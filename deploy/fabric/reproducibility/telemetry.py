#!/usr/bin/env python3
"""Wrap an existing benchmark with bounded FABRIC GPU and inter-VM measurements."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
import uuid
from resources import capture, digest, optional, utc, write_inventory
from telemetry_guest import validate as validate_guest

HERE = Path(__file__).resolve().parent


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def mark(out, event, **fields):
    """A benchmark child can call this at its actual Poisson origin/finish."""
    record = {**fields, 'event': event, 'unix_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
    with (Path(out)/'telemetry/events.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
    return record


def validate(config):
    nodes = config['nodes']
    if not nodes or len(nodes) > 32: raise ValueError('Select 1-32 experiment VMs')
    names = [n['name'] for n in nodes]
    if len(set(names)) != len(names): raise ValueError('Node names must be globally unique')
    for node in nodes:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', node['name']) or not node['slice_id']:
            raise ValueError('Each node needs a safe name and slice ID')
        validate_guest({**config, **node})
        if any(p['name'] not in names or p['name'] == node['name'] for p in node.get('peers', [])):
            raise ValueError('Peers must name another selected VM')
    return config


class Session:
    """Use as a context manager around one arm; mark the actual workload origin."""
    def __init__(self, fab, config, out, duration=1800):
        self.config = validate(config)
        if not math.isfinite(duration) or duration <= 0: raise ValueError('Finite positive duration required')
        self.fab, self.out, self.duration = fab, Path(out) / 'telemetry', duration
        self.remote = '/tmp/hexgen-telemetry-' + uuid.uuid4().hex
        self.nodes, self.slices = {}, {}
        self.manifest = {'schema_version': 1, 'started_utc': utc(), 'config': config,
                         'status': 'starting',
                         'agent_sha256': digest(HERE/'telemetry_guest.py'), 'controller_sha256': digest(__file__),
                         'max_workload_seconds': duration, 'remote_directory': self.remote, 'nodes': {}}

    def mark(self, event, **fields):
        return mark(self.out.parent, event, **fields)

    def execute(self, node, command):
        return node.execute(command, quiet=True, display=False, retry=1, read_timeout=20)[0].strip()

    def clock(self, node):
        observations = []
        for _ in range(3):
            before, tick = time.time_ns(), time.monotonic_ns()
            remote = int(self.execute(node, "python3 -c 'import time; print(time.time_ns())'"))
            after, elapsed = time.time_ns(), time.monotonic_ns() - tick
            if abs((after - before) - elapsed) > 10_000_000:
                raise RuntimeError('Controller wall clock stepped during clock measurement')
            observations.append({'controller_before_unix_ns': before, 'controller_after_unix_ns': after,
                                 'guest_unix_ns': remote, 'offset_ms': (remote - (before + after)/2)/1e6,
                                 'uncertainty_ms': elapsed/2e6})
        return min(observations, key=lambda o: o['uncertainty_ms'])

    def start_node(self, spec):
        name = spec['name']; node = self.nodes[name]
        folder = self.out/name; folder.mkdir()
        record = self.manifest['nodes'][name] = {
            'slice_id': spec['slice_id'], 'physical_host': optional(node.get_host),
            'site': optional(node.get_site), 'clock_before': optional(lambda: self.clock(node))}
        config = {**{k: v for k, v in self.config.items() if k not in ['nodes', 'context']},
                  **spec, 'node': name}
        save(folder/'config.json', config)
        remote = shlex.quote(self.remote)
        if self.execute(node, f'mkdir -m 700 -- {remote} && printf created') != 'created':
            raise RuntimeError('Could not create remote telemetry directory')
        node.upload_file(str(HERE/'telemetry_guest.py'), self.remote+'/agent.py', retry=1)
        node.upload_file(str(folder/'config.json'), self.remote+'/config.json', retry=1)
        command = shlex.join(['python3', self.remote+'/agent.py', '--config', self.remote+'/config.json',
                              '--out', self.remote+'/samples', '--duration', str(self.duration+300)])
        self.execute(node, f'nohup {command} > {remote}/agent.log 2>&1 < /dev/null &')
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            ready = self.execute(node, f'cat {remote}/samples/ready.json 2>/dev/null')
            if ready:
                record.update(status='ready', ready=json.loads(ready)); return
            time.sleep(.2)
        raise TimeoutError('Guest collector did not become ready')

    def parallel(self, function, specs):
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            pending = {pool.submit(function, spec): spec['name'] for spec in specs}
            for future in concurrent.futures.as_completed(pending):
                name = pending[future]
                try: future.result()
                except Exception as error:
                    self.manifest['nodes'].setdefault(name, {})['error_type'] = type(error).__name__
                    failures.append(name)
        return failures

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=False)
        save(self.out/'manifest.json', self.manifest)
        try:
            for spec in self.config['nodes']:
                sid = spec['slice_id']
                if sid not in self.slices: self.slices[sid] = self.fab.get_slice(slice_id=sid)
                self.nodes[spec['name']] = self.slices[sid].get_node(name=spec['name'])
            inventory = self.out.parent/'resources.json'
            if not inventory.exists(): write_inventory(self.out.parent, capture(list(self.slices.values()), guest=False))
            self.manifest['resource_inventory_sha256'] = digest(inventory)
            if self.parallel(self.start_node, self.config['nodes']):
                raise RuntimeError('Collector startup failed; see telemetry manifest and agent logs')
            self.mark('collectors_ready')
            save(self.out/'manifest.json', self.manifest)
            return self
        except BaseException as error:
            self.manifest['startup_error_type'] = type(error).__name__
            self.finish(); raise

    def finish_node(self, spec):
        name = spec['name']; node = self.nodes.get(name)
        if node is None: return
        folder = self.out/name; folder.mkdir(exist_ok=True)
        record = self.manifest['nodes'].setdefault(name, {})
        remote = shlex.quote(self.remote)
        record['clock_after'] = optional(lambda: self.clock(node))
        self.execute(node, f'touch {remote}/samples/STOP')
        deadline = time.monotonic()+120
        while time.monotonic() < deadline:
            done = self.execute(node, f'cat {remote}/samples/done.json 2>/dev/null')
            if done: record['completion'] = json.loads(done); break
            # A startup failure may have no samples directory; still retrieve its log.
            if record.get('status') != 'ready': break
            time.sleep(.2)
        record['download_errors'] = {}
        for filename in ['gpu.jsonl', 'network.jsonl', 'ready.json', 'done.json', 'agent.log']:
            path = self.remote + ('/' if filename == 'agent.log' else '/samples/') + filename
            try: node.download_file(str(folder/filename), path, retry=1)
            except Exception as error: record['download_errors'][filename] = type(error).__name__
        record['status'] = ('collected' if 'completion' in record and not record['download_errors']
                            and not record['completion']['errors'] and record['completion']['reason'] == 'requested'
                            else 'incomplete')

    def finish(self):
        self.mark('collection_stopping')
        failures = self.parallel(self.finish_node, self.config['nodes'])
        incomplete = failures or 'startup_error_type' in self.manifest or set(self.manifest['nodes']) != {
            n['name'] for n in self.config['nodes']} or any(n.get('status') != 'collected' or 'error_type' in n
                                                         for n in self.manifest['nodes'].values())
        self.manifest.update(finished_utc=utc(), status='incomplete' if incomplete else 'collected')
        self.manifest['files_sha256'] = {str(p.relative_to(self.out)): digest(p)
                                       for p in self.out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        save(self.out/'manifest.json', self.manifest)
        from summarize_telemetry import summarize
        summary = optional(lambda: summarize(self.out))
        if 'unavailable' in summary:
            self.manifest['summary_error'] = summary
            save(self.out/'manifest.json', self.manifest)

    def __exit__(self, kind, error, traceback):
        if kind: self.mark('benchmark_error', error_type=kind.__name__)
        self.finish()
        if kind is None and self.manifest['status'] != 'collected':
            raise RuntimeError('Incomplete telemetry; original benchmark output is preserved')


def run_command(session, command, duration):
    session.mark('command_started')
    process, outcome = None, 'completed'
    try:
        process = subprocess.Popen(command, start_new_session=True)
        return process.wait(timeout=duration)
    except BaseException as error:
        outcome = type(error).__name__
        if process and process.poll() is None:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
        raise
    finally:
        session.mark('command_finished', returncode=process.returncode if process else None, outcome=outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Experiment directory; creates a new telemetry/ subdirectory')
    parser.add_argument('--duration', type=float, default=1800, help='Maximum workload/capture seconds')
    parser.add_argument('--fabric-rc', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(); config = validate(json.loads(args.config.read_text()))
    if not math.isfinite(args.duration) or args.duration <= 0: parser.error('Duration must be finite and positive')
    if args.dry_run: print(json.dumps(config, indent=2)); return 0
    def interrupted(*_): raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGTERM, interrupted)
    from fabrictestbed_extensions.fablib.fablib import FablibManager
    fab = FablibManager(fabric_rc=str(args.fabric_rc) if args.fabric_rc else None, auto_token_refresh=False, log_level='ERROR')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    with Session(fab, config, args.out, args.duration) as session:
        if command: return run_command(session, command, args.duration)
        session.mark('capture_started'); time.sleep(args.duration); session.mark('capture_finished')
    return 0


if __name__ == '__main__': raise SystemExit(main())
