#!/usr/bin/env python3
"""Bounded, standard-library GPU and peer-RTT collector, run inside each VM."""
import argparse
import concurrent.futures
import csv
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

GPU_FIELDS = ('index', 'uuid', 'name', 'pci.bus_id', 'utilization.gpu', 'utilization.memory',
              'memory.used', 'memory.total', 'power.draw', 'temperature.gpu', 'clocks.sm', 'clocks.mem')
GPU_KEYS = ('index', 'uuid', 'name', 'pci_bus_id', 'gpu_utilization_pct', 'memory_activity_pct',
            'memory_used_mib', 'memory_total_mib', 'power_w', 'temperature_c', 'sm_clock_mhz', 'memory_clock_mhz')


def validate(config):
    for key, default in [('gpu_interval_s', 1), ('network_interval_s', 5), ('probe_timeout_s', 1)]:
        value = float(config.get(key, default))
        maximum = 5 if key == 'probe_timeout_s' else 3600
        if not math.isfinite(value) or not .1 <= value <= maximum:
            raise ValueError(f'{key} must be finite and between 0.1 and {maximum}')
    peers = config.get('peers', [])
    if len(peers) > 32: raise ValueError('At most 32 explicit peers per VM')
    identities = []
    for peer in peers:
        address = ipaddress.ip_address(peer['address'])
        if address.is_unspecified or address.is_multicast: raise ValueError('Use a unicast peer IP')
        if not peer['name']: raise ValueError('Peer name is required')
        if 'tcp_port' in peer and (type(peer['tcp_port']) is not int or not 1 <= peer['tcp_port'] <= 65535):
            raise ValueError('Invalid TCP port')
        identities.append((peer['name'], peer['address'], peer.get('tcp_port')))
    if len(set(identities)) != len(peers): raise ValueError('Duplicate peer probe')
    return config


def gpu_sample():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS),
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if result.returncode: return {'status': 'error', 'returncode': result.returncode, 'stderr': result.stderr}
        devices = []
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != len(GPU_KEYS): raise ValueError('Unexpected nvidia-smi columns')
            device, unavailable = {}, {}
            for i, (key, raw) in enumerate(zip(GPU_KEYS, row)):
                raw = raw.strip()
                if i < 4: device[key] = int(raw) if i == 0 else raw
                else:
                    try:
                        value = float(raw)
                        if not math.isfinite(value): raise ValueError('nonfinite')
                        device[key] = value
                    except ValueError: device[key] = None; unavailable[key] = raw
            device['unavailable_fields'] = unavailable
            devices.append(device)
        return {'status': 'ok' if devices else 'unavailable', 'devices': devices}
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return {'status': 'error', 'error_type': type(error).__name__}


def probe(peer, protocol, timeout):
    start, tick = time.time_ns(), time.monotonic_ns()
    result = {'peer': peer['name'], 'address': peer['address'], 'path': peer.get('path', 'unspecified'),
              'protocol': protocol, 'started_unix_ns': start, 'started_monotonic_ns': tick}
    try:
        if protocol == 'icmp':
            ipv6 = ipaddress.ip_address(peer['address']).version == 6
            command = ['ping'] + (['-6'] if ipv6 else []) + ['-n', '-c', '1', '-s', '56',
                       '-W', str(math.ceil(timeout)), peer['address']]
            reply = subprocess.run(command, capture_output=True, text=True, timeout=math.ceil(timeout) + 1,
                                   env={**os.environ, 'LC_ALL': 'C'})
            match = re.search(r'time([=<])([0-9.]+)\s*ms', reply.stdout)
            if reply.returncode or not match:
                result.update(status='error', returncode=reply.returncode, stderr=reply.stderr)
            else:
                result.update(status='ok', rtt_ms=float(match[2]), rtt_is_upper_bound=match[1] == '<',
                              payload_bytes=56)
        else:
            result['port'] = peer['tcp_port']
            connect_tick = time.monotonic_ns()
            with socket.create_connection((peer['address'], peer['tcp_port']), timeout=timeout):
                elapsed = (time.monotonic_ns() - connect_tick) / 1e6
            result.update(status='ok', connect_ms=elapsed)
    except (OSError, subprocess.TimeoutExpired) as error:
        result.update(status='error', error_type=type(error).__name__)
    result.update(finished_unix_ns=time.time_ns(), probe_elapsed_ms=(time.monotonic_ns() - tick) / 1e6)
    return result


def collect(config, out, duration):
    validate(config)
    if not math.isfinite(duration) or duration <= 0: raise ValueError('Finite positive duration required')
    out = Path(out); out.mkdir(exist_ok=False, parents=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    started = time.monotonic()
    counts, errors = {'gpu': 0, 'network': 0}, []
    def write(name, record):
        record.update(node=config['node'], hostname=socket.gethostname())
        with (out / (name + '.jsonl')).open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        counts[name] += 1
    # Separate workers ensure a slow/unreachable peer cannot stall GPU sampling.
    def periodic(kind, interval, sample):
        due = time.monotonic()
        try:
            while not stop.is_set():
                sample()
                due = max(due + interval, time.monotonic())
                stop.wait(max(0, due - time.monotonic()))
        except Exception as error:
            errors.append({'worker': kind, 'error_type': type(error).__name__}); stop.set()
    def gpu():
        begin, tick = time.time_ns(), time.monotonic_ns()
        data = gpu_sample()
        write('gpu', {'started_unix_ns': begin, 'finished_unix_ns': time.time_ns(),
                      'started_monotonic_ns': tick, 'sample_elapsed_ms': (time.monotonic_ns() - tick) / 1e6, **data})
    def network():
        jobs = [(p, kind) for p in config.get('peers', []) for kind in
                (['icmp', 'tcp_connect'] if 'tcp_port' in p else ['icmp'])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(probe, p, kind, float(config.get('probe_timeout_s', 1))) for p, kind in jobs]
            for future in concurrent.futures.as_completed(futures): write('network', future.result())
    # Files exist even for no peers or a missing driver; readiness is distinct from health.
    for name in counts: (out / (name + '.jsonl')).touch()
    gpu()
    (out / 'ready.json').write_text(json.dumps({'ready_unix_ns': time.time_ns(), 'hostname': socket.gethostname(),
                                               'node': config['node'], 'duration_s': duration}) + '\n')
    workers = [threading.Thread(target=periodic, args=(kind, float(config.get(key, default)), fn), daemon=True)
               for kind, key, default, fn in [('gpu', 'gpu_interval_s', 1, gpu),
                                             ('network', 'network_interval_s', 5, network)]]
    for worker in workers: worker.start()
    while not stop.wait(.1):
        if (out / 'STOP').exists() or time.monotonic() - started >= duration: break
    reason = 'requested' if (out / 'STOP').exists() else ('deadline' if not stop.is_set() else 'signal_or_error')
    stop.set()
    for worker in workers: worker.join()
    (out / 'done.json').write_text(json.dumps({'finished_unix_ns': time.time_ns(), 'reason': reason,
                                              'counts': counts, 'errors': errors}) + '\n')
    return 1 if errors else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    args = parser.parse_args()
    raise SystemExit(collect(json.loads(args.config.read_text()), args.out, args.duration))
