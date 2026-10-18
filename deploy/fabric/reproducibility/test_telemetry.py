"""Synthetic/local tests only: never contact FABRIC, allocate GPUs or run inference."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from resources import digest
from summarize_telemetry import summarize
from telemetry import Session, run_command, validate
from telemetry_guest import GPU_FIELDS, gpu_sample, probe

HERE = Path(__file__).resolve().parent
ROW = '0, GPU-synthetic, Synthetic GPU, 0000:01:00.0, 75, 20, 1024, 24576, [Not Supported], 50, 1200, 5000\n'


class LocalNode:
    """Translate each simulated VM's /tmp into its own temporary filesystem."""
    def __init__(self, root, name, bins): self.root, self.name, self.bins = root, name, bins
    def translate(self, path): return path.replace('/tmp/hexgen-telemetry-', str(self.root/'vm-'))
    def execute(self, command, **_):
        result = subprocess.run(self.translate(command), shell=True, capture_output=True, text=True,
                                timeout=25, env={**os.environ, 'PATH': str(self.bins)+os.pathsep+os.environ['PATH']})
        return result.stdout, result.stderr
    def upload_file(self, source, target, **_): shutil.copy2(source, self.translate(target))
    def download_file(self, target, source, **_): shutil.copy2(self.translate(source), target)
    def get_host(self): return 'synthetic-physical-host'
    def get_site(self): return 'TEST'


class TelemetryTests(unittest.TestCase):
    def test_gpu_units_and_unavailable_not_zero(self):
        with patch('telemetry_guest.subprocess.run', return_value=Mock(returncode=0, stdout=ROW)):
            data = gpu_sample()
        gpu = data['devices'][0]
        self.assertEqual(gpu['gpu_utilization_pct'], 75)
        self.assertEqual(gpu['memory_activity_pct'], 20)
        self.assertEqual(gpu['memory_used_mib'], 1024)
        self.assertIsNone(gpu['power_w'])
        self.assertEqual(gpu['unavailable_fields']['power_w'], '[Not Supported]')
        self.assertIn('utilization.memory', GPU_FIELDS)

    def test_missing_driver_and_malformed_reply_are_errors(self):
        with patch('telemetry_guest.subprocess.run', side_effect=FileNotFoundError):
            self.assertEqual(gpu_sample()['error_type'], 'FileNotFoundError')
        with patch('telemetry_guest.subprocess.run', return_value=Mock(returncode=0, stdout='bad,csv')):
            self.assertEqual(gpu_sample()['status'], 'error')

    def test_icmp_ipv6_bounded_and_loss_not_zero_rtt(self):
        peer = {'name': 'other', 'address': '2001:db8::2', 'path': 'pipeline-data'}
        with patch('telemetry_guest.subprocess.run', return_value=Mock(returncode=0, stdout='64 bytes: time=1.25 ms')) as call:
            sample = probe(peer, 'icmp', 1)
        self.assertEqual(sample['rtt_ms'], 1.25)
        self.assertIn('-6', call.call_args.args[0]); self.assertEqual(call.call_args.kwargs['timeout'], 2)
        with patch('telemetry_guest.subprocess.run', return_value=Mock(returncode=1, stdout='', stderr='')):
            sample = probe(peer, 'icmp', 1)
        self.assertEqual(sample['status'], 'error'); self.assertNotIn('rtt_ms', sample)

    def test_submillisecond_ping_upper_bound_is_preserved(self):
        with patch('telemetry_guest.subprocess.run', return_value=Mock(returncode=0, stdout='64 bytes: time<1 ms')):
            sample = probe({'name': 'peer', 'address': '127.0.0.1'}, 'icmp', 1)
        self.assertTrue(sample['rtt_is_upper_bound'])

    def test_tcp_connect_uses_existing_service_and_distinct_metric(self):
        with patch('telemetry_guest.socket.create_connection') as connect:
            sample = probe({'name': 'peer', 'address': '127.0.0.1', 'tcp_port': 22}, 'tcp_connect', .5)
        connect.assert_called_once_with(('127.0.0.1', 22), timeout=.5)
        self.assertEqual(sample['status'], 'ok'); self.assertIn('connect_ms', sample); self.assertNotIn('rtt_ms', sample)

    def test_real_loopback_tcp_handshake(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen()
            sample = probe({'name': 'local', 'address': '127.0.0.1', 'tcp_port': listener.getsockname()[1]}, 'tcp_connect', 1)
            self.assertEqual(sample['status'], 'ok'); self.assertGreater(sample['connect_ms'], 0)

    def test_invalid_network_targets_and_unbounded_sampling_rejected(self):
        base = {'nodes': [{'name': 'a', 'slice_id': 'slice', 'peers': [{'name': 'b', 'address': '127.0.0.1'}]},
                          {'name': 'b', 'slice_id': 'slice', 'peers': []}]}
        validate(base)
        for address in ['not-an-ip', '0.0.0.0', '224.0.0.1']:
            changed = json.loads(json.dumps(base)); changed['nodes'][0]['peers'][0]['address'] = address
            with self.assertRaises(ValueError): validate(changed)
        for interval in [float('nan'), float('inf'), -.1, 0, 10000]:
            with self.assertRaises(ValueError): validate({**base, 'gpu_interval_s': interval})
        with self.assertRaises(ValueError): validate({**base, 'probe_timeout_s': 6})

    def fixture(self, root):
        bins = root/'bin'; bins.mkdir()
        (bins/'nvidia-smi').write_text('#!'+sys.executable+'\nprint('+repr(ROW)+',end="")\n')
        (bins/'ping').write_text('#!'+sys.executable+'\nprint("64 bytes: time=0.25 ms")\n')
        for binary in bins.iterdir(): binary.chmod(0o755)
        nodes = {}
        for name in ['a', 'b']:
            directory = root/name; directory.mkdir()
            nodes[name] = LocalNode(directory, name, bins)
        fab = Mock(); fab.get_slice.return_value.get_node.side_effect = lambda name: nodes[name]
        config = {'gpu_interval_s': .1, 'network_interval_s': .1, 'context': {'synthetic': True, 'arm': 'test'},
                  'nodes': [{'name': name, 'slice_id': 'local-only', 'peers': [
                      {'name': other, 'address': '127.0.0.1', 'path': 'synthetic'}]}
                      for name, other in [('a', 'b'), ('b', 'a')]]}
        return fab, config, nodes

    def test_end_to_end_agents_around_command_and_preserved_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fab, config, _ = self.fixture(root); out = root/'run'
            with patch('telemetry.capture', return_value={'synthetic': True}):
                with Session(fab, config, out, duration=2) as session:
                    session.mark('workload_started', origin='synthetic')
                    self.assertEqual(run_command(session, [sys.executable, '-c', 'import time; time.sleep(.25)'], 2), 0)
                    session.mark('workload_finished')
            manifest = json.loads((out/'telemetry/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'collected')
            self.assertEqual(manifest['resource_inventory_sha256'], digest(out/'resources.json'))
            for name in ['a', 'b']:
                record = manifest['nodes'][name]
                self.assertEqual(record['completion']['reason'], 'requested')
                self.assertIn('uncertainty_ms', record['clock_before'])
                self.assertGreater(record['completion']['counts']['gpu'], 1)
            summary = summarize(out/'telemetry')
            self.assertEqual(len(summary['network']), 2)
            self.assertEqual(summary['nodes']['a']['devices']['GPU-synthetic']['power_w']['n'], 0)
            self.assertEqual(summary['network'][0]['rtt_ms']['p50'], .25)
            with self.assertRaises(FileExistsError):
                with Session(fab, config, out): pass

    def test_benchmark_error_still_stops_agents_and_preserves_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fab, config, _ = self.fixture(root); out = root/'run'
            with patch('telemetry.capture', return_value={'synthetic': True}):
                with self.assertRaisesRegex(ValueError, 'benchmark failed'):
                    with Session(fab, config, out, duration=2): raise ValueError('benchmark failed')
            manifest = json.loads((out/'telemetry/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'collected')
            self.assertTrue((out/'resources.txt').exists())

    def test_slow_peer_does_not_stall_gpu_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fab, config, _ = self.fixture(root); out = root/'run'
            (root/'bin/ping').write_text('#!'+sys.executable+'\nimport time\ntime.sleep(.5)\nprint("time=1 ms")\n')
            with patch('telemetry.capture', return_value={'synthetic': True}):
                with Session(fab, config, out, duration=2): time.sleep(.35)
            manifest = json.loads((out/'telemetry/manifest.json').read_text())
            for node in manifest['nodes'].values():
                counts = node['completion']['counts']
                self.assertGreater(counts['gpu'], counts['network'])

    def test_watchdog_exits_without_controller(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, _, _ = self.fixture(root)
            config = root/'config.json'; config.write_text(json.dumps({'node': 'local', 'gpu_interval_s': .1, 'peers': []}))
            subprocess.run([sys.executable, str(HERE/'telemetry_guest.py'), '--config', str(config),
                            '--out', str(root/'capture'), '--duration', '.2'], timeout=10, check=True,
                           env={**os.environ, 'PATH': str(root/'bin')+os.pathsep+os.environ['PATH']})
            self.assertEqual(json.loads((root/'capture/done.json').read_text())['reason'], 'deadline')

    def test_command_timeout_terminates_its_process_group(self):
        session = Mock()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_command(session, [sys.executable, '-c', 'import time; time.sleep(30)'], .1)
        self.assertEqual(session.mark.call_args.kwargs['outcome'], 'TimeoutExpired')

    def test_startup_api_failure_is_incomplete_not_empty_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fab, config, _ = self.fixture(root)
            fab.get_slice.side_effect = RuntimeError('API unavailable')
            with self.assertRaisesRegex(RuntimeError, 'API unavailable'):
                with Session(fab, config, root/'run', duration=2): pass
            result = json.loads((root/'run/telemetry/manifest.json').read_text())
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['startup_error_type'], 'RuntimeError')

    def test_download_failure_retains_other_data_and_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fab, config, nodes = self.fixture(root); out = root/'run'
            original = nodes['a'].download_file
            def transfer(target, source, **kwargs):
                if source.endswith('gpu.jsonl'): raise OSError('transfer interrupted')
                return original(target, source, **kwargs)
            nodes['a'].download_file = transfer
            with patch('telemetry.capture', return_value={'synthetic': True}):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete telemetry'):
                    with Session(fab, config, out, duration=2): pass
            summary = json.loads((out/'telemetry/summary.json').read_text())
            self.assertEqual(summary['collection_status'], 'incomplete')
            self.assertEqual(summary['nodes']['a']['read_errors']['gpu'], {'missing_file': 1})
            self.assertTrue((out/'telemetry/b/gpu.jsonl').exists())
            self.assertTrue((out/'resources.txt').exists())

    def test_launch_error_and_nonzero_command_exit_are_recorded(self):
        session = Mock()
        with self.assertRaises(FileNotFoundError): run_command(session, ['/no/such/binary'], 1)
        self.assertEqual(session.mark.call_args.kwargs['outcome'], 'FileNotFoundError')
        self.assertEqual(run_command(session, [sys.executable, '-c', 'raise SystemExit(17)'], 1), 17)
        self.assertEqual(session.mark.call_args.kwargs['returncode'], 17)


if __name__ == '__main__': unittest.main()
