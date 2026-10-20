#!/usr/bin/env python3
"""Summarize raw capture coverage and metrics; do not infer causal or budget wins."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from resources import digest


def stats(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values: return {'n': 0, 'min': None, 'mean': None, 'p50': None, 'p95': None, 'max': None}
    def percentile(p):
        k = (len(values)-1)*p; lower = math.floor(k); upper = math.ceil(k)
        return values[lower] + (values[upper]-values[lower])*(k-lower)
    return {'n': len(values), 'min': values[0], 'mean': sum(values)/len(values),
            'p50': percentile(.5), 'p95': percentile(.95), 'max': values[-1]}


def read_records(path):
    if not path.exists(): return [], {'missing_file': 1}
    records, errors = [], Counter()
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            if not isinstance(record, dict): raise ValueError('Expected object')
            records.append(record)
        except (ValueError, TypeError): errors['invalid_json_lines'] += 1
    return records, dict(errors)


def summarize(root):
    root = Path(root); manifest = json.loads((root/'manifest.json').read_text())
    result = {'schema_version': 1, 'summary_script_sha256': digest(__file__),
              'scope': 'entire capture, including setup and teardown; not workload-only averages',
              'context': manifest['config'].get('context', {}), 'collection_status': manifest['status'],
              'nodes': {}, 'network': [], 'input_files_sha256': {},
              'interpretation': ['ICMP RTT and TCP connect time are separate metrics, not one-way or NCCL latency.',
                                 'No GPU/probe data means unavailable, never zero utilization or zero RTT.',
                                 'Utilization samples are driver sample-window averages, not per-request GPU timings.',
                                 'Clock observations bound alignment uncertainty; command start is not the Poisson origin.',
                                 'These metrics alone do not establish equal performance, equal budget or causation.']}
    events, errors = read_records(root/'events.jsonl'); result['events'] = events; result['event_read_errors'] = errors
    for spec in manifest['config']['nodes']:
        name = spec['name']; folder = root/name
        gpu, gpu_errors = read_records(folder/'gpu.jsonl'); network, network_errors = read_records(folder/'network.jsonl')
        devices = defaultdict(list)
        for sample in gpu:
            if sample.get('status') == 'ok':
                for device in sample['devices']: devices[device['uuid']].append(device)
        record = manifest['nodes'].get(name, {})
        output = {'gpu_sample_status': dict(Counter(s.get('status', 'missing') for s in gpu)),
                  'read_errors': {'gpu': gpu_errors, 'network': network_errors}, 'devices': {},
                  'clock_before': record.get('clock_before'), 'clock_after': record.get('clock_after')}
        for identity, samples in devices.items():
            output['devices'][identity] = {'name': samples[0]['name'], 'pci_bus_id': samples[0]['pci_bus_id'],
                **{key: stats(s.get(key) for s in samples) for key in ['gpu_utilization_pct', 'memory_activity_pct',
                    'memory_used_mib', 'memory_total_mib', 'power_w', 'temperature_c', 'sm_clock_mhz', 'memory_clock_mhz']}}
        result['nodes'][name] = output
        paths = defaultdict(list)
        for sample in network:
            paths[(sample['peer'], sample['address'], sample['protocol'], sample.get('port'), sample['path'])].append(sample)
        for (peer, address, protocol, port, path), samples in paths.items():
            successful = [s for s in samples if s['status'] == 'ok']
            metric = 'rtt_ms' if protocol == 'icmp' else 'connect_ms'
            result['network'].append({'source': name, 'peer': peer, 'address': address, 'protocol': protocol,
                'port': port, 'path': path, 'attempted': len(samples), 'successful': len(successful),
                'probe_failure_fraction': (len(samples)-len(successful))/len(samples),
                'errors': dict(Counter(s.get('error_type', 'exit_' + str(s.get('returncode'))) for s in samples if s['status'] != 'ok')),
                metric: stats(s[metric] for s in successful),
                'upper_bound_samples': sum(s.get('rtt_is_upper_bound', False) for s in successful)})
        for filename in ['gpu.jsonl', 'network.jsonl', 'ready.json', 'done.json']:
            if (folder/filename).exists(): result['input_files_sha256'][name+'/'+filename] = digest(folder/filename)
    result['input_files_sha256'].update({'manifest.json': digest(root/'manifest.json'),
                                        'events.jsonl': digest(root/'events.jsonl')})
    (root/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('telemetry', type=Path)
    result = summarize(parser.parse_args().telemetry)
    print(json.dumps({'nodes': len(result['nodes']), 'network_paths': len(result['network']),
                      'collection_status': result['collection_status']}))
