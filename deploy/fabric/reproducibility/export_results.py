#!/usr/bin/env python3
"""Export measured latency and selected historical resource facts without prompts."""
import argparse
import json
from pathlib import Path
from resources import digest, utc, write_inventory

ARMS = ('homogeneous', 'heterogeneous')


def export(source, destination):
    source, destination = Path(source), Path(destination)
    runs = sorted(r for r in source.iterdir() if r.is_dir() and
                  all((r / (arm + '.json')).exists() for arm in ARMS))
    if not runs:
        raise ValueError('No paired experiment results found')
    destination.mkdir(parents=True, exist_ok=False)
    for run in runs:
        out = destination / run.name
        out.mkdir()
        manifest = json.loads((run / 'manifest.json').read_text())
        fields = ('input_tokens', 'output_tokens', 'rate', 'requests_per_arm', 'replicas_per_arm')
        (out / 'manifest.json').write_text(json.dumps(
            {k: manifest[k] for k in fields if k in manifest}, indent=2) + '\n')
        results = {}
        for arm in ARMS:
            original = run / (arm + '.json')
            data = json.loads(original.read_text())
            requests = data['requests'] if isinstance(data, dict) else data
            exported = {'requests': [{k: r[k] for k in ('success', 'latency_seconds')}
                                     for r in requests]}
            output = out / original.name
            output.write_text(json.dumps(exported, indent=2, allow_nan=False) + '\n')
            results[arm] = {'path': output.name, 'sha256': digest(output),
                            'original_result_sha256': digest(original), 'requests': len(requests)}
        recorded = json.loads((run / 'resources.json').read_text())
        if recorded['source_kind'] != 'historical_recorded_evidence':
            raise ValueError('This exporter accepts the recorded historical inventory schema only')
        inventory = {k: recorded[k] for k in ('schema_version', 'assembled_utc', 'source_kind',
                                             'note', 'not_recorded')}
        inventory.update(exported_utc=utc(), original_inventory_sha256=digest(run / 'resources.json'),
                         publication_note='Selected recorded facts; management addresses, process commands, peer IDs and controller paths omitted.',
                         sources={k: {'filename': Path(v['path']).name, 'sha256': v['sha256']}
                                  for k, v in recorded['sources'].items()},
                         experiment=json.loads((out / 'manifest.json').read_text()),
                         result_files=results, slices=[])
        for sl in recorded['slices']:
            nodes = []
            for n in sl['nodes']:
                runtime = n['post_run_runtime']
                attrs = n['attributes_recorded_by_fablib']
                nodes.append({'attributes_recorded_by_fablib': {k: attrs[k] for k in
                              ('name', 'site', 'host', 'reservation_state') if k in attrs},
                              'requested_resources_not_new_measurements': n['requested_resources_not_new_measurements'],
                              'post_run_runtime': {k: runtime[k] for k in ('gpu_runtime', 'bootstrap_evidence',
                                                   'source_hashes_match', 'source_sha256') if k in runtime}})
            inventory['slices'].append({'attributes': sl['attributes'], 'nodes': nodes})
        write_inventory(out, inventory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, help='New directory; never overwrite an export')
    args = parser.parse_args()
    export(args.source, args.destination)
