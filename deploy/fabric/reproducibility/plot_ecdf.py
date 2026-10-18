#!/usr/bin/env python3
"""Plot raw end-to-end latency ECDFs for paired experiment result directories."""
import argparse
import csv
import json
import math
import hashlib
import importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = {'homogeneous': ('Homogeneous', '#e11d25'),
        'heterogeneous': ('Heterogeneous', '#0000db')}


def save(fig, base):
    for ext in ['png', 'svg']:
        path=base.with_suffix('.'+ext);fig.savefig(path,dpi=200)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def ecdf(records):
    if not records:
        raise ValueError('No offered requests')
    finite = []
    for r in records:
        if r['success']:
            value = float(r['latency_seconds'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Successful request has invalid latency')
            finite.append(value)
    xs, counts = np.unique(finite, return_counts=True)
    return xs, np.cumsum(counts) / len(records)


def draw(ax, run):
    manifest = json.loads((run/'manifest.json').read_text()) if (run/'manifest.json').exists() else {}
    plotted, stats, maximum, lines = [], {}, 0., []
    for arm, (label, color) in ARMS.items():
        data = json.loads((run/(arm+'.json')).read_text())
        records = data['requests'] if isinstance(data, dict) else data
        xs, ys = ecdf(records)
        limit = float(xs[-1]) if len(xs) else 1.
        maximum = max(maximum, limit)
        lines.extend(ax.step(np.r_[0., xs, limit*1.03], np.r_[0., ys, ys[-1] if len(ys) else 0.],
                             where='post', lw=2, color=color, label=label))
        plotted.extend({'arm': arm, 'latency_seconds': float(x), 'ecdf': float(y),
                        'offered': len(records)} for x, y in zip(xs, ys))
        stats[arm] = {'offered':len(records), 'successful':sum(r['success'] for r in records),
                      'failed':sum(not r['success'] for r in records)}
    with (run/'latency-ecdf.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['arm','latency_seconds','ecdf','offered'],lineterminator='\n')
        writer.writeheader();writer.writerows(plotted)
    end=max(1.,maximum*1.03)
    for line in lines:line.set_xdata(np.r_[line.get_xdata()[:-1],end])
    ax.set(xlim=(0,end), ylim=(0,1.03), xlabel='End-to-end latency (seconds)', ylabel='ECDF')
    title = (f"{manifest['output_tokens']} output tokens, {manifest['rate']:g} requests/s"
             if 'output_tokens' in manifest and 'rate' in manifest else run.name)
    ax.set_title(title);ax.grid(alpha=.25);ax.spines[['top','right']].set_visible(False)
    counts='/'.join(str(s['offered']) for s in stats.values());failed=sum(s['failed'] for s in stats.values())
    ax.text(.98,.05,f'n={counts}; {failed} failures',
            transform=ax.transAxes,ha='right',fontsize=8,color='#475569')
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path, help='One experiment folder, or a batch containing experiment subfolders')
    args=p.parse_args();root=args.results
    runs=[root] if (root/'homogeneous.json').exists() else sorted(
        (r for r in root.iterdir() if r.is_dir() and (r/'homogeneous.json').exists() and (r/'heterogeneous.json').exists()),
        key=lambda r: (json.loads((r/'manifest.json').read_text()).get('output_tokens',0),
                       json.loads((r/'manifest.json').read_text()).get('rate',0),r.name)
             if (r/'manifest.json').exists() else (0,0,r.name))
    if not runs:p.error('No paired homogeneous/heterogeneous JSON results found')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    overview,axes=plt.subplots(math.ceil(len(runs)/3),min(3,len(runs)),figsize=(15,4.2*math.ceil(len(runs)/3)),squeeze=False)
    stats={}
    for run,ax in zip(runs,axes.flat):
        stats[run.name]=draw(ax,run)
        fig,one=plt.subplots(figsize=(8,5));draw(one,run);one.legend(fontsize=9)
        fig.suptitle('Empirical distribution of request latency');fig.tight_layout()
        save(fig,run/'latency-ecdf')
        plt.close(fig)
    for ax in list(axes.flat)[len(runs):]:ax.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    overview.suptitle('Latency ECDFs across experiment settings',fontsize=17)
    overview.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.035))
    overview.text(.5,.01,'Latency starts at scheduled arrival; includes dispatch, transport, queues and inference. Failures are placed at infinity.',ha='center',fontsize=9,color='#475569')
    overview.tight_layout(rect=(0,.09,1,.95))
    save(overview,root/'latency-ecdf-overview')
    (root/'latency-ecdf-summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    provenance={'generated_utc':datetime.now(timezone.utc).isoformat(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'versions':{p:importlib.metadata.version(p) for p in ['numpy','matplotlib']},
                'definition':'Successful completion latency from scheduled arrival in raw seconds; failures at infinity; all offered requests in denominator.',
                'inputs':{str(run/(arm+'.json')):hashlib.sha256((run/(arm+'.json')).read_bytes()).hexdigest() for run in runs for arm in ARMS}}
    (root/'latency-ecdf-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Generated {len(runs)} experiment ECDFs and overview in {root}')


if __name__=='__main__':main()
