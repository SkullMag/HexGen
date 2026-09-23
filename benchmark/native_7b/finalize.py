"""Re-audit seven native rates and create the compact 22-file final archive."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from paths import RUN_DIR as root
experiments = [('01_32_0.125_red',.125),('02_32_0.25_red',.25),('03_32_0.5_red',.5),
               ('04_32_01_red',1),('05_32_02_red',2),('06_32_04_red',4),('07_32_08_red',8)]
assert json.loads((root/'status.json').read_text())['phase'] == 'complete'
for directory,_ in experiments:
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit_results.py')),directory],check=True)
setup = json.loads((root/'compact/setup.json').read_text())
setup.update(included_rates=[r for _,r in experiments],requests_per_rate=500,all_rates_complete=True,
             total_requests=3500)
(root/'compact/setup.json').write_text(json.dumps(setup,indent=2)+'\n')
names = ['setup.json']+[f'{directory}/{name}' for directory,_ in experiments
                        for name in ['metrics.json','requests.csv','plot.png']]
archive = root/'native-hexgen-stream1024-red-full.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in names:z.write(root/'compact'/name,name)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist()) == 22 and set(z.namelist()) == set(names) and z.testzip() is None
    for name in names:assert z.read(name) == (root/'compact'/name).read_bytes()
    assert sum(len(list(csv.DictReader(io.StringIO(z.read(f'{d}/requests.csv').decode())))) for d,_ in experiments) == 3500

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
base = setup['base_inference_latency_seconds']
fig,axes = plt.subplots(2,4,figsize=(14,7),layout='constrained',sharex=True,sharey=True)
table=[]
for ax,(directory,rate) in zip(axes.flat,experiments):
    records = [json.loads(line) for line in (root/directory/'requests.jsonl').read_bytes().split(b'\n') if line]
    latency = np.sort([r['end_to_end_latency_s']/base for r in records if r['success']])
    x = np.array([0]+latency[latency<=20].tolist()+[20])
    y = np.searchsorted(latency,x,side='right')*100/len(records)
    ax.step(x,y,where='post',color='#cf3038',linewidth=2)
    ax.set(title=f'{rate:g} requests/s',xlabel='Deadline / base inference latency',
           ylabel='Requests meeting deadline (%)',xlim=(0,20),ylim=(0,102))
    ax.grid(alpha=.2)
    summary = json.loads((root/directory/'summary.json').read_text())
    ax.text(.96,.72,f"Median: {summary['median_s']:.3f} s\np95: {summary['p95_s']:.3f} s",
            transform=ax.transAxes,ha='right',va='top',fontsize=10,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    table.append(summary)
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.95,'Native HexGen · scaled Red\nOpenLLaMA-7B-v2 · FP16\n2 RTX 6000 hosts · 1 replica\nTP=2 · PP=1 · regular attention\n128 input / 32 output tokens\n500 LMSYS requests per rate\n\n'
                        f'Base inference: {base:.3f} s\nOne finite trial per rate',va='top',fontsize=11,transform=axes.flat[-1].transAxes)
fig.suptitle('Native Red sweep · 1024-stream limit — common deadline scale',fontsize=16)
fig.savefig(root/'native-red-overview.png',dpi=180)
fig.savefig(root/'native-red-overview.pdf')
plt.close(fig)
report = ['# Native homogeneous RTX result report','',
          f'All seven rates completed: 3500/3500 requests. Fresh TP=2 base inference reference: {base:.6f} s.',
          '', '| RPS | Requests | Median latency (s) | p95 latency (s) |','|---:|---:|---:|---:|']
for s in table:report.append(f"| {s['rate']:g} | {s['requests']} | {s['median_s']:.3f} | {s['p95_s']:.3f} |")
report += ['', 'The native OCF coordinator with a bounded 1024-stream/512-MiB per-peer HTTP resource override and error-handling corrections, NATS worker coordination, HexGen model partitioning/decoder and original request client were reused. One OpenLLaMA-7B-v2 FP16 replica spans two Quadro RTX 6000 hosts with TP=2 and PP=1 over private Ethernet.',
           '', 'This is a scaled regular-attention starting-point experiment, not a reproduction of the paper\'s 70B Red numbers. The cross-host topology, GPU type, model, controlled LMSYS subset and reference latency differ. FlashAttention 2.0.8 supplies package components, but its attention kernels are disabled on these Turing GPUs. One finite trial per rate provides no repeat-trial confidence interval; overload latency includes queue drain.',
           '', 'See EXPERIMENT_SETUP.md for exact server configuration, input preparation, timing boundaries, source reuse and all twelve patched runtime source files and the coordinator validation tests. Compact ZIP: native-hexgen-stream1024-red-full.zip (22 verified files). Raw measurements, both-rank logs, provenance and audit receipts remain in this directory.']
(root/'RESULTS_REPORT.md').write_text('\n'.join(report)+'\n')
(root/'full-audit.json').write_text(json.dumps({'passed':True,'requests':3500,'rates':7,'zip_files':22,
                                             'archive':str(archive),'visual_review_required':True},indent=2)+'\n')
print(json.dumps({'passed':True,'archive':str(archive),'plot':str(root/'native-red-overview.png')},indent=2))
