"""Scientific plotting helpers reused from the completed native experiment."""

from pathlib import Path

import numpy as np

SCALES=[0,0.5,1,1.5,2,3,4,6,8,12,16,20,32,48,64,96,128,192,256,384,512,768,1024]

def percentile(values, p):
    return float(np.percentile(values,p)) if values else None

def plot_experiment(folder, config, reference_seconds, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder)
    times=sorted(r['latency_seconds'] for r in rows if r['valid'])
    fig,ax=plt.subplots(figsize=(7,4.4),layout='constrained')
    x=[0.0]+times; y=[0.0]+[100*(i+1)/len(rows) for i in range(len(times))]
    ax.step(x,y,where='post',color='#cf3038',linewidth=2,label='Red: 2 × Quadro RTX 6000')
    ax.set(xlabel='Allowed completion time (seconds)',ylabel='Requests meeting deadline (%)',ylim=(0,101),xlim=(0,max(times,default=1)*1.05),
        title=f"32 output tokens · {config['experiment']['rate']:g} requests/s")
    ax.grid(alpha=.2); ax.legend(loc='lower right',fontsize=9)
    axis=ax.secondary_xaxis('top',functions=(lambda x:x/reference_seconds,lambda x:x*reference_seconds))
    axis.set_xlabel('Deadline / isolated RTX inference reference')
    fig.savefig(folder/'slo_attainment.png',dpi=180);fig.savefig(folder/'slo_attainment.pdf');plt.close(fig)
