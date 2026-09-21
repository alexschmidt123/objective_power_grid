"""Render saved SIR and grid single-step sPCE audits; never simulate or train."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sir',type=Path,required=True)
    p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    read=lambda path:list(csv.DictReader(path.open()))
    sir=read(a.sir/'summary.csv');grid=read(a.grid/'summary.csv')
    sm=json.loads((a.sir/'settings.json').read_text());gm=json.loads((a.grid/'settings.json').read_text())
    L=max(int(r['contrasts']) for r in grid)
    grid=[r for r in grid if int(r['contrasts'])==L]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=[(sir,'time','spce_standard_error','SIR ODE','Measurement time (model units)', '#2463A6'),
          (grid,'duration_s','standard_error','IEEE9: maximum absolute RoCoF','Hann injection duration (s)','#B35A21')]
    maxima=[]
    def draw(ax,d):
        rows,xkey,sekey,title,xlabel,color=d
        x=np.array([float(r[xkey]) for r in rows]);y=np.array([float(r['mean_spce']) for r in rows]);se=np.array([float(r[sekey]) for r in rows])
        ax.plot(x,y,color=color,lw=2)
        ax.fill_between(x,y-1.96*se,y+1.96*se,color=color,alpha=.16)
        j=int(y.argmax());ax.scatter(x[j],y[j],color=color,zorder=4,s=28)
        ax.annotate(f'Grid maximum: {x[j]:.2f}, {y[j]:.3f} nats',xy=(x[j],y[j]),xytext=(.36,.86),textcoords='axes fraction',arrowprops={'arrowstyle':'-','color':color},fontsize=9)
        ax.set(title=title,xlabel=xlabel,ylabel='Single-step sPCE (nats)',xlim=(x[0],x[-1]))
        ax.grid(alpha=.18);ax.axhline(0,color='gray',lw=.6)
        return {'benchmark':title,'design':float(x[j]),'mean_spce_nats':float(y[j])}
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,d in zip(axes,data):maxima.append(draw(ax,d))
    fig.text(.5,.015,'T = 1 · 512 evaluation systems · 1,024 contrasts · shading: pointwise ±1.96 Monte Carlo SE',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ['png','pdf','svg']:fig.savefig(a.output/f'single_step_eig_curves.{ext}',dpi=200)
    plt.close(fig)
    for name,d in zip(['sir_ode_single_step_eig','ieee9_rocof_single_step_eig'],data):
        fig,ax=plt.subplots(figsize=(6.2,4.2));draw(ax,d);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(a.output/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    report={'maxima':maxima,'sources':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.sir/'summary.csv',a.grid/'summary.csv']},
        'sir_note':'Empirical-bank sPCE diagnostic; legacy SIR experiment metric is entropy reduction. Noise SD is 1 infected individual.',
        'ieee9_note':'0.01 s duration grid, 3 s recording window, amplitude 0.05 pu, bus 1, scalar RoCoF noise SD 0.005 Hz/s; T=1 starts from equilibrium.',
        'interval_note':'Pointwise Monte Carlo uncertainty, not simultaneous bands, across-training-seed SD, or proof of superiority.',
        'sir_settings':sm,'ieee9_settings':gm}
    (a.output/'provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(maxima,indent=2))

if __name__=='__main__':main()
