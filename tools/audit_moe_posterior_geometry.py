"""Plot saved representative posterior geometry and measurement positions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def split_geometry(x):
    # Descriptive two-cluster split, not a certified count of posterior modes.
    x=x[::max(1,len(x)//4096)]
    centered=x-x.mean(0)
    _,_,axes=np.linalg.svd(centered,full_matrices=False)
    score=centered@axes[0]
    centers=np.array([x[score<=np.median(score)].mean(0),x[score>np.median(score)].mean(0)])
    for _ in range(30):
        labels=np.argmin(((x[:,None]-centers[None])**2).sum(-1),1)
        if min(np.bincount(labels,minlength=2))==0:break
        new=np.array([x[labels==i].mean(0) for i in range(2)])
        if np.allclose(new,centers):break
        centers=new
    within=float(np.mean(np.sum((x-centers[labels])**2,axis=1)))
    return dict(cluster_weights=(np.bincount(labels,minlength=2)/len(x)).tolist(),centers=centers.tolist(),
                centroid_distance=float(np.linalg.norm(centers[0]-centers[1])),within_cluster_rms=float(np.sqrt(within)),
                interpretation='Descriptive two-cluster partition only; does not establish distinct posterior modes.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);args=ap.parse_args()
    root=Path.cwd();c=Path(args.campaign).resolve();out=c/'evidence_audit';out.mkdir(exist_ok=True)
    diagnosis=json.loads((c/'checkpoint_diagnostics/diagnosis.json').read_text())
    selected=diagnosis['mse_audit']['representatives']
    paths=dict(moe=root/'experiments/10142026/10142026_moe_component_clipping/pilot',
               dad=root/'experiments/10132026/10132026_bounded_location_restart/pilot')
    fig,axs=plt.subplots(3,2,figsize=(10,13));report=[]
    for j,i in enumerate(selected):
        for k,(name,path) in enumerate(paths.items()):
            method='moe_sboed' if name=='moe' else 'dad'
            rows=json.loads((path/'eval'/f'{method}_seed1001.json').read_text());r=rows[i]
            start=i//16*16
            with np.load(path/'eval'/f'{method}_seed1001_posterior_batch{start}.npz') as f:
                samples=f['posterior_samples'][i-start]
            sources=samples.reshape(-1,2,2)
            truth=np.asarray(r['truth']).reshape(2,2);actions=np.asarray(r['actions'])
            ax=axs[j,k]
            for s,color in enumerate(['tab:blue','tab:orange']):
                ax.scatter(sources[::16,s,0],sources[::16,s,1],s=3,alpha=.2,color=color)
            dots=ax.scatter(actions[:,0],actions[:,1],c=np.arange(30),cmap='viridis',s=20,marker='o',edgecolors='black',linewidths=.3)
            ax.scatter(truth[:,0],truth[:,1],marker='*',s=160,color='red',label='true sources')
            ax.scatter(np.asarray(r['inference']['estimate']).reshape(2,2)[:,0],np.asarray(r['inference']['estimate']).reshape(2,2)[:,1],marker='x',s=70,color='black',label='posterior mean')
            ax.set(title=f"{name.upper()}, case {i}, EIG score {r['spce']:.2f}, MSE {r['inference']['mse']:.2f}",xlabel='x',ylabel='y')
            ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.15);ax.legend(fontsize=7)
            fig.colorbar(dots,ax=ax,label='measurement stage',shrink=.6)
            geom=split_geometry(samples)
            report.append(dict(method=name,trajectory_id=i,posterior_particles=len(samples),geometry=geom,
                               posterior_alignment='Saved samples already aligned to a MAP-particle reference; final MSE minimizes source-label permutations.',
                               mse=r['inference']['mse'],realized_information=r['spce']))
    fig.tight_layout();fig.savefig(out/'representative_failures.png',dpi=150);plt.close(fig)
    (out/'posterior_geometry.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
