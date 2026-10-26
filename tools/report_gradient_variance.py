"""Generate figures/tables from saved diagnostic summaries; no GPU computation."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args();C=Path(a.campaign);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 d=json.loads((C/'variance_results.json').read_text());old=json.loads((C/'saved_signal_audit.json').read_text());stages=['1','10','20','29'];names=['single_iid_L2000','mean4_iid_L2000','mean4_iid_pairs_L2000','mean4_antithetic_pairs_L2000'];labels=['Single bank','4 IID banks','8 IID banks','4 antithetic pairs'];colors=['#555555','#247c9e','#39825f','#b46b2f']
 rows=[]
 for name in names:
  for t in stages:
   r=d['gates'][name][t];rows.append(dict(estimator=name,stage=t,remaining_horizon=30-int(t),minimum_coherence=min(v for c in r['contexts'] for v in c['coherence']),minimum_SNR=min(v for c in r['contexts'] for v in c['mean_gradient_SNR']),assignment_kappa=r['assignment_kappa'],gradient_MSE_difference=r['gradient_prediction_vs_constant']['mean'],gate=r['PASS']))
 with (out/'FH_MOE_VARIANCE_NUMBERS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained');fig.suptitle('Frozen-policy gradient audit: variance reduction does not validate multistage specialization',fontsize=14)
 x=np.arange(4)
 for idx,(key,title,threshold) in enumerate([('minimum_coherence','Weakest context coherence',.1),('minimum_SNR','Weakest context mean-gradient SNR',3),('assignment_kappa','Independent-fit assignment kappa',.6)]):
  a=ax[0,idx]
  for n,label,color in zip(names,labels,colors):
   values=[r[key] for r in rows if r['estimator']==n];a.plot(x,values,'o-',label=label,color=color,markersize=4)
  a.axhline(threshold,c='black',ls='--',lw=1,label='V3 minimum');a.set(title=title,xticks=x,xticklabels=stages,xlabel='History stage')
  if idx==0:a.legend(fontsize=7)
 a=ax[1,0];vs=[d['variance'][t] for t in stages];m=np.array([r['pair_variance_ratio'] for r in vs]);lo=np.array([r['pair_variance_ratio_CI95'][0] for r in vs]);hi=np.array([r['pair_variance_ratio_CI95'][1] for r in vs]);a.errorbar(x,m,yerr=[m-lo,hi-m],fmt='o',capsize=4,color=colors[3]);a.axhline(1,c='black',ls='--',lw=1);a.set(title='Antithetic / IID-pair trace variance',xticks=x,xticklabels=stages,xlabel='History stage',ylabel='Ratio with paired bootstrap 95% interval')
 a=ax[1,1]
 for key,label in [('p01','1st percentile'),('median','Median'),('p99','99th percentile')]:a.plot(x,[old['stages'][t]['all_raw']['norm_quantiles'][key] for t in stages],'o-',label=label)
 a.set(yscale='log',title='Saved raw-gradient norm spread',xticks=x,xticklabels=stages,xlabel='History stage',ylabel='Raw adjoint norm');a.legend(fontsize=8)
 a=ax[1,2];a.bar(x,[d['variance'][t]['contrast_fraction_of_raw_variance'] for t in stages],color=colors[1]);a.set(ylim=(0,1),title='Estimated contrast share of raw variance',xticks=x,xticklabels=stages,xlabel='History stage',ylabel='Corrected moment estimate')
 for a in ax.flat:a.spines[['top','right']].set_visible(False)
 fig.savefig(out/'FH_MOE_GRADIENT_VARIANCE.png',dpi=180);fig.savefig(out/'FH_MOE_GRADIENT_VARIANCE.svg');plt.close(fig)
if __name__=='__main__':main()
