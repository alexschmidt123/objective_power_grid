"""Render saved FH-MoE-v3 discovery diagnostics; never trains or selects models."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args();c=Path(a.campaign);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 gate=json.loads((c/'context_gate.json').read_text());train=json.loads((c/'training_progress.json').read_text());stages=list(gate['stages']);r=[gate['stages'][t] for t in stages]
 fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained');fig.suptitle('FH-MoE-v3: correct optimization, insufficient context signal',fontsize=15)
 ax=axs[0,0]
 for key,label in [('shared_reference','Shared-only reference'),('native_control','Matched native-v1')]:
  ax.plot([v['update'] for v in train['records']],[v[key]['sPCE'] for v in train['records']],label=label)
 ax.set(xlabel='Preparation optimizer updates per arm',ylabel='Native training sPCE (nats)',title='Two fresh development arms');ax.legend(fontsize=8)
 for ax,values,threshold,title,ylabel in [
  (axs[0,1],[min(v for x in t['contexts'] for v in x['coherence']) for t in r],.1,'Weak mean-gradient coherence','Minimum split mean unit-gradient norm'),
  (axs[1,0],[min(v for x in t['contexts'] for v in x['mean_gradient_SNR']) for t in r],3,'At least one weak context at every stage','Minimum split mean-gradient SNR'),
  (axs[1,1],[t['assignment_kappa'] for t in r],.6,'Assignment replication at later stages','Independent-fit assignment kappa')]:
  ax.bar(stages,values,color=['#a85148' if v<threshold else '#477f77' for v in values]);ax.axhline(threshold,color='#333333',ls='--',lw=1,label='Predeclared minimum');ax.set(xlabel='Observed-history stage',ylabel=ylabel,title=title);ax.legend(fontsize=8)
 for ax in axs.flat:ax.spines[['top','right']].set_visible(False)
 fig.savefig(out/'FH_MOE_V3_DISCOVERY.png',dpi=180);fig.savefig(out/'FH_MOE_V3_DISCOVERY.svg');plt.close(fig)
if __name__=='__main__':main()
