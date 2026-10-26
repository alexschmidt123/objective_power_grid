
"""Freeze shared evaluation draws before either independent main run."""
import json,re,subprocess,hashlib
from pathlib import Path
import numpy as np,torch,yaml
from src.research.journal_validation import CommonValidation
from src.research.deepseek_sboed_experiment import write,sha
def audit_seed(seed):
 root=Path.home()/'Documents';hits=[];scanned=0
 for project in root.glob('objective_power_grid*'):
  if project.name.startswith('objective_power_grid_v1_specialization'):continue
  for p in project.glob('experiments/**/*'):
   if not p.is_file() or p.suffix not in ['.json','.yaml','.yml'] or p.stat().st_size>2000000:continue
   try:
    scanned+=1
    if re.search(r'(?<![0-9.])'+str(seed)+r'(?![0-9.])',p.read_text()):hits.append(str(p))
   except (UnicodeError,OSError):pass
 return dict(seed=seed,metadata_files_scanned=scanned,hits=hits,scope='Existing experiment JSON/YAML <=2MB, excluding this new project; integer-token match avoids decimal substrings')
def main():
 cfg=yaml.safe_load(Path('reports/DEEPSEEK_V1_SPECIALIZATION_CONFIG.yaml').read_text())
 R=Path.home()/'Documents/deepseek_v1_specialization_banks';R.mkdir(exist_ok=False)
 for name,seed,N,L in [('periodic',cfg['development_seed'],128,262144),('development',cfg['final_development_seed'],512,262144),('confirmation',cfg['confirmation_seed'],2000,1000000)]:
  if name=='confirmation':
   audit=audit_seed(seed);assert not audit['hits'],audit
   write(R/'confirmation_unused_seed_audit.json',audit)
  v=CommonValidation(cfg['benchmark'],30,seed,N,L,device='cpu')
  d=R/name;d.mkdir()
  np.savez(d/'cases.npz',truth=v.truth.numpy(),noise=v.noise.numpy());np.save(d/'contrasts.npy',v.bank.numpy())
  write(d/'manifest.json',dict(seed=seed,N=N,L=L,horizon=30,paired_draws_sha256=v.identity,file_SHA256={p.name:sha(p) for p in d.iterdir()},generated_once_on_UbuntuPC_before_main_training=True,torch_version=torch.__version__))
 print('FROZEN_BANKS',R,flush=True)
if __name__=='__main__':main()
