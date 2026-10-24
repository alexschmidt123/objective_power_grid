"""Reserve unscored confirmation arrays; audit metadata, not undocumented RNGs."""
import os,json,time,hashlib,argparse
from pathlib import Path
import yaml,numpy as np
from src.research.journal_validation import CommonValidation

def contains_seed(value,key='',seed=1043):
 if isinstance(value,dict):return any(contains_seed(v,k,seed) for k,v in value.items())
 if isinstance(value,list):return any(contains_seed(v,key,seed) for v in value)
 return 'seed' in str(key).lower() and value==seed

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--reference',required=True);a=p.parse_args();C=Path(a.campaign).resolve();start=time.perf_counter();matches=[];scanned=large=0;errors=[]
 prune={'source','frozen_source','.git','models','logs','figures','scripts','vendor','__pycache__','site-packages','frozen','source_snapshot'}
 for repo in Path('/home/grads/g/g.lin/Documents').glob('objective_power_grid*'):
  root=repo/'experiments'
  if not root.is_dir():continue
  for directory,dirs,files in os.walk(root):
   dirs[:]=[x for x in dirs if x not in prune]
   if Path(directory).is_relative_to(C):dirs[:]=[];continue
   for filename in files:
    if not filename.endswith(('.json','.yaml','.yml')) or not any(t in filename.lower() for t in ['config','manifest','registry','result','summary','freeze','seed','evaluation','protocol']):continue
    path=Path(directory)/filename
    try:
     if path.stat().st_size>2_000_000:large+=1;continue
     data=json.loads(path.read_text()) if path.suffix=='.json' else yaml.safe_load(path.read_text());scanned+=1
     if contains_seed(data):matches.append(str(path))
    except (ValueError,UnicodeError,yaml.YAMLError,OSError) as exc:errors.append(dict(file=str(path),error=str(exc)))
 report=dict(candidate_seed=1043,prior_seed_matches=matches,metadata_files_scanned=scanned,large_files_skipped=large,parse_errors=errors,scope='Existingcampaignmetadata/configs/reports<=2MB; sourcecopies/models/logsexcluded. Doesnotcertify undocumentedrandomdraws.',seconds=time.perf_counter()-start)
 (C/'source_manifest/confirmation_prior_use_audit.json').write_text(json.dumps(report,indent=2));print('AUDIT_COUNTS',scanned,large,'MATCHES',matches,'ERRORS',len(errors),flush=True)
 if matches:raise RuntimeError('Reservedseed previouslydeclared; do notgenerate oruseit')
 cfg=yaml.safe_load((Path(a.reference)/'configs/seed101.yaml').read_text());v=CommonValidation(cfg['benchmark'],30,1043,2000,1000000,batch=16,chunk=16384,device='cpu');directory=C/'evaluations/confirmation_reserved';directory.mkdir(exist_ok=True)
 if (directory/'manifest.json').exists():raise RuntimeError('Existingbankpreserved; nooverwrite')
 np.savez(directory/'cases.npz',truth=v.truth.numpy(),noise=v.noise.numpy());np.save(directory/'contrasts.npy',v.bank.numpy());hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in directory.iterdir() if x.is_file()}
 (directory/'manifest.json').write_text(json.dumps(dict(seed=1043,trajectories=2000,contrasts=1000000,paired_draws_sha256=v.identity,file_SHA256=hashes,unscored_before_v3freeze=True,selected_before_v3training=True),indent=2));print('BANK_HASHES',hashes)
if __name__=='__main__':main()
