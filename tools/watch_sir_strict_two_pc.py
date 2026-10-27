"""Long-running SSH relay for the already authorized strict SIR queue.
Run on the controlling Mac; all commits/pushes execute ONLY on LabPC.
No training source is edited. Only completed immutable runs are transferred.
"""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path
U='/home/qiqi/Documents/objective_power_grid_sir_strict_seed101/experiments/10262026/10262026_sir_strict_seed101'
L='/home/grads/g/g.lin/Documents/objective_power_grid_sir_strict_reports';H='/home/grads/g/g.lin/Documents/objective_power_grid_sir_seed101_recovery/reports';M=L+'/experiments/10262026/10262026_sir_strict_seed101/mirror';COMMIT='16632ae4b00703ffb1cc4d9b6a956e4a953f00c7'
def cmd(args,timeout=180):
 p=subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout);return p.stdout

def ssh(host,text,timeout=180):return cmd(['ssh','-o','BatchMode=yes',host,text],timeout)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--once',action='store_true');a=p.parse_args();C=Path(a.cache);C.mkdir(parents=True,exist_ok=True);statefile=C/'relay_state.json';state=json.loads(statefile.read_text()) if statefile.exists() else dict(imported={},signature=None);deadline=time.monotonic()+15*86400
 while time.monotonic()<deadline:
  try:
   raw=ssh('ubuntu-pc','cat '+shlex.quote(U+'/status.json'));status=json.loads(raw);registry=json.loads(ssh('ubuntu-pc','cat '+shlex.quote(U+'/registry.json')))
   for entry in registry:
    m=entry['method']
    if entry['probe'] or entry['status']!='COMPLETE' or m in state['imported']:continue
    archive=f'/tmp/sir_strict_{m}_{COMMIT[:12]}.tar.gz'
    remote=ssh('ubuntu-pc','test -f '+shlex.quote(archive)+' || tar -czf '+shlex.quote(archive)+' -C '+shlex.quote(U)+' '+shlex.quote('runs/'+m))
    expected=ssh('ubuntu-pc','sha256sum '+shlex.quote(archive)).split()[0];local=C/(m+'.tar.gz')
    if local.exists():assert digest(local)==expected,'Archive collision; preserve existing cache'
    else:
     partial=C/(m+'.transfer.part');cmd(['scp','-q','ubuntu-pc:'+archive,str(partial)],timeout=600);assert digest(partial)==expected;partial.rename(local)
    labarchive='/tmp/sir_strict_relay_'+m+'_'+expected+'.tar.gz';cmd(['scp','-q',str(local),'labpc:'+labarchive],timeout=600)
    receipt=M+'/'+m+'_import_receipt.json';script='from pathlib import Path;import json; p=Path('+repr(receipt)+');p.write_text(json.dumps('+repr(dict(method=m,archive_sha256=expected,origin=U+'/runs/'+m,training_commit=COMMIT))+',indent=2))'
    # Each full phase is imported once; never overwrite a differing mirror.
    ssh('labpc','mkdir -p '+shlex.quote(M)+' && test ! -e '+shlex.quote(M+'/runs/'+m)+' && tar -xzf '+shlex.quote(labarchive)+' -C '+shlex.quote(M)+' && python3 -c '+shlex.quote(script),timeout=600)
    state['imported'][m]=expected;statefile.write_text(json.dumps(state,indent=2))
   signature=json.dumps(dict(status=status,completed=sorted(state['imported']),failures=[(x['method'],x['probe'],x['status']) for x in registry if x['status']!='COMPLETE']),sort_keys=True)
   if signature!=state['signature']:
    local=C/'remote_status.json';local.write_text(json.dumps(status,indent=2));ssh('labpc','mkdir -p '+shlex.quote(M));cmd(['scp','-q',str(local),'labpc:'+M+'/remote_status.json'])
    command='cd '+shlex.quote(L)+' && /home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin/python -m tools.strict_sir_report --historical '+shlex.quote(H)+' --mirror '+shlex.quote(M)+' --status '+shlex.quote(M+'/remote_status.json')
    output=ssh('labpc',command);print(output.strip(),flush=True)
    files=['reports/SIR_SEED101_ALL_BASELINES.md','reports/SIR_SEED101_ALL_BASELINES.csv','reports/SIR_SEED101_COMPUTE.csv','reports/SIR_SEED101_PROVENANCE.json','reports/SIR_SEED101_PAIRED_SCORES.npz']
    quoted=' '.join(shlex.quote(x) for x in files);message='Update strict SIR seed101 verified completion: '+str(6+len(state['imported']))+'/10'
    publish='cd '+shlex.quote(L)+' && git add -f '+quoted+' && (git diff --cached --quiet || git commit -m '+shlex.quote(message)+') && git push origin reports/sir-strict-seed101-20261026'
    ssh('labpc',publish);state['signature']=signature;state['last_status']=status;state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());statefile.write_text(json.dumps(state,indent=2));print('PUBLISHED',status,flush=True)
   if status.get('state')=='QUEUE_FINISHED':print('RELAY_FINISHED',status,flush=True);return
   if a.once:return
   time.sleep(60)
  except Exception as exc:
   state['last_error']=repr(exc);state['error_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());statefile.write_text(json.dumps(state,indent=2));print('RELAY_ERROR',repr(exc),flush=True)
   if a.once:raise
   time.sleep(60)
 raise TimeoutError('15day relay safety cap reached; training queue untouched')
if __name__=='__main__':main()
