#!/usr/bin/env python3
"""Operate from MacBook: validated LabPC commit/push, clean Ubuntu ff-only pull."""
import argparse,json,pathlib,subprocess,sys,time
BRANCH='research/hec-moe-sboed-20261109'
LAB='Documents/objective_power_grid_temp/development/hec_git_sync'
WORKER='Documents/objective_power_grid_temp/development/hec_worker_sync'
LAB_TASK=r'''
import ast,fcntl,hashlib,json,pathlib,re,subprocess,sys
p=pathlib.Path.home()/cfg['path']
def git(*args,check=True):return subprocess.run(['git','-C',str(p),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=check)
assert git('branch','--show-current').stdout.strip()==cfg['branch'],'Wrong authoritative branch'
url=git('remote','get-url','origin').stdout.strip();assert url in ['git@github.com:alexschmidt123/objective_power_grid.git','https://github.com/alexschmidt123/objective_power_grid.git'],'Unexpected repository'
gitdir=pathlib.Path(git('rev-parse','--absolute-git-dir').stdout.strip());lock=(gitdir/'moe_sync.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
selected=set()
for name in cfg['files']:
 q=(p/name).resolve();assert q.is_relative_to(p.resolve()),'Path outside repository'
 if not q.exists():
  git('ls-files','--error-unmatch',name);selected.add(name);continue
 for f in (q.rglob('*') if q.is_dir() else [q]):
  if not f.is_file() or '__pycache__' in f.parts:continue
  assert not f.is_symlink(),'Symlink forbidden';selected.add(str(f.relative_to(p)))
changed=set(git('diff','--name-only','HEAD','-z').stdout.strip('\0').split('\0'))-{''};untracked=set(git('ls-files','--others','--exclude-standard','-z').stdout.strip('\0').split('\0'))-{''}
assert (changed|untracked)<=selected or not (changed|untracked),'Unreviewed changes outside --files;preserved: '+str(sorted((changed|untracked)-selected))
if selected:
 assert cfg['message'],'Descriptive --message required'
 for name in sorted(selected):
  f=p/name
  if not f.exists():print('VALIDATED_DELETION',name,flush=True);continue
  assert f.stat().st_size<=1048576,'Oversized file: '+name
  assert not any(part in ['.git','.ssh','inputs','arms','preparation','evaluation','TRAINING_LOGS','evidence','data','datasets','models','checkpoints'] for part in pathlib.Path(name).parts),'Output/private path: '+name
  assert f.suffix in ['.py','.md','.json','.toml','.yaml','.yml','.sh'] or f.name in ['.gitignore','requirements.txt'],'Non-source file: '+name
  content=f.read_text();assert not re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}',content),'Possible credential: '+name
  if f.suffix=='.py':compile(content,name,'exec')
  if f.suffix=='.json':json.loads(content)
  print('VALIDATED',name,hashlib.sha256(content.encode()).hexdigest(),flush=True)
 existing=set(git('diff','--cached','--name-only','-z').stdout.strip('\0').split('\0'))-{''};assert existing<=selected,'Unrelated staged work;preserved'
 normal=sorted(selected-{'AGENTS.md'})
 if normal:git('add','--',*normal)
 if 'AGENTS.md' in selected:git('add','-f','--','AGENTS.md')
 whitespace=git('diff','--cached','--check',check=False)
 if whitespace.returncode:
  assert whitespace.returncode==2,'Git diff validation failed: '+whitespace.stderr
  headers=re.findall(r'^([^:\n]+):\d+: ',whitespace.stdout,re.M);assert headers,'Unrecognized whitespace review output'
  for name in headers:
   candidates=[parent for parent in (p/name).parents if parent!=p and p in parent.parents and (parent/'SOURCE_PROVENANCE.json').is_file()]
   assert candidates,'New whitespace error outside an immutable snapshot: '+name
   parent=candidates[0];provenance=json.loads((parent/'SOURCE_PROVENANCE.json').read_text())['files'];relative=str((p/name).relative_to(parent))
   assert relative in provenance,'New whitespace error: '+name
   assert hashlib.sha256((p/name).read_bytes()).hexdigest()==provenance[relative],'Changed snapshot whitespace: '+name
  print('REVIEWED inherited frozen-source whitespace;original bytes retained',whitespace.stdout,flush=True)
 assert not git('diff','--name-only').stdout.strip(),'Working copy changed during validation'
 patch=git('diff','--cached','--binary').stdout
 audit=pathlib.Path.home()/'Documents/objective_power_grid_temp/development/workflow_migration_20261109';audit.mkdir(exist_ok=True,parents=True);(audit/'REVIEWED_PATCH.diff').write_text(patch)
 print(git('diff','--cached','--stat').stdout,flush=True)
 if cfg['review_only']:
  print('SYNC_RESULT '+json.dumps(dict(status='REVIEW_READY',patch=str(audit/'REVIEWED_PATCH.diff'))));sys.exit(0)
 if patch:print(git('commit','-m',cfg['message']).stdout,flush=True)
assert not git('status','--porcelain').stdout.strip(),'Authoritative worktree not clean'
head=git('rev-parse','HEAD').stdout.strip();print(git('push','-u','origin','HEAD:refs/heads/'+cfg['branch']).stdout,flush=True)
remote=git('ls-remote','--heads','origin','refs/heads/'+cfg['branch']).stdout.split();assert remote and remote[0]==head,'Remote commit mismatch'
print('SYNC_RESULT '+json.dumps(dict(status='PUSHED',commit=head,branch=cfg['branch'],repository='https://github.com/alexschmidt123/objective_power_grid',worktree=str(p),clean=True)))
'''
WORKER_TASK=r'''
import pathlib,subprocess,json
p=pathlib.Path.home()/cfg['path'];base=pathlib.Path.home()/'Documents/objective_power_grid'
def git(root,*args):return subprocess.run(['git','-C',str(root),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
url=git(base,'remote','get-url','origin').stdout.strip();assert url in ['git@github.com:alexschmidt123/objective_power_grid.git','https://github.com/alexschmidt123/objective_power_grid.git'],'Unexpected worker origin'
# Fetch changes references only; existing dirty checkouts are never switched or reset.
refspec='refs/heads/'+cfg['branch']+':refs/remotes/origin/'+cfg['branch']
# Existing Ubuntu repositories may fetch only main. Add this branch without
# replacing existing fetch rules or forcing any local branch/reference.
current=git(base,'config','--get-all','remote.origin.fetch').stdout.splitlines()
if refspec not in current:git(base,'config','--add','remote.origin.fetch',refspec)
git(base,'fetch','origin',refspec);remote=git(base,'rev-parse','origin/'+cfg['branch']).stdout.strip();assert remote==cfg['expected'],'Remote advanced;preserve and retry from LabPC'
if not p.exists():
 p.parent.mkdir(parents=True,exist_ok=True);assert not git(base,'branch','--list',cfg['branch']).stdout.strip(),'Local branch exists elsewhere;preserved'
 print(git(base,'worktree','add','--track','-b',cfg['branch'],str(p),'origin/'+cfg['branch']).stdout,flush=True)
assert git(p,'branch','--show-current').stdout.strip()==cfg['branch'],'Wrong worker branch'
assert not git(p,'status','--porcelain').stdout.strip(),'Worker has uncommitted work;preserved,not overwritten'
print(git(p,'pull','--ff-only','origin',cfg['branch']).stdout,flush=True)
head=git(p,'rev-parse','HEAD').stdout.strip();assert head==cfg['expected'],'Worker commit mismatch'
assert not git(p,'status','--porcelain').stdout.strip(),'Worker unexpectedly dirty'
print('SYNC_RESULT '+json.dumps(dict(status='SYNCHRONIZED',commit=head,branch=cfg['branch'],worktree=str(p),clean=True,no_jobs_started_or_interrupted=True)))
'''
def remote(host,task,cfg):
 script='cfg='+repr(cfg)+'\n'+task
 r=subprocess.run(['ssh','-q','-o','BatchMode=yes',host,'python3 -'],input=script,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
 if r.returncode:raise RuntimeError(f'{host} synchronization step failed; existing work preserved; exit {r.returncode}')
 lines=[s for s in r.stdout.splitlines() if s.startswith('SYNC_RESULT ')]
 if not lines:raise RuntimeError('Missing verified synchronization receipt')
 return json.loads(lines[-1][len('SYNC_RESULT '):])
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--message',default='');parser.add_argument('--files',nargs='*',default=[]);parser.add_argument('--review-only',action='store_true');parser.add_argument('--receipt',default=str(pathlib.Path.home()/'scratch_runs/moe_workflow_sync_20261109/LATEST_SYNC.json'));args=parser.parse_args()
 lab=remote('labpc',LAB_TASK,dict(path=LAB,branch=BRANCH,files=args.files,message=args.message,review_only=args.review_only))
 if args.review_only:return
 worker=remote('ubuntu-pc',WORKER_TASK,dict(path=WORKER,branch=BRANCH,expected=lab['commit']))
 # Verify LabPC has not advanced while the worker was pulling.
 r=subprocess.run(['ssh','-q','-o','BatchMode=yes','labpc','git -C '+str(pathlib.PurePosixPath('~/')/LAB)+' rev-parse HEAD'],capture_output=True,text=True,check=True);assert r.stdout.strip()==lab['commit']==worker['commit'],'LabPC advanced during sync;report incomplete'
 receipt=dict(status='VERIFIED',time=time.time(),repository=lab['repository'],branch=BRANCH,labpc=lab,ubuntu=worker,github_commit=lab['commit'],worker_compute_launched=False)
 p=pathlib.Path(args.receipt);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(receipt,indent=2)+'\n');print('VERIFIED_SYNC '+json.dumps(receipt))
if __name__=='__main__':
 try:main()
 except Exception as exc:print('SYNC_ERROR: '+str(exc),file=sys.stderr);sys.exit(1)
