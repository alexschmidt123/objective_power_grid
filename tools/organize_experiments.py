"""Group completed experiment campaigns by MMDDYYYY; retain an audited path map.

Run without --apply to preview. No experiment contents or historical metadata
are rewritten. Symlinks are retargeted where a moved path would break them.
"""
import argparse
from collections import Counter
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import re


def destination_name(name):
    if name=='single_step_eig_curves':
        # Creation date established by the saved September 13 plotting audit.
        return '09132026','09132026_sir_ode_ieee9_EIG_single_step_curves'
    m=re.match(r'^(\d{8})_',name)
    if m:
        datetime.strptime(m[1],'%m%d%Y')
        return m[1],name
    m=re.search(r'(?<!\d)(20\d{6})(?!\d)',name)
    if m:
        day=datetime.strptime(m[1],'%Y%m%d').strftime('%m%d%Y')
        desc=(name[:m.start()]+name[m.end():]).replace('__','_').strip('_')
        return day,day+'_'+desc
    raise ValueError(f'No verified start date for {name}; do not infer it from copy/mtime.')


def fingerprint(path):
    """Record regular-file identity and metadata, without following symlinks."""
    rows={}
    for parent,dirs,files in os.walk(path,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(parent)/d).is_symlink()]
        for name in files:
            p=Path(parent)/name
            if p.is_symlink():continue
            s=p.stat();rows[str(p.relative_to(path))]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('experiments'))
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    moves=[]
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.is_symlink():continue
        if re.fullmatch(r'\d{8}',path.name):
            datetime.strptime(path.name,'%m%d%Y');continue
        day,name=destination_name(path.name)
        dest=root/day/name
        if dest.exists():raise FileExistsError(dest)
        moves.append((path,dest))
    assert len({d for _,d in moves})==len(moves)
    print(json.dumps({'campaigns':len(moves),'date_counts':dict(sorted(Counter(d.parent.name for _,d in moves).items())),
        'renames':[(s.name,d.name) for s,d in moves if s.name!=d.name]},indent=2),flush=True)
    if not a.apply or not moves:return
    # Refuse to move the root of an active experiment process.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if 'python' in cmd and any(token in cmd for token in ['-m src.online_cli','-m src.objectives.eig.continuous_eig','-m src.experiment']):
            raise RuntimeError('Active experiment process: '+proc.name)
    mapping={str(s):str(d) for s,d in moves}
    def relocate(path):
        text=str(path)
        for old,new in mapping.items():
            if text==old or text.startswith(old+os.sep):return Path(new+text[len(old):])
        return Path(text)
    links=[]
    for parent,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            link=Path(parent)/name
            if not link.is_symlink():continue
            target=os.readlink(link)
            absolute=Path(os.path.normpath(target if os.path.isabs(target) else str(link.parent/target)))
            links.append((link,absolute,target,link.exists()))
        dirs[:]=[d for d in dirs if not (Path(parent)/d).is_symlink()]
    record=root/'relocation_manifest.json'
    if record.exists():raise FileExistsError(record)
    manifest={'date_format':'MMDDYYYY','created_at':datetime.now().isoformat(),
              'status':'in_progress','mapping':mapping,'completed':[],
              'historical_metadata':'Saved commands, hashes and absolute paths remain original provenance. Resolve relocated paths using this mapping.',
              'symlink_changes':[]}
    def save():
        temp=record.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(record)
    save()
    total_files=total_bytes=0
    for old,new in moves:
        before=fingerprint(old)
        new.parent.mkdir(exist_ok=True)
        old.rename(new)
        assert before==fingerprint(new),f'File inventory changed in {new}'
        total_files+=len(before);total_bytes+=sum(x[2] for x in before.values())
        manifest['completed'].append(str(new));save()
    for old,target,text,existed in links:
        new=relocate(old);new_target=relocate(target)
        if new_target!=target or (new!=old and not os.path.isabs(text)):
            new.unlink();new.symlink_to(os.path.relpath(new_target,new.parent))
            manifest['symlink_changes'].append({'path':str(new),'old_target':text,'new_target':os.readlink(new)})
        if existed:assert new.exists(),f'Previously valid link broke: {new}'
    manifest.update(status='complete',verified_regular_files=total_files,verified_bytes=total_bytes)
    save()
    with (root/'INDEX.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['start_date','campaign','relative_path','previous_name'])
        for old,new in moves:w.writerow([new.parent.name,new.name,str(new.relative_to(root)),old.name])
    print(json.dumps({'status':'complete','campaigns':len(moves),'verified_regular_files':total_files,
                      'repaired_links':len(manifest['symlink_changes'])}),flush=True)

if __name__=='__main__':main()
