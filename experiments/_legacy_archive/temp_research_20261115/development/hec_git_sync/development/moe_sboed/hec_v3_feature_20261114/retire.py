"""Retire failed V3 active sources; preserve all historical bytes and checkpoint paths."""
import hashlib,json,pathlib,shutil,subprocess,tarfile,time
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
OLD=['development/moe_sboed/hec_v3_v2_pathways_20261113','development/moe_sboed/hec_v3_counterfactual_router_20261114']
DEST=WORK/'development/moe_sboed/retired/hec_v3_20261114'
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8388608),b''):h.update(part)
    return h.hexdigest()
def main():
    assert not DEST.exists()
    for path in OLD:subprocess.run(['git','-C',str(WORK),'diff','--quiet','HEAD','--',path],check=True)
    ps=subprocess.check_output(['ps','-u',str(__import__('os').getuid()),'-o','pid=,args='],text=True)
    for line in ps.splitlines():
        if any('/'+path+'/' in line for path in OLD) and ('python ' in line or 'python3 ' in line):
            raise RuntimeError('A retired-source process is still running; preserve it: '+line)
    archive=ROOT/'scratch_runs/retired_hec_v3_20261114';assert not archive.exists();archive.mkdir()
    files=subprocess.check_output(['git','-C',str(WORK),'ls-files','-z','--',*OLD],text=True).strip('\0').split('\0');assert len(files)==19
    protected_paths=[ROOT/'scratch_runs/hec_v3_v2_pathways_20261113_pilot'/name for name in ['selector/actor.pt','selector/runtime.pkl','INITIAL_DISTILLED.pt']]
    protected_paths += [ROOT/'scratch_runs/hec_v3_counterfactual_router_20261114_pilot02'/f'counterfactual_round{k}/actor.pt' for k in [1,2,3]]
    protected={str(p):sha(p) for p in protected_paths}
    originals={name:sha(WORK/name) for name in files}
    with tarfile.open(archive/'FAILED_V3_SOURCE.tar.gz','w:gz') as tar:
        for name in files:tar.add(WORK/name,arcname=name,recursive=False)
    DEST.mkdir(parents=True);mapping={};relative_hashes={}
    for name in files:
        src=WORK/name;relative=pathlib.Path(name).relative_to('development/moe_sboed');target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(src),str(target));assert sha(target)==originals[name]
        mapping[name]=str(target.relative_to(WORK));relative_hashes[str(relative)]=sha(target)
    for path in OLD:
        old=WORK/path;old.mkdir(exist_ok=True)
        (old/'RETIRED.md').write_text('This failed V3 implementation is retired. Frozen source: development/moe_sboed/retired/hec_v3_20261114/. Do not restart its training. Active replacement: development/moe_sboed/hec_v3_feature_20261114/. Historical checkpoints and results remain at their original scratch paths.\n')
    (DEST/'SOURCE_PROVENANCE.json').write_text(json.dumps(dict(status='RETIRED_NO_RESTART',files=relative_hashes,original_to_retired=mapping),indent=2)+'\n')
    (DEST/'README.md').write_text('# Retired failed V3 source\n\nImmutable reference source, retained for checkpoint loading and reproducibility. No automatic training or campaign restarts. All files listed in SOURCE_PROVENANCE.json retain their original SHA256 bytes. Active replacement is hec_v3_feature_20261114.\n')
    assert all(sha(path)==h for path,h in protected.items())
    receipt=dict(status='RETIRED',source_archive=str(archive/'FAILED_V3_SOURCE.tar.gz'),source_archive_sha256=sha(archive/'FAILED_V3_SOURCE.tar.gz'),
        original_to_retired=mapping,source_hashes=originals,protected_checkpoints=protected,running_jobs_interrupted=False,models_deleted=False)
    (archive/'RETIREMENT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    report=WORK/'reports/hec_v3_feature_20261114';report.mkdir(parents=True,exist_ok=True)
    (report/'RETIREMENT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status='RETIRED',files=len(files),archive=str(archive))))
if __name__=='__main__':main()
