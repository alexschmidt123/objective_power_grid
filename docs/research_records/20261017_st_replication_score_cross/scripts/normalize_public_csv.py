import csv,io,json,hashlib,shutil
from pathlib import Path
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_st_replication_score_reverse_cross');D=Path('/home/grads/g/g.lin/Documents/objective_power_grid_sync_20261016/docs/research_records/20261017_st_replication_score_cross')
manifest=json.loads((D/'export_manifest.json').read_text());transforms=[]
for p in D.rglob('*.csv'):
 old=p.read_bytes();new=old.replace(b'\r\n',b'\n')
 assert list(csv.reader(io.StringIO(old.decode())))==list(csv.reader(io.StringIO(new.decode())))
 if old!=new:
  p.write_bytes(new);rel=str(p.relative_to(D));manifest['sha256'][rel]=hashlib.sha256(new).hexdigest();transforms.append(dict(path=rel,source_sha256=hashlib.sha256(old).hexdigest(),published_sha256=hashlib.sha256(new).hexdigest(),parsed_rows_identical=True))
script=D/'scripts/normalize_public_csv.py';shutil.copy2(C/'scripts/normalize_public_csv.py',script);manifest['sha256'][str(script.relative_to(D))]=hashlib.sha256(script.read_bytes()).hexdigest()
manifest['csv_publication_line_endings']=dict(transformation='CRLF to LF only; campaign originals unchanged',files=transforms)
(D/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PUBLIC_CSV_FORMAT_ONLY_NORMALIZATION_VERIFIED',len(transforms))
