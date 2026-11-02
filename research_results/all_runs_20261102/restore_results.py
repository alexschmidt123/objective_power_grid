"""Restore original result paths from one host's deduplicated Git snapshot."""
import argparse,csv,gzip,hashlib,pathlib
p=argparse.ArgumentParser();p.add_argument('snapshot',type=pathlib.Path);p.add_argument('destination',type=pathlib.Path);p.add_argument('--match',default='');a=p.parse_args()
count=0
with (a.snapshot/'INDEX.csv').open() as f:
 for row in csv.DictReader(f):
  original=pathlib.Path(row['scope'])/row['original_path']
  assert not original.is_absolute() and '..' not in original.parts
  if a.match not in str(original):continue
  source=a.snapshot/row['published_path']
  assert source.resolve().is_relative_to(a.snapshot.resolve())
  data=b''.join(x.read_bytes() for x in sorted(source.glob('part*'))) if source.is_dir() else source.read_bytes()
  if row['encoding'].startswith('gzip'):data=gzip.decompress(data)
  assert hashlib.sha256(data).hexdigest()==row['sha256'],f'Hash mismatch: {original}'
  target=a.destination/original;assert not target.exists(),f'Refusing overwrite: {target}';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  count+=1
assert count,'No indexed result files matched the requested filter.'
print(f'{count} result files restored and SHA256 verified.')
