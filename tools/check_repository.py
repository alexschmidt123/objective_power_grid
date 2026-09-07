#!/usr/bin/env python3
"""Check canonical folders, manuscript source consistency, and local imports.

Source-only: never invokes LaTeX or rebuilds a PDF.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]


def check():
    expected={'ieee9_eig.yaml','ieee9_mocu.yaml','ieee14_mocu.yaml',
              'ieee30_mocu.yaml','sir_ode_eig.yaml'}
    actual={p.name for p in (ROOT/'configs').iterdir()}
    assert actual==expected, ('Unexpected configs layout',actual ^ expected)
    expected_docs={'objective_driven_boed_manuscript_framework.tex','sBOED_design.tex',
                   'objective_driven_boed_refs.bib','objective_driven_boed_manuscript_framework.pdf',
                   'images','papers'}
    assert {p.name for p in (ROOT/'documents').iterdir()}==expected_docs, 'Unexpected documents root'
    manuscripts=[(ROOT/'documents'/n).read_text() for n in sorted(expected_docs) if n.endswith('.tex')]
    blocks=[re.search(r'% BEGIN SHARED FINITE-LOSS MOCU\n(.*?)% END SHARED FINITE-LOSS MOCU',s,re.S).group(1)
            for s in manuscripts]
    assert len(blocks)==2 and blocks[0]==blocks[1], 'MOCU formulations differ'
    bib=(ROOT/'documents/objective_driven_boed_refs.bib').read_text()
    keys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',bib))
    for text in manuscripts:
        cites={k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',text) for k in group.split(',')}
        assert cites<=keys, ('Missing bibliography entries',cites-keys)
        labels=re.findall(r'\\label\{([^}]+)\}',text)
        assert len(labels)==len(set(labels)), 'Duplicate equation labels'
        refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text))
        assert refs<=set(labels), ('Missing labels',refs-set(labels))
        assert '\\input{mocu_finite_loss_formulation}' not in text
    manifest=json.loads((ROOT/'documents/papers/manifest.json').read_text())
    for key in set().union(*[{k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',s) for k in group.split(',')} for s in manuscripts]):
        entry=manifest['references'][key]
        pdf=ROOT/'documents/papers'/entry['file']
        raw=pdf.read_bytes()
        assert raw.startswith(b'%PDF-'), ('Not a PDF',pdf)
        assert hashlib.sha256(raw).hexdigest()==entry['sha256'], ('PDF hash mismatch',pdf)
    for directory in ('src','tools'):
        for p in (ROOT/directory).rglob('*.py'):
            tree=ast.parse(p.read_text(),filename=str(p))
            for n in ast.walk(tree):
                names=([n.module] if isinstance(n,ast.ImportFrom) and not n.level and n.module
                       else [a.name for a in n.names] if isinstance(n,ast.Import) else [])
                for name in names:
                    if name.startswith('tools.'):
                        path=ROOT/Path(*name.split('.'))
                        assert path.is_dir() or path.with_suffix('.py').is_file(), ('Broken tools import',p,name)
    assert {p.name for p in (ROOT/'hprc').iterdir()}=={'environment.sh','experiment.slurm','master_audit.slurm'}
    print('Repository layout, manuscript references, PDF provenance, and tools imports: OK')


if __name__=='__main__':
    check()
