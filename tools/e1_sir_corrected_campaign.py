"""Corrected SIR phase execution through existing journal orchestration/run.sh."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools import journal_campaign as j
C=R/'experiments/10192026/10192026_sir_e1_corrected_implementations'
V=R/'experiments/10192026/10192026_e1_sir_verification'
OLD=Path.home()/'Documents/objective_power_grid/experiments/10182026/10182026_e1_sir_t5_seed101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 assert (C/'metadata/actual_source_inventory.json').exists()
 assert not (C/'source_manifest.json').exists(),'Never refreeze an active experiment'
 for d in ['configs','logs','reports','metadata','evaluations/development_bank','frozen_source']:(C/d).mkdir(parents=True,exist_ok=True)
 raw=yaml.safe_load((R/'configs/e1_sir_native.yaml').read_text());raw['research']['settings']['eval_contrasts']=1000000
 raw['research']['journal_protocol']['status']='ScoreBED_SIR_native_dimension_correction_development'
 raw['research']['journal_protocol']['final_evaluation'].update(contrasts=1000000,resolution_status='provisional; full paired contrast-resolution verification ongoing')
 for cap in [500000,1000000]:
  cfg=copy.deepcopy(raw);cfg['research']['journal_protocol']['stopping_rule'].update(minimum_episodes=cap,maximum_episodes=cap);cfg['research']['paper_protocol']['nominal_training_trajectories']=cap
  (C/'configs'/f'sir_{cap}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 for name,src in [('cases.npz',V/'evaluations/cases.npz'),('contrasts.npy',V/'evaluations/contrasts_nested.npy')]:shutil.copy2(src,C/'evaluations/development_bank'/name)
 j.write(C/'registry.json',[]);j.write(C/'protocol.json',raw)
 (C/'journal_protocol.md').write_text('SIR E1 implementation correction: ScoreBED scalar-data/scalar-time potential-head degeneracy confirmed by frozen-model scores, reliable independent marginal-score references and pinned upstream documentation. Use existing native potential_hidden_dim256 compatibility option; unchanged4-layer256Transformer, joint marginal score objective, score-gradient policy learning, optimizer, allocation, native keys and EMA policy, one policy, no restart selection. Exactly1M total episodes; final-budget checkpoint. Train corrected ScoreBED101 only initially; other correct seed101 policies are preserved and require reevaluation only. Development2048 paired cases/seed1001, provisional1M contrasts pending full numerical resolution freeze. Validation1003 remains diagnostic. Seeds102-105 held until source audit and evaluator freeze. Confirmation1002 remains unused for SIR.\n')
 source=[*R.glob('src/**/*.py'),*R.glob('tools/**/*.py'),*R.glob('configs/*.yaml'),R/'run.sh',R/'sweep_run.sh',*C.glob('configs/*.yaml')]
 j.write(C/'source_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),sha256={str(p.relative_to(R)):sha(p) for p in source}))
 for p in source:
  out=C/'frozen_source'/p.relative_to(R);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
 j.write(C/'location_freeze.json',dict(protocol_sha256=sha(C/'protocol.json'),written_protocol_sha256=sha(C/'journal_protocol.md')))
 j.write(C/'reports/sir_corrected_source_manifest.json',json.loads((C/'source_manifest.json').read_text()))
 print(C)
def run_score():
 j.C=C;j.NAMES['rl_sboed']='Native RL-sBOED';j.guard()
 assert json.loads((C/'reports/score_diagnostic/score_potential_head_check.json').read_text())['status']=='PASS'
 entry=j.run_phase('sir',101,'scorebed',timeout_seconds=180000,evaluation_cases=C/'evaluations/development_bank/cases.npz',evaluation_bank=C/'evaluations/development_bank/contrasts.npy')
 j.write(C/'metadata/score101_complete.json',entry)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score101']);a=p.parse_args();prepare() if a.action=='prepare' else run_score()
