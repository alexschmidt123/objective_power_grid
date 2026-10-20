"""Continue only scientifically unchanged phases in their original frozen tree."""
import argparse,importlib.util,os,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',required=True);a=p.parse_args()
r=Path.home()/'Documents/objective_power_grid_e1_sir_replication_20261019'
os.chdir(r);sys.path.insert(0,str(r))
# Existing controller/journal infrastructure and exact original source guard.
spec=importlib.util.spec_from_file_location('frozen_queue',r/'tools/e1_sir_replication.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.TASKS={'labpc':['fixed','switchmoe'],'ubuntu-pc':['base','resmoe','switchresmoe','moe']}
# Allow the already active compatible worker learner to finish; no duplicate.
import subprocess
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(20)
m.run(a.host)
