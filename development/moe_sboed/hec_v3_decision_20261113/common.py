"""Use only the hash-verified frozen V2 loader and native implementation."""
import pathlib,sys
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
SOURCE=WORK/'development/moe_sboed/hec_v2_specialization_audit_20261113'
sys.path.insert(0,str(SOURCE))
import audit
n=audit.native
