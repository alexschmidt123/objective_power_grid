import pathlib,sys
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_v2_specialization_audit_20261113'))
import audit
n=audit.native
