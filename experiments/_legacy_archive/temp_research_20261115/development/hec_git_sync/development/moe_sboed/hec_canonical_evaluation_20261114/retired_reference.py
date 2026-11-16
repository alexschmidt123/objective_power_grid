"""Read-only source loading for failed-V3 benchmark references; never launches jobs."""
import hashlib,json,pathlib,sys
def activate():
    work=pathlib.Path.home()/'Documents/objective_power_grid_temp/development/hec_git_sync'
    root=work/'development/moe_sboed/retired/hec_v3_20261114'
    provenance=json.loads((root/'SOURCE_PROVENANCE.json').read_text());assert provenance['status']=='RETIRED_NO_RESTART'
    for relative,expected in provenance['files'].items():assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected
    paths=[root/'hec_v3_counterfactual_router_20261114',root/'hec_v3_v2_pathways_20261113',work/'development/moe_sboed/hec_v3_troubleshoot_20261113']
    for path in reversed(paths):sys.path.insert(0,str(path))
    return dict(status='VERIFIED_RETIRED_REFERENCE',files=len(provenance['files']),root=str(root))
