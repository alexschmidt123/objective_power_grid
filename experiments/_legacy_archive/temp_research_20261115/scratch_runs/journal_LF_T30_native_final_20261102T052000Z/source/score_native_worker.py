"""Frozen native ScoreBED execution; selected-restart preservation only."""
import pathlib,sys
R=pathlib.Path.home()/'Documents/objective_power_grid_temp';sys.path.insert(0,str(R))
from src.research.compute_budget import ComputeClock
_can=ComputeClock.can_start
def soft_can_start(self,next_unit_bound=0.):
 return _can(self,next_unit_bound) and self.spent()+next_unit_bound+120<37800
ComputeClock.can_start=soft_can_start
path=R/'src/research/scorebed_worker.py';source=path.read_text()
needle='        params=restored_params(params,args.journal_plan)'
assert source.count(needle)==1
source=source.replace(needle,'        if restarts==1:params=restored_params(params,args.journal_plan)')
exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
