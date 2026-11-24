"""CPU-only clone/serialization check;no model/simulator training episodes."""
from pathlib import Path
import sys,importlib.util,io,json,time
M=Path.home()/"Documents/objective_power_grid"
spec=importlib.util.spec_from_file_location("spec_clone_inspection",M/"src/research/moe_sboed_v4_spec_20261118/run.py")
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
import torch,numpy as np
start=time.monotonic();torch.set_num_threads(2)
state,a,digest=r.build(r.read(r.R/"REGIMES.json")["thresholds"],"cpu")
x=torch.from_numpy(np.load(r.O/"calibration/training_only_prefixes.npz")["X"][[0,160,448,928]])
a.actor_optimizer.zero_grad();loss=-a.critic_net(torch.cat([x,a.actor_net(x)],-1)).mean();loss.backward();a.actor_optimizer.step()
folder=r.O/"audit/cpu_clone";folder.mkdir(exist_ok=False)
identity=r.save_runtime(folder/"runtime.pkl",a)
old=torch.storage._load_from_bytes;torch.storage._load_from_bytes=lambda data:torch.load(io.BytesIO(data),map_location="cpu",weights_only=False)
try:
 with (folder/"runtime.pkl").open("rb") as f:clone=r.p.n.cloudpickle.load(f)
finally:torch.storage._load_from_bytes=old
assert r.identity(clone["algorithm"],clone["rng"])==identity
parameters={id(p) for p in clone["algorithm"].actor_net.parameters()}
assert all(id(p) in parameters for group in clone["algorithm"].actor_optimizer.param_groups for p in group["params"])
r.put(r.R/"CLONE_SERIALIZATION_AUDIT.json",dict(status="PASS",seconds=time.monotonic()-start,full_algorithm_optimizer_replay_rng_roundtrip=True,optimizer_parameter_aliases_preserved=True,device="cpu",new_simulator_episodes=0,discarded_historical_prefix_gradient_updates=1,scope="CPU serialization/identity audit;actual postwarm GPU clone identities also checked by each arm at launch"))
print(json.dumps(r.read(r.R/"CLONE_SERIALIZATION_AUDIT.json")))
