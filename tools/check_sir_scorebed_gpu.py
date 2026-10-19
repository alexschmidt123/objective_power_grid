"""GPU-native ScoreBED SIR adapter versus the independent saved numerical oracle."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/research/vendor/scorebed'))
sys.path.insert(0,str(ROOT))
import numpy as np
import jax,jax.numpy as jnp
from src.research.sir_scorebed_model import SIRScoreModel

assert jax.devices()[0].platform=='gpu'
a=np.load(ROOT/'tools/tests/fixtures/sir_scorebed_ode_golden.npz')
m=SIRScoreModel(T=5)
x=jnp.asarray(a['actions']);th=jnp.asarray(a['theta'])
th=jnp.broadcast_to(th[:,None,:],(len(th),5,2))
start=time.perf_counter()
y=jax.jit(m.means)(x,th).block_until_ready()
g=jax.jit(jax.grad(lambda z:m.means(z,th).sum()))(x).block_until_ready()
ref=a['time_jacobian'].sum((0,1,2))
np.testing.assert_allclose(y,a['means'],rtol=1e-10,atol=1e-9)
np.testing.assert_allclose(g,ref,rtol=2e-6,atol=1e-7)
out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'experiments/10162026/10162026_ubuntu_sir_execution/jax_gpu_equivalence.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(status='passed',devices=[str(d) for d in jax.devices()],
    mean_max_absolute_error=float(np.max(abs(np.asarray(y)-a['means']))),
    gradient_max_absolute_error=float(np.max(abs(np.asarray(g)-ref))),
    cold_seconds=time.perf_counter()-start),indent=2))
print(out.read_text(),flush=True)
