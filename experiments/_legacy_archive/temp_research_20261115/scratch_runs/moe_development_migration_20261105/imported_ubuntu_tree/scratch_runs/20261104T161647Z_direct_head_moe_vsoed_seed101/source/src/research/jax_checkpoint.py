"""Lossless native Flax state serialization, including typed JAX PRNG keys."""
from pathlib import Path

def save_state(path,state):
    import jax,numpy as np
    from flax import serialization
    def encode(x):
        if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
        if isinstance(x,list):return [encode(v) for v in x]
        if isinstance(x,tuple):return tuple(encode(v) for v in x)
        if hasattr(x,"dtype") and jax.dtypes.issubdtype(x.dtype,jax.dtypes.prng_key):
            return {"__boed_typed_key__":True,"data":np.asarray(jax.random.key_data(x)),"implementation":str(jax.random.key_impl(x))}
        return x
    p=Path(path);tmp=p.with_suffix(p.suffix+".partial")
    tmp.write_bytes(serialization.msgpack_serialize(encode(serialization.to_state_dict(state))))
    tmp.replace(p)
def load_state(path,template):
    import jax,jax.numpy as jnp
    from flax import serialization
    def decode(x):
        if isinstance(x,dict) and x.get("__boed_typed_key__",False):return jax.random.wrap_key_data(jnp.asarray(x["data"]),impl=x["implementation"])
        if isinstance(x,dict):return {k:decode(v) for k,v in x.items()}
        if isinstance(x,list):return [decode(v) for v in x]
        if isinstance(x,tuple):return tuple(decode(v) for v in x)
        return x
    return serialization.from_state_dict(template,decode(serialization.msgpack_restore(Path(path).read_bytes())))
