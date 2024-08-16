import jax
#import jax.numpy as jnp
from optax._src import base
#from optax._src import numerics
#from optax._src.transform import AddNoiseState
from typing import NamedTuple
#import chex

class AddNoiseStateC(NamedTuple):
  """State for adding gradient noise. It keeps the Key to apply the noise and create a new key"""
  rng_key: jax.random.PRNGKey

def add_noise(
    noise_std,
    expected_bs:int,
    seed: int
) -> base.GradientTransformation:
  """Add gradient noise.

  Args:
    noise_std: Noise deviation.
    seed: Seed for random number generation.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    print("Initializing add_noise transformation", flush=True)
    return AddNoiseStateC(rng_key=jax.random.PRNGKey(seed))

  def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
    del params
    print('inside update function:',noise_std,'expected_bs',expected_bs,'seed',seed,'PRNG key',state.rng_key,flush=True)
    num_vars = len(jax.tree_util.tree_leaves(updates))
    treedef = jax.tree_util.tree_structure(updates)
    new_key,*all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
    noise = jax.tree_util.tree_map(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_util.tree_unflatten(treedef, all_keys))
    updates = jax.tree_util.tree_map(
        lambda g, n: (g + noise_std * n)/expected_bs,
        updates, noise)
    
    print(f'Noise added. New key: {new_key}', flush=True)
    
    return updates, AddNoiseStateC(rng_key=new_key)

  return base.GradientTransformation(init_fn, update_fn)

def foo(noise_std,expected_bs,rng_key,updates):
    #del params
    print('inside update function:',noise_std,'expected_bs',expected_bs,'PRNG key',rng_key,flush=True)
    num_vars = len(jax.tree_util.tree_leaves(updates))
    treedef = jax.tree_util.tree_structure(updates)
    new_key,*all_keys = jax.random.split(rng_key, num=num_vars + 1)
    noise = jax.tree_util.tree_map(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_util.tree_unflatten(treedef, all_keys))
    updates = jax.tree_util.tree_map(
        lambda g, n: (g + noise_std * n)/expected_bs,
        updates, noise)
    
    print(f'Noise added. New key: {new_key}', flush=True)
    
    return updates, new_key,noise

import numpy as np

def main():
  
  new_key = jax.random.PRNGKey(123)
  noises = []
  upds = {'params':jax.numpy.zeros(6)}
  print('original key',new_key)
  for i in range(10):
    new_updates,new_key,noise = foo(0.5,5,new_key,upds)
    noises.extend(new_updates['params'])
    print(upds,new_key,noise)
  mu = np.mean(noises)
  std = np.std(noises)
  
  print(mu,std)
#main() 