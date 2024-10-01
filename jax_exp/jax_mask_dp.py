import os
import tensorflow as tf


tf.config.experimental.set_visible_devices([], 'GPU')

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )

import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import torch
from typing import List
from torch.utils.data import Sampler, Dataset
from scipy.stats import binom
import torchvision
import math
import flax.linen as nn
from transformers import FlaxViTModel,FlaxViTForImageClassification
from private_vit import ViTModelHead
#from jax.config import config
import warnings
#config.update("jax_debug_nans", True)
#config.update("jax_debug_infs", True)

import time

from dp_accounting import dp_event,rdp
from opacus.accountants.utils import get_noise_multiplier
from flax.core.frozen_dict import unfreeze,freeze,FrozenDict

from jax.profiler import start_trace, stop_trace
from functools import partial
from jax._src.lib import xla_client

PHYSICAL_BATCH = 32

@jax.jit
def add_trees(x, y):
    #Helper function, add two tree objects
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)

@jax.jit
def noise_addition(rng_key, accumulated_clipped_grads, noise_std, C):
    """
        Add noise to the accumulated gradients.
        Keyword arguments:
        rng_key: PRNG jax key
        accumulated_clipped_grads: Tree structure, accumulated gradients
        noise_std:
        C:     
    """
    num_vars = len(jax.tree_util.tree_leaves(accumulated_clipped_grads))
    treedef = jax.tree_util.tree_structure(accumulated_clipped_grads)
    new_key, *all_keys = jax.random.split(rng_key, num=num_vars + 1)
    noise = jax.tree_util.tree_map(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        accumulated_clipped_grads, jax.tree_util.tree_unflatten(treedef, all_keys))
    updates = jax.tree_util.tree_map(
        lambda g, n: g + noise_std * C * n,
        accumulated_clipped_grads, noise)
    return updates

@jax.jit
def compute_per_example_gradients(state, batch_X, batch_y):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params, X, y):
        #logits = state.apply_fn({'params': params}, X)
        logits = state.apply_fn(X,params=params)[0]
        one_hot = jax.nn.one_hot(y, 100)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        #assert len(loss) == 1
        return loss.sum()
    
    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    px_grads = jax.vmap(grad_fn, in_axes=(0, 0))(batch_X, batch_y)
  
    return px_grads

def compute_per_example_gradients_wojit(state, batch_X, batch_y):
    """Computes per example gradients, loss and accuracy for a single batch. This function is not compiled"""

    def loss_fn(params, X, y):
        logits = state.apply_fn({'params': params}, X)
        one_hot = jax.nn.one_hot(y, 100)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        #assert len(loss) == 1
        return loss.sum()
    
    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    px_grads = jax.vmap(grad_fn, in_axes=(0, 0))(batch_X, batch_y)
  
    return px_grads

@jax.jit
def compute_gradients_non_private(state, batch_X, batch_y, mask):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params, X, y):
        logits = state.apply_fn( X,params=params)[0]
        #{'params': params},
        one_hot = jax.nn.one_hot(y, 100)
        loss = jnp.sum(mask * optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss
    
    summed_grad = jax.grad(loss_fn)(state.params, batch_X, batch_y)
  
    return summed_grad

@jax.jit
def update_model(state, grads):
    """Updates the state with the gradients, using the optimizer inside the state"""
    return state.apply_gradients(grads=grads)


def eval(state, batch_X, batch_y):
    """Computes gradients, loss and accuracy for a single batch."""

    logits = state.apply_fn({'params': state.params}, batch_X)
    one_hot = jax.nn.one_hot(batch_y, 100)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
    predicted_class = jnp.argmax(logits,axis=-1)
    
    acc = jnp.mean(predicted_class == batch_y)

    return acc

def eval_model(data_loader,state):
    # Test model on all images of a data loader and return avg loss
    accs = []
    for batch in data_loader:
        batch_X,batch_y = jnp.array(batch[0]),jnp.array(batch[1])
        acc = eval(state,batch_X,batch_y)
        accs.append(acc)
        del batch
    eval_acc = jnp.mean(jnp.array(accs))
    
    return eval_acc

def create_train_state(rng, lr,model,params):
    """Creates initial `TrainState`."""
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=tx)
    #return train_state.TrainState.create(apply_fn=jax.jit(model.apply), params=params, tx=tx)

@jax.jit
def process_a_physical_batch(px_grads, mask, C):
    """Compute the norms of the gradients, then clips them and mask the ones that we don't require"""
    def clip_mask_and_sum(x, mask, clipping_multiplier):

        new_shape = (-1,) + (1,) * (x.ndim - 1)
        mask = mask.reshape(new_shape)
        clipping_multiplier = clipping_multiplier.reshape(new_shape)

        return jnp.sum(x * mask * clipping_multiplier, axis=0)

    px_per_param_sq_norms = jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(x.shape[0], -1), axis=-1)**2, px_grads)
    flattened_px_per_param_sq_norms, tree_def = jax.tree_util.tree_flatten(px_per_param_sq_norms)

    px_grad_norms = jnp.sqrt(jnp.sum(jnp.array(flattened_px_per_param_sq_norms), axis=0))

    clipping_multiplier = jnp.minimum(1., C/px_grad_norms)

    return jax.tree_map(lambda x: clip_mask_and_sum(x, mask, clipping_multiplier), px_grads)

def private_iteration(logical_batch, state, k, q, t, noise_std, C, full_data_size,cpus,gpus):
    """Naive iteration of DPSGD, with a normal python loop"""
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x, y = logical_batch

    logical_batch_size = len(x)
    physical_batches = np.array(np.split(x, k)) # k x pbs x dim
    physical_labels = np.array(np.split(y, k))
    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])

    masks = jnp.array(jnp.split(masks, k))

    ### gradient accumulation
    accumulated_clipped_grads = jax.tree_map(lambda x: 0. * x, params)
    start_time = time.perf_counter()
    for pb, yb, mask in zip(physical_batches, physical_labels, masks):
        pb =prepare_data(gpus,pb)
        yb =prepare_data(gpus,yb)
        per_example_gradients = jax.block_until_ready(compute_per_example_gradients(state, pb, yb))
        sum_of_clipped_grads_from_pb = jax.block_until_ready(process_a_physical_batch(per_example_gradients, mask, C))
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads,sum_of_clipped_grads_from_pb)
        # accumulated_clipped_grads = jax.tree_map(lambda x,y: x+y, 
        #                                         accumulated_clipped_grads, 
        #                                         sum_of_clipped_grads_from_pb
        #                                         )

    noisy_grad = noise_addition(jax.random.PRNGKey(t), accumulated_clipped_grads, noise_std, C)

    ### update
    new_state = jax.block_until_ready(update_model(state, noisy_grad))
    batch_time = time.perf_counter() - start_time
    return new_state, actual_batch_size,logical_batch_size,batch_time


def private_iteration_wojit(logical_batch, state, k, q, t, noise_std, C, full_data_size,cpus,gpus):
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x, y = logical_batch

    logical_batch_size = len(x)
    physical_batches = np.array(np.split(x, k)) # k x pbs x dim
    physical_labels = np.array(np.split(y, k))
    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])

    masks = jnp.array(jnp.split(masks, k))

    ### gradient accumulation
    accumulated_clipped_grads = jax.tree_map(lambda x: 0. * x, params)
    start_time = time.perf_counter()
    for pb, yb, mask in zip(physical_batches, physical_labels, masks):
        pb =prepare_data(gpus,pb)
        yb =prepare_data(gpus,yb)
        per_example_gradients = jax.block_until_ready(compute_per_example_gradients_wojit(state, pb, yb))
        sum_of_clipped_grads_from_pb = jax.block_until_ready(process_a_physical_batch(per_example_gradients, mask, C))
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads,sum_of_clipped_grads_from_pb)

        # accumulated_clipped_grads = jax.tree_map(lambda x,y: x+y, 
        #                                         accumulated_clipped_grads, 
        #                                         sum_of_clipped_grads_from_pb
        #                                         )

    noisy_grad = noise_addition(jax.random.PRNGKey(t), accumulated_clipped_grads, noise_std, C)

    ### update
    new_state = jax.block_until_ready(update_model(state, noisy_grad))
    batch_time = time.perf_counter() - start_time
    return new_state, actual_batch_size,logical_batch_size,batch_time


def private_iteration_state(logical_batch, state, k, q, t, noise_std, C, full_data_size,cpus,gpus):
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x, y = logical_batch

    logical_batch_size = len(x)
    physical_batches = np.array(np.split(x, k)) # k x pbs x dim
    physical_labels = np.array(np.split(y, k))
    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])

    masks = jnp.array(jnp.split(masks, k))

    @jax.jit
    def compute_per_example_gradients(batch_X, batch_y):
        """Computes gradients, loss and accuracy for a single batch."""

        def loss_fn(params, X, y):
            logits = state.apply_fn({'params': params}, X)
            one_hot = jax.nn.one_hot(y, 100)
            loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
            #assert len(loss) == 1
            return loss.sum()
        
        grad_fn = lambda X, y: jax.grad(loss_fn)(params, X, y)
        px_grads = jax.vmap(grad_fn, in_axes=(0, 0))(batch_X, batch_y)
    
        return px_grads


    ### gradient accumulation
    accumulated_clipped_grads = jax.tree_map(lambda x: 0. * x, params)
    start_time = time.perf_counter()
    for pb, yb, mask in zip(physical_batches, physical_labels, masks):
        pb =prepare_data(gpus,pb)
        yb =prepare_data(gpus,yb)
        per_example_gradients = jax.block_until_ready(compute_per_example_gradients(pb, yb))
        sum_of_clipped_grads_from_pb = jax.block_until_ready(process_a_physical_batch(per_example_gradients, mask, C))
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads,sum_of_clipped_grads_from_pb)

    noisy_grad = noise_addition(jax.random.PRNGKey(t), accumulated_clipped_grads, noise_std, C)

    ### update
    new_state = jax.block_until_ready(update_model(state, noisy_grad))
    batch_time = time.perf_counter() - start_time
    return new_state, actual_batch_size,logical_batch_size,batch_time

@jax.jit
def body_fun_p(t, args):
    state, accumulated_grads, logical_batch_x,logical_batch_y,masks,C = args
    start_idx = t * PHYSICAL_BATCH
    x_slice = jax.lax.dynamic_slice(logical_batch_x, (start_idx,0,0,0), (PHYSICAL_BATCH,3,224,224))
    y_slice = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (PHYSICAL_BATCH,))
    masks_slice = jax.lax.dynamic_slice(masks, (start_idx,), (PHYSICAL_BATCH,))

    per_example_gradients = compute_per_example_gradients(state, x_slice,y_slice)
    sum_of_clipped_grads_from_pb = process_a_physical_batch(per_example_gradients,masks_slice, C)
    accumulated_clipped_grads = add_trees(accumulated_clipped_grads,sum_of_clipped_grads_from_pb)

    return state, accumulated_grads, logical_batch_x,logical_batch_y,masks,C


@jax.jit
def body_fun_non_p(t, args):
    state, accumulated_grads, logical_batch_x,logical_batch_y,masks = args
    #state, accumulated_grads, logical_batch_x,logical_batch_y,masks,physical_bs = args
    start_idx = t * PHYSICAL_BATCH
    x_slice = jax.lax.dynamic_slice(logical_batch_x, (start_idx,0,0,0), (PHYSICAL_BATCH,3,224,224))
    y_slice = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (PHYSICAL_BATCH,))
    masks_slice = jax.lax.dynamic_slice(masks, (start_idx,), (PHYSICAL_BATCH,))

    summed_grads_from_pb = compute_gradients_non_private(state, x_slice,y_slice, masks_slice)

    accumulated_grads = add_trees(accumulated_grads,summed_grads_from_pb)
    
    # accumulated_grads = jax.tree_map(lambda x,y: x+y, 
    #                                         accumulated_grads, 
    #                                         summed_grads_from_pb
    #                                         )
    return state, accumulated_grads, logical_batch_x,logical_batch_y,masks



def private_iteration_fori_loop(logical_batch,physical_bs, state, k, q, t, noise_std, C, full_data_size,cpus,gpus):
    """Optimized DPSGD iteration, with static sizes that complies with the poisson subsampling. It uses JAX fori_loop"""
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    #x, y = logical_batch
    x,y= prepare_data(gpus,logical_batch[0]),prepare_data(gpus,logical_batch[1])

    logical_batch_size = len(x) # k x pbs x dim

    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])
    physical_bs = int(physical_bs)
    ### gradient accumulation
    # def body_fun(t, accumulated_clipped_grads):
    #     start_idx = t * physical_bs
    #     x_slice = jax.lax.dynamic_slice(x, (start_idx,0,0,0,0), (physical_bs,1,3,224,224))
    #     y_slice = jax.lax.dynamic_slice(y, (start_idx,), (physical_bs,))
    #     masks_slice = jax.lax.dynamic_slice(masks, (start_idx,), (physical_bs,))

    #     per_example_gradients = compute_per_example_gradients(state, x_slice,y_slice)
    #     sum_of_clipped_grads_from_pb = process_a_physical_batch(per_example_gradients,masks_slice, C)
    #     accumulated_clipped_grads = add_trees(accumulated_clipped_grads,sum_of_clipped_grads_from_pb)
    #     # accumulated_clipped_grads = jax.tree_map(lambda x,y: x+y, 
    #     #                                         accumulated_clipped_grads, 
    #     #                                         sum_of_clipped_grads_from_pb
    #     #                                         )
    #     return accumulated_clipped_grads

    
    accumulated_clipped_grads0 = jax.tree_map(lambda x: 0. * x, params)

    start_time = time.perf_counter()

    _, accumulated_clipped_grads, *_  = jax.block_until_ready(jax.lax.fori_loop(0, k, body_fun_p, (state, accumulated_clipped_grads0, x,y,masks,jax.lax.convert_element_type(physical_bs, jnp.int32),jax.lax.convert_element_type(C, jnp.float32))))

    noisy_grad = noise_addition(jax.random.PRNGKey(t), accumulated_clipped_grads, noise_std, C)

    ### update
    new_state = update_model(state, noisy_grad)
    batch_time = time.perf_counter() - start_time
    return new_state, actual_batch_size,logical_batch_size,batch_time

def non_private_iteration_fori_loop(logical_batch,physical_bs, state, k, q, t, full_data_size,cpus,gpus):
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x,y= prepare_data(gpus,logical_batch[0]),prepare_data(gpus,logical_batch[1])
    logical_batch_size = len(x)
    #physical_batches = np.array(np.split(x, k)) # k x pbs x dim

    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])
    #physical_bs = logical_batch_size/k
    ### gradient accumulation
    # def body_fun(t, accumulated_grads):

    #     start_idx = t * physical_bs
    #     x_slice = jax.lax.dynamic_slice(x, (start_idx,0,0,0), (physical_bs,3,224,224))
    #     y_slice = jax.lax.dynamic_slice(y, (start_idx,), (physical_bs,))
    #     masks_slice = jax.lax.dynamic_slice(masks, (start_idx,), (physical_bs,))

    #     summed_grads_from_pb = compute_gradients_non_private(state, x_slice,y_slice, masks_slice)

    #     accumulated_grads = add_trees(accumulated_grads,summed_grads_from_pb)
        
    #     # accumulated_grads = jax.tree_map(lambda x,y: x+y, 
    #     #                                         accumulated_grads, 
    #     #                                         summed_grads_from_pb
    #     #                                         )
    #     return accumulated_grads
    
    accumulated_grads0 = jax.tree_map(lambda x: jnp.zeros_like(x), params)

    start_time = time.perf_counter()
    _, accumulated_grads, *_ = jax.block_until_ready(jax.lax.fori_loop(0, k,body_fun_non_p,(state, accumulated_grads0, x,y,masks)))

    ### update
    new_state = update_model(state, accumulated_grads)
    batch_time = time.perf_counter() - start_time
    return new_state, logical_batch_size,batch_time


def non_private_iteration(logical_batch, state, k, q, t, full_data_size,cpus,gpus):
    params = state.params
    
    sampling_rng = jax.random.PRNGKey(t + 1)
    batch_rng, binomial_rng = jax.random.split(sampling_rng, 2) 

    x, y = logical_batch

    logical_batch_size = len(x)
    physical_batches = np.array(np.split(x, k)) # k x pbs x dim
    physical_labels = np.array(np.split(y, k))
    
    # poisson subsample
    actual_batch_size = jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum()    
    n_masked_elements = logical_batch_size - actual_batch_size
    masks = jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)])
    masks = jnp.array(jnp.split(masks, k))

    ### gradient accumulation
    accumulated_grads = jax.tree_map(lambda x: 0. * x, params)
    start_time = time.perf_counter()
    for pb, yb, mask in zip(physical_batches, physical_labels, masks):
        pb =prepare_data(gpus,pb)
        yb =prepare_data(gpus,yb)
        summed_grads_from_pb = jax.block_until_ready(compute_gradients_non_private(state, pb, yb, mask))

        accumulated_grads = add_trees(accumulated_grads,summed_grads_from_pb)
        
        # accumulated_grads = jax.tree_map(lambda x,y: x+y, 
        #                                         accumulated_grads, 
        #                                         summed_grads_from_pb
        #                                         )

    ### update
    new_state = jax.block_until_ready(update_model(state, accumulated_grads))
    batch_time = time.perf_counter() - start_time
    return new_state, logical_batch_size,batch_time

class FixedBatchsizeSampler(Sampler[List[int]]):

    def __init__(
        self, *, num_samples_total: int, batch_size: int, steps: int, generator=None
    ):

        self.num_samples_total = num_samples_total
        self.batch_size = batch_size
        self.steps = steps
        
        self.generator = generator

        if self.num_samples_total <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __len__(self):
        return self.steps

    def __iter__(self):
        num_batches = self.steps
        while num_batches > 0:
            indices = torch.randperm(self.num_samples_total, generator=self.generator)[:self.batch_size]
            yield indices

            num_batches -= 1
    
def image_to_numpy_wo_t(img):
    """Transformation of the image. It normalizes and transposes it"""
    img = np.array(img, dtype=np.float32)
    img = ((img / 255.) -  np.array([0.5, 0.5, 0.5])) /  np.array([0.5, 0.5, 0.5])
    img = np.transpose(img,[2,0,1])
    return img
    
def load_dataset(dimension):
    """Load the dataset, not yet the dataloader"""
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(dimension),
        image_to_numpy_wo_t,
    ])
    trainset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=True, download=True, transform=transformation)
    testset = torchvision.datasets.CIFAR100(root='../data_cifar100/', train=False, download=True, transform=transformation)
    
    return trainset,testset

def numpy_collate(batch):
    """Collate the batch, we don't want a tensor but a numpy array"""
    if isinstance(batch[0],np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0],(tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
def load_model(rng,model_name,dimension,num_classes):
    """
        Load the model

        model_name: str. It can be a small test CNN architecture or a ViT transformer
    """
    print('load model name',model_name,flush=True)
    main_key, params_key= jax.random.split(key=rng,num=2)
    if model_name == 'small':
        class CNN(nn.Module):
            """A simple CNN model."""

            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=64, kernel_size=(7, 7),strides=2)(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                #x = nn.Conv(features=64, kernel_size=(3, 3))(x)
                #x = nn.relu(x)
                #x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = x.reshape((x.shape[0], -1))  # flatten
                x = nn.Dense(features=256)(x)
                x = nn.relu(x)
                x = nn.Dense(features=100)(x)
                return x

        model = CNN()
        input_shape = (1,3,dimension,dimension)
        #But then, we need to split it in order to get random numbers
        

        #The init function needs an example of the correct dimensions, to infer the dimensions.
        #They are not explicitly writen in the module, instead, the model infer them with the first example.
        x = jax.random.normal(params_key, input_shape)

        main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
        #Initialize the model
        #variables = model.init({'params':init_rng},x)
        variables = jax.jit(model.init)({'params':init_rng},x)
        #variables = model.init({'params':main_key}, batch)
        model.apply(variables, x)
        model = model
        params = variables['params']
    
    elif 'vit' in model_name:
        model_name = model_name
        model = FlaxViTForImageClassification.from_pretrained(model_name, num_labels=num_classes, return_dict=False, ignore_mismatched_sizes=True)
        #model = FlaxViTForImageClassification.from_pretrained(model_name)
        # model = FlaxViTModel.from_pretrained(model_name,add_pooling_layer=False)
        # module = model.module # Extract the Flax Module
        # vars = {'params': model.params} # Extract the parameters
        # #config = module.config
        # model = ViTModelHead(num_classes=num_classes,pretrained_model=model)

        # input_shape = (1,3,dimension,dimension)
        # #But then, we need to split it in order to get random numbers
        

        # #The init function needs an example of the correct dimensions, to infer the dimensions.
        # #They are not explicitly writen in the module, instead, the model infer them with the first example.
        # x = jax.random.normal(params_key, input_shape)

        # main_rng, init_rng, dropout_init_rng = jax.random.split(main_key, 3)
        # #Initialize the model
        # variables = jax.jit(model.init)({'params':init_rng},x)
        # #variables = model.init({'params':init_rng},x)

        # #So far, the parameters are initialized randomly, so we need to unfreeze them and add the pre loaded parameters.
        # params = variables['params']
        # params['vit'] = vars['params']
        #params = unfreeze(params)
        #print_param_shapes(params)
        #print(params)
        #model.apply({'params':params},x)
        #model = model
        params = model.params
    return main_key,model,params

def compute_epsilon(steps,batch_size, num_examples=60000, target_delta=1e-5,noise_multiplier=0.1):
    """Compute epsilon for DPSGD privacy accounting"""
    if num_examples * target_delta > 1.:
        warnings.warn('Your delta might be too high.')

    print('steps',steps,flush=True)

    print('noise multiplier',noise_multiplier,flush=True)

    q = batch_size / float(num_examples)
    orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders) # type: ignore
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            q, dp_event.GaussianDpEvent(noise_multiplier)), steps)
    
    epsilon = accountant.get_epsilon(target_delta)
    delta = accountant.get_delta(epsilon)

    return epsilon,delta

def calculate_noise(sample_rate,target_epsilon,target_delta,epochs,accountant):
    """Calculate the noise multiplier with Opacus implementation"""
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant=accountant
    )

    return noise_multiplier

#@jax.jit
def prepare_data(device,batch):
    """Move batch between devices"""
    return jax.device_put(batch,device[0])

def print_device(x):
    print(f"Device: {x.device()}")

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def main(args):
    print(args,flush=True)

    print('devices ',jax.devices(),flush=True)
    rng = jax.random.PRNGKey(0)

    trainset,testset = load_dataset(args.dimension)

    print('dataset loaded',flush=True)

    rng,model,params = load_model(rng,args.model,args.dimension,args.ten)
    
    print('model loaded')
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, args.lr,model,params)
    q = q = 1/math.ceil(len(trainset)/args.bs)
    n = len(trainset)
    physical_bs = args.phy_bs
    global PHYSICAL_BATCH 
    PHYSICAL_BATCH = physical_bs

    print(PHYSICAL_BATCH)
    alpha = 1e-9 # failure prob.

    from scipy.stats import binom
    k = 1
    binom_dist = binom(n, q)
    while True:
        right_prob = binom_dist.sf(k * physical_bs)
        if right_prob < alpha:
            break
        k += 1
        
    max_logical_batch_size = k*physical_bs

    steps = args.epochs * math.ceil(len(trainset)/args.bs)

    print('n',n,'q',q,'k',k,'max logical batch size',max_logical_batch_size,'steps',steps,flush=True)

    fbs = FixedBatchsizeSampler(num_samples_total=n, batch_size=max_logical_batch_size, steps=steps)

    #dataset = CustomImageDataset(train_images, train_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=fbs,collate_fn=numpy_collate,num_workers=args.n_workers,worker_init_fn=seed_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=80, shuffle=False,collate_fn=numpy_collate,num_workers=args.n_workers,worker_init_fn=seed_worker)

    clipping_mode = args.clipping_mode

    epochs = args.epochs
    iters_per_epoch = math.ceil(len(trainset)/args.bs)
    t = 0
    #for e in range(epochs): #The sampler already does the n iterations

    noise_multiplier = calculate_noise(q,args.epsilon,args.target_delta,epochs,'rdp')

    samples = 0
    epoch_time = 0
    throughtputs_b = []
    throughtputs_e = []
    throughtputs_t = []
    e = 0
    time_epoch = time.perf_counter()

    print('start training',flush=True)

    cpus = jax.devices("cpu")
    gpus = jax.devices("gpu")

    for batch_X, batch_y in trainloader:
        print('start iteration',t,flush=True)
        batch_X,batch_y = prepare_data(cpus,batch_X),prepare_data(cpus,batch_y)
        #start_trace(f'./tmp/jax-trace/{clipping_mode}',create_perfetto_trace=True)

        if clipping_mode == 'non-private':
            #batch_X = jnp.array(batch_X)
            #jax.profiler.save_device_memory_profile(f"./tmp/memory{t}.prof")
            state, logical_batch_size,batch_time = non_private_iteration((batch_X, batch_y), state, k, q, t, n,cpus,gpus)
        elif clipping_mode == 'private':
            batch_X = np.array(batch_X).reshape(-1, 1,3, args.dimension, args.dimension)
            #jax.profiler.save_device_memory_profile(f"./tmp/memory{t}.prof")
            state, actual_batch_size,logical_batch_size,batch_time = private_iteration((batch_X, batch_y), state, k, q, t, noise_multiplier, args.grad_norm, n,cpus,gpus)
            epsilon,delta = compute_epsilon(steps=t+1,batch_size=actual_batch_size,num_examples=len(trainset),target_delta=args.target_delta,noise_multiplier=noise_multiplier)
            privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
            print(privacy_results,flush=True)
        elif clipping_mode == 'private-unjit':
            batch_X = np.array(batch_X).reshape(-1, 1,3, args.dimension, args.dimension)
            #jax.profiler.save_device_memory_profile(f"./tmp/memory{t}.prof")
            state, actual_batch_size,logical_batch_size,batch_time = private_iteration_wojit((batch_X, batch_y), state, k, q, t, noise_multiplier, args.grad_norm, n,cpus,gpus)
            epsilon,delta = compute_epsilon(steps=t+1,batch_size=actual_batch_size,num_examples=len(trainset),target_delta=args.target_delta,noise_multiplier=noise_multiplier)
            privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
            print(privacy_results,flush=True)
        elif clipping_mode == 'private-state':
            batch_X = np.array(batch_X).reshape(-1, 1,3, args.dimension, args.dimension)
            #jax.profiler.save_device_memory_profile(f"./tmp/memory{t}.prof")
            state, actual_batch_size,logical_batch_size,batch_time = private_iteration_state((batch_X, batch_y), state, k, q, t, noise_multiplier, args.grad_norm, n,cpus,gpus)
            epsilon,delta = compute_epsilon(steps=t+1,batch_size=actual_batch_size,num_examples=len(trainset),target_delta=args.target_delta,noise_multiplier=noise_multiplier)
            privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
            print(privacy_results,flush=True)
        elif clipping_mode == 'non-private-fori':
            #batch_X = jnp.array(batch_X)
            state, logical_batch_size,batch_time = non_private_iteration_fori_loop((batch_X, batch_y),int(physical_bs), state, k, q, t, n,cpus,gpus)
        elif clipping_mode == 'private-fori':
            batch_X = np.array(batch_X).reshape(-1, 1,3, args.dimension, args.dimension)
            state, actual_batch_size,logical_batch_size,batch_time = private_iteration_fori_loop((batch_X, batch_y),int(physical_bs), state, k, q, t, noise_multiplier, args.grad_norm, n,cpus,gpus)
            epsilon,delta = compute_epsilon(steps=t+1,batch_size=actual_batch_size,num_examples=len(trainset),target_delta=args.target_delta,noise_multiplier=noise_multiplier)
            privacy_results = {'eps_rdp':epsilon,'delta_rdp':delta}
            print(privacy_results,flush=True)
        #stop_trace()
        del batch_X,batch_y
        xla_client._xla.collect_garbage()
        acc = eval_model(testloader,state)
        t = t+1
        throughtputs_b.append(logical_batch_size/batch_time)
        samples += logical_batch_size
        epoch_time += batch_time
        if t % iters_per_epoch == 0:
            print('finish epoch',e,'t',t)
            time_epoch_total = time.perf_counter() - time_epoch
            throughtputs_e.append(samples/epoch_time)
            throughtputs_t.append(samples/time_epoch_total)
            samples = 0
            epoch_time = 0
            e += 1 
            time_epoch = time.perf_counter()
        print('after iteration',t,'acc eval',acc,flush=True)

    print('per batch throughtput',throughtputs_b)
    print(throughtputs_e)
    return np.mean(throughtputs_e),acc
    #eval(state,)

#main(dict({'dimension':224,'epochs':2,'clipping_mode':'private','num_classes':100,'model_name':'google/vit-base-patch16-224','lr':0.00031,'bs':25000,'pbs':50}))

    # _, barely_clipped_grad, actual_batch_size = private_iteration_v2((batch_X, batch_y), state, k, q, t, 0.0, 1000.0, n)
    # _, just_clipped_grad, actual_batch_size = private_iteration_v2((batch_X, batch_y), state, k, q, t, 0.0, 1.0, n)
    # _, noisy_grad, actual_batch_size = private_iteration_v2((batch_X, batch_y), state, k, q, t, 10.0, 1.0, n)