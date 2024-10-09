
import jax, optax, itertools, collections, flax


import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.training import train_state
from flax.jax_utils import prefetch_to_device
from jax.profiler import start_trace, stop_trace
from collections import namedtuple


from opacus.accountants.utils import get_noise_multiplier

import os

from transformers import FlaxViTForImageClassification
import math
import time

DATA_MEANS = np.array([0.5, 0.5, 0.5])
DATA_STD = np.array([0.5,0.5, 0.5])

## Load data to CPU

train_images = np.load("numpy_cifar100/train_images.npy")# .to_device(device=jax.devices("cpu")[0])
train_labels = np.load("numpy_cifar100/train_labels.npy")# .to_device(device=jax.devices("cpu")[0])

train_images = jax.device_put(train_images, device=jax.devices("cpu")[0])
train_labels = jax.device_put(train_labels, device=jax.devices("cpu")[0])

test_images = np.load("numpy_cifar100/test_images.npy")# .to_device(device=jax.devices("cpu")[0])
test_labels = np.load("numpy_cifar100/test_labels.npy")# .to_device(device=jax.devices("cpu")[0])

test_images = jax.device_put(test_images, device=jax.devices("cpu")[0])
test_labels = jax.device_put(test_labels, device=jax.devices("cpu")[0])


DIMENSION = 224

## define some jax utility functions

@jax.jit
def add_trees(x, y):
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)



def normalize_and_reshape(imgs):
    normalized = ((imgs/255.) - 0.5) / 0.5
    return jax.image.resize(normalized, shape=(len(normalized), 3, 224, 224), method="bilinear")

## Main functions for DP-SGD

@jax.jit
def compute_per_example_gradients(state, batch_X, batch_y):
    """Computes gradients, loss and accuracy for a single batch."""

    resizer = lambda x: normalize_and_reshape(x)
    
    def loss_fn(params, X, y):
        resized_X = resizer(X)
        print(resized_X.shape,flush=True)
        logits = state.apply_fn(resized_X, params=params)[0]
        one_hot = jax.nn.one_hot(y, 100)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        assert len(loss) == 1
        return loss.sum()
    
    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    px_grads = jax.vmap(grad_fn, in_axes=(0, 0))(batch_X, batch_y)
  
    return px_grads

@jax.jit
def process_a_physical_batch(px_grads, mask, C):

    def clip_mask_and_sum(x, mask, clipping_multiplier):

        new_shape = (-1,) + (1,) * (x.ndim - 1)
        mask = mask.reshape(new_shape)
        clipping_multiplier = clipping_multiplier.reshape(new_shape)

        return jnp.sum(x * mask * clipping_multiplier, axis=0)

    px_per_param_sq_norms = jax.tree.map(lambda x: jnp.linalg.norm(x.reshape(x.shape[0], -1), axis=-1)**2, px_grads)
    flattened_px_per_param_sq_norms, tree_def = jax.tree_util.tree_flatten(px_per_param_sq_norms)

    px_grad_norms = jnp.sqrt(jnp.sum(jnp.array(flattened_px_per_param_sq_norms), axis=0))

    clipping_multiplier = jnp.minimum(1., C/px_grad_norms)

    return jax.tree.map(lambda x: clip_mask_and_sum(x, mask, clipping_multiplier), px_grads)

@jax.jit
def noise_addition(rng_key, accumulated_clipped_grads, noise_std, C):
    num_vars = len(jax.tree_util.tree_leaves(accumulated_clipped_grads))
    treedef = jax.tree_util.tree_structure(accumulated_clipped_grads)
    new_key, *all_keys = jax.random.split(rng_key, num=num_vars + 1)
    # draw noise
    noise = jax.tree_util.tree_map(
        lambda g, k: noise_std * C * jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        accumulated_clipped_grads, jax.tree_util.tree_unflatten(treedef, all_keys))
    
    updates = add_trees(accumulated_clipped_grads, noise)
    return updates



# ## Define a data loader with prefetch


def prepare_data(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)

def prefetch_to_device(iterator, size):
    queue = collections.deque()

    def _prefetch(xs):
        return jax.device_put(xs, jax.devices("gpu")[0])

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)


# ### Parameters for training

def create_train_state(model_name, num_labels, config):
    """Creates initial `TrainState`."""

    model = FlaxViTForImageClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=False, ignore_mismatched_sizes=True)

    # Initialize the model
    params = model.params
    
    # set the optimizer
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=jax.jit(model.__call__), params=params, tx=tx)

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

# ## NON-DP


@jax.jit
def compute_gradients_non_dp(state, batch_X, batch_y, mask):
#     """Computes gradients, loss and accuracy for a single batch."""

    resizer = lambda x: normalize_and_reshape(x)
    
    def loss_fn(params, X, y):
        resized_X = resizer(X)
        logits = state.apply_fn(resized_X, params=params)[0]
        one_hot = jax.nn.one_hot(y, 100)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        masked_loss = loss * mask
        return masked_loss.sum()

    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    sum_of_grads = grad_fn(batch_X, batch_y)

    return sum_of_grads

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


# # fori_loop

def eval_fn(state, batch_X, batch_y):
    """Computes gradients, loss and accuracy for a single batch."""

    resizer = lambda x: normalize_and_reshape(x)
    resized_X = resizer(batch_X)
    logits = state.apply_fn( resized_X,state.params)[0]
    one_hot = jax.nn.one_hot(batch_y, 100)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
    predicted_class = jnp.argmax(logits,axis=-1)
    
    acc = jnp.mean(predicted_class == batch_y)

    return acc


def model_evaluation(state,test_data,test_labels):

    accs = []

    for pb,yb in zip(test_data,test_labels):
        pb = jax.device_put(pb,jax.devices('gpu')[0])
        yb = jax.device_put(yb,jax.devices('gpu')[0])
        accs.append(eval_fn(state,pb,yb))
        
    return np.mean(np.array(accs))
    

def main(args):

    print(args,flush=True)

    steps = args.epochs * math.ceil(len(train_images)/args.bs)

    q = 1/math.ceil(len(train_images)/args.bs)

    noise_std = calculate_noise(q,args.epsilon,args.target_delta,args.epochs,args.accountant)
    C = args.grad_norm

    config = namedtuple("Config", ["momentum", "learning_rate"])
    config.momentum = 1
    config.learning_rate = args.lr
    
    state = create_train_state(
        model_name = args.model,
        num_labels = 100,
        config = config,
    )

    num_classes = args.ten
    orig_dimension = 32
    input_shape = (1, 3, DIMENSION, DIMENSION) # vit
    full_data_size = train_images.shape[0]
    physical_bs = args.phy_bs

    num_iter = steps
    dynamic_slice = True

    times = []
    logical_batch_sizes = []

    jax.clear_caches()

    splits_test = jnp.split(test_images,10)
    splits_labels = jnp.split(test_labels,10)
    private = False
    if args.clipping_mode == 'DP':
        private = True
    
    if dynamic_slice and private:
        @jax.jit
        def body_fun(t, args):
            state, accumulated_clipped_grads, logical_batch_X, logical_batch_y, masks = args
            # slice
            start_idx = t * physical_bs
            pb = jax.lax.dynamic_slice(logical_batch_X, (start_idx, 0, 0, 0, 0), (physical_bs, 1, 3, orig_dimension, orig_dimension))
            yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (physical_bs,))
            mask = jax.lax.dynamic_slice(masks, (start_idx,), (physical_bs,))

            # compute grads and clip
            per_example_gradients = compute_per_example_gradients(state, pb, yb)
            sum_of_clipped_grads_from_pb = process_a_physical_batch(per_example_gradients, mask, C)
            accumulated_clipped_grads = add_trees(accumulated_clipped_grads, sum_of_clipped_grads_from_pb)

            return state, accumulated_clipped_grads, logical_batch_X, logical_batch_y, masks
    elif dynamic_slice and not private:
        @jax.jit
        def body_fun(t, args):
            state, accumulated_clipped_grads, logical_batch_X, logical_batch_y, masks = args
            # slice
            start_idx = t * physical_bs
            pb = jax.lax.dynamic_slice(logical_batch_X, (start_idx, 0, 0, 0, 0), (physical_bs, 1, 3, orig_dimension, orig_dimension))
            yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (physical_bs,))
            mask = jax.lax.dynamic_slice(masks, (start_idx,), (physical_bs,))

            # compute grads and clip
            sum_of_grads = compute_gradients_non_dp(state,pb,yb,mask)
            accumulated_grads = add_trees(accumulated_clipped_grads, sum_of_grads)

            return state, accumulated_grads, logical_batch_X, logical_batch_y, masks

    else:
        def body_fun(t, args):
            state, accumulated_clipped_grads, logical_batch_X, logical_batch_y, masks = args
            
            # slice
            pb = logical_batch_X_split[t]
            yb = logical_batch_y_split[t]
            mask = masks[t]

            # compute grads and clip
            per_example_gradients = compute_per_example_gradients(state, pb, yb)
            sum_of_clipped_grads_from_pb = process_a_physical_batch(per_example_gradients, mask, C)
            accumulated_clipped_grads = add_trees(accumulated_clipped_grads, sum_of_clipped_grads_from_pb)

            return state, accumulated_clipped_grads, logical_batch_X, logical_batch_y, masks

    
    for t in range(num_iter):
        sampling_rng = jax.random.PRNGKey(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)

        #######
        # poisson subsample
        actual_batch_size = jax.device_put(jax.random.bernoulli(binomial_rng, shape=(full_data_size,), p=q).sum(),jax.devices("cpu")[0])
        n_physical_batches = actual_batch_size // physical_bs + 1 
        logical_batch_size = n_physical_batches * physical_bs
        n_masked_elements = logical_batch_size - actual_batch_size
        
        # take the logical batch
        indices = jax.random.permutation(batch_rng, full_data_size)[:logical_batch_size]
        logical_batch_X = train_images[indices]
        if private:
            logical_batch_X = logical_batch_X.reshape(-1, 1, 3, orig_dimension, orig_dimension)
        logical_batch_y = train_labels[indices]
        #######
        
        # masks
        masks = jax.device_put(jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)]),jax.devices("cpu")[0])
        
        # cast to GPU
        logical_batch_X = jax.device_put(logical_batch_X,jax.devices("gpu")[0])
        logical_batch_y = jax.device_put(logical_batch_y,jax.devices("gpu")[0])
        masks = jax.device_put(masks,jax.devices("gpu")[0])
        
        if not dynamic_slice:
            masks = jnp.array(jnp.split(masks, n_physical_batches))
            logical_batch_X_split = jnp.array(jnp.split(logical_batch_X, n_physical_batches))
            logical_batch_y_split = jnp.array(jnp.split(logical_batch_y, n_physical_batches))


        print("##### Starting gradient accumulation #####",flush=True)
        ### gradient accumulation
        params = state.params
        
        accumulated_clipped_grads0 = jax.tree.map(lambda x: 0. * x, params)
        
        start = time.time()        

        if private:            
        # _, accumulated_clipped_grads, *_ = jax.lax.fori_loop(0, k, body_fun, (state, accumulated_clipped_grads0, logical_batch_X, logical_batch_y, masks))
            _, accumulated_clipped_grads, *_ = jax.lax.fori_loop(0, n_physical_batches, body_fun, (state, accumulated_clipped_grads0, logical_batch_X, logical_batch_y, masks))
            noisy_grad = noise_addition(noise_rng, accumulated_clipped_grads, noise_std, C)
        
            # update
            state = jax.block_until_ready(update_model(state, noisy_grad))
        else:
        # _, accumulated_clipped_grads, *_ = jax.lax.fori_loop(0, k, body_fun, (state, accumulated_clipped_grads0, logical_batch_X, logical_batch_y, masks))
            _, accumulated_grads, *_ = jax.lax.fori_loop(0, n_physical_batches, body_fun, (state, accumulated_clipped_grads0, logical_batch_X, logical_batch_y, masks))
            # update
            state = jax.block_until_ready(update_model(state, accumulated_grads))
            
        end = time.time()
        duration = end-start
        
        times.append(duration)
        logical_batch_sizes.append(logical_batch_size)

        print(logical_batch_size/duration,flush=True)

        acc_iter = model_evaluation(state,splits_test,splits_labels)
        print('iteration',t,'acc',acc_iter,flush=True)

    acc_last = model_evaluation(state,splits_test,splits_labels)

    print('times \n',times,flush=True)

    print('batch sizes \n ',logical_batch_size,flush=True)

    print('accuracy at end of training',acc_last,flush=True)
    thr = np.mean(np.array(logical_batch_sizes) / np.array(times))
    return thr,acc_last