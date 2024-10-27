import jax
import jax.numpy as jnp
import numpy as np

from collections import namedtuple
import argparse

import os

import math
import time

from data import import_data_efficient_mask
from models import create_train_state

from dp_accounting_utils import compute_epsilon, calculate_noise

from jax_mask_efficient import (
    compute_physical_batch_per_example_gradients,
    add_trees,
    clip_and_accumulate_physical_batch,
    model_evaluation,
    add_Gaussian_noise,
    update_model,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _parse_arguments(args, dataset_size):
    num_steps = args.epochs * math.ceil(dataset_size / args.bs)

    q = 1 / math.ceil(dataset_size / args.bs)

    noise_std = calculate_noise(q, args.epsilon, args.target_delta, args.epochs, args.accountant)
    C = args.grad_norm

    optimizer_config = namedtuple("Config", ["learning_rate"])
    optimizer_config.learning_rate = args.lr

    num_classes = args.num_classes

    physical_bs = args.phy_bs

    orig_image_dimension, resized_image_dimension = 32, 224

    return (
        num_steps,
        noise_std,
        C,
        optimizer_config,
        num_classes,
        q,
        physical_bs,
        dataset_size,
        orig_image_dimension,
        resized_image_dimension,
    )


def main(args):

    jax.clear_caches()

    print(args, flush=True)

    train_images, train_labels, test_images, test_labels = import_data_efficient_mask()

    (
        num_steps,
        noise_std,
        C,
        optimizer_config,
        num_classes,
        q,
        physical_bs,
        dataset_size,
        orig_image_dimension,
        resized_image_dimension,
    ) = _parse_arguments(args=args, dataset_size=len(train_images))

    state = create_train_state(
        model_name=args.model,
        num_classes=num_classes,
        image_dimension=resized_image_dimension,
        optimizer_config=optimizer_config,
    )

    times = []
    logical_batch_sizes = []

    splits_test = jnp.split(test_images, 10)
    splits_labels = jnp.split(test_labels, 10)

    @jax.jit
    def body_fun(t, args):
        (
            state,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
        ) = args
        # slice
        start_idx = t * physical_bs
        pb = jax.lax.dynamic_slice(
            logical_batch_X,
            (start_idx, 0, 0, 0, 0),
            (physical_bs, 1, 3, orig_image_dimension, orig_image_dimension),
        )
        yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (physical_bs,))
        mask = jax.lax.dynamic_slice(masks, (start_idx,), (physical_bs,))

        # compute grads and clip
        per_example_gradients = compute_physical_batch_per_example_gradients(state, pb, yb)
        sum_of_clipped_grads_from_pb = clip_and_accumulate_physical_batch(per_example_gradients, mask, C)
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads, sum_of_clipped_grads_from_pb)

        return (
            state,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
        )

    for t in range(num_steps):
        #TODO: Is this deprecated? See https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html.
        sampling_rng = jax.random.PRNGKey(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)

        #######
        # poisson subsample
        actual_batch_size = jax.device_put(
            jax.random.bernoulli(binomial_rng, shape=(dataset_size,), p=q).sum(),
            jax.devices("cpu")[0],
        )
        n_physical_batches = actual_batch_size // physical_bs + 1
        logical_batch_size = n_physical_batches * physical_bs
        n_masked_elements = logical_batch_size - actual_batch_size

        # take the logical batch
        indices = jax.random.permutation(batch_rng, dataset_size)[:logical_batch_size]
        logical_batch_X = train_images[indices]
        logical_batch_X = logical_batch_X.reshape(-1, 1, 3, orig_image_dimension, orig_image_dimension)
        logical_batch_y = train_labels[indices]
        #######

        # masks
        masks = jax.device_put(
            jnp.concatenate([jnp.ones(actual_batch_size), jnp.zeros(n_masked_elements)]),
            jax.devices("cpu")[0],
        )

        # cast to GPU
        logical_batch_X = jax.device_put(logical_batch_X, jax.devices("gpu")[0])
        logical_batch_y = jax.device_put(logical_batch_y, jax.devices("gpu")[0])
        masks = jax.device_put(masks, jax.devices("gpu")[0])

        print("##### Starting gradient accumulation #####", flush=True)
        ### gradient accumulation
        params = state.params

        accumulated_clipped_grads0 = jax.tree.map(lambda x: 0.0 * x, params)

        start = time.time()

        # Main loop
        _, accumulated_clipped_grads, *_ = jax.lax.fori_loop(
            0,
            n_physical_batches,
            body_fun,
            (
                state,
                accumulated_clipped_grads0,
                logical_batch_X,
                logical_batch_y,
                masks,
            ),
        )
        noisy_grad = add_Gaussian_noise(noise_rng, accumulated_clipped_grads, noise_std, C)

        # update
        state = jax.block_until_ready(update_model(state, noisy_grad))

        end = time.time()
        duration = end - start

        times.append(duration)
        logical_batch_sizes.append(logical_batch_size)

        print(logical_batch_size / duration, flush=True)

        acc_iter = model_evaluation(state, splits_test, splits_labels)
        print("iteration", t, "acc", acc_iter, flush=True)

        # Compute privacy guarantees
        epsilon, delta = compute_epsilon(
            steps=t + 1,
            batch_size=actual_batch_size,
            num_examples=len(train_images),
            target_delta=args.target_delta,
            noise_multiplier=noise_std,
        )
        privacy_results = {"eps_rdp": epsilon, "delta_rdp": delta}
        print(privacy_results, flush=True)

    acc_last = model_evaluation(state, splits_test, splits_labels)

    print("times \n", times, flush=True)

    print("batch sizes \n ", logical_batch_size, flush=True)

    print("accuracy at end of training", acc_last, flush=True)
    thr = np.mean(np.array(logical_batch_sizes) / np.array(times))
    return thr, acc_last
