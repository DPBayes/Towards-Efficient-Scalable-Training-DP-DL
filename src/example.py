import argparse
import os
import math
import time
import warnings
import jax

import jax.numpy as jnp
import numpy as np

from collections import namedtuple

from data import load_from_huggingface
from models import create_train_state

from dp_accounting_utils import compute_epsilon, calculate_noise

from jax_mask_efficient import (
    compute_per_example_gradients_physical_batch,
    add_trees,
    clip_and_accumulate_physical_batch,
    get_padded_logical_batch,
    model_evaluation,
    add_Gaussian_noise,
    poisson_sample_logical_batch_size,
    setup_physical_batches,
    update_model,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _parse_arguments(args, dataset_size):
    num_steps = args.epochs * math.ceil(dataset_size / args.bs)

    if dataset_size * args.target_delta > 1.0:
        warnings.warn("Your delta might be too high.")

    q = 1 / math.ceil(dataset_size / args.bs)

    noise_std = calculate_noise(
        sample_rate=q,
        target_epsilon=args.epsilon,
        target_delta=args.target_delta,
        steps=num_steps,
        accountant=args.accountant,
    )
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

    train_images, train_labels, test_images, test_labels = load_from_huggingface("cifar100", cache_dir=None)

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
        per_example_gradients = compute_per_example_gradients_physical_batch(state, pb, yb)
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

        sampling_rng = jax.random.key(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)

        #######
        # poisson subsample
        actual_batch_size = poisson_sample_logical_batch_size(
            binomial_rng=binomial_rng, dataset_size=dataset_size, q=q
        )

        # determine padded_logical_bs so that there are full physical batches
        # and create appropriate masks to mask out unnessary elements later
        masks, n_physical_batches = setup_physical_batches(
            actual_batch_size=actual_batch_size,
            physical_bs=physical_bs,
        )

        # get random padded logical batches that are slighly larger actual batch size
        padded_logical_batch_X, padded_logical_batch_y = get_padded_logical_batch(
            batch_rng=batch_rng,
            padded_logical_batch_size=len(masks),
            train_X=train_images,
            train_y=train_labels,
        )

        padded_logical_batch_X = padded_logical_batch_X.reshape(-1, 1, 3, orig_image_dimension, orig_image_dimension)

        # cast to GPU
        padded_logical_batch_X = jax.device_put(padded_logical_batch_X, jax.devices("gpu")[0])
        padded_logical_batch_y = jax.device_put(padded_logical_batch_y, jax.devices("gpu")[0])
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
                padded_logical_batch_X,
                padded_logical_batch_y,
                masks,
            ),
        )
        noisy_grad = add_Gaussian_noise(noise_rng, accumulated_clipped_grads, noise_std, C)

        # update
        state = jax.block_until_ready(update_model(state, noisy_grad))

        end = time.time()
        duration = end - start

        times.append(duration)
        logical_batch_sizes.append(actual_batch_size)

        print(actual_batch_size / duration, flush=True)

        acc_iter = model_evaluation(state, splits_test, splits_labels)
        print("iteration", t, "acc", acc_iter, flush=True)

        # Compute privacy guarantees
        epsilon, delta = compute_epsilon(
            noise_multiplier=noise_std,
            sample_rate=q,
            steps=t + 1,
            target_delta=args.target_delta,
            accountant=args.accountant,
        )
        privacy_results = {"accountant": args.accountant, "epsilon": epsilon, "delta": delta}
        print(privacy_results, flush=True)

    acc_last = model_evaluation(state, splits_test, splits_labels)

    print("times \n", times, flush=True)

    print("batch sizes \n ", actual_batch_size, flush=True)

    print("accuracy at end of training", acc_last, flush=True)
    thr = np.mean(np.array(logical_batch_sizes) / np.array(times))
    return thr, acc_last
