import argparse
import os
import math
import time
import warnings
import jax

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


def jax_is_gpu_available():
    # https://github.com/jax-ml/jax/issues/17624
    # Get a list of devices available to JAX
    devices = jax.devices()

    # Check if any of the devices are GPUs
    for device in devices:
        if "gpu" in device.device_kind.lower():
            return True

    return False


def _parse_arguments(args, dataset_size):

    if dataset_size * args.target_delta > 1.0:
        warnings.warn("Your delta might be too high.")

    subsampling_ratio = 1 / math.ceil(dataset_size / args.logical_bs)

    optimizer_config = namedtuple("Config", ["learning_rate"])
    optimizer_config.learning_rate = args.lr

    return (
        optimizer_config,
        subsampling_ratio,
    )


def main(args):

    jax.clear_caches()

    print(args, flush=True)

    train_images, train_labels, test_images, test_labels = load_from_huggingface("cifar100", cache_dir=None)
    ORIG_IMAGE_DIMENSION, RESIZED_IMAGE_DIMENSION = 32, 224

    num_classes = len(np.unique(train_labels))
    dataset_size = len(train_labels)

    (
        optimizer_config,
        subsampling_ratio,
    ) = _parse_arguments(args=args, dataset_size=dataset_size)

    noise_std = calculate_noise(
        sample_rate=subsampling_ratio,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        steps=args.num_steps,
        accountant=args.accountant,
    )

    state = create_train_state(
        model_name=args.model,
        num_classes=num_classes,
        image_dimension=RESIZED_IMAGE_DIMENSION,
        optimizer_config=optimizer_config,
    )

    times = []
    logical_batch_sizes = []

    @jax.jit
    def body_fun(t, params):
        (
            state,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
        ) = params
        # slice
        start_idx = t * args.physical_bs
        pb = jax.lax.dynamic_slice(
            logical_batch_X,
            (start_idx, 0, 0, 0, 0),
            (args.physical_bs, 1, 3, ORIG_IMAGE_DIMENSION, ORIG_IMAGE_DIMENSION),
        )
        yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (args.physical_bs,))
        mask = jax.lax.dynamic_slice(masks, (start_idx,), (args.physical_bs,))

        # compute grads and clip
        per_example_gradients = compute_per_example_gradients_physical_batch(state, pb, yb, num_classes)
        sum_of_clipped_grads_from_pb = clip_and_accumulate_physical_batch(
            per_example_gradients, mask, args.clipping_norm
        )
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads, sum_of_clipped_grads_from_pb)

        return (
            state,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
        )

    for t in range(args.num_steps):

        sampling_rng = jax.random.key(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)

        #######
        # poisson subsample
        actual_batch_size = poisson_sample_logical_batch_size(
            binomial_rng=binomial_rng, dataset_size=dataset_size, q=subsampling_ratio
        )

        # determine padded_logical_bs so that there are full physical batches
        # and create appropriate masks to mask out unnessary elements later
        masks, n_physical_batches = setup_physical_batches(
            actual_logical_batch_size=actual_batch_size,
            physical_bs=args.physical_bs,
        )

        # get random padded logical batches that are slighly larger actual batch size
        padded_logical_batch_X, padded_logical_batch_y = get_padded_logical_batch(
            batch_rng=batch_rng,
            padded_logical_batch_size=len(masks),
            train_X=train_images,
            train_y=train_labels,
        )

        padded_logical_batch_X = padded_logical_batch_X.reshape(-1, 1, 3, ORIG_IMAGE_DIMENSION, ORIG_IMAGE_DIMENSION)

        # cast to GPU
        if jax_is_gpu_available():
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
        noisy_grad = add_Gaussian_noise(noise_rng, accumulated_clipped_grads, noise_std, args.clipping_norm)

        # update
        state = jax.block_until_ready(update_model(state, noisy_grad))

        end = time.time()
        duration = end - start

        times.append(duration)
        logical_batch_sizes.append(actual_batch_size)

        print(actual_batch_size / duration, flush=True)

        acc_iter = model_evaluation(state, test_images, test_labels, test_bs_size=10)
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

    acc_last = model_evaluation(state, test_images, test_labels, test_bs_size=10)

    print("times \n", times, flush=True)

    print("batch sizes \n ", actual_batch_size, flush=True)

    print("accuracy at end of training", acc_last, flush=True)
    thr = np.mean(np.array(logical_batch_sizes) / np.array(times))
    return thr, acc_last


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0005, type=float, help="learning rate")
    parser.add_argument("--num_steps", default=3, type=int, help="Number of steps")
    parser.add_argument("--logical_bs", default=1000, type=int, help="Logical batch size")
    parser.add_argument("--clipping_norm", default=0.1, type=float, help="max grad norm")

    parser.add_argument("--target_epsilon", default=1, type=float, help="target epsilon")
    parser.add_argument("--target_delta", default=1e-5, type=float, help="target delta")

    parser.add_argument(
        "--model",
        default="google/vit-base-patch16-224",
        type=str,
        help="The name of the model (for loading from timm library).",
    )
    parser.add_argument("--physical_bs", default=50, type=int, help="Physical Batch Size")
    parser.add_argument("--accountant", default="pld", type=str, help="The privacy accountant for DP training.")

    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()
    main(args=args)
