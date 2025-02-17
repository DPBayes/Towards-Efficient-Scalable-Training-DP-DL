import os
from collections import namedtuple

import jax
import jax.numpy as jnp
import optax
import ipdb

from src.data import load_from_huggingface
from src.dp_accounting_utils import calculate_noise, compute_epsilon
from src.jax_mask_efficient import (
    add_Gaussian_noise,
    add_trees,
    clip_physical_batch,
    accumulate_physical_batch,
    compute_per_example_gradients_physical_batch,
    get_padded_logical_batch,
    model_evaluation,
    poisson_sample_logical_batch_size,
    setup_physical_batches,
    update_model,
)
from src.models import create_train_state

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
jax.clear_caches()

USE_GPU = jax.device_count('gpu') > 0

def test_simple_end_to_end_non_DP():
    lr = 1e-3
    num_steps = 120

    train_images, train_labels, test_images, test_labels = load_from_huggingface(
        "uoft-cs/cifar10", cache_dir=None, feature_name="img"
    )
    ORIG_IMAGE_DIMENSION, RESIZED_IMAGE_DIMENSION = 32, 32
    train_images = (
        train_images[train_labels < 2]
        .transpose(0, 3, 1, 2)
        .reshape(-1, 1, 3, ORIG_IMAGE_DIMENSION, ORIG_IMAGE_DIMENSION)
    )
    train_labels = train_labels[train_labels < 2]
    test_images = test_images[test_labels < 2].transpose(0, 3, 1, 2)
    test_labels = test_labels[test_labels < 2]
    batch_size = 100

    num_classes = 2
    dataset_size = len(train_labels)

    optimizer_config = namedtuple("Config", ["learning_rate"])
    optimizer_config.learning_rate = lr

    state = create_train_state(
        model_name="small",
        num_classes=num_classes,
        image_dimension=RESIZED_IMAGE_DIMENSION,
        optimizer_config=optimizer_config,
    )

    for t in range(num_steps):
        sampling_rng = jax.random.PRNGKey(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)

        indicies = jax.random.permutation(batch_rng, jnp.arange(dataset_size))[:batch_size]

        batch_X = train_images[indicies]
        batch_y = train_labels[indicies]

        def loss_fn(params, X, y):
            resized_X = X
            logits = state.apply_fn(resized_X, params=params)
            one_hot = jax.nn.one_hot(y, num_classes=num_classes)
            loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
            return jnp.sum(loss)

        grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
        full_grads = grad_fn(batch_X, batch_y)

        state = jax.block_until_ready(update_model(state, full_grads))

    acc_last = model_evaluation(
        state, test_images, test_labels, batch_size=10, use_gpu=USE_GPU, orig_image_dimension=ORIG_IMAGE_DIMENSION
    )

    acc_train = model_evaluation(
        state,
        train_images.reshape(-1, 3, ORIG_IMAGE_DIMENSION, ORIG_IMAGE_DIMENSION),
        train_labels,
        batch_size=10,
        use_gpu=USE_GPU,
        orig_image_dimension=ORIG_IMAGE_DIMENSION,
    )

    assert acc_last > 0.8
    assert acc_train > 0.8

def test_simple_end_to_end():
    lr = 1e-3
    num_steps = 100
    subsampling_ratio = 0.05
    clipping_norm = 1
    target_epsilon = 8
    target_delta = 1e-5
    physical_bs = 2
    accountant = "pld"
    train_images, train_labels, test_images, test_labels = load_from_huggingface(
        "uoft-cs/cifar10", cache_dir=None, feature_name="img"
    )
    ORIG_IMAGE_DIMENSION, RESIZED_IMAGE_DIMENSION = 32, 32
    train_images = train_images[train_labels < 2].transpose(0, 3, 1, 2)
    train_labels = train_labels[train_labels < 2]
    test_images = test_images[test_labels < 2].transpose(0, 3, 1, 2)
    test_labels = test_labels[test_labels < 2]


    num_classes = 2
    dataset_size = len(train_labels)

    optimizer_config = namedtuple("Config", ["learning_rate"])
    optimizer_config.learning_rate = lr

    state = create_train_state(
        model_name="small",
        num_classes=num_classes,
        image_dimension=RESIZED_IMAGE_DIMENSION,
        optimizer_config=optimizer_config,
    )

    noise_std = calculate_noise(
        sample_rate=subsampling_ratio,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        steps=num_steps,
        accountant=accountant,
    )

    def process_physical_batch(t, params_tuple):
        state, accumulated_clipped_grads, logical_batch_X, logical_batch_y, masks = params_tuple
        start_idx = t * physical_bs
        pb = jax.lax.dynamic_slice(
            logical_batch_X,
            (start_idx, 0, 0, 0, 0),
            (physical_bs, 1, 3, ORIG_IMAGE_DIMENSION, ORIG_IMAGE_DIMENSION),
        )
        yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (physical_bs,))
        mask = jax.lax.dynamic_slice(masks, (start_idx,), (physical_bs,))
        per_example_gradients = compute_per_example_gradients_physical_batch(state, pb, yb, num_classes)
        clipped_grads_from_pb = clip_physical_batch(per_example_gradients, clipping_norm)
        sum_of_clipped_grads_from_pb = accumulate_physical_batch(clipped_grads_from_pb, mask)
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads, sum_of_clipped_grads_from_pb)
        return state, accumulated_clipped_grads, logical_batch_X, logical_batch_y, masks

    for t in range(num_steps):
        sampling_rng = jax.random.PRNGKey(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)
        actual_batch_size = poisson_sample_logical_batch_size(
            binomial_rng=binomial_rng, dataset_size=dataset_size, q=subsampling_ratio
        )
        masks, n_physical_batches = setup_physical_batches(
            actual_logical_batch_size=actual_batch_size,
            physical_bs=physical_bs,
        )
        padded_logical_batch_X, padded_logical_batch_y = get_padded_logical_batch(
            batch_rng=batch_rng,
            padded_logical_batch_size=len(masks),
            train_X=train_images,
            train_y=train_labels,
        )
        padded_logical_batch_X = padded_logical_batch_X.reshape(-1, 1, 3, ORIG_IMAGE_DIMENSION, ORIG_IMAGE_DIMENSION)

        if USE_GPU:
            padded_logical_batch_X = jax.device_put(padded_logical_batch_X, jax.devices("gpu")[0])
            padded_logical_batch_y = jax.device_put(padded_logical_batch_y, jax.devices("gpu")[0])
            masks = jax.device_put(masks, jax.devices("gpu")[0])

        accumulated_clipped_grads0 = jax.tree.map(lambda x: 0.0 * x, state.params)

        _, accumulated_clipped_grads, _, _, _ = jax.lax.fori_loop(
            0,
            n_physical_batches,
            process_physical_batch,
            (state, accumulated_clipped_grads0, padded_logical_batch_X, padded_logical_batch_y, masks),
        )

        noisy_grad = add_Gaussian_noise(noise_rng, accumulated_clipped_grads, noise_std, clipping_norm)
        state = jax.block_until_ready(update_model(state, noisy_grad))

    epsilon, delta = compute_epsilon(
        noise_multiplier=noise_std,
        sample_rate=subsampling_ratio,
        steps=num_steps,
        target_delta=target_delta,
        accountant=accountant,
    )

    acc_last = model_evaluation(
        state, test_images, test_labels, batch_size=10, use_gpu=USE_GPU, orig_image_dimension=ORIG_IMAGE_DIMENSION
    )

    assert epsilon <= target_epsilon
    assert acc_last > 0.8
    assert delta <= target_delta

