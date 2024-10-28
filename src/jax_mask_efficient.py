import jax, optax
import jax.numpy as jnp
import numpy as np

from flax.training import train_state

from data import normalize_and_reshape

import warnings

## define some jax utility functions


@jax.jit
def add_trees(x, y):
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)


## Main functions for DP-SGD


def poisson_sample_logical_batch_size(binomial_rng: jax.Array, dataset_size: int, q: float):
    """Sample logical batch size using Poisson subsampling with sampling probability q.

    Parameters
    ----------
    binomial_rng : jax.Array
        The PRNG key array for the sampling.
    dataset_size : int
        The size of the total training dataset.
    q : float
        The sampling probability q, must be 0 < q < 1.

    Returns
    -------
    logical_batch_size : int
        The sampled logical batch size.
    """
    logical_batch_size = jax.device_put(
        jax.random.bernoulli(binomial_rng, shape=(dataset_size,), p=q).sum(),
        jax.devices("cpu")[0],
    )
    return logical_batch_size


def get_padded_logical_batch(
    batch_rng: jax.Array, padded_logical_batch_size: int, train_X: jax.typing.ArrayLike, train_y: jax.typing.ArrayLike
):
    """Samples random padded logical batch from the data that is slightly larger than the actual logical bs
        but can be divided into n physical batches.

    Parameters
    ----------
    batch_rng : jax.Array
        The PRNG key array for sampling the batch.
    padded_logical_batch_size : int
        The size of the sampled batch (so that it can be divided into n physical batches).
    train_X : jax.typing.ArrayLike
        The training features.
    train_y : jax.typing.ArrayLike
        The training labels.

    Returns
    -------
    padded_logical_batch_X : jax.typing.ArrayLike
        The padded training features.
    padded_logical_batch_y : jax.typing.ArrayLike
        The padded logical batch training labels.
    """

    # take the logical batch
    dataset_size = len(train_y)
    indices = jax.random.permutation(batch_rng, dataset_size)[:padded_logical_batch_size]
    padded_logical_batch_X = train_X[indices]
    padded_logical_batch_y = train_y[indices]

    return padded_logical_batch_X, padded_logical_batch_y


def setup_physical_batches(
    actual_logical_batch_size: int,
    physical_bs: int,
):
    """Computed the required number of physical batches so that n (full) physical batches are created.
    This means that some elements are later thrown away.

    Parameters
    ----------
    actual_batch_size : int
        The actual sampled logical batch size.
    physical_bs : int
        The physical batch size (depends on model size and memory).

    Returns
    -------
    masks : jax.typing.ArrayLike
        A mask to throw away n_masked_elements later as they are only required for computational reasons.
    n_physical_batches : int
        The number of physical batches.
    """
    # ensure full physical batches of size `physical_bs` each
    n_physical_batches = actual_logical_batch_size // physical_bs + 1
    padded_logical_batch_size = n_physical_batches * physical_bs

    # masks (throw away n_masked_elements later as they are only required for computing)
    n_masked_elements = padded_logical_batch_size - actual_logical_batch_size
    masks = jax.device_put(
        jnp.concatenate([jnp.ones(actual_logical_batch_size), jnp.zeros(n_masked_elements)]),
        jax.devices("cpu")[0],
    )

    return masks, n_physical_batches


@jax.jit
def compute_per_example_gradients_physical_batch(
    state: train_state.TrainState, batch_X: jax.typing.ArrayLike, batch_y: jax.typing.ArrayLike, num_classes: int
):
    """Computes the per-example gradients for a physical batch.

    Parameters
    ----------
    state : train_state.TrainState
        The model train state.
    batch_X : jax.typing.ArrayLike
        The features of the physical batch.
    batch_y : jax.typing.ArrayLike
        The labels of the physical batch.
    num_classes : int
        The number of classes for one-hot encoding.

    Returns
    -------
    px_grads : jax.typing.ArrayLike
        The per-sample gradients of the physical batch.
    """

    resizer = lambda x: normalize_and_reshape(x)

    def loss_fn(params, X, y):
        resized_X = resizer(X)
        logits = state.apply_fn(resized_X, params=params)[0]
        one_hot = jax.nn.one_hot(y, num_classes=num_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        assert len(loss) == 1
        return loss.sum()

    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    px_grads = jax.vmap(grad_fn, in_axes=(0, 0))(batch_X, batch_y)

    return px_grads


@jax.jit
def clip_and_accumulate_physical_batch(px_grads: jax.typing.ArrayLike, mask: jax.typing.ArrayLike, C: float):
    """Clip and accumulate per-example gradients of a physical batch.

    Parameters
    ----------
    px_grads : jax.typing.ArrayLike
        The per-sample gradients of the physical batch.
    mask : jax.typing.ArrayLike
        A mask to filter out gradients that are discarded as a small number of per-examples gradients
        is only computed to keep the physical batch size fixed.
    C : float
        The clipping norm of DP-SGD.

    Returns
    -------
    acc_px_grads: jax.typing.ArrayLike
        The clipped and accumulated per-example gradients after discarding the additional per-example gradients.
    """

    def _clip_mask_and_sum(x: jax.typing.ArrayLike, mask: jax.typing.ArrayLike, clipping_multiplier: float):

        new_shape = (-1,) + (1,) * (x.ndim - 1)
        mask = mask.reshape(new_shape)
        clipping_multiplier = clipping_multiplier.reshape(new_shape)

        return jnp.sum(x * mask * clipping_multiplier, axis=0)

    px_per_param_sq_norms = jax.tree.map(lambda x: jnp.linalg.norm(x.reshape(x.shape[0], -1), axis=-1) ** 2, px_grads)
    flattened_px_per_param_sq_norms, tree_def = jax.tree_util.tree_flatten(px_per_param_sq_norms)

    px_grad_norms = jnp.sqrt(jnp.sum(jnp.array(flattened_px_per_param_sq_norms), axis=0))

    clipping_multiplier = jnp.minimum(1.0, C / px_grad_norms)

    return jax.tree.map(lambda x: _clip_mask_and_sum(x, mask, clipping_multiplier), px_grads)


@jax.jit
def add_Gaussian_noise(
    rng_key: jax.Array, accumulated_clipped_grads: jax.typing.ArrayLike, noise_std: float, C: float
):
    """Add Gaussian noise to the clipped and accumulated per-example gradients of a logical batch.

    Parameters
    ----------
    rng_key : jax.Array
        The PRNG key array for generating the Gaussian noise.
    accumulated_clipped_grads : jax.typing.ArrayLike
        The clipped and accumulated per-example gradients of a logical batch.
    noise_std : float
        The standard deviation to be added as computed with a privacy accountant.
    C : float
        The clipping norm of DP-SGD.

    Returns
    -------
    noisy_grad : jax.typing.ArrayLike
        The noisy gradient of the logical batch.
    """
    num_vars = len(jax.tree_util.tree_leaves(accumulated_clipped_grads))
    treedef = jax.tree_util.tree_structure(accumulated_clipped_grads)
    new_key, *all_keys = jax.random.split(rng_key, num=num_vars + 1)
    # draw noise
    noise = jax.tree_util.tree_map(
        lambda g, k: noise_std * C * jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        accumulated_clipped_grads,
        jax.tree_util.tree_unflatten(treedef, all_keys),
    )

    updates = add_trees(accumulated_clipped_grads, noise)
    return updates


@jax.jit
def update_model(state: train_state.TrainState, grads):
    return state.apply_gradients(grads=grads)


## NON-DP


@jax.jit
def compute_gradients_non_dp(
    state: train_state.TrainState,
    batch_X: jax.typing.ArrayLike,
    batch_y: jax.typing.ArrayLike,
    mask: jax.typing.ArrayLike,
    num_classes: int,
):
    """Computes the non-DP gradients for a physical batch.

    Parameters
    ----------
    state : train_state.TrainState
        The model train state.
    batch_X : jax.typing.ArrayLike
        The features of the physical batch.
    batch_y : jax.typing.ArrayLike
        The labels of the physical batch.
    mask : jax.typing.ArrayLike
        A mask to filter out gradients that are discarded as a small number of gradients
        is only computed to keep the physical batch size fixed.
    num_classes : int
        The number of classes for one-hot encoding.

    Returns
    -------
    acc_grads: jax.typing.ArrayLike
        The accumulated per-example gradients after discarding the additional gradients (see mask).
    """

    resizer = lambda x: normalize_and_reshape(x)

    def loss_fn(params, X, y):
        resized_X = resizer(X)
        logits = state.apply_fn(resized_X, params=params)[0]
        one_hot = jax.nn.one_hot(y, num_classes=num_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        masked_loss = loss * mask
        return masked_loss.sum()

    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    sum_of_grads = grad_fn(batch_X, batch_y)

    return sum_of_grads


## Evaluation


def compute_accuracy_for_batch(
    state: train_state.TrainState, batch_X: jax.typing.ArrayLike, batch_y: jax.typing.ArrayLike
):
    """Computes accuracy for a single batch."""

    resizer = lambda x: normalize_and_reshape(x)
    resized_X = resizer(batch_X)
    logits = state.apply_fn(resized_X, state.params)[0]
    predicted_class = jnp.argmax(logits, axis=-1)

    correct = jnp.sum(predicted_class == batch_y)

    return correct


def model_evaluation(
    state: train_state.TrainState, test_images: jax.typing.ArrayLike, test_labels: jax.typing.ArrayLike, batch_size:int = 50
):
    diff = len(test_images) % batch_size

    if diff != 0:
        batch_size = batch_size - diff
        warnings.warn(f'The batch size does not divide the size of the test set, fixed the new batch size to {batch_size}')

    num_splits = int(len(test_images)/batch_size)
    splits_test = jnp.split(test_images, num_splits)
    splits_test_labels = jnp.split(test_labels, num_splits)
    corr = 0 
    total = 0

    for pb, yb in zip(splits_test, splits_test_labels):
        pb = jax.device_put(pb, jax.devices("gpu")[0])
        yb = jax.device_put(yb, jax.devices("gpu")[0])
        # TODO: This won't be correct when len(pb) not the same for all pb in test_data.
        corr += compute_accuracy_for_batch(state, pb, yb)
        total += len(yb)

    return corr / total
