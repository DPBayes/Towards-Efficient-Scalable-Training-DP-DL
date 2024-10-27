import jax, optax
import jax.numpy as jnp
import numpy as np

from flax.training import train_state

from collections import namedtuple

from models import load_model
from data import normalize_and_reshape


## define some jax utility functions


@jax.jit
def add_trees(x, y):
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)


## Main functions for DP-SGD


@jax.jit
def compute_physical_batch_per_example_gradients(
    state: train_state.TrainState, batch_X: jax.typing.ArrayLike, batch_y: jax.typing.ArrayLike
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

    Returns
    -------
    px_grads : jax.typing.ArrayLike 
        The per-sample gradients of the physical batch.
    """

    resizer = lambda x: normalize_and_reshape(x)

    def loss_fn(params, X, y):
        resized_X = resizer(X)
        print(resized_X.shape, flush=True)
        logits = state.apply_fn(resized_X, params=params)[0]
        one_hot = jax.nn.one_hot(y, 100)
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
def noise_addition(rng_key: jax.Array, accumulated_clipped_grads: jax.typing.ArrayLike, noise_std: float, C: float):
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


### Parameters for training


def create_train_state(model_name: str, num_classes: int, image_dimension: int, config: namedtuple):
    """Creates initial `TrainState`."""
    rng, model, params = load_model(jax.random.PRNGKey(0), model_name, image_dimension, num_classes)

    # set the optimizer
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=jax.jit(model.__call__), params=params, tx=tx)


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
):
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


## Evaluation


def eval_fn(
    state: train_state.TrainState, batch_X: jax.typing.ArrayLike, batch_y: jax.typing.ArrayLike, num_classes: int
):
    """Computes gradients, loss and accuracy for a single batch."""

    resizer = lambda x: normalize_and_reshape(x)
    resized_X = resizer(batch_X)
    logits = state.apply_fn(resized_X, state.params)[0]
    one_hot = jax.nn.one_hot(batch_y, num_classes=num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
    predicted_class = jnp.argmax(logits, axis=-1)

    acc = jnp.mean(predicted_class == batch_y)

    return acc


def model_evaluation(
    state: train_state.TrainState, test_data: jax.typing.ArrayLike, test_labels: jax.typing.ArrayLike, num_classes: int
):

    accs = []

    for pb, yb in zip(test_data, test_labels):
        pb = jax.device_put(pb, jax.devices("gpu")[0])
        yb = jax.device_put(yb, jax.devices("gpu")[0])
        accs.append(eval_fn(state, pb, yb, num_classes=num_classes))

    return np.mean(np.array(accs))
