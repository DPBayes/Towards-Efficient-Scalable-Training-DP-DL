import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training import train_state

from src.jax_mask_efficient import (
    accumulate_physical_batch,
    clip_physical_batch,
    compute_per_example_gradients_physical_batch,
    get_padded_logical_batch,
    poisson_sample_logical_batch_size,
    setup_physical_batches,
)


def test_get_padded_logical_batch():
    N = 200
    feature_dim = 32
    train_X = np.ones((N, feature_dim))
    train_y = np.ones(N)
    padded_logical_batch_size = None
    rng = jax.random.key(42)

    for padded_logical_batch_size in [0, 100, N]:
        padded_logical_batch_X, padded_logical_batch_y = get_padded_logical_batch(
            batch_rng=rng, padded_logical_batch_size=padded_logical_batch_size, train_X=train_X, train_y=train_y
        )
        assert padded_logical_batch_X.shape == (padded_logical_batch_size, feature_dim)
        assert padded_logical_batch_y.shape == (padded_logical_batch_size,)

    for padded_logical_batch_size in [-1, N + 1]:
        with pytest.raises(ValueError):
            get_padded_logical_batch(
                batch_rng=rng, padded_logical_batch_size=padded_logical_batch_size, train_X=train_X, train_y=train_y
            )


def test_poisson_sample_logical_batch_size():
    rng = jax.random.key(42)
    n = 10000
    for q in [0.0, 1.0]:
        assert n * q == poisson_sample_logical_batch_size(binomial_rng=rng, dataset_size=n, q=q)

    samples = []
    for _ in range(5):
        samples.append(poisson_sample_logical_batch_size(binomial_rng=rng, dataset_size=n, q=0.5))

    assert all([s == samples[0] for s in samples])


def test_setup_physical_batches():
    logical_bs = 2501

    for p_bs in [-1, 0]:
        with pytest.raises(ValueError):
            setup_physical_batches(actual_logical_batch_size=logical_bs, physical_bs=p_bs)

    for p_bs in [1, logical_bs - 1, logical_bs]:
        masks, n_physical_batches = setup_physical_batches(actual_logical_batch_size=logical_bs, physical_bs=p_bs)
        assert sum(masks) == logical_bs
        assert len(masks) == math.ceil(logical_bs / p_bs) * p_bs
        assert n_physical_batches == math.ceil(logical_bs / p_bs)

    # physical_bs > logical_bs
    masks, n_physical_batches = setup_physical_batches(
        actual_logical_batch_size=logical_bs, physical_bs=logical_bs + 1
    )
    assert sum(masks) == logical_bs
    assert len(masks) == logical_bs + 1
    assert n_physical_batches == 1


def _setup_state():
    class CNN(nn.Module):
        """A simple CNN model."""

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=64, kernel_size=(7, 7), strides=2)(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(features=256)(x)
            x = nn.relu(x)
            x = nn.Dense(features=100)(x)
            return x

    model = CNN()

    input_shape = (1, 3, 32, 32)
    x = jax.random.normal(jax.random.key(42), input_shape)

    variables = model.init(jax.random.key(42), x)
    # model.apply(variables, x)
    state = train_state.TrainState.create(
        apply_fn=lambda x, params: model.apply({"params": params}, x), params=variables["params"], tx=optax.adam(0.1)
    )
    return state


def test_compute_per_example_gradients_physical_batch():
    state = _setup_state()
    n = 20
    batch_X = np.random.random_sample((n, 1, 3, 32, 32))
    batch_y = np.ones((n,), dtype=int)
    dummy_resizer = lambda x: x  # Dummy resizer
    px_grads = compute_per_example_gradients_physical_batch(
        state=state, batch_X=batch_X, batch_y=batch_y, num_classes=100, resizer=None
    )

    def loss_fn(params, X, y):
        resized_X = dummy_resizer(X)
        logits = state.apply_fn(resized_X, params=params)
        one_hot = jax.nn.one_hot(y, num_classes=100)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).flatten()
        # assert len(loss) == 1
        return np.sum(loss)

    grad_fn = lambda X, y: jax.grad(loss_fn)(state.params, X, y)
    full_grads = grad_fn(batch_X.reshape(n, 3, 32, 32), batch_y)
    summed_px_grads = jax.tree.map(lambda x: x.sum(0), px_grads)
    for key in full_grads.keys():
        for subkey in full_grads[key].keys():
            assert np.allclose(full_grads[key][subkey], summed_px_grads[key][subkey], atol=1e-6)


def test_clip_physical_batch():
    state = _setup_state()
    n = 10
    LARGE_NUMBER = 1e3

    batch_X = np.random.random_sample((n, 1, 3, 32, 32))
    batch_y = jnp.ones((n,), dtype=int)
    px_grads = compute_per_example_gradients_physical_batch(
        state=state, batch_X=batch_X, batch_y=batch_y, num_classes=100, resizer=None
    )

    big_px_grads = jax.tree.map(lambda x: jnp.ones_like(x) * LARGE_NUMBER, px_grads)
    num_parameters = sum([x.size for x in jax.tree.leaves(state.params)])

    expected_un_clipped_l2_norm = jnp.sqrt(num_parameters) * LARGE_NUMBER

    for c in [0.1, 10, expected_un_clipped_l2_norm + 1]:
        clipped_px_grads = clip_physical_batch(px_grads=big_px_grads, C=c)
        expected_norm = min(c, expected_un_clipped_l2_norm)
        squared_acc_px_grads_norms = jax.tree.map(
            lambda x: jnp.linalg.norm(x.reshape(x.shape[0], -1), axis=-1) ** 2, clipped_px_grads
        )
        actual_norm = jnp.sqrt(sum(jax.tree.flatten(squared_acc_px_grads_norms)[0]))
        assert jnp.allclose(expected_norm, actual_norm)


def test_accumulate_physical_batch():
    state = _setup_state()
    n = 10
    LARGE_NUMBER = 1e3

    batch_X = np.random.random_sample((n, 1, 3, 32, 32))
    batch_y = jnp.ones((n,), dtype=int)
    px_grads = compute_per_example_gradients_physical_batch(
        state=state, batch_X=batch_X, batch_y=batch_y, num_classes=100, resizer=None
    )

    big_px_grads = jax.tree.map(lambda x: jnp.ones_like(x) * LARGE_NUMBER, px_grads)

    for m in [0, 1, n]:
        m_mask = np.zeros(n)
        m_mask[:m] = 1
        accumulated_grads = accumulate_physical_batch(clipped_px_grads=big_px_grads, mask=m_mask)

        for key in accumulated_grads.keys():
            for subkey in accumulated_grads[key].keys():
                assert np.allclose(accumulated_grads[key][subkey], m * big_px_grads[key][subkey])


if __name__ == "__main__":
    test_accumulate_physical_batch()
