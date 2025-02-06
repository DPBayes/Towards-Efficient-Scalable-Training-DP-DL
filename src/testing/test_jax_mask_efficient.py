from src.jax_mask_efficient import get_padded_logical_batch
import jax
import numpy as np
import pytest


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
