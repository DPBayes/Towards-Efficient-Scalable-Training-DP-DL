import math

import flax.linen as nn
import ipdb
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import scipy.stats as stats
from flax.training import train_state

from src.jax_mask_efficient import (
    accumulate_physical_batch,
    add_Gaussian_noise,
    clip_physical_batch,
    compute_accuracy_for_batch,
    compute_per_example_gradients_physical_batch,
    get_padded_logical_batch,
    model_evaluation,
    poisson_sample_logical_batch_size,
    setup_physical_batches,
    update_model,
    setup_physical_batches_distributed
)


def test_get_padded_logical_batch():
    """
    Test get_padded_logical_batch returns a batch with the correct shape and raises
    a ValueError for invalid padded sizes (negative size and size larger than N).
    """
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
    """
    Test poisson_sample_logical_batch_size returns the expected logical batch size given the sampling probability,
    and that repeated sampling under the same conditions is consistent.
    """
    rng = jax.random.key(42)
    n = 10000
    for q in [0.0, 1.0]:
        assert n * q == poisson_sample_logical_batch_size(binomial_rng=rng, dataset_size=n, q=q)

    samples = []
    for _ in range(5):
        samples.append(poisson_sample_logical_batch_size(binomial_rng=rng, dataset_size=n, q=0.5))

    assert all([s == samples[0] for s in samples])


def test_poisson_sample_logical_batch_size_chisquare():
    """
    Verifies that samples from poisson_sample_logical_batch_size follow the expected binomial
    distribution using a chi-square goodness-of-fit test.
    """
    dataset_size = int(1e4)
    num_samples = int(1e5)

    for q in [0.1, 0.45, 0.9]:
        base_key = jax.random.PRNGKey(999)
        keys = jax.random.split(base_key, num_samples)

        # draw samples and bin them into dataset_size bins
        samples = jax.vmap(lambda rng_key: poisson_sample_logical_batch_size(rng_key, dataset_size, q))(keys)
        bins = np.arange(-0.5, dataset_size + 1, 1)
        observed, _ = np.histogram(samples, bins=bins)

        # compute the expected numbers of samples in each bin
        expected = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mass_within_bin = stats.binom.cdf(bins[i + 1], dataset_size, q) - stats.binom.cdf(bins[i], dataset_size, q)
            expected[i] = num_samples * mass_within_bin

        # We need to ignore bins with expected < 5 for the chi-square test because scipy notes say this is necessary
        mask = expected >= 5

        # The sum_check=False is necessary because the tolarence is tiny and cannot be changed
        chi2_stat, pvalue = stats.chisquare(observed[mask], f_exp=expected[mask], sum_check=False)
        assert pvalue > 0.05, f"Chi-square test failed q:{q}, chi2_stat={chi2_stat}, pvalue={pvalue}"


def test_setup_physical_batches():
    """
    Test setup_physical_batches creates masks of the correct length and computes the correct number
    of physical batches based on the logical batch size and physical batch size.
    """
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

def test_setup_physical_batches_distributed():
    """
    Test setup_physical_batches creates masks of the correct length and computes the correct number
    of physical batches based on the logical batch size and physical batch size.
    """
    logical_bs = 2501
    n_devices = 4

    for p_bs in [-1, 0]:
        with pytest.raises(ValueError):
            setup_physical_batches_distributed(actual_logical_batch_size=logical_bs, physical_bs=p_bs)

    for p_bs in [1,32, logical_bs - 1, logical_bs]:
        masks, n_physical_batches, worker_batch_size, n_physical_batches_worker = setup_physical_batches_distributed(actual_logical_batch_size=logical_bs, physical_bs=p_bs,world_size=n_devices)
        assert sum(masks) == logical_bs
        assert len(masks) % p_bs == 0
        assert worker_batch_size % p_bs == 0


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
    """
    Test that per-example gradients computed for a physical batch sum to the full gradient.
    """
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
    """
    Test that clip_physical_batch clips the per-example gradients to the target norm.
    """
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
    """
    Test that accumulate_physical_batch correctly aggregates clipped per-example gradients
    according to the provided mask.
    """
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


def test_compute_accuracy_for_batch():
    """
    Test compute_accuracy_for_batch returns the correct count of predictions.
    """
    state = train_state.TrainState.create(
        apply_fn=lambda x, params: (x, None), params={}, tx=optax.sgd(learning_rate=0.1)
    )

    # All correct
    batch_X = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    batch_y = jnp.array([0, 1])
    accuracy = compute_accuracy_for_batch(state, batch_X, batch_y, resizer=None)
    assert accuracy == 2

    # All wrong
    batch_X = jnp.array([[0.5, 0.9], [0.7, 0.8]])
    batch_y = jnp.array([0, 0])
    accuracy = compute_accuracy_for_batch(state, batch_X, batch_y, resizer=None)
    assert accuracy == 0

    # Empty batch
    batch_X = jnp.array([]).reshape(0, 2)
    batch_y = jnp.array([])
    accuracy = compute_accuracy_for_batch(state, batch_X, batch_y, resizer=None)
    assert accuracy == 0


def test_model_evaluation_combined():
    """
    Test model_evaluation returns the correct accuracy.
    """
    orig_image_dimension = 2
    batch_size = 2
    n_images = 4
    test_images = jnp.zeros((n_images, 3, orig_image_dimension, orig_image_dimension))
    test_labels = jnp.zeros((n_images,), dtype=jnp.int32)

    # All correct predictions.
    state_correct = train_state.TrainState.create(
        apply_fn=lambda x, params: (jnp.array([[0.9, 0.1]] * x.shape[0]), None),
        params={},
        tx=optax.sgd(learning_rate=0.1),
    )
    acc_all_correct = model_evaluation(
        state_correct, test_images, test_labels, orig_image_dimension, batch_size=batch_size, use_gpu=False
    )
    assert acc_all_correct == 1.0

    # Zero accuracy (state predicts wrong class).
    state_wrong = train_state.TrainState.create(
        apply_fn=lambda x, params: (jnp.array([[0.1, 0.9]] * x.shape[0]), None),
        params={},
        tx=optax.sgd(learning_rate=0.1),
    )
    acc_zero = model_evaluation(
        state_wrong, test_images, test_labels, orig_image_dimension, batch_size=batch_size, use_gpu=False
    )
    assert acc_zero == 0.0

    # Intermediate accuracy.
    def custom_apply(x, params):
        condition = x[:, 0, 0, 0] < 0.5
        predictions = jnp.where(condition, jnp.array([[0.9, 0.1]] * x.shape[0]), jnp.array([[0.1, 0.9]] * x.shape[0]))
        return predictions

    state_mixed = train_state.TrainState.create(
        apply_fn=custom_apply,
        params={},
        tx=optax.sgd(learning_rate=0.1),
    )
    batch1 = jnp.zeros((2, 3, orig_image_dimension, orig_image_dimension))
    batch2 = jnp.ones((2, 3, orig_image_dimension, orig_image_dimension))
    test_images_mixed = jnp.concatenate([batch1, batch2], axis=0)
    test_labels_mixed = jnp.zeros((n_images,), dtype=jnp.int32)
    acc_intermediate = model_evaluation(
        state_mixed, test_images_mixed, test_labels_mixed, orig_image_dimension, batch_size=batch_size, use_gpu=False
    )
    assert acc_intermediate == 0.5

    # No complete batches.
    batch1 = jnp.zeros((2, 3, orig_image_dimension, orig_image_dimension))
    batch2 = jnp.ones((2, 3, orig_image_dimension, orig_image_dimension))
    batch3 = jnp.ones((1, 3, orig_image_dimension, orig_image_dimension))
    test_images_incomplete = jnp.concatenate([batch1, batch2, batch3], axis=0)
    test_labels_incomplete = jnp.zeros((5,), dtype=jnp.int32)
    acc_not_complete = model_evaluation(
        state_mixed,
        test_images_incomplete,
        test_labels_incomplete,
        orig_image_dimension,
        batch_size=batch_size,
        use_gpu=False,
    )
    assert acc_not_complete == 0.4


def test_update_model():
    """
    Test update_model updates model parameters correctly using nonzero gradients
    and remains unchanged for zero gradients.
    """
    learning_rate = 0.1
    state = train_state.TrainState.create(
        apply_fn=lambda x, params: x,
        params={"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])},
        tx=optax.sgd(learning_rate=learning_rate),
    )

    # Nonzero gradients.
    nonzero_grads = {"w": jnp.array([0.1, 0.1]), "b": jnp.array([0.1])}
    updated_state = update_model(state, nonzero_grads)
    expected_params = {
        "w": state.params["w"] - learning_rate * nonzero_grads["w"],
        "b": state.params["b"] - learning_rate * nonzero_grads["b"],
    }
    assert jnp.allclose(updated_state.params["w"], expected_params["w"])
    assert jnp.allclose(updated_state.params["b"], expected_params["b"])

    # Zero gradients; parameters should remain unchanged.
    zero_grads = {"w": jnp.zeros_like(state.params["w"]), "b": jnp.zeros_like(state.params["b"])}
    updated_state_zero = update_model(state, zero_grads)
    assert jnp.allclose(updated_state_zero.params["w"], state.params["w"])
    assert jnp.allclose(updated_state_zero.params["b"], state.params["b"])


def test_add_Gaussian_noise():
    """
    Test that add_Gaussian_noise produces repeatable noise for a fixed rng key
    and varying noise for different keys.
    """
    accumulated = {"w": jnp.ones((2, 3)), "b": jnp.ones((4,))}
    noise_std_values = [0.1, 1.0, 5.0]
    C_values = [0.1, 1.0, 5.0]

    for noise_std in noise_std_values:
        for C in C_values:
            # Use a fixed rng key and verify repeatability.
            rng_key = jax.random.PRNGKey(0)
            out1 = add_Gaussian_noise(rng_key, accumulated, noise_std, C)
            out2 = add_Gaussian_noise(rng_key, accumulated, noise_std, C)
            for key in accumulated.keys():
                assert jnp.allclose(out1[key], out2[key])
            # Using a different key should change the noise.
            rng_key_diff = jax.random.PRNGKey(1)
            out_diff = add_Gaussian_noise(rng_key_diff, accumulated, noise_std, C)
            diff_found = any(not jnp.allclose(out1[key], out_diff[key]) for key in accumulated.keys())
            assert diff_found, f"Noise did not change for noise_std {noise_std} and C {C}"


def test_add_Gaussian_noise_ks():
    """
    Test that the noise added by add_Gaussian_noise passes the Kolmogorov–Smirnov test
    against a Gaussian distribution.
    """
    accumulated = {"w": jnp.ones((2, 3)), "b": jnp.ones((4,))}
    noise_std_values = [0.1, 1.0, 5.0]
    C_values = [0.1, 1.0, 5.0]
    num_samples = int(1e4)

    for noise_std in noise_std_values:
        for C in C_values:
            samples = []
            base_key = jax.random.PRNGKey(42)
            keys = jax.random.split(base_key, num_samples)
            for key in keys:
                out = add_Gaussian_noise(key, accumulated, noise_std, C)
                for param in accumulated.keys():
                    samples.append((out[param] - accumulated[param]).flatten())
            samples = jnp.concat(samples)
            expected_std = noise_std * C
            stat, pvalue = stats.kstest(samples, cdf="norm", args=(0, expected_std))
            assert pvalue > 0.05, f"KS test failed for noise_std {noise_std} and C {C}: p={pvalue}"


def test_add_Gaussian_noise_independence():
    """
    Test that add_Gaussian_noise adds independent noise to different leaves by checking
    that their correlation is near zero.
    """
    accumulated = {"a": jnp.ones((100,)), "b": jnp.ones((100,))}
    noise_std = 2.0
    C = 0.5
    rng_key = jax.random.PRNGKey(123)
    out = add_Gaussian_noise(rng_key, accumulated, noise_std, C)
    noise_a = out["a"] - accumulated["a"]
    noise_b = out["b"] - accumulated["b"]
    corr = np.corrcoef(np.array(noise_a).flatten(), np.array(noise_b).flatten())[0, 1]
    assert abs(corr) < 0.2, f"Noise between leaves are not independent, correlation={corr}"


if __name__ == "__main__":
    test_poisson_sample_logical_batch_size_chisquare()
