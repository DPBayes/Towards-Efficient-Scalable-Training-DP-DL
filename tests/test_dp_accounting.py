import pytest
from src.dp_accounting_utils import calculate_noise, compute_epsilon
import numpy as np

def test_dp_accounting_roundtrip_rdp():
    """
    This test computes the noise multiplier for given target epsilon and delta over a number of steps,
    then calculates epsilon and delta from that noise multiplier, verifying that the computed epsilon
    does not exceed the target and that the delta is as expected.
    """
    sample_rate = 0.1
    steps = 10
    target_epsilon = 1
    target_delta = 1e-5
    accountant = "rdp"
    noise_multiplier_computed = calculate_noise(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        steps=steps,
        sample_rate=sample_rate,
        accountant=accountant,
    )
    epsilon_computed, delta_computed = compute_epsilon(
        noise_multiplier=noise_multiplier_computed,
        sample_rate=sample_rate,
        steps=steps,
        target_delta=target_delta,
        accountant=accountant,
    )
    assert epsilon_computed <= target_epsilon
    assert np.isclose(target_delta, delta_computed, rtol=1e-12, atol=1e-14)

def test_dp_accounting_roundtrip_pld():
    """
    This test computes the noise multiplier for given target epsilon and delta over a number of steps with the PLD accountant,
    then calculates epsilon and delta from that noise multiplier, verifying that the computed epsilon does not exceed the target
    and that the delta is as expected.
    """
    sample_rate = 0.1
    steps = 10
    target_epsilon = 1
    target_delta = 1e-5
    accountant = "pld"
    noise_multiplier_computed = calculate_noise(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        steps=steps,
        sample_rate=sample_rate,
        accountant=accountant,
    )
    epsilon_computed, delta_computed = compute_epsilon(
        noise_multiplier=noise_multiplier_computed,
        sample_rate=sample_rate,
        steps=steps,
        target_delta=target_delta,
        accountant=accountant,
    )
    assert epsilon_computed <= target_epsilon
    assert np.isclose(target_delta, delta_computed, rtol=1e-12, atol=1e-14)

def test_invalid_accountant():
    """
    Test that using an invalid accountant parameter raises a ValueError.
    """
    sample_rate = 0.1
    steps = 10
    target_epsilon = 1
    target_delta = 1e-5
    invalid_accountant = "unknown"
    with pytest.raises(ValueError):
        calculate_noise(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            steps=steps,
            sample_rate=sample_rate,
            accountant=invalid_accountant,
        )

def test_negative_target_epsilon():
    """
    Test that providing a negative target_epsilon raises a ValueError.
    """
    sample_rate = 0.1
    steps = 10
    target_epsilon = -1
    target_delta = 1e-5
    with pytest.raises(ValueError):
        calculate_noise(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            steps=steps,
            sample_rate=sample_rate,
            accountant="pld",
        )

def test_negative_target_delta():
    """
    Test that providing a negative target_delta raises a ValueError.
    """
    sample_rate = 0.1
    steps = 10
    target_epsilon = 1
    target_delta = -1e-5
    with pytest.raises(ValueError):
        calculate_noise(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            steps=steps,
            sample_rate=sample_rate,
            accountant="rdp",
        )

def test_compute_epsilon_invalid_accountant():
    """
    Test that compute_epsilon raises a ValueError when an invalid accountant parameter is provided.
    """
    sample_rate = 0.1
    steps = 10
    target_delta = 1e-5
    noise_multiplier = 1.0
    with pytest.raises(ValueError):
        compute_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=steps,
            target_delta=target_delta,
            accountant="invalid"
        )

def test_compute_epsilon_invalid_target_delta():
    """
    Test that compute_epsilon raises a ValueError when target_delta is not in [0, 1].
    """
    sample_rate = 0.1
    steps = 10
    noise_multiplier = 1.0
    # target_delta > 1 should raise an error
    with pytest.raises(ValueError):
        compute_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=steps,
            target_delta=1.5,
            accountant="pld"
        )
    # target_delta < 0 should raise an error
    with pytest.raises(ValueError):
        compute_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=steps,
            target_delta=-1e-5,
            accountant="rdp"
        )

def test_compute_epsilon_invalid_steps():
    """
    Test that compute_epsilon raises a ValueError when steps is not an integer >= 1.
    """
    sample_rate = 0.1
    target_delta = 1e-5
    noise_multiplier = 1.0
    # steps < 1 should raise an error
    with pytest.raises(ValueError):
        compute_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=0,
            target_delta=target_delta,
            accountant="pld"
        )

def test_calculate_noise_invalid_sample_rate():
    """
    Test that calculate_noise raises a ValueError when sample_rate is not in [0, 1].
    """
    steps = 10
    target_epsilon = 1
    target_delta = 1e-5
    with pytest.raises(ValueError):
        calculate_noise(
            sample_rate=-0.1,
            steps=steps,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            accountant="pld",
        )
    with pytest.raises(ValueError):
        calculate_noise(
            sample_rate=1.1,
            steps=steps,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            accountant="rdp",
        )

def test_calculate_noise_invalid_steps():
    """
    Test that calculate_noise raises a ValueError when steps is not an integer >= 1.
    """
    sample_rate = 0.1
    target_epsilon = 1
    target_delta = 1e-5
    with pytest.raises(ValueError):
        calculate_noise(
            sample_rate=sample_rate,
            steps=0,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            accountant="pld",
        )