from dp_accounting_utils import calculate_noise, compute_epsilon
import numpy as np


def test_dp_accounting_roundtrip_rdp():
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