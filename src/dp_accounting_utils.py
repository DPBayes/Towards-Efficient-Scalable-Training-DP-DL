import warnings
import jax.numpy as jnp

from dp_accounting import rdp, pld, mechanism_calibration
from dp_accounting.dp_event import (
    PoissonSampledDpEvent,
    GaussianDpEvent,
    SelfComposedDpEvent,
)
from dp_accounting.pld.accountant import get_smallest_subsampled_gaussian_noise
from dp_accounting.pld.common import DifferentialPrivacyParameters


def calculate_noise(
    sample_rate: float,
    target_epsilon: float,
    target_delta: float,
    steps: int,
    accountant: str,
):

    dp_event = lambda sigma: SelfComposedDpEvent(
        PoissonSampledDpEvent(sample_rate, GaussianDpEvent(sigma)), steps
    )

    if accountant == "pld":
        accountant = pld.PLDAccountant(orders)
    elif accountant == "rdp":
        orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
        accountant = rdp.RdpAccountant
    else:
        raise ValueError("accountant parameter needs to be either 'prv' or 'rdp'")

    noise_multiplier = mechanism_calibration.calibrate_dp_mechanism(
        accountant,
        dp_event,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )
    return noise_multiplier


def compute_epsilon(
    steps: int,
    sample_rate: int,
    target_delta: float = 1e-5,
    noise_multiplier: float = 0.1,
):
    """Compute epsilon for DPSGD privacy accounting"""
    if num_examples * target_delta > 1.0:
        warnings.warn("Your delta might be too high.")

    print("steps", steps, flush=True)

    print("noise multiplier", noise_multiplier, flush=True)

    orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    accountant = rdp.RdpAccountant(orders)  # type: ignore
    accountant.compose(
        PoissonSampledDpEvent(sample_rate, GaussianDpEvent(noise_multiplier)),
        steps,
    )

    epsilon = accountant.get_epsilon(target_delta)
    delta = accountant.get_delta(epsilon)

    return epsilon, delta
