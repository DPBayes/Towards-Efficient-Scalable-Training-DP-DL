from opacus.accountants.utils import get_noise_multiplier

from dp_accounting import dp_event, rdp
import warnings
import jax.numpy as jnp


def calculate_noise(
    sample_rate: float,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    accountant: str,
):
    """Calculate the noise multiplier with Opacus implementation"""
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant=accountant,
    )

    return noise_multiplier


def compute_epsilon(steps, batch_size, num_examples=60000, target_delta=1e-5, noise_multiplier=0.1):
    """Compute epsilon for DPSGD privacy accounting"""
    if num_examples * target_delta > 1.0:
        warnings.warn("Your delta might be too high.")

    print("steps", steps, flush=True)

    print("noise multiplier", noise_multiplier, flush=True)

    q = batch_size / float(num_examples)
    orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)  # type: ignore
    accountant.compose(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(noise_multiplier)),
        steps,
    )

    epsilon = accountant.get_epsilon(target_delta)
    delta = accountant.get_delta(epsilon)

    return epsilon, delta
