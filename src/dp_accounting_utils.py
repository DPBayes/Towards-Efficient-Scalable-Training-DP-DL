from dp_accounting import rdp, pld, mechanism_calibration
from dp_accounting.dp_event import (
    PoissonSampledDpEvent,
    GaussianDpEvent,
    SelfComposedDpEvent,
)


def _check_params(sample_rate: float, target_delta: float, steps: int):
    if sample_rate < 0 or sample_rate > 1:
        raise ValueError("sample_rate parameter needs to be 0 <= and <= 1.")

    if target_delta < 0 or target_delta > 1:
        raise ValueError("target_delta parameter needs to be 0 <= and <= 1.")

    if steps < 1 and type(steps) == int:
        raise ValueError("steps parameter must be >= 1.")


def calculate_noise(
    sample_rate: float,
    steps: int,
    target_epsilon: float,
    target_delta: float,
    accountant: str = "pld",
):
    """
    Computes the required Gaussian noise standard deviation for DP-SGD
    given the relevant hyperparameters of DP-SGD. Note that my default
    the PLD accountant is used.

    Parameters
    ----------
    sample_rate : float
        The sampling rate for Poisson subsampling. Note that 0 <= sampling_rate <= 1.
    steps : int
        The number of steps that should be accounted for in total during training.
    target_epsilon : float
        The desired epsilon at `target_delta` that should be reached after taking all steps.
    target_delta : float
        The target delta of the DP-SGD run.
    accountant : str, optional
        The privacy accountant, can be "pld" or "rdp, by default "pld".

    Returns
    -------
    noise_multiplier : float
        The required Gaussian noise standard deviation for DP-SGD.

    Raises
    ------
    ValueError
        Raise if parameters are misspecified, e.g. negative target_epsilon.
    """

    # check those params that are common to many dp accounting methods
    _check_params(sample_rate=sample_rate, target_delta=target_delta, steps=steps)

    if target_epsilon < 0:
        raise ValueError("target_epsilon parameter needs to be positive.")

    # check and select accountant class
    if accountant == "pld":
        accountant = pld.PLDAccountant
    elif accountant == "rdp":
        accountant = rdp.RdpAccountant
    else:
        raise ValueError("accountant parameter needs to be either 'pld' or 'rdp'.")

    dp_event = lambda sigma: SelfComposedDpEvent(PoissonSampledDpEvent(sample_rate, GaussianDpEvent(sigma)), steps)

    noise_multiplier = mechanism_calibration.calibrate_dp_mechanism(
        accountant,
        dp_event,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )
    return noise_multiplier


def compute_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float = 1e-5,
    accountant: str = "pld",
):
    # check those params that are common to many dp accounting methods
    _check_params(sample_rate=sample_rate, target_delta=target_delta, steps=steps)

    if noise_multiplier < 0:
        raise ValueError("noise parameter needs to be positive.")

    # check and setup accountant
    if accountant == "pld":
        accountant = pld.PLDAccountant()
    elif accountant == "rdp":
        accountant = rdp.RdpAccountant()
    else:
        raise ValueError("accountant parameter needs to be either 'pld' or 'rdp'.")

    print("steps", steps, flush=True)

    print("noise multiplier", noise_multiplier, flush=True)

    accountant.compose(
        PoissonSampledDpEvent(sample_rate, GaussianDpEvent(noise_multiplier)),
        steps,
    )

    epsilon = accountant.get_epsilon(target_delta)
    delta = accountant.get_delta(epsilon)

    return epsilon, delta
