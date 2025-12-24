"""
Single-degree-of-freedom (SDOF) system solvers.
"""

import numpy as np
from math import sqrt, pi

from .constants import DAMPING


def piecewise_exact_solver(
    acceleration: np.ndarray,
    time_step: float,
    natural_frequency: float,
    damping_ratio: float = DAMPING
) -> np.ndarray:
    """
    Piecewise-exact SDOF solver returning absolute acceleration.

    Solves the equation of motion for a SDOF system:
        u'' + 2*damping_ratio*natural_frequency*u' + natural_frequency^2*u = -a_g(t)

    Parameters
    ----------
    acceleration : np.ndarray
        Ground acceleration time history [m/s^2]
    time_step : float
        Time step [s]
    natural_frequency : float
        Natural frequency [rad/s]
    damping_ratio : float, optional
        Damping ratio (default: DAMPING from constants)

    Returns
    -------
    absolute_acceleration : np.ndarray
        Absolute acceleration time history [m/s^2]
    """
    acceleration = np.asarray(acceleration, dtype=float)
    n = len(acceleration)

    damped_frequency = natural_frequency * sqrt(max(0.0, 1.0 - damping_ratio**2))
    stiffness = natural_frequency**2

    exponential_term = np.exp(-damping_ratio * natural_frequency * time_step)
    sin_damped_dt = np.sin(damped_frequency * time_step) if damped_frequency > 0 else 0.0
    cos_damped_dt = np.cos(damped_frequency * time_step) if damped_frequency > 0 else 1.0
    damping_sqrt = sqrt(max(1.0e-300, 1.0 - damping_ratio**2))

    # Recurrence coefficients
    coeff_A = exponential_term * (damping_ratio/damping_sqrt * sin_damped_dt + cos_damped_dt)
    coeff_B = exponential_term * (1.0/damped_frequency * sin_damped_dt) if damped_frequency > 0 else time_step * exponential_term
    coeff_C = (1.0/stiffness) * (
        (2.0*damping_ratio)/(natural_frequency*time_step)
        + exponential_term * (
            ((1.0 - 2.0*damping_ratio**2)/(damped_frequency*time_step) - damping_ratio/damping_sqrt) * sin_damped_dt
            - (1.0 + (2.0*damping_ratio)/(natural_frequency*time_step)) * cos_damped_dt
        )
    )
    coeff_D = (1.0/stiffness) * (
        1.0 - (2.0*damping_ratio)/(natural_frequency*time_step)
        + exponential_term * (
            ((2.0*damping_ratio**2 - 1.0)/(damped_frequency*time_step)) * sin_damped_dt
            + (2.0*damping_ratio)/(natural_frequency*time_step) * cos_damped_dt
        )
    )

    coeff_A1 = -exponential_term * (natural_frequency/damping_sqrt) * sin_damped_dt if damped_frequency > 0 else -natural_frequency*time_step*exponential_term
    coeff_B1 = exponential_term * (cos_damped_dt - (damping_ratio/damping_sqrt) * sin_damped_dt)
    coeff_C1 = (1.0/stiffness) * (
        -1.0/time_step + exponential_term * (((natural_frequency/damping_sqrt) + damping_ratio/(time_step*damping_sqrt)) * sin_damped_dt + (1.0/time_step) * cos_damped_dt)
    )
    coeff_D1 = (1.0/stiffness) * (1.0/time_step - (exponential_term/time_step) * ((damping_ratio/damping_sqrt) * sin_damped_dt + cos_damped_dt))

    # Initialize arrays
    displacement = np.zeros(n)
    velocity = np.zeros(n)
    relative_acceleration = np.zeros(n)

    # March forward
    for i in range(n - 1):
        force_n = -acceleration[i]
        force_np1 = -acceleration[i + 1]

        displacement_next = coeff_A * displacement[i] + coeff_B * velocity[i] + coeff_C * force_n + coeff_D * force_np1
        velocity_next = coeff_A1 * displacement[i] + coeff_B1 * velocity[i] + coeff_C1 * force_n + coeff_D1 * force_np1

        displacement[i + 1] = displacement_next
        velocity[i + 1] = velocity_next
        relative_acceleration[i + 1] = -2.0 * damping_ratio * natural_frequency * velocity_next - (natural_frequency ** 2) * displacement_next - acceleration[i + 1]

    absolute_acceleration = relative_acceleration + acceleration
    return absolute_acceleration
