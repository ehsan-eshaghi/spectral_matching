"""
Single-degree-of-freedom (SDOF) system solvers and response spectrum computation.
"""

import numpy as np
from math import sqrt, pi
from typing import Tuple

from .constants import DAMPING


def piecewise_exact_history(
    acc: np.ndarray,
    dt: float,
    wn: float,
    xi: float = DAMPING
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Piecewise-exact SDOF solver with time histories.

    Solves the equation of motion for a SDOF system:
        u'' + 2*xi*wn*u' + wn^2*u = -a_g(t)

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time history [m/s^2]
    dt : float
        Time step [s]
    wn : float
        Natural frequency [rad/s]
    xi : float, optional
        Damping ratio (default: DAMPING from constants)

    Returns
    -------
    u : np.ndarray
        Displacement time history [m]
    v : np.ndarray
        Velocity time history [m/s]
    a_rel : np.ndarray
        Relative acceleration time history [m/s^2]
    a_abs : np.ndarray
        Absolute acceleration time history [m/s^2]
    """
    acc = np.asarray(acc, dtype=float)
    n = len(acc)

    wd = wn * sqrt(max(0.0, 1.0 - xi**2))
    k = wn**2

    exp_term = np.exp(-xi * wn * dt)
    sin_wd_dt = np.sin(wd * dt) if wd > 0 else 0.0
    cos_wd_dt = np.cos(wd * dt) if wd > 0 else 1.0
    xi_sqrt = sqrt(max(1.0e-300, 1.0 - xi**2))

    # Recurrence coefficients
    A = exp_term * (xi/xi_sqrt * sin_wd_dt + cos_wd_dt)
    B = exp_term * (1.0/wd * sin_wd_dt) if wd > 0 else dt * exp_term
    C = (1.0/k) * (
        (2.0*xi)/(wn*dt)
        + exp_term * (
            ((1.0 - 2.0*xi**2)/(wd*dt) - xi/xi_sqrt) * sin_wd_dt
            - (1.0 + (2.0*xi)/(wn*dt)) * cos_wd_dt
        )
    )
    D = (1.0/k) * (
        1.0 - (2.0*xi)/(wn*dt)
        + exp_term * (
            ((2.0*xi**2 - 1.0)/(wd*dt)) * sin_wd_dt
            + (2.0*xi)/(wn*dt) * cos_wd_dt
        )
    )

    A1 = -exp_term * (wn/xi_sqrt) * sin_wd_dt if wd > 0 else -wn*dt*exp_term
    B1 = exp_term * (cos_wd_dt - (xi/xi_sqrt) * sin_wd_dt)
    C1 = (1.0/k) * (
        -1.0/dt + exp_term * (((wn/xi_sqrt) + xi/(dt*xi_sqrt)) * sin_wd_dt + (1.0/dt) * cos_wd_dt)
    )
    D1 = (1.0/k) * (1.0/dt - (exp_term/dt) * ((xi/xi_sqrt) * sin_wd_dt + cos_wd_dt))

    # Initialize arrays
    u = np.zeros(n)
    v = np.zeros(n)
    a_rel = np.zeros(n)

    # March forward
    for i in range(n - 1):
        Fn = -acc[i]
        Fnp1 = -acc[i + 1]

        u_next = A * u[i] + B * v[i] + C * Fn + D * Fnp1
        v_next = A1 * u[i] + B1 * v[i] + C1 * Fn + D1 * Fnp1

        u[i + 1] = u_next
        v[i + 1] = v_next
        a_rel[i + 1] = -2.0 * xi * wn * v_next - (wn ** 2) * u_next - acc[i + 1]

    a_abs = a_rel + acc
    return u, v, a_rel, a_abs


def response_spectrum(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float = DAMPING
) -> np.ndarray:
    """
    Compute response spectrum (Sa) using the piecewise-exact solver.

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time history [m/s^2]
    dt : float
        Time step [s]
    periods : np.ndarray
        Array of periods [s] for which to compute Sa
    damping : float, optional
        Damping ratio (default: DAMPING from constants)

    Returns
    -------
    Sa : np.ndarray
        Spectral acceleration [m/s^2] for each period
    """
    periods = np.asarray(periods, dtype=float)
    Sa = np.empty(len(periods), dtype=float)
    for i, T in enumerate(periods):
        wn = 2.0 * pi / T
        a_abs = piecewise_exact_history(acc, dt, wn, damping)[3]
        Sa[i] = np.max(np.abs(a_abs))
    return Sa

