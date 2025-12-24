"""
Earthquake intensity metrics computation.
"""

import numpy as np
from math import pi
from typing import Literal

from .constants import GRAVITY


def arias_intensity(acc: np.ndarray, dt: float) -> float:
    """
    Compute Arias Intensity (AI).

    AI = (pi / (2*g)) * integral(a(t)^2 dt)

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time history [m/s^2]
    dt : float
        Time step [s]

    Returns
    -------
    float
        Arias Intensity [m/s]
    """
    return (pi / (2.0 * GRAVITY)) * np.sum(acc ** 2) * dt


def cumulative_absolute_velocity(acc: np.ndarray, dt: float) -> float:
    """
    Compute Cumulative Absolute Velocity (CAV).

    CAV = integral(|a(t)| dt)

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time history [m/s^2]
    dt : float
        Time step [s]

    Returns
    -------
    float
        Cumulative Absolute Velocity [m/s]
    """
    return np.sum(np.abs(acc)) * dt


def cumulative_metric(
    acc: np.ndarray,
    dt: float,
    metric: Literal['AI', 'CAV'] = 'AI'
) -> np.ndarray:
    """
    Compute cumulative metric over time (AI or CAV).

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time history [m/s^2]
    dt : float
        Time step [s]
    metric : {'AI', 'CAV'}, optional
        Metric type to compute (default: 'AI')

    Returns
    -------
    np.ndarray
        Cumulative metric time history [m/s]

    Raises
    ------
    ValueError
        If metric is not 'AI' or 'CAV'
    """
    if metric == 'AI':
        return np.cumsum(acc ** 2) * dt * (pi / (2.0 * GRAVITY))
    if metric == 'CAV':
        return np.cumsum(np.abs(acc)) * dt
    raise ValueError("metric must be 'AI' or 'CAV'")

