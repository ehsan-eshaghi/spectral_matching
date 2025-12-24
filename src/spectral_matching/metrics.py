"""
Earthquake intensity metrics computation.
"""

import numpy as np
from math import pi
from typing import Literal

from .constants import GRAVITY


def arias_intensity(acceleration: np.ndarray, time_step: float) -> float:
    """
    Compute Arias Intensity (AI).

    AI = (pi / (2*g)) * integral(a(t)^2 dt)

    Parameters
    ----------
    acceleration : np.ndarray
        Ground acceleration time history [m/s^2]
    time_step : float
        Time step [s]

    Returns
    -------
    float
        Arias Intensity [m/s]
    """
    return (pi / (2.0 * GRAVITY)) * np.sum(acceleration ** 2) * time_step


def cumulative_absolute_velocity(acceleration: np.ndarray, time_step: float) -> float:
    """
    Compute Cumulative Absolute Velocity (CAV).

    CAV = integral(|a(t)| dt)

    Parameters
    ----------
    acceleration : np.ndarray
        Ground acceleration time history [m/s^2]
    time_step : float
        Time step [s]

    Returns
    -------
    float
        Cumulative Absolute Velocity [m/s]
    """
    return np.sum(np.abs(acceleration)) * time_step


def cumulative_metric(
    acceleration: np.ndarray,
    time_step: float,
    metric: Literal['AI', 'CAV'] = 'AI'
) -> np.ndarray:
    """
    Compute cumulative metric over time (AI or CAV).

    Parameters
    ----------
    acceleration : np.ndarray
        Ground acceleration time history [m/s^2]
    time_step : float
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
        return np.cumsum(acceleration ** 2) * time_step * (pi / (2.0 * GRAVITY))
    if metric == 'CAV':
        return np.cumsum(np.abs(acceleration)) * time_step
    raise ValueError("metric must be 'AI' or 'CAV'")

