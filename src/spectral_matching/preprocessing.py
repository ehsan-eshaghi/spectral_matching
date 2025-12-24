"""
Data preprocessing functions.
"""

import numpy as np
from numpy import polyfit, polyval
from typing import Tuple

from .constants import NUMERICAL_EPS, TARGET_PERIOD_BAND
from .solvers import response_spectrum


def baseline_correction(acc: np.ndarray, time: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Apply baseline correction using polynomial detrending.

    Parameters
    ----------
    acc : np.ndarray
        Acceleration time history [m/s^2]
    time : np.ndarray
        Time array [s]
    order : int, optional
        Polynomial order for detrending (default: 2)

    Returns
    -------
    np.ndarray
        Corrected acceleration [m/s^2]
    """
    trend = polyfit(time, acc, order)
    acc_corrected = acc - polyval(trend, time)
    return acc_corrected


def scale_to_target_band(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray,
    Se_target: np.ndarray,
    band: list = TARGET_PERIOD_BAND,
    damping: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    Scale acceleration record to match target spectrum in specified period band.

    Parameters
    ----------
    acc : np.ndarray
        Acceleration time history [m/s^2]
    dt : float
        Time step [s]
    periods : np.ndarray
        Period array [s]
    Se_target : np.ndarray
        Target spectral acceleration [m/s^2]
    band : list, optional
        Target period band [T_min, T_max] (default: TARGET_PERIOD_BAND)
    damping : float, optional
        Damping ratio (default: 0.05)

    Returns
    -------
    acc_scaled : np.ndarray
        Scaled acceleration [m/s^2]
    scale_factor : float
        Applied scale factor
    """
    Sa_orig = response_spectrum(acc, dt, periods, damping=damping)
    band_mask = (periods >= band[0]) & (periods <= band[1])
    scale_factor = np.mean(
        Se_target[band_mask] / (Sa_orig[band_mask] + NUMERICAL_EPS)
    )
    acc_scaled = acc * scale_factor
    return acc_scaled, scale_factor


def compute_match_statistics(
    Sa: np.ndarray,
    Se_target: np.ndarray,
    periods: np.ndarray,
    band: list = TARGET_PERIOD_BAND,
    threshold: float = 0.9
) -> float:
    """
    Compute percentage of periods in band where Sa >= threshold * Se_target.

    Parameters
    ----------
    Sa : np.ndarray
        Computed spectral acceleration [m/s^2]
    Se_target : np.ndarray
        Target spectral acceleration [m/s^2]
    periods : np.ndarray
        Period array [s]
    band : list, optional
        Target period band [T_min, T_max] (default: TARGET_PERIOD_BAND)
    threshold : float, optional
        Matching threshold (default: 0.9)

    Returns
    -------
    float
        Percentage of periods meeting the threshold
    """
    band_mask = (periods >= band[0]) & (periods <= band[1])
    pct = 100.0 * np.sum(
        Sa[band_mask] >= threshold * Se_target[band_mask]
    ) / np.sum(band_mask)
    return pct

