"""
Data preprocessing functions.
"""

import numpy as np
from numpy import polyfit, polyval
from typing import Tuple

from .constants import NUMERICAL_EPS, TARGET_PERIOD_BAND
from .matching import response_spectrum


def baseline_correction(acceleration: np.ndarray, time: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Apply baseline correction using polynomial detrending.

    Parameters
    ----------
    acceleration : np.ndarray
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
    trend = polyfit(time, acceleration, order)
    acceleration_corrected = acceleration - polyval(trend, time)
    return acceleration_corrected


def scale_to_target_band(
    acceleration: np.ndarray,
    time_step: float,
    periods: np.ndarray,
    target_spectrum: np.ndarray,
    band: list = TARGET_PERIOD_BAND,
    damping: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    Scale acceleration record to match target spectrum in specified period band.

    Parameters
    ----------
    acceleration : np.ndarray
        Acceleration time history [m/s^2]
    time_step : float
        Time step [s]
    periods : np.ndarray
        Period array [s]
    target_spectrum : np.ndarray
        Target spectral acceleration [m/s^2]
    band : list, optional
        Target period band [T_min, T_max] (default: TARGET_PERIOD_BAND)
    damping : float, optional
        Damping ratio (default: 0.05)

    Returns
    -------
    acceleration_scaled : np.ndarray
        Scaled acceleration [m/s^2]
    scale_factor : float
        Applied scale factor
    """
    spectrum_original = response_spectrum(acceleration, time_step, periods, damping=damping)
    band_mask = (periods >= band[0]) & (periods <= band[1])
    scale_factor = np.mean(
        target_spectrum[band_mask] / (spectrum_original[band_mask] + NUMERICAL_EPS)
    )
    acceleration_scaled = acceleration * scale_factor
    return acceleration_scaled, scale_factor


def compute_match_statistics(
    spectrum: np.ndarray,
    target_spectrum: np.ndarray,
    periods: np.ndarray,
    band: list = TARGET_PERIOD_BAND,
    threshold: float = 0.9
) -> float:
    """
    Compute percentage of periods in band where spectrum >= threshold * target_spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        Computed spectral acceleration [m/s^2]
    target_spectrum : np.ndarray
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
    match_percentage = 100.0 * np.sum(
        spectrum[band_mask] >= threshold * target_spectrum[band_mask]
    ) / np.sum(band_mask)
    return match_percentage

