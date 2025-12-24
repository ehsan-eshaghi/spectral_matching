"""
File I/O operations for loading and saving data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from scipy.interpolate import interp1d

from .constants import GRAVITY, PERIOD_MIN, PERIOD_MAX, NUM_PERIODS


def load_acceleration_record(filename: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load acceleration record from file.

    Expected format: two columns (time, acceleration in g)

    Parameters
    ----------
    filename : str
        Path to the data file

    Returns
    -------
    time : np.ndarray
        Time array [s]
    acceleration : np.ndarray
        Acceleration array [m/s^2]
    time_step : float
        Time step [s]
    """
    data = np.loadtxt(filename)
    time = data[:, 0]
    acceleration_raw = data[:, 1]
    time_step = time[1] - time[0]

    # Convert g to m/s^2
    acceleration = acceleration_raw * GRAVITY
    # Ensure time array is consistent
    time = np.arange(len(acceleration)) * time_step

    return time, acceleration, time_step


def load_target_spectrum(
    csv_filename: str,
    periods: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load target spectrum from CSV file and interpolate to period grid.

    Expected CSV format: columns 'Period_s' and 'Sa_g'

    Parameters
    ----------
    csv_filename : str
        Path to CSV file with target spectrum
    periods : np.ndarray, optional
        Period array for interpolation. If None, uses default grid.

    Returns
    -------
    periods : np.ndarray
        Period array [s]
    target_spectrum : np.ndarray
        Target spectral acceleration [m/s^2]
    """
    if periods is None:
        periods = np.linspace(PERIOD_MIN, PERIOD_MAX, NUM_PERIODS)

    uniform_hazard_spectrum = pd.read_csv(csv_filename)
    interpolated_spectrum = interp1d(
        uniform_hazard_spectrum['Period_s'],
        uniform_hazard_spectrum['Sa_g'] * GRAVITY,
        kind='linear',
        fill_value='extrapolate'
    )
    target_spectrum = interpolated_spectrum(periods)

    return periods, target_spectrum


def save_acceleration_record(
    filename: str,
    time: np.ndarray,
    acceleration: np.ndarray,
    header: str = "time(s) acc(g)"
) -> None:
    """
    Save acceleration record to file.

    Parameters
    ----------
    filename : str
        Output filename
    time : np.ndarray
        Time array [s]
    acceleration : np.ndarray
        Acceleration array [m/s^2] (will be converted to g)
    header : str, optional
        Header string for the file
    """
    np.savetxt(
        filename,
        np.column_stack([time, acceleration / GRAVITY]),
        header=header,
        fmt="%.6e"
    )

