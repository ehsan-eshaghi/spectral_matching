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
    acc : np.ndarray
        Acceleration array [m/s^2]
    dt : float
        Time step [s]
    """
    data = np.loadtxt(filename)
    time = data[:, 0]
    acc_raw = data[:, 1]
    dt = time[1] - time[0]

    # Convert g to m/s^2
    acc = acc_raw * GRAVITY
    # Ensure time array is consistent
    time = np.arange(len(acc)) * dt

    return time, acc, dt


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
    Se_target : np.ndarray
        Target spectral acceleration [m/s^2]
    """
    if periods is None:
        periods = np.linspace(PERIOD_MIN, PERIOD_MAX, NUM_PERIODS)

    uhs = pd.read_csv(csv_filename)
    interp_sa = interp1d(
        uhs['Period_s'],
        uhs['Sa_g'] * GRAVITY,
        kind='linear',
        fill_value='extrapolate'
    )
    Se_target = interp_sa(periods)

    return periods, Se_target


def save_acceleration_record(
    filename: str,
    time: np.ndarray,
    acc: np.ndarray,
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
    acc : np.ndarray
        Acceleration array [m/s^2] (will be converted to g)
    header : str, optional
        Header string for the file
    """
    np.savetxt(
        filename,
        np.column_stack([time, acc / GRAVITY]),
        header=header,
        fmt="%.6e"
    )

