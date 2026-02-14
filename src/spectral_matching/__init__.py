"""
Spectral matching package for earthquake ground motion records.

This package provides tools for matching earthquake acceleration records
to target response spectra using iterative FFT and Greedy Wavelet Matching methods.
"""

from .constants import (
    GRAVITY, DAMPING, TARGET_PERIOD_BAND,
    PERIOD_MIN, PERIOD_MAX, NUM_PERIODS,
    FFT_ITERS, GWM_ITERS
)
from .solvers import piecewise_exact_solver
from .metrics import arias_intensity, cumulative_absolute_velocity, cumulative_metric
from .matching import response_spectrum
from .matching import iterative_fft_match, greedy_wavelet_match, tapered_cosine_wavelet
from .io import load_acceleration_record, load_target_spectrum, save_acceleration_record
from .preprocessing import (
    baseline_correction, scale_to_target_band, compute_match_statistics
)
from .plotting import FigureSaver, plot_spectra, plot_cumulative_metric, plot_time_history

__version__ = "0.1.0"

__all__ = [
    # Constants
    'GRAVITY', 'DAMPING', 'TARGET_PERIOD_BAND',
    'PERIOD_MIN', 'PERIOD_MAX', 'NUM_PERIODS',
    'FFT_ITERS', 'GWM_ITERS',
    # Solvers
    'piecewise_exact_solver',
    # Metrics
    'arias_intensity', 'cumulative_absolute_velocity', 'cumulative_metric',
    # Matching
    'response_spectrum',
    # Matching
    'iterative_fft_match', 'greedy_wavelet_match', 'tapered_cosine_wavelet',
    # I/O
    'load_acceleration_record', 'load_target_spectrum', 'save_acceleration_record',
    # Preprocessing
    'baseline_correction', 'scale_to_target_band', 'compute_match_statistics',
    # Plotting
    'FigureSaver', 'plot_spectra', 'plot_cumulative_metric', 'plot_time_history',
]

