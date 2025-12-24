"""
Plotting functions for visualization of results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .constants import (
    GRAVITY, PERIOD_MIN, PERIOD_MAX, TARGET_PERIOD_BAND, DAMPING
)


class FigureSaver:
    """Helper class for saving figures with consistent naming."""

    def __init__(
        self,
        output_dir: str = "figures",
        run_tag: str = "run",
        period_band: list = TARGET_PERIOD_BAND,
        damping: float = DAMPING
    ):
        """
        Initialize figure saver.

        Parameters
        ----------
        output_dir : str, optional
            Output directory for figures (default: "figures")
        run_tag : str, optional
            Tag for this run (default: "run")
        period_band : list, optional
            Target period band [T_min, T_max] (default: TARGET_PERIOD_BAND)
        damping : float, optional
            Damping ratio (default: DAMPING)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.run_tag = run_tag
        self.period_band = period_band
        self.damping = damping

    def _fig_name(self, base_name: str) -> str:
        """Generate figure filename."""
        period_band_str = f"{self.period_band[0]:.2f}-{self.period_band[1]:.2f}s"
        damping_str = f"{int(round(self.damping*100))}pct"
        return f"{base_name}_{self.run_tag}_band-{period_band_str}_damp-{damping_str}"

    def save(self, base_name: str, tight_layout: bool = True, dpi: int = 200) -> Path:
        """
        Save current figure.

        Parameters
        ----------
        base_name : str
            Base name for the figure
        tight_layout : bool, optional
            Apply tight layout (default: True)
        dpi : int, optional
            Resolution in dots per inch (default: 200)

        Returns
        -------
        Path
            Path to saved figure
        """
        if tight_layout:
            plt.tight_layout()
        filename = self._fig_name(base_name)
        png_path = self.output_dir / f"{filename}.png"
        plt.savefig(png_path, dpi=dpi)
        return png_path


def plot_spectra(
    periods: np.ndarray,
    target_spectrum: np.ndarray,
    spectrum_original: np.ndarray,
    spectrum_scaled: Optional[np.ndarray] = None,
    spectrum_matched: Optional[np.ndarray] = None,
    method: str = "FFT",
    figure_size: tuple = (8, 4)
) -> plt.Figure:
    """
    Plot response spectra comparison.

    Parameters
    ----------
    periods : np.ndarray
        Period array [s]
    target_spectrum : np.ndarray
        Target spectral acceleration [m/s^2]
    spectrum_original : np.ndarray
        Original spectral acceleration [m/s^2]
    spectrum_scaled : np.ndarray, optional
        Scaled spectral acceleration [m/s^2]
    spectrum_matched : np.ndarray, optional
        Matched spectral acceleration [m/s^2]
    method : str, optional
        Method name for title (default: "FFT")
    figure_size : tuple, optional
        Figure size (default: (8, 4))

    Returns
    -------
    plt.Figure
        Figure object
    """
    figure, axes = plt.subplots(figsize=figure_size)
    axes.plot(periods, target_spectrum/GRAVITY, label='UHS (g)', color='orange')
    axes.plot(periods, spectrum_original/GRAVITY, label='Original Sa (g)', color='blue')
    if spectrum_scaled is not None:
        axes.plot(periods, spectrum_scaled/GRAVITY, label='Scaled Sa (g)', color='green')
    if spectrum_matched is not None:
        axes.plot(periods, spectrum_matched/GRAVITY, label=f'Matched Sa (g) - {method}', color='red')
    axes.axvspan(
        TARGET_PERIOD_BAND[0], TARGET_PERIOD_BAND[1],
        color='grey', alpha=0.15
    )
    axes.set_xlim(PERIOD_MIN, PERIOD_MAX)
    axes.set_xlabel('Period (s)')
    axes.set_ylabel('Sa (g)')
    title = f'Target vs Record Sa ({TARGET_PERIOD_BAND[0]}–{TARGET_PERIOD_BAND[1]} s shaded)'
    if method != "FFT":
        title = f'Target vs Record Sa ({method})'
    axes.set_title(title)
    axes.legend()
    axes.grid(True)
    return figure


def plot_cumulative_metric(
    time: np.ndarray,
    cumulative_original: np.ndarray,
    cumulative_matched: np.ndarray,
    metric: str = 'AI',
    figure_size: tuple = (8, 4)
) -> plt.Figure:
    """
    Plot cumulative metric comparison.

    Parameters
    ----------
    time : np.ndarray
        Time array [s]
    cumulative_original : np.ndarray
        Cumulative metric for original record
    cumulative_matched : np.ndarray
        Cumulative metric for matched record
    metric : str, optional
        Metric name ('AI' or 'CAV') (default: 'AI')
    figure_size : tuple, optional
        Figure size (default: (8, 4))

    Returns
    -------
    plt.Figure
        Figure object
    """
    figure, axes = plt.subplots(figsize=figure_size)
    axes.plot(time, cumulative_original, label=f'Original {metric}', color='blue')
    axes.plot(time, cumulative_matched, label=f'Matched {metric}', color='red', linestyle='--')
    axes.set_xlabel('Time [s]')
    
    if metric == 'AI':
        axes.set_ylabel('Cumulative Arias Intensity [m/s]')
        axes.set_title('Cumulative AI Comparison')
    elif metric == 'CAV':
        axes.set_ylabel('Cumulative Absolute Velocity [m/s]')
        axes.set_title('Cumulative CAV Comparison')
    
    axes.legend()
    axes.grid(True)
    return figure


def plot_time_history(
    time: np.ndarray,
    acceleration_original: np.ndarray,
    acceleration_matched: np.ndarray,
    method: str = "",
    figure_size: tuple = (8, 4)
) -> plt.Figure:
    """
    Plot time history comparison.

    Parameters
    ----------
    time : np.ndarray
        Time array [s]
    acceleration_original : np.ndarray
        Original acceleration [m/s^2]
    acceleration_matched : np.ndarray
        Matched acceleration [m/s^2]
    method : str, optional
        Method name for label (default: "")
    figure_size : tuple, optional
        Figure size (default: (8, 4))

    Returns
    -------
    plt.Figure
        Figure object
    """
    figure, axes = plt.subplots(figsize=figure_size)
    axes.plot(time, acceleration_original, label='Original EQ')
    label_text = f'Matched EQ'
    if method:
        label_text += f' ({method})'
    axes.plot(time, acceleration_matched, label=label_text)
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Acceleration [m/s^2]')
    title = 'Time History'
    if method:
        title += f' ({method})'
    axes.set_title(title)
    axes.legend()
    axes.grid(True)
    return figure

