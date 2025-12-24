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

    def _fig_name(self, base: str) -> str:
        """Generate figure filename."""
        band = f"{self.period_band[0]:.2f}-{self.period_band[1]:.2f}s"
        damp = f"{int(round(self.damping*100))}pct"
        return f"{base}_{self.run_tag}_band-{band}_damp-{damp}"

    def save(self, base: str, tight: bool = True, dpi: int = 200) -> Path:
        """
        Save current figure.

        Parameters
        ----------
        base : str
            Base name for the figure
        tight : bool, optional
            Apply tight layout (default: True)
        dpi : int, optional
            Resolution in dots per inch (default: 200)

        Returns
        -------
        Path
            Path to saved figure
        """
        if tight:
            plt.tight_layout()
        fname = self._fig_name(base)
        png_path = self.output_dir / f"{fname}.png"
        plt.savefig(png_path, dpi=dpi)
        return png_path


def plot_spectra(
    periods: np.ndarray,
    Se_target: np.ndarray,
    Sa_orig: np.ndarray,
    Sa_scaled: Optional[np.ndarray] = None,
    Sa_matched: Optional[np.ndarray] = None,
    method: str = "FFT",
    figsize: tuple = (8, 4)
) -> plt.Figure:
    """
    Plot response spectra comparison.

    Parameters
    ----------
    periods : np.ndarray
        Period array [s]
    Se_target : np.ndarray
        Target spectral acceleration [m/s^2]
    Sa_orig : np.ndarray
        Original spectral acceleration [m/s^2]
    Sa_scaled : np.ndarray, optional
        Scaled spectral acceleration [m/s^2]
    Sa_matched : np.ndarray, optional
        Matched spectral acceleration [m/s^2]
    method : str, optional
        Method name for title (default: "FFT")
    figsize : tuple, optional
        Figure size (default: (8, 4))

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(periods, Se_target/GRAVITY, label='UHS (g)', color='orange')
    ax.plot(periods, Sa_orig/GRAVITY, label='Original Sa (g)', color='blue')
    if Sa_scaled is not None:
        ax.plot(periods, Sa_scaled/GRAVITY, label='Scaled Sa (g)', color='green')
    if Sa_matched is not None:
        ax.plot(periods, Sa_matched/GRAVITY, label=f'Matched Sa (g) - {method}', color='red')
    ax.axvspan(
        TARGET_PERIOD_BAND[0], TARGET_PERIOD_BAND[1],
        color='grey', alpha=0.15
    )
    ax.set_xlim(PERIOD_MIN, PERIOD_MAX)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Sa (g)')
    title = f'Target vs Record Sa ({TARGET_PERIOD_BAND[0]}–{TARGET_PERIOD_BAND[1]} s shaded)'
    if method != "FFT":
        title = f'Target vs Record Sa ({method})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


def plot_cumulative_metric(
    time: np.ndarray,
    cum_orig: np.ndarray,
    cum_matched: np.ndarray,
    metric: str = 'AI',
    figsize: tuple = (8, 4)
) -> plt.Figure:
    """
    Plot cumulative metric comparison.

    Parameters
    ----------
    time : np.ndarray
        Time array [s]
    cum_orig : np.ndarray
        Cumulative metric for original record
    cum_matched : np.ndarray
        Cumulative metric for matched record
    metric : str, optional
        Metric name ('AI' or 'CAV') (default: 'AI')
    figsize : tuple, optional
        Figure size (default: (8, 4))

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, cum_orig, label=f'Original {metric}', color='blue')
    ax.plot(time, cum_matched, label=f'Matched {metric}', color='red', linestyle='--')
    ax.set_xlabel('Time [s]')
    
    if metric == 'AI':
        ax.set_ylabel('Cumulative Arias Intensity [m/s]')
        ax.set_title('Cumulative AI Comparison')
    elif metric == 'CAV':
        ax.set_ylabel('Cumulative Absolute Velocity [m/s]')
        ax.set_title('Cumulative CAV Comparison')
    
    ax.legend()
    ax.grid(True)
    return fig


def plot_time_history(
    time: np.ndarray,
    acc_orig: np.ndarray,
    acc_matched: np.ndarray,
    method: str = "",
    figsize: tuple = (8, 4)
) -> plt.Figure:
    """
    Plot time history comparison.

    Parameters
    ----------
    time : np.ndarray
        Time array [s]
    acc_orig : np.ndarray
        Original acceleration [m/s^2]
    acc_matched : np.ndarray
        Matched acceleration [m/s^2]
    method : str, optional
        Method name for label (default: "")
    figsize : tuple, optional
        Figure size (default: (8, 4))

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, acc_orig, label='Original EQ')
    label = f'Matched EQ'
    if method:
        label += f' ({method})'
    ax.plot(time, acc_matched, label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [m/s^2]')
    title = 'Time History'
    if method:
        title += f' ({method})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig

