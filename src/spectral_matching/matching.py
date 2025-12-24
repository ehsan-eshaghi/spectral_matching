"""
Spectral matching algorithms: Iterative FFT and Greedy Wavelet Matching.
"""

import numpy as np
from math import sqrt, pi, atan
from typing import Tuple
from scipy.signal import convolve
from scipy.signal.windows import tukey

from .constants import (
    DAMPING, FFT_ITERS, FFT_SMOOTH_WIDTH, FFT_RATIO_CLIP_MIN, FFT_RATIO_CLIP_MAX,
    FFT_GAIN_CLIP_MIN, FFT_GAIN_CLIP_MAX, FFT_TUKEY_ALPHA, TARGET_PERIOD_BAND,
    NUMERICAL_EPS, GWM_ITERS, GWM_TOL, GWM_GAMMA, GWM_AI_MULTIPLIER, GRAVITY
)
from .solvers import piecewise_exact_solver
from .metrics import arias_intensity


def iterative_fft_match(
    initial_acceleration: np.ndarray,
    time_step: float,
    periods: np.ndarray,
    target_spectrum: np.ndarray,
    num_iterations: int = FFT_ITERS,
    period_band: list = TARGET_PERIOD_BAND,
    smooth_width: int = FFT_SMOOTH_WIDTH,
    ratio_clip_min: float = FFT_RATIO_CLIP_MIN,
    ratio_clip_max: float = FFT_RATIO_CLIP_MAX,
    gain_clip_min: float = FFT_GAIN_CLIP_MIN,
    gain_clip_max: float = FFT_GAIN_CLIP_MAX,
    numerical_stability_eps: float = NUMERICAL_EPS,
    tukey_alpha: float = FFT_TUKEY_ALPHA,
    damping: float = DAMPING
) -> np.ndarray:
    """
    Iterative FFT-based spectral matching algorithm.

    Matches the response spectrum of an acceleration record to a target spectrum
    by iteratively adjusting the frequency content in the FFT domain.

    Parameters
    ----------
    initial_acceleration : np.ndarray
        Initial acceleration time history [m/s^2]
    time_step : float
        Time step [s]
    periods : np.ndarray
        Period array [s] for spectrum computation
    target_spectrum : np.ndarray
        Target spectral acceleration [m/s^2] for each period
    num_iterations : int, optional
        Number of iterations (default: FFT_ITERS)
    period_band : list, optional
        Target period band [T_min, T_max] in seconds (default: TARGET_PERIOD_BAND)
    smooth_width : int, optional
        Smoothing kernel width for gain function (default: FFT_SMOOTH_WIDTH)
    ratio_clip_min : float, optional
        Minimum ratio clipping value (default: FFT_RATIO_CLIP_MIN)
    ratio_clip_max : float, optional
        Maximum ratio clipping value (default: FFT_RATIO_CLIP_MAX)
    gain_clip_min : float, optional
        Minimum gain clipping value (default: FFT_GAIN_CLIP_MIN)
    gain_clip_max : float, optional
        Maximum gain clipping value (default: FFT_GAIN_CLIP_MAX)
    numerical_stability_eps : float, optional
        Small epsilon for numerical stability (default: NUMERICAL_EPS)
    tukey_alpha : float, optional
        Tukey window alpha parameter (default: FFT_TUKEY_ALPHA)
    damping : float, optional
        Damping ratio for response spectrum (default: DAMPING)

    Returns
    -------
    np.ndarray
        Matched acceleration time history [m/s^2]
    """
    acceleration = initial_acceleration.copy()
    n = len(acceleration)
    n_fft = int(2 ** np.ceil(np.log2(n)))

    periods = np.asarray(periods)
    target_spectrum = np.asarray(target_spectrum)
    frequency_centers = 1.0 / periods
    frequencies = np.fft.rfftfreq(n_fft, time_step)
    frequency_low, frequency_high = 1.0 / period_band[1], 1.0 / period_band[0]

    for _ in range(num_iterations):
        spectrum_current = response_spectrum(acceleration, time_step, periods, damping=damping)
        ratio = np.clip(
            target_spectrum / (spectrum_current + numerical_stability_eps),
            ratio_clip_min,
            ratio_clip_max
        )

        # Map period-domain gains to frequency grid
        sort_index = np.argsort(frequency_centers)
        gain_interpolated = np.interp(
            frequencies, frequency_centers[sort_index], ratio[sort_index],
            left=1.0, right=1.0
        )
        gain = np.ones_like(frequencies)
        band_mask = (frequencies >= frequency_low) & (frequencies <= frequency_high)
        gain[band_mask] = gain_interpolated[band_mask]

        # Smooth and clip
        if smooth_width > 1:
            kernel = np.ones(smooth_width) / smooth_width
            gain = np.clip(
                convolve(gain, kernel, mode='same'),
                gain_clip_min,
                gain_clip_max
            )
        else:
            gain = np.clip(gain, gain_clip_min, gain_clip_max)

        # Window, FFT, apply gain, IFFT
        window = tukey(n, alpha=tukey_alpha)
        acceleration_padded = np.zeros(n_fft)
        acceleration_padded[:n] = acceleration * window
        fft_coefficients = np.fft.rfft(acceleration_padded)
        acceleration_new = np.fft.irfft(fft_coefficients * gain)[:n]
        acceleration = acceleration_new

    return acceleration


def response_spectrum(
    acceleration: np.ndarray,
    time_step: float,
    periods: np.ndarray,
    damping: float = DAMPING,
    solver = None
) -> np.ndarray:
    """
    Compute response spectrum using a specified SDOF solver.

    Parameters
    ----------
    acceleration : np.ndarray
        Ground acceleration time history [m/s^2]
    time_step : float
        Time step [s]
    periods : np.ndarray
        Array of periods [s] for which to compute spectrum
    damping : float, optional
        Damping ratio (default: DAMPING from constants)
    solver : callable, optional
        SDOF solver function. Must have signature:
        solver(acceleration, time_step, natural_frequency, damping_ratio) -> np.ndarray
        returning absolute_acceleration.
        If None, uses piecewise_exact_solver from solvers module (default: None)

    Returns
    -------
    spectrum : np.ndarray
        Spectral acceleration [m/s^2] for each period
    """
    if solver is None:
        solver = piecewise_exact_solver
    
    periods = np.asarray(periods, dtype=float)
    spectrum = np.empty(len(periods), dtype=float)
    for i, period in enumerate(periods):
        natural_frequency = 2.0 * pi / period
        absolute_acceleration = solver(acceleration, time_step, natural_frequency, damping)
        spectrum[i] = np.max(np.abs(absolute_acceleration))
    return spectrum


def tapered_cosine_wavelet(
    time: np.ndarray,
    target_time: float,
    frequency: float,
    time_step: float,
    damping_ratio: float = DAMPING,
    solver = None
) -> Tuple[np.ndarray, float, float]:
    """
    Generate a tapered cosine wavelet with phase alignment.

    Parameters
    ----------
    time : np.ndarray
        Time array [s]
    target_time : float
        Target time for peak response [s]
    frequency : float
        Frequency [Hz]
    time_step : float
        Time step [s]
    damping_ratio : float, optional
        Damping ratio (default: DAMPING)
    solver : callable, optional
        SDOF solver function. Must have signature:
        solver(acceleration, time_step, natural_frequency, damping_ratio) -> np.ndarray
        returning absolute_acceleration.
        If None, uses piecewise_exact_solver from solvers module (default: None)

    Returns
    -------
    wavelet : np.ndarray
        Wavelet time history
    max_response_amplitude : float
        Maximum response amplitude
    response_sign : float
        Sign of maximum response (+1 or -1)
    """
    if solver is None:
        solver = piecewise_exact_solver
    
    angular_frequency = 2 * pi * frequency
    damped_angular_frequency = angular_frequency * sqrt(max(0.0, 1 - damping_ratio**2))
    gamma = 1.178 * frequency ** (-0.93)

    # Build wavelet with zero phase shift, find system response peak time, then align
    time_offset = time - target_time
    wavelet_temp = np.cos(damped_angular_frequency * time_offset) * np.exp(- (time_offset / gamma) ** 2)

    natural_frequency = angular_frequency
    absolute_acceleration_temp = solver(wavelet_temp, time_step, natural_frequency, damping_ratio)

    index_wavelet = np.argmax(np.abs(absolute_acceleration_temp))
    time_wavelet = time[index_wavelet]

    # Recreate with phase shift so response peaks at target_time
    time_delta = time_wavelet - target_time
    time_offset = time - target_time + time_delta
    wavelet = np.cos(damped_angular_frequency * time_offset) * np.exp(- (time_offset / gamma) ** 2)

    absolute_acceleration = solver(wavelet, time_step, natural_frequency, damping_ratio)
    index_max = np.argmax(np.abs(absolute_acceleration))
    max_response_amplitude = np.abs(absolute_acceleration[index_max])
    response_sign = np.sign(absolute_acceleration[index_max])

    return wavelet, max_response_amplitude, response_sign


def greedy_wavelet_match(
    initial_acceleration: np.ndarray,
    time_step: float,
    time: np.ndarray,
    periods: np.ndarray,
    target_spectrum: np.ndarray,
    damping: float = DAMPING,
    max_iterations: int = GWM_ITERS,
    tolerance: float = GWM_TOL,
    relaxation_factor: float = GWM_GAMMA,
    arias_intensity_max_multiplier: float = GWM_AI_MULTIPLIER,
    period_band: list = TARGET_PERIOD_BAND,
    solver = None
) -> np.ndarray:
    """
    Greedy Wavelet Matching (GWM) algorithm for spectral matching.

    Iteratively adds tapered cosine wavelets to match the target spectrum
    while constraining the Arias Intensity.

    Parameters
    ----------
    initial_acceleration : np.ndarray
        Initial acceleration time history [m/s^2]
    time_step : float
        Time step [s]
    time : np.ndarray
        Time array [s]
    periods : np.ndarray
        Period array [s] for spectrum computation
    target_spectrum : np.ndarray
        Target spectral acceleration [m/s^2] for each period
    damping : float, optional
        Damping ratio (default: DAMPING)
    max_iterations : int, optional
        Maximum number of iterations (default: GWM_ITERS)
    tolerance : float, optional
        Convergence tolerance (default: GWM_TOL)
    relaxation_factor : float, optional
        Relaxation factor (default: GWM_GAMMA)
    arias_intensity_max_multiplier : float, optional
        Maximum AI multiplier relative to initial (default: GWM_AI_MULTIPLIER)
    period_band : list, optional
        Target period band [T_min, T_max] (default: TARGET_PERIOD_BAND)
    solver : callable, optional
        SDOF solver function. Must have signature:
        solver(acceleration, time_step, natural_frequency, damping_ratio) -> np.ndarray
        returning absolute_acceleration.
        If None, uses piecewise_exact_solver from solvers module (default: None)

    Returns
    -------
    np.ndarray
        Matched acceleration time history [m/s^2]
    """
    if solver is None:
        solver = piecewise_exact_solver
    
    acceleration = initial_acceleration.copy()
    band_mask = (periods >= period_band[0]) & (periods <= period_band[1])
    periods_in_band = periods[band_mask]
    target_spectrum_in_band = target_spectrum[band_mask]
    arias_intensity_target = arias_intensity(acceleration, time_step) * arias_intensity_max_multiplier

    for iteration in range(max_iterations):
        spectrum = response_spectrum(acceleration, time_step, periods_in_band, damping, solver=solver)
        mismatches = np.abs(target_spectrum_in_band - spectrum) / (target_spectrum_in_band + NUMERICAL_EPS)
        max_mismatch = np.max(mismatches)
        if max_mismatch < tolerance:
            print(f"Converged after {iteration} iterations")
            break

        index_max = np.argmax(np.abs(target_spectrum_in_band - spectrum))
        period_max = periods_in_band[index_max]
        frequency_max = 1.0 / period_max
        natural_frequency = 2 * pi / period_max

        # Place wavelet near a dominant response of current signal
        absolute_acceleration = solver(acceleration, time_step, natural_frequency, damping)
        peak_index = np.argmax(np.abs(absolute_acceleration))
        target_time = time[peak_index]
        response_sign = np.sign(absolute_acceleration[peak_index])
        response_delta_max = (target_spectrum_in_band[index_max] - spectrum[index_max]) * response_sign

        wavelet, max_response_amplitude, wavelet_response_sign = tapered_cosine_wavelet(
            time, target_time, frequency_max, time_step, damping_ratio=damping, solver=solver
        )
        wavelet_amplitude = relaxation_factor * response_delta_max * (wavelet_response_sign / (max_response_amplitude + NUMERICAL_EPS))

        # Constrain wavelet_amplitude with an AI cap
        sum_acceleration_squared = np.sum(acceleration ** 2)
        sum_2acceleration_wavelet = np.sum(2 * acceleration * wavelet)
        sum_wavelet_squared = np.sum(wavelet ** 2)

        arias_constant = pi / (2 * GRAVITY) * time_step
        arias_intensity_current = arias_constant * sum_acceleration_squared
        arias_intensity_max = arias_intensity_target

        quadratic_coeff_a = arias_constant * sum_wavelet_squared
        quadratic_coeff_b = arias_constant * sum_2acceleration_wavelet
        quadratic_coeff_c = arias_intensity_current - arias_intensity_max

        if quadratic_coeff_a > 0 and (quadratic_coeff_b ** 2 - 4 * quadratic_coeff_a * quadratic_coeff_c) >= 0:
            discriminant = sqrt(quadratic_coeff_b ** 2 - 4 * quadratic_coeff_a * quadratic_coeff_c)
            amplitude_bound_1 = (-quadratic_coeff_b - discriminant) / (2 * quadratic_coeff_a)
            amplitude_bound_2 = (-quadratic_coeff_b + discriminant) / (2 * quadratic_coeff_a)
            if amplitude_bound_1 > amplitude_bound_2:
                amplitude_bound_1, amplitude_bound_2 = amplitude_bound_2, amplitude_bound_1
            if wavelet_amplitude < amplitude_bound_1 or wavelet_amplitude > amplitude_bound_2:
                # Clip to the closest feasible bound
                wavelet_amplitude = amplitude_bound_1 if abs(wavelet_amplitude - amplitude_bound_1) < abs(wavelet_amplitude - amplitude_bound_2) else amplitude_bound_2
                print(f"Iter {iteration}: Clipped wavelet amplitude to {wavelet_amplitude:.6e} to constrain AI")

        acceleration += wavelet_amplitude * wavelet

    return acceleration

