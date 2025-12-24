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
from .solvers import piecewise_exact_history, response_spectrum
from .metrics import arias_intensity


def iterative_fft_match(
    acc_initial: np.ndarray,
    dt: float,
    periods: np.ndarray,
    Se_target: np.ndarray,
    iters: int = FFT_ITERS,
    T_band: list = TARGET_PERIOD_BAND,
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
    acc_initial : np.ndarray
        Initial acceleration time history [m/s^2]
    dt : float
        Time step [s]
    periods : np.ndarray
        Period array [s] for spectrum computation
    Se_target : np.ndarray
        Target spectral acceleration [m/s^2] for each period
    iters : int, optional
        Number of iterations (default: FFT_ITERS)
    T_band : list, optional
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
    acc = acc_initial.copy()
    n = len(acc)
    N = int(2 ** np.ceil(np.log2(n)))

    periods = np.asarray(periods)
    Se_target = np.asarray(Se_target)
    f_centers = 1.0 / periods
    freqs = np.fft.rfftfreq(N, dt)
    f_low, f_high = 1.0 / T_band[1], 1.0 / T_band[0]

    for _ in range(iters):
        Sa_current = response_spectrum(acc, dt, periods, damping=damping)
        ratio = np.clip(
            Se_target / (Sa_current + numerical_stability_eps),
            ratio_clip_min,
            ratio_clip_max
        )

        # Map period-domain gains to frequency grid
        sort_idx = np.argsort(f_centers)
        gain_interp = np.interp(
            freqs, f_centers[sort_idx], ratio[sort_idx],
            left=1.0, right=1.0
        )
        gain = np.ones_like(freqs)
        mask_band = (freqs >= f_low) & (freqs <= f_high)
        gain[mask_band] = gain_interp[mask_band]

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
        acc_pad = np.zeros(N)
        acc_pad[:n] = acc * window
        A = np.fft.rfft(acc_pad)
        acc_new = np.fft.irfft(A * gain)[:n]
        acc = acc_new

    return acc


def tapered_cosine_wavelet(
    t: np.ndarray,
    t_i: float,
    f_i: float,
    dt: float,
    beta: float = DAMPING
) -> Tuple[np.ndarray, float, float]:
    """
    Generate a tapered cosine wavelet with phase alignment.

    Parameters
    ----------
    t : np.ndarray
        Time array [s]
    t_i : float
        Target time for peak response [s]
    f_i : float
        Frequency [Hz]
    dt : float
        Time step [s]
    beta : float, optional
        Damping ratio (default: DAMPING)

    Returns
    -------
    w : np.ndarray
        Wavelet time history
    R_w : float
        Maximum response amplitude
    P_w : float
        Sign of maximum response (+1 or -1)
    """
    omega = 2 * pi * f_i
    omega_prime = omega * sqrt(max(0.0, 1 - beta**2))
    gamma_i = 1.178 * f_i ** (-0.93)

    # Build w with zero phase shift, find system response peak time, then align
    tau = t - t_i
    w_temp = np.cos(omega_prime * tau) * np.exp(- (tau / gamma_i) ** 2)

    wn = omega
    xi = beta
    _, _, _, a_abs_w_temp = piecewise_exact_history(w_temp, dt, wn, xi)

    idx_w = np.argmax(np.abs(a_abs_w_temp))
    t_w = t[idx_w]

    # Recreate with phase shift so response peaks at t_i
    delta_t = t_w - t_i
    tau = t - t_i + delta_t
    w = np.cos(omega_prime * tau) * np.exp(- (tau / gamma_i) ** 2)

    _, _, _, a_abs_w = piecewise_exact_history(w, dt, wn, xi)
    idx_max = np.argmax(np.abs(a_abs_w))
    R_w = np.abs(a_abs_w[idx_max])
    P_w = np.sign(a_abs_w[idx_max])

    return w, R_w, P_w


def greedy_wavelet_match(
    acc_initial: np.ndarray,
    dt: float,
    t: np.ndarray,
    periods: np.ndarray,
    Se_target: np.ndarray,
    damping: float = DAMPING,
    max_iters: int = GWM_ITERS,
    tol: float = GWM_TOL,
    gamma_relax: float = GWM_GAMMA,
    ai_max_multiplier: float = GWM_AI_MULTIPLIER,
    band: list = TARGET_PERIOD_BAND
) -> np.ndarray:
    """
    Greedy Wavelet Matching (GWM) algorithm for spectral matching.

    Iteratively adds tapered cosine wavelets to match the target spectrum
    while constraining the Arias Intensity.

    Parameters
    ----------
    acc_initial : np.ndarray
        Initial acceleration time history [m/s^2]
    dt : float
        Time step [s]
    t : np.ndarray
        Time array [s]
    periods : np.ndarray
        Period array [s] for spectrum computation
    Se_target : np.ndarray
        Target spectral acceleration [m/s^2] for each period
    damping : float, optional
        Damping ratio (default: DAMPING)
    max_iters : int, optional
        Maximum number of iterations (default: GWM_ITERS)
    tol : float, optional
        Convergence tolerance (default: GWM_TOL)
    gamma_relax : float, optional
        Relaxation factor (default: GWM_GAMMA)
    ai_max_multiplier : float, optional
        Maximum AI multiplier relative to initial (default: GWM_AI_MULTIPLIER)
    band : list, optional
        Target period band [T_min, T_max] (default: TARGET_PERIOD_BAND)

    Returns
    -------
    np.ndarray
        Matched acceleration time history [m/s^2]
    """
    acc = acc_initial.copy()
    band_mask_local = (periods >= band[0]) & (periods <= band[1])
    periods_band = periods[band_mask_local]
    Se_target_band = Se_target[band_mask_local]
    ai_target = arias_intensity(acc, dt) * ai_max_multiplier

    for it in range(max_iters):
        Sa = response_spectrum(acc, dt, periods_band, damping)
        mismatches = np.abs(Se_target_band - Sa) / (Se_target_band + NUMERICAL_EPS)
        max_mismatch = np.max(mismatches)
        if max_mismatch < tol:
            print(f"Converged after {it} iterations")
            break

        i_max = np.argmax(np.abs(Se_target_band - Sa))
        T_max = periods_band[i_max]
        f_max = 1.0 / T_max
        wn = 2 * pi / T_max
        xi = damping

        # Place wavelet near a dominant response of current signal
        _, _, _, a_abs = piecewise_exact_history(acc, dt, wn, xi)
        peak_idx = np.argmax(np.abs(a_abs))
        t_i = t[peak_idx]
        P_i = np.sign(a_abs[peak_idx])
        Delta_R_max = (Se_target_band[i_max] - Sa[i_max]) * P_i

        w, R_w, P_w = tapered_cosine_wavelet(t, t_i, f_max, dt, beta=damping)
        b = gamma_relax * Delta_R_max * (P_w / (R_w + NUMERICAL_EPS))

        # Constrain b with an AI cap
        sum_acc2 = np.sum(acc ** 2)
        sum_2acc_w = np.sum(2 * acc * w)
        sum_w2 = np.sum(w ** 2)

        const = pi / (2 * GRAVITY) * dt
        ai_current = const * sum_acc2
        ai_max = ai_target

        a_q = const * sum_w2
        b_q = const * sum_2acc_w
        c_q = ai_current - ai_max

        if a_q > 0 and (b_q ** 2 - 4 * a_q * c_q) >= 0:
            disc = sqrt(b_q ** 2 - 4 * a_q * c_q)
            b1 = (-b_q - disc) / (2 * a_q)
            b2 = (-b_q + disc) / (2 * a_q)
            if b1 > b2:
                b1, b2 = b2, b1
            if b < b1 or b > b2:
                # Clip to the closest feasible bound
                b = b1 if abs(b - b1) < abs(b - b2) else b2
                print(f"Iter {it}: Clipped b to {b:.6e} to constrain AI")

        acc += b * w

    return acc

