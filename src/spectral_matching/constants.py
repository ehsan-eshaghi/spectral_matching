"""
Physical and numerical constants for spectral matching.
"""

# Physical constants
GRAVITY = 9.80665  # m/s^2

# Numerical tolerances
NUMERICAL_EPS = 1e-12

# FFT matching parameters
FFT_ITERS = 30
FFT_SMOOTH_WIDTH = 15
FFT_RATIO_CLIP_MIN = 0.4
FFT_RATIO_CLIP_MAX = 4.0
FFT_GAIN_CLIP_MIN = 0.3
FFT_GAIN_CLIP_MAX = 4.0
FFT_TUKEY_ALPHA = 0.1

# Period grid
NUM_PERIODS = 300
PERIOD_MIN = 0.05
PERIOD_MAX = 3.0

# Default damping ratio (5%)
DAMPING = 0.05

# Default target period band
TARGET_PERIOD_BAND = [0.2, 1.0]

# Greedy Wavelet Matching parameters
GWM_ITERS = 50
GWM_TOL = 0.01
GWM_GAMMA = 1.0
GWM_AI_MULTIPLIER = 1.05  # Allow AI up to 1.05 * initial AI

