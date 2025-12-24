"""
Physical and numerical constants for spectral matching.

Constants are loaded from config.ini file. The config file should be located
in the project root directory or in the same directory as this module.
"""

import configparser
from pathlib import Path
from typing import List

# Find config file - check project root first, then module directory
_config_path = None
_project_root = Path(__file__).parent.parent.parent
_module_dir = Path(__file__).parent

if (_project_root / "config.ini").exists():
    _config_path = _project_root / "config.ini"
elif (_module_dir / "config.ini").exists():
    _config_path = _module_dir / "config.ini"
else:
    # Fallback: use default config in project root
    _config_path = _project_root / "config.ini"

# Read configuration
_config = configparser.ConfigParser()
if _config_path.exists():
    _config.read(_config_path)
else:
    # If config file doesn't exist, use defaults
    _config.read_dict({
        'Physical': {'gravity': '9.80665'},
        'Numerical': {'numerical_eps': '1e-12'},
        'FFT_Matching': {
            'fft_iters': '30',
            'fft_smooth_width': '15',
            'fft_ratio_clip_min': '0.4',
            'fft_ratio_clip_max': '4.0',
            'fft_gain_clip_min': '0.3',
            'fft_gain_clip_max': '4.0',
            'fft_tukey_alpha': '0.1'
        },
        'Period_Grid': {
            'num_periods': '300',
            'period_min': '0.05',
            'period_max': '3.0'
        },
        'Default': {
            'damping': '0.05',
            'target_period_band_min': '0.2',
            'target_period_band_max': '1.0'
        },
        'GWM': {
            'gwm_iters': '50',
            'gwm_tol': '0.01',
            'gwm_gamma': '1.0',
            'gwm_ai_multiplier': '1.05'
        }
    })

# Physical constants
GRAVITY = float(_config.get('Physical', 'gravity'))

# Numerical tolerances
NUMERICAL_EPS = float(_config.get('Numerical', 'numerical_eps'))

# FFT matching parameters
FFT_ITERS = int(_config.get('FFT_Matching', 'fft_iters'))
FFT_SMOOTH_WIDTH = int(_config.get('FFT_Matching', 'fft_smooth_width'))
FFT_RATIO_CLIP_MIN = float(_config.get('FFT_Matching', 'fft_ratio_clip_min'))
FFT_RATIO_CLIP_MAX = float(_config.get('FFT_Matching', 'fft_ratio_clip_max'))
FFT_GAIN_CLIP_MIN = float(_config.get('FFT_Matching', 'fft_gain_clip_min'))
FFT_GAIN_CLIP_MAX = float(_config.get('FFT_Matching', 'fft_gain_clip_max'))
FFT_TUKEY_ALPHA = float(_config.get('FFT_Matching', 'fft_tukey_alpha'))

# Period grid
NUM_PERIODS = int(_config.get('Period_Grid', 'num_periods'))
PERIOD_MIN = float(_config.get('Period_Grid', 'period_min'))
PERIOD_MAX = float(_config.get('Period_Grid', 'period_max'))

# Default damping ratio (5%)
DAMPING = float(_config.get('Default', 'damping'))

# Default target period band
TARGET_PERIOD_BAND: List[float] = [
    float(_config.get('Default', 'target_period_band_min')),
    float(_config.get('Default', 'target_period_band_max'))
]

# Greedy Wavelet Matching parameters
GWM_ITERS = int(_config.get('GWM', 'gwm_iters'))
GWM_TOL = float(_config.get('GWM', 'gwm_tol'))
GWM_GAMMA = float(_config.get('GWM', 'gwm_gamma'))
GWM_AI_MULTIPLIER = float(_config.get('GWM', 'gwm_ai_multiplier'))

