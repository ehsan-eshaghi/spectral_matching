# Spectral Matching

A Python package for matching earthquake ground motion acceleration records to target response spectra using iterative FFT and Greedy Wavelet Matching (GWM) methods.

## Features

- **Iterative FFT Matching**: Frequency-domain spectral matching using iterative FFT adjustments
- **Greedy Wavelet Matching**: Time-domain matching using tapered cosine wavelets
- **Response Spectrum Computation**: Accurate piecewise-exact SDOF solver
- **Intensity Metrics**: Arias Intensity (AI) and Cumulative Absolute Velocity (CAV)
- **Visualization**: Comprehensive plotting tools for spectra, time histories, and cumulative metrics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spectral_matching
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
spectral_matching/
├── src/
│   ├── spectral_matching/      # Main package
│   │   ├── __init__.py         # Package initialization
│   │   ├── constants.py        # Physical and numerical constants
│   │   ├── solvers.py          # SDOF solver and response spectrum
│   │   ├── metrics.py          # Earthquake intensity metrics
│   │   ├── matching.py         # FFT and GWM matching algorithms
│   │   ├── io.py               # File I/O operations
│   │   ├── preprocessing.py    # Data preprocessing functions
│   │   └── plotting.py         # Plotting utilities
│   ├── main.py                 # Main execution script
│   ├── data/                   # Input data files
│   │   ├── elcentro_NS.dat.txt
│   │   └── uhs_el_centro.csv
│   ├── figures/                # Generated plots
│   └── output/                 # Matched acceleration records
├── requirements.txt
└── README.md
```

## Usage

### Basic Usage

Run the main script:

```bash
cd src
python main.py
```

This will:
1. Load the acceleration record from `data/elcentro_NS.dat.txt`
2. Load the target spectrum from `data/uhs_el_centro.csv`
3. Perform baseline correction and scaling
4. Apply both FFT and GWM matching methods
5. Generate plots and save matched records

### Using as a Package

```python
from spectral_matching import (
    load_acceleration_record,
    load_target_spectrum,
    iterative_fft_match,
    response_spectrum,
    plot_spectra
)

# Load data
time, acc, dt = load_acceleration_record("data/record.dat.txt")
periods, target_spectrum = load_target_spectrum("data/target.csv")

# Match spectrum
acc_matched = iterative_fft_match(acc, dt, periods, target_spectrum)

# Compute response spectrum
spectrum = response_spectrum(acc_matched, dt, periods)
```

## Configuration

Key parameters can be adjusted in `src/spectral_matching/constants.py`:

- `DAMPING`: Damping ratio (default: 0.05)
- `TARGET_PERIOD_BAND`: Target period range [T_min, T_max] in seconds
- `FFT_ITERS`: Number of FFT matching iterations
- `GWM_ITERS`: Maximum GWM iterations
- `PERIOD_MIN`, `PERIOD_MAX`, `NUM_PERIODS`: Period grid parameters

## Output

The script generates:

- **Plots** (saved to `figures/`):
  - Response spectra comparison
  - Cumulative AI and CAV plots
  - Time history comparisons

- **Matched Records** (saved to `output/`):
  - `*_matched_iterative.dat.txt`: FFT-matched record
  - `*_matched_gwm.dat.txt`: GWM-matched record

## Methods

### Iterative FFT Matching

Adjusts the frequency content of the acceleration record iteratively to match the target spectrum. Uses FFT-based frequency domain modifications with smoothing and clipping to ensure stability.

### Greedy Wavelet Matching

Adds tapered cosine wavelets to the acceleration record to iteratively improve the match. Constrains the Arias Intensity to prevent unrealistic amplification.

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0

## License

See LICENSE file for details.
