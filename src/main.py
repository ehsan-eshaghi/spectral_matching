"""
Main script for spectral matching of earthquake ground motion records.

This script demonstrates both iterative FFT matching and Greedy Wavelet Matching
methods for matching acceleration records to target response spectra.
"""

import numpy as np
from pathlib import Path

from spectral_matching import (
    # Constants
    PERIOD_MIN, PERIOD_MAX, NUM_PERIODS, DAMPING, TARGET_PERIOD_BAND, GRAVITY,
    # I/O
    load_acceleration_record, load_target_spectrum, save_acceleration_record,
    # Preprocessing
    baseline_correction, scale_to_target_band, compute_match_statistics,
    # Solvers
    response_spectrum,
    # Metrics
    arias_intensity, cumulative_absolute_velocity, cumulative_metric,
    # Matching
    iterative_fft_match, greedy_wavelet_match,
    # Plotting
    FigureSaver, plot_spectra, plot_cumulative_metric, plot_time_history,
)


def main():
    """Main execution function."""
    # ============================================================
    # Configuration
    # ============================================================
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("figures")
    OUTPUT_DIR = Path("output")
    
    # Input files
    ACC_FILENAME = DATA_DIR / "elcentro_NS.dat.txt"
    UHS_FILENAME = DATA_DIR / "uhs_el_centro.csv"
    
    # Create output directories
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ============================================================
    # Load and preprocess acceleration record
    # ============================================================
    print("Loading acceleration record...")
    time, acceleration, time_step = load_acceleration_record(str(ACC_FILENAME))
    
    # Baseline correction (quadratic detrend)
    acceleration = baseline_correction(acceleration, time, order=2)
    
    # Generate run tag from filename
    run_tag = ACC_FILENAME.stem
    if run_tag.endswith(".dat"):
        run_tag = run_tag[:-4]
    
    # Initialize figure saver
    figure_saver = FigureSaver(
        output_dir=str(FIGURES_DIR),
        run_tag=run_tag,
        period_band=TARGET_PERIOD_BAND,
        damping=DAMPING
    )
    
    # ============================================================
    # Prepare period grid and target spectrum
    # ============================================================
    print("Loading target spectrum...")
    periods = np.linspace(PERIOD_MIN, PERIOD_MAX, NUM_PERIODS)
    periods, target_spectrum = load_target_spectrum(str(UHS_FILENAME), periods)
    
    # ============================================================
    # Scale record to target band
    # ============================================================
    print("Scaling record to target band...")
    acceleration_scaled, scale_factor = scale_to_target_band(
        acceleration, time_step, periods, target_spectrum, band=TARGET_PERIOD_BAND, damping=DAMPING
    )
    
    # Compute original and scaled spectra
    spectrum_original = response_spectrum(acceleration, time_step, periods, damping=DAMPING)
    spectrum_scaled = response_spectrum(acceleration_scaled, time_step, periods, damping=DAMPING)
    
    # Baseline matching statistics
    match_percentage_scaled = compute_match_statistics(
        spectrum_scaled, target_spectrum, periods, band=TARGET_PERIOD_BAND
    )
    print(f"Match % after scaling: {match_percentage_scaled:.1f}%")
    
    # ============================================================
    # Iterative FFT Matching
    # ============================================================
    print("\nPerforming iterative FFT matching...")
    acceleration_matched_fft = iterative_fft_match(
        acceleration_scaled, time_step, periods, target_spectrum, damping=DAMPING
    )
    spectrum_matched_fft = response_spectrum(acceleration_matched_fft, time_step, periods, damping=DAMPING)
    
    # Matching statistics
    match_percentage_fft = compute_match_statistics(
        spectrum_matched_fft, target_spectrum, periods, band=TARGET_PERIOD_BAND
    )
    print(f"Match % after FFT matching: {match_percentage_fft:.1f}%")
    
    # ============================================================
    # Greedy Wavelet Matching
    # ============================================================
    print("\nPerforming Greedy Wavelet Matching...")
    acceleration_matched_gwm = greedy_wavelet_match(
        acceleration_scaled, time_step, time, periods, target_spectrum, damping=DAMPING
    )
    spectrum_matched_gwm = response_spectrum(acceleration_matched_gwm, time_step, periods, damping=DAMPING)
    
    # Matching statistics
    match_percentage_gwm = compute_match_statistics(
        spectrum_matched_gwm, target_spectrum, periods, band=TARGET_PERIOD_BAND
    )
    print(f"Match % after GWM matching: {match_percentage_gwm:.1f}%")
    
    # ============================================================
    # Compute and print metrics
    # ============================================================
    print("\n" + "="*60)
    print("Earthquake Intensity Metrics")
    print("="*60)
    
    # Original metrics
    arias_intensity_original = arias_intensity(acceleration, time_step)
    cumulative_absolute_velocity_original = cumulative_absolute_velocity(acceleration, time_step)
    
    # Scaled metrics
    arias_intensity_scaled = arias_intensity(acceleration_scaled, time_step)
    cumulative_absolute_velocity_scaled = cumulative_absolute_velocity(acceleration_scaled, time_step)
    
    # FFT matched metrics
    arias_intensity_matched_fft = arias_intensity(acceleration_matched_fft, time_step)
    cumulative_absolute_velocity_matched_fft = cumulative_absolute_velocity(acceleration_matched_fft, time_step)
    
    # GWM matched metrics
    arias_intensity_matched_gwm = arias_intensity(acceleration_matched_gwm, time_step)
    cumulative_absolute_velocity_matched_gwm = cumulative_absolute_velocity(acceleration_matched_gwm, time_step)
    
    print(f"\nArias Intensity (AI) [m/s]:")
    print(f"  Original: {arias_intensity_original:.4f}")
    print(f"  Scaled:   {arias_intensity_scaled:.4f}")
    print(f"  FFT:      {arias_intensity_matched_fft:.4f}")
    print(f"  GWM:      {arias_intensity_matched_gwm:.4f}")
    
    print(f"\nCumulative Absolute Velocity (CAV) [m/s]:")
    print(f"  Original: {cumulative_absolute_velocity_original:.4f}")
    print(f"  Scaled:   {cumulative_absolute_velocity_scaled:.4f}")
    print(f"  FFT:      {cumulative_absolute_velocity_matched_fft:.4f}")
    print(f"  GWM:      {cumulative_absolute_velocity_matched_gwm:.4f}")
    
    # ============================================================
    # Plotting - FFT Results
    # ============================================================
    print("\nGenerating plots for FFT matching...")
    
    # Spectra plot
    figure = plot_spectra(
        periods, target_spectrum, spectrum_original,
        spectrum_scaled=spectrum_scaled, spectrum_matched=spectrum_matched_fft, method="FFT"
    )
    figure_saver.save("spectra_iterfft")
    figure.show()
    
    # Cumulative AI
    cumulative_ai_original = cumulative_metric(acceleration, time_step, 'AI')
    cumulative_ai_matched = cumulative_metric(acceleration_matched_fft, time_step, 'AI')
    figure = plot_cumulative_metric(time, cumulative_ai_original, cumulative_ai_matched, metric='AI')
    figure_saver.save("cumulative_AI_iterfft")
    figure.show()
    
    # Cumulative CAV
    cumulative_cav_original = cumulative_metric(acceleration, time_step, 'CAV')
    cumulative_cav_matched = cumulative_metric(acceleration_matched_fft, time_step, 'CAV')
    figure = plot_cumulative_metric(time, cumulative_cav_original, cumulative_cav_matched, metric='CAV')
    figure_saver.save("cumulative_CAV_iterfft")
    figure.show()
    
    # Time history
    figure = plot_time_history(time, acceleration, acceleration_matched_fft, method="FFT")
    figure_saver.save("time_history_iterfft")
    figure.show()
    
    # ============================================================
    # Plotting - GWM Results
    # ============================================================
    print("Generating plots for GWM matching...")
    
    # Spectra plot
    figure = plot_spectra(
        periods, target_spectrum, spectrum_original,
        spectrum_scaled=spectrum_scaled, spectrum_matched=spectrum_matched_gwm, method="GWM"
    )
    figure_saver.save("spectra_gwm")
    figure.show()
    
    # Time history
    figure = plot_time_history(time, acceleration, acceleration_matched_gwm, method="GWM")
    figure_saver.save("time_history_gwm")
    figure.show()
    
    # ============================================================
    # Save matched records
    # ============================================================
    print("\nSaving matched records...")
    
    # FFT matched
    output_fft = OUTPUT_DIR / f"{run_tag}_matched_iterative.dat.txt"
    save_acceleration_record(
        str(output_fft),
        time, acceleration_matched_fft,
        header="time(s) acc(g) matched FFT"
    )
    print(f"  Saved: {output_fft}")
    
    # GWM matched
    output_gwm = OUTPUT_DIR / f"{run_tag}_matched_gwm.dat.txt"
    save_acceleration_record(
        str(output_gwm),
        time, acceleration_matched_gwm,
        header="time(s) acc(g) matched GWM"
    )
    print(f"  Saved: {output_gwm}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
