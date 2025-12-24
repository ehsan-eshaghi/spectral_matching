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
    time, acc, dt = load_acceleration_record(str(ACC_FILENAME))
    
    # Baseline correction (quadratic detrend)
    acc = baseline_correction(acc, time, order=2)
    
    # Generate run tag from filename
    run_tag = ACC_FILENAME.stem
    if run_tag.endswith(".dat"):
        run_tag = run_tag[:-4]
    
    # Initialize figure saver
    fig_saver = FigureSaver(
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
    periods, Se_target = load_target_spectrum(str(UHS_FILENAME), periods)
    
    # ============================================================
    # Scale record to target band
    # ============================================================
    print("Scaling record to target band...")
    acc_scaled, scale_factor = scale_to_target_band(
        acc, dt, periods, Se_target, band=TARGET_PERIOD_BAND, damping=DAMPING
    )
    
    # Compute original and scaled spectra
    Sa_orig = response_spectrum(acc, dt, periods, damping=DAMPING)
    Sa_scaled = response_spectrum(acc_scaled, dt, periods, damping=DAMPING)
    
    # Baseline matching statistics
    pct_scaled = compute_match_statistics(
        Sa_scaled, Se_target, periods, band=TARGET_PERIOD_BAND
    )
    print(f"Match % after scaling: {pct_scaled:.1f}%")
    
    # ============================================================
    # Iterative FFT Matching
    # ============================================================
    print("\nPerforming iterative FFT matching...")
    acc_matched_fft = iterative_fft_match(
        acc_scaled, dt, periods, Se_target, damping=DAMPING
    )
    Sa_matched_fft = response_spectrum(acc_matched_fft, dt, periods, damping=DAMPING)
    
    # Matching statistics
    pct_matched_fft = compute_match_statistics(
        Sa_matched_fft, Se_target, periods, band=TARGET_PERIOD_BAND
    )
    print(f"Match % after FFT matching: {pct_matched_fft:.1f}%")
    
    # ============================================================
    # Greedy Wavelet Matching
    # ============================================================
    print("\nPerforming Greedy Wavelet Matching...")
    acc_matched_gwm = greedy_wavelet_match(
        acc_scaled, dt, time, periods, Se_target, damping=DAMPING
    )
    Sa_matched_gwm = response_spectrum(acc_matched_gwm, dt, periods, damping=DAMPING)
    
    # Matching statistics
    pct_matched_gwm = compute_match_statistics(
        Sa_matched_gwm, Se_target, periods, band=TARGET_PERIOD_BAND
    )
    print(f"Match % after GWM matching: {pct_matched_gwm:.1f}%")
    
    # ============================================================
    # Compute and print metrics
    # ============================================================
    print("\n" + "="*60)
    print("Earthquake Intensity Metrics")
    print("="*60)
    
    # Original metrics
    ai_orig = arias_intensity(acc, dt)
    cav_orig = cumulative_absolute_velocity(acc, dt)
    
    # Scaled metrics
    ai_scaled = arias_intensity(acc_scaled, dt)
    cav_scaled = cumulative_absolute_velocity(acc_scaled, dt)
    
    # FFT matched metrics
    ai_matched_fft = arias_intensity(acc_matched_fft, dt)
    cav_matched_fft = cumulative_absolute_velocity(acc_matched_fft, dt)
    
    # GWM matched metrics
    ai_matched_gwm = arias_intensity(acc_matched_gwm, dt)
    cav_matched_gwm = cumulative_absolute_velocity(acc_matched_gwm, dt)
    
    print(f"\nArias Intensity (AI) [m/s]:")
    print(f"  Original: {ai_orig:.4f}")
    print(f"  Scaled:   {ai_scaled:.4f}")
    print(f"  FFT:      {ai_matched_fft:.4f}")
    print(f"  GWM:      {ai_matched_gwm:.4f}")
    
    print(f"\nCumulative Absolute Velocity (CAV) [m/s]:")
    print(f"  Original: {cav_orig:.4f}")
    print(f"  Scaled:   {cav_scaled:.4f}")
    print(f"  FFT:      {cav_matched_fft:.4f}")
    print(f"  GWM:      {cav_matched_gwm:.4f}")
    
    # ============================================================
    # Plotting - FFT Results
    # ============================================================
    print("\nGenerating plots for FFT matching...")
    
    # Spectra plot
    fig = plot_spectra(
        periods, Se_target, Sa_orig,
        Sa_scaled=Sa_scaled, Sa_matched=Sa_matched_fft, method="FFT"
    )
    fig_saver.save("spectra_iterfft")
    fig.show()
    
    # Cumulative AI
    cum_ai_orig = cumulative_metric(acc, dt, 'AI')
    cum_ai_matched = cumulative_metric(acc_matched_fft, dt, 'AI')
    fig = plot_cumulative_metric(time, cum_ai_orig, cum_ai_matched, metric='AI')
    fig_saver.save("cumulative_AI_iterfft")
    fig.show()
    
    # Cumulative CAV
    cum_cav_orig = cumulative_metric(acc, dt, 'CAV')
    cum_cav_matched = cumulative_metric(acc_matched_fft, dt, 'CAV')
    fig = plot_cumulative_metric(time, cum_cav_orig, cum_cav_matched, metric='CAV')
    fig_saver.save("cumulative_CAV_iterfft")
    fig.show()
    
    # Time history
    fig = plot_time_history(time, acc, acc_matched_fft, method="FFT")
    fig_saver.save("time_history_iterfft")
    fig.show()
    
    # ============================================================
    # Plotting - GWM Results
    # ============================================================
    print("Generating plots for GWM matching...")
    
    # Spectra plot
    fig = plot_spectra(
        periods, Se_target, Sa_orig,
        Sa_scaled=Sa_scaled, Sa_matched=Sa_matched_gwm, method="GWM"
    )
    fig_saver.save("spectra_gwm")
    fig.show()
    
    # Time history
    fig = plot_time_history(time, acc, acc_matched_gwm, method="GWM")
    fig_saver.save("time_history_gwm")
    fig.show()
    
    # ============================================================
    # Save matched records
    # ============================================================
    print("\nSaving matched records...")
    
    # FFT matched
    output_fft = OUTPUT_DIR / f"{run_tag}_matched_iterative.dat.txt"
    save_acceleration_record(
        str(output_fft),
        time, acc_matched_fft,
        header="time(s) acc(g) matched FFT"
    )
    print(f"  Saved: {output_fft}")
    
    # GWM matched
    output_gwm = OUTPUT_DIR / f"{run_tag}_matched_gwm.dat.txt"
    save_acceleration_record(
        str(output_gwm),
        time, acc_matched_gwm,
        header="time(s) acc(g) matched GWM"
    )
    print(f"  Saved: {output_gwm}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
