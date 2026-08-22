//! Discrete Fourier transform utilities.

use crate::math::constants::PI;

/// Discrete Fourier Transform: X[k] = Σ x[n]·e^(-j2πkn/N), returns (real, imag) pairs
pub fn dft(signal: &[f64]) -> Vec<(f64, f64)> {
    let n = signal.len();
    (0..n)
        .map(|k| {
            let mut real = 0.0;
            let mut imag = 0.0;
            for (idx, &sample) in signal.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * idx as f64 / n as f64;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            (real, imag)
        })
        .collect()
}

/// Inverse DFT: x[n] = (1/N)·Σ X[k]·e^(j2πkn/N)
pub fn inverse_dft(spectrum: &[(f64, f64)]) -> Vec<f64> {
    let n = spectrum.len();
    let inv_n = 1.0 / n as f64;
    (0..n)
        .map(|idx| {
            let mut sum = 0.0;
            for (k, &(re, im)) in spectrum.iter().enumerate() {
                let angle = 2.0 * PI * k as f64 * idx as f64 / n as f64;
                sum += re * angle.cos() - im * angle.sin();
            }
            sum * inv_n
        })
        .collect()
}

/// Power spectrum: |X[k]|² = Re² + Im² for each frequency bin
pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    dft(signal)
        .iter()
        .map(|&(re, im)| re * re + im * im)
        .collect()
}

/// Find the dominant frequency in a signal: f_peak = k_max · f_s / N
pub fn dominant_frequency(signal: &[f64], sample_rate: f64) -> f64 {
    let ps = power_spectrum(signal);
    let n = ps.len();
    // Only search up to Nyquist (first half)
    let half = n / 2;
    let k_max = (1..=half)
        .max_by(|&a, &b| ps[a].partial_cmp(&ps[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    k_max as f64 * sample_rate / n as f64
}
