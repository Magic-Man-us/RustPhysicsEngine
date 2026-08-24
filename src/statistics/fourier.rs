//! Discrete Fourier transform utilities.
//!
//! These are thin wrappers over `transforms::fft` (Step 0 of roadmap
//! Part 3): every length now runs in O(n log n) via the mixed-radix /
//! Bluestein FFT while keeping the original `(re, im)` tuple API.

use crate::fractals::Complex;
use crate::transforms::fft::{fft_any, ifft_any, rfft};

/// Discrete Fourier Transform: X[k] = Σ x[n]·e^(-j2πkn/N), returns (real, imag) pairs
#[must_use]
pub fn dft(signal: &[f64]) -> Vec<(f64, f64)> {
    let buf: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_any(&buf).iter().map(|c| (c.re, c.im)).collect()
}

/// Inverse DFT: x[n] = (1/N)·Σ X[k]·e^(j2πkn/N)
#[must_use]
pub fn inverse_dft(spectrum: &[(f64, f64)]) -> Vec<f64> {
    let buf: Vec<Complex> = spectrum.iter().map(|&(re, im)| Complex::new(re, im)).collect();
    ifft_any(&buf).iter().map(|c| c.re).collect()
}

/// Power spectrum: |X[k]|² = Re² + Im² for each frequency bin.
///
/// Uses the real FFT and reconstructs the upper half from conjugate
/// symmetry. Output length always equals `signal.len()`.
#[must_use]
pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let half = rfft(signal);
    let mut ps = vec![0.0; n];
    for (k, c) in half.iter().enumerate() {
        ps[k] = c.norm_sq();
        if k > 0 && k < n - k {
            ps[n - k] = ps[k]; // |X[n-k]| = |X[k]| for real input
        }
    }
    ps
}

/// Find the dominant frequency in a signal: f_peak = k_max · f_s / N
#[must_use]
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
