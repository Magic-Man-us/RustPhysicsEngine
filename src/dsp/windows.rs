//! Window functions for spectral analysis.
//!
//! Moved from `signal_processing` (Step 0 of roadmap Part 3); the classic
//! generators keep their `w[k] = f(2πk/(n−1))` symmetric convention.

use crate::math::constants::PI;

const TWO_PI: f64 = 2.0 * PI;
const FOUR_PI: f64 = 4.0 * PI;

const HANN_COEFF: f64 = 0.5;
const HAMMING_A0: f64 = 0.54;
const HAMMING_A1: f64 = 0.46;
const BLACKMAN_A0: f64 = 0.42;
const BLACKMAN_A1: f64 = 0.5;
const BLACKMAN_A2: f64 = 0.08;

/// Generate a Hann window of length n: w[k] = 0.5·(1 - cos(2πk/(n-1)))
#[must_use]
pub fn hann_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| HANN_COEFF * (1.0 - (TWO_PI * k as f64 / denom).cos()))
        .collect()
}

/// Generate a Hamming window of length n: w[k] = 0.54 - 0.46·cos(2πk/(n-1))
#[must_use]
pub fn hamming_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| HAMMING_A0 - HAMMING_A1 * (TWO_PI * k as f64 / denom).cos())
        .collect()
}

/// Generate a Blackman window of length n: w[k] = 0.42 - 0.5·cos(2πk/(n-1)) + 0.08·cos(4πk/(n-1))
#[must_use]
pub fn blackman_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| {
            let kf = k as f64;
            BLACKMAN_A0 - BLACKMAN_A1 * (TWO_PI * kf / denom).cos()
                + BLACKMAN_A2 * (FOUR_PI * kf / denom).cos()
        })
        .collect()
}

/// Generate a rectangular (uniform) window of length n: w[k] = 1 for all k
#[must_use]
pub fn rectangular_window(n: usize) -> Vec<f64> {
    vec![1.0; n]
}
