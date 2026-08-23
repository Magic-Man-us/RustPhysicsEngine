//! Infinite impulse response filters.
//!
//! The first-order RC filters moved here from `signal_processing`
//! (Step 0 of roadmap Part 3); the old paths re-export them.

use crate::signal_processing::exponential_moving_average;

/// First-order RC low-pass filter: α = dt / (RC + dt)
///
/// # Panics
/// Panics if `dt <= 0` or `rc < 0`.
#[must_use]
pub fn first_order_lowpass(signal: &[f64], dt: f64, rc: f64) -> Vec<f64> {
    assert!(dt > 0.0, "time step dt must be positive");
    assert!(rc >= 0.0, "RC time constant must be non-negative");
    let alpha = dt / (rc + dt);
    exponential_moving_average(signal, alpha)
}

/// First-order RC high-pass filter: α = RC / (RC + dt)
///
/// # Panics
/// Panics if `dt <= 0` or `rc < 0`.
#[must_use]
pub fn first_order_highpass(signal: &[f64], dt: f64, rc: f64) -> Vec<f64> {
    assert!(dt > 0.0, "time step dt must be positive");
    assert!(rc >= 0.0, "RC time constant must be non-negative");
    if signal.is_empty() {
        return Vec::new();
    }
    let alpha = rc / (rc + dt);
    let mut output = Vec::with_capacity(signal.len());
    output.push(signal[0]);
    for i in 1..signal.len() {
        let prev = output[i - 1];
        output.push(alpha * (prev + signal[i] - signal[i - 1]));
    }
    output
}
