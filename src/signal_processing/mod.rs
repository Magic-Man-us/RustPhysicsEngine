// The FFT moved to `transforms::fft` and the window generators and
// first-order RC filters to `dsp` (Step 0 of roadmap Part 3); everything
// stays importable from its old path here.
pub mod fft {
    //! Re-export of `transforms::fft` at its pre-Part-3 path.
    pub use crate::transforms::fft::*;
}

pub use crate::dsp::iir::{first_order_highpass, first_order_lowpass};
pub use crate::dsp::windows::{
    blackman_window, hamming_window, hann_window, rectangular_window,
};
pub use crate::transforms::fft::{fft, fft_convolve, ifft, next_power_of_two, rfft};

use crate::math::constants::PI;

const TWO_PI: f64 = 2.0 * PI;

// LCG parameters (Numerical Recipes)
const LCG_MULTIPLIER: u64 = 6364136223846793005;
const LCG_INCREMENT: u64 = 1442695040888963407;

// --- Convolution & Correlation ---

/// Linear convolution of signal with kernel: y[n] = Σ s[i]·k[n-i]
#[must_use]
pub fn convolve(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    if signal.is_empty() || kernel.is_empty() {
        return Vec::new();
    }
    let out_len = signal.len() + kernel.len() - 1;
    let mut output = vec![0.0; out_len];
    for (i, &s) in signal.iter().enumerate() {
        for (j, &k) in kernel.iter().enumerate() {
            output[i + j] += s * k;
        }
    }
    output
}

/// Cross-correlation of x and y via convolution with time-reversed y
#[must_use]
pub fn cross_correlate(x: &[f64], y: &[f64]) -> Vec<f64> {
    let reversed: Vec<f64> = y.iter().rev().copied().collect();
    convolve(x, &reversed)
}

/// Auto-correlation of a signal: cross-correlation of the signal with itself
#[must_use]
pub fn auto_correlate(signal: &[f64]) -> Vec<f64> {
    cross_correlate(signal, signal)
}

/// Normalize signal amplitude to [-1, 1] by dividing by peak absolute value
pub fn normalize_signal(signal: &mut [f64]) {
    if signal.is_empty() {
        return;
    }
    let mut max_abs = 0.0_f64;
    for &s in signal.iter() {
        let a = s.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        return;
    }
    let inv = 1.0 / max_abs;
    for s in signal.iter_mut() {
        *s *= inv;
    }
}

// --- Window Functions ---
// (generators now live in `dsp::windows`, re-exported above)

/// Element-wise multiplication of signal by window coefficients
#[must_use]
pub fn apply_window(signal: &[f64], window: &[f64]) -> Vec<f64> {
    signal
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| s * w)
        .collect()
}

// --- Digital Filters ---

/// Simple moving average filter with specified window size
#[must_use]
pub fn moving_average(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.is_empty() || window_size == 0 {
        return Vec::new();
    }
    let ws = window_size.min(signal.len());
    let inv_ws = 1.0 / ws as f64;
    let mut output = Vec::with_capacity(signal.len());
    let mut sum: f64 = signal[..ws].iter().sum();
    // First ws elements use growing window
    for i in 0..ws {
        let count = i + 1;
        let partial_sum: f64 = signal[..count].iter().sum();
        output.push(partial_sum / count as f64);
    }
    // Remaining elements use full sliding window
    for i in ws..signal.len() {
        sum += signal[i] - signal[i - ws];
        output.push(sum * inv_ws);
    }
    output
}

/// Exponential moving average filter: y[n] = α·x[n] + (1-α)·y[n-1]
#[must_use]
pub fn exponential_moving_average(signal: &[f64], alpha: f64) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(signal.len());
    let one_minus_alpha = 1.0 - alpha;
    output.push(signal[0]);
    for i in 1..signal.len() {
        let prev = output[i - 1];
        output.push(alpha * signal[i] + one_minus_alpha * prev);
    }
    output
}

// (first_order_lowpass / first_order_highpass now live in `dsp::iir`,
// re-exported above)

/// Median filter for impulse noise removal with specified window size
#[must_use]
pub fn median_filter(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.is_empty() || window_size == 0 {
        return Vec::new();
    }
    let half = window_size / 2;
    let mut output = Vec::with_capacity(signal.len());
    let mut window_buf = Vec::with_capacity(window_size);
    for i in 0..signal.len() {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(signal.len());
        window_buf.clear();
        window_buf.extend_from_slice(&signal[start..end]);
        window_buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = window_buf.len() / 2;
        let median = if window_buf.len() % 2 == 0 {
            (window_buf[mid - 1] + window_buf[mid]) / 2.0
        } else {
            window_buf[mid]
        };
        output.push(median);
    }
    output
}

// --- Signal Generation ---

/// Generate a sine wave: x[n] = A·sin(2πf·n/fs) for n samples over given duration
#[must_use]
pub fn sine_wave(frequency: f64, sample_rate: f64, duration: f64, amplitude: f64) -> Vec<f64> {
    assert!(sample_rate > 0.0, "sample rate must be positive");
    let n = (sample_rate * duration) as usize;
    let angular_freq = TWO_PI * frequency;
    let inv_rate = 1.0 / sample_rate;
    (0..n)
        .map(|i| amplitude * (angular_freq * i as f64 * inv_rate).sin())
        .collect()
}

/// Generate a square wave: +A for first half-period, -A for second half
#[must_use]
pub fn square_wave(frequency: f64, sample_rate: f64, duration: f64, amplitude: f64) -> Vec<f64> {
    assert!(sample_rate > 0.0, "sample rate must be positive");
    let n = (sample_rate * duration) as usize;
    let inv_rate = 1.0 / sample_rate;
    (0..n)
        .map(|i| {
            let phase = (frequency * i as f64 * inv_rate).fract();
            if phase < 0.5 {
                amplitude
            } else {
                -amplitude
            }
        })
        .collect()
}

/// Generate a sawtooth wave: linearly ramps from -A to +A each period
#[must_use]
pub fn sawtooth_wave(
    frequency: f64,
    sample_rate: f64,
    duration: f64,
    amplitude: f64,
) -> Vec<f64> {
    assert!(sample_rate > 0.0, "sample rate must be positive");
    let n = (sample_rate * duration) as usize;
    let inv_rate = 1.0 / sample_rate;
    (0..n)
        .map(|i| {
            let phase = (frequency * i as f64 * inv_rate).fract();
            amplitude * (2.0 * phase - 1.0)
        })
        .collect()
}

/// Generate deterministic pseudo-random white noise using a linear congruential generator
#[must_use]
pub fn white_noise(n: usize, amplitude: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            // Map to [-1, 1] then scale
            let normalized = (state as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            amplitude * normalized
        })
        .collect()
}

// --- Signal Analysis ---

/// Count the number of zero crossings (sign changes) in the signal
#[must_use]
pub fn zero_crossings(signal: &[f64]) -> usize {
    if signal.len() < 2 {
        return 0;
    }
    signal
        .windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count()
}

/// Root mean square level: RMS = sqrt(Σx²/n)
#[must_use]
pub fn rms_level(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&s| s * s).sum();
    (sum_sq / signal.len() as f64).sqrt()
}

/// Peak-to-peak amplitude: max(x) - min(x)
#[must_use]
pub fn peak_to_peak(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &s in signal {
        if s < min_val {
            min_val = s;
        }
        if s > max_val {
            max_val = s;
        }
    }
    max_val - min_val
}

/// Crest factor: ratio of peak absolute value to RMS level
#[must_use]
pub fn crest_factor(signal: &[f64]) -> f64 {
    let rms = rms_level(signal);
    if rms == 0.0 {
        return 0.0;
    }
    let peak = signal
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f64, f64::max);
    peak / rms
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_rel(a: f64, b: f64, rel: f64) -> bool {
        if a == 0.0 && b == 0.0 {
            return true;
        }
        let denom = a.abs().max(b.abs());
        (a - b).abs() / denom < rel
    }

    // --- Convolution & Correlation ---

    #[test]
    fn test_convolve_impulse() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0];
        let result = convolve(&signal, &kernel);
        assert_eq!(result, signal);
    }

    #[test]
    fn test_convolve_basic() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![0.5, 0.5];
        let result = convolve(&signal, &kernel);
        assert_eq!(result.len(), 4);
        assert!(approx(result[0], 0.5));
        assert!(approx(result[1], 1.5));
        assert!(approx(result[2], 2.5));
        assert!(approx(result[3], 1.5));
    }

    #[test]
    fn test_convolve_empty() {
        assert!(convolve(&[], &[1.0]).is_empty());
        assert!(convolve(&[1.0], &[]).is_empty());
    }

    #[test]
    fn test_cross_correlate_identical() {
        let x = vec![1.0, 0.0, -1.0];
        let result = cross_correlate(&x, &x);
        // Auto-correlation peak should be at the center
        let center = result.len() / 2;
        for i in 0..result.len() {
            if i != center {
                assert!(result[center] >= result[i]);
            }
        }
    }

    #[test]
    fn test_auto_correlate_peak_at_center() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = auto_correlate(&signal);
        let center = result.len() / 2;
        let peak = result[center];
        for &v in &result {
            assert!(v <= peak + TOLERANCE);
        }
    }

    #[test]
    fn test_normalize_signal() {
        let mut signal = vec![2.0, -4.0, 1.0, 3.0];
        normalize_signal(&mut signal);
        assert!(approx(signal[1], -1.0));
        for &s in &signal {
            assert!(s >= -1.0 - TOLERANCE && s <= 1.0 + TOLERANCE);
        }
    }

    #[test]
    fn test_normalize_zero_signal() {
        let mut signal = vec![0.0, 0.0, 0.0];
        normalize_signal(&mut signal);
        assert!(signal.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_normalize_empty() {
        let mut signal: Vec<f64> = Vec::new();
        normalize_signal(&mut signal);
        assert!(signal.is_empty());
    }

    // --- Window Functions ---

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(64);
        assert!(approx(w[0], 0.0));
        assert!(approx(w[63], 0.0));
    }

    #[test]
    fn test_hann_window_center() {
        let n = 65;
        let w = hann_window(n);
        assert!(approx(w[32], 1.0));
    }

    #[test]
    fn test_hann_window_single() {
        assert_eq!(hann_window(1), vec![1.0]);
    }

    #[test]
    fn test_hamming_window_length() {
        let w = hamming_window(128);
        assert_eq!(w.len(), 128);
    }

    #[test]
    fn test_hamming_window_endpoints() {
        let w = hamming_window(64);
        // Hamming endpoints are 0.08, not zero
        assert!(approx(w[0], 0.08));
    }

    #[test]
    fn test_blackman_window_endpoints() {
        let w = blackman_window(64);
        assert!(approx(w[0], 0.0));
        assert!(approx(w[63], 0.0));
    }

    #[test]
    fn test_rectangular_window() {
        let w = rectangular_window(10);
        assert!(w.iter().all(|&v| v == 1.0));
        assert_eq!(w.len(), 10);
    }

    #[test]
    fn test_apply_window() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let window = vec![0.5, 1.0, 1.0, 0.5];
        let result = apply_window(&signal, &window);
        assert!(approx(result[0], 0.5));
        assert!(approx(result[1], 2.0));
        assert!(approx(result[2], 3.0));
        assert!(approx(result[3], 2.0));
    }

    // --- Digital Filters ---

    #[test]
    fn test_moving_average_constant() {
        let signal = vec![5.0; 10];
        let result = moving_average(&signal, 3);
        for &v in &result {
            assert!(approx(v, 5.0));
        }
    }

    #[test]
    fn test_moving_average_smoothing() {
        let signal = vec![0.0, 10.0, 0.0, 10.0, 0.0];
        let result = moving_average(&signal, 3);
        // Smoothed values should have less variance
        let orig_var: f64 = signal.iter().map(|s| (s - 4.0).powi(2)).sum::<f64>() / 5.0;
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        let smooth_var: f64 = result.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / result.len() as f64;
        assert!(smooth_var < orig_var);
    }

    #[test]
    fn test_moving_average_empty() {
        assert!(moving_average(&[], 3).is_empty());
        assert!(moving_average(&[1.0], 0).is_empty());
    }

    #[test]
    fn test_ema_alpha_one() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let result = exponential_moving_average(&signal, 1.0);
        assert_eq!(result, signal);
    }

    #[test]
    fn test_ema_alpha_zero() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let result = exponential_moving_average(&signal, 0.0);
        assert!(result.iter().all(|&v| approx(v, 1.0)));
    }

    #[test]
    fn test_first_order_lowpass_dc() {
        let signal = vec![3.0; 100];
        let result = first_order_lowpass(&signal, 0.01, 0.1);
        assert!(approx(result[99], 3.0));
    }

    #[test]
    fn test_first_order_highpass_dc_rejection() {
        let signal = vec![5.0; 100];
        let result = first_order_highpass(&signal, 0.01, 0.1);
        // Highpass should attenuate DC over time
        assert!(result[99].abs() < result[0].abs());
    }

    #[test]
    fn test_median_filter_impulse_removal() {
        let mut signal = vec![1.0; 11];
        signal[5] = 100.0; // impulse
        let result = median_filter(&signal, 3);
        assert!(approx(result[5], 1.0));
    }

    #[test]
    fn test_median_filter_sorted() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = median_filter(&signal, 3);
        assert!(approx(result[2], 3.0));
    }

    #[test]
    fn test_median_filter_empty() {
        assert!(median_filter(&[], 3).is_empty());
        assert!(median_filter(&[1.0], 0).is_empty());
    }

    // --- Signal Generation ---

    #[test]
    fn test_sine_wave_length() {
        let wave = sine_wave(440.0, 44100.0, 1.0, 1.0);
        assert_eq!(wave.len(), 44100);
    }

    #[test]
    fn test_sine_wave_amplitude() {
        let amp = 2.5;
        let wave = sine_wave(10.0, 1000.0, 1.0, amp);
        let max_val = wave.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(approx_rel(max_val, amp, 0.01));
    }

    #[test]
    fn test_sine_wave_starts_at_zero() {
        let wave = sine_wave(100.0, 44100.0, 0.1, 1.0);
        assert!(approx(wave[0], 0.0));
    }

    #[test]
    fn test_square_wave_values() {
        let wave = square_wave(1.0, 100.0, 1.0, 3.0);
        for &v in &wave {
            assert!(approx(v.abs(), 3.0));
        }
    }

    #[test]
    fn test_square_wave_half_period() {
        let wave = square_wave(1.0, 100.0, 1.0, 1.0);
        // First half positive, second half negative
        assert!(wave[0] > 0.0);
        assert!(wave[50] < 0.0);
    }

    #[test]
    fn test_sawtooth_wave_range() {
        let amp = 2.0;
        let wave = sawtooth_wave(10.0, 1000.0, 1.0, amp);
        for &v in &wave {
            assert!(v >= -amp - TOLERANCE && v <= amp + TOLERANCE);
        }
    }

    #[test]
    fn test_white_noise_deterministic() {
        let a = white_noise(100, 1.0, 42);
        let b = white_noise(100, 1.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_white_noise_different_seeds() {
        let a = white_noise(100, 1.0, 42);
        let b = white_noise(100, 1.0, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn test_white_noise_amplitude() {
        let amp = 0.5;
        let noise = white_noise(10000, amp, 123);
        let max_val = noise.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(max_val <= amp + TOLERANCE);
    }

    // --- Signal Analysis ---

    #[test]
    fn test_zero_crossings_sine() {
        // A 1Hz sine over 1 second at high sample rate crosses zero ~2 times per period
        let wave = sine_wave(10.0, 10000.0, 1.0, 1.0);
        let crossings = zero_crossings(&wave);
        // 10Hz = ~20 crossings
        assert!((crossings as i64 - 20).abs() <= 1);
    }

    #[test]
    fn test_zero_crossings_constant() {
        let signal = vec![1.0; 100];
        assert_eq!(zero_crossings(&signal), 0);
    }

    #[test]
    fn test_zero_crossings_empty() {
        assert_eq!(zero_crossings(&[]), 0);
        assert_eq!(zero_crossings(&[1.0]), 0);
    }

    #[test]
    fn test_rms_level_dc() {
        let signal = vec![3.0; 100];
        assert!(approx(rms_level(&signal), 3.0));
    }

    #[test]
    fn test_rms_level_sine() {
        // RMS of a sine wave = amplitude / sqrt(2)
        let amp = 1.0;
        let wave = sine_wave(100.0, 44100.0, 1.0, amp);
        let rms = rms_level(&wave);
        assert!(approx_rel(rms, 0.707_106_781_186_547_6, 0.01));
    }

    #[test]
    fn test_rms_level_empty() {
        assert_eq!(rms_level(&[]), 0.0);
    }

    #[test]
    fn test_peak_to_peak_sine() {
        let wave = sine_wave(100.0, 44100.0, 1.0, 5.0);
        let ptp = peak_to_peak(&wave);
        assert!(approx_rel(ptp, 10.0, 0.01));
    }

    #[test]
    fn test_peak_to_peak_constant() {
        let signal = vec![7.0; 50];
        assert!(approx(peak_to_peak(&signal), 0.0));
    }

    #[test]
    fn test_peak_to_peak_empty() {
        assert_eq!(peak_to_peak(&[]), 0.0);
    }

    #[test]
    fn test_crest_factor_sine() {
        // Crest factor of a sine wave = sqrt(2) ≈ 1.414
        let wave = sine_wave(100.0, 44100.0, 1.0, 1.0);
        let cf = crest_factor(&wave);
        assert!(approx_rel(cf, 1.414_213_562_373_095, 0.01));
    }

    #[test]
    fn test_crest_factor_dc() {
        let signal = vec![4.0; 100];
        assert!(approx(crest_factor(&signal), 1.0));
    }

    #[test]
    fn test_crest_factor_empty() {
        assert_eq!(crest_factor(&[]), 0.0);
    }

    #[test]
    fn test_hamming_window_single() {
        let w = hamming_window(1);
        assert_eq!(w.len(), 1);
        assert!(approx(w[0], 1.0));
    }

    #[test]
    fn test_blackman_window_single() {
        let w = blackman_window(1);
        assert_eq!(w.len(), 1);
        assert!(approx(w[0], 1.0));
    }

    #[test]
    fn test_exponential_moving_average_empty() {
        let result = exponential_moving_average(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_first_order_highpass_empty() {
        let result = first_order_highpass(&[], 0.01, 0.1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_approx_rel_both_zero() {
        assert!(approx_rel(0.0, 0.0, 1e-6));
    }
}
