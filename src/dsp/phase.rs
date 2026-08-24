//! Phase utilities: unwrapping (1D and Itoh 2D), phase-locked loops,
//! interpolated zero crossings, and phase measurement against a
//! reference tone.

use crate::math::constants::PI;

const TWO_PI: f64 = 2.0 * PI;

/// Wrap an angle into (−π, π].
#[must_use]
pub fn wrap_phase(p: f64) -> f64 {
    let mut w = (p + PI).rem_euclid(TWO_PI) - PI;
    if w <= -PI {
        w += TWO_PI;
    }
    w
}

/// 1D phase unwrapping: remove 2π jumps between consecutive samples.
#[must_use]
pub fn unwrap_phase(p: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(p.len());
    let mut offset = 0.0;
    for (i, &v) in p.iter().enumerate() {
        if i > 0 {
            let d = v - p[i - 1];
            if d > PI {
                offset -= TWO_PI;
            } else if d < -PI {
                offset += TWO_PI;
            }
        }
        out.push(v + offset);
    }
    out
}

/// 2D phase unwrapping by Itoh's method: unwrap each row, then unwrap
/// the columns of the row-unwrapped field. Exact for residue-free
/// (consistent) phase maps.
///
/// # Panics
/// Panics unless `p.len() == w * h`.
#[must_use]
pub fn unwrap_phase_2d(p: &[f64], w: usize, h: usize) -> Vec<f64> {
    assert_eq!(p.len(), w * h, "unwrap_phase_2d expects w*h samples");
    let mut out = vec![0.0; w * h];
    // Rows.
    for y in 0..h {
        let row = unwrap_phase(&p[y * w..(y + 1) * w]);
        out[y * w..(y + 1) * w].copy_from_slice(&row);
    }
    // Columns: unwrap the column of already row-unwrapped values by
    // adjusting whole rows below each jump.
    let mut col = vec![0.0; h];
    for x in 0..w {
        for (y, c) in col.iter_mut().enumerate() {
            *c = out[y * w + x];
        }
        let un = unwrap_phase(&col);
        for y in 0..h {
            let delta = un[y] - col[y];
            out[y * w + x] += delta;
        }
    }
    out
}

/// Wrapped per-sample phase difference a − b.
///
/// # Panics
/// Panics if the lengths differ.
#[must_use]
pub fn phase_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "phase arrays must match");
    a.iter().zip(b).map(|(&x, &y)| wrap_phase(x - y)).collect()
}

/// Group delay −dφ/dω from unwrapped phase samples on an angular
/// frequency grid (central differences; one-sided at the ends).
///
/// # Panics
/// Panics if the lengths differ or fewer than 2 points.
#[must_use]
pub fn group_delay_from_phase(phase: &[f64], freqs: &[f64]) -> Vec<f64> {
    assert_eq!(phase.len(), freqs.len(), "phase and freqs must match");
    assert!(phase.len() >= 2, "need at least two points");
    let n = phase.len();
    let un = unwrap_phase(phase);
    (0..n)
        .map(|i| {
            let (i0, i1) = if i == 0 {
                (0, 1)
            } else if i == n - 1 {
                (n - 2, n - 1)
            } else {
                (i - 1, i + 1)
            };
            -(un[i1] - un[i0]) / (freqs[i1] - freqs[i0])
        })
        .collect()
}

/// Second-order phase-locked loop tracking a real tone near f0:
/// returns the NCO phase track and the instantaneous frequency estimate
/// (Hz) per sample. `bandwidth` is the loop bandwidth in Hz.
///
/// # Panics
/// Panics unless the rates are positive.
#[must_use]
pub fn phase_locked_loop(x: &[f64], fs: f64, f0: f64, bandwidth: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(fs > 0.0 && bandwidth > 0.0, "rates must be positive");
    let wn = TWO_PI * bandwidth / fs; // natural frequency per sample
    let zeta = std::f64::consts::FRAC_1_SQRT_2;
    let kp = 2.0 * zeta * wn;
    let ki = wn * wn;
    let mut theta = 0.0_f64;
    let mut freq_offset = 0.0_f64; // rad/sample beyond f0
    let base = TWO_PI * f0 / fs;
    let mut phases = Vec::with_capacity(x.len());
    let mut freqs = Vec::with_capacity(x.len());
    for &v in x {
        // Phase detector: mix input with the NCO quadrature. For x =
        // sin(φ), −cos(θ)·x ~ ... use e = x·cos(θ) which averages to
        // ½ sin(φ − θ).
        let err = v * theta.cos() * 2.0;
        freq_offset += ki * err;
        let inst = base + freq_offset + kp * err;
        theta += inst;
        phases.push(theta);
        freqs.push(inst * fs / TWO_PI);
    }
    (phases, freqs)
}

/// Linearly interpolated zero-crossing times (seconds), both directions.
#[must_use]
pub fn zero_crossing_times(x: &[f64], fs: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 1..x.len() {
        let (a, b) = (x[i - 1], x[i]);
        if a == 0.0 {
            out.push((i - 1) as f64 / fs);
        } else if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
            let frac = a / (a - b);
            out.push((i as f64 - 1.0 + frac) / fs);
        }
    }
    out
}

/// Phase (radians) of the signal's component at `ref_freq` relative to
/// cos(2π·f·t) starting at the first sample, via single-bin correlation.
#[must_use]
pub fn phase_vs_reference(x: &[f64], ref_freq: f64, fs: f64) -> f64 {
    let mut re = 0.0;
    let mut im = 0.0;
    for (i, &v) in x.iter().enumerate() {
        let ang = TWO_PI * ref_freq * i as f64 / fs;
        re += v * ang.cos();
        im -= v * ang.sin();
    }
    im.atan2(re)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(3.0 * PI) - PI).abs() < 1e-12);
        assert!((wrap_phase(-3.0 * PI) - PI).abs() < 1e-12);
        assert!((wrap_phase(0.3) - 0.3).abs() < 1e-15);
        assert!((wrap_phase(-0.3) + 0.3).abs() < 1e-15);
    }

    #[test]
    fn test_unwrap_linear_ramp() {
        // A wrapped linear ramp unwraps back to a straight line.
        let slope = 0.4;
        let wrapped: Vec<f64> = (0..500).map(|i| wrap_phase(slope * i as f64)).collect();
        let un = unwrap_phase(&wrapped);
        for (i, v) in un.iter().enumerate() {
            assert!((v - slope * i as f64).abs() < 1e-9, "at {i}");
        }
    }

    #[test]
    fn test_unwrap_2d_plane() {
        let (w, h) = (32, 24);
        let plane = |x: usize, y: usize| 0.3 * x as f64 + 0.5 * y as f64;
        let wrapped: Vec<f64> = (0..w * h)
            .map(|i| wrap_phase(plane(i % w, i / w)))
            .collect();
        let un = unwrap_phase_2d(&wrapped, w, h);
        // Same gradient everywhere; offset fixed by the first sample.
        let offset = un[0] - plane(0, 0);
        for y in 0..h {
            for x in 0..w {
                assert!(
                    (un[y * w + x] - plane(x, y) - offset).abs() < 1e-9,
                    "at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_phase_difference_wraps() {
        let a = [3.0, -3.0];
        let b = [-3.0, 3.0];
        let d = phase_difference(&a, &b);
        assert!((d[0] - (6.0 - TWO_PI)).abs() < 1e-12);
        assert!((d[1] - (TWO_PI - 6.0)).abs() < 1e-12);
    }

    #[test]
    fn test_group_delay_from_linear_phase() {
        // φ = −τω: constant group delay τ.
        let tau = 2.5;
        let freqs: Vec<f64> = (0..100).map(|i| 0.01 * i as f64).collect();
        let phase: Vec<f64> = freqs.iter().map(|&w| wrap_phase(-tau * w)).collect();
        let gd = group_delay_from_phase(&phase, &freqs);
        for v in &gd {
            assert!((v - tau).abs() < 1e-9);
        }
    }

    #[test]
    fn test_pll_locks_to_detuned_tone() {
        let fs = 8000.0;
        let f_true = 1030.0; // detuned from the 1000 Hz center
        let n = 8000; // 1 s ≈ 1000 cycles
        let x: Vec<f64> = (0..n).map(|i| (TWO_PI * f_true * i as f64 / fs).sin()).collect();
        let (_, freqs) = phase_locked_loop(&x, fs, 1000.0, 40.0);
        // Averaged frequency estimate at the tail is the true frequency.
        let cycles_100 = (100.0 * fs / f_true) as usize; // samples in 100 cycles
        let tail = &freqs[cycles_100..cycles_100 + 2000];
        let mean: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!((mean - f_true).abs() < 1.0, "locked at {mean}");
    }

    #[test]
    fn test_zero_crossing_times() {
        let fs = 100.0;
        // sin(2π·5·t): crossings every 0.1 s starting at 0.1 (skip t=0
        // since the first sample is exactly zero → reported at t=0).
        let x: Vec<f64> = (0..200).map(|i| (TWO_PI * 5.0 * i as f64 / fs).sin()).collect();
        let t = zero_crossing_times(&x, fs);
        assert!(!t.is_empty());
        for (k, &tv) in t.iter().enumerate() {
            let expect = k as f64 * 0.1;
            assert!((tv - expect).abs() < 2e-3, "crossing {k}: {tv} vs {expect}");
        }
    }

    #[test]
    fn test_phase_vs_reference() {
        let fs = 1000.0;
        let f0 = 50.0;
        for &phi in &[0.0, 0.7, -1.2, 2.9] {
            let x: Vec<f64> = (0..1000)
                .map(|i| (TWO_PI * f0 * i as f64 / fs + phi).cos())
                .collect();
            let est = phase_vs_reference(&x, f0, fs);
            assert!((wrap_phase(est - phi)).abs() < 1e-6, "phi {phi}: {est}");
        }
    }
}
