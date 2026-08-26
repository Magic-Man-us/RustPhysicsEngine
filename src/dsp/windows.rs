//! Window functions for spectral analysis and FIR design.
//!
//! [`window`] generates any of the standard windows in symmetric form
//! (filter design; endpoints at k = 0 and k = n−1) or periodic form
//! (spectral analysis; the implied period is n). [`window_metrics`]
//! measures the figures of merit from Harris (1978), *On the Use of
//! Windows for Harmonic Analysis with the DFT*.
//!
//! The pre-Part-3 generators (`hann_window`, …) are kept and wrap
//! [`window`] with their original symmetric convention.

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::special::bessel_i0;

const TWO_PI: f64 = 2.0 * PI;

/// Window families for [`window`]. Parameterized variants carry their
/// shape parameter: Kaiser β, Tukey taper fraction α ∈ [0, 1], Gaussian
/// σ (relative to the half-width), Dolph-Chebyshev sidelobe attenuation
/// in (positive) dB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowKind {
    Rect,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Nuttall,
    FlatTop,
    Kaiser(f64),
    Tukey(f64),
    Gaussian(f64),
    DolphChebyshev(f64),
    Bartlett,
    Bohman,
    Lanczos,
}

/// Generate a window of length n. `periodic` selects the DFT-even form
/// (denominator n, for spectral analysis); symmetric windows use
/// denominator n−1 (for FIR design).
#[must_use]
pub fn window(kind: WindowKind, n: usize, periodic: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    if let WindowKind::DolphChebyshev(atten_db) = kind {
        return dolph_chebyshev(n, atten_db, periodic);
    }
    let denom = if periodic { n as f64 } else { (n - 1) as f64 };
    let half = denom / 2.0;
    (0..n)
        .map(|k| {
            let kf = k as f64;
            let t = TWO_PI * kf / denom; // phase in [0, 2π)
            let u = kf / half - 1.0; // position in [-1, 1]
            match kind {
                WindowKind::Rect => 1.0,
                WindowKind::Hann => 0.5 - 0.5 * t.cos(),
                WindowKind::Hamming => 0.54 - 0.46 * t.cos(),
                WindowKind::Blackman => 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos(),
                WindowKind::BlackmanHarris => {
                    0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos()
                        - 0.01168 * (3.0 * t).cos()
                }
                WindowKind::Nuttall => {
                    0.3635819 - 0.4891775 * t.cos() + 0.1365995 * (2.0 * t).cos()
                        - 0.0106411 * (3.0 * t).cos()
                }
                WindowKind::FlatTop => {
                    0.21557895 - 0.41663158 * t.cos() + 0.277263158 * (2.0 * t).cos()
                        - 0.083578947 * (3.0 * t).cos()
                        + 0.006947368 * (4.0 * t).cos()
                }
                WindowKind::Kaiser(beta) => {
                    bessel_i0(beta * (1.0 - u * u).max(0.0).sqrt()) / bessel_i0(beta)
                }
                WindowKind::Tukey(alpha) => {
                    let a = alpha.clamp(0.0, 1.0);
                    if a == 0.0 {
                        1.0
                    } else {
                        let x = kf / denom; // [0, 1]
                        if x < a / 2.0 {
                            0.5 * (1.0 + (TWO_PI * (x / a - 0.5)).cos())
                        } else if x > 1.0 - a / 2.0 {
                            0.5 * (1.0 + (TWO_PI * ((x - 1.0) / a + 0.5)).cos())
                        } else {
                            1.0
                        }
                    }
                }
                WindowKind::Gaussian(sigma) => (-0.5 * (u / sigma).powi(2)).exp(),
                WindowKind::DolphChebyshev(_) => unreachable!(),
                WindowKind::Bartlett => 1.0 - u.abs(),
                WindowKind::Bohman => {
                    let a = u.abs().min(1.0);
                    (1.0 - a) * (PI * a).cos() + (PI * a).sin() / PI
                }
                WindowKind::Lanczos => {
                    if u == 0.0 {
                        1.0
                    } else {
                        (PI * u).sin() / (PI * u)
                    }
                }
            }
        })
        .collect()
}

/// Dolph-Chebyshev window with all sidelobes `atten_db` below the main
/// lobe, computed from its closed-form DFT (the construction used by
/// scipy's `chebwin`).
fn dolph_chebyshev(n: usize, atten_db: f64, periodic: bool) -> Vec<f64> {
    if periodic {
        // DFT-even form: symmetric window of length n+1 minus its last sample.
        let mut w = dolph_chebyshev(n + 1, atten_db, false);
        w.truncate(n);
        return w;
    }
    let m = n;
    let order = (m - 1) as f64;
    let r = 10.0_f64.powf(atten_db.abs() / 20.0);
    let beta = (r.acosh() / order).cosh();
    let p: Vec<f64> = (0..m)
        .map(|k| chebyshev_t(order, beta * (PI * k as f64 / m as f64).cos()))
        .collect();
    // Inverse construction: take the real part of the DFT of p (odd m),
    // or of p with a half-sample phase shift (even m), then mirror.
    let mut w = Vec::with_capacity(m);
    if m % 2 == 1 {
        let re: Vec<f64> = (0..m)
            .map(|j| {
                p.iter()
                    .enumerate()
                    .map(|(k, &v)| v * (TWO_PI * (j * k) as f64 / m as f64).cos())
                    .sum()
            })
            .collect();
        let half = m / 2 + 1;
        for i in (1..half).rev() {
            w.push(re[i]);
        }
        w.extend_from_slice(&re[..half]);
    } else {
        let re: Vec<f64> = (0..m)
            .map(|j| {
                p.iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        let ang = PI * k as f64 / m as f64 - TWO_PI * (j * k) as f64 / m as f64;
                        v * ang.cos()
                    })
                    .sum()
            })
            .collect();
        let half = m / 2 + 1;
        for i in (1..half).rev() {
            w.push(re[i]);
        }
        w.extend_from_slice(&re[1..half]);
    }
    let peak = w.iter().cloned().fold(f64::MIN, f64::max);
    for v in w.iter_mut() {
        *v /= peak;
    }
    w
}

/// Chebyshev polynomial of the first kind T_ν(x), extended off [−1, 1]
/// with the hyperbolic form.
fn chebyshev_t(nu: f64, x: f64) -> f64 {
    if x.abs() <= 1.0 {
        (nu * x.acos()).cos()
    } else if x > 1.0 {
        (nu * x.acosh()).cosh()
    } else {
        // x < -1
        let sign = if (nu as i64) % 2 == 0 { 1.0 } else { -1.0 };
        sign * (nu * (-x).acosh()).cosh()
    }
}

/// Figures of merit for a window (Harris 1978).
#[derive(Debug, Clone, Copy)]
pub struct WindowMetrics {
    /// DC gain relative to a rectangular window: Σw / n.
    pub coherent_gain: f64,
    /// Equivalent noise bandwidth in DFT bins: n·Σw² / (Σw)².
    pub enbw: f64,
    /// Worst-case loss for a tone midway between bins, in dB (positive).
    pub scallop_loss_db: f64,
    /// Width of the main lobe between its first nulls, in bins.
    pub main_lobe_bins: f64,
    /// Highest sidelobe level relative to the main lobe, in dB (negative).
    pub max_sidelobe_db: f64,
}

/// Measure a window's figures of merit by direct evaluation of its DTFT
/// on a fine frequency grid (64 points per bin).
#[must_use]
pub fn window_metrics(w: &[f64]) -> WindowMetrics {
    let n = w.len();
    assert!(n > 0, "window_metrics requires a non-empty window");
    let sum: f64 = w.iter().sum();
    let sum_sq: f64 = w.iter().map(|v| v * v).sum();
    let coherent_gain = sum / n as f64;
    let enbw = n as f64 * sum_sq / (sum * sum);

    // |W(f)| with f in bins.
    let mag = |f_bins: f64| -> f64 {
        let mut acc = Complex::new(0.0, 0.0);
        for (k, &v) in w.iter().enumerate() {
            let ang = -TWO_PI * f_bins * k as f64 / n as f64;
            acc = acc + Complex::new(v * ang.cos(), v * ang.sin());
        }
        acc.norm()
    };
    let peak = mag(0.0);
    let scallop_loss_db = -20.0 * (mag(0.5) / peak).log10();

    // Scan out to n/2 bins for the first null, then the max sidelobe.
    let steps_per_bin = 64;
    let max_bins = (n / 2).max(4);
    let mut first_null = f64::NAN;
    let mut prev = peak;
    let mut falling = false;
    let mut i = 1;
    while i <= max_bins * steps_per_bin {
        let f = i as f64 / steps_per_bin as f64;
        let m = mag(f);
        if m < prev {
            falling = true;
        } else if falling {
            // Local minimum at the previous sample: the first null.
            first_null = (i - 1) as f64 / steps_per_bin as f64;
            break;
        }
        prev = m;
        i += 1;
    }
    let (main_lobe_bins, max_sidelobe_db) = if first_null.is_nan() {
        (f64::NAN, f64::NEG_INFINITY)
    } else {
        let mut max_side = 0.0_f64;
        let start = (first_null * steps_per_bin as f64) as usize + 1;
        for j in start..=max_bins * steps_per_bin {
            let f = j as f64 / steps_per_bin as f64;
            max_side = max_side.max(mag(f));
        }
        (2.0 * first_null, 20.0 * (max_side / peak).log10())
    };

    WindowMetrics {
        coherent_gain,
        enbw,
        scallop_loss_db,
        main_lobe_bins,
        max_sidelobe_db,
    }
}

/// Kaiser window β for a target stopband attenuation in dB
/// (Kaiser's empirical formula).
#[must_use]
pub fn kaiser_beta_for_attenuation(db: f64) -> f64 {
    let a = db.abs();
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

// --- Pre-Part-3 generators (wrap `window` with the original symmetric
// convention) ---

/// Generate a Hann window of length n: `w[k] = 0.5·(1 - cos(2πk/(n-1)))`
#[must_use]
pub fn hann_window(n: usize) -> Vec<f64> {
    window(WindowKind::Hann, n, false)
}

/// Generate a Hamming window of length n: `w[k] = 0.54 - 0.46·cos(2πk/(n-1))`
#[must_use]
pub fn hamming_window(n: usize) -> Vec<f64> {
    window(WindowKind::Hamming, n, false)
}

/// Generate a Blackman window of length n:
    /// `w[k] = 0.42 - 0.5·cos(2πk/(n-1)) + 0.08·cos(4πk/(n-1))`
#[must_use]
pub fn blackman_window(n: usize) -> Vec<f64> {
    window(WindowKind::Blackman, n, false)
}

/// Generate a rectangular (uniform) window of length n: `w[k] = 1` for all k
#[must_use]
pub fn rectangular_window(n: usize) -> Vec<f64> {
    window(WindowKind::Rect, n, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_symmetric_windows_are_symmetric() {
        for kind in [
            WindowKind::Hann,
            WindowKind::Hamming,
            WindowKind::Blackman,
            WindowKind::BlackmanHarris,
            WindowKind::Nuttall,
            WindowKind::FlatTop,
            WindowKind::Kaiser(8.0),
            WindowKind::Tukey(0.5),
            WindowKind::Gaussian(0.4),
            WindowKind::DolphChebyshev(80.0),
            WindowKind::Bartlett,
            WindowKind::Bohman,
            WindowKind::Lanczos,
        ] {
            for &n in &[16usize, 17] {
                let w = window(kind, n, false);
                for k in 0..n {
                    assert!(
                        approx(w[k], w[n - 1 - k], 1e-9),
                        "{kind:?} not symmetric at n={n}, k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_periodic_window_extends_to_symmetric() {
        // periodic window of length n = first n samples of symmetric n+1.
        let p = window(WindowKind::Hann, 16, true);
        let s = window(WindowKind::Hann, 17, false);
        for k in 0..16 {
            assert!(approx(p[k], s[k], 1e-12));
        }
    }

    #[test]
    fn test_legacy_wrappers_match_old_formulas() {
        let n = 64;
        let denom = (n - 1) as f64;
        let hann = hann_window(n);
        let hamming = hamming_window(n);
        let blackman = blackman_window(n);
        for k in 0..n {
            let t = TWO_PI * k as f64 / denom;
            assert!(approx(hann[k], 0.5 * (1.0 - t.cos()), 1e-12));
            assert!(approx(hamming[k], 0.54 - 0.46 * t.cos(), 1e-12));
            assert!(approx(
                blackman[k],
                0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos(),
                1e-12
            ));
        }
        assert!(rectangular_window(5).iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_edge_lengths() {
        assert!(window(WindowKind::Hann, 0, false).is_empty());
        assert_eq!(window(WindowKind::Hann, 1, false), vec![1.0]);
        assert_eq!(window(WindowKind::DolphChebyshev(60.0), 1, true), vec![1.0]);
    }

    #[test]
    fn test_kaiser_zero_beta_is_rect() {
        let w = window(WindowKind::Kaiser(0.0), 8, false);
        assert!(w.iter().all(|&v| approx(v, 1.0, 1e-12)));
    }

    #[test]
    fn test_tukey_limits() {
        let rect = window(WindowKind::Tukey(0.0), 9, false);
        assert!(rect.iter().all(|&v| approx(v, 1.0, 1e-12)));
        let hann = window(WindowKind::Tukey(1.0), 9, false);
        let reference = window(WindowKind::Hann, 9, false);
        for (a, b) in hann.iter().zip(&reference) {
            assert!(approx(*a, *b, 1e-9));
        }
    }

    #[test]
    fn test_metrics_rectangular() {
        let m = window_metrics(&window(WindowKind::Rect, 64, true));
        assert!(approx(m.coherent_gain, 1.0, 1e-12));
        assert!(approx(m.enbw, 1.0, 1e-12));
        assert!(approx(m.scallop_loss_db, 3.92, 0.02));
        assert!(approx(m.main_lobe_bins, 2.0, 0.1));
        assert!(approx(m.max_sidelobe_db, -13.3, 0.3));
    }

    #[test]
    fn test_metrics_hann() {
        let m = window_metrics(&window(WindowKind::Hann, 64, true));
        assert!(approx(m.coherent_gain, 0.5, 0.01));
        assert!(approx(m.enbw, 1.5, 0.02));
        assert!(approx(m.scallop_loss_db, 1.42, 0.05));
        assert!(approx(m.main_lobe_bins, 4.0, 0.1));
        assert!(approx(m.max_sidelobe_db, -31.5, 0.5));
    }

    #[test]
    fn test_metrics_blackman_sidelobes() {
        let m = window_metrics(&window(WindowKind::Blackman, 128, true));
        assert!(m.max_sidelobe_db < -57.0, "got {}", m.max_sidelobe_db);
    }

    #[test]
    fn test_dolph_chebyshev_sidelobes_at_spec() {
        for &atten in &[60.0, 80.0] {
            let m = window_metrics(&window(WindowKind::DolphChebyshev(atten), 65, false));
            assert!(
                (m.max_sidelobe_db + atten).abs() < 1.5,
                "atten {atten}: got {}",
                m.max_sidelobe_db
            );
        }
    }

    #[test]
    fn test_kaiser_beta_for_attenuation_regimes() {
        assert!(approx(kaiser_beta_for_attenuation(10.0), 0.0, 1e-12));
        assert!(approx(kaiser_beta_for_attenuation(60.0), 0.1102 * 51.3, 1e-9));
        let b40 = kaiser_beta_for_attenuation(40.0);
        assert!(b40 > 3.0 && b40 < 4.0);
    }
}
