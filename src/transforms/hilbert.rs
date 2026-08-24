//! Hilbert transform, analytic signals, modulation, empirical mode
//! decomposition, and causality (Kramers-Kronig) tools.

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::numerical::CubicSpline;
use crate::transforms::fft::{fft_any, ifft_any};

const TWO_PI: f64 = 2.0 * PI;

/// Analytic signal x + j·H(x) by the FFT method: double the positive
/// frequencies, zero the negative ones.
#[must_use]
pub fn analytic_signal(x: &[f64]) -> Vec<Complex> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut spec = fft_any(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
    for (k, s) in spec.iter_mut().enumerate() {
        if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
            // DC and Nyquist stay
        } else if k < n.div_ceil(2) {
            *s = Complex::new(2.0 * s.re, 2.0 * s.im);
        } else {
            *s = Complex::new(0.0, 0.0);
        }
    }
    ifft_any(&spec)
}

/// Hilbert transform: the quadrature (imaginary) part of the analytic
/// signal — hilbert(cos ωt) = sin ωt.
#[must_use]
pub fn hilbert(x: &[f64]) -> Vec<f64> {
    analytic_signal(x).iter().map(|c| c.im).collect()
}

/// Instantaneous amplitude |x + jH(x)|.
#[must_use]
pub fn envelope(x: &[f64]) -> Vec<f64> {
    analytic_signal(x).iter().map(|c| c.norm()).collect()
}

/// Unwrapped instantaneous phase of the analytic signal.
#[must_use]
pub fn instantaneous_phase(x: &[f64]) -> Vec<f64> {
    let raw: Vec<f64> = analytic_signal(x).iter().map(|c| c.arg()).collect();
    // Unwrap.
    let mut out = Vec::with_capacity(raw.len());
    let mut offset = 0.0;
    for (i, &p) in raw.iter().enumerate() {
        if i > 0 {
            let d = p - raw[i - 1];
            if d > PI {
                offset -= TWO_PI;
            } else if d < -PI {
                offset += TWO_PI;
            }
        }
        out.push(p + offset);
    }
    out
}

/// Instantaneous frequency in Hz (central difference of the unwrapped
/// phase).
#[must_use]
pub fn instantaneous_frequency(x: &[f64], fs: f64) -> Vec<f64> {
    let ph = instantaneous_phase(x);
    let n = ph.len();
    (0..n)
        .map(|i| {
            let d = if n < 2 {
                0.0
            } else if i == 0 {
                ph[1] - ph[0]
            } else if i == n - 1 {
                ph[n - 1] - ph[n - 2]
            } else {
                (ph[i + 1] - ph[i - 1]) / 2.0
            };
            d * fs / TWO_PI
        })
        .collect()
}

/// FIR Hilbert transformer kernel (odd taps, antisymmetric, windowed).
///
/// # Panics
/// Panics unless `n_taps` is odd.
#[must_use]
pub fn hilbert_fir(n_taps: usize) -> Vec<f64> {
    crate::dsp::fir::fir_hilbert(n_taps)
}

/// Single-sideband modulation: upper sideband is x·cos − H(x)·sin,
/// lower is x·cos + H(x)·sin (carrier fc Hz at sample rate fs).
#[must_use]
pub fn ssb_modulate(x: &[f64], fc: f64, fs: f64, upper: bool) -> Vec<f64> {
    let h = hilbert(x);
    let s = if upper { -1.0 } else { 1.0 };
    x.iter()
        .zip(&h)
        .enumerate()
        .map(|(i, (&xv, &hv))| {
            let ph = TWO_PI * fc * i as f64 / fs;
            xv * ph.cos() + s * hv * ph.sin()
        })
        .collect()
}

/// AM envelope demodulation: the analytic-signal envelope (carrier plus
/// modulation; subtract the mean to recover the AC message).
#[must_use]
pub fn am_demodulate(x: &[f64]) -> Vec<f64> {
    envelope(x)
}

/// FM demodulation: instantaneous frequency of the analytic signal (Hz).
#[must_use]
pub fn fm_demodulate(x: &[f64], fs: f64) -> Vec<f64> {
    instantaneous_frequency(x, fs)
}

/// Local extrema (index, value) of a sequence; kind = +1 maxima, −1 minima.
fn local_extrema(x: &[f64], kind: i32) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 1..x.len().saturating_sub(1) {
        let is_ext = if kind > 0 {
            x[i] > x[i - 1] && x[i] >= x[i + 1]
        } else {
            x[i] < x[i - 1] && x[i] <= x[i + 1]
        };
        if is_ext {
            out.push((i as f64, x[i]));
        }
    }
    out
}

/// Cubic-spline envelope through extrema, mirrored at the ends to tame
/// boundary swings.
fn spline_envelope(ext: &[(f64, f64)], n: usize) -> Option<Vec<f64>> {
    if ext.len() < 2 {
        return None;
    }
    // Mirror the first and last extremum around the signal edges.
    let mut xs = Vec::with_capacity(ext.len() + 2);
    let mut ys = Vec::with_capacity(ext.len() + 2);
    xs.push(-ext[0].0.max(1.0));
    ys.push(ext[0].1);
    for &(x, y) in ext {
        xs.push(x);
        ys.push(y);
    }
    let last = ext[ext.len() - 1];
    xs.push(2.0 * (n as f64 - 1.0) - last.0);
    ys.push(last.1);
    if xs.len() < 3 {
        return None;
    }
    let spline = CubicSpline::natural(&xs, &ys).ok()?;
    Some((0..n).map(|i| spline.eval(i as f64)).collect())
}

/// Empirical mode decomposition (Huang sifting): returns the IMFs plus
/// the final residual as the last entry, so the components sum to x.
#[must_use]
pub fn empirical_mode_decomposition(x: &[f64], max_imfs: usize, sift_tol: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut components: Vec<Vec<f64>> = Vec::new();
    let mut residual = x.to_vec();
    for _ in 0..max_imfs {
        let maxima = local_extrema(&residual, 1);
        let minima = local_extrema(&residual, -1);
        if maxima.len() < 2 || minima.len() < 2 {
            break; // residual is monotone-ish: stop
        }
        let mut h = residual.clone();
        for _sift in 0..100 {
            let upper = spline_envelope(&local_extrema(&h, 1), n);
            let lower = spline_envelope(&local_extrema(&h, -1), n);
            let (Some(up), Some(lo)) = (upper, lower) else { break };
            let mean: Vec<f64> = up.iter().zip(&lo).map(|(u, l)| 0.5 * (u + l)).collect();
            let next: Vec<f64> = h.iter().zip(&mean).map(|(v, m)| v - m).collect();
            // Standard deviation criterion.
            let sd: f64 = h
                .iter()
                .zip(&next)
                .map(|(a, b)| (a - b) * (a - b) / (a * a + 1e-12))
                .sum();
            h = next;
            if sd < sift_tol {
                break;
            }
        }
        for (r, hv) in residual.iter_mut().zip(&h) {
            *r -= hv;
        }
        components.push(h);
    }
    components.push(residual);
    components
}

/// Hilbert-Huang spectrum: per IMF (excluding the residual), the
/// instantaneous frequency track and amplitude envelope.
#[must_use]
pub fn hilbert_huang_spectrum(x: &[f64], fs: f64, max_imfs: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let comps = empirical_mode_decomposition(x, max_imfs, 0.05);
    let n_imfs = comps.len().saturating_sub(1);
    comps[..n_imfs]
        .iter()
        .map(|imf| (instantaneous_frequency(imf, fs), envelope(imf)))
        .collect()
}

/// Kramers-Kronig relation: real part of a causal response from its
/// imaginary part sampled on `omega` (ω ≥ 0), by principal-value
/// trapezoid integration of (2/π)∫ ω′·Im(ω′)/(ω′² − ω²) dω′.
///
/// # Panics
/// Panics if the lengths differ.
#[must_use]
pub fn kramers_kronig(im: &[f64], omega: &[f64]) -> Vec<f64> {
    assert_eq!(im.len(), omega.len(), "im and omega must have equal length");
    let n = im.len();
    (0..n)
        .map(|i| {
            let w = omega[i];
            let mut acc = 0.0;
            for j in 0..n.saturating_sub(1) {
                // Trapezoid on [ω_j, ω_{j+1}], skipping panels touching
                // the singularity (principal value).
                if j == i || j + 1 == i {
                    continue;
                }
                let f = |k: usize| omega[k] * im[k] / (omega[k] * omega[k] - w * w);
                acc += 0.5 * (f(j) + f(j + 1)) * (omega[j + 1] - omega[j]);
            }
            2.0 / PI * acc
        })
        .collect()
}

/// Minimum-phase spectrum with the given magnitude, by the real-cepstrum
/// method: fold the causal cepstrum and re-exponentiate. `mag` samples
/// |H| on the full FFT circle (length n).
#[must_use]
pub fn minimum_phase_from_magnitude(mag: &[f64]) -> Vec<Complex> {
    let n = mag.len();
    if n == 0 {
        return Vec::new();
    }
    let log_mag: Vec<Complex> = mag
        .iter()
        .map(|&m| Complex::new(m.max(1e-300).ln(), 0.0))
        .collect();
    let cep = ifft_any(&log_mag);
    // Fold: keep c[0] (and Nyquist for even n), double 1..n/2, zero rest.
    let mut folded = vec![Complex::new(0.0, 0.0); n];
    folded[0] = cep[0];
    for k in 1..n.div_ceil(2) {
        folded[k] = Complex::new(2.0 * cep[k].re, 2.0 * cep[k].im);
    }
    if n.is_multiple_of(2) {
        folded[n / 2] = cep[n / 2];
    }
    let log_h = fft_any(&folded);
    log_h
        .iter()
        .map(|c| {
            let r = c.re.exp();
            Complex::new(r * c.im.cos(), r * c.im.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_hilbert_of_cos_is_sin() {
        let n = 256;
        let x: Vec<f64> = (0..n).map(|i| (TWO_PI * 8.0 * i as f64 / n as f64).cos()).collect();
        let h = hilbert(&x);
        for (i, v) in h.iter().enumerate() {
            let expect = (TWO_PI * 8.0 * i as f64 / n as f64).sin();
            assert!(approx(*v, expect, 1e-9), "at {i}: {v} vs {expect}");
        }
    }

    #[test]
    fn test_envelope_of_am_tone() {
        let n = 1000; // exactly 1 s so both tones are FFT-periodic
        let fs = 1000.0;
        // 100 Hz carrier, 5 Hz modulation, amplitude 2.
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                2.0 * (1.0 + 0.5 * (TWO_PI * 5.0 * t).cos()) * (TWO_PI * 100.0 * t).sin()
            })
            .collect();
        let env = envelope(&x);
        for (i, &e) in env.iter().enumerate().take(n - 100).skip(100) {
            let t = i as f64 / fs;
            let expect = 2.0 * (1.0 + 0.5 * (TWO_PI * 5.0 * t).cos());
            assert!((e - expect).abs() / expect < 0.01, "at {i}: {e} vs {expect}");
        }
    }

    #[test]
    fn test_instantaneous_frequency_tracks_chirp() {
        let fs = 1000.0;
        let n = 2000;
        // Linear chirp 50 → 150 Hz.
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let phase = TWO_PI * (50.0 * t + 25.0 * t * t); // f(t) = 50 + 50 t
                phase.sin()
            })
            .collect();
        let freq = instantaneous_frequency(&x, fs);
        for i in (200..n - 200).step_by(100) {
            let t = i as f64 / fs;
            let expect = 50.0 + 50.0 * t;
            assert!(
                (freq[i] - expect).abs() < 2.0,
                "at t={t}: {} vs {expect}",
                freq[i]
            );
        }
    }

    #[test]
    fn test_ssb_occupies_one_sideband() {
        let fs = 8000.0;
        let n = 4096;
        let fm = 200.0;
        let fc = 2000.0;
        let msg: Vec<f64> = (0..n).map(|i| (TWO_PI * fm * i as f64 / fs).cos()).collect();
        let usb = ssb_modulate(&msg, fc, fs, true);
        let spec = crate::transforms::fft::rfft(&usb);
        let bin = |f: f64| (f * n as f64 / fs).round() as usize;
        let upper = spec[bin(fc + fm)].norm();
        let lower = spec[bin(fc - fm)].norm();
        assert!(upper > 100.0 * lower, "USB {upper} vs LSB {lower}");
    }

    #[test]
    fn test_fm_demodulate() {
        let fs = 10000.0;
        let n = 4000;
        // Carrier 2 kHz, frequency swings ±100 Hz at 10 Hz rate.
        let mut phase = 0.0;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let f = 2000.0 + 100.0 * (TWO_PI * 10.0 * t).sin();
                phase += TWO_PI * f / fs;
                phase.sin()
            })
            .collect();
        let demod = fm_demodulate(&x, fs);
        for i in (400..n - 400).step_by(200) {
            let t = i as f64 / fs;
            let expect = 2000.0 + 100.0 * (TWO_PI * 10.0 * t).sin();
            assert!((demod[i] - expect).abs() < 10.0, "at {i}: {} vs {expect}", demod[i]);
        }
    }

    #[test]
    fn test_emd_components_sum_to_signal() {
        let n = 512;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (TWO_PI * 30.0 * t).sin() + 0.5 * (TWO_PI * 5.0 * t).sin() + 2.0 * t
            })
            .collect();
        let comps = empirical_mode_decomposition(&x, 4, 0.05);
        assert!(comps.len() >= 2);
        for i in 0..n {
            let sum: f64 = comps.iter().map(|c| c[i]).sum();
            assert!(approx(sum, x[i], 1e-9), "at {i}");
        }
        // First IMF should carry the fast oscillation: it should have
        // many more zero crossings than the last (residual/trend).
        let crossings = |v: &[f64]| {
            v.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count()
        };
        let first = crossings(&comps[0]);
        let last = crossings(comps.last().unwrap());
        assert!(first > 3 * (last + 1), "IMF1 {first} vs residual {last}");
    }

    #[test]
    fn test_hilbert_huang_spectrum_shapes() {
        let n = 256;
        let fs = 256.0;
        let x: Vec<f64> = (0..n).map(|i| (TWO_PI * 20.0 * i as f64 / fs).sin()).collect();
        let hhs = hilbert_huang_spectrum(&x, fs, 3);
        assert!(!hhs.is_empty());
        let (freqs, amps) = &hhs[0];
        assert_eq!(freqs.len(), n);
        assert_eq!(amps.len(), n);
        // The dominant IMF of a pure tone should sit near 20 Hz.
        let mid: f64 = freqs[n / 4..3 * n / 4].iter().sum::<f64>() / (n / 2) as f64;
        assert!((mid - 20.0).abs() < 2.0, "mean inst freq {mid}");
    }

    #[test]
    fn test_kramers_kronig_lorentzian() {
        // Damped-oscillator susceptibility: χ(ω) = 1/(ω0² − ω² − iγω);
        // Im χ = γω/D, Re χ = (ω0² − ω²)/D with D = (ω0²−ω²)² + γ²ω².
        let (w0, gamma) = (1.0, 0.2);
        let n = 4000;
        let omega: Vec<f64> = (0..n).map(|i| 5.0 * i as f64 / n as f64).collect();
        let im: Vec<f64> = omega
            .iter()
            .map(|&w| {
                let d = (w0 * w0 - w * w).powi(2) + gamma * gamma * w * w;
                gamma * w / d
            })
            .collect();
        let re = kramers_kronig(&im, &omega);
        // Compare away from the resonance and the truncated tail.
        for &wt in &[0.3, 0.5, 1.5, 2.0] {
            let i = (wt / 5.0 * n as f64) as usize;
            let w = omega[i];
            let d = (w0 * w0 - w * w).powi(2) + gamma * gamma * w * w;
            let expect = (w0 * w0 - w * w) / d;
            assert!(
                (re[i] - expect).abs() < 0.05 * expect.abs().max(0.3),
                "at ω={w}: {} vs {expect}",
                re[i]
            );
        }
    }

    #[test]
    fn test_minimum_phase_preserves_magnitude() {
        let n = 128;
        // A smooth positive magnitude on the circle (real, even).
        let mag: Vec<f64> = (0..n)
            .map(|k| 1.0 + 0.5 * (TWO_PI * k as f64 / n as f64).cos())
            .collect();
        let h = minimum_phase_from_magnitude(&mag);
        for (k, c) in h.iter().enumerate() {
            assert!(approx(c.norm(), mag[k], 1e-6), "bin {k}: {} vs {}", c.norm(), mag[k]);
        }
        // Minimum phase implies a causal impulse response: energy in the
        // "negative time" half should be negligible.
        let imp = ifft_any(&h);
        let head: f64 = imp[..n / 2].iter().map(|c| c.norm_sq()).sum();
        let tail: f64 = imp[n / 2..].iter().map(|c| c.norm_sq()).sum();
        assert!(tail < 1e-6 * head, "anticausal energy {tail} vs {head}");
    }
}
