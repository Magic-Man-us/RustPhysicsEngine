//! Sample-rate conversion: integer up/down sampling, polyphase rational
//! resampling, windowed-sinc/linear/cubic interpolation, CIC decimation,
//! and half-band filters.
//!
//! Anti-aliasing and interpolation kernels are symmetric windowed-sinc
//! filters applied centered ("same" alignment), so resampled signals
//! keep zero net delay.

use crate::dsp::windows::{kaiser_beta_for_attenuation, window, WindowKind};
use crate::math::constants::PI;

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// Symmetric Kaiser-windowed sinc low-pass, cutoff normalized to fs,
/// unit DC gain, odd length.
fn aa_kernel(cutoff: f64, taps: usize, atten_db: f64) -> Vec<f64> {
    let taps = if taps.is_multiple_of(2) { taps + 1 } else { taps };
    let beta = kaiser_beta_for_attenuation(atten_db);
    let win = window(WindowKind::Kaiser(beta), taps, false);
    let c = (taps / 2) as isize;
    let mut h: Vec<f64> = (0..taps)
        .map(|k| {
            let m = k as isize - c;
            2.0 * cutoff * sinc(2.0 * cutoff * m as f64) * win[k]
        })
        .collect();
    let sum: f64 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= sum;
    }
    h
}

/// Centered ("same") convolution with a symmetric odd-length kernel:
/// zero net delay.
fn convolve_same(h: &[f64], x: &[f64]) -> Vec<f64> {
    let c = (h.len() / 2) as isize;
    (0..x.len() as isize)
        .map(|i| {
            let mut acc = 0.0;
            for (k, &hv) in h.iter().enumerate() {
                let idx = i + c - k as isize;
                if idx >= 0 && (idx as usize) < x.len() {
                    acc += hv * x[idx as usize];
                }
            }
            acc
        })
        .collect()
}

/// Integer upsampling: zero-stuff by `factor`, then interpolate with a
/// Kaiser-windowed sinc low-pass at the original Nyquist. Output length
/// is `x.len() * factor`.
///
/// # Panics
/// Panics if `factor == 0`.
#[must_use]
pub fn upsample(x: &[f64], factor: usize) -> Vec<f64> {
    assert!(factor > 0, "factor must be positive");
    if factor == 1 || x.is_empty() {
        return x.to_vec();
    }
    let mut stuffed = vec![0.0; x.len() * factor];
    for (i, &v) in x.iter().enumerate() {
        stuffed[i * factor] = v * factor as f64;
    }
    let h = aa_kernel(0.5 / factor as f64, 32 * factor + 1, 120.0);
    convolve_same(&h, &stuffed)
}

/// Integer decimation: anti-alias low-pass at the new Nyquist, then
/// keep every `factor`-th sample. Output length ⌈n/factor⌉.
///
/// # Panics
/// Panics if `factor == 0`.
#[must_use]
pub fn decimate(x: &[f64], factor: usize) -> Vec<f64> {
    assert!(factor > 0, "factor must be positive");
    if factor == 1 || x.is_empty() {
        return x.to_vec();
    }
    let h = aa_kernel(0.5 / factor as f64 * 0.92, 32 * factor + 1, 100.0);
    let filtered = convolve_same(&h, x);
    filtered.iter().step_by(factor).copied().collect()
}

/// Rational resampling by up/down with a single polyphase Kaiser-sinc
/// kernel. Output length ⌈n·up/down⌉.
///
/// # Panics
/// Panics if `up == 0` or `down == 0`.
#[must_use]
pub fn resample_rational(x: &[f64], up: usize, down: usize) -> Vec<f64> {
    assert!(up > 0 && down > 0, "up and down must be positive");
    if x.is_empty() || up == down {
        return x.to_vec();
    }
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    let max_ud = up.max(down);
    let h = aa_kernel(0.5 / max_ud as f64 * 0.92, 32 * max_ud + 1, 100.0);
    let c = (h.len() / 2) as isize;
    let out_len = x.len() * up / down + usize::from(!(x.len() * up).is_multiple_of(down));
    // y[j] = Σ_k h[k]·stuffed[j·down + c − k]·up, only multiples of `up`
    // in the stuffed sequence are nonzero.
    (0..out_len)
        .map(|j| {
            let m = (j * down) as isize + c;
            let mut acc = 0.0;
            // k with (m − k) ≡ 0 (mod up): start at m mod up.
            let mut k = (m % up as isize + up as isize) % up as isize;
            while k < h.len() as isize {
                let src = (m - k) / up as isize;
                if src >= 0 && (src as usize) < x.len() {
                    acc += h[k as usize] * x[src as usize];
                }
                k += up as isize;
            }
            acc * up as f64
        })
        .collect()
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Resample from `fs_in` to `fs_out`, approximating the ratio with a
/// rational up/down (denominator ≤ 1000) and delegating to
/// [`resample_rational`].
///
/// # Panics
/// Panics unless both rates are positive.
#[must_use]
pub fn resample_to_rate(x: &[f64], fs_in: f64, fs_out: f64) -> Vec<f64> {
    assert!(fs_in > 0.0 && fs_out > 0.0, "rates must be positive");
    let ratio = fs_out / fs_in;
    // Best rational approximation with bounded denominator (Stern-Brocot
    // style continued fractions).
    let (mut num, mut den) = (1usize, 1usize);
    let mut best_err = (ratio - 1.0).abs();
    for d in 1..=1000usize {
        let n = (ratio * d as f64).round().max(1.0) as usize;
        let err = (ratio - n as f64 / d as f64).abs();
        if err < best_err - 1e-15 {
            best_err = err;
            num = n;
            den = d;
            if err < 1e-12 {
                break;
            }
        }
    }
    resample_rational(x, num, den)
}

/// Windowed-sinc (Hann) interpolation of the sample stream at fractional
/// index t (samples), using `half_width` taps on each side.
#[must_use]
pub fn sinc_interpolate(x: &[f64], t: f64, half_width: usize) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let center = t.floor() as isize;
    let hw = half_width.max(1) as isize;
    let mut acc = 0.0;
    for k in (center - hw + 1)..=(center + hw) {
        if k < 0 || k as usize >= x.len() {
            continue;
        }
        let d = t - k as f64;
        // Hann-windowed sinc over |d| < hw.
        let wnd = 0.5 * (1.0 + (PI * d / hw as f64).cos());
        acc += x[k as usize] * sinc(d) * wnd;
    }
    acc
}

/// Arbitrary-ratio resampling by windowed-sinc interpolation; output
/// length ⌈n·ratio⌉.
///
/// # Panics
/// Panics if `ratio <= 0`.
#[must_use]
pub fn resample_sinc(x: &[f64], ratio: f64, half_width: usize) -> Vec<f64> {
    assert!(ratio > 0.0, "ratio must be positive");
    let out_len = (x.len() as f64 * ratio).ceil() as usize;
    (0..out_len).map(|j| sinc_interpolate(x, j as f64 / ratio, half_width)).collect()
}

/// Linear-interpolation resampling (cheap, −12 dB/oct images).
///
/// # Panics
/// Panics if `ratio <= 0`.
#[must_use]
pub fn resample_linear(x: &[f64], ratio: f64) -> Vec<f64> {
    assert!(ratio > 0.0, "ratio must be positive");
    if x.is_empty() {
        return Vec::new();
    }
    let out_len = (x.len() as f64 * ratio).ceil() as usize;
    (0..out_len)
        .map(|j| {
            let t = j as f64 / ratio;
            let i = (t.floor() as usize).min(x.len() - 1);
            let f = t - i as f64;
            let a = x[i];
            let b = x[(i + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

/// Catmull-Rom cubic resampling.
///
/// # Panics
/// Panics if `ratio <= 0`.
#[must_use]
pub fn resample_cubic(x: &[f64], ratio: f64) -> Vec<f64> {
    assert!(ratio > 0.0, "ratio must be positive");
    if x.is_empty() {
        return Vec::new();
    }
    let n = x.len();
    let get = |i: isize| -> f64 { x[i.clamp(0, n as isize - 1) as usize] };
    let out_len = (n as f64 * ratio).ceil() as usize;
    (0..out_len)
        .map(|j| {
            let t = j as f64 / ratio;
            let i = t.floor() as isize;
            let f = t - i as f64;
            let (p0, p1, p2, p3) = (get(i - 1), get(i), get(i + 1), get(i + 2));
            0.5 * ((2.0 * p1)
                + (-p0 + p2) * f
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f * f
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f * f * f)
        })
        .collect()
}

/// Cascaded integrator-comb decimation: `stages` integrators, decimate
/// by `factor`, `stages` combs; output scaled by factor^stages so DC
/// gain is one.
///
/// # Panics
/// Panics if `factor == 0` or `stages == 0`.
#[must_use]
pub fn cic_decimate(x: &[f64], factor: usize, stages: usize) -> Vec<f64> {
    assert!(factor > 0 && stages > 0, "factor and stages must be positive");
    // Integrators at the high rate.
    let mut data = x.to_vec();
    for _ in 0..stages {
        let mut acc = 0.0;
        for v in data.iter_mut() {
            acc += *v;
            *v = acc;
        }
    }
    // Decimate.
    let mut dec: Vec<f64> = data.iter().step_by(factor).copied().collect();
    // Combs at the low rate (differential delay 1).
    for _ in 0..stages {
        let mut prev = 0.0;
        for v in dec.iter_mut() {
            let cur = *v;
            *v = cur - prev;
            prev = cur;
        }
    }
    let scale = (factor as f64).powi(stages as i32);
    for v in dec.iter_mut() {
        *v /= scale;
    }
    dec
}

/// Half-band FIR: odd length, every second tap zero (except the 0.5
/// center), cutoff 0.25 — the workhorse for factor-2 stages.
///
/// # Panics
/// Panics unless `n_taps` is odd and ≥ 7.
#[must_use]
pub fn half_band_filter(n_taps: usize) -> Vec<f64> {
    assert!(n_taps % 2 == 1 && n_taps >= 7, "need an odd tap count >= 7");
    let win = window(WindowKind::Kaiser(8.0), n_taps, false);
    let c = (n_taps / 2) as isize;
    let mut h: Vec<f64> = (0..n_taps)
        .map(|k| {
            let m = k as isize - c;
            if m == 0 {
                0.5
            } else if m % 2 == 0 {
                0.0 // exact half-band zeros
            } else {
                0.5 * sinc(m as f64 / 2.0) * win[k]
            }
        })
        .collect();
    // Normalize DC gain to 1 by scaling the odd taps.
    let dc: f64 = h.iter().sum();
    let err = 1.0 - dc;
    let odd_sum: f64 = h
        .iter()
        .enumerate()
        .filter(|(k, _)| (*k as isize - c) % 2 != 0)
        .map(|(_, &v)| v)
        .sum();
    for (k, v) in h.iter_mut().enumerate() {
        if (k as isize - c) % 2 != 0 {
            *v *= 1.0 + err / odd_sum;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_PI: f64 = 2.0 * PI;

    fn tone(f: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (TWO_PI * f * i as f64).sin()).collect()
    }

    fn spectrum_mag_at(x: &[f64], f: f64) -> f64 {
        // Goertzel-style projection with a Hann window.
        let n = x.len();
        let w = window(WindowKind::Hann, n, true);
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, (&v, &wv)) in x.iter().zip(&w).enumerate() {
            let ang = TWO_PI * f * i as f64;
            re += v * wv * ang.cos();
            im -= v * wv * ang.sin();
        }
        2.0 * (re * re + im * im).sqrt() / w.iter().sum::<f64>()
    }

    #[test]
    fn test_upsample_then_decimate_roundtrip() {
        let f = 0.05; // mid-band
        let n = 512;
        let x = tone(f, n);
        for &factor in &[2usize, 4] {
            let up = upsample(&x, factor);
            assert_eq!(up.len(), n * factor);
            let down = decimate(&up, factor);
            assert_eq!(down.len(), n);
            // Compare away from the edges.
            for i in 100..n - 100 {
                assert!(
                    (down[i] - x[i]).abs() < 1e-6,
                    "factor {factor} at {i}: {} vs {}",
                    down[i],
                    x[i]
                );
            }
        }
    }

    #[test]
    fn test_upsample_preserves_tone() {
        let f = 0.1;
        let n = 256;
        let factor = 4;
        let up = upsample(&tone(f, n), factor);
        // Tone moves to f/factor; interior samples match the fine tone.
        for (i, &v) in up.iter().enumerate().take(up.len() - 200).skip(200) {
            let expect = (TWO_PI * f / factor as f64 * i as f64).sin();
            assert!((v - expect).abs() < 1e-5, "at {i}: {v} vs {expect}");
        }
    }

    #[test]
    fn test_decimate_kills_alias_band() {
        // Tone above the post-decimation Nyquist must be attenuated > 60 dB.
        let factor = 4;
        let n = 2048;
        let f_alias = 0.2; // above 0.5/4 = 0.125
        let x = tone(f_alias, n);
        let y = decimate(&x, factor);
        // Its alias lands at |0.2·4 − 1| = 0.2 (normalized to the new rate).
        let alias_mag = spectrum_mag_at(&y[32..y.len() - 32], 0.2);
        assert!(
            20.0 * alias_mag.log10() < -60.0,
            "alias at {} dB",
            20.0 * alias_mag.log10()
        );
        // A tone below the new Nyquist survives intact.
        let x_ok = tone(0.02, n);
        let y_ok = decimate(&x_ok, factor);
        let keep_mag = spectrum_mag_at(&y_ok[32..y_ok.len() - 32], 0.08);
        assert!((keep_mag - 1.0).abs() < 0.01, "kept tone {keep_mag}");
    }

    #[test]
    fn test_resample_rational_3_2() {
        let f = 0.04;
        let n = 600;
        let x = tone(f, n);
        let y = resample_rational(&x, 3, 2);
        assert_eq!(y.len(), n * 3 / 2);
        // New tone frequency: f·2/3.
        for (i, &v) in y.iter().enumerate().take(y.len() - 200).skip(200) {
            let expect = (TWO_PI * f * 2.0 / 3.0 * i as f64).sin();
            assert!((v - expect).abs() < 1e-4, "at {i}: {v} vs {expect}");
        }
    }

    #[test]
    fn test_resample_to_rate() {
        let x = tone(0.03, 500);
        let y = resample_to_rate(&x, 1000.0, 1500.0);
        assert_eq!(y.len(), 750);
        let z = resample_to_rate(&x, 44100.0, 44100.0);
        assert_eq!(z.len(), 500);
    }

    #[test]
    fn test_sinc_interpolation_accuracy() {
        let f = 0.07;
        let x = tone(f, 200);
        // Interpolate between samples: matches the continuous tone.
        for &t in &[50.25, 80.5, 120.75] {
            let v = sinc_interpolate(&x, t, 16);
            let expect = (TWO_PI * f * t).sin();
            assert!((v - expect).abs() < 1e-4, "t={t}: {v} vs {expect}");
        }
        let y = resample_sinc(&x, 1.5, 16);
        assert_eq!(y.len(), 300);
        for (i, &v) in y.iter().enumerate().take(240).skip(60) {
            let expect = (TWO_PI * f * i as f64 / 1.5).sin();
            assert!((v - expect).abs() < 1e-3);
        }
    }

    #[test]
    fn test_linear_and_cubic() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // A straight line resamples exactly under both schemes.
        let lin = resample_linear(&x, 2.0);
        let cub = resample_cubic(&x, 2.0);
        for i in 4..190 {
            let expect = i as f64 / 2.0;
            assert!((lin[i] - expect).abs() < 1e-12);
            assert!((cub[i] - expect).abs() < 1e-9);
        }
        // Cubic beats linear on a curved signal.
        let f = 0.02;
        let s = tone(f, 200);
        let ls = resample_linear(&s, 3.0);
        let cs = resample_cubic(&s, 3.0);
        let mut el = 0.0;
        let mut ec = 0.0;
        for i in 30..570 {
            let expect = (TWO_PI * f * i as f64 / 3.0).sin();
            el += (ls[i] - expect).powi(2);
            ec += (cs[i] - expect).powi(2);
        }
        assert!(ec < el / 10.0, "cubic {ec} vs linear {el}");
    }

    #[test]
    fn test_cic_dc_gain_and_smoothing() {
        let x = vec![1.0; 512];
        let y = cic_decimate(&x, 8, 3);
        assert_eq!(y.len(), 64);
        // DC passes at unit gain once the pipeline fills.
        for &v in &y[8..] {
            assert!((v - 1.0).abs() < 1e-12, "{v}");
        }
        // High-frequency tone strongly attenuated.
        let hf = tone(0.45, 512);
        let yh = cic_decimate(&hf, 8, 3);
        let peak = yh[8..].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(peak < 1e-2, "CIC leak {peak}");
    }

    #[test]
    fn test_half_band_structure() {
        let h = half_band_filter(31);
        let c = 15;
        assert!((h[c] - 0.5).abs() < 1e-12);
        for k in (1..15).step_by(2) {
            assert_eq!(h[c + k + 1], 0.0);
            assert_eq!(h[c - k - 1], 0.0);
        }
        // Symmetric, DC gain 1, and |H(0.25)| = 0.5 (half-band point).
        let dc: f64 = h.iter().sum();
        assert!((dc - 1.0).abs() < 1e-12);
        let mut re = 0.0;
        let mut im = 0.0;
        for (k, &v) in h.iter().enumerate() {
            let ang = TWO_PI * 0.25 * k as f64;
            re += v * ang.cos();
            im -= v * ang.sin();
        }
        let mag = (re * re + im * im).sqrt();
        assert!((mag - 0.5).abs() < 1e-9, "quarter-band gain {mag}");
    }
}
