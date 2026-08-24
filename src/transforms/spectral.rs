//! Spectral estimation: periodogram, Welch averaging, multitaper (DPSS),
//! parametric AR models (Burg, Yule-Walker), MUSIC, cross-spectra,
//! coherence, cepstra, Lomb-Scargle, and spectrum descriptors.
//!
//! All PSDs are one-sided densities in units²/Hz: integrating them over
//! frequency (trapezoid over the returned grid) recovers the signal's
//! variance/power.

use crate::dsp::windows::{window, WindowKind};
use crate::fractals::Complex;
use crate::linalg::{eigen_symmetric, eigen_symmetric_tridiagonal, solve, Matrix};
use crate::math::constants::PI;
use crate::transforms::fft::rfft;

const TWO_PI: f64 = 2.0 * PI;

fn one_sided_freqs(n: usize, fs: f64) -> Vec<f64> {
    (0..=n / 2).map(|k| k as f64 * fs / n as f64).collect()
}

/// Windowed periodogram: (frequencies, one-sided PSD).
///
/// # Panics
/// Panics on an empty signal.
#[must_use]
pub fn periodogram(x: &[f64], fs: f64, window_kind: WindowKind) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    assert!(n > 0, "periodogram needs samples");
    let w = window(window_kind, n, true);
    let wss: f64 = w.iter().map(|v| v * v).sum();
    let tapered: Vec<f64> = x.iter().zip(&w).map(|(v, wv)| v * wv).collect();
    let spec = rfft(&tapered);
    let scale = 1.0 / (fs * wss);
    let psd: Vec<f64> = spec
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let mut p = c.norm_sq() * scale;
            if k != 0 && !(n.is_multiple_of(2) && k == n / 2) {
                p *= 2.0;
            }
            p
        })
        .collect();
    (one_sided_freqs(n, fs), psd)
}

/// Welch's method: average windowed periodograms of overlapping
/// segments (detrended by segment mean removal).
///
/// # Panics
/// Panics unless `0 < noverlap < nperseg <= x.len()`.
#[must_use]
pub fn welch(
    x: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: usize,
    window_kind: WindowKind,
) -> (Vec<f64>, Vec<f64>) {
    assert!(nperseg > 0 && nperseg <= x.len(), "invalid segment length");
    assert!(noverlap < nperseg, "overlap must be below the segment length");
    let w = window(window_kind, nperseg, true);
    let wss: f64 = w.iter().map(|v| v * v).sum();
    let step = nperseg - noverlap;
    let n_bins = nperseg / 2 + 1;
    let mut acc = vec![0.0; n_bins];
    let mut count = 0usize;
    let mut start = 0usize;
    while start + nperseg <= x.len() {
        let seg = &x[start..start + nperseg];
        let mean = seg.iter().sum::<f64>() / nperseg as f64;
        let tapered: Vec<f64> = seg.iter().zip(&w).map(|(v, wv)| (v - mean) * wv).collect();
        let spec = rfft(&tapered);
        for (k, c) in spec.iter().enumerate() {
            let mut p = c.norm_sq();
            if k != 0 && !(nperseg.is_multiple_of(2) && k == nperseg / 2) {
                p *= 2.0;
            }
            acc[k] += p;
        }
        count += 1;
        start += step;
    }
    let scale = 1.0 / (fs * wss * count.max(1) as f64);
    for v in acc.iter_mut() {
        *v *= scale;
    }
    (one_sided_freqs(nperseg, fs), acc)
}

/// Discrete prolate spheroidal (Slepian) sequences: the first k tapers
/// of length n at time-bandwidth product nw, from the tridiagonal
/// eigenproblem. Each taper has unit energy; sign convention: positive
/// mean (even tapers) / positive first lag (odd tapers).
///
/// # Panics
/// Panics if `k == 0`, `k > n`, or the eigen solve fails.
#[must_use]
pub fn dpss(n: usize, nw: f64, k: usize) -> Vec<Vec<f64>> {
    assert!(k > 0 && k <= n, "need 0 < k <= n tapers");
    let w = nw / n as f64;
    let diag: Vec<f64> = (0..n)
        .map(|i| ((n as f64 - 1.0 - 2.0 * i as f64) / 2.0).powi(2) * (TWO_PI * w).cos())
        .collect();
    let off: Vec<f64> = (0..n - 1)
        .map(|i| (i + 1) as f64 * (n - 1 - i) as f64 / 2.0)
        .collect();
    let (_, vectors) =
        eigen_symmetric_tridiagonal(&diag, &off).expect("DPSS eigen solve failed");
    // Eigenvalues ascending: the k most concentrated tapers are the last k.
    (0..k)
        .map(|j| {
            let mut v: Vec<f64> = vectors[n - 1 - j].clone();
            let norm: f64 = v.iter().map(|u| u * u).sum::<f64>().sqrt();
            for u in v.iter_mut() {
                *u /= norm;
            }
            // Fix sign: positive mean, else positive leading slope.
            let mean: f64 = v.iter().sum();
            let flip = if mean.abs() > 1e-9 {
                mean < 0.0
            } else {
                v[1] - v[0] < 0.0
            };
            if flip {
                for u in v.iter_mut() {
                    *u = -*u;
                }
            }
            v
        })
        .collect()
}

/// Thomson multitaper PSD estimate with k DPSS tapers.
#[must_use]
pub fn multitaper(x: &[f64], fs: f64, nw: f64, k: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let tapers = dpss(n, nw, k);
    let n_bins = n / 2 + 1;
    let mut acc = vec![0.0; n_bins];
    for taper in &tapers {
        let tapered: Vec<f64> = x.iter().zip(taper).map(|(v, w)| v * w).collect();
        let spec = rfft(&tapered);
        for (kb, c) in spec.iter().enumerate() {
            let mut p = c.norm_sq();
            if kb != 0 && !(n.is_multiple_of(2) && kb == n / 2) {
                p *= 2.0;
            }
            acc[kb] += p;
        }
    }
    let scale = 1.0 / (fs * k as f64);
    for v in acc.iter_mut() {
        *v *= scale;
    }
    (one_sided_freqs(n, fs), acc)
}

/// Burg's method AR(p) fit: returns (a, σ²) for the model
/// x\[n\] = Σ a\[k\]·x\[n−1−k\] + e\[n\] with prediction-error variance σ².
///
/// # Panics
/// Panics unless `0 < order < x.len()`.
#[must_use]
pub fn burg_ar(x: &[f64], order: usize) -> (Vec<f64>, f64) {
    let n = x.len();
    assert!(order > 0 && order < n, "order must be in 1..n");
    let mut f: Vec<f64> = x.to_vec(); // forward errors
    let mut b: Vec<f64> = x.to_vec(); // backward errors
    let mut a = vec![0.0; order]; // a[k] multiplies x[n-1-k]
    let mut e: f64 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    for m in 0..order {
        // Reflection coefficient.
        let mut num = 0.0;
        let mut den = 0.0;
        for i in m + 1..n {
            num += f[i] * b[i - 1];
            den += f[i] * f[i] + b[i - 1] * b[i - 1];
        }
        let kref = if den.abs() < 1e-300 { 0.0 } else { 2.0 * num / den };
        // Update AR coefficients (Levinson recursion on -a).
        let prev = a.clone();
        a[m] = kref;
        for k in 0..m {
            a[k] = prev[k] - kref * prev[m - 1 - k];
        }
        // Update the error sequences.
        for i in (m + 1..n).rev() {
            let fi = f[i];
            let bi = b[i - 1];
            f[i] = fi - kref * bi;
            b[i] = bi - kref * fi;
        }
        e *= 1.0 - kref * kref;
    }
    (a, e)
}

fn autocorr(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    (0..=max_lag)
        .map(|lag| {
            let mut acc = 0.0;
            for i in lag..n {
                acc += x[i] * x[i - lag];
            }
            acc / n as f64
        })
        .collect()
}

/// Yule-Walker AR(p) fit via the autocorrelation method (Levinson-style
/// dense solve): same conventions as [`burg_ar`].
///
/// # Panics
/// Panics unless `0 < order < x.len()` and the autocorrelation system is
/// nonsingular.
#[must_use]
pub fn yule_walker_ar(x: &[f64], order: usize) -> (Vec<f64>, f64) {
    let n = x.len();
    assert!(order > 0 && order < n, "order must be in 1..n");
    let r = autocorr(x, order);
    let mut m = Matrix::zeros(order, order);
    for i in 0..order {
        for j in 0..order {
            m.set(i, j, r[(i as isize - j as isize).unsigned_abs()]);
        }
    }
    let rhs: Vec<f64> = (1..=order).map(|i| r[i]).collect();
    let a = solve(&m, &rhs).expect("Yule-Walker system singular");
    let sigma2 = r[0] - a.iter().zip(&rhs).map(|(ai, ri)| ai * ri).sum::<f64>();
    (a, sigma2.max(0.0))
}

/// One-sided PSD of an AR model (a, σ²) on n frequency points up to
/// Nyquist: σ²/(fs·|1 − Σ a\[k\] e^(−jω(k+1))|²), doubled off DC/Nyquist.
#[must_use]
pub fn ar_psd(coeffs: &[f64], sigma2: f64, fs: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let freqs: Vec<f64> = (0..n).map(|i| fs / 2.0 * i as f64 / (n - 1).max(1) as f64).collect();
    let psd: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            let omega = TWO_PI * f / fs;
            let mut den = Complex::new(1.0, 0.0);
            for (k, &a) in coeffs.iter().enumerate() {
                let ang = -omega * (k + 1) as f64;
                den = den - Complex::new(a * ang.cos(), a * ang.sin());
            }
            let mut p = sigma2 / (fs * den.norm_sq());
            if f > 0.0 && f < fs / 2.0 {
                p *= 2.0;
            }
            p
        })
        .collect();
    (freqs, psd)
}

/// MUSIC pseudospectrum for real sinusoids: correlation matrix of
/// dimension `order`, signal subspace of dimension 2·n_sources, and the
/// noise-subspace projection evaluated at n frequencies up to Nyquist.
///
/// # Panics
/// Panics unless `2*n_sources < order < x.len()`.
#[must_use]
pub fn music(
    x: &[f64],
    n_sources: usize,
    order: usize,
    fs: f64,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    assert!(2 * n_sources < order && order < x.len(), "need 2*n_sources < order < len");
    // Sample correlation matrix (forward windows).
    let mut r = Matrix::zeros(order, order);
    let count = x.len() - order + 1;
    for s in 0..count {
        for i in 0..order {
            for j in 0..order {
                r.set(i, j, r.get(i, j) + x[s + i] * x[s + j]);
            }
        }
    }
    let eig = eigen_symmetric(&r.scale(1.0 / count as f64), 1e-12, 200)
        .expect("MUSIC eigen solve failed");
    let mut idx: Vec<usize> = (0..order).collect();
    idx.sort_by(|&a, &b| {
        eig.values[a]
            .partial_cmp(&eig.values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let noise_cols: Vec<usize> = idx[..order - 2 * n_sources].to_vec();
    let freqs: Vec<f64> = (0..n).map(|i| fs / 2.0 * i as f64 / (n - 1).max(1) as f64).collect();
    let ps: Vec<f64> = freqs
        .iter()
        .map(|&f| {
            let omega = TWO_PI * f / fs;
            let mut denom = 0.0;
            for &c in &noise_cols {
                let mut dot = Complex::new(0.0, 0.0);
                for i in 0..order {
                    let ang = -omega * i as f64;
                    let v = eig.vectors.get(i, c);
                    dot = dot + Complex::new(v * ang.cos(), v * ang.sin());
                }
                denom += dot.norm_sq();
            }
            1.0 / denom.max(1e-300)
        })
        .collect();
    (freqs, ps)
}

/// Welch-averaged cross-spectral density S_xy(f) = E\[X*(f)·Y(f)\]
/// (Hann window, 50% overlap): (frequencies, complex CSD).
///
/// # Panics
/// Panics unless both signals have at least `nperseg` samples.
#[must_use]
pub fn cross_spectral_density(
    x: &[f64],
    y: &[f64],
    fs: f64,
    nperseg: usize,
) -> (Vec<f64>, Vec<Complex>) {
    let n = x.len().min(y.len());
    assert!(nperseg > 0 && nperseg <= n, "invalid segment length");
    let w = window(WindowKind::Hann, nperseg, true);
    let wss: f64 = w.iter().map(|v| v * v).sum();
    let step = (nperseg / 2).max(1);
    let n_bins = nperseg / 2 + 1;
    let mut acc = vec![Complex::new(0.0, 0.0); n_bins];
    let mut count = 0usize;
    let mut start = 0usize;
    while start + nperseg <= n {
        let tx: Vec<f64> = x[start..start + nperseg].iter().zip(&w).map(|(v, wv)| v * wv).collect();
        let ty: Vec<f64> = y[start..start + nperseg].iter().zip(&w).map(|(v, wv)| v * wv).collect();
        let sx = rfft(&tx);
        let sy = rfft(&ty);
        for (k, slot) in acc.iter_mut().enumerate() {
            let mut c = sx[k].conjugate() * sy[k];
            if k != 0 && !(nperseg.is_multiple_of(2) && k == nperseg / 2) {
                c = Complex::new(2.0 * c.re, 2.0 * c.im);
            }
            *slot = *slot + c;
        }
        count += 1;
        start += step;
    }
    let scale = 1.0 / (fs * wss * count.max(1) as f64);
    for v in acc.iter_mut() {
        *v = Complex::new(v.re * scale, v.im * scale);
    }
    (one_sided_freqs(nperseg, fs), acc)
}

/// Magnitude-squared coherence |S_xy|²/(S_xx·S_yy) on the Welch grid.
#[must_use]
pub fn coherence(x: &[f64], y: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let (freqs, sxy) = cross_spectral_density(x, y, fs, nperseg);
    let (_, sxx) = welch(x, fs, nperseg, nperseg / 2, WindowKind::Hann);
    let (_, syy) = welch(y, fs, nperseg, nperseg / 2, WindowKind::Hann);
    let coh = sxy
        .iter()
        .zip(sxx.iter().zip(&syy))
        .map(|(pxy, (&pxx, &pyy))| {
            let den = pxx * pyy;
            if den > 1e-300 {
                (pxy.norm_sq() / den).min(1.0)
            } else {
                0.0
            }
        })
        .collect();
    (freqs, coh)
}

/// H1 transfer-function estimate S_xy/S_xx from input to output.
#[must_use]
pub fn transfer_function_estimate(
    input: &[f64],
    output: &[f64],
    fs: f64,
    nperseg: usize,
) -> (Vec<f64>, Vec<Complex>) {
    let (freqs, sxy) = cross_spectral_density(input, output, fs, nperseg);
    let (_, sxx) = welch(input, fs, nperseg, nperseg / 2, WindowKind::Hann);
    let h = sxy
        .iter()
        .zip(&sxx)
        .map(|(pxy, &pxx)| {
            if pxx > 1e-300 {
                Complex::new(pxy.re / pxx, pxy.im / pxx)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();
    (freqs, h)
}

/// Real cepstrum: IFFT of log |X(f)| (real part).
#[must_use]
pub fn cepstrum_real(x: &[f64]) -> Vec<f64> {
    use crate::transforms::fft::{fft_any, ifft_any};
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let spec = fft_any(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
    let logmag: Vec<Complex> = spec
        .iter()
        .map(|c| Complex::new(c.norm().max(1e-300).ln(), 0.0))
        .collect();
    ifft_any(&logmag).iter().map(|c| c.re).collect()
}

/// Power cepstrum: IFFT of log |X(f)|², i.e. twice the real cepstrum.
#[must_use]
pub fn cepstrum_power(x: &[f64]) -> Vec<f64> {
    cepstrum_real(x).iter().map(|v| 2.0 * v).collect()
}

/// Lomb-Scargle normalized periodogram for unevenly sampled data at the
/// requested frequencies (Hz). Values are in the classical normalization
/// (power / 2σ²).
///
/// # Panics
/// Panics if `t` and `y` lengths differ or fewer than 2 samples.
#[must_use]
pub fn lomb_scargle(t: &[f64], y: &[f64], freqs: &[f64]) -> Vec<f64> {
    assert_eq!(t.len(), y.len(), "t and y must match");
    assert!(t.len() >= 2, "need at least two samples");
    let n = t.len();
    let mean = y.iter().sum::<f64>() / n as f64;
    let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - mean).collect();
    freqs
        .iter()
        .map(|&f| {
            if f == 0.0 {
                return 0.0;
            }
            let omega = TWO_PI * f;
            // τ from tan(2ωτ) = Σ sin 2ωt / Σ cos 2ωt.
            let (mut s2, mut c2) = (0.0, 0.0);
            for &ti in t {
                s2 += (2.0 * omega * ti).sin();
                c2 += (2.0 * omega * ti).cos();
            }
            let tau = s2.atan2(c2) / (2.0 * omega);
            let (mut cy, mut sy, mut cc, mut ss) = (0.0, 0.0, 0.0, 0.0);
            for (&ti, &yi) in t.iter().zip(&yc) {
                let arg = omega * (ti - tau);
                let (s, c) = arg.sin_cos();
                cy += yi * c;
                sy += yi * s;
                cc += c * c;
                ss += s * s;
            }
            let mut p = 0.0;
            if cc > 1e-300 {
                p += cy * cy / cc;
            }
            if ss > 1e-300 {
                p += sy * sy / ss;
            }
            p / (2.0 * var)
        })
        .collect()
}

/// Spectral entropy of a PSD, normalized to \[0, 1\].
#[must_use]
pub fn spectral_entropy(psd: &[f64]) -> f64 {
    let total: f64 = psd.iter().filter(|v| **v > 0.0).sum();
    if total <= 0.0 || psd.len() < 2 {
        return 0.0;
    }
    let mut h = 0.0;
    for &p in psd {
        if p > 0.0 {
            let q = p / total;
            h -= q * q.ln();
        }
    }
    h / (psd.len() as f64).ln()
}

/// Spectral flatness (Wiener entropy): geometric over arithmetic mean.
#[must_use]
pub fn spectral_flatness(psd: &[f64]) -> f64 {
    if psd.is_empty() {
        return 0.0;
    }
    let am: f64 = psd.iter().sum::<f64>() / psd.len() as f64;
    if am <= 0.0 {
        return 0.0;
    }
    let lg: f64 = psd.iter().map(|&p| p.max(1e-300).ln()).sum::<f64>() / psd.len() as f64;
    lg.exp() / am
}

/// Remove a least-squares polynomial trend of the given order.
///
/// # Panics
/// Panics if the fit system is singular (order too high for the data).
#[must_use]
pub fn detrend(x: &[f64], order: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n <= order {
        return vec![0.0; n];
    }
    let m = order + 1;
    // Normalized abscissa in [-1, 1] for conditioning.
    let ts: Vec<f64> = (0..n)
        .map(|i| 2.0 * i as f64 / (n - 1).max(1) as f64 - 1.0)
        .collect();
    let mut ata = Matrix::zeros(m, m);
    let mut rhs = vec![0.0; m];
    for (i, &t) in ts.iter().enumerate() {
        let basis: Vec<f64> = (0..m).map(|p| t.powi(p as i32)).collect();
        for a in 0..m {
            rhs[a] += x[i] * basis[a];
            for b in 0..m {
                ata.set(a, b, ata.get(a, b) + basis[a] * basis[b]);
            }
        }
    }
    let coef = solve(&ata, &rhs).expect("detrend normal equations singular");
    x.iter()
        .zip(&ts)
        .map(|(&v, &t)| {
            let fit: f64 = coef.iter().enumerate().map(|(p, &c)| c * t.powi(p as i32)).sum();
            v - fit
        })
        .collect()
}

/// Fit PSD ≈ A·f^(−α) over \[f_min, f_max\] by log-log linear regression;
/// returns (α, A).
#[must_use]
pub fn power_law_fit(f: &[f64], psd: &[f64], f_min: f64, f_max: f64) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut n = 0.0;
    for (&fi, &pi) in f.iter().zip(psd) {
        if fi >= f_min && fi <= f_max && fi > 0.0 && pi > 0.0 {
            let lx = fi.ln();
            let ly = pi.ln();
            sx += lx;
            sy += ly;
            sxx += lx * lx;
            sxy += lx * ly;
            n += 1.0;
        }
    }
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let intercept = (sy - slope * sx) / n;
    (-slope, intercept.exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn trapz(f: &[f64], y: &[f64]) -> f64 {
        f.windows(2)
            .zip(y.windows(2))
            .map(|(fw, yw)| 0.5 * (yw[0] + yw[1]) * (fw[1] - fw[0]))
            .sum()
    }

    #[test]
    fn test_periodogram_tone_power() {
        let fs = 256.0;
        let n = 1024;
        let a = 2.0;
        let x: Vec<f64> = (0..n).map(|i| a * (TWO_PI * 32.0 * i as f64 / fs).sin()).collect();
        let (f, p) = periodogram(&x, fs, WindowKind::Rect);
        // Total power = a²/2.
        let total = trapz(&f, &p);
        assert!(approx(total, a * a / 2.0, 0.01), "total {total}");
        // Peak at 32 Hz.
        let (k, _) = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(approx(f[k], 32.0, fs / n as f64));
    }

    #[test]
    fn test_welch_integrates_to_variance() {
        let mut rng = Rng::new(7);
        let n = 8192;
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let var = {
            let m = x.iter().sum::<f64>() / n as f64;
            x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64
        };
        let fs = 100.0;
        let (f, p) = welch(&x, fs, 256, 128, WindowKind::Hann);
        let total = trapz(&f, &p);
        assert!((total - var).abs() / var < 0.05, "{total} vs {var}");
    }

    #[test]
    fn test_dpss_orthonormal_and_concentrated() {
        let n = 128;
        let tapers = dpss(n, 3.0, 5);
        assert_eq!(tapers.len(), 5);
        for i in 0..5 {
            for j in 0..5 {
                let dot: f64 = tapers[i].iter().zip(&tapers[j]).map(|(a, b)| a * b).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(approx(dot, expect, 1e-8), "({i},{j}): {dot}");
            }
        }
        // First taper: bell-shaped, all positive, energy concentrated in band.
        assert!(tapers[0].iter().all(|&v| v > 0.0));
        let spec = rfft(&tapers[0]);
        let in_band: f64 = spec.iter().take(4).map(|c| c.norm_sq()).sum();
        let total: f64 = spec.iter().map(|c| c.norm_sq()).sum();
        assert!(in_band / total > 0.99, "concentration {}", in_band / total);
    }

    #[test]
    fn test_multitaper_tone() {
        let fs = 256.0;
        let n = 512;
        let x: Vec<f64> = (0..n).map(|i| (TWO_PI * 50.0 * i as f64 / fs).sin()).collect();
        let (f, p) = multitaper(&x, fs, 4.0, 7);
        let (k, _) = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(approx(f[k], 50.0, 2.0 * fs / n as f64), "peak at {}", f[k]);
    }

    #[test]
    fn test_burg_recovers_ar2() {
        // AR(2): x[n] = 0.75 x[n-1] - 0.5 x[n-2] + e.
        let mut rng = Rng::new(21);
        let n = 20000;
        let mut x = vec![0.0; n];
        for i in 2..n {
            let e = rng.next_f64() * 2.0 - 1.0;
            x[i] = 0.75 * x[i - 1] - 0.5 * x[i - 2] + e;
        }
        let (a, sigma2) = burg_ar(&x[1000..], 2);
        assert!(approx(a[0], 0.75, 0.02), "a1 {}", a[0]);
        assert!(approx(a[1], -0.5, 0.02), "a2 {}", a[1]);
        // Uniform noise variance = 1/3.
        assert!(approx(sigma2, 1.0 / 3.0, 0.02), "sigma2 {sigma2}");
        // Yule-Walker agrees.
        let (aw, _) = yule_walker_ar(&x[1000..], 2);
        assert!(approx(aw[0], 0.75, 0.03) && approx(aw[1], -0.5, 0.03));
    }

    #[test]
    fn test_ar_psd_peak() {
        // Resonant AR(2) has a spectral peak at its pole angle.
        let r = 0.95_f64;
        let f0 = 0.2_f64; // normalized (fs = 1)
        let a1 = 2.0 * r * (TWO_PI * f0).cos();
        let a2 = -r * r;
        let (f, p) = ar_psd(&[a1, a2], 1.0, 1.0, 512);
        let (k, _) = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(approx(f[k], f0, 0.01), "peak {}", f[k]);
    }

    #[test]
    fn test_music_resolves_close_tones() {
        // Two tones one DFT-bin apart for a 256-sample record.
        let fs = 256.0;
        let n = 256;
        let (f1, f2) = (60.0, 61.0);
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (TWO_PI * f1 * t).sin() + (TWO_PI * f2 * t).sin()
            })
            .collect();
        let (f, ps) = music(&x, 2, 40, fs, 2048);
        // Count peaks near f1 and f2: the pseudospectrum should have two
        // distinct local maxima in [55, 66].
        let mut peaks = Vec::new();
        for i in 1..ps.len() - 1 {
            if ps[i] > ps[i - 1] && ps[i] > ps[i + 1] && f[i] > 55.0 && f[i] < 66.0 {
                peaks.push(f[i]);
            }
        }
        assert!(peaks.len() >= 2, "peaks {peaks:?}");
        assert!(peaks.iter().any(|&p| (p - f1).abs() < 0.5));
        assert!(peaks.iter().any(|&p| (p - f2).abs() < 0.5));
    }

    #[test]
    fn test_coherence_and_transfer() {
        // y = filtered x + independent noise: high coherence in band.
        let mut rng = Rng::new(5);
        let n = 8192;
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        // Simple 2x gain with one-sample delay.
        let mut y = vec![0.0; n];
        for i in 1..n {
            y[i] = 2.0 * x[i - 1];
        }
        let (f, coh) = coherence(&x, &y, 100.0, 256);
        let mid = coh.len() / 2;
        assert!(coh[mid] > 0.99, "coherence {}", coh[mid]);
        let (_, h) = transfer_function_estimate(&x, &y, 100.0, 256);
        assert!(approx(h[mid].norm(), 2.0, 0.05), "gain {}", h[mid].norm());
        let _ = f;
    }

    #[test]
    fn test_cepstrum_echo_peak() {
        // A signal plus its echo has a cepstral peak at the echo delay.
        let mut rng = Rng::new(9);
        let n = 512;
        let delay = 50usize;
        let base: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| base[i] + if i >= delay { 0.8 * base[i - delay] } else { 0.0 })
            .collect();
        let cep = cepstrum_real(&x);
        // Look for the echo quefrency.
        let (k, _) = cep[10..n / 2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!((k + 10) as i64 - delay as i64 == 0 || ((k + 10) as i64 - delay as i64).abs() <= 1,
            "cepstral peak at {}", k + 10);
        let cp = cepstrum_power(&x);
        assert!(approx(cp[delay], 2.0 * cep[delay], 1e-12));
    }

    #[test]
    fn test_lomb_scargle_even_matches_periodogram_peak() {
        let fs = 64.0;
        let n = 256;
        let f0 = 10.0;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let y: Vec<f64> = t.iter().map(|&ti| (TWO_PI * f0 * ti).sin()).collect();
        let freqs: Vec<f64> = (1..128).map(|i| i as f64 * 0.25).collect();
        let p = lomb_scargle(&t, &y, &freqs);
        let (k, _) = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(approx(freqs[k], f0, 0.25), "peak {}", freqs[k]);
        // Classical normalization: peak ≈ n/2 · (a²/2)/σ² ≈ n/2 for unit sine.
        assert!(p[k] > 0.8 * n as f64 / 2.0, "peak power {}", p[k]);
        // Uneven sampling still finds the tone.
        let tu: Vec<f64> = t.iter().enumerate().map(|(i, &ti)| ti + 0.003 * ((i * 7919 % 100) as f64)).collect();
        let yu: Vec<f64> = tu.iter().map(|&ti| (TWO_PI * f0 * ti).sin()).collect();
        let pu = lomb_scargle(&tu, &yu, &freqs);
        let (ku, _) = pu.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!(approx(freqs[ku], f0, 0.25));
    }

    #[test]
    fn test_entropy_flatness_extremes() {
        // Flat PSD: entropy 1, flatness 1. Single line: entropy ~0, flatness ~0.
        let flat = vec![1.0; 64];
        assert!(approx(spectral_entropy(&flat), 1.0, 1e-12));
        assert!(approx(spectral_flatness(&flat), 1.0, 1e-12));
        let mut line = vec![0.0; 64];
        line[10] = 1.0;
        assert!(spectral_entropy(&line) < 1e-9);
        assert!(spectral_flatness(&line) < 1e-3);
    }

    #[test]
    fn test_detrend_removes_polynomial() {
        let n = 200;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                3.0 + 0.1 * t - 0.002 * t * t + (t * 0.5).sin()
            })
            .collect();
        let d = detrend(&x, 2);
        // What remains should be the sine (zero-mean-ish, bounded by ~1.2).
        let mean = d.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-8, "mean {mean}");
        assert!(d.iter().all(|v| v.abs() < 1.3));
        let energy: f64 = d.iter().map(|v| v * v).sum::<f64>() / n as f64;
        assert!(approx(energy, 0.5, 0.1), "sine power {energy}");
    }

    #[test]
    fn test_power_law_fit() {
        let f: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.1).collect();
        let psd: Vec<f64> = f.iter().map(|&fi| 3.0 * fi.powf(-1.7)).collect();
        let (alpha, a) = power_law_fit(&f, &psd, 0.5, 80.0);
        assert!(approx(alpha, 1.7, 1e-9));
        assert!(approx(a, 3.0, 1e-9));
    }
}
