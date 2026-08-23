//! FIR filter design and application.
//!
//! All frequencies are normalized to the sample rate (cycles/sample), so
//! cutoffs live in (0, 0.5). Designs are linear-phase; windowed-sinc
//! designs follow Oppenheim & Schafer §7.5, the equiripple design is the
//! Parks-McClellan / Remez exchange (type I), and Savitzky-Golay follows
//! the least-squares polynomial derivation.

use crate::dsp::windows::{kaiser_beta_for_attenuation, window, WindowKind};
use crate::error::SolveError;
use crate::fractals::Complex;
use crate::linalg::{solve, Matrix};
use crate::math::constants::PI;
use crate::transforms::fft::{fft, ifft, next_power_of_two};

const TWO_PI: f64 = 2.0 * PI;

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn assert_cutoff(c: f64) {
    assert!(c > 0.0 && c < 0.5, "cutoff must be in (0, 0.5), got {c}");
}

/// Windowed-sinc low-pass FIR; unit DC gain. `cutoff` in (0, 0.5).
///
/// # Panics
/// Panics if `n_taps == 0` or the cutoff is out of range.
#[must_use]
pub fn fir_lowpass(n_taps: usize, cutoff: f64, w: WindowKind) -> Vec<f64> {
    assert!(n_taps > 0, "n_taps must be positive");
    assert_cutoff(cutoff);
    let win = window(w, n_taps, false);
    let c = (n_taps - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..n_taps)
        .map(|k| 2.0 * cutoff * sinc(2.0 * cutoff * (k as f64 - c)) * win[k])
        .collect();
    let dc: f64 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= dc;
    }
    h
}

/// Windowed-sinc high-pass FIR via spectral inversion; unit Nyquist gain.
///
/// # Panics
/// Panics unless `n_taps` is odd (type I linear phase is required for a
/// high-pass) and the cutoff is in range.
#[must_use]
pub fn fir_highpass(n_taps: usize, cutoff: f64, w: WindowKind) -> Vec<f64> {
    assert!(n_taps % 2 == 1, "fir_highpass requires an odd tap count");
    assert_cutoff(cutoff);
    let lp = fir_lowpass(n_taps, cutoff, w);
    let c = n_taps / 2;
    let mut h: Vec<f64> = lp.iter().map(|&v| -v).collect();
    h[c] += 1.0;
    // Normalize the Nyquist gain Σ (−1)^k h[k] to one.
    let g: f64 = h.iter().enumerate().map(|(k, &v)| if k % 2 == 0 { v } else { -v }).sum();
    for v in h.iter_mut() {
        *v /= g.abs();
    }
    h
}

/// Windowed-sinc band-pass FIR (difference of two low-passes); unit gain
/// at the band center (lo + hi)/2.
///
/// # Panics
/// Panics unless `0 < lo < hi < 0.5`.
#[must_use]
pub fn fir_bandpass(n_taps: usize, lo: f64, hi: f64, w: WindowKind) -> Vec<f64> {
    assert!(lo < hi, "band edges must satisfy lo < hi");
    assert_cutoff(lo);
    assert_cutoff(hi);
    let win = window(w, n_taps, false);
    let c = (n_taps - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..n_taps)
        .map(|k| {
            let t = k as f64 - c;
            (2.0 * hi * sinc(2.0 * hi * t) - 2.0 * lo * sinc(2.0 * lo * t)) * win[k]
        })
        .collect();
    let fm = 0.5 * (lo + hi);
    let g = magnitude_at(&h, fm);
    for v in h.iter_mut() {
        *v /= g;
    }
    h
}

/// Windowed-sinc band-stop FIR; unit DC gain.
///
/// # Panics
/// Panics unless `n_taps` is odd and `0 < lo < hi < 0.5`.
#[must_use]
pub fn fir_bandstop(n_taps: usize, lo: f64, hi: f64, w: WindowKind) -> Vec<f64> {
    assert!(n_taps % 2 == 1, "fir_bandstop requires an odd tap count");
    let bp = fir_bandpass(n_taps, lo, hi, w);
    let c = n_taps / 2;
    let mut h: Vec<f64> = bp.iter().map(|&v| -v).collect();
    h[c] += 1.0;
    let dc: f64 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= dc;
    }
    h
}

/// Kaiser-window low-pass design from a passband/stopband spec:
/// passband edge, stopband edge (normalized), maximum passband ripple
/// and minimum stopband attenuation in dB. Chooses the tap count and β
/// by Kaiser's formulas.
///
/// # Panics
/// Panics unless `0 < pass < stop < 0.5`.
#[must_use]
pub fn fir_kaiser_design(pass: f64, stop: f64, ripple_db: f64, atten_db: f64) -> Vec<f64> {
    assert!(pass < stop, "passband edge must be below stopband edge");
    assert_cutoff(pass);
    assert_cutoff(stop);
    // Effective attenuation from the tighter of the two ripple specs.
    let delta_p = (10.0_f64.powf(ripple_db.abs() / 20.0) - 1.0)
        / (10.0_f64.powf(ripple_db.abs() / 20.0) + 1.0);
    let delta_s = 10.0_f64.powf(-atten_db.abs() / 20.0);
    let delta = delta_p.min(delta_s);
    let a = -20.0 * delta.log10();
    let df = stop - pass;
    let mut n_taps = if a > 21.0 {
        ((a - 7.95) / (14.36 * df)).ceil() as usize + 1
    } else {
        (0.9222 / df).ceil() as usize + 1
    };
    if n_taps % 2 == 0 {
        n_taps += 1;
    }
    let beta = kaiser_beta_for_attenuation(a);
    fir_lowpass(n_taps, 0.5 * (pass + stop), WindowKind::Kaiser(beta))
}

/// Equiripple (Parks-McClellan / Remez exchange) linear-phase type I
/// design. `bands` are disjoint ascending (lo, hi) pairs in [0, 0.5];
/// `desired` and `weights` give one amplitude and weight per band.
///
/// # Errors
/// Returns `SolveError::InvalidArgument` for a malformed spec and
/// `SolveError::NoConvergence` if the exchange fails to settle.
///
/// # Panics
/// Panics unless `n_taps` is odd and ≥ 3.
pub fn fir_parks_mcclellan(
    n_taps: usize,
    bands: &[(f64, f64)],
    desired: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, SolveError> {
    assert!(n_taps >= 3 && n_taps % 2 == 1, "type I design needs an odd tap count >= 3");
    if bands.is_empty() || bands.len() != desired.len() || bands.len() != weights.len() {
        return Err(SolveError::InvalidArgument("bands/desired/weights lengths must match"));
    }
    let mut prev_hi = -1.0;
    for &(lo, hi) in bands {
        if !(0.0..=0.5).contains(&lo) || !(0.0..=0.5).contains(&hi) || lo >= hi || lo <= prev_hi {
            return Err(SolveError::InvalidArgument("bands must be ascending, disjoint, in [0, 0.5]"));
        }
        prev_hi = hi;
    }

    let r = (n_taps - 1) / 2; // cosine polynomial degree
    let n_ext = r + 2;

    // Dense frequency grid with per-band desired/weight, band edges included.
    let grid_density = 48;
    let total_width: f64 = bands.iter().map(|&(lo, hi)| hi - lo).sum();
    let mut grid: Vec<(f64, f64, f64)> = Vec::new();
    let mut band_of: Vec<usize> = Vec::new();
    for (b, &(lo, hi)) in bands.iter().enumerate() {
        let pts = (((hi - lo) / total_width) * (grid_density * (r + 1)) as f64).ceil() as usize;
        let pts = pts.max(4);
        for i in 0..=pts {
            let f = lo + (hi - lo) * i as f64 / pts as f64;
            grid.push((f, desired[b], weights[b]));
            band_of.push(b);
        }
    }
    let ng = grid.len();
    if ng < n_ext {
        return Err(SolveError::InvalidArgument("grid too coarse for the requested order"));
    }

    // Initial extremal guess: evenly spread over the grid.
    let mut ext: Vec<usize> = (0..n_ext).map(|i| i * (ng - 1) / (n_ext - 1)).collect();

    let mut delta = 0.0;
    let mut converged = false;
    for _iter in 0..250 {
        // Barycentric weights over the extremal set (x = cos 2πf).
        let x: Vec<f64> = ext.iter().map(|&i| (TWO_PI * grid[i].0).cos()).collect();
        let gamma: Vec<f64> = (0..n_ext)
            .map(|k| {
                let mut prod = 1.0;
                for j in 0..n_ext {
                    if j != k {
                        prod *= x[k] - x[j];
                    }
                }
                1.0 / prod
            })
            .collect();
        let num: f64 = (0..n_ext).map(|k| gamma[k] * grid[ext[k]].1).sum();
        let den: f64 = (0..n_ext)
            .map(|k| {
                let s = if k % 2 == 0 { 1.0 } else { -1.0 };
                s * gamma[k] / grid[ext[k]].2
            })
            .sum();
        delta = num / den;

        // Values the polynomial takes at the first r+1 extremals.
        let c: Vec<f64> = (0..=r)
            .map(|k| {
                let s = if k % 2 == 0 { 1.0 } else { -1.0 };
                grid[ext[k]].1 - s * delta / grid[ext[k]].2
            })
            .collect();
        // Second-kind barycentric weights on those r+1 nodes.
        let bw: Vec<f64> = (0..=r)
            .map(|k| {
                let mut prod = 1.0;
                for j in 0..=r {
                    if j != k {
                        prod *= x[k] - x[j];
                    }
                }
                1.0 / prod
            })
            .collect();
        let a_of = |xf: f64| -> f64 {
            let mut num = 0.0;
            let mut den = 0.0;
            for k in 0..=r {
                let dx = xf - x[k];
                if dx.abs() < 1e-14 {
                    return c[k];
                }
                let t = bw[k] / dx;
                num += t * c[k];
                den += t;
            }
            num / den
        };

        // Weighted error over the grid.
        let err: Vec<f64> = grid
            .iter()
            .map(|&(f, d, wt)| wt * (a_of((TWO_PI * f).cos()) - d))
            .collect();

        // Candidate extremals: local maxima of |E| within each band
        // (band edges count when |E| peaks there).
        let mut cand: Vec<usize> = Vec::new();
        for i in 0..ng {
            let e = err[i].abs();
            let lower = if i > 0 && band_of[i - 1] == band_of[i] {
                err[i - 1].abs()
            } else {
                f64::MIN
            };
            let upper = if i + 1 < ng && band_of[i + 1] == band_of[i] {
                err[i + 1].abs()
            } else {
                f64::MIN
            };
            if (e >= lower && e > upper) || (e > lower && e >= upper) {
                cand.push(i);
            }
        }
        // Enforce sign alternation, keeping the larger error on conflicts.
        let mut alt: Vec<usize> = Vec::new();
        for &i in &cand {
            if let Some(&last) = alt.last() {
                if (err[last] >= 0.0) == (err[i] >= 0.0) {
                    if err[i].abs() > err[last].abs() {
                        *alt.last_mut().unwrap() = i;
                    }
                    continue;
                }
            }
            alt.push(i);
        }
        // Trim to n_ext keeping the largest errors at the outside.
        while alt.len() > n_ext {
            let first = err[alt[0]].abs();
            let last = err[*alt.last().unwrap()].abs();
            if first <= last {
                alt.remove(0);
            } else {
                alt.pop();
            }
        }
        if alt.len() < n_ext {
            return Err(SolveError::NoConvergence { iters: _iter, residual: delta.abs() });
        }

        let max_err = alt.iter().map(|&i| err[i].abs()).fold(0.0_f64, f64::max);
        let settled = alt == ext;
        ext = alt;
        if settled || (max_err - delta.abs()).abs() <= 1e-6 * delta.abs().max(1e-12) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SolveError::NoConvergence { iters: 250, residual: delta.abs() });
    }

    // Final polynomial from the converged extremal set.
    let x: Vec<f64> = ext.iter().map(|&i| (TWO_PI * grid[i].0).cos()).collect();
    let c: Vec<f64> = (0..=r)
        .map(|k| {
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            grid[ext[k]].1 - s * delta / grid[ext[k]].2
        })
        .collect();
    let bw: Vec<f64> = (0..=r)
        .map(|k| {
            let mut prod = 1.0;
            for j in 0..=r {
                if j != k {
                    prod *= x[k] - x[j];
                }
            }
            1.0 / prod
        })
        .collect();
    let a_of = |xf: f64| -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 0..=r {
            let dx = xf - x[k];
            if dx.abs() < 1e-14 {
                return c[k];
            }
            let t = bw[k] / dx;
            num += t * c[k];
            den += t;
        }
        num / den
    };

    // Sample A at n_taps points and invert the cosine sum for h.
    let samples: Vec<f64> = (0..=r)
        .map(|j| a_of((TWO_PI * j as f64 / n_taps as f64).cos()))
        .collect();
    let mut h = vec![0.0; n_taps];
    let center = r;
    for m in 0..=r {
        let mut acc = samples[0];
        for (k, &s) in samples.iter().enumerate().skip(1) {
            acc += 2.0 * s * (TWO_PI * (k * m) as f64 / n_taps as f64).cos();
        }
        acc /= n_taps as f64;
        h[center + m] = acc;
        h[center - m] = acc;
    }
    Ok(h)
}

/// Least-squares linear-phase type I design over the given bands
/// (transition regions are "don't care").
///
/// # Panics
/// Panics unless `n_taps` is odd and the spec lengths match.
#[must_use]
pub fn fir_least_squares(n_taps: usize, bands: &[(f64, f64)], desired: &[f64]) -> Vec<f64> {
    assert!(n_taps % 2 == 1, "type I design needs an odd tap count");
    assert_eq!(bands.len(), desired.len(), "one desired value per band");
    let r = (n_taps - 1) / 2;
    // Dense grid.
    let mut grid: Vec<(f64, f64)> = Vec::new();
    for (b, &(lo, hi)) in bands.iter().enumerate() {
        let pts = (((hi - lo) * 40.0 * (r + 1) as f64).ceil() as usize).max(8);
        for i in 0..=pts {
            grid.push((lo + (hi - lo) * i as f64 / pts as f64, desired[b]));
        }
    }
    // Normal equations for A(f) = Σ b_k cos(2πkf).
    let m = r + 1;
    let mut q = Matrix::zeros(m, m);
    let mut rhs = vec![0.0; m];
    for &(f, d) in &grid {
        let basis: Vec<f64> = (0..m).map(|k| (TWO_PI * k as f64 * f).cos()).collect();
        for i in 0..m {
            rhs[i] += d * basis[i];
            for j in 0..m {
                q.set(i, j, q.get(i, j) + basis[i] * basis[j]);
            }
        }
    }
    let b = solve(&q, &rhs).expect("least-squares normal equations are singular");
    let mut h = vec![0.0; n_taps];
    h[r] = b[0];
    for k in 1..m {
        h[r + k] = b[k] / 2.0;
        h[r - k] = b[k] / 2.0;
    }
    h
}

/// Windowed ideal differentiator (antisymmetric, Blackman window). The
/// output of `fir_apply` approximates dx/dn (per-sample derivative)
/// delayed by (n_taps−1)/2.
///
/// # Panics
/// Panics unless `n_taps` is odd.
#[must_use]
pub fn fir_differentiator(n_taps: usize) -> Vec<f64> {
    assert!(n_taps % 2 == 1, "fir_differentiator requires an odd tap count");
    let c = (n_taps / 2) as isize;
    let win = window(WindowKind::Blackman, n_taps, false);
    (0..n_taps)
        .map(|k| {
            let m = k as isize - c;
            if m == 0 {
                0.0
            } else {
                let mf = m as f64;
                // Ideal full-band differentiator: cos(πm)/m − sin(πm)/(πm²)
                (PI * mf).cos() / mf * win[k]
            }
        })
        .collect()
}

/// Windowed ideal Hilbert transformer (antisymmetric, Blackman window):
/// shifts every positive-frequency component by −90°.
///
/// # Panics
/// Panics unless `n_taps` is odd.
#[must_use]
pub fn fir_hilbert(n_taps: usize) -> Vec<f64> {
    assert!(n_taps % 2 == 1, "fir_hilbert requires an odd tap count");
    let c = (n_taps / 2) as isize;
    let win = window(WindowKind::Blackman, n_taps, false);
    (0..n_taps)
        .map(|k| {
            let m = k as isize - c;
            if m % 2 == 0 {
                0.0
            } else {
                2.0 / (PI * m as f64) * win[k]
            }
        })
        .collect()
}

/// Raised-cosine (Nyquist) pulse: `span` symbols long at `sps` samples
/// per symbol with roll-off `beta` ∈ [0, 1]. Length span·sps + 1, peak 1,
/// zero ISI at symbol spacing.
///
/// # Panics
/// Panics if `span` or `sps` is zero, or beta is outside [0, 1].
#[must_use]
pub fn fir_raised_cosine(span: usize, sps: usize, beta: f64) -> Vec<f64> {
    assert!(span > 0 && sps > 0, "span and sps must be positive");
    assert!((0.0..=1.0).contains(&beta), "beta must be in [0, 1]");
    let n = span * sps + 1;
    let c = (n - 1) as f64 / 2.0;
    (0..n)
        .map(|k| {
            let t = (k as f64 - c) / sps as f64; // in symbols
            let denom = 1.0 - (2.0 * beta * t).powi(2);
            if denom.abs() < 1e-10 {
                // t = ±1/(2β): limit π/4 · sinc(1/(2β))
                PI / 4.0 * sinc(1.0 / (2.0 * beta))
            } else {
                sinc(t) * (PI * beta * t).cos() / denom
            }
        })
        .collect()
}

/// Root-raised-cosine pulse (same span/sps/beta conventions as
/// [`fir_raised_cosine`]); convolving it with itself gives a raised
/// cosine. Normalized to unit energy.
///
/// # Panics
/// Panics if `span` or `sps` is zero, or beta is outside [0, 1].
#[must_use]
pub fn fir_root_raised_cosine(span: usize, sps: usize, beta: f64) -> Vec<f64> {
    assert!(span > 0 && sps > 0, "span and sps must be positive");
    assert!((0.0..=1.0).contains(&beta), "beta must be in [0, 1]");
    let n = span * sps + 1;
    let c = (n - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..n)
        .map(|k| {
            let t = (k as f64 - c) / sps as f64;
            if t.abs() < 1e-10 {
                1.0 - beta + 4.0 * beta / PI
            } else if beta > 0.0 && (t.abs() - 1.0 / (4.0 * beta)).abs() < 1e-10 {
                beta / std::f64::consts::SQRT_2
                    * ((1.0 + 2.0 / PI) * (PI / (4.0 * beta)).sin()
                        + (1.0 - 2.0 / PI) * (PI / (4.0 * beta)).cos())
            } else {
                let num = (PI * t * (1.0 - beta)).sin()
                    + 4.0 * beta * t * (PI * t * (1.0 + beta)).cos();
                let den = PI * t * (1.0 - (4.0 * beta * t).powi(2));
                num / den
            }
        })
        .collect();
    let energy: f64 = h.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in h.iter_mut() {
        *v /= energy;
    }
    h
}

/// Gaussian pulse-shaping filter with bandwidth-time product `bt`
/// (bandwidth normalized to the sample rate). Unit DC gain.
///
/// # Panics
/// Panics if `n_taps == 0` or `bt <= 0`.
#[must_use]
pub fn fir_gaussian(n_taps: usize, bt: f64) -> Vec<f64> {
    assert!(n_taps > 0, "n_taps must be positive");
    assert!(bt > 0.0, "bandwidth-time product must be positive");
    let c = (n_taps - 1) as f64 / 2.0;
    let sigma = (2.0_f64.ln()).sqrt() / (TWO_PI * bt);
    let mut h: Vec<f64> = (0..n_taps)
        .map(|k| (-0.5 * ((k as f64 - c) / sigma).powi(2)).exp())
        .collect();
    let sum: f64 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= sum;
    }
    h
}

/// Savitzky-Golay convolution kernel: fits a polynomial of `order` over
/// a centered odd `window` and evaluates its `deriv`-th derivative (unit
/// sample spacing). Feeding it to [`fir_apply`] estimates the derivative
/// delayed by (window−1)/2 samples.
///
/// # Panics
/// Panics unless `window` is odd and `deriv <= order < window`.
#[must_use]
pub fn fir_savitzky_golay(window: usize, order: usize, deriv: usize) -> Vec<f64> {
    assert!(window % 2 == 1, "window must be odd");
    assert!(order < window, "order must be below the window length");
    assert!(deriv <= order, "derivative order must not exceed the fit order");
    let half = (window / 2) as isize;
    let m = order + 1;
    // Normal equations (AᵀA) c = Aᵀ e where A_{i,j} = x_i^j.
    let mut ata = Matrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for x in -half..=half {
                s += (x as f64).powi((i + j) as i32);
            }
            ata.set(i, j, s);
        }
    }
    let inv = crate::linalg::lu_decompose(&ata)
        .and_then(|lu| lu.inverse())
        .expect("Savitzky-Golay normal matrix is singular");
    // Weight for sample offset x: Σ_k inv[deriv][k] x^k, times deriv!.
    let fact: f64 = (1..=deriv).map(|v| v as f64).product::<f64>().max(1.0);
    let coeffs: Vec<f64> = (-half..=half)
        .map(|x| {
            let mut s = 0.0;
            for k in 0..m {
                s += inv.get(deriv, k) * (x as f64).powi(k as i32);
            }
            s * fact
        })
        .collect();
    // Reverse for convolution so fir_apply computes Σ w_j x[n−half+j].
    coeffs.into_iter().rev().collect()
}

/// Causal FIR filtering by direct convolution; output has the same
/// length as the input (group delay is not compensated).
#[must_use]
pub fn fir_apply(h: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for (n, out) in y.iter_mut().enumerate() {
        let kmax = h.len().min(n + 1);
        let mut acc = 0.0;
        for k in 0..kmax {
            acc += h[k] * x[n - k];
        }
        *out = acc;
    }
    y
}

/// Causal FIR filtering via overlap-save FFT blocks; identical output to
/// [`fir_apply`] but O(n log n) for long kernels.
#[must_use]
pub fn fir_apply_fft(h: &[f64], x: &[f64]) -> Vec<f64> {
    if h.is_empty() || x.is_empty() {
        return vec![0.0; x.len()];
    }
    let m = h.len();
    let nfft = next_power_of_two(8 * m.max(64));
    let step = nfft - (m - 1);
    let mut hf: Vec<Complex> = h.iter().map(|&v| Complex::new(v, 0.0)).collect();
    hf.resize(nfft, Complex::new(0.0, 0.0));
    let hf = fft(&hf);
    let mut y = vec![0.0; x.len()];
    let mut start = 0usize;
    while start < x.len() {
        // Block includes m−1 samples of history.
        let mut block = vec![Complex::new(0.0, 0.0); nfft];
        for (i, slot) in block.iter_mut().enumerate() {
            let idx = start as isize - (m as isize - 1) + i as isize;
            if idx >= 0 && (idx as usize) < x.len() {
                *slot = Complex::new(x[idx as usize], 0.0);
            }
        }
        let mut spec = fft(&block);
        for (s, hv) in spec.iter_mut().zip(&hf) {
            *s = *s * *hv;
        }
        let conv = ifft(&spec);
        let take = step.min(x.len() - start);
        for i in 0..take {
            y[start + i] = conv[m - 1 + i].re;
        }
        start += step;
    }
    y
}

/// Zero-phase filtering: filter forward, reverse, filter again, reverse.
/// The effective magnitude response is |H|².
#[must_use]
pub fn filtfilt_fir(h: &[f64], x: &[f64]) -> Vec<f64> {
    let fwd = fir_apply(h, x);
    let rev: Vec<f64> = fwd.into_iter().rev().collect();
    let back = fir_apply(h, &rev);
    back.into_iter().rev().collect()
}

fn magnitude_at(h: &[f64], f: f64) -> f64 {
    let mut acc = Complex::new(0.0, 0.0);
    for (k, &v) in h.iter().enumerate() {
        let ang = -TWO_PI * f * k as f64;
        acc = acc + Complex::new(v * ang.cos(), v * ang.sin());
    }
    acc.norm()
}

/// Frequency response of an FIR at n points spanning [0, 0.5] (normalized
/// frequency); returns (frequencies, complex response).
///
/// # Panics
/// Panics if `n < 2`.
#[must_use]
pub fn fir_freq_response(h: &[f64], n: usize) -> (Vec<f64>, Vec<Complex>) {
    assert!(n >= 2, "need at least two response points");
    let freqs: Vec<f64> = (0..n).map(|j| 0.5 * j as f64 / (n - 1) as f64).collect();
    let resp: Vec<Complex> = freqs
        .iter()
        .map(|&f| {
            let mut acc = Complex::new(0.0, 0.0);
            for (k, &v) in h.iter().enumerate() {
                let ang = -TWO_PI * f * k as f64;
                acc = acc + Complex::new(v * ang.cos(), v * ang.sin());
            }
            acc
        })
        .collect();
    (freqs, resp)
}

/// Group delay in samples: (n−1)/2 for (anti)symmetric linear-phase
/// kernels, otherwise the energy-weighted center of the impulse response.
#[must_use]
pub fn fir_group_delay(h: &[f64]) -> f64 {
    let n = h.len();
    if n == 0 {
        return 0.0;
    }
    let tol = 1e-9 * h.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-300);
    let symmetric = (0..n).all(|k| (h[k] - h[n - 1 - k]).abs() < tol);
    let antisymmetric = (0..n).all(|k| (h[k] + h[n - 1 - k]).abs() < tol);
    if symmetric || antisymmetric {
        return (n - 1) as f64 / 2.0;
    }
    let energy: f64 = h.iter().map(|v| v * v).sum();
    if energy == 0.0 {
        return (n - 1) as f64 / 2.0;
    }
    h.iter().enumerate().map(|(k, &v)| k as f64 * v * v).sum::<f64>() / energy
}

/// Streaming FIR state: one-sample-at-a-time processing with an
/// internal circular delay line.
pub struct FirState {
    h: Vec<f64>,
    buf: Vec<f64>,
    pos: usize,
}

impl FirState {
    /// Create a streaming filter from a kernel.
    ///
    /// # Panics
    /// Panics if `h` is empty.
    #[must_use]
    pub fn new(h: Vec<f64>) -> Self {
        assert!(!h.is_empty(), "kernel must be non-empty");
        let n = h.len();
        Self { h, buf: vec![0.0; n], pos: 0 }
    }

    /// Push one sample and get the filtered output.
    pub fn process(&mut self, x: f64) -> f64 {
        let n = self.h.len();
        self.buf[self.pos] = x;
        let mut acc = 0.0;
        for (k, &hk) in self.h.iter().enumerate() {
            let idx = (self.pos + n - k) % n;
            acc += hk * self.buf[idx];
        }
        self.pos = (self.pos + 1) % n;
        acc
    }

    /// Clear the delay line.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(f: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (TWO_PI * f * i as f64).sin()).collect()
    }

    fn steady_amplitude(y: &[f64]) -> f64 {
        // Peak of the second half (past the transient).
        y[y.len() / 2..].iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn test_lowpass_passes_low_blocks_high() {
        let h = fir_lowpass(101, 0.1, WindowKind::Hamming);
        let low = fir_apply(&h, &tone(0.03, 2000));
        let high = fir_apply(&h, &tone(0.25, 2000));
        assert!((steady_amplitude(&low) - 1.0).abs() < 0.01);
        assert!(steady_amplitude(&high) < 3e-3);
    }

    #[test]
    fn test_highpass_mirror() {
        let h = fir_highpass(101, 0.15, WindowKind::Hamming);
        let low = fir_apply(&h, &tone(0.03, 2000));
        let high = fir_apply(&h, &tone(0.35, 2000));
        assert!(steady_amplitude(&low) < 5e-3);
        assert!((steady_amplitude(&high) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bandpass_bandstop() {
        let bp = fir_bandpass(151, 0.15, 0.25, WindowKind::Hamming);
        assert!(magnitude_at(&bp, 0.2) > 0.99);
        assert!(magnitude_at(&bp, 0.05) < 1e-2);
        assert!(magnitude_at(&bp, 0.4) < 1e-2);
        let bs = fir_bandstop(151, 0.15, 0.25, WindowKind::Hamming);
        assert!(magnitude_at(&bs, 0.2) < 2e-2);
        assert!((magnitude_at(&bs, 0.02) - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_kaiser_design_meets_spec() {
        let h = fir_kaiser_design(0.15, 0.2, 0.1, 60.0);
        // Stopband tone attenuated by more than 60 dB.
        let stop = magnitude_at(&h, 0.22);
        assert!(20.0 * stop.log10() < -60.0, "stopband {} dB", 20.0 * stop.log10());
        // Passband within ripple.
        let pass = magnitude_at(&h, 0.1);
        assert!((pass - 1.0).abs() < 0.011, "passband {pass}");
    }

    #[test]
    fn test_parks_mcclellan_equiripple() {
        let bands = [(0.0, 0.18), (0.26, 0.5)];
        let h = fir_parks_mcclellan(31, &bands, &[1.0, 0.0], &[1.0, 1.0]).unwrap();
        assert_eq!(h.len(), 31);
        // Symmetric.
        for k in 0..31 {
            assert!((h[k] - h[30 - k]).abs() < 1e-9);
        }
        // Max error per band should match (equiripple) to 1%.
        let mut max_err = [0.0_f64; 2];
        for (b, &(lo, hi)) in bands.iter().enumerate() {
            let d = if b == 0 { 1.0 } else { 0.0 };
            for i in 0..=2048 {
                let f = lo + (hi - lo) * i as f64 / 2048.0;
                max_err[b] = max_err[b].max((magnitude_at(&h, f) - d).abs());
            }
        }
        let rel = (max_err[0] - max_err[1]).abs() / max_err[0].max(max_err[1]);
        assert!(rel < 0.01, "band errors {max_err:?} differ by {rel}");
        // And it should actually be a decent filter.
        assert!(max_err[0] < 0.1, "ripple too large: {max_err:?}");
    }

    #[test]
    fn test_parks_mcclellan_rejects_bad_spec() {
        assert!(fir_parks_mcclellan(31, &[(0.3, 0.1)], &[1.0], &[1.0]).is_err());
        assert!(fir_parks_mcclellan(31, &[(0.0, 0.2)], &[1.0, 0.0], &[1.0]).is_err());
    }

    #[test]
    fn test_least_squares_lowpass() {
        let h = fir_least_squares(41, &[(0.0, 0.15), (0.25, 0.5)], &[1.0, 0.0]);
        assert!((magnitude_at(&h, 0.05) - 1.0).abs() < 0.02);
        assert!(magnitude_at(&h, 0.35) < 0.02);
    }

    #[test]
    fn test_savitzky_golay_preserves_polynomials() {
        // Smoothing kernel must reproduce polynomial values exactly.
        for order in [2usize, 3, 4] {
            let w = 11;
            let h = fir_savitzky_golay(w, order, 0);
            let x: Vec<f64> = (0..200)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    (0..=order).map(|p| 0.3 * t.powi(p as i32)).sum()
                })
                .collect();
            let y = fir_apply(&h, &x);
            let delay = w / 2;
            for n in w..200 {
                assert!(
                    (y[n] - x[n - delay]).abs() < 1e-6,
                    "order {order}: {} vs {}",
                    y[n],
                    x[n - delay]
                );
            }
        }
    }

    #[test]
    fn test_savitzky_golay_first_derivative() {
        let w = 9;
        let h = fir_savitzky_golay(w, 3, 1);
        // Cubic: derivative recovered exactly (unit spacing).
        let x: Vec<f64> = (0..100).map(|i| {
            let t = i as f64;
            0.001 * t * t * t - 0.05 * t * t + t
        }).collect();
        let y = fir_apply(&h, &x);
        let delay = w / 2;
        for (n, &yn) in y.iter().enumerate().skip(w) {
            let t = (n - delay) as f64;
            let exact = 0.003 * t * t - 0.1 * t + 1.0;
            assert!((yn - exact).abs() < 1e-6, "{yn} vs {exact}");
        }
    }

    #[test]
    fn test_fir_apply_fft_matches_direct() {
        let h = fir_lowpass(33, 0.2, WindowKind::Hann);
        let x: Vec<f64> = (0..500).map(|i| ((i * 7919) % 100) as f64 / 50.0 - 1.0).collect();
        let direct = fir_apply(&h, &x);
        let fast = fir_apply_fft(&h, &x);
        for (a, b) in direct.iter().zip(&fast) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_filtfilt_zero_phase() {
        let h = fir_lowpass(51, 0.2, WindowKind::Hamming);
        let f = 0.05;
        let n = 1000;
        let x = tone(f, n);
        let y = filtfilt_fir(&h, &x);
        // Compare against the input in the interior: no phase shift.
        let mut dot = 0.0;
        let mut xx = 0.0;
        let mut yy = 0.0;
        for i in 200..800 {
            dot += x[i] * y[i];
            xx += x[i] * x[i];
            yy += y[i] * y[i];
        }
        let corr = dot / (xx * yy).sqrt();
        assert!(corr > 0.9999, "phase shifted: corr = {corr}");
    }

    #[test]
    fn test_raised_cosine_zero_isi() {
        let sps = 8;
        let h = fir_raised_cosine(8, sps, 0.35);
        let c = (h.len() - 1) / 2;
        assert!((h[c] - 1.0).abs() < 1e-12);
        for k in 1..=3 {
            assert!(h[c + k * sps].abs() < 1e-9, "ISI at symbol {k}");
            assert!(h[c - k * sps].abs() < 1e-9);
        }
    }

    #[test]
    fn test_rrc_convolved_is_rc() {
        let sps = 8;
        let rrc = fir_root_raised_cosine(12, sps, 0.35);
        let rc = fir_raised_cosine(12, sps, 0.35);
        let conv = crate::signal_processing::convolve(&rrc, &rrc);
        // Compare shape: normalize both to peak 1 and check symbol zeros.
        let peak = conv.iter().cloned().fold(f64::MIN, f64::max);
        let c = (conv.len() - 1) / 2;
        let rc_peak = rc.iter().cloned().fold(f64::MIN, f64::max);
        for k in 1..=4 {
            let v = conv[c + k * sps] / peak;
            let r = rc[(rc.len() - 1) / 2 + (k * sps).min((rc.len() - 1) / 2)] / rc_peak;
            assert!(v.abs() < 0.01, "RRC² ISI at symbol {k}: {v} (rc {r})");
        }
    }

    #[test]
    fn test_differentiator_gain() {
        let h = fir_differentiator(31);
        // Response to a slow tone: amplitude ≈ ω = 2πf.
        let f = 0.02;
        let x = tone(f, 2000);
        let y = fir_apply(&h, &x);
        let amp = steady_amplitude(&y);
        assert!((amp - TWO_PI * f).abs() / (TWO_PI * f) < 0.05, "amp {amp}");
    }

    #[test]
    fn test_hilbert_quadrature() {
        let h = fir_hilbert(101);
        let f = 0.1;
        let n = 2000;
        let x: Vec<f64> = (0..n).map(|i| (TWO_PI * f * i as f64).cos()).collect();
        let y = fir_apply(&h, &x);
        let delay = 50;
        // Hilbert of cos is sin (with the group delay applied).
        for (i, &yi) in y.iter().enumerate().take(1500).skip(500) {
            let expect = (TWO_PI * f * (i - delay) as f64).sin();
            assert!((yi - expect).abs() < 0.01, "at {i}: {yi} vs {expect}");
        }
    }

    #[test]
    fn test_gaussian_dc_and_shape() {
        let h = fir_gaussian(33, 0.3);
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let c = 16;
        assert!(h[c] >= h[c + 1] && h[c + 1] >= h[c + 5]);
    }

    #[test]
    fn test_group_delay_and_state() {
        let h = fir_lowpass(21, 0.2, WindowKind::Hann);
        assert!((fir_group_delay(&h) - 10.0).abs() < 1e-12);
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let batch = fir_apply(&h, &x);
        let mut st = FirState::new(h);
        let stream: Vec<f64> = x.iter().map(|&v| st.process(v)).collect();
        for (a, b) in batch.iter().zip(&stream) {
            assert!((a - b).abs() < 1e-12);
        }
        st.reset();
        assert!(st.process(0.0).abs() < 1e-15);
    }

    #[test]
    fn test_freq_response_endpoints() {
        let h = fir_lowpass(41, 0.1, WindowKind::Hamming);
        let (f, r) = fir_freq_response(&h, 129);
        assert_eq!(f[0], 0.0);
        assert!((f[128] - 0.5).abs() < 1e-12);
        assert!((r[0].norm() - 1.0).abs() < 1e-6);
        assert!(r[128].norm() < 1e-2);
    }
}
