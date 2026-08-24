//! Laplace-domain tools: numerical inverse transforms (fixed-Talbot and
//! Gaver-Stehfest), the z-transform, transfer-function responses, and a
//! discrete fractional Fourier transform.
//!
//! Polynomial coefficient conventions: s-domain polynomials are
//! highest-power-first (like `numerical::polynomial_roots`); digital
//! filter coefficient arrays are in z⁻¹ powers (b\[0\] + b\[1\]z⁻¹ + …).

use crate::error::SolveError;
use crate::fractals::Complex;
use crate::linalg::{eigen_symmetric, Matrix};
use crate::math::constants::PI;

fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

fn cexp(z: Complex) -> Complex {
    let r = z.re.exp();
    Complex::new(r * z.im.cos(), r * z.im.sin())
}

/// Fixed-Talbot inverse Laplace transform (Abate & Valkó 2004) with m
/// contour nodes: f(t) from F(s) for t > 0.
///
/// # Panics
/// Panics if `t <= 0` or `m < 2`.
#[must_use]
pub fn inverse_laplace_talbot(f: &dyn Fn(Complex) -> Complex, t: f64, m: usize) -> f64 {
    assert!(t > 0.0, "Talbot inversion needs t > 0");
    assert!(m >= 2, "need at least 2 contour nodes");
    let r = 2.0 * m as f64 / (5.0 * t);
    // θ = 0 node (real axis).
    let mut acc = 0.5 * (f(Complex::new(r, 0.0)) * Complex::new((r * t).exp(), 0.0)).re;
    for k in 1..m {
        let theta = k as f64 * PI / m as f64;
        let cot = theta.cos() / theta.sin();
        let s = Complex::new(r * theta * cot, r * theta);
        let sigma = theta + (theta * cot - 1.0) * cot;
        let term = cexp(Complex::new(s.re * t, s.im * t)) * f(s) * Complex::new(1.0, sigma);
        acc += term.re;
    }
    acc * r / m as f64
}

/// Gaver-Stehfest inverse Laplace transform: needs only real F(s)
/// evaluations. `n` must be even (12-16 is typical; larger n needs more
/// precision than f64 can give).
///
/// # Panics
/// Panics if `t <= 0`, n is odd, or n > 18.
#[must_use]
pub fn inverse_laplace_stehfest(f: &dyn Fn(f64) -> f64, t: f64, n: usize) -> f64 {
    assert!(t > 0.0, "Stehfest inversion needs t > 0");
    assert!(n >= 2 && n.is_multiple_of(2), "Stehfest n must be even");
    assert!(n <= 18, "Stehfest weights overflow f64 beyond n = 18");
    let ln2 = std::f64::consts::LN_2;
    let fact = |k: usize| -> f64 { (1..=k).map(|v| v as f64).product::<f64>().max(1.0) };
    let half = n / 2;
    let mut acc = 0.0;
    for k in 1..=n {
        let mut v = 0.0;
        let j0 = k.div_ceil(2);
        for j in j0..=k.min(half) {
            v += (j as f64).powi(half as i32) * fact(2 * j)
                / (fact(half - j) * fact(j) * fact(j - 1) * fact(k - j) * fact(2 * j - k));
        }
        if !(k + half).is_multiple_of(2) {
            v = -v;
        }
        acc += v * f(k as f64 * ln2 / t);
    }
    acc * ln2 / t
}

/// Forward Laplace transform F(s) = ∫₀^tmax f(t)e^(−st) dt by composite
/// Simpson quadrature (n panels, n rounded up to even).
#[must_use]
pub fn laplace_numeric(f: &dyn Fn(f64) -> f64, s: f64, t_max: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n.max(2) } else { n + 1 };
    let h = t_max / n as f64;
    let g = |t: f64| f(t) * (-s * t).exp();
    let mut acc = g(0.0) + g(t_max);
    for i in 1..n {
        let w = if !i.is_multiple_of(2) { 4.0 } else { 2.0 };
        acc += w * g(i as f64 * h);
    }
    acc * h / 3.0
}

/// Evaluate the (one-sided) z-transform X(z) = Σ x\[n\]·z^(−n).
#[must_use]
pub fn z_transform_eval(x: &[f64], z: Complex) -> Complex {
    // Horner in z⁻¹.
    let mut acc = Complex::new(0.0, 0.0);
    let inv = Complex::new(1.0, 0.0) / z;
    for &v in x.iter().rev() {
        acc = acc * inv + Complex::new(v, 0.0);
    }
    acc
}

/// Impulse response of the digital transfer function
/// H(z) = (num\[0\] + num\[1\]z⁻¹ + …)/(den\[0\] + den\[1\]z⁻¹ + …),
/// by running the difference equation for n samples.
///
/// # Panics
/// Panics if `den` is empty or `den[0] == 0`.
#[must_use]
pub fn impulse_response_from_tf(num: &[f64], den: &[f64], n: usize) -> Vec<f64> {
    assert!(!den.is_empty() && den[0] != 0.0, "denominator must have a nonzero leading term");
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut acc = if i < num.len() { num[i] } else { 0.0 };
        for (k, &a) in den.iter().enumerate().skip(1) {
            if i >= k {
                acc -= a * y[i - k];
            }
        }
        y[i] = acc / den[0];
    }
    y
}

fn polyval_complex(coeffs: &[f64], s: Complex) -> Complex {
    let mut acc = Complex::new(0.0, 0.0);
    for &c in coeffs {
        acc = acc * s + Complex::new(c, 0.0);
    }
    acc
}

/// Continuous-time frequency response H(jω) of num(s)/den(s)
/// (highest-power-first coefficients) at each ω.
#[must_use]
pub fn s_domain_freq_response(num: &[f64], den: &[f64], omega: &[f64]) -> Vec<Complex> {
    omega
        .iter()
        .map(|&w| {
            let s = Complex::new(0.0, w);
            polyval_complex(num, s) / polyval_complex(den, s)
        })
        .collect()
}

/// Digital frequency response of b(z⁻¹)/a(z⁻¹) at `n_points` normalized
/// frequencies spanning [0, 0.5]; returns (frequencies, response).
///
/// # Panics
/// Panics if `n_points < 2`.
#[must_use]
pub fn digital_freq_response(b: &[f64], a: &[f64], n_points: usize) -> (Vec<f64>, Vec<Complex>) {
    assert!(n_points >= 2, "need at least two response points");
    let freqs: Vec<f64> = (0..n_points).map(|j| 0.5 * j as f64 / (n_points - 1) as f64).collect();
    let resp = freqs
        .iter()
        .map(|&f| {
            let z = cis(2.0 * PI * f);
            z_transform_eval(b, z) / z_transform_eval(a, z)
        })
        .collect();
    (freqs, resp)
}

/// Discrete fractional Fourier transform of angle `alpha` (α = π/2 is
/// the unitary DFT), via the eigendecomposition of the Candan-Kutay-
/// Ozaktas commuting matrix: F^α = Σ_k e^(−i·k·α)·u_k·(u_kᵀx). O(n²)
/// after an O(n³) eigen solve, intended for moderate n.
///
/// # Errors
/// Returns an error if the eigen decomposition fails to converge.
pub fn fractional_fourier(x: &[Complex], alpha: f64) -> Result<Vec<Complex>, SolveError> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(x.to_vec());
    }
    // Commuting matrix S: near-tridiagonal, symmetric, circulant corners.
    let mut s = Matrix::zeros(n, n);
    for i in 0..n {
        s.set(i, i, 2.0 * (2.0 * PI * i as f64 / n as f64).cos() - 4.0);
        let j = (i + 1) % n;
        s.set(i, j, 1.0);
        s.set(j, i, 1.0);
    }
    let eig = eigen_symmetric(&s, 1e-12, 200)?;
    let mut vecs: Vec<Vec<f64>> = (0..n)
        .map(|c| (0..n).map(|r| eig.vectors.get(r, c)).collect())
        .collect();
    let mut vals: Vec<f64> = eig.values.clone();
    // Sort by eigenvalue so degenerate pairs sit together.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));
    vecs = idx.iter().map(|&i| vecs[i].clone()).collect();
    vals = idx.iter().map(|&i| vals[i]).collect();
    // Both S and the DFT commute with the parity flip u[i] -> u[(n-i) mod n];
    // within (near-)degenerate clusters, separate even/odd parity so every
    // basis vector is a genuine DFT eigenvector.
    let parity = |u: &[f64]| -> Vec<f64> { (0..n).map(|i| u[(n - i) % n]).collect() };
    let mut cleaned: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && (vals[end] - vals[start]).abs() < 1e-6 {
            end += 1;
        }
        // Parity projections of the cluster, then Gram-Schmidt.
        let mut cands: Vec<Vec<f64>> = Vec::new();
        for v in &vecs[start..end] {
            let p = parity(v);
            let even: Vec<f64> = v.iter().zip(&p).map(|(a, b)| 0.5 * (a + b)).collect();
            let odd: Vec<f64> = v.iter().zip(&p).map(|(a, b)| 0.5 * (a - b)).collect();
            cands.push(even);
            cands.push(odd);
        }
        for mut c in cands {
            for kept in cleaned.iter().skip(start) {
                let dot: f64 = c.iter().zip(kept).map(|(a, b)| a * b).sum();
                for (cv, kv) in c.iter_mut().zip(kept) {
                    *cv -= dot * kv;
                }
            }
            let norm: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-6 && cleaned.len() < end {
                for v in c.iter_mut() {
                    *v /= norm;
                }
                cleaned.push(c);
            }
        }
        // Numerical fallback: if parity separation lost a vector, keep
        // the originals.
        while cleaned.len() < end {
            cleaned.push(vecs[cleaned.len()].clone());
        }
        start = end;
    }
    // Hermite-index assignment: each basis vector is a DFT eigenvector
    // with eigenvalue (−i)^m; within each residue class m the Hermite
    // index increases as the S-eigenvalue decreases. Classify every
    // vector by its measured DFT eigenvalue, then hand out the known
    // index list {0, …, n−2, n} (even n) / {0, …, n−1} (odd n) per class.
    let scale = 1.0 / (n as f64).sqrt();
    let lambdas = [
        Complex::new(1.0, 0.0),
        Complex::new(0.0, -1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, 1.0),
    ];
    let class_of = |u: &[f64]| -> usize {
        let uc: Vec<Complex> = u.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let fu = crate::transforms::fft::fft_any(&uc);
        let mut best = (0usize, f64::MAX);
        for (m, lam) in lambdas.iter().enumerate() {
            let resid: f64 = fu
                .iter()
                .zip(&uc)
                .map(|(f, u)| {
                    let d = Complex::new(f.re * scale, f.im * scale) - *lam * *u;
                    d.norm_sq()
                })
                .sum();
            if resid < best.1 {
                best = (m, resid);
            }
        }
        best.0
    };
    // Vectors are already sorted by ascending S-eigenvalue; Hermite index
    // grows with descending eigenvalue, so walk them in reverse.
    let mut by_class: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for v in (0..n).rev() {
        by_class[class_of(&cleaned[v])].push(v);
    }
    let k_list: Vec<usize> = if n.is_multiple_of(2) {
        (0..n - 1).chain(std::iter::once(n)).collect()
    } else {
        (0..n).collect()
    };
    let mut assignment: Vec<(usize, usize)> = Vec::with_capacity(n); // (vector, k)
    let mut used: [usize; 4] = [0; 4];
    for &k in &k_list {
        let m = k % 4;
        let slot = used[m];
        if slot >= by_class[m].len() {
            return Err(SolveError::NoConvergence { iters: 0, residual: f64::NAN });
        }
        assignment.push((by_class[m][slot], k));
        used[m] += 1;
    }
    let mut out = vec![Complex::new(0.0, 0.0); n];
    for &(col, k) in &assignment {
        let phase = cis(-(k as f64) * alpha);
        let u = &cleaned[col];
        let mut proj = Complex::new(0.0, 0.0);
        for (r, &uv) in u.iter().enumerate() {
            proj = proj + Complex::new(uv, 0.0) * x[r];
        }
        let coeff = phase * proj;
        for (o, &uv) in out.iter_mut().zip(u.iter()) {
            *o = *o + coeff * Complex::new(uv, 0.0);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_talbot_exponential() {
        // F(s) = 1/(s+1) ↔ e^{-t}
        let f = |s: Complex| Complex::new(1.0, 0.0) / (s + Complex::new(1.0, 0.0));
        for &t in &[0.1, 0.5, 1.0, 3.0] {
            let v = inverse_laplace_talbot(&f, t, 32);
            assert!(approx(v, (-t).exp(), 1e-10), "t={t}: {v}");
        }
    }

    #[test]
    fn test_talbot_sine() {
        // F(s) = 1/(s²+1) ↔ sin t
        let f = |s: Complex| Complex::new(1.0, 0.0) / (s * s + Complex::new(1.0, 0.0));
        for &t in &[0.5, 2.0, 5.0] {
            let v = inverse_laplace_talbot(&f, t, 48);
            assert!(approx(v, t.sin(), 1e-8), "t={t}: {v} vs {}", t.sin());
        }
    }

    #[test]
    fn test_stehfest_exponential_and_step() {
        let f = |s: f64| 1.0 / (s + 1.0);
        for &t in &[0.5, 1.0, 2.0] {
            let v = inverse_laplace_stehfest(&f, t, 14);
            // Gaver-Stehfest in f64 reaches ~5 significant digits.
            assert!(approx(v, (-t).exp(), 1e-4), "t={t}: {v}");
        }
        // 1/s ↔ 1
        let step = |s: f64| 1.0 / s;
        let v = inverse_laplace_stehfest(&step, 1.7, 12);
        assert!(approx(v, 1.0, 1e-8), "{v}");
    }

    #[test]
    fn test_laplace_numeric_matches_analytic() {
        // L{e^{-2t}}(s) = 1/(s+2)
        let f = |t: f64| (-2.0 * t).exp();
        let v = laplace_numeric(&f, 1.0, 20.0, 2000);
        assert!(approx(v, 1.0 / 3.0, 1e-8), "{v}");
    }

    #[test]
    fn test_z_transform_geometric() {
        // x[n] = a^n, X(z) = 1/(1 − a/z) truncated: compare to closed sum.
        let a: f64 = 0.5;
        let x: Vec<f64> = (0..50).map(|n| a.powi(n)).collect();
        let z = Complex::new(1.2, 0.3);
        let v = z_transform_eval(&x, z);
        // Closed form of the truncated series.
        let ratio = Complex::new(a, 0.0) / z;
        let mut num = Complex::new(1.0, 0.0);
        for _ in 0..50 {
            num = num * ratio;
        }
        let expect = (Complex::new(1.0, 0.0) - num) / (Complex::new(1.0, 0.0) - ratio);
        assert!(approx(v.re, expect.re, 1e-10) && approx(v.im, expect.im, 1e-10));
    }

    #[test]
    fn test_impulse_response_one_pole() {
        // H(z) = 1/(1 − 0.9 z⁻¹): h[n] = 0.9^n
        let h = impulse_response_from_tf(&[1.0], &[1.0, -0.9], 20);
        for (n, v) in h.iter().enumerate() {
            assert!(approx(*v, 0.9_f64.powi(n as i32), 1e-12));
        }
    }

    #[test]
    fn test_s_domain_rc_filter() {
        // H(s) = 1/(s+1): |H(j·1)| = 1/√2.
        let resp = s_domain_freq_response(&[1.0], &[1.0, 1.0], &[0.0, 1.0, 10.0]);
        assert!(approx(resp[0].norm(), 1.0, 1e-12));
        assert!(approx(resp[1].norm(), std::f64::consts::FRAC_1_SQRT_2, 1e-12));
        assert!(resp[2].norm() < 0.1);
    }

    #[test]
    fn test_digital_freq_response_moving_average() {
        // 2-tap averager: unity at DC, null at Nyquist.
        let (f, h) = digital_freq_response(&[0.5, 0.5], &[1.0], 65);
        assert!(approx(h[0].norm(), 1.0, 1e-12));
        assert!(h[64].norm() < 1e-12);
        assert!(approx(f[64], 0.5, 1e-12));
    }

    #[test]
    fn test_frft_quarter_is_dft() {
        let n = 16;
        let x: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.4).sin(), (i as f64 * 0.9).cos()))
            .collect();
        let frft = fractional_fourier(&x, PI / 2.0).unwrap();
        let dft = crate::transforms::fft::fft_any(&x);
        let scale = 1.0 / (n as f64).sqrt();
        // At α = π/2 the discrete FrFT IS the unitary DFT, bin for bin.
        for (k, (a, b)) in frft.iter().zip(&dft).enumerate() {
            assert!(approx(a.re, b.re * scale, 1e-8), "bin {k} re");
            assert!(approx(a.im, b.im * scale, 1e-8), "bin {k} im");
        }
    }

    #[test]
    fn test_frft_additivity_and_identity() {
        let n = 12;
        let x: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64 * 1.1).cos(), (i as f64 * 0.2).sin()))
            .collect();
        let id = fractional_fourier(&x, 0.0).unwrap();
        for (a, b) in id.iter().zip(&x) {
            assert!(approx(a.re, b.re, 1e-9) && approx(a.im, b.im, 1e-9));
        }
        let ab = fractional_fourier(&fractional_fourier(&x, 0.4).unwrap(), 0.7).unwrap();
        let sum = fractional_fourier(&x, 1.1).unwrap();
        for (a, b) in ab.iter().zip(&sum) {
            assert!(approx(a.re, b.re, 1e-8) && approx(a.im, b.im, 1e-8));
        }
    }
}
