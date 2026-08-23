//! Discrete cosine, sine, and Hartley transforms.
//!
//! Conventions match scipy's unnormalized (`norm=None`) definitions:
//!
//! * DCT-I:   y\[k\] = x\[0\] + (−1)^k x\[N−1\] + 2 Σ_{n=1}^{N−2} x\[n\] cos(πkn/(N−1))
//! * DCT-II:  y\[k\] = 2 Σ x\[n\] cos(πk(2n+1)/(2N))
//! * DCT-III: y\[k\] = x\[0\] + 2 Σ_{n≥1} x\[n\] cos(πn(2k+1)/(2N))
//! * DCT-IV:  y\[k\] = 2 Σ x\[n\] cos(π(2k+1)(2n+1)/(4N))
//! * DST-I:   y\[k\] = 2 Σ x\[n\] sin(π(k+1)(n+1)/(N+1))
//! * DST-II:  y\[k\] = 2 Σ x\[n\] sin(π(k+1)(2n+1)/(2N))
//!
//! Everything runs in O(n log n) through the any-length FFT.

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::transforms::fft::fft_any;

const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

/// Unscaled inverse-kernel DFT (e^{+2πikn/N}, no 1/N), via the FFT.
fn dft_plus(x: &[Complex]) -> Vec<Complex> {
    let conj: Vec<Complex> = x.iter().map(|c| c.conjugate()).collect();
    fft_any(&conj).iter().map(|c| c.conjugate()).collect()
}

/// DCT-I of length N ≥ 2 (even symmetry about both endpoints).
///
/// # Panics
/// Panics if `x.len() < 2`.
#[must_use]
pub fn dct_i(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    assert!(n >= 2, "dct_i requires at least 2 samples");
    // Even extension of length 2N−2: x[0..N], x[N−2..=1].
    let m = 2 * n - 2;
    let mut z = vec![ZERO; m];
    for (i, &v) in x.iter().enumerate() {
        z[i] = Complex::new(v, 0.0);
    }
    for i in 1..n - 1 {
        z[m - i] = Complex::new(x[i], 0.0);
    }
    fft_any(&z).iter().take(n).map(|c| c.re).collect()
}

/// DCT-II (the "standard" DCT).
#[must_use]
pub fn dct_ii(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    // Mirror trick: FFT of [x, reversed x] has phase-rotated DCT-II bins.
    let m = 2 * n;
    let mut z = vec![ZERO; m];
    for (i, &v) in x.iter().enumerate() {
        z[i] = Complex::new(v, 0.0);
        z[m - 1 - i] = Complex::new(v, 0.0);
    }
    let spec = fft_any(&z);
    (0..n)
        .map(|k| {
            let ph = cis(-PI * k as f64 / m as f64);
            (ph * spec[k]).re
        })
        .collect()
}

/// DCT-III (the unnormalized inverse of DCT-II).
#[must_use]
pub fn dct_iii(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    // c[n] = Σ_k x[k] e^{iπk(2n+1)/(2N)} as a zero-padded inverse DFT.
    let m = 2 * n;
    let mut u = vec![ZERO; m];
    for (k, &v) in x.iter().enumerate() {
        u[k] = Complex::new(v, 0.0) * cis(PI * k as f64 / m as f64);
    }
    let c = dft_plus(&u);
    (0..n).map(|j| 2.0 * c[j].re - x[0]).collect()
}

/// DCT-IV (its own inverse up to a factor 2N).
#[must_use]
pub fn dct_iv(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let m = 2 * n;
    let mut v = vec![ZERO; m];
    for (i, &xv) in x.iter().enumerate() {
        v[i] = Complex::new(xv, 0.0) * cis(-PI * (2 * i + 1) as f64 / (4 * n) as f64);
    }
    let spec = fft_any(&v);
    (0..n)
        .map(|k| {
            let ph = cis(-PI * k as f64 / m as f64);
            2.0 * (ph * spec[k]).re
        })
        .collect()
}

/// Inverse of [`dct_ii`]: x = dct_iii(y) / (2N).
#[must_use]
pub fn idct_ii(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let scale = 1.0 / (2.0 * n as f64);
    dct_iii(x).iter().map(|v| v * scale).collect()
}

/// DST-I (odd symmetry about both virtual endpoints); its own inverse up
/// to a factor 2(N+1).
#[must_use]
pub fn dst_i(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    // Odd extension of length 2N+2: [0, x, 0, −reversed x].
    let m = 2 * n + 2;
    let mut z = vec![ZERO; m];
    for (i, &v) in x.iter().enumerate() {
        z[i + 1] = Complex::new(v, 0.0);
        z[m - 1 - i] = Complex::new(-v, 0.0);
    }
    let spec = fft_any(&z);
    (0..n).map(|k| -spec[k + 1].im).collect()
}

/// DST-II, via the identity DST-II(x)\[k\] = DCT-II(x·(−1)^n)\[N−1−k\].
#[must_use]
pub fn dst_ii(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let alt: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
        .collect();
    let c = dct_ii(&alt);
    (0..n).map(|k| c[n - 1 - k]).collect()
}

/// Separable 2D DCT-II of row-major data (index = y·w + x).
///
/// # Panics
/// Panics unless `x.len() == w * h`.
#[must_use]
pub fn dct_2d(x: &[f64], w: usize, h: usize) -> Vec<f64> {
    apply_2d(x, w, h, dct_ii)
}

/// Inverse of [`dct_2d`].
///
/// # Panics
/// Panics unless `x.len() == w * h`.
#[must_use]
pub fn idct_2d(x: &[f64], w: usize, h: usize) -> Vec<f64> {
    apply_2d(x, w, h, idct_ii)
}

fn apply_2d(x: &[f64], w: usize, h: usize, f: fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    assert_eq!(x.len(), w * h, "2D transform expects w*h samples");
    let mut data = x.to_vec();
    for row in 0..h {
        let out = f(&data[row * w..(row + 1) * w]);
        data[row * w..(row + 1) * w].copy_from_slice(&out);
    }
    let mut col = vec![0.0; h];
    for cx in 0..w {
        for (cy, v) in col.iter_mut().enumerate() {
            *v = data[cy * w + cx];
        }
        let out = f(&col);
        for (cy, v) in out.iter().enumerate() {
            data[cy * w + cx] = *v;
        }
    }
    data
}

/// Discrete Hartley transform: H\[k\] = Σ x\[n\]·cas(2πkn/N) with
/// cas θ = cos θ + sin θ. Self-inverse up to a factor N.
#[must_use]
pub fn hartley(x: &[f64]) -> Vec<f64> {
    let buf: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_any(&buf).iter().map(|c| c.re - c.im).collect()
}

/// Lossy compression demo: keep the largest `keep_fraction` of DCT-II
/// coefficients (by magnitude), zero the rest, and reconstruct.
#[must_use]
pub fn dct_compress(x: &[f64], keep_fraction: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let keep = ((keep_fraction.clamp(0.0, 1.0) * n as f64).round() as usize).min(n);
    let mut coeffs = dct_ii(x);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        coeffs[b]
            .abs()
            .partial_cmp(&coeffs[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in &order[keep..] {
        coeffs[i] = 0.0;
    }
    idct_ii(&coeffs)
}

/// Boundary condition for [`dct_poisson_1d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bc {
    /// u = 0 at both walls (node-centered; walls half a step outside the
    /// first/last sample... walls at the virtual nodes −1 and N).
    Dirichlet,
    /// du/dx = 0 at both walls (cell-centered mirror; the solution is
    /// determined up to a constant and returned mean-free).
    Neumann,
}

/// Solve the 1D Poisson problem u'' = rhs on a uniform grid with
/// homogeneous boundary conditions, diagonalizing the discrete
/// three-point Laplacian with the DST-I (Dirichlet) or DCT-II (Neumann).
/// The discrete residual is at roundoff.
#[must_use]
pub fn dct_poisson_1d(rhs: &[f64], dx: f64, bc: Bc) -> Vec<f64> {
    let n = rhs.len();
    if n == 0 {
        return Vec::new();
    }
    let inv_dx2 = 1.0 / (dx * dx);
    match bc {
        Bc::Dirichlet => {
            let mut coef = dst_i(rhs);
            for (k, c) in coef.iter_mut().enumerate() {
                let lambda = (2.0 * (PI * (k + 1) as f64 / (n + 1) as f64).cos() - 2.0) * inv_dx2;
                *c /= lambda;
            }
            let scale = 1.0 / (2.0 * (n + 1) as f64);
            dst_i(&coef).iter().map(|v| v * scale).collect()
        }
        Bc::Neumann => {
            let mut coef = dct_ii(rhs);
            coef[0] = 0.0;
            for (k, c) in coef.iter_mut().enumerate().skip(1) {
                let lambda = (2.0 * (PI * k as f64 / n as f64).cos() - 2.0) * inv_dx2;
                *c /= lambda;
            }
            idct_ii(&coef)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn sample(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.7).sin() + 0.3 * (i as f64 * 1.9).cos()).collect()
    }

    // Naive O(n²) references.
    fn naive_dct_i(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                let mut acc = x[0] + sign * x[n - 1];
                for (j, &v) in x.iter().enumerate().take(n - 1).skip(1) {
                    acc += 2.0 * v * (PI * (k * j) as f64 / (n - 1) as f64).cos();
                }
                acc
            })
            .collect()
    }

    fn naive_dct_ii(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                2.0 * x
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v * (PI * k as f64 * (2 * j + 1) as f64 / (2 * n) as f64).cos())
                    .sum::<f64>()
            })
            .collect()
    }

    fn naive_dct_iii(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x[0] + 2.0
                    * x.iter()
                        .enumerate()
                        .skip(1)
                        .map(|(j, &v)| {
                            v * (PI * j as f64 * (2 * k + 1) as f64 / (2 * n) as f64).cos()
                        })
                        .sum::<f64>()
            })
            .collect()
    }

    fn naive_dct_iv(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                2.0 * x
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        v * (PI * (2 * k + 1) as f64 * (2 * j + 1) as f64 / (4 * n) as f64).cos()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    fn naive_dst_i(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                2.0 * x
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        v * (PI * (k + 1) as f64 * (j + 1) as f64 / (n + 1) as f64).sin()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    fn naive_dst_ii(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                2.0 * x
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        v * (PI * (k + 1) as f64 * (2 * j + 1) as f64 / (2 * n) as f64).sin()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    #[test]
    fn test_all_match_naive() {
        for &n in &[2usize, 3, 5, 8, 12, 17] {
            let x = sample(n);
            let pairs: [(Vec<f64>, Vec<f64>, &str); 6] = [
                (dct_i(&x), naive_dct_i(&x), "dct_i"),
                (dct_ii(&x), naive_dct_ii(&x), "dct_ii"),
                (dct_iii(&x), naive_dct_iii(&x), "dct_iii"),
                (dct_iv(&x), naive_dct_iv(&x), "dct_iv"),
                (dst_i(&x), naive_dst_i(&x), "dst_i"),
                (dst_ii(&x), naive_dst_ii(&x), "dst_ii"),
            ];
            for (fast, slow, name) in &pairs {
                for (a, b) in fast.iter().zip(slow) {
                    assert!(approx(*a, *b, 1e-10), "{name} mismatch at n={n}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn test_idct_ii_roundtrip() {
        for &n in &[1usize, 2, 7, 16, 33] {
            let x = sample(n);
            let back = idct_ii(&dct_ii(&x));
            for (a, b) in x.iter().zip(&back) {
                assert!(approx(*a, *b, 1e-10), "n={n}");
            }
        }
    }

    #[test]
    fn test_dct_iv_involution() {
        let x = sample(16);
        let twice = dct_iv(&dct_iv(&x));
        for (a, b) in twice.iter().zip(&x) {
            assert!(approx(*a, b * 32.0, 1e-8)); // 2N = 32
        }
    }

    #[test]
    fn test_dst_i_involution() {
        let x = sample(9);
        let twice = dst_i(&dst_i(&x));
        for (a, b) in twice.iter().zip(&x) {
            assert!(approx(*a, b * 20.0, 1e-9)); // 2(N+1) = 20
        }
    }

    #[test]
    fn test_hartley_involution() {
        let x = sample(24);
        let twice = hartley(&hartley(&x));
        for (a, b) in twice.iter().zip(&x) {
            assert!(approx(*a, b * 24.0, 1e-9));
        }
    }

    #[test]
    fn test_dct_2d_roundtrip_and_dc() {
        let (w, h) = (8, 6);
        let x = sample(w * h);
        let spec = dct_2d(&x, w, h);
        let sum: f64 = x.iter().sum();
        assert!(approx(spec[0], 4.0 * sum, 1e-9)); // 2·2 from the two passes
        let back = idct_2d(&spec, w, h);
        for (a, b) in x.iter().zip(&back) {
            assert!(approx(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_dct_compress() {
        // A smooth signal survives 25% compression well.
        let x: Vec<f64> = (0..64).map(|i| (PI * i as f64 / 32.0).sin()).collect();
        let y = dct_compress(&x, 0.25);
        let err: f64 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / 64.0;
        assert!(err < 1e-3, "mse {err}");
        // keep_fraction = 1 is lossless.
        let z = dct_compress(&x, 1.0);
        for (a, b) in x.iter().zip(&z) {
            assert!(approx(*a, *b, 1e-10));
        }
    }

    #[test]
    fn test_poisson_dirichlet_residual() {
        let n = 40;
        let dx = 0.1;
        let rhs = sample(n);
        let u = dct_poisson_1d(&rhs, dx, Bc::Dirichlet);
        for i in 0..n {
            let um = if i == 0 { 0.0 } else { u[i - 1] };
            let up = if i == n - 1 { 0.0 } else { u[i + 1] };
            let lap = (up - 2.0 * u[i] + um) / (dx * dx);
            assert!(approx(lap, rhs[i], 1e-9), "i={i}");
        }
    }

    #[test]
    fn test_poisson_neumann_residual() {
        let n = 40;
        let dx = 0.1;
        // Neumann needs a mean-free rhs for solvability.
        let mut rhs = sample(n);
        let mean = rhs.iter().sum::<f64>() / n as f64;
        for v in rhs.iter_mut() {
            *v -= mean;
        }
        let u = dct_poisson_1d(&rhs, dx, Bc::Neumann);
        for i in 0..n {
            // Mirror ghosts: u[-1] = u[0], u[n] = u[n-1].
            let um = if i == 0 { u[0] } else { u[i - 1] };
            let up = if i == n - 1 { u[n - 1] } else { u[i + 1] };
            let lap = (up - 2.0 * u[i] + um) / (dx * dx);
            assert!(approx(lap, rhs[i], 1e-9), "i={i}");
        }
    }
}
