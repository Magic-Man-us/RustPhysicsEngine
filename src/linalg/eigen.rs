//! Eigenvalue solvers.
//!
//! Symmetric matrices use the cyclic Jacobi rotation method (Golub &
//! Van Loan §8.5), which is unconditionally convergent. General real
//! matrices are reduced to upper Hessenberg form by Gaussian similarity
//! transformations and their eigenvalues extracted with the
//! Francis-shift QR iteration (Wilkinson, *The Algebraic Eigenvalue
//! Problem*; the classic `hqr` algorithm).

use crate::error::SolveError;
use crate::fractals::Complex;
use crate::linalg::Matrix;

/// Eigen-decomposition of a symmetric matrix: A·vᵢ = λᵢ·vᵢ.
///
/// `values[i]` pairs with column i of `vectors`; entries are sorted in
/// descending eigenvalue order and the vectors are orthonormal.
#[derive(Debug, Clone, PartialEq)]
pub struct SymEigen {
    pub values: Vec<f64>,
    pub vectors: Matrix,
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix.
///
/// Sweeps Givens rotations over all off-diagonal pairs until the
/// off-diagonal Frobenius norm falls below `tol` (relative to ‖A‖) or
/// `max_sweeps` is exhausted, in which case `NoConvergence` is returned.
pub fn eigen_symmetric(a: &Matrix, tol: f64, max_sweeps: usize) -> Result<SymEigen, SolveError> {
    if !a.is_square() {
        return Err(SolveError::InvalidArgument("eigen_symmetric requires a square matrix"));
    }
    if tol <= 0.0 {
        return Err(SolveError::InvalidArgument("eigen_symmetric requires tol > 0"));
    }
    let scale = a.frobenius_norm().max(1.0);
    if !a.is_symmetric(1e-8 * scale) {
        return Err(SolveError::InvalidArgument("eigen_symmetric requires a symmetric matrix"));
    }
    let n = a.rows;
    let mut m = a.clone();
    let mut v = Matrix::identity(n);

    let off = |m: &Matrix| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    s += m.get(i, j) * m.get(i, j);
                }
            }
        }
        s.sqrt()
    };

    let threshold = tol * scale;
    let mut converged = false;
    for _ in 0..max_sweeps {
        if off(&m) <= threshold {
            converged = true;
            break;
        }
        for p in 0..n - 1 {
            for q in (p + 1)..n {
                let apq = m.get(p, q);
                if apq.abs() <= f64::EPSILON * scale {
                    continue;
                }
                // Jacobi rotation zeroing (p, q).
                let app = m.get(p, p);
                let aqq = m.get(q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update rows/columns p and q of the symmetric matrix.
                for k in 0..n {
                    let mkp = m.get(k, p);
                    let mkq = m.get(k, q);
                    m.set(k, p, c * mkp - s * mkq);
                    m.set(k, q, s * mkp + c * mkq);
                }
                for k in 0..n {
                    let mpk = m.get(p, k);
                    let mqk = m.get(q, k);
                    m.set(p, k, c * mpk - s * mqk);
                    m.set(q, k, s * mpk + c * mqk);
                }
                // Accumulate the eigenvector rotation.
                for k in 0..n {
                    let vkp = v.get(k, p);
                    let vkq = v.get(k, q);
                    v.set(k, p, c * vkp - s * vkq);
                    v.set(k, q, s * vkp + c * vkq);
                }
            }
        }
    }
    if !converged && off(&m) > threshold {
        return Err(SolveError::NoConvergence { iters: max_sweeps, residual: off(&m) });
    }

    // Extract, then sort descending, permuting vector columns alongside.
    let mut order: Vec<usize> = (0..n).collect();
    let values: Vec<f64> = (0..n).map(|i| m.get(i, i)).collect();
    order.sort_by(|&i, &j| values[j].partial_cmp(&values[i]).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    let mut vectors = Matrix::zeros(n, n);
    for (new_c, &old_c) in order.iter().enumerate() {
        for r in 0..n {
            vectors.set(r, new_c, v.get(r, old_c));
        }
    }
    Ok(SymEigen { values: sorted_values, vectors })
}

/// Reduces a square matrix to upper Hessenberg form by stabilized
/// elementary similarity transformations (the `elmhes` scheme).
fn hessenberg(a: &Matrix) -> Matrix {
    let n = a.rows;
    let mut h = a.clone();
    for m in 1..n.saturating_sub(1) {
        // Pivot: largest |h[i][m-1]| for i >= m.
        let mut x = 0.0_f64;
        let mut i_max = m;
        for i in m..n {
            if h.get(i, m - 1).abs() > x.abs() {
                x = h.get(i, m - 1);
                i_max = i;
            }
        }
        if i_max != m {
            for j in (m - 1)..n {
                let tmp = h.get(i_max, j);
                h.set(i_max, j, h.get(m, j));
                h.set(m, j, tmp);
            }
            for j in 0..n {
                let tmp = h.get(j, i_max);
                h.set(j, i_max, h.get(j, m));
                h.set(j, m, tmp);
            }
        }
        if x != 0.0 {
            for i in (m + 1)..n {
                let mut y = h.get(i, m - 1);
                if y != 0.0 {
                    y /= x;
                    h.set(i, m - 1, y);
                    for j in m..n {
                        let val = h.get(i, j) - y * h.get(m, j);
                        h.set(i, j, val);
                    }
                    for j in 0..n {
                        let val = h.get(j, m) + y * h.get(j, i);
                        h.set(j, m, val);
                    }
                }
            }
        }
    }
    // Zero the sub-Hessenberg entries left behind by the elimination.
    for i in 2..n {
        for j in 0..(i - 1) {
            h.set(i, j, 0.0);
        }
    }
    h
}

/// All eigenvalues (possibly complex) of a general real square matrix,
/// by Hessenberg reduction followed by Francis-shift QR iteration.
///
/// `max_iter` bounds the QR iterations spent per eigenvalue (30 is the
/// classical choice).
pub fn eigenvalues_general(a: &Matrix, max_iter: usize) -> Result<Vec<Complex>, SolveError> {
    if !a.is_square() {
        return Err(SolveError::InvalidArgument("eigenvalues_general requires a square matrix"));
    }
    if max_iter == 0 {
        return Err(SolveError::InvalidArgument("eigenvalues_general requires max_iter > 0"));
    }
    let n = a.rows;
    let mut h = hessenberg(a);
    let mut eig = vec![Complex::new(0.0, 0.0); n];

    let anorm: f64 = {
        let mut s = 0.0;
        for i in 0..n {
            for j in i.saturating_sub(1)..n {
                s += h.get(i, j).abs();
            }
        }
        s.max(f64::MIN_POSITIVE)
    };

    let mut nn = n as isize - 1;
    let mut t = 0.0; // accumulated exceptional shifts
    while nn >= 0 {
        let mut its = 0usize;
        loop {
            // Find small subdiagonal element: l is start of active block.
            let mut l = nn;
            while l > 0 {
                let s = h.get(l as usize - 1, l as usize - 1).abs()
                    + h.get(l as usize, l as usize).abs();
                let s = if s == 0.0 { anorm } else { s };
                if h.get(l as usize, l as usize - 1).abs() <= f64::EPSILON * s {
                    h.set(l as usize, l as usize - 1, 0.0);
                    break;
                }
                l -= 1;
            }

            let x = h.get(nn as usize, nn as usize);
            if l == nn {
                // One real eigenvalue deflated.
                eig[nn as usize] = Complex::new(x + t, 0.0);
                nn -= 1;
                break;
            }
            let y = h.get(nn as usize - 1, nn as usize - 1);
            let w = h.get(nn as usize, nn as usize - 1) * h.get(nn as usize - 1, nn as usize);
            if l == nn - 1 {
                // 2×2 block: deflate a real or complex-conjugate pair.
                let p = 0.5 * (y - x);
                let q = p * p + w;
                let z = q.abs().sqrt();
                let x_shifted = x + t;
                if q >= 0.0 {
                    let z = p + if p >= 0.0 { z } else { -z };
                    eig[nn as usize - 1] = Complex::new(x_shifted + z, 0.0);
                    eig[nn as usize] = if z != 0.0 {
                        Complex::new(x_shifted - w / z, 0.0)
                    } else {
                        Complex::new(x_shifted + z, 0.0)
                    };
                } else {
                    eig[nn as usize - 1] = Complex::new(x_shifted + p, z);
                    eig[nn as usize] = Complex::new(x_shifted + p, -z);
                }
                nn -= 2;
                break;
            }

            // No deflation yet: one Francis double-shift QR step.
            if its >= max_iter {
                return Err(SolveError::NoConvergence {
                    iters: its,
                    residual: h.get(nn as usize, nn as usize - 1).abs(),
                });
            }
            let (mut x, mut y, mut w) = (x, y, w);
            if its > 0 && its.is_multiple_of(10) {
                // Exceptional shift to break rare cycling.
                t += x;
                for i in 0..=nn as usize {
                    let d = h.get(i, i) - x;
                    h.set(i, i, d);
                }
                let s = h.get(nn as usize, nn as usize - 1).abs()
                    + h.get(nn as usize - 1, nn as usize - 2).abs();
                x = 0.75 * s;
                y = x;
                w = -0.4375 * s * s;
            }
            its += 1;

            // Look for two consecutive small subdiagonal elements.
            let mut m = nn - 2;
            let (mut p, mut q, mut r) = (0.0, 0.0, 0.0);
            while m >= l {
                let z = h.get(m as usize, m as usize);
                let rr = x - z;
                let ss = y - z;
                p = (rr * ss - w) / h.get(m as usize + 1, m as usize)
                    + h.get(m as usize, m as usize + 1);
                q = h.get(m as usize + 1, m as usize + 1) - z - rr - ss;
                r = h.get(m as usize + 2, m as usize + 1);
                let s = p.abs() + q.abs() + r.abs();
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                let u = h.get(m as usize, m as usize - 1).abs() * (q.abs() + r.abs());
                let v = p.abs()
                    * (h.get(m as usize - 1, m as usize - 1).abs()
                        + z.abs()
                        + h.get(m as usize + 1, m as usize + 1).abs());
                if u <= f64::EPSILON * v {
                    break;
                }
                m -= 1;
            }
            for i in (m as usize + 2)..=(nn as usize) {
                h.set(i, i - 2, 0.0);
                if i > m as usize + 2 {
                    h.set(i, i - 3, 0.0);
                }
            }

            // Double QR step on rows/columns l..=nn.
            for k in (m as usize)..(nn as usize) {
                if k != m as usize {
                    p = h.get(k, k - 1);
                    q = h.get(k + 1, k - 1);
                    r = if k != nn as usize - 1 { h.get(k + 2, k - 1) } else { 0.0 };
                    x = p.abs() + q.abs() + r.abs();
                    if x != 0.0 {
                        p /= x;
                        q /= x;
                        r /= x;
                    }
                }
                let s_len = (p * p + q * q + r * r).sqrt();
                let s = if p >= 0.0 { s_len } else { -s_len };
                if s == 0.0 {
                    continue;
                }
                if k == m as usize {
                    if l != m {
                        let v = -h.get(k, k - 1);
                        h.set(k, k - 1, v);
                    }
                } else {
                    h.set(k, k - 1, -s * x);
                }
                let p_new = p + s;
                let x2 = p_new / s;
                let y2 = q / s;
                let z2 = r / s;
                let q2 = q / p_new;
                let r2 = r / p_new;
                // Row modification.
                for j in k..=(nn as usize) {
                    let mut pj = h.get(k, j) + q2 * h.get(k + 1, j);
                    if k != nn as usize - 1 {
                        pj += r2 * h.get(k + 2, j);
                        let val = h.get(k + 2, j) - pj * z2;
                        h.set(k + 2, j, val);
                    }
                    let val1 = h.get(k + 1, j) - pj * y2;
                    h.set(k + 1, j, val1);
                    let val0 = h.get(k, j) - pj * x2;
                    h.set(k, j, val0);
                }
                // Column modification.
                let mmin = if (nn as usize) < k + 3 { nn as usize } else { k + 3 };
                for i in (l as usize)..=mmin {
                    let mut pi = x2 * h.get(i, k) + y2 * h.get(i, k + 1);
                    if k != nn as usize - 1 {
                        pi += z2 * h.get(i, k + 2);
                        let val = h.get(i, k + 2) - pi * r2;
                        h.set(i, k + 2, val);
                    }
                    let val1 = h.get(i, k + 1) - pi * q2;
                    h.set(i, k + 1, val1);
                    let val0 = h.get(i, k) - pi;
                    h.set(i, k, val0);
                }
            }
        }
    }
    Ok(eig)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_symmetric_diagonal() {
        let a = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 1.0]]).unwrap();
        let e = eigen_symmetric(&a, 1e-12, 50).unwrap();
        assert!(approx(e.values[0], 3.0, 1e-12));
        assert!(approx(e.values[1], 1.0, 1e-12));
    }

    #[test]
    fn test_symmetric_2x2_known() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1.
        let a = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]).unwrap();
        let e = eigen_symmetric(&a, 1e-12, 50).unwrap();
        assert!(approx(e.values[0], 3.0, 1e-10));
        assert!(approx(e.values[1], 1.0, 1e-10));
        // Eigenvector for lambda=3 is (1,1)/sqrt(2) up to sign.
        let v0 = (e.vectors.get(0, 0), e.vectors.get(1, 0));
        assert!(approx(v0.0.abs(), std::f64::consts::FRAC_1_SQRT_2, 1e-10));
        assert!(approx((v0.0 - v0.1).abs(), 0.0, 1e-10));
    }

    #[test]
    fn test_symmetric_residual_and_orthogonality() {
        let a = Matrix::from_rows(&[
            &[4.0, 1.0, -2.0],
            &[1.0, 2.0, 0.0],
            &[-2.0, 0.0, 3.0],
        ])
        .unwrap();
        let e = eigen_symmetric(&a, 1e-13, 100).unwrap();
        for k in 0..3 {
            let v: Vec<f64> = (0..3).map(|r| e.vectors.get(r, k)).collect();
            let av = a.mul_vec(&v).unwrap();
            for r in 0..3 {
                assert!(approx(av[r], e.values[k] * v[r], 1e-9));
            }
        }
        let vtv = e.vectors.transpose().mul(&e.vectors).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(vtv.get(i, j), expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_symmetric_rejects_asymmetric() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[0.0, 1.0]]).unwrap();
        assert!(matches!(
            eigen_symmetric(&a, 1e-12, 50),
            Err(SolveError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_general_real_eigenvalues() {
        // Upper triangular: eigenvalues are the diagonal.
        let a = Matrix::from_rows(&[&[1.0, 5.0, 3.0], &[0.0, 2.0, 7.0], &[0.0, 0.0, 4.0]])
            .unwrap();
        let mut eig = eigenvalues_general(&a, 60).unwrap();
        eig.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        assert!(approx(eig[0].re, 1.0, 1e-9) && approx(eig[0].im, 0.0, 1e-12));
        assert!(approx(eig[1].re, 2.0, 1e-9));
        assert!(approx(eig[2].re, 4.0, 1e-9));
    }

    #[test]
    fn test_general_complex_pair() {
        // Rotation-like matrix [[0,-1],[1,0]] has eigenvalues ±i.
        let a = Matrix::from_rows(&[&[0.0, -1.0], &[1.0, 0.0]]).unwrap();
        let eig = eigenvalues_general(&a, 60).unwrap();
        assert!(approx(eig[0].re, 0.0, 1e-10));
        assert!(approx(eig[0].im.abs(), 1.0, 1e-10));
        assert!(approx(eig[1].im, -eig[0].im, 1e-10));
    }

    #[test]
    fn test_general_matches_trace_det_4x4() {
        let a = Matrix::from_rows(&[
            &[2.0, 1.0, 0.0, 3.0],
            &[-1.0, 4.0, 1.0, 0.0],
            &[0.5, 0.0, 1.0, 2.0],
            &[1.0, 1.0, -1.0, 3.0],
        ])
        .unwrap();
        let eig = eigenvalues_general(&a, 60).unwrap();
        let trace: f64 = (0..4).map(|i| a.get(i, i)).sum();
        let sum_re: f64 = eig.iter().map(|c| c.re).sum();
        let sum_im: f64 = eig.iter().map(|c| c.im).sum();
        assert!(approx(sum_re, trace, 1e-8), "trace {trace} vs {sum_re}");
        assert!(approx(sum_im, 0.0, 1e-8));

        // Product of eigenvalues = determinant.
        let det = crate::linalg::lu_decompose(&a).unwrap().determinant();
        let mut prod = Complex::new(1.0, 0.0);
        for &e in &eig {
            prod = prod * e;
        }
        assert!(approx(prod.re, det, 1e-6 * det.abs().max(1.0)), "det {det} vs {}", prod.re);
        assert!(approx(prod.im, 0.0, 1e-6));
    }

    #[test]
    fn test_general_1x1_and_shape_errors() {
        let a = Matrix::from_rows(&[&[7.0]]).unwrap();
        let eig = eigenvalues_general(&a, 30).unwrap();
        assert!(approx(eig[0].re, 7.0, 1e-12));
        assert!(eigenvalues_general(&Matrix::zeros(2, 3), 30).is_err());
    }
}
