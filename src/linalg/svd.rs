//! Singular value decomposition by one-sided Jacobi rotations.
//!
//! Reference: Golub & Van Loan §8.6 / Demmel & Veselić, "Jacobi's
//! method is more accurate than QR". Produces the thin decomposition
//! A = U·Σ·Vᵀ with U m×n (orthonormal columns where σ > 0), Σ the
//! non-negative singular values in descending order, and Vᵀ n×n.

use crate::error::SolveError;
use crate::linalg::{Mat3, Matrix};
use crate::math::Vec3;

const MAX_SWEEPS: usize = 60;
const JACOBI_EPS: f64 = 1e-15;

/// Thin SVD: A = U·Σ·Vᵀ.
#[derive(Debug, Clone, PartialEq)]
pub struct Svd {
    pub u: Matrix,
    pub sigma: Vec<f64>,
    pub vt: Matrix,
}

/// One-sided Jacobi SVD of an m×n matrix with m ≥ n; for m < n the
/// transpose is factored and the roles of U and V are swapped.
pub fn svd(a: &Matrix) -> Result<Svd, SolveError> {
    if a.rows < a.cols {
        // A = U Σ Vᵀ  ⇔  Aᵀ = V Σ Uᵀ.
        let t = svd(&a.transpose())?;
        return Ok(Svd { u: t.vt.transpose(), sigma: t.sigma, vt: t.u.transpose() });
    }
    let m = a.rows;
    let n = a.cols;
    let mut u = a.clone(); // columns rotated in place
    let mut v = Matrix::identity(n);

    let mut converged = false;
    for _ in 0..MAX_SWEEPS {
        let mut off = 0.0_f64;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                // Gram entries for the (p, q) column pair.
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..m {
                    let up = u.get(i, p);
                    let uq = u.get(i, q);
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if alpha * beta == 0.0 {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt());
                if gamma.abs() <= JACOBI_EPS * (alpha * beta).sqrt() {
                    continue;
                }
                // Rotation angle zeroing the off-diagonal Gram entry.
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let up = u.get(i, p);
                    let uq = u.get(i, q);
                    u.set(i, p, c * up - s * uq);
                    u.set(i, q, s * up + c * uq);
                }
                for i in 0..n {
                    let vp = v.get(i, p);
                    let vq = v.get(i, q);
                    v.set(i, p, c * vp - s * vq);
                    v.set(i, q, s * vp + c * vq);
                }
            }
        }
        if off <= JACOBI_EPS {
            converged = true;
            break;
        }
    }
    if !converged && n > 1 {
        return Err(SolveError::NoConvergence { iters: MAX_SWEEPS, residual: JACOBI_EPS });
    }

    // Column norms are the singular values; normalize U's columns.
    let mut sigma: Vec<f64> = (0..n)
        .map(|c| (0..m).map(|i| u.get(i, c) * u.get(i, c)).sum::<f64>().sqrt())
        .collect();
    // Sort descending, permuting U and V columns together.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| sigma[j].partial_cmp(&sigma[i]).unwrap_or(std::cmp::Ordering::Equal));
    let mut u_sorted = Matrix::zeros(m, n);
    let mut v_sorted = Matrix::zeros(n, n);
    let mut sigma_sorted = vec![0.0; n];
    for (new_c, &old_c) in order.iter().enumerate() {
        sigma_sorted[new_c] = sigma[old_c];
        for i in 0..m {
            u_sorted.set(i, new_c, u.get(i, old_c));
        }
        for i in 0..n {
            v_sorted.set(i, new_c, v.get(i, old_c));
        }
    }
    sigma = sigma_sorted;
    for c in 0..n {
        if sigma[c] > 0.0 {
            for i in 0..m {
                let val = u_sorted.get(i, c) / sigma[c];
                u_sorted.set(i, c, val);
            }
        }
    }
    Ok(Svd { u: u_sorted, sigma, vt: v_sorted.transpose() })
}

/// Moore-Penrose pseudoinverse A⁺ = V·Σ⁺·Uᵀ; singular values below
/// `rcond · σ_max` are treated as zero.
pub fn pseudoinverse(a: &Matrix, rcond: f64) -> Result<Matrix, SolveError> {
    if rcond < 0.0 {
        return Err(SolveError::InvalidArgument("pseudoinverse requires rcond >= 0"));
    }
    let Svd { u, sigma, vt } = svd(a)?;
    let smax = sigma.first().copied().unwrap_or(0.0);
    let cutoff = rcond * smax;
    let n = sigma.len();
    // A+ = V * diag(1/sigma_i) * U^T over the kept singular values.
    let mut inv_sigma_ut = Matrix::zeros(n, a.rows);
    for i in 0..n {
        if sigma[i] > cutoff && sigma[i] > 0.0 {
            let inv = 1.0 / sigma[i];
            for j in 0..a.rows {
                inv_sigma_ut.set(i, j, inv * u.get(j, i));
            }
        }
    }
    vt.transpose().mul(&inv_sigma_ut)
}

/// Numerical rank: the number of singular values greater than `tol`.
#[must_use]
pub fn rank(a: &Matrix, tol: f64) -> usize {
    match svd(a) {
        Ok(s) => s.sigma.iter().filter(|&&x| x > tol).count(),
        Err(_) => 0,
    }
}

/// Kabsch algorithm: the rotation R minimizing Σ‖R·pᵢ − qᵢ‖².
///
/// Both point sets are used as given (no centroid subtraction); center
/// them first for the usual superposition problem. Fails with
/// `DimensionMismatch` when the sets differ in length and
/// `InvalidArgument` when fewer than 3 points are supplied.
pub fn kabsch(p: &[Vec3], q: &[Vec3]) -> Result<Mat3, SolveError> {
    if p.len() != q.len() {
        return Err(SolveError::DimensionMismatch { expected: p.len(), got: q.len() });
    }
    if p.len() < 3 {
        return Err(SolveError::InvalidArgument("kabsch requires at least 3 points"));
    }
    // Cross-covariance H = Σ qᵢ·pᵢᵀ.
    let mut h = Matrix::zeros(3, 3);
    for (pi, qi) in p.iter().zip(q.iter()) {
        let pv = [pi.x, pi.y, pi.z];
        let qv = [qi.x, qi.y, qi.z];
        for r in 0..3 {
            for c in 0..3 {
                let val = h.get(r, c) + qv[r] * pv[c];
                h.set(r, c, val);
            }
        }
    }
    let Svd { u, sigma: _, vt } = svd(&h)?;
    // R = U·diag(1,1,d)·Vᵀ with d = sign(det(U·Vᵀ)) to exclude reflections.
    let uvt = u.mul(&vt)?;
    let det = Mat3::from_rows(
        [uvt.get(0, 0), uvt.get(0, 1), uvt.get(0, 2)],
        [uvt.get(1, 0), uvt.get(1, 1), uvt.get(1, 2)],
        [uvt.get(2, 0), uvt.get(2, 1), uvt.get(2, 2)],
    )
    .determinant();
    let d = if det < 0.0 { -1.0 } else { 1.0 };
    let mut u_fixed = u;
    for i in 0..3 {
        let val = u_fixed.get(i, 2) * d;
        u_fixed.set(i, 2, val);
    }
    let r = u_fixed.mul(&vt)?;
    Ok(Mat3::from_rows(
        [r.get(0, 0), r.get(0, 1), r.get(0, 2)],
        [r.get(1, 0), r.get(1, 1), r.get(1, 2)],
        [r.get(2, 0), r.get(2, 1), r.get(2, 2)],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::rotation_axis_angle;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn reconstruct(s: &Svd) -> Matrix {
        let n = s.sigma.len();
        let mut sig = Matrix::zeros(n, n);
        for i in 0..n {
            sig.set(i, i, s.sigma[i]);
        }
        s.u.mul(&sig).unwrap().mul(&s.vt).unwrap()
    }

    #[test]
    fn test_svd_diagonal() {
        let a = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.0]]).unwrap();
        let s = svd(&a).unwrap();
        assert!(approx(s.sigma[0], 4.0, 1e-12));
        assert!(approx(s.sigma[1], 3.0, 1e-12));
    }

    #[test]
    fn test_svd_reconstructs_rectangular() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]).unwrap();
        let s = svd(&a).unwrap();
        let back = reconstruct(&s);
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx(back.get(i, j), a.get(i, j), 1e-10));
            }
        }
        // Descending, non-negative.
        assert!(s.sigma[0] >= s.sigma[1] && s.sigma[1] >= 0.0);
    }

    #[test]
    fn test_svd_wide_matrix() {
        let a = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
        let s = svd(&a).unwrap();
        let k = s.sigma.len();
        let mut sig = Matrix::zeros(k, k);
        for i in 0..k {
            sig.set(i, i, s.sigma[i]);
        }
        let back = s.u.mul(&sig).unwrap().mul(&s.vt).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx(back.get(i, j), a.get(i, j), 1e-10), "at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_pseudoinverse_of_invertible_is_inverse() {
        let a = Matrix::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]).unwrap();
        let pinv = pseudoinverse(&a, 1e-12).unwrap();
        let prod = a.mul(&pinv).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(prod.get(i, j), expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_pseudoinverse_rank_deficient() {
        // Rank-1 matrix: A+ satisfies A A+ A = A.
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        let pinv = pseudoinverse(&a, 1e-10).unwrap();
        let apa = a.mul(&pinv).unwrap().mul(&a).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx(apa.get(i, j), a.get(i, j), 1e-9));
            }
        }
    }

    #[test]
    fn test_rank() {
        let full = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]]).unwrap();
        assert_eq!(rank(&full, 1e-10), 2);
        let deficient = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        assert_eq!(rank(&deficient, 1e-10), 1);
    }

    #[test]
    fn test_kabsch_recovers_rotation() {
        let r_true = rotation_axis_angle(Vec3::new(1.0, 2.0, 0.5), 0.8);
        let pts = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let rotated: Vec<Vec3> = pts.iter().map(|&p| r_true.mul_vec(p)).collect();
        let r_est = kabsch(&pts, &rotated).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx(r_est.data[i][j], r_true.data[i][j], 1e-9),
                    "R mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_kabsch_errors() {
        let p = [Vec3::new(1.0, 0.0, 0.0)];
        let q = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        assert!(matches!(kabsch(&p, &q), Err(SolveError::DimensionMismatch { .. })));
        assert!(matches!(kabsch(&p, &p), Err(SolveError::InvalidArgument(_))));
    }
}
