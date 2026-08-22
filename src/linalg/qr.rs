//! QR decomposition by Householder reflections and least-squares solve.
//!
//! Reference: Golub & Van Loan, *Matrix Computations*, §5.2:
//! A = Q·R with Q orthogonal (m×m) and R upper trapezoidal (m×n).

use crate::error::SolveError;
use crate::linalg::Matrix;

/// Diagonal entries of R below this threshold are treated as rank
/// deficiency in `least_squares`.
const RANK_THRESHOLD: f64 = 1e-12;

/// QR factorization A = Q·R.
#[derive(Debug, Clone, PartialEq)]
pub struct Qr {
    pub q: Matrix,
    pub r: Matrix,
}

/// Factors A (m×n) as Q·R using Householder reflections.
///
/// Q is m×m orthogonal; R is m×n with zeros below the diagonal.
#[must_use]
pub fn qr_householder(a: &Matrix) -> Qr {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let mut q = Matrix::identity(m);
    let steps = if m > 1 { n.min(m - 1) } else { 0 };

    let mut v = vec![0.0; m];
    for k in 0..steps {
        // Householder vector for column k, rows k..m.
        let mut norm_sq = 0.0;
        for i in k..m {
            let x = r.get(i, k);
            v[i] = x;
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if v[k] >= 0.0 { -norm } else { norm };
        v[k] -= alpha;
        let vtv: f64 = (k..m).map(|i| v[i] * v[i]).sum();
        if vtv == 0.0 {
            continue;
        }

        // R ← H·R with H = I - 2·v·vᵀ/(vᵀv), applied to columns k..n.
        for c in k..n {
            let dot: f64 = (k..m).map(|i| v[i] * r.get(i, c)).sum();
            let scale = 2.0 * dot / vtv;
            for i in k..m {
                let val = r.get(i, c) - scale * v[i];
                r.set(i, c, val);
            }
        }
        // Q ← Q·H (accumulates the product of reflectors so A = Q·R).
        for row in 0..m {
            let dot: f64 = (k..m).map(|i| q.get(row, i) * v[i]).sum();
            let scale = 2.0 * dot / vtv;
            for i in k..m {
                let val = q.get(row, i) - scale * v[i];
                q.set(row, i, val);
            }
        }
        // Zero out the annihilated entries explicitly.
        r.set(k, k, alpha);
        for i in (k + 1)..m {
            r.set(i, k, 0.0);
        }
    }

    Qr { q, r }
}

/// Solves the least-squares problem min ‖A·x − b‖₂ via QR.
///
/// Requires m ≥ n and full column rank; returns `Singular` when R has a
/// negligible diagonal entry, `DimensionMismatch` when `b.len() != m`,
/// and `InvalidArgument` when the system is underdetermined (m < n).
pub fn least_squares(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, SolveError> {
    let m = a.rows;
    let n = a.cols;
    if m < n {
        return Err(SolveError::InvalidArgument("least_squares requires rows >= cols"));
    }
    if b.len() != m {
        return Err(SolveError::DimensionMismatch { expected: m, got: b.len() });
    }
    let Qr { q, r } = qr_householder(a);
    // c = Qᵀ·b (only the first n components are needed).
    let mut c = vec![0.0; n];
    for (j, cj) in c.iter_mut().enumerate() {
        *cj = (0..m).map(|i| q.get(i, j) * b[i]).sum();
    }
    // Back substitution on the top n×n block of R.
    let scale = r.frobenius_norm().max(1.0);
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = c[i];
        for j in (i + 1)..n {
            s -= r.get(i, j) * x[j];
        }
        let d = r.get(i, i);
        if d.abs() < RANK_THRESHOLD * scale {
            return Err(SolveError::Singular);
        }
        x[i] = s / d;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_qr_reconstructs_square() {
        let a = Matrix::from_rows(&[&[12.0, -51.0, 4.0], &[6.0, 167.0, -68.0], &[-4.0, 24.0, -41.0]])
            .unwrap();
        let Qr { q, r } = qr_householder(&a);
        let qr = q.mul(&r).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx(qr.get(i, j), a.get(i, j), 1e-10));
            }
        }
        // R upper triangular
        for i in 1..3 {
            for j in 0..i {
                assert!(approx(r.get(i, j), 0.0, 1e-12));
            }
        }
    }

    #[test]
    fn test_qr_orthogonality() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]).unwrap();
        let Qr { q, .. } = qr_householder(&a);
        let qtq = q.transpose().mul(&q).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(qtq.get(i, j), expected, 1e-12));
            }
        }
    }

    #[test]
    fn test_least_squares_exact_system() {
        let a = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 3.0]]).unwrap();
        let x = least_squares(&a, &[4.0, 9.0]).unwrap();
        assert!(approx(x[0], 2.0, 1e-12) && approx(x[1], 3.0, 1e-12));
    }

    #[test]
    fn test_least_squares_overdetermined_line() {
        // Fit y = 2x + 1 through exact points: residual should be zero.
        let xs = [0.0, 1.0, 2.0, 3.0];
        let mut a = Matrix::zeros(4, 2);
        let mut b = [0.0; 4];
        for (i, &x) in xs.iter().enumerate() {
            a.set(i, 0, 1.0);
            a.set(i, 1, x);
            b[i] = 2.0 * x + 1.0;
        }
        let sol = least_squares(&a, &b).unwrap();
        assert!(approx(sol[0], 1.0, 1e-10) && approx(sol[1], 2.0, 1e-10));
    }

    #[test]
    fn test_least_squares_rank_deficient() {
        let a = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]]).unwrap();
        assert_eq!(least_squares(&a, &[1.0, 2.0, 3.0]).unwrap_err(), SolveError::Singular);
    }

    #[test]
    fn test_least_squares_shape_errors() {
        let a = Matrix::zeros(2, 3);
        assert!(matches!(
            least_squares(&a, &[1.0, 2.0]),
            Err(SolveError::InvalidArgument(_))
        ));
        let a = Matrix::identity(2);
        assert!(matches!(
            least_squares(&a, &[1.0, 2.0, 3.0]),
            Err(SolveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_qr_single_row() {
        let a = Matrix::from_rows(&[&[3.0, 4.0]]).unwrap();
        let Qr { q, r } = qr_householder(&a);
        let qr = q.mul(&r).unwrap();
        assert!(approx(qr.get(0, 0), 3.0, 1e-12) && approx(qr.get(0, 1), 4.0, 1e-12));
    }
}
