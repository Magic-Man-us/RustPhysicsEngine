//! Cholesky factorization of symmetric positive-definite matrices.
//!
//! Reference: Golub & Van Loan, *Matrix Computations*, §4.2:
//! A = L·Lᵀ with L lower triangular and positive diagonal.

use crate::error::SolveError;
use crate::linalg::Matrix;

/// Relative tolerance used for the symmetry pre-check.
const SYMMETRY_TOL: f64 = 1e-8;

/// Factors a symmetric positive-definite matrix as A = L·Lᵀ, returning
/// the lower-triangular factor L.
///
/// Returns `InvalidArgument` for non-square or asymmetric input and
/// `NotPositiveDefinite` when a diagonal pivot is not strictly positive.
pub fn cholesky(a: &Matrix) -> Result<Matrix, SolveError> {
    if !a.is_square() {
        return Err(SolveError::InvalidArgument("cholesky requires a square matrix"));
    }
    let scale = a.frobenius_norm().max(1.0);
    if !a.is_symmetric(SYMMETRY_TOL * scale) {
        return Err(SolveError::InvalidArgument("cholesky requires a symmetric matrix"));
    }
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = a.get(i, j);
            for k in 0..j {
                s -= l.get(i, k) * l.get(j, k);
            }
            if i == j {
                if s <= 0.0 {
                    return Err(SolveError::NotPositiveDefinite);
                }
                l.set(i, j, s.sqrt());
            } else {
                l.set(i, j, s / l.get(j, j));
            }
        }
    }
    Ok(l)
}

/// Solves A·x = b given the Cholesky factor L of A (A = L·Lᵀ), by one
/// forward and one back substitution.
pub fn cholesky_solve(l: &Matrix, b: &[f64]) -> Result<Vec<f64>, SolveError> {
    if !l.is_square() {
        return Err(SolveError::InvalidArgument("cholesky_solve requires a square factor"));
    }
    let n = l.rows;
    if b.len() != n {
        return Err(SolveError::DimensionMismatch { expected: n, got: b.len() });
    }
    // Forward: L·y = b.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l.get(i, j) * y[j];
        }
        let d = l.get(i, i);
        if d == 0.0 {
            return Err(SolveError::Singular);
        }
        y[i] = s / d;
    }
    // Back: Lᵀ·x = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l.get(j, i) * x[j];
        }
        x[i] = s / l.get(i, i);
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
    fn test_cholesky_known_factor() {
        // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
        let a = Matrix::from_rows(&[&[4.0, 2.0], &[2.0, 3.0]]).unwrap();
        let l = cholesky(&a).unwrap();
        assert!(approx(l.get(0, 0), 2.0, 1e-12));
        assert!(approx(l.get(1, 0), 1.0, 1e-12));
        assert!(approx(l.get(1, 1), 2.0_f64.sqrt(), 1e-12));
        assert!(approx(l.get(0, 1), 0.0, 1e-15));
    }

    #[test]
    fn test_cholesky_reconstructs() {
        let a = Matrix::from_rows(&[
            &[6.0, 3.0, 4.0],
            &[3.0, 6.0, 5.0],
            &[4.0, 5.0, 10.0],
        ])
        .unwrap();
        let l = cholesky(&a).unwrap();
        let llt = l.mul(&l.transpose()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx(llt.get(i, j), a.get(i, j), 1e-10));
            }
        }
    }

    #[test]
    fn test_not_positive_definite() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]).unwrap();
        assert_eq!(cholesky(&a).unwrap_err(), SolveError::NotPositiveDefinite);
    }

    #[test]
    fn test_asymmetric_rejected() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[0.0, 1.0]]).unwrap();
        assert!(matches!(cholesky(&a), Err(SolveError::InvalidArgument(_))));
    }

    #[test]
    fn test_cholesky_solve() {
        let a = Matrix::from_rows(&[&[4.0, 2.0], &[2.0, 3.0]]).unwrap();
        let l = cholesky(&a).unwrap();
        let b = [8.0, 7.0];
        let x = cholesky_solve(&l, &b).unwrap();
        let back = a.mul_vec(&x).unwrap();
        assert!(approx(back[0], b[0], 1e-12) && approx(back[1], b[1], 1e-12));
    }

    #[test]
    fn test_cholesky_solve_dimension_mismatch() {
        let l = cholesky(&Matrix::identity(2)).unwrap();
        assert!(matches!(
            cholesky_solve(&l, &[1.0, 2.0, 3.0]),
            Err(SolveError::DimensionMismatch { .. })
        ));
    }
}
