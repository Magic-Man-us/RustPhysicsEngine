//! LU decomposition with partial pivoting (Doolittle form).
//!
//! Reference: Golub & Van Loan, *Matrix Computations*, §3.4.

use crate::error::SolveError;
use crate::linalg::Matrix;

/// Pivots with absolute value below this threshold are treated as zero.
const PIVOT_THRESHOLD: f64 = 1e-12;

/// Packed LU factorization P·A = L·U.
///
/// `lu` stores U on and above the diagonal and the unit-lower-triangular
/// L (implicit ones on the diagonal) below it. `perm[i]` is the row of A
/// that ended up in position i; `sign` is the permutation's parity (±1).
#[derive(Debug, Clone, PartialEq)]
pub struct Lu {
    pub lu: Matrix,
    pub perm: Vec<usize>,
    pub sign: f64,
}

/// Factors a square matrix as P·A = L·U with partial (row) pivoting.
///
/// Returns `SolveError::InvalidArgument` for non-square input and
/// `SolveError::Singular` when a pivot falls below the threshold.
pub fn lu_decompose(a: &Matrix) -> Result<Lu, SolveError> {
    if !a.is_square() {
        return Err(SolveError::InvalidArgument("lu_decompose requires a square matrix"));
    }
    let n = a.rows;
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;

    for k in 0..n {
        // Find the pivot row for column k.
        let mut pivot_row = k;
        let mut pivot_val = lu.get(k, k).abs();
        for r in (k + 1)..n {
            let v = lu.get(r, k).abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val < PIVOT_THRESHOLD {
            return Err(SolveError::Singular);
        }
        if pivot_row != k {
            for c in 0..n {
                let tmp = lu.get(k, c);
                lu.set(k, c, lu.get(pivot_row, c));
                lu.set(pivot_row, c, tmp);
            }
            perm.swap(k, pivot_row);
            sign = -sign;
        }

        let pivot = lu.get(k, k);
        for r in (k + 1)..n {
            let factor = lu.get(r, k) / pivot;
            lu.set(r, k, factor);
            for c in (k + 1)..n {
                let v = lu.get(r, c) - factor * lu.get(k, c);
                lu.set(r, c, v);
            }
        }
    }

    Ok(Lu { lu, perm, sign })
}

impl Lu {
    /// Solves A·x = b by forward and back substitution on the stored
    /// factors. Fails with `DimensionMismatch` if `b.len() != n`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        let n = self.lu.rows;
        if b.len() != n {
            return Err(SolveError::DimensionMismatch { expected: n, got: b.len() });
        }
        // Forward substitution with permuted b: L·y = P·b.
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[self.perm[i]];
            for j in 0..i {
                s -= self.lu.get(i, j) * y[j];
            }
            y[i] = s;
        }
        // Back substitution: U·x = y.
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n {
                s -= self.lu.get(i, j) * x[j];
            }
            let d = self.lu.get(i, i);
            if d.abs() < PIVOT_THRESHOLD {
                return Err(SolveError::Singular);
            }
            x[i] = s / d;
        }
        Ok(x)
    }

    /// Solves A·X = B column by column.
    pub fn solve_matrix(&self, b: &Matrix) -> Result<Matrix, SolveError> {
        let n = self.lu.rows;
        if b.rows != n {
            return Err(SolveError::DimensionMismatch { expected: n, got: b.rows });
        }
        let mut out = Matrix::zeros(n, b.cols);
        let mut col = vec![0.0; n];
        for c in 0..b.cols {
            for r in 0..n {
                col[r] = b.get(r, c);
            }
            let x = self.solve(&col)?;
            for r in 0..n {
                out.set(r, c, x[r]);
            }
        }
        Ok(out)
    }

    /// Determinant of A: sign · Π uᵢᵢ.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let n = self.lu.rows;
        let mut det = self.sign;
        for i in 0..n {
            det *= self.lu.get(i, i);
        }
        det
    }

    /// Inverse of A, computed by solving A·X = I.
    pub fn inverse(&self) -> Result<Matrix, SolveError> {
        self.solve_matrix(&Matrix::identity(self.lu.rows))
    }
}

/// Convenience wrapper: factor `a` and solve A·x = b in one call.
pub fn solve(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, SolveError> {
    lu_decompose(a)?.solve(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_solve_known_system() {
        // 2x + y = 5, x + 3y = 10 → x = 1, y = 3
        let a = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]).unwrap();
        let x = solve(&a, &[5.0, 10.0]).unwrap();
        assert!(approx(x[0], 1.0, 1e-12) && approx(x[1], 3.0, 1e-12));
    }

    #[test]
    fn test_singular_detected() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        assert_eq!(lu_decompose(&a).unwrap_err(), SolveError::Singular);
    }

    #[test]
    fn test_non_square_rejected() {
        let a = Matrix::zeros(2, 3);
        assert!(matches!(lu_decompose(&a), Err(SolveError::InvalidArgument(_))));
    }

    #[test]
    fn test_determinant_with_pivoting() {
        // Requires a row swap; det = -2 for [[0,1],[2,3]] → 0*3 - 1*2 = -2.
        let a = Matrix::from_rows(&[&[0.0, 1.0], &[2.0, 3.0]]).unwrap();
        let lu = lu_decompose(&a).unwrap();
        assert!(approx(lu.determinant(), -2.0, 1e-12));
    }

    #[test]
    fn test_determinant_identity() {
        let lu = lu_decompose(&Matrix::identity(4)).unwrap();
        assert!(approx(lu.determinant(), 1.0, 1e-12));
    }

    #[test]
    fn test_inverse_roundtrip() {
        let a = Matrix::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]).unwrap();
        let inv = lu_decompose(&a).unwrap().inverse().unwrap();
        let prod = a.mul(&inv).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(prod.get(i, j), expected, 1e-12));
            }
        }
    }

    #[test]
    fn test_solve_matrix_multiple_rhs() {
        let a = Matrix::from_rows(&[&[3.0, 1.0], &[1.0, 2.0]]).unwrap();
        let b = Matrix::from_rows(&[&[9.0, 4.0], &[8.0, 3.0]]).unwrap();
        let x = lu_decompose(&a).unwrap().solve_matrix(&b).unwrap();
        let back = a.mul(&x).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx(back.get(i, j), b.get(i, j), 1e-12));
            }
        }
    }

    #[test]
    fn test_solve_wrong_rhs_length() {
        let a = Matrix::identity(3);
        let lu = lu_decompose(&a).unwrap();
        assert!(matches!(lu.solve(&[1.0]), Err(SolveError::DimensionMismatch { .. })));
    }
}
