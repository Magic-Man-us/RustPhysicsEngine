//! Tridiagonal linear solve (Thomas algorithm).
//!
//! Reference: Press et al., *Numerical Recipes*, §2.4. Solves
//! sub[i-1]·x[i-1] + diag[i]·x[i] + sup[i]·x[i+1] = rhs[i] in O(n).

use crate::error::SolveError;

/// Pivots below this threshold abort with `Singular`.
const PIVOT_THRESHOLD: f64 = 1e-300;

/// Solves a tridiagonal system with the Thomas algorithm.
///
/// `diag` and `rhs` have length n; `sub` (below-diagonal) and `sup`
/// (above-diagonal) have length n−1. Numerically stable for diagonally
/// dominant or symmetric positive-definite systems.
pub fn thomas_solve(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, SolveError> {
    let n = diag.len();
    if n == 0 {
        return Err(SolveError::InvalidArgument("thomas_solve requires a non-empty diagonal"));
    }
    if rhs.len() != n {
        return Err(SolveError::DimensionMismatch { expected: n, got: rhs.len() });
    }
    if sub.len() != n - 1 {
        return Err(SolveError::DimensionMismatch { expected: n - 1, got: sub.len() });
    }
    if sup.len() != n - 1 {
        return Err(SolveError::DimensionMismatch { expected: n - 1, got: sup.len() });
    }

    // Forward sweep.
    let mut c_prime = vec![0.0; n - 1];
    let mut d_prime = vec![0.0; n];
    if diag[0].abs() < PIVOT_THRESHOLD {
        return Err(SolveError::Singular);
    }
    if n > 1 {
        c_prime[0] = sup[0] / diag[0];
    }
    d_prime[0] = rhs[0] / diag[0];
    for i in 1..n {
        let denom = diag[i] - sub[i - 1] * c_prime[i - 1];
        if denom.abs() < PIVOT_THRESHOLD {
            return Err(SolveError::Singular);
        }
        if i < n - 1 {
            c_prime[i] = sup[i] / denom;
        }
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution.
    let mut x = d_prime;
    for i in (0..n - 1).rev() {
        x[i] -= c_prime[i] * x[i + 1];
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
    fn test_known_system() {
        // [[2,1,0],[1,2,1],[0,1,2]] x = [4,8,8] → x = [1,2,3]
        let x = thomas_solve(&[1.0, 1.0], &[2.0, 2.0, 2.0], &[1.0, 1.0], &[4.0, 8.0, 8.0]).unwrap();
        assert!(approx(x[0], 1.0, 1e-12));
        assert!(approx(x[1], 2.0, 1e-12));
        assert!(approx(x[2], 3.0, 1e-12));
    }

    #[test]
    fn test_single_equation() {
        let x = thomas_solve(&[], &[4.0], &[], &[8.0]).unwrap();
        assert!(approx(x[0], 2.0, 1e-15));
    }

    #[test]
    fn test_dimension_mismatch() {
        assert!(matches!(
            thomas_solve(&[1.0], &[2.0, 2.0], &[1.0], &[1.0]),
            Err(SolveError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            thomas_solve(&[1.0, 1.0], &[2.0, 2.0], &[1.0], &[1.0, 1.0]),
            Err(SolveError::DimensionMismatch { .. })
        ));
        assert!(thomas_solve(&[], &[], &[], &[]).is_err());
    }

    #[test]
    fn test_singular() {
        assert_eq!(
            thomas_solve(&[], &[0.0], &[], &[1.0]).unwrap_err(),
            SolveError::Singular
        );
    }
}
