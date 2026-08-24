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


/// Eigen-decomposition of a symmetric tridiagonal matrix by the QL
/// algorithm with implicit shifts (EISPACK `tql2`): returns eigenvalues
/// (ascending) and matching orthonormal eigenvectors.
///
/// # Errors
/// Returns `DimensionMismatch` for inconsistent inputs and
/// `NoConvergence` if an eigenvalue fails to settle in 50 iterations.
pub fn eigen_symmetric_tridiagonal(
    diag: &[f64],
    off: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), SolveError> {
    let n = diag.len();
    if n == 0 {
        return Err(SolveError::InvalidArgument("empty matrix"));
    }
    if off.len() + 1 != n {
        return Err(SolveError::DimensionMismatch { expected: n - 1, got: off.len() });
    }
    let mut d = diag.to_vec();
    let mut e = vec![0.0; n];
    e[..n - 1].copy_from_slice(off);
    // z[k][i]: component k of the (evolving) i-th eigenvector.
    let mut z = vec![vec![0.0; n]; n];
    for (i, row) in z.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for l in 0..n {
        let mut iter = 0;
        loop {
            // Look for a negligible off-diagonal element.
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 50 {
                return Err(SolveError::NoConvergence { iters: 50, residual: e[l].abs() });
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            let sign_r = if g >= 0.0 { r.abs() } else { -r.abs() };
            g = d[m] - d[l] + e[l] / (g + sign_r);
            let (mut s, mut c) = (1.0, 1.0);
            let mut p = 0.0;
            let mut i = m as isize - 1;
            while i >= l as isize {
                let iu = i as usize;
                let mut f = s * e[iu];
                let b = c * e[iu];
                r = f.hypot(g);
                e[iu + 1] = r;
                if r == 0.0 {
                    d[iu + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[iu + 1] - p;
                r = (d[iu] - g) * s + 2.0 * c * b;
                p = s * r;
                d[iu + 1] = g + p;
                g = c * r - b;
                for zk in z.iter_mut() {
                    f = zk[iu + 1];
                    zk[iu + 1] = s * zk[iu] + c * f;
                    zk[iu] = c * zk[iu] - s * f;
                }
                i -= 1;
            }
            if r == 0.0 && i >= l as isize {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    // Sort ascending, carrying eigenvectors.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let vectors: Vec<Vec<f64>> = idx
        .iter()
        .map(|&i| (0..n).map(|k| z[k][i]).collect())
        .collect();
    Ok((values, vectors))
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
    fn test_tridiagonal_eigen_matches_dense() {
        // -2/1 Laplacian: eigenvalues -2 + 2cos(kπ/(n+1)).
        let n = 12;
        let diag = vec![-2.0; n];
        let off = vec![1.0; n - 1];
        let (vals, vecs) = eigen_symmetric_tridiagonal(&diag, &off).unwrap();
        for (k, &v) in vals.iter().enumerate() {
            let exact = -2.0 + 2.0 * ((n - k) as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            assert!(approx(v, exact, 1e-10), "eigenvalue {k}: {v} vs {exact}");
        }
        // Residual ‖T v − λ v‖ and orthonormality.
        for (k, v) in vecs.iter().enumerate() {
            for i in 0..n {
                let mut tv = diag[i] * v[i];
                if i > 0 {
                    tv += off[i - 1] * v[i - 1];
                }
                if i + 1 < n {
                    tv += off[i] * v[i + 1];
                }
                assert!(approx(tv, vals[k] * v[i], 1e-9));
            }
            let norm: f64 = v.iter().map(|u| u * u).sum();
            assert!(approx(norm, 1.0, 1e-10));
        }
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
