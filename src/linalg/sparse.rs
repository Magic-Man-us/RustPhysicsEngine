//! Compressed sparse row (CSR) matrices and conjugate-gradient solvers.
//!
//! Reference: Golub & Van Loan §11.5 (CG), Saad, *Iterative Methods for
//! Sparse Linear Systems* §9.2 (Jacobi-preconditioned CG).

use crate::error::SolveError;
use crate::linalg::Matrix;

/// Sparse matrix in CSR form: row r's entries live at indices
/// `row_ptr[r]..row_ptr[r+1]` of `col_idx`/`vals`.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<f64>,
}

impl CsrMatrix {
    /// Builds a CSR matrix from (row, col, value) triplets. Duplicate
    /// positions are summed; explicit zeros are kept.
    ///
    /// # Panics
    /// Panics if any triplet lies outside the given shape.
    #[must_use]
    pub fn from_triplets(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> Self {
        for &(r, c, _) in entries {
            assert!(r < rows && c < cols, "triplet ({r}, {c}) outside {rows}x{cols}");
        }
        let mut sorted: Vec<(usize, usize, f64)> = entries.to_vec();
        sorted.sort_by_key(|&(r, c, _)| (r, c));

        let mut merged: Vec<(usize, usize, f64)> = Vec::with_capacity(sorted.len());
        for &(r, c, v) in &sorted {
            match merged.last_mut() {
                Some(last) if last.0 == r && last.1 == c => last.2 += v,
                _ => merged.push((r, c, v)),
            }
        }

        let mut row_ptr = vec![0usize; rows + 1];
        for &(r, _, _) in &merged {
            row_ptr[r + 1] += 1;
        }
        for r in 0..rows {
            row_ptr[r + 1] += row_ptr[r];
        }
        let col_idx = merged.iter().map(|e| e.1).collect();
        let vals = merged.iter().map(|e| e.2).collect();
        Self { rows, cols, row_ptr, col_idx, vals }
    }

    /// Converts a dense matrix, dropping entries with |v| ≤ tol.
    #[must_use]
    pub fn from_dense(m: &Matrix, tol: f64) -> Self {
        let mut row_ptr = Vec::with_capacity(m.rows + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);
        for r in 0..m.rows {
            for c in 0..m.cols {
                let v = m.get(r, c);
                if v.abs() > tol {
                    col_idx.push(c);
                    vals.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Self { rows: m.rows, cols: m.cols, row_ptr, col_idx, vals }
    }

    /// Sparse matrix-vector product A·v.
    ///
    /// # Panics
    /// Panics if `v.len() != self.cols`.
    #[must_use]
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert!(v.len() == self.cols, "mul_vec length mismatch");
        let mut out = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut s = 0.0;
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                s += self.vals[k] * v[self.col_idx[k]];
            }
            out[r] = s;
        }
        out
    }

    /// Negative 2-D Laplacian (SPD) on an nx×ny grid of interior points
    /// with spacing h and Dirichlet (zero) boundary: the 5-point stencil
    /// (4·u − neighbors)/h². Unknown (i, j) has index i·ny + j.
    ///
    /// # Panics
    /// Panics unless nx, ny ≥ 1 and h > 0.
    #[must_use]
    pub fn laplacian_2d(nx: usize, ny: usize, h: f64) -> Self {
        assert!(nx >= 1 && ny >= 1, "laplacian_2d requires nx, ny >= 1");
        assert!(h > 0.0, "laplacian_2d requires h > 0");
        let n = nx * ny;
        let inv_h2 = 1.0 / (h * h);
        let mut entries = Vec::with_capacity(5 * n);
        for i in 0..nx {
            for j in 0..ny {
                let idx = i * ny + j;
                entries.push((idx, idx, 4.0 * inv_h2));
                if i > 0 {
                    entries.push((idx, idx - ny, -inv_h2));
                }
                if i + 1 < nx {
                    entries.push((idx, idx + ny, -inv_h2));
                }
                if j > 0 {
                    entries.push((idx, idx - 1, -inv_h2));
                }
                if j + 1 < ny {
                    entries.push((idx, idx + 1, -inv_h2));
                }
            }
        }
        Self::from_triplets(n, n, &entries)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Conjugate gradient for SPD systems A·x = b starting from `x0`.
///
/// Converges when ‖r‖₂ ≤ tol·max(‖b‖₂, 1); returns `NoConvergence`
/// with the final residual otherwise.
pub fn conjugate_gradient(
    a: &CsrMatrix,
    b: &[f64],
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, SolveError> {
    if a.rows != a.cols {
        return Err(SolveError::InvalidArgument("conjugate_gradient requires a square matrix"));
    }
    if b.len() != a.rows {
        return Err(SolveError::DimensionMismatch { expected: a.rows, got: b.len() });
    }
    if x0.len() != a.rows {
        return Err(SolveError::DimensionMismatch { expected: a.rows, got: x0.len() });
    }
    if tol <= 0.0 {
        return Err(SolveError::InvalidArgument("conjugate_gradient requires tol > 0"));
    }
    let threshold = tol * dot(b, b).sqrt().max(1.0);
    let mut x = x0.to_vec();
    let ax = a.mul_vec(&x);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);
    if rs_old.sqrt() <= threshold {
        return Ok(x);
    }
    for _ in 0..max_iter {
        let ap = a.mul_vec(&p);
        let p_ap = dot(&p, &ap);
        if p_ap <= 0.0 {
            return Err(SolveError::NotPositiveDefinite);
        }
        let alpha = rs_old / p_ap;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() <= threshold {
            return Ok(x);
        }
        let beta = rs_new / rs_old;
        for i in 0..p.len() {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    Err(SolveError::NoConvergence { iters: max_iter, residual: rs_old.sqrt() })
}

/// Jacobi (diagonal) preconditioned conjugate gradient with x₀ = 0.
///
/// Requires strictly positive diagonal entries (fails with
/// `NotPositiveDefinite` otherwise). Convergence criterion matches
/// [`conjugate_gradient`].
pub fn pcg_jacobi(
    a: &CsrMatrix,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, SolveError> {
    if a.rows != a.cols {
        return Err(SolveError::InvalidArgument("pcg_jacobi requires a square matrix"));
    }
    if b.len() != a.rows {
        return Err(SolveError::DimensionMismatch { expected: a.rows, got: b.len() });
    }
    if tol <= 0.0 {
        return Err(SolveError::InvalidArgument("pcg_jacobi requires tol > 0"));
    }
    let n = a.rows;
    // Extract the diagonal for the preconditioner M = diag(A).
    let mut inv_diag = vec![0.0; n];
    for r in 0..n {
        let mut d = 0.0;
        for k in a.row_ptr[r]..a.row_ptr[r + 1] {
            if a.col_idx[k] == r {
                d += a.vals[k];
            }
        }
        if d <= 0.0 {
            return Err(SolveError::NotPositiveDefinite);
        }
        inv_diag[r] = 1.0 / d;
    }

    let threshold = tol * dot(b, b).sqrt().max(1.0);
    let mut x = vec![0.0; n];
    let mut r: Vec<f64> = b.to_vec();
    if dot(&r, &r).sqrt() <= threshold {
        return Ok(x);
    }
    let mut z: Vec<f64> = r.iter().zip(&inv_diag).map(|(ri, di)| ri * di).collect();
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    for _ in 0..max_iter {
        let ap = a.mul_vec(&p);
        let p_ap = dot(&p, &ap);
        if p_ap <= 0.0 {
            return Err(SolveError::NotPositiveDefinite);
        }
        let alpha = rz_old / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= threshold {
            return Ok(x);
        }
        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }
    Err(SolveError::NoConvergence { iters: max_iter, residual: dot(&r, &r).sqrt() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_from_triplets_and_mul_vec() {
        // [[2, 0, 1], [0, 3, 0]]
        let m = CsrMatrix::from_triplets(2, 3, &[(0, 0, 2.0), (0, 2, 1.0), (1, 1, 3.0)]);
        let y = m.mul_vec(&[1.0, 2.0, 3.0]);
        assert_eq!(y, vec![5.0, 6.0]);
    }

    #[test]
    fn test_from_triplets_duplicates_summed() {
        let m = CsrMatrix::from_triplets(1, 1, &[(0, 0, 1.0), (0, 0, 2.5)]);
        assert_eq!(m.vals, vec![3.5]);
        assert_eq!(m.mul_vec(&[2.0]), vec![7.0]);
    }

    #[test]
    fn test_from_triplets_empty_rows() {
        let m = CsrMatrix::from_triplets(3, 3, &[(2, 1, 4.0)]);
        assert_eq!(m.mul_vec(&[0.0, 1.0, 0.0]), vec![0.0, 0.0, 4.0]);
    }

    #[test]
    fn test_from_dense_roundtrip() {
        let d = Matrix::from_rows(&[&[1.0, 0.0], &[1e-14, 2.0]]).unwrap();
        let s = CsrMatrix::from_dense(&d, 1e-12);
        assert_eq!(s.vals.len(), 2);
        assert_eq!(s.mul_vec(&[1.0, 1.0]), vec![1.0, 2.0]);
    }

    #[test]
    fn test_laplacian_2d_row_sums() {
        // Interior unknown of a 3x3 grid: full stencil sums to 0 within the
        // grid only for the center; corner rows sum to 2/h^2.
        let l = CsrMatrix::laplacian_2d(3, 3, 1.0);
        let ones = vec![1.0; 9];
        let y = l.mul_vec(&ones);
        assert!(approx(y[4], 0.0, 1e-12)); // center
        assert!(approx(y[0], 2.0, 1e-12)); // corner
    }

    #[test]
    fn test_cg_solves_spd() {
        let a = CsrMatrix::from_triplets(
            2,
            2,
            &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
        );
        let b = [1.0, 2.0];
        let x = conjugate_gradient(&a, &b, &[0.0, 0.0], 1e-12, 100).unwrap();
        let back = a.mul_vec(&x);
        assert!(approx(back[0], 1.0, 1e-9) && approx(back[1], 2.0, 1e-9));
    }

    #[test]
    fn test_pcg_matches_cg() {
        let a = CsrMatrix::laplacian_2d(4, 4, 0.5);
        let b: Vec<f64> = (0..16).map(|i| (i as f64 * 0.37).sin()).collect();
        let x_cg = conjugate_gradient(&a, &b, &vec![0.0; 16], 1e-12, 1000).unwrap();
        let x_pcg = pcg_jacobi(&a, &b, 1e-12, 1000).unwrap();
        for (u, v) in x_cg.iter().zip(&x_pcg) {
            assert!(approx(*u, *v, 1e-8));
        }
    }

    #[test]
    fn test_cg_rejects_indefinite() {
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, -1.0), (1, 1, -1.0)]);
        assert!(matches!(
            conjugate_gradient(&a, &[1.0, 1.0], &[0.0, 0.0], 1e-10, 50),
            Err(SolveError::NotPositiveDefinite)
        ));
        assert!(matches!(
            pcg_jacobi(&a, &[1.0, 1.0], 1e-10, 50),
            Err(SolveError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn test_cg_dimension_errors() {
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        assert!(conjugate_gradient(&a, &[1.0], &[0.0, 0.0], 1e-10, 10).is_err());
        assert!(pcg_jacobi(&a, &[1.0], 1e-10, 10).is_err());
    }

    #[test]
    fn test_cg_no_convergence_reported() {
        let a = CsrMatrix::laplacian_2d(5, 5, 0.1);
        let b = vec![1.0; 25];
        assert!(matches!(
            conjugate_gradient(&a, &b, &vec![0.0; 25], 1e-14, 1),
            Err(SolveError::NoConvergence { .. })
        ));
    }
}
