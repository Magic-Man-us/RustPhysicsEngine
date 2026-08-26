//! Dense row-major matrix of `f64`.

use std::ops::{Index, IndexMut};

use crate::error::SolveError;
use crate::linalg::Mat3;

/// Dense matrix with row-major storage: element (r, c) lives at
/// `data[r * cols + c]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Matrix of the given shape filled with zeros.
    ///
    /// # Panics
    /// Panics if `rows` or `cols` is zero.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "matrix dimensions must be positive");
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// n×n identity matrix.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Builds a matrix from row slices. All rows must be non-empty and of
    /// equal length; otherwise `SolveError::DimensionMismatch` is returned.
    pub fn from_rows(rows: &[&[f64]]) -> Result<Self, SolveError> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(SolveError::InvalidArgument("from_rows requires non-empty rows"));
        }
        let cols = rows[0].len();
        for r in rows {
            if r.len() != cols {
                return Err(SolveError::DimensionMismatch { expected: cols, got: r.len() });
            }
        }
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            data.extend_from_slice(r);
        }
        Ok(Self { rows: rows.len(), cols, data })
    }

    /// Builds a matrix by evaluating `f(row, col)` at every position.
    ///
    /// # Panics
    /// Panics if `rows` or `cols` is zero.
    #[must_use]
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        assert!(rows > 0 && cols > 0, "matrix dimensions must be positive");
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    /// Bounds-checked element read.
    ///
    /// # Panics
    /// Panics if `r >= rows` or `c >= cols`.
    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        self.data[r * self.cols + c]
    }

    /// Bounds-checked element write.
    ///
    /// # Panics
    /// Panics if `r >= rows` or `c >= cols`.
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        self.data[r * self.cols + c] = v;
    }

    /// Borrow row `r` as a slice.
    ///
    /// # Panics
    /// Panics if `r >= rows`.
    #[must_use]
    pub fn row(&self, r: usize) -> &[f64] {
        assert!(r < self.rows, "row index out of bounds");
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Transpose: `B[c][r] = A[r][c]`.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }

    /// Matrix product A·B; fails with `DimensionMismatch` unless
    /// `self.cols == other.rows`.
    pub fn mul(&self, other: &Self) -> Result<Self, SolveError> {
        if self.cols != other.rows {
            return Err(SolveError::DimensionMismatch { expected: self.cols, got: other.rows });
        }
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self.data[i * self.cols + k];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.data[i * other.cols + j] += aik * other.data[k * other.cols + j];
                }
            }
        }
        Ok(out)
    }

    /// Matrix-vector product A·v; fails with `DimensionMismatch` unless
    /// `self.cols == v.len()`.
    pub fn mul_vec(&self, v: &[f64]) -> Result<Vec<f64>, SolveError> {
        if self.cols != v.len() {
            return Err(SolveError::DimensionMismatch { expected: self.cols, got: v.len() });
        }
        Ok((0..self.rows)
            .map(|i| {
                self.row(i)
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &x)| a * x)
                    .sum()
            })
            .collect())
    }

    /// Element-wise sum A + B; fails with `DimensionMismatch` on shape
    /// disagreement.
    pub fn add(&self, other: &Self) -> Result<Self, SolveError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SolveError::DimensionMismatch {
                expected: self.rows * self.cols,
                got: other.rows * other.cols,
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(Self { rows: self.rows, cols: self.cols, data })
    }

    /// Scalar multiple k·A.
    #[must_use]
    pub fn scale(&self, k: f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&a| a * k).collect(),
        }
    }

    /// Frobenius norm: sqrt(Σ aᵢⱼ²).
    #[must_use]
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&a| a * a).sum::<f64>().sqrt()
    }

    /// True when the matrix is square.
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// True when the matrix is square and |A - Aᵀ| ≤ tol element-wise.
    #[must_use]
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        for r in 0..self.rows {
            for c in (r + 1)..self.cols {
                if (self.get(r, c) - self.get(c, r)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Converts a fixed-size `Mat3` into a 3×3 `Matrix`.
    #[must_use]
    pub fn from_mat3(m: &Mat3) -> Self {
        Self::from_fn(3, 3, |r, c| m.data[r][c])
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        &mut self.data[r * self.cols + c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn test_zeros_identity() {
        let z = Matrix::zeros(2, 3);
        assert_eq!(z.rows, 2);
        assert_eq!(z.cols, 3);
        assert!(z.data.iter().all(|&x| x == 0.0));

        let i = Matrix::identity(3);
        assert!(approx(i.get(0, 0), 1.0) && approx(i.get(1, 2), 0.0));
    }

    #[test]
    fn test_from_rows_and_get_set() {
        let mut m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert!(approx(m.get(1, 0), 3.0));
        m.set(1, 0, 7.0);
        assert!(approx(m.get(1, 0), 7.0));
        assert!(approx(m[(0, 1)], 2.0));
        m[(0, 1)] = 9.0;
        assert!(approx(m.get(0, 1), 9.0));
    }

    #[test]
    fn test_from_rows_ragged_fails() {
        let e = Matrix::from_rows(&[&[1.0, 2.0], &[3.0]]).unwrap_err();
        assert_eq!(e, SolveError::DimensionMismatch { expected: 2, got: 1 });
        assert!(Matrix::from_rows(&[]).is_err());
    }

    #[test]
    fn test_from_fn_row_major() {
        let m = Matrix::from_fn(2, 3, |r, c| (r * 10 + c) as f64);
        assert_eq!(m.data, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
        assert_eq!(m.row(1), &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!(approx(t.get(2, 1), 6.0));
        assert_eq!(t.transpose(), m);
    }

    #[test]
    fn test_mul_identity_and_shapes() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let i = Matrix::identity(2);
        assert_eq!(a.mul(&i).unwrap(), a);
        assert!(a.mul(&Matrix::zeros(3, 2)).is_err());
    }

    #[test]
    fn test_mul_known_product() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_mul_vec() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let y = a.mul_vec(&[1.0, 1.0]).unwrap();
        assert_eq!(y, vec![3.0, 7.0]);
        assert!(a.mul_vec(&[1.0]).is_err());
    }

    #[test]
    fn test_add_scale_norm() {
        let a = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.0]]).unwrap();
        let s = a.add(&a).unwrap();
        assert!(approx(s.get(0, 0), 6.0));
        assert!(a.add(&Matrix::zeros(3, 3)).is_err());
        assert!(approx(a.scale(2.0).get(1, 1), 8.0));
        assert!(approx(a.frobenius_norm(), 5.0));
    }

    #[test]
    fn test_is_square_symmetric() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 5.0]]).unwrap();
        assert!(a.is_square());
        assert!(a.is_symmetric(1e-12));
        let b = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 5.0]]).unwrap();
        assert!(!b.is_symmetric(1e-12));
        assert!(!Matrix::zeros(2, 3).is_symmetric(1e-12));
    }

    #[test]
    fn test_from_mat3() {
        let m3 = Mat3::from_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        let m = Matrix::from_mat3(&m3);
        assert!(approx(m.get(2, 1), 8.0));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds_panics() {
        let _ = Matrix::zeros(2, 2).get(2, 0);
    }
}
