pub mod cholesky;
pub mod eigen;
pub mod lu;
pub mod matrix;
pub mod qr;
pub mod sparse;
pub mod svd;
pub mod tridiagonal;

pub use cholesky::{cholesky, cholesky_solve};
pub use eigen::{eigen_symmetric, eigenvalues_general, SymEigen};
pub use lu::{lu_decompose, solve, Lu};
pub use matrix::Matrix;
pub use qr::{least_squares, qr_householder, Qr};
pub use sparse::{conjugate_gradient, pcg_jacobi, CsrMatrix};
pub use svd::{kabsch, pseudoinverse, rank, svd, Svd};
pub use tridiagonal::{eigen_symmetric_tridiagonal, thomas_solve};

use std::ops::{Add, Mul, Sub};

use crate::math::Vec3;

const SINGULARITY_THRESHOLD: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub data: [[f64; 3]; 3],
}

impl Mat3 {
    /// Returns the 3x3 zero matrix.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            data: [[0.0; 3]; 3],
        }
    }

    /// Returns the 3x3 identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Constructs a 3x3 matrix from three row arrays.
    #[must_use]
    pub fn from_rows(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> Self {
        Self { data: [r0, r1, r2] }
    }

    /// Computes the determinant using the Sarrus rule (cofactor expansion along the first row).
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let d = &self.data;
        d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
    }

    /// Returns the transpose of this matrix: A^T[i][j] = A[j][i].
    #[must_use]
    pub fn transpose(&self) -> Self {
        let d = &self.data;
        Self {
            data: [
                [d[0][0], d[1][0], d[2][0]],
                [d[0][1], d[1][1], d[2][1]],
                [d[0][2], d[1][2], d[2][2]],
            ],
        }
    }

    /// Computes the matrix inverse via the adjugate method: A⁻¹ = adj(A) / det(A).
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < SINGULARITY_THRESHOLD {
            return None;
        }
        let d = &self.data;
        let inv_det = 1.0 / det;

        // Cofactor matrix, transposed (adjugate), scaled by 1/det
        Some(Self {
            data: [
                [
                    (d[1][1] * d[2][2] - d[1][2] * d[2][1]) * inv_det,
                    (d[0][2] * d[2][1] - d[0][1] * d[2][2]) * inv_det,
                    (d[0][1] * d[1][2] - d[0][2] * d[1][1]) * inv_det,
                ],
                [
                    (d[1][2] * d[2][0] - d[1][0] * d[2][2]) * inv_det,
                    (d[0][0] * d[2][2] - d[0][2] * d[2][0]) * inv_det,
                    (d[0][2] * d[1][0] - d[0][0] * d[1][2]) * inv_det,
                ],
                [
                    (d[1][0] * d[2][1] - d[1][1] * d[2][0]) * inv_det,
                    (d[0][1] * d[2][0] - d[0][0] * d[2][1]) * inv_det,
                    (d[0][0] * d[1][1] - d[0][1] * d[1][0]) * inv_det,
                ],
            ],
        })
    }

    /// Returns the trace (sum of diagonal elements) of the matrix.
    #[must_use]
    pub fn trace(&self) -> f64 {
        self.data[0][0] + self.data[1][1] + self.data[2][2]
    }

    /// Multiplies this matrix by a column vector: result = A × v.
    #[must_use]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        let d = &self.data;
        Vec3::new(
            d[0][0] * v.x + d[0][1] * v.y + d[0][2] * v.z,
            d[1][0] * v.x + d[1][1] * v.y + d[1][2] * v.z,
            d[2][0] * v.x + d[2][1] * v.y + d[2][2] * v.z,
        )
    }

    /// Multiplies two 3x3 matrices: result = A × B.
    #[must_use]
    pub fn mul_mat(&self, other: &Mat3) -> Mat3 {
        let a = &self.data;
        let b = &other.data;
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Mat3 { data: result }
    }

    /// Returns a uniform scaling matrix: diag(s, s, s).
    #[must_use]
    pub fn scale(s: f64) -> Self {
        Self {
            data: [
                [s, 0.0, 0.0],
                [0.0, s, 0.0],
                [0.0, 0.0, s],
            ],
        }
    }

    /// Principal axes of a symmetric 3×3 matrix (e.g. an inertia
    /// tensor): eigenvalues in descending order paired with unit
    /// eigenvectors as the columns of the returned matrix.
    ///
    /// Fails with `InvalidArgument` when the matrix is not symmetric.
    pub fn principal_axes(&self) -> Result<([f64; 3], Mat3), crate::error::SolveError> {
        let e = eigen_symmetric(&Matrix::from_mat3(self), 1e-13, 100)?;
        let values = [e.values[0], e.values[1], e.values[2]];
        let v = &e.vectors;
        let axes = Mat3::from_rows(
            [v.get(0, 0), v.get(0, 1), v.get(0, 2)],
            [v.get(1, 0), v.get(1, 1), v.get(1, 2)],
            [v.get(2, 0), v.get(2, 1), v.get(2, 2)],
        );
        Ok((values, axes))
    }

    /// Eigen-decomposition of a symmetric 3×3 matrix by a local cyclic
    /// Jacobi iteration (no dense-matrix machinery): eigenvalues in
    /// descending order paired with unit eigenvector columns.
    ///
    /// # Panics
    /// Panics if the matrix is not symmetric within 1e-8·‖A‖.
    #[must_use]
    pub fn principal_axes_3x3(&self) -> ([f64; 3], Mat3) {
        let d = &self.data;
        let scale = self
            .data
            .iter()
            .flatten()
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1.0);
        assert!(
            (d[0][1] - d[1][0]).abs() <= 1e-8 * scale
                && (d[0][2] - d[2][0]).abs() <= 1e-8 * scale
                && (d[1][2] - d[2][1]).abs() <= 1e-8 * scale,
            "principal_axes_3x3 requires a symmetric matrix"
        );
        let mut a = self.data;
        let mut v = [[0.0; 3]; 3];
        for (i, row) in v.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for _sweep in 0..64 {
            let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
            if off < 1e-30 * scale * scale {
                break;
            }
            for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
                let apq = a[p][q];
                if apq.abs() <= f64::EPSILON * scale {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..3 {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..3 {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
        // Sort eigenvalues descending, permuting the eigenvector columns.
        let mut order = [0usize, 1, 2];
        let vals = [a[0][0], a[1][1], a[2][2]];
        order.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap_or(std::cmp::Ordering::Equal));
        let sorted_vals = [vals[order[0]], vals[order[1]], vals[order[2]]];
        let mut axes = [[0.0; 3]; 3];
        for (new_c, &old_c) in order.iter().enumerate() {
            for r in 0..3 {
                axes[r][new_c] = v[r][old_c];
            }
        }
        (sorted_vals, Mat3 { data: axes })
    }

    /// Multiplies every element of the matrix by a scalar.
    #[must_use]
    pub fn mul_scalar(&self, s: f64) -> Self {
        let d = &self.data;
        Self {
            data: [
                [d[0][0] * s, d[0][1] * s, d[0][2] * s],
                [d[1][0] * s, d[1][1] * s, d[1][2] * s],
                [d[2][0] * s, d[2][1] * s, d[2][2] * s],
            ],
        }
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;
    fn mul(self, rhs: Mat3) -> Mat3 {
        self.mul_mat(&rhs)
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        self.mul_vec(rhs)
    }
}

impl Add<Mat3> for Mat3 {
    type Output = Mat3;
    fn add(self, rhs: Mat3) -> Mat3 {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        Mat3 { data: result }
    }
}

impl Sub<Mat3> for Mat3 {
    type Output = Mat3;
    fn sub(self, rhs: Mat3) -> Mat3 {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        Mat3 { data: result }
    }
}

// --- Rotation matrices ---

/// Rotation matrix about the x-axis by the given angle in radians.
#[must_use]
pub fn rotation_x(angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    Mat3::from_rows(
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    )
}

/// Rotation matrix about the y-axis by the given angle in radians.
#[must_use]
pub fn rotation_y(angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    Mat3::from_rows(
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    )
}

/// Rotation matrix about the z-axis by the given angle in radians.
#[must_use]
pub fn rotation_z(angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    Mat3::from_rows(
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    )
}

/// Rodrigues' rotation formula: rotate by `angle` radians about `axis`.
/// The axis is normalized internally.
#[must_use]
pub fn rotation_axis_angle(axis: Vec3, angle: f64) -> Mat3 {
    let n = axis.normalized();
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;

    Mat3::from_rows(
        [
            t * n.x * n.x + c,
            t * n.x * n.y - s * n.z,
            t * n.x * n.z + s * n.y,
        ],
        [
            t * n.y * n.x + s * n.z,
            t * n.y * n.y + c,
            t * n.y * n.z - s * n.x,
        ],
        [
            t * n.z * n.x - s * n.y,
            t * n.z * n.y + s * n.x,
            t * n.z * n.z + c,
        ],
    )
}

// --- Coordinate transformations ---

/// Returns (r, theta, phi) where theta is the polar angle from +z and phi is the azimuthal angle from +x.
#[must_use]
pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    if r < SINGULARITY_THRESHOLD {
        return (0.0, 0.0, 0.0);
    }
    let theta = (z / r).clamp(-1.0, 1.0).acos();
    let phi = y.atan2(x);
    (r, theta, phi)
}

/// Converts spherical coordinates (r, theta, phi) to Cartesian (x, y, z).
#[must_use]
pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    (
        r * sin_theta * cos_phi,
        r * sin_theta * sin_phi,
        r * cos_theta,
    )
}

/// Returns (rho, phi, z) where rho is the radial distance in the xy-plane and phi is the azimuthal angle from +x.
#[must_use]
pub fn cartesian_to_cylindrical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rho = (x * x + y * y).sqrt();
    let phi = y.atan2(x);
    (rho, phi, z)
}

/// Converts cylindrical coordinates (rho, phi, z) to Cartesian (x, y, z).
#[must_use]
pub fn cylindrical_to_cartesian(rho: f64, phi: f64, z: f64) -> (f64, f64, f64) {
    let (sin_phi, cos_phi) = phi.sin_cos();
    (rho * cos_phi, rho * sin_phi, z)
}

/// Converts 2D polar coordinates (r, theta) to Cartesian (x, y).
#[must_use]
pub fn polar_to_cartesian(r: f64, theta: f64) -> (f64, f64) {
    let (sin_t, cos_t) = theta.sin_cos();
    (r * cos_t, r * sin_t)
}

/// Returns (r, theta) where theta is the angle from +x.
#[must_use]
pub fn cartesian_to_polar(x: f64, y: f64) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    let theta = y.atan2(x);
    (r, theta)
}

/// A dense 4x4 matrix with fixed-size storage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub data: [[f64; 4]; 4],
}

impl Mat4 {
    /// The 4x4 zero matrix.
    #[must_use]
    pub fn zero() -> Self {
        Self { data: [[0.0; 4]; 4] }
    }

    /// The 4x4 identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        let mut m = Self::zero();
        for i in 0..4 {
            m.data[i][i] = 1.0;
        }
        m
    }

    /// Construct from four row arrays.
    #[must_use]
    pub fn from_rows(r0: [f64; 4], r1: [f64; 4], r2: [f64; 4], r3: [f64; 4]) -> Self {
        Self {
            data: [r0, r1, r2, r3],
        }
    }

    /// Matrix product.
    #[must_use]
    pub fn mul_mat(&self, other: &Mat4) -> Mat4 {
        let mut out = Mat4::zero();
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for (k, ok) in other.data.iter().enumerate() {
                    s += self.data[i][k] * ok[j];
                }
                out.data[i][j] = s;
            }
        }
        out
    }

    /// Matrix-vector product.
    #[must_use]
    pub fn mul_vec4(&self, v: [f64; 4]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for (o, row) in out.iter_mut().zip(&self.data) {
            *o = row.iter().zip(&v).map(|(a, b)| a * b).sum();
        }
        out
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Mat4 {
        let mut out = Mat4::zero();
        for i in 0..4 {
            for j in 0..4 {
                out.data[i][j] = self.data[j][i];
            }
        }
        out
    }

    /// Trace.
    #[must_use]
    pub fn trace(&self) -> f64 {
        (0..4).map(|i| self.data[i][i]).sum()
    }

    /// Determinant via LU on the general dense matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        match lu_decompose(&self.to_matrix()) {
            Ok(lu) => lu.determinant(),
            Err(_) => 0.0,
        }
    }

    /// Inverse (None if singular).
    #[must_use]
    pub fn inverse(&self) -> Option<Mat4> {
        let inv = lu_decompose(&self.to_matrix()).and_then(|lu| lu.inverse()).ok()?;
        Some(Mat4::from_matrix(&inv))
    }

    /// Convert to a general dense matrix.
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        Matrix::from_fn(4, 4, |i, j| self.data[i][j])
    }

    /// Convert from a 4x4 general dense matrix.
    ///
    /// # Panics
    /// Panics unless `m` is 4x4.
    #[must_use]
    pub fn from_matrix(m: &Matrix) -> Mat4 {
        assert!(m.rows == 4 && m.cols == 4, "Mat4::from_matrix needs 4x4");
        let mut out = Mat4::zero();
        for i in 0..4 {
            for j in 0..4 {
                out.data[i][j] = m.get(i, j);
            }
        }
        out
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;
    fn mul(self, rhs: Mat4) -> Mat4 {
        self.mul_mat(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    const APPROX_EPSILON: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < APPROX_EPSILON
    }

    fn mat3_approx_eq(a: &Mat3, b: &Mat3) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if !approx(a.data[i][j], b.data[i][j]) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_identity_determinant() {
        assert!(approx(Mat3::identity().determinant(), 1.0));
    }

    #[test]
    fn test_zero_determinant() {
        assert!(approx(Mat3::zero().determinant(), 0.0));
    }

    #[test]
    fn test_transpose_identity() {
        assert_eq!(Mat3::identity().transpose(), Mat3::identity());
    }

    #[test]
    fn test_trace() {
        let m = Mat3::from_rows([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]);
        assert!(approx(m.trace(), 10.0));
    }

    #[test]
    fn test_inverse_times_original_is_identity() {
        let m = Mat3::from_rows([1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]);
        let inv = m.inverse().expect("matrix should be invertible");
        let product = m * inv;
        assert!(
            mat3_approx_eq(&product, &Mat3::identity()),
            "M * M^-1 should equal I, got {:?}",
            product
        );
    }

    #[test]
    fn test_singular_matrix_has_no_inverse() {
        let m = Mat3::from_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        assert!(m.inverse().is_none());
    }

    #[test]
    fn test_mul_vec() {
        let m = Mat3::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert!(approx(result.x, 1.0) && approx(result.y, 2.0) && approx(result.z, 3.0));
    }

    #[test]
    fn test_mul_mat_identity() {
        let m = Mat3::from_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        let result = m * Mat3::identity();
        assert!(mat3_approx_eq(&result, &m));
    }

    #[test]
    fn test_add_sub() {
        let a = Mat3::identity();
        let b = Mat3::identity();
        let sum = a + b;
        assert!(approx(sum.data[0][0], 2.0));
        let diff = sum - a;
        assert!(mat3_approx_eq(&diff, &Mat3::identity()));
    }

    #[test]
    fn test_scale() {
        let s = Mat3::scale(3.0);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = s * v;
        assert!(approx(result.x, 3.0) && approx(result.y, 6.0) && approx(result.z, 9.0));
    }

    #[test]
    fn test_mul_scalar() {
        let m = Mat3::identity();
        let scaled = m.mul_scalar(5.0);
        assert!(approx(scaled.data[0][0], 5.0));
        assert!(approx(scaled.data[0][1], 0.0));
    }

    // --- Rotation tests ---

    #[test]
    fn test_rotation_z_90_maps_x_to_y() {
        let r = rotation_z(PI / 2.0);
        let x_hat = Vec3::new(1.0, 0.0, 0.0);
        let result = r * x_hat;
        assert!(
            approx(result.x, 0.0) && approx(result.y, 1.0) && approx(result.z, 0.0),
            "90-deg rotation about z should map x-hat to y-hat, got {:?}",
            result
        );
    }

    #[test]
    fn test_rotation_x_90_maps_y_to_z() {
        let r = rotation_x(PI / 2.0);
        let y_hat = Vec3::new(0.0, 1.0, 0.0);
        let result = r * y_hat;
        assert!(
            approx(result.x, 0.0) && approx(result.y, 0.0) && approx(result.z, 1.0),
            "90-deg rotation about x should map y-hat to z-hat, got {:?}",
            result
        );
    }

    #[test]
    fn test_rotation_y_90_maps_z_to_x() {
        let r = rotation_y(PI / 2.0);
        let z_hat = Vec3::new(0.0, 0.0, 1.0);
        let result = r * z_hat;
        assert!(
            approx(result.x, 1.0) && approx(result.y, 0.0) && approx(result.z, 0.0),
            "90-deg rotation about y should map z-hat to x-hat, got {:?}",
            result
        );
    }

    #[test]
    fn test_rotation_matrix_is_orthogonal() {
        let r = rotation_axis_angle(Vec3::new(1.0, 1.0, 1.0), 1.23);
        let rt_r = r.transpose() * r;
        assert!(
            mat3_approx_eq(&rt_r, &Mat3::identity()),
            "R^T * R should equal I for rotation matrices, got {:?}",
            rt_r
        );
    }

    #[test]
    fn test_rotation_matrix_determinant_is_one() {
        let r = rotation_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.75);
        let det = r.determinant();
        assert!(
            approx(det, 1.0),
            "Rotation matrix determinant should be 1, got {det}",
        );
    }

    #[test]
    fn test_axis_angle_matches_rotation_z() {
        let angle = 1.2;
        let rz = rotation_z(angle);
        let raa = rotation_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle);
        assert!(
            mat3_approx_eq(&rz, &raa),
            "Axis-angle about z should match rotation_z"
        );
    }

    // --- Coordinate transformation roundtrip tests ---

    #[test]
    fn test_cartesian_spherical_roundtrip() {
        let (x, y, z) = (3.0, 4.0, 5.0);
        let (r, theta, phi) = cartesian_to_spherical(x, y, z);
        let (x2, y2, z2) = spherical_to_cartesian(r, theta, phi);
        assert!(
            approx(x, x2) && approx(y, y2) && approx(z, z2),
            "Spherical roundtrip failed: ({x}, {y}, {z}) -> ({x2}, {y2}, {z2})"
        );
    }

    #[test]
    fn test_cartesian_cylindrical_roundtrip() {
        let (x, y, z) = (-2.0, 7.0, 3.5);
        let (rho, phi, z_cyl) = cartesian_to_cylindrical(x, y, z);
        let (x2, y2, z2) = cylindrical_to_cartesian(rho, phi, z_cyl);
        assert!(
            approx(x, x2) && approx(y, y2) && approx(z, z2),
            "Cylindrical roundtrip failed: ({x}, {y}, {z}) -> ({x2}, {y2}, {z2})"
        );
    }

    #[test]
    fn test_polar_roundtrip() {
        let (x, y) = (3.0, -4.0);
        let (r, theta) = cartesian_to_polar(x, y);
        let (x2, y2) = polar_to_cartesian(r, theta);
        assert!(
            approx(x, x2) && approx(y, y2),
            "Polar roundtrip failed: ({x}, {y}) -> ({x2}, {y2})"
        );
    }

    #[test]
    fn test_spherical_known_values() {
        // Point on +z axis
        let (r, theta, _phi) = cartesian_to_spherical(0.0, 0.0, 5.0);
        assert!(approx(r, 5.0));
        assert!(approx(theta, 0.0));

        // Point on +x axis
        let (r, theta, phi) = cartesian_to_spherical(3.0, 0.0, 0.0);
        assert!(approx(r, 3.0));
        assert!(approx(theta, PI / 2.0));
        assert!(approx(phi, 0.0));
    }

    #[test]
    fn test_origin_spherical() {
        let (r, theta, phi) = cartesian_to_spherical(0.0, 0.0, 0.0);
        assert!(approx(r, 0.0) && approx(theta, 0.0) && approx(phi, 0.0));
    }

    #[test]
    fn test_mul_vec_non_identity() {
        let m = Mat3::from_rows([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = m.mul_vec(v);
        assert!(approx(result.x, 2.0) && approx(result.y, 6.0) && approx(result.z, 12.0));
    }

    #[test]
    fn test_mul_mat_non_trivial() {
        let a = Mat3::from_rows([1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let b = Mat3::from_rows([1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let c = a.mul_mat(&b);
        // c[0][0] = 1*1+2*3+0*0 = 7, c[0][1] = 1*0+2*1+0*0 = 2
        assert!(approx(c.data[0][0], 7.0), "got {}", c.data[0][0]);
        assert!(approx(c.data[0][1], 2.0), "got {}", c.data[0][1]);
        assert!(approx(c.data[1][0], 3.0), "got {}", c.data[1][0]);
    }

    // --- Mat4 tests ---

    fn mat4_approx_eq(a: &Mat4, b: &Mat4, tol: f64) -> bool {
        (0..4).all(|i| (0..4).all(|j| (a.data[i][j] - b.data[i][j]).abs() < tol))
    }

    /// Diagonally dominant (hence well-conditioned) random 4×4 matrix.
    fn random_well_conditioned(rng: &mut crate::monte_carlo::Rng) -> Mat4 {
        let mut m = Mat4::zero();
        for i in 0..4 {
            for j in 0..4 {
                m.data[i][j] = rng.next_f64() * 2.0 - 1.0;
            }
            m.data[i][i] += 5.0;
        }
        m
    }

    #[test]
    fn test_mat4_trace() {
        let m = Mat4::from_rows(
            [1.0, 9.0, 9.0, 9.0],
            [9.0, 2.0, 9.0, 9.0],
            [9.0, 9.0, 3.0, 9.0],
            [9.0, 9.0, 9.0, 4.0],
        );
        // Sum of the diagonal only.
        assert!(approx(m.trace(), 10.0));
        assert!(approx(Mat4::identity().trace(), 4.0));
        assert!(approx(Mat4::zero().trace(), 0.0));
        // Transpose-invariance and the cyclic property tr(AB) = tr(BA).
        assert!(approx(m.trace(), m.transpose().trace()));
        let mut rng = crate::monte_carlo::Rng::new(4_101);
        let a = random_well_conditioned(&mut rng);
        let b = random_well_conditioned(&mut rng);
        assert!(approx(a.mul_mat(&b).trace(), b.mul_mat(&a).trace()));
        // Trace of a similarity transform is invariant: tr(P⁻¹AP)=tr(A).
        let p_inv = b.inverse().unwrap();
        let similar = p_inv.mul_mat(&a).mul_mat(&b);
        assert!((similar.trace() - a.trace()).abs() < 1e-9);
    }

    #[test]
    fn test_mat4_inverse() {
        // Known diagonal matrix: inverse is the reciprocal diagonal.
        let d = Mat4::from_rows(
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
        );
        let di = d.inverse().unwrap();
        let expect = Mat4::from_rows(
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.125],
        );
        assert!(mat4_approx_eq(&di, &expect, 1e-12));

        // Seeded well-conditioned matrices: A·A⁻¹ = A⁻¹·A = I.
        let mut rng = crate::monte_carlo::Rng::new(4_102);
        for _ in 0..10 {
            let a = random_well_conditioned(&mut rng);
            let inv = a.inverse().unwrap();
            assert!(mat4_approx_eq(&a.mul_mat(&inv), &Mat4::identity(), 1e-10));
            assert!(mat4_approx_eq(&inv.mul_mat(&a), &Mat4::identity(), 1e-10));
            // det(A⁻¹) = 1/det(A) and (A⁻¹)⁻¹ = A.
            assert!((inv.determinant() * a.determinant() - 1.0).abs() < 1e-9);
            assert!(mat4_approx_eq(&inv.inverse().unwrap(), &a, 1e-9));
            // The inverse transpose commutes: (Aᵀ)⁻¹ = (A⁻¹)ᵀ.
            assert!(mat4_approx_eq(
                &a.transpose().inverse().unwrap(),
                &inv.transpose(),
                1e-9
            ));
        }
        // A singular matrix (duplicated row) has no inverse.
        let singular = Mat4::from_rows(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 1.0, 2.0, 3.0],
        );
        assert!(singular.inverse().is_none());
        assert!(Mat4::zero().inverse().is_none());
    }

    #[test]
    fn test_mat4_mul_operator_matches_mul_mat() {
        let mut rng = crate::monte_carlo::Rng::new(4_103);
        let a = random_well_conditioned(&mut rng);
        let b = random_well_conditioned(&mut rng);
        let c = random_well_conditioned(&mut rng);
        // The operator is exactly the named method.
        assert_eq!(a * b, a.mul_mat(&b));
        // Identity is neutral on both sides.
        assert_eq!(a * Mat4::identity(), a);
        assert_eq!(Mat4::identity() * a, a);
        // Associativity, and agreement with the matrix-vector product.
        assert!(mat4_approx_eq(&((a * b) * c), &(a * (b * c)), 1e-9));
        let v = [1.0, -2.0, 0.5, 3.0];
        let via_matrix = (a * b).mul_vec4(v);
        let via_sequence = a.mul_vec4(b.mul_vec4(v));
        for k in 0..4 {
            assert!((via_matrix[k] - via_sequence[k]).abs() < 1e-9);
        }
        // det(AB) = det(A)·det(B).
        assert!(((a * b).determinant() - a.determinant() * b.determinant()).abs() < 1e-6);
    }

    #[test]
    fn test_principal_axes_of_symmetric_matrices() {
        // Diagonal inertia-like tensor: eigenvalues are the diagonal in
        // descending order and the axes are the coordinate directions.
        let d = Mat3::from_rows([5.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]);
        let (vals, axes) = d.principal_axes().unwrap();
        assert!(approx(vals[0], 5.0) && approx(vals[1], 3.0) && approx(vals[2], 1.0));
        for c in 0..3 {
            let v = Vec3::new(axes.data[0][c], axes.data[1][c], axes.data[2][c]);
            // Unit eigenvector along a coordinate axis (sign-agnostic).
            assert!(approx(v.magnitude(), 1.0));
            let comps = [v.x.abs(), v.y.abs(), v.z.abs()];
            assert!(comps.iter().filter(|&&x| x > 1e-9).count() == 1);
        }

        // General symmetric matrix: A·v = λ·v with orthonormal columns.
        let a = Mat3::from_rows(
            [4.0, 1.0, -2.0],
            [1.0, 3.0, 0.5],
            [-2.0, 0.5, 6.0],
        );
        let (vals, axes) = a.principal_axes().unwrap();
        assert!(vals[0] >= vals[1] && vals[1] >= vals[2], "not descending: {vals:?}");
        for (c, &lambda) in vals.iter().enumerate() {
            let v = Vec3::new(axes.data[0][c], axes.data[1][c], axes.data[2][c]);
            assert!(approx(v.magnitude(), 1.0), "column {c} not unit");
            let av = a.mul_vec(v);
            assert!(av.distance_to(&(v * lambda)) < 1e-8, "A·v != λ·v for column {c}");
        }
        // Columns are mutually orthogonal: VᵀV = I.
        assert!(mat3_approx_eq(&axes.transpose().mul_mat(&axes), &Mat3::identity()));
        // Invariants: eigenvalues reproduce the trace and determinant.
        assert!(approx(vals.iter().sum::<f64>(), a.trace()));
        assert!((vals[0] * vals[1] * vals[2] - a.determinant()).abs() < 1e-8);
        // Reconstruction: A = V·diag(λ)·Vᵀ.
        let diag = Mat3::from_rows(
            [vals[0], 0.0, 0.0],
            [0.0, vals[1], 0.0],
            [0.0, 0.0, vals[2]],
        );
        assert!(mat3_approx_eq(&axes.mul_mat(&diag).mul_mat(&axes.transpose()), &a));
        // Non-symmetric input is rejected.
        assert!(Mat3::from_rows([1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
            .principal_axes()
            .is_err());
    }

    #[test]
    fn test_mat3_approx_eq_different() {
        let a = Mat3::identity();
        let b = Mat3::from_rows([2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(!mat3_approx_eq(&a, &b));
    }
}
