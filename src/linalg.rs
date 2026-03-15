use std::ops::{Add, Mul, Sub};

use crate::math::constants::PI;
use crate::math::Vec3;

const SINGULARITY_THRESHOLD: f64 = 1e-12;
const APPROX_EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub data: [[f64; 3]; 3],
}

impl Mat3 {
    #[must_use]
    pub fn zero() -> Self {
        Self {
            data: [[0.0; 3]; 3],
        }
    }

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

    #[must_use]
    pub fn from_rows(r0: [f64; 3], r1: [f64; 3], r2: [f64; 3]) -> Self {
        Self { data: [r0, r1, r2] }
    }

    #[must_use]
    pub fn determinant(&self) -> f64 {
        let d = &self.data;
        d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
    }

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

    #[must_use]
    pub fn trace(&self) -> f64 {
        self.data[0][0] + self.data[1][1] + self.data[2][2]
    }

    #[must_use]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        let d = &self.data;
        Vec3::new(
            d[0][0] * v.x + d[0][1] * v.y + d[0][2] * v.z,
            d[1][0] * v.x + d[1][1] * v.y + d[1][2] * v.z,
            d[2][0] * v.x + d[2][1] * v.y + d[2][2] * v.z,
        )
    }

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

#[must_use]
pub fn rotation_x(angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    Mat3::from_rows(
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    )
}

#[must_use]
pub fn rotation_y(angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    Mat3::from_rows(
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    )
}

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

#[must_use]
pub fn cylindrical_to_cartesian(rho: f64, phi: f64, z: f64) -> (f64, f64, f64) {
    let (sin_phi, cos_phi) = phi.sin_cos();
    (rho * cos_phi, rho * sin_phi, z)
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(
            approx(r.determinant(), 1.0),
            "Rotation matrix determinant should be 1, got {}",
            r.determinant()
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
}
