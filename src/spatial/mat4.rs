//! 4×4 homogeneous transform matrix (row-major storage, column-vector
//! convention: p' = M·p).
//!
//! References: Foley et al., *Computer Graphics: Principles and
//! Practice*; the OpenGL clip-space conventions for `perspective` and
//! `orthographic` (z mapped to [−1, 1]).

use std::ops::Mul;

use crate::linalg::Mat3;
use crate::math::Vec3;
use crate::quaternion::Quaternion;

const EPS: f64 = 1e-12;

/// 4×4 matrix, row-major: `data[row][col]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub data: [[f64; 4]; 4],
}

impl Mat4 {
    /// Identity transform.
    #[must_use]
    pub fn identity() -> Self {
        let mut data = [[0.0; 4]; 4];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self { data }
    }

    /// Builds from four row arrays.
    #[must_use]
    pub fn from_rows(r0: [f64; 4], r1: [f64; 4], r2: [f64; 4], r3: [f64; 4]) -> Self {
        Self { data: [r0, r1, r2, r3] }
    }

    /// Translation by t.
    #[must_use]
    pub fn translation(t: Vec3) -> Self {
        Self::from_rows(
            [1.0, 0.0, 0.0, t.x],
            [0.0, 1.0, 0.0, t.y],
            [0.0, 0.0, 1.0, t.z],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Per-axis scaling.
    #[must_use]
    pub fn scaling(s: Vec3) -> Self {
        Self::from_rows(
            [s.x, 0.0, 0.0, 0.0],
            [0.0, s.y, 0.0, 0.0],
            [0.0, 0.0, s.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Rotation from a unit quaternion.
    #[must_use]
    pub fn rotation(q: &Quaternion) -> Self {
        let m = q.normalize().to_rotation_matrix();
        Self::from_rows(
            [m[0][0], m[0][1], m[0][2], 0.0],
            [m[1][0], m[1][1], m[1][2], 0.0],
            [m[2][0], m[2][1], m[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Embeds a 3×3 linear map in the upper-left block.
    #[must_use]
    pub fn from_mat3(m: &Mat3) -> Self {
        let d = &m.data;
        Self::from_rows(
            [d[0][0], d[0][1], d[0][2], 0.0],
            [d[1][0], d[1][1], d[1][2], 0.0],
            [d[2][0], d[2][1], d[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Composite transform T·R·S (scale, then rotate, then translate).
    #[must_use]
    pub fn from_trs(t: Vec3, r: &Quaternion, s: Vec3) -> Self {
        Self::translation(t) * Self::rotation(r) * Self::scaling(s)
    }

    /// Right-handed view matrix: camera at `eye` looking toward
    /// `target` with the −Z axis forward in view space.
    ///
    /// # Panics
    /// Panics when eye == target or `up` is parallel to the view
    /// direction.
    #[must_use]
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalized();
        assert!(f.magnitude_squared() > 0.0, "look_at requires eye != target");
        let s = f.cross(&up).normalized();
        assert!(s.magnitude_squared() > 0.0, "look_at requires up not parallel to view");
        let u = s.cross(&f);
        Self::from_rows(
            [s.x, s.y, s.z, -s.dot(&eye)],
            [u.x, u.y, u.z, -u.dot(&eye)],
            [-f.x, -f.y, -f.z, f.dot(&eye)],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// OpenGL-style perspective projection mapping the view frustum to
    /// the clip cube [−1, 1]³ (near plane → z = −1, far → z = +1).
    ///
    /// # Panics
    /// Panics unless 0 < near < far, 0 < fov_y < π, and aspect > 0.
    #[must_use]
    pub fn perspective(fov_y_rad: f64, aspect: f64, near: f64, far: f64) -> Self {
        assert!(near > 0.0 && far > near, "perspective requires 0 < near < far");
        assert!(fov_y_rad > 0.0 && fov_y_rad < crate::math::constants::PI);
        assert!(aspect > 0.0, "perspective requires aspect > 0");
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        Self::from_rows(
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        )
    }

    /// OpenGL-style orthographic projection onto the clip cube.
    ///
    /// # Panics
    /// Panics on a zero-extent box.
    #[must_use]
    pub fn orthographic(l: f64, r: f64, b: f64, t: f64, near: f64, far: f64) -> Self {
        assert!(r != l && t != b && far != near, "orthographic requires non-zero extents");
        Self::from_rows(
            [2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l)],
            [0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b)],
            [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Matrix product self·other.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut out = [[0.0; 4]; 4];
        for (i, out_row) in out.iter_mut().enumerate() {
            for j in 0..4 {
                out_row[j] = (0..4).map(|k| self.data[i][k] * other.data[k][j]).sum();
            }
        }
        Self { data: out }
    }

    /// Transforms a point (w = 1) with perspective division; a point
    /// mapped to w' = 0 is returned undivided.
    #[must_use]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let h = self.transform_homogeneous([p.x, p.y, p.z, 1.0]);
        if h[3].abs() > EPS {
            Vec3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
        } else {
            Vec3::new(h[0], h[1], h[2])
        }
    }

    /// Transforms a direction (w = 0): translation has no effect.
    #[must_use]
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        let h = self.transform_homogeneous([v.x, v.y, v.z, 0.0]);
        Vec3::new(h[0], h[1], h[2])
    }

    /// Full homogeneous transform.
    #[must_use]
    pub fn transform_homogeneous(&self, p: [f64; 4]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = (0..4).map(|k| self.data[i][k] * p[k]).sum();
        }
        out
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut out = [[0.0; 4]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.data[j][i];
            }
        }
        Self { data: out }
    }

    /// Determinant by cofactor expansion over 2×2 sub-determinants.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let m = &self.data;
        let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
        let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
        let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
        let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
        let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
        let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];
        let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
        s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    }

    /// General inverse via the 2×2-subdeterminant (Laplace) expansion;
    /// `None` when the determinant is negligible.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.data;
        let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
        let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
        let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
        let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
        let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
        let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];
        let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.abs() < EPS {
            return None;
        }
        let inv = 1.0 / det;
        let mut o = [[0.0; 4]; 4];
        o[0][0] = (m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * inv;
        o[0][1] = (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * inv;
        o[0][2] = (m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * inv;
        o[0][3] = (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * inv;
        o[1][0] = (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * inv;
        o[1][1] = (m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * inv;
        o[1][2] = (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * inv;
        o[1][3] = (m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * inv;
        o[2][0] = (m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * inv;
        o[2][1] = (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * inv;
        o[2][2] = (m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * inv;
        o[2][3] = (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * inv;
        o[3][0] = (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * inv;
        o[3][1] = (m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * inv;
        o[3][2] = (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * inv;
        o[3][3] = (m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * inv;
        Some(Self { data: o })
    }

    /// Fast inverse for affine matrices (last row 0 0 0 1):
    /// M⁻¹ = [A⁻¹, −A⁻¹·t]. `None` if the matrix is not affine or A is
    /// singular.
    #[must_use]
    pub fn inverse_affine(&self) -> Option<Self> {
        let m = &self.data;
        if (m[3][0]).abs() > EPS
            || (m[3][1]).abs() > EPS
            || (m[3][2]).abs() > EPS
            || (m[3][3] - 1.0).abs() > EPS
        {
            return None;
        }
        let a_inv = self.to_mat3().inverse()?;
        let t = Vec3::new(m[0][3], m[1][3], m[2][3]);
        let ti = a_inv.mul_vec(t) * -1.0;
        let d = &a_inv.data;
        Some(Self::from_rows(
            [d[0][0], d[0][1], d[0][2], ti.x],
            [d[1][0], d[1][1], d[1][2], ti.y],
            [d[2][0], d[2][1], d[2][2], ti.z],
            [0.0, 0.0, 0.0, 1.0],
        ))
    }

    /// Upper-left 3×3 block.
    #[must_use]
    pub fn to_mat3(&self) -> Mat3 {
        let m = &self.data;
        Mat3::from_rows(
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]],
        )
    }

    /// Recovers (translation, rotation, scale) from an affine T·R·S
    /// matrix without shear. `None` for non-affine input or a
    /// degenerate (zero) scale. A negative determinant is folded into
    /// the x scale.
    #[must_use]
    pub fn decompose_trs(&self) -> Option<(Vec3, Quaternion, Vec3)> {
        let m = &self.data;
        if (m[3][0]).abs() > EPS
            || (m[3][1]).abs() > EPS
            || (m[3][2]).abs() > EPS
            || (m[3][3] - 1.0).abs() > EPS
        {
            return None;
        }
        let t = Vec3::new(m[0][3], m[1][3], m[2][3]);
        let cols = [
            Vec3::new(m[0][0], m[1][0], m[2][0]),
            Vec3::new(m[0][1], m[1][1], m[2][1]),
            Vec3::new(m[0][2], m[1][2], m[2][2]),
        ];
        let mut s = Vec3::new(cols[0].magnitude(), cols[1].magnitude(), cols[2].magnitude());
        if s.x < EPS || s.y < EPS || s.z < EPS {
            return None;
        }
        if self.to_mat3().determinant() < 0.0 {
            s.x = -s.x;
        }
        let r = [
            [cols[0].x / s.x, cols[1].x / s.y, cols[2].x / s.z],
            [cols[0].y / s.x, cols[1].y / s.y, cols[2].y / s.z],
            [cols[0].z / s.x, cols[1].z / s.y, cols[2].z / s.z],
        ];
        Some((t, Quaternion::from_rotation_matrix(&r), s))
    }

    /// Normal matrix: inverse-transpose of the upper-left 3×3 (falls
    /// back to the block itself when singular).
    #[must_use]
    pub fn normal_matrix(&self) -> Mat3 {
        let a = self.to_mat3();
        match a.inverse() {
            Some(inv) => inv.transpose(),
            None => a,
        }
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, rhs: Mat4) -> Mat4 {
        Mat4::mul(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn mats_close(a: &Mat4, b: &Mat4, tol: f64) -> bool {
        a.data
            .iter()
            .flatten()
            .zip(b.data.iter().flatten())
            .all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_translation_and_vector() {
        let m = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        let p = m.transform_point(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(p, Vec3::new(2.0, 3.0, 4.0));
        // Directions ignore translation.
        let v = m.transform_vector(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(v, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_inverse_general_and_affine() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 2.0, 0.5), 0.7);
        let m = Mat4::from_trs(Vec3::new(2.0, -1.0, 4.0), &q, Vec3::new(2.0, 3.0, 0.5));
        let inv = m.inverse().unwrap();
        assert!(mats_close(&(m * inv), &Mat4::identity(), 1e-12));
        let inv_a = m.inverse_affine().unwrap();
        assert!(mats_close(&inv, &inv_a, 1e-10));
        // Perspective is not affine.
        assert!(Mat4::perspective(1.0, 1.5, 0.1, 100.0).inverse_affine().is_none());
    }

    #[test]
    fn test_determinant_scaling() {
        let m = Mat4::scaling(Vec3::new(2.0, 3.0, 4.0));
        assert!(approx(m.determinant(), 24.0, 1e-12));
        assert!(approx(Mat4::identity().determinant(), 1.0, 1e-15));
    }

    #[test]
    fn test_trs_decompose_roundtrip() {
        let t = Vec3::new(1.0, -2.0, 0.5);
        let q = Quaternion::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 1.1);
        let s = Vec3::new(2.0, 0.5, 1.5);
        let m = Mat4::from_trs(t, &q, s);
        let (t2, q2, s2) = m.decompose_trs().unwrap();
        assert!(t2.distance_to(&t) < 1e-12);
        assert!(s2.distance_to(&s) < 1e-12);
        let m2 = Mat4::from_trs(t2, &q2, s2);
        assert!(mats_close(&m, &m2, 1e-10));
    }

    #[test]
    fn test_perspective_depth_range() {
        let m = Mat4::perspective(PI / 3.0, 16.0 / 9.0, 0.5, 50.0);
        let near = m.transform_point(Vec3::new(0.0, 0.0, -0.5));
        let far = m.transform_point(Vec3::new(0.0, 0.0, -50.0));
        assert!(approx(near.z, -1.0, 1e-12), "near z = {}", near.z);
        assert!(approx(far.z, 1.0, 1e-12), "far z = {}", far.z);
    }

    #[test]
    fn test_orthographic_maps_corners() {
        let m = Mat4::orthographic(-2.0, 2.0, -1.0, 1.0, 0.1, 10.0);
        let p = m.transform_point(Vec3::new(2.0, 1.0, -10.0));
        assert!(approx(p.x, 1.0, 1e-12) && approx(p.y, 1.0, 1e-12) && approx(p.z, 1.0, 1e-12));
    }

    #[test]
    fn test_look_at_puts_target_on_negative_z() {
        let eye = Vec3::new(3.0, 2.0, 5.0);
        let target = Vec3::new(-1.0, 0.5, 0.0);
        let m = Mat4::look_at(eye, target, Vec3::new(0.0, 1.0, 0.0));
        let p = m.transform_point(target);
        assert!(p.x.abs() < 1e-12 && p.y.abs() < 1e-12);
        assert!(p.z < 0.0);
        // Eye maps to origin.
        let o = m.transform_point(eye);
        assert!(o.magnitude() < 1e-12);
    }

    #[test]
    fn test_normal_matrix_perpendicularity() {
        // Normals transformed by the normal matrix stay perpendicular
        // to transformed tangents under non-uniform scale.
        let m = Mat4::scaling(Vec3::new(2.0, 1.0, 1.0));
        let tangent = Vec3::new(1.0, 1.0, 0.0).normalized();
        let normal = Vec3::new(-1.0, 1.0, 0.0).normalized();
        let t2 = m.transform_vector(tangent);
        let n2 = m.normal_matrix().mul_vec(normal);
        assert!(t2.dot(&n2).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_matches_quaternion() {
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let m = Mat4::rotation(&q);
        let p = m.transform_point(Vec3::new(1.0, 0.0, 0.0));
        assert!(p.distance_to(&Vec3::new(0.0, 1.0, 0.0)) < 1e-12);
    }
}
