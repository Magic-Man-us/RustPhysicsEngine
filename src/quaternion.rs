use std::ops::{Add, Mul, Neg, Sub};

use crate::math::constants::PI;
use crate::math::Vec3;

const SLERP_DOT_THRESHOLD: f64 = 0.9995;
const UNIT_QUATERNION_DEFAULT_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        let a = axis.normalized();
        Self {
            w: half.cos(),
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
        }
    }

    /// ZYX convention: roll (X), pitch (Y), yaw (Z) applied as Z * Y * X.
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sr, cr) = (roll * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();

        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            return Self::identity();
        }
        let inv = 1.0 / n;
        Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn inverse(&self) -> Self {
        let norm_sq = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        let inv = 1.0 / norm_sq;
        Self {
            w: self.w * inv,
            x: -self.x * inv,
            y: -self.y * inv,
            z: -self.z * inv,
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn rotate_vec(&self, v: Vec3) -> Vec3 {
        let qv = Quaternion::new(0.0, v.x, v.y, v.z);
        let rotated = *self * qv * self.conjugate();
        Vec3::new(rotated.x, rotated.y, rotated.z)
    }

    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);

        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ]
    }

    pub fn to_axis_angle(&self) -> (Vec3, f64) {
        let n = self.normalize();
        let angle = 2.0 * n.w.clamp(-1.0, 1.0).acos();
        let s = (1.0 - n.w * n.w).sqrt();
        if s < 1e-10 {
            return (Vec3::new(1.0, 0.0, 0.0), angle);
        }
        let axis = Vec3::new(n.x / s, n.y / s, n.z / s);
        (axis, angle)
    }

    /// Returns (roll, pitch, yaw) using ZYX convention.
    pub fn to_euler(&self) -> (f64, f64, f64) {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);

        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            (PI / 2.0).copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    pub fn angle_between(&self, other: &Self) -> f64 {
        let d = self.dot(other).abs().clamp(0.0, 1.0);
        2.0 * d.acos()
    }

    pub fn is_unit(&self, tolerance: f64) -> bool {
        (self.norm() - 1.0).abs() <= tolerance
    }
}

// Hamilton product
impl Mul for Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

impl Mul<f64> for Quaternion {
    type Output = Quaternion;
    fn mul(self, rhs: f64) -> Quaternion {
        Quaternion {
            w: self.w * rhs,
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Add for Quaternion {
    type Output = Quaternion;
    fn add(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Quaternion {
    type Output = Quaternion;
    fn sub(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Quaternion {
        Quaternion {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

pub fn slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Quaternion {
    let mut dot = q1.dot(q2);

    // If dot is negative, negate one quaternion to take the shorter arc.
    let q2_adj = if dot < 0.0 {
        dot = -dot;
        -*q2
    } else {
        *q2
    };

    if dot > SLERP_DOT_THRESHOLD {
        return nlerp(q1, &q2_adj, t);
    }

    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let a = ((1.0 - t) * theta).sin() / sin_theta;
    let b = (t * theta).sin() / sin_theta;

    *q1 * a + q2_adj * b
}

pub fn nlerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Quaternion {
    let mut q2_adj = *q2;
    if q1.dot(q2) < 0.0 {
        q2_adj = -*q2;
    }
    (*q1 * (1.0 - t) + q2_adj * t).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn vec3_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn quat_approx(a: Quaternion, b: Quaternion) -> bool {
        approx(a.w, b.w) && approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    #[test]
    fn identity_rotation_preserves_vector() {
        let q = Quaternion::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx(q.rotate_vec(v), v));
    }

    #[test]
    fn rotate_90_about_z() {
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = q.rotate_vec(v);
        assert!(vec3_approx(result, Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn rotate_180_about_z() {
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = q.rotate_vec(v);
        assert!(vec3_approx(result, Vec3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn q_times_inverse_is_identity() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 1.23);
        let product = q * q.inverse();
        assert!(quat_approx(product, Quaternion::identity()));
    }

    #[test]
    fn slerp_endpoints() {
        let q1 = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.0);
        let q2 = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI / 2.0);

        let at_0 = slerp(&q1, &q2, 0.0);
        let at_1 = slerp(&q1, &q2, 1.0);

        assert!(quat_approx(at_0, q1));
        assert!(quat_approx(at_1, q2));
    }

    #[test]
    fn rotation_matrix_matches_rotate_vec() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), PI / 3.0);
        let v = Vec3::new(0.0, 1.0, 0.0);

        let via_quat = q.rotate_vec(v);

        let m = q.to_rotation_matrix();
        let via_matrix = Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        );

        assert!(vec3_approx(via_quat, via_matrix));
    }

    #[test]
    fn euler_roundtrip() {
        let (roll, pitch, yaw) = (0.3, 0.5, 0.7);
        let q = Quaternion::from_euler(roll, pitch, yaw);
        let (r2, p2, y2) = q.to_euler();
        assert!(approx(roll, r2));
        assert!(approx(pitch, p2));
        assert!(approx(yaw, y2));
    }

    #[test]
    fn is_unit_after_from_axis_angle() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 1.0);
        assert!(q.is_unit(UNIT_QUATERNION_DEFAULT_TOLERANCE));
    }

    #[test]
    fn nlerp_endpoints() {
        let q1 = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.5);
        let q2 = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.0);

        let at_0 = nlerp(&q1, &q2, 0.0);
        let at_1 = nlerp(&q1, &q2, 1.0);

        assert!(quat_approx(at_0, q1));
        assert!(quat_approx(at_1, q2));
    }

    #[test]
    fn conjugate_of_unit_is_inverse() {
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.8);
        let conj = q.conjugate();
        let inv = q.inverse();
        assert!(quat_approx(conj, inv));
    }

    #[test]
    fn angle_between_same_is_zero() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert!(approx(q.angle_between(&q), 0.0));
    }
}
