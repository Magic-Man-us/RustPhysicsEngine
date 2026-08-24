//! Rigid coordinate frames (origin + unit-quaternion rotation).
//!
//! `to_world(p_local) = origin + R·p_local`; composition and inverses
//! follow the usual rigid-motion group structure SE(3).

use crate::math::Vec3;
use crate::quaternion::{slerp, Quaternion};
use crate::spatial::mat4::Mat4;

/// A rigid frame: position and orientation of a local coordinate
/// system expressed in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub origin: Vec3,
    pub rotation: Quaternion,
}

impl Frame {
    /// World frame: origin at zero, identity rotation.
    #[must_use]
    pub fn identity() -> Self {
        Self { origin: Vec3::ZERO, rotation: Quaternion::identity() }
    }

    /// Frame from origin and rotation (normalized).
    #[must_use]
    pub fn new(origin: Vec3, rotation: Quaternion) -> Self {
        Self { origin, rotation: rotation.normalize() }
    }

    /// Frame whose x axis points along `x` and whose y axis is the
    /// component of `y` perpendicular to it (Gram-Schmidt); z = x × y.
    ///
    /// # Panics
    /// Panics when `x` is zero or `y` is parallel to `x`.
    #[must_use]
    pub fn from_axes(origin: Vec3, x: Vec3, y: Vec3) -> Self {
        let xn = x.normalized();
        assert!(xn.magnitude_squared() > 0.0, "from_axes requires non-zero x");
        let z = x.cross(&y).normalized();
        assert!(z.magnitude_squared() > 0.0, "from_axes requires y not parallel to x");
        let yn = z.cross(&xn);
        let m = [[xn.x, yn.x, z.x], [xn.y, yn.y, z.y], [xn.z, yn.z, z.z]];
        Self { origin, rotation: Quaternion::from_rotation_matrix(&m) }
    }

    /// World point → local coordinates.
    #[must_use]
    pub fn to_local(&self, world_p: Vec3) -> Vec3 {
        self.rotation.conjugate().rotate_vec(world_p - self.origin)
    }

    /// Local point → world coordinates.
    #[must_use]
    pub fn to_world(&self, local_p: Vec3) -> Vec3 {
        self.origin + self.rotation.rotate_vec(local_p)
    }

    /// World direction → local (rotation only).
    #[must_use]
    pub fn to_local_vector(&self, v: Vec3) -> Vec3 {
        self.rotation.conjugate().rotate_vec(v)
    }

    /// Local direction → world (rotation only).
    #[must_use]
    pub fn to_world_vector(&self, v: Vec3) -> Vec3 {
        self.rotation.rotate_vec(v)
    }

    /// A child frame expressed in `self` coordinates, re-expressed in
    /// world coordinates.
    #[must_use]
    pub fn compose(&self, child: &Frame) -> Frame {
        Frame {
            origin: self.to_world(child.origin),
            rotation: (self.rotation * child.rotation).normalize(),
        }
    }

    /// The inverse rigid motion.
    #[must_use]
    pub fn inverse(&self) -> Frame {
        let inv_rot = self.rotation.conjugate();
        Frame { origin: inv_rot.rotate_vec(-self.origin), rotation: inv_rot }
    }

    /// This frame expressed relative to `other`:
    /// other.compose(result) == self.
    #[must_use]
    pub fn relative_to(&self, other: &Frame) -> Frame {
        other.inverse().compose(self)
    }

    /// Screw-free interpolation: lerp the origin, slerp the rotation.
    #[must_use]
    pub fn interpolate(&self, other: &Frame, t: f64) -> Frame {
        Frame {
            origin: self.origin.lerp(&other.origin, t),
            rotation: slerp(&self.rotation, &other.rotation, t),
        }
    }

    /// Equivalent homogeneous matrix (unit scale).
    #[must_use]
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_trs(self.origin, &self.rotation, Vec3::new(1.0, 1.0, 1.0))
    }

    /// Local +X axis in world coordinates.
    #[must_use]
    pub fn x_axis(&self) -> Vec3 {
        self.rotation.rotate_vec(Vec3::new(1.0, 0.0, 0.0))
    }

    /// Local +Y axis in world coordinates.
    #[must_use]
    pub fn y_axis(&self) -> Vec3 {
        self.rotation.rotate_vec(Vec3::new(0.0, 1.0, 0.0))
    }

    /// Local +Z axis in world coordinates.
    #[must_use]
    pub fn z_axis(&self) -> Vec3 {
        self.rotation.rotate_vec(Vec3::new(0.0, 0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    fn frame() -> Frame {
        Frame::new(
            Vec3::new(1.0, -2.0, 3.0),
            Quaternion::from_axis_angle(Vec3::new(0.2, 1.0, 0.5), 0.9),
        )
    }

    fn close(a: Vec3, b: Vec3, tol: f64) -> bool {
        a.distance_to(&b) < tol
    }

    #[test]
    fn test_local_world_roundtrip() {
        let f = frame();
        let p = Vec3::new(0.4, 2.0, -1.5);
        assert!(close(f.to_world(f.to_local(p)), p, 1e-12));
        assert!(close(f.to_local(f.to_world(p)), p, 1e-12));
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(close(f.to_world_vector(f.to_local_vector(v)), v, 1e-12));
    }

    #[test]
    fn test_compose_inverse_is_identity() {
        let f = frame();
        let id = f.compose(&f.inverse());
        assert!(close(id.origin, Vec3::ZERO, 1e-12));
        let p = Vec3::new(5.0, -1.0, 2.0);
        assert!(close(id.to_world(p), p, 1e-12));
    }

    #[test]
    fn test_relative_to_recovers() {
        let a = frame();
        let b = Frame::new(
            Vec3::new(-2.0, 1.0, 0.5),
            Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3),
        );
        let rel = a.relative_to(&b);
        let back = b.compose(&rel);
        let p = Vec3::new(0.7, 0.2, -0.9);
        assert!(close(back.to_world(p), a.to_world(p), 1e-12));
    }

    #[test]
    fn test_matches_mat4() {
        let f = frame();
        let m = f.to_mat4();
        let p = Vec3::new(1.1, -0.4, 0.8);
        assert!(close(m.transform_point(p), f.to_world(p), 1e-12));
    }

    #[test]
    fn test_axes_orthonormal_right_handed() {
        let f = Frame::from_axes(
            Vec3::ZERO,
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 3.0, 1.0),
        );
        let (x, y, z) = (f.x_axis(), f.y_axis(), f.z_axis());
        assert!((x.magnitude() - 1.0).abs() < 1e-12);
        assert!(x.dot(&y).abs() < 1e-12);
        assert!(close(x.cross(&y), z, 1e-12));
        // x preserved up to normalization.
        assert!(close(x, Vec3::new(2.0, 1.0, 0.0).normalized(), 1e-12));
    }

    #[test]
    fn test_interpolate_endpoints_and_half() {
        let a = Frame::identity();
        let b = Frame::new(
            Vec3::new(2.0, 0.0, 0.0),
            Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI / 2.0),
        );
        assert!(close(a.interpolate(&b, 0.0).origin, a.origin, 1e-12));
        assert!(close(a.interpolate(&b, 1.0).origin, b.origin, 1e-12));
        let mid = a.interpolate(&b, 0.5);
        assert!(close(mid.origin, Vec3::new(1.0, 0.0, 0.0), 1e-12));
        let rotated = mid.rotation.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        let expected = Vec3::new((PI / 4.0).cos(), (PI / 4.0).sin(), 0.0);
        assert!(close(rotated, expected, 1e-12));
    }
}
