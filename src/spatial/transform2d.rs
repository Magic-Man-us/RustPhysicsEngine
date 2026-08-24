//! 2-D affine transforms stored as 3×3 homogeneous matrices (last row
//! 0 0 1), column-vector convention: p' = M·p.

use crate::math::Vec2;

const EPS: f64 = 1e-12;

/// Affine map of the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2 {
    pub m: [[f64; 3]; 3],
}

impl Affine2 {
    /// Identity map.
    #[must_use]
    pub fn identity() -> Self {
        Self { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }
    }

    /// Translation by t.
    #[must_use]
    pub fn translation(t: Vec2) -> Self {
        Self { m: [[1.0, 0.0, t.x], [0.0, 1.0, t.y], [0.0, 0.0, 1.0]] }
    }

    /// Counter-clockwise rotation about the origin.
    #[must_use]
    pub fn rotation(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self { m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]] }
    }

    /// Rotation about an arbitrary center: T(c)·R·T(−c).
    #[must_use]
    pub fn rotation_about(angle: f64, center: Vec2) -> Self {
        Self::translation(center)
            .compose(&Self::rotation(angle))
            .compose(&Self::translation(-center))
    }

    /// Axis-aligned scaling.
    #[must_use]
    pub fn scaling(sx: f64, sy: f64) -> Self {
        Self { m: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]] }
    }

    /// Shear: x' = x + kx·y, y' = y + ky·x.
    #[must_use]
    pub fn shear(kx: f64, ky: f64) -> Self {
        Self { m: [[1.0, kx, 0.0], [ky, 1.0, 0.0], [0.0, 0.0, 1.0]] }
    }

    /// Reflection across the line through the origin with the given
    /// direction.
    ///
    /// # Panics
    /// Panics on a zero direction vector.
    #[must_use]
    pub fn reflection(line_through_origin_dir: Vec2) -> Self {
        let d = line_through_origin_dir;
        assert!(d.magnitude_squared() > 0.0, "reflection requires a non-zero direction");
        let n = d.normalized();
        // Householder in 2D: R = 2 n nᵀ − I over the line direction.
        let (x, y) = (n.x, n.y);
        Self {
            m: [
                [2.0 * x * x - 1.0, 2.0 * x * y, 0.0],
                [2.0 * x * y, 2.0 * y * y - 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// The unique affine map taking three source points to three
    /// destination points: M = Q·P⁻¹ with homogeneous point columns.
    /// `None` when the source triple is collinear.
    #[must_use]
    pub fn from_three_points(src: [Vec2; 3], dst: [Vec2; 3]) -> Option<Self> {
        // P columns are homogeneous source points.
        let det = src[0].x * (src[1].y - src[2].y) - src[1].x * (src[0].y - src[2].y)
            + src[2].x * (src[0].y - src[1].y);
        if det.abs() < EPS {
            return None;
        }
        // Solve M · [sx, sy, 1]^T = [dx, dy] row-wise: two 3-var systems
        // sharing the same coefficient matrix; use Cramer's rule.
        let solve_row = |d0: f64, d1: f64, d2: f64| -> [f64; 3] {
            // coefficient matrix rows: [src[i].x, src[i].y, 1]
            let a = [
                [src[0].x, src[0].y, 1.0],
                [src[1].x, src[1].y, 1.0],
                [src[2].x, src[2].y, 1.0],
            ];
            let rhs = [d0, d1, d2];
            let det3 = |c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]| {
                c0[0] * (c1[1] * c2[2] - c1[2] * c2[1])
                    - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
                    + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
            };
            let col = |j: usize| [a[0][j], a[1][j], a[2][j]];
            let d_full = det3(col(0), col(1), col(2));
            [
                det3(rhs, col(1), col(2)) / d_full,
                det3(col(0), rhs, col(2)) / d_full,
                det3(col(0), col(1), rhs) / d_full,
            ]
        };
        let r0 = solve_row(dst[0].x, dst[1].x, dst[2].x);
        let r1 = solve_row(dst[0].y, dst[1].y, dst[2].y);
        Some(Self { m: [r0, r1, [0.0, 0.0, 1.0]] })
    }

    /// Composition self ∘ other (apply `other` first).
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| self.m[i][k] * other.m[k][j]).sum();
            }
        }
        Self { m: out }
    }

    /// Applies to a point (uses the translation column).
    #[must_use]
    pub fn apply(&self, p: Vec2) -> Vec2 {
        Vec2::new(
            self.m[0][0] * p.x + self.m[0][1] * p.y + self.m[0][2],
            self.m[1][0] * p.x + self.m[1][1] * p.y + self.m[1][2],
        )
    }

    /// Applies to a direction (ignores translation).
    #[must_use]
    pub fn apply_vector(&self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y,
            self.m[1][0] * v.x + self.m[1][1] * v.y,
        )
    }

    /// Inverse map; `None` when the linear part is singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let [a, b] = [self.m[0][0], self.m[0][1]];
        let [c, d] = [self.m[1][0], self.m[1][1]];
        let det = a * d - b * c;
        if det.abs() < EPS {
            return None;
        }
        let (tx, ty) = (self.m[0][2], self.m[1][2]);
        let inv = 1.0 / det;
        let (ia, ib, ic, id) = (d * inv, -b * inv, -c * inv, a * inv);
        Some(Self {
            m: [
                [ia, ib, -(ia * tx + ib * ty)],
                [ic, id, -(ic * tx + id * ty)],
                [0.0, 0.0, 1.0],
            ],
        })
    }

    /// Decomposition A = R(rot)·[[sx, shear·sy], [0, sy]] plus the
    /// translation: returns (t, rot, (sx, sy), shear). Recompose with
    /// `translation(t)·rotation(rot)·[[sx, shear·sy],[0, sy]]`.
    #[must_use]
    pub fn decompose(&self) -> (Vec2, f64, Vec2, f64) {
        let t = Vec2::new(self.m[0][2], self.m[1][2]);
        let col0 = Vec2::new(self.m[0][0], self.m[1][0]);
        let col1 = Vec2::new(self.m[0][1], self.m[1][1]);
        let sx = col0.magnitude();
        let rot = col0.y.atan2(col0.x);
        let r0 = col0.normalized();
        let h = r0.dot(&col1); // shear·sy in the rotated frame
        let perp = col1 - r0 * h;
        let mut sy = perp.magnitude();
        // Sign of sy from the determinant (reflection folds into sy).
        if col0.cross(&col1) < 0.0 {
            sy = -sy;
        }
        let shear = if sy.abs() > EPS { h / sy } else { 0.0 };
        (t, rot, Vec2::new(sx, sy), shear)
    }

    /// True for a rigid motion (rotation + translation, no reflection):
    /// AᵀA = I and det A = +1 within tol.
    #[must_use]
    pub fn is_rigid(&self, tol: f64) -> bool {
        let (a, b) = (self.m[0][0], self.m[0][1]);
        let (c, d) = (self.m[1][0], self.m[1][1]);
        let det = a * d - b * c;
        ((a * a + c * c) - 1.0).abs() <= tol
            && ((b * b + d * d) - 1.0).abs() <= tol
            && (a * b + c * d).abs() <= tol
            && (det - 1.0).abs() <= tol
    }

    /// True for a similarity (uniform scale + rotation ± reflection):
    /// AᵀA = s²·I within tol.
    #[must_use]
    pub fn is_similarity(&self, tol: f64) -> bool {
        let (a, b) = (self.m[0][0], self.m[0][1]);
        let (c, d) = (self.m[1][0], self.m[1][1]);
        let n0 = a * a + c * c;
        let n1 = b * b + d * d;
        (n0 - n1).abs() <= tol * n0.max(n1).max(1.0) && (a * b + c * d).abs() <= tol * n0.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    fn close(a: Vec2, b: Vec2, tol: f64) -> bool {
        a.distance_to(&b) < tol
    }

    #[test]
    fn test_basic_maps() {
        let t = Affine2::translation(Vec2::new(1.0, 2.0));
        assert!(close(t.apply(Vec2::ZERO), Vec2::new(1.0, 2.0), 1e-15));
        assert!(close(t.apply_vector(Vec2::new(3.0, 4.0)), Vec2::new(3.0, 4.0), 1e-15));

        let r = Affine2::rotation(PI / 2.0);
        assert!(close(r.apply(Vec2::new(1.0, 0.0)), Vec2::new(0.0, 1.0), 1e-12));

        let s = Affine2::scaling(2.0, 3.0);
        assert!(close(s.apply(Vec2::new(1.0, 1.0)), Vec2::new(2.0, 3.0), 1e-15));

        let sh = Affine2::shear(0.5, 0.0);
        assert!(close(sh.apply(Vec2::new(0.0, 2.0)), Vec2::new(1.0, 2.0), 1e-15));
    }

    #[test]
    fn test_rotation_about_center_fixes_center() {
        let c = Vec2::new(3.0, -1.0);
        let r = Affine2::rotation_about(1.2, c);
        assert!(close(r.apply(c), c, 1e-12));
        // Distance to center preserved.
        let p = Vec2::new(5.0, 2.0);
        assert!((r.apply(p).distance_to(&c) - p.distance_to(&c)).abs() < 1e-12);
    }

    #[test]
    fn test_reflection_is_involution() {
        let m = Affine2::reflection(Vec2::new(1.0, 2.0));
        let p = Vec2::new(-3.0, 0.7);
        assert!(close(m.apply(m.apply(p)), p, 1e-12));
        // Points on the line are fixed.
        let on_line = Vec2::new(2.0, 4.0);
        assert!(close(m.apply(on_line), on_line, 1e-12));
    }

    #[test]
    fn test_from_three_points_exact() {
        let src = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
        let dst = [Vec2::new(2.0, 1.0), Vec2::new(3.0, 2.0), Vec2::new(1.0, 3.0)];
        let m = Affine2::from_three_points(src, dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            assert!(close(m.apply(*s), *d, 1e-10));
        }
        // Collinear source has no unique preimage.
        let bad = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0)];
        assert!(Affine2::from_three_points(bad, dst).is_none());
    }

    #[test]
    fn test_inverse_and_compose() {
        let m = Affine2::translation(Vec2::new(1.0, -2.0))
            .compose(&Affine2::rotation(0.7))
            .compose(&Affine2::scaling(2.0, 0.5));
        let inv = m.inverse().unwrap();
        let p = Vec2::new(0.3, -1.1);
        assert!(close(inv.apply(m.apply(p)), p, 1e-12));
        assert!(Affine2::scaling(0.0, 1.0).inverse().is_none());
    }

    #[test]
    fn test_decompose_recompose_roundtrip() {
        let (t, rot, s, k) = (Vec2::new(1.0, 2.0), 0.6, Vec2::new(2.0, 0.7), 0.3);
        let shear_m = Affine2 {
            m: [[s.x, k * s.y, 0.0], [0.0, s.y, 0.0], [0.0, 0.0, 1.0]],
        };
        let m = Affine2::translation(t).compose(&Affine2::rotation(rot)).compose(&shear_m);
        let (t2, rot2, s2, k2) = m.decompose();
        assert!(close(t2, t, 1e-12));
        assert!((rot2 - rot).abs() < 1e-12);
        assert!(close(s2, s, 1e-12));
        assert!((k2 - k).abs() < 1e-12);
    }

    #[test]
    fn test_rigid_and_similarity_predicates() {
        let rigid = Affine2::translation(Vec2::new(1.0, 1.0)).compose(&Affine2::rotation(0.4));
        assert!(rigid.is_rigid(1e-12));
        assert!(rigid.is_similarity(1e-12));
        let sim = rigid.compose(&Affine2::scaling(2.0, 2.0));
        assert!(!sim.is_rigid(1e-9));
        assert!(sim.is_similarity(1e-12));
        let general = Affine2::scaling(2.0, 1.0);
        assert!(!general.is_similarity(1e-9));
        let reflect = Affine2::reflection(Vec2::new(1.0, 0.0));
        assert!(!reflect.is_rigid(1e-9), "reflection is not orientation-preserving");
    }
}
