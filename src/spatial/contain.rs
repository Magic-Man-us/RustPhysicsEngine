//! Orientation predicates and containment tests.
//!
//! `orient2d_exact` follows Shewchuk's approach: a floating-point
//! filter with a proven error bound, falling back to exact expansion
//! arithmetic (error-free two_sum / two_product transforms) when the
//! filter cannot decide (Shewchuk, "Adaptive precision floating-point
//! arithmetic and fast robust geometric predicates", 1997).

use crate::math::{Vec2, Vec3};
use crate::spatial::primitives::{
    Aabb, Capsule, Cylinder, Obb, Polygon2, Sphere, Triangle, Triangle2,
};

/// Twice the signed area of (a, b, c): positive for a CCW turn.
#[must_use]
pub fn orient2d(a: Vec2, b: Vec2, c: Vec2) -> f64 {
    (b - a).cross(&(c - a))
}

/// Six times the signed volume of tetrahedron (a, b, c, d): positive
/// when d lies on the positive side of the CCW plane (a, b, c).
#[must_use]
pub fn orient3d(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    (b - a).cross(&(c - a)).dot(&(d - a))
}

/// In-circle predicate: > 0 iff d lies inside the circumcircle of the
/// CCW triangle (a, b, c) (4×4 determinant form).
#[must_use]
pub fn in_circle(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> f64 {
    let (adx, ady) = (a.x - d.x, a.y - d.y);
    let (bdx, bdy) = (b.x - d.x, b.y - d.y);
    let (cdx, cdy) = (c.x - d.x, c.y - d.y);
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;
    alift * (bdx * cdy - cdx * bdy) + blift * (cdx * ady - adx * cdy)
        + clift * (adx * bdy - bdx * ady)
}

/// In-sphere predicate: > 0 iff e lies inside the circumsphere of the
/// positively oriented tetrahedron (a, b, c, d).
#[must_use]
pub fn in_sphere(a: Vec3, b: Vec3, c: Vec3, d: Vec3, e: Vec3) -> f64 {
    let rows = [a - e, b - e, c - e, d - e];
    let lifts: Vec<f64> = rows.iter().map(|r| r.magnitude_squared()).collect();
    // 4x4 determinant | x y z lift | expanded along the lift column,
    // negated so that "inside" is positive for positively oriented
    // tetrahedra (the 3-D parity is opposite to the 2-D incircle case).
    let det3 = |p: Vec3, q: Vec3, r: Vec3| p.dot(&q.cross(&r));
    -(-lifts[0] * det3(rows[1], rows[2], rows[3]) + lifts[1] * det3(rows[0], rows[2], rows[3])
        - lifts[2] * det3(rows[0], rows[1], rows[3])
        + lifts[3] * det3(rows[0], rows[1], rows[2]))
}

// ── Exact orientation via expansion arithmetic ──────────────────────

/// Error-free sum: a + b = x + y exactly, |y| ≤ ulp(x)/2.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    let av = x - bv;
    let y = (a - av) + (b - bv);
    (x, y)
}

/// Error-free product via FMA: a·b = x + y exactly.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let y = a.mul_add(b, -x);
    (x, y)
}

/// Adds a scalar to an expansion (nonoverlapping, increasing order).
fn grow_expansion(e: &[f64], b: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(e.len() + 1);
    let mut q = b;
    for &ei in e {
        let (sum, err) = two_sum(q, ei);
        if err != 0.0 {
            out.push(err);
        }
        q = sum;
    }
    out.push(q);
    out
}

fn expansion_sum(e: &[f64], f: &[f64]) -> Vec<f64> {
    let mut acc = e.to_vec();
    for &fi in f {
        acc = grow_expansion(&acc, fi);
    }
    acc
}

fn expansion_sign(e: &[f64]) -> i8 {
    // The largest-magnitude component is last; its sign is the sign of
    // the sum for nonoverlapping expansions.
    for &v in e.iter().rev() {
        if v != 0.0 {
            return if v > 0.0 { 1 } else { -1 };
        }
    }
    0
}

/// Exact sign of orient2d: −1, 0, or 1, never wrong.
///
/// A Shewchuk-style floating-point filter answers the easy cases; the
/// hard ones are decided by exact expansion evaluation of the 6-term
/// determinant ax·by − ax·cy + bx·cy − bx·ay + cx·ay − cx·by.
#[must_use]
pub fn orient2d_exact(a: Vec2, b: Vec2, c: Vec2) -> i8 {
    let detleft = (a.x - c.x) * (b.y - c.y);
    let detright = (a.y - c.y) * (b.x - c.x);
    let det = detleft - detright;
    // Filter (Shewchuk's ccwerrboundA ≈ 3.33e-16 scaled sum).
    let detsum = detleft.abs() + detright.abs();
    let errbound = 3.330_669_073_875_471_6e-16 * detsum;
    if det > errbound {
        return 1;
    }
    if det < -errbound {
        return -1;
    }
    // Exact: expand all six products of the unrounded determinant.
    let mut e: Vec<f64> = vec![0.0];
    for (p, q, sign) in [
        (a.x, b.y, 1.0),
        (a.x, c.y, -1.0),
        (b.x, c.y, 1.0),
        (b.x, a.y, -1.0),
        (c.x, a.y, 1.0),
        (c.x, b.y, -1.0),
    ] {
        let (hi, lo) = two_product(p * sign, q);
        e = expansion_sum(&e, &[lo, hi]);
    }
    expansion_sign(&e)
}

// ── Containment tests ───────────────────────────────────────────────

/// Point in 2-D triangle (boundary counts as inside), robust to either
/// winding.
#[must_use]
pub fn point_in_triangle_2d(p: Vec2, t: &Triangle2) -> bool {
    let d1 = orient2d(t.a, t.b, p);
    let d2 = orient2d(t.b, t.c, p);
    let d3 = orient2d(t.c, t.a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Point in 3-D triangle: within `tol` of the plane and barycentrics
/// in [0, 1].
#[must_use]
pub fn point_in_triangle(p: Vec3, t: &Triangle, tol: f64) -> bool {
    let n = t.normal();
    if n.magnitude_squared() == 0.0 {
        return false;
    }
    if n.dot(&(p - t.a)).abs() > tol {
        return false;
    }
    let (u, v, w) = t.barycentric(p);
    let lo = -1e-12;
    let hi = 1.0 + 1e-12;
    u >= lo && u <= hi && v >= lo && v <= hi && w >= lo && w <= hi
}

/// Even-odd (crossing number) point-in-polygon test.
#[must_use]
pub fn point_in_polygon_2d(p: Vec2, poly: &Polygon2) -> bool {
    let n = poly.vertices.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let vi = poly.vertices[i];
        let vj = poly.vertices[j];
        if (vi.y > p.y) != (vj.y > p.y) {
            let x_int = vi.x + (p.y - vi.y) / (vj.y - vi.y) * (vj.x - vi.x);
            if p.x < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Winding number of a polygon about p (0 for outside points of simple
/// polygons; ±1 inside depending on orientation).
#[must_use]
pub fn winding_number_2d(p: Vec2, poly: &Polygon2) -> i32 {
    let n = poly.vertices.len();
    let mut wn = 0i32;
    for i in 0..n {
        let vi = poly.vertices[i];
        let vj = poly.vertices[(i + 1) % n];
        if vi.y <= p.y {
            if vj.y > p.y && orient2d(vi, vj, p) > 0.0 {
                wn += 1;
            }
        } else if vj.y <= p.y && orient2d(vi, vj, p) < 0.0 {
            wn -= 1;
        }
    }
    wn
}

/// O(log n) point-in-convex-polygon by binary search on the fan from
/// vertex 0 (polygon must be convex and CCW).
#[must_use]
pub fn point_in_convex_polygon_2d(p: Vec2, poly: &Polygon2) -> bool {
    let v = &poly.vertices;
    let n = v.len();
    if n < 3 {
        return false;
    }
    // Outside the wedge at v0?
    if orient2d(v[0], v[1], p) < 0.0 || orient2d(v[0], v[n - 1], p) > 0.0 {
        return false;
    }
    // Binary search for the fan triangle containing the direction v0→p.
    let mut lo = 1;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if orient2d(v[0], v[mid], p) >= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    orient2d(v[lo], v[hi], p) >= 0.0
}

/// Point inside a convex hull given as outward-oriented triangles: on
/// or behind every face plane.
#[must_use]
pub fn point_in_convex_hull_3d(p: Vec3, hull_tris: &[Triangle]) -> bool {
    hull_tris.iter().all(|t| orient3d(t.a, t.b, t.c, p) <= 1e-9)
}

/// Generalized winding number test for closed (possibly non-convex)
/// triangle meshes: the summed signed solid angle is ±4π inside and ~0
/// outside (van Oosterom & Strackee 1983 per-triangle solid angle).
#[must_use]
pub fn point_in_mesh(p: Vec3, tris: &[Triangle]) -> bool {
    let mut total = 0.0;
    for t in tris {
        let a = t.a - p;
        let b = t.b - p;
        let c = t.c - p;
        let la = a.magnitude();
        let lb = b.magnitude();
        let lc = c.magnitude();
        let numer = a.dot(&b.cross(&c));
        let denom =
            la * lb * lc + a.dot(&b) * lc + b.dot(&c) * la + c.dot(&a) * lb;
        total += 2.0 * numer.atan2(denom);
    }
    total.abs() > 2.0 * crate::math::constants::PI
}

/// Point in AABB (closed).
#[must_use]
pub fn point_in_aabb(p: Vec3, b: &Aabb) -> bool {
    b.contains_point(p)
}

/// Point in OBB: local coordinates within the half extents.
#[must_use]
pub fn point_in_obb(p: Vec3, b: &Obb) -> bool {
    let d = p - b.center;
    let axes = b.axes();
    let h = [b.half_extents.x, b.half_extents.y, b.half_extents.z];
    (0..3).all(|i| d.dot(&axes[i]).abs() <= h[i] + 1e-12)
}

/// Point in sphere (closed).
#[must_use]
pub fn point_in_sphere(p: Vec3, s: &Sphere) -> bool {
    p.distance_to(&s.center) <= s.radius
}

/// Point in capsule: within radius of the core segment.
#[must_use]
pub fn point_in_capsule(p: Vec3, c: &Capsule) -> bool {
    let seg = crate::spatial::primitives::Segment { a: c.a, b: c.b };
    crate::spatial::distance::distance_point_segment(p, &seg) <= c.radius
}

/// Point in finite cylinder: axial span and radial distance.
#[must_use]
pub fn point_in_cylinder(p: Vec3, c: &Cylinder) -> bool {
    let axis = c.b - c.a;
    let len2 = axis.magnitude_squared();
    if len2 == 0.0 {
        return false;
    }
    let t = (p - c.a).dot(&axis) / len2;
    if !(0.0..=1.0).contains(&t) {
        return false;
    }
    let foot = c.a + axis * t;
    p.distance_to(&foot) <= c.radius
}

/// Point in tetrahedron: consistent orientation with respect to all
/// four faces.
#[must_use]
pub fn point_in_tetrahedron(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    let s0 = orient3d(a, b, c, d);
    if s0.abs() < 1e-30 {
        return false; // degenerate
    }
    let signs = [
        orient3d(a, b, c, p),
        orient3d(a, d, b, p),
        orient3d(b, d, c, p),
        orient3d(c, d, a, p),
    ];
    let reference = s0.signum();
    signs.iter().all(|&s| s.signum() == reference || s.abs() < 1e-12)
}

/// Full containment of one AABB in another (closed).
#[must_use]
pub fn aabb_contains_aabb(outer: &Aabb, inner: &Aabb) -> bool {
    outer.contains_point(inner.min) && outer.contains_point(inner.max)
}

/// Sphere containing an entire AABB (all corners inside).
#[must_use]
pub fn sphere_contains_aabb(s: &Sphere, b: &Aabb) -> bool {
    b.corners().iter().all(|&c| point_in_sphere(c, s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Mat3;

    #[test]
    fn test_orient_predicates() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);
        assert!(orient2d(a, b, c) > 0.0);
        assert!((orient2d(a, b, c) + orient2d(b, a, c)).abs() < 1e-15);
        assert_eq!(orient2d(a, b, Vec2::new(2.0, 0.0)), 0.0);

        let o3 = orient3d(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(o3 > 0.0);
    }

    #[test]
    fn test_in_circle_and_in_sphere() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let c = Vec2::new(0.0, 2.0);
        // Circumcircle center (1,1), radius sqrt(2).
        assert!(in_circle(a, b, c, Vec2::new(1.0, 1.0)) > 0.0);
        assert!(in_circle(a, b, c, Vec2::new(5.0, 5.0)) < 0.0);

        let t = (
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        );
        assert!(orient3d(t.0, t.1, t.2, t.3) > 0.0);
        assert!(in_sphere(t.0, t.1, t.2, t.3, Vec3::new(0.5, 0.5, 0.5)) > 0.0);
        assert!(in_sphere(t.0, t.1, t.2, t.3, Vec3::new(9.0, 9.0, 9.0)) < 0.0);
    }

    #[test]
    fn test_orient2d_exact_hard_cases() {
        // Clearly oriented: agrees with the float sign.
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        assert_eq!(orient2d_exact(a, b, Vec2::new(0.5, 1.0)), 1);
        assert_eq!(orient2d_exact(a, b, Vec2::new(0.5, -1.0)), -1);
        // Exactly collinear even with awkward magnitudes.
        assert_eq!(orient2d_exact(a, b, Vec2::new(1e17, 0.0)), 0);
        // Nearly collinear where the naive filter is inconclusive:
        // p one ulp above the line y = x (ulp at 3e16 is 4).
        let big = 1.0e16;
        let p = Vec2::new(big, big);
        let q = Vec2::new(2.0 * big, 2.0 * big);
        let r_above = Vec2::new(3.0 * big, (3.0 * big) + 4.0);
        let r_on = Vec2::new(3.0 * big, 3.0 * big);
        assert_eq!(orient2d_exact(p, q, r_on), 0);
        let s = orient2d_exact(p, q, r_above);
        assert_eq!(s, 1, "exact sign must detect the one-ulp offset");
    }

    #[test]
    fn test_point_in_triangles() {
        let t2 = Triangle2 {
            a: Vec2::new(0.0, 0.0),
            b: Vec2::new(2.0, 0.0),
            c: Vec2::new(0.0, 2.0),
        };
        assert!(point_in_triangle_2d(Vec2::new(0.5, 0.5), &t2));
        assert!(point_in_triangle_2d(Vec2::new(1.0, 0.0), &t2)); // edge
        assert!(!point_in_triangle_2d(Vec2::new(2.0, 2.0), &t2));
        // Clockwise winding also works.
        let t2cw = Triangle2 { a: t2.a, b: t2.c, c: t2.b };
        assert!(point_in_triangle_2d(Vec2::new(0.5, 0.5), &t2cw));

        let t3 = Triangle {
            a: Vec3::ZERO,
            b: Vec3::new(2.0, 0.0, 0.0),
            c: Vec3::new(0.0, 2.0, 0.0),
        };
        assert!(point_in_triangle(Vec3::new(0.5, 0.5, 0.0), &t3, 1e-9));
        assert!(!point_in_triangle(Vec3::new(0.5, 0.5, 0.5), &t3, 1e-9));
        assert!(!point_in_triangle(Vec3::new(3.0, 3.0, 0.0), &t3, 1e-9));
    }

    #[test]
    fn test_polygon_containment_agreement() {
        let poly = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 3.0),
            Vec2::new(0.0, 3.0),
        ]);
        for (p, expect) in [
            (Vec2::new(2.0, 1.5), true),
            (Vec2::new(-1.0, 1.0), false),
            (Vec2::new(5.0, 1.0), false),
            (Vec2::new(2.0, 4.0), false),
        ] {
            assert_eq!(point_in_polygon_2d(p, &poly), expect, "{p:?}");
            assert_eq!(winding_number_2d(p, &poly) != 0, expect, "wn {p:?}");
            assert_eq!(point_in_convex_polygon_2d(p, &poly), expect, "convex {p:?}");
        }
    }

    #[test]
    fn test_point_in_volumes() {
        let b = Aabb { min: Vec3::ZERO, max: Vec3::new(1.0, 1.0, 1.0) };
        assert!(point_in_aabb(Vec3::new(0.5, 0.5, 0.5), &b));
        assert!(!point_in_aabb(Vec3::new(1.5, 0.5, 0.5), &b));

        let obb = Obb {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 2.0, 3.0),
            rotation: Mat3::identity(),
        };
        assert!(point_in_obb(Vec3::new(0.9, -1.9, 2.9), &obb));
        assert!(!point_in_obb(Vec3::new(1.1, 0.0, 0.0), &obb));

        let s = Sphere { center: Vec3::ZERO, radius: 1.0 };
        assert!(point_in_sphere(Vec3::new(0.0, 1.0, 0.0), &s));
        assert!(!point_in_sphere(Vec3::new(0.0, 1.01, 0.0), &s));

        let cap = Capsule { a: Vec3::ZERO, b: Vec3::new(0.0, 2.0, 0.0), radius: 0.5 };
        assert!(point_in_capsule(Vec3::new(0.3, 1.0, 0.0), &cap));
        assert!(point_in_capsule(Vec3::new(0.0, -0.4, 0.0), &cap)); // cap end
        assert!(!point_in_capsule(Vec3::new(0.6, 1.0, 0.0), &cap));

        let cyl = Cylinder { a: Vec3::ZERO, b: Vec3::new(0.0, 2.0, 0.0), radius: 0.5 };
        assert!(point_in_cylinder(Vec3::new(0.3, 1.0, 0.0), &cyl));
        assert!(!point_in_cylinder(Vec3::new(0.0, -0.1, 0.0), &cyl)); // beyond cap

        assert!(point_in_tetrahedron(
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0)
        ));
        assert!(!point_in_tetrahedron(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0)
        ));
    }

    #[test]
    fn test_hull_and_mesh_containment() {
        // Unit tetrahedron as outward-oriented triangles.
        let (a, b, c, d) = (
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let tris = vec![
            Triangle { a, b: c, c: b },
            Triangle { a, b, c: d },
            Triangle { a: b, b: c, c: d },
            Triangle { a, b: d, c },
        ];
        let inside = Vec3::new(0.2, 0.2, 0.2);
        let outside = Vec3::new(1.0, 1.0, 1.0);
        assert!(point_in_convex_hull_3d(inside, &tris));
        assert!(!point_in_convex_hull_3d(outside, &tris));
        assert!(point_in_mesh(inside, &tris));
        assert!(!point_in_mesh(outside, &tris));
    }

    #[test]
    fn test_aabb_and_sphere_containment() {
        let outer = Aabb { min: Vec3::ZERO, max: Vec3::new(4.0, 4.0, 4.0) };
        let inner = Aabb { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(2.0, 2.0, 2.0) };
        assert!(aabb_contains_aabb(&outer, &inner));
        assert!(!aabb_contains_aabb(&inner, &outer));
        let s = Sphere { center: Vec3::new(1.5, 1.5, 1.5), radius: 1.0 };
        assert!(sphere_contains_aabb(&s, &inner));
        assert!(!sphere_contains_aabb(&s, &outer));
    }
}
