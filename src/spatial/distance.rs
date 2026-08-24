//! Closest-point queries and set distances.
//!
//! References: Ericson, *Real-Time Collision Detection*, ch. 5 (point
//! and segment queries); Eiter & Mannila 1994 (discrete Fréchet
//! distance).

use crate::math::{Vec2, Vec3};
use crate::spatial::primitives::{
    Aabb, Obb, Plane, Polygon2, Polyline, Segment, Segment2, Sphere, Triangle,
};

/// Closest point on a segment and its parameter t ∈ [0, 1].
#[must_use]
pub fn closest_point_segment(p: Vec3, s: &Segment) -> (Vec3, f64) {
    let ab = s.b - s.a;
    let len2 = ab.magnitude_squared();
    if len2 == 0.0 {
        return (s.a, 0.0);
    }
    let t = ((p - s.a).dot(&ab) / len2).clamp(0.0, 1.0);
    (s.a + ab * t, t)
}

/// 2-D closest point on a segment and its parameter.
#[must_use]
pub fn closest_point_segment_2d(p: Vec2, s: &Segment2) -> (Vec2, f64) {
    let ab = s.b - s.a;
    let len2 = ab.magnitude_squared();
    if len2 == 0.0 {
        return (s.a, 0.0);
    }
    let t = ((p - s.a).dot(&ab) / len2).clamp(0.0, 1.0);
    (s.a + ab * t, t)
}

/// Closest point on a triangle (Ericson RTCD §5.1.5 Voronoi-region
/// walk).
#[must_use]
pub fn closest_point_triangle(p: Vec3, t: &Triangle) -> Vec3 {
    let (a, b, c) = (t.a, t.b, t.c);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Orthogonal projection onto a plane.
#[must_use]
pub fn closest_point_plane(p: Vec3, pl: &Plane) -> Vec3 {
    pl.project(p)
}

/// Componentwise clamp onto an AABB.
#[must_use]
pub fn closest_point_aabb(p: Vec3, b: &Aabb) -> Vec3 {
    Vec3::new(
        p.x.clamp(b.min.x, b.max.x),
        p.y.clamp(b.min.y, b.max.y),
        p.z.clamp(b.min.z, b.max.z),
    )
}

/// Clamp in the box's local frame (RTCD §5.1.4).
#[must_use]
pub fn closest_point_obb(p: Vec3, b: &Obb) -> Vec3 {
    let d = p - b.center;
    let axes = b.axes();
    let h = [b.half_extents.x, b.half_extents.y, b.half_extents.z];
    let mut q = b.center;
    for i in 0..3 {
        let dist = d.dot(&axes[i]).clamp(-h[i], h[i]);
        q = q + axes[i] * dist;
    }
    q
}

/// Closest point on a sphere's surface (center maps to +radius·x̂).
#[must_use]
pub fn closest_point_sphere(p: Vec3, s: &Sphere) -> Vec3 {
    let d = p - s.center;
    let m = d.magnitude();
    if m == 0.0 {
        return s.center + Vec3::new(s.radius, 0.0, 0.0);
    }
    s.center + d * (s.radius / m)
}

/// Closest points between two segments and their distance
/// (RTCD §5.1.9).
#[must_use]
pub fn closest_points_segments(s1: &Segment, s2: &Segment) -> (Vec3, Vec3, f64) {
    let d1 = s1.b - s1.a;
    let d2 = s2.b - s2.a;
    let r = s1.a - s2.a;
    let a = d1.magnitude_squared();
    let e = d2.magnitude_squared();
    let f = d2.dot(&r);
    let (s, t);
    if a <= f64::EPSILON && e <= f64::EPSILON {
        s = 0.0;
        t = 0.0;
    } else if a <= f64::EPSILON {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(&r);
        if e <= f64::EPSILON {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(&d2);
            let denom = a * e - b * b;
            let mut s_val = if denom != 0.0 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut t_val = (b * s_val + f) / e;
            if t_val < 0.0 {
                t_val = 0.0;
                s_val = (-c / a).clamp(0.0, 1.0);
            } else if t_val > 1.0 {
                t_val = 1.0;
                s_val = ((b - c) / a).clamp(0.0, 1.0);
            }
            s = s_val;
            t = t_val;
        }
    }
    let p1 = s1.a + d1 * s;
    let p2 = s2.a + d2 * t;
    (p1, p2, p1.distance_to(&p2))
}

/// Closest points between two infinite lines; `None` when parallel.
#[must_use]
pub fn closest_points_lines(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> Option<(Vec3, Vec3)> {
    let a = d1.magnitude_squared();
    let b = d1.dot(&d2);
    let e = d2.magnitude_squared();
    let denom = a * e - b * b;
    if denom.abs() < 1e-12 * a.max(e).max(1.0) {
        return None;
    }
    let r = p1 - p2;
    let c = d1.dot(&r);
    let f = d2.dot(&r);
    let s = (b * f - c * e) / denom;
    let t = (a * f - b * c) / denom;
    Some((p1 + d1 * s, p2 + d2 * t))
}

/// Distance from a point to a segment.
#[must_use]
pub fn distance_point_segment(p: Vec3, s: &Segment) -> f64 {
    closest_point_segment(p, s).0.distance_to(&p)
}

/// Distance from a point to a triangle.
#[must_use]
pub fn distance_point_triangle(p: Vec3, t: &Triangle) -> f64 {
    closest_point_triangle(p, t).distance_to(&p)
}

/// Distance from a point to a polyline, with the segment index and
/// parameter of the closest point.
///
/// # Panics
/// Panics on a polyline with no segments.
#[must_use]
pub fn distance_point_polyline(p: Vec3, pl: &Polyline) -> (f64, usize, f64) {
    let n = pl.segment_count();
    assert!(n > 0, "distance_point_polyline requires segments");
    let mut best = (f64::INFINITY, 0usize, 0.0);
    for i in 0..n {
        let seg = pl.segment(i);
        let (q, t) = closest_point_segment(p, &seg);
        let d = q.distance_to(&p);
        if d < best.0 {
            best = (d, i, t);
        }
    }
    best
}

/// Unsigned distance from a point to the boundary of a polygon.
#[must_use]
pub fn distance_point_polygon_2d(p: Vec2, poly: &Polygon2) -> f64 {
    let n = poly.vertices.len();
    let mut best = f64::INFINITY;
    for i in 0..n {
        let s = Segment2 { a: poly.vertices[i], b: poly.vertices[(i + 1) % n] };
        let (q, _) = closest_point_segment_2d(p, &s);
        best = best.min(q.distance_to(&p));
    }
    best
}

/// Distance between a segment and a triangle: minimum over segment vs
/// the three edges and the endpoints vs the face.
#[must_use]
pub fn distance_segment_triangle(s: &Segment, t: &Triangle) -> f64 {
    // Intersection through the face means distance zero.
    let dir = s.b - s.a;
    let len = dir.magnitude();
    if len > 0.0 {
        let ray = crate::spatial::primitives::Ray { origin: s.a, dir: dir * (1.0 / len) };
        if let Some((hit, _)) = crate::spatial::intersect::ray_triangle(&ray, t, false) {
            if hit.t <= len {
                return 0.0;
            }
        }
    }
    let edges = [
        Segment { a: t.a, b: t.b },
        Segment { a: t.b, b: t.c },
        Segment { a: t.c, b: t.a },
    ];
    let mut best = f64::INFINITY;
    for e in &edges {
        best = best.min(closest_points_segments(s, e).2);
    }
    best = best.min(distance_point_triangle(s.a, t));
    best = best.min(distance_point_triangle(s.b, t));
    best
}

/// Distance between two AABBs (0 when overlapping).
#[must_use]
pub fn distance_aabb_aabb(a: &Aabb, b: &Aabb) -> f64 {
    let dx = (a.min.x - b.max.x).max(b.min.x - a.max.x).max(0.0);
    let dy = (a.min.y - b.max.y).max(b.min.y - a.max.y).max(0.0);
    let dz = (a.min.z - b.max.z).max(b.min.z - a.max.z).max(0.0);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn directed_hausdorff<P, F>(a: &[P], b: &[P], dist: F) -> f64
where
    P: Copy,
    F: Fn(P, P) -> f64,
{
    let mut worst = 0.0_f64;
    for &pa in a {
        let mut best = f64::INFINITY;
        for &pb in b {
            best = best.min(dist(pa, pb));
        }
        worst = worst.max(best);
    }
    worst
}

/// Symmetric Hausdorff distance between two 3-D point sets.
///
/// # Panics
/// Panics when either set is empty.
#[must_use]
pub fn hausdorff_distance(a: &[Vec3], b: &[Vec3]) -> f64 {
    assert!(!a.is_empty() && !b.is_empty(), "hausdorff_distance requires non-empty sets");
    let d = |p: Vec3, q: Vec3| p.distance_to(&q);
    directed_hausdorff(a, b, d).max(directed_hausdorff(b, a, d))
}

/// Symmetric Hausdorff distance between two 2-D point sets.
///
/// # Panics
/// Panics when either set is empty.
#[must_use]
pub fn hausdorff_distance_2d(a: &[Vec2], b: &[Vec2]) -> f64 {
    assert!(!a.is_empty() && !b.is_empty(), "hausdorff_distance_2d requires non-empty sets");
    let d = |p: Vec2, q: Vec2| p.distance_to(&q);
    directed_hausdorff(a, b, d).max(directed_hausdorff(b, a, d))
}

/// Discrete Fréchet distance between two 2-D polygonal curves
/// (Eiter & Mannila dynamic program).
///
/// # Panics
/// Panics when either curve is empty.
#[must_use]
pub fn frechet_distance_2d(a: &[Vec2], b: &[Vec2]) -> f64 {
    assert!(!a.is_empty() && !b.is_empty(), "frechet_distance_2d requires non-empty curves");
    let n = a.len();
    let m = b.len();
    let mut ca = vec![vec![-1.0_f64; m]; n];
    ca[0][0] = a[0].distance_to(&b[0]);
    for i in 0..n {
        for j in 0..m {
            if i == 0 && j == 0 {
                continue;
            }
            let d = a[i].distance_to(&b[j]);
            let prev = if i > 0 && j > 0 {
                ca[i - 1][j].min(ca[i][j - 1]).min(ca[i - 1][j - 1])
            } else if i > 0 {
                ca[i - 1][j]
            } else {
                ca[i][j - 1]
            };
            ca[i][j] = prev.max(d);
        }
    }
    ca[n - 1][m - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closest_point_segment_regions() {
        let s = Segment { a: Vec3::ZERO, b: Vec3::new(2.0, 0.0, 0.0) };
        // Interior.
        let (q, t) = closest_point_segment(Vec3::new(1.0, 3.0, 0.0), &s);
        assert!(q.distance_to(&Vec3::new(1.0, 0.0, 0.0)) < 1e-12 && (t - 0.5).abs() < 1e-12);
        // Clamped ends.
        assert_eq!(closest_point_segment(Vec3::new(-5.0, 1.0, 0.0), &s).1, 0.0);
        assert_eq!(closest_point_segment(Vec3::new(9.0, 1.0, 0.0), &s).1, 1.0);
        // Degenerate segment.
        let pt = Segment { a: Vec3::new(1.0, 1.0, 1.0), b: Vec3::new(1.0, 1.0, 1.0) };
        assert_eq!(closest_point_segment(Vec3::ZERO, &pt).0, pt.a);
    }

    #[test]
    fn test_closest_point_triangle_all_regions() {
        let t = Triangle {
            a: Vec3::ZERO,
            b: Vec3::new(2.0, 0.0, 0.0),
            c: Vec3::new(0.0, 2.0, 0.0),
        };
        // Interior projects straight down.
        let q = closest_point_triangle(Vec3::new(0.5, 0.5, 3.0), &t);
        assert!(q.distance_to(&Vec3::new(0.5, 0.5, 0.0)) < 1e-12);
        // Vertex regions.
        assert!(closest_point_triangle(Vec3::new(-1.0, -1.0, 0.0), &t).distance_to(&t.a) < 1e-12);
        assert!(closest_point_triangle(Vec3::new(4.0, -1.0, 0.0), &t).distance_to(&t.b) < 1e-12);
        assert!(closest_point_triangle(Vec3::new(-1.0, 4.0, 0.0), &t).distance_to(&t.c) < 1e-12);
        // Edge region (hypotenuse).
        let q2 = closest_point_triangle(Vec3::new(2.0, 2.0, 0.0), &t);
        assert!(q2.distance_to(&Vec3::new(1.0, 1.0, 0.0)) < 1e-12);
    }

    #[test]
    fn test_closest_point_plane() {
        let pl = Plane::from_point_normal(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 2.0, -2.0));
        for p in [
            Vec3::new(5.0, -1.0, 2.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(-3.0, 7.0, 1.5),
        ] {
            let q = closest_point_plane(p, &pl);
            // The returned point satisfies the plane equation n·q + d = 0.
            assert!(pl.signed_distance(q).abs() < 1e-12);
            // The displacement is parallel to the normal.
            assert!((p - q).cross(&pl.normal).magnitude() < 1e-12);
            // Its length is exactly |signed distance| (orthogonality:
            // no point of the plane is closer).
            assert!((p.distance_to(&q) - pl.signed_distance(p).abs()).abs() < 1e-12);
            // A point already on the plane is its own projection.
            assert!(closest_point_plane(q, &pl).distance_to(&q) < 1e-12);
        }
    }

    #[test]
    fn test_closest_point_boxes_and_sphere() {
        let b = Aabb { min: Vec3::ZERO, max: Vec3::new(1.0, 1.0, 1.0) };
        assert!(closest_point_aabb(Vec3::new(2.0, 0.5, -1.0), &b)
            .distance_to(&Vec3::new(1.0, 0.5, 0.0))
            < 1e-12);
        let s = Sphere { center: Vec3::ZERO, radius: 2.0 };
        let q = closest_point_sphere(Vec3::new(5.0, 0.0, 0.0), &s);
        assert!(q.distance_to(&Vec3::new(2.0, 0.0, 0.0)) < 1e-12);
        let obb = Obb {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 2.0, 3.0),
            rotation: crate::linalg::Mat3::identity(),
        };
        let qo = closest_point_obb(Vec3::new(5.0, 0.0, 0.0), &obb);
        assert!(qo.distance_to(&Vec3::new(1.0, 0.0, 0.0)) < 1e-12);
    }

    #[test]
    fn test_segments_and_lines() {
        let s1 = Segment { a: Vec3::ZERO, b: Vec3::new(1.0, 0.0, 0.0) };
        let s2 = Segment { a: Vec3::new(0.0, 1.0, 1.0), b: Vec3::new(1.0, 1.0, 1.0) };
        let (p1, p2, d) = closest_points_segments(&s1, &s2);
        assert!((d - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((p1.y - 0.0).abs() < 1e-12 && (p2.y - 1.0).abs() < 1e-12);

        let (l1, l2) = closest_points_lines(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 2.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();
        assert!(l1.distance_to(&Vec3::ZERO) < 1e-12);
        assert!(l2.distance_to(&Vec3::new(0.0, 1.0, 0.0)) < 1e-12);
        assert!(closest_points_lines(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0)
        )
        .is_none());
    }

    #[test]
    fn test_polyline_and_polygon_distance() {
        let pl = Polyline {
            points: vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
            closed: false,
        };
        let (d, seg, t) = distance_point_polyline(Vec3::new(1.0, 0.5, 2.0), &pl);
        assert!((d - 2.0).abs() < 1e-12);
        assert_eq!(seg, 1);
        assert!((t - 0.5).abs() < 1e-12);

        let poly = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]);
        // Interior point: distance to the nearest edge.
        assert!((distance_point_polygon_2d(Vec2::new(0.5, 1.0), &poly) - 0.5).abs() < 1e-12);
        assert!((distance_point_polygon_2d(Vec2::new(3.0, 1.0), &poly) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_segment_triangle_and_aabb_distances() {
        let t = Triangle {
            a: Vec3::ZERO,
            b: Vec3::new(2.0, 0.0, 0.0),
            c: Vec3::new(0.0, 2.0, 0.0),
        };
        // Piercing segment: zero.
        let pierce = Segment { a: Vec3::new(0.5, 0.5, -1.0), b: Vec3::new(0.5, 0.5, 1.0) };
        assert_eq!(distance_segment_triangle(&pierce, &t), 0.0);
        // Parallel above.
        let above = Segment { a: Vec3::new(0.2, 0.2, 1.0), b: Vec3::new(0.8, 0.2, 1.0) };
        assert!((distance_segment_triangle(&above, &t) - 1.0).abs() < 1e-12);

        let a = Aabb { min: Vec3::ZERO, max: Vec3::new(1.0, 1.0, 1.0) };
        let b = Aabb { min: Vec3::new(3.0, 0.0, 0.0), max: Vec3::new(4.0, 1.0, 1.0) };
        assert!((distance_aabb_aabb(&a, &b) - 2.0).abs() < 1e-12);
        assert_eq!(distance_aabb_aabb(&a, &a), 0.0);
    }

    #[test]
    fn test_hausdorff_and_frechet() {
        let a = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)];
        let b = vec![Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(2.0, 1.0)];
        assert!((hausdorff_distance_2d(&a, &b) - 1.0).abs() < 1e-12);
        assert!((frechet_distance_2d(&a, &b) - 1.0).abs() < 1e-12);
        // Frechet respects ordering: reversed curve is farther.
        let b_rev: Vec<Vec2> = b.iter().rev().copied().collect();
        assert!(frechet_distance_2d(&a, &b_rev) > 1.5);

        let a3 = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let b3 = vec![Vec3::new(0.0, 0.0, 3.0)];
        assert!((hausdorff_distance(&a3, &b3) - 10.0_f64.sqrt()).abs() < 1e-12);
    }
}
