//! Intersection tests between the spatial primitives.
//!
//! References: Ericson, *Real-Time Collision Detection* (RTCD);
//! Möller & Trumbore 1997 (ray-triangle); Akenine-Möller 2001
//! (triangle-box SAT). Ray parameters are along the (normalized) ray
//! direction; only t ≥ 0 counts as a hit.

use crate::math::{Vec2, Vec3};
use crate::spatial::primitives::{
    Aabb, Capsule, Circle, Cylinder, Obb, Plane, Polygon2, Ray, Rect, Segment2, Sphere, Triangle,
};

const EPS: f64 = 1e-12;

/// A ray hit: parameter, position, and surface normal (facing the ray).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub t: f64,
    pub point: Vec3,
    pub normal: Vec3,
}

/// Ray vs sphere: nearest hit with t ≥ 0 (RTCD §5.3.2).
#[must_use]
pub fn ray_sphere(r: &Ray, s: &Sphere) -> Option<RayHit> {
    let m = r.origin - s.center;
    let b = m.dot(&r.dir);
    let c = m.magnitude_squared() - s.radius * s.radius;
    if c > 0.0 && b > 0.0 {
        return None; // outside and pointing away
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()).max(0.0);
    let point = r.at(t);
    Some(RayHit { t, point, normal: (point - s.center).normalized() })
}

/// Ray vs plane: `None` when parallel or hitting behind the origin.
#[must_use]
pub fn ray_plane(r: &Ray, p: &Plane) -> Option<RayHit> {
    let denom = p.normal.dot(&r.dir);
    if denom.abs() < EPS {
        return None;
    }
    let t = -(p.normal.dot(&r.origin) + p.d) / denom;
    if t < 0.0 {
        return None;
    }
    let normal = if denom < 0.0 { p.normal } else { -p.normal };
    Some(RayHit { t, point: r.at(t), normal })
}

/// Möller-Trumbore ray-triangle intersection; also returns the
/// barycentric coordinates (u, v, w) of the hit with respect to
/// (a, b, c). With `cull_backface`, only front faces (CCW seen from
/// the ray origin) hit.
#[must_use]
pub fn ray_triangle(
    r: &Ray,
    t: &Triangle,
    cull_backface: bool,
) -> Option<(RayHit, (f64, f64, f64))> {
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let pvec = r.dir.cross(&e2);
    let det = e1.dot(&pvec);
    if cull_backface {
        if det < EPS {
            return None;
        }
    } else if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = r.origin - t.a;
    let u = tvec.dot(&pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(&e1);
    let v = r.dir.dot(&qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let hit_t = e2.dot(&qvec) * inv_det;
    if hit_t < 0.0 {
        return None;
    }
    let geom_n = t.normal();
    let normal = if geom_n.dot(&r.dir) > 0.0 { -geom_n } else { geom_n };
    Some((
        RayHit { t: hit_t, point: r.at(hit_t), normal },
        (1.0 - u - v, u, v),
    ))
}

/// Slab-method ray vs AABB: (t_enter, t_exit) of the overlap with
/// t ≥ 0, `None` on a miss (RTCD §5.3.3).
#[must_use]
pub fn ray_aabb(r: &Ray, b: &Aabb) -> Option<(f64, f64)> {
    let mut t_enter = 0.0_f64;
    let mut t_exit = f64::INFINITY;
    let o = [r.origin.x, r.origin.y, r.origin.z];
    let d = [r.dir.x, r.dir.y, r.dir.z];
    let lo = [b.min.x, b.min.y, b.min.z];
    let hi = [b.max.x, b.max.y, b.max.z];
    for i in 0..3 {
        if d[i].abs() < EPS {
            if o[i] < lo[i] || o[i] > hi[i] {
                return None;
            }
        } else {
            let inv = 1.0 / d[i];
            let mut t1 = (lo[i] - o[i]) * inv;
            let mut t2 = (hi[i] - o[i]) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_enter = t_enter.max(t1);
            t_exit = t_exit.min(t2);
            if t_enter > t_exit {
                return None;
            }
        }
    }
    Some((t_enter, t_exit))
}

/// Ray vs OBB: the slab test in the box's local frame.
#[must_use]
pub fn ray_obb(r: &Ray, b: &Obb) -> Option<(f64, f64)> {
    // Express the ray in box coordinates (rotation columns are axes).
    let [ax, ay, az] = b.axes();
    let rel = r.origin - b.center;
    let local_o = Vec3::new(rel.dot(&ax), rel.dot(&ay), rel.dot(&az));
    let local_d = Vec3::new(r.dir.dot(&ax), r.dir.dot(&ay), r.dir.dot(&az));
    let h = b.half_extents;
    let aabb = Aabb { min: -h, max: h };
    // Reuse the slab test without renormalizing (local_d is unit).
    let local_ray = Ray { origin: local_o, dir: local_d };
    ray_aabb(&local_ray, &aabb)
}

/// Ray vs capsule: cylinder body plus spherical caps, nearest hit.
#[must_use]
pub fn ray_capsule(r: &Ray, c: &Capsule) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    let mut consider = |hit: Option<RayHit>| {
        if let Some(h) = hit {
            if best.is_none() || h.t < best.unwrap().t {
                best = Some(h);
            }
        }
    };
    // Spherical caps.
    consider(ray_sphere(r, &Sphere { center: c.a, radius: c.radius }));
    consider(ray_sphere(r, &Sphere { center: c.b, radius: c.radius }));
    // Infinite cylinder clipped to the segment span.
    let axis = c.b - c.a;
    let len2 = axis.magnitude_squared();
    if len2 > EPS {
        let d = r.dir;
        let m = r.origin - c.a;
        let nd = d - axis * (d.dot(&axis) / len2);
        let nm = m - axis * (m.dot(&axis) / len2);
        let a = nd.magnitude_squared();
        if a > EPS {
            let b_half = nd.dot(&nm);
            let cc = nm.magnitude_squared() - c.radius * c.radius;
            let disc = b_half * b_half - a * cc;
            if disc >= 0.0 {
                let t = (-b_half - disc.sqrt()) / a;
                if t >= 0.0 {
                    let p = r.at(t);
                    let s = (p - c.a).dot(&axis) / len2;
                    if (0.0..=1.0).contains(&s) {
                        let foot = c.a + axis * s;
                        consider(Some(RayHit {
                            t,
                            point: p,
                            normal: (p - foot).normalized(),
                        }));
                    }
                }
            }
        }
    }
    best
}

/// Ray vs finite cylinder: lateral surface and both cap disks.
#[must_use]
pub fn ray_cylinder(r: &Ray, c: &Cylinder) -> Option<RayHit> {
    let axis_full = c.b - c.a;
    let len2 = axis_full.magnitude_squared();
    if len2 < EPS {
        return None;
    }
    let axis = axis_full * (1.0 / len2.sqrt());
    let mut best: Option<RayHit> = None;
    let mut consider = |h: RayHit| {
        if best.is_none() || h.t < best.unwrap().t {
            best = Some(h);
        }
    };
    // Lateral surface.
    let d = r.dir;
    let m = r.origin - c.a;
    let nd = d - axis * d.dot(&axis);
    let nm = m - axis * m.dot(&axis);
    let a = nd.magnitude_squared();
    if a > EPS {
        let b_half = nd.dot(&nm);
        let cc = nm.magnitude_squared() - c.radius * c.radius;
        let disc = b_half * b_half - a * cc;
        if disc >= 0.0 {
            for sign in [-1.0, 1.0] {
                let t = (-b_half + sign * disc.sqrt()) / a;
                if t >= 0.0 {
                    let p = r.at(t);
                    let s = (p - c.a).dot(&axis);
                    if s >= 0.0 && s * s <= len2 {
                        let foot = c.a + axis * s;
                        consider(RayHit { t, point: p, normal: (p - foot).normalized() });
                    }
                }
            }
        }
    }
    // Cap disks.
    for (center, n) in [(c.a, -axis), (c.b, axis)] {
        let denom = n.dot(&r.dir);
        if denom.abs() > EPS {
            let t = (center - r.origin).dot(&n) / denom;
            if t >= 0.0 {
                let p = r.at(t);
                if (p - center).magnitude_squared() <= c.radius * c.radius {
                    let normal = if denom < 0.0 { n } else { -n };
                    consider(RayHit { t, point: p, normal });
                }
            }
        }
    }
    best
}

/// Proper 2-D segment intersection (interiors cross): the point, or
/// `None` for disjoint, touching, or collinear segments.
#[must_use]
pub fn segment_segment_2d(s1: &Segment2, s2: &Segment2) -> Option<Vec2> {
    let (t, u) = segment_segment_2d_params(s1, s2)?;
    if t <= 0.0 || t >= 1.0 || u <= 0.0 || u >= 1.0 {
        return None;
    }
    Some(s1.a.lerp(&s1.b, t))
}

/// Parameters (t, u) with s1(t) = s2(u), both in [0, 1] (endpoints
/// included); `None` for parallel/collinear or non-intersecting pairs.
#[must_use]
pub fn segment_segment_2d_params(s1: &Segment2, s2: &Segment2) -> Option<(f64, f64)> {
    let r = s1.b - s1.a;
    let s = s2.b - s2.a;
    let denom = r.cross(&s);
    if denom.abs() < EPS {
        return None;
    }
    let qp = s2.a - s1.a;
    let t = qp.cross(&s) / denom;
    let u = qp.cross(&r) / denom;
    if (-EPS..=1.0 + EPS).contains(&t) && (-EPS..=1.0 + EPS).contains(&u) {
        Some((t.clamp(0.0, 1.0), u.clamp(0.0, 1.0)))
    } else {
        None
    }
}

/// Infinite line (point + direction) vs circle: the two intersection
/// points ordered along the direction (equal at tangency).
#[must_use]
pub fn line_circle(p: Vec2, dir: Vec2, c: &Circle) -> Option<(Vec2, Vec2)> {
    let d = dir.normalized();
    if d.magnitude_squared() == 0.0 {
        return None;
    }
    let m = p - c.center;
    let b = m.dot(&d);
    let cc = m.magnitude_squared() - c.radius * c.radius;
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    Some((p + d * (-b - sq), p + d * (-b + sq)))
}

/// Circle-circle intersection points; `None` when separate, nested, or
/// coincident.
#[must_use]
pub fn circle_circle(a: &Circle, b: &Circle) -> Option<(Vec2, Vec2)> {
    let d = a.center.distance_to(&b.center);
    if d < EPS {
        return None; // concentric
    }
    if d > a.radius + b.radius || d < (a.radius - b.radius).abs() {
        return None;
    }
    let l = (a.radius * a.radius - b.radius * b.radius + d * d) / (2.0 * d);
    let h2 = a.radius * a.radius - l * l;
    let h = h2.max(0.0).sqrt();
    let dir = (b.center - a.center) * (1.0 / d);
    let mid = a.center + dir * l;
    let perp = dir.perp();
    Some((mid + perp * h, mid - perp * h))
}

/// Sphere overlap test.
#[must_use]
pub fn sphere_sphere(a: &Sphere, b: &Sphere) -> bool {
    let r = a.radius + b.radius;
    a.center.distance_to(&b.center) <= r
}

/// Sphere contact: unit normal from a toward b and penetration depth,
/// `Some` only when overlapping.
#[must_use]
pub fn sphere_sphere_contact(a: &Sphere, b: &Sphere) -> Option<(Vec3, f64)> {
    let delta = b.center - a.center;
    let d = delta.magnitude();
    let pen = a.radius + b.radius - d;
    if pen < 0.0 {
        return None;
    }
    let normal = if d > EPS { delta * (1.0 / d) } else { Vec3::new(1.0, 0.0, 0.0) };
    Some((normal, pen))
}

/// AABB overlap test (closed).
#[must_use]
pub fn aabb_aabb(a: &Aabb, b: &Aabb) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

/// Rectangle overlap test (closed).
#[must_use]
pub fn rect_rect(a: &Rect, b: &Rect) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y
}

/// Sphere vs AABB (closest-point distance).
#[must_use]
pub fn sphere_aabb(s: &Sphere, b: &Aabb) -> bool {
    let p = Vec3::new(
        s.center.x.clamp(b.min.x, b.max.x),
        s.center.y.clamp(b.min.y, b.max.y),
        s.center.z.clamp(b.min.z, b.max.z),
    );
    p.distance_to(&s.center) <= s.radius
}

/// Sphere vs triangle: contact point on the triangle and penetration
/// depth when overlapping.
#[must_use]
pub fn sphere_triangle(s: &Sphere, t: &Triangle) -> Option<(Vec3, f64)> {
    let closest = crate::spatial::distance::closest_point_triangle(s.center, t);
    let d = closest.distance_to(&s.center);
    if d <= s.radius {
        Some((closest, s.radius - d))
    } else {
        None
    }
}

/// OBB-OBB overlap by the separating axis theorem over the 15
/// candidate axes (RTCD §4.4.1).
#[must_use]
pub fn obb_obb(a: &Obb, b: &Obb) -> bool {
    let axes_a = a.axes();
    let axes_b = b.axes();
    let translation = b.center - a.center;
    let mut axes: Vec<Vec3> = Vec::with_capacity(15);
    axes.extend_from_slice(&axes_a);
    axes.extend_from_slice(&axes_b);
    for i in 0..3 {
        for j in 0..3 {
            let c = axes_a[i].cross(&axes_b[j]);
            if c.magnitude_squared() > EPS {
                axes.push(c.normalized());
            }
        }
    }
    for axis in axes {
        let ra = axes_a
            .iter()
            .zip([a.half_extents.x, a.half_extents.y, a.half_extents.z])
            .map(|(ax, h)| (ax.dot(&axis) * h).abs())
            .sum::<f64>();
        let rb = axes_b
            .iter()
            .zip([b.half_extents.x, b.half_extents.y, b.half_extents.z])
            .map(|(ax, h)| (ax.dot(&axis) * h).abs())
            .sum::<f64>();
        if translation.dot(&axis).abs() > ra + rb + EPS {
            return false;
        }
    }
    true
}

fn project_triangle(t: &Triangle, axis: Vec3) -> (f64, f64) {
    let d = [t.a.dot(&axis), t.b.dot(&axis), t.c.dot(&axis)];
    (
        d.iter().fold(f64::INFINITY, |m, &x| m.min(x)),
        d.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x)),
    )
}

/// Triangle-triangle overlap by SAT over both normals and the nine
/// edge-pair cross products, with a coplanar 2-D SAT fallback
/// (equivalent to the interval tests of Möller 1997).
#[must_use]
pub fn triangle_triangle(a: &Triangle, b: &Triangle) -> bool {
    let ea = [a.b - a.a, a.c - a.b, a.a - a.c];
    let eb = [b.b - b.a, b.c - b.b, b.a - b.c];
    let na = ea[0].cross(&(a.c - a.a));
    let nb = eb[0].cross(&(b.c - b.a));

    let mut axes: Vec<Vec3> = Vec::with_capacity(17);
    if na.magnitude_squared() > EPS {
        axes.push(na);
    }
    if nb.magnitude_squared() > EPS {
        axes.push(nb);
    }
    for i in 0..3 {
        for j in 0..3 {
            let c = ea[i].cross(&eb[j]);
            if c.magnitude_squared() > EPS {
                axes.push(c);
            }
        }
    }
    // Coplanar (or degenerate) case: cross-product axes vanish; add the
    // in-plane edge normals as 2-D SAT axes.
    if na.magnitude_squared() > EPS && nb.magnitude_squared() > EPS {
        let n = na.normalized();
        if n.cross(&nb.normalized()).magnitude() < 1e-9 {
            for e in ea.iter().chain(eb.iter()) {
                let axis = n.cross(e);
                if axis.magnitude_squared() > EPS {
                    axes.push(axis);
                }
            }
        }
    }
    for axis in axes {
        let (amin, amax) = project_triangle(a, axis);
        let (bmin, bmax) = project_triangle(b, axis);
        if amax < bmin - EPS || bmax < amin - EPS {
            return false;
        }
    }
    true
}

/// Plane-plane intersection line; `None` for (near-)parallel planes.
#[must_use]
pub fn plane_plane(a: &Plane, b: &Plane) -> Option<Ray> {
    let dir = a.normal.cross(&b.normal);
    if dir.magnitude_squared() < EPS {
        return None;
    }
    // Point on both planes in span{n1, n2}: solve the 2x2 Gram system.
    let n1n1 = a.normal.dot(&a.normal);
    let n1n2 = a.normal.dot(&b.normal);
    let n2n2 = b.normal.dot(&b.normal);
    let det = n1n1 * n2n2 - n1n2 * n1n2;
    let c1 = (-a.d * n2n2 + b.d * n1n2) / det;
    let c2 = (-b.d * n1n1 + a.d * n1n2) / det;
    let point = a.normal * c1 + b.normal * c2;
    Some(Ray { origin: point, dir: dir.normalized() })
}

/// Common point of three planes; `None` when any pair is parallel or
/// the normals are linearly dependent.
#[must_use]
pub fn three_planes(a: &Plane, b: &Plane, c: &Plane) -> Option<Vec3> {
    let m = crate::linalg::Mat3::from_rows(
        [a.normal.x, a.normal.y, a.normal.z],
        [b.normal.x, b.normal.y, b.normal.z],
        [c.normal.x, c.normal.y, c.normal.z],
    );
    let inv = m.inverse()?;
    Some(inv.mul_vec(Vec3::new(-a.d, -b.d, -c.d)))
}

/// Triangle vs AABB by the 13-axis SAT of Akenine-Möller.
#[must_use]
pub fn triangle_aabb(t: &Triangle, b: &Aabb) -> bool {
    let c = b.center();
    let h = b.extents();
    // Translate so the box is centered at the origin.
    let v = [t.a - c, t.b - c, t.c - c];
    let e = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];
    let box_axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let mut axes: Vec<Vec3> = Vec::with_capacity(13);
    axes.extend_from_slice(&box_axes);
    let n = e[0].cross(&e[1]);
    if n.magnitude_squared() > EPS {
        axes.push(n);
    }
    for be in &box_axes {
        for te in &e {
            let cr = be.cross(te);
            if cr.magnitude_squared() > EPS {
                axes.push(cr);
            }
        }
    }
    for axis in axes {
        let p = [v[0].dot(&axis), v[1].dot(&axis), v[2].dot(&axis)];
        let tmin = p.iter().fold(f64::INFINITY, |m, &x| m.min(x));
        let tmax = p.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x));
        let r = h.x * axis.x.abs() + h.y * axis.y.abs() + h.z * axis.z.abs();
        if tmin > r + EPS || tmax < -r - EPS {
            return false;
        }
    }
    true
}

/// Polygon overlap: SAT when both are convex, otherwise any edge
/// crossing or mutual containment.
#[must_use]
pub fn polygon_polygon_2d(a: &Polygon2, b: &Polygon2) -> bool {
    if a.is_convex() && b.is_convex() {
        // SAT over the edge normals of both polygons.
        for poly in [a, b] {
            let n = poly.vertices.len();
            for i in 0..n {
                let edge = poly.vertices[(i + 1) % n] - poly.vertices[i];
                let axis = edge.perp();
                let project = |p: &Polygon2| {
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for &v in &p.vertices {
                        let d = v.dot(&axis);
                        lo = lo.min(d);
                        hi = hi.max(d);
                    }
                    (lo, hi)
                };
                let (a_lo, a_hi) = project(a);
                let (b_lo, b_hi) = project(b);
                if a_hi < b_lo - EPS || b_hi < a_lo - EPS {
                    return false;
                }
            }
        }
        return true;
    }
    // General case: edge-edge crossing or one containing the other.
    let na = a.vertices.len();
    let nb = b.vertices.len();
    for i in 0..na {
        let s1 = Segment2 { a: a.vertices[i], b: a.vertices[(i + 1) % na] };
        for j in 0..nb {
            let s2 = Segment2 { a: b.vertices[j], b: b.vertices[(j + 1) % nb] };
            if segment_segment_2d_params(&s1, &s2).is_some() {
                return true;
            }
        }
    }
    crate::spatial::contain::point_in_polygon_2d(a.vertices[0], b)
        || crate::spatial::contain::point_in_polygon_2d(b.vertices[0], a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::rotation_z;
    use crate::linalg::Mat3;

    fn xhat() -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
    }

    #[test]
    fn test_ray_sphere_hit_distance() {
        let r = Ray::new(Vec3::new(-5.0, 0.0, 0.0), xhat());
        let s = Sphere { center: Vec3::ZERO, radius: 2.0 };
        let hit = ray_sphere(&r, &s).unwrap();
        assert!((hit.t - 3.0).abs() < 1e-12);
        assert!((hit.point.distance_to(&s.center) - 2.0).abs() < 1e-12);
        assert!(hit.normal.distance_to(&Vec3::new(-1.0, 0.0, 0.0)) < 1e-12);
        // Inside: t clamps to 0.
        let inside = Ray::new(Vec3::ZERO, xhat());
        assert_eq!(ray_sphere(&inside, &s).unwrap().t, 0.0);
        // Pointing away: miss.
        let away = Ray::new(Vec3::new(5.0, 0.0, 0.0), xhat());
        assert!(ray_sphere(&away, &s).is_none());
    }

    #[test]
    fn test_ray_plane_and_triangle() {
        let r = Ray::new(Vec3::new(0.2, 0.2, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let p = Plane::from_point_normal(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        let hit = ray_plane(&r, &p).unwrap();
        assert!((hit.t - 5.0).abs() < 1e-12);
        assert!(hit.normal.z > 0.0);

        let t = Triangle {
            a: Vec3::ZERO,
            b: Vec3::new(1.0, 0.0, 0.0),
            c: Vec3::new(0.0, 1.0, 0.0),
        };
        let (h, (u, v, w)) = ray_triangle(&r, &t, false).unwrap();
        assert!((h.t - 5.0).abs() < 1e-12);
        assert!((u + v + w - 1.0).abs() < 1e-12);
        assert!(u >= 0.0 && v >= 0.0 && w >= 0.0);
        // Outside the triangle: miss.
        let r2 = Ray::new(Vec3::new(0.9, 0.9, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(ray_triangle(&r2, &t, false).is_none());
        // Backface culling: the same ray from below misses a CCW-up tri.
        let r3 = Ray::new(Vec3::new(0.2, 0.2, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(ray_triangle(&r3, &t, true).is_none());
        assert!(ray_triangle(&r3, &t, false).is_some());
    }

    #[test]
    fn test_ray_aabb_and_obb_agree_for_identity() {
        let b = Aabb { min: Vec3::new(-1.0, -1.0, -1.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let obb = Obb {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 1.0, 1.0),
            rotation: Mat3::identity(),
        };
        let r = Ray::new(Vec3::new(-3.0, 0.2, 0.3), xhat());
        let (e1, x1) = ray_aabb(&r, &b).unwrap();
        let (e2, x2) = ray_obb(&r, &obb).unwrap();
        assert!((e1 - e2).abs() < 1e-12 && (x1 - x2).abs() < 1e-12);
        assert!((e1 - 2.0).abs() < 1e-12 && (x1 - 4.0).abs() < 1e-12);
        // Miss case.
        let miss = Ray::new(Vec3::new(-3.0, 5.0, 0.0), xhat());
        assert!(ray_aabb(&miss, &b).is_none());
        assert!(ray_obb(&miss, &obb).is_none());
    }

    #[test]
    fn test_ray_capsule_and_cylinder() {
        let cap = Capsule { a: Vec3::new(0.0, -1.0, 0.0), b: Vec3::new(0.0, 1.0, 0.0), radius: 0.5 };
        let r = Ray::new(Vec3::new(-5.0, 0.0, 0.0), xhat());
        let hit = ray_capsule(&r, &cap).unwrap();
        assert!((hit.t - 4.5).abs() < 1e-12);
        // Above the segment: hits the cap sphere.
        let r_top = Ray::new(Vec3::new(-5.0, 1.3, 0.0), xhat());
        let hit_top = ray_capsule(&r_top, &cap).unwrap();
        assert!(hit_top.point.y > 1.0);

        let cyl = Cylinder { a: Vec3::new(0.0, -1.0, 0.0), b: Vec3::new(0.0, 1.0, 0.0), radius: 0.5 };
        let hit_c = ray_cylinder(&r, &cyl).unwrap();
        assert!((hit_c.t - 4.5).abs() < 1e-12);
        // Straight down the axis: hits the top cap.
        let r_axis = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit_cap = ray_cylinder(&r_axis, &cyl).unwrap();
        assert!((hit_cap.t - 4.0).abs() < 1e-12);
        assert!((hit_cap.point.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_segment_segment() {
        let s1 = Segment2 { a: Vec2::new(0.0, 0.0), b: Vec2::new(2.0, 2.0) };
        let s2 = Segment2 { a: Vec2::new(0.0, 2.0), b: Vec2::new(2.0, 0.0) };
        let p = segment_segment_2d(&s1, &s2).unwrap();
        assert!(p.distance_to(&Vec2::new(1.0, 1.0)) < 1e-12);
        // Symmetric.
        let q = segment_segment_2d(&s2, &s1).unwrap();
        assert!(p.distance_to(&q) < 1e-12);
        // Touching at an endpoint is not a proper intersection.
        let s3 = Segment2 { a: Vec2::new(2.0, 2.0), b: Vec2::new(3.0, 0.0) };
        assert!(segment_segment_2d(&s1, &s3).is_none());
        assert!(segment_segment_2d_params(&s1, &s3).is_some());
        // Disjoint.
        let s4 = Segment2 { a: Vec2::new(5.0, 5.0), b: Vec2::new(6.0, 5.0) };
        assert!(segment_segment_2d_params(&s1, &s4).is_none());
    }

    #[test]
    fn test_line_circle_and_circle_circle() {
        let c = Circle { center: Vec2::new(1.0, 0.0), radius: 1.0 };
        let (p1, p2) = line_circle(Vec2::new(-5.0, 0.0), Vec2::new(1.0, 0.0), &c).unwrap();
        assert!(p1.distance_to(&Vec2::new(0.0, 0.0)) < 1e-12);
        assert!(p2.distance_to(&Vec2::new(2.0, 0.0)) < 1e-12);
        assert!(line_circle(Vec2::new(-5.0, 3.0), Vec2::new(1.0, 0.0), &c).is_none());

        let a = Circle { center: Vec2::new(0.0, 0.0), radius: 1.0 };
        let b = Circle { center: Vec2::new(1.0, 0.0), radius: 1.0 };
        let (q1, q2) = circle_circle(&a, &b).unwrap();
        for q in [q1, q2] {
            assert!((q.distance_to(&a.center) - 1.0).abs() < 1e-12);
            assert!((q.distance_to(&b.center) - 1.0).abs() < 1e-12);
        }
        let far = Circle { center: Vec2::new(5.0, 0.0), radius: 1.0 };
        assert!(circle_circle(&a, &far).is_none());
    }

    #[test]
    fn test_sphere_overlap_and_contact() {
        let a = Sphere { center: Vec3::ZERO, radius: 1.0 };
        let b = Sphere { center: Vec3::new(1.5, 0.0, 0.0), radius: 1.0 };
        assert!(sphere_sphere(&a, &b));
        let (n, pen) = sphere_sphere_contact(&a, &b).unwrap();
        assert!(n.distance_to(&xhat()) < 1e-12);
        assert!((pen - 0.5).abs() < 1e-12);
        let c = Sphere { center: Vec3::new(3.0, 0.0, 0.0), radius: 1.0 };
        assert!(!sphere_sphere(&a, &c));
        assert!(sphere_sphere_contact(&a, &c).is_none());
    }

    #[test]
    fn test_box_overlap_family() {
        let a = Aabb { min: Vec3::ZERO, max: Vec3::new(2.0, 2.0, 2.0) };
        let b = Aabb { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(3.0, 3.0, 3.0) };
        let c = Aabb { min: Vec3::new(5.0, 5.0, 5.0), max: Vec3::new(6.0, 6.0, 6.0) };
        assert!(aabb_aabb(&a, &b));
        assert!(!aabb_aabb(&a, &c));
        assert!(rect_rect(
            &Rect { min: Vec2::ZERO, max: Vec2::new(1.0, 1.0) },
            &Rect { min: Vec2::new(0.5, 0.5), max: Vec2::new(2.0, 2.0) }
        ));
        assert!(sphere_aabb(&Sphere { center: Vec3::new(3.0, 1.0, 1.0), radius: 1.1 }, &a));
        assert!(!sphere_aabb(&Sphere { center: Vec3::new(3.5, 1.0, 1.0), radius: 1.0 }, &a));
    }

    #[test]
    fn test_sphere_triangle() {
        let t = Triangle {
            a: Vec3::ZERO,
            b: Vec3::new(2.0, 0.0, 0.0),
            c: Vec3::new(0.0, 2.0, 0.0),
        };
        // Sphere centered over the interior at distance 0.6 < r = 1.
        let s = Sphere { center: Vec3::new(0.5, 0.5, 0.6), radius: 1.0 };
        let (contact, pen) = sphere_triangle(&s, &t).unwrap();
        assert!(contact.distance_to(&Vec3::new(0.5, 0.5, 0.0)) < 1e-12);
        assert!((pen - 0.4).abs() < 1e-12);
        // Contact point lies on the triangle plane and is the closest point.
        assert!(
            (contact.distance_to(&s.center)
                - crate::spatial::distance::distance_point_triangle(s.center, &t))
            .abs()
                < 1e-12
        );
        // Far sphere misses.
        let far = Sphere { center: Vec3::new(0.5, 0.5, 5.0), radius: 1.0 };
        assert!(sphere_triangle(&far, &t).is_none());
        // Grazing at exactly r counts as contact (closed test) with zero
        // penetration.
        let graze = Sphere { center: Vec3::new(0.5, 0.5, 1.0), radius: 1.0 };
        let (gc, gp) = sphere_triangle(&graze, &t).unwrap();
        assert!(gc.distance_to(&Vec3::new(0.5, 0.5, 0.0)) < 1e-12);
        assert!(gp.abs() < 1e-12);
        // Near a vertex: contact clamps to the vertex region.
        let corner = Sphere { center: Vec3::new(-0.3, -0.3, 0.0), radius: 0.5 };
        let (vc, vp) = sphere_triangle(&corner, &t).unwrap();
        assert!(vc.distance_to(&t.a) < 1e-12);
        assert!((vp - (0.5 - 0.3 * 2.0_f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn test_obb_obb_matches_aabb_for_identity() {
        let mk = |center: Vec3, h: Vec3| Obb {
            center,
            half_extents: h,
            rotation: Mat3::identity(),
        };
        let a = mk(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = mk(Vec3::new(1.5, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let c = mk(Vec3::new(3.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5));
        assert!(obb_obb(&a, &b));
        assert!(!obb_obb(&a, &c));
        // Rotated case: diagonal box overlaps where axis-aligned wouldn't.
        let rot = Obb {
            center: Vec3::new(2.0, 0.0, 0.0),
            half_extents: Vec3::new(1.4, 0.1, 0.1),
            rotation: rotation_z(std::f64::consts::FRAC_PI_4),
        };
        assert!(obb_obb(&a, &rot));
    }

    #[test]
    fn test_triangle_triangle() {
        let a = Triangle {
            a: Vec3::ZERO,
            b: Vec3::new(2.0, 0.0, 0.0),
            c: Vec3::new(0.0, 2.0, 0.0),
        };
        // Piercing triangle.
        let b = Triangle {
            a: Vec3::new(0.5, 0.5, -1.0),
            b: Vec3::new(0.5, 0.5, 1.0),
            c: Vec3::new(1.5, 1.5, 1.0),
        };
        assert!(triangle_triangle(&a, &b));
        // Far away.
        let c = Triangle {
            a: Vec3::new(5.0, 5.0, 5.0),
            b: Vec3::new(6.0, 5.0, 5.0),
            c: Vec3::new(5.0, 6.0, 5.0),
        };
        assert!(!triangle_triangle(&a, &c));
        // Coplanar overlapping.
        let d = Triangle {
            a: Vec3::new(0.5, 0.5, 0.0),
            b: Vec3::new(2.5, 0.5, 0.0),
            c: Vec3::new(0.5, 2.5, 0.0),
        };
        assert!(triangle_triangle(&a, &d));
        // Coplanar disjoint.
        let e = Triangle {
            a: Vec3::new(5.0, 5.0, 0.0),
            b: Vec3::new(6.0, 5.0, 0.0),
            c: Vec3::new(5.0, 6.0, 0.0),
        };
        assert!(!triangle_triangle(&a, &e));
    }

    #[test]
    fn test_plane_plane_and_three_planes() {
        let a = Plane::from_point_normal(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let b = Plane::from_point_normal(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let line = plane_plane(&a, &b).unwrap();
        assert!(line.dir.cross(&Vec3::new(0.0, 0.0, 1.0)).magnitude() < 1e-12);
        assert!(a.signed_distance(line.origin).abs() < 1e-12);
        assert!(b.signed_distance(line.origin).abs() < 1e-12);
        assert!(plane_plane(&a, &a).is_none());

        let c = Plane::from_point_normal(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 1.0));
        let p = three_planes(&a, &b, &c).unwrap();
        assert!(p.distance_to(&Vec3::new(0.0, 0.0, 2.0)) < 1e-12);
    }

    #[test]
    fn test_triangle_aabb() {
        let b = Aabb { min: Vec3::new(-1.0, -1.0, -1.0), max: Vec3::new(1.0, 1.0, 1.0) };
        let inside = Triangle {
            a: Vec3::new(-0.5, -0.5, 0.0),
            b: Vec3::new(0.5, -0.5, 0.0),
            c: Vec3::new(0.0, 0.5, 0.0),
        };
        assert!(triangle_aabb(&inside, &b));
        // Large triangle slicing through the box.
        let slicing = Triangle {
            a: Vec3::new(-5.0, 0.0, -5.0),
            b: Vec3::new(5.0, 0.0, -5.0),
            c: Vec3::new(0.0, 0.0, 5.0),
        };
        assert!(triangle_aabb(&slicing, &b));
        let outside = Triangle {
            a: Vec3::new(3.0, 3.0, 3.0),
            b: Vec3::new(4.0, 3.0, 3.0),
            c: Vec3::new(3.0, 4.0, 3.0),
        };
        assert!(!triangle_aabb(&outside, &b));
    }

    #[test]
    fn test_polygon_polygon() {
        let a = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]);
        let b = Polygon2::new(vec![
            Vec2::new(1.0, 1.0),
            Vec2::new(3.0, 1.0),
            Vec2::new(3.0, 3.0),
            Vec2::new(1.0, 3.0),
        ]);
        let far = Polygon2::new(vec![
            Vec2::new(10.0, 10.0),
            Vec2::new(11.0, 10.0),
            Vec2::new(10.0, 11.0),
        ]);
        assert!(polygon_polygon_2d(&a, &b));
        assert!(!polygon_polygon_2d(&a, &far));
        // Containment without edge crossings.
        let inner = Polygon2::new(vec![
            Vec2::new(0.5, 0.5),
            Vec2::new(1.5, 0.5),
            Vec2::new(1.0, 1.5),
        ]);
        assert!(polygon_polygon_2d(&a, &inner));
    }
}
