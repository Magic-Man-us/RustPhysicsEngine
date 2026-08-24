//! Signed distance fields: primitives, combinators, domain operators,
//! and queries (sphere tracing, normals, AO, soft shadows).
//!
//! Primitive formulas follow Inigo Quilez's reference catalogue
//! (iquilezles.org/articles/distfunctions). Negative inside, positive
//! outside; all primitive SDFs are exact unless noted.

use crate::math::{Vec2, Vec3};
use crate::spatial::intersect::RayHit;
use crate::spatial::primitives::{Aabb, Polygon2, Ray, Rect};

/// Boxed 3-D signed distance function.
pub type Sdf3 = Box<dyn Fn(Vec3) -> f64>;
/// Boxed 2-D signed distance function.
pub type Sdf2 = Box<dyn Fn(Vec2) -> f64>;

// ── 3-D primitives ──────────────────────────────────────────────────

/// Sphere of radius r at the origin: |p| − r.
#[must_use]
pub fn sd_sphere(p: Vec3, r: f64) -> f64 {
    p.magnitude() - r
}

/// Axis-aligned box with the given half extents.
#[must_use]
pub fn sd_box(p: Vec3, half: Vec3) -> f64 {
    let q = Vec3::new(p.x.abs() - half.x, p.y.abs() - half.y, p.z.abs() - half.z);
    let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).magnitude();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

/// Box with edges rounded by radius r.
#[must_use]
pub fn sd_rounded_box(p: Vec3, half: Vec3, r: f64) -> f64 {
    sd_box(p, half) - r
}

/// Torus in the xz-plane: major radius to the tube center, minor tube
/// radius.
#[must_use]
pub fn sd_torus(p: Vec3, major: f64, minor: f64) -> f64 {
    let q = Vec2::new(Vec2::new(p.x, p.z).magnitude() - major, p.y);
    q.magnitude() - minor
}

/// Capsule between a and b with radius r.
#[must_use]
pub fn sd_capsule(p: Vec3, a: Vec3, b: Vec3, r: f64) -> f64 {
    let pa = p - a;
    let ba = b - a;
    let h = (pa.dot(&ba) / ba.magnitude_squared()).clamp(0.0, 1.0);
    (pa - ba * h).magnitude() - r
}

/// Finite capped cylinder between a and b with radius r (exact).
#[must_use]
pub fn sd_cylinder(p: Vec3, a: Vec3, b: Vec3, r: f64) -> f64 {
    let ba = b - a;
    let pa = p - a;
    let baba = ba.magnitude_squared();
    let paba = pa.dot(&ba);
    let x = (pa * baba - ba * paba).magnitude() - r * baba;
    let y = (paba - baba * 0.5).abs() - baba * 0.5;
    let x2 = x * x;
    let y2 = y * y * baba;
    let d = if x.max(y) < 0.0 {
        -x2.min(y2)
    } else {
        (if x > 0.0 { x2 } else { 0.0 }) + (if y > 0.0 { y2 } else { 0.0 })
    };
    d.signum() * d.abs().sqrt() / baba
}

/// Infinite-precision capped cone with apex at the origin opening
/// along −y: half-angle `angle`, height h (IQ's sdCone, exact).
#[must_use]
pub fn sd_cone(p: Vec3, angle: f64, h: f64) -> f64 {
    let q = Vec2::new(h * angle.tan(), -h); // base radius, -height
    let w = Vec2::new(Vec2::new(p.x, p.z).magnitude(), p.y);
    let a = w - q * (w.dot(&q) / q.magnitude_squared()).clamp(0.0, 1.0);
    let b = w - Vec2::new(q.x * (w.x / q.x).clamp(0.0, 1.0), q.y);
    let k = q.y.signum();
    let d = a.magnitude_squared().min(b.magnitude_squared());
    let s = (k * (w.x * q.y - w.y * q.x)).max(k * (w.y - q.y));
    d.sqrt() * s.signum()
}

/// Half-space n·p + d = 0 (n need not be unit; it is normalized).
#[must_use]
pub fn sd_plane(p: Vec3, n: Vec3, d: f64) -> f64 {
    p.dot(&n.normalized()) + d
}

/// Ellipsoid with the given semi-axes (IQ's bound-improved
/// approximation; not exact away from the axes).
#[must_use]
pub fn sd_ellipsoid(p: Vec3, radii: Vec3) -> f64 {
    let k0 = Vec3::new(p.x / radii.x, p.y / radii.y, p.z / radii.z).magnitude();
    let k1 = Vec3::new(
        p.x / (radii.x * radii.x),
        p.y / (radii.y * radii.y),
        p.z / (radii.z * radii.z),
    )
    .magnitude();
    if k1 == 0.0 {
        return -radii.x.min(radii.y).min(radii.z);
    }
    k0 * (k0 - 1.0) / k1
}

/// Regular octahedron with "radius" s (exact).
#[must_use]
pub fn sd_octahedron(p: Vec3, s: f64) -> f64 {
    let p = Vec3::new(p.x.abs(), p.y.abs(), p.z.abs());
    let m = p.x + p.y + p.z - s;
    let q = if 3.0 * p.x < m {
        p
    } else if 3.0 * p.y < m {
        Vec3::new(p.y, p.z, p.x)
    } else if 3.0 * p.z < m {
        Vec3::new(p.z, p.x, p.y)
    } else {
        return m * (1.0 / 3.0_f64.sqrt());
    };
    let k = (0.5 * (q.z - q.y + s)).clamp(0.0, s);
    Vec3::new(q.x, q.y - s + k, q.z - k).magnitude()
}

// ── 2-D primitives ──────────────────────────────────────────────────

/// Circle of radius r at the origin.
#[must_use]
pub fn sd_circle(p: Vec2, r: f64) -> f64 {
    p.magnitude() - r
}

/// Axis-aligned rectangle with the given half extents.
#[must_use]
pub fn sd_rect(p: Vec2, half: Vec2) -> f64 {
    let q = Vec2::new(p.x.abs() - half.x, p.y.abs() - half.y);
    Vec2::new(q.x.max(0.0), q.y.max(0.0)).magnitude() + q.x.max(q.y).min(0.0)
}

/// Unsigned distance to a 2-D segment minus nothing (a "line" SDF).
#[must_use]
pub fn sd_segment_2d(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let pa = p - a;
    let ba = b - a;
    let denom = ba.magnitude_squared();
    let h = if denom > 0.0 { (pa.dot(&ba) / denom).clamp(0.0, 1.0) } else { 0.0 };
    (pa - ba * h).magnitude()
}

/// Signed distance to a simple polygon (negative inside).
#[must_use]
pub fn sd_polygon_2d(p: Vec2, poly: &Polygon2) -> f64 {
    let d = crate::spatial::distance::distance_point_polygon_2d(p, poly);
    if crate::spatial::contain::point_in_polygon_2d(p, poly) {
        -d
    } else {
        d
    }
}

/// Regular hexagon with circumscribed radius derived from apothem r
/// (IQ's sdHexagon: r is the apothem / inradius).
#[must_use]
pub fn sd_hexagon(p: Vec2, r: f64) -> f64 {
    let k = (-0.866_025_403_784_438_6, 0.5, 0.577_350_269_189_625_8);
    let mut p = Vec2::new(p.x.abs(), p.y.abs());
    let kxy = Vec2::new(k.0, k.1);
    let dot = kxy.dot(&p).min(0.0);
    p = p - kxy * (2.0 * dot);
    p = p - Vec2::new(p.x.clamp(-k.2 * r, k.2 * r), r);
    p.magnitude() * (p.y).signum()
}

/// n-pointed star with outer radius r and inner-radius factor set by m
/// (IQ's sdStar; m between 2 and n controls pointiness).
#[must_use]
pub fn sd_star(p: Vec2, r: f64, n: u32, m: f64) -> f64 {
    let an = std::f64::consts::PI / n as f64;
    let en = std::f64::consts::PI / m;
    let acs = Vec2::new(an.cos(), an.sin());
    let ecs = Vec2::new(en.cos(), en.sin());
    let bn = p.y.atan2(p.x).abs().rem_euclid(2.0 * an) - an;
    let mut q = Vec2::new(bn.cos(), bn.sin().abs()) * p.magnitude();
    q = q - acs * r;
    q = q + ecs * (-q.dot(&ecs)).clamp(0.0, r * acs.y / ecs.y);
    q.magnitude() * q.x.signum()
}

// ── Operators ───────────────────────────────────────────────────────

/// Union: min(a, b).
#[must_use]
pub fn op_union(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Subtraction (a minus b): max(a, −b).
#[must_use]
pub fn op_subtract(a: f64, b: f64) -> f64 {
    a.max(-b)
}

/// Intersection: max(a, b).
#[must_use]
pub fn op_intersect(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Polynomial smooth union with blending radius k.
#[must_use]
pub fn op_smooth_union(a: f64, b: f64, k: f64) -> f64 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

/// Smooth subtraction.
#[must_use]
pub fn op_smooth_subtract(a: f64, b: f64, k: f64) -> f64 {
    -op_smooth_union(-a, b, k)
}

/// Smooth intersection.
#[must_use]
pub fn op_smooth_intersect(a: f64, b: f64, k: f64) -> f64 {
    -op_smooth_union(-a, -b, k)
}

/// Rounds a shape outward by r.
#[must_use]
pub fn op_round(d: f64, r: f64) -> f64 {
    d - r
}

/// Hollows a shape into a shell of the given thickness.
#[must_use]
pub fn op_onion(d: f64, thickness: f64) -> f64 {
    d.abs() - thickness
}

/// Infinite domain repetition with the given period per axis
/// (returns the point folded into the central cell).
#[must_use]
pub fn op_repeat(p: Vec3, period: Vec3) -> Vec3 {
    let fold = |x: f64, c: f64| {
        if c > 0.0 {
            (x + 0.5 * c).rem_euclid(c) - 0.5 * c
        } else {
            x
        }
    };
    Vec3::new(fold(p.x, period.x), fold(p.y, period.y), fold(p.z, period.z))
}

/// Limited repetition: at most `count` cells either side per axis.
#[must_use]
pub fn op_repeat_limited(p: Vec3, period: Vec3, count: [i32; 3]) -> Vec3 {
    let fold = |x: f64, c: f64, n: i32| {
        if c > 0.0 && n > 0 {
            x - c * (x / c).round().clamp(-(n as f64), n as f64)
        } else {
            x
        }
    };
    Vec3::new(
        fold(p.x, period.x, count[0]),
        fold(p.y, period.y, count[1]),
        fold(p.z, period.z, count[2]),
    )
}

/// Mirror the chosen axes (|x| fold).
#[must_use]
pub fn op_mirror(p: Vec3, axes: [bool; 3]) -> Vec3 {
    Vec3::new(
        if axes[0] { p.x.abs() } else { p.x },
        if axes[1] { p.y.abs() } else { p.y },
        if axes[2] { p.z.abs() } else { p.z },
    )
}

/// Twist about the y axis by k radians per unit height.
#[must_use]
pub fn op_twist(p: Vec3, k: f64) -> Vec3 {
    let (s, c) = (k * p.y).sin_cos();
    Vec3::new(c * p.x - s * p.z, p.y, s * p.x + c * p.z)
}

/// Bend about the z axis with curvature k.
#[must_use]
pub fn op_bend(p: Vec3, k: f64) -> Vec3 {
    let (s, c) = (k * p.x).sin_cos();
    Vec3::new(c * p.x - s * p.y, s * p.x + c * p.y, p.z)
}

/// Elongation: stretches the shape by clamping the sample point.
#[must_use]
pub fn op_elongate(p: Vec3, h: Vec3) -> Vec3 {
    Vec3::new(
        p.x - p.x.clamp(-h.x, h.x),
        p.y - p.y.clamp(-h.y, h.y),
        p.z - p.z.clamp(-h.z, h.z),
    )
}

/// Polar repetition: folds the plane into one of n angular sectors.
#[must_use]
pub fn op_polar_repeat_2d(p: Vec2, n: u32) -> Vec2 {
    let sector = std::f64::consts::TAU / n.max(1) as f64;
    let angle = p.y.atan2(p.x).rem_euclid(sector) - 0.5 * sector;
    let r = p.magnitude();
    Vec2::new(r * angle.cos(), r * angle.sin())
}

// ── Queries ─────────────────────────────────────────────────────────

/// Central-difference gradient normalized to a surface normal.
#[must_use]
pub fn sdf_normal(f: &dyn Fn(Vec3) -> f64, p: Vec3, eps: f64) -> Vec3 {
    let dx = f(Vec3::new(p.x + eps, p.y, p.z)) - f(Vec3::new(p.x - eps, p.y, p.z));
    let dy = f(Vec3::new(p.x, p.y + eps, p.z)) - f(Vec3::new(p.x, p.y - eps, p.z));
    let dz = f(Vec3::new(p.x, p.y, p.z + eps)) - f(Vec3::new(p.x, p.y, p.z - eps));
    Vec3::new(dx, dy, dz).normalized()
}

/// Sphere tracing: march the ray by the SDF value until |f| < eps.
#[must_use]
pub fn sdf_raymarch(
    f: &dyn Fn(Vec3) -> f64,
    r: &Ray,
    max_dist: f64,
    eps: f64,
    max_steps: usize,
) -> Option<RayHit> {
    let mut t = 0.0;
    for _ in 0..max_steps {
        let p = r.at(t);
        let d = f(p);
        if d.abs() < eps {
            return Some(RayHit { t, point: p, normal: sdf_normal(f, p, eps.max(1e-6)) });
        }
        t += d.max(eps); // always advance to escape grazing regions
        if t > max_dist {
            return None;
        }
    }
    None
}

/// Samples the SDF on a regular grid (x-fastest order:
/// `data[k*ny*nx + j*nx + i]`), suitable for marching cubes.
///
/// # Panics
/// Panics if any resolution is < 2.
#[must_use]
pub fn sdf_to_grid(f: &dyn Fn(Vec3) -> f64, bounds: &Aabb, res: [usize; 3]) -> Vec<f64> {
    assert!(res.iter().all(|&r| r >= 2), "sdf_to_grid requires res >= 2 per axis");
    let [nx, ny, nz] = res;
    let mut out = Vec::with_capacity(nx * ny * nz);
    let d = bounds.max - bounds.min;
    for k in 0..nz {
        let z = bounds.min.z + d.z * k as f64 / (nz - 1) as f64;
        for j in 0..ny {
            let y = bounds.min.y + d.y * j as f64 / (ny - 1) as f64;
            for i in 0..nx {
                let x = bounds.min.x + d.x * i as f64 / (nx - 1) as f64;
                out.push(f(Vec3::new(x, y, z)));
            }
        }
    }
    out
}

/// 2-D grid sampling (row-major, `data[j*nx + i]`).
///
/// # Panics
/// Panics if any resolution is < 2.
#[must_use]
pub fn sdf_to_grid_2d(f: &dyn Fn(Vec2) -> f64, bounds: &Rect, res: [usize; 2]) -> Vec<f64> {
    assert!(res.iter().all(|&r| r >= 2), "sdf_to_grid_2d requires res >= 2 per axis");
    let [nx, ny] = res;
    let mut out = Vec::with_capacity(nx * ny);
    let d = bounds.max - bounds.min;
    for j in 0..ny {
        let y = bounds.min.y + d.y * j as f64 / (ny - 1) as f64;
        for i in 0..nx {
            let x = bounds.min.x + d.x * i as f64 / (nx - 1) as f64;
            out.push(f(Vec2::new(x, y)));
        }
    }
    out
}

/// Screen-space-style ambient occlusion: samples the SDF along the
/// normal; 1 = fully open, 0 = fully occluded.
#[must_use]
pub fn sdf_ambient_occlusion(
    f: &dyn Fn(Vec3) -> f64,
    p: Vec3,
    n: Vec3,
    steps: usize,
    step_size: f64,
) -> f64 {
    let mut occlusion = 0.0;
    let mut weight = 1.0;
    for i in 1..=steps {
        let d = step_size * i as f64;
        let sample = f(p + n * d);
        occlusion += weight * (d - sample).max(0.0);
        weight *= 0.5;
    }
    (1.0 - occlusion).clamp(0.0, 1.0)
}

/// IQ soft shadow: marches from `origin` along `dir` and darkens by the
/// closest approach scaled by k (larger k = harder shadow). Returns a
/// factor in [0, 1].
#[must_use]
pub fn sdf_soft_shadow(f: &dyn Fn(Vec3) -> f64, origin: Vec3, dir: Vec3, k: f64) -> f64 {
    let d = dir.normalized();
    if d.magnitude_squared() == 0.0 {
        return 1.0;
    }
    let mut res = 1.0_f64;
    let mut t = 1e-3;
    for _ in 0..128 {
        let h = f(origin + d * t);
        if h < 1e-6 {
            return 0.0;
        }
        res = res.min(k * h / t);
        t += h;
        if t > 100.0 {
            break;
        }
    }
    res.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_exact() {
        for &(p, r) in &[(Vec3::new(3.0, 0.0, 0.0), 1.0), (Vec3::new(0.0, 0.5, 0.0), 2.0)] {
            assert_eq!(sd_sphere(p, r), p.magnitude() - r);
        }
    }

    #[test]
    fn test_box_inside_outside() {
        let h = Vec3::new(1.0, 1.0, 1.0);
        assert!((sd_box(Vec3::new(2.0, 0.0, 0.0), h) - 1.0).abs() < 1e-12);
        assert!((sd_box(Vec3::ZERO, h) + 1.0).abs() < 1e-12);
        assert!(sd_box(Vec3::new(1.0, 0.0, 0.0), h).abs() < 1e-12);
        // Corner distance.
        let corner = sd_box(Vec3::new(2.0, 2.0, 2.0), h);
        assert!((corner - 3.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_torus_capsule_cylinder() {
        // Point on the tube center circle.
        assert!((sd_torus(Vec3::new(2.0, 0.0, 0.0), 2.0, 0.5) + 0.5).abs() < 1e-12);
        assert!(sd_torus(Vec3::new(2.5, 0.0, 0.0), 2.0, 0.5).abs() < 1e-12);

        let a = Vec3::new(0.0, -1.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((sd_capsule(Vec3::new(1.0, 0.0, 0.0), a, b, 0.25) - 0.75).abs() < 1e-12);
        assert!((sd_capsule(Vec3::new(0.0, 2.0, 0.0), a, b, 0.25) - 0.75).abs() < 1e-12);

        assert!(sd_cylinder(Vec3::new(0.5, 0.0, 0.0), a, b, 0.5).abs() < 1e-9);
        assert!(sd_cylinder(Vec3::ZERO, a, b, 0.5) < 0.0);
        assert!((sd_cylinder(Vec3::new(0.0, 2.0, 0.0), a, b, 0.5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_plane_and_octahedron() {
        assert!((sd_plane(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, 2.0, 0.0), -1.0) - 2.0).abs() < 1e-12);
        assert!(sd_octahedron(Vec3::new(1.0, 0.0, 0.0), 1.0).abs() < 1e-12);
        assert!(sd_octahedron(Vec3::ZERO, 1.0) < 0.0);
    }

    #[test]
    fn test_2d_primitives() {
        assert!((sd_circle(Vec2::new(3.0, 4.0), 2.0) - 3.0).abs() < 1e-12);
        assert!((sd_rect(Vec2::new(2.0, 0.0), Vec2::new(1.0, 1.0)) - 1.0).abs() < 1e-12);
        assert!((sd_segment_2d(Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0)) - 1.0).abs() < 1e-12);
        // Hexagon: on-edge along y at the apothem.
        assert!(sd_hexagon(Vec2::new(0.0, 1.0), 1.0).abs() < 1e-9);
        // Star: outer point is on the surface.
        let tip = sd_star(Vec2::new(1.0, 0.0), 1.0, 5, 3.0);
        assert!(tip.abs() < 1e-9, "star tip distance {tip}");

        let poly = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]);
        assert!((sd_polygon_2d(Vec2::new(1.0, 1.0), &poly) + 1.0).abs() < 1e-12);
        assert!((sd_polygon_2d(Vec2::new(3.0, 1.0), &poly) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_operators() {
        let (a, b) = (1.5, -0.5);
        assert_eq!(op_union(a, b), -0.5);
        assert_eq!(op_intersect(a, b), 1.5);
        assert_eq!(op_subtract(a, b), 1.5);
        // Smooth union lower-bounds the hard union and matches far away.
        assert!(op_smooth_union(a, b, 0.2) <= op_union(a, b) + 1e-12);
        assert!((op_smooth_union(10.0, 0.0, 0.1) - 0.0).abs() < 1e-9);
        assert_eq!(op_round(1.0, 0.25), 0.75);
        assert_eq!(op_onion(-0.5, 0.1), 0.4);
    }

    #[test]
    fn test_domain_operators() {
        // Repetition folds distant cells onto the origin cell.
        let p = op_repeat(Vec3::new(5.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        assert!((p.x - 1.0).abs() < 1e-12 || (p.x + 1.0).abs() < 1e-12);
        let pl = op_repeat_limited(Vec3::new(9.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), [2, 0, 0]);
        assert!((pl.x - 5.0).abs() < 1e-12); // clamped to 2 cells
        let m = op_mirror(Vec3::new(-1.0, 2.0, -3.0), [true, false, true]);
        assert_eq!(m, Vec3::new(1.0, 2.0, 3.0));
        // Twist/bend/elongate keep the y magnitude reasonable.
        let tw = op_twist(Vec3::new(1.0, 0.5, 0.0), 1.0);
        assert!((Vec2::new(tw.x, tw.z).magnitude() - 1.0).abs() < 1e-12);
        let el = op_elongate(Vec3::new(3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(el, Vec3::new(2.0, 0.0, 0.0));
        let pr = op_polar_repeat_2d(Vec2::new(0.0, 2.0), 4);
        assert!((pr.magnitude() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_normal_and_lipschitz() {
        let f = |p: Vec3| sd_sphere(p, 1.0);
        let p = Vec3::new(2.0, 1.0, -0.5);
        let n = sdf_normal(&f, p, 1e-6);
        assert!(n.distance_to(&p.normalized()) < 1e-6);
        // |grad| ~ 1 for exact SDFs away from the surface/center.
        for prim in [
            Box::new(|q: Vec3| sd_box(q, Vec3::new(1.0, 0.5, 0.7))) as Box<dyn Fn(Vec3) -> f64>,
            Box::new(|q: Vec3| sd_torus(q, 2.0, 0.5)),
            Box::new(|q: Vec3| sd_capsule(q, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.3)),
        ] {
            let q = Vec3::new(1.7, 2.3, -1.1);
            let eps = 1e-5;
            let g = Vec3::new(
                prim(Vec3::new(q.x + eps, q.y, q.z)) - prim(Vec3::new(q.x - eps, q.y, q.z)),
                prim(Vec3::new(q.x, q.y + eps, q.z)) - prim(Vec3::new(q.x, q.y - eps, q.z)),
                prim(Vec3::new(q.x, q.y, q.z + eps)) - prim(Vec3::new(q.x, q.y, q.z - eps)),
            ) * (1.0 / (2.0 * eps));
            assert!((g.magnitude() - 1.0).abs() < 1e-3, "gradient {}", g.magnitude());
        }
    }

    #[test]
    fn test_raymarch_matches_analytic_sphere() {
        let f = |p: Vec3| sd_sphere(p, 1.0);
        let r = Ray::new(Vec3::new(-5.0, 0.2, 0.1), Vec3::new(1.0, 0.0, 0.0));
        let marched = sdf_raymarch(&f, &r, 100.0, 1e-9, 256).unwrap();
        let analytic = crate::spatial::intersect::ray_sphere(
            &r,
            &crate::spatial::primitives::Sphere { center: Vec3::ZERO, radius: 1.0 },
        )
        .unwrap();
        assert!((marched.t - analytic.t).abs() < 1e-6);
        assert!(marched.normal.distance_to(&analytic.normal) < 1e-4);
        // Miss goes past max distance.
        let miss = Ray::new(Vec3::new(-5.0, 3.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(sdf_raymarch(&f, &miss, 20.0, 1e-9, 256).is_none());
    }

    #[test]
    fn test_grids_ao_shadow() {
        let f = |p: Vec3| sd_sphere(p, 1.0);
        let bounds = Aabb { min: Vec3::new(-2.0, -2.0, -2.0), max: Vec3::new(2.0, 2.0, 2.0) };
        let grid = sdf_to_grid(&f, &bounds, [5, 5, 5]);
        assert_eq!(grid.len(), 125);
        // Center sample is the deepest inside.
        assert!((grid[2 * 25 + 2 * 5 + 2] + 1.0).abs() < 1e-12);

        let f2 = |p: Vec2| sd_circle(p, 1.0);
        let g2 = sdf_to_grid_2d(&f2, &Rect { min: Vec2::new(-2.0, -2.0), max: Vec2::new(2.0, 2.0) }, [5, 5]);
        assert_eq!(g2.len(), 25);
        assert!((g2[2 * 5 + 2] + 1.0).abs() < 1e-12);

        // Open sky: AO near 1; facing a wall: reduced.
        let p = Vec3::new(1.0, 0.0, 0.0);
        let ao_open = sdf_ambient_occlusion(&f, p, Vec3::new(1.0, 0.0, 0.0), 5, 0.1);
        assert!(ao_open > 0.95);
        // Shadow toward the sphere is dark; away is bright.
        let origin = Vec3::new(3.0, 0.0, 0.0);
        assert!(sdf_soft_shadow(&f, origin, Vec3::new(-1.0, 0.0, 0.0), 8.0) < 0.05);
        assert!(sdf_soft_shadow(&f, origin, Vec3::new(1.0, 0.0, 0.0), 8.0) > 0.95);
    }
}
