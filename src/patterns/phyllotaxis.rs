//! Phyllotactic patterns and spirals: Vogel sunflowers, Fibonacci
//! point sets, the classical spiral family, and parastichy analysis.

use crate::math::{Vec2, Vec3};

/// The golden angle 2π(1 − 1/φ) ≈ 137.5°, the turn between
/// successive florets in phyllotaxis.
pub const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

/// Vogel's sunflower model (Vogel 1979): floret i at radius
/// `scale · √i` and angle `i · GOLDEN_ANGLE`.
#[must_use]
pub fn vogel_sunflower(n: usize, scale: f64) -> Vec<Vec2> {
    vogel_sunflower_angle(n, scale, GOLDEN_ANGLE)
}

/// Vogel model with an arbitrary divergence angle.
#[must_use]
pub fn vogel_sunflower_angle(n: usize, scale: f64, angle: f64) -> Vec<Vec2> {
    (0..n)
        .map(|i| {
            let r = scale * (i as f64).sqrt();
            let t = i as f64 * angle;
            Vec2::new(r * t.cos(), r * t.sin())
        })
        .collect()
}

/// Near-uniform points on the unit sphere: latitude strips of equal
/// area, longitude advanced by the golden angle.
///
/// # Panics
/// Panics unless `n >= 1`.
#[must_use]
pub fn fibonacci_sphere(n: usize) -> Vec<Vec3> {
    assert!(n >= 1, "fibonacci_sphere requires n >= 1");
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let t = i as f64 * GOLDEN_ANGLE;
            Vec3::new(r * t.cos(), y, r * t.sin())
        })
        .collect()
}

/// Near-uniform points on the unit disk (Vogel pattern scaled so the
/// n-th floret reaches radius 1).
///
/// # Panics
/// Panics unless `n >= 1`.
#[must_use]
pub fn fibonacci_disk(n: usize) -> Vec<Vec2> {
    assert!(n >= 1, "fibonacci_disk requires n >= 1");
    (0..n)
        .map(|i| {
            let r = ((i as f64 + 0.5) / n as f64).sqrt();
            let t = i as f64 * GOLDEN_ANGLE;
            Vec2::new(r * t.cos(), r * t.sin())
        })
        .collect()
}

/// Near-uniform points on the upper (y > 0) unit hemisphere.
///
/// # Panics
/// Panics unless `n >= 1`.
#[must_use]
pub fn fibonacci_hemisphere(n: usize) -> Vec<Vec3> {
    assert!(n >= 1, "fibonacci_hemisphere requires n >= 1");
    (0..n)
        .map(|i| {
            let y = (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let t = i as f64 * GOLDEN_ANGLE;
            Vec3::new(r * t.cos(), y, r * t.sin())
        })
        .collect()
}

/// Golden spiral: logarithmic spiral growing by φ every quarter turn,
/// starting radius `a`.
///
/// # Panics
/// Panics unless `turns > 0`, `points_per_turn >= 1`, `a > 0`.
#[must_use]
pub fn golden_spiral(turns: f64, points_per_turn: usize, a: f64) -> Vec<Vec2> {
    assert!(turns > 0.0 && points_per_turn >= 1 && a > 0.0, "invalid golden_spiral arguments");
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    let b = phi.ln() / std::f64::consts::FRAC_PI_2;
    logarithmic_spiral(a, b, turns * 2.0 * std::f64::consts::PI, (turns * points_per_turn as f64).ceil() as usize)
}

/// Archimedean spiral r = a + bθ sampled on `n` points over
/// θ ∈ [0, theta_max].
///
/// # Panics
/// Panics unless `n >= 2`.
#[must_use]
pub fn archimedean_spiral(a: f64, b: f64, theta_max: f64, n: usize) -> Vec<Vec2> {
    assert!(n >= 2, "spiral needs >= 2 points");
    (0..n)
        .map(|i| {
            let t = theta_max * i as f64 / (n - 1) as f64;
            let r = a + b * t;
            Vec2::new(r * t.cos(), r * t.sin())
        })
        .collect()
}

/// Logarithmic spiral r = a e^{bθ}.
///
/// # Panics
/// Panics unless `n >= 2`.
#[must_use]
pub fn logarithmic_spiral(a: f64, b: f64, theta_max: f64, n: usize) -> Vec<Vec2> {
    assert!(n >= 2, "spiral needs >= 2 points");
    (0..n)
        .map(|i| {
            let t = theta_max * i as f64 / (n - 1) as f64;
            let r = a * (b * t).exp();
            Vec2::new(r * t.cos(), r * t.sin())
        })
        .collect()
}

/// Fermat (parabolic) spiral r = a √θ.
///
/// # Panics
/// Panics unless `n >= 2` and `theta_max >= 0`.
#[must_use]
pub fn fermat_spiral(a: f64, theta_max: f64, n: usize) -> Vec<Vec2> {
    assert!(n >= 2 && theta_max >= 0.0, "invalid fermat_spiral arguments");
    (0..n)
        .map(|i| {
            let t = theta_max * i as f64 / (n - 1) as f64;
            let r = a * t.sqrt();
            Vec2::new(r * t.cos(), r * t.sin())
        })
        .collect()
}

/// Hyperbolic spiral r = a/θ over `theta_range` (which must exclude
/// 0).
///
/// # Panics
/// Panics unless `n >= 2` and the range excludes zero.
#[must_use]
pub fn hyperbolic_spiral(a: f64, theta_range: (f64, f64), n: usize) -> Vec<Vec2> {
    assert!(n >= 2, "spiral needs >= 2 points");
    assert!(
        theta_range.0 * theta_range.1 > 0.0,
        "theta range must exclude zero"
    );
    (0..n)
        .map(|i| {
            let t = theta_range.0
                + (theta_range.1 - theta_range.0) * i as f64 / (n - 1) as f64;
            let r = a / t;
            Vec2::new(r * t.cos(), r * t.sin())
        })
        .collect()
}

/// Lituus r = a/√θ over `theta_range` (positive).
///
/// # Panics
/// Panics unless `n >= 2` and the range is positive.
#[must_use]
pub fn lituus(a: f64, theta_range: (f64, f64), n: usize) -> Vec<Vec2> {
    assert!(n >= 2, "spiral needs >= 2 points");
    assert!(theta_range.0 > 0.0 && theta_range.1 > theta_range.0, "range must be positive");
    (0..n)
        .map(|i| {
            let t = theta_range.0
                + (theta_range.1 - theta_range.0) * i as f64 / (n - 1) as f64;
            let r = a / t.sqrt();
            Vec2::new(r * t.cos(), r * t.sin())
        })
        .collect()
}

/// Euler spiral (clothoid): curvature grows linearly with arclength,
/// κ(s) = s. Points via composite-Simpson evaluation of the Fresnel
/// integrals x = ∫cos(t²/2)dt, y = ∫sin(t²/2)dt.
///
/// # Panics
/// Panics unless `n >= 2` and `length > 0`.
#[must_use]
pub fn euler_spiral(length: f64, n: usize) -> Vec<Vec2> {
    assert!(n >= 2 && length > 0.0, "invalid euler_spiral arguments");
    let mut out = Vec::with_capacity(n);
    let mut p = Vec2::ZERO;
    out.push(p);
    let h = length / (n - 1) as f64;
    for i in 1..n {
        let s0 = (i - 1) as f64 * h;
        // Simpson's rule over [s0, s0 + h].
        let f = |s: f64| Vec2::new((s * s / 2.0).cos(), (s * s / 2.0).sin());
        let seg = (f(s0) + f(s0 + h / 2.0) * 4.0 + f(s0 + h)) * (h / 6.0);
        p = p + seg;
        out.push(p);
    }
    out
}

/// Spiral of Theodorus (square-root spiral): `n` right triangles with
/// unit legs; vertex k lies at radius √(k+1).
///
/// # Panics
/// Panics unless `n >= 1`.
#[must_use]
pub fn spiral_of_theodorus(n: usize) -> Vec<Vec2> {
    assert!(n >= 1, "spiral needs n >= 1");
    let mut out = Vec::with_capacity(n);
    let mut p = Vec2::new(1.0, 0.0);
    out.push(p);
    for _ in 1..n {
        p = p + p.perp() * (1.0 / p.magnitude());
        out.push(p);
    }
    out
}

/// Conical spiral: radius `a + b t`, height `h t`, `turns` full turns
/// over t ∈ [0, 1], axis y.
///
/// # Panics
/// Panics unless `n >= 2`.
#[must_use]
pub fn conical_spiral(a: f64, b: f64, h: f64, turns: f64, n: usize) -> Vec<Vec3> {
    assert!(n >= 2, "spiral needs >= 2 points");
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let theta = 2.0 * std::f64::consts::PI * turns * t;
            let r = a + b * t;
            Vec3::new(r * theta.cos(), h * t, r * theta.sin())
        })
        .collect()
}

/// Spherical spiral on the unit sphere: polar angle sweeps 0..π while
/// the azimuth makes `turns` turns (axis y).
///
/// # Panics
/// Panics unless `n >= 2`.
#[must_use]
pub fn spherical_spiral(turns: f64, n: usize) -> Vec<Vec3> {
    assert!(n >= 2, "spiral needs >= 2 points");
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let theta = std::f64::consts::PI * t;
            let phi = 2.0 * std::f64::consts::PI * turns * t;
            Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin())
        })
        .collect()
}

/// Detects the two dominant parastichy (visible spiral) families:
/// the two most common index differences between each floret and its
/// nearest neighbors, returned ascending. For golden-angle patterns
/// these are consecutive Fibonacci numbers.
///
/// # Panics
/// Panics unless at least 8 points are given.
#[must_use]
pub fn parastichy_counts(points: &[Vec2]) -> (usize, usize) {
    assert!(points.len() >= 8, "parastichy detection needs >= 8 points");
    let mut votes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    // Skip the innermost florets where the pattern is degenerate.
    let start = points.len() / 8;
    for i in start..points.len() {
        // Two nearest neighbors (excluding self).
        let mut best = [(f64::INFINITY, 0usize); 2];
        for (j, q) in points.iter().enumerate() {
            if j == i {
                continue;
            }
            let d = q.distance_to(&points[i]);
            if d < best[0].0 {
                best[1] = best[0];
                best[0] = (d, j.abs_diff(i));
            } else if d < best[1].0 {
                best[1] = (d, j.abs_diff(i));
            }
        }
        for (_, diff) in best {
            *votes.entry(diff).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<(usize, usize)> = votes.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let (a, b) = (ranked[0].0, ranked.get(1).map_or(ranked[0].0, |r| r.0));
    (a.min(b), a.max(b))
}

/// Helical (cylindrical) phyllotaxis: point i at height `i · rise`
/// and azimuth `i · angle` on a cylinder of the given radius.
#[must_use]
pub fn cylinder_phyllotaxis(n: usize, rise: f64, angle: f64, radius: f64) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let t = i as f64 * angle;
            Vec3::new(radius * t.cos(), i as f64 * rise, radius * t.sin())
        })
        .collect()
}

/// Logarithmic-spiral arc from `a` to `b` about the origin: `n`
/// points (inclusive) interpolating radius geometrically and angle
/// linearly (shortest way around).
///
/// # Panics
/// Panics unless `n >= 2` and both points are away from the origin.
#[must_use]
pub fn spiral_interpolate_sequence(a: Vec2, b: Vec2, n: usize) -> Vec<Vec2> {
    assert!(n >= 2, "need >= 2 points");
    let (ra, rb) = (a.magnitude(), b.magnitude());
    assert!(ra > 0.0 && rb > 0.0, "points must be away from the origin");
    let ta = a.y.atan2(a.x);
    let mut tb = b.y.atan2(b.x);
    while tb - ta > std::f64::consts::PI {
        tb -= 2.0 * std::f64::consts::PI;
    }
    while ta - tb > std::f64::consts::PI {
        tb += 2.0 * std::f64::consts::PI;
    }
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let r = ra * (rb / ra).powf(t);
            let th = ta + (tb - ta) * t;
            Vec2::new(r * th.cos(), r * th.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_angle_value() {
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let expect = 2.0 * std::f64::consts::PI * (1.0 - 1.0 / phi);
        assert!((GOLDEN_ANGLE - expect).abs() < 1e-12);
    }

    #[test]
    fn test_fibonacci_sphere_uniformity() {
        let pts = fibonacci_sphere(1000);
        let mut nn = Vec::with_capacity(pts.len());
        for (i, p) in pts.iter().enumerate() {
            assert!((p.magnitude() - 1.0).abs() < 1e-12, "unit length");
            let d = pts
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, q)| q.distance_to(p))
                .fold(f64::INFINITY, f64::min);
            nn.push(d);
        }
        let mean: f64 = nn.iter().sum::<f64>() / nn.len() as f64;
        let var: f64 = nn.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / nn.len() as f64;
        let cv = var.sqrt() / mean;
        assert!(cv < 0.15, "nearest-neighbor CV {cv} too high");
        // Mean position near the center.
        let mut sum = Vec3::ZERO;
        for p in &pts {
            sum = sum + *p;
        }
        assert!((sum * (1.0 / 1000.0)).magnitude() < 0.01);
    }

    #[test]
    fn test_vogel_parastichy_fibonacci() {
        let pts = vogel_sunflower(500, 1.0);
        let (a, b) = parastichy_counts(&pts);
        let fib = [1usize, 2, 3, 5, 8, 13, 21, 34, 55];
        let consecutive = fib.windows(2).any(|w| w[0] == a && w[1] == b);
        assert!(consecutive, "parastichy counts ({a}, {b}) not consecutive Fibonacci");
    }

    #[test]
    fn test_disk_and_hemisphere() {
        let pts = fibonacci_disk(500);
        for p in &pts {
            assert!(p.magnitude() <= 1.0 + 1e-12);
        }
        // Quadrant balance.
        let q1 = pts.iter().filter(|p| p.x > 0.0 && p.y > 0.0).count();
        assert!((q1 as f64 - 125.0).abs() < 30.0);
        for p in fibonacci_hemisphere(200) {
            assert!((p.magnitude() - 1.0).abs() < 1e-12);
            assert!(p.y > 0.0);
        }
    }

    #[test]
    fn test_spiral_shapes() {
        // Archimedean: radius grows linearly with angle.
        let s = archimedean_spiral(0.0, 1.0, 4.0 * std::f64::consts::PI, 200);
        assert!((s[199].magnitude() - 4.0 * std::f64::consts::PI).abs() < 1e-9);
        // Logarithmic: constant ratio per fixed angle step.
        let l = logarithmic_spiral(1.0, 0.1, 6.0, 61);
        let r1 = l[10].magnitude() / l[0].magnitude();
        let r2 = l[20].magnitude() / l[10].magnitude();
        assert!((r1 - r2).abs() < 1e-9, "log spiral has constant growth");
        // Golden spiral quarter-turn ratio is phi.
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let g = golden_spiral(2.0, 90, 1.0);
        // Points sampled at (turns * ppt) steps over full range; find
        // ratio over a quarter turn = 22.5 steps... instead check the
        // analytic property r(theta) = exp(ln(phi) * theta/(pi/2)).
        let b = phi.ln() / std::f64::consts::FRAC_PI_2;
        for (i, p) in g.iter().enumerate() {
            let theta = 4.0 * std::f64::consts::PI * i as f64 / (g.len() - 1) as f64;
            assert!((p.magnitude() - (b * theta).exp()).abs() < 1e-9);
        }
        // Fermat, hyperbolic, lituus radii.
        let f = fermat_spiral(2.0, 9.0, 10);
        assert!((f[9].magnitude() - 6.0).abs() < 1e-12);
        let h = hyperbolic_spiral(3.0, (1.0, 3.0), 3);
        assert!((h[0].magnitude() - 3.0).abs() < 1e-12);
        assert!((h[2].magnitude() - 1.0).abs() < 1e-12);
        let li = lituus(2.0, (1.0, 4.0), 4);
        assert!((li[0].magnitude() - 2.0).abs() < 1e-12);
        assert!((li[3].magnitude() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_euler_spiral_curvature() {
        let n = 2001;
        let length = 3.0;
        let pts = euler_spiral(length, n);
        let h = length / (n - 1) as f64;
        // Discrete curvature at samples: turn angle / arclength ~ s.
        for &k in &[400usize, 1000, 1600] {
            let (a, b, c) = (pts[k - 1], pts[k], pts[k + 1]);
            let t1 = b - a;
            let t2 = c - b;
            let kappa = (t1.cross(&t2) / (t1.magnitude() * t2.magnitude())).asin() / h;
            let s = k as f64 * h;
            assert!((kappa - s).abs() < 0.01, "clothoid curvature {kappa} vs {s}");
        }
        // Unit speed: consecutive points are h apart.
        for w in pts.windows(2).take(50) {
            assert!((w[0].distance_to(&w[1]) - h).abs() < 1e-6);
        }
    }

    #[test]
    fn test_theodorus_radii() {
        let pts = spiral_of_theodorus(20);
        for (k, p) in pts.iter().enumerate() {
            assert!((p.magnitude() - ((k + 1) as f64).sqrt()).abs() < 1e-12);
        }
        // Unit legs.
        for w in pts.windows(2) {
            assert!((w[0].distance_to(&w[1]) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_3d_spirals_and_interpolation() {
        let c = conical_spiral(2.0, -1.0, 3.0, 4.0, 100);
        assert!((Vec2::new(c[0].x, c[0].z).magnitude() - 2.0).abs() < 1e-12);
        assert!((Vec2::new(c[99].x, c[99].z).magnitude() - 1.0).abs() < 1e-12);
        assert!((c[99].y - 3.0).abs() < 1e-12);
        for p in spherical_spiral(5.0, 200) {
            assert!((p.magnitude() - 1.0).abs() < 1e-12);
        }
        let seq = spiral_interpolate_sequence(Vec2::new(1.0, 0.0), Vec2::new(0.0, 4.0), 9);
        assert_eq!(seq.len(), 9);
        assert!((seq[0] - Vec2::new(1.0, 0.0)).magnitude() < 1e-12);
        assert!((seq[8] - Vec2::new(0.0, 4.0)).magnitude() < 1e-12);
        // Geometric radius progression.
        let ratio = seq[1].magnitude() / seq[0].magnitude();
        for w in seq.windows(2) {
            assert!((w[1].magnitude() / w[0].magnitude() - ratio).abs() < 1e-9);
        }
        let cp = cylinder_phyllotaxis(50, 0.2, GOLDEN_ANGLE, 1.5);
        for (i, p) in cp.iter().enumerate() {
            assert!((Vec2::new(p.x, p.z).magnitude() - 1.5).abs() < 1e-12);
            assert!((p.y - 0.2 * i as f64).abs() < 1e-12);
        }
    }
}
