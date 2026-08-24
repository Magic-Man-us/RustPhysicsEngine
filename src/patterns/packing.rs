//! Circle and sphere packings: Descartes/Apollonian circles, lattice
//! packings, random sequential adsorption, Doyle spirals, Ford
//! circles, Steiner chains, and the problem of Apollonius.
//!
//! Lattice generators include every circle/sphere whose *center* lies
//! in the half-open region, so exact-multiple regions give the exact
//! lattice density.

use crate::fractals::Complex;
use crate::math::{Vec2, Vec3};
use crate::monte_carlo::Rng;
use crate::spatial::primitives::{Aabb, Circle, Polygon2, Rect, Sphere};

/// Descartes circle theorem: curvatures of the two circles tangent to
/// three mutually tangent circles with curvatures k1, k2, k3:
/// k4 = k1 + k2 + k3 ± 2√(k1k2 + k2k3 + k3k1).
///
/// # Panics
/// Panics when the discriminant is negative (not a tangent triple).
#[must_use]
pub fn descartes_fourth_circle(k1: f64, k2: f64, k3: f64) -> (f64, f64) {
    let disc = k1 * k2 + k2 * k3 + k3 * k1;
    assert!(disc >= 0.0, "invalid curvature triple");
    let s = 2.0 * disc.sqrt();
    (k1 + k2 + k3 + s, k1 + k2 + k3 - s)
}

fn csqrt(z: Complex) -> Complex {
    let r = z.norm().sqrt();
    let a = z.arg() / 2.0;
    Complex::new(r * a.cos(), r * a.sin())
}

/// A circle as (curvature, curvature x center) for the complex
/// Descartes theorem.
#[derive(Clone, Copy)]
struct KCircle {
    k: f64,
    kz: Complex,
}

impl KCircle {
    fn from_circle(c: &Circle, negative: bool) -> Self {
        let k = if negative { -1.0 / c.radius } else { 1.0 / c.radius };
        Self { k, kz: Complex::new(c.center.x * k, c.center.y * k) }
    }

    fn to_circle(self) -> Circle {
        Circle {
            center: Vec2::new(self.kz.re / self.k, self.kz.im / self.k),
            radius: (1.0 / self.k).abs(),
        }
    }
}

/// Reflection identity: given a tangent quadruple, the other solution
/// touching circles 1-3 is k4' = 2(k1+k2+k3) − k4, and likewise for
/// the (curvature x center) vectors.
fn reflect(a: KCircle, b: KCircle, c: KCircle, old: KCircle) -> KCircle {
    KCircle {
        k: 2.0 * (a.k + b.k + c.k) - old.k,
        kz: (a.kz + b.kz + c.kz) * Complex::new(2.0, 0.0) - old.kz,
    }
}

type CircleKey = (i64, i64, i64);

fn circle_key(c: &KCircle) -> CircleKey {
    let circ = c.to_circle();
    (
        (circ.center.x * 1e9).round() as i64,
        (circ.center.y * 1e9).round() as i64,
        (circ.radius * 1e9).round() as i64,
    )
}

fn gasket_recurse(
    a: KCircle,
    b: KCircle,
    c: KCircle,
    d: KCircle,
    depth: usize,
    out: &mut Vec<Circle>,
    seen: &mut std::collections::HashSet<CircleKey>,
) {
    if depth == 0 {
        return;
    }
    // Reflect each circle of the quadruple through the other three.
    for (p, q, r, old) in [(b, c, d, a), (a, c, d, b), (a, b, d, c), (a, b, c, d)] {
        let new = reflect(p, q, r, old);
        if new.k > old.k.abs().max(1e-12) && new.k < 1e9 && seen.insert(circle_key(&new)) {
            out.push(new.to_circle());
            gasket_recurse(p, q, r, new, depth - 1, out, seen);
        }
    }
}

/// Apollonian gasket inside `outer`: the two seed circles have
/// curvatures `k2`, `k3` (both tangent to the outer circle and each
/// other, placed on the horizontal axis), recursively filled to
/// `depth`. Returns all circles including the outer and seeds.
///
/// # Panics
/// Panics unless the curvatures are compatible: `k2, k3 > 1/R` and
/// `1/k2 + 1/k3 = 2R - ...` — concretely both seed radii must fit:
/// `1/k2 + 1/k3 == R` is required for a tangent chain on the axis.
#[must_use]
pub fn apollonian_gasket(outer: &Circle, k2: f64, k3: f64, depth: usize) -> Vec<Circle> {
    let r = outer.radius;
    assert!(k2 > 0.0 && k3 > 0.0, "seed curvatures must be positive");
    let (r2, r3) = (1.0 / k2, 1.0 / k3);
    assert!(
        (r2 + r3 - r).abs() < 1e-9 * r,
        "seed radii must span the diameter: 1/k2 + 1/k3 == R"
    );
    let c1 = KCircle::from_circle(outer, true);
    let c2 = KCircle::from_circle(
        &Circle { center: outer.center + Vec2::new(r - r2, 0.0), radius: r2 },
        false,
    );
    let c3 = KCircle::from_circle(
        &Circle { center: outer.center - Vec2::new(r - r3, 0.0), radius: r3 },
        false,
    );
    // Fourth circle via the complex Descartes theorem:
    // z4 k4 = z1k1 + z2k2 + z3k3 ± 2 sqrt(k1k2 z1z2 + k2k3 z2z3 + k3k1 z3z1).
    let (k4, _) = descartes_fourth_circle(c1.k, c2.k, c3.k);
    let sum = c1.kz + c2.kz + c3.kz;
    let root = csqrt(c1.kz * c2.kz + c2.kz * c3.kz + c3.kz * c1.kz);
    let mut best: Option<KCircle> = None;
    for sign in [1.0, -1.0] {
        let kz = sum + root * Complex::new(2.0 * sign, 0.0);
        let cand = KCircle { k: k4, kz };
        let circ = cand.to_circle();
        // Validate tangency against the two seeds.
        let ok = (circ.center.distance_to(&c2.to_circle().center) - (circ.radius + r2)).abs()
            < 1e-6 * r
            && (circ.center.distance_to(&c3.to_circle().center) - (circ.radius + r3)).abs()
                < 1e-6 * r;
        if ok {
            best = Some(cand);
            break;
        }
    }
    let c4 = best.expect("Descartes solution must exist for a valid seed triple");
    let mut out = vec![*outer, c2.to_circle(), c3.to_circle(), c4.to_circle()];
    // The mirror twin of c4.
    let c4b = reflect(c1, c2, c3, c4);
    out.push(c4b.to_circle());
    let mut seen: std::collections::HashSet<CircleKey> =
        [&c1, &c2, &c3, &c4, &c4b].iter().map(|c| circle_key(c)).collect();
    gasket_recurse(c1, c2, c3, c4, depth, &mut out, &mut seen);
    gasket_recurse(c1, c2, c3, c4b, depth, &mut out, &mut seen);
    out
}

/// The classic integral Apollonian gasket with curvatures
/// (−1, 2, 2, 3, 3): outer unit circle, two half circles.
#[must_use]
pub fn apollonian_gasket_integral(depth: usize) -> Vec<Circle> {
    apollonian_gasket(&Circle { center: Vec2::ZERO, radius: 1.0 }, 2.0, 2.0, depth)
}

/// Greedy random packing: for each radius in order, up to `attempts`
/// random placements inside the polygon (respecting the boundary and
/// previously placed circles); radii that do not fit are skipped.
#[must_use]
pub fn circle_pack_greedy(
    region: &Polygon2,
    radii: &[f64],
    rng: &mut Rng,
    attempts: usize,
) -> Vec<Circle> {
    let bbox = region.bounding_rect();
    let size = bbox.max - bbox.min;
    let v = &region.vertices;
    let n = v.len();
    let inside = |p: Vec2| {
        let mut ins = false;
        for i in 0..n {
            let (a, b) = (v[i], v[(i + 1) % n]);
            if (a.y > p.y) != (b.y > p.y)
                && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x)
            {
                ins = !ins;
            }
        }
        ins
    };
    let boundary_dist = |p: Vec2| -> f64 {
        let mut d = f64::INFINITY;
        for i in 0..n {
            let (a, b) = (v[i], v[(i + 1) % n]);
            let e = b - a;
            let t = ((p - a).dot(&e) / e.magnitude_squared()).clamp(0.0, 1.0);
            d = d.min(p.distance_to(&(a + e * t)));
        }
        d
    };
    let mut placed: Vec<Circle> = Vec::new();
    for &r in radii {
        for _ in 0..attempts {
            let p = Vec2::new(
                bbox.min.x + rng.next_f64() * size.x,
                bbox.min.y + rng.next_f64() * size.y,
            );
            if !inside(p) || boundary_dist(p) < r {
                continue;
            }
            if placed.iter().all(|c| c.center.distance_to(&p) >= c.radius + r) {
                placed.push(Circle { center: p, radius: r });
                break;
            }
        }
    }
    placed
}

/// Hexagonal (densest) circle packing: circles of radius `r` whose
/// centers lie in the half-open region.
///
/// # Panics
/// Panics unless `r > 0`.
#[must_use]
pub fn circle_pack_hex(region: &Rect, r: f64) -> Vec<Circle> {
    assert!(r > 0.0, "radius must be positive");
    let mut out = Vec::new();
    let dy = 3.0f64.sqrt() * r;
    let mut j = 0i64;
    loop {
        let y = region.min.y + dy * (j as f64 + 0.5);
        if y >= region.max.y {
            break;
        }
        let offset = if j % 2 == 0 { r } else { 0.0 };
        let mut i = 0i64;
        loop {
            let x = region.min.x + offset + 2.0 * r * i as f64;
            if x >= region.max.x {
                break;
            }
            out.push(Circle { center: Vec2::new(x, y), radius: r });
            i += 1;
        }
        j += 1;
    }
    out
}

/// Square-lattice circle packing (centers in the half-open region).
///
/// # Panics
/// Panics unless `r > 0`.
#[must_use]
pub fn circle_pack_square(region: &Rect, r: f64) -> Vec<Circle> {
    assert!(r > 0.0, "radius must be positive");
    let mut out = Vec::new();
    let mut j = 0i64;
    loop {
        let y = region.min.y + r * (2.0 * j as f64 + 1.0);
        if y >= region.max.y {
            break;
        }
        let mut i = 0i64;
        loop {
            let x = region.min.x + r * (2.0 * i as f64 + 1.0);
            if x >= region.max.x {
                break;
            }
            out.push(Circle { center: Vec2::new(x, y), radius: r });
            i += 1;
        }
        j += 1;
    }
    out
}

/// Relaxes overlapping circles by symmetric push-apart steps, keeping
/// centers at least their radius away from the rectangle boundary.
pub fn circle_pack_relax(circles: &mut [Circle], region: &Rect, iterations: usize) {
    for _ in 0..iterations {
        for i in 0..circles.len() {
            for j in i + 1..circles.len() {
                let d = circles[j].center - circles[i].center;
                let dist = d.magnitude();
                let need = circles[i].radius + circles[j].radius;
                if dist < need && dist > 1e-12 {
                    let push = d * ((need - dist) / dist * 0.5);
                    circles[i].center = circles[i].center - push;
                    circles[j].center = circles[j].center + push;
                } else if dist <= 1e-12 {
                    circles[j].center = circles[j].center + Vec2::new(need * 0.5, 0.0);
                }
            }
        }
        for c in circles.iter_mut() {
            c.center = Vec2::new(
                c.center.x.clamp(region.min.x + c.radius, region.max.x - c.radius),
                c.center.y.clamp(region.min.y + c.radius, region.max.y - c.radius),
            );
        }
    }
}

fn lattice_pack(
    region: &Aabb,
    r: f64,
    cell: f64,
    basis: &[Vec3],
) -> Vec<Sphere> {
    let size = region.max - region.min;
    let (nx, ny, nz) = (
        (size.x / cell).ceil() as i64 + 1,
        (size.y / cell).ceil() as i64 + 1,
        (size.z / cell).ceil() as i64 + 1,
    );
    let mut out = Vec::new();
    for k in -1..nz {
        for j in -1..ny {
            for i in -1..nx {
                for b in basis {
                    let p = region.min
                        + Vec3::new(
                            (i as f64 + b.x) * cell,
                            (j as f64 + b.y) * cell,
                            (k as f64 + b.z) * cell,
                        );
                    if p.x >= region.min.x
                        && p.x < region.max.x
                        && p.y >= region.min.y
                        && p.y < region.max.y
                        && p.z >= region.min.z
                        && p.z < region.max.z
                    {
                        out.push(Sphere { center: p, radius: r });
                    }
                }
            }
        }
    }
    out
}

/// Face-centered-cubic sphere packing (density π/(3√2) ≈ 0.7405),
/// centers in the half-open box.
///
/// # Panics
/// Panics unless `r > 0`.
#[must_use]
pub fn sphere_pack_fcc(region: &Aabb, r: f64) -> Vec<Sphere> {
    assert!(r > 0.0, "radius must be positive");
    let a = 2.0 * std::f64::consts::SQRT_2 * r;
    lattice_pack(
        region,
        r,
        a,
        &[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(0.5, 0.0, 0.5),
            Vec3::new(0.0, 0.5, 0.5),
        ],
    )
}

/// Hexagonal-close-packed spheres (same density as FCC), ABAB layer
/// stacking along z; centers in the half-open box.
///
/// # Panics
/// Panics unless `r > 0`.
#[must_use]
pub fn sphere_pack_hcp(region: &Aabb, r: f64) -> Vec<Sphere> {
    assert!(r > 0.0, "radius must be positive");
    let size = region.max - region.min;
    let dx = 2.0 * r;
    let dy = 3.0f64.sqrt() * r;
    let dz = 2.0 * (6.0f64).sqrt() / 3.0 * r;
    let mut out = Vec::new();
    let nz = (size.z / dz).ceil() as i64 + 1;
    let ny = (size.y / dy).ceil() as i64 + 1;
    let nx = (size.x / dx).ceil() as i64 + 2;
    for k in 0..nz {
        let z = region.min.z + (k as f64 + 0.5) * dz;
        if z >= region.max.z {
            break;
        }
        // B layers shift by (r, r/sqrt(3)).
        let (bx, by) = if k % 2 == 0 { (0.0, 0.0) } else { (r, r / 3.0f64.sqrt()) };
        for j in -1..ny {
            let row_off = if j % 2 == 0 { 0.0 } else { r };
            let y = region.min.y + by + (j as f64 + 0.5) * dy;
            if y < region.min.y || y >= region.max.y {
                continue;
            }
            for i in -1..nx {
                let x = region.min.x + bx + row_off + i as f64 * dx;
                if x < region.min.x || x >= region.max.x {
                    continue;
                }
                out.push(Sphere { center: Vec3::new(x, y, z), radius: r });
            }
        }
    }
    out
}

/// Body-centered-cubic spheres (density π√3/8 ≈ 0.6802), centers in
/// the half-open box.
///
/// # Panics
/// Panics unless `r > 0`.
#[must_use]
pub fn sphere_pack_bcc(region: &Aabb, r: f64) -> Vec<Sphere> {
    assert!(r > 0.0, "radius must be positive");
    let a = 4.0 / 3.0f64.sqrt() * r;
    lattice_pack(region, r, a, &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5)])
}

/// Random sequential adsorption: spheres placed uniformly at random,
/// rejected on overlap, until `max_attempts` placements fail
/// (saturation density ≈ 0.38).
///
/// # Panics
/// Panics unless `r > 0`.
#[must_use]
pub fn sphere_pack_random_sequential(
    region: &Aabb,
    r: f64,
    rng: &mut Rng,
    max_attempts: usize,
) -> Vec<Sphere> {
    assert!(r > 0.0, "radius must be positive");
    let size = region.max - region.min;
    let mut out: Vec<Sphere> = Vec::new();
    let mut failures = 0usize;
    while failures < max_attempts {
        let p = region.min
            + Vec3::new(
                rng.next_f64() * size.x,
                rng.next_f64() * size.y,
                rng.next_f64() * size.z,
            );
        if p.x < region.min.x + r
            || p.x > region.max.x - r
            || p.y < region.min.y + r
            || p.y > region.max.y - r
            || p.z < region.min.z + r
            || p.z > region.max.z - r
            || out.iter().any(|s| s.center.distance_to(&p) < 2.0 * r)
        {
            failures += 1;
        } else {
            out.push(Sphere { center: p, radius: r });
            failures = 0;
        }
    }
    out
}

/// Fraction of the region area covered, counting each circle's full
/// area (consistent with the centers-in-region conventions above).
#[must_use]
pub fn packing_density_2d(circles: &[Circle], region: &Rect) -> f64 {
    let total: f64 =
        circles.iter().map(|c| std::f64::consts::PI * c.radius * c.radius).sum();
    total / region.area()
}

/// Fraction of the box volume covered, counting each sphere's full
/// volume.
#[must_use]
pub fn packing_density_3d(spheres: &[Sphere], region: &Aabb) -> f64 {
    let total: f64 = spheres
        .iter()
        .map(|s| 4.0 / 3.0 * std::f64::consts::PI * s.radius.powi(3))
        .sum();
    total / region.volume()
}

/// Doyle spiral circle packing with `p` and `q` arms: each circle is
/// tangent to its neighbors along both spiral directions. The moduli
/// of the two spiral generators are solved numerically (Newton with
/// numeric Jacobian) so all three tangency ratios agree.
///
/// # Panics
/// Panics unless `1 <= p < q` and the solver converges.
#[must_use]
pub fn doyle_spiral(p: u32, q: u32, count: usize) -> Vec<Circle> {
    assert!(p >= 1 && p < q, "requires 1 <= p < q");
    // Doyle spiral structure: generators A = e^{a + i alpha},
    // B = e^{b + i beta} subject to the (p, q) closure relation
    // A^p = B^q e^{2 pi i} (p steps of one family equal q steps of the
    // other plus a full turn), with the common tangency ratio
    // s(w) = |w - 1| / (|w| + 1) equal for the hexagonal neighbor
    // ratios A, B, and A B. Solved by grid search plus Newton; roots
    // whose orbits self-overlap (spurious sign solutions) are
    // rejected.
    let tau = 2.0 * std::f64::consts::PI;
    let s_of = |ln_r: f64, ang: f64| -> f64 {
        if ln_r.abs() > 500.0 {
            return 1.0;
        }
        let w = Complex::new(ln_r.exp() * ang.cos(), ln_r.exp() * ang.sin());
        (w - Complex::new(1.0, 0.0)).norm() / (w.norm() + 1.0)
    };
    let (pf, qf) = (f64::from(p), f64::from(q));
    let derived = |a: f64, al: f64| (pf * a / qf, (pf * al - tau) / qf);
    let f = |a: f64, al: f64| -> (f64, f64) {
        let (b, be) = derived(a, al);
        let sa = s_of(a, al);
        (sa - s_of(b, be), sa - s_of(a + b, al + be))
    };
    let is_packing = |a: f64, al: f64| -> bool {
        let (b, be) = derived(a, al);
        let s = s_of(a, al);
        if !(0.01..0.98).contains(&s) {
            return false;
        }
        for j in -12i64..=12 {
            for k in -12i64..=12 {
                if (j, k) == (0, 0) {
                    continue;
                }
                // Skip lattice vectors identified by the closure
                // relation (multiples of (p, -q)).
                if j * i64::from(q) + k * i64::from(p) == 0 && j % i64::from(p) == 0 {
                    continue;
                }
                let ln_r = j as f64 * a + k as f64 * b;
                let ang = j as f64 * al + k as f64 * be;
                if s_of(ln_r, ang) < s - 1e-9 {
                    return false;
                }
            }
        }
        true
    };
    let mut solution: Option<(f64, f64)> = None;
    'grid: for ai in 1..120 {
        let a0 = ai as f64 * 0.02;
        for ti in 1..300 {
            let al0 = ti as f64 * 0.02;
            let (f1, f2) = f(a0, al0);
            if f1.abs() + f2.abs() > 0.05 {
                continue;
            }
            let (mut a, mut al) = (a0, al0);
            for _ in 0..80 {
                let (f1, f2) = f(a, al);
                if f1.abs() + f2.abs() < 1e-13 {
                    break;
                }
                let h = 1e-8;
                let (f1a, f2a) = f(a + h, al);
                let (f1b, f2b) = f(a, al + h);
                let j11 = (f1a - f1) / h;
                let j12 = (f1b - f1) / h;
                let j21 = (f2a - f2) / h;
                let j22 = (f2b - f2) / h;
                let det = j11 * j22 - j12 * j21;
                if det.abs() < 1e-16 || a.abs() > 60.0 {
                    break;
                }
                a -= (f1 * j22 - f2 * j12) / det;
                al -= (f2 * j11 - f1 * j21) / det;
            }
            let (f1, f2) = f(a, al);
            if f1.abs() + f2.abs() < 1e-11 && a > 1e-6 && is_packing(a, al) {
                solution = Some((a, al));
                break 'grid;
            }
        }
    }
    let (a, alpha) = solution.expect("Doyle spiral solver failed to converge");
    let (b, beta) = derived(a, alpha);
    let s = s_of(a, alpha);
    let big_a = Complex::new(a.exp() * alpha.cos(), a.exp() * alpha.sin());
    let big_b = Complex::new(b.exp() * beta.cos(), b.exp() * beta.sin());
    let mut out = Vec::with_capacity(count);
    let per_arm = count.div_ceil(p as usize);
    for arm in 0..p as usize {
        // Center each arm's index range around zero for a balanced
        // spiral.
        let mut z = Complex::new(1.0, 0.0);
        for _ in 0..arm {
            z = z * big_a;
        }
        let lo = -(per_arm as i64) / 2;
        for k in 0..per_arm as i64 {
            let e = lo + k;
            let mut w = z;
            if e >= 0 {
                for _ in 0..e {
                    w = w * big_b;
                }
            } else {
                for _ in 0..(-e) {
                    w = w / big_b;
                }
            }
            out.push(Circle { center: Vec2::new(w.re, w.im), radius: s * w.norm() });
            if out.len() >= count {
                return out;
            }
        }
    }
    out
}

/// Greatest common divisor.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Ford circles: for every reduced fraction p/q with
/// `q <= max_denominator` in [0, 1], the circle tangent to the x axis
/// at p/q with radius 1/(2q²).
///
/// # Panics
/// Panics unless `max_denominator >= 1`.
#[must_use]
pub fn ford_circles(max_denominator: u32) -> Vec<Circle> {
    assert!(max_denominator >= 1, "need at least denominator 1");
    let mut out = Vec::new();
    for q in 1..=max_denominator {
        for p in 0..=q {
            if gcd(p, q) == 1 || (p == 0 && q == 1) {
                let r = 0.5 / f64::from(q).powi(2);
                out.push(Circle {
                    center: Vec2::new(f64::from(p) / f64::from(q), r),
                    radius: r,
                });
            }
        }
    }
    out
}

/// Inverts a circle not passing through the origin-based inversion
/// center `o` (unit power).
fn invert_circle(c: &Circle, o: Vec2) -> Circle {
    let d = c.center - o;
    let denom = d.magnitude_squared() - c.radius * c.radius;
    Circle {
        center: o + d * (1.0 / denom),
        radius: (c.radius / denom).abs(),
    }
}

/// Steiner chain of `n` circles in the annular region between `inner`
/// and `outer` (inner strictly inside outer). Returns `None` when the
/// pair does not admit a closed chain of exactly `n` circles
/// (Steiner's porism: feasibility depends only on the inversive
/// distance).
///
/// # Panics
/// Panics unless `n >= 3` and `inner` is strictly inside `outer`.
#[must_use]
pub fn steiner_chain(outer: &Circle, inner: &Circle, n: usize) -> Option<Vec<Circle>> {
    assert!(n >= 3, "chain needs >= 3 circles");
    let sep = outer.center.distance_to(&inner.center);
    assert!(
        sep + inner.radius < outer.radius,
        "inner circle must be strictly inside outer"
    );
    let build_concentric = |center: Vec2, big_r: f64, small_r: f64| -> Option<Vec<Circle>> {
        let ratio = (big_r - small_r) / (big_r + small_r);
        if ((std::f64::consts::PI / n as f64).sin() - ratio).abs() > 1e-6 {
            return None;
        }
        let ring = (big_r + small_r) / 2.0;
        let rho = (big_r - small_r) / 2.0;
        Some(
            (0..n)
                .map(|k| {
                    let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                    Circle {
                        center: center + Vec2::new(ring * a.cos(), ring * a.sin()),
                        radius: rho,
                    }
                })
                .collect(),
        )
    };
    if sep < 1e-12 * outer.radius {
        return build_concentric(outer.center, outer.radius, inner.radius);
    }
    // Find the inversion center (a limiting point of the coaxial
    // pencil) on the center line by bisection on the concentricity
    // defect.
    let axis = (inner.center - outer.center) * (1.0 / sep);
    let concentricity = |t: f64| -> f64 {
        let o = outer.center + axis * t;
        let io = invert_circle(outer, o);
        let ii = invert_circle(inner, o);
        (io.center - ii.center).dot(&axis)
    };
    // Limiting points lie inside the inner circle's side; scan for a
    // sign change strictly inside the annulus hole.
    let (mut lo, mut hi) = (None, None);
    let steps = 4000;
    let t0 = sep - inner.radius + 1e-9;
    let t1 = sep + inner.radius - 1e-9;
    let mut prev: Option<(f64, f64)> = None;
    for k in 0..=steps {
        let t = t0 + (t1 - t0) * k as f64 / steps as f64;
        let v = concentricity(t);
        if let Some((pt, pv)) = prev {
            if pv * v <= 0.0 && pv.is_finite() && v.is_finite() {
                lo = Some(pt);
                hi = Some(t);
                break;
            }
        }
        prev = Some((t, v));
    }
    let (mut lo, mut hi) = (lo?, hi?);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if concentricity(lo) * concentricity(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let o = outer.center + axis * ((lo + hi) / 2.0);
    let io = invert_circle(outer, o);
    let ii = invert_circle(inner, o);
    let (big, small) = if io.radius > ii.radius { (io, ii) } else { (ii, io) };
    let ring = build_concentric(big.center, big.radius, small.radius)?;
    // Invert the chain back.
    Some(ring.iter().map(|c| invert_circle(c, o)).collect())
}

/// The problem of Apollonius: circles tangent to three given circles
/// (up to 8 solutions, one per internal/external tangency sign
/// choice). Solved by reducing the tangency equations to a linear
/// system plus a quadratic in the radius.
#[must_use]
pub fn tangent_circles_to_three(c1: &Circle, c2: &Circle, c3: &Circle) -> Vec<Circle> {
    let mut out: Vec<Circle> = Vec::new();
    let cs = [c1, c2, c3];
    for signs in 0..8u32 {
        let s: Vec<f64> = (0..3)
            .map(|i| if (signs >> i) & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        // (x - xi)^2 + (y - yi)^2 = (r + si ri)^2. Subtracting the
        // first equation gives two linear equations in (x, y, r).
        let (x1, y1, r1) = (cs[0].center.x, cs[0].center.y, s[0] * cs[0].radius);
        let mut m = [[0.0f64; 4]; 2];
        for row in 0..2 {
            let c = cs[row + 1];
            let ri = s[row + 1] * c.radius;
            m[row][0] = 2.0 * (x1 - c.center.x);
            m[row][1] = 2.0 * (y1 - c.center.y);
            m[row][2] = 2.0 * (ri - r1);
            m[row][3] = (x1 * x1 + y1 * y1 - r1 * r1)
                - (c.center.x * c.center.x + c.center.y * c.center.y - ri * ri);
        }
        // Solve x = ax + bx r, y = ay + by r.
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if det.abs() < 1e-12 {
            continue;
        }
        let ax = (m[0][3] * m[1][1] - m[0][1] * m[1][3]) / det;
        let bx = -(m[0][2] * m[1][1] - m[0][1] * m[1][2]) / det;
        let ay = (m[0][0] * m[1][3] - m[0][3] * m[1][0]) / det;
        let by = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / det;
        // Substitute into the first tangency equation.
        let dx0 = ax - x1;
        let dy0 = ay - y1;
        let qa = bx * bx + by * by - 1.0;
        let qb = 2.0 * (dx0 * bx + dy0 * by - r1);
        let qc = dx0 * dx0 + dy0 * dy0 - r1 * r1;
        let roots: Vec<f64> = if qa.abs() < 1e-12 {
            if qb.abs() < 1e-12 {
                Vec::new()
            } else {
                vec![-qc / qb]
            }
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                Vec::new()
            } else {
                let sq = disc.sqrt();
                vec![(-qb + sq) / (2.0 * qa), (-qb - sq) / (2.0 * qa)]
            }
        };
        for r in roots {
            if r <= 1e-12 || !r.is_finite() {
                continue;
            }
            let cand = Circle { center: Vec2::new(ax + bx * r, ay + by * r), radius: r };
            // The squared equations admit sign-spurious roots: verify
            // genuine tangency against every input circle.
            let scale = r.max(1.0);
            let tangent_all = cs.iter().all(|c| {
                let d = cand.center.distance_to(&c.center);
                (d - (cand.radius + c.radius)).abs() < 1e-7 * scale
                    || (d - (cand.radius - c.radius).abs()).abs() < 1e-7 * scale
            });
            if !tangent_all {
                continue;
            }
            // Deduplicate across sign choices.
            if out.iter().all(|c| {
                c.center.distance_to(&cand.center) > 1e-7 || (c.radius - cand.radius).abs() > 1e-7
            }) {
                out.push(cand);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descartes_theorem() {
        // Three unit circles: k4 = 3 +- 2 sqrt(3).
        let (inner, outer) = descartes_fourth_circle(1.0, 1.0, 1.0);
        assert!((inner - (3.0 + 2.0 * 3.0f64.sqrt())).abs() < 1e-12);
        assert!((outer - (3.0 - 2.0 * 3.0f64.sqrt())).abs() < 1e-12);
    }

    fn assert_gasket_valid(circles: &[Circle], outer: &Circle) {
        // Interior circles inside the outer one, disjoint interiors.
        for (i, c) in circles.iter().enumerate().skip(1) {
            assert!(
                c.center.distance_to(&outer.center) + c.radius <= outer.radius + 1e-6,
                "circle {i} escapes the outer boundary"
            );
            for d in circles.iter().skip(i + 1) {
                let dist = c.center.distance_to(&d.center);
                assert!(
                    dist >= c.radius + d.radius - 1e-6,
                    "circles overlap: {dist} vs {} + {}",
                    c.radius,
                    d.radius
                );
            }
        }
    }

    #[test]
    fn test_apollonian_integral_gasket() {
        let circles = apollonian_gasket_integral(3);
        assert!(circles.len() > 20);
        let outer = circles[0];
        assert_gasket_valid(&circles, &outer);
        // Integral gasket: all curvatures are (near) integers.
        for c in &circles[1..] {
            let k = 1.0 / c.radius;
            assert!((k - k.round()).abs() < 1e-6, "curvature {k} not integral");
        }
        // The classic first few: 2, 2, 3, 3.
        let mut ks: Vec<i64> = circles[1..5].iter().map(|c| (1.0 / c.radius).round() as i64).collect();
        ks.sort_unstable();
        assert_eq!(ks, vec![2, 2, 3, 3]);
        // Descartes holds for the seed quadruple.
        let (k4, _) = descartes_fourth_circle(-1.0, 2.0, 2.0);
        assert!((k4 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_apollonian_asymmetric() {
        let outer = Circle { center: Vec2::new(1.0, -2.0), radius: 2.0 };
        // r2 + r3 = R: r2 = 1.25, r3 = 0.75.
        let circles = apollonian_gasket(&outer, 0.8, 1.0 / 0.75, 2);
        assert!(circles.len() > 10);
        assert_gasket_valid(&circles, &outer);
    }

    #[test]
    fn test_lattice_packings_exact_density() {
        let r = 1.0;
        // 2-D hex: region an exact multiple of the 2r x sqrt(3) r cell.
        let region = Rect {
            min: Vec2::ZERO,
            max: Vec2::new(2.0 * r * 10.0, 3.0f64.sqrt() * r * 10.0),
        };
        let hex = circle_pack_hex(&region, r);
        assert_eq!(hex.len(), 100);
        let density = packing_density_2d(&hex, &region);
        assert!(
            (density - std::f64::consts::PI / (2.0 * 3.0f64.sqrt())).abs() < 1e-6,
            "hex density {density}"
        );
        for (i, a) in hex.iter().enumerate() {
            for b in hex.iter().skip(i + 1) {
                assert!(a.center.distance_to(&b.center) >= 2.0 * r - 1e-9);
            }
        }
        // Square packing on an exact-multiple region: density pi/4.
        let sq_region = Rect { min: Vec2::ZERO, max: Vec2::new(20.0, 20.0) };
        let sq = circle_pack_square(&sq_region, r);
        assert_eq!(sq.len(), 100);
        let density = packing_density_2d(&sq, &sq_region);
        assert!((density - std::f64::consts::PI / 4.0).abs() < 1e-9);

        // FCC: box an exact multiple of the cubic cell.
        let a = 2.0 * std::f64::consts::SQRT_2 * r;
        let bx = Aabb { min: Vec3::ZERO, max: Vec3::new(4.0 * a, 4.0 * a, 4.0 * a) };
        let fcc = sphere_pack_fcc(&bx, r);
        assert_eq!(fcc.len(), 4 * 64);
        let density = packing_density_3d(&fcc, &bx);
        assert!(
            (density - std::f64::consts::PI / (3.0 * std::f64::consts::SQRT_2)).abs() < 1e-6,
            "FCC density {density}"
        );
        for (i, s) in fcc.iter().enumerate() {
            for t in fcc.iter().skip(i + 1) {
                assert!(s.center.distance_to(&t.center) >= 2.0 * r - 1e-9);
            }
        }
        // BCC: nearest neighbors touch, density pi sqrt(3)/8.
        let ab = 4.0 / 3.0f64.sqrt() * r;
        let bxb = Aabb { min: Vec3::ZERO, max: Vec3::new(4.0 * ab, 4.0 * ab, 4.0 * ab) };
        let bcc = sphere_pack_bcc(&bxb, r);
        assert_eq!(bcc.len(), 2 * 64);
        let density = packing_density_3d(&bcc, &bxb);
        assert!((density - std::f64::consts::PI * 3.0f64.sqrt() / 8.0).abs() < 1e-6);
        // HCP: no overlaps, near-FCC density on a big box.
        let hbox = Aabb { min: Vec3::ZERO, max: Vec3::new(20.0, 20.0, 20.0) };
        let hcp = sphere_pack_hcp(&hbox, r);
        for (i, s) in hcp.iter().enumerate() {
            for t in hcp.iter().skip(i + 1) {
                assert!(
                    s.center.distance_to(&t.center) >= 2.0 * r - 1e-9,
                    "HCP overlap"
                );
            }
        }
        let density = packing_density_3d(&hcp, &hbox);
        assert!(density > 0.6, "HCP fills the box ({density})");
    }

    #[test]
    fn test_greedy_relax_rsa() {
        let mut rng = Rng::new(60);
        let poly = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 6.0),
            Vec2::new(0.0, 6.0),
        ]);
        let radii: Vec<f64> = (0..40).map(|i| 1.2 - 0.02 * i as f64).collect();
        let placed = circle_pack_greedy(&poly, &radii, &mut rng, 200);
        assert!(placed.len() > 10);
        for (i, a) in placed.iter().enumerate() {
            assert!(a.center.x - a.radius >= -1e-9 && a.center.x + a.radius <= 10.0 + 1e-9);
            for b in placed.iter().skip(i + 1) {
                assert!(a.center.distance_to(&b.center) >= a.radius + b.radius - 1e-9);
            }
        }
        // Relaxation resolves overlaps.
        let region = Rect { min: Vec2::ZERO, max: Vec2::new(10.0, 10.0) };
        let mut circles: Vec<Circle> = (0..20)
            .map(|_| Circle {
                center: Vec2::new(4.0 + rng.next_f64(), 4.0 + rng.next_f64()),
                radius: 0.5,
            })
            .collect();
        circle_pack_relax(&mut circles, &region, 500);
        for (i, a) in circles.iter().enumerate() {
            for b in circles.iter().skip(i + 1) {
                assert!(
                    a.center.distance_to(&b.center) >= 1.0 - 1e-6,
                    "relaxation left an overlap"
                );
            }
        }
        // RSA: no overlaps, density in the known ballpark.
        let bx = Aabb { min: Vec3::ZERO, max: Vec3::new(12.0, 12.0, 12.0) };
        let rsa = sphere_pack_random_sequential(&bx, 1.0, &mut rng, 4000);
        for (i, s) in rsa.iter().enumerate() {
            for t in rsa.iter().skip(i + 1) {
                assert!(s.center.distance_to(&t.center) >= 2.0 - 1e-9);
            }
        }
        let density = packing_density_3d(&rsa, &bx);
        assert!(density > 0.15 && density < 0.45, "RSA density {density}");
    }

    #[test]
    fn test_doyle_spiral_tangencies() {
        let p = 8u32;
        let q = 16u32;
        let circles = doyle_spiral(p, q, 64);
        assert_eq!(circles.len(), 64);
        // Every circle's radius is proportional to its distance from
        // the origin.
        let ratio = circles[0].radius / circles[0].center.magnitude();
        for c in &circles {
            assert!((c.radius / c.center.magnitude() - ratio).abs() < 1e-9);
        }
        // Tangency: each circle is tangent to at least two others in
        // the emitted set (its spiral neighbors).
        for (i, a) in circles.iter().enumerate() {
            let mut tangent = 0;
            for (j, b) in circles.iter().enumerate() {
                if i == j {
                    continue;
                }
                let gap = a.center.distance_to(&b.center) - (a.radius + b.radius);
                if gap.abs() < 1e-6 * a.radius.max(b.radius) {
                    tangent += 1;
                }
                assert!(gap > -1e-6 * a.radius.max(b.radius), "Doyle circles overlap");
            }
            if i > 8 && i < 48 {
                assert!(tangent >= 2, "circle {i} has {tangent} tangencies");
            }
        }
    }

    #[test]
    fn test_ford_circles() {
        let circles = ford_circles(5);
        // Count of reduced fractions in [0, 1]: 1 + sum of phi(q).
        // q=1: 0/1, 1/1; q=2: 1/2; q=3: 2; q=4: 2; q=5: 4 => 11.
        assert_eq!(circles.len(), 11);
        // All tangent to the x axis and pairwise non-overlapping;
        // Farey neighbors are tangent.
        for (i, a) in circles.iter().enumerate() {
            assert!((a.center.y - a.radius).abs() < 1e-12);
            for b in circles.iter().skip(i + 1) {
                let gap = a.center.distance_to(&b.center) - (a.radius + b.radius);
                assert!(gap > -1e-9, "Ford circles overlap");
            }
        }
        // 1/2 and 1/3 are Farey neighbors: tangent.
        let get = |p: u32, q: u32| {
            circles
                .iter()
                .find(|c| {
                    (c.center.x - f64::from(p) / f64::from(q)).abs() < 1e-12
                        && (c.radius - 0.5 / f64::from(q * q)).abs() < 1e-12
                })
                .expect("circle present")
                .clone()
        };
        let (a, b) = (get(1, 2), get(1, 3));
        assert!((a.center.distance_to(&b.center) - (a.radius + b.radius)).abs() < 1e-12);
    }

    #[test]
    fn test_steiner_chain() {
        // Concentric feasible pair for n = 6: sin(pi/6) = 1/2 =>
        // (R - r)/(R + r) = 1/2 => R = 3r.
        let outer = Circle { center: Vec2::ZERO, radius: 3.0 };
        let inner = Circle { center: Vec2::ZERO, radius: 1.0 };
        let chain = steiner_chain(&outer, &inner, 6).expect("feasible chain");
        assert_eq!(chain.len(), 6);
        for (i, c) in chain.iter().enumerate() {
            // Tangent to both boundary circles.
            assert!((c.center.magnitude() + c.radius - 3.0).abs() < 1e-9);
            assert!((c.center.magnitude() - c.radius - 1.0).abs() < 1e-9);
            let next = &chain[(i + 1) % 6];
            assert!(
                (c.center.distance_to(&next.center) - (c.radius + next.radius)).abs() < 1e-9,
                "chain neighbors tangent"
            );
        }
        // Infeasible count returns None.
        assert!(steiner_chain(&outer, &inner, 7).is_none());
        // Non-concentric compatible pair: invert the feasible pair
        // through an external point (images remain a valid pair).
        let o = Vec2::new(5.0, 1.0);
        let i_outer = invert_circle(&outer, o);
        let i_inner = invert_circle(&inner, o);
        let (big, small) = if i_outer.radius > i_inner.radius {
            (i_outer, i_inner)
        } else {
            (i_inner, i_outer)
        };
        if big.center.distance_to(&small.center) + small.radius < big.radius {
            let chain = steiner_chain(&big, &small, 6).expect("inverted pair stays feasible");
            assert_eq!(chain.len(), 6);
            for (i, c) in chain.iter().enumerate() {
                assert!(
                    (c.center.distance_to(&big.center) + c.radius - big.radius).abs() < 1e-6
                );
                assert!(
                    (c.center.distance_to(&small.center) - c.radius - small.radius).abs()
                        < 1e-6
                );
                let next = &chain[(i + 1) % 6];
                assert!(
                    (c.center.distance_to(&next.center) - (c.radius + next.radius)).abs()
                        < 1e-6
                );
            }
        }
    }

    #[test]
    fn test_apollonius_problem() {
        // Three mutually tangent unit circles at the corners of an
        // equilateral triangle with side 2.
        let h = 3.0f64.sqrt();
        let c1 = Circle { center: Vec2::new(-1.0, 0.0), radius: 1.0 };
        let c2 = Circle { center: Vec2::new(1.0, 0.0), radius: 1.0 };
        let c3 = Circle { center: Vec2::new(0.0, h), radius: 1.0 };
        let sols = tangent_circles_to_three(&c1, &c2, &c3);
        assert!(!sols.is_empty());
        // The inner Soddy circle: k = 3 + 2 sqrt(3).
        let r_inner = 1.0 / (3.0 + 2.0 * h);
        let found_inner = sols.iter().any(|c| (c.radius - r_inner).abs() < 1e-6);
        assert!(found_inner, "inner Soddy circle missing");
        // Every solution is tangent to all three inputs (internally
        // or externally).
        for s in &sols {
            for c in [&c1, &c2, &c3] {
                let d = s.center.distance_to(&c.center);
                let ext = (d - (s.radius + c.radius)).abs();
                let int = (d - (s.radius - c.radius).abs()).abs();
                assert!(ext < 1e-6 || int < 1e-6, "not tangent");
            }
        }
    }
}
