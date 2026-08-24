//! Random and low-discrepancy sampling: Poisson disk (Bridson),
//! blue-noise ranking, stratified jitter, uniform samplers over
//! shapes, random polygons (Valtr), random rotations (Shoemake), and
//! Lloyd relaxation.

use crate::math::{Vec2, Vec3};
use crate::mesh::Mesh;
use crate::monte_carlo::Rng;
use crate::patterns::polygon_ops::triangulate_ear_clipping;
use crate::quaternion::Quaternion;
use crate::spatial::primitives::{
    Aabb, Circle, Obb, Polygon2, Rect, Sphere, Triangle, Triangle2,
};
use std::collections::HashMap;

/// Bridson's Poisson disk sampling in a rectangle ("Fast Poisson Disk
/// Sampling in Arbitrary Dimensions", SIGGRAPH 2007): no two samples
/// closer than `min_dist`, maximal up to `k` candidate attempts per
/// active sample.
///
/// # Panics
/// Panics unless `min_dist > 0` and `k >= 1`.
#[must_use]
pub fn poisson_disk_2d(region: &Rect, min_dist: f64, k: usize, rng: &mut Rng) -> Vec<Vec2> {
    assert!(min_dist > 0.0 && k >= 1, "requires min_dist > 0, k >= 1");
    let cell = min_dist / std::f64::consts::SQRT_2;
    let size = region.max - region.min;
    let (nx, ny) = ((size.x / cell).ceil() as i64 + 1, (size.y / cell).ceil() as i64 + 1);
    let mut grid: Vec<Option<usize>> = vec![None; (nx * ny) as usize];
    let idx = |p: Vec2| -> usize {
        let i = (((p.x - region.min.x) / cell) as i64).clamp(0, nx - 1);
        let j = (((p.y - region.min.y) / cell) as i64).clamp(0, ny - 1);
        (j * nx + i) as usize
    };
    let mut points: Vec<Vec2> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let first = Vec2::new(
        region.min.x + rng.next_f64() * size.x,
        region.min.y + rng.next_f64() * size.y,
    );
    grid[idx(first)] = Some(0);
    points.push(first);
    active.push(0);
    let fits = |p: Vec2, points: &[Vec2], grid: &[Option<usize>]| -> bool {
        if p.x < region.min.x || p.x > region.max.x || p.y < region.min.y || p.y > region.max.y {
            return false;
        }
        let ci = (((p.x - region.min.x) / cell) as i64).clamp(0, nx - 1);
        let cj = (((p.y - region.min.y) / cell) as i64).clamp(0, ny - 1);
        for dj in -2..=2i64 {
            for di in -2..=2i64 {
                let (i, j) = (ci + di, cj + dj);
                if i < 0 || i >= nx || j < 0 || j >= ny {
                    continue;
                }
                if let Some(q) = grid[(j * nx + i) as usize] {
                    if points[q].distance_to(&p) < min_dist {
                        return false;
                    }
                }
            }
        }
        true
    };
    while let Some(&seed) = active.last() {
        let base = points[seed];
        let mut placed = false;
        for _ in 0..k {
            let r = min_dist * (1.0 + rng.next_f64());
            let a = rng.next_f64() * 2.0 * std::f64::consts::PI;
            let p = base + Vec2::new(r * a.cos(), r * a.sin());
            if fits(p, &points, &grid) {
                grid[idx(p)] = Some(points.len());
                active.push(points.len());
                points.push(p);
                placed = true;
                break;
            }
        }
        if !placed {
            active.pop();
        }
    }
    points
}

/// Bridson Poisson disk sampling in a box (3-D).
///
/// # Panics
/// Panics unless `min_dist > 0` and `k >= 1`.
#[must_use]
pub fn poisson_disk_3d(region: &Aabb, min_dist: f64, k: usize, rng: &mut Rng) -> Vec<Vec3> {
    assert!(min_dist > 0.0 && k >= 1, "requires min_dist > 0, k >= 1");
    let cell = min_dist / 3.0f64.sqrt();
    let size = region.max - region.min;
    let n = [
        (size.x / cell).ceil() as i64 + 1,
        (size.y / cell).ceil() as i64 + 1,
        (size.z / cell).ceil() as i64 + 1,
    ];
    let mut grid: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let key = |p: Vec3| {
        (
            (((p.x - region.min.x) / cell) as i64).clamp(0, n[0] - 1),
            (((p.y - region.min.y) / cell) as i64).clamp(0, n[1] - 1),
            (((p.z - region.min.z) / cell) as i64).clamp(0, n[2] - 1),
        )
    };
    let mut points: Vec<Vec3> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let first = region.min
        + Vec3::new(rng.next_f64() * size.x, rng.next_f64() * size.y, rng.next_f64() * size.z);
    grid.insert(key(first), 0);
    points.push(first);
    active.push(0);
    while let Some(&seed) = active.last() {
        let base = points[seed];
        let mut placed = false;
        for _ in 0..k {
            let p = base + uniform_in_shell(min_dist, 2.0 * min_dist, rng);
            if !region.contains_point(p) {
                continue;
            }
            let (ci, cj, ck) = key(p);
            let mut ok = true;
            'check: for dk in -2..=2i64 {
                for dj in -2..=2i64 {
                    for di in -2..=2i64 {
                        if let Some(&q) = grid.get(&(ci + di, cj + dj, ck + dk)) {
                            if points[q].distance_to(&p) < min_dist {
                                ok = false;
                                break 'check;
                            }
                        }
                    }
                }
            }
            if ok {
                grid.insert((ci, cj, ck), points.len());
                active.push(points.len());
                points.push(p);
                placed = true;
                break;
            }
        }
        if !placed {
            active.pop();
        }
    }
    points
}

fn uniform_in_shell(r0: f64, r1: f64, rng: &mut Rng) -> Vec3 {
    let dir = random_unit_vector(rng);
    let r = (r0 * r0 * r0 + (r1 * r1 * r1 - r0 * r0 * r0) * rng.next_f64()).cbrt();
    dir * r
}

/// Poisson disk sampling restricted to a polygon: Bridson over the
/// bounding rectangle, samples outside the polygon rejected.
///
/// # Panics
/// Panics unless `min_dist > 0`, `k >= 1`, and the polygon has >= 3
/// vertices.
#[must_use]
pub fn poisson_disk_polygon(poly: &Polygon2, min_dist: f64, k: usize, rng: &mut Rng) -> Vec<Vec2> {
    assert!(poly.vertices.len() >= 3, "polygon needs >= 3 vertices");
    let inside = |p: Vec2| {
        let v = &poly.vertices;
        let n = v.len();
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
    poisson_disk_2d(&poly.bounding_rect(), min_dist, k, rng)
        .into_iter()
        .filter(|&p| inside(p))
        .collect()
}

/// Variable-density Poisson disk sampling: `density` maps a point to
/// its local minimum distance (larger density value = larger
/// spacing). Dart throwing against a conflict grid keyed by the
/// smallest local radius.
///
/// # Panics
/// Panics unless `k >= 1` and `density` returns positive values over
/// the region (sampled at the corners and center).
#[must_use]
pub fn poisson_disk_variable(
    region: &Rect,
    density: &dyn Fn(Vec2) -> f64,
    k: usize,
    rng: &mut Rng,
) -> Vec<Vec2> {
    assert!(k >= 1, "k must be >= 1");
    let size = region.max - region.min;
    let probes = [
        region.min,
        region.max,
        region.center(),
        Vec2::new(region.min.x, region.max.y),
        Vec2::new(region.max.x, region.min.y),
    ];
    let mut r_min = f64::INFINITY;
    for p in probes {
        let r = density(p);
        assert!(r > 0.0, "density must return positive spacing");
        r_min = r_min.min(r);
    }
    // Dart throwing with a budget proportional to the area at the
    // finest spacing.
    let attempts = (k as f64 * (size.x * size.y) / (r_min * r_min)).ceil() as usize;
    let mut points: Vec<Vec2> = Vec::new();
    for _ in 0..attempts {
        let p = Vec2::new(
            region.min.x + rng.next_f64() * size.x,
            region.min.y + rng.next_f64() * size.y,
        );
        let r = density(p);
        if points
            .iter()
            .all(|q| q.distance_to(&p) >= r.min(density(*q)))
        {
            points.push(p);
        }
    }
    points
}

/// Poisson disk sampling on a mesh surface by dart throwing over
/// area-weighted surface samples.
///
/// # Panics
/// Panics unless `min_dist > 0` and the mesh has positive area.
#[must_use]
pub fn poisson_disk_surface(mesh: &Mesh, min_dist: f64, rng: &mut Rng) -> Vec<Vec3> {
    assert!(min_dist > 0.0, "min_dist must be positive");
    let area = mesh.surface_area();
    assert!(area > 0.0, "mesh must have positive area");
    let target = (4.0 * area / (min_dist * min_dist)).ceil() as usize;
    let candidates = mesh.sample_surface(target.max(64), rng);
    let mut accepted: Vec<Vec3> = Vec::new();
    let cell = min_dist;
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let key = |p: Vec3| {
        (
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        )
    };
    'outer: for p in candidates {
        let (ci, cj, ck) = key(p);
        for di in -1..=1i64 {
            for dj in -1..=1i64 {
                for dk in -1..=1i64 {
                    if let Some(list) = grid.get(&(ci + di, cj + dj, ck + dk)) {
                        for &q in list {
                            if accepted[q].distance_to(&p) < min_dist {
                                continue 'outer;
                            }
                        }
                    }
                }
            }
        }
        grid.entry((ci, cj, ck)).or_default().push(accepted.len());
        accepted.push(p);
    }
    accepted
}

/// Blue-noise point ranking on a `w` x `h` grid by the void-and-cluster
/// method (Ulichney 1993, toroidal Gaussian energy): returns the `n`
/// best-spread grid cell centers.
///
/// # Panics
/// Panics unless `n <= w * h / 2` and the grid is nonempty.
#[must_use]
pub fn blue_noise_void_cluster(w: usize, h: usize, n: usize) -> Vec<Vec2> {
    assert!(w >= 2 && h >= 2, "grid must be at least 2x2");
    assert!(n <= w * h / 2, "n must be at most half the grid");
    let sigma = 1.0f64;
    let mut on = vec![false; w * h];
    // Deterministic scrambled start.
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut placed = 0;
    while placed < n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let c = (state >> 33) as usize % (w * h);
        if !on[c] {
            on[c] = true;
            placed += 1;
        }
    }
    let energy = |on: &[bool], i: usize| -> f64 {
        let (xi, yi) = ((i % w) as f64, (i / w) as f64);
        let mut e = 0.0;
        for (j, &b) in on.iter().enumerate() {
            if !b || j == i {
                continue;
            }
            let (xj, yj) = ((j % w) as f64, (j / w) as f64);
            let dx = (xi - xj).abs().min(w as f64 - (xi - xj).abs());
            let dy = (yi - yj).abs().min(h as f64 - (yi - yj).abs());
            e += (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
        }
        e
    };
    // Swap the tightest cluster point into the largest void until
    // stable, keeping the lowest-energy pattern seen (the greedy
    // dynamics can limit-cycle).
    let total_energy = |on: &[bool]| -> f64 {
        (0..w * h).filter(|&i| on[i]).map(|i| energy(on, i)).sum()
    };
    let mut best = on.clone();
    let mut best_e = total_energy(&on);
    for _ in 0..4 * w * h {
        let cluster = (0..w * h)
            .filter(|&i| on[i])
            .max_by(|&a, &b| energy(&on, a).total_cmp(&energy(&on, b)))
            .expect("some point is on");
        on[cluster] = false;
        let void = (0..w * h)
            .filter(|&i| !on[i])
            .min_by(|&a, &b| energy(&on, a).total_cmp(&energy(&on, b)))
            .expect("some point is off");
        on[void] = true;
        if void == cluster {
            break;
        }
        let e = total_energy(&on);
        if e < best_e {
            best_e = e;
            best = on.clone();
        }
    }
    (0..w * h)
        .filter(|&i| best[i])
        .map(|i| Vec2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5))
        .collect()
}

/// Stratified jittered samples on the unit square: one sample per
/// cell of an `nx` x `ny` grid, jittered by `jitter` in [0, 1].
///
/// # Panics
/// Panics unless `nx, ny >= 1` and `jitter` is in [0, 1].
#[must_use]
pub fn stratified_2d(nx: usize, ny: usize, jitter: f64, rng: &mut Rng) -> Vec<Vec2> {
    assert!(nx >= 1 && ny >= 1, "grid must be nonempty");
    assert!((0.0..=1.0).contains(&jitter), "jitter must be in [0, 1]");
    let mut out = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let dx = 0.5 + jitter * (rng.next_f64() - 0.5);
            let dy = 0.5 + jitter * (rng.next_f64() - 0.5);
            out.push(Vec2::new((i as f64 + dx) / nx as f64, (j as f64 + dy) / ny as f64));
        }
    }
    out
}

/// Uniform point in a 2-D triangle by the square-root warp.
#[must_use]
pub fn uniform_in_triangle(t: &Triangle2, rng: &mut Rng) -> Vec2 {
    let s = rng.next_f64().sqrt();
    let r = rng.next_f64();
    t.a * (1.0 - s) + t.b * (s * (1.0 - r)) + t.c * (s * r)
}

/// Uniform point in a 3-D triangle.
#[must_use]
pub fn uniform_in_triangle_3d(t: &Triangle, rng: &mut Rng) -> Vec3 {
    let s = rng.next_f64().sqrt();
    let r = rng.next_f64();
    t.a * (1.0 - s) + t.b * (s * (1.0 - r)) + t.c * (s * r)
}

/// Uniform point in a simple polygon: triangulate, pick a triangle by
/// area, sample it.
///
/// # Panics
/// Panics when the polygon cannot be triangulated.
#[must_use]
pub fn uniform_in_polygon(poly: &Polygon2, rng: &mut Rng) -> Vec2 {
    let tris = triangulate_ear_clipping(poly).expect("polygon must be triangulable");
    let v = &poly.vertices;
    let mut cdf = Vec::with_capacity(tris.len());
    let mut total = 0.0;
    for &[a, b, c] in &tris {
        total += (v[b] - v[a]).cross(&(v[c] - v[a])).abs() / 2.0;
        cdf.push(total);
    }
    let u = rng.next_f64() * total;
    let k = cdf.partition_point(|&a| a < u).min(tris.len() - 1);
    let [a, b, c] = tris[k];
    uniform_in_triangle(&Triangle2 { a: v[a], b: v[b], c: v[c] }, rng)
}

/// Uniform point inside a circle.
#[must_use]
pub fn uniform_in_circle(c: &Circle, rng: &mut Rng) -> Vec2 {
    let r = c.radius * rng.next_f64().sqrt();
    let a = rng.next_f64() * 2.0 * std::f64::consts::PI;
    c.center + Vec2::new(r * a.cos(), r * a.sin())
}

/// Uniform point on a circle's boundary.
#[must_use]
pub fn uniform_on_circle(c: &Circle, rng: &mut Rng) -> Vec2 {
    let a = rng.next_f64() * 2.0 * std::f64::consts::PI;
    c.center + Vec2::new(a.cos(), a.sin()) * c.radius
}

/// Uniform point inside a sphere (cube-root radial warp).
#[must_use]
pub fn uniform_in_sphere(s: &Sphere, rng: &mut Rng) -> Vec3 {
    s.center + random_unit_vector(rng) * (s.radius * rng.next_f64().cbrt())
}

/// Uniform point on a sphere's surface.
#[must_use]
pub fn uniform_on_sphere(s: &Sphere, rng: &mut Rng) -> Vec3 {
    s.center + random_unit_vector(rng) * s.radius
}

/// Uniform direction on the unit hemisphere around `n`.
///
/// # Panics
/// Panics when `n` is zero.
#[must_use]
pub fn uniform_on_hemisphere(n: Vec3, rng: &mut Rng) -> Vec3 {
    assert!(n.magnitude() > 0.0, "hemisphere axis must be nonzero");
    let d = random_unit_vector(rng);
    if d.dot(&n) < 0.0 {
        -d
    } else {
        d
    }
}

/// Cosine-weighted direction on the hemisphere around `n` (Malley's
/// method: uniform disk lifted to the sphere).
///
/// # Panics
/// Panics when `n` is zero.
#[must_use]
pub fn cosine_weighted_hemisphere(n: Vec3, rng: &mut Rng) -> Vec3 {
    assert!(n.magnitude() > 0.0, "hemisphere axis must be nonzero");
    let n = n.normalized();
    let r = rng.next_f64().sqrt();
    let a = rng.next_f64() * 2.0 * std::f64::consts::PI;
    let (x, y) = (r * a.cos(), r * a.sin());
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    // Orthonormal basis around n.
    let t = if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let u = n.cross(&t).normalized();
    let v = n.cross(&u);
    u * x + v * y + n * z
}

/// Uniform point inside an axis-aligned box.
#[must_use]
pub fn uniform_in_aabb(b: &Aabb, rng: &mut Rng) -> Vec3 {
    let e = b.max - b.min;
    b.min + Vec3::new(rng.next_f64() * e.x, rng.next_f64() * e.y, rng.next_f64() * e.z)
}

/// Uniform point inside an oriented box.
#[must_use]
pub fn uniform_in_obb(b: &Obb, rng: &mut Rng) -> Vec3 {
    let axes = b.axes();
    let h = b.half_extents;
    b.center
        + axes[0] * ((rng.next_f64() * 2.0 - 1.0) * h.x)
        + axes[1] * ((rng.next_f64() * 2.0 - 1.0) * h.y)
        + axes[2] * ((rng.next_f64() * 2.0 - 1.0) * h.z)
}

/// Uniform point on the surface of an axis-aligned box
/// (area-weighted face choice).
#[must_use]
pub fn uniform_on_aabb_surface(b: &Aabb, rng: &mut Rng) -> Vec3 {
    let e = b.max - b.min;
    let areas = [e.y * e.z, e.y * e.z, e.x * e.z, e.x * e.z, e.x * e.y, e.x * e.y];
    let total: f64 = areas.iter().sum();
    let mut u = rng.next_f64() * total;
    let mut face = 0;
    for (i, &a) in areas.iter().enumerate() {
        if u < a {
            face = i;
            break;
        }
        u -= a;
    }
    let (s, t) = (rng.next_f64(), rng.next_f64());
    match face {
        0 => Vec3::new(b.min.x, b.min.y + s * e.y, b.min.z + t * e.z),
        1 => Vec3::new(b.max.x, b.min.y + s * e.y, b.min.z + t * e.z),
        2 => Vec3::new(b.min.x + s * e.x, b.min.y, b.min.z + t * e.z),
        3 => Vec3::new(b.min.x + s * e.x, b.max.y, b.min.z + t * e.z),
        4 => Vec3::new(b.min.x + s * e.x, b.min.y + t * e.y, b.min.z),
        _ => Vec3::new(b.min.x + s * e.x, b.min.y + t * e.y, b.max.z),
    }
}

/// Uniform point in the annulus between `r_in` and `r_out`.
///
/// # Panics
/// Panics unless `0 <= r_in < r_out`.
#[must_use]
pub fn uniform_in_annulus(c: Vec2, r_in: f64, r_out: f64, rng: &mut Rng) -> Vec2 {
    assert!(r_in >= 0.0 && r_out > r_in, "requires 0 <= r_in < r_out");
    let r = (r_in * r_in + (r_out * r_out - r_in * r_in) * rng.next_f64()).sqrt();
    let a = rng.next_f64() * 2.0 * std::f64::consts::PI;
    c + Vec2::new(r * a.cos(), r * a.sin())
}

/// Uniform direction within the cone of half-angle `angle` around
/// `axis` (solid-angle uniform).
///
/// # Panics
/// Panics unless `axis` is nonzero and `angle` is in (0, π].
#[must_use]
pub fn uniform_in_cone(axis: Vec3, angle: f64, rng: &mut Rng) -> Vec3 {
    assert!(axis.magnitude() > 0.0, "cone axis must be nonzero");
    assert!(angle > 0.0 && angle <= std::f64::consts::PI, "angle in (0, pi]");
    let n = axis.normalized();
    let cos_t = 1.0 - rng.next_f64() * (1.0 - angle.cos());
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = rng.next_f64() * 2.0 * std::f64::consts::PI;
    let t = if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let u = n.cross(&t).normalized();
    let v = n.cross(&u);
    u * (sin_t * phi.cos()) + v * (sin_t * phi.sin()) + n * cos_t
}

/// Random convex polygon with `n` vertices by Valtr's algorithm
/// (uniform over convex polygons in the unit square), counterclockwise.
///
/// # Panics
/// Panics unless `n >= 3`.
#[must_use]
pub fn random_convex_polygon(n: usize, rng: &mut Rng) -> Polygon2 {
    assert!(n >= 3, "polygon needs >= 3 vertices");
    let chains = |rng: &mut Rng| -> Vec<f64> {
        let mut xs: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        xs.sort_by(f64::total_cmp);
        let (lo, hi) = (xs[0], xs[n - 1]);
        // Split interior points into two chains and emit signed spans.
        let mut deltas = Vec::with_capacity(n);
        let (mut last_top, mut last_bot) = (lo, lo);
        for &x in &xs[1..n - 1] {
            if rng.next_f64() < 0.5 {
                deltas.push(x - last_top);
                last_top = x;
            } else {
                deltas.push(last_bot - x);
                last_bot = x;
            }
        }
        deltas.push(hi - last_top);
        deltas.push(last_bot - hi);
        deltas
    };
    let dx = chains(rng);
    let mut dy = chains(rng);
    // Random pairing of x and y spans.
    for i in (1..dy.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        dy.swap(i, j);
    }
    let mut edges: Vec<Vec2> = dx.iter().zip(&dy).map(|(&x, &y)| Vec2::new(x, y)).collect();
    edges.sort_by(|a, b| a.y.atan2(a.x).total_cmp(&b.y.atan2(b.x)));
    let mut p = Vec2::ZERO;
    let mut pts = Vec::with_capacity(n);
    for e in edges {
        pts.push(p);
        p = p + e;
    }
    Polygon2::new(pts)
}

/// Random simple polygon: random points untangled by repeatedly
/// swapping crossing edges (2-opt), which strictly shortens the
/// perimeter and therefore terminates.
///
/// # Panics
/// Panics unless `n >= 3`.
#[must_use]
pub fn random_simple_polygon(n: usize, rng: &mut Rng) -> Polygon2 {
    use crate::spatial::intersect::segment_segment_2d_params;
    use crate::spatial::primitives::Segment2;
    assert!(n >= 3, "polygon needs >= 3 vertices");
    let mut pts: Vec<Vec2> =
        (0..n).map(|_| Vec2::new(rng.next_f64(), rng.next_f64())).collect();
    let crossing = |pts: &[Vec2], i: usize, j: usize| -> bool {
        let m = pts.len();
        if i == j || (i + 1) % m == j || (j + 1) % m == i {
            return false;
        }
        let s1 = Segment2 { a: pts[i], b: pts[(i + 1) % m] };
        let s2 = Segment2 { a: pts[j], b: pts[(j + 1) % m] };
        segment_segment_2d_params(&s1, &s2)
            .is_some_and(|(t, u)| (1e-12..=1.0 - 1e-12).contains(&t) && (1e-12..=1.0 - 1e-12).contains(&u))
    };
    let mut changed = true;
    let mut guard = 0usize;
    while changed && guard < n * n * n {
        changed = false;
        'search: for i in 0..n {
            for j in i + 1..n {
                if crossing(&pts, i, j) {
                    pts[i + 1..=j].reverse();
                    changed = true;
                    guard += 1;
                    break 'search;
                }
            }
        }
    }
    Polygon2::new(pts)
}

/// Uniform random rotation by Shoemake's subgroup algorithm (uniform
/// over SO(3)).
#[must_use]
pub fn random_rotation(rng: &mut Rng) -> Quaternion {
    let (u1, u2, u3) = (rng.next_f64(), rng.next_f64(), rng.next_f64());
    let tau = 2.0 * std::f64::consts::PI;
    Quaternion::new(
        (1.0 - u1).sqrt() * (tau * u2).sin(),
        (1.0 - u1).sqrt() * (tau * u2).cos(),
        u1.sqrt() * (tau * u3).sin(),
        u1.sqrt() * (tau * u3).cos(),
    )
}

/// Uniform random unit vector (normalized Gaussian triple).
#[must_use]
pub fn random_unit_vector(rng: &mut Rng) -> Vec3 {
    loop {
        let v = Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian());
        let m = v.magnitude();
        if m > 1e-12 {
            return v * (1.0 / m);
        }
    }
}

/// Lloyd relaxation toward a centroidal Voronoi arrangement: each
/// iteration moves every point to the centroid of its (grid-sampled)
/// Voronoi cell within `region`.
///
/// # Panics
/// Panics when `points` is empty.
pub fn lloyd_relaxation(points: &mut [Vec2], region: &Rect, iterations: usize) {
    assert!(!points.is_empty(), "lloyd_relaxation requires points");
    let res = 8 * (points.len() as f64).sqrt().ceil() as usize;
    let size = region.max - region.min;
    for _ in 0..iterations {
        let mut acc = vec![(Vec2::ZERO, 0.0f64); points.len()];
        for j in 0..res {
            for i in 0..res {
                let p = region.min
                    + Vec2::new(
                        size.x * (i as f64 + 0.5) / res as f64,
                        size.y * (j as f64 + 0.5) / res as f64,
                    );
                let nearest = points
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.distance_to(&p).total_cmp(&b.distance_to(&p))
                    })
                    .expect("points nonempty")
                    .0;
                acc[nearest].0 = acc[nearest].0 + p;
                acc[nearest].1 += 1.0;
            }
        }
        for (pt, (sum, count)) in points.iter_mut().zip(&acc) {
            if *count > 0.0 {
                *pt = *sum * (1.0 / count);
            }
        }
    }
}

/// Weighted stippling: `n` seed points relaxed by density-weighted
/// Lloyd iterations, so point density tracks `density`.
///
/// # Panics
/// Panics unless `n >= 1` and `density` is nonnegative where sampled.
#[must_use]
pub fn stipple(
    density: &dyn Fn(Vec2) -> f64,
    region: &Rect,
    n: usize,
    iterations: usize,
    rng: &mut Rng,
) -> Vec<Vec2> {
    assert!(n >= 1, "stipple requires n >= 1");
    let size = region.max - region.min;
    // Rejection-sample initial points from the density.
    let mut peak = 1e-12f64;
    for _ in 0..256 {
        let p = region.min
            + Vec2::new(rng.next_f64() * size.x, rng.next_f64() * size.y);
        let d = density(p);
        assert!(d >= 0.0, "density must be nonnegative");
        peak = peak.max(d);
    }
    let mut points = Vec::with_capacity(n);
    let mut guard = 0usize;
    while points.len() < n && guard < 100_000 * n {
        guard += 1;
        let p = region.min
            + Vec2::new(rng.next_f64() * size.x, rng.next_f64() * size.y);
        if rng.next_f64() * peak <= density(p) {
            points.push(p);
        }
    }
    let res = 8 * (n as f64).sqrt().ceil() as usize;
    for _ in 0..iterations {
        let mut acc = vec![(Vec2::ZERO, 0.0f64); points.len()];
        for j in 0..res {
            for i in 0..res {
                let p = region.min
                    + Vec2::new(
                        size.x * (i as f64 + 0.5) / res as f64,
                        size.y * (j as f64 + 0.5) / res as f64,
                    );
                let w = density(p);
                if w <= 0.0 {
                    continue;
                }
                let nearest = points
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.distance_to(&p).total_cmp(&b.distance_to(&p))
                    })
                    .expect("points nonempty")
                    .0;
                acc[nearest].0 = acc[nearest].0 + p * w;
                acc[nearest].1 += w;
            }
        }
        for (pt, (sum, weight)) in points.iter_mut().zip(&acc) {
            if *weight > 0.0 {
                *pt = *sum * (1.0 / weight);
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_disk_2d_separation_and_maximality() {
        let mut rng = Rng::new(50);
        let region = Rect { min: Vec2::ZERO, max: Vec2::new(10.0, 10.0) };
        let d = 0.5;
        let pts = poisson_disk_2d(&region, d, 30, &mut rng);
        assert!(pts.len() > 100, "should fill the region ({} points)", pts.len());
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                assert!(pts[i].distance_to(&pts[j]) >= d - 1e-12, "separation violated");
            }
        }
        // Maximality: no empty disk of radius 2d (probe a grid).
        for gy in 0..20 {
            for gx in 0..20 {
                let p = Vec2::new(0.25 + gx as f64 * 0.5, 0.25 + gy as f64 * 0.5);
                let near = pts.iter().map(|q| q.distance_to(&p)).fold(f64::INFINITY, f64::min);
                assert!(near < 2.0 * d, "empty disk at {p:?}");
            }
        }
    }

    #[test]
    fn test_poisson_disk_3d_and_surface() {
        let mut rng = Rng::new(51);
        let region = Aabb { min: Vec3::ZERO, max: Vec3::new(4.0, 4.0, 4.0) };
        let d = 0.8;
        let pts = poisson_disk_3d(&region, d, 30, &mut rng);
        assert!(pts.len() > 30);
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                assert!(pts[i].distance_to(&pts[j]) >= d - 1e-12);
            }
        }
        let mesh = crate::mesh::generate::icosphere(1.0, 3);
        let on = poisson_disk_surface(&mesh, 0.4, &mut rng);
        assert!(on.len() > 20);
        for i in 0..on.len() {
            assert!((on[i].magnitude() - 1.0).abs() < 0.05, "sample on the sphere");
            for j in i + 1..on.len() {
                assert!(on[i].distance_to(&on[j]) >= 0.4 - 1e-12);
            }
        }
    }

    #[test]
    fn test_poisson_variants() {
        let mut rng = Rng::new(52);
        let poly = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]);
        let pts = poisson_disk_polygon(&poly, 0.4, 30, &mut rng);
        assert!(pts.len() > 20);
        for p in &pts {
            assert!(p.x >= 0.0 && p.x <= 4.0 && p.y >= 0.0 && p.y <= 2.0);
        }
        let region = Rect { min: Vec2::ZERO, max: Vec2::new(4.0, 4.0) };
        let dens = |p: Vec2| 0.2 + 0.2 * p.x / 4.0;
        let pts = poisson_disk_variable(&region, &dens, 8, &mut rng);
        assert!(pts.len() > 50);
        for (i, a) in pts.iter().enumerate() {
            for b in pts.iter().skip(i + 1) {
                let r = dens(*a).min(dens(*b));
                assert!(a.distance_to(b) >= r - 1e-12, "variable-density separation");
            }
        }
        // Denser (more points) on the low-spacing side.
        let left = pts.iter().filter(|p| p.x < 2.0).count();
        let right = pts.len() - left;
        assert!(left > right);
    }

    #[test]
    fn test_blue_noise_and_stratified() {
        let pts = blue_noise_void_cluster(16, 16, 64);
        assert_eq!(pts.len(), 64);
        // Distinct cells, and much better average spacing than a
        // random pattern (which averages ~1 cell at this density).
        let mut min_d = f64::INFINITY;
        let mut nn_sum = 0.0;
        for i in 0..pts.len() {
            let mut nn = f64::INFINITY;
            for j in 0..pts.len() {
                if i != j {
                    nn = nn.min(pts[i].distance_to(&pts[j]));
                }
            }
            min_d = min_d.min(nn);
            nn_sum += nn;
        }
        assert!(min_d >= 1.0 - 1e-12, "cells distinct");
        let avg = nn_sum / pts.len() as f64;
        assert!(avg > 1.4, "blue noise average spacing {avg} (random is ~1.0)");
        let mut rng = Rng::new(53);
        let s = stratified_2d(8, 8, 1.0, &mut rng);
        assert_eq!(s.len(), 64);
        for (k, p) in s.iter().enumerate() {
            let (i, j) = (k % 8, k / 8);
            assert!(p.x >= i as f64 / 8.0 && p.x <= (i + 1) as f64 / 8.0);
            assert!(p.y >= j as f64 / 8.0 && p.y <= (j + 1) as f64 / 8.0);
        }
    }

    #[test]
    fn test_uniform_samplers_statistics() {
        let mut rng = Rng::new(54);
        // On-sphere: unit length, small mean.
        let s = Sphere { center: Vec3::ZERO, radius: 1.0 };
        let mut mean = Vec3::ZERO;
        let n = 20_000;
        for _ in 0..n {
            let p = uniform_on_sphere(&s, &mut rng);
            assert!((p.magnitude() - 1.0).abs() < 1e-12);
            mean = mean + p;
        }
        assert!((mean * (1.0 / n as f64)).magnitude() < 0.02);
        // In-circle: chi-squared over 100 equal-area radial bins.
        let c = Circle { center: Vec2::ZERO, radius: 1.0 };
        let mut bins = [0usize; 100];
        let n = 100_000;
        for _ in 0..n {
            let p = uniform_in_circle(&c, &mut rng);
            let r2 = p.magnitude_squared();
            bins[((r2 * 100.0) as usize).min(99)] += 1;
        }
        let expected = n as f64 / 100.0;
        let chi2: f64 =
            bins.iter().map(|&b| (b as f64 - expected).powi(2) / expected).sum();
        // 99 dof: p > 0.001 needs chi2 < ~148.
        assert!(chi2 < 148.0, "chi-squared {chi2}");
        // Hemisphere and cone respect their bounds.
        let axis = Vec3::new(0.3, 0.8, -0.5);
        for _ in 0..2000 {
            assert!(uniform_on_hemisphere(axis, &mut rng).dot(&axis) >= 0.0);
            let d = cosine_weighted_hemisphere(axis, &mut rng);
            assert!((d.magnitude() - 1.0).abs() < 1e-9);
            assert!(d.dot(&axis.normalized()) >= -1e-12);
            let d = uniform_in_cone(axis, 0.4, &mut rng);
            assert!(d.dot(&axis.normalized()) >= 0.4f64.cos() - 1e-9);
        }
        // Annulus radii in range.
        for _ in 0..2000 {
            let p = uniform_in_annulus(Vec2::ZERO, 0.5, 1.0, &mut rng);
            let r = p.magnitude();
            assert!((0.5..=1.0 + 1e-12).contains(&r));
        }
    }

    #[test]
    fn test_shape_samplers_contained() {
        let mut rng = Rng::new(55);
        let tri = Triangle2 { a: Vec2::ZERO, b: Vec2::new(2.0, 0.0), c: Vec2::new(0.0, 1.0) };
        for _ in 0..500 {
            let p = uniform_in_triangle(&tri, &mut rng);
            let (u, v, w) = tri.barycentric(p);
            assert!(u >= -1e-9 && v >= -1e-9 && w >= -1e-9);
        }
        let poly = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]);
        // Uniformity across the two rectangles of the L.
        let mut low = 0usize;
        let n = 20_000;
        for _ in 0..n {
            let p = uniform_in_polygon(&poly, &mut rng);
            assert!(p.x >= 0.0 && p.y >= 0.0);
            if p.y < 1.0 {
                low += 1;
            }
        }
        let frac = low as f64 / n as f64;
        assert!((frac - 2.0 / 3.0).abs() < 0.02, "area-weighted sampling {frac}");
        let b = Aabb { min: Vec3::ZERO, max: Vec3::new(1.0, 2.0, 3.0) };
        for _ in 0..500 {
            assert!(b.contains_point(uniform_in_aabb(&b, &mut rng)));
            let p = uniform_on_aabb_surface(&b, &mut rng);
            let on_face = p.x.abs() < 1e-12
                || (p.x - 1.0).abs() < 1e-12
                || p.y.abs() < 1e-12
                || (p.y - 2.0).abs() < 1e-12
                || p.z.abs() < 1e-12
                || (p.z - 3.0).abs() < 1e-12;
            assert!(on_face);
        }
    }

    #[test]
    fn test_uniform_in_sphere_radial_law() {
        let mut rng = Rng::new(70);
        let s = Sphere { center: Vec3::new(-1.0, 2.0, 0.5), radius: 2.5 };
        let n = 100_000;
        let mut sum_r3 = 0.0;
        let mut sum_r = 0.0;
        let mut mean = Vec3::ZERO;
        // Equal-volume shells: r³ is uniform on [0, R³], so the count
        // in each of 20 shells of equal volume is n/20.
        let mut shells = [0usize; 20];
        for _ in 0..n {
            let p = uniform_in_sphere(&s, &mut rng);
            let r = (p - s.center).magnitude();
            assert!(r <= s.radius + 1e-12, "sample outside the sphere ({r})");
            let u = r / s.radius;
            sum_r3 += u * u * u;
            sum_r += u;
            shells[((u * u * u * 20.0) as usize).min(19)] += 1;
            mean = mean + p;
        }
        // E[(r/R)³] = 1/2 and E[r/R] = 3/4 for a uniform ball.
        let m3 = sum_r3 / n as f64;
        let m1 = sum_r / n as f64;
        assert!((m3 - 0.5).abs() < 0.01, "E[(r/R)^3] = {m3}, expected 0.5");
        assert!((m1 - 0.75).abs() < 0.01, "E[r/R] = {m1}, expected 0.75");
        // Chi-squared over the equal-volume shells (19 dof: p > 0.001
        // needs chi2 < ~44).
        let expected = n as f64 / 20.0;
        let chi2: f64 =
            shells.iter().map(|&b| (b as f64 - expected).powi(2) / expected).sum();
        assert!(chi2 < 44.0, "equal-volume shell chi-squared {chi2}");
        // The sample mean converges to the centre.
        let mean = mean * (1.0 / n as f64);
        assert!(
            (mean - s.center).magnitude() < 0.05,
            "mean {mean:?} vs centre {:?}",
            s.center
        );
    }

    #[test]
    fn test_uniform_on_circle_is_on_the_rim_and_isotropic() {
        let mut rng = Rng::new(71);
        let c = Circle { center: Vec2::new(3.0, -2.0), radius: 1.5 };
        let n = 60_000;
        let bins_count = 36;
        let mut bins = vec![0usize; bins_count];
        let mut mean = Vec2::ZERO;
        for _ in 0..n {
            let p = uniform_on_circle(&c, &mut rng);
            // Exactly on the boundary, never inside.
            let r = (p - c.center).magnitude();
            assert!((r - c.radius).abs() < 1e-12, "off the rim by {}", r - c.radius);
            let mut a = (p.y - c.center.y).atan2(p.x - c.center.x);
            if a < 0.0 {
                a += 2.0 * std::f64::consts::PI;
            }
            let b = ((a / (2.0 * std::f64::consts::PI)) * bins_count as f64) as usize;
            bins[b.min(bins_count - 1)] += 1;
            mean = mean + p;
        }
        // Angles are uniform: chi-squared with 35 dof stays below ~67
        // at p = 0.001.
        let expected = n as f64 / bins_count as f64;
        let chi2: f64 =
            bins.iter().map(|&b| (b as f64 - expected).powi(2) / expected).sum();
        assert!(chi2 < 67.0, "angle chi-squared {chi2}");
        // Isotropy: the mean of the rim points is the centre.
        let mean = mean * (1.0 / n as f64);
        assert!((mean - c.center).magnitude() < 0.02, "mean {mean:?}");
        // A unit circle at the origin: every sample has magnitude 1.
        let unit = Circle { center: Vec2::ZERO, radius: 1.0 };
        for _ in 0..1000 {
            let p = uniform_on_circle(&unit, &mut rng);
            assert!((p.magnitude() - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_uniform_in_triangle_3d_stays_in_the_plane_and_is_area_uniform() {
        let mut rng = Rng::new(72);
        let t = Triangle {
            a: Vec3::new(1.0, 0.0, 0.0),
            b: Vec3::new(0.0, 2.0, 1.0),
            c: Vec3::new(-1.0, 0.5, 3.0),
        };
        let n = 40_000;
        let normal = t.normal();
        let d = normal.dot(&t.a);
        let mut centroid = Vec3::ZERO;
        // Split the triangle by the barycentric coordinate u: the
        // region u > 1/2 is a similar triangle of 1/4 the area.
        let mut u_half = 0usize;
        for _ in 0..n {
            let p = uniform_in_triangle_3d(&t, &mut rng);
            // In the triangle's plane, exactly.
            assert!(
                (normal.dot(&p) - d).abs() < 1e-12,
                "off-plane by {}",
                normal.dot(&p) - d
            );
            let (u, v, w) = t.barycentric(p);
            assert!(u >= -1e-12 && v >= -1e-12 && w >= -1e-12, "({u}, {v}, {w})");
            assert!(u <= 1.0 + 1e-12 && v <= 1.0 + 1e-12 && w <= 1.0 + 1e-12);
            assert!((u + v + w - 1.0).abs() < 1e-12, "barycentric sum");
            if u > 0.5 {
                u_half += 1;
            }
            centroid = centroid + p;
        }
        // Uniform sampling puts a quarter of the mass in that corner.
        let frac = u_half as f64 / n as f64;
        assert!((frac - 0.25).abs() < 0.01, "corner fraction {frac}, expected 0.25");
        // The sample mean is the triangle's centroid (each barycentric
        // coordinate has mean 1/3).
        let centroid = centroid * (1.0 / n as f64);
        assert!(
            (centroid - t.centroid()).magnitude() < 0.02,
            "mean {centroid:?} vs centroid {:?}",
            t.centroid()
        );
        // A degenerate (zero-area) triangle collapses onto its point.
        let deg = Triangle { a: t.a, b: t.a, c: t.a };
        for _ in 0..50 {
            assert!((uniform_in_triangle_3d(&deg, &mut rng) - t.a).magnitude() < 1e-15);
        }
    }

    #[test]
    fn test_uniform_in_obb_maps_back_into_the_unit_box() {
        let mut rng = Rng::new(73);
        let rotation = crate::linalg::rotation_axis_angle(
            Vec3::new(1.0, 2.0, -0.5).normalized(),
            0.9,
        );
        let b = Obb {
            center: Vec3::new(2.0, -1.0, 0.5),
            half_extents: Vec3::new(1.0, 3.0, 0.25),
            rotation,
        };
        let axes = b.axes();
        let h = [b.half_extents.x, b.half_extents.y, b.half_extents.z];
        let n = 40_000;
        // Octant counts: a uniform box puts n/8 in each.
        let mut octants = [0usize; 8];
        let mut mean = Vec3::ZERO;
        let mut extremes = [f64::NEG_INFINITY; 3];
        for _ in 0..n {
            let p = uniform_in_obb(&b, &mut rng);
            let rel = p - b.center;
            let mut octant = 0usize;
            for k in 0..3 {
                // Project onto the local axes: |coordinate| <= half
                // extent is exactly "inside the box".
                let s = rel.dot(&axes[k]);
                assert!(
                    s.abs() <= h[k] + 1e-12,
                    "axis {k} coordinate {s} outside +-{}",
                    h[k]
                );
                extremes[k] = extremes[k].max(s.abs() / h[k]);
                if s > 0.0 {
                    octant |= 1 << k;
                }
            }
            octants[octant] += 1;
            mean = mean + p;
        }
        let expected = n as f64 / 8.0;
        let chi2: f64 =
            octants.iter().map(|&o| (o as f64 - expected).powi(2) / expected).sum();
        // 7 dof: p > 0.001 needs chi2 < ~24.3.
        assert!(chi2 < 24.3, "octant chi-squared {chi2}");
        // Samples reach close to every face (the box is filled).
        for (k, &e) in extremes.iter().enumerate() {
            assert!(e > 0.99, "axis {k} only reached {e} of the half extent");
        }
        let mean = mean * (1.0 / n as f64);
        assert!(
            (mean - b.center).magnitude() < 0.05,
            "mean {mean:?} vs centre {:?}",
            b.center
        );
        // With the identity rotation the OBB is an AABB and the samples
        // must satisfy the AABB containment test.
        let axis_aligned = Obb {
            center: Vec3::new(0.0, 0.0, 0.0),
            half_extents: Vec3::new(1.0, 2.0, 3.0),
            rotation: crate::linalg::Mat3::identity(),
        };
        let aabb = Aabb {
            min: Vec3::new(-1.0, -2.0, -3.0),
            max: Vec3::new(1.0, 2.0, 3.0),
        };
        for _ in 0..2000 {
            let p = uniform_in_obb(&axis_aligned, &mut rng);
            assert!(aabb.contains_point(p), "{p:?} outside the equivalent AABB");
        }
    }

    #[test]
    fn test_random_polygons_and_rotations() {
        let mut rng = Rng::new(56);
        for n in [3usize, 5, 8, 20] {
            let p = random_convex_polygon(n, &mut rng);
            assert_eq!(p.vertices.len(), n);
            assert!(p.is_convex(), "Valtr output must be convex");
            assert!(p.is_ccw());
        }
        for n in [4usize, 8, 12] {
            let p = random_simple_polygon(n, &mut rng);
            assert_eq!(p.vertices.len(), n);
            assert!(p.is_simple(), "2-opt untangling must yield a simple polygon");
        }
        // Shoemake rotations: unit quaternions, isotropic axes.
        let mut mean = Vec3::ZERO;
        let n = 5000;
        for _ in 0..n {
            let q = random_rotation(&mut rng);
            let norm = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
            mean = mean + q.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        }
        assert!((mean * (1.0 / n as f64)).magnitude() < 0.05, "rotations isotropic");
    }

    #[test]
    fn test_lloyd_and_stipple() {
        let mut rng = Rng::new(57);
        let region = Rect { min: Vec2::ZERO, max: Vec2::new(1.0, 1.0) };
        let mut pts: Vec<Vec2> =
            (0..16).map(|_| Vec2::new(rng.next_f64() * 0.2, rng.next_f64() * 0.2)).collect();
        let spread = |pts: &[Vec2]| {
            let mut m = f64::INFINITY;
            for i in 0..pts.len() {
                for j in i + 1..pts.len() {
                    m = m.min(pts[i].distance_to(&pts[j]));
                }
            }
            m
        };
        let before = spread(&pts);
        lloyd_relaxation(&mut pts, &region, 100);
        let after = spread(&pts);
        assert!(after > before, "Lloyd spreads clustered points");
        assert!(after > 0.15, "near-CVT spacing for 16 points, got {after}");

        let dens = |p: Vec2| if p.x < 0.5 { 4.0 } else { 1.0 };
        let st = stipple(&dens, &region, 60, 10, &mut rng);
        assert_eq!(st.len(), 60);
        let left = st.iter().filter(|p| p.x < 0.5).count();
        assert!(left > 35, "stippling follows density ({left} of 60 on the dense side)");
    }
}
