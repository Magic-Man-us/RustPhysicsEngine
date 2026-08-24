//! 2-D polygon algorithms: triangulation, simplification, offsetting,
//! Minkowski sums, boolean operations, clipping, decomposition, hulls,
//! skeletons, enclosing/inscribed shapes, and fill patterns.

use crate::error::GeomError;
use crate::math::{Vec2, Vec3};
use crate::mesh::Mesh;
use crate::spatial::primitives::{Circle, Polygon2, Rect, Segment2};
use std::collections::HashMap;

fn cross3(o: Vec2, a: Vec2, b: Vec2) -> f64 {
    (a - o).cross(&(b - o))
}

/// Even-odd point-in-polygon test on a raw vertex slice.
fn point_in_poly(p: Vec2, poly: &[Vec2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a.y > p.y) != (b.y > p.y) {
            let x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if p.x < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// Ear clipping on a counterclockwise index loop; returns triangles.
fn ear_clip_indices(pts: &[Vec2], order: &[usize]) -> Option<Vec<[usize; 3]>> {
    let mut remaining: Vec<usize> = order.to_vec();
    let mut tris = Vec::new();
    let mut stuck = 0usize;
    while remaining.len() > 3 {
        let m = remaining.len();
        let mut clipped = false;
        for k in 0..m {
            let (ip, ic, inx) = (remaining[(k + m - 1) % m], remaining[k], remaining[(k + 1) % m]);
            let (p, c, nx) = (pts[ip], pts[ic], pts[inx]);
            if cross3(p, c, nx) <= 0.0 {
                continue;
            }
            let blocked = remaining.iter().any(|&j| {
                if j == ip || j == ic || j == inx {
                    return false;
                }
                let q = pts[j];
                // Coincident duplicates (hole-bridge vertices) do not
                // block an ear.
                if q.distance_to(&p) < 1e-12
                    || q.distance_to(&c) < 1e-12
                    || q.distance_to(&nx) < 1e-12
                {
                    return false;
                }
                cross3(p, c, q) >= 0.0 && cross3(c, nx, q) >= 0.0 && cross3(nx, p, q) >= 0.0
            });
            if !blocked {
                tris.push([ip, ic, inx]);
                remaining.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Tolerate collinear runs: clip any strictly convex corner.
            let m = remaining.len();
            let k = (0..m).find(|&k| {
                cross3(
                    pts[remaining[(k + m - 1) % m]],
                    pts[remaining[k]],
                    pts[remaining[(k + 1) % m]],
                ) > 0.0
            })?;
            tris.push([
                remaining[(k + m - 1) % m],
                remaining[k],
                remaining[(k + 1) % m],
            ]);
            remaining.remove(k);
            stuck += 1;
            if stuck > order.len() {
                return None;
            }
        }
    }
    tris.push([remaining[0], remaining[1], remaining[2]]);
    Some(tris)
}

/// Ear-clipping triangulation of a simple polygon. Indices refer to
/// the polygon's own vertex order (clockwise input is handled).
///
/// # Errors
/// [`GeomError::InvalidArgument`] for fewer than 3 vertices;
/// [`GeomError::Degenerate`] for zero area or self-intersecting input.
pub fn triangulate_ear_clipping(poly: &Polygon2) -> Result<Vec<[usize; 3]>, GeomError> {
    if poly.vertices.len() < 3 {
        return Err(GeomError::InvalidArgument("polygon needs >= 3 vertices"));
    }
    if poly.area() == 0.0 {
        return Err(GeomError::Degenerate("polygon has zero area"));
    }
    if !poly.is_simple() {
        return Err(GeomError::Degenerate("polygon self-intersects"));
    }
    let n = poly.vertices.len();
    let order: Vec<usize> =
        if poly.is_ccw() { (0..n).collect() } else { (0..n).rev().collect() };
    ear_clip_indices(&poly.vertices, &order)
        .ok_or(GeomError::Degenerate("ear clipping failed"))
}

/// Triangulates a polygon with holes by bridging each hole to the
/// outer boundary (rightmost-vertex visibility bridge) and ear
/// clipping the result. Returns the combined vertex list (outer, then
/// holes in bridging order, with two duplicated bridge vertices per
/// hole) and triangles into it.
///
/// # Errors
/// Propagates the failure modes of [`triangulate_ear_clipping`];
/// holes must be strictly inside the outer polygon and disjoint.
pub fn triangulate_with_holes(
    outer: &Polygon2,
    holes: &[Polygon2],
) -> Result<(Vec<Vec2>, Vec<[usize; 3]>), GeomError> {
    if outer.vertices.len() < 3 {
        return Err(GeomError::InvalidArgument("polygon needs >= 3 vertices"));
    }
    // Outer counterclockwise; holes clockwise.
    let mut boundary: Vec<Vec2> = if outer.is_ccw() {
        outer.vertices.clone()
    } else {
        outer.vertices.iter().rev().copied().collect()
    };
    let mut ordered: Vec<Vec<Vec2>> = holes
        .iter()
        .map(|h| {
            if h.is_ccw() {
                h.vertices.iter().rev().copied().collect()
            } else {
                h.vertices.clone()
            }
        })
        .collect();
    // Bridge holes from the rightmost first (standard order).
    ordered.sort_by(|a, b| {
        let mx = |h: &Vec<Vec2>| h.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        mx(b).total_cmp(&mx(a))
    });
    for hole in ordered {
        // Rightmost hole vertex.
        let (mi, &m) = hole
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.x.total_cmp(&b.x))
            .expect("hole nonempty");
        // Closest visible boundary vertex by ray casting toward +x.
        let nb = boundary.len();
        let mut best: Option<(f64, usize)> = None;
        for i in 0..nb {
            let (a, b) = (boundary[i], boundary[(i + 1) % nb]);
            if (a.y > m.y) == (b.y > m.y) {
                continue;
            }
            let x = a.x + (m.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if x >= m.x - 1e-12 {
                // Bridge to the edge endpoint on the ray's right side
                // with the larger x (visible corner heuristic).
                let cand = if a.x > b.x { i } else { (i + 1) % nb };
                if best.is_none_or(|(bx, _)| x < bx) {
                    best = Some((x, cand));
                }
            }
        }
        let (_, bi) =
            best.ok_or(GeomError::InvalidArgument("hole is not inside the outer polygon"))?;
        // Refine: among boundary vertices inside triangle (m, i, p)
        // pick the one with the smallest angle to +x (Eberly).
        let isect = Vec2::new(best.expect("checked").0, m.y);
        let mut bridge = bi;
        let mut best_angle = f64::INFINITY;
        for (j, &q) in boundary.iter().enumerate() {
            if q.x < m.x {
                continue;
            }
            let inside_tri = cross3(m, isect, q) >= -1e-12
                && cross3(isect, boundary[bridge], q) >= -1e-12
                && cross3(boundary[bridge], m, q) >= -1e-12;
            if inside_tri {
                let angle = (q.y - m.y).atan2(q.x - m.x).abs();
                if angle < best_angle {
                    best_angle = angle;
                    bridge = j;
                }
            }
        }
        // Splice: boundary[..=bridge], hole[mi..], hole[..=mi], bridge.
        let mut merged = Vec::with_capacity(boundary.len() + hole.len() + 2);
        merged.extend_from_slice(&boundary[..=bridge]);
        for k in 0..=hole.len() {
            merged.push(hole[(mi + k) % hole.len()]);
        }
        merged.push(boundary[bridge]);
        merged.extend_from_slice(&boundary[bridge + 1..]);
        boundary = merged;
    }
    let order: Vec<usize> = (0..boundary.len()).collect();
    let tris = ear_clip_indices(&boundary, &order)
        .ok_or(GeomError::Degenerate("ear clipping failed"))?;
    Ok((boundary, tris))
}

/// Ramer-Douglas-Peucker polyline simplification: keeps points whose
/// deviation exceeds `epsilon`. Endpoints are always kept.
///
/// # Panics
/// Panics unless `epsilon >= 0`.
#[must_use]
pub fn simplify_douglas_peucker(pts: &[Vec2], epsilon: f64) -> Vec<Vec2> {
    assert!(epsilon >= 0.0, "epsilon must be nonnegative");
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    fn recurse(pts: &[Vec2], eps: f64, out: &mut Vec<Vec2>) {
        let (a, b) = (pts[0], pts[pts.len() - 1]);
        let dir = b - a;
        let len = dir.magnitude();
        let mut worst = (0.0f64, 0usize);
        for (i, &p) in pts.iter().enumerate().skip(1).take(pts.len() - 2) {
            let d = if len == 0.0 {
                p.distance_to(&a)
            } else {
                (dir.cross(&(p - a)) / len).abs()
            };
            if d > worst.0 {
                worst = (d, i);
            }
        }
        if worst.0 > eps {
            recurse(&pts[..=worst.1], eps, out);
            out.pop(); // avoid duplicating the split point
            recurse(&pts[worst.1..], eps, out);
        } else {
            out.push(a);
            out.push(b);
        }
    }
    let mut out = Vec::new();
    recurse(pts, epsilon, &mut out);
    out
}

/// Visvalingam-Whyatt simplification: repeatedly removes the interior
/// point spanning the smallest triangle until every remaining point
/// spans at least `min_area`.
///
/// # Panics
/// Panics unless `min_area >= 0`.
#[must_use]
pub fn simplify_visvalingam(pts: &[Vec2], min_area: f64) -> Vec<Vec2> {
    assert!(min_area >= 0.0, "min_area must be nonnegative");
    let mut kept: Vec<Vec2> = pts.to_vec();
    while kept.len() > 2 {
        let mut smallest = (f64::INFINITY, 0usize);
        for i in 1..kept.len() - 1 {
            let a = cross3(kept[i - 1], kept[i], kept[i + 1]).abs() / 2.0;
            if a < smallest.0 {
                smallest = (a, i);
            }
        }
        if smallest.0 >= min_area {
            break;
        }
        kept.remove(smallest.1);
    }
    kept
}

/// How offset corners are joined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JoinStyle {
    /// Sharp corner, clamped to the given miter limit (multiples of
    /// the offset distance; beveled beyond it).
    Miter(f64),
    /// Circular arc with the given number of segments.
    Round(usize),
    /// Straight cut between the two offset edges.
    Bevel,
}

/// Splits a possibly self-intersecting ring into simple loops and
/// keeps counterclockwise ones with meaningful area.
fn clean_ring(ring: Vec<Vec2>, min_area: f64) -> Vec<Polygon2> {
    use crate::spatial::intersect::segment_segment_2d_params;
    let n = ring.len();
    if n < 3 {
        return Vec::new();
    }
    for i in 0..n {
        for j in i + 1..n {
            // Skip adjacent edges (shared endpoint).
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let s1 = Segment2 { a: ring[i], b: ring[(i + 1) % n] };
            let s2 = Segment2 { a: ring[j], b: ring[(j + 1) % n] };
            if let Some((t, _)) = segment_segment_2d_params(&s1, &s2) {
                if !(1e-12..=1.0 - 1e-12).contains(&t) {
                    continue;
                }
                let x = s1.a.lerp(&s1.b, t);
                // Loop 1: ring[..=i], x, ring[j+1..]; loop 2: x,
                // ring[i+1..=j].
                let mut l1: Vec<Vec2> = ring[..=i].to_vec();
                l1.push(x);
                l1.extend_from_slice(&ring[j + 1..]);
                let mut l2: Vec<Vec2> = vec![x];
                l2.extend_from_slice(&ring[i + 1..=j]);
                let mut out = clean_ring(l1, min_area);
                out.extend(clean_ring(l2, min_area));
                return out;
            }
        }
    }
    let poly = Polygon2::new(ring);
    if poly.area_signed() > min_area {
        vec![poly]
    } else {
        Vec::new()
    }
}

/// Offsets a simple polygon outward (`distance > 0`) or inward
/// (`distance < 0`), joining corners by `join`. Self-intersections of
/// the raw offset ring (spikes collapsing under inset, etc.) are
/// resolved by splitting into simple loops and keeping
/// counterclockwise ones; an inset larger than the inradius returns
/// an empty vector. Input orientation does not matter; outputs are
/// counterclockwise.
///
/// # Panics
/// Panics unless the polygon has >= 3 vertices and `distance != 0`.
#[must_use]
pub fn offset_polygon(poly: &Polygon2, distance: f64, join: JoinStyle) -> Vec<Polygon2> {
    assert!(poly.vertices.len() >= 3, "offset_polygon requires >= 3 vertices");
    assert!(distance != 0.0, "offset distance must be nonzero");
    let pts: Vec<Vec2> = if poly.is_ccw() {
        poly.vertices.clone()
    } else {
        poly.vertices.iter().rev().copied().collect()
    };
    let n = pts.len();
    // Outward normal of each counterclockwise edge is its right-hand
    // perpendicular.
    let normal: Vec<Vec2> = (0..n)
        .map(|i| {
            let e = pts[(i + 1) % n] - pts[i];
            Vec2::new(e.y, -e.x).normalized()
        })
        .collect();
    let mut ring: Vec<Vec2> = Vec::new();
    for i in 0..n {
        let prev = (i + n - 1) % n;
        let p = pts[i];
        let (n0, n1) = (normal[prev], normal[i]);
        let (a, b) = (p + n0 * distance, p + n1 * distance);
        let turn = n0.cross(&n1); // > 0 at convex corners (counterclockwise)
        let opens = turn * distance > 0.0;
        if !opens {
            // The offset edges overlap: their line intersection is the
            // natural corner.
            let denom = n0.cross(&n1);
            if denom.abs() > 1e-12 {
                // Solve p + n0 d + s e0 = p + n1 d + t e1 for the two
                // offset lines (directions are the edges).
                let e0 = pts[i] - pts[prev];
                let e1 = pts[(i + 1) % n] - pts[i];
                let diff = b - a;
                let d = e0.cross(&e1);
                if d.abs() > 1e-12 {
                    let s = diff.cross(&e1) / d;
                    ring.push(a + e0 * s);
                    continue;
                }
            }
            ring.push((a + b) * 0.5);
            continue;
        }
        match join {
            JoinStyle::Bevel => {
                ring.push(a);
                ring.push(b);
            }
            JoinStyle::Miter(limit) => {
                let bis = (n0 + n1).normalized();
                let cos_half = bis.dot(&n1).max(1e-9);
                let miter_len = distance.abs() / cos_half;
                if miter_len <= limit.max(1.0) * distance.abs() {
                    ring.push(p + bis * (miter_len * distance.signum()));
                } else {
                    ring.push(a);
                    ring.push(b);
                }
            }
            JoinStyle::Round(segments) => {
                let segments = segments.max(1);
                let tau = 2.0 * std::f64::consts::PI;
                let a0 = n0.y.atan2(n0.x);
                let mut a1 = n1.y.atan2(n1.x);
                // Sweep the offset normal from n0 to n1 the way the
                // corner opens: counterclockwise when growing,
                // clockwise when shrinking.
                if distance > 0.0 {
                    while a1 < a0 {
                        a1 += tau;
                    }
                } else {
                    while a1 > a0 {
                        a1 -= tau;
                    }
                }
                for k in 0..=segments {
                    let t = a0 + (a1 - a0) * k as f64 / segments as f64;
                    ring.push(p + Vec2::new(t.cos(), t.sin()) * distance);
                }
            }
        }
    }
    let scale = ring.iter().map(|p| p.magnitude()).fold(1.0, f64::max);
    let loops = clean_ring(ring, 1e-12 * scale * scale);
    // Validity filter: every vertex of a true offset loop lies at
    // distance >= |d| from the input boundary (spike-collapse and
    // inverted loops fail this).
    let dist_to_boundary = |p: Vec2| -> f64 {
        let mut d = f64::INFINITY;
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let e = b - a;
            let t = ((p - a).dot(&e) / e.magnitude_squared()).clamp(0.0, 1.0);
            d = d.min(p.distance_to(&(a + e * t)));
        }
        d
    };
    let slack = distance.abs() * 1e-6 + 1e-9 * scale;
    loops
        .into_iter()
        .filter(|l| l.vertices.iter().all(|&v| dist_to_boundary(v) >= distance.abs() - slack))
        .collect()
}

/// Minkowski sum of two convex polygons by the edge-merge
/// (convolution) construction; output is convex and counterclockwise.
///
/// # Panics
/// Panics unless both polygons are convex with >= 3 vertices.
#[must_use]
pub fn minkowski_sum_convex(a: &Polygon2, b: &Polygon2) -> Polygon2 {
    assert!(a.vertices.len() >= 3 && b.vertices.len() >= 3, "polygons need >= 3 vertices");
    assert!(a.is_convex() && b.is_convex(), "minkowski_sum_convex requires convex input");
    let norm = |p: &Polygon2| -> Vec<Vec2> {
        let v: Vec<Vec2> = if p.is_ccw() {
            p.vertices.clone()
        } else {
            p.vertices.iter().rev().copied().collect()
        };
        // Rotate so the bottom-most (then left-most) vertex is first.
        let start = v
            .iter()
            .enumerate()
            .min_by(|(_, p), (_, q)| (p.y, p.x).partial_cmp(&(q.y, q.x)).expect("finite"))
            .expect("nonempty")
            .0;
        (0..v.len()).map(|i| v[(start + i) % v.len()]).collect()
    };
    let (pa, pb) = (norm(a), norm(b));
    let (na, nb) = (pa.len(), pb.len());
    let mut out = Vec::with_capacity(na + nb);
    let (mut i, mut j) = (0usize, 0usize);
    while i < na || j < nb {
        out.push(pa[i % na] + pb[j % nb]);
        let ea = pa[(i + 1) % na] - pa[i % na];
        let eb = pb[(j + 1) % nb] - pb[j % nb];
        let c = ea.cross(&eb);
        if i >= na || (j < nb && c < 0.0) {
            j += 1;
        } else if j >= nb || c > 0.0 {
            i += 1;
        } else {
            // Parallel edges advance together.
            i += 1;
            j += 1;
        }
    }
    Polygon2::new(out)
}

/// Minkowski sum of two simple polygons via convex decomposition:
/// pairwise convex sums, unioned together.
#[must_use]
pub fn minkowski_sum(a: &Polygon2, b: &Polygon2) -> Vec<Polygon2> {
    let da = convex_decomposition(a);
    let db = convex_decomposition(b);
    let mut acc: Vec<Polygon2> = Vec::new();
    for pa in &da {
        for pb in &db {
            // Merge the new piece into any accumulated piece it
            // overlaps, repeating until it is disjoint from the rest.
            let mut cur = minkowski_sum_convex(pa, pb);
            loop {
                let mut merged = false;
                for i in 0..acc.len() {
                    let u = boolean_union(&acc[i], &cur);
                    if u.len() == 1 {
                        cur = u.into_iter().next().expect("single loop");
                        acc.swap_remove(i);
                        merged = true;
                        break;
                    }
                }
                if !merged {
                    break;
                }
            }
            acc.push(cur);
        }
    }
    acc
}

// ---------------------------------------------------------------------
// Greiner-Hormann boolean operations.
// ---------------------------------------------------------------------

#[derive(Clone, Copy)]
struct GhNode {
    p: Vec2,
    next: usize,
    prev: usize,
    neighbor: usize,
    entry: bool,
    processed: bool,
    is_x: bool,
}

/// One crossing between edge `ia` of A (at parameter `ta`) and edge
/// `ib` of B (at `tb`), at point `p`.
struct GhCrossing {
    ia: usize,
    ta: f64,
    ib: usize,
    tb: f64,
    p: Vec2,
}

/// Collects proper interior edge intersections; `Err(())` signals a
/// degenerate configuration (endpoint touch or collinear overlap).
#[allow(clippy::result_unit_err)]
fn gh_intersections(pa: &[Vec2], pb: &[Vec2]) -> Result<Vec<GhCrossing>, ()> {
    let mut out = Vec::new();
    let eps = 1e-11;
    for (i, sa) in pa.iter().enumerate() {
        let ea = pa[(i + 1) % pa.len()] - *sa;
        for (j, sb) in pb.iter().enumerate() {
            let eb = pb[(j + 1) % pb.len()] - *sb;
            let denom = ea.cross(&eb);
            let diff = *sb - *sa;
            if denom.abs() < 1e-14 * (ea.magnitude() * eb.magnitude()).max(1e-30) {
                // Parallel: overlap is degenerate.
                if diff.cross(&ea).abs() < eps * ea.magnitude().max(1.0) {
                    let t0 = diff.dot(&ea) / ea.magnitude_squared();
                    let t1 = t0 + eb.dot(&ea) / ea.magnitude_squared();
                    if t0.max(t1) > eps && t0.min(t1) < 1.0 - eps {
                        return Err(());
                    }
                }
                continue;
            }
            let t = diff.cross(&eb) / denom;
            let u = diff.cross(&ea) / denom;
            if (-eps..=1.0 + eps).contains(&t) && (-eps..=1.0 + eps).contains(&u) {
                if t < eps || t > 1.0 - eps || u < eps || u > 1.0 - eps {
                    return Err(()); // endpoint degeneracy
                }
                out.push(GhCrossing { ia: i, ta: t, ib: j, tb: u, p: *sa + ea * t });
            }
        }
    }
    Ok(out)
}

fn gh_build(pa: &[Vec2], pb: &[Vec2]) -> Option<(Vec<GhNode>, usize)> {
    let xs = gh_intersections(pa, pb).ok()?;
    let node = |p: Vec2| GhNode {
        p,
        next: 0,
        prev: 0,
        neighbor: usize::MAX,
        entry: false,
        processed: false,
        is_x: false,
    };
    let mut arena: Vec<GhNode> = Vec::new();
    // Ring A with intersections inserted in order.
    let mut a_nodes_of_x: HashMap<usize, usize> = HashMap::new();
    let mut order: Vec<usize> = Vec::new();
    for (i, &p) in pa.iter().enumerate() {
        order.push(arena.len());
        arena.push(node(p));
        let mut on_edge: Vec<(f64, usize)> = xs
            .iter()
            .enumerate()
            .filter(|(_, x)| x.ia == i)
            .map(|(xi, x)| (x.ta, xi))
            .collect();
        on_edge.sort_by(|x, y| x.0.total_cmp(&y.0));
        for (_, xi) in on_edge {
            a_nodes_of_x.insert(xi, arena.len());
            order.push(arena.len());
            let mut v = node(xs[xi].p);
            v.is_x = true;
            arena.push(v);
        }
    }
    let na = order.len();
    for (k, &id) in order.iter().enumerate() {
        arena[id].next = order[(k + 1) % na];
        arena[id].prev = order[(k + na - 1) % na];
    }
    let a_head = order[0];
    // Ring B.
    let b_start = arena.len();
    let mut order_b: Vec<usize> = Vec::new();
    for (j, &p) in pb.iter().enumerate() {
        order_b.push(arena.len());
        arena.push(node(p));
        let mut on_edge: Vec<(f64, usize)> = xs
            .iter()
            .enumerate()
            .filter(|(_, x)| x.ib == j)
            .map(|(xi, x)| (x.tb, xi))
            .collect();
        on_edge.sort_by(|x, y| x.0.total_cmp(&y.0));
        for (_, xi) in on_edge {
            let b_id = arena.len();
            let mut v = node(xs[xi].p);
            v.is_x = true;
            v.neighbor = a_nodes_of_x[&xi];
            order_b.push(b_id);
            arena.push(v);
            let a_id = a_nodes_of_x[&xi];
            arena[a_id].neighbor = b_id;
        }
    }
    let nb = order_b.len();
    for (k, &id) in order_b.iter().enumerate() {
        arena[id].next = order_b[(k + 1) % nb];
        arena[id].prev = order_b[(k + nb - 1) % nb];
    }
    let b_head = order_b[0];
    // Entry/exit marking (even-odd).
    for (head, other) in [(a_head, pb), (b_head, pa)] {
        let mut status = point_in_poly(arena[head].p, other);
        let mut cur = head;
        loop {
            if arena[cur].is_x {
                arena[cur].entry = !status;
                status = !status;
            }
            cur = arena[cur].next;
            if cur == head {
                break;
            }
        }
    }
    Some((arena, b_start))
}

/// A point strictly inside a simple polygon (any orientation): the
/// classic convex-corner/diagonal construction.
fn interior_point(poly: &Polygon2) -> Vec2 {
    let v = &poly.vertices;
    let n = v.len();
    let (vi, &p) = v
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a.x, a.y).partial_cmp(&(b.x, b.y)).expect("finite"))
        .expect("nonempty");
    let a = v[(vi + n - 1) % n];
    let b = v[(vi + 1) % n];
    let mut best: Option<(f64, Vec2)> = None;
    for (j, &q) in v.iter().enumerate() {
        if j == vi || j == (vi + n - 1) % n || j == (vi + 1) % n {
            continue;
        }
        let inside = cross3(a, p, q) * cross3(a, p, b) > 0.0
            && cross3(p, b, q) * cross3(p, b, a) > 0.0
            && cross3(b, a, q) * cross3(b, a, p) > 0.0;
        if inside {
            let d = q.distance_to(&p);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, q));
            }
        }
    }
    match best {
        Some((_, q)) => (p + q) * 0.5,
        None => (a + p + b) * (1.0 / 3.0),
    }
}

/// Orients disjoint simple loops by containment depth: even depth
/// (outer boundaries) counterclockwise, odd depth (holes) clockwise.
fn orient_loops(mut loops: Vec<Polygon2>) -> Vec<Polygon2> {
    let reps: Vec<Vec2> = loops.iter().map(interior_point).collect();
    for i in 0..loops.len() {
        let depth = loops
            .iter()
            .enumerate()
            .filter(|(j, l)| *j != i && point_in_poly(reps[i], &l.vertices))
            .count();
        let want_ccw = depth % 2 == 0;
        if loops[i].is_ccw() != want_ccw {
            loops[i].reverse();
        }
    }
    loops
}

fn gh_traverse(arena: &mut [GhNode]) -> Vec<Polygon2> {
    let mut result = Vec::new();
    while let Some(start) = arena.iter().position(|n| n.is_x && !n.processed) {
        let mut poly = vec![arena[start].p];
        let mut cur = start;
        let limit = 4 * arena.len();
        loop {
            arena[cur].processed = true;
            let nb = arena[cur].neighbor;
            arena[nb].processed = true;
            if arena[cur].entry {
                loop {
                    cur = arena[cur].next;
                    poly.push(arena[cur].p);
                    if arena[cur].is_x {
                        break;
                    }
                }
            } else {
                loop {
                    cur = arena[cur].prev;
                    poly.push(arena[cur].p);
                    if arena[cur].is_x {
                        break;
                    }
                }
            }
            arena[cur].processed = true;
            cur = arena[cur].neighbor;
            if cur == start || poly.len() > limit {
                break;
            }
        }
        while poly.len() > 1 && poly[0].distance_to(poly.last().expect("nonempty")) < 1e-12 {
            poly.pop();
        }
        if poly.len() >= 3 {
            result.push(Polygon2::new(poly));
        }
    }
    result
}

#[derive(Clone, Copy, PartialEq)]
enum BoolOp {
    Union,
    Intersection,
    Difference,
}

fn normalize_ccw(p: &Polygon2) -> Vec<Vec2> {
    if p.is_ccw() {
        p.vertices.clone()
    } else {
        p.vertices.iter().rev().copied().collect()
    }
}

/// Greiner-Hormann clipping ("Efficient Clipping of Arbitrary
/// Polygons", TOG 1998). Degenerate configurations (shared vertices,
/// collinear edge overlaps) are handled by perturbing `b` by a
/// deterministic sub-1e-8 offset and retrying, so results are exact
/// up to that perturbation.
fn gh_boolean(a: &Polygon2, b: &Polygon2, op: BoolOp) -> Vec<Polygon2> {
    assert!(a.vertices.len() >= 3 && b.vertices.len() >= 3, "polygons need >= 3 vertices");
    let pa = normalize_ccw(a);
    let pb = normalize_ccw(b);
    let diameter = pa
        .iter()
        .chain(&pb)
        .map(|p| p.magnitude())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let mut built = None;
    for attempt in 0..6 {
        let trial: Vec<Vec2> = if attempt == 0 {
            pb.clone()
        } else {
            let d = diameter * 1e-9 * attempt as f64;
            let shift = Vec2::new(0.754_877_666_2 * d, 0.569_840_290_9 * d);
            pb.iter().map(|&p| p + shift).collect()
        };
        if let Some(ab) = gh_build(&pa, &trial) {
            built = Some(ab);
            break;
        }
    }
    let Some((mut arena, b_start)) = built else {
        return no_intersection_result(a, b, op);
    };
    let has_x = arena.iter().any(|n| n.is_x);
    if !has_x {
        return no_intersection_result(a, b, op);
    }
    // Entry flags as built give the intersection; invert both rings
    // for union, only the clip ring for difference.
    for (i, n) in arena.iter_mut().enumerate() {
        if !n.is_x {
            continue;
        }
        let invert = match op {
            BoolOp::Intersection => false,
            BoolOp::Union => true,
            BoolOp::Difference => i >= b_start,
        };
        if invert {
            n.entry = !n.entry;
        }
    }
    orient_loops(gh_traverse(&mut arena))
}

/// Containment-based result when the boundaries do not cross. Holes
/// are returned as clockwise (negative-area) loops.
fn no_intersection_result(a: &Polygon2, b: &Polygon2, op: BoolOp) -> Vec<Polygon2> {
    let pa = normalize_ccw(a);
    let pb = normalize_ccw(b);
    let a_in_b = point_in_poly(pa[0], &pb);
    let b_in_a = point_in_poly(pb[0], &pa);
    let ccw = |v: Vec<Vec2>| Polygon2::new(v);
    let cw = |v: Vec<Vec2>| {
        let mut r = v;
        r.reverse();
        Polygon2::new(r)
    };
    match op {
        BoolOp::Union => {
            if a_in_b {
                vec![ccw(pb)]
            } else if b_in_a {
                vec![ccw(pa)]
            } else {
                vec![ccw(pa), ccw(pb)]
            }
        }
        BoolOp::Intersection => {
            if a_in_b {
                vec![ccw(pa)]
            } else if b_in_a {
                vec![ccw(pb)]
            } else {
                Vec::new()
            }
        }
        BoolOp::Difference => {
            if a_in_b {
                Vec::new()
            } else if b_in_a {
                vec![ccw(pa), cw(pb)]
            } else {
                vec![ccw(pa)]
            }
        }
    }
}

/// Union of two simple polygons. Outer loops come out
/// counterclockwise; holes (e.g. two C shapes closing a ring)
/// clockwise.
#[must_use]
pub fn boolean_union(a: &Polygon2, b: &Polygon2) -> Vec<Polygon2> {
    gh_boolean(a, b, BoolOp::Union)
}

/// Intersection of two simple polygons (possibly several pieces).
#[must_use]
pub fn boolean_intersection(a: &Polygon2, b: &Polygon2) -> Vec<Polygon2> {
    gh_boolean(a, b, BoolOp::Intersection)
}

/// Difference a − b; a hole fully inside `a` is returned as a
/// clockwise loop.
#[must_use]
pub fn boolean_difference(a: &Polygon2, b: &Polygon2) -> Vec<Polygon2> {
    gh_boolean(a, b, BoolOp::Difference)
}

/// Symmetric difference: (a − b) ∪ (b − a), returned as the two
/// difference loop sets concatenated.
#[must_use]
pub fn boolean_xor(a: &Polygon2, b: &Polygon2) -> Vec<Polygon2> {
    let mut out = boolean_difference(a, b);
    out.extend(boolean_difference(b, a));
    out
}

/// Sutherland-Hodgman clipping of an arbitrary subject polygon
/// against a convex clip polygon.
///
/// # Panics
/// Panics unless `clip` is convex with >= 3 vertices.
#[must_use]
pub fn clip_polygon_convex(subject: &Polygon2, clip: &Polygon2) -> Polygon2 {
    assert!(clip.vertices.len() >= 3 && clip.is_convex(), "clip polygon must be convex");
    let cl = normalize_ccw(clip);
    let mut pts = normalize_ccw(subject);
    let n = cl.len();
    for i in 0..n {
        if pts.is_empty() {
            break;
        }
        let (a, b) = (cl[i], cl[(i + 1) % n]);
        let inside = |p: Vec2| cross3(a, b, p) >= 0.0;
        let mut out = Vec::with_capacity(pts.len() + 4);
        for k in 0..pts.len() {
            let (p, q) = (pts[k], pts[(k + 1) % pts.len()]);
            let (ip, iq) = (inside(p), inside(q));
            if ip {
                out.push(p);
            }
            if ip != iq {
                let denom = cross3(a, b, p) - cross3(a, b, q);
                if denom.abs() > 0.0 {
                    let t = cross3(a, b, p) / denom;
                    out.push(p.lerp(&q, t));
                }
            }
        }
        pts = out;
    }
    Polygon2::new(pts)
}

/// Clips a polygon to an axis-aligned rectangle
/// (Sutherland-Hodgman).
#[must_use]
pub fn clip_polygon_rect(subject: &Polygon2, rect: &Rect) -> Polygon2 {
    let c = rect.corners();
    clip_polygon_convex(subject, &Polygon2::new(vec![c[0], c[1], c[2], c[3]]))
}

/// Liang-Barsky segment clipping against a rectangle; `None` when the
/// segment misses it entirely.
#[must_use]
pub fn clip_line_rect(a: Vec2, b: Vec2, rect: &Rect) -> Option<(Vec2, Vec2)> {
    let d = b - a;
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    for (p, q) in [
        (-d.x, a.x - rect.min.x),
        (d.x, rect.max.x - a.x),
        (-d.y, a.y - rect.min.y),
        (d.y, rect.max.y - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            t0 = t0.max(r);
        } else {
            t1 = t1.min(r);
        }
        if t0 > t1 {
            return None;
        }
    }
    Some((a.lerp(&b, t0), a.lerp(&b, t1)))
}

/// Hertel-Mehlhorn convex decomposition: triangulate, then greedily
/// remove inessential diagonals. At most 4x the optimal piece count.
///
/// # Panics
/// Panics when the polygon cannot be triangulated (see
/// [`triangulate_ear_clipping`] for the failure modes).
#[must_use]
pub fn convex_decomposition(poly: &Polygon2) -> Vec<Polygon2> {
    if poly.vertices.len() >= 3 && poly.is_convex() {
        return vec![Polygon2::new(normalize_ccw(poly))];
    }
    let tris = triangulate_ear_clipping(poly).expect("polygon must be triangulable");
    let pts = &poly.vertices;
    // Pieces as counterclockwise index loops.
    let mut pieces: Vec<Option<Vec<usize>>> = tris.iter().map(|t| Some(t.to_vec())).collect();
    let convex = |loop_: &[usize]| {
        let m = loop_.len();
        (0..m).all(|k| {
            cross3(pts[loop_[k]], pts[loop_[(k + 1) % m]], pts[loop_[(k + 2) % m]]) >= -1e-12
        })
    };
    let mut merged = true;
    while merged {
        merged = false;
        // Diagonal -> owning pieces, rebuilt each pass.
        let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (pi, piece) in pieces.iter().enumerate() {
            if let Some(loop_) = piece {
                let m = loop_.len();
                for k in 0..m {
                    let (u, v) = (loop_[k], loop_[(k + 1) % m]);
                    by_edge.entry((u.min(v), u.max(v))).or_default().push(pi);
                }
            }
        }
        'merge: for (_, owners) in by_edge.iter().filter(|(_, o)| o.len() == 2) {
            let (pi, qi) = (owners[0], owners[1]);
            let (Some(pl), Some(ql)) = (pieces[pi].clone(), pieces[qi].clone()) else {
                continue;
            };
            // Find the shared edge in p (u followed by v).
            let m = pl.len();
            for k in 0..m {
                let (u, v) = (pl[k], pl[(k + 1) % m]);
                let qm = ql.len();
                let Some(pos) = (0..qm).find(|&t| ql[t] == v && ql[(t + 1) % qm] == u) else {
                    continue;
                };
                // Splice q (minus the shared edge) into p.
                let mut joined: Vec<usize> = Vec::with_capacity(m + qm - 2);
                for t in 0..m {
                    joined.push(pl[(k + 1 + t) % m]);
                }
                // joined now starts at v and ends at u; insert q's
                // path from u..v exclusive after u.
                for t in 2..qm {
                    joined.push(ql[(pos + t) % qm]);
                }
                if convex(&joined) {
                    pieces[pi] = Some(joined);
                    pieces[qi] = None;
                    merged = true;
                    break 'merge;
                }
            }
        }
    }
    pieces
        .into_iter()
        .flatten()
        .map(|loop_| Polygon2::new(loop_.iter().map(|&i| pts[i]).collect()))
        .collect()
}

/// Convex hull by Andrew's monotone chain, counterclockwise, minimal
/// vertex set (collinear points dropped).
///
/// # Panics
/// Panics with fewer than 3 input points.
#[must_use]
pub fn convex_hull_2d(points: &[Vec2]) -> Polygon2 {
    assert!(points.len() >= 3, "convex_hull_2d requires >= 3 points");
    let mut pts: Vec<Vec2> = points.to_vec();
    pts.sort_by(|a, b| (a.x, a.y).partial_cmp(&(b.x, b.y)).expect("finite coordinates"));
    pts.dedup_by(|a, b| a.distance_to(b) == 0.0);
    let mut hull: Vec<Vec2> = Vec::with_capacity(pts.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &Vec2>> =
            if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &p in iter {
            while hull.len() >= start + 2
                && cross3(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop(); // endpoint repeats as the next pass's start
    }
    Polygon2::new(hull)
}

/// Convex hull of 3-D points as a triangle mesh (incremental hull,
/// outward-facing counterclockwise faces).
///
/// # Panics
/// Panics with fewer than 4 points or fully coplanar input.
#[must_use]
pub fn convex_hull_3d(points: &[Vec3]) -> Mesh {
    let faces = crate::geometry::hull::convex_hull_3d(points);
    let mut m = Mesh::new(points.to_vec(), faces).expect("hull indices are valid");
    m.remove_unused_vertices();
    if m.volume() < 0.0 {
        m.flip_normals();
    }
    m
}

/// Straight skeleton arcs of a simple polygon by the
/// shrinking-wavefront (roof) construction in the style of Felkel &
/// Obdržálek 1998, processing edge events (wavefront edges collapsing
/// as vertices meet). Split events of reflex vertices are not
/// resolved, so results are exact for convex polygons and approximate
/// for mildly non-convex ones. Each arc runs from a wavefront vertex
/// (original or intermediate) to the event point that consumed it.
///
/// # Panics
/// Panics unless the polygon is simple with >= 3 vertices.
#[must_use]
pub fn straight_skeleton(poly: &Polygon2) -> Vec<Segment2> {
    assert!(poly.vertices.len() >= 3, "straight_skeleton requires >= 3 vertices");
    assert!(poly.is_simple(), "straight_skeleton requires a simple polygon");
    let pts = normalize_ccw(poly);

    // Wavefront vertex: position at time 0 and unit-speed velocity
    // (the angular bisector scaled so edges advance at speed 1).
    #[derive(Clone, Copy)]
    struct Wv {
        pos: Vec2,
        vel: Vec2,
        born: f64, // time when this vertex appeared
        prev: usize,
        next: usize,
        dead: bool,
    }
    let n = pts.len();
    let edge_normal = |a: Vec2, b: Vec2| {
        let e = b - a;
        Vec2::new(-e.y, e.x).normalized() // inward for counterclockwise
    };
    let bisector_vel = |n0: Vec2, n1: Vec2| -> Vec2 {
        // Velocity v with v·n0 = v·n1 = 1 (edges advance at unit
        // speed): solve the 2x2 system.
        let det = n0.x * n1.y - n0.y * n1.x;
        if det.abs() < 1e-12 {
            n0 // parallel edges: move along the shared normal
        } else {
            Vec2::new((n1.y - n0.y) / det, (n0.x - n1.x) / det)
        }
    };
    let mut wv: Vec<Wv> = (0..n)
        .map(|i| {
            let p = pts[i];
            let n0 = edge_normal(pts[(i + n - 1) % n], p);
            let n1 = edge_normal(p, pts[(i + 1) % n]);
            Wv {
                pos: p,
                vel: bisector_vel(n0, n1),
                born: 0.0,
                prev: (i + n - 1) % n,
                next: (i + 1) % n,
                dead: false,
            }
        })
        .collect();
    let pos_at = |v: &Wv, t: f64| v.pos + v.vel * (t - v.born);
    // Edge event time for consecutive vertices i -> j: when their
    // positions coincide.
    let edge_event = |a: &Wv, b: &Wv| -> Option<(f64, Vec2)> {
        let dp = pos_at(b, 0.0_f64.max(a.born.max(b.born))) - pos_at(a, a.born.max(b.born));
        let t0 = a.born.max(b.born);
        let dv = b.vel - a.vel;
        let vv = dv.magnitude_squared();
        if vv < 1e-18 {
            return None;
        }
        let t = t0 - dp.dot(&dv) / vv;
        if t <= t0 + 1e-12 {
            return None;
        }
        let pa = pos_at(a, t);
        let pb = pos_at(b, t);
        if pa.distance_to(&pb) < 1e-7 * (1.0 + pa.magnitude()) {
            Some((t, (pa + pb) * 0.5))
        } else {
            None
        }
    };
    let mut arcs: Vec<Segment2> = Vec::new();
    let mut alive = n;
    let mut guard = 0usize;
    while alive > 2 && guard < 4 * n * n {
        guard += 1;
        // Next edge event over all wavefront edges.
        let mut best: Option<(f64, usize, Vec2)> = None;
        for (i, v) in wv.iter().enumerate() {
            if v.dead {
                continue;
            }
            let j = v.next;
            if let Some((t, p)) = edge_event(v, &wv[j]) {
                if best.is_none_or(|(bt, _, _)| t < bt) {
                    best = Some((t, i, p));
                }
            }
        }
        let Some((t, i, p)) = best else { break };
        let j = wv[i].next;
        // Emit skeleton arcs from both collapsing vertices.
        arcs.push(Segment2 { a: wv[i].pos, b: p });
        arcs.push(Segment2 { a: wv[j].pos, b: p });
        // Merge i and j into one wavefront vertex at p.
        let prev = wv[i].prev;
        let next = wv[j].next;
        wv[j].dead = true;
        alive -= 1;
        if prev == j || next == i {
            // Triangle collapse: nothing left of this loop.
            wv[i].dead = true;
            alive -= 1;
            continue;
        }
        // New bisector from the neighboring wavefront edges.
        let a0 = pos_at(&wv[prev], t);
        let a1 = pos_at(&wv[next], t);
        let n0 = edge_normal(a0, p);
        let n1 = edge_normal(p, a1);
        wv[i] = Wv {
            pos: p,
            vel: bisector_vel(n0, n1),
            born: t,
            prev,
            next,
            dead: false,
        };
        wv[prev].next = i;
        wv[next].prev = i;
    }
    // Connect the final surviving pair.
    let last: Vec<usize> = (0..wv.len()).filter(|&i| !wv[i].dead).collect();
    if last.len() == 2 {
        let (a, b) = (last[0], last[1]);
        if let Some((t, p)) = edge_event(&wv[a], &wv[b]) {
            let _ = t;
            arcs.push(Segment2 { a: wv[a].pos, b: p });
            arcs.push(Segment2 { a: wv[b].pos, b: p });
        } else {
            arcs.push(Segment2 { a: wv[a].pos, b: wv[b].pos });
        }
    }
    arcs
}

/// Largest inscribed circle (pole of inaccessibility) by Mapbox's
/// polylabel quadtree refinement.
///
/// # Panics
/// Panics unless the polygon has >= 3 vertices and `precision > 0`
/// would hold for the derived tolerance (bbox-scaled 1e-6).
#[must_use]
pub fn largest_inscribed_circle(poly: &Polygon2) -> Circle {
    assert!(poly.vertices.len() >= 3, "polygon needs >= 3 vertices");
    let pts = &poly.vertices;
    let bbox = poly.bounding_rect();
    let precision = (bbox.max - bbox.min).magnitude() * 1e-6;
    // Signed distance to the polygon boundary (positive inside).
    let dist = |p: Vec2| -> f64 {
        let mut d = f64::INFINITY;
        let n = pts.len();
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let e = b - a;
            let t = ((p - a).dot(&e) / e.magnitude_squared()).clamp(0.0, 1.0);
            d = d.min(p.distance_to(&(a + e * t)));
        }
        if point_in_poly(p, pts) {
            d
        } else {
            -d
        }
    };
    #[derive(PartialEq)]
    struct Cell {
        max_potential: f64,
        d: f64,
        c: Vec2,
        h: f64,
    }
    impl Eq for Cell {}
    impl Ord for Cell {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.max_potential.total_cmp(&other.max_potential)
        }
    }
    impl PartialOrd for Cell {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    let cell = |c: Vec2, h: f64| Cell {
        max_potential: dist(c) + h * std::f64::consts::SQRT_2,
        d: dist(c),
        c,
        h,
    };
    let mut heap = std::collections::BinaryHeap::new();
    let size = bbox.max - bbox.min;
    let h0 = size.x.min(size.y) / 2.0;
    let mut y = bbox.min.y + h0;
    while y < bbox.max.y {
        let mut x = bbox.min.x + h0;
        while x < bbox.max.x {
            heap.push(cell(Vec2::new(x, y), h0));
            x += 2.0 * h0;
        }
        y += 2.0 * h0;
    }
    let centroid = poly.centroid();
    let mut best = Cell { max_potential: 0.0, d: dist(centroid), c: centroid, h: 0.0 };
    while let Some(c) = heap.pop() {
        if c.d > best.d {
            best = Cell { max_potential: c.max_potential, d: c.d, c: c.c, h: c.h };
        }
        if c.max_potential - best.d <= precision {
            continue;
        }
        let h = c.h / 2.0;
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            heap.push(cell(c.c + Vec2::new(dx * h, dy * h), h));
        }
    }
    Circle { center: best.c, radius: best.d.max(0.0) }
}

fn circle_from_2(a: Vec2, b: Vec2) -> Circle {
    Circle { center: (a + b) * 0.5, radius: a.distance_to(&b) / 2.0 }
}

fn circle_from_3(a: Vec2, b: Vec2, c: Vec2) -> Circle {
    crate::spatial::primitives::Triangle2 { a, b, c }.circumcircle()
}

fn circle_contains(c: &Circle, p: Vec2, eps: f64) -> bool {
    p.distance_to(&c.center) <= c.radius + eps
}

/// Smallest enclosing circle by Welzl's expected-linear incremental
/// algorithm (deterministically shuffled).
///
/// # Panics
/// Panics on empty input.
#[must_use]
pub fn smallest_enclosing_circle(points: &[Vec2]) -> Circle {
    assert!(!points.is_empty(), "smallest_enclosing_circle requires points");
    let mut pts = points.to_vec();
    // Deterministic shuffle.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for i in (1..pts.len()).rev() {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        pts.swap(i, (state >> 33) as usize % (i + 1));
    }
    let eps = 1e-10;
    let mut c = Circle { center: pts[0], radius: 0.0 };
    for i in 1..pts.len() {
        if circle_contains(&c, pts[i], eps) {
            continue;
        }
        c = Circle { center: pts[i], radius: 0.0 };
        for j in 0..i {
            if circle_contains(&c, pts[j], eps) {
                continue;
            }
            c = circle_from_2(pts[i], pts[j]);
            for k in 0..j {
                if circle_contains(&c, pts[k], eps) {
                    continue;
                }
                c = circle_from_3(pts[i], pts[j], pts[k]);
            }
        }
    }
    c
}

/// Minimum-area oriented bounding rectangle by rotating calipers over
/// the convex hull: returns `(center, half_extents, angle)`, the
/// rectangle's local x axis rotated by `angle` from world x.
///
/// # Panics
/// Panics with fewer than 3 points.
#[must_use]
pub fn minimum_bounding_rect(points: &[Vec2]) -> (Vec2, Vec2, f64) {
    let hull = convex_hull_2d(points).vertices;
    let n = hull.len();
    let mut best = (f64::INFINITY, Vec2::ZERO, Vec2::ZERO, 0.0);
    for i in 0..n {
        let e = hull[(i + 1) % n] - hull[i];
        let angle = e.y.atan2(e.x);
        let (mut lo, mut hi) = (Vec2::new(f64::INFINITY, f64::INFINITY), Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY));
        for &p in &hull {
            let q = p.rotate(-angle);
            lo = Vec2::new(lo.x.min(q.x), lo.y.min(q.y));
            hi = Vec2::new(hi.x.max(q.x), hi.y.max(q.y));
        }
        let size = hi - lo;
        let area = size.x * size.y;
        if area < best.0 {
            let center_local = (lo + hi) * 0.5;
            best = (area, center_local.rotate(angle), size * 0.5, angle);
        }
    }
    (best.1, best.2, best.3)
}

/// Farthest vertex pair (diameter) of a polygon: indices and
/// distance.
///
/// # Panics
/// Panics with fewer than 2 vertices.
#[must_use]
pub fn polygon_diameter(poly: &Polygon2) -> (usize, usize, f64) {
    assert!(poly.vertices.len() >= 2, "polygon_diameter requires >= 2 vertices");
    let v = &poly.vertices;
    let mut best = (0usize, 1usize, 0.0f64);
    for i in 0..v.len() {
        for j in i + 1..v.len() {
            let d = v[i].distance_to(&v[j]);
            if d > best.2 {
                best = (i, j, d);
            }
        }
    }
    best
}

/// Minimum width of the polygon: the smallest distance between
/// parallel supporting lines (over hull edge directions).
///
/// # Panics
/// Panics with fewer than 3 vertices.
#[must_use]
pub fn polygon_width(poly: &Polygon2) -> f64 {
    let hull = convex_hull_2d(&poly.vertices).vertices;
    let n = hull.len();
    let mut width = f64::INFINITY;
    for i in 0..n {
        let a = hull[i];
        let e = (hull[(i + 1) % n] - a).normalized();
        let far = hull
            .iter()
            .map(|&p| e.cross(&(p - a)).abs())
            .fold(0.0f64, f64::max);
        width = width.min(far);
    }
    width
}

/// Resamples the polygon boundary into `n` equally spaced points
/// (by arclength) starting at vertex 0.
///
/// # Panics
/// Panics unless `n >= 3` and the polygon has positive perimeter.
#[must_use]
pub fn resample_polygon(poly: &Polygon2, n: usize) -> Polygon2 {
    assert!(n >= 3, "resample requires n >= 3");
    let perimeter = poly.perimeter();
    assert!(perimeter > 0.0, "polygon must have positive perimeter");
    let v = &poly.vertices;
    let m = v.len();
    let mut out = Vec::with_capacity(n);
    let step = perimeter / n as f64;
    let mut target = 0.0;
    let mut walked = 0.0;
    let mut seg = 0usize;
    for _ in 0..n {
        while seg < m {
            let (a, b) = (v[seg], v[(seg + 1) % m]);
            let len = a.distance_to(&b);
            if walked + len >= target - 1e-12 {
                let t = if len == 0.0 { 0.0 } else { ((target - walked) / len).clamp(0.0, 1.0) };
                out.push(a.lerp(&b, t));
                break;
            }
            walked += len;
            seg += 1;
        }
        target += step;
    }
    Polygon2::new(out)
}

/// Chaikin corner cutting (closed polygon): each iteration replaces
/// every edge with its 1/4 and 3/4 points, converging to a smooth
/// quadratic B-spline.
#[must_use]
pub fn smooth_chaikin(poly: &Polygon2, iterations: usize) -> Polygon2 {
    let mut v = poly.vertices.clone();
    for _ in 0..iterations {
        let m = v.len();
        let mut next = Vec::with_capacity(2 * m);
        for i in 0..m {
            let (a, b) = (v[i], v[(i + 1) % m]);
            next.push(a.lerp(&b, 0.25));
            next.push(a.lerp(&b, 0.75));
        }
        v = next;
    }
    Polygon2::new(v)
}

/// Replaces each corner by a circular arc of the given radius
/// (clamped to half of the shorter adjacent edge), sampled with
/// `segments` points.
///
/// # Panics
/// Panics unless `radius > 0` and `segments >= 1`.
#[must_use]
pub fn round_corners(poly: &Polygon2, radius: f64, segments: usize) -> Polygon2 {
    assert!(radius > 0.0 && segments >= 1, "requires radius > 0 and segments >= 1");
    let v = &poly.vertices;
    let n = v.len();
    let mut out = Vec::with_capacity(n * (segments + 1));
    for i in 0..n {
        let p = v[i];
        let a = v[(i + n - 1) % n];
        let b = v[(i + 1) % n];
        let da = (a - p).normalized();
        let db = (b - p).normalized();
        let cos2 = da.dot(&db).clamp(-1.0, 1.0);
        let angle = cos2.acos();
        if angle.sin().abs() < 1e-9 {
            out.push(p);
            continue;
        }
        // Distance along each edge to the tangent points.
        let cut = (radius / (angle / 2.0).tan())
            .min(0.5 * (a - p).magnitude())
            .min(0.5 * (b - p).magnitude());
        let r = cut * (angle / 2.0).tan();
        let t0 = p + da * cut;
        let t1 = p + db * cut;
        let center = p + (da + db).normalized() * (r.hypot(cut));
        let a0 = (t0 - center).y.atan2((t0 - center).x);
        let mut a1 = (t1 - center).y.atan2((t1 - center).x);
        // Take the short way around.
        while a1 - a0 > std::f64::consts::PI {
            a1 -= 2.0 * std::f64::consts::PI;
        }
        while a0 - a1 > std::f64::consts::PI {
            a1 += 2.0 * std::f64::consts::PI;
        }
        for k in 0..=segments {
            let t = a0 + (a1 - a0) * k as f64 / segments as f64;
            out.push(center + Vec2::new(t.cos(), t.sin()) * r);
        }
    }
    Polygon2::new(out)
}

/// Parallel hatch lines filling the polygon: scanlines spaced by
/// `spacing`, rotated by `angle` radians from the x axis (even-odd
/// filled).
///
/// # Panics
/// Panics unless `spacing > 0`.
#[must_use]
pub fn hatch_fill(poly: &Polygon2, spacing: f64, angle: f64) -> Vec<Segment2> {
    assert!(spacing > 0.0, "hatch spacing must be positive");
    let pts: Vec<Vec2> = poly.vertices.iter().map(|p| p.rotate(-angle)).collect();
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in &pts {
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }
    let n = pts.len();
    let mut out = Vec::new();
    let mut y = ymin + spacing / 2.0;
    while y < ymax {
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            if (a.y > y) != (b.y > y) {
                xs.push(a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x));
            }
        }
        xs.sort_by(f64::total_cmp);
        for &[x0, x1] in xs.as_chunks::<2>().0 {
            out.push(Segment2 {
                a: Vec2::new(x0, y).rotate(angle),
                b: Vec2::new(x1, y).rotate(angle),
            });
        }
        y += spacing;
    }
    out
}

/// Concentric fill: repeated inward offsets by `spacing` until the
/// polygon vanishes.
///
/// # Panics
/// Panics unless `spacing > 0`.
#[must_use]
pub fn contour_fill(poly: &Polygon2, spacing: f64) -> Vec<Polygon2> {
    assert!(spacing > 0.0, "contour spacing must be positive");
    let mut out = Vec::new();
    let mut frontier = vec![Polygon2::new(normalize_ccw(poly))];
    let mut guard = 0usize;
    while !frontier.is_empty() && guard < 10_000 {
        guard += 1;
        let mut next = Vec::new();
        for p in &frontier {
            next.extend(offset_polygon(p, -spacing, JoinStyle::Miter(2.0)));
        }
        out.extend(frontier);
        frontier = next;
    }
    out
}

/// Triangulates a polygon (optionally with holes) into a flat mesh at
/// z = 0, facing +z.
///
/// # Panics
/// Panics when triangulation fails (non-simple input).
#[must_use]
pub fn polygon_to_mesh_2d(poly: &Polygon2, holes: &[Polygon2]) -> Mesh {
    let (pts, tris) =
        triangulate_with_holes(poly, holes).expect("polygon must be triangulable");
    Mesh {
        vertices: pts.iter().map(|p| Vec3::new(p.x, p.y, 0.0)).collect(),
        indices: tris,
        normals: None,
        uvs: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn square(s: f64) -> Polygon2 {
        Polygon2::new(vec![
            Vec2::new(-s, -s),
            Vec2::new(s, -s),
            Vec2::new(s, s),
            Vec2::new(-s, s),
        ])
    }

    fn lshape() -> Polygon2 {
        Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ])
    }

    fn loops_area(loops: &[Polygon2]) -> f64 {
        loops.iter().map(Polygon2::area_signed).sum()
    }

    #[test]
    fn test_triangulation_area() {
        for poly in [square(1.0), lshape()] {
            let tris = triangulate_ear_clipping(&poly).unwrap();
            assert_eq!(tris.len(), poly.vertices.len() - 2);
            let sum: f64 = tris
                .iter()
                .map(|&[a, b, c]| {
                    cross3(poly.vertices[a], poly.vertices[b], poly.vertices[c]) / 2.0
                })
                .sum();
            assert!((sum - poly.area()).abs() < 1e-12);
        }
        // Zero-area (collinear) polygon rejected.
        let flat = Polygon2::new(vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)]);
        assert!(triangulate_ear_clipping(&flat).is_err());
        // Self-intersecting bowtie rejected.
        let bow = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ]);
        assert!(triangulate_ear_clipping(&bow).is_err());
    }

    #[test]
    fn test_triangulation_with_holes_area() {
        let outer = square(2.0);
        let hole = square(1.0);
        let (pts, tris) = triangulate_with_holes(&outer, &[hole]).unwrap();
        let sum: f64 = tris.iter().map(|&[a, b, c]| cross3(pts[a], pts[b], pts[c]) / 2.0).sum();
        assert!((sum - (16.0 - 4.0)).abs() < 1e-9, "annulus area {sum}");
        let m = polygon_to_mesh_2d(&square(2.0), &[square(1.0)]);
        assert!((m.surface_area() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_simplification() {
        // A noisy straight line simplifies to its endpoints.
        let pts: Vec<Vec2> =
            (0..=20).map(|i| Vec2::new(i as f64, if i % 2 == 0 { 0.0 } else { 0.01 })).collect();
        let dp = simplify_douglas_peucker(&pts, 0.1);
        assert_eq!(dp.len(), 2);
        assert_eq!(dp[0], pts[0]);
        assert_eq!(dp[1], pts[20]);
        // A right angle survives.
        let corner = vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)];
        assert_eq!(simplify_douglas_peucker(&corner, 0.1).len(), 3);
        let vw = simplify_visvalingam(&pts, 0.5);
        assert!(vw.len() < pts.len());
        assert_eq!(vw[0], pts[0]);
    }

    #[test]
    fn test_offset_roundtrip_convex() {
        let p = square(1.0);
        // Grow-then-shrink is exact for miter joins; round and bevel
        // joins clip the corners by design, bevel the most.
        for (join, tol) in [
            (JoinStyle::Miter(4.0), 1e-9),
            (JoinStyle::Round(16), 0.03),
            (JoinStyle::Bevel, 0.1),
        ] {
            let grown = offset_polygon(&p, 0.5, join);
            assert_eq!(grown.len(), 1);
            assert!(grown[0].area() > p.area());
            let back = offset_polygon(&grown[0], -0.5, join);
            assert_eq!(back.len(), 1);
            assert!(
                (back[0].area() - p.area()).abs() < tol,
                "offset round-trip area {} vs {}",
                back[0].area(),
                p.area()
            );
        }
        // Miter square grows by exactly d on each side.
        let g = offset_polygon(&p, 0.25, JoinStyle::Miter(4.0));
        assert!((g[0].area() - 2.5 * 2.5).abs() < 1e-9);
        // Inset past the inradius vanishes.
        assert!(offset_polygon(&p, -1.5, JoinStyle::Bevel).is_empty());
    }

    #[test]
    fn test_minkowski_sums() {
        let a = square(1.0);
        let b = square(0.5);
        let s = minkowski_sum_convex(&a, &b);
        assert!(s.is_convex());
        assert!((s.area() - 9.0).abs() < 1e-9, "sum of squares is a square: {}", s.area());
        // Nonconvex: L-shape + small square = dilated L.
        let l = lshape();
        let parts = minkowski_sum(&l, &square(0.1));
        assert!(!parts.is_empty());
        let area = loops_area(&parts);
        // Dilation area = A + P*r ... for square structuring element
        // of half-width 0.1: A + P*0.1 + 4*0.1^2 (with corner terms;
        // reflex corner subtracts). Just sanity-bound it.
        assert!(area > l.area() && area < l.area() + 2.0);
    }

    #[test]
    fn test_boolean_ops_partial_overlap() {
        let a = square(1.0);
        let mut bv = square(1.0).vertices;
        for v in &mut bv {
            *v = *v + Vec2::new(1.0, 1.0);
        }
        let b = Polygon2::new(bv);
        let (aa, ab) = (a.area(), b.area());
        let union = boolean_union(&a, &b);
        let inter = boolean_intersection(&a, &b);
        let xor = boolean_xor(&a, &b);
        let (ua, ia, xa) = (loops_area(&union), loops_area(&inter), loops_area(&xor));
        assert!((ia - 1.0).abs() < 1e-6, "unit overlap, got {ia}");
        assert!((ua - 7.0).abs() < 1e-6, "union area {ua}");
        assert!(ua <= aa + ab + 1e-9 && ua >= aa.max(ab) - 1e-9);
        assert!((ia + xa - ua).abs() < 1e-6, "inclusion-exclusion");
        let diff = boolean_difference(&a, &b);
        assert!((loops_area(&diff) - (aa - ia)).abs() < 1e-6);
    }

    #[test]
    fn test_boolean_disjoint_and_nested() {
        let a = square(2.0);
        let b = square(0.5);
        assert_eq!(boolean_union(&a, &b).len(), 1);
        assert!((loops_area(&boolean_intersection(&a, &b)) - 1.0).abs() < 1e-12);
        let d = boolean_difference(&a, &b);
        assert_eq!(d.len(), 2, "difference leaves a hole loop");
        assert!((loops_area(&d) - 15.0).abs() < 1e-12, "hole loop is clockwise");
        // Disjoint.
        let mut fv = square(0.5).vertices;
        for v in &mut fv {
            *v = *v + Vec2::new(5.0, 0.0);
        }
        let far = Polygon2::new(fv);
        assert_eq!(boolean_union(&a, &far).len(), 2);
        assert!(boolean_intersection(&a, &far).is_empty());
        assert_eq!(boolean_difference(&a, &far).len(), 1);
    }

    #[test]
    fn test_boolean_degenerate_shared_edge() {
        // Two unit squares sharing an edge: degeneracy handling kicks
        // in (perturbation), union close to a 2x1 rectangle.
        let a = square(0.5);
        let mut bv = square(0.5).vertices;
        for v in &mut bv {
            *v = *v + Vec2::new(1.0, 0.0);
        }
        let b = Polygon2::new(bv);
        let u = boolean_union(&a, &b);
        assert!((loops_area(&u) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_clipping() {
        let subject = lshape();
        let clip = square(0.75);
        let out = clip_polygon_convex(&subject, &clip);
        // L-shape ∩ [-0.75,0.75]^2 = [0,0.75]^2.
        assert!((out.area() - 0.5625).abs() < 1e-12);
        let r = Rect { min: Vec2::new(0.25, 0.25), max: Vec2::new(0.5, 3.0) };
        let rc = clip_polygon_rect(&subject, &r);
        assert!((rc.area() - 0.25 * 1.75).abs() < 1e-12);
        // Liang-Barsky.
        let rect = Rect { min: Vec2::new(0.0, 0.0), max: Vec2::new(1.0, 1.0) };
        let (a, b) =
            clip_line_rect(Vec2::new(-1.0, 0.5), Vec2::new(2.0, 0.5), &rect).unwrap();
        assert!((a - Vec2::new(0.0, 0.5)).magnitude() < 1e-12);
        assert!((b - Vec2::new(1.0, 0.5)).magnitude() < 1e-12);
        assert!(clip_line_rect(Vec2::new(-1.0, 2.0), Vec2::new(2.0, 2.0), &rect).is_none());
    }

    #[test]
    fn test_convex_decomposition() {
        let l = lshape();
        let parts = convex_decomposition(&l);
        assert!(parts.len() >= 2 && parts.len() <= 4, "L-shape splits into few pieces");
        let total: f64 = parts.iter().map(Polygon2::area).sum();
        assert!((total - l.area()).abs() < 1e-9);
        for p in &parts {
            assert!(p.is_convex());
        }
        assert_eq!(convex_decomposition(&square(1.0)).len(), 1);
    }

    #[test]
    fn test_hulls() {
        let mut rng = Rng::new(41);
        let pts: Vec<Vec2> = (0..200)
            .map(|_| Vec2::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0))
            .collect();
        let hull = convex_hull_2d(&pts);
        assert!(hull.is_convex() && hull.is_ccw());
        for &p in &pts {
            let n = hull.vertices.len();
            for i in 0..n {
                assert!(
                    cross3(hull.vertices[i], hull.vertices[(i + 1) % n], p) >= -1e-9,
                    "point outside hull"
                );
            }
        }
        let cube: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    f64::from(i & 1) * 2.0 - 1.0,
                    f64::from((i >> 1) & 1) * 2.0 - 1.0,
                    f64::from((i >> 2) & 1) * 2.0 - 1.0,
                )
            })
            .chain(std::iter::once(Vec3::ZERO))
            .collect();
        let m = convex_hull_3d(&cube);
        assert!((m.volume() - 8.0).abs() < 1e-9);
        assert_eq!(m.vertices.len(), 8, "interior point dropped");
    }

    #[test]
    fn test_straight_skeleton_rectangle() {
        // Rectangle skeleton: 4 corner arcs meeting a horizontal
        // ridge between (-0.5, 0) and (0.5, 0) for a 3x2 rectangle...
        let r = Polygon2::new(vec![
            Vec2::new(-1.5, -1.0),
            Vec2::new(1.5, -1.0),
            Vec2::new(1.5, 1.0),
            Vec2::new(-1.5, 1.0),
        ]);
        let arcs = straight_skeleton(&r);
        assert!(!arcs.is_empty());
        // All arc endpoints are inside (or on) the rectangle.
        for s in &arcs {
            for p in [s.a, s.b] {
                assert!(p.x.abs() <= 1.5 + 1e-6 && p.y.abs() <= 1.0 + 1e-6);
            }
        }
        // The ridge endpoints (+-0.5, 0) appear among arc endpoints.
        for target in [Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0)] {
            assert!(
                arcs.iter().any(|s| s.b.distance_to(&target) < 1e-6
                    || s.a.distance_to(&target) < 1e-6),
                "missing ridge endpoint {target:?}"
            );
        }
        // Square: all four bisectors meet the center.
        let arcs = straight_skeleton(&square(1.0));
        assert!(arcs.iter().filter(|s| s.b.magnitude() < 1e-6).count() >= 3);
    }

    #[test]
    fn test_inscribed_and_enclosing_circles() {
        let sq = square(1.0);
        let inc = largest_inscribed_circle(&sq);
        assert!(inc.center.magnitude() < 1e-3);
        assert!((inc.radius - 1.0).abs() < 1e-3);
        // L-shape: the largest circle sits on the diagonal, touching
        // both outer edges and the reentrant corner: r = 2 - sqrt(2).
        let l = lshape();
        let c = largest_inscribed_circle(&l);
        assert!(
            (c.radius - (2.0 - std::f64::consts::SQRT_2)).abs() < 1e-3,
            "L-shape inradius {}",
            c.radius
        );

        let mut rng = Rng::new(42);
        let pts: Vec<Vec2> = (0..500)
            .map(|_| Vec2::new(rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 2.0 - 1.0))
            .collect();
        let c = smallest_enclosing_circle(&pts);
        for &p in &pts {
            assert!(p.distance_to(&c.center) <= c.radius + 1e-9);
        }
        // Tight: some point on the boundary.
        let on_boundary = pts
            .iter()
            .filter(|p| (p.distance_to(&c.center) - c.radius).abs() < 1e-7)
            .count();
        assert!(on_boundary >= 2);
    }

    #[test]
    fn test_calipers() {
        // A rotated 4x2 rectangle.
        let angle = 0.35;
        let base = [
            Vec2::new(-2.0, -1.0),
            Vec2::new(2.0, -1.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(-2.0, 1.0),
        ];
        let pts: Vec<Vec2> = base.iter().map(|p| p.rotate(angle)).collect();
        let (center, half, found) = minimum_bounding_rect(&pts);
        assert!(center.magnitude() < 1e-9);
        let (lo, hi) = (half.x.min(half.y), half.x.max(half.y));
        assert!((lo - 1.0).abs() < 1e-9 && (hi - 2.0).abs() < 1e-9);
        // Angle matches up to k*pi/2.
        let diff = (found - angle).rem_euclid(std::f64::consts::FRAC_PI_2);
        assert!(!(1e-9..=std::f64::consts::FRAC_PI_2 - 1e-9).contains(&diff));

        let poly = Polygon2::new(pts.clone());
        let (_, _, diam) = polygon_diameter(&poly);
        assert!((diam - 20.0f64.sqrt()).abs() < 1e-9);
        assert!((polygon_width(&poly) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_resample_chaikin_round() {
        let sq = square(1.0);
        let r = resample_polygon(&sq, 16);
        assert_eq!(r.vertices.len(), 16);
        assert!((r.perimeter() - 8.0).abs() < 1e-6);
        let step = 8.0 / 16.0;
        for i in 0..16 {
            let d = r.vertices[i].distance_to(&r.vertices[(i + 1) % 16]);
            assert!((d - step).abs() < 1e-9, "equal spacing");
        }
        let ch = smooth_chaikin(&sq, 3);
        assert_eq!(ch.vertices.len(), 4 * 8);
        assert!(ch.is_convex());
        assert!(ch.area() < sq.area());
        let rc = round_corners(&sq, 0.25, 4);
        assert!(rc.area() < sq.area());
        // Rounded square area = 4 - (4 - pi) r^2.
        let expect = 4.0 - (4.0 - std::f64::consts::PI) * 0.0625;
        assert!((rc.area() - expect).abs() < 0.01);
    }

    #[test]
    fn test_fills() {
        let sq = square(1.0);
        let hatch = hatch_fill(&sq, 0.25, 0.0);
        assert_eq!(hatch.len(), 8);
        let total: f64 = hatch.iter().map(|s| s.a.distance_to(&s.b)).sum();
        assert!((total - 16.0).abs() < 1e-9);
        // 45-degree hatches stay inside.
        for s in hatch_fill(&sq, 0.3, std::f64::consts::FRAC_PI_4) {
            for p in [s.a, s.b] {
                assert!(p.x.abs() <= 1.0 + 1e-9 && p.y.abs() <= 1.0 + 1e-9);
            }
        }
        let rings = contour_fill(&sq, 0.3);
        assert!(rings.len() >= 3);
        // Strictly shrinking areas.
        for w in rings.windows(2) {
            assert!(w[1].area() < w[0].area());
        }
    }
}
