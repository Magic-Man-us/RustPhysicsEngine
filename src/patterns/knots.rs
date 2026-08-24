//! Knots and space curves: parametric knot families, Frenet and
//! rotation-minimizing frames, curvature/torsion estimates, and the
//! classical knot invariants computable from a curve in space —
//! writhe and linking number by the Gauss integral, crossing numbers
//! of projections, and the Alexander polynomial from a knot diagram.

use crate::math::{Vec2, Vec3};
use crate::mesh::Mesh;
use crate::quaternion::Quaternion;
use crate::spatial::frame::Frame;
use crate::spatial::primitives::Polyline;

/// Point on the (p, q) torus knot at parameter `t` ∈ [0, 2π):
/// winds `p` times around the torus axis and `q` times through the
/// hole of the torus with radii `r_major` > `r_minor`.
///
/// x = (R + r cos qt) cos pt, y = (R + r cos qt) sin pt, z = r sin qt.
///
/// # Panics
/// Panics unless `p, q >= 1` and `r_major > r_minor > 0`.
#[must_use]
pub fn torus_knot(p: u32, q: u32, r_major: f64, r_minor: f64, t: f64) -> Vec3 {
    assert!(p >= 1 && q >= 1, "torus knot needs p, q >= 1");
    assert!(r_major > r_minor && r_minor > 0.0, "torus knot needs R > r > 0");
    let (pf, qf) = (f64::from(p), f64::from(q));
    let w = r_major + r_minor * (qf * t).cos();
    Vec3::new(w * (pf * t).cos(), w * (pf * t).sin(), r_minor * (qf * t).sin())
}

/// Closed polyline sampling of the (p, q) torus knot with `n`
/// vertices.
///
/// # Panics
/// Panics unless `n >= 3` (and the `torus_knot` preconditions hold).
#[must_use]
pub fn torus_knot_curve(p: u32, q: u32, r_major: f64, r_minor: f64, n: usize) -> Polyline {
    assert!(n >= 3, "polyline needs at least 3 vertices");
    let points = (0..n)
        .map(|i| torus_knot(p, q, r_major, r_minor, i as f64 / n as f64 * std::f64::consts::TAU))
        .collect();
    Polyline { points, closed: true }
}

/// Point on a Lissajous knot: x = cos(nx t + φx), y = cos(ny t + φy),
/// z = cos(nz t + φz). Coprime frequencies with generic phases give
/// knotted closed curves (e.g. (3, 2, 7) with φ = (0.7, 0.2, 0)).
#[must_use]
pub fn lissajous_knot(
    nx: u32,
    ny: u32,
    nz: u32,
    phase_x: f64,
    phase_y: f64,
    phase_z: f64,
    t: f64,
) -> Vec3 {
    Vec3::new(
        (f64::from(nx) * t + phase_x).cos(),
        (f64::from(ny) * t + phase_y).cos(),
        (f64::from(nz) * t + phase_z).cos(),
    )
}

/// The trefoil knot 3₁ in its symmetric parametrization:
/// (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t), t ∈ [0, 2π).
#[must_use]
pub fn trefoil(t: f64) -> Vec3 {
    Vec3::new(
        t.sin() + 2.0 * (2.0 * t).sin(),
        t.cos() - 2.0 * (2.0 * t).cos(),
        -(3.0 * t).sin(),
    )
}

/// The figure-eight knot 4₁:
/// ((2 + cos 2t) cos 3t, (2 + cos 2t) sin 3t, sin 4t), t ∈ [0, 2π).
#[must_use]
pub fn figure_eight_knot(t: f64) -> Vec3 {
    let w = 2.0 + (2.0 * t).cos();
    Vec3::new(w * (3.0 * t).cos(), w * (3.0 * t).sin(), (4.0 * t).sin())
}

/// The cinquefoil (Solomon's seal) knot 5₁ = (2, 5) torus knot on
/// the torus R = 2, r = 1.
#[must_use]
pub fn cinquefoil(t: f64) -> Vec3 {
    torus_knot(2, 5, 2.0, 1.0, t)
}

/// Frenet frame of a curve at `t` by central differences with step
/// `h`: x axis = unit tangent T, y = principal normal N, z =
/// binormal B = T × N. `None` where the frame is undefined (zero
/// speed or zero curvature).
///
/// # Panics
/// Panics unless `h > 0`.
#[must_use]
pub fn frenet_frame(curve: &dyn Fn(f64) -> Vec3, t: f64, h: f64) -> Option<Frame> {
    assert!(h > 0.0, "step must be positive");
    let (pm, p0, pp) = (curve(t - h), curve(t), curve(t + h));
    let d1 = (pp - pm) * (0.5 / h);
    let d2 = (pp - p0 - p0 + pm) * (1.0 / (h * h));
    let speed = d1.magnitude();
    if speed < 1e-12 {
        return None;
    }
    let tangent = d1 * (1.0 / speed);
    // Normal: acceleration component perpendicular to the tangent.
    let n = d2 - tangent * d2.dot(&tangent);
    if n.magnitude() < 1e-9 * speed * speed {
        return None;
    }
    Some(Frame::from_axes(p0, tangent, n))
}

fn vertex_tangents(pl: &Polyline) -> Vec<Vec3> {
    let n = pl.points.len();
    (0..n)
        .map(|i| {
            let prev = if i > 0 {
                pl.points[i - 1]
            } else if pl.closed {
                pl.points[n - 1]
            } else {
                pl.points[0]
            };
            let next = if i + 1 < n {
                pl.points[i + 1]
            } else if pl.closed {
                pl.points[0]
            } else {
                pl.points[n - 1]
            };
            let d = next - prev;
            if d.magnitude() > 0.0 { d.normalized() } else { Vec3::new(1.0, 0.0, 0.0) }
        })
        .collect()
}

fn any_perpendicular(v: Vec3) -> Vec3 {
    let candidate = if v.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    (candidate - v * candidate.dot(&v)).normalized()
}

/// Discrete Frenet frames at every vertex of a polyline (tangent by
/// central difference, normal from the discrete curvature vector).
/// Straight stretches inherit the previous normal so the field stays
/// continuous.
///
/// # Panics
/// Panics unless the polyline has at least 2 points.
#[must_use]
pub fn frenet_frames_polyline(pl: &Polyline) -> Vec<Frame> {
    assert!(pl.points.len() >= 2, "polyline needs at least 2 points");
    let tangents = vertex_tangents(pl);
    let n = pl.points.len();
    let mut normals: Vec<Vec3> = Vec::with_capacity(n);
    for i in 0..n {
        let prev_t = tangents[if i > 0 { i - 1 } else if pl.closed { n - 1 } else { 0 }];
        let next_t = tangents[if i + 1 < n { i + 1 } else if pl.closed { 0 } else { n - 1 }];
        let dt = next_t - prev_t;
        let normal = dt - tangents[i] * dt.dot(&tangents[i]);
        if normal.magnitude() > 1e-9 {
            normals.push(normal.normalized());
        } else if let Some(&last) = normals.last() {
            let projected = last - tangents[i] * last.dot(&tangents[i]);
            normals.push(if projected.magnitude() > 1e-12 {
                projected.normalized()
            } else {
                any_perpendicular(tangents[i])
            });
        } else {
            normals.push(any_perpendicular(tangents[i]));
        }
    }
    (0..n).map(|i| Frame::from_axes(pl.points[i], tangents[i], normals[i])).collect()
}

/// Rotation-minimizing frames along a polyline by the double
/// reflection method (Wang, Jüttler, Zheng & Liu 2008): each step
/// reflects the previous frame in the chord bisector plane and then
/// in the tangent bisector plane, which transports the normal with
/// no spurious twist (fourth-order accurate for smooth curves).
///
/// # Panics
/// Panics unless the polyline has at least 2 points.
#[must_use]
pub fn parallel_transport_frames(pl: &Polyline) -> Vec<Frame> {
    assert!(pl.points.len() >= 2, "polyline needs at least 2 points");
    let tangents = vertex_tangents(pl);
    let n = pl.points.len();
    let mut normals = Vec::with_capacity(n);
    normals.push(any_perpendicular(tangents[0]));
    for i in 0..n - 1 {
        let v1 = pl.points[i + 1] - pl.points[i];
        let c1 = v1.dot(&v1);
        let (r_next, t_next) = if c1 > 0.0 {
            let rl = normals[i] - v1 * (2.0 / c1 * v1.dot(&normals[i]));
            let tl = tangents[i] - v1 * (2.0 / c1 * v1.dot(&tangents[i]));
            let v2 = tangents[i + 1] - tl;
            let c2 = v2.dot(&v2);
            if c2 > 1e-24 {
                (rl - v2 * (2.0 / c2 * v2.dot(&rl)), tangents[i + 1])
            } else {
                (rl, tangents[i + 1])
            }
        } else {
            (normals[i], tangents[i + 1])
        };
        let projected = r_next - t_next * r_next.dot(&t_next);
        normals.push(if projected.magnitude() > 1e-12 {
            projected.normalized()
        } else {
            any_perpendicular(t_next)
        });
    }
    (0..n).map(|i| Frame::from_axes(pl.points[i], tangents[i], normals[i])).collect()
}

/// Curvature and torsion of a curve at `t` by finite differences:
/// κ = |c′ × c″| / |c′|³ and τ = (c′ × c″)·c‴ / |c′ × c″|².
///
/// # Panics
/// Panics unless `h > 0`.
#[must_use]
pub fn curvature_torsion(curve: &dyn Fn(f64) -> Vec3, t: f64, h: f64) -> (f64, f64) {
    assert!(h > 0.0, "step must be positive");
    let (pm2, pm, p0, pp, pp2) =
        (curve(t - 2.0 * h), curve(t - h), curve(t), curve(t + h), curve(t + 2.0 * h));
    let d1 = (pp - pm) * (0.5 / h);
    let d2 = (pp - p0 - p0 + pm) * (1.0 / (h * h));
    let d3 = (pp2 - pp * 2.0 + pm * 2.0 - pm2) * (0.5 / (h * h * h));
    let cx = d1.cross(&d2);
    let speed = d1.magnitude();
    let kappa = if speed > 0.0 { cx.magnitude() / speed.powi(3) } else { 0.0 };
    let tau = if cx.magnitude_squared() > 0.0 { cx.dot(&d3) / cx.magnitude_squared() } else { 0.0 };
    (kappa, tau)
}

/// Total curvature of a polyline: the sum of exterior turning angles
/// between consecutive segments. For closed knotted curves this is
/// at least 4π (Fáry-Milnor).
#[must_use]
pub fn total_curvature(pl: &Polyline) -> f64 {
    let segs = pl.segment_count();
    if segs < 2 {
        return 0.0;
    }
    let dirs: Vec<Vec3> = (0..segs)
        .map(|i| {
            let s = pl.segment(i);
            (s.b - s.a).normalized()
        })
        .collect();
    let pairs = if pl.closed { segs } else { segs - 1 };
    (0..pairs)
        .map(|i| {
            let d = dirs[i].dot(&dirs[(i + 1) % segs]).clamp(-1.0, 1.0);
            d.acos()
        })
        .sum()
}

/// Signed solid-angle contribution of the ordered segment pair
/// (p1→p2, p3→p4) to the Gauss integral, by the tetrahedral formula
/// of Klenin & Langowski 2000 (method 1a).
fn gauss_pair(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> f64 {
    let r13 = p3 - p1;
    let r14 = p4 - p1;
    let r23 = p3 - p2;
    let r24 = p4 - p2;
    let unit = |v: Vec3| {
        let m = v.magnitude();
        if m > 0.0 { Some(v * (1.0 / m)) } else { None }
    };
    let (Some(n1), Some(n2), Some(n3), Some(n4)) = (
        unit(r13.cross(&r14)),
        unit(r14.cross(&r24)),
        unit(r24.cross(&r23)),
        unit(r23.cross(&r13)),
    ) else {
        return 0.0;
    };
    let s = |a: Vec3, b: Vec3| a.dot(&b).clamp(-1.0, 1.0).asin();
    let omega = s(n1, n2) + s(n2, n3) + s(n3, n4) + s(n4, n1);
    let sign = (p4 - p3).cross(&(p2 - p1)).dot(&r13);
    // Coplanar pairs contribute nothing (the Gauss integrand is a
    // triple product of coplanar vectors); the formula's asin terms
    // degenerate there, so detect it via the same triple product.
    let scale = (p2 - p1).magnitude() * (p4 - p3).magnitude() * r13.magnitude();
    if sign.abs() <= 1e-12 * scale {
        0.0
    } else if sign > 0.0 {
        omega
    } else {
        -omega
    }
}

/// Writhe of a closed polyline: the Gauss double integral
/// Wr = (1/4π) ∮∮ (dr₁ × dr₂)·(r₁ − r₂)/|r₁ − r₂|³, evaluated
/// exactly over segment pairs by the solid-angle formula. Planar
/// curves have writhe 0.
///
/// # Panics
/// Panics unless the polyline is closed with at least 3 points.
#[must_use]
pub fn writhe(pl: &Polyline) -> f64 {
    assert!(pl.closed && pl.points.len() >= 3, "writhe needs a closed polyline");
    let n = pl.segment_count();
    let mut total = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue; // adjacent segments contribute zero
            }
            let (si, sj) = (pl.segment(i), pl.segment(j));
            total += gauss_pair(si.a, si.b, sj.a, sj.b);
        }
    }
    total / (2.0 * std::f64::consts::PI)
}

/// Linking number of two closed polylines by the Gauss double sum;
/// the result is an integer for disjoint closed curves.
///
/// # Panics
/// Panics unless both polylines are closed with at least 3 points.
#[must_use]
pub fn linking_number(a: &Polyline, b: &Polyline) -> i32 {
    assert!(a.closed && a.points.len() >= 3, "linking number needs closed curves");
    assert!(b.closed && b.points.len() >= 3, "linking number needs closed curves");
    let mut total = 0.0;
    for i in 0..a.segment_count() {
        for j in 0..b.segment_count() {
            let (si, sj) = (a.segment(i), b.segment(j));
            total += gauss_pair(si.a, si.b, sj.a, sj.b);
        }
    }
    (total / (4.0 * std::f64::consts::PI)).round() as i32
}

/// Orthonormal basis (u, v) of the plane perpendicular to `dir`.
fn projection_basis(dir: Vec3) -> (Vec3, Vec3) {
    let d = dir.normalized();
    let u = any_perpendicular(d);
    let v = d.cross(&u);
    (u, v)
}

/// A crossing of the projected diagram.
struct Crossing {
    /// Position along the curve (segment index + parameter) of the
    /// strand passing over / under.
    over_pos: f64,
    under_pos: f64,
    /// Sign of the crossing (right-handed = +1).
    sign: i32,
}

fn diagram_crossings(pl: &Polyline, direction: Vec3) -> Vec<Crossing> {
    let (u, v) = projection_basis(direction);
    let d = direction.normalized();
    let n = pl.segment_count();
    let proj: Vec<(Vec2, f64)> =
        pl.points.iter().map(|p| (Vec2::new(p.dot(&u), p.dot(&v)), p.dot(&d))).collect();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let (a1, h1a) = proj[i];
            let (a2, h1b) = proj[(i + 1) % pl.points.len()];
            let (b1, h2a) = proj[j];
            let (b2, h2b) = proj[(j + 1) % pl.points.len()];
            let da = a2 - a1;
            let db = b2 - b1;
            let denom = da.cross(&db);
            if denom.abs() < 1e-14 {
                continue;
            }
            let s = (b1 - a1).cross(&db) / denom;
            let t = (b1 - a1).cross(&da) / denom;
            if !(0.0..=1.0).contains(&s) || !(0.0..=1.0).contains(&t) {
                continue;
            }
            let hi = h1a + (h1b - h1a) * s;
            let hj = h2a + (h2b - h2a) * t;
            let (over_pos, under_pos, d_over, d_under) = if hi > hj {
                (i as f64 + s, j as f64 + t, da, db)
            } else {
                (j as f64 + t, i as f64 + s, db, da)
            };
            let sign = if d_over.cross(&d_under) > 0.0 { 1 } else { -1 };
            out.push(Crossing { over_pos, under_pos, sign });
        }
    }
    out
}

/// Number of crossings in the projection of the polyline along
/// `direction` (transverse double points of the diagram).
///
/// # Panics
/// Panics unless the polyline is closed and `direction` is non-zero.
#[must_use]
pub fn crossing_number_projection(pl: &Polyline, direction: Vec3) -> usize {
    assert!(pl.closed, "knot diagrams come from closed curves");
    assert!(direction.magnitude() > 0.0, "projection direction must be non-zero");
    diagram_crossings(pl, direction).len()
}

/// Bareiss fraction-free determinant of an integer matrix.
fn det_i128(mut m: Vec<Vec<i128>>) -> i128 {
    let n = m.len();
    if n == 0 {
        return 1;
    }
    let mut sign = 1i128;
    let mut prev = 1i128;
    for k in 0..n - 1 {
        if m[k][k] == 0 {
            let Some(swap) = (k + 1..n).find(|&r| m[r][k] != 0) else { return 0 };
            m.swap(k, swap);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                m[i][j] = (m[k][k] * m[i][j] - m[i][k] * m[k][j]) / prev;
            }
            m[i][k] = 0;
        }
        prev = m[k][k];
    }
    sign * m[n - 1][n - 1]
}

/// Alexander polynomial coefficients (lowest degree first) computed
/// from the diagram of the closed polyline projected along +z. Arcs
/// run between undercrossings; each crossing contributes the
/// abelianized Fox-derivative row of its Wirtinger relation
/// (over-arc 1 − t, incoming under-arc t, outgoing under-arc −1 for
/// a positive crossing), one row and one column are deleted, and the
/// determinant is recovered by evaluation at integer points and
/// Lagrange interpolation. Normalized so the constant term is
/// non-zero and the leading coefficient positive; the unknot (no
/// crossings) gives `[1]`.
///
/// The projection must be regular: only transverse double points.
/// Sample the curve finely enough that no segment participates in
/// two crossings with nearly equal positions.
///
/// # Panics
/// Panics unless the polyline is closed with at least 3 points.
#[must_use]
pub fn alexander_polynomial_coeffs(pl: &Polyline) -> Vec<i64> {
    assert!(pl.closed && pl.points.len() >= 3, "Alexander polynomial needs a closed curve");
    let crossings = diagram_crossings(pl, Vec3::new(0.0, 0.0, 1.0));
    let n = crossings.len();
    if n < 2 {
        return vec![1]; // unknot diagram (0 or 1 crossing)
    }
    // Arcs: pieces of the curve between consecutive undercrossings,
    // in curve order. Arc k runs from under-event k to k+1 (mod n).
    let mut under_order: Vec<usize> = (0..n).collect();
    under_order.sort_by(|&a, &b| crossings[a].under_pos.total_cmp(&crossings[b].under_pos));
    let arc_of = |pos: f64| -> usize {
        // The arc that contains position `pos`: the number of
        // under-events at position <= pos, minus one (mod n).
        let count = under_order
            .iter()
            .filter(|&&c| crossings[c].under_pos <= pos)
            .count();
        (count + n - 1) % n
    };
    // For crossing c (the k-th undercrossing along the curve), the
    // incoming under-arc is k-1..k, the outgoing is k..k+1.
    let mut under_rank = vec![0usize; n];
    for (rank, &c) in under_order.iter().enumerate() {
        under_rank[c] = rank;
    }
    // Row entries as polynomials in t: coefficient arrays [c0, c1].
    let mut rows: Vec<Vec<[i64; 2]>> = vec![vec![[0, 0]; n]; n];
    for (c, crossing) in crossings.iter().enumerate() {
        let over = arc_of(crossing.over_pos);
        let incoming = (under_rank[c] + n - 1) % n;
        let outgoing = under_rank[c];
        let row = &mut rows[c];
        if crossing.sign > 0 {
            // 1 - t at the over-arc, t at incoming, -1 at outgoing.
            row[over][0] += 1;
            row[over][1] -= 1;
            row[incoming][1] += 1;
            row[outgoing][0] -= 1;
        } else {
            // t - 1 at the over-arc, 1 at incoming, -t at outgoing.
            row[over][0] -= 1;
            row[over][1] += 1;
            row[incoming][0] += 1;
            row[outgoing][1] -= 1;
        }
    }
    // Delete the last row and column; determinant by evaluation at
    // m+1 integer points and Lagrange interpolation (degree <= n-1).
    let size = n - 1;
    let degree = size; // each entry is degree <= 1
    let xs: Vec<i128> = (0..=degree as i128).map(|k| k + 2).collect();
    let ys: Vec<i128> = xs
        .iter()
        .map(|&x| {
            let m: Vec<Vec<i128>> = (0..size)
                .map(|i| {
                    (0..size)
                        .map(|j| i128::from(rows[i][j][0]) + i128::from(rows[i][j][1]) * x)
                        .collect()
                })
                .collect();
            det_i128(m)
        })
        .collect();
    // Lagrange interpolation with exact rational arithmetic; the
    // result is an integer polynomial.
    let m = xs.len();
    let mut num = vec![0i128; m]; // accumulated numerator coefficients
    let mut den = 1i128;
    for k in 0..m {
        // Basis polynomial numerator: prod_{j != k} (x - xs[j]).
        let mut basis = vec![0i128; m];
        basis[0] = 1;
        let mut deg = 0;
        let mut basis_den = 1i128;
        for j in 0..m {
            if j == k {
                continue;
            }
            basis_den *= xs[k] - xs[j];
            for d in (0..=deg).rev() {
                basis[d + 1] += basis[d];
                basis[d] *= -xs[j];
            }
            deg += 1;
        }
        // num/den += ys[k] * basis / basis_den
        let new_den = den * basis_den;
        for c in num.iter_mut() {
            *c *= basis_den;
        }
        for (c, b) in num.iter_mut().zip(&basis) {
            *c += ys[k] * b * den;
        }
        den = new_den;
    }
    let mut coeffs: Vec<i64> = num
        .iter()
        .map(|&c| {
            assert!(c % den == 0, "Alexander determinant interpolation is integral");
            i64::try_from(c / den).expect("Alexander coefficients fit in i64")
        })
        .collect();
    // Normalize up to +/- t^k: strip trailing zeros above the top
    // degree, shift out powers of t, fix the sign.
    while coeffs.len() > 1 && *coeffs.last().unwrap() == 0 {
        coeffs.pop();
    }
    let shift = coeffs.iter().take_while(|&&c| c == 0).count();
    coeffs.drain(..shift.min(coeffs.len().saturating_sub(1)));
    if coeffs.iter().all(|&c| c == 0) {
        return vec![0];
    }
    if *coeffs.last().unwrap() < 0 {
        for c in &mut coeffs {
            *c = -*c;
        }
    }
    coeffs
}

/// Sweeps a circle of `radius` along the polyline (delegates to
/// `mesh::generate::tube_along_polyline`).
#[must_use]
pub fn knot_tube(pl: &Polyline, radius: f64, segments: usize) -> Mesh {
    crate::mesh::generate::tube_along_polyline(pl, radius, segments)
}

/// Circular helix of given radius, pitch (rise per turn), and number
/// of turns, sampled at `n` points.
///
/// # Panics
/// Panics unless `radius > 0`, `turns > 0`, and `n >= 2`.
#[must_use]
pub fn helix(radius: f64, pitch: f64, turns: f64, n: usize) -> Polyline {
    assert!(radius > 0.0 && turns > 0.0, "helix needs radius > 0 and turns > 0");
    assert!(n >= 2, "polyline needs at least 2 points");
    let points = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64 * turns * std::f64::consts::TAU;
            Vec3::new(
                radius * t.cos(),
                radius * t.sin(),
                pitch * t / std::f64::consts::TAU,
            )
        })
        .collect();
    Polyline { points, closed: false }
}

/// Two helices on the same axis separated by `phase` radians (DNA
/// uses phase ≈ 2.1 rad for the minor/major groove asymmetry).
#[must_use]
pub fn double_helix(
    radius: f64,
    pitch: f64,
    turns: f64,
    n: usize,
    phase: f64,
) -> (Polyline, Polyline) {
    let first = helix(radius, pitch, turns, n);
    let rot = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), phase);
    let points = first.points.iter().map(|&p| rot.rotate_vec(p)).collect();
    (first, Polyline { points, closed: false })
}

/// Viviani's curve: the intersection of the sphere of radius 2a with
/// the cylinder of radius a tangent to its vertical axis:
/// (a(1 + cos t), a sin t, 2a sin(t/2)), t ∈ [0, 4π) for the full
/// figure-eight.
///
/// # Panics
/// Panics unless `a > 0`.
#[must_use]
pub fn viviani_curve(a: f64, t: f64) -> Vec3 {
    assert!(a > 0.0, "Viviani curve needs a > 0");
    Vec3::new(a * (1.0 + t.cos()), a * t.sin(), 2.0 * a * (0.5 * t).sin())
}

/// Tennis-ball seam curve: (a cos t + b cos 3t, a sin t − b sin 3t,
/// 2 √(ab) sin 2t) lies on the sphere of radius a + b.
///
/// # Panics
/// Panics unless `a, b > 0`.
#[must_use]
pub fn tennis_ball_curve(a: f64, b: f64, t: f64) -> Vec3 {
    assert!(a > 0.0 && b > 0.0, "tennis ball curve needs a, b > 0");
    Vec3::new(
        a * t.cos() + b * (3.0 * t).cos(),
        a * t.sin() - b * (3.0 * t).sin(),
        2.0 * (a * b).sqrt() * (2.0 * t).sin(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_torus_knot_on_torus() {
        for i in 0..200 {
            let t = i as f64 / 200.0 * TAU;
            let p = torus_knot(2, 3, 3.0, 1.0, t);
            // Distance to the torus surface (R = 3, r = 1).
            let ring = (p.x * p.x + p.y * p.y).sqrt() - 3.0;
            let d = (ring * ring + p.z * p.z).sqrt() - 1.0;
            assert!(d.abs() < 1e-9, "torus knot on torus ({d})");
        }
        let c = torus_knot_curve(2, 3, 3.0, 1.0, 100);
        assert!(c.closed);
        assert_eq!(c.points.len(), 100);
    }

    #[test]
    fn test_helix_curvature_torsion() {
        let (r, pitch) = (2.0, 3.0);
        let c = pitch / TAU;
        let curve = |t: f64| Vec3::new(r * t.cos(), r * t.sin(), c * t);
        let denom = r * r + c * c;
        for i in 0..10 {
            let t = i as f64 * 0.7;
            // h large enough that the third-difference roundoff
            // (~eps/h^3) stays below the truncation error.
            let (kappa, tau) = curvature_torsion(&curve, t, 1e-3);
            assert!((kappa - r / denom).abs() < 1e-5, "helix curvature {kappa}");
            assert!((tau - c / denom).abs() < 1e-5, "helix torsion {tau}");
            let f = frenet_frame(&curve, t, 1e-4).expect("helix frame exists");
            // Principal normal of a helix points inward horizontally.
            let n = f.to_world_vector(Vec3::new(0.0, 1.0, 0.0));
            let inward = Vec3::new(-t.cos(), -t.sin(), 0.0);
            assert!((n - inward).magnitude() < 1e-5);
        }
    }

    #[test]
    fn test_parallel_transport_zero_twist() {
        let pl = torus_knot_curve(2, 3, 3.0, 1.0, 2000);
        let frames = parallel_transport_frames(&pl);
        assert_eq!(frames.len(), pl.points.len());
        for w in frames.windows(2) {
            let t_next = w[1].to_world_vector(Vec3::new(1.0, 0.0, 0.0));
            let n0 = w[0].to_world_vector(Vec3::new(0.0, 1.0, 0.0));
            let n1 = w[1].to_world_vector(Vec3::new(0.0, 1.0, 0.0));
            // Twist: angle between the transported previous normal
            // and the next normal, measured in the next normal plane.
            let transported = (n0 - t_next * n0.dot(&t_next)).normalized();
            let angle = transported.dot(&n1).clamp(-1.0, 1.0).acos();
            assert!(angle < 1e-4, "twist {angle} between adjacent frames");
        }
        // Frenet frames also exist everywhere on this knot.
        let ff = frenet_frames_polyline(&pl);
        assert_eq!(ff.len(), pl.points.len());
    }

    #[test]
    fn test_total_curvature_fary_milnor() {
        // Circle: total curvature 2 pi.
        let circle = Polyline {
            points: (0..200)
                .map(|i| {
                    let t = i as f64 / 200.0 * TAU;
                    Vec3::new(t.cos(), t.sin(), 0.0)
                })
                .collect(),
            closed: true,
        };
        assert!((total_curvature(&circle) - TAU).abs() < 1e-3);
        // Knotted curve: > 4 pi (Fary-Milnor).
        let tref = Polyline {
            points: (0..400).map(|i| trefoil(i as f64 / 400.0 * TAU)).collect(),
            closed: true,
        };
        assert!(total_curvature(&tref) > 2.0 * TAU);
    }

    #[test]
    fn test_writhe_planar_and_linking() {
        let circle = Polyline {
            points: (0..64)
                .map(|i| {
                    let t = i as f64 / 64.0 * TAU;
                    Vec3::new(t.cos(), t.sin(), 0.0)
                })
                .collect(),
            closed: true,
        };
        assert!(writhe(&circle).abs() < 1e-9, "planar curve has zero writhe");
        // Hopf link: two unit circles through each other's centers.
        let other = Polyline {
            points: (0..64)
                .map(|i| {
                    let t = i as f64 / 64.0 * TAU;
                    Vec3::new(1.0 + t.cos(), 0.0, t.sin())
                })
                .collect(),
            closed: true,
        };
        assert_eq!(linking_number(&circle, &other).abs(), 1, "Hopf link");
        // Distant circles are unlinked.
        let far = Polyline {
            points: other.points.iter().map(|&p| p + Vec3::new(10.0, 0.0, 0.0)).collect(),
            closed: true,
        };
        assert_eq!(linking_number(&circle, &far), 0);
    }

    #[test]
    fn test_crossing_numbers() {
        let tref = Polyline {
            points: (0..600).map(|i| trefoil(i as f64 / 600.0 * TAU)).collect(),
            closed: true,
        };
        // The standard trefoil projection along z has 3 crossings.
        assert_eq!(crossing_number_projection(&tref, Vec3::new(0.0, 0.0, 1.0)), 3);
        let fig8 = Polyline {
            points: (0..800).map(|i| figure_eight_knot(i as f64 / 800.0 * TAU)).collect(),
            closed: true,
        };
        // This parametrization projects with more than the minimal 4
        // crossings, but always an even count for a generic diagram
        // of a closed curve... not guaranteed; just regularity:
        assert!(crossing_number_projection(&fig8, Vec3::new(0.0, 0.0, 1.0)) >= 4);
    }

    #[test]
    fn test_alexander_polynomials() {
        let tref = Polyline {
            points: (0..600).map(|i| trefoil(i as f64 / 600.0 * TAU)).collect(),
            closed: true,
        };
        assert_eq!(alexander_polynomial_coeffs(&tref), vec![1, -1, 1], "trefoil t^2 - t + 1");
        let circle = Polyline {
            points: (0..64)
                .map(|i| {
                    let t = i as f64 / 64.0 * TAU;
                    Vec3::new(t.cos(), t.sin(), (2.0 * t).sin() * 0.01)
                })
                .collect(),
            closed: true,
        };
        assert_eq!(alexander_polynomial_coeffs(&circle), vec![1], "unknot");
        let cinq = Polyline {
            points: (0..800).map(|i| cinquefoil(i as f64 / 800.0 * TAU)).collect(),
            closed: true,
        };
        assert_eq!(
            alexander_polynomial_coeffs(&cinq),
            vec![1, -1, 1, -1, 1],
            "cinquefoil t^4 - t^3 + t^2 - t + 1"
        );
    }

    #[test]
    fn test_named_curves_and_tube() {
        for i in 0..100 {
            let t = i as f64 / 100.0 * 2.0 * TAU;
            let v = viviani_curve(1.5, t);
            // On the sphere of radius 2a about the origin.
            assert!((v.magnitude() - 3.0).abs() < 1e-9, "Viviani on sphere");
            let s = tennis_ball_curve(2.0, 1.0, t);
            assert!((s.magnitude() - 3.0).abs() < 1e-9, "tennis ball on sphere");
        }
        let h = helix(1.0, 0.5, 3.0, 100);
        assert_eq!(h.points.len(), 100);
        assert!((h.points[99].z - 1.5).abs() < 1e-12, "pitch 0.5 over 3 turns");
        let (a, b) = double_helix(1.0, 0.5, 2.0, 50, std::f64::consts::PI);
        assert!((a.points[0] + b.points[0]).magnitude() < 1e-12, "opposite start points");
        let tube = knot_tube(&torus_knot_curve(2, 3, 3.0, 1.0, 60), 0.3, 8);
        assert!(!tube.vertices.is_empty());
        let liss = lissajous_knot(3, 2, 7, 0.7, 0.2, 0.0, 1.0);
        assert!(liss.x.abs() <= 1.0 && liss.y.abs() <= 1.0 && liss.z.abs() <= 1.0);
    }
}
