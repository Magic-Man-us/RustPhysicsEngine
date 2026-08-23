//! Aperiodic tilings: Penrose P2 (kite/dart) and P3 (rhombs) by
//! Robinson-triangle deflation, de Bruijn multigrid projection,
//! Ammann-Beenker, the hat and spectre monotiles (ported from the
//! reference implementations accompanying Smith, Myers, Kaplan &
//! Goodman-Strauss 2023), the pinwheel tiling, and 1-D quasiperiodic
//! sequences.

use crate::math::Vec2;
use crate::spatial::primitives::{Polygon2, Rect};

const PHI: f64 = 1.618_033_988_749_895;

/// Penrose tile kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PenroseTile {
    Kite,
    Dart,
    ThinRhomb,
    ThickRhomb,
}

/// A placed Penrose tile. Vertices are ordered so positions 1 and 3
/// are the tile's internal axis/diagonal (the symmetry axis for
/// kites and darts, the splitting diagonal for rhombs).
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedTile {
    pub kind: PenroseTile,
    pub vertices: Vec<Vec2>,
}

/// Robinson half-tile in apex-first convention: `acute` triangles
/// (36°-72°-72°) have the apex (36°) first and equal legs; `gnomon`
/// triangles (36°-36°-108°) have the apex (108°) first and equal
/// short sides.
#[derive(Clone, Copy)]
struct Half {
    gnomon: bool,
    a: Vec2,
    b: Vec2,
    c: Vec2,
}

/// P2 half-tile carrying the matching-rule decoration as a boolean
/// "color" per vertex (Robinson's vertex 2-coloring). Acute halves
/// (36-72-72, half a kite) keep the 36° apex in `a`; obtuse halves
/// (36-36-108, half a dart) keep the 36° vertex on the dart axis in
/// `a`. The colors decide where each subdivision cut goes, which is
/// what keeps children of neighboring halves mirror-compatible: P2
/// deflation without the decoration produces geometrically valid
/// triangles that violate the matching rules and cannot be merged
/// back into kites and darts.
#[derive(Clone, Copy)]
struct P2Half {
    obtuse: bool,
    a: (Vec2, bool),
    b: (Vec2, bool),
    c: (Vec2, bool),
}

/// One round of P2 (kite/dart) deflation on decorated half-tiles.
/// Acute -> two acute + one obtuse; obtuse -> one of each. The cut
/// choices and the recoloring follow the standard decorated
/// substitution (see e.g. the PenroseGenerator reference
/// implementation); edge lengths shrink by 1/φ.
fn deflate_p2(halves: &[P2Half]) -> Vec<P2Half> {
    let mut out = Vec::with_capacity(halves.len() * 3);
    for h in halves {
        let (a, ca) = h.a;
        if !h.obtuse {
            // Pick the leg toward the base vertex whose color equals
            // the apex color for the long cut P (at the base length
            // from A) and the other leg for the short cut Q (at base
            // length / φ).
            let ((pb, _), (qb, _)) = if h.b.1 == ca { (h.b, h.c) } else { (h.c, h.b) };
            let sz = pb.distance_to(&qb);
            let p = a + (pb - a) * (sz / a.distance_to(&pb));
            let q = a + (qb - a) * (sz / (PHI * a.distance_to(&qb)));
            let vp = (p, ca);
            let vq = (q, !ca);
            out.push(P2Half {
                obtuse: false,
                a: (qb, ca),
                b: vq,
                c: vp,
            });
            out.push(P2Half {
                obtuse: false,
                a: (qb, ca),
                b: (pb, !ca),
                c: vp,
            });
            out.push(P2Half {
                obtuse: true,
                a: (a, !ca),
                b: vp,
                c: vq,
            });
        } else {
            // Cut the long side A->bisect (the base) at 1/φ, where
            // `bisect` is the base vertex colored unlike A.
            let (bisect, unmodified) = if h.b.1 != ca { (h.b, h.c) } else { (h.c, h.b) };
            let vp = (a + (bisect.0 - a) * (1.0 / PHI), bisect.1);
            out.push(P2Half {
                obtuse: true,
                a: bisect,
                b: vp,
                c: unmodified,
            });
            out.push(P2Half {
                obtuse: false,
                a: (a, ca),
                b: vp,
                c: unmodified,
            });
        }
    }
    out
}

fn deflate_p3(halves: &[Half]) -> Vec<Half> {
    let mut out = Vec::with_capacity(halves.len() * 3);
    for h in halves {
        if !h.gnomon {
            // Half-thin: apex A, legs AB and AC (rhomb edges).
            let p = h.a + (h.b - h.a) * (1.0 / PHI);
            out.push(Half {
                gnomon: false,
                a: h.c,
                b: p,
                c: h.b,
            });
            out.push(Half {
                gnomon: true,
                a: p,
                b: h.c,
                c: h.a,
            });
        } else {
            // Half-thick: apex A (108°), base BC (long diagonal).
            let q = h.b + (h.a - h.b) * (1.0 / PHI);
            let r = h.b + (h.c - h.b) * (1.0 / PHI);
            out.push(Half {
                gnomon: true,
                a: r,
                b: h.c,
                c: h.a,
            });
            out.push(Half {
                gnomon: true,
                a: q,
                b: r,
                c: h.b,
            });
            out.push(Half {
                gnomon: false,
                a: r,
                b: q,
                c: h.a,
            });
        }
    }
    out
}

fn quantize(p: Vec2, q: f64) -> (i64, i64) {
    ((p.x / q).round() as i64, (p.y / q).round() as i64)
}

/// Half-tile indices bucketed by a quantized undirected edge plus
/// the gnomon/obtuse flag.
type EdgeBuckets = std::collections::HashMap<((i64, i64), (i64, i64), bool), Vec<usize>>;

/// Interior angles of a quad respecting winding (reflex angles come
/// out > π).
fn quad_angles(v: &[Vec2; 4]) -> [f64; 4] {
    let ccw = {
        let mut s = 0.0;
        for i in 0..4 {
            s += v[i].cross(&v[(i + 1) % 4]);
        }
        s > 0.0
    };
    let mut out = [0.0; 4];
    for i in 0..4 {
        let p = v[(i + 3) % 4] - v[i];
        let n = v[(i + 1) % 4] - v[i];
        let ang = n.angle_between(&p);
        let convex = if ccw {
            n.cross(&p) > 0.0
        } else {
            n.cross(&p) < 0.0
        };
        out[i] = if convex {
            ang
        } else {
            2.0 * std::f64::consts::PI - ang
        };
    }
    out
}

fn signature_matches(kind: PenroseTile, angles: &[f64; 4]) -> bool {
    let d = std::f64::consts::PI / 180.0;
    let close = |a: f64, deg: f64| (a - deg * d).abs() < 1e-6;
    let count = |deg: f64| angles.iter().filter(|&&a| close(a, deg)).count();
    match kind {
        PenroseTile::Kite => count(72.0) == 3 && count(144.0) == 1,
        PenroseTile::Dart => count(36.0) == 2 && count(72.0) == 1 && count(216.0) == 1,
        PenroseTile::ThinRhomb => count(36.0) == 2 && count(144.0) == 2,
        PenroseTile::ThickRhomb => count(72.0) == 2 && count(108.0) == 2,
    }
}

/// Merges mirror-pair half-tiles into full tiles. Pairs share their
/// axis edge and produce the correct angle signature; each half is
/// used once.
fn merge_halves(halves: &[Half], p2: bool) -> Vec<PlacedTile> {
    // Scale-aware quantum from the smallest side.
    let scale = halves
        .iter()
        .map(|h| h.a.distance_to(&h.b).min(h.a.distance_to(&h.c)))
        .fold(f64::INFINITY, f64::min);
    let q = scale * 1e-6;
    let mut by_edge: EdgeBuckets = std::collections::HashMap::new();
    for (i, h) in halves.iter().enumerate() {
        for (u, v) in [(h.a, h.b), (h.b, h.c), (h.c, h.a)] {
            let (ku, kv) = (quantize(u, q), quantize(v, q));
            let key = (ku.min(kv), ku.max(kv), h.gnomon);
            by_edge.entry(key).or_default().push(i);
        }
    }
    let mut used = vec![false; halves.len()];
    let mut out = Vec::new();
    for (i, h) in halves.iter().enumerate() {
        if used[i] {
            continue;
        }
        'edges: for (u, v) in [(h.a, h.b), (h.b, h.c), (h.c, h.a)] {
            let (ku, kv) = (quantize(u, q), quantize(v, q));
            let key = (ku.min(kv), ku.max(kv), h.gnomon);
            let Some(candidates) = by_edge.get(&key) else {
                continue;
            };
            for &j in candidates {
                if j == i || used[j] {
                    continue;
                }
                let g = halves[j];
                // The two non-shared vertices.
                let others = |h: &Half| -> Vec2 {
                    for p in [h.a, h.b, h.c] {
                        if quantize(p, q) != ku && quantize(p, q) != kv {
                            return p;
                        }
                    }
                    unreachable!("triangle has a non-shared vertex")
                };
                let (x1, x2) = (others(h), others(&g));
                if x1.distance_to(&x2) < q {
                    continue; // identical twin, not a mirror
                }
                let quad = [x1, u, x2, v];
                let angles = quad_angles(&quad);
                let kind = if h.gnomon {
                    if p2 {
                        PenroseTile::Dart
                    } else {
                        PenroseTile::ThickRhomb
                    }
                } else if p2 {
                    PenroseTile::Kite
                } else {
                    PenroseTile::ThinRhomb
                };
                if signature_matches(kind, &angles) {
                    used[i] = true;
                    used[j] = true;
                    // Counterclockwise output.
                    let mut verts = quad.to_vec();
                    if Polygon2::new(verts.clone()).area_signed() < 0.0 {
                        verts.reverse();
                        verts.rotate_right(1); // keep axis at 1 and 3
                    }
                    out.push(PlacedTile {
                        kind,
                        vertices: verts,
                    });
                    break 'edges;
                }
            }
        }
    }
    out
}

/// Merges decorated P2 halves into kites and darts. The decoration
/// makes the pairing unambiguous: in every half exactly one edge has
/// equal endpoint colors, and that edge is the axis its mirror twin
/// shares (greedy geometric pairing can join two halves apex-to-apex
/// across an edge that is not a kite axis, stranding their real
/// twins). Halves whose twin lies outside the patch are dropped.
fn merge_halves_p2(halves: &[P2Half]) -> Vec<PlacedTile> {
    let scale = halves
        .iter()
        .map(|h| h.a.0.distance_to(&h.b.0).min(h.a.0.distance_to(&h.c.0)))
        .fold(f64::INFINITY, f64::min);
    let q = scale * 1e-6;
    // The axis edge (equal endpoint colors) and the off-axis vertex.
    let axis = |h: &P2Half| -> ((Vec2, Vec2), Vec2) {
        if h.a.1 == h.b.1 && h.a.1 != h.c.1 {
            ((h.a.0, h.b.0), h.c.0)
        } else if h.a.1 == h.c.1 && h.a.1 != h.b.1 {
            ((h.a.0, h.c.0), h.b.0)
        } else {
            ((h.b.0, h.c.0), h.a.0)
        }
    };
    let mut by_axis: EdgeBuckets = std::collections::HashMap::new();
    for (i, h) in halves.iter().enumerate() {
        let ((u, v), _) = axis(h);
        let (ku, kv) = (quantize(u, q), quantize(v, q));
        by_axis
            .entry((ku.min(kv), ku.max(kv), h.obtuse))
            .or_default()
            .push(i);
    }
    let mut out = Vec::new();
    for (i, h) in halves.iter().enumerate() {
        let ((u, v), x1) = axis(h);
        let (ku, kv) = (quantize(u, q), quantize(v, q));
        let key = (ku.min(kv), ku.max(kv), h.obtuse);
        let twins = &by_axis[&key];
        // Emit once per pair, from the lower index.
        let Some(&j) = twins.iter().find(|&&j| j > i) else {
            continue;
        };
        if twins.iter().any(|&k| k < i) {
            continue;
        }
        let (_, x2) = axis(&halves[j]);
        let quad = [x1, u, x2, v];
        let angles = quad_angles(&quad);
        let kind = if h.obtuse {
            PenroseTile::Dart
        } else {
            PenroseTile::Kite
        };
        assert!(
            signature_matches(kind, &angles),
            "decorated halves sharing an axis edge form a {kind:?}"
        );
        let mut verts = quad.to_vec();
        if Polygon2::new(verts.clone()).area_signed() < 0.0 {
            verts.reverse();
            verts.rotate_right(1); // keep axis at 1 and 3
        }
        out.push(PlacedTile {
            kind,
            vertices: verts,
        });
    }
    out
}

/// Splits placed tiles back into apex-first halves.
fn split_tiles(tiles: &[PlacedTile]) -> Vec<Half> {
    let mut out = Vec::with_capacity(tiles.len() * 2);
    for t in tiles {
        assert_eq!(t.vertices.len(), 4, "Penrose tiles are quadrilaterals");
        let v: [Vec2; 4] = [t.vertices[0], t.vertices[1], t.vertices[2], t.vertices[3]];
        let angles = quad_angles(&v);
        assert!(
            signature_matches(t.kind, &angles),
            "tile vertices do not match its kind (axis must be at positions 1 and 3)"
        );
        let d = std::f64::consts::PI / 180.0;
        match t.kind {
            PenroseTile::Kite => {
                // Apex (36° halves) is the axis corner with angle 72°.
                let (apex, other) = if (angles[1] - 72.0 * d).abs() < 1e-6 {
                    (v[1], v[3])
                } else {
                    (v[3], v[1])
                };
                out.push(Half {
                    gnomon: false,
                    a: apex,
                    b: v[0],
                    c: other,
                });
                out.push(Half {
                    gnomon: false,
                    a: apex,
                    b: v[2],
                    c: other,
                });
            }
            PenroseTile::Dart => {
                // Apex (108° halves) is the reflex axis corner.
                let (apex, tip) = if angles[1] > std::f64::consts::PI {
                    (v[1], v[3])
                } else {
                    (v[3], v[1])
                };
                out.push(Half {
                    gnomon: true,
                    a: apex,
                    b: v[0],
                    c: tip,
                });
                out.push(Half {
                    gnomon: true,
                    a: apex,
                    b: v[2],
                    c: tip,
                });
            }
            PenroseTile::ThinRhomb => {
                out.push(Half {
                    gnomon: false,
                    a: v[0],
                    b: v[1],
                    c: v[3],
                });
                out.push(Half {
                    gnomon: false,
                    a: v[2],
                    b: v[3],
                    c: v[1],
                });
            }
            PenroseTile::ThickRhomb => {
                out.push(Half {
                    gnomon: true,
                    a: v[0],
                    b: v[1],
                    c: v[3],
                });
                out.push(Half {
                    gnomon: true,
                    a: v[2],
                    b: v[3],
                    c: v[1],
                });
            }
        }
    }
    out
}

/// Splits kites and darts into decorated halves, assigning each
/// tile the canonical P2 vertex coloring (every kite carries the
/// same decoration, likewise every dart): kite apex and axis end
/// false, wings true; dart axis vertices true, wings false. Patches
/// produced by this module's seeds and deflations recolor
/// consistently; arbitrary hand-built patches must respect the
/// matching rules for their deflation to merge fully.
fn split_tiles_p2(tiles: &[PlacedTile]) -> Vec<P2Half> {
    let mut out = Vec::with_capacity(tiles.len() * 2);
    for t in tiles {
        assert_eq!(t.vertices.len(), 4, "Penrose tiles are quadrilaterals");
        let v: [Vec2; 4] = [t.vertices[0], t.vertices[1], t.vertices[2], t.vertices[3]];
        let angles = quad_angles(&v);
        assert!(
            signature_matches(t.kind, &angles),
            "tile vertices do not match its kind (axis must be at positions 1 and 3)"
        );
        let d = std::f64::consts::PI / 180.0;
        match t.kind {
            PenroseTile::Kite => {
                // Axis runs from the 72° apex to the 144° corner.
                let (apex, tail) = if (angles[1] - 72.0 * d).abs() < 1e-6 {
                    (v[1], v[3])
                } else {
                    (v[3], v[1])
                };
                for wing in [v[0], v[2]] {
                    out.push(P2Half {
                        obtuse: false,
                        a: (apex, false),
                        b: (wing, true),
                        c: (tail, false),
                    });
                }
            }
            PenroseTile::Dart => {
                // Axis runs from the 72° tip to the 216° reflex corner.
                let (tip, reflex) = if angles[1] > std::f64::consts::PI {
                    (v[3], v[1])
                } else {
                    (v[1], v[3])
                };
                for wing in [v[0], v[2]] {
                    out.push(P2Half {
                        obtuse: true,
                        a: (tip, true),
                        b: (wing, false),
                        c: (reflex, true),
                    });
                }
            }
            PenroseTile::ThinRhomb | PenroseTile::ThickRhomb => {
                panic!("P2 deflation takes kites and darts")
            }
        }
    }
    out
}

/// Deflates P2 (kite/dart) tiles `iterations` times; each round
/// shrinks edges by 1/φ.
#[must_use]
pub fn penrose_p2_deflate(tiles: &[PlacedTile], iterations: usize) -> Vec<PlacedTile> {
    let mut halves = split_tiles_p2(tiles);
    for _ in 0..iterations {
        halves = deflate_p2(&halves);
    }
    merge_halves_p2(&halves)
}

/// Deflates P3 (rhomb) tiles `iterations` times.
#[must_use]
pub fn penrose_p3_deflate(tiles: &[PlacedTile], iterations: usize) -> Vec<PlacedTile> {
    let mut halves = split_tiles(tiles);
    for _ in 0..iterations {
        halves = deflate_p3(&halves);
    }
    merge_halves(&halves, false)
}

fn wheel(radius: f64, gnomon: bool, star: bool) -> Vec<Half> {
    (0..10)
        .map(|i| {
            let a1 = (2 * i - 1) as f64 * std::f64::consts::PI / 10.0;
            let a2 = (2 * i + 1) as f64 * std::f64::consts::PI / 10.0;
            let mut b = Vec2::new(radius * a1.cos(), radius * a1.sin());
            let mut c = Vec2::new(radius * a2.cos(), radius * a2.sin());
            if (i % 2 == 0) != star {
                std::mem::swap(&mut b, &mut c);
            }
            Half {
                gnomon,
                a: Vec2::ZERO,
                b,
                c,
            }
        })
        .collect()
}

/// P3 "sun" seed: ten half-thin triangles around the origin (rhomb
/// edges of length `radius`).
///
/// # Panics
/// Panics unless `radius > 0`.
#[must_use]
pub fn penrose_p3_sun(radius: f64) -> Vec<PlacedTile> {
    assert!(radius > 0.0, "radius must be positive");
    merge_halves(&deflate_p3(&wheel(radius, false, false)), false)
}

/// P3 "star" seed: the mirrored wheel, deflated once so full rhombs
/// exist.
///
/// # Panics
/// Panics unless `radius > 0`.
#[must_use]
pub fn penrose_p3_star(radius: f64) -> Vec<PlacedTile> {
    assert!(radius > 0.0, "radius must be positive");
    merge_halves(&deflate_p3(&wheel(radius, false, true)), false)
}

/// P2 "sun" seed: five kites around the origin (kite long edges of
/// length `radius`).
///
/// # Panics
/// Panics unless `radius > 0`.
#[must_use]
pub fn penrose_p2_seed(radius: f64) -> Vec<PlacedTile> {
    assert!(radius > 0.0, "radius must be positive");
    merge_halves(&wheel(radius, false, false), true)
}

/// Number ratio of thick rhombs (plus kites) to thin rhombs (plus
/// darts); converges to φ under deflation.
///
/// # Panics
/// Panics when the denominator count is zero.
#[must_use]
pub fn ratio_thick_to_thin(tiles: &[PlacedTile]) -> f64 {
    let thick = tiles
        .iter()
        .filter(|t| matches!(t.kind, PenroseTile::ThickRhomb | PenroseTile::Kite))
        .count();
    let thin = tiles.len() - thick;
    assert!(thin > 0, "no thin tiles to compare against");
    thick as f64 / thin as f64
}

/// Generic de Bruijn multigrid dual: `dirs` families of parallel
/// lines with normals at angles `theta(j)` and offsets `gamma[j]`;
/// every line intersection dualizes to a rhomb.
fn multigrid_rhombs(
    angles: &[f64],
    gamma: &[f64],
    extent: &Rect,
) -> Vec<(usize, usize, Vec<Vec2>)> {
    let dirs: Vec<Vec2> = angles
        .iter()
        .map(|&a| Vec2::new(a.cos(), a.sin()))
        .collect();
    let n = dirs.len();
    let radius = extent
        .corners()
        .iter()
        .map(Vec2::magnitude)
        .fold(0.0f64, f64::max);
    let range = radius.ceil() as i64 + 2;
    let mut out = Vec::new();
    for j in 0..n {
        for k in j + 1..n {
            let det = dirs[j].cross(&dirs[k]);
            if det.abs() < 1e-12 {
                continue;
            }
            for a in -range..=range {
                for b in -range..=range {
                    // Solve p·e_j = a + γ_j, p·e_k = b + γ_k.
                    let (ra, rb) = (a as f64 + gamma[j], b as f64 + gamma[k]);
                    let p = Vec2::new(
                        (ra * dirs[k].y - rb * dirs[j].y) / det,
                        (rb * dirs[j].x - ra * dirs[k].x) / det,
                    );
                    if p.magnitude() > radius + 1.0 {
                        continue;
                    }
                    // Dual rhomb vertex indices.
                    let base: Vec<f64> = (0..n)
                        .map(|m| {
                            if m == j || m == k {
                                0.0
                            } else {
                                (p.dot(&dirs[m]) - gamma[m]).ceil()
                            }
                        })
                        .collect();
                    let corner = |dj: f64, dk: f64| -> Vec2 {
                        let mut v = Vec2::ZERO;
                        for m in 0..n {
                            let km = if m == j {
                                a as f64 + dj
                            } else if m == k {
                                b as f64 + dk
                            } else {
                                base[m]
                            };
                            v = v + dirs[m] * km;
                        }
                        v
                    };
                    let quad = vec![
                        corner(0.0, 0.0),
                        corner(1.0, 0.0),
                        corner(1.0, 1.0),
                        corner(0.0, 1.0),
                    ];
                    let centroid = (quad[0] + quad[1] + quad[2] + quad[3]) * 0.25;
                    if extent.contains_point(centroid) {
                        out.push((j, k, quad));
                    }
                }
            }
        }
    }
    out
}

/// Penrose P3 rhombs by de Bruijn's pentagrid projection: five line
/// grids with the given offsets (their sum should be an integer for a
/// true Penrose tiling; generic values give a generalized tiling).
#[must_use]
pub fn penrose_by_projection(extent: &Rect, offsets: [f64; 5]) -> Vec<PlacedTile> {
    let angles: Vec<f64> = (0..5)
        .map(|j| 2.0 * std::f64::consts::PI * j as f64 / 5.0)
        .collect();
    multigrid_rhombs(&angles, &offsets, extent)
        .into_iter()
        .map(|(j, k, mut quad)| {
            let diff = (k - j) % 5;
            let kind = if diff == 1 || diff == 4 {
                PenroseTile::ThickRhomb
            } else {
                PenroseTile::ThinRhomb
            };
            if Polygon2::new(quad.clone()).area_signed() < 0.0 {
                quad.reverse();
            }
            // Put the splitting diagonal at positions 1/3: thick
            // splits through its 72° corners, thin through 144°.
            let v: [Vec2; 4] = [quad[0], quad[1], quad[2], quad[3]];
            let angles4 = quad_angles(&v);
            let target = if kind == PenroseTile::ThickRhomb {
                72.0
            } else {
                144.0
            };
            let d = std::f64::consts::PI / 180.0;
            if (angles4[0] - target * d).abs() < 1e-6 {
                quad.rotate_left(1);
            }
            PlacedTile {
                kind,
                vertices: quad,
            }
        })
        .collect()
}

/// Ammann-Beenker (octagonal) tiling of squares and 45° rhombs by the
/// four-grid de Bruijn dual. The `iterations` argument scales the
/// generated patch density (offsets stay fixed), kept for signature
/// compatibility with substitution-style generators.
#[must_use]
pub fn ammann_beenker(extent: &Rect, iterations: usize) -> Vec<Polygon2> {
    let scale = (iterations.max(1)) as f64;
    let scaled = Rect {
        min: extent.min * scale,
        max: extent.max * scale,
    };
    let angles: Vec<f64> = (0..4)
        .map(|j| std::f64::consts::PI * j as f64 / 4.0)
        .collect();
    let gamma = [0.234_567_9, 0.376_543_2, 0.129_876_5, 0.298_765_4];
    multigrid_rhombs(&angles, &gamma, &scaled)
        .into_iter()
        .map(|(_, _, quad)| Polygon2::new(quad.into_iter().map(|p| p * (1.0 / scale)).collect()))
        .collect()
}

// ---------------------------------------------------------------------
// Hat monotile (port of Kaplan's `hatviz`).
// ---------------------------------------------------------------------

type Aff = [f64; 6];

const IDENT: Aff = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

fn amul(a: &Aff, b: &Aff) -> Aff {
    [
        a[0] * b[0] + a[1] * b[3],
        a[0] * b[1] + a[1] * b[4],
        a[0] * b[2] + a[1] * b[5] + a[2],
        a[3] * b[0] + a[4] * b[3],
        a[3] * b[1] + a[4] * b[4],
        a[3] * b[2] + a[4] * b[5] + a[5],
    ]
}

fn ainv(t: &Aff) -> Aff {
    let det = t[0] * t[4] - t[1] * t[3];
    [
        t[4] / det,
        -t[1] / det,
        (t[1] * t[5] - t[2] * t[4]) / det,
        -t[3] / det,
        t[0] / det,
        (t[2] * t[3] - t[0] * t[5]) / det,
    ]
}

fn apply(t: &Aff, p: Vec2) -> Vec2 {
    Vec2::new(
        t[0] * p.x + t[1] * p.y + t[2],
        t[3] * p.x + t[4] * p.y + t[5],
    )
}

fn ttrans(x: f64, y: f64) -> Aff {
    [1.0, 0.0, x, 0.0, 1.0, y]
}

fn trot(ang: f64) -> Aff {
    let (s, c) = ang.sin_cos();
    [c, -s, 0.0, s, c, 0.0]
}

fn rot_about(p: Vec2, ang: f64) -> Aff {
    amul(&ttrans(p.x, p.y), &amul(&trot(ang), &ttrans(-p.x, -p.y)))
}

/// Maps the unit interval onto segment p→q.
fn match_seg(p: Vec2, q: Vec2) -> Aff {
    [q.x - p.x, p.y - q.y, p.x, q.y - p.y, q.x - p.x, p.y]
}

/// Maps segment p1→q1 onto p2→q2 (with a flip convention matching the
/// reference implementation).
fn match_two(p1: Vec2, q1: Vec2, p2: Vec2, q2: Vec2) -> Aff {
    amul(&match_seg(p2, q2), &ainv(&match_seg(p1, q1)))
}

fn line_intersect(p1: Vec2, q1: Vec2, p2: Vec2, q2: Vec2) -> Vec2 {
    let d = (q2.y - p2.y) * (q1.x - p1.x) - (q2.x - p2.x) * (q1.y - p1.y);
    let ua = ((q2.x - p2.x) * (p1.y - p2.y) - (q2.y - p2.y) * (p1.x - p2.x)) / d;
    p1 + (q1 - p1) * ua
}

const HR3: f64 = 0.866_025_403_784_438_6;

fn hex_pt(x: f64, y: f64) -> Vec2 {
    Vec2::new(x + 0.5 * y, HR3 * y)
}

fn hat_outline() -> Vec<Vec2> {
    vec![
        hex_pt(0.0, 0.0),
        hex_pt(-1.0, -1.0),
        hex_pt(0.0, -2.0),
        hex_pt(2.0, -2.0),
        hex_pt(2.0, -1.0),
        hex_pt(4.0, -2.0),
        hex_pt(5.0, -1.0),
        hex_pt(4.0, 0.0),
        hex_pt(3.0, 0.0),
        hex_pt(2.0, 2.0),
        hex_pt(0.0, 3.0),
        hex_pt(0.0, 2.0),
        hex_pt(-1.0, 2.0),
    ]
}

/// Metatile tree: node 0 is the hat itself; others carry an outline
/// and transformed children.
#[derive(Clone)]
struct MetaNode {
    outline: Vec<Vec2>,
    children: Vec<(Aff, usize)>,
    is_hat: bool,
}

struct HatSystem {
    nodes: Vec<MetaNode>,
}

impl HatSystem {
    fn add(&mut self, outline: Vec<Vec2>) -> usize {
        self.nodes.push(MetaNode {
            outline,
            children: Vec::new(),
            is_hat: false,
        });
        self.nodes.len() - 1
    }

    fn eval_child(&self, node: usize, child: usize, vertex: usize) -> Vec2 {
        let (t, g) = self.nodes[node].children[child];
        apply(&t, self.nodes[g].outline[vertex])
    }

    fn recentre(&mut self, node: usize) {
        let c = self.nodes[node]
            .outline
            .iter()
            .fold(Vec2::ZERO, |s, &p| s + p)
            * (1.0 / self.nodes[node].outline.len() as f64);
        for p in &mut self.nodes[node].outline {
            *p = *p - c;
        }
        let m = ttrans(-c.x, -c.y);
        for ch in &mut self.nodes[node].children {
            ch.0 = amul(&m, &ch.0);
        }
    }

    fn collect_hats(&self, node: usize, t: &Aff, out: &mut Vec<Vec<Vec2>>) {
        if self.nodes[node].is_hat {
            out.push(
                self.nodes[node]
                    .outline
                    .iter()
                    .map(|&p| apply(t, p))
                    .collect(),
            );
            return;
        }
        for &(ct, g) in &self.nodes[node].children {
            self.collect_hats(g, &amul(t, &ct), out);
        }
    }
}

/// Builds the base H, T, P, F metatiles (each containing hats).
fn hat_base() -> (HatSystem, [usize; 4]) {
    let mut sys = HatSystem { nodes: Vec::new() };
    let hat = sys.add(hat_outline());
    sys.nodes[hat].is_hat = true;
    let ho = hat_outline();

    let h_outline = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 0.0),
        Vec2::new(4.5, HR3),
        Vec2::new(2.5, 5.0 * HR3),
        Vec2::new(1.5, 5.0 * HR3),
        Vec2::new(-0.5, HR3),
    ];
    let h = sys.add(h_outline.clone());
    sys.nodes[h]
        .children
        .push((match_two(ho[5], ho[7], h_outline[5], h_outline[0]), hat));
    sys.nodes[h]
        .children
        .push((match_two(ho[9], ho[11], h_outline[1], h_outline[2]), hat));
    sys.nodes[h]
        .children
        .push((match_two(ho[5], ho[7], h_outline[3], h_outline[4]), hat));
    sys.nodes[h].children.push((
        amul(
            &ttrans(2.5, HR3),
            &amul(
                &[-0.5, -HR3, 0.0, HR3, -0.5, 0.0],
                &[0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
            ),
        ),
        hat,
    ));

    let t_outline = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(3.0, 0.0),
        Vec2::new(1.5, 3.0 * HR3),
    ];
    let t = sys.add(t_outline);
    sys.nodes[t]
        .children
        .push(([0.5, 0.0, 0.5, 0.0, 0.5, HR3], hat));

    let p_outline = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 0.0),
        Vec2::new(3.0, 2.0 * HR3),
        Vec2::new(-1.0, 2.0 * HR3),
    ];
    let p = sys.add(p_outline);
    sys.nodes[p]
        .children
        .push(([0.5, 0.0, 1.5, 0.0, 0.5, HR3], hat));
    sys.nodes[p].children.push((
        amul(
            &ttrans(0.0, 2.0 * HR3),
            &amul(
                &[0.5, HR3, 0.0, -HR3, 0.5, 0.0],
                &[0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            ),
        ),
        hat,
    ));

    let f_outline = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(3.0, 0.0),
        Vec2::new(3.5, HR3),
        Vec2::new(3.0, 2.0 * HR3),
        Vec2::new(-1.0, 2.0 * HR3),
    ];
    let f = sys.add(f_outline);
    sys.nodes[f]
        .children
        .push(([0.5, 0.0, 1.5, 0.0, 0.5, HR3], hat));
    sys.nodes[f].children.push((
        amul(
            &ttrans(0.0, 2.0 * HR3),
            &amul(
                &[0.5, HR3, 0.0, -HR3, 0.5, 0.0],
                &[0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            ),
        ),
        hat,
    ));

    (sys, [h, t, p, f])
}

/// One substitution round: builds the patch of 29 metatiles, then
/// carves the four next-level supertiles out of it (the exact rules
/// of the reference implementation).
fn hat_substitute(sys: &mut HatSystem, tiles: [usize; 4]) -> [usize; 4] {
    let [h, t, p, f] = tiles;
    #[derive(Clone, Copy)]
    enum Rule {
        Seed,
        Edge(usize, usize, usize, usize),
        Two(usize, usize, usize, usize, usize, usize),
    }
    use Rule::*;
    let shapes = |c: usize| match c {
        0 => h,
        1 => t,
        2 => p,
        _ => f,
    };
    // (child_index, child_vertex, [child2, vertex2], shape, shape_vertex)
    let rules: Vec<Rule> = vec![
        Seed,
        Edge(0, 0, 2, 2),
        Edge(1, 0, 0, 2),
        Edge(2, 0, 2, 2),
        Edge(3, 0, 0, 2),
        Edge(4, 4, 2, 2),
        Edge(0, 4, 3, 3),
        Edge(2, 4, 3, 3),
        Two(4, 1, 3, 2, 3, 0),
        Edge(8, 3, 0, 0),
        Edge(9, 2, 2, 0),
        Edge(10, 2, 0, 0),
        Edge(11, 4, 2, 2),
        Edge(12, 0, 0, 2),
        Edge(13, 0, 3, 3),
        Edge(14, 2, 3, 1),
        Edge(15, 3, 0, 4),
        Edge(8, 2, 3, 1),
        Edge(17, 3, 0, 0),
        Edge(18, 2, 2, 0),
        Edge(19, 2, 0, 2),
        Edge(20, 4, 3, 3),
        Edge(20, 0, 2, 2),
        Edge(22, 0, 0, 2),
        Edge(23, 4, 3, 3),
        Edge(23, 0, 3, 3),
        Edge(16, 0, 2, 2),
        Two(9, 4, 0, 2, 1, 2),
        Edge(4, 0, 3, 3),
    ];
    let patch = sys.add(Vec::new());
    for r in rules {
        let (transform, geom) = match r {
            Seed => (IDENT, h),
            Edge(child, vertex, shape, svert) => {
                let (ct, cg) = sys.nodes[patch].children[child];
                let poly_len = sys.nodes[cg].outline.len();
                let pp = apply(&ct, sys.nodes[cg].outline[(vertex + 1) % poly_len]);
                let qq = apply(&ct, sys.nodes[cg].outline[vertex]);
                let ns = shapes(shape);
                let npoly_len = sys.nodes[ns].outline.len();
                (
                    match_two(
                        sys.nodes[ns].outline[svert],
                        sys.nodes[ns].outline[(svert + 1) % npoly_len],
                        pp,
                        qq,
                    ),
                    ns,
                )
            }
            Two(c1, v1, c2, v2, shape, svert) => {
                let pp = sys.eval_child(patch, c2, v2);
                let qq = sys.eval_child(patch, c1, v1);
                let ns = shapes(shape);
                let npoly_len = sys.nodes[ns].outline.len();
                (
                    match_two(
                        sys.nodes[ns].outline[svert],
                        sys.nodes[ns].outline[(svert + 1) % npoly_len],
                        pp,
                        qq,
                    ),
                    ns,
                )
            }
        };
        sys.nodes[patch].children.push((transform, geom));
    }
    // Carve the new metatiles (constructMetatiles).
    let bps1 = sys.eval_child(patch, 8, 2);
    let bps2 = sys.eval_child(patch, 21, 2);
    let rbps = apply(&rot_about(bps1, -2.0 * std::f64::consts::PI / 3.0), bps2);
    let p72 = sys.eval_child(patch, 7, 2);
    let p252 = sys.eval_child(patch, 25, 2);
    let c62 = sys.eval_child(patch, 6, 2);
    let llc = line_intersect(bps1, rbps, c62, p72);
    let mut w = c62 - llc;

    let mut new_h_outline = vec![llc, bps1];
    w = apply(&trot(-std::f64::consts::FRAC_PI_3), w);
    new_h_outline.push(new_h_outline[1] + w);
    new_h_outline.push(sys.eval_child(patch, 14, 2));
    w = apply(&trot(-std::f64::consts::FRAC_PI_3), w);
    new_h_outline.push(new_h_outline[3] - w);
    new_h_outline.push(c62);
    let new_h = sys.add(new_h_outline.clone());
    for ch in [0usize, 9, 16, 27, 26, 6, 1, 8, 10, 15] {
        let child = sys.nodes[patch].children[ch];
        sys.nodes[new_h].children.push(child);
    }

    let new_p_outline = vec![p72, p72 + (bps1 - llc), bps1, llc];
    let new_p = sys.add(new_p_outline);
    for ch in [7usize, 2, 3, 4, 28] {
        let child = sys.nodes[patch].children[ch];
        sys.nodes[new_p].children.push(child);
    }

    let new_f_outline = vec![
        bps2,
        sys.eval_child(patch, 24, 2),
        sys.eval_child(patch, 25, 0),
        p252,
        p252 + (llc - bps1),
    ];
    let new_f = sys.add(new_f_outline);
    for ch in [21usize, 20, 22, 23, 24, 25] {
        let child = sys.nodes[patch].children[ch];
        sys.nodes[new_f].children.push(child);
    }

    let aaa = new_h_outline[2];
    let bbb = new_h_outline[1] + (new_h_outline[4] - new_h_outline[5]);
    let ccc = apply(&rot_about(bbb, -std::f64::consts::FRAC_PI_3), aaa);
    let new_t = sys.add(vec![bbb, ccc, aaa]);
    let child = sys.nodes[patch].children[11];
    sys.nodes[new_t].children.push(child);

    for n in [new_h, new_t, new_p, new_f] {
        sys.recentre(n);
    }
    [new_h, new_t, new_p, new_f]
}

/// A patch of hat monotiles (Smith, Myers, Kaplan & Goodman-Strauss
/// 2023) built by `iterations` rounds of the H/T/P/F metatile
/// substitution; hats whose centroid lies in `extent` are returned
/// (hat edge lengths 1 and √3, fixed scale — grow the extent or the
/// iteration count for more tiles).
#[must_use]
pub fn hat_monotile(extent: &Rect, iterations: usize) -> Vec<Polygon2> {
    let (mut sys, mut tiles) = hat_base();
    for _ in 0..iterations {
        tiles = hat_substitute(&mut sys, tiles);
    }
    let mut hats = Vec::new();
    sys.collect_hats(tiles[0], &IDENT, &mut hats);
    hats.into_iter()
        .map(Polygon2::new)
        .filter(|p| extent.contains_point(p.centroid()))
        .collect()
}

// ---------------------------------------------------------------------
// Spectre monotile (port of the reference 9-metatile substitution).
// ---------------------------------------------------------------------

fn spectre_points() -> Vec<Vec2> {
    let r3h = 3.0f64.sqrt() / 2.0;
    vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.5, -r3h),
        Vec2::new(1.5 + r3h, 0.5 - r3h),
        Vec2::new(1.5 + r3h, 1.5 - r3h),
        Vec2::new(2.5 + r3h, 1.5 - r3h),
        Vec2::new(3.0 + r3h, 1.5),
        Vec2::new(3.0, 2.0),
        Vec2::new(3.0 - r3h, 1.5),
        Vec2::new(2.5 - r3h, 1.5 + r3h),
        Vec2::new(1.5 - r3h, 1.5 + r3h),
        Vec2::new(0.5 - r3h, 1.5 + r3h),
        Vec2::new(-r3h, 1.5),
        Vec2::new(0.0, 1.0),
    ]
}

/// One tile system level: 9 labeled metatiles, each a set of
/// transformed spectres plus 4 quad anchor points.
#[derive(Clone)]
struct SpectreMeta {
    /// (transform, spectre?) — leaves carry the outline directly.
    placements: Vec<Aff>,
    quad: [Vec2; 4],
}

const SPECTRE_LABELS: usize = 9; // Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi

fn spectre_base() -> Vec<SpectreMeta> {
    let pts = spectre_points();
    let quad = [pts[3], pts[5], pts[7], pts[11]];
    let single = SpectreMeta {
        placements: vec![IDENT],
        quad,
    };
    let mut out = vec![single.clone(); SPECTRE_LABELS];
    // Gamma is the "Mystic": two spectres.
    out[0] = SpectreMeta {
        placements: vec![
            IDENT,
            amul(
                &ttrans(pts[8].x, pts[8].y),
                &trot(std::f64::consts::FRAC_PI_6),
            ),
        ],
        quad,
    };
    out
}

fn spectre_supertiles(system: &[SpectreMeta]) -> Vec<SpectreMeta> {
    let quad = system[1].quad; // Delta's quad
    let r_flip: Aff = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    // (rotation degrees, from-quad-point, to-quad-point)
    let rules: [(f64, usize, usize); 7] = [
        (60.0, 3, 1),
        (0.0, 2, 0),
        (60.0, 3, 1),
        (60.0, 3, 1),
        (0.0, 2, 0),
        (60.0, 3, 1),
        (-120.0, 3, 3),
    ];
    let mut transformations: Vec<Aff> = vec![IDENT];
    let mut total_angle = 0.0f64;
    let mut rotation = IDENT;
    let mut transformed_quad = quad;
    for (angle, from, to) in rules {
        if angle != 0.0 {
            total_angle += angle;
            rotation = trot(total_angle.to_radians());
            for (dst, src) in transformed_quad.iter_mut().zip(&quad) {
                *dst = apply(&rotation, *src);
            }
        }
        let last = transformations.last().expect("nonempty");
        let target = apply(last, quad[from]);
        let shift = target - transformed_quad[to];
        transformations.push(amul(&ttrans(shift.x, shift.y), &rotation));
    }
    for t in &mut transformations {
        *t = amul(&r_flip, t);
    }
    // Substitution table (None = empty slot). Labels:
    // 0 Gamma, 1 Delta, 2 Theta, 3 Lambda, 4 Xi, 5 Pi, 6 Sigma,
    // 7 Phi, 8 Psi.
    let super_rules: [[Option<usize>; 8]; 9] = [
        [
            Some(5),
            Some(1),
            None,
            Some(2),
            Some(6),
            Some(4),
            Some(7),
            Some(0),
        ],
        [
            Some(4),
            Some(1),
            Some(4),
            Some(7),
            Some(6),
            Some(5),
            Some(7),
            Some(0),
        ],
        [
            Some(8),
            Some(1),
            Some(5),
            Some(7),
            Some(6),
            Some(5),
            Some(7),
            Some(0),
        ],
        [
            Some(8),
            Some(1),
            Some(4),
            Some(7),
            Some(6),
            Some(5),
            Some(7),
            Some(0),
        ],
        [
            Some(8),
            Some(1),
            Some(5),
            Some(7),
            Some(6),
            Some(8),
            Some(7),
            Some(0),
        ],
        [
            Some(8),
            Some(1),
            Some(4),
            Some(7),
            Some(6),
            Some(8),
            Some(7),
            Some(0),
        ],
        [
            Some(4),
            Some(1),
            Some(4),
            Some(7),
            Some(6),
            Some(5),
            Some(3),
            Some(0),
        ],
        [
            Some(8),
            Some(1),
            Some(8),
            Some(7),
            Some(6),
            Some(5),
            Some(7),
            Some(0),
        ],
        [
            Some(8),
            Some(1),
            Some(8),
            Some(7),
            Some(6),
            Some(8),
            Some(7),
            Some(0),
        ],
    ];
    let super_quad = [
        apply(&transformations[6], quad[2]),
        apply(&transformations[5], quad[1]),
        apply(&transformations[3], quad[2]),
        apply(&transformations[0], quad[1]),
    ];
    super_rules
        .iter()
        .map(|subs| {
            let mut placements = Vec::new();
            for (sub, t) in subs.iter().zip(&transformations) {
                if let Some(label) = sub {
                    for pt in &system[*label].placements {
                        placements.push(amul(t, pt));
                    }
                }
            }
            SpectreMeta {
                placements,
                quad: super_quad,
            }
        })
        .collect()
}

/// A patch of spectre monotiles ("A chiral aperiodic monotile",
/// Smith, Myers, Kaplan & Goodman-Strauss 2023) built by `iterations`
/// substitution rounds; spectres with centroid inside `extent` are
/// returned (unit edge length, fixed scale).
#[must_use]
pub fn spectre_monotile(extent: &Rect, iterations: usize) -> Vec<Polygon2> {
    let mut system = spectre_base();
    for _ in 0..iterations {
        system = spectre_supertiles(&system);
    }
    let pts = spectre_points();
    system[1] // Delta
        .placements
        .iter()
        .map(|t| Polygon2::new(pts.iter().map(|&p| apply(t, p)).collect()))
        .filter(|p| extent.contains_point(p.centroid()))
        .collect()
}

// ---------------------------------------------------------------------
// Pinwheel tiling.
// ---------------------------------------------------------------------

/// A 1:2:√5 right triangle as (right-angle vertex, short-leg end,
/// long-leg end).
#[derive(Clone, Copy)]
struct PinTri {
    r: Vec2,
    a: Vec2,
    b: Vec2,
}

fn pinwheel_subdivide(t: &PinTri) -> [PinTri; 5] {
    // Foot of the altitude from the right angle onto the hypotenuse.
    let ab = t.b - t.a;
    let s = (t.r - t.a).dot(&ab) / ab.magnitude_squared();
    let h = t.a + ab * s;
    let m = |p: Vec2, q: Vec2| (p + q) * 0.5;
    [
        // Small corner triangle at the short-leg end.
        PinTri {
            r: h,
            a: t.a,
            b: t.r,
        },
        // The big half quartered: corner at h, corner at r, corner at
        // b, and the central inverted copy.
        PinTri {
            r: h,
            a: m(h, t.r),
            b: m(h, t.b),
        },
        PinTri {
            r: m(t.r, h),
            a: t.r,
            b: m(t.r, t.b),
        },
        PinTri {
            r: m(t.b, h),
            a: m(t.b, t.r),
            b: t.b,
        },
        PinTri {
            r: m(t.r, t.b),
            a: m(h, t.b),
            b: m(h, t.r),
        },
    ]
}

/// Pinwheel tiling (Radin 1994): 1:2:√5 right triangles subdivided
/// `iterations` times, seeded by two triangles covering the extent.
#[must_use]
pub fn pinwheel(extent: &Rect, iterations: usize) -> Vec<Polygon2> {
    // Seed rectangle split into two 1:2 right triangles.
    let c = extent.center();
    let half = (extent.max - extent.min) * 0.5;
    let s = half.x.max(half.y * 2.0);
    let (w, hgt) = (s, s / 2.0);
    let corners = [
        c + Vec2::new(-w, -hgt),
        c + Vec2::new(w, -hgt),
        c + Vec2::new(w, hgt),
        c + Vec2::new(-w, hgt),
    ];
    let mut tris = vec![
        PinTri {
            r: corners[1],
            a: corners[2],
            b: corners[0],
        },
        PinTri {
            r: corners[3],
            a: corners[0],
            b: corners[2],
        },
    ];
    for _ in 0..iterations {
        tris = tris
            .iter()
            .flat_map(|t| pinwheel_subdivide(t).into_iter())
            .collect();
    }
    tris.into_iter()
        .map(|t| Polygon2::new(vec![t.r, t.a, t.b]))
        .filter(|p| extent.contains_point(p.centroid()))
        .collect()
}

/// The Fibonacci word: fixed point of a→ab, b→a. Returns the first
/// `n` letters (`true` = a).
#[must_use]
pub fn fibonacci_word(n: usize) -> Vec<bool> {
    let mut word = vec![true];
    while word.len() < n {
        let mut next = Vec::with_capacity(word.len() * 2);
        for &c in &word {
            if c {
                next.push(true);
                next.push(false);
            } else {
                next.push(true);
            }
        }
        word = next;
    }
    word.truncate(n.max(0));
    word
}

/// 1-D quasicrystal by the canonical cut-and-project scheme: lattice
/// points of ℤ² whose perpendicular coordinate falls in the canonical
/// window are projected onto the line of the given slope. Returns
/// sorted positions with |x| <= extent. Irrational slopes give
/// aperiodic point sets (slope 1/φ gives the Fibonacci chain).
///
/// # Panics
/// Panics unless `extent > 0`.
#[must_use]
pub fn cut_and_project_1d(slope: f64, extent: f64) -> Vec<f64> {
    assert!(extent > 0.0, "extent must be positive");
    let norm = (1.0 + slope * slope).sqrt();
    let (ex, ey) = (1.0 / norm, slope / norm); // parallel direction
                                               // Perpendicular direction (-ey, ex); canonical window = projection
                                               // of the unit square onto it.
    let w = ex + ey;
    let range = (extent * norm) as i64 + 2;
    let mut out = Vec::new();
    for i in -range..=range {
        for j in -range..=range {
            let par = i as f64 * ex + j as f64 * ey;
            let perp = -(i as f64) * ey + j as f64 * ex;
            if perp >= 0.0 && perp < w && par.abs() <= extent {
                out.push(par);
            }
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_area(tiles: &[PlacedTile]) -> f64 {
        tiles
            .iter()
            .map(|t| Polygon2::new(t.vertices.clone()).area())
            .sum()
    }

    fn assert_no_overlaps(polys: &[Polygon2], probes: usize, seed: u64) {
        let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in polys {
            for v in &p.vertices {
                lo = Vec2::new(lo.x.min(v.x), lo.y.min(v.y));
                hi = Vec2::new(hi.x.max(v.x), hi.y.max(v.y));
            }
        }
        let mut state = seed;
        let mut rand = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let inside = |poly: &Polygon2, p: Vec2| {
            let v = &poly.vertices;
            let n = v.len();
            let mut ins = false;
            for i in 0..n {
                let (a, b) = (v[i], v[(i + 1) % n]);
                if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x)
                {
                    ins = !ins;
                }
            }
            ins
        };
        for _ in 0..probes {
            let p = Vec2::new(lo.x + rand() * (hi.x - lo.x), lo.y + rand() * (hi.y - lo.y));
            let count = polys.iter().filter(|poly| inside(poly, p)).count();
            assert!(count <= 1, "tiles overlap at {p:?}");
        }
    }

    #[test]
    fn test_p2_deflation_ratio_and_structure() {
        let seed = penrose_p2_seed(1.0);
        assert_eq!(seed.len(), 5, "sun seed has 5 kites");
        assert!(seed.iter().all(|t| t.kind == PenroseTile::Kite));
        let a0 = total_area(&seed);
        let tiles = penrose_p2_deflate(&seed, 5);
        // Halves on the patch boundary lack a mirror partner and are
        // dropped, so the merged area is a large fraction of the
        // original, never more.
        let a = total_area(&tiles);
        assert!(
            a <= a0 * (1.0 + 1e-9) && a > 0.7 * a0,
            "merged area {a} vs seed {a0}"
        );
        // Edge lengths shrink by phi^-5: each tile has edges of two
        // lengths with ratio phi.
        for t in &tiles {
            let v = &t.vertices;
            let mut lens: Vec<f64> = (0..4).map(|i| v[i].distance_to(&v[(i + 1) % 4])).collect();
            lens.sort_by(f64::total_cmp);
            assert!((lens[1] / lens[0] - 1.0).abs() < 1e-9);
            assert!((lens[3] / lens[2] - 1.0).abs() < 1e-9);
            assert!((lens[3] / lens[0] - PHI).abs() < 1e-9, "edge ratio phi");
        }
        let r = ratio_thick_to_thin(&tiles);
        assert!((r - PHI).abs() < 0.05, "kite/dart ratio {r} near phi");
        let polys: Vec<Polygon2> = tiles
            .iter()
            .map(|t| Polygon2::new(t.vertices.clone()))
            .collect();
        assert_no_overlaps(&polys, 2000, 11);
    }

    #[test]
    fn test_p3_deflation_ratio_and_structure() {
        let sun = penrose_p3_sun(1.0);
        assert!(!sun.is_empty());
        let a0 = total_area(&sun);
        let tiles = penrose_p3_deflate(&sun, 6);
        let a = total_area(&tiles);
        assert!(
            a <= a0 * (1.0 + 1e-9) && a > 0.6 * a0,
            "merged area {a} vs {a0}"
        );
        for t in &tiles {
            // All rhombs: 4 equal edges.
            let v = &t.vertices;
            let lens: Vec<f64> = (0..4).map(|i| v[i].distance_to(&v[(i + 1) % 4])).collect();
            for l in &lens {
                assert!((l - lens[0]).abs() < 1e-9, "rhomb edges equal");
            }
        }
        let r = ratio_thick_to_thin(&tiles);
        assert!((r - PHI).abs() < 0.08, "thick/thin ratio {r} near phi");
        let polys: Vec<Polygon2> = tiles
            .iter()
            .map(|t| Polygon2::new(t.vertices.clone()))
            .collect();
        assert_no_overlaps(&polys, 2000, 12);
        // Star seed also works.
        let star = penrose_p3_star(1.0);
        assert!(!star.is_empty());
    }

    #[test]
    fn test_pentagrid_projection() {
        let extent = Rect {
            min: Vec2::new(-4.0, -4.0),
            max: Vec2::new(4.0, 4.0),
        };
        let offsets = [0.17, 0.23, 0.11, 0.31, 0.18];
        let tiles = penrose_by_projection(&extent, offsets);
        assert!(tiles.len() > 30);
        let d = std::f64::consts::PI / 180.0;
        for t in &tiles {
            let v: [Vec2; 4] = [t.vertices[0], t.vertices[1], t.vertices[2], t.vertices[3]];
            // Unit rhombs of the right shapes.
            for i in 0..4 {
                assert!((v[i].distance_to(&v[(i + 1) % 4]) - 1.0).abs() < 1e-9);
            }
            let angles = quad_angles(&v);
            let small = angles.iter().cloned().fold(f64::INFINITY, f64::min);
            match t.kind {
                PenroseTile::ThinRhomb => assert!((small - 36.0 * d).abs() < 1e-9),
                PenroseTile::ThickRhomb => assert!((small - 72.0 * d).abs() < 1e-9),
                _ => panic!("pentagrid produces rhombs"),
            }
        }
        let polys: Vec<Polygon2> = tiles
            .iter()
            .map(|t| Polygon2::new(t.vertices.clone()))
            .collect();
        assert_no_overlaps(&polys, 1500, 13);
    }

    #[test]
    fn test_ammann_beenker() {
        let extent = Rect {
            min: Vec2::new(-4.0, -4.0),
            max: Vec2::new(4.0, 4.0),
        };
        let tiles = ammann_beenker(&extent, 1);
        assert!(tiles.len() > 20);
        let d = std::f64::consts::PI / 180.0;
        let mut squares = 0usize;
        let mut rhombs = 0usize;
        for p in &tiles {
            let v: [Vec2; 4] = [p.vertices[0], p.vertices[1], p.vertices[2], p.vertices[3]];
            for i in 0..4 {
                assert!((v[i].distance_to(&v[(i + 1) % 4]) - 1.0).abs() < 1e-9);
            }
            let angles = quad_angles(&v);
            let small = angles.iter().cloned().fold(f64::INFINITY, f64::min);
            if (small - 90.0 * d).abs() < 1e-9 {
                squares += 1;
            } else if (small - 45.0 * d).abs() < 1e-9 {
                rhombs += 1;
            } else {
                panic!("Ammann-Beenker tile with angle {small}");
            }
        }
        assert!(squares > 0 && rhombs > 0);
        assert_no_overlaps(&tiles, 1500, 14);
    }

    #[test]
    fn test_hat_monotile() {
        let extent = Rect {
            min: Vec2::new(-30.0, -30.0),
            max: Vec2::new(30.0, 30.0),
        };
        let hats = hat_monotile(&extent, 2);
        assert!(hats.len() > 20, "got {} hats", hats.len());
        // Every hat is congruent (the metatiles place hats at half
        // the outline scale; reflected copies appear, as hat tilings
        // require): 13 vertices, one common area, edges of length a
        // and a*sqrt(3), plus the one doubled edge 2a spanning the
        // flat vertex of the underlying 14-vertex polykite outline.
        let a0 = hats[0].area();
        for h in &hats {
            assert_eq!(h.vertices.len(), 13);
            assert!((h.area() - a0).abs() < 1e-6);
            let lens: Vec<f64> = (0..13)
                .map(|i| h.vertices[i].distance_to(&h.vertices[(i + 1) % 13]))
                .collect();
            let l0 = lens.iter().cloned().fold(f64::INFINITY, f64::min);
            for l in &lens {
                assert!(
                    (l - l0).abs() < 1e-6
                        || (l - l0 * 3.0f64.sqrt()).abs() < 1e-6
                        || (l - l0 * 2.0).abs() < 1e-6,
                    "hat edge length {l} vs {l0}"
                );
            }
        }
        assert_no_overlaps(&hats, 2000, 15);
    }

    #[test]
    fn test_spectre_monotile() {
        let extent = Rect {
            min: Vec2::new(-30.0, -30.0),
            max: Vec2::new(30.0, 30.0),
        };
        let tiles = spectre_monotile(&extent, 2);
        assert!(tiles.len() > 20, "got {} spectres", tiles.len());
        let a0 = Polygon2::new(spectre_points()).area();
        for t in &tiles {
            assert_eq!(t.vertices.len(), 14);
            assert!((t.area() - a0).abs() < 1e-6, "congruent spectres");
            for i in 0..14 {
                let l = t.vertices[i].distance_to(&t.vertices[(i + 1) % 14]);
                assert!((l - 1.0).abs() < 1e-6, "spectre edges are unit ({l})");
            }
        }
        assert_no_overlaps(&tiles, 2000, 16);
    }

    #[test]
    fn test_pinwheel() {
        let extent = Rect {
            min: Vec2::new(-2.0, -1.0),
            max: Vec2::new(2.0, 1.0),
        };
        let tiles = pinwheel(&extent, 3);
        assert!(tiles.len() > 50);
        for t in &tiles {
            let v = &t.vertices;
            let mut lens: Vec<f64> = (0..3).map(|i| v[i].distance_to(&v[(i + 1) % 3])).collect();
            lens.sort_by(f64::total_cmp);
            // 1 : 2 : sqrt(5).
            assert!((lens[1] / lens[0] - 2.0).abs() < 1e-9);
            assert!((lens[2] / lens[0] - 5.0f64.sqrt()).abs() < 1e-9);
        }
        assert_no_overlaps(&tiles, 1500, 17);
        // Areas conserved across a subdivision level (5x count, /5
        // area).
        let t1 = pinwheel(&extent, 1);
        let a1: f64 = t1.iter().map(Polygon2::area).sum();
        let t2 = pinwheel(&extent, 2);
        let a2: f64 = t2.iter().map(Polygon2::area).sum();
        assert!((a1 - a2).abs() < 1e-9 * a1.max(1.0));
    }

    #[test]
    fn test_fibonacci_word_and_cut_project() {
        let w = fibonacci_word(13);
        // a b a a b a b a a b a a b
        let expect = [
            true, false, true, true, false, true, false, true, true, false, true, true, false,
        ];
        assert_eq!(w, expect);
        // Letter frequency ratio approaches phi.
        let w = fibonacci_word(10_000);
        let a = w.iter().filter(|&&c| c).count();
        let b = w.len() - a;
        assert!((a as f64 / b as f64 - PHI).abs() < 1e-3);

        // Fibonacci chain from slope 1/phi: two gap lengths, ratio phi.
        let pts = cut_and_project_1d(1.0 / PHI, 30.0);
        assert!(pts.len() > 30);
        let mut gaps: Vec<f64> = pts.windows(2).map(|w| w[1] - w[0]).collect();
        gaps.sort_by(f64::total_cmp);
        let short = gaps[0];
        let long = *gaps.last().expect("nonempty");
        assert!(
            (long / short - PHI).abs() < 1e-6,
            "gap ratio {}",
            long / short
        );
        for g in &gaps {
            assert!(
                (g - short).abs() < 1e-9 || (g - long).abs() < 1e-9,
                "only two gap lengths"
            );
        }
        // Aperiodic mix: both gaps occur.
        let n_short = gaps.iter().filter(|&&g| (g - short).abs() < 1e-9).count();
        assert!(n_short > 0 && n_short < gaps.len());
    }
}
