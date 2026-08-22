//! Plane tilings: regular and Archimedean (uniform) tilings, their
//! Laves duals, hex-grid coordinate algebra, and a few classic
//! non-edge-to-edge patterns (brick, herringbone).

use crate::math::Vec2;
use crate::spatial::primitives::{Polygon2, Rect};
use std::collections::HashMap;

/// A tiling as an indexed face set: `faces` are counterclockwise
/// vertex loops, `edges` the unique undirected edges.
#[derive(Debug, Clone, PartialEq)]
pub struct Tiling {
    pub vertices: Vec<Vec2>,
    pub faces: Vec<Vec<usize>>,
    pub edges: Vec<(usize, usize)>,
}

/// Incremental builder deduplicating vertices on a quantized grid.
struct TilingBuilder {
    quantum: f64,
    map: HashMap<(i64, i64), usize>,
    vertices: Vec<Vec2>,
    faces: Vec<Vec<usize>>,
}

impl TilingBuilder {
    fn new(quantum: f64) -> Self {
        Self { quantum, map: HashMap::new(), vertices: Vec::new(), faces: Vec::new() }
    }

    fn vertex(&mut self, p: Vec2) -> usize {
        let key = (
            (p.x / self.quantum).round() as i64,
            (p.y / self.quantum).round() as i64,
        );
        *self.map.entry(key).or_insert_with(|| {
            self.vertices.push(p);
            self.vertices.len() - 1
        })
    }

    fn face(&mut self, loop_: &[Vec2]) {
        let ids: Vec<usize> = loop_.iter().map(|&p| self.vertex(p)).collect();
        self.faces.push(ids);
    }

    fn build(self) -> Tiling {
        let mut edges: Vec<(usize, usize)> = self
            .faces
            .iter()
            .flat_map(|f| {
                (0..f.len()).map(move |k| {
                    let (a, b) = (f[k], f[(k + 1) % f.len()]);
                    (a.min(b), a.max(b))
                })
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        Tiling { vertices: self.vertices, faces: self.faces, edges }
    }
}

impl Tiling {
    /// Keeps only faces whose vertices all lie inside the rectangle
    /// (closed), reindexing vertices.
    #[must_use]
    pub fn clip_to_rect(&self, rect: &Rect) -> Tiling {
        let eps = 1e-9 * (rect.max - rect.min).magnitude().max(1.0);
        let inside = |p: Vec2| {
            p.x >= rect.min.x - eps
                && p.x <= rect.max.x + eps
                && p.y >= rect.min.y - eps
                && p.y <= rect.max.y + eps
        };
        let mut remap: HashMap<usize, usize> = HashMap::new();
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for f in &self.faces {
            if f.iter().all(|&v| inside(self.vertices[v])) {
                faces.push(
                    f.iter()
                        .map(|&v| {
                            *remap.entry(v).or_insert_with(|| {
                                vertices.push(self.vertices[v]);
                                vertices.len() - 1
                            })
                        })
                        .collect(),
                );
            }
        }
        let mut edges: Vec<(usize, usize)> = faces
            .iter()
            .flat_map(|f: &Vec<usize>| {
                (0..f.len()).map(move |k| {
                    let (a, b) = (f[k], f[(k + 1) % f.len()]);
                    (a.min(b), a.max(b))
                })
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        Tiling { vertices, faces, edges }
    }

    /// Faces as polygons.
    #[must_use]
    pub fn polygons(&self) -> Vec<Polygon2> {
        self.faces
            .iter()
            .map(|f| Polygon2::new(f.iter().map(|&v| self.vertices[v]).collect()))
            .collect()
    }

    /// Centroid of every face.
    #[must_use]
    pub fn face_centroids(&self) -> Vec<Vec2> {
        self.faces
            .iter()
            .map(|f| {
                f.iter().map(|&v| self.vertices[v]).fold(Vec2::ZERO, |s, p| s + p)
                    * (1.0 / f.len() as f64)
            })
            .collect()
    }

    /// Dual tiling: one vertex per face (at its centroid), one face
    /// per interior tiling vertex (a vertex is interior when its
    /// incident face angles sum to 2π). Boundary vertices produce no
    /// dual face.
    #[must_use]
    pub fn dual(&self) -> Tiling {
        let centroids = self.face_centroids();
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); self.vertices.len()];
        let mut angle_sum = vec![0.0f64; self.vertices.len()];
        for (fi, f) in self.faces.iter().enumerate() {
            let m = f.len();
            for k in 0..m {
                let v = f[k];
                incident[v].push(fi);
                let prev = self.vertices[f[(k + m - 1) % m]];
                let next = self.vertices[f[(k + 1) % m]];
                let p = self.vertices[v];
                angle_sum[v] += (prev - p).angle_between(&(next - p));
            }
        }
        let mut builder = TilingBuilder::new(1e-9);
        for (v, faces) in incident.iter().enumerate() {
            if faces.len() < 3 || (angle_sum[v] - 2.0 * std::f64::consts::PI).abs() > 1e-6 {
                continue;
            }
            let p = self.vertices[v];
            let mut ordered: Vec<usize> = faces.clone();
            ordered.sort_by(|&a, &b| {
                let pa = centroids[a] - p;
                let pb = centroids[b] - p;
                pa.y.atan2(pa.x).total_cmp(&pb.y.atan2(pb.x))
            });
            let loop_: Vec<Vec2> = ordered.iter().map(|&fi| centroids[fi]).collect();
            builder.face(&loop_);
        }
        builder.build()
    }
}

/// The 11 Archimedean (uniform) tilings by vertex configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Archimedean {
    /// 3.3.3.3.3.3 (triangular)
    T3_3_3_3_3_3,
    /// 4.4.4.4 (square)
    T4_4_4_4,
    /// 6.6.6 (hexagonal)
    T6_6_6,
    /// 3.3.3.3.6 (snub hexagonal)
    T3_3_3_3_6,
    /// 3.3.3.4.4 (elongated triangular)
    T3_3_3_4_4,
    /// 3.3.4.3.4 (snub square)
    T3_3_4_3_4,
    /// 3.4.6.4 (rhombitrihexagonal)
    T3_4_6_4,
    /// 3.6.3.6 (trihexagonal / kagome)
    T3_6_3_6,
    /// 3.12.12 (truncated hexagonal)
    T3_12_12,
    /// 4.6.12 (truncated trihexagonal)
    T4_6_12,
    /// 4.8.8 (truncated square)
    T4_8_8,
}

fn regular_polygon(center: Vec2, radius: f64, n: usize, angle0: f64) -> Vec<Vec2> {
    (0..n)
        .map(|k| {
            let a = angle0 + 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            center + Vec2::new(radius * a.cos(), radius * a.sin())
        })
        .collect()
}

/// Finds the unit-edge equilateral triangles among `verts` whose
/// centroid is not inside any of the given polygons (used to fill the
/// gaps of the snub tilings).
fn fill_unit_triangles(verts: &[Vec2], occupied: &[Vec<Vec2>]) -> Vec<Vec<Vec2>> {
    let inside_any = |p: Vec2| {
        occupied.iter().any(|poly| {
            let n = poly.len();
            let mut ins = false;
            for i in 0..n {
                let (a, b) = (poly[i], poly[(i + 1) % n]);
                if (a.y > p.y) != (b.y > p.y)
                    && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x)
                {
                    ins = !ins;
                }
            }
            ins
        })
    };
    let mut out = Vec::new();
    let mut seen: Vec<(i64, i64)> = Vec::new();
    let n = verts.len();
    for i in 0..n {
        for j in i + 1..n {
            if (verts[i].distance_to(&verts[j]) - 1.0).abs() > 1e-6 {
                continue;
            }
            for k in j + 1..n {
                if (verts[i].distance_to(&verts[k]) - 1.0).abs() > 1e-6
                    || (verts[j].distance_to(&verts[k]) - 1.0).abs() > 1e-6
                {
                    continue;
                }
                let c = (verts[i] + verts[j] + verts[k]) * (1.0 / 3.0);
                if inside_any(c) {
                    continue;
                }
                let key = ((c.x * 1e6).round() as i64, (c.y * 1e6).round() as i64);
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                // Counterclockwise order.
                let mut tri = vec![verts[i], verts[j], verts[k]];
                if (tri[1] - tri[0]).cross(&(tri[2] - tri[0])) < 0.0 {
                    tri.swap(1, 2);
                }
                out.push(tri);
            }
        }
    }
    out
}

/// Periodic cell description: basis vectors and faces (unit edge
/// length).
fn archimedean_cell(kind: Archimedean) -> (Vec2, Vec2, Vec<Vec<Vec2>>) {
    let s3 = 3.0f64.sqrt();
    match kind {
        Archimedean::T4_4_4_4 => (
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            vec![vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ]],
        ),
        Archimedean::T3_3_3_3_3_3 => (
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, s3 / 2.0),
            vec![
                vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.5, s3 / 2.0)],
                vec![
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.5, s3 / 2.0),
                    Vec2::new(0.5, s3 / 2.0),
                ],
            ],
        ),
        Archimedean::T6_6_6 => (
            Vec2::new(s3, 0.0),
            Vec2::new(s3 / 2.0, 1.5),
            vec![regular_polygon(Vec2::ZERO, 1.0, 6, std::f64::consts::FRAC_PI_6)],
        ),
        Archimedean::T3_6_3_6 => (
            Vec2::new(2.0, 0.0),
            Vec2::new(1.0, s3),
            vec![
                regular_polygon(Vec2::ZERO, 1.0, 6, 0.0),
                vec![
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.5, s3 / 2.0),
                    Vec2::new(0.5, s3 / 2.0),
                ],
                vec![
                    Vec2::new(1.0, 0.0),
                    Vec2::new(0.5, -s3 / 2.0),
                    Vec2::new(1.5, -s3 / 2.0),
                ],
            ],
        ),
        Archimedean::T4_8_8 => {
            let a = 1.0 + std::f64::consts::SQRT_2;
            let c = 0.5 + std::f64::consts::FRAC_1_SQRT_2;
            let d = 0.5 + std::f64::consts::SQRT_2;
            (
                Vec2::new(a, 0.0),
                Vec2::new(0.0, a),
                vec![
                    regular_polygon(
                        Vec2::ZERO,
                        (0.25 + c * c).sqrt(),
                        8,
                        (0.5f64).atan2(c),
                    ),
                    vec![
                        Vec2::new(c, 0.5),
                        Vec2::new(d, c),
                        Vec2::new(c, d),
                        Vec2::new(0.5, c),
                    ],
                ],
            )
        }
        Archimedean::T3_12_12 => {
            let d = 2.0 + s3;
            let r12 = 0.5 / (std::f64::consts::PI / 12.0).sin();
            let v15 = |ang: f64| Vec2::new(r12 * ang.cos(), r12 * ang.sin());
            let a15 = std::f64::consts::PI / 12.0;
            (
                Vec2::new(d, 0.0),
                Vec2::new(d / 2.0, d * s3 / 2.0),
                vec![
                    regular_polygon(Vec2::ZERO, r12, 12, a15),
                    vec![
                        v15(a15),
                        Vec2::new(d / 2.0, d * s3 / 2.0) + v15(a15 * 17.0), // 255 deg
                        Vec2::new(d / 2.0, d * s3 / 2.0) + v15(a15 * 19.0), // 285 deg
                    ],
                    vec![
                        v15(-a15),
                        Vec2::new(d / 2.0, -d * s3 / 2.0) + v15(-a15 * 19.0),
                        Vec2::new(d / 2.0, -d * s3 / 2.0) + v15(-a15 * 17.0),
                    ],
                ],
            )
        }
        Archimedean::T4_6_12 => {
            let d = 3.0 + s3;
            let r12 = 0.5 / (std::f64::consts::PI / 12.0).sin();
            let a15 = std::f64::consts::PI / 12.0;
            let x1 = d / 2.0 - 0.5;
            let x2 = d / 2.0 + 0.5;
            let hex_c = Vec2::new(d / 2.0, d * s3 / 6.0);
            let square = vec![
                Vec2::new(x1, -0.5),
                Vec2::new(x2, -0.5),
                Vec2::new(x2, 0.5),
                Vec2::new(x1, 0.5),
            ];
            let rot = |poly: &[Vec2], ang: f64| -> Vec<Vec2> {
                poly.iter().map(|p| p.rotate(ang)).collect()
            };
            let deg60 = std::f64::consts::FRAC_PI_3;
            (
                Vec2::new(d, 0.0),
                Vec2::new(d / 2.0, d * s3 / 2.0),
                vec![
                    regular_polygon(Vec2::ZERO, r12, 12, a15),
                    square.clone(),
                    rot(&square, deg60),
                    rot(&square, 2.0 * deg60),
                    regular_polygon(hex_c, 1.0, 6, 0.0),
                    regular_polygon(Vec2::new(hex_c.x, -hex_c.y), 1.0, 6, 0.0),
                ],
            )
        }
        Archimedean::T3_4_6_4 => {
            let d = 1.0 + s3;
            let square = vec![
                Vec2::new(s3 / 2.0, -0.5),
                Vec2::new(s3 / 2.0 + 1.0, -0.5),
                Vec2::new(s3 / 2.0 + 1.0, 0.5),
                Vec2::new(s3 / 2.0, 0.5),
            ];
            let rot = |poly: &[Vec2], ang: f64| -> Vec<Vec2> {
                poly.iter().map(|p| p.rotate(ang)).collect()
            };
            let deg60 = std::f64::consts::FRAC_PI_3;
            (
                Vec2::new(d, 0.0),
                Vec2::new(d / 2.0, d * s3 / 2.0),
                vec![
                    regular_polygon(Vec2::ZERO, 1.0, 6, std::f64::consts::FRAC_PI_6),
                    square.clone(),
                    rot(&square, deg60),
                    rot(&square, 2.0 * deg60),
                    vec![
                        Vec2::new(s3 / 2.0, 0.5),
                        Vec2::new(s3 / 2.0 + 1.0, 0.5),
                        Vec2::new(d / 2.0, d * s3 / 2.0 - 1.0),
                    ],
                    vec![
                        Vec2::new(s3 / 2.0, -0.5),
                        Vec2::new(d / 2.0, -(d * s3 / 2.0 - 1.0)),
                        Vec2::new(s3 / 2.0 + 1.0, -0.5),
                    ],
                ],
            )
        }
        Archimedean::T3_3_3_4_4 => (
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0 + s3 / 2.0),
            vec![
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(0.0, 1.0),
                ],
                vec![
                    Vec2::new(0.0, 1.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(0.5, 1.0 + s3 / 2.0),
                ],
                vec![
                    Vec2::new(1.0, 1.0),
                    Vec2::new(1.5, 1.0 + s3 / 2.0),
                    Vec2::new(0.5, 1.0 + s3 / 2.0),
                ],
            ],
        ),
        Archimedean::T3_3_4_3_4 => {
            let l = (2.0 + s3).sqrt();
            let deg = std::f64::consts::PI / 180.0;
            let sq = |center: Vec2, rot: f64| {
                regular_polygon(center, std::f64::consts::FRAC_1_SQRT_2, 4, rot)
            };
            let squares = vec![
                sq(Vec2::ZERO, 60.0 * deg),
                sq(Vec2::new(l / 2.0, l / 2.0), 30.0 * deg),
            ];
            // Collect vertices from a 3x3 block of cells and detect
            // the gap triangles of the base cell.
            let mut verts = Vec::new();
            for i in -1..=1 {
                for j in -1..=1 {
                    let off = Vec2::new(l * i as f64, l * j as f64);
                    for s in &squares {
                        for &p in s {
                            verts.push(p + off);
                        }
                    }
                }
            }
            let occupied: Vec<Vec<Vec2>> = (-1..=1)
                .flat_map(|i| {
                    let squares = squares.clone();
                    (-1..=1).flat_map(move |j| {
                        let off = Vec2::new(l * f64::from(i), l * f64::from(j));
                        squares
                            .iter()
                            .map(|s| s.iter().map(|&p| p + off).collect::<Vec<_>>())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let mut faces = squares;
            for tri in fill_unit_triangles(&verts, &occupied) {
                let c = (tri[0] + tri[1] + tri[2]) * (1.0 / 3.0);
                if c.x >= -1e-9 && c.x < l - 1e-9 && c.y >= -1e-9 && c.y < l - 1e-9 {
                    faces.push(tri);
                }
            }
            (Vec2::new(l, 0.0), Vec2::new(0.0, l), faces)
        }
        Archimedean::T3_3_3_3_6 => {
            let va = Vec2::new(2.5, s3 / 2.0);
            let vb = Vec2::new(0.5, 3.0 * s3 / 2.0);
            let hex = regular_polygon(Vec2::ZERO, 1.0, 6, 0.0);
            let mut verts = Vec::new();
            let mut occupied = Vec::new();
            for i in -1..=1 {
                for j in -1..=1 {
                    let off = va * f64::from(i) + vb * f64::from(j);
                    let moved: Vec<Vec2> = hex.iter().map(|&p| p + off).collect();
                    verts.extend(moved.iter().copied());
                    occupied.push(moved);
                }
            }
            let mut faces = vec![hex];
            for tri in fill_unit_triangles(&verts, &occupied) {
                let c = (tri[0] + tri[1] + tri[2]) * (1.0 / 3.0);
                // Keep triangles whose centroid falls in the base
                // cell (lattice coordinates in [0, 1)).
                let det = va.x * vb.y - va.y * vb.x;
                let u = (c.x * vb.y - c.y * vb.x) / det;
                let v = (va.x * c.y - va.y * c.x) / det;
                if (-1e-9..1.0 - 1e-9).contains(&u) && (-1e-9..1.0 - 1e-9).contains(&v) {
                    faces.push(tri);
                }
            }
            (va, vb, faces)
        }
    }
}

/// Replicates a periodic cell over the extent, keeping faces whose
/// centroid lies inside it. `size` scales the edge length.
fn replicate(
    va: Vec2,
    vb: Vec2,
    cell_faces: &[Vec<Vec2>],
    extent: &Rect,
    size: f64,
) -> Tiling {
    assert!(size > 0.0, "tile size must be positive");
    let (va, vb) = (va * size, vb * size);
    let det = va.x * vb.y - va.y * vb.x;
    // Lattice index bounds from the extent corners.
    let (mut lo_i, mut hi_i) = (i64::MAX, i64::MIN);
    let (mut lo_j, mut hi_j) = (i64::MAX, i64::MIN);
    for corner in extent.corners() {
        let u = (corner.x * vb.y - corner.y * vb.x) / det;
        let v = (va.x * corner.y - va.y * corner.x) / det;
        lo_i = lo_i.min(u.floor() as i64 - 2);
        hi_i = hi_i.max(u.ceil() as i64 + 2);
        lo_j = lo_j.min(v.floor() as i64 - 2);
        hi_j = hi_j.max(v.ceil() as i64 + 2);
    }
    let mut builder = TilingBuilder::new(1e-6 * size);
    for i in lo_i..=hi_i {
        for j in lo_j..=hi_j {
            let off = va * i as f64 + vb * j as f64;
            for face in cell_faces {
                let moved: Vec<Vec2> = face.iter().map(|&p| p * size + off).collect();
                let c = moved.iter().fold(Vec2::ZERO, |s, &p| s + p) * (1.0 / moved.len() as f64);
                if extent.contains_point(c) {
                    builder.face(&moved);
                }
            }
        }
    }
    builder.build()
}

/// Square grid of `nx` x `ny` cells with the given cell size.
#[must_use]
pub fn square_grid(nx: usize, ny: usize, size: f64) -> Tiling {
    assert!(nx >= 1 && ny >= 1 && size > 0.0, "invalid grid");
    let mut b = TilingBuilder::new(1e-9 * size);
    for j in 0..ny {
        for i in 0..nx {
            let p = Vec2::new(i as f64 * size, j as f64 * size);
            b.face(&[
                p,
                p + Vec2::new(size, 0.0),
                p + Vec2::new(size, size),
                p + Vec2::new(0.0, size),
            ]);
        }
    }
    b.build()
}

/// Triangular grid: `nx` x `ny` rhombi split into unit triangles of
/// the given edge length.
#[must_use]
pub fn triangular_grid(nx: usize, ny: usize, size: f64) -> Tiling {
    assert!(nx >= 1 && ny >= 1 && size > 0.0, "invalid grid");
    let s3 = 3.0f64.sqrt();
    let mut b = TilingBuilder::new(1e-9 * size);
    for j in 0..ny {
        for i in 0..nx {
            let p = Vec2::new((i as f64 + j as f64 * 0.5) * size, j as f64 * s3 / 2.0 * size);
            let right = Vec2::new(size, 0.0);
            let up = Vec2::new(size * 0.5, size * s3 / 2.0);
            b.face(&[p, p + right, p + up]);
            b.face(&[p + right, p + right + up, p + up]);
        }
    }
    b.build()
}

/// Hexagonal grid: `nx` x `ny` hexagons of circumradius `size`.
/// `pointy_top` orients a vertex upward; otherwise an edge is up.
#[must_use]
pub fn hexagonal_grid(nx: usize, ny: usize, size: f64, pointy_top: bool) -> Tiling {
    assert!(nx >= 1 && ny >= 1 && size > 0.0, "invalid grid");
    let mut b = TilingBuilder::new(1e-9 * size);
    let s3 = 3.0f64.sqrt();
    for j in 0..ny {
        for i in 0..nx {
            let center = if pointy_top {
                Vec2::new(
                    (i as f64 + 0.5 * (j % 2) as f64) * s3 * size,
                    j as f64 * 1.5 * size,
                )
            } else {
                Vec2::new(
                    i as f64 * 1.5 * size,
                    (j as f64 + 0.5 * (i % 2) as f64) * s3 * size,
                )
            };
            let angle0 = if pointy_top { std::f64::consts::FRAC_PI_2 } else { 0.0 };
            b.face(&regular_polygon(center, size, 6, angle0));
        }
    }
    b.build()
}

/// Archimedean (uniform) tiling of the given kind with edge length
/// `size`, covering `extent` (faces with centroid inside).
#[must_use]
pub fn archimedean(kind: Archimedean, extent: &Rect, size: f64) -> Tiling {
    let (va, vb, faces) = archimedean_cell(kind);
    replicate(va, vb, &faces, extent, size)
}

/// Laves tiling: the dual of the corresponding Archimedean tiling.
#[must_use]
pub fn laves(kind: Archimedean, extent: &Rect, size: f64) -> Tiling {
    // Build on a padded extent so the dual covers the requested one.
    let pad = 4.0 * size;
    let padded = Rect {
        min: extent.min - Vec2::new(pad, pad),
        max: extent.max + Vec2::new(pad, pad),
    };
    archimedean(kind, &padded, size).dual().clip_to_rect(&padded)
}

/// Axial hex-grid coordinate (Red Blob Games convention); the third
/// cube coordinate is `s = -q - r`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hex {
    pub q: i32,
    pub r: i32,
}

impl Hex {
    #[must_use]
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Third cube coordinate.
    #[must_use]
    pub fn s(&self) -> i32 {
        -self.q - self.r
    }

    #[must_use]
    pub fn add(&self, other: Hex) -> Hex {
        Hex::new(self.q + other.q, self.r + other.r)
    }

    #[must_use]
    pub fn sub(&self, other: Hex) -> Hex {
        Hex::new(self.q - other.q, self.r - other.r)
    }

    #[must_use]
    pub fn scale(&self, k: i32) -> Hex {
        Hex::new(self.q * k, self.r * k)
    }

    /// The six neighbors, counterclockwise from +q.
    #[must_use]
    pub fn neighbors(&self) -> [Hex; 6] {
        [
            self.add(Hex::new(1, 0)),
            self.add(Hex::new(1, -1)),
            self.add(Hex::new(0, -1)),
            self.add(Hex::new(-1, 0)),
            self.add(Hex::new(-1, 1)),
            self.add(Hex::new(0, 1)),
        ]
    }

    /// Hex (cube) distance.
    #[must_use]
    pub fn distance(&self, other: Hex) -> i32 {
        let d = self.sub(other);
        (d.q.abs() + d.r.abs() + d.s().abs()) / 2
    }

    /// Center position of the hex cell.
    #[must_use]
    pub fn to_pixel(&self, size: f64, pointy: bool) -> Vec2 {
        let (q, r) = (f64::from(self.q), f64::from(self.r));
        let s3 = 3.0f64.sqrt();
        if pointy {
            Vec2::new(size * (s3 * q + s3 / 2.0 * r), size * 1.5 * r)
        } else {
            Vec2::new(size * 1.5 * q, size * (s3 / 2.0 * q + s3 * r))
        }
    }

    /// Inverse of [`Hex::to_pixel`] with cube rounding.
    #[must_use]
    pub fn from_pixel(p: Vec2, size: f64, pointy: bool) -> Hex {
        let s3 = 3.0f64.sqrt();
        let (qf, rf) = if pointy {
            (
                (s3 / 3.0 * p.x - p.y / 3.0) / size,
                2.0 / 3.0 * p.y / size,
            )
        } else {
            (
                2.0 / 3.0 * p.x / size,
                (-p.x / 3.0 + s3 / 3.0 * p.y) / size,
            )
        };
        // Cube round.
        let sf = -qf - rf;
        let (mut q, mut r, s) = (qf.round(), rf.round(), sf.round());
        let (dq, dr, ds) = ((q - qf).abs(), (r - rf).abs(), (s - sf).abs());
        if dq > dr && dq > ds {
            q = -r - s;
        } else if dr > ds {
            r = -q - s;
        }
        Hex::new(q as i32, r as i32)
    }

    /// The ring of hexes at exactly the given radius.
    ///
    /// # Panics
    /// Panics for negative radius.
    #[must_use]
    pub fn ring(&self, radius: i32) -> Vec<Hex> {
        assert!(radius >= 0, "radius must be nonnegative");
        if radius == 0 {
            return vec![*self];
        }
        let dirs = [
            Hex::new(1, 0),
            Hex::new(1, -1),
            Hex::new(0, -1),
            Hex::new(-1, 0),
            Hex::new(-1, 1),
            Hex::new(0, 1),
        ];
        let mut out = Vec::with_capacity(6 * radius as usize);
        let mut h = self.add(dirs[4].scale(radius));
        for dir in dirs {
            for _ in 0..radius {
                out.push(h);
                h = h.add(dir);
            }
        }
        out
    }

    /// All hexes within the radius, spiraling outward ring by ring.
    ///
    /// # Panics
    /// Panics for negative radius.
    #[must_use]
    pub fn spiral(&self, radius: i32) -> Vec<Hex> {
        let mut out = Vec::new();
        for r in 0..=radius {
            out.extend(self.ring(r));
        }
        out
    }

    /// Hexes on the line to `other` (inclusive), by cube
    /// interpolation and rounding.
    #[must_use]
    pub fn line_to(&self, other: Hex) -> Vec<Hex> {
        let n = self.distance(other).max(1);
        (0..=n)
            .map(|k| {
                let t = f64::from(k) / f64::from(n);
                // Lerp in fractional pixel space (pointy, unit size).
                let a = self.to_pixel(1.0, true);
                let b = other.to_pixel(1.0, true);
                Hex::from_pixel(a.lerp(&b, t), 1.0, true)
            })
            .collect()
    }

    /// Rotation by 60° counterclockwise about the origin.
    #[must_use]
    pub fn rotate60(&self) -> Hex {
        Hex::new(-self.r, self.q + self.r)
    }

    /// Reflection fixing the q axis.
    #[must_use]
    pub fn reflect_q(&self) -> Hex {
        Hex::new(self.q, self.s())
    }
}

/// All hexes within `radius` of `center` (hex-distance ball).
///
/// # Panics
/// Panics for negative radius.
#[must_use]
pub fn hex_range(center: Hex, radius: i32) -> Vec<Hex> {
    assert!(radius >= 0, "radius must be nonnegative");
    let mut out = Vec::new();
    for q in -radius..=radius {
        let lo = (-radius).max(-q - radius);
        let hi = radius.min(-q + radius);
        for r in lo..=hi {
            out.push(center.add(Hex::new(q, r)));
        }
    }
    out
}

/// Cairo pentagonal tiling: the dual of the snub square tiling.
#[must_use]
pub fn cairo_pentagonal(extent: &Rect, size: f64) -> Tiling {
    laves(Archimedean::T3_3_4_3_4, extent, size)
}

/// Rhombille (tumbling blocks) tiling: the dual of the trihexagonal
/// tiling.
#[must_use]
pub fn rhombille(extent: &Rect, size: f64) -> Tiling {
    laves(Archimedean::T3_6_3_6, extent, size)
}

/// Running-bond brick pattern: rows of `w` x `h` bricks, each row
/// shifted by `offset` (in units of `w`).
///
/// # Panics
/// Panics unless `w, h > 0`.
#[must_use]
pub fn brick(extent: &Rect, w: f64, h: f64, offset: f64) -> Tiling {
    assert!(w > 0.0 && h > 0.0, "brick size must be positive");
    let mut b = TilingBuilder::new(1e-9 * w.min(h));
    let rows = (((extent.max.y - extent.min.y) / h).ceil() as i64) + 1;
    let cols = (((extent.max.x - extent.min.x) / w).ceil() as i64) + 2;
    for j in 0..rows {
        let y = extent.min.y + j as f64 * h;
        let shift = (j as f64 * offset * w).rem_euclid(w);
        for i in -1..cols {
            let x = extent.min.x + i as f64 * w + shift;
            let c = Vec2::new(x + w / 2.0, y + h / 2.0);
            if extent.contains_point(c) {
                b.face(&[
                    Vec2::new(x, y),
                    Vec2::new(x + w, y),
                    Vec2::new(x + w, y + h),
                    Vec2::new(x, y + h),
                ]);
            }
        }
    }
    b.build()
}

/// Herringbone pattern of `w` x `h` bricks (alternating horizontal
/// and vertical along the diagonals).
///
/// # Panics
/// Panics unless `0 < h < w`.
#[must_use]
pub fn herringbone(extent: &Rect, w: f64, h: f64) -> Tiling {
    assert!(h > 0.0 && w > h, "requires 0 < h < w");
    // One horizontal + one vertical brick per cell; diagonal lattice.
    let hbrick = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(w, 0.0),
        Vec2::new(w, h),
        Vec2::new(0.0, h),
    ];
    let vbrick = vec![
        Vec2::new(w, h - w),
        Vec2::new(w + h, h - w),
        Vec2::new(w + h, h),
        Vec2::new(w, h),
    ];
    let va = Vec2::new(h, h);
    let vb = Vec2::new(w + h, -(w - h));
    replicate(va, vb, &[hbrick, vbrick], extent, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cyclic sequence of face sizes around each interior vertex,
    /// compared against the expected configuration (any rotation or
    /// reflection).
    fn check_vertex_config(t: &Tiling, expected: &[usize]) -> usize {
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); t.vertices.len()];
        let mut angle_sum = vec![0.0f64; t.vertices.len()];
        for (fi, f) in t.faces.iter().enumerate() {
            let m = f.len();
            for k in 0..m {
                let v = f[k];
                incident[v].push(fi);
                let prev = t.vertices[f[(k + m - 1) % m]];
                let next = t.vertices[f[(k + 1) % m]];
                let p = t.vertices[v];
                angle_sum[v] += (prev - p).angle_between(&(next - p));
            }
        }
        let centroids = t.face_centroids();
        let mut checked = 0;
        for (v, faces) in incident.iter().enumerate() {
            if (angle_sum[v] - 2.0 * std::f64::consts::PI).abs() > 1e-6 {
                continue; // boundary vertex
            }
            let p = t.vertices[v];
            let mut ordered: Vec<(f64, usize)> = faces
                .iter()
                .map(|&fi| {
                    let d = centroids[fi] - p;
                    (d.y.atan2(d.x), t.faces[fi].len())
                })
                .collect();
            ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
            let sizes: Vec<usize> = ordered.iter().map(|&(_, s)| s).collect();
            let n = sizes.len();
            assert_eq!(n, expected.len(), "vertex degree mismatch: {sizes:?}");
            let matches = (0..n).any(|shift| {
                (0..n).all(|k| sizes[(shift + k) % n] == expected[k])
                    || (0..n).all(|k| sizes[(shift + n - k) % n] == expected[k])
            });
            assert!(matches, "vertex config {sizes:?} != {expected:?}");
            checked += 1;
        }
        checked
    }

    #[test]
    fn test_regular_grids() {
        let sq = square_grid(4, 3, 1.0);
        assert_eq!(sq.faces.len(), 12);
        assert_eq!(sq.vertices.len(), 20);
        assert_eq!(check_vertex_config(&sq, &[4, 4, 4, 4]), 6);

        let tri = triangular_grid(6, 6, 1.0);
        assert!(check_vertex_config(&tri, &[3, 3, 3, 3, 3, 3]) > 10);

        let hexp = hexagonal_grid(5, 5, 1.0, true);
        assert_eq!(hexp.faces.len(), 25);
        assert!(check_vertex_config(&hexp, &[6, 6, 6]) > 10);
        let hexf = hexagonal_grid(5, 5, 1.0, false);
        assert!(check_vertex_config(&hexf, &[6, 6, 6]) > 10);
    }

    #[test]
    fn test_archimedean_vertex_configs() {
        let extent = Rect { min: Vec2::new(-8.0, -8.0), max: Vec2::new(8.0, 8.0) };
        let cases: Vec<(Archimedean, Vec<usize>)> = vec![
            (Archimedean::T3_3_3_3_3_3, vec![3, 3, 3, 3, 3, 3]),
            (Archimedean::T4_4_4_4, vec![4, 4, 4, 4]),
            (Archimedean::T6_6_6, vec![6, 6, 6]),
            (Archimedean::T3_3_3_3_6, vec![3, 3, 3, 3, 6]),
            (Archimedean::T3_3_3_4_4, vec![3, 3, 3, 4, 4]),
            (Archimedean::T3_3_4_3_4, vec![3, 3, 4, 3, 4]),
            (Archimedean::T3_4_6_4, vec![3, 4, 6, 4]),
            (Archimedean::T3_6_3_6, vec![3, 6, 3, 6]),
            (Archimedean::T3_12_12, vec![3, 12, 12]),
            (Archimedean::T4_6_12, vec![4, 6, 12]),
            (Archimedean::T4_8_8, vec![4, 8, 8]),
        ];
        for (kind, expected) in cases {
            let t = archimedean(kind, &extent, 1.0);
            let interior = check_vertex_config(&t, &expected);
            assert!(interior >= 4, "{kind:?}: only {interior} interior vertices checked");
            // All edges have unit length.
            for &(a, b) in &t.edges {
                assert!(
                    (t.vertices[a].distance_to(&t.vertices[b]) - 1.0).abs() < 1e-6,
                    "{kind:?}: non-unit edge"
                );
            }
        }
    }

    #[test]
    fn test_dual_and_laves() {
        let extent = Rect { min: Vec2::new(-6.0, -6.0), max: Vec2::new(6.0, 6.0) };
        // Dual of the square tiling is a square tiling: every dual
        // face is a quad.
        let sq = archimedean(Archimedean::T4_4_4_4, &extent, 1.0);
        let d = sq.dual();
        assert!(!d.faces.is_empty());
        assert!(d.faces.iter().all(|f| f.len() == 4));
        // Dual of dual lands back on original vertices.
        let dd = d.dual();
        assert!(!dd.faces.is_empty());
        for &v in dd.faces.iter().flatten() {
            let p = dd.vertices[v];
            let nearest = sq
                .vertices
                .iter()
                .map(|q| q.distance_to(&p))
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 1e-6, "dual of dual strays from original vertices");
        }
        // Laves duals: cairo pentagons (5-gons), rhombille rhombi
        // (4-gons).
        let cairo = cairo_pentagonal(&extent, 1.0);
        assert!(!cairo.faces.is_empty());
        assert!(cairo.faces.iter().all(|f| f.len() == 5), "cairo tiles are pentagons");
        let rh = rhombille(&extent, 1.0);
        assert!(!rh.faces.is_empty());
        assert!(rh.faces.iter().all(|f| f.len() == 4), "rhombille tiles are rhombi");
        for f in &rh.faces {
            let d1 = rh.vertices[f[0]].distance_to(&rh.vertices[f[1]]);
            let d2 = rh.vertices[f[1]].distance_to(&rh.vertices[f[2]]);
            assert!((d1 - d2).abs() < 1e-6, "rhombus sides equal");
        }
    }

    #[test]
    fn test_hex_algebra() {
        let h = Hex::new(3, -2);
        assert_eq!(h.s(), -1);
        assert_eq!(h.add(Hex::new(1, 1)), Hex::new(4, -1));
        assert_eq!(h.distance(Hex::new(0, 0)), 3);
        // Round trips for both orientations.
        for pointy in [true, false] {
            for q in -5..=5 {
                for r in -5..=5 {
                    let h = Hex::new(q, r);
                    let p = h.to_pixel(0.8, pointy);
                    assert_eq!(Hex::from_pixel(p, 0.8, pointy), h, "pixel round trip");
                }
            }
        }
        // Triangle inequality.
        let (a, b, c) = (Hex::new(0, 0), Hex::new(3, -1), Hex::new(-2, 4));
        assert!(a.distance(c) <= a.distance(b) + b.distance(c));
        // Neighbors are at distance 1 and adjacent pixels.
        for n in h.neighbors() {
            assert_eq!(h.distance(n), 1);
        }
        // Ring and spiral counts.
        assert_eq!(h.ring(3).len(), 18);
        assert_eq!(h.spiral(3).len(), 1 + 6 + 12 + 18);
        assert_eq!(hex_range(h, 3).len(), 37);
        // Line endpoints and step size.
        let line = a.line_to(b);
        assert_eq!(*line.first().unwrap(), a);
        assert_eq!(*line.last().unwrap(), b);
        for w in line.windows(2) {
            assert_eq!(w[0].distance(w[1]), 1);
        }
        // Rotation has order 6; reflection is an involution.
        let mut r6 = h;
        for _ in 0..6 {
            r6 = r6.rotate60();
        }
        assert_eq!(r6, h);
        assert_eq!(h.reflect_q().reflect_q(), h);
    }

    #[test]
    fn test_brick_and_herringbone_cover() {
        let extent = Rect { min: Vec2::ZERO, max: Vec2::new(10.0, 6.0) };
        let b = brick(&extent, 2.0, 1.0, 0.5);
        assert!(!b.faces.is_empty());
        let total: f64 = b.polygons().iter().map(Polygon2::area).sum();
        // Bricks with centroid inside cover the extent up to boundary
        // overhang (at most half a brick per boundary brick).
        assert!(total < 70.0);
        assert!(total > 45.0);

        let hb = herringbone(&extent, 2.0, 1.0);
        assert!(!hb.faces.is_empty());
        // Point-coverage check well inside the extent (where every
        // covering brick's centroid is inside too): each point lies
        // in exactly one brick — no overlaps, no gaps.
        let polys = hb.polygons();
        let mut state = 7u64;
        let mut rand = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        for _ in 0..300 {
            let p = Vec2::new(3.0 + rand() * 4.0, 2.0 + rand() * 2.0);
            let count = polys
                .iter()
                .filter(|poly| {
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
                })
                .count();
            assert_eq!(count, 1, "herringbone coverage broken at {p:?}");
        }
    }
}
