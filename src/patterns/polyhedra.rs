//! Polyhedra: Platonic/Archimedean/Catalan/Johnson solids, Goldberg
//! and geodesic polyhedra, and Conway polyhedron operators.

use crate::error::GeomError;
use crate::math::Vec3;
use crate::mesh::Mesh;
use crate::patterns::symmetry::PointGroup3;
use std::collections::HashMap;

/// A polyhedron with polygonal faces (counterclockwise seen from
/// outside).
#[derive(Debug, Clone, PartialEq)]
pub struct Polyhedron {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<Vec<usize>>,
}

fn newell_normal(pts: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        n = n + Vec3::new(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    n
}

impl Polyhedron {
    /// Fan-triangulates every face into a [`Mesh`].
    #[must_use]
    pub fn to_mesh(&self) -> Mesh {
        let mut indices = Vec::new();
        for f in &self.faces {
            for k in 1..f.len() - 1 {
                indices.push([f[0], f[k], f[k + 1]]);
            }
        }
        Mesh {
            vertices: self.vertices.clone(),
            indices,
            normals: None,
            uvs: None,
        }
    }

    /// Unique undirected edges, sorted.
    #[must_use]
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut e: Vec<(usize, usize)> = self
            .faces
            .iter()
            .flat_map(|f| {
                (0..f.len()).map(move |k| {
                    let (a, b) = (f[k], f[(k + 1) % f.len()]);
                    (a.min(b), a.max(b))
                })
            })
            .collect();
        e.sort_unstable();
        e.dedup();
        e
    }

    /// Euler characteristic V − E + F.
    #[must_use]
    pub fn euler(&self) -> i64 {
        self.vertices.len() as i64 - self.edges().len() as i64 + self.faces.len() as i64
    }

    /// Centroid of every face.
    #[must_use]
    pub fn face_centroids(&self) -> Vec<Vec3> {
        self.faces
            .iter()
            .map(|f| {
                f.iter()
                    .map(|&v| self.vertices[v])
                    .fold(Vec3::ZERO, |s, p| s + p)
                    * (1.0 / f.len() as f64)
            })
            .collect()
    }

    /// Unit face normals (Newell's method, robust for near-planar
    /// polygons).
    #[must_use]
    pub fn face_normals(&self) -> Vec<Vec3> {
        self.faces
            .iter()
            .map(|f| {
                let pts: Vec<Vec3> = f.iter().map(|&v| self.vertices[v]).collect();
                newell_normal(&pts).normalized()
            })
            .collect()
    }

    /// Signed volume (positive for outward-facing faces).
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.to_mesh().volume()
    }

    /// Total face area.
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        self.to_mesh().surface_area()
    }

    /// True when every vertex lies on or behind every face plane.
    #[must_use]
    pub fn is_convex(&self) -> bool {
        let normals = self.face_normals();
        let centroids = self.face_centroids();
        let scale = self
            .vertices
            .iter()
            .map(|v| v.magnitude())
            .fold(0.0f64, f64::max)
            .max(1e-12);
        for (f, (n, c)) in normals.iter().zip(&centroids).enumerate() {
            let _ = f;
            for v in &self.vertices {
                if (*v - *c).dot(n) > 1e-8 * scale {
                    return false;
                }
            }
        }
        true
    }

    /// Map from directed edge (a, b) to the face containing it.
    fn directed_edge_faces(&self) -> HashMap<(usize, usize), usize> {
        let mut map = HashMap::new();
        for (fi, f) in self.faces.iter().enumerate() {
            for k in 0..f.len() {
                map.insert((f[k], f[(k + 1) % f.len()]), fi);
            }
        }
        map
    }

    /// Faces around vertex `v` in rotational order (walking across
    /// shared edges).
    ///
    /// # Panics
    /// Panics when the polyhedron is not closed (a directed edge has
    /// no partner) or `v` is unused.
    #[must_use]
    pub fn faces_around_vertex(&self, v: usize) -> Vec<usize> {
        let def = self.directed_edge_faces();
        // Successor of v in face f.
        let next_in = |fi: usize| -> usize {
            let f = &self.faces[fi];
            let k = f.iter().position(|&x| x == v).expect("vertex in face");
            f[(k + 1) % f.len()]
        };
        let start = self
            .faces
            .iter()
            .position(|f| f.contains(&v))
            .expect("vertex used by some face");
        let mut order = vec![start];
        let mut cur = start;
        loop {
            let n = next_in(cur);
            let next_face = *def.get(&(n, v)).expect("closed polyhedron");
            if next_face == start {
                break;
            }
            order.push(next_face);
            cur = next_face;
            assert!(order.len() <= self.faces.len(), "broken vertex fan");
        }
        order
    }

    /// The neighbors of `v` in rotational order (the vertex figure).
    ///
    /// # Panics
    /// Panics when the polyhedron is not closed or `v` is unused.
    #[must_use]
    pub fn vertex_figure(&self, v: usize) -> Vec<usize> {
        let order = self.faces_around_vertex(v);
        order
            .iter()
            .map(|&fi| {
                let f = &self.faces[fi];
                let k = f.iter().position(|&x| x == v).expect("vertex in face");
                f[(k + 1) % f.len()]
            })
            .collect()
    }

    /// Ensures faces are counterclockwise seen from outside (flips
    /// faces whose normal points inward relative to the centroid) —
    /// valid for star-shaped solids, which covers everything built
    /// here.
    fn orient_outward(mut self) -> Self {
        let center = self.vertices.iter().fold(Vec3::ZERO, |s, &p| s + p)
            * (1.0 / self.vertices.len() as f64);
        let centroids = self.face_centroids();
        let normals = self.face_normals();
        for (f, (n, c)) in self.faces.iter_mut().zip(normals.iter().zip(&centroids)) {
            if n.dot(&(*c - center)) < 0.0 {
                f.reverse();
            }
        }
        self
    }

    /// Scales every vertex to the given distance from the origin.
    ///
    /// # Panics
    /// Panics unless `radius > 0`.
    #[must_use]
    pub fn normalize(&self, radius: f64) -> Polyhedron {
        assert!(radius > 0.0, "radius must be positive");
        Polyhedron {
            vertices: self
                .vertices
                .iter()
                .map(|v| v.normalized() * radius)
                .collect(),
            faces: self.faces.clone(),
        }
    }

    /// Hart's canonical form iteration: edges tangent to the unit
    /// sphere, faces planar, centroid at the origin. Converges to the
    /// canonical (maximally symmetric) shape for convex polyhedra.
    #[must_use]
    pub fn canonicalize(&self, iterations: usize) -> Polyhedron {
        let mut p = self.clone();
        let edges = p.edges();
        for _ in 0..iterations {
            // Tangentify: nudge edges toward unit distance from the
            // origin.
            for &(a, b) in &edges {
                let (va, vb) = (p.vertices[a], p.vertices[b]);
                let e = vb - va;
                let t = (-va.dot(&e) / e.magnitude_squared()).clamp(0.0, 1.0);
                let c = va + e * t;
                let m = c.magnitude();
                if m > 1e-12 {
                    let corr = c * ((1.0 - m) / m * 0.5);
                    p.vertices[a] = p.vertices[a] + corr;
                    p.vertices[b] = p.vertices[b] + corr;
                }
            }
            // Recenter.
            let center =
                p.vertices.iter().fold(Vec3::ZERO, |s, &v| s + v) * (1.0 / p.vertices.len() as f64);
            for v in &mut p.vertices {
                *v = *v - center;
            }
            // Planarize: project vertices toward each face's best
            // plane.
            let centroids = p.face_centroids();
            let normals = p.face_normals();
            let mut acc = vec![(Vec3::ZERO, 0.0f64); p.vertices.len()];
            for (f, (n, c)) in p.faces.iter().zip(normals.iter().zip(&centroids)) {
                for &v in f {
                    let d = (p.vertices[v] - *c).dot(n);
                    acc[v].0 = acc[v].0 + (p.vertices[v] - *n * d);
                    acc[v].1 += 1.0;
                }
            }
            for (v, (sum, count)) in p.vertices.iter_mut().zip(&acc) {
                if *count > 0.0 {
                    *v = *v * 0.5 + *sum * (0.5 / count);
                }
            }
        }
        p
    }

    /// Dual polyhedron: face centroids become vertices, vertices
    /// become faces.
    #[must_use]
    pub fn dual(&self) -> Polyhedron {
        let centroids = self.face_centroids();
        let faces = (0..self.vertices.len())
            .map(|v| self.faces_around_vertex(v))
            .collect();
        Polyhedron {
            vertices: centroids,
            faces,
        }
        .orient_outward()
    }

    /// Detects the rotational symmetry group of the vertex set:
    /// tetrahedral, octahedral, icosahedral, or a single-axis
    /// cyclic/dihedral group. Returns `None` when no nontrivial
    /// rotation (beyond identity, up to order 12 axes) fits.
    #[must_use]
    pub fn symmetry_group(&self) -> Option<PointGroup3> {
        let scale = self
            .vertices
            .iter()
            .map(|v| v.magnitude())
            .fold(0.0f64, f64::max)
            .max(1e-12);
        let tol = 1e-6 * scale;
        let maps_to_self = |axis: Vec3, angle: f64| -> bool {
            let q = crate::quaternion::Quaternion::from_axis_angle(axis, angle);
            self.vertices.iter().all(|&v| {
                let w = q.rotate_vec(v);
                self.vertices.iter().any(|&u| u.distance_to(&w) <= tol)
            })
        };
        // Candidate axes: vertices, face centroids, edge midpoints.
        let mut axes: Vec<Vec3> = Vec::new();
        for v in &self.vertices {
            axes.push(*v);
        }
        for c in self.face_centroids() {
            axes.push(c);
        }
        for (a, b) in self.edges() {
            axes.push((self.vertices[a] + self.vertices[b]) * 0.5);
        }
        axes.retain(|a| a.magnitude() > tol);
        // Highest rotation order per axis (deduplicated by direction).
        let mut best: Vec<(Vec3, u32)> = Vec::new();
        for a in axes {
            let dir = a.normalized();
            if best.iter().any(|(d, _)| d.cross(&dir).magnitude() < 1e-6) {
                continue;
            }
            let mut order = 1u32;
            for n in 2..=12u32 {
                if maps_to_self(dir, 2.0 * std::f64::consts::PI / f64::from(n)) {
                    order = order.max(n);
                }
            }
            if order > 1 {
                best.push((dir, order));
            }
        }
        let count_order = |k: u32| best.iter().filter(|(_, o)| *o % k == 0 && *o >= k).count();
        let has5 = best.iter().any(|(_, o)| *o % 5 == 0);
        let max_order = best.iter().map(|(_, o)| *o).max().unwrap_or(1);
        if has5 && count_order(3) >= 4 {
            return Some(PointGroup3::Icosahedral);
        }
        if best.iter().any(|(_, o)| *o % 4 == 0) && count_order(3) >= 4 {
            return Some(PointGroup3::Octahedral);
        }
        if count_order(3) >= 4 {
            return Some(PointGroup3::Tetrahedral);
        }
        if max_order > 1 {
            let (main, order) = *best.iter().max_by_key(|(_, o)| *o).expect("nonempty");
            // Perpendicular twofold axes make it dihedral.
            let dihedral = best
                .iter()
                .any(|(d, o)| *o % 2 == 0 && d.dot(&main).abs() < 1e-6);
            return Some(if dihedral {
                PointGroup3::Dn(order)
            } else {
                PointGroup3::Cn(order)
            });
        }
        None
    }
}

// ---------------------------------------------------------------------
// Conway operator machinery: vertices are created on demand under
// structural keys so shared vertices unify exactly.
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum VKey {
    /// Original vertex.
    V(usize),
    /// Face centroid (optionally lifted).
    C(usize),
    /// Point on directed edge (a, b) (e.g. one third along).
    E(usize, usize),
    /// Per-face, per-corner point.
    I(usize, usize),
}

struct FlagBuilder {
    map: HashMap<VKey, usize>,
    positions: Vec<Vec3>,
    faces: Vec<Vec<usize>>,
}

impl FlagBuilder {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            positions: Vec::new(),
            faces: Vec::new(),
        }
    }

    fn vertex(&mut self, key: VKey, pos: Vec3) -> usize {
        *self.map.entry(key).or_insert_with(|| {
            self.positions.push(pos);
            self.positions.len() - 1
        })
    }

    fn face(&mut self, ids: Vec<usize>) {
        self.faces.push(ids);
    }

    fn build(self) -> Polyhedron {
        Polyhedron {
            vertices: self.positions,
            faces: self.faces,
        }
        .orient_outward()
    }
}

/// Conway dual (alias of [`Polyhedron::dual`]).
#[must_use]
pub fn conway_dual(p: &Polyhedron) -> Polyhedron {
    p.dual()
}

/// Conway kis: a pyramid of the given apex height over every face.
#[must_use]
pub fn kis(p: &Polyhedron, apex_height: f64) -> Polyhedron {
    let centroids = p.face_centroids();
    let normals = p.face_normals();
    let mut b = FlagBuilder::new();
    for (fi, f) in p.faces.iter().enumerate() {
        let apex = b.vertex(VKey::C(fi), centroids[fi] + normals[fi] * apex_height);
        for k in 0..f.len() {
            let (v, w) = (f[k], f[(k + 1) % f.len()]);
            let a = b.vertex(VKey::V(v), p.vertices[v]);
            let c = b.vertex(VKey::V(w), p.vertices[w]);
            b.face(vec![a, c, apex]);
        }
    }
    b.build()
}

/// Conway ambo (rectification): vertices at edge midpoints.
#[must_use]
pub fn ambo(p: &Polyhedron) -> Polyhedron {
    let mid = |a: usize, c: usize| (p.vertices[a] + p.vertices[c]) * 0.5;
    let ekey = |a: usize, c: usize| VKey::E(a.min(c), a.max(c));
    let mut b = FlagBuilder::new();
    for f in &p.faces {
        let ids: Vec<usize> = (0..f.len())
            .map(|k| {
                let (v, w) = (f[k], f[(k + 1) % f.len()]);
                b.vertex(ekey(v, w), mid(v, w))
            })
            .collect();
        b.face(ids);
    }
    for v in 0..p.vertices.len() {
        let around = p.faces_around_vertex(v);
        let ids: Vec<usize> = around
            .iter()
            .map(|&fi| {
                let f = &p.faces[fi];
                let k = f.iter().position(|&x| x == v).expect("vertex in face");
                let w = f[(k + 1) % f.len()];
                b.vertex(ekey(v, w), mid(v, w))
            })
            .collect();
        b.face(ids);
    }
    b.build()
}

/// Conway truncate: cuts each corner, moving `ratio` along every
/// edge (1/3 turns regular triangles into regular hexagons).
///
/// # Panics
/// Panics unless `0 < ratio < 1/2`.
#[must_use]
pub fn truncate(p: &Polyhedron, ratio: f64) -> Polyhedron {
    assert!(ratio > 0.0 && ratio < 0.5, "truncation ratio in (0, 1/2)");
    let cut = |a: usize, c: usize| p.vertices[a] + (p.vertices[c] - p.vertices[a]) * ratio;
    let mut b = FlagBuilder::new();
    for f in &p.faces {
        let mut ids = Vec::with_capacity(2 * f.len());
        for k in 0..f.len() {
            let (v, w) = (f[k], f[(k + 1) % f.len()]);
            ids.push(b.vertex(VKey::E(v, w), cut(v, w)));
            ids.push(b.vertex(VKey::E(w, v), cut(w, v)));
        }
        b.face(ids);
    }
    for v in 0..p.vertices.len() {
        let around = p.faces_around_vertex(v);
        let ids: Vec<usize> = around
            .iter()
            .map(|&fi| {
                let f = &p.faces[fi];
                let k = f.iter().position(|&x| x == v).expect("vertex in face");
                let w = f[(k + 1) % f.len()];
                b.vertex(VKey::E(v, w), cut(v, w))
            })
            .collect();
        b.face(ids);
    }
    b.build()
}

/// Conway chamfer: shrinks faces in-plane by `ratio` and replaces
/// each edge with a hexagon (original vertices kept).
///
/// # Panics
/// Panics unless `0 < ratio < 1`.
#[must_use]
pub fn chamfer(p: &Polyhedron, ratio: f64) -> Polyhedron {
    assert!(ratio > 0.0 && ratio < 1.0, "chamfer ratio in (0, 1)");
    let centroids = p.face_centroids();
    let def = p.directed_edge_faces();
    let mut b = FlagBuilder::new();
    let shrunk = |fi: usize, v: usize| -> Vec3 {
        centroids[fi] + (p.vertices[v] - centroids[fi]) * (1.0 - ratio)
    };
    for (fi, f) in p.faces.iter().enumerate() {
        let ids: Vec<usize> = f
            .iter()
            .map(|&v| b.vertex(VKey::I(fi, v), shrunk(fi, v)))
            .collect();
        b.face(ids);
    }
    for &(a, c) in &p.edges() {
        let f1 = def[&(a, c)];
        let f2 = def[&(c, a)];
        let va = b.vertex(VKey::V(a), p.vertices[a]);
        let vc = b.vertex(VKey::V(c), p.vertices[c]);
        let f1a = b.vertex(VKey::I(f1, a), shrunk(f1, a));
        let f1c = b.vertex(VKey::I(f1, c), shrunk(f1, c));
        let f2a = b.vertex(VKey::I(f2, a), shrunk(f2, a));
        let f2c = b.vertex(VKey::I(f2, c), shrunk(f2, c));
        b.face(vec![va, f1a, f1c, vc, f2c, f2a]);
    }
    b.build()
}

/// Conway gyro: pentagonal faces, one per (face, edge) incidence.
#[must_use]
pub fn gyro(p: &Polyhedron) -> Polyhedron {
    let centroids = p.face_centroids();
    let third = |a: usize, c: usize| p.vertices[a] + (p.vertices[c] - p.vertices[a]) * (1.0 / 3.0);
    let mut b = FlagBuilder::new();
    for (fi, f) in p.faces.iter().enumerate() {
        let n = f.len();
        for k in 0..n {
            let (v1, v2, v3) = (f[k], f[(k + 1) % n], f[(k + 2) % n]);
            let c = b.vertex(VKey::C(fi), centroids[fi]);
            let t12 = b.vertex(VKey::E(v1, v2), third(v1, v2));
            let t21 = b.vertex(VKey::E(v2, v1), third(v2, v1));
            let vv2 = b.vertex(VKey::V(v2), p.vertices[v2]);
            let t23 = b.vertex(VKey::E(v2, v3), third(v2, v3));
            b.face(vec![c, t12, t21, vv2, t23]);
        }
    }
    b.build()
}

/// Conway propellor: each face spins off a smaller rotated copy
/// surrounded by quads.
#[must_use]
pub fn propellor(p: &Polyhedron) -> Polyhedron {
    let third = |a: usize, c: usize| p.vertices[a] + (p.vertices[c] - p.vertices[a]) * (1.0 / 3.0);
    let mut b = FlagBuilder::new();
    for f in &p.faces {
        let n = f.len();
        let center: Vec<usize> = (0..n)
            .map(|k| b.vertex(VKey::E(f[k], f[(k + 1) % n]), third(f[k], f[(k + 1) % n])))
            .collect();
        b.face(center);
        for k in 0..n {
            let (v1, v2, v3) = (f[k], f[(k + 1) % n], f[(k + 2) % n]);
            let t12 = b.vertex(VKey::E(v1, v2), third(v1, v2));
            let t21 = b.vertex(VKey::E(v2, v1), third(v2, v1));
            let vv2 = b.vertex(VKey::V(v2), p.vertices[v2]);
            let t23 = b.vertex(VKey::E(v2, v3), third(v2, v3));
            b.face(vec![t12, t21, vv2, t23]);
        }
    }
    b.build()
}

/// Conway whirl: hexagons spiral around shrunken rotated faces.
#[must_use]
pub fn whirl(p: &Polyhedron) -> Polyhedron {
    let centroids = p.face_centroids();
    let third = |a: Vec3, c: Vec3| a + (c - a) * (1.0 / 3.0);
    let mut b = FlagBuilder::new();
    for (fi, f) in p.faces.iter().enumerate() {
        let n = f.len();
        let inner: Vec<usize> = (0..n)
            .map(|k| {
                let (v1, v2) = (f[k], f[(k + 1) % n]);
                let t12 = third(p.vertices[v1], p.vertices[v2]);
                b.vertex(VKey::I(fi, v1), third(centroids[fi], t12))
            })
            .collect();
        b.face(inner.clone());
        for k in 0..n {
            let (v1, v2, v3) = (f[k], f[(k + 1) % n], f[(k + 2) % n]);
            let t12 = b.vertex(VKey::E(v1, v2), third(p.vertices[v1], p.vertices[v2]));
            let t21 = b.vertex(VKey::E(v2, v1), third(p.vertices[v2], p.vertices[v1]));
            let vv2 = b.vertex(VKey::V(v2), p.vertices[v2]);
            let t23 = b.vertex(VKey::E(v2, v3), third(p.vertices[v2], p.vertices[v3]));
            let w2 = inner[(k + 1) % n];
            let w1 = inner[k];
            b.face(vec![t12, t21, vv2, t23, w2, w1]);
        }
    }
    b.build()
}

fn default_kis_height(p: &Polyhedron) -> f64 {
    let edges = p.edges();
    let mean: f64 = edges
        .iter()
        .map(|&(a, b)| p.vertices[a].distance_to(&p.vertices[b]))
        .sum::<f64>()
        / edges.len().max(1) as f64;
    0.25 * mean
}

/// Conway join = dual(ambo): rhombic faces over each original edge.
#[must_use]
pub fn join(p: &Polyhedron) -> Polyhedron {
    ambo(p).dual()
}

/// Conway needle = kis(dual).
#[must_use]
pub fn needle(p: &Polyhedron) -> Polyhedron {
    let d = p.dual();
    let h = default_kis_height(&d);
    kis(&d, h)
}

/// Conway zip = dual(kis).
#[must_use]
pub fn zip(p: &Polyhedron) -> Polyhedron {
    kis(p, default_kis_height(p)).dual()
}

/// Conway ortho = join(join).
#[must_use]
pub fn ortho(p: &Polyhedron) -> Polyhedron {
    join(&join(p))
}

/// Conway expand = ambo(ambo).
#[must_use]
pub fn expand(p: &Polyhedron) -> Polyhedron {
    ambo(&ambo(p))
}

/// Conway bevel = truncate(ambo).
#[must_use]
pub fn bevel(p: &Polyhedron) -> Polyhedron {
    truncate(&ambo(p), 1.0 / 3.0)
}

/// Conway meta = kis(join).
#[must_use]
pub fn meta(p: &Polyhedron) -> Polyhedron {
    let j = join(p);
    let h = default_kis_height(&j);
    kis(&j, h)
}

/// Conway snub = dual(gyro).
#[must_use]
pub fn snub(p: &Polyhedron) -> Polyhedron {
    gyro(p).dual()
}

/// Applies a Conway notation string, e.g. `"tkT"` or `"dsI"`: the
/// rightmost character may be a seed (T, C, O, D, I); otherwise the
/// operators apply to `p`. Operators: d a k t j n z o e b m s g p c w.
///
/// # Errors
/// Returns [`GeomError::InvalidArgument`] for an unknown character.
pub fn conway_apply(p: &Polyhedron, notation: &str) -> Result<Polyhedron, GeomError> {
    let chars: Vec<char> = notation.chars().collect();
    let mut idx = chars.len();
    let mut current = match chars.last() {
        Some('T') => {
            idx -= 1;
            tetrahedron()
        }
        Some('C') => {
            idx -= 1;
            cube()
        }
        Some('O') => {
            idx -= 1;
            octahedron()
        }
        Some('D') => {
            idx -= 1;
            dodecahedron()
        }
        Some('I') => {
            idx -= 1;
            icosahedron()
        }
        _ => p.clone(),
    };
    for k in (0..idx).rev() {
        current = match chars[k] {
            'd' => current.dual(),
            'a' => ambo(&current),
            'k' => kis(&current, default_kis_height(&current)),
            't' => truncate(&current, 1.0 / 3.0),
            'j' => join(&current),
            'n' => needle(&current),
            'z' => zip(&current),
            'o' => ortho(&current),
            'e' => expand(&current),
            'b' => bevel(&current),
            'm' => meta(&current),
            's' => snub(&current),
            'g' => gyro(&current),
            'p' => propellor(&current),
            'c' => chamfer(&current, 0.5),
            'w' => whirl(&current),
            _ => return Err(GeomError::InvalidArgument("unknown Conway operator")),
        };
    }
    Ok(current)
}

// ---------------------------------------------------------------------
// Seeds and named solids.
// ---------------------------------------------------------------------

/// Regular tetrahedron (edge 2√2).
#[must_use]
pub fn tetrahedron() -> Polyhedron {
    Polyhedron {
        vertices: vec![
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ],
        faces: vec![vec![0, 1, 2], vec![0, 3, 1], vec![0, 2, 3], vec![1, 3, 2]],
    }
    .orient_outward()
}

/// Cube (edge 2).
#[must_use]
pub fn cube() -> Polyhedron {
    let vertices = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            )
        })
        .collect();
    Polyhedron {
        vertices,
        faces: vec![
            vec![0, 2, 3, 1],
            vec![4, 5, 7, 6],
            vec![0, 1, 5, 4],
            vec![3, 2, 6, 7],
            vec![0, 4, 6, 2],
            vec![1, 3, 7, 5],
        ],
    }
    .orient_outward()
}

/// Regular octahedron (edge √2).
#[must_use]
pub fn octahedron() -> Polyhedron {
    Polyhedron {
        vertices: vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ],
        faces: vec![
            vec![0, 2, 4],
            vec![2, 1, 4],
            vec![1, 3, 4],
            vec![3, 0, 4],
            vec![2, 0, 5],
            vec![1, 2, 5],
            vec![3, 1, 5],
            vec![0, 3, 5],
        ],
    }
    .orient_outward()
}

/// Regular icosahedron.
#[must_use]
pub fn icosahedron() -> Polyhedron {
    let t = (1.0 + 5.0f64.sqrt()) / 2.0;
    let raw = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ];
    Polyhedron {
        vertices: raw.iter().map(|&(x, y, z)| Vec3::new(x, y, z)).collect(),
        faces: vec![
            vec![0, 11, 5],
            vec![0, 5, 1],
            vec![0, 1, 7],
            vec![0, 7, 10],
            vec![0, 10, 11],
            vec![1, 5, 9],
            vec![5, 11, 4],
            vec![11, 10, 2],
            vec![10, 7, 6],
            vec![7, 1, 8],
            vec![3, 9, 4],
            vec![3, 4, 2],
            vec![3, 2, 6],
            vec![3, 6, 8],
            vec![3, 8, 9],
            vec![4, 9, 5],
            vec![2, 4, 11],
            vec![6, 2, 10],
            vec![8, 6, 7],
            vec![9, 8, 1],
        ],
    }
    .orient_outward()
}

/// Regular dodecahedron (dual of the icosahedron).
#[must_use]
pub fn dodecahedron() -> Polyhedron {
    icosahedron().dual()
}

/// Right prism over a regular n-gon (unit edge circumcircle scaled so
/// the polygon edge is 1), height `h`.
///
/// # Panics
/// Panics unless `n >= 3` and `h > 0`.
#[must_use]
pub fn prism(n: usize, h: f64) -> Polyhedron {
    assert!(n >= 3 && h > 0.0, "prism requires n >= 3, h > 0");
    let r = 0.5 / (std::f64::consts::PI / n as f64).sin();
    let mut vertices = Vec::with_capacity(2 * n);
    for &z in &[-h / 2.0, h / 2.0] {
        for k in 0..n {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            vertices.push(Vec3::new(r * a.cos(), r * a.sin(), z));
        }
    }
    let mut faces = vec![
        (0..n).rev().collect::<Vec<_>>(),
        (n..2 * n).collect::<Vec<_>>(),
    ];
    for k in 0..n {
        faces.push(vec![k, (k + 1) % n, n + (k + 1) % n, n + k]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

/// Antiprism over a regular n-gon (unit polygon edge), height `h`.
///
/// # Panics
/// Panics unless `n >= 3` and `h > 0`.
#[must_use]
pub fn antiprism(n: usize, h: f64) -> Polyhedron {
    assert!(n >= 3 && h > 0.0, "antiprism requires n >= 3, h > 0");
    let r = 0.5 / (std::f64::consts::PI / n as f64).sin();
    let mut vertices = Vec::with_capacity(2 * n);
    for k in 0..n {
        let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        vertices.push(Vec3::new(r * a.cos(), r * a.sin(), -h / 2.0));
    }
    for k in 0..n {
        let a = 2.0 * std::f64::consts::PI * (k as f64 + 0.5) / n as f64;
        vertices.push(Vec3::new(r * a.cos(), r * a.sin(), h / 2.0));
    }
    let mut faces = vec![
        (0..n).rev().collect::<Vec<_>>(),
        (n..2 * n).collect::<Vec<_>>(),
    ];
    for k in 0..n {
        faces.push(vec![k, (k + 1) % n, n + k]);
        faces.push(vec![(k + 1) % n, n + (k + 1) % n, n + k]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

/// Pyramid over a regular n-gon (unit edge base), apex height `h`.
///
/// # Panics
/// Panics unless `n >= 3` and `h > 0`.
#[must_use]
pub fn pyramid(n: usize, h: f64) -> Polyhedron {
    assert!(n >= 3 && h > 0.0, "pyramid requires n >= 3, h > 0");
    let r = 0.5 / (std::f64::consts::PI / n as f64).sin();
    let mut vertices: Vec<Vec3> = (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            Vec3::new(r * a.cos(), r * a.sin(), 0.0)
        })
        .collect();
    vertices.push(Vec3::new(0.0, 0.0, h));
    let mut faces = vec![(0..n).rev().collect::<Vec<_>>()];
    for k in 0..n {
        faces.push(vec![k, (k + 1) % n, n]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

/// Bipyramid over a regular n-gon (unit edge equator), apexes at ±h.
///
/// # Panics
/// Panics unless `n >= 3` and `h > 0`.
#[must_use]
pub fn bipyramid(n: usize, h: f64) -> Polyhedron {
    assert!(n >= 3 && h > 0.0, "bipyramid requires n >= 3, h > 0");
    let r = 0.5 / (std::f64::consts::PI / n as f64).sin();
    let mut vertices: Vec<Vec3> = (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            Vec3::new(r * a.cos(), r * a.sin(), 0.0)
        })
        .collect();
    vertices.push(Vec3::new(0.0, 0.0, h));
    vertices.push(Vec3::new(0.0, 0.0, -h));
    let mut faces = Vec::new();
    for k in 0..n {
        faces.push(vec![k, (k + 1) % n, n]);
        faces.push(vec![(k + 1) % n, k, n + 1]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

/// Builds a polyhedron as the convex hull of a point set, merging
/// coplanar triangles into polygon faces.
///
/// # Panics
/// Panics with fewer than 4 points or degenerate input.
#[must_use]
pub fn from_convex_points(points: &[Vec3]) -> Polyhedron {
    let tris = crate::geometry::hull::convex_hull_3d(points);
    let mut mesh = Mesh::new(points.to_vec(), tris).expect("hull indices valid");
    if mesh.volume() < 0.0 {
        mesh.flip_normals();
    }
    let scale = points.iter().map(|p| p.magnitude()).fold(1e-12, f64::max);
    // Union-find over triangles with matching planes.
    let normals: Vec<Vec3> = mesh.triangles().map(|t| t.normal()).collect();
    let offsets: Vec<f64> = mesh
        .triangles()
        .zip(&normals)
        .map(|(t, n)| n.dot(&t.a))
        .collect();
    let nf = mesh.indices.len();
    let mut parent: Vec<usize> = (0..nf).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let fa = mesh.face_adjacency();
    for (i, row) in fa.iter().enumerate() {
        for j in row.iter().flatten() {
            let same = normals[i].dot(&normals[*j]) > 1.0 - 1e-9
                && (offsets[i] - offsets[*j]).abs() < 1e-7 * scale;
            if same {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, *j));
                if ri != rj {
                    parent[ri.max(rj)] = ri.min(rj);
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..nf {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    // Boundary loop of each group.
    let mut faces = Vec::new();
    for group in groups.values() {
        let mut boundary: HashMap<usize, usize> = HashMap::new();
        let mut counts: HashMap<(usize, usize), i32> = HashMap::new();
        for &t in group {
            let [a, b, c] = mesh.indices[t];
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *counts.entry((u.min(v), u.max(v))).or_insert(0) += 1;
            }
        }
        for &t in group {
            let [a, b, c] = mesh.indices[t];
            for (u, v) in [(a, b), (b, c), (c, a)] {
                if counts[&(u.min(v), u.max(v))] == 1 {
                    boundary.insert(u, v);
                }
            }
        }
        let &start = boundary.keys().next().expect("nonempty boundary");
        let mut loop_ = vec![start];
        let mut cur = boundary[&start];
        while cur != start {
            loop_.push(cur);
            cur = boundary[&cur];
            assert!(loop_.len() <= boundary.len(), "broken face boundary");
        }
        faces.push(loop_);
    }
    let mut p = Polyhedron {
        vertices: mesh.vertices,
        faces,
    };
    // Drop unused vertices (hull interior points).
    let mut used = vec![false; p.vertices.len()];
    for f in &p.faces {
        for &v in f {
            used[v] = true;
        }
    }
    let mut remap = vec![usize::MAX; p.vertices.len()];
    let mut verts = Vec::new();
    for (i, &u) in used.iter().enumerate() {
        if u {
            remap[i] = verts.len();
            verts.push(p.vertices[i]);
        }
    }
    for f in &mut p.faces {
        for v in f.iter_mut() {
            *v = remap[*v];
        }
    }
    p.vertices = verts;
    p.orient_outward()
}

/// The 13 Archimedean solids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchimedeanSolid {
    TruncatedTetrahedron,
    Cuboctahedron,
    TruncatedCube,
    TruncatedOctahedron,
    Rhombicuboctahedron,
    TruncatedCuboctahedron,
    SnubCube,
    Icosidodecahedron,
    TruncatedDodecahedron,
    TruncatedIcosahedron,
    Rhombicosidodecahedron,
    TruncatedIcosidodecahedron,
    SnubDodecahedron,
}

fn all_permutations_signed(base: &[f64; 3]) -> Vec<Vec3> {
    let perms = [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut out = Vec::new();
    for p in perms {
        for s in 0..8 {
            let v = Vec3::new(
                base[p[0]] * if s & 1 == 0 { 1.0 } else { -1.0 },
                base[p[1]] * if s & 2 == 0 { 1.0 } else { -1.0 },
                base[p[2]] * if s & 4 == 0 { 1.0 } else { -1.0 },
            );
            if out.iter().all(|q: &Vec3| q.distance_to(&v) > 1e-9) {
                out.push(v);
            }
        }
    }
    out
}

fn cyclic_permutations_signed(bases: &[[f64; 3]]) -> Vec<Vec3> {
    let mut out = Vec::new();
    for base in bases {
        for p in [[0usize, 1, 2], [1, 2, 0], [2, 0, 1]] {
            for s in 0..8 {
                let v = Vec3::new(
                    base[p[0]] * if s & 1 == 0 { 1.0 } else { -1.0 },
                    base[p[1]] * if s & 2 == 0 { 1.0 } else { -1.0 },
                    base[p[2]] * if s & 4 == 0 { 1.0 } else { -1.0 },
                );
                if out.iter().all(|q: &Vec3| q.distance_to(&v) > 1e-9) {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// Constructs an Archimedean solid: exact coordinates or exact Conway
/// constructions everywhere except the snub dodecahedron, which is
/// built combinatorially by the snub operator and canonicalized (its
/// coordinates are then approximate).
#[must_use]
pub fn archimedean(kind: ArchimedeanSolid) -> Polyhedron {
    use ArchimedeanSolid::*;
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    let s2 = std::f64::consts::SQRT_2;
    match kind {
        TruncatedTetrahedron => truncate(&tetrahedron(), 1.0 / 3.0),
        Cuboctahedron => ambo(&cube()),
        TruncatedCube => truncate(&cube(), 1.0 / (2.0 + s2)),
        TruncatedOctahedron => truncate(&octahedron(), 1.0 / 3.0),
        Rhombicuboctahedron => from_convex_points(&all_permutations_signed(&[1.0, 1.0, 1.0 + s2])),
        TruncatedCuboctahedron => {
            from_convex_points(&all_permutations_signed(&[1.0, 1.0 + s2, 1.0 + 2.0 * s2]))
        }
        SnubCube => {
            // Tribonacci constant t: t^3 = t^2 + t + 1.
            let mut t = 1.8f64;
            for _ in 0..60 {
                t = (t * t + t + 1.0).cbrt();
            }
            // Even permutations with an even number of minus signs and
            // odd permutations with an odd number.
            let base = [1.0, 1.0 / t, t];
            let even_perms = [[0usize, 1, 2], [1, 2, 0], [2, 0, 1]];
            let odd_perms = [[0usize, 2, 1], [1, 0, 2], [2, 1, 0]];
            let mut pts = Vec::new();
            for s in 0..8u32 {
                let signs = [
                    if s & 1 == 0 { 1.0 } else { -1.0 },
                    if s & 2 == 0 { 1.0 } else { -1.0 },
                    if s & 4 == 0 { 1.0 } else { -1.0 },
                ];
                let minus = s.count_ones() % 2;
                let perms: &[[usize; 3]] = if minus == 0 { &even_perms } else { &odd_perms };
                for p in perms {
                    pts.push(Vec3::new(
                        base[p[0]] * signs[0],
                        base[p[1]] * signs[1],
                        base[p[2]] * signs[2],
                    ));
                }
            }
            from_convex_points(&pts)
        }
        Icosidodecahedron => ambo(&dodecahedron()),
        TruncatedDodecahedron => truncate(&dodecahedron(), 1.0 / (2.0 + phi)),
        TruncatedIcosahedron => truncate(&icosahedron(), 1.0 / 3.0),
        Rhombicosidodecahedron => from_convex_points(&cyclic_permutations_signed(&[
            [1.0, 1.0, phi * phi * phi],
            [phi * phi, phi, 2.0 * phi],
            [2.0 + phi, 0.0, phi * phi],
        ])),
        TruncatedIcosidodecahedron => from_convex_points(&cyclic_permutations_signed(&[
            [1.0 / phi, 1.0 / phi, 3.0 + phi],
            [2.0 / phi, phi, 1.0 + 2.0 * phi],
            [1.0 / phi, phi * phi, 3.0 * phi - 1.0],
            [2.0 * phi - 1.0, 2.0, 2.0 + phi],
            [phi, 3.0, 2.0 * phi],
        ])),
        SnubDodecahedron => snub(&dodecahedron()).canonicalize(8000),
    }
}

/// Catalan solid: the dual of the corresponding Archimedean solid
/// (canonicalized so faces are planar and congruent).
#[must_use]
pub fn catalan(kind: ArchimedeanSolid) -> Polyhedron {
    archimedean(kind).canonicalize(100).dual()
}

/// The first 20 Johnson solids J1..J20 with unit edges; `None` for
/// n = 0 or n > 20.
#[must_use]
pub fn johnson(n: u8) -> Option<Polyhedron> {
    // Pyramid apex height for unit lateral edges over a unit n-gon.
    let pyramid_h = |k: usize| -> f64 {
        let r = 0.5 / (std::f64::consts::PI / k as f64).sin();
        (1.0 - r * r).sqrt()
    };
    // Antiprism height for unit edges.
    let antiprism_h = |k: usize| -> f64 {
        let r = 0.5 / (std::f64::consts::PI / k as f64).sin();
        let d = 2.0 * r * (std::f64::consts::PI / (2 * k) as f64).sin();
        (1.0 - d * d).sqrt()
    };
    let p = match n {
        1 => johnson_pyramid(4, pyramid_h(4)),
        2 => johnson_pyramid(5, pyramid_h(5)),
        3 => cupola(3),
        4 => cupola(4),
        5 => cupola(5),
        6 => rotunda(),
        7 => elongated_pyramid(3, pyramid_h(3)),
        8 => elongated_pyramid(4, pyramid_h(4)),
        9 => elongated_pyramid(5, pyramid_h(5)),
        10 => gyroelongated_pyramid(4, pyramid_h(4), antiprism_h(4)),
        11 => gyroelongated_pyramid(5, pyramid_h(5), antiprism_h(5)),
        12 => johnson_bipyramid(3, pyramid_h(3)),
        13 => johnson_bipyramid(5, pyramid_h(5)),
        14 => elongated_bipyramid(3, pyramid_h(3)),
        15 => elongated_bipyramid(4, pyramid_h(4)),
        16 => elongated_bipyramid(5, pyramid_h(5)),
        17 => gyroelongated_bipyramid(4, pyramid_h(4), antiprism_h(4)),
        18 => elongated_cupola(3),
        19 => elongated_cupola(4),
        20 => elongated_cupola(5),
        _ => return None,
    };
    Some(p)
}

fn ring(k: usize, radius: f64, z: f64, angle0: f64) -> Vec<Vec3> {
    (0..k)
        .map(|i| {
            let a = angle0 + 2.0 * std::f64::consts::PI * i as f64 / k as f64;
            Vec3::new(radius * a.cos(), radius * a.sin(), z)
        })
        .collect()
}

fn johnson_pyramid(k: usize, h: f64) -> Polyhedron {
    pyramid(k, h)
}

fn johnson_bipyramid(k: usize, h: f64) -> Polyhedron {
    bipyramid(k, h)
}

fn elongated_pyramid(k: usize, apex_h: f64) -> Polyhedron {
    let r = 0.5 / (std::f64::consts::PI / k as f64).sin();
    let mut vertices = ring(k, r, 0.0, 0.0);
    vertices.extend(ring(k, r, 1.0, 0.0));
    vertices.push(Vec3::new(0.0, 0.0, 1.0 + apex_h));
    let mut faces = vec![(0..k).rev().collect::<Vec<_>>()];
    for i in 0..k {
        faces.push(vec![i, (i + 1) % k, k + (i + 1) % k, k + i]);
        faces.push(vec![k + i, k + (i + 1) % k, 2 * k]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

fn gyroelongated_pyramid(k: usize, apex_h: f64, anti_h: f64) -> Polyhedron {
    let r = 0.5 / (std::f64::consts::PI / k as f64).sin();
    let half = std::f64::consts::PI / k as f64;
    let mut vertices = ring(k, r, 0.0, 0.0);
    vertices.extend(ring(k, r, anti_h, half));
    vertices.push(Vec3::new(0.0, 0.0, anti_h + apex_h));
    let mut faces = vec![(0..k).rev().collect::<Vec<_>>()];
    for i in 0..k {
        faces.push(vec![i, (i + 1) % k, k + i]);
        faces.push(vec![(i + 1) % k, k + (i + 1) % k, k + i]);
        faces.push(vec![k + i, k + (i + 1) % k, 2 * k]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

fn elongated_bipyramid(k: usize, apex_h: f64) -> Polyhedron {
    let r = 0.5 / (std::f64::consts::PI / k as f64).sin();
    let mut vertices = ring(k, r, 0.0, 0.0);
    vertices.extend(ring(k, r, 1.0, 0.0));
    vertices.push(Vec3::new(0.0, 0.0, 1.0 + apex_h));
    vertices.push(Vec3::new(0.0, 0.0, -apex_h));
    let mut faces = Vec::new();
    for i in 0..k {
        faces.push(vec![i, (i + 1) % k, k + (i + 1) % k, k + i]);
        faces.push(vec![k + i, k + (i + 1) % k, 2 * k]);
        faces.push(vec![(i + 1) % k, i, 2 * k + 1]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

fn gyroelongated_bipyramid(k: usize, apex_h: f64, anti_h: f64) -> Polyhedron {
    let r = 0.5 / (std::f64::consts::PI / k as f64).sin();
    let half = std::f64::consts::PI / k as f64;
    let mut vertices = ring(k, r, 0.0, 0.0);
    vertices.extend(ring(k, r, anti_h, half));
    vertices.push(Vec3::new(0.0, 0.0, anti_h + apex_h));
    vertices.push(Vec3::new(0.0, 0.0, -apex_h));
    let mut faces = Vec::new();
    for i in 0..k {
        faces.push(vec![i, (i + 1) % k, k + i]);
        faces.push(vec![(i + 1) % k, k + (i + 1) % k, k + i]);
        faces.push(vec![k + i, k + (i + 1) % k, 2 * k]);
        faces.push(vec![(i + 1) % k, i, 2 * k + 1]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

/// n-gonal cupola (J3, J4, J5 for n = 3, 4, 5): unit-edge top n-gon
/// over a unit-edge 2n-gon.
fn cupola(k: usize) -> Polyhedron {
    let r2 = 0.5 / (std::f64::consts::PI / (2 * k) as f64).sin();
    let r1 = 0.5 / (std::f64::consts::PI / k as f64).sin();
    let step = std::f64::consts::PI / k as f64; // bottom angular step
                                                // Top vertex j sits above the midpoint of bottom edge (2j, 2j+1).
    let d2 = {
        let b = Vec3::new(r2, 0.0, 0.0);
        let t = Vec3::new(r1 * (0.5 * step).cos(), r1 * (0.5 * step).sin(), 0.0);
        (t - b).magnitude_squared()
    };
    let h = (1.0 - d2).max(0.0).sqrt();
    let mut vertices = ring(2 * k, r2, 0.0, 0.0);
    vertices.extend(ring(k, r1, h, 0.5 * step));
    let mut faces = vec![
        (0..2 * k).rev().collect::<Vec<_>>(),
        (2 * k..3 * k).collect::<Vec<_>>(),
    ];
    for j in 0..k {
        // Triangle over bottom edge (2j, 2j+1); quad over (2j+1, 2j+2).
        faces.push(vec![2 * j, (2 * j + 1) % (2 * k), 2 * k + j]);
        faces.push(vec![
            (2 * j + 1) % (2 * k),
            (2 * j + 2) % (2 * k),
            2 * k + (j + 1) % k,
            2 * k + j,
        ]);
    }
    Polyhedron { vertices, faces }.orient_outward()
}

fn elongated_cupola(k: usize) -> Polyhedron {
    let c = cupola(k);
    let r2 = 0.5 / (std::f64::consts::PI / (2 * k) as f64).sin();
    // Stack the cupola on a 2k-prism of height 1.
    let mut vertices = ring(2 * k, r2, -1.0, 0.0);
    let base_count = vertices.len();
    for v in &c.vertices {
        vertices.push(*v);
    }
    let mut faces: Vec<Vec<usize>> = vec![(0..2 * k).rev().collect()];
    for i in 0..2 * k {
        faces.push(vec![
            i,
            (i + 1) % (2 * k),
            base_count + (i + 1) % (2 * k),
            base_count + i,
        ]);
    }
    for f in &c.faces {
        // Skip the cupola's bottom face (the first, the reversed 2k-gon).
        if f.len() == 2 * k {
            continue;
        }
        faces.push(f.iter().map(|&v| v + base_count).collect());
    }
    Polyhedron { vertices, faces }.orient_outward()
}

/// Pentagonal rotunda (J6): half an icosidodecahedron closed by its
/// equatorial decagon.
fn rotunda() -> Polyhedron {
    let icosi = ambo(&dodecahedron());
    // Unit edges: scale so the shortest edge is 1.
    let e = icosi.edges();
    let len = icosi.vertices[e[0].0].distance_to(&icosi.vertices[e[0].1]);
    let icosi = Polyhedron {
        vertices: icosi.vertices.iter().map(|v| *v * (1.0 / len)).collect(),
        faces: icosi.faces,
    };
    // A 5-fold axis: any pentagon face normal direction.
    let pent = icosi
        .faces
        .iter()
        .position(|f| f.len() == 5)
        .expect("icosidodecahedron has pentagons");
    let axis = icosi.face_centroids()[pent].normalized();
    // Keep faces entirely in the closed upper half-space.
    let height = |v: usize| icosi.vertices[v].dot(&axis);
    let mut faces: Vec<Vec<usize>> = icosi
        .faces
        .iter()
        .filter(|f| f.iter().all(|&v| height(v) > -1e-9))
        .cloned()
        .collect();
    // Close with the equatorial decagon: boundary edges of the kept
    // faces.
    let mut counts: HashMap<(usize, usize), i32> = HashMap::new();
    let mut succ: HashMap<usize, usize> = HashMap::new();
    for f in &faces {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            *counts.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    for f in &faces {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            if counts[&(a.min(b), a.max(b))] == 1 {
                // Reverse orientation for the closing face.
                succ.insert(b, a);
            }
        }
    }
    let &start = succ.keys().next().expect("boundary exists");
    let mut loop_ = vec![start];
    let mut cur = succ[&start];
    while cur != start {
        loop_.push(cur);
        cur = succ[&cur];
        assert!(loop_.len() <= succ.len(), "broken rotunda boundary");
    }
    faces.push(loop_);
    // Compact unused vertices.
    let mut used = vec![false; icosi.vertices.len()];
    for f in &faces {
        for &v in f {
            used[v] = true;
        }
    }
    let mut remap = vec![usize::MAX; icosi.vertices.len()];
    let mut verts = Vec::new();
    for (i, &u) in used.iter().enumerate() {
        if u {
            remap[i] = verts.len();
            verts.push(icosi.vertices[i]);
        }
    }
    for f in &mut faces {
        for v in f.iter_mut() {
            *v = remap[*v];
        }
    }
    Polyhedron {
        vertices: verts,
        faces,
    }
    .orient_outward()
}

/// Class I geodesic sphere: each icosahedron face subdivided into
/// `frequency`² triangles, projected to the unit sphere.
///
/// # Panics
/// Panics unless `frequency >= 1`.
#[must_use]
pub fn geodesic_sphere(frequency: u32) -> Polyhedron {
    assert!(frequency >= 1, "frequency must be >= 1");
    let f = frequency as usize;
    let ico = icosahedron();
    let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut faces = Vec::new();
    let mut vid = |p: Vec3, vertices: &mut Vec<Vec3>| -> usize {
        let q = p.normalized();
        let key = (
            (q.x * 1e8).round() as i64,
            (q.y * 1e8).round() as i64,
            (q.z * 1e8).round() as i64,
        );
        *map.entry(key).or_insert_with(|| {
            vertices.push(q);
            vertices.len() - 1
        })
    };
    for face in &ico.faces {
        let (a, b, c) = (
            ico.vertices[face[0]],
            ico.vertices[face[1]],
            ico.vertices[face[2]],
        );
        let at = |i: usize, j: usize| -> Vec3 {
            let (i, j) = (i as f64, j as f64);
            let k = f as f64;
            a * ((k - i - j) / k) + b * (i / k) + c * (j / k)
        };
        for j in 0..f {
            for i in 0..f - j {
                let p0 = vid(at(i, j), &mut vertices);
                let p1 = vid(at(i + 1, j), &mut vertices);
                let p2 = vid(at(i, j + 1), &mut vertices);
                faces.push(vec![p0, p1, p2]);
                if i + j < f - 1 {
                    let p3 = vid(at(i + 1, j + 1), &mut vertices);
                    faces.push(vec![p1, p3, p2]);
                }
            }
        }
    }
    Polyhedron { vertices, faces }.orient_outward()
}

/// Goldberg polyhedron GP(m, n): hexagons plus 12 pentagons. Class I
/// (n = 0) and class II (m = n) are supported (class II via a √3
/// refinement of the class I triangulation); general class III is
/// not.
///
/// # Panics
/// Panics unless `m >= 1` and (`n == 0` or `n == m`).
#[must_use]
pub fn goldberg(m: u32, n: u32) -> Polyhedron {
    assert!(m >= 1, "m must be >= 1");
    assert!(
        n == 0 || n == m,
        "only class I (n = 0) and class II (m = n) supported"
    );
    let geo = geodesic_sphere(m);
    let tri = if n == m {
        // sqrt(3) refinement triples the triangle count (class II).
        let mesh = crate::mesh::subdivide::sqrt3_subdivide(&geo.to_mesh());
        let vertices: Vec<Vec3> = mesh.vertices.iter().map(|v| v.normalized()).collect();
        Polyhedron {
            vertices,
            faces: mesh.indices.iter().map(|f| f.to_vec()).collect(),
        }
    } else {
        geo
    };
    tri.dual()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_lengths(p: &Polyhedron) -> Vec<f64> {
        p.edges()
            .iter()
            .map(|&(a, b)| p.vertices[a].distance_to(&p.vertices[b]))
            .collect()
    }

    fn faces_regular(p: &Polyhedron, tol: f64) -> bool {
        p.faces.iter().all(|f| {
            let n = f.len();
            let pts: Vec<Vec3> = f.iter().map(|&v| p.vertices[v]).collect();
            let sides: Vec<f64> = (0..n)
                .map(|k| pts[k].distance_to(&pts[(k + 1) % n]))
                .collect();
            let angles: Vec<f64> = (0..n)
                .map(|k| {
                    let a = pts[(k + n - 1) % n] - pts[k];
                    let b = pts[(k + 1) % n] - pts[k];
                    a.angle_between(&b)
                })
                .collect();
            let s0 = sides[0];
            let a0 = angles[0];
            sides.iter().all(|s| (s - s0).abs() < tol)
                && angles.iter().all(|a| (a - a0).abs() < tol)
        })
    }

    #[test]
    fn test_platonic_solids_regular() {
        let solids: [(Polyhedron, usize, usize, usize); 5] = [
            (tetrahedron(), 4, 6, 4),
            (cube(), 8, 12, 6),
            (octahedron(), 6, 12, 8),
            (dodecahedron(), 20, 30, 12),
            (icosahedron(), 12, 30, 20),
        ];
        for (p, v, e, f) in solids {
            assert_eq!(p.vertices.len(), v);
            assert_eq!(p.edges().len(), e);
            assert_eq!(p.faces.len(), f);
            assert_eq!(p.euler(), 2);
            assert!(p.is_convex());
            assert!(p.volume() > 0.0);
            let lens = edge_lengths(&p);
            let l0 = lens[0];
            assert!(lens.iter().all(|l| (l - l0).abs() < 1e-9), "equal edges");
            assert!(faces_regular(&p, 1e-9), "regular faces");
        }
    }

    #[test]
    fn test_dual_of_dual_combinatorics() {
        for p in [
            cube(),
            dodecahedron(),
            archimedean(ArchimedeanSolid::Cuboctahedron),
        ] {
            let dd = p.dual().dual();
            assert_eq!(dd.vertices.len(), p.vertices.len());
            assert_eq!(dd.faces.len(), p.faces.len());
            assert_eq!(dd.edges().len(), p.edges().len());
            let mut sizes: Vec<usize> = p.faces.iter().map(Vec::len).collect();
            let mut dsizes: Vec<usize> = dd.faces.iter().map(Vec::len).collect();
            sizes.sort_unstable();
            dsizes.sort_unstable();
            assert_eq!(sizes, dsizes);
        }
    }

    #[test]
    fn test_truncated_cube_counts() {
        let tc = truncate(&cube(), 1.0 / 3.0);
        assert_eq!(tc.faces.len(), 14);
        assert_eq!(tc.vertices.len(), 24);
        assert_eq!(tc.edges().len(), 36);
        let mut sizes: Vec<usize> = tc.faces.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(&sizes[..8], &[3; 8]);
        assert_eq!(&sizes[8..], &[8; 6]);
        // The Archimedean truncated cube (exact ratio) has equal edges.
        let atc = archimedean(ArchimedeanSolid::TruncatedCube);
        let lens = edge_lengths(&atc);
        let l0 = lens[0];
        assert!(lens.iter().all(|l| (l - l0).abs() < 1e-9));
    }

    #[test]
    fn test_conway_ac_is_cuboctahedron() {
        let ac = conway_apply(&cube(), "aC").unwrap();
        let co = archimedean(ArchimedeanSolid::Cuboctahedron);
        assert_eq!(ac.vertices.len(), co.vertices.len());
        assert_eq!(ac.faces.len(), co.faces.len());
        let mut a: Vec<usize> = ac.faces.iter().map(Vec::len).collect();
        let mut b: Vec<usize> = co.faces.iter().map(Vec::len).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        // Same shape up to scale: sorted radii proportional.
        let mut ra: Vec<f64> = ac.vertices.iter().map(|v| v.magnitude()).collect();
        let mut rb: Vec<f64> = co.vertices.iter().map(|v| v.magnitude()).collect();
        ra.sort_by(f64::total_cmp);
        rb.sort_by(f64::total_cmp);
        let k = ra[0] / rb[0];
        for (x, y) in ra.iter().zip(&rb) {
            assert!((x / y - k).abs() < 1e-9);
        }
        assert!(conway_apply(&cube(), "qX").is_err());
    }

    #[test]
    fn test_conway_operator_counts() {
        let c = cube();
        // Known V/E/F for operators applied to the cube.
        let cases: Vec<(Polyhedron, usize, usize, usize, &str)> = vec![
            (ambo(&c), 12, 24, 14, "aC"),
            (kis(&c, 0.3), 14, 36, 24, "kC"),
            (join(&c), 14, 24, 12, "jC"),
            (truncate(&c, 0.3), 24, 36, 14, "tC"),
            (expand(&c), 24, 48, 26, "eC"),
            (gyro(&c), 38, 60, 24, "gC"),
            (snub(&c), 24, 60, 38, "sC"),
            (chamfer(&c, 0.5), 32, 48, 18, "cC"),
            (propellor(&c), 32, 60, 30, "pC"),
            (whirl(&c), 56, 84, 30, "wC"),
            (bevel(&c), 48, 72, 26, "bC"),
            (meta(&c), 26, 72, 48, "mC"),
            (ortho(&c), 26, 48, 24, "oC"),
        ];
        for (p, v, e, f, name) in cases {
            assert_eq!(p.vertices.len(), v, "{name} vertices");
            assert_eq!(p.edges().len(), e, "{name} edges");
            assert_eq!(p.faces.len(), f, "{name} faces");
            assert_eq!(p.euler(), 2, "{name} Euler");
            assert!(p.volume() > 0.0, "{name} orientation");
        }
    }

    #[test]
    fn test_archimedean_all_thirteen() {
        use ArchimedeanSolid::*;
        let expected: Vec<(ArchimedeanSolid, usize, usize, usize)> = vec![
            (TruncatedTetrahedron, 12, 18, 8),
            (Cuboctahedron, 12, 24, 14),
            (TruncatedCube, 24, 36, 14),
            (TruncatedOctahedron, 24, 36, 14),
            (Rhombicuboctahedron, 24, 48, 26),
            (TruncatedCuboctahedron, 48, 72, 26),
            (SnubCube, 24, 60, 38),
            (Icosidodecahedron, 30, 60, 32),
            (TruncatedDodecahedron, 60, 90, 32),
            (TruncatedIcosahedron, 60, 90, 32),
            (Rhombicosidodecahedron, 60, 120, 62),
            (TruncatedIcosidodecahedron, 120, 180, 62),
            (SnubDodecahedron, 60, 150, 92),
        ];
        for (kind, v, e, f) in expected {
            let p = archimedean(kind);
            assert_eq!(p.vertices.len(), v, "{kind:?} V");
            assert_eq!(p.edges().len(), e, "{kind:?} E");
            assert_eq!(p.faces.len(), f, "{kind:?} F");
            assert_eq!(p.euler(), 2);
            assert!(p.is_convex(), "{kind:?} convex");
            // Uniformity: equal edges (exact constructions tight, the
            // canonicalized snub dodecahedron looser).
            let lens = edge_lengths(&p);
            let l0 = lens[0];
            let tol = if kind == SnubDodecahedron { 0.02 } else { 1e-6 };
            for l in &lens {
                assert!(
                    (l - l0).abs() < tol * l0.max(1.0),
                    "{kind:?} edge {l} vs {l0}"
                );
            }
        }
    }

    #[test]
    fn test_catalan_duals() {
        // Rhombic dodecahedron: dual of the cuboctahedron.
        let rd = catalan(ArchimedeanSolid::Cuboctahedron);
        assert_eq!(rd.faces.len(), 12);
        assert_eq!(rd.vertices.len(), 14);
        assert!(rd.faces.iter().all(|f| f.len() == 4));
        // Face transitivity proxy: all faces have the same area.
        let mesh_areas: Vec<f64> = rd
            .faces
            .iter()
            .map(|f| {
                let pts: Vec<Vec3> = f.iter().map(|&v| rd.vertices[v]).collect();
                newell_normal(&pts).magnitude() / 2.0
            })
            .collect();
        let a0 = mesh_areas[0];
        for a in &mesh_areas {
            assert!((a - a0).abs() < 1e-6 * a0);
        }
    }

    #[test]
    fn test_prisms_and_pyramids() {
        let p = prism(6, 1.0);
        assert_eq!(p.euler(), 2);
        assert_eq!(p.faces.len(), 8);
        let lens = edge_lengths(&p);
        assert!(
            lens.iter().all(|l| (l - 1.0).abs() < 1e-9),
            "uniform hexagonal prism"
        );
        let a = antiprism(4, 1.0);
        assert_eq!(a.euler(), 2);
        assert_eq!(a.faces.len(), 10);
        let py = pyramid(4, 0.8);
        assert_eq!(py.euler(), 2);
        let bp = bipyramid(3, 0.9);
        assert_eq!(bp.euler(), 2);
        assert_eq!(bp.faces.len(), 6);
    }

    #[test]
    fn test_johnson_solids_unit_edges() {
        for n in 1..=20u8 {
            let p = johnson(n).unwrap();
            assert_eq!(p.euler(), 2, "J{n} Euler");
            assert!(p.volume() > 0.0, "J{n} orientation");
            assert!(p.is_convex(), "J{n} convex");
            let lens = edge_lengths(&p);
            for l in &lens {
                assert!((l - 1.0).abs() < 1e-9, "J{n} edge length {l}");
            }
            assert!(faces_regular(&p, 1e-7), "J{n} regular faces");
        }
        assert!(johnson(0).is_none());
        assert!(johnson(21).is_none());
        // Spot checks.
        assert_eq!(johnson(1).unwrap().faces.len(), 5);
        assert_eq!(johnson(6).unwrap().faces.len(), 17);
        assert_eq!(johnson(17).unwrap().faces.len(), 16);
    }

    #[test]
    fn test_geodesic_and_goldberg() {
        for f in [1u32, 2, 3, 4] {
            let g = geodesic_sphere(f);
            let t = (f * f) as usize;
            assert_eq!(g.faces.len(), 20 * t);
            assert_eq!(g.euler(), 2);
            for v in &g.vertices {
                assert!((v.magnitude() - 1.0).abs() < 1e-9);
            }
            let gb = goldberg(f, 0);
            assert_eq!(gb.faces.len(), 10 * t + 2);
            let pentagons = gb.faces.iter().filter(|f| f.len() == 5).count();
            assert_eq!(pentagons, 12, "always exactly 12 pentagons");
            assert!(gb.faces.iter().all(|f| f.len() == 5 || f.len() == 6));
        }
        // Class II: GP(1,1) is the truncated icosahedron pattern.
        let g11 = goldberg(1, 1);
        assert_eq!(g11.faces.len(), 32);
        assert_eq!(g11.faces.iter().filter(|f| f.len() == 5).count(), 12);
        assert_eq!(g11.faces.iter().filter(|f| f.len() == 6).count(), 20);
    }

    #[test]
    fn test_canonicalize_improves_uniformity() {
        // aa on a dodecahedron has golden-ratio edge spread;
        // canonicalization evens it out.
        let p = expand(&dodecahedron());
        let spread = |p: &Polyhedron| {
            let l = edge_lengths(p);
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for x in l {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            hi / lo
        };
        let before = spread(&p);
        let after = spread(&p.canonicalize(300));
        assert!(after < before, "canonicalization tightens edge spread");
        assert!(after < 1.1, "close to uniform (spread {after})");
    }

    #[test]
    fn test_symmetry_groups() {
        assert_eq!(cube().symmetry_group(), Some(PointGroup3::Octahedral));
        assert_eq!(octahedron().symmetry_group(), Some(PointGroup3::Octahedral));
        assert_eq!(
            tetrahedron().symmetry_group(),
            Some(PointGroup3::Tetrahedral)
        );
        assert_eq!(
            icosahedron().symmetry_group(),
            Some(PointGroup3::Icosahedral)
        );
        assert_eq!(
            dodecahedron().symmetry_group(),
            Some(PointGroup3::Icosahedral)
        );
        match prism(5, 1.3).symmetry_group() {
            Some(PointGroup3::Dn(5)) => {}
            other => panic!("pentagonal prism should be D5, got {other:?}"),
        }
        match johnson(2).unwrap().symmetry_group() {
            Some(PointGroup3::Cn(5)) => {}
            other => panic!("pentagonal pyramid should be C5, got {other:?}"),
        }
    }

    #[test]
    fn test_vertex_figure_and_mesh() {
        let ico = icosahedron();
        for v in 0..12 {
            assert_eq!(ico.vertex_figure(v).len(), 5, "icosahedron valence 5");
        }
        let m = dodecahedron().to_mesh();
        assert_eq!(m.indices.len(), 12 * 3);
        assert!(m.volume() > 0.0);
    }
}
