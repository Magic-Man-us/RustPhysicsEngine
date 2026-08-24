//! Four-dimensional polytopes: the six regular 4-polytopes with their full
//! combinatorics, prisms and products, projections and cross-sections,
//! duals, Coxeter-plane pictures, exceptional root systems and lattices,
//! and curse-of-dimensionality demonstrations.

use crate::geometry::mesh::Mesh;
use crate::linalg::{eigen_symmetric, Matrix};
use crate::manifold::lie::So4;
use crate::manifold::vecn::VecN;
use crate::math::{Vec2, Vec3};

const PHI: f64 = 1.618_033_988_749_895;
const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Vec4
// ---------------------------------------------------------------------------

/// A 4D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Vec4 { x, y, z, w }
    }

    #[must_use]
    pub fn dot(&self, o: &Vec4) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }

    #[must_use]
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    #[must_use]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            *self
        } else {
            self.scale(1.0 / n)
        }
    }

    #[must_use]
    pub fn add(&self, o: &Vec4) -> Self {
        Vec4::new(self.x + o.x, self.y + o.y, self.z + o.z, self.w + o.w)
    }

    #[must_use]
    pub fn sub(&self, o: &Vec4) -> Self {
        Vec4::new(self.x - o.x, self.y - o.y, self.z - o.z, self.w - o.w)
    }

    #[must_use]
    pub fn scale(&self, k: f64) -> Self {
        Vec4::new(self.x * k, self.y * k, self.z * k, self.w * k)
    }

    fn get(&self, i: usize) -> f64 {
        match i {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => self.w,
        }
    }

    fn set(&mut self, i: usize, v: f64) {
        match i {
            0 => self.x = v,
            1 => self.y = v,
            2 => self.z = v,
            _ => self.w = v,
        }
    }
}

// ---------------------------------------------------------------------------
// Polytope4
// ---------------------------------------------------------------------------

/// A 4-polytope: vertices with edge, face (2D), and cell (3D facet)
/// combinatorics. Faces and cells list vertex indices.
#[derive(Debug, Clone)]
pub struct Polytope4 {
    pub vertices: Vec<Vec4>,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<Vec<usize>>,
    pub cells: Vec<Vec<usize>>,
}

fn edges_by_min_distance(vertices: &[Vec4]) -> Vec<(usize, usize)> {
    let n = vertices.len();
    let mut min_d = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = vertices[i].sub(&vertices[j]).norm();
            if d > 1e-9 {
                min_d = min_d.min(d);
            }
        }
    }
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = vertices[i].sub(&vertices[j]).norm();
            if (d - min_d).abs() < 1e-6 * min_d {
                edges.push((i, j));
            }
        }
    }
    edges
}

/// Facet (cell) vertex sets found as the argmax sets of the given outward
/// directions.
fn cells_from_directions(vertices: &[Vec4], dirs: &[Vec4]) -> Vec<Vec<usize>> {
    dirs.iter()
        .map(|d| {
            let dn = d.normalized();
            let best = vertices
                .iter()
                .map(|v| v.dot(&dn))
                .fold(f64::MIN, f64::max);
            (0..vertices.len())
                .filter(|&i| vertices[i].dot(&dn) > best - 1e-6 * best.abs().max(1.0))
                .collect()
        })
        .collect()
}

/// Faces as intersections of adjacent cells (3 or more shared vertices),
/// ordered around the face centroid.
fn faces_from_cells(vertices: &[Vec4], cells: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut faces: Vec<Vec<usize>> = Vec::new();
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    for i in 0..cells.len() {
        let set_i: std::collections::HashSet<usize> = cells[i].iter().copied().collect();
        for cell_j in cells.iter().skip(i + 1) {
            let mut shared: Vec<usize> = cell_j
                .iter()
                .copied()
                .filter(|v| set_i.contains(v))
                .collect();
            if shared.len() < 3 {
                continue;
            }
            shared.sort_unstable();
            if seen.contains(&shared) {
                continue;
            }
            seen.insert(shared.clone());
            faces.push(order_face(vertices, &shared));
        }
    }
    faces
}

/// Order face vertices around their centroid within the face plane.
fn order_face(vertices: &[Vec4], face: &[usize]) -> Vec<usize> {
    let n = face.len() as f64;
    let mut centroid = Vec4::new(0.0, 0.0, 0.0, 0.0);
    for &v in face {
        centroid = centroid.add(&vertices[v]);
    }
    centroid = centroid.scale(1.0 / n);
    // basis in the face plane
    let a = vertices[face[0]].sub(&centroid);
    let e1 = a.normalized();
    let mut e2 = Vec4::new(0.0, 0.0, 0.0, 0.0);
    for &v in &face[1..] {
        let b = vertices[v].sub(&centroid);
        let perp = b.sub(&e1.scale(b.dot(&e1)));
        if perp.norm() > 1e-9 {
            e2 = perp.normalized();
            break;
        }
    }
    let mut with_angle: Vec<(f64, usize)> = face
        .iter()
        .map(|&v| {
            let d = vertices[v].sub(&centroid);
            (d.dot(&e2).atan2(d.dot(&e1)), v)
        })
        .collect();
    with_angle.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    with_angle.into_iter().map(|(_, v)| v).collect()
}

/// The 120 unit icosians (vertices of the 600-cell).
fn icosians() -> Vec<Vec4> {
    let mut v = Vec::new();
    // 8 unit axis vectors
    for i in 0..4 {
        for &s in &[1.0, -1.0] {
            let mut p = Vec4::new(0.0, 0.0, 0.0, 0.0);
            p.set(i, s);
            v.push(p);
        }
    }
    // 16 half-integer points
    for &s0 in &[0.5, -0.5] {
        for &s1 in &[0.5, -0.5] {
            for &s2 in &[0.5, -0.5] {
                for &s3 in &[0.5, -0.5] {
                    v.push(Vec4::new(s0, s1, s2, s3));
                }
            }
        }
    }
    // 96 even permutations of (±phi, ±1, ±1/phi, 0)/2
    let a = PHI / 2.0;
    let b = 0.5;
    let c = 0.5 / PHI;
    // even permutations of indices (0, 1, 2, 3) applied to (a, b, c, 0)
    let even_perms = [
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 2, 1, 0],
    ];
    for perm in &even_perms {
        for &sa in &[1.0_f64, -1.0] {
            for &sb in &[1.0_f64, -1.0] {
                for &sc in &[1.0_f64, -1.0] {
                    let vals = [sa * a, sb * b, sc * c, 0.0];
                    let mut p = Vec4::new(0.0, 0.0, 0.0, 0.0);
                    for (slot, &src) in perm.iter().enumerate() {
                        p.set(slot, vals[src]);
                    }
                    v.push(p);
                }
            }
        }
    }
    v
}

impl Polytope4 {
    fn from_vertices_and_dirs(vertices: Vec<Vec4>, dirs: &[Vec4]) -> Self {
        let edges = edges_by_min_distance(&vertices);
        let cells = cells_from_directions(&vertices, dirs);
        let faces = faces_from_cells(&vertices, &cells);
        Polytope4 {
            vertices,
            edges,
            faces,
            cells,
        }
    }

    /// The 8-cell (tesseract).
    #[must_use]
    pub fn tesseract() -> Self {
        let mut vertices = Vec::new();
        for &x in &[0.5, -0.5] {
            for &y in &[0.5, -0.5] {
                for &z in &[0.5, -0.5] {
                    for &w in &[0.5, -0.5] {
                        vertices.push(Vec4::new(x, y, z, w));
                    }
                }
            }
        }
        // cells in the +-axis directions
        let mut dirs = Vec::new();
        for i in 0..4 {
            for &s in &[1.0, -1.0] {
                let mut d = Vec4::new(0.0, 0.0, 0.0, 0.0);
                d.set(i, s);
                dirs.push(d);
            }
        }
        Self::from_vertices_and_dirs(vertices, &dirs)
    }

    /// The 16-cell (4D cross-polytope).
    #[must_use]
    pub fn cell16() -> Self {
        let mut vertices = Vec::new();
        for i in 0..4 {
            for &s in &[1.0, -1.0] {
                let mut p = Vec4::new(0.0, 0.0, 0.0, 0.0);
                p.set(i, s);
                vertices.push(p);
            }
        }
        // cells toward tesseract vertices
        let mut dirs = Vec::new();
        for &x in &[1.0, -1.0] {
            for &y in &[1.0, -1.0] {
                for &z in &[1.0, -1.0] {
                    for &w in &[1.0, -1.0] {
                        dirs.push(Vec4::new(x, y, z, w));
                    }
                }
            }
        }
        Self::from_vertices_and_dirs(vertices, &dirs)
    }

    /// The self-dual 24-cell.
    #[must_use]
    pub fn cell24() -> Self {
        let mut vertices = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                for &si in &[1.0, -1.0] {
                    for &sj in &[1.0, -1.0] {
                        let mut p = Vec4::new(0.0, 0.0, 0.0, 0.0);
                        p.set(i, si);
                        p.set(j, sj);
                        vertices.push(p);
                    }
                }
            }
        }
        // dual 24-cell directions: 8 axes + 16 half-integer points
        let mut dirs = Vec::new();
        for i in 0..4 {
            for &s in &[1.0, -1.0] {
                let mut d = Vec4::new(0.0, 0.0, 0.0, 0.0);
                d.set(i, s);
                dirs.push(d);
            }
        }
        for &x in &[0.5, -0.5] {
            for &y in &[0.5, -0.5] {
                for &z in &[0.5, -0.5] {
                    for &w in &[0.5, -0.5] {
                        dirs.push(Vec4::new(x, y, z, w));
                    }
                }
            }
        }
        Self::from_vertices_and_dirs(vertices, &dirs)
    }

    /// The regular 5-cell (4-simplex).
    #[must_use]
    pub fn simplex5() -> Self {
        // 5 vertices of a regular simplex centered at the origin
        let mut vertices = vec![
            Vec4::new(1.0, 1.0, 1.0, -1.0 / 5.0_f64.sqrt()),
            Vec4::new(1.0, -1.0, -1.0, -1.0 / 5.0_f64.sqrt()),
            Vec4::new(-1.0, 1.0, -1.0, -1.0 / 5.0_f64.sqrt()),
            Vec4::new(-1.0, -1.0, 1.0, -1.0 / 5.0_f64.sqrt()),
            Vec4::new(0.0, 0.0, 0.0, 5.0_f64.sqrt() - 1.0 / 5.0_f64.sqrt()),
        ];
        // center
        let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for v in &vertices {
            c = c.add(v);
        }
        c = c.scale(1.0 / 5.0);
        for v in vertices.iter_mut() {
            *v = v.sub(&c);
        }
        // cells opposite each vertex
        let dirs: Vec<Vec4> = vertices.iter().map(|v| v.scale(-1.0)).collect();
        Self::from_vertices_and_dirs(vertices, &dirs)
    }

    /// The 600-cell (vertices are the 120 unit icosians; cells are the
    /// tetrahedral 4-cliques of the edge graph).
    #[must_use]
    pub fn cell600() -> Self {
        let vertices = icosians();
        let edges = edges_by_min_distance(&vertices);
        let n = vertices.len();
        let mut adj = vec![vec![false; n]; n];
        let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &edges {
            adj[a][b] = true;
            adj[b][a] = true;
            nbrs[a].push(b);
            nbrs[b].push(a);
        }
        // tetrahedral cells: 4-cliques
        let mut cells = Vec::new();
        for (v, nb) in nbrs.iter().enumerate() {
            for (ai, &a) in nb.iter().enumerate() {
                if a < v {
                    continue;
                }
                for (bi, &b) in nb.iter().enumerate().skip(ai + 1) {
                    if b < v || !adj[a][b] {
                        continue;
                    }
                    for &c in nb.iter().skip(bi + 1) {
                        if c < v || !adj[a][c] || !adj[b][c] {
                            continue;
                        }
                        cells.push(vec![v, a, b, c]);
                    }
                }
            }
        }
        // triangular faces: 3-cliques
        let mut faces = Vec::new();
        for (v, nb) in nbrs.iter().enumerate() {
            for (ai, &a) in nb.iter().enumerate() {
                if a < v {
                    continue;
                }
                for &b in nb.iter().skip(ai + 1) {
                    if b < v || !adj[a][b] {
                        continue;
                    }
                    faces.push(vec![v, a, b]);
                }
            }
        }
        Polytope4 {
            vertices,
            edges,
            faces,
            cells,
        }
    }

    /// The 120-cell: vertices are the 600 cell centers of the 600-cell,
    /// with dodecahedral cells found from the 600-cell vertex directions.
    #[must_use]
    pub fn cell120() -> Self {
        let c600 = Polytope4::cell600();
        let vertices: Vec<Vec4> = c600
            .cells
            .iter()
            .map(|cell| {
                let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
                for &v in cell {
                    c = c.add(&c600.vertices[v]);
                }
                c.normalized()
            })
            .collect();
        let dirs = c600.vertices.clone();
        Self::from_vertices_and_dirs(vertices, &dirs)
    }

    /// p,q-duoprism: the product of a p-gon and a q-gon.
    #[must_use]
    pub fn duoprism(p: usize, q: usize) -> Self {
        let mut vertices = Vec::new();
        for i in 0..p {
            let a = 2.0 * PI * i as f64 / p as f64;
            for j in 0..q {
                let b = 2.0 * PI * j as f64 / q as f64;
                vertices.push(Vec4::new(a.cos(), a.sin(), b.cos(), b.sin()));
            }
        }
        let idx = |i: usize, j: usize| i * q + j;
        let mut edges = Vec::new();
        let mut faces = Vec::new();
        let mut cells = Vec::new();
        for i in 0..p {
            for j in 0..q {
                edges.push((idx(i, j), idx((i + 1) % p, j)));
                edges.push((idx(i, j), idx(i, (j + 1) % q)));
                // square faces
                faces.push(vec![
                    idx(i, j),
                    idx((i + 1) % p, j),
                    idx((i + 1) % p, (j + 1) % q),
                    idx(i, (j + 1) % q),
                ]);
            }
        }
        // p-gon faces (fixed j) and q-gon faces (fixed i)
        for j in 0..q {
            faces.push((0..p).map(|i| idx(i, j)).collect());
        }
        for i in 0..p {
            faces.push((0..q).map(|j| idx(i, j)).collect());
        }
        // cells: p prisms over the q-gon and q prisms over the p-gon
        for i in 0..p {
            let mut cell: Vec<usize> = (0..q).map(|j| idx(i, j)).collect();
            cell.extend((0..q).map(|j| idx((i + 1) % p, j)));
            cells.push(cell);
        }
        for j in 0..q {
            let mut cell: Vec<usize> = (0..p).map(|i| idx(i, j)).collect();
            cell.extend((0..p).map(|i| idx(i, (j + 1) % q)));
            cells.push(cell);
        }
        Polytope4 {
            vertices,
            edges,
            faces,
            cells,
        }
    }

    /// Discretized duocylinder (n x n grid on the Clifford-torus ridge);
    /// vertices and edges only.
    #[must_use]
    pub fn duocylinder(n: usize) -> Self {
        let p = Polytope4::duoprism(n, n);
        Polytope4 {
            vertices: p.vertices,
            edges: p.edges,
            faces: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Grand antiprism: the 600-cell with two orthogonal rings of ten
    /// vertices removed. Vertices and edges only.
    #[must_use]
    pub fn grand_antiprism() -> Self {
        let all = icosians();
        // ring 1: a great decagon built by the golden recurrence
        // v_{k+1} = phi v_k - v_{k-1} from a vertex and one of its neighbors
        let v0 = all[0];
        let v1 = *all
            .iter()
            .find(|v| (v.dot(&v0) - PHI / 2.0).abs() < 1e-9)
            .expect("neighbor at pi/5");
        let mut ring: Vec<Vec4> = vec![v0, v1];
        for k in 2..10 {
            let next = ring[k - 1].scale(PHI).sub(&ring[k - 2]);
            ring.push(next);
        }
        // ring 2: the completely orthogonal decagon
        let in_ring1 = |v: &Vec4| ring.iter().any(|r| r.sub(v).norm() < 1e-6);
        let orthogonal =
            |v: &Vec4| v.dot(&v0).abs() < 1e-9 && v.dot(&v1).abs() < 1e-9;
        let keep: Vec<Vec4> = all
            .into_iter()
            .filter(|v| !in_ring1(v) && !orthogonal(v))
            .collect();
        let edges = edges_by_min_distance(&keep);
        Polytope4 {
            vertices: keep,
            edges,
            faces: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Discretized cubinder (cylinder x square); vertices/edges only.
    #[must_use]
    pub fn cubinder(n: usize) -> Self {
        let mut vertices = Vec::new();
        for i in 0..n {
            let a = 2.0 * PI * i as f64 / n as f64;
            for &z in &[-0.5, 0.5] {
                for &w in &[-0.5, 0.5] {
                    vertices.push(Vec4::new(a.cos(), a.sin(), z, w));
                }
            }
        }
        let edges = edges_by_min_distance(&vertices);
        Polytope4 {
            vertices,
            edges,
            faces: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Discretized spherinder (sphere x segment); vertices/edges only.
    #[must_use]
    pub fn spherinder(n: usize) -> Self {
        let mut vertices = Vec::new();
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        for k in 0..n {
            let kf = k as f64 + 0.5;
            let z = 1.0 - 2.0 * kf / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = 2.0 * PI * kf * golden;
            for &w in &[-0.5, 0.5] {
                vertices.push(Vec4::new(r * th.cos(), r * th.sin(), z, w));
            }
        }
        let edges = edges_by_min_distance(&vertices);
        Polytope4 {
            vertices,
            edges,
            faces: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Rectified polytope: vertices at edge midpoints (vertices and edges).
    #[must_use]
    pub fn rectified(&self) -> Self {
        let vertices: Vec<Vec4> = self
            .edges
            .iter()
            .map(|&(a, b)| self.vertices[a].add(&self.vertices[b]).scale(0.5))
            .collect();
        let edges = edges_by_min_distance(&vertices);
        Polytope4 {
            vertices,
            edges,
            faces: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Truncated polytope: two vertices per edge at the one-third points
    /// (vertices and edges).
    #[must_use]
    pub fn truncated(&self) -> Self {
        let mut vertices = Vec::new();
        for &(a, b) in &self.edges {
            let va = self.vertices[a];
            let vb = self.vertices[b];
            vertices.push(va.scale(2.0 / 3.0).add(&vb.scale(1.0 / 3.0)));
            vertices.push(va.scale(1.0 / 3.0).add(&vb.scale(2.0 / 3.0)));
        }
        let edges = edges_by_min_distance(&vertices);
        Polytope4 {
            vertices,
            edges,
            faces: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Dual polytope: vertices at cell centers, full combinatorics from the
    /// original vertex directions.
    #[must_use]
    pub fn dual(&self) -> Self {
        let vertices: Vec<Vec4> = self
            .cells
            .iter()
            .map(|cell| {
                let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
                for &v in cell {
                    c = c.add(&self.vertices[v]);
                }
                c.normalized()
            })
            .collect();
        let dirs = self.vertices.clone();
        Self::from_vertices_and_dirs(vertices, &dirs)
    }

    /// Euler characteristic V - E + F - C (zero for all convex 4-polytopes).
    #[must_use]
    pub fn euler_characteristic(&self) -> i64 {
        self.vertices.len() as i64 - self.edges.len() as i64 + self.faces.len() as i64
            - self.cells.len() as i64
    }

    /// (V, E, F, C).
    #[must_use]
    pub fn f_vector(&self) -> [usize; 4] {
        [
            self.vertices.len(),
            self.edges.len(),
            self.faces.len(),
            self.cells.len(),
        ]
    }

    /// Common edge length.
    #[must_use]
    pub fn edge_length(&self) -> f64 {
        let (a, b) = self.edges[0];
        self.vertices[a].sub(&self.vertices[b]).norm()
    }

    /// Rotate all vertices by a 4D rotation.
    #[must_use]
    pub fn rotate(&self, r: &So4) -> Self {
        let vertices = self
            .vertices
            .iter()
            .map(|v| {
                let p = r.apply([v.x, v.y, v.z, v.w]);
                Vec4::new(p[0], p[1], p[2], p[3])
            })
            .collect();
        Polytope4 {
            vertices,
            edges: self.edges.clone(),
            faces: self.faces.clone(),
            cells: self.cells.clone(),
        }
    }

    /// Perspective projection from w = `distance` into 3D, triangulating the
    /// faces into a mesh.
    #[must_use]
    pub fn project_perspective(&self, distance: f64) -> Mesh {
        let verts3: Vec<Vec3> = self
            .vertices
            .iter()
            .map(|v| {
                let s = 1.0 / (distance - v.w);
                Vec3::new(v.x * s, v.y * s, v.z * s)
            })
            .collect();
        let mut mesh = Mesh::new();
        mesh.vertices = verts3;
        for face in &self.faces {
            for k in 1..face.len() - 1 {
                mesh.triangles.push([face[0], face[k], face[k + 1]]);
            }
        }
        mesh.materials = vec![0; mesh.triangles.len()];
        mesh
    }

    /// Orthographic projection dropping one axis.
    #[must_use]
    pub fn project_orthographic(&self, drop_axis: usize) -> Vec<Vec3> {
        self.vertices
            .iter()
            .map(|v| {
                let mut out = [0.0; 3];
                let mut k = 0;
                for i in 0..4 {
                    if i != drop_axis {
                        out[k] = v.get(i);
                        k += 1;
                    }
                }
                Vec3::new(out[0], out[1], out[2])
            })
            .collect()
    }

    /// Stereographic projection from S3 (for unit-radius polytopes such as
    /// the 120-cell and 600-cell).
    #[must_use]
    pub fn project_stereographic(&self) -> Vec<Vec3> {
        self.vertices
            .iter()
            .map(|v| {
                let u = v.normalized();
                let s = 1.0 / (1.0 - u.w).max(1e-9);
                Vec3::new(u.x * s, u.y * s, u.z * s)
            })
            .collect()
    }

    /// 3D cross-section by the hyperplane w = const (convex hull of the
    /// edge-hyperplane intersection points).
    #[must_use]
    pub fn cross_section(&self, w: f64) -> Mesh {
        self.cross_section_oriented(Vec4::new(0.0, 0.0, 0.0, 1.0), w)
    }

    /// Cross-section by the hyperplane normal . x = offset.
    #[must_use]
    pub fn cross_section_oriented(&self, normal: Vec4, offset: f64) -> Mesh {
        let n = normal.normalized();
        // basis of the hyperplane
        let mut basis = Vec::new();
        for i in 0..4 {
            let mut e = Vec4::new(0.0, 0.0, 0.0, 0.0);
            e.set(i, 1.0);
            let perp = e.sub(&n.scale(e.dot(&n)));
            if perp.norm() > 1e-9 {
                let mut b = perp;
                for prev in &basis {
                    let p: &Vec4 = prev;
                    b = b.sub(&p.scale(b.dot(p)));
                }
                if b.norm() > 1e-9 {
                    basis.push(b.normalized());
                }
            }
            if basis.len() == 3 {
                break;
            }
        }
        let mut pts3 = Vec::new();
        for &(a, b) in &self.edges {
            let da = self.vertices[a].dot(&n) - offset;
            let db = self.vertices[b].dot(&n) - offset;
            if (da > 0.0) != (db > 0.0) {
                let t = da / (da - db);
                let p = self.vertices[a].scale(1.0 - t).add(&self.vertices[b].scale(t));
                let p3 = Vec3::new(p.dot(&basis[0]), p.dot(&basis[1]), p.dot(&basis[2]));
                if pts3
                    .iter()
                    .all(|q: &Vec3| (*q - p3).magnitude() > 1e-9)
                {
                    pts3.push(p3);
                }
            } else if da.abs() < 1e-12 {
                let v = self.vertices[a];
                let p3 = Vec3::new(v.dot(&basis[0]), v.dot(&basis[1]), v.dot(&basis[2]));
                if pts3.iter().all(|q: &Vec3| (*q - p3).magnitude() > 1e-9) {
                    pts3.push(p3);
                }
            }
        }
        let mut mesh = Mesh::new();
        if pts3.len() >= 4 {
            let tris = crate::geometry::hull::convex_hull_3d(&pts3);
            mesh.vertices = pts3;
            mesh.triangles = tris;
            mesh.materials = vec![0; mesh.triangles.len()];
        } else {
            mesh.vertices = pts3;
        }
        mesh
    }

    /// Unfold the cells into disjoint 3D meshes (a simple net: each cell is
    /// projected into its own hyperplane coordinates and translated apart;
    /// for the tesseract this is the classical 8-cube cross layout).
    #[must_use]
    pub fn unfold_net(&self) -> Vec<Mesh> {
        let mut out = Vec::new();
        for (k, cell) in self.cells.iter().enumerate() {
            // cell center and hyperplane basis
            let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
            for &v in cell {
                c = c.add(&self.vertices[v]);
            }
            c = c.scale(1.0 / cell.len() as f64);
            let nrm = c.normalized();
            let mut basis: Vec<Vec4> = Vec::new();
            for i in 0..4 {
                let mut e = Vec4::new(0.0, 0.0, 0.0, 0.0);
                e.set(i, 1.0);
                let mut b = e.sub(&nrm.scale(e.dot(&nrm)));
                for prev in &basis {
                    b = b.sub(&prev.scale(b.dot(prev)));
                }
                if b.norm() > 1e-9 {
                    basis.push(b.normalized());
                }
                if basis.len() == 3 {
                    break;
                }
            }
            // classical cross layout for 8 cells, else a row
            let offset = if self.cells.len() == 8 {
                match k {
                    0 => Vec3::new(0.0, 0.0, 0.0),
                    1 => Vec3::new(1.05, 0.0, 0.0),
                    2 => Vec3::new(-1.05, 0.0, 0.0),
                    3 => Vec3::new(0.0, 1.05, 0.0),
                    4 => Vec3::new(0.0, -1.05, 0.0),
                    5 => Vec3::new(0.0, 0.0, 1.05),
                    6 => Vec3::new(0.0, 0.0, -1.05),
                    _ => Vec3::new(0.0, -2.1, 0.0),
                }
            } else {
                Vec3::new(1.1 * k as f64 * self.edge_length(), 0.0, 0.0)
            };
            let mut mesh = Mesh::new();
            let local: Vec<Vec3> = cell
                .iter()
                .map(|&v| {
                    let d = self.vertices[v].sub(&c);
                    Vec3::new(d.dot(&basis[0]), d.dot(&basis[1]), d.dot(&basis[2])) + offset
                })
                .collect();
            mesh.vertices = local;
            if cell.len() >= 4 {
                let tris = crate::geometry::hull::convex_hull_3d(&mesh.vertices);
                mesh.triangles = tris;
            }
            mesh.materials = vec![0; mesh.triangles.len()];
            out.push(mesh);
        }
        out
    }

    /// Schlegel diagram: perspective projection from just outside the
    /// center of the chosen cell.
    #[must_use]
    pub fn schlegel_diagram(&self, cell: usize) -> Vec<Vec3> {
        let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for &v in &self.cells[cell] {
            c = c.add(&self.vertices[v]);
        }
        c = c.scale(1.0 / self.cells[cell].len() as f64);
        let n = c.normalized();
        let eye = n.scale(self.circumradius() * 1.5);
        // basis orthogonal to n
        let mut basis: Vec<Vec4> = Vec::new();
        for i in 0..4 {
            let mut e = Vec4::new(0.0, 0.0, 0.0, 0.0);
            e.set(i, 1.0);
            let mut b = e.sub(&n.scale(e.dot(&n)));
            for prev in &basis {
                b = b.sub(&prev.scale(b.dot(prev)));
            }
            if b.norm() > 1e-9 {
                basis.push(b.normalized());
            }
            if basis.len() == 3 {
                break;
            }
        }
        self.vertices
            .iter()
            .map(|v| {
                let d = v.sub(&eye);
                let depth = -d.dot(&n);
                let s = 1.0 / depth.max(1e-9);
                Vec3::new(
                    d.dot(&basis[0]) * s,
                    d.dot(&basis[1]) * s,
                    d.dot(&basis[2]) * s,
                )
            })
            .collect()
    }

    /// Vertex figure: the polyhedron whose vertices are the neighbors of
    /// `v`, in the neighbors' midpoint positions (returned as a hull mesh).
    #[must_use]
    pub fn vertex_figure(&self, v: usize) -> Mesh {
        let nbrs: Vec<usize> = self
            .edges
            .iter()
            .filter_map(|&(a, b)| {
                if a == v {
                    Some(b)
                } else if b == v {
                    Some(a)
                } else {
                    None
                }
            })
            .collect();
        let center = self.vertices[v];
        // midpoints projected into the tangent hyperplane at v
        let n = center.normalized();
        let mut basis: Vec<Vec4> = Vec::new();
        for i in 0..4 {
            let mut e = Vec4::new(0.0, 0.0, 0.0, 0.0);
            e.set(i, 1.0);
            let mut b = e.sub(&n.scale(e.dot(&n)));
            for prev in &basis {
                b = b.sub(&prev.scale(b.dot(prev)));
            }
            if b.norm() > 1e-9 {
                basis.push(b.normalized());
            }
            if basis.len() == 3 {
                break;
            }
        }
        let pts: Vec<Vec3> = nbrs
            .iter()
            .map(|&u| {
                let m = self.vertices[u].add(&center).scale(0.5).sub(&center);
                Vec3::new(m.dot(&basis[0]), m.dot(&basis[1]), m.dot(&basis[2]))
            })
            .collect();
        let mut mesh = Mesh::new();
        if pts.len() >= 4 {
            mesh.triangles = crate::geometry::hull::convex_hull_3d(&pts);
        }
        mesh.vertices = pts;
        mesh.materials = vec![0; mesh.triangles.len()];
        mesh
    }

    /// Order of the full symmetry group (regular polytopes only).
    #[must_use]
    pub fn symmetry_order(&self) -> usize {
        match self.cells.len() {
            5 => 120,
            8 | 16 => 384,
            24 => 1152,
            120 | 600 => 14400,
            _ => 0,
        }
    }

    /// A handful of rotations generating (a subgroup of) the symmetry
    /// group: rotations by the face angle in coordinate planes that map the
    /// vertex set to itself.
    #[must_use]
    pub fn coxeter_group_generators(&self) -> Vec<So4> {
        let planes = rotation_4d_planes();
        let mut gens = Vec::new();
        let is_symmetry = |r: &So4| -> bool {
            self.vertices.iter().all(|v| {
                let img = r.apply([v.x, v.y, v.z, v.w]);
                let ip = Vec4::new(img[0], img[1], img[2], img[3]);
                self.vertices.iter().any(|u| u.sub(&ip).norm() < 1e-6)
            })
        };
        for &plane in &planes {
            for k in 1..=6 {
                let angle = 2.0 * PI / (k as f64 + 1.0);
                let r = So4::simple_rotation(plane, angle);
                if is_symmetry(&r) {
                    gens.push(r);
                    break;
                }
            }
        }
        gens
    }

    /// Construct from a Schlafli/Wythoff symbol.
    #[must_use]
    pub fn from_wythoff(symbol: &str) -> Option<Self> {
        match symbol {
            "{3,3,3}" => Some(Polytope4::simplex5()),
            "{4,3,3}" => Some(Polytope4::tesseract()),
            "{3,3,4}" => Some(Polytope4::cell16()),
            "{3,4,3}" => Some(Polytope4::cell24()),
            "{5,3,3}" => Some(Polytope4::cell120()),
            "{3,3,5}" => Some(Polytope4::cell600()),
            "t{4,3,3}" => Some(Polytope4::tesseract().truncated()),
            "t{3,3,4}" => Some(Polytope4::cell16().truncated()),
            "r{4,3,3}" => Some(Polytope4::tesseract().rectified()),
            _ => None,
        }
    }

    /// Dihedral angle between adjacent cells (across a shared face).
    #[must_use]
    pub fn dihedral_angle(&self) -> f64 {
        // find two cells sharing a face; the dihedral angle is pi minus the
        // angle between their outward hyperplane normals (cell centers)
        for i in 0..self.cells.len() {
            let si: std::collections::HashSet<usize> = self.cells[i].iter().copied().collect();
            for j in (i + 1)..self.cells.len() {
                let shared = self.cells[j].iter().filter(|v| si.contains(v)).count();
                if shared >= 3 {
                    let center = |c: &[usize]| {
                        let mut s = Vec4::new(0.0, 0.0, 0.0, 0.0);
                        for &v in c {
                            s = s.add(&self.vertices[v]);
                        }
                        s.normalized()
                    };
                    let n1 = center(&self.cells[i]);
                    let n2 = center(&self.cells[j]);
                    return PI - n1.dot(&n2).clamp(-1.0, 1.0).acos();
                }
            }
        }
        0.0
    }

    /// Circumradius (max vertex distance from the origin).
    #[must_use]
    pub fn circumradius(&self) -> f64 {
        self.vertices.iter().map(Vec4::norm).fold(0.0, f64::max)
    }

    /// Inradius (distance from the origin to a cell hyperplane).
    #[must_use]
    pub fn inradius(&self) -> f64 {
        self.cells
            .iter()
            .map(|cell| {
                let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
                for &v in cell {
                    c = c.add(&self.vertices[v]);
                }
                let n = c.normalized();
                self.vertices[cell[0]].dot(&n)
            })
            .fold(f64::MAX, f64::min)
    }

    /// Volume of one cell (3D) by fan decomposition into tetrahedra from
    /// the cell centroid over its (triangulated) faces.
    fn cell_volume(&self, ci: usize) -> f64 {
        let cell = &self.cells[ci];
        let cset: std::collections::HashSet<usize> = cell.iter().copied().collect();
        let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for &v in cell {
            c = c.add(&self.vertices[v]);
        }
        c = c.scale(1.0 / cell.len() as f64);
        let n = c.normalized();
        // hyperplane basis
        let mut basis: Vec<Vec4> = Vec::new();
        for i in 0..4 {
            let mut e = Vec4::new(0.0, 0.0, 0.0, 0.0);
            e.set(i, 1.0);
            let mut b = e.sub(&n.scale(e.dot(&n)));
            for prev in &basis {
                b = b.sub(&prev.scale(b.dot(prev)));
            }
            if b.norm() > 1e-9 {
                basis.push(b.normalized());
            }
            if basis.len() == 3 {
                break;
            }
        }
        let to3 = |v: usize| {
            let d = self.vertices[v].sub(&c);
            Vec3::new(d.dot(&basis[0]), d.dot(&basis[1]), d.dot(&basis[2]))
        };
        let mut vol = 0.0;
        for face in &self.faces {
            if !face.iter().all(|v| cset.contains(v)) {
                continue;
            }
            for k in 1..face.len() - 1 {
                let (a, b, d) = (to3(face[0]), to3(face[k]), to3(face[k + 1]));
                vol += a.dot(&b.cross(&d)).abs() / 6.0;
            }
        }
        vol
    }

    /// 4D hypervolume: sum of cone volumes over cells,
    /// (1/4) * cell volume * inradius contribution.
    #[must_use]
    pub fn hypervolume(&self) -> f64 {
        (0..self.cells.len())
            .map(|ci| {
                let cell = &self.cells[ci];
                let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
                for &v in cell {
                    c = c.add(&self.vertices[v]);
                }
                let n = c.normalized();
                let h = self.vertices[cell[0]].dot(&n);
                self.cell_volume(ci) * h / 4.0
            })
            .sum()
    }

    /// Total surface volume: sum of the 3D volumes of all cells.
    #[must_use]
    pub fn surface_volume(&self) -> f64 {
        (0..self.cells.len()).map(|ci| self.cell_volume(ci)).sum()
    }
}

// ---------------------------------------------------------------------------
// 4D rotation helpers and tori
// ---------------------------------------------------------------------------

/// The six coordinate rotation planes of R4.
#[must_use]
pub fn rotation_4d_planes() -> [(usize, usize); 6] {
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
}

/// Rotate a point in a single coordinate plane.
#[must_use]
pub fn rotate_4d(p: Vec4, plane: (usize, usize), angle: f64) -> Vec4 {
    let (a, b) = plane;
    let (s, c) = angle.sin_cos();
    let mut out = p;
    let pa = p.get(a);
    let pb = p.get(b);
    out.set(a, c * pa - s * pb);
    out.set(b, s * pa + c * pb);
    out
}

/// Double rotation: xy-plane by `angle_xy`, zw-plane by `angle_zw`.
#[must_use]
pub fn rotate_4d_double(p: Vec4, angle_xy: f64, angle_zw: f64) -> Vec4 {
    rotate_4d(rotate_4d(p, (0, 1), angle_xy), (2, 3), angle_zw)
}

/// Point on the Clifford torus in S3: parameter angles (u, v), aspect r.
#[must_use]
pub fn clifford_torus(u: f64, v: f64, r: f64) -> Vec4 {
    let c1 = r / (1.0 + r * r).sqrt();
    let c2 = 1.0 / (1.0 + r * r).sqrt();
    Vec4::new(c1 * u.cos(), c1 * u.sin(), c2 * v.cos(), c2 * v.sin())
}

/// Sampled Clifford torus grid.
#[must_use]
pub fn clifford_torus_mesh(nu: usize, nv: usize, r: f64) -> Vec<Vec4> {
    let mut out = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        for j in 0..nv {
            out.push(clifford_torus(
                2.0 * PI * i as f64 / nu as f64,
                2.0 * PI * j as f64 / nv as f64,
                r,
            ));
        }
    }
    out
}

/// Near-uniform points on S3 (from the super-Fibonacci quaternions).
#[must_use]
pub fn hypersphere_s3_points(n: usize) -> Vec<Vec4> {
    crate::manifold::spherical::s3_uniform_points(n)
        .into_iter()
        .map(|q| Vec4::new(q.w, q.x, q.y, q.z))
        .collect()
}

/// Volume of the n-ball of radius r (alias into the spherical module).
#[must_use]
pub fn hypersphere_volume(r: f64, n: usize) -> f64 {
    crate::manifold::spherical::sphere_volume_n(r, n)
}

// ---------------------------------------------------------------------------
// Regular polytopes in any dimension
// ---------------------------------------------------------------------------

/// Regular n-simplex: n+1 vertices in R^n, unit circumradius, with edges.
#[must_use]
pub fn simplex_n(n: usize) -> (Vec<VecN>, Vec<(usize, usize)>) {
    // n+1 points: e_i embedded then centered and normalized
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let mut v = VecN::zeros(n + 1);
        v.data[i] = 1.0;
        pts.push(v);
    }
    let center = VecN::ones(n + 1).scale(1.0 / (n as f64 + 1.0));
    // project into n-dim subspace orthogonal to (1,..,1): use Gram-Schmidt
    let mut basis = Vec::new();
    for i in 0..n {
        let mut v = VecN::zeros(n + 1);
        v.data[i] = 1.0;
        v.data[n] = -1.0;
        for b in &basis {
            let bb: &VecN = b;
            v = v.sub(&bb.scale(v.dot(bb)));
        }
        basis.push(v.normalized());
    }
    let verts: Vec<VecN> = pts
        .iter()
        .map(|p| {
            let d = p.sub(&center);
            VecN::from(
                &basis
                    .iter()
                    .map(|b| d.dot(b))
                    .collect::<Vec<f64>>(),
            )
            .normalized()
        })
        .collect();
    let mut edges = Vec::new();
    for i in 0..=n {
        for j in (i + 1)..=n {
            edges.push((i, j));
        }
    }
    (verts, edges)
}

/// n-cube vertices (coordinates +-1/2) with edges.
#[must_use]
pub fn hypercube_n(n: usize) -> (Vec<VecN>, Vec<(usize, usize)>) {
    let count = 1usize << n;
    let verts: Vec<VecN> = (0..count)
        .map(|mask| {
            VecN::from(
                &(0..n)
                    .map(|b| if mask >> b & 1 == 1 { 0.5 } else { -0.5 })
                    .collect::<Vec<f64>>(),
            )
        })
        .collect();
    (verts, hypercube_graph_n(n))
}

/// n-dimensional cross-polytope (unit vertices +-e_i) with edges.
#[must_use]
pub fn cross_polytope_n(n: usize) -> (Vec<VecN>, Vec<(usize, usize)>) {
    let mut verts = Vec::with_capacity(2 * n);
    for i in 0..n {
        for &s in &[1.0, -1.0] {
            let mut v = VecN::zeros(n);
            v.data[i] = s;
            verts.push(v);
        }
    }
    let mut edges = Vec::new();
    for a in 0..2 * n {
        for b in (a + 1)..2 * n {
            // connect unless antipodal (same axis)
            if a / 2 != b / 2 {
                edges.push((a, b));
            }
        }
    }
    (verts, edges)
}

/// Edges of the n-cube graph (bitmask vertices, Hamming distance 1).
#[must_use]
pub fn hypercube_graph_n(n: usize) -> Vec<(usize, usize)> {
    let count = 1usize << n;
    let mut edges = Vec::new();
    for v in 0..count {
        for b in 0..n {
            let u = v ^ (1 << b);
            if u > v {
                edges.push((v, u));
            }
        }
    }
    edges
}

/// Project n-dimensional points into 3D with the given (orthonormal) basis.
#[must_use]
pub fn project_n_to_3(points: &[VecN], basis: [VecN; 3]) -> Vec<Vec3> {
    points
        .iter()
        .map(|p| Vec3::new(p.dot(&basis[0]), p.dot(&basis[1]), p.dot(&basis[2])))
        .collect()
}

/// Project n-dimensional points into 2D.
#[must_use]
pub fn project_n_to_2(points: &[VecN], basis: [VecN; 2]) -> Vec<Vec2> {
    points
        .iter()
        .map(|p| Vec2::new(p.dot(&basis[0]), p.dot(&basis[1])))
        .collect()
}

/// Petrie polygon projection of a regular 4-polytope: its vertices
/// projected into the Coxeter plane of the matching symmetry group, using
/// root systems realized in the polytope's own coordinates.
#[must_use]
pub fn petrie_polygon_projection(p: &Polytope4) -> Vec<Vec2> {
    let verts: Vec<VecN> = p
        .vertices
        .iter()
        .map(|v| VecN::from(&[v.x, v.y, v.z, v.w]))
        .collect();
    match p.cells.len() {
        5 => {
            // A4 roots: normalized differences of the 5 vertices
            let mut roots = Vec::new();
            for i in 0..p.vertices.len() {
                for j in 0..p.vertices.len() {
                    if i != j {
                        let d = p.vertices[i].sub(&p.vertices[j]);
                        roots.push(VecN::from(&[d.x, d.y, d.z, d.w]).normalized());
                    }
                }
            }
            let gram = chain_gram(&[3.0, 3.0, 3.0]);
            project_via_roots(&verts, &roots, &gram, 5)
        }
        24 => coxeter_plane_projection(&verts, "F4"),
        120 | 600 => coxeter_plane_projection(&verts, "H4"),
        _ => coxeter_plane_projection(&verts, "B4"),
    }
}

/// Gram matrix of unit simple roots for a linear Coxeter diagram with the
/// given edge labels.
fn chain_gram(labels: &[f64]) -> Vec<Vec<f64>> {
    let n = labels.len() + 1;
    let mut g = vec![vec![0.0; n]; n];
    for (i, row) in g.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for (k, &m) in labels.iter().enumerate() {
        let c = -(PI / m).cos();
        g[k][k + 1] = c;
        g[k + 1][k] = c;
    }
    g
}

/// Backtracking search for a simple system with the target Gram matrix
/// inside a root set.
fn find_simple_system(roots: &[VecN], gram: &[Vec<f64>]) -> Option<Vec<VecN>> {
    let n = gram.len();
    fn go(
        roots: &[VecN],
        gram: &[Vec<f64>],
        chosen: &mut Vec<usize>,
        budget: &mut usize,
    ) -> bool {
        if chosen.len() == gram.len() {
            return true;
        }
        let k = chosen.len();
        for cand in 0..roots.len() {
            if *budget == 0 {
                return false;
            }
            *budget -= 1;
            let ok = chosen.iter().enumerate().all(|(j, &cj)| {
                (roots[cand].dot(&roots[cj]) - gram[k][j]).abs() < 1e-6
            });
            if ok {
                chosen.push(cand);
                if go(roots, gram, chosen, budget) {
                    return true;
                }
                chosen.pop();
            }
        }
        false
    }
    let mut chosen = Vec::with_capacity(n);
    let mut budget = 2_000_000usize;
    if go(roots, gram, &mut chosen, &mut budget) {
        Some(chosen.into_iter().map(|i| roots[i].clone()).collect())
    } else {
        None
    }
}

fn project_via_roots(
    vertices: &[VecN],
    roots: &[VecN],
    gram: &[Vec<f64>],
    h: usize,
) -> Vec<Vec2> {
    let simple = find_simple_system(roots, gram).expect("no simple system found");
    let dim = simple[0].dim();
    let reflect = |r: &VecN| -> Matrix {
        let rr = r.dot(r);
        Matrix::from_fn(dim, dim, |i, j| {
            (if i == j { 1.0 } else { 0.0 }) - 2.0 * r[i] * r[j] / rr
        })
    };
    let mut w = Matrix::identity(dim);
    for r in &simple {
        w = w.mul(&reflect(r)).unwrap();
    }
    let sym = Matrix::from_fn(dim, dim, |i, j| 0.5 * (w.get(i, j) + w.get(j, i)));
    let target = (2.0 * PI / h as f64).cos();
    let eig = eigen_symmetric(&sym, 1e-12, 300).expect("eigen failed");
    let mut idx: Vec<usize> = (0..eig.values.len()).collect();
    idx.sort_by(|&a, &b| {
        (eig.values[a] - target)
            .abs()
            .partial_cmp(&(eig.values[b] - target).abs())
            .unwrap()
    });
    let u = VecN::from(
        &(0..dim)
            .map(|r| eig.vectors.get(r, idx[0]))
            .collect::<Vec<f64>>(),
    )
    .normalized();
    let mut v = VecN::from(
        &(0..dim)
            .map(|r| eig.vectors.get(r, idx[1]))
            .collect::<Vec<f64>>(),
    );
    v = v.sub(&u.scale(v.dot(&u))).normalized();
    vertices
        .iter()
        .map(|p| Vec2::new(p.dot(&u), p.dot(&v)))
        .collect()
}

/// Project points into the Coxeter plane of the given group ("B4", "D4",
/// "F4", "H4", "E8"), with root systems in standard coordinates. The plane
/// is the invariant plane of a Coxeter element, found as the 2D eigenspace
/// of (w + w^T)/2 with eigenvalue cos(2 pi/h).
#[must_use]
pub fn coxeter_plane_projection(vertices: &[VecN], group: &str) -> Vec<Vec2> {
    match group {
        "H4" => {
            let roots: Vec<VecN> = icosians()
                .iter()
                .map(|v| VecN::from(&[v.x, v.y, v.z, v.w]))
                .collect();
            let gram = chain_gram(&[5.0, 3.0, 3.0]);
            project_via_roots(vertices, &roots, &gram, 30)
        }
        "F4" => {
            let roots: Vec<VecN> = f4_roots()
                .iter()
                .map(|v| VecN::from(&[v.x, v.y, v.z, v.w]).normalized())
                .collect();
            let gram = chain_gram(&[3.0, 4.0, 3.0]);
            project_via_roots(vertices, &roots, &gram, 12)
        }
        "B4" => {
            let mut roots = Vec::new();
            for i in 0..4 {
                for j in 0..4 {
                    if i == j {
                        continue;
                    }
                    for &si in &[1.0_f64, -1.0] {
                        for &sj in &[1.0_f64, -1.0] {
                            let mut v = VecN::zeros(4);
                            v.data[i] = si;
                            v.data[j] = sj;
                            roots.push(v.normalized());
                        }
                    }
                }
            }
            for i in 0..4 {
                for &s in &[1.0_f64, -1.0] {
                    let mut v = VecN::zeros(4);
                    v.data[i] = s;
                    roots.push(v);
                }
            }
            let gram = chain_gram(&[4.0, 3.0, 3.0]);
            project_via_roots(vertices, &roots, &gram, 8)
        }
        "D4" => {
            let mut roots = Vec::new();
            for i in 0..4 {
                for j in (i + 1)..4 {
                    for &si in &[1.0_f64, -1.0] {
                        for &sj in &[1.0_f64, -1.0] {
                            let mut v = VecN::zeros(4);
                            v.data[i] = si;
                            v.data[j] = sj;
                            roots.push(v.normalized());
                        }
                    }
                }
            }
            // D4 star diagram: center node 1 connected to 0, 2, 3
            let c = -0.5;
            let gram = vec![
                vec![1.0, c, 0.0, 0.0],
                vec![c, 1.0, c, c],
                vec![0.0, c, 1.0, 0.0],
                vec![0.0, c, 0.0, 1.0],
            ];
            project_via_roots(vertices, &roots, &gram, 6)
        }
        "E8" => {
            let roots: Vec<VecN> = e8_roots()
                .iter()
                .map(|r| r.scale(1.0 / 2.0_f64.sqrt()))
                .collect();
            // E8 diagram: chain 0-1-2-3-4-5-6 with node 7 attached to node 4
            let c = -0.5;
            let mut gram = vec![vec![0.0; 8]; 8];
            for (i, row) in gram.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            for k in 0..6 {
                gram[k][k + 1] = c;
                gram[k + 1][k] = c;
            }
            gram[4][7] = c;
            gram[7][4] = c;
            project_via_roots(vertices, &roots, &gram, 30)
        }
        _ => vertices
            .iter()
            .map(|p| Vec2::new(p[0], p[1]))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Root systems and lattices
// ---------------------------------------------------------------------------

/// The 240 roots of E8 (norm sqrt 2).
#[must_use]
pub fn e8_roots() -> Vec<VecN> {
    let mut roots = Vec::with_capacity(240);
    // 112 integer roots: +-e_i +- e_j
    for i in 0..8 {
        for j in (i + 1)..8 {
            for &si in &[1.0, -1.0] {
                for &sj in &[1.0, -1.0] {
                    let mut v = VecN::zeros(8);
                    v.data[i] = si;
                    v.data[j] = sj;
                    roots.push(v);
                }
            }
        }
    }
    // 128 half-integer roots with an even number of minus signs
    for mask in 0..256usize {
        if mask.count_ones().is_multiple_of(2) {
            let v = VecN::from(
                &(0..8)
                    .map(|b| if mask >> b & 1 == 1 { -0.5 } else { 0.5 })
                    .collect::<Vec<f64>>(),
            );
            roots.push(v);
        }
    }
    roots
}

/// Nearest E8 lattice point (D8 plus glue-vector decoding).
#[must_use]
pub fn e8_lattice_nearest(p: &VecN) -> VecN {
    // E8 = D8 union (D8 + (1/2,...,1/2)); decode both cosets, keep the
    // nearer
    let decode_d8 = |q: &VecN| -> VecN {
        // round each coordinate; if the sum is odd, flip the coordinate
        // with the largest rounding error
        let mut r: Vec<f64> = q.data.iter().map(|&x| x.round()).collect();
        let sum: f64 = r.iter().sum();
        if (sum.rem_euclid(2.0)).abs() > 0.5 {
            let mut worst = 0;
            let mut werr = -1.0;
            for (i, (&x, &ri)) in q.data.iter().zip(&r).enumerate() {
                let err = (x - ri).abs();
                if err > werr {
                    werr = err;
                    worst = i;
                }
            }
            let x = q.data[worst];
            r[worst] = if x > r[worst] { r[worst] + 1.0 } else { r[worst] - 1.0 };
        }
        VecN::from(&r)
    };
    let c1 = decode_d8(p);
    let shifted = p.sub(&VecN::ones(8).scale(0.5));
    let c2 = decode_d8(&shifted).add(&VecN::ones(8).scale(0.5));
    if p.sub(&c1).norm() <= p.sub(&c2).norm() {
        c1
    } else {
        c2
    }
}

/// The Leech lattice minimal-vector count (kissing number in 24D).
#[must_use]
pub fn leech_lattice_min_vectors_count() -> usize {
    196_560
}

/// D4 lattice points (integer coordinates, even sum) within radius r.
#[must_use]
pub fn d4_lattice_points(r: f64) -> Vec<Vec4> {
    let m = r.ceil() as i64;
    let mut out = Vec::new();
    for x in -m..=m {
        for y in -m..=m {
            for z in -m..=m {
                for w in -m..=m {
                    if (x + y + z + w) % 2 == 0 {
                        let p = Vec4::new(x as f64, y as f64, z as f64, w as f64);
                        if p.norm() <= r + 1e-12 {
                            out.push(p);
                        }
                    }
                }
            }
        }
    }
    out
}

/// The 48 roots of F4.
#[must_use]
pub fn f4_roots() -> Vec<Vec4> {
    let mut out = Vec::new();
    // 24 long roots: +-e_i +- e_j
    for i in 0..4 {
        for j in (i + 1)..4 {
            for &si in &[1.0, -1.0] {
                for &sj in &[1.0, -1.0] {
                    let mut p = Vec4::new(0.0, 0.0, 0.0, 0.0);
                    p.set(i, si);
                    p.set(j, sj);
                    out.push(p);
                }
            }
        }
    }
    // 8 short: +-e_i
    for i in 0..4 {
        for &s in &[1.0, -1.0] {
            let mut p = Vec4::new(0.0, 0.0, 0.0, 0.0);
            p.set(i, s);
            out.push(p);
        }
    }
    // 16 short: (+-1/2)^4
    for &x in &[0.5, -0.5] {
        for &y in &[0.5, -0.5] {
            for &z in &[0.5, -0.5] {
                for &w in &[0.5, -0.5] {
                    out.push(Vec4::new(x, y, z, w));
                }
            }
        }
    }
    out
}

/// The 120 roots of H4 (the unit icosians).
#[must_use]
pub fn h4_roots() -> Vec<Vec4> {
    icosians()
}

/// Known kissing numbers by dimension.
#[must_use]
pub fn kissing_number_known(n: usize) -> Option<usize> {
    match n {
        1 => Some(2),
        2 => Some(6),
        3 => Some(12),
        4 => Some(24),
        8 => Some(240),
        24 => Some(196_560),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// High-dimensional phenomena
// ---------------------------------------------------------------------------

/// (n-1)-volume of the slice of the unit n-cube [0,1]^n by the hyperplane
/// sum(x) = s, times sqrt(n) (the Irwin-Hall density scaled to a volume).
#[must_use]
pub fn hypercube_slicing_volume(n: usize, s: f64) -> f64 {
    // Irwin-Hall pdf: f(s) = 1/(n-1)! sum_k (-1)^k C(n,k) (s-k)^{n-1}_+
    if s < 0.0 || s > n as f64 {
        return 0.0;
    }
    let mut fact = 1.0;
    for k in 1..n {
        fact *= k as f64;
    }
    let mut sum = 0.0;
    let mut binom = 1.0;
    for k in 0..=(s.floor() as usize) {
        let term = binom * (s - k as f64).powi(n as i32 - 1);
        sum += if k % 2 == 0 { term } else { -term };
        binom = binom * (n - k) as f64 / (k as f64 + 1.0);
    }
    (n as f64).sqrt() * sum / fact
}

/// Fraction of the (n-1)-sphere's surface within angle theta of a pole:
/// regularized incomplete beta I_{sin^2 theta}((n-1)/2, 1/2) / 2 for
/// theta <= pi/2.
#[must_use]
pub fn hypersphere_cap_fraction(n: usize, theta: f64) -> f64 {
    if theta <= 0.0 {
        return 0.0;
    }
    if theta >= PI {
        return 1.0;
    }
    let a = (n as f64 - 1.0) / 2.0;
    let x = theta.sin().powi(2);
    let half = 0.5 * crate::special::beta_inc(a, 0.5, x);
    if theta <= 0.5 * PI {
        half
    } else {
        1.0 - half
    }
}

/// Gaussian mass concentrates at radius sqrt(n).
#[must_use]
pub fn gaussian_concentration_radius(n: usize) -> f64 {
    (n as f64).sqrt()
}

/// Monte Carlo probability that a simple random walk on Z^n returns to the
/// origin within `steps` steps (deterministic internal seed).
#[must_use]
pub fn random_walk_n_return_prob(n: usize, steps: usize) -> f64 {
    let trials = 4000;
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut rand = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as usize
    };
    let mut returned = 0;
    for _ in 0..trials {
        let mut pos = vec![0i64; n];
        for _ in 0..steps {
            let r = rand();
            let axis = r % n;
            let dir = if (r / n).is_multiple_of(2) { 1 } else { -1 };
            pos[axis] += dir;
            if pos.iter().all(|&x| x == 0) {
                returned += 1;
                break;
            }
        }
    }
    returned as f64 / trials as f64
}

/// Ratio of the volume of the inscribed ball to the unit cube in n
/// dimensions (goes to zero fast).
#[must_use]
pub fn volume_ball_vs_cube_ratio(n: usize) -> f64 {
    hypersphere_volume(0.5, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_f_vectors() {
        let cases: Vec<(Polytope4, [usize; 4])> = vec![
            (Polytope4::simplex5(), [5, 10, 10, 5]),
            (Polytope4::tesseract(), [16, 32, 24, 8]),
            (Polytope4::cell16(), [8, 24, 32, 16]),
            (Polytope4::cell24(), [24, 96, 96, 24]),
        ];
        for (p, f) in &cases {
            assert_eq!(&p.f_vector(), f, "f-vector");
            assert_eq!(p.euler_characteristic(), 0);
        }
        let c600 = Polytope4::cell600();
        assert_eq!(c600.f_vector(), [120, 720, 1200, 600]);
        assert_eq!(c600.euler_characteristic(), 0);
        let c120 = Polytope4::cell120();
        assert_eq!(c120.f_vector(), [600, 1200, 720, 120]);
        assert_eq!(c120.euler_characteristic(), 0);
        // 120-cell cells are dodecahedra (20 vertices), faces are pentagons
        assert!(c120.cells.iter().all(|c| c.len() == 20));
        assert!(c120.faces.iter().all(|f| f.len() == 5));
        // 600-cell cells are tetrahedra, faces triangles
        assert!(c600.cells.iter().all(|c| c.len() == 4));
        assert!(c600.faces.iter().all(|f| f.len() == 3));
    }

    #[test]
    fn test_dual_relations() {
        // dual of the 120-cell is the 600-cell combinatorially
        let c120 = Polytope4::cell120();
        let dual = c120.dual();
        assert_eq!(dual.f_vector(), [120, 720, 1200, 600]);
        assert_eq!(dual.euler_characteristic(), 0);
        // tesseract dual is the 16-cell
        let td = Polytope4::tesseract().dual();
        assert_eq!(td.f_vector(), [8, 24, 32, 16]);
        // 24-cell is self-dual
        let sd = Polytope4::cell24().dual();
        assert_eq!(sd.f_vector(), [24, 96, 96, 24]);
    }

    #[test]
    fn test_cross_sections_and_projections() {
        let t = Polytope4::tesseract();
        // w = 0 slice of the tesseract is a cube
        let slice = t.cross_section(0.0);
        assert_eq!(slice.vertices.len(), 8, "cube has 8 vertices");
        for v in &slice.vertices {
            assert!((v.x.abs() - 0.5).abs() < 1e-9);
            assert!((v.y.abs() - 0.5).abs() < 1e-9);
            assert!((v.z.abs() - 0.5).abs() < 1e-9);
        }
        assert_eq!(slice.triangles.len(), 12, "cube hull triangulation");
        // oblique slice near a vertex is a small tetrahedron-like cell
        let corner = t.cross_section_oriented(Vec4::new(1.0, 1.0, 1.0, 1.0), 0.9);
        assert!(corner.vertices.len() >= 4);
        // perspective projection keeps all vertices finite
        let mesh = t.project_perspective(3.0);
        assert_eq!(mesh.vertices.len(), 16);
        assert!(mesh.vertices.iter().all(|v| v.magnitude().is_finite()));
        assert!(!mesh.triangles.is_empty());
        // orthographic drop-w of the tesseract gives two nested cubes'
        // vertex set (8 unique 3D points, each hit twice)
        let ortho = t.project_orthographic(3);
        assert_eq!(ortho.len(), 16);
        // stereographic projection of the 600-cell: all finite
        let s = Polytope4::cell600().project_stereographic();
        assert!(s.iter().all(|v| v.magnitude().is_finite()));
        // Schlegel diagram is finite
        let sd = t.schlegel_diagram(0);
        assert!(sd.iter().all(|v| v.magnitude().is_finite()));
        // unfold: tesseract has 8 cube cells in the net
        let net = t.unfold_net();
        assert_eq!(net.len(), 8);
        for m in &net {
            assert_eq!(m.vertices.len(), 8);
        }
        // vertex figure of a tesseract vertex is a tetrahedron (4 neighbors)
        let vf = t.vertex_figure(0);
        assert_eq!(vf.vertices.len(), 4);
        assert_eq!(vf.triangles.len(), 4);
    }

    #[test]
    fn test_metrics_and_rotation() {
        let t = Polytope4::tesseract();
        assert!((t.edge_length() - 1.0).abs() < 1e-12);
        assert!((t.circumradius() - 1.0).abs() < 1e-12);
        assert!((t.inradius() - 0.5).abs() < 1e-12);
        assert!((t.hypervolume() - 1.0).abs() < 1e-9, "tesseract volume {}", t.hypervolume());
        assert!((t.surface_volume() - 8.0).abs() < 1e-9);
        // dihedral angles
        assert!((t.dihedral_angle() - PI / 2.0).abs() < 1e-9);
        let c16 = Polytope4::cell16();
        assert!((c16.dihedral_angle() - (2.0 * PI / 3.0)).abs() < 1e-9);
        // 16-cell hypervolume = 2^4/4! * (edge sqrt2)... vertices +-e_i:
        // V = 2/3 for the standard cross-polytope
        assert!((c16.hypervolume() - 2.0 / 3.0).abs() < 1e-9, "16-cell volume {}", c16.hypervolume());
        // 120-cell dihedral 144 degrees
        let c120 = Polytope4::cell120();
        assert!(
            (c120.dihedral_angle() - 144.0_f64.to_radians()).abs() < 1e-6,
            "120-cell dihedral {}",
            c120.dihedral_angle().to_degrees()
        );
        // rotation preserves edge lengths and f-vector
        let r = So4::double_rotation(0.3, 0.7);
        let tr = t.rotate(&r);
        assert!((tr.edge_length() - 1.0).abs() < 1e-12);
        assert_eq!(tr.f_vector(), t.f_vector());
        // symmetry orders
        assert_eq!(t.symmetry_order(), 384);
        assert_eq!(Polytope4::simplex5().symmetry_order(), 120);
        assert_eq!(Polytope4::cell24().symmetry_order(), 1152);
        // generators actually map vertices to vertices
        let gens = t.coxeter_group_generators();
        assert!(!gens.is_empty());
        // wythoff lookup
        assert_eq!(
            Polytope4::from_wythoff("{3,4,3}").unwrap().f_vector(),
            [24, 96, 96, 24]
        );
        assert!(Polytope4::from_wythoff("{9,9,9}").is_none());
    }

    #[test]
    fn test_products_and_modified() {
        // 3,5-duoprism: 15 vertices, 30 edges, 23 faces, 8 cells
        let dp = Polytope4::duoprism(3, 5);
        assert_eq!(dp.f_vector(), [15, 30, 23, 8]);
        assert_eq!(dp.euler_characteristic(), 0);
        // rectified tesseract has 32 vertices (one per edge)
        let rt = Polytope4::tesseract().rectified();
        assert_eq!(rt.vertices.len(), 32);
        // truncated tesseract has 64 vertices
        let tt = Polytope4::tesseract().truncated();
        assert_eq!(tt.vertices.len(), 64);
        // grand antiprism: 100 vertices (600-cell minus two decagon rings)
        let ga = Polytope4::grand_antiprism();
        assert_eq!(ga.vertices.len(), 100);
        // discretized solids have vertices on expected radii
        let dc = Polytope4::duocylinder(12);
        assert!(dc
            .vertices
            .iter()
            .all(|v| ((v.x * v.x + v.y * v.y) - 1.0).abs() < 1e-9));
        let cb = Polytope4::cubinder(8);
        assert_eq!(cb.vertices.len(), 32);
        let sp = Polytope4::spherinder(20);
        assert_eq!(sp.vertices.len(), 40);
    }

    #[test]
    fn test_rotations_and_torus() {
        let p = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let q = rotate_4d(p, (0, 1), PI / 2.0);
        assert!((q.y - 1.0).abs() < 1e-12 && q.x.abs() < 1e-12);
        let d = rotate_4d_double(Vec4::new(1.0, 0.0, 1.0, 0.0), PI, PI);
        assert!((d.x + 1.0).abs() < 1e-12 && (d.z + 1.0).abs() < 1e-12);
        assert_eq!(rotation_4d_planes().len(), 6);
        // Clifford torus lies on S3
        for i in 0..8 {
            for j in 0..8 {
                let u = 2.0 * PI * i as f64 / 8.0;
                let v = 2.0 * PI * j as f64 / 8.0;
                let p = clifford_torus(u, v, 1.5);
                assert!((p.norm() - 1.0).abs() < 1e-12);
            }
        }
        assert_eq!(clifford_torus_mesh(6, 7, 1.0).len(), 42);
        let s3 = hypersphere_s3_points(50);
        assert!(s3.iter().all(|p| (p.norm() - 1.0).abs() < 1e-9));
        assert!((hypersphere_volume(1.0, 4) - PI * PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_ndim_polytopes() {
        // regular simplex: all pairwise distances equal
        let (verts, edges) = simplex_n(5);
        assert_eq!(verts.len(), 6);
        assert_eq!(edges.len(), 15);
        let d0 = verts[0].sub(&verts[1]).norm();
        for i in 0..6 {
            for j in (i + 1)..6 {
                assert!((verts[i].sub(&verts[j]).norm() - d0).abs() < 1e-10);
            }
        }
        // hypercube graph: n 2^{n-1} edges
        for n in 2..=6 {
            assert_eq!(hypercube_graph_n(n).len(), n * (1 << (n - 1)));
        }
        let (hv, he) = hypercube_n(4);
        assert_eq!(hv.len(), 16);
        assert_eq!(he.len(), 32);
        let (cv, ce) = cross_polytope_n(4);
        assert_eq!(cv.len(), 8);
        assert_eq!(ce.len(), 24);
        // projection helpers
        let basis3 = [
            VecN::unit(4, 0),
            VecN::unit(4, 1),
            VecN::unit(4, 2),
        ];
        let p3 = project_n_to_3(&hv, basis3);
        assert_eq!(p3.len(), 16);
        let basis2 = [VecN::unit(4, 0), VecN::unit(4, 1)];
        assert_eq!(project_n_to_2(&hv, basis2).len(), 16);
    }

    fn outer_ring_count(pts: &[Vec2]) -> usize {
        let rmax = pts
            .iter()
            .map(|p| (p.x * p.x + p.y * p.y).sqrt())
            .fold(0.0, f64::max);
        pts.iter()
            .filter(|p| ((p.x * p.x + p.y * p.y).sqrt() - rmax).abs() < 1e-6 * rmax)
            .count()
    }

    #[test]
    fn test_coxeter_projections() {
        // E8 Gosset picture: outermost ring of the 240 roots is a 30-gon
        let roots = e8_roots();
        let proj = coxeter_plane_projection(&roots, "E8");
        assert_eq!(proj.len(), 240);
        assert_eq!(outer_ring_count(&proj), 30, "E8 outer ring");
        // 600-cell Petrie polygon: outer ring of 30
        let c600 = Polytope4::cell600();
        let petrie = petrie_polygon_projection(&c600);
        assert_eq!(outer_ring_count(&petrie), 30, "600-cell outer ring");
        // 24-cell (F4, h = 12): outer ring of 12
        let c24 = Polytope4::cell24();
        let p24 = petrie_polygon_projection(&c24);
        assert_eq!(outer_ring_count(&p24), 12, "24-cell outer ring");
        // tesseract (B4, h = 8): outer ring of 8
        let t = Polytope4::tesseract();
        let pt = petrie_polygon_projection(&t);
        assert_eq!(outer_ring_count(&pt), 8, "tesseract outer ring");
        // 5-cell (A4, h = 5): all 5 vertices on one ring
        let s5 = Polytope4::simplex5();
        let ps = petrie_polygon_projection(&s5);
        assert_eq!(outer_ring_count(&ps), 5, "5-cell ring");
    }

    #[test]
    fn test_e8_and_lattices() {
        let roots = e8_roots();
        assert_eq!(roots.len(), 240);
        for r in &roots {
            assert!((r.norm() - 2.0_f64.sqrt()).abs() < 1e-12);
        }
        // each root has 56 neighbors at inner product 1
        let r0 = &roots[0];
        let n56 = roots.iter().filter(|r| (r0.dot(r) - 1.0).abs() < 1e-9).count();
        assert_eq!(n56, 56, "E8 neighbors");
        // lattice decoding: roots decode to themselves
        for r in roots.iter().step_by(17) {
            let near = e8_lattice_nearest(r);
            assert!(near.sub(r).norm() < 1e-12);
        }
        // perturbed points decode to the unperturbed lattice point
        let p = roots[3].add(&VecN::from(&[0.1, -0.2, 0.05, 0.1, -0.1, 0.2, 0.0, 0.1]));
        let near = e8_lattice_nearest(&p);
        assert!(near.sub(&roots[3]).norm() < 1e-12);
        assert_eq!(leech_lattice_min_vectors_count(), 196_560);
        // D4: kissing number 24 at radius sqrt 2
        let d4 = d4_lattice_points(2.0_f64.sqrt() + 1e-9);
        let kiss = d4
            .iter()
            .filter(|p| (p.norm() - 2.0_f64.sqrt()).abs() < 1e-9)
            .count();
        assert_eq!(kiss, 24);
        assert_eq!(f4_roots().len(), 48);
        assert_eq!(h4_roots().len(), 120);
        assert_eq!(kissing_number_known(8), Some(240));
        assert_eq!(kissing_number_known(5), None);
    }

    #[test]
    fn test_high_dim_phenomena() {
        // slicing: n=2, the diagonal slice at s=1 has length sqrt(2)
        assert!((hypercube_slicing_volume(2, 1.0) - 2.0_f64.sqrt()).abs() < 1e-12);
        // integral over s of V/sqrt(n) ds = cube volume 1
        for n in [2usize, 3, 5] {
            let steps = 2000;
            let mut total = 0.0;
            for k in 0..steps {
                let s = n as f64 * (k as f64 + 0.5) / steps as f64;
                total += hypercube_slicing_volume(n, s) / (n as f64).sqrt()
                    * (n as f64 / steps as f64);
            }
            assert!((total - 1.0).abs() < 1e-6, "n = {n}: {total}");
        }
        // cap fraction: hemisphere is half; 3D matches (1 - cos)/2
        for n in [3usize, 5, 8] {
            assert!((hypersphere_cap_fraction(n, PI / 2.0) - 0.5).abs() < 1e-9);
        }
        let th = 0.7;
        assert!(
            (hypersphere_cap_fraction(3, th) - 0.5 * (1.0 - th.cos())).abs() < 1e-9
        );
        // caps concentrate near the equator as n grows
        assert!(hypersphere_cap_fraction(50, 1.2) < hypersphere_cap_fraction(3, 1.2));
        assert!((gaussian_concentration_radius(100) - 10.0).abs() < 1e-12);
        // random walks return less often in higher dimensions
        let p1 = random_walk_n_return_prob(1, 200);
        let p3 = random_walk_n_return_prob(3, 200);
        assert!(p1 > 0.8, "1D walk returns: {p1}");
        assert!(p3 < p1, "3D transient vs 1D recurrent");
        // ball-in-cube ratio collapses
        assert!(volume_ball_vs_cube_ratio(2) > volume_ball_vs_cube_ratio(5));
        assert!(volume_ball_vs_cube_ratio(10) < 0.01);
        assert!((volume_ball_vs_cube_ratio(3) - PI / 6.0).abs() < 1e-12);
    }
}
