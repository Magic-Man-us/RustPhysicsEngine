//! Subdivision surfaces (Loop, Catmull-Clark, sqrt(3), midpoint) and
//! Laplacian-family smoothing.

use crate::math::Vec3;
use crate::mesh::Mesh;
use std::collections::HashMap;

type EdgeMap<T> = HashMap<(usize, usize), T>;

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// One level of Loop subdivision (Loop 1987): each triangle splits
/// into four; new edge vertices use the 3/8-1/8 stencil, old vertices
/// the valence-dependent β stencil, with the standard boundary rules
/// (midpoint and 3/4-1/8-1/8).
#[must_use]
pub fn loop_subdivide(m: &Mesh) -> Mesh {
    // Edge -> (opposite vertices, face count).
    let mut edges: EdgeMap<(Vec<usize>, usize)> = HashMap::new();
    for &[a, b, c] in &m.indices {
        for (u, v, w) in [(a, b, c), (b, c, a), (c, a, b)] {
            let e = edges.entry(edge_key(u, v)).or_insert((Vec::new(), 0));
            e.0.push(w);
            e.1 += 1;
        }
    }
    // Adjacency and boundary flags.
    let n = m.vertices.len();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut boundary_nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (&(a, b), &(_, count)) in &edges {
        neighbors[a].push(b);
        neighbors[b].push(a);
        if count == 1 {
            boundary_nbrs[a].push(b);
            boundary_nbrs[b].push(a);
        }
    }
    let mut face_count = vec![0usize; n];
    for &[a, b, c] in &m.indices {
        for v in [a, b, c] {
            face_count[v] += 1;
        }
    }
    // Even (old) vertices; corners (a single incident face) stay put.
    let mut vertices: Vec<Vec3> = (0..n)
        .map(|i| {
            let v = m.vertices[i];
            if !boundary_nbrs[i].is_empty() {
                if boundary_nbrs[i].len() == 2 && face_count[i] > 1 {
                    (boundary_nbrs[i].iter().map(|&j| m.vertices[j]).fold(Vec3::ZERO, |s, p| s + p))
                        * 0.125
                        + v * 0.75
                } else {
                    v // corner / non-manifold boundary: keep
                }
            } else {
                let k = neighbors[i].len();
                if k < 3 {
                    v
                } else {
                    let kf = k as f64;
                    let c = 0.375 + 0.25 * (2.0 * std::f64::consts::PI / kf).cos();
                    let beta = (0.625 - c * c) / kf;
                    let sum =
                        neighbors[i].iter().map(|&j| m.vertices[j]).fold(Vec3::ZERO, |s, p| s + p);
                    v * (1.0 - kf * beta) + sum * beta
                }
            }
        })
        .collect();
    // Odd (edge) vertices.
    let mut edge_vertex: EdgeMap<usize> = HashMap::new();
    for (&(a, b), (opposite, count)) in &edges {
        let p = if *count == 2 {
            (m.vertices[a] + m.vertices[b]) * 0.375
                + (m.vertices[opposite[0]] + m.vertices[opposite[1]]) * 0.125
        } else {
            (m.vertices[a] + m.vertices[b]) * 0.5
        };
        edge_vertex.insert((a, b), vertices.len());
        vertices.push(p);
    }
    let mut indices = Vec::with_capacity(m.indices.len() * 4);
    for &[a, b, c] in &m.indices {
        let ab = edge_vertex[&edge_key(a, b)];
        let bc = edge_vertex[&edge_key(b, c)];
        let ca = edge_vertex[&edge_key(c, a)];
        indices.push([a, ab, ca]);
        indices.push([b, bc, ab]);
        indices.push([c, ca, bc]);
        indices.push([ab, bc, ca]);
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// `n` levels of Loop subdivision.
#[must_use]
pub fn loop_subdivide_n(m: &Mesh, n: usize) -> Mesh {
    let mut out = m.clone();
    for _ in 0..n {
        out = loop_subdivide(&out);
    }
    out
}

/// Quadrilateral mesh (faces as counterclockwise vertex quadruples).
#[derive(Debug, Clone, PartialEq)]
pub struct QuadMesh {
    pub vertices: Vec<Vec3>,
    pub quads: Vec<[usize; 4]>,
}

impl QuadMesh {
    /// Triangulates each quad along its 0-2 diagonal.
    #[must_use]
    pub fn to_triangles(&self) -> Mesh {
        let mut indices = Vec::with_capacity(self.quads.len() * 2);
        for &[a, b, c, d] in &self.quads {
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
        Mesh { vertices: self.vertices.clone(), indices, normals: None, uvs: None }
    }

    /// Axis-aligned box of the given half extents: 8 vertices, 6
    /// outward quads.
    ///
    /// # Panics
    /// Panics unless all half extents are positive.
    #[must_use]
    pub fn from_box(half: Vec3) -> Self {
        assert!(
            half.x > 0.0 && half.y > 0.0 && half.z > 0.0,
            "from_box requires positive half extents"
        );
        let vertices = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -half.x } else { half.x },
                    if i & 2 == 0 { -half.y } else { half.y },
                    if i & 4 == 0 { -half.z } else { half.z },
                )
            })
            .collect();
        let quads = vec![
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [3, 2, 6, 7],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        Self { vertices, quads }
    }

    /// Flat grid of `nx` x `nz` quads in the xz plane (normals +y),
    /// centered at the origin.
    ///
    /// # Panics
    /// Panics unless `width, depth > 0` and `nx, nz >= 1`.
    #[must_use]
    pub fn from_grid(width: f64, depth: f64, nx: usize, nz: usize) -> Self {
        assert!(width > 0.0 && depth > 0.0, "from_grid requires positive size");
        assert!(nx >= 1 && nz >= 1, "from_grid requires at least one quad per axis");
        let mut vertices = Vec::with_capacity((nx + 1) * (nz + 1));
        for j in 0..=nz {
            for i in 0..=nx {
                vertices.push(Vec3::new(
                    width * (i as f64 / nx as f64 - 0.5),
                    0.0,
                    depth * (j as f64 / nz as f64 - 0.5),
                ));
            }
        }
        let at = |i: usize, j: usize| j * (nx + 1) + i;
        let mut quads = Vec::with_capacity(nx * nz);
        for j in 0..nz {
            for i in 0..nx {
                quads.push([at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)]);
            }
        }
        Self { vertices, quads }
    }
}

/// One level of Catmull-Clark subdivision (Catmull & Clark 1978):
/// face points at centroids, edge points averaging endpoints and face
/// points, old vertices moved by (F + 2R + (n-3)P)/n, standard
/// boundary rules. Each quad becomes four.
#[must_use]
pub fn catmull_clark(q: &QuadMesh) -> QuadMesh {
    let nv = q.vertices.len();
    let face_point: Vec<Vec3> = q
        .quads
        .iter()
        .map(|f| f.iter().map(|&i| q.vertices[i]).fold(Vec3::ZERO, |s, p| s + p) * 0.25)
        .collect();
    // Edge -> adjacent faces.
    let mut edge_faces: EdgeMap<Vec<usize>> = HashMap::new();
    for (fi, f) in q.quads.iter().enumerate() {
        for k in 0..4 {
            edge_faces.entry(edge_key(f[k], f[(k + 1) % 4])).or_default().push(fi);
        }
    }
    // Edge points.
    let mut edge_point: EdgeMap<usize> = HashMap::new();
    let mut vertices = vec![Vec3::ZERO; nv];
    let mut new_pts: Vec<Vec3> = face_point.clone();
    for (&(a, b), faces) in &edge_faces {
        let mid = (q.vertices[a] + q.vertices[b]) * 0.5;
        let p = if faces.len() == 2 {
            (q.vertices[a] + q.vertices[b] + face_point[faces[0]] + face_point[faces[1]]) * 0.25
        } else {
            mid
        };
        edge_point.insert((a, b), nv + new_pts.len());
        new_pts.push(p);
    }
    // Old vertices.
    let mut incident_faces: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (fi, f) in q.quads.iter().enumerate() {
        for &v in f {
            incident_faces[v].push(fi);
        }
    }
    let mut incident_edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];
    let mut boundary_nbrs: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (&(a, b), faces) in &edge_faces {
        incident_edges[a].push((a, b));
        incident_edges[b].push((a, b));
        if faces.len() == 1 {
            boundary_nbrs[a].push(b);
            boundary_nbrs[b].push(a);
        }
    }
    for v in 0..nv {
        let p = q.vertices[v];
        vertices[v] = if !boundary_nbrs[v].is_empty() {
            // Corners (a single incident face) stay put.
            if boundary_nbrs[v].len() == 2 && incident_faces[v].len() > 1 {
                p * 0.75
                    + (q.vertices[boundary_nbrs[v][0]] + q.vertices[boundary_nbrs[v][1]]) * 0.125
            } else {
                p
            }
        } else {
            let n = incident_edges[v].len() as f64;
            let f = incident_faces[v]
                .iter()
                .map(|&fi| face_point[fi])
                .fold(Vec3::ZERO, |s, x| s + x)
                * (1.0 / incident_faces[v].len() as f64);
            let r = incident_edges[v]
                .iter()
                .map(|&(a, b)| (q.vertices[a] + q.vertices[b]) * 0.5)
                .fold(Vec3::ZERO, |s, x| s + x)
                * (1.0 / n);
            (f + r * 2.0 + p * (n - 3.0)) * (1.0 / n)
        };
    }
    vertices.extend(new_pts);
    // Face points occupy indices nv..nv+nf.
    let fp_index = |fi: usize| nv + fi;
    let mut quads = Vec::with_capacity(q.quads.len() * 4);
    for (fi, f) in q.quads.iter().enumerate() {
        for k in 0..4 {
            let prev = f[(k + 3) % 4];
            let cur = f[k];
            let next = f[(k + 1) % 4];
            quads.push([
                cur,
                edge_point[&edge_key(cur, next)],
                fp_index(fi),
                edge_point[&edge_key(prev, cur)],
            ]);
        }
    }
    QuadMesh { vertices, quads }
}

/// `n` levels of Catmull-Clark.
#[must_use]
pub fn catmull_clark_n(q: &QuadMesh, n: usize) -> QuadMesh {
    let mut out = q.clone();
    for _ in 0..n {
        out = catmull_clark(&out);
    }
    out
}

/// One level of sqrt(3) subdivision (Kobbelt 2000): a centroid vertex
/// per face, original interior edges flipped, old vertices smoothed by
/// the α_n stencil (boundary vertices stay).
#[must_use]
pub fn sqrt3_subdivide(m: &Mesh) -> Mesh {
    let n = m.vertices.len();
    let mut edge_faces: EdgeMap<Vec<usize>> = HashMap::new();
    for (fi, &[a, b, c]) in m.indices.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            edge_faces.entry(edge_key(u, v)).or_default().push(fi);
        }
    }
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut on_boundary = vec![false; n];
    for (&(a, b), faces) in &edge_faces {
        neighbors[a].push(b);
        neighbors[b].push(a);
        if faces.len() == 1 {
            on_boundary[a] = true;
            on_boundary[b] = true;
        }
    }
    // Smooth old vertices.
    let mut vertices: Vec<Vec3> = (0..n)
        .map(|i| {
            let v = m.vertices[i];
            let k = neighbors[i].len();
            if on_boundary[i] || k < 3 {
                v
            } else {
                let kf = k as f64;
                let alpha = (4.0 - 2.0 * (2.0 * std::f64::consts::PI / kf).cos()) / 9.0;
                let avg =
                    neighbors[i].iter().map(|&j| m.vertices[j]).fold(Vec3::ZERO, |s, p| s + p)
                        * (1.0 / kf);
                v * (1.0 - alpha) + avg * alpha
            }
        })
        .collect();
    // Face centroids.
    let centroid_index: Vec<usize> = m
        .indices
        .iter()
        .map(|&[a, b, c]| {
            let idx = vertices.len();
            vertices.push((m.vertices[a] + m.vertices[b] + m.vertices[c]) * (1.0 / 3.0));
            idx
        })
        .collect();
    // Directed original edges: face containing a->b.
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for (fi, &[a, b, c]) in m.indices.iter().enumerate() {
        directed.insert((a, b), fi);
        directed.insert((b, c), fi);
        directed.insert((c, a), fi);
    }
    let mut indices = Vec::new();
    for (&(a, b), faces) in &edge_faces {
        if faces.len() == 2 {
            // Flip: connect the two centroids across the edge.
            let f1 = directed[&(a, b)]; // face with a->b (b left of a? consistent orientation)
            let f2 = directed[&(b, a)];
            let (c1, c2) = (centroid_index[f1], centroid_index[f2]);
            indices.push([a, c2, c1]);
            indices.push([c2, b, c1]);
        } else {
            // Boundary edge: keep the corner triangle with the centroid.
            let fi = *directed.get(&(a, b)).unwrap_or_else(|| &directed[&(b, a)]);
            let (u, v) = if directed.contains_key(&(a, b)) { (a, b) } else { (b, a) };
            indices.push([u, v, centroid_index[fi]]);
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// One 1-to-4 split at edge midpoints with no repositioning: geometry
/// is unchanged (flat faces stay flat).
#[must_use]
pub fn midpoint_subdivide(m: &Mesh) -> Mesh {
    let mut vertices = m.vertices.clone();
    let mut edge_vertex: EdgeMap<usize> = HashMap::new();
    let mut midpoint = |a: usize, b: usize, vertices: &mut Vec<Vec3>| {
        *edge_vertex.entry(edge_key(a, b)).or_insert_with(|| {
            vertices.push((m.vertices[a] + m.vertices[b]) * 0.5);
            vertices.len() - 1
        })
    };
    let mut indices = Vec::with_capacity(m.indices.len() * 4);
    for &[a, b, c] in &m.indices {
        let ab = midpoint(a, b, &mut vertices);
        let bc = midpoint(b, c, &mut vertices);
        let ca = midpoint(c, a, &mut vertices);
        indices.push([a, ab, ca]);
        indices.push([b, bc, ab]);
        indices.push([c, ca, bc]);
        indices.push([ab, bc, ca]);
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

fn uniform_laplacian_step(m: &mut Mesh, adjacency: &[Vec<usize>], factor: f64) {
    let old = m.vertices.clone();
    for (i, v) in m.vertices.iter_mut().enumerate() {
        if adjacency[i].is_empty() {
            continue;
        }
        let avg = adjacency[i].iter().map(|&j| old[j]).fold(Vec3::ZERO, |s, p| s + p)
            * (1.0 / adjacency[i].len() as f64);
        *v = *v + (avg - *v) * factor;
    }
}

/// Uniform-weight Laplacian smoothing: each iteration moves every
/// vertex by `lambda` toward the average of its neighbors. Shrinks
/// closed meshes.
pub fn laplacian_smooth(m: &mut Mesh, iterations: usize, lambda: f64) {
    let adj = m.adjacency();
    for _ in 0..iterations {
        uniform_laplacian_step(m, &adj, lambda);
    }
}

/// Taubin's λ|μ smoothing (Taubin 1995): alternating positive
/// (`lambda`) and negative (`mu`, with `mu < -lambda` typically)
/// steps smooth without significant shrinkage.
pub fn taubin_smooth(m: &mut Mesh, iterations: usize, lambda: f64, mu: f64) {
    let adj = m.adjacency();
    for _ in 0..iterations {
        uniform_laplacian_step(m, &adj, lambda);
        uniform_laplacian_step(m, &adj, mu);
    }
}

/// HC-Laplacian smoothing (Vollmer, Mencl & Müller 1999): a Laplacian
/// step followed by a correction that pushes points back toward a
/// blend of their original (`alpha`) and previous positions, the
/// correction itself averaged over neighbors (`beta`).
pub fn hc_laplacian_smooth(m: &mut Mesh, iterations: usize, alpha: f64, beta: f64) {
    let adj = m.adjacency();
    let original = m.vertices.clone();
    for _ in 0..iterations {
        let q = m.vertices.clone();
        // Plain Laplacian average.
        let p: Vec<Vec3> = (0..q.len())
            .map(|i| {
                if adj[i].is_empty() {
                    q[i]
                } else {
                    adj[i].iter().map(|&j| q[j]).fold(Vec3::ZERO, |s, v| s + v)
                        * (1.0 / adj[i].len() as f64)
                }
            })
            .collect();
        let b: Vec<Vec3> = (0..q.len())
            .map(|i| p[i] - (original[i] * alpha + q[i] * (1.0 - alpha)))
            .collect();
        for i in 0..q.len() {
            let avg_b = if adj[i].is_empty() {
                b[i]
            } else {
                adj[i].iter().map(|&j| b[j]).fold(Vec3::ZERO, |s, v| s + v)
                    * (1.0 / adj[i].len() as f64)
            };
            m.vertices[i] = p[i] - (b[i] * beta + avg_b * (1.0 - beta));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generate::icosphere;

    fn euler(m: &Mesh) -> i64 {
        m.vertices.len() as i64 - m.edges().len() as i64 + m.indices.len() as i64
    }

    fn max_radius_dev(m: &Mesh, r: f64) -> f64 {
        m.vertices.iter().map(|v| (v.magnitude() - r).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn test_loop_converges_smoothly() {
        let mut m = icosphere(1.0, 1);
        assert_eq!(euler(&m), 2);
        // The mesh gets smoother every level (max angle between
        // adjacent face normals shrinks) and stays near the sphere.
        let max_dihedral = |m: &Mesh| {
            crate::mesh::analyze::dihedral_angles(m).iter().copied().fold(0.0f64, f64::max)
        };
        let mut prev = max_dihedral(&m);
        for _ in 0..3 {
            m = loop_subdivide(&m);
            assert_eq!(euler(&m), 2, "Loop preserves topology");
            let d = max_dihedral(&m);
            assert!(d < prev, "smoothness should improve: {d} vs {prev}");
            prev = d;
        }
        // Near the (slightly aspherical) limit surface: within 1% of
        // the best-fit radius.
        let mean_r: f64 =
            m.vertices.iter().map(|v| v.magnitude()).sum::<f64>() / m.vertices.len() as f64;
        assert!(max_radius_dev(&m, mean_r) < 0.01 * mean_r);
        assert!(m.volume() > 0.0);
    }

    #[test]
    fn test_loop_boundary_rules() {
        // A fan of two triangles with a boundary: subdividing keeps
        // boundary endpoints fixed only under the corner rule; the
        // middle boundary vertex moves along the boundary.
        let m = Mesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 3], [1, 2, 3]],
        )
        .unwrap();
        let s = loop_subdivide(&m);
        assert_eq!(s.indices.len(), 8);
        // All new points stay in the z = 0 plane.
        for v in &s.vertices {
            assert!(v.z.abs() < 1e-15);
        }
        // Boundary edge 0-1 gets a plain midpoint.
        assert!(s
            .vertices
            .iter()
            .any(|v| (*v - Vec3::new(0.5, 0.0, 0.0)).magnitude() < 1e-12));
    }

    #[test]
    fn test_catmull_clark_box() {
        let q = QuadMesh::from_box(Vec3::new(1.0, 1.0, 1.0));
        let tri = q.to_triangles();
        assert_eq!(euler(&tri), 2);
        let mut cc = q.clone();
        for level in 0..3 {
            cc = catmull_clark(&cc);
            let m = cc.to_triangles();
            assert_eq!(euler(&m), 2, "Catmull-Clark preserves Euler at level {level}");
            assert!(m.volume() > 0.0);
        }
        assert_eq!(cc.quads.len(), 6 * 4 * 4 * 4);
        // The limit surface of a cube is rounded: strictly inside the
        // cube, strictly containing a concentric sphere of radius 0.5.
        for v in &cc.vertices {
            assert!(v.x.abs() <= 1.0 + 1e-12 && v.y.abs() <= 1.0 + 1e-12 && v.z.abs() <= 1.0 + 1e-12);
        }
        let m = catmull_clark_n(&QuadMesh::from_box(Vec3::new(1.0, 1.0, 1.0)), 3).to_triangles();
        assert!(m.volume() < 8.0 && m.volume() > 2.0);
    }

    #[test]
    fn test_catmull_clark_grid_boundary() {
        let g = QuadMesh::from_grid(2.0, 2.0, 2, 2);
        let cc = catmull_clark(&g);
        assert_eq!(cc.quads.len(), 16);
        // Everything stays planar.
        for v in &cc.vertices {
            assert!(v.y.abs() < 1e-15);
        }
        // Grid corners are boundary corners (valence-2 boundary): kept.
        assert!(cc
            .vertices
            .iter()
            .any(|v| (*v - Vec3::new(-1.0, 0.0, -1.0)).magnitude() < 1e-12));
    }

    #[test]
    fn test_sqrt3_topology_and_convergence() {
        let m = icosphere(1.0, 1);
        let s = sqrt3_subdivide(&m);
        assert_eq!(euler(&s), 2);
        assert_eq!(s.indices.len(), m.indices.len() * 3);
        assert!(s.volume() > 0.0, "sqrt3 keeps orientation");
        let s2 = sqrt3_subdivide(&s);
        assert_eq!(euler(&s2), 2);
        assert_eq!(s2.indices.len(), m.indices.len() * 9);
        let mean_r: f64 =
            s2.vertices.iter().map(|v| v.magnitude()).sum::<f64>() / s2.vertices.len() as f64;
        assert!(max_radius_dev(&s2, mean_r) < 0.02 * mean_r, "stays near a sphere");
        // Smoother than the input.
        let max_dihedral = |m: &Mesh| {
            crate::mesh::analyze::dihedral_angles(m).iter().copied().fold(0.0f64, f64::max)
        };
        assert!(max_dihedral(&s2) < max_dihedral(&m));
    }

    #[test]
    fn test_midpoint_preserves_geometry() {
        let m = icosphere(1.0, 1);
        let s = midpoint_subdivide(&m);
        assert_eq!(s.indices.len(), m.indices.len() * 4);
        assert_eq!(euler(&s), 2);
        assert!((s.volume() - m.volume()).abs() < 1e-12);
        assert!((s.surface_area() - m.surface_area()).abs() < 1e-12);
    }

    #[test]
    fn test_taubin_resists_shrinkage() {
        let base = icosphere(1.0, 3);
        let v0 = base.volume();

        let mut lap = base.clone();
        laplacian_smooth(&mut lap, 10, 0.5);
        let shrink = (v0 - lap.volume()) / v0;
        assert!(shrink > 0.01, "Laplacian should visibly shrink (got {shrink})");

        let mut tau = base.clone();
        taubin_smooth(&mut tau, 10, 0.33, -0.34);
        let change = (tau.volume() - v0).abs() / v0;
        assert!(change < 0.01, "Taubin volume change {change} should stay under 1%");

        let mut hc = base.clone();
        hc_laplacian_smooth(&mut hc, 10, 0.1, 0.6);
        let change = (hc.volume() - v0).abs() / v0;
        assert!(change < 0.02, "HC-Laplacian volume change {change} too large");
    }
}
