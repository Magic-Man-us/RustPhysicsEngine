//! Indexed triangle meshes: construction, mass properties, cleanup,
//! spatial queries, and OBJ/STL interchange.

pub mod generate;
pub mod isosurface;

use crate::error::GeomError;
use crate::linalg::Mat3;
use crate::math::{Vec2, Vec3};
use crate::monte_carlo::Rng;
use crate::quaternion::Quaternion;
use crate::spatial::intersect::{ray_triangle, RayHit};
use crate::spatial::primitives::{Aabb, Ray, Sphere, Triangle};
use crate::spatial::{Bvh, Mat4};
use std::collections::HashMap;

/// Indexed triangle mesh with optional per-vertex normals and UVs.
///
/// `normals` and `uvs`, when present, are parallel to `vertices`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[usize; 3]>,
    pub normals: Option<Vec<Vec3>>,
    pub uvs: Option<Vec<Vec2>>,
}

impl Mesh {
    /// Builds a mesh, validating that every index is in range.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] when a face references a
    /// vertex index `>= vertices.len()`.
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[usize; 3]>) -> Result<Self, GeomError> {
        let n = vertices.len();
        for f in &indices {
            if f[0] >= n || f[1] >= n || f[2] >= n {
                return Err(GeomError::InvalidArgument("face index out of range"));
            }
        }
        Ok(Self { vertices, indices, normals: None, uvs: None })
    }

    /// The i-th face as a [`Triangle`].
    ///
    /// # Panics
    /// Panics when `i >= self.indices.len()`.
    #[must_use]
    pub fn triangle(&self, i: usize) -> Triangle {
        let [a, b, c] = self.indices[i];
        Triangle { a: self.vertices[a], b: self.vertices[b], c: self.vertices[c] }
    }

    /// Iterator over all faces as triangles.
    pub fn triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
        self.indices.iter().map(|&[a, b, c]| Triangle {
            a: self.vertices[a],
            b: self.vertices[b],
            c: self.vertices[c],
        })
    }

    /// All faces collected as triangles.
    #[must_use]
    pub fn to_triangles(&self) -> Vec<Triangle> {
        self.triangles().collect()
    }

    /// Unit normal of every face (zero vector for degenerate faces).
    #[must_use]
    pub fn face_normals(&self) -> Vec<Vec3> {
        self.triangles().map(|t| t.normal()).collect()
    }

    /// Computes area-weighted per-vertex normals and stores them in
    /// `self.normals`.
    ///
    /// Each face contributes its (unnormalized) cross product, whose
    /// magnitude is twice the face area, so large faces dominate.
    pub fn compute_vertex_normals(&mut self) {
        let mut acc = vec![Vec3::ZERO; self.vertices.len()];
        for &[a, b, c] in &self.indices {
            let n = (self.vertices[b] - self.vertices[a])
                .cross(&(self.vertices[c] - self.vertices[a]));
            acc[a] = acc[a] + n;
            acc[b] = acc[b] + n;
            acc[c] = acc[c] + n;
        }
        for n in &mut acc {
            let m = n.magnitude();
            if m > 0.0 {
                *n = *n * (1.0 / m);
            }
        }
        self.normals = Some(acc);
    }

    /// Total surface area (sum of face areas).
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        self.triangles().map(|t| t.area()).sum()
    }

    /// Signed enclosed volume via the divergence theorem:
    /// V = Σ aᵢ · (bᵢ × cᵢ) / 6. Positive for a closed mesh with
    /// outward-facing (counterclockwise) triangles.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.triangles()
            .map(|t| t.a.dot(&t.b.cross(&t.c)) / 6.0)
            .sum()
    }

    /// Centroid of the enclosed volume (center of mass at uniform
    /// density), from the signed tetrahedron decomposition against the
    /// origin.
    ///
    /// # Panics
    /// Panics when the signed volume is zero.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let mut vol = 0.0;
        let mut c = Vec3::ZERO;
        for t in self.triangles() {
            let v6 = t.a.dot(&t.b.cross(&t.c));
            vol += v6 / 6.0;
            // ∫ x dV over the tet (0, a, b, c) equals V (a + b + c) / 4.
            c = c + (t.a + t.b + t.c) * (v6 / 24.0);
        }
        assert!(vol != 0.0, "centroid requires nonzero enclosed volume");
        c * (1.0 / vol)
    }

    /// Area-weighted centroid of the surface (center of mass of a thin
    /// shell of uniform surface density).
    ///
    /// # Panics
    /// Panics when the total surface area is zero.
    #[must_use]
    pub fn center_of_mass_surface(&self) -> Vec3 {
        let mut area = 0.0;
        let mut c = Vec3::ZERO;
        for t in self.triangles() {
            let a = t.area();
            area += a;
            c = c + t.centroid() * a;
        }
        assert!(area > 0.0, "center_of_mass_surface requires nonzero area");
        c * (1.0 / area)
    }

    /// Inertia tensor about the center of mass of the enclosed solid at
    /// the given uniform density, by signed tetrahedron decomposition
    /// (equivalent to Mirtich's polyhedral mass-property integrals).
    ///
    /// Each face forms the tet (0, a, b, c); its second-moment
    /// (covariance) integral is det(J) · J C J^T where J = [a b c] and
    /// C is the canonical tetrahedron covariance (1/60 diagonal, 1/120
    /// off-diagonal). Source: Mirtich, "Fast and Accurate Computation
    /// of Polyhedral Mass Properties", JGT 1996.
    ///
    /// # Panics
    /// Panics when the signed volume is zero.
    #[must_use]
    pub fn inertia_tensor(&self, density: f64) -> Mat3 {
        let mut cov = [[0.0f64; 3]; 3];
        let mut vol = 0.0;
        let mut com = Vec3::ZERO;
        for t in self.triangles() {
            let det = t.a.dot(&t.b.cross(&t.c));
            vol += det / 6.0;
            com = com + (t.a + t.b + t.c) * (det / 24.0);
            let j = [t.a, t.b, t.c];
            let col = |v: &Vec3, k: usize| match k {
                0 => v.x,
                1 => v.y,
                _ => v.z,
            };
            // det · J C Jᵀ with C = 1/60 on the diagonal, 1/120 off it:
            // (J C Jᵀ)_{rs} = Σ_{p,q} J_{rp} C_{pq} J_{sq}.
            for (r, row) in cov.iter_mut().enumerate() {
                for (s, entry) in row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (p, jp) in j.iter().enumerate() {
                        for (q, jq) in j.iter().enumerate() {
                            let cpq = if p == q { 1.0 / 60.0 } else { 1.0 / 120.0 };
                            sum += col(jp, r) * cpq * col(jq, s);
                        }
                    }
                    *entry += det * sum;
                }
            }
        }
        assert!(vol != 0.0, "inertia_tensor requires nonzero enclosed volume");
        let com = com * (1.0 / vol);
        let mass = density * vol;
        // Shift the second moments to the center of mass, then convert
        // the covariance to the inertia tensor: I = tr(C) 1 − C.
        let cvec = [com.x, com.y, com.z];
        let mut c = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for s in 0..3 {
                c[r][s] = density * cov[r][s] - mass * cvec[r] * cvec[s];
            }
        }
        let tr = c[0][0] + c[1][1] + c[2][2];
        let mut i = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for s in 0..3 {
                i[r][s] = if r == s { tr - c[r][s] } else { -c[r][s] };
            }
        }
        Mat3 { data: i }
    }

    /// Principal moments of inertia (descending) and the rotation whose
    /// columns are the principal axes.
    ///
    /// # Panics
    /// Panics when the signed volume is zero.
    #[must_use]
    pub fn principal_inertia(&self, density: f64) -> ([f64; 3], Mat3) {
        self.inertia_tensor(density).principal_axes_3x3()
    }

    /// Axis-aligned bounding box of all vertices.
    ///
    /// # Panics
    /// Panics when the mesh has no vertices.
    #[must_use]
    pub fn bounding_box(&self) -> Aabb {
        assert!(!self.vertices.is_empty(), "bounding_box requires vertices");
        Aabb::from_points(&self.vertices)
    }

    /// Approximate minimal bounding sphere by Ritter's two-pass
    /// algorithm (at most ~5% larger than optimal).
    ///
    /// # Panics
    /// Panics when the mesh has no vertices.
    #[must_use]
    pub fn bounding_sphere(&self) -> Sphere {
        assert!(!self.vertices.is_empty(), "bounding_sphere requires vertices");
        let x = self.vertices[0];
        let farthest = |from: Vec3| {
            let mut best = from;
            let mut d2 = -1.0;
            for &v in &self.vertices {
                let d = (v - from).magnitude_squared();
                if d > d2 {
                    d2 = d;
                    best = v;
                }
            }
            best
        };
        let y = farthest(x);
        let z = farthest(y);
        let mut center = (y + z) * 0.5;
        let mut radius = y.distance_to(&z) * 0.5;
        for &v in &self.vertices {
            let d = v.distance_to(&center);
            if d > radius {
                // Grow to just enclose v, keeping the far side fixed.
                let new_r = 0.5 * (radius + d);
                center = center + (v - center) * ((d - new_r) / d);
                radius = new_r;
            }
        }
        Sphere { center, radius }
    }

    /// Applies a general 4x4 transform to vertices; normals are mapped
    /// by the normal matrix (inverse transpose) and renormalized.
    ///
    /// # Panics
    /// Panics (inside [`Mat4::normal_matrix`]) when the mesh has
    /// normals and the linear part of `m` is singular.
    pub fn transform(&mut self, m: &Mat4) {
        for v in &mut self.vertices {
            *v = m.transform_point(*v);
        }
        if let Some(normals) = &mut self.normals {
            let nm = m.normal_matrix();
            for n in normals {
                *n = nm.mul_vec(*n).normalized();
            }
        }
    }

    /// Translates every vertex by `offset`.
    pub fn translate(&mut self, offset: Vec3) {
        for v in &mut self.vertices {
            *v = *v + offset;
        }
    }

    /// Uniformly scales every vertex about the origin.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.vertices {
            *v = *v * factor;
        }
    }

    /// Rotates vertices (and normals) about the origin.
    pub fn rotate(&mut self, q: &Quaternion) {
        for v in &mut self.vertices {
            *v = q.rotate_vec(*v);
        }
        if let Some(normals) = &mut self.normals {
            for n in normals {
                *n = q.rotate_vec(*n);
            }
        }
    }

    /// Appends another mesh. Optional attributes are kept only when
    /// both meshes carry them.
    pub fn merge(&mut self, other: &Mesh) {
        let base = self.vertices.len();
        self.vertices.extend_from_slice(&other.vertices);
        self.indices
            .extend(other.indices.iter().map(|&[a, b, c]| [a + base, b + base, c + base]));
        self.normals = match (self.normals.take(), &other.normals) {
            (Some(mut a), Some(b)) => {
                a.extend_from_slice(b);
                Some(a)
            }
            _ => None,
        };
        self.uvs = match (self.uvs.take(), &other.uvs) {
            (Some(mut a), Some(b)) => {
                a.extend_from_slice(b);
                Some(a)
            }
            _ => None,
        };
    }

    /// Reverses the winding of every face and negates stored normals.
    pub fn flip_normals(&mut self) {
        for f in &mut self.indices {
            f.swap(1, 2);
        }
        if let Some(normals) = &mut self.normals {
            for n in normals {
                *n = -*n;
            }
        }
    }

    /// Merges vertices closer than `tol` (grid hashing with neighbor
    /// search, so any pair within `tol` of a common representative
    /// merges). Faces left with a repeated index are removed; stored
    /// normals and UVs are dropped. Returns the number of vertices
    /// removed.
    ///
    /// # Panics
    /// Panics unless `tol > 0` and finite.
    pub fn weld_vertices(&mut self, tol: f64) -> usize {
        assert!(tol > 0.0 && tol.is_finite(), "weld tolerance must be positive");
        let key = |v: &Vec3| {
            (
                (v.x / tol).floor() as i64,
                (v.y / tol).floor() as i64,
                (v.z / tol).floor() as i64,
            )
        };
        let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        let mut remap = vec![usize::MAX; self.vertices.len()];
        let mut kept: Vec<Vec3> = Vec::new();
        for (i, v) in self.vertices.iter().enumerate() {
            let (kx, ky, kz) = key(v);
            let mut found = None;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(list) = cells.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &j in list {
                                if kept[j].distance_to(v) <= tol {
                                    found = Some(j);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
            remap[i] = match found {
                Some(j) => j,
                None => {
                    let j = kept.len();
                    kept.push(*v);
                    cells.entry((kx, ky, kz)).or_default().push(j);
                    j
                }
            };
        }
        let removed = self.vertices.len() - kept.len();
        self.vertices = kept;
        self.indices = self
            .indices
            .iter()
            .map(|&[a, b, c]| [remap[a], remap[b], remap[c]])
            .filter(|&[a, b, c]| a != b && b != c && a != c)
            .collect();
        self.normals = None;
        self.uvs = None;
        removed
    }

    /// Removes vertices referenced by no face, compacting attributes.
    pub fn remove_unused_vertices(&mut self) {
        let mut used = vec![false; self.vertices.len()];
        for &[a, b, c] in &self.indices {
            used[a] = true;
            used[b] = true;
            used[c] = true;
        }
        let mut remap = vec![usize::MAX; self.vertices.len()];
        let mut next = 0usize;
        for (i, &u) in used.iter().enumerate() {
            if u {
                remap[i] = next;
                next += 1;
            }
        }
        let old = std::mem::take(&mut self.vertices);
        self.vertices =
            old.into_iter().enumerate().filter(|&(i, _)| used[i]).map(|(_, v)| v).collect();
        if let Some(normals) = self.normals.take() {
            self.normals = Some(
                normals.into_iter().enumerate().filter(|&(i, _)| used[i]).map(|(_, n)| n).collect(),
            );
        }
        if let Some(uvs) = self.uvs.take() {
            self.uvs = Some(
                uvs.into_iter().enumerate().filter(|&(i, _)| used[i]).map(|(_, t)| t).collect(),
            );
        }
        for f in &mut self.indices {
            *f = [remap[f[0]], remap[f[1]], remap[f[2]]];
        }
    }

    /// Removes faces with area below `area_tol` or with repeated
    /// indices; returns how many were removed.
    pub fn remove_degenerate_triangles(&mut self, area_tol: f64) -> usize {
        let before = self.indices.len();
        let vertices = &self.vertices;
        self.indices.retain(|&[a, b, c]| {
            if a == b || b == c || a == c {
                return false;
            }
            let t = Triangle { a: vertices[a], b: vertices[b], c: vertices[c] };
            t.area() > area_tol
        });
        before - self.indices.len()
    }

    /// Unique undirected edges as sorted `(min, max)` index pairs,
    /// lexicographically ordered.
    #[must_use]
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut set: Vec<(usize, usize)> = self
            .indices
            .iter()
            .flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)])
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        set.sort_unstable();
        set.dedup();
        set
    }

    /// Vertex-to-neighbor-vertices adjacency (each list sorted,
    /// deduplicated).
    #[must_use]
    pub fn adjacency(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.vertices.len()];
        for (a, b) in self.edges() {
            adj[a].push(b);
            adj[b].push(a);
        }
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
        }
        adj
    }

    /// For each face, the neighboring face across each of its edges
    /// `(v0,v1), (v1,v2), (v2,v0)`, or `None` on a boundary. When an
    /// edge is shared by more than two faces, an arbitrary neighbor is
    /// reported.
    #[must_use]
    pub fn face_adjacency(&self) -> Vec<[Option<usize>; 3]> {
        let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (fi, &[a, b, c]) in self.indices.iter().enumerate() {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                by_edge.entry((u.min(v), u.max(v))).or_default().push(fi);
            }
        }
        self.indices
            .iter()
            .enumerate()
            .map(|(fi, &[a, b, c])| {
                let neighbor = |u: usize, v: usize| {
                    by_edge[&(u.min(v), u.max(v))].iter().copied().find(|&g| g != fi)
                };
                [neighbor(a, b), neighbor(b, c), neighbor(c, a)]
            })
            .collect()
    }

    /// Builds a BVH over the faces (indices refer to face order).
    ///
    /// # Panics
    /// Panics when the mesh has no faces.
    #[must_use]
    pub fn build_bvh(&self) -> Bvh {
        Bvh::build_triangles(&self.to_triangles())
    }

    /// Nearest ray hit as `(face index, hit)`. Pass a BVH built by
    /// [`Mesh::build_bvh`] to accelerate; `None` falls back to brute
    /// force.
    #[must_use]
    pub fn raycast(&self, r: &Ray, bvh: Option<&Bvh>) -> Option<(usize, RayHit)> {
        match bvh {
            Some(bvh) => bvh.closest_hit(r, &self.to_triangles()),
            None => {
                let mut best: Option<(usize, RayHit)> = None;
                for (i, t) in self.triangles().enumerate() {
                    if let Some((hit, _)) = ray_triangle(r, &t, false) {
                        if best.as_ref().is_none_or(|(_, b)| hit.t < b.t) {
                            best = Some((i, hit));
                        }
                    }
                }
                best
            }
        }
    }

    /// Draws `n` points uniformly over the surface: faces are chosen
    /// with probability proportional to area, positions by the
    /// square-root barycentric warp.
    ///
    /// # Panics
    /// Panics when the total surface area is zero.
    #[must_use]
    pub fn sample_surface(&self, n: usize, rng: &mut Rng) -> Vec<Vec3> {
        let mut cdf = Vec::with_capacity(self.indices.len());
        let mut total = 0.0;
        for t in self.triangles() {
            total += t.area();
            cdf.push(total);
        }
        assert!(total > 0.0, "sample_surface requires nonzero area");
        (0..n)
            .map(|_| {
                let u = rng.next_f64() * total;
                let fi = cdf.partition_point(|&a| a < u).min(cdf.len() - 1);
                let t = self.triangle(fi);
                let s = rng.next_f64().sqrt();
                let r2 = rng.next_f64();
                t.a * (1.0 - s) + t.b * (s * (1.0 - r2)) + t.c * (s * r2)
            })
            .collect()
    }

    /// Serializes to Wavefront OBJ (1-indexed; `vn`/`vt` written when
    /// present, referenced with the same index as the position).
    #[must_use]
    pub fn to_obj(&self) -> String {
        let mut s = String::new();
        for v in &self.vertices {
            s.push_str(&format!("v {} {} {}\n", v.x, v.y, v.z));
        }
        if let Some(uvs) = &self.uvs {
            for t in uvs {
                s.push_str(&format!("vt {} {}\n", t.x, t.y));
            }
        }
        if let Some(normals) = &self.normals {
            for n in normals {
                s.push_str(&format!("vn {} {} {}\n", n.x, n.y, n.z));
            }
        }
        for &[a, b, c] in &self.indices {
            let idx = |i: usize| match (&self.uvs, &self.normals) {
                (Some(_), Some(_)) => format!("{0}/{0}/{0}", i + 1),
                (Some(_), None) => format!("{0}/{0}", i + 1),
                (None, Some(_)) => format!("{0}//{0}", i + 1),
                (None, None) => format!("{}", i + 1),
            };
            s.push_str(&format!("f {} {} {}\n", idx(a), idx(b), idx(c)));
        }
        s
    }

    /// Parses Wavefront OBJ. Faces with more than three corners are
    /// fan-triangulated. Normals and UVs are kept only when every face
    /// corner references the attribute with the same index as its
    /// position and the counts match; otherwise they are dropped.
    /// Negative (relative) indices are resolved against the counts seen
    /// so far.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] on malformed numbers or
    /// out-of-range indices.
    pub fn from_obj(s: &str) -> Result<Self, GeomError> {
        let bad = GeomError::InvalidArgument("malformed OBJ");
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut uvs: Vec<Vec2> = Vec::new();
        let mut indices: Vec<[usize; 3]> = Vec::new();
        let mut attrs_consistent = true;
        for line in s.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let tag = parts.next().unwrap_or("");
            let read = |p: Option<&str>| -> Result<f64, GeomError> {
                p.ok_or(bad.clone())?.parse::<f64>().map_err(|_| bad.clone())
            };
            match tag {
                "v" => {
                    let (x, y, z) =
                        (read(parts.next())?, read(parts.next())?, read(parts.next())?);
                    vertices.push(Vec3::new(x, y, z));
                }
                "vn" => {
                    let (x, y, z) =
                        (read(parts.next())?, read(parts.next())?, read(parts.next())?);
                    normals.push(Vec3::new(x, y, z));
                }
                "vt" => {
                    let (x, y) = (read(parts.next())?, read(parts.next())?);
                    uvs.push(Vec2::new(x, y));
                }
                "f" => {
                    let mut corners: Vec<usize> = Vec::new();
                    for corner in parts {
                        let mut fields = corner.split('/');
                        let vi = fields.next().ok_or(bad.clone())?;
                        let vi: i64 = vi.parse().map_err(|_| bad.clone())?;
                        let resolve = |i: i64, len: usize| -> Result<usize, GeomError> {
                            let idx = if i > 0 {
                                i - 1
                            } else if i < 0 {
                                len as i64 + i
                            } else {
                                return Err(bad.clone());
                            };
                            if idx < 0 || idx as usize >= len {
                                return Err(GeomError::InvalidArgument(
                                    "OBJ face index out of range",
                                ));
                            }
                            Ok(idx as usize)
                        };
                        let v = resolve(vi, vertices.len())?;
                        // vt then vn; empty field means absent.
                        for (field, len) in
                            [(fields.next(), uvs.len()), (fields.next(), normals.len())]
                        {
                            match field {
                                None | Some("") => {}
                                Some(t) => {
                                    let i: i64 = t.parse().map_err(|_| bad.clone())?;
                                    if resolve(i, len)? != v {
                                        attrs_consistent = false;
                                    }
                                }
                            }
                        }
                        corners.push(v);
                    }
                    if corners.len() < 3 {
                        return Err(bad.clone());
                    }
                    for k in 1..corners.len() - 1 {
                        indices.push([corners[0], corners[k], corners[k + 1]]);
                    }
                }
                _ => {}
            }
        }
        let mut mesh = Mesh::new(vertices, indices)?;
        if attrs_consistent && normals.len() == mesh.vertices.len() {
            mesh.normals = Some(normals);
        }
        if attrs_consistent && uvs.len() == mesh.vertices.len() {
            mesh.uvs = Some(uvs);
        }
        Ok(mesh)
    }

    /// Serializes to ASCII STL (facet normals recomputed from
    /// geometry).
    #[must_use]
    pub fn to_stl_ascii(&self) -> String {
        let mut s = String::from("solid mesh\n");
        for t in self.triangles() {
            let n = t.normal();
            s.push_str(&format!("facet normal {} {} {}\n", n.x, n.y, n.z));
            s.push_str("  outer loop\n");
            for v in [t.a, t.b, t.c] {
                s.push_str(&format!("    vertex {} {} {}\n", v.x, v.y, v.z));
            }
            s.push_str("  endloop\nendfacet\n");
        }
        s.push_str("endsolid mesh\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::generate::{box_mesh, icosphere};
    use super::*;

    fn tetra() -> Mesh {
        // Regular-ish positively oriented tetrahedron around the origin.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        // Outward-facing windings.
        let f = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        Mesh::new(v, f).unwrap()
    }

    #[test]
    fn test_new_validates_indices() {
        let v = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        assert!(Mesh::new(v.clone(), vec![[0, 1, 2]]).is_err());
        assert!(Mesh::new(v, vec![[0, 1, 1]]).is_ok());
    }

    #[test]
    fn test_tetra_mass_properties() {
        let m = tetra();
        assert!((m.volume() - 1.0 / 6.0).abs() < 1e-12);
        let area =
            0.5 * 3.0 + (Vec3::new(-1.0, 1.0, 0.0).cross(&Vec3::new(-1.0, 0.0, 1.0))).magnitude()
                / 2.0;
        assert!((m.surface_area() - area).abs() < 1e-12);
        let c = m.centroid();
        assert!((c - Vec3::new(0.25, 0.25, 0.25)).magnitude() < 1e-12);
    }

    #[test]
    fn test_box_inertia_matches_closed_form() {
        let half = Vec3::new(0.5, 1.0, 1.5);
        let m = box_mesh(half);
        let rho = 2.5;
        let mass = rho * m.volume();
        let i = m.inertia_tensor(rho);
        let expect = |a: f64, b: f64| mass / 3.0 * (a * a + b * b);
        assert!((i.data[0][0] - expect(half.y, half.z)).abs() < 1e-9);
        assert!((i.data[1][1] - expect(half.x, half.z)).abs() < 1e-9);
        assert!((i.data[2][2] - expect(half.x, half.y)).abs() < 1e-9);
        for r in 0..3 {
            for s in 0..3 {
                if r != s {
                    assert!(i.data[r][s].abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn test_inertia_translation_invariant_about_com() {
        let mut m = icosphere(1.0, 2);
        let i0 = m.inertia_tensor(1.0);
        m.translate(Vec3::new(3.0, -2.0, 5.0));
        let i1 = m.inertia_tensor(1.0);
        for r in 0..3 {
            for s in 0..3 {
                assert!((i0.data[r][s] - i1.data[r][s]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_bounding_volumes() {
        let m = icosphere(2.0, 1);
        let bb = m.bounding_box();
        assert!(bb.min.x >= -2.0 - 1e-12 && bb.max.x <= 2.0 + 1e-12);
        let bs = m.bounding_sphere();
        for v in &m.vertices {
            assert!(v.distance_to(&bs.center) <= bs.radius + 1e-9);
        }
        assert!(bs.radius < 2.2);
    }

    #[test]
    fn test_vertex_normals_outward_on_sphere() {
        let mut m = icosphere(1.0, 2);
        m.compute_vertex_normals();
        let normals = m.normals.as_ref().unwrap();
        for (v, n) in m.vertices.iter().zip(normals) {
            assert!(v.normalized().dot(n) > 0.9);
            assert!((n.magnitude() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_flip_normals_negates_volume() {
        let mut m = tetra();
        let v = m.volume();
        m.flip_normals();
        assert!((m.volume() + v).abs() < 1e-15);
    }

    #[test]
    fn test_merge_and_remove_unused() {
        let mut a = tetra();
        let b = tetra();
        a.merge(&b);
        assert_eq!(a.vertices.len(), 8);
        assert_eq!(a.indices.len(), 8);
        // Orphan a vertex by dropping the faces of the second copy.
        a.indices.truncate(4);
        a.remove_unused_vertices();
        assert_eq!(a.vertices.len(), 4);
        assert!((a.volume() - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_weld_and_degenerate_removal() {
        let mut m = tetra();
        let shifted = tetra();
        m.merge(&shifted);
        // The two copies coincide; welding halves the vertex count and
        // duplicate faces remain (not degenerate).
        assert_eq!(m.weld_vertices(1e-9), 4);
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.indices.len(), 8);
        m.indices.push([0, 0, 1]);
        m.indices.push([0, 1, 1]);
        assert_eq!(m.remove_degenerate_triangles(0.0), 2);
    }

    #[test]
    fn test_edges_adjacency_face_adjacency() {
        let m = tetra();
        let e = m.edges();
        assert_eq!(e.len(), 6);
        let adj = m.adjacency();
        for list in &adj {
            assert_eq!(list.len(), 3);
        }
        let fa = m.face_adjacency();
        for row in &fa {
            for n in row {
                assert!(n.is_some(), "closed tetra has no boundary edges");
            }
        }
    }

    #[test]
    fn test_raycast_brute_and_bvh_agree() {
        let m = icosphere(1.0, 2);
        let bvh = m.build_bvh();
        let r = Ray::new(Vec3::new(3.0, 0.1, -0.2), Vec3::new(-1.0, 0.0, 0.0));
        let (fa, ha) = m.raycast(&r, None).unwrap();
        let (fb, hb) = m.raycast(&r, Some(&bvh)).unwrap();
        assert_eq!(fa, fb);
        assert!((ha.t - hb.t).abs() < 1e-12);
        assert!((ha.point.magnitude() - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_sample_surface_on_sphere() {
        let m = icosphere(1.0, 3);
        let mut rng = Rng::new(42);
        let pts = m.sample_surface(500, &mut rng);
        assert_eq!(pts.len(), 500);
        let mut mean = Vec3::ZERO;
        for p in &pts {
            assert!((p.magnitude() - 1.0).abs() < 0.05);
            mean = mean + *p;
        }
        // Uniform sphere samples average near the center.
        assert!((mean * (1.0 / 500.0)).magnitude() < 0.12);
    }

    #[test]
    fn test_obj_roundtrip_exact() {
        let mut m = icosphere(1.0, 1);
        let back = Mesh::from_obj(&m.to_obj()).unwrap();
        assert_eq!(m, back);
        m.compute_vertex_normals();
        let back = Mesh::from_obj(&m.to_obj()).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn test_obj_parses_quads_and_relative_indices() {
        let src = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nf -4 -3 -2\n";
        let m = Mesh::from_obj(src).unwrap();
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.indices.len(), 3);
        assert_eq!(m.indices[0], [0, 1, 2]);
        assert_eq!(m.indices[1], [0, 2, 3]);
        assert_eq!(m.indices[2], [0, 1, 2]);
        assert!(Mesh::from_obj("f 1 2 5\nv 0 0 0").is_err());
        assert!(Mesh::from_obj("v 0 0\n").is_err());
    }

    #[test]
    fn test_stl_ascii_structure() {
        let m = tetra();
        let s = m.to_stl_ascii();
        assert!(s.starts_with("solid"));
        assert!(s.ends_with("endsolid mesh\n"));
        assert_eq!(s.matches("facet normal").count(), 4);
        assert_eq!(s.matches("vertex").count(), 12);
    }

    #[test]
    fn test_transform_matches_translate_scale_rotate() {
        let mut a = tetra();
        let mut b = tetra();
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.7);
        a.rotate(&q);
        a.scale(2.0);
        a.translate(Vec3::new(1.0, 2.0, 3.0));
        let m = Mat4::from_trs(Vec3::new(1.0, 2.0, 3.0), &q, Vec3::new(2.0, 2.0, 2.0));
        b.transform(&m);
        for (u, v) in a.vertices.iter().zip(&b.vertices) {
            assert!(u.distance_to(v) < 1e-12);
        }
    }
}
