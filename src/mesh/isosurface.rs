//! Isosurface and isocontour extraction from sampled scalar fields:
//! marching squares/cubes/tetrahedra, surface nets, dual contouring,
//! and metaballs.
//!
//! Convention: a sample is "inside" when its value is below the iso
//! level (matching signed distance fields, negative inside). Output
//! triangles are wound counterclockwise seen from the outside (normals
//! point toward values above the iso level); 2-D contours keep the
//! inside region on their left, so they run counterclockwise around
//! regions below the iso level.

use crate::math::{Vec2, Vec3};
use crate::mesh::Mesh;
use crate::spatial::primitives::{Aabb, Rect, Segment2};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Scalar samples on a regular 2-D grid (`width` x `height` samples,
/// x-fastest layout: `data[j * width + i]`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField2 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
    pub bounds: Rect,
}

impl ScalarField2 {
    /// Samples `f` on a `width` x `height` grid spanning `bounds`.
    ///
    /// # Panics
    /// Panics unless `width >= 2` and `height >= 2`.
    #[must_use]
    pub fn from_fn(bounds: Rect, width: usize, height: usize, f: &dyn Fn(Vec2) -> f64) -> Self {
        assert!(width >= 2 && height >= 2, "ScalarField2 requires >= 2 samples per axis");
        let mut data = Vec::with_capacity(width * height);
        for j in 0..height {
            for i in 0..width {
                data.push(f(Self::grid_position(bounds, width, height, i, j)));
            }
        }
        Self { width, height, data, bounds }
    }

    fn grid_position(bounds: Rect, width: usize, height: usize, i: usize, j: usize) -> Vec2 {
        let e = bounds.max - bounds.min;
        Vec2::new(
            bounds.min.x + e.x * i as f64 / (width - 1) as f64,
            bounds.min.y + e.y * j as f64 / (height - 1) as f64,
        )
    }

    /// Sample value at grid coordinates.
    ///
    /// # Panics
    /// Panics when out of range.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.width && j < self.height, "ScalarField2 index out of range");
        self.data[j * self.width + i]
    }

    /// World position of the sample at grid coordinates.
    ///
    /// # Panics
    /// Panics when out of range.
    #[must_use]
    pub fn position(&self, i: usize, j: usize) -> Vec2 {
        assert!(i < self.width && j < self.height, "ScalarField2 index out of range");
        Self::grid_position(self.bounds, self.width, self.height, i, j)
    }
}

/// Scalar samples on a regular 3-D grid (`nx` x `ny` x `nz` samples,
/// x-fastest layout: `data[(k * ny + j) * nx + i]`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField3 {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub data: Vec<f64>,
    pub bounds: Aabb,
}

impl ScalarField3 {
    /// Samples `f` on an `nx` x `ny` x `nz` grid spanning `bounds`.
    ///
    /// # Panics
    /// Panics unless every axis has at least 2 samples.
    #[must_use]
    pub fn from_fn(
        bounds: Aabb,
        nx: usize,
        ny: usize,
        nz: usize,
        f: &dyn Fn(Vec3) -> f64,
    ) -> Self {
        assert!(nx >= 2 && ny >= 2 && nz >= 2, "ScalarField3 requires >= 2 samples per axis");
        let mut data = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    data.push(f(Self::grid_position(&bounds, nx, ny, nz, i, j, k)));
                }
            }
        }
        Self { nx, ny, nz, data, bounds }
    }

    /// Samples a signed distance function (alias of [`Self::from_fn`],
    /// spelled out because SDF sampling is the common case).
    #[must_use]
    pub fn from_sdf(
        bounds: Aabb,
        nx: usize,
        ny: usize,
        nz: usize,
        sdf: &dyn Fn(Vec3) -> f64,
    ) -> Self {
        Self::from_fn(bounds, nx, ny, nz, sdf)
    }

    fn grid_position(
        bounds: &Aabb,
        nx: usize,
        ny: usize,
        nz: usize,
        i: usize,
        j: usize,
        k: usize,
    ) -> Vec3 {
        let e = bounds.max - bounds.min;
        Vec3::new(
            bounds.min.x + e.x * i as f64 / (nx - 1) as f64,
            bounds.min.y + e.y * j as f64 / (ny - 1) as f64,
            bounds.min.z + e.z * k as f64 / (nz - 1) as f64,
        )
    }

    /// Sample value at grid coordinates.
    ///
    /// # Panics
    /// Panics when out of range.
    #[must_use]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        assert!(i < self.nx && j < self.ny && k < self.nz, "ScalarField3 index out of range");
        self.data[(k * self.ny + j) * self.nx + i]
    }

    /// World position of the sample at grid coordinates.
    ///
    /// # Panics
    /// Panics when out of range.
    #[must_use]
    pub fn position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        assert!(i < self.nx && j < self.ny && k < self.nz, "ScalarField3 index out of range");
        Self::grid_position(&self.bounds, self.nx, self.ny, self.nz, i, j, k)
    }

    /// Trilinear interpolation of the field at a world position
    /// (clamped to the grid).
    #[must_use]
    pub fn sample_trilinear(&self, p: Vec3) -> f64 {
        let e = self.bounds.max - self.bounds.min;
        let fx = ((p.x - self.bounds.min.x) / e.x * (self.nx - 1) as f64)
            .clamp(0.0, (self.nx - 1) as f64);
        let fy = ((p.y - self.bounds.min.y) / e.y * (self.ny - 1) as f64)
            .clamp(0.0, (self.ny - 1) as f64);
        let fz = ((p.z - self.bounds.min.z) / e.z * (self.nz - 1) as f64)
            .clamp(0.0, (self.nz - 1) as f64);
        let (i, j, k) = (
            (fx.floor() as usize).min(self.nx - 2),
            (fy.floor() as usize).min(self.ny - 2),
            (fz.floor() as usize).min(self.nz - 2),
        );
        let (tx, ty, tz) = (fx - i as f64, fy - j as f64, fz - k as f64);
        let mut acc = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { tx } else { 1.0 - tx })
                        * (if dy == 1 { ty } else { 1.0 - ty })
                        * (if dz == 1 { tz } else { 1.0 - tz });
                    acc += w * self.get(i + dx, j + dy, k + dz);
                }
            }
        }
        acc
    }

    /// Central-difference gradient at a grid point (one-sided on the
    /// boundary), in world units.
    ///
    /// # Panics
    /// Panics when out of range.
    #[must_use]
    pub fn gradient(&self, i: usize, j: usize, k: usize) -> Vec3 {
        assert!(i < self.nx && j < self.ny && k < self.nz, "ScalarField3 index out of range");
        let e = self.bounds.max - self.bounds.min;
        let (i0, i1) = (i.saturating_sub(1), (i + 1).min(self.nx - 1));
        let (j0, j1) = (j.saturating_sub(1), (j + 1).min(self.ny - 1));
        let (k0, k1) = (k.saturating_sub(1), (k + 1).min(self.nz - 1));
        let hx = e.x / (self.nx - 1) as f64;
        let hy = e.y / (self.ny - 1) as f64;
        let hz = e.z / (self.nz - 1) as f64;
        Vec3::new(
            (self.get(i1, j, k) - self.get(i0, j, k)) / ((i1 - i0) as f64 * hx),
            (self.get(i, j1, k) - self.get(i, j0, k)) / ((j1 - j0) as f64 * hy),
            (self.get(i, j, k1) - self.get(i, j, k0)) / ((k1 - k0) as f64 * hz),
        )
    }
}

/// Marching-squares connectivity for one cell face. Corners are listed
/// counterclockwise (seen from outside); bit k of `mask` is set when
/// corner k is inside (below iso). Local edge k joins corners k and
/// k+1. Each `(enter, exit)` pair is a directed surface crossing with
/// the inside region on its left; ambiguous diagonal cases separate
/// the inside corners, which depends only on the face's corner signs
/// and therefore agrees between the two cells sharing the face.
fn ms_links(mask: u8) -> &'static [(u8, u8)] {
    match mask {
        1 => &[(0, 3)],
        2 => &[(1, 0)],
        4 => &[(2, 1)],
        8 => &[(3, 2)],
        14 => &[(3, 0)],
        13 => &[(0, 1)],
        11 => &[(1, 2)],
        7 => &[(2, 3)],
        3 => &[(1, 3)],
        6 => &[(2, 0)],
        12 => &[(3, 1)],
        9 => &[(0, 2)],
        5 => &[(0, 3), (2, 1)],
        10 => &[(1, 0), (3, 2)],
        _ => &[],
    }
}

/// All isocontour crossings as directed segments (inside on the left).
#[must_use]
pub fn marching_squares(field: &ScalarField2, iso: f64) -> Vec<Segment2> {
    let mut out = Vec::new();
    for j in 0..field.height - 1 {
        for i in 0..field.width - 1 {
            // Corners counterclockwise starting at (i, j).
            let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let vals = corners.map(|(a, b)| field.get(a, b));
            let mask = (0..4).fold(0u8, |m, k| m | (u8::from(vals[k] < iso) << k));
            let crossing = |k: usize| {
                let (a, b) = (corners[k], corners[(k + 1) % 4]);
                let (va, vb) = (vals[k], vals[(k + 1) % 4]);
                let t = ((iso - va) / (vb - va)).clamp(0.0, 1.0);
                field.position(a.0, a.1).lerp(&field.position(b.0, b.1), t)
            };
            for &(en, ex) in ms_links(mask) {
                out.push(Segment2 { a: crossing(en as usize), b: crossing(ex as usize) });
            }
        }
    }
    out
}

/// Grid-edge identifier: lower sample coordinates plus 0 (edge toward
/// +x) or 1 (edge toward +y).
type EdgeKey2 = (usize, usize, u8);

/// Isocontours joined into polylines. Closed loops repeat their first
/// point at the end; open chains (which begin and end on the grid
/// boundary) do not. Inside (< iso) lies on the left of the direction
/// of travel.
#[must_use]
pub fn marching_squares_polylines(field: &ScalarField2, iso: f64) -> Vec<Vec<Vec2>> {
    let mut next: HashMap<EdgeKey2, EdgeKey2> = HashMap::new();
    let mut pos: HashMap<EdgeKey2, Vec2> = HashMap::new();
    let mut has_in: HashMap<EdgeKey2, bool> = HashMap::new();
    for j in 0..field.height - 1 {
        for i in 0..field.width - 1 {
            let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let vals = corners.map(|(a, b)| field.get(a, b));
            let mask = (0..4).fold(0u8, |m, k| m | (u8::from(vals[k] < iso) << k));
            // Local edge k as a grid-edge key.
            let key = |k: usize| -> EdgeKey2 {
                match k {
                    0 => (i, j, 0),
                    1 => (i + 1, j, 1),
                    2 => (i, j + 1, 0),
                    _ => (i, j, 1),
                }
            };
            let crossing = |k: usize| {
                let (a, b) = (corners[k], corners[(k + 1) % 4]);
                let (va, vb) = (vals[k], vals[(k + 1) % 4]);
                let t = ((iso - va) / (vb - va)).clamp(0.0, 1.0);
                field.position(a.0, a.1).lerp(&field.position(b.0, b.1), t)
            };
            for &(en, ex) in ms_links(mask) {
                let (ke, kx) = (key(en as usize), key(ex as usize));
                pos.insert(ke, crossing(en as usize));
                pos.insert(kx, crossing(ex as usize));
                next.insert(ke, kx);
                has_in.entry(ke).or_insert(false);
                has_in.insert(kx, true);
            }
        }
    }
    let mut out = Vec::new();
    let mut visited: HashMap<EdgeKey2, bool> = HashMap::new();
    // Open chains start at nodes with no incoming link.
    let mut starts: Vec<EdgeKey2> =
        next.keys().filter(|k| !has_in.get(*k).copied().unwrap_or(false)).copied().collect();
    starts.sort_unstable();
    for start in starts {
        let mut chain = vec![pos[&start]];
        visited.insert(start, true);
        let mut cur = start;
        while let Some(&nx) = next.get(&cur) {
            chain.push(pos[&nx]);
            if visited.insert(nx, true).is_some() {
                break;
            }
            cur = nx;
        }
        out.push(chain);
    }
    // Remaining links belong to closed loops.
    let mut loop_starts: Vec<EdgeKey2> =
        next.keys().filter(|k| !visited.contains_key(*k)).copied().collect();
    loop_starts.sort_unstable();
    for start in loop_starts {
        if visited.contains_key(&start) {
            continue;
        }
        let mut chain = vec![pos[&start]];
        visited.insert(start, true);
        let mut cur = next[&start];
        while cur != start {
            visited.insert(cur, true);
            chain.push(pos[&cur]);
            cur = next[&cur];
        }
        chain.push(pos[&start]);
        out.push(chain);
    }
    out
}

/// Joined contours for each requested level.
#[must_use]
pub fn contour_levels(field: &ScalarField2, levels: &[f64]) -> Vec<(f64, Vec<Vec<Vec2>>)> {
    levels.iter().map(|&l| (l, marching_squares_polylines(field, l))).collect()
}

/// Local cube corner offsets `(dx, dy, dz)`, Lorensen numbering.
const MC_CORNER: [(usize, usize, usize); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
];

/// The 12 cube edges as corner pairs, Lorensen numbering.
const MC_EDGE: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Cube faces, corners counterclockwise seen from outside.
const MC_FACE: [[usize; 4]; 6] = [
    [0, 3, 2, 1], // z = 0
    [4, 5, 6, 7], // z = 1
    [0, 1, 5, 4], // y = 0
    [2, 3, 7, 6], // y = 1
    [0, 4, 7, 3], // x = 0
    [1, 2, 6, 5], // x = 1
];

/// The 256-case marching-cubes triangle table (triples of cube edge
/// indices), generated at first use by the standard cycle
/// construction: every face pairs its crossings by [`ms_links`] and
/// the pairs are walked into closed polygons, which are then
/// fan-triangulated. Because the pairing depends only on each face's
/// corner signs, adjacent cells always agree along a shared face and
/// the surface is watertight; triangles are wound with normals toward
/// values above the iso level.
fn mc_table() -> &'static Vec<Vec<[usize; 3]>> {
    static TABLE: OnceLock<Vec<Vec<[usize; 3]>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let edge_of = |a: usize, b: usize| {
            MC_EDGE
                .iter()
                .position(|&(u, v)| (u == a && v == b) || (u == b && v == a))
                .expect("cube face edges are cube edges")
        };
        (0usize..256)
            .map(|mask| {
                let mut next = [usize::MAX; 12];
                for f in &MC_FACE {
                    let fmask =
                        (0..4).fold(0u8, |m, k| m | ((((mask >> f[k]) & 1) as u8) << k));
                    for &(en, ex) in ms_links(fmask) {
                        let local = |k: u8| edge_of(f[k as usize], f[(k as usize + 1) % 4]);
                        let (ge, gx) = (local(en), local(ex));
                        assert_eq!(next[ge], usize::MAX, "edge entered twice");
                        next[ge] = gx;
                    }
                }
                // Every crossed edge must have exactly one outgoing and
                // one incoming link.
                for (e, &(a, b)) in MC_EDGE.iter().enumerate() {
                    let crossed = (mask >> a) & 1 != (mask >> b) & 1;
                    assert_eq!(next[e] != usize::MAX, crossed, "inconsistent face pairing");
                }
                let mut tris = Vec::new();
                let mut visited = [false; 12];
                for start in 0..12 {
                    if next[start] == usize::MAX || visited[start] {
                        continue;
                    }
                    let mut cycle = vec![start];
                    visited[start] = true;
                    let mut cur = next[start];
                    while cur != start {
                        visited[cur] = true;
                        cycle.push(cur);
                        cur = next[cur];
                    }
                    assert!(cycle.len() >= 3, "degenerate surface cycle");
                    // The walk keeps the inside on the left, which
                    // yields inward normals; reverse for outward.
                    cycle.reverse();
                    for t in 1..cycle.len() - 1 {
                        tris.push([cycle[0], cycle[t], cycle[t + 1]]);
                    }
                }
                tris
            })
            .collect()
    })
}

/// Extracts the isosurface by marching cubes (Lorensen & Cline 1987;
/// the case table is generated by face-consistent cycle construction,
/// see [`mc_table`]). Output vertices are shared across cells (keyed
/// by grid edge), so the mesh is watertight wherever the surface does
/// not leave the grid.
#[must_use]
pub fn marching_cubes(field: &ScalarField3, iso: f64) -> Mesh {
    let table = mc_table();
    let (nx, ny) = (field.nx, field.ny);
    let gid = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut vmap: HashMap<(usize, usize), usize> = HashMap::new();
    let mut indices: Vec<[usize; 3]> = Vec::new();
    for k in 0..field.nz - 1 {
        for j in 0..field.ny - 1 {
            for i in 0..field.nx - 1 {
                let corner = |m: usize| {
                    let (dx, dy, dz) = MC_CORNER[m];
                    (i + dx, j + dy, k + dz)
                };
                let mask = (0..8).fold(0usize, |acc, m| {
                    let (ci, cj, ck) = corner(m);
                    acc | (usize::from(field.get(ci, cj, ck) < iso) << m)
                });
                if mask == 0 || mask == 255 {
                    continue;
                }
                for tri in &table[mask] {
                    let vids = tri.map(|e| {
                        let (a, b) = MC_EDGE[e];
                        let (ca, cb) = (corner(a), corner(b));
                        let (ga, gb) = (gid(ca.0, ca.1, ca.2), gid(cb.0, cb.1, cb.2));
                        let key = (ga.min(gb), ga.max(gb));
                        *vmap.entry(key).or_insert_with(|| {
                            let va = field.get(ca.0, ca.1, ca.2);
                            let vb = field.get(cb.0, cb.1, cb.2);
                            let t = ((iso - va) / (vb - va)).clamp(0.0, 1.0);
                            let pa = field.position(ca.0, ca.1, ca.2);
                            let pb = field.position(cb.0, cb.1, cb.2);
                            vertices.push(pa.lerp(&pb, t));
                            vertices.len() - 1
                        })
                    });
                    indices.push(vids);
                }
            }
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Marching-tetrahedra case table for a positively oriented
/// tetrahedron `(v0, v1, v2, v3)`: triangles as pairs of tetrahedron
/// vertices whose connecting edge is crossed, wound with normals
/// toward values above the iso level.
fn mt_cases(mask: u8) -> &'static [[(usize, usize); 3]] {
    match mask {
        1 => &[[(0, 1), (0, 2), (0, 3)]],
        2 => &[[(1, 0), (1, 3), (1, 2)]],
        4 => &[[(2, 0), (2, 1), (2, 3)]],
        8 => &[[(3, 0), (3, 2), (3, 1)]],
        3 => &[[(0, 2), (0, 3), (1, 3)], [(0, 2), (1, 3), (1, 2)]],
        5 => &[[(0, 1), (2, 1), (2, 3)], [(0, 1), (2, 3), (0, 3)]],
        9 => &[[(0, 1), (0, 2), (3, 2)], [(0, 1), (3, 2), (3, 1)]],
        6 => &[[(0, 1), (3, 1), (3, 2)], [(0, 1), (3, 2), (0, 2)]],
        10 => &[[(0, 1), (0, 3), (2, 3)], [(0, 1), (2, 3), (2, 1)]],
        12 => &[[(0, 2), (1, 2), (1, 3)], [(0, 2), (1, 3), (0, 3)]],
        7 => &[[(3, 0), (3, 1), (3, 2)]],
        11 => &[[(2, 0), (2, 3), (2, 1)]],
        13 => &[[(1, 0), (1, 2), (1, 3)]],
        14 => &[[(0, 1), (0, 3), (0, 2)]],
        _ => &[],
    }
}

/// Extracts the isosurface by marching tetrahedra over the Freudenthal
/// (Kuhn) 6-tetrahedra decomposition of each cell. The decomposition
/// cuts every cell face along its min-to-max diagonal, which depends
/// only on global grid coordinates, so adjacent cells agree and the
/// output is watertight. No ambiguous cases exist; the surface is
/// finer (more triangles) than marching cubes for the same grid.
#[must_use]
pub fn marching_tetrahedra(field: &ScalarField3, iso: f64) -> Mesh {
    const PERMS: [[usize; 3]; 6] =
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let (nx, ny) = (field.nx, field.ny);
    let gid = |c: [usize; 3]| (c[2] * ny + c[1]) * nx + c[0];
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut vmap: HashMap<(usize, usize), usize> = HashMap::new();
    let mut indices: Vec<[usize; 3]> = Vec::new();
    for k in 0..field.nz - 1 {
        for j in 0..field.ny - 1 {
            for i in 0..field.nx - 1 {
                for perm in &PERMS {
                    let mut corners = [[i, j, k]; 4];
                    for step in 0..3 {
                        corners[step + 1] = corners[step];
                        corners[step + 1][perm[step]] += 1;
                    }
                    let vals = corners.map(|c| field.get(c[0], c[1], c[2]));
                    let mask =
                        (0..4).fold(0u8, |m, v| m | (u8::from(vals[v] < iso) << v));
                    if mask == 0 || mask == 15 {
                        continue;
                    }
                    let pos = corners.map(|c| field.position(c[0], c[1], c[2]));
                    let flip = (pos[1] - pos[0])
                        .cross(&(pos[2] - pos[0]))
                        .dot(&(pos[3] - pos[0]))
                        < 0.0;
                    for tri in mt_cases(mask) {
                        let mut vids = tri.map(|(a, b)| {
                            let (ga, gb) = (gid(corners[a]), gid(corners[b]));
                            let key = (ga.min(gb), ga.max(gb));
                            *vmap.entry(key).or_insert_with(|| {
                                let t = ((iso - vals[a]) / (vals[b] - vals[a])).clamp(0.0, 1.0);
                                vertices.push(pos[a].lerp(&pos[b], t));
                                vertices.len() - 1
                            })
                        });
                        if flip {
                            vids.swap(1, 2);
                        }
                        indices.push(vids);
                    }
                }
            }
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Places a dual vertex for the cell at the given grid coordinates,
/// from the cell's edge-crossing points.
type PlaceVertex<'a> = dyn FnMut(usize, usize, usize, &[Vec3]) -> Vec3 + 'a;

/// Shared dual-method topology: one vertex per sign-crossed cell, one
/// quad around every interior sign-crossed grid edge, wound toward
/// values above the iso level.
fn dual_mesh(field: &ScalarField3, iso: f64, place_vertex: &mut PlaceVertex) -> Mesh {
    let mut cell_vertex: HashMap<(usize, usize, usize), usize> = HashMap::new();
    let mut vertices = Vec::new();
    // Pass 1: place a vertex in every cell the surface crosses.
    for k in 0..field.nz - 1 {
        for j in 0..field.ny - 1 {
            for i in 0..field.nx - 1 {
                let mut crossings: Vec<Vec3> = Vec::new();
                for &(a, b) in &MC_EDGE {
                    let (da, db) = (MC_CORNER[a], MC_CORNER[b]);
                    let va = field.get(i + da.0, j + da.1, k + da.2);
                    let vb = field.get(i + db.0, j + db.1, k + db.2);
                    if (va < iso) != (vb < iso) {
                        let t = ((iso - va) / (vb - va)).clamp(0.0, 1.0);
                        let pa = field.position(i + da.0, j + da.1, k + da.2);
                        let pb = field.position(i + db.0, j + db.1, k + db.2);
                        crossings.push(pa.lerp(&pb, t));
                    }
                }
                if !crossings.is_empty() {
                    cell_vertex.insert((i, j, k), vertices.len());
                    vertices.push(place_vertex(i, j, k, &crossings));
                }
            }
        }
    }
    // Pass 2: a quad of the four adjacent cell vertices around every
    // interior grid edge the surface crosses.
    let mut indices = Vec::new();
    let dims = [field.nx, field.ny, field.nz];
    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        // Quadrant offsets in (u, v), counterclockwise around +axis.
        let quad_off: [(i64, i64); 4] = [(-1, -1), (0, -1), (0, 0), (-1, 0)];
        for k in 0..field.nz {
            for j in 0..field.ny {
                for i in 0..field.nx {
                    let c = [i, j, k];
                    if c[axis] + 1 >= dims[axis] {
                        continue;
                    }
                    if c[u] == 0 || c[u] + 1 >= dims[u] || c[v] == 0 || c[v] + 1 >= dims[v] {
                        continue; // boundary edge: not all four cells exist
                    }
                    let va = field.get(c[0], c[1], c[2]);
                    let mut ch = c;
                    ch[axis] += 1;
                    let vb = field.get(ch[0], ch[1], ch[2]);
                    if (va < iso) == (vb < iso) {
                        continue;
                    }
                    let mut quad = [0usize; 4];
                    let mut ok = true;
                    for (q, &(du, dv)) in quad_off.iter().enumerate() {
                        let mut cc = [c[0] as i64, c[1] as i64, c[2] as i64];
                        cc[u] += du;
                        cc[v] += dv;
                        match cell_vertex.get(&(cc[0] as usize, cc[1] as usize, cc[2] as usize)) {
                            Some(&idx) => quad[q] = idx,
                            None => ok = false,
                        }
                    }
                    if !ok {
                        continue;
                    }
                    // Counterclockwise around +axis faces +axis; flip
                    // when the outside is on the low end.
                    if va < iso {
                        indices.push([quad[0], quad[1], quad[2]]);
                        indices.push([quad[0], quad[2], quad[3]]);
                    } else {
                        indices.push([quad[3], quad[2], quad[1]]);
                        indices.push([quad[3], quad[1], quad[0]]);
                    }
                }
            }
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Naive surface nets (Gibson 1998): each crossed cell gets the
/// centroid of its edge crossings; quads connect cells around crossed
/// interior grid edges. Smoother than marching cubes at the same
/// resolution but not guaranteed to stay inside each cell.
#[must_use]
pub fn surface_nets(field: &ScalarField3, iso: f64) -> Mesh {
    dual_mesh(field, iso, &mut |_, _, _, crossings| {
        let mut c = Vec3::ZERO;
        for p in crossings {
            c = c + *p;
        }
        c * (1.0 / crossings.len() as f64)
    })
}

/// Dual contouring (Ju et al. 2002): like surface nets, but each cell
/// vertex minimizes the quadratic error function
/// Σ (nᵢ · (x − pᵢ))² over the cell's edge crossings, with normals
/// supplied by `normals` (e.g. an SDF gradient). Reproduces sharp
/// features. The QEF is solved by regularized normal equations
/// (Gaussian elimination with partial pivoting), and the vertex is
/// clamped into its cell.
#[must_use]
pub fn dual_contouring(
    field: &ScalarField3,
    iso: f64,
    normals: &dyn Fn(Vec3) -> Vec3,
) -> Mesh {
    dual_mesh(field, iso, &mut |i, j, k, crossings| {
        let mut centroid = Vec3::ZERO;
        for p in crossings {
            centroid = centroid + *p;
        }
        centroid = centroid * (1.0 / crossings.len() as f64);
        let mut a = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for p in crossings {
            let n = normals(*p);
            let m = n.magnitude();
            if m == 0.0 {
                continue;
            }
            let n = n * (1.0 / m);
            let nv = [n.x, n.y, n.z];
            let d = n.dot(p);
            for r in 0..3 {
                for s in 0..3 {
                    a[r][s] += nv[r] * nv[s];
                }
                b[r] += nv[r] * d;
            }
        }
        // Tikhonov regularization pulls rank-deficient solves toward
        // the centroid.
        let lambda = 0.05;
        let cv = [centroid.x, centroid.y, centroid.z];
        for r in 0..3 {
            a[r][r] += lambda;
            b[r] += lambda * cv[r];
        }
        // 3x3 Gaussian elimination with partial pivoting.
        let mut m = [[0.0f64; 4]; 3];
        for r in 0..3 {
            m[r][..3].copy_from_slice(&a[r]);
            m[r][3] = b[r];
        }
        for col in 0..3 {
            let piv = (col..3)
                .max_by(|&x, &y| m[x][col].abs().total_cmp(&m[y][col].abs()))
                .expect("nonempty range");
            m.swap(col, piv);
            let pivot_row = m[col];
            for row in m.iter_mut().skip(col + 1) {
                let f = row[col] / pivot_row[col];
                for (c, pv) in pivot_row.iter().enumerate().skip(col) {
                    row[c] -= f * pv;
                }
            }
        }
        let mut x = [0.0f64; 3];
        for row in (0..3).rev() {
            let mut acc = m[row][3];
            for c in row + 1..3 {
                acc -= m[row][c] * x[c];
            }
            x[row] = acc / m[row][row];
        }
        // Clamp into the cell.
        let lo = field.position(i, j, k);
        let hi = field.position(i + 1, j + 1, k + 1);
        Vec3::new(x[0].clamp(lo.x, hi.x), x[1].clamp(lo.y, hi.y), x[2].clamp(lo.z, hi.z))
    })
}

/// Polygonizes a metaball (blobby) surface: field
/// Σ rᵢ² / |p − cᵢ|² compared against `threshold`, marched on a
/// `res`³ grid over `bounds`. Larger `threshold` shrinks the blobs.
///
/// # Panics
/// Panics unless `threshold > 0`, `res >= 2`, and `centers` is
/// nonempty.
#[must_use]
pub fn metaballs(centers: &[(Vec3, f64)], bounds: &Aabb, res: usize, threshold: f64) -> Mesh {
    assert!(!centers.is_empty(), "metaballs requires at least one center");
    assert!(threshold > 0.0 && res >= 2, "metaballs requires threshold > 0, res >= 2");
    let f = |p: Vec3| {
        let strength: f64 = centers
            .iter()
            .map(|&(c, r)| {
                let d2 = (p - c).magnitude_squared();
                if d2 == 0.0 {
                    f64::INFINITY
                } else {
                    r * r / d2
                }
            })
            .sum();
        // Below iso 0 inside, matching the SDF convention.
        threshold - strength
    };
    let field = ScalarField3::from_fn(*bounds, res, res, res, &f);
    marching_cubes(&field, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::volume_sphere;
    use crate::spatial::sdf::{sd_sphere, sd_torus};

    fn sphere_field(r: f64, n: usize) -> ScalarField3 {
        let bounds =
            Aabb { min: Vec3::new(-1.5 * r, -1.5 * r, -1.5 * r), max: Vec3::new(1.5 * r, 1.5 * r, 1.5 * r) };
        ScalarField3::from_sdf(bounds, n, n, n, &|p| sd_sphere(p, r))
    }

    fn assert_watertight(m: &Mesh) {
        let mut counts: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for &[a, b, c] in &m.indices {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let e = counts.entry((u.min(v), u.max(v))).or_insert((0, 0));
                if u < v {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                }
            }
        }
        for (&edge, &(fwd, rev)) in &counts {
            assert_eq!((fwd, rev), (1, 1), "edge {edge:?} not manifold");
        }
    }

    fn euler(m: &Mesh) -> i64 {
        m.vertices.len() as i64 - m.edges().len() as i64 + m.indices.len() as i64
    }

    #[test]
    fn test_mc_table_structural_invariants() {
        let table = mc_table();
        assert_eq!(table.len(), 256);
        assert!(table[0].is_empty() && table[255].is_empty());
        for (mask, tris) in table.iter().enumerate() {
            for tri in tris {
                for &e in tri {
                    let (a, b) = MC_EDGE[e];
                    assert_ne!(
                        (mask >> a) & 1,
                        (mask >> b) & 1,
                        "case {mask} uses uncrossed edge {e}"
                    );
                }
            }
        }
        // Single-corner case: one triangle, outward orientation.
        assert_eq!(table[1].len(), 1);
        let tri = table[1][0];
        let mid = |e: usize| {
            let (a, b) = MC_EDGE[e];
            let pa = MC_CORNER[a];
            let pb = MC_CORNER[b];
            Vec3::new(
                (pa.0 + pb.0) as f64 / 2.0,
                (pa.1 + pb.1) as f64 / 2.0,
                (pa.2 + pb.2) as f64 / 2.0,
            )
        };
        let (p0, p1, p2) = (mid(tri[0]), mid(tri[1]), mid(tri[2]));
        let n = (p1 - p0).cross(&(p2 - p0));
        // Corner 0 is inside; the normal must point away from it.
        assert!(n.dot(&Vec3::new(1.0, 1.0, 1.0)) > 0.0);
    }

    #[test]
    fn test_marching_cubes_sphere() {
        let r = 1.0;
        let field = sphere_field(r, 64);
        let m = marching_cubes(&field, 0.0);
        assert_watertight(&m);
        assert_eq!(euler(&m), 2);
        let v = m.volume();
        let exact = volume_sphere(r);
        assert!(
            (v - exact).abs() / exact < 0.02,
            "marching cubes sphere volume {v} vs {exact}"
        );
        // Every vertex lies close to the true surface.
        let cell = 3.0 * r / 63.0;
        for p in &m.vertices {
            assert!((p.magnitude() - r).abs() < cell);
            assert!(field.sample_trilinear(*p).abs() < cell);
        }
    }

    #[test]
    fn test_marching_cubes_torus_genus() {
        let bounds =
            Aabb { min: Vec3::new(-2.0, -2.0, -2.0), max: Vec3::new(2.0, 2.0, 2.0) };
        let field =
            ScalarField3::from_sdf(bounds, 48, 48, 48, &|p| sd_torus(p, 1.2, 0.4));
        let m = marching_cubes(&field, 0.0);
        assert_watertight(&m);
        assert_eq!(euler(&m), 0);
    }

    #[test]
    fn test_marching_tetrahedra_sphere() {
        let r = 1.0;
        let field = sphere_field(r, 48);
        let m = marching_tetrahedra(&field, 0.0);
        assert_watertight(&m);
        assert_eq!(euler(&m), 2);
        let exact = volume_sphere(r);
        assert!((m.volume() - exact).abs() / exact < 0.02);
    }

    #[test]
    fn test_surface_nets_and_dual_contouring_sphere() {
        let r = 1.0;
        let field = sphere_field(r, 40);
        let exact = volume_sphere(r);

        let sn = surface_nets(&field, 0.0);
        assert_watertight(&sn);
        assert_eq!(euler(&sn), 2);
        assert!((sn.volume() - exact).abs() / exact < 0.05);

        let dc = dual_contouring(&field, 0.0, &|p| p.normalized());
        assert_watertight(&dc);
        assert_eq!(euler(&dc), 2);
        assert!((dc.volume() - exact).abs() / exact < 0.05);
        // Dual contouring vertices sit on the sphere (normals are
        // exact), tighter than plain surface nets.
        for p in &dc.vertices {
            assert!((p.magnitude() - r).abs() < 0.02);
        }
    }

    #[test]
    fn test_dual_contouring_recovers_box_corners() {
        use crate::spatial::sdf::sd_box;
        let bounds =
            Aabb { min: Vec3::new(-1.6, -1.6, -1.6), max: Vec3::new(1.6, 1.6, 1.6) };
        let half = Vec3::new(1.0, 1.0, 1.0);
        let field = ScalarField3::from_sdf(bounds, 33, 33, 33, &|p| sd_box(p, half));
        let grad = |p: Vec3| {
            let h = 1e-5;
            Vec3::new(
                sd_box(p + Vec3::new(h, 0.0, 0.0), half) - sd_box(p - Vec3::new(h, 0.0, 0.0), half),
                sd_box(p + Vec3::new(0.0, h, 0.0), half) - sd_box(p - Vec3::new(0.0, h, 0.0), half),
                sd_box(p + Vec3::new(0.0, 0.0, h), half) - sd_box(p - Vec3::new(0.0, 0.0, h), half),
            )
        };
        let m = dual_contouring(&field, 0.0, &grad);
        assert_watertight(&m);
        // A vertex lands (nearly) on each of the 8 sharp corners.
        for corner in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        ] {
            let d = m
                .vertices
                .iter()
                .map(|v| v.distance_to(&corner))
                .fold(f64::INFINITY, f64::min);
            assert!(d < 0.05, "no vertex near corner {corner:?} (closest {d})");
        }
        let exact = 8.0;
        assert!((m.volume() - exact).abs() / exact < 0.02);
    }

    #[test]
    fn test_marching_squares_circle() {
        let r = 1.0;
        let bounds = Rect { min: Vec2::new(-2.0, -2.0), max: Vec2::new(2.0, 2.0) };
        let field = ScalarField2::from_fn(bounds, 128, 128, &|p| p.magnitude() - r);
        let segments = marching_squares(&field, 0.0);
        let total: f64 = segments.iter().map(|s| s.a.distance_to(&s.b)).sum();
        let exact = 2.0 * std::f64::consts::PI * r;
        assert!((total - exact).abs() / exact < 0.01, "contour length {total} vs {exact}");
        // Inside on the left means counterclockwise around the disk:
        // positive signed area via the shoelace sum over segments.
        let area: f64 = segments.iter().map(|s| s.a.cross(&s.b) / 2.0).sum();
        assert!((area - std::f64::consts::PI * r * r).abs() / (std::f64::consts::PI) < 0.01);
    }

    #[test]
    fn test_marching_squares_polylines_join() {
        let bounds = Rect { min: Vec2::new(-2.0, -2.0), max: Vec2::new(2.0, 2.0) };
        let field = ScalarField2::from_fn(bounds, 96, 96, &|p| p.magnitude() - 1.0);
        let loops = marching_squares_polylines(&field, 0.0);
        assert_eq!(loops.len(), 1, "one circle gives one loop");
        let lp = &loops[0];
        assert!(lp.first().unwrap().distance_to(lp.last().unwrap()) < 1e-12, "loop closes");
        // An open contour: iso line y = 0.5 crossing the whole grid.
        let field2 = ScalarField2::from_fn(bounds, 32, 32, &|p| p.y);
        let chains = marching_squares_polylines(&field2, 0.5);
        assert_eq!(chains.len(), 1);
        let ch = &chains[0];
        assert!(ch.first().unwrap().distance_to(ch.last().unwrap()) > 3.0, "chain stays open");
        for p in ch {
            assert!((p.y - 0.5).abs() < 1e-9);
        }
        let levels = contour_levels(&field, &[-0.5, 0.0, 0.5]);
        assert_eq!(levels.len(), 3);
        for (lvl, loops) in &levels {
            assert_eq!(loops.len(), 1);
            let expected_r = 1.0 + lvl;
            for p in &loops[0] {
                assert!((p.magnitude() - expected_r).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_metaballs_single_ball_is_sphere() {
        let bounds =
            Aabb { min: Vec3::new(-2.0, -2.0, -2.0), max: Vec3::new(2.0, 2.0, 2.0) };
        // One ball: r^2/d^2 = 1 at d = r.
        let m = metaballs(&[(Vec3::ZERO, 1.0)], &bounds, 48, 1.0);
        assert_watertight(&m);
        assert_eq!(euler(&m), 2);
        for v in &m.vertices {
            assert!((v.magnitude() - 1.0).abs() < 0.05);
        }
        // Two overlapping balls fuse into one component (genus 0).
        let m2 = metaballs(
            &[(Vec3::new(-0.5, 0.0, 0.0), 1.0), (Vec3::new(0.5, 0.0, 0.0), 1.0)],
            &bounds,
            48,
            1.0,
        );
        assert_watertight(&m2);
        assert_eq!(euler(&m2), 2);
        assert!(m2.volume() > volume_sphere(1.0));
    }

    #[test]
    fn test_scalar_field3_accessors() {
        let bounds = Aabb { min: Vec3::ZERO, max: Vec3::new(1.0, 2.0, 3.0) };
        let field = ScalarField3::from_fn(bounds, 5, 5, 5, &|p| p.x + 2.0 * p.y - p.z);
        assert!((field.get(4, 0, 0) - 1.0).abs() < 1e-12);
        assert!((field.get(0, 4, 0) - 4.0).abs() < 1e-12);
        let p = Vec3::new(0.3, 0.9, 1.7);
        // A linear field is reproduced exactly by trilinear sampling.
        assert!((field.sample_trilinear(p) - (p.x + 2.0 * p.y - p.z)).abs() < 1e-12);
        let g = field.gradient(2, 2, 2);
        assert!((g - Vec3::new(1.0, 2.0, -1.0)).magnitude() < 1e-9);
        let g = field.gradient(0, 0, 4);
        assert!((g - Vec3::new(1.0, 2.0, -1.0)).magnitude() < 1e-9);
    }
}
