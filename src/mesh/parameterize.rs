//! Mesh parameterization: closed-form spherical/planar/cylindrical
//! projections, harmonic (cotangent-Laplace) disk parameterization
//! with fixed boundaries, least-squares conformal maps (Lévy,
//! Petitjean, Ray & Maillot 2002), and per-triangle conformal and
//! area distortion measures.

use crate::linalg::sparse::{conjugate_gradient, CsrMatrix};
use crate::math::{Vec2, Vec3};
use crate::mesh::Mesh;

/// Target shape for the fixed boundary of a harmonic
/// parameterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryShape {
    /// Unit circle, uniform vertex spacing.
    Circle,
    /// Unit square, uniform vertex spacing along the perimeter.
    Square,
    /// Unit circle with spacing proportional to the boundary's 3-D
    /// edge lengths.
    Free,
}

/// Spherical texture coordinates for a (genus-0) mesh: each vertex
/// direction from the centroid maps to
/// u = ½ + atan2(d_y, d_x)/2π, v = acos(d_z)/π. Writes `m.uvs`.
///
/// # Panics
/// Panics on an empty mesh.
pub fn spherical_uv(m: &mut Mesh) {
    assert!(!m.vertices.is_empty(), "mesh has no vertices");
    let centroid = m
        .vertices
        .iter()
        .fold(Vec3::ZERO, |acc, &v| acc + v)
        * (1.0 / m.vertices.len() as f64);
    let uvs = m
        .vertices
        .iter()
        .map(|&v| {
            let d = v - centroid;
            let len = d.magnitude();
            if len == 0.0 {
                return Vec2::new(0.5, 0.5);
            }
            let d = d * (1.0 / len);
            Vec2::new(
                0.5 + d.y.atan2(d.x) / std::f64::consts::TAU,
                (d.z.clamp(-1.0, 1.0)).acos() / std::f64::consts::PI,
            )
        })
        .collect();
    m.uvs = Some(uvs);
}

fn plane_basis(normal: Vec3) -> (Vec3, Vec3) {
    let n = normal.normalized();
    let helper =
        if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let t1 = (helper - n * helper.dot(&n)).normalized();
    let t2 = n.cross(&t1);
    (t1, t2)
}

/// Planar projection along `normal`, normalized so the projected
/// bounding box spans [0, 1]². Writes `m.uvs`.
///
/// # Panics
/// Panics on an empty mesh or a zero normal.
pub fn planar_uv(m: &mut Mesh, normal: Vec3) {
    assert!(!m.vertices.is_empty(), "mesh has no vertices");
    assert!(normal.magnitude() > 0.0, "normal must be non-zero");
    let (t1, t2) = plane_basis(normal);
    let projected: Vec<Vec2> =
        m.vertices.iter().map(|v| Vec2::new(v.dot(&t1), v.dot(&t2))).collect();
    let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
    let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in &projected {
        lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
        hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
    }
    let span = Vec2::new((hi.x - lo.x).max(1e-300), (hi.y - lo.y).max(1e-300));
    m.uvs = Some(
        projected
            .iter()
            .map(|p| Vec2::new((p.x - lo.x) / span.x, (p.y - lo.y) / span.y))
            .collect(),
    );
}

/// Cylindrical projection about `axis` through the centroid:
/// u = angle/2π around the axis, v = normalized height along it.
/// Writes `m.uvs`.
///
/// # Panics
/// Panics on an empty mesh or a zero axis.
pub fn cylindrical_uv(m: &mut Mesh, axis: Vec3) {
    assert!(!m.vertices.is_empty(), "mesh has no vertices");
    assert!(axis.magnitude() > 0.0, "axis must be non-zero");
    let a = axis.normalized();
    let (t1, t2) = plane_basis(a);
    let centroid = m
        .vertices
        .iter()
        .fold(Vec3::ZERO, |acc, &v| acc + v)
        * (1.0 / m.vertices.len() as f64);
    let heights: Vec<f64> = m.vertices.iter().map(|v| (*v - centroid).dot(&a)).collect();
    let (h_lo, h_hi) = heights
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &h| (lo.min(h), hi.max(h)));
    let span = (h_hi - h_lo).max(1e-300);
    m.uvs = Some(
        m.vertices
            .iter()
            .zip(&heights)
            .map(|(v, &h)| {
                let d = *v - centroid;
                let angle = d.dot(&t2).atan2(d.dot(&t1));
                Vec2::new(0.5 + angle / std::f64::consts::TAU, (h - h_lo) / span)
            })
            .collect(),
    );
}

/// The single boundary loop of a disk-topology mesh.
fn single_boundary_loop(m: &Mesh) -> Vec<usize> {
    let loops = crate::mesh::analyze::boundary_loops(m);
    assert_eq!(
        loops.len(),
        1,
        "harmonic parameterization needs disk topology (exactly one boundary loop)"
    );
    loops.into_iter().next().expect("one loop")
}

/// Positions for the boundary vertices on the target shape.
fn boundary_positions(m: &Mesh, boundary: &[usize], shape: BoundaryShape) -> Vec<Vec2> {
    let n = boundary.len();
    // Cumulative parameter along the loop: uniform by count, or by
    // 3-D arclength for Free.
    let mut t = Vec::with_capacity(n);
    match shape {
        BoundaryShape::Circle | BoundaryShape::Square => {
            for i in 0..n {
                t.push(i as f64 / n as f64);
            }
        }
        BoundaryShape::Free => {
            let mut acc = 0.0;
            for i in 0..n {
                t.push(acc);
                acc += m.vertices[boundary[i]]
                    .distance_to(&m.vertices[boundary[(i + 1) % n]]);
            }
            let total = acc.max(1e-300);
            for ti in &mut t {
                *ti /= total;
            }
        }
    }
    t.iter()
        .map(|&ti| match shape {
            BoundaryShape::Circle | BoundaryShape::Free => {
                let a = ti * std::f64::consts::TAU;
                Vec2::new(a.cos(), a.sin())
            }
            BoundaryShape::Square => {
                // Walk the unit square perimeter counterclockwise.
                let s = ti * 4.0;
                if s < 1.0 {
                    Vec2::new(s, 0.0)
                } else if s < 2.0 {
                    Vec2::new(1.0, s - 1.0)
                } else if s < 3.0 {
                    Vec2::new(3.0 - s, 1.0)
                } else {
                    Vec2::new(0.0, 4.0 - s)
                }
            }
        })
        .collect()
}

/// Cotangent edge weights: w_ij = ½(cot α + cot β) over the angles
/// opposite edge (i, j).
fn cotan_weights(m: &Mesh) -> std::collections::HashMap<(usize, usize), f64> {
    let mut weights = std::collections::HashMap::new();
    for &[a, b, c] in &m.indices {
        for (i, j, k) in [(a, b, c), (b, c, a), (c, a, b)] {
            // Angle at k, opposite edge (i, j).
            let u = m.vertices[i] - m.vertices[k];
            let v = m.vertices[j] - m.vertices[k];
            let cross = u.cross(&v).magnitude();
            let cot = if cross > 1e-14 { u.dot(&v) / cross } else { 0.0 };
            let key = (i.min(j), i.max(j));
            *weights.entry(key).or_insert(0.0) += 0.5 * cot;
        }
    }
    weights
}

/// Harmonic (cotangent-weight) parameterization of a disk-topology
/// mesh: the boundary loop is pinned to the target shape and the
/// interior solves the discrete Laplace equation, giving the unique
/// harmonic extension (identity up to similarity on flat meshes;
/// convexity of the target keeps the map injective for meshes with
/// non-negative weights). Returns one UV per vertex.
///
/// # Panics
/// Panics unless the mesh has disk topology and the linear solve
/// converges.
#[must_use]
pub fn harmonic_parameterization(m: &Mesh, boundary_shape: BoundaryShape) -> Vec<Vec2> {
    let boundary = single_boundary_loop(m);
    let target = boundary_positions(m, &boundary, boundary_shape);
    let n = m.vertices.len();
    let mut is_boundary = vec![false; n];
    let mut fixed = vec![Vec2::ZERO; n];
    for (bi, &v) in boundary.iter().enumerate() {
        is_boundary[v] = true;
        fixed[v] = target[bi];
    }
    // Interior index mapping.
    let mut interior = Vec::new();
    let mut index_of = vec![usize::MAX; n];
    for v in 0..n {
        if !is_boundary[v] {
            index_of[v] = interior.len();
            interior.push(v);
        }
    }
    if interior.is_empty() {
        return fixed;
    }
    let weights = cotan_weights(m);
    // Assemble L x = b over interior vertices for both coordinates.
    let mut triplets = Vec::new();
    let mut diag = vec![0.0; interior.len()];
    let mut bu = vec![0.0; interior.len()];
    let mut bv = vec![0.0; interior.len()];
    for (&(i, j), &w) in &weights {
        for (from, to) in [(i, j), (j, i)] {
            if is_boundary[from] {
                continue;
            }
            let row = index_of[from];
            diag[row] += w;
            if is_boundary[to] {
                bu[row] += w * fixed[to].x;
                bv[row] += w * fixed[to].y;
            } else {
                triplets.push((row, index_of[to], -w));
            }
        }
    }
    for (row, &d) in diag.iter().enumerate() {
        triplets.push((row, row, d));
    }
    let laplacian = CsrMatrix::from_triplets(interior.len(), interior.len(), &triplets);
    let x0 = vec![0.0; interior.len()];
    let us = conjugate_gradient(&laplacian, &bu, &x0, 1e-12, 20 * interior.len() + 100)
        .expect("harmonic system solves");
    let vs = conjugate_gradient(&laplacian, &bv, &x0, 1e-12, 20 * interior.len() + 100)
        .expect("harmonic system solves");
    let mut uv = fixed;
    for (k, &v) in interior.iter().enumerate() {
        uv[v] = Vec2::new(us[k], vs[k]);
    }
    uv
}

/// Local orthonormal 2-D coordinates of a triangle's vertices.
fn triangle_local(p0: Vec3, p1: Vec3, p2: Vec3) -> Option<[Vec2; 3]> {
    let e1 = p1 - p0;
    let len = e1.magnitude();
    if len <= 0.0 {
        return None;
    }
    let x = e1 * (1.0 / len);
    let e2 = p2 - p0;
    let ny = e2 - x * e2.dot(&x);
    let h = ny.magnitude();
    if h <= 0.0 {
        return None;
    }
    Some([
        Vec2::new(0.0, 0.0),
        Vec2::new(len, 0.0),
        Vec2::new(e2.dot(&x), h),
    ])
}

/// Least-squares conformal map: minimizes the conformal energy
/// Σ_T A_T |∇u rotated 90° − ∇v|² with two pinned vertices removing
/// the similarity ambiguity. Solved through the normal equations by
/// conjugate gradients. Returns one UV per vertex.
///
/// # Panics
/// Panics unless the two pins are distinct valid vertices and the
/// solve converges.
#[must_use]
pub fn lscm(m: &Mesh, pinned: [(usize, Vec2); 2]) -> Vec<Vec2> {
    let n = m.vertices.len();
    assert!(pinned[0].0 < n && pinned[1].0 < n, "pinned vertices out of range");
    assert_ne!(pinned[0].0, pinned[1].0, "pins must be distinct");
    // Unknown layout: u_i -> 2i, v_i -> 2i+1; pinned entries removed.
    let mut pin_value = std::collections::HashMap::new();
    pin_value.insert(2 * pinned[0].0, pinned[0].1.x);
    pin_value.insert(2 * pinned[0].0 + 1, pinned[0].1.y);
    pin_value.insert(2 * pinned[1].0, pinned[1].1.x);
    pin_value.insert(2 * pinned[1].0 + 1, pinned[1].1.y);
    let mut col_of = vec![usize::MAX; 2 * n];
    let mut cols = 0usize;
    for (i, slot) in col_of.iter_mut().enumerate() {
        if !pin_value.contains_key(&i) {
            *slot = cols;
            cols += 1;
        }
    }
    // Normal equations assembled triangle by triangle.
    let mut ata = std::collections::HashMap::new();
    let mut atb = vec![0.0; cols];
    for &[a, b, c] in &m.indices {
        let Some(local) = triangle_local(m.vertices[a], m.vertices[b], m.vertices[c]) else {
            continue;
        };
        let area = 0.5 * (local[1] - local[0]).cross(&(local[2] - local[0]));
        if area <= 1e-14 {
            continue;
        }
        // Hat-function gradients in local coordinates.
        let verts = [a, b, c];
        let grads: [Vec2; 3] = [
            rot90(local[2] - local[1]) * (1.0 / (2.0 * area)),
            rot90(local[0] - local[2]) * (1.0 / (2.0 * area)),
            rot90(local[1] - local[0]) * (1.0 / (2.0 * area)),
        ];
        let s = area.sqrt();
        // Two residual rows: R∇u − ∇v = 0, weighted by sqrt(area).
        // Row 1 (x component): Σ_j (−g_j.y·u_j − g_j.x·v_j)
        // Row 2 (y component): Σ_j ( g_j.x·u_j − g_j.y·v_j)
        let mut rows: [Vec<(usize, f64)>; 2] = [Vec::new(), Vec::new()];
        for (j, &vid) in verts.iter().enumerate() {
            rows[0].push((2 * vid, -grads[j].y * s));
            rows[0].push((2 * vid + 1, -grads[j].x * s));
            rows[1].push((2 * vid, grads[j].x * s));
            rows[1].push((2 * vid + 1, -grads[j].y * s));
        }
        for row in &rows {
            for &(gi, ci) in row {
                match pin_value.get(&gi) {
                    Some(_) => {}
                    None => {
                        for &(gj, cj) in row {
                            if let Some(&val) = pin_value.get(&gj) {
                                atb[col_of[gi]] -= ci * cj * val;
                            } else {
                                *ata.entry((col_of[gi], col_of[gj])).or_insert(0.0) += ci * cj;
                            }
                        }
                    }
                }
            }
        }
    }
    let triplets: Vec<(usize, usize, f64)> =
        ata.into_iter().map(|((i, j), v)| (i, j, v)).collect();
    let matrix = CsrMatrix::from_triplets(cols, cols, &triplets);
    let x0 = vec![0.0; cols];
    let solution = conjugate_gradient(&matrix, &atb, &x0, 1e-12, 40 * cols + 200)
        .expect("LSCM system solves");
    (0..n)
        .map(|i| {
            let u = pin_value
                .get(&(2 * i))
                .copied()
                .unwrap_or_else(|| solution[col_of[2 * i]]);
            let v = pin_value
                .get(&(2 * i + 1))
                .copied()
                .unwrap_or_else(|| solution[col_of[2 * i + 1]]);
            Vec2::new(u, v)
        })
        .collect()
}

fn rot90(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

/// Singular values of the per-triangle 3D→UV Jacobian.
fn triangle_singular_values(m: &Mesh, uv: &[Vec2], tri: [usize; 3]) -> Option<(f64, f64)> {
    let [a, b, c] = tri;
    let local = triangle_local(m.vertices[a], m.vertices[b], m.vertices[c])?;
    let e1 = local[1] - local[0];
    let e2 = local[2] - local[0];
    let f1 = uv[b] - uv[a];
    let f2 = uv[c] - uv[a];
    // J solves J [e1 e2] = [f1 f2].
    let det = e1.x * e2.y - e1.y * e2.x;
    if det.abs() < 1e-300 {
        return None;
    }
    let inv = [
        [e2.y / det, -e2.x / det],
        [-e1.y / det, e1.x / det],
    ];
    let j = [
        [f1.x * inv[0][0] + f2.x * inv[1][0], f1.x * inv[0][1] + f2.x * inv[1][1]],
        [f1.y * inv[0][0] + f2.y * inv[1][0], f1.y * inv[0][1] + f2.y * inv[1][1]],
    ];
    // Singular values from JᵀJ.
    let a11 = j[0][0] * j[0][0] + j[1][0] * j[1][0];
    let a12 = j[0][0] * j[0][1] + j[1][0] * j[1][1];
    let a22 = j[0][1] * j[0][1] + j[1][1] * j[1][1];
    let mean = 0.5 * (a11 + a22);
    let d = (0.25 * (a11 - a22) * (a11 - a22) + a12 * a12).sqrt();
    Some(((mean + d).max(0.0).sqrt(), (mean - d).max(0.0).sqrt()))
}

/// Per-triangle conformal distortion σ₁/σ₂ of the parameterization
/// (1 = angle-preserving; degenerate triangles report 1).
///
/// # Panics
/// Panics unless `uv` has one entry per vertex.
#[must_use]
pub fn conformal_distortion(m: &Mesh, uv: &[Vec2]) -> Vec<f64> {
    assert_eq!(uv.len(), m.vertices.len(), "one UV per vertex");
    m.indices
        .iter()
        .map(|&tri| match triangle_singular_values(m, uv, tri) {
            Some((s1, s2)) if s2 > 1e-14 => s1 / s2,
            _ => 1.0,
        })
        .collect()
}

/// Per-triangle area distortion: the UV/3-D area ratio normalized by
/// the global ratio, so a globally scaled isometry reports 1
/// everywhere.
///
/// # Panics
/// Panics unless `uv` has one entry per vertex.
#[must_use]
pub fn area_distortion(m: &Mesh, uv: &[Vec2]) -> Vec<f64> {
    assert_eq!(uv.len(), m.vertices.len(), "one UV per vertex");
    let areas: Vec<(f64, f64)> = m
        .indices
        .iter()
        .map(|&[a, b, c]| {
            let a3 = 0.5
                * (m.vertices[b] - m.vertices[a])
                    .cross(&(m.vertices[c] - m.vertices[a]))
                    .magnitude();
            let a2 = 0.5 * (uv[b] - uv[a]).cross(&(uv[c] - uv[a])).abs();
            (a2, a3)
        })
        .collect();
    let total2: f64 = areas.iter().map(|a| a.0).sum();
    let total3: f64 = areas.iter().map(|a| a.1).sum();
    let global = if total3 > 0.0 { total2 / total3 } else { 1.0 };
    areas
        .iter()
        .map(|&(a2, a3)| {
            if a3 > 1e-300 && global > 0.0 { a2 / a3 / global } else { 1.0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generate;

    /// Flat multi-ring disk in the xy plane: `rings` concentric
    /// rings of `segments` vertices plus the center.
    fn ring_disk(rings: usize, segments: usize, radius: f64) -> Mesh {
        let mut vertices = vec![Vec3::ZERO];
        for k in 1..=rings {
            let r = radius * k as f64 / rings as f64;
            for j in 0..segments {
                let a = std::f64::consts::TAU * j as f64 / segments as f64;
                vertices.push(Vec3::new(r * a.cos(), r * a.sin(), 0.0));
            }
        }
        let ring_start = |k: usize| 1 + (k - 1) * segments;
        let mut indices = Vec::new();
        for j in 0..segments {
            indices.push([0, ring_start(1) + j, ring_start(1) + (j + 1) % segments]);
        }
        for k in 1..rings {
            for j in 0..segments {
                let (a, b) = (ring_start(k) + j, ring_start(k) + (j + 1) % segments);
                let (c, d) = (ring_start(k + 1) + j, ring_start(k + 1) + (j + 1) % segments);
                indices.push([a, c, d]);
                indices.push([a, d, b]);
            }
        }
        Mesh { vertices, indices, normals: None, uvs: None }
    }

    /// Open cylinder patch (θ from 0 to `sweep`), nu×nv grid.
    fn open_cylinder(sweep: f64, height: f64, nu: usize, nv: usize) -> Mesh {
        let mut vertices = Vec::new();
        for j in 0..nv {
            for i in 0..nu {
                let theta = sweep * i as f64 / (nu - 1) as f64;
                let z = height * j as f64 / (nv - 1) as f64;
                vertices.push(Vec3::new(theta.cos(), theta.sin(), z));
            }
        }
        let mut indices = Vec::new();
        for j in 0..nv - 1 {
            for i in 0..nu - 1 {
                let a = j * nu + i;
                let b = a + 1;
                let c = a + nu;
                let d = c + 1;
                indices.push([a, b, d]);
                indices.push([a, d, c]);
            }
        }
        Mesh { vertices, indices, normals: None, uvs: None }
    }

    #[test]
    fn test_projection_uvs() {
        let mut sphere = generate::uv_sphere(2.0, 16, 8);
        spherical_uv(&mut sphere);
        let uvs = sphere.uvs.as_ref().expect("uvs written");
        assert_eq!(uvs.len(), sphere.vertices.len());
        assert!(uvs.iter().all(|uv| (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)));
        // Planar projection of a plane grid spans [0, 1]^2 and is
        // affine in the grid coordinates.
        let mut plane = generate::plane_grid(2.0, 3.0, 4, 5);
        planar_uv(&mut plane, Vec3::new(0.0, 1.0, 0.0));
        let uvs = plane.uvs.as_ref().expect("uvs written");
        let (mut lo, mut hi) = (1.0f64, 0.0f64);
        for uv in uvs {
            lo = lo.min(uv.x.min(uv.y));
            hi = hi.max(uv.x.max(uv.y));
        }
        assert!(lo.abs() < 1e-12 && (hi - 1.0).abs() < 1e-12, "spans the unit square");
        // Cylindrical projection of a cylinder: v matches height.
        let mut cyl = generate::cylinder(1.0, 2.0, 24, false);
        cylindrical_uv(&mut cyl, Vec3::new(0.0, 1.0, 0.0));
        let uvs = cyl.uvs.as_ref().expect("uvs written");
        for (vert, uv) in cyl.vertices.iter().zip(uvs) {
            assert!((uv.y - (vert.y + 1.0) / 2.0).abs() < 1e-9, "v is normalized height");
        }
    }

    #[test]
    fn test_harmonic_flat_disk_is_identity_up_to_similarity() {
        let disk = ring_disk(4, 16, 2.0);
        let uv = harmonic_parameterization(&disk, BoundaryShape::Circle);
        // Fit the similarity uv = s·R(θ)·p by matching the first
        // boundary vertex, then every vertex must agree: harmonic
        // coordinates reproduce linear functions on a flat mesh.
        let b0 = 1 + 3 * 16; // first vertex of the outer ring
        let p = disk.vertices[b0];
        let q = uv[b0];
        let scale = q.magnitude() / Vec2::new(p.x, p.y).magnitude();
        let rot = q.y.atan2(q.x) - p.y.atan2(p.x);
        let (s, c) = rot.sin_cos();
        for (vert, got) in disk.vertices.iter().zip(&uv) {
            let expect = Vec2::new(
                scale * (c * vert.x - s * vert.y),
                scale * (s * vert.x + c * vert.y),
            );
            assert!(
                (expect - *got).magnitude() < 1e-7,
                "harmonic map of flat disk is a similarity ({expect:?} vs {got:?})"
            );
        }
        // Square and Free targets produce injective maps: all UV
        // triangles keep positive orientation.
        for shape in [BoundaryShape::Square, BoundaryShape::Free] {
            let uv = harmonic_parameterization(&disk, shape);
            for &[a, b, c] in &disk.indices {
                let area = (uv[b] - uv[a]).cross(&(uv[c] - uv[a]));
                assert!(area > 0.0, "{shape:?} keeps orientation");
            }
        }
    }

    #[test]
    fn test_lscm_developable_is_conformal() {
        let sweep = 1.5 * std::f64::consts::PI;
        let m = open_cylinder(sweep, 2.0, 20, 10);
        // Pin two bottom corners to their isometric flattening.
        let uv = lscm(&m, [(0, Vec2::new(0.0, 0.0)), (19, Vec2::new(sweep, 0.0))]);
        let distortion = conformal_distortion(&m, &uv);
        for d in &distortion {
            assert!((d - 1.0).abs() < 0.01, "developable LSCM is conformal ({d})");
        }
        // The flattening is also an isometry up to global scale.
        for a in area_distortion(&m, &uv) {
            assert!((a - 1.0).abs() < 0.02, "area preserved up to scale ({a})");
        }
        // The u coordinate unrolls the angle: width ~ sweep, height ~ 2.
        let (mut ulo, mut uhi, mut vlo, mut vhi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for p in &uv {
            ulo = ulo.min(p.x);
            uhi = uhi.max(p.x);
            vlo = vlo.min(p.y);
            vhi = vhi.max(p.y);
        }
        assert!(((uhi - ulo) - sweep).abs() < 0.05, "unrolled width {}", uhi - ulo);
        assert!(((vhi - vlo) - 2.0).abs() < 0.05, "unrolled height {}", vhi - vlo);
    }

    #[test]
    fn test_lscm_flat_mesh_identity() {
        // LSCM of a flat disk with two pins at true positions
        // reproduces the plane exactly (zero conformal energy is
        // attainable).
        let disk = ring_disk(3, 12, 1.0);
        let i0 = 1 + 2 * 12; // inner-ring vertex on the +x axis
        let p0 = Vec2::new(disk.vertices[i0].x, disk.vertices[i0].y);
        let p1 = Vec2::new(disk.vertices[0].x, disk.vertices[0].y);
        let uv = lscm(&disk, [(i0, p0), (0, p1)]);
        for (vert, got) in disk.vertices.iter().zip(&uv) {
            let expect = Vec2::new(vert.x, vert.y);
            assert!(
                (expect - *got).magnitude() < 1e-7,
                "flat LSCM is the identity ({expect:?} vs {got:?})"
            );
        }
        let d = conformal_distortion(&disk, &uv);
        assert!(d.iter().all(|&x| (x - 1.0).abs() < 1e-9));
        let a = area_distortion(&disk, &uv);
        assert!(a.iter().all(|&x| (x - 1.0).abs() < 1e-9));
    }

    #[test]
    fn test_distortion_measures_detect_stretching() {
        let disk = ring_disk(2, 10, 1.0);
        // Anisotropic squash: u = x, v = y/2.
        let uv: Vec<Vec2> =
            disk.vertices.iter().map(|v| Vec2::new(v.x, 0.5 * v.y)).collect();
        let d = conformal_distortion(&disk, &uv);
        let max_d = d.iter().cloned().fold(0.0f64, f64::max);
        assert!((max_d - 2.0).abs() < 1e-9, "squash has distortion 2 ({max_d})");
        // Area distortion of a *uniform* scale is 1 everywhere.
        let uv2: Vec<Vec2> =
            disk.vertices.iter().map(|v| Vec2::new(3.0 * v.x, 3.0 * v.y)).collect();
        for a in area_distortion(&disk, &uv2) {
            assert!((a - 1.0).abs() < 1e-12);
        }
    }
}
