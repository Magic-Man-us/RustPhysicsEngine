//! Procedural mesh generators. Closed shapes are watertight (shared
//! seam vertices, no duplicates) with outward-facing counterclockwise
//! winding.

use crate::math::{Vec2, Vec3};
use crate::mesh::Mesh;
use crate::spatial::primitives::{Polygon2, Polyline};

/// Latitude/longitude sphere: `rings` latitude bands, `segments`
/// meridians, poles as single vertices. Closed manifold (Euler
/// characteristic 2).
///
/// # Panics
/// Panics unless `radius > 0`, `segments >= 3`, and `rings >= 2`.
#[must_use]
pub fn uv_sphere(radius: f64, segments: usize, rings: usize) -> Mesh {
    assert!(radius > 0.0, "uv_sphere requires radius > 0");
    assert!(segments >= 3 && rings >= 2, "uv_sphere requires segments >= 3, rings >= 2");
    let mut vertices = vec![Vec3::new(0.0, radius, 0.0)];
    for i in 1..rings {
        let theta = std::f64::consts::PI * i as f64 / rings as f64;
        for j in 0..segments {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
            vertices.push(Vec3::new(
                radius * theta.sin() * phi.cos(),
                radius * theta.cos(),
                radius * theta.sin() * phi.sin(),
            ));
        }
    }
    vertices.push(Vec3::new(0.0, -radius, 0.0));
    let south = vertices.len() - 1;
    let ring = |i: usize, j: usize| 1 + (i - 1) * segments + j % segments;
    let mut indices = Vec::new();
    for j in 0..segments {
        indices.push([0, ring(1, j + 1), ring(1, j)]);
        indices.push([south, ring(rings - 1, j), ring(rings - 1, j + 1)]);
    }
    for i in 1..rings - 1 {
        for j in 0..segments {
            let (a, b) = (ring(i, j), ring(i, j + 1));
            let (d, c) = (ring(i + 1, j), ring(i + 1, j + 1));
            indices.push([a, b, d]);
            indices.push([b, c, d]);
        }
    }
    Mesh::new(vertices, indices).expect("uv_sphere indices are valid")
}

/// Geodesic sphere: an icosahedron subdivided `subdivisions` times,
/// vertices projected to the sphere. Closed manifold.
///
/// # Panics
/// Panics unless `radius > 0`.
#[must_use]
pub fn icosphere(radius: f64, subdivisions: usize) -> Mesh {
    assert!(radius > 0.0, "icosphere requires radius > 0");
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
    let mut vertices: Vec<Vec3> = raw
        .iter()
        .map(|&(x, y, z)| Vec3::new(x, y, z).normalized() * radius)
        .collect();
    let mut indices: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut cache: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let mut midpoint = |a: usize, b: usize, vertices: &mut Vec<Vec3>| {
            let key = (a.min(b), a.max(b));
            *cache.entry(key).or_insert_with(|| {
                let m = ((vertices[a] + vertices[b]) * 0.5).normalized() * radius;
                vertices.push(m);
                vertices.len() - 1
            })
        };
        let mut next = Vec::with_capacity(indices.len() * 4);
        for [a, b, c] in indices {
            let ab = midpoint(a, b, &mut vertices);
            let bc = midpoint(b, c, &mut vertices);
            let ca = midpoint(c, a, &mut vertices);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        indices = next;
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Axis-aligned box with the given half extents: 8 shared vertices, 12
/// triangles. Closed manifold.
///
/// # Panics
/// Panics unless every half extent is positive.
#[must_use]
pub fn box_mesh(half: Vec3) -> Mesh {
    assert!(
        half.x > 0.0 && half.y > 0.0 && half.z > 0.0,
        "box_mesh requires positive half extents"
    );
    let vertices: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { -half.x } else { half.x },
                if i & 2 == 0 { -half.y } else { half.y },
                if i & 4 == 0 { -half.z } else { half.z },
            )
        })
        .collect();
    // Faces counterclockwise seen from outside.
    let quads: [[usize; 4]; 6] = [
        [0, 2, 3, 1], // z = -hz
        [4, 5, 7, 6], // z = +hz
        [0, 1, 5, 4], // y = -hy
        [3, 2, 6, 7], // y = +hy
        [0, 4, 6, 2], // x = -hx
        [1, 3, 7, 5], // x = +hx
    ];
    let mut indices = Vec::with_capacity(12);
    for [a, b, c, d] in quads {
        indices.push([a, b, c]);
        indices.push([a, c, d]);
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Cylinder along the y axis, centered at the origin, spanning
/// `[-height/2, height/2]`. With `capped`, both ends are closed by
/// fans sharing the rim vertices (closed manifold).
///
/// # Panics
/// Panics unless `radius > 0`, `height > 0`, `segments >= 3`.
#[must_use]
pub fn cylinder(radius: f64, height: f64, segments: usize, capped: bool) -> Mesh {
    assert!(radius > 0.0 && height > 0.0, "cylinder requires positive radius and height");
    assert!(segments >= 3, "cylinder requires segments >= 3");
    let h = height / 2.0;
    let mut vertices = Vec::new();
    for &y in &[-h, h] {
        for j in 0..segments {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
            vertices.push(Vec3::new(radius * phi.cos(), y, radius * phi.sin()));
        }
    }
    let bot = |j: usize| j % segments;
    let top = |j: usize| segments + j % segments;
    let mut indices = Vec::new();
    for j in 0..segments {
        indices.push([bot(j), top(j), top(j + 1)]);
        indices.push([bot(j), top(j + 1), bot(j + 1)]);
    }
    if capped {
        let cb = vertices.len();
        vertices.push(Vec3::new(0.0, -h, 0.0));
        let ct = vertices.len();
        vertices.push(Vec3::new(0.0, h, 0.0));
        for j in 0..segments {
            indices.push([cb, bot(j), bot(j + 1)]);
            indices.push([ct, top(j + 1), top(j)]);
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Cone with its base disk in the y = 0 plane (centered at the origin)
/// and apex at `(0, height, 0)`. Closed manifold.
///
/// # Panics
/// Panics unless `radius > 0`, `height > 0`, `segments >= 3`.
#[must_use]
pub fn cone(radius: f64, height: f64, segments: usize) -> Mesh {
    assert!(radius > 0.0 && height > 0.0, "cone requires positive radius and height");
    assert!(segments >= 3, "cone requires segments >= 3");
    let mut vertices: Vec<Vec3> = (0..segments)
        .map(|j| {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
            Vec3::new(radius * phi.cos(), 0.0, radius * phi.sin())
        })
        .collect();
    let apex = vertices.len();
    vertices.push(Vec3::new(0.0, height, 0.0));
    let center = vertices.len();
    vertices.push(Vec3::ZERO);
    let mut indices = Vec::new();
    for j in 0..segments {
        let (a, b) = (j, (j + 1) % segments);
        indices.push([apex, b, a]);
        indices.push([center, a, b]);
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Torus around the y axis: tube of radius `minor` swept along a
/// circle of radius `major` in the xz plane. Closed manifold (Euler
/// characteristic 0).
///
/// # Panics
/// Panics unless `0 < minor < major` and both segment counts are >= 3.
#[must_use]
pub fn torus(major: f64, minor: f64, major_segs: usize, minor_segs: usize) -> Mesh {
    assert!(minor > 0.0 && major > minor, "torus requires 0 < minor < major");
    assert!(major_segs >= 3 && minor_segs >= 3, "torus requires >= 3 segments each way");
    let mut vertices = Vec::with_capacity(major_segs * minor_segs);
    for i in 0..major_segs {
        let u = 2.0 * std::f64::consts::PI * i as f64 / major_segs as f64;
        for j in 0..minor_segs {
            let v = 2.0 * std::f64::consts::PI * j as f64 / minor_segs as f64;
            let r = major + minor * v.cos();
            vertices.push(Vec3::new(r * u.cos(), minor * v.sin(), r * u.sin()));
        }
    }
    let at = |i: usize, j: usize| (i % major_segs) * minor_segs + j % minor_segs;
    let mut indices = Vec::new();
    for i in 0..major_segs {
        for j in 0..minor_segs {
            // Quad ordered so normals face away from the tube center.
            let a = at(i, j);
            let b = at(i, j + 1);
            let c = at(i + 1, j + 1);
            let d = at(i + 1, j);
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Flat grid in the xz plane centered at the origin, normals facing
/// +y, `nx` by `nz` cells. Open (has a boundary).
///
/// # Panics
/// Panics unless `width > 0`, `depth > 0`, `nx >= 1`, `nz >= 1`.
#[must_use]
pub fn plane_grid(width: f64, depth: f64, nx: usize, nz: usize) -> Mesh {
    assert!(width > 0.0 && depth > 0.0, "plane_grid requires positive size");
    assert!(nx >= 1 && nz >= 1, "plane_grid requires at least one cell per axis");
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
    let mut indices = Vec::new();
    for j in 0..nz {
        for i in 0..nx {
            indices.push([at(i, j), at(i, j + 1), at(i + 1, j + 1)]);
            indices.push([at(i, j), at(i + 1, j + 1), at(i + 1, j)]);
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Capsule along the y axis: a cylinder of length `height` between two
/// hemispherical caps of the given radius (`rings` latitude bands per
/// hemisphere). Closed manifold.
///
/// # Panics
/// Panics unless `radius > 0`, `height >= 0`, `segments >= 3`,
/// `rings >= 1`.
#[must_use]
pub fn capsule(radius: f64, height: f64, segments: usize, rings: usize) -> Mesh {
    assert!(radius > 0.0 && height >= 0.0, "capsule requires radius > 0, height >= 0");
    assert!(segments >= 3 && rings >= 1, "capsule requires segments >= 3, rings >= 1");
    let h = height / 2.0;
    let mut vertices = vec![Vec3::new(0.0, radius + h, 0.0)];
    // Northern hemisphere rings (equator last), then southern rings
    // (equator first), mirrored.
    for i in 1..=rings {
        let theta = std::f64::consts::FRAC_PI_2 * i as f64 / rings as f64;
        for j in 0..segments {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
            vertices.push(Vec3::new(
                radius * theta.sin() * phi.cos(),
                h + radius * theta.cos(),
                radius * theta.sin() * phi.sin(),
            ));
        }
    }
    for i in 0..rings {
        let theta = std::f64::consts::FRAC_PI_2 * (1.0 - i as f64 / rings as f64);
        for j in 0..segments {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
            vertices.push(Vec3::new(
                radius * theta.sin() * phi.cos(),
                -h - radius * theta.cos(),
                radius * theta.sin() * phi.sin(),
            ));
        }
    }
    vertices.push(Vec3::new(0.0, -radius - h, 0.0));
    let south = vertices.len() - 1;
    let nrings = 2 * rings;
    let ring = |i: usize, j: usize| 1 + (i - 1) * segments + j % segments;
    let mut indices = Vec::new();
    for j in 0..segments {
        indices.push([0, ring(1, j + 1), ring(1, j)]);
        indices.push([south, ring(nrings, j), ring(nrings, j + 1)]);
    }
    for i in 1..nrings {
        for j in 0..segments {
            let (a, b) = (ring(i, j), ring(i, j + 1));
            let (d, c) = (ring(i + 1, j), ring(i + 1, j + 1));
            indices.push([a, b, d]);
            indices.push([b, c, d]);
        }
    }
    Mesh::new(vertices, indices).expect("capsule indices are valid")
}

/// Flat disk in the y = 0 plane centered at the origin, normal +y.
/// Open (has a boundary).
///
/// # Panics
/// Panics unless `radius > 0` and `segments >= 3`.
#[must_use]
pub fn disk(radius: f64, segments: usize) -> Mesh {
    assert!(radius > 0.0, "disk requires radius > 0");
    assert!(segments >= 3, "disk requires segments >= 3");
    let mut vertices = vec![Vec3::ZERO];
    for j in 0..segments {
        let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
        vertices.push(Vec3::new(radius * phi.cos(), 0.0, radius * phi.sin()));
    }
    let mut indices = Vec::new();
    for j in 0..segments {
        indices.push([0, 1 + (j + 1) % segments, 1 + j]);
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Tube of the given radius swept along a polyline using parallel
/// transport frames (rotation-minimizing, so the tube does not twist).
/// A closed path joins the last ring back to the first; an open path
/// leaves the ends uncapped.
///
/// # Panics
/// Panics unless `radius > 0`, `segments >= 3`, and the path has at
/// least two points with nonzero consecutive tangents.
#[must_use]
pub fn tube_along_polyline(path: &Polyline, radius: f64, segments: usize) -> Mesh {
    assert!(radius > 0.0 && segments >= 3, "tube requires radius > 0, segments >= 3");
    let pts = &path.points;
    assert!(pts.len() >= 2, "tube requires at least two path points");
    let n = pts.len();
    // Tangent at each point (central where possible).
    let tangent = |i: usize| -> Vec3 {
        let t = if path.closed {
            pts[(i + 1) % n] - pts[(i + n - 1) % n]
        } else if i == 0 {
            pts[1] - pts[0]
        } else if i == n - 1 {
            pts[n - 1] - pts[n - 2]
        } else {
            pts[i + 1] - pts[i - 1]
        };
        assert!(t.magnitude() > 0.0, "tube path has coincident points");
        t.normalized()
    };
    let t0 = tangent(0);
    // Any vector not parallel to t0 seeds the first normal.
    let seed = if t0.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let mut normal = (seed - t0 * seed.dot(&t0)).normalized();
    let mut vertices = Vec::with_capacity(n * segments);
    let mut prev_t = t0;
    for (i, &center) in pts.iter().enumerate() {
        let t = tangent(i);
        // Parallel transport: rotate the frame by the rotation taking
        // prev_t to t (Rodrigues about their cross product).
        let axis = prev_t.cross(&t);
        let s = axis.magnitude();
        let c = prev_t.dot(&t).clamp(-1.0, 1.0);
        if s > 1e-12 {
            let a = axis * (1.0 / s);
            let angle = s.atan2(c);
            let (sa, ca) = angle.sin_cos();
            normal = normal * ca + a.cross(&normal) * sa + a * (a.dot(&normal) * (1.0 - ca));
        }
        normal = (normal - t * normal.dot(&t)).normalized();
        let binormal = t.cross(&normal);
        for j in 0..segments {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
            vertices.push(center + normal * (radius * phi.cos()) + binormal * (radius * phi.sin()));
        }
        prev_t = t;
    }
    let at = |i: usize, j: usize| (i % n) * segments + j % segments;
    let spans = if path.closed { n } else { n - 1 };
    let mut indices = Vec::new();
    for i in 0..spans {
        for j in 0..segments {
            let a = at(i, j);
            let b = at(i, j + 1);
            let c = at(i + 1, j + 1);
            let d = at(i + 1, j);
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Ear-clipping triangulation of a simple counterclockwise polygon
/// (indices into `pts`).
fn ear_clip(pts: &[Vec2]) -> Vec<[usize; 3]> {
    let mut remaining: Vec<usize> = (0..pts.len()).collect();
    let mut tris = Vec::new();
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).cross(&(b - o));
    while remaining.len() > 3 {
        let m = remaining.len();
        let mut clipped = false;
        for k in 0..m {
            let (ip, ic, inx) = (remaining[(k + m - 1) % m], remaining[k], remaining[(k + 1) % m]);
            let (p, c, nx) = (pts[ip], pts[ic], pts[inx]);
            if cross(p, c, nx) <= 0.0 {
                continue; // reflex or collinear corner
            }
            let contains_other = remaining.iter().any(|&j| {
                if j == ip || j == ic || j == inx {
                    return false;
                }
                let q = pts[j];
                cross(p, c, q) >= 0.0 && cross(c, nx, q) >= 0.0 && cross(nx, p, q) >= 0.0
            });
            if !contains_other {
                tris.push([ip, ic, inx]);
                remaining.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Numerically stuck (e.g. collinear runs): clip the first
            // strictly convex corner regardless.
            let k = (0..m)
                .find(|&k| {
                    cross(
                        pts[remaining[(k + m - 1) % m]],
                        pts[remaining[k]],
                        pts[remaining[(k + 1) % m]],
                    ) > 0.0
                })
                .unwrap_or(0);
            tris.push([
                remaining[(k + m - 1) % m],
                remaining[k],
                remaining[(k + 1) % m],
            ]);
            remaining.remove(k);
        }
    }
    tris.push([remaining[0], remaining[1], remaining[2]]);
    tris
}

/// Extrudes a simple polygon (in the xy plane) along +z from z = 0 to
/// z = `height`, with ear-clipped caps. Closed manifold for a simple
/// polygon. Clockwise input is treated as its counterclockwise
/// reversal.
///
/// # Panics
/// Panics unless the polygon has at least 3 vertices and
/// `height > 0`.
#[must_use]
pub fn extrude_polygon(poly: &Polygon2, height: f64) -> Mesh {
    assert!(poly.vertices.len() >= 3, "extrude_polygon requires >= 3 vertices");
    assert!(height > 0.0, "extrude_polygon requires height > 0");
    let mut pts = poly.vertices.clone();
    if Polygon2::new(pts.clone()).area_signed() < 0.0 {
        pts.reverse();
    }
    let n = pts.len();
    let mut vertices = Vec::with_capacity(2 * n);
    for p in &pts {
        vertices.push(Vec3::new(p.x, p.y, 0.0));
    }
    for p in &pts {
        vertices.push(Vec3::new(p.x, p.y, height));
    }
    let mut indices = Vec::new();
    // Sides: outward for a counterclockwise polygon.
    for i in 0..n {
        let j = (i + 1) % n;
        indices.push([i, j, n + j]);
        indices.push([i, n + j, n + i]);
    }
    // Caps: top faces +z (counterclockwise in the plane), bottom
    // reversed.
    for [a, b, c] in ear_clip(&pts) {
        indices.push([n + a, n + b, n + c]);
        indices.push([a, c, b]);
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Revolves a profile polyline around the y axis. Each profile point
/// `(x, y)` gives radius `x` at height `y`; points with `x == 0`
/// become poles. The profile should ascend in y for outward normals;
/// a profile that starts and ends on the axis yields a closed
/// manifold.
///
/// # Panics
/// Panics unless the profile has >= 2 points, `segments >= 3`, and no
/// profile radius is negative.
#[must_use]
pub fn revolve_profile(profile: &[Vec2], segments: usize) -> Mesh {
    assert!(profile.len() >= 2, "revolve_profile requires >= 2 profile points");
    assert!(segments >= 3, "revolve_profile requires segments >= 3");
    assert!(profile.iter().all(|p| p.x >= 0.0), "profile radii must be nonnegative");
    let is_pole = |p: &Vec2| p.x == 0.0;
    // First vertex index of each profile row (a pole row has 1 vertex).
    let mut row_start = Vec::with_capacity(profile.len());
    let mut vertices = Vec::new();
    for p in profile {
        row_start.push(vertices.len());
        if is_pole(p) {
            vertices.push(Vec3::new(0.0, p.y, 0.0));
        } else {
            for j in 0..segments {
                let phi = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
                vertices.push(Vec3::new(p.x * phi.cos(), p.y, p.x * phi.sin()));
            }
        }
    }
    let mut indices = Vec::new();
    for i in 0..profile.len() - 1 {
        let (lo, hi) = (&profile[i], &profile[i + 1]);
        let (rl, rh) = (row_start[i], row_start[i + 1]);
        match (is_pole(lo), is_pole(hi)) {
            (false, false) => {
                for j in 0..segments {
                    let a = rl + j;
                    let b = rh + j;
                    let c = rh + (j + 1) % segments;
                    let d = rl + (j + 1) % segments;
                    indices.push([a, b, c]);
                    indices.push([a, c, d]);
                }
            }
            (true, false) => {
                // Bottom pole fan.
                for j in 0..segments {
                    indices.push([rl, rh + j, rh + (j + 1) % segments]);
                }
            }
            (false, true) => {
                // Top pole fan.
                for j in 0..segments {
                    indices.push([rh, rl + (j + 1) % segments, rl + j]);
                }
            }
            (true, true) => {}
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Height field surface: vertex `(i, j)` sits at
/// `(i * dx, heights[j * nx + i], j * dz)`, triangles facing +y for
/// positive `dx`, `dz`. Open.
///
/// # Panics
/// Panics unless `nx >= 2`, `nz >= 2`,
/// `heights.len() == nx * nz`, and `dx, dz > 0`.
#[must_use]
pub fn heightfield(heights: &[f64], nx: usize, nz: usize, dx: f64, dz: f64) -> Mesh {
    assert!(nx >= 2 && nz >= 2, "heightfield requires nx, nz >= 2");
    assert_eq!(heights.len(), nx * nz, "heightfield requires nx * nz samples");
    assert!(dx > 0.0 && dz > 0.0, "heightfield requires positive spacing");
    let mut vertices = Vec::with_capacity(nx * nz);
    for j in 0..nz {
        for i in 0..nx {
            vertices.push(Vec3::new(i as f64 * dx, heights[j * nx + i], j as f64 * dz));
        }
    }
    let at = |i: usize, j: usize| j * nx + i;
    let mut indices = Vec::new();
    for j in 0..nz - 1 {
        for i in 0..nx - 1 {
            indices.push([at(i, j), at(i, j + 1), at(i + 1, j + 1)]);
            indices.push([at(i, j), at(i + 1, j + 1), at(i + 1, j)]);
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Samples a parametric surface on an `nu` by `nv` cell grid.
/// `close_u`/`close_v` wrap the respective direction (the last row of
/// samples is omitted and faces reference the first, so periodic
/// surfaces come out watertight). Triangles are wound so normals
/// follow `∂f/∂u × ∂f/∂v`.
///
/// # Panics
/// Panics unless `nu >= 1`, `nv >= 1` (>= 3 for a closed direction)
/// and each range is nonempty.
#[must_use]
pub fn from_parametric(
    f: &dyn Fn(f64, f64) -> Vec3,
    u_range: (f64, f64),
    v_range: (f64, f64),
    nu: usize,
    nv: usize,
    close_u: bool,
    close_v: bool,
) -> Mesh {
    assert!(nu >= 1 && nv >= 1, "from_parametric requires nu, nv >= 1");
    assert!(u_range.1 > u_range.0 && v_range.1 > v_range.0, "empty parameter range");
    assert!(!close_u || nu >= 3, "closed u direction requires >= 3 cells");
    assert!(!close_v || nv >= 3, "closed v direction requires >= 3 cells");
    let su = if close_u { nu } else { nu + 1 };
    let sv = if close_v { nv } else { nv + 1 };
    let mut vertices = Vec::with_capacity(su * sv);
    for i in 0..su {
        let u = u_range.0 + (u_range.1 - u_range.0) * i as f64 / nu as f64;
        for j in 0..sv {
            let v = v_range.0 + (v_range.1 - v_range.0) * j as f64 / nv as f64;
            vertices.push(f(u, v));
        }
    }
    let at = |i: usize, j: usize| (i % su) * sv + j % sv;
    let mut indices = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let a = at(i, j);
            let b = at(i + 1, j);
            let c = at(i + 1, j + 1);
            let d = at(i, j + 1);
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }
    Mesh { vertices, indices, normals: None, uvs: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::volume_sphere;

    fn euler(m: &Mesh) -> i64 {
        m.vertices.len() as i64 - m.edges().len() as i64 + m.indices.len() as i64
    }

    fn assert_closed(m: &Mesh) {
        // Every undirected edge must appear in exactly two faces, once
        // per direction.
        let mut counts: std::collections::HashMap<(usize, usize), (usize, usize)> =
            std::collections::HashMap::new();
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

    #[test]
    fn test_spheres_closed_and_accurate() {
        for m in [uv_sphere(1.5, 32, 24), icosphere(1.5, 3)] {
            assert_closed(&m);
            assert_eq!(euler(&m), 2);
            let v = m.volume();
            assert!(v > 0.0 && (v - volume_sphere(1.5)).abs() / volume_sphere(1.5) < 0.02);
            for vert in &m.vertices {
                assert!((vert.magnitude() - 1.5).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_box_exact() {
        let m = box_mesh(Vec3::new(0.5, 1.0, 2.0));
        assert_closed(&m);
        assert_eq!(euler(&m), 2);
        assert!((m.volume() - 8.0 * 0.5 * 1.0 * 2.0).abs() < 1e-12);
        let (hx, hy, hz) = (0.5, 1.0, 2.0);
        let area = 8.0 * (hx * hy + hy * hz + hx * hz);
        assert!((m.surface_area() - area).abs() < 1e-12);
    }

    #[test]
    fn test_cylinder_cone_capsule_closed() {
        let c = cylinder(1.0, 2.0, 32, true);
        assert_closed(&c);
        assert_eq!(euler(&c), 2);
        let analytic = std::f64::consts::PI * 2.0; // r^2 h pi
        assert!((c.volume() - analytic).abs() / analytic < 0.01);

        let open = cylinder(1.0, 2.0, 16, false);
        assert_eq!(open.indices.len(), 32);

        let k = cone(1.0, 3.0, 48);
        assert_closed(&k);
        assert_eq!(euler(&k), 2);
        let analytic = std::f64::consts::PI / 3.0 * 3.0;
        assert!((k.volume() - analytic).abs() / analytic < 0.01);

        let cap = capsule(0.5, 2.0, 32, 12);
        assert_closed(&cap);
        assert_eq!(euler(&cap), 2);
        let analytic = std::f64::consts::PI * 0.25 * 2.0 + volume_sphere(0.5);
        assert!((cap.volume() - analytic).abs() / analytic < 0.02);
    }

    #[test]
    fn test_torus_genus_one() {
        let m = torus(2.0, 0.5, 48, 24);
        assert_closed(&m);
        assert_eq!(euler(&m), 0);
        // V = 2 pi^2 R r^2.
        let analytic = 2.0 * std::f64::consts::PI.powi(2) * 2.0 * 0.25;
        assert!((m.volume() - analytic).abs() / analytic < 0.02);
    }

    #[test]
    fn test_plane_disk_heightfield_open() {
        let p = plane_grid(2.0, 3.0, 4, 5);
        assert_eq!(p.vertices.len(), 30);
        assert_eq!(p.indices.len(), 40);
        assert!((p.surface_area() - 6.0).abs() < 1e-12);
        for t in p.triangles() {
            assert!(t.normal().y > 0.99);
        }
        let d = disk(2.0, 64);
        let analytic = std::f64::consts::PI * 4.0;
        assert!((d.surface_area() - analytic).abs() / analytic < 0.01);
        for t in d.triangles() {
            assert!(t.normal().y > 0.99);
        }
        let h = heightfield(&[0.0; 12], 4, 3, 0.5, 0.5);
        assert_eq!(h.vertices.len(), 12);
        assert!((h.surface_area() - 1.5 * 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_extrude_square_matches_box() {
        let poly = Polygon2::new(vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ]);
        let m = extrude_polygon(&poly, 2.0);
        assert_closed(&m);
        assert_eq!(euler(&m), 2);
        assert!((m.volume() - 8.0).abs() < 1e-12);
        // Clockwise input gives the same solid.
        let mut cw = poly.clone();
        cw.reverse();
        let m2 = extrude_polygon(&cw, 2.0);
        assert!((m2.volume() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_extrude_nonconvex() {
        // L-shape.
        let poly = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ]);
        let m = extrude_polygon(&poly, 1.0);
        assert_closed(&m);
        assert!((m.volume() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_revolve_sphere_profile() {
        let n = 32;
        let profile: Vec<Vec2> = (0..=n)
            .map(|i| {
                let t = std::f64::consts::PI * i as f64 / n as f64;
                // Force exact poles so the end rows collapse.
                let r = if i == 0 || i == n { 0.0 } else { t.sin() };
                Vec2::new(r, -t.cos())
            })
            .collect();
        let m = revolve_profile(&profile, 48);
        assert_closed(&m);
        assert_eq!(euler(&m), 2);
        assert!((m.volume() - volume_sphere(1.0)).abs() / volume_sphere(1.0) < 0.01);
    }

    #[test]
    fn test_tube_along_polyline() {
        let path = Polyline {
            points: (0..20).map(|i| Vec3::new(i as f64 * 0.3, (i as f64 * 0.4).sin(), 0.0)).collect(),
            closed: false,
        };
        let m = tube_along_polyline(&path, 0.1, 12);
        assert_eq!(m.vertices.len(), 20 * 12);
        assert_eq!(m.indices.len(), 19 * 12 * 2);
        // Rings keep their radius: every vertex lies within radius of
        // the path.
        for v in &m.vertices {
            let d = path
                .points
                .iter()
                .map(|p| p.distance_to(v))
                .fold(f64::INFINITY, f64::min);
            assert!(d < 0.1 + 1e-9);
        }
        // Closed loop tube is watertight.
        let ring_path = Polyline {
            points: (0..24)
                .map(|i| {
                    let a = 2.0 * std::f64::consts::PI * i as f64 / 24.0;
                    Vec3::new(a.cos(), 0.0, a.sin())
                })
                .collect(),
            closed: true,
        };
        let ring = tube_along_polyline(&ring_path, 0.2, 12);
        assert_closed(&ring);
        assert_eq!(euler(&ring), 0);
        assert!(ring.volume() > 0.0);
    }

    #[test]
    fn test_from_parametric_torus() {
        let f = |v: f64, u: f64| {
            let r = 2.0 + 0.5 * v.cos();
            Vec3::new(r * u.cos(), 0.5 * v.sin(), r * u.sin())
        };
        let tau = 2.0 * std::f64::consts::PI;
        let m = from_parametric(&f, (0.0, tau), (0.0, tau), 24, 48, true, true);
        assert_closed(&m);
        assert_eq!(euler(&m), 0);
        let analytic = 2.0 * std::f64::consts::PI.powi(2) * 2.0 * 0.25;
        assert!((m.volume() - analytic).abs() / analytic < 0.02);
    }
}
