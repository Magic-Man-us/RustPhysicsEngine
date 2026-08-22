//! Convex hulls and polygon predicates.
//!
//! 2-D hull: Andrew's monotone chain (O(n log n)), CCW output.
//! 3-D hull: incremental visible-face (quickhull-style) algorithm
//! returning triangle index triples with outward-facing orientation.

use crate::math::Vec3;

/// Twice the signed area of triangle (o, a, b): > 0 for a CCW turn.
fn cross2(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Convex hull of 2-D points by monotone chain, returned in
/// counter-clockwise order without repetition of the first vertex.
/// Collinear boundary points are dropped. Fewer than 3 distinct points
/// return the distinct points themselves.
#[must_use]
pub fn convex_hull_2d(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup();
    let n = pts.len();
    if n < 3 {
        return pts;
    }
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(2 * n);
    // Lower hull.
    for &p in &pts {
        while hull.len() >= 2 && cross2(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    // Upper hull.
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower_len
            && cross2(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop(); // last point equals the first
    if hull.len() < 3 {
        // All points collinear: return the two extremes.
        return vec![pts[0], pts[n - 1]];
    }
    hull
}

/// Shoelace formula: positive for counter-clockwise vertex order.
///
/// # Panics
/// Panics if the polygon has fewer than 3 vertices.
#[must_use]
pub fn polygon_area_signed(poly: &[(f64, f64)]) -> f64 {
    assert!(poly.len() >= 3, "polygon_area_signed requires at least 3 vertices");
    let n = poly.len();
    let mut s = 0.0;
    for i in 0..n {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % n];
        s += x1 * y2 - x2 * y1;
    }
    0.5 * s
}

/// Even-odd (ray casting) point-in-polygon test; boundary points count
/// as inside up to floating-point tolerance.
///
/// # Panics
/// Panics if the polygon has fewer than 3 vertices.
#[must_use]
pub fn point_in_polygon(p: (f64, f64), poly: &[(f64, f64)]) -> bool {
    assert!(poly.len() >= 3, "point_in_polygon requires at least 3 vertices");
    let n = poly.len();
    let mut inside = false;
    let (px, py) = p;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        // Boundary check on this edge.
        let cross = (xj - xi) * (py - yi) - (px - xi) * (yj - yi);
        let within_x = px >= xi.min(xj) - 1e-12 && px <= xi.max(xj) + 1e-12;
        let within_y = py >= yi.min(yj) - 1e-12 && py <= yi.max(yj) + 1e-12;
        if cross.abs() < 1e-12 && within_x && within_y {
            return true;
        }
        if (yi > py) != (yj > py) {
            let x_int = xi + (py - yi) / (yj - yi) * (xj - xi);
            if px < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Convex hull of 3-D points as outward-oriented triangle index
/// triples, by the incremental visible-face (quickhull-style)
/// algorithm.
///
/// # Panics
/// Panics with fewer than 4 points or fully degenerate (coplanar)
/// input.
#[must_use]
pub fn convex_hull_3d(points: &[Vec3]) -> Vec<[usize; 3]> {
    let n = points.len();
    assert!(n >= 4, "convex_hull_3d requires at least 4 points");
    let sub = |a: Vec3, b: Vec3| Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z);
    let cross = |a: Vec3, b: Vec3| {
        Vec3::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    };
    let dot = |a: Vec3, b: Vec3| a.x * b.x + a.y * b.y + a.z * b.z;
    // Signed volume of tetrahedron (a, b, c, d): > 0 when d is on the
    // positive side of triangle (a, b, c).
    let orient = |a: usize, b: usize, c: usize, d: usize| -> f64 {
        let ab = sub(points[b], points[a]);
        let ac = sub(points[c], points[a]);
        let ad = sub(points[d], points[a]);
        dot(cross(ab, ac), ad)
    };

    // Initial simplex: two extreme points, then max-distance point from
    // the line, then max-distance point from the plane.
    let (mut i0, mut i1) = (0, 1);
    let mut best = -1.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = sub(points[i], points[j]);
            let dsq = dot(d, d);
            if dsq > best {
                best = dsq;
                i0 = i;
                i1 = j;
            }
        }
    }
    assert!(best > 0.0, "convex_hull_3d requires non-coincident points");
    let mut i2 = usize::MAX;
    best = 1e-20;
    for i in 0..n {
        if i == i0 || i == i1 {
            continue;
        }
        let c = cross(sub(points[i1], points[i0]), sub(points[i], points[i0]));
        let dsq = dot(c, c);
        if dsq > best {
            best = dsq;
            i2 = i;
        }
    }
    assert!(i2 != usize::MAX, "convex_hull_3d requires non-collinear points");
    let mut i3 = usize::MAX;
    best = 1e-12;
    for i in 0..n {
        if i == i0 || i == i1 || i == i2 {
            continue;
        }
        let v = orient(i0, i1, i2, i).abs();
        if v > best {
            best = v;
            i3 = i;
        }
    }
    assert!(i3 != usize::MAX, "convex_hull_3d requires non-coplanar points");

    // Orient the initial tetrahedron consistently (outward normals).
    let mut faces: Vec<[usize; 3]> = if orient(i0, i1, i2, i3) < 0.0 {
        vec![[i0, i1, i2], [i0, i3, i1], [i1, i3, i2], [i2, i3, i0]]
    } else {
        vec![[i0, i2, i1], [i0, i1, i3], [i1, i2, i3], [i2, i0, i3]]
    };

    const EPS: f64 = 1e-10;
    for p in 0..n {
        if p == i0 || p == i1 || p == i2 || p == i3 {
            continue;
        }
        // Faces visible from p (p strictly outside).
        let visible: Vec<usize> = (0..faces.len())
            .filter(|&f| orient(faces[f][0], faces[f][1], faces[f][2], p) > EPS)
            .collect();
        if visible.is_empty() {
            continue;
        }
        // Horizon: directed edges of visible faces whose reverse edge
        // does not belong to any visible face (consistent orientation
        // makes shared edges appear once per direction).
        use std::collections::HashSet;
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        for &f in &visible {
            let t = faces[f];
            edges.insert((t[0], t[1]));
            edges.insert((t[1], t[2]));
            edges.insert((t[2], t[0]));
        }
        let horizon: Vec<(usize, usize)> = edges
            .iter()
            .filter(|&&(a, b)| !edges.contains(&(b, a)))
            .copied()
            .collect();
        // Remove visible faces (descending indices), add cone from p.
        let mut vis_sorted = visible;
        vis_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for f in vis_sorted {
            faces.swap_remove(f);
        }
        for (a, b) in horizon {
            faces.push([a, b, p]);
        }
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hull_2d_square_with_interior() {
        let pts = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (1.0, 1.0),
            (0.5, 1.5),
        ];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 4);
        assert!(polygon_area_signed(&hull) > 0.0, "hull must be CCW");
        assert!((polygon_area_signed(&hull) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_hull_2d_degenerate() {
        assert_eq!(convex_hull_2d(&[(1.0, 1.0)]), vec![(1.0, 1.0)]);
        let collinear = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        let hull = convex_hull_2d(&collinear);
        assert_eq!(hull, vec![(0.0, 0.0), (3.0, 3.0)]);
    }

    #[test]
    fn test_polygon_area_signed_orientation() {
        let ccw = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!((polygon_area_signed(&ccw) - 1.0).abs() < 1e-15);
        let cw: Vec<(f64, f64)> = ccw.iter().rev().copied().collect();
        assert!((polygon_area_signed(&cw) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_point_in_polygon() {
        let poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
        assert!(point_in_polygon((2.0, 1.5), &poly));
        assert!(!point_in_polygon((5.0, 1.0), &poly));
        assert!(point_in_polygon((0.0, 1.0), &poly)); // boundary
        assert!(point_in_polygon((4.0, 3.0), &poly)); // vertex
        assert!(!point_in_polygon((-0.001, 1.0), &poly));
    }

    #[test]
    fn test_hull_3d_tetrahedron() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let faces = convex_hull_3d(&pts);
        assert_eq!(faces.len(), 4);
    }

    #[test]
    fn test_hull_3d_cube_with_interior_point() {
        let mut pts = Vec::new();
        for &x in &[0.0, 1.0] {
            for &y in &[0.0, 1.0] {
                for &z in &[0.0, 1.0] {
                    pts.push(Vec3::new(x, y, z));
                }
            }
        }
        pts.push(Vec3::new(0.5, 0.5, 0.5)); // interior
        let faces = convex_hull_3d(&pts);
        // Cube hull: 8 vertices, 12 triangles (Euler: F = 2V - 4).
        assert_eq!(faces.len(), 12);
        assert!(faces.iter().all(|t| t.iter().all(|&i| i < 8)), "interior point on hull");
    }

    #[test]
    #[should_panic(expected = "non-coplanar")]
    fn test_hull_3d_coplanar_panics() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let _ = convex_hull_3d(&pts);
    }
}
