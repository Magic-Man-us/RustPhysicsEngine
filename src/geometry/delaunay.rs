//! Delaunay triangulation and Voronoi diagrams in the plane.
//!
//! Triangulation: Bowyer-Watson incremental insertion with a
//! super-triangle. Voronoi cells: half-plane intersection of the
//! perpendicular bisectors (the dual definition), clipped to the
//! bounding box of the sites — robust for boundary cells.

/// Circumcircle of triangle (a, b, c): (center, radius).
///
/// # Panics
/// Panics if the points are collinear.
#[must_use]
pub fn circumcircle(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> ((f64, f64), f64) {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    assert!(d.abs() > 1e-14, "circumcircle requires non-collinear points");
    let a2 = a.0 * a.0 + a.1 * a.1;
    let b2 = b.0 * b.0 + b.1 * b.1;
    let c2 = c.0 * c.0 + c.1 * c.1;
    let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
    let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
    let r = ((a.0 - ux).powi(2) + (a.1 - uy).powi(2)).sqrt();
    ((ux, uy), r)
}

fn in_circumcircle(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let (center, r) = circumcircle(a, b, c);
    let dx = p.0 - center.0;
    let dy = p.1 - center.1;
    (dx * dx + dy * dy).sqrt() < r * (1.0 - 1e-12)
}

/// Delaunay triangulation by Bowyer-Watson insertion; returns triangle
/// index triples into `points`.
///
/// # Panics
/// Panics with fewer than 3 points or if all points are collinear.
#[must_use]
pub fn delaunay_2d(points: &[(f64, f64)]) -> Vec<[usize; 3]> {
    let n = points.len();
    assert!(n >= 3, "delaunay_2d requires at least 3 points");

    // Super-triangle enclosing all points.
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let d = (max_x - min_x).max(max_y - min_y).max(1.0) * 20.0;
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;
    // Indices n, n+1, n+2 refer to the super-triangle vertices.
    let all_point = |idx: usize| -> (f64, f64) {
        match idx.checked_sub(n) {
            None => points[idx],
            Some(0) => (mid_x - d, mid_y - d),
            Some(1) => (mid_x + d, mid_y - d),
            Some(_) => (mid_x, mid_y + d),
        }
    };

    let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];
    for p in 0..n {
        let pt = points[p];
        // Bad triangles: circumcircle contains the new point.
        let mut bad = Vec::new();
        for (t, tri) in triangles.iter().enumerate() {
            let (a, b, c) = (all_point(tri[0]), all_point(tri[1]), all_point(tri[2]));
            if in_circumcircle(pt, a, b, c) {
                bad.push(t);
            }
        }
        // Boundary polygon: edges of bad triangles not shared by two.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for &t in &bad {
            let tri = triangles[t];
            for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if let Some(pos) =
                    edges.iter().position(|&(a, b)| (a, b) == (e.1, e.0) || (a, b) == e)
                {
                    edges.remove(pos);
                } else {
                    edges.push(e);
                }
            }
        }
        for t in bad.iter().rev() {
            triangles.swap_remove(*t);
        }
        for (a, b) in edges {
            triangles.push([a, b, p]);
        }
    }
    // Drop triangles touching the super-triangle; canonicalize CCW.
    triangles.retain(|t| t.iter().all(|&i| i < n));
    for tri in triangles.iter_mut() {
        let (a, b, c) = (points[tri[0]], points[tri[1]], points[tri[2]]);
        let area2 = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if area2 < 0.0 {
            tri.swap(1, 2);
        }
    }
    triangles
}

/// Clips a convex polygon against the half-plane
/// {q : (q − mid)·n ≤ 0} (Sutherland-Hodgman step).
fn clip_half_plane(
    poly: &[(f64, f64)],
    mid: (f64, f64),
    normal: (f64, f64),
) -> Vec<(f64, f64)> {
    let side = |q: (f64, f64)| (q.0 - mid.0) * normal.0 + (q.1 - mid.1) * normal.1;
    let mut out = Vec::with_capacity(poly.len() + 1);
    let m = poly.len();
    for i in 0..m {
        let cur = poly[i];
        let next = poly[(i + 1) % m];
        let sc = side(cur);
        let sn = side(next);
        if sc <= 0.0 {
            out.push(cur);
        }
        if (sc < 0.0 && sn > 0.0) || (sc > 0.0 && sn < 0.0) {
            let t = sc / (sc - sn);
            out.push((cur.0 + t * (next.0 - cur.0), cur.1 + t * (next.1 - cur.1)));
        }
    }
    out
}

/// Voronoi cell polygons for each site, clipped to the sites' bounding
/// box (expanded by 10%). Cell i is the intersection of the half-planes
/// bounded by the perpendicular bisectors toward every other site —
/// the exact dual of the Delaunay triangulation.
///
/// # Panics
/// Panics with fewer than 2 sites.
#[must_use]
pub fn voronoi_cells_2d(points: &[(f64, f64)]) -> Vec<Vec<(f64, f64)>> {
    let n = points.len();
    assert!(n >= 2, "voronoi_cells_2d requires at least 2 sites");
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let pad_x = 0.1 * (max_x - min_x).max(1e-9);
    let pad_y = 0.1 * (max_y - min_y).max(1e-9);
    let bbox = vec![
        (min_x - pad_x, min_y - pad_y),
        (max_x + pad_x, min_y - pad_y),
        (max_x + pad_x, max_y + pad_y),
        (min_x - pad_x, max_y + pad_y),
    ];

    (0..n)
        .map(|i| {
            let mut cell = bbox.clone();
            for j in 0..n {
                if i == j || cell.is_empty() {
                    continue;
                }
                let mid = (
                    (points[i].0 + points[j].0) / 2.0,
                    (points[i].1 + points[j].1) / 2.0,
                );
                let normal = (points[j].0 - points[i].0, points[j].1 - points[i].1);
                cell = clip_half_plane(&cell, mid, normal);
            }
            cell
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::hull::{point_in_polygon, polygon_area_signed};

    #[test]
    fn test_circumcircle_right_triangle() {
        // Right triangle: circumcenter at the hypotenuse midpoint.
        let ((cx, cy), r) = circumcircle((0.0, 0.0), (2.0, 0.0), (0.0, 2.0));
        assert!((cx - 1.0).abs() < 1e-12 && (cy - 1.0).abs() < 1e-12);
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "non-collinear")]
    fn test_circumcircle_collinear_panics() {
        let _ = circumcircle((0.0, 0.0), (1.0, 1.0), (2.0, 2.0));
    }

    #[test]
    fn test_delaunay_square() {
        let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let tris = delaunay_2d(&pts);
        assert_eq!(tris.len(), 2);
        // Union of areas equals the square.
        let total: f64 = tris
            .iter()
            .map(|t| {
                polygon_area_signed(&[pts[t[0]], pts[t[1]], pts[t[2]]])
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_delaunay_empty_circumcircle() {
        let pts = [
            (0.0, 0.0),
            (2.0, 0.1),
            (1.1, 1.9),
            (-0.5, 1.2),
            (0.9, 0.8),
            (1.7, 1.1),
        ];
        let tris = delaunay_2d(&pts);
        for t in &tris {
            let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
            for (i, &p) in pts.iter().enumerate() {
                if t.contains(&i) {
                    continue;
                }
                assert!(
                    !in_circumcircle(p, a, b, c),
                    "point {i} inside circumcircle of {t:?}"
                );
            }
        }
    }

    #[test]
    fn test_voronoi_two_sites() {
        let cells = voronoi_cells_2d(&[(0.0, 0.0), (2.0, 0.0)]);
        assert_eq!(cells.len(), 2);
        // Each site lies inside its own cell; the bisector x = 1 splits them.
        assert!(cells[0].iter().all(|&(x, _)| x <= 1.0 + 1e-9));
        assert!(cells[1].iter().all(|&(x, _)| x >= 1.0 - 1e-9));
    }

    #[test]
    fn test_voronoi_sites_inside_their_cells() {
        let pts = [(0.0, 0.0), (2.0, 0.3), (1.0, 2.0), (-1.0, 1.4), (0.7, 0.9)];
        let cells = voronoi_cells_2d(&pts);
        for (site, cell) in pts.iter().zip(&cells) {
            assert!(cell.len() >= 3, "degenerate cell");
            assert!(point_in_polygon(*site, cell), "site outside its cell");
        }
    }
}
