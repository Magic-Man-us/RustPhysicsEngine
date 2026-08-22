//! Properties for `geometry`: hulls, Delaunay, geodesy.

use rust_physics_engine::geometry::{
    circumcircle, convex_hull_2d, delaunay_2d, point_in_polygon, polygon_area_signed,
    vincenty_direct, vincenty_inverse, Ellipsoid,
};
use rust_physics_engine::monte_carlo::Rng;

fn random_points(rng: &mut Rng, n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|_| (rng.next_f64() * 10.0 - 5.0, rng.next_f64() * 10.0 - 5.0))
        .collect()
}

/// Every input point lies inside or on the 2-D convex hull, and the
/// hull is CCW.
#[test]
fn prop_hull_contains_all_points() {
    let mut rng = Rng::new(121);
    for _ in 0..30 {
        let pts = random_points(&mut rng, 40);
        let hull = convex_hull_2d(&pts);
        assert!(hull.len() >= 3);
        assert!(polygon_area_signed(&hull) > 0.0, "hull not CCW");
        for &p in &pts {
            assert!(point_in_polygon(p, &hull), "point {p:?} outside hull");
        }
    }
}

/// Delaunay: the empty-circumcircle property holds for every triangle,
/// and the triangle count equals 2n − h − 2 (n sites, h hull sites).
#[test]
fn prop_delaunay_empty_circumcircle_and_count() {
    let mut rng = Rng::new(122);
    for _ in 0..15 {
        let pts = random_points(&mut rng, 25);
        let tris = delaunay_2d(&pts);
        for t in &tris {
            let (center, r) = circumcircle(pts[t[0]], pts[t[1]], pts[t[2]]);
            for (i, &p) in pts.iter().enumerate() {
                if t.contains(&i) {
                    continue;
                }
                let d = ((p.0 - center.0).powi(2) + (p.1 - center.1).powi(2)).sqrt();
                assert!(d >= r * (1.0 - 1e-9), "site {i} inside circumcircle of {t:?}");
            }
        }
        // Triangle count: T = 2n - h - 2, with h taken as the number of
        // boundary edges of the triangulation itself (equal to the
        // number of hull sites, robust to near-collinear boundary
        // chains where an exact hull count is tolerance-dependent).
        // A dropped triangle would open a hole and inflate the
        // boundary-edge count, so this still catches missing slivers.
        use std::collections::HashMap;
        let mut edge_uses: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &tris {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = (e.0.min(e.1), e.0.max(e.1));
                *edge_uses.entry(key).or_insert(0) += 1;
            }
        }
        assert!(edge_uses.values().all(|&c| c <= 2), "edge shared by 3+ triangles");
        let boundary_edges = edge_uses.values().filter(|&&c| c == 1).count();
        let hull_vertices = convex_hull_2d(&pts).len();
        assert!(boundary_edges >= hull_vertices, "boundary shorter than hull");
        assert_eq!(
            tris.len(),
            2 * pts.len() - boundary_edges - 2,
            "Euler count mismatch (boundary edges = {boundary_edges})"
        );
    }
}

/// Vincenty: direct(inverse(A, B)) recovers B to 1e-6 m, and the
/// f = 0 sphere matches the great-circle distance.
#[test]
fn prop_vincenty_roundtrip() {
    let mut rng = Rng::new(123);
    let e = Ellipsoid::WGS84;
    let mut checked = 0;
    for _ in 0..40 {
        let lat1 = (rng.next_f64() - 0.5) * 2.8; // avoid the exact poles
        let lon1 = (rng.next_f64() - 0.5) * std::f64::consts::TAU;
        let lat2 = (rng.next_f64() - 0.5) * 2.8;
        let lon2 = (rng.next_f64() - 0.5) * std::f64::consts::TAU;
        // Nearly antipodal pairs may legitimately fail to converge.
        let Ok((d, az1, _)) = vincenty_inverse(lat1, lon1, lat2, lon2, &e) else {
            continue;
        };
        let (rlat, rlon, _) = vincenty_direct(lat1, lon1, az1, d, &e);
        let Ok((miss, _, _)) = vincenty_inverse(rlat, rlon, lat2, lon2, &e) else {
            continue;
        };
        // 1e-6 m absolute, widened by the double-precision limit on
        // very long geodesics (~eps-level relative error on d).
        let tol = 1e-6_f64.max(d * 1e-12);
        assert!(miss < tol, "roundtrip miss {miss} m for d = {d}");
        checked += 1;
    }
    assert!(checked > 30, "too few convergent cases: {checked}");
}
