//! Properties for `mesh`: mass properties of generated solids, OBJ
//! round-trips, and isosurface extraction invariants.

use rust_physics_engine::geometry::volume_sphere;
use rust_physics_engine::math::{Vec2, Vec3};
use rust_physics_engine::mesh::generate::{
    box_mesh, capsule, cone, cylinder, extrude_polygon, icosphere, torus, uv_sphere,
};
use rust_physics_engine::mesh::isosurface::{
    marching_cubes, marching_squares, marching_squares_polylines, marching_tetrahedra,
    ScalarField2, ScalarField3,
};
use rust_physics_engine::mesh::Mesh;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::spatial::primitives::{Polygon2, Rect};
use rust_physics_engine::spatial::sdf::sd_sphere;
use rust_physics_engine::spatial::Aabb;
use std::collections::HashMap;

fn euler_characteristic(m: &Mesh) -> i64 {
    m.vertices.len() as i64 - m.edges().len() as i64 + m.indices.len() as i64
}

/// Every undirected edge is used by exactly two faces, once per
/// direction (closed, consistently oriented 2-manifold).
fn assert_watertight(m: &Mesh, what: &str) {
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
        assert_eq!((fwd, rev), (1, 1), "{what}: edge {edge:?} not manifold");
    }
}

#[test]
fn prop_icosphere_volume_and_area_converge() {
    let r = 1.3;
    let m = icosphere(r, 4);
    let vol = volume_sphere(r);
    assert!((m.volume() - vol).abs() / vol < 0.01, "icosphere volume off: {}", m.volume());
    let area = 4.0 * std::f64::consts::PI * r * r;
    assert!((m.surface_area() - area).abs() / area < 0.01);
    // Refinement monotonically improves the volume.
    let mut prev = f64::INFINITY;
    for sub in 0..4 {
        let err = (icosphere(r, sub).volume() - vol).abs();
        assert!(err < prev);
        prev = err;
    }
}

#[test]
fn prop_random_box_inertia_matches_closed_form() {
    let mut rng = Rng::new(901);
    for _ in 0..20 {
        let half = Vec3::new(
            0.2 + rng.next_f64() * 2.0,
            0.2 + rng.next_f64() * 2.0,
            0.2 + rng.next_f64() * 2.0,
        );
        let rho = 0.5 + rng.next_f64() * 3.0;
        let m = box_mesh(half);
        let mass = rho * 8.0 * half.x * half.y * half.z;
        let i = m.inertia_tensor(rho);
        let expect = [
            mass / 3.0 * (half.y * half.y + half.z * half.z),
            mass / 3.0 * (half.x * half.x + half.z * half.z),
            mass / 3.0 * (half.x * half.x + half.y * half.y),
        ];
        for r in 0..3 {
            for s in 0..3 {
                let want = if r == s { expect[r] } else { 0.0 };
                assert!(
                    (i.data[r][s] - want).abs() < 1e-9 * mass.max(1.0),
                    "I[{r}][{s}] = {} want {want}",
                    i.data[r][s]
                );
            }
        }
    }
}

#[test]
fn prop_generators_closed_with_outward_normals() {
    // (mesh, expected Euler characteristic)
    let solids: Vec<(Mesh, i64, &str)> = vec![
        (uv_sphere(1.0, 20, 12), 2, "uv_sphere"),
        (icosphere(1.0, 2), 2, "icosphere"),
        (box_mesh(Vec3::new(1.0, 0.7, 0.4)), 2, "box"),
        (cylinder(0.8, 2.0, 24, true), 2, "cylinder"),
        (cone(1.0, 1.5, 24), 2, "cone"),
        (capsule(0.5, 1.0, 16, 6), 2, "capsule"),
        (torus(2.0, 0.6, 32, 16), 0, "torus"),
    ];
    for (m, chi, name) in &solids {
        assert_watertight(m, name);
        assert_eq!(euler_characteristic(m), *chi, "{name} Euler characteristic");
        assert!(m.volume() > 0.0, "{name} volume must be positive (outward winding)");
    }
    // Convex solids: every face normal points away from the centroid.
    for (m, _, name) in solids.iter().filter(|(_, chi, _)| *chi == 2) {
        let c = m.centroid();
        for t in m.triangles() {
            assert!(
                t.normal().dot(&(t.centroid() - c)) > 0.0,
                "{name}: inward-facing face"
            );
        }
    }
}

#[test]
fn prop_obj_roundtrip_random_meshes() {
    let mut rng = Rng::new(902);
    for i in 0..5 {
        let mut m = match i {
            0 => icosphere(0.5 + rng.next_f64(), 2),
            1 => torus(2.0, 0.5, 16, 8),
            2 => cylinder(1.0, 2.0, 12, true),
            3 => box_mesh(Vec3::new(1.0, 2.0, 3.0)),
            _ => cone(1.0, 2.0, 9),
        };
        // Random rigid motion produces awkward float coordinates.
        m.translate(Vec3::new(
            (rng.next_f64() - 0.5) * 10.0,
            (rng.next_f64() - 0.5) * 10.0,
            (rng.next_f64() - 0.5) * 10.0,
        ));
        let back = Mesh::from_obj(&m.to_obj()).expect("own OBJ output parses");
        assert_eq!(m, back, "OBJ round-trip must be exact");
        m.compute_vertex_normals();
        let back = Mesh::from_obj(&m.to_obj()).expect("own OBJ output with normals parses");
        assert_eq!(m, back);
    }
}

#[test]
fn prop_extruded_polygon_volume_matches_area() {
    let mut rng = Rng::new(903);
    for _ in 0..10 {
        // Random star-shaped (hence simple) polygon around the origin.
        let n = 5 + (rng.next_u64() % 8) as usize;
        let pts: Vec<Vec2> = (0..n)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                let r = 0.5 + rng.next_f64() * 1.5;
                Vec2::new(r * a.cos(), r * a.sin())
            })
            .collect();
        let poly = Polygon2::new(pts);
        let h = 0.3 + rng.next_f64() * 2.0;
        let m = extrude_polygon(&poly, h);
        assert_watertight(&m, "extrusion");
        assert_eq!(euler_characteristic(&m), 2);
        let expect = poly.area() * h;
        assert!(
            (m.volume() - expect).abs() < 1e-9 * expect.max(1.0),
            "prism volume {} vs area*h {expect}",
            m.volume()
        );
    }
}

#[test]
fn prop_marching_cubes_sphere_res64() {
    let r = 1.0;
    let bounds = Aabb {
        min: Vec3::new(-1.5, -1.5, -1.5),
        max: Vec3::new(1.5, 1.5, 1.5),
    };
    let field = ScalarField3::from_sdf(bounds, 64, 64, 64, &|p| sd_sphere(p, r));
    let m = marching_cubes(&field, 0.0);
    assert_watertight(&m, "marching cubes sphere");
    assert_eq!(euler_characteristic(&m), 2);
    let vol = volume_sphere(r);
    assert!((m.volume() - vol).abs() / vol < 0.02, "volume {} vs {vol}", m.volume());
    // Every vertex is within a cell of the true surface, and the
    // interpolated field there is below the cell-size * gradient bound
    // (the SDF has |grad| = 1).
    let cell = 3.0 / 63.0;
    for v in &m.vertices {
        assert!((v.magnitude() - r).abs() < cell);
        assert!(field.sample_trilinear(*v).abs() < cell);
    }
}

#[test]
fn prop_marching_methods_agree_on_offset_sphere() {
    // An off-center sphere exercises asymmetric cell configurations.
    let c = Vec3::new(0.13, -0.21, 0.07);
    let r = 0.9;
    let bounds = Aabb {
        min: Vec3::new(-1.5, -1.5, -1.5),
        max: Vec3::new(1.5, 1.5, 1.5),
    };
    let field = ScalarField3::from_sdf(bounds, 40, 40, 40, &|p| (p - c).magnitude() - r);
    let vol = volume_sphere(r);
    let mc = marching_cubes(&field, 0.0);
    let mt = marching_tetrahedra(&field, 0.0);
    for (m, name) in [(&mc, "cubes"), (&mt, "tetrahedra")] {
        assert_watertight(m, name);
        assert_eq!(euler_characteristic(m), 2, "marching {name}");
        assert!((m.volume() - vol).abs() / vol < 0.03, "marching {name} volume");
        let cc = m.centroid();
        assert!(cc.distance_to(&c) < 0.01, "marching {name} centroid drifted");
    }
}

#[test]
fn prop_marching_squares_circle_length_and_area() {
    let r = 1.0;
    let bounds = Rect { min: Vec2::new(-2.0, -2.0), max: Vec2::new(2.0, 2.0) };
    let field = ScalarField2::from_fn(bounds, 128, 128, &|p| p.magnitude() - r);
    let segs = marching_squares(&field, 0.0);
    let total: f64 = segs.iter().map(|s| s.a.distance_to(&s.b)).sum();
    let exact = 2.0 * std::f64::consts::PI * r;
    assert!((total - exact).abs() / exact < 0.01, "contour length {total} vs {exact}");
    // Joined polylines: one closed loop with the same length.
    let loops = marching_squares_polylines(&field, 0.0);
    assert_eq!(loops.len(), 1);
    let lp = &loops[0];
    assert!(lp.first().unwrap().distance_to(lp.last().unwrap()) < 1e-12);
    let joined: f64 = lp.windows(2).map(|w| w[0].distance_to(&w[1])).sum();
    assert!((joined - total).abs() < 1e-9);
}
