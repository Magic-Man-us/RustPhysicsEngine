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

#[test]
fn prop_gauss_bonnet_survives_random_deformation() {
    use rust_physics_engine::mesh::analyze::discrete_gaussian_curvature;
    let mut rng = Rng::new(904);
    for trial in 0..5 {
        let (mut m, chi) = if trial % 2 == 0 {
            (icosphere(1.0, 2), 2.0)
        } else {
            (torus(2.0, 0.5, 16, 8), 0.0)
        };
        // Angle deficit is purely intrinsic-topological in total: any
        // vertex perturbation that keeps faces non-degenerate leaves
        // the sum at 2 pi chi.
        for v in &mut m.vertices {
            *v = *v
                + Vec3::new(
                    (rng.next_f64() - 0.5) * 0.1,
                    (rng.next_f64() - 0.5) * 0.1,
                    (rng.next_f64() - 0.5) * 0.1,
                );
        }
        let total: f64 = discrete_gaussian_curvature(&m).iter().sum();
        assert!(
            (total - 2.0 * std::f64::consts::PI * chi).abs() < 1e-9,
            "Gauss-Bonnet after deformation: {total}"
        );
    }
}

#[test]
fn prop_parametric_curvatures_random_points() {
    use rust_physics_engine::mesh::surfaces::{
        catenoid, fundamental_forms, gaussian_curvature, helicoid, mean_curvature,
    };
    let mut rng = Rng::new(905);
    let r = 1.7;
    let sphere =
        |u: f64, v: f64| Vec3::new(r * v.sin() * u.cos(), r * v.cos(), r * v.sin() * u.sin());
    for _ in 0..20 {
        let u = rng.next_f64() * 6.0;
        let v = 0.3 + rng.next_f64() * 2.5;
        let ff = fundamental_forms(&sphere, u, v, 1e-4);
        assert!((gaussian_curvature(&ff) - 1.0 / (r * r)).abs() < 1e-3);

        let cat = |u: f64, v: f64| catenoid(u, v, 0.8);
        let ffc = fundamental_forms(&cat, u, v - 1.5, 1e-4);
        assert!(mean_curvature(&ffc).abs() < 1e-3, "catenoid is minimal");
        let hel = |u: f64, v: f64| helicoid(u, v, 1.1);
        let ffh = fundamental_forms(&hel, u, v, 1e-4);
        assert!(mean_curvature(&ffh).abs() < 1e-3, "helicoid is minimal");
    }
}

#[test]
fn prop_nurbs_quadrics_exact_random_params() {
    use rust_physics_engine::mesh::surfaces::NurbsSurface;
    let mut rng = Rng::new(906);
    let s = NurbsSurface::sphere(2.5);
    let t = NurbsSurface::torus(3.0, 0.75);
    for _ in 0..100 {
        let (u, v) = (rng.next_f64(), rng.next_f64());
        let p = s.eval(u, v);
        assert!((p.magnitude() - 2.5).abs() < 1e-12, "NURBS sphere exact");
        let q = t.eval(u, v);
        let radial = (q.x * q.x + q.z * q.z).sqrt();
        let d = ((radial - 3.0).powi(2) + q.y * q.y).sqrt();
        assert!((d - 0.75).abs() < 1e-12, "NURBS torus exact");
    }
}

#[test]
fn prop_subdivision_topology_and_smoothing() {
    use rust_physics_engine::mesh::subdivide::{
        catmull_clark_n, laplacian_smooth, loop_subdivide_n, midpoint_subdivide, sqrt3_subdivide,
        taubin_smooth, QuadMesh,
    };
    let base = icosphere(1.0, 1);
    let l = loop_subdivide_n(&base, 2);
    assert_eq!(euler_characteristic(&l), 2);
    assert_watertight(&l, "loop");
    assert_eq!(l.indices.len(), base.indices.len() * 16);

    let s3 = sqrt3_subdivide(&base);
    assert_eq!(euler_characteristic(&s3), 2);
    assert_watertight(&s3, "sqrt3");

    let mp = midpoint_subdivide(&base);
    assert!((mp.volume() - base.volume()).abs() < 1e-12, "midpoint keeps geometry");

    let cc = catmull_clark_n(&QuadMesh::from_box(Vec3::new(1.0, 1.0, 1.0)), 2);
    let cct = cc.to_triangles();
    assert_eq!(euler_characteristic(&cct), 2, "Catmull-Clark preserves Euler");
    assert!(cct.volume() > 0.0);

    // Taubin keeps volume; plain Laplacian shrinks.
    let smooth_base = icosphere(1.0, 3);
    let v0 = smooth_base.volume();
    let mut a = smooth_base.clone();
    taubin_smooth(&mut a, 10, 0.33, -0.34);
    assert!((a.volume() - v0).abs() / v0 < 0.01);
    let mut b = smooth_base.clone();
    laplacian_smooth(&mut b, 10, 0.5);
    assert!(b.volume() < v0 * 0.99);
}

#[test]
fn prop_decimation_random_bumpy_sphere_keeps_genus() {
    use rust_physics_engine::mesh::analyze::{decimate_edge_collapse, stats};
    let mut rng = Rng::new(907);
    let mut m = icosphere(1.0, 3);
    for v in &mut m.vertices {
        *v = *v * (1.0 + 0.1 * (rng.next_f64() - 0.5));
    }
    let d = decimate_edge_collapse(&m, 200);
    assert!(d.indices.len() <= 200 && d.indices.len() > 50);
    let s = stats(&d);
    assert!(s.is_manifold && s.is_closed && s.is_oriented);
    assert_eq!(s.genus, Some(0));
    // Volume roughly preserved.
    assert!((d.volume() - m.volume()).abs() / m.volume() < 0.1);
}

#[test]
fn prop_parameterization_invariance_and_injectivity() {
    use rust_physics_engine::mesh::parameterize::{
        area_distortion, conformal_distortion, harmonic_parameterization, lscm, BoundaryShape,
    };
    use rust_physics_engine::quaternion::Quaternion;
    // Random bumpy disk meshes: harmonic maps into a convex target
    // stay injective, and LSCM's conformal distortion is invariant
    // under rigid motion of the input.
    let mut rng = Rng::new(901);
    for trial in 0..5 {
        let rings = 3 + trial % 2;
        let segments = 10 + 2 * trial;
        let mut vertices = vec![Vec3::ZERO];
        for k in 1..=rings {
            let r = k as f64 / rings as f64;
            for j in 0..segments {
                let a = std::f64::consts::TAU * j as f64 / segments as f64;
                // Gentle out-of-plane bumps keep the mesh a graph
                // over the disk.
                let z = 0.15 * rng.next_f64() * (3.0 * a).sin();
                vertices.push(Vec3::new(r * a.cos(), r * a.sin(), z));
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
        let m = Mesh { vertices, indices, normals: None, uvs: None };
        for shape in [BoundaryShape::Circle, BoundaryShape::Square, BoundaryShape::Free] {
            let uv = harmonic_parameterization(&m, shape);
            for &[a, b, c] in &m.indices {
                let area = (uv[b] - uv[a]).cross(&(uv[c] - uv[a]));
                assert!(area > 0.0, "harmonic {shape:?} map stays injective");
            }
        }
        // LSCM distortion is a rigid invariant.
        let pins = [(1usize, Vec2::new(0.0, 0.0)), (1 + segments / 2, Vec2::new(1.0, 0.0))];
        let d0 = conformal_distortion(&m, &lscm(&m, pins));
        let q = Quaternion::from_axis_angle(
            Vec3::new(rng.next_f64(), rng.next_f64(), 1.0).normalized(),
            rng.next_f64() * 3.0,
        );
        let shift = Vec3::new(5.0, -2.0, 1.0);
        let moved = Mesh {
            vertices: m.vertices.iter().map(|&v| q.rotate_vec(v) + shift).collect(),
            indices: m.indices.clone(),
            normals: None,
            uvs: None,
        };
        let d1 = conformal_distortion(&moved, &lscm(&moved, pins));
        for (a, b) in d0.iter().zip(&d1) {
            assert!((a - b).abs() < 1e-6, "distortion is rigid-invariant ({a} vs {b})");
        }
        // Area distortion of the harmonic circle map averages to 1.
        let uv = harmonic_parameterization(&m, BoundaryShape::Circle);
        let ad = area_distortion(&m, &uv);
        let mean: f64 = ad.iter().sum::<f64>() / ad.len() as f64;
        assert!(mean > 0.5 && mean < 2.0, "area distortion near its normalized mean ({mean})");
    }
}
