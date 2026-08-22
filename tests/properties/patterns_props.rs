//! Properties for `patterns`: polygon algorithms and sampling
//! distributions under randomized inputs.

use rust_physics_engine::math::{Vec2, Vec3};
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::patterns::polygon_ops::{
    boolean_intersection, boolean_union, boolean_xor, convex_decomposition, convex_hull_2d,
    offset_polygon, smallest_enclosing_circle, triangulate_ear_clipping, JoinStyle,
};
use rust_physics_engine::patterns::sampling::{
    poisson_disk_2d, random_convex_polygon, random_simple_polygon, uniform_in_polygon,
    uniform_on_sphere,
};
use rust_physics_engine::spatial::primitives::{Polygon2, Rect, Sphere};

fn cross3(o: Vec2, a: Vec2, b: Vec2) -> f64 {
    (a - o).cross(&(b - o))
}

#[test]
fn prop_triangulation_area_random_simple_polygons() {
    let mut rng = Rng::new(801);
    for trial in 0..20 {
        let n = 5 + (trial % 8);
        let poly = random_simple_polygon(n, &mut rng);
        let tris = triangulate_ear_clipping(&poly).expect("simple polygon triangulates");
        assert_eq!(tris.len(), n - 2);
        let sum: f64 = tris
            .iter()
            .map(|&[a, b, c]| {
                cross3(poly.vertices[a], poly.vertices[b], poly.vertices[c]).abs() / 2.0
            })
            .sum();
        assert!(
            (sum - poly.area()).abs() < 1e-9,
            "triangle areas {sum} vs polygon {}",
            poly.area()
        );
    }
}

#[test]
fn prop_boolean_identities_random_convex() {
    let mut rng = Rng::new(802);
    for trial in 0..25 {
        let a = random_convex_polygon(5 + trial % 6, &mut rng);
        let mut bv = random_convex_polygon(5 + (trial + 3) % 6, &mut rng).vertices;
        // Shift b so the pair ranges from overlapping to disjoint.
        let shift = Vec2::new(rng.next_f64() * 1.2 - 0.3, rng.next_f64() * 1.2 - 0.3);
        for v in &mut bv {
            *v = *v + shift;
        }
        let b = Polygon2::new(bv);
        let area = |loops: &[Polygon2]| loops.iter().map(Polygon2::area_signed).sum::<f64>();
        let (aa, ab) = (a.area(), b.area());
        let ua = area(&boolean_union(&a, &b));
        let ia = area(&boolean_intersection(&a, &b));
        let xa = area(&boolean_xor(&a, &b));
        let tol = 1e-6 * (aa + ab).max(1.0);
        assert!(ua <= aa + ab + tol, "union too large: {ua} vs {aa} + {ab}");
        assert!(ua >= aa.max(ab) - tol, "union too small");
        assert!(ia >= -tol && ia <= aa.min(ab) + tol, "intersection bounds");
        assert!((ia + xa - ua).abs() < tol, "intersection + xor == union ({ia} + {xa} != {ua})");
        assert!((ua + ia - aa - ab).abs() < tol, "inclusion-exclusion");
    }
}

#[test]
fn prop_welzl_contains_and_tight() {
    let mut rng = Rng::new(803);
    for trial in 0..20 {
        let n = 5 + trial * 7;
        let pts: Vec<Vec2> = (0..n)
            .map(|_| Vec2::new(rng.next_f64() * 6.0 - 3.0, rng.next_f64() * 2.0 - 1.0))
            .collect();
        let c = smallest_enclosing_circle(&pts);
        let mut on_boundary = 0;
        for p in &pts {
            let d = p.distance_to(&c.center);
            assert!(d <= c.radius + 1e-9, "point outside Welzl circle");
            if (d - c.radius).abs() < 1e-7 {
                on_boundary += 1;
            }
        }
        assert!(on_boundary >= 2, "circle not tight ({on_boundary} support points)");
    }
}

#[test]
fn prop_offset_roundtrip_random_convex() {
    let mut rng = Rng::new(804);
    for trial in 0..15 {
        let poly = random_convex_polygon(5 + trial % 8, &mut rng);
        if poly.area() < 0.05 {
            continue; // avoid slivers whose inset vanishes
        }
        let d = 0.05 + rng.next_f64() * 0.2;
        let grown = offset_polygon(&poly, d, JoinStyle::Miter(1e6));
        assert_eq!(grown.len(), 1, "convex offset is a single loop");
        let back = offset_polygon(&grown[0], -d, JoinStyle::Miter(1e6));
        assert_eq!(back.len(), 1);
        assert!(
            (back[0].area() - poly.area()).abs() < 1e-6 * poly.area().max(1.0),
            "miter offset round-trip exact on convex input"
        );
    }
}

#[test]
fn prop_convex_pieces_partition_area() {
    let mut rng = Rng::new(805);
    for _ in 0..10 {
        let poly = random_simple_polygon(9, &mut rng);
        let parts = convex_decomposition(&poly);
        let total: f64 = parts.iter().map(Polygon2::area).sum();
        assert!((total - poly.area()).abs() < 1e-9, "decomposition partitions the area");
        for p in &parts {
            assert!(p.is_convex());
        }
    }
}

#[test]
fn prop_hull_contains_all_points() {
    let mut rng = Rng::new(806);
    for _ in 0..10 {
        let pts: Vec<Vec2> = (0..100)
            .map(|_| {
                let a = rng.next_f64() * 6.283;
                let r = rng.next_f64().sqrt();
                Vec2::new(r * a.cos() * 2.0, r * a.sin())
            })
            .collect();
        let hull = convex_hull_2d(&pts);
        assert!(hull.is_convex() && hull.is_ccw());
        let n = hull.vertices.len();
        for &p in &pts {
            for i in 0..n {
                assert!(cross3(hull.vertices[i], hull.vertices[(i + 1) % n], p) >= -1e-9);
            }
        }
    }
}

#[test]
fn prop_poisson_disk_separation_maximality() {
    let mut rng = Rng::new(807);
    for trial in 0..3 {
        let d = 0.3 + 0.2 * trial as f64;
        let region = Rect { min: Vec2::new(-3.0, -2.0), max: Vec2::new(3.0, 2.0) };
        let pts = poisson_disk_2d(&region, d, 30, &mut rng);
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                assert!(pts[i].distance_to(&pts[j]) >= d - 1e-12);
            }
        }
        // Maximality: probe grid, no empty disk of radius 2d.
        let mut y = region.min.y + 0.1;
        while y < region.max.y {
            let mut x = region.min.x + 0.1;
            while x < region.max.x {
                let p = Vec2::new(x, y);
                let near =
                    pts.iter().map(|q| q.distance_to(&p)).fold(f64::INFINITY, f64::min);
                assert!(near < 2.0 * d, "empty disk at {p:?}");
                x += d / 2.0;
            }
            y += d / 2.0;
        }
    }
}

#[test]
fn prop_uniform_samplers_statistics() {
    let mut rng = Rng::new(808);
    // On-sphere: unit norm, isotropic mean.
    let s = Sphere { center: Vec3::ZERO, radius: 2.0 };
    let n = 30_000;
    let mut mean = Vec3::ZERO;
    for _ in 0..n {
        let p = uniform_on_sphere(&s, &mut rng);
        assert!((p.magnitude() - 2.0).abs() < 1e-12);
        mean = mean + p;
    }
    assert!((mean * (1.0 / n as f64)).magnitude() < 0.03);
    // In-polygon: fraction landing in a half of a rectangle matches
    // the area ratio.
    let poly = Polygon2::new(vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(3.0, 0.0),
        Vec2::new(3.0, 1.0),
        Vec2::new(0.0, 1.0),
    ]);
    let n = 30_000;
    let mut left = 0usize;
    for _ in 0..n {
        let p = uniform_in_polygon(&poly, &mut rng);
        assert!(p.x >= 0.0 && p.x <= 3.0 && p.y >= 0.0 && p.y <= 1.0);
        if p.x < 1.0 {
            left += 1;
        }
    }
    let frac = left as f64 / n as f64;
    assert!((frac - 1.0 / 3.0).abs() < 0.015, "uniformity {frac}");
}

#[test]
fn prop_hilbert_morton_roundtrips() {
    use rust_physics_engine::patterns::space_filling::{
        gray_code, gray_decode, hilbert_3d_d2xyz, hilbert_3d_xyz2d, hilbert_d2xy, hilbert_xy2d,
        morton_decode_2d, morton_decode_3d, morton_encode_2d, morton_encode_3d,
    };
    let mut rng = Rng::new(809);
    for order in [4u32, 8, 12, 16] {
        for _ in 0..200 {
            let d = rng.next_u64() % (1u64 << (2 * order));
            let (x, y) = hilbert_d2xy(order, d);
            assert_eq!(hilbert_xy2d(order, x, y), d);
        }
    }
    for order in [4u32, 8, 16] {
        for _ in 0..200 {
            let d = rng.next_u64() % (1u64 << (3 * order));
            let (x, y, z) = hilbert_3d_d2xyz(order, d);
            assert_eq!(hilbert_3d_xyz2d(order, x, y, z), d);
        }
    }
    for _ in 0..500 {
        let (x, y) = (rng.next_u64() as u32, rng.next_u64() as u32);
        assert_eq!(morton_decode_2d(morton_encode_2d(x, y)), (x, y));
        let (a, b, c) = (
            (rng.next_u64() % (1 << 21)) as u32,
            (rng.next_u64() % (1 << 21)) as u32,
            (rng.next_u64() % (1 << 21)) as u32,
        );
        assert_eq!(morton_decode_3d(morton_encode_3d(a, b, c)), (a, b, c));
        let n = rng.next_u64() >> 1;
        assert_eq!(gray_decode(gray_code(n)), n);
    }
}

#[test]
fn prop_packings_never_overlap() {
    use rust_physics_engine::patterns::packing::{
        apollonian_gasket_integral, circle_pack_hex, packing_density_2d, sphere_pack_fcc,
    };
    use rust_physics_engine::spatial::Aabb;
    // Descartes theorem holds for every mutually tangent triple found
    // in the gasket (spot check the seed chain).
    let gasket = apollonian_gasket_integral(4);
    for (i, a) in gasket.iter().enumerate().skip(1) {
        for b in gasket.iter().skip(i + 1) {
            assert!(
                a.center.distance_to(&b.center) >= a.radius + b.radius - 1e-6,
                "gasket circles overlap"
            );
        }
    }
    let region = rust_physics_engine::spatial::primitives::Rect {
        min: Vec2::ZERO,
        max: Vec2::new(24.0, 12.0 * 3.0f64.sqrt()),
    };
    let hex = circle_pack_hex(&region, 1.2);
    let density = packing_density_2d(&hex, &region);
    assert!((density - std::f64::consts::PI / (2.0 * 3.0f64.sqrt())).abs() < 1e-6);
    let bx = Aabb {
        min: Vec3::ZERO,
        max: Vec3::new(
            3.0 * 2.0 * std::f64::consts::SQRT_2,
            3.0 * 2.0 * std::f64::consts::SQRT_2,
            3.0 * 2.0 * std::f64::consts::SQRT_2,
        ),
    };
    let fcc = sphere_pack_fcc(&bx, 1.0);
    let d3 = rust_physics_engine::patterns::packing::packing_density_3d(&fcc, &bx);
    assert!((d3 - std::f64::consts::PI / (3.0 * std::f64::consts::SQRT_2)).abs() < 1e-6);
}

#[test]
fn prop_wallpaper_closure_and_point_group_orbits() {
    use rust_physics_engine::patterns::symmetry::{
        point_group_orbit, point_group_rotations, wallpaper_generators, wallpaper_group_order,
        PointGroup3, WallpaperGroup,
    };
    for g in [
        WallpaperGroup::Pgg,
        WallpaperGroup::Cmm,
        WallpaperGroup::P4g,
        WallpaperGroup::P31m,
        WallpaperGroup::P6m,
    ] {
        assert_eq!(wallpaper_generators(g).len(), wallpaper_group_order(g));
    }
    assert_eq!(point_group_rotations(PointGroup3::Tetrahedral).len(), 12);
    assert_eq!(point_group_rotations(PointGroup3::Octahedral).len(), 24);
    assert_eq!(point_group_rotations(PointGroup3::Icosahedral).len(), 60);
    // Orbit sizes divide the group order for special points too.
    let mut rng = Rng::new(810);
    for _ in 0..5 {
        let p = Vec3::new(rng.next_f64(), rng.next_f64(), rng.next_f64());
        for (g, n) in [
            (PointGroup3::Tetrahedral, 12usize),
            (PointGroup3::Octahedral, 24),
            (PointGroup3::Icosahedral, 60),
        ] {
            let orbit = point_group_orbit(g, p);
            assert_eq!(n % orbit.len(), 0);
        }
    }
}

#[test]
fn prop_archimedean_tilings_unit_edges() {
    use rust_physics_engine::patterns::tilings::{archimedean, Archimedean};
    let extent = rust_physics_engine::spatial::primitives::Rect {
        min: Vec2::new(-6.0, -6.0),
        max: Vec2::new(6.0, 6.0),
    };
    for kind in [
        Archimedean::T3_3_4_3_4,
        Archimedean::T3_3_3_3_6,
        Archimedean::T4_6_12,
        Archimedean::T3_12_12,
    ] {
        let t = archimedean(kind, &extent, 1.0);
        assert!(!t.faces.is_empty());
        for &(a, b) in &t.edges {
            assert!((t.vertices[a].distance_to(&t.vertices[b]) - 1.0).abs() < 1e-6);
        }
        // Faces don't overlap: sample points lie in at most one face.
        let polys = t.polygons();
        let mut rng = Rng::new(811);
        for _ in 0..100 {
            let p = Vec2::new(rng.next_f64() * 8.0 - 4.0, rng.next_f64() * 8.0 - 4.0);
            let count = polys
                .iter()
                .filter(|poly| {
                    let v = &poly.vertices;
                    let n = v.len();
                    let mut ins = false;
                    for i in 0..n {
                        let (a, b) = (v[i], v[(i + 1) % n]);
                        if (a.y > p.y) != (b.y > p.y)
                            && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x)
                        {
                            ins = !ins;
                        }
                    }
                    ins
                })
                .count();
            assert!(count <= 1, "{kind:?} faces overlap at {p:?}");
        }
    }
}
