//! Properties for `spatial`: transforms, frames, projective maps, and
//! primitives.

use rust_physics_engine::math::{Vec2, Vec3};
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::quaternion::Quaternion;
use rust_physics_engine::spatial::{Aabb, Affine2, Frame, Homography, Mat4, Triangle};

fn rand_vec3(rng: &mut Rng, s: f64) -> Vec3 {
    Vec3::new(
        (rng.next_f64() - 0.5) * s,
        (rng.next_f64() - 0.5) * s,
        (rng.next_f64() - 0.5) * s,
    )
}

fn rand_vec2(rng: &mut Rng, s: f64) -> Vec2 {
    Vec2::new((rng.next_f64() - 0.5) * s, (rng.next_f64() - 0.5) * s)
}

fn rand_quat(rng: &mut Rng) -> Quaternion {
    Quaternion::from_axis_angle(rand_vec3(rng, 2.0) + Vec3::new(0.0, 0.0, 1e-3), rng.next_f64() * 6.0)
}

fn mats_close(a: &Mat4, b: &Mat4, tol: f64) -> bool {
    a.data
        .iter()
        .flatten()
        .zip(b.data.iter().flatten())
        .all(|(x, y)| (x - y).abs() < tol)
}

/// M · M⁻¹ == I for random invertible TRS matrices.
#[test]
fn prop_mat4_inverse() {
    let mut rng = Rng::new(201);
    for _ in 0..50 {
        let m = Mat4::from_trs(
            rand_vec3(&mut rng, 10.0),
            &rand_quat(&mut rng),
            Vec3::new(
                0.5 + rng.next_f64() * 2.0,
                0.5 + rng.next_f64() * 2.0,
                0.5 + rng.next_f64() * 2.0,
            ),
        );
        let inv = m.inverse().expect("TRS with positive scale is invertible");
        assert!(mats_close(&(m * inv), &Mat4::identity(), 1e-9));
    }
}

/// from_trs(decompose_trs(M)) == M for random TRS matrices.
#[test]
fn prop_mat4_trs_roundtrip() {
    let mut rng = Rng::new(202);
    for _ in 0..50 {
        let m = Mat4::from_trs(
            rand_vec3(&mut rng, 5.0),
            &rand_quat(&mut rng),
            Vec3::new(
                0.2 + rng.next_f64() * 3.0,
                0.2 + rng.next_f64() * 3.0,
                0.2 + rng.next_f64() * 3.0,
            ),
        );
        let (t, q, s) = m.decompose_trs().expect("affine TRS decomposes");
        assert!(mats_close(&m, &Mat4::from_trs(t, &q, s), 1e-9));
    }
}

/// Affine2::from_three_points maps its defining points exactly, and
/// rotations preserve distances.
#[test]
fn prop_affine2_three_points_and_rigidity() {
    let mut rng = Rng::new(203);
    for _ in 0..50 {
        let src = [rand_vec2(&mut rng, 4.0), rand_vec2(&mut rng, 4.0), rand_vec2(&mut rng, 4.0)];
        let area2 = (src[1] - src[0]).cross(&(src[2] - src[0]));
        if area2.abs() < 1e-3 {
            continue;
        }
        let dst = [rand_vec2(&mut rng, 4.0), rand_vec2(&mut rng, 4.0), rand_vec2(&mut rng, 4.0)];
        let m = Affine2::from_three_points(src, dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            assert!(m.apply(*s).distance_to(d) < 1e-8);
        }

        let rot = Affine2::rotation(rng.next_f64() * 6.0);
        let p = rand_vec2(&mut rng, 5.0);
        let q = rand_vec2(&mut rng, 5.0);
        assert!((rot.apply(p).distance_to(&rot.apply(q)) - p.distance_to(&q)).abs() < 1e-10);
        assert!(rot.is_rigid(1e-10));
    }
}

/// Affine2 decompose → recompose reproduces the map.
#[test]
fn prop_affine2_decompose_roundtrip() {
    let mut rng = Rng::new(204);
    for _ in 0..50 {
        let m = Affine2::translation(rand_vec2(&mut rng, 5.0))
            .compose(&Affine2::rotation(rng.next_f64() * 6.0))
            .compose(&Affine2::scaling(
                0.3 + rng.next_f64() * 2.0,
                0.3 + rng.next_f64() * 2.0,
            ))
            .compose(&Affine2::shear(rng.next_f64() - 0.5, 0.0));
        let (t, rot, s, k) = m.decompose();
        let rebuilt = Affine2::translation(t)
            .compose(&Affine2::rotation(rot))
            .compose(&Affine2 {
                m: [[s.x, k * s.y, 0.0], [0.0, s.y, 0.0], [0.0, 0.0, 1.0]],
            });
        for _ in 0..5 {
            let p = rand_vec2(&mut rng, 3.0);
            assert!(m.apply(p).distance_to(&rebuilt.apply(p)) < 1e-9);
        }
    }
}

/// Frames: to_world ∘ to_local is the identity; compose(inverse) is the
/// identity; to_mat4 agrees with the frame maps.
#[test]
fn prop_frame_roundtrips() {
    let mut rng = Rng::new(205);
    for _ in 0..50 {
        let f = Frame::new(rand_vec3(&mut rng, 8.0), rand_quat(&mut rng));
        let p = rand_vec3(&mut rng, 6.0);
        assert!(f.to_world(f.to_local(p)).distance_to(&p) < 1e-10);
        let id = f.compose(&f.inverse());
        assert!(id.to_world(p).distance_to(&p) < 1e-10);
        assert!(f.to_mat4().transform_point(p).distance_to(&f.to_world(p)) < 1e-10);
    }
}

/// Homographies map their four defining points exactly and preserve
/// collinearity.
#[test]
fn prop_homography_points_and_lines() {
    let mut rng = Rng::new(206);
    let mut done = 0;
    while done < 30 {
        let base = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let dst = [
            base[0] + rand_vec2(&mut rng, 0.4),
            base[1] + rand_vec2(&mut rng, 0.4),
            base[2] + rand_vec2(&mut rng, 0.4),
            base[3] + rand_vec2(&mut rng, 0.4),
        ];
        let Some(h) = Homography::from_four_points(base, dst) else {
            continue;
        };
        for (s, d) in base.iter().zip(&dst) {
            assert!(h.apply(*s).unwrap().distance_to(d) < 1e-8);
        }
        // Collinear triple stays collinear.
        let t0 = rng.next_f64();
        let t1 = rng.next_f64();
        let dir = rand_vec2(&mut rng, 1.0);
        let o = rand_vec2(&mut rng, 0.5) + Vec2::new(0.5, 0.5);
        let (Some(a), Some(b), Some(c)) = (
            h.apply(o),
            h.apply(o + dir * t0),
            h.apply(o + dir * (t0 + t1)),
        ) else {
            continue;
        };
        let area2 = (b - a).cross(&(c - a));
        assert!(area2.abs() < 1e-6, "collinearity broken: {area2}");
        done += 1;
    }
}

/// Triangle barycentric ↔ Cartesian roundtrip; Aabb union contains both
/// operands' corners.
#[test]
fn prop_triangle_and_aabb() {
    let mut rng = Rng::new(207);
    for _ in 0..50 {
        let t = Triangle {
            a: rand_vec3(&mut rng, 5.0),
            b: rand_vec3(&mut rng, 5.0),
            c: rand_vec3(&mut rng, 5.0),
        };
        if t.area() < 1e-3 {
            continue;
        }
        let u = rng.next_f64();
        let v = rng.next_f64() * (1.0 - u);
        let w = 1.0 - u - v;
        let p = t.from_barycentric(u, v, w);
        let (u2, v2, w2) = t.barycentric(p);
        assert!((u - u2).abs() < 1e-8 && (v - v2).abs() < 1e-8 && (w - w2).abs() < 1e-8);

        let a = Aabb::from_points(&[rand_vec3(&mut rng, 4.0), rand_vec3(&mut rng, 4.0)]);
        let b = Aabb::from_points(&[rand_vec3(&mut rng, 4.0), rand_vec3(&mut rng, 4.0)]);
        let un = a.union(&b);
        for c in a.corners().iter().chain(b.corners().iter()) {
            assert!(un.expand(1e-12).contains_point(*c));
        }
    }
}

// ── Intersection, distance, and containment properties ──────────────

use rust_physics_engine::linalg::Mat3;
use rust_physics_engine::spatial::contain::{
    orient2d, orient2d_exact, point_in_polygon_2d, winding_number_2d,
};
use rust_physics_engine::spatial::distance::{
    closest_point_segment, closest_point_triangle, distance_point_triangle,
};
use rust_physics_engine::spatial::intersect::{
    aabb_aabb, obb_obb, ray_aabb, ray_obb, ray_sphere, ray_triangle, segment_segment_2d,
};
use rust_physics_engine::spatial::{Obb, Polygon2, Ray, Segment, Segment2, Sphere};

/// ray_sphere hits land exactly on the sphere surface.
#[test]
fn prop_ray_sphere_on_surface() {
    let mut rng = Rng::new(211);
    let mut hits = 0;
    for _ in 0..300 {
        let s = Sphere { center: rand_vec3(&mut rng, 4.0), radius: 0.5 + rng.next_f64() * 2.0 };
        let origin = s.center + rand_vec3(&mut rng, 1.0).normalized() * (s.radius + 2.0 + rng.next_f64() * 8.0);
        // Aim at a jittered point near the center so most rays hit.
        let target = s.center + rand_vec3(&mut rng, s.radius);
        let dir = target - origin;
        if dir.magnitude_squared() < 1e-6 {
            continue;
        }
        let r = Ray::new(origin, dir);
        if let Some(hit) = ray_sphere(&r, &s) {
            if hit.t > 0.0 {
                assert!(
                    (hit.point.distance_to(&s.center) - s.radius).abs() < 1e-9,
                    "hit off surface"
                );
                hits += 1;
            }
        }
    }
    assert!(hits > 10, "too few hits to be meaningful: {hits}");
}

/// ray_triangle barycentrics sum to 1 with components in [0, 1], and
/// reproduce the hit point.
#[test]
fn prop_ray_triangle_barycentric() {
    let mut rng = Rng::new(212);
    let mut hits = 0;
    while hits < 50 {
        let t = rust_physics_engine::spatial::Triangle {
            a: rand_vec3(&mut rng, 3.0),
            b: rand_vec3(&mut rng, 3.0),
            c: rand_vec3(&mut rng, 3.0),
        };
        if t.area() < 0.1 {
            continue;
        }
        // Aim at a random interior point.
        let u = 0.1 + rng.next_f64() * 0.5;
        let v = 0.1 + rng.next_f64() * (0.8 - u);
        let target = t.from_barycentric(1.0 - u - v, u, v);
        let origin = target + t.normal() * (1.0 + rng.next_f64() * 4.0);
        let r = Ray::new(origin, target - origin);
        let Some((hit, (bu, bv, bw))) = ray_triangle(&r, &t, false) else {
            continue;
        };
        assert!((bu + bv + bw - 1.0).abs() < 1e-9);
        assert!(bu >= -1e-9 && bv >= -1e-9 && bw >= -1e-9);
        assert!(t.from_barycentric(bu, bv, bw).distance_to(&hit.point) < 1e-8);
        hits += 1;
    }
}

/// segment_segment_2d is symmetric in its arguments.
#[test]
fn prop_segment_intersection_symmetric() {
    let mut rng = Rng::new(213);
    for _ in 0..300 {
        let s1 = Segment2 { a: rand_vec2(&mut rng, 4.0), b: rand_vec2(&mut rng, 4.0) };
        let s2 = Segment2 { a: rand_vec2(&mut rng, 4.0), b: rand_vec2(&mut rng, 4.0) };
        match (segment_segment_2d(&s1, &s2), segment_segment_2d(&s2, &s1)) {
            (Some(p), Some(q)) => assert!(p.distance_to(&q) < 1e-9),
            (None, None) => {}
            _ => panic!("asymmetric intersection result"),
        }
    }
}

/// ray_aabb agrees with ray_obb at identity rotation; obb_obb agrees
/// with aabb_aabb at identity rotation.
#[test]
fn prop_identity_obb_matches_aabb() {
    let mut rng = Rng::new(214);
    for _ in 0..200 {
        let c = rand_vec3(&mut rng, 4.0);
        let h = Vec3::new(
            0.2 + rng.next_f64(),
            0.2 + rng.next_f64(),
            0.2 + rng.next_f64(),
        );
        let aabb = Aabb { min: c - h, max: c + h };
        let obb = Obb { center: c, half_extents: h, rotation: Mat3::identity() };
        let dir = rand_vec3(&mut rng, 2.0);
        if dir.magnitude_squared() < 1e-6 {
            continue;
        }
        let r = Ray::new(rand_vec3(&mut rng, 10.0), dir);
        match (ray_aabb(&r, &aabb), ray_obb(&r, &obb)) {
            (Some((a1, a2)), Some((b1, b2))) => {
                assert!((a1 - b1).abs() < 1e-9 && (a2 - b2).abs() < 1e-9);
            }
            (None, None) => {}
            other => panic!("ray aabb/obb disagree: {other:?}"),
        }

        let c2 = rand_vec3(&mut rng, 4.0);
        let h2 = Vec3::new(
            0.2 + rng.next_f64(),
            0.2 + rng.next_f64(),
            0.2 + rng.next_f64(),
        );
        let aabb2 = Aabb { min: c2 - h2, max: c2 + h2 };
        let obb2 = Obb { center: c2, half_extents: h2, rotation: Mat3::identity() };
        assert_eq!(aabb_aabb(&aabb, &aabb2), obb_obb(&obb, &obb2));
    }
}

/// Closest-point queries: the reported point is on the primitive and no
/// random sample on the primitive is closer.
#[test]
fn prop_closest_point_optimality() {
    let mut rng = Rng::new(215);
    for _ in 0..20 {
        let t = rust_physics_engine::spatial::Triangle {
            a: rand_vec3(&mut rng, 3.0),
            b: rand_vec3(&mut rng, 3.0),
            c: rand_vec3(&mut rng, 3.0),
        };
        if t.area() < 0.05 {
            continue;
        }
        let p = rand_vec3(&mut rng, 6.0);
        let q = closest_point_triangle(p, &t);
        let d = distance_point_triangle(p, &t);
        assert!((q.distance_to(&p) - d).abs() < 1e-12);
        // Closest point lies on (or numerically on) the triangle.
        let (u, v, w) = t.barycentric(q);
        assert!(u > -1e-9 && v > -1e-9 && w > -1e-9 && (u + v + w - 1.0).abs() < 1e-9);
        for _ in 0..1000 {
            let ru = rng.next_f64();
            let rv = rng.next_f64() * (1.0 - ru);
            let sample = t.from_barycentric(1.0 - ru - rv, ru, rv);
            assert!(sample.distance_to(&p) >= d - 1e-9, "sample closer than closest point");
        }

        let s = Segment { a: rand_vec3(&mut rng, 3.0), b: rand_vec3(&mut rng, 3.0) };
        let (qs, ts) = closest_point_segment(p, &s);
        assert!((0.0..=1.0).contains(&ts));
        let ds = qs.distance_to(&p);
        for _ in 0..200 {
            let sample = s.a.lerp(&s.b, rng.next_f64());
            assert!(sample.distance_to(&p) >= ds - 1e-9);
        }
    }
}

/// orient2d antisymmetry; polygon containment tests agree; the exact
/// predicate agrees with the float sign away from degeneracy.
#[test]
fn prop_orientation_and_containment() {
    let mut rng = Rng::new(216);
    for _ in 0..300 {
        let a = rand_vec2(&mut rng, 5.0);
        let b = rand_vec2(&mut rng, 5.0);
        let c = rand_vec2(&mut rng, 5.0);
        assert!((orient2d(a, b, c) + orient2d(b, a, c)).abs() < 1e-9);
        let det = orient2d(a, b, c);
        if det.abs() > 1e-9 {
            assert_eq!(orient2d_exact(a, b, c) as f64, det.signum());
        }
    }
    // Random star-shaped simple polygon: even-odd matches winding.
    for _ in 0..20 {
        let n = 6 + (rng.next_u64() % 6) as usize;
        let verts: Vec<Vec2> = (0..n)
            .map(|i| {
                let ang = std::f64::consts::TAU * i as f64 / n as f64;
                let r = 1.0 + rng.next_f64();
                Vec2::new(r * ang.cos(), r * ang.sin())
            })
            .collect();
        let poly = Polygon2::new(verts);
        for _ in 0..100 {
            let p = rand_vec2(&mut rng, 5.0);
            assert_eq!(
                point_in_polygon_2d(p, &poly),
                winding_number_2d(p, &poly) != 0,
                "even-odd vs winding disagree at {p:?}"
            );
        }
    }
}

// ── SDF and acceleration-structure properties ───────────────────────

use rust_physics_engine::spatial::sdf::{
    op_union, sd_box, sd_capsule, sd_sphere, sd_torus, sdf_raymarch,
};
use rust_physics_engine::spatial::{Bvh, KdTree, Quadtree, Rect};

/// SDF invariants: sd_sphere is exact, primitive gradients have unit
/// magnitude away from the surface, op_union equals min, and sphere
/// tracing matches the analytic ray-sphere hit.
#[test]
fn prop_sdf_invariants() {
    let mut rng = Rng::new(221);
    for _ in 0..200 {
        let p = rand_vec3(&mut rng, 8.0);
        let r = 0.5 + rng.next_f64() * 2.0;
        assert_eq!(sd_sphere(p, r), p.magnitude() - r);
        let (a, b) = (rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 4.0 - 2.0);
        assert_eq!(op_union(a, b), a.min(b));
    }
    // Lipschitz |grad| ~ 1 sampled away from surfaces/skeletons.
    let prims: Vec<Box<dyn Fn(Vec3) -> f64>> = vec![
        Box::new(|q| sd_sphere(q, 1.0)),
        Box::new(|q| sd_box(q, Vec3::new(1.0, 0.6, 0.8))),
        Box::new(|q| sd_torus(q, 2.0, 0.4)),
        Box::new(|q| sd_capsule(q, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.3)),
    ];
    for f in &prims {
        let mut checked = 0;
        while checked < 50 {
            let q = rand_vec3(&mut rng, 8.0);
            let d = f(q);
            if d.abs() < 0.2 {
                continue; // stay off the surface for clean differences
            }
            let eps = 1e-5;
            let g = Vec3::new(
                f(Vec3::new(q.x + eps, q.y, q.z)) - f(Vec3::new(q.x - eps, q.y, q.z)),
                f(Vec3::new(q.x, q.y + eps, q.z)) - f(Vec3::new(q.x, q.y - eps, q.z)),
                f(Vec3::new(q.x, q.y, q.z + eps)) - f(Vec3::new(q.x, q.y, q.z - eps)),
            ) * (1.0 / (2.0 * eps));
            assert!(g.magnitude() < 1.0 + 1e-3, "gradient above 1: {}", g.magnitude());
            checked += 1;
        }
    }
    // Raymarch vs analytic sphere hit.
    for _ in 0..20 {
        let origin = rand_vec3(&mut rng, 1.0).normalized() * 5.0;
        let ray = Ray::new(origin, -origin);
        let f = |p: Vec3| sd_sphere(p, 1.0);
        let marched = sdf_raymarch(&f, &ray, 100.0, 1e-10, 512).unwrap();
        let analytic = ray_sphere(&ray, &Sphere { center: Vec3::ZERO, radius: 1.0 }).unwrap();
        assert!((marched.t - analytic.t).abs() < 1e-6);
    }
}

/// Acceleration structures answer exactly like brute force on 1000
/// random items.
#[test]
fn prop_acceleration_structures_exact() {
    let mut rng = Rng::new(222);
    // BVH sphere query.
    let boxes: Vec<Aabb> = (0..1000)
        .map(|_| {
            let c = rand_vec3(&mut rng, 20.0);
            let h = Vec3::new(
                0.1 + rng.next_f64() * 0.5,
                0.1 + rng.next_f64() * 0.5,
                0.1 + rng.next_f64() * 0.5,
            );
            Aabb { min: c - h, max: c + h }
        })
        .collect();
    let bvh = Bvh::build(&boxes);
    for _ in 0..10 {
        let s = Sphere { center: rand_vec3(&mut rng, 20.0), radius: 1.0 + rng.next_f64() * 3.0 };
        let mut fast = bvh.query_sphere(&s);
        fast.sort_unstable();
        let brute: Vec<usize> = (0..boxes.len())
            .filter(|&i| rust_physics_engine::spatial::intersect::sphere_aabb(&s, &boxes[i]))
            .collect();
        assert_eq!(fast, brute);
    }
    // KdTree k-nearest ascending + exact.
    let pts: Vec<Vec3> = (0..1000).map(|_| rand_vec3(&mut rng, 20.0)).collect();
    let tree = KdTree::build(&pts);
    for _ in 0..10 {
        let q = rand_vec3(&mut rng, 20.0);
        let knn = tree.k_nearest(q, 5);
        for w in knn.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
        let mut brute: Vec<f64> = pts.iter().map(|p| p.distance_to(&q)).collect();
        brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, &(_, d)) in knn.iter().enumerate() {
            assert!((d - brute[i]).abs() < 1e-12);
        }
    }
    // Quadtree rect query.
    let bounds = Rect { min: Vec2::new(-10.0, -10.0), max: Vec2::new(10.0, 10.0) };
    let mut qt = Quadtree::new(bounds, 6, 10);
    let pts2: Vec<Vec2> = (0..1000).map(|_| rand_vec2(&mut rng, 20.0)).collect();
    for (i, &p) in pts2.iter().enumerate() {
        qt.insert(p, i);
    }
    for _ in 0..10 {
        let a = rand_vec2(&mut rng, 16.0);
        let r = Rect { min: a, max: a + Vec2::new(3.0, 3.0) };
        let mut fast: Vec<usize> = qt.query_rect(&r).into_iter().map(|(_, i)| i).collect();
        fast.sort_unstable();
        let brute: Vec<usize> = (0..pts2.len()).filter(|&i| r.contains_point(pts2[i])).collect();
        assert_eq!(fast, brute);
    }
}
