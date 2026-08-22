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
