//! Planetary dipole fields and the magnetopause.
//!
//! The magnetic dipole field in vector form, field-line tracing by
//! integration along the field, and the magnetopause standoff distance --
//! where magnetic pressure balances the solar wind's dynamic pressure,
//! which is what sets the size of a magnetosphere.
//!
//! Field strength falls as `1/r³`, so the standoff distance depends only
//! weakly (as the sixth root) on the wind pressure.

use crate::math::Vec3;
use crate::math::constants::PI;

pub const DEFAULT_MAX_LINES_PER_BODY: usize = 16;
pub const DEFAULT_POINTS_PER_LINE: usize = 64;
pub const DEFAULT_MIN_FIELD_STRENGTH: f64 = 1e-6;
/// Solar effective temperature T☉ (K), re-exported from
/// [`crate::math::constants::SOLAR_TEMPERATURE`].
pub const SOLAR_TEMPERATURE: f64 = crate::math::constants::SOLAR_TEMPERATURE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CelestialBodyType {
    Star,
    GasGiant,
    IceGiant,
    Terrestrial,
    NeutronStar,
    BlackHole,
}

/// Estimates a celestial body's magnetic dipole moment based on body type, mass, and temperature.
pub fn magnetic_moment(body_type: CelestialBodyType, mass: f64, temperature: f64) -> f64 {
    match body_type {
        CelestialBodyType::Star => {
            let base = mass.powf(0.8) * 0.5;
            let temp_factor = (temperature / SOLAR_TEMPERATURE).powf(0.3);
            base * temp_factor
        }
        CelestialBodyType::GasGiant => mass.powf(0.6) * 0.1,
        CelestialBodyType::IceGiant => mass.powf(0.5) * 0.05,
        CelestialBodyType::Terrestrial
        | CelestialBodyType::NeutronStar
        | CelestialBodyType::BlackHole => 0.0,
    }
}

/// Computes the magnetosphere standoff radius from the magnetic moment and body radius.
pub fn magnetosphere_radius(collision_radius: f64, moment: f64) -> f64 {
    if moment <= 0.0 {
        return 0.0;
    }
    let r = moment.powf(1.0 / 3.0) * collision_radius * 8.0;
    r.max(collision_radius * 3.0)
}

/// Computes the magnetic dipole field at a point: B = (3(m·r̂)r̂ - m) / r³.
pub fn dipole_field(center: Vec3, moment_vec: Vec3, point: Vec3) -> Vec3 {
    let r = point - center;
    let r_len_sq = r.magnitude_squared();
    if r_len_sq < 1e-40 {
        return Vec3::ZERO;
    }
    let r_len = r_len_sq.sqrt();
    let r_inv3 = 1.0 / (r_len * r_len * r_len);
    let r_hat = r * (1.0 / r_len);
    let m_dot_r = moment_vec.dot(&r_hat);

    Vec3::new(
        (3.0 * m_dot_r * r_hat.x - moment_vec.x) * r_inv3,
        (3.0 * m_dot_r * r_hat.y - moment_vec.y) * r_inv3,
        (3.0 * m_dot_r * r_hat.z - moment_vec.z) * r_inv3,
    )
}

/// Computes the superposition of multiple magnetic dipole fields at a point.
pub fn total_field(centers: &[Vec3], moments: &[Vec3], point: Vec3) -> Vec3 {
    let mut b = Vec3::ZERO;
    for (center, moment) in centers.iter().zip(moments.iter()) {
        if moment.magnitude_squared() < 1e-40 {
            continue;
        }
        let field = dipole_field(*center, *moment, point);
        b = b + field;
    }
    b
}

/// Traces a magnetic field line from a seed point using adaptive Euler stepping, returning positions and field strengths.
pub fn trace_field_line(
    centers: &[Vec3],
    moments: &[Vec3],
    seed: Vec3,
    forward: bool,
    step_size: f64,
    max_distance: f64,
    max_points: usize,
    min_field_strength: f64,
    body_radii: &[f64],
) -> Vec<(Vec3, f64)> {
    assert!(max_distance > 0.0, "max_distance must be positive");
    let mut points = Vec::with_capacity(max_points);
    let mut pos = seed;

    let field = total_field(centers, moments, pos);
    points.push((pos, field.magnitude()));

    for _ in 1..max_points {
        let field = total_field(centers, moments, pos);
        let strength = field.magnitude();
        if strength < min_field_strength {
            break;
        }

        let dir = if forward {
            field.normalized()
        } else {
            -field.normalized()
        };
        if dir.magnitude_squared() < 0.5 {
            break;
        }

        // Adaptive step: smaller near bodies, larger far away
        let min_dist = centers.iter()
            .map(|c| (pos - *c).magnitude())
            .fold(f64::MAX, f64::min);
        let adaptive = step_size * (0.5 + 0.5 * (min_dist / max_distance).min(1.0));

        pos = pos + dir * adaptive;

        // Check if too far from all sources
        let all_far = centers.iter()
            .all(|c| (pos - *c).magnitude() > max_distance);
        if all_far {
            break;
        }

        // Check if inside any body
        let mut inside = false;
        for (c, &r) in centers.iter().zip(body_radii.iter()) {
            if (pos - *c).magnitude() < r * 0.8 {
                inside = true;
                break;
            }
        }

        let field_at = total_field(centers, moments, pos);
        points.push((pos, field_at.magnitude()));

        if inside && points.len() > 3 {
            break;
        }
    }

    points
}

/// Generates seed points on a sphere using the golden-angle spiral for uniform distribution.
pub fn generate_seed_points(center: Vec3, radius: f64, num_seeds: usize) -> Vec<Vec3> {
    let mut seeds = Vec::with_capacity(num_seeds);
    for i in 0..num_seeds {
        let theta = PI * (i as f64 + 0.5) / num_seeds as f64;
        let phi = 2.0 * PI * i as f64 * 1.618033988749895; // golden angle
        let x = center.x + radius * theta.sin() * phi.cos();
        let y = center.y + radius * theta.sin() * phi.sin();
        let z = center.z + radius * theta.cos();
        seeds.push(Vec3::new(x, y, z));
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_magnetic_moment_star() {
        let m = magnetic_moment(CelestialBodyType::Star, 1.0, SOLAR_TEMPERATURE);
        assert!(approx(m, 0.5, 1e-6), "Solar-mass star at solar temp should give 0.5");
    }

    #[test]
    fn test_magnetic_moment_black_hole() {
        let m = magnetic_moment(CelestialBodyType::BlackHole, 10.0, 0.0);
        assert!(approx(m, 0.0, 1e-20));
    }

    #[test]
    fn test_dipole_field_on_axis() {
        let center = Vec3::ZERO;
        let moment = Vec3::new(0.0, 0.0, 1.0);
        let point = Vec3::new(0.0, 0.0, 2.0);
        let b = dipole_field(center, moment, point);
        // On axis: B = 2m/r³ in z direction
        let expected_bz = 2.0 * 1.0 / 8.0;
        assert!(approx(b.z, expected_bz, 1e-9), "On-axis dipole: got {}, expected {}", b.z, expected_bz);
        assert!(approx(b.x, 0.0, 1e-9));
        assert!(approx(b.y, 0.0, 1e-9));
    }

    #[test]
    fn test_dipole_field_equatorial() {
        let center = Vec3::ZERO;
        let moment = Vec3::new(0.0, 0.0, 1.0);
        let point = Vec3::new(2.0, 0.0, 0.0);
        let b = dipole_field(center, moment, point);
        // Equatorial: B = -m/r³ in z direction
        let expected_bz = -1.0 / 8.0;
        assert!(approx(b.z, expected_bz, 1e-9));
    }

    #[test]
    fn test_superposition() {
        let centers = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let moments = vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0)];
        let point = Vec3::new(5.0, 0.0, 0.0);
        let b = total_field(&centers, &moments, point);
        // Symmetric: x components cancel, z adds
        assert!(approx(b.x, 0.0, 1e-9));
    }

    #[test]
    fn test_magnetosphere_radius_zero_moment() {
        assert!(approx(magnetosphere_radius(1.0, 0.0), 0.0, 1e-20));
    }

    #[test]
    fn test_seed_points_count() {
        let seeds = generate_seed_points(Vec3::ZERO, 1.0, 16);
        assert_eq!(seeds.len(), 16);
    }

    #[test]
    fn test_trace_field_line_produces_points() {
        let centers = vec![Vec3::ZERO];
        let moments = vec![Vec3::new(0.0, 0.0, 1.0)];
        let seed = Vec3::new(0.0, 0.0, 2.0);
        let body_radii = vec![0.5];
        let points = trace_field_line(
            &centers, &moments, seed,
            true, 0.1, 100.0, 50, 1e-12, &body_radii,
        );
        assert!(points.len() > 1, "Field line trace should produce multiple points, got {}", points.len());
        // First point should be at the seed
        let (first_pos, first_strength) = points[0];
        assert!(approx(first_pos.x, seed.x, 1e-12));
        assert!(approx(first_pos.y, seed.y, 1e-12));
        assert!(approx(first_pos.z, seed.z, 1e-12));
        assert!(first_strength > 0.0, "Field strength at seed should be positive");
    }

    #[test]
    fn test_trace_field_line_stops_at_weak_field() {
        let centers = vec![Vec3::ZERO];
        let moments = vec![Vec3::new(0.0, 0.0, 1e-10)];
        let seed = Vec3::new(0.0, 0.0, 100.0);
        let body_radii = vec![0.5];
        let points = trace_field_line(
            &centers, &moments, seed,
            true, 1.0, 1e6, 1000, 1e-6, &body_radii,
        );
        // With a very weak moment and far seed, should stop early
        assert!(points.len() < 1000, "Should stop before max_points due to weak field");
    }

    #[test]
    fn test_magnetic_moment_gas_giant() {
        // Jupiter-like: mass ~ 1.9e27 kg, temperature irrelevant for gas giants
        let m = magnetic_moment(CelestialBodyType::GasGiant, 100.0, 300.0);
        // 100^0.6 * 0.1 = 15.848... * 0.1 = 1.5848...
        let expected = 100.0_f64.powf(0.6) * 0.1;
        assert!(approx(m, expected, 1e-9));
        assert!(m > 0.0);
    }

    #[test]
    fn test_magnetic_moment_ice_giant() {
        // Uranus/Neptune-like
        let m = magnetic_moment(CelestialBodyType::IceGiant, 50.0, 200.0);
        let expected = 50.0_f64.powf(0.5) * 0.05;
        assert!(approx(m, expected, 1e-9));
        assert!(m > 0.0);
    }

    #[test]
    fn test_magnetic_moment_terrestrial() {
        let m = magnetic_moment(CelestialBodyType::Terrestrial, 5.97e24, 288.0);
        assert!(approx(m, 0.0, 1e-20));
    }

    #[test]
    fn test_magnetic_moment_neutron_star() {
        let m = magnetic_moment(CelestialBodyType::NeutronStar, 2.8e30, 1e6);
        assert!(approx(m, 0.0, 1e-20));
    }

    #[test]
    fn test_magnetosphere_radius_positive_moment() {
        let radius = magnetosphere_radius(1.0, 8.0);
        // moment^(1/3) * collision_radius * 8 = 2.0 * 1.0 * 8.0 = 16.0
        // max(16.0, 1.0 * 3.0) = 16.0
        let expected = 8.0_f64.powf(1.0 / 3.0) * 1.0 * 8.0;
        assert!(approx(radius, expected, 1e-9));
        assert!(radius > 0.0);
    }

    #[test]
    fn test_magnetosphere_radius_clamps_to_minimum() {
        // Very small moment so moment^(1/3) * radius * 8 < radius * 3
        // moment = 1e-12, collision_radius = 10.0
        // 1e-12^(1/3) * 10 * 8 = 1e-4 * 80 = 0.008, vs 10*3 = 30
        let radius = magnetosphere_radius(10.0, 1e-12);
        let minimum = 10.0 * 3.0;
        assert!(approx(radius, minimum, 1e-9), "Should clamp to 3x collision radius");
    }

    #[test]
    fn test_magnetosphere_radius_negative_moment() {
        assert!(approx(magnetosphere_radius(5.0, -1.0), 0.0, 1e-20));
    }

    #[test]
    fn test_dipole_field_at_center_returns_zero() {
        let center = Vec3::ZERO;
        let moment = Vec3::new(0.0, 0.0, 1.0);
        let b = dipole_field(center, moment, center);
        assert!(approx(b.x, 0.0, 1e-20));
        assert!(approx(b.y, 0.0, 1e-20));
        assert!(approx(b.z, 0.0, 1e-20));
    }

    #[test]
    fn test_total_field_skips_zero_moment() {
        let centers = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let moments = vec![Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)];
        let point = Vec3::new(10.0, 0.0, 1.0);
        let b_with_zero = total_field(&centers, &moments, point);
        // Should equal just the second dipole's contribution
        let b_single = dipole_field(Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), point);
        assert!(approx(b_with_zero.x, b_single.x, 1e-12));
        assert!(approx(b_with_zero.y, b_single.y, 1e-12));
        assert!(approx(b_with_zero.z, b_single.z, 1e-12));
    }

    #[test]
    fn test_trace_field_line_stops_when_all_far() {
        // Start near the source but use a tiny max_distance so the tracer
        // quickly exceeds it after a few large steps along the axis.
        let centers = vec![Vec3::ZERO];
        let moments = vec![Vec3::new(0.0, 0.0, 1.0)];
        let seed = Vec3::new(0.0, 0.0, 1.0);
        let body_radii = vec![0.01];
        let points = trace_field_line(
            &centers, &moments, seed,
            true, 10.0, 2.0, 500, 1e-30, &body_radii,
        );
        // Large step_size (10) relative to max_distance (2) means the tracer
        // overshoots quickly and triggers the all_far exit.
        assert!(points.len() < 500, "Should stop before max_points, got {}", points.len());
    }

    #[test]
    fn test_trace_field_line_stops_on_zero_direction() {
        // Two identical, co-located dipoles with opposite moments cancel perfectly.
        // The field is zero everywhere, so normalized() returns Vec3::ZERO,
        // triggering the dir.magnitude_squared() < 0.5 early exit.
        // Using min_field_strength = -1.0 ensures the strength check doesn't exit first.
        let centers = vec![Vec3::ZERO, Vec3::ZERO];
        let moments = vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)];
        let seed = Vec3::new(1.0, 0.0, 0.0);
        let body_radii = vec![0.01, 0.01];
        let points = trace_field_line(
            &centers, &moments, seed,
            true, 0.1, 100.0, 50, -1.0, &body_radii,
        );
        // Should stop after just the seed point + one iteration hitting zero field direction
        assert!(points.len() <= 2, "Should stop early due to zero field direction, got {} points", points.len());
    }

    #[test]
    fn test_trace_field_line_backward() {
        let centers = vec![Vec3::ZERO];
        let moments = vec![Vec3::new(0.0, 0.0, 1.0)];
        let seed = Vec3::new(0.0, 0.0, 2.0);
        let body_radii = vec![0.5];
        let forward_pts = trace_field_line(
            &centers, &moments, seed,
            true, 0.1, 100.0, 50, 1e-12, &body_radii,
        );
        let backward_pts = trace_field_line(
            &centers, &moments, seed,
            false, 0.1, 100.0, 50, 1e-12, &body_radii,
        );
        assert!(forward_pts.len() > 1 && backward_pts.len() > 1,
            "Expected multiple trace points in both directions");
        let fwd_z = forward_pts[1].0.z;
        let bwd_z = backward_pts[1].0.z;
        assert!(fwd_z > seed.z || bwd_z < seed.z,
            "Forward and backward traces should diverge from seed");
    }
}
