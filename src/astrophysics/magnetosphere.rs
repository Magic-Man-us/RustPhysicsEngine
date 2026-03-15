use crate::math::Vec3;
use crate::math::constants::PI;

pub const DEFAULT_MAX_LINES_PER_BODY: usize = 16;
pub const DEFAULT_POINTS_PER_LINE: usize = 64;
pub const DEFAULT_MIN_FIELD_STRENGTH: f64 = 1e-6;
pub const SOLAR_TEMPERATURE: f64 = 5778.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CelestialBodyType {
    Star,
    GasGiant,
    IceGiant,
    Terrestrial,
    NeutronStar,
    BlackHole,
}

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

pub fn magnetosphere_radius(collision_radius: f64, moment: f64) -> f64 {
    if moment <= 0.0 {
        return 0.0;
    }
    let r = moment.powf(1.0 / 3.0) * collision_radius * 8.0;
    r.max(collision_radius * 3.0)
}

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
}
