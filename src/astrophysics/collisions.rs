use crate::math::Vec3;
use crate::math::constants::G;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionKind {
    Merger,
    PlanetaryDestruction,
    StellarMerger,
    BlackHoleAbsorption,
    TidalDisruption,
    GrazingCollision,
    GiantImpact,
}

#[derive(Debug, Clone, Copy)]
pub struct DebrisParams {
    pub count: usize,
    pub speed_factor: f64,
    pub mass_fraction: f64,
    pub base_temperature: f64,
}

#[derive(Debug, Clone)]
pub struct CollisionResult {
    pub kind: CollisionKind,
    pub merged_mass: f64,
    pub merged_velocity: Vec3,
    pub merged_radius: f64,
    pub temperature_increase: f64,
    pub debris: DebrisParams,
}

pub fn impact_angle(pos1: Vec3, vel1: Vec3, pos2: Vec3, vel2: Vec3) -> f64 {
    let delta = pos2 - pos1;
    let dist = delta.magnitude();
    if dist < 1e-12 {
        return 0.0;
    }

    let rel_vel = vel2 - vel1;
    let rel_speed = rel_vel.magnitude();
    if rel_speed < 1e-12 {
        return 0.0;
    }

    let radial_unit = delta * (1.0 / dist);
    let radial_speed = rel_vel.dot(&radial_unit).abs();
    let cos_angle = (radial_speed / rel_speed).clamp(0.0, 1.0);
    cos_angle.acos()
}

pub fn impact_speed(vel1: Vec3, vel2: Vec3) -> f64 {
    (vel1 - vel2).magnitude()
}

pub fn merge_velocity(m1: f64, v1: Vec3, m2: f64, v2: Vec3) -> Vec3 {
    let total = m1 + m2;
    if total <= 0.0 {
        return Vec3::ZERO;
    }
    (v1 * m1 + v2 * m2) * (1.0 / total)
}

pub fn merge_radius(r1: f64, r2: f64) -> f64 {
    (r1.powi(3) + r2.powi(3)).cbrt()
}

pub fn collision_energy(m1: f64, v1: Vec3, m2: f64, v2: Vec3) -> f64 {
    let v_cm = merge_velocity(m1, v1, m2, v2);
    let ke1 = 0.5 * m1 * (v1 - v_cm).magnitude_squared();
    let ke2 = 0.5 * m2 * (v2 - v_cm).magnitude_squared();
    ke1 + ke2
}

pub fn escape_speed(mass: f64, radius: f64) -> f64 {
    if radius <= 0.0 {
        return 0.0;
    }
    (2.0 * G * mass / radius).sqrt()
}

pub fn debris_params(kind: CollisionKind) -> DebrisParams {
    match kind {
        CollisionKind::Merger => DebrisParams {
            count: 6, speed_factor: 0.6, mass_fraction: 0.1, base_temperature: 1500.0,
        },
        CollisionKind::PlanetaryDestruction => DebrisParams {
            count: 24, speed_factor: 1.5, mass_fraction: 0.6, base_temperature: 3000.0,
        },
        CollisionKind::StellarMerger => DebrisParams {
            count: 32, speed_factor: 2.0, mass_fraction: 0.15, base_temperature: 15000.0,
        },
        CollisionKind::BlackHoleAbsorption => DebrisParams {
            count: 16, speed_factor: 0.5, mass_fraction: 0.1, base_temperature: 20000.0,
        },
        CollisionKind::TidalDisruption => DebrisParams {
            count: 28, speed_factor: 1.0, mass_fraction: 0.7, base_temperature: 5000.0,
        },
        CollisionKind::GrazingCollision => DebrisParams {
            count: 12, speed_factor: 0.8, mass_fraction: 0.2, base_temperature: 2000.0,
        },
        CollisionKind::GiantImpact => DebrisParams {
            count: 20, speed_factor: 1.2, mass_fraction: 0.4, base_temperature: 2500.0,
        },
    }
}

pub fn resolve_collision(
    m1: f64, r1: f64, v1: Vec3,
    m2: f64, r2: f64, v2: Vec3,
    kind: CollisionKind,
) -> CollisionResult {
    let total_mass = m1 + m2;
    let merged_vel = merge_velocity(m1, v1, m2, v2);
    let merged_rad = merge_radius(r1, r2);
    let energy = collision_energy(m1, v1, m2, v2);
    let temp_increase = if total_mass > 0.0 { energy * 100.0 / total_mass } else { 0.0 };
    let debris = debris_params(kind);

    CollisionResult {
        kind,
        merged_mass: total_mass,
        merged_velocity: merged_vel,
        merged_radius: merged_rad,
        temperature_increase: temp_increase,
        debris,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_merge_velocity_conservation() {
        let v = merge_velocity(
            2.0, Vec3::new(3.0, 0.0, 0.0),
            3.0, Vec3::new(-2.0, 0.0, 0.0),
        );
        assert!(approx(v.x, 0.0, 1e-9));
    }

    #[test]
    fn test_merge_radius_volume_conservation() {
        let r = merge_radius(3.0, 4.0);
        let v_sum = 3.0_f64.powi(3) + 4.0_f64.powi(3);
        assert!(approx(r.powi(3), v_sum, 1e-9));
    }

    #[test]
    fn test_impact_angle_head_on() {
        let angle = impact_angle(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0),
        );
        assert!(approx(angle, 0.0, 1e-6), "Head-on should be 0, got {angle}");
    }

    #[test]
    fn test_debris_params_stellar_merger() {
        let dp = debris_params(CollisionKind::StellarMerger);
        assert_eq!(dp.count, 32);
        assert!(approx(dp.speed_factor, 2.0, 1e-9));
    }

    #[test]
    fn test_collision_energy_symmetric() {
        let e = collision_energy(
            1.0, Vec3::new(10.0, 0.0, 0.0),
            1.0, Vec3::new(-10.0, 0.0, 0.0),
        );
        assert!(e > 0.0);
        // In CM frame each has v=10, so KE = 2 * 0.5 * 1.0 * 100 = 100
        assert!(approx(e, 100.0, 1e-9));
    }
}
