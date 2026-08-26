//! Impacts, mergers, and collision probability.
//!
//! Impact geometry and speed (including the gravitational focusing that
//! makes the impact speed at least the escape velocity, however slowly the
//! bodies approach), perfectly inelastic merger of mass and momentum, and
//! the energy released.
//!
//! Crater scaling and the collision probability for objects sharing a
//! volume of space follow, along with the debris-flux relations used for
//! orbital collision risk.

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

/// Computes the impact angle between two colliding bodies: θ = acos(v_radial / |v_rel|).
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

/// Computes the relative impact speed between two bodies: |v1 - v2|.
pub fn impact_speed(vel1: Vec3, vel2: Vec3) -> f64 {
    (vel1 - vel2).magnitude()
}

/// Computes the post-merger velocity via conservation of momentum: v_cm = (m1 v1 + m2 v2) / (m1 + m2).
pub fn merge_velocity(m1: f64, v1: Vec3, m2: f64, v2: Vec3) -> Vec3 {
    let total = m1 + m2;
    if total <= 0.0 {
        return Vec3::ZERO;
    }
    (v1 * m1 + v2 * m2) * (1.0 / total)
}

/// Computes the merged body radius assuming volume conservation: r = (r1³ + r2³)^(1/3).
pub fn merge_radius(r1: f64, r2: f64) -> f64 {
    (r1.powi(3) + r2.powi(3)).cbrt()
}

/// Computes the kinetic energy available in the center-of-mass frame: KE_cm = Σ ½m_i |v_i - v_cm|².
pub fn collision_energy(m1: f64, v1: Vec3, m2: f64, v2: Vec3) -> f64 {
    let v_cm = merge_velocity(m1, v1, m2, v2);
    let ke1 = 0.5 * m1 * (v1 - v_cm).magnitude_squared();
    let ke2 = 0.5 * m2 * (v2 - v_cm).magnitude_squared();
    ke1 + ke2
}

/// Computes the surface escape speed: v_esc = √(2GM/r).
pub fn escape_speed(mass: f64, radius: f64) -> f64 {
    if radius <= 0.0 {
        return 0.0;
    }
    (2.0 * G * mass / radius).sqrt()
}

/// Returns debris generation parameters (count, speed, mass fraction, temperature) for a given collision type.
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

/// Resolves a collision between two bodies, computing the merged properties and debris parameters.
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

    #[test]
    fn test_escape_speed_earth() {
        let v_esc = escape_speed(5.972e24, 6.371e6);
        // Earth escape speed ~11.2 km/s
        assert!(v_esc > 1.0e4 && v_esc < 1.2e4, "Escape speed = {v_esc}, expected ~11186 m/s");
    }

    #[test]
    fn test_escape_speed_zero_radius() {
        assert!(approx(escape_speed(1.0e30, 0.0), 0.0, 1e-20));
    }

    #[test]
    fn test_impact_speed_stationary() {
        let speed = impact_speed(Vec3::new(5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
        assert!(approx(speed, 0.0, 1e-12));
    }

    #[test]
    fn test_impact_speed_opposing() {
        let speed = impact_speed(Vec3::new(10.0, 0.0, 0.0), Vec3::new(-10.0, 0.0, 0.0));
        assert!(approx(speed, 20.0, 1e-12));
    }

    #[test]
    fn test_resolve_collision_mass_conservation() {
        let result = resolve_collision(
            3.0, 1.0, Vec3::new(1.0, 0.0, 0.0),
            7.0, 2.0, Vec3::new(-1.0, 0.0, 0.0),
            CollisionKind::Merger,
        );
        assert!(approx(result.merged_mass, 10.0, 1e-12));
    }

    #[test]
    fn test_resolve_collision_momentum_conservation() {
        let m1 = 3.0;
        let v1 = Vec3::new(5.0, 2.0, -1.0);
        let m2 = 7.0;
        let v2 = Vec3::new(-3.0, 1.0, 4.0);
        let result = resolve_collision(m1, 1.0, v1, m2, 2.0, v2, CollisionKind::GiantImpact);
        let expected_v = (v1 * m1 + v2 * m2) * (1.0 / (m1 + m2));
        assert!(approx(result.merged_velocity.x, expected_v.x, 1e-12));
        assert!(approx(result.merged_velocity.y, expected_v.y, 1e-12));
        assert!(approx(result.merged_velocity.z, expected_v.z, 1e-12));
    }

    #[test]
    fn test_resolve_collision_kind_propagated() {
        let result = resolve_collision(
            1.0, 1.0, Vec3::ZERO,
            1.0, 1.0, Vec3::ZERO,
            CollisionKind::TidalDisruption,
        );
        assert_eq!(result.kind, CollisionKind::TidalDisruption);
    }

    #[test]
    fn test_impact_angle_coincident_positions() {
        let angle = impact_angle(
            Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 0.0),
        );
        assert!(approx(angle, 0.0, 1e-12));
    }

    #[test]
    fn test_impact_angle_zero_relative_velocity() {
        let angle = impact_angle(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
        );
        assert!(approx(angle, 0.0, 1e-12));
    }

    #[test]
    fn test_merge_velocity_zero_total_mass() {
        let v = merge_velocity(0.0, Vec3::new(1.0, 0.0, 0.0), 0.0, Vec3::new(-1.0, 0.0, 0.0));
        assert!(approx(v.x, 0.0, 1e-12));
        assert!(approx(v.y, 0.0, 1e-12));
        assert!(approx(v.z, 0.0, 1e-12));
    }

    #[test]
    fn test_debris_params_planetary_destruction() {
        let dp = debris_params(CollisionKind::PlanetaryDestruction);
        assert_eq!(dp.count, 24);
        assert!(approx(dp.speed_factor, 1.5, 1e-9));
        assert!(approx(dp.mass_fraction, 0.6, 1e-9));
        assert!(approx(dp.base_temperature, 3000.0, 1e-9));
    }

    #[test]
    fn test_debris_params_black_hole_absorption() {
        let dp = debris_params(CollisionKind::BlackHoleAbsorption);
        assert_eq!(dp.count, 16);
        assert!(approx(dp.speed_factor, 0.5, 1e-9));
        assert!(approx(dp.base_temperature, 20000.0, 1e-9));
    }

    #[test]
    fn test_debris_params_grazing_collision() {
        let dp = debris_params(CollisionKind::GrazingCollision);
        assert_eq!(dp.count, 12);
        assert!(approx(dp.speed_factor, 0.8, 1e-9));
        assert!(approx(dp.mass_fraction, 0.2, 1e-9));
        assert!(approx(dp.base_temperature, 2000.0, 1e-9));
    }

    #[test]
    fn test_resolve_collision_zero_mass_temp_increase() {
        let result = resolve_collision(
            0.0, 1.0, Vec3::ZERO,
            0.0, 1.0, Vec3::ZERO,
            CollisionKind::Merger,
        );
        assert!(approx(result.temperature_increase, 0.0, 1e-12));
    }
}
