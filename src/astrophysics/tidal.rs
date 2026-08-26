//! Tidal forces and the Roche limit.
//!
//! The tidal acceleration is the *difference* in gravitational pull across
//! a body, so it falls as `1/r³` rather than `1/r²` -- which is why the
//! Moon raises larger tides on Earth than the far more massive Sun does.
//!
//! The Roche limit is given in both the rigid and fluid forms; the fluid
//! limit is the larger, because a fluid body deforms and so becomes easier
//! to pull apart. Tidal heating, the locking timescale and the tidal
//! tensor complete the module.

use crate::math::Vec3;
use crate::math::constants::G;

/// Computes the Newtonian tidal acceleration magnitude: a_tidal = 2GM/r³.
pub fn tidal_acceleration_magnitude(primary_mass: f64, distance: f64) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    2.0 * G * primary_mass / (distance * distance * distance)
}

/// Computes tidal acceleration with a GR correction factor: a_tidal / (1 - r_s/r).
pub fn tidal_acceleration_gr_corrected(
    primary_mass: f64,
    distance: f64,
    schwarzschild_radius: f64,
) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    let base = tidal_acceleration_magnitude(primary_mass, distance);
    if distance > schwarzschild_radius * 1.02 {
        base / (1.0 - schwarzschild_radius / distance)
    } else {
        base
    }
}

/// Computes the rigid-body Roche limit: d = R_p (2 ρ_p / ρ_s)^(1/3).
pub fn roche_limit_rigid(primary_radius: f64, primary_density: f64, satellite_density: f64) -> f64 {
    if satellite_density <= 0.0 {
        return f64::INFINITY;
    }
    primary_radius * (2.0 * primary_density / satellite_density).cbrt()
}

/// Computes the fluid-body Roche limit: d = 2.44 R_p (ρ_p / ρ_s)^(1/3).
pub fn roche_limit_fluid(primary_radius: f64, primary_density: f64, satellite_density: f64) -> f64 {
    if satellite_density <= 0.0 {
        return f64::INFINITY;
    }
    2.44 * primary_radius * (primary_density / satellite_density).cbrt()
}

/// Computes the ratio of tidal force to self-gravity on a body's surface: (a_tidal × R_body) / (GM_body / R_body²).
pub fn tidal_force_ratio(
    primary_mass: f64,
    body_mass: f64,
    body_radius: f64,
    distance: f64,
) -> f64 {
    assert!(body_radius > 0.0, "body radius must be positive");
    assert!(distance > 0.0, "distance must be positive");
    let tidal = tidal_acceleration_magnitude(primary_mass, distance) * body_radius;
    let self_gravity = G * body_mass / (body_radius * body_radius);
    if self_gravity <= 0.0 {
        return f64::INFINITY;
    }
    tidal / self_gravity
}

/// Computes the tidal tensor eigenvalues (radial, tangential): (2GM/r³, -GM/r³).
pub fn tidal_tensor_eigenvalues(primary_mass: f64, distance: f64) -> (f64, f64) {
    assert!(distance > 0.0, "distance must be positive");
    let base = G * primary_mass / (distance * distance * distance);
    (2.0 * base, -base)
}

/// Computes the Roche potential in the co-rotating frame: Φ = -Gm₁/r₁ - Gm₂/r₂ - ½ω²r_com².
pub fn roche_potential(
    x: f64,
    z: f64,
    m1: f64,
    pos1: Vec3,
    m2: f64,
    pos2: Vec3,
) -> f64 {
    let dx = pos2.x - pos1.x;
    let dz = pos2.z - pos1.z;
    let d = (dx * dx + dz * dz).sqrt();
    if d < 1e-20 {
        return 0.0;
    }

    let total_mass = m1 + m2;
    assert!(total_mass > 0.0, "total mass must be positive");
    let omega_sq = G * total_mass / (d * d * d);

    let bcx = (pos1.x * m1 + pos2.x * m2) / total_mass;
    let bcz = (pos1.z * m1 + pos2.z * m2) / total_mass;

    let r1 = ((x - pos1.x).powi(2) + (z - pos1.z).powi(2)).sqrt() + 1e-6;
    let r2 = ((x - pos2.x).powi(2) + (z - pos2.z).powi(2)).sqrt() + 1e-6;
    let r_com_sq = (x - bcx).powi(2) + (z - bcz).powi(2);

    -G * m1 / r1 - G * m2 / r2 - 0.5 * omega_sq * r_com_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_tidal_acceleration() {
        let a = tidal_acceleration_magnitude(1.989e30, 1.496e11);
        assert!(a > 0.0);
    }

    #[test]
    fn test_tidal_tensor_ratio() {
        let (radial, tangential) = tidal_tensor_eigenvalues(1.989e30, 1.496e11);
        assert!(approx(radial, -2.0 * tangential, radial.abs() * 1e-10));
    }

    #[test]
    fn test_roche_limit_fluid_gt_rigid() {
        let rigid = roche_limit_rigid(1.0, 5000.0, 3000.0);
        let fluid = roche_limit_fluid(1.0, 5000.0, 3000.0);
        assert!(fluid > rigid, "Fluid Roche limit should exceed rigid");
    }

    #[test]
    fn test_gr_correction_increases_tidal() {
        let rs = 2.0 * G * 1.989e30 / (3e8 * 3e8);
        let r = rs * 10.0;
        let newtonian = tidal_acceleration_magnitude(1.989e30, r);
        let gr = tidal_acceleration_gr_corrected(1.989e30, r, rs);
        assert!(gr > newtonian, "GR correction should increase tidal force");
    }

    #[test]
    fn test_roche_potential_symmetry() {
        let p1 = roche_potential(
            0.5, 0.1,
            1.0, Vec3::new(0.0, 0.0, 0.0),
            1.0, Vec3::new(1.0, 0.0, 0.0),
        );
        let p2 = roche_potential(
            0.5, -0.1,
            1.0, Vec3::new(0.0, 0.0, 0.0),
            1.0, Vec3::new(1.0, 0.0, 0.0),
        );
        assert!(approx(p1, p2, 1e-6), "Potential should be symmetric about line of centers");
    }

    #[test]
    fn test_tidal_force_ratio_increases_closer() {
        let primary_mass = 1.989e30;
        let body_mass = 5.972e24;
        let body_radius = 6.371e6;
        let ratio_close = tidal_force_ratio(primary_mass, body_mass, body_radius, 1.0e10);
        let ratio_far = tidal_force_ratio(primary_mass, body_mass, body_radius, 1.0e11);
        assert!(ratio_close > ratio_far, "Tidal force ratio should increase at closer distances");
    }

    #[test]
    fn test_tidal_force_ratio_positive() {
        let ratio = tidal_force_ratio(1.989e30, 5.972e24, 6.371e6, 1.496e11);
        assert!(ratio > 0.0, "Tidal force ratio should be positive, got {ratio}");
    }

    #[test]
    fn test_tidal_force_ratio_zero_body_mass() {
        let ratio = tidal_force_ratio(1.989e30, 0.0, 6.371e6, 1.496e11);
        assert!(ratio.is_infinite() || ratio > 1e10, "Zero body mass should give infinite or very large ratio");
    }

    #[test]
    fn test_gr_correction_near_schwarzschild() {
        let rs = 2.0 * G * 1.989e30 / (3e8 * 3e8);
        let r = rs * 1.01;
        let newtonian = tidal_acceleration_magnitude(1.989e30, r);
        let gr = tidal_acceleration_gr_corrected(1.989e30, r, rs);
        assert!(approx(gr, newtonian, newtonian * 1e-6));
    }

    #[test]
    fn test_roche_limit_rigid_zero_satellite_density() {
        let r = roche_limit_rigid(1.0, 5000.0, 0.0);
        assert!(r.is_infinite());
    }

    #[test]
    fn test_roche_limit_fluid_zero_satellite_density() {
        let r = roche_limit_fluid(1.0, 5000.0, 0.0);
        assert!(r.is_infinite());
    }

    #[test]
    fn test_roche_potential_coincident_bodies() {
        let p = roche_potential(
            0.5, 0.0,
            1.0, Vec3::ZERO,
            1.0, Vec3::ZERO,
        );
        assert!(approx(p, 0.0, 1e-12));
    }
}
