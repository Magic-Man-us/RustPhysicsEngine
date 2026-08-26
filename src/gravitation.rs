//! Newtonian gravity and two-body orbits.
//!
//! The inverse-square force and its potential energy, the field of a point
//! mass, escape and circular orbital velocity, and Kepler's third law in
//! both directions. The vis-viva equation `v² = μ(2/r − 1/a)` ties speed
//! to position on any conic orbit, and the specific orbital energy fixes
//! which conic it is.
//!
//! Also the Roche limit, the Hill sphere, the Schwarzschild radius and
//! gravitational time dilation -- the last two are the points at which
//! Newtonian gravity stops being enough; see [`crate::general_relativity`].
//!
//! For orbits propagated rather than characterised, and for transfers
//! between them, see [`crate::astrophysics`].

use crate::math::{Vec3, constants};

/// Gravitational force magnitude between two masses: F = G * m1 * m2 / r^2
pub fn gravitational_force(m1: f64, m2: f64, distance: f64) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    constants::G * m1 * m2 / (distance * distance)
}

/// Gravitational force vector from body1 toward body2.
pub fn gravitational_force_vec(m1: f64, pos1: Vec3, m2: f64, pos2: Vec3) -> Vec3 {
    let r = pos2 - pos1;
    let dist = r.magnitude();
    if dist == 0.0 {
        return Vec3::ZERO;
    }
    let f_mag = constants::G * m1 * m2 / (dist * dist);
    r.normalized() * f_mag
}

/// Gravitational potential energy: U = -G * m1 * m2 / r
pub fn gravitational_potential_energy(m1: f64, m2: f64, distance: f64) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    -constants::G * m1 * m2 / distance
}

/// Gravitational field strength at distance r from mass M: g = G * M / r^2
pub fn gravitational_field(mass: f64, distance: f64) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    constants::G * mass / (distance * distance)
}

/// Escape velocity from a body of mass M and radius r: v = sqrt(2GM/r)
pub fn escape_velocity(mass: f64, radius: f64) -> f64 {
    assert!(radius > 0.0, "radius must be positive");
    (2.0 * constants::G * mass / radius).sqrt()
}

/// Orbital velocity for a circular orbit: v = sqrt(GM/r)
pub fn orbital_velocity(central_mass: f64, orbital_radius: f64) -> f64 {
    assert!(orbital_radius > 0.0, "orbital_radius must be positive");
    (constants::G * central_mass / orbital_radius).sqrt()
}

/// Orbital period (Kepler's third law): T = 2π * sqrt(r^3 / (G*M))
pub fn orbital_period(central_mass: f64, orbital_radius: f64) -> f64 {
    assert!(central_mass > 0.0, "central_mass must be positive");
    assert!(orbital_radius > 0.0, "orbital_radius must be positive");
    2.0 * constants::PI * (orbital_radius.powi(3) / (constants::G * central_mass)).sqrt()
}

/// Semi-major axis from orbital period (inverse Kepler's third law):
/// a = (G*M*T^2 / (4π^2))^(1/3)
pub fn semi_major_axis_from_period(central_mass: f64, period: f64) -> f64 {
    assert!(central_mass > 0.0, "central_mass must be positive");
    assert!(period > 0.0, "period must be positive");
    (constants::G * central_mass * period * period / (4.0 * constants::PI * constants::PI))
        .powf(1.0 / 3.0)
}

/// Schwarzschild radius of a black hole: r_s = 2GM / c^2
pub fn schwarzschild_radius(mass: f64) -> f64 {
    2.0 * constants::G * mass / (constants::C * constants::C)
}

/// Gravitational time dilation factor at distance r from mass M:
/// sqrt(1 - 2GM/(rc^2))
pub fn gravitational_time_dilation(mass: f64, distance: f64) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    (1.0 - 2.0 * constants::G * mass / (distance * constants::C * constants::C)).sqrt()
}

/// Roche limit (fluid body): d = R * (2 * ρ_M / ρ_m)^(1/3)
/// R = radius of primary, ρ_M = density of primary, ρ_m = density of satellite
pub fn roche_limit(primary_radius: f64, primary_density: f64, satellite_density: f64) -> f64 {
    assert!(satellite_density > 0.0, "satellite_density must be positive");
    primary_radius * (2.0 * primary_density / satellite_density).powf(1.0 / 3.0)
}

/// Vis-viva equation: v^2 = GM * (2/r - 1/a)
/// Returns the orbital speed at distance r for an orbit with semi-major axis a.
pub fn vis_viva(central_mass: f64, distance: f64, semi_major_axis: f64) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    assert!(semi_major_axis > 0.0, "semi_major_axis must be positive");
    (constants::G * central_mass * (2.0 / distance - 1.0 / semi_major_axis)).sqrt()
}

/// Specific orbital energy: ε = -GM / (2a)
pub fn specific_orbital_energy(central_mass: f64, semi_major_axis: f64) -> f64 {
    assert!(semi_major_axis > 0.0, "semi_major_axis must be positive");
    -constants::G * central_mass / (2.0 * semi_major_axis)
}

/// Hill sphere radius: r_H ≈ a * (m / (3M))^(1/3)
pub fn hill_sphere_radius(semi_major_axis: f64, orbiting_mass: f64, central_mass: f64) -> f64 {
    assert!(central_mass > 0.0, "central_mass must be positive");
    semi_major_axis * (orbiting_mass / (3.0 * central_mass)).powf(1.0 / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_gravitational_force() {
        let f = gravitational_force(5.97e24, 7.35e22, 3.84e8);
        assert!(approx_rel(f, 1.98e20, 0.02));
    }

    #[test]
    fn test_escape_velocity_earth() {
        let v = escape_velocity(5.97e24, 6.371e6);
        assert!(approx_rel(v, 11186.0, 0.01));
    }

    #[test]
    fn test_orbital_velocity() {
        let v = orbital_velocity(5.97e24, 6.771e6);
        assert!(approx_rel(v, 7670.0, 0.01));
    }

    #[test]
    fn test_orbital_period() {
        // ISS orbit ~408km altitude
        let t = orbital_period(5.97e24, 6.771e6);
        assert!(approx_rel(t, 5549.0, 0.01));
    }

    #[test]
    fn test_schwarzschild_radius_sun() {
        let r = schwarzschild_radius(1.989e30);
        assert!(approx_rel(r, 2953.0, 0.01));
    }

    #[test]
    fn test_gravitational_potential_energy() {
        let u = gravitational_potential_energy(5.97e24, 1.0, 6.371e6);
        assert!(u < 0.0);
    }

    #[test]
    fn test_gravitational_field_earth_surface() {
        // g = GM/r^2, Earth surface ≈ 9.82 m/s^2
        let g = gravitational_field(5.97e24, 6.371e6);
        assert!(approx_rel(g, 9.82, 0.01));
    }

    #[test]
    fn test_gravitational_force_vec() {
        // Two 1 kg masses 1 m apart along x-axis
        let pos1 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
        let pos2 = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
        let f = gravitational_force_vec(1.0, pos1, 1.0, pos2);
        // Force should point in +x direction (toward pos2)
        assert!(approx_rel(f.x, constants::G, 1e-6));
        assert!(f.y.abs() < 1e-30);
        assert!(f.z.abs() < 1e-30);
    }

    #[test]
    fn test_gravitational_force_vec_zero_distance() {
        let pos = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
        let f = gravitational_force_vec(1.0, pos, 1.0, pos);
        assert!(f.x.abs() < 1e-30);
        assert!(f.y.abs() < 1e-30);
        assert!(f.z.abs() < 1e-30);
    }

    #[test]
    fn test_gravitational_time_dilation() {
        // Far from any mass → factor ≈ 1
        let factor = gravitational_time_dilation(1.0, 1e20);
        assert!(approx_rel(factor, 1.0, 1e-6));
        // On Earth's surface, factor ≈ 1 - 6.95e-10 (very close to 1 but < 1)
        let factor_earth = gravitational_time_dilation(5.97e24, 6.371e6);
        assert!(factor_earth < 1.0);
        assert!(factor_earth > 0.999_999_999);
    }

    #[test]
    fn test_hill_sphere_radius() {
        // Earth orbiting Sun: a=1.496e11 m, m_earth=5.97e24, M_sun=1.989e30
        // r_H ≈ 1.496e11 * (5.97e24 / (3 * 1.989e30))^(1/3) ≈ 1.5e9 m
        let r_h = hill_sphere_radius(1.496e11, 5.97e24, 1.989e30);
        assert!(approx_rel(r_h, 1.496e9, 0.02));
    }

    #[test]
    fn test_roche_limit() {
        // Earth-Moon: R_earth=6.371e6, ρ_earth≈5515, ρ_moon≈3340
        // d = R*(2*ρ_M/ρ_m)^(1/3) = 6.371e6 * (2*5515/3340)^(1/3)
        let d = roche_limit(6.371e6, 5515.0, 3340.0);
        // 6.371e6 * (11030/3340)^(1/3) = 6.371e6 * 1.4890 ≈ 9.486e6
        assert!(approx_rel(d, 9.486e6, 0.01));
    }

    #[test]
    fn test_semi_major_axis_from_period() {
        // ISS orbit period ≈ 5549 s, recover semi-major axis ≈ 6.771e6 m
        let a_recovered = semi_major_axis_from_period(5.97e24, 5549.0);
        assert!(approx_rel(a_recovered, 6.771e6, 0.01));
    }

    #[test]
    fn test_specific_orbital_energy() {
        // ε = -GM/(2a) = -6.6743e-11 * 5.97e24 / (2 * 6.771e6) ≈ -2.943e7 J/kg
        let eps = specific_orbital_energy(5.97e24, 6.771e6);
        assert!(eps < 0.0);
        assert!(approx_rel(eps, -2.943e7, 0.01));
    }

    #[test]
    fn test_vis_viva_circular() {
        // For circular orbit at ISS altitude: v = √(GM/r) ≈ 7672 m/s
        let v = vis_viva(5.97e24, 6.771e6, 6.771e6);
        assert!(approx_rel(v, 7672.0, 0.01));
    }
}
