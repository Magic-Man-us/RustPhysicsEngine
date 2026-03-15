use crate::math::{Vec3, constants};

/// Gravitational force magnitude between two masses: F = G * m1 * m2 / r^2
pub fn gravitational_force(m1: f64, m2: f64, distance: f64) -> f64 {
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
    -constants::G * m1 * m2 / distance
}

/// Gravitational field strength at distance r from mass M: g = G * M / r^2
pub fn gravitational_field(mass: f64, distance: f64) -> f64 {
    constants::G * mass / (distance * distance)
}

/// Escape velocity from a body of mass M and radius r: v = sqrt(2GM/r)
pub fn escape_velocity(mass: f64, radius: f64) -> f64 {
    (2.0 * constants::G * mass / radius).sqrt()
}

/// Orbital velocity for a circular orbit: v = sqrt(GM/r)
pub fn orbital_velocity(central_mass: f64, orbital_radius: f64) -> f64 {
    (constants::G * central_mass / orbital_radius).sqrt()
}

/// Orbital period (Kepler's third law): T = 2π * sqrt(r^3 / (G*M))
pub fn orbital_period(central_mass: f64, orbital_radius: f64) -> f64 {
    2.0 * constants::PI * (orbital_radius.powi(3) / (constants::G * central_mass)).sqrt()
}

/// Semi-major axis from orbital period (inverse Kepler's third law):
/// a = (G*M*T^2 / (4π^2))^(1/3)
pub fn semi_major_axis_from_period(central_mass: f64, period: f64) -> f64 {
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
    (1.0 - 2.0 * constants::G * mass / (distance * constants::C * constants::C)).sqrt()
}

/// Roche limit (fluid body): d = R * (2 * ρ_M / ρ_m)^(1/3)
/// R = radius of primary, ρ_M = density of primary, ρ_m = density of satellite
pub fn roche_limit(primary_radius: f64, primary_density: f64, satellite_density: f64) -> f64 {
    primary_radius * (2.0 * primary_density / satellite_density).powf(1.0 / 3.0)
}

/// Vis-viva equation: v^2 = GM * (2/r - 1/a)
/// Returns the orbital speed at distance r for an orbit with semi-major axis a.
pub fn vis_viva(central_mass: f64, distance: f64, semi_major_axis: f64) -> f64 {
    (constants::G * central_mass * (2.0 / distance - 1.0 / semi_major_axis)).sqrt()
}

/// Specific orbital energy: ε = -GM / (2a)
pub fn specific_orbital_energy(central_mass: f64, semi_major_axis: f64) -> f64 {
    -constants::G * central_mass / (2.0 * semi_major_axis)
}

/// Hill sphere radius: r_H ≈ a * (m / (3M))^(1/3)
pub fn hill_sphere_radius(semi_major_axis: f64, orbiting_mass: f64, central_mass: f64) -> f64 {
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
}
