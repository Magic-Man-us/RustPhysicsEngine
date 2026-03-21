use crate::math::constants;

/// Lorentz factor: γ = 1 / sqrt(1 - v^2/c^2)
pub fn lorentz_factor(velocity: f64) -> f64 {
    assert!(velocity.abs() < constants::C, "velocity must be less than speed of light");
    let beta = velocity / constants::C;
    1.0 / (1.0 - beta * beta).sqrt()
}

/// Beta factor: β = v / c
pub fn beta(velocity: f64) -> f64 {
    velocity / constants::C
}

/// Time dilation: Δt = γ * Δt_proper
pub fn time_dilation(proper_time: f64, velocity: f64) -> f64 {
    lorentz_factor(velocity) * proper_time
}

/// Length contraction: L = L_proper / γ
pub fn length_contraction(proper_length: f64, velocity: f64) -> f64 {
    proper_length / lorentz_factor(velocity)
}

/// Relativistic momentum: p = γ * m * v
pub fn relativistic_momentum(mass: f64, velocity: f64) -> f64 {
    lorentz_factor(velocity) * mass * velocity
}

/// Relativistic kinetic energy: KE = (γ - 1) * m * c^2
pub fn relativistic_kinetic_energy(mass: f64, velocity: f64) -> f64 {
    (lorentz_factor(velocity) - 1.0) * mass * constants::C * constants::C
}

/// Total relativistic energy: E = γ * m * c^2
pub fn relativistic_total_energy(mass: f64, velocity: f64) -> f64 {
    lorentz_factor(velocity) * mass * constants::C * constants::C
}

/// Rest energy: E = m * c^2
pub fn rest_energy(mass: f64) -> f64 {
    mass * constants::C * constants::C
}

/// Energy-momentum relation: E^2 = (pc)^2 + (mc^2)^2
/// Returns total energy given momentum and rest mass.
pub fn energy_from_momentum(momentum: f64, mass: f64) -> f64 {
    let pc = momentum * constants::C;
    let mc2 = mass * constants::C * constants::C;
    (pc * pc + mc2 * mc2).sqrt()
}

/// Relativistic velocity addition: u = (v + u') / (1 + v*u'/c^2)
pub fn velocity_addition(v: f64, u_prime: f64) -> f64 {
    (v + u_prime) / (1.0 + v * u_prime / (constants::C * constants::C))
}

/// Lorentz transformation of position: x' = γ * (x - v*t)
pub fn lorentz_transform_x(x: f64, v: f64, t: f64) -> f64 {
    lorentz_factor(v) * (x - v * t)
}

/// Lorentz transformation of time: t' = γ * (t - v*x/c^2)
pub fn lorentz_transform_t(t: f64, v: f64, x: f64) -> f64 {
    lorentz_factor(v) * (t - v * x / (constants::C * constants::C))
}

/// Relativistic Doppler effect (approaching): f' = f * sqrt((1+β)/(1-β))
pub fn relativistic_doppler_approaching(frequency: f64, velocity: f64) -> f64 {
    let b = beta(velocity);
    assert!(b < 1.0, "velocity must be less than speed of light");
    frequency * ((1.0 + b) / (1.0 - b)).sqrt()
}

/// Relativistic Doppler effect (receding): f' = f * sqrt((1-β)/(1+β))
pub fn relativistic_doppler_receding(frequency: f64, velocity: f64) -> f64 {
    let b = beta(velocity);
    assert!(b < 1.0, "velocity must be less than speed of light");
    frequency * ((1.0 - b) / (1.0 + b)).sqrt()
}

/// Gravitational redshift: f_obs = f_emit * sqrt(1 - 2GM/(rc^2))
pub fn gravitational_redshift(emitted_freq: f64, mass: f64, radius: f64) -> f64 {
    assert!(radius > 0.0, "radius must be positive");
    emitted_freq * (1.0 - 2.0 * constants::G * mass / (radius * constants::C * constants::C)).sqrt()
}

/// Relativistic mass (apparent mass at velocity v): m_rel = γ * m_0
pub fn relativistic_mass(rest_mass: f64, velocity: f64) -> f64 {
    lorentz_factor(velocity) * rest_mass
}

/// Proper time interval from coordinate time: Δτ = Δt / γ
pub fn proper_time(coordinate_time: f64, velocity: f64) -> f64 {
    coordinate_time / lorentz_factor(velocity)
}

/// Spacetime interval: s^2 = (cΔt)^2 - Δx^2 - Δy^2 - Δz^2
pub fn spacetime_interval_squared(dt: f64, dx: f64, dy: f64, dz: f64) -> f64 {
    (constants::C * dt).powi(2) - dx * dx - dy * dy - dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_lorentz_factor_at_rest() {
        assert!(approx(lorentz_factor(0.0), 1.0, 1e-9));
    }

    #[test]
    fn test_lorentz_factor_half_c() {
        let gamma = lorentz_factor(0.5 * constants::C);
        assert!(approx_rel(gamma, 1.1547, 0.001));
    }

    #[test]
    fn test_time_dilation() {
        let dilated = time_dilation(1.0, 0.8 * constants::C);
        assert!(approx_rel(dilated, 5.0 / 3.0, 1e-6));
    }

    #[test]
    fn test_length_contraction() {
        let contracted = length_contraction(100.0, 0.6 * constants::C);
        assert!(approx(contracted, 80.0, 0.1));
    }

    #[test]
    fn test_rest_energy_electron() {
        let e = rest_energy(constants::M_ELECTRON);
        assert!(approx_rel(e, 8.187e-14, 0.01));
    }

    #[test]
    fn test_velocity_addition_classical() {
        // At low speeds, should be approximately v + u'
        let u = velocity_addition(10.0, 20.0);
        assert!(approx(u, 30.0, 0.01));
    }

    #[test]
    fn test_velocity_addition_relativistic() {
        // c + c should still be c
        let u = velocity_addition(0.9 * constants::C, 0.9 * constants::C);
        assert!(u < constants::C);
    }

    #[test]
    fn test_energy_momentum_relation_at_rest() {
        // At rest, E = mc^2
        let e = energy_from_momentum(0.0, 1.0);
        assert!(approx(e, constants::C * constants::C, 1.0));
    }

    #[test]
    fn test_spacetime_interval_lightlike() {
        // For light: ds^2 = 0
        let ds2 = spacetime_interval_squared(1.0, constants::C, 0.0, 0.0);
        assert!(approx(ds2, 0.0, 1.0));
    }

    #[test]
    fn test_proper_time() {
        let tau = proper_time(2.0, 0.8 * constants::C);
        assert!(approx(tau, 1.2, 1e-6));
    }

    // ── Tests for previously untested functions ──

    #[test]
    fn test_beta() {
        assert!(approx(beta(0.0), 0.0, 1e-15));
        assert!(approx(beta(constants::C), 1.0, 1e-9));
        assert!(approx(beta(0.5 * constants::C), 0.5, 1e-9));
    }

    #[test]
    fn test_relativistic_momentum() {
        let p = relativistic_momentum(1.0, 0.9 * constants::C);
        assert!(approx_rel(p, 6.191e8, 1e-3));
    }

    #[test]
    fn test_relativistic_momentum_low_speed() {
        let v = 100.0;
        let p = relativistic_momentum(2.0, v);
        assert!(approx(p, 2.0 * v, 1e-6));
    }

    #[test]
    fn test_relativistic_kinetic_energy() {
        let ke = relativistic_kinetic_energy(constants::M_ELECTRON, 0.8 * constants::C);
        assert!(approx_rel(ke, 5.458e-14, 1e-3));
    }

    #[test]
    fn test_relativistic_kinetic_energy_at_rest() {
        let ke = relativistic_kinetic_energy(1.0, 0.0);
        assert!(approx(ke, 0.0, 1e-9));
    }

    #[test]
    fn test_relativistic_total_energy() {
        let e = relativistic_total_energy(constants::M_ELECTRON, 0.8 * constants::C);
        assert!(approx_rel(e, 1.3645e-13, 1e-3));
    }

    #[test]
    fn test_relativistic_total_energy_at_rest() {
        let e = relativistic_total_energy(1.0, 0.0);
        assert!(approx(e, constants::C * constants::C, 1.0));
    }

    #[test]
    fn test_relativistic_mass() {
        let m = relativistic_mass(constants::M_ELECTRON, 0.9 * constants::C);
        assert!(approx_rel(m, 2.090e-30, 1e-3));
    }

    #[test]
    fn test_relativistic_mass_at_rest() {
        let m = relativistic_mass(1.0, 0.0);
        assert!(approx(m, 1.0, 1e-9));
    }

    #[test]
    fn test_lorentz_transform_x() {
        let x_prime = lorentz_transform_x(0.0, 0.0, 1.0);
        assert!(approx(x_prime, 0.0, 1e-9));

        let v = 0.5 * constants::C;
        let x_prime2 = lorentz_transform_x(1e8, v, 0.0);
        assert!(approx_rel(x_prime2, 1.154_700_538e8, 1e-6));
    }

    #[test]
    fn test_lorentz_transform_t() {
        let t_prime = lorentz_transform_t(0.0, 0.0, 0.0);
        assert!(approx(t_prime, 0.0, 1e-15));

        let v = 0.5 * constants::C;
        let t_prime2 = lorentz_transform_t(1.0, v, 0.0);
        assert!(approx_rel(t_prime2, 1.154_700_538, 1e-6));
    }

    #[test]
    fn test_relativistic_doppler_approaching() {
        let f_obs = relativistic_doppler_approaching(1e9, 0.5 * constants::C);
        assert!(approx_rel(f_obs, 1.732_050_808e9, 1e-6));
        assert!(f_obs > 1e9); // blueshift
    }

    #[test]
    fn test_relativistic_doppler_receding() {
        let f_obs = relativistic_doppler_receding(1e9, 0.5 * constants::C);
        assert!(approx_rel(f_obs, 5.773_508_076e8, 1e-6));
        assert!(f_obs < 1e9); // redshift
    }

    #[test]
    fn test_doppler_approaching_receding_inverse() {
        let f0 = 1e9;
        let v = 0.3 * constants::C;
        let f_app = relativistic_doppler_approaching(f0, v);
        let f_rec = relativistic_doppler_receding(f0, v);
        assert!(approx_rel(f_app * f_rec, f0 * f0, 1e-6));
    }

    #[test]
    fn test_gravitational_redshift() {
        let f_obs = gravitational_redshift(1e14, constants::EARTH_MASS, constants::EARTH_RADIUS);
        assert!(f_obs < 1e14);
        assert!(f_obs > 0.0);

        let ratio = f_obs / 1e14;
        assert!(approx_rel(ratio, 1.0 - 6.964e-10, 1e-6));
    }
}
