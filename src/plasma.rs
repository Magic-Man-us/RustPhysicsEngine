use crate::math::constants::{C, E_CHARGE, EPSILON_0, K_B, M_ELECTRON, MU_0, PI};

/// λD = √(ε₀kT / (nq²))
#[must_use]
pub fn debye_length(temperature: f64, density: f64, charge: f64) -> f64 {
    (EPSILON_0 * K_B * temperature / (density * charge * charge)).sqrt()
}

/// ωp = √(ne² / (mₑε₀))
#[must_use]
pub fn plasma_frequency_electron(density: f64) -> f64 {
    (density * E_CHARGE * E_CHARGE / (M_ELECTRON * EPSILON_0)).sqrt()
}

/// ωp = √(ne² / (m_ion × ε₀))
#[must_use]
pub fn plasma_frequency_ion(density: f64, ion_mass: f64) -> f64 {
    (density * E_CHARGE * E_CHARGE / (ion_mass * EPSILON_0)).sqrt()
}

/// ωc = eB / mₑ
#[must_use]
pub fn cyclotron_frequency_electron(b_field: f64) -> f64 {
    E_CHARGE * b_field / M_ELECTRON
}

/// ωc = qB / m
#[must_use]
pub fn cyclotron_frequency_ion(b_field: f64, ion_mass: f64, charge: f64) -> f64 {
    charge * b_field / ion_mass
}

/// rL = mv⊥ / (|q|B)
#[must_use]
pub fn larmor_radius(velocity_perp: f64, mass: f64, charge: f64, b_field: f64) -> f64 {
    mass * velocity_perp / (charge.abs() * b_field)
}

/// β = 2μ₀p / B²
#[must_use]
pub fn plasma_beta(pressure: f64, b_field: f64) -> f64 {
    2.0 * MU_0 * pressure / (b_field * b_field)
}

/// vA = B / √(μ₀ρ)
#[must_use]
pub fn alfven_speed(b_field: f64, density: f64) -> f64 {
    b_field / (MU_0 * density).sqrt()
}

/// cs = √(γkT / m)
#[must_use]
pub fn sound_speed_plasma(gamma: f64, temperature: f64, ion_mass: f64) -> f64 {
    (gamma * K_B * temperature / ion_mass).sqrt()
}

/// vth = √(2kT / m)
#[must_use]
pub fn thermal_velocity(temperature: f64, mass: f64) -> f64 {
    (2.0 * K_B * temperature / mass).sqrt()
}

/// ND = n × (4π/3)λD³
#[must_use]
pub fn debye_number(density: f64, debye_len: f64) -> f64 {
    density * (4.0 * PI / 3.0) * debye_len.powi(3)
}

/// lnΛ = ln(12π × ND)
#[must_use]
pub fn coulomb_logarithm(temperature: f64, density: f64) -> f64 {
    let lambda_d = debye_length(temperature, density, E_CHARGE);
    let nd = debye_number(density, lambda_d);
    (12.0 * PI * nd).ln()
}

/// Pm = B² / (2μ₀)
#[must_use]
pub fn magnetic_pressure(b_field: f64) -> f64 {
    b_field * b_field / (2.0 * MU_0)
}

/// vms = √(vA² + cs²)
#[must_use]
pub fn magnetosonic_speed(alfven: f64, sound: f64) -> f64 {
    (alfven * alfven + sound * sound).sqrt()
}

/// δ = c / ωp
#[must_use]
pub fn skin_depth_plasma(plasma_freq: f64) -> f64 {
    C / plasma_freq
}

/// ν = nq⁴lnΛ / (4πε₀²m²vth³)
#[must_use]
pub fn collision_frequency(
    density: f64,
    temperature: f64,
    coulomb_log: f64,
    mass: f64,
    charge: f64,
) -> f64 {
    let vth = thermal_velocity(temperature, mass);
    let numerator = density * charge.powi(4) * coulomb_log;
    let denominator = 4.0 * PI * EPSILON_0 * EPSILON_0 * mass * mass * vth.powi(3);
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::M_PROTON;

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b == 0.0 {
            return a.abs() < tol;
        }
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_debye_length() {
        // Typical fusion plasma: T = 1 keV ≈ 1.16e7 K, n = 1e20 m⁻³
        let temp = 1.16e7;
        let n = 1e20;
        let ld = debye_length(temp, n, E_CHARGE);
        // Expected ~7.4e-5 m
        assert!(approx_rel(ld, 2.35e-5, 0.05));
    }

    #[test]
    fn test_plasma_frequency_electron() {
        let n = 1e18;
        let wp = plasma_frequency_electron(n);
        // Expected ~5.64e10 rad/s
        assert!(approx_rel(wp, 5.64e10, 0.02));
    }

    #[test]
    fn test_plasma_frequency_ion() {
        let n = 1e18;
        let wp_ion = plasma_frequency_ion(n, M_PROTON);
        let wp_e = plasma_frequency_electron(n);
        // Ion frequency should be sqrt(m_e/m_ion) times electron frequency
        let ratio = wp_ion / wp_e;
        let expected_ratio = (M_ELECTRON / M_PROTON).sqrt();
        assert!(approx_rel(ratio, expected_ratio, 1e-6));
    }

    #[test]
    fn test_cyclotron_frequency_electron() {
        let b = 1.0; // 1 Tesla
        let wc = cyclotron_frequency_electron(b);
        // Expected ~1.76e11 rad/s
        assert!(approx_rel(wc, 1.76e11, 0.01));
    }

    #[test]
    fn test_cyclotron_frequency_ion() {
        let b = 1.0;
        let wc = cyclotron_frequency_ion(b, M_PROTON, E_CHARGE);
        // Expected ~9.58e7 rad/s
        assert!(approx_rel(wc, 9.58e7, 0.01));
    }

    #[test]
    fn test_larmor_radius() {
        let v_perp = 1e6; // 1e6 m/s
        let b = 1.0;
        let rl = larmor_radius(v_perp, M_ELECTRON, E_CHARGE, b);
        // rL = m_e * v / (e * B) ≈ 9.109e-31 * 1e6 / (1.602e-19 * 1) ≈ 5.69e-6 m
        assert!(approx_rel(rl, 5.69e-6, 0.01));
    }

    #[test]
    fn test_plasma_beta() {
        let pressure = 1e5; // 1 atm
        let b = 1.0;
        let beta = plasma_beta(pressure, b);
        // β = 2 * 1.257e-6 * 1e5 / 1 ≈ 0.2514
        assert!(approx_rel(beta, 0.2514, 0.01));
    }

    #[test]
    fn test_alfven_speed() {
        let b = 1e-3; // 1 mT
        let rho = 1e-12; // kg/m³ (very low density)
        let va = alfven_speed(b, rho);
        // vA = 1e-3 / sqrt(1.257e-6 * 1e-12) ≈ 8.92e5 m/s
        assert!(approx_rel(va, 8.92e5, 0.02));
    }

    #[test]
    fn test_sound_speed_plasma() {
        let gamma = 5.0 / 3.0;
        let temp = 1e6; // 1 MK
        let cs = sound_speed_plasma(gamma, temp, M_PROTON);
        // cs = sqrt(5/3 * 1.381e-23 * 1e6 / 1.673e-27) ≈ 1.17e5 m/s
        assert!(approx_rel(cs, 1.17e5, 0.02));
    }

    #[test]
    fn test_thermal_velocity() {
        let temp = 1e6;
        let vth = thermal_velocity(temp, M_ELECTRON);
        // vth = sqrt(2 * 1.381e-23 * 1e6 / 9.109e-31) ≈ 5.51e6 m/s
        assert!(approx_rel(vth, 5.51e6, 0.02));
    }

    #[test]
    fn test_debye_number() {
        let n = 1e20;
        let ld = 2.35e-5;
        let nd = debye_number(n, ld);
        // ND = 1e20 * (4π/3) * (2.35e-5)^3 ≈ 5.44e6
        assert!(approx_rel(nd, 5.44e6, 0.05));
    }

    #[test]
    fn test_coulomb_logarithm() {
        let temp = 1.16e7;
        let n = 1e20;
        let ln_lambda = coulomb_logarithm(temp, n);
        // For fusion plasmas, lnΛ is typically 15-20
        assert!(ln_lambda > 10.0 && ln_lambda < 30.0);
    }

    #[test]
    fn test_magnetic_pressure() {
        let b = 1.0;
        let pm = magnetic_pressure(b);
        // Pm = 1 / (2 * 1.257e-6) ≈ 3.98e5 Pa
        assert!(approx_rel(pm, 3.98e5, 0.01));
    }

    #[test]
    fn test_magnetosonic_speed() {
        let va = 1000.0;
        let cs = 500.0;
        let vms = magnetosonic_speed(va, cs);
        let expected = (1000.0_f64.powi(2) + 500.0_f64.powi(2)).sqrt();
        assert!(approx_rel(vms, expected, 1e-10));
    }

    #[test]
    fn test_skin_depth_plasma() {
        let wp = 5.64e10;
        let delta = skin_depth_plasma(wp);
        // δ = 2.998e8 / 5.64e10 ≈ 5.31e-3 m
        assert!(approx_rel(delta, 5.31e-3, 0.02));
    }

    #[test]
    fn test_collision_frequency() {
        let n = 1e20;
        let temp = 1.16e7;
        let ln_lambda = 17.0;
        let nu = collision_frequency(n, temp, ln_lambda, M_ELECTRON, E_CHARGE);
        // Should be a positive finite number in a reasonable range
        assert!(nu > 0.0 && nu.is_finite());
    }
}
