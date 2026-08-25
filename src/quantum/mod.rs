//! Quantum mechanics: the elementary relations here, with the
//! wavefunction machinery and the Schrodinger solvers in submodules.

pub mod algorithms;
pub mod circuit;
pub mod schrodinger;
pub mod solid_state;
pub mod spin;
pub mod wavefunction;

use crate::math::constants;

// ── Wave-Particle Duality ──

/// de Broglie wavelength: λ = h / p = h / (m * v)
pub fn de_broglie_wavelength(mass: f64, velocity: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    assert!(velocity != 0.0, "velocity must not be zero");
    constants::H / (mass * velocity)
}

/// de Broglie wavelength from kinetic energy: λ = h / sqrt(2mE)
pub fn de_broglie_wavelength_from_energy(mass: f64, kinetic_energy: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    assert!(kinetic_energy > 0.0, "kinetic energy must be positive");
    constants::H / (2.0 * mass * kinetic_energy).sqrt()
}

/// Photon momentum: p = h / λ = h*f / c
pub fn photon_momentum(wavelength: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    constants::H / wavelength
}

/// Photon energy: E = h * f
pub fn photon_energy(frequency: f64) -> f64 {
    constants::H * frequency
}

/// Photon energy from wavelength: E = h * c / λ
pub fn photon_energy_from_wavelength(wavelength: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    constants::H * constants::C / wavelength
}

// ── Photoelectric Effect ──

/// Photoelectric effect: KE_max = h*f - φ (work function)
/// Returns max kinetic energy of emitted electron.
pub fn photoelectric_ke(frequency: f64, work_function: f64) -> f64 {
    constants::H * frequency - work_function
}

/// Threshold frequency: f_0 = φ / h
pub fn threshold_frequency(work_function: f64) -> f64 {
    assert!(work_function > 0.0, "work function must be positive");
    work_function / constants::H
}

/// Threshold wavelength: λ_0 = h * c / φ
pub fn threshold_wavelength(work_function: f64) -> f64 {
    assert!(work_function > 0.0, "work function must be positive");
    constants::H * constants::C / work_function
}

/// Stopping potential: V_s = KE_max / e
pub fn stopping_potential(max_kinetic_energy: f64) -> f64 {
    max_kinetic_energy / constants::E_CHARGE
}

// ── Uncertainty Principle ──

/// Heisenberg uncertainty principle (position-momentum): Δx * Δp ≥ ℏ/2
/// Returns minimum uncertainty in momentum given position uncertainty.
pub fn min_momentum_uncertainty(position_uncertainty: f64) -> f64 {
    assert!(position_uncertainty > 0.0, "position uncertainty must be positive");
    constants::HBAR / (2.0 * position_uncertainty)
}

/// Minimum position uncertainty given momentum uncertainty.
pub fn min_position_uncertainty(momentum_uncertainty: f64) -> f64 {
    assert!(momentum_uncertainty > 0.0, "momentum uncertainty must be positive");
    constants::HBAR / (2.0 * momentum_uncertainty)
}

/// Energy-time uncertainty: ΔE * Δt ≥ ℏ/2
pub fn min_energy_uncertainty(time_uncertainty: f64) -> f64 {
    assert!(time_uncertainty > 0.0, "time uncertainty must be positive");
    constants::HBAR / (2.0 * time_uncertainty)
}

/// Minimum time uncertainty given energy uncertainty: Δt ≥ ℏ / (2ΔE)
pub fn min_time_uncertainty(energy_uncertainty: f64) -> f64 {
    assert!(energy_uncertainty > 0.0, "energy uncertainty must be positive");
    constants::HBAR / (2.0 * energy_uncertainty)
}

// ── Hydrogen Atom / Bohr Model ──

/// Bohr radius: a_0 = ℏ^2 / (m_e * k_e * e^2)
pub fn bohr_radius() -> f64 {
    constants::HBAR * constants::HBAR
        / (constants::M_ELECTRON * constants::K_E * constants::E_CHARGE * constants::E_CHARGE)
}

/// Energy levels of hydrogen atom: E_n = -13.6 eV / n^2
/// Returns energy in Joules.
pub fn hydrogen_energy_level(n: u32) -> f64 {
    let ev_13_6 = 13.6 * constants::E_CHARGE;
    -ev_13_6 / (n as f64 * n as f64)
}

/// Energy of photon emitted in hydrogen transition: E = 13.6 eV * (1/n_f^2 - 1/n_i^2)
/// Returns energy in Joules (positive for emission when n_i > n_f).
pub fn hydrogen_transition_energy(n_initial: u32, n_final: u32) -> f64 {
    let ev_13_6 = 13.6 * constants::E_CHARGE;
    ev_13_6 * (1.0 / (n_final as f64).powi(2) - 1.0 / (n_initial as f64).powi(2))
}

/// Wavelength of photon from hydrogen transition (Rydberg formula):
/// 1/λ = R_H * (1/n_f^2 - 1/n_i^2)
pub fn hydrogen_transition_wavelength(n_initial: u32, n_final: u32) -> f64 {
    let energy = hydrogen_transition_energy(n_initial, n_final);
    if energy <= 0.0 {
        return f64::INFINITY;
    }
    constants::H * constants::C / energy
}

/// Orbital radius of nth level in hydrogen: r_n = n^2 * a_0
pub fn hydrogen_orbital_radius(n: u32) -> f64 {
    (n as f64).powi(2) * bohr_radius()
}

/// Orbital velocity in nth Bohr orbit: v_n = e^2 / (4πε_0 * n * ℏ)
pub fn hydrogen_orbital_velocity(n: u32) -> f64 {
    constants::K_E * constants::E_CHARGE * constants::E_CHARGE
        / (n as f64 * constants::HBAR)
}

// ── Quantum Tunneling ──

/// Transmission coefficient for a rectangular barrier (approximate, E < V):
/// T ≈ e^(-2κL) where κ = sqrt(2m(V-E)) / ℏ
pub fn tunneling_transmission(mass: f64, barrier_height: f64, particle_energy: f64, barrier_width: f64) -> f64 {
    if particle_energy >= barrier_height {
        return 1.0;
    }
    let kappa = (2.0 * mass * (barrier_height - particle_energy)).sqrt() / constants::HBAR;
    (-2.0 * kappa * barrier_width).exp()
}

// ── Particle in a Box ──

/// Energy levels of a particle in a 1D infinite potential well:
/// E_n = n^2 * π^2 * ℏ^2 / (2 * m * L^2)
pub fn particle_in_box_energy(n: u32, mass: f64, box_length: f64) -> f64 {
    assert!(n > 0, "quantum number n must be positive");
    assert!(mass > 0.0, "mass must be positive");
    assert!(box_length > 0.0, "box length must be positive");
    let n = n as f64;
    n * n * constants::PI * constants::PI * constants::HBAR * constants::HBAR
        / (2.0 * mass * box_length * box_length)
}

/// Zero-point energy (ground state, n=1):
pub fn zero_point_energy(mass: f64, box_length: f64) -> f64 {
    particle_in_box_energy(1, mass, box_length)
}

// ── Compton Scattering ──

/// Compton wavelength shift: Δλ = (h / (m_e * c)) * (1 - cos(θ))
pub fn compton_wavelength_shift(scattering_angle_rad: f64) -> f64 {
    (constants::H / (constants::M_ELECTRON * constants::C)) * (1.0 - scattering_angle_rad.cos())
}

/// Compton wavelength of the electron: λ_C = h / (m_e * c)
pub fn compton_wavelength_electron() -> f64 {
    constants::H / (constants::M_ELECTRON * constants::C)
}

// ── Blackbody Radiation ──

/// Wien's displacement law: λ_max = b / T where b ≈ 2.898e-3 m·K
pub fn wien_peak_wavelength(temperature: f64) -> f64 {
    assert!(temperature > 0.0, "temperature must be positive");
    2.898e-3 / temperature
}

/// Stefan-Boltzmann law (total power): P = σ * A * T^4
pub fn blackbody_power(area: f64, temperature: f64) -> f64 {
    constants::SIGMA * area * temperature.powi(4)
}

/// Planck's law (spectral radiance): B(λ,T) = (2hc^2/λ^5) / (e^(hc/(λkT)) - 1)
pub fn planck_spectral_radiance(wavelength: f64, temperature: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    assert!(temperature > 0.0, "temperature must be positive");
    let numerator = 2.0 * constants::H * constants::C * constants::C / wavelength.powi(5);
    let exponent = constants::H * constants::C / (wavelength * constants::K_B * temperature);
    numerator / (exponent.exp() - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b == 0.0 { return a.abs() < tol; }
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_de_broglie_electron() {
        // Electron at 1e6 m/s
        let wl = de_broglie_wavelength(constants::M_ELECTRON, 1e6);
        assert!(approx_rel(wl, 7.27e-10, 0.02));
    }

    #[test]
    fn test_photoelectric_effect() {
        // UV light at 1e15 Hz, work function 4 eV
        let ke = photoelectric_ke(1e15, 4.0 * constants::E_CHARGE);
        assert!(ke > 0.0);
    }

    #[test]
    fn test_hydrogen_ground_state_energy() {
        let e = hydrogen_energy_level(1);
        assert!(approx_rel(e, -13.6 * constants::E_CHARGE, 0.01));
    }

    #[test]
    fn test_hydrogen_transition_lyman_alpha() {
        let wl = hydrogen_transition_wavelength(2, 1);
        assert!(approx_rel(wl, 121.6e-9, 0.02));
    }

    #[test]
    fn test_bohr_radius() {
        let a0 = bohr_radius();
        assert!(approx_rel(a0, 5.29e-11, 0.02));
    }

    #[test]
    fn test_uncertainty_principle() {
        let dp = min_momentum_uncertainty(1e-10);
        assert!(dp > 0.0);
        let dx = min_position_uncertainty(dp);
        assert!(approx_rel(dx, 1e-10, 0.01));
    }

    #[test]
    fn test_compton_wavelength() {
        let lc = compton_wavelength_electron();
        assert!(approx_rel(lc, 2.426e-12, 0.02));
    }

    #[test]
    fn test_wien_sun() {
        // Sun surface ~5778K → peak ~502nm
        let wl = wien_peak_wavelength(5778.0);
        assert!(approx_rel(wl, 501.5e-9, 0.02));
    }

    #[test]
    fn test_particle_in_box() {
        // Electron in 1nm box, n=1
        let e = particle_in_box_energy(1, constants::M_ELECTRON, 1e-9);
        assert!(approx_rel(e, 6.024e-20, 0.05));
    }

    #[test]
    fn test_tunneling_zero_barrier() {
        let t = tunneling_transmission(constants::M_ELECTRON, 1.0, 2.0, 1e-10);
        assert!((t - 1.0).abs() < 1e-9);
    }

    // ── Tests for previously untested functions ──

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_de_broglie_wavelength_from_energy() {
        let ke = 1.0 * constants::E_CHARGE; // 1 eV
        let wl = de_broglie_wavelength_from_energy(constants::M_ELECTRON, ke);
        assert!(approx_rel(wl, 1.226_39e-9, 1e-3));
    }

    #[test]
    fn test_photon_energy() {
        let e = photon_energy(5e14);
        assert!(approx_rel(e, 3.313_035_075e-19, 1e-6));
    }

    #[test]
    fn test_photon_energy_from_wavelength() {
        let e = photon_energy_from_wavelength(500e-9);
        assert!(approx_rel(e, 3.972_891e-19, 1e-4));
    }

    #[test]
    fn test_photon_momentum() {
        let p = photon_momentum(500e-9);
        assert!(approx_rel(p, 1.325_214_03e-27, 1e-6));
    }

    #[test]
    fn test_threshold_frequency() {
        let work = 4.0 * constants::E_CHARGE;
        let f0 = threshold_frequency(work);
        assert!(approx_rel(f0, 9.6719e14, 1e-4));
    }

    #[test]
    fn test_threshold_wavelength() {
        let work = 4.0 * constants::E_CHARGE;
        let lam = threshold_wavelength(work);
        assert!(approx_rel(lam, 3.0996e-7, 1e-3));
    }

    #[test]
    fn test_stopping_potential() {
        let ke_max = 2.0 * constants::E_CHARGE;
        let vs = stopping_potential(ke_max);
        assert!(approx(vs, 2.0, 1e-6));
    }

    #[test]
    fn test_min_energy_uncertainty() {
        let de = min_energy_uncertainty(1e-15);
        assert!(approx_rel(de, 5.272_859_085e-20, 1e-6));
    }

    #[test]
    fn test_min_time_uncertainty() {
        let dt = min_time_uncertainty(1.0 * constants::E_CHARGE);
        assert!(approx_rel(dt, 3.2918e-16, 1e-3));
    }

    #[test]
    fn test_hydrogen_transition_energy() {
        let e = hydrogen_transition_energy(2, 1);
        assert!(approx_rel(e, 1.634_22e-18, 1e-3));
    }

    #[test]
    fn test_hydrogen_orbital_radius() {
        let r1 = hydrogen_orbital_radius(1);
        assert!(approx_rel(r1, 5.29e-11, 0.01));
        let r2 = hydrogen_orbital_radius(2);
        assert!(approx_rel(r2, 2.116e-10, 0.01));
    }

    #[test]
    fn test_hydrogen_orbital_velocity() {
        let v1 = hydrogen_orbital_velocity(1);
        assert!(approx_rel(v1, 2.187_69e6, 1e-3));
        let v2 = hydrogen_orbital_velocity(2);
        assert!(approx_rel(v2, 1.093_85e6, 1e-3));
    }

    #[test]
    fn test_compton_wavelength_shift() {
        let shift_90 = compton_wavelength_shift(constants::PI / 2.0);
        assert!(approx_rel(shift_90, 2.4263e-12, 1e-3));

        let shift_0 = compton_wavelength_shift(0.0);
        assert!(approx(shift_0, 0.0, 1e-20));
    }

    #[test]
    fn test_blackbody_power() {
        let p = blackbody_power(1.0, 5778.0);
        assert!(approx_rel(p, 6.320e7, 1e-3));
    }

    #[test]
    fn test_planck_spectral_radiance() {
        let b = planck_spectral_radiance(500e-9, 5778.0);
        assert!(b > 0.0);
        assert!(b.is_finite());
    }

    #[test]
    fn test_zero_point_energy() {
        let e_zp = zero_point_energy(constants::M_ELECTRON, 1e-9);
        assert!(approx_rel(e_zp, 6.024e-20, 0.05));
    }

    #[test]
    fn test_hydrogen_transition_wavelength_downward() {
        let lambda = hydrogen_transition_wavelength(1, 2);
        assert!(lambda.is_infinite());
    }

    #[test]
    fn test_tunneling_transmission_above_barrier() {
        let t = tunneling_transmission(constants::M_ELECTRON, 1.0e-19, 2.0e-19, 1.0e-10);
        assert!((t - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tunneling_transmission_below_barrier() {
        let t = tunneling_transmission(constants::M_ELECTRON, 2.0e-19, 1.0e-19, 1.0e-10);
        assert!(t > 0.0 && t < 1.0);
    }
}
