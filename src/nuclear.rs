//! Radioactive decay, nuclear binding, and dosimetry.
//!
//! Exponential decay in its several parameterisations -- decay constant,
//! half-life, mean lifetime -- and activity. Binding energy from the mass
//! defect, binding energy per nucleon (the curve whose peak at iron-56 is
//! why both fission and fusion release energy), and reaction Q-values.
//!
//! Nuclear size follows `R = R₀A^(1/3)`, giving a roughly constant
//! nuclear density. Dosimetry covers absorbed and equivalent dose and the
//! inverse-square falloff of intensity with distance.
//!
//! For reactor-scale neutron transport see [`crate::neutronics`].

use crate::math::constants;

// ── Radioactive Decay ──

/// Number of remaining nuclei: N(t) = N_0 * e^(-λt)
pub fn remaining_nuclei(initial_count: f64, decay_constant: f64, time: f64) -> f64 {
    initial_count * (-decay_constant * time).exp()
}

/// Activity (decay rate): A = λ * N = λ * N_0 * e^(-λt)
pub fn activity(initial_count: f64, decay_constant: f64, time: f64) -> f64 {
    decay_constant * remaining_nuclei(initial_count, decay_constant, time)
}

/// Half-life from decay constant: t_1/2 = ln(2) / λ
pub fn half_life(decay_constant: f64) -> f64 {
    assert!(decay_constant > 0.0, "decay constant must be positive");
    2.0_f64.ln() / decay_constant
}

/// Decay constant from half-life: λ = ln(2) / t_1/2
pub fn decay_constant(half_life: f64) -> f64 {
    assert!(half_life > 0.0, "half-life must be positive");
    2.0_f64.ln() / half_life
}

/// Mean lifetime: τ = 1 / λ
pub fn mean_lifetime(decay_constant: f64) -> f64 {
    assert!(decay_constant > 0.0, "decay constant must be positive");
    1.0 / decay_constant
}

/// Number of nuclei remaining after n half-lives: N = N_0 / 2^n
pub fn remaining_after_half_lives(initial_count: f64, num_half_lives: f64) -> f64 {
    initial_count / 2.0_f64.powf(num_half_lives)
}

/// Number of half-lives elapsed: n = t / t_1/2
pub fn num_half_lives(time: f64, half_life: f64) -> f64 {
    assert!(half_life > 0.0, "half-life must be positive");
    time / half_life
}

// ── Nuclear Binding Energy ──

/// Mass defect: Δm = Z*m_p + N*m_n - M_nucleus
pub fn mass_defect(
    num_protons: u32,
    num_neutrons: u32,
    nucleus_mass: f64,
) -> f64 {
    num_protons as f64 * constants::M_PROTON + num_neutrons as f64 * constants::M_NEUTRON
        - nucleus_mass
}

/// Binding energy from mass defect: E_b = Δm * c^2
pub fn binding_energy(mass_defect: f64) -> f64 {
    mass_defect * constants::C * constants::C
}

/// Binding energy per nucleon: E_b / A
pub fn binding_energy_per_nucleon(total_binding_energy: f64, mass_number: u32) -> f64 {
    assert!(mass_number > 0, "mass number must be positive");
    total_binding_energy / mass_number as f64
}

/// Q-value of a nuclear reaction: Q = (m_reactants - m_products) * c^2
pub fn q_value(reactant_mass: f64, product_mass: f64) -> f64 {
    (reactant_mass - product_mass) * constants::C * constants::C
}

// ── Nuclear Energy ──

/// Energy released from mass conversion: E = Δm * c^2
pub fn mass_energy(mass: f64) -> f64 {
    mass * constants::C * constants::C
}

/// Energy from fission/fusion (given mass difference in amu):
/// E = Δm(amu) * 931.5 MeV
pub fn energy_from_amu(mass_difference_amu: f64) -> f64 {
    mass_difference_amu * 931.5 * 1e6 * constants::E_CHARGE
}

// ── Cross Section and Reaction Rates ──

/// Reaction rate: R = n * σ * Φ
/// n = target number density, σ = cross section, Φ = flux
pub fn reaction_rate(number_density: f64, cross_section: f64, flux: f64) -> f64 {
    number_density * cross_section * flux
}

/// Mean free path in nuclear context: λ = 1 / (n * σ)
pub fn nuclear_mean_free_path(number_density: f64, cross_section: f64) -> f64 {
    assert!(number_density > 0.0, "number density must be positive");
    assert!(cross_section > 0.0, "cross section must be positive");
    1.0 / (number_density * cross_section)
}

// ── Nuclear Radius ──

/// Nuclear radius (empirical): R = R_0 * A^(1/3) where R_0 ≈ 1.2 fm
pub fn nuclear_radius(mass_number: u32) -> f64 {
    1.2e-15 * (mass_number as f64).powf(1.0 / 3.0)
}

/// Nuclear density (approximately constant): ρ ≈ 3m_p / (4π * R_0^3)
pub fn nuclear_density() -> f64 {
    let r0 = 1.2e-15;
    3.0 * constants::M_PROTON / (4.0 * constants::PI * r0 * r0 * r0)
}

// ── Radiation ──

/// Absorbed dose: D = E / m (Gray, Gy = J/kg)
pub fn absorbed_dose(energy: f64, mass: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    energy / mass
}

/// Equivalent dose: H = D * w_R (Sievert, Sv)
/// w_R is the radiation weighting factor
pub fn equivalent_dose(absorbed_dose: f64, weighting_factor: f64) -> f64 {
    absorbed_dose * weighting_factor
}

/// Inverse square law for radiation intensity: I = I_0 * (r_0 / r)^2
pub fn radiation_intensity_distance(
    initial_intensity: f64,
    initial_distance: f64,
    new_distance: f64,
) -> f64 {
    assert!(new_distance > 0.0, "new distance must be positive");
    initial_intensity * (initial_distance / new_distance).powi(2)
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
    fn test_half_life() {
        // C-14 half-life: 5730 yr × 365.25 × 24 × 3600 = 1.80825048e11 s
        let lambda = decay_constant(1.80825048e11);
        let hl = half_life(lambda);
        assert!(approx_rel(hl, 1.80825048e11, 0.001));
    }

    #[test]
    fn test_remaining_nuclei() {
        // After one half-life, half remain
        let lambda = decay_constant(100.0);
        let n = remaining_nuclei(1000.0, lambda, 100.0);
        assert!(approx(n, 500.0, 0.1));
    }

    #[test]
    fn test_remaining_after_half_lives() {
        let n = remaining_after_half_lives(1000.0, 3.0);
        assert!(approx(n, 125.0, 0.01));
    }

    #[test]
    fn test_mass_defect_helium4() {
        // He-4: 2 protons + 2 neutrons
        let m_he4 = 6.6447e-27; // approximate
        let dm = mass_defect(2, 2, m_he4);
        assert!(dm > 0.0); // Binding releases energy
    }

    #[test]
    fn test_binding_energy() {
        let dm = 0.03_f64 * constants::AMU; // ~0.03 amu mass defect
        let e = binding_energy(dm);
        assert!(e > 0.0);
    }

    #[test]
    fn test_nuclear_radius() {
        // Iron-56
        let r = nuclear_radius(56);
        assert!(approx_rel(r, 4.59e-15, 0.05));
    }

    #[test]
    fn test_radiation_inverse_square() {
        let i = radiation_intensity_distance(100.0, 1.0, 2.0);
        assert!(approx(i, 25.0, 1e-6));
    }

    #[test]
    fn test_activity() {
        let lambda = decay_constant(100.0);
        let a = activity(1000.0, lambda, 0.0);
        // A = λ*N = (ln(2)/100)*1000 = 10*ln(2) ≈ 6.93147
        assert!(approx(a, 6.93147, 1e-3));
    }

    #[test]
    fn test_q_value() {
        // If products are lighter, Q > 0 (exothermic)
        let q = q_value(2.0 * constants::AMU, 1.99 * constants::AMU);
        assert!(q > 0.0);
    }

    #[test]
    fn test_energy_from_amu() {
        let e = energy_from_amu(1.0);
        // E = 931.5e6 * 1.602176634e-19 ≈ 1.49243e-10 J
        assert!(approx_rel(e, 1.49243e-10, 0.001));
    }

    // ── Untested functions ──

    #[test]
    fn test_mean_lifetime() {
        // λ = ln(2)/100 → τ = 1/λ = 100/ln(2) ≈ 144.27 s
        let lambda = decay_constant(100.0);
        let tau = mean_lifetime(lambda);
        // τ = 100/ln(2) ≈ 144.2695 s
        assert!(approx_rel(tau, 144.2695, 1e-4));
    }

    #[test]
    fn test_num_half_lives() {
        // t=300 s, t_1/2=100 s → n = 3.0
        let n = num_half_lives(300.0, 100.0);
        assert!(approx(n, 3.0, 1e-9));
    }

    #[test]
    fn test_binding_energy_per_nucleon() {
        // E_b=1.4e-11 J, A=56 → E_b/A ≈ 2.5e-13 J
        let bepn = binding_energy_per_nucleon(1.4e-11, 56);
        assert!(approx_rel(bepn, 2.5e-13, 0.01));
    }

    #[test]
    fn test_mass_energy() {
        // 1 kg → E = c² ≈ 8.988e16 J
        let e = mass_energy(1.0);
        // E = c² = (299792458)² ≈ 8.98755179e16 J
        assert!(approx_rel(e, 8.98755179e16, 1e-6));
    }

    #[test]
    fn test_reaction_rate() {
        // n=1e28, σ=1e-24 cm² = 1e-28 m², Φ=1e14
        // R = 1e28 * 1e-28 * 1e14 = 1e14
        let r = reaction_rate(1e28, 1e-28, 1e14);
        assert!(approx_rel(r, 1e14, 1e-9));
    }

    #[test]
    fn test_nuclear_mean_free_path() {
        // n=1e28 m⁻³, σ=1e-28 m² → λ = 1/(1e28*1e-28) = 1.0 m
        let mfp = nuclear_mean_free_path(1e28, 1e-28);
        assert!(approx(mfp, 1.0, 1e-9));
    }

    #[test]
    fn test_nuclear_density() {
        // Nuclear density is approximately 2.3e17 kg/m³
        let rho = nuclear_density();
        assert!(approx_rel(rho, 2.3e17, 0.1));
    }

    #[test]
    fn test_absorbed_dose() {
        // E=0.01 J, m=70 kg → D = 1.429e-4 Gy
        let d = absorbed_dose(0.01, 70.0);
        // D = 0.01/70 ≈ 1.42857e-4 Gy
        assert!(approx_rel(d, 1.42857e-4, 1e-4));
    }

    #[test]
    fn test_equivalent_dose() {
        // D=0.001 Gy, w_R=20 (alpha) → H = 0.02 Sv
        let h = equivalent_dose(0.001, 20.0);
        assert!(approx(h, 0.02, 1e-9));
    }
}
