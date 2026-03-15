use crate::math::constants;

/// Faraday constant (C/mol)
pub const FARADAY: f64 = 96485.0;

// ── Reaction Kinetics ──

/// Arrhenius equation: k = A × exp(-Ea / (RT))
pub fn arrhenius_rate(pre_exponential: f64, activation_energy: f64, temperature: f64) -> f64 {
    pre_exponential * (-activation_energy / (constants::R * temperature)).exp()
}

/// First-order half-life: t½ = ln(2) / k
pub fn half_life_first_order(rate_constant: f64) -> f64 {
    f64::ln(2.0) / rate_constant
}

/// First-order concentration decay: [A] = [A]₀ × e^(-kt)
pub fn concentration_first_order(c0: f64, rate_constant: f64, time: f64) -> f64 {
    c0 * (-rate_constant * time).exp()
}

/// Second-order integrated rate law: 1/[A] = 1/[A]₀ + kt, returns [A]
pub fn concentration_second_order(c0: f64, rate_constant: f64, time: f64) -> f64 {
    1.0 / (1.0 / c0 + rate_constant * time)
}

/// General rate law: r = k × Π([Ci]^ni)
pub fn reaction_rate(k: f64, concentrations: &[f64], orders: &[f64]) -> f64 {
    assert_eq!(
        concentrations.len(),
        orders.len(),
        "concentrations and orders must have equal length"
    );
    let product: f64 = concentrations
        .iter()
        .zip(orders.iter())
        .map(|(c, n)| c.powf(*n))
        .product();
    k * product
}

// ── Thermochemistry ──

/// Gibbs free energy: ΔG = ΔH - TΔS
pub fn gibbs_free_energy(enthalpy: f64, temperature: f64, entropy: f64) -> f64 {
    enthalpy - temperature * entropy
}

/// Equilibrium constant from Gibbs energy: K = exp(-ΔG / (RT))
pub fn equilibrium_constant_from_gibbs(delta_g: f64, temperature: f64) -> f64 {
    (-delta_g / (constants::R * temperature)).exp()
}

/// Van't Hoff equation: ln(K2/K1) = -ΔH/R × (1/T2 - 1/T1), returns K2
pub fn vant_hoff(k1: f64, delta_h: f64, t1: f64, t2: f64) -> f64 {
    let exponent = -delta_h / constants::R * (1.0 / t2 - 1.0 / t1);
    k1 * exponent.exp()
}

/// Hess's law: ΔH_rxn = Σ(ci × ΔHi)
pub fn hess_law(enthalpies: &[f64], coefficients: &[f64]) -> f64 {
    assert_eq!(
        enthalpies.len(),
        coefficients.len(),
        "enthalpies and coefficients must have equal length"
    );
    enthalpies
        .iter()
        .zip(coefficients.iter())
        .map(|(h, c)| c * h)
        .sum()
}

// ── Electrochemistry ──

/// Nernst equation: E = E° - (RT / (nF)) × ln(Q)
pub fn nernst_potential(
    e_standard: f64,
    temperature: f64,
    n_electrons: f64,
    reaction_quotient: f64,
) -> f64 {
    e_standard
        - (constants::R * temperature / (n_electrons * FARADAY)) * reaction_quotient.ln()
}

/// Cell potential: E_cell = E_cathode - E_anode
pub fn cell_potential(e_cathode: f64, e_anode: f64) -> f64 {
    e_cathode - e_anode
}

/// Faraday's law of electrolysis: m = (I × t × M) / (n × F)
pub fn faraday_electrolysis(
    current: f64,
    time: f64,
    molar_mass: f64,
    n_electrons: f64,
) -> f64 {
    (current * time * molar_mass) / (n_electrons * FARADAY)
}

// ── Solution Chemistry ──

/// pH = -log₁₀([H⁺])
pub fn ph(h_concentration: f64) -> f64 {
    -h_concentration.log10()
}

/// pOH = -log₁₀([OH⁻])
pub fn poh(oh_concentration: f64) -> f64 {
    -oh_concentration.log10()
}

/// [H⁺] = 10^(-pH)
pub fn h_from_ph(ph: f64) -> f64 {
    10.0_f64.powf(-ph)
}

/// Osmotic pressure: Π = iMRT
pub fn osmotic_pressure(molarity: f64, temperature: f64, i_factor: f64) -> f64 {
    i_factor * molarity * constants::R * temperature
}

/// Molarity: M = n / V
pub fn molarity(moles: f64, volume_liters: f64) -> f64 {
    moles / volume_liters
}

/// Dilution: C2 = C1 × V1 / V2
pub fn dilution(c1: f64, v1: f64, v2: f64) -> f64 {
    c1 * v1 / v2
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_rel(a: f64, b: f64, rel_tol: f64) -> bool {
        if b == 0.0 {
            a.abs() < TOLERANCE
        } else {
            ((a - b) / b).abs() < rel_tol
        }
    }

    // ── Reaction Kinetics ──

    #[test]
    fn test_arrhenius_rate() {
        let k = arrhenius_rate(1e13, 75000.0, 300.0);
        let expected = 1e13 * (-75000.0 / (constants::R * 300.0)).exp();
        assert!(approx_rel(k, expected, 1e-9));
    }

    #[test]
    fn test_half_life_first_order() {
        let t = half_life_first_order(0.05);
        assert!(approx_rel(t, f64::ln(2.0) / 0.05, 1e-9));
    }

    #[test]
    fn test_concentration_first_order() {
        let c = concentration_first_order(1.0, 0.1, 10.0);
        let expected = (-1.0_f64).exp();
        assert!(approx_rel(c, expected, 1e-9));
    }

    #[test]
    fn test_concentration_second_order() {
        let c = concentration_second_order(1.0, 0.5, 2.0);
        let expected = 1.0 / (1.0 + 0.5 * 2.0);
        assert!(approx(c, expected));
    }

    #[test]
    fn test_reaction_rate() {
        let r = reaction_rate(0.5, &[2.0, 3.0], &[1.0, 2.0]);
        let expected = 0.5 * 2.0 * 9.0;
        assert!(approx(r, expected));
    }

    #[test]
    #[should_panic(expected = "concentrations and orders must have equal length")]
    fn test_reaction_rate_mismatched_lengths() {
        reaction_rate(1.0, &[1.0, 2.0], &[1.0]);
    }

    // ── Thermochemistry ──

    #[test]
    fn test_gibbs_free_energy() {
        let dg = gibbs_free_energy(-100000.0, 298.15, -200.0);
        let expected = -100000.0 - 298.15 * (-200.0);
        assert!(approx(dg, expected));
    }

    #[test]
    fn test_equilibrium_constant_from_gibbs() {
        let k = equilibrium_constant_from_gibbs(0.0, 298.15);
        assert!(approx(k, 1.0));
    }

    #[test]
    fn test_equilibrium_constant_negative_dg() {
        let k = equilibrium_constant_from_gibbs(-5000.0, 298.15);
        let expected = (5000.0 / (constants::R * 298.15)).exp();
        assert!(approx_rel(k, expected, 1e-9));
    }

    #[test]
    fn test_vant_hoff() {
        let k2 = vant_hoff(1.0, -40000.0, 300.0, 350.0);
        let exponent = 40000.0 / constants::R * (1.0 / 350.0 - 1.0 / 300.0);
        let expected = exponent.exp();
        assert!(approx_rel(k2, expected, 1e-9));
    }

    #[test]
    fn test_hess_law() {
        let dh = hess_law(&[-100.0, 50.0, -200.0], &[1.0, -2.0, 1.0]);
        let expected = -100.0 + (-100.0) + (-200.0);
        assert!(approx(dh, expected));
    }

    #[test]
    #[should_panic(expected = "enthalpies and coefficients must have equal length")]
    fn test_hess_law_mismatched_lengths() {
        hess_law(&[1.0], &[1.0, 2.0]);
    }

    // ── Electrochemistry ──

    #[test]
    fn test_nernst_potential_standard_conditions() {
        let e = nernst_potential(0.76, 298.15, 2.0, 1.0);
        assert!(approx(e, 0.76));
    }

    #[test]
    fn test_nernst_potential_nonstandard() {
        let e = nernst_potential(0.76, 298.15, 2.0, 0.01);
        let correction = (constants::R * 298.15 / (2.0 * FARADAY)) * 0.01_f64.ln();
        let expected = 0.76 - correction;
        assert!(approx_rel(e, expected, 1e-9));
    }

    #[test]
    fn test_cell_potential() {
        let e = cell_potential(0.34, -0.76);
        assert!(approx(e, 1.10));
    }

    #[test]
    fn test_faraday_electrolysis() {
        let m = faraday_electrolysis(10.0, 3600.0, 63.546, 2.0);
        let expected = (10.0 * 3600.0 * 63.546) / (2.0 * FARADAY);
        assert!(approx_rel(m, expected, 1e-9));
    }

    // ── Solution Chemistry ──

    #[test]
    fn test_ph() {
        let p = ph(1e-7);
        assert!(approx(p, 7.0));
    }

    #[test]
    fn test_poh() {
        let p = poh(1e-7);
        assert!(approx(p, 7.0));
    }

    #[test]
    fn test_h_from_ph() {
        let h = h_from_ph(7.0);
        assert!(approx_rel(h, 1e-7, 1e-9));
    }

    #[test]
    fn test_ph_h_roundtrip() {
        let original_ph = 4.5;
        let h = h_from_ph(original_ph);
        let recovered = ph(h);
        assert!(approx(recovered, original_ph));
    }

    #[test]
    fn test_osmotic_pressure() {
        let pi = osmotic_pressure(0.1, 298.15, 2.0);
        let expected = 2.0 * 0.1 * constants::R * 298.15;
        assert!(approx_rel(pi, expected, 1e-9));
    }

    #[test]
    fn test_molarity() {
        let m = molarity(0.5, 2.0);
        assert!(approx(m, 0.25));
    }

    #[test]
    fn test_dilution() {
        let c2 = dilution(1.0, 0.5, 2.0);
        assert!(approx(c2, 0.25));
    }
}
