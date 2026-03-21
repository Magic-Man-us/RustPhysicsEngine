use crate::math::constants;

// ── Neutron Multiplication ──

/// Effective multiplication factor: k_eff = production_rate / loss_rate
#[must_use]
pub fn k_effective(production_rate: f64, loss_rate: f64) -> f64 {
    assert!(loss_rate > 0.0, "loss_rate must be positive");
    production_rate / loss_rate
}

/// Reactivity: ρ = (k - 1) / k
#[must_use]
pub fn reactivity(k_eff: f64) -> f64 {
    assert!(k_eff > 0.0, "k_eff must be positive");
    (k_eff - 1.0) / k_eff
}

/// Doubling time for supercritical reactor: T = l × ln(2) / (k - 1)
/// Returns f64::INFINITY if k_eff <= 1.0 (not supercritical).
#[must_use]
pub fn doubling_time(k_eff: f64, neutron_lifetime: f64) -> f64 {
    if k_eff <= 1.0 {
        return f64::INFINITY;
    }
    neutron_lifetime * 2.0_f64.ln() / (k_eff - 1.0)
}

/// Six-factor formula: k_eff = η × f × p × ε × P_FNL × P_TNL
#[must_use]
pub fn six_factor_formula(
    eta: f64,
    f: f64,
    p: f64,
    epsilon: f64,
    p_fnl: f64,
    p_tnl: f64,
) -> f64 {
    eta * f * p * epsilon * p_fnl * p_tnl
}

/// Reproduction factor: η = ν × σ_f / σ_a
#[must_use]
pub fn reproduction_factor(nu: f64, sigma_f: f64, sigma_a: f64) -> f64 {
    assert!(sigma_a > 0.0, "sigma_a must be positive");
    nu * sigma_f / sigma_a
}

// ── Neutron Diffusion ──

/// Diffusion coefficient: D = λ_tr / 3
#[must_use]
pub fn diffusion_coefficient(transport_mfp: f64) -> f64 {
    transport_mfp / 3.0
}

/// Diffusion length: L = √(D / Σ_a)
#[must_use]
pub fn diffusion_length(diffusion_coeff: f64, absorption_xs: f64) -> f64 {
    assert!(absorption_xs > 0.0, "absorption_xs must be positive");
    (diffusion_coeff / absorption_xs).sqrt()
}

/// Migration length: M = √(L² + τ) where τ is the Fermi age (slowing-down area)
#[must_use]
pub fn migration_length(diffusion_length: f64, slowing_down_length: f64) -> f64 {
    (diffusion_length * diffusion_length + slowing_down_length).sqrt()
}

/// Thermal utilization factor: f = Σ_a_fuel / Σ_a_total
#[must_use]
pub fn thermal_utilization(sigma_a_fuel: f64, sigma_a_total: f64) -> f64 {
    assert!(sigma_a_total > 0.0, "sigma_a_total must be positive");
    sigma_a_fuel / sigma_a_total
}

/// Neutron flux in a slab reactor with uniform source:
/// φ(x) = (S / Σ_a) × (1 - cosh(x/L) / cosh(a/L))
/// where L = √(D/Σ_a) and a = half-thickness (extrapolated).
#[must_use]
pub fn neutron_flux_slab(
    source: f64,
    diffusion_coeff: f64,
    sigma_a: f64,
    x: f64,
    half_thickness: f64,
) -> f64 {
    let l = diffusion_length(diffusion_coeff, sigma_a);
    let base_flux = source / sigma_a;
    base_flux * (1.0 - (x / l).cosh() / (half_thickness / l).cosh())
}

// ── Cross Sections ──

/// Macroscopic cross section from microscopic: Σ = N × σ
#[must_use]
pub fn microscopic_to_macroscopic(micro_xs: f64, number_density: f64) -> f64 {
    number_density * micro_xs
}

/// Number density from bulk density and molar mass: N = ρ × N_A / M
#[must_use]
pub fn number_density(density: f64, molar_mass: f64) -> f64 {
    assert!(molar_mass > 0.0, "molar_mass must be positive");
    density * constants::N_A / molar_mass
}

/// Mean free path for neutrons: λ = 1 / Σ
#[must_use]
pub fn mean_free_path_neutron(macro_xs: f64) -> f64 {
    assert!(macro_xs > 0.0, "macro_xs must be positive");
    1.0 / macro_xs
}

/// Neutron reaction rate: R = Σ × φ
#[must_use]
pub fn reaction_rate_neutron(macro_xs: f64, flux: f64) -> f64 {
    macro_xs * flux
}

/// 1/v cross section law for thermal neutrons: σ(E) = σ₀ × √(E₀ / E)
#[must_use]
pub fn one_over_v_xs(sigma_0: f64, e_0: f64, energy: f64) -> f64 {
    assert!(energy > 0.0, "energy must be positive");
    sigma_0 * (e_0 / energy).sqrt()
}

// ── Reactor Power ──

/// Reactor thermal power: P = R_f × E_f
#[must_use]
pub fn reactor_power(fission_rate: f64, energy_per_fission: f64) -> f64 {
    fission_rate * energy_per_fission
}

/// Burnup: BU = P × t / M (MWd/kg when units are consistent)
#[must_use]
pub fn burnup(power: f64, time: f64, mass_heavy_metal: f64) -> f64 {
    assert!(mass_heavy_metal > 0.0, "mass_heavy_metal must be positive");
    power * time / mass_heavy_metal
}

/// Decay heat fraction (Way-Wigner approximation for long prior operation):
/// P/P₀ ≈ 0.066 × t^(-0.2)
#[must_use]
pub fn decay_heat_fraction(time_after_shutdown: f64) -> f64 {
    assert!(time_after_shutdown > 0.0, "time_after_shutdown must be positive");
    const WAY_WIGNER_COEFF: f64 = 0.066;
    const WAY_WIGNER_EXP: f64 = -0.2;
    WAY_WIGNER_COEFF * time_after_shutdown.powf(WAY_WIGNER_EXP)
}

// ── Shielding ──

/// Transmission factor (uncollided): I/I₀ = e^(-Σx)
#[must_use]
pub fn transmission_factor(macro_xs: f64, thickness: f64) -> f64 {
    (-macro_xs * thickness).exp()
}

/// Half-value layer: HVL = ln(2) / Σ
#[must_use]
pub fn half_value_layer(macro_xs: f64) -> f64 {
    assert!(macro_xs > 0.0, "macro_xs must be positive");
    2.0_f64.ln() / macro_xs
}

/// Tenth-value layer: TVL = ln(10) / Σ
#[must_use]
pub fn tenth_value_layer(macro_xs: f64) -> f64 {
    assert!(macro_xs > 0.0, "macro_xs must be positive");
    10.0_f64.ln() / macro_xs
}

/// Linear buildup factor approximation for thin shields: B ≈ 1 + Σx
#[must_use]
pub fn buildup_factor_approx(macro_xs: f64, thickness: f64) -> f64 {
    1.0 + macro_xs * thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b.abs() < 1e-30 {
            return a.abs() < tol;
        }
        ((a - b) / b).abs() < tol
    }

    // ── Multiplication ──

    #[test]
    fn test_k_effective() {
        assert!(approx(k_effective(110.0, 100.0), 1.1, 1e-9));
    }

    #[test]
    fn test_reactivity_critical() {
        assert!(approx(reactivity(1.0), 0.0, 1e-9));
    }

    #[test]
    fn test_reactivity_supercritical() {
        let rho = reactivity(1.05);
        assert!(approx_rel(rho, 0.05 / 1.05, 1e-9));
    }

    #[test]
    fn test_doubling_time_supercritical() {
        let t = doubling_time(1.001, 1e-3);
        let expected = 0.6931471805599453;
        assert!(approx_rel(t, expected, 1e-9));
    }

    #[test]
    fn test_doubling_time_subcritical() {
        assert!(doubling_time(0.99, 1e-3).is_infinite());
    }

    #[test]
    fn test_doubling_time_critical() {
        assert!(doubling_time(1.0, 1e-3).is_infinite());
    }

    #[test]
    fn test_six_factor_formula() {
        let k = six_factor_formula(2.0, 0.9, 0.8, 1.05, 0.97, 0.99);
        let expected = 1.4519736;
        assert!(approx_rel(k, expected, 1e-9));
    }

    #[test]
    fn test_reproduction_factor() {
        let eta = reproduction_factor(2.5, 580.0, 680.0);
        assert!(approx_rel(eta, 2.1323529411764706, 1e-9));
    }

    // ── Diffusion ──

    #[test]
    fn test_diffusion_coefficient() {
        assert!(approx(diffusion_coefficient(3.0), 1.0, 1e-9));
    }

    #[test]
    fn test_diffusion_length() {
        let l = diffusion_length(1.0, 0.01);
        assert!(approx(l, 10.0, 1e-9));
    }

    #[test]
    fn test_migration_length() {
        // L = 10 cm, τ = 50 cm² => M = √(100 + 50) = √150
        let m = migration_length(10.0, 50.0);
        assert!(approx_rel(m, 12.24744871391589, 1e-9));
    }

    #[test]
    fn test_thermal_utilization() {
        assert!(approx(thermal_utilization(0.8, 1.0), 0.8, 1e-9));
    }

    #[test]
    fn test_neutron_flux_slab_center() {
        // At x=0, cosh(0)=1, so φ(0) = (S/Σa)×(1 - 1/cosh(a/L))
        let phi = neutron_flux_slab(1e12, 1.0, 0.01, 0.0, 50.0);
        let expected = 9.865247177786955e13;
        assert!(approx_rel(phi, expected, 1e-9));
    }

    #[test]
    fn test_neutron_flux_slab_boundary() {
        // At x = a, φ should be approximately 0
        let phi = neutron_flux_slab(1e12, 1.0, 0.01, 50.0, 50.0);
        assert!(approx(phi, 0.0, 1e-6));
    }

    // ── Cross Sections ──

    #[test]
    fn test_microscopic_to_macroscopic() {
        let sigma = microscopic_to_macroscopic(1e-24, 1e28);
        assert!(approx_rel(sigma, 1e4, 1e-9));
    }

    #[test]
    fn test_number_density() {
        // Water: ρ ≈ 1000 kg/m³, M ≈ 0.018 kg/mol
        let n = number_density(1000.0, 0.018);
        let expected = 3.345633755555556e28;
        assert!(approx_rel(n, expected, 1e-6));
    }

    #[test]
    fn test_mean_free_path_neutron() {
        assert!(approx(mean_free_path_neutron(0.5), 2.0, 1e-9));
    }

    #[test]
    fn test_reaction_rate_neutron() {
        assert!(approx(reaction_rate_neutron(0.1, 1e14), 1e13, 1e-9));
    }

    #[test]
    fn test_one_over_v_xs() {
        // At 4× the reference energy, σ should be σ₀/2
        let sigma = one_over_v_xs(100.0, 0.0253, 0.0253 * 4.0);
        assert!(approx_rel(sigma, 50.0, 1e-9));
    }

    // ── Reactor Power ──

    #[test]
    fn test_reactor_power() {
        let p = reactor_power(3.1e19, 3.2e-11);
        assert!(approx_rel(p, 3.1e19 * 3.2e-11, 1e-9));
    }

    #[test]
    fn test_burnup() {
        // 1 MW for 1 day on 1 kg = 1 MWd/kg
        assert!(approx(burnup(1.0, 1.0, 1.0), 1.0, 1e-9));
    }

    #[test]
    fn test_decay_heat_fraction() {
        // At t=1s, fraction = 0.066 × 1^(-0.2) = 0.066
        assert!(approx(decay_heat_fraction(1.0), 0.066, 1e-9));
    }

    #[test]
    fn test_decay_heat_decreases() {
        let early = decay_heat_fraction(10.0);
        let late = decay_heat_fraction(1000.0);
        assert!(early > late);
    }

    // ── Shielding ──

    #[test]
    fn test_transmission_factor() {
        // Zero thickness => full transmission
        assert!(approx(transmission_factor(1.0, 0.0), 1.0, 1e-9));
    }

    #[test]
    fn test_transmission_at_hvl() {
        let sigma = 0.5;
        let hvl = half_value_layer(sigma);
        let t = transmission_factor(sigma, hvl);
        assert!(approx(t, 0.5, 1e-9));
    }

    #[test]
    fn test_half_value_layer() {
        assert!(approx(half_value_layer(1.0), 2.0_f64.ln(), 1e-9));
    }

    #[test]
    fn test_tenth_value_layer() {
        assert!(approx(tenth_value_layer(1.0), 10.0_f64.ln(), 1e-9));
    }

    #[test]
    fn test_transmission_at_tvl() {
        let sigma = 0.3;
        let tvl = tenth_value_layer(sigma);
        let t = transmission_factor(sigma, tvl);
        assert!(approx(t, 0.1, 1e-9));
    }

    #[test]
    fn test_buildup_factor_approx() {
        assert!(approx(buildup_factor_approx(0.5, 2.0), 2.0, 1e-9));
    }

    #[test]
    fn test_buildup_factor_zero_thickness() {
        assert!(approx(buildup_factor_approx(1.0, 0.0), 1.0, 1e-9));
    }

    #[test]
    fn test_approx_rel_near_zero_b() {
        assert!(approx_rel(0.0, 0.0, 1e-6));
        assert!(!approx_rel(1.0, 0.0, 0.5));
    }
}
