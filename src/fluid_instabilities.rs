//! When a fluid configuration stops being stable, and how fast it comes
//! apart.
//!
//! Each entry here is a growth rate or a threshold. Rayleigh-Taylor for a
//! heavy fluid over a light one, with the Atwood number and the most
//! unstable wavelength; Kelvin-Helmholtz for a velocity shear;
//! Rayleigh-Bénard convection through the Rayleigh number and its critical
//! value; Plateau-Rayleigh for the breakup of a liquid column into drops;
//! Richtmyer-Meshkov for a shock crossing an interface; and the Jeans
//! criterion, which is the same instability applied to a self-gravitating
//! gas cloud and so sets the mass at which a cloud collapses into a star.
//!
//! The Richardson number and its stability test cover stratified shear
//! flow.

use crate::math::constants;

// ── Constants ────────────────────────────────────────────────────────────────

/// Critical Rayleigh number for onset of convection (rigid-rigid boundaries).
const CRITICAL_RAYLEIGH: f64 = 1708.0;

/// Critical Richardson number below which shear flow becomes dynamically unstable.
const CRITICAL_RICHARDSON: f64 = 0.25;

/// Plateau-Rayleigh most-unstable wavelength coefficient: λ_max ≈ 9.02 × r
const PLATEAU_RAYLEIGH_MOST_UNSTABLE_COEFF: f64 = 9.02;

/// Nusselt-Rayleigh prefactor for turbulent natural convection (Globe & Dropkin).
const NUSSELT_PREFACTOR: f64 = 0.069;

/// Nusselt-Rayleigh exponent (1/3).
const NUSSELT_EXPONENT: f64 = 1.0 / 3.0;

/// √3 used in RT most-unstable wavelength.
const SQRT_3: f64 = 1.732_050_808_068_87;

// ── Rayleigh-Taylor Instability ──────────────────────────────────────────────

/// Growth rate of the Rayleigh-Taylor instability: γ = √(A g k).
///
/// `atwood` is the Atwood number A = (ρ₂ − ρ₁)/(ρ₂ + ρ₁), `g` the gravitational
/// acceleration, and `wavenumber` the perturbation wavenumber k.
pub fn rayleigh_taylor_growth_rate(g: f64, atwood: f64, wavenumber: f64) -> f64 {
    (atwood * g * wavenumber).sqrt()
}

/// Atwood number: A = (ρ_heavy − ρ_light) / (ρ_heavy + ρ_light).
///
/// Ranges from −1 to 1. Positive when `density_heavy > density_light`.
pub fn atwood_number(density_heavy: f64, density_light: f64) -> f64 {
    assert!((density_heavy + density_light) != 0.0, "sum of densities must be non-zero");
    (density_heavy - density_light) / (density_heavy + density_light)
}

/// Critical wavelength for capillary stabilization of RT instability:
/// λ_c = 2π √(σ / (Δρ g)).
///
/// Perturbations shorter than λ_c are stabilized by surface tension σ.
pub fn rt_critical_wavelength(surface_tension: f64, density_diff: f64, g: f64) -> f64 {
    assert!(density_diff > 0.0, "density_diff must be positive");
    assert!(g > 0.0, "gravitational acceleration must be positive");
    constants::TAU * (surface_tension / (density_diff * g)).sqrt()
}

/// Most unstable RT wavelength: λ_max = √3 × λ_c.
pub fn rt_most_unstable_wavelength(surface_tension: f64, density_diff: f64, g: f64) -> f64 {
    SQRT_3 * rt_critical_wavelength(surface_tension, density_diff, g)
}

// ── Kelvin-Helmholtz Instability ─────────────────────────────────────────────

/// Growth rate of the Kelvin-Helmholtz instability:
/// γ = k |ΔV| √(ρ₁ρ₂) / (ρ₁ + ρ₂).
pub fn kh_growth_rate(
    density1: f64,
    density2: f64,
    velocity_diff: f64,
    wavenumber: f64,
) -> f64 {
    assert!((density1 + density2) > 0.0, "sum of densities must be positive");
    wavenumber * velocity_diff.abs() * (density1 * density2).sqrt() / (density1 + density2)
}

/// Critical velocity difference for onset of KH instability:
/// ΔV_c² = (ρ₁ + ρ₂)/(ρ₁ρ₂) × [g(ρ₂ − ρ₁)/k + σk].
pub fn kh_critical_velocity(
    density1: f64,
    density2: f64,
    surface_tension: f64,
    wavenumber: f64,
    g: f64,
) -> f64 {
    assert!(density1 > 0.0, "density1 must be positive");
    assert!(density2 > 0.0, "density2 must be positive");
    assert!(wavenumber > 0.0, "wavenumber must be positive");
    let density_sum = density1 + density2;
    let density_product = density1 * density2;
    let gravity_term = g * (density2 - density1) / wavenumber;
    let capillary_term = surface_tension * wavenumber;
    ((density_sum / density_product) * (gravity_term + capillary_term)).sqrt()
}

// ── Rayleigh-Bénard Convection ───────────────────────────────────────────────

/// Thermal Rayleigh number: Ra = g β ΔT H³ / (ν α).
///
/// `beta` is the thermal expansion coefficient, `delta_t` the temperature
/// difference across the layer, `height` the layer thickness, `kinematic_viscosity`
/// is ν, and `thermal_diffusivity` is α.
pub fn rayleigh_number_thermal(
    g: f64,
    beta: f64,
    delta_t: f64,
    height: f64,
    kinematic_viscosity: f64,
    thermal_diffusivity: f64,
) -> f64 {
    assert!(kinematic_viscosity > 0.0, "kinematic_viscosity must be positive");
    assert!(thermal_diffusivity > 0.0, "thermal_diffusivity must be positive");
    g * beta * delta_t * height.powi(3) / (kinematic_viscosity * thermal_diffusivity)
}

/// Critical Rayleigh number for rigid-rigid boundaries: Ra_c = 1708.
pub fn critical_rayleigh_number() -> f64 {
    CRITICAL_RAYLEIGH
}

/// Nusselt number from Rayleigh number for turbulent natural convection
/// (simplified for air, Pr ≈ 0.71): Nu = 0.069 × Ra^(1/3).
///
/// Returns 1.0 (pure conduction) when Ra < Ra_c.
pub fn nusselt_from_rayleigh(rayleigh: f64) -> f64 {
    if rayleigh < CRITICAL_RAYLEIGH {
        return 1.0;
    }
    NUSSELT_PREFACTOR * rayleigh.powf(NUSSELT_EXPONENT)
}

/// Returns `true` if the Rayleigh number exceeds the critical value (Ra > 1708),
/// indicating onset of convective motion.
pub fn is_convecting(rayleigh: f64) -> bool {
    rayleigh > CRITICAL_RAYLEIGH
}

// ── Jeans Instability ────────────────────────────────────────────────────────

/// Jeans length: λ_J = c_s √(π / (G ρ)).
///
/// Density perturbations larger than λ_J undergo gravitational collapse.
pub fn jeans_length(sound_speed: f64, density: f64) -> f64 {
    assert!(density > 0.0, "density must be positive");
    sound_speed * (constants::PI / (constants::G * density)).sqrt()
}

/// Jeans mass: M_J = (π/6) ρ λ_J³.
pub fn jeans_mass(sound_speed: f64, density: f64) -> f64 {
    let lambda = jeans_length(sound_speed, density);
    (constants::PI / 6.0) * density * lambda.powi(3)
}

/// Jeans angular frequency: ω_J = √(4πGρ).
///
/// This is the frequency at the Jeans wavenumber k_J where ω² = k²c_s² − 4πGρ = 0.
pub fn jeans_frequency(sound_speed: f64, density: f64) -> f64 {
    let _ = sound_speed; // included for API consistency; ω_J depends only on density
    (4.0 * constants::PI * constants::G * density).sqrt()
}

// ── Plateau-Rayleigh (Capillary) Instability ─────────────────────────────────

/// Growth rate of the Plateau-Rayleigh instability (simplified):
/// γ² = (σ / (ρ r³)) × (kr)(1 − (kr)²).
///
/// Valid for kr < 1 (long-wavelength regime). Returns 0 for kr ≥ 1.
pub fn plateau_rayleigh_growth_rate(
    surface_tension: f64,
    density: f64,
    radius: f64,
    wavenumber: f64,
) -> f64 {
    assert!(density > 0.0, "density must be positive");
    assert!(radius > 0.0, "radius must be positive");
    let kr = wavenumber * radius;
    if kr >= 1.0 {
        return 0.0;
    }
    let gamma_sq = (surface_tension / (density * radius.powi(3))) * kr * (1.0 - kr * kr);
    gamma_sq.sqrt()
}

/// Critical wavelength for Plateau-Rayleigh instability: λ_c = 2πr.
///
/// Perturbations with wavelength > λ_c (i.e., longer than the circumference) are unstable.
pub fn plateau_rayleigh_critical_wavelength(radius: f64) -> f64 {
    constants::TAU * radius
}

/// Most unstable wavelength for Plateau-Rayleigh instability: λ_max ≈ 9.02 r.
pub fn plateau_rayleigh_most_unstable(radius: f64) -> f64 {
    PLATEAU_RAYLEIGH_MOST_UNSTABLE_COEFF * radius
}

// ── Richtmyer-Meshkov Instability ────────────────────────────────────────────

/// Linear growth rate of the Richtmyer-Meshkov instability:
/// dh/dt = A × ΔV × k × h₀.
///
/// Unlike RT, RM growth is linear (not exponential). Returns the velocity
/// of the perturbation amplitude growth for unit initial amplitude (h₀ = 1).
pub fn richtmyer_meshkov_growth_rate(
    atwood: f64,
    velocity_jump: f64,
    wavenumber: f64,
) -> f64 {
    atwood * velocity_jump * wavenumber
}

// ── General Stability Criteria ───────────────────────────────────────────────

/// Gradient Richardson number: Ri = g (dρ/dz) / (ρ (du/dz)²).
///
/// `density_gradient` is dρ/dz and `velocity_gradient` is du/dz.
pub fn richardson_number(
    g: f64,
    density_gradient: f64,
    density: f64,
    velocity_gradient: f64,
) -> f64 {
    assert!(density > 0.0, "density must be positive");
    assert!(velocity_gradient != 0.0, "velocity_gradient must be non-zero");
    g * density_gradient / (density * velocity_gradient * velocity_gradient)
}

/// Returns `true` if the gradient Richardson number exceeds the critical value
/// of 0.25, indicating dynamic stability against shear-driven turbulence.
pub fn is_dynamically_stable(richardson: f64) -> bool {
    richardson > CRITICAL_RICHARDSON
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn rel_error(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    // ── Rayleigh-Taylor ──

    #[test]
    fn rt_growth_rate_positive_for_heavy_on_light() {
        let atwood = atwood_number(2.0, 1.0);
        let gamma = rayleigh_taylor_growth_rate(9.81, atwood, 1.0);
        assert!(gamma > 0.0, "RT growth rate must be positive for heavy-on-light");
    }

    #[test]
    fn rt_growth_rate_value() {
        // A = 1/3, g = 9.81, k = 2.0 → γ = √(1/3 × 9.81 × 2.0) = √6.54 ≈ 2.5573
        let atwood = atwood_number(2.0, 1.0); // A = 1/3
        let gamma = rayleigh_taylor_growth_rate(9.81, atwood, 2.0);
        assert!(rel_error(gamma, 2.557_342_262_890_562) < 1e-6);
    }

    #[test]
    fn atwood_number_range() {
        let a1 = atwood_number(10.0, 1.0);
        assert!(a1 > -1.0 && a1 < 1.0, "Atwood number must be in (-1, 1) for finite densities");

        let a2 = atwood_number(1.0, 1.0);
        assert!(approx(a2, 0.0), "Equal densities must give Atwood = 0");

        let a3 = atwood_number(1.0, 2.0);
        assert!(a3 < 0.0, "Light-on-heavy must give negative Atwood");
    }

    #[test]
    fn rt_most_unstable_is_sqrt3_times_critical() {
        let sigma = 0.072;
        let drho = 100.0;
        let g = 9.81;
        let lambda_c = rt_critical_wavelength(sigma, drho, g);
        let lambda_max = rt_most_unstable_wavelength(sigma, drho, g);
        assert!(rel_error(lambda_max, SQRT_3 * lambda_c) < 1e-10);
    }

    // ── Kelvin-Helmholtz ──

    #[test]
    fn kh_growth_rate_increases_with_velocity() {
        let gamma1 = kh_growth_rate(1.0, 1.2, 1.0, 1.0);
        let gamma2 = kh_growth_rate(1.0, 1.2, 2.0, 1.0);
        assert!(
            gamma2 > gamma1,
            "KH growth rate must increase with velocity difference"
        );
    }

    #[test]
    fn kh_growth_rate_symmetric_in_velocity_sign() {
        let gamma_pos = kh_growth_rate(1.0, 1.2, 5.0, 1.0);
        let gamma_neg = kh_growth_rate(1.0, 1.2, -5.0, 1.0);
        assert!(approx(gamma_pos, gamma_neg));
    }

    // ── Rayleigh-Bénard ──

    #[test]
    fn critical_rayleigh_is_1708() {
        assert!(approx(critical_rayleigh_number(), 1708.0));
    }

    #[test]
    fn below_critical_rayleigh_not_convecting() {
        assert!(!is_convecting(1000.0));
        assert!(!is_convecting(1708.0));
    }

    #[test]
    fn above_critical_rayleigh_is_convecting() {
        assert!(is_convecting(1709.0));
        assert!(is_convecting(1e6));
    }

    #[test]
    fn nusselt_below_critical_is_one() {
        assert!(approx(nusselt_from_rayleigh(1000.0), 1.0));
    }

    #[test]
    fn nusselt_above_critical_exceeds_one() {
        let nu = nusselt_from_rayleigh(1e6);
        assert!(nu > 1.0, "Nusselt above critical Ra must exceed 1");
    }

    #[test]
    fn rayleigh_number_scales_with_height_cubed() {
        let ra1 = rayleigh_number_thermal(9.81, 3.4e-3, 10.0, 1.0, 1.5e-5, 2.0e-5);
        let ra2 = rayleigh_number_thermal(9.81, 3.4e-3, 10.0, 2.0, 1.5e-5, 2.0e-5);
        assert!(rel_error(ra2 / ra1, 8.0) < 1e-10, "Ra must scale as H³");
    }

    // ── Jeans ──

    #[test]
    fn jeans_length_positive() {
        let lambda = jeans_length(1000.0, 1e-20);
        assert!(lambda > 0.0, "Jeans length must be positive");
    }

    #[test]
    fn jeans_mass_positive() {
        let m = jeans_mass(1000.0, 1e-20);
        assert!(m > 0.0, "Jeans mass must be positive");
    }

    #[test]
    fn jeans_frequency_positive() {
        let omega = jeans_frequency(1000.0, 1e-20);
        assert!(omega > 0.0, "Jeans frequency must be positive");
    }

    #[test]
    fn jeans_frequency_value() {
        let density = 1e-18;
        let omega = jeans_frequency(1000.0, density);
        assert!(rel_error(omega, 2.896_5e-14) < 1e-3);
    }

    // ── Plateau-Rayleigh ──

    #[test]
    fn plateau_rayleigh_zero_beyond_critical() {
        let gamma = plateau_rayleigh_growth_rate(0.072, 1000.0, 0.001, 1000.1);
        assert!(approx(gamma, 0.0), "PR growth rate must be zero for kr >= 1");
    }

    #[test]
    fn plateau_rayleigh_critical_wavelength_is_circumference() {
        let r = 0.5;
        let lambda_c = plateau_rayleigh_critical_wavelength(r);
        assert!(approx(lambda_c, constants::TAU * r));
    }

    #[test]
    fn plateau_rayleigh_most_unstable_value() {
        let r = 1.0;
        let lambda_max = plateau_rayleigh_most_unstable(r);
        assert!(approx(lambda_max, PLATEAU_RAYLEIGH_MOST_UNSTABLE_COEFF));
    }

    // ── Richtmyer-Meshkov ──

    #[test]
    fn richtmyer_meshkov_linear_in_wavenumber() {
        let g1 = richtmyer_meshkov_growth_rate(0.5, 100.0, 1.0);
        let g2 = richtmyer_meshkov_growth_rate(0.5, 100.0, 2.0);
        assert!(rel_error(g2 / g1, 2.0) < 1e-10, "RM growth must scale linearly with k");
    }

    // ── Richardson ──

    #[test]
    fn richardson_above_quarter_is_stable() {
        let ri = richardson_number(9.81, 1.0, 1.0, 1.0);
        // Ri = 9.81 > 0.25
        assert!(is_dynamically_stable(ri));
    }

    #[test]
    fn richardson_below_quarter_is_unstable() {
        let ri = richardson_number(9.81, 0.001, 1.0, 10.0);
        // Ri = 9.81 * 0.001 / (1.0 * 100) = 0.0000981
        assert!(!is_dynamically_stable(ri));
    }

    #[test]
    fn richardson_at_quarter_is_not_stable() {
        assert!(!is_dynamically_stable(CRITICAL_RICHARDSON));
    }

    #[test]
    fn kh_critical_velocity_positive() {
        let v_c = kh_critical_velocity(1.0, 1.2, 0.072, 1.0, 9.81);
        assert!(v_c > 0.0, "critical velocity must be positive, got {v_c}");
    }

    #[test]
    fn kh_critical_velocity_value() {
        let rho1 = 1.0;
        let rho2 = 1.2;
        let sigma = 0.072;
        let k = 2.0;
        let g = 9.81;
        let v_c = kh_critical_velocity(rho1, rho2, sigma, k, g);
        assert!(rel_error(v_c, 1.436_141_116_206_173) < 1e-6);
    }

    #[test]
    fn plateau_rayleigh_growth_rate_unstable_mode() {
        let gamma = plateau_rayleigh_growth_rate(0.072, 1000.0, 0.001, 500.0);
        assert!(gamma > 0.0, "Growth rate should be positive for unstable mode, got {gamma}");
    }
}
