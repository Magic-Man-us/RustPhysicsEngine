use crate::math::constants::{K_B, MU_0, PI};

// ── Frozen-in flux threshold ────────────────────────────────────────────
const FROZEN_IN_THRESHOLD: f64 = 100.0;

// ── MHD Parameters ──────────────────────────────────────────────────────

/// Magnetic Reynolds number: Rm = μ₀ σ v L.
/// Quantifies the ratio of magnetic advection to diffusion.
#[must_use]
pub fn magnetic_reynolds_number(
    velocity: f64,
    length: f64,
    conductivity: f64,
) -> f64 {
    MU_0 * conductivity * velocity * length
}

/// Magnetic diffusivity: η = 1/(μ₀ σ).
#[must_use]
pub fn magnetic_diffusivity(conductivity: f64) -> f64 {
    assert!(conductivity > 0.0, "conductivity must be positive");
    1.0 / (MU_0 * conductivity)
}

/// Lundquist number: S = vₐ L / η.
/// Ratio of resistive diffusion time to Alfvén transit time.
#[must_use]
pub fn lundquist_number(alfven_speed: f64, length: f64, diffusivity: f64) -> f64 {
    assert!(diffusivity > 0.0, "diffusivity must be positive");
    alfven_speed * length / diffusivity
}

/// Hartmann number: Ha = B L √(σ / μ_visc).
/// Ratio of electromagnetic to viscous forces in a conducting fluid.
#[must_use]
pub fn hartmann_number(
    b_field: f64,
    length: f64,
    conductivity: f64,
    dynamic_viscosity: f64,
) -> f64 {
    assert!(dynamic_viscosity > 0.0, "dynamic_viscosity must be positive");
    b_field * length * (conductivity / dynamic_viscosity).sqrt()
}

/// Magnetic pressure: P_B = B² / (2 μ₀).
#[must_use]
pub fn magnetic_pressure(b_field: f64) -> f64 {
    b_field * b_field / (2.0 * MU_0)
}

/// Total pressure (gas + magnetic): P_total = P + B² / (2 μ₀).
#[must_use]
pub fn total_pressure(gas_pressure: f64, b_field: f64) -> f64 {
    gas_pressure + magnetic_pressure(b_field)
}

/// Plasma beta: β = 2 μ₀ P / B².
/// Ratio of gas pressure to magnetic pressure.
#[must_use]
pub fn plasma_beta(gas_pressure: f64, b_field: f64) -> f64 {
    assert!(b_field != 0.0, "magnetic field must be non-zero");
    2.0 * MU_0 * gas_pressure / (b_field * b_field)
}

// ── MHD Wave Speeds ─────────────────────────────────────────────────────

/// Alfvén speed: vₐ = B / √(μ₀ ρ).
#[must_use]
pub fn alfven_speed(b_field: f64, density: f64) -> f64 {
    assert!(density > 0.0, "density must be positive");
    b_field / (MU_0 * density).sqrt()
}

/// Slow magnetosonic speed (perpendicular propagation): v_slow = min(vₐ, cₛ).
#[must_use]
pub fn slow_magnetosonic_speed(alfven: f64, sound: f64) -> f64 {
    alfven.min(sound)
}

/// Fast magnetosonic speed (perpendicular propagation): v_fast = √(vₐ² + cₛ²).
#[must_use]
pub fn fast_magnetosonic_speed(alfven: f64, sound: f64) -> f64 {
    (alfven * alfven + sound * sound).sqrt()
}

/// Magnetosonic Mach number: M_ms = v / v_fast.
#[must_use]
pub fn magnetosonic_mach(velocity: f64, alfven: f64, sound: f64) -> f64 {
    let v_fast = fast_magnetosonic_speed(alfven, sound);
    velocity / v_fast
}

// ── MHD Equilibrium ─────────────────────────────────────────────────────

/// Z-pinch pressure balance.
/// Computes the magnetic pressure from the azimuthal field B_θ = μ₀ I / (2π r).
#[must_use]
pub fn pinch_pressure_balance(current: f64, radius: f64) -> f64 {
    assert!(radius > 0.0, "radius must be positive");
    let b_theta = MU_0 * current / (2.0 * PI * radius);
    magnetic_pressure(b_theta)
}

/// Bennett pinch condition: checks whether I² ≈ 8π N k_B T / μ₀.
/// Returns `true` when the plasma is in pressure balance.
#[must_use]
pub fn bennett_pinch_condition(
    current: f64,
    line_density: f64,
    temperature: f64,
) -> bool {
    let rhs = 8.0 * PI * line_density * K_B * temperature / MU_0;
    let lhs = current * current;
    let ratio = lhs / rhs;
    (0.9..=1.1).contains(&ratio)
}

/// Rough Troyon-like beta limit: β_max ≈ 1 / aspect_ratio.
#[must_use]
pub fn grad_shafranov_beta_limit(aspect_ratio: f64) -> f64 {
    assert!(aspect_ratio > 0.0, "aspect_ratio must be positive");
    1.0 / aspect_ratio
}

// ── Magnetic Reconnection ───────────────────────────────────────────────

/// Sweet-Parker reconnection rate: v_in / vₐ = 1 / √S.
#[must_use]
pub fn sweet_parker_rate(alfven_speed: f64, lundquist: f64) -> f64 {
    assert!(lundquist > 0.0, "Lundquist number must be positive");
    alfven_speed / lundquist.sqrt()
}

/// Reconnection electric field: E = v_in × B (magnitude).
#[must_use]
pub fn reconnection_electric_field(b_field: f64, inflow_velocity: f64) -> f64 {
    inflow_velocity * b_field
}

// ── Induction Equation Helpers ──────────────────────────────────────────

/// Magnetic diffusion time: τ_d = L² / η.
#[must_use]
pub fn magnetic_diffusion_time(length: f64, diffusivity: f64) -> f64 {
    assert!(diffusivity > 0.0, "diffusivity must be positive");
    length * length / diffusivity
}

/// Advection time: τ_a = L / v.
#[must_use]
pub fn advection_time(length: f64, velocity: f64) -> f64 {
    assert!(velocity > 0.0, "velocity must be positive");
    length / velocity
}

/// Returns `true` when Rm > 100, indicating the magnetic field is frozen into the plasma.
#[must_use]
pub fn is_frozen_in(reynolds_mag: f64) -> bool {
    reynolds_mag > FROZEN_IN_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn test_alfven_speed_solar_corona() {
        // Typical solar corona: B ~ 1e-4 T (1 G), ρ ~ 1e-12 kg/m³
        let va = alfven_speed(1e-4, 1e-12);
        // Expected ~ B / sqrt(mu0 * rho) ≈ 1e-4 / sqrt(1.257e-6 * 1e-12)
        //   = 1e-4 / sqrt(1.257e-18) ≈ 1e-4 / 3.545e-9 ≈ 2.82e4 m/s
        assert!(va > 1e4, "Alfvén speed in solar corona should be > 10 km/s, got {va}");
        assert!(va < 1e6, "Alfvén speed in solar corona should be < 1000 km/s, got {va}");

        // 1e-4 / √(μ₀ × 1e-12) ≈ 89206.2 m/s
        assert!(rel_err(va, 89_206.2) < 1e-3);
    }

    #[test]
    fn test_fast_magnetosonic_exceeds_components() {
        let va = 500.0;
        let cs = 300.0;
        let v_fast = fast_magnetosonic_speed(va, cs);
        let v_slow = slow_magnetosonic_speed(va, cs);

        assert!(v_fast > va, "Fast magnetosonic must exceed Alfvén speed");
        assert!(v_fast > cs, "Fast magnetosonic must exceed sound speed");
        assert!(v_slow <= va && v_slow <= cs, "Slow magnetosonic must not exceed either component");

        assert!(rel_err(v_fast, 583.095_189_484_530_05) < 1e-10);
    }

    #[test]
    fn test_magnetic_pressure_positive() {
        for &b in &[1e-6, 1e-3, 1.0, 10.0, 100.0] {
            let p = magnetic_pressure(b);
            assert!(p > 0.0, "Magnetic pressure must be positive for B = {b}");
        }

        let p = magnetic_pressure(1.0);
        assert!(rel_err(p, 397_887.36) < 1e-3);
    }

    #[test]
    fn test_sweet_parker_rate_decreases_with_lundquist() {
        let va = 1e5;
        let s1 = 1e4;
        let s2 = 1e8;

        let rate1 = sweet_parker_rate(va, s1);
        let rate2 = sweet_parker_rate(va, s2);
        assert!(
            rate2 < rate1,
            "Sweet-Parker rate must decrease with increasing Lundquist number"
        );

        assert!(rel_err(rate1, 1000.0) < 1e-12);
    }

    #[test]
    fn test_magnetic_reynolds_scales_with_conductivity() {
        let v = 1e3;
        let l = 1.0;
        let sigma1 = 1e3;
        let sigma2 = 1e6;

        let rm1 = magnetic_reynolds_number(v, l, sigma1);
        let rm2 = magnetic_reynolds_number(v, l, sigma2);

        assert!(rm2 > rm1, "Rm must increase with conductivity");
        assert!(
            rel_err(rm2 / rm1, sigma2 / sigma1) < 1e-10,
            "Rm must scale linearly with conductivity"
        );
    }

    #[test]
    fn test_plasma_beta() {
        let p_gas = 1e5;
        let b = 0.01;
        let beta = plasma_beta(p_gas, b);
        assert!(rel_err(beta, 2513.274_124_24) < 1e-6);
    }

    #[test]
    fn test_total_pressure() {
        let p_gas = 1e5;
        let b = 0.1;
        let p_total = total_pressure(p_gas, b);
        assert!(p_total > p_gas, "Total pressure must exceed gas pressure");
        assert!(rel_err(p_total, 103_978.873_6) < 1e-3);
    }

    #[test]
    fn test_magnetic_diffusivity() {
        let sigma = 1e6;
        let eta = magnetic_diffusivity(sigma);
        assert!(rel_err(eta, 0.795_774_715_459_477) < 1e-6);
    }

    #[test]
    fn test_hartmann_number() {
        let ha = hartmann_number(1.0, 0.1, 1e6, 1e-3);
        assert!(rel_err(ha, 3162.277_660_168_38) < 1e-6);
    }

    #[test]
    fn test_lundquist_number() {
        let s = lundquist_number(1e5, 1.0, 0.1);
        assert!(rel_err(s, 1e6) < 1e-10);
    }

    #[test]
    fn test_frozen_in_condition() {
        assert!(is_frozen_in(1e6), "Rm = 1e6 should be frozen-in");
        assert!(is_frozen_in(101.0), "Rm = 101 should be frozen-in");
        assert!(!is_frozen_in(50.0), "Rm = 50 should NOT be frozen-in");
        assert!(!is_frozen_in(100.0), "Rm = 100 (boundary) should NOT be frozen-in");
    }

    #[test]
    fn test_diffusion_and_advection_times() {
        let l = 1e6;
        let eta = 1.0;
        let v = 1e3;

        let tau_d = magnetic_diffusion_time(l, eta);
        let tau_a = advection_time(l, v);

        assert!(rel_err(tau_d, 1e12) < 1e-12);
        assert!(rel_err(tau_a, 1e3) < 1e-12);
    }

    #[test]
    fn test_bennett_pinch_condition() {
        let n_line = 1e20;
        let t = 1e7;
        let i_squared = 8.0 * PI * n_line * K_B * t / MU_0;
        let current = i_squared.sqrt();

        assert!(
            bennett_pinch_condition(current, n_line, t),
            "Exact Bennett current should satisfy condition"
        );
        assert!(
            !bennett_pinch_condition(current * 2.0, n_line, t),
            "Double current should violate condition"
        );
    }

    #[test]
    fn test_pinch_pressure_balance() {
        let current = 1e6;
        let radius = 0.1;
        let p = pinch_pressure_balance(current, radius);
        assert!(rel_err(p, 1_591_549.430_918_95) < 1e-3);
    }

    #[test]
    fn test_reconnection_electric_field() {
        let e = reconnection_electric_field(1e-4, 1e3);
        assert!(rel_err(e, 0.1) < 1e-12);
    }

    #[test]
    fn test_magnetosonic_mach() {
        let v = 1000.0;
        let va = 500.0;
        let cs = 300.0;
        let m = magnetosonic_mach(v, va, cs);
        assert!(rel_err(m, 1.714_985_850_992_862_5) < 1e-6);
    }

    #[test]
    fn test_grad_shafranov_beta_limit() {
        let a_r = 3.0;
        let beta_max = grad_shafranov_beta_limit(a_r);
        assert!(rel_err(beta_max, 1.0 / 3.0) < 1e-12);
    }
}
