use crate::math::constants::{C, E_CHARGE, HBAR, K_E};

// ---------------------------------------------------------------------------
// Particle masses (kg)
// ---------------------------------------------------------------------------
pub const M_MUON: f64 = 1.884e-28;
pub const M_TAU: f64 = 3.168e-27;
pub const M_PION_CHARGED: f64 = 2.488e-28;
pub const M_PION_NEUTRAL: f64 = 2.406e-28;
pub const M_KAON: f64 = 8.800e-28;
pub const M_W_BOSON: f64 = 1.433e-25; // 80.4 GeV/c²
pub const M_Z_BOSON: f64 = 1.626e-25; // 91.2 GeV/c²
pub const M_HIGGS: f64 = 2.230e-25; // 125.1 GeV/c²
pub const M_TOP_QUARK: f64 = 3.078e-25; // 173 GeV/c²

// ---------------------------------------------------------------------------
// Fundamental charges in units of e
// ---------------------------------------------------------------------------
pub const CHARGE_UP: f64 = 2.0 / 3.0;
pub const CHARGE_DOWN: f64 = -1.0 / 3.0;

// ---------------------------------------------------------------------------
// Coupling constants
// ---------------------------------------------------------------------------
pub const FINE_STRUCTURE: f64 = 7.297e-3; // α ≈ 1/137
pub const WEAK_MIXING_ANGLE_SIN2: f64 = 0.2312; // sin²θ_W
pub const STRONG_COUPLING: f64 = 0.1179; // α_s at M_Z

// ---------------------------------------------------------------------------
// Relativistic kinematics
// ---------------------------------------------------------------------------

const C2: f64 = C * C;

/// Invariant mass from total energy and scalar momentum magnitude.
/// m = √(E² - p²c²) / c²
pub fn invariant_mass(energy: f64, momentum: f64) -> f64 {
    let m2c4 = energy * energy - momentum * momentum * C2;
    (m2c4.max(0.0)).sqrt() / C2
}

/// Invariant mass of a two-body system from individual four-momenta.
/// m² = (E1+E2)² - |p1+p2|²c²,  return m/c².
pub fn invariant_mass_two_body(
    e1: f64, px1: f64, py1: f64, pz1: f64,
    e2: f64, px2: f64, py2: f64, pz2: f64,
) -> f64 {
    let e_tot = e1 + e2;
    let px = px1 + px2;
    let py = py1 + py2;
    let pz = pz1 + pz2;
    let m2c4 = e_tot * e_tot - (px * px + py * py + pz * pz) * C2;
    (m2c4.max(0.0)).sqrt() / C2
}

/// Center-of-mass energy √s for two particles with given energies and momenta.
/// √s = √((E1+E2)² - (p1+p2)²c²)
pub fn center_of_mass_energy(
    e_beam: f64, e_target: f64,
    p_beam: f64, p_target: f64,
) -> f64 {
    let e_tot = e_beam + e_target;
    let p_tot = p_beam + p_target;
    let s = e_tot * e_tot - p_tot * p_tot * C2;
    s.max(0.0).sqrt()
}

/// Fixed-target center-of-mass energy (high-energy approximation).
/// √s ≈ √(2 × E_beam × m_target × c²)
pub fn fixed_target_com_energy(beam_energy: f64, target_mass: f64) -> f64 {
    (2.0 * beam_energy * target_mass * C2).sqrt()
}

/// Lorentz boost of energy along z.  E' = γ(E - β pz c)
pub fn lorentz_boost_energy(energy: f64, momentum_z: f64, beta: f64) -> f64 {
    assert!(beta.abs() < 1.0, "beta must be less than 1");
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    gamma * (energy - beta * momentum_z * C)
}

/// Lorentz boost of z-momentum.  pz' = γ(pz - β E/c)
pub fn lorentz_boost_pz(energy: f64, momentum_z: f64, beta: f64) -> f64 {
    assert!(beta.abs() < 1.0, "beta must be less than 1");
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    gamma * (momentum_z - beta * energy / C)
}

/// Rapidity y = 0.5 × ln((E + pz c) / (E - pz c))
pub fn rapidity(energy: f64, pz: f64) -> f64 {
    assert!(energy > (pz * C).abs(), "energy must exceed |pz*c| for valid rapidity");
    0.5 * ((energy + pz * C) / (energy - pz * C)).ln()
}

/// Pseudorapidity η = -ln(tan(θ/2))
pub fn pseudorapidity(theta: f64) -> f64 {
    let tan_half = (theta / 2.0).tan();
    assert!(tan_half > 0.0, "theta must be in (0, pi)");
    -tan_half.ln()
}

/// Transverse momentum pT = √(px² + py²)
pub fn transverse_momentum(px: f64, py: f64) -> f64 {
    (px * px + py * py).sqrt()
}

// ---------------------------------------------------------------------------
// Cross sections & decay
// ---------------------------------------------------------------------------

/// Rutherford scattering differential cross section.
/// dσ/dΩ = (Z1 Z2 k_e e² / (4E))² / sin⁴(θ/2)
pub fn rutherford_cross_section(z1: f64, z2: f64, energy: f64, angle: f64) -> f64 {
    assert!(energy > 0.0, "energy must be positive");
    let sin_half = (angle / 2.0).sin();
    assert!(sin_half.abs() > 0.0, "angle must not be a multiple of 2*pi");
    let numerator = z1 * z2 * K_E * E_CHARGE * E_CHARGE;
    let term = numerator / (4.0 * energy);
    let sin4 = sin_half.powi(4);
    (term * term) / sin4
}

/// Non-relativistic Breit-Wigner resonance (normalized to peak = 1).
/// BW(E) = (Γ/2)² / ((E - M)² + (Γ/2)²)
pub fn breit_wigner(energy: f64, mass: f64, width: f64) -> f64 {
    let half_width = width / 2.0;
    let de = energy - mass;
    (half_width * half_width) / (de * de + half_width * half_width)
}

/// Decay rate from lifetime.  Γ = ℏ / τ
pub fn decay_rate_from_lifetime(lifetime: f64) -> f64 {
    assert!(lifetime > 0.0, "lifetime must be positive");
    HBAR / lifetime
}

/// Lifetime from decay width.  τ = ℏ / Γ
pub fn lifetime_from_width(width_joules: f64) -> f64 {
    assert!(width_joules > 0.0, "width_joules must be positive");
    HBAR / width_joules
}

/// Branching ratio BR = Γ_i / Γ_total
pub fn branching_ratio(partial_width: f64, total_width: f64) -> f64 {
    assert!(total_width > 0.0, "total_width must be positive");
    partial_width / total_width
}

/// Mean free path λ = 1 / (n σ)
pub fn mean_free_path_particle(cross_section: f64, number_density: f64) -> f64 {
    assert!(cross_section > 0.0, "cross_section must be positive");
    assert!(number_density > 0.0, "number_density must be positive");
    1.0 / (number_density * cross_section)
}

/// Event rate R = L × σ
pub fn luminosity_to_event_rate(luminosity: f64, cross_section: f64) -> f64 {
    luminosity * cross_section
}

// ---------------------------------------------------------------------------
// Conservation laws
// ---------------------------------------------------------------------------

const CHARGE_TOLERANCE: f64 = 1e-9;

/// Check charge conservation: sum of input charges ≈ sum of output charges.
pub fn is_charge_conserved(charges_in: &[f64], charges_out: &[f64]) -> bool {
    let sum_in: f64 = charges_in.iter().sum();
    let sum_out: f64 = charges_out.iter().sum();
    (sum_in - sum_out).abs() < CHARGE_TOLERANCE
}

/// Check lepton number conservation.
pub fn is_lepton_number_conserved(leptons_in: &[i32], leptons_out: &[i32]) -> bool {
    let sum_in: i32 = leptons_in.iter().sum();
    let sum_out: i32 = leptons_out.iter().sum();
    sum_in == sum_out
}

/// Check baryon number conservation.
pub fn is_baryon_number_conserved(baryons_in: &[i32], baryons_out: &[i32]) -> bool {
    let sum_in: i32 = baryons_in.iter().sum();
    let sum_out: i32 = baryons_out.iter().sum();
    sum_in == sum_out
}

/// Four-momentum magnitude (invariant mass × c).
/// √(E² - p²c²) / c  =  mc
pub fn four_momentum_magnitude(energy: f64, px: f64, py: f64, pz: f64) -> f64 {
    let p2 = px * px + py * py + pz * pz;
    let val = energy * energy - p2 * C2;
    val.max(0.0).sqrt() / C
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::{M_ELECTRON, M_PROTON, PI};

    fn approx(a: f64, b: f64, rel_tol: f64) -> bool {
        if a == 0.0 && b == 0.0 {
            return true;
        }
        let denom = a.abs().max(b.abs());
        (a - b).abs() / denom < rel_tol
    }

    #[test]
    fn test_constants_sanity() {
        assert!(M_MUON > M_ELECTRON);
        assert!(M_TAU > M_MUON);
        assert!(M_HIGGS > M_Z_BOSON);
        assert!(M_Z_BOSON > M_W_BOSON);
        assert!(FINE_STRUCTURE > 0.0 && FINE_STRUCTURE < 0.01);
        assert!(CHARGE_UP > 0.0);
        assert!(CHARGE_DOWN < 0.0);
        assert!(approx(CHARGE_UP, 2.0 / 3.0, 1e-12));
        assert!(approx(CHARGE_DOWN, -1.0 / 3.0, 1e-12));
    }

    #[test]
    fn test_invariant_mass_at_rest() {
        // Particle at rest: E = mc², p = 0  =>  m_out = m
        let m = M_PROTON;
        let energy = m * C2;
        let result = invariant_mass(energy, 0.0);
        assert!(approx(result, m, 1e-6));
    }

    #[test]
    fn test_invariant_mass_two_body_at_rest() {
        // Two equal-mass particles at rest
        let m = M_ELECTRON;
        let e = m * C2;
        let result = invariant_mass_two_body(e, 0.0, 0.0, 0.0, e, 0.0, 0.0, 0.0);
        assert!(approx(result, 2.0 * m, 1e-6));
    }

    #[test]
    fn test_center_of_mass_energy_equal_beams() {
        // Two equal beams head-on: E each, p equal and opposite => √s = 2E
        let e = 1.0e-9; // 1 nJ
        let p = e / C; // ultra-relativistic
        let sqrt_s = center_of_mass_energy(e, e, p, -p);
        assert!(approx(sqrt_s, 2.0 * e, 1e-4));
    }

    #[test]
    fn test_fixed_target_com_energy() {
        let e_beam = 1.0e-6;
        let m_target = M_PROTON;
        let sqrt_s = fixed_target_com_energy(e_beam, m_target);
        let expected = 1.73394210744484e-8;
        assert!(approx(sqrt_s, expected, 1e-10));
    }

    #[test]
    fn test_lorentz_boost_at_rest() {
        // Zero boost should return original values
        let e = 1.0e-9;
        let pz = 3.0e-18;
        let e_prime = lorentz_boost_energy(e, pz, 0.0);
        let pz_prime = lorentz_boost_pz(e, pz, 0.0);
        assert!(approx(e_prime, e, 1e-10));
        assert!(approx(pz_prime, pz, 1e-10));
    }

    #[test]
    fn test_rapidity_zero_pz() {
        // pz = 0 => rapidity = 0
        let y = rapidity(1.0e-9, 0.0);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn test_pseudorapidity_ninety_degrees() {
        // θ = π/2 => η = -ln(tan(π/4)) = -ln(1) = 0
        let eta = pseudorapidity(PI / 2.0);
        assert!(eta.abs() < 1e-10);
    }

    #[test]
    fn test_transverse_momentum() {
        let pt = transverse_momentum(3.0, 4.0);
        assert!(approx(pt, 5.0, 1e-10));
    }

    #[test]
    fn test_rutherford_cross_section_positive() {
        let ds = rutherford_cross_section(1.0, 1.0, 1.0e-13, PI / 4.0);
        assert!(ds > 0.0);
        assert!(ds.is_finite());
    }

    #[test]
    fn test_rutherford_angle_dependence() {
        // Smaller angle => larger cross section (1/sin⁴(θ/2))
        let ds_small = rutherford_cross_section(1.0, 1.0, 1.0e-13, 0.3);
        let ds_large = rutherford_cross_section(1.0, 1.0, 1.0e-13, 1.0);
        assert!(ds_small > ds_large);
    }

    #[test]
    fn test_breit_wigner_peak() {
        // At resonance E = M, BW should be 1.0
        let bw = breit_wigner(91.2, 91.2, 2.5);
        assert!(approx(bw, 1.0, 1e-10));
    }

    #[test]
    fn test_breit_wigner_off_peak() {
        // Far off-resonance, BW should be small
        let bw = breit_wigner(50.0, 91.2, 2.5);
        assert!(bw < 0.01);
    }

    #[test]
    fn test_decay_rate_lifetime_roundtrip() {
        let tau = 2.2e-6; // muon lifetime
        let gamma = decay_rate_from_lifetime(tau);
        let tau_back = lifetime_from_width(gamma);
        assert!(approx(tau_back, tau, 1e-6));
    }

    #[test]
    fn test_branching_ratio() {
        let br = branching_ratio(0.3, 1.0);
        assert!(approx(br, 0.3, 1e-10));
    }

    #[test]
    fn test_mean_free_path() {
        let sigma = 1.0e-28; // ~1 barn
        let n = 1.0e28;
        let mfp = mean_free_path_particle(sigma, n);
        assert!(approx(mfp, 1.0, 1e-10));
    }

    #[test]
    fn test_luminosity_to_event_rate() {
        let lumi = 1.0e34;
        let sigma = 1.0e-28;
        let rate = luminosity_to_event_rate(lumi, sigma);
        assert!(approx(rate, 1.0e6, 1e-6));
    }

    #[test]
    fn test_charge_conserved() {
        assert!(is_charge_conserved(&[1.0, -1.0], &[1.0, -1.0]));
        assert!(!is_charge_conserved(&[1.0], &[-1.0]));
    }

    #[test]
    fn test_lepton_number_conserved() {
        assert!(is_lepton_number_conserved(&[1, -1], &[1, -1]));
        assert!(!is_lepton_number_conserved(&[1], &[1, 1]));
    }

    #[test]
    fn test_baryon_number_conserved() {
        assert!(is_baryon_number_conserved(&[1, 1], &[1, 1]));
        assert!(!is_baryon_number_conserved(&[1], &[0]));
    }

    #[test]
    fn test_four_momentum_magnitude_at_rest() {
        // Particle at rest: E = mc², p = 0 => |P| = mc
        let m = M_PROTON;
        let e = m * C2;
        let mag = four_momentum_magnitude(e, 0.0, 0.0, 0.0);
        assert!(approx(mag, m * C, 1e-6));
    }

    #[test]
    fn test_approx_both_zero() {
        assert!(approx(0.0, 0.0, 1e-6));
    }
}
