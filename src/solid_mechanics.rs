//! Strength of materials: stress, strain, elastic constants and beams.
//!
//! Engineering and true stress and strain, and the elastic constants with
//! the identities that connect them -- any two of `E`, `G`, `K` and `ν`
//! determine the other two for an isotropic material, and the conversions
//! are all here.
//!
//! Beam bending: cantilever and simply-supported deflections, bending
//! moment and stress, and second moments of area for rectangular and
//! circular sections. Design closes with the von Mises equivalent stress,
//! the safety factor, and strain energy density.
//!
//! For the tensor formulation and yield surfaces see
//! [`crate::continuum_mechanics`]; for finite-element beams and modal
//! analysis see [`crate::resonance::structural`].

use crate::math::constants::PI;

// ---------------------------------------------------------------------------
// Stress & Strain
// ---------------------------------------------------------------------------

/// Tensile stress: σ = F/A
#[must_use]
pub fn tensile_stress(force: f64, area: f64) -> f64 {
    assert!(area > 0.0, "area must be positive");
    force / area
}

/// Tensile strain: ε = ΔL/L₀
#[must_use]
pub fn tensile_strain(delta_l: f64, original_l: f64) -> f64 {
    assert!(original_l > 0.0, "original_l must be positive");
    delta_l / original_l
}

/// Shear stress: τ = F/A
#[must_use]
pub fn shear_stress(force: f64, area: f64) -> f64 {
    assert!(area > 0.0, "area must be positive");
    force / area
}

/// Shear strain: γ = Δx/h
#[must_use]
pub fn shear_strain(displacement: f64, height: f64) -> f64 {
    assert!(height > 0.0, "height must be positive");
    displacement / height
}

/// Volumetric strain: εv = ΔV/V₀
#[must_use]
pub fn volumetric_strain(delta_v: f64, original_v: f64) -> f64 {
    assert!(original_v > 0.0, "original_v must be positive");
    delta_v / original_v
}

/// True stress from engineering values: σ_true = σ_eng(1 + ε_eng)
#[must_use]
pub fn true_stress(engineering_stress: f64, engineering_strain: f64) -> f64 {
    engineering_stress * (1.0 + engineering_strain)
}

/// True strain from engineering strain: ε_true = ln(1 + ε_eng)
#[must_use]
pub fn true_strain(engineering_strain: f64) -> f64 {
    (1.0 + engineering_strain).ln()
}

// ---------------------------------------------------------------------------
// Elastic Moduli
// ---------------------------------------------------------------------------

/// Young's modulus (elastic modulus): E = σ/ε
#[must_use]
pub fn youngs_modulus(stress: f64, strain: f64) -> f64 {
    assert!(strain != 0.0, "strain must not be zero");
    stress / strain
}

/// Shear modulus: G = τ/γ
#[must_use]
pub fn shear_modulus(shear_stress: f64, shear_strain: f64) -> f64 {
    assert!(shear_strain != 0.0, "shear_strain must not be zero");
    shear_stress / shear_strain
}

/// Bulk modulus: K = -P/εv
#[must_use]
pub fn bulk_modulus(pressure: f64, volumetric_strain: f64) -> f64 {
    assert!(volumetric_strain != 0.0, "volumetric_strain must not be zero");
    -pressure / volumetric_strain
}

/// Poisson's ratio from elastic moduli: ν = E/(2G) - 1
#[must_use]
pub fn poisson_ratio_from_moduli(e: f64, g: f64) -> f64 {
    assert!(g > 0.0, "shear modulus must be positive");
    e / (2.0 * g) - 1.0
}

/// Young's modulus from bulk and shear moduli: E = 9KG/(3K + G)
#[must_use]
pub fn e_from_k_and_g(bulk: f64, shear: f64) -> f64 {
    assert!(3.0 * bulk + shear != 0.0, "3K + G must not be zero");
    9.0 * bulk * shear / (3.0 * bulk + shear)
}

/// Bulk modulus from Young's modulus and Poisson's ratio: K = E/(3(1 - 2ν))
#[must_use]
pub fn bulk_from_e_and_nu(e: f64, nu: f64) -> f64 {
    assert!((1.0 - 2.0 * nu).abs() > 0.0, "1 - 2*nu must not be zero");
    e / (3.0 * (1.0 - 2.0 * nu))
}

/// Shear modulus from Young's modulus and Poisson's ratio: G = E/(2(1 + ν))
#[must_use]
pub fn shear_from_e_and_nu(e: f64, nu: f64) -> f64 {
    assert!((1.0 + nu).abs() > 0.0, "1 + nu must not be zero");
    e / (2.0 * (1.0 + nu))
}

// ---------------------------------------------------------------------------
// Beam Mechanics
// ---------------------------------------------------------------------------

/// Cantilever beam tip deflection under point load: δ = FL³/(3EI)
#[must_use]
pub fn beam_deflection_cantilever_point(force: f64, length: f64, e: f64, i: f64) -> f64 {
    assert!(e > 0.0, "elastic modulus must be positive");
    assert!(i > 0.0, "second moment of area must be positive");
    force * length.powi(3) / (3.0 * e * i)
}

/// Simply supported beam center deflection under point load: δ = FL³/(48EI)
#[must_use]
pub fn beam_deflection_simply_supported_center(force: f64, length: f64, e: f64, i: f64) -> f64 {
    assert!(e > 0.0, "elastic modulus must be positive");
    assert!(i > 0.0, "second moment of area must be positive");
    force * length.powi(3) / (48.0 * e * i)
}

/// Bending moment: M = F·d
#[must_use]
pub fn bending_moment(force: f64, distance: f64) -> f64 {
    force * distance
}

/// Bending stress at distance y from neutral axis: σ = My/I
#[must_use]
pub fn bending_stress(moment: f64, y: f64, i: f64) -> f64 {
    assert!(i > 0.0, "second moment of area must be positive");
    moment * y / i
}

/// Second moment of area for a rectangle: I = bh³/12
#[must_use]
pub fn second_moment_rectangle(width: f64, height: f64) -> f64 {
    width * height.powi(3) / 12.0
}

/// Second moment of area for a circle: I = πr⁴/4
#[must_use]
pub fn second_moment_circle(radius: f64) -> f64 {
    PI * radius.powi(4) / 4.0
}

// ---------------------------------------------------------------------------
// Failure Criteria
// ---------------------------------------------------------------------------

/// Von Mises equivalent stress: σ_vm = √(((σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²)/2)
#[must_use]
pub fn von_mises_stress(s1: f64, s2: f64, s3: f64) -> f64 {
    let term = (s1 - s2).powi(2) + (s2 - s3).powi(2) + (s3 - s1).powi(2);
    (term / 2.0).sqrt()
}

/// Safety factor: n = σ_yield/σ_applied
#[must_use]
pub fn safety_factor(yield_strength: f64, applied_stress: f64) -> f64 {
    assert!(applied_stress != 0.0, "applied_stress must not be zero");
    yield_strength / applied_stress
}

/// Elastic strain energy density: u = σε/2
#[must_use]
pub fn strain_energy_density(stress: f64, strain: f64) -> f64 {
    stress * strain / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_rel(a: f64, b: f64) -> bool {
        if b.abs() < TOLERANCE {
            return a.abs() < TOLERANCE;
        }
        ((a - b) / b).abs() < 1e-6
    }

    // -- Stress & Strain --

    #[test]
    fn test_tensile_stress() {
        assert!(approx(tensile_stress(1000.0, 0.01), 100_000.0));
    }

    #[test]
    fn test_tensile_strain() {
        assert!(approx(tensile_strain(0.002, 1.0), 0.002));
    }

    #[test]
    fn test_shear_stress() {
        assert!(approx(shear_stress(500.0, 0.05), 10_000.0));
    }

    #[test]
    fn test_shear_strain() {
        assert!(approx(shear_strain(0.01, 0.5), 0.02));
    }

    #[test]
    fn test_volumetric_strain() {
        assert!(approx(volumetric_strain(-0.001, 1.0), -0.001));
    }

    #[test]
    fn test_true_stress() {
        // 200e6 * (1 + 0.05) = 210e6
        assert!(approx_rel(true_stress(200e6, 0.05), 210e6));
    }

    #[test]
    fn test_true_strain() {
        // ln(1.05) ≈ 0.04879016
        assert!(approx_rel(true_strain(0.05), 0.04879016));
    }

    // -- Elastic Moduli --

    #[test]
    fn test_youngs_modulus() {
        assert!(approx(youngs_modulus(200e6, 0.001), 200e9));
    }

    #[test]
    fn test_shear_modulus() {
        assert!(approx(shear_modulus(80e6, 0.001), 80e9));
    }

    #[test]
    fn test_bulk_modulus() {
        assert!(approx(bulk_modulus(-100e6, -0.001), -100e9));
    }

    #[test]
    fn test_poisson_ratio_from_moduli() {
        // Steel: E=200 GPa, G=77 GPa => ν = 200/(2*77) - 1 = 1.2987 - 1 = 0.2987
        let nu = poisson_ratio_from_moduli(200e9, 77e9);
        assert!(approx_rel(nu, 0.2987013));
    }

    #[test]
    fn test_e_from_k_and_g() {
        // 9*160e9*80e9 / (3*160e9 + 80e9) = 115200e18 / 560e9 ≈ 205.714e9
        let e = e_from_k_and_g(160e9, 80e9);
        assert!(approx_rel(e, 205.7142857e9));
    }

    #[test]
    fn test_bulk_from_e_and_nu() {
        // 200e9 / (3*(1-0.6)) = 200e9 / 1.2 ≈ 166.667e9
        assert!(approx_rel(bulk_from_e_and_nu(200e9, 0.3), 166.6667e9));
    }

    #[test]
    fn test_shear_from_e_and_nu() {
        // 200e9 / (2*1.3) = 200e9 / 2.6 ≈ 76.923e9
        assert!(approx_rel(shear_from_e_and_nu(200e9, 0.3), 76.92308e9));
    }

    #[test]
    fn test_moduli_roundtrip() {
        // K=166.667e9, G=76.923e9 → E = 9KG/(3K+G) ≈ 200e9
        let k = bulk_from_e_and_nu(200e9, 0.3);
        let g = shear_from_e_and_nu(200e9, 0.3);
        let e_recovered = e_from_k_and_g(k, g);
        assert!(approx_rel(e_recovered, 200e9));
    }

    // -- Beam Mechanics --

    #[test]
    fn test_cantilever_deflection() {
        // FL³/(3EI) = 1000*8 / (3*200e9*8.33e-6) = 8000/4998 ≈ 0.001601 m
        assert!(approx_rel(
            beam_deflection_cantilever_point(1000.0, 2.0, 200e9, 8.33e-6),
            0.0016006403,
        ));
    }

    #[test]
    fn test_simply_supported_deflection() {
        // FL³/(48EI) = 1000*64 / (48*200e9*8.33e-6) = 64000/79968 ≈ 0.0008003 m
        assert!(approx_rel(
            beam_deflection_simply_supported_center(1000.0, 4.0, 200e9, 8.33e-6),
            0.00080032013,
        ));
    }

    #[test]
    fn test_bending_moment() {
        assert!(approx(bending_moment(500.0, 3.0), 1500.0));
    }

    #[test]
    fn test_bending_stress() {
        // M*y/I = 1500*0.05 / 8.33e-6 = 75 / 8.33e-6 ≈ 9_003_601.44 Pa
        assert!(approx_rel(bending_stress(1500.0, 0.05, 8.33e-6), 9_003_601.44));
    }

    #[test]
    fn test_second_moment_rectangle() {
        // bh³/12 = 0.1 * 0.008 / 12 = 6.6667e-5
        assert!(approx_rel(second_moment_rectangle(0.1, 0.2), 6.666667e-5));
    }

    #[test]
    fn test_second_moment_circle() {
        // πr⁴/4 = π * 6.25e-6 / 4 ≈ 4.90874e-6
        assert!(approx_rel(second_moment_circle(0.05), 4.90874e-6));
    }

    // -- Failure Criteria --

    #[test]
    fn test_von_mises_uniaxial() {
        // Uniaxial tension: s1 = sigma, s2 = s3 = 0 => von Mises = sigma
        let sigma = 250e6;
        assert!(approx_rel(von_mises_stress(sigma, 0.0, 0.0), sigma));
    }

    #[test]
    fn test_von_mises_hydrostatic() {
        // Hydrostatic: s1 = s2 = s3 => von Mises = 0
        let p = 100e6;
        assert!(approx(von_mises_stress(p, p, p), 0.0));
    }

    #[test]
    fn test_von_mises_pure_shear() {
        // Pure shear: s1=τ, s2=0, s3=-τ => von Mises = τ√3 = 100e6 * 1.73205 ≈ 173.205e6
        assert!(approx_rel(von_mises_stress(100e6, 0.0, -100e6), 173.205e6));
    }

    #[test]
    fn test_safety_factor() {
        assert!(approx(safety_factor(250e6, 125e6), 2.0));
    }

    #[test]
    fn test_strain_energy_density() {
        assert!(approx(strain_energy_density(200e6, 0.001), 100e3));
    }

    #[test]
    fn test_approx_rel_near_zero_b() {
        assert!(approx_rel(0.0, 0.0));
        assert!(!approx_rel(1.0, 0.0));
    }
}
