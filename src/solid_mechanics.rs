use crate::math::constants::PI;

// ---------------------------------------------------------------------------
// Stress & Strain
// ---------------------------------------------------------------------------

#[must_use]
pub fn tensile_stress(force: f64, area: f64) -> f64 {
    force / area
}

#[must_use]
pub fn tensile_strain(delta_l: f64, original_l: f64) -> f64 {
    delta_l / original_l
}

#[must_use]
pub fn shear_stress(force: f64, area: f64) -> f64 {
    force / area
}

#[must_use]
pub fn shear_strain(displacement: f64, height: f64) -> f64 {
    displacement / height
}

#[must_use]
pub fn volumetric_strain(delta_v: f64, original_v: f64) -> f64 {
    delta_v / original_v
}

#[must_use]
pub fn true_stress(engineering_stress: f64, engineering_strain: f64) -> f64 {
    engineering_stress * (1.0 + engineering_strain)
}

#[must_use]
pub fn true_strain(engineering_strain: f64) -> f64 {
    (1.0 + engineering_strain).ln()
}

// ---------------------------------------------------------------------------
// Elastic Moduli
// ---------------------------------------------------------------------------

#[must_use]
pub fn youngs_modulus(stress: f64, strain: f64) -> f64 {
    stress / strain
}

#[must_use]
pub fn shear_modulus(shear_stress: f64, shear_strain: f64) -> f64 {
    shear_stress / shear_strain
}

#[must_use]
pub fn bulk_modulus(pressure: f64, volumetric_strain: f64) -> f64 {
    -pressure / volumetric_strain
}

#[must_use]
pub fn poisson_ratio_from_moduli(e: f64, g: f64) -> f64 {
    e / (2.0 * g) - 1.0
}

#[must_use]
pub fn e_from_k_and_g(bulk: f64, shear: f64) -> f64 {
    9.0 * bulk * shear / (3.0 * bulk + shear)
}

#[must_use]
pub fn bulk_from_e_and_nu(e: f64, nu: f64) -> f64 {
    e / (3.0 * (1.0 - 2.0 * nu))
}

#[must_use]
pub fn shear_from_e_and_nu(e: f64, nu: f64) -> f64 {
    e / (2.0 * (1.0 + nu))
}

// ---------------------------------------------------------------------------
// Beam Mechanics
// ---------------------------------------------------------------------------

#[must_use]
pub fn beam_deflection_cantilever_point(force: f64, length: f64, e: f64, i: f64) -> f64 {
    force * length.powi(3) / (3.0 * e * i)
}

#[must_use]
pub fn beam_deflection_simply_supported_center(force: f64, length: f64, e: f64, i: f64) -> f64 {
    force * length.powi(3) / (48.0 * e * i)
}

#[must_use]
pub fn bending_moment(force: f64, distance: f64) -> f64 {
    force * distance
}

#[must_use]
pub fn bending_stress(moment: f64, y: f64, i: f64) -> f64 {
    moment * y / i
}

#[must_use]
pub fn second_moment_rectangle(width: f64, height: f64) -> f64 {
    width * height.powi(3) / 12.0
}

#[must_use]
pub fn second_moment_circle(radius: f64) -> f64 {
    PI * radius.powi(4) / 4.0
}

// ---------------------------------------------------------------------------
// Failure Criteria
// ---------------------------------------------------------------------------

#[must_use]
pub fn von_mises_stress(s1: f64, s2: f64, s3: f64) -> f64 {
    let term = (s1 - s2).powi(2) + (s2 - s3).powi(2) + (s3 - s1).powi(2);
    (term / 2.0).sqrt()
}

#[must_use]
pub fn safety_factor(yield_strength: f64, applied_stress: f64) -> f64 {
    yield_strength / applied_stress
}

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
        let eng_stress = 200e6;
        let eng_strain = 0.05;
        let expected = eng_stress * 1.05;
        assert!(approx_rel(true_stress(eng_stress, eng_strain), expected));
    }

    #[test]
    fn test_true_strain() {
        let eng_strain = 0.05;
        let expected = (1.05_f64).ln();
        assert!(approx_rel(true_strain(eng_strain), expected));
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
        // Steel: E ~ 200 GPa, G ~ 77 GPa => nu ~ 0.2987
        let nu = poisson_ratio_from_moduli(200e9, 77e9);
        assert!(approx_rel(nu, 200e9 / (2.0 * 77e9) - 1.0));
    }

    #[test]
    fn test_e_from_k_and_g() {
        let k = 160e9;
        let g = 80e9;
        let e = e_from_k_and_g(k, g);
        let expected = 9.0 * k * g / (3.0 * k + g);
        assert!(approx_rel(e, expected));
    }

    #[test]
    fn test_bulk_from_e_and_nu() {
        let e = 200e9;
        let nu = 0.3;
        let expected = e / (3.0 * (1.0 - 2.0 * nu));
        assert!(approx_rel(bulk_from_e_and_nu(e, nu), expected));
    }

    #[test]
    fn test_shear_from_e_and_nu() {
        let e = 200e9;
        let nu = 0.3;
        let expected = e / (2.0 * (1.0 + nu));
        assert!(approx_rel(shear_from_e_and_nu(e, nu), expected));
    }

    #[test]
    fn test_moduli_roundtrip() {
        let e = 200e9;
        let nu = 0.3;
        let k = bulk_from_e_and_nu(e, nu);
        let g = shear_from_e_and_nu(e, nu);
        let e_recovered = e_from_k_and_g(k, g);
        assert!(approx_rel(e_recovered, e));
    }

    // -- Beam Mechanics --

    #[test]
    fn test_cantilever_deflection() {
        let force = 1000.0;
        let length = 2.0;
        let e = 200e9;
        let i = 8.33e-6;
        let expected = force * (length as f64).powi(3) / (3.0 * e * i);
        assert!(approx_rel(
            beam_deflection_cantilever_point(force, length, e, i),
            expected,
        ));
    }

    #[test]
    fn test_simply_supported_deflection() {
        let force = 1000.0;
        let length = 4.0;
        let e = 200e9;
        let i = 8.33e-6;
        let expected = force * (length as f64).powi(3) / (48.0 * e * i);
        assert!(approx_rel(
            beam_deflection_simply_supported_center(force, length, e, i),
            expected,
        ));
    }

    #[test]
    fn test_bending_moment() {
        assert!(approx(bending_moment(500.0, 3.0), 1500.0));
    }

    #[test]
    fn test_bending_stress() {
        let moment = 1500.0;
        let y = 0.05;
        let i = 8.33e-6;
        let expected = moment * y / i;
        assert!(approx_rel(bending_stress(moment, y, i), expected));
    }

    #[test]
    fn test_second_moment_rectangle() {
        let w = 0.1;
        let h = 0.2;
        let expected = w * (h as f64).powi(3) / 12.0;
        assert!(approx_rel(second_moment_rectangle(w, h), expected));
    }

    #[test]
    fn test_second_moment_circle() {
        let r = 0.05;
        let expected = PI * (r as f64).powi(4) / 4.0;
        assert!(approx_rel(second_moment_circle(r), expected));
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
        // Pure shear: s1 = tau, s2 = 0, s3 = -tau => von Mises = tau * sqrt(3)
        let tau = 100e6;
        let expected = tau * 3.0_f64.sqrt();
        assert!(approx_rel(von_mises_stress(tau, 0.0, -tau), expected));
    }

    #[test]
    fn test_safety_factor() {
        assert!(approx(safety_factor(250e6, 125e6), 2.0));
    }

    #[test]
    fn test_strain_energy_density() {
        assert!(approx(strain_energy_density(200e6, 0.001), 100e3));
    }
}
