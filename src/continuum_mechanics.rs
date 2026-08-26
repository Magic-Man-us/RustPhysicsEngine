//! Stress and strain as tensors, and the yield criteria built on them.
//!
//! The stress tensor with its invariants, principal stresses, and the
//! split into hydrostatic and deviatoric parts -- the split that matters
//! because metals yield on the deviatoric part alone, which is why the von
//! Mises criterion ignores hydrostatic pressure entirely.
//!
//! Strain in both the small-strain and Green-Lagrange forms, the isotropic
//! 3-D Hooke's law `σᵢⱼ = λ δᵢⱼ ε_kk + 2μ εᵢⱼ` and its compliance
//! inverse, plane stress and plane strain, and the von Mises, Tresca,
//! Mohr-Coulomb and Drucker-Prager yield criteria.

use crate::linalg::Mat3;

// ── Stress Tensor ────────────────────────────────────────────────────────────

/// Constructs a symmetric 3x3 Cauchy stress tensor from six independent components.
#[must_use]
pub fn stress_tensor(sxx: f64, syy: f64, szz: f64, sxy: f64, sxz: f64, syz: f64) -> Mat3 {
    Mat3::from_rows(
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz],
    )
}

/// Computes the three stress invariants (I1, I2, I3) of a symmetric stress tensor.
///
/// I1 = tr(sigma), I2 = (tr^2 - tr(sigma^2))/2, I3 = det(sigma).
#[must_use]
pub fn stress_invariants(stress: &Mat3) -> (f64, f64, f64) {
    let i1 = stress.trace();
    let sq = stress.mul_mat(stress);
    let i2 = (i1 * i1 - sq.trace()) / 2.0;
    let i3 = stress.determinant();
    (i1, i2, i3)
}

/// Computes the three principal stresses (eigenvalues) of a symmetric 3x3 tensor,
/// returned in descending order: sigma1 >= sigma2 >= sigma3.
///
/// Uses the analytical cubic solution via the characteristic equation det(sigma - lambda*I) = 0.
#[must_use]
pub fn principal_stresses(stress: &Mat3) -> [f64; 3] {
    let (i1, i2, i3) = stress_invariants(stress);

    // Depressed cubic: t^3 + p*t + q = 0 where lambda = t + I1/3
    let i1_over_3 = i1 / 3.0;
    let p = i2 - i1 * i1 / 3.0;
    let q = 2.0 * i1 * i1 * i1 / 27.0 - i1 * i2 / 3.0 + i3;

    // For a real symmetric matrix, the discriminant guarantees three real roots.
    // Use trigonometric method for three real roots of the depressed cubic.
    let r_sq = -p / 3.0;
    if r_sq <= 0.0 {
        // Hydrostatic or near-zero: all eigenvalues equal
        return [i1_over_3, i1_over_3, i1_over_3];
    }
    let r = r_sq.sqrt();
    let cos_arg = (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
    let theta = cos_arg.acos();

    let mut vals = [
        2.0 * r * (theta / 3.0).cos() + i1_over_3,
        2.0 * r * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() + i1_over_3,
        2.0 * r * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos() + i1_over_3,
    ];

    // Sort descending
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    vals
}

/// Hydrostatic (mean) stress: sigma_h = tr(sigma) / 3.
#[must_use]
pub fn hydrostatic_stress(stress: &Mat3) -> f64 {
    stress.trace() / 3.0
}

/// Deviatoric stress tensor: s = sigma - sigma_h * I.
#[must_use]
pub fn deviatoric_stress(stress: &Mat3) -> Mat3 {
    let sh = hydrostatic_stress(stress);
    *stress - Mat3::scale(sh)
}

/// Von Mises equivalent stress from a full stress tensor: sigma_vm = sqrt(3/2 * s:s).
#[must_use]
pub fn von_mises_from_tensor(stress: &Mat3) -> f64 {
    let s = deviatoric_stress(stress);
    let s_sq = s.mul_mat(&s);
    (1.5 * s_sq.trace()).sqrt()
}

/// Maximum shear stress: tau_max = (sigma1 - sigma3) / 2.
#[must_use]
pub fn max_shear_stress(stress: &Mat3) -> f64 {
    let p = principal_stresses(stress);
    (p[0] - p[2]) / 2.0
}

// ── Strain Tensor ────────────────────────────────────────────────────────────

/// Constructs a symmetric 3x3 strain tensor from six independent components.
#[must_use]
pub fn strain_tensor(exx: f64, eyy: f64, ezz: f64, exy: f64, exz: f64, eyz: f64) -> Mat3 {
    Mat3::from_rows(
        [exx, exy, exz],
        [exy, eyy, eyz],
        [exz, eyz, ezz],
    )
}

/// Volumetric strain: epsilon_v = tr(epsilon).
#[must_use]
pub fn volumetric_strain(strain: &Mat3) -> f64 {
    strain.trace()
}

/// Deviatoric strain tensor: e = epsilon - (epsilon_v / 3) * I.
#[must_use]
pub fn deviatoric_strain(strain: &Mat3) -> Mat3 {
    let ev = volumetric_strain(strain) / 3.0;
    *strain - Mat3::scale(ev)
}

/// Small (infinitesimal) strain from the displacement gradient: epsilon = (grad_u + grad_u^T) / 2.
#[must_use]
pub fn strain_from_displacement_gradient(grad_u: &Mat3) -> Mat3 {
    let sum = *grad_u + grad_u.transpose();
    sum.mul_scalar(0.5)
}

/// Green-Lagrange finite strain tensor: E = (F^T F - I) / 2.
#[must_use]
pub fn green_lagrange_strain(deformation_gradient: &Mat3) -> Mat3 {
    let f = deformation_gradient;
    let ft_f = f.transpose().mul_mat(f);
    (ft_f - Mat3::identity()).mul_scalar(0.5)
}

// ── Constitutive Models ──────────────────────────────────────────────────────

/// 3D isotropic linear elastic (Hooke's law):
/// sigma_ij = lambda * tr(epsilon) * delta_ij + 2 * mu * epsilon_ij.
#[must_use]
pub fn hooke_3d(strain: &Mat3, youngs: f64, poisson: f64) -> Mat3 {
    assert!(youngs > 0.0, "Young's modulus must be positive");
    assert!((1.0 + poisson) != 0.0, "poisson must not equal -1");
    assert!((1.0 - 2.0 * poisson) != 0.0, "poisson must not equal 0.5");
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = youngs / (2.0 * (1.0 + poisson));
    let tr_eps = strain.trace();
    Mat3::scale(lambda * tr_eps) + strain.mul_scalar(2.0 * mu)
}

/// Returns the six key elastic constants for an isotropic material:
/// [C11, C12, C44, lambda, mu (shear modulus), K (bulk modulus)].
#[must_use]
pub fn compliance_matrix_isotropic(youngs: f64, poisson: f64) -> [f64; 6] {
    assert!(youngs > 0.0, "Young's modulus must be positive");
    assert!((1.0 + poisson) != 0.0, "poisson must not equal -1");
    assert!((1.0 - 2.0 * poisson) != 0.0, "poisson must not equal 0.5");
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = youngs / (2.0 * (1.0 + poisson));
    let k = youngs / (3.0 * (1.0 - 2.0 * poisson));
    let c11 = lambda + 2.0 * mu;
    let c12 = lambda;
    let c44 = mu;
    [c11, c12, c44, lambda, mu, k]
}

/// Plane stress: returns (sigma_xx, sigma_yy, tau_xy) given in-plane strains.
/// Uses the constitutive relation sigma = E/(1-nu^2) * [1,nu; nu,1] * epsilon for normal,
/// and tau_xy = G * gamma_xy where G = E/(2(1+nu)).
#[must_use]
pub fn plane_stress(
    strain_xx: f64,
    strain_yy: f64,
    strain_xy: f64,
    youngs: f64,
    poisson: f64,
) -> (f64, f64, f64) {
    assert!(youngs > 0.0, "Young's modulus must be positive");
    assert!((1.0 - poisson * poisson) != 0.0, "poisson must not equal +/-1");
    let factor = youngs / (1.0 - poisson * poisson);
    let sigma_xx = factor * (strain_xx + poisson * strain_yy);
    let sigma_yy = factor * (poisson * strain_xx + strain_yy);
    let tau_xy = youngs / (2.0 * (1.0 + poisson)) * 2.0 * strain_xy;
    (sigma_xx, sigma_yy, tau_xy)
}

/// Plane strain: returns (sigma_xx, sigma_yy, tau_xy) given in-plane strains.
/// This is the 3D Hooke's law with epsilon_zz = 0 and the z-normal stress is nonzero but not returned.
#[must_use]
pub fn plane_strain(
    strain_xx: f64,
    strain_yy: f64,
    strain_xy: f64,
    youngs: f64,
    poisson: f64,
) -> (f64, f64, f64) {
    assert!(youngs > 0.0, "Young's modulus must be positive");
    assert!((1.0 + poisson) != 0.0, "poisson must not equal -1");
    assert!((1.0 - 2.0 * poisson) != 0.0, "poisson must not equal 0.5");
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = youngs / (2.0 * (1.0 + poisson));
    let tr_eps = strain_xx + strain_yy; // epsilon_zz = 0
    let sigma_xx = lambda * tr_eps + 2.0 * mu * strain_xx;
    let sigma_yy = lambda * tr_eps + 2.0 * mu * strain_yy;
    let tau_xy = 2.0 * mu * strain_xy;
    (sigma_xx, sigma_yy, tau_xy)
}

// ── Failure Criteria ─────────────────────────────────────────────────────────

/// Tresca equivalent stress: max(|s1-s2|, |s2-s3|, |s3-s1|).
#[must_use]
pub fn tresca_stress(stress: &Mat3) -> f64 {
    let p = principal_stresses(stress);
    let d01 = (p[0] - p[1]).abs();
    let d12 = (p[1] - p[2]).abs();
    let d20 = (p[2] - p[0]).abs();
    d01.max(d12).max(d20)
}

/// Mohr-Coulomb failure criterion: tau - c - sigma * tan(phi).
/// Returns positive if the stress state violates the criterion (failure).
/// `friction_angle` is in radians.
#[must_use]
pub fn mohr_coulomb(normal_stress: f64, shear_stress: f64, cohesion: f64, friction_angle: f64) -> f64 {
    shear_stress - cohesion - normal_stress * friction_angle.tan()
}

/// Drucker-Prager failure criterion: sqrt(J2) + alpha * I1 - k.
/// Returns positive if the stress state violates the criterion (failure).
/// Alpha and k are derived from cohesion c and friction angle phi (radians)
/// using the inscribed-cone approximation (matching Mohr-Coulomb for compression).
#[must_use]
pub fn drucker_prager(stress: &Mat3, cohesion: f64, friction_angle: f64) -> f64 {
    let sin_phi = friction_angle.sin();
    let cos_phi = friction_angle.cos();

    // Inscribed cone (compression meridian match)
    let alpha = 2.0 * sin_phi / (3.0_f64.sqrt() * (3.0 - sin_phi));
    let k = 6.0 * cohesion * cos_phi / (3.0_f64.sqrt() * (3.0 - sin_phi));

    let (i1, _, _) = stress_invariants(stress);
    let s = deviatoric_stress(stress);
    let j2 = s.mul_mat(&s).trace() / 2.0;

    j2.sqrt() + alpha * i1 - k
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;
    const LOOSE_TOLERANCE: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b.abs() < TOLERANCE {
            a.abs() < tol
        } else {
            ((a - b) / b).abs() < tol
        }
    }

    // ── Stress tensor tests ──────────────────────────────────────────────

    #[test]
    fn test_principal_stresses_of_diagonal() {
        let s = stress_tensor(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let p = principal_stresses(&s);
        assert!(approx(p[0], 300.0), "sigma1={}, expected 300", p[0]);
        assert!(approx(p[1], 200.0), "sigma2={}, expected 200", p[1]);
        assert!(approx(p[2], 100.0), "sigma3={}, expected 100", p[2]);
    }

    #[test]
    fn test_hydrostatic_equals_trace_over_3() {
        let s = stress_tensor(10.0, 20.0, 30.0, 5.0, 3.0, 7.0);
        let sh = hydrostatic_stress(&s);
        assert!(approx(sh, 20.0));
    }

    #[test]
    fn test_deviatoric_trace_is_zero() {
        let s = stress_tensor(10.0, 20.0, 30.0, 5.0, 3.0, 7.0);
        let dev = deviatoric_stress(&s);
        let tr = dev.trace();
        assert!(
            tr.abs() < TOLERANCE,
            "deviatoric trace = {tr}, expected 0",
        );
    }

    #[test]
    fn test_von_mises_uniaxial() {
        let sigma = 250.0;
        let s = stress_tensor(sigma, 0.0, 0.0, 0.0, 0.0, 0.0);
        let vm = von_mises_from_tensor(&s);
        assert!(
            approx(vm, sigma),
            "von Mises of uniaxial sigma={} should be {}, got {}",
            sigma, sigma, vm
        );
    }

    #[test]
    fn test_von_mises_hydrostatic_is_zero() {
        let p = 100.0;
        let s = stress_tensor(p, p, p, 0.0, 0.0, 0.0);
        let vm = von_mises_from_tensor(&s);
        assert!(vm.abs() < TOLERANCE, "hydrostatic von Mises should be 0, got {}", vm);
    }

    #[test]
    fn test_stress_invariants_diagonal() {
        let s = stress_tensor(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        let (i1, i2, i3) = stress_invariants(&s);
        assert!(approx(i1, 6.0), "I1={}, expected 6", i1);
        assert!(approx(i2, 11.0), "I2={}, expected 11", i2);
        assert!(approx(i3, 6.0), "I3={}, expected 6", i3);
    }

    #[test]
    fn test_max_shear_stress() {
        let s = stress_tensor(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let tau = max_shear_stress(&s);
        assert!(approx(tau, 100.0), "max shear = {}, expected 100", tau);
    }

    // ── Strain tensor tests ──────────────────────────────────────────────

    #[test]
    fn test_volumetric_strain() {
        let e = strain_tensor(0.001, 0.002, 0.003, 0.0, 0.0, 0.0);
        assert!(approx(volumetric_strain(&e), 0.006));
    }

    #[test]
    fn test_deviatoric_strain_trace_zero() {
        let e = strain_tensor(0.01, 0.02, 0.03, 0.005, 0.003, 0.007);
        let dev = deviatoric_strain(&e);
        assert!(dev.trace().abs() < TOLERANCE);
    }

    #[test]
    fn test_strain_from_displacement_gradient_symmetric() {
        let grad_u = Mat3::from_rows(
            [0.01, 0.02, 0.0],
            [0.04, 0.05, 0.0],
            [0.0, 0.0, 0.03],
        );
        let eps = strain_from_displacement_gradient(&grad_u);
        // Should be symmetric
        assert!(approx(eps.data[0][1], eps.data[1][0]));
        // epsilon_xy = (0.02 + 0.04) / 2 = 0.03
        assert!(approx(eps.data[0][1], 0.03));
    }

    #[test]
    fn test_green_lagrange_reduces_to_small_strain() {
        // For small deformations, F = I + grad_u where grad_u is small,
        // Green-Lagrange E = (F^T F - I)/2 ≈ (grad_u + grad_u^T)/2.
        let eps = 1e-6;
        let f = Mat3::from_rows(
            [1.0 + eps, eps, 0.0],
            [0.0, 1.0 + eps, 0.0],
            [0.0, 0.0, 1.0],
        );
        let e_gl = green_lagrange_strain(&f);

        let grad_u = Mat3::from_rows(
            [eps, eps, 0.0],
            [0.0, eps, 0.0],
            [0.0, 0.0, 0.0],
        );
        let e_small = strain_from_displacement_gradient(&grad_u);

        for i in 0..3 {
            for j in 0..3 {
                let gl_val = e_gl.data[i][j];
                let sm_val = e_small.data[i][j];
                assert!(
                    approx_rel(gl_val, sm_val, 1e-4),
                    "GL[{i}][{j}]={gl_val} vs small[{i}][{j}]={sm_val}",
                );
            }
        }
    }

    // ── Constitutive model tests ─────────────────────────────────────────

    #[test]
    fn test_hooke_uniaxial() {
        // Uniaxial strain: epsilon_xx = 0.001, all others zero
        let youngs = 200e9; // Steel-like
        let poisson = 0.3;
        let eps_xx = 0.001;
        let strain = strain_tensor(eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0);
        let stress = hooke_3d(&strain, youngs, poisson);

        let expected_sxx = 2.692_307_692_307_692e8;
        let expected_syy = 1.153_846_153_846_154e8;

        let sxx = stress.data[0][0];
        assert!(
            approx_rel(sxx, expected_sxx, LOOSE_TOLERANCE),
            "sigma_xx = {sxx}, expected {expected_sxx}",
        );
        let syy = stress.data[1][1];
        assert!(
            approx_rel(syy, expected_syy, LOOSE_TOLERANCE),
            "sigma_yy = {syy}, expected {expected_syy}",
        );
    }

    #[test]
    fn test_compliance_matrix_isotropic_values() {
        let youngs = 200e9;
        let poisson = 0.3;
        let [c11, c12, c44, _lambda, _mu, k] = compliance_matrix_isotropic(youngs, poisson);

        assert!(approx_rel(c11, 2.692_307_692_307_692e11, LOOSE_TOLERANCE));
        assert!(approx_rel(c12, 1.153_846_153_846_154e11, LOOSE_TOLERANCE));
        assert!(approx_rel(c44, 7.692_307_692_307_692e10, LOOSE_TOLERANCE));

        assert!(approx_rel(k, 1.666_666_666_666_667e11, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_plane_stress_uniaxial() {
        let youngs = 200e9;
        let poisson = 0.3;
        let eps = 0.001;
        let (sxx, syy, txy) = plane_stress(eps, 0.0, 0.0, youngs, poisson);
        assert!(approx_rel(sxx, 2.197_802_197_802_198e8, LOOSE_TOLERANCE));
        assert!(approx_rel(syy, 6.593_406_593_406_593e7, LOOSE_TOLERANCE));
        assert!(txy.abs() < TOLERANCE);
    }

    #[test]
    fn test_plane_strain_uniaxial() {
        let youngs = 200e9;
        let poisson = 0.3;
        let eps = 0.001;
        let (sxx, syy, txy) = plane_strain(eps, 0.0, 0.0, youngs, poisson);

        assert!(approx_rel(sxx, 2.692_307_692_307_692e8, LOOSE_TOLERANCE));
        assert!(approx_rel(syy, 1.153_846_153_846_154e8, LOOSE_TOLERANCE));
        assert!(txy.abs() < TOLERANCE);
    }

    // ── Failure criteria tests ───────────────────────────────────────────

    #[test]
    fn test_tresca_uniaxial() {
        let sigma = 250.0;
        let s = stress_tensor(sigma, 0.0, 0.0, 0.0, 0.0, 0.0);
        let tresca = tresca_stress(&s);
        assert!(approx(tresca, sigma), "tresca = {}, expected {}", tresca, sigma);
    }

    #[test]
    fn test_tresca_equals_twice_max_shear() {
        let s = stress_tensor(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let tresca = tresca_stress(&s);
        let tau = max_shear_stress(&s);
        assert!(approx(tresca, 2.0 * tau));
    }

    #[test]
    fn test_mohr_coulomb_no_shear_compressive() {
        let cohesion = 50.0;
        let friction_angle = 30.0_f64.to_radians();
        // Compressive normal stress (negative convention for compression in some formulations,
        // but here we use positive compression)
        let result = mohr_coulomb(100.0, 0.0, cohesion, friction_angle);
        // tau(0) - c - sigma*tan(phi) should be negative (safe)
        assert!(result < 0.0, "should be safe: f = {}", result);
    }

    #[test]
    fn test_drucker_prager_hydrostatic_compression() {
        let cohesion = 50.0;
        let friction_angle = 30.0_f64.to_radians();
        // Large hydrostatic compression should be safe (negative I1 for compression)
        let s = stress_tensor(-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0);
        let result = drucker_prager(&s, cohesion, friction_angle);
        assert!(result < 0.0, "hydrostatic compression should be safe: f = {}", result);
    }

    #[test]
    fn test_principal_stresses_hydrostatic() {
        let s = stress_tensor(100.0, 100.0, 100.0, 0.0, 0.0, 0.0);
        let p = principal_stresses(&s);
        for val in &p {
            assert!((*val - 100.0).abs() < TOLERANCE, "Hydrostatic: all principals should be 100, got {val}");
        }
    }
}
