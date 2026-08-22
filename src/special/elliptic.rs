//! Elliptic integrals and physical applications.
//!
//! Complete integrals use the arithmetic-geometric mean (Abramowitz &
//! Stegun §17.6); incomplete integrals use the Carlson symmetric forms
//! R_F and R_D (Carlson 1979; NR §6.11). The parameter is m = k².

use crate::error::SolveError;
use crate::math::constants::PI;

const AGM_TOL: f64 = 1e-15;
const AGM_MAX_ITER: usize = 60;

/// Complete elliptic integral of the first kind K(m), parameter m = k²:
/// K(m) = ∫₀^{π/2} dθ/√(1 − m·sin²θ) = π / (2·AGM(1, √(1−m))).
///
/// # Panics
/// Panics unless 0 ≤ m < 1.
#[must_use]
pub fn elliptic_k(m: f64) -> f64 {
    assert!((0.0..1.0).contains(&m), "elliptic_k requires 0 <= m < 1");
    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..AGM_MAX_ITER {
        if (a - b).abs() < AGM_TOL * a {
            break;
        }
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
    }
    PI / (2.0 * a)
}

/// Complete elliptic integral of the second kind E(m), parameter m = k²:
/// E(m) = ∫₀^{π/2} √(1 − m·sin²θ) dθ, via the AGM with the
/// c²-correction sum: E = K·(1 − Σ 2^{n−1}·cₙ²).
///
/// # Panics
/// Panics unless 0 ≤ m ≤ 1 (E(1) = 1 exactly).
#[must_use]
pub fn elliptic_e(m: f64) -> f64 {
    assert!((0.0..=1.0).contains(&m), "elliptic_e requires 0 <= m <= 1");
    if m == 1.0 {
        return 1.0;
    }
    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    let mut c = m.sqrt();
    let mut sum = 0.5 * c * c; // 2^{-1} c_0^2
    let mut pow2 = 1.0;
    for _ in 0..AGM_MAX_ITER {
        if c.abs() < AGM_TOL {
            break;
        }
        let an = 0.5 * (a + b);
        let bn = (a * b).sqrt();
        c = 0.5 * (a - b);
        a = an;
        b = bn;
        sum += pow2 * c * c;
        pow2 *= 2.0;
    }
    let k = PI / (2.0 * a);
    k * (1.0 - sum)
}

const RF_ERRTOL: f64 = 0.0025;
const RD_ERRTOL: f64 = 0.0015;

/// Carlson symmetric integral R_F(x, y, z) (NR `rf`).
fn carlson_rf(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    loop {
        let sqrt_x = x.sqrt();
        let sqrt_y = y.sqrt();
        let sqrt_z = z.sqrt();
        let lambda = sqrt_x * (sqrt_y + sqrt_z) + sqrt_y * sqrt_z;
        x = 0.25 * (x + lambda);
        y = 0.25 * (y + lambda);
        z = 0.25 * (z + lambda);
        let ave = (x + y + z) / 3.0;
        let dx = (ave - x) / ave;
        let dy = (ave - y) / ave;
        let dz = (ave - z) / ave;
        if dx.abs().max(dy.abs()).max(dz.abs()) < RF_ERRTOL {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            return (1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0) / ave.sqrt();
        }
    }
}

/// Carlson symmetric integral R_D(x, y, z) (NR `rd`).
fn carlson_rd(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    let mut sum = 0.0;
    let mut fac = 1.0;
    loop {
        let sqrt_x = x.sqrt();
        let sqrt_y = y.sqrt();
        let sqrt_z = z.sqrt();
        let lambda = sqrt_x * (sqrt_y + sqrt_z) + sqrt_y * sqrt_z;
        sum += fac / (sqrt_z * (z + lambda));
        fac *= 0.25;
        x = 0.25 * (x + lambda);
        y = 0.25 * (y + lambda);
        z = 0.25 * (z + lambda);
        let ave = 0.2 * (x + y + 3.0 * z);
        let dx = (ave - x) / ave;
        let dy = (ave - y) / ave;
        let dz = (ave - z) / ave;
        if dx.abs().max(dy.abs()).max(dz.abs()) < RD_ERRTOL {
            let ea = dx * dy;
            let eb = dz * dz;
            let ec = ea - eb;
            let ed = ea - 6.0 * eb;
            let ee = ed + ec + ec;
            let c1 = 3.0 / 14.0;
            let c2 = 1.0 / 6.0;
            let c3 = 9.0 / 22.0;
            let c4 = 3.0 / 26.0;
            let c5 = 0.25 * c3;
            let c6 = 1.5 * c4;
            return 3.0 * sum
                + fac
                    * (1.0
                        + ed * (-c1 + c5 * ed - c6 * dz * ee)
                        + dz * (c2 * ee + dz * (-c3 * ec + dz * c4 * ea)))
                    / (ave * ave.sqrt());
        }
    }
}

/// Incomplete elliptic integral of the first kind F(φ | m) via Carlson
/// R_F: F = sinφ·R_F(cos²φ, 1 − m·sin²φ, 1).
///
/// # Panics
/// Panics unless 0 ≤ φ ≤ π/2 and m·sin²φ < 1.
#[must_use]
pub fn elliptic_f(phi: f64, m: f64) -> f64 {
    assert!((0.0..=PI / 2.0).contains(&phi), "elliptic_f requires 0 <= phi <= pi/2");
    let s = phi.sin();
    assert!(m * s * s < 1.0, "elliptic_f requires m sin^2(phi) < 1");
    if s == 0.0 {
        return 0.0;
    }
    let c2 = phi.cos() * phi.cos();
    s * carlson_rf(c2, 1.0 - m * s * s, 1.0)
}

/// Incomplete elliptic integral of the second kind E(φ | m) via Carlson
/// forms: E = sinφ·R_F − (m/3)·sin³φ·R_D.
///
/// # Panics
/// Panics unless 0 ≤ φ ≤ π/2 and m·sin²φ < 1.
#[must_use]
pub fn elliptic_e_inc(phi: f64, m: f64) -> f64 {
    assert!((0.0..=PI / 2.0).contains(&phi), "elliptic_e_inc requires 0 <= phi <= pi/2");
    let s = phi.sin();
    assert!(m * s * s < 1.0, "elliptic_e_inc requires m sin^2(phi) < 1");
    if s == 0.0 {
        return 0.0;
    }
    let c2 = phi.cos() * phi.cos();
    let q = 1.0 - m * s * s;
    s * carlson_rf(c2, q, 1.0) - (m / 3.0) * s * s * s * carlson_rd(c2, q, 1.0)
}

/// Exact large-amplitude pendulum period:
/// T = 4·√(L/g)·K(sin²(θ₀/2)).
///
/// Reduces to 2π√(L/g) as the amplitude → 0. Fails with
/// `InvalidArgument` for non-positive length/gravity or amplitude
/// outside [0, π).
pub fn pendulum_period_exact(length: f64, g: f64, amplitude_rad: f64) -> Result<f64, SolveError> {
    if !(length > 0.0) || !(g > 0.0) {
        return Err(SolveError::InvalidArgument(
            "pendulum_period_exact requires positive length and gravity",
        ));
    }
    if !(0.0..PI).contains(&amplitude_rad) {
        return Err(SolveError::InvalidArgument(
            "pendulum_period_exact requires amplitude in [0, pi)",
        ));
    }
    let m = (amplitude_rad / 2.0).sin().powi(2);
    Ok(4.0 * (length / g).sqrt() * elliptic_k(m))
}

/// Exact ellipse perimeter: P = 4·a·E(m) with m = 1 − (b/a)² for
/// a ≥ b (arguments may be given in either order).
///
/// # Panics
/// Panics unless both semi-axes are positive.
#[must_use]
pub fn ellipse_perimeter_exact(a: f64, b: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "ellipse_perimeter_exact requires positive semi-axes");
    let (major, minor) = if a >= b { (a, b) } else { (b, a) };
    let m = 1.0 - (minor / major) * (minor / major);
    4.0 * major * elliptic_e(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_complete_first_kind_known() {
        assert!(approx(elliptic_k(0.0), PI / 2.0, 1e-14));
        // K(0.5) = 1.8540746773013719 (mpmath, parameter convention)
        assert!(approx(elliptic_k(0.5), 1.854_074_677_301_371_9, 1e-12));
    }

    #[test]
    fn test_complete_second_kind_known() {
        assert!(approx(elliptic_e(0.0), PI / 2.0, 1e-14));
        assert!(approx(elliptic_e(1.0), 1.0, 1e-14));
        // E(0.5) = 1.3506438810476755 (mpmath)
        assert!(approx(elliptic_e(0.5), 1.350_643_881_047_675_5, 1e-12));
    }

    #[test]
    fn test_incomplete_reduce_to_complete() {
        for &m in &[0.0, 0.3, 0.8] {
            assert!(approx(elliptic_f(PI / 2.0, m), elliptic_k(m), 1e-9), "m={m}");
            assert!(approx(elliptic_e_inc(PI / 2.0, m), elliptic_e(m), 1e-9), "m={m}");
        }
    }

    #[test]
    fn test_incomplete_zero_amplitude_and_m0() {
        assert_eq!(elliptic_f(0.0, 0.5), 0.0);
        assert_eq!(elliptic_e_inc(0.0, 0.5), 0.0);
        // m = 0: F(phi|0) = E(phi|0) = phi.
        for &phi in &[0.3, 1.0, 1.5] {
            assert!(approx(elliptic_f(phi, 0.0), phi, 1e-9));
            assert!(approx(elliptic_e_inc(phi, 0.0), phi, 1e-9));
        }
    }

    #[test]
    fn test_pendulum_period() {
        let (l, g) = (1.0, 9.81);
        let small = pendulum_period_exact(l, g, 1e-6).unwrap();
        assert!(approx(small, 2.0 * PI * (l / g).sqrt(), 1e-9));
        // 90-degree amplitude: T = 4 sqrt(L/g) K(1/2) ≈ 1.18034 * T_small.
        let quarter = pendulum_period_exact(l, g, PI / 2.0).unwrap();
        let ratio = quarter / (2.0 * PI * (l / g).sqrt());
        assert!(approx(ratio, 1.180_340_599_016_096, 1e-9), "ratio {ratio}");
        assert!(pendulum_period_exact(-1.0, g, 0.1).is_err());
        assert!(pendulum_period_exact(l, g, PI).is_err());
    }

    #[test]
    fn test_ellipse_perimeter() {
        // Circle: 2 pi r.
        assert!(approx(ellipse_perimeter_exact(2.0, 2.0), 4.0 * PI, 1e-10));
        // a=2, b=1: known perimeter 9.688448220547676 (mpmath).
        assert!(approx(ellipse_perimeter_exact(2.0, 1.0), 9.688_448_220_547_676, 1e-9));
        // Argument order does not matter.
        assert!(approx(
            ellipse_perimeter_exact(1.0, 2.0),
            ellipse_perimeter_exact(2.0, 1.0),
            1e-12
        ));
    }
}
