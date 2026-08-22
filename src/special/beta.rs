//! Beta function and regularized incomplete beta.
//!
//! B(a,b) = Γ(a)Γ(b)/Γ(a+b), evaluated in log space. The regularized
//! incomplete beta I_x(a,b) uses the modified Lentz continued fraction
//! of Numerical Recipes §6.4.

use crate::special::gamma::lgamma;

const MAX_ITER: usize = 500;
const EPS: f64 = 1e-15;
const FPMIN: f64 = 1e-300;

/// Complete beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b).
///
/// # Panics
/// Panics unless a > 0 and b > 0.
#[must_use]
pub fn beta(a: f64, b: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "beta requires a > 0 and b > 0");
    (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
}

/// Lentz continued fraction for the incomplete beta (NR `betacf`).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function I_x(a,b), the CDF of the Beta
/// distribution.
///
/// # Panics
/// Panics unless a > 0, b > 0, and x ∈ [0, 1].
#[must_use]
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "beta_inc requires a > 0 and b > 0");
    assert!((0.0..=1.0).contains(&x), "beta_inc requires x in [0, 1]");
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    let ln_front =
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = ln_front.exp();
    // Continued fraction converges fastest for x < (a+1)/(a+b+2);
    // otherwise use the symmetry I_x(a,b) = 1 − I_{1−x}(b,a).
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_beta_known_values() {
        // B(1,1) = 1; B(2,3) = 1/12; B(0.5,0.5) = π
        assert!(approx(beta(1.0, 1.0), 1.0, 1e-12));
        assert!(approx(beta(2.0, 3.0), 1.0 / 12.0, 1e-13));
        assert!(approx(beta(0.5, 0.5), PI, 1e-10));
    }

    #[test]
    fn test_beta_inc_uniform_case() {
        // I_x(1,1) = x
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(approx(beta_inc(1.0, 1.0, x), x, 1e-13));
        }
    }

    #[test]
    fn test_beta_inc_known_value() {
        // I_{0.5}(2,2) = 0.5 by symmetry; I_{0.25}(2,2) = 0.15625 (exact: x^2(3-2x))
        assert!(approx(beta_inc(2.0, 2.0, 0.5), 0.5, 1e-13));
        assert!(approx(beta_inc(2.0, 2.0, 0.25), 0.15625, 1e-13));
    }

    #[test]
    fn test_beta_inc_symmetry() {
        for &(a, b) in &[(2.0, 5.0), (0.5, 0.5), (3.5, 1.2)] {
            for &x in &[0.1, 0.4, 0.9] {
                let lhs = beta_inc(a, b, x);
                let rhs = 1.0 - beta_inc(b, a, 1.0 - x);
                assert!(approx(lhs, rhs, 1e-12), "a={a} b={b} x={x}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "x in [0, 1]")]
    fn test_beta_inc_rejects_out_of_range() {
        let _ = beta_inc(1.0, 1.0, 1.5);
    }
}
