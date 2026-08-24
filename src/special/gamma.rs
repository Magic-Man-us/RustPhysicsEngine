//! Gamma function family.
//!
//! `gamma` uses the Lanczos approximation (g = 7, n = 9); `lgamma` is
//! the same approximation carried in log space so it does not overflow
//! up to very large arguments. The regularized incomplete functions
//! P(a,x) (`gamma_p`) and Q(a,x) (`gamma_q`) follow Numerical Recipes
//! ch. 6.2 (series for x < a+1, Lentz continued fraction otherwise).

use crate::math::constants::PI;

// Lanczos approximation coefficients (g=7, n=9)
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

const MAX_ITER: usize = 500;
const EPS: f64 = 1e-15;
const FPMIN: f64 = 1e-300;

/// Gamma function Γ(z) via the Lanczos approximation, with the
/// reflection formula for z < 0.5.
#[must_use]
pub fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) × Γ(1-z))
        return PI / ((PI * z).sin() * gamma(1.0 - z));
    }

    let z = z - 1.0;
    let mut x = LANCZOS_COEFFICIENTS[0];
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        x += coeff / (z + i as f64);
    }

    let t = z + LANCZOS_G + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

/// Natural log of |Γ(z)|, computed in log space so it stays finite up to
/// z ≈ 1e300 (e.g. `lgamma(1e6)` is exact to ~1e-13 relative).
///
/// # Panics
/// Panics for z ≤ 0 (poles and the reflection region are out of scope
/// for the real-valued solvers this supports).
#[must_use]
pub fn lgamma(z: f64) -> f64 {
    assert!(z > 0.0, "lgamma requires z > 0");
    if z < 0.5 {
        // ln Γ(z) = ln π − ln sin(πz) − ln Γ(1−z)
        return PI.ln() - (PI * z).sin().ln() - lgamma(1.0 - z);
    }
    let zm1 = z - 1.0;
    let mut x = LANCZOS_COEFFICIENTS[0];
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        x += coeff / (zm1 + i as f64);
    }
    let t = zm1 + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (zm1 + 0.5) * t.ln() - t + x.ln()
}

/// Series expansion for P(a,x), valid (fast) for x < a + 1.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Lentz continued fraction for Q(a,x), valid (fast) for x ≥ a + 1.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
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
    (-x + a * x.ln() - lgamma(a)).exp() * h
}

/// Regularized lower incomplete gamma P(a,x) = γ(a,x)/Γ(a).
///
/// # Panics
/// Panics unless a > 0 and x ≥ 0.
#[must_use]
pub fn gamma_p(a: f64, x: f64) -> f64 {
    assert!(a > 0.0, "gamma_p requires a > 0");
    assert!(x >= 0.0, "gamma_p requires x >= 0");
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_cf(a, x)
    }
}

/// Regularized upper incomplete gamma Q(a,x) = 1 − P(a,x), computed by
/// continued fraction for x > a + 1 so large-x values keep precision.
///
/// # Panics
/// Panics unless a > 0 and x ≥ 0.
#[must_use]
pub fn gamma_q(a: f64, x: f64) -> f64 {
    assert!(a > 0.0, "gamma_q requires a > 0");
    assert!(x >= 0.0, "gamma_q requires x >= 0");
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_gamma_integers() {
        assert!(approx(gamma(5.0), 24.0, 1e-10));
        assert!(approx(gamma(1.0), 1.0, 1e-12));
        assert!(approx(gamma(0.5), PI.sqrt(), 1e-12));
    }

    #[test]
    fn test_lgamma_matches_ln_gamma_small() {
        for &z in &[0.1, 0.5, 1.0, 2.5, 10.0, 30.0] {
            assert!(
                approx(lgamma(z), gamma(z).ln(), 1e-10),
                "z={z}: {} vs {}",
                lgamma(z),
                gamma(z).ln()
            );
        }
    }

    #[test]
    fn test_lgamma_no_overflow_large() {
        // lgamma(1e6) = 12815504.569147771 (mpmath); relative check.
        let v = lgamma(1e6);
        assert!(v.is_finite());
        assert!((v / 12_815_504.569_147_771 - 1.0).abs() < 1e-12, "got {v}");
    }

    #[test]
    #[should_panic(expected = "z > 0")]
    fn test_lgamma_rejects_nonpositive() {
        let _ = lgamma(0.0);
    }

    #[test]
    fn test_gamma_p_known() {
        // P(1, x) = 1 - e^{-x}
        for &x in &[0.1, 1.0, 3.0] {
            assert!(approx(gamma_p(1.0, x), 1.0 - (-x).exp(), 1e-13));
        }
        // P(a, 0) = 0, Q(a, 0) = 1
        assert_eq!(gamma_p(2.5, 0.0), 0.0);
        assert_eq!(gamma_q(2.5, 0.0), 1.0);
    }

    #[test]
    fn test_gamma_q_large_x_precision() {
        // Q(2, 50) = 51*e^{-50} = 51 × 1.9287498479639178e-22
        let q = gamma_q(2.0, 50.0);
        let expected = 51.0 * (-50.0_f64).exp();
        assert!((q / expected - 1.0).abs() < 1e-12, "got {q}, expected {expected}");
    }

    #[test]
    fn test_p_plus_q_is_one() {
        for &a in &[0.3, 1.0, 2.5, 10.0] {
            for &x in &[0.1, 1.0, 5.0, 20.0] {
                let s = gamma_p(a, x) + gamma_q(a, x);
                assert!(approx(s, 1.0, 1e-12), "a={a}, x={x}: {s}");
            }
        }
    }
}
