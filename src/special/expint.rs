//! Exponential integrals Ei(x) and E1(x).

/// Euler-Mascheroni constant.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Exponential integral E1(x) = ∫_x^∞ e^{-t}/t dt for x > 0.
///
/// Power series for x ≤ 1, continued fraction (modified Lentz) for x > 1.
/// Returns infinity at x = 0 and NaN for x < 0.
#[must_use]
pub fn e1(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 1.0 {
        // E1(x) = -gamma - ln x + sum_{k>=1} (-1)^{k+1} x^k / (k * k!)
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..=60 {
            term *= -x / k as f64;
            let add = -term / k as f64;
            sum += add;
            if add.abs() < 1e-18 * sum.abs().max(1e-300) {
                break;
            }
        }
        -EULER_GAMMA - x.ln() + sum
    } else {
        // continued fraction: E1(x) = e^{-x} / (x + 1/(1 + 1/(x + 2/(1 + ...))))
        let tiny = 1e-300;
        let mut b = x + 1.0;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=200 {
            let a = -(i as f64) * (i as f64);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * (-x).exp()
    }
}

/// Exponential integral Ei(x) (Cauchy principal value for x > 0).
///
/// For x < 0, Ei(x) = -E1(-x). Returns -infinity at x = 0.
#[must_use]
pub fn exponential_integral(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return -e1(-x);
    }
    if x < 40.0 {
        // Ei(x) = gamma + ln x + sum_{k>=1} x^k / (k * k!)
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..=120 {
            term *= x / k as f64;
            let add = term / k as f64;
            sum += add;
            if add < 1e-18 * sum {
                break;
            }
        }
        EULER_GAMMA + x.ln() + sum
    } else {
        // asymptotic: Ei(x) ~ e^x/x (1 + 1!/x + 2!/x^2 + ...), truncate at
        // the smallest term
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..=60 {
            let next = term * k as f64 / x;
            if next >= term {
                break;
            }
            term = next;
            sum += term;
        }
        x.exp() / x * sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e1_known_values() {
        // Abramowitz & Stegun table values
        assert!((e1(0.5) - 0.559_773_594_776_160_2).abs() < 1e-12);
        assert!((e1(1.0) - 0.219_383_934_395_520_3).abs() < 1e-12);
        assert!((e1(2.0) - 0.048_900_510_708_061_12).abs() < 1e-12);
        assert!((e1(10.0) - 4.156_968_929_685_325e-6).abs() < 1e-16);
        // small-argument expansion: E1(x) ~ -gamma - ln x
        let x = 1e-8;
        assert!((e1(x) - (-EULER_GAMMA - x.ln())).abs() < 1e-7);
        assert!(e1(0.0).is_infinite());
        assert!(e1(-1.0).is_nan());
    }

    #[test]
    fn test_ei_known_values() {
        assert!((exponential_integral(1.0) - 1.895_117_816_355_936_8).abs() < 1e-12);
        assert!((exponential_integral(2.0) - 4.954_234_356_001_890).abs() < 1e-11);
        assert!(
            (exponential_integral(10.0) - 2_492.228_976_241_877_4).abs() / 2492.0 < 1e-12
        );
        // Ei(-x) = -E1(x)
        assert!((exponential_integral(-1.0) + e1(1.0)).abs() < 1e-14);
        // large-argument asymptotic branch stays consistent with the series
        let a = exponential_integral(39.9);
        let b = exponential_integral(40.1);
        assert!(b > a && (b / a - (0.2_f64).exp() * 39.9 / 40.1).abs() < 1e-3);
    }
}
