//! Continued fractions: expansions, convergents, the periodic expansion of
//! a square root, Pell's equation, generalized continued fractions by the
//! modified Lentz algorithm, and the Gauss-map statistics.

use crate::exact::bigint::BigInt;
use crate::exact::rational::Rational;

/// The simple continued-fraction expansion of a float, `[a0; a1, a2, ...]`.
///
/// Stops after `max_terms`, or earlier once the remaining fractional part
/// is too small to yield a meaningful term. Only the leading terms of the
/// result describe the intended real number: an `f64` carries about 53
/// bits, so terms beyond roughly the twentieth describe the rounding of
/// the input rather than the number itself.
///
/// # Panics
/// Panics if `x` is not finite.
#[must_use]
pub fn continued_fraction_f64(x: f64, max_terms: usize) -> Vec<i64> {
    assert!(x.is_finite(), "x must be finite");
    let mut out = Vec::with_capacity(max_terms);
    let mut v = x;
    for _ in 0..max_terms {
        let a = v.floor();
        if a.abs() > i64::MAX as f64 {
            break;
        }
        out.push(a as i64);
        let frac = v - a;
        if frac.abs() < 1e-12 {
            break;
        }
        v = 1.0 / frac;
    }
    out
}

/// The convergents `h_k / k_k` of a simple continued fraction.
///
/// Uses the standard recurrence `h_k = a_k h_{k-1} + h_{k-2}`, and the
/// same for the denominators.
#[must_use]
pub fn convergents(cf: &[i64]) -> Vec<Rational> {
    let (mut h_prev, mut h) = (BigInt::zero(), BigInt::one());
    let (mut k_prev, mut k) = (BigInt::one(), BigInt::zero());
    let mut out = Vec::with_capacity(cf.len());
    for &a in cf {
        let ab = BigInt::from_i64(a);
        let h_next = ab.mul(&h).add(&h_prev);
        let k_next = ab.mul(&k).add(&k_prev);
        h_prev = std::mem::replace(&mut h, h_next);
        k_prev = std::mem::replace(&mut k, k_next);
        if let Some(q) = Rational::new(h.clone(), k.clone()) {
            out.push(q);
        }
    }
    out
}

/// The periodic continued fraction of `sqrt(n)`, as `(head, period)` with
/// `sqrt(n) = [head; period repeated]`.
///
/// For a perfect square the period is empty. Otherwise the expansion is
/// purely periodic after the first term and the period always ends with
/// `2*a0`, which is the termination test used here.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn periodic_cf_sqrt(n: u64) -> (Vec<u64>, Vec<u64>) {
    assert!(n > 0, "n must be positive");
    let a0 = (n as f64).sqrt() as u64;
    // Guard the float square root against an off-by-one at large n.
    let a0 = (a0 + 2).min(n);
    let a0 = (0..=a0).rev().find(|&c| c * c <= n).expect("a0 exists");
    if a0 * a0 == n {
        return (vec![a0], Vec::new());
    }
    let mut period = Vec::new();
    let (mut m, mut d, mut a) = (0u64, 1u64, a0);
    loop {
        m = d * a - m;
        d = (n - m * m) / d;
        a = (a0 + m) / d;
        period.push(a);
        if a == 2 * a0 {
            break;
        }
    }
    (vec![a0], period)
}

/// The fundamental solution of Pell's equation `x^2 - d y^2 = 1`.
///
/// Returns `None` when `d` is a perfect square, where the equation has
/// only the trivial solution. Otherwise the smallest solution with
/// `y > 0` is a convergent of the continued fraction of `sqrt(d)`.
///
/// # Panics
/// Panics if `d` is zero.
#[must_use]
pub fn pell_fundamental_solution(d: u64) -> Option<(BigInt, BigInt)> {
    assert!(d > 0, "d must be positive");
    let (head, period) = periodic_cf_sqrt(d);
    if period.is_empty() {
        return None;
    }
    let db = BigInt::from_u64(d);
    // Walk the convergents, extending through the period as needed. The
    // solution appears at the end of the first period when its length is
    // even, and at the end of the second when it is odd.
    let (mut h_prev, mut h) = (BigInt::zero(), BigInt::one());
    let (mut k_prev, mut k) = (BigInt::one(), BigInt::zero());
    let terms = head.iter().copied().chain(period.iter().copied().cycle());
    for a in terms.take(1 + 2 * period.len()) {
        let ab = BigInt::from_u64(a);
        let h_next = ab.mul(&h).add(&h_prev);
        let k_next = ab.mul(&k).add(&k_prev);
        h_prev = std::mem::replace(&mut h, h_next);
        k_prev = std::mem::replace(&mut k, k_next);
        if k.is_zero() {
            continue;
        }
        let lhs = h.mul(&h).sub(&db.mul(&k.mul(&k)));
        if lhs == BigInt::one() {
            return Some((h, k));
        }
    }
    None
}

/// Evaluate a generalized continued fraction
/// `b(0) + a(1)/(b(1) + a(2)/(b(2) + ...))` to `n` levels by the modified
/// Lentz algorithm.
///
/// Lentz builds the value from the top down with multiplicative updates,
/// so it never forms the deep nested quotient directly and cannot lose the
/// tail to cancellation. Zero intermediates are nudged to a tiny value,
/// which is the "modified" part.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn generalized_cf_eval(
    a: &dyn Fn(usize) -> f64,
    b: &dyn Fn(usize) -> f64,
    n: usize,
) -> f64 {
    assert!(n > 0, "n must be positive");
    const TINY: f64 = 1e-300;
    let nudge = |v: f64| if v == 0.0 { TINY } else { v };
    let mut f = nudge(b(0));
    let mut c = f;
    let mut d = 0.0f64;
    for j in 1..=n {
        d = nudge(b(j) + a(j) * d);
        c = nudge(b(j) + a(j) / c);
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        // Stop once delta is within one ulp of 1. A tighter threshold can
        // never be met in f64, which would force every call to run all `n`
        // levels and pay a rounding for each: on the golden ratio that
        // costs 3.6e-14 of accuracy against an exactly-correct result here.
        if (delta - 1.0).abs() < f64::EPSILON {
            break;
        }
    }
    f
}

/// The first `n` terms of the continued fraction of `e`.
///
/// `e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, ...]`: after the leading 2 the
/// terms run in blocks of `1, 2k, 1`.
#[must_use]
pub fn cf_e(n: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    out.push(2);
    let mut k = 1i64;
    while out.len() < n {
        for t in [1, 2 * k, 1] {
            if out.len() == n {
                break;
            }
            out.push(t);
        }
        k += 1;
    }
    out
}

/// Compute `pi * scale` as a `BigInt`, using Machin's formula
/// `pi/4 = 4 atan(1/5) - atan(1/239)` in fixed point.
fn pi_scaled(scale: &BigInt) -> BigInt {
    fn atan_inv(x: u64, scale: &BigInt) -> BigInt {
        let xb = BigInt::from_u64(x);
        let x2 = xb.mul(&xb);
        let mut power = scale.div_rem(&xb).0; // scale / x^(2k+1)
        let mut sum = power.clone();
        let mut k = 1u64;
        loop {
            power = power.div_rem(&x2).0;
            if power.is_zero() {
                break;
            }
            let term = power.div_rem(&BigInt::from_u64(2 * k + 1)).0;
            if term.is_zero() {
                break;
            }
            // atan(z) = z - z^3/3 + z^5/5 - ...
            sum = if k % 2 == 1 { sum.sub(&term) } else { sum.add(&term) };
            k += 1;
        }
        sum
    }
    let four = BigInt::from_u64(4);
    four.mul(&four.mul(&atan_inv(5, scale)).sub(&atan_inv(239, scale)))
}

/// The first `n` terms of the continued fraction of `pi`.
///
/// `pi` has no known pattern, so the terms are read off a high-precision
/// value computed here in fixed point rather than from an `f64`, which
/// would only support about twenty correct terms. The working precision
/// is chosen generously against the number of terms requested.
#[must_use]
pub fn cf_pi_terms(n: usize) -> Vec<i64> {
    if n == 0 {
        return Vec::new();
    }
    // Continued-fraction terms consume roughly one decimal digit each on
    // average (Levy's constant is about 3.28 per convergent denominator),
    // so ask for several digits per term plus a fixed margin.
    let digits = 4 * n + 40;
    let scale = BigInt::from_u64(10).pow(digits as u64);
    let pi = pi_scaled(&scale);
    let mut value = Rational::new(pi, scale).expect("scale is non-zero");
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let a = value.floor();
        let Some(ai) = a.to_i64() else { break };
        out.push(ai);
        let frac = value.sub(&Rational::from_int(a));
        let Some(inv) = frac.recip() else { break };
        value = inv;
    }
    out
}

/// The orbit of `x` under the Gauss map `G(x) = frac(1/x)`, `n` steps.
///
/// The Gauss map is the shift on continued-fraction expansions: the
/// integer parts of the reciprocals along the orbit are exactly the
/// partial quotients.
///
/// # Panics
/// Panics if `x` is not finite.
#[must_use]
pub fn gauss_map_orbit(x: f64, n: usize) -> Vec<f64> {
    assert!(x.is_finite(), "x must be finite");
    let mut out = Vec::with_capacity(n);
    let mut v = x.fract().abs();
    for _ in 0..n {
        if v == 0.0 {
            break;
        }
        v = (1.0 / v).fract();
        out.push(v);
    }
    out
}

/// The geometric mean of the first `n` continued-fraction terms of `x`.
///
/// For almost every irrational this tends to Khinchin's constant,
/// about 2.685452001. Convergence is very slow, so a short orbit only
/// lands in the neighbourhood.
///
/// # Panics
/// Panics if `x` is not finite.
#[must_use]
pub fn khinchin_estimate(x: f64, n: usize) -> f64 {
    let cf = continued_fraction_f64(x, n + 1);
    let terms: Vec<i64> = cf.into_iter().skip(1).filter(|&a| a > 0).collect();
    if terms.is_empty() {
        return f64::NAN;
    }
    // Average the logarithms rather than multiplying, which would overflow.
    let mean_ln = terms.iter().map(|&a| (a as f64).ln()).sum::<f64>() / terms.len() as f64;
    mean_ln.exp()
}

/// Estimate Levy's constant from the growth of the convergent
/// denominators of `x`: `q_n^(1/n)` tends to `exp(pi^2 / (12 ln 2))`,
/// about 3.275822918.
///
/// # Panics
/// Panics if `x` is not finite.
#[must_use]
pub fn levy_constant_estimate(x: f64, n: usize) -> f64 {
    let cf = continued_fraction_f64(x, n);
    let conv = convergents(&cf);
    let Some(last) = conv.last() else {
        return f64::NAN;
    };
    let k = conv.len() as f64;
    // ln q_n / n, exponentiated; use bit length to avoid overflowing f64.
    let ln_q = last.den.bits() as f64 * std::f64::consts::LN_2;
    (ln_q / k).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expansion_and_convergents() {
        // Rational inputs terminate at their exact expansion.
        assert_eq!(continued_fraction_f64(0.5, 10), [0, 2]);
        assert_eq!(continued_fraction_f64(4.0, 10), [4]);
        assert_eq!(continued_fraction_f64(-2.5, 10), [-3, 2], "floor, not truncation");
        // The golden ratio is all ones, which is what makes it the hardest
        // number to approximate rationally.
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        assert!(continued_fraction_f64(phi, 20).iter().all(|&a| a == 1));

        // Convergents obey the unimodular relation h_k k_{k-1} - h_{k-1} k_k
        // = (-1)^(k+1), and bracket the target from alternating sides.
        let cf = continued_fraction_f64(std::f64::consts::PI, 8);
        let conv = convergents(&cf);
        assert_eq!(conv[0].to_string(), "3");
        assert_eq!(conv[1].to_string(), "22/7");
        assert_eq!(conv[2].to_string(), "333/106");
        assert_eq!(conv[3].to_string(), "355/113");
        for w in conv.windows(2) {
            let d = w[1].num.mul(&w[0].den).sub(&w[0].num.mul(&w[1].den));
            assert_eq!(d.abs(), BigInt::one(), "consecutive convergents not unimodular");
        }
        let pi = std::f64::consts::PI;
        for w in conv.windows(2) {
            assert!((w[0].to_f64() - pi).signum() != (w[1].to_f64() - pi).signum(),
                    "convergents must alternate around the target");
            assert!((w[1].to_f64() - pi).abs() < (w[0].to_f64() - pi).abs(),
                    "convergents must improve");
        }
        assert!(convergents(&[]).is_empty());
    }

    #[test]
    fn test_periodic_sqrt_expansions() {
        // Published expansions.
        assert_eq!(periodic_cf_sqrt(2), (vec![1], vec![2]));
        assert_eq!(periodic_cf_sqrt(3), (vec![1], vec![1, 2]));
        assert_eq!(periodic_cf_sqrt(7), (vec![2], vec![1, 1, 1, 4]));
        assert_eq!(periodic_cf_sqrt(13), (vec![3], vec![1, 1, 1, 1, 6]));
        assert_eq!(periodic_cf_sqrt(61), (vec![7], vec![1, 4, 3, 1, 2, 2, 1, 3, 4, 1, 14]));
        // Perfect squares have no period.
        for k in 1..=20u64 {
            assert_eq!(periodic_cf_sqrt(k * k), (vec![k], Vec::new()), "sqrt({}) is exact", k * k);
        }
        for n in 2..=200u64 {
            let (head, period) = periodic_cf_sqrt(n);
            let a0 = head[0];
            assert!(a0 * a0 <= n && (a0 + 1) * (a0 + 1) > n, "a0 wrong for {n}");
            if period.is_empty() {
                continue;
            }
            // The period always closes on 2*a0, and apart from that last
            // term it is a palindrome -- both classical facts.
            assert_eq!(*period.last().unwrap(), 2 * a0, "period of {n} must end in 2*a0");
            let body = &period[..period.len() - 1];
            let rev: Vec<u64> = body.iter().rev().copied().collect();
            assert_eq!(body, rev.as_slice(), "period body of {n} is not a palindrome");
            // Rebuilding the expansion reproduces sqrt(n) closely.
            let mut terms: Vec<i64> = vec![a0 as i64];
            // Convergent error falls like 1/k^2, so take enough terms that
            // the comparison is limited by f64 rather than by truncation.
            let repeats = 40 / period.len() + 2;
            for _ in 0..repeats {
                terms.extend(period.iter().map(|&a| a as i64));
            }
            let approx = convergents(&terms).last().unwrap().to_f64();
            assert!((approx - (n as f64).sqrt()).abs() < 1e-12, "rebuild failed for {n}");
        }
    }

    #[test]
    fn test_pell_equation() {
        // The roadmap's property: d = 61 is the classic hard case.
        let (x, y) = pell_fundamental_solution(61).expect("61 is not a square");
        assert_eq!(x.to_string_radix(10), "1766319049");
        assert_eq!(y.to_string_radix(10), "226153980");
        // Known small solutions.
        for (d, ex, ey) in [(2u64, "3", "2"), (3, "2", "1"), (5, "9", "4"), (6, "5", "2"),
                            (7, "8", "3"), (13, "649", "180")] {
            let (x, y) = pell_fundamental_solution(d).expect("non-square");
            assert_eq!((x.to_string_radix(10), y.to_string_radix(10)),
                       (ex.to_string(), ey.to_string()), "Pell d={d}");
        }
        // A larger odd-period case.
        let (x, y) = pell_fundamental_solution(109).unwrap();
        assert_eq!(x.to_string_radix(10), "158070671986249");
        assert_eq!(y.to_string_radix(10), "15140424455100");
        // Every returned pair satisfies the equation exactly, and no
        // smaller positive y does.
        for d in 2..=60u64 {
            let Some((x, y)) = pell_fundamental_solution(d) else {
                assert_eq!(((d as f64).sqrt() as u64).pow(2), d, "only squares return None");
                continue;
            };
            let db = BigInt::from_u64(d);
            assert_eq!(x.mul(&x).sub(&db.mul(&y.mul(&y))), BigInt::one(), "d={d}");
            assert!(!y.is_zero());
            // Minimality, checked by brute force where y is small enough.
            if let Some(ys) = y.to_i64() {
                if ys < 20_000 {
                    for cand in 1..ys {
                        let t = d as i128 * cand as i128 * cand as i128 + 1;
                        let s = (t as f64).sqrt() as i128;
                        assert!(!(s * s == t || (s + 1) * (s + 1) == t),
                                "smaller solution y={cand} exists for d={d}");
                    }
                }
            }
        }
        // Squares have no non-trivial solution.
        assert!(pell_fundamental_solution(4).is_none());
        assert!(pell_fundamental_solution(9).is_none());
    }

    #[test]
    fn test_generalized_cf_lentz() {
        // The golden ratio: 1 + 1/(1 + 1/(1 + ...)).
        let phi = generalized_cf_eval(&|_| 1.0, &|_| 1.0, 200);
        assert!((phi - (1.0 + 5f64.sqrt()) / 2.0).abs() < 1e-14, "phi = {phi}");
        // sqrt(2) = 1 + 1/(2 + 1/(2 + ...)).
        let s2 = generalized_cf_eval(&|_| 1.0, &|j| if j == 0 { 1.0 } else { 2.0 }, 200);
        assert!((s2 - 2f64.sqrt()).abs() < 1e-14, "sqrt2 = {s2}");
        // Lord Brouncker's expansion:
        // 4/pi = 1 + 1^2/(2 + 3^2/(2 + 5^2/(2 + ...))).
        let four_over_pi = generalized_cf_eval(
            &|j| ((2 * j - 1) as f64).powi(2),
            &|j| if j == 0 { 1.0 } else { 2.0 },
            60_000,
        );
        assert!((4.0 / four_over_pi - std::f64::consts::PI).abs() < 1e-4,
                "Brouncker gave pi = {}", 4.0 / four_over_pi);
        // Gauss's expansion for tan: tan(x) = x/(1 - x^2/(3 - x^2/(5 - ...))).
        for x in [0.3f64, 0.7, 1.0, -0.5] {
            let t = generalized_cf_eval(
                &|j| if j == 1 { x } else { -x * x },
                &|j| if j == 0 { 0.0 } else { (2 * j - 1) as f64 },
                60,
            );
            assert!((t - x.tan()).abs() < 1e-12, "tan({x}) = {t} vs {}", x.tan());
        }
    }

    #[test]
    fn test_known_expansions_of_e_and_pi() {
        // e = [2; 1,2,1, 1,4,1, 1,6,1, ...].
        assert_eq!(cf_e(11), [2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1]);
        assert_eq!(cf_e(1), [2]);
        assert!(cf_e(0).is_empty());
        // Rebuilding from the pattern converges on e.
        let approx = convergents(&cf_e(30)).last().unwrap().to_f64();
        assert!((approx - std::f64::consts::E).abs() < 1e-15, "e = {approx}");
        // Cross-check the pattern against a direct expansion of the f64,
        // which is only trustworthy for the leading terms.
        let direct = continued_fraction_f64(std::f64::consts::E, 12);
        assert_eq!(direct[..10], cf_e(10)[..10]);

        // pi has no pattern, so the terms come from a high-precision value.
        let want = [3i64, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, 1, 84, 2, 1, 1];
        assert_eq!(cf_pi_terms(25), want);
        assert_eq!(cf_pi_terms(1), [3]);
        assert!(cf_pi_terms(0).is_empty());
        // The 84 at index 21 is far past what an f64 expansion can reach,
        // which is the reason this routine does not use one.
        assert_eq!(cf_pi_terms(60).len(), 60);
        let approx = convergents(&cf_pi_terms(30)).last().unwrap().to_f64();
        assert!((approx - std::f64::consts::PI).abs() < 1e-15, "pi = {approx}");
    }

    #[test]
    fn test_gauss_map_and_constants() {
        // The Gauss map is the shift on continued fractions: the integer
        // part of each reciprocal along the orbit is the next term.
        let x = std::f64::consts::PI.fract();
        let orbit = gauss_map_orbit(x, 6);
        let cf = continued_fraction_f64(std::f64::consts::PI, 8);
        let mut v = x;
        for (k, term) in cf[1..].iter().take(5).enumerate() {
            assert_eq!((1.0 / v).floor() as i64, *term, "orbit term {k}");
            v = orbit[k];
        }
        // A rational orbit terminates.
        assert!(gauss_map_orbit(0.5, 10).len() < 10);
        assert!(gauss_map_orbit(0.0, 5).is_empty());
        assert!(orbit.iter().all(|&v| (0.0..1.0).contains(&v)), "orbit leaves the unit interval");

        // Khinchin and Levy: an f64 supports only about twenty terms, so
        // these land in the neighbourhood rather than converging. The
        // assertions are deliberately loose for that reason.
        let k = khinchin_estimate(std::f64::consts::PI, 20);
        assert!(k > 1.2 && k < 12.0, "Khinchin estimate off-scale: {k}");
        // Quadratic irrationals are the classic exceptions: sqrt(2) has all
        // terms equal to 2, so its geometric mean is exactly 2.
        assert!((khinchin_estimate(2f64.sqrt(), 20) - 2.0).abs() < 1e-9);
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        assert!((khinchin_estimate(phi, 20) - 1.0).abs() < 1e-9, "golden ratio is all ones");

        // Levy's constant for the golden ratio is phi itself, since
        // q_n are the Fibonacci numbers and q_n^(1/n) -> phi.
        let l = levy_constant_estimate(phi, 30);
        assert!((l - phi).abs() < 0.05, "golden-ratio Levy estimate {l} vs {phi}");
        let lp = levy_constant_estimate(std::f64::consts::PI, 18);
        assert!(lp > 1.5 && lp < 8.0, "Levy estimate off-scale: {lp}");
    }
}
