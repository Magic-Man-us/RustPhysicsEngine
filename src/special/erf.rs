//! Error function family.
//!
//! `erf`/`erfc` implement W. J. Cody's rational Chebyshev approximations
//! ("Rational Chebyshev approximation for the error function",
//! Math. Comp. 23, 1969; the SPECFUN `CALERF` algorithm), accurate to
//! full double precision. `erfinv` uses M. Giles' polynomial
//! approximation ("Approximating the erfinv function", GPU Computing
//! Gems, 2012) polished with Newton steps on `erf`.

use crate::math::constants::PI;

// Cody coefficients, |x| <= 0.46875 branch.
const A: [f64; 5] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_6e2,
    3.774_852_376_853_020_2e2,
    3.209_377_589_138_469_4e3,
    1.857_777_061_846_031_5e-1,
];
const B: [f64; 4] = [
    2.360_129_095_234_412_1e1,
    2.440_246_379_344_441_7e2,
    1.282_616_526_077_372_3e3,
    2.844_236_833_439_170_6e3,
];
// Cody coefficients, 0.46875 < |x| <= 4 branch (erfc).
const C: [f64; 9] = [
    5.641_884_969_886_700_9e-1,
    8.883_149_794_388_375_9e0,
    6.611_919_063_714_163e1,
    2.986_351_381_974_001_3e2,
    8.819_522_212_417_691e2,
    1.712_047_612_634_070_6e3,
    2.051_078_377_826_071_5e3,
    1.230_339_354_797_997_2e3,
    2.153_115_354_744_038_5e-8,
];
const D: [f64; 8] = [
    1.574_492_611_070_983_5e1,
    1.176_939_508_913_125e2,
    5.371_811_018_620_098_6e2,
    1.621_389_574_566_690_2e3,
    3.290_799_235_733_459_6e3,
    4.362_619_090_143_247e3,
    3.439_367_674_143_721_6e3,
    1.230_339_354_803_749_4e3,
];
// Cody coefficients, |x| > 4 branch (asymptotic erfc).
const P: [f64; 6] = [
    3.053_266_349_612_323_4e-1,
    3.603_448_999_498_044_4e-1,
    1.257_817_261_112_292_5e-1,
    1.608_378_514_874_227_7e-2,
    6.587_491_615_298_378e-4,
    1.631_538_713_730_209_8e-2,
];
const Q: [f64; 5] = [
    2.568_520_192_289_822_4e0,
    1.872_952_849_923_460_5e0,
    5.279_051_029_514_284_1e-1,
    6.051_834_131_244_131_9e-2,
    2.335_204_976_268_691_8e-3,
];

const ONE_OVER_SQRT_PI: f64 = 5.641_895_835_477_562_9e-1;
const X_SMALL: f64 = 1.11e-16;
const X_BIG: f64 = 26.543; // erfc underflows to 0 beyond this

/// erfc(y)·exp(y²) evaluated for 0.46875 < y ≤ 4 (rational part), then
/// rescaled by an accurately split exp(−y²).
fn erfc_mid(y: f64) -> f64 {
    let mut num = C[8] * y;
    let mut den = y;
    for i in 0..7 {
        num = (num + C[i]) * y;
        den = (den + D[i]) * y;
    }
    let ratio = (num + C[7]) / (den + D[7]);
    // Split y² so exp(-y²) keeps full precision for large y.
    let ysq = (y * 16.0).floor() / 16.0;
    let del = (y - ysq) * (y + ysq);
    (-ysq * ysq).exp() * (-del).exp() * ratio
}

/// erfc(y) for y > 4 via the asymptotic rational approximation.
fn erfc_tail(y: f64) -> f64 {
    if y >= X_BIG {
        return 0.0;
    }
    let z = 1.0 / (y * y);
    let mut num = P[5] * z;
    let mut den = z;
    for i in 0..4 {
        num = (num + P[i]) * z;
        den = (den + Q[i]) * z;
    }
    let mut ratio = z * (num + P[4]) / (den + Q[4]);
    ratio = (ONE_OVER_SQRT_PI - ratio) / y;
    let ysq = (y * 16.0).floor() / 16.0;
    let del = (y - ysq) * (y + ysq);
    (-ysq * ysq).exp() * (-del).exp() * ratio
}

/// Error function erf(x) = (2/√π)·∫₀ˣ e^(−t²) dt, full double precision.
#[must_use]
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();
    if y <= 0.46875 {
        let z = if y > X_SMALL { y * y } else { 0.0 };
        let mut num = A[4] * z;
        let mut den = z;
        for i in 0..3 {
            num = (num + A[i]) * z;
            den = (den + B[i]) * z;
        }
        return x * (num + A[3]) / (den + B[3]);
    }
    let erfc_abs = if y <= 4.0 { erfc_mid(y) } else { erfc_tail(y) };
    if x < 0.0 {
        erfc_abs - 1.0
    } else {
        1.0 - erfc_abs
    }
}

/// Complementary error function erfc(x) = 1 − erf(x), computed without
/// cancellation for large positive x.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();
    if y <= 0.46875 {
        return 1.0 - erf(x);
    }
    let erfc_abs = if y <= 4.0 { erfc_mid(y) } else { erfc_tail(y) };
    if x < 0.0 {
        2.0 - erfc_abs
    } else {
        erfc_abs
    }
}

// Giles (2012) double-precision polynomial branches for erfinv.
const GILES_CENTRAL: [f64; 23] = [
    -3.6444120640178196996e-21,
    -1.685059138182016589e-19,
    1.2858480715256400167e-18,
    1.115787767802518096e-17,
    -1.333171662854620906e-16,
    2.0972767875968561637e-17,
    6.6376381343583238325e-15,
    -4.0545662729752068639e-14,
    -8.1519341976054721522e-14,
    2.6335093153082322977e-12,
    -1.2975133253453532498e-11,
    -5.4154120542946279317e-11,
    1.051212273321532285e-09,
    -4.1126339803469836976e-09,
    -2.9070369957882005086e-08,
    4.2347877827932403518e-07,
    -1.3654692000834678645e-06,
    -1.3882523362786468719e-05,
    0.0001867342080340571352,
    -0.00074070253416626697512,
    -0.0060336708714301490533,
    0.24015818242558961693,
    1.6536545626831027356,
];
const GILES_MID: [f64; 19] = [
    2.2137376921775787049e-09,
    9.0756561938885390979e-08,
    -2.7517406297064545428e-07,
    1.8239629214389227755e-08,
    1.5027403968909827627e-06,
    -4.013867526981545969e-06,
    2.9234449089955446044e-06,
    1.2475304481671778723e-05,
    -4.7318229009055733981e-05,
    6.8284851459573175448e-05,
    2.4031110387097893999e-05,
    -0.0003550375203628474796,
    0.00095328937973738049703,
    -0.0016882755560235047313,
    0.0024914420961078508066,
    -0.0037512085075692412107,
    0.005370914553590063617,
    1.0052589676941592334,
    3.0838856104922207635,
];
const GILES_TAIL: [f64; 17] = [
    -2.7109920616438573243e-11,
    -2.5556418169965252055e-10,
    1.5076572693500548083e-09,
    -3.7894654401267369937e-09,
    7.6157012080783393804e-09,
    -1.4960026627149240478e-08,
    2.9147953450901080826e-08,
    -6.7711997758452339498e-08,
    2.2900482228026654717e-07,
    -9.9298272942317002539e-07,
    4.5260625972231537039e-06,
    -1.9681778105531670567e-05,
    7.5995277030017761139e-05,
    -0.00021503011930044477347,
    -0.00013871931833623122026,
    1.0103004648645343977,
    4.8499064014085844221,
];

fn polyval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// Inverse error function: erfinv(erf(x)) = x for p ∈ (−1, 1).
///
/// Returns ±∞ at p = ±1 and NaN outside [−1, 1].
#[must_use]
pub fn erfinv(p: f64) -> f64 {
    if p.is_nan() || p < -1.0 || p > 1.0 {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == -1.0 {
        return f64::NEG_INFINITY;
    }
    if p == 0.0 {
        return 0.0;
    }

    let w = -((1.0 - p) * (1.0 + p)).ln();
    let mut x = if w < 6.25 {
        p * polyval(&GILES_CENTRAL, w - 3.125)
    } else if w < 16.0 {
        p * polyval(&GILES_MID, w.sqrt() - 3.25)
    } else {
        p * polyval(&GILES_TAIL, w.sqrt() - 5.0)
    };

    // Newton polish on erf(x) - p = 0; derivative is (2/sqrt(pi)) e^{-x^2}.
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..2 {
        let err = erf(x) - p;
        let deriv = two_over_sqrt_pi * (-x * x).exp();
        if deriv == 0.0 {
            break;
        }
        x -= err / deriv;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_erf_known_values() {
        // Reference values from Abramowitz & Stegun / mpmath.
        assert!(approx(erf(0.0), 0.0, 1e-15));
        assert!(approx(erf(0.5), 0.5204998778130465, 1e-14));
        assert!(approx(erf(1.0), 0.8427007929497149, 1e-14));
        assert!(approx(erf(2.0), 0.9953222650189527, 1e-14));
        assert!(approx(erf(3.0), 0.9999779095030014, 1e-14));
    }

    #[test]
    fn test_erfc_large_x_no_underflow_to_wrong_value() {
        // erfc(5) = 1.5374597944280349e-12 (mpmath)
        assert!((erfc(5.0) / 1.5374597944280349e-12 - 1.0).abs() < 1e-12);
        // erfc(10) = 2.0884875837625447e-45
        assert!((erfc(10.0) / 2.0884875837625447e-45 - 1.0).abs() < 1e-12);
        assert_eq!(erfc(27.0), 0.0);
    }

    #[test]
    fn test_erf_odd_symmetry() {
        for &x in &[0.1, 0.5, 1.0, 2.5, 4.5] {
            assert!(approx(erf(-x), -erf(x), 1e-15));
        }
    }

    #[test]
    fn test_erf_plus_erfc_is_one() {
        for &x in &[-6.0, -2.0, -0.3, 0.0, 0.3, 1.0, 2.0, 4.0, 6.0] {
            assert!(approx(erf(x) + erfc(x), 1.0, 1e-14), "x={x}");
        }
    }

    #[test]
    fn test_erfinv_roundtrip() {
        // For |x| ≲ 3.2 the roundtrip is exact to 1e-12. Beyond that,
        // p = erf(x) saturates toward 1 and the ulp of p alone limits
        // recoverable accuracy to ~eps / erf'(x); test against that bound.
        for i in -49..=49 {
            let x = i as f64 / 10.0; // -4.9 ..= 4.9
            let p = erf(x);
            let back = erfinv(p);
            let deriv = (2.0 / PI.sqrt()) * (-x * x).exp();
            let ulp_bound = 4.0 * f64::EPSILON / deriv;
            let tol = 1e-12_f64.max(ulp_bound);
            assert!(approx(back, x, tol), "x={x}, back={back}, tol={tol}");
        }
        // Direct check well inside the representable region.
        for i in -30..=30 {
            let x = i as f64 / 10.0;
            assert!(approx(erfinv(erf(x)), x, 1e-12), "x={x}");
        }
    }

    #[test]
    fn test_erfinv_edge_cases() {
        assert_eq!(erfinv(1.0), f64::INFINITY);
        assert_eq!(erfinv(-1.0), f64::NEG_INFINITY);
        assert_eq!(erfinv(0.0), 0.0);
        assert!(erfinv(1.5).is_nan());
        assert!(erfinv(f64::NAN).is_nan());
    }

    #[test]
    fn test_nan_propagation() {
        assert!(erf(f64::NAN).is_nan());
        assert!(erfc(f64::NAN).is_nan());
    }
}
