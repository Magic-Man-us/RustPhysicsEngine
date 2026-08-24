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
    3.774_852_376_853_02e2,
    3.209_377_589_138_469_4e3,
    1.857_777_061_846_031_5e-1,
];
const B: [f64; 4] = [
    2.360_129_095_234_412_1e1,
    2.440_246_379_344_441_7e2,
    1.282_616_526_077_372_3e3,
    2.844_236_833_439_171e3,
];
// Cody coefficients, 0.46875 < |x| <= 4 branch (erfc).
const C: [f64; 9] = [
    5.641_884_969_886_701e-1,
    8.883_149_794_388_375,
    6.611_919_063_714_163e1,
    2.986_351_381_974_001e2,
    8.819_522_212_417_69e2,
    1.712_047_612_634_070_6e3,
    2.051_078_377_826_071_5e3,
    1.230_339_354_797_997_2e3,
    2.153_115_354_744_038_5e-8,
];
const D: [f64; 8] = [
    1.574_492_611_070_983_5e1,
    1.176_939_508_913_125e2,
    5.371_811_018_620_099e2,
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
    1.608_378_514_874_228e-2,
    6.587_491_615_298_378e-4,
    1.631_538_713_730_209_8e-2,
];
const Q: [f64; 5] = [
    2.568_520_192_289_822,
    1.872_952_849_923_460_5e0,
    5.279_051_029_514_285e-1,
    6.051_834_131_244_132e-2,
    2.335_204_976_268_691_8e-3,
];

const ONE_OVER_SQRT_PI: f64 = 5.641_895_835_477_563e-1;
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
    -3.644_412_064_017_82e-21,
    -1.685_059_138_182_016_6e-19,
    1.285_848_071_525_64e-18,
    1.115_787_767_802_518_1e-17,
    -1.333_171_662_854_621e-16,
    2.097_276_787_596_856_2e-17,
    6.637_638_134_358_324e-15,
    -4.054_566_272_975_207e-14,
    -8.151_934_197_605_472e-14,
    2.633_509_315_308_232_3e-12,
    -1.297_513_325_345_353_2e-11,
    -5.415_412_054_294_628e-11,
    1.051_212_273_321_532_3e-9,
    -4.112_633_980_346_984e-9,
    -2.907_036_995_788_200_5e-8,
    4.234_787_782_793_240_4e-7,
    -1.365_469_200_083_467_9e-6,
    -1.388_252_336_278_646_9e-5,
    0.000_186_734_208_034_057_14,
    -0.000_740_702_534_166_267,
    -0.006_033_670_871_430_149,
    0.240_158_182_425_589_62,
    1.653_654_562_683_102_7,
];
const GILES_MID: [f64; 19] = [
    2.213_737_692_177_578_7e-9,
    9.075_656_193_888_539e-8,
    -2.751_740_629_706_454_5e-7,
    1.823_962_921_438_922_8e-8,
    1.502_740_396_890_982_8e-6,
    -4.013_867_526_981_546e-6,
    2.923_444_908_995_544_6e-6,
    1.247_530_448_167_177_9e-5,
    -4.731_822_900_905_573_4e-5,
    6.828_485_145_957_318e-5,
    2.403_111_038_709_789_4e-5,
    -0.000_355_037_520_362_847_5,
    0.000_953_289_379_737_380_5,
    -0.001_688_275_556_023_504_7,
    0.002_491_442_096_107_851,
    -0.003_751_208_507_569_241,
    0.005_370_914_553_590_064,
    1.005_258_967_694_159_2,
    3.083_885_610_492_220_8,
];
const GILES_TAIL: [f64; 17] = [
    -2.710_992_061_643_857_3e-11,
    -2.555_641_816_996_525e-10,
    1.507_657_269_350_054_8e-9,
    -3.789_465_440_126_737e-9,
    7.615_701_208_078_34e-9,
    -1.496_002_662_714_924e-8,
    2.914_795_345_090_108e-8,
    -6.771_199_775_845_234e-8,
    2.290_048_222_802_665_5e-7,
    -9.929_827_294_231_7e-7,
    4.526_062_597_223_154e-6,
    -1.968_177_810_553_167e-5,
    7.599_527_703_001_776e-5,
    -0.000_215_030_119_300_444_77,
    -0.000_138_719_318_336_231_22,
    1.010_300_464_864_534_4,
    4.849_906_401_408_584,
];

fn polyval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// Inverse error function: erfinv(erf(x)) = x for p ∈ (−1, 1).
///
/// Returns ±∞ at p = ±1 and NaN outside [−1, 1].
#[must_use]
pub fn erfinv(p: f64) -> f64 {
    if p.is_nan() || !(-1.0..=1.0).contains(&p) {
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
        // erfc(5) = 1.537459794428035e-12 (mpmath)
        assert!((erfc(5.0) / 1.537459794428035e-12 - 1.0).abs() < 1e-12);
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
