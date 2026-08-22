//! Bessel functions of integer order.
//!
//! J and Y use the rational approximations of Numerical Recipes §6.5
//! (about 1e-8 absolute accuracy); higher orders use upward recurrence
//! for Y and Miller's downward-recurrence algorithm for J when the
//! argument is smaller than the order. I and K follow the polynomial
//! approximations of Abramowitz & Stegun §9.8 (as in NR §6.6).

use crate::error::SolveError;
use crate::numerical::roots::brent_root;

const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;

/// Bessel function of the first kind, order 0: J₀(x).
#[must_use]
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = 57_568_490_574.0
            + y * (-13_362_590_354.0
                + y * (651_619_640.7
                    + y * (-11_214_424.18 + y * (77_392.330_17 + y * (-184.905_245_6)))));
        let den = 57_568_490_411.0
            + y * (1_029_532_985.0
                + y * (9_494_680.718 + y * (59_272.648_53 + y * (267.853_271_2 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785_398_164;
        let p0 = 1.0
            + y * (-0.109_862_862_7e-2
                + y * (0.273_451_040_7e-4
                    + y * (-0.207_337_063_9e-5 + y * 0.209_388_721_1e-6)));
        let q0 = -0.156_249_999_5e-1
            + y * (0.143_048_876_5e-3
                + y * (-0.691_114_765_1e-5
                    + y * (0.762_109_516_1e-6 + y * (-0.934_935_152e-7))));
        (TWO_OVER_PI / ax).sqrt() * (xx.cos() * p0 - z * xx.sin() * q0)
    }
}

/// Bessel function of the first kind, order 1: J₁(x).
#[must_use]
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439
                            + y * (15_704.482_60 + y * (-30.160_366_06))))));
        let den = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74 + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_491;
        let p1 = 1.0
            + y * (0.183_105e-2
                + y * (-0.351_639_649_6e-4
                    + y * (0.245_752_017_4e-5 + y * (-0.240_337_019e-6))));
        let q1 = 0.046_874_999_95
            + y * (-0.200_269_087_3e-3
                + y * (0.844_919_909_6e-5
                    + y * (-0.882_289_87e-6 + y * 0.105_787_412e-6)));
        let ans = (TWO_OVER_PI / ax).sqrt() * (xx.cos() * p1 - z * xx.sin() * q1);
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    }
}

const MILLER_ACC: f64 = 40.0;
const MILLER_BIG: f64 = 1e10;
const MILLER_BIG_INV: f64 = 1e-10;

/// Bessel function of the first kind, integer order n: Jₙ(x).
/// Upward recurrence for x > n; Miller's downward algorithm otherwise.
#[must_use]
pub fn bessel_jn(n: u32, x: f64) -> f64 {
    match n {
        0 => return bessel_j0(x),
        1 => return bessel_j1(x),
        _ => {}
    }
    let ax = x.abs();
    if ax == 0.0 {
        return 0.0;
    }
    let n_f = n as f64;
    let ans = if ax > n_f {
        // Stable upward recurrence.
        let tox = 2.0 / ax;
        let mut bjm = bessel_j0(ax);
        let mut bj = bessel_j1(ax);
        for j in 1..n {
            let bjp = j as f64 * tox * bj - bjm;
            bjm = bj;
            bj = bjp;
        }
        bj
    } else {
        // Miller's downward recurrence with normalization.
        let tox = 2.0 / ax;
        let m = 2 * ((n + (MILLER_ACC * n_f).sqrt() as u32) / 2);
        let mut jsum = false;
        let mut sum = 0.0;
        let mut ans = 0.0;
        let mut bjp = 0.0;
        let mut bj = 1.0;
        for j in (1..=m).rev() {
            let bjm = j as f64 * tox * bj - bjp;
            bjp = bj;
            bj = bjm;
            if bj.abs() > MILLER_BIG {
                bj *= MILLER_BIG_INV;
                bjp *= MILLER_BIG_INV;
                ans *= MILLER_BIG_INV;
                sum *= MILLER_BIG_INV;
            }
            if jsum {
                sum += bj;
            }
            jsum = !jsum;
            if j == n {
                ans = bjp;
            }
        }
        sum = 2.0 * sum - bj;
        ans / sum
    };
    if x < 0.0 && n % 2 == 1 {
        -ans
    } else {
        ans
    }
}

/// Bessel function of the second kind, order 0: Y₀(x).
///
/// # Panics
/// Panics unless x > 0.
#[must_use]
pub fn bessel_y0(x: f64) -> f64 {
    assert!(x > 0.0, "bessel_y0 requires x > 0");
    if x < 8.0 {
        let y = x * x;
        let num = -2_957_821_389.0
            + y * (7_062_834_065.0
                + y * (-512_359_803.6
                    + y * (10_879_881.29 + y * (-86_327.927_57 + y * 228.462_273_3))));
        let den = 40_076_544_269.0
            + y * (745_249_964.8
                + y * (7_189_466.438 + y * (47_447.264_70 + y * (226.103_024_4 + y))));
        num / den + TWO_OVER_PI * bessel_j0(x) * x.ln()
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 0.785_398_164;
        let p0 = 1.0
            + y * (-0.109_862_862_7e-2
                + y * (0.273_451_040_7e-4
                    + y * (-0.207_337_063_9e-5 + y * 0.209_388_721_1e-6)));
        let q0 = -0.156_249_999_5e-1
            + y * (0.143_048_876_5e-3
                + y * (-0.691_114_765_1e-5
                    + y * (0.762_109_516_1e-6 + y * (-0.934_935_152e-7))));
        (TWO_OVER_PI / x).sqrt() * (xx.sin() * p0 + z * xx.cos() * q0)
    }
}

/// Bessel function of the second kind, order 1: Y₁(x).
///
/// # Panics
/// Panics unless x > 0.
#[must_use]
pub fn bessel_y1(x: f64) -> f64 {
    assert!(x > 0.0, "bessel_y1 requires x > 0");
    if x < 8.0 {
        let y = x * x;
        let num = x
            * (-4.900_604_943e12
                + y * (1.275_274_390e12
                    + y * (-5.153_438_139e10
                        + y * (7.349_264_551e8
                            + y * (-4.237_922_726e6 + y * 8.511_937_935e3)))));
        let den = 2.499_580_570e13
            + y * (4.244_419_664e11
                + y * (3.733_650_367e9
                    + y * (2.245_904_002e7
                        + y * (1.020_426_050e5 + y * (3.549_632_885e2 + y)))));
        num / den + TWO_OVER_PI * (bessel_j1(x) * x.ln() - 1.0 / x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 2.356_194_491;
        let p1 = 1.0
            + y * (0.183_105e-2
                + y * (-0.351_639_649_6e-4
                    + y * (0.245_752_017_4e-5 + y * (-0.240_337_019e-6))));
        let q1 = 0.046_874_999_95
            + y * (-0.200_269_087_3e-3
                + y * (0.844_919_909_6e-5
                    + y * (-0.882_289_87e-6 + y * 0.105_787_412e-6)));
        (TWO_OVER_PI / x).sqrt() * (xx.sin() * p1 + z * xx.cos() * q1)
    }
}

/// Bessel function of the second kind, integer order n: Yₙ(x), by
/// stable upward recurrence.
///
/// # Panics
/// Panics unless x > 0.
#[must_use]
pub fn bessel_yn(n: u32, x: f64) -> f64 {
    assert!(x > 0.0, "bessel_yn requires x > 0");
    match n {
        0 => return bessel_y0(x),
        1 => return bessel_y1(x),
        _ => {}
    }
    let tox = 2.0 / x;
    let mut bym = bessel_y0(x);
    let mut by = bessel_y1(x);
    for j in 1..n {
        let byp = j as f64 * tox * by - bym;
        bym = by;
        by = byp;
    }
    by
}

/// Modified Bessel function of the first kind, order 0: I₀(x).
#[must_use]
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = x / 3.75;
        let y = t * t;
        1.0 + y * (3.515_622_9
            + y * (3.089_942_4
                + y * (1.206_749_2
                    + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

/// Modified Bessel function of the first kind, order 1: I₁(x).
#[must_use]
pub fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let ans = if ax < 3.75 {
        let t = x / 3.75;
        let y = t * t;
        ax * (0.5
            + y * (0.878_905_94
                + y * (0.514_988_69
                    + y * (0.150_849_34
                        + y * (0.026_587_33 + y * (0.003_015_32 + y * 0.000_324_11))))))
    } else {
        let y = 3.75 / ax;
        let mut a = 0.022_829_67 + y * (-0.028_953_12 + y * (0.017_876_54 - y * 0.004_200_59));
        a = 0.398_942_28
            + y * (-0.039_880_24
                + y * (-0.003_620_18 + y * (0.001_638_01 + y * (-0.010_315_55 + y * a))));
        a * (ax.exp() / ax.sqrt())
    };
    if x < 0.0 {
        -ans
    } else {
        ans
    }
}

/// Modified Bessel function of the second kind, order 0: K₀(x).
///
/// # Panics
/// Panics unless x > 0.
#[must_use]
pub fn bessel_k0(x: f64) -> f64 {
    assert!(x > 0.0, "bessel_k0 requires x > 0");
    if x <= 2.0 {
        let y = x * x / 4.0;
        -(x / 2.0).ln() * bessel_i0(x)
            + (-0.577_215_66
                + y * (0.422_784_20
                    + y * (0.230_697_56
                        + y * (0.034_885_90
                            + y * (0.002_626_98 + y * (0.000_107_50 + y * 0.000_007_4))))))
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.253_314_14
                + y * (-0.078_323_58
                    + y * (0.021_895_68
                        + y * (-0.010_624_46
                            + y * (0.005_878_72 + y * (-0.002_515_40 + y * 0.000_532_08))))))
    }
}

/// Modified Bessel function of the second kind, order 1: K₁(x).
///
/// # Panics
/// Panics unless x > 0.
#[must_use]
pub fn bessel_k1(x: f64) -> f64 {
    assert!(x > 0.0, "bessel_k1 requires x > 0");
    if x <= 2.0 {
        let y = x * x / 4.0;
        (x / 2.0).ln() * bessel_i1(x)
            + (1.0 / x)
                * (1.0
                    + y * (0.154_431_44
                        + y * (-0.672_785_79
                            + y * (-0.181_568_97
                                + y * (-0.019_194_02
                                    + y * (-0.001_104_04 + y * (-0.000_046_86)))))))
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.253_314_14
                + y * (0.234_986_19
                    + y * (-0.036_556_20
                        + y * (0.015_042_68
                            + y * (-0.007_803_53 + y * (0.003_256_14 + y * (-0.000_682_45)))))))
    }
}

/// First `count` positive zeros of Jₙ, found by scanning for sign
/// changes (step π/4 starting past x = n) and refining each bracket
/// with Brent's method.
#[must_use]
pub fn bessel_j_zeros(n: u32, count: usize) -> Vec<f64> {
    let f = |x: f64| bessel_jn(n, x);
    let mut zeros = Vec::with_capacity(count);
    let step = std::f64::consts::FRAC_PI_4;
    let mut x = (n as f64).max(0.5); // first zero of J_n exceeds n
    let mut fx = f(x);
    while zeros.len() < count {
        let x_next = x + step;
        let fx_next = f(x_next);
        if fx == 0.0 {
            zeros.push(x);
        } else if fx * fx_next < 0.0 {
            if let Ok(z) = brent_root(&f, x, x_next, 1e-14, 200) {
                zeros.push(z);
            }
        }
        x = x_next;
        fx = fx_next;
    }
    zeros
}

/// Convenience alias family used by some references: Jₙ zeros with a
/// `Result` wrapper for invalid counts.
pub fn bessel_j_zeros_checked(n: u32, count: usize) -> Result<Vec<f64>, SolveError> {
    if count == 0 {
        return Err(SolveError::InvalidArgument("bessel_j_zeros requires count > 0"));
    }
    Ok(bessel_j_zeros(n, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_j0_j1_known_values() {
        // Reference: A&S tables / mpmath.
        // NR rational approximations are accurate to ~1e-8, not machine eps.
        assert!(approx(bessel_j0(0.0), 1.0, 1e-8));
        assert!(approx(bessel_j0(1.0), 0.765_197_686_557_966_6, 1e-8));
        assert!(approx(bessel_j0(10.0), -0.245_935_764_451_348_3, 1e-8));
        assert!(approx(bessel_j1(0.0), 0.0, 1e-12));
        assert!(approx(bessel_j1(1.0), 0.440_050_585_744_933_5, 1e-8));
        assert!(approx(bessel_j1(10.0), 0.043_472_746_168_861_44, 1e-8));
    }

    #[test]
    fn test_y0_y1_known_values() {
        assert!(approx(bessel_y0(1.0), 0.088_256_964_215_676_96, 1e-7));
        assert!(approx(bessel_y1(1.0), -0.781_212_821_300_288_7, 1e-7));
        assert!(approx(bessel_y0(10.0), 0.055_671_167_283_599_4, 1e-7));
    }

    #[test]
    fn test_jn_matches_series_small_x() {
        // J_3(1) = 0.019563353982668405 (mpmath)
        assert!(approx(bessel_jn(3, 1.0), 0.019_563_353_982_668_405, 1e-8));
        // J_5(10) = -0.23406152818679364
        assert!(approx(bessel_jn(5, 10.0), -0.234_061_528_186_793_64, 1e-7));
    }

    #[test]
    fn test_jn_negative_x_parity() {
        assert!(approx(bessel_jn(2, -3.0), bessel_jn(2, 3.0), 1e-12));
        assert!(approx(bessel_jn(3, -3.0), -bessel_jn(3, 3.0), 1e-12));
    }

    #[test]
    fn test_i_and_k_known_values() {
        // I0(1) = 1.2660658777520084, I1(1) = 0.5651591039924851
        assert!(approx(bessel_i0(1.0), 1.266_065_877_752_008_4, 1e-7));
        assert!(approx(bessel_i1(1.0), 0.565_159_103_992_485_1, 1e-7));
        // K0(1) = 0.42102443824070834, K1(1) = 0.6019072301972346
        assert!(approx(bessel_k0(1.0), 0.421_024_438_240_708_34, 1e-7));
        assert!(approx(bessel_k1(1.0), 0.601_907_230_197_234_6, 1e-7));
        // Large-argument branches.
        assert!(approx(bessel_i0(5.0), 27.239_871_823_604_45, 1e-4));
        assert!(approx(bessel_k0(5.0), 0.003_691_098_334_042_594, 1e-8));
    }

    #[test]
    fn test_wronskian_identity() {
        // J1(x) Y0(x) - J0(x) Y1(x) = 2/(pi x)
        for &x in &[0.5, 1.0, 3.0, 7.0, 12.0] {
            let w = bessel_j1(x) * bessel_y0(x) - bessel_j0(x) * bessel_y1(x);
            assert!(approx(w, 2.0 / (PI * x), 1e-6), "x={x}: W={w}");
        }
    }

    #[test]
    fn test_j_zeros_first_of_j0() {
        let zeros = bessel_j_zeros(0, 3);
        // First three zeros of J0: 2.404825557, 5.520078110, 8.653727913
        assert!(approx(zeros[0], 2.404_825_557_695_773, 1e-7));
        assert!(approx(zeros[1], 5.520_078_110_286_311, 1e-7));
        assert!(approx(zeros[2], 8.653_727_912_911_012, 1e-7));
        assert!(bessel_j_zeros_checked(0, 0).is_err());
    }

    #[test]
    fn test_j_zeros_of_j1() {
        let zeros = bessel_j_zeros(1, 2);
        assert!(approx(zeros[0], 3.831_705_970_207_512, 1e-7));
        assert!(approx(zeros[1], 7.015_586_669_815_619, 1e-7));
    }
}
