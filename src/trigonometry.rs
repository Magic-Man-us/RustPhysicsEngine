use crate::math::constants::PI;

const TWO_PI: f64 = 2.0 * PI;
const HALF_PI: f64 = PI / 2.0;

// --- Triangle Laws ---

#[must_use]
pub fn law_of_cosines_side(a: f64, b: f64, angle_c: f64) -> f64 {
    (a * a + b * b - 2.0 * a * b * angle_c.cos()).sqrt()
}

#[must_use]
pub fn law_of_cosines_angle(a: f64, b: f64, c: f64) -> f64 {
    let numerator = a * a + b * b - c * c;
    let denominator = 2.0 * a * b;
    (numerator / denominator).clamp(-1.0, 1.0).acos()
}

#[must_use]
pub fn law_of_sines_side(a: f64, angle_a: f64, angle_b: f64) -> f64 {
    a * angle_b.sin() / angle_a.sin()
}

#[must_use]
pub fn law_of_sines_angle(a: f64, b: f64, angle_a: f64) -> f64 {
    (b * angle_a.sin() / a).clamp(-1.0, 1.0).asin()
}

#[must_use]
pub fn triangle_area_sas(a: f64, b: f64, angle_c: f64) -> f64 {
    0.5 * a * b * angle_c.sin()
}

// --- Trig Identities ---

#[must_use]
pub fn sin_sum(a: f64, b: f64) -> f64 {
    a.sin() * b.cos() + a.cos() * b.sin()
}

#[must_use]
pub fn cos_sum(a: f64, b: f64) -> f64 {
    a.cos() * b.cos() - a.sin() * b.sin()
}

#[must_use]
pub fn sin_diff(a: f64, b: f64) -> f64 {
    a.sin() * b.cos() - a.cos() * b.sin()
}

#[must_use]
pub fn cos_diff(a: f64, b: f64) -> f64 {
    a.cos() * b.cos() + a.sin() * b.sin()
}

#[must_use]
pub fn tan_sum(a: f64, b: f64) -> f64 {
    let tan_a = a.tan();
    let tan_b = b.tan();
    (tan_a + tan_b) / (1.0 - tan_a * tan_b)
}

#[must_use]
pub fn double_angle_sin(a: f64) -> f64 {
    2.0 * a.sin() * a.cos()
}

#[must_use]
pub fn double_angle_cos(a: f64) -> f64 {
    let c = a.cos();
    let s = a.sin();
    c * c - s * s
}

#[must_use]
pub fn half_angle_sin(a: f64) -> f64 {
    ((1.0 - a.cos()) / 2.0).abs().sqrt()
}

#[must_use]
pub fn half_angle_cos(a: f64) -> f64 {
    ((1.0 + a.cos()) / 2.0).abs().sqrt()
}

#[must_use]
pub fn product_to_sum_sin_sin(a: f64, b: f64) -> f64 {
    0.5 * ((a - b).cos() - (a + b).cos())
}

#[must_use]
pub fn product_to_sum_cos_cos(a: f64, b: f64) -> f64 {
    0.5 * ((a - b).cos() + (a + b).cos())
}

// --- Hyperbolic Functions ---

#[must_use]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

#[must_use]
pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

#[must_use]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[must_use]
pub fn sech(x: f64) -> f64 {
    1.0 / x.cosh()
}

#[must_use]
pub fn csch(x: f64) -> f64 {
    1.0 / x.sinh()
}

#[must_use]
pub fn coth(x: f64) -> f64 {
    x.cosh() / x.sinh()
}

#[must_use]
pub fn asinh(x: f64) -> f64 {
    (x + (x * x + 1.0).sqrt()).ln()
}

#[must_use]
pub fn acosh(x: f64) -> f64 {
    (x + (x * x - 1.0).sqrt()).ln()
}

#[must_use]
pub fn atanh(x: f64) -> f64 {
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

// --- Angle Utilities ---

#[must_use]
pub fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle % TWO_PI;
    if a < 0.0 {
        a += TWO_PI;
    }
    a
}

#[must_use]
pub fn normalize_angle_signed(angle: f64) -> f64 {
    let mut a = angle % TWO_PI;
    if a >= PI {
        a -= TWO_PI;
    } else if a < -PI {
        a += TWO_PI;
    }
    a
}

#[must_use]
pub fn angular_difference(a: f64, b: f64) -> f64 {
    normalize_angle_signed(b - a)
}

#[must_use]
pub fn is_acute(angle: f64) -> bool {
    let a = normalize_angle(angle);
    a > 0.0 && a < HALF_PI
}

#[must_use]
pub fn is_right(angle: f64, tolerance: f64) -> bool {
    (normalize_angle(angle) - HALF_PI).abs() < tolerance
}

#[must_use]
pub fn is_obtuse(angle: f64) -> bool {
    let a = normalize_angle(angle);
    a > HALF_PI && a < PI
}

#[must_use]
pub fn complementary(angle: f64) -> f64 {
    HALF_PI - angle
}

#[must_use]
pub fn supplementary(angle: f64) -> f64 {
    PI - angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    const EPSILON: f64 = 1e-10;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    // --- Triangle Laws (30-60-90 triangle) ---
    // Sides: 1, sqrt(3), 2. Angles: PI/6, PI/3, PI/2.

    #[test]
    fn test_law_of_cosines_side_30_60_90() {
        let a = 1.0;
        let b = 3.0_f64.sqrt();
        let angle_c = FRAC_PI_2;
        let c = law_of_cosines_side(a, b, angle_c);
        assert!(approx(c, 2.0));
    }

    #[test]
    fn test_law_of_cosines_angle_30_60_90() {
        let a = 1.0;
        let b = 3.0_f64.sqrt();
        let c = 2.0;
        let angle = law_of_cosines_angle(a, b, c);
        assert!(approx(angle, FRAC_PI_2));
    }

    #[test]
    fn test_law_of_sines_side() {
        let a = 1.0;
        let angle_a = PI / 6.0;
        let angle_b = PI / 3.0;
        let b = law_of_sines_side(a, angle_a, angle_b);
        assert!(approx(b, 3.0_f64.sqrt()));
    }

    #[test]
    fn test_law_of_sines_angle() {
        let a = 2.0;
        let b = 1.0;
        let angle_a = FRAC_PI_2;
        let angle_b = law_of_sines_angle(a, b, angle_a);
        assert!(approx(angle_b, PI / 6.0));
    }

    #[test]
    fn test_triangle_area_sas() {
        let a = 1.0;
        let b = 3.0_f64.sqrt();
        let angle_c = FRAC_PI_2;
        let area = triangle_area_sas(a, b, angle_c);
        assert!(approx(area, 3.0_f64.sqrt() / 2.0));
    }

    // --- Identities vs std lib ---

    #[test]
    fn test_sin_sum() {
        let a = PI / 5.0;
        let b = PI / 7.0;
        assert!(approx(sin_sum(a, b), (a + b).sin()));
    }

    #[test]
    fn test_cos_sum() {
        let a = PI / 5.0;
        let b = PI / 7.0;
        assert!(approx(cos_sum(a, b), (a + b).cos()));
    }

    #[test]
    fn test_sin_diff() {
        let a = PI / 3.0;
        let b = PI / 4.0;
        assert!(approx(sin_diff(a, b), (a - b).sin()));
    }

    #[test]
    fn test_cos_diff() {
        let a = PI / 3.0;
        let b = PI / 4.0;
        assert!(approx(cos_diff(a, b), (a - b).cos()));
    }

    #[test]
    fn test_tan_sum() {
        let a = PI / 5.0;
        let b = PI / 7.0;
        assert!(approx(tan_sum(a, b), (a + b).tan()));
    }

    #[test]
    fn test_double_angle_sin() {
        let a = PI / 5.0;
        assert!(approx(double_angle_sin(a), (2.0 * a).sin()));
    }

    #[test]
    fn test_double_angle_cos() {
        let a = PI / 5.0;
        assert!(approx(double_angle_cos(a), (2.0 * a).cos()));
    }

    #[test]
    fn test_half_angle_sin() {
        let a = PI / 3.0;
        assert!(approx(half_angle_sin(a), (a / 2.0).sin()));
    }

    #[test]
    fn test_half_angle_cos() {
        let a = PI / 3.0;
        assert!(approx(half_angle_cos(a), (a / 2.0).cos()));
    }

    #[test]
    fn test_product_to_sum_sin_sin() {
        let a = PI / 4.0;
        let b = PI / 6.0;
        assert!(approx(product_to_sum_sin_sin(a, b), a.sin() * b.sin()));
    }

    #[test]
    fn test_product_to_sum_cos_cos() {
        let a = PI / 4.0;
        let b = PI / 6.0;
        assert!(approx(product_to_sum_cos_cos(a, b), a.cos() * b.cos()));
    }

    // --- Hyperbolic ---

    #[test]
    fn test_sinh_cosh_identity() {
        let x = 1.5;
        let s = sinh(x);
        let c = cosh(x);
        assert!(approx(c * c - s * s, 1.0));
    }

    #[test]
    fn test_tanh() {
        let x = 0.8;
        assert!(approx(tanh(x), x.tanh()));
    }

    #[test]
    fn test_sech() {
        let x = 1.0;
        assert!(approx(sech(x), 1.0 / x.cosh()));
    }

    #[test]
    fn test_csch() {
        let x = 1.0;
        assert!(approx(csch(x), 1.0 / x.sinh()));
    }

    #[test]
    fn test_coth() {
        let x = 1.0;
        assert!(approx(coth(x), x.cosh() / x.sinh()));
    }

    #[test]
    fn test_asinh_roundtrip() {
        let x = 2.5;
        assert!(approx(sinh(asinh(x)), x));
    }

    #[test]
    fn test_acosh_roundtrip() {
        let x = 3.0;
        assert!(approx(cosh(acosh(x)), x));
    }

    #[test]
    fn test_atanh_roundtrip() {
        let x = 0.7;
        assert!(approx(tanh(atanh(x)), x));
    }

    // --- Angle Utilities ---

    #[test]
    fn test_normalize_angle() {
        assert!(approx(normalize_angle(-PI / 2.0), 3.0 * FRAC_PI_2));
        assert!(approx(normalize_angle(3.0 * PI), PI));
        assert!(approx(normalize_angle(0.0), 0.0));
    }

    #[test]
    fn test_normalize_angle_signed() {
        assert!(approx(normalize_angle_signed(3.0 * PI / 2.0), -FRAC_PI_2));
        assert!(approx(normalize_angle_signed(-PI / 4.0), -PI / 4.0));
    }

    #[test]
    fn test_angular_difference() {
        assert!(approx(angular_difference(0.1, 0.2), 0.1));
        assert!(approx(angular_difference(0.0, 2.0 * PI - 0.1), -0.1));
    }

    #[test]
    fn test_is_acute() {
        assert!(is_acute(PI / 4.0));
        assert!(!is_acute(PI / 2.0));
        assert!(!is_acute(PI));
    }

    #[test]
    fn test_is_right() {
        assert!(is_right(FRAC_PI_2, 1e-9));
        assert!(!is_right(PI / 4.0, 1e-9));
    }

    #[test]
    fn test_is_obtuse() {
        assert!(is_obtuse(2.0 * PI / 3.0));
        assert!(!is_obtuse(PI / 4.0));
        assert!(!is_obtuse(PI));
    }

    #[test]
    fn test_complementary() {
        assert!(approx(complementary(PI / 6.0), PI / 3.0));
    }

    #[test]
    fn test_supplementary() {
        assert!(approx(supplementary(PI / 3.0), 2.0 * PI / 3.0));
    }
}
