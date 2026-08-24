/// Visible spectrum lower bound (nm).
const WAVELENGTH_MIN_NM: f64 = 380.0;
/// Visible spectrum upper bound (nm).
const WAVELENGTH_MAX_NM: f64 = 780.0;

// Piecewise wavelength band boundaries (nm).
const BAND_VIOLET_END: f64 = 440.0;
const BAND_BLUE_END: f64 = 490.0;
const BAND_CYAN_END: f64 = 510.0;
const BAND_GREEN_END: f64 = 580.0;
const BAND_YELLOW_END: f64 = 645.0;

// Intensity falloff boundaries (nm).
const INTENSITY_FULL_LOW: f64 = 420.0;
const INTENSITY_FULL_HIGH: f64 = 700.0;

// Blackbody temperature range (K).
const BLACKBODY_MIN_K: f64 = 1000.0;
const BLACKBODY_MAX_K: f64 = 40000.0;
const BLACKBODY_WARM_THRESHOLD: f64 = 6600.0;
const BLACKBODY_BLUE_CUTOFF: f64 = 1900.0;

// Tanner Helland approximation coefficients.
const HELLAND_RED_COEFF: f64 = 329.698_727_446;
const HELLAND_RED_EXP: f64 = -0.133_204_759_2;
const HELLAND_GREEN_WARM_COEFF: f64 = 99.470_802_586_1;
const HELLAND_GREEN_WARM_OFFSET: f64 = 161.119_568_166_1;
const HELLAND_GREEN_COOL_COEFF: f64 = 288.122_169_528_3;
const HELLAND_GREEN_COOL_EXP: f64 = -0.075_514_849_3;
const HELLAND_BLUE_COEFF: f64 = 138.517_731_223_1;
const HELLAND_BLUE_OFFSET: f64 = 305.044_792_730_7;

// sRGB gamma correction thresholds.
const SRGB_LINEAR_THRESHOLD: f64 = 0.003_130_8;
const SRGB_LINEAR_SCALE: f64 = 12.92;
const SRGB_GAMMA_SCALE: f64 = 1.055;
const SRGB_GAMMA_OFFSET: f64 = 0.055;
const SRGB_GAMMA_EXP: f64 = 1.0 / 2.4;
const SRGB_INV_THRESHOLD: f64 = 0.040_45;

// sRGB <-> XYZ (D65) matrices.
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
];

// Luminance coefficients (ITU-R BT.709).
const LUMINANCE_R: f64 = 0.2126;
const LUMINANCE_G: f64 = 0.7152;
const LUMINANCE_B: f64 = 0.0722;

// WCAG contrast ratio offset.
const CONTRAST_OFFSET: f64 = 0.05;

// McCamy's CCT approximation coefficients.
const MCCAMY_C3: f64 = 449.0;
const MCCAMY_C2: f64 = 3525.0;
const MCCAMY_C1: f64 = 6823.3;
const MCCAMY_C0: f64 = 5520.33;
const MCCAMY_X_REF: f64 = 0.3320;
const MCCAMY_Y_REF: f64 = 0.1858;

const CHANNEL_MAX: f64 = 255.0;
const DEGREES_PER_SECTOR: f64 = 60.0;
const HUE_FULL_CIRCLE: f64 = 360.0;

/// Convert a visible light wavelength (380-780 nm) to linear RGB in [0, 1].
///
/// Uses a standard piecewise approximation with intensity falloff at the
/// edges of the visible spectrum.
#[must_use]
pub fn wavelength_to_rgb(wavelength_nm: f64) -> (f64, f64, f64) {
    let (mut r, mut g, mut b) = if !(WAVELENGTH_MIN_NM..=WAVELENGTH_MAX_NM).contains(&wavelength_nm)
    {
        (0.0, 0.0, 0.0)
    } else if wavelength_nm < BAND_VIOLET_END {
        (
            (BAND_VIOLET_END - wavelength_nm) / (BAND_VIOLET_END - WAVELENGTH_MIN_NM),
            0.0,
            1.0,
        )
    } else if wavelength_nm < BAND_BLUE_END {
        (
            0.0,
            (wavelength_nm - BAND_VIOLET_END) / (BAND_BLUE_END - BAND_VIOLET_END),
            1.0,
        )
    } else if wavelength_nm < BAND_CYAN_END {
        (
            0.0,
            1.0,
            (BAND_CYAN_END - wavelength_nm) / (BAND_CYAN_END - BAND_BLUE_END),
        )
    } else if wavelength_nm < BAND_GREEN_END {
        (
            (wavelength_nm - BAND_CYAN_END) / (BAND_GREEN_END - BAND_CYAN_END),
            1.0,
            0.0,
        )
    } else if wavelength_nm < BAND_YELLOW_END {
        (
            1.0,
            (BAND_YELLOW_END - wavelength_nm) / (BAND_YELLOW_END - BAND_GREEN_END),
            0.0,
        )
    } else {
        (1.0, 0.0, 0.0)
    };

    let factor = if wavelength_nm < INTENSITY_FULL_LOW {
        0.3 + 0.7 * (wavelength_nm - WAVELENGTH_MIN_NM)
            / (INTENSITY_FULL_LOW - WAVELENGTH_MIN_NM)
    } else if wavelength_nm > INTENSITY_FULL_HIGH {
        0.3 + 0.7 * (WAVELENGTH_MAX_NM - wavelength_nm)
            / (WAVELENGTH_MAX_NM - INTENSITY_FULL_HIGH)
    } else {
        1.0
    };

    r *= factor;
    g *= factor;
    b *= factor;

    (r, g, b)
}

/// Convert a blackbody temperature (1000-40000 K) to linear RGB in [0, 1].
///
/// Uses the Tanner Helland approximation.
#[must_use]
pub fn blackbody_to_rgb(temperature_k: f64) -> (f64, f64, f64) {
    let temp = temperature_k.clamp(BLACKBODY_MIN_K, BLACKBODY_MAX_K);
    let temp_100 = temp / 100.0;

    let r = if temp <= BLACKBODY_WARM_THRESHOLD {
        CHANNEL_MAX
    } else {
        HELLAND_RED_COEFF * (temp_100 - 60.0).powf(HELLAND_RED_EXP)
    }
    .clamp(0.0, CHANNEL_MAX);

    let g = if temp <= BLACKBODY_WARM_THRESHOLD {
        HELLAND_GREEN_WARM_COEFF * temp_100.ln() - HELLAND_GREEN_WARM_OFFSET
    } else {
        HELLAND_GREEN_COOL_COEFF * (temp_100 - 60.0).powf(HELLAND_GREEN_COOL_EXP)
    }
    .clamp(0.0, CHANNEL_MAX);

    let b = if temp >= BLACKBODY_WARM_THRESHOLD {
        CHANNEL_MAX
    } else if temp <= BLACKBODY_BLUE_CUTOFF {
        0.0
    } else {
        HELLAND_BLUE_COEFF * (temp_100 - 10.0).ln() - HELLAND_BLUE_OFFSET
    }
    .clamp(0.0, CHANNEL_MAX);

    (r / CHANNEL_MAX, g / CHANNEL_MAX, b / CHANNEL_MAX)
}

/// Convert linear RGB [0,1] to HSV. H in [0,360), S and V in [0,1].
#[must_use]
pub fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        DEGREES_PER_SECTOR * (((g - b) / delta) % 6.0)
    } else if max == g {
        DEGREES_PER_SECTOR * ((b - r) / delta + 2.0)
    } else {
        DEGREES_PER_SECTOR * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + HUE_FULL_CIRCLE } else { h };

    (h, s, v)
}

/// Convert HSV to linear RGB. H in [0,360), S and V in [0,1].
#[must_use]
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let h_prime = h / DEGREES_PER_SECTOR;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r1 + m, g1 + m, b1 + m)
}

/// Convert linear RGB [0,1] to HSL. H in [0,360), S and L in [0,1].
#[must_use]
pub fn rgb_to_hsl(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let l = (max + min) / 2.0;

    let s = if delta == 0.0 {
        0.0
    } else {
        delta / (1.0 - (2.0 * l - 1.0).abs())
    };

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        DEGREES_PER_SECTOR * (((g - b) / delta) % 6.0)
    } else if max == g {
        DEGREES_PER_SECTOR * ((b - r) / delta + 2.0)
    } else {
        DEGREES_PER_SECTOR * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + HUE_FULL_CIRCLE } else { h };

    (h, s, l)
}

/// Convert HSL to linear RGB. H in [0,360), S and L in [0,1].
#[must_use]
pub fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / DEGREES_PER_SECTOR;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r1 + m, g1 + m, b1 + m)
}

/// Apply sRGB gamma correction to a linear channel value.
#[must_use]
pub fn linear_to_srgb(c: f64) -> f64 {
    if c <= SRGB_LINEAR_THRESHOLD {
        SRGB_LINEAR_SCALE * c
    } else {
        SRGB_GAMMA_SCALE * c.powf(SRGB_GAMMA_EXP) - SRGB_GAMMA_OFFSET
    }
}

/// Convert an sRGB gamma-encoded channel value back to linear.
#[must_use]
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= SRGB_INV_THRESHOLD {
        c / SRGB_LINEAR_SCALE
    } else {
        ((c + SRGB_GAMMA_OFFSET) / SRGB_GAMMA_SCALE).powf(2.4)
    }
}

/// Convert linear sRGB to CIE XYZ (D65 illuminant).
#[must_use]
pub fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let x = SRGB_TO_XYZ[0][0] * r + SRGB_TO_XYZ[0][1] * g + SRGB_TO_XYZ[0][2] * b;
    let y = SRGB_TO_XYZ[1][0] * r + SRGB_TO_XYZ[1][1] * g + SRGB_TO_XYZ[1][2] * b;
    let z = SRGB_TO_XYZ[2][0] * r + SRGB_TO_XYZ[2][1] * g + SRGB_TO_XYZ[2][2] * b;
    (x, y, z)
}

/// Convert CIE XYZ (D65 illuminant) to linear sRGB.
#[must_use]
pub fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = XYZ_TO_SRGB[0][0] * x + XYZ_TO_SRGB[0][1] * y + XYZ_TO_SRGB[0][2] * z;
    let g = XYZ_TO_SRGB[1][0] * x + XYZ_TO_SRGB[1][1] * y + XYZ_TO_SRGB[1][2] * z;
    let b = XYZ_TO_SRGB[2][0] * x + XYZ_TO_SRGB[2][1] * y + XYZ_TO_SRGB[2][2] * z;
    (r, g, b)
}

/// Compute correlated color temperature from CIE xy chromaticity using
/// McCamy's approximation.
#[must_use]
pub fn correlated_color_temperature(x: f64, y: f64) -> f64 {
    assert!((y - MCCAMY_Y_REF).abs() > f64::EPSILON, "y must differ from McCamy reference (0.1858)");
    let n = (x - MCCAMY_X_REF) / (y - MCCAMY_Y_REF);
    MCCAMY_C3 * n * n * n + MCCAMY_C2 * n * n + MCCAMY_C1 * n + MCCAMY_C0
}

/// Euclidean color difference in RGB space.
#[must_use]
pub fn color_difference_euclidean(
    r1: f64, g1: f64, b1: f64,
    r2: f64, g2: f64, b2: f64,
) -> f64 {
    let dr = r1 - r2;
    let dg = g1 - g2;
    let db = b1 - b2;
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Relative luminance per ITU-R BT.709.
#[must_use]
pub fn luminance(r: f64, g: f64, b: f64) -> f64 {
    LUMINANCE_R * r + LUMINANCE_G * g + LUMINANCE_B * b
}

/// WCAG contrast ratio between two relative luminance values.
#[must_use]
pub fn contrast_ratio(l1: f64, l2: f64) -> f64 {
    let lighter = l1.max(l2);
    let darker = l1.min(l2);
    (lighter + CONTRAST_OFFSET) / (darker + CONTRAST_OFFSET)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 0.05;
    const TIGHT_TOLERANCE: f64 = 1e-6;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn red_light_700nm() {
        let (r, g, b) = wavelength_to_rgb(700.0);
        assert!(r > 0.5, "red channel should dominate: r={r}");
        assert!(g < 0.05, "green should be near zero: g={g}");
        assert!(b < 0.05, "blue should be near zero: b={b}");
    }

    #[test]
    fn green_light_530nm() {
        let (r, g, b) = wavelength_to_rgb(530.0);
        assert!(g > 0.5, "green channel should dominate: g={g}");
        assert!(r < g, "red should be less than green: r={r}, g={g}");
        assert!(b < 0.05, "blue should be near zero: b={b}");
    }

    #[test]
    fn blue_light_450nm() {
        let (r, g, b) = wavelength_to_rgb(450.0);
        assert!(b > 0.5, "blue channel should dominate: b={b}");
        assert!(r < 0.3, "red should be low: r={r}");
        assert!(g < 0.3, "green should be low: g={g}");
    }

    #[test]
    fn outside_visible_spectrum() {
        let (r, g, b) = wavelength_to_rgb(300.0);
        assert_eq!((r, g, b), (0.0, 0.0, 0.0));
        let (r, g, b) = wavelength_to_rgb(800.0);
        assert_eq!((r, g, b), (0.0, 0.0, 0.0));
    }

    #[test]
    fn hsv_roundtrip() {
        let original = (0.8, 0.3, 0.6);
        let (h, s, v) = rgb_to_hsv(original.0, original.1, original.2);
        let (r, g, b) = hsv_to_rgb(h, s, v);
        assert!(approx(r, original.0, TIGHT_TOLERANCE), "r mismatch: {r} vs {}", original.0);
        assert!(approx(g, original.1, TIGHT_TOLERANCE), "g mismatch: {g} vs {}", original.1);
        assert!(approx(b, original.2, TIGHT_TOLERANCE), "b mismatch: {b} vs {}", original.2);
    }

    #[test]
    fn hsl_roundtrip() {
        let original = (0.4, 0.7, 0.2);
        let (h, s, l) = rgb_to_hsl(original.0, original.1, original.2);
        let (r, g, b) = hsl_to_rgb(h, s, l);
        assert!(approx(r, original.0, TIGHT_TOLERANCE), "r mismatch: {r} vs {}", original.0);
        assert!(approx(g, original.1, TIGHT_TOLERANCE), "g mismatch: {g} vs {}", original.1);
        assert!(approx(b, original.2, TIGHT_TOLERANCE), "b mismatch: {b} vs {}", original.2);
    }

    #[test]
    fn srgb_gamma_roundtrip() {
        for val in [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let encoded = linear_to_srgb(val);
            let decoded = srgb_to_linear(encoded);
            assert!(
                approx(decoded, val, TIGHT_TOLERANCE),
                "roundtrip failed for {val}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    #[test]
    fn blackbody_6500k_approximately_white() {
        let (r, g, b) = blackbody_to_rgb(6500.0);
        assert!(r > 0.9, "red should be high for ~daylight: r={r}");
        assert!(g > 0.9, "green should be high for ~daylight: g={g}");
        assert!(b > 0.9, "blue should be high for ~daylight: b={b}");
        let spread = (r - g).abs().max((g - b).abs()).max((r - b).abs());
        assert!(spread < 0.15, "channels should be close for white: spread={spread}");
    }

    #[test]
    fn blackbody_low_temp_is_reddish() {
        let (r, g, b) = blackbody_to_rgb(2000.0);
        assert!(r > g, "red should exceed green at 2000K");
        assert!(r > b, "red should exceed blue at 2000K");
    }

    #[test]
    fn xyz_roundtrip() {
        let (r0, g0, b0) = (0.5, 0.3, 0.8);
        let (x, y, z) = rgb_to_xyz(r0, g0, b0);
        let (r, g, b) = xyz_to_rgb(x, y, z);
        assert!(approx(r, r0, TIGHT_TOLERANCE), "r mismatch");
        assert!(approx(g, g0, TIGHT_TOLERANCE), "g mismatch");
        assert!(approx(b, b0, TIGHT_TOLERANCE), "b mismatch");
    }

    #[test]
    fn luminance_white_is_one() {
        let l = luminance(1.0, 1.0, 1.0);
        assert!(approx(l, 1.0, TIGHT_TOLERANCE));
    }

    #[test]
    fn contrast_ratio_black_white() {
        let ratio = contrast_ratio(1.0, 0.0);
        assert!(approx(ratio, 21.0, TOLERANCE));
    }

    #[test]
    fn color_difference_same_color() {
        let d = color_difference_euclidean(0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
        assert!(approx(d, 0.0, TIGHT_TOLERANCE));
    }

    #[test]
    fn correlated_color_temperature_mccamy() {
        // McCamy's approximation: n = (x - 0.3320) / (y - 0.1858)
        // For x=0.3320, n=0, CCT = C0 = 5520.33
        let cct_at_ref = correlated_color_temperature(MCCAMY_X_REF, MCCAMY_Y_REF + 0.15);
        // n = 0 => CCT = 5520.33
        assert!(
            (cct_at_ref - MCCAMY_C0).abs() < 1.0,
            "at reference x, CCT should be C0={MCCAMY_C0}, got {cct_at_ref}"
        );
    }

    // --- wavelength_to_rgb: cover violet band (380-440 nm) ---
    #[test]
    fn violet_light_400nm() {
        // 400 nm is in [380, 440): violet band with intensity falloff
        let (r, g, b) = wavelength_to_rgb(400.0);
        // r = (440 - 400) / (440 - 380) = 40/60 = 0.6667
        // g = 0, b = 1.0
        // factor = 0.3 + 0.7 * (400 - 380) / (420 - 380) = 0.3 + 0.7 * 0.5 = 0.65
        assert!(r > 0.0, "violet has some red: r={r}");
        assert!(approx(g, 0.0, TIGHT_TOLERANCE), "green should be zero: g={g}");
        assert!(b > r, "blue should exceed red in violet: b={b}, r={r}");
        let expected_r = (40.0 / 60.0) * 0.65;
        let expected_b = 1.0 * 0.65;
        assert!(approx(r, expected_r, TIGHT_TOLERANCE), "r={r}, expected {expected_r}");
        assert!(approx(b, expected_b, TIGHT_TOLERANCE), "b={b}, expected {expected_b}");
    }

    // --- wavelength_to_rgb: cover cyan band (490-510 nm) ---
    #[test]
    fn cyan_light_500nm() {
        // 500 nm is in [490, 510): cyan band, full intensity
        let (r, g, b) = wavelength_to_rgb(500.0);
        // r = 0, g = 1.0, b = (510 - 500) / (510 - 490) = 0.5
        // factor = 1.0 (between 420 and 700)
        assert!(approx(r, 0.0, TIGHT_TOLERANCE), "red should be zero: r={r}");
        assert!(approx(g, 1.0, TIGHT_TOLERANCE), "green should be 1.0: g={g}");
        assert!(approx(b, 0.5, TIGHT_TOLERANCE), "blue should be 0.5: b={b}");
    }

    // --- wavelength_to_rgb: cover yellow band (580-645 nm) ---
    #[test]
    fn yellow_light_600nm() {
        // 600 nm is in [580, 645): yellow band, full intensity
        let (r, g, b) = wavelength_to_rgb(600.0);
        // r = 1.0, g = (645 - 600) / (645 - 580) = 45/65, b = 0
        // factor = 1.0 (between 420 and 700)
        let expected_g = 45.0 / 65.0;
        assert!(approx(r, 1.0, TIGHT_TOLERANCE), "red should be 1.0: r={r}");
        assert!(approx(g, expected_g, TIGHT_TOLERANCE), "g={g}, expected {expected_g}");
        assert!(approx(b, 0.0, TIGHT_TOLERANCE), "blue should be zero: b={b}");
    }

    // --- blackbody_to_rgb: cover high temperature (> 6600 K) branches ---
    #[test]
    fn blackbody_high_temp_10000k() {
        // At 10000 K (> 6600), red uses Helland formula, green uses cool formula, blue = 255
        let (r, g, b) = blackbody_to_rgb(10000.0);
        assert!(b > 0.99, "blue should be ~1.0 at 10000K: b={b}");
        assert!(r < 1.0, "red should be below 1.0 at 10000K: r={r}");
        assert!(r > 0.5, "red should still be significant: r={r}");
        assert!(g > 0.5, "green should still be significant: g={g}");
    }

    // --- blackbody_to_rgb: cover very low temperature (<= 1900 K) for blue = 0 ---
    #[test]
    fn blackbody_very_low_temp_1500k() {
        // At 1500 K (<= 1900), blue channel should be 0
        let (r, g, b) = blackbody_to_rgb(1500.0);
        assert!(approx(b, 0.0, TIGHT_TOLERANCE), "blue should be 0 at 1500K: b={b}");
        assert!(r > g, "red should exceed green at very low temp");
        assert!(approx(r, 1.0, TIGHT_TOLERANCE), "red should be 1.0 (clamped to 255/255)");
    }

    // --- rgb_to_hsv: achromatic (black) ---
    #[test]
    fn hsv_black() {
        let (h, s, v) = rgb_to_hsv(0.0, 0.0, 0.0);
        assert!(approx(h, 0.0, TIGHT_TOLERANCE));
        assert!(approx(s, 0.0, TIGHT_TOLERANCE));
        assert!(approx(v, 0.0, TIGHT_TOLERANCE));
    }

    // --- rgb_to_hsv: achromatic (gray) ---
    #[test]
    fn hsv_gray() {
        let (h, s, v) = rgb_to_hsv(0.5, 0.5, 0.5);
        assert!(approx(h, 0.0, TIGHT_TOLERANCE), "hue undefined for gray: h={h}");
        assert!(approx(s, 0.0, TIGHT_TOLERANCE), "saturation should be 0: s={s}");
        assert!(approx(v, 0.5, TIGHT_TOLERANCE), "value should be 0.5: v={v}");
    }

    // --- rgb_to_hsv: max == green ---
    #[test]
    fn hsv_green_dominant() {
        // Pure green: (0, 1, 0) => H=120, S=1, V=1
        let (h, s, v) = rgb_to_hsv(0.0, 1.0, 0.0);
        assert!(approx(h, 120.0, TIGHT_TOLERANCE), "h={h}");
        assert!(approx(s, 1.0, TIGHT_TOLERANCE), "s={s}");
        assert!(approx(v, 1.0, TIGHT_TOLERANCE), "v={v}");
    }

    // --- rgb_to_hsv: max == blue ---
    #[test]
    fn hsv_blue_dominant() {
        // Pure blue: (0, 0, 1) => H=240, S=1, V=1
        let (h, s, v) = rgb_to_hsv(0.0, 0.0, 1.0);
        assert!(approx(h, 240.0, TIGHT_TOLERANCE), "h={h}");
        assert!(approx(s, 1.0, TIGHT_TOLERANCE), "s={s}");
        assert!(approx(v, 1.0, TIGHT_TOLERANCE), "v={v}");
    }

    // --- rgb_to_hsv: negative hue wrapping ---
    #[test]
    fn hsv_negative_hue_wrapping() {
        // RGB with max=R where g < b produces negative hue before wrapping
        // e.g., (0.8, 0.1, 0.5): max=0.8(R), delta=0.7
        //   h_raw = 60 * ((0.1 - 0.5) / 0.7 % 6) = 60 * (-0.5714 % 6) = 60 * 5.4286 = 325.7
        // Actually, Rust's % can produce negative: (g-b)/delta = -0.4/0.7 = -0.5714
        // -0.5714 % 6.0 in Rust = -0.5714 (remainder preserves sign)
        // h_raw = 60 * -0.5714 = -34.28... < 0 => h = -34.28 + 360 = 325.7
        let (h, s, v) = rgb_to_hsv(0.8, 0.1, 0.5);
        assert!(h > 300.0, "hue should be in magenta range after wrapping: h={h}");
        assert!(h < 360.0, "hue should be < 360: h={h}");
        let (r2, g2, b2) = hsv_to_rgb(h, s, v);
        assert!(approx(r2, 0.8, TIGHT_TOLERANCE), "roundtrip r");
        assert!(approx(g2, 0.1, TIGHT_TOLERANCE), "roundtrip g");
        assert!(approx(b2, 0.5, TIGHT_TOLERANCE), "roundtrip b");
    }

    // --- hsv_to_rgb: cover all 6 sectors ---
    #[test]
    fn hsv_sector_1() {
        // H in [60, 120): sector 1
        let (r, g, b) = hsv_to_rgb(90.0, 1.0, 1.0);
        let (h, s, v) = rgb_to_hsv(r, g, b);
        assert!(approx(h, 90.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
        assert!(approx(s, 1.0, TIGHT_TOLERANCE));
        assert!(approx(v, 1.0, TIGHT_TOLERANCE));
    }

    #[test]
    fn hsv_sector_2() {
        // H in [120, 180): sector 2
        let (r, g, b) = hsv_to_rgb(150.0, 0.8, 0.9);
        let (h, s, v) = rgb_to_hsv(r, g, b);
        assert!(approx(h, 150.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
        assert!(approx(s, 0.8, TIGHT_TOLERANCE));
        assert!(approx(v, 0.9, TIGHT_TOLERANCE));
    }

    #[test]
    fn hsv_sector_3() {
        // H in [180, 240): sector 3
        let (r, g, b) = hsv_to_rgb(200.0, 0.7, 0.6);
        let (h, s, v) = rgb_to_hsv(r, g, b);
        assert!(approx(h, 200.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
        assert!(approx(s, 0.7, TIGHT_TOLERANCE));
        assert!(approx(v, 0.6, TIGHT_TOLERANCE));
    }

    #[test]
    fn hsv_sector_4() {
        // H in [240, 300): sector 4
        let (r, g, b) = hsv_to_rgb(270.0, 0.5, 0.8);
        let (h, s, v) = rgb_to_hsv(r, g, b);
        assert!(approx(h, 270.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
        assert!(approx(s, 0.5, TIGHT_TOLERANCE));
        assert!(approx(v, 0.8, TIGHT_TOLERANCE));
    }

    #[test]
    fn hsv_sector_5() {
        // H in [300, 360): sector 5
        let (r, g, b) = hsv_to_rgb(330.0, 0.9, 0.7);
        let (h, s, v) = rgb_to_hsv(r, g, b);
        assert!(approx(h, 330.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
        assert!(approx(s, 0.9, TIGHT_TOLERANCE));
        assert!(approx(v, 0.7, TIGHT_TOLERANCE));
    }

    // --- rgb_to_hsl: cover all branches ---
    #[test]
    fn hsl_achromatic() {
        // Gray: delta=0, h=0, s=0
        let (h, s, l) = rgb_to_hsl(0.5, 0.5, 0.5);
        assert!(approx(h, 0.0, TIGHT_TOLERANCE));
        assert!(approx(s, 0.0, TIGHT_TOLERANCE));
        assert!(approx(l, 0.5, TIGHT_TOLERANCE));
    }

    #[test]
    fn hsl_green_dominant() {
        // Pure green: H=120, S=1, L=0.5
        let (h, s, l) = rgb_to_hsl(0.0, 1.0, 0.0);
        assert!(approx(h, 120.0, TIGHT_TOLERANCE), "h={h}");
        assert!(approx(s, 1.0, TIGHT_TOLERANCE), "s={s}");
        assert!(approx(l, 0.5, TIGHT_TOLERANCE), "l={l}");
    }

    #[test]
    fn hsl_blue_dominant() {
        // Pure blue: H=240, S=1, L=0.5
        let (h, s, l) = rgb_to_hsl(0.0, 0.0, 1.0);
        assert!(approx(h, 240.0, TIGHT_TOLERANCE), "h={h}");
        assert!(approx(s, 1.0, TIGHT_TOLERANCE), "s={s}");
        assert!(approx(l, 0.5, TIGHT_TOLERANCE), "l={l}");
    }

    #[test]
    fn hsl_negative_hue_wrapping() {
        // Same magenta-ish color that produces negative hue
        let (h, s, l) = rgb_to_hsl(0.8, 0.1, 0.5);
        assert!(h > 300.0, "hue should wrap to positive: h={h}");
        let (r2, g2, b2) = hsl_to_rgb(h, s, l);
        assert!(approx(r2, 0.8, TIGHT_TOLERANCE), "roundtrip r");
        assert!(approx(g2, 0.1, TIGHT_TOLERANCE), "roundtrip g");
        assert!(approx(b2, 0.5, TIGHT_TOLERANCE), "roundtrip b");
    }

    // --- hsl_to_rgb: cover all 6 sectors ---
    #[test]
    fn hsl_sector_1() {
        let (r, g, b) = hsl_to_rgb(90.0, 0.8, 0.5);
        let (h, _, _) = rgb_to_hsl(r, g, b);
        assert!(approx(h, 90.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
    }

    #[test]
    fn hsl_sector_2() {
        let (r, g, b) = hsl_to_rgb(150.0, 0.8, 0.5);
        let (h, _, _) = rgb_to_hsl(r, g, b);
        assert!(approx(h, 150.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
    }

    #[test]
    fn hsl_sector_3() {
        let (r, g, b) = hsl_to_rgb(200.0, 0.8, 0.5);
        let (h, _, _) = rgb_to_hsl(r, g, b);
        assert!(approx(h, 200.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
    }

    #[test]
    fn hsl_sector_4() {
        let (r, g, b) = hsl_to_rgb(270.0, 0.8, 0.5);
        let (h, _, _) = rgb_to_hsl(r, g, b);
        assert!(approx(h, 270.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
    }

    #[test]
    fn hsl_sector_5() {
        let (r, g, b) = hsl_to_rgb(330.0, 0.8, 0.5);
        let (h, _, _) = rgb_to_hsl(r, g, b);
        assert!(approx(h, 330.0, TIGHT_TOLERANCE), "h roundtrip: {h}");
    }

    // --- color_difference_euclidean: non-zero distance ---
    #[test]
    fn color_difference_known_distance() {
        // Distance between (1,0,0) and (0,1,0) = sqrt(2)
        let d = color_difference_euclidean(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        assert!(approx(d, std::f64::consts::SQRT_2, TIGHT_TOLERANCE), "d={d}");
    }

    // --- contrast_ratio: same luminance ---
    #[test]
    fn contrast_ratio_same_luminance() {
        let ratio = contrast_ratio(0.5, 0.5);
        assert!(approx(ratio, 1.0, TIGHT_TOLERANCE), "same luminance => ratio 1.0: {ratio}");
    }

    // --- luminance: pure channels ---
    #[test]
    fn luminance_pure_red() {
        let l = luminance(1.0, 0.0, 0.0);
        assert!(approx(l, LUMINANCE_R, TIGHT_TOLERANCE));
    }

    #[test]
    fn luminance_black() {
        let l = luminance(0.0, 0.0, 0.0);
        assert!(approx(l, 0.0, TIGHT_TOLERANCE));
    }

    // --- hsv_to_rgb / hsl_to_rgb: sector 0 (H in [0, 60)) ---
    #[test]
    fn hsv_sector_0() {
        // H=30 (orange), S=1, V=1 => h_prime=0.5, c=1, x=0.5, m=0
        // sector 0: (c, x, 0) = (1.0, 0.5, 0.0)
        let (r, g, b) = hsv_to_rgb(30.0, 1.0, 1.0);
        assert!(approx(r, 1.0, TIGHT_TOLERANCE), "r={r}");
        assert!(approx(g, 0.5, TIGHT_TOLERANCE), "g={g}");
        assert!(approx(b, 0.0, TIGHT_TOLERANCE), "b={b}");
    }

    #[test]
    fn hsl_sector_0() {
        // H=30, S=1, L=0.5 => c=1, h_prime=0.5, x=0.5, m=0
        // sector 0: (c, x, 0) + m = (1.0, 0.5, 0.0)
        let (r, g, b) = hsl_to_rgb(30.0, 1.0, 0.5);
        assert!(approx(r, 1.0, TIGHT_TOLERANCE), "r={r}");
        assert!(approx(g, 0.5, TIGHT_TOLERANCE), "g={g}");
        assert!(approx(b, 0.0, TIGHT_TOLERANCE), "b={b}");
    }

    // --- correlated_color_temperature: panic on degenerate y ---
    #[test]
    #[should_panic(expected = "y must differ from McCamy reference")]
    fn cct_panics_at_reference_y() {
        let _ = correlated_color_temperature(0.4, MCCAMY_Y_REF);
    }
}
