use crate::math::constants;

// Atmospheric constants
pub const EARTH_ROTATION_RATE: f64 = 7.2921e-5; // rad/s
/// Standard atmosphere (Pa), re-exported from [`constants::ATM`].
pub const STANDARD_PRESSURE: f64 = constants::ATM;
pub const STANDARD_TEMPERATURE: f64 = 288.15; // K (15°C)
pub const DRY_AIR_MOLAR_MASS: f64 = 0.028_97; // kg/mol

// Magnus formula coefficients for dew point / relative humidity
const MAGNUS_A: f64 = 17.27;
const MAGNUS_B: f64 = 237.7;

// Wind chill validity thresholds (Environment Canada / NWS)
const WIND_CHILL_MAX_TEMP_C: f64 = 10.0;
const WIND_CHILL_MIN_SPEED_KMH: f64 = 4.8;

const MAX_BEAUFORT: u32 = 12;
const BEAUFORT_COEFFICIENT: f64 = 0.836;

// Density altitude pilot formula coefficient (meters per °C deviation)
const DENSITY_ALT_COEFF: f64 = 36.576;

// Standard atmosphere exponent: R * Γ / (M * g) ≈ 0.1903
const STD_ATMO_EXPONENT: f64 = 0.1903;

// ── Wind Dynamics ──────────────────────────────────────────────────

/// Environment Canada / NWS wind chill formula.
/// Valid for temperature <= 10°C and wind speed >= 4.8 km/h.
/// Returns the wind chill temperature in °C.
#[must_use]
pub fn wind_chill(temperature_c: f64, wind_speed_kmh: f64) -> f64 {
    debug_assert!(
        temperature_c <= WIND_CHILL_MAX_TEMP_C,
        "wind chill formula valid for T <= 10°C"
    );
    debug_assert!(
        wind_speed_kmh >= WIND_CHILL_MIN_SPEED_KMH,
        "wind chill formula valid for V >= 4.8 km/h"
    );

    let v_pow = wind_speed_kmh.powf(0.16);
    13.12 + 0.6215 * temperature_c - 11.37 * v_pow + 0.3965 * temperature_c * v_pow
}

/// Convert Beaufort scale number to approximate wind speed in m/s.
/// Uses v = 0.836 × B^(3/2).
#[must_use]
pub fn beaufort_to_speed(beaufort: u32) -> f64 {
    BEAUFORT_COEFFICIENT * (beaufort as f64).powf(1.5)
}

/// Convert wind speed in m/s to Beaufort scale number (clamped 0–12).
/// Inverse of `beaufort_to_speed`.
#[must_use]
pub fn speed_to_beaufort(speed_ms: f64) -> u32 {
    if speed_ms <= 0.0 {
        return 0;
    }
    let b = (speed_ms / BEAUFORT_COEFFICIENT).powf(2.0 / 3.0);
    (b.round() as u32).min(MAX_BEAUFORT)
}

/// Vertical wind shear: dv/dz = (v_top - v_bottom) / Δz.
/// Returns shear in s⁻¹.
#[must_use]
pub fn wind_shear(v_top: f64, v_bottom: f64, height_diff: f64) -> f64 {
    assert!(height_diff != 0.0, "height_diff must be non-zero");
    (v_top - v_bottom) / height_diff
}

/// Wind power density: P/A = ½ρv³ (W/m²).
#[must_use]
pub fn wind_power_density(density: f64, velocity: f64) -> f64 {
    0.5 * density * velocity * velocity * velocity
}

// ── Coriolis Effect ────────────────────────────────────────────────

/// Coriolis parameter: f = 2Ω sin(φ).
#[must_use]
pub fn coriolis_parameter(latitude_rad: f64) -> f64 {
    2.0 * EARTH_ROTATION_RATE * latitude_rad.sin()
}

/// Coriolis acceleration: a = 2Ωv sin(φ).
#[must_use]
pub fn coriolis_acceleration(velocity: f64, latitude_rad: f64) -> f64 {
    coriolis_parameter(latitude_rad) * velocity
}

// ── Atmospheric Structure ──────────────────────────────────────────

/// Barometric pressure at a given height using the hypsometric equation.
/// P = P₀ × exp(-Mgh / (RT))
#[must_use]
pub fn barometric_pressure(
    p0: f64,
    molar_mass: f64,
    g: f64,
    height: f64,
    temperature: f64,
) -> f64 {
    assert!(temperature > 0.0, "temperature must be positive");
    p0 * (-molar_mass * g * height / (constants::R * temperature)).exp()
}

/// Dry adiabatic lapse rate: Γ = g / cp (K/m).
#[must_use]
pub fn dry_adiabatic_lapse_rate(g: f64, cp: f64) -> f64 {
    assert!(cp > 0.0, "specific heat capacity must be positive");
    g / cp
}

/// Temperature at altitude: T = T₀ - Γ × h.
#[must_use]
pub fn temperature_at_altitude(t0: f64, lapse_rate: f64, altitude: f64) -> f64 {
    t0 - lapse_rate * altitude
}

/// Pressure altitude using the standard atmosphere simplification:
/// h = (T₀ / Γ) × (1 - (P / P₀)^0.1903)
#[must_use]
pub fn pressure_altitude(p0: f64, pressure: f64, lapse_rate: f64, t0: f64) -> f64 {
    assert!(p0 > 0.0, "reference pressure must be positive");
    assert!(lapse_rate != 0.0, "lapse_rate must be non-zero");
    (t0 / lapse_rate) * (1.0 - (pressure / p0).powf(STD_ATMO_EXPONENT))
}

/// Density altitude from pressure altitude and temperature deviation:
/// DA = PA + 36.576 × (T - T_std) meters.
#[must_use]
pub fn density_altitude(
    pressure_alt: f64,
    temperature_c: f64,
    standard_temp_c: f64,
) -> f64 {
    pressure_alt + DENSITY_ALT_COEFF * (temperature_c - standard_temp_c)
}

/// Scale height: H = RT / (Mg).
#[must_use]
pub fn scale_height(temperature: f64, molar_mass: f64, g: f64) -> f64 {
    assert!(molar_mass > 0.0, "molar_mass must be positive");
    assert!(g > 0.0, "gravitational acceleration must be positive");
    constants::R * temperature / (molar_mass * g)
}

// ── Humidity ───────────────────────────────────────────────────────

/// Dew point via the Magnus formula.
/// α = (a×T)/(b+T) + ln(RH), then Td = (b×α)/(a-α).
/// `relative_humidity` is fractional (0.0–1.0).
#[must_use]
pub fn dew_point(temperature_c: f64, relative_humidity: f64) -> f64 {
    assert!(relative_humidity > 0.0, "relative_humidity must be positive for ln()");
    assert!((MAGNUS_B + temperature_c) != 0.0, "temperature_c must not equal -MAGNUS_B");
    let alpha =
        (MAGNUS_A * temperature_c) / (MAGNUS_B + temperature_c) + relative_humidity.ln();
    assert!((MAGNUS_A - alpha) != 0.0, "degenerate dew point formula: alpha equals MAGNUS_A");
    (MAGNUS_B * alpha) / (MAGNUS_A - alpha)
}

/// Relative humidity from temperature and dew point (returns fractional 0.0–1.0).
/// RH = exp((a×Td)/(b+Td) - (a×T)/(b+T))
#[must_use]
pub fn relative_humidity(temperature_c: f64, dew_point_c: f64) -> f64 {
    ((MAGNUS_A * dew_point_c) / (MAGNUS_B + dew_point_c)
        - (MAGNUS_A * temperature_c) / (MAGNUS_B + temperature_c))
        .exp()
}

/// Heat index via the Rothfusz regression (simplified).
/// Takes temperature in °C and relative_humidity as percentage (0–100).
#[must_use]
pub fn heat_index(temperature_c: f64, relative_humidity: f64) -> f64 {
    let t = temperature_c;
    let rh = relative_humidity;
    let t2 = t * t;
    let rh2 = rh * rh;

    -8.785 + 1.611 * t + 2.339 * rh - 0.1461 * t * rh - 0.012_31 * t2 - 0.016_42 * rh2
        + 0.002_212 * t2 * rh
        + 0.000_725 * t * rh2
        - 0.000_003_582 * t2 * rh2
}

/// Absolute humidity in g/m³.
/// AH = (6.112 × e^(17.67T/(T+243.5)) × RH × 2.1674) / (273.15 + T)
/// `relative_humidity` is fractional (0.0–1.0).
#[must_use]
pub fn absolute_humidity(relative_humidity: f64, temperature_c: f64) -> f64 {
    let t = temperature_c;
    assert!((t + 243.5) != 0.0, "temperature_c must not equal -243.5");
    assert!((273.15 + t) != 0.0, "temperature_c must not equal -273.15");
    let saturation = 6.112 * (17.67 * t / (t + 243.5)).exp();
    (saturation * relative_humidity * 2.1674) / (273.15 + t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-3;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // ── Wind Dynamics ──────────────────────────────────────────

    #[test]
    fn test_wind_chill() {
        // -10°C, 30 km/h => well-known result around -20°C
        let wc = wind_chill(-10.0, 30.0);
        assert!(
            approx(wc, -19.52, 0.1),
            "wind chill at -10°C, 30km/h was {wc}"
        );
    }

    #[test]
    fn test_wind_chill_zero_temp() {
        let wc = wind_chill(0.0, 10.0);
        // 13.12 + 0 - 11.37*(10^0.16) + 0 = 13.12 - 11.37*1.4454 ≈ -3.31
        assert!(approx(wc, -3.305, 0.1), "wind chill at 0°C, 10km/h was {wc}");
    }

    #[test]
    fn test_beaufort_to_speed() {
        // Beaufort 0 => 0 m/s
        assert!(approx(beaufort_to_speed(0), 0.0, EPSILON));
        // Beaufort 1 => 0.836 m/s
        assert!(approx(beaufort_to_speed(1), 0.836, EPSILON));
        // Beaufort 4 => 0.836 * 8.0 = 6.688 m/s
        assert!(approx(beaufort_to_speed(4), 0.836 * 8.0, EPSILON));
    }

    #[test]
    fn test_speed_to_beaufort() {
        assert_eq!(speed_to_beaufort(0.0), 0);
        assert_eq!(speed_to_beaufort(0.836), 1);
        assert_eq!(speed_to_beaufort(100.0), MAX_BEAUFORT);
    }

    #[test]
    fn test_beaufort_roundtrip() {
        for b in 0..=MAX_BEAUFORT {
            let speed = beaufort_to_speed(b);
            assert_eq!(speed_to_beaufort(speed), b, "roundtrip failed for B={b}");
        }
    }

    #[test]
    fn test_wind_shear() {
        assert!(approx(wind_shear(20.0, 10.0, 500.0), 0.02, EPSILON));
    }

    #[test]
    fn test_wind_power_density() {
        // ½ × 1.225 × 10³ = 612.5
        assert!(approx(wind_power_density(1.225, 10.0), 612.5, EPSILON));
    }

    // ── Coriolis Effect ────────────────────────────────────────

    #[test]
    fn test_coriolis_parameter_equator() {
        assert!(approx(coriolis_parameter(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_coriolis_parameter_45deg() {
        let lat = constants::PI / 4.0;
        let f = coriolis_parameter(lat);
        assert!(approx(f, 1.031_26e-4, 1e-8));
    }

    #[test]
    fn test_coriolis_acceleration() {
        let lat = constants::PI / 4.0;
        let a = coriolis_acceleration(10.0, lat);
        assert!(approx(a, 1.031_26e-3, 1e-7));
    }

    // ── Atmospheric Structure ──────────────────────────────────

    #[test]
    fn test_barometric_pressure_sea_level() {
        let p = barometric_pressure(
            STANDARD_PRESSURE,
            DRY_AIR_MOLAR_MASS,
            constants::G_ACCEL,
            0.0,
            STANDARD_TEMPERATURE,
        );
        assert!(approx(p, STANDARD_PRESSURE, EPSILON));
    }

    #[test]
    fn test_barometric_pressure_at_altitude() {
        let p = barometric_pressure(
            STANDARD_PRESSURE,
            DRY_AIR_MOLAR_MASS,
            constants::G_ACCEL,
            1000.0,
            STANDARD_TEMPERATURE,
        );
        // ~89,875 Pa at 1 km
        assert!(p < STANDARD_PRESSURE);
        assert!(approx(p, 89_994.0, 50.0));
    }

    #[test]
    fn test_dry_adiabatic_lapse_rate() {
        let lr = dry_adiabatic_lapse_rate(constants::G_ACCEL, 1005.0);
        // ~0.00976 K/m ≈ 9.76 K/km
        assert!(approx(lr, 0.009_76, 0.001));
    }

    #[test]
    fn test_temperature_at_altitude() {
        let t = temperature_at_altitude(288.15, 0.0065, 1000.0);
        assert!(approx(t, 281.65, EPSILON));
    }

    #[test]
    fn test_pressure_altitude() {
        let pa = pressure_altitude(STANDARD_PRESSURE, 90_000.0, 0.0065, STANDARD_TEMPERATURE);
        assert!(pa > 0.0);
        // Roughly ~1000 m for 90 kPa
        assert!(approx(pa, 988.0, 50.0));
    }

    #[test]
    fn test_density_altitude() {
        // PA=1000m, T=25°C, std=15°C => DA = 1000 + 36.576*10 = 1365.76
        let da = density_altitude(1000.0, 25.0, 15.0);
        assert!(approx(da, 1365.76, EPSILON));
    }

    #[test]
    fn test_density_altitude_standard_conditions() {
        let da = density_altitude(1000.0, 15.0, 15.0);
        assert!(approx(da, 1000.0, EPSILON));
    }

    #[test]
    fn test_scale_height() {
        let h = scale_height(STANDARD_TEMPERATURE, DRY_AIR_MOLAR_MASS, constants::G_ACCEL);
        // ~8435 m for standard atmosphere
        assert!(approx(h, 8435.0, 50.0));
    }

    // ── Humidity ───────────────────────────────────────────────

    #[test]
    fn test_dew_point() {
        // 25°C, 50% RH => Td ≈ 13.9°C
        let td = dew_point(25.0, 0.50);
        assert!(approx(td, 13.9, 0.5), "dew point was {td}");
    }

    #[test]
    fn test_relative_humidity() {
        // 25°C, Td=25°C => RH = 1.0
        let rh = relative_humidity(25.0, 25.0);
        assert!(approx(rh, 1.0, EPSILON));
    }

    #[test]
    fn test_humidity_roundtrip() {
        let t = 30.0;
        let rh_in = 0.65;
        let td = dew_point(t, rh_in);
        let rh_out = relative_humidity(t, td);
        assert!(
            approx(rh_out, rh_in, 0.001),
            "humidity roundtrip failed: {rh_out} vs {rh_in}"
        );
    }

    #[test]
    fn test_heat_index() {
        // 33°C, 70% RH => roughly 41°C heat index
        let hi = heat_index(33.0, 70.0);
        assert!(approx(hi, 43.5, 1.0), "heat index was {hi}");
    }

    #[test]
    fn test_absolute_humidity() {
        // 20°C, 50% RH => ~8.65 g/m³
        let ah = absolute_humidity(0.50, 20.0);
        assert!(approx(ah, 0.0864, 0.01), "absolute humidity was {ah}");
    }
}
