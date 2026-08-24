use crate::math::constants;

// Wien's displacement constant in the frequency domain: f_max = WIEN_FREQ_CONSTANT * T
const WIEN_FREQ_CONSTANT: f64 = 5.879e10;

// Wien's displacement constant in the wavelength domain: λ_max * T = b
const WIEN_DISPLACEMENT_CONSTANT: f64 = 2.898e-3;

// ── Blackbody & Thermal Emission ──

/// Total emissive power of a perfect blackbody (ε=1): E = σT⁴
pub fn total_emissive_power(temperature: f64) -> f64 {
    constants::SIGMA * temperature.powi(4)
}

/// Wien's law in the frequency domain: f_max = 5.879×10¹⁰ × T
pub fn spectral_peak_frequency(temperature: f64) -> f64 {
    WIEN_FREQ_CONSTANT * temperature
}

/// Inverse Wien's law: T = b / λ_max (color temperature from peak wavelength)
pub fn color_temperature(peak_wavelength: f64) -> f64 {
    assert!(peak_wavelength > 0.0, "peak_wavelength must be positive");
    WIEN_DISPLACEMENT_CONSTANT / peak_wavelength
}

/// Brightness temperature via the Rayleigh-Jeans approximation: T_b = Ic² / (2kf²)
pub fn brightness_temperature(intensity: f64, frequency: f64) -> f64 {
    assert!(frequency > 0.0, "frequency must be positive");
    intensity * constants::C * constants::C / (2.0 * constants::K_B * frequency * frequency)
}

// ── Radiation Transport ──

/// Optical depth: τ = κ × s
pub fn optical_depth(absorption_coeff: f64, path_length: f64) -> f64 {
    absorption_coeff * path_length
}

/// Beer-Lambert law: I = I₀ × e^(-κs)
pub fn beer_lambert(initial_intensity: f64, absorption_coeff: f64, path_length: f64) -> f64 {
    initial_intensity * (-absorption_coeff * path_length).exp()
}

/// Photon mean free path: l = 1/κ
pub fn mean_free_path_photon(absorption_coeff: f64) -> f64 {
    assert!(absorption_coeff > 0.0, "absorption_coeff must be positive");
    1.0 / absorption_coeff
}

/// Radiation pressure for fully absorbed radiation: P = I/c
pub fn radiation_pressure(intensity: f64) -> f64 {
    intensity / constants::C
}

/// Radiation pressure for fully reflected radiation: P = 2I/c
pub fn radiation_pressure_reflected(intensity: f64) -> f64 {
    2.0 * intensity / constants::C
}

// ── Kirchhoff's Law ──

/// At thermal equilibrium, emissivity equals absorptivity: ε = α
pub fn emissivity_from_absorptivity(absorptivity: f64) -> f64 {
    absorptivity
}

// ── View Factors (Radiative Exchange) ──

/// View factor for two identical, directly opposed, parallel rectangles of
/// width W and height H separated by distance D.
///
/// Uses the exact analytical formula:
/// F = (2 / (πXY)) * [ ln(√((1+X²)(1+Y²)/(1+X²+Y²)))
///                     + X√(1+Y²) atan(X/√(1+Y²))
///                     + Y√(1+X²) atan(Y/√(1+X²))
///                     - X atan(X) - Y atan(Y) ]
/// where X = W/D and Y = H/D.
pub fn view_factor_parallel_plates(width: f64, height: f64, separation: f64) -> f64 {
    assert!(separation > 0.0, "separation must be positive");
    let x = width / separation;
    let y = height / separation;
    let x2 = x * x;

    let a = 1.0 + x2;
    let b = 1.0 + y * y;
    let c = 1.0 + x2 + y * y;

    let term_ln = (a * b / c).sqrt().ln();
    let term_x = x * b.sqrt() * (x / b.sqrt()).atan() - x * x.atan();
    let term_y = y * a.sqrt() * (y / a.sqrt()).atan() - y * y.atan();

    let factor = 2.0 / (constants::PI * x * y) * (term_ln + term_x + term_y);
    factor.clamp(0.0, 1.0)
}

/// Radiative heat exchange between two infinite parallel gray surfaces:
/// Q = σA(T₁⁴ - T₂⁴) / (1/ε₁ + 1/ε₂ - 1)
pub fn radiative_exchange(
    emissivity1: f64,
    emissivity2: f64,
    area: f64,
    t1: f64,
    t2: f64,
) -> f64 {
    assert!(emissivity1 > 0.0, "emissivity1 must be positive");
    assert!(emissivity2 > 0.0, "emissivity2 must be positive");
    let resistance = 1.0 / emissivity1 + 1.0 / emissivity2 - 1.0;
    constants::SIGMA * area * (t1.powi(4) - t2.powi(4)) / resistance
}

// ── Inverse Square Law ──

/// Intensity at distance from a point source: I = L / (4πd²)
pub fn intensity_at_distance(luminosity: f64, distance: f64) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    luminosity / (4.0 * constants::PI * distance * distance)
}

/// Luminosity from measured intensity and distance: L = I × 4πd²
pub fn luminosity_from_intensity(intensity: f64, distance: f64) -> f64 {
    intensity * 4.0 * constants::PI * distance * distance
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b == 0.0 {
            return a.abs() < tol;
        }
        ((a - b) / b).abs() < tol
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── Blackbody & Thermal Emission ──

    #[test]
    fn test_total_emissive_power_sun() {
        // Sun surface ~5778 K → σT⁴ ≈ 6.32×10⁷ W/m²
        let e = total_emissive_power(5778.0);
        assert!(approx_rel(e, 6.32e7, 0.02));
    }

    #[test]
    fn test_total_emissive_power_room_temp() {
        // 300 K → σ(300)⁴ ≈ 459 W/m²
        let e = total_emissive_power(300.0);
        assert!(approx_rel(e, 459.3, 0.02));
    }

    #[test]
    fn test_spectral_peak_frequency_sun() {
        // 5778 K → f_max ≈ 3.40×10¹⁴ Hz
        let f = spectral_peak_frequency(5778.0);
        assert!(approx_rel(f, 3.397e14, 0.01));
    }

    #[test]
    fn test_color_temperature_sun() {
        // Peak at 502 nm → T ≈ 5773 K
        let t = color_temperature(502e-9);
        assert!(approx_rel(t, 5773.0, 0.01));
    }

    #[test]
    fn test_color_temperature_inverse_of_wien() {
        // Round-trip: wien_peak → color_temperature should recover T
        let temp = 3000.0;
        let peak = WIEN_DISPLACEMENT_CONSTANT / temp;
        let recovered = color_temperature(peak);
        assert!(approx_rel(recovered, temp, 1e-9));
    }

    #[test]
    fn test_brightness_temperature() {
        // For a source with known intensity at a given frequency,
        // verify T_b = Ic²/(2kf²)
        let freq = 1e9; // 1 GHz (radio)
        let intensity = 2.0 * constants::K_B * 5000.0 * freq * freq / (constants::C * constants::C);
        let tb = brightness_temperature(intensity, freq);
        assert!(approx_rel(tb, 5000.0, 1e-6));
    }

    // ── Radiation Transport ──

    #[test]
    fn test_optical_depth() {
        let tau = optical_depth(0.5, 10.0);
        assert!(approx(tau, 5.0, 1e-9));
    }

    #[test]
    fn test_beer_lambert_zero_depth() {
        let i = beer_lambert(100.0, 0.0, 10.0);
        assert!(approx(i, 100.0, 1e-9));
    }

    #[test]
    fn test_beer_lambert_attenuation() {
        // After one mean free path (κ=1, s=1), intensity drops to I₀/e
        let i = beer_lambert(1000.0, 1.0, 1.0);
        assert!(approx_rel(i, 1000.0 / std::f64::consts::E, 1e-6));
    }

    #[test]
    fn test_beer_lambert_thick_medium() {
        // Very large optical depth should give near-zero intensity
        let i = beer_lambert(1000.0, 10.0, 10.0);
        assert!(i < 1e-30);
    }

    #[test]
    fn test_mean_free_path_photon() {
        let l = mean_free_path_photon(0.25);
        assert!(approx(l, 4.0, 1e-9));
    }

    #[test]
    fn test_radiation_pressure_absorbed() {
        // Solar constant ~1361 W/m² → P ≈ 4.54 μPa
        let p = radiation_pressure(1361.0);
        assert!(approx_rel(p, 4.54e-6, 0.02));
    }

    #[test]
    fn test_radiation_pressure_reflected() {
        let p_abs = radiation_pressure(1361.0);
        let p_ref = radiation_pressure_reflected(1361.0);
        assert!(approx_rel(p_ref, 2.0 * p_abs, 1e-9));
    }

    // ── Kirchhoff's Law ──

    #[test]
    fn test_emissivity_from_absorptivity() {
        assert!(approx(emissivity_from_absorptivity(0.75), 0.75, 1e-15));
    }

    // ── View Factors ──

    #[test]
    fn test_view_factor_parallel_plates_bounded() {
        let f = view_factor_parallel_plates(1.0, 1.0, 1.0);
        assert!(f > 0.0 && f < 1.0);
    }

    #[test]
    fn test_view_factor_parallel_plates_large_separation() {
        // Very far apart → view factor near zero
        let f = view_factor_parallel_plates(1.0, 1.0, 1000.0);
        assert!(f < 0.01);
    }

    #[test]
    fn test_view_factor_parallel_plates_known_value() {
        // Two 1×1 squares separated by distance 1: known F ≈ 0.1998
        let f = view_factor_parallel_plates(1.0, 1.0, 1.0);
        assert!(approx_rel(f, 0.1998, 0.02));
    }

    #[test]
    fn test_radiative_exchange_blackbodies() {
        // Two blackbodies (ε=1): Q = σA(T₁⁴-T₂⁴)
        let q = radiative_exchange(1.0, 1.0, 1.0, 500.0, 300.0);
        let expected = 3084.683683936;
        assert!(approx_rel(q, expected, 1e-9));
    }

    #[test]
    fn test_radiative_exchange_gray_surfaces() {
        // ε₁=0.5, ε₂=0.8 → resistance = 2.0 + 1.25 - 1.0 = 2.25
        let q = radiative_exchange(0.5, 0.8, 2.0, 600.0, 300.0);
        let expected = 6124.00437252;
        assert!(approx_rel(q, expected, 1e-9));
    }

    #[test]
    fn test_radiative_exchange_equal_temps() {
        // Same temperature → zero net exchange
        let q = radiative_exchange(0.9, 0.9, 5.0, 400.0, 400.0);
        assert!(approx(q, 0.0, 1e-9));
    }

    // ── Inverse Square Law ──

    #[test]
    fn test_intensity_at_distance_sun() {
        // Sun luminosity ~3.828×10²⁶ W at 1 AU (1.496×10¹¹ m) → ~1361 W/m²
        let i = intensity_at_distance(3.828e26, 1.496e11);
        assert!(approx_rel(i, 1361.0, 0.02));
    }

    #[test]
    fn test_luminosity_from_intensity_roundtrip() {
        let luminosity = 1e26;
        let distance = 1e10;
        let i = intensity_at_distance(luminosity, distance);
        let recovered = luminosity_from_intensity(i, distance);
        assert!(approx_rel(recovered, luminosity, 1e-9));
    }

    #[test]
    fn test_inverse_square_law_double_distance() {
        // Doubling distance quarters intensity
        let i1 = intensity_at_distance(1e26, 1e10);
        let i2 = intensity_at_distance(1e26, 2e10);
        assert!(approx_rel(i2, i1 / 4.0, 1e-9));
    }

    #[test]
    fn test_approx_rel_zero_b() {
        assert!(approx_rel(0.0, 0.0, 1e-6));
        assert!(!approx_rel(1.0, 0.0, 0.5));
    }
}
