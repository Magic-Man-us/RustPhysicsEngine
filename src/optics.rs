use crate::math::constants;

// ── Reflection and Refraction ──

/// Snell's law: n1 * sin(θ1) = n2 * sin(θ2) → θ2 = asin(n1 * sin(θ1) / n2)
/// Returns the refraction angle in radians, or None for total internal reflection.
pub fn snells_law(n1: f64, angle1_rad: f64, n2: f64) -> Option<f64> {
    let sin_theta2 = n1 * angle1_rad.sin() / n2;
    if sin_theta2.abs() > 1.0 {
        None // Total internal reflection
    } else {
        Some(sin_theta2.asin())
    }
}

/// Critical angle for total internal reflection: θ_c = asin(n2 / n1)
/// Only valid when n1 > n2. Returns None otherwise.
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 <= n2 {
        None
    } else {
        Some((n2 / n1).asin())
    }
}

/// Brewster's angle: θ_B = atan(n2 / n1)
pub fn brewster_angle(n1: f64, n2: f64) -> f64 {
    (n2 / n1).atan()
}

/// Index of refraction: n = c / v
pub fn refractive_index(speed_in_medium: f64) -> f64 {
    constants::C / speed_in_medium
}

/// Speed of light in a medium: v = c / n
pub fn speed_in_medium(refractive_index: f64) -> f64 {
    constants::C / refractive_index
}

// ── Mirrors and Lenses ──

/// Mirror/thin lens equation: 1/f = 1/d_o + 1/d_i → d_i = f*d_o / (d_o - f)
pub fn image_distance(focal_length: f64, object_distance: f64) -> f64 {
    focal_length * object_distance / (object_distance - focal_length)
}

/// Magnification: m = -d_i / d_o
pub fn magnification(image_distance: f64, object_distance: f64) -> f64 {
    -image_distance / object_distance
}

/// Magnification from image and object heights: m = h_i / h_o
pub fn magnification_from_heights(image_height: f64, object_height: f64) -> f64 {
    image_height / object_height
}

/// Lens maker's equation: 1/f = (n-1) * (1/R1 - 1/R2)
pub fn lens_focal_length(n: f64, r1: f64, r2: f64) -> f64 {
    1.0 / ((n - 1.0) * (1.0 / r1 - 1.0 / r2))
}

/// Power of a lens: P = 1/f (in diopters when f is in meters)
pub fn lens_power(focal_length: f64) -> f64 {
    1.0 / focal_length
}

/// Combined focal length of two thin lenses in contact: 1/f = 1/f1 + 1/f2
pub fn combined_focal_length(f1: f64, f2: f64) -> f64 {
    1.0 / (1.0 / f1 + 1.0 / f2)
}

/// Mirror radius of curvature: R = 2f
pub fn radius_of_curvature(focal_length: f64) -> f64 {
    2.0 * focal_length
}

// ── Diffraction ──

/// Single slit diffraction minima: a * sin(θ) = m * λ → θ = asin(m * λ / a)
/// Returns angle in radians for the m-th minimum.
pub fn single_slit_minimum(order: i32, wavelength: f64, slit_width: f64) -> Option<f64> {
    let sin_theta = order as f64 * wavelength / slit_width;
    if sin_theta.abs() > 1.0 {
        None
    } else {
        Some(sin_theta.asin())
    }
}

/// Double slit maxima: d * sin(θ) = m * λ → θ = asin(m * λ / d)
pub fn double_slit_maximum(order: i32, wavelength: f64, slit_separation: f64) -> Option<f64> {
    let sin_theta = order as f64 * wavelength / slit_separation;
    if sin_theta.abs() > 1.0 {
        None
    } else {
        Some(sin_theta.asin())
    }
}

/// Diffraction grating: d * sin(θ) = m * λ
pub fn diffraction_grating_angle(
    order: i32,
    wavelength: f64,
    grating_spacing: f64,
) -> Option<f64> {
    let sin_theta = order as f64 * wavelength / grating_spacing;
    if sin_theta.abs() > 1.0 {
        None
    } else {
        Some(sin_theta.asin())
    }
}

/// Rayleigh criterion (angular resolution): θ = 1.22 * λ / D
pub fn rayleigh_resolution(wavelength: f64, aperture_diameter: f64) -> f64 {
    1.22 * wavelength / aperture_diameter
}

// ── Interference ──

/// Thin film interference (constructive, normal incidence):
/// 2 * n * t = (m + 0.5) * λ for reflection with one phase change
pub fn thin_film_constructive_thickness(order: u32, wavelength: f64, film_index: f64) -> f64 {
    (order as f64 + 0.5) * wavelength / (2.0 * film_index)
}

/// Path difference for constructive interference: Δ = m * λ
pub fn constructive_path_diff(order: i32, wavelength: f64) -> f64 {
    order as f64 * wavelength
}

/// Path difference for destructive interference: Δ = (m + 0.5) * λ
pub fn destructive_path_diff(order: i32, wavelength: f64) -> f64 {
    (order as f64 + 0.5) * wavelength
}

// ── Polarization ──

/// Malus's law: I = I_0 * cos^2(θ)
pub fn malus_law(initial_intensity: f64, angle_rad: f64) -> f64 {
    initial_intensity * angle_rad.cos().powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_snells_law() {
        // Air to glass (n=1.5), 30 degrees
        let angle = snells_law(1.0, 30.0_f64.to_radians(), 1.5).unwrap();
        assert!(approx(angle.to_degrees(), 19.47, 0.1));
    }

    #[test]
    fn test_total_internal_reflection() {
        // Glass to air at steep angle
        let result = snells_law(1.5, 45.0_f64.to_radians(), 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_critical_angle() {
        let angle = critical_angle(1.5, 1.0).unwrap();
        assert!(approx(angle.to_degrees(), 41.81, 0.1));
    }

    #[test]
    fn test_image_distance_converging() {
        // Object at 30cm from lens with f=20cm
        let di = image_distance(20.0, 30.0);
        assert!(approx(di, 60.0, 0.01));
    }

    #[test]
    fn test_magnification() {
        let m = magnification(60.0, 30.0);
        assert!(approx(m, -2.0, 1e-9));
    }

    #[test]
    fn test_lens_power() {
        assert!(approx(lens_power(0.5), 2.0, 1e-9));
    }

    #[test]
    fn test_rayleigh() {
        let theta = rayleigh_resolution(550e-9, 0.1);
        assert!(approx(theta, 6.71e-6, 1e-7));
    }

    #[test]
    fn test_malus_law() {
        let i = malus_law(100.0, 0.0);
        assert!(approx(i, 100.0, 1e-9));
        let i2 = malus_law(100.0, constants::PI / 2.0);
        assert!(approx(i2, 0.0, 1e-9));
    }

    #[test]
    fn test_brewster_angle() {
        let b = brewster_angle(1.0, 1.5);
        assert!(approx(b.to_degrees(), 56.31, 0.1));
    }
}
