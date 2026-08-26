//! Habitable zone boundaries and tidal locking.
//!
//! Inner and outer edges scale as the square root of the stellar
//! luminosity, with the conventional coefficients: 0.95 AU and 1.37 AU per
//! square root of a solar luminosity.
//!
//! Also the mass-luminosity relation for main-sequence stars, equilibrium
//! temperature for a given albedo, and the tidal locking timescale --
//! which matters here because low-mass stars have close-in habitable
//! zones, so their habitable planets are likely to be locked.

use crate::math::constants;

/// Solar luminosity L☉ (W), re-exported from [`constants`].
pub const SOLAR_LUMINOSITY: f64 = constants::SOLAR_LUMINOSITY;
/// Solar effective temperature T☉ (K), re-exported from [`constants`].
pub const SOLAR_TEMPERATURE: f64 = constants::SOLAR_TEMPERATURE;
pub const HZ_INNER_COEFFICIENT: f64 = 0.95;
pub const HZ_OUTER_COEFFICIENT: f64 = 1.37;

/// Computes the inner edge of the habitable zone in AU: d_inner = √L × 0.95.
pub fn habitable_zone_inner(luminosity_solar: f64) -> f64 {
    luminosity_solar.sqrt() * HZ_INNER_COEFFICIENT
}

/// Computes the outer edge of the habitable zone in AU: d_outer = √L × 1.37.
pub fn habitable_zone_outer(luminosity_solar: f64) -> f64 {
    luminosity_solar.sqrt() * HZ_OUTER_COEFFICIENT
}

/// Returns the (inner, outer) habitable zone boundaries in AU for a given stellar luminosity in solar units.
pub fn habitable_zone(luminosity_solar: f64) -> (f64, f64) {
    (
        habitable_zone_inner(luminosity_solar),
        habitable_zone_outer(luminosity_solar),
    )
}

/// Returns true if a body at the given distance (AU) lies within the habitable zone.
pub fn is_in_habitable_zone(luminosity_solar: f64, distance_au: f64) -> bool {
    let (inner, outer) = habitable_zone(luminosity_solar);
    distance_au >= inner && distance_au <= outer
}

/// Estimates stellar luminosity from mass using the mass-luminosity relation: L = M^3.5 (in solar units).
pub fn luminosity_from_mass(mass_solar: f64) -> f64 {
    mass_solar.powf(3.5)
}

/// Computes stellar luminosity from the Stefan-Boltzmann law: L = R² (T/T_sun)⁴ (in solar units).
pub fn luminosity_from_temperature_radius(temperature: f64, radius_solar: f64) -> f64 {
    let temp_ratio = temperature / SOLAR_TEMPERATURE;
    radius_solar * radius_solar * temp_ratio.powi(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_solar_habitable_zone() {
        let (inner, outer) = habitable_zone(1.0);
        assert!(approx(inner, 0.95, 1e-6));
        assert!(approx(outer, 1.37, 1e-6));
    }

    #[test]
    fn test_earth_in_hz() {
        assert!(is_in_habitable_zone(1.0, 1.0));
    }

    #[test]
    fn test_mercury_not_in_hz() {
        assert!(!is_in_habitable_zone(1.0, 0.39));
    }

    #[test]
    fn test_luminosity_from_mass_sun() {
        let l = luminosity_from_mass(1.0);
        assert!(approx(l, 1.0, 1e-9));
    }

    #[test]
    fn test_luminosity_from_temp_radius_sun() {
        let l = luminosity_from_temperature_radius(SOLAR_TEMPERATURE, 1.0);
        assert!(approx(l, 1.0, 1e-6));
    }

    #[test]
    fn test_brighter_star_wider_hz() {
        let (_, outer_sun) = habitable_zone(1.0);
        let (_, outer_bright) = habitable_zone(4.0);
        assert!(outer_bright > outer_sun);
    }

    #[test]
    fn test_habitable_zone_inner_solar() {
        let inner = habitable_zone_inner(1.0);
        assert!(approx(inner, HZ_INNER_COEFFICIENT, 1e-9), "Inner HZ for solar luminosity should be {}, got {inner}", HZ_INNER_COEFFICIENT);
    }

    #[test]
    fn test_habitable_zone_outer_solar() {
        let outer = habitable_zone_outer(1.0);
        assert!(approx(outer, HZ_OUTER_COEFFICIENT, 1e-9), "Outer HZ for solar luminosity should be {}, got {outer}", HZ_OUTER_COEFFICIENT);
    }

    #[test]
    fn test_habitable_zone_inner_brighter_star() {
        let inner_sun = habitable_zone_inner(1.0);
        let inner_bright = habitable_zone_inner(4.0);
        assert!(inner_bright > inner_sun, "Brighter star should have farther inner HZ edge");
        assert!(approx(inner_bright, 2.0 * HZ_INNER_COEFFICIENT, 1e-9));
    }

    #[test]
    fn test_habitable_zone_outer_brighter_star() {
        let outer_sun = habitable_zone_outer(1.0);
        let outer_bright = habitable_zone_outer(4.0);
        assert!(outer_bright > outer_sun, "Brighter star should have farther outer HZ edge");
        assert!(approx(outer_bright, 2.0 * HZ_OUTER_COEFFICIENT, 1e-9));
    }
}
