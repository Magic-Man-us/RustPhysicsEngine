pub const SOLAR_LUMINOSITY: f64 = 3.828e26;
pub const SOLAR_TEMPERATURE: f64 = 5778.0;
pub const HZ_INNER_COEFFICIENT: f64 = 0.95;
pub const HZ_OUTER_COEFFICIENT: f64 = 1.37;

pub fn habitable_zone_inner(luminosity_solar: f64) -> f64 {
    luminosity_solar.sqrt() * HZ_INNER_COEFFICIENT
}

pub fn habitable_zone_outer(luminosity_solar: f64) -> f64 {
    luminosity_solar.sqrt() * HZ_OUTER_COEFFICIENT
}

pub fn habitable_zone(luminosity_solar: f64) -> (f64, f64) {
    (
        habitable_zone_inner(luminosity_solar),
        habitable_zone_outer(luminosity_solar),
    )
}

pub fn is_in_habitable_zone(luminosity_solar: f64, distance_au: f64) -> bool {
    let (inner, outer) = habitable_zone(luminosity_solar);
    distance_au >= inner && distance_au <= outer
}

pub fn luminosity_from_mass(mass_solar: f64) -> f64 {
    mass_solar.powf(3.5)
}

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
}
