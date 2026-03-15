use crate::math::Vec3;
use crate::math::constants::{G, C, PI};

pub fn gw_luminosity(m1: f64, m2: f64, separation: f64) -> f64 {
    if separation <= 0.0 {
        return 0.0;
    }
    let c5 = C.powi(5);
    let g4 = G.powi(4);
    let m_product_sq = (m1 * m2) * (m1 * m2);
    let m_total = m1 + m2;
    (32.0 / 5.0) * g4 * m_product_sq * m_total / (c5 * separation.powi(5))
}

pub fn gw_frequency(m1: f64, m2: f64, separation: f64) -> f64 {
    if separation <= 0.0 {
        return 0.0;
    }
    let mu = G * (m1 + m2);
    (1.0 / PI) * (mu / separation.powi(3)).sqrt()
}

pub fn gw_strain(m1: f64, m2: f64, separation: f64, distance_to_observer: f64) -> f64 {
    if separation <= 0.0 || distance_to_observer <= 0.0 {
        return 0.0;
    }
    let c4 = C.powi(4);
    (4.0 * G * G * m1 * m2) / (c4 * separation * distance_to_observer)
}

pub fn inspiral_time(m1: f64, m2: f64, separation: f64) -> f64 {
    if m1 <= 0.0 || m2 <= 0.0 {
        return f64::INFINITY;
    }
    let c5 = C.powi(5);
    let g3 = G.powi(3);
    (5.0 / 256.0) * c5 * separation.powi(4) / (g3 * m1 * m2 * (m1 + m2))
}

pub fn chirp_mass(m1: f64, m2: f64) -> f64 {
    let product = m1 * m2;
    let total = m1 + m2;
    if total <= 0.0 {
        return 0.0;
    }
    product.powf(3.0 / 5.0) / total.powf(1.0 / 5.0)
}

pub fn innermost_stable_circular_orbit(total_mass: f64) -> f64 {
    6.0 * G * total_mass / (C * C)
}

pub fn find_strongest_source(masses: &[f64], positions: &[Vec3]) -> Option<(usize, usize, f64)> {
    if masses.len() < 2 {
        return None;
    }

    let mut best: Option<(usize, usize, f64)> = None;

    for i in 0..masses.len() {
        for j in (i + 1)..masses.len() {
            let d = positions[i].distance_to(&positions[j]);
            if d < 1e-20 {
                continue;
            }
            let lum = gw_luminosity(masses[i], masses[j], d);
            match best {
                Some((_, _, best_lum)) if lum <= best_lum => {}
                _ => best = Some((i, j, lum)),
            }
        }
    }

    best
}

pub fn merger_energy(m1: f64, m2: f64) -> f64 {
    let total = m1 + m2;
    if total <= 0.0 {
        return 0.0;
    }
    let eta = m1 * m2 / (total * total);
    eta * total * C * C
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chirp_mass_equal() {
        let mc = chirp_mass(10.0, 10.0);
        // For equal masses: Mc = M / 2^(1/5) * 2^(3/5) = M * 2^(2/5)
        // = m * 2^(2/5) where m = individual mass... let's just check > 0
        assert!(mc > 0.0);
        // Mc = (m*m)^(3/5) / (2m)^(1/5) = m^(6/5) / (2m)^(1/5) = m * (m/2)^(1/5)
        // For m=10: Mc = 10 * (5)^(1/5) ≈ 10 * 1.38 = ~13.8? No...
        // Mc = (100)^0.6 / (20)^0.2 = 15.85 / 1.82 = 8.71
        let expected = 100.0_f64.powf(0.6) / 20.0_f64.powf(0.2);
        assert!((mc - expected).abs() < 1e-6, "chirp mass = {mc}, expected {expected}");
    }

    #[test]
    fn test_gw_luminosity_closer_is_brighter() {
        let l1 = gw_luminosity(1.989e30, 1.989e30, 1e9);
        let l2 = gw_luminosity(1.989e30, 1.989e30, 2e9);
        assert!(l1 > l2, "Closer binary should radiate more");
    }

    #[test]
    fn test_inspiral_time_positive() {
        let t = inspiral_time(1.989e30, 1.989e30, 1e9);
        assert!(t > 0.0);
    }

    #[test]
    fn test_isco_schwarzschild() {
        let m = 1.989e30; // solar mass
        let r = innermost_stable_circular_orbit(m);
        let rs = 2.0 * G * m / (C * C);
        assert!((r - 3.0 * rs).abs() / rs < 1e-6, "ISCO should be 3×Schwarzschild radius");
    }

    #[test]
    fn test_find_strongest_source() {
        let masses = vec![1.989e30, 1.989e30, 1.0e20];
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e9, 0.0, 0.0),
            Vec3::new(1e12, 0.0, 0.0),
        ];
        let result = find_strongest_source(&masses, &positions);
        assert!(result.is_some());
        let (i, j, _) = result.unwrap();
        assert_eq!(i, 0);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_merger_energy_positive() {
        let e = merger_energy(1.989e30, 1.989e30);
        assert!(e > 0.0);
    }
}
