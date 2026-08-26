//! Gravitational radiation from a compact binary.
//!
//! Quadrupole-formula results for an inspiralling binary: the emitted
//! luminosity, the wave frequency (twice the orbital frequency), the
//! strain amplitude at a given distance, and the time remaining to merger.
//!
//! The chirp mass `ℳ = (m₁m₂)^(3/5)/(m₁+m₂)^(1/5)` is the combination
//! that governs all of them -- it is the parameter the inspiral waveform
//! actually determines, which is why it is measured far better than either
//! individual mass.

use crate::math::Vec3;
use crate::math::constants::{G, C, PI};

/// Computes gravitational wave luminosity for a circular binary: L = (32/5) G⁴m₁²m₂²(m₁+m₂) / (c⁵ a⁵).
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

/// Computes the gravitational wave frequency (twice the orbital frequency): f_gw = (1/π)√(G(m₁+m₂)/a³).
pub fn gw_frequency(m1: f64, m2: f64, separation: f64) -> f64 {
    if separation <= 0.0 {
        return 0.0;
    }
    let mu = G * (m1 + m2);
    (1.0 / PI) * (mu / separation.powi(3)).sqrt()
}

/// Computes the dimensionless gravitational wave strain amplitude: h = 4G²m₁m₂ / (c⁴ a D).
pub fn gw_strain(m1: f64, m2: f64, separation: f64, distance_to_observer: f64) -> f64 {
    if separation <= 0.0 || distance_to_observer <= 0.0 {
        return 0.0;
    }
    let c4 = C.powi(4);
    (4.0 * G * G * m1 * m2) / (c4 * separation * distance_to_observer)
}

/// Computes the Peters inspiral time for a circular binary: t = (5/256) c⁵ a⁴ / (G³ m₁ m₂ (m₁+m₂)).
pub fn inspiral_time(m1: f64, m2: f64, separation: f64) -> f64 {
    if m1 <= 0.0 || m2 <= 0.0 {
        return f64::INFINITY;
    }
    let c5 = C.powi(5);
    let g3 = G.powi(3);
    (5.0 / 256.0) * c5 * separation.powi(4) / (g3 * m1 * m2 * (m1 + m2))
}

/// Computes the chirp mass of a binary system: M_c = (m₁ m₂)^(3/5) / (m₁ + m₂)^(1/5).
pub fn chirp_mass(m1: f64, m2: f64) -> f64 {
    let product = m1 * m2;
    let total = m1 + m2;
    if total <= 0.0 {
        return 0.0;
    }
    product.powf(3.0 / 5.0) / total.powf(1.0 / 5.0)
}

/// Computes the Schwarzschild ISCO radius: r_isco = 6GM/c².
pub fn innermost_stable_circular_orbit(total_mass: f64) -> f64 {
    6.0 * G * total_mass / (C * C)
}

/// Finds the pair of bodies with the highest gravitational wave luminosity, returning their indices and luminosity.
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

/// Estimates the energy radiated during merger: E = η M c², where η = m₁m₂/(m₁+m₂)².
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
        assert!((mc - 8.7055).abs() < 1e-3, "chirp mass = {mc}, expected 8.7055");
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
        let m = 1.989e30;
        let r = innermost_stable_circular_orbit(m);
        assert!((r - 8862.38).abs() < 1.0, "ISCO should be ~8862 m, got {r}");
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

    #[test]
    fn test_gw_frequency_positive() {
        let f = gw_frequency(1.989e30, 1.989e30, 1e9);
        assert!(f > 0.0, "GW frequency should be positive, got {f}");
    }

    #[test]
    fn test_gw_frequency_closer_is_higher() {
        let f1 = gw_frequency(1.989e30, 1.989e30, 1e9);
        let f2 = gw_frequency(1.989e30, 1.989e30, 2e9);
        assert!(f1 > f2, "Closer binary should have higher GW frequency");
    }

    #[test]
    fn test_gw_frequency_zero_separation() {
        let f = gw_frequency(1.989e30, 1.989e30, 0.0);
        assert!((f - 0.0).abs() < 1e-20, "Zero separation should return 0");
    }

    #[test]
    fn test_gw_strain_positive() {
        let h = gw_strain(1.989e30, 1.989e30, 1e9, 1e22);
        assert!(h > 0.0, "Strain should be positive, got {h}");
    }

    #[test]
    fn test_gw_strain_decreases_with_distance() {
        let h1 = gw_strain(1.989e30, 1.989e30, 1e9, 1e22);
        let h2 = gw_strain(1.989e30, 1.989e30, 1e9, 2e22);
        assert!(h1 > h2, "Strain should decrease with observer distance");
    }

    #[test]
    fn test_gw_strain_zero_separation() {
        let h = gw_strain(1.989e30, 1.989e30, 0.0, 1e22);
        assert!((h - 0.0).abs() < 1e-20);
    }

    #[test]
    fn test_gw_luminosity_zero_separation() {
        let l = gw_luminosity(1.989e30, 1.989e30, 0.0);
        assert!((l - 0.0).abs() < 1e-20);
    }

    #[test]
    fn test_inspiral_time_zero_mass() {
        let t = inspiral_time(0.0, 1.989e30, 1e9);
        assert!(t.is_infinite());
    }

    #[test]
    fn test_chirp_mass_zero_total() {
        let mc = chirp_mass(0.0, 0.0);
        assert!((mc - 0.0).abs() < 1e-20);
    }

    #[test]
    fn test_find_strongest_source_single_body() {
        let masses = vec![1.989e30];
        let positions = vec![Vec3::new(0.0, 0.0, 0.0)];
        let result = find_strongest_source(&masses, &positions);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_strongest_source_coincident() {
        let masses = vec![1.989e30, 1.989e30];
        let positions = vec![Vec3::ZERO, Vec3::ZERO];
        let result = find_strongest_source(&masses, &positions);
        assert!(result.is_none());
    }

    #[test]
    fn test_merger_energy_zero_mass() {
        let e = merger_energy(0.0, 0.0);
        assert!((e - 0.0).abs() < 1e-20);
    }
}
