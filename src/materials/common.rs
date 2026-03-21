/// An engineering material with mechanical and thermal properties.
///
/// All values use SI units:
/// - `density`: kg/m^3
/// - `youngs_modulus`, `yield_strength`, `tensile_strength`: Pa
/// - `thermal_conductivity`: W/(m*K)
/// - `specific_heat`: J/(kg*K)
/// - `thermal_expansion`: 1/K
/// - `melting_point`: K
#[derive(Debug, Clone)]
pub struct Material {
    pub name: &'static str,
    pub density: f64,
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub yield_strength: f64,
    pub tensile_strength: f64,
    pub thermal_conductivity: f64,
    pub specific_heat: f64,
    pub thermal_expansion: f64,
    pub melting_point: f64,
}

static MATERIALS: [Material; 20] = [
    // Mild Steel (AISI 1020)
    Material {
        name: "Steel",
        density: 7850.0,
        youngs_modulus: 200.0e9,
        poisson_ratio: 0.29,
        yield_strength: 250.0e6,
        tensile_strength: 400.0e6,
        thermal_conductivity: 51.9,
        specific_heat: 486.0,
        thermal_expansion: 11.7e-6,
        melting_point: 1793.0,
    },
    // Stainless Steel 304
    Material {
        name: "Stainless Steel 304",
        density: 8000.0,
        youngs_modulus: 193.0e9,
        poisson_ratio: 0.29,
        yield_strength: 215.0e6,
        tensile_strength: 505.0e6,
        thermal_conductivity: 16.2,
        specific_heat: 500.0,
        thermal_expansion: 17.3e-6,
        melting_point: 1723.0,
    },
    // Aluminum 6061-T6
    Material {
        name: "Aluminum 6061",
        density: 2700.0,
        youngs_modulus: 68.9e9,
        poisson_ratio: 0.33,
        yield_strength: 276.0e6,
        tensile_strength: 310.0e6,
        thermal_conductivity: 167.0,
        specific_heat: 896.0,
        thermal_expansion: 23.6e-6,
        melting_point: 855.0,
    },
    // Copper (pure, annealed)
    Material {
        name: "Copper",
        density: 8960.0,
        youngs_modulus: 110.0e9,
        poisson_ratio: 0.34,
        yield_strength: 33.3e6,
        tensile_strength: 210.0e6,
        thermal_conductivity: 401.0,
        specific_heat: 385.0,
        thermal_expansion: 16.5e-6,
        melting_point: 1357.77,
    },
    // Titanium Ti-6Al-4V
    Material {
        name: "Titanium Ti-6Al-4V",
        density: 4430.0,
        youngs_modulus: 113.8e9,
        poisson_ratio: 0.342,
        yield_strength: 880.0e6,
        tensile_strength: 950.0e6,
        thermal_conductivity: 6.7,
        specific_heat: 526.3,
        thermal_expansion: 8.6e-6,
        melting_point: 1933.0,
    },
    // Glass (soda-lime)
    Material {
        name: "Glass",
        density: 2500.0,
        youngs_modulus: 72.0e9,
        poisson_ratio: 0.22,
        yield_strength: 0.0,
        tensile_strength: 33.0e6,
        thermal_conductivity: 1.05,
        specific_heat: 840.0,
        thermal_expansion: 8.5e-6,
        melting_point: 1273.0,
    },
    // Concrete (typical)
    Material {
        name: "Concrete",
        density: 2400.0,
        youngs_modulus: 30.0e9,
        poisson_ratio: 0.20,
        yield_strength: 0.0,
        tensile_strength: 3.0e6,
        thermal_conductivity: 1.7,
        specific_heat: 880.0,
        thermal_expansion: 12.0e-6,
        melting_point: 1773.0,
    },
    // Wood (oak, along grain)
    Material {
        name: "Wood (Oak)",
        density: 750.0,
        youngs_modulus: 12.0e9,
        poisson_ratio: 0.35,
        yield_strength: 41.0e6,
        tensile_strength: 78.0e6,
        thermal_conductivity: 0.17,
        specific_heat: 2380.0,
        thermal_expansion: 5.4e-6,
        melting_point: 0.0,
    },
    // Natural Rubber
    Material {
        name: "Rubber",
        density: 920.0,
        youngs_modulus: 0.01e9,
        poisson_ratio: 0.4999,
        yield_strength: 0.0,
        tensile_strength: 25.0e6,
        thermal_conductivity: 0.13,
        specific_heat: 1880.0,
        thermal_expansion: 77.0e-6,
        melting_point: 0.0,
    },
    // Diamond
    Material {
        name: "Diamond",
        density: 3510.0,
        youngs_modulus: 1220.0e9,
        poisson_ratio: 0.07,
        yield_strength: 0.0,
        tensile_strength: 2800.0e6,
        thermal_conductivity: 2200.0,
        specific_heat: 509.0,
        thermal_expansion: 1.0e-6,
        melting_point: 3823.0,
    },
    // Tungsten
    Material {
        name: "Tungsten",
        density: 19250.0,
        youngs_modulus: 411.0e9,
        poisson_ratio: 0.28,
        yield_strength: 550.0e6,
        tensile_strength: 980.0e6,
        thermal_conductivity: 173.0,
        specific_heat: 132.0,
        thermal_expansion: 4.5e-6,
        melting_point: 3695.0,
    },
    // Brass (70/30)
    Material {
        name: "Brass",
        density: 8530.0,
        youngs_modulus: 110.0e9,
        poisson_ratio: 0.34,
        yield_strength: 75.0e6,
        tensile_strength: 325.0e6,
        thermal_conductivity: 120.0,
        specific_heat: 380.0,
        thermal_expansion: 19.0e-6,
        melting_point: 1203.0,
    },
    // Bronze (phosphor)
    Material {
        name: "Bronze",
        density: 8800.0,
        youngs_modulus: 110.0e9,
        poisson_ratio: 0.34,
        yield_strength: 130.0e6,
        tensile_strength: 350.0e6,
        thermal_conductivity: 50.0,
        specific_heat: 380.0,
        thermal_expansion: 17.8e-6,
        melting_point: 1273.0,
    },
    // Cast Iron (gray)
    Material {
        name: "Cast Iron",
        density: 7200.0,
        youngs_modulus: 100.0e9,
        poisson_ratio: 0.26,
        yield_strength: 0.0,
        tensile_strength: 200.0e6,
        thermal_conductivity: 52.0,
        specific_heat: 490.0,
        thermal_expansion: 10.5e-6,
        melting_point: 1473.0,
    },
    // Lead
    Material {
        name: "Lead",
        density: 11340.0,
        youngs_modulus: 16.0e9,
        poisson_ratio: 0.44,
        yield_strength: 8.0e6,
        tensile_strength: 18.0e6,
        thermal_conductivity: 35.3,
        specific_heat: 129.0,
        thermal_expansion: 28.9e-6,
        melting_point: 600.61,
    },
    // Gold
    Material {
        name: "Gold",
        density: 19300.0,
        youngs_modulus: 79.0e9,
        poisson_ratio: 0.44,
        yield_strength: 25.0e6,
        tensile_strength: 120.0e6,
        thermal_conductivity: 318.0,
        specific_heat: 129.0,
        thermal_expansion: 14.2e-6,
        melting_point: 1337.33,
    },
    // Silver
    Material {
        name: "Silver",
        density: 10490.0,
        youngs_modulus: 83.0e9,
        poisson_ratio: 0.37,
        yield_strength: 55.0e6,
        tensile_strength: 170.0e6,
        thermal_conductivity: 429.0,
        specific_heat: 235.0,
        thermal_expansion: 18.9e-6,
        melting_point: 1234.93,
    },
    // Platinum
    Material {
        name: "Platinum",
        density: 21450.0,
        youngs_modulus: 168.0e9,
        poisson_ratio: 0.38,
        yield_strength: 14.0e6,
        tensile_strength: 125.0e6,
        thermal_conductivity: 71.6,
        specific_heat: 133.0,
        thermal_expansion: 8.8e-6,
        melting_point: 2041.4,
    },
    // Nylon 6/6
    Material {
        name: "Nylon",
        density: 1140.0,
        youngs_modulus: 2.8e9,
        poisson_ratio: 0.39,
        yield_strength: 45.0e6,
        tensile_strength: 75.0e6,
        thermal_conductivity: 0.25,
        specific_heat: 1700.0,
        thermal_expansion: 80.0e-6,
        melting_point: 533.0,
    },
    // PTFE (Teflon)
    Material {
        name: "PTFE",
        density: 2200.0,
        youngs_modulus: 0.5e9,
        poisson_ratio: 0.46,
        yield_strength: 10.0e6,
        tensile_strength: 25.0e6,
        thermal_conductivity: 0.25,
        specific_heat: 1000.0,
        thermal_expansion: 135.0e-6,
        melting_point: 600.0,
    },
];

/// Looks up a material by name using case-insensitive ASCII comparison.
///
/// Zero-allocation: uses `eq_ignore_ascii_case` instead of `to_lowercase`.
pub fn by_name(name: &str) -> Option<&'static Material> {
    MATERIALS.iter().find(|m| m.name.eq_ignore_ascii_case(name))
}

/// Returns a slice of all common engineering materials.
pub fn all() -> &'static [Material] {
    &MATERIALS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steel_youngs_modulus() {
        let steel = by_name("Steel").expect("Steel should exist");
        let modulus_gpa = steel.youngs_modulus / 1.0e9;
        assert!((modulus_gpa - 200.0).abs() < 5.0, "Steel E ~ 200 GPa, got {modulus_gpa}");
    }

    #[test]
    fn case_insensitive_lookup() {
        assert!(by_name("steel").is_some());
        assert!(by_name("STEEL").is_some());
        assert!(by_name("COPPER").is_some());
        assert!(by_name("ptfe").is_some());
    }

    #[test]
    fn nonexistent_material_returns_none() {
        assert!(by_name("nonexistent").is_none());
        assert!(by_name("Vibranium").is_none());
    }

    #[test]
    fn diamond_has_highest_youngs_modulus() {
        let diamond = by_name("Diamond").unwrap();
        for m in MATERIALS.iter() {
            if m.name != "Diamond" {
                assert!(diamond.youngs_modulus > m.youngs_modulus,
                    "Diamond E should exceed {}", m.name);
            }
        }
    }

    #[test]
    fn all_returns_full_list() {
        let materials = all();
        assert_eq!(materials.len(), 20, "should have 20 materials, got {}", materials.len());
        assert!(materials.iter().any(|m| m.name == "Steel"));
        assert!(materials.iter().any(|m| m.name == "PTFE"));
    }
}
