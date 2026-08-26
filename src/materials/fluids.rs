//! Common liquids.
//!
//! Density, dynamic and kinematic viscosity, surface tension, speed of
//! sound, specific heat, and boiling and freezing points, at room
//! temperature and one atmosphere.
//!
//! Viscosity is the strongly temperature-dependent one: it can change by a
//! factor of several over a few tens of degrees, so a single figure is
//! only a starting point.

/// A fluid (liquid) with its mechanical and thermal properties at 20 degrees C
/// unless otherwise noted in the entry.
///
/// All values use SI units:
/// - `density`: kg/m^3
/// - `dynamic_viscosity`: Pa*s
/// - `kinematic_viscosity`: m^2/s
/// - `surface_tension`: N/m
/// - `specific_heat`: J/(kg*K)
/// - `thermal_conductivity`: W/(m*K)
/// - `boiling_point`, `freezing_point`: K
#[derive(Debug, Clone)]
pub struct Fluid {
    pub name: &'static str,
    pub density: f64,
    pub dynamic_viscosity: f64,
    pub kinematic_viscosity: f64,
    pub surface_tension: f64,
    pub specific_heat: f64,
    pub thermal_conductivity: f64,
    pub boiling_point: f64,
    pub freezing_point: f64,
}

static FLUIDS: [Fluid; 16] = [
    // Water at 20 C
    Fluid {
        name: "Water",
        density: 998.2,
        dynamic_viscosity: 1.002e-3,
        kinematic_viscosity: 1.004e-6,
        surface_tension: 0.0728,
        specific_heat: 4182.0,
        thermal_conductivity: 0.598,
        boiling_point: 373.15,
        freezing_point: 273.15,
    },
    // Seawater (3.5% salinity) at 20 C
    Fluid {
        name: "Seawater",
        density: 1025.0,
        dynamic_viscosity: 1.08e-3,
        kinematic_viscosity: 1.054e-6,
        surface_tension: 0.0735,
        specific_heat: 3993.0,
        thermal_conductivity: 0.596,
        boiling_point: 373.65,
        freezing_point: 271.35,
    },
    // Mercury at 20 C
    Fluid {
        name: "Mercury",
        density: 13546.0,
        dynamic_viscosity: 1.526e-3,
        kinematic_viscosity: 1.126e-7,
        surface_tension: 0.487,
        specific_heat: 139.3,
        thermal_conductivity: 8.69,
        boiling_point: 629.88,
        freezing_point: 234.32,
    },
    // Ethanol at 20 C
    Fluid {
        name: "Ethanol",
        density: 789.0,
        dynamic_viscosity: 1.20e-3,
        kinematic_viscosity: 1.52e-6,
        surface_tension: 0.0223,
        specific_heat: 2440.0,
        thermal_conductivity: 0.167,
        boiling_point: 351.44,
        freezing_point: 159.05,
    },
    // Methanol at 20 C
    Fluid {
        name: "Methanol",
        density: 791.0,
        dynamic_viscosity: 5.9e-4,
        kinematic_viscosity: 7.46e-7,
        surface_tension: 0.0226,
        specific_heat: 2530.0,
        thermal_conductivity: 0.200,
        boiling_point: 337.7,
        freezing_point: 175.47,
    },
    // Glycerin at 20 C
    Fluid {
        name: "Glycerin",
        density: 1261.0,
        dynamic_viscosity: 1.412,
        kinematic_viscosity: 1.12e-3,
        surface_tension: 0.0634,
        specific_heat: 2430.0,
        thermal_conductivity: 0.285,
        boiling_point: 563.0,
        freezing_point: 291.15,
    },
    // Motor Oil (SAE 30) at 20 C
    Fluid {
        name: "Motor Oil (SAE 30)",
        density: 891.0,
        dynamic_viscosity: 0.29,
        kinematic_viscosity: 3.25e-4,
        surface_tension: 0.035,
        specific_heat: 1880.0,
        thermal_conductivity: 0.145,
        boiling_point: 588.0,
        freezing_point: 243.0,
    },
    // Gasoline at 20 C
    Fluid {
        name: "Gasoline",
        density: 737.0,
        dynamic_viscosity: 6.0e-4,
        kinematic_viscosity: 8.14e-7,
        surface_tension: 0.022,
        specific_heat: 2220.0,
        thermal_conductivity: 0.15,
        boiling_point: 373.0,
        freezing_point: 213.0,
    },
    // Diesel at 20 C
    Fluid {
        name: "Diesel",
        density: 832.0,
        dynamic_viscosity: 2.6e-3,
        kinematic_viscosity: 3.12e-6,
        surface_tension: 0.028,
        specific_heat: 2050.0,
        thermal_conductivity: 0.13,
        boiling_point: 533.0,
        freezing_point: 253.0,
    },
    // Acetone at 20 C
    Fluid {
        name: "Acetone",
        density: 790.0,
        dynamic_viscosity: 3.16e-4,
        kinematic_viscosity: 4.0e-7,
        surface_tension: 0.0237,
        specific_heat: 2160.0,
        thermal_conductivity: 0.161,
        boiling_point: 329.2,
        freezing_point: 178.5,
    },
    // Sulfuric Acid (98%) at 20 C
    Fluid {
        name: "Sulfuric Acid",
        density: 1830.0,
        dynamic_viscosity: 2.42e-2,
        kinematic_viscosity: 1.32e-5,
        surface_tension: 0.0735,
        specific_heat: 1380.0,
        thermal_conductivity: 0.35,
        boiling_point: 610.0,
        freezing_point: 283.46,
    },
    // Liquid Nitrogen at -196 C (77 K)
    Fluid {
        name: "Liquid Nitrogen",
        density: 808.0,
        dynamic_viscosity: 1.58e-4,
        kinematic_viscosity: 1.95e-7,
        surface_tension: 0.00885,
        specific_heat: 2042.0,
        thermal_conductivity: 0.1396,
        boiling_point: 77.36,
        freezing_point: 63.15,
    },
    // Liquid Helium at -269 C (4.2 K)
    Fluid {
        name: "Liquid Helium",
        density: 125.0,
        dynamic_viscosity: 3.3e-6,
        kinematic_viscosity: 2.64e-8,
        surface_tension: 0.000354,
        specific_heat: 4545.0,
        thermal_conductivity: 0.0187,
        boiling_point: 4.222,
        freezing_point: 0.0,
    },
    // Blood (human, whole) at 37 C
    Fluid {
        name: "Blood",
        density: 1060.0,
        dynamic_viscosity: 3.5e-3,
        kinematic_viscosity: 3.30e-6,
        surface_tension: 0.058,
        specific_heat: 3617.0,
        thermal_conductivity: 0.492,
        boiling_point: 373.0,
        freezing_point: 272.55,
    },
    // Honey at 20 C
    Fluid {
        name: "Honey",
        density: 1420.0,
        dynamic_viscosity: 10.0,
        kinematic_viscosity: 7.04e-3,
        surface_tension: 0.050,
        specific_heat: 2430.0,
        thermal_conductivity: 0.50,
        boiling_point: 0.0,
        freezing_point: 0.0,
    },
    // Olive Oil at 20 C
    Fluid {
        name: "Olive Oil",
        density: 916.0,
        dynamic_viscosity: 8.4e-2,
        kinematic_viscosity: 9.17e-5,
        surface_tension: 0.032,
        specific_heat: 1970.0,
        thermal_conductivity: 0.17,
        boiling_point: 573.0,
        freezing_point: 263.0,
    },
];

/// Looks up a fluid by name using case-insensitive ASCII comparison.
///
/// Zero-allocation: uses `eq_ignore_ascii_case` instead of `to_lowercase`.
pub fn by_name(name: &str) -> Option<&'static Fluid> {
    FLUIDS.iter().find(|f| f.name.eq_ignore_ascii_case(name))
}

/// Returns a slice of all fluids in the database.
pub fn all() -> &'static [Fluid] {
    &FLUIDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_density() {
        let water = by_name("Water").expect("Water should exist");
        assert!((water.density - 998.0).abs() < 5.0,
            "Water density ~ 998 kg/m3, got {}", water.density);
    }

    #[test]
    fn mercury_is_densest_fluid() {
        let mercury = by_name("Mercury").unwrap();
        for f in FLUIDS.iter() {
            if f.name != "Mercury" {
                assert!(mercury.density > f.density,
                    "Mercury should be denser than {}", f.name);
            }
        }
    }

    #[test]
    fn honey_is_most_viscous() {
        let honey = by_name("Honey").unwrap();
        for f in FLUIDS.iter() {
            if f.name != "Honey" {
                assert!(honey.dynamic_viscosity > f.dynamic_viscosity,
                    "Honey should be more viscous than {}", f.name);
            }
        }
    }

    #[test]
    fn nonexistent_fluid_returns_none() {
        assert!(by_name("nonexistent").is_none());
        assert!(by_name("Liquid Unobtanium").is_none());
    }

    #[test]
    fn case_insensitive_lookup() {
        assert!(by_name("water").is_some());
        assert!(by_name("MERCURY").is_some());
        assert!(by_name("ethanol").is_some());
    }

    #[test]
    fn kinematic_viscosity_consistent() {
        for f in FLUIDS.iter() {
            assert!(f.density > 0.0, "{}: density must be positive", f.name);
            assert!(f.kinematic_viscosity > 0.0, "{}: kinematic_viscosity must be positive", f.name);
            let computed = f.dynamic_viscosity / f.density;
            let rel_error = (computed - f.kinematic_viscosity).abs() / f.kinematic_viscosity;
            assert!(rel_error < 0.1,
                "{}: nu={}, but mu/rho={computed} (rel error {rel_error:.4})",
                f.name, f.kinematic_viscosity);
        }
    }

    #[test]
    fn all_returns_full_list() {
        let fluids = all();
        assert_eq!(fluids.len(), 16, "should have 16 fluids, got {}", fluids.len());
        assert!(fluids.iter().any(|f| f.name == "Water"));
        assert!(fluids.iter().any(|f| f.name == "Olive Oil"));
    }
}
