/// A gas with its thermodynamic and transport properties at STP.
///
/// All values use SI units:
/// - `molar_mass`: kg/mol
/// - `cp`, `cv`: J/(kg*K)
/// - `density_stp`: kg/m^3
/// - `dynamic_viscosity`: Pa*s
/// - `thermal_conductivity`: W/(m*K)
#[derive(Debug, Clone)]
pub struct Gas {
    pub name: &'static str,
    pub formula: &'static str,
    pub molar_mass: f64,
    pub specific_heat_ratio: f64,
    pub cp: f64,
    pub cv: f64,
    pub density_stp: f64,
    pub dynamic_viscosity: f64,
    pub thermal_conductivity: f64,
}

static GASES: [Gas; 12] = [
    // Air (dry, standard composition)
    Gas {
        name: "Air",
        formula: "N2/O2",
        molar_mass: 0.02897,
        specific_heat_ratio: 1.400,
        cp: 1005.0,
        cv: 717.86,
        density_stp: 1.293,
        dynamic_viscosity: 1.81e-5,
        thermal_conductivity: 0.0257,
    },
    // Nitrogen
    Gas {
        name: "Nitrogen",
        formula: "N2",
        molar_mass: 0.02802,
        specific_heat_ratio: 1.400,
        cp: 1040.0,
        cv: 742.86,
        density_stp: 1.2506,
        dynamic_viscosity: 1.76e-5,
        thermal_conductivity: 0.02583,
    },
    // Oxygen
    Gas {
        name: "Oxygen",
        formula: "O2",
        molar_mass: 0.032,
        specific_heat_ratio: 1.395,
        cp: 918.0,
        cv: 658.1,
        density_stp: 1.429,
        dynamic_viscosity: 2.04e-5,
        thermal_conductivity: 0.02658,
    },
    // Hydrogen
    Gas {
        name: "Hydrogen",
        formula: "H2",
        molar_mass: 0.002016,
        specific_heat_ratio: 1.410,
        cp: 14304.0,
        cv: 10142.0,
        density_stp: 0.08988,
        dynamic_viscosity: 8.76e-6,
        thermal_conductivity: 0.1805,
    },
    // Helium
    Gas {
        name: "Helium",
        formula: "He",
        molar_mass: 0.004003,
        specific_heat_ratio: 1.667,
        cp: 5193.0,
        cv: 3116.0,
        density_stp: 0.1786,
        dynamic_viscosity: 1.96e-5,
        thermal_conductivity: 0.1513,
    },
    // Carbon Dioxide
    Gas {
        name: "Carbon Dioxide",
        formula: "CO2",
        molar_mass: 0.04401,
        specific_heat_ratio: 1.289,
        cp: 844.0,
        cv: 655.0,
        density_stp: 1.977,
        dynamic_viscosity: 1.47e-5,
        thermal_conductivity: 0.01662,
    },
    // Argon
    Gas {
        name: "Argon",
        formula: "Ar",
        molar_mass: 0.03995,
        specific_heat_ratio: 1.667,
        cp: 520.0,
        cv: 312.0,
        density_stp: 1.784,
        dynamic_viscosity: 2.23e-5,
        thermal_conductivity: 0.01772,
    },
    // Methane
    Gas {
        name: "Methane",
        formula: "CH4",
        molar_mass: 0.01604,
        specific_heat_ratio: 1.320,
        cp: 2226.0,
        cv: 1687.0,
        density_stp: 0.717,
        dynamic_viscosity: 1.10e-5,
        thermal_conductivity: 0.0343,
    },
    // Steam (at 100 C, 1 atm)
    Gas {
        name: "Steam",
        formula: "H2O",
        molar_mass: 0.01802,
        specific_heat_ratio: 1.330,
        cp: 2080.0,
        cv: 1564.0,
        density_stp: 0.5974,
        dynamic_viscosity: 1.27e-5,
        thermal_conductivity: 0.0248,
    },
    // Neon
    Gas {
        name: "Neon",
        formula: "Ne",
        molar_mass: 0.02018,
        specific_heat_ratio: 1.667,
        cp: 1030.0,
        cv: 618.0,
        density_stp: 0.9002,
        dynamic_viscosity: 3.13e-5,
        thermal_conductivity: 0.0491,
    },
    // Xenon
    Gas {
        name: "Xenon",
        formula: "Xe",
        molar_mass: 0.13129,
        specific_heat_ratio: 1.667,
        cp: 158.3,
        cv: 95.0,
        density_stp: 5.894,
        dynamic_viscosity: 2.27e-5,
        thermal_conductivity: 0.00569,
    },
    // Krypton
    Gas {
        name: "Krypton",
        formula: "Kr",
        molar_mass: 0.08380,
        specific_heat_ratio: 1.667,
        cp: 248.0,
        cv: 149.0,
        density_stp: 3.749,
        dynamic_viscosity: 2.51e-5,
        thermal_conductivity: 0.00943,
    },
];

/// Looks up a gas by name using case-insensitive ASCII comparison.
///
/// Zero-allocation: uses `eq_ignore_ascii_case` instead of `to_lowercase`.
pub fn by_name(name: &str) -> Option<&'static Gas> {
    GASES.iter().find(|g| g.name.eq_ignore_ascii_case(name))
}

/// Returns a slice of all gases in the database.
pub fn all() -> &'static [Gas] {
    &GASES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_specific_heat_ratio() {
        let air = by_name("Air").expect("Air should exist");
        assert!((air.specific_heat_ratio - 1.4).abs() < 0.01,
            "Air gamma ~ 1.4, got {}", air.specific_heat_ratio);
    }

    #[test]
    fn gamma_equals_cp_over_cv() {
        for gas in GASES.iter() {
            let computed_gamma = gas.cp / gas.cv;
            let rel_error = (computed_gamma - gas.specific_heat_ratio).abs() / gas.specific_heat_ratio;
            assert!(rel_error < 0.02,
                "{}: gamma={}, but Cp/Cv={computed_gamma} (rel error {rel_error:.4})",
                gas.name, gas.specific_heat_ratio);
        }
    }

    #[test]
    fn noble_gases_have_monatomic_gamma() {
        for name in &["Helium", "Argon", "Neon", "Xenon", "Krypton"] {
            let gas = by_name(name).unwrap();
            assert!((gas.specific_heat_ratio - 1.667).abs() < 0.01,
                "{name} should have gamma ~ 5/3");
        }
    }

    #[test]
    fn nonexistent_gas_returns_none() {
        assert!(by_name("nonexistent").is_none());
        assert!(by_name("Phlogiston").is_none());
    }

    #[test]
    fn case_insensitive_lookup() {
        assert!(by_name("air").is_some());
        assert!(by_name("HELIUM").is_some());
        assert!(by_name("carbon dioxide").is_some());
    }

    #[test]
    fn all_returns_full_list() {
        let gases = all();
        assert_eq!(gases.len(), 12, "should have 12 gases, got {}", gases.len());
        assert!(gases.iter().any(|g| g.name == "Air"));
        assert!(gases.iter().any(|g| g.name == "Krypton"));
    }
}
