use crate::math::constants;

// ── Ideal Gas Law ──

/// Ideal gas law: PV = nRT. Solve for pressure: P = nRT / V
pub fn ideal_gas_pressure(moles: f64, temperature: f64, volume: f64) -> f64 {
    moles * constants::R * temperature / volume
}

/// Solve for volume: V = nRT / P
pub fn ideal_gas_volume(moles: f64, temperature: f64, pressure: f64) -> f64 {
    moles * constants::R * temperature / pressure
}

/// Solve for temperature: T = PV / (nR)
pub fn ideal_gas_temperature(pressure: f64, volume: f64, moles: f64) -> f64 {
    pressure * volume / (moles * constants::R)
}

/// Number of moles: n = PV / (RT)
pub fn ideal_gas_moles(pressure: f64, volume: f64, temperature: f64) -> f64 {
    pressure * volume / (constants::R * temperature)
}

// ── Kinetic Theory ──

/// Average kinetic energy of a gas molecule: KE = (3/2) * k_B * T
pub fn average_kinetic_energy(temperature: f64) -> f64 {
    1.5 * constants::K_B * temperature
}

/// RMS speed of gas molecules: v_rms = sqrt(3 * k_B * T / m)
pub fn rms_speed(temperature: f64, molecular_mass: f64) -> f64 {
    (3.0 * constants::K_B * temperature / molecular_mass).sqrt()
}

/// Mean free path: λ = 1 / (√2 * π * d^2 * n/V)
pub fn mean_free_path(molecular_diameter: f64, number_density: f64) -> f64 {
    1.0 / (std::f64::consts::SQRT_2 * constants::PI * molecular_diameter * molecular_diameter * number_density)
}

// ── Heat Transfer ──

/// Heat transfer: Q = m * c * ΔT
pub fn heat_transfer(mass: f64, specific_heat: f64, delta_temp: f64) -> f64 {
    mass * specific_heat * delta_temp
}

/// Heat conduction (Fourier's law): Q/t = k * A * ΔT / d
pub fn heat_conduction_rate(
    conductivity: f64,
    area: f64,
    delta_temp: f64,
    thickness: f64,
) -> f64 {
    conductivity * area * delta_temp / thickness
}

/// Heat radiation (Stefan-Boltzmann law): P = ε * σ * A * T^4
pub fn heat_radiation_power(emissivity: f64, area: f64, temperature: f64) -> f64 {
    emissivity * constants::SIGMA * area * temperature.powi(4)
}

/// Net radiative heat transfer: P = ε * σ * A * (T_hot^4 - T_cold^4)
pub fn net_radiation_power(
    emissivity: f64,
    area: f64,
    t_hot: f64,
    t_cold: f64,
) -> f64 {
    emissivity * constants::SIGMA * area * (t_hot.powi(4) - t_cold.powi(4))
}

/// Newton's law of cooling: dT/dt = -k * (T - T_env)
/// Returns temperature at time t: T(t) = T_env + (T0 - T_env) * e^(-k*t)
pub fn newton_cooling(t_initial: f64, t_environment: f64, cooling_constant: f64, time: f64) -> f64 {
    t_environment + (t_initial - t_environment) * (-cooling_constant * time).exp()
}

// ── Thermodynamic Processes ──

/// Work done by an ideal gas during isothermal expansion: W = nRT * ln(V2/V1)
pub fn work_isothermal(moles: f64, temperature: f64, v1: f64, v2: f64) -> f64 {
    moles * constants::R * temperature * (v2 / v1).ln()
}

/// Work done during isobaric (constant pressure) process: W = P * ΔV
pub fn work_isobaric(pressure: f64, delta_v: f64) -> f64 {
    pressure * delta_v
}

/// Work done during adiabatic process: W = (P1*V1 - P2*V2) / (γ - 1)
pub fn work_adiabatic(p1: f64, v1: f64, p2: f64, v2: f64, gamma: f64) -> f64 {
    (p1 * v1 - p2 * v2) / (gamma - 1.0)
}

/// Adiabatic relation: P1 * V1^γ = P2 * V2^γ → P2 = P1 * (V1/V2)^γ
pub fn adiabatic_final_pressure(p1: f64, v1: f64, v2: f64, gamma: f64) -> f64 {
    p1 * (v1 / v2).powf(gamma)
}

// ── Entropy ──

/// Entropy change for heat transfer at constant temperature: ΔS = Q / T
pub fn entropy_change_isothermal(heat: f64, temperature: f64) -> f64 {
    heat / temperature
}

/// Entropy change for an ideal gas: ΔS = n*Cv*ln(T2/T1) + n*R*ln(V2/V1)
pub fn entropy_change_ideal_gas(
    moles: f64,
    cv: f64,
    t1: f64,
    t2: f64,
    v1: f64,
    v2: f64,
) -> f64 {
    moles * cv * (t2 / t1).ln() + moles * constants::R * (v2 / v1).ln()
}

// ── Heat Engines ──

/// Carnot efficiency: η = 1 - T_cold / T_hot
pub fn carnot_efficiency(t_cold: f64, t_hot: f64) -> f64 {
    1.0 - t_cold / t_hot
}

/// Thermal efficiency: η = W / Q_hot
pub fn thermal_efficiency(work: f64, heat_input: f64) -> f64 {
    work / heat_input
}

/// Coefficient of performance (refrigerator): COP = Q_cold / W
pub fn cop_refrigerator(heat_removed: f64, work: f64) -> f64 {
    heat_removed / work
}

/// Coefficient of performance (heat pump): COP = Q_hot / W
pub fn cop_heat_pump(heat_delivered: f64, work: f64) -> f64 {
    heat_delivered / work
}

// ── Phase Changes ──

/// Heat for phase change: Q = m * L (latent heat)
pub fn latent_heat(mass: f64, specific_latent_heat: f64) -> f64 {
    mass * specific_latent_heat
}

/// Clausius-Clapeyron (approximate): ln(P2/P1) = (L/R) * (1/T1 - 1/T2)
/// Returns P2 given P1, T1, T2, and molar latent heat L.
pub fn clausius_clapeyron(p1: f64, t1: f64, t2: f64, molar_latent_heat: f64) -> f64 {
    p1 * (molar_latent_heat / constants::R * (1.0 / t1 - 1.0 / t2)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_ideal_gas_pressure() {
        // 1 mol at 273 K in 0.0224 m^3 ≈ 101325 Pa
        let p = ideal_gas_pressure(1.0, 273.15, 0.02241);
        assert!(approx_rel(p, 101325.0, 0.01));
    }

    #[test]
    fn test_heat_transfer() {
        // 1 kg water, c=4186, ΔT=10 → Q = 41860
        let q = heat_transfer(1.0, 4186.0, 10.0);
        assert!(approx(q, 41860.0, 0.1));
    }

    #[test]
    fn test_carnot_efficiency() {
        let eff = carnot_efficiency(300.0, 600.0);
        assert!(approx(eff, 0.5, 1e-9));
    }

    #[test]
    fn test_work_isothermal() {
        let w = work_isothermal(1.0, 300.0, 1.0, 2.0);
        assert!(approx_rel(w, 1728.85, 0.01));
    }

    #[test]
    fn test_newton_cooling() {
        // After infinite time, should approach environment temp
        let t = newton_cooling(100.0, 20.0, 0.1, 1000.0);
        assert!(approx(t, 20.0, 0.01));
    }

    #[test]
    fn test_entropy_change_isothermal() {
        let ds = entropy_change_isothermal(1000.0, 500.0);
        assert!(approx(ds, 2.0, 1e-9));
    }

    #[test]
    fn test_latent_heat() {
        // 2 kg of ice, latent heat of fusion = 334000 J/kg
        let q = latent_heat(2.0, 334000.0);
        assert!(approx(q, 668000.0, 0.1));
    }
}
