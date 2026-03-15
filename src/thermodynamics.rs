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

// ── Convective Heat Transfer ──

/// Newton's law of convection: Q/t = h×A×ΔT
pub fn convective_heat_rate(h: f64, area: f64, delta_temp: f64) -> f64 {
    h * area * delta_temp
}

/// Thermal diffusivity: α = k/(ρ×cₚ)
pub fn thermal_diffusivity(conductivity: f64, density: f64, specific_heat: f64) -> f64 {
    conductivity / (density * specific_heat)
}

// ── Dimensionless Numbers ──

/// Grashof number: Gr = gβΔTL³/ν²
pub fn grashof_number(
    g: f64,
    beta: f64,
    delta_temp: f64,
    length: f64,
    kinematic_viscosity: f64,
) -> f64 {
    g * beta * delta_temp * length.powi(3) / kinematic_viscosity.powi(2)
}

/// Rayleigh number: Ra = Gr × Pr
pub fn rayleigh_number(grashof: f64, prandtl: f64) -> f64 {
    grashof * prandtl
}

/// Prandtl number: Pr = ν/α
pub fn prandtl_number(kinematic_viscosity: f64, thermal_diffusivity: f64) -> f64 {
    kinematic_viscosity / thermal_diffusivity
}

/// Nusselt number: Nu = hL/k
pub fn nusselt_number(h: f64, length: f64, conductivity: f64) -> f64 {
    h * length / conductivity
}

/// Biot number: Bi = hL/k (external convection vs internal conduction)
pub fn biot_number(h: f64, length: f64, conductivity: f64) -> f64 {
    h * length / conductivity
}

// ── Heat Equation (1D transient) ──

/// Explicit finite difference: T_i^(n+1) = T_i^n + α×dt/dx² × (T_(i+1) - 2T_i + T_(i-1))
/// Fixed boundary conditions (first and last elements unchanged).
pub fn heat_equation_step_1d(temperatures: &mut [f64], dx: f64, dt: f64, diffusivity: f64) {
    let n = temperatures.len();
    if n < 3 {
        return;
    }
    let r = diffusivity * dt / (dx * dx);
    let old: Vec<f64> = temperatures.to_vec();
    for i in 1..n - 1 {
        temperatures[i] = old[i] + r * (old[i + 1] - 2.0 * old[i] + old[i - 1]);
    }
}

/// Maximum stable time step for explicit finite difference: dt_max = dx²/(2α)
pub fn heat_equation_stability(dx: f64, diffusivity: f64) -> f64 {
    dx * dx / (2.0 * diffusivity)
}

// ── Thermal Radiation (extended) ──

/// Wien's displacement law: λ_max = b/T where b = 2.898e-3 m·K
pub fn wien_displacement(temperature: f64) -> f64 {
    const WIEN_B: f64 = 2.898e-3;
    WIEN_B / temperature
}

/// Planck's law: M = (2πhc²/λ⁵) × 1/(e^(hc/λkT) - 1)
pub fn spectral_exitance(wavelength: f64, temperature: f64) -> f64 {
    let c = constants::C;
    let h = constants::H;
    let k = constants::K_B;
    let numerator = 2.0 * constants::PI * h * c * c / wavelength.powi(5);
    let exponent = h * c / (wavelength * k * temperature);
    numerator / (exponent.exp() - 1.0)
}

/// Radiative equilibrium temperature: T = ((L(1-a))/(16πσd²))^(1/4)
pub fn radiative_equilibrium_temperature(luminosity: f64, distance: f64, albedo: f64) -> f64 {
    let numerator = luminosity * (1.0 - albedo);
    let denominator = 16.0 * constants::PI * constants::SIGMA * distance * distance;
    (numerator / denominator).powf(0.25)
}

// ── Temperature Scales ──

/// Celsius to Kelvin: K = C + 273.15
pub fn celsius_to_kelvin(c: f64) -> f64 {
    c + 273.15
}

/// Kelvin to Celsius: C = K - 273.15
pub fn kelvin_to_celsius(k: f64) -> f64 {
    k - 273.15
}

/// Celsius to Fahrenheit: F = C × 9/5 + 32
pub fn celsius_to_fahrenheit(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

/// Fahrenheit to Celsius: C = (F - 32) × 5/9
pub fn fahrenheit_to_celsius(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0
}

/// Fahrenheit to Kelvin via Celsius
pub fn fahrenheit_to_kelvin(f: f64) -> f64 {
    celsius_to_kelvin(fahrenheit_to_celsius(f))
}

/// Kelvin to Fahrenheit via Celsius
pub fn kelvin_to_fahrenheit(k: f64) -> f64 {
    celsius_to_fahrenheit(kelvin_to_celsius(k))
}

/// Celsius to Rankine: R = (C + 273.15) × 9/5
pub fn celsius_to_rankine(c: f64) -> f64 {
    (c + 273.15) * 9.0 / 5.0
}

/// Rankine to Celsius: C = R × 5/9 - 273.15
pub fn rankine_to_celsius(r: f64) -> f64 {
    r * 5.0 / 9.0 - 273.15
}

// ── Phase Change Physics (extended) ──

/// Boiling point elevation: ΔTb = Kb × m
pub fn boiling_point_elevation(kb: f64, molality: f64) -> f64 {
    kb * molality
}

/// Freezing point depression: ΔTf = Kf × m
pub fn freezing_point_depression(kf: f64, molality: f64) -> f64 {
    kf * molality
}

/// Antoine equation: log10(P) = A - B/(C+T), returns P
pub fn saturation_pressure(t: f64, a: f64, b: f64, c: f64) -> f64 {
    10.0_f64.powf(a - b / (c + t))
}

/// Trouton's rule: ΔHvap ≈ 88 × Tb (J/mol)
pub fn heat_of_vaporization_trouton(boiling_point_k: f64) -> f64 {
    const TROUTON_CONSTANT: f64 = 88.0;
    TROUTON_CONSTANT * boiling_point_k
}

/// Degree of superheat: ΔT = T_actual - T_sat
pub fn superheat_degree(actual_temp: f64, saturation_temp: f64) -> f64 {
    actual_temp - saturation_temp
}

/// Degree of subcooling: ΔT = T_sat - T_actual
pub fn subcool_degree(saturation_temp: f64, actual_temp: f64) -> f64 {
    saturation_temp - actual_temp
}

/// Steam quality (dryness fraction): x = m_vapor / m_total
pub fn quality(mass_vapor: f64, mass_total: f64) -> f64 {
    mass_vapor / mass_total
}

/// Specific enthalpy of wet steam: h = hf + x × hfg
pub fn specific_enthalpy_wet(hf: f64, hfg: f64, quality: f64) -> f64 {
    hf + quality * hfg
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

    // ── Convective Heat Transfer ──

    #[test]
    fn test_convective_heat_rate() {
        // h=25 W/(m²·K), A=2 m², ΔT=50 K → Q/t = 2500 W
        let q = convective_heat_rate(25.0, 2.0, 50.0);
        assert!(approx(q, 2500.0, 1e-9));
    }

    #[test]
    fn test_thermal_diffusivity() {
        // Aluminum: k=237, ρ=2700, c=900 → α ≈ 9.753e-5 m²/s
        let alpha = thermal_diffusivity(237.0, 2700.0, 900.0);
        assert!(approx_rel(alpha, 9.753e-5, 0.01));
    }

    // ── Dimensionless Numbers ──

    #[test]
    fn test_grashof_number() {
        // g=9.81, β=3.4e-3, ΔT=20, L=0.5, ν=1.5e-5
        let gr = grashof_number(9.81, 3.4e-3, 20.0, 0.5, 1.5e-5);
        let expected = 9.81 * 3.4e-3 * 20.0 * 0.5_f64.powi(3) / (1.5e-5_f64).powi(2);
        assert!(approx_rel(gr, expected, 1e-9));
    }

    #[test]
    fn test_rayleigh_number() {
        let ra = rayleigh_number(1e6, 0.71);
        assert!(approx_rel(ra, 7.1e5, 1e-9));
    }

    #[test]
    fn test_prandtl_number() {
        // Air: ν=1.5e-5, α=2.1e-5 → Pr ≈ 0.714
        let pr = prandtl_number(1.5e-5, 2.1e-5);
        assert!(approx_rel(pr, 0.7143, 0.01));
    }

    #[test]
    fn test_nusselt_number() {
        // h=50, L=0.1, k=0.6 → Nu ≈ 8.333
        let nu = nusselt_number(50.0, 0.1, 0.6);
        assert!(approx_rel(nu, 8.333, 0.01));
    }

    #[test]
    fn test_biot_number() {
        // h=100, L=0.01, k=50 → Bi = 0.02
        let bi = biot_number(100.0, 0.01, 50.0);
        assert!(approx(bi, 0.02, 1e-9));
    }

    // ── Heat Equation (1D transient) ──

    #[test]
    fn test_heat_equation_step_1d() {
        // Rod with fixed ends at 100 and 0, interior starts at 0
        let mut temps = vec![100.0, 0.0, 0.0, 0.0, 0.0];
        let dx = 0.01;
        let alpha = 1e-4;
        let dt = heat_equation_stability(dx, alpha) * 0.5; // stable step
        heat_equation_step_1d(&mut temps, dx, dt, alpha);
        // Boundaries unchanged
        assert!(approx(temps[0], 100.0, 1e-9));
        assert!(approx(temps[4], 0.0, 1e-9));
        // Interior node next to hot boundary should have increased
        assert!(temps[1] > 0.0);
    }

    #[test]
    fn test_heat_equation_step_1d_short_array() {
        // Arrays shorter than 3 should be unchanged
        let mut temps = vec![100.0, 50.0];
        heat_equation_step_1d(&mut temps, 0.01, 0.001, 1e-4);
        assert!(approx(temps[0], 100.0, 1e-9));
        assert!(approx(temps[1], 50.0, 1e-9));
    }

    #[test]
    fn test_heat_equation_stability() {
        // dx=0.01, α=1e-4 → dt_max = 0.01²/(2×1e-4) = 0.5
        let dt_max = heat_equation_stability(0.01, 1e-4);
        assert!(approx(dt_max, 0.5, 1e-9));
    }

    // ── Thermal Radiation (extended) ──

    #[test]
    fn test_wien_displacement() {
        // Sun surface ~5778 K → λ_max ≈ 501 nm
        let lambda = wien_displacement(5778.0);
        assert!(approx_rel(lambda, 5.014e-7, 0.01));
    }

    #[test]
    fn test_spectral_exitance() {
        // At peak wavelength of 5778 K, exitance should be positive and large
        let lambda = wien_displacement(5778.0);
        let m = spectral_exitance(lambda, 5778.0);
        assert!(m > 0.0);
        // Peak spectral exitance for Sun ~8.3e13 W/m³
        assert!(approx_rel(m, 8.3e13, 0.05));
    }

    #[test]
    fn test_radiative_equilibrium_temperature() {
        // Earth: L_sun=3.846e26 W, d=1.496e11 m, a=0.3 → T ≈ 255 K
        let t = radiative_equilibrium_temperature(3.846e26, 1.496e11, 0.3);
        assert!(approx_rel(t, 255.0, 0.03));
    }

    // ── Temperature Scales ──

    #[test]
    fn test_celsius_to_kelvin() {
        assert!(approx(celsius_to_kelvin(0.0), 273.15, 1e-9));
        assert!(approx(celsius_to_kelvin(100.0), 373.15, 1e-9));
        assert!(approx(celsius_to_kelvin(-273.15), 0.0, 1e-9));
    }

    #[test]
    fn test_kelvin_to_celsius() {
        assert!(approx(kelvin_to_celsius(273.15), 0.0, 1e-9));
        assert!(approx(kelvin_to_celsius(0.0), -273.15, 1e-9));
    }

    #[test]
    fn test_celsius_to_fahrenheit() {
        assert!(approx(celsius_to_fahrenheit(0.0), 32.0, 1e-9));
        assert!(approx(celsius_to_fahrenheit(100.0), 212.0, 1e-9));
        assert!(approx(celsius_to_fahrenheit(-40.0), -40.0, 1e-9));
    }

    #[test]
    fn test_fahrenheit_to_celsius() {
        assert!(approx(fahrenheit_to_celsius(32.0), 0.0, 1e-9));
        assert!(approx(fahrenheit_to_celsius(212.0), 100.0, 1e-9));
        assert!(approx(fahrenheit_to_celsius(-40.0), -40.0, 1e-9));
    }

    #[test]
    fn test_fahrenheit_to_kelvin() {
        assert!(approx(fahrenheit_to_kelvin(32.0), 273.15, 1e-9));
        assert!(approx(fahrenheit_to_kelvin(212.0), 373.15, 1e-9));
    }

    #[test]
    fn test_kelvin_to_fahrenheit() {
        assert!(approx(kelvin_to_fahrenheit(273.15), 32.0, 1e-9));
        assert!(approx(kelvin_to_fahrenheit(373.15), 212.0, 1e-9));
    }

    #[test]
    fn test_celsius_to_rankine() {
        // 0°C = 273.15 K = 491.67 R
        assert!(approx(celsius_to_rankine(0.0), 491.67, 0.01));
        // 100°C = 373.15 K = 671.67 R
        assert!(approx(celsius_to_rankine(100.0), 671.67, 0.01));
    }

    #[test]
    fn test_rankine_to_celsius() {
        assert!(approx(rankine_to_celsius(491.67), 0.0, 0.01));
        assert!(approx(rankine_to_celsius(671.67), 100.0, 0.01));
    }

    // ── Phase Change Physics (extended) ──

    #[test]
    fn test_boiling_point_elevation() {
        // Water: Kb=0.512, 1 molal → ΔTb = 0.512
        let dt = boiling_point_elevation(0.512, 1.0);
        assert!(approx(dt, 0.512, 1e-9));
    }

    #[test]
    fn test_freezing_point_depression() {
        // Water: Kf=1.86, 2 molal → ΔTf = 3.72
        let dt = freezing_point_depression(1.86, 2.0);
        assert!(approx(dt, 3.72, 1e-9));
    }

    #[test]
    fn test_saturation_pressure() {
        // Antoine for water (NIST): A=8.07131, B=1730.63, C=233.426 at T=100°C
        // log10(P) = 8.07131 - 1730.63/(233.426+100) ≈ 2.884 → P ≈ 766 mmHg
        let p = saturation_pressure(100.0, 8.07131, 1730.63, 233.426);
        assert!(approx_rel(p, 766.0, 0.01));
    }

    #[test]
    fn test_heat_of_vaporization_trouton() {
        // Water: Tb=373 K → ΔHvap ≈ 88×373 = 32824 J/mol
        let dh = heat_of_vaporization_trouton(373.0);
        assert!(approx(dh, 32824.0, 1e-9));
    }

    #[test]
    fn test_superheat_degree() {
        let dt = superheat_degree(120.0, 100.0);
        assert!(approx(dt, 20.0, 1e-9));
    }

    #[test]
    fn test_subcool_degree() {
        let dt = subcool_degree(100.0, 85.0);
        assert!(approx(dt, 15.0, 1e-9));
    }

    #[test]
    fn test_quality() {
        let x = quality(0.3, 1.0);
        assert!(approx(x, 0.3, 1e-9));
    }

    #[test]
    fn test_specific_enthalpy_wet() {
        // hf=417.5 kJ/kg, hfg=2258 kJ/kg, x=0.8 → h = 417.5 + 0.8×2258 = 2223.9
        let h = specific_enthalpy_wet(417.5, 2258.0, 0.8);
        assert!(approx(h, 2223.9, 0.1));
    }
}
