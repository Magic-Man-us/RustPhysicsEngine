use crate::math::{Vec3, constants};

// ── Electrostatics ──

/// Coulomb's law: F = k_e * |q1 * q2| / r^2
pub fn coulomb_force(q1: f64, q2: f64, distance: f64) -> f64 {
    constants::K_E * (q1 * q2).abs() / (distance * distance)
}

/// Coulomb force (signed, 1D): positive = repulsive, negative = attractive
pub fn coulomb_force_signed(q1: f64, q2: f64, distance: f64) -> f64 {
    constants::K_E * q1 * q2 / (distance * distance)
}

/// Coulomb force vector from charge at pos1 to charge at pos2.
pub fn coulomb_force_vec(q1: f64, pos1: Vec3, q2: f64, pos2: Vec3) -> Vec3 {
    let r = pos1 - pos2;
    let dist = r.magnitude();
    if dist == 0.0 {
        return Vec3::ZERO;
    }
    r.normalized() * (constants::K_E * q1 * q2 / (dist * dist))
}

/// Electric field due to a point charge: E = k_e * q / r^2
pub fn electric_field_point_charge(charge: f64, distance: f64) -> f64 {
    constants::K_E * charge / (distance * distance)
}

/// Electric field vector at a point due to a charge at a given position.
pub fn electric_field_vec(charge: f64, charge_pos: Vec3, field_point: Vec3) -> Vec3 {
    let r = field_point - charge_pos;
    let dist = r.magnitude();
    if dist == 0.0 {
        return Vec3::ZERO;
    }
    r.normalized() * (constants::K_E * charge / (dist * dist))
}

/// Electric potential due to a point charge: V = k_e * q / r
pub fn electric_potential(charge: f64, distance: f64) -> f64 {
    constants::K_E * charge / distance
}

/// Electric potential energy: U = k_e * q1 * q2 / r
pub fn electric_potential_energy(q1: f64, q2: f64, distance: f64) -> f64 {
    constants::K_E * q1 * q2 / distance
}

/// Electric flux through a surface (Gauss's law): Φ = q_enclosed / ε_0
pub fn electric_flux_gauss(enclosed_charge: f64) -> f64 {
    enclosed_charge / constants::EPSILON_0
}

/// Capacitance of a parallel plate capacitor: C = ε_0 * A / d
pub fn capacitance_parallel_plate(area: f64, separation: f64) -> f64 {
    constants::EPSILON_0 * area / separation
}

/// Energy stored in a capacitor: U = 0.5 * C * V^2
pub fn capacitor_energy(capacitance: f64, voltage: f64) -> f64 {
    0.5 * capacitance * voltage * voltage
}

// ── Electric Circuits ──

/// Ohm's law: V = I * R
pub fn ohms_law_voltage(current: f64, resistance: f64) -> f64 {
    current * resistance
}

/// Ohm's law: I = V / R
pub fn ohms_law_current(voltage: f64, resistance: f64) -> f64 {
    voltage / resistance
}

/// Ohm's law: R = V / I
pub fn ohms_law_resistance(voltage: f64, current: f64) -> f64 {
    voltage / current
}

/// Electrical power: P = V * I
pub fn electrical_power(voltage: f64, current: f64) -> f64 {
    voltage * current
}

/// Electrical power: P = I^2 * R
pub fn electrical_power_from_current(current: f64, resistance: f64) -> f64 {
    current * current * resistance
}

/// Resistors in series: R_total = R1 + R2 + ...
pub fn resistors_series(resistances: &[f64]) -> f64 {
    resistances.iter().sum()
}

/// Resistors in parallel: 1/R_total = 1/R1 + 1/R2 + ...
pub fn resistors_parallel(resistances: &[f64]) -> f64 {
    let sum: f64 = resistances.iter().map(|r| 1.0 / r).sum();
    1.0 / sum
}

/// Capacitors in series: 1/C_total = 1/C1 + 1/C2 + ...
pub fn capacitors_series(capacitances: &[f64]) -> f64 {
    let sum: f64 = capacitances.iter().map(|c| 1.0 / c).sum();
    1.0 / sum
}

/// Capacitors in parallel: C_total = C1 + C2 + ...
pub fn capacitors_parallel(capacitances: &[f64]) -> f64 {
    capacitances.iter().sum()
}

/// RC time constant: τ = R * C
pub fn rc_time_constant(resistance: f64, capacitance: f64) -> f64 {
    resistance * capacitance
}

/// Voltage across charging capacitor: V(t) = V0 * (1 - e^(-t/RC))
pub fn rc_charging_voltage(v0: f64, resistance: f64, capacitance: f64, time: f64) -> f64 {
    v0 * (1.0 - (-time / (resistance * capacitance)).exp())
}

// ── Magnetism ──

/// Magnetic force on a moving charge: F = q * v * B * sin(θ)
pub fn magnetic_force_on_charge(charge: f64, velocity: f64, b_field: f64, angle_rad: f64) -> f64 {
    (charge * velocity * b_field * angle_rad.sin()).abs()
}

/// Lorentz force: F = q * (E + v × B)
pub fn lorentz_force(charge: f64, e_field: Vec3, velocity: Vec3, b_field: Vec3) -> Vec3 {
    (e_field + velocity.cross(&b_field)) * charge
}

/// Magnetic field from a long straight wire: B = μ_0 * I / (2π * r)
pub fn magnetic_field_wire(current: f64, distance: f64) -> f64 {
    constants::MU_0 * current / (2.0 * constants::PI * distance)
}

/// Magnetic force between two parallel wires per unit length: F/L = μ_0 * I1 * I2 / (2π * d)
pub fn force_between_wires(i1: f64, i2: f64, distance: f64) -> f64 {
    constants::MU_0 * i1 * i2 / (2.0 * constants::PI * distance)
}

/// Cyclotron radius: r = m*v / (|q|*B)
pub fn cyclotron_radius(mass: f64, velocity: f64, charge: f64, b_field: f64) -> f64 {
    mass * velocity / (charge.abs() * b_field)
}

/// Cyclotron frequency: f = |q|*B / (2π*m)
pub fn cyclotron_frequency(charge: f64, b_field: f64, mass: f64) -> f64 {
    charge.abs() * b_field / (2.0 * constants::PI * mass)
}

// ── Electromagnetic Induction ──

/// Faraday's law (magnitude): EMF = -N * dΦ/dt
pub fn faraday_emf(num_turns: f64, delta_flux: f64, delta_time: f64) -> f64 {
    -(num_turns * delta_flux / delta_time)
}

/// Motional EMF: ε = B * L * v
pub fn motional_emf(b_field: f64, length: f64, velocity: f64) -> f64 {
    b_field * length * velocity
}

/// Inductance energy: U = 0.5 * L * I^2
pub fn inductor_energy(inductance: f64, current: f64) -> f64 {
    0.5 * inductance * current * current
}

// ── Electromagnetic Waves ──

/// Relationship between wavelength and frequency: c = λ * f
pub fn wavelength_from_frequency(frequency: f64) -> f64 {
    constants::C / frequency
}

pub fn frequency_from_wavelength(wavelength: f64) -> f64 {
    constants::C / wavelength
}

/// Poynting vector magnitude (EM wave intensity): S = E * B / μ_0
pub fn poynting_magnitude(e_field: f64, b_field: f64) -> f64 {
    e_field * b_field / constants::MU_0
}

// ── Advanced Magnetism ──

/// Solenoid magnetic field: B = μ₀nI
pub fn solenoid_field(mu0: f64, turns_per_length: f64, current: f64) -> f64 {
    mu0 * turns_per_length * current
}

/// Toroid magnetic field: B = μ₀NI/(2πr)
pub fn toroid_field(mu0: f64, total_turns: f64, current: f64, radius: f64) -> f64 {
    mu0 * total_turns * current / (2.0 * constants::PI * radius)
}

/// Magnetic flux: Φ = BA cos(θ)
pub fn magnetic_flux(b_field: f64, area: f64, angle: f64) -> f64 {
    b_field * area * angle.cos()
}

/// Magnetic energy density: u = B²/(2μ₀)
pub fn magnetic_energy_density(b_field: f64) -> f64 {
    b_field * b_field / (2.0 * constants::MU_0)
}

/// Mutual inductance of coaxial solenoids: M = μ₀n₁n₂AL
pub fn mutual_inductance_coaxial(mu0: f64, n1: f64, n2: f64, area: f64, length: f64) -> f64 {
    mu0 * n1 * n2 * area * length
}

/// Self-inductance of a solenoid: L = μ₀N²A/l
pub fn self_inductance_solenoid(mu0: f64, turns: f64, area: f64, length: f64) -> f64 {
    mu0 * turns * turns * area / length
}

/// Magnetic dipole moment: m = IA
pub fn magnetic_dipole_moment(current: f64, area: f64) -> f64 {
    current * area
}

/// Torque on a magnetic dipole: τ = mB sin(θ)
pub fn torque_on_dipole(moment: f64, b_field: f64, angle: f64) -> f64 {
    moment * b_field * angle.sin()
}

// ── AC Circuits ──

/// Capacitive reactance: Xc = 1/(2πfC)
pub fn capacitive_reactance(frequency: f64, capacitance: f64) -> f64 {
    1.0 / (2.0 * constants::PI * frequency * capacitance)
}

/// Inductive reactance: XL = 2πfL
pub fn inductive_reactance(frequency: f64, inductance: f64) -> f64 {
    2.0 * constants::PI * frequency * inductance
}

/// Impedance of a series RLC circuit: Z = √(R² + (XL - XC)²)
pub fn impedance_rlc_series(resistance: f64, inductive_reactance: f64, capacitive_reactance: f64) -> f64 {
    let diff = inductive_reactance - capacitive_reactance;
    (resistance * resistance + diff * diff).sqrt()
}

/// Resonant frequency of an LC circuit: f₀ = 1/(2π√(LC))
pub fn resonant_frequency_lc(inductance: f64, capacitance: f64) -> f64 {
    1.0 / (2.0 * constants::PI * (inductance * capacitance).sqrt())
}

/// Power factor: cos(φ) = R/Z
pub fn power_factor(resistance: f64, impedance: f64) -> f64 {
    resistance / impedance
}

/// RMS voltage: V_rms = V_peak/√2
pub fn rms_voltage(peak: f64) -> f64 {
    peak / 2.0_f64.sqrt()
}

/// RMS current: I_rms = I_peak/√2
pub fn rms_current(peak: f64) -> f64 {
    peak / 2.0_f64.sqrt()
}

/// Average AC power: P = V_rms × I_rms × cos(φ)
pub fn ac_power_average(vrms: f64, irms: f64, power_factor: f64) -> f64 {
    vrms * irms * power_factor
}

/// Quality factor of an RLC circuit: Q = (1/R)√(L/C)
pub fn quality_factor_rlc(inductance: f64, capacitance: f64, resistance: f64) -> f64 {
    (inductance / capacitance).sqrt() / resistance
}

/// Bandwidth of an RLC circuit: BW = f₀/Q
pub fn bandwidth_rlc(resonant_freq: f64, quality: f64) -> f64 {
    resonant_freq / quality
}

// ── Electromagnetic Wave Properties ──

/// EM wave speed in a medium: v = 1/√(εμ)
pub fn em_wave_speed(permittivity: f64, permeability: f64) -> f64 {
    1.0 / (permittivity * permeability).sqrt()
}

/// Refractive index from relative permittivity and permeability: n = √(ε_r × μ_r)
pub fn refractive_index_from_em(permittivity_rel: f64, permeability_rel: f64) -> f64 {
    (permittivity_rel * permeability_rel).sqrt()
}

/// Characteristic impedance of a medium: η = √(μ/ε)
pub fn characteristic_impedance(permeability: f64, permittivity: f64) -> f64 {
    (permeability / permittivity).sqrt()
}

/// Free-space impedance: η₀ = √(μ₀/ε₀) ≈ 377 Ω
pub fn free_space_impedance() -> f64 {
    (constants::MU_0 / constants::EPSILON_0).sqrt()
}

/// Total EM energy density: u = ε₀E²/2 + B²/(2μ₀)
pub fn energy_density_em(e_field: f64, b_field: f64) -> f64 {
    constants::EPSILON_0 * e_field * e_field / 2.0 + b_field * b_field / (2.0 * constants::MU_0)
}

/// Radiation intensity of a Hertzian dipole: I(θ) = (3P/(8π)) × sin²(θ)
pub fn radiation_intensity_dipole(power: f64, angle: f64) -> f64 {
    3.0 * power / (8.0 * constants::PI) * angle.sin().powi(2)
}

/// Larmor radiated power: P = q²a²/(6πε₀c³)
pub fn larmor_power(charge: f64, acceleration: f64) -> f64 {
    let c = constants::C;
    charge * charge * acceleration * acceleration / (6.0 * constants::PI * constants::EPSILON_0 * c * c * c)
}

// ── Transformer ──

/// Transformer secondary voltage: V₂ = V₁ × N₂/N₁
pub fn transformer_voltage(v_primary: f64, n_primary: f64, n_secondary: f64) -> f64 {
    v_primary * n_secondary / n_primary
}

/// Transformer secondary current: I₂ = I₁ × N₁/N₂
pub fn transformer_current(i_primary: f64, n_primary: f64, n_secondary: f64) -> f64 {
    i_primary * n_primary / n_secondary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b == 0.0 { return a.abs() < tol; }
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_coulomb_force() {
        let f = coulomb_force(1.0e-6, 1.0e-6, 1.0);
        assert!(approx_rel(f, 8.988e-3, 0.01));
    }

    #[test]
    fn test_ohms_law() {
        assert!(approx(ohms_law_voltage(2.0, 5.0), 10.0, 1e-9));
        assert!(approx(ohms_law_current(10.0, 5.0), 2.0, 1e-9));
    }

    #[test]
    fn test_resistors_series() {
        assert!(approx(resistors_series(&[10.0, 20.0, 30.0]), 60.0, 1e-9));
    }

    #[test]
    fn test_resistors_parallel() {
        let r = resistors_parallel(&[10.0, 10.0]);
        assert!(approx(r, 5.0, 1e-9));
    }

    #[test]
    fn test_capacitor_energy() {
        let u = capacitor_energy(1e-6, 100.0);
        assert!(approx(u, 0.005, 1e-6));
    }

    #[test]
    fn test_wavelength_frequency() {
        let wl = wavelength_from_frequency(5e14);
        let f = frequency_from_wavelength(wl);
        assert!(approx_rel(f, 5e14, 1e-6));
    }

    #[test]
    fn test_magnetic_field_wire() {
        let b = magnetic_field_wire(10.0, 0.05);
        assert!(approx_rel(b, 4e-5, 0.01));
    }

    #[test]
    fn test_lorentz_force() {
        let f = lorentz_force(
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        // E contribution: (1,0,0), v×B = (0,1,0)×(0,0,1) = (1,0,0)
        // Total: (2,0,0)
        assert!(approx(f.x, 2.0, 1e-9));
        assert!(approx(f.y, 0.0, 1e-9));
    }

    #[test]
    fn test_rc_charging() {
        // After 5 time constants, should be ~99.3% charged
        let v = rc_charging_voltage(10.0, 1000.0, 1e-3, 5.0);
        assert!(v > 9.9 && v < 10.0);
    }

    // ── Advanced Magnetism Tests ──

    #[test]
    fn test_solenoid_field() {
        let b = solenoid_field(constants::MU_0, 1000.0, 2.0);
        assert!(approx_rel(b, constants::MU_0 * 1000.0 * 2.0, 1e-9));
    }

    #[test]
    fn test_toroid_field() {
        let b = toroid_field(constants::MU_0, 500.0, 3.0, 0.1);
        let expected = constants::MU_0 * 500.0 * 3.0 / (2.0 * constants::PI * 0.1);
        assert!(approx_rel(b, expected, 1e-9));
    }

    #[test]
    fn test_magnetic_flux() {
        let phi = magnetic_flux(0.5, 0.02, 0.0);
        assert!(approx(phi, 0.01, 1e-9));

        let phi_angled = magnetic_flux(0.5, 0.02, constants::PI / 3.0);
        assert!(approx_rel(phi_angled, 0.005, 1e-6));
    }

    #[test]
    fn test_magnetic_energy_density() {
        let u = magnetic_energy_density(1.0);
        let expected = 1.0 / (2.0 * constants::MU_0);
        assert!(approx_rel(u, expected, 1e-9));
    }

    #[test]
    fn test_mutual_inductance_coaxial() {
        let m = mutual_inductance_coaxial(constants::MU_0, 100.0, 200.0, 0.01, 0.5);
        let expected = constants::MU_0 * 100.0 * 200.0 * 0.01 * 0.5;
        assert!(approx_rel(m, expected, 1e-9));
    }

    #[test]
    fn test_self_inductance_solenoid() {
        let l = self_inductance_solenoid(constants::MU_0, 1000.0, 0.01, 0.5);
        let expected = constants::MU_0 * 1e6 * 0.01 / 0.5;
        assert!(approx_rel(l, expected, 1e-9));
    }

    #[test]
    fn test_magnetic_dipole_moment() {
        assert!(approx(magnetic_dipole_moment(5.0, 0.02), 0.1, 1e-9));
    }

    #[test]
    fn test_torque_on_dipole() {
        let tau = torque_on_dipole(0.1, 0.5, constants::PI / 2.0);
        assert!(approx(tau, 0.05, 1e-9));

        let tau_zero = torque_on_dipole(0.1, 0.5, 0.0);
        assert!(approx(tau_zero, 0.0, 1e-9));
    }

    // ── AC Circuits Tests ──

    #[test]
    fn test_capacitive_reactance() {
        let xc = capacitive_reactance(60.0, 10e-6);
        let expected = 1.0 / (2.0 * constants::PI * 60.0 * 10e-6);
        assert!(approx_rel(xc, expected, 1e-9));
    }

    #[test]
    fn test_inductive_reactance() {
        let xl = inductive_reactance(60.0, 0.1);
        let expected = 2.0 * constants::PI * 60.0 * 0.1;
        assert!(approx_rel(xl, expected, 1e-9));
    }

    #[test]
    fn test_impedance_rlc_series() {
        let z = impedance_rlc_series(100.0, 50.0, 50.0);
        assert!(approx(z, 100.0, 1e-9));

        let z2 = impedance_rlc_series(3.0, 8.0, 4.0);
        assert!(approx(z2, 5.0, 1e-9));
    }

    #[test]
    fn test_resonant_frequency_lc() {
        let f0 = resonant_frequency_lc(1e-3, 1e-6);
        let expected = 1.0 / (2.0 * constants::PI * (1e-9_f64).sqrt());
        assert!(approx_rel(f0, expected, 1e-9));
    }

    #[test]
    fn test_power_factor() {
        assert!(approx(power_factor(50.0, 100.0), 0.5, 1e-9));
        assert!(approx(power_factor(100.0, 100.0), 1.0, 1e-9));
    }

    #[test]
    fn test_rms_voltage() {
        let vrms = rms_voltage(170.0);
        assert!(approx_rel(vrms, 170.0 / 2.0_f64.sqrt(), 1e-9));
    }

    #[test]
    fn test_rms_current() {
        let irms = rms_current(10.0);
        assert!(approx_rel(irms, 10.0 / 2.0_f64.sqrt(), 1e-9));
    }

    #[test]
    fn test_ac_power_average() {
        let p = ac_power_average(120.0, 5.0, 0.8);
        assert!(approx(p, 480.0, 1e-9));
    }

    #[test]
    fn test_quality_factor_rlc() {
        let q = quality_factor_rlc(0.1, 1e-6, 10.0);
        let expected = (0.1 / 1e-6_f64).sqrt() / 10.0;
        assert!(approx_rel(q, expected, 1e-9));
    }

    #[test]
    fn test_bandwidth_rlc() {
        assert!(approx(bandwidth_rlc(1000.0, 50.0), 20.0, 1e-9));
    }

    // ── Electromagnetic Wave Properties Tests ──

    #[test]
    fn test_em_wave_speed() {
        let v = em_wave_speed(constants::EPSILON_0, constants::MU_0);
        assert!(approx_rel(v, constants::C, 0.01));
    }

    #[test]
    fn test_refractive_index_from_em() {
        assert!(approx(refractive_index_from_em(1.0, 1.0), 1.0, 1e-9));
        assert!(approx(refractive_index_from_em(4.0, 1.0), 2.0, 1e-9));
    }

    #[test]
    fn test_characteristic_impedance() {
        let eta = characteristic_impedance(constants::MU_0, constants::EPSILON_0);
        assert!(approx_rel(eta, 377.0, 0.01));
    }

    #[test]
    fn test_free_space_impedance() {
        let eta0 = free_space_impedance();
        assert!(approx_rel(eta0, 377.0, 0.01));
    }

    #[test]
    fn test_energy_density_em() {
        let u = energy_density_em(100.0, 0.0);
        let expected = constants::EPSILON_0 * 100.0 * 100.0 / 2.0;
        assert!(approx_rel(u, expected, 1e-9));

        let u2 = energy_density_em(0.0, 0.5);
        let expected2 = 0.25 / (2.0 * constants::MU_0);
        assert!(approx_rel(u2, expected2, 1e-9));
    }

    #[test]
    fn test_radiation_intensity_dipole() {
        let i_at_90 = radiation_intensity_dipole(100.0, constants::PI / 2.0);
        assert!(approx_rel(i_at_90, 3.0 * 100.0 / (8.0 * constants::PI), 1e-6));

        let i_at_0 = radiation_intensity_dipole(100.0, 0.0);
        assert!(approx(i_at_0, 0.0, 1e-9));
    }

    #[test]
    fn test_larmor_power() {
        let p = larmor_power(constants::E_CHARGE, 1e15);
        let c = constants::C;
        let expected = constants::E_CHARGE * constants::E_CHARGE * 1e30
            / (6.0 * constants::PI * constants::EPSILON_0 * c * c * c);
        assert!(approx_rel(p, expected, 1e-6));
    }

    // ── Transformer Tests ──

    #[test]
    fn test_transformer_voltage() {
        let v2 = transformer_voltage(120.0, 100.0, 500.0);
        assert!(approx(v2, 600.0, 1e-9));

        let v2_step_down = transformer_voltage(120.0, 500.0, 100.0);
        assert!(approx(v2_step_down, 24.0, 1e-9));
    }

    #[test]
    fn test_transformer_current() {
        let i2 = transformer_current(10.0, 100.0, 500.0);
        assert!(approx(i2, 2.0, 1e-9));
    }
}
