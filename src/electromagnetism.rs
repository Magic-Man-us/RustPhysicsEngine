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

/// Energy of a photon: E = h * f
pub fn photon_energy(frequency: f64) -> f64 {
    constants::H * frequency
}

/// Photon energy from wavelength: E = h * c / λ
pub fn photon_energy_from_wavelength(wavelength: f64) -> f64 {
    constants::H * constants::C / wavelength
}

/// Poynting vector magnitude (EM wave intensity): S = E * B / μ_0
pub fn poynting_magnitude(e_field: f64, b_field: f64) -> f64 {
    e_field * b_field / constants::MU_0
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
    fn test_photon_energy() {
        // Visible light ~5e14 Hz
        let e = photon_energy(5e14);
        assert!(approx_rel(e, 3.313e-19, 0.01));
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
}
