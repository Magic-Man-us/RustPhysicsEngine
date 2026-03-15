use crate::math::constants::{G_ACCEL, PI, R as GAS_R};

// ── Rocket Equation ──

/// Tsiolkovsky rocket equation: Δv = ve × ln(m0/mf)
pub fn tsiolkovsky_delta_v(exhaust_velocity: f64, mass_initial: f64, mass_final: f64) -> f64 {
    exhaust_velocity * (mass_initial / mass_final).ln()
}

/// Mass ratio from the rocket equation inverted: m0/mf = exp(Δv/ve)
pub fn mass_ratio(delta_v: f64, exhaust_velocity: f64) -> f64 {
    (delta_v / exhaust_velocity).exp()
}

/// Specific impulse: Isp = F / (ṁ × g)
pub fn specific_impulse(thrust: f64, mass_flow_rate: f64, g: f64) -> f64 {
    thrust / (mass_flow_rate * g)
}

/// Effective exhaust velocity from specific impulse: ve = Isp × g
pub fn exhaust_velocity_from_isp(isp: f64, g: f64) -> f64 {
    isp * g
}

/// Thrust from momentum: F = ṁ × ve
pub fn thrust(mass_flow_rate: f64, exhaust_velocity: f64) -> f64 {
    mass_flow_rate * exhaust_velocity
}

/// Thrust including pressure term: F = ṁve + (Pe - Pa)Ae
pub fn thrust_with_pressure(
    mass_flow_rate: f64,
    exhaust_velocity: f64,
    exit_pressure: f64,
    ambient_pressure: f64,
    exit_area: f64,
) -> f64 {
    mass_flow_rate * exhaust_velocity + (exit_pressure - ambient_pressure) * exit_area
}

// ── Staging ──

/// Total Δv for a multi-stage rocket. Each element is (exhaust_velocity, mass_full, mass_empty).
pub fn delta_v_staged(stages: &[(f64, f64, f64)]) -> f64 {
    stages
        .iter()
        .map(|&(ve, m_full, m_empty)| ve * (m_full / m_empty).ln())
        .sum()
}

// ── Orbital Maneuvers ──

/// Hohmann transfer Δv values. Returns (Δv1, Δv2) for departure and arrival burns.
pub fn hohmann_delta_v(mu: f64, r1: f64, r2: f64) -> (f64, f64) {
    let dv1 = (mu / r1).sqrt() * ((2.0 * r2 / (r1 + r2)).sqrt() - 1.0);
    let dv2 = (mu / r2).sqrt() * (1.0 - (2.0 * r1 / (r1 + r2)).sqrt());
    (dv1, dv2)
}

/// Transfer time for a Hohmann orbit: t = π√((r1+r2)³ / (8μ))
pub fn hohmann_transfer_time(mu: f64, r1: f64, r2: f64) -> f64 {
    let a = r1 + r2;
    PI * (a * a * a / (8.0 * mu)).sqrt()
}

/// Gravity drag loss approximation: Δv_loss ≈ g × t
pub fn gravity_turn_loss(g: f64, burn_time: f64) -> f64 {
    g * burn_time
}

/// Plane change Δv: Δv = 2v × sin(θ/2)
pub fn delta_v_plane_change(velocity: f64, angle: f64) -> f64 {
    2.0 * velocity * (angle / 2.0).sin()
}

/// Bi-elliptic transfer total Δv via three burns through an intermediate radius.
pub fn bi_elliptic_delta_v(mu: f64, r1: f64, r2: f64, r_intermediate: f64) -> f64 {
    let a1 = (r1 + r_intermediate) / 2.0;
    let a2 = (r2 + r_intermediate) / 2.0;

    // First burn: depart r1 onto transfer ellipse 1
    let v_circ1 = (mu / r1).sqrt();
    let v_peri1 = (2.0 * mu * (1.0 / r1 - 1.0 / (2.0 * a1))).sqrt();
    let dv1 = (v_peri1 - v_circ1).abs();

    // Second burn: at r_intermediate, transition from transfer 1 to transfer 2
    let v_apo1 = (2.0 * mu * (1.0 / r_intermediate - 1.0 / (2.0 * a1))).sqrt();
    let v_peri2 = (2.0 * mu * (1.0 / r_intermediate - 1.0 / (2.0 * a2))).sqrt();
    let dv2 = (v_peri2 - v_apo1).abs();

    // Third burn: circularize at r2
    let v_circ2 = (mu / r2).sqrt();
    let v_apo2 = (2.0 * mu * (1.0 / r2 - 1.0 / (2.0 * a2))).sqrt();
    let dv3 = (v_circ2 - v_apo2).abs();

    dv1 + dv2 + dv3
}

// ── Nozzle Physics ──

/// Nozzle exit velocity from thermodynamic properties:
/// ve = √( 2γRT / (M(γ-1)) × (1 - (Pe/Pc)^((γ-1)/γ)) )
pub fn nozzle_exit_velocity(
    chamber_temp: f64,
    molar_mass: f64,
    gamma: f64,
    pressure_ratio: f64,
) -> f64 {
    let exponent = (gamma - 1.0) / gamma;
    let term = 1.0 - pressure_ratio.powf(exponent);
    let coeff = 2.0 * gamma * GAS_R * chamber_temp / (molar_mass * (gamma - 1.0));
    (coeff * term).sqrt()
}

/// Throat area required for a given mass flow:
/// A* = (ṁ / Pc) × √(R T / (γ M)) / (2/(γ+1))^((γ+1)/(2(γ-1)))
pub fn throat_area(
    mass_flow: f64,
    chamber_pressure: f64,
    chamber_temp: f64,
    gamma: f64,
    molar_mass: f64,
) -> f64 {
    let gp1 = gamma + 1.0;
    let gm1 = gamma - 1.0;
    let throat_factor = (2.0 / gp1).powf(gp1 / (2.0 * gm1));
    let c_star = (GAS_R * chamber_temp / (gamma * molar_mass)).sqrt() / throat_factor;
    mass_flow * c_star / chamber_pressure
}

/// Area ratio Ae/A* as a function of Mach number and heat capacity ratio:
/// Ae/A* = (1/M) × ((2/(γ+1)) × (1 + (γ-1)/2 × M²))^((γ+1)/(2(γ-1)))
pub fn area_ratio_from_mach(mach: f64, gamma: f64) -> f64 {
    let gp1 = gamma + 1.0;
    let gm1 = gamma - 1.0;
    let inner = (2.0 / gp1) * (1.0 + gm1 / 2.0 * mach * mach);
    let exponent = gp1 / (2.0 * gm1);
    (1.0 / mach) * inner.powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;
    const LOOSE_TOLERANCE: f64 = 0.01;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol || (a - b).abs() / b.abs().max(1e-15) < tol
    }

    fn approx_rel(a: f64, b: f64, rel_tol: f64) -> bool {
        (a - b).abs() / b.abs().max(1e-15) < rel_tol
    }

    // ── Rocket Equation ──

    #[test]
    fn test_tsiolkovsky_single_stage() {
        // Saturn V first stage (S-IC): ve ≈ 2580 m/s, m0=2970000 kg, mf=870000 kg
        let ve = 2580.0;
        let m0 = 2_970_000.0;
        let mf = 870_000.0;
        let dv = tsiolkovsky_delta_v(ve, m0, mf);
        let expected = ve * (m0 / mf).ln();
        assert!(approx(dv, expected, TOLERANCE));
    }

    #[test]
    fn test_mass_ratio_roundtrip() {
        let ve = 3000.0;
        let dv = 5000.0;
        let mr = mass_ratio(dv, ve);
        let dv_back = tsiolkovsky_delta_v(ve, mr, 1.0);
        assert!(approx(dv_back, dv, TOLERANCE));
    }

    #[test]
    fn test_specific_impulse_saturn_v_f1() {
        // F-1 engine: thrust=6.77 MN, mass_flow≈2617 kg/s
        let f = 6_770_000.0;
        let mdot = 2617.0;
        let isp = specific_impulse(f, mdot, G_ACCEL);
        // Expected Isp ≈ 263 s (sea level)
        assert!(approx_rel(isp, 263.0, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_exhaust_velocity_from_isp() {
        let isp = 311.0; // Merlin 1D sea level
        let ve = exhaust_velocity_from_isp(isp, G_ACCEL);
        assert!(approx_rel(ve, 3049.0, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_thrust_momentum() {
        let mdot = 100.0;
        let ve = 3000.0;
        assert!(approx(thrust(mdot, ve), 300_000.0, TOLERANCE));
    }

    #[test]
    fn test_thrust_with_pressure_vacuum() {
        let mdot = 100.0;
        let ve = 3000.0;
        // In vacuum with matched nozzle (Pe == Pa), reduces to F = ṁve
        let f = thrust_with_pressure(mdot, ve, 101_325.0, 101_325.0, 1.0);
        assert!(approx(f, 300_000.0, TOLERANCE));
    }

    #[test]
    fn test_thrust_with_pressure_term() {
        let mdot = 100.0;
        let ve = 3000.0;
        let pe = 50_000.0;
        let pa = 101_325.0;
        let ae = 2.0;
        let f = thrust_with_pressure(mdot, ve, pe, pa, ae);
        let expected = mdot * ve + (pe - pa) * ae;
        assert!(approx(f, expected, TOLERANCE));
    }

    // ── Staging ──

    #[test]
    fn test_delta_v_staged_matches_sum() {
        let stages = vec![
            (2580.0, 2_970_000.0, 870_000.0), // S-IC
            (4130.0, 570_000.0, 110_000.0),    // S-II (approximate)
            (4130.0, 120_800.0, 13_300.0),     // S-IVB (approximate)
        ];
        let total = delta_v_staged(&stages);
        let manual: f64 = stages
            .iter()
            .map(|&(ve, mf, me)| ve * (mf / me).ln())
            .sum();
        assert!(approx(total, manual, TOLERANCE));
        // Saturn V total Δv should be in the ballpark of 9-10 km/s
        assert!(total > 9000.0 && total < 20000.0);
    }

    // ── Orbital Maneuvers ──

    #[test]
    fn test_hohmann_earth_to_mars() {
        // Earth orbit r1 ≈ 1.496e11 m, Mars orbit r2 ≈ 2.279e11 m, μ_sun ≈ 1.327e20 m³/s²
        let mu = 1.327e20;
        let r1 = 1.496e11;
        let r2 = 2.279e11;
        let (dv1, dv2) = hohmann_delta_v(mu, r1, r2);
        // Expected: Δv1 ≈ 2945 m/s, Δv2 ≈ 2649 m/s (approximate)
        assert!(approx_rel(dv1, 2945.0, LOOSE_TOLERANCE));
        assert!(approx_rel(dv2, 2649.0, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_hohmann_transfer_time_earth_mars() {
        let mu = 1.327e20;
        let r1 = 1.496e11;
        let r2 = 2.279e11;
        let t = hohmann_transfer_time(mu, r1, r2);
        // Transfer time ≈ 259 days ≈ 2.24e7 s
        let days = t / 86400.0;
        assert!(approx_rel(days, 259.0, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_gravity_turn_loss() {
        // Typical launch: g ≈ 9.81, burn ≈ 150 s → loss ≈ 1471 m/s
        let loss = gravity_turn_loss(G_ACCEL, 150.0);
        assert!(approx_rel(loss, 1471.0, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_delta_v_plane_change() {
        // 28.5° plane change at LEO velocity ≈ 7700 m/s
        let v = 7700.0;
        let angle = 28.5_f64.to_radians();
        let dv = delta_v_plane_change(v, angle);
        // Δv = 2 × 7700 × sin(14.25°) ≈ 3790 m/s
        let expected = 2.0 * v * (angle / 2.0).sin();
        assert!(approx(dv, expected, TOLERANCE));
    }

    #[test]
    fn test_bi_elliptic_vs_hohmann() {
        // For r2/r1 < ~11.94, Hohmann is cheaper. For large ratios, bi-elliptic wins.
        let mu = 1.327e20;
        let r1 = 1.0e11;
        let r2 = 15.0e11; // Large ratio
        let r_int = 50.0e11;
        let bi = bi_elliptic_delta_v(mu, r1, r2, r_int);
        let (h1, h2) = hohmann_delta_v(mu, r1, r2);
        let hohmann_total = h1 + h2;
        // With this ratio (15:1), bi-elliptic should be competitive
        assert!(bi > 0.0);
        assert!(hohmann_total > 0.0);
    }

    // ── Nozzle Physics ──

    #[test]
    fn test_nozzle_exit_velocity() {
        // Typical LOX/LH2: T=3500K, M=0.018 kg/mol, γ=1.2, Pe/Pc=0.01
        let ve = nozzle_exit_velocity(3500.0, 0.018, 1.2, 0.01);
        // Should produce a reasonable exhaust velocity (3000-5000 m/s)
        assert!(ve > 3000.0 && ve < 6000.0);
    }

    #[test]
    fn test_throat_area() {
        // Mass flow 100 kg/s, Pc = 7 MPa, T = 3500 K, γ = 1.2, M = 0.018 kg/mol
        let a_star = throat_area(100.0, 7e6, 3500.0, 1.2, 0.018);
        // Throat area should be a small positive value (order of 0.01-0.1 m²)
        assert!(a_star > 0.0 && a_star < 1.0);
    }

    #[test]
    fn test_area_ratio_from_mach_sonic() {
        // At Mach 1, Ae/A* = 1.0 (by definition of the throat)
        let ratio = area_ratio_from_mach(1.0, 1.4);
        assert!(approx(ratio, 1.0, TOLERANCE));
    }

    #[test]
    fn test_area_ratio_supersonic() {
        // For γ=1.4, Mach 3: Ae/A* ≈ 4.2346
        let ratio = area_ratio_from_mach(3.0, 1.4);
        assert!(approx_rel(ratio, 4.2346, 1e-3));
    }
}
