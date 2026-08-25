//! Biophysics: the elementary membrane, transport and mechanics relations
//! here, with the population-scale models in submodules.
//!
//! The roadmap calls this area `bio`; it lives under the existing
//! `biophysics` module instead, so that there is one home for the subject
//! rather than two.

pub mod epidemiology;
pub mod phylo;
pub mod population;
pub mod seq_align;

use crate::error::GeomError;

/// Integrates a first-order system with adaptive Runge-Kutta and a
/// step-doubling error estimate.
///
/// The population and epidemic models here are not stiff in the way a
/// chemical network is -- their rates are all of the same order -- so an
/// explicit method is the right choice, and the adaptivity is there to
/// resolve the peaks and turning points where the curvature concentrates.
pub(crate) fn integrate_adaptive(
    derivative: impl Fn(&[f64]) -> Vec<f64>,
    y0: &[f64],
    t_end: f64,
    rtol: f64,
) -> Result<Vec<(f64, Vec<f64>)>, GeomError> {
    if !(t_end > 0.0) || !(rtol > 0.0) || rtol >= 1.0 {
        return Err(GeomError::InvalidArgument("integrate: bad parameters"));
    }
    let n = y0.len();
    let rk4 = |y: &[f64], h: f64| -> Vec<f64> {
        let k1 = derivative(y);
        let mid1: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
        let k2 = derivative(&mid1);
        let mid2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k2[i]).collect();
        let k3 = derivative(&mid2);
        let end: Vec<f64> = (0..n).map(|i| y[i] + h * k3[i]).collect();
        let k4 = derivative(&end);
        (0..n)
            .map(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .collect()
    };
    let mut out = vec![(0.0, y0.to_vec())];
    let mut t = 0.0;
    let mut y = y0.to_vec();
    let mut h = (t_end * 1e-4).min(0.1);
    let smallest = t_end * 1e-12;
    while t < t_end {
        // The accumulated time overshoots `t_end` by a rounding residue --
        // here of order 1e-14 on a span of 235 -- and the loop condition is
        // still true. That leftover is not a step to be taken; treating it
        // as one and then finding it smaller than the working precision
        // reported a numerical breakdown for an integration that had in
        // fact finished. The two conditions are distinct: an interval too
        // short to bother with, and an error controller forced into a step
        // it cannot take.
        let remaining = t_end - t;
        if remaining <= smallest {
            break;
        }
        h = h.min(remaining);
        if h < smallest {
            return Err(GeomError::Degenerate("the step collapsed below the working precision"));
        }
        let coarse = rk4(&y, h);
        let fine = rk4(&rk4(&y, 0.5 * h), 0.5 * h);
        let error = (0..n)
            .map(|i| (coarse[i] - fine[i]).abs() / fine[i].abs().max(1e-3))
            .fold(0.0, f64::max);
        if error <= rtol {
            // Richardson: RK4's error is fourth order, so the two-step
            // result plus a fifteenth of the gap is fifth order.
            y = (0..n).map(|i| fine[i] + (fine[i] - coarse[i]) / 15.0).collect();
            t += h;
            out.push((t, y.clone()));
        }
        let growth = if error > 0.0 { 0.9 * (rtol / error).powf(0.2) } else { 4.0 };
        h *= growth.clamp(0.2, 4.0);
        if out.len() > 500_000 {
            return Err(GeomError::Degenerate("the integration did not reach the end time"));
        }
    }
    Ok(out)
}


use crate::math::constants;
use crate::chemistry::FARADAY;

// ── Constants ──

/// Faraday constant for biophysics context (C/mol), re-exported from chemistry
pub const FARADAY_BIO: f64 = FARADAY;
/// Universal gas constant (J/(mol·K)), re-exported for convenience
pub const GAS_CONSTANT: f64 = constants::R;
/// Standard human body temperature (K), 37°C
pub const BODY_TEMP: f64 = 310.15;
/// Typical resting membrane potential for a neuron (V)
const RESTING_POTENTIAL_V: f64 = -70e-3;

// ── Membrane Potentials ──

/// Nernst equation: E = (RT/(zF)) × ln(c_out/c_in)
pub fn nernst_potential(temperature: f64, z: f64, c_out: f64, c_in: f64) -> f64 {
    assert!(z != 0.0, "valence z must be nonzero");
    assert!(c_in > 0.0, "c_in must be positive");
    assert!(c_out > 0.0, "c_out must be positive");
    (constants::R * temperature / (z * FARADAY)) * (c_out / c_in).ln()
}

/// Goldman-Hodgkin-Katz voltage equation for K⁺, Na⁺, and Cl⁻.
/// Vm = (RT/F) × ln((Pk[K]o + Pna[Na]o + Pcl[Cl]i) / (Pk[K]i + Pna[Na]i + Pcl[Cl]o))
pub fn goldman_potential(
    temperature: f64,
    pk: f64,
    pna: f64,
    pcl: f64,
    k_out: f64,
    k_in: f64,
    na_out: f64,
    na_in: f64,
    cl_out: f64,
    cl_in: f64,
) -> f64 {
    let numerator = pk * k_out + pna * na_out + pcl * cl_in;
    let denominator = pk * k_in + pna * na_in + pcl * cl_out;
    assert!(denominator > 0.0, "denominator (Pk*K_in + Pna*Na_in + Pcl*Cl_out) must be positive");
    assert!(numerator > 0.0, "numerator (Pk*K_out + Pna*Na_out + Pcl*Cl_in) must be positive");
    (constants::R * temperature / FARADAY) * (numerator / denominator).ln()
}

/// Typical resting membrane potential for a neuron: -70 mV
pub fn resting_membrane_potential_typical() -> f64 {
    RESTING_POTENTIAL_V
}

// ── Enzyme Kinetics ──

/// Michaelis-Menten kinetics: v = Vmax × [S] / (Km + [S])
pub fn michaelis_menten(vmax: f64, km: f64, substrate: f64) -> f64 {
    vmax * substrate / (km + substrate)
}

/// Competitive inhibition: v = Vmax × [S] / (Km(1 + [I]/Ki) + [S])
pub fn michaelis_menten_inhibited(
    vmax: f64,
    km: f64,
    substrate: f64,
    inhibitor: f64,
    ki: f64,
) -> f64 {
    assert!(ki > 0.0, "ki must be positive");
    vmax * substrate / (km * (1.0 + inhibitor / ki) + substrate)
}

/// Lineweaver-Burk transform: returns (1/[S], 1/v) for double-reciprocal plot
pub fn lineweaver_burk(vmax: f64, km: f64, substrate: f64) -> (f64, f64) {
    assert!(substrate > 0.0, "substrate must be positive");
    let v = michaelis_menten(vmax, km, substrate);
    assert!(v > 0.0, "velocity must be positive");
    (1.0 / substrate, 1.0 / v)
}

/// Hill equation for cooperative binding: v = Vmax × [S]^n / (K^n + [S]^n)
pub fn hill_equation(vmax: f64, k: f64, substrate: f64, n: f64) -> f64 {
    let s_n = substrate.powf(n);
    let k_n = k.powf(n);
    vmax * s_n / (k_n + s_n)
}

/// Derive Hill coefficient from two (substrate, velocity) data points.
/// n = log((v1/(Vmax-v1)) / (v2/(Vmax-v2))) / log(s1/s2)
pub fn hill_coefficient_from_data(s1: f64, v1: f64, s2: f64, v2: f64, vmax: f64) -> f64 {
    assert!(vmax > v1, "vmax must be greater than v1");
    assert!(vmax > v2, "vmax must be greater than v2");
    assert!(s1 > 0.0, "s1 must be positive");
    assert!(s2 > 0.0, "s2 must be positive");
    assert!((s1 - s2).abs() > 0.0, "s1 and s2 must differ");
    let ratio1 = v1 / (vmax - v1);
    let ratio2 = v2 / (vmax - v2);
    (ratio1 / ratio2).ln() / (s1 / s2).ln()
}

// ── Population Dynamics ──

/// Exponential growth: N = N₀ × e^(rt)
pub fn exponential_growth(n0: f64, rate: f64, time: f64) -> f64 {
    n0 * (rate * time).exp()
}

/// Logistic growth: N = K / (1 + ((K - N₀)/N₀) × e^(-rt))
pub fn logistic_growth(n0: f64, k: f64, r: f64, time: f64) -> f64 {
    assert!(n0 > 0.0, "initial population n0 must be positive");
    k / (1.0 + ((k - n0) / n0) * (-r * time).exp())
}

/// Doubling time: td = ln(2) / r
pub fn doubling_time_population(rate: f64) -> f64 {
    assert!(rate > 0.0, "growth rate must be positive");
    2.0_f64.ln() / rate
}

/// Lotka-Volterra prey rate: dx/dt = αx - βxy
pub fn lotka_volterra_prey(prey: f64, predator: f64, alpha: f64, beta: f64) -> f64 {
    alpha * prey - beta * prey * predator
}

/// Lotka-Volterra predator rate: dy/dt = δxy - γy
pub fn lotka_volterra_predator(prey: f64, predator: f64, delta: f64, gamma: f64) -> f64 {
    delta * prey * predator - gamma * predator
}

// ── Hemodynamics ──

/// Cardiac output: CO = SV × HR (L/min when SV in L/beat and HR in bpm)
pub fn cardiac_output(stroke_volume: f64, heart_rate: f64) -> f64 {
    stroke_volume * heart_rate
}

/// Mean arterial pressure: MAP = DBP + (SBP - DBP)/3
pub fn mean_arterial_pressure(systolic: f64, diastolic: f64) -> f64 {
    diastolic + (systolic - diastolic) / 3.0
}

/// Vascular resistance: R = ΔP / Q
pub fn vascular_resistance(pressure_drop: f64, flow: f64) -> f64 {
    assert!(flow != 0.0, "flow must be nonzero");
    pressure_drop / flow
}

/// Poiseuille flow in a vessel: Q = πr⁴ΔP / (8μL)
pub fn poiseuille_blood_flow(radius: f64, pressure_drop: f64, viscosity: f64, length: f64) -> f64 {
    assert!(viscosity > 0.0, "viscosity must be positive");
    assert!(length > 0.0, "length must be positive");
    constants::PI * radius.powi(4) * pressure_drop / (8.0 * viscosity * length)
}

// ── Dose-Response ──

/// Sigmoid function: f = 1 / (1 + exp(-slope × (x - x50)))
pub fn sigmoid(x: f64, x50: f64, slope: f64) -> f64 {
    1.0 / (1.0 + (-slope * (x - x50)).exp())
}

/// LD50 probit model: probability = sigmoid(ln(dose), ln(ld50), slope)
pub fn ld50_probit(dose: f64, ld50: f64, slope: f64) -> f64 {
    assert!(dose > 0.0, "dose must be positive");
    assert!(ld50 > 0.0, "ld50 must be positive");
    sigmoid(dose.ln(), ld50.ln(), slope)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b.abs() < 1e-15 {
            return a.abs() < tol;
        }
        ((a - b) / b).abs() < tol
    }

    // ── Membrane Potentials ──

    #[test]
    fn test_nernst_potential_potassium() {
        // K+ at body temp: z=1, [K]out=5mM, [K]in=140mM
        let e = nernst_potential(BODY_TEMP, 1.0, 5.0, 140.0);
        // Should be approximately -90 mV
        assert!(approx_rel(e, -0.0902, 0.02));
    }

    #[test]
    fn test_nernst_potential_monovalent_37c() {
        let e = nernst_potential(BODY_TEMP, 1.0, 5.0, 140.0);
        assert!(approx_rel(e, -0.089_058, 1e-3));
    }

    #[test]
    fn test_goldman_potential() {
        // Typical neuron values: Pk:Pna:Pcl = 1:0.04:0.45
        let vm = goldman_potential(
            BODY_TEMP,
            1.0, 0.04, 0.45,   // permeabilities
            5.0, 140.0,         // K out/in
            145.0, 12.0,        // Na out/in
            120.0, 4.0,         // Cl out/in
        );
        // Should be near -70 mV
        assert!(vm < -0.050 && vm > -0.090);
    }

    #[test]
    fn test_resting_membrane_potential() {
        assert!(approx(resting_membrane_potential_typical(), -0.070));
    }

    // ── Enzyme Kinetics ──

    #[test]
    fn test_michaelis_menten_half_vmax() {
        // When [S] = Km, v = Vmax/2
        let v = michaelis_menten(100.0, 10.0, 10.0);
        assert!(approx(v, 50.0));
    }

    #[test]
    fn test_michaelis_menten_saturation() {
        // When [S] >> Km, v approaches Vmax
        let v = michaelis_menten(100.0, 10.0, 10_000.0);
        assert!(approx_rel(v, 100.0, 0.01));
    }

    #[test]
    fn test_michaelis_menten_inhibited() {
        let v_normal = michaelis_menten(100.0, 10.0, 20.0);
        let v_inhibited = michaelis_menten_inhibited(100.0, 10.0, 20.0, 10.0, 10.0);
        // Competitive inhibition reduces velocity
        assert!(v_inhibited < v_normal);
        // With [I]=Ki, apparent Km doubles: v = Vmax*[S]/(2Km+[S]) = 100*20/(20+20) = 50
        assert!(approx(v_inhibited, 50.0));
    }

    #[test]
    fn test_lineweaver_burk() {
        let (inv_s, inv_v) = lineweaver_burk(100.0, 10.0, 20.0);
        assert!(approx(inv_s, 0.05));
        assert!(approx(inv_v, 0.015));
    }

    #[test]
    fn test_hill_equation_n1_matches_mm() {
        // Hill with n=1 should equal Michaelis-Menten
        let v_hill = hill_equation(100.0, 10.0, 20.0, 1.0);
        let v_mm = michaelis_menten(100.0, 10.0, 20.0);
        assert!(approx(v_hill, v_mm));
    }

    #[test]
    fn test_hill_equation_cooperative() {
        // With n>1, sigmoidal curve is steeper at half-max
        let v_n1 = hill_equation(100.0, 10.0, 10.0, 1.0);
        let v_n4 = hill_equation(100.0, 10.0, 10.0, 4.0);
        // At [S]=K, both should give Vmax/2
        assert!(approx(v_n1, 50.0));
        assert!(approx(v_n4, 50.0));
    }

    #[test]
    fn test_hill_coefficient_from_data() {
        // Generate two points from Hill equation with known n=2
        let vmax = 100.0;
        let k = 10.0;
        let n_expected = 2.0;
        let s1 = 5.0;
        let s2 = 20.0;
        let v1 = hill_equation(vmax, k, s1, n_expected);
        let v2 = hill_equation(vmax, k, s2, n_expected);
        let n_calc = hill_coefficient_from_data(s1, v1, s2, v2, vmax);
        assert!(approx_rel(n_calc, n_expected, 1e-6));
    }

    // ── Population Dynamics ──

    #[test]
    fn test_exponential_growth() {
        let n = exponential_growth(100.0, 0.1, 10.0);
        assert!(approx_rel(n, 271.828_183, 1e-4));
    }

    #[test]
    fn test_logistic_growth_approaches_capacity() {
        let n = logistic_growth(10.0, 1000.0, 0.5, 100.0);
        // After long time, should approach carrying capacity K
        assert!(approx_rel(n, 1000.0, 0.01));
    }

    #[test]
    fn test_logistic_growth_initial() {
        let n = logistic_growth(100.0, 1000.0, 0.5, 0.0);
        assert!(approx(n, 100.0));
    }

    #[test]
    fn test_doubling_time() {
        let td = doubling_time_population(0.1);
        assert!(approx_rel(td, 6.9315, 1e-3));
    }

    #[test]
    fn test_lotka_volterra_prey() {
        // With no predators, prey grows exponentially
        let rate = lotka_volterra_prey(100.0, 0.0, 1.5, 0.1);
        assert!(approx(rate, 150.0));
    }

    #[test]
    fn test_lotka_volterra_predator() {
        // With no prey, predators decline
        let rate = lotka_volterra_predator(0.0, 50.0, 0.01, 0.5);
        assert!(approx(rate, -25.0));
    }

    // ── Hemodynamics ──

    #[test]
    fn test_cardiac_output() {
        // 70 mL/beat × 72 bpm = 5040 mL/min
        let co = cardiac_output(0.070, 72.0);
        assert!(approx(co, 5.04));
    }

    #[test]
    fn test_mean_arterial_pressure() {
        // 120/80 mmHg → MAP = 80 + 40/3 ≈ 93.33
        let map = mean_arterial_pressure(120.0, 80.0);
        assert!(approx_rel(map, 93.333, 1e-3));
    }

    #[test]
    fn test_vascular_resistance() {
        let r = vascular_resistance(10.0, 5.0);
        assert!(approx(r, 2.0));
    }

    #[test]
    fn test_poiseuille_blood_flow() {
        let q = poiseuille_blood_flow(0.01, 1000.0, 0.003, 0.1);
        assert!(approx_rel(q, 0.013_09, 1e-3));
    }

    // ── Dose-Response ──

    #[test]
    fn test_sigmoid_midpoint() {
        // At x = x50, sigmoid = 0.5
        let f = sigmoid(5.0, 5.0, 1.0);
        assert!(approx(f, 0.5));
    }

    #[test]
    fn test_sigmoid_extremes() {
        // Far above x50 → ~1, far below → ~0
        let high = sigmoid(100.0, 5.0, 1.0);
        let low = sigmoid(-100.0, 5.0, 1.0);
        assert!(high > 0.999);
        assert!(low < 0.001);
    }

    #[test]
    fn test_ld50_probit_at_ld50() {
        // At dose = LD50, probability should be 0.5
        let p = ld50_probit(50.0, 50.0, 1.0);
        assert!(approx(p, 0.5));
    }

    #[test]
    fn test_approx_rel_near_zero_b() {
        assert!(approx_rel(1e-16, 0.0, 1e-6));
        assert!(!approx_rel(1.0, 0.0, 0.5));
    }
}
