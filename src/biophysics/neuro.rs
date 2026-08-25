//! Computational neuroscience: single neurons, spike trains, synapses and
//! the small networks built from them.
//!
//! # Units
//!
//! The conductance-based models use the squid axon's units throughout:
//! millivolts, milliseconds, microfarads and microamps per square
//! centimetre, and millisiemens per square centimetre. A rate is therefore
//! a count per millisecond unless a function says otherwise, and the
//! spike frequencies reported by the F-I curves are converted to hertz
//! where that is the useful number. The reduced models -- FitzHugh-Nagumo
//! and the drift-diffusion process -- carry no units at all.
//!
//! # What a spike is here
//!
//! Every model that fires does so by one of two mechanisms, and the
//! difference decides what can be asked of it. Hodgkin-Huxley, Morris-Lecar
//! and FitzHugh-Nagumo generate the spike from their own dynamics: the
//! upstroke is a solution of the equations and the threshold is not a
//! parameter but an emergent property of the vector field. The
//! integrate-and-fire family -- LIF, Izhikevich, AdEx -- *stipulates* the
//! spike: the equations describe only the approach, and a rule replaces
//! the voltage when it crosses a number. The second kind is far cheaper
//! and reproduces firing statistics well; it has no answer to questions
//! about the spike's shape, because the shape was never computed.
//!
//! Spikes are detected in a trace by an upward crossing of a fixed level,
//! which is the right test for a model whose spikes are tall and brief.
//!
//! # Equilibrium potentials
//!
//! [`crate::biophysics::nernst_potential`] and
//! [`crate::biophysics::goldman_potential`] already provide the reversal
//! potentials these models take as constants, and are not repeated here.

use crate::error::GeomError;
use crate::linalg::Matrix;
use crate::monte_carlo::Rng;
use crate::numerical::ode::rk4_step_vec;

/// The level a trace must cross upward to count as a spike, in millivolts.
const SPIKE_LEVEL: f64 = 0.0;

/// The times at which a voltage trace crosses `level` going up.
///
/// The crossing time is interpolated between the bracketing samples, so
/// the answer does not jump in steps of `dt`.
fn upward_crossings(trace: &[(f64, f64)], level: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for pair in trace.windows(2) {
        let (t0, v0) = pair[0];
        let (t1, v1) = pair[1];
        if v0 < level && v1 >= level {
            let fraction = (level - v0) / (v1 - v0);
            out.push(t0 + fraction * (t1 - t0));
        }
    }
    out
}

/// The times at which a voltage trace crosses `level` upward, interpolated
/// between samples.
///
/// The level is the caller's because the models here peak at very
/// different voltages: Hodgkin-Huxley and Izhikevich overshoot well past
/// zero, while [`adex`] tops out at `v_t + 10 * slope`, which is usually
/// still negative. A detector fixed at zero would report that an AdEx
/// neuron never fires.
#[must_use]
pub fn spike_times(trace: &[(f64, f64)], level: f64) -> Vec<f64> {
    upward_crossings(trace, level)
}

fn check_run(t_end: f64, dt: f64, largest: f64) -> Result<usize, GeomError> {
    if !(t_end > 0.0) || !(dt > 0.0) || dt > largest || dt >= t_end {
        return Err(GeomError::InvalidArgument("the run length or step size is out of range"));
    }
    let steps = (t_end / dt).ceil();
    if steps > 2e7 {
        return Err(GeomError::InvalidArgument("that many steps would not finish"));
    }
    Ok(steps as usize)
}

// ---------------------------------------------------------------------------
// Hodgkin-Huxley
// ---------------------------------------------------------------------------

/// Membrane capacitance, uF/cm^2.
pub const HH_C_M: f64 = 1.0;
/// Maximal sodium conductance, mS/cm^2.
pub const HH_G_NA: f64 = 120.0;
/// Maximal potassium conductance, mS/cm^2.
pub const HH_G_K: f64 = 36.0;
/// Leak conductance, mS/cm^2.
pub const HH_G_L: f64 = 0.3;
/// Sodium reversal potential, mV.
pub const HH_E_NA: f64 = 50.0;
/// Potassium reversal potential, mV.
pub const HH_E_K: f64 = -77.0;
/// Leak reversal potential, mV, chosen so the model rests at -65 mV.
pub const HH_E_L: f64 = -54.387;
/// The model's resting potential, mV.
pub const HH_V_REST: f64 = -65.0;

/// `x / (exp(x / y) - 1)`, continued through the removable singularity.
///
/// Three of the six Hodgkin-Huxley rate constants have this form and each
/// is `0/0` at one particular voltage -- `alpha_m` at -40 mV, `alpha_n` at
/// -55 mV. Evaluated naively those give NaN at exactly those voltages and
/// lose precision near them, which a simulation reaches sooner or later.
/// The limit is `y`, and near the singularity the series `y - x/2` is both
/// accurate and finite.
fn exprel(x: f64, y: f64) -> f64 {
    if (x / y).abs() < 1e-6 {
        y - 0.5 * x
    } else {
        x / ((x / y).exp() - 1.0)
    }
}

/// The six voltage-dependent rate constants, per millisecond.
fn hh_rates(v: f64) -> [f64; 6] {
    let alpha_m = 0.1 * exprel(-(v + 40.0), 10.0);
    let beta_m = 4.0 * (-(v + 65.0) / 18.0).exp();
    let alpha_h = 0.07 * (-(v + 65.0) / 20.0).exp();
    let beta_h = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
    let alpha_n = 0.01 * exprel(-(v + 55.0), 10.0);
    let beta_n = 0.125 * (-(v + 65.0) / 80.0).exp();
    [alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n]
}

/// The steady-state gating variables at a holding potential.
///
/// A gate settles at `alpha / (alpha + beta)`; starting a run anywhere
/// else adds a transient that has nothing to do with the stimulus.
#[must_use]
pub fn hh_steady_state(v: f64) -> (f64, f64, f64) {
    let [am, bm, ah, bh, an, bn] = hh_rates(v);
    (am / (am + bm), ah / (ah + bh), an / (an + bn))
}

/// The Hodgkin-Huxley membrane, integrated with fixed-step RK4.
///
/// Returns `(t, V, m, h, n)` per step. The run starts from the gating
/// variables' steady state at [`HH_V_REST`], so an unstimulated axon stays
/// where it is instead of relaxing through a spurious transient.
///
/// The action potential is not built in. Sodium activation `m` is fast and
/// its cube makes the inward current explosive; inactivation `h` and
/// potassium activation `n` are ten times slower and end it. That
/// separation of timescales is the whole mechanism, and it is why the
/// threshold is a property of the trajectory rather than a parameter.
///
/// A strongly *hyperpolarising* current is the one thing this integrator
/// cannot take. Below about -25 uA/cm^2 the voltage falls far enough that
/// `beta_m`, which grows exponentially as the membrane hyperpolarises,
/// reaches thousands per millisecond and a fixed step of 0.01 ms is no
/// longer stable. That is reported as a breakdown rather than returned as
/// a trace full of nonsense. Depolarising currents have no such limit:
/// hundreds of uA/cm^2 integrate cleanly, and simply drive the model into
/// depolarisation block.
///
/// # Errors
/// Returns an error for a non-positive `t_end`, a `dt` outside `(0, 0.05]`
/// -- above which fixed-step RK4 loses the upstroke -- a `dt` that is not
/// smaller than `t_end`, or an integration that diverges.
pub fn hodgkin_huxley(
    i_ext: &dyn Fn(f64) -> f64,
    t_end: f64,
    dt: f64,
) -> Result<Vec<(f64, f64, f64, f64, f64)>, GeomError> {
    let steps = check_run(t_end, dt, 0.05)?;
    let (m0, h0, n0) = hh_steady_state(HH_V_REST);
    let derivative = |t: f64, y: &[f64]| -> Vec<f64> {
        let (v, m, h, n) = (y[0], y[1], y[2], y[3]);
        let [am, bm, ah, bh, an, bn] = hh_rates(v);
        let i_na = HH_G_NA * m * m * m * h * (v - HH_E_NA);
        let i_k = HH_G_K * n * n * n * n * (v - HH_E_K);
        let i_l = HH_G_L * (v - HH_E_L);
        vec![
            (i_ext(t) - i_na - i_k - i_l) / HH_C_M,
            am * (1.0 - m) - bm * m,
            ah * (1.0 - h) - bh * h,
            an * (1.0 - n) - bn * n,
        ]
    };
    let mut state = vec![HH_V_REST, m0, h0, n0];
    let mut out = Vec::with_capacity(steps + 1);
    out.push((0.0, state[0], state[1], state[2], state[3]));
    for step in 0..steps {
        let t = step as f64 * dt;
        state = rk4_step_vec(&derivative, t, &state, dt);
        if !state.iter().all(|x| x.is_finite()) {
            return Err(GeomError::Degenerate("the Hodgkin-Huxley integration diverged"));
        }
        out.push((t + dt, state[0], state[1], state[2], state[3]));
    }
    Ok(out)
}

/// The spike times in a Hodgkin-Huxley trace.
#[must_use]
pub fn hh_spike_times(trace: &[(f64, f64, f64, f64, f64)]) -> Vec<f64> {
    let voltage: Vec<(f64, f64)> = trace.iter().map(|row| (row.0, row.1)).collect();
    upward_crossings(&voltage, SPIKE_LEVEL)
}

/// The smallest sustained current, in uA/cm^2, that makes the model fire.
///
/// Found by bisection on "does a 120 ms step produce a spike". This is the
/// rheobase, and it is not the same thing as a voltage threshold: a brief
/// pulse well above this current can fail to fire, and the model has no
/// single voltage at which firing becomes inevitable.
#[must_use]
pub fn hh_spike_threshold_estimate() -> f64 {
    let fires = |current: f64| -> bool {
        let trace = hodgkin_huxley(&|_| current, 120.0, 0.01).expect("fixed valid parameters");
        !hh_spike_times(&trace).is_empty()
    };
    let (mut low, mut high) = (0.0, 20.0);
    for _ in 0..24 {
        let mid = 0.5 * (low + high);
        if fires(mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
    0.5 * (low + high)
}

/// The firing rate in hertz against sustained current, for each current in
/// `currents`.
///
/// Hodgkin-Huxley's F-I curve is discontinuous: at the rheobase the rate
/// jumps to about 50 Hz rather than rising from zero, because the
/// oscillation is born through a subcritical Hopf bifurcation with a
/// finite frequency. A neuron whose rate can be tuned smoothly to
/// arbitrarily low values -- a type I neuron -- needs a different
/// bifurcation, which [`morris_lecar`] can be parameterised to show.
///
/// The first 30 ms of each run are discarded so the onset transient does
/// not enter the rate.
///
/// # Errors
/// Returns an error if `currents` is empty or holds a value that is not
/// finite.
pub fn hh_fi_curve(currents: &[f64]) -> Result<Vec<(f64, f64)>, GeomError> {
    if currents.is_empty() || currents.iter().any(|c| !c.is_finite()) {
        return Err(GeomError::InvalidArgument("hh_fi_curve: bad currents"));
    }
    let settle = 30.0;
    let t_end = 230.0;
    currents
        .iter()
        .map(|current| {
            let trace = hodgkin_huxley(&|_| *current, t_end, 0.01)?;
            let counted = hh_spike_times(&trace).into_iter().filter(|t| *t >= settle).count();
            // Spikes per millisecond, reported per second.
            Ok((*current, 1000.0 * counted as f64 / (t_end - settle)))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Reduced and integrate-and-fire models
// ---------------------------------------------------------------------------

/// FitzHugh-Nagumo, the two-variable caricature of an excitable membrane.
///
/// `dv/dt = v - v^3/3 - w + I`, `dw/dt = (v + a - b w) / tau`. Returns
/// `(t, v, w)` per step, dimensionless throughout.
///
/// The point of the reduction is that two variables can be drawn: the
/// cubic `v` nullcline and the straight `w` nullcline cross at a fixed
/// point, and whether that crossing sits on the cubic's middle branch
/// decides whether the neuron rests or oscillates. Excitability -- a small
/// push decaying, a slightly larger one taking a long excursion -- is
/// visible in the phase plane in a way it is not in four dimensions.
///
/// # Errors
/// Returns an error for a non-positive `tau`, or a run length or step size
/// out of range.
pub fn fitzhugh_nagumo_neuron(
    a: f64,
    b: f64,
    tau: f64,
    current: f64,
    v0: f64,
    w0: f64,
    t_end: f64,
    dt: f64,
) -> Result<Vec<(f64, f64, f64)>, GeomError> {
    if !(tau > 0.0) || ![a, b, current, v0, w0].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("fitzhugh_nagumo_neuron: bad parameters"));
    }
    let steps = check_run(t_end, dt, 0.5)?;
    let derivative = |_: f64, y: &[f64]| -> Vec<f64> {
        vec![y[0] - y[0].powi(3) / 3.0 - y[1] + current, (y[0] + a - b * y[1]) / tau]
    };
    let mut state = vec![v0, w0];
    let mut out = vec![(0.0, v0, w0)];
    for step in 0..steps {
        let t = step as f64 * dt;
        state = rk4_step_vec(&derivative, t, &state, dt);
        if !state.iter().all(|x| x.is_finite()) {
            return Err(GeomError::Degenerate("the FitzHugh-Nagumo integration diverged"));
        }
        out.push((t + dt, state[0], state[1]));
    }
    Ok(out)
}

/// Morris-Lecar's parameters, in the squid axon's units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorrisLecar {
    /// Membrane capacitance, uF/cm^2.
    pub c_m: f64,
    /// Leak, calcium and potassium conductances, mS/cm^2.
    pub g_l: f64,
    /// Calcium conductance, mS/cm^2.
    pub g_ca: f64,
    /// Potassium conductance, mS/cm^2.
    pub g_k: f64,
    /// Leak reversal potential, mV.
    pub v_l: f64,
    /// Calcium reversal potential, mV.
    pub v_ca: f64,
    /// Potassium reversal potential, mV.
    pub v_k: f64,
    /// Half-activation and slope of the calcium gate, mV.
    pub v1: f64,
    /// Slope of the calcium gate, mV.
    pub v2: f64,
    /// Half-activation of the potassium gate, mV.
    pub v3: f64,
    /// Slope of the potassium gate, mV.
    pub v4: f64,
    /// Rate scaling of the potassium gate, per ms.
    pub phi: f64,
}

impl MorrisLecar {
    /// The Hopf parameter set: a type II neuron, whose firing rate jumps
    /// to a finite value at threshold as Hodgkin-Huxley's does.
    #[must_use]
    pub fn hopf() -> Self {
        Self {
            c_m: 20.0,
            g_l: 2.0,
            g_ca: 4.4,
            g_k: 8.0,
            v_l: -60.0,
            v_ca: 120.0,
            v_k: -84.0,
            v1: -1.2,
            v2: 18.0,
            v3: 2.0,
            v4: 30.0,
            phi: 0.04,
        }
    }

    /// The saddle-node-on-a-circle parameter set: a type I neuron, which
    /// can fire arbitrarily slowly just above threshold because the limit
    /// cycle is born with infinite period.
    #[must_use]
    pub fn saddle_node() -> Self {
        Self { g_ca: 4.0, v3: 12.0, v4: 17.4, phi: 0.0667, ..Self::hopf() }
    }
}

/// Morris-Lecar, a calcium-potassium membrane with one gating variable.
///
/// Returns `(t, V, w)` per step. The calcium current is instantaneous,
/// which is what removes the second gate: only potassium activation `w`
/// has its own equation.
///
/// # Errors
/// Returns an error for a non-positive capacitance or slope, or a run
/// length or step size out of range.
pub fn morris_lecar(
    params: &MorrisLecar,
    current: f64,
    v0: f64,
    w0: f64,
    t_end: f64,
    dt: f64,
) -> Result<Vec<(f64, f64, f64)>, GeomError> {
    let p = *params;
    if !(p.c_m > 0.0) || !(p.v2 > 0.0) || !(p.v4 > 0.0) || !(p.phi > 0.0) {
        return Err(GeomError::InvalidArgument("morris_lecar: bad parameters"));
    }
    let steps = check_run(t_end, dt, 1.0)?;
    let derivative = |_: f64, y: &[f64]| -> Vec<f64> {
        let (v, w) = (y[0], y[1]);
        let m_inf = 0.5 * (1.0 + ((v - p.v1) / p.v2).tanh());
        let w_inf = 0.5 * (1.0 + ((v - p.v3) / p.v4).tanh());
        let tau_w = 1.0 / ((v - p.v3) / (2.0 * p.v4)).cosh();
        let ionic = p.g_l * (v - p.v_l)
            + p.g_ca * m_inf * (v - p.v_ca)
            + p.g_k * w * (v - p.v_k);
        vec![(current - ionic) / p.c_m, p.phi * (w_inf - w) * tau_w]
    };
    let mut state = vec![v0, w0];
    let mut out = vec![(0.0, v0, w0)];
    for step in 0..steps {
        let t = step as f64 * dt;
        state = rk4_step_vec(&derivative, t, &state, dt);
        if !state.iter().all(|x| x.is_finite()) {
            return Err(GeomError::Degenerate("the Morris-Lecar integration diverged"));
        }
        out.push((t + dt, state[0], state[1]));
    }
    Ok(out)
}

/// Izhikevich's two-variable spiking model.
///
/// `v' = 0.04 v^2 + 5 v + 140 - u + I` and `u' = a (b v - u)`, with the
/// reset `v <- c`, `u <- u + d` once `v` reaches 30 mV. Returns `(t, v)`
/// per step, with the spike sample set to the 30 mV peak so a trace can be
/// plotted without the reset looking like a downstroke.
///
/// The quadratic term is what makes it a spike generator rather than a
/// leaky integrator: above the unstable fixed point `v` runs away in finite
/// time, and the reset is what stops it. Two parameters then buy most of
/// the qualitative variety real neurons show -- see
/// [`izhikevich_presets`].
///
/// The published implementation advances `v` in two half-steps for
/// stability, and that is what is done here; `dt` is the reporting step.
///
/// # Errors
/// Returns an error for a non-positive `a`, or a run length or step size
/// out of range.
pub fn izhikevich(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    current: f64,
    t_end: f64,
    dt: f64,
) -> Result<Vec<(f64, f64)>, GeomError> {
    if !(a > 0.0) || ![b, c, d, current].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("izhikevich: bad parameters"));
    }
    let steps = check_run(t_end, dt, 1.0)?;
    let mut v = c;
    let mut u = b * v;
    let mut out = vec![(0.0, v)];
    for step in 0..steps {
        let t = (step + 1) as f64 * dt;
        let mut peaked = false;
        for _ in 0..2 {
            v += 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + current);
            if v >= 30.0 {
                peaked = true;
                break;
            }
        }
        u += dt * a * (b * v - u);
        if peaked {
            out.push((t, 30.0));
            v = c;
            u += d;
        } else {
            out.push((t, v));
        }
        if !v.is_finite() || !u.is_finite() {
            return Err(GeomError::Degenerate("the Izhikevich integration diverged"));
        }
    }
    Ok(out)
}

/// The five firing patterns Izhikevich's paper names, as `(a, b, c, d)`.
///
/// Regular spiking, intrinsically bursting, chattering, fast spiking and
/// low-threshold spiking. `c` and `d` set what happens after a spike, so
/// they are what separates a regular spiker from a burster; `a` and `b`
/// set the recovery variable's speed and its coupling to voltage.
#[must_use]
pub fn izhikevich_presets() -> Vec<(&'static str, [f64; 4])> {
    vec![
        ("RS", [0.02, 0.2, -65.0, 8.0]),
        ("IB", [0.02, 0.2, -55.0, 4.0]),
        ("CH", [0.02, 0.2, -50.0, 2.0]),
        ("FS", [0.1, 0.2, -65.0, 2.0]),
        ("LTS", [0.02, 0.25, -65.0, 2.0]),
    ]
}

/// The adaptive exponential integrate-and-fire neuron.
///
/// `C dV/dt = -g_L (V - E_L) + g_L dt_slope exp((V - v_t)/dt_slope) - w + I`
/// with `tau_w dw/dt = a (V - E_L) - w`, and the reset `V <- v_reset`,
/// `w <- w + b` at the peak. Returns `(t, V, w)` per step.
///
/// The exponential term is fitted to the sodium activation curve, so the
/// upstroke's *shape* near threshold is right even though the spike itself
/// is still stipulated. The adaptation current `w` is what the leaky
/// integrator lacks: it accumulates over a spike train and slows it, which
/// is the commonest firing pattern in cortex and cannot be produced by a
/// model with one variable.
///
/// The recorded spike sample sits at the peak `v_t + 10 * slope`, which is
/// where [`spike_times`] should be pointed to count them.
///
/// # Errors
/// Returns an error for a non-positive capacitance, conductance, slope or
/// adaptation time constant, or a run length or step size out of range.
pub fn adex(
    c_m: f64,
    g_l: f64,
    e_l: f64,
    slope: f64,
    v_t: f64,
    tau_w: f64,
    a: f64,
    b: f64,
    v_reset: f64,
    current: f64,
    t_end: f64,
    dt: f64,
) -> Result<Vec<(f64, f64, f64)>, GeomError> {
    if !(c_m > 0.0) || !(g_l > 0.0) || !(slope > 0.0) || !(tau_w > 0.0) {
        return Err(GeomError::InvalidArgument("adex: bad parameters"));
    }
    let steps = check_run(t_end, dt, 0.5)?;
    let peak = v_t + 10.0 * slope;
    let mut v = e_l;
    let mut w = 0.0;
    let mut out = vec![(0.0, v, w)];
    for step in 0..steps {
        let t = (step + 1) as f64 * dt;
        let exponential = (((v - v_t) / slope).min(50.0)).exp();
        let dv = (-g_l * (v - e_l) + g_l * slope * exponential - w + current) / c_m;
        let dw = (a * (v - e_l) - w) / tau_w;
        v += dt * dv;
        w += dt * dw;
        if v >= peak {
            out.push((t, peak, w + b));
            v = v_reset;
            w += b;
        } else {
            out.push((t, v, w));
        }
        if !v.is_finite() || !w.is_finite() {
            return Err(GeomError::Degenerate("the AdEx integration diverged"));
        }
    }
    Ok(out)
}

/// The leaky integrate-and-fire neuron's spike times.
///
/// `tau dV/dt = -(V - V_rest) + R I`, with a spike and a reset to
/// `v_reset` whenever `V` reaches `v_th`, and an absolute refractory
/// period during which the voltage is clamped. Gaussian current noise of
/// standard deviation `noise` is added per unit time, scaled so the
/// result does not depend on `dt`.
///
/// The voltage between spikes carries no information the times do not, so
/// only the times are returned.
///
/// # Errors
/// Returns an error for a non-positive `tau`, a negative refractory period
/// or noise, a threshold at or below the reset, or a run length or step
/// size out of range.
pub fn lif_neuron(
    current: f64,
    tau: f64,
    v_th: f64,
    v_reset: f64,
    refractory: f64,
    noise: f64,
    t_end: f64,
    dt: f64,
    rng: &mut Rng,
) -> Result<Vec<f64>, GeomError> {
    if !(tau > 0.0) || refractory < 0.0 || noise < 0.0 || v_th <= v_reset {
        return Err(GeomError::InvalidArgument("lif_neuron: bad parameters"));
    }
    let steps = check_run(t_end, dt, tau)?;
    let mut v = v_reset;
    let mut blocked_until = f64::NEG_INFINITY;
    let mut spikes = Vec::new();
    for step in 0..steps {
        let t = (step + 1) as f64 * dt;
        if t < blocked_until {
            v = v_reset;
            continue;
        }
        // The noise enters as a Wiener increment, so its size grows with
        // the square root of the step and the trajectory's statistics do
        // not depend on how finely it was sampled.
        let kick = noise * dt.sqrt() * rng.next_gaussian();
        v += dt * (-v + current) / tau + kick / tau;
        if v >= v_th {
            spikes.push(t);
            v = v_reset;
            blocked_until = t + refractory;
        }
    }
    Ok(spikes)
}

/// The leaky integrate-and-fire firing rate in the noiseless case, exactly.
///
/// `1 / (t_ref + tau ln((I - V_reset)/(I - V_th)))` for a current above
/// threshold, and zero otherwise. The rest potential is taken as zero, so
/// `I` is measured in the same units as the voltages.
///
/// The logarithm is what makes the curve saturate: doubling a large
/// current barely changes the rate, because the refractory period comes to
/// dominate. Below `v_th` the neuron never fires however long you wait --
/// the exact zero, not a very small number.
///
/// # Errors
/// Returns an error for a non-positive `tau`, a negative refractory
/// period, or a threshold at or below the reset.
pub fn lif_fi_exact(
    current: f64,
    tau: f64,
    v_th: f64,
    v_reset: f64,
    refractory: f64,
) -> Result<f64, GeomError> {
    if !(tau > 0.0) || refractory < 0.0 || v_th <= v_reset {
        return Err(GeomError::InvalidArgument("lif_fi_exact: bad parameters"));
    }
    if current <= v_th {
        return Ok(0.0);
    }
    let interval = refractory + tau * ((current - v_reset) / (current - v_th)).ln();
    Ok(1.0 / interval)
}

// ---------------------------------------------------------------------------
// Spike train statistics
// ---------------------------------------------------------------------------

/// The gaps between successive spikes.
///
/// # Errors
/// Returns an error if the times are not sorted, since an unsorted train
/// would silently produce negative intervals.
pub fn interspike_intervals(spikes: &[f64]) -> Result<Vec<f64>, GeomError> {
    if spikes.windows(2).any(|p| p[1] < p[0]) {
        return Err(GeomError::InvalidArgument("the spike times are not in order"));
    }
    Ok(spikes.windows(2).map(|p| p[1] - p[0]).collect())
}

/// The coefficient of variation of the interspike intervals.
///
/// One for a Poisson process, because an exponential distribution's
/// standard deviation equals its mean; near zero for a regular pacemaker;
/// and above one for a bursting cell, whose intervals come in two very
/// different sizes. It is a measure of *irregularity*, not of rate: it is
/// unchanged by running the clock faster.
///
/// # Errors
/// Returns an error for fewer than three spikes, an unsorted train, or a
/// mean interval of zero.
pub fn cv_isi(spikes: &[f64]) -> Result<f64, GeomError> {
    let intervals = interspike_intervals(spikes)?;
    if intervals.len() < 2 {
        return Err(GeomError::InvalidArgument("the coefficient needs at least three spikes"));
    }
    let n = intervals.len() as f64;
    let mean = intervals.iter().sum::<f64>() / n;
    if !(mean > 0.0) {
        return Err(GeomError::Degenerate("every spike arrived at the same instant"));
    }
    let variance = intervals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok(variance.sqrt() / mean)
}

/// The Fano factor of a set of counts: variance over mean.
///
/// One for a Poisson process. Unlike [`cv_isi`] this is measured over a
/// window, so the two can disagree: a train with regular intervals but a
/// drifting rate has a low CV and a high Fano factor, because the
/// irregularity is between windows rather than within them.
///
/// # Errors
/// Returns an error for fewer than two counts or a mean of zero.
pub fn fano_factor(counts: &[u64]) -> Result<f64, GeomError> {
    if counts.len() < 2 {
        return Err(GeomError::InvalidArgument("the Fano factor needs at least two windows"));
    }
    let n = counts.len() as f64;
    let mean = counts.iter().map(|c| *c as f64).sum::<f64>() / n;
    if !(mean > 0.0) {
        return Err(GeomError::Degenerate("no spikes were counted"));
    }
    let variance = counts.iter().map(|c| (*c as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok(variance / mean)
}

/// A homogeneous Poisson spike train on `[0, t_end)`.
///
/// Generated by accumulating exponential waiting times, which is exact --
/// there is no time step and so no chance of two spikes in one bin.
///
/// # Errors
/// Returns an error for a non-positive rate or run length, or an expected
/// count above ten million.
pub fn poisson_spike_train(rate: f64, t_end: f64, rng: &mut Rng) -> Result<Vec<f64>, GeomError> {
    if !(rate > 0.0) || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("poisson_spike_train: bad parameters"));
    }
    if rate * t_end > 1e7 {
        return Err(GeomError::InvalidArgument("that many spikes would not fit in memory"));
    }
    let mut out = Vec::new();
    let mut t = 0.0;
    loop {
        t += -(1.0 - rng.next_f64()).ln() / rate;
        if t >= t_end {
            return Ok(out);
        }
        out.push(t);
    }
}

/// The peri-stimulus time histogram: the mean firing rate in each bin,
/// across trials.
///
/// Dividing by the bin width and the trial count is what makes this a
/// rate rather than a count, and is what lets histograms with different
/// binnings be compared. The bin width is the whole choice in a PSTH: too
/// wide and a transient response is smeared into the background, too
/// narrow and every bin is zero or one.
///
/// # Errors
/// Returns an error for no trials, a non-positive bin width or window, or
/// a spike time outside `[0, t_end)`.
pub fn psth(trains: &[Vec<f64>], bin: f64, t_end: f64) -> Result<Vec<f64>, GeomError> {
    if trains.is_empty() || !(bin > 0.0) || !(t_end > 0.0) || bin > t_end {
        return Err(GeomError::InvalidArgument("psth: bad parameters"));
    }
    let bins = (t_end / bin).ceil() as usize;
    let mut counts = vec![0.0f64; bins];
    for train in trains {
        for spike in train {
            if !(0.0..t_end).contains(spike) {
                return Err(GeomError::InvalidArgument("a spike falls outside the window"));
            }
            counts[((spike / bin) as usize).min(bins - 1)] += 1.0;
        }
    }
    let scale = bin * trains.len() as f64;
    Ok(counts.into_iter().map(|c| c / scale).collect())
}

/// Every spike as a `(time, trial)` pair, sorted by time.
///
/// The raster is the raw data a PSTH averages away, and the two answer
/// different questions: a response present on every trial and one present
/// on half the trials at twice the rate give the same histogram.
#[must_use]
pub fn raster_data(trains: &[Vec<f64>]) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = trains
        .iter()
        .enumerate()
        .flat_map(|(trial, train)| train.iter().map(move |t| (*t, trial)))
        .collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// The spike-triggered average: the mean stimulus in the `window` samples
/// before a spike.
///
/// Returned oldest sample first, so the last entry is the stimulus at the
/// spike itself. Spikes too early for a full window are skipped, and the
/// count of those that contributed decides the divisor.
///
/// This estimates the neuron's linear filter only if the stimulus is white:
/// any correlation in the stimulus appears in the average and will be
/// mistaken for structure in the neuron. The usual remedy is to whiten by
/// the stimulus autocorrelation, which is a different calculation from
/// this one.
///
/// # Errors
/// Returns an error for an empty stimulus, a non-positive sampling step, a
/// zero window, a window longer than the stimulus, or no usable spike.
pub fn spike_triggered_average(
    stimulus: &[f64],
    dt: f64,
    spikes: &[f64],
    window: usize,
) -> Result<Vec<f64>, GeomError> {
    if stimulus.is_empty() || !(dt > 0.0) || window == 0 || window > stimulus.len() {
        return Err(GeomError::InvalidArgument("spike_triggered_average: bad parameters"));
    }
    let mut sum = vec![0.0f64; window];
    let mut used = 0usize;
    for spike in spikes {
        if !spike.is_finite() || *spike < 0.0 {
            return Err(GeomError::InvalidArgument("a spike time is negative or not finite"));
        }
        let index = (spike / dt) as usize;
        if index + 1 < window || index >= stimulus.len() {
            continue;
        }
        used += 1;
        for (slot, offset) in sum.iter_mut().zip((0..window).rev()) {
            *slot += stimulus[index - offset];
        }
    }
    if used == 0 {
        return Err(GeomError::Degenerate("no spike had a full window of stimulus before it"));
    }
    Ok(sum.into_iter().map(|s| s / used as f64).collect())
}

/// Fits `r(theta) = amplitude * exp(kappa * cos(theta - preferred))` to a
/// set of angles and rates, returning `(preferred, kappa, amplitude)`.
///
/// Taking logarithms turns the von Mises form into
/// `ln r = ln A + (kappa cos mu) cos theta + (kappa sin mu) sin theta`,
/// which is linear in three coefficients and so is solved exactly rather
/// than searched for. `preferred` comes back in `(-pi, pi]`.
///
/// The price of the linearisation is that it fits the log rate, so it
/// weights a doubling at a low rate as heavily as a doubling at the peak.
/// With noiseless data that costs nothing and the fit is exact; with noisy
/// data it biases toward the flanks.
///
/// # Errors
/// Returns an error for fewer than three points, mismatched lengths, a
/// non-positive rate, or angles that do not determine the fit -- all equal,
/// or spread over too little of the circle.
pub fn tuning_curve_fit_von_mises(
    angles: &[f64],
    rates: &[f64],
) -> Result<(f64, f64, f64), GeomError> {
    if angles.len() < 3 || angles.len() != rates.len() {
        return Err(GeomError::InvalidArgument("the fit needs at least three matched points"));
    }
    if rates.iter().any(|r| !(*r > 0.0)) || angles.iter().any(|a| !a.is_finite()) {
        return Err(GeomError::InvalidArgument("a rate is not positive or an angle is not finite"));
    }
    // Normal equations for the design matrix [1, cos, sin].
    let n = angles.len() as f64;
    let (mut sc, mut ss, mut scc, mut sss, mut scs) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sy, mut syc, mut sys) = (0.0, 0.0, 0.0);
    for (angle, rate) in angles.iter().zip(rates.iter()) {
        let (s, c) = angle.sin_cos();
        let y = rate.ln();
        sc += c;
        ss += s;
        scc += c * c;
        sss += s * s;
        scs += c * s;
        sy += y;
        syc += y * c;
        sys += y * s;
    }
    let matrix = [[n, sc, ss], [sc, scc, scs], [ss, scs, sss]];
    let rhs = [sy, syc, sys];
    let solved = solve3(&matrix, &rhs)
        .ok_or(GeomError::Degenerate("the angles do not determine a tuning curve"))?;
    let (log_amplitude, x, y) = (solved[0], solved[1], solved[2]);
    let kappa = x.hypot(y);
    let preferred = y.atan2(x);
    Ok((preferred, kappa, log_amplitude.exp()))
}

/// Gaussian elimination on a 3x3 system, or `None` if it is singular.
fn solve3(matrix: &[[f64; 3]; 3], rhs: &[f64; 3]) -> Option<[f64; 3]> {
    let mut a = [
        [matrix[0][0], matrix[0][1], matrix[0][2], rhs[0]],
        [matrix[1][0], matrix[1][1], matrix[1][2], rhs[1]],
        [matrix[2][0], matrix[2][1], matrix[2][2], rhs[2]],
    ];
    let scale = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for column in 0..3 {
        let pivot = (column..3).max_by(|i, j| {
            a[*i][column].abs().partial_cmp(&a[*j][column].abs()).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        a.swap(column, pivot);
        if a[column][column].abs() < 1e-12 * scale {
            return None;
        }
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = a[row][column] / a[column][column];
            for k in column..4 {
                a[row][k] -= factor * a[column][k];
            }
        }
    }
    Some([a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]])
}

// ---------------------------------------------------------------------------
// Synapses and plasticity
// ---------------------------------------------------------------------------

/// The conductance of an exponential synapse at time `t`, given the
/// presynaptic spike times.
///
/// Each spike adds `g_max` instantaneously and it decays as
/// `exp(-(t - t_spike)/tau)`. Conductances sum, so a burst arriving within
/// a time constant produces more than one spike's worth -- which is what
/// makes a synapse a low-pass filter of its input rather than a repeater.
///
/// # Errors
/// Returns an error for a non-positive `tau` or an unsorted spike train.
pub fn synapse_exp(g_max: f64, tau: f64, spikes: &[f64], t: f64) -> Result<f64, GeomError> {
    if !(tau > 0.0) || spikes.windows(2).any(|p| p[1] < p[0]) {
        return Err(GeomError::InvalidArgument("synapse_exp: bad time constant or train"));
    }
    Ok(spikes
        .iter()
        .filter(|s| **s <= t)
        .map(|s| g_max * (-(t - s) / tau).exp())
        .sum())
}

/// The conductance of an alpha synapse at time `t`.
///
/// `g_max * x * exp(1 - x)` with `x = (t - t_spike)/tau`, which peaks at
/// exactly `g_max` one time constant after the spike. The rise is what
/// distinguishes it from [`synapse_exp`]: a real conductance cannot jump,
/// and the delay to peak matters when the question is whether two inputs
/// coincide.
///
/// # Errors
/// Returns an error for a non-positive `tau` or an unsorted spike train.
pub fn alpha_synapse(g_max: f64, tau: f64, spikes: &[f64], t: f64) -> Result<f64, GeomError> {
    if !(tau > 0.0) || spikes.windows(2).any(|p| p[1] < p[0]) {
        return Err(GeomError::InvalidArgument("alpha_synapse: bad time constant or train"));
    }
    Ok(spikes
        .iter()
        .filter(|s| **s <= t)
        .map(|s| {
            let x = (t - s) / tau;
            g_max * x * (1.0 - x).exp()
        })
        .sum())
}

/// The spike-timing-dependent plasticity window: the weight change for a
/// post-minus-pre interval of `delta`.
///
/// Positive `delta` -- the postsynaptic spike came second -- potentiates by
/// `a_plus exp(-delta/tau_plus)`; negative depresses by
/// `-a_minus exp(delta/tau_minus)`. Exactly simultaneous spikes give zero,
/// which is the discontinuity at the origin the rule is known for: a
/// millisecond either way is the difference between strengthening and
/// weakening.
///
/// # Errors
/// Returns an error for a non-positive time constant or a negative
/// amplitude.
pub fn stdp_window(
    delta: f64,
    a_plus: f64,
    a_minus: f64,
    tau_plus: f64,
    tau_minus: f64,
) -> Result<f64, GeomError> {
    if !(tau_plus > 0.0) || !(tau_minus > 0.0) || a_plus < 0.0 || a_minus < 0.0 {
        return Err(GeomError::InvalidArgument("stdp_window: bad parameters"));
    }
    if delta > 0.0 {
        Ok(a_plus * (-delta / tau_plus).exp())
    } else if delta < 0.0 {
        Ok(-a_minus * (delta / tau_minus).exp())
    } else {
        Ok(0.0)
    }
}

/// The total weight change from every pre-post pair in two trains.
///
/// This is the all-to-all rule: each presynaptic spike is paired with each
/// postsynaptic spike. It is the simplest interpretation and not the only
/// one -- nearest-neighbour pairing gives noticeably less potentiation at
/// high rates, because a burst's later spikes no longer each count against
/// every earlier one.
///
/// # Errors
/// Returns an error for a bad window parameter, an unsorted train, or more
/// than ten million pairs.
pub fn stdp_train(
    pre: &[f64],
    post: &[f64],
    a_plus: f64,
    a_minus: f64,
    tau_plus: f64,
    tau_minus: f64,
) -> Result<f64, GeomError> {
    if pre.windows(2).any(|p| p[1] < p[0]) || post.windows(2).any(|p| p[1] < p[0]) {
        return Err(GeomError::InvalidArgument("a spike train is not in order"));
    }
    if pre.len().saturating_mul(post.len()) > 10_000_000 {
        return Err(GeomError::InvalidArgument("that many pairings would not finish"));
    }
    let mut total = 0.0;
    for before in pre {
        for after in post {
            total += stdp_window(after - before, a_plus, a_minus, tau_plus, tau_minus)?;
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Networks
// ---------------------------------------------------------------------------

/// Izhikevich's randomly connected network of excitatory and inhibitory
/// neurons, returning every spike as `(time in ms, neuron index)`.
///
/// Excitatory neurons are regular spikers scattered toward chattering by a
/// squared random factor, inhibitory ones toward fast spiking, exactly as
/// in the published network; each neuron receives a random thalamic drive
/// each millisecond, with the excitatory population driven harder. All
/// weights are all-to-all with random excitatory strengths and stronger
/// fixed inhibitory ones.
///
/// The behaviour worth looking for is that the population synchronises
/// into gamma-band rhythms without any oscillator being built in: the
/// rhythm is a property of the excitatory-inhibitory loop, not of the
/// cells. Inhibition being both stronger and faster than excitation is
/// what produces it.
///
/// # Errors
/// Returns an error for no excitatory or no inhibitory neurons, more than
/// four thousand in total, or a non-positive run length.
pub fn izhikevich_network(
    n_exc: usize,
    n_inh: usize,
    t_end: f64,
    rng: &mut Rng,
) -> Result<Vec<(f64, usize)>, GeomError> {
    if n_exc == 0 || n_inh == 0 || n_exc + n_inh > 4000 || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("izhikevich_network: bad parameters"));
    }
    let n = n_exc + n_inh;
    let steps = (t_end.ceil()) as usize;
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        let r = rng.next_f64();
        if i < n_exc {
            a[i] = 0.02;
            b[i] = 0.2;
            c[i] = -65.0 + 15.0 * r * r;
            d[i] = 8.0 - 6.0 * r * r;
        } else {
            a[i] = 0.02 + 0.08 * r;
            b[i] = 0.25 - 0.05 * r;
            c[i] = -65.0;
            d[i] = 2.0;
        }
    }
    let mut weight = vec![0.0f64; n * n];
    for target in 0..n {
        for source in 0..n {
            weight[target * n + source] =
                if source < n_exc { 0.5 * rng.next_f64() } else { -rng.next_f64() };
        }
    }
    let mut v: Vec<f64> = (0..n).map(|i| c[i]).collect();
    let mut u: Vec<f64> = (0..n).map(|i| b[i] * v[i]).collect();
    let mut out = Vec::new();
    for step in 0..steps {
        let t = step as f64;
        let mut input: Vec<f64> = (0..n)
            .map(|i| {
                let drive = if i < n_exc { 5.0 } else { 2.0 };
                drive * rng.next_gaussian()
            })
            .collect();
        let fired: Vec<usize> = (0..n).filter(|i| v[*i] >= 30.0).collect();
        for i in &fired {
            out.push((t, *i));
            v[*i] = c[*i];
            u[*i] += d[*i];
        }
        for target in 0..n {
            for source in &fired {
                input[target] += weight[target * n + source];
            }
        }
        for i in 0..n {
            // Two half-millisecond steps for the voltage, as published.
            for _ in 0..2 {
                v[i] += 0.5 * (0.04 * v[i] * v[i] + 5.0 * v[i] + 140.0 - u[i] + input[i]);
            }
            v[i] = v[i].min(30.0);
            u[i] += a[i] * (b[i] * v[i] - u[i]);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Hopfield networks
// ---------------------------------------------------------------------------

/// The Hebbian weight matrix storing a set of +-1 patterns.
///
/// `w_ij = (1/n) sum_p x_i^p x_j^p` with a zero diagonal. The rule is
/// local and one-shot: each pattern is written by a single pass and never
/// revisited, which is why the network cannot unlearn and why capacity is
/// the limiting resource rather than training time.
///
/// # Errors
/// Returns an error for no patterns, patterns of differing or zero length,
/// or an entry that is not exactly +1 or -1.
pub fn hopfield_store(patterns: &[Vec<i8>]) -> Result<Matrix, GeomError> {
    let n = patterns.first().map_or(0, Vec::len);
    if patterns.is_empty() || n == 0 || patterns.iter().any(|p| p.len() != n) {
        return Err(GeomError::InvalidArgument("hopfield_store: bad patterns"));
    }
    if patterns.iter().flatten().any(|s| *s != 1 && *s != -1) {
        return Err(GeomError::InvalidArgument("a pattern entry is not plus or minus one"));
    }
    let mut w = Matrix::zeros(n, n);
    for pattern in patterns {
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let value = w.get(i, j) + f64::from(pattern[i]) * f64::from(pattern[j]) / n as f64;
                    w.set(i, j, value);
                }
            }
        }
    }
    Ok(w)
}

/// Recalls from a probe by sweeping the units in index order, stopping
/// early once a whole sweep changes nothing. `steps` counts sweeps.
///
/// The updates are sequential rather than simultaneous, and the difference
/// is not cosmetic. Flipping one unit at a time against the current state
/// can only lower the energy `-1/2 x' W x` when the weights are symmetric
/// with a zero diagonal, so recall converges to a fixed point. Updating
/// every unit at once against the *old* state has no such guarantee: it
/// can raise the energy and settle into a two-cycle that oscillates
/// forever between two states, neither of them stored. What it converges *to* need not be a stored
/// pattern: mixtures of three stored patterns are also minima, and so are
/// the negatives of everything stored, since flipping every unit leaves
/// the energy unchanged.
///
/// # Errors
/// Returns an error for a non-square matrix, a probe of the wrong length,
/// or a probe entry that is not exactly +1 or -1.
pub fn hopfield_recall(w: &Matrix, probe: &[i8], steps: usize) -> Result<Vec<i8>, GeomError> {
    let n = w.rows;
    if w.cols != n || probe.len() != n {
        return Err(GeomError::InvalidArgument("hopfield_recall: mismatched shapes"));
    }
    if probe.iter().any(|s| *s != 1 && *s != -1) {
        return Err(GeomError::InvalidArgument("a probe entry is not plus or minus one"));
    }
    let mut state = probe.to_vec();
    for _ in 0..steps {
        let mut changed = false;
        for i in 0..n {
            let field: f64 = (0..n).map(|j| w.get(i, j) * f64::from(state[j])).sum();
            // A unit with no field keeps its state rather than flipping
            // arbitrarily.
            let next = if field > 0.0 {
                1
            } else if field < 0.0 {
                -1
            } else {
                state[i]
            };
            if next != state[i] {
                state[i] = next;
                changed = true;
            }
        }
        if !changed {
            return Ok(state);
        }
    }
    Ok(state)
}

/// The energy of a state under a Hopfield weight matrix.
///
/// # Errors
/// Returns an error for a non-square matrix or a state of the wrong length.
pub fn hopfield_energy(w: &Matrix, state: &[i8]) -> Result<f64, GeomError> {
    let n = w.rows;
    if w.cols != n || state.len() != n {
        return Err(GeomError::InvalidArgument("hopfield_energy: mismatched shapes"));
    }
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            total -= 0.5 * w.get(i, j) * f64::from(state[i]) * f64::from(state[j]);
        }
    }
    Ok(total)
}

/// The fraction of stored patterns recalled exactly from themselves, over
/// `trials` random pattern sets of size `stored`.
///
/// Recall from the pattern itself is the easiest possible test, so this
/// measures storage rather than error correction. It falls off sharply
/// near `0.138 n` patterns: below that the stored patterns are stable, and
/// above it the crosstalk between them overwhelms the signal and the
/// network forgets everything at once rather than degrading gracefully.
///
/// # Errors
/// Returns an error for a network or trial count of zero, no patterns to
/// store, or a request above five hundred units.
pub fn hopfield_capacity_check(
    n: usize,
    stored: usize,
    trials: usize,
    rng: &mut Rng,
) -> Result<f64, GeomError> {
    if n == 0 || n > 500 || stored == 0 || trials == 0 {
        return Err(GeomError::InvalidArgument("hopfield_capacity_check: bad parameters"));
    }
    let mut recalled = 0usize;
    for _ in 0..trials {
        let patterns: Vec<Vec<i8>> = (0..stored)
            .map(|_| (0..n).map(|_| if rng.next_f64() < 0.5 { -1i8 } else { 1 }).collect())
            .collect();
        let w = hopfield_store(&patterns)?;
        for pattern in &patterns {
            if hopfield_recall(&w, pattern, 1)? == *pattern {
                recalled += 1;
            }
        }
    }
    Ok(recalled as f64 / (trials * stored) as f64)
}

// ---------------------------------------------------------------------------
// Population dynamics and cables
// ---------------------------------------------------------------------------

/// The Wilson-Cowan equations for coupled excitatory and inhibitory
/// populations, returning `(t, E, I)`.
///
/// `tau_e dE/dt = -E + S(c_ee E - c_ei I + p_e)` and the matching
/// equation for `I`, with `S` the logistic function. `E` and `I` are
/// fractions of each population active, so they stay in `[0, 1]`.
///
/// This is a mean-field model: it describes what a population does on
/// average and says nothing about individual spikes or their timing.
/// Oscillations here are oscillations of the *rate*, which is a different
/// claim from the synchrony a spiking network shows, and the two need not
/// coincide.
///
/// # Errors
/// Returns an error for a non-positive time constant or slope, initial
/// activity outside `[0, 1]`, or a run length or step size out of range.
pub fn wilson_cowan(
    c_ee: f64,
    c_ei: f64,
    c_ie: f64,
    c_ii: f64,
    p_e: f64,
    p_i: f64,
    tau_e: f64,
    tau_i: f64,
    slope: f64,
    threshold: f64,
    e0: f64,
    i0: f64,
    t_end: f64,
    dt: f64,
) -> Result<Vec<(f64, f64, f64)>, GeomError> {
    if !(tau_e > 0.0) || !(tau_i > 0.0) || !(slope > 0.0) {
        return Err(GeomError::InvalidArgument("wilson_cowan: bad parameters"));
    }
    if !(0.0..=1.0).contains(&e0) || !(0.0..=1.0).contains(&i0) {
        return Err(GeomError::InvalidArgument("the activities must start as fractions"));
    }
    let steps = check_run(t_end, dt, tau_e.min(tau_i))?;
    let response = |x: f64| 1.0 / (1.0 + (-slope * (x - threshold)).exp());
    let derivative = |_: f64, y: &[f64]| -> Vec<f64> {
        let (e, i) = (y[0], y[1]);
        vec![
            (-e + response(c_ee * e - c_ei * i + p_e)) / tau_e,
            (-i + response(c_ie * e - c_ii * i + p_i)) / tau_i,
        ]
    };
    let mut state = vec![e0, i0];
    let mut out = vec![(0.0, e0, i0)];
    for step in 0..steps {
        let t = step as f64 * dt;
        state = rk4_step_vec(&derivative, t, &state, dt);
        if !state.iter().all(|x| x.is_finite()) {
            return Err(GeomError::Degenerate("the Wilson-Cowan integration diverged"));
        }
        out.push((t + dt, state[0], state[1]));
    }
    Ok(out)
}

/// The passive cable's length constant `sqrt(d R_m / (4 R_i))`.
///
/// With `r_m` in ohm-cm^2, `r_i` in ohm-cm and the diameter in cm, the
/// answer is in cm. The square root is the reason thin processes are
/// electrically short: halving the diameter shortens the reach only by
/// `sqrt(2)`, but that is enough that a dendritic spine's neck is a
/// different electrical world from its parent branch.
///
/// # Errors
/// Returns an error for a non-positive resistance or diameter.
pub fn length_constant(r_m: f64, r_i: f64, diameter: f64) -> Result<f64, GeomError> {
    if !(r_m > 0.0) || !(r_i > 0.0) || !(diameter > 0.0) {
        return Err(GeomError::InvalidArgument("length_constant: bad parameters"));
    }
    Ok((diameter * r_m / (4.0 * r_i)).sqrt())
}

/// The steady-state voltage along a finite passive cable with current
/// injected at one end and the far end sealed.
///
/// Returns `points` samples of `V(x)` over `[0, length]`, solved from the
/// discretised cable equation `lambda^2 V'' = V` rather than from the
/// closed form, so the boundary conditions are imposed rather than
/// assumed. The analytic answer for a sealed end is
/// `V(x) = V(0) cosh((L - x)/lambda) / cosh(L/lambda)`.
///
/// A sealed end is not a neutral choice. Current that reaches it has
/// nowhere to go, so the voltage there is *higher* than an infinite cable
/// would give -- an end effect that grows as the cable shortens relative
/// to its length constant.
///
/// # Errors
/// Returns an error for a non-positive length or length constant, fewer
/// than three points, or a singular system.
pub fn cable_equation_1d(
    length: f64,
    lambda: f64,
    v_injected: f64,
    points: usize,
) -> Result<Vec<f64>, GeomError> {
    if !(length > 0.0) || !(lambda > 0.0) || points < 3 {
        return Err(GeomError::InvalidArgument("cable_equation_1d: bad parameters"));
    }
    let h = length / (points - 1) as f64;
    let k = (h / lambda).powi(2);
    let mut sub = vec![0.0; points - 1];
    let mut diag = vec![0.0; points];
    let mut sup = vec![0.0; points - 1];
    let mut rhs = vec![0.0; points];
    // A clamped voltage at the injection site.
    diag[0] = 1.0;
    sup[0] = 0.0;
    rhs[0] = v_injected;
    for i in 1..points - 1 {
        sub[i - 1] = 1.0;
        diag[i] = -(2.0 + k);
        sup[i] = 1.0;
    }
    // Sealed end: no axial current, so the gradient vanishes there. Setting
    // a ghost node equal to its mirror image and substituting it into the
    // interior stencil gives 2 V[n-2] - (2 + k) V[n-1] = 0. The obvious
    // one-sided difference V[n-1] = V[n-2] also imposes a zero gradient,
    // but only to first order, and it drags the whole solution down with
    // it: the interior is second order and the boundary decides the rate.
    sub[points - 2] = 2.0;
    diag[points - 1] = -(2.0 + k);
    crate::linalg::thomas_solve(&sub, &diag, &sup, &rhs)
        .map_err(|_| GeomError::Degenerate("the cable system is singular"))
}

// ---------------------------------------------------------------------------
// Decision making
// ---------------------------------------------------------------------------

/// Simulated reaction times from the drift-diffusion model, as
/// `(time, chose the positive bound)`.
///
/// Evidence accumulates from zero with constant `drift` and Gaussian noise
/// until it reaches `+threshold` or `-threshold`. The model's appeal is
/// that one mechanism produces both the choice and its latency, and it
/// predicts the awkward fact that errors and correct responses have
/// nearly the same distribution of times when the starting point is
/// unbiased.
///
/// # Errors
/// Returns an error for a non-positive threshold, noise or step, no
/// trials, or a run that exhausts the fifty-million-step budget shared
/// across all trials.
pub fn reaction_time_ddm(
    drift: f64,
    threshold: f64,
    noise: f64,
    dt: f64,
    trials: usize,
    rng: &mut Rng,
) -> Result<Vec<(f64, bool)>, GeomError> {
    if !(threshold > 0.0) || !(noise > 0.0) || !(dt > 0.0) || trials == 0 {
        return Err(GeomError::InvalidArgument("reaction_time_ddm: bad parameters"));
    }
    // The budget is shared across trials rather than imposed on each. A
    // decision time has a long tail, so a per-trial cap would throw away
    // exactly the slow trials the distribution is about; a total budget
    // still catches a drift and noise so small that nothing ever decides.
    let mut budget = 50_000_000usize;
    let mut out = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut evidence = 0.0;
        let mut steps = 0usize;
        loop {
            evidence += drift * dt + noise * dt.sqrt() * rng.next_gaussian();
            steps += 1;
            if evidence >= threshold {
                out.push((steps as f64 * dt, true));
                break;
            }
            if evidence <= -threshold {
                out.push((steps as f64 * dt, false));
                break;
            }
            if steps > budget {
                return Err(GeomError::Degenerate(
                    "the evidence never reached a bound within the step budget",
                ));
            }
        }
        budget = budget.saturating_sub(steps);
    }
    Ok(out)
}

/// The exact probability that unbiased evidence reaches the positive
/// bound: `1 / (1 + exp(-2 * drift * threshold / noise^2))`.
///
/// This is the gambler's-ruin answer for Brownian motion with drift
/// between symmetric absorbing barriers, and it depends on the three
/// parameters only through `drift * threshold / noise^2`. Doubling the
/// drift and the noise variance together therefore changes the accuracy
/// not at all, only the time taken.
///
/// # Errors
/// Returns an error for a non-positive threshold or noise.
pub fn ddm_analytic_accuracy(drift: f64, threshold: f64, noise: f64) -> Result<f64, GeomError> {
    if !(threshold > 0.0) || !(noise > 0.0) {
        return Err(GeomError::InvalidArgument("ddm_analytic_accuracy: bad parameters"));
    }
    Ok(1.0 / (1.0 + (-2.0 * drift * threshold / (noise * noise)).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rectangular current pulse.
    fn pulse(amplitude: f64, start: f64, width: f64) -> impl Fn(f64) -> f64 {
        move |t: f64| if t >= start && t < start + width { amplitude } else { 0.0 }
    }

    fn peak(trace: &[(f64, f64, f64, f64, f64)]) -> f64 {
        trace.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn the_rate_constants_are_finite_where_their_formulas_are_not() {
        // alpha_m is 0/0 at -40 mV and alpha_n at -55 mV. Evaluated as
        // written they give NaN exactly there and lose precision nearby,
        // which a simulation will reach.
        for v in [-40.0, -55.0] {
            let rates = hh_rates(v);
            assert!(rates.iter().all(|r| r.is_finite() && *r >= 0.0), "rates at {v}: {rates:?}");
        }
        // And the continued values agree with the limit approached from
        // both sides.
        for (v, index) in [(-40.0, 0usize), (-55.0, 4)] {
            let here = hh_rates(v)[index];
            let below = hh_rates(v - 1e-4)[index];
            let above = hh_rates(v + 1e-4)[index];
            assert!(
                (here - 0.5 * (below + above)).abs() < 1e-8,
                "at {v} the rate {here} does not match its neighbours {below} and {above}"
            );
        }
        // alpha_m's limit at -40 mV is exactly 1 per ms.
        assert!((hh_rates(-40.0)[0] - 1.0).abs() < 1e-12);
        // alpha_n's at -55 mV is 0.1.
        assert!((hh_rates(-55.0)[4] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn an_unstimulated_axon_stays_where_it_started() {
        // The gates begin at their steady state, so there is no transient
        // to relax through and nothing to mistake for a response.
        let trace = hodgkin_huxley(&|_| 0.0, 50.0, 0.01).unwrap();
        assert!(hh_spike_times(&trace).is_empty());
        for row in &trace {
            assert!((row.1 - HH_V_REST).abs() < 0.02, "the resting voltage drifted to {}", row.1);
        }
        let (m, h, n) = hh_steady_state(HH_V_REST);
        assert!((trace[0].2 - m).abs() < 1e-15);
        assert!((trace[0].3 - h).abs() < 1e-15);
        assert!((trace[0].4 - n).abs() < 1e-15);
    }

    #[test]
    fn the_gating_variables_are_probabilities_throughout_a_spike() {
        // m, h and n are fractions of channels open. A value outside the
        // unit interval is meaningless, and an integrator that overshoots
        // would produce one.
        let trace = hodgkin_huxley(&|_| 15.0, 120.0, 0.01).unwrap();
        assert!(hh_spike_times(&trace).len() > 5);
        for row in &trace {
            for gate in [row.2, row.3, row.4] {
                assert!((0.0..=1.0).contains(&gate), "a gate reached {gate}");
            }
            assert!(row.1 > HH_E_K - 5.0 && row.1 < HH_E_NA + 5.0, "voltage left the reversals");
        }
    }

    #[test]
    fn the_action_potential_is_all_or_none() {
        // A half-millisecond pulse either fails entirely or produces a
        // full-sized spike. Doubling a suprathreshold pulse changes the
        // peak by a millivolt or two, not by a factor of two -- which is
        // the observation the whole conductance mechanism was built to
        // explain.
        let small = hodgkin_huxley(&pulse(10.0, 5.0, 0.5), 40.0, 0.01).unwrap();
        assert!(hh_spike_times(&small).is_empty());
        assert!(peak(&small) < -55.0, "a subthreshold pulse reached {}", peak(&small));

        let once = hodgkin_huxley(&pulse(20.0, 5.0, 0.5), 40.0, 0.01).unwrap();
        let twice = hodgkin_huxley(&pulse(40.0, 5.0, 0.5), 40.0, 0.01).unwrap();
        assert_eq!(hh_spike_times(&once).len(), 1);
        assert_eq!(hh_spike_times(&twice).len(), 1);
        assert!(peak(&once) > 30.0 && peak(&twice) > 30.0);
        assert!(
            (peak(&twice) - peak(&once)).abs() < 3.0,
            "doubling the stimulus moved the peak from {} to {}",
            peak(&once),
            peak(&twice)
        );
    }

    #[test]
    fn a_second_pulse_too_soon_after_the_first_produces_nothing() {
        // Refractoriness is not a rule in the model; it is inactivation h
        // having not yet recovered. A pulse that fires the resting axon
        // fails a few milliseconds after a spike and succeeds later.
        let count = |gap: f64| {
            let stimulus = move |t: f64| {
                if (5.0..5.5).contains(&t) || (5.0 + gap..5.5 + gap).contains(&t) {
                    20.0
                } else {
                    0.0
                }
            };
            hh_spike_times(&hodgkin_huxley(&stimulus, 60.0, 0.01).unwrap()).len()
        };
        assert_eq!(count(3.0), 1, "an early second pulse should be refused");
        assert_eq!(count(8.0), 1);
        assert_eq!(count(20.0), 2, "a late second pulse should succeed");
    }

    #[test]
    fn the_firing_threshold_and_the_repetitive_firing_threshold_are_different_numbers() {
        // A step of 2.24 uA/cm^2 makes the model fire once and then sit
        // still; repetitive firing needs nearly three times that. The
        // rheobase for "a spike" and the rheobase for "a spike train" are
        // not the same quantity, which the bistability around a subcritical
        // Hopf bifurcation is what produces.
        let threshold = hh_spike_threshold_estimate();
        assert!((2.0..2.6).contains(&threshold), "the estimate came out at {threshold}");

        let below = hodgkin_huxley(&|_| threshold * 0.98, 120.0, 0.01).unwrap();
        let above = hodgkin_huxley(&|_| threshold * 1.02, 120.0, 0.01).unwrap();
        assert!(hh_spike_times(&below).is_empty(), "it fired below its own estimate");
        assert!(!hh_spike_times(&above).is_empty(), "it did not fire above its own estimate");

        // But a sustained step at that current settles after the transient.
        let sustained = hh_fi_curve(&[threshold * 1.5]).unwrap();
        assert!(sustained[0].1 < 1.0, "it fired repeatedly at {}", sustained[0].0);
    }

    #[test]
    fn the_f_i_curve_starts_abruptly_and_then_rises() {
        // Type II excitability: the rate does not grow from zero. Between
        // 6.0 and 6.3 uA/cm^2 it goes from silence to about fifty hertz,
        // and no current produces a rate in between.
        let currents = [0.0, 3.0, 6.0, 6.3, 7.0, 10.0, 20.0];
        let curve = hh_fi_curve(&currents).unwrap();
        assert_eq!(curve.len(), currents.len());
        for (index, (current, rate)) in curve.iter().enumerate() {
            assert!((current - currents[index]).abs() < 1e-15);
            assert!(*rate >= 0.0);
        }
        assert_eq!(curve[0].1, 0.0);
        assert_eq!(curve[1].1, 0.0);
        assert_eq!(curve[2].1, 0.0, "6.0 uA/cm^2 should not fire repetitively");
        assert!(curve[3].1 > 40.0, "the onset rate was only {}", curve[3].1);
        for pair in curve.windows(2) {
            assert!(pair[1].1 >= pair[0].1 - 1e-9, "the curve went down");
        }
        assert!(curve.last().unwrap().1 < 200.0, "the rate is beyond what the model can do");
    }

    #[test]
    fn the_integrator_refuses_a_step_that_would_lose_the_upstroke() {
        assert!(hodgkin_huxley(&|_| 0.0, 10.0, 0.1).is_err());
        assert!(hodgkin_huxley(&|_| 0.0, 10.0, 0.0).is_err());
        assert!(hodgkin_huxley(&|_| 0.0, 10.0, -0.01).is_err());
        assert!(hodgkin_huxley(&|_| 0.0, 0.0, 0.01).is_err());
        assert!(hodgkin_huxley(&|_| 0.0, 0.005, 0.01).is_err());
        assert!(hh_fi_curve(&[]).is_err());
        assert!(hh_fi_curve(&[f64::NAN]).is_err());
    }

    #[test]
    fn fitzhugh_nagumo_decays_a_small_push_and_takes_an_excursion_from_a_larger_one() {
        // Excitability in two variables: the response is not proportional
        // to the stimulus, and the boundary between the two behaviours is
        // sharp.
        let rest = fitzhugh_nagumo_neuron(0.7, 0.8, 12.5, 0.0, -1.2, -0.62, 400.0, 0.05).unwrap();
        let (v_rest, w_rest) = (rest.last().unwrap().1, rest.last().unwrap().2);
        assert!((v_rest - -1.1994).abs() < 1e-3, "the rest point moved to {v_rest}");

        let response = |kick: f64| {
            let run =
                fitzhugh_nagumo_neuron(0.7, 0.8, 12.5, 0.0, v_rest + kick, w_rest, 200.0, 0.05)
                    .unwrap();
            run.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max)
        };
        // A small push never rises above where it was put.
        assert!((response(0.1) - (v_rest + 0.1)).abs() < 1e-6);
        assert!((response(0.5) - (v_rest + 0.5)).abs() < 0.2);
        // A larger one runs to the far branch of the cubic.
        assert!(response(0.8) > 1.5, "the large kick only reached {}", response(0.8));
    }

    #[test]
    fn a_current_above_the_bifurcation_makes_fitzhugh_nagumo_oscillate_forever() {
        let run = fitzhugh_nagumo_neuron(0.7, 0.8, 12.5, 0.5, -1.2, -0.62, 400.0, 0.05).unwrap();
        let tail: Vec<f64> = run.iter().filter(|r| r.0 > 200.0).map(|r| r.1).collect();
        let swing = tail.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            - tail.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        assert!(swing > 3.0, "the oscillation died back to a swing of {swing}");

        // With no current it settles instead, so the swing is a property
        // of the current and not of the initial condition.
        let quiet = fitzhugh_nagumo_neuron(0.7, 0.8, 12.5, 0.0, -1.2, -0.62, 400.0, 0.05).unwrap();
        let settled: Vec<f64> = quiet.iter().filter(|r| r.0 > 200.0).map(|r| r.1).collect();
        let residue = settled.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            - settled.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        assert!(residue < 1e-3, "the unstimulated model still swings by {residue}");
        assert!(fitzhugh_nagumo_neuron(0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 10.0, 0.05).is_err());
    }

    #[test]
    fn the_two_bifurcations_give_firing_rates_that_begin_differently() {
        // A saddle-node on an invariant circle is born with infinite
        // period, so a type I neuron can be tuned to fire arbitrarily
        // slowly. A Hopf bifurcation is born with a finite frequency, so a
        // type II neuron's rate jumps. The two parameter sets differ only
        // in the potassium gate.
        let rate = |params: &MorrisLecar, current: f64, span: f64| -> f64 {
            let run = morris_lecar(params, current, -60.0, 0.0, span, 0.05).unwrap();
            let trace: Vec<(f64, f64)> = run.iter().map(|r| (r.0, r.1)).collect();
            let counted =
                spike_times(&trace, 0.0).into_iter().filter(|t| *t > span * 0.3).count();
            1000.0 * counted as f64 / (span * 0.7)
        };
        let two = MorrisLecar::hopf();
        assert_eq!(rate(&two, 88.0, 3000.0), 0.0, "type II fired below its threshold");
        let onset_two = rate(&two, 90.0, 3000.0);
        assert!(onset_two > 5.0, "type II started at only {onset_two} Hz");

        let one = MorrisLecar::saddle_node();
        assert_eq!(rate(&one, 39.0, 6000.0), 0.0, "type I fired below its threshold");
        let onset_one = rate(&one, 40.0, 6000.0);
        assert!(onset_one > 0.0, "type I did not start firing");
        assert!(
            onset_one < 0.5 * onset_two,
            "type I began at {onset_one} Hz against type II's {onset_two} Hz"
        );
        // And type I's rate keeps climbing where type II's is already
        // nearly saturated.
        assert!(rate(&one, 60.0, 6000.0) > 3.0 * onset_one);
        assert!(rate(&two, 120.0, 3000.0) < 2.0 * onset_two);
    }

    #[test]
    fn morris_lecar_refuses_parameters_that_are_not_conductances() {
        let mut bad = MorrisLecar::hopf();
        bad.c_m = 0.0;
        assert!(morris_lecar(&bad, 50.0, -60.0, 0.0, 100.0, 0.05).is_err());
        bad = MorrisLecar::hopf();
        bad.v2 = 0.0;
        assert!(morris_lecar(&bad, 50.0, -60.0, 0.0, 100.0, 0.05).is_err());
        bad = MorrisLecar::hopf();
        bad.phi = -1.0;
        assert!(morris_lecar(&bad, 50.0, -60.0, 0.0, 100.0, 0.05).is_err());
        assert!(morris_lecar(&MorrisLecar::hopf(), 50.0, -60.0, 0.0, 100.0, 2.0).is_err());
    }

    #[test]
    fn each_izhikevich_preset_produces_the_pattern_it_is_named_for() {
        let presets = izhikevich_presets();
        assert_eq!(presets.len(), 5);
        let mut rates = std::collections::HashMap::new();
        let mut irregularity = std::collections::HashMap::new();
        for (name, p) in &presets {
            let run = izhikevich(p[0], p[1], p[2], p[3], 10.0, 400.0, 0.25).unwrap();
            assert!(run.iter().all(|r| r.1 <= 30.0 + 1e-9), "{name} overshot the peak");
            let spikes = spike_times(&run, 20.0);
            assert!(spikes.len() > 3, "{name} barely fired");
            rates.insert(*name, spikes.len());
            irregularity.insert(*name, cv_isi(&spikes).unwrap());
        }
        // Fast spiking is the fastest and the most regular of the five.
        assert!(rates["FS"] > rates["RS"], "FS did not outpace RS");
        assert!(rates["FS"] > rates["IB"]);
        assert!(irregularity["FS"] < 0.15, "FS was irregular at {}", irregularity["FS"]);
        // Chattering fires in bursts, which shows up as intervals of two
        // very different sizes and so a coefficient of variation above one.
        assert!(
            irregularity["CH"] > 1.0,
            "the chattering preset was regular at {}",
            irregularity["CH"]
        );
        assert!(irregularity["CH"] > 4.0 * irregularity["FS"]);
    }

    #[test]
    fn izhikevich_starts_at_rest_and_needs_a_current_to_fire() {
        let quiet = izhikevich(0.02, 0.2, -65.0, 8.0, 0.0, 200.0, 0.25).unwrap();
        assert!(spike_times(&quiet, 20.0).is_empty(), "it fired with no input");
        assert!((quiet[0].1 - -65.0).abs() < 1e-12);
        // The resting voltage is a fixed point of the quadratic, so it
        // stays put rather than drifting.
        assert!(quiet.iter().all(|r| r.1 < -50.0));
        assert!(izhikevich(0.0, 0.2, -65.0, 8.0, 10.0, 100.0, 0.25).is_err());
        assert!(izhikevich(0.02, 0.2, -65.0, 8.0, 10.0, 100.0, 2.0).is_err());
    }

    #[test]
    fn adaptation_lengthens_an_adex_spike_train_and_its_absence_does_not() {
        // Two mechanisms, both absent from a leaky integrator: `b` adds a
        // fixed current at every spike, `a` couples the current to voltage.
        // Either lengthens the intervals through a train; without them the
        // intervals are constant.
        let train = |a: f64, b: f64| -> Vec<f64> {
            let run =
                adex(200.0, 10.0, -70.0, 2.0, -50.0, 100.0, a, b, -58.0, 500.0, 400.0, 0.05)
                    .unwrap();
            let trace: Vec<(f64, f64)> = run.iter().map(|r| (r.0, r.1)).collect();
            interspike_intervals(&spike_times(&trace, -32.0)).unwrap()
        };
        let steady = train(0.0, 0.0);
        assert!(steady.len() > 20);
        let spread = steady.last().unwrap() - steady.first().unwrap();
        assert!(spread.abs() < 0.1, "an unadapting neuron drifted by {spread} ms");

        let spike_triggered = train(0.0, 60.0);
        assert!(spike_triggered.len() > 5);
        assert!(
            spike_triggered.last().unwrap() > &(2.0 * spike_triggered.first().unwrap()),
            "spike-triggered adaptation went from {:?} to {:?}",
            spike_triggered.first(),
            spike_triggered.last()
        );
        // And adaptation costs spikes: the adapting neuron fires fewer.
        assert!(spike_triggered.len() < steady.len());

        let subthreshold = train(4.0, 0.0);
        assert!(subthreshold.last().unwrap() > subthreshold.first().unwrap());
        assert!(adex(0.0, 10.0, -70.0, 2.0, -50.0, 100.0, 0.0, 0.0, -58.0, 500.0, 100.0, 0.05).is_err());
        assert!(adex(200.0, 10.0, -70.0, 0.0, -50.0, 100.0, 0.0, 0.0, -58.0, 500.0, 100.0, 0.05).is_err());
    }

    #[test]
    fn the_simulated_leaky_integrator_fires_at_the_rate_the_formula_gives() {
        // Noiseless, the interspike interval is the time for an
        // exponential charging curve to cross threshold, and that has a
        // closed form. Simulation and formula must agree to the resolution
        // of the step.
        let mut rng = Rng::new(0x0E0E_1001);
        for current in [1.05f64, 1.2, 2.0, 5.0, 20.0] {
            let spikes =
                lif_neuron(current, 10.0, 1.0, 0.0, 2.0, 0.0, 4000.0, 0.005, &mut rng).unwrap();
            let simulated = spikes.len() as f64 / 4000.0;
            let exact = lif_fi_exact(current, 10.0, 1.0, 0.0, 2.0).unwrap();
            assert!(
                (simulated - exact).abs() < 0.02 * exact,
                "at I={current} simulation gave {simulated} against {exact}"
            );
        }
        // Below threshold it never fires, exactly.
        let silent = lif_neuron(0.99, 10.0, 1.0, 0.0, 2.0, 0.0, 2000.0, 0.01, &mut rng).unwrap();
        assert!(silent.is_empty());
        assert_eq!(lif_fi_exact(0.99, 10.0, 1.0, 0.0, 2.0).unwrap(), 0.0);
        assert_eq!(lif_fi_exact(1.0, 10.0, 1.0, 0.0, 2.0).unwrap(), 0.0);
    }

    #[test]
    fn the_leaky_integrator_saturates_at_the_refractory_period() {
        // However large the current, the rate cannot exceed one spike per
        // refractory period. The logarithm is what enforces it.
        let ceiling = 1.0 / 2.0;
        let mut previous = 0.0;
        for current in [1.5f64, 3.0, 10.0, 100.0, 1e6] {
            let rate = lif_fi_exact(current, 10.0, 1.0, 0.0, 2.0).unwrap();
            assert!(rate > previous, "the curve was not increasing at {current}");
            assert!(rate < ceiling, "at {current} the rate {rate} beat the refractory limit");
            previous = rate;
        }
        assert!((lif_fi_exact(1e12, 10.0, 1.0, 0.0, 2.0).unwrap() - ceiling).abs() < 1e-6);
        // Without a refractory period there is no ceiling.
        assert!(lif_fi_exact(1e6, 10.0, 1.0, 0.0, 0.0).unwrap() > 100.0);
        assert!(lif_fi_exact(2.0, 0.0, 1.0, 0.0, 1.0).is_err());
        assert!(lif_fi_exact(2.0, 10.0, 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn noise_makes_a_subthreshold_leaky_integrator_fire_anyway() {
        // A current below threshold produces no spikes at all in the
        // deterministic model; with noise the same current fires at a low
        // rate, which is how a stochastic neuron has no hard threshold.
        let mut rng = Rng::new(0x0E0E_1002);
        let quiet = lif_neuron(0.9, 10.0, 1.0, 0.0, 2.0, 0.0, 5000.0, 0.01, &mut rng).unwrap();
        assert!(quiet.is_empty());
        let noisy = lif_neuron(0.9, 10.0, 1.0, 0.0, 2.0, 0.5, 5000.0, 0.01, &mut rng).unwrap();
        assert!(!noisy.is_empty(), "noise produced no spikes at all");
        // And the firing it produces is irregular, unlike the
        // deterministic case where every interval is identical.
        assert!(cv_isi(&noisy).unwrap() > 0.3, "the noisy train was suspiciously regular");
        assert!(lif_neuron(1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.01, &mut rng).is_err());
        assert!(lif_neuron(1.0, 10.0, 0.0, 1.0, 0.0, 0.0, 10.0, 0.01, &mut rng).is_err());
        assert!(lif_neuron(1.0, 10.0, 1.0, 0.0, -1.0, 0.0, 10.0, 0.01, &mut rng).is_err());
    }

    #[test]
    fn a_poisson_train_has_the_rate_and_the_irregularity_it_should() {
        // Both statistics are one for a Poisson process, and they are
        // computed by different routes -- one from intervals, one from
        // counts in windows -- so agreeing is a real check.
        let mut rng = Rng::new(0x0E0E_1003);
        let rate = 0.05;
        let span = 200_000.0;
        let train = poisson_spike_train(rate, span, &mut rng).unwrap();
        let observed = train.len() as f64 / span;
        assert!((observed - rate).abs() < 0.05 * rate, "the rate came out at {observed}");
        assert!(train.windows(2).all(|p| p[1] > p[0]), "the train is not ordered");
        assert!(train.iter().all(|t| (0.0..span).contains(t)));

        let cv = cv_isi(&train).unwrap();
        assert!((cv - 1.0).abs() < 0.05, "the coefficient of variation was {cv}");

        let window = 100.0;
        let bins = (span / window) as usize;
        let mut counts = vec![0u64; bins];
        for spike in &train {
            counts[((spike / window) as usize).min(bins - 1)] += 1;
        }
        let fano = fano_factor(&counts).unwrap();
        assert!((fano - 1.0).abs() < 0.1, "the Fano factor was {fano}");
    }

    #[test]
    fn the_two_irregularity_measures_answer_different_questions() {
        // A perfectly regular train has a coefficient of variation of zero
        // and a Fano factor of zero. A train that is regular within each
        // block but changes rate between them still has a low CV and a
        // large Fano factor, because the variability is between windows.
        let regular: Vec<f64> = (0..1000).map(|k| k as f64 * 10.0).collect();
        assert!(cv_isi(&regular).unwrap() < 1e-12);
        let counts: Vec<u64> = (0..100).map(|_| 10u64).collect();
        assert!(fano_factor(&counts).unwrap() < 1e-12);

        let mut drifting: Vec<f64> = Vec::new();
        let mut t = 0.0f64;
        for block in 0..40 {
            let gap = if block % 2 == 0 { 4.0 } else { 40.0 };
            for _ in 0..25 {
                drifting.push(t);
                t += gap;
            }
        }
        let block_counts: Vec<u64> = (0..40).map(|_| 25u64).collect();
        assert!(fano_factor(&block_counts).unwrap() < 1e-12);
        // Counted in fixed windows instead, the same train is bursty.
        let window = 200.0;
        let bins = (t / window).ceil() as usize;
        let mut windowed = vec![0u64; bins];
        for spike in &drifting {
            windowed[((spike / window) as usize).min(bins - 1)] += 1;
        }
        assert!(fano_factor(&windowed).unwrap() > 5.0, "the drifting train looked Poisson");
    }

    #[test]
    fn the_irregularity_measures_reject_what_they_cannot_describe() {
        assert!(interspike_intervals(&[1.0, 0.5]).is_err());
        assert!(cv_isi(&[1.0, 2.0]).is_err());
        assert!(cv_isi(&[1.0, 1.0, 1.0]).is_err());
        assert!(fano_factor(&[3]).is_err());
        assert!(fano_factor(&[0, 0, 0]).is_err());
        assert!(interspike_intervals(&[]).unwrap().is_empty());
        let mut rng = Rng::new(1);
        assert!(poisson_spike_train(0.0, 10.0, &mut rng).is_err());
        assert!(poisson_spike_train(1.0, 0.0, &mut rng).is_err());
        assert!(poisson_spike_train(1e9, 1e9, &mut rng).is_err());
    }

    #[test]
    fn a_histogram_of_poisson_trials_recovers_the_rate_it_was_drawn_from() {
        // The PSTH divides by the bin width and the trial count, so a
        // constant-rate process gives back that rate however it is binned.
        let mut rng = Rng::new(0x0E0E_1004);
        let rate = 0.08;
        let span = 500.0;
        let trains: Vec<Vec<f64>> =
            (0..400).map(|_| poisson_spike_train(rate, span, &mut rng).unwrap()).collect();
        for bin in [5.0, 25.0, 100.0] {
            let histogram = psth(&trains, bin, span).unwrap();
            assert_eq!(histogram.len(), (span / bin) as usize);
            let mean = histogram.iter().sum::<f64>() / histogram.len() as f64;
            assert!((mean - rate).abs() < 0.1 * rate, "bin {bin} gave a mean rate of {mean}");
        }
        // The histogram's integral is the mean spike count per trial.
        let bin = 10.0;
        let histogram = psth(&trains, bin, span).unwrap();
        let integral: f64 = histogram.iter().map(|r| r * bin).sum();
        let counted = trains.iter().map(Vec::len).sum::<usize>() as f64 / trains.len() as f64;
        assert!((integral - counted).abs() < 1e-9, "{integral} against {counted}");
    }

    #[test]
    fn the_raster_keeps_every_spike_and_the_trial_it_came_from() {
        let trains = vec![vec![1.0, 4.0, 9.0], vec![2.0, 3.0], vec![], vec![0.5, 7.0]];
        let raster = raster_data(&trains);
        assert_eq!(raster.len(), 7);
        assert!(raster.windows(2).all(|p| p[1].0 >= p[0].0), "the raster is not sorted");
        for (trial, train) in trains.iter().enumerate() {
            for spike in train {
                assert!(raster.contains(&(*spike, trial)), "{spike} from trial {trial} was lost");
            }
        }
        assert!(raster_data(&[]).is_empty());
        assert!(psth(&[], 1.0, 10.0).is_err());
        assert!(psth(&trains, 0.0, 10.0).is_err());
        assert!(psth(&trains, 20.0, 10.0).is_err());
        assert!(psth(&trains, 1.0, 5.0).is_err(), "a spike outside the window was accepted");
    }

    #[test]
    fn the_spike_triggered_average_recovers_a_feature_the_spikes_were_locked_to() {
        // Spikes placed exactly where a marker was written into otherwise
        // random noise must return that marker, scaled down by nothing
        // because every spike saw it.
        let mut rng = Rng::new(0x0E0E_1005);
        let dt = 1.0;
        let marker = [-1.0, 0.5, 2.0, 3.0];
        let mut stimulus: Vec<f64> = (0..4000).map(|_| rng.next_gaussian()).collect();
        let mut spikes = Vec::new();
        let mut at = 50usize;
        while at + marker.len() < stimulus.len() {
            stimulus[at..at + marker.len()].copy_from_slice(&marker);
            spikes.push((at + marker.len() - 1) as f64 * dt);
            at += 40;
        }
        let average = spike_triggered_average(&stimulus, dt, &spikes, marker.len()).unwrap();
        for (got, want) in average.iter().zip(marker.iter()) {
            assert!((got - want).abs() < 1e-9, "recovered {average:?} not {marker:?}");
        }

        // Spikes unrelated to the stimulus average to nothing.
        let noise: Vec<f64> = (0..40_000).map(|_| rng.next_gaussian()).collect();
        let scattered: Vec<f64> = (0..8000).map(|k| (10 + 4 * k) as f64).collect();
        let flat = spike_triggered_average(&noise, 1.0, &scattered, 6).unwrap();
        for value in &flat {
            assert!(value.abs() < 0.1, "an unrelated average came out at {value}");
        }
    }

    #[test]
    fn the_spike_triggered_average_refuses_what_it_cannot_average() {
        let stimulus: Vec<f64> = (0..20).map(|k| k as f64).collect();
        assert!(spike_triggered_average(&[], 1.0, &[5.0], 3).is_err());
        assert!(spike_triggered_average(&stimulus, 0.0, &[5.0], 3).is_err());
        assert!(spike_triggered_average(&stimulus, 1.0, &[5.0], 0).is_err());
        assert!(spike_triggered_average(&stimulus, 1.0, &[5.0], 30).is_err());
        assert!(spike_triggered_average(&stimulus, 1.0, &[-1.0], 3).is_err());
        // Every spike too early for a full window is a degenerate request,
        // not a silently shortened average.
        assert!(spike_triggered_average(&stimulus, 1.0, &[1.0], 5).is_err());
        assert!(spike_triggered_average(&stimulus, 1.0, &[], 3).is_err());
        // The window ends at the spike itself.
        let average = spike_triggered_average(&stimulus, 1.0, &[10.0], 3).unwrap();
        assert_eq!(average, vec![8.0, 9.0, 10.0]);
    }

    #[test]
    fn the_von_mises_fit_is_exact_on_data_that_came_from_a_von_mises_curve() {
        // The log form is linear in three coefficients, so a noiseless
        // curve is recovered by solving rather than by searching.
        let angles: Vec<f64> = (0..16)
            .map(|k| -std::f64::consts::PI + k as f64 * std::f64::consts::TAU / 16.0)
            .collect();
        for (preferred, kappa, amplitude) in
            [(0.7f64, 2.5f64, 3.0f64), (-2.0, 0.4, 12.0), (3.0, 8.0, 0.05)]
        {
            let rates: Vec<f64> =
                angles.iter().map(|a| amplitude * (kappa * (a - preferred).cos()).exp()).collect();
            let (mu, k, amp) = tuning_curve_fit_von_mises(&angles, &rates).unwrap();
            let offset = (mu - preferred).sin().atan2((mu - preferred).cos()).abs();
            assert!(offset < 1e-9, "preferred angle {mu} against {preferred}");
            assert!((k - kappa).abs() < 1e-9, "concentration {k} against {kappa}");
            assert!((amp - amplitude).abs() < 1e-9 * amplitude, "amplitude {amp}");
        }
    }

    #[test]
    fn the_von_mises_fit_is_blind_to_the_turn_of_the_circle_and_scales_with_the_rates() {
        let angles: Vec<f64> = (0..12)
            .map(|k| -std::f64::consts::PI + k as f64 * std::f64::consts::TAU / 12.0)
            .collect();
        let rates: Vec<f64> = angles.iter().map(|a| 4.0 * (1.8 * (a - 0.3).cos()).exp()).collect();
        let (mu, kappa, amplitude) = tuning_curve_fit_von_mises(&angles, &rates).unwrap();

        // Shifting every angle by a full turn is the same measurement.
        let turned: Vec<f64> = angles.iter().map(|a| a + std::f64::consts::TAU).collect();
        let (mu2, kappa2, amplitude2) = tuning_curve_fit_von_mises(&turned, &rates).unwrap();
        assert!((mu - mu2).abs() < 1e-8 && (kappa - kappa2).abs() < 1e-8);
        assert!((amplitude - amplitude2).abs() < 1e-8);

        // Doubling every rate doubles the amplitude and moves nothing else.
        let doubled: Vec<f64> = rates.iter().map(|r| 2.0 * r).collect();
        let (mu3, kappa3, amplitude3) = tuning_curve_fit_von_mises(&angles, &doubled).unwrap();
        assert!((mu - mu3).abs() < 1e-8 && (kappa - kappa3).abs() < 1e-8);
        assert!((amplitude3 - 2.0 * amplitude).abs() < 1e-8);

        // A flat curve has no preferred direction, which shows as a
        // concentration of zero rather than an arbitrary angle.
        let flat = vec![5.0; angles.len()];
        let (_, kappa_flat, amplitude_flat) =
            tuning_curve_fit_von_mises(&angles, &flat).unwrap();
        assert!(kappa_flat < 1e-9, "a flat curve claimed a concentration of {kappa_flat}");
        assert!((amplitude_flat - 5.0).abs() < 1e-9);
    }

    #[test]
    fn the_von_mises_fit_refuses_data_that_does_not_determine_it() {
        assert!(tuning_curve_fit_von_mises(&[0.0, 1.0], &[1.0, 2.0]).is_err());
        assert!(tuning_curve_fit_von_mises(&[0.0, 1.0, 2.0], &[1.0, 2.0]).is_err());
        assert!(tuning_curve_fit_von_mises(&[0.0, 1.0, 2.0], &[1.0, 0.0, 2.0]).is_err());
        assert!(tuning_curve_fit_von_mises(&[0.0, 1.0, 2.0], &[1.0, -1.0, 2.0]).is_err());
        assert!(tuning_curve_fit_von_mises(&[0.0, f64::NAN, 2.0], &[1.0, 1.0, 2.0]).is_err());
        // Three measurements at the same angle cannot fix three
        // coefficients, whatever the rates.
        assert!(tuning_curve_fit_von_mises(&[0.5, 0.5, 0.5], &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn the_alpha_synapse_peaks_at_its_maximum_exactly_one_time_constant_late() {
        // g_max * x * exp(1 - x) equals g_max at x = 1 and less everywhere
        // else, which is what makes g_max a peak conductance rather than
        // an amplitude with no direct meaning.
        let tau = 3.0;
        let g_max = 0.7;
        assert!((alpha_synapse(g_max, tau, &[0.0], tau).unwrap() - g_max).abs() < 1e-12);
        for t in [0.5f64, 1.0, 2.0, 4.0, 8.0, 20.0] {
            let value = alpha_synapse(g_max, tau, &[0.0], t).unwrap();
            assert!(value <= g_max + 1e-12, "at {t} the conductance reached {value}");
            assert!(value >= 0.0);
        }
        // It starts at zero, unlike the exponential synapse, which jumps.
        assert!(alpha_synapse(g_max, tau, &[0.0], 0.0).unwrap().abs() < 1e-15);
        assert!((synapse_exp(g_max, tau, &[0.0], 0.0).unwrap() - g_max).abs() < 1e-15);
    }

    #[test]
    fn the_exponential_synapse_decays_by_half_every_tau_ln_two() {
        let tau = 5.0;
        let half_life = tau * std::f64::consts::LN_2;
        let mut expected = 1.0;
        for k in 0..6 {
            let value = synapse_exp(1.0, tau, &[0.0], k as f64 * half_life).unwrap();
            assert!((value - expected).abs() < 1e-12, "at half-life {k} it was {value}");
            expected *= 0.5;
        }
        // Nothing before the spike.
        assert_eq!(synapse_exp(1.0, tau, &[10.0], 9.99).unwrap(), 0.0);
    }

    #[test]
    fn synaptic_conductances_add_so_a_burst_outweighs_a_single_spike() {
        // Superposition is what makes a synapse integrate rather than
        // repeat: three spikes inside a time constant leave more
        // conductance behind than any one of them could.
        let tau = 10.0;
        let burst = [0.0, 2.0, 4.0];
        for t in [5.0f64, 12.0, 30.0] {
            let together = synapse_exp(1.0, tau, &burst, t).unwrap();
            let apart: f64 =
                burst.iter().map(|s| synapse_exp(1.0, tau, &[*s], t).unwrap()).sum();
            assert!((together - apart).abs() < 1e-12);
            assert!(together > synapse_exp(1.0, tau, &[burst[0]], t).unwrap());
        }
        let alpha_together = alpha_synapse(1.0, tau, &burst, 8.0).unwrap();
        let alpha_apart: f64 =
            burst.iter().map(|s| alpha_synapse(1.0, tau, &[*s], 8.0).unwrap()).sum();
        assert!((alpha_together - alpha_apart).abs() < 1e-12);

        assert!(synapse_exp(1.0, 0.0, &[0.0], 1.0).is_err());
        assert!(alpha_synapse(1.0, -1.0, &[0.0], 1.0).is_err());
        assert!(synapse_exp(1.0, 1.0, &[3.0, 1.0], 5.0).is_err());
    }

    #[test]
    fn the_plasticity_window_changes_sign_across_a_zero_millisecond_gap() {
        // Order decides direction. The rule's whole content is that a
        // millisecond either way separates strengthening from weakening,
        // and the discontinuity at zero is where that lives.
        let (a_plus, a_minus, tau_plus, tau_minus) = (0.01, 0.012, 20.0, 20.0);
        let at = |d: f64| stdp_window(d, a_plus, a_minus, tau_plus, tau_minus).unwrap();
        assert_eq!(at(0.0), 0.0);
        assert!(at(1e-9) > 0.0 && at(-1e-9) < 0.0);
        assert!((at(1e-9) - a_plus).abs() < 1e-9);
        assert!((at(-1e-9) + a_minus).abs() < 1e-9);
        // Both sides decay, and neither ever changes sign again.
        let mut previous = a_plus;
        for delta in [5.0f64, 10.0, 20.0, 60.0] {
            assert!(at(delta) < previous && at(delta) > 0.0);
            assert!(at(-delta) > -previous && at(-delta) < 0.0);
            previous = at(delta);
        }
        // Depression outweighing potentiation is what keeps the rule from
        // running away, and it is a choice of amplitudes, not a law.
        let balance: f64 = (1..2000).map(|k| at(k as f64 * 0.1) + at(-(k as f64) * 0.1)).sum();
        assert!(balance < 0.0, "the window integrates to {balance}, which would only potentiate");
        assert!(stdp_window(1.0, 0.01, 0.01, 0.0, 20.0).is_err());
        assert!(stdp_window(1.0, -0.01, 0.01, 20.0, 20.0).is_err());
    }

    #[test]
    fn causal_pairing_potentiates_and_reversing_the_order_depresses() {
        // The same two trains, with the postsynaptic one shifted either
        // side of the presynaptic one.
        let pre: Vec<f64> = (0..20).map(|k| k as f64 * 50.0).collect();
        let causal: Vec<f64> = pre.iter().map(|t| t + 5.0).collect();
        let anticausal: Vec<f64> = pre.iter().map(|t| t - 5.0).collect();
        let up = stdp_train(&pre, &causal, 0.01, 0.012, 20.0, 20.0).unwrap();
        let down = stdp_train(&pre, &anticausal, 0.01, 0.012, 20.0, 20.0).unwrap();
        assert!(up > 0.0, "causal pairing gave {up}");
        assert!(down < 0.0, "anticausal pairing gave {down}");

        // And the total is exactly the sum over pairs, not an approximation.
        let mut by_hand = 0.0;
        for before in &pre {
            for after in &causal {
                by_hand += stdp_window(after - before, 0.01, 0.012, 20.0, 20.0).unwrap();
            }
        }
        assert!((up - by_hand).abs() < 1e-12);
        assert!(stdp_train(&[2.0, 1.0], &causal, 0.01, 0.012, 20.0, 20.0).is_err());
        assert_eq!(stdp_train(&[], &causal, 0.01, 0.012, 20.0, 20.0).unwrap(), 0.0);
    }

    #[test]
    fn a_stored_pattern_is_a_fixed_point_of_the_network_that_stored_it() {
        let mut rng = Rng::new(0x0E0E_2001);
        let n = 80;
        let patterns: Vec<Vec<i8>> = (0..4)
            .map(|_| (0..n).map(|_| if rng.next_f64() < 0.5 { -1i8 } else { 1 }).collect())
            .collect();
        let w = hopfield_store(&patterns).unwrap();
        // The weight matrix is symmetric with a zero diagonal, which is
        // what makes the energy a Lyapunov function.
        for i in 0..n {
            assert!(w.get(i, i).abs() < 1e-15);
            for j in 0..n {
                assert!((w.get(i, j) - w.get(j, i)).abs() < 1e-15);
            }
        }
        for pattern in &patterns {
            assert_eq!(&hopfield_recall(&w, pattern, 10).unwrap(), pattern);
            // Flipping every unit leaves the energy alone, so the negative
            // of a stored pattern is stored too, whether or not you wanted
            // it.
            let mirrored: Vec<i8> = pattern.iter().map(|s| -s).collect();
            assert_eq!(hopfield_recall(&w, &mirrored, 10).unwrap(), mirrored);
            assert!(
                (hopfield_energy(&w, pattern).unwrap()
                    - hopfield_energy(&w, &mirrored).unwrap())
                .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn recall_lowers_the_energy_and_repairs_a_corrupted_probe_almost_always() {
        // The energy claim is a theorem and holds every time; the repair
        // is a statistical property of the basins and does not.
        let mut rng = Rng::new(0x0E0E_2002);
        let n = 100;
        let (mut repaired, mut attempts) = (0usize, 0usize);
        for _ in 0..12 {
            let patterns: Vec<Vec<i8>> = (0..5)
                .map(|_| (0..n).map(|_| if rng.next_f64() < 0.5 { -1i8 } else { 1 }).collect())
                .collect();
            let w = hopfield_store(&patterns).unwrap();
            for pattern in &patterns {
                let mut probe = pattern.clone();
                for slot in probe.iter_mut() {
                    if rng.next_f64() < 0.2 {
                        *slot = -*slot;
                    }
                }
                let before = hopfield_energy(&w, &probe).unwrap();
                let recalled = hopfield_recall(&w, &probe, 50).unwrap();
                let after = hopfield_energy(&w, &recalled).unwrap();
                assert!(after <= before + 1e-9, "the energy rose from {before} to {after}");
                // Whatever it settled on is a fixed point.
                assert_eq!(hopfield_recall(&w, &recalled, 50).unwrap(), recalled);
                attempts += 1;
                if recalled == *pattern {
                    repaired += 1;
                }
            }
        }
        let fraction = repaired as f64 / attempts as f64;
        assert!(fraction > 0.95, "a fifth of the bits flipped was repaired only {fraction} of the time");
    }

    #[test]
    fn recall_can_settle_in_a_spurious_state_deeper_than_the_pattern_it_came_from() {
        // Descending the energy finds *a* minimum, not the right one.
        // Storing patterns writes minima into the landscape but does not
        // stop others appearing, and one of those can be deeper than the
        // pattern whose neighbourhood the probe started in -- after which
        // no amount of further recall finds the way back.
        let mut rng = Rng::new(0x0E0E_2007);
        let n = 100;
        let mut found = None;
        'search: for _ in 0..40 {
            let patterns: Vec<Vec<i8>> = (0..8)
                .map(|_| (0..n).map(|_| if rng.next_f64() < 0.5 { -1i8 } else { 1 }).collect())
                .collect();
            let w = hopfield_store(&patterns).unwrap();
            for pattern in &patterns {
                let mut probe = pattern.clone();
                for slot in probe.iter_mut() {
                    if rng.next_f64() < 0.25 {
                        *slot = -*slot;
                    }
                }
                let recalled = hopfield_recall(&w, &probe, 50).unwrap();
                let stored = patterns
                    .iter()
                    .any(|p| recalled == *p || recalled.iter().zip(p).all(|(a, b)| *a == -*b));
                let deeper = hopfield_energy(&w, &recalled).unwrap()
                    < hopfield_energy(&w, pattern).unwrap() - 1e-9;
                if !stored && deeper {
                    found = Some((w, recalled, pattern.clone(), patterns.clone()));
                    break 'search;
                }
            }
        }
        let (w, recalled, pattern, patterns) =
            found.expect("no spurious minimum turned up in forty attempts");
        // It is a genuine fixed point, and none of the stored patterns.
        assert_eq!(hopfield_recall(&w, &recalled, 50).unwrap(), recalled);
        for stored in &patterns {
            assert_ne!(&recalled, stored);
        }
        assert!(hopfield_energy(&w, &recalled).unwrap() < hopfield_energy(&w, &pattern).unwrap());
    }

    #[test]
    fn hopfield_capacity_collapses_near_fourteen_percent_of_the_units() {
        // Below the load the stored patterns are stable; above it the
        // crosstalk wins and the network forgets nearly everything. The
        // collapse is abrupt, which is the point: capacity is a cliff, not
        // a gradual decline.
        let mut rng = Rng::new(0x0E0E_2003);
        let n = 100;
        let easy = hopfield_capacity_check(n, 5, 6, &mut rng).unwrap();
        let critical = hopfield_capacity_check(n, 14, 3, &mut rng).unwrap();
        let overloaded = hopfield_capacity_check(n, 30, 3, &mut rng).unwrap();
        assert!(easy > 0.98, "a light load recalled only {easy}");
        assert!(critical < easy, "the critical load did no worse than the light one");
        assert!(overloaded < 0.1, "an overloaded network still recalled {overloaded}");
        assert!((0.0..=1.0).contains(&critical));

        assert!(hopfield_store(&[]).is_err());
        assert!(hopfield_store(&[vec![1, -1], vec![1]]).is_err());
        assert!(hopfield_store(&[vec![1, 0]]).is_err());
        let w = hopfield_store(&[vec![1, -1, 1]]).unwrap();
        assert!(hopfield_recall(&w, &[1, -1], 5).is_err());
        assert!(hopfield_recall(&w, &[1, 0, 1], 5).is_err());
        assert!(hopfield_energy(&w, &[1, -1]).is_err());
        assert!(hopfield_capacity_check(0, 2, 2, &mut rng).is_err());
        assert!(hopfield_capacity_check(600, 2, 2, &mut rng).is_err());
        assert!(hopfield_capacity_check(10, 0, 2, &mut rng).is_err());
    }

    #[test]
    fn the_izhikevich_network_fires_at_a_cortical_rate_without_running_away() {
        let mut rng = Rng::new(0x0E0E_2004);
        let (excitatory, inhibitory, span) = (80usize, 20usize, 400.0);
        let spikes = izhikevich_network(excitatory, inhibitory, span, &mut rng).unwrap();
        let neurons = excitatory + inhibitory;
        let rate = 1000.0 * spikes.len() as f64 / (neurons as f64 * span);
        assert!((0.5..60.0).contains(&rate), "the network fired at {rate} Hz per neuron");
        assert!(spikes.iter().all(|s| s.1 < neurons && (0.0..span).contains(&s.0)));
        // Both populations take part; a network where only one fires has
        // lost the loop that makes it interesting.
        assert!(spikes.iter().any(|s| s.1 < excitatory));
        assert!(spikes.iter().any(|s| s.1 >= excitatory));

        assert!(izhikevich_network(0, 20, 100.0, &mut rng).is_err());
        assert!(izhikevich_network(80, 0, 100.0, &mut rng).is_err());
        assert!(izhikevich_network(3000, 2000, 100.0, &mut rng).is_err());
        assert!(izhikevich_network(80, 20, 0.0, &mut rng).is_err());
    }

    #[test]
    fn wilson_cowan_activities_stay_fractions_and_settle_when_the_loop_is_weak() {
        let run =
            wilson_cowan(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.2, 50.0, 0.01)
                .unwrap();
        for row in &run {
            assert!((0.0..=1.0).contains(&row.1), "E left the unit interval at {}", row.1);
            assert!((0.0..=1.0).contains(&row.2), "I left the unit interval at {}", row.2);
        }
        let tail: Vec<f64> = run.iter().filter(|r| r.0 > 30.0).map(|r| r.1).collect();
        let swing = tail.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            - tail.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        assert!(swing < 1e-6, "a weakly coupled pair still swings by {swing}");
        assert!(wilson_cowan(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 10.0, 0.01).is_err());
        assert!(wilson_cowan(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.5, 0.5, 10.0, 0.01).is_err());
    }

    #[test]
    fn a_strong_excitatory_inhibitory_loop_oscillates_where_a_weak_one_does_not() {
        // Neither population oscillates alone. The rhythm comes from
        // excitation driving inhibition which then shuts the excitation
        // down, and it needs the gain to be steep enough -- 1.3 here, but
        // not 1.0, with everything else held fixed.
        let swing = |slope: f64| -> f64 {
            let run = wilson_cowan(
                16.0, 12.0, 15.0, 3.0, 1.25, 0.0, 1.0, 1.0, slope, 4.0, 0.2, 0.1, 300.0, 0.01,
            )
            .unwrap();
            let tail: Vec<f64> = run.iter().filter(|r| r.0 > 200.0).map(|r| r.1).collect();
            tail.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
                - tail.iter().fold(f64::INFINITY, |a, b| a.min(*b))
        };
        assert!(swing(1.0) < 1e-6, "the shallow response oscillated by {}", swing(1.0));
        assert!(swing(1.3) > 0.2, "the steep response only swung by {}", swing(1.3));

        // Cutting the inhibitory feedback stops it: the loop, not the
        // excitatory population, is the oscillator.
        let run = wilson_cowan(
            16.0, 0.0, 0.0, 3.0, 1.25, 0.0, 1.0, 1.0, 1.3, 4.0, 0.2, 0.1, 300.0, 0.01,
        )
        .unwrap();
        let tail: Vec<f64> = run.iter().filter(|r| r.0 > 200.0).map(|r| r.1).collect();
        let residue = tail.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            - tail.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        assert!(residue < 1e-6, "the excitatory population oscillated alone, by {residue}");
    }

    #[test]
    fn the_cable_solution_matches_the_hyperbolic_cosine_and_converges_at_second_order() {
        // The discretised system is solved with the boundary conditions
        // imposed, so agreeing with the closed form is a real check on
        // both -- and halving the spacing must quarter the error.
        let (length, lambda) = (2.0, 0.5);
        let analytic = |x: f64| ((length - x) / lambda).cosh() / (length / lambda).cosh();
        let worst = |points: usize| -> f64 {
            let v = cable_equation_1d(length, lambda, 1.0, points).unwrap();
            (0..points)
                .map(|i| {
                    let x = length * i as f64 / (points - 1) as f64;
                    (v[i] - analytic(x)).abs()
                })
                .fold(0.0, f64::max)
        };
        let coarse = worst(101);
        let fine = worst(201);
        let finer = worst(401);
        assert!(coarse < 2e-3, "the coarse grid was off by {coarse}");
        let ratio = coarse / fine;
        assert!((3.5..4.5).contains(&ratio), "halving the spacing cut the error by {ratio}");
        assert!((fine / finer > 3.5) && (fine / finer < 4.5));
    }

    #[test]
    fn a_sealed_end_holds_the_voltage_up_where_a_long_cable_would_have_decayed() {
        // Current that reaches a sealed end has nowhere to go. On a cable
        // four length constants long the end effect is negligible; on one
        // a single length constant long it is not.
        let short = cable_equation_1d(0.5, 0.5, 1.0, 201).unwrap();
        let long = cable_equation_1d(4.0, 0.5, 1.0, 1601).unwrap();
        assert!((short[0] - 1.0).abs() < 1e-12 && (long[0] - 1.0).abs() < 1e-12);
        // The infinite-cable answer one length constant out is exp(-1).
        let infinite = (-1.0f64).exp();
        assert!(short.last().unwrap() > &infinite, "the sealed end did not hold the voltage up");
        let at_lambda = long[200];
        assert!((at_lambda - infinite).abs() < 1e-3, "a long cable gave {at_lambda} not {infinite}");
        // And the profile falls monotonically in both cases.
        assert!(short.windows(2).all(|p| p[1] <= p[0] + 1e-12));
        assert!(long.windows(2).all(|p| p[1] <= p[0] + 1e-12));

        assert!(cable_equation_1d(0.0, 0.5, 1.0, 10).is_err());
        assert!(cable_equation_1d(1.0, 0.0, 1.0, 10).is_err());
        assert!(cable_equation_1d(1.0, 0.5, 1.0, 2).is_err());
    }

    #[test]
    fn the_length_constant_grows_with_the_square_root_of_the_diameter() {
        // Which is why a thin neurite is electrically short: four times
        // the diameter buys only twice the reach.
        let base = length_constant(20_000.0, 100.0, 1e-4).unwrap();
        let wider = length_constant(20_000.0, 100.0, 4e-4).unwrap();
        assert!((wider / base - 2.0).abs() < 1e-12);
        // Raising the membrane resistance has the same square-root effect,
        // and raising the axial resistance the inverse one.
        assert!((length_constant(80_000.0, 100.0, 1e-4).unwrap() / base - 2.0).abs() < 1e-12);
        assert!((length_constant(20_000.0, 400.0, 1e-4).unwrap() / base - 0.5).abs() < 1e-12);
        // 20 kohm-cm^2, 100 ohm-cm and a micron give about 0.7 mm.
        assert!((base - 0.0707).abs() < 1e-4, "the length constant came out at {base} cm");
        assert!(length_constant(0.0, 100.0, 1e-4).is_err());
        assert!(length_constant(20_000.0, 0.0, 1e-4).is_err());
        assert!(length_constant(20_000.0, 100.0, 0.0).is_err());
    }

    #[test]
    fn simulated_decisions_are_as_accurate_as_the_gamblers_ruin_formula_says() {
        let mut rng = Rng::new(0x0E0E_2005);
        for (drift, threshold, noise) in [(0.5f64, 1.0f64, 1.0f64), (1.0, 0.8, 1.2), (0.0, 1.0, 1.0)]
        {
            let trials = 4000;
            let runs = reaction_time_ddm(drift, threshold, noise, 0.001, trials, &mut rng).unwrap();
            assert_eq!(runs.len(), trials);
            assert!(runs.iter().all(|r| r.0 > 0.0));
            let observed = runs.iter().filter(|r| r.1).count() as f64 / trials as f64;
            let exact = ddm_analytic_accuracy(drift, threshold, noise).unwrap();
            let error = (exact * (1.0 - exact) / trials as f64).sqrt();
            assert!(
                (observed - exact).abs() < 4.0 * error + 0.01,
                "drift {drift} gave {observed} against {exact}"
            );
        }
    }

    #[test]
    fn the_decision_depends_only_on_the_drift_scaled_by_the_noise_power() {
        // Accuracy is a function of drift * threshold / noise^2 alone, so
        // three parameter sets with the same combination give the same
        // answer while taking very different times.
        let reference = ddm_analytic_accuracy(0.5, 1.0, 1.0).unwrap();
        assert!((ddm_analytic_accuracy(1.0, 1.0, 2.0f64.sqrt()).unwrap() - reference).abs() < 1e-12);
        assert!((ddm_analytic_accuracy(0.25, 2.0, 1.0).unwrap() - reference).abs() < 1e-12);
        // Zero drift is a coin flip; a large one is certainty; and
        // reversing the drift reflects the probability.
        assert!((ddm_analytic_accuracy(0.0, 1.0, 1.0).unwrap() - 0.5).abs() < 1e-15);
        assert!(ddm_analytic_accuracy(50.0, 1.0, 1.0).unwrap() > 1.0 - 1e-12);
        assert!(
            (ddm_analytic_accuracy(0.5, 1.0, 1.0).unwrap()
                + ddm_analytic_accuracy(-0.5, 1.0, 1.0).unwrap()
                - 1.0)
                .abs()
                < 1e-15
        );
        assert!(ddm_analytic_accuracy(1.0, 0.0, 1.0).is_err());
        assert!(ddm_analytic_accuracy(1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn stronger_evidence_is_decided_faster_and_a_higher_bound_more_slowly() {
        // The speed-accuracy trade-off, which is the model's reason for
        // existing: raising the threshold buys accuracy and costs time,
        // with no change to the evidence itself.
        let mut rng = Rng::new(0x0E0E_2006);
        let mean = |drift: f64, threshold: f64, rng: &mut Rng| -> f64 {
            let runs = reaction_time_ddm(drift, threshold, 1.0, 0.001, 1500, rng).unwrap();
            runs.iter().map(|r| r.0).sum::<f64>() / runs.len() as f64
        };
        let weak = mean(0.3, 1.0, &mut rng);
        let strong = mean(1.5, 1.0, &mut rng);
        assert!(strong < weak, "strong evidence took {strong} against weak evidence's {weak}");
        let cautious = mean(0.3, 2.0, &mut rng);
        assert!(cautious > weak, "a higher bound was decided in {cautious} against {weak}");
        let careful = ddm_analytic_accuracy(0.3, 2.0, 1.0).unwrap();
        assert!(careful > ddm_analytic_accuracy(0.3, 1.0, 1.0).unwrap());

        assert!(reaction_time_ddm(1.0, 0.0, 1.0, 0.001, 10, &mut rng).is_err());
        assert!(reaction_time_ddm(1.0, 1.0, 0.0, 0.001, 10, &mut rng).is_err());
        assert!(reaction_time_ddm(1.0, 1.0, 1.0, 0.0, 10, &mut rng).is_err());
        assert!(reaction_time_ddm(1.0, 1.0, 1.0, 0.001, 0, &mut rng).is_err());
    }
}
