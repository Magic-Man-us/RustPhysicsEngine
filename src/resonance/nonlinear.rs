//! Nonlinear resonance: the Duffing and van der Pol oscillators,
//! parametric (Mathieu) stability, Fano interference, synchronization
//! pulling/locking, and generic harmonic-balance machinery.
//!
//! The Duffing convention throughout is
//! x″ + δ·x′ + α·x + β·x³ = γ·cos(ωt).

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::numerical::{polynomial_roots, rk4_step_vec};
use crate::optimization::levenberg_marquardt;

const TWO_PI: f64 = 2.0 * PI;

/// Steady single-harmonic response amplitudes of the Duffing oscillator
/// at drive frequency ω (harmonic balance): up to three coexisting
/// branches, ascending.
#[must_use]
pub fn duffing_response_amplitude(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    omega: f64,
) -> Vec<f64> {
    // [(α − ω² + ¾βz)² + (δω)²]·z = γ², z = a².
    let d = alpha - omega * omega;
    let c3 = 9.0 / 16.0 * beta * beta;
    let c2 = 1.5 * beta * d;
    let c1 = d * d + delta * delta * omega * omega;
    let c0 = -gamma * gamma;
    let roots = if beta == 0.0 {
        vec![Complex::new(-c0 / c1, 0.0)]
    } else {
        polynomial_roots(&[c3, c2, c1, c0]).unwrap_or_default()
    };
    let mut amps: Vec<f64> = roots
        .iter()
        .filter(|r| r.im.abs() < 1e-8 * (1.0 + r.re.abs()) && r.re > 0.0)
        .map(|r| r.re.sqrt())
        .collect();
    amps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    amps.dedup_by(|a, b| (*a - *b).abs() < 1e-9 * (1.0 + b.abs()));
    amps
}

/// Backbone curve: free-vibration frequency at amplitude a,
/// ω = √(α + ¾βa²).
#[must_use]
pub fn duffing_backbone(alpha: f64, beta: f64, amplitude: f64) -> f64 {
    (alpha + 0.75 * beta * amplitude * amplitude).sqrt()
}

/// Jump (saddle-node) frequencies of the forced Duffing sweep: the ω
/// interval where three branches coexist, found by scanning the
/// harmonic-balance solution count. None when no bistability exists.
#[must_use]
pub fn duffing_jump_frequencies(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
) -> Option<(f64, f64)> {
    let w0 = alpha.max(1e-9).sqrt();
    let steps = 6000;
    let w_max = 4.0 * w0 + 4.0 * gamma.abs();
    let mut lo = None;
    let mut hi = None;
    for i in 1..steps {
        let w = w_max * i as f64 / steps as f64;
        let n = duffing_response_amplitude(alpha, beta, delta, gamma, w).len();
        if n >= 3 {
            if lo.is_none() {
                lo = Some(w);
            }
            hi = Some(w);
        }
    }
    match (lo, hi) {
        (Some(a), Some(b)) if b > a => Some((a, b)),
        _ => None,
    }
}

fn duffing_rhs(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    omega: f64,
) -> impl Fn(f64, &[f64]) -> Vec<f64> {
    move |t: f64, y: &[f64]| {
        vec![
            y[1],
            gamma * (omega * t).cos() - delta * y[1] - alpha * y[0] - beta * y[0].powi(3),
        ]
    }
}

/// RK4 trajectory (t, x, v) of the forced Duffing oscillator.
#[must_use]
#[allow(clippy::too_many_arguments)] // signature fixed by the roadmap
pub fn duffing_simulate(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    omega: f64,
    x0: f64,
    v0: f64,
    t_end: f64,
    dt: f64,
) -> Vec<(f64, f64, f64)> {
    let f = duffing_rhs(alpha, beta, delta, gamma, omega);
    let mut y = vec![x0, v0];
    let mut t = 0.0;
    let mut out = vec![(0.0, x0, v0)];
    while t < t_end - 1e-12 {
        y = rk4_step_vec(&f, t, &y, dt);
        t += dt;
        out.push((t, y[0], y[1]));
    }
    out
}

/// Poincaré section of the Duffing oscillator: (x, v) sampled once per
/// forcing period, after discarding 100 transient periods.
#[must_use]
#[allow(clippy::too_many_arguments)] // signature fixed by the roadmap
pub fn duffing_poincare(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    omega: f64,
    x0: f64,
    v0: f64,
    n_points: usize,
) -> Vec<(f64, f64)> {
    let f = duffing_rhs(alpha, beta, delta, gamma, omega);
    let period = TWO_PI / omega;
    let steps_per_period = 256usize;
    let dt = period / steps_per_period as f64;
    let mut y = vec![x0, v0];
    let mut t = 0.0;
    let transient = 100;
    let mut out = Vec::with_capacity(n_points);
    for p in 0..transient + n_points {
        for _ in 0..steps_per_period {
            y = rk4_step_vec(&f, t, &y, dt);
            t += dt;
        }
        if p >= transient {
            out.push((y[0], y[1]));
        }
    }
    out
}

/// Floquet monodromy matrix of x″ + (a − 2q·cos 2t)·x = 0 over one
/// coefficient period π.
fn mathieu_monodromy(a: f64, q: f64) -> [[f64; 2]; 2] {
    let f = |t: f64, y: &[f64]| -> Vec<f64> {
        vec![y[1], -(a - 2.0 * q * (2.0 * t).cos()) * y[0]]
    };
    let steps = 2000;
    let dt = PI / steps as f64;
    let mut cols = [[1.0, 0.0], [0.0, 1.0]];
    for col in cols.iter_mut() {
        let mut y = vec![col[0], col[1]];
        let mut t = 0.0;
        for _ in 0..steps {
            y = rk4_step_vec(&f, t, &y, dt);
            t += dt;
        }
        *col = [y[0], y[1]];
    }
    // Columns of the monodromy matrix.
    [[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]]
}

/// Mathieu equation stability of x″ + (a − 2q·cos 2t)x = 0 by the
/// Floquet criterion |tr M| ≤ 2.
#[must_use]
pub fn mathieu_stability(a: f64, q: f64) -> bool {
    let m = mathieu_monodromy(a, q);
    (m[0][0] + m[1][1]).abs() <= 2.0
}

/// Stability chart over a (rows) × q (columns) grids of n points each.
#[must_use]
pub fn mathieu_stability_chart(
    a_range: (f64, f64),
    q_range: (f64, f64),
    n: usize,
) -> Vec<Vec<bool>> {
    (0..n)
        .map(|i| {
            let a = a_range.0 + (a_range.1 - a_range.0) * i as f64 / (n - 1).max(1) as f64;
            (0..n)
                .map(|j| {
                    let q =
                        q_range.0 + (q_range.1 - q_range.0) * j as f64 / (n - 1).max(1) as f64;
                    mathieu_stability(a, q)
                })
                .collect()
        })
        .collect()
}

/// Pump-amplitude threshold h_c of the n-th parametric instability
/// tongue for x″ + 2λx′ + ω₀²(1 + h·cos(Ωt))x = 0 with Ω = 2ω₀/n,
/// found by bisecting the damped Floquet spectral radius.
///
/// # Panics
/// Panics unless n ≥ 1 and the parameters are positive.
#[must_use]
pub fn parametric_resonance_threshold(omega0: f64, damping: f64, n: usize) -> f64 {
    assert!(n >= 1 && omega0 > 0.0 && damping > 0.0, "invalid parameters");
    let pump = 2.0 * omega0 / n as f64;
    let period = TWO_PI / pump;
    let unstable = |h: f64| -> bool {
        let f = |t: f64, y: &[f64]| -> Vec<f64> {
            vec![
                y[1],
                -2.0 * damping * y[1] - omega0 * omega0 * (1.0 + h * (pump * t).cos()) * y[0],
            ]
        };
        let steps = 2000;
        let dt = period / steps as f64;
        let mut cols = [[1.0, 0.0], [0.0, 1.0]];
        for col in cols.iter_mut() {
            let mut y = vec![col[0], col[1]];
            let mut t = 0.0;
            for _ in 0..steps {
                y = rk4_step_vec(&f, t, &y, dt);
                t += dt;
            }
            *col = [y[0], y[1]];
        }
        let (tr, det) = (
            cols[0][0] + cols[1][1],
            cols[0][0] * cols[1][1] - cols[1][0] * cols[0][1],
        );
        // Spectral radius of the 2x2 monodromy matrix.
        let disc = tr * tr / 4.0 - det;
        let rho = if disc >= 0.0 {
            (tr.abs() / 2.0 + disc.sqrt()).abs()
        } else {
            det.sqrt()
        };
        rho > 1.0
    };
    let mut lo = 0.0;
    let mut hi = 2.0;
    if !unstable(hi) {
        return f64::INFINITY;
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if unstable(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Kapitza inverted pendulum stability: a²ω² > 2·g·l.
#[must_use]
pub fn kapitza_pendulum_stable(l: f64, g: f64, a: f64, omega: f64) -> bool {
    a * a * omega * omega > 2.0 * g * l
}

/// RK4 trajectory (t, x, v) of the van der Pol oscillator
/// x″ − μ(1 − x²)x′ + ω²x = 0.
#[must_use]
pub fn van_der_pol_simulate(
    mu: f64,
    omega: f64,
    x0: f64,
    v0: f64,
    t_end: f64,
    dt: f64,
) -> Vec<(f64, f64, f64)> {
    let f = move |_t: f64, y: &[f64]| -> Vec<f64> {
        vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - omega * omega * y[0]]
    };
    let mut y = vec![x0, v0];
    let mut t = 0.0;
    let mut out = vec![(0.0, x0, v0)];
    while t < t_end - 1e-12 {
        y = rk4_step_vec(&f, t, &y, dt);
        t += dt;
        out.push((t, y[0], y[1]));
    }
    out
}

/// Limit-cycle amplitude of the van der Pol oscillator (numerically
/// settled; → 2 as μ → 0).
#[must_use]
pub fn van_der_pol_limit_cycle_amplitude(mu: f64) -> f64 {
    let t_settle = 60.0 / mu.clamp(0.05, 1.0);
    let traj = van_der_pol_simulate(mu, 1.0, 0.5, 0.0, t_settle + 20.0, 0.002);
    let tail_start = ((t_settle / 0.002) as usize).min(traj.len() - 1);
    traj[tail_start..].iter().map(|&(_, x, _)| x.abs()).fold(0.0_f64, f64::max)
}

/// Adler entrainment (lock-in) band of a weakly forced van der Pol
/// oscillator with unit natural frequency: ω ∈ 1 ± F/4 for weak
/// forcing F on the a = 2 limit cycle.
#[must_use]
pub fn van_der_pol_entrainment_range(mu: f64, forcing_amp: f64) -> (f64, f64) {
    let _ = mu; // weak-coupling averaging is μ-independent at leading order
    let half = forcing_amp / 4.0;
    (1.0 - half, 1.0 + half)
}

/// Fano lineshape (q + ε)²/(1 + ε²), ε = 2(ω − ω₀)/γ (background → 1).
#[must_use]
pub fn fano_lineshape(omega: f64, omega0: f64, gamma: f64, q: f64) -> f64 {
    let eps = 2.0 * (omega - omega0) / gamma;
    (q + eps).powi(2) / (1.0 + eps * eps)
}

/// Fit A·(q+ε)²/(1+ε²) to data; returns (ω₀, γ, q, A).
///
/// # Panics
/// Panics if the fit fails or fewer than 5 points are supplied.
#[must_use]
pub fn fano_fit(omega: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    assert!(omega.len() >= 5 && omega.len() == y.len(), "need >= 5 points");
    // Initial guesses: dip/peak location and span.
    let (imax, &ymax) = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let span = omega[omega.len() - 1] - omega[0];
    let om = omega.to_vec();
    let yv = y.to_vec();
    let resid = move |p: &[f64]| -> Vec<f64> {
        let (w0, g, q, a) = (p[0], p[1], p[2], p[3]);
        om.iter()
            .zip(&yv)
            .map(|(&w, &yy)| {
                let eps = 2.0 * (w - w0) / g;
                a * (q + eps).powi(2) / (1.0 + eps * eps) - yy
            })
            .collect()
    };
    let p0 = [omega[imax], span / 10.0, 1.0, ymax / 2.0];
    let fit = levenberg_marquardt(&resid, None, &p0, 1e-12, 800).expect("Fano fit failed");
    (fit.params[0], fit.params[1].abs(), fit.params[2], fit.params[3])
}

/// Autoresonance capture threshold for a swept-drive Duffing-type
/// oscillator: ε_c = 0.41·(dω/dt)^(3/4)/√|α_nl| (Fajans-Friedland
/// scaling law).
#[must_use]
pub fn autoresonance_threshold(alpha: f64, sweep_rate: f64) -> f64 {
    0.41 * sweep_rate.abs().powf(0.75) / alpha.abs().sqrt()
}

/// Resonator frequency pulling by a detuned load:
/// f = f₀·(1 + Δ/(2Q)).
#[must_use]
pub fn frequency_pulling(f0: f64, q: f64, coupling_detuning: f64) -> f64 {
    f0 * (1.0 + coupling_detuning / (2.0 * q))
}

/// Adler injection-locking half-range Δf = f₀·ρ/(2Q) for injection
/// amplitude ratio ρ.
#[must_use]
pub fn injection_locking_range(f0: f64, q: f64, injection_ratio: f64) -> f64 {
    f0 * injection_ratio / (2.0 * q)
}

/// Weak-signal stochastic-resonance SNR of a bistable well
/// (McNamara-Wiesenfeld form): √2·(a·ΔU/D²)²·... reduced to the
/// standard shape SNR ∝ (a²ΔU²/D²)·e^(−ΔU/D), which is maximized at
/// D = ΔU/2. `omega` enters only beyond the adiabatic limit and is
/// ignored here.
#[must_use]
pub fn stochastic_resonance_snr(a: f64, d: f64, noise: f64, omega: f64) -> f64 {
    let _ = omega;
    let du = d;
    std::f64::consts::SQRT_2 * (a * du / (noise * noise)).powi(2) * (-du / noise).exp()
}

/// Enclosed area of a swept-response hysteresis loop (trapezoid of
/// up-sweep minus down-sweep).
///
/// # Panics
/// Panics on mismatched lengths.
#[must_use]
pub fn hysteresis_loop(f_sweep: &[f64], response_up: &[f64], response_down: &[f64]) -> f64 {
    assert!(
        f_sweep.len() == response_up.len() && f_sweep.len() == response_down.len(),
        "sweep arrays must match"
    );
    let mut area = 0.0;
    for i in 1..f_sweep.len() {
        let df = f_sweep[i] - f_sweep[i - 1];
        let d1 = response_up[i - 1] - response_down[i - 1];
        let d2 = response_up[i] - response_down[i];
        area += 0.5 * (d1 + d2) * df;
    }
    area.abs()
}

/// Generic harmonic balance for x″ + f(x, x′, t) = 0 with f
/// 2π/ω-periodic in t: Newton iteration on the truncated Fourier series
/// x(t) = c₀ + Σ_k \[a_k cos kωt + b_k sin kωt\], collocated at
/// 4·n_harmonics + 2 points. Returns coefficients c_k = a_k − j·b_k
/// (c₀ real) for k = 0..=n_harmonics.
///
/// # Panics
/// Panics if the Newton solve fails to converge.
#[must_use]
pub fn harmonic_balance(
    f: &dyn Fn(f64, f64, f64) -> f64,
    omega: f64,
    n_harmonics: usize,
    amplitude_guess: f64,
) -> Vec<Complex> {
    let nh = n_harmonics.max(1);
    let n_par = 2 * nh + 1; // c0, a_k, b_k
    let n_col = 2 * n_par; // oversampled collocation
    let eval = |p: &[f64], t: f64| -> (f64, f64, f64) {
        // x, x', x''
        let mut x = p[0];
        let mut xd = 0.0;
        let mut xdd = 0.0;
        for k in 1..=nh {
            let (a, b) = (p[2 * k - 1], p[2 * k]);
            let kw = k as f64 * omega;
            let (s, c) = (kw * t).sin_cos();
            x += a * c + b * s;
            xd += -a * kw * s + b * kw * c;
            xdd += -a * kw * kw * c - b * kw * kw * s;
        }
        (x, xd, xdd)
    };
    let resid = |p: &[f64]| -> Vec<f64> {
        (0..n_col)
            .map(|i| {
                let t = TWO_PI / omega * i as f64 / n_col as f64;
                let (x, xd, xdd) = eval(p, t);
                xdd + f(x, xd, t)
            })
            .collect()
    };
    let mut p0 = vec![0.0; n_par];
    p0[1] = amplitude_guess;
    let fit = levenberg_marquardt(&resid, None, &p0, 1e-12, 800)
        .expect("harmonic balance failed to converge");
    let p = fit.params;
    let mut out = vec![Complex::new(p[0], 0.0)];
    for k in 1..=nh {
        out.push(Complex::new(p[2 * k - 1], -p[2 * k]));
    }
    out
}

/// Sinusoidal-input describing function of a static nonlinearity:
/// N(A) = (b₁ + j·a₁)/A from the first Fourier component of
/// f(A·sin θ).
#[must_use]
pub fn describing_function(nonlinearity: &dyn Fn(f64) -> f64, amplitude: f64) -> Complex {
    let n = 4096;
    let mut b1 = 0.0;
    let mut a1 = 0.0;
    for i in 0..n {
        let theta = TWO_PI * i as f64 / n as f64;
        let v = nonlinearity(amplitude * theta.sin());
        b1 += v * theta.sin();
        a1 += v * theta.cos();
    }
    let scale = 2.0 / n as f64;
    Complex::new(b1 * scale / amplitude, a1 * scale / amplitude)
}

/// Steady-state spectral amplitudes of the driven Duffing response at
/// \[ω/3, ω/2, ω, 2ω, 3ω\] — nonzero sub/superharmonic content flags
/// period-multiplied responses.
#[must_use]
pub fn subharmonic_response(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    omega: f64,
) -> Vec<f64> {
    // 6-periods-long steady window sampled after a transient, so ω/3 and
    // ω/2 land on bins.
    let period = TWO_PI / omega;
    let steps_per_period = 128usize;
    let dt = period / steps_per_period as f64;
    let f = duffing_rhs(alpha, beta, delta, gamma, omega);
    let mut y = vec![0.1, 0.0];
    let mut t = 0.0;
    for _ in 0..200 * steps_per_period {
        y = rk4_step_vec(&f, t, &y, dt);
        t += dt;
    }
    let n = 6 * steps_per_period;
    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        y = rk4_step_vec(&f, t, &y, dt);
        t += dt;
        xs.push(y[0]);
    }
    let spec = crate::transforms::fft::fft_any(
        &xs.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>(),
    );
    // Bin of ω is 6 (6 periods in the window).
    [2usize, 3, 6, 12, 18]
        .iter()
        .map(|&k| 2.0 * spec[k].norm() / n as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_duffing_linear_limit_matches_oscillator() {
        // β = 0 reduces to the damped linear oscillator.
        let (alpha, delta, gamma) = (4.0, 0.3, 1.0);
        let osc = super::super::oscillator::DampedOscillator { m: 1.0, c: delta, k: alpha };
        for &w in &[0.5, 1.9, 2.0, 3.0] {
            let amps = duffing_response_amplitude(alpha, 0.0, delta, gamma, w);
            assert_eq!(amps.len(), 1);
            assert!(
                approx(amps[0], osc.steady_state_amplitude(gamma, w), 1e-9),
                "w={w}"
            );
        }
    }

    #[test]
    fn test_duffing_bistability_and_jumps() {
        // Classic hardening spring with weak damping.
        let (alpha, beta, delta, gamma) = (1.0, 0.4, 0.05, 0.2);
        let jumps = duffing_jump_frequencies(alpha, beta, delta, gamma);
        let (w_lo, w_hi) = jumps.expect("expected a bistable band");
        assert!(w_lo > 1.0 && w_hi > w_lo, "band ({w_lo}, {w_hi})");
        // Inside the band: three branches; outside: one.
        let mid = 0.5 * (w_lo + w_hi);
        assert_eq!(duffing_response_amplitude(alpha, beta, delta, gamma, mid).len(), 3);
        assert_eq!(duffing_response_amplitude(alpha, beta, delta, gamma, 0.5).len(), 1);
        // Backbone passes through the band.
        let a_mid = duffing_response_amplitude(alpha, beta, delta, gamma, mid)[2];
        let wb = duffing_backbone(alpha, beta, a_mid);
        assert!(wb > w_lo * 0.9 && wb < w_hi * 1.1, "backbone {wb}");
    }

    #[test]
    fn test_duffing_simulation_matches_harmonic_balance() {
        let (alpha, beta, delta, gamma, omega) = (1.0, 0.05, 0.2, 0.3, 1.2);
        let amps = duffing_response_amplitude(alpha, beta, delta, gamma, omega);
        let traj = duffing_simulate(alpha, beta, delta, gamma, omega, 0.0, 0.0, 400.0, 0.005);
        let tail: Vec<f64> =
            traj[traj.len() - 3000..].iter().map(|&(_, x, _)| x.abs()).collect();
        let peak = tail.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            (peak - amps[0]).abs() / amps[0] < 0.05,
            "sim {peak} vs HB {:?}",
            amps
        );
    }

    #[test]
    fn test_duffing_poincare_periodic_vs_chaotic() {
        // Weakly forced: period-1 orbit → all section points coincide.
        let pts = duffing_poincare(1.0, 0.05, 0.2, 0.3, 1.2, 0.1, 0.0, 20);
        let spread = pts
            .iter()
            .map(|&(x, v)| ((x - pts[0].0).abs()).max((v - pts[0].1).abs()))
            .fold(0.0_f64, f64::max);
        assert!(spread < 1e-4, "period-1 spread {spread}");
        // Classic chaotic parameters (Ueda-like): points scatter.
        let pts_c = duffing_poincare(-1.0, 1.0, 0.3, 0.5, 1.2, 0.1, 0.0, 40);
        let spread_c = pts_c
            .iter()
            .map(|&(x, v)| ((x - pts_c[0].0).abs()).max((v - pts_c[0].1).abs()))
            .fold(0.0_f64, f64::max);
        assert!(spread_c > 0.1, "chaotic spread {spread_c}");
    }

    #[test]
    fn test_mathieu_first_tongue() {
        // The first instability tongue emanates from a = 1 (equation in
        // the a − 2q cos 2t form): at q = 0.2, a = 1 is unstable while
        // a = 0.5 and a = 2.0 are stable.
        assert!(!mathieu_stability(1.0, 0.2));
        assert!(mathieu_stability(0.5, 0.2));
        assert!(mathieu_stability(2.0, 0.2));
        // q = 0: stable for a > 0.
        assert!(mathieu_stability(1.3, 0.0));
        let chart = mathieu_stability_chart((0.5, 1.5), (0.0, 0.5), 11);
        // Middle row (a = 1) flips to unstable as q grows.
        assert!(chart[5][0]);
        assert!(!chart[5][10]);
    }

    #[test]
    fn test_parametric_threshold_matches_landau() {
        // First tongue: h_c = 4λ/ω0.
        let (w0, lam) = (2.0, 0.02);
        let hc = parametric_resonance_threshold(w0, lam, 1);
        assert!(approx(hc, 4.0 * lam / w0, 0.003), "h_c {hc}");
        // Higher tongues need much more pumping.
        let hc2 = parametric_resonance_threshold(w0, lam, 2);
        assert!(hc2 > 3.0 * hc, "second tongue {hc2}");
        assert!(kapitza_pendulum_stable(0.2, 9.81, 0.02, 400.0));
        assert!(!kapitza_pendulum_stable(0.2, 9.81, 0.02, 50.0));
    }

    #[test]
    fn test_van_der_pol_amplitude_and_entrainment() {
        for &mu in &[0.05, 0.2] {
            let a = van_der_pol_limit_cycle_amplitude(mu);
            assert!((a - 2.0).abs() < 0.1, "mu={mu}: a={a}");
        }
        let (lo, hi) = van_der_pol_entrainment_range(0.1, 0.2);
        assert!(approx(hi - lo, 0.1, 1e-12));
    }

    #[test]
    fn test_fano_shape_and_fit() {
        // q = 0: symmetric dip to zero at resonance; large q → Lorentzian.
        assert!(fano_lineshape(5.0, 5.0, 0.4, 0.0) < 1e-12);
        assert!(fano_lineshape(50.0, 5.0, 0.4, 0.0) > 0.99);
        // Fit recovery.
        let (w0, g, q, a) = (10.0, 0.8, 1.5, 2.0);
        let omega: Vec<f64> = (0..600).map(|i| 6.0 + 8.0 * i as f64 / 600.0).collect();
        let y: Vec<f64> = omega.iter().map(|&w| a * fano_lineshape(w, w0, g, q)).collect();
        let (w0f, gf, qf, af) = fano_fit(&omega, &y);
        assert!(approx(w0f, w0, 1e-4) && approx(gf, g, 1e-4));
        assert!(approx(qf, q, 1e-4) && approx(af, a, 1e-4));
    }

    #[test]
    fn test_pulling_locking_snr_hysteresis() {
        assert!(approx(frequency_pulling(1e6, 100.0, 0.01), 1e6 * (1.0 + 0.00005), 1e-6));
        assert!(approx(injection_locking_range(1e6, 50.0, 0.01), 100.0, 1e-9));
        // SNR maximized near D = ΔU/2.
        let snr = |d: f64| stochastic_resonance_snr(0.1, 1.0, d, 0.0);
        assert!(snr(0.5) > snr(0.1) && snr(0.5) > snr(2.0));
        // Hysteresis area of a simple rectangle loop.
        let f: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let up = vec![1.0; 11];
        let down = vec![0.0; 11];
        assert!(approx(hysteresis_loop(&f, &up, &down), 10.0, 1e-12));
        assert!(autoresonance_threshold(1.0, 1e-4) > 0.0);
    }

    #[test]
    fn test_harmonic_balance_generic() {
        // Duffing via the generic engine matches the closed-form HB.
        let (alpha, beta, delta, gamma, omega) = (1.0, 0.05, 0.2, 0.3, 1.2);
        let coeffs = harmonic_balance(
            &|x, v, t| delta * v + alpha * x + beta * x * x * x - gamma * (omega * t).cos(),
            omega,
            5,
            0.5,
        );
        let a1 = coeffs[1].norm();
        let hb = duffing_response_amplitude(alpha, beta, delta, gamma, omega)[0];
        assert!((a1 - hb).abs() / hb < 0.02, "generic {a1} vs single-term {hb}");
    }

    #[test]
    fn test_describing_function_relay_and_cubic() {
        // Ideal relay ±M: N = 4M/(πA).
        let m = 2.5;
        for &a in &[0.5, 1.0, 3.0] {
            let n = describing_function(&move |x: f64| m * x.signum(), a);
            assert!(approx(n.re, 4.0 * m / (PI * a), 1e-3), "A={a}");
            assert!(n.im.abs() < 1e-9);
        }
        // Cubic: N = ¾A².
        let n = describing_function(&|x: f64| x * x * x, 2.0);
        assert!(approx(n.re, 3.0, 1e-6));
    }

    #[test]
    fn test_subharmonic_response_fundamental_dominates() {
        let amps = subharmonic_response(1.0, 0.05, 0.2, 0.3, 1.2);
        // [ω/3, ω/2, ω, 2ω, 3ω]: the drive line dominates; the cubic
        // feeds the 3ω line more than the even harmonics.
        assert!(amps[2] > 10.0 * amps[0] && amps[2] > 10.0 * amps[1]);
        assert!(amps[4] > amps[3]);
    }
}
