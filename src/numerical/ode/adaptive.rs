//! Adaptive Runge-Kutta integration: Dormand-Prince 5(4).
//!
//! Reference: Dormand & Prince, "A family of embedded Runge-Kutta
//! formulae" (1980); Hairer, Nørsett & Wanner, *Solving ODEs I*, §II.4.
//! The embedded 4th-order solution provides the error estimate; steps
//! use the FSAL (first-same-as-last) property.

use crate::error::SolveError;

// Dormand-Prince 5(4) Butcher tableau.
const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
const A: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
];
// 5th-order weights (equal to the last A row; FSAL).
const B5: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];
// Embedded 4th-order weights.
const B4: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

const SAFETY: f64 = 0.9;
const MIN_SCALE: f64 = 0.2;
const MAX_SCALE: f64 = 5.0;
const MAX_STEPS: usize = 10_000_000;

/// Result of an adaptive integration: accepted step times, states, and
/// the number of rejected trial steps.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveResult {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub steps_rejected: usize,
}

struct StepAttempt {
    y_new: Vec<f64>,
    k_last: Vec<f64>,
    err_norm: f64,
}

/// One trial DOPRI5 step from (t, y) with derivative k1 already known.
fn try_step(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    t: f64,
    y: &[f64],
    k1: &[f64],
    h: f64,
    rtol: f64,
    atol: f64,
) -> StepAttempt {
    let n = y.len();
    let mut k: Vec<Vec<f64>> = Vec::with_capacity(7);
    k.push(k1.to_vec());
    for s in 1..7 {
        let mut ys = y.to_vec();
        for (j, kj) in k.iter().enumerate() {
            let a = A[s - 1][j];
            if a != 0.0 {
                for i in 0..n {
                    ys[i] += h * a * kj[i];
                }
            }
        }
        k.push(f(t + C[s] * h, &ys));
    }
    let mut y_new = y.to_vec();
    let mut y4 = y.to_vec();
    for (j, kj) in k.iter().enumerate() {
        for i in 0..n {
            y_new[i] += h * B5[j] * kj[i];
            y4[i] += h * B4[j] * kj[i];
        }
    }
    // RMS of the componentwise scaled error.
    let mut err_sq = 0.0;
    for i in 0..n {
        let scale = atol + rtol * y[i].abs().max(y_new[i].abs());
        let e = (y_new[i] - y4[i]) / scale;
        err_sq += e * e;
    }
    let err_norm = (err_sq / n as f64).sqrt();
    StepAttempt { y_new, k_last: k.pop().unwrap(), err_norm }
}

fn validate(
    t0: f64,
    t1: f64,
    y0: &[f64],
    rtol: f64,
    atol: f64,
    h0: f64,
) -> Result<(), SolveError> {
    if y0.is_empty() {
        return Err(SolveError::InvalidArgument("dormand_prince requires a non-empty state"));
    }
    if !(t1 > t0) {
        return Err(SolveError::InvalidArgument("dormand_prince requires t1 > t0"));
    }
    if !(rtol > 0.0) || !(atol > 0.0) {
        return Err(SolveError::InvalidArgument("dormand_prince requires rtol, atol > 0"));
    }
    if !(h0 > 0.0) {
        return Err(SolveError::InvalidArgument("dormand_prince requires h0 > 0"));
    }
    Ok(())
}

/// Integrates dy/dt = f(t, y) from t0 to t1 with adaptive step size,
/// recording every accepted step.
///
/// The local error per step is kept near `atol + rtol·|y|` (RMS over
/// components). Fails with `NoConvergence` if the step count budget is
/// exhausted or the step size underflows.
pub fn dormand_prince(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    t1: f64,
    y0: &[f64],
    rtol: f64,
    atol: f64,
    h0: f64,
) -> Result<AdaptiveResult, SolveError> {
    validate(t0, t1, y0, rtol, atol, h0)?;
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = h0.min(t1 - t0);
    let mut k1 = f(t, &y);
    let mut out_t = vec![t0];
    let mut out_y = vec![y.clone()];
    let mut rejected = 0usize;

    for _ in 0..MAX_STEPS {
        if t >= t1 {
            return Ok(AdaptiveResult { t: out_t, y: out_y, steps_rejected: rejected });
        }
        h = h.min(t1 - t);
        if h <= f64::EPSILON * t.abs().max(1.0) {
            return Err(SolveError::NoConvergence { iters: out_t.len(), residual: h });
        }
        let attempt = try_step(f, t, &y, &k1, h, rtol, atol);
        if attempt.err_norm <= 1.0 {
            t += h;
            y = attempt.y_new;
            k1 = attempt.k_last; // FSAL
            out_t.push(t);
            out_y.push(y.clone());
            let scale = if attempt.err_norm == 0.0 {
                MAX_SCALE
            } else {
                (SAFETY * attempt.err_norm.powf(-0.2)).clamp(MIN_SCALE, MAX_SCALE)
            };
            h *= scale;
        } else {
            rejected += 1;
            h *= (SAFETY * attempt.err_norm.powf(-0.2)).clamp(MIN_SCALE, 1.0);
        }
    }
    Err(SolveError::NoConvergence { iters: MAX_STEPS, residual: t1 - t })
}

/// Like [`dormand_prince`] but returns the solution interpolated at
/// `sample_times` (cubic Hermite between accepted steps, using the
/// stored derivatives at the step endpoints).
///
/// `sample_times` must be non-decreasing and lie within [t0, t1].
pub fn dormand_prince_dense(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    t1: f64,
    y0: &[f64],
    rtol: f64,
    atol: f64,
    h0: f64,
    sample_times: &[f64],
) -> Result<AdaptiveResult, SolveError> {
    validate(t0, t1, y0, rtol, atol, h0)?;
    for w in sample_times.windows(2) {
        if w[1] < w[0] {
            return Err(SolveError::InvalidArgument("sample_times must be non-decreasing"));
        }
    }
    if sample_times.iter().any(|&s| s < t0 || s > t1) {
        return Err(SolveError::InvalidArgument("sample_times must lie within [t0, t1]"));
    }

    let steps = dormand_prince(f, t0, t1, y0, rtol, atol, h0)?;
    let n = y0.len();
    // Derivatives at each accepted node (recomputed; cheap relative to
    // the integration itself).
    let derivs: Vec<Vec<f64>> = steps.t.iter().zip(&steps.y).map(|(&ti, yi)| f(ti, yi)).collect();

    let mut out_y = Vec::with_capacity(sample_times.len());
    let mut seg = 0usize;
    for &s in sample_times {
        while seg + 1 < steps.t.len() - 1 && steps.t[seg + 1] < s {
            seg += 1;
        }
        let (ta, tb) = (steps.t[seg], steps.t[seg + 1]);
        let hseg = tb - ta;
        let theta = if hseg > 0.0 { (s - ta) / hseg } else { 0.0 };
        // Cubic Hermite basis.
        let h00 = (1.0 + 2.0 * theta) * (1.0 - theta) * (1.0 - theta);
        let h10 = theta * (1.0 - theta) * (1.0 - theta);
        let h01 = theta * theta * (3.0 - 2.0 * theta);
        let h11 = theta * theta * (theta - 1.0);
        let mut yi = vec![0.0; n];
        for i in 0..n {
            yi[i] = h00 * steps.y[seg][i]
                + h10 * hseg * derivs[seg][i]
                + h01 * steps.y[seg + 1][i]
                + h11 * hseg * derivs[seg + 1][i];
        }
        out_y.push(yi);
    }
    Ok(AdaptiveResult {
        t: sample_times.to_vec(),
        y: out_y,
        steps_rejected: steps.steps_rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay_matches_analytic() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let r = dormand_prince(&f, 0.0, 5.0, &[1.0], 1e-10, 1e-12, 0.1).unwrap();
        let (tf, yf) = (r.t.last().unwrap(), r.y.last().unwrap());
        assert!((tf - 5.0).abs() < 1e-12);
        assert!((yf[0] - (-5.0_f64).exp()).abs() < 1e-9, "got {}", yf[0]);
    }

    #[test]
    fn test_exponential_growth_matches_rtol() {
        let f = |_t: f64, y: &[f64]| vec![y[0]];
        let rtol = 1e-9;
        let r = dormand_prince(&f, 0.0, 3.0, &[1.0], rtol, 1e-14, 0.05).unwrap();
        let yf = r.y.last().unwrap()[0];
        let exact = 3.0_f64.exp();
        assert!(
            ((yf - exact) / exact).abs() < 100.0 * rtol,
            "relative error {}",
            ((yf - exact) / exact).abs()
        );
    }

    #[test]
    fn test_harmonic_oscillator_accuracy() {
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let r = dormand_prince(&f, 0.0, 10.0, &[1.0, 0.0], 1e-10, 1e-12, 0.1).unwrap();
        let yf = r.y.last().unwrap();
        assert!((yf[0] - 10.0_f64.cos()).abs() < 1e-7);
        assert!((yf[1] + 10.0_f64.sin()).abs() < 1e-7);
    }

    #[test]
    fn test_step_rejection_happens_on_kick() {
        // A sharp transition forces rejections at loose initial steps.
        let f = |t: f64, y: &[f64]| vec![if t < 1.0 { 0.0 } else { -50.0 * y[0] }];
        let r = dormand_prince(&f, 0.0, 2.0, &[1.0], 1e-8, 1e-10, 0.5).unwrap();
        assert!(r.steps_rejected > 0);
    }

    #[test]
    fn test_dense_output_matches_analytic() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let samples: Vec<f64> = (0..=50).map(|i| i as f64 * 0.1).collect();
        let r = dormand_prince_dense(&f, 0.0, 5.0, &[1.0], 1e-10, 1e-12, 0.1, &samples).unwrap();
        assert_eq!(r.t.len(), samples.len());
        for (ti, yi) in r.t.iter().zip(&r.y) {
            assert!((yi[0] - (-ti).exp()).abs() < 1e-6, "t={ti}");
        }
    }

    #[test]
    fn test_invalid_arguments() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        assert!(dormand_prince(&f, 0.0, -1.0, &[1.0], 1e-8, 1e-8, 0.1).is_err());
        assert!(dormand_prince(&f, 0.0, 1.0, &[], 1e-8, 1e-8, 0.1).is_err());
        assert!(dormand_prince(&f, 0.0, 1.0, &[1.0], -1.0, 1e-8, 0.1).is_err());
        assert!(dormand_prince(&f, 0.0, 1.0, &[1.0], 1e-8, 1e-8, 0.0).is_err());
        assert!(dormand_prince_dense(&f, 0.0, 1.0, &[1.0], 1e-8, 1e-8, 0.1, &[2.0]).is_err());
        assert!(
            dormand_prince_dense(&f, 0.0, 1.0, &[1.0], 1e-8, 1e-8, 0.1, &[0.5, 0.2]).is_err()
        );
    }
}
