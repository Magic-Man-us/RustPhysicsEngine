//! Nonlinear least squares: Levenberg-Marquardt.
//!
//! Reference: Marquardt (1963); Nocedal & Wright, *Numerical
//! Optimization*, §10.3. Minimizes ½‖r(p)‖² by solving
//! (JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr and adapting λ on accept/reject.

use crate::error::SolveError;
use crate::linalg::{lu_decompose, Matrix};

const LAMBDA_INIT: f64 = 1e-3;
const LAMBDA_UP: f64 = 10.0;
const LAMBDA_DOWN: f64 = 0.1;
const LAMBDA_MAX: f64 = 1e12;

/// Result of a Levenberg-Marquardt fit.
///
/// `residual` is the final sum of squared residuals ‖r(p)‖²;
/// `covariance` is s²·(JᵀJ)⁻¹ with s² = SSR/(m−n) when m > n and JᵀJ is
/// invertible, `None` otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct LmResult {
    pub params: Vec<f64>,
    pub residual: f64,
    pub iters: usize,
    pub covariance: Option<Matrix>,
}

/// Forward-difference Jacobian of the residual vector.
fn fd_jacobian(residuals: &dyn Fn(&[f64]) -> Vec<f64>, p: &[f64], r0: &[f64]) -> Matrix {
    let n = p.len();
    let m = r0.len();
    let mut j = Matrix::zeros(m, n);
    let mut pp = p.to_vec();
    for c in 0..n {
        let h = f64::EPSILON.sqrt() * p[c].abs().max(1.0);
        pp[c] = p[c] + h;
        let rp = residuals(&pp);
        pp[c] = p[c];
        for row in 0..m {
            j.set(row, c, (rp[row] - r0[row]) / h);
        }
    }
    j
}

fn ssr(r: &[f64]) -> f64 {
    r.iter().map(|x| x * x).sum()
}

/// Levenberg-Marquardt minimization of ‖residuals(p)‖².
///
/// `jacobian`, when given, returns the m×n Jacobian of the residual
/// vector; otherwise a forward-difference approximation is used.
/// Converges when the gradient ‖Jᵀr‖∞, the step, or the cost decrease
/// falls below `tol`; fails with `NoConvergence` after `max_iter`
/// accepted iterations without meeting the criterion.
pub fn levenberg_marquardt(
    residuals: &dyn Fn(&[f64]) -> Vec<f64>,
    jacobian: Option<&dyn Fn(&[f64]) -> Matrix>,
    p0: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<LmResult, SolveError> {
    if p0.is_empty() {
        return Err(SolveError::InvalidArgument("levenberg_marquardt requires parameters"));
    }
    if !(tol > 0.0) {
        return Err(SolveError::InvalidArgument("levenberg_marquardt requires tol > 0"));
    }
    let n = p0.len();
    let mut p = p0.to_vec();
    let mut r = residuals(&p);
    let m = r.len();
    if m < n {
        return Err(SolveError::InvalidArgument(
            "levenberg_marquardt requires at least as many residuals as parameters",
        ));
    }
    let mut cost = ssr(&r);
    let mut lambda = LAMBDA_INIT;

    let make_jacobian = |p: &[f64], r: &[f64]| -> Result<Matrix, SolveError> {
        let j = match jacobian {
            Some(jf) => jf(p),
            None => fd_jacobian(residuals, p, r),
        };
        if j.rows != m || j.cols != n {
            return Err(SolveError::DimensionMismatch { expected: m * n, got: j.rows * j.cols });
        }
        Ok(j)
    };

    let mut iters = 0usize;
    let mut j = make_jacobian(&p, &r)?;
    while iters < max_iter {
        iters += 1;
        let jt = j.transpose();
        let jtj = jt.mul(&j)?;
        let jtr = jt.mul_vec(&r)?;

        let grad_inf = jtr.iter().fold(0.0_f64, |a, &g| a.max(g.abs()));
        if grad_inf < tol {
            break;
        }

        // Try steps, inflating lambda until one is accepted.
        let mut accepted = false;
        let mut converged = false;
        while lambda <= LAMBDA_MAX {
            let mut a = jtj.clone();
            for i in 0..n {
                let d = jtj.get(i, i);
                let damped = d + lambda * d.max(1e-12);
                a.set(i, i, damped);
            }
            let delta = match lu_decompose(&a).and_then(|f| f.solve(&jtr)) {
                Ok(d) => d,
                Err(_) => {
                    lambda *= LAMBDA_UP;
                    continue;
                }
            };
            let p_new: Vec<f64> = p.iter().zip(&delta).map(|(pi, di)| pi - di).collect();
            let r_new = residuals(&p_new);
            let cost_new = ssr(&r_new);
            if cost_new < cost {
                let step_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
                let cost_drop = cost - cost_new;
                p = p_new;
                r = r_new;
                cost = cost_new;
                lambda = (lambda * LAMBDA_DOWN).max(1e-12);
                accepted = true;
                converged = step_norm < tol || cost_drop < tol * tol;
                break;
            }
            lambda *= LAMBDA_UP;
        }
        if !accepted || converged {
            break; // lambda exhausted (local minimum) or step converged
        }
        j = make_jacobian(&p, &r)?;
    }

    if iters >= max_iter {
        let jt = j.transpose();
        let jtr = jt.mul_vec(&r)?;
        let grad_inf = jtr.iter().fold(0.0_f64, |a, &g| a.max(g.abs()));
        if grad_inf >= tol {
            return Err(SolveError::NoConvergence { iters, residual: cost });
        }
    }

    // Covariance from the final Jacobian.
    let covariance = if m > n {
        let jt = make_jacobian(&p, &r)?.transpose();
        let jtj = jt.mul(&jt.transpose())?;
        match lu_decompose(&jtj).and_then(|f| f.inverse()) {
            Ok(inv) => {
                let s2 = cost / (m - n) as f64;
                Some(inv.scale(s2))
            }
            Err(_) => None,
        }
    } else {
        None
    };

    Ok(LmResult { params: p, residual: cost, iters, covariance })
}

/// Fits y ≈ A·e^(−k·t), returning (A, k). Initial guess from a
/// log-linear regression over the positive samples, refined by LM.
///
/// Fails with `InvalidArgument` unless there are ≥ 2 samples with
/// matching lengths and at least two positive y values.
pub fn fit_exponential_decay(t: &[f64], y: &[f64]) -> Result<(f64, f64), SolveError> {
    if t.len() != y.len() {
        return Err(SolveError::DimensionMismatch { expected: t.len(), got: y.len() });
    }
    if t.len() < 2 {
        return Err(SolveError::InvalidArgument("fit_exponential_decay requires >= 2 samples"));
    }
    // Log-linear initial guess: ln y = ln A - k t.
    let pos: Vec<(f64, f64)> = t
        .iter()
        .zip(y.iter())
        .filter(|(_, &yi)| yi > 0.0)
        .map(|(&ti, &yi)| (ti, yi.ln()))
        .collect();
    if pos.len() < 2 {
        return Err(SolveError::InvalidArgument(
            "fit_exponential_decay requires at least two positive y values",
        ));
    }
    let nn = pos.len() as f64;
    let sx: f64 = pos.iter().map(|p| p.0).sum();
    let sy: f64 = pos.iter().map(|p| p.1).sum();
    let sxx: f64 = pos.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pos.iter().map(|p| p.0 * p.1).sum();
    let denom = nn * sxx - sx * sx;
    let (k0, ln_a0) = if denom.abs() > 1e-12 {
        let slope = (nn * sxy - sx * sy) / denom;
        (-slope, (sy - slope * sx) / nn)
    } else {
        (1.0, pos[0].1)
    };

    let t_owned = t.to_vec();
    let y_owned = y.to_vec();
    let residuals = move |p: &[f64]| -> Vec<f64> {
        t_owned
            .iter()
            .zip(&y_owned)
            .map(|(&ti, &yi)| p[0] * (-p[1] * ti).exp() - yi)
            .collect()
    };
    let fit = levenberg_marquardt(&residuals, None, &[ln_a0.exp(), k0], 1e-12, 200)?;
    Ok((fit.params[0], fit.params[1]))
}

/// Fits y ≈ A·exp(−(x−μ)²/(2σ²)), returning (A, μ, σ). Initial guess
/// from the sample peak and moment-based width, refined by LM.
pub fn fit_gaussian_peak(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64), SolveError> {
    if x.len() != y.len() {
        return Err(SolveError::DimensionMismatch { expected: x.len(), got: y.len() });
    }
    if x.len() < 3 {
        return Err(SolveError::InvalidArgument("fit_gaussian_peak requires >= 3 samples"));
    }
    // Peak-based initial guess.
    let (mut a0, mut mu0) = (f64::MIN, 0.0);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if yi > a0 {
            a0 = yi;
            mu0 = xi;
        }
    }
    if !(a0 > 0.0) {
        return Err(SolveError::InvalidArgument(
            "fit_gaussian_peak requires a positive peak value",
        ));
    }
    // Moment-based width from the y-weighted spread (clamped positive).
    let wsum: f64 = y.iter().filter(|&&v| v > 0.0).sum();
    let sigma0 = if wsum > 0.0 {
        let mean: f64 = x
            .iter()
            .zip(y.iter())
            .filter(|(_, &v)| v > 0.0)
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f64>()
            / wsum;
        let var: f64 = x
            .iter()
            .zip(y.iter())
            .filter(|(_, &v)| v > 0.0)
            .map(|(&xi, &yi)| yi * (xi - mean) * (xi - mean))
            .sum::<f64>()
            / wsum;
        var.sqrt().max(1e-3)
    } else {
        1.0
    };

    let x_owned = x.to_vec();
    let y_owned = y.to_vec();
    let residuals = move |p: &[f64]| -> Vec<f64> {
        let (a, mu, s) = (p[0], p[1], p[2]);
        x_owned
            .iter()
            .zip(&y_owned)
            .map(|(&xi, &yi)| {
                let z = (xi - mu) / s;
                a * (-0.5 * z * z).exp() - yi
            })
            .collect()
    };
    let fit = levenberg_marquardt(&residuals, None, &[a0, mu0, sigma0], 1e-12, 300)?;
    Ok((fit.params[0], fit.params[1], fit.params[2].abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_lm_linear_problem_exact() {
        // Residuals linear in p: converges immediately to the LS solution.
        let residuals = |p: &[f64]| vec![p[0] - 1.0, p[1] - 2.0, p[0] + p[1] - 3.0];
        let fit = levenberg_marquardt(&residuals, None, &[10.0, -4.0], 1e-12, 100).unwrap();
        assert!(approx(fit.params[0], 1.0, 1e-8));
        assert!(approx(fit.params[1], 2.0, 1e-8));
        assert!(fit.residual < 1e-15);
    }

    #[test]
    fn test_lm_rosenbrock_valley() {
        // Rosenbrock as least squares: r = (10(y - x²), 1 - x); min at (1, 1).
        let residuals = |p: &[f64]| vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]];
        let fit = levenberg_marquardt(&residuals, None, &[-1.2, 1.0], 1e-12, 500).unwrap();
        assert!(approx(fit.params[0], 1.0, 1e-6), "x = {}", fit.params[0]);
        assert!(approx(fit.params[1], 1.0, 1e-6), "y = {}", fit.params[1]);
    }

    #[test]
    fn test_lm_with_analytic_jacobian() {
        let residuals = |p: &[f64]| vec![p[0] * p[0] - 4.0];
        let jac = |p: &[f64]| {
            let mut j = Matrix::zeros(1, 1);
            j.set(0, 0, 2.0 * p[0]);
            j
        };
        let fit = levenberg_marquardt(&residuals, Some(&jac), &[1.0], 1e-14, 100).unwrap();
        assert!(approx(fit.params[0].abs(), 2.0, 1e-7));
    }

    #[test]
    fn test_lm_invalid_args() {
        let residuals = |_p: &[f64]| vec![0.0];
        assert!(levenberg_marquardt(&residuals, None, &[], 1e-10, 10).is_err());
        assert!(levenberg_marquardt(&residuals, None, &[1.0, 2.0], 1e-10, 10).is_err());
        assert!(levenberg_marquardt(&residuals, None, &[1.0], 0.0, 10).is_err());
    }

    #[test]
    fn test_fit_exponential_decay_with_noise() {
        let mut rng = Rng::new(61);
        let (a_true, k_true) = (2.5, 0.8);
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| a_true * (-k_true * ti).exp() * (1.0 + 0.001 * (rng.next_f64() - 0.5)))
            .collect();
        let (a, k) = fit_exponential_decay(&t, &y).unwrap();
        assert!(approx(a, a_true, 5e-3), "A = {a}");
        assert!(approx(k, k_true, 5e-3), "k = {k}");
    }

    #[test]
    fn test_fit_gaussian_peak_with_noise() {
        let mut rng = Rng::new(62);
        let (a_true, mu_true, s_true) = (3.0, 1.5, 0.4);
        let x: Vec<f64> = (0..80).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let z = (xi - mu_true) / s_true;
                a_true * (-0.5 * z * z).exp() + 0.001 * (rng.next_f64() - 0.5)
            })
            .collect();
        let (a, mu, s) = fit_gaussian_peak(&x, &y).unwrap();
        assert!(approx(a, a_true, 1e-2), "A = {a}");
        assert!(approx(mu, mu_true, 1e-2), "mu = {mu}");
        assert!(approx(s, s_true, 1e-2), "sigma = {s}");
    }

    #[test]
    fn test_covariance_present_for_overdetermined() {
        let residuals = |p: &[f64]| vec![p[0] - 1.0, p[0] - 1.1, p[0] - 0.9];
        let fit = levenberg_marquardt(&residuals, None, &[0.0], 1e-12, 100).unwrap();
        let cov = fit.covariance.expect("covariance expected");
        assert_eq!(cov.rows, 1);
        assert!(cov.get(0, 0) > 0.0);
    }
}
