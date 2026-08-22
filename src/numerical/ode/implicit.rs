//! Implicit (stiff-stable) ODE steps: backward Euler and BDF2.
//!
//! Both solve their implicit update equation with damped-free Newton
//! iteration; the Jacobian of f is user-supplied or, when `None`,
//! approximated by forward finite differences (an opaque `f64` closure
//! cannot be differentiated with `core::dual`, so finite differences
//! stand in for it here).

use crate::error::SolveError;
use crate::linalg::{lu_decompose, Matrix};

/// Forward-difference Jacobian of f(t, ·) at y.
fn fd_jacobian(f: &dyn Fn(f64, &[f64]) -> Vec<f64>, t: f64, y: &[f64], fy: &[f64]) -> Matrix {
    let n = y.len();
    let mut j = Matrix::zeros(n, n);
    let mut yp = y.to_vec();
    for c in 0..n {
        let h = (f64::EPSILON.sqrt()) * y[c].abs().max(1.0);
        yp[c] = y[c] + h;
        let fp = f(t, &yp);
        yp[c] = y[c];
        for r in 0..n {
            j.set(r, c, (fp[r] - fy[r]) / h);
        }
    }
    j
}

/// Newton solve of u − beta·dt·f(t_new, u) = rhs, shared by both steps.
#[allow(clippy::too_many_arguments)]
fn newton_implicit(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    jac: Option<&dyn Fn(f64, &[f64]) -> Matrix>,
    t_new: f64,
    rhs: &[f64],
    u0: &[f64],
    beta_dt: f64,
    newton_tol: f64,
    newton_iter: usize,
) -> Result<Vec<f64>, SolveError> {
    let n = rhs.len();
    let mut u = u0.to_vec();
    for _ in 0..newton_iter {
        let fu = f(t_new, &u);
        if fu.len() != n {
            return Err(SolveError::DimensionMismatch { expected: n, got: fu.len() });
        }
        // Residual G(u) = u - beta*dt*f - rhs.
        let g: Vec<f64> = (0..n).map(|i| u[i] - beta_dt * fu[i] - rhs[i]).collect();
        let gnorm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if gnorm <= newton_tol {
            return Ok(u);
        }
        // J_G = I - beta*dt*J_f.
        let jf = match jac {
            Some(jfn) => jfn(t_new, &u),
            None => fd_jacobian(f, t_new, &u, &fu),
        };
        if jf.rows != n || jf.cols != n {
            return Err(SolveError::DimensionMismatch { expected: n, got: jf.rows });
        }
        let mut jg = jf.scale(-beta_dt);
        for i in 0..n {
            let v = jg.get(i, i) + 1.0;
            jg.set(i, i, v);
        }
        let delta = lu_decompose(&jg)?.solve(&g)?;
        for i in 0..n {
            u[i] -= delta[i];
        }
    }
    // Converged?
    let fu = f(t_new, &u);
    let g: Vec<f64> = (0..n).map(|i| u[i] - beta_dt * fu[i] - rhs[i]).collect();
    let gnorm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
    if gnorm <= newton_tol {
        Ok(u)
    } else {
        Err(SolveError::NoConvergence { iters: newton_iter, residual: gnorm })
    }
}

/// One backward-Euler step: solves y₁ = y + dt·f(t+dt, y₁) by Newton
/// iteration. A-stable, first order; stable on stiff systems (e.g.
/// y' = −1000·y with dt = 0.1) where explicit RK4 diverges.
#[allow(clippy::too_many_arguments)]
pub fn backward_euler(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    jac: Option<&dyn Fn(f64, &[f64]) -> Matrix>,
    t: f64,
    y: &[f64],
    dt: f64,
    newton_tol: f64,
    newton_iter: usize,
) -> Result<Vec<f64>, SolveError> {
    if y.is_empty() {
        return Err(SolveError::InvalidArgument("backward_euler requires a non-empty state"));
    }
    if !(dt > 0.0) || !(newton_tol > 0.0) || newton_iter == 0 {
        return Err(SolveError::InvalidArgument(
            "backward_euler requires dt > 0, newton_tol > 0, newton_iter > 0",
        ));
    }
    newton_implicit(f, jac, t + dt, y, y, dt, newton_tol, newton_iter)
}

/// One BDF2 step: solves
/// y₂ − (4/3)y₁ + (1/3)y₀ = (2/3)·dt·f(t+dt, y₂)
/// where `y` is the current state y₁ at time t and `y_prev` is y₀ one
/// step earlier. Second order, A-stable.
#[allow(clippy::too_many_arguments)]
pub fn bdf2(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    jac: Option<&dyn Fn(f64, &[f64]) -> Matrix>,
    t: f64,
    y_prev: &[f64],
    y: &[f64],
    dt: f64,
    newton_tol: f64,
    newton_iter: usize,
) -> Result<Vec<f64>, SolveError> {
    if y.is_empty() {
        return Err(SolveError::InvalidArgument("bdf2 requires a non-empty state"));
    }
    if y_prev.len() != y.len() {
        return Err(SolveError::DimensionMismatch { expected: y.len(), got: y_prev.len() });
    }
    if !(dt > 0.0) || !(newton_tol > 0.0) || newton_iter == 0 {
        return Err(SolveError::InvalidArgument(
            "bdf2 requires dt > 0, newton_tol > 0, newton_iter > 0",
        ));
    }
    let n = y.len();
    let rhs: Vec<f64> = (0..n).map(|i| (4.0 * y[i] - y_prev[i]) / 3.0).collect();
    newton_implicit(f, jac, t + dt, &rhs, y, 2.0 * dt / 3.0, newton_tol, newton_iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::ode::rk4_step_vec;

    #[test]
    fn test_backward_euler_stiff_stable_where_rk4_diverges() {
        // y' = -1000 y, dt = 0.1: RK4 blows up, backward Euler decays.
        let f = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
        let dt = 0.1;
        let mut y_be = vec![1.0];
        let mut y_rk = vec![1.0];
        for k in 0..50 {
            y_be = backward_euler(&f, None, k as f64 * dt, &y_be, dt, 1e-12, 50).unwrap();
            y_rk = rk4_step_vec(&f, k as f64 * dt, &y_rk, dt);
        }
        assert!(y_be[0].abs() < 1e-10, "backward Euler should decay, got {}", y_be[0]);
        assert!(
            !y_rk[0].is_finite() || y_rk[0].abs() > 1e10,
            "RK4 should diverge, got {}",
            y_rk[0]
        );
    }

    #[test]
    fn test_backward_euler_accuracy_first_order() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let dt = 0.001;
        let mut y = vec![1.0];
        for k in 0..1000 {
            y = backward_euler(&f, None, k as f64 * dt, &y, dt, 1e-13, 20).unwrap();
        }
        assert!((y[0] - (-1.0_f64).exp()).abs() < 1e-3);
    }

    #[test]
    fn test_backward_euler_with_analytic_jacobian() {
        let f = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
        let jac = |_t: f64, _y: &[f64]| {
            let mut m = Matrix::zeros(1, 1);
            m.set(0, 0, -1000.0);
            m
        };
        let y = backward_euler(&f, Some(&jac), 0.0, &[1.0], 0.1, 1e-12, 20).unwrap();
        // Exact backward Euler update: 1/(1 + 1000*0.1)
        assert!((y[0] - 1.0 / 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_bdf2_stiff_stable() {
        let f = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
        let dt = 0.1;
        // Bootstrap the two-step history with one backward Euler step.
        let y0 = vec![1.0];
        let y1 = backward_euler(&f, None, 0.0, &y0, dt, 1e-12, 50).unwrap();
        let (mut y_prev, mut y) = (y0, y1);
        for k in 1..50 {
            let y_next = bdf2(&f, None, k as f64 * dt, &y_prev, &y, dt, 1e-12, 50).unwrap();
            y_prev = y;
            y = y_next;
        }
        assert!(y[0].abs() < 1e-10, "BDF2 should decay, got {}", y[0]);
    }

    #[test]
    fn test_bdf2_second_order_accuracy() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let dt = 0.01;
        let y0 = vec![1.0];
        // Exact first step to avoid polluting the order measurement.
        let y1 = vec![(-dt_f(dt)).exp()];
        let (mut y_prev, mut y) = (y0, y1);
        let steps = 100;
        for k in 1..steps {
            let y_next = bdf2(&f, None, k as f64 * dt, &y_prev, &y, dt, 1e-14, 20).unwrap();
            y_prev = y;
            y = y_next;
        }
        let t_end = steps as f64 * dt;
        // Second order: global error ~ C·dt² = O(1e-4) at dt = 0.01.
        assert!((y[0] - (-t_end).exp()).abs() < 1e-4, "err {}", (y[0] - (-t_end).exp()).abs());
    }

    fn dt_f(dt: f64) -> f64 {
        dt
    }

    #[test]
    fn test_invalid_arguments() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        assert!(backward_euler(&f, None, 0.0, &[], 0.1, 1e-10, 10).is_err());
        assert!(backward_euler(&f, None, 0.0, &[1.0], -0.1, 1e-10, 10).is_err());
        assert!(bdf2(&f, None, 0.0, &[1.0, 2.0], &[1.0], 0.1, 1e-10, 10).is_err());
    }
}
