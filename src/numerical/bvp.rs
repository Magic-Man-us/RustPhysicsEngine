//! Two-point boundary value problems.
//!
//! `shooting` reduces y'' = f(t, y, y') with y(t0) = y0, y(t1) = y1 to
//! root finding on the initial slope (integrated with Dormand-Prince,
//! slope found with Brent's method). `finite_difference_linear_bvp`
//! discretizes the linear problem y'' + p·y' + q·y = r on a uniform
//! grid and solves the tridiagonal system with the Thomas algorithm
//! (Burden & Faires, *Numerical Analysis*, §11.3).

use crate::error::SolveError;
use crate::linalg::thomas_solve;
use crate::numerical::ode::dormand_prince;
use crate::numerical::roots::brent_root;

const SHOOT_RTOL: f64 = 1e-10;
const SHOOT_ATOL: f64 = 1e-12;

/// Solves y'' = f(t, y, y') with y(t0) = y0 and y(t1) = y1_target by
/// the shooting method: the unknown initial slope is bracketed by
/// [guess_lo, guess_hi] and found with Brent's method.
///
/// Returns the (t, y) trajectory of the converged solution. Fails with
/// `InvalidArgument` if the bracket does not straddle the target.
#[allow(clippy::too_many_arguments)]
pub fn shooting(
    f: &dyn Fn(f64, f64, f64) -> f64,
    t0: f64,
    t1: f64,
    y0: f64,
    y1_target: f64,
    guess_lo: f64,
    guess_hi: f64,
    tol: f64,
) -> Result<Vec<(f64, f64)>, SolveError> {
    if !(t1 > t0) {
        return Err(SolveError::InvalidArgument("shooting requires t1 > t0"));
    }
    if !(tol > 0.0) {
        return Err(SolveError::InvalidArgument("shooting requires tol > 0"));
    }
    let system = |t: f64, s: &[f64]| vec![s[1], f(t, s[0], s[1])];
    let endpoint = |slope: f64| -> Result<f64, SolveError> {
        let r = dormand_prince(
            &system,
            t0,
            t1,
            &[y0, slope],
            SHOOT_RTOL,
            SHOOT_ATOL,
            (t1 - t0) / 100.0,
        )?;
        Ok(r.y.last().unwrap()[0])
    };

    // Brent on g(slope) = y(t1; slope) - y1_target.
    let miss = |slope: f64| match endpoint(slope) {
        Ok(y_end) => y_end - y1_target,
        Err(_) => f64::NAN,
    };
    let slope = brent_root(&miss, guess_lo, guess_hi, tol, 200)?;

    let r = dormand_prince(
        &system,
        t0,
        t1,
        &[y0, slope],
        SHOOT_RTOL,
        SHOOT_ATOL,
        (t1 - t0) / 100.0,
    )?;
    Ok(r.t.iter().zip(&r.y).map(|(&t, s)| (t, s[0])).collect())
}

/// Solves the linear BVP y'' + p(x)·y' + q(x)·y = r(x) on [a, b] with
/// y(a) = ya, y(b) = yb using n interior points and second-order
/// central differences; the tridiagonal system goes to `thomas_solve`.
///
/// Returns the full grid of n + 2 values including both boundaries.
#[allow(clippy::too_many_arguments)]
pub fn finite_difference_linear_bvp(
    p: &dyn Fn(f64) -> f64,
    q: &dyn Fn(f64) -> f64,
    r: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    ya: f64,
    yb: f64,
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    if !(b > a) {
        return Err(SolveError::InvalidArgument("finite_difference_linear_bvp requires b > a"));
    }
    if n == 0 {
        return Err(SolveError::InvalidArgument(
            "finite_difference_linear_bvp requires at least one interior point",
        ));
    }
    let h = (b - a) / (n + 1) as f64;
    let inv_h2 = 1.0 / (h * h);

    let mut sub = vec![0.0; n - 1];
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n - 1];
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        let x = a + (i + 1) as f64 * h;
        let pi = p(x);
        let qi = q(x);
        // (y_{i-1} - 2 y_i + y_{i+1})/h^2 + p_i (y_{i+1}-y_{i-1})/(2h) + q_i y_i = r_i
        let lower = inv_h2 - pi / (2.0 * h);
        let upper = inv_h2 + pi / (2.0 * h);
        diag[i] = -2.0 * inv_h2 + qi;
        rhs[i] = r(x);
        if i > 0 {
            sub[i - 1] = lower;
        } else {
            rhs[i] -= lower * ya;
        }
        if i < n - 1 {
            sup[i] = upper;
        } else {
            rhs[i] -= upper * yb;
        }
    }
    let interior = thomas_solve(&sub, &diag, &sup, &rhs)?;

    let mut full = Vec::with_capacity(n + 2);
    full.push(ya);
    full.extend(interior);
    full.push(yb);
    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    #[test]
    fn test_shooting_recovers_sin() {
        // y'' = -y, y(0) = 0, y(pi/2) = 1 → y = sin(t)
        let f = |_t: f64, y: f64, _dy: f64| -y;
        let sol = shooting(&f, 0.0, PI / 2.0, 0.0, 1.0, 0.0, 2.0, 1e-10).unwrap();
        for &(t, y) in &sol {
            assert!((y - t.sin()).abs() < 1e-7, "t={t}: y={y} vs {}", t.sin());
        }
    }

    #[test]
    fn test_shooting_no_bracket() {
        let f = |_t: f64, y: f64, _dy: f64| -y;
        // Both slopes overshoot: g has the same sign at both ends.
        assert!(shooting(&f, 0.0, PI / 2.0, 0.0, 1.0, 5.0, 9.0, 1e-10).is_err());
    }

    #[test]
    fn test_fd_bvp_recovers_sin() {
        // y'' + y = 0, y(0) = 0, y(pi/2) = 1 → sin
        let sol = finite_difference_linear_bvp(
            &|_| 0.0,
            &|_| 1.0,
            &|_| 0.0,
            0.0,
            PI / 2.0,
            0.0,
            1.0,
            199,
        )
        .unwrap();
        assert_eq!(sol.len(), 201);
        let h = (PI / 2.0) / 200.0;
        for (i, &y) in sol.iter().enumerate() {
            let x = i as f64 * h;
            assert!((y - x.sin()).abs() < 1e-4, "x={x}: {y} vs {}", x.sin());
        }
    }

    #[test]
    fn test_fd_bvp_linear_exact() {
        // y'' = 0 with linear boundary data is reproduced exactly.
        let sol =
            finite_difference_linear_bvp(&|_| 0.0, &|_| 0.0, &|_| 0.0, 0.0, 1.0, 2.0, 5.0, 9)
                .unwrap();
        for (i, &y) in sol.iter().enumerate() {
            let x = i as f64 / 10.0;
            assert!((y - (2.0 + 3.0 * x)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fd_bvp_invalid_args() {
        assert!(
            finite_difference_linear_bvp(&|_| 0.0, &|_| 0.0, &|_| 0.0, 1.0, 0.0, 0.0, 0.0, 5)
                .is_err()
        );
        assert!(
            finite_difference_linear_bvp(&|_| 0.0, &|_| 0.0, &|_| 0.0, 0.0, 1.0, 0.0, 0.0, 0)
                .is_err()
        );
    }
}
