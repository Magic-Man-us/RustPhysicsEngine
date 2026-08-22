//! Numerical methods: quadrature, root finding, ODE solvers, and
//! interpolation. Submodules are re-exported so historical paths such as
//! `crate::numerical::trapezoid` keep working.

pub mod integrate;
pub mod interpolate;
pub mod ode;
pub mod roots;

pub use integrate::{gaussian_quadrature_5, simpson, trapezoid};
pub use interpolate::{cubic_interp, lerp, linear_interp};
pub use ode::{euler_step, rk4_solve, rk4_step, rk4_step_vec};
pub use roots::{bisection, newton_raphson, secant};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    const INTEGRATION_TOL: f64 = 1e-6;
    const ROOT_TOL: f64 = 1e-10;
    const ODE_TOL: f64 = 1e-4;
    const MAX_ITER: usize = 1000;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── Integration tests: ∫sin(x)dx from 0 to π = 2.0 ────────────

    #[test]
    fn test_trapezoid_sin() {
        let result = trapezoid(&f64::sin, 0.0, PI, 10_000);
        assert!(
            approx_eq(result, 2.0, INTEGRATION_TOL),
            "trapezoid: expected 2.0, got {result}"
        );
    }

    #[test]
    fn test_simpson_sin() {
        let result = simpson(&f64::sin, 0.0, PI, 1_000);
        assert!(
            approx_eq(result, 2.0, INTEGRATION_TOL),
            "simpson: expected 2.0, got {result}"
        );
    }

    #[test]
    fn test_simpson_clamps_odd_n() {
        let result = simpson(&f64::sin, 0.0, PI, 999);
        assert!(
            approx_eq(result, 2.0, INTEGRATION_TOL),
            "simpson with odd n: expected 2.0, got {result}"
        );
    }

    #[test]
    fn test_gaussian_quadrature_sin() {
        let result = gaussian_quadrature_5(&f64::sin, 0.0, PI);
        assert!(
            approx_eq(result, 2.0, 1e-5),
            "gauss5: expected 2.0, got {result}"
        );
    }

    // ── Root finding tests: x² - 2 = 0, root = √2 ─────────────────

    #[test]
    fn test_bisection_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = bisection(&f, 1.0, 2.0, ROOT_TOL, MAX_ITER).unwrap();
        assert!(
            approx_eq(root, std::f64::consts::SQRT_2, ROOT_TOL),
            "bisection: expected √2, got {root}"
        );
    }

    #[test]
    fn test_bisection_same_sign_returns_none() {
        let f = |x: f64| x * x + 1.0;
        assert!(bisection(&f, 0.0, 1.0, ROOT_TOL, MAX_ITER).is_none());
    }

    #[test]
    fn test_newton_raphson_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let root = newton_raphson(&f, &df, 1.5, ROOT_TOL, MAX_ITER).unwrap();
        assert!(
            approx_eq(root, std::f64::consts::SQRT_2, ROOT_TOL),
            "newton: expected √2, got {root}"
        );
    }

    #[test]
    fn test_newton_raphson_zero_derivative() {
        assert!(newton_raphson(&|_: f64| 1.0, &|_: f64| 0.0, 1.0, ROOT_TOL, 10).is_none());
    }

    #[test]
    fn test_secant_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = secant(&f, 1.0, 2.0, ROOT_TOL, MAX_ITER).unwrap();
        assert!(
            approx_eq(root, std::f64::consts::SQRT_2, ROOT_TOL),
            "secant: expected √2, got {root}"
        );
    }

    // ── ODE tests: dy/dt = -y, y(0) = 1 → y(t) = e^(-t) ──────────

    #[test]
    fn test_euler_step() {
        let f = |_t: f64, y: f64| -y;
        let y1 = euler_step(&f, 0.0, 1.0, 0.01);
        assert!(
            approx_eq(y1, 0.99, 1e-12),
            "euler step: expected 0.99, got {y1}"
        );
    }

    #[test]
    fn test_rk4_step() {
        let f = |_t: f64, y: f64| -y;
        let dt = 0.1;
        let y1 = rk4_step(&f, 0.0, 1.0, dt);
        let expected = 0.9048374180359595;
        assert!(
            approx_eq(y1, expected, 1e-6),
            "rk4 step: expected {expected}, got {y1}"
        );
    }

    #[test]
    fn test_rk4_solve_exponential_decay() {
        let f = |_t: f64, y: f64| -y;
        let dt = 0.01;
        let results = rk4_solve(&f, 0.0, 1.0, 5.0, dt);
        for &(t, y) in &results {
            let exact = (-t).exp();
            assert!(
                approx_eq(y, exact, ODE_TOL),
                "rk4 solve at t={t}: expected {exact}, got {y}"
            );
        }
    }

    #[test]
    fn test_rk4_step_vec_system() {
        // Coupled system: harmonic oscillator x'' = -x
        // State: [x, v], derivatives: [v, -x]
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let y0 = vec![1.0, 0.0]; // x=1, v=0 → x(t) = cos(t)
        let dt = 0.01;
        let mut t = 0.0;
        let mut y = y0;
        let steps = 100;
        for _ in 0..steps {
            y = rk4_step_vec(&f, t, &y, dt);
            t += dt;
        }
        let exact_x = t.cos();
        let got_x = y[0];
        assert!(
            approx_eq(got_x, exact_x, 1e-6),
            "rk4 vec at t={t}: expected x={exact_x}, got {got_x}",
        );
    }

    // ── Interpolation tests ────────────────────────────────────────

    #[test]
    fn test_lerp() {
        assert!(approx_eq(lerp(0.0, 10.0, 0.5), 5.0, 1e-15));
        assert!(approx_eq(lerp(0.0, 10.0, 0.0), 0.0, 1e-15));
        assert!(approx_eq(lerp(0.0, 10.0, 1.0), 10.0, 1e-15));
    }

    #[test]
    fn test_linear_interp_exact_points() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 2.0, 4.0, 6.0];
        assert!(approx_eq(linear_interp(&xs, &ys, 1.5), 3.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 0.0), 0.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 3.0), 6.0, 1e-12));
    }

    #[test]
    fn test_linear_interp_clamping() {
        let xs = [1.0, 2.0, 3.0];
        let ys = [10.0, 20.0, 30.0];
        assert!(approx_eq(linear_interp(&xs, &ys, 0.0), 10.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 5.0), 30.0, 1e-12));
    }

    #[test]
    fn test_cubic_interp_quadratic() {
        // y = x^2 sampled at 5 points; natural cubic spline is close but not exact
        let xs: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        for (&x, &expected) in [0.5, 1.5, 2.5, 3.5].iter().zip(&[0.25, 2.25, 6.25, 12.25]) {
            let got = cubic_interp(&xs, &ys, x);
            assert!(
                approx_eq(got, expected, 0.5),
                "cubic at x={x}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_cubic_interp_two_points_falls_back() {
        let xs = [0.0, 1.0];
        let ys = [0.0, 1.0];
        let got = cubic_interp(&xs, &ys, 0.5);
        let expected = 0.5;
        assert!(
            approx_eq(got, expected, 1e-12),
            "cubic 2-point fallback: expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_cubic_interp_few_points_falls_back() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 4.0];
        let got = cubic_interp(&xs, &ys, 0.5);
        let expected = 0.5;
        assert!(
            approx_eq(got, expected, 1e-12),
            "cubic fallback: expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_integration_polynomial() {
        // ∫(x^2)dx from 0 to 3 = 9
        let f = |x: f64| x * x;
        let trap = trapezoid(&f, 0.0, 3.0, 10_000);
        let simp = simpson(&f, 0.0, 3.0, 100);
        let gauss = gaussian_quadrature_5(&f, 0.0, 3.0);
        assert!(approx_eq(trap, 9.0, 1e-4), "trap x^2: got {trap}");
        assert!(approx_eq(simp, 9.0, 1e-10), "simpson x^2: got {simp}");
        assert!(approx_eq(gauss, 9.0, 1e-10), "gauss x^2: got {gauss}");
    }

    #[test]
    fn test_bisection_max_iter_reached() {
        let f = |x: f64| x.sin();
        let result = bisection(&f, 2.5, 3.8, 1e-20, 3);
        assert!(result.is_some());
    }

    #[test]
    fn test_newton_raphson_no_convergence() {
        // f(x) = x^2 + 1 has no real roots; Newton's method oscillates
        let f = |x: f64| x * x + 1.0;
        let df = |x: f64| 2.0 * x;
        let result = newton_raphson(&f, &df, 1.0, 1e-12, 100);
        assert!(result.is_none());
    }

    #[test]
    fn test_newton_raphson_max_iter() {
        let f = |x: f64| x.sin();
        let df = |x: f64| x.cos();
        let result = newton_raphson(&f, &df, 1.0, 1e-100, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_secant_no_convergence() {
        let f = |x: f64| x * x + 1.0;
        let result = secant(&f, 0.0, 0.0, 1e-12, 100);
        assert!(result.is_none());
    }

    #[test]
    fn test_secant_max_iter() {
        let f = |x: f64| x.sin();
        let result = secant(&f, 2.0, 3.5, 1e-100, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_cubic_spline_two_points() {
        let x_data = vec![0.0, 1.0];
        let y_data = vec![0.0, 2.0];
        let y = cubic_interp(&x_data, &y_data, 0.5);
        assert!(approx_eq(y, 1.0, 1e-12));
    }

    #[test]
    fn test_simpson_n_less_than_2() {
        let result = simpson(&|x: f64| x * x, 0.0, 1.0, 1);
        // n=1 gets rounded up to n=2; Simpson's rule is exact for polynomials up to degree 3
        assert!(approx_eq(result, 1.0 / 3.0, 1e-12));
    }
}
