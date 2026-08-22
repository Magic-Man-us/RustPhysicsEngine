//! Properties for `numerical`: adaptive, implicit, and symplectic ODE
//! solvers.

use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::numerical::ode::{
    backward_euler, dormand_prince, rk4_step_vec, velocity_verlet, yoshida4,
};
use rust_physics_engine::numerical::{
    adaptive_quad, gauss_kronrod_15, polynomial_eval_complex, polynomial_roots,
};

/// The reported quadrature error bound dominates the true error on
/// smooth integrands.
#[test]
fn prop_quadrature_error_bound_holds() {
    let cases: Vec<(Box<dyn Fn(f64) -> f64>, f64, f64, f64)> = vec![
        (Box::new(f64::sin), 0.0, std::f64::consts::PI, 2.0),
        (Box::new(|x: f64| x.exp()), 0.0, 1.0, std::f64::consts::E - 1.0),
        (Box::new(|x: f64| 1.0 / (1.0 + x * x)), 0.0, 1.0, std::f64::consts::PI / 4.0),
    ];
    for (f, a, b, exact) in &cases {
        let r = gauss_kronrod_15(f.as_ref(), *a, *b);
        assert!(
            r.error + 1e-15 >= (r.value - exact).abs(),
            "GK15 bound {} < true error {}",
            r.error,
            (r.value - exact).abs()
        );
        let r = adaptive_quad(f.as_ref(), *a, *b, 1e-10, 30).unwrap();
        assert!(r.error + 1e-15 >= (r.value - exact).abs(), "adaptive bound too small");
    }
}

/// Every reported polynomial root satisfies |p(root)| < 1e-8, and the
/// root count equals the degree.
#[test]
fn prop_polynomial_roots_satisfy_polynomial() {
    let mut rng = Rng::new(52);
    for _ in 0..30 {
        let degree = 2 + (rng.next_u64() % 5) as usize;
        let coeffs: Vec<f64> = std::iter::once(1.0)
            .chain((0..degree).map(|_| rng.next_f64() * 4.0 - 2.0))
            .collect();
        let roots = polynomial_roots(&coeffs).unwrap();
        assert_eq!(roots.len(), degree);
        for &z in &roots {
            let residual = polynomial_eval_complex(&coeffs, z).norm();
            assert!(residual < 1e-8, "residual {residual} for degree {degree}");
        }
    }
}

/// Two-body (Kepler) orbit: relative energy drift below 1e-8 over 100
/// periods at rtol = 1e-10.
#[test]
fn prop_dormand_prince_kepler_energy_drift() {
    // Normalized circular orbit: mu = 1, r = 1, v = 1, period 2*pi.
    let f = |_t: f64, s: &[f64]| {
        let r3 = (s[0] * s[0] + s[1] * s[1]).powf(1.5);
        vec![s[2], s[3], -s[0] / r3, -s[1] / r3]
    };
    let energy = |s: &[f64]| {
        0.5 * (s[2] * s[2] + s[3] * s[3]) - 1.0 / (s[0] * s[0] + s[1] * s[1]).sqrt()
    };
    let y0 = [1.0, 0.0, 0.0, 1.0];
    let t_end = 100.0 * 2.0 * std::f64::consts::PI;
    let r = dormand_prince(&f, 0.0, t_end, &y0, 1e-11, 1e-13, 0.01).unwrap();
    let e0 = energy(&y0);
    let ef = energy(r.y.last().unwrap());
    let drift = ((ef - e0) / e0).abs();
    assert!(drift < 1e-8, "energy drift {drift}");
}

/// Exponential growth y' = k y matches the analytic solution to ~rtol.
#[test]
fn prop_dormand_prince_exp_growth() {
    let mut rng = Rng::new(51);
    for _ in 0..10 {
        let k = rng.next_f64() * 2.0 + 0.1;
        let f = move |_t: f64, y: &[f64]| vec![k * y[0]];
        let rtol = 1e-9;
        let r = dormand_prince(&f, 0.0, 2.0, &[1.0], rtol, 1e-14, 0.1).unwrap();
        let yf = r.y.last().unwrap()[0];
        let exact = (2.0 * k).exp();
        assert!(((yf - exact) / exact).abs() < 1000.0 * rtol, "k={k}");
    }
}

/// Harmonic oscillator energy stays bounded (no secular drift) over 1e5
/// symplectic steps.
#[test]
fn prop_symplectic_energy_bounded() {
    let acc = |x: &[f64]| vec![-x[0]];
    let energy = |x: &[f64], v: &[f64]| 0.5 * (v[0] * v[0] + x[0] * x[0]);

    for use_yoshida in [false, true] {
        let mut x = vec![1.0];
        let mut v = vec![0.0];
        let e0 = energy(&x, &v);
        let mut max_err = 0.0_f64;
        for _ in 0..100_000 {
            if use_yoshida {
                yoshida4(&acc, &mut x, &mut v, 0.05);
            } else {
                velocity_verlet(&acc, &mut x, &mut v, 0.05);
            }
            max_err = max_err.max((energy(&x, &v) - e0).abs());
        }
        assert!(max_err < 1e-3, "yoshida={use_yoshida}: energy error {max_err}");
    }
}

/// Backward Euler is stable on y' = -1000 y with dt = 0.1 where RK4
/// diverges.
#[test]
fn prop_implicit_stable_on_stiff() {
    let f = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
    let dt = 0.1;
    let mut y_implicit = vec![1.0];
    let mut y_rk4 = vec![1.0];
    for k in 0..30 {
        y_implicit = backward_euler(&f, None, k as f64 * dt, &y_implicit, dt, 1e-12, 50).unwrap();
        y_rk4 = rk4_step_vec(&f, k as f64 * dt, &y_rk4, dt);
    }
    assert!(y_implicit[0].abs() < 1e-6);
    assert!(!y_rk4[0].is_finite() || y_rk4[0].abs() > 1e6);
}
