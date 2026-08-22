//! Properties for `special`: erf family, gamma family, beta.

use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::special::{beta_inc, erf, erfc, erfinv, gamma, gamma_p, gamma_q};
use rust_physics_engine::statistics::factorial;

/// erf(-x) == -erf(x)
#[test]
fn prop_erf_odd() {
    let mut rng = Rng::new(21);
    for _ in 0..200 {
        let x = (rng.next_f64() - 0.5) * 12.0;
        assert!((erf(-x) + erf(x)).abs() < 1e-15, "x={x}");
    }
}

/// erf(x) + erfc(x) == 1
#[test]
fn prop_erf_plus_erfc() {
    let mut rng = Rng::new(22);
    for _ in 0..200 {
        let x = (rng.next_f64() - 0.5) * 12.0;
        assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-14, "x={x}");
    }
}

/// erfinv(erf(x)) == x to 1e-12 on |x| < 5, up to the ulp-of-p limit:
/// for |x| ≳ 3.2 the rounding of p = erf(x) toward 1 alone perturbs the
/// preimage by ~eps/erf'(x), so the bound widens accordingly there.
#[test]
fn prop_erfinv_roundtrip() {
    let mut rng = Rng::new(23);
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    for _ in 0..500 {
        let x = (rng.next_f64() - 0.5) * 10.0;
        let back = erfinv(erf(x));
        let deriv = two_over_sqrt_pi * (-x * x).exp();
        let tol = 1e-12_f64.max(4.0 * f64::EPSILON / deriv);
        assert!((back - x).abs() < tol, "x={x}, back={back}, tol={tol}");
    }
}

/// gamma(n+1) == n! for n < 20.
#[test]
fn prop_gamma_matches_factorial() {
    for n in 0..20u64 {
        let g = gamma(n as f64 + 1.0);
        let f = factorial(n);
        assert!(
            (g / f - 1.0).abs() < 1e-11,
            "n={n}: gamma={g}, factorial={f}"
        );
    }
}

/// gamma_p + gamma_q == 1.
#[test]
fn prop_gamma_p_plus_q() {
    let mut rng = Rng::new(24);
    for _ in 0..200 {
        let a = rng.next_f64() * 20.0 + 0.05;
        let x = rng.next_f64() * 40.0;
        let s = gamma_p(a, x) + gamma_q(a, x);
        assert!((s - 1.0).abs() < 1e-12, "a={a}, x={x}: {s}");
    }
}

/// beta_inc(a, b, x) == 1 - beta_inc(b, a, 1-x).
#[test]
fn prop_beta_inc_symmetry() {
    let mut rng = Rng::new(25);
    for _ in 0..200 {
        let a = rng.next_f64() * 10.0 + 0.1;
        let b = rng.next_f64() * 10.0 + 0.1;
        let x = rng.next_f64();
        let lhs = beta_inc(a, b, x);
        let rhs = 1.0 - beta_inc(b, a, 1.0 - x);
        assert!((lhs - rhs).abs() < 1e-11, "a={a}, b={b}, x={x}");
    }
}
