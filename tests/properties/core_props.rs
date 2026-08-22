//! Properties for `core::compensated` and `core::dual`.

use rust_physics_engine::core::compensated::{dot_compensated, sum_neumaier, sum_pairwise};
use rust_physics_engine::core::dual::{gradient, Dual};
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::optimization::numerical_gradient_vec;

/// Dual-number gradients match the finite-difference gradient to 1e-6
/// and are exact on polynomials.
#[test]
fn prop_dual_gradient_matches_numerical() {
    let mut rng = Rng::new(41);
    let f_dual = |v: &[Dual]| {
        v[0] * v[0] * v[1] + (v[1] * v[2]).sin() + v[2].exp() / (v[0] + Dual::constant(3.0))
    };
    let f_num =
        |v: &[f64]| v[0] * v[0] * v[1] + (v[1] * v[2]).sin() + v[2].exp() / (v[0] + 3.0);
    for _ in 0..50 {
        let x: Vec<f64> = (0..3).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let g_ad = gradient(f_dual, &x);
        let g_fd = numerical_gradient_vec(&f_num, &x, 1e-6);
        for (a, b) in g_ad.iter().zip(&g_fd) {
            assert!((a - b).abs() < 1e-5, "AD {a} vs FD {b}");
        }
    }
}

/// Exactness on polynomials: derivative of x^3 y^2 has no rounding
/// beyond the arithmetic itself.
#[test]
fn prop_dual_polynomial_exact() {
    let mut rng = Rng::new(42);
    let f = |v: &[Dual]| v[0].powi(3) * v[1].powi(2);
    for _ in 0..100 {
        let x = rng.next_f64() * 4.0 - 2.0;
        let y = rng.next_f64() * 4.0 - 2.0;
        let g = gradient(f, &[x, y]);
        assert_eq!(g[0], 3.0 * x.powi(2) * y.powi(2));
        assert_eq!(g[1], x.powi(3) * 2.0 * y);
    }
}

/// sum_neumaier of 1e6 copies of 0.1 is within 1e-9 of 1e5; the naive sum is not.
#[test]
fn prop_neumaier_beats_naive_on_tenths() {
    let xs = vec![0.1_f64; 1_000_000];
    let naive: f64 = xs.iter().sum();
    let compensated = sum_neumaier(&xs);
    assert!(
        (compensated - 1e5).abs() < 1e-9,
        "compensated sum off by {}",
        (compensated - 1e5).abs()
    );
    assert!(
        (naive - 1e5).abs() >= 1e-9,
        "naive sum unexpectedly accurate: off by {}",
        (naive - 1e5).abs()
    );
}

/// Compensated and pairwise sums agree with each other on random data to
/// much tighter tolerance than the naive error bound.
#[test]
fn prop_sums_agree_on_random_data() {
    let mut rng = Rng::new(42);
    for _ in 0..20 {
        let xs: Vec<f64> = (0..10_000).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let a = sum_neumaier(&xs);
        let b = sum_pairwise(&xs);
        assert!((a - b).abs() < 1e-9, "neumaier {a} vs pairwise {b}");
    }
}

/// dot_compensated matches the naive dot product on well-conditioned data.
#[test]
fn prop_dot_matches_naive_when_well_conditioned() {
    let mut rng = Rng::new(7);
    for _ in 0..20 {
        let a: Vec<f64> = (0..1000).map(|_| rng.next_f64()).collect();
        let b: Vec<f64> = (0..1000).map(|_| rng.next_f64()).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let comp = dot_compensated(&a, &b);
        assert!((naive - comp).abs() < 1e-9);
    }
}
