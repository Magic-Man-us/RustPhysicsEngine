//! Properties for `core::compensated`.

use rust_physics_engine::core::compensated::{dot_compensated, sum_neumaier, sum_pairwise};
use rust_physics_engine::monte_carlo::Rng;

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
