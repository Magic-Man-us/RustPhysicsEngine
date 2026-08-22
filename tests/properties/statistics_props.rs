//! Properties for `statistics::distributions`, inference, and
//! resampling.

use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::numerical::adaptive_quad;
use rust_physics_engine::statistics::{
    Beta, ChiSquared, Distribution, Exponential, FDist, Gamma, LogNormal, Normal, StudentT,
    Weibull,
};

fn continuous_distributions() -> Vec<(&'static str, Box<dyn Distribution>)> {
    vec![
        ("normal", Box::new(Normal::new(1.0, 2.0))),
        ("student_t", Box::new(StudentT::new(7.0))),
        ("chi_squared", Box::new(ChiSquared::new(4.0))),
        ("f", Box::new(FDist::new(5.0, 12.0))),
        ("gamma", Box::new(Gamma::new(2.5, 1.5))),
        ("beta", Box::new(Beta::new(2.0, 3.0))),
        ("exponential", Box::new(Exponential::new(0.7))),
        ("lognormal", Box::new(LogNormal::new(0.2, 0.5))),
        ("weibull", Box::new(Weibull::new(1.8, 1.2))),
    ]
}

/// cdf(quantile(p)) == p for p in (0.001, 0.999).
#[test]
fn prop_quantile_cdf_roundtrip() {
    let mut rng = Rng::new(71);
    for (name, d) in &continuous_distributions() {
        for _ in 0..40 {
            let p = 0.001 + rng.next_f64() * 0.998;
            let x = d.quantile(p);
            let back = d.cdf(x);
            assert!((back - p).abs() < 1e-8, "{name}: p={p}, back={back}");
        }
    }
}

/// The pdf integrates to 1 (over a quantile-bracketed support).
#[test]
fn prop_pdf_integrates_to_one() {
    for (name, d) in &continuous_distributions() {
        let lo = d.quantile(1e-9);
        let hi = d.quantile(1.0 - 1e-9);
        let f = |x: f64| d.pdf(x);
        let r = adaptive_quad(&f, lo, hi, 1e-10, 40).unwrap();
        assert!((r.value - 1.0).abs() < 1e-6, "{name}: integral {}", r.value);
    }
}

/// The mean of 1e5 draws lies within 4 standard errors of `mean()`.
#[test]
fn prop_sample_mean_matches() {
    let mut rng = Rng::new(72);
    let n = 100_000;
    for (name, d) in &continuous_distributions() {
        let sum: f64 = (0..n).map(|_| d.sample(&mut rng)).sum();
        let mean = sum / n as f64;
        let se = (d.variance() / n as f64).sqrt();
        assert!(
            (mean - d.mean()).abs() < 4.0 * se,
            "{name}: sample mean {mean} vs {} (se {se})",
            d.mean()
        );
    }
}

/// Under the null hypothesis, p-values are uniform: KS test on 1000
/// simulated one-sample t-test p-values against the Uniform CDF,
/// rejecting only below 1e-4.
#[test]
fn prop_p_values_uniform_under_null() {
    use rust_physics_engine::statistics::{ks_test_one_sample, t_test_one_sample};
    let mut rng = Rng::new(73);
    let p_values: Vec<f64> = (0..1000)
        .map(|_| {
            let sample: Vec<f64> = (0..20).map(|_| rng.next_gaussian()).collect();
            t_test_one_sample(&sample, 0.0).p_value
        })
        .collect();
    let ks = ks_test_one_sample(&p_values, &|v| v.clamp(0.0, 1.0));
    assert!(
        ks.p_value > 1e-4,
        "p-values not uniform under null: KS D = {}, p = {}",
        ks.statistic,
        ks.p_value
    );
}

/// Bootstrap percentile CI covers the true mean at roughly the nominal
/// rate (loose sanity band).
#[test]
fn prop_bootstrap_coverage() {
    use rust_physics_engine::statistics::bootstrap;
    let mut rng = Rng::new(74);
    let mean_stat = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;
    let mut covered = 0;
    let trials = 100;
    for _ in 0..trials {
        let data: Vec<f64> = (0..50).map(|_| rng.next_gaussian() * 2.0 + 1.0).collect();
        let r = bootstrap(&data, &mean_stat, 500, 0.9, &mut rng);
        if r.ci_low <= 1.0 && 1.0 <= r.ci_high {
            covered += 1;
        }
    }
    // Nominal 90%; accept a generous band for 100 trials.
    assert!((75..=100).contains(&covered), "coverage {covered}/100");
}
