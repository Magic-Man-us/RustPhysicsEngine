//! Resampling methods: bootstrap, BCa bootstrap, permutation tests,
//! and the jackknife.
//!
//! Reference: Efron & Tibshirani, *An Introduction to the Bootstrap*
//! (1993), ch. 6 (percentile), ch. 14 (BCa), ch. 15 (permutation).

use crate::monte_carlo::Rng;
use crate::special::erf::erfinv;
use crate::statistics::distributions::{Distribution, Normal};

/// Bootstrap summary: point estimate on the original data, bootstrap
/// standard error, and the confidence interval bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapResult {
    pub estimate: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

fn resample(data: &[f64], rng: &mut Rng, out: &mut Vec<f64>) {
    out.clear();
    for _ in 0..data.len() {
        let idx = (rng.next_u64() % data.len() as u64) as usize;
        out.push(data[idx]);
    }
}

fn bootstrap_replicates(
    data: &[f64],
    statistic: &dyn Fn(&[f64]) -> f64,
    n_resamples: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut buf = Vec::with_capacity(data.len());
    let mut thetas = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        resample(data, rng, &mut buf);
        thetas.push(statistic(&buf));
    }
    thetas
}

fn se_of(thetas: &[f64]) -> f64 {
    let n = thetas.len() as f64;
    let m: f64 = thetas.iter().sum::<f64>() / n;
    (thetas.iter().map(|t| (t - m) * (t - m)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Percentile bootstrap for an arbitrary statistic at the given
/// confidence level (e.g. 0.95).
///
/// # Panics
/// Panics unless data is non-empty, n_resamples ≥ 2, and
/// level ∈ (0, 1).
#[must_use]
pub fn bootstrap(
    data: &[f64],
    statistic: &dyn Fn(&[f64]) -> f64,
    n_resamples: usize,
    level: f64,
    rng: &mut Rng,
) -> BootstrapResult {
    assert!(!data.is_empty(), "bootstrap requires data");
    assert!(n_resamples >= 2, "bootstrap requires n_resamples >= 2");
    assert!((0.0..1.0).contains(&level) && level > 0.0, "level must be in (0, 1)");
    let estimate = statistic(data);
    let mut thetas = bootstrap_replicates(data, statistic, n_resamples, rng);
    let se = se_of(&thetas);
    thetas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = (1.0 - level) / 2.0;
    let lo_idx = ((alpha * n_resamples as f64) as usize).min(n_resamples - 1);
    let hi_idx = (((1.0 - alpha) * n_resamples as f64) as usize).min(n_resamples - 1);
    BootstrapResult { estimate, se, ci_low: thetas[lo_idx], ci_high: thetas[hi_idx] }
}

/// Bias-corrected and accelerated (BCa) bootstrap: percentile interval
/// with the bias correction z₀ from the replicate distribution and the
/// acceleration a from the jackknife influence values.
///
/// # Panics
/// Panics unless data has ≥ 2 points, n_resamples ≥ 2, and
/// level ∈ (0, 1).
#[must_use]
pub fn bootstrap_bca(
    data: &[f64],
    statistic: &dyn Fn(&[f64]) -> f64,
    n_resamples: usize,
    level: f64,
    rng: &mut Rng,
) -> BootstrapResult {
    assert!(data.len() >= 2, "bootstrap_bca requires at least 2 points");
    assert!(n_resamples >= 2, "bootstrap_bca requires n_resamples >= 2");
    assert!((0.0..1.0).contains(&level) && level > 0.0, "level must be in (0, 1)");
    let estimate = statistic(data);
    let mut thetas = bootstrap_replicates(data, statistic, n_resamples, rng);
    let se = se_of(&thetas);

    // Bias correction from the fraction of replicates below the estimate.
    let below = thetas.iter().filter(|&&t| t < estimate).count() as f64;
    let prop = (below / n_resamples as f64).clamp(1e-10, 1.0 - 1e-10);
    let z0 = std::f64::consts::SQRT_2 * erfinv(2.0 * prop - 1.0);

    // Acceleration from jackknife influence values.
    let n = data.len();
    let mut jack = Vec::with_capacity(n);
    let mut buf = Vec::with_capacity(n - 1);
    for i in 0..n {
        buf.clear();
        buf.extend(data.iter().take(i).chain(data.iter().skip(i + 1)));
        jack.push(statistic(&buf));
    }
    let jack_mean: f64 = jack.iter().sum::<f64>() / n as f64;
    let num: f64 = jack.iter().map(|&j| (jack_mean - j).powi(3)).sum();
    let den: f64 = jack.iter().map(|&j| (jack_mean - j).powi(2)).sum();
    let a = if den > 0.0 { num / (6.0 * den.powf(1.5)) } else { 0.0 };

    let normal = Normal::new(0.0, 1.0);
    let alpha = (1.0 - level) / 2.0;
    let z_alpha = normal.quantile(alpha);
    let z_1malpha = normal.quantile(1.0 - alpha);
    let adjusted = |z: f64| normal.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)));
    let a1 = adjusted(z_alpha).clamp(0.0, 1.0);
    let a2 = adjusted(z_1malpha).clamp(0.0, 1.0);

    thetas.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let idx = |q: f64| ((q * n_resamples as f64) as usize).min(n_resamples - 1);
    BootstrapResult { estimate, se, ci_low: thetas[idx(a1)], ci_high: thetas[idx(a2)] }
}

/// Two-sample permutation test. `statistic` maps (x, y) to a test
/// statistic (e.g. difference of means); the returned p-value is the
/// fraction of label permutations with |T*| ≥ |T| (with the +1
/// continuity correction).
///
/// # Panics
/// Panics unless both samples are non-empty and n_perm ≥ 1.
#[must_use]
pub fn permutation_test(
    x: &[f64],
    y: &[f64],
    statistic: &dyn Fn(&[f64], &[f64]) -> f64,
    n_perm: usize,
    rng: &mut Rng,
) -> f64 {
    assert!(!x.is_empty() && !y.is_empty(), "permutation_test requires data");
    assert!(n_perm >= 1, "permutation_test requires n_perm >= 1");
    let observed = statistic(x, y).abs();
    let mut pooled: Vec<f64> = x.iter().chain(y.iter()).copied().collect();
    let nx = x.len();
    let mut count = 0usize;
    for _ in 0..n_perm {
        // Fisher-Yates shuffle of the pooled sample.
        for i in (1..pooled.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            pooled.swap(i, j);
        }
        let t = statistic(&pooled[..nx], &pooled[nx..]).abs();
        if t >= observed {
            count += 1;
        }
    }
    (count as f64 + 1.0) / (n_perm as f64 + 1.0)
}

/// Jackknife estimate and standard error of a statistic:
/// SE² = (n−1)/n · Σ (θ₍ᵢ₎ − θ̄)².
///
/// # Panics
/// Panics unless data has at least 2 points.
#[must_use]
pub fn jackknife(data: &[f64], statistic: &dyn Fn(&[f64]) -> f64) -> (f64, f64) {
    assert!(data.len() >= 2, "jackknife requires at least 2 points");
    let n = data.len();
    let mut buf = Vec::with_capacity(n - 1);
    let mut thetas = Vec::with_capacity(n);
    for i in 0..n {
        buf.clear();
        buf.extend(data.iter().take(i).chain(data.iter().skip(i + 1)));
        thetas.push(statistic(&buf));
    }
    let theta_bar: f64 = thetas.iter().sum::<f64>() / n as f64;
    let var: f64 =
        (n as f64 - 1.0) / n as f64 * thetas.iter().map(|t| (t - theta_bar).powi(2)).sum::<f64>();
    (statistic(data), var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::descriptive::mean;

    #[test]
    fn test_bootstrap_mean_ci_covers_truth() {
        let mut rng = Rng::new(81);
        let data: Vec<f64> = (0..200).map(|_| rng.next_gaussian() + 5.0).collect();
        let r = bootstrap(&data, &mean, 2000, 0.95, &mut rng);
        assert!((r.estimate - 5.0).abs() < 0.3);
        assert!(r.ci_low < r.estimate && r.estimate < r.ci_high);
        assert!(r.ci_low < 5.0 && 5.0 < r.ci_high, "CI [{}, {}]", r.ci_low, r.ci_high);
        // Bootstrap SE of the mean approx sigma/sqrt(n) = 1/sqrt(200).
        assert!((r.se - 1.0 / (200.0_f64).sqrt()).abs() < 0.03, "se = {}", r.se);
    }

    #[test]
    fn test_bootstrap_bca_reasonable() {
        let mut rng = Rng::new(82);
        let data: Vec<f64> = (0..150).map(|_| rng.next_gaussian() * 2.0 + 1.0).collect();
        let r = bootstrap_bca(&data, &mean, 2000, 0.9, &mut rng);
        assert!(r.ci_low < r.estimate && r.estimate < r.ci_high);
        // For a symmetric statistic BCa should be near the percentile CI.
        let p = bootstrap(&data, &mean, 2000, 0.9, &mut rng);
        assert!((r.ci_low - p.ci_low).abs() < 0.2);
        assert!((r.ci_high - p.ci_high).abs() < 0.2);
    }

    #[test]
    fn test_permutation_test_detects_shift() {
        let mut rng = Rng::new(83);
        let x: Vec<f64> = (0..40).map(|_| rng.next_gaussian()).collect();
        let y_far: Vec<f64> = (0..40).map(|_| rng.next_gaussian() + 3.0).collect();
        let diff = |a: &[f64], b: &[f64]| mean(a) - mean(b);
        let p_far = permutation_test(&x, &y_far, &diff, 500, &mut rng);
        assert!(p_far < 0.01, "p = {p_far}");
        let y_same: Vec<f64> = (0..40).map(|_| rng.next_gaussian()).collect();
        let p_same = permutation_test(&x, &y_same, &diff, 500, &mut rng);
        assert!(p_same > 0.05, "p = {p_same}");
    }

    #[test]
    fn test_jackknife_mean_matches_formula() {
        // For the mean, jackknife SE equals the usual s/sqrt(n).
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (est, se) = jackknife(&data, &mean);
        assert!((est - 5.0).abs() < 1e-12);
        let s = crate::statistics::descriptive::sample_std_deviation(&data);
        assert!((se - s / (8.0_f64).sqrt()).abs() < 1e-10, "se = {se}");
    }
}
