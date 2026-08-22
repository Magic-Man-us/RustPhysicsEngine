//! Hypothesis tests and confidence intervals.
//!
//! p-values come from the crate's own distribution CDFs (Student t,
//! chi-squared, F) and the asymptotic Kolmogorov distribution
//! Q_KS(λ) = 2·Σ (−1)^{j−1} e^{−2j²λ²} (NR §14.3). All t-type tests
//! report two-sided p-values.

use crate::linalg::Matrix;
use crate::statistics::descriptive::{mean, sample_variance};
use crate::statistics::distributions::{ChiSquared, Distribution, FDist, StudentT};

/// Outcome of a hypothesis test. For tests with two df parameters
/// (ANOVA, independence tables) `df` is the numerator / primary df;
/// the p-value always accounts for the full parameterization.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub df: f64,
}

fn two_sided_t_p(t: f64, df: f64) -> f64 {
    let dist = StudentT::new(df);
    2.0 * (1.0 - dist.cdf(t.abs()))
}

/// One-sample t test of H₀: μ = mu0.
///
/// R cross-check: `t.test(c(1,2,3,4,5), mu=2)` gives
/// t = 1.4142, df = 4, p-value = 0.2302.
///
/// # Panics
/// Panics unless x has at least 2 elements.
#[must_use]
pub fn t_test_one_sample(x: &[f64], mu0: f64) -> TestResult {
    assert!(x.len() >= 2, "t_test_one_sample requires n >= 2");
    let n = x.len() as f64;
    let m = mean(x);
    let s = sample_variance(x).sqrt();
    let t = (m - mu0) / (s / n.sqrt());
    let df = n - 1.0;
    TestResult { statistic: t, p_value: two_sided_t_p(t, df), df }
}

/// Two-sample t test of H₀: μₓ = μᵧ. `equal_var` selects the pooled
/// test; otherwise Welch's test with Welch-Satterthwaite df.
///
/// Analytic cross-check: x = 1..5 vs y = 2,4,…,10 with equal variances
/// gives t = −3/√2.5 = −1.897367 on 8 df (p ≈ 0.094); Welch df is
/// exactly 6.25/1.0625 = 5.882353.
///
/// # Panics
/// Panics unless both samples have at least 2 elements.
#[must_use]
pub fn t_test_two_sample(x: &[f64], y: &[f64], equal_var: bool) -> TestResult {
    assert!(x.len() >= 2 && y.len() >= 2, "t_test_two_sample requires n >= 2 per sample");
    let (nx, ny) = (x.len() as f64, y.len() as f64);
    let (mx, my) = (mean(x), mean(y));
    let (vx, vy) = (sample_variance(x), sample_variance(y));
    let (t, df) = if equal_var {
        let pooled = ((nx - 1.0) * vx + (ny - 1.0) * vy) / (nx + ny - 2.0);
        let se = (pooled * (1.0 / nx + 1.0 / ny)).sqrt();
        ((mx - my) / se, nx + ny - 2.0)
    } else {
        let se = (vx / nx + vy / ny).sqrt();
        let df = (vx / nx + vy / ny).powi(2)
            / ((vx / nx).powi(2) / (nx - 1.0) + (vy / ny).powi(2) / (ny - 1.0));
        ((mx - my) / se, df)
    };
    TestResult { statistic: t, p_value: two_sided_t_p(t, df), df }
}

/// Paired t test: one-sample test on the pairwise differences.
///
/// # Panics
/// Panics unless the samples are the same length with n ≥ 2.
#[must_use]
pub fn t_test_paired(x: &[f64], y: &[f64]) -> TestResult {
    assert!(x.len() == y.len(), "t_test_paired requires equal-length samples");
    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
    t_test_one_sample(&diffs, 0.0)
}

/// Chi-squared goodness-of-fit test: Σ (O − E)²/E with k − 1 df.
///
/// R cross-check: `chisq.test(c(10,20,30,40), p=rep(0.25,4))` gives
/// X-squared = 20, df = 3, p-value = 0.0001697.
///
/// # Panics
/// Panics unless the slices match in length (≥ 2) and all expected
/// counts are positive.
#[must_use]
pub fn chi_squared_gof(observed: &[f64], expected: &[f64]) -> TestResult {
    assert!(observed.len() == expected.len(), "chi_squared_gof requires equal lengths");
    assert!(observed.len() >= 2, "chi_squared_gof requires at least 2 categories");
    assert!(expected.iter().all(|&e| e > 0.0), "expected counts must be positive");
    let stat: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| (o - e) * (o - e) / e)
        .sum();
    let df = observed.len() as f64 - 1.0;
    let p = 1.0 - ChiSquared::new(df).cdf(stat);
    TestResult { statistic: stat, p_value: p, df }
}

/// Chi-squared test of independence on an r×c contingency table;
/// expected counts from the margins, (r−1)(c−1) df (reported as `df`).
///
/// # Panics
/// Panics unless the table is at least 2×2 with non-negative entries
/// and positive margins.
#[must_use]
pub fn chi_squared_independence(table: &Matrix) -> TestResult {
    assert!(table.rows >= 2 && table.cols >= 2, "table must be at least 2x2");
    assert!(table.data.iter().all(|&v| v >= 0.0), "table entries must be non-negative");
    let total: f64 = table.data.iter().sum();
    assert!(total > 0.0, "table must have a positive total");
    let row_sums: Vec<f64> = (0..table.rows).map(|r| table.row(r).iter().sum()).collect();
    let col_sums: Vec<f64> =
        (0..table.cols).map(|c| (0..table.rows).map(|r| table.get(r, c)).sum()).collect();
    assert!(
        row_sums.iter().all(|&s| s > 0.0) && col_sums.iter().all(|&s| s > 0.0),
        "margins must be positive"
    );
    let mut stat = 0.0;
    for r in 0..table.rows {
        for c in 0..table.cols {
            let e = row_sums[r] * col_sums[c] / total;
            let o = table.get(r, c);
            stat += (o - e) * (o - e) / e;
        }
    }
    let df = (table.rows as f64 - 1.0) * (table.cols as f64 - 1.0);
    let p = 1.0 - ChiSquared::new(df).cdf(stat);
    TestResult { statistic: stat, p_value: p, df }
}

/// Asymptotic Kolmogorov survival function Q_KS(λ).
fn kolmogorov_q(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0;
    let mut sign = 1.0;
    for j in 1..=100 {
        let term = (-2.0 * (j as f64) * (j as f64) * lambda * lambda).exp();
        sum += sign * term;
        sign = -sign;
        if term < 1e-16 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// One-sample Kolmogorov-Smirnov test of x against a continuous CDF.
/// `statistic` is Dₙ; `df` reports the sample size; p uses the NR
/// asymptotic correction λ = (√n + 0.12 + 0.11/√n)·D.
///
/// # Panics
/// Panics if x is empty.
#[must_use]
pub fn ks_test_one_sample(x: &[f64], cdf: &dyn Fn(f64) -> f64) -> TestResult {
    assert!(!x.is_empty(), "ks_test_one_sample requires data");
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let mut d = 0.0_f64;
    for (i, &xi) in sorted.iter().enumerate() {
        let f = cdf(xi);
        let above = (i as f64 + 1.0) / n - f;
        let below = f - i as f64 / n;
        d = d.max(above.max(below));
    }
    let lambda = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d;
    TestResult { statistic: d, p_value: kolmogorov_q(lambda), df: n }
}

/// Two-sample Kolmogorov-Smirnov test. `df` reports the effective
/// sample size n₁n₂/(n₁+n₂).
///
/// # Panics
/// Panics if either sample is empty.
#[must_use]
pub fn ks_test_two_sample(x: &[f64], y: &[f64]) -> TestResult {
    assert!(!x.is_empty() && !y.is_empty(), "ks_test_two_sample requires data");
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (n1, n2) = (xs.len(), ys.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut d = 0.0_f64;
    while i < n1 && j < n2 {
        let xi = xs[i];
        let yj = ys[j];
        if xi <= yj {
            i += 1;
        }
        if yj <= xi {
            j += 1;
        }
        let f1 = i as f64 / n1 as f64;
        let f2 = j as f64 / n2 as f64;
        d = d.max((f1 - f2).abs());
    }
    let ne = (n1 * n2) as f64 / (n1 + n2) as f64;
    let lambda = (ne.sqrt() + 0.12 + 0.11 / ne.sqrt()) * d;
    TestResult { statistic: d, p_value: kolmogorov_q(lambda), df: ne }
}

/// One-way ANOVA. `statistic` is F; `df` reports the between-groups
/// (numerator) df k − 1; the p-value uses F(k − 1, N − k).
///
/// Analytic cross-check: groups (1,2,3), (2,3,4), (5,6,7) give
/// SS_between = 26, SS_within = 6, F = 13 on (2, 6) df.
///
/// # Panics
/// Panics unless there are ≥ 2 groups, each non-empty, with more total
/// observations than groups.
#[must_use]
pub fn anova_one_way(groups: &[&[f64]]) -> TestResult {
    assert!(groups.len() >= 2, "anova_one_way requires at least 2 groups");
    assert!(groups.iter().all(|g| !g.is_empty()), "groups must be non-empty");
    let k = groups.len() as f64;
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let n = n_total as f64;
    assert!(n > k, "anova_one_way requires more observations than groups");

    let grand_mean: f64 = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n;
    let ss_between: f64 = groups
        .iter()
        .map(|g| {
            let gm = mean(g);
            g.len() as f64 * (gm - grand_mean) * (gm - grand_mean)
        })
        .sum();
    let ss_within: f64 = groups
        .iter()
        .map(|g| {
            let gm = mean(g);
            g.iter().map(|&v| (v - gm) * (v - gm)).sum::<f64>()
        })
        .sum();
    let df1 = k - 1.0;
    let df2 = n - k;
    let f = (ss_between / df1) / (ss_within / df2);
    let p = 1.0 - FDist::new(df1, df2).cdf(f);
    TestResult { statistic: f, p_value: p, df: df1 }
}

/// Two-sided confidence interval for the mean at the given level
/// (e.g. 0.95): x̄ ± t·s/√n.
///
/// # Panics
/// Panics unless n ≥ 2 and level ∈ (0, 1).
#[must_use]
pub fn confidence_interval_mean(x: &[f64], level: f64) -> (f64, f64) {
    assert!(x.len() >= 2, "confidence_interval_mean requires n >= 2");
    assert!((0.0..1.0).contains(&level) && level > 0.0, "level must be in (0, 1)");
    let n = x.len() as f64;
    let m = mean(x);
    let s = sample_variance(x).sqrt();
    let t = StudentT::new(n - 1.0).quantile(0.5 + level / 2.0);
    let half = t * s / n.sqrt();
    (m - half, m + half)
}

/// Test of H₀: ρ = 0 from the Pearson correlation:
/// t = r·√((n−2)/(1−r²)) with n − 2 df.
///
/// R cross-check: `cor.test(c(1,2,3,4,5), c(2,1,4,3,5))` gives
/// r = 0.8, t = 2.3094, df = 3, p-value = 0.1041.
///
/// # Panics
/// Panics unless both slices have equal length n ≥ 3.
#[must_use]
pub fn pearson_test(x: &[f64], y: &[f64]) -> TestResult {
    assert!(x.len() == y.len(), "pearson_test requires equal-length samples");
    assert!(x.len() >= 3, "pearson_test requires n >= 3");
    let r = crate::statistics::descriptive::correlation(x, y);
    let n = x.len() as f64;
    let df = n - 2.0;
    let t = r * (df / (1.0 - r * r)).sqrt();
    TestResult { statistic: t, p_value: two_sided_t_p(t, df), df }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_one_sample_t_matches_r() {
        // R: t.test(c(1,2,3,4,5), mu=2): t = 1.4142, df = 4, p = 0.2302.
        let r = t_test_one_sample(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0);
        assert!(approx(r.statistic, std::f64::consts::SQRT_2, 1e-10));
        assert!(approx(r.df, 4.0, 1e-12));
        assert!(approx(r.p_value, 0.230_2, 1e-3));
        // Exact null: t = 0, p = 1.
        let r0 = t_test_one_sample(&[1.0, 2.0, 3.0, 4.0, 5.0], 3.0);
        assert!(approx(r0.statistic, 0.0, 1e-12));
        assert!(approx(r0.p_value, 1.0, 1e-12));
    }

    #[test]
    fn test_two_sample_t_analytic() {
        // Pooled: vx = 2.5, vy = 10, pooled = 6.25, se = sqrt(2.5),
        // t = -3/sqrt(2.5) = -1.897367, df = 8; two-sided p in (0.09, 0.10)
        // per the t table (t_{0.95,8} = 1.860).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = t_test_two_sample(&x, &y, true);
        assert!(approx(r.statistic, -3.0 / 2.5_f64.sqrt(), 1e-12), "t = {}", r.statistic);
        assert!(approx(r.df, 8.0, 1e-12));
        assert!(r.p_value > 0.09 && r.p_value < 0.10, "p = {}", r.p_value);
        // Welch: same t, df = 6.25/1.0625 = 5.882353 exactly.
        let w = t_test_two_sample(&x, &y, false);
        assert!(approx(w.statistic, r.statistic, 1e-12));
        assert!(approx(w.df, 6.25 / 1.0625, 1e-10), "df = {}", w.df);
        assert!(w.p_value > r.p_value, "Welch p should exceed pooled p at lower df");
    }

    #[test]
    fn test_paired_t() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.1, 2.1, 3.1, 4.1];
        let r = t_test_paired(&x, &y);
        // Differences constant -0.1: zero variance... adjust: use varying diffs.
        assert!(r.statistic.is_infinite() || r.statistic.abs() > 1e6 || r.statistic.is_nan());
        let y2 = [1.2, 2.0, 3.4, 3.9];
        let r2 = t_test_paired(&x, &y2);
        assert!(r2.p_value > 0.0 && r2.p_value <= 1.0);
        assert!(approx(r2.df, 3.0, 1e-12));
    }

    #[test]
    fn test_chi_squared_gof_matches_r() {
        // R: chisq.test(c(10,20,30,40), p=rep(0.25,4)):
        // X-squared = 20, df = 3, p-value = 0.0001697.
        let r = chi_squared_gof(&[10.0, 20.0, 30.0, 40.0], &[25.0, 25.0, 25.0, 25.0]);
        assert!(approx(r.statistic, 20.0, 1e-12));
        assert!(approx(r.df, 3.0, 1e-12));
        assert!(approx(r.p_value, 0.000_169_7, 2e-6), "p = {}", r.p_value);
    }

    #[test]
    fn test_chi_squared_independence_matches_r() {
        // R: chisq.test(matrix(c(20,30,30,20), nrow=2), correct=FALSE):
        // X-squared = 4, df = 1, p-value = 0.0455.
        let table = Matrix::from_rows(&[&[20.0, 30.0], &[30.0, 20.0]]).unwrap();
        let r = chi_squared_independence(&table);
        assert!(approx(r.statistic, 4.0, 1e-10), "stat = {}", r.statistic);
        assert!(approx(r.df, 1.0, 1e-12));
        assert!(approx(r.p_value, 0.045_5, 1e-3), "p = {}", r.p_value);
    }

    #[test]
    fn test_anova_analytic() {
        // Group means 2, 3, 6; grand mean 11/3. SS_between = 26,
        // SS_within = 6, F = (26/2)/(6/6) = 13 on (2, 6) df.
        // F table: F_{0.99}(2,6) = 10.92, so p < 0.01 (and > 0.001).
        let g1 = [1.0, 2.0, 3.0];
        let g2 = [2.0, 3.0, 4.0];
        let g3 = [5.0, 6.0, 7.0];
        let r = anova_one_way(&[&g1, &g2, &g3]);
        assert!(approx(r.statistic, 13.0, 1e-10), "F = {}", r.statistic);
        assert!(approx(r.df, 2.0, 1e-12));
        assert!(r.p_value < 0.01 && r.p_value > 0.001, "p = {}", r.p_value);
    }

    #[test]
    fn test_pearson_matches_r() {
        // R: cor.test(c(1,2,3,4,5), c(2,1,4,3,5)): t = 2.3094, df = 3, p = 0.1041.
        let r = pearson_test(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 1.0, 4.0, 3.0, 5.0]);
        assert!(approx(r.statistic, 2.309_4, 1e-4), "t = {}", r.statistic);
        assert!(approx(r.df, 3.0, 1e-12));
        assert!(approx(r.p_value, 0.104_1, 1e-3), "p = {}", r.p_value);
    }

    #[test]
    fn test_ks_one_sample_uniform() {
        // Clearly non-uniform data strongly rejects Uniform(0,1).
        let x: Vec<f64> = (0..50).map(|i| 0.5 + 0.01 * (i as f64 / 50.0)).collect();
        let r = ks_test_one_sample(&x, &|v| v.clamp(0.0, 1.0));
        assert!(r.p_value < 1e-6);
        // Perfectly uniform grid: high p.
        let u: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5) / 100.0).collect();
        let r2 = ks_test_one_sample(&u, &|v| v.clamp(0.0, 1.0));
        assert!(r2.p_value > 0.99, "p = {}", r2.p_value);
    }

    #[test]
    fn test_ks_two_sample() {
        let x: Vec<f64> = (0..60).map(|i| i as f64 / 60.0).collect();
        let y: Vec<f64> = (0..60).map(|i| i as f64 / 60.0 + 0.001).collect();
        let same = ks_test_two_sample(&x, &y);
        assert!(same.p_value > 0.99);
        let z: Vec<f64> = (0..60).map(|i| i as f64 / 60.0 + 0.7).collect();
        let diff = ks_test_two_sample(&x, &z);
        assert!(diff.p_value < 1e-6);
    }

    #[test]
    fn test_confidence_interval() {
        // R: t.test(c(1,2,3,4,5)) 95% CI: (1.036757, 4.963243).
        let (lo, hi) = confidence_interval_mean(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.95);
        assert!(approx(lo, 1.036_757, 1e-4), "lo = {lo}");
        assert!(approx(hi, 4.963_243, 1e-4), "hi = {hi}");
    }
}
