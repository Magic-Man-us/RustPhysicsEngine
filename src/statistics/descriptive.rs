//! Descriptive statistics, error propagation, and weighted means.

use crate::core::compensated::sum_neumaier;

/// Arithmetic mean of a data set: μ = (Σxᵢ) / n
/// The sum is computed with Neumaier compensated summation.
pub fn mean(data: &[f64]) -> f64 {
    assert!(!data.is_empty(), "mean requires non-empty data");
    sum_neumaier(data) / data.len() as f64
}

/// Population variance: σ² = Σ(xᵢ - μ)² / n
/// The sum of squared deviations is computed with Neumaier compensated summation.
pub fn variance(data: &[f64]) -> f64 {
    let mu = mean(data);
    let sq_dev: Vec<f64> = data.iter().map(|&x| (x - mu).powi(2)).collect();
    sum_neumaier(&sq_dev) / data.len() as f64
}

/// Population standard deviation: σ = sqrt(σ²)
pub fn std_deviation(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Sample variance with Bessel's correction: s² = Σ(xᵢ - x̄)² / (n - 1)
pub fn sample_variance(data: &[f64]) -> f64 {
    assert!(data.len() >= 2, "sample_variance requires at least 2 data points");
    let mu = mean(data);
    let sq_dev: Vec<f64> = data.iter().map(|&x| (x - mu).powi(2)).collect();
    sum_neumaier(&sq_dev) / (data.len() - 1) as f64
}

/// Sample standard deviation: s = sqrt(s²)
pub fn sample_std_deviation(data: &[f64]) -> f64 {
    sample_variance(data).sqrt()
}

/// Median of a data set (sorts the slice in place)
pub fn median(data: &mut [f64]) -> f64 {
    assert!(!data.is_empty(), "median requires non-empty data");
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = data.len();
    if n % 2 == 0 {
        (data[n / 2 - 1] + data[n / 2]) / 2.0
    } else {
        data[n / 2]
    }
}

/// Population covariance of two data sets: cov(X,Y) = Σ(xᵢ - μₓ)(yᵢ - μᵧ) / n
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "covariance requires equal-length slices");
    assert!(!x.is_empty(), "covariance requires non-empty data");
    let mu_x = mean(x);
    let mu_y = mean(y);
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mu_x) * (yi - mu_y))
        .sum::<f64>()
        / x.len() as f64
}

/// Pearson correlation coefficient: r = cov(X,Y) / (σₓ · σᵧ)
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let cov = covariance(x, y);
    let sx = std_deviation(x);
    let sy = std_deviation(y);
    assert!(sx > 0.0 && sy > 0.0, "correlation requires non-zero standard deviations");
    cov / (sx * sy)
}

/// Error propagation for sums: δ_total = sqrt(Σδᵢ²)
pub fn error_propagation_sum(errors: &[f64]) -> f64 {
    errors.iter().map(|&e| e * e).sum::<f64>().sqrt()
}

/// Error propagation for products using relative errors: δ_rel = sqrt(Σ(δᵢ/vᵢ)²)
pub fn error_propagation_product(values: &[f64], relative_errors: &[f64]) -> f64 {
    assert_eq!(
        values.len(),
        relative_errors.len(),
        "values and relative_errors must have equal length"
    );
    values
        .iter()
        .zip(relative_errors.iter())
        .map(|(&v, &e)| {
            assert!(v.abs() > 0.0, "values must be non-zero for product error propagation");
            (e / v).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

/// Weighted mean: x̄_w = Σ(wᵢ·xᵢ) / Σwᵢ
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(values.len(), weights.len(), "values and weights must have equal length");
    let total_weight: f64 = weights.iter().sum();
    assert!(total_weight > 0.0, "total weight must be positive");
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * v)
        .sum::<f64>()
        / total_weight
}

/// Weighted mean uncertainty: δ = 1 / sqrt(Σwᵢ)
pub fn weighted_mean_error(weights: &[f64]) -> f64 {
    assert!(!weights.is_empty(), "weights must be non-empty");
    let sum: f64 = weights.iter().sum();
    assert!(sum > 0.0, "sum of weights must be positive");
    1.0 / sum.sqrt()
}
