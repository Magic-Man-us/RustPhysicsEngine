//! Probability distributions: densities, mass functions, and CDFs.

use crate::math::constants::PI;
use crate::special::erf::erfc;

use super::{factorial, gamma_lanczos};

/// Gaussian probability density function: f(x) = (1/(σ√(2π))) · exp(-½((x-μ)/σ)²)
pub fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0, "sigma must be positive");
    let z = (x - mu) / sigma;
    (1.0 / (sigma * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
}

/// Gaussian CDF: Φ(x) = ½·erfc(−(x−μ)/(σ√2)), full double precision.
pub fn gaussian_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0, "sigma must be positive");
    let z = (x - mu) / sigma;
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Deprecated alias of [`gaussian_cdf`]; the historical Abramowitz &
/// Stegun approximation has been replaced by the exact erfc form.
#[deprecated(note = "use gaussian_cdf; this alias now computes the exact value")]
pub fn gaussian_cdf_approx(x: f64, mu: f64, sigma: f64) -> f64 {
    gaussian_cdf(x, mu, sigma)
}

/// Poisson probability mass function: P(k;λ) = λᵏ · e⁻λ / k!
pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    lambda.powi(k as i32) * (-lambda).exp() / factorial(k)
}

/// Exponential probability density function: f(x;λ) = λ · e⁻ˡˣ for x ≥ 0
pub fn exponential_pdf(x: f64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    if x < 0.0 {
        return 0.0;
    }
    lambda * (-lambda * x).exp()
}

/// Exponential cumulative distribution function: F(x;λ) = 1 - e⁻ˡˣ for x ≥ 0
pub fn exponential_cdf(x: f64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    if x < 0.0 {
        return 0.0;
    }
    1.0 - (-lambda * x).exp()
}

/// Chi-squared PDF: f(x;k) = x^(k/2-1)·e^(-x/2) / (2^(k/2)·Γ(k/2))
pub fn chi_squared_pdf(x: f64, k: u32) -> f64 {
    assert!(k > 0, "degrees of freedom must be positive");
    if x <= 0.0 {
        return 0.0;
    }
    let half_k = k as f64 / 2.0;
    x.powf(half_k - 1.0) * (-x / 2.0).exp()
        / (2.0_f64.powf(half_k) * gamma_lanczos(half_k))
}
