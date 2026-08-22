//! Probability distributions: densities, mass functions, and CDFs.

use crate::math::constants::PI;

use super::{factorial, gamma_lanczos};

// Abramowitz & Stegun CDF approximation constants
const AS_B1: f64 = 0.436_183_6;
const AS_B2: f64 = -0.120_167_6;
const AS_B3: f64 = 0.937_298_0;
const AS_P: f64 = 0.332_67;

/// Gaussian probability density function: f(x) = (1/(σ√(2π))) · exp(-½((x-μ)/σ)²)
pub fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0, "sigma must be positive");
    let z = (x - mu) / sigma;
    (1.0 / (sigma * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
}

/// Approximate Gaussian CDF using the Abramowitz & Stegun method: Φ(x) ≈ 1 - φ(x)·P(t)
pub fn gaussian_cdf_approx(x: f64, mu: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0, "sigma must be positive");
    let z = (x - mu) / sigma;
    if z < 0.0 {
        return 1.0 - gaussian_cdf_approx(mu - (x - mu), mu, sigma);
    }
    let phi_z = (-0.5 * z * z).exp() / (2.0 * PI).sqrt();
    let t = 1.0 / (1.0 + AS_P * z);
    1.0 - phi_z * (AS_B1 * t + AS_B2 * t * t + AS_B3 * t * t * t)
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
