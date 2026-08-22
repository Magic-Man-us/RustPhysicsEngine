//! Statistics: descriptive measures, probability distributions, and
//! Fourier utilities. Submodules are re-exported so historical paths such
//! as `crate::statistics::mean` keep working.

use crate::math::constants::PI;

pub mod descriptive;
pub mod distributions;
pub mod fourier;

pub use descriptive::{
    correlation, covariance, error_propagation_product, error_propagation_sum, mean, median,
    sample_std_deviation, sample_variance, std_deviation, variance, weighted_mean,
    weighted_mean_error,
};
pub use distributions::{
    chi_squared_pdf, exponential_cdf, exponential_pdf, gaussian, gaussian_cdf_approx, poisson_pmf,
};
pub use fourier::{dft, dominant_frequency, inverse_dft, power_spectrum};

// Lanczos approximation coefficients (g=7, n=9)
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Compute factorial of n: n! = 1 × 2 × ... × n
pub fn factorial(n: u64) -> f64 {
    (1..=n).fold(1.0, |acc, i| acc * i as f64)
}

/// Compute the gamma function via Lanczos approximation: Γ(z)
pub fn gamma_lanczos(z: f64) -> f64 {
    if z < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) × Γ(1-z))
        return PI / ((PI * z).sin() * gamma_lanczos(1.0 - z));
    }

    let z = z - 1.0;
    let mut x = LANCZOS_COEFFICIENTS[0];
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        x += coeff / (z + i as f64);
    }

    let t = z + LANCZOS_G + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;
    const LOOSE_EPSILON: f64 = 1e-4;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_loose(a: f64, b: f64) -> bool {
        (a - b).abs() < LOOSE_EPSILON
    }

    #[test]
    fn test_mean_variance_std() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(approx(mean(&data), 5.0));
        assert!(approx(variance(&data), 4.0));
        assert!(approx(std_deviation(&data), 2.0));
    }

    #[test]
    fn test_sample_variance() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(approx(sample_variance(&data), 4.571_428_571_428_571));
    }

    #[test]
    fn test_median_odd() {
        let mut data = [3.0, 1.0, 2.0];
        assert!(approx(median(&mut data), 2.0));
    }

    #[test]
    fn test_median_even() {
        let mut data = [4.0, 1.0, 3.0, 2.0];
        assert!(approx(median(&mut data), 2.5));
    }

    #[test]
    fn test_correlation_perfect() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!(approx(correlation(&x, &y), 1.0));
    }

    #[test]
    fn test_gaussian_standard_normal_at_zero() {
        assert!(approx_loose(gaussian(0.0, 0.0, 1.0), 0.3989));
    }

    #[test]
    fn test_gaussian_cdf_symmetry() {
        let cdf_0 = gaussian_cdf_approx(0.0, 0.0, 1.0);
        assert!(approx_loose(cdf_0, 0.5));
    }

    #[test]
    fn test_gaussian_cdf_nonunit_sigma() {
        // Φ(mu + sigma) should be ~0.8413 for any sigma
        let cdf = gaussian_cdf_approx(10.0, 5.0, 5.0);
        assert!(approx_loose(cdf, 0.8413));
        // Φ(mu - sigma) should be ~0.1587
        let cdf_neg = gaussian_cdf_approx(0.0, 5.0, 5.0);
        assert!(approx_loose(cdf_neg, 0.1587));
    }

    #[test]
    fn test_poisson() {
        // P(k=3, λ=2) = 8 × e^(-2) / 6 ≈ 0.18045
        let p = poisson_pmf(3, 2.0);
        assert!(approx_loose(p, 0.1804));
    }

    #[test]
    fn test_exponential_cdf_at_mean() {
        // F(1/λ) = 1 - e^(-1) ≈ 0.6321
        let cdf = exponential_cdf(1.0, 1.0);
        assert!(approx_loose(cdf, 0.6321));
    }

    #[test]
    fn test_error_propagation_sum() {
        let errors = [3.0, 4.0];
        assert!(approx(error_propagation_sum(&errors), 5.0));
    }

    #[test]
    fn test_error_propagation_product() {
        let values = [10.0, 20.0];
        let errors = [1.0, 2.0];
        // relative errors: 0.1, 0.1 → sqrt(0.01 + 0.01) = sqrt(0.02)
        let result = error_propagation_product(&values, &errors);
        assert!(approx(result, 0.02_f64.sqrt()));
    }

    #[test]
    fn test_weighted_mean() {
        let values = [10.0, 20.0, 30.0];
        let weights = [1.0, 1.0, 1.0];
        assert!(approx(weighted_mean(&values, &weights), 20.0));
    }

    #[test]
    fn test_dft_sine_peak() {
        // Pure sine at bin 3 in a 32-sample signal
        const N: usize = 32;
        const TARGET_BIN: usize = 3;
        let signal: Vec<f64> = (0..N)
            .map(|n| (2.0 * PI * TARGET_BIN as f64 * n as f64 / N as f64).sin())
            .collect();

        let ps = power_spectrum(&signal);
        let half = N / 2;
        let peak_bin = (1..=half)
            .max_by(|&a, &b| ps[a].partial_cmp(&ps[b]).unwrap())
            .unwrap();

        assert_eq!(peak_bin, TARGET_BIN);
    }

    #[test]
    fn test_dft_inverse_roundtrip() {
        let signal = vec![1.0, 0.0, -1.0, 0.0, 0.5, -0.5, 0.25, -0.25];
        let spectrum = dft(&signal);
        let recovered = inverse_dft(&spectrum);

        for (original, rec) in signal.iter().zip(recovered.iter()) {
            assert!(
                approx(*original, *rec),
                "roundtrip failed: {original} vs {rec}"
            );
        }
    }

    #[test]
    fn test_dominant_frequency() {
        const SAMPLE_RATE: f64 = 100.0;
        const FREQ: f64 = 10.0;
        const N: usize = 100;
        let signal: Vec<f64> = (0..N)
            .map(|n| (2.0 * PI * FREQ * n as f64 / SAMPLE_RATE).sin())
            .collect();

        let dom = dominant_frequency(&signal, SAMPLE_RATE);
        assert!(approx(dom, FREQ));
    }

    #[test]
    fn test_factorial() {
        assert!(approx(factorial(0), 1.0));
        assert!(approx(factorial(5), 120.0));
        assert!(approx(factorial(10), 3_628_800.0));
    }

    #[test]
    fn test_gamma_lanczos() {
        // Γ(5) = 4! = 24
        assert!(approx_loose(gamma_lanczos(5.0), 24.0));
        // Γ(0.5) = √π
        assert!(approx_loose(gamma_lanczos(0.5), 1.772_453_850_905_516));
    }

    #[test]
    fn test_chi_squared_pdf_nonzero() {
        // χ²(x=2, k=2) = 0.5 × e^(-1) ≈ 0.1839
        let val = chi_squared_pdf(2.0, 2);
        assert!(approx_loose(val, 0.1839));
    }

    #[test]
    fn test_covariance_identical() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        // cov(X, X) = var(X)
        let cov = covariance(&x, &x);
        let var = variance(&x);
        assert!(approx(cov, var), "cov(X,X)={cov} should equal var(X)={var}");
    }

    #[test]
    fn test_covariance_uncorrelated() {
        // x = [1, -1, 1, -1], y = [1, 1, -1, -1]
        let x = [1.0, -1.0, 1.0, -1.0];
        let y = [1.0, 1.0, -1.0, -1.0];
        let cov = covariance(&x, &y);
        assert!(approx(cov, 0.0), "uncorrelated data should have cov=0, got {cov}");
    }

    #[test]
    fn test_exponential_pdf_at_zero() {
        // f(0; λ) = λ
        let lambda = 3.0;
        let val = exponential_pdf(0.0, lambda);
        assert!(approx(val, lambda), "f(0)={val}, expected {lambda}");
    }

    #[test]
    fn test_exponential_pdf_negative_x() {
        let val = exponential_pdf(-1.0, 2.0);
        assert!(approx(val, 0.0), "f(x<0) should be 0, got {val}");
    }

    #[test]
    fn test_sample_std_deviation() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = sample_std_deviation(&data);
        assert!(approx(s, 2.138_089_935_299_395), "s={s}");
    }

    #[test]
    fn test_weighted_mean_error() {
        let weights = [4.0, 1.0];
        // δ = 1/√(4+1) = 1/√5
        let err = weighted_mean_error(&weights);
        assert!(approx(err, 0.447_213_595_499_958), "err={err}");
    }

    #[test]
    fn test_gamma_lanczos_negative_half() {
        let g = gamma_lanczos(0.25);
        assert!((g - 3.625_609_908_221_908).abs() < 1e-6, "gamma(0.25)={g}");
    }

    #[test]
    fn test_exponential_pdf_negative_x_returns_zero() {
        let p = exponential_pdf(-1.0, 1.0);
        assert!(approx(p, 0.0));
    }

    #[test]
    fn test_chi_squared_pdf_zero_x() {
        let p = chi_squared_pdf(0.0, 2);
        assert!(approx(p, 0.0));
    }

    #[test]
    fn test_exponential_cdf_negative_x() {
        let cdf = exponential_cdf(-1.0, 2.0);
        assert!(approx(cdf, 0.0));
    }
}
