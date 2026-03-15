use crate::math::constants::PI;

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

// Abramowitz & Stegun CDF approximation constants
const AS_B1: f64 = 0.436_183_6;
const AS_B2: f64 = -0.120_167_6;
const AS_B3: f64 = 0.937_298_0;
const AS_P: f64 = 0.332_67;

pub fn factorial(n: u64) -> f64 {
    (1..=n).fold(1.0, |acc, i| acc * i as f64)
}

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

// ---------------------------------------------------------------------------
// Descriptive statistics
// ---------------------------------------------------------------------------

pub fn mean(data: &[f64]) -> f64 {
    assert!(!data.is_empty(), "mean requires non-empty data");
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn variance(data: &[f64]) -> f64 {
    let mu = mean(data);
    data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / data.len() as f64
}

pub fn std_deviation(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

pub fn sample_variance(data: &[f64]) -> f64 {
    assert!(data.len() >= 2, "sample_variance requires at least 2 data points");
    let mu = mean(data);
    data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn sample_std_deviation(data: &[f64]) -> f64 {
    sample_variance(data).sqrt()
}

pub fn median(data: &mut [f64]) -> f64 {
    assert!(!data.is_empty(), "median requires non-empty data");
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    if n % 2 == 0 {
        (data[n / 2 - 1] + data[n / 2]) / 2.0
    } else {
        data[n / 2]
    }
}

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

pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let cov = covariance(x, y);
    let sx = std_deviation(x);
    let sy = std_deviation(y);
    assert!(sx > 0.0 && sy > 0.0, "correlation requires non-zero standard deviations");
    cov / (sx * sy)
}

// ---------------------------------------------------------------------------
// Probability distributions
// ---------------------------------------------------------------------------

pub fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0, "sigma must be positive");
    let z = (x - mu) / sigma;
    (1.0 / (sigma * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
}

pub fn gaussian_cdf_approx(x: f64, mu: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0, "sigma must be positive");
    let z = (x - mu) / sigma;
    if z < 0.0 {
        return 1.0 - gaussian_cdf_approx(mu - (x - mu), mu, sigma);
    }
    let phi_z = gaussian(x, mu, sigma);
    let t = 1.0 / (1.0 + AS_P * z);
    1.0 - phi_z * (AS_B1 * t + AS_B2 * t * t + AS_B3 * t * t * t)
}

pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    lambda.powi(k as i32) * (-lambda).exp() / factorial(k)
}

pub fn exponential_pdf(x: f64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    if x < 0.0 {
        return 0.0;
    }
    lambda * (-lambda * x).exp()
}

pub fn exponential_cdf(x: f64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    if x < 0.0 {
        return 0.0;
    }
    1.0 - (-lambda * x).exp()
}

pub fn chi_squared_pdf(x: f64, k: u32) -> f64 {
    assert!(k > 0, "degrees of freedom must be positive");
    if x <= 0.0 {
        return 0.0;
    }
    let half_k = k as f64 / 2.0;
    x.powf(half_k - 1.0) * (-x / 2.0).exp()
        / (2.0_f64.powf(half_k) * gamma_lanczos(half_k))
}

// ---------------------------------------------------------------------------
// Error propagation
// ---------------------------------------------------------------------------

pub fn error_propagation_sum(errors: &[f64]) -> f64 {
    errors.iter().map(|&e| e * e).sum::<f64>().sqrt()
}

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

pub fn weighted_mean_error(weights: &[f64]) -> f64 {
    assert!(!weights.is_empty(), "weights must be non-empty");
    let sum: f64 = weights.iter().sum();
    assert!(sum > 0.0, "sum of weights must be positive");
    1.0 / sum.sqrt()
}

// ---------------------------------------------------------------------------
// Fourier transform
// ---------------------------------------------------------------------------

pub fn dft(signal: &[f64]) -> Vec<(f64, f64)> {
    let n = signal.len();
    (0..n)
        .map(|k| {
            let mut real = 0.0;
            let mut imag = 0.0;
            for (idx, &sample) in signal.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * idx as f64 / n as f64;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            (real, imag)
        })
        .collect()
}

pub fn inverse_dft(spectrum: &[(f64, f64)]) -> Vec<f64> {
    let n = spectrum.len();
    let inv_n = 1.0 / n as f64;
    (0..n)
        .map(|idx| {
            let mut sum = 0.0;
            for (k, &(re, im)) in spectrum.iter().enumerate() {
                let angle = 2.0 * PI * k as f64 * idx as f64 / n as f64;
                sum += re * angle.cos() - im * angle.sin();
            }
            sum * inv_n
        })
        .collect()
}

pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    dft(signal)
        .iter()
        .map(|&(re, im)| re * re + im * im)
        .collect()
}

pub fn dominant_frequency(signal: &[f64], sample_rate: f64) -> f64 {
    let ps = power_spectrum(signal);
    let n = ps.len();
    // Only search up to Nyquist (first half)
    let half = n / 2;
    let k_max = (1..=half)
        .max_by(|&a, &b| ps[a].partial_cmp(&ps[b]).unwrap())
        .unwrap_or(0);
    k_max as f64 * sample_rate / n as f64
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
        let expected = 32.0 / 7.0; // 4.571...
        assert!(approx(sample_variance(&data), expected));
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
        assert!(approx_loose(gamma_lanczos(0.5), PI.sqrt()));
    }

    #[test]
    fn test_chi_squared_pdf_nonzero() {
        // χ²(x=2, k=2) = 0.5 × e^(-1) ≈ 0.1839
        let val = chi_squared_pdf(2.0, 2);
        assert!(approx_loose(val, 0.1839));
    }
}
