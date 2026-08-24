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

// ---------------------------------------------------------------------------
// Distribution trait and concrete distributions
// ---------------------------------------------------------------------------

use crate::monte_carlo::Rng;
use crate::numerical::roots::brent_root;
use crate::special::beta::beta_inc;
use crate::special::erf::erfinv;
use crate::special::gamma::{gamma_p, gamma_q, lgamma};

const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Common interface for probability distributions.
///
/// For discrete distributions `pdf` is the probability mass at the
/// nearest integer and `quantile` returns the smallest k with
/// F(k) ≥ p. Moments that do not exist return `f64::NAN`.
pub trait Distribution {
    /// Probability density (or mass) at x.
    fn pdf(&self, x: f64) -> f64;
    /// Cumulative distribution F(x) = P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;
    /// Inverse CDF: the x with F(x) = p, for p ∈ (0, 1).
    fn quantile(&self, p: f64) -> f64;
    /// Expected value.
    fn mean(&self) -> f64;
    /// Variance.
    fn variance(&self) -> f64;
    /// Draws one sample using the crate's deterministic `Rng`.
    fn sample(&self, rng: &mut Rng) -> f64;
}

/// Bracket-expanding Brent inversion of a CDF, for distributions with
/// no closed-form quantile.
fn quantile_by_root(
    cdf: &dyn Fn(f64) -> f64,
    p: f64,
    mut lo: f64,
    mut hi: f64,
) -> f64 {
    assert!((0.0..=1.0).contains(&p), "quantile requires p in [0, 1]");
    let g = |x: f64| cdf(x) - p;
    for _ in 0..200 {
        if g(lo) <= 0.0 {
            break;
        }
        lo -= (hi - lo).abs().max(1.0);
    }
    for _ in 0..200 {
        if g(hi) >= 0.0 {
            break;
        }
        hi += (hi - lo).abs().max(1.0);
    }
    brent_root(&g, lo, hi, 1e-13, 300).unwrap_or(f64::NAN)
}

/// Marsaglia-Tsang gamma variate with the given shape and rate 1.
fn sample_gamma_shape(shape: f64, rng: &mut Rng) -> f64 {
    if shape < 1.0 {
        // Boost to shape+1 and correct with U^{1/shape}.
        let u = rng.next_f64().max(1e-300);
        return sample_gamma_shape(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.next_gaussian();
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.next_f64().max(1e-300);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Normal distribution N(μ, σ²); quantile via `erfinv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}

impl Normal {
    /// # Panics
    /// Panics unless σ > 0.
    #[must_use]
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "Normal requires sigma > 0");
        Self { mu, sigma }
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        gaussian(x, self.mu, self.sigma)
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 * erfc(-(x - self.mu) / (self.sigma * SQRT_2))
    }
    fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..=1.0).contains(&p), "quantile requires p in [0, 1]");
        self.mu + self.sigma * SQRT_2 * erfinv(2.0 * p - 1.0)
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        self.mu + self.sigma * rng.next_gaussian()
    }
}

/// Student's t distribution with ν degrees of freedom; CDF via the
/// regularized incomplete beta function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT {
    pub nu: f64,
}

impl StudentT {
    /// # Panics
    /// Panics unless ν > 0.
    #[must_use]
    pub fn new(nu: f64) -> Self {
        assert!(nu > 0.0, "StudentT requires nu > 0");
        Self { nu }
    }
}

impl Distribution for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        let nu = self.nu;
        let ln_norm = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * (nu * PI).ln();
        (ln_norm - 0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln()).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        let nu = self.nu;
        let ib = beta_inc(nu / 2.0, 0.5, nu / (nu + x * x));
        if x >= 0.0 {
            1.0 - 0.5 * ib
        } else {
            0.5 * ib
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        quantile_by_root(&|x| self.cdf(x), p, -10.0, 10.0)
    }
    fn mean(&self) -> f64 {
        if self.nu > 1.0 {
            0.0
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.nu > 2.0 {
            self.nu / (self.nu - 2.0)
        } else {
            f64::NAN
        }
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        let z = rng.next_gaussian();
        let chi2 = 2.0 * sample_gamma_shape(self.nu / 2.0, rng);
        z / (chi2 / self.nu).sqrt()
    }
}

/// Chi-squared distribution with k degrees of freedom; CDF via P(k/2, x/2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    pub k: f64,
}

impl ChiSquared {
    /// # Panics
    /// Panics unless k > 0.
    #[must_use]
    pub fn new(k: f64) -> Self {
        assert!(k > 0.0, "ChiSquared requires k > 0");
        Self { k }
    }
}

impl Distribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let half_k = self.k / 2.0;
        ((half_k - 1.0) * x.ln() - x / 2.0 - half_k * 2.0_f64.ln() - lgamma(half_k)).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        gamma_p(self.k / 2.0, x / 2.0)
    }
    fn quantile(&self, p: f64) -> f64 {
        quantile_by_root(&|x| self.cdf(x), p, 1e-12, self.k + 10.0)
    }
    fn mean(&self) -> f64 {
        self.k
    }
    fn variance(&self) -> f64 {
        2.0 * self.k
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        2.0 * sample_gamma_shape(self.k / 2.0, rng)
    }
}

/// Fisher-Snedecor F distribution with (d1, d2) degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FDist {
    pub d1: f64,
    pub d2: f64,
}

impl FDist {
    /// # Panics
    /// Panics unless d1 > 0 and d2 > 0.
    #[must_use]
    pub fn new(d1: f64, d2: f64) -> Self {
        assert!(d1 > 0.0 && d2 > 0.0, "FDist requires d1, d2 > 0");
        Self { d1, d2 }
    }
}

impl Distribution for FDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        let half1 = d1 / 2.0;
        let half2 = d2 / 2.0;
        let ln_b = lgamma(half1) + lgamma(half2) - lgamma(half1 + half2);
        (half1 * (d1 / d2).ln() + (half1 - 1.0) * x.ln()
            - (half1 + half2) * (1.0 + d1 * x / d2).ln()
            - ln_b)
            .exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = self.d1 * x / (self.d1 * x + self.d2);
        beta_inc(self.d1 / 2.0, self.d2 / 2.0, z)
    }
    fn quantile(&self, p: f64) -> f64 {
        quantile_by_root(&|x| self.cdf(x), p, 1e-12, 100.0)
    }
    fn mean(&self) -> f64 {
        if self.d2 > 2.0 {
            self.d2 / (self.d2 - 2.0)
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        let (d1, d2) = (self.d1, self.d2);
        if d2 > 4.0 {
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
        } else {
            f64::NAN
        }
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        let x1 = 2.0 * sample_gamma_shape(self.d1 / 2.0, rng);
        let x2 = 2.0 * sample_gamma_shape(self.d2 / 2.0, rng);
        (x1 / self.d1) / (x2 / self.d2)
    }
}

/// Gamma distribution with shape α and rate β (mean α/β).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    pub shape: f64,
    pub rate: f64,
}

impl Gamma {
    /// # Panics
    /// Panics unless shape > 0 and rate > 0.
    #[must_use]
    pub fn new(shape: f64, rate: f64) -> Self {
        assert!(shape > 0.0 && rate > 0.0, "Gamma requires shape, rate > 0");
        Self { shape, rate }
    }
}

impl Distribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (a, b) = (self.shape, self.rate);
        (a * b.ln() + (a - 1.0) * x.ln() - b * x - lgamma(a)).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        gamma_p(self.shape, self.rate * x)
    }
    fn quantile(&self, p: f64) -> f64 {
        quantile_by_root(&|x| self.cdf(x), p, 1e-12, (self.shape + 10.0) / self.rate)
    }
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }
    fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        sample_gamma_shape(self.shape, rng) / self.rate
    }
}

/// Beta distribution on [0, 1] with shape parameters (a, b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    pub a: f64,
    pub b: f64,
}

impl Beta {
    /// # Panics
    /// Panics unless a > 0 and b > 0.
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a > 0.0 && b > 0.0, "Beta requires a, b > 0");
        Self { a, b }
    }
}

impl Distribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        if x == 0.0 || x == 1.0 {
            // Density endpoints: finite only for a,b >= 1.
            return if (x == 0.0 && self.a < 1.0) || (x == 1.0 && self.b < 1.0) {
                f64::INFINITY
            } else if (x == 0.0 && self.a > 1.0) || (x == 1.0 && self.b > 1.0) {
                0.0
            } else {
                let ln_b = lgamma(self.a) + lgamma(self.b) - lgamma(self.a + self.b);
                (-ln_b).exp()
            };
        }
        let ln_b = lgamma(self.a) + lgamma(self.b) - lgamma(self.a + self.b);
        ((self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln() - ln_b).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            beta_inc(self.a, self.b, x)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        quantile_by_root(&|x| self.cdf(x), p, 1e-12, 1.0 - 1e-12)
    }
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        let x = sample_gamma_shape(self.a, rng);
        let y = sample_gamma_shape(self.b, rng);
        x / (x + y)
    }
}

/// Exponential distribution with the given rate λ (mean 1/λ), wrapping
/// the module's free `exponential_pdf`/`exponential_cdf`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    pub rate: f64,
}

impl Exponential {
    /// # Panics
    /// Panics unless rate > 0.
    #[must_use]
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "Exponential requires rate > 0");
        Self { rate }
    }
}

impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        exponential_pdf(x, self.rate)
    }
    fn cdf(&self, x: f64) -> f64 {
        exponential_cdf(x, self.rate)
    }
    fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..1.0).contains(&p), "quantile requires p in [0, 1)");
        -(1.0 - p).ln() / self.rate
    }
    fn mean(&self) -> f64 {
        1.0 / self.rate
    }
    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        -(1.0 - rng.next_f64()).max(1e-300).ln() / self.rate
    }
}

/// Poisson distribution (discrete); `pdf` is the mass at round(x) and
/// `cdf` uses P(X ≤ k) = Q(k+1, λ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    /// # Panics
    /// Panics unless λ > 0.
    #[must_use]
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "Poisson requires lambda > 0");
        Self { lambda }
    }

    /// Probability mass P(X = k), computed in log space.
    #[must_use]
    pub fn pmf(&self, k: u64) -> f64 {
        (k as f64 * self.lambda.ln() - self.lambda - lgamma(k as f64 + 1.0)).exp()
    }
}

impl Distribution for Poisson {
    fn pdf(&self, x: f64) -> f64 {
        if x < -0.5 {
            return 0.0;
        }
        self.pmf(x.round().max(0.0) as u64)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let k = x.floor();
        gamma_q(k + 1.0, self.lambda)
    }
    fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..1.0).contains(&p), "quantile requires p in [0, 1)");
        let mut k = 0.0;
        while self.cdf(k) < p && k < 1e9 {
            k += 1.0;
        }
        k
    }
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        if self.lambda < 30.0 {
            // Knuth's product method.
            let l = (-self.lambda).exp();
            let mut k = 0.0;
            let mut p = 1.0;
            loop {
                p *= rng.next_f64();
                if p <= l {
                    return k;
                }
                k += 1.0;
            }
        }
        // Normal approximation with continuity correction for large lambda.
        (self.lambda + self.lambda.sqrt() * rng.next_gaussian()).round().max(0.0)
    }
}

/// Binomial distribution (discrete) with n trials and success
/// probability p; CDF via the regularized incomplete beta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}

impl Binomial {
    /// # Panics
    /// Panics unless 0 ≤ p ≤ 1 and n ≥ 1.
    #[must_use]
    pub fn new(n: u64, p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "Binomial requires p in [0, 1]");
        assert!(n >= 1, "Binomial requires n >= 1");
        Self { n, p }
    }

    /// Probability mass P(X = k), computed in log space.
    #[must_use]
    pub fn pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        if self.p == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        if self.p == 1.0 {
            return if k == self.n { 1.0 } else { 0.0 };
        }
        let n = self.n as f64;
        let kf = k as f64;
        let ln_choose = lgamma(n + 1.0) - lgamma(kf + 1.0) - lgamma(n - kf + 1.0);
        (ln_choose + kf * self.p.ln() + (n - kf) * (1.0 - self.p).ln()).exp()
    }
}

impl Distribution for Binomial {
    fn pdf(&self, x: f64) -> f64 {
        if x < -0.5 {
            return 0.0;
        }
        self.pmf(x.round().max(0.0) as u64)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let k = x.floor() as u64;
        if k >= self.n {
            return 1.0;
        }
        if self.p == 0.0 {
            return 1.0;
        }
        if self.p == 1.0 {
            return 0.0;
        }
        // P(X <= k) = I_{1-p}(n-k, k+1)
        beta_inc((self.n - k) as f64, k as f64 + 1.0, 1.0 - self.p)
    }
    fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..1.0).contains(&p), "quantile requires p in [0, 1)");
        let mut k = 0.0;
        while self.cdf(k) < p && k < self.n as f64 {
            k += 1.0;
        }
        k
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        // O(n) Bernoulli sum; adequate for the crate's simulation sizes.
        let mut count = 0u64;
        for _ in 0..self.n {
            if rng.next_f64() < self.p {
                count += 1;
            }
        }
        count as f64
    }
}

/// Log-normal distribution: ln X ~ N(μ, σ²).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl LogNormal {
    /// # Panics
    /// Panics unless σ > 0.
    #[must_use]
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "LogNormal requires sigma > 0");
        Self { mu, sigma }
    }
}

impl Distribution for LogNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (x * self.sigma * (2.0 * PI).sqrt())
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        0.5 * erfc(-(x.ln() - self.mu) / (self.sigma * SQRT_2))
    }
    fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..=1.0).contains(&p), "quantile requires p in [0, 1]");
        (self.mu + self.sigma * SQRT_2 * erfinv(2.0 * p - 1.0)).exp()
    }
    fn mean(&self) -> f64 {
        (self.mu + 0.5 * self.sigma * self.sigma).exp()
    }
    fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (s2.exp() - 1.0) * (2.0 * self.mu + s2).exp()
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        (self.mu + self.sigma * rng.next_gaussian()).exp()
    }
}

/// Weibull distribution with shape k and scale λ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull {
    pub k: f64,
    pub lambda: f64,
}

impl Weibull {
    /// # Panics
    /// Panics unless k > 0 and λ > 0.
    #[must_use]
    pub fn new(k: f64, lambda: f64) -> Self {
        assert!(k > 0.0 && lambda > 0.0, "Weibull requires k, lambda > 0");
        Self { k, lambda }
    }
}

impl Distribution for Weibull {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let z = x / self.lambda;
        (self.k / self.lambda) * z.powf(self.k - 1.0) * (-z.powf(self.k)).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        1.0 - (-(x / self.lambda).powf(self.k)).exp()
    }
    fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..1.0).contains(&p), "quantile requires p in [0, 1)");
        self.lambda * (-(1.0 - p).ln()).powf(1.0 / self.k)
    }
    fn mean(&self) -> f64 {
        self.lambda * crate::special::gamma::gamma(1.0 + 1.0 / self.k)
    }
    fn variance(&self) -> f64 {
        let g1 = crate::special::gamma::gamma(1.0 + 1.0 / self.k);
        let g2 = crate::special::gamma::gamma(1.0 + 2.0 / self.k);
        self.lambda * self.lambda * (g2 - g1 * g1)
    }
    fn sample(&self, rng: &mut Rng) -> f64 {
        self.lambda * (-(1.0 - rng.next_f64()).max(1e-300).ln()).powf(1.0 / self.k)
    }
}

#[cfg(test)]
mod distribution_tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_normal_quantile_cdf_roundtrip() {
        let d = Normal::new(2.0, 3.0);
        for &p in &[0.01, 0.25, 0.5, 0.9, 0.999] {
            assert!(approx(d.cdf(d.quantile(p)), p, 1e-12), "p={p}");
        }
        assert!(approx(d.quantile(0.5), 2.0, 1e-12));
    }

    #[test]
    fn test_student_t_matches_normal_large_nu() {
        let t = StudentT::new(1e6);
        let n = Normal::new(0.0, 1.0);
        for &x in &[-2.0, -0.5, 0.0, 1.0, 2.5] {
            assert!(approx(t.cdf(x), n.cdf(x), 1e-4), "x={x}");
            assert!(approx(t.pdf(x), n.pdf(x), 1e-4));
        }
    }

    #[test]
    fn test_student_t_known_value() {
        // t distribution with nu=10: P(T <= 1.812461) ≈ 0.95 (t-table).
        let t = StudentT::new(10.0);
        assert!(approx(t.cdf(1.812_461_122_811_676), 0.95, 1e-6));
        assert!(approx(t.quantile(0.95), 1.812_461_122_811_676, 1e-5));
    }

    #[test]
    fn test_chi_squared_consistency_with_free_pdf() {
        let d = ChiSquared::new(4.0);
        for &x in &[0.5, 2.0, 7.0] {
            assert!(approx(d.pdf(x), chi_squared_pdf(x, 4), 1e-12), "x={x}");
        }
        assert!(approx(d.cdf(d.quantile(0.3)), 0.3, 1e-10));
    }

    #[test]
    fn test_f_distribution_known() {
        // F(2, 10): P(F <= 4.102821) ≈ 0.95.
        let d = FDist::new(2.0, 10.0);
        assert!(approx(d.cdf(4.102_821), 0.95, 1e-4));
        assert!(approx(d.quantile(0.95), 4.102_821, 1e-3));
    }

    #[test]
    fn test_gamma_exponential_special_case() {
        // Gamma(1, rate) is Exponential(rate).
        let g = Gamma::new(1.0, 2.0);
        let e = Exponential::new(2.0);
        for &x in &[0.1, 0.5, 2.0] {
            assert!(approx(g.pdf(x), e.pdf(x), 1e-12));
            assert!(approx(g.cdf(x), e.cdf(x), 1e-12));
        }
    }

    #[test]
    fn test_beta_uniform_special_case() {
        let b = Beta::new(1.0, 1.0);
        for &x in &[0.2, 0.5, 0.9] {
            assert!(approx(b.pdf(x), 1.0, 1e-12));
            assert!(approx(b.cdf(x), x, 1e-12));
        }
        assert!(approx(b.mean(), 0.5, 1e-15));
    }

    #[test]
    fn test_poisson_pmf_cdf_consistency() {
        let d = Poisson::new(3.0);
        assert!(approx(d.pmf(3), poisson_pmf(3, 3.0), 1e-12));
        let sum: f64 = (0..=7).map(|k| d.pmf(k)).sum();
        assert!(approx(sum, d.cdf(7.0), 1e-10));
        assert_eq!(d.quantile(d.cdf(4.0)), 4.0);
    }

    #[test]
    fn test_gaussian_cdf_approx_matches_the_exact_erfc_form() {
        // The alias is documented as now computing the exact value, so
        // it must agree with `gaussian_cdf` bit for bit; it is also
        // strictly monotone and hits the standard reference points.
        #[allow(deprecated)]
        let approx_cdf = |x: f64, mu: f64, sigma: f64| gaussian_cdf_approx(x, mu, sigma);
        for &(mu, sigma) in &[(0.0, 1.0), (2.5, 0.5), (-1.0, 3.0)] {
            let mut prev = f64::NEG_INFINITY;
            for k in -80..=80 {
                let x = mu + sigma * k as f64 / 10.0;
                let v = approx_cdf(x, mu, sigma);
                assert_eq!(v, gaussian_cdf(x, mu, sigma), "exact alias at x={x}");
                assert!(v > prev, "monotone at x={x}: {v} <= {prev}");
                assert!((0.0..=1.0).contains(&v), "CDF in [0, 1] at x={x}");
                prev = v;
            }
            // Symmetry about the mean and the classic 1σ/2σ/3σ masses.
            assert!(approx(approx_cdf(mu, mu, sigma), 0.5, 1e-15));
            for k in 1..=3 {
                let hi = approx_cdf(mu + k as f64 * sigma, mu, sigma);
                let lo = approx_cdf(mu - k as f64 * sigma, mu, sigma);
                assert!(approx(hi + lo, 1.0, 1e-14), "symmetry at {k} sigma");
            }
            assert!(approx(
                approx_cdf(mu + sigma, mu, sigma) - approx_cdf(mu - sigma, mu, sigma),
                0.682_689_492_137_086,
                1e-4
            ));
            assert!(approx(
                approx_cdf(mu + 2.0 * sigma, mu, sigma)
                    - approx_cdf(mu - 2.0 * sigma, mu, sigma),
                0.954_499_736_103_642,
                1e-4
            ));
            assert!(approx(
                approx_cdf(mu + 3.0 * sigma, mu, sigma)
                    - approx_cdf(mu - 3.0 * sigma, mu, sigma),
                0.997_300_203_936_740,
                1e-4
            ));
        }
        // Standard-normal table values.
        assert!(approx(approx_cdf(1.96, 0.0, 1.0), 0.975_002_104_851_780, 1e-4));
        assert!(approx(approx_cdf(-1.644_853_626_951_472, 0.0, 1.0), 0.05, 1e-4));
    }

    #[test]
    fn test_poisson_distribution_trait_impl() {
        for &lambda in &[0.5_f64, 3.0, 12.0] {
            let d = Poisson::new(lambda);
            // The trait mean/variance are both λ.
            assert_eq!(d.mean(), lambda);
            assert_eq!(d.variance(), lambda);
            // pdf is the mass at the nearest integer: it agrees with
            // pmf and is invariant to sub-half perturbations of x.
            let kmax = (lambda + 12.0 * lambda.sqrt()).ceil() as u64 + 40;
            let mut total = 0.0;
            for k in 0..=kmax {
                let m = d.pmf(k);
                assert!(approx(d.pdf(k as f64), m, 1e-15), "pdf({k}) vs pmf");
                assert!(approx(d.pdf(k as f64 + 0.3), m, 1e-15), "rounds down");
                assert!(approx(d.pdf(k as f64 - 0.3), m, 1e-15), "rounds up");
                total += m;
            }
            // The mass function sums to 1 over its whole support.
            assert!(approx(total, 1.0, 1e-12), "lambda={lambda} total {total}");
            // Negative arguments carry no mass.
            assert_eq!(d.pdf(-1.0), 0.0);
            assert_eq!(d.pdf(-7.5), 0.0);
            // Mean and variance recovered from the mass function.
            let mean: f64 = (0..=kmax).map(|k| k as f64 * d.pmf(k)).sum();
            let var: f64 =
                (0..=kmax).map(|k| (k as f64 - lambda).powi(2) * d.pmf(k)).sum();
            assert!(approx(mean, d.mean(), 1e-9), "E[X] {mean} vs {lambda}");
            assert!(approx(var, d.variance(), 1e-9), "Var[X] {var} vs {lambda}");
            // Recurrence P(k+1) = P(k)·λ/(k+1).
            for k in 0..20u64 {
                assert!(approx(
                    d.pdf(k as f64 + 1.0),
                    d.pdf(k as f64) * lambda / (k as f64 + 1.0),
                    1e-14
                ));
            }
            // quantile inverts the CDF on the lattice.
            for k in 0..8u64 {
                assert_eq!(d.quantile(d.cdf(k as f64)), k as f64, "quantile roundtrip");
            }
        }
    }

    #[test]
    fn test_binomial_distribution_trait_impl() {
        for &(n, p) in &[(10u64, 0.3_f64), (25, 0.5), (7, 0.85)] {
            let d = Binomial::new(n, p);
            // pdf is the mass at the nearest integer and sums to 1.
            let mut total = 0.0;
            for k in 0..=n {
                let m = d.pmf(k);
                assert!(approx(d.pdf(k as f64), m, 1e-15), "pdf({k})");
                assert!(approx(d.pdf(k as f64 + 0.4), m, 1e-15));
                assert!(approx(d.pdf(k as f64 - 0.4), m, 1e-15));
                total += m;
            }
            assert!(approx(total, 1.0, 1e-12), "n={n} p={p} total {total}");
            // No mass outside 0..=n.
            assert_eq!(d.pdf(-1.0), 0.0);
            assert_eq!(d.pdf(n as f64 + 1.0), 0.0);
            // Moments from the mass function match the closed forms.
            let mean: f64 = (0..=n).map(|k| k as f64 * d.pmf(k)).sum();
            let var: f64 =
                (0..=n).map(|k| (k as f64 - d.mean()).powi(2) * d.pmf(k)).sum();
            assert!(approx(mean, n as f64 * p, 1e-12));
            assert!(approx(var, n as f64 * p * (1.0 - p), 1e-12));
            // quantile(cdf(k)) == k on the lattice, for every k whose
            // mass is resolvable (F is strictly increasing there).
            for k in 0..n {
                let c = d.cdf(k as f64);
                if c < 1.0 && d.pmf(k) > 1e-12 {
                    assert_eq!(d.quantile(c), k as f64, "n={n} p={p} k={k}");
                }
            }
            // quantile is the smallest k with F(k) >= q, hence monotone
            // and bracketed by the CDF.
            for q in [0.01, 0.25, 0.5, 0.75, 0.99] {
                let k = d.quantile(q);
                assert!(d.cdf(k) >= q - 1e-12, "F(quantile) >= q");
                if k > 0.0 {
                    assert!(d.cdf(k - 1.0) < q, "quantile is minimal");
                }
            }
        }
        // Degenerate success probabilities are point masses.
        let sure = Binomial::new(5, 1.0);
        assert_eq!(sure.pdf(5.0), 1.0);
        assert_eq!(sure.pdf(4.0), 0.0);
        let never = Binomial::new(5, 0.0);
        assert_eq!(never.pdf(0.0), 1.0);
        assert_eq!(never.pdf(1.0), 0.0);
    }

    #[test]
    fn test_binomial_pmf_and_cdf() {
        let d = Binomial::new(10, 0.3);
        // P(X = 3) for B(10, 0.3) = 0.266827932
        assert!(approx(d.pmf(3), 0.266_827_932, 1e-8));
        let sum: f64 = (0..=4).map(|k| d.pmf(k)).sum();
        assert!(approx(sum, d.cdf(4.0), 1e-10));
        assert!(approx(d.mean(), 3.0, 1e-12));
        assert!(approx(d.variance(), 2.1, 1e-12));
    }

    #[test]
    fn test_lognormal_and_weibull_quantiles() {
        let ln = LogNormal::new(0.5, 0.8);
        for &p in &[0.1, 0.5, 0.95] {
            assert!(approx(ln.cdf(ln.quantile(p)), p, 1e-10));
        }
        let w = Weibull::new(1.5, 2.0);
        for &p in &[0.1, 0.5, 0.95] {
            assert!(approx(w.cdf(w.quantile(p)), p, 1e-12));
        }
        // Weibull(1, lambda) is Exponential(1/lambda).
        let w1 = Weibull::new(1.0, 2.0);
        let e = Exponential::new(0.5);
        assert!(approx(w1.cdf(1.7), e.cdf(1.7), 1e-12));
    }

    #[test]
    fn test_sample_means() {
        let mut rng = Rng::new(99);
        let n = 20_000;
        let dists: Vec<(Box<dyn Distribution>, f64)> = vec![
            (Box::new(Normal::new(1.0, 2.0)), 1.0),
            (Box::new(Exponential::new(0.5)), 2.0),
            (Box::new(Gamma::new(3.0, 2.0)), 1.5),
            (Box::new(Poisson::new(4.0)), 4.0),
            (Box::new(Binomial::new(20, 0.25)), 5.0),
        ];
        for (d, expected_mean) in &dists {
            let mean: f64 = (0..n).map(|_| d.sample(&mut rng)).sum::<f64>() / n as f64;
            let se = (d.variance() / n as f64).sqrt();
            assert!(
                (mean - expected_mean).abs() < 5.0 * se,
                "sample mean {mean} vs {expected_mean} (se {se})"
            );
        }
    }
}
