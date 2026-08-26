//! Time series analysis: correlation structure, stationarity, ARMA models,
//! smoothing, volatility, and change detection.
//!
//! A time series differs from a sample only in that the order matters, and
//! every tool here is a way of asking how much it matters. The
//! autocorrelation function measures it directly; the partial
//! autocorrelation strips out what is already explained by the lags in
//! between; the spectral density says the same thing in the frequency
//! domain. An ARMA model is a compact parameterisation of that structure,
//! and its impulse-response weights are the bridge between the two views --
//! they generate the autocovariances, the forecast error variances, and the
//! spectral density alike.
//!
//! Stationarity is the assumption the whole apparatus rests on, so it is
//! tested rather than assumed. The augmented Dickey-Fuller test takes a unit
//! root as the null and looks for evidence against it; the KPSS test takes
//! stationarity as the null and looks for evidence against *that*. They are
//! deliberately opposed: agreeing on a rejection is much stronger evidence
//! than either alone, and disagreement is a signal that the series is
//! neither cleanly one nor the other.
//!
//! The p-values for both come from tabulated quantiles of their non-standard
//! null distributions, interpolated. Neither statistic is asymptotically
//! normal or chi-squared -- a Dickey-Fuller `t`-ratio is not a `t` at all --
//! so a p-value computed from a standard distribution would be wrong rather
//! than approximate. The tables are documented where they are used.

use crate::error::GeomError;
use crate::fractals::Complex;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;
use crate::optimization::least_squares::levenberg_marquardt;
use crate::statistics::descriptive::mean;
use crate::statistics::distributions::{ChiSquared, Distribution, FDist};
use crate::statistics::inference::TestResult;

// ---------------------------------------------------------------------------
// Correlation structure
// ---------------------------------------------------------------------------

/// Sample autocorrelation at lags `0..=max_lag`.
///
/// Uses the divide-by-`n` estimator rather than dividing each lag by its own
/// count. That biases individual lags toward zero, but it is the choice that
/// makes the resulting sequence positive semi-definite, which is what lets
/// [`pacf`] and the Yule-Walker equations be solved at all. The
/// divide-by-`n-k` version can produce a sequence no stationary process
/// possesses, and Durbin-Levinson then divides by a negative variance.
///
/// Element 0 is 1 by construction.
///
/// # Panics
/// Panics unless the series has at least two points and `max_lag < n`.
#[must_use]
pub fn acf(x: &[f64], max_lag: usize) -> Vec<f64> {
    assert!(x.len() >= 2, "acf requires at least two observations");
    assert!(max_lag < x.len(), "acf requires max_lag < n");
    let n = x.len();
    let m = mean(x);
    let c0: f64 = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
    (0..=max_lag)
        .map(|k| {
            if c0 <= 0.0 {
                return if k == 0 { 1.0 } else { 0.0 };
            }
            let ck: f64 =
                (k..n).map(|t| (x[t] - m) * (x[t - k] - m)).sum::<f64>() / n as f64;
            ck / c0
        })
        .collect()
}

/// Sample partial autocorrelation at lags `0..=max_lag`, by the
/// Durbin-Levinson recursion.
///
/// The partial autocorrelation at lag `k` is the correlation between `x_t`
/// and `x_{t-k}` once the intervening lags are projected out -- equivalently,
/// the last coefficient of the best linear predictor of order `k`. For an
/// AR(p) process it is exactly zero beyond lag `p`, which is what makes it
/// the tool for choosing `p`.
///
/// Element 0 is 1, matching [`acf`].
///
/// # Panics
/// Panics under the same conditions as [`acf`].
#[must_use]
pub fn pacf(x: &[f64], max_lag: usize) -> Vec<f64> {
    let r = acf(x, max_lag);
    let mut out = vec![1.0; max_lag + 1];
    if max_lag == 0 {
        return out;
    }
    // phi holds the order-k predictor coefficients; each step extends it.
    let mut phi = vec![0.0f64; max_lag + 1];
    let mut prev = vec![0.0f64; max_lag + 1];
    phi[1] = r[1];
    out[1] = r[1];
    let mut v = 1.0 - r[1] * r[1];

    for k in 2..=max_lag {
        prev[..k].copy_from_slice(&phi[..k]);
        let num: f64 = r[k] - (1..k).map(|j| prev[j] * r[k - j]).sum::<f64>();
        let kappa = if v.abs() < 1e-15 { 0.0 } else { num / v };
        phi[k] = kappa;
        for j in 1..k {
            phi[j] = prev[j] - kappa * prev[k - j];
        }
        v *= 1.0 - kappa * kappa;
        out[k] = kappa;
    }
    out
}

/// Cross-correlation of `x` and `y` at lags `-max_lag..=max_lag`.
///
/// Element `max_lag + k` is the correlation between `x_t` and `y_{t+k}`, so a
/// peak at positive `k` means `x` leads `y` by `k` steps.
///
/// # Panics
/// Panics unless both series have the same length, at least two points, and
/// `max_lag < n`.
#[must_use]
pub fn cross_correlation_lags(x: &[f64], y: &[f64], max_lag: usize) -> Vec<f64> {
    assert!(x.len() == y.len(), "cross_correlation_lags requires equal lengths");
    assert!(x.len() >= 2, "cross_correlation_lags requires at least two observations");
    assert!(max_lag < x.len(), "cross_correlation_lags requires max_lag < n");
    let n = x.len();
    let (mx, my) = (mean(x), mean(y));
    let sx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum::<f64>().sqrt();
    let sy: f64 = y.iter().map(|v| (v - my) * (v - my)).sum::<f64>().sqrt();
    let denom = sx * sy;
    (0..2 * max_lag + 1)
        .map(|i| {
            if denom <= 0.0 {
                return 0.0;
            }
            let k = i as isize - max_lag as isize;
            let mut acc = 0.0;
            for t in 0..n {
                let u = t as isize + k;
                if u >= 0 && (u as usize) < n {
                    acc += (x[t] - mx) * (y[u as usize] - my);
                }
            }
            acc / denom
        })
        .collect()
}

/// The Ljung-Box portmanteau test for autocorrelation up to lag `lags`.
///
/// `Q = n(n+2) sum_{k=1}^{h} r_k^2 / (n-k)`, which is asymptotically
/// chi-squared on `h` degrees of freedom under the null that the series is
/// uncorrelated. A small p-value says the series has structure a white-noise
/// model would not produce.
///
/// # Panics
/// Panics unless `lags >= 1` and `lags < n`.
#[must_use]
pub fn ljung_box(x: &[f64], lags: usize) -> TestResult {
    assert!(lags >= 1, "ljung_box requires at least one lag");
    let n = x.len();
    assert!(lags < n, "ljung_box requires lags < n");
    let r = acf(x, lags);
    let q: f64 = (1..=lags).map(|k| r[k] * r[k] / (n - k) as f64).sum::<f64>()
        * (n * (n + 2)) as f64;
    let df = lags as f64;
    TestResult { statistic: q, p_value: 1.0 - ChiSquared::new(df).cdf(q), df }
}

// ---------------------------------------------------------------------------
// Differencing
// ---------------------------------------------------------------------------

/// The `d`-th successive difference of `x`, shortening it by `d`.
///
/// # Panics
/// Panics if `d >= x.len()`.
#[must_use]
pub fn difference(x: &[f64], d: usize) -> Vec<f64> {
    assert!(d < x.len(), "difference requires d < n");
    let mut out = x.to_vec();
    for _ in 0..d {
        out = out.windows(2).map(|w| w[1] - w[0]).collect();
    }
    out
}

/// The seasonal difference `x_t - x_{t-s}`, shortening the series by `s`.
///
/// # Panics
/// Panics unless `1 <= s < x.len()`.
#[must_use]
pub fn seasonal_difference(x: &[f64], s: usize) -> Vec<f64> {
    assert!(s >= 1 && s < x.len(), "seasonal_difference requires 1 <= s < n");
    (s..x.len()).map(|t| x[t] - x[t - s]).collect()
}

/// Rebuilds a series from its differences.
///
/// `initial` holds the first element of each successive difference, lowest
/// order first: `initial[j]` is `difference(x, j)[0]`, so `initial[0]` is
/// `x[0]`. Its length sets the differencing order being undone. Exactly
/// inverts [`difference`].
///
/// # Panics
/// Panics if `initial` is empty.
#[must_use]
pub fn undifference(diffed: &[f64], initial: &[f64]) -> Vec<f64> {
    assert!(!initial.is_empty(), "undifference requires at least one initial value");
    let mut out = diffed.to_vec();
    // Work outward from the innermost difference: each cumulative sum, seeded
    // with that level's first value, undoes one differencing step.
    for j in (0..initial.len()).rev() {
        let mut level = Vec::with_capacity(out.len() + 1);
        let mut acc = initial[j];
        level.push(acc);
        for &v in &out {
            acc += v;
            level.push(acc);
        }
        out = level;
    }
    out
}

// ---------------------------------------------------------------------------
// Stationarity
// ---------------------------------------------------------------------------

/// Quantiles of the Dickey-Fuller `tau` statistic for the constant-no-trend
/// case in large samples, as `(p, tau)` pairs.
///
/// The statistic looks like a `t`-ratio but is not one: under a unit root the
/// regressor is non-stationary, so the usual limit theory does not apply and
/// the distribution is skewed far to the left of a `t`. These are the
/// standard tabulated values (Fuller 1976, Table 8.5.2, `n -> infinity`).
const DF_TAU_TABLE: [(f64, f64); 11] = [
    (0.010, -3.43),
    (0.025, -3.12),
    (0.050, -2.86),
    (0.100, -2.57),
    (0.250, -2.16),
    (0.500, -1.57),
    (0.750, -1.04),
    (0.900, -0.44),
    (0.950, -0.07),
    (0.975, 0.23),
    (0.990, 0.60),
];

/// Quantiles of the Engle-Granger cointegration statistic with one regressor
/// and a constant. The residuals are estimated rather than observed, which
/// shifts the null distribution further left than the plain Dickey-Fuller
/// table above; using the wrong one of the two rejects far too readily.
const EG_TAU_TABLE: [(f64, f64); 7] = [
    (0.010, -3.90),
    (0.050, -3.34),
    (0.100, -3.04),
    (0.250, -2.58),
    (0.500, -2.13),
    (0.750, -1.71),
    (0.900, -1.35),
];

/// Quantiles of the KPSS statistic for level stationarity (Kwiatkowski et al.
/// 1992, Table 1). The statistic is a positive functional of a Brownian
/// bridge, so large values argue against the null of stationarity.
const KPSS_TABLE: [(f64, f64); 6] =
    [(0.900, 0.347), (0.950, 0.463), (0.975, 0.574), (0.990, 0.739), (0.500, 0.211), (0.100, 0.119)];

/// Linear interpolation of a p-value from a table of `(p, statistic)` pairs.
///
/// The table is sorted on the statistic first, so entries may be supplied in
/// any order. Values beyond either end are clamped rather than extrapolated:
/// a statistic far into the tail is reported at the tail's tabulated p-value,
/// which understates the significance but never invents a figure the table
/// does not support.
fn interpolate_p(table: &[(f64, f64)], statistic: f64) -> f64 {
    let mut pts: Vec<(f64, f64)> = table.iter().map(|&(p, s)| (s, p)).collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if statistic <= pts[0].0 {
        return pts[0].1;
    }
    if statistic >= pts[pts.len() - 1].0 {
        return pts[pts.len() - 1].1;
    }
    for w in pts.windows(2) {
        let ((s0, p0), (s1, p1)) = (w[0], w[1]);
        if statistic <= s1 {
            let t = if (s1 - s0).abs() < 1e-15 { 0.0 } else { (statistic - s0) / (s1 - s0) };
            return p0 + t * (p1 - p0);
        }
    }
    pts[pts.len() - 1].1
}

/// Ordinary least squares with an intercept, returning
/// `(coefficients, residuals, residual sum of squares)`.
///
/// The intercept is the first coefficient.
///
/// Solved through the normal equations `X'X beta = X'y` with a Cholesky
/// factorisation rather than by a QR of `X` itself. Every regression in this
/// module has far more rows than columns -- a few thousand observations
/// against a handful of lags -- and the crate's Householder QR accumulates an
/// explicit `n x n` orthogonal factor, which costs `O(n^2 k)`: over a billion
/// operations for a long pilot autoregression, against `O(n k^2)` here.
///
/// The trade is the usual one. Forming `X'X` squares the condition number, so
/// this loses roughly half the available digits on an ill-conditioned design
/// where QR would not. That is acceptable for regressions on lagged values of
/// a series, which are well scaled and nowhere near collinear unless the
/// series is degenerate -- and in that case the Cholesky factorisation fails
/// outright, which is reported rather than silently absorbed.
fn ols_with_intercept(
    predictors: &[Vec<f64>],
    y: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, f64), GeomError> {
    let n = y.len();
    let k = predictors.len() + 1;
    if n <= k {
        return Err(GeomError::InvalidArgument("regression has too few observations"));
    }
    if predictors.iter().any(|c| c.len() != n) {
        return Err(GeomError::InvalidArgument("regression predictor length mismatch"));
    }

    // Column j of the design is the constant for j = 0 and predictor j - 1
    // otherwise; `column` avoids materialising the n x k matrix twice.
    let column = |j: usize, i: usize| -> f64 {
        if j == 0 {
            1.0
        } else {
            predictors[j - 1][i]
        }
    };

    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for a in 0..k {
        for b in a..k {
            let v: f64 = (0..n).map(|i| column(a, i) * column(b, i)).sum();
            xtx.set(a, b, v);
            xtx.set(b, a, v);
        }
        xty[a] = (0..n).map(|i| column(a, i) * y[i]).sum();
    }

    let l = crate::linalg::cholesky::cholesky(&xtx)
        .map_err(|_| GeomError::Degenerate("regression design matrix is rank deficient"))?;
    let beta = crate::linalg::cholesky::cholesky_solve(&l, &xty)
        .map_err(|_| GeomError::Degenerate("regression design matrix is rank deficient"))?;
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(GeomError::Degenerate("regression produced a non-finite coefficient"));
    }

    let resid: Vec<f64> = (0..n)
        .map(|i| y[i] - (0..k).map(|j| beta[j] * column(j, i)).sum::<f64>())
        .collect();
    let rss: f64 = resid.iter().map(|r| r * r).sum();
    Ok((beta, resid, rss))
}

/// The augmented Dickey-Fuller test for a unit root, with a constant and no
/// trend.
///
/// Regresses `dy_t` on `y_{t-1}`, a constant, and `lags` lagged differences;
/// the statistic is the `t`-ratio on the `y_{t-1}` coefficient. The null is
/// that a unit root is present, so a *small* p-value is evidence the series
/// is stationary. `df` reports the residual degrees of freedom.
///
/// The p-value is interpolated from the module's table of Dickey-Fuller
/// quantiles; see the note on that table for why a `t` distribution would be
/// the wrong reference.
///
/// # Errors
/// Returns an error if the series is too short for the requested lag order or
/// the regression is rank deficient.
pub fn adf_test(x: &[f64], lags: usize) -> Result<TestResult, GeomError> {
    let n = x.len();
    if n < lags + 4 {
        return Err(GeomError::InvalidArgument("adf_test: series too short for the lag order"));
    }
    let dy = difference(x, 1);
    // Row t of the regression uses dy[t], y[t] (which is y_{t-1} for dy[t]),
    // and dy[t-1..t-lags].
    let start = lags;
    let rows = dy.len() - start;
    if rows <= lags + 2 {
        return Err(GeomError::InvalidArgument("adf_test: too few usable rows"));
    }
    let y: Vec<f64> = (start..dy.len()).map(|t| dy[t]).collect();
    let mut predictors: Vec<Vec<f64>> = Vec::with_capacity(lags + 1);
    predictors.push((start..dy.len()).map(|t| x[t]).collect());
    for l in 1..=lags {
        predictors.push((start..dy.len()).map(|t| dy[t - l]).collect());
    }

    let (beta, _, rss) = ols_with_intercept(&predictors, &y)?;
    let k = predictors.len() + 1;
    let df = (rows - k) as f64;
    let s2 = rss / df;

    // Standard error of the y_{t-1} coefficient: sqrt(s^2 (X'X)^{-1}_{11}).
    let mut a = Matrix::zeros(rows, k);
    for i in 0..rows {
        a.set(i, 0, 1.0);
        for (j, col) in predictors.iter().enumerate() {
            a.set(i, j + 1, col[i]);
        }
    }
    let xtx = a.transpose().mul(&a).map_err(|_| GeomError::Degenerate("adf_test: shape error"))?;
    let inv = crate::linalg::lu::lu_decompose(&xtx)
        .and_then(|d| d.inverse())
        .map_err(|_| GeomError::Degenerate("adf_test: design matrix is singular"))?;
    let se = (s2 * inv.get(1, 1)).sqrt();
    let tau = if se > 0.0 { beta[1] / se } else { 0.0 };

    Ok(TestResult { statistic: tau, p_value: interpolate_p(&DF_TAU_TABLE, tau), df })
}

/// The KPSS test for level stationarity.
///
/// The statistic is `sum_t S_t^2 / (n^2 s^2(l))`, where `S_t` is the partial
/// sum of deviations from the mean and `s^2(l)` is a Newey-West long-run
/// variance with the usual `l = floor(4 (n/100)^{1/4})` bandwidth. Here
/// stationarity is the *null*, so a small p-value is evidence against it --
/// the opposite polarity to [`adf_test`], which is the point of running both.
///
/// `df` is reported as the bandwidth actually used.
///
/// # Errors
/// Returns an error for a series shorter than four points or one with no
/// variation at all.
pub fn kpss_test(x: &[f64]) -> Result<TestResult, GeomError> {
    let n = x.len();
    if n < 4 {
        return Err(GeomError::InvalidArgument("kpss_test requires at least four observations"));
    }
    let m = mean(x);
    let e: Vec<f64> = x.iter().map(|v| v - m).collect();
    let mut s = 0.0;
    let mut acc = 0.0;
    for v in &e {
        s += v;
        acc += s * s;
    }

    let l = (4.0 * (n as f64 / 100.0).powf(0.25)).floor().max(1.0) as usize;
    let gamma0: f64 = e.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let mut long_run = gamma0;
    for j in 1..=l.min(n - 1) {
        let gj: f64 = (j..n).map(|t| e[t] * e[t - j]).sum::<f64>() / n as f64;
        // Bartlett weights taper the higher lags so the estimate stays
        // non-negative whatever the sample happens to produce.
        long_run += 2.0 * (1.0 - j as f64 / (l + 1) as f64) * gj;
    }
    if !(long_run > 0.0) {
        return Err(GeomError::Degenerate("kpss_test: long-run variance is not positive"));
    }

    let stat = acc / ((n * n) as f64 * long_run);
    Ok(TestResult {
        statistic: stat,
        // The table is indexed by the upper tail, so the p-value is one minus
        // the interpolated quantile position.
        p_value: 1.0 - interpolate_p(&KPSS_TABLE, stat),
        df: l as f64,
    })
}

// ---------------------------------------------------------------------------
// ARMA
// ---------------------------------------------------------------------------

/// An autoregressive moving-average model.
///
/// The process is written around its mean:
/// `(x_t - mu) = sum_i phi_i (x_{t-i} - mu) + e_t + sum_j theta_j e_{t-j}`,
/// with `e_t` white noise of variance `sigma2`. The sign convention on the
/// moving-average side is the additive one, matching the Box-Jenkins form.
#[derive(Debug, Clone, PartialEq)]
pub struct Arma {
    /// Autoregressive coefficients `phi_1 ..= phi_p`.
    pub ar: Vec<f64>,
    /// Moving-average coefficients `theta_1 ..= theta_q`.
    pub ma: Vec<f64>,
    /// Innovation variance.
    pub sigma2: f64,
    /// Process mean.
    pub mean: f64,
}

/// Anything larger than this in a conditional residual means the parameters
/// have wandered somewhere explosive; saturating there keeps the optimiser's
/// cost finite so it can step back rather than seeing a NaN.
const RESIDUAL_CLAMP: f64 = 1e12;

impl Arma {
    /// A model with the given coefficients.
    #[must_use]
    pub fn new(ar: Vec<f64>, ma: Vec<f64>, sigma2: f64, mean: f64) -> Self {
        Self { ar, ma, sigma2, mean }
    }

    /// Autoregressive order.
    #[must_use]
    pub fn p(&self) -> usize {
        self.ar.len()
    }

    /// Moving-average order.
    #[must_use]
    pub fn q(&self) -> usize {
        self.ma.len()
    }

    /// The conditional innovations implied by the model and the data.
    ///
    /// Pre-sample observations are replaced by the mean and pre-sample
    /// innovations by zero, which is the "conditional" in conditional sum of
    /// squares. The first few residuals therefore carry that assumption, and
    /// its influence dies out at the rate the moving-average part is
    /// invertible.
    #[must_use]
    pub fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut e = vec![0.0; n];
        for t in 0..n {
            let mut acc = x[t] - self.mean;
            for (i, &phi) in self.ar.iter().enumerate() {
                let lag = i + 1;
                let past = if t >= lag { x[t - lag] - self.mean } else { 0.0 };
                acc -= phi * past;
            }
            for (j, &theta) in self.ma.iter().enumerate() {
                let lag = j + 1;
                let past = if t >= lag { e[t - lag] } else { 0.0 };
                acc -= theta * past;
            }
            e[t] = acc.clamp(-RESIDUAL_CLAMP, RESIDUAL_CLAMP);
        }
        e
    }

    /// Fits by conditional sum of squares: choose the mean and the
    /// coefficients that make the conditional innovations as small as
    /// possible in the least-squares sense.
    ///
    /// The innovations *are* the residual vector, so this is a plain
    /// nonlinear least-squares problem and Levenberg-Marquardt solves it
    /// directly. `sigma2` is then the residual mean square on `n - k` degrees
    /// of freedom.
    ///
    /// # Errors
    /// Returns an error if the series is too short or the optimiser fails to
    /// converge.
    pub fn fit_css(x: &[f64], p: usize, q: usize) -> Result<Self, GeomError> {
        let n = x.len();
        let k = p + q + 1;
        if n < k + 5 {
            return Err(GeomError::InvalidArgument("fit_css: series too short for the orders"));
        }
        let residuals = |params: &[f64]| -> Vec<f64> {
            let model = Arma {
                mean: params[0],
                ar: params[1..1 + p].to_vec(),
                ma: params[1 + p..1 + p + q].to_vec(),
                sigma2: 1.0,
            };
            model.residuals(x)
        };
        // Start from a white-noise model at the sample mean: zero coefficients
        // is the one starting point that is always inside the stationary and
        // invertible region.
        let mut p0 = vec![0.0; k];
        p0[0] = mean(x);
        let fit = levenberg_marquardt(&residuals, None, &p0, 1e-10, 500)
            .map_err(|_| GeomError::Degenerate("fit_css: the optimiser did not converge"))?;

        let params = fit.params;
        let model = Arma {
            mean: params[0],
            ar: params[1..1 + p].to_vec(),
            ma: params[1 + p..1 + p + q].to_vec(),
            sigma2: 1.0,
        };
        let e = model.residuals(x);
        let rss: f64 = e.iter().map(|v| v * v).sum();
        Ok(Arma { sigma2: rss / (n - k) as f64, ..model })
    }

    /// Fits by the Hannan-Rissanen two-stage procedure.
    ///
    /// A long autoregression approximates the innovations, and those
    /// estimated innovations then enter a second regression as if they were
    /// observed, turning a nonlinear problem into two linear ones. It costs
    /// some efficiency against [`Arma::fit_css`] but needs no starting values
    /// and no iteration, which makes it a good source of starting values.
    ///
    /// # Errors
    /// Returns an error if the series is too short or either regression is
    /// rank deficient.
    pub fn fit_hannan_rissanen(x: &[f64], p: usize, q: usize) -> Result<Self, GeomError> {
        let n = x.len();
        // The pilot order has to grow with the sample for the approximation to
        // be consistent, but stay well short of n; log(n)^2 is the usual rule.
        let lower = p + q + 1;
        let upper = n / 4;
        if upper < lower {
            return Err(GeomError::InvalidArgument(
                "fit_hannan_rissanen: series too short for the orders",
            ));
        }
        let m = ((n as f64).ln().powi(2).ceil() as usize).clamp(lower, upper).max(1);
        if n < 4 * (m + p + q + 2) {
            return Err(GeomError::InvalidArgument(
                "fit_hannan_rissanen: series too short for the orders",
            ));
        }

        // Stage one: a long autoregression, whose residuals stand in for the
        // unobserved innovations.
        let rows = n - m;
        let y: Vec<f64> = (m..n).map(|t| x[t]).collect();
        let pilot: Vec<Vec<f64>> = (1..=m).map(|l| (m..n).map(|t| x[t - l]).collect()).collect();
        let (_, resid, _) = ols_with_intercept(&pilot, &y)?;
        // resid[i] is the innovation at time m + i.
        let mut eps = vec![0.0; n];
        for (i, r) in resid.iter().enumerate() {
            eps[m + i] = *r;
        }
        debug_assert_eq!(resid.len(), rows);

        // Stage two: regress on both the observed lags and the estimated
        // innovations.
        let start = m + p.max(q);
        if n <= start + p + q + 2 {
            return Err(GeomError::InvalidArgument("fit_hannan_rissanen: too few usable rows"));
        }
        let y2: Vec<f64> = (start..n).map(|t| x[t]).collect();
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(p + q);
        for l in 1..=p {
            cols.push((start..n).map(|t| x[t - l]).collect());
        }
        for l in 1..=q {
            cols.push((start..n).map(|t| eps[t - l]).collect());
        }
        let (beta, resid2, rss) = ols_with_intercept(&cols, &y2)?;

        let ar: Vec<f64> = beta[1..1 + p].to_vec();
        let ma: Vec<f64> = beta[1 + p..1 + p + q].to_vec();
        // The intercept is mu (1 - sum phi), so recover mu from it.
        let phi_sum: f64 = ar.iter().sum();
        let mu = if (1.0 - phi_sum).abs() > 1e-12 { beta[0] / (1.0 - phi_sum) } else { mean(x) };
        let dof = (resid2.len() - (p + q + 1)) as f64;
        Ok(Arma { ar, ma, sigma2: rss / dof.max(1.0), mean: mu })
    }

    /// Generates `n` observations, discarding a burn-in long enough for the
    /// transient from the zero start to decay.
    ///
    /// # Panics
    /// Panics if `n` is zero or `sigma2` is negative.
    #[must_use]
    pub fn simulate(&self, n: usize, rng: &mut Rng) -> Vec<f64> {
        assert!(n > 0, "simulate requires n > 0");
        assert!(self.sigma2 >= 0.0, "simulate requires a non-negative variance");
        let burn = 500 + 20 * (self.p() + self.q());
        let total = n + burn;
        let sd = self.sigma2.sqrt();
        let mut e = vec![0.0; total];
        let mut y = vec![0.0; total];
        for t in 0..total {
            e[t] = sd * rng.next_gaussian();
            let mut acc = e[t];
            for (i, &phi) in self.ar.iter().enumerate() {
                if t > i {
                    acc += phi * y[t - i - 1];
                }
            }
            for (j, &theta) in self.ma.iter().enumerate() {
                if t > j {
                    acc += theta * e[t - j - 1];
                }
            }
            y[t] = acc;
        }
        y[burn..].iter().map(|v| v + self.mean).collect()
    }

    /// The impulse-response (psi) weights: the coefficients of the model's
    /// infinite moving-average representation.
    ///
    /// `psi_0 = 1` and `psi_j = theta_j + sum_i phi_i psi_{j-i}`. These are
    /// the single most useful derived quantity in the module -- forecast
    /// error variances, the autocovariances, and the spectral density are all
    /// expressible in them.
    ///
    /// Returns `n` weights, `psi_0` first.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    #[must_use]
    pub fn impulse_response(&self, n: usize) -> Vec<f64> {
        assert!(n > 0, "impulse_response requires n > 0");
        let mut psi = vec![0.0; n];
        psi[0] = 1.0;
        for j in 1..n {
            let mut acc = if j <= self.q() { self.ma[j - 1] } else { 0.0 };
            for (i, &phi) in self.ar.iter().enumerate() {
                let lag = i + 1;
                if j >= lag {
                    acc += phi * psi[j - lag];
                }
            }
            psi[j] = acc;
        }
        psi
    }

    /// The spectral density at each supplied angular frequency.
    ///
    /// `f(w) = (sigma2 / 2 pi) |theta(e^{-iw})|^2 / |phi(e^{-iw})|^2`. It
    /// integrates over `[-pi, pi]` to the process variance, which is the
    /// frequency-domain statement of the same second-order structure the
    /// autocovariances describe.
    #[must_use]
    pub fn spectral_density(&self, freqs: &[f64]) -> Vec<f64> {
        freqs
            .iter()
            .map(|&w| {
                let (mut ar_re, mut ar_im) = (1.0f64, 0.0f64);
                for (i, &phi) in self.ar.iter().enumerate() {
                    let angle = -(i as f64 + 1.0) * w;
                    ar_re -= phi * angle.cos();
                    ar_im -= phi * angle.sin();
                }
                let (mut ma_re, mut ma_im) = (1.0f64, 0.0f64);
                for (j, &theta) in self.ma.iter().enumerate() {
                    let angle = -(j as f64 + 1.0) * w;
                    ma_re += theta * angle.cos();
                    ma_im += theta * angle.sin();
                }
                let den = ar_re * ar_re + ar_im * ar_im;
                if den <= 0.0 {
                    return f64::INFINITY;
                }
                self.sigma2 / (2.0 * std::f64::consts::PI) * (ma_re * ma_re + ma_im * ma_im) / den
            })
            .collect()
    }

    /// `(stationary, invertible)`.
    ///
    /// Stationarity asks that every root of `1 - phi_1 z - ... - phi_p z^p`
    /// lie outside the unit circle; invertibility asks the same of
    /// `1 + theta_1 z + ... + theta_q z^q`. An empty side is trivially both.
    ///
    /// A root exactly on the circle counts as failing, since the boundary is
    /// where stationarity breaks down.
    #[must_use]
    pub fn roots_check(&self) -> (bool, bool) {
        (roots_outside_unit_circle(&self.ar, -1.0), roots_outside_unit_circle(&self.ma, 1.0))
    }

    /// The conditional (Gaussian) log-likelihood of `x` under this model.
    ///
    /// Conditional because the pre-sample values are fixed rather than
    /// integrated out; it is the quantity [`Arma::fit_css`] maximises, and the
    /// one [`Arma::aic`] and [`Arma::bic`] penalise.
    #[must_use]
    pub fn log_likelihood(&self, x: &[f64]) -> f64 {
        if !(self.sigma2 > 0.0) {
            return f64::NEG_INFINITY;
        }
        let e = self.residuals(x);
        let rss: f64 = e.iter().map(|v| v * v).sum();
        let n = x.len() as f64;
        -0.5 * n * (2.0 * std::f64::consts::PI * self.sigma2).ln() - rss / (2.0 * self.sigma2)
    }

    /// Number of free parameters: the coefficients, the mean, and the
    /// innovation variance.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.p() + self.q() + 2
    }

    /// Akaike's information criterion, `-2 ln L + 2k`. Lower is better.
    #[must_use]
    pub fn aic(&self, x: &[f64]) -> f64 {
        -2.0 * self.log_likelihood(x) + 2.0 * self.n_params() as f64
    }

    /// The Bayesian information criterion, `-2 ln L + k ln n`. Penalises
    /// extra parameters harder than [`Arma::aic`] for any sample past
    /// `n = e^2`, so it selects more parsimonious models.
    #[must_use]
    pub fn bic(&self, x: &[f64]) -> f64 {
        -2.0 * self.log_likelihood(x) + self.n_params() as f64 * (x.len() as f64).ln()
    }

    /// `h`-step-ahead forecasts and their standard errors.
    ///
    /// Point forecasts run the model recursion forward with future
    /// innovations set to their expectation of zero. The standard errors are
    /// `sigma sqrt(sum_{j<h} psi_j^2)`, which grows with the horizon and, for
    /// a stationary model, converges to the process standard deviation --
    /// beyond the memory of the process the best forecast is the mean and the
    /// uncertainty is the unconditional one.
    ///
    /// # Panics
    /// Panics if `h` is zero or `x` is empty.
    #[must_use]
    pub fn forecast(&self, x: &[f64], h: usize) -> (Vec<f64>, Vec<f64>) {
        assert!(h > 0, "forecast requires h > 0");
        assert!(!x.is_empty(), "forecast requires observations");
        let e = self.residuals(x);
        let n = x.len();
        let mut future = Vec::with_capacity(h);
        for step in 0..h {
            let mut acc = 0.0;
            for (i, &phi) in self.ar.iter().enumerate() {
                let lag = i + 1;
                // Reach back into the forecasts first, then the data.
                let past = if step >= lag {
                    future[step - lag] - self.mean
                } else if n + step >= lag {
                    x[n + step - lag] - self.mean
                } else {
                    0.0
                };
                acc += phi * past;
            }
            for (j, &theta) in self.ma.iter().enumerate() {
                let lag = j + 1;
                // Innovations past the end of the data are zero in expectation.
                if step < lag && n + step >= lag {
                    acc += theta * e[n + step - lag];
                }
            }
            future.push(acc + self.mean);
        }

        let psi = self.impulse_response(h);
        let sd = self.sigma2.max(0.0).sqrt();
        let mut cumulative = 0.0;
        let errors = psi
            .iter()
            .map(|&p| {
                cumulative += p * p;
                sd * cumulative.sqrt()
            })
            .collect();
        (future, errors)
    }
}

/// Whether every root of `1 + sign * (c_1 z + ... + c_k z^k)` lies strictly
/// outside the unit circle.
///
/// `sign` is `-1` for the autoregressive polynomial, whose coefficients enter
/// with a minus, and `+1` for the moving-average one.
fn roots_outside_unit_circle(coeffs: &[f64], sign: f64) -> bool {
    // Drop trailing zeros: a coefficient of zero at the top does not
    // contribute a root, it just lowers the degree.
    let mut trimmed = coeffs;
    while let Some((&last, rest)) = trimmed.split_last() {
        if last == 0.0 {
            trimmed = rest;
        } else {
            break;
        }
    }
    if trimmed.is_empty() {
        return true;
    }
    // polynomial_roots takes the highest power first, so reverse.
    let mut poly: Vec<f64> = trimmed.iter().rev().map(|&c| sign * c).collect();
    poly.push(1.0);
    match crate::numerical::roots::polynomial_roots(&poly) {
        Ok(roots) => roots.iter().all(|r| r.norm() > 1.0 + 1e-9),
        Err(_) => false,
    }
}

/// An ARIMA model: an [`Arma`] fitted to the `d`-th difference.
#[derive(Debug, Clone, PartialEq)]
pub struct Arima {
    /// Order of differencing applied before the ARMA part.
    pub d: usize,
    /// The model for the differenced series.
    pub arma: Arma,
    /// The first value of each successive difference of the training data,
    /// which is what [`undifference`] needs to put a forecast back on the
    /// original scale.
    pub initial: Vec<f64>,
    /// The tail of the training series, kept so forecasts can be integrated
    /// back up without the caller re-supplying it.
    tail: Vec<f64>,
}

impl Arima {
    /// Differences `d` times, then fits an ARMA(`p`, `q`) by conditional sum
    /// of squares.
    ///
    /// # Errors
    /// Returns an error if the series is too short or the ARMA fit fails.
    pub fn fit(x: &[f64], p: usize, d: usize, q: usize) -> Result<Self, GeomError> {
        if d >= x.len() {
            return Err(GeomError::InvalidArgument("Arima::fit: d must be less than n"));
        }
        let diffed = difference(x, d);
        let arma = Arma::fit_css(&diffed, p, q)?;
        let initial = (0..d).map(|j| difference(x, j)[0]).collect();
        Ok(Self { d, arma, initial, tail: x.to_vec() })
    }

    /// `h`-step forecasts on the original scale, with standard errors.
    ///
    /// The ARMA part forecasts the differenced series; integrating those
    /// forecasts back up is a cumulative sum, so the errors accumulate too --
    /// the standard error of the `h`-step forecast of an integrated series is
    /// the norm of the *partial sums* of the psi weights, not of the weights
    /// themselves. That is why an ARIMA forecast interval keeps widening
    /// without bound while a stationary ARMA one levels off.
    ///
    /// # Panics
    /// Panics if `h` is zero.
    #[must_use]
    pub fn forecast(&self, h: usize) -> (Vec<f64>, Vec<f64>) {
        assert!(h > 0, "forecast requires h > 0");
        let diffed = difference(&self.tail, self.d);
        let (point, _) = self.arma.forecast(&diffed, h);

        // Integrate the point forecasts back up, seeded with the last observed
        // value at each differencing level.
        let mut level = point;
        for j in (0..self.d).rev() {
            let base = *difference(&self.tail, j).last().unwrap_or(&0.0);
            let mut acc = base;
            level = level
                .iter()
                .map(|v| {
                    acc += v;
                    acc
                })
                .collect();
        }

        // Cumulate the psi weights d times to get the integrated ones.
        let mut psi = self.arma.impulse_response(h);
        for _ in 0..self.d {
            let mut acc = 0.0;
            psi = psi
                .iter()
                .map(|v| {
                    acc += v;
                    acc
                })
                .collect();
        }
        let sd = self.arma.sigma2.max(0.0).sqrt();
        let mut cumulative = 0.0;
        let errors = psi
            .iter()
            .map(|&p| {
                cumulative += p * p;
                sd * cumulative.sqrt()
            })
            .collect();
        (level, errors)
    }
}

/// Selects `(p, d, q)` by minimising AIC over a grid, choosing `d` by
/// differencing until an augmented Dickey-Fuller test rejects a unit root.
///
/// Differencing order is settled first and separately, because AIC cannot
/// compare across it: differencing changes the data the likelihood is
/// computed on, so the numbers are not on the same scale.
///
/// # Errors
/// Returns an error if no candidate model in the grid can be fitted.
pub fn auto_arima(
    x: &[f64],
    max_p: usize,
    max_d: usize,
    max_q: usize,
) -> Result<Arima, GeomError> {
    let mut d = 0usize;
    while d < max_d {
        let level = difference(x, d);
        match adf_test(&level, 1) {
            // A p-value at or below 0.05 is evidence against the unit root,
            // so stop differencing.
            Ok(t) if t.p_value <= 0.05 => break,
            Ok(_) => d += 1,
            Err(_) => break,
        }
    }

    let diffed = difference(x, d);
    let mut best: Option<(f64, usize, usize)> = None;
    for p in 0..=max_p {
        for q in 0..=max_q {
            if p == 0 && q == 0 {
                continue;
            }
            if let Ok(model) = Arma::fit_css(&diffed, p, q) {
                let (stationary, invertible) = model.roots_check();
                if !stationary || !invertible {
                    continue;
                }
                let score = model.aic(&diffed);
                if score.is_finite() && best.is_none_or(|(b, _, _)| score < b) {
                    best = Some((score, p, q));
                }
            }
        }
    }
    let (_, p, q) = best.ok_or(GeomError::Degenerate("auto_arima: no candidate model fitted"))?;
    Arima::fit(x, p, d, q)
}

/// A seasonal ARIMA model, `(p, d, q) x (P, D, Q)_s`.
///
/// Fitted by applying the seasonal difference `D` times and the ordinary
/// difference `d` times, then estimating the non-seasonal and seasonal
/// polynomials on the doubly differenced series. The seasonal part is modelled
/// as an ARMA in lags that are multiples of `s`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sarima {
    /// Non-seasonal differencing order.
    pub d: usize,
    /// Seasonal differencing order.
    pub seasonal_d: usize,
    /// Season length.
    pub s: usize,
    /// The model fitted to the doubly differenced series, with the seasonal
    /// terms sitting at lags `s, 2s, ...` of an otherwise sparse polynomial.
    pub arma: Arma,
    /// Seasonally differenced training data, kept for forecasting.
    working: Vec<f64>,
    /// The untouched training series.
    original: Vec<f64>,
}

impl Sarima {
    /// Fits a `(p, d, q) x (P, D, Q)_s` model.
    ///
    /// The combined autoregressive polynomial has non-zero coefficients at
    /// lags `1..=p` and at `s, 2s, ...` up to `P s`; the moving-average side
    /// likewise. Cross-product terms of the multiplicative form are omitted,
    /// which makes this the additive rather than the strictly multiplicative
    /// SARIMA -- the difference is second order and the additive form is what
    /// conditional least squares can identify without a much longer series.
    ///
    /// # Errors
    /// Returns an error if the series is too short after differencing or the
    /// fit fails.
    pub fn fit(
        x: &[f64],
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        s: usize,
    ) -> Result<Self, GeomError> {
        if s < 2 {
            return Err(GeomError::InvalidArgument("Sarima::fit requires a season of at least 2"));
        }
        let mut working = x.to_vec();
        for _ in 0..seasonal_d {
            if working.len() <= s + 1 {
                return Err(GeomError::InvalidArgument("Sarima::fit: series too short"));
            }
            working = seasonal_difference(&working, s);
        }
        let ordinary = difference(&working, d);

        let ar_len = p.max(seasonal_p * s);
        let ma_len = q.max(seasonal_q * s);
        let free_ar: Vec<usize> =
            (1..=ar_len).filter(|&l| l <= p || (l % s == 0 && l / s <= seasonal_p)).collect();
        let free_ma: Vec<usize> =
            (1..=ma_len).filter(|&l| l <= q || (l % s == 0 && l / s <= seasonal_q)).collect();
        if free_ar.is_empty() && free_ma.is_empty() {
            return Err(GeomError::InvalidArgument("Sarima::fit: no free parameters"));
        }

        let k = free_ar.len() + free_ma.len() + 1;
        if ordinary.len() < k + ar_len.max(ma_len) + 5 {
            return Err(GeomError::InvalidArgument("Sarima::fit: series too short for the orders"));
        }

        // Only the lags the orders actually name are free; everything between
        // them is pinned at zero, which is what makes a seasonal model cheap
        // to estimate at a long season.
        let expand = |params: &[f64]| -> Arma {
            let mut ar = vec![0.0; ar_len];
            let mut ma = vec![0.0; ma_len];
            for (i, &lag) in free_ar.iter().enumerate() {
                ar[lag - 1] = params[1 + i];
            }
            for (j, &lag) in free_ma.iter().enumerate() {
                ma[lag - 1] = params[1 + free_ar.len() + j];
            }
            Arma { ar, ma, sigma2: 1.0, mean: params[0] }
        };

        let target = ordinary.clone();
        let residuals = |params: &[f64]| -> Vec<f64> { expand(params).residuals(&target) };
        let mut p0 = vec![0.0; k];
        p0[0] = mean(&ordinary);
        let fit = levenberg_marquardt(&residuals, None, &p0, 1e-10, 500)
            .map_err(|_| GeomError::Degenerate("Sarima::fit: the optimiser did not converge"))?;

        let model = expand(&fit.params);
        let e = model.residuals(&ordinary);
        let rss: f64 = e.iter().map(|v| v * v).sum();
        let arma = Arma { sigma2: rss / (ordinary.len() - k) as f64, ..model };
        Ok(Self { d, seasonal_d, s, arma, working, original: x.to_vec() })
    }

    /// `h`-step forecasts on the original scale.
    ///
    /// # Panics
    /// Panics if `h` is zero.
    #[must_use]
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        assert!(h > 0, "forecast requires h > 0");
        let ordinary = difference(&self.working, self.d);
        let (point, _) = self.arma.forecast(&ordinary, h);

        // Undo the ordinary differencing.
        let mut level = point;
        for j in (0..self.d).rev() {
            let base = *difference(&self.working, j).last().unwrap_or(&0.0);
            let mut acc = base;
            level = level
                .iter()
                .map(|v| {
                    acc += v;
                    acc
                })
                .collect();
        }

        // Undo the seasonal differencing, one level at a time: the forecast at
        // step t adds back the value one season earlier, which may itself be a
        // forecast once t exceeds the season length.
        let mut history_levels: Vec<Vec<f64>> = Vec::with_capacity(self.seasonal_d);
        let mut cur = self.original.clone();
        for _ in 0..self.seasonal_d {
            history_levels.push(cur.clone());
            cur = seasonal_difference(&cur, self.s);
        }
        for hist in history_levels.iter().rev() {
            let mut extended = hist.clone();
            for (i, v) in level.iter().enumerate() {
                let base = extended[hist.len() + i - self.s];
                extended.push(base + v);
            }
            level = extended[hist.len()..].to_vec();
        }
        level
    }
}

// ---------------------------------------------------------------------------
// Exponential smoothing
// ---------------------------------------------------------------------------

/// Simple exponential smoothing: `s_t = alpha x_t + (1 - alpha) s_{t-1}`,
/// seeded at `x_0`.
///
/// The smoothed value is a geometrically weighted average of the whole past,
/// and the weights sum to one, so a constant series is reproduced exactly at
/// any `alpha`.
///
/// # Panics
/// Panics unless `x` is non-empty and `alpha` is in `[0, 1]`.
#[must_use]
pub fn exponential_smoothing(x: &[f64], alpha: f64) -> Vec<f64> {
    assert!(!x.is_empty(), "exponential_smoothing requires observations");
    assert!((0.0..=1.0).contains(&alpha), "exponential_smoothing requires alpha in [0, 1]");
    let mut s = x[0];
    x.iter()
        .map(|&v| {
            s = alpha * v + (1.0 - alpha) * s;
            s
        })
        .collect()
}

/// Holt's linear method: a smoothed level and a smoothed slope.
///
/// Element `t` of the result is the one-step-ahead prediction of `x[t]`, made
/// from the state after seeing `x[t-1]` -- the same convention as
/// [`holt_winters`]. Unlike simple smoothing this tracks a linear trend
/// without lagging behind it.
///
/// The state is seeded one step *before* the data: the slope from the first
/// two points, and a level back-extrapolated so that `level + trend` equals
/// `x[0]`. Seeding the level at `x[0]` itself, as is often done, puts the
/// state half a step ahead of where the recursion expects it and leaves a
/// transient that takes tens of observations to decay -- on an exact straight
/// line, which the method should reproduce perfectly from the first step.
///
/// # Panics
/// Panics unless `x` has at least two points and both parameters lie in
/// `[0, 1]`.
#[must_use]
pub fn double_exponential(x: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
    assert!(x.len() >= 2, "double_exponential requires at least two observations");
    assert!((0.0..=1.0).contains(&alpha), "double_exponential requires alpha in [0, 1]");
    assert!((0.0..=1.0).contains(&beta), "double_exponential requires beta in [0, 1]");
    let mut trend = x[1] - x[0];
    let mut level = x[0] - trend;
    let mut out = Vec::with_capacity(x.len());
    for &v in x {
        out.push(level + trend);
        let previous = level;
        level = alpha * v + (1.0 - alpha) * (level + trend);
        trend = beta * (level - previous) + (1.0 - beta) * trend;
    }
    out
}

/// The smoothing state left behind by [`holt_winters`], enough to continue
/// the recursion or to forecast forward.
#[derive(Debug, Clone, PartialEq)]
pub struct HwState {
    /// Final level.
    pub level: f64,
    /// Final slope.
    pub trend: f64,
    /// Final seasonal factors, oldest phase first.
    pub seasonal: Vec<f64>,
    /// Whether the seasonal component multiplies rather than adds.
    pub multiplicative: bool,
}

impl HwState {
    /// `h`-step forecasts continuing from this state.
    ///
    /// # Panics
    /// Panics if `h` is zero or the seasonal vector is empty.
    #[must_use]
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        assert!(h > 0, "forecast requires h > 0");
        let s = self.seasonal.len();
        assert!(s > 0, "forecast requires a seasonal period");
        (1..=h)
            .map(|k| {
                let base = self.level + k as f64 * self.trend;
                let factor = self.seasonal[(k - 1) % s];
                if self.multiplicative {
                    base * factor
                } else {
                    base + factor
                }
            })
            .collect()
    }
}

/// Holt-Winters triple exponential smoothing.
///
/// Tracks a level, a slope, and a set of seasonal factors, each updated by
/// its own smoothing constant. Returns the one-step-ahead fitted values
/// alongside the final state.
///
/// The seasonal factors are initialised from the first complete season and,
/// in the additive case, centred so they sum to zero -- otherwise the level
/// and the seasonal component are not separately identified and the pair can
/// drift apart while their sum stays right.
///
/// # Panics
/// Panics unless the series covers at least two full seasons, `season_len` is
/// at least 2, and all three parameters lie in `[0, 1]`.
#[must_use]
pub fn holt_winters(
    x: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    season_len: usize,
    multiplicative: bool,
) -> (Vec<f64>, HwState) {
    assert!(season_len >= 2, "holt_winters requires a season of at least 2");
    assert!(x.len() >= 2 * season_len, "holt_winters requires at least two full seasons");
    assert!((0.0..=1.0).contains(&alpha), "holt_winters requires alpha in [0, 1]");
    assert!((0.0..=1.0).contains(&beta), "holt_winters requires beta in [0, 1]");
    assert!((0.0..=1.0).contains(&gamma), "holt_winters requires gamma in [0, 1]");

    let s = season_len;
    let first: f64 = x[..s].iter().sum::<f64>() / s as f64;
    let second: f64 = x[s..2 * s].iter().sum::<f64>() / s as f64;
    let mut trend = (second - first) / s as f64;
    // `first` is the mean of the opening season, which describes the middle of
    // that window rather than its start. Back-extrapolating it to one step
    // before the data puts the level where the recursion expects it, and the
    // seasonal factors are then taken against that trend line rather than
    // against a flat mean -- otherwise each factor absorbs part of the slope
    // and the two components take tens of seasons to sort themselves out.
    let mut level = first - trend * (s as f64 + 1.0) / 2.0;
    let mut seasonal: Vec<f64> = (0..s)
        .map(|i| {
            let baseline = level + trend * (i + 1) as f64;
            if multiplicative {
                if baseline.abs() > 1e-12 {
                    x[i] / baseline
                } else {
                    1.0
                }
            } else {
                x[i] - baseline
            }
        })
        .collect();

    let mut fitted = Vec::with_capacity(x.len());
    for (t, &v) in x.iter().enumerate() {
        let idx = t % s;
        let season = seasonal[idx];
        let predicted =
            if multiplicative { (level + trend) * season } else { level + trend + season };
        fitted.push(predicted);

        let previous = level;
        if multiplicative {
            let deseasonalised = if season.abs() > 1e-12 { v / season } else { v };
            level = alpha * deseasonalised + (1.0 - alpha) * (level + trend);
            trend = beta * (level - previous) + (1.0 - beta) * trend;
            if level.abs() > 1e-12 {
                seasonal[idx] = gamma * (v / level) + (1.0 - gamma) * season;
            }
        } else {
            level = alpha * (v - season) + (1.0 - alpha) * (level + trend);
            trend = beta * (level - previous) + (1.0 - beta) * trend;
            seasonal[idx] = gamma * (v - level) + (1.0 - gamma) * season;
        }
    }

    // Rotate so element 0 is the phase the next observation would land on.
    let phase = x.len() % s;
    seasonal.rotate_left(phase);
    (fitted, HwState { level, trend, seasonal, multiplicative })
}

/// Chooses `(alpha, beta, gamma)` by minimising the one-step-ahead sum of
/// squared errors over a coarse grid followed by a local refinement.
///
/// A grid rather than a gradient method: the Holt-Winters error surface is
/// not convex in the three constants and has flat regions near the corners of
/// the unit cube, where a local method started badly will simply stop.
///
/// # Panics
/// Panics under the same conditions as [`holt_winters`].
#[must_use]
pub fn holt_winters_optimize(x: &[f64], season_len: usize) -> (f64, f64, f64) {
    let sse = |a: f64, b: f64, g: f64| -> f64 {
        let (fitted, _) = holt_winters(x, a, b, g, season_len, false);
        // Skip the first season: those fitted values are dominated by the
        // initialisation rather than by the parameters being scored.
        fitted
            .iter()
            .zip(x)
            .skip(season_len)
            .map(|(f, v)| (f - v) * (f - v))
            .sum::<f64>()
    };

    let grid = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9];
    let mut best = (grid[0], grid[0], grid[0]);
    let mut best_score = f64::INFINITY;
    for &a in &grid {
        for &b in &grid {
            for &g in &grid {
                let score = sse(a, b, g);
                if score < best_score {
                    best_score = score;
                    best = (a, b, g);
                }
            }
        }
    }

    // Refine by halving steps around the grid winner.
    let mut step = 0.1;
    for _ in 0..6 {
        let mut improved = false;
        for &(da, db, dg) in &[
            (step, 0.0, 0.0),
            (-step, 0.0, 0.0),
            (0.0, step, 0.0),
            (0.0, -step, 0.0),
            (0.0, 0.0, step),
            (0.0, 0.0, -step),
        ] {
            let candidate = (
                (best.0 + da).clamp(0.0, 1.0),
                (best.1 + db).clamp(0.0, 1.0),
                (best.2 + dg).clamp(0.0, 1.0),
            );
            let score = sse(candidate.0, candidate.1, candidate.2);
            if score < best_score {
                best_score = score;
                best = candidate;
                improved = true;
            }
        }
        if !improved {
            step /= 2.0;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Volatility
// ---------------------------------------------------------------------------

/// A GARCH(1,1) volatility model:
/// `sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2`.
///
/// The single most used model in the family, because two parameters are
/// enough to reproduce the two features that matter: volatility clusters, and
/// it mean-reverts. `alpha + beta` is the persistence, and the model is
/// stationary only while that sum is below one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Garch11 {
    /// Constant term; must be positive for the variance to stay positive.
    pub omega: f64,
    /// Weight on the previous squared return.
    pub alpha: f64,
    /// Weight on the previous variance.
    pub beta: f64,
}

impl Garch11 {
    /// `alpha + beta`: how much of today's variance shock survives to
    /// tomorrow. At 1 the process has a unit root in variance and no
    /// unconditional variance exists.
    #[must_use]
    pub fn persistence(&self) -> f64 {
        self.alpha + self.beta
    }

    /// `omega / (1 - alpha - beta)`, the level volatility reverts to.
    ///
    /// Infinite once persistence reaches one.
    #[must_use]
    pub fn unconditional_variance(&self) -> f64 {
        let p = self.persistence();
        if p >= 1.0 {
            f64::INFINITY
        } else {
            self.omega / (1.0 - p)
        }
    }

    /// The filtered conditional variance for each return, seeded at the
    /// unconditional variance where one exists and at the sample variance
    /// otherwise.
    ///
    /// # Panics
    /// Panics if `returns` is empty.
    #[must_use]
    pub fn conditional_variance(&self, returns: &[f64]) -> Vec<f64> {
        assert!(!returns.is_empty(), "conditional_variance requires returns");
        let seed = if self.persistence() < 1.0 {
            self.unconditional_variance()
        } else {
            returns.iter().map(|r| r * r).sum::<f64>() / returns.len() as f64
        };
        let mut v = seed.max(1e-300);
        let mut out = Vec::with_capacity(returns.len());
        for t in 0..returns.len() {
            if t > 0 {
                let prev = returns[t - 1];
                v = self.omega + self.alpha * prev * prev + self.beta * v;
            }
            out.push(v);
        }
        out
    }

    /// Fits by maximising the Gaussian quasi-likelihood over a Nelder-Mead
    /// simplex, parameterised so the constraints hold by construction.
    ///
    /// `omega` is optimised on the log scale, keeping it positive, and
    /// `(alpha, beta)` through a softmax-style map onto the simplex
    /// `alpha, beta > 0`, `alpha + beta < 1`, which is where the model is
    /// stationary. An unconstrained fit routinely wanders to a negative
    /// variance, where the likelihood is not merely bad but undefined.
    ///
    /// # Errors
    /// Returns an error for a series too short to identify three parameters.
    pub fn fit(returns: &[f64]) -> Result<Self, GeomError> {
        if returns.len() < 30 {
            return Err(GeomError::InvalidArgument("Garch11::fit requires at least 30 returns"));
        }
        let sample_var: f64 =
            returns.iter().map(|r| r * r).sum::<f64>() / returns.len() as f64;
        if !(sample_var > 0.0) {
            return Err(GeomError::Degenerate("Garch11::fit: returns have no variation"));
        }

        // Map R^3 onto the stationary region.
        let unpack = |p: &[f64]| -> Garch11 {
            let omega = p[0].clamp(-40.0, 40.0).exp();
            // Logistic on the total, then split it between alpha and beta.
            let total = 0.999 / (1.0 + (-p[1].clamp(-40.0, 40.0)).exp());
            let share = 1.0 / (1.0 + (-p[2].clamp(-40.0, 40.0)).exp());
            Garch11 { omega, alpha: total * share, beta: total * (1.0 - share) }
        };

        let negative_ll = |p: &[f64]| -> f64 {
            let model = unpack(p);
            let v = model.conditional_variance(returns);
            let mut acc = 0.0;
            for (r, s2) in returns.iter().zip(&v) {
                if !(*s2 > 0.0) || !s2.is_finite() {
                    return f64::MAX;
                }
                acc += s2.ln() + r * r / s2;
            }
            if acc.is_finite() {
                0.5 * acc
            } else {
                f64::MAX
            }
        };

        // Start from a persistence of 0.9 split 1:8 between alpha and beta,
        // which is where financial return series overwhelmingly land.
        let start =
            [(sample_var * 0.1).max(1e-12).ln(), (0.9f64 / (0.999 - 0.9)).ln(), (0.1f64 / 0.9).ln()];
        let best = crate::optimization::nelder_mead(&negative_ll, &start, 0.5, 1e-10, 4000);
        let model = unpack(&best);
        if !model.omega.is_finite() || !(model.omega > 0.0) {
            return Err(GeomError::Degenerate("Garch11::fit: the optimiser produced no model"));
        }
        Ok(model)
    }

    /// Simulates `n` returns with Gaussian innovations.
    ///
    /// # Panics
    /// Panics if `n` is zero or the parameters are not non-negative.
    #[must_use]
    pub fn simulate(&self, n: usize, rng: &mut Rng) -> Vec<f64> {
        assert!(n > 0, "simulate requires n > 0");
        assert!(
            self.omega > 0.0 && self.alpha >= 0.0 && self.beta >= 0.0,
            "simulate requires omega > 0 and non-negative alpha, beta"
        );
        let burn = 500;
        let mut v = if self.persistence() < 1.0 {
            self.unconditional_variance()
        } else {
            self.omega
        };
        let mut previous = 0.0f64;
        let mut out = Vec::with_capacity(n);
        for t in 0..n + burn {
            v = self.omega + self.alpha * previous * previous + self.beta * v;
            let r = v.max(0.0).sqrt() * rng.next_gaussian();
            previous = r;
            if t >= burn {
                out.push(r);
            }
        }
        out
    }

    /// Variance forecasts `1..=h` steps ahead from the end of `returns`.
    ///
    /// Each step pulls the forecast toward the unconditional variance at rate
    /// `persistence`, so the sequence is monotone and converges there
    /// geometrically.
    ///
    /// # Panics
    /// Panics if `h` is zero or `returns` is empty.
    #[must_use]
    pub fn forecast_variance(&self, returns: &[f64], h: usize) -> Vec<f64> {
        assert!(h > 0, "forecast_variance requires h > 0");
        assert!(!returns.is_empty(), "forecast_variance requires returns");
        let filtered = self.conditional_variance(returns);
        let last_r = returns[returns.len() - 1];
        let last_v = filtered[filtered.len() - 1];
        // One step is exact; beyond that E[r^2] is replaced by its forecast.
        let mut v = self.omega + self.alpha * last_r * last_r + self.beta * last_v;
        let mut out = Vec::with_capacity(h);
        for _ in 0..h {
            out.push(v);
            v = self.omega + self.persistence() * v;
        }
        out
    }
}

/// The RiskMetrics exponentially weighted variance,
/// `v_t = lambda v_{t-1} + (1 - lambda) r_{t-1}^2`.
///
/// A GARCH(1,1) with `omega = 0` and unit persistence: no mean reversion, so
/// the variance wanders rather than settling.
///
/// # Panics
/// Panics unless `returns` is non-empty and `lambda` lies in `[0, 1)`.
#[must_use]
pub fn ewma_variance(returns: &[f64], lambda: f64) -> Vec<f64> {
    assert!(!returns.is_empty(), "ewma_variance requires returns");
    assert!((0.0..1.0).contains(&lambda), "ewma_variance requires lambda in [0, 1)");
    let mut v = returns[0] * returns[0];
    returns
        .iter()
        .enumerate()
        .map(|(t, _)| {
            if t > 0 {
                let prev = returns[t - 1];
                v = lambda * v + (1.0 - lambda) * prev * prev;
            }
            v
        })
        .collect()
}

/// Engle's ARCH LM test for conditional heteroskedasticity.
///
/// Regresses squared returns on their own lags; the statistic `n R^2` is
/// asymptotically chi-squared on `lags` degrees of freedom under the null of
/// no ARCH effect. A small p-value says the size of a return predicts the
/// size of the next one, which is precisely what a GARCH model is for.
///
/// # Errors
/// Returns an error if the series is too short or the regression is
/// degenerate.
pub fn arch_lm_test(returns: &[f64], lags: usize) -> Result<TestResult, GeomError> {
    if lags == 0 {
        return Err(GeomError::InvalidArgument("arch_lm_test requires at least one lag"));
    }
    let sq: Vec<f64> = returns.iter().map(|r| r * r).collect();
    if sq.len() < 3 * (lags + 2) {
        return Err(GeomError::InvalidArgument("arch_lm_test: series too short"));
    }
    let n = sq.len() - lags;
    let y: Vec<f64> = sq[lags..].to_vec();
    let cols: Vec<Vec<f64>> =
        (1..=lags).map(|l| (lags..sq.len()).map(|t| sq[t - l]).collect()).collect();
    let (_, _, rss) = ols_with_intercept(&cols, &y)?;
    let m = mean(&y);
    let tss: f64 = y.iter().map(|v| (v - m) * (v - m)).sum();
    if !(tss > 0.0) {
        return Err(GeomError::Degenerate("arch_lm_test: squared returns are constant"));
    }
    let r2 = 1.0 - rss / tss;
    let stat = n as f64 * r2;
    let df = lags as f64;
    Ok(TestResult { statistic: stat, p_value: 1.0 - ChiSquared::new(df).cdf(stat), df })
}

// ---------------------------------------------------------------------------
// Causality and cointegration
// ---------------------------------------------------------------------------

/// Tests whether `x` Granger-causes `y`: whether past `x` improves a forecast
/// of `y` that already uses past `y`.
///
/// An `F` test of the restricted regression of `y` on its own lags against
/// the unrestricted one that adds the lags of `x`. The name is a term of art
/// -- it is predictive precedence, not causation, and a common driver of both
/// series will produce it.
///
/// # Errors
/// Returns an error if the series differ in length, are too short, or either
/// regression is degenerate.
pub fn granger_causality(x: &[f64], y: &[f64], lags: usize) -> Result<TestResult, GeomError> {
    if x.len() != y.len() {
        return Err(GeomError::InvalidArgument("granger_causality requires equal lengths"));
    }
    if lags == 0 {
        return Err(GeomError::InvalidArgument("granger_causality requires at least one lag"));
    }
    let n = y.len();
    if n < 4 * lags + 8 {
        return Err(GeomError::InvalidArgument("granger_causality: series too short"));
    }
    let rows = n - lags;
    let target: Vec<f64> = y[lags..].to_vec();
    let own: Vec<Vec<f64>> =
        (1..=lags).map(|l| (lags..n).map(|t| y[t - l]).collect()).collect();
    let (_, _, rss_r) = ols_with_intercept(&own, &target)?;

    let mut full = own;
    for l in 1..=lags {
        full.push((lags..n).map(|t| x[t - l]).collect());
    }
    let (_, _, rss_u) = ols_with_intercept(&full, &target)?;

    let df1 = lags as f64;
    let df2 = (rows - (2 * lags + 1)) as f64;
    if !(rss_u > 0.0) || df2 <= 0.0 {
        return Err(GeomError::Degenerate("granger_causality: unrestricted fit is exact"));
    }
    let f = ((rss_r - rss_u) / df1) / (rss_u / df2);
    let p = if f > 0.0 { 1.0 - FDist::new(df1, df2).cdf(f) } else { 1.0 };
    Ok(TestResult { statistic: f, p_value: p, df: df1 })
}

/// The Engle-Granger two-step test for cointegration between `x` and `y`.
///
/// Regresses `y` on `x` with an intercept, then tests the residual for a unit
/// root. Rejecting means some linear combination of two individually
/// non-stationary series is stationary -- they share a stochastic trend.
///
/// The p-value comes from the module's Engle-Granger table rather than its
/// plain Dickey-Fuller one: the residual is fitted rather than observed, and the
/// regression has already worked to make it look stationary, so the null
/// distribution sits further left. Using the ordinary table here is a common
/// way to find cointegration that is not there.
///
/// # Errors
/// Returns an error if the series differ in length, are too short, or the
/// first-stage regression is degenerate.
pub fn cointegration_engle_granger(x: &[f64], y: &[f64]) -> Result<TestResult, GeomError> {
    if x.len() != y.len() {
        return Err(GeomError::InvalidArgument("cointegration requires equal lengths"));
    }
    if x.len() < 20 {
        return Err(GeomError::InvalidArgument("cointegration requires at least 20 observations"));
    }
    let (_, resid, _) = ols_with_intercept(&[x.to_vec()], y)?;
    let adf = adf_test(&resid, 1)?;
    Ok(TestResult {
        statistic: adf.statistic,
        p_value: interpolate_p(&EG_TAU_TABLE, adf.statistic),
        df: adf.df,
    })
}

/// A vector autoregression: each series regressed on `p` lags of every series.
#[derive(Debug, Clone, PartialEq)]
pub struct Var {
    /// `coeffs[i]` is the coefficient matrix on lag `i + 1`; entry `(r, c)`
    /// multiplies series `c` at that lag when predicting series `r`.
    pub coeffs: Vec<Matrix>,
    /// Per-series constant.
    pub intercept: Vec<f64>,
    /// Residual sum of squares per series, kept for the causality tests.
    rss: Vec<f64>,
    /// Rows used in the fit.
    rows: usize,
}

impl Var {
    /// Number of series.
    #[must_use]
    pub fn k(&self) -> usize {
        self.intercept.len()
    }

    /// Lag order.
    #[must_use]
    pub fn p(&self) -> usize {
        self.coeffs.len()
    }

    /// Fits by equation-by-equation least squares.
    ///
    /// Every equation has the same right-hand side, so the seemingly
    /// unrelated regression collapses to ordinary least squares run
    /// separately -- there is nothing to gain from estimating them jointly.
    ///
    /// `data[t]` holds all series at time `t`.
    ///
    /// # Errors
    /// Returns an error if the series are ragged, too short, or the design is
    /// rank deficient.
    pub fn fit(data: &[Vec<f64>], p: usize) -> Result<Self, GeomError> {
        if data.is_empty() || p == 0 {
            return Err(GeomError::InvalidArgument("Var::fit requires data and p >= 1"));
        }
        let k = data[0].len();
        if k == 0 || data.iter().any(|row| row.len() != k) {
            return Err(GeomError::InvalidArgument("Var::fit requires rectangular data"));
        }
        let n = data.len();
        let rows = n.saturating_sub(p);
        if rows <= k * p + 2 {
            return Err(GeomError::InvalidArgument("Var::fit: too few observations for the order"));
        }

        // One shared design matrix: a constant, then every series at every lag.
        let mut predictors: Vec<Vec<f64>> = Vec::with_capacity(k * p);
        for l in 1..=p {
            for j in 0..k {
                predictors.push((p..n).map(|t| data[t - l][j]).collect());
            }
        }

        let mut coeffs = vec![Matrix::zeros(k, k); p];
        let mut intercept = vec![0.0; k];
        let mut rss = vec![0.0; k];
        for i in 0..k {
            let y: Vec<f64> = (p..n).map(|t| data[t][i]).collect();
            let (beta, _, r) = ols_with_intercept(&predictors, &y)?;
            intercept[i] = beta[0];
            rss[i] = r;
            for l in 0..p {
                for j in 0..k {
                    coeffs[l].set(i, j, beta[1 + l * k + j]);
                }
            }
        }
        Ok(Self { coeffs, intercept, rss, rows })
    }

    /// `h`-step forecasts, each row one time step.
    ///
    /// # Errors
    /// Returns an error if `data` is too short or shaped wrongly.
    pub fn forecast(&self, data: &[Vec<f64>], h: usize) -> Result<Vec<Vec<f64>>, GeomError> {
        let k = self.k();
        if h == 0 {
            return Err(GeomError::InvalidArgument("Var::forecast requires h >= 1"));
        }
        if data.len() < self.p() || data.iter().any(|r| r.len() != k) {
            return Err(GeomError::InvalidArgument("Var::forecast: history too short or ragged"));
        }
        let mut history: Vec<Vec<f64>> = data[data.len() - self.p()..].to_vec();
        let mut out = Vec::with_capacity(h);
        for _ in 0..h {
            let mut next = self.intercept.clone();
            for (l, a) in self.coeffs.iter().enumerate() {
                let lagged = &history[history.len() - 1 - l];
                for r in 0..k {
                    for c in 0..k {
                        next[r] += a.get(r, c) * lagged[c];
                    }
                }
            }
            history.push(next.clone());
            out.push(next);
        }
        Ok(out)
    }

    /// The moving-average (impulse-response) matrices `Psi_0 ..= Psi_{h}`.
    ///
    /// `Psi_0` is the identity and `Psi_m = sum_l A_l Psi_{m-l}`. Entry
    /// `(r, c)` of `Psi_m` is the response of series `r` at horizon `m` to a
    /// unit shock in series `c` now.
    ///
    /// # Panics
    /// Panics if the coefficient matrices are not square and conformable.
    #[must_use]
    pub fn impulse_response(&self, h: usize) -> Vec<Matrix> {
        let k = self.k();
        let mut psi = vec![Matrix::zeros(k, k); h + 1];
        psi[0] = Matrix::identity(k);
        for m in 1..=h {
            let mut acc = Matrix::zeros(k, k);
            for (l, a) in self.coeffs.iter().enumerate() {
                if m > l {
                    let term = a.mul(&psi[m - l - 1]).expect("conformable by construction");
                    acc = acc.add(&term).expect("same shape");
                }
            }
            psi[m] = acc;
        }
        psi
    }

    /// A matrix of Granger-causality p-values: entry `(i, j)` tests whether
    /// series `j` helps predict series `i` given the rest of the system.
    ///
    /// The diagonal is set to 1: a series always predicts itself, so the
    /// question is not meaningful there.
    ///
    /// # Errors
    /// Returns an error if a restricted regression is degenerate.
    pub fn granger_matrix(&self, data: &[Vec<f64>]) -> Result<Matrix, GeomError> {
        let k = self.k();
        let p = self.p();
        let n = data.len();
        if n <= p {
            return Err(GeomError::InvalidArgument("granger_matrix: history too short"));
        }
        let mut out = Matrix::zeros(k, k);
        for i in 0..k {
            let y: Vec<f64> = (p..n).map(|t| data[t][i]).collect();
            for j in 0..k {
                if i == j {
                    out.set(i, j, 1.0);
                    continue;
                }
                // Restricted: every lag of every series except those of j.
                let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k * p - p);
                for l in 1..=p {
                    for c in 0..k {
                        if c != j {
                            cols.push((p..n).map(|t| data[t - l][c]).collect());
                        }
                    }
                }
                let (_, _, rss_r) = ols_with_intercept(&cols, &y)?;
                let rss_u = self.rss[i];
                let df1 = p as f64;
                let df2 = (self.rows - (k * p + 1)) as f64;
                if !(rss_u > 0.0) || df2 <= 0.0 {
                    out.set(i, j, 1.0);
                    continue;
                }
                let f = ((rss_r - rss_u) / df1) / (rss_u / df2);
                let pv = if f > 0.0 { 1.0 - FDist::new(df1, df2).cdf(f) } else { 1.0 };
                out.set(i, j, pv);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Decomposition and change detection
// ---------------------------------------------------------------------------

/// Additive seasonal decomposition into `(trend, seasonal, residual)`.
///
/// The trend is a centred moving average over one full period; the seasonal
/// component is the average detrended value at each phase, centred to sum to
/// zero; the residual is whatever is left. Near the ends, where the moving
/// average has no window, the trend is held at the nearest value it does
/// have -- so the three components add back to the input exactly at every
/// index, which is the property that makes the decomposition usable rather
/// than merely indicative.
///
/// # Panics
/// Panics unless `period >= 2` and the series covers at least two periods.
#[must_use]
pub fn seasonal_decompose_stl_lite(
    x: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert!(period >= 2, "seasonal_decompose_stl_lite requires a period of at least 2");
    assert!(
        x.len() >= 2 * period,
        "seasonal_decompose_stl_lite requires at least two full periods"
    );
    let n = x.len();
    let half = period / 2;

    // Centred moving average. An even period needs the half-weight end points
    // so the window is symmetric about an integer index rather than a half one.
    let mut trend = vec![f64::NAN; n];
    for t in half..n.saturating_sub(half) {
        let value = if period.is_multiple_of(2) {
            let inner: f64 = ((t - half + 1)..(t + half)).map(|u| x[u]).sum();
            (inner + 0.5 * x[t - half] + 0.5 * x[t + half]) / period as f64
        } else {
            ((t - half)..=(t + half)).map(|u| x[u]).sum::<f64>() / period as f64
        };
        trend[t] = value;
    }
    // Extend the ends by the nearest defined value.
    let first = trend.iter().position(|v| v.is_finite()).unwrap_or(0);
    let last = trend.iter().rposition(|v| v.is_finite()).unwrap_or(n - 1);
    for t in 0..first {
        trend[t] = trend[first];
    }
    for t in last + 1..n {
        trend[t] = trend[last];
    }

    // Seasonal averages by phase, over the region where the trend was real.
    let mut sums = vec![0.0; period];
    let mut counts = vec![0usize; period];
    for t in first..=last {
        sums[t % period] += x[t] - trend[t];
        counts[t % period] += 1;
    }
    let mut phase: Vec<f64> = (0..period)
        .map(|i| if counts[i] > 0 { sums[i] / counts[i] as f64 } else { 0.0 })
        .collect();
    // Centre so the seasonal component carries no level of its own; otherwise
    // it and the trend are not separately identified.
    let offset = phase.iter().sum::<f64>() / period as f64;
    for v in &mut phase {
        *v -= offset;
    }

    let seasonal: Vec<f64> = (0..n).map(|t| phase[t % period]).collect();
    let residual: Vec<f64> =
        (0..n).map(|t| x[t] - trend[t] - seasonal[t]).collect();
    (trend, seasonal, residual)
}

/// Sum of squared deviations of `x[a..b]` from its own mean: the cost of
/// describing that segment by a single level.
fn segment_cost(prefix: &[f64], prefix_sq: &[f64], a: usize, b: usize) -> f64 {
    if b <= a {
        return 0.0;
    }
    let n = (b - a) as f64;
    let s = prefix[b] - prefix[a];
    let ss = prefix_sq[b] - prefix_sq[a];
    (ss - s * s / n).max(0.0)
}

/// Prefix sums of `x` and of `x^2`, for constant-time segment costs.
fn prefix_sums(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut p = Vec::with_capacity(x.len() + 1);
    let mut q = Vec::with_capacity(x.len() + 1);
    p.push(0.0);
    q.push(0.0);
    for &v in x {
        p.push(p[p.len() - 1] + v);
        q.push(q[q.len() - 1] + v * v);
    }
    (p, q)
}

/// Change-in-mean detection by PELT (pruned exact linear time).
///
/// Finds the segmentation minimising the total within-segment sum of squares
/// plus `penalty` per changepoint. Unlike binary segmentation this is exact:
/// dynamic programming considers every segmentation, and the pruning step
/// discards only candidates that provably cannot start an optimal segment,
/// so the answer is the global optimum rather than a greedy approximation.
///
/// Returns the interior changepoint indices, each the first index of a new
/// segment, in increasing order.
///
/// # Panics
/// Panics if `penalty` is negative.
#[must_use]
pub fn changepoint_pelt(x: &[f64], penalty: f64) -> Vec<usize> {
    assert!(penalty >= 0.0, "changepoint_pelt requires a non-negative penalty");
    let n = x.len();
    if n < 2 {
        return Vec::new();
    }
    let (prefix, prefix_sq) = prefix_sums(x);

    let mut best = vec![f64::INFINITY; n + 1];
    let mut last = vec![0usize; n + 1];
    best[0] = -penalty;
    // Candidate starting points that have not been pruned away.
    let mut candidates: Vec<usize> = vec![0];

    for t in 1..=n {
        let mut best_cost = f64::INFINITY;
        let mut best_start = 0usize;
        for &s in &candidates {
            let cost = best[s] + segment_cost(&prefix, &prefix_sq, s, t) + penalty;
            if cost < best_cost {
                best_cost = cost;
                best_start = s;
            }
        }
        best[t] = best_cost;
        last[t] = best_start;

        // Pruning: a start whose own cost already exceeds the best total can
        // never be beaten into an optimal segmentation later, because the
        // segment cost only grows as the segment lengthens.
        candidates.retain(|&s| best[s] + segment_cost(&prefix, &prefix_sq, s, t) <= best[t]);
        candidates.push(t);
    }

    let mut points = Vec::new();
    let mut t = n;
    while t > 0 {
        let s = last[t];
        if s > 0 {
            points.push(s);
        }
        t = s;
    }
    points.reverse();
    points
}

/// Change-in-mean detection by recursive binary segmentation.
///
/// Splits at the point giving the largest reduction in sum of squares, then
/// recurses into both halves, stopping at `max_k` changepoints. Greedy rather
/// than exact -- it can miss a pair of changes whose individual effects
/// cancel -- but it is fast and needs no penalty to be chosen.
///
/// Returns changepoint indices in increasing order.
#[must_use]
pub fn changepoint_binary_segmentation(x: &[f64], max_k: usize) -> Vec<usize> {
    let n = x.len();
    if n < 4 || max_k == 0 {
        return Vec::new();
    }
    let (prefix, prefix_sq) = prefix_sums(x);
    let mut points: Vec<usize> = Vec::new();
    let mut segments: Vec<(usize, usize)> = vec![(0, n)];

    for _ in 0..max_k {
        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, usize, usize)> = None;
        for (idx, &(a, b)) in segments.iter().enumerate() {
            if b - a < 4 {
                continue;
            }
            let whole = segment_cost(&prefix, &prefix_sq, a, b);
            for s in a + 2..b - 1 {
                let gain = whole
                    - segment_cost(&prefix, &prefix_sq, a, s)
                    - segment_cost(&prefix, &prefix_sq, s, b);
                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((idx, s, b));
                }
            }
        }
        let Some((idx, s, b)) = best_split else { break };
        let (a, _) = segments[idx];
        segments[idx] = (a, s);
        segments.push((s, b));
        points.push(s);
    }
    points.sort_unstable();
    points
}

/// Two-sided cumulative sum control statistics, `(upper, lower)`.
///
/// `S+_t = max(0, S+_{t-1} + (x_t - target) - k)` and the mirror image for
/// the lower arm. The slack `k` is what stops the statistic drifting on
/// ordinary noise: with `k` set to half the shift worth detecting, the
/// statistic stays near zero while the process is on target and climbs
/// roughly linearly once it is not.
///
/// # Panics
/// Panics if `k` is negative.
#[must_use]
pub fn cusum(x: &[f64], target: f64, k: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(k >= 0.0, "cusum requires a non-negative slack");
    let mut up = 0.0f64;
    let mut down = 0.0f64;
    let mut hi = Vec::with_capacity(x.len());
    let mut lo = Vec::with_capacity(x.len());
    for &v in x {
        let d = v - target;
        up = (up + d - k).max(0.0);
        down = (down - d - k).max(0.0);
        hi.push(up);
        lo.push(down);
    }
    (hi, lo)
}

/// The matrix profile of `x` for subsequences of length `m`:
/// `(distance to the nearest other subsequence, its index)`.
///
/// Distances are z-normalised Euclidean, so a match is about shape rather
/// than level or amplitude. Overlapping neighbours are excluded -- a
/// subsequence's closest match is always the one shifted by one sample, which
/// says nothing -- using the usual exclusion zone of half the window.
///
/// The smallest entries locate the repeated motifs; the largest locates the
/// discord, the least-like-anything-else stretch.
///
/// # Panics
/// Panics unless `m >= 2` and the series holds at least two non-overlapping
/// windows.
#[must_use]
pub fn matrix_profile_lite(x: &[f64], m: usize) -> (Vec<f64>, Vec<usize>) {
    assert!(m >= 2, "matrix_profile_lite requires m >= 2");
    assert!(x.len() >= 2 * m, "matrix_profile_lite requires at least two windows");
    let count = x.len() - m + 1;
    let exclusion = (m / 2).max(1);

    // z-normalise each window once rather than inside the pair loop.
    let normalised: Vec<Vec<f64>> = (0..count)
        .map(|i| {
            let w = &x[i..i + m];
            let mu = mean(w);
            let var = w.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / m as f64;
            let sd = var.sqrt();
            if sd <= 1e-12 {
                vec![0.0; m]
            } else {
                w.iter().map(|v| (v - mu) / sd).collect()
            }
        })
        .collect();

    let mut profile = vec![f64::INFINITY; count];
    let mut index = vec![0usize; count];
    for i in 0..count {
        for j in 0..count {
            if i.abs_diff(j) < exclusion {
                continue;
            }
            let d: f64 = normalised[i]
                .iter()
                .zip(&normalised[j])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if d < profile[i] {
                profile[i] = d;
                index[i] = j;
            }
        }
    }
    (profile, index)
}

// ---------------------------------------------------------------------------
// Complexity measures
// ---------------------------------------------------------------------------

/// Counts template matches of length `len` under the Chebyshev metric at
/// tolerance `tol`, optionally excluding the self-match.
fn template_matches(x: &[f64], len: usize, tol: f64, include_self: bool) -> (usize, usize) {
    let count = x.len() + 1 - len;
    let mut matches = 0usize;
    let mut pairs = 0usize;
    for i in 0..count {
        for j in 0..count {
            if !include_self && i == j {
                continue;
            }
            pairs += 1;
            let d = (0..len)
                .map(|k| (x[i + k] - x[j + k]).abs())
                .fold(0.0f64, f64::max);
            if d <= tol {
                matches += 1;
            }
        }
    }
    (matches, pairs)
}

/// Sample entropy: the negative log probability that two sequences matching
/// for `m` points go on matching for `m + 1`.
///
/// `tol` is given as a multiple of the series standard deviation. Unlike
/// [`approximate_entropy`] the self-match is excluded, which removes the bias
/// that otherwise makes a short series look more regular than it is.
///
/// Returns infinity when no `m+1`-length match occurs at all, which is the
/// honest answer -- the estimator has run out of data rather than found zero
/// probability.
///
/// # Panics
/// Panics unless `m >= 1`, `tol > 0`, and the series holds at least `m + 2`
/// points.
#[must_use]
pub fn sample_entropy(x: &[f64], m: usize, tol: f64) -> f64 {
    assert!(m >= 1, "sample_entropy requires m >= 1");
    assert!(tol > 0.0, "sample_entropy requires a positive tolerance");
    assert!(x.len() >= m + 2, "sample_entropy requires at least m + 2 observations");
    let mu = mean(x);
    let sd = (x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / x.len() as f64).sqrt();
    let r = tol * sd;
    if r <= 0.0 {
        return 0.0;
    }
    let (b, _) = template_matches(x, m, r, false);
    let (a, _) = template_matches(x, m + 1, r, false);
    if a == 0 || b == 0 {
        return f64::INFINITY;
    }
    -((a as f64) / (b as f64)).ln()
}

/// Approximate entropy, the older cousin of [`sample_entropy`].
///
/// Includes the self-match, which guarantees the logarithm is defined but
/// biases the estimate toward regularity, the more so the shorter the series.
/// Kept because it is what a great deal of published work reports.
///
/// # Panics
/// Panics under the same conditions as [`sample_entropy`].
#[must_use]
pub fn approximate_entropy(x: &[f64], m: usize, tol: f64) -> f64 {
    assert!(m >= 1, "approximate_entropy requires m >= 1");
    assert!(tol > 0.0, "approximate_entropy requires a positive tolerance");
    assert!(x.len() >= m + 2, "approximate_entropy requires at least m + 2 observations");
    let mu = mean(x);
    let sd = (x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / x.len() as f64).sqrt();
    let r = tol * sd;
    if r <= 0.0 {
        return 0.0;
    }
    let phi = |len: usize| -> f64 {
        let count = x.len() + 1 - len;
        let mut acc = 0.0;
        for i in 0..count {
            let mut hits = 0usize;
            for j in 0..count {
                let d = (0..len).map(|k| (x[i + k] - x[j + k]).abs()).fold(0.0f64, f64::max);
                if d <= r {
                    hits += 1;
                }
            }
            acc += ((hits as f64) / (count as f64)).ln();
        }
        acc / count as f64
    };
    phi(m) - phi(m + 1)
}

/// Permutation entropy: the Shannon entropy of the ordinal patterns of length
/// `order` sampled at spacing `delay`, normalised to `[0, 1]`.
///
/// Only the ranking within each window matters, so the measure is invariant
/// to any monotone transformation of the series and needs no tolerance
/// parameter. A monotone series visits one pattern and scores 0; independent
/// noise visits all `order!` patterns equally and scores 1.
///
/// # Panics
/// Panics unless `order` is between 2 and 8, `delay >= 1`, and the series is
/// long enough to hold at least two windows.
#[must_use]
pub fn permutation_entropy(x: &[f64], order: usize, delay: usize) -> f64 {
    assert!((2..=8).contains(&order), "permutation_entropy requires 2 <= order <= 8");
    assert!(delay >= 1, "permutation_entropy requires delay >= 1");
    let span = (order - 1) * delay;
    assert!(x.len() > span + 1, "permutation_entropy requires a longer series");

    let mut counts: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    let windows = x.len() - span;
    for t in 0..windows {
        let vals: Vec<f64> = (0..order).map(|k| x[t + k * delay]).collect();
        // The ordinal pattern is the permutation that sorts the window.
        let mut idx: Vec<usize> = (0..order).collect();
        idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));
        *counts.entry(idx).or_insert(0) += 1;
    }

    let total = windows as f64;
    let h: f64 = counts
        .values()
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum();
    let max = (1..=order).map(|i| i as f64).product::<f64>().ln();
    if max <= 0.0 {
        0.0
    } else {
        h / max
    }
}

/// One IAAFT surrogate: a series with the same amplitude distribution as `x`
/// and, as closely as the two constraints allow, the same power spectrum.
///
/// The iteration alternates two projections -- impose the target spectrum in
/// the frequency domain, then impose the target amplitudes by rank-ordering
/// in the time domain -- neither of which preserves the other, so it
/// converges to a compromise rather than a fixed point.
fn iaaft_surrogate(x: &[f64], rng: &mut Rng, iterations: usize) -> Vec<f64> {
    let n = x.len();
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let target_amplitude: Vec<f64> = crate::transforms::fft::fft_any(
        &x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>(),
    )
    .iter()
    .map(|c| c.norm())
    .collect();

    // Start from a random shuffle: the right values in the wrong order.
    let mut y = x.to_vec();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        y.swap(i, j);
    }

    for _ in 0..iterations {
        // Impose the spectrum, keeping the current phases.
        let spectrum =
            crate::transforms::fft::fft_any(&y.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
        let adjusted: Vec<Complex> = spectrum
            .iter()
            .zip(&target_amplitude)
            .map(|(c, &a)| {
                let norm = c.norm();
                if norm <= 1e-300 {
                    Complex::new(a, 0.0)
                } else {
                    Complex::new(c.re / norm * a, c.im / norm * a)
                }
            })
            .collect();
        let back = crate::transforms::fft::ifft_any(&adjusted);
        let mut candidate: Vec<f64> = back.iter().map(|c| c.re).collect();

        // Impose the amplitudes: replace each value by the sorted original of
        // the same rank, which restores the distribution exactly.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            candidate[a].partial_cmp(&candidate[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, &pos) in order.iter().enumerate() {
            candidate[pos] = sorted[rank];
        }
        y = candidate;
    }
    y
}

/// A surrogate-data test: how extreme `statistic(x)` is against the
/// distribution it takes on IAAFT surrogates of `x`.
///
/// The surrogates share the series' amplitude distribution and power
/// spectrum, hence all of its linear structure. Rejecting therefore points at
/// something a linear Gaussian process could not produce -- nonlinearity --
/// rather than merely at "not white noise", which is what a test against
/// shuffled data would show.
///
/// Returns the two-sided rank p-value `(1 + #{|s_i - mean| >= |s_x - mean|}) /
/// (1 + n)`, which is exact for finite `n` rather than asymptotic.
///
/// # Panics
/// Panics if `n_surrogates` is zero or the series is shorter than four points.
#[must_use]
pub fn surrogate_test_iaaft(
    x: &[f64],
    statistic: &dyn Fn(&[f64]) -> f64,
    n_surrogates: usize,
    rng: &mut Rng,
) -> f64 {
    assert!(n_surrogates > 0, "surrogate_test_iaaft requires at least one surrogate");
    assert!(x.len() >= 4, "surrogate_test_iaaft requires at least four observations");
    let observed = statistic(x);
    let values: Vec<f64> = (0..n_surrogates)
        .map(|_| statistic(&iaaft_surrogate(x, rng, 40)))
        .collect();
    let m = mean(&values);
    let reference = (observed - m).abs();
    let extreme = values.iter().filter(|&&v| (v - m).abs() >= reference).count();
    (1 + extreme) as f64 / (1 + n_surrogates) as f64
}

// ---------------------------------------------------------------------------
// State space
// ---------------------------------------------------------------------------

/// The local level model: `x_t = mu_t + e_t`, `mu_t = mu_{t-1} + n_t`.
///
/// Returns `(smoothed level, signal variance, observation variance)`. The two
/// variances are estimated by maximising the Gaussian likelihood from the
/// Kalman filter; only their ratio -- the signal-to-noise ratio, or hyper-
/// parameter `q` -- affects the filtered path, so it is that ratio the
/// optimiser searches over, with the overall scale then available in closed
/// form.
///
/// The model is the state-space form of simple exponential smoothing: the
/// steady-state Kalman gain *is* the smoothing constant, so an estimated `q`
/// and an estimated `alpha` carry the same information.
///
/// # Errors
/// Returns an error for a series shorter than five points or with no
/// variation.
pub fn state_space_local_level(x: &[f64]) -> Result<(Vec<f64>, f64, f64), GeomError> {
    let n = x.len();
    if n < 5 {
        return Err(GeomError::InvalidArgument("state_space_local_level requires n >= 5"));
    }
    let mu = mean(x);
    let var: f64 = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / n as f64;
    if !(var > 0.0) {
        return Err(GeomError::Degenerate("state_space_local_level: series is constant"));
    }

    // Run the filter with the observation variance fixed at one; the
    // likelihood is then concentrated and the scale recovered afterwards.
    let filter = |q: f64| -> (f64, f64) {
        let (mut a, mut p) = (x[0], 1e6);
        let mut acc_v = 0.0;
        let mut acc_f = 0.0;
        for &v in x.iter() {
            let f = p + 1.0;
            let innovation = v - a;
            acc_v += innovation * innovation / f;
            acc_f += f.ln();
            let k = p / f;
            a += k * innovation;
            p = p * (1.0 - k) + q;
        }
        (acc_v, acc_f)
    };
    let negative_ll = |theta: &[f64]| -> f64 {
        let q = theta[0].clamp(-30.0, 30.0).exp();
        let (acc_v, acc_f) = filter(q);
        if !acc_v.is_finite() || !acc_f.is_finite() || acc_v <= 0.0 {
            return f64::MAX;
        }
        // Concentrated likelihood: profile out the common scale.
        0.5 * (acc_f + n as f64 * (acc_v / n as f64).ln())
    };
    let best = crate::optimization::nelder_mead(&negative_ll, &[0.0], 0.5, 1e-10, 800);
    let q = best[0].clamp(-30.0, 30.0).exp();
    let (acc_v, _) = filter(q);
    let sigma2_eps = acc_v / n as f64;
    let sigma2_eta = q * sigma2_eps;

    // A second pass at the fitted parameters, this time keeping the filtered
    // states, then the Rauch-Tung-Striebel backward recursion to smooth them.
    let mut a_pred = vec![0.0; n];
    let mut p_pred = vec![0.0; n];
    let mut a_filt = vec![0.0; n];
    let mut p_filt = vec![0.0; n];
    let (mut a, mut p) = (x[0], 1e6 * sigma2_eps);
    for t in 0..n {
        a_pred[t] = a;
        p_pred[t] = p;
        let f = p + sigma2_eps;
        let k = p / f;
        a_filt[t] = a + k * (x[t] - a);
        p_filt[t] = p * (1.0 - k);
        a = a_filt[t];
        p = p_filt[t] + sigma2_eta;
    }
    let mut smoothed = a_filt.clone();
    for t in (0..n - 1).rev() {
        // The transition is the identity, so the smoother gain is just the
        // ratio of filtered to predicted variance.
        let gain = if p_pred[t + 1] > 0.0 { p_filt[t] / p_pred[t + 1] } else { 0.0 };
        smoothed[t] = a_filt[t] + gain * (smoothed[t + 1] - a_pred[t + 1]);
    }
    Ok((smoothed, sigma2_eta, sigma2_eps))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    fn white_noise(n: usize, sd: f64, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| sd * rng.next_gaussian()).collect()
    }

    fn random_walk(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut acc = 0.0;
        (0..n)
            .map(|_| {
                acc += rng.next_gaussian();
                acc
            })
            .collect()
    }

    // -----------------------------------------------------------------
    // Correlation structure
    // -----------------------------------------------------------------

    #[test]
    fn acf_is_a_correlation_and_starts_at_one() {
        for seed in [1u64, 2, 3] {
            let x = white_noise(600, 2.0, seed);
            let r = acf(&x, 20);
            assert_eq!(r[0], 1.0);
            assert!(r.iter().all(|v| v.abs() <= 1.0), "an autocorrelation left [-1, 1]");
            // Under white noise each lag is roughly N(0, 1/n), so exceeding
            // four standard errors at any of twenty lags would be remarkable.
            let se = 1.0 / (x.len() as f64).sqrt();
            assert!(
                r[1..].iter().all(|v| v.abs() < 4.0 * se),
                "white noise showed structure: {:?}",
                &r[1..5]
            );
        }
    }

    #[test]
    fn ar1_autocorrelation_decays_at_the_coefficient() {
        // For x_t = phi x_{t-1} + e_t the theoretical acf is exactly phi^k.
        for phi in [0.7f64, -0.5, 0.3] {
            let model = Arma::new(vec![phi], vec![], 1.0, 0.0);
            let mut rng = Rng::new(0x71_0001 + (phi.abs() * 1000.0) as u64);
            let x = model.simulate(20_000, &mut rng);
            let r = acf(&x, 5);
            for k in 1..=5 {
                assert!(
                    (r[k] - phi.powi(k as i32)).abs() < 0.05,
                    "phi = {phi}, lag {k}: {} against {}",
                    r[k],
                    phi.powi(k as i32)
                );
            }
        }
    }

    #[test]
    fn pacf_cuts_off_beyond_the_autoregressive_order() {
        // The defining property of the partial autocorrelation: for an AR(p)
        // it is zero past lag p, while the plain acf decays forever.
        let model = Arma::new(vec![0.5, 0.3], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0002);
        let x = model.simulate(20_000, &mut rng);
        let pa = pacf(&x, 10);
        let r = acf(&x, 10);
        assert!((pa[1] - r[1]).abs() < 1e-12, "the first partial must equal the first ordinary");
        assert!(pa[2].abs() > 0.2, "the lag-2 partial should be substantial, got {}", pa[2]);
        let se = 1.0 / (x.len() as f64).sqrt();
        for k in 3..=10 {
            assert!(pa[k].abs() < 4.0 * se, "lag {k} partial {} did not cut off", pa[k]);
        }
        // The ordinary acf, by contrast, is still clearly non-zero at lag 3.
        assert!(r[3].abs() > 8.0 * se, "the acf should not have cut off");
    }

    #[test]
    fn pacf_of_an_ma1_matches_its_closed_form() {
        // For an MA(1) with coefficient theta the partial autocorrelations are
        // -(-theta)^k (1 - theta^2) / (1 - theta^{2(k+1)}).
        let theta = 0.6f64;
        let model = Arma::new(vec![], vec![theta], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0003);
        let x = model.simulate(40_000, &mut rng);
        let pa = pacf(&x, 4);
        for k in 1..=4i32 {
            let expected = -(-theta).powi(k) * (1.0 - theta * theta)
                / (1.0 - theta.powi(2 * (k + 1)));
            assert!(
                (pa[k as usize] - expected).abs() < 0.03,
                "lag {k}: {} against {expected}",
                pa[k as usize]
            );
        }
    }

    #[test]
    fn cross_correlation_peaks_at_the_true_lead() {
        let base = white_noise(2000, 1.0, 0x71_0004);
        let shift = 7usize;
        // y trails x by `shift` steps.
        let mut y = vec![0.0; base.len()];
        y[shift..].copy_from_slice(&base[..base.len() - shift]);
        let cc = cross_correlation_lags(&base, &y, 20);
        let peak = cc
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i as isize - 20)
            .unwrap();
        assert_eq!(peak, shift as isize, "the peak landed at lag {peak}, not {shift}");
        assert!(cc[20 + shift] > 0.9);
    }

    #[test]
    fn ljung_box_separates_white_noise_from_an_autoregression() {
        let noise = white_noise(500, 1.0, 0x71_0005);
        let clean = ljung_box(&noise, 10);
        assert!(clean.p_value > 0.05, "white noise was flagged, p = {}", clean.p_value);
        assert_eq!(clean.df, 10.0);

        let model = Arma::new(vec![0.6], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0006);
        let structured = model.simulate(500, &mut rng);
        let flagged = ljung_box(&structured, 10);
        assert!(flagged.p_value < 1e-6, "an AR(1) went undetected, p = {}", flagged.p_value);
        assert!(flagged.statistic > clean.statistic);
    }

    // -----------------------------------------------------------------
    // The regression kernel everything else is built on
    // -----------------------------------------------------------------

    #[test]
    fn the_normal_equation_solver_agrees_with_a_householder_qr() {
        // Every test here rests on this regression, and it takes the cheaper
        // of two routes to the same answer. On a well-conditioned design the
        // two must agree to near machine precision; the QR is the reference
        // because it never forms X'X.
        let mut rng = Rng::new(0x71_00A1);
        let n = 400usize;
        let cols: Vec<Vec<f64>> = (0..4)
            .map(|j| (0..n).map(|i| (i as f64 * (0.13 + 0.07 * j as f64)).sin()).collect())
            .collect();
        let truth = [2.0, -1.5, 0.75, 3.0, -0.25];
        let y: Vec<f64> = (0..n)
            .map(|i| {
                truth[0]
                    + (0..4).map(|j| truth[j + 1] * cols[j][i]).sum::<f64>()
                    + 0.05 * rng.next_gaussian()
            })
            .collect();

        let (beta, resid, rss) = ols_with_intercept(&cols, &y).unwrap();
        let mut design = Matrix::zeros(n, 5);
        for i in 0..n {
            design.set(i, 0, 1.0);
            for j in 0..4 {
                design.set(i, j + 1, cols[j][i]);
            }
        }
        let reference = crate::linalg::qr::least_squares(&design, &y).unwrap();
        for j in 0..5 {
            assert!(
                (beta[j] - reference[j]).abs() < 1e-9,
                "coefficient {j}: normal equations {} against QR {}",
                beta[j],
                reference[j]
            );
            assert!((beta[j] - truth[j]).abs() < 0.02, "coefficient {j} came out {}", beta[j]);
        }
        // The residuals must be orthogonal to every column of the design --
        // the defining property of a least-squares fit.
        assert!(resid.iter().sum::<f64>().abs() < 1e-8, "the residuals have a mean");
        for c in &cols {
            let dot: f64 = resid.iter().zip(c).map(|(r, v)| r * v).sum();
            assert!(dot.abs() < 1e-8, "the residuals correlate with a regressor: {dot}");
        }
        assert!((rss - resid.iter().map(|r| r * r).sum::<f64>()).abs() < 1e-12);
    }

    #[test]
    fn the_regression_reports_a_rank_deficient_design_rather_than_guessing() {
        let base: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        // An exact duplicate column leaves the coefficients unidentified.
        let doubled: Vec<f64> = base.iter().map(|v| 2.0 * v).collect();
        let y: Vec<f64> = base.iter().map(|v| 3.0 * v).collect();
        assert!(ols_with_intercept(&[base.clone(), doubled], &y).is_err());
        // A constant regressor duplicates the intercept.
        assert!(ols_with_intercept(&[vec![1.0; 100]], &y).is_err());
        // Too few rows for the parameters.
        assert!(ols_with_intercept(&[vec![1.0, 2.0], vec![3.0, 1.0]], &[1.0, 2.0]).is_err());
        // Ragged input.
        assert!(ols_with_intercept(&[vec![1.0; 50]], &y).is_err());
    }

    // -----------------------------------------------------------------
    // Differencing
    // -----------------------------------------------------------------

    #[test]
    fn differencing_round_trips_at_every_order() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) * 0.7 + (i as f64 * 0.3).sin() * 5.0).collect();
        for d in 1..=4usize {
            let diffed = difference(&x, d);
            assert_eq!(diffed.len(), x.len() - d);
            let initial: Vec<f64> = (0..d).map(|j| difference(&x, j)[0]).collect();
            let back = undifference(&diffed, &initial);
            assert_eq!(back.len(), x.len());
            for (a, b) in back.iter().zip(&x) {
                assert!((a - b).abs() < 1e-9, "round trip at d = {d} lost {a} vs {b}");
            }
        }
    }

    #[test]
    fn differencing_annihilates_a_polynomial_of_matching_degree() {
        // The d-th difference of a degree-d polynomial is the constant d!
        // times the leading coefficient, and the (d+1)-th is zero.
        for d in 1..=4usize {
            let x: Vec<f64> = (0..30).map(|i| (i as f64).powi(d as i32)).collect();
            let flat = difference(&x, d);
            let expected: f64 = (1..=d).map(|i| i as f64).product();
            assert!(
                flat.iter().all(|v| (v - expected).abs() < 1e-6),
                "the {d}-th difference is not the constant {expected}: {flat:?}"
            );
            let gone = difference(&x, d + 1);
            assert!(gone.iter().all(|v| v.abs() < 1e-6), "degree {d} survived {} differences", d + 1);
        }
    }

    #[test]
    fn seasonal_differencing_removes_a_pure_seasonal_pattern() {
        let s = 12usize;
        let x: Vec<f64> =
            (0..60).map(|i| ((i % s) as f64 * 0.5).sin() * 10.0 + 3.0).collect();
        let d = seasonal_difference(&x, s);
        assert_eq!(d.len(), x.len() - s);
        assert!(d.iter().all(|v| v.abs() < 1e-9), "a pure seasonal pattern survived");
    }

    // -----------------------------------------------------------------
    // Stationarity, with the two tests pointing opposite ways
    // -----------------------------------------------------------------

    #[test]
    fn adf_and_kpss_disagree_in_the_right_direction() {
        // Two series, two tests, four verdicts. ADF's null is a unit root and
        // KPSS's is stationarity, so a correct pair of tests gives opposite
        // rejections on the same data.
        let walk = random_walk(800, 0x71_0007);
        let model = Arma::new(vec![0.5], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0008);
        let stable = model.simulate(800, &mut rng);

        let adf_walk = adf_test(&walk, 2).unwrap();
        let adf_stable = adf_test(&stable, 2).unwrap();
        assert!(adf_walk.p_value > 0.10, "ADF rejected a random walk, p = {}", adf_walk.p_value);
        // 0.010 is the smallest value the table carries, so a decisive
        // rejection lands exactly on it; the statistic itself is the sharper
        // statement.
        assert!(
            adf_stable.p_value <= 0.01,
            "ADF failed to reject on a stationary AR(1), p = {}",
            adf_stable.p_value
        );
        assert!(
            adf_stable.statistic < -3.43,
            "tau = {} is not past the 1% critical value",
            adf_stable.statistic
        );
        assert!(
            adf_stable.statistic < adf_walk.statistic,
            "the stationary series should give the more negative tau"
        );

        let kpss_walk = kpss_test(&walk).unwrap();
        let kpss_stable = kpss_test(&stable).unwrap();
        assert!(
            kpss_walk.p_value < 0.05,
            "KPSS accepted stationarity for a random walk, p = {}",
            kpss_walk.p_value
        );
        assert!(
            kpss_stable.p_value > 0.05,
            "KPSS rejected stationarity for an AR(1), p = {}",
            kpss_stable.p_value
        );
        assert!(kpss_walk.statistic > kpss_stable.statistic);
    }

    #[test]
    fn differencing_a_random_walk_makes_it_stationary_to_both_tests() {
        let walk = random_walk(800, 0x71_0009);
        let d = difference(&walk, 1);
        assert!(adf_test(&d, 2).unwrap().p_value <= 0.01, "ADF still sees a unit root");
        assert!(adf_test(&d, 2).unwrap().statistic < -3.43);
        assert!(kpss_test(&d).unwrap().p_value > 0.05, "KPSS still rejects stationarity");
    }

    #[test]
    fn the_p_value_table_is_monotone_and_clamps_rather_than_extrapolates() {
        let mut previous = 0.0;
        for i in 0..=80 {
            let tau = -5.0 + i as f64 * 0.1;
            let p = interpolate_p(&DF_TAU_TABLE, tau);
            assert!(p >= previous - 1e-12, "the p-value fell at tau = {tau}");
            assert!((0.0..=1.0).contains(&p));
            previous = p;
        }
        // Far outside the table the answer is the tabulated end, not an
        // extrapolated number the table cannot support.
        assert_eq!(interpolate_p(&DF_TAU_TABLE, -50.0), 0.010);
        assert_eq!(interpolate_p(&DF_TAU_TABLE, 50.0), 0.990);
    }

    #[test]
    fn stationarity_tests_reject_impossible_input() {
        assert!(adf_test(&[1.0, 2.0, 3.0], 5).is_err());
        assert!(kpss_test(&[1.0, 2.0]).is_err());
        assert!(kpss_test(&[3.0; 30]).is_err());
    }

    // -----------------------------------------------------------------
    // ARMA
    // -----------------------------------------------------------------

    #[test]
    fn impulse_response_of_an_ar1_is_the_geometric_sequence() {
        let phi = 0.6f64;
        let psi = Arma::new(vec![phi], vec![], 1.0, 0.0).impulse_response(10);
        for (j, p) in psi.iter().enumerate() {
            assert!((p - phi.powi(j as i32)).abs() < 1e-12, "psi_{j} = {p}");
        }
        // An MA(q) has exactly q + 1 non-zero weights and nothing beyond.
        let ma = Arma::new(vec![], vec![0.4, -0.2], 1.0, 0.0).impulse_response(6);
        assert_eq!(ma[0], 1.0);
        assert!((ma[1] - 0.4).abs() < 1e-12);
        assert!((ma[2] + 0.2).abs() < 1e-12);
        assert!(ma[3..].iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn the_spectral_density_integrates_to_the_process_variance() {
        // Integral over [-pi, pi] of f(w) dw = gamma_0 = sigma2 sum psi_j^2.
        // The frequency-domain and time-domain descriptions of second-order
        // structure have to agree.
        for model in [
            Arma::new(vec![0.6], vec![], 2.0, 0.0),
            Arma::new(vec![], vec![0.5, -0.3], 1.5, 0.0),
            Arma::new(vec![0.4, 0.2], vec![0.3], 1.0, 0.0),
        ] {
            let m = 40_000usize;
            let freqs: Vec<f64> = (0..m)
                .map(|i| -std::f64::consts::PI + (i as f64 + 0.5) * 2.0 * std::f64::consts::PI / m as f64)
                .collect();
            let dens = model.spectral_density(&freqs);
            let integral: f64 =
                dens.iter().sum::<f64>() * 2.0 * std::f64::consts::PI / m as f64;

            let psi = model.impulse_response(4000);
            let gamma0 = model.sigma2 * psi.iter().map(|p| p * p).sum::<f64>();
            assert!(
                close(integral, gamma0, 1e-5),
                "spectral integral {integral} against psi-weight variance {gamma0}"
            );
            assert!(dens.iter().all(|&v| v >= 0.0), "a spectral density went negative");
        }
    }

    #[test]
    fn spectral_density_peaks_where_the_autoregressive_root_sits() {
        // A positive phi concentrates power at low frequency, a negative one
        // at the Nyquist end. Same magnitude, mirrored spectrum.
        let freqs: Vec<f64> = (0..200).map(|i| i as f64 * std::f64::consts::PI / 199.0).collect();
        let positive = Arma::new(vec![0.8], vec![], 1.0, 0.0).spectral_density(&freqs);
        let negative = Arma::new(vec![-0.8], vec![], 1.0, 0.0).spectral_density(&freqs);
        assert!(positive[0] > positive[199], "positive phi did not peak at zero frequency");
        assert!(negative[199] > negative[0], "negative phi did not peak at the Nyquist end");
        for i in 0..200 {
            assert!(
                (positive[i] - negative[199 - i]).abs() < 1e-9,
                "the two spectra are not mirror images at index {i}"
            );
        }
    }

    #[test]
    fn roots_check_finds_the_boundary_of_stationarity() {
        assert_eq!(Arma::new(vec![0.9], vec![], 1.0, 0.0).roots_check(), (true, true));
        assert!(!Arma::new(vec![1.1], vec![], 1.0, 0.0).roots_check().0);
        // A root exactly on the unit circle is not stationary either.
        assert!(!Arma::new(vec![1.0], vec![], 1.0, 0.0).roots_check().0);
        // An MA is always stationary; only its invertibility is in question.
        assert_eq!(Arma::new(vec![], vec![2.0], 1.0, 0.0).roots_check(), (true, false));
        assert_eq!(Arma::new(vec![], vec![0.5], 1.0, 0.0).roots_check(), (true, true));
        // A white-noise model is trivially both.
        assert_eq!(Arma::new(vec![], vec![], 1.0, 0.0).roots_check(), (true, true));
        // AR(2) stationarity triangle: phi1 + phi2 < 1, phi2 - phi1 < 1,
        // |phi2| < 1. A point just outside must fail.
        assert!(Arma::new(vec![0.5, 0.4], vec![], 1.0, 0.0).roots_check().0);
        assert!(!Arma::new(vec![0.5, 0.6], vec![], 1.0, 0.0).roots_check().0);
        // |phi2| < 1 is the third side of the triangle: (0.1, -0.95) satisfies
        // every condition and is stationary, while (0.1, -1.05) fails only
        // this one.
        assert!(Arma::new(vec![0.1, -0.95], vec![], 1.0, 0.0).roots_check().0);
        assert!(!Arma::new(vec![0.1, -1.05], vec![], 1.0, 0.0).roots_check().0);
    }

    #[test]
    fn conditional_sum_of_squares_recovers_the_generating_parameters() {
        for (ar, ma) in [
            (vec![0.6], vec![]),
            (vec![], vec![0.5]),
            (vec![0.5, -0.25], vec![]),
            (vec![0.6], vec![0.4]),
        ] {
            let truth = Arma::new(ar.clone(), ma.clone(), 1.0, 3.0);
            let mut rng = Rng::new(0x71_0010 + ar.len() as u64 * 7 + ma.len() as u64);
            let x = truth.simulate(6000, &mut rng);
            let fit = Arma::fit_css(&x, ar.len(), ma.len()).unwrap();
            assert!((fit.mean - 3.0).abs() < 0.15, "mean came out {}", fit.mean);
            for (i, &t) in ar.iter().enumerate() {
                assert!((fit.ar[i] - t).abs() < 0.06, "phi_{i}: {} against {t}", fit.ar[i]);
            }
            for (j, &t) in ma.iter().enumerate() {
                assert!((fit.ma[j] - t).abs() < 0.06, "theta_{j}: {} against {t}", fit.ma[j]);
            }
            assert!((fit.sigma2 - 1.0).abs() < 0.1, "sigma2 came out {}", fit.sigma2);
            assert_eq!(fit.roots_check(), (true, true));
        }
    }

    #[test]
    fn hannan_rissanen_lands_near_the_least_squares_answer() {
        let truth = Arma::new(vec![0.6], vec![0.4], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0011);
        let x = truth.simulate(4000, &mut rng);
        let hr = Arma::fit_hannan_rissanen(&x, 1, 1).unwrap();
        assert!((hr.ar[0] - 0.6).abs() < 0.10, "phi came out {}", hr.ar[0]);
        assert!((hr.ma[0] - 0.4).abs() < 0.10, "theta came out {}", hr.ma[0]);
        // It should be close to, but generally not better than, the CSS fit.
        let css = Arma::fit_css(&x, 1, 1).unwrap();
        assert!(css.log_likelihood(&x) >= hr.log_likelihood(&x) - 1e-6);
    }

    #[test]
    fn residuals_of_a_correctly_specified_model_are_uncorrelated() {
        // A strong second lag: an AR(1) cannot mimic this, whereas a weak one
        // it approximates well enough that the portmanteau test sees nothing.
        let truth = Arma::new(vec![0.3, 0.5], vec![0.3], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0012);
        let x = truth.simulate(6000, &mut rng);
        // The wrong model leaves structure behind; the right one does not.
        let under = Arma::fit_css(&x, 1, 0).unwrap();
        let right = Arma::fit_css(&x, 2, 1).unwrap();
        let bad = ljung_box(&under.residuals(&x), 12);
        let good = ljung_box(&right.residuals(&x), 12);
        assert!(bad.p_value < 0.01, "an under-specified fit left no trace, p = {}", bad.p_value);
        assert!(good.p_value > 0.05, "the correct fit left structure, p = {}", good.p_value);
    }

    #[test]
    fn forecasts_of_an_ar1_follow_the_analytic_recursion() {
        // The h-step forecast is mu + phi^h (x_n - mu), and the error variance
        // is sigma2 sum_{j<h} phi^{2j}, which converges to the process
        // variance sigma2 / (1 - phi^2).
        let (phi, mu, sigma2) = (0.7f64, 5.0, 2.0);
        let model = Arma::new(vec![phi], vec![], sigma2, mu);
        let x: Vec<f64> = vec![4.0, 6.0, 5.5, 7.0, 6.2];
        let h = 30usize;
        let (point, se) = model.forecast(&x, h);
        let last = x[x.len() - 1];
        for k in 1..=h {
            let expected = mu + phi.powi(k as i32) * (last - mu);
            assert!(
                (point[k - 1] - expected).abs() < 1e-9,
                "step {k}: {} against {expected}",
                point[k - 1]
            );
        }
        assert!((se[0] - sigma2.sqrt()).abs() < 1e-12, "the one-step error is not sigma");
        assert!(se.windows(2).all(|w| w[1] >= w[0] - 1e-12), "the error band shrank");
        let limit = (sigma2 / (1.0 - phi * phi)).sqrt();
        assert!(
            (se[h - 1] - limit).abs() < 1e-6,
            "the band settled at {} rather than the process sd {limit}",
            se[h - 1]
        );
    }

    #[test]
    fn bic_penalises_extra_parameters_harder_than_aic() {
        let truth = Arma::new(vec![0.6], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0013);
        let x = truth.simulate(2000, &mut rng);
        let small = Arma::fit_css(&x, 1, 0).unwrap();
        let large = Arma::fit_css(&x, 3, 2).unwrap();
        // The larger model fits at least as well by likelihood alone.
        assert!(large.log_likelihood(&x) >= small.log_likelihood(&x) - 1e-6);
        // Both criteria should prefer the true order, and BIC by more.
        assert!(small.aic(&x) < large.aic(&x), "AIC preferred the over-fitted model");
        assert!(small.bic(&x) < large.bic(&x), "BIC preferred the over-fitted model");
        let aic_margin = large.aic(&x) - small.aic(&x);
        let bic_margin = large.bic(&x) - small.bic(&x);
        assert!(bic_margin > aic_margin, "BIC did not penalise more than AIC");
    }

    #[test]
    fn arma_rejects_a_series_too_short_for_its_orders() {
        assert!(Arma::fit_css(&[1.0, 2.0, 3.0], 2, 2).is_err());
        assert!(Arma::fit_hannan_rissanen(&[1.0, 2.0, 3.0, 4.0], 1, 1).is_err());
    }

    // -----------------------------------------------------------------
    // ARIMA
    // -----------------------------------------------------------------

    #[test]
    fn arima_forecast_bands_widen_without_bound_while_arma_bands_level_off() {
        // This is the practical difference between a differenced and an
        // undifferenced model: integration turns the psi weights into their
        // partial sums, and those do not square-sum to anything finite.
        let walk = random_walk(600, 0x71_0014);
        let integrated = Arima::fit(&walk, 1, 1, 0).unwrap();
        let (_, se_i) = integrated.forecast(40);
        assert!(se_i.windows(2).all(|w| w[1] > w[0]), "the ARIMA band stopped widening");
        assert!(se_i[39] > 3.0 * se_i[0], "the ARIMA band barely grew");

        let stable = Arma::new(vec![0.5], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0015);
        let x = stable.simulate(600, &mut rng);
        let fit = Arma::fit_css(&x, 1, 0).unwrap();
        let (_, se_a) = fit.forecast(&x, 40);
        assert!(
            (se_a[39] - se_a[30]).abs() < 1e-6,
            "the stationary band was still growing: {} then {}",
            se_a[30],
            se_a[39]
        );
    }

    #[test]
    fn a_random_walk_forecasts_flat_at_its_last_value() {
        // ARIMA(0,1,0) is the driftless random walk, whose optimal forecast at
        // every horizon is the last observation.
        let walk = random_walk(400, 0x71_0016);
        let model = Arima { d: 1, arma: Arma::new(vec![], vec![], 1.0, 0.0), initial: vec![walk[0]], tail: walk.clone() };
        let (point, se) = model.forecast(12);
        let last = walk[walk.len() - 1];
        assert!(point.iter().all(|v| (v - last).abs() < 1e-9), "the forecast was not flat");
        // And the band grows as sqrt(h), the random walk's own spread.
        for h in 1..=12usize {
            assert!(
                (se[h - 1] - (h as f64).sqrt()).abs() < 1e-9,
                "step {h} band {} against sqrt(h)",
                se[h - 1]
            );
        }
    }

    #[test]
    fn arima_recovers_a_trend_it_was_differenced_out_of() {
        // A linear trend plus AR(1) noise: after one difference the trend is a
        // constant, so the model should forecast the slope back.
        let slope = 0.4;
        let noise = Arma::new(vec![0.5], vec![], 0.25, 0.0);
        let mut rng = Rng::new(0x71_0017);
        let e = noise.simulate(800, &mut rng);
        let x: Vec<f64> = e.iter().enumerate().map(|(i, v)| slope * i as f64 + v).collect();
        let model = Arima::fit(&x, 1, 1, 0).unwrap();
        let (point, _) = model.forecast(20);
        // Successive forecasts should step up by roughly the slope.
        let steps: Vec<f64> = point.windows(2).map(|w| w[1] - w[0]).collect();
        let average = mean(&steps);
        assert!((average - slope).abs() < 0.1, "the recovered slope was {average}, not {slope}");
    }

    #[test]
    fn auto_arima_differences_a_walk_and_leaves_a_stationary_series_alone() {
        let walk = random_walk(600, 0x71_0018);
        let chosen = auto_arima(&walk, 2, 2, 1).unwrap();
        assert!(chosen.d >= 1, "auto_arima left a random walk undifferenced");

        let model = Arma::new(vec![0.6], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0019);
        let x = model.simulate(600, &mut rng);
        let picked = auto_arima(&x, 2, 2, 1).unwrap();
        assert_eq!(picked.d, 0, "auto_arima over-differenced a stationary series");
        assert!(picked.arma.roots_check().0, "auto_arima chose a non-stationary model");
    }

    // -----------------------------------------------------------------
    // SARIMA
    // -----------------------------------------------------------------

    #[test]
    fn sarima_reproduces_a_seasonal_pattern_it_was_shown() {
        let s = 12usize;
        let season: Vec<f64> = (0..s).map(|i| (i as f64 * 0.6).sin() * 8.0).collect();
        let mut rng = Rng::new(0x71_001A);
        let x: Vec<f64> = (0..240)
            .map(|t| 20.0 + season[t % s] + 0.3 * rng.next_gaussian())
            .collect();
        let model = Sarima::fit(&x, 1, 0, 0, 0, 1, 0, s).unwrap();
        let forecast = model.forecast(s);
        for k in 0..s {
            let expected = 20.0 + season[(240 + k) % s];
            assert!(
                (forecast[k] - expected).abs() < 2.0,
                "step {k}: forecast {} against pattern {expected}",
                forecast[k]
            );
        }
    }

    #[test]
    fn sarima_rejects_impossible_configurations() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(Sarima::fit(&x, 1, 0, 0, 0, 0, 0, 1).is_err(), "a season of 1 was accepted");
        assert!(Sarima::fit(&x, 0, 0, 0, 0, 0, 0, 12).is_err(), "a model with no parameters fitted");
        assert!(Sarima::fit(&x[..20], 1, 0, 0, 1, 1, 1, 12).is_err(), "a short series was accepted");
    }

    // -----------------------------------------------------------------
    // Smoothing
    // -----------------------------------------------------------------

    #[test]
    fn exponential_smoothing_preserves_a_constant_and_reduces_to_its_limits() {
        let constant = vec![7.0; 50];
        for alpha in [0.0, 0.1, 0.5, 1.0] {
            let s = exponential_smoothing(&constant, alpha);
            assert!(s.iter().all(|v| (v - 7.0).abs() < 1e-12), "alpha = {alpha} moved a constant");
        }
        let x = white_noise(60, 1.0, 0x71_001B);
        // alpha = 1 passes the data through untouched.
        assert_eq!(exponential_smoothing(&x, 1.0), x);
        // alpha = 0 never moves off the seed.
        assert!(exponential_smoothing(&x, 0.0).iter().all(|v| (v - x[0]).abs() < 1e-12));
        // Smoothing reduces variance.
        let smoothed = exponential_smoothing(&x, 0.2);
        let var = |v: &[f64]| {
            let m = mean(v);
            v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / v.len() as f64
        };
        assert!(var(&smoothed) < var(&x), "smoothing did not reduce variance");
    }

    #[test]
    fn holt_tracks_a_linear_trend_without_lagging() {
        // Simple smoothing sits below a rising line forever; Holt's slope term
        // is exactly what removes that bias.
        let x: Vec<f64> = (0..60).map(|i| 3.0 + 2.0 * i as f64).collect();
        let holt = double_exponential(&x, 0.6, 0.4);
        // With the state seeded a step before the data, Holt reproduces an
        // exact straight line exactly -- from the very first prediction, not
        // merely once a transient has decayed.
        for t in 0..60 {
            assert!(
                (holt[t] - x[t]).abs() < 1e-9,
                "at t = {t} Holt predicted {} for {}",
                holt[t],
                x[t]
            );
        }
        let simple = exponential_smoothing(&x, 0.6);
        assert!(
            simple[50] < x[50] - 1.0,
            "simple smoothing should lag a trend, but tracked it"
        );
    }

    #[test]
    fn holt_winters_reconstructs_a_trend_plus_season() {
        let s = 4usize;
        let season = [3.0, -1.0, -4.0, 2.0];
        let x: Vec<f64> =
            (0..80).map(|i| 10.0 + 0.5 * i as f64 + season[i % s]).collect();
        let (fitted, state) = holt_winters(&x, 0.4, 0.2, 0.3, s, false);
        // Element t predicts x[t]. On a noiseless trend-plus-season the seeded
        // state is already exact, so every prediction is too.
        for t in 0..80 {
            assert!(
                (fitted[t] - x[t]).abs() < 1e-9,
                "at t = {t} the fit was {} for {}",
                fitted[t],
                x[t]
            );
        }
        // And the state forecasts forward on the same pattern.
        let ahead = state.forecast(8);
        for k in 0..8 {
            let expected = 10.0 + 0.5 * (80 + k) as f64 + season[(80 + k) % s];
            assert!(
                (ahead[k] - expected).abs() < 1e-9,
                "step {k}: {} against {expected}",
                ahead[k]
            );
        }
    }

    #[test]
    fn holt_winters_optimize_beats_an_arbitrary_parameter_choice() {
        let s = 4usize;
        let season = [2.0, -3.0, 1.0, 0.0];
        let mut rng = Rng::new(0x71_001C);
        let x: Vec<f64> = (0..120)
            .map(|i| 5.0 + 0.3 * i as f64 + season[i % s] + 0.4 * rng.next_gaussian())
            .collect();
        let (a, b, g) = holt_winters_optimize(&x, s);
        assert!((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b) && (0.0..=1.0).contains(&g));

        let sse = |a: f64, b: f64, g: f64| {
            let (f, _) = holt_winters(&x, a, b, g, s, false);
            f.iter().zip(&x).skip(s).map(|(p, v)| (p - v) * (p - v)).sum::<f64>()
        };
        let best = sse(a, b, g);
        for &(ta, tb, tg) in
            &[(0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (0.5, 0.5, 0.5), (0.2, 0.8, 0.4)]
        {
            assert!(best <= sse(ta, tb, tg) + 1e-9, "({ta}, {tb}, {tg}) beat the optimiser");
        }
    }

    #[test]
    fn multiplicative_holt_winters_handles_growing_seasonal_amplitude() {
        // Seasonal swings proportional to the level: the case the additive
        // form cannot represent.
        let s = 4usize;
        let factor = [1.2, 0.8, 0.9, 1.1];
        let x: Vec<f64> =
            (0..80).map(|i| (10.0 + 2.0 * i as f64) * factor[i % s]).collect();
        let (mult, _) = holt_winters(&x, 0.4, 0.2, 0.3, s, true);
        let (add, _) = holt_winters(&x, 0.4, 0.2, 0.3, s, false);
        let err = |f: &[f64]| {
            f.iter().zip(&x).skip(2 * s).map(|(p, v)| (p - v) * (p - v)).sum::<f64>()
        };
        assert!(
            err(&mult) < err(&add),
            "the multiplicative form ({}) did not beat the additive one ({})",
            err(&mult),
            err(&add)
        );
    }

    // -----------------------------------------------------------------
    // Volatility
    // -----------------------------------------------------------------

    #[test]
    fn garch_unconditional_variance_is_the_fixed_point_of_its_recursion() {
        let g = Garch11 { omega: 0.02, alpha: 0.1, beta: 0.85 };
        let v = g.unconditional_variance();
        assert!((v - (g.omega + g.persistence() * v)).abs() < 1e-12, "v is not a fixed point");
        assert!((g.persistence() - 0.95).abs() < 1e-12);
        // At unit persistence there is no finite level to revert to.
        assert!(Garch11 { omega: 0.01, alpha: 0.1, beta: 0.9 }.unconditional_variance().is_infinite());
    }

    #[test]
    fn garch_variance_forecasts_converge_monotonically_to_the_unconditional_level() {
        let g = Garch11 { omega: 0.02, alpha: 0.1, beta: 0.85 };
        let mut rng = Rng::new(0x71_001D);
        let r = g.simulate(1000, &mut rng);
        let f = g.forecast_variance(&r, 300);
        let target = g.unconditional_variance();
        // Each step closes the gap by exactly the persistence factor.
        for w in f.windows(2) {
            let expected = g.omega + g.persistence() * w[0];
            assert!((w[1] - expected).abs() < 1e-12);
        }
        let gaps: Vec<f64> = f.iter().map(|v| (v - target).abs()).collect();
        assert!(gaps.windows(2).all(|w| w[1] <= w[0] + 1e-15), "the forecast moved away");
        assert!(gaps[299] < 1e-6, "still {} from the target after 300 steps", gaps[299]);
    }

    #[test]
    fn garch_simulation_reproduces_its_own_unconditional_variance() {
        let g = Garch11 { omega: 0.05, alpha: 0.08, beta: 0.87 };
        let mut rng = Rng::new(0x71_001E);
        let r = g.simulate(200_000, &mut rng);
        let realised = r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64;
        assert!(
            close(realised, g.unconditional_variance(), 0.08),
            "realised {realised} against {}",
            g.unconditional_variance()
        );
        // And the kurtosis exceeds a Gaussian's three: volatility clustering
        // makes the marginal distribution fat-tailed even with normal shocks.
        let m4 = r.iter().map(|v| v.powi(4)).sum::<f64>() / r.len() as f64;
        let kurtosis = m4 / (realised * realised);
        assert!(kurtosis > 3.3, "kurtosis was only {kurtosis}");
    }

    #[test]
    fn garch_fit_recovers_the_parameters_it_simulated_from() {
        let truth = Garch11 { omega: 0.05, alpha: 0.10, beta: 0.85 };
        let mut rng = Rng::new(0x71_001F);
        let r = truth.simulate(20_000, &mut rng);
        let fit = Garch11::fit(&r).unwrap();
        assert!(fit.omega > 0.0, "omega came out non-positive");
        assert!(fit.alpha >= 0.0 && fit.beta >= 0.0, "a weight came out negative");
        assert!(fit.persistence() < 1.0, "the fit is not stationary");
        assert!(
            (fit.persistence() - truth.persistence()).abs() < 0.05,
            "persistence {} against {}",
            fit.persistence(),
            truth.persistence()
        );
        assert!(
            close(fit.unconditional_variance(), truth.unconditional_variance(), 0.2),
            "unconditional variance {} against {}",
            fit.unconditional_variance(),
            truth.unconditional_variance()
        );
        assert!((fit.alpha - truth.alpha).abs() < 0.05, "alpha came out {}", fit.alpha);
    }

    #[test]
    fn the_conditional_variance_filter_is_positive_and_tracks_the_shocks() {
        let g = Garch11 { omega: 0.02, alpha: 0.2, beta: 0.7 };
        let mut r = white_noise(400, 0.3, 0x71_0020);
        // Plant a burst of large returns and check the filter responds.
        for t in 200..220 {
            r[t] *= 8.0;
        }
        let v = g.conditional_variance(&r);
        assert!(v.iter().all(|&x| x > 0.0), "the variance went non-positive");
        let quiet = mean(&v[150..190]);
        let loud = mean(&v[205..225]);
        assert!(loud > 4.0 * quiet, "the filter barely reacted: {quiet} then {loud}");
        // And it decays back afterwards.
        assert!(mean(&v[330..380]) < loud / 2.0, "the variance never came back down");
    }

    #[test]
    fn ewma_is_the_zero_intercept_unit_persistence_limit_of_garch() {
        let r = white_noise(500, 1.0, 0x71_0021);
        let lambda = 0.94;
        let ewma = ewma_variance(&r, lambda);
        let equivalent = Garch11 { omega: 0.0, alpha: 1.0 - lambda, beta: lambda };
        let garch = equivalent.conditional_variance(&r);
        // Both obey exactly the same recursion, step for step.
        for t in 1..500 {
            let step = |prev: f64| lambda * prev + (1.0 - lambda) * r[t - 1] * r[t - 1];
            assert!((ewma[t] - step(ewma[t - 1])).abs() < 1e-12);
            assert!((garch[t] - step(garch[t - 1])).abs() < 1e-12);
        }
        // They differ only in their seed, whose influence decays as lambda^t;
        // by t = 400 that factor is under 1e-10.
        let seed_gap = (ewma[0] - garch[0]).abs();
        for t in 100..500 {
            let bound = seed_gap * lambda.powi(t as i32) + 1e-9;
            assert!(
                (ewma[t] - garch[t]).abs() <= bound,
                "at t = {t} the gap {} exceeded the decayed seed bound {bound}",
                (ewma[t] - garch[t]).abs()
            );
        }
        assert!(ewma.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn arch_lm_separates_clustered_volatility_from_constant_volatility() {
        let g = Garch11 { omega: 0.05, alpha: 0.15, beta: 0.8 };
        let mut rng = Rng::new(0x71_0022);
        let clustered = g.simulate(3000, &mut rng);
        let flagged = arch_lm_test(&clustered, 5).unwrap();
        assert!(flagged.p_value < 1e-6, "GARCH data went undetected, p = {}", flagged.p_value);

        let plain = white_noise(3000, 1.0, 0x71_0023);
        let clean = arch_lm_test(&plain, 5).unwrap();
        assert!(clean.p_value > 0.05, "constant volatility was flagged, p = {}", clean.p_value);
        assert_eq!(clean.df, 5.0);
    }

    #[test]
    fn volatility_routines_reject_impossible_input() {
        assert!(Garch11::fit(&[0.1; 10]).is_err());
        assert!(Garch11::fit(&[0.0; 100]).is_err());
        assert!(arch_lm_test(&[0.1, 0.2], 5).is_err());
        assert!(arch_lm_test(&white_noise(200, 1.0, 5), 0).is_err());
    }

    // -----------------------------------------------------------------
    // Causality, cointegration, VAR
    // -----------------------------------------------------------------

    #[test]
    fn granger_finds_a_planted_lead_and_not_its_reverse() {
        let mut rng = Rng::new(0x71_0024);
        let n = 1200;
        let x: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        // y depends on x two steps back and on nothing else.
        let mut y = vec![0.0; n];
        for t in 2..n {
            y[t] = 0.6 * x[t - 2] + 0.3 * y[t - 1] + rng.next_gaussian();
        }
        let forward = granger_causality(&x, &y, 3).unwrap();
        let backward = granger_causality(&y, &x, 3).unwrap();
        assert!(forward.p_value < 1e-6, "the planted lead was missed, p = {}", forward.p_value);
        assert!(
            backward.p_value > 0.05,
            "a spurious reverse causality was found, p = {}",
            backward.p_value
        );
        assert_eq!(forward.df, 3.0);
    }

    #[test]
    fn granger_finds_nothing_between_independent_series() {
        let a = white_noise(1000, 1.0, 0x71_0025);
        let b = white_noise(1000, 1.0, 0x71_0026);
        let t = granger_causality(&a, &b, 4).unwrap();
        assert!(t.p_value > 0.05, "independent noise showed causality, p = {}", t.p_value);
    }

    #[test]
    fn cointegration_separates_a_shared_trend_from_two_independent_walks() {
        let mut rng = Rng::new(0x71_0027);
        let n = 600;
        // A common stochastic trend with stationary deviations around it.
        let mut trend = 0.0;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            trend += rng.next_gaussian();
            x.push(trend + 0.5 * rng.next_gaussian());
            y.push(2.0 * trend + 1.0 + 0.5 * rng.next_gaussian());
        }
        let linked = cointegration_engle_granger(&x, &y).unwrap();
        assert!(linked.p_value < 0.05, "a shared trend was missed, p = {}", linked.p_value);

        let a = random_walk(600, 0x71_0028);
        let b = random_walk(600, 0x71_0029);
        let unlinked = cointegration_engle_granger(&a, &b).unwrap();
        assert!(
            unlinked.p_value > 0.05,
            "independent walks looked cointegrated, p = {}",
            unlinked.p_value
        );
        // The tables must differ, or the whole distinction is lost.
        assert!(
            interpolate_p(&EG_TAU_TABLE, -3.2) > interpolate_p(&DF_TAU_TABLE, -3.2),
            "the Engle-Granger table is not more conservative than the Dickey-Fuller one"
        );
    }

    #[test]
    fn var_recovers_the_matrix_it_was_simulated_from() {
        // A stable bivariate VAR(1).
        let a = [[0.5, 0.2], [-0.1, 0.6]];
        let mut rng = Rng::new(0x71_002A);
        let n = 4000;
        let mut data: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];
        for t in 1..n {
            let p = &data[t - 1];
            data.push(vec![
                a[0][0] * p[0] + a[0][1] * p[1] + rng.next_gaussian(),
                a[1][0] * p[0] + a[1][1] * p[1] + rng.next_gaussian(),
            ]);
        }
        let fit = Var::fit(&data, 1).unwrap();
        assert_eq!(fit.k(), 2);
        assert_eq!(fit.p(), 1);
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (fit.coeffs[0].get(r, c) - a[r][c]).abs() < 0.05,
                    "A[{r}][{c}] came out {}",
                    fit.coeffs[0].get(r, c)
                );
            }
            assert!(fit.intercept[r].abs() < 0.1, "a spurious intercept appeared");
        }
    }

    #[test]
    fn var_forecast_is_the_recursion_applied_by_hand() {
        let a = [[0.5, 0.2], [-0.1, 0.6]];
        let mut rng = Rng::new(0x71_002B);
        let mut data: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];
        for t in 1..500 {
            let p = &data[t - 1];
            data.push(vec![
                a[0][0] * p[0] + a[0][1] * p[1] + rng.next_gaussian(),
                a[1][0] * p[0] + a[1][1] * p[1] + rng.next_gaussian(),
            ]);
        }
        let fit = Var::fit(&data, 1).unwrap();
        let f = fit.forecast(&data, 3).unwrap();

        let mut manual = data[data.len() - 1].clone();
        for step in 0..3 {
            let next: Vec<f64> = (0..2)
                .map(|r| {
                    fit.intercept[r]
                        + (0..2).map(|c| fit.coeffs[0].get(r, c) * manual[c]).sum::<f64>()
                })
                .collect();
            for i in 0..2 {
                assert!(
                    (f[step][i] - next[i]).abs() < 1e-9,
                    "step {step}, series {i}: {} against {}",
                    f[step][i],
                    next[i]
                );
            }
            manual = next;
        }
    }

    #[test]
    fn var_impulse_responses_start_at_the_identity_and_decay() {
        let a = [[0.5, 0.2], [-0.1, 0.6]];
        let mut rng = Rng::new(0x71_002C);
        let mut data: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];
        for t in 1..1500 {
            let p = &data[t - 1];
            data.push(vec![
                a[0][0] * p[0] + a[0][1] * p[1] + rng.next_gaussian(),
                a[1][0] * p[0] + a[1][1] * p[1] + rng.next_gaussian(),
            ]);
        }
        let fit = Var::fit(&data, 1).unwrap();
        let psi = fit.impulse_response(30);
        // The instantaneous response to a unit shock is the shock itself.
        assert_eq!(psi[0], Matrix::identity(2));
        // For a VAR(1) the m-step response is A^m, so it must equal the
        // fitted matrix raised to that power.
        assert!(psi[1].add(&fit.coeffs[0].scale(-1.0)).unwrap().frobenius_norm() < 1e-12);
        let squared = fit.coeffs[0].mul(&fit.coeffs[0]).unwrap();
        assert!(psi[2].add(&squared.scale(-1.0)).unwrap().frobenius_norm() < 1e-12);
        // A stable system forgets a shock.
        assert!(psi[30].frobenius_norm() < 1e-4, "the response did not decay");
        assert!(psi[5].frobenius_norm() < psi[1].frobenius_norm());
    }

    #[test]
    fn var_granger_matrix_finds_the_one_directed_link() {
        // Series 1 is driven by series 0; series 0 is driven by nothing.
        let mut rng = Rng::new(0x71_002D);
        let n = 2000;
        let mut data: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        for t in 2..n {
            let x = 0.4 * data[t - 1][0] + rng.next_gaussian();
            let y = 0.3 * data[t - 1][1] + 0.7 * data[t - 1][0] + rng.next_gaussian();
            data.push(vec![x, y]);
        }
        let fit = Var::fit(&data, 2).unwrap();
        let g = fit.granger_matrix(&data).unwrap();
        assert_eq!(g.get(0, 0), 1.0);
        assert_eq!(g.get(1, 1), 1.0);
        // 0 causes 1, so entry (1, 0) is small; the reverse is not.
        assert!(g.get(1, 0) < 1e-6, "the planted link was missed, p = {}", g.get(1, 0));
        assert!(g.get(0, 1) > 0.05, "a reverse link appeared, p = {}", g.get(0, 1));
    }

    #[test]
    fn var_rejects_ragged_or_short_input() {
        assert!(Var::fit(&[], 1).is_err());
        assert!(Var::fit(&[vec![1.0, 2.0], vec![3.0]], 1).is_err());
        assert!(Var::fit(&[vec![1.0], vec![2.0], vec![3.0]], 1).is_err());
        // Two independent shapes: a design whose columns are proportional is
        // rank deficient and would fail the fit for an unrelated reason.
        let data: Vec<Vec<f64>> =
            (0..300).map(|i| vec![(i as f64 * 0.1).sin(), (i as f64 * 0.37).cos()]).collect();
        let fit = Var::fit(&data, 1).unwrap();
        assert!(fit.forecast(&data, 0).is_err());
        assert!(fit.forecast(&[], 3).is_err());
    }

    // -----------------------------------------------------------------
    // Decomposition and change detection
    // -----------------------------------------------------------------

    #[test]
    fn the_decomposition_adds_back_to_the_input_exactly() {
        let mut rng = Rng::new(0x71_002E);
        let period = 12usize;
        let x: Vec<f64> = (0..120)
            .map(|i| {
                0.2 * i as f64
                    + 5.0 * ((i % period) as f64 * 0.5).sin()
                    + 0.3 * rng.next_gaussian()
            })
            .collect();
        let (trend, seasonal, resid) = seasonal_decompose_stl_lite(&x, period);
        for i in 0..x.len() {
            assert!(
                (trend[i] + seasonal[i] + resid[i] - x[i]).abs() < 1e-9,
                "the parts do not sum to the whole at index {i}"
            );
            assert!(trend[i].is_finite(), "the trend is undefined at index {i}");
        }
        // The seasonal component carries no level of its own.
        assert!(
            seasonal[..period].iter().sum::<f64>().abs() < 1e-9,
            "the seasonal factors do not sum to zero"
        );
        // And it repeats exactly.
        for i in period..x.len() {
            assert!((seasonal[i] - seasonal[i - period]).abs() < 1e-12);
        }
    }

    #[test]
    fn the_decomposition_recovers_a_planted_seasonal_shape() {
        let period = 4usize;
        let shape = [3.0, -1.0, -4.0, 2.0];
        let x: Vec<f64> = (0..100).map(|i| 10.0 + 0.1 * i as f64 + shape[i % period]).collect();
        let (_, seasonal, resid) = seasonal_decompose_stl_lite(&x, period);
        let centred: Vec<f64> = {
            let m = shape.iter().sum::<f64>() / period as f64;
            shape.iter().map(|v| v - m).collect()
        };
        for i in 0..period {
            assert!(
                (seasonal[i] - centred[i]).abs() < 0.2,
                "phase {i}: {} against {}",
                seasonal[i],
                centred[i]
            );
        }
        // With no noise the interior residual is essentially zero.
        assert!(
            resid[period..100 - period].iter().all(|v| v.abs() < 0.2),
            "a noiseless series left a residual"
        );
    }

    #[test]
    fn pelt_finds_planted_level_shifts() {
        let mut rng = Rng::new(0x71_002F);
        let mut x = Vec::new();
        for _ in 0..120 {
            x.push(0.0 + 0.4 * rng.next_gaussian());
        }
        for _ in 0..120 {
            x.push(5.0 + 0.4 * rng.next_gaussian());
        }
        for _ in 0..120 {
            x.push(1.0 + 0.4 * rng.next_gaussian());
        }
        let points = changepoint_pelt(&x, 20.0);
        assert_eq!(points.len(), 2, "found {points:?} rather than two changes");
        assert!((points[0] as isize - 120).abs() <= 3, "first break at {}", points[0]);
        assert!((points[1] as isize - 240).abs() <= 3, "second break at {}", points[1]);
        assert!(points.windows(2).all(|w| w[0] < w[1]), "the breaks are not ordered");
    }

    #[test]
    fn a_larger_penalty_never_yields_more_changepoints() {
        let mut rng = Rng::new(0x71_0030);
        let x: Vec<f64> = (0..300)
            .map(|i| if i < 100 { 0.0 } else if i < 200 { 3.0 } else { 1.0 })
            .map(|v: f64| v + 0.5 * rng.next_gaussian())
            .collect();
        let mut previous = usize::MAX;
        for penalty in [2.0, 10.0, 30.0, 100.0, 1000.0, 100_000.0] {
            let k = changepoint_pelt(&x, penalty).len();
            assert!(k <= previous, "penalty {penalty} produced more breaks than a smaller one");
            previous = k;
        }
        assert_eq!(previous, 0, "an enormous penalty still found breaks");
        // And a constant series has nothing to find at any penalty.
        assert!(changepoint_pelt(&[4.0; 200], 1.0).is_empty());
    }

    #[test]
    fn pelt_is_at_least_as_good_as_binary_segmentation_on_the_same_data() {
        // PELT is exact, so its segmentation cost can never exceed the greedy
        // one at the same number of breaks.
        let mut rng = Rng::new(0x71_0031);
        let x: Vec<f64> = (0..240)
            .map(|i| if i < 80 { 0.0 } else if i < 160 { 4.0 } else { 2.0 })
            .map(|v: f64| v + 0.6 * rng.next_gaussian())
            .collect();
        let pelt = changepoint_pelt(&x, 25.0);
        let binseg = changepoint_binary_segmentation(&x, pelt.len());
        assert_eq!(binseg.len(), pelt.len());

        let (prefix, prefix_sq) = prefix_sums(&x);
        let total = |breaks: &[usize]| -> f64 {
            let mut bounds = vec![0usize];
            bounds.extend_from_slice(breaks);
            bounds.push(x.len());
            bounds.windows(2).map(|w| segment_cost(&prefix, &prefix_sq, w[0], w[1])).sum()
        };
        assert!(
            total(&pelt) <= total(&binseg) + 1e-9,
            "PELT cost {} exceeded binary segmentation's {}",
            total(&pelt),
            total(&binseg)
        );
        // Both should land on the true breaks here.
        for (a, b) in binseg.iter().zip(&[80usize, 160]) {
            assert!((*a as isize - *b as isize).abs() <= 4, "binseg break at {a}, expected {b}");
        }
    }

    #[test]
    fn cusum_stays_flat_on_target_and_climbs_after_a_shift() {
        let mut on = white_noise(400, 1.0, 0x71_0032);
        let (hi, lo) = cusum(&on, 0.0, 0.5);
        assert!(hi.iter().all(|&v| v >= 0.0) && lo.iter().all(|&v| v >= 0.0));
        let quiet_max = hi.iter().cloned().fold(0.0f64, f64::max);

        // Now shift the mean up by two standard deviations part way through.
        for v in on.iter_mut().skip(200) {
            *v += 2.0;
        }
        let (hi2, lo2) = cusum(&on, 0.0, 0.5);
        assert!(hi2[399] > 10.0 * quiet_max.max(1.0), "the upper arm barely moved: {}", hi2[399]);
        assert!(hi2[199] <= quiet_max + 1e-9, "the arm rose before the shift");
        // The lower arm should be untouched by an upward shift.
        assert!(lo2[399] < 5.0, "the lower arm reacted to an upward shift: {}", lo2[399]);
        assert!(mean(&lo) >= 0.0);
    }

    #[test]
    fn the_matrix_profile_locates_a_planted_motif_and_a_planted_discord() {
        let m = 20usize;
        let mut rng = Rng::new(0x71_0033);
        let mut x: Vec<f64> = (0..300).map(|_| rng.next_gaussian()).collect();
        // Plant the same shape at two well-separated places.
        let motif: Vec<f64> = (0..m).map(|i| (i as f64 * 0.4).sin() * 3.0).collect();
        x[40..40 + m].copy_from_slice(&motif);
        x[200..200 + m].copy_from_slice(&motif);
        let (profile, index) = matrix_profile_lite(&x, m);
        assert_eq!(profile.len(), x.len() - m + 1);
        assert!(profile.iter().all(|v| v.is_finite() && *v >= 0.0));

        // The two planted windows are each other's nearest neighbour.
        assert!(profile[40] < 1e-6, "the motif did not match: {}", profile[40]);
        assert!(profile[200] < 1e-6, "the motif did not match: {}", profile[200]);
        assert_eq!(index[40], 200);
        assert_eq!(index[200], 40);
        // And the motif is the global minimum.
        let argmin = profile
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(argmin == 40 || argmin == 200, "the minimum sat at {argmin}");
        // No window matches itself or an overlapping neighbour.
        for (i, &j) in index.iter().enumerate() {
            assert!(i.abs_diff(j) >= m / 2, "window {i} matched its overlapping neighbour {j}");
        }
    }

    // -----------------------------------------------------------------
    // Complexity
    // -----------------------------------------------------------------

    #[test]
    fn permutation_entropy_spans_its_full_range() {
        // A monotone series visits one ordinal pattern; noise visits all of
        // them equally often.
        let rising: Vec<f64> = (0..500).map(|i| i as f64).collect();
        assert!(permutation_entropy(&rising, 3, 1) < 1e-12, "a ramp was not maximally regular");
        let falling: Vec<f64> = (0..500).map(|i| -(i as f64)).collect();
        assert!(permutation_entropy(&falling, 3, 1) < 1e-12);

        let noise = white_noise(20_000, 1.0, 0x71_0034);
        let h = permutation_entropy(&noise, 3, 1);
        assert!(h > 0.98, "noise scored only {h}");
        assert!(h <= 1.0 + 1e-12, "the normalised entropy exceeded one: {h}");

        // Invariant to any increasing transformation of the values.
        let stretched: Vec<f64> = noise.iter().map(|v| v.exp()).collect();
        assert!(
            (permutation_entropy(&stretched, 3, 1) - h).abs() < 1e-12,
            "a monotone transform changed the ordinal patterns"
        );

        // A periodic signal sits between the two extremes.
        let periodic: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.7).sin()).collect();
        let hp = permutation_entropy(&periodic, 4, 1);
        assert!(hp > 0.0 && hp < 0.7, "a sine wave scored {hp}");
    }

    #[test]
    fn sample_entropy_ranks_regularity_the_way_it_should() {
        // A clean sine is far more predictable than noise.
        let periodic: Vec<f64> = (0..600).map(|i| (i as f64 * 0.3).sin()).collect();
        let noise = white_noise(600, 1.0, 0x71_0035);
        let regular = sample_entropy(&periodic, 2, 0.2);
        let random = sample_entropy(&noise, 2, 0.2);
        assert!(regular < random, "the sine ({regular}) scored above noise ({random})");
        assert!(random > 1.0, "white noise scored only {random}");
        assert!(regular >= 0.0);

        // Approximate entropy orders them the same way but reads lower,
        // because counting the self-match biases it toward regularity.
        let ap_regular = approximate_entropy(&periodic, 2, 0.2);
        let ap_random = approximate_entropy(&noise, 2, 0.2);
        assert!(ap_regular < ap_random);
        assert!(ap_random < random, "approximate entropy was not the more biased of the two");
    }

    #[test]
    fn sample_entropy_reports_infinity_rather_than_inventing_a_number() {
        // With a tolerance far too tight for the data no template matches at
        // all, and there is no ratio to take.
        let x: Vec<f64> = (0..80).map(|i| i as f64).collect();
        assert!(sample_entropy(&x, 2, 1e-12).is_infinite());
    }

    #[test]
    fn iaaft_surrogates_keep_the_distribution_and_the_spectrum() {
        let model = Arma::new(vec![0.7], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0036);
        let x = model.simulate(256, &mut rng);
        let s = iaaft_surrogate(&x, &mut rng, 60);

        // The amplitude constraint is exact by construction: the surrogate is
        // a permutation of the original values.
        let mut a = x.clone();
        let mut b = s.clone();
        a.sort_by(|p, q| p.partial_cmp(q).unwrap());
        b.sort_by(|p, q| p.partial_cmp(q).unwrap());
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-12, "the surrogate changed the value set");
        }
        // The spectral constraint is approximate; the two projections fight.
        let power = |v: &[f64]| -> Vec<f64> {
            crate::transforms::fft::fft_any(
                &v.iter().map(|&z| Complex::new(z, 0.0)).collect::<Vec<_>>(),
            )
            .iter()
            .map(|c| c.norm_sq())
            .collect()
        };
        let (px, ps) = (power(&x), power(&s));
        let total: f64 = px.iter().sum();
        let error: f64 = px.iter().zip(&ps).map(|(p, q)| (p - q).abs()).sum();
        assert!(error / total < 0.06, "the spectrum drifted by {}", error / total);
        // And it is genuinely a different series, not a copy.
        assert!(
            x.iter().zip(&s).filter(|(p, q)| (*p - *q).abs() > 1e-12).count() > x.len() / 4,
            "the surrogate is barely distinguishable from the original"
        );
    }

    #[test]
    fn the_surrogate_test_accepts_a_linear_process_and_flags_a_nonlinear_one() {
        // A statistic sensitive to asymmetry under time reversal: zero in
        // expectation for a linear Gaussian process, non-zero for many
        // nonlinear ones.
        let reversibility = |v: &[f64]| -> f64 {
            let m = mean(v);
            let n = v.len();
            (1..n).map(|t| (v[t] - v[t - 1]).powi(3)).sum::<f64>() / n as f64 - m * 0.0
        };

        let linear = Arma::new(vec![0.6], vec![], 1.0, 0.0);
        let mut rng = Rng::new(0x71_0037);
        let x = linear.simulate(256, &mut rng);
        let p_linear = surrogate_test_iaaft(&x, &reversibility, 39, &mut rng);
        assert!(p_linear > 0.05, "a linear process was flagged, p = {p_linear}");

        // A series with a deterministic sawtooth: sharply time-irreversible,
        // yet with the same kind of spectrum a linear model could produce.
        let saw: Vec<f64> = (0..256).map(|i| (i % 16) as f64).collect();
        let p_nonlinear = surrogate_test_iaaft(&saw, &reversibility, 39, &mut rng);
        assert!(p_nonlinear <= 0.05, "a sawtooth went undetected, p = {p_nonlinear}");
        assert!((0.0..=1.0).contains(&p_linear) && (0.0..=1.0).contains(&p_nonlinear));
    }

    // -----------------------------------------------------------------
    // State space
    // -----------------------------------------------------------------

    #[test]
    fn the_local_level_model_recovers_its_variance_ratio() {
        // Signal-to-noise 1:4. The estimator has to separate a wandering level
        // from the noise sitting on top of it.
        let (sig, obs) = (0.25f64, 1.0f64);
        let mut rng = Rng::new(0x71_0038);
        let mut level = 0.0;
        let x: Vec<f64> = (0..4000)
            .map(|_| {
                level += sig.sqrt() * rng.next_gaussian();
                level + obs.sqrt() * rng.next_gaussian()
            })
            .collect();
        let (smoothed, eta, eps) = state_space_local_level(&x).unwrap();
        assert_eq!(smoothed.len(), x.len());
        assert!(eta > 0.0 && eps > 0.0, "a variance came out non-positive");
        let ratio = eta / eps;
        assert!(
            (ratio - sig / obs).abs() < 0.12,
            "the signal-to-noise ratio came out {ratio}, not {}",
            sig / obs
        );
        // The smoothed level is less variable than the raw series, since the
        // observation noise has been taken out.
        let var = |v: &[f64]| {
            let m = mean(v);
            v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / v.len() as f64
        };
        assert!(var(&smoothed) < var(&x), "smoothing did not reduce variance");
    }

    #[test]
    fn a_pure_noise_series_gets_a_flat_level() {
        // With no signal the optimiser should drive the state variance toward
        // zero, leaving the level essentially constant.
        let x = white_noise(1500, 1.0, 0x71_0039);
        let (smoothed, eta, eps) = state_space_local_level(&x).unwrap();
        assert!(eta / eps < 0.02, "a signal was found in pure noise: ratio {}", eta / eps);
        let spread = smoothed.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - smoothed.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(spread < 0.6, "the level wandered by {spread} on noise alone");
        assert!((mean(&smoothed) - mean(&x)).abs() < 0.2);
    }

    #[test]
    fn state_space_rejects_degenerate_input() {
        assert!(state_space_local_level(&[1.0, 2.0]).is_err());
        assert!(state_space_local_level(&[3.0; 40]).is_err());
    }
}
