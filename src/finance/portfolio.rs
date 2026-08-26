//! Portfolio construction and performance measurement.
//!
//! # What mean-variance optimisation actually does
//!
//! Markowitz's problem is: given expected returns and a covariance matrix,
//! find the weights minimising variance at each level of expected return.
//! It has a closed form, and that is both its appeal and its trap. The
//! optimiser is an *error maximiser*: it puts weight where the estimated
//! return is highest relative to the estimated risk, which is exactly
//! where the estimates are most likely to be wrong. Expected returns
//! estimated from a decade of monthly data carry standard errors of the
//! same order as the differences between assets, so the "optimal"
//! portfolio is often a leveraged bet on estimation noise.
//!
//! Nothing here shrinks, regularises or constrains, because the roadmap's
//! signatures do not. [`min_variance_weights`] uses only the covariance
//! matrix, which is estimated far more reliably than the mean, and is for
//! that reason the one output here that survives contact with real data.
//!
//! # Returns compound, and that decides which average to use
//!
//! [`returns_from_prices`] gives simple returns, whose *arithmetic* mean
//! is the expected one-period return. [`log_returns`] gives continuously
//! compounded returns, which add across periods, so their *sum* is the
//! total log return. Mixing them up produces the standard error of
//! quoting an arithmetic mean as though it were achievable: a series that
//! gains 50% then loses 50% has an arithmetic mean return of zero and has
//! lost a quarter of its value.

use crate::error::GeomError;
use crate::linalg::Matrix;

/// Simple period returns `p[t]/p[t-1] - 1`.
///
/// # Errors
/// Returns an error for fewer than two prices, or a non-positive or
/// non-finite price.
pub fn returns_from_prices(prices: &[f64]) -> Result<Vec<f64>, GeomError> {
    if prices.len() < 2 || prices.iter().any(|p| !(*p > 0.0) || !p.is_finite()) {
        return Err(GeomError::InvalidArgument("returns_from_prices: bad price series"));
    }
    Ok(prices.windows(2).map(|w| w[1] / w[0] - 1.0).collect())
}

/// Continuously compounded returns `ln(p[t]/p[t-1])`.
///
/// These add across periods, which is what makes them the right thing to
/// average when the question is about growth over time rather than about
/// the next period. They are always smaller than the simple return, by
/// roughly half the variance, which is the whole content of the
/// arithmetic-geometric gap.
///
/// # Errors
/// As [`returns_from_prices`].
pub fn log_returns(prices: &[f64]) -> Result<Vec<f64>, GeomError> {
    if prices.len() < 2 || prices.iter().any(|p| !(*p > 0.0) || !p.is_finite()) {
        return Err(GeomError::InvalidArgument("log_returns: bad price series"));
    }
    Ok(prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect())
}

fn check_covariance(cov: &Matrix) -> Result<usize, GeomError> {
    let n = cov.rows;
    if n == 0 || cov.cols != n {
        return Err(GeomError::InvalidArgument("the covariance matrix must be square"));
    }
    for i in 0..n {
        if !(cov.get(i, i) > 0.0) {
            return Err(GeomError::InvalidArgument("an asset has non-positive variance"));
        }
        for j in 0..n {
            if !cov.get(i, j).is_finite() {
                return Err(GeomError::InvalidArgument("a covariance is not finite"));
            }
            if (cov.get(i, j) - cov.get(j, i)).abs() > 1e-9 * cov.get(i, i).max(cov.get(j, j)) {
                return Err(GeomError::InvalidArgument("the covariance matrix is not symmetric"));
            }
        }
    }
    Ok(n)
}

/// The portfolio variance `w' C w`.
///
/// # Errors
/// Returns an error for a malformed covariance matrix or mismatched
/// weights.
pub fn portfolio_variance(cov: &Matrix, weights: &[f64]) -> Result<f64, GeomError> {
    let n = check_covariance(cov)?;
    if weights.len() != n {
        return Err(GeomError::InvalidArgument("the weights do not match the covariance matrix"));
    }
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            total += weights[i] * weights[j] * cov.get(i, j);
        }
    }
    Ok(total)
}

/// Solves `C x = b` for a symmetric positive-definite covariance matrix.
fn solve_covariance(cov: &Matrix, b: &[f64]) -> Result<Vec<f64>, GeomError> {
    crate::linalg::solve(cov, b)
        .map_err(|_| GeomError::Degenerate("the covariance matrix is singular"))
}

/// The global minimum-variance weights, which sum to one.
///
/// `w = C^-1 1 / (1' C^-1 1)`. Expected returns do not appear, which is
/// why this is the mean-variance output that survives real data: a
/// covariance matrix estimated from the same sample that produced a
/// hopeless mean estimate is still usually good enough to rank risk.
///
/// Weights may be negative -- the problem as posed allows short positions,
/// and with correlated assets the minimum-variance solution frequently
/// takes them.
///
/// # Errors
/// Returns an error for a malformed or singular covariance matrix, or one
/// whose implied weights do not sum to a usable total.
pub fn min_variance_weights(cov: &Matrix) -> Result<Vec<f64>, GeomError> {
    let n = check_covariance(cov)?;
    let ones = vec![1.0; n];
    let solved = solve_covariance(cov, &ones)?;
    let total: f64 = solved.iter().sum();
    if total.abs() < 1e-300 {
        return Err(GeomError::Degenerate("the minimum-variance weights do not normalise"));
    }
    Ok(solved.into_iter().map(|x| x / total).collect())
}

/// The tangency portfolio: the weights maximising the Sharpe ratio at a
/// given risk-free rate.
///
/// `w = C^-1 (mu - rf) / (1' C^-1 (mu - rf))`. Every portfolio on the
/// efficient frontier with a risk-free asset available is a mix of this
/// one and cash, which is the two-fund separation theorem -- and it is
/// what makes "the market portfolio" a meaningful object in CAPM.
///
/// The normalisation fails when the excess returns are orthogonal to the
/// inverse-covariance-weighted ones, and flips sign when the excess
/// returns are net negative, at which point the "tangency portfolio" is a
/// short position and the geometry has broken down. Both are reported
/// rather than returned as numbers.
///
/// # Errors
/// Returns an error for a malformed or singular covariance matrix,
/// mismatched means, or excess returns that do not determine a tangency.
pub fn tangency_portfolio(
    mu: &[f64],
    cov: &Matrix,
    risk_free: f64,
) -> Result<Vec<f64>, GeomError> {
    let n = check_covariance(cov)?;
    if mu.len() != n || mu.iter().any(|m| !m.is_finite()) || !risk_free.is_finite() {
        return Err(GeomError::InvalidArgument("tangency_portfolio: bad expected returns"));
    }
    let excess: Vec<f64> = mu.iter().map(|m| m - risk_free).collect();
    let solved = solve_covariance(cov, &excess)?;
    let total: f64 = solved.iter().sum();
    if total.abs() < 1e-12 {
        return Err(GeomError::Degenerate("the excess returns do not determine a tangency"));
    }
    Ok(solved.into_iter().map(|x| x / total).collect())
}

/// The efficient frontier as `(standard deviation, expected return,
/// weights)`, from the minimum-variance point up to the highest mean.
///
/// Each point solves the two-constraint problem exactly through the
/// standard `a, b, c` scalars, so no numerical optimisation is involved.
/// The frontier is a hyperbola in mean-standard-deviation space and a
/// parabola in mean-variance space, and its lower half -- the same
/// variances at lower returns -- is dominated and not returned.
///
/// Short positions are permitted throughout. A frontier computed with a
/// no-short constraint is a different and much better behaved object,
/// and it has no closed form.
///
/// # Errors
/// Returns an error for a malformed or singular covariance matrix,
/// mismatched means, fewer than two points, more than ten thousand, or
/// means that are all equal, where the frontier degenerates to a point.
pub fn markowitz_frontier(
    mu: &[f64],
    cov: &Matrix,
    points: usize,
) -> Result<Vec<(f64, f64, Vec<f64>)>, GeomError> {
    let n = check_covariance(cov)?;
    if mu.len() != n || mu.iter().any(|m| !m.is_finite()) {
        return Err(GeomError::InvalidArgument("markowitz_frontier: bad expected returns"));
    }
    if !(2..=10_000).contains(&points) {
        return Err(GeomError::InvalidArgument("markowitz_frontier: bad point count"));
    }
    let ones = vec![1.0; n];
    let inv_ones = solve_covariance(cov, &ones)?;
    let inv_mu = solve_covariance(cov, mu)?;
    // The three scalars every closed-form frontier is built from.
    let a: f64 = mu.iter().zip(inv_mu.iter()).map(|(m, x)| m * x).sum();
    let b: f64 = mu.iter().zip(inv_ones.iter()).map(|(m, x)| m * x).sum();
    let c: f64 = inv_ones.iter().sum();
    let determinant = a * c - b * b;
    if !(determinant > 1e-300) || !(c > 0.0) {
        return Err(GeomError::Degenerate(
            "the expected returns do not span a frontier: they are all equal or the matrix is ill-conditioned",
        ));
    }
    let smallest = b / c;
    let largest = mu.iter().fold(f64::NEG_INFINITY, |x, y| x.max(*y));
    let top = if largest > smallest { largest } else { smallest + 1.0 };
    let mut out = Vec::with_capacity(points);
    for step in 0..points {
        let target = smallest + (top - smallest) * step as f64 / (points - 1) as f64;
        // w = ((c target - b) inv_mu + (a - b target) inv_ones) / det
        let lambda = (c * target - b) / determinant;
        let gamma = (a - b * target) / determinant;
        let weights: Vec<f64> =
            (0..n).map(|i| lambda * inv_mu[i] + gamma * inv_ones[i]).collect();
        let variance = portfolio_variance(cov, &weights)?;
        out.push((variance.max(0.0).sqrt(), target, weights));
    }
    Ok(out)
}

/// Risk-parity weights: each asset contributes the same share of total
/// portfolio risk.
///
/// The condition is `w_i (C w)_i` equal across assets, which has no closed
/// form. It is solved here by the fixed point of
/// `w_i <- sqrt(w_i / (C w)_i)`, renormalised each pass: at rest that
/// gives `w_i^2 = k^2 w_i / (C w)_i`, so `w_i (C w)_i` is the same
/// constant for every asset, which is the condition itself.
///
/// The square root is not decoration. The undamped update
/// `w_i <- w_i / (C w)_i` converges to `(C w)_i` equal across assets --
/// which is the *minimum-variance* condition, not this one, and gives
/// visibly different weights whenever the assets differ in volatility.
///
/// This is not the same as equal weights, nor as inverse-volatility
/// weights -- those coincide with it only when correlations are all
/// equal. The appeal is that it needs no expected returns at all, which
/// removes the input mean-variance optimisation is most damaged by.
///
/// Weights are constrained positive, which is what makes the problem well
/// posed: the equal-risk-contribution condition has no positive solution
/// requirement built in, and shorting breaks the interpretation.
///
/// # Errors
/// Returns an error for a malformed covariance matrix, or an iteration
/// that does not converge.
pub fn risk_parity_weights(cov: &Matrix) -> Result<Vec<f64>, GeomError> {
    let n = check_covariance(cov)?;
    let mut weights = vec![1.0 / n as f64; n];
    for _ in 0..10_000 {
        let mut marginal = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                marginal[i] += cov.get(i, j) * weights[j];
            }
        }
        if marginal.iter().any(|m| !(*m > 0.0)) {
            return Err(GeomError::Degenerate("a marginal risk contribution went non-positive"));
        }
        let updated: Vec<f64> = (0..n).map(|i| (weights[i] / marginal[i]).sqrt()).collect();
        let total: f64 = updated.iter().sum();
        let normalised: Vec<f64> = updated.into_iter().map(|w| w / total).collect();
        let moved: f64 =
            normalised.iter().zip(weights.iter()).map(|(a, b)| (a - b).abs()).sum();
        weights = normalised;
        if moved < 1e-14 {
            return Ok(weights);
        }
    }
    Err(GeomError::Degenerate("risk parity did not converge"))
}

/// Each asset's share of total portfolio risk: `w_i (C w)_i / (w' C w)`.
///
/// The shares sum to one by construction, which is what makes "risk
/// contribution" a decomposition rather than a metaphor -- variance is a
/// quadratic form and Euler's theorem splits it exactly.
///
/// # Errors
/// As [`portfolio_variance`], plus a portfolio with no variance.
pub fn risk_contributions(cov: &Matrix, weights: &[f64]) -> Result<Vec<f64>, GeomError> {
    let n = check_covariance(cov)?;
    if weights.len() != n {
        return Err(GeomError::InvalidArgument("the weights do not match the matrix"));
    }
    let variance = portfolio_variance(cov, weights)?;
    if !(variance > 0.0) {
        return Err(GeomError::Degenerate("the portfolio has no variance to attribute"));
    }
    Ok((0..n)
        .map(|i| {
            let marginal: f64 = (0..n).map(|j| cov.get(i, j) * weights[j]).sum();
            weights[i] * marginal / variance
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Performance measurement
// ---------------------------------------------------------------------------

fn mean_and_deviation(values: &[f64]) -> Result<(f64, f64), GeomError> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(GeomError::InvalidArgument("at least two finite observations are required"));
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok((mean, variance.sqrt()))
}

/// The Sharpe ratio: mean excess return divided by its standard deviation.
///
/// Per period, not annualised -- multiplying by the square root of the
/// periods per year is the usual annualisation and it assumes returns are
/// independent, which is exactly what a trending or mean-reverting series
/// is not.
///
/// The denominator penalises upside and downside alike. A strategy that
/// occasionally doubles is punished for it, which is what [`sortino`]
/// addresses, and a strategy that sells insurance -- small steady gains
/// and a rare catastrophe -- scores well right up until the catastrophe.
/// The ratio says nothing about the shape of the distribution beyond its
/// first two moments.
///
/// # Errors
/// Returns an error for fewer than two returns, a non-finite value, or a
/// series with no variation.
pub fn sharpe(returns: &[f64], risk_free: f64) -> Result<f64, GeomError> {
    if !risk_free.is_finite() {
        return Err(GeomError::InvalidArgument("the risk-free rate is not finite"));
    }
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free).collect();
    let (mean, deviation) = mean_and_deviation(&excess)?;
    if !(deviation > 0.0) {
        return Err(GeomError::Degenerate("the returns have no variation"));
    }
    Ok(mean / deviation)
}

/// The Sortino ratio: mean excess return over the downside deviation.
///
/// The denominator is the root mean square of the shortfalls below
/// `target`, counting periods above it as zero rather than dropping them.
/// That choice matters: dividing by the count of losing periods instead
/// would make a strategy look better simply for losing less often, and
/// the two conventions differ by a factor that grows as losses get rarer.
///
/// # Errors
/// Returns an error for fewer than two returns, a non-finite value, or a
/// series that never falls below the target.
pub fn sortino(returns: &[f64], risk_free: f64, target: f64) -> Result<f64, GeomError> {
    if returns.len() < 2 || returns.iter().any(|r| !r.is_finite()) {
        return Err(GeomError::InvalidArgument("sortino: bad returns"));
    }
    if !risk_free.is_finite() || !target.is_finite() {
        return Err(GeomError::InvalidArgument("sortino: bad rate or target"));
    }
    let n = returns.len() as f64;
    let mean = returns.iter().map(|r| r - risk_free).sum::<f64>() / n;
    let downside =
        (returns.iter().map(|r| (r - target).min(0.0).powi(2)).sum::<f64>() / n).sqrt();
    if !(downside > 0.0) {
        return Err(GeomError::Degenerate("the series never fell below its target"));
    }
    Ok(mean / downside)
}

/// The maximum drawdown: the largest peak-to-trough fall, as a positive
/// fraction of the peak.
///
/// Computed against the running maximum, so it is a property of the path
/// and not of the endpoints. Two series with the same start and end can
/// have wildly different drawdowns, which is the point -- it measures what
/// an investor would have had to sit through.
///
/// # Errors
/// Returns an error for fewer than two prices, or a non-positive price.
pub fn max_drawdown(prices: &[f64]) -> Result<f64, GeomError> {
    if prices.len() < 2 || prices.iter().any(|p| !(*p > 0.0) || !p.is_finite()) {
        return Err(GeomError::InvalidArgument("max_drawdown: bad price series"));
    }
    let mut peak = prices[0];
    let mut worst = 0.0f64;
    for price in prices {
        peak = peak.max(*price);
        worst = worst.max((peak - price) / peak);
    }
    Ok(worst)
}

/// The Calmar ratio: annualised return divided by maximum drawdown.
///
/// `periods_per_year` converts the series' own period into a year. The
/// return used is the *geometric* one -- the constant rate that would have
/// produced the same total growth -- because that is what an investor
/// actually earned, unlike the arithmetic mean.
///
/// # Errors
/// Returns an error for fewer than two prices, a non-positive price or
/// period count, or a series with no drawdown to divide by.
pub fn calmar(prices: &[f64], periods_per_year: f64) -> Result<f64, GeomError> {
    if !(periods_per_year > 0.0) || !periods_per_year.is_finite() {
        return Err(GeomError::InvalidArgument("calmar: bad period count"));
    }
    let drawdown = max_drawdown(prices)?;
    if !(drawdown > 0.0) {
        return Err(GeomError::Degenerate("the series never fell, so there is nothing to divide by"));
    }
    let periods = (prices.len() - 1) as f64;
    let growth = prices[prices.len() - 1] / prices[0];
    let annual = growth.powf(periods_per_year / periods) - 1.0;
    Ok(annual / drawdown)
}

/// The information ratio: mean active return over its tracking error.
///
/// Active return is the portfolio's minus the benchmark's, period by
/// period. It is the Sharpe ratio of a long-short position against the
/// benchmark, which is why it is the natural measure for a manager judged
/// relative to an index rather than to cash.
///
/// # Errors
/// Returns an error for mismatched or too-short series, a non-finite
/// value, or an active series with no variation.
pub fn information_ratio(portfolio: &[f64], benchmark: &[f64]) -> Result<f64, GeomError> {
    if portfolio.len() != benchmark.len() {
        return Err(GeomError::InvalidArgument("the two series must have the same length"));
    }
    let active: Vec<f64> = portfolio.iter().zip(benchmark.iter()).map(|(p, b)| p - b).collect();
    let (mean, deviation) = mean_and_deviation(&active)?;
    if !(deviation > 0.0) {
        return Err(GeomError::Degenerate("the portfolio tracks the benchmark exactly"));
    }
    Ok(mean / deviation)
}

/// The CAPM regression of an asset on the market, returning
/// `(alpha, beta)`.
///
/// Beta is `cov(asset, market) / var(market)` and alpha is the intercept
/// that remains. Beta is an estimate of sensitivity and nothing more: it
/// is a single number summarising a scatter that may not be linear, it is
/// unstable across sample periods, and a high R-squared is required before
/// it means very much at all.
///
/// # Errors
/// Returns an error for mismatched or too-short series, a non-finite
/// value, or a market series with no variation.
pub fn capm_beta(asset: &[f64], market: &[f64]) -> Result<(f64, f64), GeomError> {
    if asset.len() != market.len() || asset.len() < 2 {
        return Err(GeomError::InvalidArgument("capm_beta: mismatched or too-short series"));
    }
    if asset.iter().chain(market.iter()).any(|x| !x.is_finite()) {
        return Err(GeomError::InvalidArgument("capm_beta: a value is not finite"));
    }
    let n = asset.len() as f64;
    let mean_asset = asset.iter().sum::<f64>() / n;
    let mean_market = market.iter().sum::<f64>() / n;
    let covariance: f64 = asset
        .iter()
        .zip(market.iter())
        .map(|(a, m)| (a - mean_asset) * (m - mean_market))
        .sum::<f64>()
        / (n - 1.0);
    let variance: f64 =
        market.iter().map(|m| (m - mean_market).powi(2)).sum::<f64>() / (n - 1.0);
    if !(variance > 0.0) {
        return Err(GeomError::Degenerate("the market series has no variation"));
    }
    let beta = covariance / variance;
    Ok((mean_asset - beta * mean_market, beta))
}

/// The Kelly fraction for a discrete bet won with probability `p` paying
/// `b` to one: `p - (1 - p)/b`.
///
/// Maximises the expected *logarithm* of wealth, which is the growth rate
/// achieved almost surely over many repetitions. A negative answer means
/// the bet has no edge and the optimal stake is nothing.
///
/// The fraction assumes the edge is known exactly. Overestimating it
/// pushes the stake past the growth-optimal point, where growth falls
/// faster than it rose: staking twice the Kelly fraction earns no more
/// than the risk-free rate however large the edge, and beyond that it
/// loses. That is why practitioners bet a fraction of it.
///
/// # Errors
/// Returns an error for a probability outside `[0, 1]` or a non-positive
/// payout.
pub fn kelly_fraction(p: f64, b: f64) -> Result<f64, GeomError> {
    if !(0.0..=1.0).contains(&p) || !(b > 0.0) || !b.is_finite() {
        return Err(GeomError::InvalidArgument("kelly_fraction: bad probability or payout"));
    }
    Ok(p - (1.0 - p) / b)
}

/// The continuous Kelly fraction `(mu - rf)/sigma^2`.
///
/// The same object for a lognormal asset: the leverage maximising the
/// long-run growth rate. It is also the tangency portfolio's leverage
/// under one asset, which is not a coincidence -- both maximise the
/// Sharpe-like quantity `(mu - rf)/sigma` scaled by the risk taken.
///
/// # Errors
/// Returns an error for a non-positive volatility or a non-finite input.
pub fn kelly_continuous(mu: f64, sigma: f64, risk_free: f64) -> Result<f64, GeomError> {
    if !(sigma > 0.0) || ![mu, sigma, risk_free].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("kelly_continuous: bad parameters"));
    }
    Ok((mu - risk_free) / (sigma * sigma))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A three-asset covariance matrix with real correlation.
    fn sample_covariance() -> Matrix {
        let vols = [0.15f64, 0.22, 0.30];
        let corr = [[1.0, 0.3, 0.1], [0.3, 1.0, 0.5], [0.1, 0.5, 1.0]];
        let mut cov = Matrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                cov.set(i, j, vols[i] * vols[j] * corr[i][j]);
            }
        }
        cov
    }

    #[test]
    fn a_log_return_is_the_logarithm_of_one_plus_the_simple_one() {
        let prices = [100.0, 110.0, 99.0, 123.75];
        let simple = returns_from_prices(&prices).unwrap();
        let logs = log_returns(&prices).unwrap();
        assert_eq!(simple.len(), 3);
        assert!((simple[0] - 0.1).abs() < 1e-15);
        assert!((simple[1] - -0.1).abs() < 1e-15);
        assert!((simple[2] - 0.25).abs() < 1e-15);
        for (r, l) in simple.iter().zip(logs.iter()) {
            assert!((l - (1.0 + r).ln()).abs() < 1e-15);
            // A log return is always the smaller of the two.
            assert!(*l <= r + 1e-15);
        }
        // Log returns add to the total, simple ones do not.
        let total: f64 = logs.iter().sum();
        assert!((total.exp() - prices[3] / prices[0]).abs() < 1e-13);
        assert!(returns_from_prices(&[100.0]).is_err());
        assert!(log_returns(&[100.0, 0.0]).is_err());
        assert!(returns_from_prices(&[100.0, -5.0]).is_err());
    }

    #[test]
    fn up_fifty_then_down_fifty_is_a_loss_the_arithmetic_mean_hides() {
        // The single clearest reason to keep the two averages apart.
        let prices = [100.0, 150.0, 75.0];
        let simple = returns_from_prices(&prices).unwrap();
        let arithmetic: f64 = simple.iter().sum::<f64>() / 2.0;
        assert!(arithmetic.abs() < 1e-15, "the arithmetic mean was {arithmetic}");
        assert!((prices[2] / prices[0] - 0.75).abs() < 1e-15, "a quarter of the value is gone");
        // The log returns tell the truth: they sum to ln(0.75) < 0.
        let logs = log_returns(&prices).unwrap();
        assert!((logs.iter().sum::<f64>() - 0.75f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn the_minimum_variance_portfolio_really_is_the_minimum() {
        // Perturbing along any direction that keeps the weights summing to
        // one must raise the variance, and by a second-order amount, which
        // is what a minimum looks like.
        let cov = sample_covariance();
        let weights = min_variance_weights(&cov).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let base = portfolio_variance(&cov, &weights).unwrap();
        for direction in [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [1.0, 0.5, -1.5]] {
            for size in [0.01f64, -0.01, 0.1, -0.1] {
                let moved: Vec<f64> =
                    (0..3).map(|i| weights[i] + size * direction[i]).collect();
                assert!((moved.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                let variance = portfolio_variance(&cov, &moved).unwrap();
                assert!(variance > base, "moving by {size} lowered the variance");
                // Second order: the excess scales with the square.
                let excess = variance - base;
                let tenth = {
                    let smaller: Vec<f64> =
                        (0..3).map(|i| weights[i] + 0.1 * size * direction[i]).collect();
                    portfolio_variance(&cov, &smaller).unwrap() - base
                };
                assert!(
                    (excess / tenth - 100.0).abs() < 1e-6,
                    "the excess did not scale quadratically: {}",
                    excess / tenth
                );
            }
        }
    }

    #[test]
    fn independent_assets_get_minimum_variance_weights_in_inverse_variance() {
        // With no correlation the closed form is exactly proportional to
        // the reciprocal of each variance, which is a case with an answer
        // known in advance.
        let variances = [0.01f64, 0.04, 0.25];
        let mut cov = Matrix::zeros(3, 3);
        for (i, v) in variances.iter().enumerate() {
            cov.set(i, i, *v);
        }
        let weights = min_variance_weights(&cov).unwrap();
        let total: f64 = variances.iter().map(|v| 1.0 / v).sum();
        for (i, v) in variances.iter().enumerate() {
            assert!(
                (weights[i] - (1.0 / v) / total).abs() < 1e-12,
                "asset {i} got {} not {}",
                weights[i],
                (1.0 / v) / total
            );
        }
    }

    #[test]
    fn the_tangency_portfolio_has_the_highest_sharpe_ratio_there_is() {
        let cov = sample_covariance();
        let mu = [0.06f64, 0.09, 0.12];
        let risk_free = 0.02;
        let weights = tangency_portfolio(&mu, &cov, risk_free).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let ratio = |w: &[f64]| {
            let ret: f64 = w.iter().zip(mu.iter()).map(|(x, m)| x * m).sum();
            (ret - risk_free) / portfolio_variance(&cov, w).unwrap().sqrt()
        };
        let best = ratio(&weights);
        assert!(best > 0.0);
        for direction in [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 2.0, -1.0]] {
            for size in [0.02f64, -0.02, 0.2, -0.2] {
                let moved: Vec<f64> = (0..3).map(|i| weights[i] + size * direction[i]).collect();
                assert!(ratio(&moved) < best, "moving by {size} raised the Sharpe ratio");
            }
        }
    }

    #[test]
    fn every_frontier_point_is_the_least_variance_at_its_own_return() {
        let cov = sample_covariance();
        let mu = [0.06f64, 0.09, 0.12];
        let frontier = markowitz_frontier(&mu, &cov, 9).unwrap();
        assert_eq!(frontier.len(), 9);
        // The lowest point is the global minimum-variance portfolio.
        let minimum = min_variance_weights(&cov).unwrap();
        for (a, b) in frontier[0].2.iter().zip(minimum.iter()) {
            assert!((a - b).abs() < 1e-10, "the frontier does not start at the minimum");
        }
        // A direction that preserves both the budget and the expected
        // return, so a move along it stays at the same point of the
        // frontier's vertical axis.
        let neutral = [mu[1] - mu[2], mu[2] - mu[0], mu[0] - mu[1]];
        for (deviation, target, weights) in &frontier {
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
            assert!((achieved - target).abs() < 1e-10, "it returned {achieved} not {target}");
            let variance = portfolio_variance(cov_ref(&cov), weights).unwrap();
            assert!((variance.sqrt() - deviation).abs() < 1e-12);
            for size in [0.05f64, -0.05] {
                let moved: Vec<f64> =
                    (0..3).map(|i| weights[i] + size * neutral[i]).collect();
                let shifted: f64 = moved.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
                assert!((shifted - target).abs() < 1e-10, "the move changed the return");
                assert!(
                    portfolio_variance(&cov, &moved).unwrap() > variance,
                    "a same-return portfolio had less variance"
                );
            }
        }
        // Risk rises with return above the minimum-variance point.
        for pair in frontier.windows(2) {
            assert!(pair[1].1 > pair[0].1, "the return did not rise");
            assert!(pair[1].0 > pair[0].0, "the risk did not rise with it");
        }
    }

    /// Borrow helper so the loop above reads naturally.
    fn cov_ref(m: &Matrix) -> &Matrix {
        m
    }

    #[test]
    fn risk_parity_gives_every_asset_the_same_share_of_the_risk() {
        // Which is the definition, and it is not the same portfolio as the
        // minimum-variance one -- an undamped iteration converges to that
        // instead, and the equal shares are what tell the two apart.
        let cov = sample_covariance();
        let weights = risk_parity_weights(&cov).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|w| *w > 0.0), "a weight went negative");
        let shares = risk_contributions(&cov, &weights).unwrap();
        for share in &shares {
            assert!((share - 1.0 / 3.0).abs() < 1e-9, "a share was {share}");
        }
        assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let minimum = min_variance_weights(&cov).unwrap();
        let apart: f64 =
            weights.iter().zip(minimum.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(apart > 0.1, "risk parity landed on the minimum-variance weights");
        // Risk parity takes more variance than the minimum, by
        // construction.
        assert!(
            portfolio_variance(&cov, &weights).unwrap()
                > portfolio_variance(&cov, &minimum).unwrap()
        );
    }

    #[test]
    fn independent_assets_get_risk_parity_weights_in_inverse_volatility() {
        // With no correlation, equal risk contribution reduces to equal
        // volatility contribution, which is the reciprocal of each
        // standard deviation -- not of each variance.
        let vols = [0.1f64, 0.2, 0.5];
        let mut cov = Matrix::zeros(3, 3);
        for (i, v) in vols.iter().enumerate() {
            cov.set(i, i, v * v);
        }
        let weights = risk_parity_weights(&cov).unwrap();
        let total: f64 = vols.iter().map(|v| 1.0 / v).sum();
        for (i, v) in vols.iter().enumerate() {
            assert!(
                (weights[i] - (1.0 / v) / total).abs() < 1e-9,
                "asset {i} got {} not {}",
                weights[i],
                (1.0 / v) / total
            );
        }
    }

    #[test]
    fn the_portfolio_builders_refuse_a_matrix_that_is_not_a_covariance() {
        let mut asymmetric = sample_covariance();
        asymmetric.set(0, 1, 0.9);
        assert!(min_variance_weights(&asymmetric).is_err());
        let mut negative = Matrix::zeros(2, 2);
        negative.set(0, 0, -1.0);
        negative.set(1, 1, 1.0);
        assert!(min_variance_weights(&negative).is_err());
        let cov = sample_covariance();
        assert!(tangency_portfolio(&[0.05, 0.06], &cov, 0.02).is_err());
        assert!(markowitz_frontier(&[0.06, 0.09, 0.12], &cov, 1).is_err());
        // Equal expected returns leave no frontier to trace.
        assert!(markowitz_frontier(&[0.07, 0.07, 0.07], &cov, 5).is_err());
        // And excess returns that cancel leave no tangency.
        let mut orthogonal = Matrix::zeros(2, 2);
        orthogonal.set(0, 0, 0.04);
        orthogonal.set(1, 1, 0.04);
        assert!(tangency_portfolio(&[0.05, -0.01], &orthogonal, 0.02).is_err());
        assert!(portfolio_variance(&cov, &[1.0, 0.0]).is_err());
        assert!(risk_contributions(&cov, &[0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn the_performance_ratios_are_the_quantities_they_are_named_after() {
        // A series with a known mean and deviation, so the ratio has an
        // arithmetic answer rather than a plausible one.
        let returns = [0.02f64, -0.01, 0.03, 0.00, 0.01];
        let mean = 0.01;
        let deviation = {
            let ss: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum();
            (ss / 4.0).sqrt()
        };
        assert!((sharpe(&returns, 0.0).unwrap() - mean / deviation).abs() < 1e-15);
        // Subtracting a constant rate shifts the mean and nothing else.
        assert!(
            (sharpe(&returns, 0.005).unwrap() - (mean - 0.005) / deviation).abs() < 1e-15
        );
        // Sortino counts only the shortfalls, so it exceeds Sharpe on a
        // series whose losses are milder than its gains.
        let skewed = [0.10f64, -0.01, 0.08, -0.02, 0.09];
        assert!(sortino(&skewed, 0.0, 0.0).unwrap() > sharpe(&skewed, 0.0).unwrap());
        assert!(sharpe(&[0.01, 0.01, 0.01], 0.0).is_err(), "no variation, no ratio");
        assert!(sortino(&[0.01, 0.02], 0.0, 0.0).is_err(), "never below target");
    }

    #[test]
    fn a_drawdown_is_a_property_of_the_path_and_not_of_its_ends() {
        // Two series with the same start and finish and very different
        // experiences in between.
        let smooth = [100.0f64, 105.0, 110.0, 115.0, 120.0];
        let rough = [100.0f64, 150.0, 60.0, 90.0, 120.0];
        assert!(max_drawdown(&smooth).unwrap() < 1e-15, "a rising series has no drawdown");
        // 150 down to 60 is a fall of 60%.
        assert!((max_drawdown(&rough).unwrap() - 0.6).abs() < 1e-15);
        assert_eq!(smooth[4], rough[4]);

        // Calmar divides the annualised growth by that drawdown, so the
        // rough path scores far worse for the same total return.
        assert!(calmar(&smooth, 252.0).is_err(), "no drawdown, nothing to divide by");
        let calm = calmar(&rough, 4.0).unwrap();
        let growth = (120.0f64 / 100.0).powf(4.0 / 4.0) - 1.0;
        assert!((calm - growth / 0.6).abs() < 1e-12, "got {calm}");
        assert!(max_drawdown(&[100.0]).is_err());
    }

    #[test]
    fn the_regression_recovers_a_beta_that_was_put_there_on_purpose() {
        // An asset built as alpha plus beta times the market must give
        // exactly those two back, with no residual to confuse them.
        let market = [0.01f64, -0.02, 0.03, 0.00, 0.015, -0.005, 0.02];
        for (alpha, beta) in [(0.001f64, 1.5f64), (0.0, 0.4), (-0.002, -0.8)] {
            let asset: Vec<f64> = market.iter().map(|m| alpha + beta * m).collect();
            let (a, b) = capm_beta(&asset, &market).unwrap();
            assert!((b - beta).abs() < 1e-12, "beta came back {b} not {beta}");
            assert!((a - alpha).abs() < 1e-12, "alpha came back {a} not {alpha}");
        }
        // Against itself the beta is one and the alpha nothing.
        let (a, b) = capm_beta(&market, &market).unwrap();
        assert!((b - 1.0).abs() < 1e-14 && a.abs() < 1e-16);
        assert!(capm_beta(&market, &[0.01; 7]).is_err(), "a flat market has no beta");
        assert!(capm_beta(&market, &market[..3]).is_err());
    }

    #[test]
    fn the_information_ratio_is_the_sharpe_ratio_of_the_active_position() {
        let portfolio = [0.02f64, -0.01, 0.03, 0.00, 0.01];
        let benchmark = [0.01f64, -0.02, 0.02, 0.01, 0.00];
        let active: Vec<f64> =
            portfolio.iter().zip(benchmark.iter()).map(|(p, b)| p - b).collect();
        let ratio = information_ratio(&portfolio, &benchmark).unwrap();
        assert!((ratio - sharpe(&active, 0.0).unwrap()).abs() < 1e-15);
        // Tracking the benchmark exactly leaves nothing to measure.
        assert!(information_ratio(&portfolio, &portfolio).is_err());
        assert!(information_ratio(&portfolio, &benchmark[..3]).is_err());
    }

    #[test]
    fn staking_twice_the_kelly_fraction_earns_nothing_over_the_risk_free_rate() {
        // The growth rate is quadratic in the stake with its peak at the
        // Kelly fraction, so it returns to the risk-free rate at twice it
        // and falls below beyond -- which is why an overestimated edge is
        // worse than a halved one.
        let (mu, sigma, risk_free) = (0.10, 0.20, 0.02);
        let kelly = kelly_continuous(mu, sigma, risk_free).unwrap();
        assert!((kelly - 2.0).abs() < 1e-15, "the fraction was {kelly}");
        let growth = |f: f64| risk_free + f * (mu - risk_free) - 0.5 * f * f * sigma * sigma;
        assert!(growth(kelly) > growth(0.0));
        assert!((growth(2.0 * kelly) - risk_free).abs() < 1e-15);
        assert!(growth(2.5 * kelly) < risk_free);
        // Half Kelly keeps three quarters of the excess growth for a
        // quarter of the variance drag.
        let excess = growth(kelly) - risk_free;
        assert!(((growth(0.5 * kelly) - risk_free) / excess - 0.75).abs() < 1e-12);

        // The discrete form: an even-money bet needs an edge to be worth
        // taking at all.
        assert!((kelly_fraction(0.6, 1.0).unwrap() - 0.2).abs() < 1e-15);
        assert!(kelly_fraction(0.5, 1.0).unwrap().abs() < 1e-15);
        assert!(kelly_fraction(0.4, 1.0).unwrap() < 0.0, "a losing bet should be refused");
        // Longer odds make a smaller edge worth taking.
        assert!(kelly_fraction(0.3, 4.0).unwrap() > 0.0);
        assert!(kelly_fraction(1.5, 1.0).is_err());
        assert!(kelly_continuous(0.1, 0.0, 0.02).is_err());
    }
}
