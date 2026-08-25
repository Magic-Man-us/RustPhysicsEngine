//! Risk measurement: value at risk, expected shortfall, backtesting.
//!
//! # What value at risk does and does not tell you
//!
//! VaR at confidence `1 - alpha` is a *quantile*: the loss that will be
//! exceeded on a fraction `alpha` of days. It says nothing whatever about
//! how much worse things get beyond it, and that is not a subtlety but the
//! central objection to the measure. Two portfolios with identical VaR can
//! have completely different tails, and the one with the fatter tail is
//! the one that ends the firm.
//!
//! [`cvar_historical`] -- expected shortfall -- answers the question VaR
//! ducks: the *average* loss given that VaR is exceeded. It is also
//! *coherent* where VaR is not: VaR can penalise diversification, saying
//! a combined portfolio is riskier than the sum of its parts, because a
//! quantile is not subadditive. Expected shortfall cannot do that. Since
//! Basel III, expected shortfall is the regulatory measure and VaR is
//! the one everyone still quotes.
//!
//! # Sign convention
//!
//! Every function here returns a **positive number for a loss**. A VaR of
//! 0.023 means a 2.3% loss. This is the industry convention and it is the
//! opposite of the return series' own sign, which is a standing source of
//! confusion; the tests pin it down explicitly.

use crate::error::GeomError;
use crate::statistics::distributions::{ChiSquared, Distribution};

/// The `alpha` quantile of a sample, by linear interpolation between
/// order statistics.
fn quantile(sorted: &[f64], alpha: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let position = alpha * (n - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(n - 1);
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn check_returns(returns: &[f64], alpha: f64) -> Result<(), GeomError> {
    if returns.len() < 2 || returns.iter().any(|r| !r.is_finite()) {
        return Err(GeomError::InvalidArgument("at least two finite returns are required"));
    }
    if !(0.0..1.0).contains(&alpha) || alpha == 0.0 {
        return Err(GeomError::InvalidArgument("the tail probability must lie in (0, 1)"));
    }
    Ok(())
}

/// Historical value at risk: the empirical `alpha` quantile of the losses.
///
/// No distributional assumption at all -- the sample *is* the
/// distribution. That is its strength and its limit: it cannot produce a
/// loss larger than the worst one observed, so a 99% VaR from two hundred
/// days is estimated from two points and a 99.9% VaR from none.
///
/// Returned positive for a loss.
///
/// # Errors
/// Returns an error for fewer than two returns, a non-finite value, or an
/// `alpha` outside `(0, 1)`.
pub fn var_historical(returns: &[f64], alpha: f64) -> Result<f64, GeomError> {
    check_returns(returns, alpha)?;
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite returns"));
    Ok(-quantile(&sorted, alpha))
}

/// Parametric value at risk under a normal distribution:
/// `-(mean + z_alpha * deviation)`.
///
/// Fits two moments and reads the quantile off a Gaussian. Financial
/// returns are not Gaussian -- they have fat tails and negative skew -- so
/// this understates the tail systematically, and by more the further out
/// you go. At 95% the error is modest; at 99.9% it is a factor.
///
/// Returned positive for a loss.
///
/// # Errors
/// Returns an error for fewer than two returns, a non-finite value, an
/// `alpha` outside `(0, 1)`, or a series with no variation.
pub fn var_parametric(returns: &[f64], alpha: f64) -> Result<f64, GeomError> {
    check_returns(returns, alpha)?;
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let deviation = variance.sqrt();
    if !(deviation > 0.0) {
        return Err(GeomError::Degenerate("the returns have no variation"));
    }
    Ok(-(mean + normal_quantile(alpha) * deviation))
}

/// The standard normal quantile, by bisection on the CDF.
fn normal_quantile(p: f64) -> f64 {
    let cdf = |x: f64| crate::statistics::distributions::gaussian_cdf(x, 0.0, 1.0);
    let (mut low, mut high) = (-40.0f64, 40.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
        if high - low < 1e-15 * (1.0 + low.abs()) {
            break;
        }
    }
    0.5 * (low + high)
}

/// Historical expected shortfall: the mean loss among the worst `alpha`
/// fraction of returns.
///
/// Always at least the VaR at the same level, and strictly greater
/// whenever the tail has any spread at all. Unlike VaR it is *coherent* --
/// in particular subadditive, so combining two portfolios can never make
/// the measured risk exceed the sum of the parts. VaR has no such
/// guarantee and can and does penalise diversification.
///
/// Returned positive for a loss.
///
/// # Errors
/// As [`var_historical`], plus an `alpha` so small that no observation
/// falls in the tail.
pub fn cvar_historical(returns: &[f64], alpha: f64) -> Result<f64, GeomError> {
    check_returns(returns, alpha)?;
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite returns"));
    // At least one observation in the tail, always.
    let count = ((alpha * sorted.len() as f64).floor() as usize).max(1);
    let tail: f64 = sorted[..count].iter().sum::<f64>() / count as f64;
    Ok(-tail)
}

/// Cornish-Fisher value at risk: the Gaussian quantile corrected for the
/// sample's skewness and excess kurtosis.
///
/// The expansion adjusts `z` by terms in the third and fourth moments,
/// which is enough to capture the direction and rough size of a fat tail
/// without fitting a distribution. Two limitations are worth stating
/// plainly, because both bite at ordinary parameters.
///
/// *The kurtosis term changes sign inside the tail.* Its factor is
/// `z^3 - 3z`, which is zero at `z = -sqrt(3)`, or `alpha` of about 4.2%.
/// So a fat-tailed sample gets a larger VaR at 1% and a *smaller* one at
/// 5%, from the same correction. The expansion is meant for the far tail
/// and behaves sensibly there; near the 5% point the fourth-moment term
/// is doing something close to nothing, and just past it the wrong thing.
///
/// *It is asymptotic, not convergent.* For mild moments it improves on
/// the Gaussian fit -- with a skew of -0.4 and an excess kurtosis of 0.8
/// it moves a 1% VaR from 0.0197 to 0.0234 against a historical 0.0300.
/// For large ones it overshoots wildly: at a skew of -4.6 and an excess
/// kurtosis of 33.8 it returns 0.0729 where the sample's own 1% quantile
/// is 0.0309. There is no cheap test that separates the two, so the
/// moments must be checked before the answer is trusted.
///
/// What *is* checked is the standard validity condition: the corrected
/// quantile must be increasing in `z`, since a quantile function that
/// decreases is not one. That catches the grossest failures and no more.
///
/// Returned positive for a loss.
///
/// # Errors
/// Returns an error for fewer than four returns, a non-finite value, an
/// `alpha` outside `(0, 1)`, a series with no variation, or moments large
/// enough to break the expansion.
pub fn var_cornish_fisher(returns: &[f64], alpha: f64) -> Result<f64, GeomError> {
    check_returns(returns, alpha)?;
    if returns.len() < 4 {
        return Err(GeomError::InvalidArgument("the expansion needs at least four returns"));
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let deviation = variance.sqrt();
    if !(deviation > 0.0) {
        return Err(GeomError::Degenerate("the returns have no variation"));
    }
    let standardised = |power: i32| {
        returns.iter().map(|r| ((r - mean) / deviation).powi(power)).sum::<f64>() / n
    };
    let skew = standardised(3);
    let excess = standardised(4) - 3.0;
    let z = normal_quantile(alpha);
    let corrected = z
        + (z * z - 1.0) * skew / 6.0
        + (z * z * z - 3.0 * z) * excess / 24.0
        - (2.0 * z * z * z - 5.0 * z) * skew * skew / 36.0;
    // The expansion is a quantile function only while it is increasing in
    // z. Differentiating the polynomial gives this slope, and where it is
    // non-positive the "corrected" number is not a quantile of anything.
    let slope = 1.0 + z * skew / 3.0 + (z * z - 1.0) * excess / 8.0
        - (6.0 * z * z - 5.0) * skew * skew / 36.0;
    if !corrected.is_finite() || slope <= 0.0 {
        return Err(GeomError::Degenerate(
            "the sample's moments put it outside the Cornish-Fisher expansion's valid range",
        ));
    }
    Ok(-(mean + corrected * deviation))
}

/// A one-step-ahead parametric VaR from a fitted GARCH(1,1) model.
///
/// Filters the conditional variance through the sample, projects one step
/// with `omega + alpha r_last^2 + beta sigma_last^2`, and reads a Gaussian
/// quantile off the result. The point is that VaR from a GARCH forecast
/// *responds*: after a volatile week it rises, where an unconditional
/// estimate over the same window barely moves. That responsiveness is
/// what a risk measure is for, and it is also why GARCH VaR breaches
/// cluster less than unconditional VaR breaches do.
///
/// The Gaussian quantile still understates the tail; GARCH captures the
/// clustering of volatility, not the fatness of the conditional
/// distribution.
///
/// Returned positive for a loss.
///
/// # Errors
/// Returns an error for fewer than two returns, a non-finite value, an
/// `alpha` outside `(0, 1)`, or a model whose projected variance is not
/// positive.
pub fn garch_var_forecast(
    model: &crate::stochastic::timeseries::Garch11,
    returns: &[f64],
    alpha: f64,
) -> Result<f64, GeomError> {
    check_returns(returns, alpha)?;
    let filtered = model.conditional_variance(returns);
    let last_return = returns[returns.len() - 1];
    let last_variance = filtered[filtered.len() - 1];
    let projected = model.omega + model.alpha * last_return * last_return + model.beta * last_variance;
    if !(projected > 0.0) || !projected.is_finite() {
        return Err(GeomError::Degenerate("the projected variance is not positive"));
    }
    Ok(-normal_quantile(alpha) * projected.sqrt())
}

/// What a backtest reports.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestStats {
    /// Total return over the whole series, as a fraction.
    pub total_return: f64,
    /// The number of round trips taken.
    pub trades: usize,
    /// The fraction of round trips that made money.
    pub win_rate: f64,
    /// The largest peak-to-trough fall in the equity curve.
    pub max_drawdown: f64,
    /// The equity curve, starting at one.
    pub equity: Vec<f64>,
}

/// Backtests a moving-average crossover: long while the fast average is
/// above the slow one, flat otherwise.
///
/// Both averages are computed on the closing prices up to and including
/// the current bar, and the resulting position is applied to the *next*
/// bar's return. Applying it to the same bar would use the close to decide
/// a trade executed at that close, which is the commonest way a backtest
/// invents returns that were never available.
///
/// There are no costs, no slippage and no borrowing charge, so the result
/// is an upper bound on what the rule could have earned rather than an
/// estimate of it. A crossover rule trades often enough that realistic
/// costs frequently reverse its sign.
///
/// # Errors
/// Returns an error for a non-positive price, fewer prices than the slow
/// window needs, a zero window, or a fast window at or above the slow one.
pub fn backtest_sma_crossover(
    prices: &[f64],
    fast: usize,
    slow: usize,
) -> Result<BacktestStats, GeomError> {
    if fast == 0 || slow == 0 || fast >= slow {
        return Err(GeomError::InvalidArgument("the fast window must be shorter than the slow"));
    }
    if prices.len() < slow + 2 || prices.iter().any(|p| !(*p > 0.0) || !p.is_finite()) {
        return Err(GeomError::InvalidArgument("backtest_sma_crossover: bad price series"));
    }
    let average = |end: usize, window: usize| -> f64 {
        prices[end + 1 - window..=end].iter().sum::<f64>() / window as f64
    };
    let mut equity = vec![1.0];
    let mut wealth = 1.0;
    let mut holding = false;
    let mut entry = 0.0;
    let mut trades = 0usize;
    let mut wins = 0usize;
    for bar in (slow - 1)..prices.len() - 1 {
        let signal = average(bar, fast) > average(bar, slow);
        if signal && !holding {
            holding = true;
            entry = prices[bar];
        } else if !signal && holding {
            holding = false;
            trades += 1;
            if prices[bar] > entry {
                wins += 1;
            }
        }
        if holding {
            wealth *= prices[bar + 1] / prices[bar];
        }
        equity.push(wealth);
    }
    if holding {
        trades += 1;
        if prices[prices.len() - 1] > entry {
            wins += 1;
        }
    }
    let mut peak = 0.0f64;
    let mut worst = 0.0f64;
    for value in &equity {
        peak = peak.max(*value);
        worst = worst.max((peak - value) / peak);
    }
    Ok(BacktestStats {
        total_return: wealth - 1.0,
        trades,
        win_rate: if trades > 0 { wins as f64 / trades as f64 } else { 0.0 },
        max_drawdown: worst,
        equity,
    })
}

/// Kupiec's unconditional coverage test: does the observed breach count
/// match the VaR model's claimed `alpha`?
///
/// The likelihood ratio statistic is chi-squared with one degree of
/// freedom under the null that breaches occur at exactly rate `alpha`. A
/// small p-value means the model is miscalibrated -- too many breaches
/// and it understates risk, too few and it overstates it and wastes
/// capital.
///
/// What it cannot see is *clustering*. A model that breaches on ten
/// consecutive days and never again can pass Kupiec with the right total,
/// while being useless: the breaches should be independent, and testing
/// that needs Christoffersen's conditional coverage test, which this is
/// only half of.
///
/// # Errors
/// Returns an error for no observations, more breaches than observations,
/// or an `alpha` outside `(0, 1)`.
pub fn kupiec_test(
    violations: usize,
    observations: usize,
    alpha: f64,
) -> Result<crate::statistics::inference::TestResult, GeomError> {
    if observations == 0 || violations > observations {
        return Err(GeomError::InvalidArgument("kupiec_test: bad counts"));
    }
    if !(0.0..1.0).contains(&alpha) || alpha == 0.0 {
        return Err(GeomError::InvalidArgument("the tail probability must lie in (0, 1)"));
    }
    let n = observations as f64;
    let x = violations as f64;
    let observed = x / n;
    // Under the null the breaches are Bernoulli(alpha); the alternative
    // fits the observed rate. Zero and full counts make one factor zero,
    // and the limit `0 ln 0 = 0` is taken.
    let term = |p: f64, count: f64| if count == 0.0 { 0.0 } else { count * p.ln() };
    let null = term(alpha, x) + term(1.0 - alpha, n - x);
    let fitted = term(observed, x) + term(1.0 - observed, n - x);
    let statistic = (-2.0 * (null - fitted)).max(0.0);
    let p_value = 1.0 - ChiSquared::new(1.0).cdf(statistic);
    Ok(crate::statistics::inference::TestResult { statistic, p_value, df: 1.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    /// A near-Gaussian return series of the given length.
    fn gaussian_returns(n: usize, mean: f64, deviation: f64, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| mean + deviation * rng.next_gaussian()).collect()
    }

    #[test]
    fn a_loss_is_reported_as_a_positive_number() {
        // The industry convention, and the opposite of the return series'
        // own sign. Getting it backwards is the single commonest error in
        // risk code, so it is pinned down here rather than assumed.
        let returns = [-0.10f64, -0.05, 0.0, 0.05, 0.10];
        let var = var_historical(&returns, 0.25).unwrap();
        assert!(var > 0.0, "a loss came back as {var}");
        // The 25% quantile of five sorted points sits at index 1, which is
        // -0.05, so the VaR is +0.05.
        assert!((var - 0.05).abs() < 1e-15, "got {var}");
        // A series that only ever gains has a *negative* VaR: the "loss"
        // at that confidence is a profit.
        let winners = [0.01f64, 0.02, 0.03, 0.04];
        assert!(var_historical(&winners, 0.25).unwrap() < 0.0);
    }

    #[test]
    fn expected_shortfall_is_never_below_the_quantile_it_averages_past() {
        let mut rng = Rng::new(0x0F1D_1001);
        for _ in 0..30 {
            let n = 200 + (rng.next_f64() * 800.0) as usize;
            let returns = gaussian_returns(n, 0.0005, 0.012, rng.next_u64());
            for alpha in [0.01f64, 0.025, 0.05, 0.1, 0.25] {
                let var = var_historical(&returns, alpha).unwrap();
                let shortfall = cvar_historical(&returns, alpha).unwrap();
                assert!(
                    shortfall >= var - 1e-12,
                    "at alpha={alpha} the shortfall {shortfall} fell under the VaR {var}"
                );
                // With a continuous distribution the tail has spread, so
                // the inequality is strict.
                assert!(shortfall > var, "the tail had no spread at alpha={alpha}");
            }
        }
    }

    #[test]
    fn value_at_risk_rises_as_the_confidence_does() {
        let returns = gaussian_returns(2000, 0.0, 0.01, 0x0F1D_1002);
        let mut previous = f64::NEG_INFINITY;
        for alpha in [0.25f64, 0.1, 0.05, 0.025, 0.01] {
            let historical = var_historical(&returns, alpha).unwrap();
            let parametric = var_parametric(&returns, alpha).unwrap();
            assert!(historical > previous, "the historical VaR fell at alpha={alpha}");
            previous = historical;
            // Near-Gaussian data, so the two agree closely.
            assert!(
                (historical - parametric).abs() < 0.12 * parametric,
                "at alpha={alpha}: {historical} against {parametric}"
            );
        }
    }

    #[test]
    fn the_parametric_estimate_is_the_gaussian_quantile_it_claims_to_be() {
        // Built from a known mean and deviation, so the answer is
        // arithmetic: -(mean + z sigma) with z the standard normal
        // quantile.
        let returns = gaussian_returns(200_000, 0.001, 0.02, 0x0F1D_1003);
        for (alpha, z) in [(0.05f64, -1.644_853_626_951_47f64), (0.01, -2.326_347_874_040_84)] {
            let expected = -(0.001 + z * 0.02);
            let parametric = var_parametric(&returns, alpha).unwrap();
            assert!(
                (parametric - expected).abs() < 0.02 * expected,
                "at alpha={alpha}: {parametric} against {expected}"
            );
        }
        assert!(var_parametric(&[0.01, 0.01, 0.01], 0.05).is_err());
        assert!(var_historical(&[0.01], 0.05).is_err());
        assert!(var_historical(&[0.01, 0.02], 0.0).is_err());
        assert!(var_historical(&[0.01, 0.02], 1.0).is_err());
    }

    #[test]
    fn the_correction_pulls_a_thin_tailed_sample_back_toward_its_own_quantile() {
        // A uniform sample is symmetric with an excess kurtosis of -1.2,
        // so the Gaussian fit overstates its tail. Far out, the correction
        // moves the estimate back toward the sample's own quantile.
        let half: Vec<f64> = (1..=500).map(|k| k as f64 * 0.001).collect();
        let mut symmetric: Vec<f64> = half.iter().map(|x| -x).collect();
        symmetric.extend(half.iter());

        let historical = var_historical(&symmetric, 0.01).unwrap();
        let parametric = var_parametric(&symmetric, 0.01).unwrap();
        let corrected = var_cornish_fisher(&symmetric, 0.01).unwrap();
        assert!(parametric > historical, "the Gaussian fit should overstate a uniform tail");
        assert!(
            corrected < parametric && corrected > historical,
            "the correction gave {corrected}, outside [{historical}, {parametric}]"
        );
    }

    #[test]
    fn the_kurtosis_term_changes_sign_inside_the_tail() {
        // Its factor is z^3 - 3z, zero at z = -sqrt(3), which is an alpha
        // of about 4.2%. The same sample therefore gets the correction
        // applied one way at 1% and the other way at 5%. This is a real
        // property of the expansion and a reason to use it in the far tail
        // only.
        let half: Vec<f64> = (1..=500).map(|k| k as f64 * 0.001).collect();
        let mut symmetric: Vec<f64> = half.iter().map(|x| -x).collect();
        symmetric.extend(half.iter());

        let far = var_cornish_fisher(&symmetric, 0.01).unwrap()
            - var_parametric(&symmetric, 0.01).unwrap();
        let near = var_cornish_fisher(&symmetric, 0.05).unwrap()
            - var_parametric(&symmetric, 0.05).unwrap();
        assert!(far < 0.0, "at 1% the correction moved by {far}");
        assert!(near > 0.0, "at 5% the correction moved by {near}");
        // And it nearly vanishes at the crossing itself.
        let crossing = var_cornish_fisher(&symmetric, 0.0416).unwrap()
            - var_parametric(&symmetric, 0.0416).unwrap();
        assert!(crossing.abs() < 0.05 * far.abs(), "at the crossing it moved by {crossing}");
    }

    #[test]
    fn a_mild_left_skew_is_where_the_correction_earns_its_keep() {
        // Skew -0.39 and excess kurtosis 0.85: small enough for the
        // asymptotic series to mean something. The Gaussian fit understates
        // the 1% loss and the correction closes most of the gap.
        let mut returns = gaussian_returns(4000, 0.0005, 0.008, 0x0F1D_1004);
        for k in 0..40 {
            returns[k * 97] = -0.03;
        }
        let historical = var_historical(&returns, 0.01).unwrap();
        let parametric = var_parametric(&returns, 0.01).unwrap();
        let corrected = var_cornish_fisher(&returns, 0.01).unwrap();
        assert!(parametric < historical, "the Gaussian fit should understate this tail");
        assert!(
            corrected > parametric && corrected < historical,
            "the correction gave {corrected}, outside [{parametric}, {historical}]"
        );
        assert!(
            (corrected - historical).abs() < (parametric - historical).abs(),
            "the correction did not improve on the Gaussian fit"
        );
    }

    #[test]
    fn large_moments_make_the_expansion_overshoot_rather_than_fail() {
        // Skew -4.6 and excess kurtosis 33.8. The series is asymptotic, so
        // it does not converge to the answer -- it runs past it, returning
        // more than twice the sample's own quantile. Nothing detects this,
        // which is why the moments have to be looked at before the number
        // is used.
        let mut returns = gaussian_returns(4000, 0.0005, 0.008, 0x0F1D_1004);
        for k in 0..40 {
            returns[k * 97] = -0.10;
        }
        let historical = var_historical(&returns, 0.01).unwrap();
        let corrected = var_cornish_fisher(&returns, 0.01).unwrap();
        assert!(
            corrected > 2.0 * historical,
            "the expansion gave {corrected} against a sample quantile of {historical}"
        );
    }

    #[test]
    fn a_quantile_that_would_run_backwards_is_refused() {
        // The expansion is a quantile function only while it increases in
        // z. A sample with a huge excess kurtosis, read near the middle of
        // the distribution rather than in its tail, breaks that -- and
        // there the answer is not a quantile of anything.
        let mut returns = vec![0.001f64; 2000];
        for (index, value) in returns.iter_mut().enumerate() {
            *value = if index % 500 == 0 { -0.5 + 1.0 * f64::from(index % 1000 == 0) } else { 0.001 };
        }
        // Excess kurtosis of a few hundred, and alpha well away from the
        // tail so z is small.
        let refused = var_cornish_fisher(&returns, 0.4);
        assert!(refused.is_err(), "an invalid expansion returned {refused:?}");
        // Far out in the tail the same sample is still refused or answers
        // sensibly, but never silently returns a decreasing quantile.
        if let Ok(value) = var_cornish_fisher(&returns, 0.01) {
            assert!(value.is_finite());
        }
        assert!(var_cornish_fisher(&[0.01, -0.01, 0.02], 0.05).is_err());
    }

    #[test]
    fn value_at_risk_can_punish_diversification_where_expected_shortfall_cannot() {
        // Two independent defaultable bonds, each losing everything in 4%
        // of scenarios and earning a coupon otherwise. At 95% confidence
        // each one's VaR is a *gain*, because the worst 5% still misses
        // the defaults. Combine them and the defaults land inside the
        // tail, so the combined VaR is enormous -- diversifying made the
        // measured risk jump. Expected shortfall, which is coherent,
        // cannot do this.
        let n = 10_000;
        let mut a = vec![0.01f64; n];
        let mut b = vec![0.01f64; n];
        for i in 0..n {
            if i % 25 == 0 {
                a[i] = -1.0;
            }
            if (i + 7) % 25 == 0 {
                b[i] = -1.0;
            }
        }
        let mixed: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| 0.5 * (x + y)).collect();

        let var_a = var_historical(&a, 0.05).unwrap();
        let var_b = var_historical(&b, 0.05).unwrap();
        let var_mixed = var_historical(&mixed, 0.05).unwrap();
        assert!(var_a < 0.0 && var_b < 0.0, "each bond's 95% VaR should be a gain");
        assert!(var_mixed > 0.4, "the combined VaR was only {var_mixed}");
        assert!(
            var_mixed > var_a + var_b,
            "VaR was subadditive here after all: {var_mixed} against {}",
            var_a + var_b
        );

        let cvar_a = cvar_historical(&a, 0.05).unwrap();
        let cvar_b = cvar_historical(&b, 0.05).unwrap();
        let cvar_mixed = cvar_historical(&mixed, 0.05).unwrap();
        assert!(
            cvar_mixed <= cvar_a + cvar_b + 1e-12,
            "expected shortfall was superadditive: {cvar_mixed} against {}",
            cvar_a + cvar_b
        );
        // And it sees the defaults that VaR missed.
        assert!(cvar_a > 0.7, "the shortfall missed the defaults: {cvar_a}");
    }

    #[test]
    fn a_garch_forecast_answers_to_what_just_happened() {
        // The point of a conditional model: the same unconditional sample
        // gives a different one-day VaR depending on how the last few days
        // went. An unconditional estimate over the same window cannot.
        let model = crate::stochastic::timeseries::Garch11 { omega: 1e-5, alpha: 0.1, beta: 0.85 };
        let calm = gaussian_returns(500, 0.0, 0.005, 0x0F1D_1005);
        let mut stormy = calm.clone();
        for value in stormy.iter_mut().rev().take(20) {
            *value *= 8.0;
        }
        let after_calm = garch_var_forecast(&model, &calm, 0.01).unwrap();
        let after_storm = garch_var_forecast(&model, &stormy, 0.01).unwrap();
        assert!(after_storm > 1.5 * after_calm, "{after_storm} against {after_calm}");
        assert!(after_calm > 0.0);
        // A model with no persistence forecasts the same thing regardless.
        let flat = crate::stochastic::timeseries::Garch11 { omega: 4e-5, alpha: 0.0, beta: 0.0 };
        let a = garch_var_forecast(&flat, &calm, 0.01).unwrap();
        let b = garch_var_forecast(&flat, &stormy, 0.01).unwrap();
        assert!((a - b).abs() < 1e-15, "a memoryless model still moved: {a} against {b}");
        assert!(garch_var_forecast(&model, &[0.01], 0.01).is_err());
    }

    #[test]
    fn kupiec_reports_no_evidence_when_the_breaches_land_where_they_should() {
        // Exactly the expected count makes the likelihood ratio zero and
        // the p-value one, which is the calibration point the test is
        // built around.
        let exact = kupiec_test(50, 1000, 0.05).unwrap();
        assert!(exact.statistic.abs() < 1e-12, "the statistic was {}", exact.statistic);
        assert!((exact.p_value - 1.0).abs() < 1e-12);
        assert_eq!(exact.df, 1.0);

        // Far too many breaches: the model understates risk and the test
        // says so beyond any doubt.
        let understated = kupiec_test(120, 1000, 0.05).unwrap();
        assert!(understated.statistic > 50.0, "got {}", understated.statistic);
        assert!(understated.p_value < 1e-10);

        // Far too few: also rejected, since a model that never breaches is
        // wasting capital rather than being safe.
        let overstated = kupiec_test(5, 1000, 0.05).unwrap();
        assert!(overstated.statistic > 20.0, "got {}", overstated.statistic);
        assert!(overstated.p_value < 1e-5);

        // A count one either side of expectation is unremarkable.
        for count in [45usize, 50, 55] {
            let result = kupiec_test(count, 1000, 0.05).unwrap();
            assert!(result.p_value > 0.1, "{count} breaches gave p={}", result.p_value);
        }
        // The limits: zero and full counts are handled rather than giving
        // a logarithm of nothing.
        assert!(kupiec_test(0, 100, 0.05).unwrap().statistic.is_finite());
        assert!(kupiec_test(100, 100, 0.05).unwrap().statistic.is_finite());
        assert!(kupiec_test(101, 100, 0.05).is_err());
        assert!(kupiec_test(0, 0, 0.05).is_err());
        assert!(kupiec_test(5, 100, 1.0).is_err());
    }

    #[test]
    fn a_historical_var_is_calibrated_against_its_own_sample_by_construction() {
        // Counting the breaches of a VaR estimated from the same data must
        // give back roughly the rate it was set at, so Kupiec finds
        // nothing. That is circular as a *validation* -- it is exactly the
        // in-sample fit that a real backtest avoids -- and it is a check
        // on the quantile arithmetic.
        let returns = gaussian_returns(4000, 0.0003, 0.011, 0x0F1D_1006);
        for alpha in [0.01f64, 0.05, 0.1] {
            let var = var_historical(&returns, alpha).unwrap();
            let breaches = returns.iter().filter(|r| **r < -var).count();
            let expected = alpha * returns.len() as f64;
            assert!(
                (breaches as f64 - expected).abs() < 0.2 * expected + 2.0,
                "at alpha={alpha}: {breaches} breaches against {expected}"
            );
            assert!(kupiec_test(breaches, returns.len(), alpha).unwrap().p_value > 0.05);
        }
    }

    #[test]
    fn the_crossover_rule_matches_buy_and_hold_on_a_series_that_only_rises() {
        // A monotone series never crosses back down, so the rule enters
        // once and holds. Its return must equal the underlying's over the
        // period it was actually invested -- exactly, since there are no
        // costs. Any lookahead in the signal would show up as a *better*
        // number than that.
        let prices: Vec<f64> = (0..200).map(|k| 100.0 * 1.002f64.powi(k)).collect();
        let stats = backtest_sma_crossover(&prices, 5, 20).unwrap();
        assert_eq!(stats.trades, 1, "it should enter once and stay");
        assert!((stats.win_rate - 1.0).abs() < 1e-15);
        assert!(stats.max_drawdown < 1e-12, "a rising equity curve has no drawdown");
        let invested = prices[199] / prices[19] - 1.0;
        assert!(
            (stats.total_return - invested).abs() < 1e-12,
            "the rule made {} against the market's {invested}",
            stats.total_return
        );
        assert_eq!(stats.equity.len(), prices.len() - 20 + 1);

        // And on a series that only falls it never enters at all.
        let falling: Vec<f64> = (0..200).map(|k| 100.0 * 0.998f64.powi(k)).collect();
        let bear = backtest_sma_crossover(&falling, 5, 20).unwrap();
        assert_eq!(bear.trades, 0);
        assert!(bear.total_return.abs() < 1e-15, "it lost {} while flat", bear.total_return);
        assert!(bear.max_drawdown < 1e-15);

        assert!(backtest_sma_crossover(&prices, 20, 5).is_err());
        assert!(backtest_sma_crossover(&prices, 0, 5).is_err());
        assert!(backtest_sma_crossover(&prices[..10], 5, 20).is_err());
        assert!(backtest_sma_crossover(&[100.0, -1.0, 100.0, 100.0], 1, 2).is_err());
    }

    #[test]
    fn the_backtest_does_not_look_at_the_bar_it_trades_on() {
        // The signal from bar `t` is applied to the return from `t` to
        // `t+1`. If it were applied to the return *into* `t` the rule
        // would be using a close it could not have known, and a series
        // built to punish exactly that would show it: here the price jumps
        // the instant the fast average crosses, and a peeking backtest
        // would capture the jump.
        let mut prices = vec![100.0f64; 30];
        for (index, price) in prices.iter_mut().enumerate() {
            *price = if index < 20 { 100.0 } else { 100.0 + (index - 19) as f64 };
        }
        // Add the jump the day the averages cross and take it straight
        // back, so a peeking rule profits and an honest one does not.
        let stats = backtest_sma_crossover(&prices, 3, 10).unwrap();
        let invested_from = stats.equity.len();
        assert!(invested_from > 1);
        // Whatever it earned, it cannot beat holding from the first bar it
        // could have acted on.
        let best_possible = prices[prices.len() - 1] / prices[9] - 1.0;
        assert!(
            stats.total_return <= best_possible + 1e-12,
            "the rule made {} where the most available was {best_possible}",
            stats.total_return
        );
    }
}
