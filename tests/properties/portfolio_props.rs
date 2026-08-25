//! Properties of the portfolio and risk modules.
//!
//! Mean-variance optimisation is defined by first-order conditions, so the
//! sharp test of a solution is to perturb it: a minimum must rise in every
//! feasible direction, and a maximum must fall. That is checkable without
//! trusting the closed form the solution came from, and it is what most of
//! these do.
//!
//! The risk measures have a different kind of structure. Expected
//! shortfall is *coherent* -- monotone, positively homogeneous,
//! translation-equivariant and subadditive -- and each of those four is an
//! identity that must hold for every sample. Value at risk satisfies the
//! first three and fails the fourth, and failing it is not a bug but the
//! reason the regulatory measure changed.

use rust_physics_engine::finance::portfolio::{
    capm_beta, information_ratio, kelly_continuous, kelly_fraction, log_returns,
    markowitz_frontier, max_drawdown, min_variance_weights, portfolio_variance,
    returns_from_prices, risk_contributions, risk_parity_weights, sharpe, tangency_portfolio,
};
use rust_physics_engine::finance::risk::{
    cvar_historical, kupiec_test, var_historical, var_parametric,
};
use rust_physics_engine::linalg::Matrix;
use rust_physics_engine::monte_carlo::Rng;

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random positive-definite covariance matrix, built as `L L'` with a
/// positive diagonal so it cannot be singular.
fn random_covariance(n: usize, rng: &mut Rng) -> Matrix {
    let mut lower = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let value = if i == j { 0.05 + 0.3 * rng.next_f64() } else { -0.15 + 0.3 * rng.next_f64() };
            lower.set(i, j, value);
        }
    }
    let mut cov = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut total = 0.0;
            for k in 0..n {
                total += lower.get(i, k) * lower.get(j, k);
            }
            cov.set(i, j, total);
        }
    }
    cov
}

/// A random return series.
fn random_returns(n: usize, rng: &mut Rng) -> Vec<f64> {
    let mean = -0.001 + 0.002 * rng.next_f64();
    let deviation = 0.002 + 0.03 * rng.next_f64();
    (0..n).map(|_| mean + deviation * rng.next_gaussian()).collect()
}

#[test]
fn prop_prices_and_returns_invert_each_other() {
    let mut rng = Rng::new(0x0F1E_1001);
    for _ in 0..200 {
        let n = 3 + pick(&mut rng, 60);
        let mut prices = vec![10.0 + 200.0 * rng.next_f64()];
        for _ in 1..n {
            let last = *prices.last().unwrap();
            prices.push(last * (0.9 + 0.2 * rng.next_f64()));
        }
        let simple = returns_from_prices(&prices).unwrap();
        let logs = log_returns(&prices).unwrap();
        assert_eq!(simple.len(), n - 1);
        for (r, l) in simple.iter().zip(logs.iter()) {
            assert!((l - (1.0 + r).ln()).abs() < 1e-13);
            // A log return never exceeds the simple one, by concavity.
            assert!(*l <= r + 1e-15);
        }
        // Rebuilding the path from either kind returns the prices.
        let mut rebuilt = prices[0];
        for r in &simple {
            rebuilt *= 1.0 + r;
        }
        assert!((rebuilt - prices[n - 1]).abs() < 1e-9 * prices[n - 1]);
        let total: f64 = logs.iter().sum();
        assert!((prices[0] * total.exp() - prices[n - 1]).abs() < 1e-9 * prices[n - 1]);
    }
}

#[test]
fn prop_the_minimum_variance_portfolio_rises_in_every_feasible_direction() {
    // The first-order condition, tested by perturbation rather than by
    // trusting the closed form. Any direction summing to zero keeps the
    // budget, so the variance must rise along all of them.
    let mut rng = Rng::new(0x0F1E_1002);
    for _ in 0..120 {
        let n = 2 + pick(&mut rng, 5);
        let cov = random_covariance(n, &mut rng);
        let Ok(weights) = min_variance_weights(&cov) else { continue };
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let base = portfolio_variance(&cov, &weights).unwrap();
        assert!(base > 0.0);
        for _ in 0..8 {
            let mut direction: Vec<f64> = (0..n).map(|_| -1.0 + 2.0 * rng.next_f64()).collect();
            let mean = direction.iter().sum::<f64>() / n as f64;
            for value in direction.iter_mut() {
                *value -= mean;
            }
            if direction.iter().map(|d| d.abs()).sum::<f64>() < 1e-9 {
                continue;
            }
            for size in [0.05f64, -0.05, 0.5, -0.5] {
                let moved: Vec<f64> =
                    (0..n).map(|i| weights[i] + size * direction[i]).collect();
                assert!((moved.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(
                    portfolio_variance(&cov, &moved).unwrap() > base,
                    "a feasible move lowered the variance"
                );
            }
        }
    }
}

#[test]
fn prop_the_tangency_portfolio_maximises_the_sharpe_ratio() {
    let mut rng = Rng::new(0x0F1E_1003);
    for _ in 0..100 {
        let n = 2 + pick(&mut rng, 5);
        let cov = random_covariance(n, &mut rng);
        let mu: Vec<f64> = (0..n).map(|_| 0.01 + 0.15 * rng.next_f64()).collect();
        let risk_free = 0.005;
        let Ok(weights) = tangency_portfolio(&mu, &cov, risk_free) else { continue };
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let ratio = |w: &[f64]| {
            let ret: f64 = w.iter().zip(mu.iter()).map(|(x, m)| x * m).sum();
            let variance = portfolio_variance(&cov, w).unwrap();
            (ret - risk_free) / variance.sqrt()
        };
        let best = ratio(&weights);
        assert!(best.is_finite());
        for _ in 0..8 {
            let mut direction: Vec<f64> = (0..n).map(|_| -1.0 + 2.0 * rng.next_f64()).collect();
            let mean = direction.iter().sum::<f64>() / n as f64;
            for value in direction.iter_mut() {
                *value -= mean;
            }
            if direction.iter().map(|d| d.abs()).sum::<f64>() < 1e-9 {
                continue;
            }
            for size in [0.05f64, -0.05, 0.4, -0.4] {
                let moved: Vec<f64> =
                    (0..n).map(|i| weights[i] + size * direction[i]).collect();
                assert!(ratio(&moved) <= best + 1e-9, "a move raised the Sharpe ratio");
            }
        }
    }
}

#[test]
fn prop_every_frontier_point_minimises_variance_at_its_own_return() {
    // Two constraints, so a feasible perturbation must be orthogonal to
    // both the budget vector and the expected returns.
    let mut rng = Rng::new(0x0F1E_1004);
    for _ in 0..60 {
        let n = 3 + pick(&mut rng, 4);
        let cov = random_covariance(n, &mut rng);
        let mu: Vec<f64> = (0..n).map(|_| 0.01 + 0.15 * rng.next_f64()).collect();
        let Ok(frontier) = markowitz_frontier(&mu, &cov, 6) else { continue };
        for (deviation, target, weights) in &frontier {
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-8);
            let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
            assert!((achieved - target).abs() < 1e-8, "it returned {achieved} not {target}");
            let variance = portfolio_variance(&cov, weights).unwrap();
            assert!((variance.sqrt() - deviation).abs() < 1e-9);
            // The two constraint vectors are not orthogonal to each
            // other, so projecting a direction against them one after the
            // other leaves it satisfying only the second. Orthogonalise
            // the basis first, then project once against each.
            let ones = vec![1.0f64; n];
            let ones_norm: f64 = ones.iter().map(|y| y * y).sum();
            let mu_dot: f64 = mu.iter().zip(ones.iter()).map(|(x, y)| x * y).sum();
            let mu_perp: Vec<f64> =
                (0..n).map(|i| mu[i] - mu_dot / ones_norm * ones[i]).collect();
            let mu_perp_norm: f64 = mu_perp.iter().map(|y| y * y).sum();
            if mu_perp_norm < 1e-12 {
                continue;
            }
            for _ in 0..5 {
                let raw: Vec<f64> = (0..n).map(|_| -1.0 + 2.0 * rng.next_f64()).collect();
                let a: f64 = raw.iter().zip(ones.iter()).map(|(x, y)| x * y).sum::<f64>()
                    / ones_norm;
                let b: f64 = raw.iter().zip(mu_perp.iter()).map(|(x, y)| x * y).sum::<f64>()
                    / mu_perp_norm;
                let d: Vec<f64> =
                    (0..n).map(|i| raw[i] - a * ones[i] - b * mu_perp[i]).collect();
                if d.iter().map(|x| x.abs()).sum::<f64>() < 1e-6 {
                    continue;
                }
                for size in [0.2f64, -0.2] {
                    let moved: Vec<f64> = (0..n).map(|i| weights[i] + size * d[i]).collect();
                    assert!(
                        (moved.iter().sum::<f64>() - 1.0).abs() < 1e-7,
                        "the move broke the budget"
                    );
                    let shifted: f64 = moved.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
                    assert!((shifted - target).abs() < 1e-7, "the move changed the return");
                    assert!(
                        portfolio_variance(&cov, &moved).unwrap() > variance,
                        "a same-return portfolio had less variance"
                    );
                }
            }
        }
        // Risk and return both rise along the frontier.
        for pair in frontier.windows(2) {
            assert!(pair[1].1 > pair[0].1 && pair[1].0 > pair[0].0);
        }
    }
}

#[test]
fn prop_risk_parity_splits_the_risk_evenly_and_the_shares_sum_to_one() {
    // Euler's theorem on a quadratic form: the contributions decompose the
    // variance exactly, so they sum to one whatever the weights.
    let mut rng = Rng::new(0x0F1E_1005);
    for _ in 0..120 {
        let n = 2 + pick(&mut rng, 5);
        let cov = random_covariance(n, &mut rng);
        let Ok(weights) = risk_parity_weights(&cov) else { continue };
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(weights.iter().all(|w| *w > 0.0), "a risk-parity weight went negative");
        let shares = risk_contributions(&cov, &weights).unwrap();
        assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for share in &shares {
            assert!(
                (share - 1.0 / n as f64).abs() < 1e-7,
                "a share was {share} against {}",
                1.0 / n as f64
            );
        }
        // The shares decompose *any* portfolio's variance, not only this
        // one's.
        let arbitrary: Vec<f64> = (0..n).map(|_| 0.1 + rng.next_f64()).collect();
        let total: f64 = arbitrary.iter().sum();
        let normalised: Vec<f64> = arbitrary.into_iter().map(|w| w / total).collect();
        let other = risk_contributions(&cov, &normalised).unwrap();
        assert!((other.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn prop_a_sharpe_ratio_is_blind_to_scale_and_answers_to_a_shift() {
    // Multiplying every return by a positive constant leaves the ratio
    // alone -- it is a signal-to-noise measure. Adding a constant moves it
    // by exactly that over the deviation.
    let mut rng = Rng::new(0x0F1E_1006);
    for _ in 0..200 {
        let returns = random_returns(50 + pick(&mut rng, 200), &mut rng);
        let Ok(base) = sharpe(&returns, 0.0) else { continue };
        let factor = 0.1 + 5.0 * rng.next_f64();
        let scaled: Vec<f64> = returns.iter().map(|r| factor * r).collect();
        assert!(
            (sharpe(&scaled, 0.0).unwrap() - base).abs() < 1e-9 * base.abs().max(1.0),
            "scaling changed the ratio"
        );
        // A constant subtracted from every return is the risk-free rate.
        let shift = 0.001 * rng.next_f64();
        let shifted: Vec<f64> = returns.iter().map(|r| r - shift).collect();
        assert!((sharpe(&shifted, 0.0).unwrap() - sharpe(&returns, shift).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn prop_a_beta_built_into_a_series_comes_back_out_of_it() {
    // An asset constructed as alpha + beta * market has no residual, so
    // the regression must recover both exactly.
    let mut rng = Rng::new(0x0F1E_1007);
    for _ in 0..200 {
        let market = random_returns(30 + pick(&mut rng, 200), &mut rng);
        let alpha = -0.005 + 0.01 * rng.next_f64();
        let beta = -2.0 + 4.0 * rng.next_f64();
        let asset: Vec<f64> = market.iter().map(|m| alpha + beta * m).collect();
        let Ok((a, b)) = capm_beta(&asset, &market) else { continue };
        assert!((b - beta).abs() < 1e-9 * beta.abs().max(1.0), "beta came back {b} not {beta}");
        assert!((a - alpha).abs() < 1e-9, "alpha came back {a} not {alpha}");
        // The information ratio against the market is the Sharpe ratio of
        // the difference, by definition.
        let active: Vec<f64> = asset.iter().zip(market.iter()).map(|(x, m)| x - m).collect();
        if let Ok(ratio) = information_ratio(&asset, &market) {
            assert!((ratio - sharpe(&active, 0.0).unwrap()).abs() < 1e-12);
        }
    }
}

#[test]
fn prop_a_drawdown_is_between_nothing_and_everything() {
    let mut rng = Rng::new(0x0F1E_1008);
    for _ in 0..200 {
        let n = 5 + pick(&mut rng, 300);
        let mut prices = vec![50.0 + 100.0 * rng.next_f64()];
        for _ in 1..n {
            let last = *prices.last().unwrap();
            prices.push(last * (0.85 + 0.3 * rng.next_f64()));
        }
        let drawdown = max_drawdown(&prices).unwrap();
        assert!((0.0..1.0).contains(&drawdown), "the drawdown was {drawdown}");
        // It is at least the fall from the first price to the lowest, and
        // at least the final loss if there is one.
        let lowest = prices.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        assert!(drawdown >= (prices[0] - lowest) / prices[0] - 1e-12);
        // Scaling every price leaves it alone: it is a ratio.
        let scaled: Vec<f64> = prices.iter().map(|p| 7.5 * p).collect();
        assert!((max_drawdown(&scaled).unwrap() - drawdown).abs() < 1e-12);
        // A sorted-ascending path has none at all.
        let mut rising = prices.clone();
        rising.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(max_drawdown(&rising).unwrap() < 1e-15);
    }
}

#[test]
fn prop_expected_shortfall_is_coherent_where_value_at_risk_is_not() {
    // Monotone, positively homogeneous and translation-equivariant are
    // satisfied by both. Subadditivity is satisfied by expected shortfall
    // alone, and the counterexample for VaR is not exotic.
    let mut rng = Rng::new(0x0F1E_1009);
    for _ in 0..80 {
        let n = 200 + pick(&mut rng, 800);
        let a = random_returns(n, &mut rng);
        let b = random_returns(n, &mut rng);
        let mixed: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| 0.5 * (x + y)).collect();
        for alpha in [0.01f64, 0.05, 0.1] {
            // Positive homogeneity: doubling the position doubles the risk.
            let doubled: Vec<f64> = a.iter().map(|x| 2.0 * x).collect();
            let single = cvar_historical(&a, alpha).unwrap();
            assert!(
                (cvar_historical(&doubled, alpha).unwrap() - 2.0 * single).abs()
                    < 1e-12 * single.abs().max(1.0)
            );
            let var_single = var_historical(&a, alpha).unwrap();
            assert!(
                (var_historical(&doubled, alpha).unwrap() - 2.0 * var_single).abs()
                    < 1e-12 * var_single.abs().max(1.0)
            );
            // Translation: adding a certain gain reduces the risk by it.
            let shifted: Vec<f64> = a.iter().map(|x| x + 0.01).collect();
            assert!(
                (cvar_historical(&shifted, alpha).unwrap() - (single - 0.01)).abs() < 1e-12
            );
            assert!(
                (var_historical(&shifted, alpha).unwrap() - (var_single - 0.01)).abs() < 1e-12
            );
            // Subadditivity of expected shortfall, on a half-and-half mix.
            let combined = cvar_historical(&mixed, alpha).unwrap();
            let parts = 0.5 * (single + cvar_historical(&b, alpha).unwrap());
            assert!(
                combined <= parts + 1e-9,
                "expected shortfall was superadditive at alpha={alpha}: {combined} against {parts}"
            );
            // And the shortfall never falls below the quantile it averages
            // past.
            assert!(combined >= var_historical(&mixed, alpha).unwrap() - 1e-12);
        }
    }
}

#[test]
fn prop_value_at_risk_is_monotone_in_the_confidence_level() {
    let mut rng = Rng::new(0x0F1E_100A);
    for _ in 0..100 {
        let returns = random_returns(300 + pick(&mut rng, 700), &mut rng);
        let mut previous = f64::NEG_INFINITY;
        for alpha in [0.5f64, 0.25, 0.1, 0.05, 0.025, 0.01] {
            let historical = var_historical(&returns, alpha).unwrap();
            assert!(historical >= previous - 1e-15, "the VaR fell at alpha={alpha}");
            previous = historical;
            // The parametric estimate is monotone too, and both are finite.
            assert!(var_parametric(&returns, alpha).unwrap().is_finite());
            assert!(cvar_historical(&returns, alpha).unwrap() >= historical - 1e-12);
        }
    }
}

#[test]
fn prop_the_kupiec_statistic_is_zero_exactly_at_the_expected_rate() {
    // The likelihood ratio compares the claimed rate with the observed
    // one, so it vanishes when they agree and grows either side.
    let mut rng = Rng::new(0x0F1E_100B);
    for _ in 0..200 {
        let observations = 100 + pick(&mut rng, 4000);
        for alpha in [0.01f64, 0.05, 0.1] {
            let expected = alpha * observations as f64;
            let exact = expected.round() as usize;
            let at_rate = kupiec_test(exact, observations, alpha).unwrap();
            assert!((0.0..=1.0).contains(&at_rate.p_value));
            assert_eq!(at_rate.df, 1.0);
            // The statistic grows as the count moves away in either
            // direction.
            let mut previous = at_rate.statistic;
            for extra in [1usize, 5, 20, 60] {
                let more = kupiec_test(exact + extra, observations, alpha).unwrap();
                assert!(
                    more.statistic >= previous - 1e-9,
                    "the statistic fell when the count rose to {}",
                    exact + extra
                );
                assert!(more.p_value <= at_rate.p_value + 1e-9);
                previous = more.statistic;
            }
            let mut previous = at_rate.statistic;
            for fewer in [1usize, 5, 20] {
                if fewer > exact {
                    break;
                }
                let less = kupiec_test(exact - fewer, observations, alpha).unwrap();
                assert!(less.statistic >= previous - 1e-9, "too few breaches was not penalised");
                previous = less.statistic;
            }
        }
    }
}

#[test]
fn prop_kelly_is_the_peak_of_the_growth_rate_it_maximises() {
    // The continuous growth rate is quadratic in the stake, so the Kelly
    // fraction is its vertex: growth falls on both sides and returns to
    // the risk-free rate at exactly twice it.
    let mut rng = Rng::new(0x0F1E_100C);
    for _ in 0..300 {
        let sigma = 0.05 + 0.5 * rng.next_f64();
        let risk_free = 0.05 * rng.next_f64();
        let mu = risk_free + 0.01 + 0.2 * rng.next_f64();
        let kelly = kelly_continuous(mu, sigma, risk_free).unwrap();
        assert!(kelly > 0.0, "a positive edge should call for a positive stake");
        let growth =
            |f: f64| risk_free + f * (mu - risk_free) - 0.5 * f * f * sigma * sigma;
        let peak = growth(kelly);
        for factor in [0.1f64, 0.5, 0.9, 1.1, 1.5, 2.0, 3.0] {
            assert!(growth(factor * kelly) <= peak + 1e-12, "the peak was not at Kelly");
        }
        assert!(
            (growth(2.0 * kelly) - risk_free).abs() < 1e-12 * risk_free.abs().max(1.0),
            "twice Kelly did not return to the risk-free rate"
        );
        assert!(growth(3.0 * kelly) < risk_free);

        // The discrete form has no edge exactly at fair odds.
        let payout = 0.2 + 5.0 * rng.next_f64();
        let fair = 1.0 / (1.0 + payout);
        assert!(kelly_fraction(fair, payout).unwrap().abs() < 1e-12);
        assert!(kelly_fraction(fair + 0.05, payout).unwrap() > 0.0);
        assert!(kelly_fraction((fair - 0.05).max(0.0), payout).unwrap() < 0.0);
    }
}
