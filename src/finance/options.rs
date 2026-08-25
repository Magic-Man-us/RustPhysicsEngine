//! Option pricing: closed forms, lattices, Monte Carlo and a PDE solver.
//!
//! # Conventions
//!
//! Rates and volatilities are continuously compounded and annualised;
//! time is in years. `q` is a continuous dividend yield, which also serves
//! as a foreign interest rate for a currency option and as a convenience
//! yield for a commodity. A `call: bool` argument names the payoff:
//! `max(S - K, 0)` when true and `max(K - S, 0)` when false.
//!
//! # Why there are so many methods for one number
//!
//! They price different things, and where they overlap they check each
//! other. [`black_scholes`] is exact but only for a European payoff on a
//! lognormal process. A lattice ([`binomial_crr`], [`trinomial`]) handles
//! early exercise, at the cost of converging to the closed form only in
//! the limit -- and it converges by oscillating around the answer, not by
//! approaching it from one side. Monte Carlo ([`monte_carlo_european`] and
//! the path-dependent payoffs) handles anything you can simulate, and
//! pays for that with an error that falls like the square root of the
//! path count, which is why the variance reduction here is not an
//! optimisation but the difference between usable and not.
//!
//! # The volatility argument is the whole problem
//!
//! Black-Scholes takes one volatility for all strikes. Real option prices
//! do not admit one: the implied volatilities of options on the same
//! underlying and expiry form a smile, and a model with a single sigma
//! cannot produce it. That is not a defect in the arithmetic, it is the
//! lognormal assumption failing. [`merton_jump_price`] and the Heston
//! model add mechanisms that generate a smile, and
//! [`volatility_smile_svi`] simply parameterises one without a mechanism.

use crate::error::GeomError;
use crate::monte_carlo::Rng;
use crate::statistics::distributions::{gaussian, gaussian_cdf};

/// The standard normal cumulative distribution.
fn n(x: f64) -> f64 {
    gaussian_cdf(x, 0.0, 1.0)
}

/// The standard normal density.
fn phi(x: f64) -> f64 {
    gaussian(x, 0.0, 1.0)
}

/// The first-order sensitivities of an option price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    /// Sensitivity to the underlying price.
    pub delta: f64,
    /// Sensitivity of delta to the underlying price.
    pub gamma: f64,
    /// Sensitivity to volatility, per unit of volatility.
    pub vega: f64,
    /// Sensitivity to the passage of time, per year.
    pub theta: f64,
    /// Sensitivity to the interest rate.
    pub rho: f64,
}

fn check_inputs(s: f64, k: f64, t: f64, sigma: f64) -> Result<(), GeomError> {
    if !(s > 0.0) || !(k > 0.0) || !(t >= 0.0) || !(sigma >= 0.0) {
        return Err(GeomError::InvalidArgument(
            "price, strike, time and volatility must be positive and finite",
        ));
    }
    if !s.is_finite() || !k.is_finite() || !t.is_finite() || !sigma.is_finite() {
        return Err(GeomError::InvalidArgument("an option parameter is not finite"));
    }
    Ok(())
}

/// `d1` and `d2` of the Black-Scholes formula.
fn d1_d2(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> (f64, f64) {
    let vol = sigma * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vol;
    (d1, d1 - vol)
}

/// The value at expiry, which is also the value of a zero-volatility or
/// zero-maturity option discounted appropriately.
fn intrinsic(s: f64, k: f64, t: f64, r: f64, q: f64, call: bool) -> f64 {
    let forward = s * (-q * t).exp();
    let discounted = k * (-r * t).exp();
    if call {
        (forward - discounted).max(0.0)
    } else {
        (discounted - forward).max(0.0)
    }
}

/// The Black-Scholes-Merton price of a European option.
///
/// `S e^(-qT) N(d1) - K e^(-rT) N(d2)` for a call, and the mirror for a
/// put. The two terms are not "probability times payoff": the first is the
/// value of receiving the share if exercised, computed under a measure in
/// which the share is the numeraire, and the second is the strike times
/// the risk-neutral probability of exercise. Reading `N(d2)` as a
/// real-world probability is the commonest misreading of the formula --
/// it is a probability under a measure chosen to make discounted prices
/// martingales, and has nothing to say about what the share will do.
///
/// Zero volatility or zero time to expiry both collapse the formula to
/// the discounted intrinsic value, which is handled directly rather than
/// left to divide by zero.
///
/// # Errors
/// Returns an error for a non-positive price or strike, a negative time or
/// volatility, or any input that is not finite.
pub fn black_scholes(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
) -> Result<f64, GeomError> {
    check_inputs(s, k, t, sigma)?;
    if t == 0.0 || sigma == 0.0 {
        return Ok(intrinsic(s, k, t, r, q, call));
    }
    let (d1, d2) = d1_d2(s, k, t, r, sigma, q);
    let discounted_spot = s * (-q * t).exp();
    let discounted_strike = k * (-r * t).exp();
    Ok(if call {
        discounted_spot * n(d1) - discounted_strike * n(d2)
    } else {
        discounted_strike * n(-d2) - discounted_spot * n(-d1)
    })
}

/// The Black-Scholes Greeks.
///
/// `vega` is per unit of volatility (so divide by 100 for "per volatility
/// point"), `theta` is per year (divide by 365 for a daily decay), and
/// `rho` is per unit of rate. Those conventions differ between desks and
/// are the commonest source of a factor of a hundred.
///
/// Gamma and vega are the same for a call and a put, because the two
/// differ by a forward contract, which is linear in the spot and does not
/// depend on volatility at all. That identity is exact and is what the
/// tests check rather than the individual numbers.
///
/// # Errors
/// Returns an error for the same inputs as [`black_scholes`], and for a
/// zero time or volatility, where the derivatives do not exist.
pub fn bs_greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
) -> Result<Greeks, GeomError> {
    check_inputs(s, k, t, sigma)?;
    if t == 0.0 || sigma == 0.0 {
        return Err(GeomError::Degenerate("the Greeks are undefined at zero time or volatility"));
    }
    let (d1, d2) = d1_d2(s, k, t, r, sigma, q);
    let root_t = t.sqrt();
    let carry = (-q * t).exp();
    let discount = (-r * t).exp();
    let gamma = carry * phi(d1) / (s * sigma * root_t);
    let vega = s * carry * phi(d1) * root_t;
    let (delta, theta, rho) = if call {
        (
            carry * n(d1),
            -s * carry * phi(d1) * sigma / (2.0 * root_t) - r * k * discount * n(d2)
                + q * s * carry * n(d1),
            k * t * discount * n(d2),
        )
    } else {
        (
            -carry * n(-d1),
            -s * carry * phi(d1) * sigma / (2.0 * root_t) + r * k * discount * n(-d2)
                - q * s * carry * n(-d1),
            -k * t * discount * n(-d2),
        )
    };
    Ok(Greeks { delta, gamma, vega, theta, rho })
}

/// The volatility that reproduces an observed price, or `None` if no
/// volatility does.
///
/// Price is strictly increasing in volatility, so the root is unique where
/// it exists; the search brackets it by doubling and then bisects, taking
/// Newton steps where vega is large enough to trust and falling back to
/// bisection where it is not. Deep out-of-the-money options have vega
/// near zero over a wide range of volatilities, which is exactly where a
/// pure Newton iteration diverges and where the answer is least
/// meaningful.
///
/// `None` means no volatility can be recovered, for either of two
/// reasons. The price may be outside the model's range -- below the
/// no-arbitrage floor (the discounted intrinsic value), above the
/// ceiling, or unreachable at any volatility the doubling search reaches.
/// Or the price may simply not determine one: a deep in-the-money option
/// with weeks left has a vega around `1e-13`, and prices identically at
/// 5% and at 20% volatility to the last bit of a double. Returning a
/// number there would be reporting rounding noise as a measurement, so
/// the answer is withheld when vega falls below `1e-8` relative to the
/// price.
///
/// # Errors
/// Returns an error for a non-positive price or strike, a non-positive
/// time, or a negative observed price.
pub fn implied_volatility(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    call: bool,
) -> Result<Option<f64>, GeomError> {
    check_inputs(s, k, t, 0.0)?;
    if !(t > 0.0) || price < 0.0 || !price.is_finite() {
        return Err(GeomError::InvalidArgument("implied_volatility: bad price or maturity"));
    }
    let floor = intrinsic(s, k, t, r, q, call);
    let ceiling = if call { s * (-q * t).exp() } else { k * (-r * t).exp() };
    if price < floor - 1e-12 || price > ceiling + 1e-12 {
        return Ok(None);
    }
    // Bracket by doubling from a sensible first guess.
    let mut low = 1e-9;
    let mut high = 0.2;
    let mut attempts = 0;
    while black_scholes(s, k, t, r, high, q, call)? < price {
        low = high;
        high *= 2.0;
        attempts += 1;
        if attempts > 12 {
            return Ok(None);
        }
    }
    let mut sigma = 0.5 * (low + high);
    for _ in 0..200 {
        // The stopping test is on the *volatility* bracket, not on the
        // price. Stopping when the price matches would return the first
        // volatility whose price is indistinguishable from the target,
        // and where vega is small that is a wide range: a deep in-the-
        // money option with weeks to run prices identically at 5% and at
        // 20% volatility to the last bit of a double.
        if high - low < 1e-13 * (1.0 + low) {
            break;
        }
        let value = black_scholes(s, k, t, r, sigma, q, call)?;
        let error = value - price;
        if error > 0.0 {
            high = sigma;
        } else {
            low = sigma;
        }
        let vega = bs_greeks(s, k, t, r, sigma, q, call)?.vega;
        // A Newton step is only worth taking where vega is large enough
        // for the derivative to mean something and where it lands inside
        // the bracket; otherwise bisect, which cannot fail.
        let stepped = sigma - error / vega;
        sigma = if vega > 1e-8 && stepped > low && stepped < high {
            stepped
        } else {
            0.5 * (low + high)
        };
    }
    // Having found the root, ask whether the price determined it. Vega is
    // the rate at which the price carries information about volatility;
    // where it is negligible, the price is flat to within double
    // precision and any answer here would be an artefact of rounding.
    let vega = bs_greeks(s, k, t, r, sigma, q, call).map_or(0.0, |g| g.vega);
    if vega < 1e-8 * price.max(1.0) {
        return Ok(None);
    }
    Ok(Some(sigma))
}

/// The put-call parity residual: `C - P - S e^(-qT) + K e^(-rT)`.
///
/// Zero for any pair of European prices that admit no arbitrage,
/// *whatever* model produced them, because the identity follows from the
/// payoffs alone: holding a call and selling a put is the same as holding
/// the forward. A residual is therefore a statement about the prices, not
/// about the model, and this is the sharpest check available on a pricing
/// routine that has no closed form to compare with.
#[must_use]
pub fn put_call_parity_check(call: f64, put: f64, s: f64, k: f64, t: f64, r: f64, q: f64) -> f64 {
    call - put - s * (-q * t).exp() + k * (-r * t).exp()
}

/// The Cox-Ross-Rubinstein binomial tree.
///
/// Up and down moves of `e^(±sigma sqrt(dt))` with the risk-neutral
/// probability that makes the discounted price a martingale. Set
/// `american` to allow exercise at every node.
///
/// Convergence to Black-Scholes is `O(1/steps)` but *oscillatory*: the
/// error alternates in sign as the strike moves between two adjacent
/// terminal nodes, so a tree with 101 steps can be further from the answer
/// than one with 100. Averaging two consecutive step counts removes most
/// of it, and is why an odd-even pair is the honest way to quote a
/// lattice price.
///
/// # Errors
/// Returns an error for bad option parameters, zero steps, more than
/// twenty thousand steps, or a `dt` so large that the risk-neutral
/// probability leaves `[0, 1]` -- which happens when the drift outruns
/// what the volatility can span in one step.
pub fn binomial_crr(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    steps: usize,
    call: bool,
    american: bool,
) -> Result<f64, GeomError> {
    check_inputs(s, k, t, sigma)?;
    if steps == 0 || steps > 20_000 {
        return Err(GeomError::InvalidArgument("binomial_crr: bad step count"));
    }
    if t == 0.0 || sigma == 0.0 {
        return Ok(intrinsic(s, k, t, r, q, call));
    }
    let dt = t / steps as f64;
    let up = (sigma * dt.sqrt()).exp();
    let down = 1.0 / up;
    let growth = ((r - q) * dt).exp();
    let p = (growth - down) / (up - down);
    if !(0.0..=1.0).contains(&p) {
        return Err(GeomError::Degenerate(
            "the risk-neutral probability left [0, 1]: the step is too coarse for this drift",
        ));
    }
    let discount = (-r * dt).exp();
    let payoff = |price: f64| if call { (price - k).max(0.0) } else { (k - price).max(0.0) };
    // Terminal layer: node j has j up moves.
    let mut values: Vec<f64> =
        (0..=steps).map(|j| payoff(s * up.powi(j as i32) * down.powi((steps - j) as i32))).collect();
    for layer in (0..steps).rev() {
        for j in 0..=layer {
            let held = discount * (p * values[j + 1] + (1.0 - p) * values[j]);
            values[j] = if american {
                let price = s * up.powi(j as i32) * down.powi((layer - j) as i32);
                held.max(payoff(price))
            } else {
                held
            };
        }
    }
    Ok(values[0])
}

/// A trinomial tree with an up, down and unchanged move.
///
/// The third branch buys a free parameter, used here to set the space step
/// to `sigma sqrt(3 dt)`, which is the choice that makes the tree stable
/// and its convergence smoother than the binomial's. It is the same
/// explicit finite-difference scheme as the binomial in different
/// clothing, and the extra branch is what keeps the scheme's coefficients
/// positive over a wider range of steps.
///
/// The probabilities here match the first two moments of the *log* price.
/// That is the usual construction and it has a consequence worth knowing:
/// unlike Cox-Ross-Rubinstein, whose up-probability is chosen to make the
/// price itself a martingale exactly, this tree is a martingale only to
/// `O(dt^2)`. So its call and put prices satisfy put-call parity only to
/// that order -- a residual of about `2e-3` on a two-and-a-half-year
/// option at seven steps, falling as `1/steps^2` and reaching `4e-8` by
/// sixteen hundred. The tree is arbitrage-free in the limit and not
/// before it. Use [`binomial_crr`] where an exactly consistent call and
/// put matter more than a smooth convergence.
///
/// # Errors
/// As [`binomial_crr`], with a lower step ceiling since the work is
/// quadratic in the step count.
pub fn trinomial(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    steps: usize,
    call: bool,
    american: bool,
) -> Result<f64, GeomError> {
    check_inputs(s, k, t, sigma)?;
    if steps == 0 || steps > 5_000 {
        return Err(GeomError::InvalidArgument("trinomial: bad step count"));
    }
    if t == 0.0 || sigma == 0.0 {
        return Ok(intrinsic(s, k, t, r, q, call));
    }
    let dt = t / steps as f64;
    let dx = sigma * (3.0 * dt).sqrt();
    let drift = r - q - 0.5 * sigma * sigma;
    let variance = sigma * sigma * dt;
    let mean = drift * dt;
    let p_up = 0.5 * ((variance + mean * mean) / (dx * dx) + mean / dx);
    let p_down = 0.5 * ((variance + mean * mean) / (dx * dx) - mean / dx);
    let p_mid = 1.0 - p_up - p_down;
    if p_up < 0.0 || p_down < 0.0 || p_mid < 0.0 {
        return Err(GeomError::Degenerate(
            "a transition probability went negative: the step is too coarse for this drift",
        ));
    }
    let discount = (-r * dt).exp();
    let payoff = |price: f64| if call { (price - k).max(0.0) } else { (k - price).max(0.0) };
    let width = 2 * steps + 1;
    let price_at = |node: isize| s * (node as f64 * dx).exp();
    let mut values: Vec<f64> =
        (0..width).map(|index| payoff(price_at(index as isize - steps as isize))).collect();
    for layer in (0..steps).rev() {
        let span = 2 * layer + 1;
        let mut next = vec![0.0; span];
        for index in 0..span {
            // The previous layer spans two more nodes and its levels are
            // shifted by one, so node `index` here reads `index`,
            // `index + 1` and `index + 2` there -- down, unchanged, up.
            let held = discount
                * (p_down * values[index] + p_mid * values[index + 1] + p_up * values[index + 2]);
            next[index] = if american {
                held.max(payoff(price_at(index as isize - layer as isize)))
            } else {
                held
            };
        }
        values = next;
    }
    Ok(values[0])
}

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

/// One terminal price from a lognormal path, given a standard normal draw.
fn terminal_price(s: f64, t: f64, r: f64, sigma: f64, q: f64, z: f64) -> f64 {
    s * ((r - q - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp()
}

fn check_paths(paths: usize) -> Result<(), GeomError> {
    if !(2..=20_000_000).contains(&paths) {
        return Err(GeomError::InvalidArgument("the path count is zero or beyond the budget"));
    }
    Ok(())
}

/// The sample mean and standard error of a set of discounted payoffs.
fn summarise(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// A European option by Monte Carlo, returning `(price, standard error)`.
///
/// Two variance reductions are applied, and both are exact rather than
/// heuristic:
///
/// *Antithetic variates* price each draw with `z` and `-z`. The pair has
/// the same distribution as two independent draws, so the estimator stays
/// unbiased, and the negative correlation between the two payoffs shrinks
/// the variance of their mean.
///
/// *A control variate* uses the discounted terminal price, whose expected
/// value under the risk-neutral measure is exactly `S e^(-qT)` -- known,
/// not estimated. Subtracting `beta` times its error from each payoff
/// cannot bias the result whatever `beta` is, and choosing `beta` by
/// regression on the same sample minimises the variance.
///
/// The reported standard error is the error *of the reduced estimator*,
/// so it is the honest one to compare against the closed form: a price
/// two standard errors from Black-Scholes is a failure, and the tests
/// treat it as one.
///
/// # Errors
/// Returns an error for bad option parameters or a path count outside
/// `[2, 2e7]`.
pub fn monte_carlo_european(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
    paths: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    check_inputs(s, k, t, sigma)?;
    check_paths(paths)?;
    let discount = (-r * t).exp();
    let payoff = |price: f64| if call { (price - k).max(0.0) } else { (k - price).max(0.0) };
    let mut payoffs = Vec::with_capacity(paths);
    let mut controls = Vec::with_capacity(paths);
    let pairs = paths.div_ceil(2);
    for _ in 0..pairs {
        let z = rng.next_gaussian();
        for sign in [1.0, -1.0] {
            let price = terminal_price(s, t, r, sigma, q, sign * z);
            payoffs.push(discount * payoff(price));
            controls.push(discount * price);
        }
    }
    payoffs.truncate(paths);
    controls.truncate(paths);
    // The control's expectation is known exactly under the risk-neutral
    // measure: the discounted spot grows at the carry.
    let expected_control = s * (-q * t).exp();
    let count = paths as f64;
    let mean_control = controls.iter().sum::<f64>() / count;
    let mean_payoff = payoffs.iter().sum::<f64>() / count;
    let covariance: f64 = payoffs
        .iter()
        .zip(controls.iter())
        .map(|(p, c)| (p - mean_payoff) * (c - mean_control))
        .sum::<f64>();
    let control_variance: f64 = controls.iter().map(|c| (c - mean_control).powi(2)).sum::<f64>();
    let beta = if control_variance > 0.0 { covariance / control_variance } else { 0.0 };
    let adjusted: Vec<f64> = payoffs
        .iter()
        .zip(controls.iter())
        .map(|(p, c)| p - beta * (c - expected_control))
        .collect();
    Ok(summarise(&adjusted))
}

/// A lognormal path sampled at `steps` equal intervals, returned without
/// the initial price.
fn lognormal_path(
    s: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    steps: usize,
    rng: &mut Rng,
    into: &mut Vec<f64>,
) {
    into.clear();
    let dt = t / steps as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();
    let mut price = s;
    for _ in 0..steps {
        price *= (drift + diffusion * rng.next_gaussian()).exp();
        into.push(price);
    }
}

/// An arithmetic-average Asian option by Monte Carlo, returning
/// `(price, standard error)`.
///
/// The average is taken over the `steps` monitoring dates, excluding the
/// start. Averaging is what makes the option cheaper than its European
/// twin: the average of a lognormal has lower variance than its terminal
/// value, and lower variance means a lower option price at the same
/// forward.
///
/// There is no closed form for the arithmetic average -- the sum of
/// lognormals is not lognormal -- which is why this is a simulation and
/// not a formula. The *geometric* average does have one, and that is what
/// makes a geometric control variate the standard variance reduction
/// here; it is not applied, so expect the error to fall only as the
/// square root of the path count.
///
/// # Errors
/// Returns an error for bad option parameters, a path count outside
/// `[2, 2e7]`, a step count of zero, or more than fifty million total
/// steps.
pub fn monte_carlo_asian(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
    steps: usize,
    paths: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    check_inputs(s, k, t, sigma)?;
    check_paths(paths)?;
    if steps == 0 || steps.saturating_mul(paths) > 50_000_000 {
        return Err(GeomError::InvalidArgument("monte_carlo_asian: bad step count"));
    }
    let discount = (-r * t).exp();
    let mut path = Vec::with_capacity(steps);
    let mut values = Vec::with_capacity(paths);
    for _ in 0..paths {
        lognormal_path(s, t, r, sigma, q, steps, rng, &mut path);
        let average = path.iter().sum::<f64>() / steps as f64;
        let payoff = if call { (average - k).max(0.0) } else { (k - average).max(0.0) };
        values.push(discount * payoff);
    }
    Ok(summarise(&values))
}

/// Which barrier a knock-out or knock-in option watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Barrier {
    /// Dies if the price ever rises to the barrier.
    UpAndOut,
    /// Dies if the price ever falls to the barrier.
    DownAndOut,
    /// Pays only if the price rises to the barrier at some point.
    UpAndIn,
    /// Pays only if the price falls to the barrier at some point.
    DownAndIn,
}

/// A barrier option by Monte Carlo, returning `(price, standard error)`.
///
/// The barrier is checked only at the `steps` monitoring dates. That is a
/// *discretely monitored* option and it is worth strictly more than a
/// continuously monitored one, because a path can cross the barrier and
/// come back between observations. The gap closes slowly, like
/// `1/sqrt(steps)`, so a daily-monitored option priced with twelve steps
/// is materially mispriced -- the discretisation is a modelling choice
/// here, not a numerical detail.
///
/// The in-out parity holds by construction: a knock-in and its matching
/// knock-out sum to the vanilla option, since every path pays into exactly
/// one of them.
///
/// # Errors
/// As [`monte_carlo_asian`], plus a non-positive barrier level.
pub fn monte_carlo_barrier(
    s: f64,
    k: f64,
    barrier: f64,
    kind: Barrier,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
    steps: usize,
    paths: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    check_inputs(s, k, t, sigma)?;
    check_paths(paths)?;
    if !(barrier > 0.0) || !barrier.is_finite() {
        return Err(GeomError::InvalidArgument("the barrier must be positive and finite"));
    }
    if steps == 0 || steps.saturating_mul(paths) > 50_000_000 {
        return Err(GeomError::InvalidArgument("monte_carlo_barrier: bad step count"));
    }
    let discount = (-r * t).exp();
    let mut path = Vec::with_capacity(steps);
    let mut values = Vec::with_capacity(paths);
    for _ in 0..paths {
        lognormal_path(s, t, r, sigma, q, steps, rng, &mut path);
        let touched = match kind {
            Barrier::UpAndOut | Barrier::UpAndIn => path.iter().any(|p| *p >= barrier),
            Barrier::DownAndOut | Barrier::DownAndIn => path.iter().any(|p| *p <= barrier),
        };
        let alive = match kind {
            Barrier::UpAndOut | Barrier::DownAndOut => !touched,
            Barrier::UpAndIn | Barrier::DownAndIn => touched,
        };
        let terminal = *path.last().expect("at least one step");
        let payoff = if call { (terminal - k).max(0.0) } else { (k - terminal).max(0.0) };
        values.push(if alive { discount * payoff } else { 0.0 });
    }
    Ok(summarise(&values))
}

/// A fixed-strike lookback option by Monte Carlo, returning
/// `(price, standard error)`.
///
/// A call pays on the running maximum and a put on the running minimum, so
/// the holder is credited with the best price the path ever reached. It is
/// therefore worth at least as much as the European option with the same
/// strike, always and path by path, and the tests use that as an ordering
/// rather than a number.
///
/// Discrete monitoring cuts the price for the same reason it raises a
/// knock-out's: the sampled extremum is closer to the terminal value than
/// the continuous one.
///
/// # Errors
/// As [`monte_carlo_asian`].
pub fn monte_carlo_lookback(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
    steps: usize,
    paths: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    check_inputs(s, k, t, sigma)?;
    check_paths(paths)?;
    if steps == 0 || steps.saturating_mul(paths) > 50_000_000 {
        return Err(GeomError::InvalidArgument("monte_carlo_lookback: bad step count"));
    }
    let discount = (-r * t).exp();
    let mut path = Vec::with_capacity(steps);
    let mut values = Vec::with_capacity(paths);
    for _ in 0..paths {
        lognormal_path(s, t, r, sigma, q, steps, rng, &mut path);
        let payoff = if call {
            (path.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)) - k).max(0.0)
        } else {
            (k - path.iter().fold(f64::INFINITY, |a, b| a.min(*b))).max(0.0)
        };
        values.push(discount * payoff);
    }
    Ok(summarise(&values))
}

/// The Longstaff-Schwartz price of an American option by least-squares
/// Monte Carlo.
///
/// Working backwards from expiry, the continuation value at each exercise
/// date is regressed on a quadratic in the current price, using only the
/// paths that are in the money -- and the *fitted* value, not the
/// realised one, decides whether to exercise. Using the realised future
/// payoff to make the decision would be looking ahead, and would produce a
/// price above the true one.
///
/// The estimate is biased low in principle, because the exercise rule
/// comes from a finite regression and any suboptimal rule undervalues the
/// option. In practice with a low-order basis it can also come out high
/// on the same sample the rule was fitted on, which is why the tests
/// compare it against a binomial tree with a tolerance rather than
/// asserting a direction.
///
/// # Errors
/// As [`monte_carlo_asian`], and for fewer than two exercise dates.
pub fn longstaff_schwartz_american(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
    steps: usize,
    paths: usize,
    rng: &mut Rng,
) -> Result<f64, GeomError> {
    check_inputs(s, k, t, sigma)?;
    check_paths(paths)?;
    if steps < 2 || steps.saturating_mul(paths) > 50_000_000 {
        return Err(GeomError::InvalidArgument("longstaff_schwartz_american: bad step count"));
    }
    let dt = t / steps as f64;
    let discount = (-r * dt).exp();
    let payoff = |price: f64| if call { (price - k).max(0.0) } else { (k - price).max(0.0) };

    // Store every path so the recursion can walk backwards through them.
    let mut grid = vec![0.0f64; paths * steps];
    let mut path = Vec::with_capacity(steps);
    for index in 0..paths {
        lognormal_path(s, t, r, sigma, q, steps, rng, &mut path);
        grid[index * steps..(index + 1) * steps].copy_from_slice(&path);
    }
    let mut cash: Vec<f64> = (0..paths).map(|i| payoff(grid[i * steps + steps - 1])).collect();
    for step in (0..steps - 1).rev() {
        for value in cash.iter_mut() {
            *value *= discount;
        }
        // Regress the discounted continuation value on 1, S and S^2 over
        // the in-the-money paths only: elsewhere the decision is not in
        // question and the fit would waste its degrees of freedom.
        let live: Vec<usize> =
            (0..paths).filter(|i| payoff(grid[i * steps + step]) > 0.0).collect();
        if live.len() < 4 {
            continue;
        }
        let mut moments = [0.0f64; 5];
        let mut rhs = [0.0f64; 3];
        for index in &live {
            let x = grid[index * steps + step];
            let y = cash[*index];
            let powers = [1.0, x, x * x, x * x * x, x * x * x * x];
            for (slot, value) in moments.iter_mut().zip(powers.iter()) {
                *slot += value;
            }
            rhs[0] += y;
            rhs[1] += y * x;
            rhs[2] += y * x * x;
        }
        let matrix = [
            [moments[0], moments[1], moments[2]],
            [moments[1], moments[2], moments[3]],
            [moments[2], moments[3], moments[4]],
        ];
        let Some(beta) = solve3(&matrix, &rhs) else { continue };
        for index in &live {
            let x = grid[index * steps + step];
            let continuation = beta[0] + beta[1] * x + beta[2] * x * x;
            let immediate = payoff(x);
            if immediate > continuation {
                cash[*index] = immediate;
            }
        }
    }
    let mean = cash.iter().sum::<f64>() / paths as f64;
    Ok(discount * mean)
}

/// Gaussian elimination on a 3x3 system, or `None` if it is singular.
fn solve3(matrix: &[[f64; 3]; 3], rhs: &[f64; 3]) -> Option<[f64; 3]> {
    let mut a = [
        [matrix[0][0], matrix[0][1], matrix[0][2], rhs[0]],
        [matrix[1][0], matrix[1][1], matrix[1][2], rhs[1]],
        [matrix[2][0], matrix[2][1], matrix[2][2], rhs[2]],
    ];
    let scale = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for column in 0..3 {
        let pivot = (column..3).max_by(|i, j| {
            a[*i][column]
                .abs()
                .partial_cmp(&a[*j][column].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        a.swap(column, pivot);
        if a[column][column].abs() < 1e-12 * scale {
            return None;
        }
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = a[row][column] / a[column][column];
            for entry in column..4 {
                a[row][entry] -= factor * a[column][entry];
            }
        }
    }
    Some([a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]])
}

// ---------------------------------------------------------------------------
// Models that produce a smile
// ---------------------------------------------------------------------------

/// Merton's jump-diffusion price, as a Poisson-weighted sum of
/// Black-Scholes prices.
///
/// A jump arriving at rate `lambda` multiplies the price by a lognormal
/// factor with log-mean `jump_mean` and log-standard-deviation
/// `jump_vol`. Conditioning on the number of jumps makes each term
/// lognormal again, so the price is an exact infinite sum of Black-Scholes
/// prices with adjusted rate and volatility, truncated here once the
/// Poisson weights are exhausted.
///
/// The drift compensator `-lambda * (e^(jump_mean + jump_vol^2/2) - 1)`
/// is what keeps the discounted price a martingale: jumps add expected
/// return, and it must be taken back out of the diffusion or the model
/// prices an arbitrage.
///
/// Jumps are what generate a smile. A single lognormal cannot make
/// out-of-the-money options expensive relative to at-the-money ones; a
/// mixture over jump counts has fatter tails and does exactly that.
///
/// # Errors
/// Returns an error for bad option parameters, a negative jump intensity
/// or volatility, or a non-positive maturity.
pub fn merton_jump_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    lambda: f64,
    jump_mean: f64,
    jump_vol: f64,
    call: bool,
) -> Result<f64, GeomError> {
    check_inputs(s, k, t, sigma)?;
    if lambda < 0.0 || jump_vol < 0.0 || !lambda.is_finite() || !jump_vol.is_finite() {
        return Err(GeomError::InvalidArgument("merton_jump_price: bad jump parameters"));
    }
    if !jump_mean.is_finite() {
        return Err(GeomError::InvalidArgument("merton_jump_price: bad jump mean"));
    }
    if !(t > 0.0) {
        return Ok(intrinsic(s, k, t, r, q, call));
    }
    let expected_jump = (jump_mean + 0.5 * jump_vol * jump_vol).exp() - 1.0;
    let compensator = lambda * expected_jump;
    // The Poisson weights carry the *risk-neutral* intensity
    // `lambda (1 + k)`, not `lambda`. With the bare intensity the series
    // still converges and still looks like a price, but the discount
    // factors no longer sum to `e^(-rT)` and the call and put prices stop
    // satisfying put-call parity -- an arbitrage in a model that is
    // supposed to be free of them.
    let intensity = lambda * (1.0 + expected_jump) * t;
    let mut total = 0.0;
    let mut weight = (-intensity).exp();
    for count in 0..200usize {
        if count > 0 {
            weight *= intensity / count as f64;
        }
        if weight < 1e-16 && count > (intensity as usize + 10) {
            break;
        }
        let jumps = count as f64;
        let variance = sigma * sigma + jumps * jump_vol * jump_vol / t;
        let rate = r - compensator + jumps * (jump_mean + 0.5 * jump_vol * jump_vol) / t;
        total += weight * black_scholes(s, k, t, rate, variance.sqrt(), q, call)?;
    }
    Ok(total)
}

/// A Heston stochastic-volatility price by Monte Carlo, returning
/// `(price, standard error)`.
///
/// The variance follows `dv = kappa (theta - v) dt + xi sqrt(v) dW`, with
/// the variance's Brownian motion correlated with the price's at `rho`.
/// That correlation is the model's point: a negative `rho` makes the
/// volatility rise as the price falls, which produces the downward-sloping
/// implied volatility skew that equity markets actually show, and which no
/// symmetric model can.
///
/// The variance is simulated with a full-truncation Euler scheme -- the
/// variance is floored at zero wherever a step takes it negative. Exact
/// simulation of the variance process is possible but expensive, and
/// full truncation is the standard compromise; it biases the price
/// slightly, and the bias falls with the step count rather than the path
/// count, so refining paths alone will not remove it.
///
/// # Errors
/// Returns an error for bad option parameters, a negative initial or
/// long-run variance, a non-positive mean reversion or volatility of
/// volatility, a correlation outside `[-1, 1]`, or a step or path count
/// outside its budget.
pub fn heston_price_mc(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    call: bool,
    steps: usize,
    paths: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    check_inputs(s, k, t, 0.0)?;
    check_paths(paths)?;
    if v0 < 0.0 || theta < 0.0 || !(kappa > 0.0) || !(xi > 0.0) || !(-1.0..=1.0).contains(&rho) {
        return Err(GeomError::InvalidArgument("heston_price_mc: bad model parameters"));
    }
    if steps == 0 || steps.saturating_mul(paths) > 50_000_000 || !(t > 0.0) {
        return Err(GeomError::InvalidArgument("heston_price_mc: bad step count or maturity"));
    }
    let dt = t / steps as f64;
    let discount = (-r * t).exp();
    let correlate = (1.0 - rho * rho).max(0.0).sqrt();
    let mut values = Vec::with_capacity(paths);
    for _ in 0..paths {
        let mut price = s;
        let mut variance = v0;
        for _ in 0..steps {
            let z1 = rng.next_gaussian();
            let z2 = rho * z1 + correlate * rng.next_gaussian();
            let used = variance.max(0.0);
            let root = used.sqrt();
            price *= ((r - q - 0.5 * used) * dt + root * dt.sqrt() * z1).exp();
            variance += kappa * (theta - used) * dt + xi * root * dt.sqrt() * z2;
        }
        let payoff = if call { (price - k).max(0.0) } else { (k - price).max(0.0) };
        values.push(discount * payoff);
    }
    Ok(summarise(&values))
}

/// The raw SVI parameterisation of a volatility smile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Svi {
    /// The overall level of variance.
    pub a: f64,
    /// The wing spread; non-negative.
    pub b: f64,
    /// The skew, in `[-1, 1]`.
    pub rho: f64,
    /// The horizontal shift of the smile's minimum.
    pub m: f64,
    /// The curvature at the minimum; positive.
    pub sigma: f64,
}

/// Total implied variance under raw SVI:
/// `a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))`.
///
/// `k` is log-moneyness `ln(K/F)` and the result is *total* variance
/// `sigma_implied^2 * T`, not annualised variance. SVI is a shape, not a
/// model: it has no process behind it and makes no prediction, and its
/// value is that five parameters fit an observed smile closely and the
/// wings are linear in `k`, which is what Lee's moment formula requires of
/// any arbitrage-free smile.
///
/// # Errors
/// Returns an error for a negative `b`, a non-positive `sigma`, a `rho`
/// outside `[-1, 1]`, or a total variance that comes out negative -- which
/// is an arbitrage, not a small numerical matter.
pub fn volatility_smile_svi(params: &Svi, k: f64) -> Result<f64, GeomError> {
    let p = *params;
    if p.b < 0.0 || !(p.sigma > 0.0) || !(-1.0..=1.0).contains(&p.rho) || !k.is_finite() {
        return Err(GeomError::InvalidArgument("volatility_smile_svi: bad parameters"));
    }
    let shifted = k - p.m;
    let variance = p.a + p.b * (p.rho * shifted + (shifted * shifted + p.sigma * p.sigma).sqrt());
    if variance < 0.0 {
        return Err(GeomError::Degenerate("the SVI parameters imply a negative total variance"));
    }
    Ok(variance)
}

/// Fits raw SVI to observed total variances by Nelder-Mead on the sum of
/// squared errors.
///
/// The objective is not convex and the parameters trade off against each
/// other -- `b` and `sigma` in particular are nearly degenerate for a
/// shallow smile -- so the search is restarted from the best point found,
/// which is what rescues it from the flat valley a single pass stalls in.
/// A good fit here means the shape matches, not that the parameters are
/// identified.
///
/// # Errors
/// Returns an error for fewer than five points, mismatched lengths, or a
/// non-positive total variance among the targets.
pub fn svi_fit(log_moneyness: &[f64], total_variance: &[f64]) -> Result<Svi, GeomError> {
    if log_moneyness.len() < 5 || log_moneyness.len() != total_variance.len() {
        return Err(GeomError::InvalidArgument("svi_fit needs at least five matched points"));
    }
    if total_variance.iter().any(|v| !(*v > 0.0)) || log_moneyness.iter().any(|k| !k.is_finite()) {
        return Err(GeomError::InvalidArgument("svi_fit: bad observations"));
    }
    let smallest = total_variance.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    let objective = |p: &[f64]| -> f64 {
        let candidate =
            Svi { a: p[0], b: p[1].abs(), rho: p[2].clamp(-0.999, 0.999), m: p[3], sigma: p[4].abs().max(1e-6) };
        log_moneyness
            .iter()
            .zip(total_variance.iter())
            .map(|(k, target)| match volatility_smile_svi(&candidate, *k) {
                Ok(value) => (value - target).powi(2),
                Err(_) => 1e12,
            })
            .sum()
    };
    let mut best = vec![smallest * 0.5, 0.1, -0.3, 0.0, 0.1];
    for _ in 0..3 {
        best = nelder_mead(&objective, &best, 4000);
    }
    Ok(Svi {
        a: best[0],
        b: best[1].abs(),
        rho: best[2].clamp(-0.999, 0.999),
        m: best[3],
        sigma: best[4].abs().max(1e-6),
    })
}

/// A compact Nelder-Mead simplex search.
fn nelder_mead(objective: &dyn Fn(&[f64]) -> f64, start: &[f64], iterations: usize) -> Vec<f64> {
    let n = start.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(start.to_vec());
    for axis in 0..n {
        let mut point = start.to_vec();
        let step = if point[axis].abs() > 1e-8 { 0.1 * point[axis] } else { 0.05 };
        point[axis] += step;
        simplex.push(point);
    }
    let mut values: Vec<f64> = simplex.iter().map(|p| objective(p)).collect();
    for _ in 0..iterations {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap_or(std::cmp::Ordering::Equal));
        simplex = order.iter().map(|i| simplex[*i].clone()).collect();
        values = order.iter().map(|i| values[*i]).collect();
        let centroid: Vec<f64> =
            (0..n).map(|axis| simplex[..n].iter().map(|p| p[axis]).sum::<f64>() / n as f64).collect();
        let reflected: Vec<f64> =
            (0..n).map(|axis| centroid[axis] + (centroid[axis] - simplex[n][axis])).collect();
        let reflected_value = objective(&reflected);
        if reflected_value < values[0] {
            let expanded: Vec<f64> = (0..n)
                .map(|axis| centroid[axis] + 2.0 * (centroid[axis] - simplex[n][axis]))
                .collect();
            let expanded_value = objective(&expanded);
            if expanded_value < reflected_value {
                simplex[n] = expanded;
                values[n] = expanded_value;
            } else {
                simplex[n] = reflected;
                values[n] = reflected_value;
            }
        } else if reflected_value < values[n - 1] {
            simplex[n] = reflected;
            values[n] = reflected_value;
        } else {
            let contracted: Vec<f64> = (0..n)
                .map(|axis| centroid[axis] + 0.5 * (simplex[n][axis] - centroid[axis]))
                .collect();
            let contracted_value = objective(&contracted);
            if contracted_value < values[n] {
                simplex[n] = contracted;
                values[n] = contracted_value;
            } else {
                for index in 1..=n {
                    for axis in 0..n {
                        simplex[index][axis] =
                            simplex[0][axis] + 0.5 * (simplex[index][axis] - simplex[0][axis]);
                    }
                    values[index] = objective(&simplex[index]);
                }
            }
        }
    }
    let best = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(index, _)| index);
    simplex[best].clone()
}

// ---------------------------------------------------------------------------
// The PDE
// ---------------------------------------------------------------------------

/// The Black-Scholes PDE solved by Crank-Nicolson on a log-price grid.
///
/// Solves `dV/dt + (r - q - sigma^2/2) dV/dx + (sigma^2/2) d2V/dx2 = rV`
/// backwards from the payoff, on `space` points spanning six standard
/// deviations either side of the log spot, with Dirichlet boundaries set
/// to the discounted no-arbitrage values. Set `american` to apply the
/// early-exercise constraint after each step, which makes the scheme a
/// projected one and costs its second-order accuracy in time near the
/// exercise boundary.
///
/// Crank-Nicolson is used rather than a fully implicit scheme because it
/// is second order in time as well as space. The price of that is that it
/// is only *A*-stable and not *L*-stable: it damps high-frequency error
/// slowly, so the kink in the payoff at the strike rings for several steps
/// rather than being smoothed away, and the Greeks near the strike are
/// visibly noisier than the price. Starting with a few fully implicit
/// steps -- Rannacher smoothing -- is the standard remedy and is what the
/// first two steps here do.
///
/// # Errors
/// Returns an error for bad option parameters, fewer than eleven space
/// points, no time steps, more than ten million grid cells, or a
/// tridiagonal system that will not solve.
pub fn bs_pde_crank_nicolson(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    call: bool,
    american: bool,
    space: usize,
    time_steps: usize,
) -> Result<f64, GeomError> {
    check_inputs(s, k, t, sigma)?;
    if space < 11 || time_steps == 0 || space.saturating_mul(time_steps) > 10_000_000 {
        return Err(GeomError::InvalidArgument("bs_pde_crank_nicolson: bad grid"));
    }
    if t == 0.0 || sigma == 0.0 {
        return Ok(intrinsic(s, k, t, r, q, call));
    }
    let width = 6.0 * sigma * t.sqrt();
    let centre = s.ln();
    let (low, high) = (centre - width, centre + width);
    let dx = (high - low) / (space - 1) as f64;
    let dt = t / time_steps as f64;
    let drift = r - q - 0.5 * sigma * sigma;
    let diffusion = 0.5 * sigma * sigma;
    let payoff = |x: f64| {
        let price = x.exp();
        if call {
            (price - k).max(0.0)
        } else {
            (k - price).max(0.0)
        }
    };
    let x_of = |index: usize| low + index as f64 * dx;
    let mut values: Vec<f64> = (0..space).map(|index| payoff(x_of(index))).collect();

    // Operator coefficients: L V = alpha V_{i-1} + beta V_i + gamma V_{i+1}.
    let alpha = diffusion / (dx * dx) - drift / (2.0 * dx);
    let beta = -2.0 * diffusion / (dx * dx) - r;
    let gamma = diffusion / (dx * dx) + drift / (2.0 * dx);
    let interior = space - 2;

    for step in 0..time_steps {
        // Rannacher smoothing: the first two steps are fully implicit,
        // which damps the payoff's kink that Crank-Nicolson would ring on.
        let weight = if step < 2 { 1.0 } else { 0.5 };
        let elapsed = (step + 1) as f64 * dt;
        let boundary = |x: f64| {
            let price = x.exp();
            if call {
                (price * (-q * elapsed).exp() - k * (-r * elapsed).exp()).max(0.0)
            } else {
                (k * (-r * elapsed).exp() - price * (-q * elapsed).exp()).max(0.0)
            }
        };
        let mut sub = vec![0.0; interior.saturating_sub(1)];
        let mut diag = vec![0.0; interior];
        let mut sup = vec![0.0; interior.saturating_sub(1)];
        let mut rhs = vec![0.0; interior];
        for row in 0..interior {
            let index = row + 1;
            diag[row] = 1.0 - weight * dt * beta;
            if row > 0 {
                sub[row - 1] = -weight * dt * alpha;
            }
            if row + 1 < interior {
                sup[row] = -weight * dt * gamma;
            }
            let explicit = (1.0 - weight) * dt
                * (alpha * values[index - 1] + beta * values[index] + gamma * values[index + 1]);
            rhs[row] = values[index] + explicit;
        }
        let low_boundary = boundary(x_of(0));
        let high_boundary = boundary(x_of(space - 1));
        rhs[0] += weight * dt * alpha * low_boundary;
        rhs[interior - 1] += weight * dt * gamma * high_boundary;

        let solved = crate::linalg::thomas_solve(&sub, &diag, &sup, &rhs)
            .map_err(|_| GeomError::Degenerate("the Crank-Nicolson system is singular"))?;
        values[0] = low_boundary;
        values[space - 1] = high_boundary;
        for (row, value) in solved.into_iter().enumerate() {
            values[row + 1] = if american { value.max(payoff(x_of(row + 1))) } else { value };
        }
    }

    // Interpolate at the spot, which sits at the grid's centre.
    let position = (centre - low) / dx;
    let left = (position.floor() as usize).min(space - 2);
    let fraction = position - left as f64;
    Ok(values[left] * (1.0 - fraction) + values[left + 1] * fraction)
}

// ---------------------------------------------------------------------------
// Hedging
// ---------------------------------------------------------------------------

/// Simulates delta hedging a short European option, returning
/// `(mean profit and loss, standard deviation)`.
///
/// The option is sold at its Black-Scholes price and the position is
/// rehedged `rebalances` times at the model delta; the P&L is what remains
/// at expiry after the payoff is settled.
///
/// The mean is near zero because the option was sold at its fair price,
/// but the *standard deviation* is the point: it falls like
/// `1/sqrt(rebalances)`, so cutting the residual risk in half costs four
/// times as many trades. That trade-off, not the mean, is what makes
/// continuous hedging a limit rather than a procedure -- with any
/// transaction cost at all, the total cost grows as `sqrt(rebalances)`
/// while the risk falls as `1/sqrt(rebalances)`, and an optimum exists at
/// a finite frequency.
///
/// A hedge run at a volatility different from the one the path was
/// generated with does not have a zero mean; the difference is the
/// volatility arbitrage, and it is what the tests check rather than the
/// noise.
///
/// # Errors
/// Returns an error for bad option parameters, no rebalances, a
/// non-positive maturity, or a path count outside `[2, 2e7]`.
pub fn delta_hedging_sim(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    hedge_vol: f64,
    realised_vol: f64,
    q: f64,
    call: bool,
    rebalances: usize,
    paths: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    check_inputs(s, k, t, hedge_vol)?;
    check_inputs(s, k, t, realised_vol)?;
    check_paths(paths)?;
    if rebalances == 0 || !(t > 0.0) || rebalances.saturating_mul(paths) > 50_000_000 {
        return Err(GeomError::InvalidArgument("delta_hedging_sim: bad rebalance count"));
    }
    if !(hedge_vol > 0.0) {
        return Err(GeomError::InvalidArgument("the hedge volatility must be positive"));
    }
    let dt = t / rebalances as f64;
    let premium = black_scholes(s, k, t, r, hedge_vol, q, call)?;
    let mut results = Vec::with_capacity(paths);
    for _ in 0..paths {
        let mut price = s;
        let mut remaining = t;
        let mut shares = bs_greeks(s, k, t, r, hedge_vol, q, call)?.delta;
        // Sold the option, bought `shares`; the rest sits in cash.
        let mut cash = premium - shares * s;
        for _ in 0..rebalances {
            cash *= (r * dt).exp();
            cash += shares * price * (q * dt).exp() - shares * price;
            price *= ((r - q - 0.5 * realised_vol * realised_vol) * dt
                + realised_vol * dt.sqrt() * rng.next_gaussian())
            .exp();
            remaining -= dt;
            let target = if remaining > 1e-10 {
                bs_greeks(price, k, remaining, r, hedge_vol, q, call)?.delta
            } else if call {
                f64::from(price > k)
            } else {
                -f64::from(price < k)
            };
            cash -= (target - shares) * price;
            shares = target;
        }
        let settled = if call { (price - k).max(0.0) } else { (k - price).max(0.0) };
        results.push(cash + shares * price - settled);
    }
    let count = paths as f64;
    let mean = results.iter().sum::<f64>() / count;
    let variance = results.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    Ok((mean, variance.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The textbook case: at the money, one year, 5% rates, 20% vol.
    const CASE: (f64, f64, f64, f64, f64, f64) = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0);

    #[test]
    fn the_closed_form_reproduces_the_textbook_number_and_its_parity() {
        let (s, k, t, r, sigma, q) = CASE;
        let call = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let put = black_scholes(s, k, t, r, sigma, q, false).unwrap();
        assert!((call - 10.450_583_572_185_565).abs() < 1e-12, "the call came out at {call}");
        assert!((put - 5.573_526_022_256_971).abs() < 1e-12, "the put came out at {put}");
        // Parity is an identity between the two, not an approximation.
        assert!(put_call_parity_check(call, put, s, k, t, r, q).abs() < 1e-13);
    }

    #[test]
    fn parity_holds_across_every_strike_maturity_and_dividend_yield() {
        // The identity follows from the payoffs, so no parameter can break
        // it. A pricing routine that got d1 and d2 subtly wrong would.
        for k in [50.0f64, 90.0, 100.0, 130.0, 400.0] {
            for t in [0.01f64, 0.5, 2.0, 30.0] {
                for q in [0.0f64, 0.03, 0.12] {
                    for r in [-0.01f64, 0.0, 0.05, 0.2] {
                        for sigma in [0.05f64, 0.3, 1.2] {
                            let call = black_scholes(100.0, k, t, r, sigma, q, true).unwrap();
                            let put = black_scholes(100.0, k, t, r, sigma, q, false).unwrap();
                            let residue =
                                put_call_parity_check(call, put, 100.0, k, t, r, q).abs();
                            assert!(
                                residue < 1e-10 * call.max(put).max(1.0),
                                "K={k} T={t} q={q} r={r} vol={sigma} left {residue}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_price_stays_inside_the_bounds_arbitrage_would_close() {
        // Below the discounted intrinsic value or above the underlying and
        // the option is free money. These are model-free bounds and hold
        // for every parameter set.
        for k in [60.0f64, 100.0, 150.0] {
            for t in [0.1f64, 1.0, 5.0] {
                for sigma in [0.01f64, 0.2, 0.9] {
                    let (s, r, q) = (100.0, 0.04, 0.02);
                    let call = black_scholes(s, k, t, r, sigma, q, true).unwrap();
                    let put = black_scholes(s, k, t, r, sigma, q, false).unwrap();
                    let forward = s * (-q * t).exp();
                    let strike = k * (-r * t).exp();
                    assert!(call >= (forward - strike).max(0.0) - 1e-12, "call under its floor");
                    assert!(call <= forward + 1e-12, "call above the share itself");
                    assert!(put >= (strike - forward).max(0.0) - 1e-12, "put under its floor");
                    assert!(put <= strike + 1e-12, "put above the discounted strike");
                }
            }
        }
    }

    #[test]
    fn a_price_rises_with_volatility_and_with_time_but_only_one_way_in_the_spot() {
        let (s, k, t, r, _, q) = CASE;
        let mut previous = 0.0;
        for sigma in [0.01f64, 0.05, 0.1, 0.3, 0.6, 1.5, 4.0] {
            let call = black_scholes(s, k, t, r, sigma, q, true).unwrap();
            assert!(call > previous, "the call fell as volatility rose to {sigma}");
            previous = call;
            // Both sides rise: vega has the same sign for a call and a put.
            let put = black_scholes(s, k, t, r, sigma, q, false).unwrap();
            assert!(put > 0.0);
        }
        let mut previous = 0.0;
        for spot in [50.0f64, 80.0, 100.0, 130.0, 200.0] {
            let call = black_scholes(spot, k, t, r, 0.2, q, true).unwrap();
            let put = black_scholes(spot, k, t, r, 0.2, q, false).unwrap();
            assert!(call > previous);
            previous = call;
            assert!(put < black_scholes(spot - 1.0, k, t, r, 0.2, q, false).unwrap());
        }
    }

    #[test]
    fn zero_volatility_or_zero_time_gives_the_discounted_intrinsic_value() {
        // Both limits are handled directly, since the formula divides by
        // sigma sqrt(T).
        for (t, sigma) in [(1.0f64, 0.0f64), (0.0, 0.2), (0.0, 0.0)] {
            let call = black_scholes(100.0, 90.0, t, 0.05, sigma, 0.01, true).unwrap();
            let expected = (100.0 * (-0.01 * t).exp() - 90.0 * (-0.05 * t).exp()).max(0.0);
            assert!((call - expected).abs() < 1e-12, "t={t} vol={sigma} gave {call}");
            let put = black_scholes(100.0, 90.0, t, 0.05, sigma, 0.01, false).unwrap();
            let expected_put = (90.0 * (-0.05 * t).exp() - 100.0 * (-0.01 * t).exp()).max(0.0);
            assert!((put - expected_put).abs() < 1e-12);
        }
        // And the Greeks are refused there rather than returning infinity.
        assert!(bs_greeks(100.0, 100.0, 0.0, 0.05, 0.2, 0.0, true).is_err());
        assert!(bs_greeks(100.0, 100.0, 1.0, 0.05, 0.0, 0.0, true).is_err());
    }

    #[test]
    fn the_greeks_are_the_derivatives_they_claim_to_be() {
        // Each Greek is checked against a central difference of the price
        // it differentiates. This is the test that catches a sign or a
        // missing carry term, which no self-consistency check would.
        for (s, k, t, sigma, q) in
            [(100.0, 100.0, 1.0, 0.2, 0.0), (80.0, 100.0, 0.5, 0.35, 0.03), (130.0, 100.0, 2.0, 0.15, 0.06)]
        {
            let r = 0.04;
            for call in [true, false] {
                let g = bs_greeks(s, k, t, r, sigma, q, call).unwrap();
                let price = |s: f64, t: f64, sigma: f64, r: f64| {
                    black_scholes(s, k, t, r, sigma, q, call).unwrap()
                };
                let h = 1e-4;
                let delta = (price(s + h, t, sigma, r) - price(s - h, t, sigma, r)) / (2.0 * h);
                assert!((g.delta - delta).abs() < 1e-6, "delta {} against {delta}", g.delta);
                let gamma = (price(s + h, t, sigma, r) - 2.0 * price(s, t, sigma, r)
                    + price(s - h, t, sigma, r))
                    / (h * h);
                assert!((g.gamma - gamma).abs() < 1e-4, "gamma {} against {gamma}", g.gamma);
                let vega = (price(s, t, sigma + h, r) - price(s, t, sigma - h, r)) / (2.0 * h);
                assert!((g.vega - vega).abs() < 1e-5, "vega {} against {vega}", g.vega);
                let rho = (price(s, t, sigma, r + h) - price(s, t, sigma, r - h)) / (2.0 * h);
                assert!((g.rho - rho).abs() < 1e-5, "rho {} against {rho}", g.rho);
                // Theta is minus the derivative in maturity: less time
                // left is what decay means.
                let theta = -(price(s, t + h, sigma, r) - price(s, t - h, sigma, r)) / (2.0 * h);
                assert!((g.theta - theta).abs() < 1e-5, "theta {} against {theta}", g.theta);
            }
        }
    }

    #[test]
    fn gamma_and_vega_do_not_know_whether_the_option_is_a_call() {
        // A call minus a put is a forward, which is linear in the spot and
        // has no volatility exposure, so the second derivative and the
        // volatility derivative must agree exactly.
        for (s, k, t, sigma, q) in
            [(100.0, 100.0, 1.0, 0.2, 0.0), (70.0, 120.0, 0.25, 0.5, 0.04), (150.0, 100.0, 3.0, 0.1, 0.02)]
        {
            let call = bs_greeks(s, k, t, 0.05, sigma, q, true).unwrap();
            let put = bs_greeks(s, k, t, 0.05, sigma, q, false).unwrap();
            assert!((call.gamma - put.gamma).abs() < 1e-15);
            assert!((call.vega - put.vega).abs() < 1e-13);
            // And the deltas differ by exactly the forward's, e^(-qT).
            assert!((call.delta - put.delta - (-q * t).exp()).abs() < 1e-13);
            assert!((0.0..=(-q * t).exp() + 1e-15).contains(&call.delta));
            assert!(put.delta <= 0.0);
            assert!(call.gamma > 0.0 && call.vega > 0.0);
        }
    }

    #[test]
    fn implied_volatility_inverts_the_formula_it_was_given() {
        // Wherever vega is meaningful the inversion is exact to a part in
        // a hundred million, for calls and puts alike.
        for k in [70.0f64, 100.0, 140.0] {
            for t in [0.05f64, 1.0, 4.0] {
                for sigma in [0.05f64, 0.2, 0.8, 2.0] {
                    for call in [true, false] {
                        let price = black_scholes(100.0, k, t, 0.04, sigma, 0.01, call).unwrap();
                        let vega = bs_greeks(100.0, k, t, 0.04, sigma, 0.01, call).unwrap().vega;
                        let recovered =
                            implied_volatility(price, 100.0, k, t, 0.04, 0.01, call).unwrap();
                        // The same threshold the solver documents.
                        if vega < 1e-8 * price.max(1.0) {
                            // The price carries no information about
                            // volatility here, and the solver says so.
                            assert_eq!(recovered, None, "K={k} T={t} vol={sigma} answered anyway");
                            continue;
                        }
                        let found = recovered.unwrap_or_else(|| {
                            panic!("no volatility found for K={k} T={t} vol={sigma}, vega {vega}")
                        });
                        assert!(
                            (found - sigma).abs() < 1e-8,
                            "K={k} T={t}: recovered {found} not {sigma}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_price_that_does_not_determine_a_volatility_gets_no_answer() {
        // A call struck at 70 with the share at 100 and eighteen days to
        // run is worth its intrinsic value whatever the volatility: vega
        // is about 1e-13, and 5% and 20% give the same price to the last
        // bit of a double. Reporting a number there would be reporting
        // rounding noise, so nothing is reported.
        let deep = black_scholes(100.0, 70.0, 0.05, 0.04, 0.05, 0.01, true).unwrap();
        let same = black_scholes(100.0, 70.0, 0.05, 0.04, 0.2, 0.01, true).unwrap();
        assert_eq!(deep, same, "the two volatilities were distinguishable after all");
        assert_eq!(implied_volatility(deep, 100.0, 70.0, 0.05, 0.04, 0.01, true).unwrap(), None);

        // Far out of the money is the same problem from the other side.
        let worthless = black_scholes(100.0, 140.0, 0.05, 0.04, 0.05, 0.01, true).unwrap();
        assert!(worthless < 1e-100, "the option was worth {worthless}");
        assert_eq!(
            implied_volatility(worthless, 100.0, 140.0, 0.05, 0.04, 0.01, true).unwrap(),
            None
        );

        // Give the same option four years instead and the price becomes
        // informative again.
        let readable = black_scholes(100.0, 140.0, 4.0, 0.04, 0.05, 0.01, true).unwrap();
        let recovered =
            implied_volatility(readable, 100.0, 140.0, 4.0, 0.04, 0.01, true).unwrap().unwrap();
        assert!((recovered - 0.05).abs() < 1e-8, "recovered {recovered}");
    }

    #[test]
    fn a_price_outside_the_no_arbitrage_range_has_no_implied_volatility() {
        let (s, k, t, r, _, q) = CASE;
        // Below the floor: no volatility makes an option worth less than
        // exercising it.
        assert_eq!(implied_volatility(0.0, s, 50.0, t, r, q, true).unwrap(), None);
        // Above the underlying itself.
        assert_eq!(implied_volatility(200.0, s, k, t, r, q, true).unwrap(), None);
        assert_eq!(implied_volatility(500.0, s, k, t, r, q, false).unwrap(), None);
        // Exactly at the floor the implied volatility is zero, and so is
        // vega: the price has stopped depending on volatility, so there
        // is nothing to report rather than a spurious zero.
        let floor = (s - k * (-r * t).exp()).max(0.0);
        assert_eq!(implied_volatility(floor, s, k, t, r, q, true).unwrap(), None);
        // A hair above it and the answer is a small positive number.
        let just_above = implied_volatility(floor + 0.5, s, k, t, r, q, true).unwrap();
        assert!(just_above.is_some_and(|v| v > 0.0 && v < 0.1), "got {just_above:?}");
        assert!(implied_volatility(-1.0, s, k, t, r, q, true).is_err());
        assert!(implied_volatility(10.0, s, k, 0.0, r, q, true).is_err());
        assert!(implied_volatility(10.0, 0.0, k, t, r, q, true).is_err());
    }

    #[test]
    fn the_binomial_tree_converges_by_oscillating_and_the_trinomial_does_not() {
        // The binomial error alternates in sign as the strike moves
        // between adjacent terminal nodes, so more steps is not reliably
        // better: 101 steps here lands further from the answer than 100
        // and on the other side of it. The trinomial's extra branch keeps
        // a node on the strike and removes the effect.
        let (s, k, t, r, sigma, q) = CASE;
        let exact = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let at_100 = binomial_crr(s, k, t, r, sigma, q, 100, true, false).unwrap() - exact;
        let at_101 = binomial_crr(s, k, t, r, sigma, q, 101, true, false).unwrap() - exact;
        assert!(at_100 < 0.0 && at_101 > 0.0, "the errors were {at_100} and {at_101}");

        let tri_100 = trinomial(s, k, t, r, sigma, q, 100, true, false).unwrap() - exact;
        let tri_101 = trinomial(s, k, t, r, sigma, q, 101, true, false).unwrap() - exact;
        assert!(tri_100 < 0.0 && tri_101 < 0.0, "the trinomial changed sign");
        assert!((tri_100 - tri_101).abs() < 0.1 * tri_100.abs(), "the trinomial jumped");

        // Both converge at first order in the step count.
        for method in [0, 1] {
            let error = |steps: usize| {
                let value = if method == 0 {
                    binomial_crr(s, k, t, r, sigma, q, steps, true, false).unwrap()
                } else {
                    trinomial(s, k, t, r, sigma, q, steps, true, false).unwrap()
                };
                (value - exact).abs()
            };
            let ratio = error(250) / error(1000);
            assert!((3.0..5.5).contains(&ratio), "quadrupling the steps cut the error by {ratio}");
        }
    }

    #[test]
    fn an_american_call_on_a_share_that_pays_nothing_is_never_exercised_early() {
        // The classic result: exercising throws away the interest on the
        // strike and the remaining optionality, and buys nothing, so the
        // American call is worth exactly the European one. With a dividend
        // it is worth strictly more, because waiting now costs something.
        let (s, k, t, r, sigma, _) = CASE;
        let european = binomial_crr(s, k, t, r, sigma, 0.0, 2000, true, false).unwrap();
        let american = binomial_crr(s, k, t, r, sigma, 0.0, 2000, true, true).unwrap();
        assert!((american - european).abs() < 1e-12, "early exercise gained {}", american - european);

        let paid = binomial_crr(s, k, t, r, sigma, 0.08, 2000, true, true).unwrap();
        let held = binomial_crr(s, k, t, r, sigma, 0.08, 2000, true, false).unwrap();
        assert!(paid > held + 1e-4, "a dividend did not create an early-exercise premium");

        // An American put always carries a premium, dividend or not,
        // because exercising banks the strike and starts earning on it.
        let euro_put = binomial_crr(s, 110.0, t, r, sigma, 0.0, 2000, false, false).unwrap();
        let amer_put = binomial_crr(s, 110.0, t, r, sigma, 0.0, 2000, false, true).unwrap();
        assert!(amer_put > euro_put + 1.0, "the put premium was only {}", amer_put - euro_put);
        // And an American option is never worth less than exercising now.
        assert!(amer_put >= 110.0 - s - 1e-9);
    }

    #[test]
    fn the_lattices_refuse_a_step_too_coarse_for_the_drift() {
        let (s, k, t, _, sigma, q) = CASE;
        assert!(binomial_crr(s, k, t, 0.05, sigma, q, 0, true, false).is_err());
        assert!(binomial_crr(s, k, t, 0.05, sigma, q, 100_000, true, false).is_err());
        assert!(trinomial(s, k, t, 0.05, sigma, q, 0, true, false).is_err());
        // A huge rate with a single step puts the risk-neutral probability
        // outside [0, 1], which is an arbitrage in the tree rather than a
        // small numerical matter.
        assert!(binomial_crr(s, k, t, 5.0, 0.05, q, 1, true, false).is_err());
        assert!(trinomial(s, k, t, 5.0, 0.05, q, 1, true, false).is_err());
        assert!(binomial_crr(0.0, k, t, 0.05, sigma, q, 10, true, false).is_err());
        assert!(binomial_crr(s, k, -1.0, 0.05, sigma, q, 10, true, false).is_err());
        // Zero volatility still prices, as the discounted intrinsic.
        assert!(binomial_crr(s, k, t, 0.05, 0.0, q, 10, true, false).is_ok());
    }

    #[test]
    fn monte_carlo_agrees_with_the_closed_form_within_its_own_error_bar() {
        // The standard error is the estimator's own claim about how far it
        // might be. A price several errors from the exact answer is a
        // failure of the estimator, not bad luck.
        let (s, k, t, r, sigma, q) = CASE;
        let mut rng = Rng::new(0x0F1A_1001);
        for call in [true, false] {
            let exact = black_scholes(s, k, t, r, sigma, q, call).unwrap();
            for paths in [4_000usize, 40_000] {
                let (price, error) =
                    monte_carlo_european(s, k, t, r, sigma, q, call, paths, &mut rng).unwrap();
                assert!(error > 0.0);
                assert!(
                    (price - exact).abs() < 3.0 * error,
                    "{paths} paths gave {price} +- {error} against {exact}"
                );
            }
        }
    }

    #[test]
    fn the_monte_carlo_error_falls_as_the_square_root_of_the_path_count() {
        let (s, k, t, r, sigma, q) = CASE;
        let mut rng = Rng::new(0x0F1A_1002);
        let (_, coarse) = monte_carlo_european(s, k, t, r, sigma, q, true, 5_000, &mut rng).unwrap();
        let (_, fine) = monte_carlo_european(s, k, t, r, sigma, q, true, 80_000, &mut rng).unwrap();
        let ratio = coarse / fine;
        // Sixteen times the paths should be four times the accuracy.
        assert!((3.0..5.0).contains(&ratio), "the error fell by {ratio}");
    }

    #[test]
    fn the_control_variate_cannot_bias_the_estimate_and_does_shrink_it() {
        // Subtracting beta times a quantity of known mean leaves the
        // expectation alone whatever beta is. The check is that the
        // reduced estimator is both unbiased and much tighter than the
        // raw payoff standard error on the same sample.
        let (s, k, t, r, sigma, q) = CASE;
        let mut rng = Rng::new(0x0F1A_1003);
        let paths = 40_000;
        let (price, reduced) =
            monte_carlo_european(s, k, t, r, sigma, q, true, paths, &mut rng).unwrap();
        // The raw estimator's error, computed independently here.
        let mut rng = Rng::new(0x0F1A_1003);
        let discount = (-r * t).exp();
        let raw: Vec<f64> = (0..paths)
            .map(|_| {
                let z = rng.next_gaussian();
                discount * (terminal_price(s, t, r, sigma, q, z) - k).max(0.0)
            })
            .collect();
        let mean = raw.iter().sum::<f64>() / paths as f64;
        let variance = raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (paths as f64 - 1.0);
        let plain = (variance / paths as f64).sqrt();
        assert!(reduced < 0.5 * plain, "the reduction gave {reduced} against {plain}");
        let exact = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        assert!((price - exact).abs() < 3.0 * reduced);
    }

    #[test]
    fn averaging_makes_an_asian_option_cheaper_than_its_european_twin() {
        // The average of a lognormal path has lower variance than its
        // endpoint, and lower variance at the same forward is a lower
        // option price. This is an ordering, not a number, and it holds
        // however the path is sampled.
        let (s, k, t, r, sigma, q) = CASE;
        let european = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let mut rng = Rng::new(0x0F1A_1004);
        let (asian, error) =
            monte_carlo_asian(s, k, t, r, sigma, q, true, 50, 30_000, &mut rng).unwrap();
        assert!(asian > 0.0);
        assert!(asian + 3.0 * error < european, "the Asian at {asian} was not cheaper");
        // More monitoring dates means more averaging and a lower price.
        let (few, _) = monte_carlo_asian(s, k, t, r, sigma, q, true, 2, 30_000, &mut rng).unwrap();
        let (many, _) = monte_carlo_asian(s, k, t, r, sigma, q, true, 200, 30_000, &mut rng).unwrap();
        assert!(many < few, "averaging over more dates raised the price: {many} against {few}");
    }

    #[test]
    fn a_knock_in_and_a_knock_out_add_up_to_the_option_without_a_barrier() {
        // Every path pays into exactly one of them, so on the *same* paths
        // the two sum to the vanilla price exactly -- not to within a
        // standard error. Priced on different draws they would only agree
        // statistically, which is a much weaker statement.
        let (s, k, t, r, sigma, q) = CASE;
        let seed = 0x0F1A_1005;
        let price = |kind: Barrier, level: f64| {
            let mut rng = Rng::new(seed);
            monte_carlo_barrier(s, k, level, kind, t, r, sigma, q, true, 100, 20_000, &mut rng)
                .unwrap()
                .0
        };
        // A barrier this far away is never touched, so up-and-out is the
        // vanilla option on these paths.
        let vanilla = price(Barrier::UpAndOut, 1e9);
        let out = price(Barrier::UpAndOut, 130.0);
        let knocked_in = price(Barrier::UpAndIn, 130.0);
        assert!(
            (out + knocked_in - vanilla).abs() < 1e-9,
            "{out} + {knocked_in} against {vanilla}"
        );
        assert!(out > 0.0 && knocked_in > 0.0, "one side of the barrier never paid");
        // The same holds downward.
        let down_out = price(Barrier::DownAndOut, 80.0);
        let down_in = price(Barrier::DownAndIn, 80.0);
        assert!((down_out + down_in - vanilla).abs() < 1e-9);
        // A knock-out is worth less than the vanilla, always.
        assert!(out < vanilla && down_out < vanilla);
    }

    #[test]
    fn watching_the_barrier_more_often_kills_more_options() {
        // Discrete monitoring is a modelling choice, not a numerical
        // detail: a path can cross the barrier and come back between
        // observations, so a knock-out watched twelve times a year is
        // worth strictly more than one watched daily.
        let (s, k, t, r, sigma, q) = CASE;
        let price = |steps: usize| {
            let mut rng = Rng::new(0x0F1A_1006);
            monte_carlo_barrier(
                s, k, 120.0, Barrier::UpAndOut, t, r, sigma, q, true, steps, 20_000, &mut rng,
            )
            .unwrap()
            .0
        };
        let rarely = price(12);
        let often = price(250);
        assert!(often < rarely, "daily monitoring gave {often} against monthly's {rarely}");
    }

    #[test]
    fn a_lookback_pays_at_least_what_the_european_would() {
        // Path by path the running maximum is at least the terminal
        // price, so the payoff dominates and so must the price.
        let (s, k, t, r, sigma, q) = CASE;
        let european = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let mut rng = Rng::new(0x0F1A_1007);
        let (lookback, error) =
            monte_carlo_lookback(s, k, t, r, sigma, q, true, 100, 20_000, &mut rng).unwrap();
        assert!(lookback > european + 3.0 * error, "the lookback was only {lookback}");
        // Sampling the extremum more finely can only find a larger one.
        let coarse = {
            let mut rng = Rng::new(0x0F1A_1008);
            monte_carlo_lookback(s, k, t, r, sigma, q, true, 10, 20_000, &mut rng).unwrap().0
        };
        let fine = {
            let mut rng = Rng::new(0x0F1A_1008);
            monte_carlo_lookback(s, k, t, r, sigma, q, true, 200, 20_000, &mut rng).unwrap().0
        };
        assert!(fine > coarse, "finer monitoring gave {fine} against {coarse}");
    }

    #[test]
    fn least_squares_monte_carlo_finds_the_price_the_tree_does() {
        // Two entirely different methods for the same American put: a
        // backward recursion on a lattice, and a regression on simulated
        // paths. Agreement to under half a percent is a real check on
        // both.
        let (s, _, t, r, sigma, q) = CASE;
        let k = 110.0;
        let tree = binomial_crr(s, k, t, r, sigma, q, 2000, false, true).unwrap();
        let mut rng = Rng::new(0x0F1A_1009);
        let regressed =
            longstaff_schwartz_american(s, k, t, r, sigma, q, false, 50, 40_000, &mut rng).unwrap();
        assert!(
            (regressed - tree).abs() < 0.01 * tree,
            "the regression gave {regressed} against the tree's {tree}"
        );
        // And it is worth at least the European put, which is what the
        // early exercise right buys.
        let european = black_scholes(s, k, t, r, sigma, q, false).unwrap();
        assert!(regressed > european, "{regressed} against a European {european}");
        assert!(longstaff_schwartz_american(s, k, t, r, sigma, q, false, 1, 100, &mut rng).is_err());
    }

    #[test]
    fn merton_with_no_jumps_is_black_scholes_to_the_last_bit() {
        // The Poisson sum collapses to its zeroth term, which is exactly
        // the closed form. Nothing less than equality would do here: any
        // difference is an error in the compensator or the weights.
        let (s, k, t, r, sigma, q) = CASE;
        for strike in [70.0f64, 100.0, 130.0] {
            for call in [true, false] {
                let exact = black_scholes(s, strike, t, r, sigma, q, call).unwrap();
                let jumped =
                    merton_jump_price(s, strike, t, r, sigma, q, 0.0, 0.0, 0.0, call).unwrap();
                assert!((jumped - exact).abs() < 1e-13, "{jumped} against {exact}");
            }
        }
        let _ = k;
    }

    #[test]
    fn jumps_create_a_smile_that_a_single_volatility_cannot() {
        // With a negative mean jump the implied volatilities fall as the
        // strike rises: a skew. Black-Scholes prices every strike off one
        // number and produces a flat line, so any slope at all is the
        // jumps talking.
        let (s, _, t, r, sigma, q) = CASE;
        let implied = |strike: f64, jump_mean: f64| {
            let price =
                merton_jump_price(s, strike, t, r, sigma, q, 0.5, jump_mean, 0.15, true).unwrap();
            implied_volatility(price, s, strike, t, r, q, true).unwrap().expect("a readable price")
        };
        let strikes = [80.0f64, 90.0, 100.0, 110.0, 120.0];
        let down: Vec<f64> = strikes.iter().map(|k| implied(*k, -0.1)).collect();
        for pair in down.windows(2) {
            assert!(pair[1] < pair[0], "the skew was not downward: {down:?}");
        }
        // Every one of them exceeds the diffusion volatility: jumps add
        // variance whichever way they point.
        assert!(down.iter().all(|v| *v > sigma), "{down:?}");

        // A positive mean jump tilts it the other way.
        let up: Vec<f64> = strikes.iter().map(|k| implied(*k, 0.1)).collect();
        assert!(up.last().unwrap() > up.first().unwrap(), "the skew did not reverse: {up:?}");
    }

    #[test]
    fn heston_with_no_volatility_of_volatility_is_black_scholes_again() {
        // Set xi to nothing and start the variance at its long-run level:
        // the variance never moves, and the model degenerates to a
        // lognormal with that volatility.
        let (s, k, t, r, sigma, q) = CASE;
        let exact = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let mut rng = Rng::new(0x0F1A_100A);
        let (price, error) = heston_price_mc(
            s, k, t, r, q, sigma * sigma, 2.0, sigma * sigma, 1e-8, 0.0, true, 200, 20_000,
            &mut rng,
        )
        .unwrap();
        assert!((price - exact).abs() < 3.0 * error, "{price} +- {error} against {exact}");
    }

    #[test]
    fn a_negative_correlation_is_what_tilts_the_heston_smile() {
        // The volatility rising as the price falls is what makes puts
        // expensive relative to calls. With zero correlation the smile is
        // symmetric; with negative correlation the low strike costs more.
        let (s, _, t, r, _, q) = CASE;
        let price = |strike: f64, rho: f64, seed: u64| {
            let mut rng = Rng::new(seed);
            heston_price_mc(
                s, strike, t, r, q, 0.04, 2.0, 0.04, 0.5, rho, true, 100, 60_000, &mut rng,
            )
            .unwrap()
        };
        let seed = 0x0F1A_100B;
        let (low_flat, _) = price(85.0, 0.0, seed);
        let (high_flat, _) = price(115.0, 0.0, seed);
        let (low_skew, low_err) = price(85.0, -0.8, seed);
        let (high_skew, high_err) = price(115.0, -0.8, seed);
        // The correlation makes the downside dearer and the upside
        // cheaper, relative to the uncorrelated case.
        assert!(
            low_skew > low_flat + 2.0 * low_err,
            "the low strike went from {low_flat} to {low_skew}"
        );
        assert!(
            high_skew < high_flat - 2.0 * high_err,
            "the high strike went from {high_flat} to {high_skew}"
        );
    }

    #[test]
    fn the_crank_nicolson_grid_converges_at_second_order() {
        // Doubling both the space and time resolution must quarter the
        // error. A first-order boundary or a mis-set theta weight would
        // show up here as a ratio near two.
        let (s, k, t, r, sigma, q) = CASE;
        let exact = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let error = |space: usize, steps: usize| {
            (bs_pde_crank_nicolson(s, k, t, r, sigma, q, true, false, space, steps).unwrap()
                - exact)
                .abs()
        };
        let coarse = error(201, 100);
        let fine = error(401, 200);
        let finer = error(801, 400);
        assert!(coarse < 1e-2, "the coarse grid was off by {coarse}");
        assert!((3.0..5.0).contains(&(coarse / fine)), "first refinement gave {}", coarse / fine);
        assert!((3.0..5.0).contains(&(fine / finer)), "second refinement gave {}", fine / finer);
    }

    #[test]
    fn the_grid_prices_an_american_put_where_the_tree_does() {
        let (s, _, t, r, sigma, q) = CASE;
        let k = 110.0;
        let tree = binomial_crr(s, k, t, r, sigma, q, 4000, false, true).unwrap();
        let grid = bs_pde_crank_nicolson(s, k, t, r, sigma, q, false, true, 801, 400).unwrap();
        assert!((grid - tree).abs() < 0.01 * tree, "the grid gave {grid} against {tree}");
        // The constraint is applied, so the value never falls below
        // exercising now.
        assert!(grid >= k - s - 1e-9);
        assert!(bs_pde_crank_nicolson(s, k, t, r, sigma, q, true, false, 5, 10).is_err());
        assert!(bs_pde_crank_nicolson(s, k, t, r, sigma, q, true, false, 101, 0).is_err());
    }

    #[test]
    fn hedging_more_often_halves_the_risk_for_four_times_the_trades() {
        // The residual standard deviation falls like one over the square
        // root of the rebalance count. That is the whole speed-cost
        // trade-off of a discrete hedge, and it is a rate rather than a
        // level, so it is checkable without knowing the constant.
        let (s, k, t, r, sigma, q) = CASE;
        let mut rng = Rng::new(0x0F1A_100C);
        let mut spreads = Vec::new();
        for rebalances in [8usize, 32, 128] {
            let (mean, spread) =
                delta_hedging_sim(s, k, t, r, sigma, sigma, q, true, rebalances, 4_000, &mut rng)
                    .unwrap();
            // Sold at the fair price and hedged at the realised
            // volatility, so the expected profit is nothing.
            assert!(
                mean.abs() < 4.0 * spread / (4_000.0f64).sqrt() + 0.02,
                "{rebalances} rebalances made {mean} on average"
            );
            spreads.push(spread);
        }
        for pair in spreads.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!((1.6..2.4).contains(&ratio), "quadrupling the trades cut the risk by {ratio}");
        }
    }

    #[test]
    fn hedging_at_the_wrong_volatility_is_where_the_money_is_lost() {
        // Selling an option at 20% and finding the world moves at 30%
        // loses money on average, and the loss is not noise: it is the
        // gamma of the position integrated against the variance
        // difference. Selling at 30% into a 20% world makes it back.
        let (s, k, t, r, _, q) = CASE;
        let mut rng = Rng::new(0x0F1A_100D);
        let (sold_cheap, spread) =
            delta_hedging_sim(s, k, t, r, 0.2, 0.3, q, true, 200, 4_000, &mut rng).unwrap();
        let error = spread / (4_000.0f64).sqrt();
        assert!(sold_cheap < -4.0 * error, "underpricing volatility made {sold_cheap}");

        let (sold_dear, spread) =
            delta_hedging_sim(s, k, t, r, 0.3, 0.2, q, true, 200, 4_000, &mut rng).unwrap();
        let error = spread / (4_000.0f64).sqrt();
        assert!(sold_dear > 4.0 * error, "overpricing volatility made {sold_dear}");

        assert!(delta_hedging_sim(s, k, t, r, 0.0, 0.2, q, true, 10, 100, &mut rng).is_err());
        assert!(delta_hedging_sim(s, k, t, r, 0.2, 0.2, q, true, 0, 100, &mut rng).is_err());
    }

    #[test]
    fn the_svi_fit_recovers_the_smile_it_was_given() {
        let truth = Svi { a: 0.02, b: 0.1, rho: -0.4, m: 0.02, sigma: 0.1 };
        let strikes: Vec<f64> = (0..15).map(|i| -0.7 + 0.1 * i as f64).collect();
        let variances: Vec<f64> =
            strikes.iter().map(|k| volatility_smile_svi(&truth, *k).unwrap()).collect();
        let fitted = svi_fit(&strikes, &variances).unwrap();
        for k in &strikes {
            let want = volatility_smile_svi(&truth, *k).unwrap();
            let got = volatility_smile_svi(&fitted, *k).unwrap();
            assert!((got - want).abs() < 1e-8, "at k={k} the fit gave {got} not {want}");
        }
        // The shape is what is fitted, and it is what the parameters mean.
        assert!(fitted.b >= 0.0 && fitted.sigma > 0.0);
        assert!((-1.0..=1.0).contains(&fitted.rho));
    }

    #[test]
    fn the_svi_wings_are_linear_and_its_minimum_sits_where_m_says() {
        // Lee's moment formula requires total variance to grow at most
        // linearly in log-moneyness, and SVI is built so that it grows
        // exactly linearly far out, with slopes b(1 - rho) and b(1 + rho).
        let params = Svi { a: 0.02, b: 0.1, rho: -0.4, m: 0.05, sigma: 0.1 };
        let far = 400.0;
        let right = volatility_smile_svi(&params, far).unwrap();
        let further = volatility_smile_svi(&params, far + 1.0).unwrap();
        assert!(
            (further - right - params.b * (1.0 + params.rho)).abs() < 1e-6,
            "the right wing's slope was {}",
            further - right
        );
        let left = volatility_smile_svi(&params, -far).unwrap();
        let farther = volatility_smile_svi(&params, -far - 1.0).unwrap();
        assert!((farther - left - params.b * (1.0 - params.rho)).abs() < 1e-6);

        // The minimum is at m when there is no skew, and moves off it
        // when there is.
        let flat = Svi { rho: 0.0, ..params };
        let at_m = volatility_smile_svi(&flat, flat.m).unwrap();
        for offset in [-0.3f64, -0.05, 0.05, 0.3] {
            assert!(volatility_smile_svi(&flat, flat.m + offset).unwrap() > at_m);
        }
        // Total variance is never negative, and the parameters that would
        // make it so are refused rather than priced.
        let arbitrage = Svi { a: -1.0, ..params };
        assert!(volatility_smile_svi(&arbitrage, 0.0).is_err());
        assert!(volatility_smile_svi(&Svi { b: -0.1, ..params }, 0.0).is_err());
        assert!(volatility_smile_svi(&Svi { sigma: 0.0, ..params }, 0.0).is_err());
        assert!(volatility_smile_svi(&Svi { rho: 1.5, ..params }, 0.0).is_err());
        assert!(svi_fit(&[0.0, 0.1], &[0.02, 0.02]).is_err());
        assert!(svi_fit(&[0.0, 0.1, 0.2, 0.3, 0.4], &[0.02, 0.02, 0.02, 0.02, -1.0]).is_err());
    }

    #[test]
    fn the_simulations_refuse_what_they_cannot_simulate() {
        let (s, k, t, r, sigma, q) = CASE;
        let mut rng = Rng::new(1);
        assert!(monte_carlo_european(s, k, t, r, sigma, q, true, 1, &mut rng).is_err());
        assert!(monte_carlo_european(0.0, k, t, r, sigma, q, true, 100, &mut rng).is_err());
        assert!(monte_carlo_asian(s, k, t, r, sigma, q, true, 0, 100, &mut rng).is_err());
        assert!(
            monte_carlo_barrier(s, k, 0.0, Barrier::UpAndOut, t, r, sigma, q, true, 10, 100, &mut rng)
                .is_err()
        );
        assert!(monte_carlo_lookback(s, k, t, r, sigma, q, true, 0, 100, &mut rng).is_err());
        assert!(merton_jump_price(s, k, t, r, sigma, q, -1.0, 0.0, 0.1, true).is_err());
        assert!(merton_jump_price(s, k, t, r, sigma, q, 0.5, 0.0, -0.1, true).is_err());
        assert!(
            heston_price_mc(s, k, t, r, q, 0.04, 0.0, 0.04, 0.5, 0.0, true, 10, 100, &mut rng)
                .is_err()
        );
        assert!(
            heston_price_mc(s, k, t, r, q, 0.04, 2.0, 0.04, 0.5, 1.5, true, 10, 100, &mut rng)
                .is_err()
        );
        assert!(
            heston_price_mc(s, k, t, r, q, -0.01, 2.0, 0.04, 0.5, 0.0, true, 10, 100, &mut rng)
                .is_err()
        );
    }
}
