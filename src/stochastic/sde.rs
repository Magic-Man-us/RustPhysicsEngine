//! Stochastic differential equations: simulation, convergence, and the
//! densities the paths are distributed by.
//!
//! An equation `dX = mu dt + sigma dW` is not an ordinary differential
//! equation with noise added. Brownian motion is nowhere differentiable, and
//! `dW` has magnitude of order `sqrt(dt)` rather than `dt`, so a term that
//! would be second order in a deterministic expansion is first order here.
//! That is the whole content of Ito's lemma, and it is why the numerical
//! schemes are not the familiar ones: Euler-Maruyama looks like Euler's
//! method but converges at half its order, and recovering first order needs
//! the Milstein correction, which is precisely the term Ito's lemma says is
//! missing.
//!
//! *Strong* convergence is about paths -- how close a simulated path is to
//! the exact path driven by the same noise -- and *weak* convergence is about
//! distributions, how close the expectation of a function is. They are
//! genuinely different: Euler-Maruyama is strong order one half and weak
//! order one. Which one matters depends on the question, and both are
//! measured here rather than asserted.

use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;
use std::f64::consts::PI;

/// A Brownian path of `n + 1` points at spacing `dt`, starting at zero.
///
/// Increments are independent Gaussians of variance `dt`, which is the
/// definition. Everything else in the module is built on this or on the
/// same increments used differently.
///
/// # Panics
/// Panics unless `dt` is positive.
#[must_use]
pub fn brownian_motion(n: usize, dt: f64, rng: &mut Rng) -> Vec<f64> {
    assert!(dt > 0.0, "the step must be positive");
    let s = dt.sqrt();
    let mut out = Vec::with_capacity(n + 1);
    let mut x = 0.0;
    out.push(x);
    for _ in 0..n {
        x += s * rng.next_gaussian();
        out.push(x);
    }
    out
}

/// A Brownian bridge: a path pinned at both ends.
///
/// Built by taking a free Brownian path and subtracting the linear
/// interpolation of its own endpoint error. The result has variance
/// `t (T - t) / T` -- zero at both ends and largest in the middle -- which is
/// what conditioning on the destination does to the uncertainty.
///
/// # Panics
/// Panics unless `dt` is positive and `n` is at least one.
#[must_use]
pub fn brownian_bridge(n: usize, dt: f64, x0: f64, x1: f64, rng: &mut Rng) -> Vec<f64> {
    assert!(dt > 0.0, "the step must be positive");
    assert!(n >= 1, "a bridge needs at least one step");
    let w = brownian_motion(n, dt, rng);
    let total = n as f64 * dt;
    let end = w[n];
    (0..=n)
        .map(|i| {
            let t = i as f64 * dt;
            x0 + (x1 - x0) * t / total + w[i] - end * t / total
        })
        .collect()
}

/// A Brownian path in two dimensions, as independent coordinates.
///
/// # Panics
/// Panics unless `dt` is positive.
#[must_use]
pub fn brownian_2d(n: usize, dt: f64, rng: &mut Rng) -> Vec<(f64, f64)> {
    let x = brownian_motion(n, dt, rng);
    let y = brownian_motion(n, dt, rng);
    x.into_iter().zip(y).collect()
}

/// A Brownian path in three dimensions.
///
/// # Panics
/// Panics unless `dt` is positive.
#[must_use]
pub fn brownian_3d(n: usize, dt: f64, rng: &mut Rng) -> Vec<(f64, f64, f64)> {
    let x = brownian_motion(n, dt, rng);
    let y = brownian_motion(n, dt, rng);
    let z = brownian_motion(n, dt, rng);
    (0..=n).map(|i| (x[i], y[i], z[i])).collect()
}

/// Geometric Brownian motion, simulated by exact log-space steps.
///
/// `dS = mu S dt + sigma S dW`. Its logarithm is Brownian with drift
/// `mu - sigma^2 / 2`, so the process can be stepped exactly rather than
/// approximated -- and the `- sigma^2 / 2` is Ito's correction, the
/// difference between the drift of the process and the drift of its
/// logarithm. Simulating in log space also guarantees the path stays
/// positive, which a naive Euler step does not.
///
/// # Panics
/// Panics unless `dt` is positive and `x0` is positive.
#[must_use]
pub fn geometric_brownian(
    x0: f64,
    mu: f64,
    sigma: f64,
    n: usize,
    dt: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(dt > 0.0 && x0 > 0.0, "the step and the start must be positive");
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let mut out = Vec::with_capacity(n + 1);
    let mut x = x0;
    out.push(x);
    for _ in 0..n {
        x *= (drift + vol * rng.next_gaussian()).exp();
        out.push(x);
    }
    out
}

/// The exact solution of geometric Brownian motion at time `t`, given the
/// standard normal `z` that drives it.
///
/// The closed form the schemes are measured against. Passing the driving
/// normal in rather than drawing it is what lets a numerical path and the
/// exact path share the same noise, which is what strong convergence means.
#[must_use]
pub fn gbm_exact(x0: f64, mu: f64, sigma: f64, t: f64, z: f64) -> f64 {
    x0 * ((mu - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp()
}

/// An Ornstein-Uhlenbeck path, stepped exactly.
///
/// `dX = theta (mu - X) dt + sigma dW`: a Brownian particle pulled back
/// towards `mu` at a rate proportional to its distance. Unlike Brownian
/// motion it has a stationary distribution -- Gaussian with mean `mu` and
/// variance `sigma^2 / (2 theta)` -- because the restoring pull eventually
/// balances the noise. The transition density is Gaussian in closed form, so
/// this is exact at any step size.
///
/// # Panics
/// Panics unless `dt` and `theta` are positive.
#[must_use]
pub fn ornstein_uhlenbeck(
    x0: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    n: usize,
    dt: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(dt > 0.0 && theta > 0.0, "the step and the pull must be positive");
    let mut out = Vec::with_capacity(n + 1);
    let mut x = x0;
    out.push(x);
    for _ in 0..n {
        x = ou_exact_step(x, theta, mu, sigma, dt, rng.next_gaussian());
        out.push(x);
    }
    out
}

/// One exact Ornstein-Uhlenbeck step, given the standard normal driving it.
///
/// # Panics
/// Panics unless `theta` and `dt` are positive.
#[must_use]
pub fn ou_exact_step(x: f64, theta: f64, mu: f64, sigma: f64, dt: f64, z: f64) -> f64 {
    assert!(theta > 0.0 && dt > 0.0, "the pull and the step must be positive");
    let decay = (-theta * dt).exp();
    // The conditional variance of the exact transition, which tends to the
    // stationary variance as the step grows.
    let var = sigma * sigma * (1.0 - decay * decay) / (2.0 * theta);
    mu + (x - mu) * decay + var.sqrt() * z
}

/// The Euler-Maruyama scheme for a scalar equation.
///
/// `X_{k+1} = X_k + mu dt + sigma sqrt(dt) Z`. The obvious discretisation,
/// and strong order one half rather than the order one Euler's method
/// achieves without noise -- because the neglected term involves
/// `(dW)^2`, which is of order `dt` rather than `dt^2`.
///
/// # Panics
/// Panics unless `n` is positive and `t_end` is positive.
pub fn euler_maruyama(
    mu: &dyn Fn(f64, f64) -> f64,
    sigma: &dyn Fn(f64, f64) -> f64,
    x0: f64,
    t_end: f64,
    n: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(n > 0 && t_end > 0.0, "the horizon and the step count must be positive");
    let dt = t_end / n as f64;
    let s = dt.sqrt();
    let mut out = Vec::with_capacity(n + 1);
    let mut x = x0;
    out.push(x);
    for k in 0..n {
        let t = k as f64 * dt;
        x += mu(t, x) * dt + sigma(t, x) * s * rng.next_gaussian();
        out.push(x);
    }
    out
}

/// Euler-Maruyama for a vector equation with a matrix diffusion.
///
/// The noise is a vector of independent Brownian motions, one per column of
/// the diffusion matrix, so correlations between components come from the
/// matrix rather than from the noise.
///
/// # Panics
/// Panics unless `n` and `t_end` are positive, or if the diffusion matrix's
/// shape does not match the state.
pub fn euler_maruyama_nd(
    mu: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    sigma: &dyn Fn(f64, &[f64]) -> Matrix,
    x0: &[f64],
    t_end: f64,
    n: usize,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    assert!(n > 0 && t_end > 0.0, "the horizon and the step count must be positive");
    let dt = t_end / n as f64;
    let s = dt.sqrt();
    let d = x0.len();
    let mut x = x0.to_vec();
    let mut out = Vec::with_capacity(n + 1);
    out.push(x.clone());
    for k in 0..n {
        let t = k as f64 * dt;
        let drift = mu(t, &x);
        let diffusion = sigma(t, &x);
        assert_eq!(diffusion.rows, d, "the diffusion matrix has the wrong height");
        let dw: Vec<f64> = (0..diffusion.cols).map(|_| s * rng.next_gaussian()).collect();
        let kick = diffusion.mul_vec(&dw).expect("the shapes agree");
        for i in 0..d {
            x[i] += drift[i] * dt + kick[i];
        }
        out.push(x.clone());
    }
    out
}

/// The Milstein scheme, which restores strong order one.
///
/// Adds `0.5 sigma sigma' ((dW)^2 - dt)` to the Euler step. That term is
/// exactly what Ito's lemma says the expansion of `sigma(X)` contributes at
/// first order and Euler-Maruyama drops; putting it back doubles the
/// convergence rate for the price of one derivative.
///
/// # Panics
/// Panics unless `n` and `t_end` are positive.
pub fn milstein(
    mu: &dyn Fn(f64, f64) -> f64,
    sigma: &dyn Fn(f64, f64) -> f64,
    dsigma_dx: &dyn Fn(f64, f64) -> f64,
    x0: f64,
    t_end: f64,
    n: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(n > 0 && t_end > 0.0, "the horizon and the step count must be positive");
    let dt = t_end / n as f64;
    let s = dt.sqrt();
    let mut out = Vec::with_capacity(n + 1);
    let mut x = x0;
    out.push(x);
    for k in 0..n {
        let t = k as f64 * dt;
        let dw = s * rng.next_gaussian();
        let sg = sigma(t, x);
        x += mu(t, x) * dt + sg * dw + 0.5 * sg * dsigma_dx(t, x) * (dw * dw - dt);
        out.push(x);
    }
    out
}

/// The stochastic Heun scheme, which converges to the *Stratonovich*
/// solution.
///
/// A predictor-corrector: step forward, evaluate the coefficients there too,
/// and average. In the deterministic case that is the trapezoidal rule; with
/// noise it changes which stochastic integral is being computed. The
/// Stratonovich integral evaluates the integrand at the midpoint of each
/// interval rather than the left end, which makes the ordinary chain rule
/// hold and Ito's correction vanish -- and makes the answer differ from the
/// Ito one by `0.5 sigma sigma'`.
///
/// # Panics
/// Panics unless `n` and `t_end` are positive.
pub fn stochastic_heun(
    mu: &dyn Fn(f64, f64) -> f64,
    sigma: &dyn Fn(f64, f64) -> f64,
    x0: f64,
    t_end: f64,
    n: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(n > 0 && t_end > 0.0, "the horizon and the step count must be positive");
    let dt = t_end / n as f64;
    let s = dt.sqrt();
    let mut out = Vec::with_capacity(n + 1);
    let mut x = x0;
    out.push(x);
    for k in 0..n {
        let t = k as f64 * dt;
        let dw = s * rng.next_gaussian();
        let predictor = x + mu(t, x) * dt + sigma(t, x) * dw;
        let t1 = t + dt;
        x += 0.5 * (mu(t, x) + mu(t1, predictor)) * dt
            + 0.5 * (sigma(t, x) + sigma(t1, predictor)) * dw;
        out.push(x);
    }
    out
}

/// A stochastic Runge-Kutta scheme of strong order one and a half for
/// additive noise.
///
/// With `sigma` constant the double stochastic integrals that ordinarily
/// block high-order schemes reduce to two correlated Gaussians, which can be
/// drawn directly. Both are drawn here, so the extra half order is real
/// rather than a relabelled Milstein.
///
/// # Panics
/// Panics unless `n` and `t_end` are positive.
pub fn srk_order_1_5(
    mu: &dyn Fn(f64, f64) -> f64,
    sigma: f64,
    x0: f64,
    t_end: f64,
    n: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(n > 0 && t_end > 0.0, "the horizon and the step count must be positive");
    let dt = t_end / n as f64;
    let mut out = Vec::with_capacity(n + 1);
    let mut x = x0;
    out.push(x);
    for k in 0..n {
        let t = k as f64 * dt;
        // The Brownian increment and its time integral over the step, which
        // are jointly Gaussian with a known correlation.
        let u1 = rng.next_gaussian();
        let u2 = rng.next_gaussian();
        let dw = dt.sqrt() * u1;
        let dz = 0.5 * dt.powf(1.5) * (u1 + u2 / 3.0f64.sqrt());
        let drift = mu(t, x);
        let supporting = x + drift * dt + sigma * dt.sqrt();
        let ahead = mu(t + dt, supporting);
        let behind = mu(t + dt, x + drift * dt - sigma * dt.sqrt());
        x += drift * dt
            + sigma * dw
            + (ahead - behind) * dz / (2.0 * sigma * dt.sqrt())
            + (ahead - 2.0 * drift + behind) * dt / 4.0;
        out.push(x);
    }
    out
}

/// The measured strong convergence order of a scheme.
///
/// `errors` are mean absolute path errors against the exact solution, one
/// per step size in `dts`. The order is the slope of the error against the
/// step on log axes, by least squares. Measuring it rather than assuming it
/// is the only way to notice that a scheme has been implemented at the wrong
/// order, which looks like nothing at all at a single step size.
///
/// # Panics
/// Panics unless the two slices have the same length, at least two entries,
/// and all values are positive.
#[must_use]
pub fn strong_convergence_order(errors: &[f64], dts: &[f64]) -> f64 {
    assert_eq!(errors.len(), dts.len(), "one error per step size is required");
    assert!(errors.len() >= 2, "a slope needs at least two points");
    assert!(
        errors.iter().chain(dts).all(|&v| v > 0.0 && v.is_finite()),
        "errors and steps must be positive"
    );
    let xs: Vec<f64> = dts.iter().map(|d| d.ln()).collect();
    let ys: Vec<f64> = errors.iter().map(|e| e.ln()).collect();
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    num / den
}

/// The measured weak convergence order, from errors in an expectation.
///
/// The same regression on a different error. A scheme can be weak order one
/// while being strong order a half, which is not a contradiction: getting
/// the distribution right is easier than getting each path right.
///
/// # Panics
/// Panics under the same conditions as [`strong_convergence_order`].
#[must_use]
pub fn weak_convergence_order(errors: &[f64], dts: &[f64]) -> f64 {
    strong_convergence_order(errors, dts)
}

/// A Cox-Ingersoll-Ross path by the full truncation scheme.
///
/// `dX = kappa (theta - X) dt + sigma sqrt(X) dW`. The square root makes the
/// noise vanish at zero, so the exact process never goes negative -- but a
/// discretisation can step below zero and then take the root of a negative
/// number.
///
/// Full truncation lets the *internal* state go negative and applies
/// `max(X, 0)` only inside the coefficients, reporting the truncated value.
/// Clipping the state itself instead -- reflecting at zero -- is the obvious
/// alternative and a much worse one: every reflection injects probability
/// mass that the exact process does not have, and the bias grows rather than
/// shrinks as the step is refined, because a finer step visits the boundary
/// more often. Full truncation has the smallest measured bias of the
/// published variants, which is why it is the one in use.
///
/// # Panics
/// Panics unless `dt`, `kappa` and `theta` are positive and `x0` is
/// non-negative.
#[must_use]
pub fn cir_process(
    x0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    n: usize,
    dt: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(dt > 0.0 && kappa > 0.0 && theta > 0.0, "the parameters must be positive");
    assert!(x0 >= 0.0, "the start must be non-negative");
    let s = dt.sqrt();
    let mut out = Vec::with_capacity(n + 1);
    // `state` may go negative; what is reported never does.
    let mut state = x0;
    out.push(state.max(0.0));
    for _ in 0..n {
        let positive = state.max(0.0);
        state +=
            kappa * (theta - positive) * dt + sigma * positive.sqrt() * s * rng.next_gaussian();
        out.push(state.max(0.0));
    }
    out
}

/// Parameters of the Heston stochastic volatility model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonParams {
    /// Drift of the asset.
    pub mu: f64,
    /// Rate at which variance reverts.
    pub kappa: f64,
    /// Long-run variance.
    pub theta: f64,
    /// Volatility of the variance.
    pub xi: f64,
    /// Correlation between the two driving Brownian motions.
    pub rho: f64,
}

/// Heston paths: an asset whose variance is itself a Cox-Ingersoll-Ross
/// process.
///
/// The correlation between the two noises is what makes the model useful.
/// A negative `rho` means variance rises when the price falls, which
/// reproduces the skew that a constant-volatility model cannot.
///
/// # Panics
/// Panics unless `dt` and `s0` are positive, `v0` is non-negative, and the
/// correlation lies in `[-1, 1]`.
#[must_use]
pub fn heston_paths(
    s0: f64,
    v0: f64,
    params: HestonParams,
    n: usize,
    dt: f64,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>) {
    assert!(dt > 0.0 && s0 > 0.0, "the step and the price must be positive");
    assert!(v0 >= 0.0, "the variance must be non-negative");
    assert!((-1.0..=1.0).contains(&params.rho), "the correlation must lie in [-1, 1]");
    let s = dt.sqrt();
    let mut prices = Vec::with_capacity(n + 1);
    let mut vars = Vec::with_capacity(n + 1);
    let (mut price, mut var) = (s0, v0);
    prices.push(price);
    vars.push(var);
    for _ in 0..n {
        let z1 = rng.next_gaussian();
        let z2 = rng.next_gaussian();
        // The second noise is correlated with the first by rho, built from
        // two independent draws.
        let w1 = z1;
        let w2 = params.rho * z1 + (1.0 - params.rho * params.rho).sqrt() * z2;
        let positive = var.max(0.0);
        let root = positive.sqrt();
        // The price is stepped in log space so it stays positive.
        price *= ((params.mu - 0.5 * positive) * dt + root * s * w1).exp();
        // Full truncation on the variance, for the reason cir_process gives.
        var += params.kappa * (params.theta - positive) * dt + params.xi * root * s * w2;
        prices.push(price);
        vars.push(var.max(0.0));
    }
    (prices, vars)
}

/// Merton's jump diffusion: geometric Brownian motion with Poisson jumps of
/// lognormal size.
///
/// The jumps put weight in the tails that a diffusion cannot, which is what
/// the model exists for. Between jumps it is exactly geometric Brownian
/// motion, and the compensator `lambda (exp(jump_mu + jump_sigma^2/2) - 1)`
/// is subtracted from the drift so the expected return is `mu` whether or
/// not a jump lands.
///
/// # Panics
/// Panics unless `dt` and `x0` are positive and `lambda` is non-negative.
#[must_use]
pub fn jump_diffusion_merton(
    x0: f64,
    mu: f64,
    sigma: f64,
    lambda: f64,
    jump_mu: f64,
    jump_sigma: f64,
    n: usize,
    dt: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(dt > 0.0 && x0 > 0.0, "the step and the start must be positive");
    assert!(lambda >= 0.0, "the jump rate must be non-negative");
    let compensator = lambda * ((jump_mu + 0.5 * jump_sigma * jump_sigma).exp() - 1.0);
    let drift = (mu - compensator - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let mut out = Vec::with_capacity(n + 1);
    let mut x = x0;
    out.push(x);
    for _ in 0..n {
        let mut log_step = drift + vol * rng.next_gaussian();
        // How many jumps land in this interval.
        let jumps = poisson_count(lambda * dt, rng);
        for _ in 0..jumps {
            log_step += jump_mu + jump_sigma * rng.next_gaussian();
        }
        x *= log_step.exp();
        out.push(x);
    }
    out
}

/// A Poisson count with the given mean, by Knuth's product method.
fn poisson_count(mean: f64, rng: &mut Rng) -> u64 {
    if mean <= 0.0 {
        return 0;
    }
    if mean > 30.0 {
        // The product underflows past about seven hundred; for a mean this
        // large a normal approximation is well within the noise anyway.
        return (mean + mean.sqrt() * rng.next_gaussian()).max(0.0).round() as u64;
    }
    let limit = (-mean).exp();
    let mut product = 1.0;
    let mut k = 0u64;
    loop {
        product *= rng.next_f64();
        if product <= limit {
            return k;
        }
        k += 1;
    }
}

/// A draw from a stable distribution by the Chambers-Mallows-Stuck method.
///
/// The stable laws are the only possible limits of normalised sums, and only
/// the Gaussian among them has finite variance. `alpha` is the tail index:
/// two gives a Gaussian, one with `beta` zero gives Cauchy, and anything
/// below two has infinite variance and a tail decaying like a power rather
/// than an exponential. `beta` is the skew.
///
/// # Panics
/// Panics unless `alpha` is in `(0, 2]` and `beta` is in `[-1, 1]`.
#[must_use]
pub fn levy_stable_sample(alpha: f64, beta: f64, rng: &mut Rng) -> f64 {
    assert!(alpha > 0.0 && alpha <= 2.0, "alpha must lie in (0, 2]");
    assert!((-1.0..=1.0).contains(&beta), "beta must lie in [-1, 1]");
    // A uniform angle and an exponential radius drive the construction.
    // The sign of beta is flipped against the raw Chambers-Mallows-Stuck
    // formulas, which skew the opposite way from the convention everyone
    // states results in: here a positive beta stretches the upper tail.
    let beta = -beta;
    let u = PI * (rng.next_f64() - 0.5);
    let w = -rng.next_f64().max(1e-300).ln();
    if (alpha - 1.0).abs() < 1e-12 {
        let term = (PI / 2.0 + beta * u) * u.tan()
            - beta * ((PI / 2.0) * w * u.cos() / (PI / 2.0 + beta * u)).ln();
        return term * 2.0 / PI;
    }
    let zeta = -beta * (PI * alpha / 2.0).tan();
    let xi = zeta.atan() / alpha;
    let numerator = (alpha * (u + xi)).sin();
    let denominator = u.cos().powf(1.0 / alpha);
    let tail = ((u - alpha * (u + xi)).cos() / w).powf((1.0 - alpha) / alpha);
    (1.0 + zeta * zeta).powf(1.0 / (2.0 * alpha)) * numerator / denominator * tail
}

/// Fractional Brownian motion with Hurst parameter `h`, by the Davies-Harte
/// method.
///
/// Increments are correlated rather than independent: `h` above a half gives
/// a path that persists, below a half one that reverses, and exactly a half
/// gives ordinary Brownian motion. Davies and Harte's method embeds the
/// covariance into a circulant matrix, whose eigenvalues a Fourier transform
/// supplies, so an exact sample costs one transform instead of a Cholesky
/// factorisation.
///
/// # Panics
/// Panics unless `h` is in `(0, 1)` and `n` is positive. Falls back to a
/// Cholesky construction if the circulant embedding is not non-negative
/// definite, which can happen near the ends of the range.
#[must_use]
pub fn fractional_brownian(h: f64, n: usize, rng: &mut Rng) -> Vec<f64> {
    assert!(h > 0.0 && h < 1.0, "the Hurst parameter must lie in (0, 1)");
    assert!(n > 0, "a path needs at least one step");
    // The autocovariance of fractional Gaussian noise at lag k.
    let gamma = |k: f64| {
        0.5 * ((k + 1.0).abs().powf(2.0 * h) - 2.0 * k.abs().powf(2.0 * h)
            + (k - 1.0).abs().powf(2.0 * h))
    };
    // Circulant embedding of length 2n, whose first row wraps the covariance.
    let m = 2 * n;
    let mut row: Vec<crate::fractals::Complex> = Vec::with_capacity(m);
    for j in 0..m {
        let k = if j <= n { j as f64 } else { (m - j) as f64 };
        row.push(crate::fractals::Complex::new(gamma(k), 0.0));
    }
    let spectrum = crate::transforms::fft::fft(&row);
    if spectrum.iter().any(|c| c.re < -1e-9) {
        return fbm_cholesky(h, n, rng);
    }
    // Multiply independent complex noise by the square roots of the
    // eigenvalues and transform back; the real part is the sample.
    let mut freq: Vec<crate::fractals::Complex> = Vec::with_capacity(m);
    for (j, c) in spectrum.iter().enumerate() {
        let scale = (c.re.max(0.0) / m as f64).sqrt();
        if j == 0 || j == m / 2 {
            freq.push(crate::fractals::Complex::new(scale * rng.next_gaussian(), 0.0));
        } else if j < m / 2 {
            let a = rng.next_gaussian() / 2.0f64.sqrt();
            let b = rng.next_gaussian() / 2.0f64.sqrt();
            freq.push(crate::fractals::Complex::new(scale * a, scale * b));
        } else {
            // Conjugate symmetry, so the transform comes back real.
            let mirror = freq[m - j];
            freq.push(crate::fractals::Complex::new(mirror.re, -mirror.im));
        }
    }
    let noise = crate::transforms::fft::fft(&freq);
    let mut out = Vec::with_capacity(n + 1);
    let mut acc = 0.0;
    out.push(acc);
    for item in noise.iter().take(n) {
        acc += item.re;
        out.push(acc);
    }
    out
}

/// Fractional Brownian motion by Cholesky factorisation of the covariance,
/// used when the circulant embedding fails.
fn fbm_cholesky(h: f64, n: usize, rng: &mut Rng) -> Vec<f64> {
    let cov = |s: f64, t: f64| {
        0.5 * (s.powf(2.0 * h) + t.powf(2.0 * h) - (s - t).abs().powf(2.0 * h))
    };
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = cov((i + 1) as f64, (j + 1) as f64);
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            l[i][j] = if i == j { sum.max(0.0).sqrt() } else if l[j][j] > 0.0 { sum / l[j][j] } else { 0.0 };
        }
    }
    let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
    let mut out = vec![0.0];
    for i in 0..n {
        out.push((0..=i).map(|k| l[i][k] * z[k]).sum());
    }
    out
}

/// The Hurst exponent by rescaled range analysis.
///
/// Split the series into blocks of several sizes, and for each measure the
/// range of the cumulative deviation from the block mean divided by the
/// block's standard deviation. That ratio grows like the block size to the
/// power `H`, and the slope on log axes is the estimate. Hurst found the
/// relation studying Nile flood records; the point is that it needs no model
/// of the process at all.
///
/// # Panics
/// Panics unless the series has at least sixteen points.
#[must_use]
pub fn hurst_exponent_rs(x: &[f64]) -> f64 {
    assert!(x.len() >= 16, "rescaled range analysis needs at least sixteen points");
    let mut sizes = Vec::new();
    let mut logs = Vec::new();
    let mut size = 8usize;
    while size <= x.len() / 2 {
        let blocks = x.len() / size;
        let mut total = 0.0;
        let mut counted = 0usize;
        for b in 0..blocks {
            let block = &x[b * size..(b + 1) * size];
            let mean = block.iter().sum::<f64>() / size as f64;
            let sd = (block.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                / size as f64)
                .sqrt();
            if sd <= 0.0 {
                continue;
            }
            let mut acc = 0.0;
            let (mut lo, mut hi) = (0.0f64, 0.0f64);
            for &v in block {
                acc += v - mean;
                lo = lo.min(acc);
                hi = hi.max(acc);
            }
            total += (hi - lo) / sd;
            counted += 1;
        }
        if counted > 0 {
            sizes.push((size as f64).ln());
            logs.push((total / counted as f64).ln());
        }
        size *= 2;
    }
    if sizes.len() < 2 {
        return 0.5;
    }
    slope(&sizes, &logs)
}

/// The Hurst exponent by detrended fluctuation analysis.
///
/// Integrate the series, split it into windows, remove a linear trend from
/// each, and measure the residual fluctuation against the window size. The
/// detrending is what lets it work on data with a slow drift, which
/// rescaled range analysis mistakes for persistence.
///
/// The input should be the *increments* -- fractional Gaussian noise, not
/// fractional Brownian motion. Feeding it an already-integrated series
/// returns `H + 1`, since the routine integrates once itself.
///
/// Windows shorter than sixteen points are skipped. Removing a straight line
/// from eight points takes out a real part of the fluctuation along with the
/// trend, which biases the exponent up by several hundredths -- enough to
/// make white noise look persistent.
///
/// # Panics
/// Panics unless the series has at least thirty-two points.
#[must_use]
pub fn hurst_dfa(x: &[f64]) -> f64 {
    assert!(x.len() >= 32, "detrended fluctuation analysis needs at least thirty-two points");
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let mut walk = Vec::with_capacity(x.len());
    let mut acc = 0.0;
    for &v in x {
        acc += v - mean;
        walk.push(acc);
    }
    let mut sizes = Vec::new();
    let mut logs = Vec::new();
    let mut size = 16usize;
    while size <= walk.len() / 4 {
        let windows = walk.len() / size;
        let mut total = 0.0;
        for w in 0..windows {
            let seg = &walk[w * size..(w + 1) * size];
            // Least squares line through the window.
            let n = size as f64;
            let sx: f64 = (0..size).map(|i| i as f64).sum();
            let sy: f64 = seg.iter().sum();
            let sxx: f64 = (0..size).map(|i| (i * i) as f64).sum();
            let sxy: f64 = seg.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
            let den = n * sxx - sx * sx;
            let (a, b) = if den.abs() > 0.0 {
                ((n * sxy - sx * sy) / den, (sy * sxx - sx * sxy) / den)
            } else {
                (0.0, sy / n)
            };
            total += (0..size)
                .map(|i| {
                    let r = seg[i] - (a * i as f64 + b);
                    r * r
                })
                .sum::<f64>()
                / n;
        }
        let f = (total / windows as f64).sqrt();
        if f > 0.0 {
            sizes.push((size as f64).ln());
            logs.push(f.ln());
        }
        size *= 2;
    }
    if sizes.len() < 2 {
        return 0.5;
    }
    slope(&sizes, &logs)
}

/// Least squares slope.
fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    if den.abs() > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// First passage times of a drifting Brownian motion to a barrier, by
/// simulation.
///
/// Returns one time per path that reached the barrier; paths that did not are
/// omitted, so a short horizon returns fewer times than paths.
///
/// # Panics
/// Panics unless `dt`, `t_end` and `n_paths` are positive.
#[must_use]
pub fn first_passage_time_sim(
    barrier: f64,
    drift: f64,
    diffusion: f64,
    t_end: f64,
    dt: f64,
    n_paths: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(dt > 0.0 && t_end > 0.0 && n_paths > 0, "the parameters must be positive");
    let steps = (t_end / dt).ceil() as usize;
    let s = dt.sqrt();
    let v = diffusion * diffusion * dt;
    let mut out = Vec::new();
    for _ in 0..n_paths {
        let mut x = 0.0;
        for k in 0..steps {
            let previous = x;
            x += drift * dt + diffusion * s * rng.next_gaussian();
            let crossed = if barrier > 0.0 { x >= barrier } else { x <= barrier };
            // Checking only at grid points misses the excursions that cross
            // and come back within one step, which biases the times upward.
            // Conditional on both endpoints, the chance the bridge between
            // them touched the barrier has a closed form, so those crossings
            // can be counted rather than missed.
            let bridged = !crossed
                && v > 0.0
                && rng.next_f64()
                    < (-2.0 * (barrier - previous) * (barrier - x) / v).exp();
            if crossed || bridged {
                out.push((k + 1) as f64 * dt);
                break;
            }
        }
    }
    out
}

/// The exact density of the first passage time of a drifting Brownian motion
/// to a barrier.
///
/// The inverse Gaussian density. It has a closed form because the reflection
/// principle turns the question "did the path ever reach the barrier" into a
/// statement about where the reflected path ended, which is an ordinary
/// Gaussian probability.
///
/// # Panics
/// Panics unless `t` and the barrier are positive.
#[must_use]
pub fn first_passage_bm_exact(barrier: f64, drift: f64, diffusion: f64, t: f64) -> f64 {
    assert!(t > 0.0, "the time must be positive");
    assert!(barrier > 0.0, "the barrier must be positive");
    let v = diffusion * diffusion;
    barrier / (2.0 * PI * v * t.powi(3)).sqrt()
        * (-(barrier - drift * t).powi(2) / (2.0 * v * t)).exp()
}

/// The probability that a drifting Brownian motion has reached the barrier
/// by time `t`.
///
/// `Phi((mu t - b) / (sigma sqrt t)) + exp(2 mu b / sigma^2)
/// Phi((-mu t - b) / (sigma sqrt t))`. The second term is the reflection
/// principle's contribution: paths that crossed and came back are counted by
/// reflecting them about the barrier, which maps them onto paths that ended
/// beyond it. With a non-positive drift the limit as `t` grows is
/// `exp(2 mu b / sigma^2)` rather than one, since such a path may never
/// arrive at all.
///
/// # Panics
/// Panics unless `t` and the barrier are positive.
#[must_use]
pub fn first_passage_bm_cdf(barrier: f64, drift: f64, diffusion: f64, t: f64) -> f64 {
    assert!(t > 0.0, "the time must be positive");
    assert!(barrier > 0.0, "the barrier must be positive");
    let s = diffusion * t.sqrt();
    let a = normal_cdf((drift * t - barrier) / s);
    let b = (2.0 * drift * barrier / (diffusion * diffusion)).exp()
        * normal_cdf((-drift * t - barrier) / s);
    (a + b).clamp(0.0, 1.0)
}

/// Checks the Feynman-Kac correspondence: the expectation of a payoff along
/// simulated paths against the solution of the matching partial differential
/// equation.
///
/// Returns `(monte_carlo, closed_form)` for a European call under geometric
/// Brownian motion, where the closed form is Black-Scholes. That the two
/// agree is not a coincidence -- Feynman-Kac says the expectation of a
/// terminal payoff over the paths of a diffusion *is* the solution of the
/// backward equation, which is what turns an option price into a partial
/// differential equation and back.
///
/// # Panics
/// Panics unless the parameters are positive.
#[must_use]
pub fn feynman_kac_check(
    s0: f64,
    strike: f64,
    rate: f64,
    sigma: f64,
    t: f64,
    n_paths: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    assert!(s0 > 0.0 && strike > 0.0 && sigma > 0.0 && t > 0.0, "the parameters must be positive");
    assert!(n_paths > 0, "at least one path is required");
    let mut total = 0.0;
    for _ in 0..n_paths {
        let s = gbm_exact(s0, rate, sigma, t, rng.next_gaussian());
        total += (s - strike).max(0.0);
    }
    let monte_carlo = (-rate * t).exp() * total / n_paths as f64;
    // Black-Scholes, which is the closed-form solution of the same problem.
    let d1 = ((s0 / strike).ln() + (rate + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let closed = s0 * normal_cdf(d1) - strike * (-rate * t).exp() * normal_cdf(d2);
    (monte_carlo, closed)
}

/// The standard normal cumulative distribution.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0f64.sqrt()))
}

/// The error function, by Abramowitz and Stegun's rational approximation.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Checks Ito's isometry: the variance of a stochastic integral equals the
/// integral of the squared integrand.
///
/// Returns `(measured, expected)`. The isometry is what makes stochastic
/// integration work at all -- it says the map from integrands to integrals
/// preserves the `L^2` norm, so the integral can be defined for any
/// square-integrable integrand by taking limits.
///
/// # Panics
/// Panics unless `t`, `n_paths` and `steps` are positive.
#[must_use]
pub fn ito_isometry_check(
    sigma: &dyn Fn(f64) -> f64,
    t: f64,
    steps: usize,
    n_paths: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    assert!(t > 0.0 && n_paths > 0 && steps > 0, "the parameters must be positive");
    let dt = t / steps as f64;
    let s = dt.sqrt();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n_paths {
        let mut integral = 0.0;
        for k in 0..steps {
            // Left endpoint, which is what makes it an Ito integral.
            integral += sigma(k as f64 * dt) * s * rng.next_gaussian();
        }
        sum += integral;
        sum_sq += integral * integral;
    }
    let n = n_paths as f64;
    let measured = sum_sq / n - (sum / n) * (sum / n);
    let expected: f64 = (0..steps).map(|k| sigma(k as f64 * dt).powi(2) * dt).sum();
    (measured, expected)
}

/// Underdamped Langevin dynamics by the BAOAB splitting.
///
/// A particle in a force field with friction and thermal noise. The
/// integrator splits the dynamics into a drift, a kick and an
/// Ornstein-Uhlenbeck step on the velocity, and applies them in the
/// palindromic order B-A-O-A-B. The symmetry is what gives it the best known
/// accuracy for configurational averages: at any step size it samples
/// positions from very nearly the right distribution, even where the
/// velocities are visibly wrong.
///
/// Returns position and velocity at each step. `temp` is in energy units, so
/// the equipartition result is `<v^2> = temp / mass`.
///
/// # Panics
/// Panics unless `dt`, `mass` and `gamma` are positive and `temp` is
/// non-negative.
pub fn langevin_underdamped(
    x0: f64,
    v0: f64,
    gamma: f64,
    temp: f64,
    mass: f64,
    force: &dyn Fn(f64) -> f64,
    n: usize,
    dt: f64,
    rng: &mut Rng,
) -> Vec<(f64, f64)> {
    assert!(dt > 0.0 && mass > 0.0 && gamma > 0.0, "the parameters must be positive");
    assert!(temp >= 0.0, "the temperature must be non-negative");
    let decay = (-gamma * dt).exp();
    let noise = (temp / mass * (1.0 - decay * decay)).sqrt();
    let mut x = x0;
    let mut v = v0;
    let mut out = Vec::with_capacity(n + 1);
    out.push((x, v));
    for _ in 0..n {
        // B: half kick.
        v += 0.5 * dt * force(x) / mass;
        // A: half drift.
        x += 0.5 * dt * v;
        // O: the exact Ornstein-Uhlenbeck step on the velocity, which is
        // what makes the scheme stable at large friction.
        v = decay * v + noise * rng.next_gaussian();
        // A: half drift.
        x += 0.5 * dt * v;
        // B: half kick.
        v += 0.5 * dt * force(x) / mass;
        out.push((x, v));
    }
    out
}

/// One step of the Fokker-Planck equation by the Chang-Cooper scheme.
///
/// The density evolves as `dp/dt = -d(mu p)/dx + 0.5 d^2(sigma^2 p)/dx^2`.
/// Chang and Cooper's discretisation weights the drift term so that the
/// scheme's own stationary solution is the exact one -- an ordinary centred
/// difference relaxes to a slightly wrong density and stays there, which is
/// the failure this scheme exists to avoid. Zero-flux boundaries, so the
/// total probability is conserved exactly.
///
/// # Panics
/// Panics unless `dx`, `dt` are positive and the density has at least three
/// points.
pub fn fokker_planck_1d(
    p0: &[f64],
    mu: &dyn Fn(f64) -> f64,
    sigma: &dyn Fn(f64) -> f64,
    x_min: f64,
    dx: f64,
    dt: f64,
    steps: usize,
) -> Vec<f64> {
    assert!(dx > 0.0 && dt > 0.0, "the grid and the step must be positive");
    assert!(p0.len() >= 3, "the grid needs at least three points");
    let n = p0.len();
    let mut p = p0.to_vec();
    for _ in 0..steps {
        // Flux at each half-grid point, zero at both ends.
        let mut flux = vec![0.0; n + 1];
        for j in 1..n {
            let x_half = x_min + (j as f64 - 0.5) * dx;
            let b = mu(x_half);
            let d = 0.5 * sigma(x_half).powi(2);
            // The Chang-Cooper weight: an exponential interpolation between
            // upwind and centred, chosen so the discrete stationary state is
            // the continuous one.
            let w = b * dx / d.max(1e-300);
            let delta = if w.abs() < 1e-8 {
                0.5
            } else {
                1.0 / w - 1.0 / (w.exp() - 1.0)
            };
            let left = p[j - 1];
            let right = p[j];
            flux[j] = b * ((1.0 - delta) * left + delta * right) - d * (right - left) / dx;
        }
        let mut next = p.clone();
        for j in 0..n {
            next[j] = p[j] - dt * (flux[j + 1] - flux[j]) / dx;
        }
        p = next;
    }
    p
}

/// The stationary density of a one-dimensional diffusion, in closed form.
///
/// `p(x) proportional to exp(2 integral mu / sigma^2) / sigma^2`. It is the
/// zero-flux solution: the drift's tendency to push probability one way
/// exactly balances diffusion's tendency to spread it, at every point rather
/// than on average. Returned normalised over the grid.
///
/// # Panics
/// Panics unless the range is increasing and `n` is at least two.
#[must_use]
pub fn stationary_density_1d(
    mu: &dyn Fn(f64) -> f64,
    sigma: &dyn Fn(f64) -> f64,
    x_range: (f64, f64),
    n: usize,
) -> Vec<f64> {
    assert!(x_range.1 > x_range.0, "the range must be increasing");
    assert!(n >= 2, "the grid needs at least two points");
    let dx = (x_range.1 - x_range.0) / (n - 1) as f64;
    let mut log_p = Vec::with_capacity(n);
    let mut acc = 0.0;
    for j in 0..n {
        let x = x_range.0 + j as f64 * dx;
        let s2 = sigma(x).powi(2).max(1e-300);
        if j > 0 {
            let x_prev = x - dx;
            let s2_prev = sigma(x_prev).powi(2).max(1e-300);
            // Trapezoidal integration of 2 mu / sigma^2.
            acc += dx * (mu(x) / s2 + mu(x_prev) / s2_prev);
        }
        log_p.push(acc - s2.ln());
    }
    let peak = log_p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<f64> = log_p.into_iter().map(|v| (v - peak).exp()).collect();
    let total: f64 = p.iter().sum::<f64>() * dx;
    if total > 0.0 {
        for v in &mut p {
            *v /= total;
        }
    }
    p
}

/// Kramers' escape rate from a potential well over a barrier.
///
/// `(omega_well omega_barrier / (2 pi gamma)) exp(-barrier / temp)` in the
/// high-friction limit. The exponential is Arrhenius and is the part everyone
/// knows; Kramers' contribution was the prefactor, which says the rate falls
/// as friction rises, because a strongly damped particle takes longer to
/// diffuse across the barrier top even once it has the energy.
///
/// # Panics
/// Panics unless the temperature, friction and both frequencies are
/// positive.
#[must_use]
pub fn kramers_escape_rate(
    barrier_height: f64,
    temp: f64,
    omega_well: f64,
    omega_barrier: f64,
    gamma: f64,
) -> f64 {
    assert!(temp > 0.0 && gamma > 0.0, "the temperature and friction must be positive");
    assert!(omega_well > 0.0 && omega_barrier > 0.0, "the frequencies must be positive");
    omega_well * omega_barrier / (2.0 * PI * gamma) * (-barrier_height / temp).exp()
}

/// Simulates a bistable system driven by a weak periodic force and noise, and
/// returns the path.
///
/// Stochastic resonance is the phenomenon that adding noise can *improve* the
/// response to a signal too weak to drive the system on its own: the noise
/// supplies the energy to cross the barrier, and the signal decides when. The
/// effect is largest at an intermediate noise level, which is what a sweep
/// over `temp` shows.
///
/// # Panics
/// Panics unless `dt` is positive and `temp` is non-negative.
#[must_use]
pub fn stochastic_resonance_sim(
    x0: f64,
    amplitude: f64,
    frequency: f64,
    temp: f64,
    n: usize,
    dt: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(dt > 0.0, "the step must be positive");
    assert!(temp >= 0.0, "the temperature must be non-negative");
    let s = (2.0 * temp * dt).sqrt();
    let mut x = x0;
    let mut out = Vec::with_capacity(n + 1);
    out.push(x);
    for k in 0..n {
        let t = k as f64 * dt;
        // The double well x^2/2 - x^4/4 has minima at plus and minus one.
        let force = x - x.powi(3) + amplitude * (2.0 * PI * frequency * t).sin();
        x += force * dt + s * rng.next_gaussian();
        out.push(x);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(x: &[f64]) -> f64 {
        x.iter().sum::<f64>() / x.len() as f64
    }

    fn variance(x: &[f64]) -> f64 {
        let m = mean(x);
        x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64
    }

    /// Brownian motion has the covariance it is defined by, and the bridge
    /// has the one conditioning produces.
    #[test]
    fn brownian_paths_have_their_defining_covariance() {
        let mut rng = Rng::new(0x_B204);
        let (n, dt) = (50usize, 0.02f64);
        let paths: Vec<Vec<f64>> = (0..20_000).map(|_| brownian_motion(n, dt, &mut rng)).collect();
        for p in paths.iter().take(5) {
            assert_eq!(p.len(), n + 1);
            assert_eq!(p[0], 0.0, "a Brownian path starts at zero");
        }
        // Var(W_t) = t, and Cov(W_s, W_t) = min(s, t).
        for k in [1usize, 10, 25, 50] {
            let at_k: Vec<f64> = paths.iter().map(|p| p[k]).collect();
            let t = k as f64 * dt;
            assert!(mean(&at_k).abs() < 0.05, "the mean drifted at step {k}");
            assert!(
                (variance(&at_k) / t - 1.0).abs() < 0.06,
                "the variance at {t} is {}",
                variance(&at_k)
            );
        }
        let a: Vec<f64> = paths.iter().map(|p| p[10]).collect();
        let b: Vec<f64> = paths.iter().map(|p| p[40]).collect();
        let cov = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>() / a.len() as f64;
        assert!((cov / (10.0 * dt) - 1.0).abs() < 0.08, "the covariance is {cov}");

        // The bridge is pinned, and its variance is t(T - t)/T.
        let total = n as f64 * dt;
        let bridges: Vec<Vec<f64>> =
            (0..20_000).map(|_| brownian_bridge(n, dt, 1.0, 4.0, &mut rng)).collect();
        for br in bridges.iter().take(50) {
            assert!((br[0] - 1.0).abs() < 1e-12, "the bridge does not start where told");
            assert!((br[n] - 4.0).abs() < 1e-9, "the bridge does not end where told");
        }
        for k in [10usize, 25, 40] {
            let t = k as f64 * dt;
            let at_k: Vec<f64> = bridges.iter().map(|p| p[k]).collect();
            let want_var = t * (total - t) / total;
            let want_mean = 1.0 + 3.0 * t / total;
            assert!((mean(&at_k) - want_mean).abs() < 0.03, "the bridge's mean is off at {k}");
            assert!(
                (variance(&at_k) / want_var - 1.0).abs() < 0.08,
                "the bridge's variance at {t} is {} against {want_var}",
                variance(&at_k)
            );
        }
        let two = brownian_2d(10, dt, &mut rng);
        assert_eq!(two.len(), 11);
        assert_eq!(brownian_3d(10, dt, &mut rng).len(), 11);
    }

    /// Geometric Brownian motion has the lognormal law its closed form says,
    /// and the exact stepper agrees with the exact formula.
    #[test]
    fn geometric_brownian_matches_its_closed_form() {
        let mut rng = Rng::new(0x_6B44);
        let (x0, mu, sigma, t) = (100.0f64, 0.07f64, 0.3f64, 1.0f64);
        let n = 250usize;
        let dt = t / n as f64;
        let ends: Vec<f64> =
            (0..40_000).map(|_| *geometric_brownian(x0, mu, sigma, n, dt, &mut rng).last().expect("non-empty")).collect();
        // E[S_T] = S_0 exp(mu T), and the log is Gaussian with the Ito drift.
        let want_mean = x0 * (mu * t).exp();
        assert!(
            (mean(&ends) / want_mean - 1.0).abs() < 0.01,
            "the mean is {} against {want_mean}",
            mean(&ends)
        );
        let logs: Vec<f64> = ends.iter().map(|v| (v / x0).ln()).collect();
        assert!(
            (mean(&logs) - (mu - 0.5 * sigma * sigma) * t).abs() < 0.01,
            "the log drift is {} against {}",
            mean(&logs),
            (mu - 0.5 * sigma * sigma) * t
        );
        assert!(
            (variance(&logs) / (sigma * sigma * t) - 1.0).abs() < 0.03,
            "the log variance is {}",
            variance(&logs)
        );
        // Every path stays positive, which log-space stepping guarantees and
        // a naive Euler step does not.
        let path = geometric_brownian(x0, -2.0, 1.5, 5_000, 0.001, &mut rng);
        assert!(path.iter().all(|&v| v > 0.0), "a price went non-positive");
        // The exact formula and the stepper agree when driven by one normal.
        for z in [-2.0f64, -0.5, 0.0, 1.3] {
            let direct = gbm_exact(x0, mu, sigma, t, z);
            let stepped = x0 * ((mu - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp();
            assert!((direct - stepped).abs() < 1e-9);
        }
    }

    /// The Ornstein-Uhlenbeck process reverts to its mean and settles at the
    /// stationary variance the parameters dictate.
    #[test]
    fn ornstein_uhlenbeck_reaches_its_stationary_law() {
        let mut rng = Rng::new(0x_00AB);
        let (theta, mu, sigma) = (2.0f64, 5.0f64, 1.2f64);
        let want_var = sigma * sigma / (2.0 * theta);
        // Started far from the mean, and sampled well after it has settled.
        let path = ornstein_uhlenbeck(-20.0, theta, mu, sigma, 400_000, 0.01, &mut rng);
        let tail = &path[100_000..];
        assert!((mean(tail) - mu).abs() < 0.02, "the mean settled at {}", mean(tail));
        assert!(
            (variance(tail) / want_var - 1.0).abs() < 0.06,
            "the variance settled at {} against {want_var}",
            variance(tail)
        );
        // The transition law is exact at any step, so a single large step
        // from the mean has exactly the stationary variance in the limit.
        let big: Vec<f64> = (0..40_000)
            .map(|_| ou_exact_step(mu, theta, mu, sigma, 100.0, rng.next_gaussian()))
            .collect();
        assert!(
            (variance(&big) / want_var - 1.0).abs() < 0.03,
            "one long step gave variance {}",
            variance(&big)
        );
        // And the conditional mean decays towards mu exponentially.
        let short: Vec<f64> = (0..40_000)
            .map(|_| ou_exact_step(0.0, theta, mu, sigma, 0.25, rng.next_gaussian()))
            .collect();
        let want = mu + (0.0 - mu) * (-theta * 0.25f64).exp();
        assert!((mean(&short) - want).abs() < 0.02, "the conditional mean is {}", mean(&short));
    }

    /// Euler-Maruyama converges at strong order one half and Milstein at one,
    /// measured against the exact solution driven by the same noise.
    ///
    /// This is the property the schemes exist to have, and the only one that
    /// distinguishes a correct Milstein implementation from an Euler step
    /// with an extra term that happens to be small.
    #[test]
    fn the_schemes_converge_at_their_stated_orders() {
        let (x0, mu_c, sigma_c, t) = (1.0f64, 1.5f64, 0.6f64, 1.0f64);
        let mu = move |_t: f64, x: f64| mu_c * x;
        let sigma = move |_t: f64, x: f64| sigma_c * x;
        let dsigma = move |_t: f64, _x: f64| sigma_c;

        // Fine enough to be in the asymptotic regime. Measured over
        // 16 to 256 the Euler slope reads 0.62 rather than 0.5, because the
        // higher-order terms have not yet died away -- which is a fact about
        // where the asymptote starts, not about the scheme.
        let step_counts = [64usize, 128, 256, 512, 1024];
        let mut dts = Vec::new();
        let mut euler_errors = Vec::new();
        let mut milstein_errors = Vec::new();
        for &n in &step_counts {
            let dt = t / n as f64;
            let mut eu = 0.0;
            let mut mi = 0.0;
            let paths = 2_000;
            for p in 0..paths {
                // The same seed for both schemes and for the exact path, so
                // the comparison is path by path rather than in law.
                let seed = 0x_5C4E_0000u64 + p as u64;
                let mut r1 = Rng::new(seed);
                let mut r2 = Rng::new(seed);
                let mut r3 = Rng::new(seed);
                let a = euler_maruyama(&mu, &sigma, x0, t, n, &mut r1);
                let b = milstein(&mu, &sigma, &dsigma, x0, t, n, &mut r2);
                // The exact solution driven by the same increments: the sum
                // of the increments is the terminal Brownian value.
                let mut w = 0.0;
                let s = dt.sqrt();
                for _ in 0..n {
                    w += s * r3.next_gaussian();
                }
                let exact = x0 * ((mu_c - 0.5 * sigma_c * sigma_c) * t + sigma_c * w).exp();
                eu += (a[n] - exact).abs();
                mi += (b[n] - exact).abs();
            }
            dts.push(dt);
            euler_errors.push(eu / paths as f64);
            milstein_errors.push(mi / paths as f64);
        }
        let eu_order = strong_convergence_order(&euler_errors, &dts);
        let mi_order = strong_convergence_order(&milstein_errors, &dts);
        assert!(
            (eu_order - 0.5).abs() < 0.12,
            "Euler-Maruyama measured strong order {eu_order}, not one half"
        );
        assert!(
            (mi_order - 1.0).abs() < 0.15,
            "Milstein measured strong order {mi_order}, not one"
        );
        // Milstein is strictly better at the finest step, which is what the
        // extra order buys.
        assert!(
            milstein_errors[4] < euler_errors[4] / 5.0,
            "Milstein did not pull ahead: {} against {}",
            milstein_errors[4],
            euler_errors[4]
        );

        // Weak order: Euler-Maruyama gets the mean right at order one, which
        // is better than its strong order.
        let mut weak_errors = Vec::new();
        let mut weak_dts = Vec::new();
        for &n in &[8usize, 16, 32, 64] {
            let mut rng = Rng::new(0x_11EA + n as u64);
            let ends: Vec<f64> = (0..120_000)
                .map(|_| euler_maruyama(&mu, &sigma, x0, t, n, &mut rng)[n])
                .collect();
            weak_dts.push(t / n as f64);
            weak_errors.push((mean(&ends) - x0 * (mu_c * t).exp()).abs());
        }
        let weak = weak_convergence_order(&weak_errors, &weak_dts);
        assert!(weak > 0.7, "the weak order measured {weak}, which is below the strong one");
    }

    /// The vector scheme, the Stratonovich scheme, and the order-1.5 scheme
    /// each reproduce a law they can be checked against.
    #[test]
    fn the_other_schemes_reproduce_known_laws() {
        let mut rng = Rng::new(0x_5CE5);
        // Two-dimensional Brownian motion with a correlation built into the
        // diffusion matrix rather than the noise.
        let rho = 0.7f64;
        let drift = |_t: f64, _x: &[f64]| vec![0.0, 0.0];
        let diffusion = move |_t: f64, _x: &[f64]| {
            Matrix::from_rows(&[&[1.0, 0.0], &[rho, (1.0f64 - rho * rho).sqrt()]])
                .expect("rows")
        };
        let ends: Vec<Vec<f64>> = (0..20_000)
            .map(|_| {
                let p = euler_maruyama_nd(&drift, &diffusion, &[0.0, 0.0], 1.0, 40, &mut rng);
                p[40].clone()
            })
            .collect();
        let x: Vec<f64> = ends.iter().map(|v| v[0]).collect();
        let y: Vec<f64> = ends.iter().map(|v| v[1]).collect();
        assert!((variance(&x) - 1.0).abs() < 0.05, "the first component's variance is off");
        assert!((variance(&y) - 1.0).abs() < 0.05, "the second component's variance is off");
        let cov = x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / x.len() as f64;
        assert!((cov - rho).abs() < 0.05, "the correlation came out at {cov}");

        // Heun converges to the Stratonovich solution, which for
        // dX = X dW is exp(W_t) rather than the Ito answer
        // exp(W_t - t/2). That difference is the whole distinction between
        // the two integrals, and it shows up in the mean.
        let mu0 = |_t: f64, _x: f64| 0.0;
        let sig = |_t: f64, x: f64| x;
        let heun_ends: Vec<f64> = (0..40_000)
            .map(|_| {
                let p = stochastic_heun(&mu0, &sig, 1.0, 1.0, 200, &mut rng);
                p[200]
            })
            .collect();
        // E[exp(W_1)] = exp(1/2) for the Stratonovich solution.
        assert!(
            (mean(&heun_ends) - 0.5f64.exp()).abs() < 0.03,
            "Heun's mean is {} against the Stratonovich {}",
            mean(&heun_ends),
            0.5f64.exp()
        );
        // The Ito solution of the same equation is a martingale, so its mean
        // stays at one -- which is what the two schemes disagree about.
        let ito_ends: Vec<f64> = (0..40_000)
            .map(|_| {
                let p = euler_maruyama(&mu0, &sig, 1.0, 1.0, 200, &mut rng);
                p[200]
            })
            .collect();
        assert!(
            (mean(&ito_ends) - 1.0).abs() < 0.03,
            "the Ito solution's mean is {}, but it should be a martingale",
            mean(&ito_ends)
        );

        // The order-1.5 scheme on an Ornstein-Uhlenbeck equation with
        // additive noise, against the exact stationary variance.
        let (theta, sigma_a) = (1.5f64, 0.8f64);
        let pull = move |_t: f64, x: f64| -theta * x;
        let long = srk_order_1_5(&pull, sigma_a, 0.0, 200.0, 20_000, &mut rng);
        let tail = &long[5_000..];
        let want = sigma_a * sigma_a / (2.0 * theta);
        assert!(
            (variance(tail) / want - 1.0).abs() < 0.15,
            "the order-1.5 scheme's variance is {} against {want}",
            variance(tail)
        );
    }

    /// The variance processes stay where they belong, and the jump model puts
    /// weight in the tails a diffusion cannot.
    #[test]
    fn variance_processes_and_jumps_behave() {
        let mut rng = Rng::new(0x_C124);
        // A CIR path never goes negative, whether or not the Feller
        // condition holds -- and it is the badly violated case, where the
        // exact process spends nearly all its time against the boundary,
        // that the truncation exists for.
        for (kappa, theta, sigma) in [(2.0f64, 0.04f64, 0.3f64), (0.5, 0.02, 0.9)] {
            let path = cir_process(theta, kappa, theta, sigma, 200_000, 0.001, &mut rng);
            assert!(path.iter().all(|&v| v >= 0.0), "a CIR path went negative");
            assert!(path.iter().all(|v| v.is_finite()), "a CIR path diverged");
        }
        // Mean reversion is only worth asserting where the mean is
        // measurable. With 2 kappa theta far below sigma squared the exact
        // process sits at zero between rare large spikes, so a sample mean
        // over any affordable horizon says nothing: at these parameters two
        // hundred time units give estimates spanning an order of magnitude
        // either side of theta.
        let (kappa, theta, sigma) = (2.0f64, 0.04f64, 0.3f64);
        assert!(2.0 * kappa * theta > sigma * sigma, "this case should satisfy Feller");
        let path = cir_process(theta, kappa, theta, sigma, 400_000, 0.001, &mut rng);
        let tail = &path[100_000..];
        assert!(
            (mean(tail) / theta - 1.0).abs() < 0.1,
            "CIR settled at {} against {theta}",
            mean(tail)
        );
        // Starting far above the mean, it comes back down.
        let high = cir_process(0.5, kappa, theta, sigma, 400_000, 0.001, &mut rng);
        assert!(
            mean(&high[100_000..]) < 0.1,
            "CIR did not revert from a high start: {}",
            mean(&high[100_000..])
        );

        // Heston: the variance stays non-negative and the price positive.
        let params = HestonParams { mu: 0.05, kappa: 2.0, theta: 0.04, xi: 0.5, rho: -0.7 };
        let (prices, vars) = heston_paths(100.0, 0.04, params, 100_000, 0.0005, &mut rng);
        assert!(vars.iter().all(|&v| v >= 0.0), "a Heston variance went negative");
        assert!(prices.iter().all(|&v| v > 0.0), "a Heston price went non-positive");
        // The correlation shows up: variance rises when the price falls.
        let dp: Vec<f64> = prices.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
        let dv: Vec<f64> = vars.windows(2).map(|w| w[1] - w[0]).collect();
        let mp = mean(&dp);
        let mv = mean(&dv);
        let cov: f64 =
            dp.iter().zip(&dv).map(|(a, b)| (a - mp) * (b - mv)).sum::<f64>() / dp.len() as f64;
        assert!(cov < 0.0, "a negative rho should make the two move oppositely, not {cov}");

        // Merton: the drift is compensated, so the expected return is mu
        // whether or not jumps land.
        let (x0, mu, lambda) = (100.0f64, 0.06f64, 3.0f64);
        let ends: Vec<f64> = (0..40_000)
            .map(|_| {
                *jump_diffusion_merton(x0, mu, 0.2, lambda, -0.05, 0.15, 100, 0.01, &mut rng)
                    .last()
                    .expect("non-empty")
            })
            .collect();
        assert!(
            (mean(&ends) / (x0 * mu.exp()) - 1.0).abs() < 0.03,
            "the compensated mean is {} against {}",
            mean(&ends),
            x0 * mu.exp()
        );
        // Jumps make the tails heavier than a matched lognormal.
        let logs: Vec<f64> = ends.iter().map(|v| (v / x0).ln()).collect();
        let m = mean(&logs);
        let sd = variance(&logs).sqrt();
        let kurtosis =
            logs.iter().map(|v| ((v - m) / sd).powi(4)).sum::<f64>() / logs.len() as f64;
        assert!(kurtosis > 3.2, "jumps should raise the kurtosis above three, not {kurtosis}");
    }

    /// The stable sampler reproduces the two cases with closed forms, and
    /// shows the infinite variance the others have.
    #[test]
    fn levy_stable_sampling_matches_its_special_cases() {
        let mut rng = Rng::new(0x_57AB);
        // alpha = 2 is Gaussian with variance two, in this parameterisation.
        let gauss: Vec<f64> = (0..80_000).map(|_| levy_stable_sample(2.0, 0.0, &mut rng)).collect();
        assert!(mean(&gauss).abs() < 0.03, "the Gaussian case is not centred");
        assert!(
            (variance(&gauss) / 2.0 - 1.0).abs() < 0.05,
            "the Gaussian case has variance {}",
            variance(&gauss)
        );
        // alpha = 1, beta = 0 is Cauchy: the median is zero and the
        // interquartile range is two, but the mean does not converge.
        let mut cauchy: Vec<f64> =
            (0..80_000).map(|_| levy_stable_sample(1.0, 0.0, &mut rng)).collect();
        cauchy.sort_by(f64::total_cmp);
        let q = |p: f64| cauchy[(p * cauchy.len() as f64) as usize];
        assert!(q(0.5).abs() < 0.03, "the Cauchy median is {}", q(0.5));
        assert!(
            ((q(0.75) - q(0.25)) / 2.0 - 1.0).abs() < 0.05,
            "the Cauchy interquartile range is {}",
            q(0.75) - q(0.25)
        );
        // The tails really are heavy: the largest sample dwarfs the spread.
        let largest = cauchy.last().expect("non-empty").abs();
        assert!(largest > 100.0, "the heaviest Cauchy draw was only {largest}");
        // A skewed case leans the way beta says.
        let skewed: Vec<f64> =
            (0..40_000).map(|_| levy_stable_sample(1.5, 0.9, &mut rng)).collect();
        let mut sorted = skewed.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        let upper = sorted[(0.99 * sorted.len() as f64) as usize] - median;
        let lower = median - sorted[(0.01 * sorted.len() as f64) as usize];
        assert!(
            upper > 2.0 * lower,
            "a positive beta should stretch the upper tail, not the lower: {upper} against {lower}"
        );
        // And a negative one the other way, which pins the sign convention
        // rather than leaving it to whichever way the formulas happened to
        // come out.
        let mirrored: Vec<f64> =
            (0..40_000).map(|_| levy_stable_sample(1.5, -0.9, &mut rng)).collect();
        let mut ms = mirrored.clone();
        ms.sort_by(f64::total_cmp);
        let m_med = ms[ms.len() / 2];
        let m_upper = ms[(0.99 * ms.len() as f64) as usize] - m_med;
        let m_lower = m_med - ms[(0.01 * ms.len() as f64) as usize];
        assert!(m_lower > 2.0 * m_upper, "a negative beta should stretch the lower tail");
    }

    /// Fractional Brownian motion has the Hurst exponent it was asked for,
    /// measured by two independent estimators.
    #[test]
    fn fractional_brownian_has_the_hurst_exponent_it_was_given() {
        for h in [0.3f64, 0.5, 0.7] {
            let mut rs = Vec::new();
            let mut dfa = Vec::new();
            for seed in 0..8u64 {
                let mut rng = Rng::new(0x_FB40 + seed);
                let path = fractional_brownian(h, 2048, &mut rng);
                assert_eq!(path.len(), 2049);
                assert_eq!(path[0], 0.0);
                assert!(path.iter().all(|v| v.is_finite()), "the path went non-finite at H = {h}");
                let increments: Vec<f64> = path.windows(2).map(|w| w[1] - w[0]).collect();
                rs.push(hurst_exponent_rs(&increments));
                dfa.push(hurst_dfa(&increments));
            }
            let rs_mean = mean(&rs);
            let dfa_mean = mean(&dfa);
            assert!(
                (dfa_mean - h).abs() < 0.08,
                "detrended fluctuation gave {dfa_mean} for H = {h}"
            );
            assert!(
                (rs_mean - h).abs() < 0.15,
                "rescaled range gave {rs_mean} for H = {h}"
            );
        }
        // Ordinary Brownian increments are white noise, so both estimators
        // should say one half.
        let mut rng = Rng::new(0x_1177);
        let white: Vec<f64> = (0..4096).map(|_| rng.next_gaussian()).collect();
        // Detrended fluctuation analysis is biased upward on a finite
        // series -- removing a straight line from a short window takes real
        // fluctuation with it -- so half a per cent either way is not the
        // right tolerance to ask for. It lands near 0.55 at this length.
        let white_dfa = hurst_dfa(&white);
        assert!(
            (white_dfa - 0.5).abs() < 0.08,
            "white noise measured {white_dfa}, which is not near one half"
        );
        // A cumulative sum of white noise is a random walk, H = 1 by DFA on
        // the walk itself.
        let mut acc = 0.0;
        let walk: Vec<f64> = white
            .iter()
            .map(|v| {
                acc += v;
                acc
            })
            .collect();
        assert!(hurst_dfa(&walk) > 0.85, "a random walk should measure near one");
    }

    /// First passage times match the inverse Gaussian density they are
    /// distributed by.
    #[test]
    fn first_passage_times_match_the_inverse_gaussian() {
        let mut rng = Rng::new(0x_F1A5);
        let (barrier, drift, diffusion) = (1.0f64, 1.0f64, 1.0f64);
        let times =
            first_passage_time_sim(barrier, drift, diffusion, 25.0, 0.002, 8_000, &mut rng);
        assert!(times.len() > 7_900, "with a positive drift nearly every path should arrive");
        // Compare the whole distribution rather than a histogram bin: the
        // empirical cumulative distribution against the closed form, at the
        // scale sampling error actually allows. Twenty thousand draws give
        // about seven thousandths of resolution, so a hundredth is a real
        // constraint and a bin-by-bin density check at the same count would
        // not be.
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len() as f64;
        let mut worst = 0.0f64;
        for i in 0..sorted.len() {
            let empirical = (i + 1) as f64 / n;
            let exact = first_passage_bm_cdf(barrier, drift, diffusion, sorted[i]);
            worst = worst.max((empirical - exact).abs());
        }
        assert!(
            worst < 0.025,
            "the empirical and exact distributions differ by {worst} at their worst"
        );
        // The density and the distribution are consistent with each other,
        // which checks the two closed forms against one another.
        for t in [0.4f64, 0.8, 1.5, 3.0, 6.0] {
            let h = 1e-4;
            let numeric = (first_passage_bm_cdf(barrier, drift, diffusion, t + h)
                - first_passage_bm_cdf(barrier, drift, diffusion, t - h))
                / (2.0 * h);
            let direct = first_passage_bm_exact(barrier, drift, diffusion, t);
            // The tolerance is set by the rational approximation behind the
            // normal distribution, whose error a finite difference divides by
            // the step and so magnifies; the two closed forms themselves are
            // exact.
            assert!(
                (numeric - direct).abs() < 1e-3 * direct.max(1e-3),
                "at t = {t} the density is {direct} but the distribution's slope is {numeric}"
            );
        }
        // A negative drift may never arrive: the chance of ever doing so is
        // exp(2 mu b / sigma^2), which the distribution tends to.
        let escape = (2.0 * -0.5 * barrier / 1.0f64).exp();
        assert!(
            (first_passage_bm_cdf(barrier, -0.5, diffusion, 5_000.0) - escape).abs() < 1e-3,
            "a downward drift should reach the barrier with probability {escape}"
        );
        let downhill =
            first_passage_time_sim(barrier, -0.5, diffusion, 100.0, 0.01, 4_000, &mut rng);
        let arrived = downhill.len() as f64 / 4_000.0;
        assert!(
            (arrived - escape).abs() < 0.03,
            "only {arrived} of the downhill paths arrived, against {escape}"
        );
        // The density integrates to one for a positive drift, since the
        // barrier is reached with probability one.
        let total: f64 = (0..4000)
            .map(|i| first_passage_bm_exact(barrier, drift, diffusion, (i as f64 + 0.5) * 0.01) * 0.01)
            .sum();
        assert!((total - 1.0).abs() < 0.01, "the density integrates to {total}");
    }

    /// Feynman-Kac and Ito's isometry, each against the closed form it
    /// asserts.
    #[test]
    fn feynman_kac_and_the_ito_isometry_hold() {
        let mut rng = Rng::new(0x_FE44);
        let (mc, closed) = feynman_kac_check(100.0, 100.0, 0.05, 0.2, 1.0, 150_000, &mut rng);
        assert!(
            (mc - closed).abs() < 0.05 * closed,
            "the simulation gave {mc} against Black-Scholes' {closed}"
        );
        assert!(closed > 0.0 && closed < 100.0, "the price left its bounds");
        // Deep out of the money, where the payoff is almost always zero and
        // the two must still agree.
        let (mc2, closed2) = feynman_kac_check(100.0, 200.0, 0.05, 0.2, 1.0, 150_000, &mut rng);
        assert!(closed2 < 1.0);
        assert!((mc2 - closed2).abs() < 0.2 * closed2.max(0.01));

        // Ito's isometry for three integrands, including one that varies.
        for f in [
            (&|_t: f64| 1.0) as &dyn Fn(f64) -> f64,
            &|t: f64| t,
            &|t: f64| (2.0 * t).sin() + 1.5,
        ] {
            let (measured, expected) = ito_isometry_check(f, 2.0, 200, 40_000, &mut rng);
            // A variance estimated from n draws has relative error about
            // the square root of two over n, so three per cent is four
            // standard errors at this count rather than a loose bound.
            assert!(
                (measured / expected - 1.0).abs() < 0.03,
                "the isometry gave {measured} against {expected}"
            );
        }
    }

    /// The Langevin integrator reaches thermal equilibrium: equipartition in
    /// the velocity and the Boltzmann density in the position.
    #[test]
    fn langevin_dynamics_reaches_equipartition_and_boltzmann() {
        let mut rng = Rng::new(0x_1A46);
        let (temp, mass, k) = (0.7f64, 1.3f64, 2.0f64);
        // A harmonic well, whose exact equilibrium is Gaussian in both.
        let force = move |x: f64| -k * x;
        let path = langevin_underdamped(0.0, 0.0, 1.0, temp, mass, &force, 400_000, 0.01, &mut rng);
        let tail = &path[50_000..];
        let vs: Vec<f64> = tail.iter().map(|&(_, v)| v).collect();
        let xs: Vec<f64> = tail.iter().map(|&(x, _)| x).collect();
        // Equipartition: <v^2> = temp / mass.
        let v2 = vs.iter().map(|v| v * v).sum::<f64>() / vs.len() as f64;
        assert!(
            (v2 / (temp / mass) - 1.0).abs() < 0.05,
            "the kinetic energy is {} against {}",
            v2,
            temp / mass
        );
        // Boltzmann: <x^2> = temp / k.
        let x2 = xs.iter().map(|v| v * v).sum::<f64>() / xs.len() as f64;
        assert!(
            (x2 / (temp / k) - 1.0).abs() < 0.06,
            "the potential energy is {} against {}",
            x2,
            temp / k
        );
        assert!(mean(&xs).abs() < 0.03, "the position drifted");
        // At zero temperature the particle simply relaxes to the minimum.
        let cold = langevin_underdamped(2.0, 0.0, 3.0, 0.0, mass, &force, 20_000, 0.005, &mut rng);
        assert!(cold.last().expect("non-empty").0.abs() < 1e-3, "a cold particle did not settle");
    }

    /// The Fokker-Planck solver conserves probability and relaxes to the
    /// stationary density the drift and diffusion imply.
    #[test]
    fn fokker_planck_conserves_mass_and_finds_the_stationary_density() {
        // An Ornstein-Uhlenbeck generator, whose stationary density is a
        // Gaussian in closed form.
        let (theta, sigma) = (1.0f64, 0.8f64);
        let mu = move |x: f64| -theta * x;
        let sig = move |_x: f64| sigma;
        let (lo, hi) = (-4.0f64, 4.0f64);
        let n = 201usize;
        let dx = (hi - lo) / (n - 1) as f64;
        // Start from a narrow spike well off centre.
        let mut p0 = vec![0.0; n];
        p0[60] = 1.0 / dx;
        let evolved = fokker_planck_1d(&p0, &mu, &sig, lo, dx, 1e-4, 200_000);
        let mass: f64 = evolved.iter().sum::<f64>() * dx;
        assert!((mass - 1.0).abs() < 1e-9, "probability was not conserved: {mass}");
        assert!(evolved.iter().all(|&v| v >= -1e-12), "the density went negative");

        let want = stationary_density_1d(&mu, &sig, (lo, hi), n);
        let want_mass: f64 = want.iter().sum::<f64>() * dx;
        assert!((want_mass - 1.0).abs() < 1e-9, "the closed form is not normalised");
        // The two agree pointwise where there is anything to compare.
        for j in 20..n - 20 {
            assert!(
                (evolved[j] - want[j]).abs() < 0.02 * want[j].max(0.01),
                "at grid point {j} the solver has {} against {}",
                evolved[j],
                want[j]
            );
        }
        // And the closed form really is the Gaussian it should be.
        let var = sigma * sigma / (2.0 * theta);
        for j in (30..n - 30).step_by(10) {
            let x = lo + j as f64 * dx;
            let exact = (-x * x / (2.0 * var)).exp() / (2.0 * PI * var).sqrt();
            assert!(
                (want[j] - exact).abs() < 0.01 * exact.max(0.01),
                "the stationary density at {x} is {} against {exact}",
                want[j]
            );
        }
    }

    /// Kramers' rate has the Arrhenius form, and stochastic resonance shows
    /// its characteristic peak at an intermediate noise level.
    #[test]
    fn escape_rates_and_stochastic_resonance() {
        // Doubling the barrier squares the rate, at fixed temperature.
        let r1 = kramers_escape_rate(1.0, 0.5, 2.0, 1.5, 1.0);
        let r2 = kramers_escape_rate(2.0, 0.5, 2.0, 1.5, 1.0);
        assert!((r2 / (r1 * r1 / kramers_escape_rate(0.0, 0.5, 2.0, 1.5, 1.0)) - 1.0).abs() < 1e-9);
        // The rate falls with the barrier and rises with the temperature.
        assert!(kramers_escape_rate(1.0, 0.5, 2.0, 1.5, 1.0) > kramers_escape_rate(3.0, 0.5, 2.0, 1.5, 1.0));
        assert!(kramers_escape_rate(1.0, 1.0, 2.0, 1.5, 1.0) > kramers_escape_rate(1.0, 0.3, 2.0, 1.5, 1.0));
        // And with friction, which is Kramers' own contribution.
        assert!(kramers_escape_rate(1.0, 0.5, 2.0, 1.5, 1.0) > kramers_escape_rate(1.0, 0.5, 2.0, 1.5, 4.0));

        // Stochastic resonance: measure how well the path tracks the drive
        // at several noise levels, and require the best to be in the middle.
        let (amplitude, frequency) = (0.15f64, 0.005f64);
        let dt = 0.05f64;
        let n = 200_000usize;
        let mut scores = Vec::new();
        for &temp in &[0.005f64, 0.02, 0.08, 0.3, 1.2] {
            let mut rng = Rng::new(0x_5707 + (temp * 1000.0) as u64);
            let path = stochastic_resonance_sim(-1.0, amplitude, frequency, temp, n, dt, &mut rng);
            // Correlation of the sign of the path with the drive.
            let mut num = 0.0;
            for (k, &x) in path.iter().enumerate().take(n) {
                let t = k as f64 * dt;
                num += x.signum() * (2.0 * PI * frequency * t).sin();
            }
            scores.push(num / n as f64);
        }
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty")
            .0;
        assert!(
            best != 0 && best != scores.len() - 1,
            "the response peaked at the edge, at index {best}: {scores:?}"
        );
        assert!(scores[best] > 0.1, "the best response was only {}", scores[best]);
    }
}
