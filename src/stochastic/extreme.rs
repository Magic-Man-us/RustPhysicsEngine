//! Extreme value theory and copulas: the distribution of maxima, the
//! distribution of exceedances, and the dependence structure between them.
//!
//! Ordinary statistics describes the middle of a distribution, where there is
//! data. Extreme value theory describes the edge, where by construction there
//! is almost none, and it does so by an argument that parallels the central
//! limit theorem. Just as a normalised *sum* of independent variables has
//! only one possible limit whatever the summands, a normalised *maximum* has
//! only three -- Gumbel, Frechet, Weibull -- and the generalised extreme
//! value family holds all three, distinguished by the sign of a single shape
//! parameter. That is what licenses extrapolating past the largest
//! observation: the tail shape is not assumed, it is forced.
//!
//! Two routes lead to the same place. Taking the maximum of each block and
//! fitting a GEV throws away every observation but one per block. Taking
//! every exceedance over a high threshold instead keeps far more of the data,
//! and the Pickands-Balkema-de Haan theorem says those exceedances follow a
//! generalised Pareto distribution with the *same* shape parameter. The
//! threshold approach is usually the better estimator; the block approach is
//! easier to explain and needs no threshold chosen.
//!
//! The shape parameter is the whole story. Negative means a bounded tail with
//! a finite upper endpoint; zero means an exponential tail, where every
//! moment exists; positive means a power-law tail, where moments beyond
//! `1/xi` do not. A hundred-year return level computed under the wrong sign
//! is not slightly wrong.
//!
//! Copulas answer the other half of the question. Marginal tails say how
//! extreme each variable gets; a copula says whether they get extreme
//! together. The distinction matters because correlation does not capture it:
//! a Gaussian copula has zero tail dependence at any correlation below one,
//! so two variables can be strongly correlated in the body and yet
//! asymptotically independent in the tail, which is precisely the failure
//! mode a correlation-based risk model cannot see.

use crate::error::GeomError;
use crate::linalg::cholesky::cholesky;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;
use crate::statistics::distributions::{Distribution, Normal, StudentT};

/// Below this the shape parameter is treated as zero and the limiting
/// Gumbel form is used, since the general form divides by it.
const SHAPE_TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// The generalised extreme value distribution
// ---------------------------------------------------------------------------

/// `1 + xi (x - mu) / sigma`, the quantity that must stay positive for `x` to
/// be inside the GEV support.
fn gev_reduced(x: f64, mu: f64, sigma: f64, xi: f64) -> f64 {
    1.0 + xi * (x - mu) / sigma
}

/// The generalised extreme value density.
///
/// Outside the support -- above the upper endpoint when `xi < 0`, below the
/// lower one when `xi > 0` -- the density is zero.
///
/// # Panics
/// Panics unless `sigma` is positive.
#[must_use]
pub fn gev_pdf(x: f64, mu: f64, sigma: f64, xi: f64) -> f64 {
    assert!(sigma > 0.0, "gev_pdf requires a positive scale");
    let z = (x - mu) / sigma;
    if xi.abs() < SHAPE_TOL {
        // Gumbel: the xi -> 0 limit, where the support is the whole line.
        return (-z - (-z).exp()).exp() / sigma;
    }
    let s = gev_reduced(x, mu, sigma, xi);
    if s <= 0.0 {
        return 0.0;
    }
    let t = s.powf(-1.0 / xi);
    t.powf(xi + 1.0) * (-t).exp() / sigma
}

/// The generalised extreme value distribution function,
/// `exp(-[1 + xi (x - mu)/sigma]^(-1/xi))`.
///
/// # Panics
/// Panics unless `sigma` is positive.
#[must_use]
pub fn gev_cdf(x: f64, mu: f64, sigma: f64, xi: f64) -> f64 {
    assert!(sigma > 0.0, "gev_cdf requires a positive scale");
    let z = (x - mu) / sigma;
    if xi.abs() < SHAPE_TOL {
        return (-(-z).exp()).exp();
    }
    let s = gev_reduced(x, mu, sigma, xi);
    if s <= 0.0 {
        // Below the lower endpoint for a heavy tail, above the upper one for
        // a bounded one.
        return if xi > 0.0 { 0.0 } else { 1.0 };
    }
    (-s.powf(-1.0 / xi)).exp()
}

/// The GEV quantile at probability `p`.
///
/// `mu + (sigma / xi) [(-ln p)^(-xi) - 1]`, or the Gumbel form
/// `mu - sigma ln(-ln p)` when the shape vanishes.
///
/// # Panics
/// Panics unless `sigma` is positive and `p` lies strictly in `(0, 1)`.
#[must_use]
pub fn gev_quantile(p: f64, mu: f64, sigma: f64, xi: f64) -> f64 {
    assert!(sigma > 0.0, "gev_quantile requires a positive scale");
    assert!(p > 0.0 && p < 1.0, "gev_quantile requires p in (0, 1)");
    let y = -p.ln();
    if xi.abs() < SHAPE_TOL {
        mu - sigma * y.ln()
    } else {
        mu + sigma * (y.powf(-xi) - 1.0) / xi
    }
}

/// Negative log-likelihood of a GEV fit, infinite where any observation falls
/// outside the implied support.
fn gev_nll(data: &[f64], mu: f64, sigma: f64, xi: f64) -> f64 {
    if !(sigma > 0.0) || !sigma.is_finite() {
        return f64::MAX;
    }
    let n = data.len() as f64;
    if xi.abs() < SHAPE_TOL {
        let mut acc = n * sigma.ln();
        for &x in data {
            let z = (x - mu) / sigma;
            acc += z + (-z).exp();
        }
        return if acc.is_finite() { acc } else { f64::MAX };
    }
    let mut acc = n * sigma.ln();
    for &x in data {
        let s = gev_reduced(x, mu, sigma, xi);
        // An observation outside the support has zero density, so the
        // likelihood is zero and the negative log-likelihood infinite. This
        // is what confines the optimiser to feasible parameters.
        if s <= 1e-300 {
            return f64::MAX;
        }
        acc += (1.0 + 1.0 / xi) * s.ln() + s.powf(-1.0 / xi);
    }
    if acc.is_finite() {
        acc
    } else {
        f64::MAX
    }
}

/// Fits a GEV to block maxima by maximum likelihood, returning
/// `(location, scale, shape)`.
///
/// The scale is optimised on the log scale so it cannot go negative, and the
/// likelihood is infinite wherever an observation would fall outside the
/// support, which keeps the search inside the feasible region without an
/// explicit constraint. Started from the moment-matched Gumbel fit, which is
/// the shape-zero member of the family and a reliable neighbourhood to
/// descend from.
///
/// # Errors
/// Returns an error for fewer than ten observations, or if no feasible
/// parameter set is found.
pub fn gev_fit(maxima: &[f64]) -> Result<(f64, f64, f64), GeomError> {
    if maxima.len() < 10 {
        return Err(GeomError::InvalidArgument("gev_fit requires at least ten maxima"));
    }
    let (g_mu, g_sigma) = gumbel_moment_start(maxima)?;
    let objective = |p: &[f64]| -> f64 {
        gev_nll(maxima, p[0], p[1].clamp(-40.0, 40.0).exp(), p[2])
    };
    let start = [g_mu, g_sigma.ln(), 0.05];
    let best = crate::optimization::nelder_mead(&objective, &start, 0.2, 1e-12, 4000);
    let (mu, sigma, xi) = (best[0], best[1].clamp(-40.0, 40.0).exp(), best[2]);
    if !mu.is_finite() || !(sigma > 0.0) || !xi.is_finite() {
        return Err(GeomError::Degenerate("gev_fit: the optimiser produced no fit"));
    }
    if gev_nll(maxima, mu, sigma, xi) >= f64::MAX {
        return Err(GeomError::Degenerate("gev_fit: no feasible parameters found"));
    }
    Ok((mu, sigma, xi))
}

/// Moment-matched Gumbel parameters, used as a starting point.
///
/// A Gumbel has variance `pi^2 sigma^2 / 6` and mean `mu + gamma sigma`, so
/// both parameters follow from the sample mean and standard deviation.
fn gumbel_moment_start(data: &[f64]) -> Result<(f64, f64), GeomError> {
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    if !(var > 0.0) {
        return Err(GeomError::Degenerate("the sample has no variation"));
    }
    let sigma = (6.0 * var).sqrt() / std::f64::consts::PI;
    const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
    Ok((mean - EULER_MASCHERONI * sigma, sigma))
}

/// Fits a Gumbel distribution -- the GEV with shape fixed at zero -- by
/// maximum likelihood, returning `(location, scale)`.
///
/// Worth fitting separately rather than reading off a GEV fit: with the shape
/// pinned, the two remaining parameters are far better determined, and the
/// difference in log-likelihood against the free-shape fit is the natural
/// test of whether the tail is exponential.
///
/// # Errors
/// Returns an error for fewer than five observations or a constant sample.
pub fn gumbel_fit(maxima: &[f64]) -> Result<(f64, f64), GeomError> {
    if maxima.len() < 5 {
        return Err(GeomError::InvalidArgument("gumbel_fit requires at least five maxima"));
    }
    let (mu0, sigma0) = gumbel_moment_start(maxima)?;
    let objective =
        |p: &[f64]| -> f64 { gev_nll(maxima, p[0], p[1].clamp(-40.0, 40.0).exp(), 0.0) };
    let best = crate::optimization::nelder_mead(&objective, &[mu0, sigma0.ln()], 0.2, 1e-12, 3000);
    let sigma = best[1].clamp(-40.0, 40.0).exp();
    if !best[0].is_finite() || !(sigma > 0.0) {
        return Err(GeomError::Degenerate("gumbel_fit: the optimiser produced no fit"));
    }
    Ok((best[0], sigma))
}

// ---------------------------------------------------------------------------
// The generalised Pareto distribution
// ---------------------------------------------------------------------------

/// The generalised Pareto density for an exceedance `y > 0`.
///
/// # Panics
/// Panics unless `sigma` is positive.
#[must_use]
pub fn gpd_pdf(y: f64, sigma: f64, xi: f64) -> f64 {
    assert!(sigma > 0.0, "gpd_pdf requires a positive scale");
    if y < 0.0 {
        return 0.0;
    }
    if xi.abs() < SHAPE_TOL {
        return (-y / sigma).exp() / sigma;
    }
    let s = 1.0 + xi * y / sigma;
    if s <= 0.0 {
        return 0.0;
    }
    s.powf(-1.0 / xi - 1.0) / sigma
}

/// The generalised Pareto distribution function,
/// `1 - (1 + xi y / sigma)^(-1/xi)`.
///
/// # Panics
/// Panics unless `sigma` is positive.
#[must_use]
pub fn gpd_cdf(y: f64, sigma: f64, xi: f64) -> f64 {
    assert!(sigma > 0.0, "gpd_cdf requires a positive scale");
    if y <= 0.0 {
        return 0.0;
    }
    if xi.abs() < SHAPE_TOL {
        return 1.0 - (-y / sigma).exp();
    }
    let s = 1.0 + xi * y / sigma;
    if s <= 0.0 {
        // Past the finite upper endpoint of a bounded tail.
        return 1.0;
    }
    1.0 - s.powf(-1.0 / xi)
}

/// The generalised Pareto quantile at probability `p`.
///
/// # Panics
/// Panics unless `sigma` is positive and `p` lies in `[0, 1)`.
#[must_use]
pub fn gpd_quantile(p: f64, sigma: f64, xi: f64) -> f64 {
    assert!(sigma > 0.0, "gpd_quantile requires a positive scale");
    assert!((0.0..1.0).contains(&p), "gpd_quantile requires p in [0, 1)");
    if xi.abs() < SHAPE_TOL {
        -sigma * (1.0 - p).ln()
    } else {
        sigma * ((1.0 - p).powf(-xi) - 1.0) / xi
    }
}

/// Fits a generalised Pareto distribution to threshold exceedances by
/// maximum likelihood, returning `(scale, shape)`.
///
/// The exceedances must already be measured from the threshold, so they are
/// all positive. This is the peaks-over-threshold half of the theory: by
/// Pickands-Balkema-de Haan the shape here is the same shape a GEV fit to
/// block maxima of the same data would find, but estimated from every large
/// observation rather than one per block.
///
/// # Errors
/// Returns an error for fewer than ten exceedances, a non-positive
/// exceedance, or a failure to find feasible parameters.
pub fn gpd_fit(exceedances: &[f64]) -> Result<(f64, f64), GeomError> {
    if exceedances.len() < 10 {
        return Err(GeomError::InvalidArgument("gpd_fit requires at least ten exceedances"));
    }
    if exceedances.iter().any(|&y| !(y > 0.0)) {
        return Err(GeomError::InvalidArgument("gpd_fit requires positive exceedances"));
    }
    let n = exceedances.len() as f64;
    let mean: f64 = exceedances.iter().sum::<f64>() / n;

    let nll = |sigma: f64, xi: f64| -> f64 {
        if !(sigma > 0.0) || !sigma.is_finite() {
            return f64::MAX;
        }
        if xi.abs() < SHAPE_TOL {
            let acc = n * sigma.ln() + exceedances.iter().sum::<f64>() / sigma;
            return if acc.is_finite() { acc } else { f64::MAX };
        }
        let mut acc = n * sigma.ln();
        for &y in exceedances {
            let s = 1.0 + xi * y / sigma;
            if s <= 1e-300 {
                return f64::MAX;
            }
            acc += (1.0 / xi + 1.0) * s.ln();
        }
        if acc.is_finite() {
            acc
        } else {
            f64::MAX
        }
    };

    let objective = |p: &[f64]| -> f64 { nll(p[0].clamp(-40.0, 40.0).exp(), p[1]) };
    // The exponential fit is the shape-zero member and always feasible.
    let best = crate::optimization::nelder_mead(&objective, &[mean.ln(), 0.05], 0.2, 1e-12, 4000);
    let (sigma, xi) = (best[0].clamp(-40.0, 40.0).exp(), best[1]);
    if !(sigma > 0.0) || !xi.is_finite() || nll(sigma, xi) >= f64::MAX {
        return Err(GeomError::Degenerate("gpd_fit: no feasible parameters found"));
    }
    Ok((sigma, xi))
}

/// The mean excess over each threshold: the average of `x - u` across the
/// observations that exceed `u`.
///
/// The standard threshold-selection diagnostic. If the exceedances over some
/// `u` follow a generalised Pareto, the mean excess above any higher
/// threshold is `(sigma + xi u) / (1 - xi)` -- *linear* in the threshold. So
/// the point above which the plot straightens is the point above which the
/// asymptotic theory has taken hold, and a slope of zero means an
/// exponential tail.
///
/// A threshold exceeded by nothing yields NaN, which is reported rather than
/// silently dropped.
#[must_use]
pub fn mean_residual_life(x: &[f64], thresholds: &[f64]) -> Vec<f64> {
    thresholds
        .iter()
        .map(|&u| {
            let excesses: Vec<f64> = x.iter().filter(|&&v| v > u).map(|&v| v - u).collect();
            if excesses.is_empty() {
                f64::NAN
            } else {
                excesses.iter().sum::<f64>() / excesses.len() as f64
            }
        })
        .collect()
}

/// The Hill estimator of the tail index from the `k` largest observations.
///
/// `(1/k) sum_{i=1}^{k} ln X_(i) - ln X_(k+1)`, where `X_(1)` is the largest.
/// Estimates `xi` for a heavy tail, and only for a heavy one: the derivation
/// assumes a regularly varying tail, so a negative or zero shape is outside
/// its scope and the estimator will still return a positive number there.
///
/// Choosing `k` is the usual bias-variance trade: too small and the estimate
/// is noisy, too large and observations from the body contaminate it.
///
/// # Panics
/// Panics unless `1 <= k < n` and all of the top `k + 1` observations are
/// positive.
#[must_use]
pub fn hill_estimator(x: &[f64], k: usize) -> f64 {
    assert!(k >= 1, "hill_estimator requires k >= 1");
    assert!(k < x.len(), "hill_estimator requires k < n");
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    assert!(
        sorted[k] > 0.0,
        "hill_estimator requires the top k + 1 observations to be positive"
    );
    let anchor = sorted[k].ln();
    (0..k).map(|i| sorted[i].ln() - anchor).sum::<f64>() / k as f64
}

/// The level exceeded on average once every `period` blocks, under a GEV fit.
///
/// The quantile at `1 - 1/period`. A hundred-year level is not the largest
/// value seen in a century; it is the level with a one-in-a-hundred chance of
/// being exceeded in any given year.
///
/// # Panics
/// Panics unless `sigma` is positive and `period` exceeds one.
#[must_use]
pub fn return_level(mu: f64, sigma: f64, xi: f64, period: f64) -> f64 {
    assert!(period > 1.0, "return_level requires a period above one");
    gev_quantile(1.0 - 1.0 / period, mu, sigma, xi)
}

/// The average number of blocks between exceedances of `level`, the exact
/// inverse of [`return_level`].
///
/// Infinite for a level at or above the finite upper endpoint of a bounded
/// tail, which is the honest answer: such a level is never exceeded.
///
/// # Panics
/// Panics unless `sigma` is positive.
#[must_use]
pub fn return_period(mu: f64, sigma: f64, xi: f64, level: f64) -> f64 {
    let p = gev_cdf(level, mu, sigma, xi);
    if p >= 1.0 {
        f64::INFINITY
    } else {
        1.0 / (1.0 - p)
    }
}

/// The maximum of each consecutive block of `block` observations.
///
/// A trailing partial block is dropped: its maximum is drawn from fewer
/// observations and is not comparable with the rest, and including it biases
/// the fit downward.
///
/// # Panics
/// Panics if `block` is zero.
#[must_use]
pub fn block_maxima(x: &[f64], block: usize) -> Vec<f64> {
    assert!(block > 0, "block_maxima requires a positive block size");
    x.chunks_exact(block)
        .map(|c| c.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .collect()
}

/// The extremal index by the Ferro-Segers intervals estimator.
///
/// Roughly the reciprocal of the mean cluster size: 1 when exceedances arrive
/// independently, below 1 when they arrive in bursts. It matters because
/// clustering does not change *how many* exceedances there are but does
/// change how many *distinct events* they represent, and a return period
/// computed as though every exceedance were its own event overstates the
/// frequency by exactly this factor.
///
/// The intervals estimator works from the gaps between exceedances rather
/// than from a declustering rule, so it needs no run length chosen.
///
/// Returns 1 when there are too few exceedances to say anything.
///
/// # Panics
/// Panics if `x` is empty.
#[must_use]
pub fn extremal_index(x: &[f64], threshold: f64) -> f64 {
    assert!(!x.is_empty(), "extremal_index requires observations");
    let positions: Vec<usize> =
        x.iter().enumerate().filter(|(_, &v)| v > threshold).map(|(i, _)| i).collect();
    let n = positions.len();
    if n < 3 {
        return 1.0;
    }
    let gaps: Vec<f64> =
        positions.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let count = (n - 1) as f64;
    let max_gap = gaps.iter().copied().fold(0.0f64, f64::max);

    // Ferro and Segers give two forms. The second is used once any gap
    // exceeds two, where subtracting one from each gap removes the bias that
    // arises because an interexceedance time is at least one by construction.
    let theta = if max_gap <= 2.0 {
        let s: f64 = gaps.iter().sum();
        let ss: f64 = gaps.iter().map(|t| t * t).sum();
        if ss <= 0.0 {
            return 1.0;
        }
        2.0 * s * s / (count * ss)
    } else {
        let s: f64 = gaps.iter().map(|t| t - 1.0).sum();
        let ss: f64 = gaps.iter().map(|t| (t - 1.0) * (t - 2.0)).sum();
        if ss <= 0.0 {
            return 1.0;
        }
        2.0 * s * s / (count * ss)
    };
    theta.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Rank correlation
// ---------------------------------------------------------------------------

/// Kendall's tau: the probability of concordance minus the probability of
/// discordance, estimated over all pairs.
///
/// Ties in either coordinate contribute nothing to either count. Unlike
/// Pearson correlation this depends only on the ranks, so it is invariant
/// under any increasing transformation of either variable -- which is exactly
/// what makes it a property of the copula rather than of the margins, and
/// what lets a copula parameter be recovered from it.
///
/// # Panics
/// Panics unless the series have equal length and at least two points.
#[must_use]
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    assert!(x.len() == y.len(), "kendall_tau requires equal lengths");
    assert!(x.len() >= 2, "kendall_tau requires at least two points");
    let n = x.len();
    let pairs = (n * (n - 1) / 2) as i64;
    if pairs <= 0 {
        return 0.0;
    }

    // Knight's algorithm. Comparing every pair directly is the definition but
    // costs O(n^2), which is minutes rather than milliseconds on the sample
    // sizes a copula fit wants. Sorting by x and counting inversions in the
    // resulting y sequence gives the same discordance count in O(n log n),
    // because a pair is discordant exactly when it is out of order in y once
    // ordered by x.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Pairs tied in x, in y, and in both. Ties are concordant with nothing and
    // discordant with nothing, so they are removed from the comparable total.
    let tied = |v: &[f64]| -> i64 {
        let mut sorted = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut acc = 0i64;
        let mut run = 1i64;
        for i in 1..sorted.len() {
            if sorted[i] == sorted[i - 1] {
                run += 1;
            } else {
                acc += run * (run - 1) / 2;
                run = 1;
            }
        }
        acc + run * (run - 1) / 2
    };
    let tied_x = tied(x);
    let tied_y = tied(y);
    // Pairs tied in both coordinates: runs within the jointly sorted order.
    let mut tied_both = 0i64;
    let mut run = 1i64;
    for i in 1..n {
        let (a, b) = (order[i], order[i - 1]);
        if x[a] == x[b] && y[a] == y[b] {
            run += 1;
        } else {
            tied_both += run * (run - 1) / 2;
            run = 1;
        }
    }
    tied_both += run * (run - 1) / 2;

    let sequence: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let discordant = count_inversions(&sequence);
    let comparable = pairs - tied_x - tied_y + tied_both;
    (comparable - 2 * discordant) as f64 / pairs as f64
}

/// The number of strictly out-of-order pairs in `v`, by merge sort.
///
/// Equal neighbours are not inversions, which is what makes the count equal
/// the number of strictly discordant pairs rather than merely the
/// non-concordant ones.
fn count_inversions(v: &[f64]) -> i64 {
    let mut work = v.to_vec();
    let mut buffer = work.clone();
    merge_count(&mut work, &mut buffer, 0, v.len())
}

fn merge_count(v: &mut [f64], buffer: &mut [f64], lo: usize, hi: usize) -> i64 {
    if hi - lo < 2 {
        return 0;
    }
    let mid = lo + (hi - lo) / 2;
    let mut count = merge_count(v, buffer, lo, mid) + merge_count(v, buffer, mid, hi);
    let (mut i, mut j, mut k) = (lo, mid, lo);
    while i < mid && j < hi {
        // Take from the left while it is not strictly greater, so equal
        // values never register as an inversion.
        if v[i] <= v[j] {
            buffer[k] = v[i];
            i += 1;
        } else {
            // Every remaining element on the left is greater than v[j].
            count += (mid - i) as i64;
            buffer[k] = v[j];
            j += 1;
        }
        k += 1;
    }
    while i < mid {
        buffer[k] = v[i];
        i += 1;
        k += 1;
    }
    while j < hi {
        buffer[k] = v[j];
        j += 1;
        k += 1;
    }
    v[lo..hi].copy_from_slice(&buffer[lo..hi]);
    count
}

/// Ranks with ties averaged, one-based.
fn ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        // A run of equal values shares the average of the ranks it spans.
        let average = ((i + j) as f64) / 2.0 + 1.0;
        for &k in &order[i..=j] {
            out[k] = average;
        }
        i = j + 1;
    }
    out
}

/// Spearman's rho: Pearson correlation applied to the ranks.
///
/// Like Kendall's tau it is a function of the copula alone, but it weights
/// the whole distribution more evenly, so the two disagree in a way that is
/// itself informative about the shape of the dependence.
///
/// # Panics
/// Panics unless the series have equal length and at least two points.
#[must_use]
pub fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    assert!(x.len() == y.len(), "spearman_rho requires equal lengths");
    assert!(x.len() >= 2, "spearman_rho requires at least two points");
    let (rx, ry) = (ranks(x), ranks(y));
    let n = x.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for i in 0..x.len() {
        let (a, b) = (rx[i] - mx, ry[i] - my);
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    if dx <= 0.0 || dy <= 0.0 {
        0.0
    } else {
        num / (dx * dy).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Copulas
// ---------------------------------------------------------------------------

/// The Archimedean and elliptical families supported here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopulaFamily {
    /// The dependence of a multivariate normal. No tail dependence at any
    /// correlation below one.
    Gaussian,
    /// Lower tail dependence, upper tail independence.
    Clayton,
    /// Upper tail dependence, lower tail independence.
    Gumbel,
    /// Symmetric, with no tail dependence in either direction.
    Frank,
}

/// Samples `n` points from a Gaussian copula with the given correlation
/// matrix.
///
/// Draws from a multivariate normal by a Cholesky factor and maps each margin
/// through the standard normal distribution function, which is what leaves
/// uniform margins and keeps only the dependence.
///
/// # Errors
/// Returns an error if the matrix is not a valid correlation matrix -- not
/// square, not symmetric, or not positive definite.
pub fn copula_gaussian_sample(
    corr: &Matrix,
    n: usize,
    rng: &mut Rng,
) -> Result<Vec<Vec<f64>>, GeomError> {
    let l = correlation_factor(corr)?;
    let d = corr.rows;
    let normal = Normal::new(0.0, 1.0);
    Ok((0..n)
        .map(|_| {
            let z: Vec<f64> = (0..d).map(|_| rng.next_gaussian()).collect();
            (0..d)
                .map(|i| {
                    let v: f64 = (0..=i).map(|j| l.get(i, j) * z[j]).sum();
                    normal.cdf(v)
                })
                .collect()
        })
        .collect())
}

/// Samples `n` points from a `t` copula with `df` degrees of freedom.
///
/// The same construction as the Gaussian copula but with a shared chi-squared
/// scaling across all coordinates. That single shared factor is what creates
/// tail dependence: occasionally it is small, every coordinate is inflated at
/// once, and the sample lands in a corner. The Gaussian copula has no such
/// mechanism, which is why its tail dependence is exactly zero.
///
/// # Errors
/// Returns an error for an invalid correlation matrix or `df` below one.
pub fn copula_t_sample(
    corr: &Matrix,
    df: f64,
    n: usize,
    rng: &mut Rng,
) -> Result<Vec<Vec<f64>>, GeomError> {
    if !(df >= 1.0) {
        return Err(GeomError::InvalidArgument("copula_t_sample requires df >= 1"));
    }
    let l = correlation_factor(corr)?;
    let d = corr.rows;
    let t = StudentT::new(df);
    let k = df.round().max(1.0) as usize;
    Ok((0..n)
        .map(|_| {
            let z: Vec<f64> = (0..d).map(|_| rng.next_gaussian()).collect();
            // Chi-squared with k degrees of freedom as a sum of squares.
            let chi: f64 = (0..k).map(|_| rng.next_gaussian().powi(2)).sum();
            let scale = (df / chi.max(1e-300)).sqrt();
            (0..d)
                .map(|i| {
                    let v: f64 = (0..=i).map(|j| l.get(i, j) * z[j]).sum();
                    t.cdf(v * scale)
                })
                .collect()
        })
        .collect())
}

/// Validates a correlation matrix and returns its Cholesky factor.
fn correlation_factor(corr: &Matrix) -> Result<Matrix, GeomError> {
    if !corr.is_square() || corr.rows == 0 {
        return Err(GeomError::InvalidArgument("copula requires a square correlation matrix"));
    }
    if !corr.is_symmetric(1e-9) {
        return Err(GeomError::InvalidArgument("copula requires a symmetric correlation matrix"));
    }
    for i in 0..corr.rows {
        if (corr.get(i, i) - 1.0).abs() > 1e-9 {
            return Err(GeomError::InvalidArgument("a correlation matrix has unit diagonal"));
        }
    }
    cholesky(corr).map_err(|_| GeomError::Degenerate("the correlation matrix is not positive definite"))
}

/// Samples `n` pairs from a bivariate Clayton copula by conditional
/// inversion.
///
/// `theta > 0`. Clayton concentrates its dependence in the *lower* tail: its
/// coefficient of lower tail dependence is `2^(-1/theta)`, while the upper is
/// zero. That asymmetry is the reason to reach for it -- joint crashes
/// without joint booms.
///
/// # Panics
/// Panics unless `theta` is positive.
#[must_use]
pub fn copula_clayton(theta: f64, n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    assert!(theta > 0.0, "copula_clayton requires theta > 0");
    (0..n)
        .map(|_| {
            let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
            let w = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
            // Inverting the conditional distribution of V given U = u.
            let v = (u.powf(-theta) * (w.powf(-theta / (1.0 + theta)) - 1.0) + 1.0)
                .powf(-1.0 / theta);
            vec![u, v.clamp(0.0, 1.0)]
        })
        .collect()
}

/// Samples `n` pairs from a bivariate Gumbel copula.
///
/// `theta >= 1`. The mirror image of Clayton: upper tail dependence
/// `2 - 2^(1/theta)` and none in the lower tail.
///
/// The conditional distribution has no closed-form inverse, so this uses the
/// Marshall-Olkin frailty construction instead. The Gumbel generator is the
/// Laplace transform of a positive stable law, so drawing one such variate
/// and dividing two independent exponentials by it produces the copula
/// directly. The stable variate comes from Kanter's algorithm.
///
/// # Panics
/// Panics unless `theta >= 1`.
#[must_use]
pub fn copula_gumbel(theta: f64, n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    assert!(theta >= 1.0, "copula_gumbel requires theta >= 1");
    if (theta - 1.0).abs() < SHAPE_TOL {
        // theta = 1 is independence, and Kanter's formula degenerates there.
        return (0..n).map(|_| vec![rng.next_f64(), rng.next_f64()]).collect();
    }
    let alpha = 1.0 / theta;
    (0..n)
        .map(|_| {
            let s = positive_stable(alpha, rng);
            let e1 = -rng.next_f64().max(1e-300).ln();
            let e2 = -rng.next_f64().max(1e-300).ln();
            vec![
                (-(e1 / s).powf(alpha)).exp().clamp(0.0, 1.0),
                (-(e2 / s).powf(alpha)).exp().clamp(0.0, 1.0),
            ]
        })
        .collect()
}

/// A positive stable variate with Laplace transform `exp(-t^alpha)`, by
/// Kanter's algorithm.
fn positive_stable(alpha: f64, rng: &mut Rng) -> f64 {
    let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12) * std::f64::consts::PI;
    let w = -rng.next_f64().max(1e-300).ln();
    let a = (alpha * u).sin() / u.sin().powf(1.0 / alpha);
    let b = ((1.0 - alpha) * u).sin() / w;
    a * b.powf((1.0 - alpha) / alpha)
}

/// Samples `n` pairs from a bivariate Frank copula by conditional inversion.
///
/// `theta` may be any non-zero real: positive for positive dependence,
/// negative for negative. Frank is the symmetric Archimedean copula, with no
/// tail dependence in either direction -- useful precisely when dependence in
/// the body should not imply dependence in the extremes.
///
/// # Panics
/// Panics if `theta` is zero, where the family degenerates to independence.
#[must_use]
pub fn copula_frank(theta: f64, n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    assert!(theta.abs() > SHAPE_TOL, "copula_frank requires a non-zero theta");
    let a = (-theta).exp() - 1.0;
    (0..n)
        .map(|_| {
            let u = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
            let w = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
            let eu = (-theta * u).exp();
            // v = -(1/theta) ln[1 + w a / (e^{-theta u} - w (e^{-theta u} - 1))].
            let denominator = eu - w * (eu - 1.0);
            let v = -(1.0 + w * a / denominator).ln() / theta;
            vec![u, v.clamp(0.0, 1.0)]
        })
        .collect()
}

/// The Debye function of order one, `D_1(theta) = (1/theta) int_0^theta
/// t / (e^t - 1) dt`, which appears in Frank's Kendall tau.
fn debye1(theta: f64) -> f64 {
    if theta.abs() < 1e-8 {
        // The integrand tends to one at the origin.
        return 1.0 - theta / 4.0;
    }
    let steps = 4000usize;
    let h = theta / steps as f64;
    let mut acc = 0.0;
    for k in 0..steps {
        let t = (k as f64 + 0.5) * h;
        let d = t.exp() - 1.0;
        acc += if d.abs() < 1e-12 { 1.0 } else { t / d } * h;
    }
    acc / theta
}

/// Kendall's tau implied by a copula family at parameter `theta`.
///
/// Each family has a closed-form relation, which is what makes inversion
/// possible: `2 arcsin(rho) / pi` for the Gaussian, `theta / (theta + 2)` for
/// Clayton, `1 - 1/theta` for Gumbel, and for Frank
/// `1 - 4 (1 - D_1(theta)) / theta` with `D_1` the Debye function.
#[must_use]
pub fn copula_tau(family: CopulaFamily, theta: f64) -> f64 {
    match family {
        CopulaFamily::Gaussian => 2.0 * theta.clamp(-1.0, 1.0).asin() / std::f64::consts::PI,
        CopulaFamily::Clayton => theta / (theta + 2.0),
        CopulaFamily::Gumbel => 1.0 - 1.0 / theta,
        CopulaFamily::Frank => {
            if theta.abs() < 1e-8 {
                0.0
            } else {
                1.0 - 4.0 * (1.0 - debye1(theta)) / theta
            }
        }
    }
}

/// Fits a copula parameter by inverting Kendall's tau.
///
/// The method of moments applied to a rank statistic: measure tau from the
/// data, then solve the family's tau-theta relation for theta. It needs no
/// likelihood and no numerical optimisation for three of the four families,
/// and because tau depends only on the ranks the answer is unaffected by
/// whatever the margins happen to be -- which is the entire point of
/// separating a copula from its margins.
///
/// `data` holds one row per observation with two columns.
///
/// # Errors
/// Returns an error for the wrong shape, or for a sample tau outside the
/// range the family can represent -- Clayton and Gumbel model only positive
/// dependence, so a negative tau has no solution.
pub fn copula_fit_tau(data: &[Vec<f64>], family: CopulaFamily) -> Result<f64, GeomError> {
    if data.len() < 3 || data.iter().any(|r| r.len() != 2) {
        return Err(GeomError::InvalidArgument("copula_fit_tau requires at least three pairs"));
    }
    let x: Vec<f64> = data.iter().map(|r| r[0]).collect();
    let y: Vec<f64> = data.iter().map(|r| r[1]).collect();
    let tau = kendall_tau(&x, &y);

    match family {
        CopulaFamily::Gaussian => Ok((std::f64::consts::PI * tau / 2.0).sin()),
        CopulaFamily::Clayton => {
            if tau <= 0.0 || tau >= 1.0 {
                return Err(GeomError::InvalidArgument(
                    "Clayton represents only positive dependence",
                ));
            }
            Ok(2.0 * tau / (1.0 - tau))
        }
        CopulaFamily::Gumbel => {
            if tau <= 0.0 || tau >= 1.0 {
                return Err(GeomError::InvalidArgument(
                    "Gumbel represents only positive dependence",
                ));
            }
            Ok(1.0 / (1.0 - tau))
        }
        CopulaFamily::Frank => {
            if tau.abs() < 1e-9 {
                return Err(GeomError::InvalidArgument("Frank is undefined at zero dependence"));
            }
            // tau is strictly increasing in theta, so bisect.
            let (mut lo, mut hi) = if tau > 0.0 { (1e-6, 200.0) } else { (-200.0, -1e-6) };
            if (copula_tau(family, lo) - tau).signum() == (copula_tau(family, hi) - tau).signum() {
                return Err(GeomError::InvalidArgument("tau is outside Frank's range"));
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if (copula_tau(family, mid) - tau).signum()
                    == (copula_tau(family, lo) - tau).signum()
                {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Ok(0.5 * (lo + hi))
        }
    }
}

/// The pseudo-observations of a sample: each column replaced by its ranks
/// divided by `n + 1`.
///
/// This is the empirical copula transform. Dividing by `n + 1` rather than
/// `n` keeps every value strictly inside `(0, 1)`, which matters because the
/// copula densities and tail statistics below take logarithms of them.
/// Whatever the marginal distributions were, the result has approximately
/// uniform margins and retains exactly the original dependence.
///
/// # Errors
/// Returns an error for empty or ragged input.
pub fn empirical_copula(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GeomError> {
    if data.is_empty() {
        return Err(GeomError::Empty);
    }
    let d = data[0].len();
    if d == 0 || data.iter().any(|r| r.len() != d) {
        return Err(GeomError::InvalidArgument("empirical_copula requires rectangular data"));
    }
    let n = data.len();
    let scale = (n + 1) as f64;
    let columns: Vec<Vec<f64>> = (0..d)
        .map(|j| ranks(&data.iter().map(|r| r[j]).collect::<Vec<f64>>()))
        .collect();
    Ok((0..n).map(|i| (0..d).map(|j| columns[j][i] / scale).collect()).collect())
}

/// Empirical coefficients of `(lower, upper)` tail dependence at quantile
/// level `q`.
///
/// The lower coefficient estimates `P(V <= q | U <= q)` and the upper
/// `P(V > q | U > q)`, both computed on pseudo-observations so the margins
/// are irrelevant. Only one of the pair is informative at any given `q`: read
/// the lower coefficient at a small `q` and the upper at a `q` near one. At
/// `q = 0.01` the upper coefficient is the probability both variables exceed
/// their first percentile, which is close to one for any sample and says
/// nothing about the tail. As `q` approaches its limit these tend to the theoretical
/// coefficients: `2^(-1/theta)` and 0 for Clayton, 0 and `2 - 2^(1/theta)`
/// for Gumbel, and 0 for both under any Gaussian copula with correlation
/// below one.
///
/// The last of those is the practically important one. Two variables can have
/// a correlation of 0.9 and still, under a Gaussian copula, become
/// independent in the limit of extreme events.
///
/// # Errors
/// Returns an error for the wrong shape or a `q` outside `(0, 1)`.
pub fn tail_dependence_coefficient(
    data: &[Vec<f64>],
    q: f64,
) -> Result<(f64, f64), GeomError> {
    if !(q > 0.0 && q < 1.0) {
        return Err(GeomError::InvalidArgument("tail dependence requires q in (0, 1)"));
    }
    if data.len() < 4 || data.iter().any(|r| r.len() != 2) {
        return Err(GeomError::InvalidArgument("tail dependence requires at least four pairs"));
    }
    let pseudo = empirical_copula(data)?;
    let n = pseudo.len() as f64;

    let both_below = pseudo.iter().filter(|r| r[0] <= q && r[1] <= q).count() as f64 / n;
    let first_below = pseudo.iter().filter(|r| r[0] <= q).count() as f64 / n;
    let both_above = pseudo.iter().filter(|r| r[0] > q && r[1] > q).count() as f64 / n;
    let first_above = pseudo.iter().filter(|r| r[0] > q).count() as f64 / n;

    let lower = if first_below > 0.0 { both_below / first_below } else { 0.0 };
    let upper = if first_above > 0.0 { both_above / first_above } else { 0.0 };
    Ok((lower, upper))
}

/// The Pickands dependence function estimated at `t`, for a bivariate
/// extreme-value copula.
///
/// An extreme-value copula is determined entirely by a convex function `A` on
/// `[0, 1]` satisfying `max(t, 1-t) <= A(t) <= 1`. The two bounds are the two
/// extremes of dependence: `A == 1` is independence, and `A(t) = max(t, 1-t)`
/// is perfect dependence. Everything in between is a real dependence
/// structure, and `A` is the whole of it.
///
/// Estimated by Pickands' original construction, the reciprocal of the mean
/// of `min(xi/(1-t), eta/t)` over the transformed data.
///
/// # Errors
/// Returns an error for the wrong shape or a `t` outside `(0, 1)`.
pub fn pickands_dependence(data: &[Vec<f64>], t: f64) -> Result<f64, GeomError> {
    if !(t > 0.0 && t < 1.0) {
        return Err(GeomError::InvalidArgument("pickands_dependence requires t in (0, 1)"));
    }
    if data.len() < 4 || data.iter().any(|r| r.len() != 2) {
        return Err(GeomError::InvalidArgument("pickands_dependence requires at least four pairs"));
    }
    let pseudo = empirical_copula(data)?;
    let n = pseudo.len() as f64;
    let mut acc = 0.0;
    for row in &pseudo {
        // Unit Frechet-style transform: -ln u is standard exponential when u
        // is uniform, so these are the exponential scales Pickands works on.
        let xi = -row[0].ln();
        let eta = -row[1].ln();
        acc += (xi / (1.0 - t)).min(eta / t);
    }
    let mean = acc / n;
    if !(mean > 0.0) {
        return Err(GeomError::Degenerate("pickands_dependence: degenerate sample"));
    }
    // The estimator is not guaranteed to respect the bounds in a finite
    // sample, so clamp it into the region the function is defined on.
    Ok((1.0 / mean).clamp(t.max(1.0 - t), 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    /// Draws from a GEV by inverting its distribution function.
    fn gev_sample(n: usize, mu: f64, sigma: f64, xi: f64, rng: &mut Rng) -> Vec<f64> {
        (0..n)
            .map(|_| gev_quantile(rng.next_f64().clamp(1e-12, 1.0 - 1e-12), mu, sigma, xi))
            .collect()
    }

    /// Draws from a GPD by inverting its distribution function.
    fn gpd_sample(n: usize, sigma: f64, xi: f64, rng: &mut Rng) -> Vec<f64> {
        (0..n)
            .map(|_| gpd_quantile(rng.next_f64().clamp(0.0, 1.0 - 1e-12), sigma, xi))
            .collect()
    }

    // -----------------------------------------------------------------
    // The GEV family is a distribution family
    // -----------------------------------------------------------------

    #[test]
    fn the_gev_distribution_function_is_the_integral_of_its_density() {
        for &(mu, sigma, xi) in
            &[(0.0, 1.0, 0.0), (2.0, 1.5, 0.3), (-1.0, 0.7, -0.25), (0.0, 1.0, 0.8)]
        {
            // Integrate the density up to a point and compare with the
            // distribution function evaluated there.
            let lo = gev_quantile(1e-9, mu, sigma, xi);
            for &p in &[0.05f64, 0.25, 0.5, 0.9, 0.99] {
                let x = gev_quantile(p, mu, sigma, xi);
                let steps = 200_000usize;
                let h = (x - lo) / steps as f64;
                let mass: f64 =
                    (0..steps).map(|k| gev_pdf(lo + (k as f64 + 0.5) * h, mu, sigma, xi) * h).sum();
                assert!(
                    (mass - p).abs() < 1e-5,
                    "xi = {xi}, p = {p}: integrated {mass} against cdf {}",
                    gev_cdf(x, mu, sigma, xi)
                );
                // And the quantile really is the inverse of the cdf.
                assert!(
                    (gev_cdf(x, mu, sigma, xi) - p).abs() < 1e-12,
                    "xi = {xi}: the quantile and cdf disagree at p = {p}"
                );
            }
            assert!(gev_pdf(lo - 1e6, mu, sigma, xi) >= 0.0);
        }
    }

    #[test]
    fn the_sign_of_the_shape_decides_whether_the_tail_is_bounded() {
        // A negative shape gives a finite upper endpoint at mu - sigma/xi; a
        // positive one gives a power-law tail with no endpoint at all. This is
        // the single most consequential fact in the subject.
        let (mu, sigma, xi) = (0.0f64, 1.0f64, -0.5f64);
        let endpoint = mu - sigma / xi;
        assert!((endpoint - 2.0).abs() < 1e-12, "the endpoint is {endpoint}");
        assert_eq!(gev_cdf(endpoint + 1e-9, mu, sigma, xi), 1.0);
        assert_eq!(gev_pdf(endpoint + 1e-9, mu, sigma, xi), 0.0);
        assert!(gev_cdf(endpoint - 1e-6, mu, sigma, xi) < 1.0);
        // Nothing is ever exceeded past the endpoint.
        assert!(return_period(mu, sigma, xi, endpoint + 1.0).is_infinite());

        // Heavy tail: the survival function decays like a power, so a
        // thousand-year level is far beyond a hundred-year one.
        let heavy = (0.0, 1.0, 0.5);
        let hundred = return_level(heavy.0, heavy.1, heavy.2, 100.0);
        let thousand = return_level(heavy.0, heavy.1, heavy.2, 1000.0);
        assert!(thousand > 2.5 * hundred, "{thousand} is not far past {hundred}");
        // Light tail: the same ratio of periods buys much less.
        let light = (0.0, 1.0, 0.0);
        let l100 = return_level(light.0, light.1, light.2, 100.0);
        let l1000 = return_level(light.0, light.1, light.2, 1000.0);
        assert!(l1000 < 1.6 * l100, "an exponential tail grew too fast");
    }

    #[test]
    fn return_level_and_return_period_invert_each_other() {
        for &(mu, sigma, xi) in &[(10.0, 2.0, 0.0), (0.0, 1.0, 0.25), (5.0, 3.0, -0.2)] {
            for &period in &[2.0f64, 10.0, 50.0, 100.0, 500.0] {
                let level = return_level(mu, sigma, xi, period);
                let back = return_period(mu, sigma, xi, level);
                assert!(
                    close(back, period, 1e-9),
                    "xi = {xi}: period {period} became {back} through level {level}"
                );
            }
            // Longer periods mean higher levels, always.
            let levels: Vec<f64> =
                [2.0f64, 5.0, 20.0, 100.0].iter().map(|&t| return_level(mu, sigma, xi, t)).collect();
            assert!(levels.windows(2).all(|w| w[1] > w[0]), "return levels are not increasing");
        }
    }

    #[test]
    fn gev_fitting_recovers_the_parameters_it_sampled_from() {
        for &(mu, sigma, xi) in &[(0.0f64, 1.0f64, 0.0f64), (3.0, 2.0, 0.3), (1.0, 1.0, -0.25)] {
            let mut rng = Rng::new(0x06E7_0001 + (xi.abs() * 1000.0) as u64);
            let sample = gev_sample(4000, mu, sigma, xi, &mut rng);
            let (m, s, x) = gev_fit(&sample).unwrap();
            assert!((m - mu).abs() < 0.12, "location {m} against {mu}");
            assert!(close(s, sigma, 0.10), "scale {s} against {sigma}");
            assert!((x - xi).abs() < 0.08, "shape {x} against {xi}");
        }
    }

    #[test]
    fn a_gumbel_fit_is_the_shape_zero_member_of_the_gev_family() {
        let mut rng = Rng::new(0x06E7_0002);
        let sample = gev_sample(3000, 5.0, 2.0, 0.0, &mut rng);
        let (gm, gs) = gumbel_fit(&sample).unwrap();
        assert!((gm - 5.0).abs() < 0.15, "location {gm}");
        assert!(close(gs, 2.0, 0.08), "scale {gs}");

        // The free-shape fit should land near zero and cannot fit worse.
        let (fm, fs, fx) = gev_fit(&sample).unwrap();
        assert!(fx.abs() < 0.06, "the free shape came out {fx}");
        let restricted = gev_nll(&sample, gm, gs, 0.0);
        let free = gev_nll(&sample, fm, fs, fx);
        assert!(
            free <= restricted + 1e-6,
            "the free fit ({free}) was worse than the restricted one ({restricted})"
        );
        // Nesting means the extra parameter buys little on Gumbel data.
        assert!(restricted - free < 5.0, "the shape parameter bought {}", restricted - free);
    }

    #[test]
    fn gev_fitting_rejects_input_it_cannot_use() {
        assert!(gev_fit(&[1.0; 5]).is_err());
        assert!(gev_fit(&[3.0; 40]).is_err());
        assert!(gumbel_fit(&[1.0, 2.0]).is_err());
        assert!(gumbel_fit(&[7.0; 40]).is_err());
    }

    // -----------------------------------------------------------------
    // Peaks over threshold
    // -----------------------------------------------------------------

    #[test]
    fn the_generalised_pareto_is_a_distribution_and_inverts_its_own_quantile() {
        for &(sigma, xi) in &[(1.0, 0.0), (2.0, 0.4), (1.5, -0.3)] {
            for &p in &[0.1f64, 0.5, 0.9, 0.99] {
                let y = gpd_quantile(p, sigma, xi);
                assert!((gpd_cdf(y, sigma, xi) - p).abs() < 1e-12, "xi = {xi}, p = {p}");
            }
            // The density integrates to the distribution function.
            let top = gpd_quantile(0.999, sigma, xi);
            let steps = 400_000usize;
            let h = top / steps as f64;
            let mass: f64 = (0..steps).map(|k| gpd_pdf((k as f64 + 0.5) * h, sigma, xi) * h).sum();
            assert!((mass - 0.999).abs() < 1e-5, "xi = {xi} integrated to {mass}");
            assert_eq!(gpd_cdf(-1.0, sigma, xi), 0.0);
            assert_eq!(gpd_pdf(-1.0, sigma, xi), 0.0);
        }
        // A negative shape bounds the excess at -sigma/xi.
        let (sigma, xi) = (1.0, -0.5);
        assert_eq!(gpd_cdf(2.0 + 1e-9, sigma, xi), 1.0);
        assert_eq!(gpd_pdf(2.0 + 1e-9, sigma, xi), 0.0);
    }

    #[test]
    fn gpd_fitting_recovers_the_parameters_it_sampled_from() {
        for &(sigma, xi) in &[(1.0f64, 0.0f64), (2.0, 0.3), (1.0, -0.2)] {
            let mut rng = Rng::new(0x06D0_0001 + (xi.abs() * 1000.0) as u64);
            let sample = gpd_sample(5000, sigma, xi, &mut rng);
            let (s, x) = gpd_fit(&sample).unwrap();
            assert!(close(s, sigma, 0.10), "scale {s} against {sigma}");
            assert!((x - xi).abs() < 0.06, "shape {x} against {xi}");
        }
        assert!(gpd_fit(&[1.0; 5]).is_err());
        assert!(gpd_fit(&[1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]).is_err());
    }

    #[test]
    fn block_maxima_and_threshold_exceedances_find_the_same_shape() {
        // Pickands-Balkema-de Haan: the two routes into the tail estimate the
        // same shape parameter. This is the theorem that makes the
        // peaks-over-threshold approach worth preferring, since it uses far
        // more of the data to get there.
        let xi = 0.3f64;
        let mut rng = Rng::new(0xB07A_0011);
        // Pareto data with tail index 1/xi, so the extreme value shape is xi.
        let raw: Vec<f64> = (0..40_000)
            .map(|_| rng.next_f64().clamp(1e-12, 1.0 - 1e-12).powf(-xi))
            .collect();

        let maxima = block_maxima(&raw, 200);
        assert_eq!(maxima.len(), 200);
        let (_, _, block_shape) = gev_fit(&maxima).unwrap();

        let mut sorted = raw.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = sorted[raw.len() - 2000];
        let excesses: Vec<f64> =
            raw.iter().filter(|&&v| v > threshold).map(|&v| v - threshold).collect();
        let (_, pot_shape) = gpd_fit(&excesses).unwrap();

        assert!((block_shape - xi).abs() < 0.12, "the block route gave {block_shape}");
        assert!((pot_shape - xi).abs() < 0.08, "the threshold route gave {pot_shape}");
        assert!(
            (block_shape - pot_shape).abs() < 0.15,
            "the two routes disagree: {block_shape} against {pot_shape}"
        );
    }

    #[test]
    fn block_maxima_drops_a_partial_trailing_block() {
        let x: Vec<f64> = (0..23).map(|i| i as f64).collect();
        let m = block_maxima(&x, 5);
        // Four complete blocks; the last three observations are discarded
        // because their maximum is drawn from fewer draws.
        assert_eq!(m, vec![4.0, 9.0, 14.0, 19.0]);
        assert_eq!(block_maxima(&x, 30), Vec::<f64>::new());
        assert_eq!(block_maxima(&x, 1).len(), 23);
    }

    #[test]
    fn the_mean_excess_is_linear_in_the_threshold_for_pareto_tails() {
        // The threshold diagnostic: if exceedances are generalised Pareto with
        // shape xi < 1, the mean excess above u is (sigma + xi u) / (1 - xi),
        // a straight line of slope xi / (1 - xi).
        let (sigma, xi) = (1.0, 0.25f64);
        let mut rng = Rng::new(0x06D0_11FE);
        let sample = gpd_sample(200_000, sigma, xi, &mut rng);
        let thresholds: Vec<f64> = (0..8).map(|k| k as f64 * 0.5).collect();
        let excess = mean_residual_life(&sample, &thresholds);

        for (u, e) in thresholds.iter().zip(&excess) {
            let expected = (sigma + xi * u) / (1.0 - xi);
            assert!(close(*e, expected, 0.06), "at u = {u} the mean excess is {e}, not {expected}");
        }
        // The slope really is xi / (1 - xi), not zero.
        let slope = (excess[7] - excess[0]) / (thresholds[7] - thresholds[0]);
        assert!(
            close(slope, xi / (1.0 - xi), 0.10),
            "the slope is {slope}, not {}",
            xi / (1.0 - xi)
        );

        // An exponential tail is memoryless, so its mean excess is flat.
        let mut rng = Rng::new(0x06D0_11F0);
        let exponential = gpd_sample(200_000, 1.0, 0.0, &mut rng);
        let flat = mean_residual_life(&exponential, &thresholds);
        for e in &flat {
            assert!(close(*e, 1.0, 0.06), "an exponential mean excess came out {e}");
        }
        // A threshold nothing exceeds is reported, not hidden.
        assert!(mean_residual_life(&exponential, &[1e9])[0].is_nan());
    }

    #[test]
    fn the_hill_estimator_recovers_a_power_law_index() {
        // A Pareto tail with index alpha has extreme value shape 1/alpha, and
        // that is what Hill estimates.
        for &alpha in &[1.5f64, 2.0, 4.0] {
            let mut rng = Rng::new(0x41BB_0000 + (alpha * 10.0) as u64);
            let sample: Vec<f64> = (0..40_000)
                .map(|_| rng.next_f64().clamp(1e-12, 1.0 - 1e-12).powf(-1.0 / alpha))
                .collect();
            let estimate = hill_estimator(&sample, 2000);
            assert!(
                close(estimate, 1.0 / alpha, 0.10),
                "alpha = {alpha}: Hill gave {estimate}, not {}",
                1.0 / alpha
            );
        }
        // Using more order statistics reduces the variance but pulls in the
        // body, so the two ends of the k range bracket the truth differently.
        let mut rng = Rng::new(0x41BB_0002);
        let sample: Vec<f64> =
            (0..40_000).map(|_| rng.next_f64().clamp(1e-12, 1.0 - 1e-12).powf(-0.5)).collect();
        for &k in &[500usize, 2000, 8000] {
            let e = hill_estimator(&sample, k);
            assert!(e > 0.0 && e.is_finite(), "k = {k} gave {e}");
            assert!((e - 0.5).abs() < 0.12, "k = {k} gave {e}");
        }
    }

    #[test]
    fn the_extremal_index_separates_clustered_exceedances_from_isolated_ones() {
        // Independent observations: exceedances arrive one at a time, so the
        // index is one.
        let mut rng = Rng::new(0x0E27_0001);
        let independent: Vec<f64> = (0..20_000).map(|_| rng.next_gaussian()).collect();
        let threshold = 2.0;
        let solo = extremal_index(&independent, threshold);
        assert!(solo > 0.85, "independent exceedances gave an index of {solo}");
        assert!(solo <= 1.0);

        // A moving maximum over a window of m has extremal index 1/m: each
        // large value is echoed m times, so exceedances arrive in clusters of
        // that size.
        for m in [2usize, 4] {
            let mut rng = Rng::new(0x0E27_0002 + m as u64);
            let base: Vec<f64> = (0..40_000).map(|_| rng.next_gaussian()).collect();
            let clustered: Vec<f64> = (m - 1..base.len())
                .map(|t| base[t + 1 - m..=t].iter().copied().fold(f64::NEG_INFINITY, f64::max))
                .collect();
            let index = extremal_index(&clustered, 2.2);
            assert!(
                (index - 1.0 / m as f64).abs() < 0.18,
                "a window of {m} gave an index of {index}, not {}",
                1.0 / m as f64
            );
            assert!(index < solo, "clustering did not lower the index");
        }
        // Too few exceedances to say anything.
        assert_eq!(extremal_index(&[0.0, 0.0, 0.0, 5.0], 1.0), 1.0);
    }

    // -----------------------------------------------------------------
    // Rank correlation
    // -----------------------------------------------------------------

    #[test]
    fn rank_correlations_are_invariant_to_monotone_transformation() {
        // The property that makes them properties of the copula rather than
        // the margins.
        let mut rng = Rng::new(0x002A_0001);
        let x: Vec<f64> = (0..300).map(|_| rng.next_gaussian()).collect();
        let y: Vec<f64> = x.iter().map(|v| 0.6 * v + 0.8 * rng.next_gaussian()).collect();
        let (tau, rho) = (kendall_tau(&x, &y), spearman_rho(&x, &y));

        for f in [
            (|v: f64| v.exp()) as fn(f64) -> f64,
            (|v: f64| v * 3.0 + 7.0) as fn(f64) -> f64,
            (|v: f64| v.tanh()) as fn(f64) -> f64,
        ] {
            let fx: Vec<f64> = x.iter().map(|&v| f(v)).collect();
            let fy: Vec<f64> = y.iter().map(|&v| f(v)).collect();
            assert!((kendall_tau(&fx, &fy) - tau).abs() < 1e-12, "tau moved under a transform");
            assert!((spearman_rho(&fx, &fy) - rho).abs() < 1e-9, "rho moved under a transform");
        }
        // Both agree on the sign and both are bounded.
        assert!(tau > 0.0 && rho > 0.0);
        assert!(tau.abs() <= 1.0 && rho.abs() <= 1.0);
        // Spearman weights the whole distribution, so it reads larger than
        // Kendall for the same monotone dependence.
        assert!(rho > tau, "rho {rho} did not exceed tau {tau}");
    }

    #[test]
    fn the_fast_kendall_agrees_with_comparing_every_pair() {
        // The merge-sort count replaces the O(n^2) definition, so it has to
        // reproduce it exactly -- including the tie handling, which is where
        // an inversion count and a pair count most easily diverge.
        let direct = |x: &[f64], y: &[f64]| -> f64 {
            let n = x.len();
            let (mut c, mut d) = (0i64, 0i64);
            for i in 0..n {
                for j in i + 1..n {
                    use std::cmp::Ordering::Equal;
                    let a = x[j].partial_cmp(&x[i]).unwrap();
                    let b = y[j].partial_cmp(&y[i]).unwrap();
                    if a == Equal || b == Equal {
                        continue;
                    }
                    if a == b {
                        c += 1;
                    } else {
                        d += 1;
                    }
                }
            }
            (c - d) as f64 / (n * (n - 1) / 2) as f64
        };

        let mut rng = Rng::new(0x002A_FA57);
        for round in 0..40 {
            let n = 12 + round * 3;
            // Coarse rounding deliberately manufactures ties in both
            // coordinates, and sometimes in the same pair.
            let grid = 1.0 + (round % 5) as f64 * 2.0;
            let x: Vec<f64> = (0..n).map(|_| (rng.next_gaussian() * grid).round()).collect();
            let y: Vec<f64> = x
                .iter()
                .map(|v| ((0.7 * v + rng.next_gaussian()) * grid).round())
                .collect();
            let fast = kendall_tau(&x, &y);
            let slow = direct(&x, &y);
            assert!(
                (fast - slow).abs() < 1e-12,
                "round {round}: merge count gave {fast}, pair count {slow}"
            );
            assert!(fast.abs() <= 1.0 + 1e-12);
        }
        // Every value identical in one coordinate: all pairs tied, so tau is
        // zero and nothing is comparable.
        assert_eq!(kendall_tau(&[1.0, 2.0, 3.0, 4.0], &[5.0; 4]), 0.0);
        // Every value identical in both.
        assert_eq!(kendall_tau(&[2.0; 6], &[9.0; 6]), 0.0);
    }

    #[test]
    fn perfect_and_reversed_orderings_hit_the_ends_of_the_range() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let up: Vec<f64> = x.iter().map(|v| v * 2.0 + 1.0).collect();
        let down: Vec<f64> = x.iter().map(|v| -v).collect();
        assert!((kendall_tau(&x, &up) - 1.0).abs() < 1e-12);
        assert!((spearman_rho(&x, &up) - 1.0).abs() < 1e-12);
        assert!((kendall_tau(&x, &down) + 1.0).abs() < 1e-12);
        assert!((spearman_rho(&x, &down) + 1.0).abs() < 1e-12);
        // A constant series has no ranks to correlate.
        assert_eq!(kendall_tau(&x, &vec![4.0; 50]), 0.0);
        assert_eq!(spearman_rho(&x, &vec![4.0; 50]), 0.0);
        // Ties are averaged rather than broken arbitrarily.
        let tied = [1.0, 2.0, 2.0, 4.0];
        assert_eq!(ranks(&tied), vec![1.0, 2.5, 2.5, 4.0]);
    }

    // -----------------------------------------------------------------
    // Copulas
    // -----------------------------------------------------------------

    #[test]
    fn every_copula_sampler_produces_uniform_margins() {
        // The defining property: whatever the dependence, each margin is
        // uniform on the unit interval.
        let mut rng = Rng::new(0x00C0_0001);
        let corr = Matrix::from_rows(&[&[1.0, 0.6], &[0.6, 1.0]]).unwrap();
        let samples: Vec<(&str, Vec<Vec<f64>>)> = vec![
            ("gaussian", copula_gaussian_sample(&corr, 20_000, &mut rng).unwrap()),
            ("t", copula_t_sample(&corr, 4.0, 20_000, &mut rng).unwrap()),
            ("clayton", copula_clayton(2.0, 20_000, &mut rng)),
            ("gumbel", copula_gumbel(2.0, 20_000, &mut rng)),
            ("frank", copula_frank(5.0, 20_000, &mut rng)),
        ];
        for (name, data) in samples {
            assert_eq!(data.len(), 20_000);
            for j in 0..2 {
                let column: Vec<f64> = data.iter().map(|r| r[j]).collect();
                assert!(
                    column.iter().all(|&v| (0.0..=1.0).contains(&v)),
                    "{name} margin {j} left the unit interval"
                );
                let mean: f64 = column.iter().sum::<f64>() / column.len() as f64;
                assert!(close(mean, 0.5, 0.03), "{name} margin {j} has mean {mean}");
                // A uniform has variance 1/12.
                let var: f64 = column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                    / column.len() as f64;
                assert!(close(var, 1.0 / 12.0, 0.05), "{name} margin {j} has variance {var}");
                // Every decile should hold about a tenth of the mass.
                for d in 0..10 {
                    let lo = d as f64 / 10.0;
                    let share = column.iter().filter(|&&v| v >= lo && v < lo + 0.1).count() as f64
                        / column.len() as f64;
                    assert!(close(share, 0.1, 0.10), "{name} decile {d} holds {share}");
                }
            }
        }
    }

    #[test]
    fn kendall_tau_inversion_recovers_the_parameter_that_generated_the_sample() {
        // The tau-theta relation of each family, checked against data rather
        // than against itself.
        let mut rng = Rng::new(0x00C0_0002);
        for &theta in &[1.5f64, 3.0, 6.0] {
            let data = copula_clayton(theta, 30_000, &mut rng);
            let fitted = copula_fit_tau(&data, CopulaFamily::Clayton).unwrap();
            assert!(close(fitted, theta, 0.10), "Clayton {theta} came back as {fitted}");
        }
        for &theta in &[1.5f64, 2.5, 5.0] {
            let data = copula_gumbel(theta, 30_000, &mut rng);
            let fitted = copula_fit_tau(&data, CopulaFamily::Gumbel).unwrap();
            assert!(close(fitted, theta, 0.10), "Gumbel {theta} came back as {fitted}");
        }
        for &theta in &[2.0f64, 8.0, -5.0] {
            let data = copula_frank(theta, 30_000, &mut rng);
            let fitted = copula_fit_tau(&data, CopulaFamily::Frank).unwrap();
            assert!(close(fitted, theta, 0.12), "Frank {theta} came back as {fitted}");
        }
        for &rho in &[0.3f64, 0.7, -0.5] {
            let corr = Matrix::from_rows(&[&[1.0, rho], &[rho, 1.0]]).unwrap();
            let data = copula_gaussian_sample(&corr, 30_000, &mut rng).unwrap();
            let fitted = copula_fit_tau(&data, CopulaFamily::Gaussian).unwrap();
            assert!(close(fitted, rho, 0.06), "Gaussian {rho} came back as {fitted}");
        }
    }

    #[test]
    fn the_tau_relations_are_monotone_and_span_the_dependence_range() {
        // tau must increase with the parameter in every family, and reach the
        // right limits: independence at the bottom of each range.
        assert!((copula_tau(CopulaFamily::Clayton, 1e-9)).abs() < 1e-8);
        assert!((copula_tau(CopulaFamily::Gumbel, 1.0)).abs() < 1e-12);
        assert!((copula_tau(CopulaFamily::Gaussian, 0.0)).abs() < 1e-12);
        assert!((copula_tau(CopulaFamily::Frank, 0.0)).abs() < 1e-12);
        assert!((copula_tau(CopulaFamily::Gaussian, 1.0) - 1.0).abs() < 1e-12);

        let mut previous = -2.0;
        for k in 1..60 {
            let t = copula_tau(CopulaFamily::Clayton, k as f64 * 0.5);
            assert!(t > previous, "Clayton tau is not increasing at {k}");
            assert!((0.0..1.0).contains(&t));
            previous = t;
        }
        let mut previous = -2.0;
        for k in 0..60 {
            let t = copula_tau(CopulaFamily::Frank, -20.0 + k as f64 * 0.7);
            assert!(t > previous, "Frank tau is not increasing at {k}");
            assert!((-1.0..1.0).contains(&t));
            previous = t;
        }
        // Frank is antisymmetric in its parameter.
        for &theta in &[1.0f64, 4.0, 12.0] {
            assert!(
                (copula_tau(CopulaFamily::Frank, theta)
                    + copula_tau(CopulaFamily::Frank, -theta))
                .abs()
                    < 1e-6,
                "Frank tau is not odd at {theta}"
            );
        }
    }

    #[test]
    fn tail_dependence_separates_the_families_where_correlation_cannot() {
        // The practical point of the whole section. All four samples here are
        // strongly dependent by any rank measure, and they behave completely
        // differently in the corners.
        let mut rng = Rng::new(0x00C0_7A11);
        let n = 60_000usize;
        let q = 0.01f64;

        // Clayton: lower dependence 2^(-1/theta), upper zero.
        let theta = 2.0f64;
        let clayton = copula_clayton(theta, n, &mut rng);
        let (lower, _) = tail_dependence_coefficient(&clayton, q).unwrap();
        let expected_lower = 2.0f64.powf(-1.0 / theta);
        assert!(
            (lower - expected_lower).abs() < 0.12,
            "Clayton lower tail {lower} against {expected_lower}"
        );
        // The upper coefficient has to be read at a high quantile: at q = 0.01
        // it is the chance both exceed their first percentile, which is near
        // one whatever the copula.
        // Clayton's upper coefficient at a finite q is
        // (1 - 2q + C(q, q)) / (1 - q) with C(q, q) = (2 q^-theta - 1)^(-1/theta),
        // which is small but not zero.
        let high = 1.0 - q;
        let (_, upper) = tail_dependence_coefficient(&clayton, high).unwrap();
        let diagonal = (2.0 * high.powf(-theta) - 1.0).powf(-1.0 / theta);
        let exact_upper = (1.0 - 2.0 * high + diagonal) / (1.0 - high);
        assert!(
            (upper - exact_upper).abs() < 0.10,
            "Clayton upper tail {upper} against the finite-q value {exact_upper}"
        );
        assert!(upper < 0.2, "Clayton showed substantial upper tail dependence: {upper}");

        // Gumbel: the mirror image, 2 - 2^(1/theta) above and nothing below.
        let gumbel = copula_gumbel(theta, n, &mut rng);
        let (_, upper) = tail_dependence_coefficient(&gumbel, 1.0 - q).unwrap();
        let expected_upper = 2.0 - 2.0f64.powf(1.0 / theta);
        assert!(
            (upper - expected_upper).abs() < 0.12,
            "Gumbel upper tail {upper} against {expected_upper}"
        );

        // Gumbel's lower tail dependence is zero only in the limit. At a
        // finite q the exact value is q^(2^(1/theta) - 1), which at q = 0.01
        // and theta = 2 is about 0.15 -- so comparing the estimate against
        // zero would be comparing it against the wrong number. The asymptotic
        // statement is that it falls toward zero as q does, and that is what
        // to check.
        let mut previous = f64::INFINITY;
        for &level in &[0.10f64, 0.05, 0.02, 0.01] {
            let (low_end, _) = tail_dependence_coefficient(&gumbel, level).unwrap();
            let exact = level.powf(2.0f64.powf(1.0 / theta) - 1.0);
            assert!(
                (low_end - exact).abs() < 0.05,
                "Gumbel lower tail at q = {level} is {low_end}, not {exact}"
            );
            assert!(low_end < previous, "the lower coefficient rose as q fell to {level}");
            previous = low_end;
        }
        assert!(previous < 0.35 * upper, "the lower tail did not decay against the upper one");

        // The sharpest contrast in the section, and the one a correlation
        // cannot see. Both samples below have correlation 0.7 and nearly the
        // same rank dependence; one is asymptotically independent in the tails
        // and the other is not.
        let rho = 0.7f64;
        let corr = Matrix::from_rows(&[&[1.0, rho], &[rho, 1.0]]).unwrap();
        let gaussian = copula_gaussian_sample(&corr, n, &mut rng).unwrap();
        let df = 3.0f64;
        let t_sample = copula_t_sample(&corr, df, n, &mut rng).unwrap();

        let gx: Vec<f64> = gaussian.iter().map(|r| r[0]).collect();
        let gy: Vec<f64> = gaussian.iter().map(|r| r[1]).collect();
        let tx: Vec<f64> = t_sample.iter().map(|r| r[0]).collect();
        let ty: Vec<f64> = t_sample.iter().map(|r| r[1]).collect();
        let (g_tau, t_tau) = (kendall_tau(&gx, &gy), kendall_tau(&tx, &ty));
        assert!(g_tau > 0.4, "the Gaussian sample is not strongly dependent");
        assert!(
            (g_tau - t_tau).abs() < 0.05,
            "the two samples differ in rank dependence ({g_tau} against {t_tau}), so the tail \
             comparison would not be like for like"
        );

        // The t copula has an exact limit: 2 t_{df+1}(-sqrt((df+1)(1-rho)/(1+rho))).
        let limit = 2.0
            * StudentT::new(df + 1.0)
                .cdf(-(((df + 1.0) * (1.0 - rho)) / (1.0 + rho)).sqrt());
        assert!(limit > 0.3, "the t copula's theoretical tail dependence is only {limit}");

        // The Gaussian's coefficient falls toward zero as the quantile
        // tightens -- slowly, since the decay is logarithmic, which is why
        // comparing it against zero at any finite q would be the wrong test.
        // The t copula's does not fall: it settles on its limit.
        // Deeper in the tail the two separate. At a loose quantile they are
        // barely distinguishable -- which is the trap: a risk model calibrated
        // on the body of the distribution cannot tell these apart at all.
        let mut previous = f64::INFINITY;
        let mut gaps = Vec::new();
        for &level in &[0.10f64, 0.05, 0.02, 0.01] {
            let (g_low, _) = tail_dependence_coefficient(&gaussian, level).unwrap();
            let (t_low, _) = tail_dependence_coefficient(&t_sample, level).unwrap();
            assert!(g_low < previous, "the Gaussian coefficient rose at q = {level}");
            assert!(t_low > g_low, "at q = {level} the t copula did not exceed the Gaussian");
            assert!(
                (t_low - limit).abs() < 0.12,
                "at q = {level} the t copula gave {t_low} against its limit {limit}"
            );
            gaps.push(t_low - g_low);
            previous = g_low;
        }
        assert!(
            gaps.windows(2).all(|w| w[1] > w[0]),
            "the gap did not widen as the quantile tightened: {gaps:?}"
        );
        assert!(gaps[0] < 0.10, "the two were already separated in the body: {}", gaps[0]);
        assert!(gaps[3] > 0.15, "the two never separated in the tail: {}", gaps[3]);
        let (g_start, _) = tail_dependence_coefficient(&gaussian, 0.10).unwrap();
        assert!(previous < 0.75 * g_start, "the Gaussian tail did not decay: {g_start} to {previous}");

        // Symmetric families, so the same holds in the upper tail.
        let (_, g_high) = tail_dependence_coefficient(&gaussian, 1.0 - q).unwrap();
        let (_, t_high) = tail_dependence_coefficient(&t_sample, 1.0 - q).unwrap();
        assert!(t_high > g_high + 0.10, "the t copula's upper tail ({t_high}) matched the Gaussian's");
        assert!((t_high - limit).abs() < 0.12, "the t upper tail {t_high} against limit {limit}");
    }

    #[test]
    fn the_empirical_copula_transform_uniformises_any_margins() {
        let mut rng = Rng::new(0xC0E1_1000);
        let raw: Vec<Vec<f64>> = (0..2000)
            .map(|_| {
                let a = rng.next_gaussian();
                // Wildly different marginal scales and shapes.
                vec![a.exp() * 1000.0, (0.8 * a + 0.6 * rng.next_gaussian()).tanh()]
            })
            .collect();
        let pseudo = empirical_copula(&raw).unwrap();
        assert_eq!(pseudo.len(), raw.len());

        for j in 0..2 {
            let column: Vec<f64> = pseudo.iter().map(|r| r[j]).collect();
            // Strictly inside the unit interval, so logarithms are safe.
            assert!(column.iter().all(|&v| v > 0.0 && v < 1.0), "column {j} touched an endpoint");
            let mut sorted = column.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // With no ties the ranks are a permutation, so the sorted
            // pseudo-observations are exactly i/(n+1).
            for (i, v) in sorted.iter().enumerate() {
                assert!(
                    (v - (i + 1) as f64 / 2001.0).abs() < 1e-12,
                    "column {j} entry {i} is {v}"
                );
            }
        }
        // The transform is rank-based, so it leaves the dependence untouched.
        let rx: Vec<f64> = raw.iter().map(|r| r[0]).collect();
        let ry: Vec<f64> = raw.iter().map(|r| r[1]).collect();
        let px: Vec<f64> = pseudo.iter().map(|r| r[0]).collect();
        let py: Vec<f64> = pseudo.iter().map(|r| r[1]).collect();
        assert!((kendall_tau(&rx, &ry) - kendall_tau(&px, &py)).abs() < 1e-12);

        assert!(empirical_copula(&[]).is_err());
        assert!(empirical_copula(&[vec![1.0, 2.0], vec![3.0]]).is_err());
    }

    #[test]
    fn the_pickands_function_stays_between_its_two_bounds() {
        // max(t, 1-t) <= A(t) <= 1, with the upper bound attained under
        // independence and the lower under perfect dependence.
        let mut rng = Rng::new(0x91C1_0003);
        let independent: Vec<Vec<f64>> =
            (0..4000).map(|_| vec![rng.next_f64(), rng.next_f64()]).collect();
        let comonotone: Vec<Vec<f64>> = (0..4000)
            .map(|_| {
                let u = rng.next_f64();
                vec![u, u]
            })
            .collect();
        let gumbel = copula_gumbel(2.0, 4000, &mut rng);

        for &t in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
            let bound = t.max(1.0 - t);
            for (name, data) in
                [("independent", &independent), ("comonotone", &comonotone), ("gumbel", &gumbel)]
            {
                let a = pickands_dependence(data, t).unwrap();
                assert!(
                    (bound - 1e-9..=1.0 + 1e-9).contains(&a),
                    "{name} at t = {t} gave A = {a}, outside [{bound}, 1]"
                );
            }
            // Independence sits at the top of the range, perfect dependence at
            // the bottom, and Gumbel strictly in between.
            let ind = pickands_dependence(&independent, t).unwrap();
            let com = pickands_dependence(&comonotone, t).unwrap();
            let gum = pickands_dependence(&gumbel, t).unwrap();
            assert!(ind > 0.9, "independence gave A({t}) = {ind}");
            assert!(com < bound + 0.05, "perfect dependence gave A({t}) = {com}");
            assert!(gum < ind + 1e-9 && gum > com - 1e-9, "Gumbel A({t}) = {gum} is not between");
        }
        // A Gumbel copula with a larger parameter is more dependent, so its
        // Pickands function sits lower.
        let strong = copula_gumbel(5.0, 4000, &mut rng);
        assert!(
            pickands_dependence(&strong, 0.5).unwrap()
                < pickands_dependence(&gumbel, 0.5).unwrap(),
            "stronger dependence did not lower A(1/2)"
        );

        assert!(pickands_dependence(&gumbel, 0.0).is_err());
        assert!(pickands_dependence(&gumbel, 1.0).is_err());
        assert!(pickands_dependence(&[vec![0.5, 0.5]], 0.5).is_err());
    }

    #[test]
    fn the_copula_samplers_reject_malformed_input() {
        let mut rng = Rng::new(9);
        let not_square = Matrix::zeros(2, 3);
        assert!(copula_gaussian_sample(&not_square, 10, &mut rng).is_err());
        let not_correlation = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 2.0]]).unwrap();
        assert!(copula_gaussian_sample(&not_correlation, 10, &mut rng).is_err());
        let not_positive_definite =
            Matrix::from_rows(&[&[1.0, 1.5], &[1.5, 1.0]]).unwrap();
        assert!(copula_gaussian_sample(&not_positive_definite, 10, &mut rng).is_err());
        let fine = Matrix::from_rows(&[&[1.0, 0.4], &[0.4, 1.0]]).unwrap();
        assert!(copula_t_sample(&fine, 0.5, 10, &mut rng).is_err());
        assert!(copula_fit_tau(&[vec![0.1, 0.2]], CopulaFamily::Clayton).is_err());
        // Clayton and Gumbel cannot represent negative dependence.
        let negative: Vec<Vec<f64>> =
            (0..500).map(|i| vec![i as f64, -(i as f64)]).collect();
        assert!(copula_fit_tau(&negative, CopulaFamily::Clayton).is_err());
        assert!(copula_fit_tau(&negative, CopulaFamily::Gumbel).is_err());
        // Frank spans negative dependence, but only strictly inside (-1, 1):
        // its tau approaches -1 asymptotically and never reaches it, so the
        // perfectly reversed sample above has no solution either.
        assert!(copula_fit_tau(&negative, CopulaFamily::Frank).is_err());
        let mut rng = Rng::new(0xF2A4_0001);
        let moderate = copula_frank(-4.0, 3000, &mut rng);
        let fitted = copula_fit_tau(&moderate, CopulaFamily::Frank).unwrap();
        assert!(fitted < 0.0, "a negatively dependent sample fitted {fitted}");
        assert!(tail_dependence_coefficient(&negative, 0.0).is_err());
        assert!(tail_dependence_coefficient(&negative, 1.0).is_err());
    }
}
