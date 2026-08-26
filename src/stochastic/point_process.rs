//! Point processes: random collections of points in time or space.
//!
//! The Poisson process is the reference against which every other is
//! described. It has no memory -- the chance of an event in the next instant
//! does not depend on what happened before -- and everything else follows:
//! counts in disjoint sets are independent and Poisson, waiting times are
//! exponential, and given the count in an interval the points are uniformly
//! scattered in it.
//!
//! The other processes here are departures from that in one of two
//! directions. *Clustered* processes -- Hawkes, Cox, Matern, Thomas -- put
//! more points near other points, either because events trigger events or
//! because the rate is itself random. *Regular* processes have points that
//! avoid each other. Ripley's `K` function and the pair correlation measure
//! which of the three a pattern is, by comparing what is seen at each
//! distance against what a Poisson process would give.

use crate::error::GeomError;
use crate::math::Vec2;
use crate::monte_carlo::Rng;
use crate::spatial::primitives::Rect;
use crate::statistics::inference::TestResult;
use std::f64::consts::PI;

/// Event times of a homogeneous Poisson process on `[0, t_end]`.
///
/// Generated from exponential waiting times, which is the process's own
/// definition rather than a device: the memorylessness of the exponential is
/// exactly the memorylessness of the process.
///
/// # Panics
/// Panics unless the rate is non-negative and `t_end` is positive.
#[must_use]
pub fn poisson_process(rate: f64, t_end: f64, rng: &mut Rng) -> Vec<f64> {
    assert!(rate >= 0.0, "the rate must be non-negative");
    assert!(t_end > 0.0, "the horizon must be positive");
    let mut out = Vec::new();
    if rate == 0.0 {
        return out;
    }
    let mut t = 0.0;
    loop {
        t += -rng.next_f64().max(1e-300).ln() / rate;
        if t > t_end {
            return out;
        }
        out.push(t);
    }
}

/// Event times of a Poisson process whose rate varies with time, by thinning.
///
/// Generate a homogeneous process at the maximum rate, then keep each point
/// with probability equal to the ratio of the true rate there to the
/// maximum. Lewis and Shedler's construction, and it is exact rather than an
/// approximation: the retained points have precisely the right intensity,
/// whatever shape the rate function has.
///
/// # Panics
/// Panics unless `rate_max` is positive, `t_end` is positive, or if the rate
/// function exceeds the stated maximum, which would make the thinning wrong
/// rather than merely inefficient.
pub fn poisson_inhomogeneous(
    rate_fn: &dyn Fn(f64) -> f64,
    rate_max: f64,
    t_end: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(rate_max > 0.0, "the bounding rate must be positive");
    assert!(t_end > 0.0, "the horizon must be positive");
    let mut out = Vec::new();
    let mut t = 0.0;
    loop {
        t += -rng.next_f64().max(1e-300).ln() / rate_max;
        if t > t_end {
            return out;
        }
        let r = rate_fn(t);
        assert!(
            r <= rate_max + 1e-9,
            "the rate {r} at {t} exceeds the stated maximum {rate_max}"
        );
        if rng.next_f64() < r / rate_max {
            out.push(t);
        }
    }
}

/// A Poisson point pattern in a rectangle.
///
/// The count is Poisson with mean `rate` times the area, and given the count
/// the points are independent and uniform -- which is the cleanest statement
/// of what complete spatial randomness means.
///
/// # Panics
/// Panics unless the rate is non-negative and the rectangle has positive
/// area.
#[must_use]
pub fn poisson_2d(rate: f64, region: &Rect, rng: &mut Rng) -> Vec<Vec2> {
    assert!(rate >= 0.0, "the rate must be non-negative");
    let w = region.max.x - region.min.x;
    let h = region.max.y - region.min.y;
    assert!(w > 0.0 && h > 0.0, "the region must have positive area");
    let n = poisson_count(rate * w * h, rng);
    (0..n)
        .map(|_| {
            Vec2::new(region.min.x + w * rng.next_f64(), region.min.y + h * rng.next_f64())
        })
        .collect()
}

/// A Poisson point pattern in a box.
///
/// # Panics
/// Panics unless the rate is non-negative and every side is positive.
#[must_use]
pub fn poisson_3d(
    rate: f64,
    min: (f64, f64, f64),
    max: (f64, f64, f64),
    rng: &mut Rng,
) -> Vec<(f64, f64, f64)> {
    assert!(rate >= 0.0, "the rate must be non-negative");
    let (dx, dy, dz) = (max.0 - min.0, max.1 - min.1, max.2 - min.2);
    assert!(dx > 0.0 && dy > 0.0 && dz > 0.0, "the box must have positive volume");
    let n = poisson_count(rate * dx * dy * dz, rng);
    (0..n)
        .map(|_| {
            (
                min.0 + dx * rng.next_f64(),
                min.1 + dy * rng.next_f64(),
                min.2 + dz * rng.next_f64(),
            )
        })
        .collect()
}

/// A Poisson count with the given mean.
fn poisson_count(mean: f64, rng: &mut Rng) -> usize {
    if mean <= 0.0 {
        return 0;
    }
    if mean > 30.0 {
        // Knuth's product underflows past about seven hundred, and at this
        // mean a normal approximation is inside the sampling noise anyway.
        return (mean + mean.sqrt() * rng.next_gaussian()).max(0.0).round() as usize;
    }
    let limit = (-mean).exp();
    let mut product = 1.0;
    let mut k = 0usize;
    loop {
        product *= rng.next_f64();
        if product <= limit {
            return k;
        }
        k += 1;
    }
}

/// A compound Poisson process: events at Poisson times, each carrying a
/// random mark.
///
/// Returns `(time, mark)` pairs. The running total of the marks is the
/// process usually meant -- insurance claims, trade volumes -- and its
/// variance is `rate * t * E[mark^2]`, not `rate * t * Var[mark]`, because
/// the number of terms is random too.
///
/// # Panics
/// Panics unless the rate is non-negative and `t_end` is positive.
pub fn compound_poisson(
    rate: f64,
    jump_dist: &dyn Fn(&mut Rng) -> f64,
    t_end: f64,
    rng: &mut Rng,
) -> Vec<(f64, f64)> {
    poisson_process(rate, t_end, rng)
        .into_iter()
        .map(|t| {
            let j = jump_dist(rng);
            (t, j)
        })
        .collect()
}

/// The conditional intensity of a Hawkes process with an exponential kernel.
///
/// `mu + sum over past events of alpha exp(-beta (t - t_i))`. Each event
/// raises the chance of the next, and the excitation decays; the process is
/// its own trigger, which is what makes it a model for earthquakes and for
/// order flow alike.
#[must_use]
pub fn hawkes_intensity(events: &[f64], mu: f64, alpha: f64, beta: f64, t: f64) -> f64 {
    mu + events
        .iter()
        .filter(|&&s| s < t)
        .map(|&s| alpha * (-beta * (t - s)).exp())
        .sum::<f64>()
}

/// The branching ratio `alpha / beta`: the expected number of events each
/// event directly triggers.
///
/// Below one the process is stationary; at or above one it explodes, because
/// each generation of offspring is at least as large as the last. It is the
/// mean of a Galton-Watson offspring distribution wearing different clothes.
#[must_use]
pub fn hawkes_branching_ratio(alpha: f64, beta: f64) -> f64 {
    if beta > 0.0 {
        alpha / beta
    } else {
        f64::INFINITY
    }
}

/// A Hawkes process with an exponential kernel, by Ogata's thinning.
///
/// The intensity only ever falls between events, so it can be bounded by its
/// value just after the last one; propose from a homogeneous process at that
/// bound and accept in proportion. Rebounding after each event is what keeps
/// the acceptance rate high.
///
/// # Panics
/// Panics unless `mu` and `beta` are positive, `alpha` is non-negative, and
/// the branching ratio is below one -- above it the process explodes and no
/// simulation terminates.
#[must_use]
pub fn hawkes_process(mu: f64, alpha: f64, beta: f64, t_end: f64, rng: &mut Rng) -> Vec<f64> {
    assert!(mu > 0.0 && beta > 0.0, "the background rate and decay must be positive");
    assert!(alpha >= 0.0, "the excitation must be non-negative");
    assert!(
        hawkes_branching_ratio(alpha, beta) < 1.0,
        "a branching ratio at or above one explodes"
    );
    assert!(t_end > 0.0, "the horizon must be positive");
    let mut events: Vec<f64> = Vec::new();
    let mut t = 0.0f64;
    // The excitation carried forward, so the intensity costs one exponential
    // per proposal instead of a sum over the whole history. Resumming would
    // make a run with n events cost n squared, which is the difference
    // between seconds and hours on a long horizon.
    let mut excitation = 0.0f64;
    loop {
        // The intensity only falls between events, so its value now bounds
        // it until the next one arrives.
        let bound = mu + excitation;
        t += -rng.next_f64().max(1e-300).ln() / bound;
        if t > t_end {
            return events;
        }
        // Decay the excitation forward to the proposed time.
        let decayed = excitation * (-beta * (t - events.last().copied().unwrap_or(0.0))).exp();
        let intensity = mu + decayed;
        if rng.next_f64() * bound <= intensity {
            events.push(t);
            excitation = decayed + alpha;
        } else {
            // The proposal was rejected, but the clock still moved, so the
            // excitation must be carried to where it now sits.
            excitation = decayed;
        }
    }
}

/// The log-likelihood of a Hawkes process with an exponential kernel.
///
/// `sum log lambda(t_i) - integral lambda`. The integral has a closed form
/// for this kernel, and the sum can be accumulated in one pass by the same
/// recursion, so the whole thing is linear in the event count rather than
/// quadratic.
#[must_use]
pub fn hawkes_log_likelihood(events: &[f64], t_end: f64, mu: f64, alpha: f64, beta: f64) -> f64 {
    if mu <= 0.0 || beta <= 0.0 || alpha < 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut total = -mu * t_end;
    // The compensator's excitation part: each event contributes
    // (alpha / beta)(1 - exp(-beta (T - t_i))).
    for &t in events {
        total -= alpha / beta * (1.0 - (-beta * (t_end - t)).exp());
    }
    // The recursion: A_i = exp(-beta (t_i - t_{i-1})) (1 + A_{i-1}).
    let mut a = 0.0f64;
    for (i, &t) in events.iter().enumerate() {
        if i > 0 {
            a = (-beta * (t - events[i - 1])).exp() * (1.0 + a);
        }
        let lambda = mu + alpha * a;
        if lambda <= 0.0 {
            return f64::NEG_INFINITY;
        }
        total += lambda.ln();
    }
    total
}

/// Maximum likelihood estimates of a Hawkes process's parameters.
///
/// Returns `(mu, alpha, beta)`, found by a coordinate search over the
/// log-likelihood. The likelihood is not concave in these coordinates, so
/// this is a local optimiser started from moment-based guesses rather than a
/// guarantee.
///
/// # Panics
/// Panics unless there are at least two events and `t_end` is positive.
#[must_use]
pub fn hawkes_fit_mle(events: &[f64], t_end: f64) -> (f64, f64, f64) {
    assert!(events.len() >= 2, "fitting needs at least two events");
    assert!(t_end > 0.0, "the horizon must be positive");
    // A moment start: the observed rate is mu / (1 - alpha/beta), and the
    // mean gap sets the scale of the decay.
    let observed = events.len() as f64 / t_end;
    let mean_gap = t_end / events.len() as f64;
    let mut best = (0.5 * observed, 0.4 / mean_gap, 1.0 / mean_gap);
    let mut best_ll = hawkes_log_likelihood(events, t_end, best.0, best.1, best.2);
    let mut scale = 0.5f64;
    for _ in 0..60 {
        let mut improved = false;
        for axis in 0..3 {
            for direction in [1.0f64, -1.0] {
                let mut candidate = best;
                let factor = (1.0 + scale * direction).max(0.05);
                match axis {
                    0 => candidate.0 *= factor,
                    1 => candidate.1 *= factor,
                    _ => candidate.2 *= factor,
                }
                // Stay inside the stationary region, where the likelihood is
                // the one being maximised.
                if candidate.1 >= candidate.2 {
                    continue;
                }
                let ll = hawkes_log_likelihood(events, t_end, candidate.0, candidate.1, candidate.2);
                if ll > best_ll {
                    best_ll = ll;
                    best = candidate;
                    improved = true;
                }
            }
        }
        if !improved {
            scale *= 0.6;
            if scale < 1e-4 {
                break;
            }
        }
    }
    best
}

/// A renewal process: event times from independent waiting times of any
/// distribution.
///
/// The Poisson process is the special case where the waits are exponential,
/// and it is the only one that is memoryless -- for any other law, how long
/// you have waited tells you something about how much longer you will.
///
/// # Panics
/// Panics if `t_end` is not positive, or if the interarrival draw is not
/// positive, which would make the process explode.
pub fn renewal_process(
    interarrival: &dyn Fn(&mut Rng) -> f64,
    t_end: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    assert!(t_end > 0.0, "the horizon must be positive");
    let mut out = Vec::new();
    let mut t = 0.0;
    loop {
        let gap = interarrival(rng);
        assert!(gap > 0.0 && gap.is_finite(), "an interarrival time must be positive");
        t += gap;
        if t > t_end {
            return out;
        }
        out.push(t);
    }
}

/// The renewal function: the expected number of events by time `t`,
/// estimated by simulation.
///
/// Asymptotically `t / mean_gap`, whatever the waiting law -- the elementary
/// renewal theorem, which says the long-run rate depends on the mean alone
/// and not on the shape.
///
/// # Panics
/// Panics unless `t` is positive and `n_paths` is positive.
pub fn renewal_function_estimate(
    interarrival: &dyn Fn(&mut Rng) -> f64,
    t: f64,
    n_paths: usize,
    rng: &mut Rng,
) -> f64 {
    assert!(t > 0.0 && n_paths > 0, "the horizon and the path count must be positive");
    let total: usize = (0..n_paths).map(|_| renewal_process(interarrival, t, rng).len()).sum();
    total as f64 / n_paths as f64
}

/// A Cox process: a Poisson process whose rate is itself random.
///
/// Also called doubly stochastic. Drawing the rate first and then the points
/// makes the counts *over*-dispersed relative to Poisson -- the variance
/// exceeds the mean, because the randomness of the rate adds to the
/// randomness of the count. That is the signature to look for when a count
/// is more variable than Poisson allows.
///
/// # Panics
/// Panics unless `t_end` is positive or if the drawn rate is negative.
pub fn cox_process(
    rate_dist: &dyn Fn(&mut Rng) -> f64,
    t_end: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let rate = rate_dist(rng);
    assert!(rate >= 0.0, "a drawn rate must be non-negative");
    poisson_process(rate, t_end, rng)
}

/// A Matern cluster process: Poisson parents, each surrounded by a Poisson
/// number of daughters uniformly inside a disc.
///
/// Only the daughters are returned. Parents outside the region still throw
/// daughters into it, so they are generated over a margin as wide as the
/// cluster radius; omitting that margin would thin the pattern near the
/// edges and is the standard way a clustered simulation comes out wrong.
///
/// # Panics
/// Panics unless the rates and the radius are positive and the region has
/// positive area.
#[must_use]
pub fn matern_cluster_process(
    parent_rate: f64,
    cluster_radius: f64,
    daughter_mean: f64,
    region: &Rect,
    rng: &mut Rng,
) -> Vec<Vec2> {
    assert!(parent_rate > 0.0 && daughter_mean > 0.0, "the rates must be positive");
    assert!(cluster_radius > 0.0, "the radius must be positive");
    let grown = Rect {
        min: Vec2::new(region.min.x - cluster_radius, region.min.y - cluster_radius),
        max: Vec2::new(region.max.x + cluster_radius, region.max.y + cluster_radius),
    };
    let parents = poisson_2d(parent_rate, &grown, rng);
    let mut out = Vec::new();
    for p in parents {
        for _ in 0..poisson_count(daughter_mean, rng) {
            // Uniform in the disc needs the square root, or the points pile
            // up at the centre.
            let r = cluster_radius * rng.next_f64().sqrt();
            let a = 2.0 * PI * rng.next_f64();
            let q = Vec2::new(p.x + r * a.cos(), p.y + r * a.sin());
            if q.x >= region.min.x && q.x <= region.max.x && q.y >= region.min.y && q.y <= region.max.y
            {
                out.push(q);
            }
        }
    }
    out
}

/// A Thomas process: the same as Matern, with daughters scattered by a
/// Gaussian instead of uniformly in a disc.
///
/// The Gaussian has no hard edge, so the clusters blend rather than ending
/// abruptly; the margin is taken at four standard deviations, past which the
/// contribution is negligible.
///
/// # Panics
/// Panics unless the rates and the spread are positive.
#[must_use]
pub fn thomas_process(
    parent_rate: f64,
    spread: f64,
    daughter_mean: f64,
    region: &Rect,
    rng: &mut Rng,
) -> Vec<Vec2> {
    assert!(parent_rate > 0.0 && daughter_mean > 0.0, "the rates must be positive");
    assert!(spread > 0.0, "the spread must be positive");
    let margin = 4.0 * spread;
    let grown = Rect {
        min: Vec2::new(region.min.x - margin, region.min.y - margin),
        max: Vec2::new(region.max.x + margin, region.max.y + margin),
    };
    let parents = poisson_2d(parent_rate, &grown, rng);
    let mut out = Vec::new();
    for p in parents {
        for _ in 0..poisson_count(daughter_mean, rng) {
            let q = Vec2::new(
                p.x + spread * rng.next_gaussian(),
                p.y + spread * rng.next_gaussian(),
            );
            if q.x >= region.min.x && q.x <= region.max.x && q.y >= region.min.y && q.y <= region.max.y
            {
                out.push(q);
            }
        }
    }
    out
}

/// Ripley's `K` function: the expected number of further points within `r` of
/// a typical point, divided by the intensity.
///
/// For complete spatial randomness it is `pi r^2` at every distance, because
/// the expected count in a disc is the intensity times its area and the
/// division cancels the intensity. Above that means clustering and below
/// means regularity, so the whole diagnostic is a comparison against a
/// parabola.
///
/// Edge effects are handled by Ripley's isotropic correction: a point near
/// the boundary sees only part of its own circle, so each neighbour is
/// weighted by the reciprocal of the fraction of that circle lying inside
/// the region. Without it every pattern looks regular near the edges.
///
/// The correction is trustworthy only while the radius stays well inside the
/// window -- a quarter of the shorter side is the usual limit. Beyond that a
/// point near a corner has most of its circle outside, the weight it earns is
/// large, and the estimate becomes both noisy and biased upward.
///
/// # Panics
/// Panics unless the region has positive area and the radii are positive.
#[must_use]
pub fn ripley_k(points: &[Vec2], region: &Rect, r_values: &[f64]) -> Vec<f64> {
    let w = region.max.x - region.min.x;
    let h = region.max.y - region.min.y;
    assert!(w > 0.0 && h > 0.0, "the region must have positive area");
    assert!(r_values.iter().all(|&r| r > 0.0), "the radii must be positive");
    let n = points.len();
    let area = w * h;
    if n < 2 {
        return vec![0.0; r_values.len()];
    }
    let intensity = n as f64 / area;
    r_values
        .iter()
        .map(|&r| {
            let mut total = 0.0;
            for (i, p) in points.iter().enumerate() {
                let weight = ripley_weight(*p, r, region);
                for (j, q) in points.iter().enumerate() {
                    if i != j && p.distance_to(q) <= r {
                        total += weight;
                    }
                }
            }
            total / (n as f64 * intensity)
        })
        .collect()
}

/// The reciprocal of the fraction of the circle of radius `r` about `p` that
/// lies inside the rectangle, approximated by sampling the circumference.
fn ripley_weight(p: Vec2, r: f64, region: &Rect) -> f64 {
    const SAMPLES: usize = 72;
    let inside = (0..SAMPLES)
        .filter(|&k| {
            let a = 2.0 * PI * k as f64 / SAMPLES as f64;
            let x = p.x + r * a.cos();
            let y = p.y + r * a.sin();
            x >= region.min.x && x <= region.max.x && y >= region.min.y && y <= region.max.y
        })
        .count();
    if inside == 0 {
        1.0
    } else {
        SAMPLES as f64 / inside as f64
    }
}

/// Besag's `L` function: `sqrt(K / pi)`, which is `r` itself under complete
/// spatial randomness.
///
/// The point of the transformation is that a straight line is far easier to
/// read a departure from than a parabola, and it stabilises the variance
/// along the way.
///
/// # Panics
/// Panics under the same conditions as [`ripley_k`].
#[must_use]
pub fn l_function(points: &[Vec2], region: &Rect, r_values: &[f64]) -> Vec<f64> {
    ripley_k(points, region, r_values).into_iter().map(|k| (k / PI).sqrt()).collect()
}

/// The pair correlation function: the density of points at distance `r` from
/// a typical point, relative to the intensity.
///
/// One everywhere under complete spatial randomness. Where `K` accumulates
/// everything within `r` and so smears features together, this looks at a
/// shell of width `dr` and shows the distance at which clustering actually
/// happens.
///
/// # Panics
/// Panics unless the region has positive area and `r` and `dr` are positive
/// with `dr` below `r`.
#[must_use]
pub fn pair_correlation(points: &[Vec2], region: &Rect, r: f64, dr: f64) -> f64 {
    let w = region.max.x - region.min.x;
    let h = region.max.y - region.min.y;
    assert!(w > 0.0 && h > 0.0, "the region must have positive area");
    assert!(r > 0.0 && dr > 0.0 && dr < r, "the shell must be inside the radius");
    let n = points.len();
    if n < 2 {
        return 0.0;
    }
    let area = w * h;
    let intensity = n as f64 / area;
    let mut total = 0.0;
    for (i, p) in points.iter().enumerate() {
        let weight = ripley_weight(*p, r, region);
        for (j, q) in points.iter().enumerate() {
            let d = p.distance_to(q);
            if i != j && (d - r).abs() <= dr / 2.0 {
                total += weight;
            }
        }
    }
    // The shell's area, against which the count is normalised.
    let shell = 2.0 * PI * r * dr;
    total / (n as f64 * intensity * shell)
}

/// The Clark-Evans nearest neighbour index: the mean nearest-neighbour
/// distance divided by what a Poisson pattern of the same intensity would
/// give.
///
/// One for complete spatial randomness, below one for clustering, above for
/// regularity. The expected distance under randomness is
/// `1 / (2 sqrt(intensity))`, which follows from the void probability: the
/// chance that the nearest neighbour is beyond `r` is the chance a disc of
/// radius `r` is empty.
///
/// # Panics
/// Panics unless the region has positive area and there are at least two
/// points.
#[must_use]
pub fn nearest_neighbor_index(points: &[Vec2], region: &Rect) -> f64 {
    let w = region.max.x - region.min.x;
    let h = region.max.y - region.min.y;
    assert!(w > 0.0 && h > 0.0, "the region must have positive area");
    assert!(points.len() >= 2, "the index needs at least two points");
    let intensity = points.len() as f64 / (w * h);
    let mean_observed: f64 = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            points
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, q)| p.distance_to(q))
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / points.len() as f64;
    let expected = 1.0 / (2.0 * intensity.sqrt());
    mean_observed / expected
}

/// The quadrat test: divide the region into cells and test whether the counts
/// look Poisson.
///
/// Under complete spatial randomness every cell has the same expected count,
/// so a chi-squared goodness-of-fit against a flat expectation is the test.
/// It sees departures in the *variance* of the counts and is blind to
/// anything at a scale finer than a cell, which is why it is a first look
/// rather than a conclusion.
///
/// # Errors
/// Returns an error unless the grid is at least two by two and there are at
/// least as many points as cells.
pub fn quadrat_test(
    points: &[Vec2],
    region: &Rect,
    nx: usize,
    ny: usize,
) -> Result<TestResult, GeomError> {
    if nx < 2 || ny < 2 {
        return Err(GeomError::InvalidArgument("the grid must be at least two by two"));
    }
    let w = region.max.x - region.min.x;
    let h = region.max.y - region.min.y;
    if w <= 0.0 || h <= 0.0 {
        return Err(GeomError::InvalidArgument("the region must have positive area"));
    }
    let cells = nx * ny;
    if points.len() < cells {
        return Err(GeomError::InvalidArgument("too few points for this many cells"));
    }
    let mut counts = vec![0.0f64; cells];
    for p in points {
        let cx = (((p.x - region.min.x) / w * nx as f64) as usize).min(nx - 1);
        let cy = (((p.y - region.min.y) / h * ny as f64) as usize).min(ny - 1);
        counts[cy * nx + cx] += 1.0;
    }
    let expected = vec![points.len() as f64 / cells as f64; cells];
    Ok(crate::statistics::inference::chi_squared_gof(&counts, &expected))
}

/// Tests whether the gaps between events look exponential, which is what a
/// Poisson process requires.
///
/// A Kolmogorov-Smirnov test against the exponential distribution with the
/// observed mean. A small p-value says the process is not Poisson; a large
/// one says only that this particular test did not notice.
///
/// # Errors
/// Returns an error unless there are at least three events with positive
/// gaps.
pub fn ks_test_exponential_interarrivals(events: &[f64]) -> Result<TestResult, GeomError> {
    if events.len() < 3 {
        return Err(GeomError::InvalidArgument("at least three events are required"));
    }
    let gaps: Vec<f64> = std::iter::once(events[0])
        .chain(events.windows(2).map(|w| w[1] - w[0]))
        .collect();
    if gaps.iter().any(|&g| g <= 0.0) {
        return Err(GeomError::InvalidArgument("the gaps must be positive"));
    }
    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    let cdf = move |x: f64| if x <= 0.0 { 0.0 } else { 1.0 - (-x / mean).exp() };
    Ok(crate::statistics::inference::ks_test_one_sample(&gaps, &cdf))
}

/// A Galton-Watson branching process: the population size at each
/// generation.
///
/// Every individual independently has a random number of offspring from the
/// same distribution. The population dies out with probability one when the
/// mean offspring count is at most one -- including exactly one, which is the
/// surprise: a population that replaces itself on average still goes extinct
/// unless the count is deterministic.
///
/// A supercritical population is held once it passes two thousand: beyond
/// that its extinction probability is smaller than any double can represent,
/// so the remaining generations carry no information and every one of them
/// would cost time proportional to the population.
///
/// # Panics
/// Panics unless the offspring distribution is a probability vector.
#[must_use]
pub fn branching_process_gw(
    offspring_pmf: &[f64],
    generations: usize,
    rng: &mut Rng,
) -> Vec<u64> {
    assert!(!offspring_pmf.is_empty(), "the offspring distribution must not be empty");
    assert!(offspring_pmf.iter().all(|&p| p >= 0.0), "a probability is negative");
    let total: f64 = offspring_pmf.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "the offspring distribution must sum to one");
    let mut sizes = Vec::with_capacity(generations + 1);
    let mut n = 1u64;
    sizes.push(n);
    for _ in 0..generations {
        let mut next = 0u64;
        for _ in 0..n {
            let u = rng.next_f64();
            let mut acc = 0.0;
            for (k, &p) in offspring_pmf.iter().enumerate() {
                acc += p;
                if u < acc {
                    next += k as u64;
                    break;
                }
            }
        }
        n = next;
        sizes.push(n);
        if n == 0 {
            // Extinction is absorbing, so the rest of the run is zeros.
            sizes.resize(generations + 1, 0);
            break;
        }
        // Past this size extinction has probability below any double can
        // represent, so continuing only costs time proportional to the
        // population. The size is held rather than grown further.
        if n > 2_000 {
            sizes.resize(generations + 1, n);
            break;
        }
    }
    sizes
}

/// The extinction probability of a branching process: the smallest fixed
/// point of the offspring generating function in `[0, 1]`.
///
/// One when the mean offspring count is at most one, and strictly below one
/// above it. The fixed point equation says that a lineage dies out exactly
/// when every one of its founder's children's lineages does, which is the
/// whole argument in one line.
///
/// # Panics
/// Panics unless the coefficients are a probability vector.
#[must_use]
pub fn extinction_probability(offspring_pgf_coeffs: &[f64]) -> f64 {
    assert!(!offspring_pgf_coeffs.is_empty(), "the distribution must not be empty");
    assert!(offspring_pgf_coeffs.iter().all(|&p| p >= 0.0), "a probability is negative");
    let total: f64 = offspring_pgf_coeffs.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "the distribution must sum to one");
    let pgf = |s: f64| -> f64 {
        offspring_pgf_coeffs.iter().enumerate().map(|(k, &p)| p * s.powi(k as i32)).sum()
    };
    let mean: f64 = offspring_pgf_coeffs.iter().enumerate().map(|(k, &p)| k as f64 * p).sum();
    if mean <= 1.0 {
        return 1.0;
    }
    // Iterating the generating function from zero converges upward to the
    // smallest fixed point, which is exactly the extinction probability.
    let mut q = 0.0f64;
    for _ in 0..10_000 {
        let next = pgf(q);
        if (next - q).abs() < 1e-15 {
            break;
        }
        q = next;
    }
    q.clamp(0.0, 1.0)
}

/// A Yule process: pure birth, each individual splitting at a constant rate.
///
/// Returns the times at which the population grew. The population at time `t`
/// is geometric with mean `exp(birth_rate t)`, which is the continuous-time
/// analogue of a branching process that never dies.
///
/// # Panics
/// Panics unless the rate and the horizon are positive.
#[must_use]
pub fn yule_process(birth_rate: f64, t_end: f64, rng: &mut Rng) -> Vec<f64> {
    assert!(birth_rate > 0.0 && t_end > 0.0, "the rate and horizon must be positive");
    let mut out = Vec::new();
    let mut t = 0.0;
    let mut n = 1u64;
    loop {
        // The total birth rate scales with the population, so the waits
        // shorten as it grows.
        t += -rng.next_f64().max(1e-300).ln() / (birth_rate * n as f64);
        if t > t_end || n > 100_000 {
            return out;
        }
        n += 1;
        out.push(t);
    }
}

/// A linear birth-death process, by Gillespie's direct method.
///
/// Returns `(time, population)` after each event. The population dies out
/// with probability one when the death rate is at least the birth rate, and
/// with probability `(death / birth)^n0` when it is not -- which is the
/// branching process's extinction probability again, in continuous time.
///
/// A population past a thousand is held, for the reason
/// [`branching_process_gw`] gives.
///
/// # Panics
/// Panics unless the rates are non-negative and the horizon is positive.
#[must_use]
pub fn birth_death_simulate(
    birth: f64,
    death: f64,
    n0: u64,
    t_end: f64,
    rng: &mut Rng,
) -> Vec<(f64, u64)> {
    assert!(birth >= 0.0 && death >= 0.0, "the rates must be non-negative");
    assert!(t_end > 0.0, "the horizon must be positive");
    let mut out = vec![(0.0, n0)];
    let mut t = 0.0;
    let mut n = n0;
    loop {
        // The same reasoning as the branching process: a population this
        // large will not die out, and every further event costs time.
        if n == 0 || n > 1_000 {
            return out;
        }
        let total = (birth + death) * n as f64;
        if total <= 0.0 {
            return out;
        }
        t += -rng.next_f64().max(1e-300).ln() / total;
        if t > t_end {
            return out;
        }
        // Which of the two competing events fired, in proportion to its rate.
        if rng.next_f64() < birth / (birth + death) {
            n += 1;
        } else {
            n -= 1;
        }
        out.push((t, n));
    }
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

    fn unit_square() -> Rect {
        Rect { min: Vec2::new(0.0, 0.0), max: Vec2::new(1.0, 1.0) }
    }

    /// A Poisson process has Poisson counts, exponential gaps, and uniform
    /// points given the count -- the three statements that define it.
    #[test]
    fn the_poisson_process_has_its_three_defining_properties() {
        let mut rng = Rng::new(0x_9015);
        let rate = 4.0f64;
        let t_end = 3.0f64;
        let runs: Vec<Vec<f64>> = (0..25_000).map(|_| poisson_process(rate, t_end, &mut rng)).collect();
        let counts: Vec<f64> = runs.iter().map(|r| r.len() as f64).collect();
        let expected = rate * t_end;
        // Mean and variance both equal the rate times the horizon, which is
        // the Poisson signature and rules out most alternatives at once.
        assert!((mean(&counts) - expected).abs() < 0.06, "the mean count is {}", mean(&counts));
        assert!(
            (variance(&counts) / expected - 1.0).abs() < 0.04,
            "the variance is {} against {expected}",
            variance(&counts)
        );
        // The count distribution itself, against the Poisson mass function.
        let mut observed = vec![0.0f64; 30];
        for &c in &counts {
            if (c as usize) < 30 {
                observed[c as usize] += 1.0;
            }
        }
        let mut factorial = 1.0f64;
        for k in 0..30usize {
            if k > 0 {
                factorial *= k as f64;
            }
            let pk = (-expected).exp() * expected.powi(k as i32) / factorial;
            let seen = observed[k] / counts.len() as f64;
            assert!(
                (seen - pk).abs() < 0.008,
                "P(N = {k}) came out at {seen} against {pk}"
            );
        }
        // Events are ordered, inside the horizon, and their gaps exponential.
        for r in runs.iter().take(200) {
            assert!(r.windows(2).all(|w| w[0] < w[1]), "the events are out of order");
            assert!(r.iter().all(|&t| t > 0.0 && t <= t_end), "an event left the horizon");
        }
        let long = poisson_process(rate, 2_000.0, &mut rng);
        let ks = ks_test_exponential_interarrivals(&long).expect("enough events");
        assert!(ks.p_value > 0.001, "the gaps failed the exponential test at p = {}", ks.p_value);
        // Given the count, the points are uniform: their mean should sit at
        // the middle of the horizon.
        let all: Vec<f64> = runs.iter().flatten().copied().collect();
        assert!((mean(&all) - t_end / 2.0).abs() < 0.01, "the points are not uniform");
        assert!(poisson_process(0.0, 1.0, &mut rng).is_empty());
    }

    /// Thinning produces exactly the requested intensity, however the rate
    /// varies.
    #[test]
    fn thinning_reproduces_a_varying_rate() {
        let mut rng = Rng::new(0x_7417);
        // A rate that doubles across the window, so a constant-rate process
        // could not be mistaken for it.
        let rate_fn = |t: f64| 2.0 + 2.0 * t;
        let t_end = 4.0f64;
        let runs: Vec<Vec<f64>> = (0..15_000)
            .map(|_| poisson_inhomogeneous(&rate_fn, 10.0, t_end, &mut rng))
            .collect();
        // The expected count is the integral of the rate.
        let expected = 2.0 * t_end + t_end * t_end;
        let counts: Vec<f64> = runs.iter().map(|r| r.len() as f64).collect();
        assert!(
            (mean(&counts) - expected).abs() < 0.08,
            "the mean count is {} against {expected}",
            mean(&counts)
        );
        assert!(
            (variance(&counts) / expected - 1.0).abs() < 0.05,
            "an inhomogeneous Poisson count should still have variance equal to its mean"
        );
        // The counts in the two halves are in the ratio the rate dictates.
        let first: f64 =
            runs.iter().map(|r| r.iter().filter(|&&t| t < 2.0).count() as f64).sum::<f64>()
                / runs.len() as f64;
        let second = mean(&counts) - first;
        let want_first = 2.0 * 2.0 + 4.0;
        assert!((first - want_first).abs() < 0.06, "the first half holds {first}");
        assert!((second - (expected - want_first)).abs() < 0.06, "the second half holds {second}");
        // A rate function exceeding its stated bound is caught rather than
        // silently producing the wrong process.
        let bad = |_t: f64| 100.0f64;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut r = Rng::new(1);
            poisson_inhomogeneous(&bad, 1.0, 1.0, &mut r)
        }))
        .is_err());
    }

    /// A spatial Poisson pattern is completely random by every measure meant
    /// to detect that it is not.
    #[test]
    fn a_spatial_poisson_pattern_looks_completely_random() {
        let mut rng = Rng::new(0x_5A47);
        let region = unit_square();
        // Ripley's K against pi r squared, which is the definition of
        // complete spatial randomness.
        // Kept below a quarter of the side, where the edge correction is
        // reliable; see the note on ripley_k.
        let radii = [0.05f64, 0.08, 0.12, 0.16];
        let mut totals = vec![0.0f64; radii.len()];
        let reps = 20;
        for _ in 0..reps {
            let pts = poisson_2d(400.0, &region, &mut rng);
            for (i, k) in ripley_k(&pts, &region, &radii).into_iter().enumerate() {
                totals[i] += k;
            }
        }
        for (i, &r) in radii.iter().enumerate() {
            let observed = totals[i] / reps as f64;
            let want = PI * r * r;
            assert!(
                (observed / want - 1.0).abs() < 0.1,
                "K({r}) came out at {observed} against {want}"
            );
        }
        // The L function is the identity, which is the same statement made
        // easier to read.
        let pts = poisson_2d(900.0, &region, &mut rng);
        for (i, l) in l_function(&pts, &region, &radii).into_iter().enumerate() {
            assert!(
                (l - radii[i]).abs() < 0.02,
                "L({}) came out at {l}",
                radii[i]
            );
        }
        // The pair correlation is one at every distance.
        for r in [0.06f64, 0.1, 0.15] {
            let g = pair_correlation(&pts, &region, r, 0.02);
            assert!((g - 1.0).abs() < 0.15, "the pair correlation at {r} is {g}");
        }
        // Clark-Evans is one, and the quadrat test does not reject.
        let index = nearest_neighbor_index(&pts, &region);
        assert!((index - 1.0).abs() < 0.06, "the nearest neighbour index is {index}");
        let q = quadrat_test(&pts, &region, 6, 6).expect("enough points");
        assert!(q.p_value > 0.001, "a random pattern was rejected at p = {}", q.p_value);
        // The count is Poisson with mean the rate times the area.
        let counts: Vec<f64> =
            (0..8_000).map(|_| poisson_2d(20.0, &region, &mut rng).len() as f64).collect();
        assert!((mean(&counts) - 20.0).abs() < 0.2);
        assert!((variance(&counts) / 20.0 - 1.0).abs() < 0.08);
        let boxed = poisson_3d(1_000.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), &mut rng);
        assert!((boxed.len() as f64 - 1_000.0).abs() < 150.0);
    }

    /// Clustered patterns are detected as clustered by every diagnostic, and
    /// the diagnostics disagree with what they said about a random one.
    #[test]
    fn clustered_patterns_are_detected_as_clustered() {
        let mut rng = Rng::new(0x_C1005);
        let region = unit_square();
        let radii = [0.04f64, 0.08, 0.12];
        for (name, pts) in [
            ("Matern", matern_cluster_process(40.0, 0.04, 25.0, &region, &mut rng)),
            ("Thomas", thomas_process(40.0, 0.02, 25.0, &region, &mut rng)),
        ] {
            assert!(pts.len() > 300, "{name} produced only {} points", pts.len());
            assert!(
                pts.iter().all(|p| p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0),
                "{name} put a point outside the region"
            );
            // K above pi r squared at every scale below the cluster size.
            for (i, k) in ripley_k(&pts, &region, &radii).into_iter().enumerate() {
                let csr = PI * radii[i] * radii[i];
                assert!(k > 1.5 * csr, "{name}: K({}) is {k} against {csr}", radii[i]);
            }
            // Points sit closer together than randomness would put them.
            let index = nearest_neighbor_index(&pts, &region);
            assert!(index < 0.8, "{name}: the nearest neighbour index is {index}");
            // And the quadrat counts are over-dispersed enough to reject.
            let q = quadrat_test(&pts, &region, 6, 6).expect("enough points");
            assert!(q.p_value < 0.01, "{name}: the quadrat test did not reject, p = {}", q.p_value);
            // The pair correlation exceeds one at short range and settles.
            let close = pair_correlation(&pts, &region, 0.03, 0.02);
            assert!(close > 1.5, "{name}: the short-range correlation is only {close}");
        }
        // Clusters must not thin out at the edges, which is what the margin
        // in the parent process prevents. Compare the density in the middle
        // against the density in the border strip.
        let pts = matern_cluster_process(80.0, 0.05, 20.0, &region, &mut rng);
        let border = pts
            .iter()
            .filter(|p| p.x < 0.1 || p.x > 0.9 || p.y < 0.1 || p.y > 0.9)
            .count() as f64;
        let border_area = 1.0 - 0.8 * 0.8;
        let density_border = border / border_area;
        let density_all = pts.len() as f64;
        assert!(
            (density_border / density_all - 1.0).abs() < 0.25,
            "the border density is {density_border} against {density_all}, so the margin is wrong"
        );
    }

    /// A Hawkes process excites itself: its intensity follows the rule it is
    /// defined by, its rate exceeds the background, and the fitted parameters
    /// come back.
    #[test]
    fn the_hawkes_process_excites_itself_and_can_be_fitted() {
        let (mu, alpha, beta) = (0.8f64, 0.6f64, 1.4f64);
        let ratio = hawkes_branching_ratio(alpha, beta);
        assert!((ratio - alpha / beta).abs() < 1e-12);
        assert!(ratio < 1.0);
        // The intensity against its definition, on a hand-built history.
        let history = [0.5f64, 1.0, 2.5];
        let want = mu
            + alpha * (-beta * (3.0 - 0.5f64)).exp()
            + alpha * (-beta * (3.0 - 1.0f64)).exp()
            + alpha * (-beta * (3.0 - 2.5f64)).exp();
        assert!((hawkes_intensity(&history, mu, alpha, beta, 3.0) - want).abs() < 1e-12);
        // Events after the query time do not count.
        assert!((hawkes_intensity(&history, mu, alpha, beta, 0.25) - mu).abs() < 1e-12);

        let mut rng = Rng::new(0x_4A00);
        let t_end = 8_000.0f64;
        let events = hawkes_process(mu, alpha, beta, t_end, &mut rng);
        // The stationary rate is mu / (1 - alpha/beta), which is the
        // background inflated by every generation of offspring.
        let observed = events.len() as f64 / t_end;
        let want_rate = mu / (1.0 - ratio);
        assert!(
            (observed / want_rate - 1.0).abs() < 0.06,
            "the rate came out at {observed} against {want_rate}"
        );
        // Clustering: the gaps are over-dispersed relative to exponential,
        // so the exponential test rejects where it would not for Poisson.
        let ks = ks_test_exponential_interarrivals(&events).expect("enough events");
        assert!(ks.p_value < 1e-6, "the Hawkes gaps looked exponential, p = {}", ks.p_value);

        // The likelihood is maximised near the truth.
        let truth_ll = hawkes_log_likelihood(&events, t_end, mu, alpha, beta);
        for (dm, da, db) in [(0.5f64, 1.0f64, 1.0f64), (1.0, 0.4, 1.0), (1.0, 1.0, 2.0)] {
            let off = hawkes_log_likelihood(&events, t_end, mu * dm, alpha * da, beta * db);
            assert!(off < truth_ll, "a displaced parameter scored higher: {off} against {truth_ll}");
        }
        // And the fit recovers them.
        let (fm, fa, fb) = hawkes_fit_mle(&events, t_end);
        assert!((fm / mu - 1.0).abs() < 0.15, "mu was fitted at {fm} against {mu}");
        assert!(
            (fa / fb / ratio - 1.0).abs() < 0.15,
            "the branching ratio was fitted at {} against {ratio}",
            fa / fb
        );
        assert!(hawkes_log_likelihood(&events, t_end, fm, fa, fb) >= truth_ll - 1.0);
        // An explosive branching ratio is refused rather than hanging.
        assert!(std::panic::catch_unwind(|| {
            let mut r = Rng::new(1);
            hawkes_process(1.0, 2.0, 1.0, 10.0, &mut r)
        })
        .is_err());
    }

    /// Renewal and Cox processes differ from Poisson in the two ways they are
    /// supposed to.
    #[test]
    fn renewal_and_cox_processes_depart_from_poisson_as_they_should() {
        let mut rng = Rng::new(0x_2E4E);
        // A renewal process with deterministic-ish gaps is far more regular
        // than Poisson: the count has much less than Poisson variance.
        let tight = |r: &mut Rng| 1.0 + 0.05 * r.next_gaussian();
        let counts: Vec<f64> =
            (0..4_000).map(|_| renewal_process(&tight, 50.0, &mut rng).len() as f64).collect();
        assert!((mean(&counts) - 49.0).abs() < 1.5, "the mean count is {}", mean(&counts));
        assert!(
            variance(&counts) < 0.2 * mean(&counts),
            "a near-deterministic renewal process should be under-dispersed, not {}",
            variance(&counts)
        );
        // The elementary renewal theorem: the rate is one over the mean gap,
        // whatever the shape of the waiting law.
        for (name, gap, want) in [
            ("tight", &tight as &dyn Fn(&mut Rng) -> f64, 1.0f64),
            ("exponential", &|r: &mut Rng| -r.next_f64().max(1e-300).ln() / 2.0, 0.5),
            ("uniform", &|r: &mut Rng| 0.2 + 1.6 * r.next_f64(), 1.0),
        ] {
            let m = renewal_function_estimate(gap, 200.0, 200, &mut rng);
            assert!(
                (m / (200.0 / want) - 1.0).abs() < 0.05,
                "{name}: the renewal function is {m} against {}",
                200.0 / want
            );
        }
        // A Cox process is over-dispersed: the randomness of the rate adds
        // to the randomness of the count.
        let rate_dist = |r: &mut Rng| if r.next_f64() < 0.5 { 1.0 } else { 9.0 };
        let cox: Vec<f64> =
            (0..15_000).map(|_| cox_process(&rate_dist, 4.0, &mut rng).len() as f64).collect();
        let m = mean(&cox);
        assert!((m - 20.0).abs() < 0.5, "the Cox mean is {m}");
        // Variance is the mean plus the variance the rate contributes:
        // 20 + 16 * 16 = 276.
        assert!(
            variance(&cox) > 3.0 * m,
            "a Cox process should be over-dispersed: variance {} against mean {m}",
            variance(&cox)
        );
    }

    /// Compound Poisson sums have the mean and variance Wald's identity
    /// gives.
    #[test]
    fn compound_poisson_sums_match_walds_identity() {
        let mut rng = Rng::new(0x_C044);
        let rate = 3.0f64;
        let t_end = 5.0f64;
        // Marks with a known mean and second moment.
        let jump = |r: &mut Rng| 2.0 + r.next_gaussian();
        let totals: Vec<f64> = (0..25_000)
            .map(|_| compound_poisson(rate, &jump, t_end, &mut rng).iter().map(|&(_, j)| j).sum())
            .collect();
        let lambda_t = rate * t_end;
        // E[S] = lambda t E[J], and Var[S] = lambda t E[J^2] -- the second
        // moment, not the variance, because the number of terms is random.
        let want_mean = lambda_t * 2.0;
        let want_var = lambda_t * (2.0f64 * 2.0 + 1.0);
        assert!(
            (mean(&totals) - want_mean).abs() < 0.15,
            "the mean is {} against {want_mean}",
            mean(&totals)
        );
        assert!(
            (variance(&totals) / want_var - 1.0).abs() < 0.05,
            "the variance is {} against {want_var}",
            variance(&totals)
        );
        // The times are a Poisson process and the marks are attached in
        // order.
        let one = compound_poisson(rate, &jump, t_end, &mut rng);
        assert!(one.windows(2).all(|w| w[0].0 < w[1].0), "the marked times are out of order");
    }

    /// Extinction probabilities match the generating function's fixed point,
    /// and simulation agrees with the closed form.
    #[test]
    fn branching_processes_go_extinct_as_predicted() {
        // Subcritical and critical processes die out with probability one --
        // including the critical case, where the population replaces itself
        // on average.
        assert!((extinction_probability(&[0.6, 0.4]) - 1.0).abs() < 1e-12);
        assert!((extinction_probability(&[0.5, 0.5]) - 1.0).abs() < 1e-12);
        assert!((extinction_probability(&[0.25, 0.5, 0.25]) - 1.0).abs() < 1e-12, "critical");
        // A supercritical one has a fixed point strictly inside.
        // For p0 = 1/4, p2 = 3/4 the equation q = 1/4 + 3/4 q^2 gives 1/3.
        let q = extinction_probability(&[0.25, 0.0, 0.75]);
        assert!((q - 1.0 / 3.0).abs() < 1e-9, "the fixed point came out at {q}");
        // It really is a fixed point of the generating function.
        for pmf in [
            vec![0.25f64, 0.0, 0.75],
            vec![0.3, 0.2, 0.5],
            vec![0.1, 0.1, 0.3, 0.5],
        ] {
            let q = extinction_probability(&pmf);
            let g: f64 = pmf.iter().enumerate().map(|(k, &p)| p * q.powi(k as i32)).sum();
            assert!((g - q).abs() < 1e-9, "g({q}) is {g}, so it is not a fixed point");
            // And the smallest one: nothing below it is fixed.
            if q > 1e-6 {
                let below = q * 0.5;
                let gb: f64 = pmf.iter().enumerate().map(|(k, &p)| p * below.powi(k as i32)).sum();
                assert!(gb > below, "a smaller fixed point exists");
            }
        }

        // Simulation agrees with the closed form.
        let mut rng = Rng::new(0x_6704);
        let pmf = [0.25f64, 0.0, 0.75];
        let want = extinction_probability(&pmf);
        // Twelve thousand trials give a standard error of about four
        // thousandths on a probability near a third, so two hundredths is
        // roughly five of them.
        let trials = 12_000;
        let extinct = (0..trials)
            .filter(|_| *branching_process_gw(&pmf, 40, &mut rng).last().expect("non-empty") == 0)
            .count() as f64
            / trials as f64;
        assert!(
            (extinct - want).abs() < 0.02,
            "{extinct} of the lineages died out against a predicted {want}"
        );
        // A subcritical process dies out essentially always.
        let sub = [0.7f64, 0.3];
        let survived = (0..2_000)
            .filter(|_| *branching_process_gw(&sub, 60, &mut rng).last().expect("non-empty") > 0)
            .count();
        assert_eq!(survived, 0, "a subcritical process survived {survived} times");
        // Extinction is absorbing: once zero, always zero.
        for _ in 0..200 {
            let path = branching_process_gw(&pmf, 40, &mut rng);
            if let Some(first_zero) = path.iter().position(|&v| v == 0) {
                assert!(path[first_zero..].iter().all(|&v| v == 0), "a lineage came back");
            }
        }
        assert!(std::panic::catch_unwind(|| extinction_probability(&[0.5, 0.6])).is_err());
    }

    /// The continuous-time birth and death processes have the growth and the
    /// extinction their rates dictate.
    #[test]
    fn birth_and_death_processes_grow_and_die_as_predicted() {
        let mut rng = Rng::new(0x_B124);
        // A Yule process grows exponentially: the population at time t has
        // mean exp(rate t).
        let rate = 0.5f64;
        let t = 4.0f64;
        let sizes: Vec<f64> = (0..15_000)
            .map(|_| 1.0 + yule_process(rate, t, &mut rng).len() as f64)
            .collect();
        let want = (rate * t).exp();
        assert!(
            (mean(&sizes) / want - 1.0).abs() < 0.05,
            "the Yule population is {} against {want}",
            mean(&sizes)
        );
        // It is geometric, so the variance is exp(2rt) - exp(rt).
        let want_var = (2.0 * rate * t).exp() - want;
        assert!(
            (variance(&sizes) / want_var - 1.0).abs() < 0.1,
            "the Yule variance is {} against {want_var}",
            variance(&sizes)
        );
        // Births are ordered and inside the horizon.
        let one = yule_process(rate, t, &mut rng);
        assert!(one.windows(2).all(|w| w[0] < w[1]));
        assert!(one.iter().all(|&s| s > 0.0 && s <= t));

        // A birth-death process started at n0 dies out with probability
        // (death / birth)^n0 when births outpace deaths.
        let (birth, death, n0) = (1.0f64, 0.4f64, 2u64);
        let want_extinct = (death / birth).powi(n0 as i32);
        let trials = 3_000;
        let extinct = (0..trials)
            .filter(|_| {
                let path = birth_death_simulate(birth, death, n0, 60.0, &mut rng);
                path.last().expect("non-empty").1 == 0
            })
            .count() as f64
            / trials as f64;
        assert!(
            (extinct - want_extinct).abs() < 0.03,
            "{extinct} died out against a predicted {want_extinct}"
        );
        // With deaths outpacing births, extinction is certain.
        let doomed = (0..500)
            .filter(|_| {
                birth_death_simulate(0.3, 1.0, 3, 300.0, &mut rng).last().expect("non-empty").1 == 0
            })
            .count();
        assert_eq!(doomed, 500, "a subcritical birth-death process survived");
        // The population moves by one at each event, and the times increase.
        let path = birth_death_simulate(1.0, 0.9, 5, 50.0, &mut rng);
        assert!(path.windows(2).all(|w| w[0].0 < w[1].0), "the event times are out of order");
        assert!(
            path.windows(2).all(|w| w[0].1.abs_diff(w[1].1) == 1),
            "the population jumped by more than one"
        );
    }
}
