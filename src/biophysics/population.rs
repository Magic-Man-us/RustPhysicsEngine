//! Population dynamics and population genetics: growth laws, interacting
//! species, age-structured projection, discrete maps, and the drift,
//! selection and coalescent theory that describes gene frequencies.
//!
//! # Two kinds of model, and why they disagree
//!
//! The deterministic models here describe a population large enough that
//! averages are the whole story. The genetic models mostly do not: drift is
//! the *variance* introduced by finite sampling, and it vanishes from any
//! model that tracks only the mean. A Wright-Fisher population's expected
//! allele frequency never changes at all, and yet every such population
//! eventually fixes one allele or the other -- so the mean is not merely an
//! approximation here, it is silent about the outcome. Where a function
//! reports an expectation, it says so.
//!
//! # Units
//!
//! Times are in whatever unit the caller uses for rates. Genetic models work
//! in generations, and `n` is the number of *diploid* individuals unless a
//! function says otherwise, so a population of `n` carries `2n` gene copies
//! -- the factor that makes heterozygosity decay as `1 - 1/(2n)` rather than
//! `1 - 1/n`.

use crate::biophysics::integrate_adaptive;
use crate::error::GeomError;
use crate::linalg::Matrix;
use crate::monte_carlo::Rng;
use crate::statistics::inference::{chi_squared_gof, TestResult};

// ---------------------------------------------------------------------------
// Single-species growth
// ---------------------------------------------------------------------------

/// Logistic growth in closed form: `N = K N0 e^(rt) / (K + N0 (e^(rt) - 1))`.
///
/// Evaluated from the analytic solution rather than integrated, so it is
/// exact at every time and costs nothing at large `t`.
///
/// # Errors
/// Returns an error for a non-positive carrying capacity or a negative
/// initial population.
pub fn logistic_growth(r: f64, k: f64, n0: f64, t: f64) -> Result<f64, GeomError> {
    if !(k > 0.0) || n0 < 0.0 {
        return Err(GeomError::InvalidArgument("logistic_growth: bad parameters"));
    }
    if n0 == 0.0 {
        return Ok(0.0);
    }
    let growth = (r * t).exp();
    // Written to avoid overflow at large rt: divide through by e^(rt).
    if growth.is_infinite() {
        return Ok(k);
    }
    Ok(k * n0 * growth / (k + n0 * (growth - 1.0)))
}

/// Gompertz growth: `N = K exp(ln(N0/K) e^(-rt))`.
///
/// Differs from the logistic in where it turns: the inflection is at `K/e`,
/// about 37 per cent of capacity, rather than at half. That asymmetry is why
/// it fits tumour and organ growth better than the logistic does -- those
/// slow down earlier than a symmetric curve allows.
///
/// # Errors
/// Returns an error for a non-positive capacity or initial population.
pub fn gompertz(r: f64, k: f64, n0: f64, t: f64) -> Result<f64, GeomError> {
    if !(k > 0.0) || !(n0 > 0.0) {
        return Err(GeomError::InvalidArgument("gompertz: bad parameters"));
    }
    Ok(k * ((n0 / k).ln() * (-r * t).exp()).exp())
}

/// Richards growth, which contains both: `nu = 1` is logistic and the limit
/// `nu -> 0` is Gompertz.
///
/// `N = K (1 + q e^(-r t))^(-1/nu)` with `q = (K/N0)^nu - 1`, solving
/// `dN/dt = (r/nu) N (1 - (N/K)^nu)`.
///
/// The `r/nu` in that equation is not decoration, and writing the solution
/// with `e^(-r nu t)` instead -- which solves the tidier-looking
/// `dN/dt = r N (1 - (N/K)^nu)` -- destroys the Gompertz limit. Under that
/// convention the effective rate is `r nu`, so letting `nu -> 0` at fixed
/// `r` freezes the curve at its initial value rather than approaching
/// anything. Here `r` is the intrinsic rate in both limits, which is what
/// makes the family a genuine interpolation rather than two special cases
/// with a gap between them.
///
/// # Errors
/// Returns an error for a non-positive capacity, initial population or
/// shape.
pub fn richards(r: f64, k: f64, nu: f64, n0: f64, t: f64) -> Result<f64, GeomError> {
    if !(k > 0.0) || !(n0 > 0.0) || !(nu > 0.0) {
        return Err(GeomError::InvalidArgument("richards: bad parameters"));
    }
    let q = (k / n0).powf(nu) - 1.0;
    Ok(k * (1.0 + q * (-r * t).exp()).powf(-1.0 / nu))
}

/// Growth with a strong Allee effect:
/// `dN/dt = r N (N/A - 1) (1 - N/K)`.
///
/// Below the threshold `A` the growth rate is *negative* and the population
/// collapses however far it is from the capacity. That is the qualitative
/// difference from logistic growth, where any positive population recovers:
/// here there is a point of no return, which is why a species can be
/// committed to extinction while individuals are still alive.
///
/// # Errors
/// Returns an error for a threshold not below the capacity, a negative
/// initial population, or a non-positive end time.
pub fn allee_effect_ode(
    r: f64,
    a: f64,
    k: f64,
    n0: f64,
    t_end: f64,
) -> Result<Vec<(f64, f64)>, GeomError> {
    if !(a > 0.0) || !(k > a) || n0 < 0.0 {
        return Err(GeomError::InvalidArgument("allee_effect_ode: bad parameters"));
    }
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let n = y[0].max(0.0);
        vec![r * n * (n / a - 1.0) * (1.0 - n / k)]
    };
    Ok(integrate_adaptive(derivative, &[n0], t_end, 1e-9)?
        .into_iter()
        .map(|(t, y)| (t, y[0]))
        .collect())
}

// ---------------------------------------------------------------------------
// Interacting species
// ---------------------------------------------------------------------------

/// The Lotka-Volterra predator-prey system, with its conserved quantity.
///
/// `dx/dt = alpha x - beta x y`, `dy/dt = delta x y - gamma y`. Returns
/// `(time, prey, predator)` together with
/// `V = delta x - gamma ln x + beta y - alpha ln y`, which is constant along
/// every orbit.
///
/// That constant is the reason the orbits are closed curves rather than a
/// limit cycle: the system is conservative, so its amplitude is set by where
/// it started and never forgets. A model that damped onto a single cycle
/// would be a different system, and returning the invariant lets a caller
/// see the integrator's drift rather than take it on trust.
///
/// # Errors
/// Returns an error for non-positive rates or a non-positive initial
/// population, for which the invariant is undefined.
pub fn lotka_volterra(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    x0: f64,
    y0: f64,
    t_end: f64,
) -> Result<(Vec<(f64, f64, f64)>, Vec<f64>), GeomError> {
    if !(alpha > 0.0) || !(beta > 0.0) || !(delta > 0.0) || !(gamma > 0.0) {
        return Err(GeomError::InvalidArgument("lotka_volterra: the rates must be positive"));
    }
    if !(x0 > 0.0) || !(y0 > 0.0) {
        return Err(GeomError::InvalidArgument("both populations must start positive"));
    }
    let derivative = move |v: &[f64]| -> Vec<f64> {
        let (x, y) = (v[0].max(0.0), v[1].max(0.0));
        vec![alpha * x - beta * x * y, delta * x * y - gamma * y]
    };
    let raw = integrate_adaptive(derivative, &[x0, y0], t_end, 1e-10)?;
    let invariant = raw
        .iter()
        .map(|(_, v)| {
            let (x, y) = (v[0].max(1e-300), v[1].max(1e-300));
            delta * x - gamma * x.ln() + beta * y - alpha * y.ln()
        })
        .collect();
    Ok((raw.into_iter().map(|(t, v)| (t, v[0], v[1])).collect(), invariant))
}

/// The Rosenzweig-MacArthur predator-prey model: logistic prey with a
/// saturating (Holling type II) predator response.
///
/// `dx/dt = r x (1 - x/K) - a x y / (1 + a h x)`,
/// `dy/dt = e a x y / (1 + a h x) - m y`.
///
/// The saturating response is what produces the *paradox of enrichment*:
/// raising the prey's carrying capacity destabilises the coexistence
/// equilibrium into a limit cycle of growing amplitude, so enriching the
/// system makes extinction more likely rather than less. The plain
/// Lotka-Volterra model, whose response is linear, cannot show this.
///
/// # Errors
/// Returns an error for non-positive parameters or a non-positive initial
/// population.
pub fn rosenzweig_macarthur(
    r: f64,
    k: f64,
    attack: f64,
    handling: f64,
    efficiency: f64,
    mortality: f64,
    x0: f64,
    y0: f64,
    t_end: f64,
) -> Result<Vec<(f64, f64, f64)>, GeomError> {
    if !(r > 0.0) || !(k > 0.0) || !(attack > 0.0) || handling < 0.0 {
        return Err(GeomError::InvalidArgument("rosenzweig_macarthur: bad parameters"));
    }
    if !(efficiency > 0.0) || !(mortality > 0.0) || !(x0 > 0.0) || !(y0 > 0.0) {
        return Err(GeomError::InvalidArgument("rosenzweig_macarthur: bad parameters"));
    }
    let derivative = move |v: &[f64]| -> Vec<f64> {
        let (x, y) = (v[0].max(0.0), v[1].max(0.0));
        let intake = attack * x / (1.0 + attack * handling * x);
        vec![r * x * (1.0 - x / k) - intake * y, efficiency * intake * y - mortality * y]
    };
    Ok(integrate_adaptive(derivative, &[x0, y0], t_end, 1e-10)?
        .into_iter()
        .map(|(t, v)| (t, v[0], v[1]))
        .collect())
}

/// The prey density at which the Rosenzweig-MacArthur coexistence
/// equilibrium loses stability, `K = (1 + a h x*) / (a h - ...)`, expressed
/// as the critical carrying capacity.
///
/// The equilibrium prey density is `x* = m / (a (e - m h))`, independent of
/// `K`, and the equilibrium is stable while `K < x* + 1/(a h)` and unstable
/// above -- the Hopf bifurcation of the paradox of enrichment.
///
/// # Errors
/// Returns an error for parameters that admit no coexistence equilibrium:
/// the predator must gain more from a prey item than it spends handling it.
pub fn enrichment_critical_capacity(
    attack: f64,
    handling: f64,
    efficiency: f64,
    mortality: f64,
) -> Result<f64, GeomError> {
    if !(attack > 0.0) || !(handling > 0.0) || !(efficiency > 0.0) || !(mortality > 0.0) {
        return Err(GeomError::InvalidArgument("enrichment_critical_capacity: bad parameters"));
    }
    let denominator = attack * (efficiency - mortality * handling);
    if !(denominator > 0.0) {
        return Err(GeomError::Degenerate("no coexistence equilibrium exists"));
    }
    let prey_star = mortality / denominator;
    Ok(prey_star + 1.0 / (attack * handling))
}

/// Which of the four outcomes a two-species Lotka-Volterra competition has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Competition {
    /// Both persist: each limits itself more than it limits the other.
    Coexistence,
    /// Species one excludes species two, from any starting point.
    FirstExcludes,
    /// Species two excludes species one, from any starting point.
    SecondExcludes,
    /// Both exclusion states are stable; which one is reached depends on the
    /// initial densities.
    FounderControl,
}

/// The outcome of two-species competition, from the competition
/// coefficients and capacities alone.
///
/// Coexistence requires each species to limit *itself* more than it limits
/// the other -- `alpha12 < K1/K2` and `alpha21 < K2/K1`. If both
/// inequalities reverse, both exclusion equilibria are stable and the winner
/// is decided by the starting densities rather than by the parameters. This
/// is the content of the competitive exclusion principle, and it is a
/// statement about niche overlap rather than about which species is
/// "stronger".
///
/// # Errors
/// Returns an error for non-positive capacities or negative coefficients.
pub fn coexistence_condition(
    k1: f64,
    k2: f64,
    alpha12: f64,
    alpha21: f64,
) -> Result<Competition, GeomError> {
    if !(k1 > 0.0) || !(k2 > 0.0) || alpha12 < 0.0 || alpha21 < 0.0 {
        return Err(GeomError::InvalidArgument("coexistence_condition: bad parameters"));
    }
    let first_survives = alpha12 < k1 / k2;
    let second_survives = alpha21 < k2 / k1;
    Ok(match (first_survives, second_survives) {
        (true, true) => Competition::Coexistence,
        (true, false) => Competition::FirstExcludes,
        (false, true) => Competition::SecondExcludes,
        (false, false) => Competition::FounderControl,
    })
}

/// Two-species Lotka-Volterra competition, integrated.
///
/// # Errors
/// Returns an error for non-positive capacities or rates, or negative
/// initial densities.
pub fn competition_lv(
    r1: f64,
    r2: f64,
    k1: f64,
    k2: f64,
    alpha12: f64,
    alpha21: f64,
    n1: f64,
    n2: f64,
    t_end: f64,
) -> Result<Vec<(f64, f64, f64)>, GeomError> {
    if !(r1 > 0.0) || !(r2 > 0.0) || !(k1 > 0.0) || !(k2 > 0.0) {
        return Err(GeomError::InvalidArgument("competition_lv: bad parameters"));
    }
    if n1 < 0.0 || n2 < 0.0 || alpha12 < 0.0 || alpha21 < 0.0 {
        return Err(GeomError::InvalidArgument("competition_lv: bad parameters"));
    }
    let derivative = move |v: &[f64]| -> Vec<f64> {
        let (a, b) = (v[0].max(0.0), v[1].max(0.0));
        vec![
            r1 * a * (1.0 - (a + alpha12 * b) / k1),
            r2 * b * (1.0 - (b + alpha21 * a) / k2),
        ]
    };
    Ok(integrate_adaptive(derivative, &[n1, n2], t_end, 1e-9)?
        .into_iter()
        .map(|(t, v)| (t, v[0], v[1]))
        .collect())
}

/// The Levins metapopulation model: `dp/dt = c p (1 - p) - e p`.
///
/// The equilibrium occupancy is `1 - e/c`, and the population persists only
/// while colonisation outpaces extinction. Note what it says about habitat
/// loss: destroying a fraction `D` of patches replaces the equilibrium with
/// `1 - D - e/c`, so a metapopulation goes extinct while a fraction `e/c` of
/// its habitat still remains -- the extinction debt.
///
/// # Errors
/// Returns an error for negative rates or an occupancy outside zero to one.
pub fn metapopulation_levins(
    c: f64,
    e: f64,
    p0: f64,
    t_end: f64,
) -> Result<Vec<(f64, f64)>, GeomError> {
    if c < 0.0 || e < 0.0 || !(0.0..=1.0).contains(&p0) {
        return Err(GeomError::InvalidArgument("metapopulation_levins: bad parameters"));
    }
    let derivative = move |v: &[f64]| -> Vec<f64> {
        let p = v[0].clamp(0.0, 1.0);
        vec![c * p * (1.0 - p) - e * p]
    };
    Ok(integrate_adaptive(derivative, &[p0], t_end, 1e-10)?
        .into_iter()
        .map(|(t, v)| (t, v[0]))
        .collect())
}

// ---------------------------------------------------------------------------
// Age structure
// ---------------------------------------------------------------------------

/// The Leslie projection matrix from age-specific fecundity and survival.
///
/// `fecundity[i]` is the expected offspring of an individual in class `i`
/// over one time step, and `survival[i]` the probability of surviving from
/// class `i` to `i + 1`, so `survival` is one shorter than `fecundity`.
///
/// # Errors
/// Returns an error for empty input, a mismatched length, a negative
/// fecundity, or a survival outside zero to one.
pub fn leslie_matrix(fecundity: &[f64], survival: &[f64]) -> Result<Matrix, GeomError> {
    let classes = fecundity.len();
    if classes == 0 || survival.len() + 1 != classes {
        return Err(GeomError::InvalidArgument("leslie_matrix: mismatched input"));
    }
    if fecundity.iter().any(|f| *f < 0.0) {
        return Err(GeomError::InvalidArgument("fecundity must be non-negative"));
    }
    if survival.iter().any(|s| !(0.0..=1.0).contains(s)) {
        return Err(GeomError::InvalidArgument("survival must be a probability"));
    }
    let mut m = Matrix::zeros(classes, classes);
    for (j, f) in fecundity.iter().enumerate() {
        m.set(0, j, *f);
    }
    for (i, s) in survival.iter().enumerate() {
        m.set(i + 1, i, *s);
    }
    Ok(m)
}

/// The asymptotic growth rate and stable age distribution of a Leslie
/// matrix, by power iteration.
///
/// Returns `(lambda, distribution)` with the distribution normalised to sum
/// to one. Perron-Frobenius guarantees the dominant eigenvalue of a
/// primitive non-negative matrix is real, positive and simple, which is what
/// makes power iteration the right method here rather than a general
/// eigensolver.
///
/// The strong ergodic theorem is the substance: *whatever* age distribution
/// a population starts with, it converges to this one and then grows by
/// `lambda` per step. The transient depends on the start; the asymptote does
/// not.
///
/// # Errors
/// Returns an error for a non-square matrix or one whose iteration does not
/// converge -- which happens when the matrix is imprimitive, for instance a
/// species that reproduces at exactly one age, whose age classes then cycle
/// forever instead of settling.
pub fn leslie_growth_rate(l: &Matrix) -> Result<(f64, Vec<f64>), GeomError> {
    if !l.is_square() || l.rows == 0 {
        return Err(GeomError::InvalidArgument("leslie_growth_rate needs a square matrix"));
    }
    let n = l.rows;
    let mut v = vec![1.0 / n as f64; n];
    let mut lambda = 0.0;
    let mut converged = false;
    for _ in 0..200_000 {
        let next: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| l.get(i, j) * v[j]).sum())
            .collect();
        let norm: f64 = next.iter().sum();
        if !(norm > 0.0) {
            return Err(GeomError::Degenerate("the population dies out entirely"));
        }
        let scaled: Vec<f64> = next.iter().map(|x| x / norm).collect();
        let moved = (0..n).map(|k| (scaled[k] - v[k]).abs()).fold(0.0, f64::max);
        v = scaled;
        lambda = norm;
        if moved < 1e-14 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(GeomError::Degenerate(
            "the age distribution cycles rather than settling: the matrix is imprimitive",
        ));
    }
    Ok((lambda, v))
}

/// The stable age distribution alone.
///
/// # Errors
/// Returns an error on the same conditions as [`leslie_growth_rate`].
pub fn stable_age_distribution(l: &Matrix) -> Result<Vec<f64>, GeomError> {
    Ok(leslie_growth_rate(l)?.1)
}

/// Solves the Euler-Lotka equation `sum l_x m_x r^(-x) = 1` for the growth
/// rate `r` per time step.
///
/// `lx[i]` is survivorship to age `i + 1` and `mx[i]` the fecundity there,
/// so the first entry describes age one. The left side is strictly
/// decreasing in `r`, so bisection cannot fail; it is the same growth rate
/// [`leslie_growth_rate`] finds, reached from the life table rather than
/// from the matrix.
///
/// # Errors
/// Returns an error for mismatched lengths, a survivorship outside zero to
/// one, or a population with no reproduction at all.
pub fn euler_lotka_solve(lx: &[f64], mx: &[f64]) -> Result<f64, GeomError> {
    if lx.is_empty() || lx.len() != mx.len() {
        return Err(GeomError::InvalidArgument("euler_lotka_solve: mismatched input"));
    }
    if lx.iter().any(|s| !(0.0..=1.0).contains(s)) || mx.iter().any(|f| *f < 0.0) {
        return Err(GeomError::InvalidArgument("euler_lotka_solve: bad life table"));
    }
    let net: f64 = lx.iter().zip(mx).map(|(l, m)| l * m).sum();
    if !(net > 0.0) {
        return Err(GeomError::Degenerate("the population never reproduces"));
    }
    let f = |r: f64| -> f64 {
        lx.iter()
            .zip(mx)
            .enumerate()
            .map(|(k, (l, m))| l * m * r.powi(-(k as i32 + 1)))
            .sum::<f64>()
            - 1.0
    };
    // Strictly decreasing in r, so any bracket that straddles one works.
    let (mut lo, mut hi) = (1e-8f64, 1.0f64);
    while f(hi) > 0.0 && hi < 1e8 {
        hi *= 2.0;
    }
    if f(hi) > 0.0 {
        return Err(GeomError::Degenerate("the growth rate exceeds the search range"));
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// Discrete maps
// ---------------------------------------------------------------------------

/// The Ricker map `N -> N exp(r (1 - N/K))`, iterated.
///
/// Overcompensating density dependence: a population far above capacity
/// crashes below it rather than settling, and as `r` grows the fixed point
/// period-doubles into chaos. That a deterministic single-species model with
/// no environmental variation produces apparently random fluctuations is the
/// point -- population data need not be noisy to look noisy.
///
/// # Errors
/// Returns an error for a non-positive capacity or negative start.
pub fn ricker_map(r: f64, k: f64, n0: f64, steps: usize) -> Result<Vec<f64>, GeomError> {
    if !(k > 0.0) || n0 < 0.0 || steps > 10_000_000 {
        return Err(GeomError::InvalidArgument("ricker_map: bad parameters"));
    }
    let mut n = n0;
    let mut out = Vec::with_capacity(steps + 1);
    out.push(n);
    for _ in 0..steps {
        n = (n * (r * (1.0 - n / k)).exp()).min(1e300);
        out.push(n);
    }
    Ok(out)
}

/// The Beverton-Holt map `N -> R N / (1 + (R - 1) N / K)`, iterated.
///
/// Compensating rather than overcompensating: however far above capacity the
/// population starts it approaches `K` monotonically and never overshoots,
/// so unlike Ricker it has no route to chaos at any `R`. The two models
/// differ in nothing but the shape of the density dependence, and that
/// single difference is the whole distinction between a stable fishery model
/// and a chaotic one.
///
/// It also has a closed-form solution, which is what the tests check against.
///
/// # Errors
/// Returns an error for a growth ratio at or below one, a non-positive
/// capacity, or a negative start.
pub fn beverton_holt(ratio: f64, k: f64, n0: f64, steps: usize) -> Result<Vec<f64>, GeomError> {
    if !(ratio > 1.0) || !(k > 0.0) || n0 < 0.0 || steps > 10_000_000 {
        return Err(GeomError::InvalidArgument("beverton_holt: bad parameters"));
    }
    let mut n = n0;
    let mut out = Vec::with_capacity(steps + 1);
    out.push(n);
    for _ in 0..steps {
        n = ratio * n / (1.0 + (ratio - 1.0) * n / k);
        out.push(n);
    }
    Ok(out)
}

/// The attractor of the Ricker map at each of a range of growth rates: the
/// bifurcation diagram.
///
/// Returns `(r, attractor points)` per rate, with the transient discarded
/// and the remaining points deduplicated so a period-`p` cycle reports `p`
/// values.
///
/// # Errors
/// Returns an error for an empty or descending range, or bad map parameters.
pub fn bifurcation_ricker(
    r_lo: f64,
    r_hi: f64,
    samples: usize,
    transient: usize,
    keep: usize,
) -> Result<Vec<(f64, Vec<f64>)>, GeomError> {
    if !(r_hi > r_lo) || samples < 2 || keep == 0 || keep > 4_096 {
        return Err(GeomError::InvalidArgument("bifurcation_ricker: bad range"));
    }
    (0..samples)
        .map(|s| {
            let r = r_lo + (r_hi - r_lo) * s as f64 / (samples - 1) as f64;
            let trace = ricker_map(r, 1.0, 0.6, transient + keep)?;
            let mut tail: Vec<f64> = trace[transient..].to_vec();
            tail.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            tail.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            Ok((r, tail))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Drift
// ---------------------------------------------------------------------------

/// A Wright-Fisher allele-frequency trajectory: each generation resamples
/// `2n` gene copies binomially from the previous frequency.
///
/// The expected frequency never changes -- drift is a martingale -- and yet
/// every trajectory eventually fixes at zero or one. That is the whole point
/// of the model, and the reason no deterministic account of it is possible:
/// the mean is constant while the outcome is certain to be extreme.
///
/// # Errors
/// Returns an error for no individuals or a frequency outside zero to one.
pub fn wright_fisher(
    n: u64,
    p0: f64,
    generations: usize,
    rng: &mut Rng,
) -> Result<Vec<f64>, GeomError> {
    if n == 0 || !(0.0..=1.0).contains(&p0) || generations > 10_000_000 {
        return Err(GeomError::InvalidArgument("wright_fisher: bad parameters"));
    }
    let copies = 2 * n;
    let mut count = (p0 * copies as f64).round() as u64;
    let mut out = Vec::with_capacity(generations + 1);
    out.push(count as f64 / copies as f64);
    for _ in 0..generations {
        let p = count as f64 / copies as f64;
        // Binomial by direct sampling; the populations here are small
        // enough that this is cheaper than a rejection method and it is
        // exact at any size.
        count = (0..copies).filter(|_| rng.next_f64() < p).count() as u64;
        out.push(count as f64 / copies as f64);
        if count == 0 || count == copies {
            // Fixed: the frequency cannot change again, so the rest of the
            // trajectory is a constant and is filled in directly.
            let final_p = count as f64 / copies as f64;
            while out.len() <= generations {
                out.push(final_p);
            }
            break;
        }
    }
    Ok(out)
}

/// A Moran process: one birth and one death per step, with the mutant type
/// having relative fitness `r`.
///
/// Returns `(fixed, steps)` -- whether the mutant fixed rather than being
/// lost, and how many steps it took. Unlike Wright-Fisher the population
/// overlaps generations, and the fixation probability has an exact closed
/// form; see [`fixation_probability_moran`].
///
/// # Errors
/// Returns an error for an empty population, a starting count above it, or a
/// non-positive fitness.
pub fn moran_process(
    n: u64,
    i0: u64,
    fitness: f64,
    rng: &mut Rng,
) -> Result<(bool, u64), GeomError> {
    if n == 0 || i0 > n || !(fitness > 0.0) {
        return Err(GeomError::InvalidArgument("moran_process: bad parameters"));
    }
    let mut i = i0;
    let mut steps = 0u64;
    while i > 0 && i < n {
        let mutants = i as f64;
        let residents = (n - i) as f64;
        let total_fitness = fitness * mutants + residents;
        // A mutant is born with probability proportional to its fitness
        // share, and a uniformly chosen individual dies.
        let mutant_born = rng.next_f64() * total_fitness < fitness * mutants;
        let mutant_dies = rng.next_f64() * (n as f64) < mutants;
        match (mutant_born, mutant_dies) {
            (true, false) => i += 1,
            (false, true) => i -= 1,
            _ => {}
        }
        steps += 1;
        if steps > 200_000_000 {
            return Err(GeomError::Degenerate("the Moran process did not absorb"));
        }
    }
    Ok((i == n, steps))
}

/// The exact fixation probability of `i` mutants of relative fitness `r` in
/// a Moran population of `n`: `(1 - r^-i) / (1 - r^-n)`.
///
/// At `r = 1` it degenerates to `i/n` -- a neutral mutant fixes with
/// probability equal to its initial frequency, which is the cleanest
/// statement of what drift alone does. A single advantageous mutant with
/// `r = 1.01` fixes with probability about `1/100` rather than the certainty
/// a deterministic model would predict: even a beneficial mutation is
/// usually lost.
///
/// # Errors
/// Returns an error for an empty population, a count above it, or a
/// non-positive fitness.
pub fn fixation_probability_moran(n: u64, i: u64, r: f64) -> Result<f64, GeomError> {
    if n == 0 || i > n || !(r > 0.0) {
        return Err(GeomError::InvalidArgument("fixation_probability_moran: bad parameters"));
    }
    if i == 0 {
        return Ok(0.0);
    }
    if i == n {
        return Ok(1.0);
    }
    if (r - 1.0).abs() < 1e-12 {
        return Ok(i as f64 / n as f64);
    }
    let inverse = 1.0 / r;
    Ok((1.0 - inverse.powi(i as i32)) / (1.0 - inverse.powi(n as i32)))
}

/// The expected heterozygosity after `t` generations of drift:
/// `H_t = H_0 (1 - 1/(2N))^t`.
///
/// The `2N` rather than `N` is the diploid gene copy count, and getting it
/// wrong halves the predicted rate of decay. Variation is lost at a rate set
/// by the population size alone -- no selection is involved -- which is why
/// small populations lose diversity even when nothing is wrong with them.
///
/// # Errors
/// Returns an error for an empty population or a heterozygosity outside zero
/// to one.
pub fn genetic_drift_heterozygosity(n: u64, h0: f64, t: f64) -> Result<f64, GeomError> {
    if n == 0 || !(0.0..=1.0).contains(&h0) || t < 0.0 {
        return Err(GeomError::InvalidArgument("genetic_drift_heterozygosity: bad parameters"));
    }
    Ok(h0 * (1.0 - 1.0 / (2.0 * n as f64)).powf(t))
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Hardy-Weinberg genotype frequencies `(p^2, 2pq, q^2)`.
///
/// # Errors
/// Returns an error for an allele frequency outside zero to one.
pub fn hardy_weinberg(p: f64) -> Result<(f64, f64, f64), GeomError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(GeomError::InvalidArgument("the allele frequency must be in [0, 1]"));
    }
    let q = 1.0 - p;
    Ok((p * p, 2.0 * p * q, q * q))
}

/// A chi-squared test of observed genotype counts against Hardy-Weinberg
/// proportions, with the allele frequency estimated from the same data.
///
/// One degree of freedom, not two: estimating `p` from the counts costs one,
/// which is why the standard `k - 1` rule does not apply here. Reported
/// through [`chi_squared_gof`], whose degrees of freedom are corrected
/// afterwards.
///
/// # Errors
/// Returns an error for a negative count or an empty sample.
pub fn hw_chi_square_test(observed: [f64; 3]) -> Result<TestResult, GeomError> {
    if observed.iter().any(|c| *c < 0.0) {
        return Err(GeomError::InvalidArgument("the counts must be non-negative"));
    }
    let total: f64 = observed.iter().sum();
    if !(total > 0.0) {
        return Err(GeomError::InvalidArgument("the sample is empty"));
    }
    // p estimated from the allele counts, which is what makes this one
    // degree of freedom rather than two.
    let p = (2.0 * observed[0] + observed[1]) / (2.0 * total);
    let (aa, ab, bb) = hardy_weinberg(p)?;
    let expected = [aa * total, ab * total, bb * total];
    if expected.iter().any(|e| *e <= 0.0) {
        return Err(GeomError::Degenerate("an allele is absent, so the test does not apply"));
    }
    let mut result = chi_squared_gof(&observed, &expected);
    // chi_squared_gof assumes k - 1 = 2; one parameter was estimated.
    result.df = 1.0;
    result.p_value = 1.0 - crate::special::gamma::gamma_p(0.5, result.statistic / 2.0);
    Ok(result)
}

/// One generation at a time of selection at a single diploid locus, with
/// genotype fitnesses `[w_AA, w_Aa, w_aa]`.
///
/// Returns the allele frequency each generation. Which allele wins is not
/// decided by fitness alone: with heterozygote advantage neither fixes and
/// the population settles at a polymorphic equilibrium, while with
/// heterozygote *disadvantage* both fixations are stable and the outcome
/// depends on where it started. Directional selection is only one of three
/// possibilities.
///
/// # Errors
/// Returns an error for a frequency outside zero to one, a negative fitness,
/// or a population with no viable genotype.
pub fn selection_one_locus(
    p0: f64,
    w: [f64; 3],
    generations: usize,
) -> Result<Vec<f64>, GeomError> {
    if !(0.0..=1.0).contains(&p0) || w.iter().any(|x| *x < 0.0) || generations > 10_000_000 {
        return Err(GeomError::InvalidArgument("selection_one_locus: bad parameters"));
    }
    if w.iter().all(|x| *x == 0.0) {
        return Err(GeomError::Degenerate("no genotype is viable"));
    }
    let mut p = p0;
    let mut out = Vec::with_capacity(generations + 1);
    out.push(p);
    for _ in 0..generations {
        let q = 1.0 - p;
        let mean = w[0] * p * p + w[1] * 2.0 * p * q + w[2] * q * q;
        if !(mean > 0.0) {
            // Every surviving genotype has died out; the frequency is
            // undefined from here and is held rather than invented.
            while out.len() <= generations {
                out.push(p);
            }
            break;
        }
        p = (w[0] * p * p + w[1] * p * q) / mean;
        out.push(p);
    }
    Ok(out)
}

/// The polymorphic equilibrium of a locus with heterozygote advantage:
/// `p* = (w_Aa - w_aa) / (2 w_Aa - w_AA - w_aa)`.
///
/// # Errors
/// Returns an error unless the heterozygote is strictly the fittest, in
/// which case there is no interior equilibrium to report.
pub fn balanced_polymorphism(w: [f64; 3]) -> Result<f64, GeomError> {
    if !(w[1] > w[0] && w[1] > w[2]) {
        return Err(GeomError::InvalidArgument(
            "an interior equilibrium needs heterozygote advantage",
        ));
    }
    Ok((w[1] - w[2]) / (2.0 * w[1] - w[0] - w[2]))
}

/// The equilibrium frequency of a deleterious allele maintained by
/// mutation.
///
/// For a fully recessive allele the balance is `sqrt(mu/s)`; with any
/// dominance `h > 0` it is `mu/(h s)` instead. The difference is large: at
/// `mu = 1e-6` and `s = 0.1` a recessive allele sits at 0.32 per cent while
/// one with `h = 0.1` sits at 0.01 per cent, some thirty times rarer.
/// Selection acts on heterozygotes far more often than on the rare
/// homozygote, so even slight dominance dominates the balance.
///
/// # Errors
/// Returns an error for a non-positive selection coefficient, a negative
/// mutation rate, or a dominance outside zero to one.
pub fn mutation_selection_balance(mu: f64, s: f64, h: f64) -> Result<f64, GeomError> {
    if mu < 0.0 || !(s > 0.0) || !(0.0..=1.0).contains(&h) {
        return Err(GeomError::InvalidArgument("mutation_selection_balance: bad parameters"));
    }
    if h * s <= mu {
        // Dominance too weak to matter: the recessive form applies.
        return Ok((mu / s).sqrt().min(1.0));
    }
    Ok((mu / (h * s)).min(1.0))
}

/// Hamilton's rule: an altruistic act spreads when `r b > c`.
///
/// # Errors
/// Returns an error for a relatedness outside zero to one.
pub fn kin_selection_hamilton(r: f64, b: f64, c: f64) -> Result<bool, GeomError> {
    if !(0.0..=1.0).contains(&r) {
        return Err(GeomError::InvalidArgument("relatedness must be in [0, 1]"));
    }
    Ok(r * b > c)
}

/// The Price equation, decomposing the change in a mean trait into
/// selection and transmission.
///
/// Returns `(selection, transmission)` with
/// `selection = Cov(w, z) / w_bar` and
/// `transmission = E[w dz] / w_bar`, whose sum is exactly the change in the
/// mean trait. This is an *identity*, not a model -- it assumes nothing
/// about inheritance or fitness and holds for any population whatever, which
/// is what makes it useful for deciding whether an observed change was
/// selection at all.
///
/// # Errors
/// Returns an error for mismatched lengths, an empty population, a negative
/// fitness, or a mean fitness of zero.
pub fn price_equation_decompose(
    trait_values: &[f64],
    fitness: &[f64],
    offspring_trait: &[f64],
) -> Result<(f64, f64), GeomError> {
    let n = trait_values.len();
    if n == 0 || fitness.len() != n || offspring_trait.len() != n {
        return Err(GeomError::InvalidArgument("price_equation_decompose: mismatched input"));
    }
    if fitness.iter().any(|w| *w < 0.0) {
        return Err(GeomError::InvalidArgument("fitness must be non-negative"));
    }
    let count = n as f64;
    let mean_w: f64 = fitness.iter().sum::<f64>() / count;
    if !(mean_w > 0.0) {
        return Err(GeomError::Degenerate("the population left no offspring"));
    }
    let mean_z: f64 = trait_values.iter().sum::<f64>() / count;
    let covariance: f64 = (0..n)
        .map(|k| (fitness[k] - mean_w) * (trait_values[k] - mean_z))
        .sum::<f64>()
        / count;
    let transmission: f64 = (0..n)
        .map(|k| fitness[k] * (offspring_trait[k] - trait_values[k]))
        .sum::<f64>()
        / count;
    Ok((covariance / mean_w, transmission / mean_w))
}

// ---------------------------------------------------------------------------
// The coalescent and sequence diversity
// ---------------------------------------------------------------------------

/// The expected time, in generations, during which a sample of `n` lineages
/// has exactly `k` ancestors: `E[T_k] = 4N / (k (k - 1))`.
///
/// The `4N` is the diploid gene-copy convention: there are `2N` copies, and
/// the coalescence rate for `k` lineages is `C(k,2) / (2N)`. The
/// distribution's shape is the striking part -- `T_2` alone is `2N`
/// generations, longer than every other interval put together, so the
/// genealogy of a sample is dominated by its deepest branch and estimates of
/// ancient history rest on very little independent information.
///
/// # Errors
/// Returns an error for fewer than two lineages or an empty population.
pub fn coalescent_time_expected(n: u64, k: u64) -> Result<f64, GeomError> {
    if n == 0 || k < 2 {
        return Err(GeomError::InvalidArgument("coalescent_time_expected: bad parameters"));
    }
    Ok(4.0 * n as f64 / (k as f64 * (k as f64 - 1.0)))
}

/// The expected time to the most recent common ancestor of a sample of `k`:
/// `4N (1 - 1/k)` generations.
///
/// Bounded above by `4N` however large the sample: adding sequences barely
/// deepens the tree, because new lineages coalesce almost immediately with
/// the ones already there. Sampling more individuals buys resolution near
/// the tips and almost nothing at the root.
///
/// # Errors
/// Returns an error for fewer than two lineages or an empty population.
pub fn coalescent_tmrca_expected(n: u64, k: u64) -> Result<f64, GeomError> {
    if n == 0 || k < 2 {
        return Err(GeomError::InvalidArgument("coalescent_tmrca_expected: bad parameters"));
    }
    Ok(4.0 * n as f64 * (1.0 - 1.0 / k as f64))
}

/// One realisation of the coalescent: the waiting times, in generations,
/// while the sample has `k, k-1, ..., 2` ancestors.
///
/// Returns the intervals in that order, so the total tree height is their
/// sum. Each is exponential with rate `C(k,2)/(2N)`.
///
/// The tree *topology* belongs with the phylogenetics module; this reports
/// the times, which is what the diversity statistics here need.
///
/// # Errors
/// Returns an error for fewer than two lineages or an empty population.
pub fn coalescent_simulate(
    n: u64,
    samples: u64,
    rng: &mut Rng,
) -> Result<Vec<f64>, GeomError> {
    if n == 0 || !(2..=100_000).contains(&samples) {
        return Err(GeomError::InvalidArgument("coalescent_simulate: bad parameters"));
    }
    Ok((2..=samples)
        .rev()
        .map(|k| {
            let rate = k as f64 * (k as f64 - 1.0) / 2.0 / (2.0 * n as f64);
            -(1.0 - rng.next_f64()).ln() / rate
        })
        .collect())
}

/// The `n`-th harmonic-like sum `a_n = sum_{i=1}^{n-1} 1/i`, the
/// normalising constant of Watterson's estimator.
fn watterson_a(n: u64) -> f64 {
    (1..n).map(|i| 1.0 / i as f64).sum()
}

/// Watterson's estimator of `theta = 4 N mu` from the number of segregating
/// sites: `theta_W = S / a_n`.
///
/// The division by `a_n` rather than by `n` is the whole content: the number
/// of segregating sites grows only logarithmically with the sample, because
/// each additional sequence adds a shorter and shorter branch to the
/// genealogy. Dividing by the sample size would make the estimate fall
/// steadily as more data arrived.
///
/// # Errors
/// Returns an error for fewer than two sequences or a negative site count.
pub fn watterson_theta(segregating: f64, n: u64) -> Result<f64, GeomError> {
    if n < 2 || segregating < 0.0 {
        return Err(GeomError::InvalidArgument("watterson_theta: bad parameters"));
    }
    Ok(segregating / watterson_a(n))
}

/// Nucleotide diversity `pi`: the mean number of differences between a pair
/// of sequences.
///
/// # Errors
/// Returns an error for fewer than two sequences or sequences of differing
/// length.
pub fn nucleotide_diversity(sequences: &[Vec<u8>]) -> Result<f64, GeomError> {
    if sequences.len() < 2 {
        return Err(GeomError::InvalidArgument("nucleotide_diversity needs two sequences"));
    }
    let length = sequences[0].len();
    if sequences.iter().any(|s| s.len() != length) {
        return Err(GeomError::InvalidArgument("the sequences differ in length"));
    }
    let n = sequences.len();
    let mut total = 0.0;
    let mut pairs = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            total += (0..length).filter(|k| sequences[i][*k] != sequences[j][*k]).count() as f64;
            pairs += 1.0;
        }
    }
    Ok(total / pairs)
}

/// The number of segregating sites in an alignment.
///
/// # Errors
/// Returns an error for fewer than two sequences or sequences of differing
/// length.
pub fn segregating_sites(sequences: &[Vec<u8>]) -> Result<usize, GeomError> {
    if sequences.len() < 2 {
        return Err(GeomError::InvalidArgument("segregating_sites needs two sequences"));
    }
    let length = sequences[0].len();
    if sequences.iter().any(|s| s.len() != length) {
        return Err(GeomError::InvalidArgument("the sequences differ in length"));
    }
    Ok((0..length)
        .filter(|k| sequences.iter().any(|s| s[*k] != sequences[0][*k]))
        .count())
}

/// Tajima's D: the standardised difference between nucleotide diversity and
/// Watterson's estimator.
///
/// Both estimate the same `theta` under neutrality and constant size, so
/// their difference is zero in expectation and any departure is evidence
/// that one of those assumptions fails. The sign carries the interpretation:
/// negative means an excess of rare variants -- a recent expansion or a
/// selective sweep -- and positive means an excess of intermediate ones,
/// as under balancing selection or population structure. It cannot
/// distinguish demography from selection, which is why a significant D is a
/// question rather than an answer.
///
/// # Errors
/// Returns an error for fewer than four sequences, below which the variance
/// is not defined, or for an alignment with no variation.
pub fn tajima_d(sequences: &[Vec<u8>]) -> Result<f64, GeomError> {
    let n = sequences.len() as u64;
    if n < 4 {
        return Err(GeomError::InvalidArgument("tajima_d needs four sequences"));
    }
    let s = segregating_sites(sequences)? as f64;
    if !(s > 0.0) {
        return Err(GeomError::Degenerate("the alignment has no variation"));
    }
    let pi = nucleotide_diversity(sequences)?;
    let a1 = watterson_a(n);
    let a2: f64 = (1..n).map(|i| 1.0 / (i * i) as f64).sum();
    let nf = n as f64;
    let b1 = (nf + 1.0) / (3.0 * (nf - 1.0));
    let b2 = 2.0 * (nf * nf + nf + 3.0) / (9.0 * nf * (nf - 1.0));
    let c1 = b1 - 1.0 / a1;
    let c2 = b2 - (nf + 2.0) / (a1 * nf) + a2 / (a1 * a1);
    let e1 = c1 / a1;
    let e2 = c2 / (a1 * a1 + a2);
    let variance = e1 * s + e2 * s * (s - 1.0);
    if !(variance > 0.0) {
        return Err(GeomError::Degenerate("the variance of D is not positive"));
    }
    Ok((pi - s / a1) / variance.sqrt())
}

/// Wright's `F_ST` from subpopulation allele frequencies:
/// `(H_T - H_S) / H_T`.
///
/// Zero when the subpopulations have identical frequencies and one when each
/// is fixed for a different allele. It measures how much of the total
/// heterozygosity is *lost* by subdivision, so it is a statement about
/// variance in frequency rather than about how different the populations
/// look.
///
/// # Errors
/// Returns an error for fewer than two subpopulations, a frequency outside
/// zero to one, or a set of populations all fixed for the same allele, for
/// which there is no heterozygosity to partition.
pub fn fst(subpop_freqs: &[f64]) -> Result<f64, GeomError> {
    if subpop_freqs.len() < 2 {
        return Err(GeomError::InvalidArgument("fst needs two subpopulations"));
    }
    if subpop_freqs.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return Err(GeomError::InvalidArgument("every frequency must be in [0, 1]"));
    }
    let k = subpop_freqs.len() as f64;
    let mean: f64 = subpop_freqs.iter().sum::<f64>() / k;
    let h_total = 2.0 * mean * (1.0 - mean);
    let h_sub: f64 = subpop_freqs.iter().map(|p| 2.0 * p * (1.0 - p)).sum::<f64>() / k;
    if !(h_total > 0.0) {
        return Err(GeomError::Degenerate("there is no variation to partition"));
    }
    Ok(((h_total - h_sub) / h_total).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------
    // Growth
    // -----------------------------------------------------------------

    #[test]
    fn the_growth_laws_satisfy_the_equations_they_solve() {
        // Each is a closed-form solution of a differential equation, so the
        // check is that it satisfies that equation -- a central difference
        // of the formula against the right-hand side. That tests the algebra
        // rather than remembering a curve.
        let h = 1e-6;
        for &(r, k, n0) in &[(0.5f64, 100.0f64, 1.0f64), (1.5, 10.0, 9.0), (0.2, 1e6, 1e3)] {
            for step in 1..=20 {
                let t = f64::from(step) * 0.4 / r;
                // Logistic: dN/dt = r N (1 - N/K).
                let n = logistic_growth(r, k, n0, t).unwrap();
                let numeric = (logistic_growth(r, k, n0, t + h).unwrap()
                    - logistic_growth(r, k, n0, t - h).unwrap())
                    / (2.0 * h);
                let expected = r * n * (1.0 - n / k);
                assert!(
                    close(numeric, expected, 1e-4 * expected.abs().max(1.0)),
                    "logistic at t = {t}: {numeric} against {expected}"
                );

                // Gompertz: dN/dt = r N ln(K/N).
                let g = gompertz(r, k, n0, t).unwrap();
                let g_numeric = (gompertz(r, k, n0, t + h).unwrap()
                    - gompertz(r, k, n0, t - h).unwrap())
                    / (2.0 * h);
                let g_expected = r * g * (k / g).ln();
                assert!(
                    close(g_numeric, g_expected, 1e-4 * g_expected.abs().max(1.0)),
                    "Gompertz at t = {t}: {g_numeric} against {g_expected}"
                );
            }
            // Both start where they are told and end at capacity.
            assert!(close(logistic_growth(r, k, n0, 0.0).unwrap(), n0, 1e-12));
            assert!(close(gompertz(r, k, n0, 0.0).unwrap(), n0, 1e-9 * n0));
            assert!(close(logistic_growth(r, k, n0, 200.0 / r).unwrap(), k, 1e-6 * k));
            assert!(close(gompertz(r, k, n0, 200.0 / r).unwrap(), k, 1e-6 * k));
        }
        // The inflections differ, which is the whole reason to have both:
        // the logistic turns at K/2 and Gompertz at K/e.
        let (r, k, n0) = (1.0f64, 100.0f64, 1.0f64);
        let fastest = |f: &dyn Fn(f64) -> f64| -> f64 {
            let mut best = (0.0, f64::NEG_INFINITY);
            for step in 0..20_000 {
                let t = f64::from(step) * 0.001;
                let rate = (f(t + h) - f(t - h)) / (2.0 * h);
                if rate > best.1 {
                    best = (t, rate);
                }
            }
            f(best.0)
        };
        assert!(close(
            fastest(&|t| logistic_growth(r, k, n0, t).unwrap()),
            k / 2.0,
            0.5
        ));
        assert!(close(
            fastest(&|t| gompertz(r, k, n0, t).unwrap()),
            k / std::f64::consts::E,
            0.5
        ));
        assert!(logistic_growth(r, 0.0, n0, 1.0).is_err());
        assert!(logistic_growth(r, k, -1.0, 1.0).is_err());
        assert!(gompertz(r, k, 0.0, 1.0).is_err());
        assert!(close(logistic_growth(r, k, 0.0, 5.0).unwrap(), 0.0, 1e-15));
    }

    #[test]
    fn richards_contains_the_logistic_and_approaches_gompertz() {
        // The claim that makes the extra parameter worth having: nu = 1 must
        // reproduce the logistic exactly, and small nu must approach
        // Gompertz. Both are checked as limits rather than asserted.
        let (r, k, n0) = (0.7f64, 50.0f64, 2.0f64);
        for step in 0..=30 {
            let t = f64::from(step) * 0.5;
            assert!(
                close(
                    richards(r, k, 1.0, n0, t).unwrap(),
                    logistic_growth(r, k, n0, t).unwrap(),
                    1e-9 * k
                ),
                "Richards at nu = 1 differs from the logistic at t = {t}"
            );
        }
        // As nu falls the curve approaches Gompertz, monotonically in the
        // worst-case gap.
        let mut previous = f64::INFINITY;
        for shift in 0..5 {
            let nu = 0.1 / f64::from(1 << shift);
            let gap = (0..=30)
                .map(|step| {
                    let t = f64::from(step) * 0.5;
                    (richards(r, k, nu, n0, t).unwrap() - gompertz(r, k, n0, t).unwrap()).abs()
                })
                .fold(0.0f64, f64::max);
            assert!(gap < previous, "a smaller nu moved away from Gompertz: {gap} after {previous}");
            previous = gap;
        }
        assert!(previous < 0.02 * k, "the Gompertz limit was not reached: {previous}");
        // And the solution solves the equation it claims to:
        // dN/dt = (r/nu) N (1 - (N/K)^nu).
        let h = 1e-6;
        for &nu in &[0.4f64, 1.0, 2.5] {
            for step in 1..=15 {
                let t = f64::from(step) * 0.4;
                let n = richards(r, k, nu, n0, t).unwrap();
                let numeric = (richards(r, k, nu, n0, t + h).unwrap()
                    - richards(r, k, nu, n0, t - h).unwrap())
                    / (2.0 * h);
                let expected = (r / nu) * n * (1.0 - (n / k).powf(nu));
                assert!(
                    close(numeric, expected, 1e-4 * expected.abs().max(1.0)),
                    "Richards at nu = {nu}, t = {t}: {numeric} against {expected}"
                );
            }
        }
        assert!(richards(r, k, 0.0, n0, 1.0).is_err());
        assert!(richards(r, 0.0, 1.0, n0, 1.0).is_err());
    }

    #[test]
    fn the_allee_effect_makes_extinction_a_one_way_door() {
        // The qualitative difference from logistic growth: below the
        // threshold the growth rate is negative and the population collapses
        // however far it is from capacity. Checked from both sides of the
        // threshold and arbitrarily close to it.
        let (r, a, k) = (1.0f64, 20.0f64, 100.0f64);
        for &offset in &[-5.0f64, -0.5, -0.01] {
            let trace = allee_effect_ode(r, a, k, a + offset, 200.0).unwrap();
            let end = trace.last().unwrap().1;
            assert!(end < 1e-3, "starting {offset} below the threshold reached {end}");
        }
        for &offset in &[0.01f64, 0.5, 5.0] {
            let trace = allee_effect_ode(r, a, k, a + offset, 200.0).unwrap();
            let end = trace.last().unwrap().1;
            assert!(close(end, k, 1e-3 * k), "starting {offset} above it reached {end}");
        }
        // The threshold itself is an unstable equilibrium: it stays put.
        let poised = allee_effect_ode(r, a, k, a, 200.0).unwrap();
        assert!(close(poised.last().unwrap().1, a, 1e-6 * a));
        // And logistic growth has no such door -- any positive start
        // recovers. That contrast is what makes the Allee model different.
        assert!(logistic_growth(r, k, 1e-6, 200.0).unwrap() > 0.99 * k);
        for (_, n) in &poised {
            assert!(*n >= -1e-9 && n.is_finite());
        }
        assert!(allee_effect_ode(r, 0.0, k, 10.0, 10.0).is_err());
        assert!(allee_effect_ode(r, 50.0, 20.0, 10.0, 10.0).is_err());
        assert!(allee_effect_ode(r, a, k, -1.0, 10.0).is_err());
    }


    // -----------------------------------------------------------------
    // Age structure
    // -----------------------------------------------------------------

    #[test]
    fn the_leslie_growth_rate_and_the_euler_lotka_solution_are_the_same_number() {
        // Two entirely separate routes: one iterates a matrix to its
        // dominant eigenvalue, the other bisects a transcendental equation
        // built from the life table. They must agree, and agreement across
        // random-ish life tables is evidence about both.
        let tables: [(Vec<f64>, Vec<f64>); 4] = [
            (vec![0.0, 1.0, 2.0], vec![0.8, 0.5]),
            (vec![0.0, 0.0, 3.0, 1.0], vec![0.9, 0.7, 0.4]),
            (vec![0.5, 2.0], vec![0.6]),
            (vec![0.0, 4.0, 0.2, 0.1, 0.05], vec![0.95, 0.9, 0.6, 0.3]),
        ];
        for (fecundity, survival) in &tables {
            let l = leslie_matrix(fecundity, survival).unwrap();
            let (lambda, distribution) = leslie_growth_rate(&l).unwrap();
            assert!(lambda > 0.0 && lambda.is_finite());
            assert!(close(distribution.iter().sum::<f64>(), 1.0, 1e-12));
            assert!(distribution.iter().all(|x| *x >= -1e-12));

            // The life table: lx[i] is survivorship to age i + 1.
            let mut lx = Vec::with_capacity(fecundity.len());
            let mut running = 1.0;
            for i in 0..fecundity.len() {
                if i > 0 {
                    running *= survival[i - 1];
                }
                lx.push(running);
            }
            let mx = fecundity.clone();
            let r = euler_lotka_solve(&lx, &mx).unwrap();
            assert!(
                close(r, lambda, 1e-6 * lambda),
                "Euler-Lotka gives {r} against the matrix's {lambda}"
            );

            // The stable distribution really is the eigenvector: applying
            // the matrix scales it by lambda and nothing else.
            let applied: Vec<f64> = (0..l.rows)
                .map(|i| (0..l.cols).map(|j| l.get(i, j) * distribution[j]).sum())
                .collect();
            for i in 0..l.rows {
                assert!(
                    close(applied[i], lambda * distribution[i], 1e-8 * lambda),
                    "the distribution is not an eigenvector at class {i}"
                );
            }
        }
    }

    #[test]
    fn a_population_forgets_its_starting_age_structure() {
        // The strong ergodic theorem: whatever distribution it starts with,
        // a population converges to the same one and then grows by lambda
        // per step. Checked from three deliberately extreme starts, since
        // the theorem is about the asymptote and not the transient.
        let l = leslie_matrix(&[0.0, 1.5, 1.2, 0.3], &[0.85, 0.7, 0.4]).unwrap();
        let stable = stable_age_distribution(&l).unwrap();
        for start in [vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 1.0], vec![0.1, 0.6, 0.2, 0.1]]
        {
            let mut v = start.clone();
            for _ in 0..400 {
                let next: Vec<f64> = (0..4)
                    .map(|i| (0..4).map(|j| l.get(i, j) * v[j]).sum())
                    .collect();
                let total: f64 = next.iter().sum();
                v = next.iter().map(|x| x / total).collect();
            }
            for i in 0..4 {
                assert!(
                    close(v[i], stable[i], 1e-6),
                    "from {start:?} class {i} settled at {} rather than {}",
                    v[i],
                    stable[i]
                );
            }
        }
        // A species that reproduces at exactly one age is imprimitive: its
        // age classes cycle forever instead of settling, and the function
        // says so rather than returning a meaningless average.
        let cyclic = leslie_matrix(&[0.0, 0.0, 4.0], &[1.0, 1.0]).unwrap();
        assert!(leslie_growth_rate(&cyclic).is_err());
        // A population that never reproduces dies out, and is reported as
        // degenerate rather than as growth rate zero.
        let doomed = leslie_matrix(&[0.0, 0.0], &[0.5]).unwrap();
        assert!(leslie_growth_rate(&doomed).is_err());
        assert!(leslie_matrix(&[], &[]).is_err());
        assert!(leslie_matrix(&[1.0, 1.0], &[0.5, 0.5]).is_err());
        assert!(leslie_matrix(&[1.0, -1.0], &[0.5]).is_err());
        assert!(leslie_matrix(&[1.0, 1.0], &[1.5]).is_err());
        assert!(euler_lotka_solve(&[0.5], &[0.0]).is_err());
        assert!(euler_lotka_solve(&[0.5, 0.2], &[1.0]).is_err());
        assert!(euler_lotka_solve(&[1.5], &[1.0]).is_err());
    }

    #[test]
    fn a_stationary_population_has_growth_rate_one() {
        // The calibration point: a life table whose net reproductive rate is
        // exactly one must give lambda = 1, and both routes must say so.
        // Built by construction rather than by search.
        for classes in 2..=6usize {
            let survival = vec![0.8f64; classes - 1];
            let mut lx = Vec::with_capacity(classes);
            let mut running = 1.0;
            for i in 0..classes {
                if i > 0 {
                    running *= survival[i - 1];
                }
                lx.push(running);
            }
            // Choose a single fecundity at the last class making
            // sum lx mx / lambda^x = 1 at lambda = 1, i.e. lx[last] * m = 1.
            let mut fecundity = vec![0.0f64; classes];
            fecundity[classes - 1] = 1.0 / lx[classes - 1];
            let l = leslie_matrix(&fecundity, &survival).unwrap();
            let net: f64 = lx.iter().zip(&fecundity).map(|(a, b)| a * b).sum();
            assert!(close(net, 1.0, 1e-12), "the fixture is not stationary: R0 = {net}");
            // Reproduction at a single age is imprimitive, so the matrix
            // route is refused; Euler-Lotka has no such restriction and is
            // the right tool here.
            assert!(leslie_growth_rate(&l).is_err());
            let r = euler_lotka_solve(&lx, &fecundity).unwrap();
            assert!(close(r, 1.0, 1e-9), "a stationary table gave {r}");
            // Doubling every fecundity must raise it above one.
            let doubled: Vec<f64> = fecundity.iter().map(|f| f * 2.0).collect();
            assert!(euler_lotka_solve(&lx, &doubled).unwrap() > 1.0);
            let halved: Vec<f64> = fecundity.iter().map(|f| f * 0.5).collect();
            assert!(euler_lotka_solve(&lx, &halved).unwrap() < 1.0);
        }
    }

    // -----------------------------------------------------------------
    // Discrete maps
    // -----------------------------------------------------------------

    #[test]
    fn beverton_holt_matches_its_closed_form_and_never_overshoots() {
        // The exact solution is
        // N_t = K N_0 / (N_0 + (K - N_0) R^(-t)), which the iteration must
        // reproduce to rounding at every step.
        for &ratio in &[1.2f64, 2.0, 8.0] {
            for &n0 in &[0.1f64, 50.0, 400.0] {
                let k = 100.0;
                let trace = beverton_holt(ratio, k, n0, 40).unwrap();
                for (t, n) in trace.iter().enumerate() {
                    let expected =
                        k * n0 / (n0 + (k - n0) * ratio.powi(-(t as i32)));
                    assert!(
                        close(*n, expected, 1e-9 * expected.abs().max(1.0)),
                        "R = {ratio}, N0 = {n0}, t = {t}: {n} against {expected}"
                    );
                }
                // Monotone toward K from either side, never past it: that is
                // what "compensating" means, and it is why this model has no
                // route to chaos at any R.
                for pair in trace.windows(2) {
                    if n0 < k {
                        assert!(pair[1] >= pair[0] - 1e-12 && pair[1] <= k + 1e-9);
                    } else {
                        assert!(pair[1] <= pair[0] + 1e-12 && pair[1] >= k - 1e-9);
                    }
                }
                assert!(close(trace[40], k, 1e-3 * k) || ratio < 1.3);
            }
        }
        assert!(beverton_holt(1.0, 100.0, 1.0, 10).is_err());
        assert!(beverton_holt(2.0, 0.0, 1.0, 10).is_err());
        assert!(beverton_holt(2.0, 100.0, -1.0, 10).is_err());
    }

    #[test]
    fn the_ricker_map_period_doubles_into_chaos_where_it_should() {
        // The route is the point: a stable fixed point up to r = 2, then a
        // two-cycle, a four-cycle, and chaos beyond about 2.692. The
        // bifurcation diagram is asked for the *number* of attractor points,
        // which is an exact integer at each stage rather than a picture.
        let period = |r: f64| -> usize {
            let diagram = bifurcation_ricker(r, r + 1e-9, 2, 4_000, 512).unwrap();
            diagram[0].1.len()
        };
        assert_eq!(period(1.5), 1, "below r = 2 the fixed point is stable");
        assert_eq!(period(1.9), 1);
        assert_eq!(period(2.2), 2, "just above r = 2 there is a two-cycle");
        // The four-cycle begins at about 2.526, not at 2.5 -- the windows
        // narrow fast and guessing at their edges gets them wrong.
        assert_eq!(period(2.5), 2, "2.5 is still inside the two-cycle window");
        assert_eq!(period(2.6), 4, "the four-cycle is missing");
        assert_eq!(period(2.67), 8, "the eight-cycle is missing");
        assert!(period(3.0) > 16, "r = 3 is not chaotic: {} points", period(3.0));

        // The cascade converges at Feigenbaum's constant, 4.669, which is
        // universal: the same number for every map with a quadratic
        // maximum, and it appears here without being put in anywhere.
        //
        // The bifurcation points are found from the *multiplier* of the
        // cycle rather than by counting attractor points. Counting fails
        // near a boundary for a real reason: convergence there is algebraic
        // rather than geometric, so no finite transient settles and a
        // 4-cycle is indistinguishable from an 8-cycle just below the split.
        // The multiplier has no such problem -- a period-p cycle loses
        // stability exactly where the product of f' around it passes -1.
        let multiplier = |r: f64, p: usize| -> f64 {
            let step = |n: f64| n * (r * (1.0 - n)).exp();
            let mut n = 0.6;
            for _ in 0..20_000 {
                n = step(n);
            }
            let mut product = 1.0;
            for _ in 0..p {
                product *= (r * (1.0 - n)).exp() * (1.0 - r * n);
                n = step(n);
            }
            product
        };
        let onset = |p: usize, lo: f64, hi: f64| -> f64 {
            let (mut lo, mut hi) = (lo, hi);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if multiplier(mid, p) > -1.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        // The first is exact and needs no search: the fixed point is N = K,
        // where f'(K) = 1 - r, so it destabilises at r = 2 precisely.
        let first_point = 2.0;
        assert!(close(1.0 - first_point, -1.0, 1e-15));
        let second_point = onset(2, 2.05, 2.6);
        let third_point = onset(4, 2.53, 2.66);
        let fourth_point = onset(8, 2.66, 2.688);
        assert!(close(second_point, 2.5263, 1e-3), "the 2-to-4 split is at {second_point}");
        assert!(close(third_point, 2.6563, 1e-3), "the 4-to-8 split is at {third_point}");
        assert!(close(fourth_point, 2.6846, 1e-3), "the 8-to-16 split is at {fourth_point}");
        let ratio_one = (second_point - first_point) / (third_point - second_point);
        let ratio_two = (third_point - second_point) / (fourth_point - third_point);
        assert!(
            ratio_two > ratio_one,
            "the ratios are not converging: {ratio_one} then {ratio_two}"
        );
        assert!(
            close(ratio_two, 4.669, 0.15),
            "the second Feigenbaum ratio is {ratio_two}, not near 4.669"
        );

        // Exactly at a bifurcation point the approach is algebraic rather
        // than geometric, so no finite transient settles and the attractor
        // looks continuous. That is a property of the map, not of the
        // sampling, and it is why the checks above avoid the exact
        // thresholds.
        assert!(period(2.0) > 100, "r = 2.0 settled, which it cannot do in finite time");

        // The fixed point is exactly K below the threshold.
        let settled = ricker_map(1.5, 80.0, 10.0, 400).unwrap();
        assert!(close(settled[400], 80.0, 1e-6 * 80.0));
        // Overcompensation: from far above capacity it crashes below.
        let crash = ricker_map(2.5, 1.0, 4.0, 3).unwrap();
        assert!(crash[1] < 1.0, "the population did not overshoot downward");
        // Nothing goes negative or diverges.
        for r in [0.5f64, 1.0, 2.0, 3.0] {
            for n in ricker_map(r, 1.0, 0.6, 2_000).unwrap() {
                assert!(n >= 0.0 && n.is_finite(), "Ricker at r = {r} produced {n}");
            }
        }
        assert!(ricker_map(1.0, 0.0, 1.0, 10).is_err());
        assert!(ricker_map(1.0, 1.0, -1.0, 10).is_err());
        assert!(bifurcation_ricker(3.0, 1.0, 10, 100, 100).is_err());
        assert!(bifurcation_ricker(1.0, 3.0, 1, 100, 100).is_err());
        assert!(bifurcation_ricker(1.0, 3.0, 10, 100, 0).is_err());
    }

    // -----------------------------------------------------------------
    // Drift
    // -----------------------------------------------------------------

    #[test]
    fn wright_fisher_drift_is_a_martingale_that_nonetheless_always_fixes() {
        // The two facts together are the whole model, and neither alone
        // describes it: the expected frequency never moves, and yet every
        // population ends at zero or one. A deterministic account of drift
        // is not merely inaccurate -- it is silent about the outcome.
        let mut rng = Rng::new(0x0B10_1001);
        for &p0 in &[0.2f64, 0.5, 0.8] {
            let n = 40u64;
            let runs = 4_000;
            let mut sum_p = 0.0;
            let mut fixed_high = 0;
            let mut still_going = 0;
            for _ in 0..runs {
                let trace = wright_fisher(n, p0, 600, &mut rng).unwrap();
                let end = *trace.last().unwrap();
                sum_p += end;
                if end >= 1.0 - 1e-12 {
                    fixed_high += 1;
                } else if end > 1e-12 {
                    still_going += 1;
                }
            }
            let mean = sum_p / f64::from(runs);
            assert!(
                close(mean, p0, 0.02),
                "the mean frequency drifted from {p0} to {mean}"
            );
            // The fixation probability equals the starting frequency, which
            // is the martingale property expressed at the absorbing states.
            let fixation = f64::from(fixed_high) / f64::from(runs);
            assert!(
                close(fixation, p0, 0.02),
                "at p0 = {p0} the fixation rate is {fixation}"
            );
            // And essentially everything has absorbed by 600 generations,
            // which is well beyond the 4N = 160 scale.
            assert!(
                f64::from(still_going) / f64::from(runs) < 0.02,
                "{still_going} of {runs} runs were still segregating"
            );
        }
        // The variance grows as p0 (1 - p0) (1 - (1 - 1/2N)^t), a closed
        // form that a mean-only account cannot produce.
        let n = 25u64;
        let p0 = 0.5;
        for &t in &[5usize, 20, 60] {
            let runs = 6_000;
            let mut values = Vec::with_capacity(runs);
            for _ in 0..runs {
                values.push(*wright_fisher(n, p0, t, &mut rng).unwrap().last().unwrap());
            }
            let mean: f64 = values.iter().sum::<f64>() / runs as f64;
            let variance: f64 =
                values.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / runs as f64;
            let expected =
                p0 * (1.0 - p0) * (1.0 - (1.0 - 1.0 / (2.0 * n as f64)).powi(t as i32));
            assert!(
                close(variance, expected, 0.12 * expected),
                "after {t} generations the variance is {variance} against {expected}"
            );
        }
        assert!(wright_fisher(0, 0.5, 10, &mut rng).is_err());
        assert!(wright_fisher(10, 1.5, 10, &mut rng).is_err());
        // Fixed populations stay fixed.
        assert!(wright_fisher(10, 0.0, 50, &mut rng).unwrap().iter().all(|p| *p == 0.0));
        assert!(wright_fisher(10, 1.0, 50, &mut rng).unwrap().iter().all(|p| *p == 1.0));
    }

    #[test]
    fn the_moran_simulation_fixes_at_the_rate_its_closed_form_predicts() {
        // The formula and the simulation are independent, so agreement is
        // evidence about both -- and the neutral case, i/n, is the cleanest
        // statement of what drift alone does.
        let mut rng = Rng::new(0x0B10_1002);
        for &(n, i0, fitness) in &[(20u64, 5u64, 1.0f64), (20, 5, 1.5), (30, 3, 0.7), (16, 8, 2.0)] {
            let predicted = fixation_probability_moran(n, i0, fitness).unwrap();
            let runs = 6_000;
            let mut fixed = 0;
            for _ in 0..runs {
                if moran_process(n, i0, fitness, &mut rng).unwrap().0 {
                    fixed += 1;
                }
            }
            let observed = f64::from(fixed) / f64::from(runs);
            assert!(
                close(observed, predicted, 0.025),
                "n = {n}, i0 = {i0}, r = {fitness}: {observed} against {predicted}"
            );
        }
        // The neutral case is exactly i/n.
        for n in [5u64, 17, 100] {
            for i in 0..=n {
                assert!(close(
                    fixation_probability_moran(n, i, 1.0).unwrap(),
                    i as f64 / n as f64,
                    1e-12
                ));
            }
        }
        // Even a beneficial mutant is usually lost: at r = 1.01 a single
        // copy fixes about one time in a hundred, not with certainty.
        let lucky = fixation_probability_moran(1_000, 1, 1.01).unwrap();
        assert!(lucky > 0.005 && lucky < 0.02, "a 1% advantage fixed with probability {lucky}");
        // Fitness monotonically helps.
        let mut previous = 0.0;
        for step in 1..=40 {
            let r = f64::from(step) * 0.1;
            let p = fixation_probability_moran(50, 5, r).unwrap();
            assert!(p > previous, "fitness {r} fixed less often than the one below");
            previous = p;
        }
        assert!(close(fixation_probability_moran(10, 0, 2.0).unwrap(), 0.0, 1e-15));
        assert!(close(fixation_probability_moran(10, 10, 2.0).unwrap(), 1.0, 1e-15));
        assert!(fixation_probability_moran(0, 0, 1.0).is_err());
        assert!(fixation_probability_moran(10, 11, 1.0).is_err());
        assert!(fixation_probability_moran(10, 5, 0.0).is_err());
        assert!(moran_process(10, 11, 1.0, &mut rng).is_err());
    }

    #[test]
    fn heterozygosity_decays_at_the_rate_the_population_size_sets() {
        // H_t = H_0 (1 - 1/(2N))^t, and the 2N is the diploid gene copy
        // count -- using N would halve the predicted rate. Checked against a
        // Wright-Fisher simulation, which is an independent route to the
        // same decay.
        let mut rng = Rng::new(0x0B10_1003);
        for &n in &[10u64, 25] {
            let generations = 40;
            let runs = 4_000;
            let p0 = 0.5;
            let mut heterozygosity = vec![0.0f64; generations + 1];
            for _ in 0..runs {
                let trace = wright_fisher(n, p0, generations, &mut rng).unwrap();
                for (t, p) in trace.iter().enumerate() {
                    heterozygosity[t] += 2.0 * p * (1.0 - p);
                }
            }
            for (t, h) in heterozygosity.iter().enumerate() {
                let observed = h / f64::from(runs);
                let predicted =
                    genetic_drift_heterozygosity(n, 2.0 * p0 * (1.0 - p0), t as f64).unwrap();
                assert!(
                    close(observed, predicted, 0.03),
                    "at N = {n}, t = {t} the heterozygosity is {observed} against {predicted}"
                );
            }
        }
        // The closed form itself: halving the population doubles the decay
        // rate, and nothing survives indefinitely.
        assert!(close(genetic_drift_heterozygosity(50, 0.5, 0.0).unwrap(), 0.5, 1e-15));
        assert!(genetic_drift_heterozygosity(50, 0.5, 1e5).unwrap() < 1e-9);
        let small = genetic_drift_heterozygosity(25, 0.5, 20.0).unwrap();
        let large = genetic_drift_heterozygosity(50, 0.5, 20.0).unwrap();
        assert!(small < large, "the smaller population lost less variation");
        assert!(genetic_drift_heterozygosity(0, 0.5, 1.0).is_err());
        assert!(genetic_drift_heterozygosity(10, 1.5, 1.0).is_err());
    }


    // -----------------------------------------------------------------
    // Selection
    // -----------------------------------------------------------------

    #[test]
    fn hardy_weinberg_holds_where_it_should_and_the_test_detects_where_it_does_not() {
        // The proportions themselves are arithmetic; the test is what makes
        // them useful, so it is checked against data that does conform and
        // data that does not.
        for step in 0..=20 {
            let p = f64::from(step) / 20.0;
            let (aa, ab, bb) = hardy_weinberg(p).unwrap();
            assert!(close(aa + ab + bb, 1.0, 1e-15), "the genotypes do not sum to one");
            assert!(close(aa, p * p, 1e-15) && close(bb, (1.0 - p) * (1.0 - p), 1e-15));
            // The allele frequency is recovered from the genotypes.
            assert!(close(aa + ab / 2.0, p, 1e-12));
            // Heterozygosity peaks at p = 1/2 and is at most a half.
            assert!(ab <= 0.5 + 1e-15);
        }
        assert!(hardy_weinberg(-0.1).is_err());
        assert!(hardy_weinberg(1.1).is_err());

        // Conforming counts pass.
        let p = 0.3;
        let total = 1_000.0;
        let (aa, ab, bb) = hardy_weinberg(p).unwrap();
        let conforming = [aa * total, ab * total, bb * total];
        let result = hw_chi_square_test(conforming).unwrap();
        assert!(close(result.statistic, 0.0, 1e-9), "exact proportions gave {}", result.statistic);
        assert!(close(result.df, 1.0, 1e-15), "the degrees of freedom are {}", result.df);
        assert!(result.p_value > 0.99);

        // Complete inbreeding -- no heterozygotes at all -- is rejected
        // decisively at the same allele frequency.
        let inbred = [p * total, 0.0, (1.0 - p) * total];
        let rejected = hw_chi_square_test(inbred).unwrap();
        assert!(
            rejected.p_value < 1e-12,
            "a population with no heterozygotes passed at p = {}",
            rejected.p_value
        );
        assert!(rejected.statistic > result.statistic);
        // A mild excess of heterozygotes is detected too, but less strongly.
        let mild = [aa * total * 0.9, ab * total * 1.2, bb * total * 0.9];
        let middling = hw_chi_square_test(mild).unwrap();
        assert!(middling.p_value < 0.05 && middling.p_value > 1e-12);
        assert!(hw_chi_square_test([0.0, 0.0, 0.0]).is_err());
        assert!(hw_chi_square_test([-1.0, 1.0, 1.0]).is_err());
        // An absent allele leaves nothing to test.
        assert!(hw_chi_square_test([100.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn selection_has_three_outcomes_and_the_equilibrium_is_where_the_algebra_says() {
        // Directional selection is only one of them. Heterozygote advantage
        // gives a polymorphic equilibrium at a point with a closed form, and
        // heterozygote disadvantage makes both fixations stable so the
        // outcome depends on the start.
        // Directional: the fitter allele fixes from any interior start.
        for &p0 in &[0.01f64, 0.5, 0.99] {
            let trace = selection_one_locus(p0, [1.0, 0.9, 0.8], 4_000).unwrap();
            assert!(close(*trace.last().unwrap(), 1.0, 1e-6), "A did not fix from {p0}");
            // And monotonically, since there is no interior equilibrium.
            for pair in trace.windows(2) {
                assert!(pair[1] >= pair[0] - 1e-15, "the frequency fell under directional selection");
            }
        }

        // Heterozygote advantage: a polymorphic equilibrium, reached from
        // both sides, at (w12 - w22) / (2 w12 - w11 - w22).
        let w = [0.8f64, 1.0, 0.6];
        let star = balanced_polymorphism(w).unwrap();
        assert!(close(star, 0.4 / 0.6, 1e-12), "the equilibrium is {star}");
        for &p0 in &[0.05f64, 0.4, 0.95] {
            let trace = selection_one_locus(p0, w, 6_000).unwrap();
            assert!(
                close(*trace.last().unwrap(), star, 1e-6),
                "from {p0} the frequency settled at {} rather than {star}",
                trace.last().unwrap()
            );
        }
        // Heterozygote disadvantage: both fixations are stable and the
        // unstable equilibrium separates their basins.
        let bad = [1.0f64, 0.5, 0.9];
        assert!(balanced_polymorphism(bad).is_err());
        let unstable = (bad[1] - bad[2]) / (2.0 * bad[1] - bad[0] - bad[2]);
        assert!((0.0..1.0).contains(&unstable), "the fixture has no interior point");
        let below = selection_one_locus(unstable - 0.02, bad, 6_000).unwrap();
        let above = selection_one_locus(unstable + 0.02, bad, 6_000).unwrap();
        assert!(close(*below.last().unwrap(), 0.0, 1e-6), "below the ridge A did not vanish");
        assert!(close(*above.last().unwrap(), 1.0, 1e-6), "above the ridge A did not fix");
        // Neutrality changes nothing at all.
        let flat = selection_one_locus(0.37, [1.0, 1.0, 1.0], 500).unwrap();
        assert!(flat.iter().all(|p| close(*p, 0.37, 1e-12)));
        assert!(selection_one_locus(1.5, [1.0; 3], 10).is_err());
        assert!(selection_one_locus(0.5, [0.0; 3], 10).is_err());
        assert!(selection_one_locus(0.5, [1.0, -1.0, 1.0], 10).is_err());
    }

    #[test]
    fn dominance_dominates_the_mutation_selection_balance() {
        // The point worth making: even slight dominance changes the answer
        // by orders of magnitude, because selection sees heterozygotes far
        // more often than the rare homozygote.
        let (mu, s) = (1e-6f64, 0.1f64);
        let recessive = mutation_selection_balance(mu, s, 0.0).unwrap();
        assert!(close(recessive, (mu / s).sqrt(), 1e-12), "the recessive balance is {recessive}");
        assert!(close(recessive, 3.162e-3, 1e-5));
        let partial = mutation_selection_balance(mu, s, 0.1).unwrap();
        assert!(close(partial, mu / (0.1 * s), 1e-12), "the partial balance is {partial}");
        // Thirty-one fold, which is sqrt(s/mu) * h = sqrt(1e5) * 0.1. The
        // ratio has a closed form and is worth checking against it rather
        // than against a round number picked by eye.
        assert!(
            close(recessive / partial, (s / mu).sqrt() * 0.1, 1e-9),
            "the ratio is {}",
            recessive / partial
        );
        assert!(
            recessive / partial > 25.0,
            "dominance changed the answer by only a factor of {}",
            recessive / partial
        );
        // More dominance means rarer, monotonically; more mutation means
        // commoner; more selection means rarer.
        let mut previous = f64::INFINITY;
        for step in 1..=20 {
            let h = f64::from(step) * 0.05;
            let q = mutation_selection_balance(mu, s, h).unwrap();
            assert!(q < previous, "dominance {h} gave a commoner allele");
            previous = q;
        }
        assert!(mutation_selection_balance(1e-5, s, 0.5).unwrap() > partial);
        assert!(mutation_selection_balance(mu, 0.5, 0.5).unwrap()
            < mutation_selection_balance(mu, 0.05, 0.5).unwrap());
        // A frequency is never above one, however extreme the parameters.
        assert!(mutation_selection_balance(0.5, 1e-6, 0.0).unwrap() <= 1.0);
        assert!(mutation_selection_balance(mu, 0.0, 0.5).is_err());
        assert!(mutation_selection_balance(-1.0, s, 0.5).is_err());
        assert!(mutation_selection_balance(mu, s, 1.5).is_err());
    }

    #[test]
    fn hamiltons_rule_and_the_price_equation_say_what_they_claim() {
        // Hamilton's rule is a comparison; the Price equation is an
        // identity, and identities are checkable exactly.
        assert!(kin_selection_hamilton(0.5, 3.0, 1.0).unwrap());
        assert!(!kin_selection_hamilton(0.5, 3.0, 2.0).unwrap());
        assert!(!kin_selection_hamilton(0.0, 100.0, 0.001).unwrap());
        assert!(kin_selection_hamilton(1.0, 1.001, 1.0).unwrap());
        assert!(kin_selection_hamilton(1.5, 1.0, 1.0).is_err());

        // The Price identity: the two terms sum exactly to the change in
        // the mean trait, whatever the numbers.
        let mut rng = Rng::new(0x0B10_1004);
        for _ in 0..200 {
            let n = 3 + ((rng.next_f64() * 12.0) as usize);
            let z: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
            let w: Vec<f64> = (0..n).map(|_| rng.next_f64() * 3.0).collect();
            let z_offspring: Vec<f64> =
                (0..n).map(|k| z[k] + (rng.next_f64() - 0.5)).collect();
            let mean_w: f64 = w.iter().sum::<f64>() / n as f64;
            if !(mean_w > 1e-9) {
                continue;
            }
            let (selection, transmission) =
                price_equation_decompose(&z, &w, &z_offspring).unwrap();
            // The mean trait among offspring, weighted by fitness.
            let offspring_mean: f64 =
                (0..n).map(|k| w[k] * z_offspring[k]).sum::<f64>() / (n as f64 * mean_w);
            let parent_mean: f64 = z.iter().sum::<f64>() / n as f64;
            assert!(
                close(selection + transmission, offspring_mean - parent_mean, 1e-9),
                "the Price terms sum to {} against a change of {}",
                selection + transmission,
                offspring_mean - parent_mean
            );
        }
        // Perfect transmission puts everything in the selection term.
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0.5, 1.0, 1.5, 2.0];
        let (selection, transmission) = price_equation_decompose(&z, &w, &z).unwrap();
        assert!(close(transmission, 0.0, 1e-15), "faithful inheritance transmitted {transmission}");
        assert!(selection > 0.0, "fitness correlated with the trait but selection was {selection}");
        // Equal fitness puts everything in the transmission term.
        let flat = vec![1.0; 4];
        let drifted: Vec<f64> = z.iter().map(|x| x + 0.5).collect();
        let (s2, t2) = price_equation_decompose(&z, &flat, &drifted).unwrap();
        assert!(close(s2, 0.0, 1e-15), "equal fitness selected {s2}");
        assert!(close(t2, 0.5, 1e-15), "the transmission term is {t2}");
        assert!(price_equation_decompose(&[], &[], &[]).is_err());
        assert!(price_equation_decompose(&z, &w[..2], &z).is_err());
        assert!(price_equation_decompose(&z, &[0.0; 4], &z).is_err());
        assert!(price_equation_decompose(&z, &[-1.0, 1.0, 1.0, 1.0], &z).is_err());
    }

    // -----------------------------------------------------------------
    // Coalescent and diversity
    // -----------------------------------------------------------------

    #[test]
    fn the_coalescent_intervals_have_the_expectations_the_theory_gives() {
        // Simulated against closed forms: each interval is exponential with
        // rate C(k,2)/(2N), the total height is 4N(1 - 1/k), and T_2 alone
        // is half of it -- the genealogy is dominated by its deepest branch,
        // which is why ancient history is estimated from very little.
        let mut rng = Rng::new(0x0B10_1005);
        let n = 500u64;
        for &samples in &[2u64, 5, 20] {
            let runs = 20_000;
            let mut totals = vec![0.0f64; (samples - 1) as usize];
            let mut height = 0.0;
            for _ in 0..runs {
                let intervals = coalescent_simulate(n, samples, &mut rng).unwrap();
                assert_eq!(intervals.len(), (samples - 1) as usize);
                for (i, t) in intervals.iter().enumerate() {
                    assert!(*t >= 0.0 && t.is_finite());
                    totals[i] += t;
                }
                height += intervals.iter().sum::<f64>();
            }
            // Interval i corresponds to k = samples - i lineages.
            for (i, total) in totals.iter().enumerate() {
                let k = samples - i as u64;
                let observed = total / f64::from(runs);
                let expected = coalescent_time_expected(n, k).unwrap();
                assert!(
                    close(observed, expected, 0.06 * expected),
                    "with {k} lineages the mean interval is {observed} against {expected}"
                );
            }
            let mean_height = height / f64::from(runs);
            let expected_height = coalescent_tmrca_expected(n, samples).unwrap();
            assert!(
                close(mean_height, expected_height, 0.05 * expected_height),
                "for {samples} samples the height is {mean_height} against {expected_height}"
            );
        }
        // The deepest branch dominates: T_2 is 2N and the whole tree is at
        // most 4N, so the last coalescence is at least half the height
        // however large the sample.
        for samples in [4u64, 50, 5_000] {
            let t2 = coalescent_time_expected(n, 2).unwrap();
            let total = coalescent_tmrca_expected(n, samples).unwrap();
            assert!(t2 >= 0.5 * total, "T_2 is {t2} of a height of {total}");
            assert!(total < 4.0 * n as f64);
        }
        // Sampling more buys almost nothing at the root.
        let hundred = coalescent_tmrca_expected(n, 100).unwrap();
        let million = coalescent_tmrca_expected(n, 1_000_000).unwrap();
        assert!(million / hundred < 1.02, "a ten-thousandfold sample deepened the tree");
        assert!(coalescent_time_expected(n, 1).is_err());
        assert!(coalescent_time_expected(0, 5).is_err());
        assert!(coalescent_tmrca_expected(n, 1).is_err());
        assert!(coalescent_simulate(n, 1, &mut rng).is_err());
        assert!(coalescent_simulate(0, 5, &mut rng).is_err());
    }

    #[test]
    fn the_diversity_statistics_measure_what_they_are_defined_as() {
        // Small alignments where every quantity can be counted by hand.
        let sequences = vec![
            b"AAAA".to_vec(),
            b"AAAT".to_vec(),
            b"AACT".to_vec(),
            b"AACT".to_vec(),
        ];
        // Sites 3 and 4 vary; sites 1 and 2 do not.
        assert_eq!(segregating_sites(&sequences).unwrap(), 2);
        // Pairwise differences: (1,2)=1, (1,3)=2, (1,4)=2, (2,3)=1,
        // (2,4)=1, (3,4)=0, so pi = 7/6.
        assert!(close(nucleotide_diversity(&sequences).unwrap(), 7.0 / 6.0, 1e-12));
        // Watterson: a_4 = 1 + 1/2 + 1/3 = 11/6, so theta = 2 / (11/6).
        assert!(close(watterson_theta(2.0, 4).unwrap(), 12.0 / 11.0, 1e-12));

        // An invariant alignment has no diversity at all.
        let same = vec![b"GGGG".to_vec(); 5];
        assert_eq!(segregating_sites(&same).unwrap(), 0);
        assert!(close(nucleotide_diversity(&same).unwrap(), 0.0, 1e-15));
        assert!(tajima_d(&same).is_err());

        // Watterson grows only logarithmically with the sample, which is why
        // it is divided by a_n rather than by n: at a fixed theta the
        // expected number of segregating sites is theta * a_n.
        let mut previous = 0.0;
        for n in 2..=200u64 {
            let a = (1..n).map(|i| 1.0 / i as f64).sum::<f64>();
            assert!(a > previous);
            previous = a;
            // Recovering theta from the sites it would produce is exact.
            assert!(close(watterson_theta(5.0 * a, n).unwrap(), 5.0, 1e-9));
        }
        assert!(previous < 7.0, "a_200 should be about 5.9, not {previous}");
        assert!(watterson_theta(1.0, 1).is_err());
        assert!(watterson_theta(-1.0, 5).is_err());
        assert!(nucleotide_diversity(&sequences[..1]).is_err());
        let ragged = vec![b"AA".to_vec(), b"AAA".to_vec()];
        assert!(nucleotide_diversity(&ragged).is_err());
        assert!(segregating_sites(&ragged).is_err());
    }

    #[test]
    fn tajimas_d_is_near_zero_under_neutrality_and_negative_after_a_sweep() {
        // Both estimators target the same theta under neutrality, so their
        // difference is zero in expectation; a star genealogy -- one
        // ancestor, all differences private -- is the signature of an
        // expansion or a sweep and drives D negative.
        let mut rng = Rng::new(0x0B10_1006);
        let n = 12usize;
        let length = 400usize;

        // Neutral: mutations placed on a coalescent genealogy. A lineage
        // that splits early carries its mutations to many descendants,
        // which is what gives intermediate-frequency variants.
        let mut totals = 0.0;
        let runs = 200;
        for _ in 0..runs {
            let mut sequences = vec![vec![b'A'; length]; n];
            // Build a random bifurcating history by repeatedly merging.
            let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
            let mut site = 0usize;
            while groups.len() > 1 && site + 4 < length {
                let a = (rng.next_f64() * groups.len() as f64) as usize % groups.len();
                let mut b = (rng.next_f64() * groups.len() as f64) as usize % groups.len();
                if b == a {
                    b = (b + 1) % groups.len();
                }
                // Mutations on the branch above each group, in proportion to
                // the time it existed -- longer for fewer lineages.
                let branch = 2.0 / (groups.len() as f64 * (groups.len() as f64 - 1.0));
                for group in [a, b] {
                    let count = (branch * 40.0 * length as f64 / 100.0).round() as usize;
                    for _ in 0..count.min(3) {
                        if site >= length {
                            break;
                        }
                        for &member in &groups[group] {
                            sequences[member][site] = b'T';
                        }
                        site += 1;
                    }
                }
                let (lo, hi) = (a.min(b), a.max(b));
                let merged: Vec<usize> =
                    groups[lo].iter().chain(&groups[hi]).copied().collect();
                groups.remove(hi);
                groups.remove(lo);
                groups.push(merged);
            }
            if let Ok(d) = tajima_d(&sequences) {
                totals += d;
            }
        }
        let neutral_mean = totals / f64::from(runs);
        assert!(
            neutral_mean.abs() < 1.5,
            "a coalescent genealogy gave a mean D of {neutral_mean}"
        );

        // A star genealogy: every sequence carries its own private
        // mutations and shares none. Every variant is a singleton, so pi is
        // small against S and D is strongly negative.
        let mut star = vec![vec![b'A'; length]; n];
        for (i, seq) in star.iter_mut().enumerate() {
            for k in 0..8 {
                seq[i * 8 + k] = b'T';
            }
        }
        let swept = tajima_d(&star).unwrap();
        assert!(swept < -1.0, "a star genealogy gave D = {swept}");
        assert!(swept < neutral_mean, "the sweep signature is not below neutrality");

        // Balancing selection: two deeply diverged haplotypes, so every
        // variant is at frequency one half and pi is large against S.
        let mut balanced = vec![vec![b'A'; length]; n];
        for (i, seq) in balanced.iter_mut().enumerate() {
            if i % 2 == 0 {
                for site in seq.iter_mut().take(40) {
                    *site = b'T';
                }
            }
        }
        let held = tajima_d(&balanced).unwrap();
        assert!(held > 1.0, "two diverged haplotypes gave D = {held}");
        assert!(tajima_d(&star[..3]).is_err());
    }

    #[test]
    fn fst_partitions_the_heterozygosity_it_is_defined_from() {
        // Zero when the subpopulations are identical and one when each is
        // fixed for a different allele, with the intermediate values being
        // the fraction of variation *lost* to subdivision.
        for step in 0..=20 {
            let p = f64::from(step) / 20.0;
            if p > 0.0 && p < 1.0 {
                assert!(
                    close(fst(&[p, p, p]).unwrap(), 0.0, 1e-12),
                    "identical subpopulations gave a positive Fst"
                );
            }
        }
        assert!(close(fst(&[0.0, 1.0]).unwrap(), 1.0, 1e-12));
        assert!(close(fst(&[0.0, 1.0, 0.0, 1.0]).unwrap(), 1.0, 1e-12));
        // Symmetric halves around a mean of a half: Fst = (p - q)^2 form.
        for step in 1..=9 {
            let d = f64::from(step) / 20.0;
            let value = fst(&[0.5 - d, 0.5 + d]).unwrap();
            // H_T = 1/2, H_S = 2(1/4 - d^2), so Fst = 4 d^2.
            assert!(close(value, 4.0 * d * d, 1e-12), "at d = {d} Fst is {value}");
        }
        // More divergence means more Fst, monotonically.
        let mut previous = -1.0;
        for step in 0..=10 {
            let d = f64::from(step) / 22.0;
            let value = fst(&[0.5 - d, 0.5 + d]).unwrap();
            assert!(value > previous);
            previous = value;
        }
        // Always in range.
        let mut rng = Rng::new(0x0B10_1007);
        for _ in 0..500 {
            let freqs: Vec<f64> = (0..2 + (rng.next_f64() * 6.0) as usize)
                .map(|_| rng.next_f64())
                .collect();
            let value = fst(&freqs).unwrap();
            assert!((0.0..=1.0).contains(&value), "Fst left [0, 1]: {value}");
        }
        assert!(fst(&[0.5]).is_err());
        assert!(fst(&[0.5, 1.5]).is_err());
        assert!(fst(&[0.0, 0.0]).is_err());
        assert!(fst(&[1.0, 1.0]).is_err());
    }

    // -----------------------------------------------------------------
    // Interacting species
    // -----------------------------------------------------------------

    #[test]
    fn the_lotka_volterra_orbits_are_closed_and_remember_where_they_started() {
        // A conservative system, so the invariant is constant and the
        // amplitude depends on the initial condition. Both are checked,
        // because holding the first without the second would mean the
        // integrator had damped the orbit onto a spurious limit cycle.
        let (alpha, beta, delta, gamma) = (1.0f64, 0.5f64, 0.4f64, 0.8f64);
        let (trace, invariant) =
            lotka_volterra(alpha, beta, delta, gamma, 3.0, 1.5, 60.0).unwrap();
        let first = invariant[0];
        for (k, v) in invariant.iter().enumerate() {
            assert!(
                close(*v, first, 1e-5 * first.abs().max(1.0)),
                "the invariant drifted from {first} to {v} at step {k}"
            );
        }
        let swing = |t: &[(f64, f64, f64)], which: usize| -> f64 {
            let end = t.last().unwrap().0;
            let tail: Vec<f64> = t
                .iter()
                .filter(|(time, _, _)| *time > 0.5 * end)
                .map(|(_, x, y)| if which == 0 { *x } else { *y })
                .collect();
            tail.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - tail.iter().copied().fold(f64::INFINITY, f64::min)
        };
        assert!(swing(&trace, 0) > 0.5, "the prey barely moved");
        assert!(swing(&trace, 1) > 0.2, "the predator barely moved");
        // A wider start gives a wider orbit: the system is conservative,
        // not a limit cycle.
        let (wide, _) = lotka_volterra(alpha, beta, delta, gamma, 6.0, 0.6, 60.0).unwrap();
        assert!(
            swing(&wide, 0) > 1.5 * swing(&trace, 0),
            "a wider start gave the same orbit, so this behaves as a limit cycle"
        );
        // The fixed point is (gamma/delta, alpha/beta) and stays put.
        let (fixed, _) =
            lotka_volterra(alpha, beta, delta, gamma, gamma / delta, alpha / beta, 40.0).unwrap();
        assert!(swing(&fixed, 0) < 1e-6, "the coexistence equilibrium moved");
        // Both populations stay positive, always.
        for (_, x, y) in &trace {
            assert!(*x > 0.0 && *y > 0.0);
        }
        assert!(lotka_volterra(0.0, beta, delta, gamma, 1.0, 1.0, 1.0).is_err());
        assert!(lotka_volterra(alpha, beta, delta, gamma, 0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn enrichment_destabilises_the_predator_prey_equilibrium() {
        // The paradox of enrichment, as a measurement rather than a slogan:
        // below the critical capacity the system settles, above it the
        // amplitude grows, and the crossing is where the closed form says.
        let (r, attack, handling) = (1.0f64, 1.0f64, 0.4f64);
        let (efficiency, mortality) = (0.5f64, 0.3f64);
        let critical =
            enrichment_critical_capacity(attack, handling, efficiency, mortality).unwrap();
        assert!(critical > 0.0 && critical.is_finite());
        let swing = |k: f64| -> f64 {
            let trace = rosenzweig_macarthur(
                r, k, attack, handling, efficiency, mortality, 0.5, 0.3, 900.0,
            )
            .unwrap();
            let end = trace.last().unwrap().0;
            let tail: Vec<f64> = trace
                .iter()
                .filter(|(t, _, _)| *t > 0.7 * end)
                .map(|(_, x, _)| *x)
                .collect();
            tail.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - tail.iter().copied().fold(f64::INFINITY, f64::min)
        };
        // Comfortably below: it settles.
        assert!(swing(critical * 0.7) < 1e-3, "below the threshold it still oscillates");
        // Comfortably above: a limit cycle, and a wider one further up.
        let just_above = swing(critical * 1.4);
        let far_above = swing(critical * 2.5);
        assert!(just_above > 1e-2, "above the threshold it did not oscillate: {just_above}");
        assert!(
            far_above > just_above,
            "more enrichment gave a smaller cycle: {far_above} against {just_above}"
        );
        // The equilibrium prey density does not depend on K at all, which is
        // the counterintuitive part -- enriching the system feeds only the
        // predator.
        let prey_star = mortality / (attack * (efficiency - mortality * handling));
        let settled = rosenzweig_macarthur(
            r, critical * 0.7, attack, handling, efficiency, mortality, 0.5, 0.3, 900.0,
        )
        .unwrap();
        assert!(
            close(settled.last().unwrap().1, prey_star, 1e-3 * prey_star),
            "the prey settled at {} rather than {prey_star}",
            settled.last().unwrap().1
        );
        assert!(enrichment_critical_capacity(attack, handling, 0.1, 1.0).is_err());
        assert!(enrichment_critical_capacity(0.0, handling, efficiency, mortality).is_err());
    }

    #[test]
    fn competition_has_four_outcomes_and_the_integration_agrees_with_the_criterion() {
        // The criterion is algebraic and the integration is not, so agreeing
        // on all four cases is evidence about both.
        let (r1, r2) = (0.8f64, 0.9f64);
        let cases: [(f64, f64, f64, f64, Competition); 4] = [
            (100.0, 100.0, 0.5, 0.5, Competition::Coexistence),
            (100.0, 100.0, 0.5, 1.6, Competition::FirstExcludes),
            (100.0, 100.0, 1.6, 0.5, Competition::SecondExcludes),
            (100.0, 100.0, 1.6, 1.6, Competition::FounderControl),
        ];
        for (k1, k2, a12, a21, expected) in cases {
            assert_eq!(coexistence_condition(k1, k2, a12, a21).unwrap(), expected);
            let trace = competition_lv(r1, r2, k1, k2, a12, a21, 10.0, 10.0, 900.0).unwrap();
            let (_, n1, n2) = *trace.last().unwrap();
            match expected {
                Competition::Coexistence => {
                    assert!(n1 > 1.0 && n2 > 1.0, "coexistence gave {n1} and {n2}");
                    // The interior equilibrium, in closed form.
                    let denominator = 1.0 - a12 * a21;
                    let want1 = (k1 - a12 * k2) / denominator;
                    let want2 = (k2 - a21 * k1) / denominator;
                    assert!(close(n1, want1, 1e-3 * want1), "N1 is {n1} against {want1}");
                    assert!(close(n2, want2, 1e-3 * want2), "N2 is {n2} against {want2}");
                }
                Competition::FirstExcludes => {
                    assert!(close(n1, k1, 1e-3 * k1) && n2 < 1e-3, "gave {n1} and {n2}");
                }
                Competition::SecondExcludes => {
                    assert!(close(n2, k2, 1e-3 * k2) && n1 < 1e-3, "gave {n1} and {n2}");
                }
                Competition::FounderControl => {
                    // One or the other wins outright; from an equal start
                    // the faster grower does.
                    assert!(
                        (n1 < 1e-3) != (n2 < 1e-3),
                        "founder control left both at {n1} and {n2}"
                    );
                    // And the outcome depends on the start, which is the
                    // defining property.
                    let ahead = competition_lv(r1, r2, k1, k2, a12, a21, 40.0, 1.0, 900.0).unwrap();
                    let behind =
                        competition_lv(r1, r2, k1, k2, a12, a21, 1.0, 40.0, 900.0).unwrap();
                    assert!(ahead.last().unwrap().1 > ahead.last().unwrap().2);
                    assert!(behind.last().unwrap().2 > behind.last().unwrap().1);
                }
            }
            for (_, a, b) in &trace {
                assert!(*a >= -1e-9 && *b >= -1e-9 && a.is_finite() && b.is_finite());
            }
        }
        assert!(coexistence_condition(0.0, 100.0, 0.5, 0.5).is_err());
        assert!(coexistence_condition(100.0, 100.0, -0.5, 0.5).is_err());
        assert!(competition_lv(r1, r2, 0.0, 100.0, 0.5, 0.5, 1.0, 1.0, 10.0).is_err());
    }

    #[test]
    fn the_metapopulation_persists_only_while_colonisation_beats_extinction() {
        // The equilibrium is 1 - e/c exactly, and the threshold at c = e is
        // sharp: below it the occupancy decays to zero however high it
        // started.
        for &c in &[0.2f64, 0.5, 1.0] {
            for &e in &[0.05f64, 0.15, 0.4] {
                let trace = metapopulation_levins(c, e, 0.5, 900.0).unwrap();
                let end = trace.last().unwrap().1;
                if c > e {
                    assert!(
                        close(end, 1.0 - e / c, 1e-4),
                        "c = {c}, e = {e} settled at {end} rather than {}",
                        1.0 - e / c
                    );
                } else {
                    assert!(end < 1e-3, "c = {c}, e = {e} persisted at {end}");
                }
                for (_, p) in &trace {
                    assert!((-1e-9..=1.0 + 1e-9).contains(p), "occupancy left [0, 1]: {p}");
                }
            }
        }
        // Habitat loss: destroying a fraction D shifts the equilibrium to
        // 1 - D - e/c, so extinction arrives while a fraction e/c of the
        // habitat still stands. That is the extinction debt, and it is a
        // sharper statement than "habitat loss is bad".
        let (c, e) = (0.5f64, 0.2f64);
        let doomed_at = 1.0 - e / c;
        let survivor = metapopulation_levins(c * (1.0 - doomed_at * 0.9), e, 0.5, 900.0).unwrap();
        assert!(survivor.last().unwrap().1 > 1e-3, "the metapopulation died too early");
        let lost = metapopulation_levins(c * (1.0 - doomed_at * 1.1), e, 0.5, 900.0).unwrap();
        assert!(lost.last().unwrap().1 < 1e-3, "the metapopulation survived past its debt");
        assert!(metapopulation_levins(c, e, 1.5, 10.0).is_err());
        assert!(metapopulation_levins(-1.0, e, 0.5, 10.0).is_err());
    }
}
