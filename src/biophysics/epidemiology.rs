//! Compartment models of epidemics, their stochastic counterparts, and the
//! quantities estimated from case data.
//!
//! # Units and conventions
//!
//! Compartments are *fractions* of the population and sum to one, so a model
//! is independent of the population size and the numbers can be read as
//! probabilities. The stochastic models work in whole individuals instead,
//! because the questions they answer -- will this outbreak die out, how long
//! until it does -- are questions about integers and have no meaning in a
//! continuum. Rates are per unit time in whatever unit the caller uses for
//! `t_end`; the recovery rate `gamma` is the reciprocal of the mean
//! infectious period, so a two-week illness with time in days is
//! `gamma = 1/14`.
//!
//! # What the basic reproduction number is and is not
//!
//! `R0 = beta / gamma` is the expected number of secondary cases from one
//! case in a *wholly susceptible* population. It is a property of the
//! pathogen and the contact structure together, not of the pathogen alone,
//! and it stops describing the epidemic the moment susceptibles are
//! depleted -- which is what the effective reproduction number is for. Two
//! populations with the same `R0` and different contact heterogeneity do not
//! have the same epidemic; see [`epidemic_threshold_network`], where the
//! threshold is set by the largest eigenvalue of the contact graph rather
//! than by any average.

use crate::error::GeomError;
use crate::biophysics::integrate_adaptive as integrate;
use crate::graph::Graph;
use crate::monte_carlo::Rng;

/// One sample of a compartment trajectory: `(time, S, E, I, R)`.
///
/// Models without an exposed class report `E = 0`, so a caller can plot any
/// of them the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpidemicSample {
    /// Elapsed time.
    pub t: f64,
    /// Susceptible fraction.
    pub s: f64,
    /// Exposed (infected, not yet infectious) fraction.
    pub e: f64,
    /// Infectious fraction.
    pub i: f64,
    /// Removed (recovered or dead) fraction.
    pub r: f64,
}

impl EpidemicSample {
    /// The total, which every model here conserves.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.s + self.e + self.i + self.r
    }
}

fn check_initial(s0: f64, e0: f64, i0: f64) -> Result<(), GeomError> {
    if s0 < 0.0 || e0 < 0.0 || i0 < 0.0 {
        return Err(GeomError::InvalidArgument("the compartments must be non-negative"));
    }
    if s0 + e0 + i0 > 1.0 + 1e-12 {
        return Err(GeomError::InvalidArgument("the compartments exceed the whole population"));
    }
    Ok(())
}

fn to_samples(raw: Vec<(f64, Vec<f64>)>, exposed: bool) -> Vec<EpidemicSample> {
    raw.into_iter()
        .map(|(t, y)| {
            if exposed {
                EpidemicSample { t, s: y[0], e: y[1], i: y[2], r: y[3] }
            } else {
                EpidemicSample { t, s: y[0], e: 0.0, i: y[1], r: y[2] }
            }
        })
        .collect()
}

/// The classical SIR model.
///
/// # Errors
/// Returns an error for negative rates, a bad initial condition, or a
/// non-positive end time.
pub fn sir(
    beta: f64,
    gamma: f64,
    s0: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<EpidemicSample>, GeomError> {
    if beta < 0.0 || gamma < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    check_initial(s0, 0.0, i0)?;
    let r0 = 1.0 - s0 - i0;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (s, i) = (y[0].max(0.0), y[1].max(0.0));
        vec![-beta * s * i, beta * s * i - gamma * i, gamma * i]
    };
    Ok(to_samples(integrate(derivative, &[s0, i0, r0], t_end, 1e-9)?, false))
}

/// SIS: recovery returns an individual to the susceptible pool, so there is
/// no removed class and the disease can persist indefinitely.
///
/// The distinction from SIR is not a detail. With no removed class the
/// epidemic has an *endemic equilibrium* at `1 - 1/R0` rather than burning
/// out, which is why the same pathogen parameters give a one-off wave in one
/// model and a permanent prevalence in the other.
///
/// # Errors
/// Returns an error on the same conditions as [`sir`].
pub fn sis(
    beta: f64,
    gamma: f64,
    s0: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<EpidemicSample>, GeomError> {
    if beta < 0.0 || gamma < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    check_initial(s0, 0.0, i0)?;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (s, i) = (y[0].max(0.0), y[1].max(0.0));
        vec![-beta * s * i + gamma * i, beta * s * i - gamma * i, 0.0]
    };
    Ok(to_samples(integrate(derivative, &[s0, i0, 0.0], t_end, 1e-9)?, false))
}

/// SIRS: immunity wanes at rate `omega`, returning the removed to the
/// susceptible pool.
///
/// # Errors
/// Returns an error on the same conditions as [`sir`].
pub fn sirs(
    beta: f64,
    gamma: f64,
    omega: f64,
    s0: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<EpidemicSample>, GeomError> {
    if beta < 0.0 || gamma < 0.0 || omega < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    check_initial(s0, 0.0, i0)?;
    let r0 = 1.0 - s0 - i0;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (s, i, r) = (y[0].max(0.0), y[1].max(0.0), y[2].max(0.0));
        vec![-beta * s * i + omega * r, beta * s * i - gamma * i, gamma * i - omega * r]
    };
    Ok(to_samples(integrate(derivative, &[s0, i0, r0], t_end, 1e-9)?, false))
}

/// SEIR: an exposed class that is infected but not yet infectious, entered
/// at the infection rate and left at rate `sigma`.
///
/// The latent period does not change the final size at all -- that depends
/// on `R0` alone -- but it slows the *growth rate*, which is what makes two
/// pathogens with the same `R0` and different incubation periods look so
/// different in the first month.
///
/// # Errors
/// Returns an error on the same conditions as [`sir`].
pub fn seir(
    beta: f64,
    sigma: f64,
    gamma: f64,
    s0: f64,
    e0: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<EpidemicSample>, GeomError> {
    if beta < 0.0 || sigma < 0.0 || gamma < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    check_initial(s0, e0, i0)?;
    let r0 = 1.0 - s0 - e0 - i0;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (s, e, i) = (y[0].max(0.0), y[1].max(0.0), y[2].max(0.0));
        vec![-beta * s * i, beta * s * i - sigma * e, sigma * e - gamma * i, gamma * i]
    };
    Ok(to_samples(integrate(derivative, &[s0, e0, i0, r0], t_end, 1e-9)?, true))
}

/// SEIRS: SEIR with waning immunity.
///
/// # Errors
/// Returns an error on the same conditions as [`sir`].
pub fn seirs(
    beta: f64,
    sigma: f64,
    gamma: f64,
    omega: f64,
    s0: f64,
    e0: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<EpidemicSample>, GeomError> {
    if beta < 0.0 || sigma < 0.0 || gamma < 0.0 || omega < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    check_initial(s0, e0, i0)?;
    let r0 = 1.0 - s0 - e0 - i0;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (s, e, i, r) = (y[0].max(0.0), y[1].max(0.0), y[2].max(0.0), y[3].max(0.0));
        vec![
            -beta * s * i + omega * r,
            beta * s * i - sigma * e,
            sigma * e - gamma * i,
            gamma * i - omega * r,
        ]
    };
    Ok(to_samples(integrate(derivative, &[s0, e0, i0, r0], t_end, 1e-9)?, true))
}

/// MSIR: an additional class of infants protected by maternal antibodies,
/// which are lost at rate `delta`.
///
/// Returns `(time, M, S, I, R)`. The maternal class is why measles
/// vaccination is not given at birth: the antibodies that protect the infant
/// also neutralise the vaccine.
///
/// # Errors
/// Returns an error on the same conditions as [`sir`].
pub fn msir(
    beta: f64,
    gamma: f64,
    delta: f64,
    m0: f64,
    s0: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<(f64, f64, f64, f64, f64)>, GeomError> {
    if beta < 0.0 || gamma < 0.0 || delta < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    if m0 < 0.0 || m0 + s0 + i0 > 1.0 + 1e-12 {
        return Err(GeomError::InvalidArgument("the compartments exceed the whole population"));
    }
    check_initial(s0, 0.0, i0)?;
    let r0 = 1.0 - m0 - s0 - i0;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (m, s, i) = (y[0].max(0.0), y[1].max(0.0), y[2].max(0.0));
        vec![-delta * m, delta * m - beta * s * i, beta * s * i - gamma * i, gamma * i]
    };
    Ok(integrate(derivative, &[m0, s0, i0, r0], t_end, 1e-9)?
        .into_iter()
        .map(|(t, y)| (t, y[0], y[1], y[2], y[3]))
        .collect())
}

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

/// `R0 = beta / gamma` for the SIR model.
///
/// # Errors
/// Returns an error for a non-positive recovery rate, for which the
/// infectious period is unbounded and `R0` is not defined.
pub fn r0_sir(beta: f64, gamma: f64) -> Result<f64, GeomError> {
    if !(gamma > 0.0) || beta < 0.0 {
        return Err(GeomError::InvalidArgument("r0_sir: bad rates"));
    }
    Ok(beta / gamma)
}

/// The herd immunity threshold `1 - 1/R0`: the immune fraction at which the
/// effective reproduction number falls to one.
///
/// This is the threshold for the epidemic to stop *growing*, not the
/// fraction that ends up infected. An epidemic that reaches the threshold
/// keeps going and overshoots it, because the people already infectious at
/// that moment go on to infect others; see [`final_size_equation`], whose
/// answer is always larger.
///
/// # Errors
/// Returns an error for `R0` below one, where no immunity is needed.
pub fn herd_immunity_threshold(r0: f64) -> Result<f64, GeomError> {
    if !(r0 >= 1.0) {
        return Err(GeomError::InvalidArgument("below R0 = 1 no threshold is needed"));
    }
    Ok(1.0 - 1.0 / r0)
}

/// The final size of an epidemic: the fraction ever infected, from the
/// implicit relation `1 - z = exp(-R0 z)`.
///
/// Solved by bisection, which is unconditionally safe here because
/// `f(z) = 1 - z - exp(-R0 z)` vanishes at zero, is *positive* just above it
/// for every `R0 > 1` -- its slope there is `R0 - 1` -- and is `-exp(-R0)`
/// at one. So the sought root is bracketed with `f` positive at the low end
/// and negative at the high end, which is the opposite of the usual
/// arrangement and the easy thing to get backwards. Newton's method on the
/// same equation converges too, but from a poor start it can step outside
/// `[0, 1]`, where the epidemic fraction has no meaning.
///
/// # Errors
/// Returns an error for a negative `R0`.
pub fn final_size_equation(r0: f64) -> Result<f64, GeomError> {
    if r0 < 0.0 {
        return Err(GeomError::InvalidArgument("R0 must be non-negative"));
    }
    if r0 <= 1.0 {
        // Below threshold the only root is zero: an introduction dies out.
        return Ok(0.0);
    }
    let f = |z: f64| 1.0 - z - (-r0 * z).exp();
    let (mut lo, mut hi) = (1e-12, 1.0);
    debug_assert!(f(lo) > 0.0 && f(hi) < 0.0, "the root is not bracketed");
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// The probability that an introduction of `i0` infectious individuals dies
/// out rather than becoming an epidemic.
///
/// From the branching-process approximation, valid while susceptibles are
/// undepleted: each case's offspring are geometric with mean `R0`, the
/// extinction probability of one chain is `1/R0`, and independent chains
/// multiply. So even a pathogen with `R0 = 3` fails to establish about a
/// third of the time from a single case -- epidemics are rarer than their
/// reproduction numbers suggest, and the ones that happen are the survivors
/// of many that did not.
///
/// # Errors
/// Returns an error for a negative `R0` or no introductions.
pub fn extinction_probability_epidemic(r0: f64, i0: u32) -> Result<f64, GeomError> {
    if r0 < 0.0 {
        return Err(GeomError::InvalidArgument("R0 must be non-negative"));
    }
    if i0 == 0 {
        return Err(GeomError::InvalidArgument("there must be at least one introduction"));
    }
    if r0 <= 1.0 {
        return Ok(1.0);
    }
    Ok((1.0 / r0).powi(i0 as i32))
}

/// The epidemic threshold of a contact network: the reciprocal of the
/// largest eigenvalue of its adjacency matrix.
///
/// A disease spreads on the network when `beta / gamma` exceeds this. The
/// mean degree is *not* the right quantity: a network with a few very
/// highly connected nodes has a spectral radius far above its mean degree,
/// and its epidemic threshold is correspondingly lower. That is why a
/// scale-free contact structure sustains an epidemic that a homogeneous
/// network with the same average contact rate would not.
///
/// # Errors
/// Returns an error for an empty graph or one with no edges, whose spectral
/// radius is zero and whose threshold is unbounded.
pub fn epidemic_threshold_network(g: &Graph) -> Result<f64, GeomError> {
    if g.n == 0 {
        return Err(GeomError::Empty);
    }
    let spectrum = crate::graph::spectral::adjacency_spectrum(g);
    let radius = spectrum.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if !(radius > 1e-12) {
        return Err(GeomError::Degenerate("the graph has no edges to spread along"));
    }
    Ok(1.0 / radius)
}

// ---------------------------------------------------------------------------
// Interventions and structure
// ---------------------------------------------------------------------------

/// SIR with a fraction `coverage` vaccinated before the epidemic begins.
///
/// Vaccination moves people straight from susceptible to removed, so it acts
/// exactly like a reduced initial susceptible fraction -- which is why the
/// effect of a vaccination campaign on the final size is entirely captured
/// by `R0 (1 - coverage)`, and why the threshold coverage is the herd
/// immunity threshold.
///
/// # Errors
/// Returns an error for a coverage outside zero to one, or on the same
/// conditions as [`sir`].
pub fn sir_with_vaccination(
    beta: f64,
    gamma: f64,
    coverage: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<EpidemicSample>, GeomError> {
    if !(0.0..=1.0).contains(&coverage) {
        return Err(GeomError::InvalidArgument("the coverage must be a fraction"));
    }
    let s0 = (1.0 - coverage - i0).max(0.0);
    // `sir` puts whatever is left over -- here exactly the vaccinated
    // fraction -- into the removed class, so nothing further is needed.
    sir(beta, gamma, s0, i0, t_end)
}

/// SIR with births and deaths at rate `mu`, both at the same rate so the
/// population is constant.
///
/// Demography is what turns a one-off epidemic into an endemic disease: the
/// birth of new susceptibles replenishes the fuel, and the trajectory spirals
/// into an equilibrium at `S* = 1/R0` rather than burning out. The damped
/// oscillation on the way there is the source of the multi-year cycles seen
/// in measles before vaccination.
///
/// # Errors
/// Returns an error for a negative rate, or on the same conditions as
/// [`sir`].
pub fn sir_with_demography(
    beta: f64,
    gamma: f64,
    mu: f64,
    s0: f64,
    i0: f64,
    t_end: f64,
) -> Result<Vec<EpidemicSample>, GeomError> {
    if beta < 0.0 || gamma < 0.0 || mu < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    check_initial(s0, 0.0, i0)?;
    let r0 = 1.0 - s0 - i0;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (s, i, r) = (y[0].max(0.0), y[1].max(0.0), y[2].max(0.0));
        vec![
            mu - beta * s * i - mu * s,
            beta * s * i - gamma * i - mu * i,
            gamma * i - mu * r,
        ]
    };
    Ok(to_samples(integrate(derivative, &[s0, i0, r0], t_end, 1e-9)?, false))
}

/// Two strains competing for the same susceptible pool, with complete
/// cross-immunity.
///
/// Returns `(time, S, I1, I2, R)`. Both strains end at zero: they compete
/// for one susceptible pool and this model does not replenish it, so the
/// epidemic ends when the susceptibles do.
///
/// Which strain infects more is *not* settled by `R0` alone. Competitive
/// exclusion -- the fitter strain driving the other out however far behind
/// it starts -- is a statement about a system with susceptible
/// replenishment, where there is an indefinite future to be excluded from.
/// Here the race is finite, and a strain with a thousandfold head start can
/// out-infect a rival with nearly twice its reproduction number before the
/// susceptibles are gone. From equal starts the fitter strain does win.
///
/// # Errors
/// Returns an error for negative rates or a bad initial condition.
pub fn two_strain(
    beta1: f64,
    gamma1: f64,
    beta2: f64,
    gamma2: f64,
    s0: f64,
    i1: f64,
    i2: f64,
    t_end: f64,
) -> Result<Vec<(f64, f64, f64, f64, f64)>, GeomError> {
    if beta1 < 0.0 || beta2 < 0.0 || gamma1 < 0.0 || gamma2 < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    if s0 < 0.0 || i1 < 0.0 || i2 < 0.0 || s0 + i1 + i2 > 1.0 + 1e-12 {
        return Err(GeomError::InvalidArgument("the compartments exceed the whole population"));
    }
    let r0 = 1.0 - s0 - i1 - i2;
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let (s, a, b) = (y[0].max(0.0), y[1].max(0.0), y[2].max(0.0));
        vec![
            -beta1 * s * a - beta2 * s * b,
            beta1 * s * a - gamma1 * a,
            beta2 * s * b - gamma2 * b,
            gamma1 * a + gamma2 * b,
        ]
    };
    Ok(integrate(derivative, &[s0, i1, i2, r0], t_end, 1e-10)?
        .into_iter()
        .map(|(t, y)| (t, y[0], y[1], y[2], y[3]))
        .collect())
}

/// An age-structured SIR with a contact matrix.
///
/// `contact[i][j]` is the rate at which a member of group `i` is contacted
/// by a member of group `j`, and `sizes` gives each group's share of the
/// population. Returns the trajectory as `(time, S, I, R)` with one entry
/// per group.
///
/// Structure changes the threshold, not just the detail. `R0` is the largest
/// eigenvalue of the next-generation matrix, not the average contact rate
/// times the infectious period, and the two differ whenever contact is
/// assortative -- which it always is by age.
///
/// # Errors
/// Returns an error for a non-square or negative contact matrix, group sizes
/// that do not sum to one, or a bad initial condition.
pub fn age_structured(
    contact: &[Vec<f64>],
    sizes: &[f64],
    gamma: f64,
    i0: &[f64],
    t_end: f64,
) -> Result<Vec<(f64, Vec<f64>, Vec<f64>, Vec<f64>)>, GeomError> {
    let groups = sizes.len();
    if groups == 0 || contact.len() != groups || contact.iter().any(|row| row.len() != groups) {
        return Err(GeomError::InvalidArgument("the contact matrix is not square"));
    }
    if contact.iter().flatten().any(|c| *c < 0.0) || gamma < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    if i0.len() != groups {
        return Err(GeomError::InvalidArgument("one initial infectious fraction per group"));
    }
    if (sizes.iter().sum::<f64>() - 1.0).abs() > 1e-9 || sizes.iter().any(|s| !(*s > 0.0)) {
        return Err(GeomError::InvalidArgument("the group sizes must be positive and sum to one"));
    }
    if i0.iter().zip(sizes).any(|(i, n)| *i < 0.0 || *i > *n + 1e-12) {
        return Err(GeomError::InvalidArgument("a group has more infectious than members"));
    }
    let contact = contact.to_vec();
    let sizes_owned = sizes.to_vec();
    let derivative = move |y: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; 3 * groups];
        for a in 0..groups {
            let s = y[a].max(0.0);
            // The force of infection on group a: contacts with each group,
            // weighted by that group's infectious *prevalence*.
            let force: f64 = (0..groups)
                .map(|b| contact[a][b] * y[groups + b].max(0.0) / sizes_owned[b])
                .sum();
            let new_cases = force * s;
            let recoveries = gamma * y[groups + a].max(0.0);
            out[a] = -new_cases;
            out[groups + a] = new_cases - recoveries;
            out[2 * groups + a] = recoveries;
        }
        out
    };
    let mut y0 = vec![0.0; 3 * groups];
    for a in 0..groups {
        y0[a] = sizes[a] - i0[a];
        y0[groups + a] = i0[a];
    }
    Ok(integrate(derivative, &y0, t_end, 1e-9)?
        .into_iter()
        .map(|(t, y)| {
            (
                t,
                y[..groups].to_vec(),
                y[groups..2 * groups].to_vec(),
                y[2 * groups..].to_vec(),
            )
        })
        .collect())
}

/// `R0` for an age-structured model: the largest eigenvalue of the
/// next-generation matrix `K[a][b] = contact[a][b] * sizes[a] / (gamma *
/// sizes[b])`.
///
/// # Errors
/// Returns an error on the same conditions as [`age_structured`], or for a
/// non-positive recovery rate.
pub fn r0_age_structured(
    contact: &[Vec<f64>],
    sizes: &[f64],
    gamma: f64,
) -> Result<f64, GeomError> {
    let groups = sizes.len();
    if groups == 0 || contact.len() != groups || contact.iter().any(|row| row.len() != groups) {
        return Err(GeomError::InvalidArgument("the contact matrix is not square"));
    }
    if !(gamma > 0.0) {
        return Err(GeomError::InvalidArgument("the recovery rate must be positive"));
    }
    if (sizes.iter().sum::<f64>() - 1.0).abs() > 1e-9 || sizes.iter().any(|s| !(*s > 0.0)) {
        return Err(GeomError::InvalidArgument("the group sizes must be positive and sum to one"));
    }
    // Power iteration on a non-negative matrix, which Perron-Frobenius
    // guarantees converges to the dominant eigenvalue.
    let mut v = vec![1.0 / groups as f64; groups];
    let mut lambda = 0.0;
    for _ in 0..10_000 {
        let next: Vec<f64> = (0..groups)
            .map(|a| {
                (0..groups)
                    .map(|b| contact[a][b] * sizes[a] / (gamma * sizes[b]) * v[b])
                    .sum()
            })
            .collect();
        let norm: f64 = next.iter().map(|x| x.abs()).sum();
        if !(norm > 0.0) {
            return Ok(0.0);
        }
        let scaled: Vec<f64> = next.iter().map(|x| x / norm).collect();
        let moved = (0..groups).map(|k| (scaled[k] - v[k]).abs()).fold(0.0, f64::max);
        v = scaled;
        lambda = norm;
        if moved < 1e-14 {
            break;
        }
    }
    Ok(lambda)
}

// ---------------------------------------------------------------------------
// Stochastic epidemics
// ---------------------------------------------------------------------------

/// An exact stochastic SIR by Gillespie's direct method, in whole
/// individuals.
///
/// Returns `(time, S, I, R)` after each event. The deterministic model
/// cannot answer the question this one is for: with `R0 > 1` the
/// deterministic epidemic always takes off, while the stochastic one dies
/// out with probability `(1/R0)^i0` -- and that difference is not a
/// correction, it is the whole behaviour at small numbers.
///
/// # Errors
/// Returns an error for negative rates, an empty population, or a
/// non-positive end time.
pub fn sir_stochastic_gillespie(
    beta: f64,
    gamma: f64,
    n: u64,
    i0: u64,
    t_end: f64,
    rng: &mut Rng,
) -> Result<Vec<(f64, u64, u64, u64)>, GeomError> {
    if beta < 0.0 || gamma < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    if n == 0 || i0 > n || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("sir_stochastic_gillespie: bad parameters"));
    }
    let (mut s, mut i, mut r) = (n - i0, i0, 0u64);
    let mut t = 0.0;
    let mut out = vec![(t, s, i, r)];
    while t < t_end && i > 0 {
        // The infection propensity uses the *density* of infectives, so the
        // model matches the deterministic one as n grows.
        let infect = beta * s as f64 * i as f64 / n as f64;
        let recover = gamma * i as f64;
        let total = infect + recover;
        if !(total > 0.0) {
            break;
        }
        t -= (1.0 - rng.next_f64()).ln() / total;
        if t > t_end {
            break;
        }
        if rng.next_f64() * total < infect {
            s -= 1;
            i += 1;
        } else {
            i -= 1;
            r += 1;
        }
        out.push((t, s, i, r));
        if out.len() > 20_000_000 {
            return Err(GeomError::Degenerate("the epidemic did not terminate"));
        }
    }
    Ok(out)
}

/// An SIR epidemic on a contact network.
///
/// Each infectious node infects each susceptible neighbour at rate `beta`
/// and recovers at rate `gamma`. Returns the `(S, I, R)` counts after each
/// event. Unlike the well-mixed model the epidemic here is limited by the
/// *local* structure: a node cannot reinfect its own neighbourhood, so the
/// final size is smaller than the well-mixed prediction at the same `R0`.
///
/// # Errors
/// Returns an error for negative rates, an empty graph, or a patient zero
/// outside it.
pub fn network_sir(
    g: &Graph,
    beta: f64,
    gamma: f64,
    patient_zero: usize,
    rng: &mut Rng,
) -> Result<Vec<(usize, usize, usize)>, GeomError> {
    if beta < 0.0 || gamma < 0.0 {
        return Err(GeomError::InvalidArgument("the rates must be non-negative"));
    }
    if g.n == 0 {
        return Err(GeomError::Empty);
    }
    if patient_zero >= g.n {
        return Err(GeomError::InvalidArgument("patient zero is not a vertex"));
    }
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        Susceptible,
        Infectious,
        Removed,
    }
    let mut state = vec![State::Susceptible; g.n];
    state[patient_zero] = State::Infectious;
    let mut counts = vec![(g.n - 1, 1usize, 0usize)];
    loop {
        // Every currently possible event and its rate.
        let mut infections: Vec<usize> = Vec::new();
        let mut infectious: Vec<usize> = Vec::new();
        for u in 0..g.n {
            if state[u] != State::Infectious {
                continue;
            }
            infectious.push(u);
            for (v, _) in &g.adj[u] {
                if state[*v] == State::Susceptible {
                    infections.push(*v);
                }
            }
        }
        if infectious.is_empty() {
            break;
        }
        // Each susceptible neighbour appears once per infectious neighbour,
        // which is exactly the multiplicity its infection rate should have.
        let infect_rate = beta * infections.len() as f64;
        let recover_rate = gamma * infectious.len() as f64;
        let total = infect_rate + recover_rate;
        if !(total > 0.0) {
            break;
        }
        if rng.next_f64() * total < infect_rate {
            let target = infections[((u128::from(rng.next_u64()) * infections.len() as u128) >> 64) as usize];
            state[target] = State::Infectious;
        } else {
            let target = infectious[((u128::from(rng.next_u64()) * infectious.len() as u128) >> 64) as usize];
            state[target] = State::Removed;
        }
        let s = state.iter().filter(|x| **x == State::Susceptible).count();
        let i = state.iter().filter(|x| **x == State::Infectious).count();
        counts.push((s, i, g.n - s - i));
    }
    Ok(counts)
}

// ---------------------------------------------------------------------------
// Estimation from case data
// ---------------------------------------------------------------------------

/// The effective reproduction number over time, by the Cori method.
///
/// `R_t` is the ratio of today's incidence to the total infectiousness
/// present, where the latter is past incidence weighted by the serial
/// interval distribution. Returns one estimate per day from `window`
/// onward, and `NaN` before that -- there is no data yet, and reporting a
/// number there would be worse than reporting nothing.
///
/// The distinction from a naive ratio of consecutive counts matters: that
/// ratio is a *growth rate*, and converting it to a reproduction number
/// requires knowing the generation time. Two epidemics doubling at the same
/// speed have very different `R_t` if one has a serial interval of three
/// days and the other of ten.
///
/// # Errors
/// Returns an error for a negative incidence, a serial interval that is not
/// a distribution, or a window longer than the record.
pub fn effective_r_estimate(
    incidence: &[f64],
    serial_interval: &[f64],
    window: usize,
) -> Result<Vec<f64>, GeomError> {
    if incidence.iter().any(|c| *c < 0.0) {
        return Err(GeomError::InvalidArgument("the incidence must be non-negative"));
    }
    if serial_interval.is_empty() || serial_interval.iter().any(|w| *w < 0.0) {
        return Err(GeomError::InvalidArgument("the serial interval must be non-negative"));
    }
    let mass: f64 = serial_interval.iter().sum();
    if !(mass > 0.0) {
        return Err(GeomError::InvalidArgument("the serial interval carries no mass"));
    }
    if window == 0 || window >= incidence.len() {
        return Err(GeomError::InvalidArgument("the window does not fit the record"));
    }
    // Normalised, so the caller may pass unnormalised weights.
    let w: Vec<f64> = serial_interval.iter().map(|x| x / mass).collect();
    let mut out = vec![f64::NAN; incidence.len()];
    for day in window..incidence.len() {
        let mut cases = 0.0;
        let mut infectiousness = 0.0;
        for back in 0..window {
            let today = day - back;
            cases += incidence[today];
            // Weight index s corresponds to a serial interval of s + 1 days.
            for (s, weight) in w.iter().enumerate() {
                if today > s {
                    infectiousness += weight * incidence[today - s - 1];
                }
            }
        }
        out[day] = if infectiousness > 0.0 { cases / infectiousness } else { f64::NAN };
    }
    Ok(out)
}

/// Fits a gamma distribution to observed serial intervals by the method of
/// moments, returning `(shape, scale)`.
///
/// # Errors
/// Returns an error for fewer than two observations, a non-positive
/// interval, or observations with no spread.
pub fn serial_interval_fit(intervals: &[f64]) -> Result<(f64, f64), GeomError> {
    if intervals.len() < 2 {
        return Err(GeomError::InvalidArgument("serial_interval_fit needs two observations"));
    }
    if intervals.iter().any(|x| !(*x > 0.0)) {
        return Err(GeomError::InvalidArgument("every interval must be positive"));
    }
    let n = intervals.len() as f64;
    let mean: f64 = intervals.iter().sum::<f64>() / n;
    let variance: f64 =
        intervals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    if !(variance > 0.0) {
        return Err(GeomError::Degenerate("every interval is identical"));
    }
    // For a gamma, mean = k theta and variance = k theta^2.
    Ok((mean * mean / variance, variance / mean))
}

/// Fits `(beta, sigma, gamma)` of an SEIR model to an incidence series by
/// Nelder-Mead on the sum of squared errors.
///
/// Fitting three rates to one incidence curve is close to the edge of what
/// the data supports: the growth rate constrains a *combination* of `beta`
/// and `sigma`, so the two trade off against each other along a valley in
/// the objective and are only weakly separated by the shape of the peak.
/// The returned fit reproduces the curve; it should not be read as three
/// independently identified parameters.
///
/// The initial infectious fraction is taken from the first observation
/// rather than estimated, so if that first point is noisy or the epidemic
/// was already under way when reporting began, the resulting time offset
/// appears as a residual that no choice of rates can remove. Fitting it as a
/// fourth parameter would trade that bias for a worse identifiability
/// problem than the one already described.
///
/// # Errors
/// Returns an error for fewer than five points, a negative incidence, or a
/// non-positive population.
pub fn seir_fit_to_incidence(
    incidence: &[f64],
    dt: f64,
    population: f64,
    guess: (f64, f64, f64),
) -> Result<(f64, f64, f64), GeomError> {
    if incidence.len() < 5 {
        return Err(GeomError::InvalidArgument("seir_fit_to_incidence needs five points"));
    }
    if incidence.iter().any(|c| *c < 0.0) || !(population > 0.0) || !(dt > 0.0) {
        return Err(GeomError::InvalidArgument("seir_fit_to_incidence: bad input"));
    }
    let total: f64 = incidence.iter().sum();
    if !(total > 0.0) {
        return Err(GeomError::Degenerate("the record contains no cases"));
    }
    let t_end = dt * (incidence.len() - 1) as f64;
    let i0 = (incidence[0] / population).max(1e-9);
    let objective = |p: &[f64; 3]| -> f64 {
        if p.iter().any(|v| !(*v > 0.0) || !v.is_finite()) {
            return f64::INFINITY;
        }
        let Ok(trace) = seir(p[0], p[1], p[2], 1.0 - i0, 0.0, i0, t_end) else {
            return f64::INFINITY;
        };
        // Modelled incidence is the rate of new infections, beta S I.
        let mut error = 0.0;
        for (k, observed) in incidence.iter().enumerate() {
            let want = dt * k as f64;
            let sample = trace
                .iter()
                .min_by(|a, b| (a.t - want).abs().partial_cmp(&(b.t - want).abs()).unwrap())
                .expect("non-empty");
            let modelled = p[0] * sample.s * sample.i * population;
            error += (modelled - observed) * (modelled - observed);
        }
        error
    };
    // Nelder-Mead in three dimensions.
    let mut simplex: Vec<([f64; 3], f64)> = Vec::with_capacity(4);
    let start = [guess.0, guess.1, guess.2];
    if start.iter().any(|v| !(*v > 0.0)) {
        return Err(GeomError::InvalidArgument("the initial guess must be positive"));
    }
    simplex.push((start, objective(&start)));
    for axis in 0..3 {
        let mut point = start;
        point[axis] *= 1.4;
        simplex.push((point, objective(&point)));
    }
    for _ in 0..2_000 {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let best = simplex[0].1;
        let worst = simplex[3].1;
        if (worst - best).abs() <= 1e-12 * best.abs().max(1e-12) {
            break;
        }
        let centroid: [f64; 3] = std::array::from_fn(|a| {
            simplex[..3].iter().map(|(p, _)| p[a]).sum::<f64>() / 3.0
        });
        let reflect: [f64; 3] =
            std::array::from_fn(|a| centroid[a] + (centroid[a] - simplex[3].0[a]));
        let reflected = objective(&reflect);
        if reflected < simplex[0].1 {
            let expand: [f64; 3] =
                std::array::from_fn(|a| centroid[a] + 2.0 * (centroid[a] - simplex[3].0[a]));
            let expanded = objective(&expand);
            simplex[3] = if expanded < reflected {
                (expand, expanded)
            } else {
                (reflect, reflected)
            };
        } else if reflected < simplex[2].1 {
            simplex[3] = (reflect, reflected);
        } else {
            let contract: [f64; 3] =
                std::array::from_fn(|a| centroid[a] + 0.5 * (simplex[3].0[a] - centroid[a]));
            let contracted = objective(&contract);
            if contracted < simplex[3].1 {
                simplex[3] = (contract, contracted);
            } else {
                let anchor = simplex[0].0;
                for entry in simplex.iter_mut().skip(1) {
                    let shrunk: [f64; 3] =
                        std::array::from_fn(|a| anchor[a] + 0.5 * (entry.0[a] - anchor[a]));
                    *entry = (shrunk, objective(&shrunk));
                }
            }
        }
    }
    // Restart from the best point with a fresh simplex. Nelder-Mead
    // contracts onto a direction and then stops exploring the others, so a
    // single run stalls short of the minimum on a valley -- which is exactly
    // the shape this objective has, since the growth rate constrains beta
    // and sigma only in combination.
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let restart = simplex[0].0;
    simplex.clear();
    simplex.push((restart, objective(&restart)));
    for axis in 0..3 {
        let mut point = restart;
        point[axis] *= 1.1;
        simplex.push((point, objective(&point)));
    }
    for _ in 0..2_000 {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let best = simplex[0].1;
        let worst = simplex[3].1;
        if (worst - best).abs() <= 1e-14 * best.abs().max(1e-14) {
            break;
        }
        let centroid: [f64; 3] = std::array::from_fn(|a| {
            simplex[..3].iter().map(|(p, _)| p[a]).sum::<f64>() / 3.0
        });
        let reflect: [f64; 3] =
            std::array::from_fn(|a| centroid[a] + (centroid[a] - simplex[3].0[a]));
        let reflected = objective(&reflect);
        if reflected < simplex[0].1 {
            let expand: [f64; 3] =
                std::array::from_fn(|a| centroid[a] + 2.0 * (centroid[a] - simplex[3].0[a]));
            let expanded = objective(&expand);
            simplex[3] = if expanded < reflected {
                (expand, expanded)
            } else {
                (reflect, reflected)
            };
        } else if reflected < simplex[2].1 {
            simplex[3] = (reflect, reflected);
        } else {
            let contract: [f64; 3] =
                std::array::from_fn(|a| centroid[a] + 0.5 * (simplex[3].0[a] - centroid[a]));
            let contracted = objective(&contract);
            if contracted < simplex[3].1 {
                simplex[3] = (contract, contracted);
            } else {
                let anchor = simplex[0].0;
                for entry in simplex.iter_mut().skip(1) {
                    let shrunk: [f64; 3] =
                        std::array::from_fn(|a| anchor[a] + 0.5 * (entry.0[a] - anchor[a]));
                    *entry = (shrunk, objective(&shrunk));
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    if !simplex[0].1.is_finite() {
        return Err(GeomError::Degenerate("the fit never found a feasible model"));
    }
    let p = simplex[0].0;
    Ok((p[0], p[1], p[2]))
}

/// The Wallinga-Teunis case reproduction number.
///
/// Where the Cori method asks "how many people is each *current* case
/// infecting", this asks "how many did each *past* case go on to infect",
/// by assigning each case's infector probabilistically among the earlier
/// cases in proportion to the serial interval. The two answer different
/// questions and disagree near the end of a record, where Wallinga-Teunis
/// is biased down because the infections have not happened yet.
///
/// # Errors
/// Returns an error for a negative incidence or a serial interval that is
/// not a distribution.
pub fn wallinga_teunis(
    incidence: &[f64],
    serial_interval: &[f64],
) -> Result<Vec<f64>, GeomError> {
    if incidence.is_empty() || incidence.iter().any(|c| *c < 0.0) {
        return Err(GeomError::InvalidArgument("the incidence must be non-negative"));
    }
    if serial_interval.is_empty() || serial_interval.iter().any(|w| *w < 0.0) {
        return Err(GeomError::InvalidArgument("the serial interval must be non-negative"));
    }
    let mass: f64 = serial_interval.iter().sum();
    if !(mass > 0.0) {
        return Err(GeomError::InvalidArgument("the serial interval carries no mass"));
    }
    let w: Vec<f64> = serial_interval.iter().map(|x| x / mass).collect();
    let days = incidence.len();
    let weight = |gap: usize| -> f64 {
        if gap == 0 || gap > w.len() {
            0.0
        } else {
            w[gap - 1]
        }
    };
    // p[j][i] is the probability that case-day j was infected from day i.
    let mut out = vec![0.0f64; days];
    for j in 0..days {
        let denominator: f64 = (0..j).map(|i| incidence[i] * weight(j - i)).sum();
        if !(denominator > 0.0) {
            continue;
        }
        for i in 0..j {
            let share = incidence[i] * weight(j - i) / denominator;
            // Every case on day j contributes that share to day i's total.
            out[i] += incidence[j] * share;
        }
    }
    Ok((0..days)
        .map(|i| if incidence[i] > 0.0 { out[i] / incidence[i] } else { f64::NAN })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------
    // Compartment models
    // -----------------------------------------------------------------

    #[test]
    fn every_compartment_model_conserves_its_population() {
        // The one invariant every model here shares, and it holds sample by
        // sample rather than on average -- the derivatives sum to zero
        // identically, so any drift is integration error and any jump is a
        // defect in the derivative.
        for &(beta, gamma) in &[(0.6f64, 0.2f64), (1.5, 1.0), (0.1, 0.5)] {
            let checks: Vec<Vec<EpidemicSample>> = vec![
                sir(beta, gamma, 0.99, 0.01, 60.0).unwrap(),
                sis(beta, gamma, 0.99, 0.01, 60.0).unwrap(),
                sirs(beta, gamma, 0.05, 0.99, 0.01, 60.0).unwrap(),
                seir(beta, 0.3, gamma, 0.99, 0.0, 0.01, 60.0).unwrap(),
                seirs(beta, 0.3, gamma, 0.05, 0.99, 0.0, 0.01, 60.0).unwrap(),
                sir_with_demography(beta, gamma, 0.01, 0.99, 0.01, 60.0).unwrap(),
            ];
            for (which, trace) in checks.iter().enumerate() {
                assert!(trace.len() > 10, "model {which} produced {} samples", trace.len());
                for sample in trace {
                    assert!(
                        close(sample.total(), 1.0, 1e-7),
                        "model {which} at t = {} sums to {}",
                        sample.t,
                        sample.total()
                    );
                    for value in [sample.s, sample.e, sample.i, sample.r] {
                        assert!(value >= -1e-9, "model {which} went negative: {value}");
                        assert!(value <= 1.0 + 1e-9, "model {which} exceeded one: {value}");
                    }
                }
                // Time advances monotonically and reaches the end.
                for pair in trace.windows(2) {
                    assert!(pair[1].t > pair[0].t);
                }
                assert!(close(trace.last().unwrap().t, 60.0, 1e-9));
            }
        }
        // MSIR and the two-strain model, which report their own tuples.
        let m = msir(0.6, 0.2, 0.1, 0.1, 0.89, 0.01, 60.0).unwrap();
        for (t, a, b, c, d) in &m {
            assert!(close(a + b + c + d, 1.0, 1e-7), "MSIR at t = {t} sums to {}", a + b + c + d);
        }
        let two = two_strain(0.6, 0.2, 0.5, 0.2, 0.98, 0.01, 0.01, 60.0).unwrap();
        for (t, a, b, c, d) in &two {
            assert!(close(a + b + c + d, 1.0, 1e-7), "two-strain at t = {t} sums to {}", a + b + c + d);
        }
    }

    #[test]
    fn the_sir_epidemic_grows_only_above_the_threshold_and_ends_at_the_final_size() {
        // Two closed-form checks. The epidemic grows if and only if
        // R0 * S0 > 1, exactly -- and the fraction ever infected is the root
        // of the final size equation, a number the integration never sees.
        for &r0 in &[0.5f64, 0.9, 1.1, 2.0, 4.0] {
            let gamma = 0.25;
            let beta = r0 * gamma;
            let i0 = 1e-6;
            // Near the threshold the epidemic is very slow: the growth rate
            // is gamma (R0 - 1), which at R0 = 1.1 is 0.025, so reaching
            // O(1) from a millionth alone takes some 550 time units. A fixed
            // horizon would truncate it and the final size would come out
            // short -- not a defect in the model but a run that had not
            // finished.
            let horizon = if r0 > 1.0 {
                400.0 + 60.0 / (gamma * (r0 - 1.0))
            } else {
                400.0
            };
            let trace = sir(beta, gamma, 1.0 - i0, i0, horizon).unwrap();
            let peak = trace.iter().map(|s| s.i).fold(0.0f64, f64::max);
            if r0 > 1.0 {
                assert!(peak > i0 * 2.0, "at R0 = {r0} the epidemic did not grow");
                assert!(
                    trace.last().unwrap().i < 1e-9,
                    "at R0 = {r0} the epidemic had not finished: {} still infectious",
                    trace.last().unwrap().i
                );
                let ever = 1.0 - trace.last().unwrap().s;
                let predicted = final_size_equation(r0).unwrap();
                assert!(
                    close(ever, predicted, 2e-3),
                    "at R0 = {r0} the epidemic reached {ever} against the predicted {predicted}"
                );
                // And it overshoots herd immunity, because the people
                // already infectious at the threshold go on infecting.
                let threshold = herd_immunity_threshold(r0).unwrap();
                assert!(
                    ever > threshold,
                    "at R0 = {r0} the final size {ever} did not overshoot the threshold {threshold}"
                );
            } else {
                assert!(peak <= i0 * 1.001, "at R0 = {r0} the epidemic grew from {i0} to {peak}");
                assert!(close(final_size_equation(r0).unwrap(), 0.0, 1e-9));
            }
        }
        // The peak occurs exactly where S crosses 1/R0, which is where the
        // growth rate changes sign.
        let (beta, gamma) = (0.75f64, 0.25f64);
        let trace = sir(beta, gamma, 1.0 - 1e-6, 1e-6, 400.0).unwrap();
        let peak = trace
            .iter()
            .max_by(|a, b| a.i.partial_cmp(&b.i).unwrap())
            .unwrap();
        assert!(
            close(peak.s, gamma / beta, 5e-3),
            "the peak was at S = {} rather than 1/R0 = {}",
            peak.s,
            gamma / beta
        );
    }

    #[test]
    fn sis_settles_at_its_endemic_equilibrium_rather_than_burning_out() {
        // The distinction from SIR: with no removed class the disease
        // persists at 1 - 1/R0 instead of running out of susceptibles.
        for &r0 in &[1.5f64, 3.0, 8.0] {
            let gamma = 0.3;
            let trace = sis(r0 * gamma, gamma, 0.99, 0.01, 400.0).unwrap();
            let settled = trace.last().unwrap();
            assert!(
                close(settled.i, 1.0 - 1.0 / r0, 1e-4),
                "at R0 = {r0} SIS settled at {} rather than {}",
                settled.i,
                1.0 - 1.0 / r0
            );
            // The corresponding SIR burns out entirely.
            let burnt = sir(r0 * gamma, gamma, 0.99, 0.01, 400.0).unwrap();
            assert!(
                burnt.last().unwrap().i < 1e-4,
                "SIR did not burn out: {} remain infectious",
                burnt.last().unwrap().i
            );
        }
        // Below threshold SIS dies out too.
        let dying = sis(0.1, 0.3, 0.99, 0.01, 400.0).unwrap();
        assert!(dying.last().unwrap().i < 1e-6);
    }

    #[test]
    fn demography_turns_a_one_off_epidemic_into_an_endemic_equilibrium() {
        // With births replenishing susceptibles the trajectory spirals into
        // S* = 1/R0 rather than burning out, and the approach is a damped
        // oscillation -- the source of the multi-year measles cycles.
        let (beta, gamma, mu) = (1.0f64, 0.2f64, 0.005f64);
        let r0 = beta / (gamma + mu);
        let trace = sir_with_demography(beta, gamma, mu, 0.99, 0.01, 4_000.0).unwrap();
        let settled = trace.last().unwrap();
        assert!(
            close(settled.s, 1.0 / r0, 5e-3),
            "the susceptible fraction settled at {} rather than {}",
            settled.s,
            1.0 / r0
        );
        assert!(settled.i > 1e-5, "the disease died out instead of becoming endemic");
        // It really oscillates on the way: the infectious fraction has more
        // than one local maximum.
        let mut peaks = 0;
        for w in trace.windows(3) {
            if w[1].i > w[0].i && w[1].i > w[2].i && w[1].i > 1e-4 {
                peaks += 1;
            }
        }
        assert!(peaks >= 2, "only {peaks} peaks: the approach is not oscillatory");
        // Without demography the same parameters burn out.
        let burnt = sir(beta, gamma, 0.99, 0.01, 4_000.0).unwrap();
        assert!(burnt.last().unwrap().i < 1e-9);
    }

    #[test]
    fn the_latent_period_slows_growth_without_changing_the_final_size() {
        // The final size depends on R0 alone, so SEIR and SIR with the same
        // R0 end in the same place however different the incubation. What
        // changes is the growth rate -- which is why two pathogens with the
        // same R0 look so different in the first month.
        let (beta, gamma) = (0.6f64, 0.2f64);
        let r0 = beta / gamma;
        let expected = final_size_equation(r0).unwrap();
        let mut times_to_peak = Vec::new();
        for &sigma in &[2.0f64, 0.5, 0.15] {
            let trace = seir(beta, sigma, gamma, 1.0 - 1e-6, 0.0, 1e-6, 1_500.0).unwrap();
            let ever = 1.0 - trace.last().unwrap().s;
            assert!(
                close(ever, expected, 3e-3),
                "at sigma = {sigma} the final size is {ever} against {expected}"
            );
            let peak = trace.iter().max_by(|a, b| a.i.partial_cmp(&b.i).unwrap()).unwrap();
            times_to_peak.push(peak.t);
        }
        // A longer latent period delays the peak, monotonically.
        for pair in times_to_peak.windows(2) {
            assert!(pair[1] > pair[0], "a longer latency did not delay the peak: {times_to_peak:?}");
        }
    }

    #[test]
    fn the_fitter_strain_wins_from_an_equal_start_and_a_head_start_can_beat_it() {
        // Competitive exclusion is a statement about a system that
        // replenishes its susceptibles -- there has to be an indefinite
        // future to be excluded from. A one-off epidemic is a finite race,
        // and both halves of that are checked here: from equal starts the
        // fitter strain wins, and given enough of a head start the less fit
        // one out-infects it before the susceptibles run out. The second
        // half is the interesting one, and asserting the textbook slogan
        // instead would have been asserting something false about this
        // model.
        let gamma = 0.2;
        let (fit, unfit) = (0.8f64, 0.5f64);
        let peaks = |head_start: f64| -> (f64, f64) {
            let trace = two_strain(
                fit,
                gamma,
                unfit,
                gamma,
                1.0 - head_start - 1e-6,
                1e-6,
                head_start,
                1_500.0,
            )
            .unwrap();
            let (_, _, i1, i2, _) = *trace.last().unwrap();
            assert!(i1 < 1e-6 && i2 < 1e-6, "a strain was still going at the end");
            (
                trace.iter().map(|(_, _, a, _, _)| *a).fold(0.0f64, f64::max),
                trace.iter().map(|(_, _, _, b, _)| *b).fold(0.0f64, f64::max),
            )
        };
        // From an equal start, fitness decides.
        let (even_fit, even_unfit) = peaks(1e-6);
        assert!(
            even_fit > 2.0 * even_unfit,
            "from equal starts the fitter strain only reached {even_fit} against {even_unfit}"
        );
        // From a thousandfold head start, it does not.
        let (behind_fit, ahead_unfit) = peaks(1e-3);
        assert!(
            ahead_unfit > behind_fit,
            "a thousandfold head start was not enough: {ahead_unfit} against {behind_fit}"
        );
        // And the advantage is monotone in the head start, so the crossover
        // is a real threshold rather than a numerical accident.
        let mut previous = f64::NEG_INFINITY;
        for step in 0..7 {
            let head_start = 10f64.powf(-6.0 + f64::from(step) * 0.6);
            let (f, u) = peaks(head_start);
            let advantage = u / f;
            assert!(
                advantage > previous,
                "a larger head start helped less: {advantage} after {previous}"
            );
            previous = advantage;
        }
        // Whatever happens, the susceptibles are what run out.
        let trace = two_strain(fit, gamma, unfit, gamma, 0.998, 0.001, 0.001, 1_500.0).unwrap();
        let (_, s, _, _, r) = *trace.last().unwrap();
        assert!(s < 0.05 && r > 0.9, "the epidemic ended with S = {s} and R = {r}");
    }

    #[test]
    fn vaccination_acts_exactly_as_a_reduced_susceptible_pool() {
        // Which is why the threshold coverage is the herd immunity
        // threshold, and why the effect on the final size is entirely
        // captured by R0 (1 - coverage).
        let (beta, gamma) = (0.75f64, 0.25f64);
        let r0 = beta / gamma;
        let threshold = herd_immunity_threshold(r0).unwrap();
        for &coverage in &[0.0f64, 0.2, 0.5] {
            let trace = sir_with_vaccination(beta, gamma, coverage, 1e-6, 600.0).unwrap();
            for sample in &trace {
                assert!(close(sample.total(), 1.0, 1e-7), "vaccination broke the total");
            }
            let ever = 1.0 - coverage - trace.last().unwrap().s;
            // The effective R0 among the unvaccinated.
            let effective = r0 * (1.0 - coverage);
            let expected = (1.0 - coverage) * final_size_equation_effective(effective);
            assert!(
                close(ever, expected, 5e-3),
                "at coverage {coverage} the epidemic reached {ever} against {expected}"
            );
        }
        // Above the threshold nothing takes off.
        let protected = sir_with_vaccination(beta, gamma, threshold + 0.05, 1e-6, 600.0).unwrap();
        let ever = 1.0 - (threshold + 0.05) - protected.last().unwrap().s;
        assert!(ever < 1e-4, "an epidemic ran despite herd immunity: {ever}");
        assert!(sir_with_vaccination(beta, gamma, 1.5, 1e-6, 10.0).is_err());
        assert!(sir_with_vaccination(beta, gamma, -0.1, 1e-6, 10.0).is_err());
    }

    /// The final size among the susceptible sub-population, for an epidemic
    /// whose effective reproduction number is `effective`.
    fn final_size_equation_effective(effective: f64) -> f64 {
        final_size_equation(effective).unwrap()
    }


    // -----------------------------------------------------------------
    // Structure
    // -----------------------------------------------------------------

    /// A complete graph, whose adjacency spectrum is known exactly.
    fn complete(n: usize) -> Graph {
        let mut g = Graph::new(n, false);
        for u in 0..n {
            for v in (u + 1)..n {
                g.add_edge(u, v, 1.0);
            }
        }
        g
    }

    /// A star: one hub joined to `n - 1` leaves.
    fn star(n: usize) -> Graph {
        let mut g = Graph::new(n, false);
        for v in 1..n {
            g.add_edge(0, v, 1.0);
        }
        g
    }

    #[test]
    fn the_network_threshold_is_set_by_the_spectral_radius_not_the_mean_degree() {
        // Two graphs with the same mean degree can have very different
        // thresholds. That is the whole point of the spectral criterion, and
        // the star is the sharpest case: its mean degree is just under two
        // however large it grows, while its spectral radius is sqrt(n - 1)
        // and grows without bound. Both are exact.
        for n in [4usize, 9, 16, 25, 50] {
            // A complete graph has spectral radius n - 1 exactly.
            let k = complete(n);
            assert!(
                close(epidemic_threshold_network(&k).unwrap(), 1.0 / (n as f64 - 1.0), 1e-8),
                "the complete graph on {n} gives {}",
                epidemic_threshold_network(&k).unwrap()
            );
            // A star has spectral radius sqrt(n - 1) exactly.
            let sn = star(n);
            assert!(
                close(
                    epidemic_threshold_network(&sn).unwrap(),
                    1.0 / (n as f64 - 1.0).sqrt(),
                    1e-8
                ),
                "the star on {n} gives {}",
                epidemic_threshold_network(&sn).unwrap()
            );
            // The star's mean degree stays below two while its threshold
            // keeps falling: an average cannot express this.
            let mean_degree = 2.0 * (n as f64 - 1.0) / n as f64;
            assert!(mean_degree < 2.0);
            assert!(
                epidemic_threshold_network(&sn).unwrap() < 1.0 / mean_degree,
                "the star's threshold is no lower than a mean-degree estimate at n = {n}"
            );
        }
        // The threshold falls as edges are added, always.
        let mut previous = f64::INFINITY;
        for n in 3..=20 {
            let t = epidemic_threshold_network(&complete(n)).unwrap();
            assert!(t < previous);
            previous = t;
        }
        assert!(epidemic_threshold_network(&Graph::new(0, false)).is_err());
        assert!(epidemic_threshold_network(&Graph::new(5, false)).is_err());
    }

    #[test]
    fn the_age_structured_r0_is_the_dominant_eigenvalue_not_the_mean_contact_rate() {
        // With uniform contact the next-generation matrix has one non-zero
        // eigenvalue and R0 reduces to the well-mixed value, which pins the
        // normalisation. With assortative contact it does not, and the
        // difference is the whole reason to structure the model.
        let gamma = 0.2;
        for groups in [2usize, 3, 4] {
            let sizes: Vec<f64> = vec![1.0 / groups as f64; groups];
            let rate = 0.5;
            let uniform: Vec<Vec<f64>> = vec![vec![rate; groups]; groups];
            let r0 = r0_age_structured(&uniform, &sizes, gamma).unwrap();
            assert!(
                close(r0, groups as f64 * rate / gamma, 1e-6 * r0),
                "uniform contact on {groups} groups gives R0 = {r0}"
            );

            // Concentrating the same contact *within* groups changes
            // nothing when the groups are the same size and equally active:
            // the next-generation matrix has the same dominant eigenvalue,
            // 5.0 either way. Assortativity alone is not what raises R0, and
            // asserting that it does would have been asserting something
            // false.
            let mut assortative = vec![vec![0.0; groups]; groups];
            for (a, row) in assortative.iter_mut().enumerate() {
                row[a] = rate * groups as f64;
            }
            let assorted = r0_age_structured(&assortative, &sizes, gamma).unwrap();
            assert!(
                close(assorted, r0, 1e-6 * r0),
                "equal-sized, equally active groups gave {assorted} against {r0}"
            );
        }

        // What *does* raise it is heterogeneous activity. Under
        // proportionate mixing the next-generation matrix has dominant
        // eigenvalue <k^2> / (<k> gamma) rather than <k> / gamma, so a
        // population with the same mean contact rate but unequal activity
        // has a strictly larger R0 -- and it grows with the spread. This is
        // the reason a mean contact rate is not enough to characterise an
        // epidemic.
        let halves = vec![0.5f64, 0.5];
        let proportionate = |k: [f64; 2]| -> Vec<Vec<f64>> {
            let mean = 0.5 * (k[0] + k[1]);
            (0..2)
                .map(|a| (0..2).map(|b| k[a] * k[b] * halves[b] / mean).collect())
                .collect()
        };
        let homogeneous = r0_age_structured(&proportionate([1.0, 1.0]), &halves, gamma).unwrap();
        assert!(close(homogeneous, 1.0 / gamma, 1e-6), "the uniform case gives {homogeneous}");
        let mut previous = homogeneous;
        for spread in [0.5f64, 0.8, 0.9] {
            let k = [1.0 - spread, 1.0 + spread];
            let heterogeneous = r0_age_structured(&proportionate(k), &halves, gamma).unwrap();
            assert!(
                heterogeneous > previous,
                "a spread of {spread} gave R0 = {heterogeneous}, no more than {previous}"
            );
            // The closed form: <k^2> / (<k> gamma), with <k> = 1.
            let expected = (1.0 + spread * spread) / gamma;
            assert!(
                close(heterogeneous, expected, 1e-6 * expected),
                "at spread {spread} R0 is {heterogeneous} against the predicted {expected}"
            );
            previous = heterogeneous;
        }
        // A small, highly connected core group. Its self-contact rate of 5
        // against a size of a tenth gives a next-generation entry of 25, and
        // that single entry sets R0 for the whole population -- far above
        // any average of the two groups' own reproduction numbers. A core
        // group can sustain an epidemic the rest of the population could
        // not, which is the practical reason to structure a model at all.
        let sizes = vec![0.9, 0.1];
        let contact = vec![vec![0.1, 0.1], vec![0.1, 5.0]];
        let r0 = r0_age_structured(&contact, &sizes, gamma).unwrap();
        assert!(r0 > 0.0 && r0.is_finite());
        let naive = (0.2 / gamma + 5.1 / gamma) / 2.0;
        assert!(
            r0 > 1.5 * naive,
            "the dominant eigenvalue {r0} did not exceed the crude average {naive}"
        );
        // It is essentially the core group's own self-reproduction number.
        assert!(close(r0, 5.0 * 0.1 / (gamma * 0.1), 0.05 * r0), "R0 came out {r0}");
        // Weakening only the core group's internal contact collapses it,
        // even though the population-average contact barely moves.
        let calmer = vec![vec![0.1, 0.1], vec![0.1, 0.3]];
        let lowered = r0_age_structured(&calmer, &sizes, gamma).unwrap();
        assert!(lowered < 0.2 * r0, "damping the core group left R0 at {lowered}");

        // And the model integrates consistently: the population is conserved
        // group by group.
        let trace = age_structured(&contact, &sizes, gamma, &[0.0, 1e-4], 200.0).unwrap();
        for (t, s, i, r) in &trace {
            for a in 0..2 {
                assert!(
                    close(s[a] + i[a] + r[a], sizes[a], 1e-7),
                    "group {a} at t = {t} sums to {}",
                    s[a] + i[a] + r[a]
                );
            }
        }
        assert!(age_structured(&contact, &[0.5, 0.4], gamma, &[0.0, 1e-4], 10.0).is_err());
        assert!(age_structured(&contact, &sizes, gamma, &[1e-4], 10.0).is_err());
        assert!(age_structured(&[vec![0.1]], &sizes, gamma, &[0.0, 1e-4], 10.0).is_err());
        assert!(r0_age_structured(&contact, &sizes, 0.0).is_err());
    }

    // -----------------------------------------------------------------
    // Stochastic epidemics
    // -----------------------------------------------------------------

    #[test]
    fn the_stochastic_epidemic_dies_out_at_the_rate_the_branching_process_predicts() {
        // The thing the deterministic model cannot express: with R0 = 3 an
        // epidemic started from one case fails about a third of the time.
        // The prediction is (1/R0)^i0 from the branching approximation, and
        // it is accurate while susceptibles are undepleted -- which is
        // exactly the regime the failures happen in.
        let mut rng = Rng::new(0x0B10_0001);
        let n = 20_000u64;
        let gamma = 1.0;
        for &r0 in &[2.0f64, 3.0, 5.0] {
            for &i0 in &[1u64, 2] {
                let runs = 3_000;
                let mut died = 0;
                for _ in 0..runs {
                    let trace =
                        sir_stochastic_gillespie(r0 * gamma, gamma, n, i0, 200.0, &mut rng).unwrap();
                    let (_, _, _, r) = *trace.last().unwrap();
                    // "Died out" means it never established: only a handful
                    // of cases before the chain broke.
                    if r < 50 {
                        died += 1;
                    }
                }
                let observed = f64::from(died) / f64::from(runs);
                let predicted = extinction_probability_epidemic(r0, i0 as u32).unwrap();
                assert!(
                    close(observed, predicted, 0.04),
                    "at R0 = {r0}, i0 = {i0}: {observed} died out against a predicted {predicted}"
                );
            }
        }
    }

    #[test]
    fn a_large_stochastic_epidemic_tracks_the_deterministic_one() {
        // The other half of the same story: conditioned on taking off, and
        // with a large population, the stochastic trajectory follows the
        // deterministic curve. If it did not, one of the two models would be
        // wrong -- they are meant to be the same system at different scales.
        let mut rng = Rng::new(0x0B10_0002);
        let n = 200_000u64;
        let (beta, gamma) = (0.6f64, 0.2f64);
        let i0 = 400u64;
        let mut finals = Vec::new();
        for _ in 0..12 {
            let trace = sir_stochastic_gillespie(beta, gamma, n, i0, 400.0, &mut rng).unwrap();
            let (_, s, i, r) = *trace.last().unwrap();
            assert_eq!(s + i + r, n, "the stochastic model lost an individual");
            finals.push(r as f64 / n as f64);
        }
        let mean: f64 = finals.iter().sum::<f64>() / finals.len() as f64;
        let deterministic = sir(beta, gamma, 1.0 - i0 as f64 / n as f64, i0 as f64 / n as f64, 400.0)
            .unwrap();
        let predicted = deterministic.last().unwrap().r;
        assert!(
            close(mean, predicted, 0.02),
            "the stochastic mean final size is {mean} against the deterministic {predicted}"
        );
        // Every trajectory is monotone in R and S.
        let one = sir_stochastic_gillespie(beta, gamma, n, i0, 400.0, &mut rng).unwrap();
        for pair in one.windows(2) {
            assert!(pair[1].1 <= pair[0].1, "susceptibles increased");
            assert!(pair[1].3 >= pair[0].3, "removed decreased");
            assert!(pair[1].0 >= pair[0].0, "time went backwards");
        }
        assert!(sir_stochastic_gillespie(beta, gamma, 0, 0, 10.0, &mut rng).is_err());
        assert!(sir_stochastic_gillespie(beta, gamma, 10, 11, 10.0, &mut rng).is_err());
        assert!(sir_stochastic_gillespie(beta, gamma, 10, 1, 0.0, &mut rng).is_err());
        assert!(sir_stochastic_gillespie(-1.0, gamma, 10, 1, 10.0, &mut rng).is_err());
    }

    #[test]
    fn the_network_epidemic_is_bounded_by_its_own_component() {
        // Structure limits the epidemic in a way the well-mixed model cannot
        // see: an epidemic cannot leave the connected component it started
        // in, whatever the transmission rate. Checked against a deliberately
        // disconnected graph, where the bound is exact and known.
        let mut rng = Rng::new(0x0B10_0003);
        let mut split = Graph::new(20, false);
        for u in 0..9 {
            split.add_edge(u, u + 1, 1.0);
        }
        for u in 10..19 {
            split.add_edge(u, u + 1, 1.0);
        }
        for _ in 0..20 {
            let trace = network_sir(&split, 50.0, 0.01, 0, &mut rng).unwrap();
            let (s, i, r) = *trace.last().unwrap();
            assert_eq!(i, 0, "the epidemic did not finish");
            assert_eq!(s + i + r, 20, "the network model lost a node");
            assert!(r <= 10, "the epidemic escaped its component: {r} infected");
            assert!(s >= 10, "the other component was touched");
        }
        // On a connected graph with overwhelming transmission, everyone gets
        // it; with none, nobody but patient zero.
        let k = complete(15);
        let all = network_sir(&k, 1_000.0, 0.001, 0, &mut rng).unwrap();
        assert_eq!(all.last().unwrap().2, 15, "a strong epidemic missed someone");
        let none = network_sir(&k, 0.0, 1.0, 0, &mut rng).unwrap();
        assert_eq!(none.last().unwrap().2, 1, "an epidemic spread with no transmission");
        // Counts are consistent at every step.
        for (s, i, r) in &all {
            assert_eq!(s + i + r, 15);
        }
        assert!(network_sir(&Graph::new(0, false), 1.0, 1.0, 0, &mut rng).is_err());
        assert!(network_sir(&k, 1.0, 1.0, 15, &mut rng).is_err());
        assert!(network_sir(&k, -1.0, 1.0, 0, &mut rng).is_err());
    }

    // -----------------------------------------------------------------
    // Estimation
    // -----------------------------------------------------------------

    #[test]
    fn the_effective_r_estimate_recovers_a_known_reproduction_number() {
        // Built from a renewal process with a chosen R and serial interval,
        // so the answer is known by construction rather than remembered.
        // Checked at several R and two serial intervals, because the point
        // of the method is that it separates the two -- a naive ratio of
        // consecutive counts cannot.
        for &weights in &[
            [0.2f64, 0.4, 0.3, 0.1].as_slice(),
            [0.05f64, 0.1, 0.2, 0.3, 0.2, 0.15].as_slice(),
        ] {
            for &r in &[0.7f64, 1.0, 1.6, 2.5] {
                let days = 90;
                let mut incidence = vec![0.0; days];
                incidence[0] = 100.0;
                for day in 1..days {
                    let force: f64 = weights
                        .iter()
                        .enumerate()
                        .filter(|(s, _)| day > *s)
                        .map(|(s, w)| w * incidence[day - s - 1])
                        .sum();
                    incidence[day] = r * force;
                }
                let estimate = effective_r_estimate(&incidence, weights, 7).unwrap();
                for day in (days / 2)..days {
                    assert!(
                        close(estimate[day], r, 0.02 * r),
                        "at R = {r} day {day} the estimate is {}",
                        estimate[day]
                    );
                }
                // Before the window there is no estimate at all, which is
                // the honest answer rather than a number.
                assert!(estimate[..7].iter().all(|v| v.is_nan()));

                // The naive ratio of consecutive counts is a *growth rate*,
                // not a reproduction number, and it differs -- by more the
                // further R is from one.
                let naive = incidence[days - 1] / incidence[days - 2];
                if (r - 1.0).abs() > 0.2 {
                    assert!(
                        (naive - r).abs() > 0.05 * r,
                        "the naive ratio {naive} matched R = {r}, so the fixture shows nothing"
                    );
                }
            }
        }
        assert!(effective_r_estimate(&[1.0, 2.0], &[1.0], 5).is_err());
        assert!(effective_r_estimate(&[1.0, -2.0, 3.0], &[1.0], 1).is_err());
        assert!(effective_r_estimate(&[1.0, 2.0, 3.0], &[], 1).is_err());
        assert!(effective_r_estimate(&[1.0, 2.0, 3.0], &[0.0], 1).is_err());
        assert!(effective_r_estimate(&[1.0, 2.0, 3.0], &[1.0], 0).is_err());
    }

    #[test]
    fn the_serial_interval_fit_recovers_the_moments_it_was_given() {
        // Method of moments, so on a sample the fit must reproduce the
        // sample's own mean and variance exactly -- that is what the method
        // *is*, and it is checkable without any distributional assumption.
        let mut rng = Rng::new(0x0B10_0010);
        for &(shape, scale) in &[(2.0f64, 1.5f64), (5.0, 0.8), (9.0, 0.5)] {
            // Gamma by summing exponentials, valid for an integer shape.
            let draws: Vec<f64> = (0..20_000)
                .map(|_| {
                    (0..shape as u32)
                        .map(|_| -scale * (1.0 - rng.next_f64()).ln())
                        .sum::<f64>()
                })
                .collect();
            let (fit_shape, fit_scale) = serial_interval_fit(&draws).unwrap();
            assert!(
                close(fit_shape, shape, 0.15 * shape),
                "the shape {shape} came back as {fit_shape}"
            );
            assert!(
                close(fit_scale, scale, 0.15 * scale),
                "the scale {scale} came back as {fit_scale}"
            );
            // Exactly reproducing the sample moments is the definition.
            let n = draws.len() as f64;
            let mean: f64 = draws.iter().sum::<f64>() / n;
            let variance: f64 =
                draws.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
            assert!(close(fit_shape * fit_scale, mean, 1e-9 * mean));
            assert!(close(fit_shape * fit_scale * fit_scale, variance, 1e-9 * variance));
        }
        assert!(serial_interval_fit(&[1.0]).is_err());
        assert!(serial_interval_fit(&[1.0, 0.0]).is_err());
        assert!(serial_interval_fit(&[2.0; 10]).is_err());
    }

    #[test]
    fn wallinga_teunis_and_cori_agree_in_the_middle_and_diverge_at_the_end() {
        // They answer different questions -- how many is each current case
        // infecting, against how many did each past case go on to infect --
        // and the difference shows where it should: at the end of the
        // record, where Wallinga-Teunis is biased down because the
        // infections have not happened yet.
        let weights = [0.3f64, 0.4, 0.2, 0.1];
        let r = 1.8;
        let days = 80;
        let mut incidence = vec![0.0; days];
        incidence[0] = 100.0;
        for day in 1..days {
            let force: f64 = weights
                .iter()
                .enumerate()
                .filter(|(s, _)| day > *s)
                .map(|(s, w)| w * incidence[day - s - 1])
                .sum();
            incidence[day] = r * force;
        }
        let wt = wallinga_teunis(&incidence, &weights).unwrap();
        let cori = effective_r_estimate(&incidence, &weights, 7).unwrap();
        // In the middle, both recover R.
        for day in 30..50 {
            assert!(
                close(wt[day], r, 0.05 * r),
                "Wallinga-Teunis at day {day} is {}",
                wt[day]
            );
            assert!(close(cori[day], r, 0.05 * r), "Cori at day {day} is {}", cori[day]);
        }
        // At the end Wallinga-Teunis collapses toward zero while Cori does
        // not, because the future infections are missing from the record.
        assert!(
            wt[days - 1] < 0.3 * r,
            "the end effect is absent: {} at the last day",
            wt[days - 1]
        );
        assert!(
            close(cori[days - 1], r, 0.05 * r),
            "Cori was affected by the end of the record: {}",
            cori[days - 1]
        );
        assert!(wallinga_teunis(&[], &weights).is_err());
        assert!(wallinga_teunis(&[1.0, -1.0], &weights).is_err());
        assert!(wallinga_teunis(&incidence, &[]).is_err());
        assert!(wallinga_teunis(&incidence, &[0.0, 0.0]).is_err());
    }

    #[test]
    fn the_seir_fit_reproduces_a_curve_it_was_generated_from() {
        // The honest claim, and the one the documentation makes: the fit
        // reproduces the incidence curve. It is not claimed to identify
        // three parameters independently, and this test does not pretend it
        // does -- it checks the curve, and separately that the recovered R0
        // is close, since that combination *is* well determined.
        let population = 1e6;
        let (beta, sigma, gamma) = (0.9f64, 0.4f64, 0.3f64);
        let dt = 1.0;
        let days = 60;
        let i0 = 1e-5;
        let truth = seir(beta, sigma, gamma, 1.0 - i0, 0.0, i0, dt * (days - 1) as f64).unwrap();
        let incidence: Vec<f64> = (0..days)
            .map(|k| {
                let want = dt * k as f64;
                let sample = truth
                    .iter()
                    .min_by(|a, b| (a.t - want).abs().partial_cmp(&(b.t - want).abs()).unwrap())
                    .unwrap();
                beta * sample.s * sample.i * population
            })
            .collect();
        let (fb, fs, fg) =
            seir_fit_to_incidence(&incidence, dt, population, (0.6, 0.6, 0.2)).unwrap();
        // The curve is reproduced.
        let fitted = seir(fb, fs, fg, 1.0 - i0, 0.0, i0, dt * (days - 1) as f64).unwrap();
        let mut worst: f64 = 0.0;
        let scale = incidence.iter().copied().fold(0.0, f64::max);
        for (k, observed) in incidence.iter().enumerate() {
            let want = dt * k as f64;
            let sample = fitted
                .iter()
                .min_by(|a, b| (a.t - want).abs().partial_cmp(&(b.t - want).abs()).unwrap())
                .unwrap();
            let modelled = fb * sample.s * sample.i * population;
            worst = worst.max((modelled - observed).abs() / scale);
        }
        // Five per cent of the peak at the worst point. The residual is
        // dominated by the initial condition, which the fit does not
        // estimate: the truth used i0 = 1e-5 while the fit infers
        // incidence[0] / population = 9e-6 from the first observation, and a
        // ten per cent offset in the seed shifts the whole curve in time in
        // a way no choice of rates can absorb.
        assert!(worst < 0.08, "the fitted curve is {worst} of the peak away at its worst");
        // And R0, which the growth rate does determine, comes back close.
        let fitted_r0 = fb / fg;
        assert!(
            close(fitted_r0, beta / gamma, 0.2 * beta / gamma),
            "R0 came back as {fitted_r0} against {}",
            beta / gamma
        );
        assert!(seir_fit_to_incidence(&incidence[..3], dt, population, (0.6, 0.6, 0.2)).is_err());
        assert!(seir_fit_to_incidence(&incidence, dt, 0.0, (0.6, 0.6, 0.2)).is_err());
        assert!(seir_fit_to_incidence(&[0.0; 10], dt, population, (0.6, 0.6, 0.2)).is_err());
        assert!(seir_fit_to_incidence(&incidence, dt, population, (0.0, 0.6, 0.2)).is_err());
    }

    // -----------------------------------------------------------------
    // Thresholds
    // -----------------------------------------------------------------

    #[test]
    fn the_threshold_quantities_agree_with_their_closed_forms() {
        for &(beta, gamma) in &[(0.6f64, 0.2f64), (1.0, 1.0), (0.1, 2.0)] {
            assert!(close(r0_sir(beta, gamma).unwrap(), beta / gamma, 1e-15));
        }
        assert!(r0_sir(0.5, 0.0).is_err());
        assert!(r0_sir(-0.5, 1.0).is_err());

        // Herd immunity rises with R0 and reaches one only in the limit.
        assert!(close(herd_immunity_threshold(1.0).unwrap(), 0.0, 1e-15));
        assert!(close(herd_immunity_threshold(2.0).unwrap(), 0.5, 1e-15));
        assert!(close(herd_immunity_threshold(4.0).unwrap(), 0.75, 1e-15));
        assert!(herd_immunity_threshold(1e12).unwrap() < 1.0);
        assert!(herd_immunity_threshold(0.9).is_err());

        // The final size solves its own equation, which is the check that
        // matters: 1 - z = exp(-R0 z) to machine precision.
        for &r0 in &[1.001f64, 1.5, 2.5, 6.0, 20.0] {
            let z = final_size_equation(r0).unwrap();
            assert!(
                close(1.0 - z, (-r0 * z).exp(), 1e-11),
                "at R0 = {r0} the root {z} does not satisfy the equation"
            );
            assert!(z > 0.0 && z < 1.0);
            assert!(z > herd_immunity_threshold(r0).unwrap(), "the final size undershot herd immunity");
        }
        // It rises with R0 and tends to one.
        let mut previous = 0.0;
        for step in 1..=40 {
            let z = final_size_equation(1.0 + f64::from(step) * 0.25).unwrap();
            assert!(z > previous);
            previous = z;
        }
        assert!(final_size_equation(50.0).unwrap() > 0.999);
        assert!(final_size_equation(-1.0).is_err());
    }

    #[test]
    fn extinction_is_certain_below_threshold_and_common_above_it() {
        // The counterintuitive part, and the reason it is worth having: an
        // R0 of three still fails from a single introduction a third of the
        // time. Epidemics are the survivors of many introductions that were
        // not.
        assert!(close(extinction_probability_epidemic(0.5, 1).unwrap(), 1.0, 1e-15));
        assert!(close(extinction_probability_epidemic(1.0, 5).unwrap(), 1.0, 1e-15));
        assert!(close(extinction_probability_epidemic(3.0, 1).unwrap(), 1.0 / 3.0, 1e-12));
        assert!(close(extinction_probability_epidemic(3.0, 2).unwrap(), 1.0 / 9.0, 1e-12));
        // More introductions make extinction rapidly less likely.
        let mut previous = 1.0;
        for i0 in 1..=20 {
            let p = extinction_probability_epidemic(2.0, i0).unwrap();
            assert!(p < previous && p > 0.0);
            previous = p;
        }
        assert!(extinction_probability_epidemic(2.0, 0).is_err());
        assert!(extinction_probability_epidemic(-1.0, 1).is_err());
    }
}
