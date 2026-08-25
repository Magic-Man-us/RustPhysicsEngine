//! Chemical kinetics: rate laws, deterministic and stochastic reaction
//! networks, enzyme saturation, equilibrium composition, oscillating
//! mechanisms, nucleation and transformation, and the acid-base and
//! electrochemical relations that share their arithmetic.
//!
//! # What lives here and what lives in `chemistry`
//!
//! The elementary single-formula relations -- the Arrhenius rate, the
//! equilibrium constant from a free energy, the Nernst potential, pH from a
//! proton concentration -- are already in `crate::chemistry`, and are not
//! duplicated. This module is the part that needs a solver: networks
//! integrated in time, fits inverted from data, compositions found by
//! root-finding, and the stochastic algorithms.
//!
//! # Units
//!
//! Concentrations are molar, times are seconds, energies are joules per mole
//! and temperatures are kelvin, so `R` rather than `k_B` appears throughout.
//! The one exception is [`kramers_rate_check`], which follows its own
//! literature convention of barrier heights in units of `k_B T`; it is
//! marked at the function.

use crate::error::GeomError;
use crate::linalg::Matrix;
use crate::math::constants;
use crate::monte_carlo::Rng;
use crate::numerical::ode::implicit::backward_euler;

/// One elementary reaction, as species indices with their stoichiometric
/// coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reaction {
    /// `(species, coefficient)` consumed.
    pub reactants: Vec<(usize, u32)>,
    /// `(species, coefficient)` produced.
    pub products: Vec<(usize, u32)>,
}

impl Reaction {
    /// A reaction from reactant and product lists.
    #[must_use]
    pub fn new(reactants: &[(usize, u32)], products: &[(usize, u32)]) -> Self {
        Self { reactants: reactants.to_vec(), products: products.to_vec() }
    }

    /// The molecularity: how many molecules meet.
    #[must_use]
    pub fn order(&self) -> u32 {
        self.reactants.iter().map(|(_, n)| *n).sum()
    }

    /// The net change in each species, indexed by species.
    #[must_use]
    pub fn net_change(&self, species: usize) -> Vec<i64> {
        let mut delta = vec![0i64; species];
        for (s, n) in &self.reactants {
            if *s < species {
                delta[*s] -= i64::from(*n);
            }
        }
        for (s, n) in &self.products {
            if *s < species {
                delta[*s] += i64::from(*n);
            }
        }
        delta
    }
}

/// The stoichiometry matrix: species by reaction, each entry the net change
/// in that species when that reaction fires once.
///
/// # Errors
/// Returns an error for no reactions, no species, or a species index outside
/// the declared count.
pub fn stoichiometry_matrix(
    reactions: &[Reaction],
    species: usize,
) -> Result<Matrix, GeomError> {
    if reactions.is_empty() || species == 0 {
        return Err(GeomError::InvalidArgument("stoichiometry_matrix: empty network"));
    }
    for r in reactions {
        if r.reactants.iter().chain(&r.products).any(|(s, _)| *s >= species) {
            return Err(GeomError::InvalidArgument("a species index is out of range"));
        }
    }
    let mut m = Matrix::zeros(species, reactions.len());
    for (j, r) in reactions.iter().enumerate() {
        for (i, d) in r.net_change(species).into_iter().enumerate() {
            m.set(i, j, d as f64);
        }
    }
    Ok(m)
}

/// The deterministic mass-action rate of each reaction at a composition.
///
/// `v_j = k_j prod_i c_i^m_ij`. Note the contrast with the stochastic
/// propensity in [`gillespie_ssa`], which uses a falling factorial rather
/// than a power: a bimolecular reaction of a species with itself has rate
/// `k c^2` in the continuum and `k x (x - 1) / 2` in molecule counts, and
/// the two agree only when the count is large. Conflating them is the
/// classic way to get a stochastic simulation that quietly disagrees with
/// its own rate equations.
///
/// # Errors
/// Returns an error for a rate constant per reaction mismatch, a negative
/// rate constant, or a species index outside the composition.
pub fn mass_action_rates(
    reactions: &[Reaction],
    k: &[f64],
    concentrations: &[f64],
) -> Result<Vec<f64>, GeomError> {
    if reactions.len() != k.len() {
        return Err(GeomError::InvalidArgument("one rate constant per reaction"));
    }
    if k.iter().any(|v| *v < 0.0 || !v.is_finite()) {
        return Err(GeomError::InvalidArgument("every rate constant must be finite and positive"));
    }
    // Every species the network mentions must have a concentration, not just
    // the ones that happen to appear as reactants: a composition too short
    // to cover the products is a mismatched network, and accepting it
    // silently would let a caller integrate the wrong system.
    let mentioned = reactions
        .iter()
        .flat_map(|r| r.reactants.iter().chain(&r.products))
        .map(|(s, _)| *s)
        .max()
        .unwrap_or(0);
    if mentioned >= concentrations.len() {
        return Err(GeomError::InvalidArgument("a species index is out of range"));
    }
    reactions
        .iter()
        .zip(k)
        .map(|(r, rate)| {
            let mut v = *rate;
            for (s, n) in &r.reactants {
                v *= concentrations[*s].max(0.0).powi(*n as i32);
            }
            Ok(v)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Deterministic integration
// ---------------------------------------------------------------------------

/// Integrates a reaction network in time with an adaptive implicit method.
///
/// Chemical networks are almost always stiff -- a fast pre-equilibrium
/// alongside a slow overall conversion means the fastest and slowest
/// timescales differ by orders of magnitude -- and an explicit integrator is
/// then limited by the *fastest* one long after it has ceased to matter.
/// The step here is backward Euler -- A-stable, and L-stable, so a mode far
/// faster than the step is damped rather than merely bounded -- taken once
/// at the full step and twice at half. The difference is the local error
/// estimate, and their Richardson combination `2 y_half - y_full` is the
/// second-order value actually kept.
///
/// A multistep formula would be the conventional choice and is the wrong one
/// here: BDF2 assumes a uniform step, and an adaptive controller varies it
/// every step, so the history it is handed is at the wrong spacing and the
/// resulting inconsistency dominates the error estimate. A one-step method
/// with Richardson has no history to get wrong.
///
/// The step is limited by the solution's own timescale as well as by the
/// error estimate, and that second limit is not redundant. On an
/// *oscillatory* system step doubling can be fooled outright: an L-stable
/// method damps hard at a step much longer than the period, so the coarse
/// and fine solutions both collapse toward the fixed point, agree closely
/// with each other, and report a small error -- whereupon the controller
/// grows the step further. A run can end up stepping clean over whole
/// oscillations while its error estimate reports success. Bounding the step
/// by `|c| / |dc/dt|` prevents that, because it looks at the dynamics rather
/// than at the difference between two equally wrong answers.
///
/// Returns `(time, composition)` at each accepted step.
///
/// # Errors
/// Returns an error for a mismatched initial composition, a non-positive
/// end time or tolerance, or if the Newton iteration inside a step fails to
/// converge even at the smallest permitted step.
pub fn rate_equations(
    stoich: &Matrix,
    rates: &dyn Fn(&[f64]) -> Vec<f64>,
    c0: &[f64],
    t_end: f64,
    rtol: f64,
) -> Result<Vec<(f64, Vec<f64>)>, GeomError> {
    let species = stoich.rows;
    if c0.len() != species {
        return Err(GeomError::InvalidArgument("the initial composition has the wrong length"));
    }
    if !(t_end > 0.0) || !(rtol > 0.0) || rtol >= 1.0 {
        return Err(GeomError::InvalidArgument("rate_equations: bad parameters"));
    }
    let derivative = |_t: f64, c: &[f64]| -> Vec<f64> {
        let v = rates(c);
        (0..species)
            .map(|i| (0..stoich.cols).map(|j| stoich.get(i, j) * v[j]).sum())
            .collect()
    };

    let scale: f64 = c0.iter().fold(1e-12, |a, b| a.max(b.abs()));
    let mut out = vec![(0.0, c0.to_vec())];
    let mut t = 0.0;
    let mut dt = (t_end * 1e-6).min(1e-3);
    let smallest = t_end * 1e-14;
    let mut current = c0.to_vec();

    while t < t_end {
        dt = dt.min(t_end - t);
        if dt < smallest {
            return Err(GeomError::Degenerate("the step collapsed below the working precision"));
        }
        let step = |from: &[f64], at: f64, h: f64| -> Option<Vec<f64>> {
            backward_euler(&derivative, None, at, from, h, 1e-12 * scale, 60).ok()
        };
        // One full step against two half steps.
        let whole = step(&current, t, dt);
        let half = step(&current, t, 0.5 * dt)
            .and_then(|mid| step(&mid, t + 0.5 * dt, 0.5 * dt));
        let (Some(whole), Some(fine)) = (whole, half) else {
            dt *= 0.25;
            continue;
        };
        let error = (0..species)
            .map(|i| (whole[i] - fine[i]).abs() / (fine[i].abs().max(scale)))
            .fold(0.0, f64::max);
        if error <= rtol || dt <= smallest * 10.0 {
            // Richardson: backward Euler's error is first order, so twice
            // the half-step result minus the full-step one cancels it and
            // leaves second order. Concentrations cannot be negative, and a
            // step that undershoots zero is a numerical artefact -- clamping
            // there is the difference between a slow decay and a blow-up.
            current = (0..species)
                .map(|i| (2.0 * fine[i] - whole[i]).max(0.0))
                .collect();
            t += dt;
            out.push((t, current.clone()));
        }
        // Backward Euler's local error is O(h^2), so the step scales with
        // the square root of the tolerance ratio.
        let growth = if error > 0.0 { 0.9 * (rtol / error).sqrt() } else { 5.0 };
        dt *= growth.clamp(0.2, 5.0);
        // And no step may outrun the solution's own timescale, whatever the
        // error estimate says. See the note above: on an oscillatory system
        // the estimate can be fooled into approving a step that skips whole
        // periods.
        let derivatives = derivative(t, &current);
        let fastest = (0..species)
            .map(|i| derivatives[i].abs() / current[i].abs().max(scale))
            .fold(0.0, f64::max);
        if fastest > 0.0 {
            dt = dt.min(0.25 / fastest);
        }
        if out.len() > 2_000_000 {
            return Err(GeomError::Degenerate("the integration did not reach the end time"));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Stochastic simulation
// ---------------------------------------------------------------------------

/// The stochastic propensity of each reaction at a molecule count.
///
/// `a_j = k_j prod_i C(x_i, m_ij) m_ij!` -- a falling factorial, not a
/// power, because the molecules are discrete and distinguishable: two
/// molecules of the same species can meet in `x (x - 1) / 2` ways, not
/// `x^2 / 2`.
fn propensities(reactions: &[Reaction], k: &[f64], x: &[u64]) -> Vec<f64> {
    reactions
        .iter()
        .zip(k)
        .map(|(r, rate)| {
            let mut a = *rate;
            for (s, n) in &r.reactants {
                let count = x[*s];
                for step in 0..u64::from(*n) {
                    a *= (count.saturating_sub(step)) as f64;
                }
                // The 1/m! from the indistinguishable ways of choosing them.
                for step in 1..=u64::from(*n) {
                    a /= step as f64;
                }
            }
            a
        })
        .collect()
}

fn check_network(reactions: &[Reaction], k: &[f64], x0: &[u64]) -> Result<(), GeomError> {
    if reactions.is_empty() || reactions.len() != k.len() {
        return Err(GeomError::InvalidArgument("one rate constant per reaction"));
    }
    if k.iter().any(|v| *v < 0.0 || !v.is_finite()) {
        return Err(GeomError::InvalidArgument("every rate constant must be finite and positive"));
    }
    if x0.is_empty() {
        return Err(GeomError::InvalidArgument("the network has no species"));
    }
    if reactions
        .iter()
        .any(|r| r.reactants.iter().chain(&r.products).any(|(s, _)| *s >= x0.len()))
    {
        return Err(GeomError::InvalidArgument("a species index is out of range"));
    }
    Ok(())
}

/// Gillespie's direct method: an exact realisation of the chemical master
/// equation.
///
/// Exact in a strong sense -- the trajectory is drawn from the true
/// distribution of the jump process, with no time discretisation at all.
/// The waiting time to the next event is exponential with rate equal to the
/// total propensity, and which reaction fires is chosen in proportion to
/// its own. Returns `(time, counts)` after each event, including the
/// initial state.
///
/// # Errors
/// Returns an error for a malformed network or a non-positive end time.
pub fn gillespie_ssa(
    reactions: &[Reaction],
    k: &[f64],
    x0: &[u64],
    t_end: f64,
    max_events: usize,
    rng: &mut Rng,
) -> Result<Vec<(f64, Vec<u64>)>, GeomError> {
    check_network(reactions, k, x0)?;
    if !(t_end > 0.0) || max_events == 0 {
        return Err(GeomError::InvalidArgument("gillespie_ssa: bad parameters"));
    }
    let species = x0.len();
    let changes: Vec<Vec<i64>> = reactions.iter().map(|r| r.net_change(species)).collect();
    let mut x = x0.to_vec();
    let mut t = 0.0;
    let mut out = vec![(t, x.clone())];
    for _ in 0..max_events {
        let a = propensities(reactions, k, &x);
        let total: f64 = a.iter().sum();
        if !(total > 0.0) {
            // Nothing can happen: the system is at an absorbing state and
            // stays there for the rest of the run.
            break;
        }
        // The waiting time, by inversion. `1 - u` rather than `u` so a
        // uniform of exactly zero cannot produce an infinite wait.
        t -= (1.0 - rng.next_f64()).ln() / total;
        if t > t_end {
            break;
        }
        let mut pick = rng.next_f64() * total;
        let mut chosen = a.len() - 1;
        for (j, value) in a.iter().enumerate() {
            if pick < *value {
                chosen = j;
                break;
            }
            pick -= *value;
        }
        for i in 0..species {
            let delta = changes[chosen][i];
            x[i] = if delta >= 0 {
                x[i] + delta as u64
            } else {
                x[i].saturating_sub((-delta) as u64)
            };
        }
        out.push((t, x.clone()));
    }
    Ok(out)
}

/// A Poisson draw, exact at every rate.
///
/// Knuth's product method below thirty, where it is fastest, and Atkinson's
/// rejection method above it, where the product would underflow.
fn poisson(lambda: f64, rng: &mut Rng) -> u64 {
    if !(lambda > 0.0) {
        return 0;
    }
    if lambda < 30.0 {
        let limit = (-lambda).exp();
        let mut product = 1.0;
        let mut count = 0u64;
        loop {
            product *= rng.next_f64();
            if product <= limit {
                return count;
            }
            count += 1;
            if count > 1_000_000 {
                return count;
            }
        }
    }
    let c = 0.767 - 3.36 / lambda;
    let beta = std::f64::consts::PI / (3.0 * lambda).sqrt();
    let alpha = beta * lambda;
    let offset = c.ln() - lambda - beta.ln();
    for _ in 0..10_000 {
        let u = rng.next_f64().clamp(1e-300, 1.0 - 1e-16);
        let x = (alpha - ((1.0 - u) / u).ln()) / beta;
        if x + 0.5 < 0.0 {
            continue;
        }
        let n = (x + 0.5).floor();
        let v = rng.next_f64().max(1e-300);
        let y = alpha - beta * x;
        let denominator = 1.0 + y.exp();
        let lhs = y + (v / (denominator * denominator)).ln();
        let rhs = offset + n * lambda.ln() - crate::special::gamma::lgamma(n + 1.0);
        if lhs <= rhs {
            return n as u64;
        }
    }
    lambda.round() as u64
}

/// Explicit tau-leaping: many reaction events per step, each count drawn
/// from a Poisson distribution.
///
/// Trades exactness for speed. Over a leap of `tau` the propensities are
/// held fixed, so the number of firings of reaction `j` is Poisson with
/// mean `a_j tau` -- correct only while `tau` is short enough that the
/// propensities really do not change much, which is the whole art of the
/// method. Too long a leap drives species negative; this implementation
/// rejects a leap that would and retries it at half the length rather than
/// clamping, since clamping silently changes the reaction network.
///
/// # Errors
/// Returns an error for a malformed network, a non-positive end time or
/// leap.
pub fn tau_leaping(
    reactions: &[Reaction],
    k: &[f64],
    x0: &[u64],
    t_end: f64,
    tau: f64,
    rng: &mut Rng,
) -> Result<Vec<(f64, Vec<u64>)>, GeomError> {
    check_network(reactions, k, x0)?;
    if !(t_end > 0.0) || !(tau > 0.0) {
        return Err(GeomError::InvalidArgument("tau_leaping: bad parameters"));
    }
    let species = x0.len();
    let changes: Vec<Vec<i64>> = reactions.iter().map(|r| r.net_change(species)).collect();
    let mut x = x0.to_vec();
    let mut t = 0.0;
    let mut out = vec![(t, x.clone())];
    let mut steps = 0usize;
    while t < t_end && steps < 10_000_000 {
        steps += 1;
        let a = propensities(reactions, k, &x);
        if a.iter().sum::<f64>() <= 0.0 {
            break;
        }
        let mut leap = tau.min(t_end - t);
        let mut accepted = None;
        for _ in 0..40 {
            let firings: Vec<u64> = a.iter().map(|value| poisson(value * leap, rng)).collect();
            let mut candidate = vec![0i64; species];
            let mut negative = false;
            for i in 0..species {
                let mut total = x[i] as i64;
                for (j, count) in firings.iter().enumerate() {
                    total += changes[j][i] * *count as i64;
                }
                if total < 0 {
                    negative = true;
                    break;
                }
                candidate[i] = total;
            }
            if !negative {
                accepted = Some(candidate);
                break;
            }
            leap *= 0.5;
        }
        let Some(candidate) = accepted else {
            return Err(GeomError::Degenerate("no leap short enough kept the counts positive"));
        };
        for i in 0..species {
            x[i] = candidate[i] as u64;
        }
        t += leap;
        out.push((t, x.clone()));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Enzyme kinetics
// ---------------------------------------------------------------------------

/// The Michaelis-Menten rate `v = vmax s / (km + s)`.
#[must_use]
pub fn michaelis_menten(s: f64, vmax: f64, km: f64) -> f64 {
    if s <= 0.0 {
        return 0.0;
    }
    vmax * s / (km + s)
}

/// The Hill rate `v = vmax s^n / (k^n + s^n)`.
///
/// The exponent is a measure of cooperativity, not a molecularity: a Hill
/// coefficient of 2.8 for haemoglobin does not mean 2.8 oxygen molecules
/// bind at once, it means four sites bind with positive cooperativity and
/// the two-state fit lands there.
#[must_use]
pub fn hill_equation(s: f64, vmax: f64, k: f64, n: f64) -> f64 {
    if s <= 0.0 || k <= 0.0 {
        return 0.0;
    }
    let sn = s.powf(n);
    vmax * sn / (k.powf(n) + sn)
}

/// Fits `vmax` and `km` to saturation data by least squares on the
/// *residuals of the rate itself*, by Gauss-Newton.
///
/// Deliberately not the Lineweaver-Burk fit. Inverting the data transforms
/// the error along with it, so the points at the lowest substrate -- where
/// the relative error is largest -- become the ones with the largest
/// leverage, and the fitted `vmax` is biased. The double-reciprocal plot
/// remains useful for *seeing* the mechanism, which is what
/// [`lineweaver_burk`] is for; it is not the way to get the numbers.
///
/// # Errors
/// Returns an error for fewer than three points, mismatched lengths,
/// negative concentrations or rates, or a fit that does not converge.
pub fn mm_fit(s: &[f64], v: &[f64]) -> Result<(f64, f64), GeomError> {
    if s.len() < 3 || s.len() != v.len() {
        return Err(GeomError::InvalidArgument("mm_fit needs three matched points"));
    }
    if s.iter().any(|x| *x < 0.0) || v.iter().any(|y| *y < 0.0) {
        return Err(GeomError::InvalidArgument("concentrations and rates must be non-negative"));
    }
    let peak = v.iter().copied().fold(0.0, f64::max);
    if !(peak > 0.0) {
        return Err(GeomError::Degenerate("every measured rate is zero"));
    }
    // Started from the double-reciprocal estimate, which is a poor fit and a
    // perfectly good starting point.
    let mut vmax = peak * 1.2;
    let mut km = s.iter().copied().fold(0.0, f64::max) * 0.5 + 1e-9;
    for _ in 0..200 {
        // Residual r_i = vmax s / (km + s) - v_i, with analytic derivatives.
        let (mut jtj00, mut jtj01, mut jtj11) = (0.0, 0.0, 0.0);
        let (mut jtr0, mut jtr1) = (0.0, 0.0);
        for (si, vi) in s.iter().zip(v) {
            let denominator = km + si;
            if denominator.abs() < 1e-300 {
                return Err(GeomError::Degenerate("the fit reached a singular denominator"));
            }
            let model = vmax * si / denominator;
            let d_vmax = si / denominator;
            let d_km = -vmax * si / (denominator * denominator);
            let residual = model - vi;
            jtj00 += d_vmax * d_vmax;
            jtj01 += d_vmax * d_km;
            jtj11 += d_km * d_km;
            jtr0 += d_vmax * residual;
            jtr1 += d_km * residual;
        }
        // A Levenberg damping term, so a flat direction cannot send the
        // step to infinity.
        let lambda = 1e-9 * (jtj00 + jtj11).max(1e-12);
        let (a, b, d) = (jtj00 + lambda, jtj01, jtj11 + lambda);
        let determinant = a * d - b * b;
        if determinant.abs() < 1e-300 {
            break;
        }
        let step_vmax = -(d * jtr0 - b * jtr1) / determinant;
        let step_km = -(a * jtr1 - b * jtr0) / determinant;
        // Both parameters are positive by construction, so a step that
        // would cross zero is halved rather than taken.
        let mut scale = 1.0f64;
        while (vmax + scale * step_vmax <= 0.0 || km + scale * step_km <= 0.0) && scale > 1e-12 {
            scale *= 0.5;
        }
        vmax += scale * step_vmax;
        km += scale * step_km;
        if (scale * step_vmax).abs() < 1e-12 * vmax && (scale * step_km).abs() < 1e-12 * km {
            return Ok((vmax, km));
        }
    }
    Ok((vmax, km))
}

/// The double-reciprocal transform: `(1/s, 1/v)` for each point, plus the
/// straight line through them as `(slope, intercept)`.
///
/// The line has slope `km / vmax` and intercept `1 / vmax`. Useful for
/// reading a mechanism off a plot -- competitive, uncompetitive and
/// non-competitive inhibition give visibly different families of lines --
/// and a poor way to extract the constants; see [`mm_fit`].
///
/// # Errors
/// Returns an error for fewer than two points, mismatched lengths, or a
/// non-positive concentration or rate, which the transform cannot represent.
pub fn lineweaver_burk(s: &[f64], v: &[f64]) -> Result<(Vec<(f64, f64)>, f64, f64), GeomError> {
    if s.len() < 2 || s.len() != v.len() {
        return Err(GeomError::InvalidArgument("lineweaver_burk needs two matched points"));
    }
    if s.iter().any(|x| !(*x > 0.0)) || v.iter().any(|y| !(*y > 0.0)) {
        return Err(GeomError::InvalidArgument("the transform needs positive values"));
    }
    let points: Vec<(f64, f64)> = s.iter().zip(v).map(|(x, y)| (1.0 / x, 1.0 / y)).collect();
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denominator = n * sxx - sx * sx;
    if denominator.abs() < 1e-300 {
        return Err(GeomError::Degenerate("every point has the same concentration"));
    }
    let slope = (n * sxy - sx * sy) / denominator;
    let intercept = (sy - slope * sx) / n;
    Ok((points, slope, intercept))
}

/// Fits the Hill parameters `(vmax, k, n)` by Gauss-Newton.
///
/// # Errors
/// Returns an error for fewer than four points, mismatched lengths, or
/// non-positive data.
pub fn hill_fit(s: &[f64], v: &[f64]) -> Result<(f64, f64, f64), GeomError> {
    if s.len() < 4 || s.len() != v.len() {
        return Err(GeomError::InvalidArgument("hill_fit needs four matched points"));
    }
    if s.iter().any(|x| !(*x > 0.0)) || v.iter().any(|y| *y < 0.0) {
        return Err(GeomError::InvalidArgument("hill_fit: bad data"));
    }
    let peak = v.iter().copied().fold(0.0, f64::max);
    if !(peak > 0.0) {
        return Err(GeomError::Degenerate("every measured rate is zero"));
    }
    let mut p = [peak * 1.1, s.iter().copied().fold(0.0, f64::max) * 0.5 + 1e-9, 1.0];
    let model = |p: &[f64; 3], x: f64| hill_equation(x, p[0], p[1], p[2]);
    for _ in 0..400 {
        let mut jtj = [[0.0f64; 3]; 3];
        let mut jtr = [0.0f64; 3];
        for (si, vi) in s.iter().zip(v) {
            let residual = model(&p, *si) - vi;
            // Numerical derivatives: the analytic ones in the exponent are
            // long and this fit is small.
            let mut grad = [0.0f64; 3];
            for a in 0..3 {
                let h = 1e-6 * p[a].abs().max(1e-6);
                let mut up = p;
                up[a] += h;
                let mut down = p;
                down[a] -= h;
                grad[a] = (model(&up, *si) - model(&down, *si)) / (2.0 * h);
            }
            for a in 0..3 {
                jtr[a] += grad[a] * residual;
                for b in 0..3 {
                    jtj[a][b] += grad[a] * grad[b];
                }
            }
        }
        let damping = 1e-8 * (jtj[0][0] + jtj[1][1] + jtj[2][2]).max(1e-12);
        let mut m = Matrix::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                m.set(a, b, jtj[a][b] + if a == b { damping } else { 0.0 });
            }
        }
        let Ok(step) = crate::linalg::lu::solve(&m, &[-jtr[0], -jtr[1], -jtr[2]]) else {
            break;
        };
        let mut scale = 1.0f64;
        while (0..3).any(|a| p[a] + scale * step[a] <= 0.0) && scale > 1e-12 {
            scale *= 0.5;
        }
        let mut moved = 0.0f64;
        for a in 0..3 {
            p[a] += scale * step[a];
            moved = moved.max((scale * step[a]).abs() / p[a].abs().max(1e-12));
        }
        if moved < 1e-12 {
            break;
        }
    }
    Ok((p[0], p[1], p[2]))
}

/// Which way an inhibitor acts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inhibition {
    /// Binds the free enzyme, so substrate can outcompete it: `km` rises,
    /// `vmax` is untouched.
    Competitive,
    /// Binds the enzyme-substrate complex only: `km` and `vmax` fall
    /// together, so their ratio is untouched.
    Uncompetitive,
    /// Binds either with equal affinity: `vmax` falls, `km` is untouched.
    NonCompetitive,
}

/// The inhibited Michaelis-Menten rate.
///
/// The three mechanisms are distinguished by *which* constant moves, not by
/// how much the rate falls -- which is why a single rate measurement can
/// never identify the mechanism and a substrate series can.
///
/// # Errors
/// Returns an error for a non-positive `km` or inhibition constant, or a
/// negative concentration.
pub fn enzyme_inhibition(
    s: f64,
    i: f64,
    vmax: f64,
    km: f64,
    ki: f64,
    kind: Inhibition,
) -> Result<f64, GeomError> {
    if !(km > 0.0) || !(ki > 0.0) || s < 0.0 || i < 0.0 {
        return Err(GeomError::InvalidArgument("enzyme_inhibition: bad parameters"));
    }
    if s == 0.0 {
        return Ok(0.0);
    }
    let alpha = 1.0 + i / ki;
    Ok(match kind {
        Inhibition::Competitive => vmax * s / (km * alpha + s),
        Inhibition::Uncompetitive => (vmax / alpha) * s / (km / alpha + s),
        Inhibition::NonCompetitive => (vmax / alpha) * s / (km + s),
    })
}

/// How far a mechanism is from its steady-state approximation, as the
/// largest relative difference in the intermediate's concentration.
///
/// The approximation holds when the intermediate is consumed as fast as it
/// is made, which for Michaelis-Menten means the enzyme is scarce beside the
/// substrate. Returns the discrepancy so the caller can see *whether* it
/// holds rather than assuming it.
///
/// # Errors
/// Returns an error for non-positive rate constants or concentrations.
pub fn steady_state_approx_check(
    e0: f64,
    s0: f64,
    k1: f64,
    k_minus1: f64,
    k2: f64,
    t_end: f64,
) -> Result<f64, GeomError> {
    if !(e0 > 0.0) || !(s0 > 0.0) || !(k1 > 0.0) || k_minus1 < 0.0 || !(k2 > 0.0) || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("steady_state_approx_check: bad parameters"));
    }
    // Species: 0 = S, 1 = E, 2 = ES, 3 = P.
    let reactions = [
        Reaction::new(&[(0, 1), (1, 1)], &[(2, 1)]),
        Reaction::new(&[(2, 1)], &[(0, 1), (1, 1)]),
        Reaction::new(&[(2, 1)], &[(1, 1), (3, 1)]),
    ];
    let k = [k1, k_minus1, k2];
    let stoich = stoichiometry_matrix(&reactions, 4)?;
    let rates = |c: &[f64]| mass_action_rates(&reactions, &k, c).unwrap_or_else(|_| vec![0.0; 3]);
    let trace = rate_equations(&stoich, &rates, &[s0, e0, 0.0, 0.0], t_end, 1e-8)?;
    let km = (k_minus1 + k2) / k1;
    // The complex fills on a timescale 1 / (k1 s0 + k_minus1 + k2), and
    // during that transient the approximation is not claimed to hold at all
    // -- it says the complex is at its steady value, and at t = 0 it is
    // zero. Skipping a *fraction of the steps* would not do: the adaptive
    // integrator spends most of its steps inside that transient, so a tenth
    // of the way through the record is still well inside it.
    let induction = 1.0 / (k1 * s0 + k_minus1 + k2);
    let start = 50.0 * induction;
    if start >= t_end {
        return Err(GeomError::InvalidArgument(
            "the run ends before the complex has had time to fill",
        ));
    }
    let mut worst: f64 = 0.0;
    for (_, c) in trace.iter().filter(|(t, _)| *t > start) {
        let (s, es) = (c[0], c[2]);
        // The steady-state value of the complex, from d[ES]/dt = 0 with the
        // enzyme conserved.
        let total_enzyme = c[1] + c[2];
        let predicted = total_enzyme * s / (km + s);
        let scale = predicted.abs().max(es.abs()).max(1e-12 * e0);
        worst = worst.max((es - predicted).abs() / scale);
    }
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Equilibrium
// ---------------------------------------------------------------------------

/// The equilibrium composition of a set of reactions with known constants,
/// found by minimising the total residual of the mass-action and
/// conservation conditions.
///
/// Each reaction contributes `prod c^nu = k_eq` and each conserved element
/// contributes a total. Solved by Newton on the logarithms of the
/// concentrations, which keeps every one positive without a constraint --
/// a composition can approach zero but never reach or cross it, which is
/// what the physical problem requires and what an unconstrained solve on the
/// concentrations themselves does not respect.
///
/// `totals` is one row per conserved quantity, giving each species' content
/// and the total amount.
///
/// # Errors
/// Returns an error for mismatched shapes, a non-positive constant or total,
/// or a system that does not converge.
pub fn equilibrium_composition(
    stoich: &Matrix,
    k_eq: &[f64],
    totals: &[(Vec<f64>, f64)],
) -> Result<Vec<f64>, GeomError> {
    let species = stoich.rows;
    let reactions = stoich.cols;
    if k_eq.len() != reactions {
        return Err(GeomError::InvalidArgument("one constant per reaction"));
    }
    if k_eq.iter().any(|k| !(*k > 0.0)) {
        return Err(GeomError::InvalidArgument("every constant must be positive"));
    }
    if reactions + totals.len() != species {
        return Err(GeomError::InvalidArgument(
            "the reactions and conservation laws must together determine the composition",
        ));
    }
    if totals.iter().any(|(row, amount)| row.len() != species || !(*amount > 0.0)) {
        return Err(GeomError::InvalidArgument("a conservation row is malformed"));
    }
    // Started from an even split of each conserved total.
    let mut log_c = vec![0.0f64; species];
    for i in 0..species {
        let mut guess = 1e-6f64;
        for (row, amount) in totals {
            if row[i] > 0.0 {
                guess = guess.max(amount / (species as f64 * row[i]));
            }
        }
        log_c[i] = guess.ln();
    }
    for _ in 0..500 {
        let c: Vec<f64> = log_c.iter().map(|l| l.exp()).collect();
        let mut residual = vec![0.0f64; species];
        let mut jacobian = Matrix::zeros(species, species);
        // Mass action, in logarithms: sum nu_i ln c_i = ln k.
        for j in 0..reactions {
            let mut sum = -k_eq[j].ln();
            for i in 0..species {
                sum += stoich.get(i, j) * log_c[i];
                jacobian.set(j, i, stoich.get(i, j));
            }
            residual[j] = sum;
        }
        // Conservation, in the concentrations themselves.
        for (r, (row, amount)) in totals.iter().enumerate() {
            let index = reactions + r;
            let mut sum = -amount;
            for i in 0..species {
                sum += row[i] * c[i];
                // d/d(ln c_i) of row_i c_i is row_i c_i.
                jacobian.set(index, i, row[i] * c[i]);
            }
            residual[index] = sum;
        }
        let worst = residual.iter().fold(0.0f64, |a, r| a.max(r.abs()));
        if worst < 1e-13 {
            return Ok(c);
        }
        let negated: Vec<f64> = residual.iter().map(|r| -r).collect();
        let Ok(step) = crate::linalg::lu::solve(&jacobian, &negated) else {
            return Err(GeomError::Degenerate("the equilibrium system is singular"));
        };
        // A trust region in the logarithms: a full Newton step early on can
        // jump twenty orders of magnitude and land outside the range where
        // the exponentials are finite.
        let longest = step.iter().fold(0.0f64, |a, s| a.max(s.abs()));
        let scale = if longest > 2.0 { 2.0 / longest } else { 1.0 };
        for i in 0..species {
            log_c[i] += scale * step[i];
        }
    }
    Err(GeomError::Degenerate("the equilibrium composition did not converge"))
}

// ---------------------------------------------------------------------------
// Oscillating and autocatalytic mechanisms
// ---------------------------------------------------------------------------

/// The Brusselator, integrated in time.
///
/// `A -> X`, `2X + Y -> 3X`, `B + X -> Y + D`, `X -> E`, with `A` and `B`
/// held fixed. The steady state `(a, b/a)` loses stability in a Hopf
/// bifurcation exactly at `b = 1 + a^2`, and above it the system settles
/// onto a limit cycle whose amplitude does not depend on where it started.
/// That sharp threshold is what makes it the standard test of an oscillating
/// mechanism: the transition is a property of the equations, not of the
/// integrator.
///
/// # Errors
/// Returns an error for non-positive parameters or a bad initial state.
pub fn oscillating_brusselator(
    a: f64,
    b: f64,
    c0: (f64, f64),
    t_end: f64,
) -> Result<Vec<(f64, Vec<f64>)>, GeomError> {
    if !(a > 0.0) || !(b > 0.0) || c0.0 < 0.0 || c0.1 < 0.0 || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("oscillating_brusselator: bad parameters"));
    }
    // Species: 0 = X, 1 = Y. Written directly rather than through the
    // network machinery, since A and B are held fixed and so are not
    // species of the dynamical system.
    let mut stoich = Matrix::zeros(2, 4);
    // A -> X
    stoich.set(0, 0, 1.0);
    // 2X + Y -> 3X
    stoich.set(0, 1, 1.0);
    stoich.set(1, 1, -1.0);
    // B + X -> Y + D
    stoich.set(0, 2, -1.0);
    stoich.set(1, 2, 1.0);
    // X -> E
    stoich.set(0, 3, -1.0);
    let rates = move |c: &[f64]| {
        let (x, y) = (c[0].max(0.0), c[1].max(0.0));
        vec![a, x * x * y, b * x, x]
    };
    rate_equations(&stoich, &rates, &[c0.0, c0.1], t_end, 1e-8)
}

/// Whether the Brusselator oscillates at these parameters: `b > 1 + a^2`.
#[must_use]
pub fn brusselator_oscillates(a: f64, b: f64) -> bool {
    b > 1.0 + a * a
}

/// The Oregonator, the Field-Noyes reduction of the Belousov-Zhabotinsky
/// reaction, in its scaled form.
///
/// Genuinely stiff: `epsilon` and `delta` are of order `10^-2` and `10^-4`,
/// so the three variables move on timescales four orders of magnitude
/// apart, and an explicit integrator would be pinned to the fastest one for
/// the whole run. This is the case the implicit solver in
/// [`rate_equations`] exists for.
///
/// # Errors
/// Returns an error for non-positive parameters or a bad initial state.
pub fn oregonator(
    epsilon: f64,
    delta: f64,
    q: f64,
    f: f64,
    c0: (f64, f64, f64),
    t_end: f64,
) -> Result<Vec<(f64, Vec<f64>)>, GeomError> {
    if !(epsilon > 0.0) || !(delta > 0.0) || !(q > 0.0) || !(f > 0.0) || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("oregonator: bad parameters"));
    }
    if c0.0 < 0.0 || c0.1 < 0.0 || c0.2 < 0.0 {
        return Err(GeomError::InvalidArgument("the initial state must be non-negative"));
    }
    // The scaled equations are not mass action, so the identity
    // stoichiometry is used and the whole derivative is supplied as a rate.
    let stoich = Matrix::identity(3);
    let rates = move |c: &[f64]| {
        let (x, y, z) = (c[0].max(0.0), c[1].max(0.0), c[2].max(0.0));
        vec![
            (q * y - x * y + x * (1.0 - x)) / epsilon,
            (-q * y - x * y + f * z) / delta,
            x - z,
        ]
    };
    rate_equations(&stoich, &rates, &[c0.0, c0.1, c0.2], t_end, 1e-7)
}

/// The chemical Lotka-Volterra mechanism: `A + X -> 2X`, `X + Y -> 2Y`,
/// `Y -> B`, with `A` held fixed.
///
/// Returns the trajectory together with the conserved quantity
/// `V = k2 x + k2 y - k3 ln x - k1 a ln y`, which is constant along every
/// orbit. That constant is the reason the orbits are closed curves rather
/// than a limit cycle: the system is conservative, and unlike the
/// Brusselator its amplitude *does* depend on where it started. Reporting
/// it lets a caller see the integrator's drift directly.
///
/// # Errors
/// Returns an error for non-positive parameters or a non-positive initial
/// state, for which the conserved quantity is undefined.
pub fn lotka_volterra_chemical(
    a: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    c0: (f64, f64),
    t_end: f64,
) -> Result<(Vec<(f64, Vec<f64>)>, Vec<f64>), GeomError> {
    if !(a > 0.0) || !(k1 > 0.0) || !(k2 > 0.0) || !(k3 > 0.0) || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("lotka_volterra_chemical: bad parameters"));
    }
    if !(c0.0 > 0.0) || !(c0.1 > 0.0) {
        return Err(GeomError::InvalidArgument("both populations must start positive"));
    }
    let mut stoich = Matrix::zeros(2, 3);
    stoich.set(0, 0, 1.0);
    stoich.set(0, 1, -1.0);
    stoich.set(1, 1, 1.0);
    stoich.set(1, 2, -1.0);
    let rates = move |c: &[f64]| {
        let (x, y) = (c[0].max(0.0), c[1].max(0.0));
        vec![k1 * a * x, k2 * x * y, k3 * y]
    };
    let trace = rate_equations(&stoich, &rates, &[c0.0, c0.1], t_end, 1e-8)?;
    let invariant = trace
        .iter()
        .map(|(_, c)| {
            let (x, y) = (c[0].max(1e-300), c[1].max(1e-300));
            k2 * x + k2 * y - k3 * x.ln() - k1 * a * y.ln()
        })
        .collect();
    Ok((trace, invariant))
}

/// The ignition time of an autocatalytic reaction `A + B -> 2B`, defined as
/// the moment the product passes half its final amount.
///
/// The closed form is the logistic inflection: with `a0 + b0` conserved,
/// `t = ln(a0 / b0) / (k (a0 + b0))`. The induction period is set by how
/// *little* product there is at the start, which is why an autocatalytic
/// reaction can sit apparently inert for a long time and then go over in a
/// moment.
///
/// # Errors
/// Returns an error for a non-positive rate constant or a non-positive
/// initial amount of either species.
pub fn autocatalysis_ignition(a0: f64, b0: f64, k: f64) -> Result<f64, GeomError> {
    if !(k > 0.0) || !(a0 > 0.0) || !(b0 > 0.0) {
        return Err(GeomError::InvalidArgument("autocatalysis_ignition: bad parameters"));
    }
    if b0 >= a0 {
        // Already past half conversion at t = 0.
        return Ok(0.0);
    }
    Ok((a0 / b0).ln() / (k * (a0 + b0)))
}

/// Whether a branching chain reaction runs away, and by how much: the
/// branching ratio `k_branch / k_term`.
///
/// Above one the chain carriers multiply and the reaction accelerates
/// without bound; below one it dies out. The threshold is exactly one and
/// nothing continuous separates the two behaviours, which is why an
/// explosion limit is a sharp line in pressure and temperature rather than
/// a gradual onset.
///
/// # Errors
/// Returns an error for a non-positive termination rate.
pub fn chain_reaction_criticality(k_branch: f64, k_term: f64) -> Result<f64, GeomError> {
    if !(k_term > 0.0) || k_branch < 0.0 {
        return Err(GeomError::InvalidArgument("chain_reaction_criticality: bad parameters"));
    }
    Ok(k_branch / k_term)
}

// ---------------------------------------------------------------------------
// Rate theory
// ---------------------------------------------------------------------------

/// The Eyring rate `(k_B T / h) exp(dS/R) exp(-dH/RT)`.
///
/// Differs from Arrhenius in what the prefactor means: here it is
/// `k_B T / h`, a universal frequency of about `6 x 10^12` per second at
/// room temperature, and all the chemistry sits in the entropy of
/// activation. The two forms fit the same data equally well and disagree
/// about why.
///
/// # Errors
/// Returns an error for a non-positive temperature.
pub fn eyring(delta_h: f64, delta_s: f64, t: f64) -> Result<f64, GeomError> {
    if !(t > 0.0) {
        return Err(GeomError::InvalidArgument("the temperature must be positive"));
    }
    const PLANCK: f64 = 6.626_070_15e-34;
    Ok((constants::K_B * t / PLANCK) * (delta_s / constants::R).exp()
        * (-delta_h / (constants::R * t)).exp())
}

/// Transition-state theory with a transmission coefficient.
///
/// `k = kappa (k_B T / h) exp(-dG/RT)`. The coefficient is the fraction of
/// trajectories that cross the barrier and *stay* crossed; transition-state
/// theory assumes it is one, which makes the theory an upper bound on the
/// true rate rather than an estimate of it.
///
/// # Errors
/// Returns an error for a non-positive temperature or a coefficient outside
/// zero to one.
pub fn transition_state_theory_rate(
    delta_g: f64,
    t: f64,
    transmission: f64,
) -> Result<f64, GeomError> {
    if !(t > 0.0) || !(0.0..=1.0).contains(&transmission) {
        return Err(GeomError::InvalidArgument("transition_state_theory_rate: bad parameters"));
    }
    const PLANCK: f64 = 6.626_070_15e-34;
    Ok(transmission * (constants::K_B * t / PLANCK) * (-delta_g / (constants::R * t)).exp())
}

/// The Kramers rate in the moderate-to-high friction regime, relative to the
/// transition-state result.
///
/// `k / k_TST = sqrt(1 + (gamma / 2 omega_b)^2) - gamma / (2 omega_b)`,
/// which is at most one and falls toward `omega_b / gamma` as the friction
/// grows: a solvent that couples strongly to the reaction coordinate makes
/// recrossing likely, and every recrossing is a barrier passage that did not
/// produce a product. This is the transmission coefficient that
/// [`transition_state_theory_rate`] takes on faith.
///
/// Barrier frequency and friction are in the same units; the ratio is what
/// matters.
///
/// # Errors
/// Returns an error for a non-positive barrier frequency or a negative
/// friction.
pub fn kramers_rate_check(gamma: f64, barrier_frequency: f64) -> Result<f64, GeomError> {
    if !(barrier_frequency > 0.0) || gamma < 0.0 {
        return Err(GeomError::InvalidArgument("kramers_rate_check: bad parameters"));
    }
    let ratio = gamma / (2.0 * barrier_frequency);
    Ok((1.0 + ratio * ratio).sqrt() - ratio)
}

/// The semiclassical kinetic isotope effect from the change in zero-point
/// energy alone.
///
/// `k_light / k_heavy = exp(h (nu_light - nu_heavy) / (2 k_B T))`. The
/// hydrogen-deuterium maximum near seven at room temperature comes out of
/// this and nothing else; a measured ratio well above it is evidence of
/// tunnelling, which this estimate deliberately omits so that the excess is
/// visible rather than absorbed into a fitted parameter.
///
/// Frequencies are in reciprocal centimetres.
///
/// # Errors
/// Returns an error for a non-positive temperature or frequency.
pub fn kinetic_isotope_effect_estimate(
    nu_light: f64,
    nu_heavy: f64,
    t: f64,
) -> Result<f64, GeomError> {
    if !(t > 0.0) || !(nu_light > 0.0) || !(nu_heavy > 0.0) {
        return Err(GeomError::InvalidArgument("kinetic_isotope_effect_estimate: bad parameters"));
    }
    const PLANCK: f64 = 6.626_070_15e-34;
    const LIGHT_SPEED_CM: f64 = 2.997_924_58e10;
    let energy = 0.5 * PLANCK * LIGHT_SPEED_CM * (nu_light - nu_heavy);
    Ok((energy / (constants::K_B * t)).exp())
}

/// The relaxation time of a reaction perturbed from equilibrium by a
/// temperature jump.
///
/// For `A <-> B` the relaxation is a single exponential with rate
/// `k_forward + k_reverse` -- the *sum*, not either one. That is what makes
/// the technique work: a single measured relaxation gives the sum, the
/// equilibrium constant gives the ratio, and together they give both rate
/// constants, which no steady-state measurement can separate.
///
/// # Errors
/// Returns an error if both rate constants are zero or either is negative.
pub fn temperature_jump_relaxation(k_forward: f64, k_reverse: f64) -> Result<f64, GeomError> {
    if k_forward < 0.0 || k_reverse < 0.0 {
        return Err(GeomError::InvalidArgument("the rate constants must be non-negative"));
    }
    let total = k_forward + k_reverse;
    if !(total > 0.0) {
        return Err(GeomError::Degenerate("nothing relaxes: both rates are zero"));
    }
    Ok(1.0 / total)
}

/// The classical nucleation rate `J = A exp(-dG* / k_B T)`.
///
/// The exponent is enormous and its argument is a cube over a square, so
/// the rate spans dozens of orders of magnitude over a small change in
/// supersaturation. That extreme sensitivity is the physics, not a defect of
/// the model: it is why nucleation appears to have a threshold.
///
/// # Errors
/// Returns an error for a non-positive temperature or prefactor.
pub fn nucleation_rate_cnt(barrier: f64, prefactor: f64, t: f64) -> Result<f64, GeomError> {
    if !(t > 0.0) || !(prefactor > 0.0) || barrier < 0.0 {
        return Err(GeomError::InvalidArgument("nucleation_rate_cnt: bad parameters"));
    }
    Ok(prefactor * (-barrier / (constants::K_B * t)).exp())
}

/// The classical nucleation barrier for a spherical nucleus:
/// `16 pi sigma^3 / (3 (n dmu)^2)`.
///
/// # Errors
/// Returns an error for a non-positive surface tension, density or driving
/// force.
pub fn nucleation_barrier(
    surface_tension: f64,
    number_density: f64,
    driving_force: f64,
) -> Result<f64, GeomError> {
    if !(surface_tension > 0.0) || !(number_density > 0.0) || !(driving_force > 0.0) {
        return Err(GeomError::InvalidArgument("nucleation_barrier: bad parameters"));
    }
    let bulk = number_density * driving_force;
    Ok(16.0 * std::f64::consts::PI * surface_tension.powi(3) / (3.0 * bulk * bulk))
}

/// The Johnson-Mehl-Avrami-Kolmogorov transformed fraction
/// `1 - exp(-(k t)^n)`.
///
/// The exponent carries the mechanism: roughly 4 for three-dimensional
/// growth from a constant nucleation rate, 3 when all sites nucleate at
/// once, and lower for growth confined to a plane or a line. The point of
/// fitting it is to read the dimensionality off the kinetics.
#[must_use]
pub fn jmak_avrami(t: f64, k: f64, n: f64) -> f64 {
    if t <= 0.0 || k <= 0.0 || n <= 0.0 {
        return 0.0;
    }
    1.0 - (-(k * t).powf(n)).exp()
}

/// Fits `(k, n)` to transformed-fraction data.
///
/// The double logarithm `ln(-ln(1 - x)) = n ln t + n ln k` makes the fit
/// linear and exact, which is the one case where a transform of the data is
/// the right thing to do: the relation is exactly linear in the transformed
/// variables, so no error is being reshaped, only re-expressed.
///
/// # Errors
/// Returns an error for fewer than two usable points -- a fraction of zero
/// or one carries no information, since the transform sends it to infinity.
pub fn avrami_fit(times: &[f64], fraction: &[f64]) -> Result<(f64, f64), GeomError> {
    if times.len() != fraction.len() {
        return Err(GeomError::InvalidArgument("avrami_fit: mismatched input"));
    }
    let points: Vec<(f64, f64)> = times
        .iter()
        .zip(fraction)
        .filter(|(t, x)| **t > 0.0 && **x > 1e-12 && **x < 1.0 - 1e-12)
        .map(|(t, x)| (t.ln(), (-(1.0 - x).ln()).ln()))
        .collect();
    if points.len() < 2 {
        return Err(GeomError::InvalidArgument("avrami_fit needs two points strictly inside"));
    }
    let count = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denominator = count * sxx - sx * sx;
    if denominator.abs() < 1e-300 {
        return Err(GeomError::Degenerate("every point is at the same time"));
    }
    let n = (count * sxy - sx * sy) / denominator;
    if !(n > 0.0) {
        return Err(GeomError::Degenerate("the fitted exponent is not positive"));
    }
    let intercept = (sy - n * sx) / count;
    Ok(((intercept / n).exp(), n))
}

/// The quantum yield: molecules transformed per photon absorbed.
///
/// A yield above one is not an error -- a chain reaction initiated by one
/// photon can transform thousands of molecules -- so no upper bound is
/// imposed.
///
/// # Errors
/// Returns an error for a non-positive photon count or a negative product
/// count.
pub fn photochemistry_quantum_yield(
    molecules: f64,
    photons_absorbed: f64,
) -> Result<f64, GeomError> {
    if !(photons_absorbed > 0.0) || molecules < 0.0 {
        return Err(GeomError::InvalidArgument("photochemistry_quantum_yield: bad input"));
    }
    Ok(molecules / photons_absorbed)
}

// ---------------------------------------------------------------------------
// Acid-base and electrochemistry
// ---------------------------------------------------------------------------

/// The pH of a solution of one or more acids, by solving the full charge
/// balance rather than any approximation.
///
/// Each acid is `(pKa, total concentration)`; `base_conc` is added strong
/// base. The equation solved is
/// `[H+] + [base] = K_w/[H+] + sum_a C_a K_a / (K_a + [H+])`,
/// which includes the water autoprotolysis and the depletion of the acid as
/// it dissociates. Neither can be dropped in general: the usual
/// `sqrt(K_a C)` shortcut assumes both, and it fails for a dilute acid
/// (where water dominates) and for a strong one (where the acid is nearly
/// all dissociated and the depletion is the whole story). Solved by
/// bisection on `pH`, which cannot diverge because the balance is monotone
/// in `[H+]`.
///
/// # Errors
/// Returns an error for a negative concentration or an empty system with no
/// base.
pub fn ph_from_equilibria(acids: &[(f64, f64)], base_conc: f64) -> Result<f64, GeomError> {
    if acids.iter().any(|(_, c)| *c < 0.0) || base_conc < 0.0 {
        return Err(GeomError::InvalidArgument("concentrations must be non-negative"));
    }
    const KW: f64 = 1e-14;
    // Excess of negative charge at a given [H+]; monotone decreasing in
    // [H+], so bisection is unconditionally safe.
    let balance = |h: f64| -> f64 {
        let mut total = KW / h - h - base_conc;
        for (pka, c) in acids {
            let ka = 10f64.powf(-pka);
            total += c * ka / (ka + h);
        }
        total
    };
    let (mut lo, mut hi) = (-1.0f64, 15.0f64);
    // lo is the most acidic pH considered, so the balance there is negative.
    if balance(10f64.powf(-lo)) > 0.0 || balance(10f64.powf(-hi)) < 0.0 {
        return Err(GeomError::Degenerate("the pH lies outside -1 to 15"));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if balance(10f64.powf(-mid)) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// A titration curve: pH against the volume of strong base added.
///
/// Returns `(volume added, pH)` at each of `points` steps up to
/// `volume_max`. Dilution is accounted for -- both the acid and the base
/// are diluted by the growing total volume -- which is what puts the
/// equivalence point of a weak acid above pH 7 rather than at it.
///
/// # Errors
/// Returns an error for a non-positive volume, concentration or point
/// count.
pub fn titration_curve(
    acid_pka: f64,
    acid_conc: f64,
    acid_volume: f64,
    base_conc: f64,
    volume_max: f64,
    points: usize,
) -> Result<Vec<(f64, f64)>, GeomError> {
    if !(acid_conc > 0.0) || !(acid_volume > 0.0) || !(base_conc > 0.0) {
        return Err(GeomError::InvalidArgument("titration_curve: bad concentrations"));
    }
    if !(volume_max > 0.0) || points < 2 {
        return Err(GeomError::InvalidArgument("titration_curve: bad sweep"));
    }
    (0..points)
        .map(|k| {
            let added = volume_max * k as f64 / (points - 1) as f64;
            let total = acid_volume + added;
            let diluted_acid = acid_conc * acid_volume / total;
            let diluted_base = base_conc * added / total;
            let ph = ph_from_equilibria(&[(acid_pka, diluted_acid)], diluted_base)?;
            Ok((added, ph))
        })
        .collect()
}

/// Henderson-Hasselbalch: `pH = pKa + log10(base / acid)`.
///
/// An approximation, and one whose failure is predictable: it assumes the
/// dissociation does not appreciably change either concentration, so it is
/// accurate within about a unit of the pKa and wrong outside that. Compare
/// against [`ph_from_equilibria`], which makes no such assumption.
///
/// # Errors
/// Returns an error for a non-positive ratio.
pub fn buffer_henderson_hasselbalch(pka: f64, ratio: f64) -> Result<f64, GeomError> {
    if !(ratio > 0.0) {
        return Err(GeomError::InvalidArgument("the ratio must be positive"));
    }
    Ok(pka + ratio.log10())
}

/// The Debye-Huckel activity coefficient of an ion.
///
/// The extended law `log10 gamma = -A z^2 sqrt(I) / (1 + sqrt I)`, with
/// `A = 0.509` for water at 25 degrees. The limiting law without the
/// denominator is only good below about `I = 0.01`; the extended form holds
/// to roughly `I = 0.1`, and above that no simple expression does.
///
/// # Errors
/// Returns an error for a negative ionic strength.
pub fn debye_huckel_activity(z: f64, ionic_strength: f64) -> Result<f64, GeomError> {
    if ionic_strength < 0.0 {
        return Err(GeomError::InvalidArgument("the ionic strength must be non-negative"));
    }
    const A: f64 = 0.509;
    let root = ionic_strength.sqrt();
    Ok(10f64.powf(-A * z * z * root / (1.0 + root)))
}

/// The Nernst potential from a concentration ratio.
///
/// A thin wrapper on [`crate::chemistry::nernst_potential`] in the form the
/// kinetics literature uses. At 25 degrees and one electron the slope is
/// 59.16 mV per decade, which is the number every ion-selective electrode
/// is calibrated against.
///
/// # Errors
/// Returns an error for a non-positive temperature, electron count or
/// ratio.
pub fn nernst(e0: f64, z: f64, ratio: f64, t: f64) -> Result<f64, GeomError> {
    if !(t > 0.0) || !(z > 0.0) || !(ratio > 0.0) {
        return Err(GeomError::InvalidArgument("nernst: bad parameters"));
    }
    Ok(crate::chemistry::nernst_potential(e0, t, z, ratio))
}

/// The Butler-Volmer current density
/// `i0 (exp(alpha z F eta / RT) - exp(-(1 - alpha) z F eta / RT))`.
///
/// At small overpotential the two exponentials cancel to leading order and
/// the current is *linear* in `eta` with a slope `i0 z F / RT` -- the
/// charge-transfer resistance. At large overpotential one term dominates
/// and the relation becomes the logarithmic Tafel law. Both limits come out
/// of the same expression, which is why fitting a Tafel slope to
/// near-equilibrium data gives a meaningless exchange current.
///
/// # Errors
/// Returns an error for a non-positive temperature or exchange current, an
/// asymmetry outside zero to one, or a non-positive electron count.
pub fn butler_volmer(
    i0: f64,
    alpha: f64,
    eta: f64,
    z: f64,
    t: f64,
) -> Result<f64, GeomError> {
    if !(i0 > 0.0) || !(0.0..=1.0).contains(&alpha) || !(t > 0.0) || !(z > 0.0) {
        return Err(GeomError::InvalidArgument("butler_volmer: bad parameters"));
    }
    let f = z * crate::chemistry::FARADAY / (constants::R * t);
    Ok(i0 * ((alpha * f * eta).exp() - (-(1.0 - alpha) * f * eta).exp()))
}

/// The Cottrell current `z F A c sqrt(D / (pi t))` for a diffusion-limited
/// electrode.
///
/// Falls as the inverse square root of time, not exponentially: the
/// depletion layer grows as `sqrt(D t)`, so the gradient that drives the
/// current thins in proportion. The same square root governs every
/// semi-infinite diffusion problem.
///
/// # Errors
/// Returns an error for a non-positive time, area, diffusion coefficient,
/// concentration or electron count.
pub fn cottrell_current(
    z: f64,
    area: f64,
    concentration: f64,
    diffusivity: f64,
    t: f64,
) -> Result<f64, GeomError> {
    if !(t > 0.0) || !(area > 0.0) || !(diffusivity > 0.0) || !(concentration > 0.0) || !(z > 0.0) {
        return Err(GeomError::InvalidArgument("cottrell_current: bad parameters"));
    }
    Ok(z * crate::chemistry::FARADAY * area * concentration
        * (diffusivity / (std::f64::consts::PI * t)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------
    // Networks
    // -----------------------------------------------------------------

    #[test]
    fn the_stoichiometry_matrix_records_the_net_change() {
        // 2 H2 + O2 -> 2 H2O, with species 0 = H2, 1 = O2, 2 = H2O.
        let burn = Reaction::new(&[(0, 2), (1, 1)], &[(2, 2)]);
        assert_eq!(burn.order(), 3);
        assert_eq!(burn.net_change(3), vec![-2, -1, 2]);
        let m = stoichiometry_matrix(std::slice::from_ref(&burn), 3).unwrap();
        assert_eq!((m.rows, m.cols), (3, 1));
        assert!(close(m.get(0, 0), -2.0, 1e-12));
        assert!(close(m.get(2, 0), 2.0, 1e-12));

        // A species on both sides nets out, which is what makes it a
        // catalyst rather than a reactant.
        let catalysed = Reaction::new(&[(0, 1), (1, 1)], &[(2, 1), (1, 1)]);
        assert_eq!(catalysed.net_change(3), vec![-1, 0, 1]);
        assert_eq!(catalysed.order(), 2, "the catalyst still sets the rate order");

        assert!(stoichiometry_matrix(&[], 3).is_err());
        assert!(stoichiometry_matrix(std::slice::from_ref(&burn), 0).is_err());
        assert!(stoichiometry_matrix(&[burn], 2).is_err());
    }

    #[test]
    fn the_mass_action_rate_is_the_product_of_the_orders() {
        let reactions = [
            Reaction::new(&[(0, 1)], &[(1, 1)]),
            Reaction::new(&[(0, 2)], &[(1, 1)]),
            Reaction::new(&[(0, 1), (1, 1)], &[(2, 1)]),
        ];
        let k = [2.0, 3.0, 5.0];
        let c = [0.5, 0.25, 0.0];
        let v = mass_action_rates(&reactions, &k, &c).unwrap();
        assert!(close(v[0], 2.0 * 0.5, 1e-12));
        assert!(close(v[1], 3.0 * 0.25, 1e-12));
        assert!(close(v[2], 5.0 * 0.5 * 0.25, 1e-12));
        // A zero concentration stops its own reaction and nothing else.
        let none = mass_action_rates(&reactions, &k, &[0.0, 0.25, 0.0]).unwrap();
        assert!(close(none[0], 0.0, 1e-12) && close(none[2], 0.0, 1e-12));
        assert!(mass_action_rates(&reactions, &k[..2], &c).is_err());
        assert!(mass_action_rates(&reactions, &[2.0, -1.0, 5.0], &c).is_err());
        assert!(mass_action_rates(&reactions, &k, &[0.5, 0.25]).is_err());
    }

    #[test]
    fn the_stochastic_propensity_is_a_falling_factorial_not_a_power() {
        // The distinction that matters: a bimolecular reaction of a species
        // with itself proceeds at k x (x - 1) / 2, not k x^2. At two
        // molecules the propensity is k, not 2k, and at one it is zero --
        // a single molecule cannot react with itself, which a power law
        // would not know.
        let dimerise = [Reaction::new(&[(0, 2)], &[(1, 1)])];
        let k = [1.0];
        assert!(close(propensities(&dimerise, &k, &[0])[0], 0.0, 1e-12));
        assert!(close(propensities(&dimerise, &k, &[1])[0], 0.0, 1e-12));
        assert!(close(propensities(&dimerise, &k, &[2])[0], 1.0, 1e-12));
        assert!(close(propensities(&dimerise, &k, &[3])[0], 3.0, 1e-12));
        assert!(close(propensities(&dimerise, &k, &[10])[0], 45.0, 1e-12));
        // And it approaches the continuum k x^2 / 2 only at large counts.
        let big = 10_000.0;
        let exact = propensities(&dimerise, &k, &[10_000])[0];
        assert!(close(exact / (0.5 * big * big), 1.0, 1e-3));
        // A bimolecular reaction between distinct species is the plain
        // product, with no factor of two.
        let cross = [Reaction::new(&[(0, 1), (1, 1)], &[(2, 1)])];
        assert!(close(propensities(&cross, &k, &[3, 4, 0])[0], 12.0, 1e-12));
    }


    // -----------------------------------------------------------------
    // Oscillating mechanisms
    // -----------------------------------------------------------------

    /// The peak-to-trough range of a component over the last third of a run,
    /// selected by *time* rather than by step index.
    ///
    /// The adaptive controller front-loads its steps into whatever transient
    /// the run begins with, so the second half of the step list can still be
    /// inside it. Every "does this settle" question in this module has to be
    /// asked of a time window.
    fn late_swing(trace: &[(f64, Vec<f64>)], component: usize) -> f64 {
        let end = trace.last().map_or(0.0, |(t, _)| *t);
        let tail: Vec<f64> = trace
            .iter()
            .filter(|(t, _)| *t > 2.0 / 3.0 * end)
            .map(|(_, c)| c[component])
            .collect();
        // One sample is enough and is itself informative: a system that has
        // settled produces exactly that, because the controller correctly
        // takes one enormous step across a stretch where nothing changes.
        // Demanding a dense tail would fail on precisely the runs that are
        // most obviously converged.
        assert!(!tail.is_empty(), "the late window is empty");
        let hi = tail.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lo = tail.iter().copied().fold(f64::INFINITY, f64::min);
        hi - lo
    }

    #[test]
    fn the_brusselator_oscillates_exactly_where_the_hopf_condition_says() {
        // The threshold b = 1 + a^2 is a property of the equations, not of
        // the integrator, and it is sharp: just below it every trajectory
        // decays onto the fixed point and just above it every trajectory
        // reaches the same cycle. Both sides are run at three values of a,
        // so the condition is being tested rather than one lucky point.
        for &a in &[0.8f64, 1.0, 1.5] {
            let threshold = 1.0 + a * a;
            assert!(brusselator_oscillates(a, threshold + 0.01));
            assert!(!brusselator_oscillates(a, threshold - 0.01));

            // Comfortably below: the fixed point (a, b/a) is stable, so a
            // trajectory started off it decays back.
            let calm = threshold - 0.5;
            let below =
                oscillating_brusselator(a, calm, (a * 1.4, calm / a * 1.4), 120.0).unwrap();
            let quiet = late_swing(&below, 0);
            assert!(quiet < 0.02, "at a = {a}, b = {calm} the swing is {quiet}");
            let (_, settled) = below.last().unwrap();
            assert!(close(settled[0], a, 0.02), "X settled at {} rather than {a}", settled[0]);
            assert!(
                close(settled[1], calm / a, 0.02),
                "Y settled at {} rather than {}",
                settled[1],
                calm / a
            );

            // Comfortably above: a limit cycle, whose amplitude does not
            // depend on where it started. That independence is what makes
            // it a *limit* cycle rather than a family of orbits, and it is
            // the property that separates the Brusselator from the
            // conservative Lotka-Volterra below.
            let lively = threshold + 1.0;
            let near = oscillating_brusselator(a, lively, (a * 1.05, lively / a * 1.05), 200.0)
                .unwrap();
            let far = oscillating_brusselator(a, lively, (a * 3.0, lively / a * 0.2), 200.0)
                .unwrap();
            let near_swing = late_swing(&near, 0);
            let far_swing = late_swing(&far, 0);
            assert!(near_swing > 0.5, "at a = {a}, b = {lively} the swing is only {near_swing}");
            assert!(
                close(near_swing, far_swing, 0.1 * near_swing),
                "two starts gave swings {near_swing} and {far_swing}"
            );
            // Nothing goes negative, and nothing runs away.
            for (_, c) in &near {
                assert!(c[0] >= 0.0 && c[1] >= 0.0 && c[0] < 100.0 && c[1] < 100.0);
            }
        }
        assert!(oscillating_brusselator(0.0, 1.0, (1.0, 1.0), 10.0).is_err());
        assert!(oscillating_brusselator(1.0, 0.0, (1.0, 1.0), 10.0).is_err());
        assert!(oscillating_brusselator(1.0, 3.0, (-1.0, 1.0), 10.0).is_err());
        assert!(oscillating_brusselator(1.0, 3.0, (1.0, 1.0), 0.0).is_err());
    }

    #[test]
    fn the_oregonator_relaxation_oscillates_over_four_decades_of_timescale() {
        // The standard parameters, where the three variables move on
        // timescales four orders of magnitude apart. The signature of a
        // relaxation oscillator is that the excursion is enormous -- x
        // sweeps several decades -- while the period stays regular, which
        // is what distinguishes it from a smooth oscillation.
        let trace = oregonator(1e-2, 2.5e-5, 2e-4, 1.0, (1.0, 1.0, 1.0), 60.0).unwrap();
        // By time, not by index: see the note on `late_swing`.
        let end = trace.last().unwrap().0;
        let tail: Vec<(f64, Vec<f64>)> =
            trace.iter().filter(|(t, _)| *t > end / 3.0).cloned().collect();
        let hi = tail.iter().map(|(_, c)| c[0]).fold(f64::NEG_INFINITY, f64::max);
        let lo = tail.iter().map(|(_, c)| c[0]).fold(f64::INFINITY, f64::min);
        assert!(hi / lo.max(1e-12) > 1e3, "x swept only {} decades", (hi / lo).log10());
        assert!(hi.is_finite() && hi < 1e6, "x ran away to {hi}");
        for (_, c) in &trace {
            assert!(c.iter().all(|v| *v >= 0.0), "a concentration went negative");
            assert!(c.iter().all(|v| v.is_finite()), "the run blew up");
        }
        // The period is regular: successive crossings of a threshold are
        // evenly spaced. A run that merely wandered would not be.
        let threshold = (hi * lo.max(1e-12)).sqrt();
        let mut crossings = Vec::new();
        for pair in tail.windows(2) {
            if pair[0].1[0] < threshold && pair[1].1[0] >= threshold {
                crossings.push(pair[1].0);
            }
        }
        assert!(crossings.len() >= 3, "only {} crossings found", crossings.len());
        let periods: Vec<f64> = crossings.windows(2).map(|p| p[1] - p[0]).collect();
        let mean: f64 = periods.iter().sum::<f64>() / periods.len() as f64;
        for period in &periods {
            assert!(
                close(*period, mean, 0.1 * mean),
                "a period of {period} against a mean of {mean}"
            );
        }
        assert!(oregonator(0.0, 1e-4, 1e-3, 1.0, (1.0, 1.0, 1.0), 1.0).is_err());
        assert!(oregonator(1e-2, 0.0, 1e-3, 1.0, (1.0, 1.0, 1.0), 1.0).is_err());
        assert!(oregonator(1e-2, 1e-4, 1e-3, 1.0, (-1.0, 1.0, 1.0), 1.0).is_err());
    }

    #[test]
    fn the_chemical_lotka_volterra_conserves_its_invariant() {
        // Unlike the Brusselator this system is conservative: the orbits are
        // closed curves labelled by a constant, and the amplitude *does*
        // depend on where it started. Both halves are checked, because
        // getting the first without the second would mean the integrator was
        // damping the orbit onto a spurious cycle.
        let (a, k1, k2, k3) = (1.0f64, 1.0f64, 1.0f64, 1.0f64);
        let (trace, invariant) = lotka_volterra_chemical(a, k1, k2, k3, (1.2, 0.9), 30.0).unwrap();
        let first = invariant[0];
        for (k, v) in invariant.iter().enumerate() {
            assert!(
                close(*v, first, 3e-4 * first.abs().max(1.0)),
                "the invariant drifted from {first} to {v} at step {k}"
            );
        }
        // It really does go round: both populations swing.
        assert!(late_swing(&trace, 0) > 0.1, "X barely moved");
        assert!(late_swing(&trace, 1) > 0.1, "Y barely moved");
        // A different start gives a different orbit, which is the mark of a
        // conservative system rather than a limit cycle.
        let (wide, _) = lotka_volterra_chemical(a, k1, k2, k3, (2.5, 0.4), 30.0).unwrap();
        assert!(
            late_swing(&wide, 0) > 1.5 * late_swing(&trace, 0),
            "a wider start gave the same orbit, so this is behaving as a limit cycle"
        );
        // And the fixed point stays put: (k3/k2, k1 a/k2).
        let (fixed, _) =
            lotka_volterra_chemical(a, k1, k2, k3, (k3 / k2, k1 * a / k2), 40.0).unwrap();
        assert!(late_swing(&fixed, 0) < 1e-3, "the fixed point moved");
        assert!(lotka_volterra_chemical(0.0, 1.0, 1.0, 1.0, (1.0, 1.0), 1.0).is_err());
        assert!(lotka_volterra_chemical(1.0, 1.0, 1.0, 1.0, (0.0, 1.0), 1.0).is_err());
    }

    #[test]
    fn autocatalysis_ignites_when_the_logistic_curve_says_it_does() {
        // The closed form against a direct integration of the same
        // mechanism, which is a check of the formula rather than of a
        // remembered number.
        for &k in &[0.5f64, 2.0, 10.0] {
            for &b0 in &[1e-6f64, 1e-3, 0.1] {
                let a0 = 1.0;
                let predicted = autocatalysis_ignition(a0, b0, k).unwrap();
                let reactions = [Reaction::new(&[(0, 1), (1, 1)], &[(1, 2)])];
                let stoich = stoichiometry_matrix(&reactions, 2).unwrap();
                let rates = |c: &[f64]| vec![k * c[0].max(0.0) * c[1].max(0.0)];
                let trace =
                    rate_equations(&stoich, &rates, &[a0, b0], 2.0 * predicted, 1e-10).unwrap();
                let half = 0.5 * (a0 + b0);
                let crossing = trace
                    .windows(2)
                    .find(|w| w[0].1[1] < half && w[1].1[1] >= half)
                    .map(|w| w[1].0);
                let crossing = crossing.expect("the reaction did not reach half conversion");
                assert!(
                    close(crossing, predicted, 0.02 * predicted),
                    "at k = {k}, b0 = {b0} the integration crossed at {crossing} against {predicted}"
                );
            }
        }
        // The induction period grows as the seed shrinks, logarithmically.
        let long = autocatalysis_ignition(1.0, 1e-9, 1.0).unwrap();
        let short = autocatalysis_ignition(1.0, 1e-3, 1.0).unwrap();
        assert!(long > short);
        assert!(close(long / short, 1e-9f64.ln() / 1e-3f64.ln(), 0.01));
        // Already past half conversion at the start.
        assert!(close(autocatalysis_ignition(1.0, 2.0, 1.0).unwrap(), 0.0, 1e-15));
        assert!(autocatalysis_ignition(1.0, 1.0, 0.0).is_err());
        assert!(autocatalysis_ignition(0.0, 1.0, 1.0).is_err());
        assert!(autocatalysis_ignition(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn the_branching_ratio_is_the_explosion_criterion() {
        assert!(close(chain_reaction_criticality(3.0, 1.5).unwrap(), 2.0, 1e-12));
        assert!(chain_reaction_criticality(1.0, 1.0).unwrap() == 1.0, "the threshold is exactly one");
        assert!(chain_reaction_criticality(0.99, 1.0).unwrap() < 1.0);
        assert!(chain_reaction_criticality(1.01, 1.0).unwrap() > 1.0);
        assert!(close(chain_reaction_criticality(0.0, 1.0).unwrap(), 0.0, 1e-15));
        assert!(chain_reaction_criticality(1.0, 0.0).is_err());
        assert!(chain_reaction_criticality(-1.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Rate theory
    // -----------------------------------------------------------------

    #[test]
    fn eyring_and_arrhenius_describe_the_same_curve_differently() {
        // Fitting an Arrhenius form to Eyring data must recover an
        // activation energy of dH + RT and a prefactor that absorbs the
        // entropy -- the two forms are reparameterisations of nearly the
        // same temperature dependence, differing by the linear factor T in
        // the Eyring prefactor. That is the whole content of the
        // relationship and it is checked by fitting rather than asserted.
        let (dh, ds) = (80_000.0f64, -50.0f64);
        let temperatures: Vec<f64> = (0..12).map(|k| 280.0 + f64::from(k) * 10.0).collect();
        let rates: Vec<f64> = temperatures.iter().map(|t| eyring(dh, ds, *t).unwrap()).collect();
        // Arrhenius fit: ln k against 1/T.
        let n = temperatures.len() as f64;
        let sx: f64 = temperatures.iter().map(|t| 1.0 / t).sum();
        let sy: f64 = rates.iter().map(|k| k.ln()).sum();
        let sxx: f64 = temperatures.iter().map(|t| 1.0 / (t * t)).sum();
        let sxy: f64 =
            temperatures.iter().zip(&rates).map(|(t, k)| k.ln() / t).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let ea = -slope * constants::R;
        let mid = 335.0;
        assert!(
            close(ea, dh + constants::R * mid, 0.02 * ea),
            "the fitted activation energy is {ea} against {}",
            dh + constants::R * mid
        );
        // A negative entropy of activation slows the reaction, an ordering
        // transition state being harder to reach.
        assert!(eyring(dh, -50.0, 300.0).unwrap() < eyring(dh, 0.0, 300.0).unwrap());
        assert!(eyring(dh, 50.0, 300.0).unwrap() > eyring(dh, 0.0, 300.0).unwrap());
        // The universal prefactor at 298 K is about 6.2e12 per second.
        assert!(close(eyring(0.0, 0.0, 298.15).unwrap() / 1e12, 6.21, 0.05));
        assert!(eyring(dh, ds, 0.0).is_err());

        // Transition-state theory is an upper bound: a transmission
        // coefficient below one can only lower it.
        let full = transition_state_theory_rate(80_000.0, 300.0, 1.0).unwrap();
        assert!(transition_state_theory_rate(80_000.0, 300.0, 0.3).unwrap() < full);
        assert!(close(
            transition_state_theory_rate(80_000.0, 300.0, 0.3).unwrap(),
            0.3 * full,
            1e-9 * full
        ));
        assert!(close(transition_state_theory_rate(80_000.0, 300.0, 0.0).unwrap(), 0.0, 1e-30));
        assert!(transition_state_theory_rate(1.0, 0.0, 1.0).is_err());
        assert!(transition_state_theory_rate(1.0, 300.0, 1.5).is_err());
        assert!(transition_state_theory_rate(1.0, 300.0, -0.1).is_err());
    }

    #[test]
    fn the_kramers_factor_falls_from_one_toward_the_inverse_friction() {
        // No friction, no recrossing: the transition-state result stands. As
        // the friction grows the factor falls toward omega_b / gamma, the
        // Smoluchowski limit, and it never exceeds one -- which is what
        // makes transition-state theory a bound.
        assert!(close(kramers_rate_check(0.0, 1.0).unwrap(), 1.0, 1e-15));
        let mut previous = 1.0;
        for step in 1..=40 {
            let gamma = f64::from(step) * 0.5;
            let factor = kramers_rate_check(gamma, 1.0).unwrap();
            assert!(factor <= 1.0 + 1e-12, "the factor {factor} exceeds one");
            assert!(factor > 0.0);
            assert!(factor < previous, "the factor rose from {previous} to {factor}");
            previous = factor;
        }
        // The high-friction limit.
        for &gamma in &[100.0f64, 1_000.0, 10_000.0] {
            let factor = kramers_rate_check(gamma, 1.0).unwrap();
            assert!(
                close(factor, 1.0 / gamma, 0.02 / gamma),
                "at gamma {gamma} the factor is {factor} against {}",
                1.0 / gamma
            );
        }
        // Only the ratio matters.
        assert!(close(
            kramers_rate_check(6.0, 3.0).unwrap(),
            kramers_rate_check(2.0, 1.0).unwrap(),
            1e-12
        ));
        assert!(kramers_rate_check(1.0, 0.0).is_err());
        assert!(kramers_rate_check(-1.0, 1.0).is_err());
    }

    #[test]
    fn the_isotope_effect_reaches_its_semiclassical_maximum_for_hydrogen() {
        // A C-H stretch near 3000 wavenumbers against a C-D stretch near
        // 2200 gives about seven at room temperature, and that number is the
        // accepted semiclassical ceiling. A measured ratio well above it is
        // evidence of tunnelling, which this estimate deliberately omits.
        let ratio = kinetic_isotope_effect_estimate(3_000.0, 2_200.0, 298.15).unwrap();
        assert!(close(ratio, 6.9, 0.4), "the C-H/C-D effect came out {ratio}");
        // It falls with temperature, since zero-point energy matters less
        // against a larger thermal energy.
        let hot = kinetic_isotope_effect_estimate(3_000.0, 2_200.0, 600.0).unwrap();
        assert!(hot < ratio && hot > 1.0);
        // Equal frequencies mean no effect at all.
        assert!(close(kinetic_isotope_effect_estimate(3_000.0, 3_000.0, 298.15).unwrap(), 1.0, 1e-12));
        // A heavier light isotope would be an inverse effect.
        assert!(kinetic_isotope_effect_estimate(2_200.0, 3_000.0, 298.15).unwrap() < 1.0);
        assert!(kinetic_isotope_effect_estimate(3_000.0, 2_200.0, 0.0).is_err());
        assert!(kinetic_isotope_effect_estimate(0.0, 2_200.0, 298.0).is_err());
    }

    #[test]
    fn a_relaxation_measures_the_sum_of_the_rates_not_either_one() {
        // The point of the technique: one relaxation gives the sum, the
        // equilibrium constant gives the ratio, and together they separate
        // two rate constants that no steady-state measurement can.
        for &(kf, kr) in &[(3.0f64, 1.0f64), (0.5, 2.5), (10.0, 10.0)] {
            let tau = temperature_jump_relaxation(kf, kr).unwrap();
            assert!(close(tau, 1.0 / (kf + kr), 1e-12));
            // Recovering both from the pair of measurements.
            let k_eq = kf / kr;
            let sum = 1.0 / tau;
            let recovered_kr = sum / (1.0 + k_eq);
            assert!(close(recovered_kr, kr, 1e-9 * kr));
            assert!(close(sum - recovered_kr, kf, 1e-9 * kf));
        }
        // A one-way reaction still relaxes, at its own rate.
        assert!(close(temperature_jump_relaxation(4.0, 0.0).unwrap(), 0.25, 1e-12));
        assert!(temperature_jump_relaxation(0.0, 0.0).is_err());
        assert!(temperature_jump_relaxation(-1.0, 1.0).is_err());
    }

    #[test]
    fn nucleation_is_as_sensitive_to_the_driving_force_as_the_theory_says() {
        // The barrier goes as the inverse square of the driving force and
        // the rate as its exponential, so a factor of two in supersaturation
        // moves the rate by dozens of orders of magnitude. That extreme
        // sensitivity is the physics -- it is why nucleation looks like it
        // has a threshold -- and a formula that merely varied smoothly would
        // be the wrong one.
        let (sigma, density) = (0.05f64, 3.3e28f64);
        let weak = nucleation_barrier(sigma, density, 1e-21).unwrap();
        let strong = nucleation_barrier(sigma, density, 2e-21).unwrap();
        assert!(close(weak / strong, 4.0, 1e-9), "the barrier is not inverse square");
        let closed = 16.0 * std::f64::consts::PI * sigma.powi(3)
            / (3.0 * (density * 1e-21) * (density * 1e-21));
        assert!(close(weak, closed, 1e-9 * closed));

        let t = 300.0;
        let slow = nucleation_rate_cnt(weak, 1e35, t).unwrap();
        let fast = nucleation_rate_cnt(strong, 1e35, t).unwrap();
        assert!(fast > slow);
        assert!(
            (fast / slow.max(1e-300)).log10() > 10.0,
            "doubling the driving force moved the rate by only {} decades",
            (fast / slow.max(1e-300)).log10()
        );
        // No barrier, no suppression.
        assert!(close(nucleation_rate_cnt(0.0, 1e35, t).unwrap(), 1e35, 1.0));
        assert!(nucleation_rate_cnt(1.0, 0.0, t).is_err());
        assert!(nucleation_rate_cnt(1.0, 1e35, 0.0).is_err());
        assert!(nucleation_barrier(0.0, density, 1e-21).is_err());
        assert!(nucleation_barrier(sigma, density, 0.0).is_err());
    }

    #[test]
    fn the_avrami_fit_reads_back_the_dimensionality_it_was_given() {
        for &n in &[1.0f64, 1.5, 2.0, 3.0, 4.0] {
            for &k in &[0.2f64, 1.0, 5.0] {
                let times: Vec<f64> = (1..=25).map(|j| f64::from(j) * 0.1 / k).collect();
                let fraction: Vec<f64> = times.iter().map(|t| jmak_avrami(*t, k, n)).collect();
                let (fit_k, fit_n) = avrami_fit(&times, &fraction).unwrap();
                assert!(close(fit_n, n, 1e-6 * n), "the exponent {n} came back as {fit_n}");
                assert!(close(fit_k, k, 1e-6 * k), "the rate {k} came back as {fit_k}");
            }
        }
        // The curve runs from nothing to everything, monotonically, and
        // passes 1 - 1/e exactly at t = 1/k whatever the exponent.
        for &n in &[1.0f64, 3.0] {
            assert!(close(jmak_avrami(1.0 / 2.0, 2.0, n), 1.0 - (-1.0f64).exp(), 1e-12));
            assert!(close(jmak_avrami(0.0, 2.0, n), 0.0, 1e-15));
            assert!(close(jmak_avrami(1e6, 2.0, n), 1.0, 1e-12));
            let mut previous = 0.0;
            for step in 1..=40 {
                // Stopping at k t = 2: beyond about (k t)^n = 37 the
                // exponential underflows and the fraction rounds to exactly
                // one, so a strict `x < 1` would be testing double precision
                // rather than the curve.
                let x = jmak_avrami(f64::from(step) * 0.025, 2.0, n);
                assert!(x > previous, "the curve went back down at step {step}");
                assert!(x < 1.0, "the curve saturated at step {step}");
                previous = x;
            }
        }
        assert!(close(jmak_avrami(-1.0, 1.0, 1.0), 0.0, 1e-15));
        assert!(close(jmak_avrami(1.0, 0.0, 1.0), 0.0, 1e-15));
        // Points at exactly nothing or everything carry no information.
        assert!(avrami_fit(&[1.0, 2.0], &[0.0, 1.0]).is_err());
        assert!(avrami_fit(&[1.0, 2.0], &[0.3]).is_err());
        assert!(avrami_fit(&[1.0, 1.0, 1.0], &[0.2, 0.3, 0.4]).is_err());
    }

    #[test]
    fn a_quantum_yield_above_one_is_a_chain_and_not_an_error() {
        assert!(close(photochemistry_quantum_yield(50.0, 100.0).unwrap(), 0.5, 1e-12));
        // A chain reaction can transform thousands of molecules per photon,
        // so no ceiling is imposed.
        assert!(close(photochemistry_quantum_yield(1e6, 1.0).unwrap(), 1e6, 1e-6));
        assert!(close(photochemistry_quantum_yield(0.0, 10.0).unwrap(), 0.0, 1e-15));
        assert!(photochemistry_quantum_yield(1.0, 0.0).is_err());
        assert!(photochemistry_quantum_yield(-1.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Acid-base and electrochemistry
    // -----------------------------------------------------------------

    #[test]
    fn the_ph_solver_agrees_with_the_quadratic_where_the_quadratic_is_valid() {
        // For a moderately concentrated weak acid the usual approximation
        // -- ignore water, allow for depletion -- is a quadratic with a
        // closed-form root, and the full solver must match it there.
        for &pka in &[3.0f64, 4.76, 7.0] {
            for &c in &[0.001f64, 0.01, 0.1, 1.0] {
                let ka = 10f64.powf(-pka);
                // ka = h^2 / (c - h).
                let h = (-ka + (ka * ka + 4.0 * ka * c).sqrt()) / 2.0;
                let quadratic = -h.log10();
                let solved = ph_from_equilibria(&[(pka, c)], 0.0).unwrap();
                assert!(
                    close(solved, quadratic, 0.01),
                    "pKa {pka} at {c} M: the solver gives {solved} against {quadratic}"
                );
            }
        }
        // 0.1 M acetic acid is pH 2.87, the standard textbook figure.
        assert!(close(ph_from_equilibria(&[(4.76, 0.1)], 0.0).unwrap(), 2.87, 0.02));
        // Pure water is neutral.
        assert!(close(ph_from_equilibria(&[], 0.0).unwrap(), 7.0, 1e-3));

        // Where the quadratic fails and the full balance does not: a very
        // dilute acid cannot be more acidic than water, so the pH must
        // approach 7 from below rather than continuing up.
        let dilute = ph_from_equilibria(&[(4.76, 1e-9)], 0.0).unwrap();
        assert!(dilute < 7.0 && dilute > 6.9, "a 1 nM acid gives pH {dilute}");
        let quadratic_would_say = {
            let ka = 10f64.powf(-4.76);
            let c = 1e-9;
            -((-ka + (ka * ka + 4.0 * ka * c).sqrt()) / 2.0).log10()
        };
        assert!(
            quadratic_would_say > 7.5,
            "the fixture does not show the failure: the quadratic says {quadratic_would_say}"
        );
        // A strong base drives it up.
        assert!(ph_from_equilibria(&[(4.76, 0.1)], 0.05).unwrap() > 4.0);
        assert!(ph_from_equilibria(&[(4.76, -0.1)], 0.0).is_err());
        assert!(ph_from_equilibria(&[], -1.0).is_err());
    }

    #[test]
    fn a_titration_passes_through_the_buffer_region_and_over_the_equivalence_point() {
        // Half way to equivalence the pH equals the pKa, which is the
        // definition of a buffer and the reason Henderson-Hasselbalch works
        // there. At equivalence a weak acid's conjugate base makes the
        // solution basic, which is what the full balance gets right and the
        // half-reaction intuition does not.
        let pka = 4.76;
        let curve = titration_curve(pka, 0.1, 50.0, 0.1, 100.0, 401).unwrap();
        assert_eq!(curve.len(), 401);
        let at = |volume: f64| -> f64 {
            curve
                .iter()
                .min_by(|a, b| {
                    (a.0 - volume).abs().partial_cmp(&(b.0 - volume).abs()).unwrap()
                })
                .unwrap()
                .1
        };
        assert!(close(at(25.0), pka, 0.03), "the half-equivalence pH is {}", at(25.0));
        let equivalence = at(50.0);
        assert!(equivalence > 8.0 && equivalence < 9.5, "the equivalence pH is {equivalence}");
        // Monotone throughout, and steepest at the equivalence point.
        for pair in curve.windows(2) {
            assert!(pair[1].1 >= pair[0].1 - 1e-9, "the curve went back down");
        }
        let slope = |volume: f64| (at(volume + 1.0) - at(volume - 1.0)).abs();
        assert!(slope(50.0) > 8.0 * slope(25.0), "the equivalence point is not a jump");
        assert!(titration_curve(pka, 0.0, 50.0, 0.1, 100.0, 10).is_err());
        assert!(titration_curve(pka, 0.1, 0.0, 0.1, 100.0, 10).is_err());
        assert!(titration_curve(pka, 0.1, 50.0, 0.1, 0.0, 10).is_err());
        assert!(titration_curve(pka, 0.1, 50.0, 0.1, 100.0, 1).is_err());
    }

    #[test]
    fn henderson_hasselbalch_is_right_near_the_pka_and_wrong_away_from_it() {
        // The approximation and its failure, both demonstrated. It assumes
        // the dissociation does not change either concentration, which is
        // true within about a unit of the pKa and false outside it.
        let pka = 4.76;
        assert!(close(buffer_henderson_hasselbalch(pka, 1.0).unwrap(), pka, 1e-12));
        assert!(close(buffer_henderson_hasselbalch(pka, 10.0).unwrap(), pka + 1.0, 1e-12));
        assert!(close(buffer_henderson_hasselbalch(pka, 0.1).unwrap(), pka - 1.0, 1e-12));
        // Against the full balance for a real buffer: an acetate buffer of
        // 0.1 M acid and 0.1 M conjugate base is 0.1 M acid plus 0.1 M
        // strong base as far as the balance is concerned.
        let full = ph_from_equilibria(&[(pka, 0.2)], 0.1).unwrap();
        assert!(
            close(full, pka, 0.02),
            "the balance gives {full} where Henderson-Hasselbalch gives {pka}"
        );
        // Where it fails is a *dilute* buffer, not merely a lopsided one:
        // the shortcut assumes the dissociation does not appreciably change
        // either concentration, and at a micromolar an acid of this
        // strength is almost entirely dissociated whatever ratio was
        // weighed out. At 0.101 M acid with 0.1 M base the two agree to
        // five decimals, so a lopsided ratio alone shows nothing.
        let lopsided = ph_from_equilibria(&[(pka, 0.101)], 0.1).unwrap();
        assert!(
            close(lopsided, buffer_henderson_hasselbalch(pka, 100.0).unwrap(), 0.01),
            "a concentrated buffer should still obey the shortcut, but gives {lopsided}"
        );
        let dilute = ph_from_equilibria(&[(pka, 1e-6)], 5e-7).unwrap();
        let approximated = buffer_henderson_hasselbalch(pka, 1.0).unwrap();
        assert!(
            (dilute - approximated).abs() > 1.0,
            "a micromolar buffer gives {dilute} against the shortcut's {approximated}"
        );
        assert!(buffer_henderson_hasselbalch(pka, 0.0).is_err());
    }

    #[test]
    fn the_activity_coefficient_falls_from_one_and_scales_with_the_charge_squared() {
        // At infinite dilution every ion is ideal; the coefficient falls as
        // the ionic strength rises, and it falls far faster for a doubly
        // charged ion -- the z^2 is the whole content of the law.
        assert!(close(debye_huckel_activity(1.0, 0.0).unwrap(), 1.0, 1e-15));
        let mut previous = 1.0;
        for step in 1..=20 {
            let i = f64::from(step) * 0.005;
            let gamma = debye_huckel_activity(1.0, i).unwrap();
            assert!(gamma < previous && gamma > 0.0);
            previous = gamma;
        }
        for &i in &[0.001f64, 0.01, 0.1] {
            let single = debye_huckel_activity(1.0, i).unwrap();
            let double = debye_huckel_activity(2.0, i).unwrap();
            // log gamma scales as z^2, so the double-charge coefficient is
            // the single one raised to the fourth power.
            assert!(
                close(double, single.powi(4), 1e-9),
                "at I = {i} the divalent coefficient is {double} against {}",
                single.powi(4)
            );
            // The sign of the charge does not matter.
            assert!(close(debye_huckel_activity(-1.0, i).unwrap(), single, 1e-15));
        }
        // The standard figure: a monovalent ion at I = 0.1 has gamma 0.755.
        assert!(close(debye_huckel_activity(1.0, 0.1).unwrap(), 0.755, 0.005));
        assert!(debye_huckel_activity(1.0, -0.1).is_err());
    }

    #[test]
    fn the_nernst_slope_is_fifty_nine_millivolts_a_decade() {
        // The number every ion-selective electrode is calibrated against,
        // and it follows from RT/F alone.
        let t = 298.15;
        let a = nernst(0.0, 1.0, 1.0, t).unwrap();
        let b = nernst(0.0, 1.0, 10.0, t).unwrap();
        assert!(close(a, 0.0, 1e-15));
        assert!(
            close((a - b) * 1000.0, 59.16, 0.05),
            "the slope is {} mV per decade",
            (a - b) * 1000.0
        );
        // Two electrons halve it.
        let two = nernst(0.0, 2.0, 10.0, t).unwrap();
        assert!(close((a - two) * 1000.0, 29.58, 0.05));
        // And it scales with temperature.
        let hot = nernst(0.0, 1.0, 10.0, 2.0 * t).unwrap();
        assert!(close(hot, 2.0 * b, 1e-9 * b.abs()));
        assert!(nernst(0.0, 1.0, 10.0, 0.0).is_err());
        assert!(nernst(0.0, 0.0, 10.0, t).is_err());
        assert!(nernst(0.0, 1.0, 0.0, t).is_err());
    }

    #[test]
    fn butler_volmer_is_linear_near_equilibrium_and_logarithmic_far_from_it() {
        // Both limits come out of the same expression, which is why fitting
        // a Tafel slope to near-equilibrium data gives a meaningless
        // exchange current -- the data there is not on the logarithmic
        // branch at all.
        let (i0, alpha, z, t) = (1e-3f64, 0.5f64, 1.0f64, 298.15f64);
        assert!(close(butler_volmer(i0, alpha, 0.0, z, t).unwrap(), 0.0, 1e-18));
        // Near equilibrium: i = i0 z F eta / RT, the charge-transfer
        // resistance.
        let f = z * crate::chemistry::FARADAY / (constants::R * t);
        for &eta in &[1e-5f64, 1e-4, 1e-3] {
            let linear = i0 * f * eta;
            let exact = butler_volmer(i0, alpha, eta, z, t).unwrap();
            assert!(
                close(exact, linear, 0.01 * linear),
                "at eta = {eta} the current is {exact} against the linear {linear}"
            );
        }
        // Far from it: a decade of current per 2.303 RT / (alpha z F) volts.
        let tafel_slope = std::f64::consts::LN_10 / (alpha * f);
        let high = butler_volmer(i0, alpha, 0.4, z, t).unwrap();
        let higher = butler_volmer(i0, alpha, 0.4 + tafel_slope, z, t).unwrap();
        assert!(
            close(higher / high, 10.0, 0.1),
            "one Tafel slope multiplied the current by {}",
            higher / high
        );
        // Odd in the overpotential at alpha = 1/2, and only then.
        assert!(close(
            butler_volmer(i0, 0.5, 0.1, z, t).unwrap(),
            -butler_volmer(i0, 0.5, -0.1, z, t).unwrap(),
            1e-12
        ));
        assert!(!close(
            butler_volmer(i0, 0.3, 0.1, z, t).unwrap(),
            -butler_volmer(i0, 0.3, -0.1, z, t).unwrap(),
            1e-6
        ));
        assert!(butler_volmer(0.0, alpha, 0.1, z, t).is_err());
        assert!(butler_volmer(i0, 1.5, 0.1, z, t).is_err());
        assert!(butler_volmer(i0, alpha, 0.1, z, 0.0).is_err());
    }

    #[test]
    fn the_cottrell_current_falls_as_the_inverse_square_root_of_time() {
        // The depletion layer grows as sqrt(D t), so the gradient driving
        // the current thins in proportion. Quadrupling the time halves the
        // current -- exactly, not approximately.
        let (z, area, c, d) = (1.0f64, 0.01f64, 1e-3f64, 1e-9f64);
        let early = cottrell_current(z, area, c, d, 1.0).unwrap();
        let late = cottrell_current(z, area, c, d, 4.0).unwrap();
        assert!(close(late * 2.0, early, 1e-9 * early));
        // And it is linear in every other argument.
        assert!(close(
            cottrell_current(z, 2.0 * area, c, d, 1.0).unwrap(),
            2.0 * early,
            1e-9 * early
        ));
        assert!(close(
            cottrell_current(z, area, 3.0 * c, d, 1.0).unwrap(),
            3.0 * early,
            1e-9 * early
        ));
        assert!(close(
            cottrell_current(z, area, c, 4.0 * d, 1.0).unwrap(),
            2.0 * early,
            1e-9 * early
        ));
        let closed = z * crate::chemistry::FARADAY * area * c
            * (d / std::f64::consts::PI).sqrt();
        assert!(close(early, closed, 1e-12 * closed));
        assert!(cottrell_current(z, area, c, d, 0.0).is_err());
        assert!(cottrell_current(z, 0.0, c, d, 1.0).is_err());
        assert!(cottrell_current(0.0, area, c, d, 1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Enzyme kinetics
    // -----------------------------------------------------------------

    #[test]
    fn the_saturation_curves_have_the_limits_that_define_them() {
        // km is the concentration at half saturation and vmax is the
        // asymptote -- that is what the two constants *mean*, so they are
        // checked as limits rather than against tabulated values.
        for &vmax in &[0.5f64, 2.0, 10.0] {
            for &km in &[0.1f64, 1.0, 7.0] {
                assert!(close(michaelis_menten(km, vmax, km), 0.5 * vmax, 1e-12));
                assert!(close(michaelis_menten(0.0, vmax, km), 0.0, 1e-12));
                assert!(close(michaelis_menten(1e9 * km, vmax, km), vmax, 1e-6 * vmax));
                // Far below saturation it is first order with slope
                // vmax / km, the specificity constant.
                let small = 1e-6 * km;
                assert!(close(
                    michaelis_menten(small, vmax, km),
                    vmax * small / km,
                    1e-9 * vmax
                ));
                // Monotone in the substrate, always.
                let mut previous = 0.0;
                for step in 1..=50 {
                    let v = michaelis_menten(f64::from(step) * km / 5.0, vmax, km);
                    assert!(v > previous);
                    previous = v;
                }
                // Hill with n = 1 is exactly Michaelis-Menten.
                for step in 1..=20 {
                    let s = f64::from(step) * km / 3.0;
                    assert!(close(
                        hill_equation(s, vmax, km, 1.0),
                        michaelis_menten(s, vmax, km),
                        1e-12
                    ));
                }
                // And a larger exponent makes the curve steeper at the
                // half point without moving it -- that is what
                // cooperativity is.
                assert!(close(hill_equation(km, vmax, km, 4.0), 0.5 * vmax, 1e-12));
                let low = 0.5 * km;
                assert!(hill_equation(low, vmax, km, 4.0) < hill_equation(low, vmax, km, 1.0));
                let high = 2.0 * km;
                assert!(hill_equation(high, vmax, km, 4.0) > hill_equation(high, vmax, km, 1.0));
            }
        }
        assert!(close(hill_equation(-1.0, 1.0, 1.0, 2.0), 0.0, 1e-15));
        assert!(close(hill_equation(1.0, 1.0, 0.0, 2.0), 0.0, 1e-15));
    }

    #[test]
    fn the_michaelis_menten_fit_recovers_the_constants_it_was_generated_from() {
        // Exact data first, where the fit must be exact, then noisy data,
        // where it must stay close. Across a range of both constants, so a
        // fit that happened to work at one scale could not pass.
        let mut rng = Rng::new(0x_C0DE_0010);
        for &vmax in &[0.4f64, 3.0, 25.0] {
            for &km in &[0.05f64, 1.0, 12.0] {
                let s: Vec<f64> = (1..=12).map(|k| km * f64::from(k) * 0.4).collect();
                let v: Vec<f64> = s.iter().map(|x| michaelis_menten(*x, vmax, km)).collect();
                let (fit_vmax, fit_km) = mm_fit(&s, &v).unwrap();
                assert!(
                    close(fit_vmax, vmax, 1e-6 * vmax),
                    "vmax {vmax} came back as {fit_vmax}"
                );
                assert!(close(fit_km, km, 1e-6 * km), "km {km} came back as {fit_km}");

                let noisy: Vec<f64> = v
                    .iter()
                    .map(|y| y * (1.0 + 0.03 * rng.next_gaussian()))
                    .map(|y| y.max(1e-12))
                    .collect();
                let (noisy_vmax, noisy_km) = mm_fit(&s, &noisy).unwrap();
                assert!(close(noisy_vmax, vmax, 0.15 * vmax), "noisy vmax {noisy_vmax}");
                assert!(close(noisy_km, km, 0.3 * km), "noisy km {noisy_km}");
            }
        }
        assert!(mm_fit(&[1.0, 2.0], &[1.0, 2.0]).is_err());
        assert!(mm_fit(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
        assert!(mm_fit(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]).is_err());
        assert!(mm_fit(&[1.0, -2.0, 3.0], &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn the_double_reciprocal_line_carries_the_constants_and_biases_the_fit() {
        // On exact data the transform is exact, so the slope and intercept
        // are km/vmax and 1/vmax to rounding. On noisy data it is *worse*
        // than the direct fit, which is the documented reason not to use it
        // for the numbers -- and a claim worth demonstrating rather than
        // asserting, since it is the whole justification for mm_fit.
        let (vmax, km) = (4.0f64, 2.0f64);
        let s: Vec<f64> = (1..=10).map(|k| km * f64::from(k) * 0.5).collect();
        let v: Vec<f64> = s.iter().map(|x| michaelis_menten(*x, vmax, km)).collect();
        let (points, slope, intercept) = lineweaver_burk(&s, &v).unwrap();
        assert_eq!(points.len(), s.len());
        assert!(close(points[0].0, 1.0 / s[0], 1e-12));
        assert!(close(intercept, 1.0 / vmax, 1e-9));
        assert!(close(slope, km / vmax, 1e-9));

        // Additive noise of a fixed size, as an instrument with a detection
        // floor would produce. That is the case the double-reciprocal plot
        // mishandles: a small absolute error at a low rate becomes a large
        // one in 1/v, and those points sit furthest out on the transformed
        // axis where they have the most leverage on the intercept. Noise
        // *proportional* to the rate would survive the transform unchanged
        // and show nothing -- and did, when it was tried.
        let mut rng = Rng::new(0x_C0DE_0011);
        let sigma = 0.05 * vmax;
        let mut reciprocal_error = 0.0;
        let mut direct_error = 0.0;
        let trials = 400;
        for _ in 0..trials {
            let noisy: Vec<f64> = v
                .iter()
                .map(|y| (y + sigma * rng.next_gaussian()).max(1e-6 * vmax))
                .collect();
            let (_, _, b) = lineweaver_burk(&s, &noisy).unwrap();
            reciprocal_error += (1.0 / b - vmax).abs();
            let (fit_vmax, _) = mm_fit(&s, &noisy).unwrap();
            direct_error += (fit_vmax - vmax).abs();
        }
        reciprocal_error /= f64::from(trials);
        direct_error /= f64::from(trials);
        assert!(
            direct_error < 0.7 * reciprocal_error,
            "the direct fit erred by {direct_error} against the transform's {reciprocal_error}"
        );
        assert!(lineweaver_burk(&[1.0], &[1.0]).is_err());
        assert!(lineweaver_burk(&[1.0, 0.0], &[1.0, 1.0]).is_err());
        assert!(lineweaver_burk(&[1.0, 2.0], &[1.0, 0.0]).is_err());
        assert!(lineweaver_burk(&[2.0, 2.0], &[1.0, 1.0]).is_err());
    }

    #[test]
    fn the_hill_fit_recovers_its_own_cooperativity() {
        for &n in &[0.8f64, 1.0, 2.0, 3.5] {
            let (vmax, k) = (5.0f64, 1.5f64);
            let s: Vec<f64> = (1..=16).map(|j| k * f64::from(j) * 0.25).collect();
            let v: Vec<f64> = s.iter().map(|x| hill_equation(*x, vmax, k, n)).collect();
            let (fit_vmax, fit_k, fit_n) = hill_fit(&s, &v).unwrap();
            assert!(close(fit_n, n, 0.02 * n), "the exponent {n} came back as {fit_n}");
            assert!(close(fit_vmax, vmax, 0.02 * vmax), "vmax came back as {fit_vmax}");
            assert!(close(fit_k, k, 0.02 * k), "k came back as {fit_k}");
        }
        assert!(hill_fit(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(hill_fit(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4]).is_err());
        assert!(hill_fit(&[1.0, 2.0, 3.0, 0.0], &[1.0; 4]).is_err());
    }

    #[test]
    fn the_three_inhibitions_move_the_constants_they_are_named_for() {
        // The mechanisms are distinguished by *which* constant moves, not by
        // how much the rate falls, which is why a single measurement can
        // never identify one and a substrate series can. Each is refitted
        // here and the apparent constants compared against the untouched
        // ones.
        let (vmax, km, ki) = (6.0f64, 2.0f64, 3.0f64);
        let i = 6.0;
        let alpha = 1.0 + i / ki;
        let s: Vec<f64> = (1..=14).map(|j| km * f64::from(j) * 0.4).collect();
        let refit = |kind: Inhibition| -> (f64, f64) {
            let v: Vec<f64> = s
                .iter()
                .map(|x| enzyme_inhibition(*x, i, vmax, km, ki, kind).unwrap())
                .collect();
            mm_fit(&s, &v).unwrap()
        };
        let (comp_vmax, comp_km) = refit(Inhibition::Competitive);
        assert!(close(comp_vmax, vmax, 1e-4 * vmax), "competitive moved vmax to {comp_vmax}");
        assert!(close(comp_km, km * alpha, 1e-4 * km * alpha), "competitive km is {comp_km}");

        let (non_vmax, non_km) = refit(Inhibition::NonCompetitive);
        assert!(close(non_vmax, vmax / alpha, 1e-4 * vmax), "non-competitive vmax is {non_vmax}");
        assert!(close(non_km, km, 1e-4 * km), "non-competitive moved km to {non_km}");

        let (un_vmax, un_km) = refit(Inhibition::Uncompetitive);
        assert!(close(un_vmax, vmax / alpha, 1e-4 * vmax), "uncompetitive vmax is {un_vmax}");
        assert!(close(un_km, km / alpha, 1e-4 * km), "uncompetitive km is {un_km}");
        // Its signature: vmax and km fall together, so the ratio is
        // untouched -- which is what makes the double-reciprocal lines
        // parallel.
        assert!(close(un_vmax / un_km, vmax / km, 1e-4 * vmax / km));

        // With no inhibitor all three collapse to the plain rate.
        for kind in [Inhibition::Competitive, Inhibition::Uncompetitive, Inhibition::NonCompetitive] {
            for x in &s {
                assert!(close(
                    enzyme_inhibition(*x, 0.0, vmax, km, ki, kind).unwrap(),
                    michaelis_menten(*x, vmax, km),
                    1e-12
                ));
            }
        }
        // And saturating substrate defeats a competitive inhibitor and
        // nothing else, which is the practical distinction.
        let huge = 1e9 * km;
        assert!(close(
            enzyme_inhibition(huge, i, vmax, km, ki, Inhibition::Competitive).unwrap(),
            vmax,
            1e-4 * vmax
        ));
        assert!(close(
            enzyme_inhibition(huge, i, vmax, km, ki, Inhibition::NonCompetitive).unwrap(),
            vmax / alpha,
            1e-4 * vmax
        ));
        assert!(enzyme_inhibition(1.0, 1.0, 1.0, 0.0, 1.0, Inhibition::Competitive).is_err());
        assert!(enzyme_inhibition(1.0, 1.0, 1.0, 1.0, 0.0, Inhibition::Competitive).is_err());
        assert!(enzyme_inhibition(-1.0, 1.0, 1.0, 1.0, 1.0, Inhibition::Competitive).is_err());
    }

    #[test]
    fn the_steady_state_approximation_holds_where_it_should_and_fails_where_it_should_not() {
        // The approximation needs the enzyme scarce beside the substrate.
        // Both regimes are run, because a check that only ever reports
        // "small" is not a check.
        let good = steady_state_approx_check(1e-4, 1.0, 1e5, 1e3, 50.0, 0.2).unwrap();
        // The run has to outlast the induction period for the check to mean
        // anything, and is refused rather than answered when it does not.
        assert!(steady_state_approx_check(1e-4, 1.0, 1e5, 1e3, 50.0, 1e-7).is_err());
        assert!(good < 0.05, "the approximation should hold here, but reads {good}");
        // Comparable enzyme and substrate: the complex is a large fraction
        // of the enzyme and the assumption is not available.
        let bad = steady_state_approx_check(1.0, 1.0, 1e5, 1e3, 50.0, 0.2).unwrap();
        assert!(bad > good, "the check reports {bad} where the approximation is worse than {good}");
        assert!(steady_state_approx_check(0.0, 1.0, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(steady_state_approx_check(1.0, 0.0, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(steady_state_approx_check(1.0, 1.0, 0.0, 1.0, 1.0, 1.0).is_err());
        assert!(steady_state_approx_check(1.0, 1.0, 1.0, 1.0, 1.0, 0.0).is_err());
    }

    // -----------------------------------------------------------------
    // Equilibrium
    // -----------------------------------------------------------------

    #[test]
    fn the_equilibrium_composition_satisfies_its_own_conditions() {
        // A + B <-> C with K = [C]/([A][B]), and the two element balances
        // A + C and B + C. The answer is a quadratic, so it can be checked
        // in closed form as well as by residual.
        let mut stoich = Matrix::zeros(3, 1);
        stoich.set(0, 0, -1.0);
        stoich.set(1, 0, -1.0);
        stoich.set(2, 0, 1.0);
        for &k in &[0.05f64, 1.0, 40.0, 5_000.0] {
            for &(a_total, b_total) in &[(1.0f64, 1.0f64), (0.4, 2.5), (3.0, 0.7)] {
                let totals = vec![
                    (vec![1.0, 0.0, 1.0], a_total),
                    (vec![0.0, 1.0, 1.0], b_total),
                ];
                let c = equilibrium_composition(&stoich, &[k], &totals).unwrap();
                assert!(c.iter().all(|v| *v > 0.0), "a concentration is not positive");
                // Mass action holds.
                assert!(
                    close(c[2] / (c[0] * c[1]), k, 1e-6 * k),
                    "at K = {k} the quotient is {}",
                    c[2] / (c[0] * c[1])
                );
                // And both balances.
                assert!(close(c[0] + c[2], a_total, 1e-9 * a_total));
                assert!(close(c[1] + c[2], b_total, 1e-9 * b_total));
                // Against the closed form: x = [C] solves
                // k (a - x)(b - x) = x.
                let (p, q, r) = (k, -(k * (a_total + b_total) + 1.0), k * a_total * b_total);
                let root = (-q - (q * q - 4.0 * p * r).sqrt()) / (2.0 * p);
                assert!(close(c[2], root, 1e-6 * root.max(1e-12)), "[C] is {} against {root}", c[2]);
            }
        }
        // A larger constant drives the reaction further, always.
        let totals = vec![(vec![1.0, 0.0, 1.0], 1.0), (vec![0.0, 1.0, 1.0], 1.0)];
        let mut previous = 0.0;
        for step in 0..10 {
            let k = 10f64.powi(step - 3);
            let c = equilibrium_composition(&stoich, &[k], &totals).unwrap();
            assert!(c[2] > previous, "a larger constant gave less product");
            previous = c[2];
        }
        assert!(equilibrium_composition(&stoich, &[1.0, 2.0], &totals).is_err());
        assert!(equilibrium_composition(&stoich, &[0.0], &totals).is_err());
        assert!(equilibrium_composition(&stoich, &[1.0], &totals[..1]).is_err());
        let bad = vec![(vec![1.0, 0.0], 1.0), (vec![0.0, 1.0, 1.0], 1.0)];
        assert!(equilibrium_composition(&stoich, &[1.0], &bad).is_err());
    }

    // -----------------------------------------------------------------
    // Deterministic integration
    // -----------------------------------------------------------------

    #[test]
    fn the_integrator_reproduces_first_order_decay_exactly() {
        // A -> B has the closed form c = c0 exp(-k t), so the integrator
        // can be checked rather than compared, across four decades of rate
        // constant.
        for &k in &[0.05f64, 1.0, 30.0, 500.0] {
            let reactions = [Reaction::new(&[(0, 1)], &[(1, 1)])];
            let stoich = stoichiometry_matrix(&reactions, 2).unwrap();
            let rates = |c: &[f64]| vec![k * c[0].max(0.0)];
            let trace = rate_equations(&stoich, &rates, &[1.0, 0.0], 3.0 / k, 1e-9).unwrap();
            for (t, c) in trace.iter().step_by(trace.len() / 20 + 1) {
                let expected = (-k * t).exp();
                assert!(
                    close(c[0], expected, 1e-5 + 1e-5 * expected),
                    "at k = {k}, t = {t} the concentration is {} against {expected}",
                    c[0]
                );
                // Mass is conserved: what leaves A arrives at B.
                assert!(close(c[0] + c[1], 1.0, 1e-6));
            }
            let (_, last) = trace.last().unwrap();
            assert!(close(last[0], (-3.0f64).exp(), 1e-5));
        }
    }

    #[test]
    fn the_integrator_survives_a_stiff_system_an_explicit_one_would_not() {
        // A fast pre-equilibrium beside a slow conversion: the rate
        // constants differ by six orders of magnitude, so an explicit
        // method would need a step set by the fastest long after it has
        // stopped mattering. The check is against the conserved total and
        // the known final state, both exact.
        let reactions = [
            Reaction::new(&[(0, 1)], &[(1, 1)]),
            Reaction::new(&[(1, 1)], &[(0, 1)]),
            Reaction::new(&[(1, 1)], &[(2, 1)]),
        ];
        let k = [1e6, 2e6, 1.0];
        let stoich = stoichiometry_matrix(&reactions, 3).unwrap();
        let rates = |c: &[f64]| mass_action_rates(&reactions, &k, c).unwrap();
        let trace = rate_equations(&stoich, &rates, &[1.0, 0.0, 0.0], 5.0, 1e-8).unwrap();
        for (_, c) in &trace {
            assert!(close(c[0] + c[1] + c[2], 1.0, 1e-6), "mass was not conserved");
            assert!(c.iter().all(|v| *v >= -1e-12), "a concentration went negative");
        }
        // The fast pair equilibrates at [B]/[A] = k1/(k2 + k3), a half to
        // seven figures, and holds it while C drains them both.
        //
        // Selected by *time*, not by step index: the controller spends most
        // of its steps resolving the transient, so the quarter-way step is
        // still inside it. The fast relaxation time is 1/(k1 + k2 + k3),
        // about 3e-7, so anything past thirty of those is well settled.
        let relaxation = 1.0 / (k[0] + k[1] + k[2]);
        let settled: Vec<&(f64, Vec<f64>)> =
            trace.iter().filter(|(t, _)| *t > 30.0 * relaxation).collect();
        assert!(settled.len() > 100, "only {} settled samples", settled.len());
        for (t, c) in &settled {
            if c[0] > 1e-6 {
                assert!(
                    close(c[1] / c[0], 0.5, 1e-3),
                    "at t = {t} the fast pair sits at {} rather than 0.5",
                    c[1] / c[0]
                );
            }
        }
        // The overall conversion is first order with an effective rate
        // k3 * fraction in B = 1 * (1/3).
        let (t_end, last) = trace.last().unwrap();
        let expected = 1.0 - (-t_end / 3.0).exp();
        assert!(
            close(last[2], expected, 0.01),
            "the product reached {} against {expected}",
            last[2]
        );
        // And the saving is the point. An explicit method is limited by the
        // fastest mode for the whole run: with a total decay rate of 3e6 it
        // would need a step below 2/3e6 to stay stable, or about seven
        // million steps to reach t = 5. The implicit one is stable at any
        // step and is limited only by accuracy.
        let explicit_steps = 5.0 / (2.0 / (k[0] + k[1] + k[2]));
        assert!(
            (trace.len() as f64) < explicit_steps / 100.0,
            "the implicit run took {} steps against an explicit method's {explicit_steps}",
            trace.len()
        );
    }

    #[test]
    fn the_integrator_rejects_what_it_cannot_integrate() {
        let reactions = [Reaction::new(&[(0, 1)], &[(1, 1)])];
        let stoich = stoichiometry_matrix(&reactions, 2).unwrap();
        let rates = |c: &[f64]| vec![c[0]];
        assert!(rate_equations(&stoich, &rates, &[1.0], 1.0, 1e-8).is_err());
        assert!(rate_equations(&stoich, &rates, &[1.0, 0.0], 0.0, 1e-8).is_err());
        assert!(rate_equations(&stoich, &rates, &[1.0, 0.0], 1.0, 0.0).is_err());
        assert!(rate_equations(&stoich, &rates, &[1.0, 0.0], 1.0, 1.5).is_err());
    }

    // -----------------------------------------------------------------
    // Stochastic simulation
    // -----------------------------------------------------------------

    #[test]
    fn the_gillespie_mean_matches_the_rate_equation_for_a_linear_network() {
        // The chemical master equation and the rate equations agree exactly
        // in the mean for a network whose propensities are linear in the
        // counts -- no large-number approximation is involved, because the
        // expectation of a linear function is the function of the
        // expectation. So this comparison is exact in the limit of many
        // runs, and any systematic gap is a defect rather than sampling
        // error.
        let reactions = [
            Reaction::new(&[(0, 1)], &[(1, 1)]),
            Reaction::new(&[(1, 1)], &[(2, 1)]),
        ];
        let k = [1.0, 0.4];
        let x0 = [200u64, 0, 0];
        let t_end = 3.0;
        let runs = 3_000;
        let mut rng = Rng::new(0x_C0DE_0001);
        let mut totals = [0.0f64; 3];
        for _ in 0..runs {
            let trace = gillespie_ssa(&reactions, &k, &x0, t_end, 100_000, &mut rng).unwrap();
            let (_, final_state) = trace.last().unwrap();
            for i in 0..3 {
                totals[i] += final_state[i] as f64;
            }
        }
        let means: Vec<f64> = totals.iter().map(|t| t / runs as f64).collect();

        let stoich = stoichiometry_matrix(&reactions, 3).unwrap();
        let rates = |c: &[f64]| mass_action_rates(&reactions, &k, c).unwrap();
        let ode = rate_equations(&stoich, &rates, &[200.0, 0.0, 0.0], t_end, 1e-10).unwrap();
        let (_, deterministic) = ode.last().unwrap();
        for i in 0..3 {
            let scale = deterministic[i].max(1.0);
            assert!(
                close(means[i], deterministic[i], 0.05 * scale),
                "species {i}: the mean of {runs} runs is {} against the ODE's {}",
                means[i],
                deterministic[i]
            );
        }
        // The trajectory is a jump process: every state is integral, and
        // time never runs backwards.
        let one = gillespie_ssa(&reactions, &k, &x0, t_end, 100_000, &mut rng).unwrap();
        for pair in one.windows(2) {
            assert!(pair[1].0 >= pair[0].0, "time went backwards");
            let changed: usize =
                (0..3).filter(|i| pair[1].1[*i] != pair[0].1[*i]).count();
            assert!(changed <= 2, "one event changed {changed} species");
        }
        // And the total molecule count is conserved by this network.
        for (_, x) in &one {
            assert_eq!(x[0] + x[1] + x[2], 200, "molecules were created or destroyed");
        }
    }

    #[test]
    fn gillespie_stops_at_an_absorbing_state_rather_than_spinning() {
        // Once nothing can react the algorithm has no next event to draw,
        // and must stop rather than divide by a zero total propensity.
        let reactions = [Reaction::new(&[(0, 1)], &[(1, 1)])];
        let mut rng = Rng::new(0x_C0DE_0002);
        let trace = gillespie_ssa(&reactions, &[5.0], &[3, 0], 1e6, 1_000, &mut rng).unwrap();
        let (_, last) = trace.last().unwrap();
        assert_eq!(*last, vec![0, 3], "the network did not run to completion");
        assert_eq!(trace.len(), 4, "three molecules should take three events");
        // The mean completion time is the sum of three exponential waits
        // with rates 15, 10 and 5 -- 1/15 + 1/10 + 1/5.
        let expected = 1.0 / 15.0 + 1.0 / 10.0 + 1.0 / 5.0;
        let mut total = 0.0;
        for _ in 0..4_000 {
            let run = gillespie_ssa(&reactions, &[5.0], &[3, 0], 1e6, 1_000, &mut rng).unwrap();
            total += run.last().unwrap().0;
        }
        let mean = total / 4_000.0;
        assert!(close(mean, expected, 0.05 * expected), "the mean wait is {mean}, not {expected}");
        assert!(gillespie_ssa(&[], &[], &[1], 1.0, 10, &mut rng).is_err());
        assert!(gillespie_ssa(&reactions, &[5.0, 1.0], &[3, 0], 1.0, 10, &mut rng).is_err());
        assert!(gillespie_ssa(&reactions, &[5.0], &[], 1.0, 10, &mut rng).is_err());
        assert!(gillespie_ssa(&reactions, &[5.0], &[3], 1.0, 10, &mut rng).is_err());
        assert!(gillespie_ssa(&reactions, &[5.0], &[3, 0], 0.0, 10, &mut rng).is_err());
        assert!(gillespie_ssa(&reactions, &[5.0], &[3, 0], 1.0, 0, &mut rng).is_err());
    }

    #[test]
    fn the_poisson_sampler_has_the_mean_and_variance_it_should() {
        // Poisson has mean equal to variance equal to lambda, which is a
        // strong pair of conditions -- a geometric or a normal draw would
        // match one and fail the other. Checked across both branches of the
        // implementation, since they are entirely different algorithms.
        let mut rng = Rng::new(0x_C0DE_0003);
        for &lambda in &[0.3f64, 3.0, 25.0, 40.0, 300.0, 2_000.0] {
            let n = 40_000;
            let draws: Vec<f64> = (0..n).map(|_| poisson(lambda, &mut rng) as f64).collect();
            let mean: f64 = draws.iter().sum::<f64>() / n as f64;
            let variance: f64 =
                draws.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
            let tolerance = 4.0 * (lambda / n as f64).sqrt();
            assert!(
                close(mean, lambda, tolerance.max(0.02)),
                "at lambda {lambda} the mean is {mean}"
            );
            assert!(
                close(variance / lambda, 1.0, 0.06),
                "at lambda {lambda} the variance ratio is {}",
                variance / lambda
            );
            assert!(draws.iter().all(|x| *x >= 0.0));
        }
        assert_eq!(poisson(0.0, &mut rng), 0);
        assert_eq!(poisson(-1.0, &mut rng), 0);
    }

    #[test]
    fn tau_leaping_converges_to_the_exact_algorithm_as_the_leap_shortens() {
        // The approximation is controlled: shortening the leap must move
        // the answer toward the exact one and keep it there. Checked as a
        // sequence rather than at one leap, since a single comparison
        // cannot distinguish a converging method from a lucky one.
        let reactions = [
            Reaction::new(&[(0, 1)], &[(1, 1)]),
            Reaction::new(&[(1, 1)], &[(0, 1)]),
        ];
        let k = [2.0, 1.0];
        let x0 = [1_000u64, 0];
        let t_end = 1.0;

        let mut rng = Rng::new(0x_C0DE_0004);
        let runs = 300;
        let exact_mean = {
            let mut total = 0.0;
            for _ in 0..runs {
                let trace = gillespie_ssa(&reactions, &k, &x0, t_end, 2_000_000, &mut rng).unwrap();
                total += trace.last().unwrap().1[1] as f64;
            }
            total / runs as f64
        };

        let mut errors = Vec::new();
        for shift in 0..4 {
            let tau = 0.2 / f64::from(1 << shift);
            let mut total = 0.0;
            for _ in 0..runs {
                let trace = tau_leaping(&reactions, &k, &x0, t_end, tau, &mut rng).unwrap();
                total += trace.last().unwrap().1[1] as f64;
            }
            errors.push((total / runs as f64 - exact_mean).abs());
        }
        assert!(
            errors[3] < errors[0],
            "shortening the leap did not help: {errors:?}"
        );
        assert!(
            errors[3] < 0.02 * exact_mean,
            "the shortest leap is still {} from the exact mean {exact_mean}",
            errors[3]
        );
        // Counts never go negative, which is what the leap rejection is
        // there to guarantee.
        let one = tau_leaping(&reactions, &k, &x0, t_end, 0.05, &mut rng).unwrap();
        for (_, x) in &one {
            assert_eq!(x[0] + x[1], 1_000, "molecules were created or destroyed");
        }
        assert!(tau_leaping(&reactions, &k, &x0, t_end, 0.0, &mut rng).is_err());
        assert!(tau_leaping(&reactions, &k, &x0, 0.0, 0.1, &mut rng).is_err());
        assert!(tau_leaping(&reactions, &k[..1], &x0, t_end, 0.1, &mut rng).is_err());
    }
}
