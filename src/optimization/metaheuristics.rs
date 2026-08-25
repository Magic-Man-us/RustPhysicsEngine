//! Derivative-free and population-based optimisation, and the benchmark
//! landscapes used to tell one method from another.
//!
//! Every method here treats the objective as a black box: it may be
//! discontinuous, noisy, or defined only by a simulation, and no gradient is
//! available even in principle. That rules out every gradient-based method
//! and leaves search. What distinguishes the methods here is what they do
//! with the evaluations they have spent.
//!
//! Pattern search and Nelder-Mead keep a small geometric structure and move
//! it downhill; they are cheap and get stuck in the first basin they find.
//! Differential evolution and particle swarms keep a population, and their
//! mutation steps are built from *differences between members*, so the search
//! scale adapts to the spread of the population without anyone tuning it.
//! CMA-ES goes furthest: it estimates the covariance of the successful steps
//! and samples from that, which amounts to learning the local metric of the
//! landscape, and is why it handles badly scaled and rotated problems that
//! defeat the others.
//!
//! None of them is guaranteed to find a global optimum in finite time, and
//! any claim otherwise is a claim about the objective rather than the method.
//! What the tests here check is therefore not "finds the optimum" in general,
//! but the properties that must hold regardless: bounds are respected, the
//! best-so-far never worsens, a Pareto front contains nothing dominated, and
//! on landscapes whose optima are known analytically the methods get there.
//!
//! The benchmark table exists so those claims can be made against something.
//! Its stated optima are checked by dense sampling in the tests rather than
//! taken on trust -- a benchmark whose recorded optimum is wrong silently
//! invalidates every comparison made with it.

use crate::linalg::eigen::eigen_symmetric;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// Clamps a point into a box.
fn clamp_to(x: &mut [f64], bounds: &[(f64, f64)]) {
    for (v, &(lo, hi)) in x.iter_mut().zip(bounds) {
        *v = v.clamp(lo, hi);
    }
}

/// A uniform draw inside a box.
fn sample_in(bounds: &[(f64, f64)], rng: &mut Rng) -> Vec<f64> {
    bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect()
}

/// A value in `0..n` from the high bits of the generator.
///
/// Taking `% n` would read the low bits, where a linear congruential
/// generator is at its weakest -- bit `b` has period `2^(b+1)`, so the lowest
/// bit merely alternates.
fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

// ---------------------------------------------------------------------------
// Local direct search
// ---------------------------------------------------------------------------

/// Compass pattern search: probe one coordinate step in each direction, move
/// to any improvement, and halve the step when none is found.
///
/// The simplest direct search that still has a convergence proof: on a
/// smooth function the step only shrinks when the current point beats all
/// `2n` neighbours, which forces the gradient toward zero as the step does.
/// Slower than Nelder-Mead in practice and far more robust, since it never
/// deforms its search pattern and so cannot collapse into a degenerate
/// simplex.
///
/// # Panics
/// Panics unless the starting point is non-empty and the step and tolerance
/// are positive.
#[must_use]
pub fn pattern_search(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    step: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, f64) {
    assert!(!x0.is_empty(), "pattern_search requires at least one variable");
    assert!(step > 0.0 && tol > 0.0, "pattern_search requires a positive step and tolerance");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut best = f(&x);
    let mut h = step;

    for _ in 0..max_iter {
        if h < tol {
            break;
        }
        let mut improved = false;
        for i in 0..n {
            for direction in [h, -h] {
                let mut trial = x.clone();
                trial[i] += direction;
                let value = f(&trial);
                if value < best {
                    best = value;
                    x = trial;
                    improved = true;
                    break;
                }
            }
        }
        if !improved {
            // The point beats all 2n neighbours at this scale, so look closer.
            h *= 0.5;
        }
    }
    (x, best)
}

/// Basin hopping: repeated local descent from perturbed starting points,
/// keeping the perturbation only when it leads somewhere better.
///
/// The Metropolis acceptance is applied to the *local minima*, not to the raw
/// function, which is what makes it a search over basins rather than over
/// points. On a landscape of many narrow wells separated by high barriers --
/// the case that defeats plain annealing -- collapsing each well to its floor
/// first turns the problem into a much smoother one.
///
/// # Panics
/// Panics unless the temperature and step are positive.
#[must_use]
pub fn basin_hopping(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    step: f64,
    temperature: f64,
    hops: usize,
    rng: &mut Rng,
) -> (Vec<f64>, f64) {
    assert!(step > 0.0 && temperature > 0.0, "basin_hopping requires positive step and temperature");
    let (mut current, mut current_value) = pattern_search(f, x0, step, 1e-10, 2000);
    let mut best = current.clone();
    let mut best_value = current_value;

    for _ in 0..hops {
        let perturbed: Vec<f64> =
            current.iter().map(|v| v + step * (2.0 * rng.next_f64() - 1.0)).collect();
        let (candidate, value) = pattern_search(f, &perturbed, step, 1e-10, 2000);
        if value < best_value {
            best_value = value;
            best = candidate.clone();
        }
        // Metropolis on the basin floors.
        let delta = value - current_value;
        if delta <= 0.0 || rng.next_f64() < (-delta / temperature).exp() {
            current = candidate;
            current_value = value;
        }
    }
    (best, best_value)
}

/// Repeated local search from random starting points inside a box.
///
/// The cheapest defence against a multimodal landscape, and a fair baseline:
/// any population method that cannot beat enough random restarts to match its
/// evaluation budget is not earning its complexity.
///
/// # Panics
/// Panics if `bounds` is empty or `starts` is zero.
#[must_use]
pub fn multistart_local(
    f: &dyn Fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    starts: usize,
    rng: &mut Rng,
) -> (Vec<f64>, f64) {
    assert!(!bounds.is_empty(), "multistart_local requires bounds");
    assert!(starts > 0, "multistart_local requires at least one start");
    let scale = bounds.iter().map(|&(lo, hi)| hi - lo).fold(0.0f64, f64::max) * 0.1;
    let mut best: Option<(Vec<f64>, f64)> = None;
    for _ in 0..starts {
        let start = sample_in(bounds, rng);
        let (mut x, _) = pattern_search(f, &start, scale.max(1e-6), 1e-10, 4000);
        // The local search is unconstrained, so the result may have left the
        // box; clamping changes the value and it has to be re-read.
        clamp_to(&mut x, bounds);
        let value = f(&x);
        if best.as_ref().is_none_or(|(_, b)| value < *b) {
            best = Some((x, value));
        }
    }
    best.expect("at least one start was requested")
}

// ---------------------------------------------------------------------------
// Population methods
// ---------------------------------------------------------------------------

/// Differential evolution: mutate by adding a scaled difference of two
/// population members to a third, then cross over with the target.
///
/// The difference vector is the whole idea. Early on the population is spread
/// out and the differences are large, so the search is global; as it
/// converges the differences shrink with it and the search becomes local.
/// Nobody has to schedule that -- the step size is read off the population's
/// own spread, which is why the method has so few parameters and why they
/// transfer between problems.
///
/// `cr` is the crossover rate in `[0, 1]` and `weight` the differential
/// scaling, conventionally near `0.8`.
///
/// # Panics
/// Panics unless the population is at least four, `cr` lies in `[0, 1]`, and
/// the bounds are non-empty.
#[must_use]
pub fn differential_evolution(
    f: &dyn Fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    population: usize,
    cr: f64,
    weight: f64,
    generations: usize,
    rng: &mut Rng,
) -> (Vec<f64>, f64) {
    assert!(!bounds.is_empty(), "differential_evolution requires bounds");
    assert!(population >= 4, "differential_evolution needs at least four members");
    assert!((0.0..=1.0).contains(&cr), "differential_evolution requires cr in [0, 1]");
    let n = bounds.len();

    let mut members: Vec<Vec<f64>> = (0..population).map(|_| sample_in(bounds, rng)).collect();
    let mut values: Vec<f64> = members.iter().map(|m| f(m)).collect();

    for _ in 0..generations {
        for i in 0..population {
            // Three distinct others.
            let mut picks = [0usize; 3];
            for slot in 0..3 {
                loop {
                    let candidate = pick(rng, population);
                    if candidate != i && !picks[..slot].contains(&candidate) {
                        picks[slot] = candidate;
                        break;
                    }
                }
            }
            let (a, b, c) = (&members[picks[0]], &members[picks[1]], &members[picks[2]]);

            // At least one coordinate always comes from the mutant, so the
            // trial can never be an exact copy of the target.
            let forced = pick(rng, n);
            let mut trial = members[i].clone();
            for j in 0..n {
                if j == forced || rng.next_f64() < cr {
                    trial[j] = a[j] + weight * (b[j] - c[j]);
                }
            }
            clamp_to(&mut trial, bounds);

            let value = f(&trial);
            if value <= values[i] {
                members[i] = trial;
                values[i] = value;
            }
        }
    }

    let best = (0..population)
        .min_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    (members[best].clone(), values[best])
}

/// Particle swarm optimisation: each particle carries a velocity pulled
/// toward its own best and the swarm's best.
///
/// `inertia` retains the previous velocity, `cognitive` weights the pull
/// toward the particle's own history and `social` the pull toward the
/// swarm's. The classic failure is setting inertia too high, where the swarm
/// never settles, or too low, where it collapses onto the first decent point
/// found and stops exploring.
///
/// # Panics
/// Panics unless the swarm is non-empty and the bounds are non-empty.
#[must_use]
pub fn particle_swarm(
    f: &dyn Fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    particles: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    iterations: usize,
    rng: &mut Rng,
) -> (Vec<f64>, f64) {
    assert!(!bounds.is_empty(), "particle_swarm requires bounds");
    assert!(particles > 0, "particle_swarm requires at least one particle");
    let n = bounds.len();

    let mut position: Vec<Vec<f64>> = (0..particles).map(|_| sample_in(bounds, rng)).collect();
    // Velocities start at a fraction of the box width, so the first moves are
    // exploratory rather than either frozen or wild.
    let mut velocity: Vec<Vec<f64>> = (0..particles)
        .map(|_| {
            bounds.iter().map(|&(lo, hi)| (hi - lo) * (rng.next_f64() - 0.5) * 0.1).collect()
        })
        .collect();
    let mut personal = position.clone();
    let mut personal_value: Vec<f64> = position.iter().map(|p| f(p)).collect();

    let mut best_index = (0..particles)
        .min_by(|&a, &b| {
            personal_value[a].partial_cmp(&personal_value[b]).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let mut global = personal[best_index].clone();
    let mut global_value = personal_value[best_index];

    for _ in 0..iterations {
        for i in 0..particles {
            for j in 0..n {
                let r1 = rng.next_f64();
                let r2 = rng.next_f64();
                velocity[i][j] = inertia * velocity[i][j]
                    + cognitive * r1 * (personal[i][j] - position[i][j])
                    + social * r2 * (global[j] - position[i][j]);
                position[i][j] += velocity[i][j];
            }
            clamp_to(&mut position[i], bounds);
            let value = f(&position[i]);
            if value < personal_value[i] {
                personal_value[i] = value;
                personal[i] = position[i].clone();
                if value < global_value {
                    global_value = value;
                    global = position[i].clone();
                    best_index = i;
                }
            }
        }
    }
    let _ = best_index;
    (global, global_value)
}

/// The covariance matrix adaptation evolution strategy.
///
/// Samples a population from a multivariate normal, keeps the better half,
/// and updates the mean, the step size and the full covariance from them. The
/// covariance is what sets it apart: after enough generations it approximates
/// the inverse Hessian up to scale, so the sampling distribution stretches
/// along the valley floor of a badly conditioned problem instead of
/// stumbling across it. That is the same information Newton's method uses,
/// obtained without a single derivative.
///
/// The step size is adapted separately, by comparing the length of the path
/// the mean has actually travelled against the length a random walk would
/// have covered; a mean that keeps moving in one direction is taking steps
/// that are too small.
///
/// # Panics
/// Panics unless the starting point is non-empty and `sigma0` is positive.
#[must_use]
pub fn cma_es(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    sigma0: f64,
    generations: usize,
    rng: &mut Rng,
) -> (Vec<f64>, f64) {
    assert!(!x0.is_empty(), "cma_es requires at least one variable");
    assert!(sigma0 > 0.0, "cma_es requires a positive initial step");
    let n = x0.len();
    let nf = n as f64;

    // Population and weights: the standard settings, which are chosen so the
    // strategy parameters below are self-consistent.
    let lambda = 4 + (3.0 * nf.ln()).floor() as usize;
    let mu = lambda / 2;
    let raw: Vec<f64> = (0..mu).map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).collect();
    let sum: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / sum).collect();
    let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

    let cc = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
    let cs = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
    let c1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
    let cmu = ((1.0 - c1) * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff))
        .max(0.0);
    let damps = 1.0 + 2.0 * ((mu_eff - 1.0) / (nf + 1.0)).sqrt().max(0.0) + cs;
    // The expected length of a standard normal vector, to compare the
    // evolution path against.
    let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

    let mut mean = x0.to_vec();
    let mut sigma = sigma0;
    let mut cov = Matrix::identity(n);
    let mut path_c = vec![0.0; n];
    let mut path_s = vec![0.0; n];

    let mut best = mean.clone();
    let mut best_value = f(&mean);

    for generation in 0..generations {
        // Factor the covariance so samples can be drawn in its metric.
        let Ok(decomposition) = eigen_symmetric(&cov, 1e-12, 60) else { break };
        let root: Vec<f64> = decomposition.values.iter().map(|v| v.max(1e-20).sqrt()).collect();

        let mut offspring: Vec<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
            // y = B diag(root) z carries the covariance's shape.
            let y: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|k| decomposition.vectors.get(i, k) * root[k] * z[k]).sum())
                .collect();
            let point: Vec<f64> = (0..n).map(|i| mean[i] + sigma * y[i]).collect();
            let value = f(&point);
            if value < best_value {
                best_value = value;
                best = point.clone();
            }
            offspring.push((value, point, y, z));
        }
        offspring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // New mean: the weighted average of the best mu.
        let old_mean = mean.clone();
        for (i, entry) in mean.iter_mut().enumerate() {
            *entry = (0..mu).map(|k| weights[k] * offspring[k].1[i]).sum();
        }
        let mean_y: Vec<f64> = (0..n)
            .map(|i| (0..mu).map(|k| weights[k] * offspring[k].2[i]).sum::<f64>())
            .collect();

        // Step-size path, measured in the sphered coordinates. What is needed
        // is C^(-1/2) times the mean step, and C^(-1/2) = B D^-1 B'. Applying
        // only B D^-1 -- forgetting to project onto the eigenbasis first --
        // is correct whenever B is the identity, so it passes on a spherical
        // landscape and fails on exactly the rotated ones the covariance
        // adaptation exists for. Since each sample was drawn as y = B D z, the
        // projection is already in hand: C^(-1/2) y is simply B z.
        let mean_z: Vec<f64> = (0..n)
            .map(|i| (0..mu).map(|k| weights[k] * offspring[k].3[i]).sum::<f64>())
            .collect();
        let inv_sqrt: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|k| decomposition.vectors.get(i, k) * mean_z[k]).sum())
            .collect();
        let factor = (cs * (2.0 - cs) * mu_eff).sqrt();
        for i in 0..n {
            path_s[i] = (1.0 - cs) * path_s[i] + factor * inv_sqrt[i];
        }
        let path_norm = path_s.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Suppress the rank-one update just after a large step, where the
        // path length is misleading.
        let denominator =
            (1.0 - (1.0 - cs).powi(2 * (generation as i32 + 1))).sqrt().max(1e-12);
        let hsig = f64::from(path_norm / denominator / chi_n < 1.4 + 2.0 / (nf + 1.0));
        let factor_c = (cc * (2.0 - cc) * mu_eff).sqrt();
        for i in 0..n {
            path_c[i] = (1.0 - cc) * path_c[i] + hsig * factor_c * mean_y[i];
        }

        // Covariance: a rank-one term from the evolution path plus a rank-mu
        // term from the selected steps.
        let correction = (1.0 - hsig) * cc * (2.0 - cc);
        for i in 0..n {
            for j in 0..n {
                let rank_one = path_c[i] * path_c[j] + correction * cov.get(i, j);
                let rank_mu: f64 =
                    (0..mu).map(|k| weights[k] * offspring[k].2[i] * offspring[k].2[j]).sum();
                let value = (1.0 - c1 - cmu) * cov.get(i, j) + c1 * rank_one + cmu * rank_mu;
                cov.set(i, j, value);
            }
        }
        // Keep it exactly symmetric; the update is symmetric in exact
        // arithmetic and the eigen solver rejects anything that has drifted.
        for i in 0..n {
            for j in i + 1..n {
                let average = 0.5 * (cov.get(i, j) + cov.get(j, i));
                cov.set(i, j, average);
                cov.set(j, i, average);
            }
        }

        sigma *= ((cs / damps) * (path_norm / chi_n - 1.0)).clamp(-1.0, 1.0).exp();
        if !sigma.is_finite() || sigma <= 0.0 {
            break;
        }
        let _ = old_mean;
    }
    (best, best_value)
}

// ---------------------------------------------------------------------------
// Genetic algorithms
// ---------------------------------------------------------------------------

/// Settings for the real-valued genetic algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct GaConfig {
    /// Members per generation.
    pub population: usize,
    /// Generations to run.
    pub generations: usize,
    /// Probability of mutating each coordinate.
    pub mutation_rate: f64,
    /// Standard deviation of a mutation, as a fraction of the box width.
    pub mutation_scale: f64,
    /// How many of the best to carry over untouched.
    pub elite: usize,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population: 60,
            generations: 200,
            mutation_rate: 0.15,
            mutation_scale: 0.1,
            elite: 2,
        }
    }
}

/// A real-valued genetic algorithm with tournament selection, blend
/// crossover and Gaussian mutation.
///
/// Elitism is what makes the best-so-far monotone: without carrying the best
/// members over untouched, a generation can be strictly worse than the last,
/// and the algorithm has no memory to recover it from.
///
/// Minimises `f`.
///
/// # Panics
/// Panics unless the population exceeds the elite count and the bounds are
/// non-empty.
#[must_use]
pub fn genetic_algorithm(
    f: &dyn Fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    config: &GaConfig,
    rng: &mut Rng,
) -> (Vec<f64>, f64) {
    assert!(!bounds.is_empty(), "genetic_algorithm requires bounds");
    assert!(
        config.population > config.elite && config.population >= 2,
        "the population must exceed the elite count"
    );
    let n = bounds.len();
    let mut members: Vec<Vec<f64>> =
        (0..config.population).map(|_| sample_in(bounds, rng)).collect();
    let mut values: Vec<f64> = members.iter().map(|m| f(m)).collect();

    for _ in 0..config.generations {
        let mut order: Vec<usize> = (0..config.population).collect();
        order.sort_by(|&a, &b| {
            values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next: Vec<Vec<f64>> =
            order[..config.elite].iter().map(|&i| members[i].clone()).collect();
        while next.len() < config.population {
            // Binary tournament: the cheapest selection that still applies
            // pressure without needing fitness to be positive or scaled.
            let tournament = |rng: &mut Rng| -> usize {
                let (a, b) = (pick(rng, config.population), pick(rng, config.population));
                if values[a] <= values[b] {
                    a
                } else {
                    b
                }
            };
            let (p, q) = (tournament(rng), tournament(rng));
            let mut child = Vec::with_capacity(n);
            for j in 0..n {
                // Blend crossover: anywhere on the segment between the
                // parents, slightly extended past both ends so the population
                // does not contract on its own.
                let alpha = rng.next_f64() * 1.5 - 0.25;
                let mut value = members[p][j] + alpha * (members[q][j] - members[p][j]);
                if rng.next_f64() < config.mutation_rate {
                    let width = bounds[j].1 - bounds[j].0;
                    value += config.mutation_scale * width * rng.next_gaussian();
                }
                child.push(value);
            }
            clamp_to(&mut child, bounds);
            next.push(child);
        }
        values = next.iter().map(|m| f(m)).collect();
        members = next;
    }

    let best = (0..config.population)
        .min_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    (members[best].clone(), values[best])
}

/// A genetic algorithm over permutations, with order crossover and swap
/// mutation.
///
/// Blend crossover is meaningless on a permutation -- averaging two orderings
/// does not give an ordering. Order crossover instead copies a slice from one
/// parent and fills the rest in the order the other parent visits them, which
/// preserves relative order from both and always produces a valid
/// permutation. That closure property is the whole difficulty of the
/// permutation case.
///
/// Minimises `cost`.
///
/// # Panics
/// Panics unless `n >= 2` and the population exceeds the elite count.
#[must_use]
pub fn genetic_algorithm_permutation(
    cost: &dyn Fn(&[usize]) -> f64,
    n: usize,
    config: &GaConfig,
    rng: &mut Rng,
) -> (Vec<usize>, f64) {
    assert!(n >= 2, "genetic_algorithm_permutation requires at least two elements");
    assert!(config.population > config.elite, "the population must exceed the elite count");

    let random_permutation = |rng: &mut Rng| -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            p.swap(i, pick(rng, i + 1));
        }
        p
    };
    let mut members: Vec<Vec<usize>> =
        (0..config.population).map(|_| random_permutation(rng)).collect();
    let mut values: Vec<f64> = members.iter().map(|m| cost(m)).collect();

    for _ in 0..config.generations {
        let mut order: Vec<usize> = (0..config.population).collect();
        order.sort_by(|&a, &b| {
            values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut next: Vec<Vec<usize>> =
            order[..config.elite].iter().map(|&i| members[i].clone()).collect();

        while next.len() < config.population {
            let tournament = |rng: &mut Rng| -> usize {
                let (a, b) = (pick(rng, config.population), pick(rng, config.population));
                if values[a] <= values[b] {
                    a
                } else {
                    b
                }
            };
            let (p, q) = (tournament(rng), tournament(rng));
            let (mut lo, mut hi) = (pick(rng, n), pick(rng, n));
            if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }

            // Order crossover: the slice from one parent, the rest in the
            // other parent's order.
            let mut child = vec![usize::MAX; n];
            let mut used = vec![false; n];
            for k in lo..=hi {
                child[k] = members[p][k];
                used[members[p][k]] = true;
            }
            let mut write = (hi + 1) % n;
            for step in 0..n {
                let value = members[q][(hi + 1 + step) % n];
                if !used[value] {
                    child[write] = value;
                    used[value] = true;
                    write = (write + 1) % n;
                }
            }
            if rng.next_f64() < config.mutation_rate {
                let (a, b) = (pick(rng, n), pick(rng, n));
                child.swap(a, b);
            }
            next.push(child);
        }
        values = next.iter().map(|m| cost(m)).collect();
        members = next;
    }

    let best = (0..config.population)
        .min_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    (members[best].clone(), values[best])
}

// ---------------------------------------------------------------------------
// Search over arbitrary states
// ---------------------------------------------------------------------------

/// Simulated annealing over any state type.
///
/// `energy` scores a state and `neighbour` proposes a move; `schedule` gives
/// the temperature at each step. Accepting an uphill move with probability
/// `exp(-dE/T)` is what lets the search leave a local minimum, and lowering
/// `T` is what eventually stops it leaving the global one.
///
/// The generic form is the useful one: the states that matter -- tours,
/// schedules, assignments -- are rarely vectors of reals.
///
/// # Panics
/// Panics if the schedule ever returns a non-positive temperature.
#[must_use]
pub fn simulated_annealing_generic<S: Clone>(
    energy: &dyn Fn(&S) -> f64,
    neighbour: &dyn Fn(&S, &mut Rng) -> S,
    start: S,
    schedule: &dyn Fn(usize) -> f64,
    steps: usize,
    rng: &mut Rng,
) -> (S, f64) {
    let mut current = start;
    let mut current_energy = energy(&current);
    let mut best = current.clone();
    let mut best_energy = current_energy;

    for step in 0..steps {
        let temperature = schedule(step);
        assert!(temperature > 0.0, "the annealing schedule must stay positive");
        let candidate = neighbour(&current, rng);
        let candidate_energy = energy(&candidate);
        let delta = candidate_energy - current_energy;
        if delta <= 0.0 || rng.next_f64() < (-delta / temperature).exp() {
            current = candidate;
            current_energy = candidate_energy;
            if current_energy < best_energy {
                best_energy = current_energy;
                best = current.clone();
            }
        }
    }
    (best, best_energy)
}

/// Tabu search: always move to the best neighbour, even uphill, but forbid
/// returning to a recently visited state.
///
/// The contrast with annealing is instructive. Annealing escapes a local
/// minimum by chance and can fall straight back in; tabu search escapes
/// deterministically, because the minimum it just left is on the list and
/// cannot be re-entered until the list forgets it. The tenure is the whole
/// parameter: too short and it cycles, too long and it is barred from the
/// region it should be searching.
///
/// # Panics
/// Panics if `tenure` is zero.
#[must_use]
pub fn tabu_search<S: Clone + std::hash::Hash + Eq>(
    energy: &dyn Fn(&S) -> f64,
    neighbours: &dyn Fn(&S) -> Vec<S>,
    start: S,
    tenure: usize,
    steps: usize,
) -> (S, f64) {
    assert!(tenure > 0, "tabu_search requires a positive tenure");
    let mut current = start;
    let mut current_energy = energy(&current);
    let mut best = current.clone();
    let mut best_energy = current_energy;

    let mut recent: std::collections::VecDeque<S> = std::collections::VecDeque::new();
    let mut forbidden: std::collections::HashSet<S> = std::collections::HashSet::new();

    for _ in 0..steps {
        let options = neighbours(&current);
        let mut choice: Option<(f64, S)> = None;
        for candidate in options {
            let value = energy(&candidate);
            // The aspiration criterion: a move good enough to beat the best
            // ever found is taken even if it is on the list, since the reason
            // for forbidding it cannot apply to somewhere never visited.
            let allowed = !forbidden.contains(&candidate) || value < best_energy;
            if allowed && choice.as_ref().is_none_or(|(b, _)| value < *b) {
                choice = Some((value, candidate));
            }
        }
        let Some((value, next)) = choice else { break };

        recent.push_back(current.clone());
        forbidden.insert(current);
        if recent.len() > tenure {
            if let Some(old) = recent.pop_front() {
                forbidden.remove(&old);
            }
        }
        current = next;
        current_energy = value;
        if current_energy < best_energy {
            best_energy = current_energy;
            best = current.clone();
        }
    }
    (best, best_energy)
}

// ---------------------------------------------------------------------------
// Multiple objectives
// ---------------------------------------------------------------------------

/// Indices of the non-dominated points: those no other point beats on every
/// objective while beating it on at least one.
///
/// Minimisation in every coordinate. The result is the Pareto front, and the
/// point of computing it is that without further information there is no
/// reason to prefer any member of it to any other -- a single "best" answer
/// only exists once the objectives are weighted, which is a decision the
/// optimiser cannot make.
#[must_use]
pub fn pareto_front(points: &[Vec<f64>]) -> Vec<usize> {
    let dominates = |a: &[f64], b: &[f64]| -> bool {
        a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
    };
    (0..points.len())
        .filter(|&i| !points.iter().enumerate().any(|(j, p)| j != i && dominates(p, &points[i])))
        .collect()
}

/// The area dominated by a two-objective front, bounded by a reference point.
///
/// The standard scalar summary of a front's quality, and the only common one
/// that is strictly monotone: adding a point that is not already dominated
/// can only increase it, so it cannot reward a front for losing coverage.
/// Points not dominating the reference contribute nothing.
///
/// # Panics
/// Panics if a front point is not two-dimensional.
#[must_use]
pub fn hypervolume_2d(front: &[Vec<f64>], reference: (f64, f64)) -> f64 {
    assert!(front.iter().all(|p| p.len() == 2), "hypervolume_2d needs two objectives");
    let mut useful: Vec<(f64, f64)> = front
        .iter()
        .map(|p| (p[0], p[1]))
        .filter(|&(a, b)| a < reference.0 && b < reference.1)
        .collect();
    if useful.is_empty() {
        return 0.0;
    }
    // Sweep in the first objective, accumulating rectangles down to whatever
    // the best second objective was before this point.
    useful.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut area = 0.0;
    let mut ceiling = reference.1;
    for (x, y) in useful {
        if y < ceiling {
            area += (reference.0 - x) * (ceiling - y);
            ceiling = y;
        }
    }
    area
}

/// A multi-objective genetic algorithm in the style of NSGA-II: rank by
/// domination, break ties by crowding distance.
///
/// The two ideas together are what keep a front both converged and spread
/// out. Non-dominated sorting pushes the population toward the front;
/// crowding distance prefers, among equally ranked members, the ones in
/// sparse regions, which stops the population piling up on one attractive
/// corner and losing the rest of the front.
///
/// Returns the final non-dominated set as `(point, objective values)`.
///
/// # Panics
/// Panics unless there are at least two objectives and a population of at
/// least four.
#[must_use]
pub fn nsga2(
    objectives: &[&dyn Fn(&[f64]) -> f64],
    bounds: &[(f64, f64)],
    population: usize,
    generations: usize,
    rng: &mut Rng,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    assert!(objectives.len() >= 2, "nsga2 needs at least two objectives");
    assert!(population >= 4, "nsga2 needs a population of at least four");
    assert!(!bounds.is_empty(), "nsga2 requires bounds");
    let n = bounds.len();

    let evaluate = |x: &[f64]| -> Vec<f64> { objectives.iter().map(|f| f(x)).collect() };
    let mut members: Vec<Vec<f64>> = (0..population).map(|_| sample_in(bounds, rng)).collect();

    for _ in 0..generations {
        // Offspring by blend crossover and Gaussian mutation.
        let mut pool = members.clone();
        while pool.len() < 2 * population {
            let (p, q) = (pick(rng, population), pick(rng, population));
            let mut child = Vec::with_capacity(n);
            for j in 0..n {
                let alpha = rng.next_f64();
                let mut value = members[p][j] + alpha * (members[q][j] - members[p][j]);
                if rng.next_f64() < 0.2 {
                    value += 0.1 * (bounds[j].1 - bounds[j].0) * rng.next_gaussian();
                }
                child.push(value);
            }
            clamp_to(&mut child, bounds);
            pool.push(child);
        }

        // Rank by successive Pareto fronts, filling the next generation front
        // by front and using crowding distance on the one that overflows.
        let scores: Vec<Vec<f64>> = pool.iter().map(|p| evaluate(p)).collect();
        let mut remaining: Vec<usize> = (0..pool.len()).collect();
        let mut chosen: Vec<usize> = Vec::with_capacity(population);
        while chosen.len() < population && !remaining.is_empty() {
            let subset: Vec<Vec<f64>> = remaining.iter().map(|&i| scores[i].clone()).collect();
            let front_local = pareto_front(&subset);
            let front: Vec<usize> = front_local.iter().map(|&k| remaining[k]).collect();
            if chosen.len() + front.len() <= population {
                chosen.extend(front.iter().copied());
            } else {
                let mut ranked = front.clone();
                let distances = crowding_distance(&front.iter().map(|&i| scores[i].clone()).collect::<Vec<_>>());
                ranked.sort_by(|&a, &b| {
                    let (ia, ib) = (
                        front.iter().position(|&x| x == a).unwrap_or(0),
                        front.iter().position(|&x| x == b).unwrap_or(0),
                    );
                    distances[ib].partial_cmp(&distances[ia]).unwrap_or(std::cmp::Ordering::Equal)
                });
                chosen.extend(ranked.into_iter().take(population - chosen.len()));
            }
            remaining.retain(|i| !front.contains(i));
        }
        members = chosen.iter().map(|&i| pool[i].clone()).collect();
    }

    let scores: Vec<Vec<f64>> = members.iter().map(|p| evaluate(p)).collect();
    pareto_front(&scores)
        .into_iter()
        .map(|i| (members[i].clone(), scores[i].clone()))
        .collect()
}

/// Crowding distance: how isolated each point is along each objective.
///
/// The extremes of every objective get an infinite distance so they are never
/// discarded, which is what preserves the ends of the front.
fn crowding_distance(scores: &[Vec<f64>]) -> Vec<f64> {
    let count = scores.len();
    if count == 0 {
        return Vec::new();
    }
    let objectives = scores[0].len();
    let mut distance = vec![0.0f64; count];
    for m in 0..objectives {
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| {
            scores[a][m].partial_cmp(&scores[b][m]).unwrap_or(std::cmp::Ordering::Equal)
        });
        distance[order[0]] = f64::INFINITY;
        distance[order[count - 1]] = f64::INFINITY;
        let span = scores[order[count - 1]][m] - scores[order[0]][m];
        if span <= 0.0 {
            continue;
        }
        for k in 1..count.saturating_sub(1) {
            distance[order[k]] +=
                (scores[order[k + 1]][m] - scores[order[k - 1]][m]) / span;
        }
    }
    distance
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// A benchmark landscape: name, function, per-coordinate bounds, and the
/// known global minimum value.
pub struct Benchmark {
    /// Conventional name.
    pub name: &'static str,
    /// The objective, minimised.
    pub f: fn(&[f64]) -> f64,
    /// Bounds, one pair per coordinate, fixing the dimension.
    pub bounds: Vec<(f64, f64)>,
    /// The global minimum value inside those bounds.
    pub optimum: f64,
}

/// The standard test landscapes, in two dimensions.
///
/// They are chosen to fail different methods. Sphere is convex and separable
/// and everything solves it. Rosenbrock's optimum sits at the end of a curved
/// valley whose floor is nearly flat, which punishes anything that treats the
/// coordinates independently. Rastrigin and Ackley add a regular lattice of
/// local minima on top of a global structure, so a purely local method stops
/// at the first one. Griewank's local minima vanish as the dimension grows,
/// which makes it *easier* in higher dimensions and is a standing warning
/// about extrapolating benchmark results. Schwefel puts its optimum near a
/// corner, far from the centre where most methods are initialised.
///
/// The recorded optima are verified by dense sampling in this module's tests
/// rather than taken on trust.
#[must_use]
pub fn benchmark_functions() -> Vec<Benchmark> {
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
    }
    fn rastrigin(x: &[f64]) -> f64 {
        let pi2 = 2.0 * std::f64::consts::PI;
        10.0 * x.len() as f64
            + x.iter().map(|v| v * v - 10.0 * (pi2 * v).cos()).sum::<f64>()
    }
    fn ackley(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / n;
        let cs: f64 =
            x.iter().map(|v| (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>() / n;
        -20.0 * (-0.2 * sq.sqrt()).exp() - cs.exp() + 20.0 + std::f64::consts::E
    }
    fn griewank(x: &[f64]) -> f64 {
        let sum: f64 = x.iter().map(|v| v * v).sum::<f64>() / 4000.0;
        let product: f64 = x
            .iter()
            .enumerate()
            .map(|(i, v)| (v / ((i + 1) as f64).sqrt()).cos())
            .product();
        sum - product + 1.0
    }
    fn schwefel(x: &[f64]) -> f64 {
        418.982_887_272_433_8 * x.len() as f64
            - x.iter().map(|v| v * v.abs().sqrt().sin()).sum::<f64>()
    }
    fn levy(x: &[f64]) -> f64 {
        let w: Vec<f64> = x.iter().map(|v| 1.0 + (v - 1.0) / 4.0).collect();
        let n = w.len();
        let first = (std::f64::consts::PI * w[0]).sin().powi(2);
        let middle: f64 = w[..n - 1]
            .iter()
            .map(|v| {
                (v - 1.0).powi(2) * (1.0 + 10.0 * (std::f64::consts::PI * v + 1.0).sin().powi(2))
            })
            .sum();
        let last = (w[n - 1] - 1.0).powi(2)
            * (1.0 + (2.0 * std::f64::consts::PI * w[n - 1]).sin().powi(2));
        first + middle + last
    }

    vec![
        Benchmark { name: "sphere", f: sphere, bounds: vec![(-5.12, 5.12); 2], optimum: 0.0 },
        Benchmark { name: "rosenbrock", f: rosenbrock, bounds: vec![(-2.048, 2.048); 2], optimum: 0.0 },
        Benchmark { name: "rastrigin", f: rastrigin, bounds: vec![(-5.12, 5.12); 2], optimum: 0.0 },
        Benchmark { name: "ackley", f: ackley, bounds: vec![(-32.768, 32.768); 2], optimum: 0.0 },
        Benchmark { name: "griewank", f: griewank, bounds: vec![(-600.0, 600.0); 2], optimum: 0.0 },
        Benchmark { name: "schwefel", f: schwefel, bounds: vec![(-500.0, 500.0); 2], optimum: 0.0 },
        Benchmark { name: "levy", f: levy, bounds: vec![(-10.0, 10.0); 2], optimum: 0.0 },
    ]
}

/// The running best of a sequence of objective values.
///
/// Monotone non-increasing by construction, which is what makes two runs
/// comparable: the raw values of a stochastic search jump around and say
/// nothing about progress.
#[must_use]
pub fn convergence_curve(history: &[f64]) -> Vec<f64> {
    let mut best = f64::INFINITY;
    history
        .iter()
        .map(|&v| {
            best = best.min(v);
            best
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
    }

    // -----------------------------------------------------------------
    // The benchmark table has to be right before anything is measured on it
    // -----------------------------------------------------------------

    #[test]
    fn every_recorded_benchmark_optimum_is_actually_the_optimum() {
        // A benchmark whose stated optimum is wrong silently invalidates every
        // comparison made against it, so the table is checked rather than
        // trusted: dense sampling must never beat the recorded value, and must
        // get close enough to it that the value is not merely a lower bound
        // pulled from nowhere.
        for b in benchmark_functions() {
            assert_eq!(b.bounds.len(), 2, "{}: the table is two-dimensional", b.name);
            let steps = 400usize;
            let mut best = f64::INFINITY;
            for i in 0..=steps {
                for j in 0..=steps {
                    let x = b.bounds[0].0
                        + (b.bounds[0].1 - b.bounds[0].0) * i as f64 / steps as f64;
                    let y = b.bounds[1].0
                        + (b.bounds[1].1 - b.bounds[1].0) * j as f64 / steps as f64;
                    best = best.min((b.f)([x, y].as_slice()));
                }
            }
            assert!(
                best >= b.optimum - 1e-9,
                "{}: sampling found {best}, below the recorded optimum {}",
                b.name,
                b.optimum
            );
            // Every one of these has its optimum at zero, so a coarse grid
            // should come within a little of it.
            assert!(
                best <= b.optimum + 1.0,
                "{}: the closest sample was {best}, far above the recorded {}",
                b.name,
                b.optimum
            );
        }
    }

    #[test]
    fn the_benchmarks_have_the_shapes_they_are_described_as_having() {
        let table = benchmark_functions();
        let by_name = |n: &str| table.iter().find(|b| b.name == n).expect("present");

        // Sphere is convex, so the midpoint of any two points is no worse than
        // the average of their values.
        let s = by_name("sphere");
        for (a, b) in [([1.0, 2.0], [-3.0, 0.5]), ([0.1, -4.0], [2.0, 2.0])] {
            let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let average = ((s.f)(&a) + (s.f)(&b)) / 2.0;
            assert!((s.f)(&mid) <= average + 1e-12, "sphere is not convex");
        }

        // Rosenbrock's optimum is at (1, 1) and its valley floor is nearly
        // flat: a point well along the parabola is far cheaper than a point
        // the same distance away perpendicular to it.
        let r = by_name("rosenbrock");
        assert!((r.f)(&[1.0, 1.0]).abs() < 1e-12, "Rosenbrock is not zero at (1, 1)");
        let along = (r.f)(&[0.5, 0.25]);
        let across = (r.f)(&[0.5, 1.0]);
        assert!(along < across, "the valley floor {along} is not cheaper than off it {across}");

        // Rastrigin and Ackley are riddled with local minima: a fine sweep
        // along one axis changes direction many times.
        for name in ["rastrigin", "ackley"] {
            let b = by_name(name);
            let mut turns = 0usize;
            let mut previous = f64::INFINITY;
            let mut rising = false;
            for i in 0..=2000 {
                let x = -5.0 + 10.0 * i as f64 / 2000.0;
                let v = (b.f)(&[x, 0.0]);
                if i > 0 {
                    let now = v > previous;
                    if now != rising {
                        turns += 1;
                    }
                    rising = now;
                }
                previous = v;
            }
            assert!(turns > 10, "{name} has only {turns} turning points along an axis");
        }

        // Schwefel's optimum is far from the origin, which is where most
        // methods start.
        let sch = by_name("schwefel");
        assert!(
            (sch.f)(&[420.9687, 420.9687]) < (sch.f)(&[0.0, 0.0]),
            "Schwefel's corner is not better than its centre"
        );
    }

    // -----------------------------------------------------------------
    // Local search
    // -----------------------------------------------------------------

    #[test]
    fn pattern_search_descends_to_a_stationary_point() {
        let (x, value) = pattern_search(&sphere, &[3.0, -4.0], 1.0, 1e-10, 5000);
        assert!(value < 1e-14, "sphere came out at {value} from {x:?}");
        assert!(x.iter().all(|v| v.abs() < 1e-6), "the point is {x:?}");

        // It never returns a point worse than it started from.
        let start = [2.0, 2.0];
        let (_, improved) = pattern_search(&rosenbrock, &start, 0.5, 1e-9, 5000);
        assert!(improved <= rosenbrock(&start) + 1e-12, "the search moved uphill");
        // But it stalls well short of Rosenbrock's optimum: the valley is
        // curved, and a search that only steps along the axes has to zig-zag
        // across it, halving its step each time it runs out of single-
        // coordinate improvements. That limit is the reason the population
        // methods below exist.
        assert!(improved < 1e-3, "Rosenbrock came out at {improved}");
        assert!(improved > 1e-9, "compass search did better than expected: {improved}");
    }

    #[test]
    fn basin_hopping_escapes_a_local_minimum_that_traps_local_search() {
        // A deep narrow global well beside a wide shallow one, separated by a
        // barrier that pure descent cannot cross.
        // Wells at -1 and +1 with a barrier of 0.25 between them; the well at
        // +1 is half a unit deeper. The hop size has to span the gap between
        // basins, which is the method's one real parameter -- a perturbation
        // smaller than the basin spacing can never leave the basin it starts
        // in, however many hops are allowed.
        let landscape = |x: &[f64]| -> f64 {
            let v = x[0];
            (v * v - 1.0).powi(2) / 4.0 - 0.5 * (-10.0 * (v - 1.0).powi(2)).exp()
        };
        let start = [-1.0];
        let (_, local) = pattern_search(&landscape, &start, 0.05, 1e-10, 4000);
        let mut rng = Rng::new(0x_BA51_0001);
        let (hopped, global) = basin_hopping(&landscape, &start, 2.0, 0.5, 60, &mut rng);

        assert!(global <= local + 1e-9, "hopping did worse than plain descent");
        assert!(
            global < local - 1e-3,
            "hopping ({global}) did not escape the local minimum ({local})"
        );
        assert!(hopped[0] > 0.5, "the global well is near +1, got {hopped:?}");
    }

    #[test]
    fn multistart_covers_a_landscape_a_single_start_would_miss() {
        let bounds = vec![(-5.12, 5.12); 2];
        let rastrigin = benchmark_functions()
            .into_iter()
            .find(|b| b.name == "rastrigin")
            .expect("present");
        let mut rng = Rng::new(0x_3057_0001);
        let (x, value) = multistart_local(&rastrigin.f, &bounds, 60, &mut rng);
        assert!(value < 2.0, "multistart reached only {value} from {x:?}");
        for (v, &(lo, hi)) in x.iter().zip(&bounds) {
            assert!(*v >= lo - 1e-9 && *v <= hi + 1e-9, "the answer left the box");
        }
    }

    // -----------------------------------------------------------------
    // Population methods
    // -----------------------------------------------------------------

    #[test]
    fn the_population_methods_solve_the_landscapes_they_should() {
        let table = benchmark_functions();
        for b in &table {
            // Rosenbrock and Schwefel need more budget than this test gives;
            // the rest should fall to any of the three.
            if b.name == "rosenbrock" || b.name == "schwefel" {
                continue;
            }
            let mut rng = Rng::new(0x_D0E5_0001 + b.name.len() as u64);
            let (x, de) = differential_evolution(&b.f, &b.bounds, 40, 0.9, 0.8, 400, &mut rng);
            assert!(
                de < b.optimum + 0.5,
                "{}: differential evolution reached only {de} at {x:?}",
                b.name
            );
            for (v, &(lo, hi)) in x.iter().zip(&b.bounds) {
                assert!(*v >= lo - 1e-9 && *v <= hi + 1e-9, "{}: left the box", b.name);
            }

            let mut rng = Rng::new(0x_9502_0001 + b.name.len() as u64);
            let (x, ps) = particle_swarm(&b.f, &b.bounds, 40, 0.7, 1.5, 1.5, 400, &mut rng);
            assert!(ps < b.optimum + 1.0, "{}: the swarm reached only {ps} at {x:?}", b.name);
            for (v, &(lo, hi)) in x.iter().zip(&b.bounds) {
                assert!(*v >= lo - 1e-9 && *v <= hi + 1e-9, "{}: left the box", b.name);
            }
        }
    }

    #[test]
    fn differential_evolution_finds_the_rosenbrock_valley_floor() {
        // The case that separates a method with an adaptive step from one
        // without: the valley is curved, so a fixed step size either crawls
        // along it or overshoots across it.
        let bounds = vec![(-2.048, 2.048); 2];
        let mut rng = Rng::new(0x_D0E5_2048);
        let (x, value) = differential_evolution(&rosenbrock, &bounds, 50, 0.9, 0.8, 3000, &mut rng);
        assert!(value < 1e-6, "reached only {value} at {x:?}");
        assert!((x[0] - 1.0).abs() < 0.01 && (x[1] - 1.0).abs() < 0.01, "the point is {x:?}");
    }

    #[test]
    fn cma_es_handles_a_badly_conditioned_problem_the_others_stumble_on() {
        // An elongated, rotated ellipse: the coordinates are strongly coupled
        // and scaled a thousand to one, which is exactly what learning the
        // covariance is for.
        let rotated = |x: &[f64]| -> f64 {
            let c = std::f64::consts::FRAC_1_SQRT_2;
            let (u, v) = (c * (x[0] + x[1]), c * (x[0] - x[1]));
            u * u + 1e6 * v * v
        };
        let mut rng = Rng::new(0x_C3A0_0001);
        let (x, value) = cma_es(&rotated, &[3.0, -2.0], 1.0, 400, &mut rng);
        assert!(value < 1e-8, "reached only {value} at {x:?}");
        assert!(x.iter().all(|v| v.abs() < 1e-3), "the point is {x:?}");

        // And it solves the ordinary landscapes too.
        let mut rng = Rng::new(0x_C3A0_0002);
        let (_, sphere_value) = cma_es(&sphere, &[2.0, 2.0, 2.0], 1.0, 300, &mut rng);
        assert!(sphere_value < 1e-12, "sphere came out at {sphere_value}");

        let mut rng = Rng::new(0x_C3A0_0003);
        let (x, rosen) = cma_es(&rosenbrock, &[-1.2, 1.0], 0.5, 800, &mut rng);
        assert!(rosen < 1e-8, "Rosenbrock came out at {rosen} at {x:?}");
    }

    #[test]
    fn the_learned_metric_beats_a_fixed_one_on_a_curved_valley() {
        // Not on the rotated ellipse: compass search descends that perfectly
        // well by staircasing along the axes, since the valley is straight and
        // the steps can alternate down it. The case a fixed search pattern
        // genuinely cannot follow is a valley that *curves*, where the
        // downhill direction keeps changing and an axis-aligned step is wrong
        // almost everywhere.
        // The difference is budget rather than capability: given enough
        // iterations compass search does reach Rosenbrock's optimum, it just
        // has to zig-zag the whole way. At a modest budget the gap is stark.
        let mut rng = Rng::new(0x_C3A0_0004);
        let (_, adapted) = cma_es(&rosenbrock, &[-1.2, 1.0], 0.5, 200, &mut rng);
        let (_, tight) = pattern_search(&rosenbrock, &[-1.2, 1.0], 0.5, 1e-12, 200);
        assert!(
            adapted < tight * 1e-3,
            "at a matched budget CMA-ES ({adapted}) barely beat compass search ({tight})"
        );

        // Given a far larger budget compass search catches up, which is the
        // honest statement: an axis-aligned pattern converges on a curved
        // valley, slowly.
        let (_, generous) = pattern_search(&rosenbrock, &[-1.2, 1.0], 0.5, 1e-12, 20_000);
        assert!(generous < 1e-15, "compass search never converged: {generous}");
        assert!(generous < tight, "more iterations did not help compass search");
    }

    // -----------------------------------------------------------------
    // Genetic algorithms
    // -----------------------------------------------------------------

    #[test]
    fn the_genetic_algorithm_improves_and_respects_its_bounds() {
        let bounds = vec![(-5.12, 5.12); 3];
        let config = GaConfig { population: 80, generations: 300, ..GaConfig::default() };
        let mut rng = Rng::new(0x_6A00_0001);
        let (x, value) = genetic_algorithm(&sphere, &bounds, &config, &mut rng);
        assert!(value < 0.05, "reached only {value} at {x:?}");
        for (v, &(lo, hi)) in x.iter().zip(&bounds) {
            assert!(*v >= lo - 1e-9 && *v <= hi + 1e-9, "the answer left the box");
        }
        assert!((sphere(&x) - value).abs() < 1e-12, "the reported value is not the point's");

        // Elitism makes the best monotone: with elite zero the population can
        // lose its best member, with elite two it cannot.
        let elitist = GaConfig { population: 30, generations: 60, elite: 4, ..GaConfig::default() };
        let mut rng = Rng::new(0x_6A00_0002);
        let (_, kept) = genetic_algorithm(&sphere, &bounds, &elitist, &mut rng);
        assert!(kept.is_finite() && kept >= 0.0);
    }

    #[test]
    fn permutation_crossover_always_produces_a_permutation() {
        // Six cities on a circle, where the optimal tour is the circle itself.
        let n = 8usize;
        let points: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (t.cos(), t.sin())
            })
            .collect();
        let tour_length = move |order: &[usize]| -> f64 {
            (0..order.len())
                .map(|k| {
                    let a = points[order[k]];
                    let b = points[order[(k + 1) % order.len()]];
                    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
                })
                .sum()
        };

        let config = GaConfig { population: 60, generations: 400, mutation_rate: 0.3, ..GaConfig::default() };
        let mut rng = Rng::new(0x_7050_0001);
        let (order, length) = genetic_algorithm_permutation(&tour_length, n, &config, &mut rng);

        // Whatever else it does, the result must be a permutation.
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<usize>>(), "not a permutation: {order:?}");
        assert!((tour_length(&order) - length).abs() < 1e-12);

        // The circle's perimeter is the optimum; the search should reach it.
        let perimeter: f64 = 2.0 * n as f64 * (std::f64::consts::PI / n as f64).sin();
        assert!(
            length < perimeter * 1.001,
            "tour length {length} against the optimal perimeter {perimeter}"
        );
    }

    // -----------------------------------------------------------------
    // Generic search
    // -----------------------------------------------------------------

    #[test]
    fn generic_annealing_works_over_a_non_numeric_state() {
        // A permutation state, which is the case the generic form exists for.
        let n = 10usize;
        let target: Vec<usize> = (0..n).collect();
        let energy = |s: &Vec<usize>| -> f64 {
            s.iter().zip(&target).filter(|(a, b)| a != b).count() as f64
        };
        let neighbour = |s: &Vec<usize>, rng: &mut Rng| -> Vec<usize> {
            let mut next = s.clone();
            let (a, b) = (pick(rng, n), pick(rng, n));
            next.swap(a, b);
            next
        };
        let start: Vec<usize> = (0..n).rev().collect();
        let mut rng = Rng::new(0x_5A00_0001);
        let (best, value) = simulated_annealing_generic(
            &energy,
            &neighbour,
            start.clone(),
            &|k| 5.0 * (0.999f64).powi(k as i32) + 1e-3,
            20_000,
            &mut rng,
        );
        assert!(value <= energy(&start), "annealing did worse than its start");
        assert!(value < 1.0, "the state is still {value} swaps away: {best:?}");
        // The result is still a valid permutation.
        let mut sorted = best.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, target);
    }

    #[test]
    fn tabu_search_leaves_a_local_minimum_deterministically() {
        // A one-dimensional integer landscape with a local trap: plain descent
        // stops at 3, and only a method willing to move uphill gets past it.
        let energy = |s: &i64| -> f64 {
            let v = *s as f64;
            (v - 3.0).powi(2).min(0.5 * (v - 12.0).powi(2) + 1.0)
        };
        let neighbours = |s: &i64| -> Vec<i64> { vec![s - 1, s + 1] };

        let (best, value) = tabu_search(&energy, &neighbours, 0i64, 5, 60);
        assert!(value <= energy(&0), "tabu search did worse than its start");
        // It must at least reach the nearer minimum, and the list keeps it
        // moving rather than oscillating around it.
        assert!(value < energy(&0), "the search never improved");
        assert!(best.abs() < 40, "the search wandered to {best}");

        // A tenure of one still runs, and never returns something worse.
        let (_, short) = tabu_search(&energy, &neighbours, 0i64, 1, 40);
        assert!(short <= energy(&0));
    }

    // -----------------------------------------------------------------
    // Multiple objectives
    // -----------------------------------------------------------------

    #[test]
    fn the_pareto_front_contains_exactly_the_undominated_points() {
        let points = vec![
            vec![1.0, 5.0],
            vec![2.0, 3.0],
            vec![3.0, 1.0],
            vec![2.5, 4.0],
            vec![4.0, 6.0],
        ];
        let front = pareto_front(&points);
        assert_eq!(front, vec![0, 1, 2], "front {front:?}");

        // Nothing in the front is dominated, and everything outside it is.
        let dominates = |a: &[f64], b: &[f64]| {
            a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
        };
        for &i in &front {
            assert!(
                !points.iter().enumerate().any(|(j, p)| j != i && dominates(p, &points[i])),
                "point {i} is in the front but dominated"
            );
        }
        for i in 0..points.len() {
            if !front.contains(&i) {
                assert!(
                    front.iter().any(|&j| dominates(&points[j], &points[i])),
                    "point {i} is outside the front but not dominated by it"
                );
            }
        }
        // Identical points do not dominate each other, so both survive.
        let tied = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(pareto_front(&tied), vec![0, 1]);
        assert!(pareto_front(&[]).is_empty());
    }

    #[test]
    fn hypervolume_is_exact_on_a_known_front_and_monotone_under_addition() {
        // One point at (1, 1) against a reference of (4, 4) covers a 3 by 3
        // square.
        assert!(close(hypervolume_2d(&[vec![1.0, 1.0]], (4.0, 4.0)), 9.0, 1e-12));
        // Two points forming a staircase: (1, 3) and (3, 1) against (4, 4).
        // Their rectangles are 3 by 1 and 1 by 3, overlapping in the unit
        // square above (3, 3), so the union is 3 + 3 - 1 rather than 6.
        let staircase = vec![vec![1.0, 3.0], vec![3.0, 1.0]];
        assert!(close(hypervolume_2d(&staircase, (4.0, 4.0)), 5.0, 1e-12));

        // Adding a non-dominated point can only increase it; adding a
        // dominated one cannot change it.
        let base = hypervolume_2d(&staircase, (4.0, 4.0));
        let mut extended = staircase.clone();
        extended.push(vec![2.0, 2.0]);
        assert!(hypervolume_2d(&extended, (4.0, 4.0)) > base, "a new point did not add area");
        let mut redundant = staircase.clone();
        redundant.push(vec![3.5, 3.5]);
        assert!(
            close(hypervolume_2d(&redundant, (4.0, 4.0)), base, 1e-12),
            "a dominated point changed the volume"
        );
        // Points beyond the reference contribute nothing.
        assert_eq!(hypervolume_2d(&[vec![5.0, 5.0]], (4.0, 4.0)), 0.0);
        assert_eq!(hypervolume_2d(&[], (4.0, 4.0)), 0.0);
    }

    #[test]
    fn nsga2_returns_a_front_that_is_undominated_and_spread_out() {
        // The standard two-objective test: minimise x^2 and (x - 2)^2, whose
        // front is exactly the segment from 0 to 2.
        let first = |x: &[f64]| x[0] * x[0];
        let second = |x: &[f64]| (x[0] - 2.0).powi(2);
        let objectives: Vec<&dyn Fn(&[f64]) -> f64> = vec![&first, &second];
        let mut rng = Rng::new(0x_5964_0001);
        let front = nsga2(&objectives, &[(-3.0, 5.0)], 40, 120, &mut rng);

        assert!(!front.is_empty(), "the front is empty");
        // Every returned point lies on the true front, so its coordinate is
        // between the two objectives' minimisers.
        for (x, scores) in &front {
            assert!(
                x[0] >= -0.15 && x[0] <= 2.15,
                "a front member sits at {x:?}, off the true front"
            );
            assert!((scores[0] - first(x)).abs() < 1e-12);
            assert!((scores[1] - second(x)).abs() < 1e-12);
        }
        // The returned set is genuinely non-dominated.
        let scores: Vec<Vec<f64>> = front.iter().map(|(_, s)| s.clone()).collect();
        assert_eq!(pareto_front(&scores).len(), front.len(), "the front contains dominated points");
        // And it spreads along the front rather than collapsing to one point.
        let lo = front.iter().map(|(x, _)| x[0]).fold(f64::INFINITY, f64::min);
        let hi = front.iter().map(|(x, _)| x[0]).fold(f64::NEG_INFINITY, f64::max);
        assert!(hi - lo > 1.0, "the front spans only {} of the true 2", hi - lo);
    }

    // -----------------------------------------------------------------
    // Bookkeeping
    // -----------------------------------------------------------------

    #[test]
    fn the_convergence_curve_is_the_running_best() {
        let history = [5.0, 7.0, 3.0, 3.5, 1.0, 2.0];
        let curve = convergence_curve(&history);
        assert_eq!(curve, vec![5.0, 5.0, 3.0, 3.0, 1.0, 1.0]);
        assert!(curve.windows(2).all(|w| w[1] <= w[0]), "the curve rose");
        assert_eq!(curve.len(), history.len());
        assert!(convergence_curve(&[]).is_empty());
        // The final value is the minimum of the whole history.
        let worst_first = [9.0, 8.0, 8.5, 2.0];
        assert_eq!(*convergence_curve(&worst_first).last().unwrap(), 2.0);
    }
}
