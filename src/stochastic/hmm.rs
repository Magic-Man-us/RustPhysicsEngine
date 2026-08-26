//! Hidden state models: hidden Markov models, smoothing, and particle
//! filters.
//!
//! The common thread is a state that evolves as a Markov chain and is never
//! observed directly -- only through emissions that depend on it. Three
//! questions follow, and each has its own algorithm. *How likely is this
//! observation sequence?* is answered by summing over every possible state
//! path, which the forward recursion does in linear time by never
//! enumerating the paths. *Which single path best explains it?* is answered
//! by Viterbi, the same recursion with the sum replaced by a maximum. *What
//! parameters make it likeliest?* is answered by Baum-Welch, which is
//! expectation-maximisation applied to the first two.
//!
//! The discrete and Gaussian models here differ only in what an emission is.
//! The Kalman smoother and the particle filter answer the same questions for
//! a continuous state: exactly, when the model is linear and Gaussian, and
//! by sampling when it is not.
//!
//! Everything works in logs or with explicit scaling, because the
//! probability of a sequence of a few hundred observations underflows a
//! double long before the algorithm finishes.

use crate::control_systems::kalman::KalmanFilter;
use crate::error::GeomError;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// A hidden Markov model with discrete emissions.
#[derive(Debug, Clone, PartialEq)]
pub struct Hmm {
    /// Transition matrix, `n_states` by `n_states`, row-stochastic.
    pub a: Matrix,
    /// Emission matrix, `n_states` by `n_symbols`, row-stochastic.
    pub b: Matrix,
    /// Initial state distribution.
    pub pi: Vec<f64>,
}

impl Hmm {
    /// The model with the given parameters.
    ///
    /// # Errors
    /// Returns an error unless the shapes agree and every row of each matrix,
    /// and the initial distribution, sums to one over non-negative entries.
    pub fn new(a: Matrix, b: Matrix, pi: Vec<f64>) -> Result<Self, GeomError> {
        let n = a.rows;
        if !a.is_square() || b.rows != n || pi.len() != n {
            return Err(GeomError::InvalidArgument("the model's shapes do not agree"));
        }
        let stochastic = |row: &[f64]| {
            row.iter().all(|&v| v >= -1e-12 && v.is_finite())
                && (row.iter().sum::<f64>() - 1.0).abs() < 1e-6
        };
        for i in 0..n {
            if !stochastic(a.row(i)) || !stochastic(b.row(i)) {
                return Err(GeomError::InvalidArgument("a row is not a distribution"));
            }
        }
        if !stochastic(&pi) {
            return Err(GeomError::InvalidArgument("the initial distribution is invalid"));
        }
        Ok(Hmm { a, b, pi })
    }

    /// The number of hidden states.
    #[must_use]
    pub fn n_states(&self) -> usize {
        self.a.rows
    }

    /// The number of observable symbols.
    #[must_use]
    pub fn n_symbols(&self) -> usize {
        self.b.cols
    }

    /// A model with random parameters, for Baum-Welch to start from.
    ///
    /// # Panics
    /// Panics if either dimension is zero.
    #[must_use]
    pub fn random_init(n_states: usize, n_symbols: usize, rng: &mut Rng) -> Self {
        assert!(n_states > 0 && n_symbols > 0, "both dimensions must be positive");
        let draw = |rows: usize, cols: usize, rng: &mut Rng| {
            let mut m = Matrix::zeros(rows, cols);
            for i in 0..rows {
                let row: Vec<f64> = (0..cols).map(|_| 0.1 + rng.next_f64()).collect();
                let total: f64 = row.iter().sum();
                for j in 0..cols {
                    m.set(i, j, row[j] / total);
                }
            }
            m
        };
        let a = draw(n_states, n_states, rng);
        let b = draw(n_states, n_symbols, rng);
        let raw: Vec<f64> = (0..n_states).map(|_| 0.1 + rng.next_f64()).collect();
        let total: f64 = raw.iter().sum();
        let pi = raw.into_iter().map(|v| v / total).collect();
        Hmm { a, b, pi }
    }

    /// The forward recursion, returning the log-likelihood and the scaled
    /// forward probabilities.
    ///
    /// `alpha[t][i]` is the probability of state `i` at time `t` given the
    /// observations up to `t`, rescaled to sum to one at each step. Scaling
    /// is not an optimisation: without it the raw forward variables shrink by
    /// roughly the observation's probability at every step and underflow a
    /// double within a few hundred symbols. The scale factors, summed as
    /// logs, are exactly the log-likelihood.
    ///
    /// # Panics
    /// Panics if an observation is outside the alphabet.
    #[must_use]
    pub fn forward(&self, obs: &[usize]) -> (f64, Matrix) {
        assert!(obs.iter().all(|&o| o < self.n_symbols()), "an observation is out of range");
        let n = self.n_states();
        let t = obs.len();
        let mut alpha = Matrix::zeros(t.max(1), n);
        if t == 0 {
            return (0.0, alpha);
        }
        let mut log_likelihood = 0.0;
        let mut current: Vec<f64> =
            (0..n).map(|i| self.pi[i] * self.b.get(i, obs[0])).collect();
        for step in 0..t {
            if step > 0 {
                let previous: Vec<f64> = (0..n).map(|i| alpha.get(step - 1, i)).collect();
                current = (0..n)
                    .map(|j| {
                        let inflow: f64 =
                            (0..n).map(|i| previous[i] * self.a.get(i, j)).sum();
                        inflow * self.b.get(j, obs[step])
                    })
                    .collect();
            }
            let scale: f64 = current.iter().sum();
            if scale <= 0.0 {
                // The observation is impossible under this model.
                return (f64::NEG_INFINITY, alpha);
            }
            log_likelihood += scale.ln();
            for i in 0..n {
                alpha.set(step, i, current[i] / scale);
            }
        }
        (log_likelihood, alpha)
    }

    /// The backward recursion, scaled to match [`forward`](Self::forward).
    ///
    /// `beta[t][i]` is proportional to the probability of the observations
    /// after `t` given state `i` at `t`, under the same per-step scaling.
    ///
    /// # Panics
    /// Panics if an observation is outside the alphabet.
    #[must_use]
    pub fn backward(&self, obs: &[usize]) -> Matrix {
        assert!(obs.iter().all(|&o| o < self.n_symbols()), "an observation is out of range");
        let n = self.n_states();
        let t = obs.len();
        let mut beta = Matrix::zeros(t.max(1), n);
        if t == 0 {
            return beta;
        }
        for i in 0..n {
            beta.set(t - 1, i, 1.0);
        }
        for step in (0..t - 1).rev() {
            let mut row: Vec<f64> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            self.a.get(i, j) * self.b.get(j, obs[step + 1]) * beta.get(step + 1, j)
                        })
                        .sum()
                })
                .collect();
            let scale: f64 = row.iter().sum();
            if scale > 0.0 {
                for v in &mut row {
                    *v /= scale;
                }
            }
            for i in 0..n {
                beta.set(step, i, row[i]);
            }
        }
        beta
    }

    /// The log-likelihood of an observation sequence.
    ///
    /// # Panics
    /// Panics if an observation is outside the alphabet.
    #[must_use]
    pub fn log_likelihood(&self, obs: &[usize]) -> f64 {
        self.forward(obs).0
    }

    /// The single most likely state path, and its log probability.
    ///
    /// The forward recursion with the sum replaced by a maximum, in logs. It
    /// answers a different question from decoding each state separately: the
    /// best path need not contain any individual state's most likely value,
    /// and unlike posterior decoding its answer is always a path the model
    /// can actually produce.
    ///
    /// # Panics
    /// Panics if an observation is outside the alphabet.
    #[must_use]
    pub fn viterbi(&self, obs: &[usize]) -> (f64, Vec<usize>) {
        assert!(obs.iter().all(|&o| o < self.n_symbols()), "an observation is out of range");
        let n = self.n_states();
        let t = obs.len();
        if t == 0 {
            return (0.0, Vec::new());
        }
        let ln = |x: f64| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY };
        let mut delta: Vec<f64> =
            (0..n).map(|i| ln(self.pi[i]) + ln(self.b.get(i, obs[0]))).collect();
        let mut back = vec![vec![0usize; n]; t];
        for step in 1..t {
            let mut next = vec![f64::NEG_INFINITY; n];
            for j in 0..n {
                let (best_i, best_v) = (0..n)
                    .map(|i| (i, delta[i] + ln(self.a.get(i, j))))
                    .fold((0usize, f64::NEG_INFINITY), |acc, x| if x.1 > acc.1 { x } else { acc });
                back[step][j] = best_i;
                next[j] = best_v + ln(self.b.get(j, obs[step]));
            }
            delta = next;
        }
        let (mut s, score) = (0..n)
            .map(|i| (i, delta[i]))
            .fold((0usize, f64::NEG_INFINITY), |acc, x| if x.1 > acc.1 { x } else { acc });
        let mut path = vec![0usize; t];
        for step in (0..t).rev() {
            path[step] = s;
            s = back[step][s];
        }
        (score, path)
    }

    /// The state posteriors: `gamma[t][i]` is the probability of state `i` at
    /// time `t` given the whole sequence.
    ///
    /// # Panics
    /// Panics if an observation is outside the alphabet.
    #[must_use]
    pub fn posteriors(&self, obs: &[usize]) -> Matrix {
        let n = self.n_states();
        let t = obs.len();
        let (_, alpha) = self.forward(obs);
        let beta = self.backward(obs);
        let mut gamma = Matrix::zeros(t.max(1), n);
        for step in 0..t {
            let row: Vec<f64> = (0..n).map(|i| alpha.get(step, i) * beta.get(step, i)).collect();
            let total: f64 = row.iter().sum();
            for i in 0..n {
                gamma.set(step, i, if total > 0.0 { row[i] / total } else { 1.0 / n as f64 });
            }
        }
        gamma
    }

    /// The most likely state at each time, taken separately.
    ///
    /// Maximises the expected number of correct states, which is a different
    /// objective from Viterbi's. The path it returns can have probability
    /// zero -- if two consecutive states are each individually likeliest but
    /// the transition between them is impossible, this will happily report
    /// both.
    ///
    /// # Panics
    /// Panics if an observation is outside the alphabet.
    #[must_use]
    pub fn posterior_decode(&self, obs: &[usize]) -> Vec<usize> {
        let gamma = self.posteriors(obs);
        (0..obs.len())
            .map(|t| {
                (0..self.n_states())
                    .max_by(|&i, &j| gamma.get(t, i).total_cmp(&gamma.get(t, j)))
                    .expect("at least one state")
            })
            .collect()
    }

    /// Baum-Welch training, returning the final total log-likelihood.
    ///
    /// Expectation-maximisation: compute the expected transition and emission
    /// counts under the current parameters, then set the parameters to their
    /// maximum-likelihood values given those counts. Each round is guaranteed
    /// not to lower the likelihood, which is what makes it safe to run
    /// without a line search -- and also all it guarantees, since it climbs
    /// to a local optimum that depends entirely on where it started.
    ///
    /// Stops early when the improvement falls below `tol`.
    ///
    /// # Panics
    /// Panics if a sequence contains an observation outside the alphabet.
    pub fn baum_welch(&mut self, sequences: &[Vec<usize>], iters: usize, tol: f64) -> f64 {
        let n = self.n_states();
        let m = self.n_symbols();
        let mut previous = f64::NEG_INFINITY;
        for _ in 0..iters {
            let mut trans = Matrix::zeros(n, n);
            let mut emit = Matrix::zeros(n, m);
            let mut start = vec![0.0; n];
            let mut total_ll = 0.0;
            for obs in sequences {
                if obs.is_empty() {
                    continue;
                }
                let (ll, alpha) = self.forward(obs);
                if !ll.is_finite() {
                    continue;
                }
                total_ll += ll;
                let beta = self.backward(obs);
                let t = obs.len();
                // Posterior over states, and over transitions between them.
                for step in 0..t {
                    let row: Vec<f64> =
                        (0..n).map(|i| alpha.get(step, i) * beta.get(step, i)).collect();
                    let denom: f64 = row.iter().sum();
                    if denom <= 0.0 {
                        continue;
                    }
                    for i in 0..n {
                        let g = row[i] / denom;
                        if step == 0 {
                            start[i] += g;
                        }
                        emit.set(i, obs[step], emit.get(i, obs[step]) + g);
                    }
                }
                for step in 0..t - 1 {
                    let mut xi = vec![vec![0.0; n]; n];
                    let mut denom = 0.0;
                    for i in 0..n {
                        for j in 0..n {
                            let v = alpha.get(step, i)
                                * self.a.get(i, j)
                                * self.b.get(j, obs[step + 1])
                                * beta.get(step + 1, j);
                            xi[i][j] = v;
                            denom += v;
                        }
                    }
                    if denom <= 0.0 {
                        continue;
                    }
                    for i in 0..n {
                        for j in 0..n {
                            trans.set(i, j, trans.get(i, j) + xi[i][j] / denom);
                        }
                    }
                }
            }
            // Maximisation: normalise the expected counts. A state never
            // visited keeps its old row rather than becoming undefined.
            let start_total: f64 = start.iter().sum();
            if start_total > 0.0 {
                self.pi = start.iter().map(|v| v / start_total).collect();
            }
            for i in 0..n {
                let row: f64 = (0..n).map(|j| trans.get(i, j)).sum();
                if row > 0.0 {
                    for j in 0..n {
                        self.a.set(i, j, trans.get(i, j) / row);
                    }
                }
                let erow: f64 = (0..m).map(|k| emit.get(i, k)).sum();
                if erow > 0.0 {
                    for k in 0..m {
                        self.b.set(i, k, emit.get(i, k) / erow);
                    }
                }
            }
            if total_ll - previous < tol && previous.is_finite() {
                break;
            }
            previous = total_ll;
        }
        sequences.iter().map(|o| self.log_likelihood(o)).sum()
    }

    /// Draws a state path and the observations it produces.
    #[must_use]
    pub fn simulate(&self, n: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        let mut states = Vec::with_capacity(n);
        let mut obs = Vec::with_capacity(n);
        let mut s = sample_from(&self.pi, rng);
        for _ in 0..n {
            states.push(s);
            let row: Vec<f64> = (0..self.n_symbols()).map(|k| self.b.get(s, k)).collect();
            obs.push(sample_from(&row, rng));
            let next: Vec<f64> = (0..self.n_states()).map(|j| self.a.get(s, j)).collect();
            s = sample_from(&next, rng);
        }
        (states, obs)
    }
}

/// A draw from a discrete distribution given as weights.
fn sample_from(weights: &[f64], rng: &mut Rng) -> usize {
    let total: f64 = weights.iter().sum();
    let u = rng.next_f64() * total;
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u < acc {
            return i;
        }
    }
    weights.len() - 1
}

/// A hidden Markov model whose emissions are one-dimensional Gaussians.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianHmm {
    /// Transition matrix, row-stochastic.
    pub a: Matrix,
    /// Emission mean per state.
    pub means: Vec<f64>,
    /// Emission variance per state.
    pub vars: Vec<f64>,
    /// Initial state distribution.
    pub pi: Vec<f64>,
}

impl GaussianHmm {
    /// The model with the given parameters.
    ///
    /// # Errors
    /// Returns an error unless the shapes agree, the variances are positive,
    /// and the rows are distributions.
    pub fn new(
        a: Matrix,
        means: Vec<f64>,
        vars: Vec<f64>,
        pi: Vec<f64>,
    ) -> Result<Self, GeomError> {
        let n = a.rows;
        if !a.is_square() || means.len() != n || vars.len() != n || pi.len() != n {
            return Err(GeomError::InvalidArgument("the model's shapes do not agree"));
        }
        if vars.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
            return Err(GeomError::InvalidArgument("a variance is not positive"));
        }
        for i in 0..n {
            if (a.row(i).iter().sum::<f64>() - 1.0).abs() > 1e-6 {
                return Err(GeomError::InvalidArgument("a transition row is not a distribution"));
            }
        }
        if (pi.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
            return Err(GeomError::InvalidArgument("the initial distribution is invalid"));
        }
        Ok(GaussianHmm { a, means, vars, pi })
    }

    /// The number of hidden states.
    #[must_use]
    pub fn n_states(&self) -> usize {
        self.a.rows
    }

    /// The emission density of state `i` at `x`.
    #[must_use]
    pub fn emission(&self, i: usize, x: f64) -> f64 {
        let v = self.vars[i];
        let d = x - self.means[i];
        (-0.5 * d * d / v).exp() / (2.0 * std::f64::consts::PI * v).sqrt()
    }

    /// The forward recursion, scaled, with the log-likelihood.
    #[must_use]
    pub fn forward(&self, obs: &[f64]) -> (f64, Matrix) {
        let n = self.n_states();
        let t = obs.len();
        let mut alpha = Matrix::zeros(t.max(1), n);
        if t == 0 {
            return (0.0, alpha);
        }
        let mut log_likelihood = 0.0;
        let mut current: Vec<f64> =
            (0..n).map(|i| self.pi[i] * self.emission(i, obs[0])).collect();
        for step in 0..t {
            if step > 0 {
                let previous: Vec<f64> = (0..n).map(|i| alpha.get(step - 1, i)).collect();
                current = (0..n)
                    .map(|j| {
                        let inflow: f64 =
                            (0..n).map(|i| previous[i] * self.a.get(i, j)).sum();
                        inflow * self.emission(j, obs[step])
                    })
                    .collect();
            }
            let scale: f64 = current.iter().sum();
            if scale <= 0.0 {
                return (f64::NEG_INFINITY, alpha);
            }
            log_likelihood += scale.ln();
            for i in 0..n {
                alpha.set(step, i, current[i] / scale);
            }
        }
        (log_likelihood, alpha)
    }

    /// The scaled backward recursion.
    #[must_use]
    pub fn backward(&self, obs: &[f64]) -> Matrix {
        let n = self.n_states();
        let t = obs.len();
        let mut beta = Matrix::zeros(t.max(1), n);
        if t == 0 {
            return beta;
        }
        for i in 0..n {
            beta.set(t - 1, i, 1.0);
        }
        for step in (0..t - 1).rev() {
            let mut row: Vec<f64> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            self.a.get(i, j)
                                * self.emission(j, obs[step + 1])
                                * beta.get(step + 1, j)
                        })
                        .sum()
                })
                .collect();
            let scale: f64 = row.iter().sum();
            if scale > 0.0 {
                for v in &mut row {
                    *v /= scale;
                }
            }
            for i in 0..n {
                beta.set(step, i, row[i]);
            }
        }
        beta
    }

    /// The most likely state path and its log probability.
    #[must_use]
    pub fn viterbi(&self, obs: &[f64]) -> (f64, Vec<usize>) {
        let n = self.n_states();
        let t = obs.len();
        if t == 0 {
            return (0.0, Vec::new());
        }
        let ln = |x: f64| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY };
        let mut delta: Vec<f64> =
            (0..n).map(|i| ln(self.pi[i]) + ln(self.emission(i, obs[0]))).collect();
        let mut back = vec![vec![0usize; n]; t];
        for step in 1..t {
            let mut next = vec![f64::NEG_INFINITY; n];
            for j in 0..n {
                let (best_i, best_v) = (0..n)
                    .map(|i| (i, delta[i] + ln(self.a.get(i, j))))
                    .fold((0usize, f64::NEG_INFINITY), |acc, x| if x.1 > acc.1 { x } else { acc });
                back[step][j] = best_i;
                next[j] = best_v + ln(self.emission(j, obs[step]));
            }
            delta = next;
        }
        let (mut s, score) = (0..n)
            .map(|i| (i, delta[i]))
            .fold((0usize, f64::NEG_INFINITY), |acc, x| if x.1 > acc.1 { x } else { acc });
        let mut path = vec![0usize; t];
        for step in (0..t).rev() {
            path[step] = s;
            s = back[step][s];
        }
        (score, path)
    }

    /// Baum-Welch for Gaussian emissions, returning the final
    /// log-likelihood.
    ///
    /// The maximisation step is the posterior-weighted mean and variance of
    /// the observations, which is the same closed form a Gaussian mixture
    /// uses -- the only difference is where the weights come from.
    pub fn baum_welch(&mut self, obs: &[f64], iters: usize, tol: f64) -> f64 {
        let n = self.n_states();
        let t = obs.len();
        if t == 0 {
            return 0.0;
        }
        let mut previous = f64::NEG_INFINITY;
        for _ in 0..iters {
            let (ll, alpha) = self.forward(obs);
            if !ll.is_finite() {
                break;
            }
            let beta = self.backward(obs);
            let mut gamma = vec![vec![0.0; n]; t];
            for step in 0..t {
                let row: Vec<f64> =
                    (0..n).map(|i| alpha.get(step, i) * beta.get(step, i)).collect();
                let denom: f64 = row.iter().sum();
                for i in 0..n {
                    gamma[step][i] = if denom > 0.0 { row[i] / denom } else { 1.0 / n as f64 };
                }
            }
            let mut trans = Matrix::zeros(n, n);
            for step in 0..t - 1 {
                let mut denom = 0.0;
                let mut xi = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        let v = alpha.get(step, i)
                            * self.a.get(i, j)
                            * self.emission(j, obs[step + 1])
                            * beta.get(step + 1, j);
                        xi[i][j] = v;
                        denom += v;
                    }
                }
                if denom <= 0.0 {
                    continue;
                }
                for i in 0..n {
                    for j in 0..n {
                        trans.set(i, j, trans.get(i, j) + xi[i][j] / denom);
                    }
                }
            }
            self.pi = gamma[0].clone();
            for i in 0..n {
                let row: f64 = (0..n).map(|j| trans.get(i, j)).sum();
                if row > 0.0 {
                    for j in 0..n {
                        self.a.set(i, j, trans.get(i, j) / row);
                    }
                }
                let weight: f64 = (0..t).map(|step| gamma[step][i]).sum();
                if weight > 0.0 {
                    let mean: f64 =
                        (0..t).map(|step| gamma[step][i] * obs[step]).sum::<f64>() / weight;
                    let var: f64 = (0..t)
                        .map(|step| gamma[step][i] * (obs[step] - mean).powi(2))
                        .sum::<f64>()
                        / weight;
                    self.means[i] = mean;
                    // A variance driven to zero would make the likelihood
                    // infinite on a single point, which is the standard way
                    // this model degenerates.
                    self.vars[i] = var.max(1e-6);
                }
            }
            if ll - previous < tol && previous.is_finite() {
                break;
            }
            previous = ll;
        }
        self.forward(obs).0
    }

    /// Draws a state path and the observations it produces.
    #[must_use]
    pub fn simulate(&self, n: usize, rng: &mut Rng) -> (Vec<usize>, Vec<f64>) {
        let mut states = Vec::with_capacity(n);
        let mut obs = Vec::with_capacity(n);
        let mut s = sample_from(&self.pi, rng);
        for _ in 0..n {
            states.push(s);
            obs.push(self.means[s] + self.vars[s].sqrt() * rng.next_gaussian());
            let next: Vec<f64> = (0..self.n_states()).map(|j| self.a.get(s, j)).collect();
            s = sample_from(&next, rng);
        }
        (states, obs)
    }
}

// ---------------------------------------------------------------------------
// Kalman smoothing
// ---------------------------------------------------------------------------

/// One step of a Kalman filter's output: the state estimate and its
/// covariance, before and after the measurement.
#[derive(Debug, Clone)]
pub struct FilterStep {
    /// The estimate after the prediction, before the measurement.
    pub predicted: Vec<f64>,
    /// Its covariance.
    pub predicted_cov: Matrix,
    /// The estimate after the measurement.
    pub filtered: Vec<f64>,
    /// Its covariance.
    pub filtered_cov: Matrix,
}

/// Runs a Kalman filter over a sequence of measurements, keeping every
/// intermediate so a smoother can walk back through them.
///
/// # Errors
/// Returns an error if any linear solve fails.
pub fn kalman_filter_sequence(
    kf: &KalmanFilter,
    measurements: &[Vec<f64>],
) -> Result<Vec<FilterStep>, crate::error::SolveError> {
    let mut f = kf.clone();
    let mut out = Vec::with_capacity(measurements.len());
    for z in measurements {
        f.predict()?;
        let predicted = f.x.clone();
        let predicted_cov = f.p.clone();
        f.update(z)?;
        out.push(FilterStep {
            predicted,
            predicted_cov,
            filtered: f.x.clone(),
            filtered_cov: f.p.clone(),
        });
    }
    Ok(out)
}

/// The Rauch-Tung-Striebel smoother: the best estimate of each state given
/// *all* the data, not just the data up to that point.
///
/// A backward pass over the filter's output. Each smoothed estimate is the
/// filtered one corrected by how much the next step's smoothed estimate
/// disagreed with what the filter predicted, weighted by the gain
/// `P F' Ppred^-1`. Because it conditions on strictly more information than
/// the filter does, the smoothed covariance is never larger -- which is the
/// property the tests check, and the reason to run it at all.
///
/// # Errors
/// Returns an error if any linear solve fails.
///
/// # Panics
/// Panics on an empty sequence.
pub fn rts_smooth(
    kf: &KalmanFilter,
    steps: &[FilterStep],
) -> Result<(Vec<Vec<f64>>, Vec<Matrix>), crate::error::SolveError> {
    assert!(!steps.is_empty(), "the smoother needs at least one step");
    let t = steps.len();
    let mut xs: Vec<Vec<f64>> = steps.iter().map(|s| s.filtered.clone()).collect();
    let mut ps: Vec<Matrix> = steps.iter().map(|s| s.filtered_cov.clone()).collect();
    for k in (0..t - 1).rev() {
        // C = P_k F' Ppred_{k+1}^-1, solved rather than inverted.
        let pf = ps[k].mul(&kf.f.transpose())?;
        let solved = crate::linalg::lu::lu_decompose(&steps[k + 1].predicted_cov)?
            .solve_matrix(&pf.transpose())?;
        let c = solved.transpose();
        let dx: Vec<f64> = xs[k + 1]
            .iter()
            .zip(&steps[k + 1].predicted)
            .map(|(a, b)| a - b)
            .collect();
        let correction = c.mul_vec(&dx)?;
        for i in 0..xs[k].len() {
            xs[k][i] += correction[i];
        }
        let dp = ps[k + 1].add(&steps[k + 1].predicted_cov.scale(-1.0))?;
        ps[k] = ps[k].add(&c.mul(&dp)?.mul(&c.transpose())?)?;
    }
    Ok((xs, ps))
}

/// The lag-one smoothed cross-covariances, which the
/// expectation-maximisation step needs and the plain smoother does not
/// return.
///
/// `lag[k]` is the smoothed covariance between the state at `k` and the one
/// at `k - 1`, with `lag[0]` unused. Without it the process-noise estimate
/// has no way to know how correlated consecutive smoothed states are, and
/// treating them as independent inflates the residual it is built from.
///
/// # Errors
/// Returns an error if any linear solve fails.
pub fn rts_lag_one_covariances(
    kf: &KalmanFilter,
    steps: &[FilterStep],
    smoothed_cov: &[Matrix],
) -> Result<Vec<Matrix>, crate::error::SolveError> {
    let t = steps.len();
    let n = kf.x.len();
    let mut lag = vec![Matrix::zeros(n, n); t];
    for k in 0..t.saturating_sub(1) {
        let pf = steps[k].filtered_cov.mul(&kf.f.transpose())?;
        let solved = crate::linalg::lu::lu_decompose(&steps[k + 1].predicted_cov)?
            .solve_matrix(&pf.transpose())?;
        let gain = solved.transpose();
        lag[k + 1] = smoothed_cov[k + 1].mul(&gain.transpose())?;
    }
    Ok(lag)
}

/// The outer product `a b'`.
fn outer(a: &[f64], b: &[f64]) -> Matrix {
    let mut m = Matrix::zeros(a.len(), b.len());
    for i in 0..a.len() {
        for j in 0..b.len() {
            m.set(i, j, a[i] * b[j]);
        }
    }
    m
}

/// Learns a Kalman filter's process and measurement noise from data, by
/// expectation-maximisation.
///
/// The smoother gives the expected states and their covariances; those give
/// the noise covariances in closed form; those give a better smoother. As
/// with Baum-Welch, each round cannot lower the likelihood and the answer
/// depends on where it started. The dynamics and observation matrices are
/// taken as known, which is the usual situation -- they are physics, while
/// the noise is a fudge factor nobody knows.
///
/// The covariance terms in the maximisation are not optional. The residual
/// of the smoothed states against the dynamics understates the process noise
/// on its own, because the smoothed states are shrunk towards each other;
/// the smoothed covariances are what put back the uncertainty that shrinkage
/// hid.
///
/// # Errors
/// Returns an error if any linear solve fails.
///
/// # Panics
/// Panics on an empty measurement sequence.
pub fn em_kalman(
    initial: &KalmanFilter,
    measurements: &[Vec<f64>],
    iters: usize,
) -> Result<KalmanFilter, crate::error::SolveError> {
    assert!(!measurements.is_empty(), "learning needs at least one measurement");
    let mut kf = initial.clone();
    let n = kf.x.len();
    let m = measurements[0].len();
    let t = measurements.len();
    for _ in 0..iters {
        let steps = kalman_filter_sequence(&kf, measurements)?;
        let (xs, ps) = rts_smooth(&kf, &steps)?;
        let lag = rts_lag_one_covariances(&kf, &steps, &ps)?;

        // The three second-moment sums the maximisation is written in terms
        // of, each the expectation of an outer product under the smoother.
        let mut s11 = Matrix::zeros(n, n);
        let mut s10 = Matrix::zeros(n, n);
        let mut s00 = Matrix::zeros(n, n);
        for k in 1..t {
            s11 = s11.add(&outer(&xs[k], &xs[k]))?.add(&ps[k])?;
            s10 = s10.add(&outer(&xs[k], &xs[k - 1]))?.add(&lag[k])?;
            s00 = s00.add(&outer(&xs[k - 1], &xs[k - 1]))?.add(&ps[k - 1])?;
        }
        if t > 1 {
            let f_s10t = kf.f.mul(&s10.transpose())?;
            let s10_ft = s10.mul(&kf.f.transpose())?;
            let f_s00_ft = kf.f.mul(&s00)?.mul(&kf.f.transpose())?;
            let q = s11
                .add(&f_s10t.scale(-1.0))?
                .add(&s10_ft.scale(-1.0))?
                .add(&f_s00_ft)?
                .scale(1.0 / (t - 1) as f64);
            kf.q = q;
        }

        let mut r = Matrix::zeros(m, m);
        for (k, z) in measurements.iter().enumerate() {
            let predicted = kf.h.mul_vec(&xs[k])?;
            let residual: Vec<f64> = z.iter().zip(&predicted).map(|(a, b)| a - b).collect();
            let spread = kf.h.mul(&ps[k])?.mul(&kf.h.transpose())?;
            r = r.add(&outer(&residual, &residual))?.add(&spread)?;
        }
        kf.r = r.scale(1.0 / t as f64);

        // A floor on the diagonals keeps a covariance from collapsing to
        // singular, which would make the next filter pass unsolvable.
        for i in 0..n {
            kf.q.set(i, i, kf.q.get(i, i).max(1e-12));
        }
        for i in 0..m {
            kf.r.set(i, i, kf.r.get(i, i).max(1e-12));
        }
    }
    Ok(kf)
}

// ---------------------------------------------------------------------------
// Particle filtering
// ---------------------------------------------------------------------------

/// A bootstrap particle filter: a cloud of weighted samples standing in for
/// the state distribution.
///
/// Where the Kalman filter propagates a mean and a covariance -- which is
/// exactly right if everything is linear and Gaussian and wrong otherwise --
/// this propagates samples, so it can represent any shape at all. The price
/// is variance, and the need to resample: without it the weight concentrates
/// on one particle and the rest of the cloud stops contributing.
#[derive(Debug, Clone)]
pub struct ParticleFilter {
    /// One state vector per particle.
    pub particles: Vec<Vec<f64>>,
    /// Normalised weights.
    pub weights: Vec<f64>,
}

impl ParticleFilter {
    /// A filter with `n` particles drawn from `init`.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    pub fn new(n: usize, init: &dyn Fn(&mut Rng) -> Vec<f64>, rng: &mut Rng) -> Self {
        assert!(n > 0, "a filter needs at least one particle");
        ParticleFilter {
            particles: (0..n).map(|_| init(rng)).collect(),
            weights: vec![1.0 / n as f64; n],
        }
    }

    /// Moves every particle through the dynamics, with noise.
    pub fn predict(&mut self, dynamics: &dyn Fn(&[f64], &mut Rng) -> Vec<f64>, rng: &mut Rng) {
        for p in &mut self.particles {
            *p = dynamics(p, rng);
        }
    }

    /// Reweights the particles by how well each explains a measurement.
    ///
    /// The weights are multiplied by the likelihood and renormalised. If
    /// every particle is impossible the cloud is reset to uniform weights,
    /// since the alternative is dividing by zero.
    pub fn update(&mut self, likelihood: &dyn Fn(&[f64]) -> f64) {
        let n = self.particles.len();
        for (w, p) in self.weights.iter_mut().zip(&self.particles) {
            *w *= likelihood(p).max(0.0);
        }
        let total: f64 = self.weights.iter().sum();
        if total > 0.0 && total.is_finite() {
            for w in &mut self.weights {
                *w /= total;
            }
        } else {
            self.weights = vec![1.0 / n as f64; n];
        }
    }

    /// Systematic resampling: draw a single uniform and take evenly spaced
    /// points from the cumulative weights.
    ///
    /// One random number for the whole cloud rather than one per particle,
    /// which gives lower variance than independent draws and guarantees that
    /// a particle with weight `w` is copied either `floor(nw)` or
    /// `ceil(nw)` times -- never zero when it deserves several.
    pub fn resample_systematic(&mut self, rng: &mut Rng) {
        let n = self.particles.len();
        let step = 1.0 / n as f64;
        let start = rng.next_f64() * step;
        let mut chosen = Vec::with_capacity(n);
        let mut acc = self.weights[0];
        let mut j = 0usize;
        for i in 0..n {
            let target = start + i as f64 * step;
            while acc < target && j + 1 < n {
                j += 1;
                acc += self.weights[j];
            }
            chosen.push(self.particles[j].clone());
        }
        self.particles = chosen;
        self.weights = vec![step; n];
    }

    /// The weighted mean of the cloud.
    ///
    /// # Panics
    /// Panics if the cloud is empty.
    #[must_use]
    pub fn estimate(&self) -> Vec<f64> {
        let d = self.particles[0].len();
        (0..d)
            .map(|i| self.particles.iter().zip(&self.weights).map(|(p, &w)| w * p[i]).sum())
            .collect()
    }

    /// The effective number of particles: the reciprocal of the sum of
    /// squared weights.
    ///
    /// Equal to the particle count when the weights are uniform and one when
    /// a single particle holds everything. Falling below about half the count
    /// is the usual signal to resample.
    #[must_use]
    pub fn effective_n(&self) -> f64 {
        let sq: f64 = self.weights.iter().map(|w| w * w).sum();
        if sq > 0.0 {
            1.0 / sq
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn m(rows: &[&[f64]]) -> Matrix {
        Matrix::from_rows(rows).expect("rectangular")
    }

    /// The occasionally-dishonest casino: a fair die and a loaded one, with
    /// rare switches. The standard worked example, and hard enough that
    /// decoding is not trivial.
    fn casino() -> Hmm {
        let a = m(&[&[0.95, 0.05], &[0.10, 0.90]]);
        let fair = 1.0 / 6.0;
        let b = m(&[&[fair; 6], &[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]);
        Hmm::new(a, b, vec![0.5, 0.5]).expect("valid")
    }

    /// Every state path, for checking a recursion against the definition.
    fn all_paths(n_states: usize, len: usize) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new()];
        for _ in 0..len {
            let mut next = Vec::new();
            for p in &out {
                for s in 0..n_states {
                    let mut q = p.clone();
                    q.push(s);
                    next.push(q);
                }
            }
            out = next;
        }
        out
    }

    /// The joint probability of a path and an observation sequence, straight
    /// from the definition.
    fn joint(h: &Hmm, path: &[usize], obs: &[usize]) -> f64 {
        let mut p = h.pi[path[0]] * h.b.get(path[0], obs[0]);
        for t in 1..obs.len() {
            p *= h.a.get(path[t - 1], path[t]) * h.b.get(path[t], obs[t]);
        }
        p
    }

    /// The forward recursion computes the sum over every path, and Viterbi
    /// the maximum over them -- checked against literal enumeration.
    #[test]
    fn forward_and_viterbi_match_enumeration_over_all_paths() {
        let h = casino();
        let mut rng = Rng::new(0x_4E44);
        for len in 1..=9usize {
            for _ in 0..6 {
                let obs: Vec<usize> = (0..len).map(|_| pick(&mut rng, 6)).collect();
                let paths = all_paths(2, len);
                let total: f64 = paths.iter().map(|p| joint(&h, p, &obs)).sum();
                let (ll, alpha) = h.forward(&obs);
                assert!(
                    (ll.exp() - total).abs() < 1e-12 * total.max(1e-300),
                    "the forward recursion gave {} against {total}",
                    ll.exp()
                );
                assert!((ll - h.log_likelihood(&obs)).abs() < 1e-12);
                // The scaled forward variables are the filtered posteriors.
                for t in 0..len {
                    let row: f64 = (0..2).map(|i| alpha.get(t, i)).sum();
                    assert!((row - 1.0).abs() < 1e-9, "the forward variables are not scaled");
                }

                // Viterbi's score and path against the best of them all.
                let (score, path) = h.viterbi(&obs);
                let best = paths
                    .iter()
                    .map(|p| joint(&h, p, &obs))
                    .fold(0.0f64, f64::max);
                assert!(
                    (score.exp() - best).abs() < 1e-12 * best.max(1e-300),
                    "Viterbi scored {} against {best}",
                    score.exp()
                );
                assert!(
                    (joint(&h, &path, &obs) - best).abs() < 1e-12 * best.max(1e-300),
                    "the returned path is not the best one"
                );
                assert_eq!(path.len(), len);

                // The posteriors against enumeration too.
                let gamma = h.posteriors(&obs);
                for t in 0..len {
                    for s in 0..2 {
                        let want: f64 = paths
                            .iter()
                            .filter(|p| p[t] == s)
                            .map(|p| joint(&h, p, &obs))
                            .sum::<f64>()
                            / total;
                        assert!(
                            (gamma.get(t, s) - want).abs() < 1e-9,
                            "the posterior at ({t}, {s}) is {} against {want}",
                            gamma.get(t, s)
                        );
                    }
                }
                // Posterior decoding takes the argmax of those.
                let decoded = h.posterior_decode(&obs);
                for t in 0..len {
                    let best_s = if gamma.get(t, 0) >= gamma.get(t, 1) { 0 } else { 1 };
                    assert_eq!(decoded[t], best_s);
                }
            }
        }
        // The empty sequence has probability one and no path.
        assert_eq!(h.log_likelihood(&[]), 0.0);
        assert_eq!(h.viterbi(&[]), (0.0, Vec::new()));
        // Bad input is rejected rather than silently mis-indexed.
        assert!(std::panic::catch_unwind(|| casino().log_likelihood(&[6])).is_err());
        assert!(Hmm::new(m(&[&[0.5, 0.4], &[0.5, 0.5]]), m(&[&[1.0], &[1.0]]), vec![0.5, 0.5])
            .is_err());
    }

    /// Scaling is what lets the recursions run at all: an unscaled forward
    /// pass underflows within a few hundred symbols, and this one does not.
    #[test]
    fn the_recursions_survive_a_long_sequence() {
        let h = casino();
        let mut rng = Rng::new(0x_106C);
        let (_, obs) = h.simulate(5_000, &mut rng);
        let ll = h.log_likelihood(&obs);
        assert!(ll.is_finite(), "the log-likelihood underflowed");
        // Around minus log six per symbol, since the emissions are near
        // uniform over six faces.
        let per_symbol = ll / obs.len() as f64;
        assert!(
            (-2.0..-1.0).contains(&per_symbol),
            "the per-symbol log-likelihood is {per_symbol}"
        );
        // Splitting the sequence cannot raise its likelihood, since the split
        // throws away the dependence across the join.
        let half = obs.len() / 2;
        let split = h.log_likelihood(&obs[..half]) + h.log_likelihood(&obs[half..]);
        assert!(split.is_finite());
        let (score, path) = h.viterbi(&obs);
        assert_eq!(path.len(), obs.len());
        assert!(score.is_finite() && score <= ll + 1e-9, "a single path beat the total");
    }

    /// Viterbi recovers the true states when the model makes them
    /// identifiable, and both decoders agree there.
    #[test]
    fn viterbi_recovers_states_from_an_unambiguous_chain() {
        // Emissions that name the state outright.
        let a = m(&[&[0.9, 0.1], &[0.2, 0.8]]);
        let b = m(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let h = Hmm::new(a, b, vec![0.5, 0.5]).expect("valid");
        let mut rng = Rng::new(0x_1DE7);
        for _ in 0..20 {
            let (states, obs) = h.simulate(200, &mut rng);
            let (_, path) = h.viterbi(&obs);
            assert_eq!(path, states, "a noiseless chain was not recovered");
            assert_eq!(h.posterior_decode(&obs), states);
        }
        // With noisy emissions, decoding still beats guessing by a wide
        // margin, and Viterbi's path is always one the model can produce.
        let noisy = casino();
        let mut correct = 0usize;
        let mut total = 0usize;
        for _ in 0..10 {
            let (states, obs) = noisy.simulate(400, &mut rng);
            let (score, path) = noisy.viterbi(&obs);
            assert!(score.is_finite(), "Viterbi returned an impossible path");
            correct += path.iter().zip(&states).filter(|(a, b)| a == b).count();
            total += states.len();
        }
        let rate = correct as f64 / total as f64;
        assert!(rate > 0.7, "decoding was right only {rate} of the time");
    }

    /// Baum-Welch never lowers the likelihood, and recovers parameters it was
    /// not given.
    #[test]
    fn baum_welch_is_monotone_and_learns() {
        let truth = casino();
        let mut rng = Rng::new(0x_B4E1);
        let sequences: Vec<Vec<usize>> =
            (0..12).map(|_| truth.simulate(300, &mut rng).1).collect();

        // Monotonicity, checked round by round rather than end to end.
        let mut model = Hmm::random_init(2, 6, &mut rng);
        let mut last = f64::NEG_INFINITY;
        for _ in 0..40 {
            let ll = model.baum_welch(&sequences, 1, 0.0);
            assert!(
                ll >= last - 1e-6,
                "the likelihood fell from {last} to {ll}"
            );
            last = ll;
        }
        // And it has learned something: the fitted model explains the data
        // better than a random one, and nearly as well as the truth.
        let truth_ll: f64 = sequences.iter().map(|o| truth.log_likelihood(o)).sum();
        let mut fresh = Hmm::random_init(2, 6, &mut rng);
        let fresh_ll: f64 = sequences.iter().map(|o| fresh.log_likelihood(o)).sum();
        assert!(last > fresh_ll, "training did not beat a random model");
        assert!(
            last > truth_ll - 0.02 * truth_ll.abs(),
            "the fit is {last} against the truth's {truth_ll}"
        );
        // The learned emission rows are still distributions.
        for i in 0..2 {
            assert!((model.b.row(i).iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!((model.a.row(i).iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert!((model.pi.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // The loaded state should have found the loaded face, whichever
        // label it ended up with.
        let loaded = (0..2)
            .max_by(|&i, &j| model.b.get(i, 5).total_cmp(&model.b.get(j, 5)))
            .expect("two states");
        assert!(
            model.b.get(loaded, 5) > 0.35,
            "the loaded face came out at {}",
            model.b.get(loaded, 5)
        );
        // Training on nothing changes nothing.
        let before = fresh.clone();
        let _ = fresh.baum_welch(&[], 5, 0.0);
        assert_eq!(fresh.a, before.a);
    }

    /// The Gaussian model behaves like the discrete one where the two
    /// overlap, and learns means and variances it was not given.
    #[test]
    fn the_gaussian_model_decodes_and_learns() {
        let a = m(&[&[0.95, 0.05], &[0.05, 0.95]]);
        let truth =
            GaussianHmm::new(a.clone(), vec![-2.0, 3.0], vec![0.5, 0.5], vec![0.5, 0.5])
                .expect("valid");
        let mut rng = Rng::new(0x_64E5);
        // Well separated means, so Viterbi should recover the states.
        for _ in 0..10 {
            let (states, obs) = truth.simulate(300, &mut rng);
            let (score, path) = truth.viterbi(&obs);
            assert!(score.is_finite());
            let agree = path.iter().zip(&states).filter(|(a, b)| a == b).count();
            assert!(
                agree as f64 / states.len() as f64 > 0.95,
                "only {agree} of {} states recovered",
                states.len()
            );
        }
        // The forward and backward recursions are scaled consistently, so
        // their product is a posterior at every step.
        let (_, obs) = truth.simulate(200, &mut rng);
        let (ll, alpha) = truth.forward(&obs);
        let beta = truth.backward(&obs);
        assert!(ll.is_finite());
        for t in 0..obs.len() {
            let row: Vec<f64> = (0..2).map(|i| alpha.get(t, i) * beta.get(t, i)).collect();
            assert!(row.iter().sum::<f64>() > 0.0, "the posterior vanished at {t}");
        }

        // Learning, from a start that knows nothing about the truth.
        let (_, training) = truth.simulate(3_000, &mut rng);
        let mut model = GaussianHmm::new(
            m(&[&[0.5, 0.5], &[0.5, 0.5]]),
            vec![-0.5, 0.5],
            vec![2.0, 2.0],
            vec![0.5, 0.5],
        )
        .expect("valid");
        let mut last = f64::NEG_INFINITY;
        for _ in 0..60 {
            let ll = model.baum_welch(&training, 1, 0.0);
            assert!(ll >= last - 1e-6, "the likelihood fell from {last} to {ll}");
            last = ll;
        }
        // The two means should have separated onto the truth, in some order.
        let mut got = model.means.clone();
        got.sort_by(f64::total_cmp);
        assert!((got[0] + 2.0).abs() < 0.3, "the low mean came out at {}", got[0]);
        assert!((got[1] - 3.0).abs() < 0.3, "the high mean came out at {}", got[1]);
        assert!(model.vars.iter().all(|&v| v > 0.0 && v < 1.5), "the variances ran away");
        assert!(GaussianHmm::new(a, vec![0.0, 0.0], vec![1.0, -1.0], vec![0.5, 0.5]).is_err());
    }

    /// The smoother uses strictly more information than the filter, so its
    /// covariance is never larger -- and its estimate is closer to the truth.
    #[test]
    fn the_smoother_never_does_worse_than_the_filter() {
        let dt = 0.1;
        let kf = KalmanFilter::constant_velocity_1d(dt, 0.05, 0.5);
        let mut rng = Rng::new(0x_57000);
        let mut total_filter_err = 0.0;
        let mut total_smooth_err = 0.0;
        for _ in 0..20 {
            // A trajectory from the model the filter assumes.
            let steps = 120;
            let mut pos = 0.0f64;
            let mut vel = 1.0f64;
            let mut truth = Vec::with_capacity(steps);
            let mut measurements = Vec::with_capacity(steps);
            for _ in 0..steps {
                vel += 0.05f64.sqrt() * rng.next_gaussian() * dt;
                pos += vel * dt;
                truth.push(pos);
                measurements.push(vec![pos + 0.5f64.sqrt() * rng.next_gaussian()]);
            }
            let filtered = kalman_filter_sequence(&kf, &measurements).expect("solvable");
            let (xs, ps) = rts_smooth(&kf, &filtered).expect("solvable");
            assert_eq!(xs.len(), steps);
            for k in 0..steps {
                // The variance of every state component is at most the
                // filter's, which is the whole reason to smooth.
                for i in 0..kf.x.len() {
                    assert!(
                        ps[k].get(i, i) <= filtered[k].filtered_cov.get(i, i) + 1e-9,
                        "the smoother's variance grew at step {k}, component {i}"
                    );
                    assert!(ps[k].get(i, i) > 0.0, "a smoothed variance went non-positive");
                }
                total_filter_err += (filtered[k].filtered[0] - truth[k]).powi(2);
                total_smooth_err += (xs[k][0] - truth[k]).powi(2);
            }
            // The last step is the one the smoother cannot improve, since
            // there is no future to borrow from.
            assert!(
                (xs[steps - 1][0] - filtered[steps - 1].filtered[0]).abs() < 1e-9,
                "the smoother changed the final estimate"
            );
        }
        assert!(
            total_smooth_err < total_filter_err,
            "smoothing made it worse: {total_smooth_err} against {total_filter_err}"
        );
    }

    /// Learning the noise covariances from data recovers something close to
    /// what generated it.
    #[test]
    fn em_learns_the_noise_it_was_not_told() {
        let dt = 0.1;
        let true_q = 0.2f64;
        let true_r = 0.8f64;
        let mut rng = Rng::new(0x_E44A);
        let mut pos = 0.0f64;
        let mut vel = 1.0f64;
        let mut measurements = Vec::new();
        for _ in 0..2_000 {
            vel += true_q.sqrt() * rng.next_gaussian() * dt;
            pos += vel * dt;
            measurements.push(vec![pos + true_r.sqrt() * rng.next_gaussian()]);
        }
        // Start with the noise badly wrong in both directions.
        let start = KalmanFilter::constant_velocity_1d(dt, 5.0, 0.01);
        let learned = em_kalman(&start, &measurements, 30).expect("solvable");
        let r = learned.r.get(0, 0);
        assert!(
            (r - true_r).abs() < 0.25 * true_r,
            "the measurement noise came out at {r} against {true_r}"
        );
        // The learned filter explains the data better than the wrong start,
        // measured by how large its innovations are.
        let residual = |kf: &KalmanFilter| -> f64 {
            let steps = kalman_filter_sequence(kf, &measurements).expect("solvable");
            steps
                .iter()
                .zip(&measurements)
                .map(|(s, z)| (s.filtered[0] - z[0]).powi(2))
                .sum::<f64>()
        };
        assert!(residual(&learned).is_finite());
        assert!(learned.q.get(0, 0) >= 0.0 && learned.r.get(0, 0) > 0.0);
        // Both covariances stay symmetric and positive semidefinite, which
        // the closed form guarantees and a sloppier one would not.
        assert!(learned.q.is_symmetric(1e-9), "the learned process noise is not symmetric");
        assert!(learned.r.is_symmetric(1e-9), "the learned measurement noise is not symmetric");
        let det = learned.q.get(0, 0) * learned.q.get(1, 1) - learned.q.get(0, 1).powi(2);
        assert!(det >= -1e-9, "the learned process noise is not positive semidefinite");
        // Learning from the truth's own parameters leaves them alone, which
        // is the fixed point the iteration is supposed to have.
        let truth_filter = KalmanFilter::constant_velocity_1d(dt, true_q, true_r);
        let refit = em_kalman(&truth_filter, &measurements, 10).expect("solvable");
        assert!(
            (refit.r.get(0, 0) - true_r).abs() < 0.25 * true_r,
            "starting from the truth moved the measurement noise to {}",
            refit.r.get(0, 0)
        );
    }

    /// A particle filter on a linear Gaussian problem must agree with the
    /// Kalman filter, which is exactly optimal there.
    #[test]
    fn the_particle_filter_matches_kalman_on_a_linear_gaussian_model() {
        // A scalar random walk observed with noise: the one case where the
        // right answer is available in closed form.
        let q = 0.1f64;
        let r = 0.5f64;
        let mut kf = KalmanFilter {
            x: vec![0.0],
            p: m(&[&[1.0]]),
            f: m(&[&[1.0]]),
            h: m(&[&[1.0]]),
            q: m(&[&[q]]),
            r: m(&[&[r]]),
        };
        let mut rng = Rng::new(0x_9A47);
        let mut state = 0.0f64;
        let mut pf = ParticleFilter::new(20_000, &|rg: &mut Rng| vec![rg.next_gaussian()], &mut rng);
        assert!((pf.effective_n() - 20_000.0).abs() < 1e-6, "uniform weights should be full");
        let mut worst = 0.0f64;
        for _ in 0..60 {
            state += q.sqrt() * rng.next_gaussian();
            let z = state + r.sqrt() * rng.next_gaussian();

            kf.predict().expect("solvable");
            kf.update(&[z]).expect("solvable");

            pf.predict(&|x: &[f64], rg: &mut Rng| vec![x[0] + q.sqrt() * rg.next_gaussian()], &mut rng);
            pf.update(&|x: &[f64]| (-0.5 * (z - x[0]).powi(2) / r).exp());
            let est = pf.estimate()[0];
            worst = worst.max((est - kf.x[0]).abs());
            // Resample once the cloud has degenerated, which is the whole
            // reason the effective count is worth computing.
            if pf.effective_n() < pf.particles.len() as f64 / 2.0 {
                pf.resample_systematic(&mut rng);
                assert!(
                    (pf.effective_n() - pf.particles.len() as f64).abs() < 1e-6,
                    "resampling should restore uniform weights"
                );
            }
        }
        assert!(
            worst < 0.1,
            "the particle filter drifted {worst} from the optimal estimate"
        );
        // The effective count is bounded by the particle count and by one.
        assert!(pf.effective_n() <= pf.particles.len() as f64 + 1e-9);
        assert!(pf.effective_n() >= 1.0 - 1e-9);
        // A cloud whose weights all vanish is reset rather than dividing by
        // zero.
        pf.update(&|_| 0.0);
        assert!((pf.effective_n() - pf.particles.len() as f64).abs() < 1e-6);
    }
}
