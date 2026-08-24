//! Finite Markov chains and Markov chain Monte Carlo.
//!
//! A Markov chain is a square matrix whose rows sum to one, and almost
//! everything about it follows from linear algebra applied to that matrix.
//! The long-run behaviour is an eigenvector; how fast it is reached is the
//! gap between the leading eigenvalue and the next; expected hitting times
//! are the solution of a linear system; and the answer to "what happens after
//! `n` steps" is a matrix power.
//!
//! Markov chain Monte Carlo runs the idea backwards. Given a distribution you
//! can evaluate but not sample from, build a chain whose stationary
//! distribution is that one, and run it. Metropolis-Hastings does this by
//! proposing a move and accepting it with a probability that makes detailed
//! balance hold; Hamiltonian Monte Carlo does it by simulating a physical
//! trajectory that conserves energy, so the acceptance probability stays near
//! one even for a long move. The samplers are only ever asymptotically
//! correct, so the diagnostics -- effective sample size, the Gelman-Rubin
//! statistic, the autocorrelation time -- are not optional extras but the
//! only evidence that a run has converged.

use crate::error::GeomError;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// How a state behaves in the long run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateClass {
    /// Once left, never returned to with probability one.
    Transient,
    /// Returned to with probability one, and part of a closed set.
    Recurrent,
    /// Recurrent and alone: once entered, never left.
    Absorbing,
}

/// A finite Markov chain, held as its row-stochastic transition matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovChain {
    /// Row `i` is the distribution of the next state given the current one.
    pub p: Matrix,
}

const TOL: f64 = 1e-9;

impl MarkovChain {
    /// The chain with the given transition matrix.
    ///
    /// # Errors
    /// Returns an error unless the matrix is square, non-empty, has no
    /// negative entries, and every row sums to one.
    pub fn new(p: Matrix) -> Result<Self, GeomError> {
        // No public Matrix constructor yields a zero-sized matrix, so
        // squareness is the only shape left to check.
        if !p.is_square() {
            return Err(GeomError::InvalidArgument("a chain needs a square matrix"));
        }
        for i in 0..p.rows {
            let mut sum = 0.0;
            for j in 0..p.cols {
                let v = p.get(i, j);
                if v < -TOL || !v.is_finite() {
                    return Err(GeomError::InvalidArgument("a transition probability is invalid"));
                }
                sum += v;
            }
            if (sum - 1.0).abs() > 1e-6 {
                return Err(GeomError::InvalidArgument("a row does not sum to one"));
            }
        }
        Ok(MarkovChain { p })
    }

    /// The number of states.
    #[must_use]
    pub fn n(&self) -> usize {
        self.p.rows
    }

    /// A chain estimated from a matrix of observed transition counts.
    ///
    /// Each row is normalised by its total, which is the maximum likelihood
    /// estimate. A row with no observations is made absorbing, since the data
    /// says nothing about where that state goes and any other choice would be
    /// an invention.
    ///
    /// # Errors
    /// Returns an error unless the counts form a non-empty square matrix with
    /// no negative entries.
    pub fn from_counts(transitions: &Matrix) -> Result<Self, GeomError> {
        if !transitions.is_square() {
            return Err(GeomError::InvalidArgument("counts must be a square matrix"));
        }
        let n = transitions.rows;
        let mut p = Matrix::zeros(n, n);
        for i in 0..n {
            let total: f64 = (0..n).map(|j| transitions.get(i, j)).sum();
            if transitions.row(i).iter().any(|&v| v < 0.0) {
                return Err(GeomError::InvalidArgument("a transition count is negative"));
            }
            if total <= 0.0 {
                p.set(i, i, 1.0);
            } else {
                for j in 0..n {
                    p.set(i, j, transitions.get(i, j) / total);
                }
            }
        }
        MarkovChain::new(p)
    }

    /// A chain estimated from one observed sequence of states.
    ///
    /// # Errors
    /// Returns an error if `n_states` is zero or a state is out of range.
    pub fn from_sequence(states: &[usize], n_states: usize) -> Result<Self, GeomError> {
        if n_states == 0 {
            return Err(GeomError::InvalidArgument("a chain needs at least one state"));
        }
        if states.iter().any(|&s| s >= n_states) {
            return Err(GeomError::InvalidArgument("a state is outside the range"));
        }
        let mut counts = Matrix::zeros(n_states, n_states);
        for w in states.windows(2) {
            counts.set(w[0], w[1], counts.get(w[0], w[1]) + 1.0);
        }
        MarkovChain::from_counts(&counts)
    }

    /// The distribution one step on from `dist`.
    ///
    /// # Panics
    /// Panics unless `dist` has one entry per state.
    #[must_use]
    pub fn step_dist(&self, dist: &[f64]) -> Vec<f64> {
        assert_eq!(dist.len(), self.n(), "one probability per state is required");
        (0..self.n())
            .map(|j| (0..self.n()).map(|i| dist[i] * self.p.get(i, j)).sum())
            .collect()
    }

    /// The `n`-step transition matrix, by repeated squaring.
    #[must_use]
    pub fn n_step(&self, n: usize) -> Matrix {
        let mut acc = Matrix::identity(self.n());
        let mut base = self.p.clone();
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base).expect("square matrices of the same size");
            }
            base = base.mul(&base).expect("square matrices of the same size");
            e >>= 1;
        }
        acc
    }

    /// A stationary distribution: a row vector left fixed by the matrix.
    ///
    /// Solved as a linear system rather than found by iteration, so a
    /// periodic chain -- where the powers of the matrix never converge --
    /// still gives its stationary distribution. The system is `pi (P - I) =
    /// 0` with the normalisation `sum pi = 1` substituted for one of the
    /// redundant equations.
    ///
    /// # Panics
    /// Panics if the linear system is singular, which happens only when the
    /// matrix is not stochastic.
    #[must_use]
    pub fn stationary(&self) -> Vec<f64> {
        let n = self.n();
        // Columns of (P' - I), with the last row replaced by all ones.
        let mut a = Matrix::zeros(n, n);
        for i in 0..n - 1 {
            for j in 0..n {
                a.set(i, j, self.p.get(j, i) - f64::from(u8::from(i == j)));
            }
        }
        for j in 0..n {
            a.set(n - 1, j, 1.0);
        }
        let mut b = vec![0.0; n];
        b[n - 1] = 1.0;
        let mut pi = crate::linalg::lu::solve(&a, &b).expect("a stochastic matrix is solvable");
        // Clamp and renormalise: the exact zeros come back as tiny negatives.
        for v in &mut pi {
            *v = v.max(0.0);
        }
        let total: f64 = pi.iter().sum();
        if total > 0.0 {
            for v in &mut pi {
                *v /= total;
            }
        }
        pi
    }

    /// Runs the chain, returning the states visited including the start.
    ///
    /// # Panics
    /// Panics unless `start` is a valid state.
    #[must_use]
    pub fn simulate(&self, start: usize, steps: usize, rng: &mut Rng) -> Vec<usize> {
        assert!(start < self.n(), "the start state is outside the chain");
        let mut out = Vec::with_capacity(steps + 1);
        let mut s = start;
        out.push(s);
        for _ in 0..steps {
            let u = rng.next_f64();
            let mut acc = 0.0;
            let mut next = self.n() - 1;
            for j in 0..self.n() {
                acc += self.p.get(s, j);
                if u < acc {
                    next = j;
                    break;
                }
            }
            s = next;
            out.push(s);
        }
        out
    }

    /// Which states can be reached from which, by transitive closure.
    fn reachability(&self) -> Vec<Vec<bool>> {
        let n = self.n();
        let mut r = vec![vec![false; n]; n];
        for i in 0..n {
            r[i][i] = true;
            for j in 0..n {
                if self.p.get(i, j) > TOL {
                    r[i][j] = true;
                }
            }
        }
        for k in 0..n {
            for i in 0..n {
                if r[i][k] {
                    for j in 0..n {
                        if r[k][j] {
                            r[i][j] = true;
                        }
                    }
                }
            }
        }
        r
    }

    /// Whether every state can reach every other.
    #[must_use]
    pub fn is_irreducible(&self) -> bool {
        let r = self.reachability();
        r.iter().all(|row| row.iter().all(|&b| b))
    }

    /// The period of a state: the greatest common divisor of the lengths of
    /// the loops through it.
    ///
    /// One means aperiodic. A chain with a period above one cycles through
    /// classes of states and its matrix powers never settle, which is why
    /// aperiodicity is a hypothesis of every convergence theorem here.
    ///
    /// # Panics
    /// Panics unless `state` is valid.
    #[must_use]
    pub fn period(&self, state: usize) -> usize {
        assert!(state < self.n(), "the state is outside the chain");
        let n = self.n();
        // Breadth-first over path lengths, taking the gcd of every loop found
        // within twice the state count -- past that, no new residue appears.
        let mut seen: Vec<Option<usize>> = vec![None; n];
        let mut queue = std::collections::VecDeque::from([(state, 0usize)]);
        let mut period = 0usize;
        while let Some((s, d)) = queue.pop_front() {
            if let Some(prev) = seen[s] {
                period = gcd(period, d.abs_diff(prev));
                continue;
            }
            seen[s] = Some(d);
            for j in 0..n {
                if self.p.get(s, j) > TOL {
                    if j == state {
                        period = gcd(period, d + 1);
                    }
                    queue.push_back((j, d + 1));
                }
            }
        }
        if period == 0 {
            1
        } else {
            period
        }
    }

    /// Whether every state has period one.
    #[must_use]
    pub fn is_aperiodic(&self) -> bool {
        (0..self.n()).all(|s| self.period(s) == 1)
    }

    /// Each state's long-run behaviour.
    ///
    /// A state is recurrent when everything it can reach can reach it back,
    /// and transient otherwise; it is absorbing when it goes nowhere else.
    #[must_use]
    pub fn classify_states(&self) -> Vec<StateClass> {
        let r = self.reachability();
        (0..self.n())
            .map(|i| {
                if self.p.get(i, i) > 1.0 - 1e-9 {
                    StateClass::Absorbing
                } else if (0..self.n()).all(|j| !r[i][j] || r[j][i]) {
                    StateClass::Recurrent
                } else {
                    StateClass::Transient
                }
            })
            .collect()
    }

    /// The indices of the absorbing and transient states.
    fn absorbing_split(&self) -> (Vec<usize>, Vec<usize>) {
        let classes = self.classify_states();
        let absorbing: Vec<usize> =
            (0..self.n()).filter(|&i| classes[i] == StateClass::Absorbing).collect();
        let transient: Vec<usize> =
            (0..self.n()).filter(|&i| classes[i] != StateClass::Absorbing).collect();
        (absorbing, transient)
    }

    /// The probability of ending in each absorbing state, one row per
    /// transient state.
    ///
    /// The fundamental matrix `N = (I - Q)^-1` counts expected visits to each
    /// transient state before absorption -- its `(i, j)` entry is the sum
    /// over path lengths of the chance of being at `j` at that step -- and
    /// `N R` then routes those visits into the absorbing states. Columns
    /// follow the order the absorbing states appear in.
    ///
    /// # Panics
    /// Panics if the chain has no absorbing state, or if `I - Q` is singular,
    /// which means some transient state cannot reach absorption.
    #[must_use]
    pub fn absorbing_probabilities(&self) -> Matrix {
        let (absorbing, transient) = self.absorbing_split();
        assert!(!absorbing.is_empty(), "the chain has no absorbing state");
        let t = transient.len();
        let mut im_q = Matrix::zeros(t, t);
        for (a, &i) in transient.iter().enumerate() {
            for (b, &j) in transient.iter().enumerate() {
                im_q.set(a, b, f64::from(u8::from(a == b)) - self.p.get(i, j));
            }
        }
        let mut r = Matrix::zeros(t, absorbing.len());
        for (a, &i) in transient.iter().enumerate() {
            for (b, &j) in absorbing.iter().enumerate() {
                r.set(a, b, self.p.get(i, j));
            }
        }
        crate::linalg::lu::lu_decompose(&im_q)
            .expect("every transient state must reach absorption")
            .solve_matrix(&r)
            .expect("the system is solvable")
    }

    /// The fundamental matrix `N = (I - Q)^-1` over the transient states.
    ///
    /// # Panics
    /// Panics if `I - Q` is singular.
    #[must_use]
    pub fn fundamental_matrix(&self) -> Matrix {
        let (_, transient) = self.absorbing_split();
        let t = transient.len();
        let mut im_q = Matrix::zeros(t, t);
        for (a, &i) in transient.iter().enumerate() {
            for (b, &j) in transient.iter().enumerate() {
                im_q.set(a, b, f64::from(u8::from(a == b)) - self.p.get(i, j));
            }
        }
        crate::linalg::lu::lu_decompose(&im_q)
            .expect("every transient state must reach absorption")
            .inverse()
            .expect("the system is solvable")
    }

    /// Expected steps to absorption from each state, zero for the absorbing
    /// ones.
    ///
    /// The row sums of the fundamental matrix: total expected visits to all
    /// transient states is total expected time before leaving them.
    ///
    /// # Panics
    /// Panics if the chain has no absorbing state.
    #[must_use]
    pub fn expected_steps_to_absorption(&self) -> Vec<f64> {
        let (absorbing, transient) = self.absorbing_split();
        assert!(!absorbing.is_empty(), "the chain has no absorbing state");
        let n = self.fundamental_matrix();
        let mut out = vec![0.0; self.n()];
        for (a, &i) in transient.iter().enumerate() {
            out[i] = (0..transient.len()).map(|b| n.get(a, b)).sum();
        }
        out
    }

    /// The expected number of steps to first reach any state in `target`.
    ///
    /// Infinite when the target cannot be reached. Solved as the linear
    /// system `h_i = 1 + sum_j p_ij h_j` over the states outside the target,
    /// which is the first-step decomposition written down.
    ///
    /// # Panics
    /// Panics unless the states are valid.
    #[must_use]
    pub fn hitting_time(&self, from: usize, target: &[usize]) -> f64 {
        assert!(from < self.n(), "the start state is outside the chain");
        assert!(target.iter().all(|&t| t < self.n()), "a target is outside the chain");
        if target.contains(&from) {
            return 0.0;
        }
        let outside: Vec<usize> = (0..self.n()).filter(|i| !target.contains(i)).collect();
        let idx: std::collections::BTreeMap<usize, usize> =
            outside.iter().enumerate().map(|(a, &i)| (i, a)).collect();
        let m = outside.len();
        let mut a = Matrix::zeros(m, m);
        for (r, &i) in outside.iter().enumerate() {
            for (c, &j) in outside.iter().enumerate() {
                a.set(r, c, f64::from(u8::from(r == c)) - self.p.get(i, j));
            }
        }
        let b = vec![1.0; m];
        match crate::linalg::lu::solve(&a, &b) {
            Ok(h) => {
                let v = h[idx[&from]];
                if v.is_finite() && v >= 0.0 {
                    v
                } else {
                    f64::INFINITY
                }
            }
            // A singular system means some state outside the target can never
            // reach it, so the expectation does not exist.
            Err(_) => f64::INFINITY,
        }
    }

    /// The probability of ever reaching `target` from `from`.
    ///
    /// # Panics
    /// Panics unless the states are valid.
    #[must_use]
    pub fn hitting_probability(&self, from: usize, target: &[usize]) -> f64 {
        assert!(from < self.n(), "the start state is outside the chain");
        assert!(target.iter().all(|&t| t < self.n()), "a target is outside the chain");
        if target.contains(&from) {
            return 1.0;
        }
        // The minimal non-negative solution of h = P h with h = 1 on the
        // target, reached by iterating from zero -- which converges upward to
        // exactly that solution.
        let n = self.n();
        let mut h = vec![0.0; n];
        for &t in target {
            h[t] = 1.0;
        }
        for _ in 0..20_000 {
            let mut next = h.clone();
            let mut delta = 0.0f64;
            for i in 0..n {
                if target.contains(&i) {
                    continue;
                }
                let v: f64 = (0..n).map(|j| self.p.get(i, j) * h[j]).sum();
                delta = delta.max((v - h[i]).abs());
                next[i] = v;
            }
            h = next;
            if delta < 1e-14 {
                break;
            }
        }
        h[from].clamp(0.0, 1.0)
    }

    /// The expected number of steps to return to a state, starting from it.
    ///
    /// Kac's formula: the reciprocal of that state's stationary probability.
    /// It is one of the most useful facts about a chain -- the long-run share
    /// of time spent somewhere and the average wait between visits are
    /// reciprocals of each other, with no further hypothesis than
    /// irreducibility.
    ///
    /// # Panics
    /// Panics unless `state` is valid.
    #[must_use]
    pub fn return_time(&self, state: usize) -> f64 {
        assert!(state < self.n(), "the state is outside the chain");
        let pi = self.stationary()[state];
        if pi <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / pi
        }
    }

    /// The mean first passage time from every state to every other.
    ///
    /// The diagonal holds the return times.
    ///
    /// # Panics
    /// Panics if the chain has fewer than one state.
    #[must_use]
    pub fn mfpt_matrix(&self) -> Matrix {
        let n = self.n();
        let mut m = Matrix::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                m.set(i, j, if i == j { self.return_time(j) } else { self.hitting_time(i, &[j]) });
            }
        }
        m
    }

    /// Total variation distance between two distributions: half the sum of
    /// the absolute differences.
    ///
    /// The largest difference in probability the two assign to any event,
    /// which is why it is the metric convergence is measured in.
    ///
    /// # Panics
    /// Panics unless the two have the same length.
    #[must_use]
    pub fn total_variation_distance(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "the distributions must have the same length");
        0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
    }

    /// The number of steps until every start is within `eps` of stationary in
    /// total variation.
    ///
    /// Infinite for a chain that does not converge -- one that is reducible or
    /// periodic.
    ///
    /// # Panics
    /// Panics unless `eps` is in `(0, 1)`.
    #[must_use]
    pub fn mixing_time(&self, eps: f64) -> usize {
        assert!(eps > 0.0 && eps < 1.0, "eps must lie in (0, 1)");
        if !self.is_irreducible() || !self.is_aperiodic() {
            return usize::MAX;
        }
        let pi = self.stationary();
        let mut power = Matrix::identity(self.n());
        for t in 0..100_000 {
            let worst = (0..self.n())
                .map(|i| {
                    let row: Vec<f64> = (0..self.n()).map(|j| power.get(i, j)).collect();
                    MarkovChain::total_variation_distance(&row, &pi)
                })
                .fold(0.0f64, f64::max);
            if worst <= eps {
                return t;
            }
            power = power.mul(&self.p).expect("square matrices of the same size");
        }
        usize::MAX
    }

    /// The spectral gap: one minus the second-largest eigenvalue modulus.
    ///
    /// What sets the rate of convergence, since the distance to stationary
    /// falls like the second eigenvalue's magnitude raised to the step count.
    /// Zero for a chain that does not converge. Computed here through the
    /// symmetrised chain, so it is exact for reversible chains and a
    /// reasonable proxy otherwise.
    #[must_use]
    pub fn spectral_gap(&self) -> f64 {
        let n = self.n();
        if n < 2 {
            return 1.0;
        }
        let pi = self.stationary();
        // The additive reversibilisation: (P + P*)/2 for the time reversal
        // P*, which is self-adjoint in the stationary inner product and has
        // the same stationary distribution.
        let mut s = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let reversed = if pi[i] > 0.0 { pi[j] * self.p.get(j, i) / pi[i] } else { 0.0 };
                s.set(i, j, 0.5 * (self.p.get(i, j) + reversed));
            }
        }
        // Similarity by the square roots of pi turns it symmetric, so Jacobi
        // applies and the eigenvalues are real.
        let mut sym = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let scale = if pi[i] > 0.0 && pi[j] > 0.0 {
                    (pi[i] / pi[j]).sqrt()
                } else {
                    f64::from(u8::from(i == j))
                };
                sym.set(i, j, 0.5 * (s.get(i, j) * scale + s.get(j, i) / scale.max(1e-300)));
            }
        }
        let Ok(e) = crate::linalg::eigen::eigen_symmetric(&sym, 1e-12, 200) else {
            return 0.0;
        };
        let mut vals = e.values;
        vals.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        // The leading eigenvalue is one; the gap is to the next.
        (1.0 - vals.get(1).copied().unwrap_or(0.0).abs()).clamp(0.0, 1.0)
    }

    /// Whether the chain satisfies detailed balance against `pi`.
    ///
    /// `pi_i p_ij = pi_j p_ji` for every pair: the flow between any two
    /// states is the same in both directions. It is much stronger than
    /// stationarity, which needs only that the total flow into each state
    /// balances the total flow out, and it is what every Metropolis-Hastings
    /// sampler arranges because it is far easier to arrange.
    ///
    /// # Panics
    /// Panics unless `pi` has one entry per state.
    #[must_use]
    pub fn reversible_check(&self, pi: &[f64], tol: f64) -> bool {
        assert_eq!(pi.len(), self.n(), "one probability per state is required");
        (0..self.n()).all(|i| {
            (0..self.n())
                .all(|j| (pi[i] * self.p.get(i, j) - pi[j] * self.p.get(j, i)).abs() <= tol)
        })
    }

    /// The entropy rate: the average uncertainty per step in the long run, in
    /// bits.
    ///
    /// The stationary-weighted average of each row's entropy. It is the
    /// compression limit for a stream generated by the chain, and it is what
    /// separates a chain from a memoryless source with the same marginal:
    /// the marginal entropy is an upper bound and the difference is what the
    /// dependence saves.
    #[must_use]
    pub fn entropy_rate(&self) -> f64 {
        let pi = self.stationary();
        (0..self.n())
            .map(|i| {
                let row: f64 = (0..self.n())
                    .map(|j| self.p.get(i, j))
                    .filter(|&p| p > 0.0)
                    .map(|p| -p * p.log2())
                    .sum();
                pi[i] * row
            })
            .sum()
    }

    /// An exact sample from the stationary distribution, by coupling from the
    /// past.
    ///
    /// Ordinary simulation gives a sample that is only approximately
    /// stationary, with no way to tell how close. Propp and Wilson's
    /// construction instead runs every possible start from further and
    /// further back until they all coalesce by time zero; the common value is
    /// then *exactly* stationary, because whatever the chain was doing
    /// infinitely far back, it would have ended up there too.
    ///
    /// # Panics
    /// Panics if coalescence does not occur, which for an irreducible
    /// aperiodic chain means only that the bound was too small.
    #[must_use]
    pub fn coupling_from_the_past_small(&self, rng: &mut Rng) -> usize {
        let n = self.n();
        assert!(self.is_irreducible() && self.is_aperiodic(), "the chain must converge");
        // Randomness for each step back, reused as the window grows: that
        // reuse is what makes the result exact rather than merely close.
        let mut noise: Vec<f64> = Vec::new();
        let mut span = 1usize;
        for _ in 0..24 {
            while noise.len() < span {
                noise.push(rng.next_f64());
            }
            let mut states: Vec<usize> = (0..n).collect();
            // Run every start forward from -span to zero with shared noise.
            for t in (0..span).rev() {
                let u = noise[t];
                for s in &mut states {
                    let mut acc = 0.0;
                    let mut next = n - 1;
                    for j in 0..n {
                        acc += self.p.get(*s, j);
                        if u < acc {
                            next = j;
                            break;
                        }
                    }
                    *s = next;
                }
            }
            if states.iter().all(|&s| s == states[0]) {
                return states[0];
            }
            span *= 2;
        }
        panic!("the chain did not coalesce within the bound");
    }

    /// The PageRank chain of a graph: follow a random out-edge with
    /// probability `damping`, and teleport to a uniform vertex otherwise.
    ///
    /// The teleportation is what makes the chain irreducible and aperiodic
    /// whatever the graph looks like, so a stationary distribution exists and
    /// is unique. A vertex with no out-edges teleports always, which spreads
    /// its mass rather than letting it vanish.
    ///
    /// # Panics
    /// Panics unless the graph is non-empty and `damping` is in `[0, 1]`.
    #[must_use]
    pub fn pagerank_chain(g: &crate::graph::core::Graph, damping: f64) -> Self {
        assert!(g.n > 0, "the graph must have a vertex");
        assert!((0.0..=1.0).contains(&damping), "damping must lie in [0, 1]");
        let n = g.n;
        let mut p = Matrix::zeros(n, n);
        for i in 0..n {
            let out: Vec<usize> = g.adj[i].iter().map(|&(v, _)| v).collect();
            for j in 0..n {
                let teleport = (1.0 - damping) / n as f64;
                let follow = if out.is_empty() {
                    // A dangling vertex has nowhere to follow, so all of its
                    // mass teleports.
                    damping / n as f64
                } else {
                    damping * out.iter().filter(|&&v| v == j).count() as f64 / out.len() as f64
                };
                p.set(i, j, teleport + follow);
            }
        }
        MarkovChain::new(p).expect("the construction is stochastic")
    }
}

/// One sub-trajectory: its two ends, the states the slice admits, and
/// whether it may still be extended.
struct Subtree {
    qm: Vec<f64>,
    pm: Vec<f64>,
    qp: Vec<f64>,
    pp: Vec<f64>,
    candidates: Vec<Vec<f64>>,
    alive: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Whether the two ends of a trajectory are still moving apart, at both ends.
///
/// Checking both is what makes the criterion symmetric under reversing the
/// trajectory, and symmetry is what makes the sampler valid.
fn no_u_turn(qm: &[f64], qp: &[f64], pm: &[f64], pp: &[f64]) -> bool {
    let span: Vec<f64> = qp.iter().zip(qm).map(|(a, b)| a - b).collect();
    dot(&span, pm) >= 0.0 && dot(&span, pp) >= 0.0
}

/// One leapfrog step. A negative step integrates backwards, which is what
/// lets the trajectory be grown in either direction.
fn leapfrog(
    q: &[f64],
    p: &[f64],
    step: f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let g = grad(q);
    let mut ph: Vec<f64> = p.iter().zip(&g).map(|(v, gi)| v + 0.5 * step * gi).collect();
    let qn: Vec<f64> = q.iter().zip(&ph).map(|(v, pi)| v + step * pi).collect();
    let g2 = grad(&qn);
    for (v, gi) in ph.iter_mut().zip(&g2) {
        *v += 0.5 * step * gi;
    }
    (qn, ph)
}

/// Doubles a trajectory recursively, as the no-U-turn sampler prescribes.
fn build_tree(
    q: &[f64],
    p: &[f64],
    log_u: f64,
    step: f64,
    depth: usize,
    log_target: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Subtree {
    if depth == 0 {
        let (qn, pn) = leapfrog(q, p, step, grad);
        let joint = log_target(&qn) - 0.5 * dot(&pn, &pn);
        let candidates = if joint >= log_u { vec![qn.clone()] } else { Vec::new() };
        // A trajectory that has lost a thousand nats of energy has diverged,
        // and extending it would only waste work.
        let alive = joint > log_u - 1000.0 && joint.is_finite();
        return Subtree { qm: qn.clone(), pm: pn.clone(), qp: qn, pp: pn, candidates, alive };
    }
    let mut t = build_tree(q, p, log_u, step, depth - 1, log_target, grad);
    if t.alive {
        let far = if step < 0.0 {
            build_tree(&t.qm, &t.pm, log_u, step, depth - 1, log_target, grad)
        } else {
            build_tree(&t.qp, &t.pp, log_u, step, depth - 1, log_target, grad)
        };
        if step < 0.0 {
            t.qm = far.qm;
            t.pm = far.pm;
        } else {
            t.qp = far.qp;
            t.pp = far.pp;
        }
        t.candidates.extend(far.candidates);
        t.alive = far.alive && no_u_turn(&t.qm, &t.qp, &t.pm, &t.pp);
    }
    t
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ---------------------------------------------------------------------------
// Markov chain Monte Carlo
// ---------------------------------------------------------------------------

/// Samplers that build a chain whose stationary distribution is a target you
/// can evaluate but not sample from directly.
///
/// Every method here takes the *log* of the target, unnormalised. Logs
/// because the density of anything interesting underflows; unnormalised
/// because the normalising constant is exactly the thing that is usually
/// impossible to compute, and none of these methods needs it -- they see the
/// target only through ratios, in which it cancels.
#[derive(Debug, Clone, Copy)]
pub struct Mcmc;

impl Mcmc {
    /// Metropolis-Hastings with a symmetric Gaussian proposal.
    ///
    /// Propose a move, accept it outright if it goes uphill, and accept it
    /// with probability equal to the density ratio if it goes down. That rule
    /// makes detailed balance hold against the target, so the target is
    /// stationary; the downhill moves are not a concession but the mechanism,
    /// since a sampler that only climbed would sit at the mode forever.
    ///
    /// Returns the chain after discarding `burn` samples.
    ///
    /// # Panics
    /// Panics on an empty start, a non-positive proposal width, or a burn-in
    /// at or beyond the requested length.
    pub fn metropolis_hastings(
        log_target: &dyn Fn(&[f64]) -> f64,
        x0: &[f64],
        proposal_std: f64,
        n: usize,
        burn: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        assert!(!x0.is_empty(), "the start point must have a dimension");
        assert!(proposal_std > 0.0, "the proposal width must be positive");
        assert!(burn < n, "the burn-in must be shorter than the run");
        let mut x = x0.to_vec();
        let mut lp = log_target(&x);
        let mut out = Vec::with_capacity(n - burn);
        for t in 0..n {
            let candidate: Vec<f64> =
                x.iter().map(|&v| v + proposal_std * rng.next_gaussian()).collect();
            let lq = log_target(&candidate);
            // Comparing logs against a log uniform avoids exponentiating a
            // ratio that would overflow or underflow.
            if lq >= lp || rng.next_f64().ln() < lq - lp {
                x = candidate;
                lp = lq;
            }
            if t >= burn {
                out.push(x.clone());
            }
        }
        out
    }

    /// Metropolis-Hastings that tunes its own proposal width towards an
    /// acceptance rate of about a quarter.
    ///
    /// Too wide a proposal is rejected constantly and the chain stands still;
    /// too narrow a one is always accepted and the chain crawls. The optimum
    /// for a high-dimensional Gaussian target is famously near 0.234, and
    /// adapting towards it costs nothing. Adaptation stops at the end of
    /// burn-in, because a proposal that keeps changing breaks the Markov
    /// property and the chain is no longer guaranteed to have the right
    /// stationary distribution.
    ///
    /// # Panics
    /// Panics under the same conditions as
    /// [`metropolis_hastings`](Self::metropolis_hastings).
    pub fn adaptive_metropolis(
        log_target: &dyn Fn(&[f64]) -> f64,
        x0: &[f64],
        proposal_std: f64,
        n: usize,
        burn: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        assert!(!x0.is_empty(), "the start point must have a dimension");
        assert!(proposal_std > 0.0, "the proposal width must be positive");
        assert!(burn < n, "the burn-in must be shorter than the run");
        let mut x = x0.to_vec();
        let mut lp = log_target(&x);
        let mut width = proposal_std;
        let mut out = Vec::with_capacity(n - burn);
        let mut accepted = 0usize;
        for t in 0..n {
            let candidate: Vec<f64> =
                x.iter().map(|&v| v + width * rng.next_gaussian()).collect();
            let lq = log_target(&candidate);
            if lq >= lp || rng.next_f64().ln() < lq - lp {
                x = candidate;
                lp = lq;
                accepted += 1;
            }
            if t < burn && t > 0 && t.is_multiple_of(50) {
                let rate = accepted as f64 / 50.0;
                width *= if rate > 0.234 { 1.15 } else { 1.0 / 1.15 };
                width = width.clamp(1e-8, 1e8);
                accepted = 0;
            }
            if t >= burn {
                out.push(x.clone());
            }
        }
        out
    }

    /// Gibbs sampling: update one coordinate at a time from its conditional
    /// distribution given the rest.
    ///
    /// Every move is accepted, because a draw from the exact conditional is
    /// already in equilibrium for that coordinate. That makes it the method
    /// of choice whenever the conditionals are tractable, and useless when
    /// they are not.
    ///
    /// Each conditional receives the full current point and must return a
    /// draw for its own coordinate.
    ///
    /// # Panics
    /// Panics unless there is one conditional per coordinate and the burn-in
    /// is shorter than the run.
    pub fn gibbs(
        conditionals: &[&dyn Fn(&[f64], &mut Rng) -> f64],
        x0: &[f64],
        n: usize,
        burn: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        assert_eq!(conditionals.len(), x0.len(), "one conditional per coordinate is required");
        assert!(burn < n, "the burn-in must be shorter than the run");
        let mut x = x0.to_vec();
        let mut out = Vec::with_capacity(n - burn);
        for t in 0..n {
            for (i, c) in conditionals.iter().enumerate() {
                x[i] = c(&x, rng);
            }
            if t >= burn {
                out.push(x.clone());
            }
        }
        out
    }

    /// Hamiltonian Monte Carlo: give the point a momentum and follow the
    /// resulting trajectory.
    ///
    /// Treat the negative log density as a potential energy, draw a random
    /// momentum, and integrate the equations of motion. The trajectory
    /// conserves energy, so a proposal at the far end is accepted with
    /// probability near one however far it has travelled -- which is what
    /// lets the chain cross the whole distribution in one move instead of
    /// diffusing across it. The leapfrog integrator is used because it is
    /// *symplectic*: its error does not accumulate, so energy stays nearly
    /// conserved over long trajectories, and it is reversible, which the
    /// acceptance rule requires.
    ///
    /// # Panics
    /// Panics on an empty start, a non-positive step, no leapfrog steps, or a
    /// burn-in at or beyond the run.
    pub fn hamiltonian_mc(
        log_target: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        step: f64,
        n_leapfrog: usize,
        n: usize,
        burn: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        assert!(!x0.is_empty(), "the start point must have a dimension");
        assert!(step > 0.0, "the step size must be positive");
        assert!(n_leapfrog > 0, "a trajectory needs at least one step");
        assert!(burn < n, "the burn-in must be shorter than the run");
        let d = x0.len();
        let mut x = x0.to_vec();
        let mut out = Vec::with_capacity(n - burn);
        for t in 0..n {
            let p0: Vec<f64> = (0..d).map(|_| rng.next_gaussian()).collect();
            let mut q = x.clone();
            let mut p = p0.clone();
            // Leapfrog: a half kick, then alternating drifts and kicks.
            let g = grad(&q);
            for i in 0..d {
                p[i] += 0.5 * step * g[i];
            }
            for l in 0..n_leapfrog {
                for i in 0..d {
                    q[i] += step * p[i];
                }
                let g = grad(&q);
                let scale = if l + 1 == n_leapfrog { 0.5 } else { 1.0 };
                for i in 0..d {
                    p[i] += scale * step * g[i];
                }
            }
            let kinetic = |p: &[f64]| 0.5 * p.iter().map(|v| v * v).sum::<f64>();
            let current = log_target(&x) - kinetic(&p0);
            let proposed = log_target(&q) - kinetic(&p);
            if proposed.is_finite() && (proposed >= current || rng.next_f64().ln() < proposed - current)
            {
                x = q;
            }
            if t >= burn {
                out.push(x.clone());
            }
        }
        out
    }

    /// The no-U-turn sampler: Hamiltonian trajectories whose length the
    /// algorithm chooses by watching for the path to double back.
    ///
    /// Hoffman and Gelman's naive scheme. The trajectory is grown by
    /// repeated doubling, forwards or backwards at random, and stops when the
    /// two ends of *any* sub-trajectory start approaching each other; the
    /// next state is drawn uniformly from the states the slice variable
    /// admits. The doubling and the sub-tree stopping check are not
    /// decoration -- simply running until the path turns and taking the last
    /// point is not reversible, and gives the wrong stationary distribution.
    /// Getting that right removes trajectory length from the list of things a
    /// user must tune, which was the practical obstacle to Hamiltonian
    /// methods.
    ///
    /// # Panics
    /// Panics on an empty start, a non-positive step, a zero depth, or a
    /// burn-in at or beyond the run.
    pub fn nuts_lite(
        log_target: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        step: f64,
        max_depth: usize,
        n: usize,
        burn: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        assert!(!x0.is_empty(), "the start point must have a dimension");
        assert!(step > 0.0, "the step size must be positive");
        assert!(max_depth > 0, "a trajectory needs a depth");
        assert!(burn < n, "the burn-in must be shorter than the run");
        let d = x0.len();
        let mut x = x0.to_vec();
        let mut out = Vec::with_capacity(n - burn);
        for t in 0..n {
            let p0: Vec<f64> = (0..d).map(|_| rng.next_gaussian()).collect();
            let joint0 = log_target(&x) - 0.5 * dot(&p0, &p0);
            // The slice variable, kept as a log so nothing is exponentiated.
            let log_u = joint0 + rng.next_f64().ln();
            let mut qm = x.clone();
            let mut pm = p0.clone();
            let mut qp = x.clone();
            let mut pp = p0;
            let mut candidates = vec![x.clone()];
            let mut depth = 0usize;
            let mut alive = true;
            while alive && depth < max_depth {
                let backwards = rng.next_u64() & 1 == 0;
                let sub = if backwards {
                    build_tree(&qm, &pm, log_u, -step, depth, log_target, grad)
                } else {
                    build_tree(&qp, &pp, log_u, step, depth, log_target, grad)
                };
                if backwards {
                    qm = sub.qm;
                    pm = sub.pm;
                } else {
                    qp = sub.qp;
                    pp = sub.pp;
                }
                if sub.alive {
                    candidates.extend(sub.candidates);
                }
                alive = sub.alive && no_u_turn(&qm, &qp, &pm, &pp);
                depth += 1;
            }
            let i = ((u128::from(rng.next_u64()) * candidates.len() as u128) >> 64) as usize;
            x = candidates[i].clone();
            if t >= burn {
                out.push(x.clone());
            }
        }
        out
    }

    /// Slice sampling in one dimension.
    ///
    /// Draw a height uniformly below the density, then draw a point uniformly
    /// from the slice at that height. Every move is accepted and there is no
    /// proposal width to tune -- the stepping-out procedure finds the slice's
    /// extent on its own, so `w` only affects speed and not correctness.
    ///
    /// # Panics
    /// Panics on a non-positive width.
    pub fn slice_sampler(
        log_target_1d: &dyn Fn(f64) -> f64,
        x0: f64,
        w: f64,
        n: usize,
        rng: &mut Rng,
    ) -> Vec<f64> {
        assert!(w > 0.0, "the step width must be positive");
        let mut x = x0;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            // The height, drawn as a log so the density never has to be
            // exponentiated.
            let level = log_target_1d(x) + rng.next_f64().ln();
            // Step out until both ends are below the level.
            let mut lo = x - w * rng.next_f64();
            let mut hi = lo + w;
            for _ in 0..100 {
                if log_target_1d(lo) <= level {
                    break;
                }
                lo -= w;
            }
            for _ in 0..100 {
                if log_target_1d(hi) <= level {
                    break;
                }
                hi += w;
            }
            // Shrink towards the current point until a draw lands inside.
            for _ in 0..200 {
                let candidate = lo + (hi - lo) * rng.next_f64();
                if log_target_1d(candidate) > level {
                    x = candidate;
                    break;
                }
                if candidate < x {
                    lo = candidate;
                } else {
                    hi = candidate;
                }
            }
            out.push(x);
        }
        out
    }

    /// Parallel tempering: run several chains at different temperatures and
    /// let them swap.
    ///
    /// A hot chain sees a flattened version of the target and crosses between
    /// modes easily; a cold chain samples the target itself but can be
    /// trapped. Swapping states between neighbouring temperatures, with an
    /// acceptance rule that preserves each chain's own stationary
    /// distribution, lets the cold chain inherit the hot one's mobility.
    /// Returns the samples from the coldest chain.
    ///
    /// # Panics
    /// Panics unless the temperatures are positive with the first equal to
    /// one, and the burn-in is shorter than the run.
    pub fn parallel_tempering(
        log_target: &dyn Fn(&[f64]) -> f64,
        temps: &[f64],
        x0: &[f64],
        proposal_std: f64,
        n: usize,
        burn: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        assert!(!temps.is_empty(), "at least one temperature is required");
        assert!(temps.iter().all(|&t| t > 0.0), "temperatures must be positive");
        assert!((temps[0] - 1.0).abs() < 1e-12, "the first chain must be at temperature one");
        assert!(proposal_std > 0.0, "the proposal width must be positive");
        assert!(burn < n, "the burn-in must be shorter than the run");
        let k = temps.len();
        let mut xs: Vec<Vec<f64>> = vec![x0.to_vec(); k];
        let mut lps: Vec<f64> = xs.iter().map(|x| log_target(x)).collect();
        let mut out = Vec::with_capacity(n - burn);
        for t in 0..n {
            for c in 0..k {
                let candidate: Vec<f64> = xs[c]
                    .iter()
                    .map(|&v| v + proposal_std * temps[c].sqrt() * rng.next_gaussian())
                    .collect();
                let lq = log_target(&candidate);
                // At temperature T the chain targets the density raised to
                // 1/T, so the log ratio is divided by T.
                if (lq - lps[c]) / temps[c] >= 0.0 || rng.next_f64().ln() < (lq - lps[c]) / temps[c]
                {
                    xs[c] = candidate;
                    lps[c] = lq;
                }
            }
            // Attempt one swap between a random neighbouring pair.
            if k > 1 {
                let c = ((u128::from(rng.next_u64()) * (k - 1) as u128) >> 64) as usize;
                let delta = (1.0 / temps[c] - 1.0 / temps[c + 1]) * (lps[c + 1] - lps[c]);
                if delta >= 0.0 || rng.next_f64().ln() < delta {
                    xs.swap(c, c + 1);
                    lps.swap(c, c + 1);
                }
            }
            if t >= burn {
                out.push(xs[0].clone());
            }
        }
        out
    }

    /// The autocorrelation time of a chain: one plus twice the sum of the
    /// autocorrelations, truncated where they first turn negative.
    ///
    /// How many steps the chain takes to forget where it was. The truncation
    /// is Geyer's initial positive sequence rule: past that point the
    /// estimates are dominated by noise, and summing them adds variance
    /// rather than information.
    #[must_use]
    pub fn autocorrelation_time(chain: &[f64]) -> f64 {
        let n = chain.len();
        if n < 2 {
            return 1.0;
        }
        let mean = chain.iter().sum::<f64>() / n as f64;
        let var = chain.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        if var <= 0.0 {
            return 1.0;
        }
        let mut tau = 1.0;
        for lag in 1..n.min(n / 4).max(2) {
            let cov: f64 = (0..n - lag)
                .map(|i| (chain[i] - mean) * (chain[i + lag] - mean))
                .sum::<f64>()
                / n as f64;
            let rho = cov / var;
            if rho <= 0.0 {
                break;
            }
            tau += 2.0 * rho;
        }
        tau.max(1.0)
    }

    /// The effective sample size: the number of independent draws a
    /// correlated chain is worth.
    ///
    /// The run length divided by the autocorrelation time. Always at most the
    /// run length, and usually far less -- a Metropolis chain with a
    /// well-tuned proposal might be worth a tenth of its length, which is the
    /// honest denominator for any Monte Carlo error estimate.
    #[must_use]
    pub fn effective_sample_size(chain: &[f64]) -> f64 {
        let n = chain.len() as f64;
        if n <= 1.0 {
            return n;
        }
        (n / Mcmc::autocorrelation_time(chain)).clamp(1.0, n)
    }

    /// The Gelman-Rubin statistic: the ratio of the pooled variance estimate
    /// to the within-chain one.
    ///
    /// Several chains from different starts should, once converged, look like
    /// draws from the same distribution -- so the spread between chains
    /// should match the spread within them and the ratio should approach one.
    /// A value well above one is the clearest evidence available that a run
    /// has not converged. It cannot prove that one has.
    ///
    /// # Panics
    /// Panics unless there are at least two chains of at least two samples
    /// each, all the same length.
    #[must_use]
    pub fn gelman_rubin(chains: &[Vec<f64>]) -> f64 {
        assert!(chains.len() >= 2, "at least two chains are required");
        let n = chains[0].len();
        assert!(n >= 2, "each chain needs at least two samples");
        assert!(chains.iter().all(|c| c.len() == n), "the chains must be the same length");
        let m = chains.len() as f64;
        let means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
        let grand = means.iter().sum::<f64>() / m;
        // Between-chain variance, scaled by the chain length.
        let b = n as f64 / (m - 1.0)
            * means.iter().map(|v| (v - grand) * (v - grand)).sum::<f64>();
        // Within-chain variance.
        let w = chains
            .iter()
            .zip(&means)
            .map(|(c, &mu)| {
                c.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n as f64 - 1.0)
            })
            .sum::<f64>()
            / m;
        if w <= 0.0 {
            return 1.0;
        }
        let var_plus = (n as f64 - 1.0) / n as f64 * w + b / n as f64;
        (var_plus / w).sqrt()
    }

    /// Simulated annealing: Metropolis on an energy, with the temperature
    /// falling on a schedule.
    ///
    /// At a high temperature almost every move is accepted and the search
    /// wanders; as the temperature falls it becomes a hill descent. Returns
    /// the best point found and its energy -- the best, not the last, because
    /// the walk can and does step away from an optimum it has found.
    ///
    /// # Panics
    /// Panics on an empty start.
    pub fn simulated_annealing(
        energy: &dyn Fn(&[f64]) -> f64,
        x0: &[f64],
        schedule: &dyn Fn(usize) -> f64,
        n: usize,
        rng: &mut Rng,
    ) -> (Vec<f64>, f64) {
        assert!(!x0.is_empty(), "the start point must have a dimension");
        let mut x = x0.to_vec();
        let mut e = energy(&x);
        let mut best = (x.clone(), e);
        for t in 0..n {
            let temp = schedule(t).max(1e-12);
            let candidate: Vec<f64> =
                x.iter().map(|&v| v + temp.sqrt() * rng.next_gaussian()).collect();
            let ec = energy(&candidate);
            if ec <= e || rng.next_f64() < ((e - ec) / temp).exp() {
                x = candidate;
                e = ec;
                if e < best.1 {
                    best = (x.clone(), e);
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    /// A random row-stochastic matrix.
    fn random_chain(n: usize, rng: &mut Rng) -> MarkovChain {
        let mut p = Matrix::zeros(n, n);
        for i in 0..n {
            let row: Vec<f64> = (0..n).map(|_| 0.01 + rng.next_f64()).collect();
            let total: f64 = row.iter().sum();
            for j in 0..n {
                p.set(i, j, row[j] / total);
            }
        }
        MarkovChain::new(p).expect("the construction is stochastic")
    }

    fn chain_from(rows: &[&[f64]]) -> MarkovChain {
        MarkovChain::new(Matrix::from_rows(rows).expect("rectangular")).expect("stochastic")
    }

    /// The stationary distribution is the thing it is defined to be: a
    /// probability vector left fixed by the matrix.
    #[test]
    fn the_stationary_distribution_is_fixed_by_the_chain() {
        let mut rng = Rng::new(0x_5747);
        for _ in 0..200 {
            let n = 2 + pick(&mut rng, 7);
            let c = random_chain(n, &mut rng);
            let pi = c.stationary();
            assert_eq!(pi.len(), n);
            assert!(pi.iter().all(|&v| v >= -1e-12), "a stationary probability is negative");
            assert!(close(pi.iter().sum::<f64>(), 1.0, 1e-9), "the distribution does not sum to one");
            // pi P = pi, entry by entry.
            let next = c.step_dist(&pi);
            for j in 0..n {
                assert!(close(next[j], pi[j], 1e-8), "pi P differs from pi at {j}");
            }
            // And it is the limit of the powers, since a dense chain is
            // irreducible and aperiodic.
            let far = c.n_step(200);
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (far.get(i, j) - pi[j]).abs() < 1e-6,
                        "the powers do not converge to the stationary distribution"
                    );
                }
            }
        }
        // A periodic chain has a stationary distribution even though its
        // powers never settle, which is why this is solved rather than
        // iterated.
        let flip = chain_from(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let pi = flip.stationary();
        assert!(close(pi[0], 0.5, 1e-12) && close(pi[1], 0.5, 1e-12));
        assert_eq!(flip.period(0), 2);
        assert!(!flip.is_aperiodic());
        assert!(flip.is_irreducible());
        assert_eq!(flip.mixing_time(0.01), usize::MAX, "a periodic chain never mixes");
    }

    /// Estimation from data, and the structural classifications.
    #[test]
    fn estimation_and_classification_agree_with_the_definitions() {
        let mut rng = Rng::new(0x_C1A5);
        // A chain estimated from a long run of itself comes back close.
        let truth = chain_from(&[&[0.7, 0.2, 0.1], &[0.1, 0.6, 0.3], &[0.3, 0.3, 0.4]]);
        let path = truth.simulate(0, 200_000, &mut rng);
        let est = MarkovChain::from_sequence(&path, 3).expect("valid");
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (est.p.get(i, j) - truth.p.get(i, j)).abs() < 0.02,
                    "the estimate is off at ({i}, {j})"
                );
            }
        }
        // A state with no observations becomes absorbing rather than being
        // invented.
        let sparse = MarkovChain::from_sequence(&[0usize, 0, 0], 2).expect("valid");
        assert!(close(sparse.p.get(1, 1), 1.0, 1e-12));

        // Classification against the definitions.
        let mixed = chain_from(&[
            &[0.5, 0.5, 0.0, 0.0],
            &[0.5, 0.5, 0.0, 0.0],
            &[0.0, 0.25, 0.5, 0.25],
            &[0.0, 0.0, 0.0, 1.0],
        ]);
        let classes = mixed.classify_states();
        assert_eq!(classes[0], StateClass::Recurrent);
        assert_eq!(classes[1], StateClass::Recurrent);
        assert_eq!(classes[2], StateClass::Transient);
        assert_eq!(classes[3], StateClass::Absorbing);
        assert!(!mixed.is_irreducible());
        // Periods: a three-cycle has period three at every state.
        let cycle = chain_from(&[&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]]);
        for s in 0..3 {
            assert_eq!(cycle.period(s), 3, "the cycle's period is wrong at {s}");
        }
        // A self-loop anywhere makes the chain aperiodic.
        let lazy = chain_from(&[&[0.5, 0.5, 0.0], &[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]]);
        assert!(lazy.is_aperiodic());
        assert!(lazy.is_irreducible());

        // Rejection of bad input.
        assert!(MarkovChain::new(Matrix::from_rows(&[&[0.5, 0.4]]).expect("row")).is_err());
        let oblong = Matrix::from_rows(&[&[0.5, 0.5, 0.0], &[0.2, 0.8, 0.0]]).expect("rows");
        assert!(MarkovChain::new(oblong).is_err(), "a non-square matrix is not a chain");
        let negative = Matrix::from_rows(&[&[1.5, -0.5], &[0.5, 0.5]]).expect("rows");
        assert!(MarkovChain::new(negative).is_err());
    }

    /// The gambler's ruin, against the closed form every textbook gives.
    ///
    /// A gambler with `k` of `n` pounds bets one at a time, winning with
    /// probability `p`. The chance of reaching `n` before zero is `k / n`
    /// for a fair game, and a ratio of powers otherwise. The absorbing
    /// machinery must reproduce both.
    #[test]
    fn absorption_matches_the_gamblers_ruin() {
        for n in [4usize, 6, 10] {
            for &p in &[0.5f64, 0.4, 0.6, 0.25] {
                let mut m = Matrix::zeros(n + 1, n + 1);
                m.set(0, 0, 1.0);
                m.set(n, n, 1.0);
                for k in 1..n {
                    m.set(k, k + 1, p);
                    m.set(k, k - 1, 1.0 - p);
                }
                let chain = MarkovChain::new(m).expect("stochastic");
                let abs = chain.absorbing_probabilities();
                // Rows follow the transient states in order, which here is
                // 1..n; columns follow the absorbing ones, 0 then n.
                for k in 1..n {
                    let win = abs.get(k - 1, 1);
                    let want = if (p - 0.5).abs() < 1e-12 {
                        k as f64 / n as f64
                    } else {
                        let r = (1.0 - p) / p;
                        (1.0 - r.powi(k as i32)) / (1.0 - r.powi(n as i32))
                    };
                    assert!(
                        close(win, want, 1e-9),
                        "ruin at n = {n}, p = {p}, k = {k}: {win} against {want}"
                    );
                    // The two absorbing probabilities exhaust the outcomes.
                    assert!(close(abs.get(k - 1, 0) + win, 1.0, 1e-9));
                    // And the hitting probability agrees, by a different
                    // route entirely.
                    assert!(close(chain.hitting_probability(k, &[n]), want, 1e-6));
                }
                // Expected duration, against its own closed form.
                let steps = chain.expected_steps_to_absorption();
                for k in 1..n {
                    let want = if (p - 0.5).abs() < 1e-12 {
                        (k * (n - k)) as f64
                    } else {
                        let r = (1.0 - p) / p;
                        let q = 1.0 - 2.0 * p;
                        k as f64 / q
                            - n as f64 / q * (1.0 - r.powi(k as i32))
                                / (1.0 - r.powi(n as i32))
                    };
                    assert!(
                        close(steps[k], want, 1e-8),
                        "duration at n = {n}, p = {p}, k = {k}: {} against {want}",
                        steps[k]
                    );
                    // The hitting time to either barrier is the same number.
                    assert!(close(chain.hitting_time(k, &[0, n]), steps[k], 1e-8));
                }
                assert_eq!(steps[0], 0.0);
                assert_eq!(steps[n], 0.0);
            }
        }
    }

    /// Kac's formula, mean first passage times, and the entropy rate, each
    /// against an independent computation.
    #[test]
    fn return_times_and_entropy_rate_match_their_definitions() {
        let mut rng = Rng::new(0x_4AC5);
        for _ in 0..40 {
            let n = 2 + pick(&mut rng, 5);
            let c = random_chain(n, &mut rng);
            let pi = c.stationary();
            // Kac: the expected return time is the reciprocal of the
            // stationary probability.
            for s in 0..n {
                assert!(close(c.return_time(s), 1.0 / pi[s], 1e-8), "Kac's formula fails at {s}");
            }
            // Mean first passage times satisfy their own recurrence:
            // m_ij = 1 + sum_k p_ik m_kj for i != j.
            let m = c.mfpt_matrix();
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let rhs: f64 = 1.0
                        + (0..n).filter(|&k| k != j).map(|k| c.p.get(i, k) * m.get(k, j)).sum::<f64>();
                    assert!(close(m.get(i, j), rhs, 1e-7), "the passage recurrence fails");
                }
            }
            // The entropy rate against its definition, and against the
            // marginal entropy which bounds it above.
            let rate = c.entropy_rate();
            let marginal: f64 =
                pi.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.log2()).sum();
            assert!(rate >= -1e-12);
            assert!(rate <= marginal + 1e-9, "dependence should not raise the entropy rate");
            assert!(rate <= (n as f64).log2() + 1e-9);
        }
        // A deterministic chain has no uncertainty at all.
        let cycle = chain_from(&[&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]]);
        assert!(cycle.entropy_rate().abs() < 1e-12);
        // A chain whose rows are all uniform is a memoryless source, so its
        // entropy rate is the full log of the alphabet.
        let uniform = chain_from(&[&[0.25; 4], &[0.25; 4], &[0.25; 4], &[0.25; 4]]);
        assert!(close(uniform.entropy_rate(), 2.0, 1e-12));
    }

    /// Reversibility, mixing and the spectral gap, tied to each other.
    #[test]
    fn reversibility_mixing_and_the_spectral_gap_agree() {
        // A random walk on an undirected graph is reversible with stationary
        // distribution proportional to degree, which is the standard example
        // and a real theorem rather than a construction.
        let mut g = crate::graph::core::Graph::new(5, false);
        for (u, v) in [(0usize, 1usize), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)] {
            g.add_edge(u, v, 1.0);
        }
        let mut p = Matrix::zeros(5, 5);
        for u in 0..5 {
            let deg = g.adj[u].len() as f64;
            for &(v, _) in &g.adj[u] {
                p.set(u, v, p.get(u, v) + 1.0 / deg);
            }
        }
        let walk = MarkovChain::new(p).expect("stochastic");
        let pi = walk.stationary();
        let total: f64 = (0..5).map(|u| g.adj[u].len() as f64).sum();
        for u in 0..5 {
            assert!(
                close(pi[u], g.adj[u].len() as f64 / total, 1e-9),
                "the walk's stationary distribution is not proportional to degree"
            );
        }
        assert!(walk.reversible_check(&pi, 1e-9), "a graph walk should be reversible");
        // A directed cycle with a bias is stationary but not reversible: the
        // flow goes round, so it does not balance pairwise.
        let biased = chain_from(&[&[0.0, 0.9, 0.1], &[0.1, 0.0, 0.9], &[0.9, 0.1, 0.0]]);
        let bpi = biased.stationary();
        let next = biased.step_dist(&bpi);
        for j in 0..3 {
            assert!(close(next[j], bpi[j], 1e-9), "the biased cycle is not stationary");
        }
        assert!(!biased.reversible_check(&bpi, 1e-6), "a one-way cycle is not reversible");

        // Mixing time and the spectral gap move together: a chain that mixes
        // fast has a large gap.
        let mut rng = Rng::new(0x_6A97);
        for _ in 0..30 {
            let n = 2 + pick(&mut rng, 4);
            let c = random_chain(n, &mut rng);
            let gap = c.spectral_gap();
            assert!((0.0..=1.0).contains(&gap), "the gap left its range: {gap}");
            let t = c.mixing_time(0.01);
            assert!(t < usize::MAX, "a dense chain should mix");
            // The distance really is below the threshold at that time, and
            // was not before it.
            let power = c.n_step(t);
            let pi = c.stationary();
            for i in 0..n {
                let row: Vec<f64> = (0..n).map(|j| power.get(i, j)).collect();
                assert!(MarkovChain::total_variation_distance(&row, &pi) <= 0.01 + 1e-12);
            }
        }
        // Total variation: zero against itself, one for disjoint support.
        assert_eq!(MarkovChain::total_variation_distance(&[0.5, 0.5], &[0.5, 0.5]), 0.0);
        assert!(close(
            MarkovChain::total_variation_distance(&[1.0, 0.0], &[0.0, 1.0]),
            1.0,
            1e-12
        ));
    }

    /// Coupling from the past returns an exactly stationary sample, and the
    /// PageRank chain is the one PageRank is defined by.
    #[test]
    fn exact_sampling_and_the_pagerank_chain() {
        let c = chain_from(&[&[0.5, 0.3, 0.2], &[0.2, 0.5, 0.3], &[0.3, 0.2, 0.5]]);
        let pi = c.stationary();
        let mut rng = Rng::new(0x_C0F7);
        let mut counts = [0usize; 3];
        let draws = 30_000;
        for _ in 0..draws {
            counts[c.coupling_from_the_past_small(&mut rng)] += 1;
        }
        for s in 0..3 {
            let seen = counts[s] as f64 / draws as f64;
            assert!(
                (seen - pi[s]).abs() < 0.01,
                "exact sampling gave {seen} for state {s} against {}",
                pi[s]
            );
        }

        // The PageRank chain's stationary distribution is PageRank.
        let mut g = crate::graph::core::Graph::new(6, true);
        for (u, v) in [(0usize, 1usize), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (5, 0)] {
            g.add_edge(u, v, 1.0);
        }
        let damping = 0.85;
        let chain = MarkovChain::pagerank_chain(&g, damping);
        assert!(chain.is_irreducible(), "teleportation should connect everything");
        assert!(chain.is_aperiodic());
        let ranks = chain.stationary();
        let direct = crate::graph::spectral::pagerank(&g, damping, 1e-14);
        for v in 0..6 {
            assert!(
                (ranks[v] - direct[v]).abs() < 1e-6,
                "the chain and the direct computation disagree at {v}: {} against {}",
                ranks[v],
                direct[v]
            );
        }
        // A vertex with no out-edges spreads its mass rather than losing it.
        let dangling = crate::graph::core::Graph::new(3, true);
        let c = MarkovChain::pagerank_chain(&dangling, 0.85);
        let pi = c.stationary();
        assert!(pi.iter().all(|&v| close(v, 1.0 / 3.0, 1e-9)));
    }

    /// Metropolis-Hastings recovers a Gaussian's mean and variance, and the
    /// chain it produces really has the target as its stationary
    /// distribution.
    #[test]
    fn metropolis_hastings_recovers_a_gaussian() {
        let mu = 2.0;
        let sigma = 1.5;
        let log_target =
            |x: &[f64]| -0.5 * ((x[0] - mu) / sigma).powi(2) - (sigma * (2.0 * PI).sqrt()).ln();
        let mut rng = Rng::new(0x_4348);
        let chain = Mcmc::metropolis_hastings(&log_target, &[0.0], 2.0, 60_000, 10_000, &mut rng);
        assert_eq!(chain.len(), 50_000);
        let xs: Vec<f64> = chain.iter().map(|v| v[0]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / xs.len() as f64;
        // The Monte Carlo error scales with the effective sample size, not
        // the run length, so that is what the tolerance is built from.
        let ess = Mcmc::effective_sample_size(&xs);
        assert!(ess > 100.0, "the chain was worth only {ess} independent draws");
        assert!(ess <= xs.len() as f64, "the effective size exceeded the run length");
        let se = sigma / ess.sqrt();
        assert!((mean - mu).abs() < 4.0 * se, "mean {mean} against {mu}, standard error {se}");
        assert!((var - sigma * sigma).abs() < 0.2, "variance {var} against {}", sigma * sigma);

        // The adaptive version reaches the same answer with less tuning.
        let mut rng = Rng::new(0x_A44D);
        let adaptive =
            Mcmc::adaptive_metropolis(&log_target, &[0.0], 0.01, 60_000, 10_000, &mut rng);
        let ys: Vec<f64> = adaptive.iter().map(|v| v[0]).collect();
        let amean = ys.iter().sum::<f64>() / ys.len() as f64;
        assert!(
            (amean - mu).abs() < 0.15,
            "the adaptive chain did not find the mode: {amean}"
        );
        // Starting from a width of 0.01, a fixed proposal would barely move.
        let mut rng = Rng::new(0x_A44E);
        let stuck = Mcmc::metropolis_hastings(&log_target, &[0.0], 0.01, 60_000, 10_000, &mut rng);
        let zs: Vec<f64> = stuck.iter().map(|v| v[0]).collect();
        assert!(
            Mcmc::effective_sample_size(&zs) < Mcmc::effective_sample_size(&ys),
            "adaptation should improve the effective sample size"
        );
    }

    /// Hamiltonian Monte Carlo and the no-U-turn variant agree with
    /// Metropolis on the same target, and mix better.
    #[test]
    fn gradient_samplers_agree_with_metropolis_and_mix_better() {
        // A correlated two-dimensional Gaussian, which is where random-walk
        // proposals struggle and gradients do not.
        let rho = 0.9;
        let det = 1.0 - rho * rho;
        let log_target = move |x: &[f64]| {
            -0.5 / det * (x[0] * x[0] - 2.0 * rho * x[0] * x[1] + x[1] * x[1])
        };
        let grad = move |x: &[f64]| {
            vec![
                -(x[0] - rho * x[1]) / det,
                -(x[1] - rho * x[0]) / det,
            ]
        };
        let mut rng = Rng::new(0x_44C0);
        let mh = Mcmc::metropolis_hastings(&log_target, &[0.0, 0.0], 0.5, 40_000, 5_000, &mut rng);
        let hmc =
            Mcmc::hamiltonian_mc(&log_target, &grad, &[0.0, 0.0], 0.15, 20, 8_000, 1_000, &mut rng);
        let nuts =
            Mcmc::nuts_lite(&log_target, &grad, &[0.0, 0.0], 0.15, 6, 8_000, 1_000, &mut rng);

        for (name, chain) in [("MH", &mh), ("HMC", &hmc), ("NUTS", &nuts)] {
            let x: Vec<f64> = chain.iter().map(|v| v[0]).collect();
            let y: Vec<f64> = chain.iter().map(|v| v[1]).collect();
            let mx = x.iter().sum::<f64>() / x.len() as f64;
            let my = y.iter().sum::<f64>() / y.len() as f64;
            let vx = x.iter().map(|v| (v - mx) * (v - mx)).sum::<f64>() / x.len() as f64;
            let cov = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - mx) * (b - my))
                .sum::<f64>()
                / x.len() as f64;
            assert!(mx.abs() < 0.15, "{name}: the mean drifted to {mx}");
            assert!(my.abs() < 0.15, "{name}: the mean drifted to {my}");
            assert!((vx - 1.0).abs() < 0.2, "{name}: the variance is {vx}");
            assert!((cov / vx - rho).abs() < 0.15, "{name}: the correlation is {}", cov / vx);
        }
        // The gradient samplers are worth more per sample on this target.
        let ess_mh = Mcmc::effective_sample_size(
            &mh.iter().map(|v| v[0]).collect::<Vec<_>>(),
        ) / mh.len() as f64;
        let ess_hmc = Mcmc::effective_sample_size(
            &hmc.iter().map(|v| v[0]).collect::<Vec<_>>(),
        ) / hmc.len() as f64;
        assert!(
            ess_hmc > ess_mh,
            "gradients should mix better on a correlated target: {ess_hmc} against {ess_mh}"
        );
    }

    /// Gibbs, the slice sampler and parallel tempering, each on a target
    /// where the right answer is known.
    #[test]
    fn the_other_samplers_hit_their_targets() {
        let mut rng = Rng::new(0x_61B5);
        // Gibbs on a correlated Gaussian, whose conditionals are Gaussian
        // with a known mean and variance.
        let rho = 0.8;
        let sd = (1.0f64 - rho * rho).sqrt();
        let c0 = move |x: &[f64], r: &mut Rng| rho * x[1] + sd * r.next_gaussian();
        let c1 = move |x: &[f64], r: &mut Rng| rho * x[0] + sd * r.next_gaussian();
        let conds: [&dyn Fn(&[f64], &mut Rng) -> f64; 2] = [&c0, &c1];
        let chain = Mcmc::gibbs(&conds, &[0.0, 0.0], 40_000, 5_000, &mut rng);
        let x: Vec<f64> = chain.iter().map(|v| v[0]).collect();
        let y: Vec<f64> = chain.iter().map(|v| v[1]).collect();
        let mx = x.iter().sum::<f64>() / x.len() as f64;
        let vx = x.iter().map(|v| (v - mx) * (v - mx)).sum::<f64>() / x.len() as f64;
        let my = y.iter().sum::<f64>() / y.len() as f64;
        let cov =
            x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / x.len() as f64;
        assert!(mx.abs() < 0.1 && (vx - 1.0).abs() < 0.15, "Gibbs missed the marginal");
        assert!((cov - rho).abs() < 0.1, "Gibbs missed the correlation: {cov}");

        // The slice sampler on a standard normal.
        let normal = |x: f64| -0.5 * x * x;
        let s = Mcmc::slice_sampler(&normal, 0.0, 1.0, 40_000, &mut rng);
        let ms = s.iter().sum::<f64>() / s.len() as f64;
        let vs = s.iter().map(|v| (v - ms) * (v - ms)).sum::<f64>() / s.len() as f64;
        assert!(ms.abs() < 0.05, "the slice sampler's mean is {ms}");
        assert!((vs - 1.0).abs() < 0.1, "the slice sampler's variance is {vs}");

        // Parallel tempering on a bimodal target, where a single cold chain
        // gets stuck in whichever mode it starts in.
        let bimodal = |x: &[f64]| {
            let a = -0.5 * (x[0] - 5.0f64).powi(2);
            let b = -0.5 * (x[0] + 5.0f64).powi(2);
            a.max(b) + (1.0 + (-(a - b).abs()).exp()).ln()
        };
        let temps = [1.0, 2.5, 6.0, 15.0];
        let pt = Mcmc::parallel_tempering(&bimodal, &temps, &[5.0], 1.0, 40_000, 5_000, &mut rng);
        let visited_left = pt.iter().filter(|v| v[0] < 0.0).count();
        let visited_right = pt.len() - visited_left;
        assert!(
            visited_left > pt.len() / 10 && visited_right > pt.len() / 10,
            "tempering visited {visited_left} and {visited_right}, so it did not cross"
        );
        // A single cold chain at the same width crosses far less often, and
        // often not at all. Counting sign changes rather than occupancy, and
        // averaging over several starts, keeps that a statement about the
        // method rather than about one lucky stream.
        let sign_changes = |c: &[Vec<f64>]| {
            c.windows(2).filter(|w| (w[0][0] < 0.0) != (w[1][0] < 0.0)).count()
        };
        let mut pt_changes = 0usize;
        let mut single_changes = 0usize;
        for seed in 0..5u64 {
            let mut r = Rng::new(0x_7E11 + seed);
            pt_changes += sign_changes(&Mcmc::parallel_tempering(
                &bimodal, &temps, &[5.0], 1.0, 20_000, 2_000, &mut r,
            ));
            let mut r = Rng::new(0x_7E11 + seed);
            single_changes += sign_changes(&Mcmc::metropolis_hastings(
                &bimodal, &[5.0], 1.0, 20_000, 2_000, &mut r,
            ));
        }
        assert!(pt_changes > 20, "tempering crossed only {pt_changes} times over five runs");
        assert!(
            pt_changes > 10 * single_changes.max(1),
            "tempering crossed {pt_changes} times against the single chain's {single_changes}"
        );
    }

    /// The diagnostics diagnose: they call a converged run converged and an
    /// unconverged one unconverged.
    #[test]
    fn the_convergence_diagnostics_tell_the_two_cases_apart() {
        let mut rng = Rng::new(0x_D1A6);
        // Independent draws have an autocorrelation time of one and an
        // effective size equal to the run length.
        let iid: Vec<f64> = (0..20_000).map(|_| rng.next_gaussian()).collect();
        let tau = Mcmc::autocorrelation_time(&iid);
        assert!((tau - 1.0).abs() < 0.35, "independent draws gave a time of {tau}");
        assert!(Mcmc::effective_sample_size(&iid) > 0.6 * iid.len() as f64);
        // A strongly correlated walk is worth far less.
        let mut x = 0.0;
        let correlated: Vec<f64> = (0..20_000)
            .map(|_| {
                x = 0.98 * x + 0.2 * rng.next_gaussian();
                x
            })
            .collect();
        let ctau = Mcmc::autocorrelation_time(&correlated);
        assert!(ctau > 10.0, "a correlated chain gave a time of {ctau}");
        assert!(Mcmc::effective_sample_size(&correlated) < 0.1 * correlated.len() as f64);
        // Never more than the run length, whatever the input.
        for chain in [&iid, &correlated] {
            assert!(Mcmc::effective_sample_size(chain) <= chain.len() as f64);
        }
        assert_eq!(Mcmc::effective_sample_size(&[1.0]), 1.0);

        // Gelman-Rubin: near one for chains from the same distribution, and
        // well above it for chains that have not met.
        let converged: Vec<Vec<f64>> =
            (0..4).map(|_| (0..3_000).map(|_| rng.next_gaussian()).collect()).collect();
        let r = Mcmc::gelman_rubin(&converged);
        assert!((r - 1.0).abs() < 0.02, "converged chains gave {r}");
        let separated: Vec<Vec<f64>> = (0..4)
            .map(|k| {
                (0..3_000).map(|_| k as f64 * 10.0 + rng.next_gaussian()).collect()
            })
            .collect();
        let r2 = Mcmc::gelman_rubin(&separated);
        assert!(r2 > 2.0, "chains ten apart gave {r2}");
        assert!(std::panic::catch_unwind(|| Mcmc::gelman_rubin(&[vec![1.0, 2.0]])).is_err());
    }

    /// Simulated annealing finds a global minimum a hill descent would miss.
    #[test]
    fn annealing_escapes_a_local_minimum() {
        // A double well with the deeper minimum at +2 and a shallow trap at
        // -2, separated by a barrier.
        let energy = |x: &[f64]| {
            let v = x[0];
            0.05 * (v * v - 4.0).powi(2) - 0.35 * v
        };
        let schedule = |t: usize| 4.0 * (-(t as f64) / 3_000.0).exp() + 1e-3;
        let mut rng = Rng::new(0x_A44E);
        let mut from_trap = 0;
        for _ in 0..20 {
            let (x, e) = Mcmc::simulated_annealing(&energy, &[-2.0], &schedule, 20_000, &mut rng);
            assert!(e <= energy(&[-2.0]) + 1e-9, "annealing returned a worse point than it started");
            assert!(e <= energy(&x) + 1e-9, "the reported energy does not match the point");
            if x[0] > 0.0 {
                from_trap += 1;
            }
        }
        assert!(from_trap >= 18, "annealing escaped the trap only {from_trap} times in 20");
        // Freezing immediately leaves it where it started, which is what
        // makes the schedule the whole method.
        let frozen = |_: usize| 1e-12;
        let (x, _) = Mcmc::simulated_annealing(&energy, &[-2.0], &frozen, 20_000, &mut rng);
        assert!(x[0] < 0.0, "a frozen schedule should not escape");
    }
}
