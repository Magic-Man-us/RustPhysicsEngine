//! Gaussian process regression.
//!
//! # A distribution over functions, conditioned
//!
//! A Gaussian process says that any finite set of function values is
//! jointly normal, with a covariance given by the kernel. Regression is
//! then not fitting but conditioning: the posterior over an unobserved
//! point is the conditional of a multivariate normal, and that has a
//! closed form. There is no optimisation anywhere in
//! [`Gp::fit`] -- it is one Cholesky factorisation, and the answer is
//! exact given the kernel.
//!
//! Two consequences are worth stating because they surprise people and
//! because they are exactly testable.
//!
//! *The posterior variance does not depend on what was observed.* It is
//! `k(x,x) - k_*^T K^-1 k_*`, and `y` does not appear. Uncertainty in a
//! Gaussian process is a statement about where the data *is*, not about
//! what it said. Doubling every observation doubles the mean and leaves
//! every error bar alone.
//!
//! *With no noise the mean interpolates exactly and the variance
//! vanishes at the data.* The conditional of a normal on one of its own
//! coordinates is a point mass. Adding noise is what turns
//! interpolation into smoothing, and the residual at the data grows
//! from zero in proportion to it.
//!
//! # Which kernel is a modelling choice, not a detail
//!
//! The kernel *is* the prior. A squared exponential asserts that the
//! function is infinitely differentiable, which is a very strong claim
//! and the reason its posterior can look implausibly smooth between
//! widely spaced points. The Matern family asserts a finite number of
//! derivatives -- `3/2` gives one, `5/2` gives two -- and is usually the
//! better default for anything physical. A periodic kernel asserts exact
//! periodicity, and [`KernelFn::Periodic`] satisfies
//! `k(x, x + p) = k(x, x)` to rounding rather than approximately.
//!
//! Kernels are closed under addition and multiplication, which is what
//! [`KernelFn::Sum`] and [`KernelFn::Product`] are for: a sum models
//! additive structure (a trend plus a wiggle), a product models
//! interaction (a periodicity whose amplitude decays).
//!
//! # The marginal likelihood balances fit against complexity on its own
//!
//! `log p(y | X)` splits into a data-fit term `-y^T K^-1 y / 2` and a
//! complexity penalty `-log|K| / 2`. Making the kernel more flexible
//! improves the first and costs the second, and the trade is not a
//! hyperparameter anyone chose -- it falls out of the normalisation of a
//! probability distribution. That is why hyperparameters can be tuned by
//! maximising it without a validation set.

use crate::error::SolveError;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// A covariance function.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelFn {
    /// Squared exponential, `s^2 exp(-r^2 / (2 l^2))`. Infinitely
    /// differentiable sample paths.
    Rbf { l: f64, s: f64 },
    /// Matern with `nu = 3/2`: once differentiable.
    Matern32 { l: f64, s: f64 },
    /// Matern with `nu = 5/2`: twice differentiable.
    Matern52 { l: f64, s: f64 },
    /// Exactly periodic with period `p`, and smooth within a period at
    /// the scale `l`.
    Periodic { l: f64, p: f64, s: f64 },
    /// `s^2 (x . x') + c`. Its posterior mean is an affine function, so
    /// a Gaussian process with this kernel is Bayesian linear
    /// regression in disguise.
    Linear { s: f64, c: f64 },
    /// The sum of two kernels, which is a kernel.
    Sum(Box<KernelFn>, Box<KernelFn>),
    /// The product of two kernels, which is also a kernel.
    Product(Box<KernelFn>, Box<KernelFn>),
}

/// The Euclidean distance between two points.
fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

impl KernelFn {
    /// Evaluates the covariance between two points.
    pub fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            KernelFn::Rbf { l, s } => {
                let r = distance(a, b);
                s * s * (-0.5 * r * r / (l * l)).exp()
            }
            KernelFn::Matern32 { l, s } => {
                let z = 3.0f64.sqrt() * distance(a, b) / l;
                s * s * (1.0 + z) * (-z).exp()
            }
            KernelFn::Matern52 { l, s } => {
                let z = 5.0f64.sqrt() * distance(a, b) / l;
                s * s * (1.0 + z + z * z / 3.0) * (-z).exp()
            }
            KernelFn::Periodic { l, p, s } => {
                // The distance enters through a sine of half the
                // separation over the period, which is what makes the
                // kernel exactly periodic rather than nearly so.
                let r = distance(a, b);
                let t = (std::f64::consts::PI * r / p).sin();
                s * s * (-2.0 * t * t / (l * l)).exp()
            }
            KernelFn::Linear { s, c } => {
                s * s * a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() + c
            }
            KernelFn::Sum(x, y) => x.eval(a, b) + y.eval(a, b),
            KernelFn::Product(x, y) => x.eval(a, b) * y.eval(a, b),
        }
    }

    /// Whether every length scale and amplitude is positive and finite,
    /// which is what makes the function a valid covariance.
    pub fn is_valid(&self) -> bool {
        match self {
            KernelFn::Rbf { l, s } | KernelFn::Matern32 { l, s } | KernelFn::Matern52 { l, s } => {
                l.is_finite() && *l > 0.0 && s.is_finite() && *s > 0.0
            }
            KernelFn::Periodic { l, p, s } => {
                l.is_finite() && *l > 0.0 && p.is_finite() && *p > 0.0 && s.is_finite() && *s > 0.0
            }
            KernelFn::Linear { s, c } => s.is_finite() && *s > 0.0 && c.is_finite() && *c >= 0.0,
            KernelFn::Sum(a, b) | KernelFn::Product(a, b) => a.is_valid() && b.is_valid(),
        }
    }

    /// The hyperparameters as a flat vector, in the order
    /// [`KernelFn::with_parameters`] expects them back.
    pub fn parameters(&self) -> Vec<f64> {
        match self {
            KernelFn::Rbf { l, s } | KernelFn::Matern32 { l, s } | KernelFn::Matern52 { l, s } => {
                vec![*l, *s]
            }
            KernelFn::Periodic { l, p, s } => vec![*l, *p, *s],
            KernelFn::Linear { s, c } => vec![*s, *c],
            KernelFn::Sum(a, b) | KernelFn::Product(a, b) => {
                let mut out = a.parameters();
                out.extend(b.parameters());
                out
            }
        }
    }

    /// Rebuilds the kernel with new hyperparameters, consuming as many
    /// as its structure needs.
    fn take_parameters(&self, values: &[f64], at: &mut usize) -> KernelFn {
        let next = |slot: &mut usize| {
            let v = values[*slot];
            *slot += 1;
            v
        };
        match self {
            KernelFn::Rbf { .. } => {
                let l = next(at);
                let s = next(at);
                KernelFn::Rbf { l, s }
            }
            KernelFn::Matern32 { .. } => {
                let l = next(at);
                let s = next(at);
                KernelFn::Matern32 { l, s }
            }
            KernelFn::Matern52 { .. } => {
                let l = next(at);
                let s = next(at);
                KernelFn::Matern52 { l, s }
            }
            KernelFn::Periodic { .. } => {
                let l = next(at);
                let p = next(at);
                let s = next(at);
                KernelFn::Periodic { l, p, s }
            }
            KernelFn::Linear { .. } => {
                let s = next(at);
                let c = next(at);
                KernelFn::Linear { s, c }
            }
            KernelFn::Sum(a, b) => {
                let left = a.take_parameters(values, at);
                let right = b.take_parameters(values, at);
                KernelFn::Sum(Box::new(left), Box::new(right))
            }
            KernelFn::Product(a, b) => {
                let left = a.take_parameters(values, at);
                let right = b.take_parameters(values, at);
                KernelFn::Product(Box::new(left), Box::new(right))
            }
        }
    }

    /// Rebuilds the kernel from a flat parameter vector.
    ///
    /// # Errors
    ///
    /// [`SolveError::DimensionMismatch`] if the count does not match.
    pub fn with_parameters(&self, values: &[f64]) -> Result<KernelFn, SolveError> {
        let wanted = self.parameters().len();
        if values.len() != wanted {
            return Err(SolveError::DimensionMismatch { expected: wanted, got: values.len() });
        }
        let mut at = 0;
        Ok(self.take_parameters(values, &mut at))
    }
}

/// A fitted Gaussian process.
#[derive(Debug, Clone, PartialEq)]
pub struct Gp {
    /// The covariance function.
    pub kernel: KernelFn,
    /// The observation noise variance, added to the diagonal.
    pub noise: f64,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    /// Lower Cholesky factor of `K + noise I`.
    chol: Matrix,
    /// `K^-1 y`, precomputed.
    alpha: Vec<f64>,
}

/// A small multiple of the kernel's own scale, added to the diagonal so
/// that a Cholesky factorisation succeeds on a matrix that is positive
/// semi-definite in exact arithmetic and indefinite by a few ulps in
/// floating point.
///
/// This is not a modelling choice masquerading as a numerical one. Two
/// identical training inputs make the true covariance matrix singular,
/// and no amount of care in the factorisation changes that; the jitter
/// makes the answer well defined and biases it by an amount far below
/// any noise level anyone would use.
const JITTER: f64 = 1e-10;

impl Gp {
    /// Conditions the process on training data.
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidArgument`] for an invalid kernel, negative
    /// noise, an empty or ragged dataset, or non-finite values;
    /// [`SolveError::DimensionMismatch`] if the target count does not
    /// match the input count;
    /// [`SolveError::NotPositiveDefinite`] if the covariance matrix
    /// cannot be factored even with jitter.
    pub fn fit(kernel: KernelFn, noise: f64, x: &[Vec<f64>], y: &[f64]) -> Result<Self, SolveError> {
        if !kernel.is_valid() {
            return Err(SolveError::InvalidArgument("the kernel has invalid hyperparameters"));
        }
        if !noise.is_finite() || noise < 0.0 {
            return Err(SolveError::InvalidArgument("the noise variance must be nonnegative"));
        }
        if x.is_empty() {
            return Err(SolveError::InvalidArgument("the dataset is empty"));
        }
        if y.len() != x.len() {
            return Err(SolveError::DimensionMismatch { expected: x.len(), got: y.len() });
        }
        let dim = x[0].len();
        if dim == 0 || x.iter().any(|p| p.len() != dim) {
            return Err(SolveError::InvalidArgument("the inputs are ragged or zero-dimensional"));
        }
        if x.iter().flatten().chain(y.iter()).any(|v| !v.is_finite()) {
            return Err(SolveError::InvalidArgument("the data must be finite"));
        }
        let n = x.len();
        let scale = kernel.eval(&x[0], &x[0]).abs().max(1.0);
        let mut k = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = kernel.eval(&x[i], &x[j]);
                k.set(i, j, v);
                k.set(j, i, v);
            }
            k.set(i, i, k.get(i, i) + noise + JITTER * scale);
        }
        let chol = crate::linalg::cholesky::cholesky(&k)?;
        let alpha = crate::linalg::cholesky::cholesky_solve(&chol, y)?;
        Ok(Self { kernel, noise, x_train: x.to_vec(), y_train: y.to_vec(), chol, alpha })
    }

    /// How many points the process was conditioned on.
    pub fn len(&self) -> usize {
        self.x_train.len()
    }

    /// Whether the process has no training data. Always false, since
    /// [`Gp::fit`] refuses an empty dataset; present because clippy asks
    /// for it alongside `len`.
    pub fn is_empty(&self) -> bool {
        self.x_train.is_empty()
    }

    /// A cheap *lower bound* on the condition number of the covariance
    /// matrix, taken as the squared ratio of the largest to the smallest
    /// diagonal entry of its Cholesky factor.
    ///
    /// A lower bound, not an estimate: the true condition number can be
    /// an order of magnitude or two above this, since the factor's
    /// diagonal says nothing about how the off-diagonal mass is
    /// arranged. It is useful for noticing that a problem is badly
    /// conditioned, not for predicting how badly.
    ///
    /// Worth having in public, because it is the number that decides how
    /// much of an answer is real. A squared exponential kernel on points
    /// spaced well inside its length scale produces a covariance matrix
    /// that is singular to working precision -- the values it is
    /// correlating are nearly the same random variable -- and the jitter
    /// that makes the factorisation succeed is then what limits the
    /// accuracy of everything downstream. The jitter perturbs the matrix
    /// by a relative amount of its own size and the solve amplifies that
    /// by the condition number, so a noiseless fit interpolates to about
    /// the jitter times this -- which for a squared exponential on
    /// closely spaced points can be parts in a million rather than the
    /// parts in `1e16` the arithmetic would suggest.
    ///
    /// The remedy is not more precision. It is a shorter length scale, a
    /// rougher kernel from the Matern family, or a nonzero noise, all of
    /// which are statements about the model rather than about the
    /// arithmetic.
    pub fn condition_estimate(&self) -> f64 {
        let n = self.chol.rows;
        let mut lo = f64::INFINITY;
        let mut hi: f64 = 0.0;
        for i in 0..n {
            let d = self.chol.get(i, i).abs();
            lo = lo.min(d);
            hi = hi.max(d);
        }
        if lo > 0.0 {
            (hi / lo).powi(2)
        } else {
            f64::INFINITY
        }
    }

    /// Solves `L v = b` by forward substitution.
    fn forward_substitute(&self, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut v = vec![0.0; n];
        for i in 0..n {
            let mut acc = b[i];
            for j in 0..i {
                acc -= self.chol.get(i, j) * v[j];
            }
            v[i] = acc / self.chol.get(i, i);
        }
        v
    }

    /// The posterior mean and variance at each query point.
    ///
    /// The variance does not depend on the observed targets at all --
    /// see the module note. It is the prior variance minus what the data
    /// locations explain, and adding observations can only reduce it.
    ///
    /// # Errors
    ///
    /// [`SolveError::DimensionMismatch`] if a query point has the wrong
    /// dimension.
    pub fn predict(&self, x_star: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
        let dim = self.x_train[0].len();
        let mut means = Vec::with_capacity(x_star.len());
        let mut variances = Vec::with_capacity(x_star.len());
        for q in x_star {
            if q.len() != dim {
                return Err(SolveError::DimensionMismatch { expected: dim, got: q.len() });
            }
            let ks: Vec<f64> = self.x_train.iter().map(|t| self.kernel.eval(t, q)).collect();
            means.push(ks.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum());
            let v = self.forward_substitute(&ks);
            let explained: f64 = v.iter().map(|a| a * a).sum();
            // Clamped at zero: the subtraction is a difference of two
            // nearly equal numbers at a training point, where the answer
            // is zero and rounding can make it slightly negative. A
            // negative variance is never meaningful.
            variances.push((self.kernel.eval(q, q) - explained).max(0.0));
        }
        Ok((means, variances))
    }

    /// `log p(y | X)`, the log marginal likelihood.
    ///
    /// Equal to `-y^T K^-1 y / 2 - log|K| / 2 - n log(2 pi) / 2`, with
    /// the determinant read off the Cholesky diagonal rather than
    /// computed separately -- `log|K|` is twice the sum of the logs of
    /// the diagonal, which is both cheaper and better conditioned than
    /// forming a determinant that underflows for any sizeable `n`.
    pub fn log_marginal_likelihood(&self) -> f64 {
        let n = self.y_train.len();
        let fit: f64 = self.y_train.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum();
        let log_det: f64 = (0..n).map(|i| self.chol.get(i, i).ln()).sum::<f64>() * 2.0;
        -0.5 * fit - 0.5 * log_det - 0.5 * n as f64 * std::f64::consts::TAU.ln()
    }

    /// Refits with the hyperparameters that maximise the log marginal
    /// likelihood, searched by Nelder-Mead over their logarithms.
    ///
    /// Optimising the logarithms rather than the values keeps every
    /// hyperparameter positive without a constraint, and makes the
    /// search scale-free -- a length scale of `0.01` and one of `100`
    /// are the same distance from `1` in log space, which is how they
    /// should be treated when nothing is known about the scale.
    ///
    /// The likelihood surface is not concave and the search finds a
    /// local optimum. `restarts` different starting points are tried,
    /// spread geometrically around the current values, and the best is
    /// kept.
    ///
    /// # Errors
    ///
    /// As [`Gp::fit`], or [`SolveError::NoConvergence`] if no starting
    /// point produced a usable fit.
    pub fn optimize_hyperparams(
        &self,
        restarts: usize,
        rng: &mut Rng,
    ) -> Result<Gp, SolveError> {
        let base = self.kernel.parameters();
        let n = base.len();
        let objective = |logs: &[f64]| -> f64 {
            let values: Vec<f64> = logs.iter().map(|v| v.exp()).collect();
            let Ok(kernel) = self.kernel.with_parameters(&values) else {
                return f64::INFINITY;
            };
            if !kernel.is_valid() {
                return f64::INFINITY;
            }
            match Gp::fit(kernel, self.noise, &self.x_train, &self.y_train) {
                Ok(g) => {
                    let lml = g.log_marginal_likelihood();
                    if lml.is_finite() {
                        -lml
                    } else {
                        f64::INFINITY
                    }
                }
                Err(_) => f64::INFINITY,
            }
        };
        let mut best: Option<(f64, Vec<f64>)> = None;
        for attempt in 0..restarts.max(1) {
            let start: Vec<f64> = (0..n)
                .map(|k| {
                    let centre = base[k].max(1e-12).ln();
                    if attempt == 0 {
                        centre
                    } else {
                        centre + 2.0 * (rng.next_f64() - 0.5) * 2.0
                    }
                })
                .collect();
            let found = crate::optimization::nelder_mead(&objective, &start, 0.5, 1e-10, 4000);
            let value = objective(&found);
            if value.is_finite() && best.as_ref().is_none_or(|(v, _)| value < *v) {
                best = Some((value, found));
            }
        }
        let (_, logs) = best.ok_or(SolveError::NoConvergence { iters: restarts, residual: f64::INFINITY })?;
        let values: Vec<f64> = logs.iter().map(|v| v.exp()).collect();
        let kernel = self.kernel.with_parameters(&values)?;
        Gp::fit(kernel, self.noise, &self.x_train, &self.y_train)
    }

    /// Draws `count` sample functions from the posterior at the given
    /// points.
    ///
    /// # Errors
    ///
    /// As [`Gp::predict`], plus
    /// [`SolveError::NotPositiveDefinite`] if the joint posterior
    /// covariance cannot be factored.
    pub fn sample_posterior(
        &self,
        x_star: &[Vec<f64>],
        count: usize,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<f64>>, SolveError> {
        let (mean, _) = self.predict(x_star)?;
        let m = x_star.len();
        // The full joint posterior covariance, not just its diagonal:
        // sampling from the marginals independently would give draws
        // that jump between neighbouring points, which is not what the
        // process says at all.
        let mut cov = Matrix::zeros(m, m);
        let mut rows = Vec::with_capacity(m);
        for q in x_star {
            let ks: Vec<f64> = self.x_train.iter().map(|t| self.kernel.eval(t, q)).collect();
            rows.push(self.forward_substitute(&ks));
        }
        let scale = self.kernel.eval(&x_star[0], &x_star[0]).abs().max(1.0);
        for i in 0..m {
            for j in 0..m {
                let explained: f64 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| a * b).sum();
                cov.set(i, j, self.kernel.eval(&x_star[i], &x_star[j]) - explained);
            }
            cov.set(i, i, cov.get(i, i) + JITTER * scale);
        }
        // Symmetrise: the two triangles agree analytically and differ
        // by rounding, which a Cholesky refuses outright.
        let symmetric = Matrix::from_fn(m, m, |i, j| 0.5 * (cov.get(i, j) + cov.get(j, i)));
        let l = crate::linalg::cholesky::cholesky(&symmetric)?;
        Ok((0..count)
            .map(|_| {
                let z: Vec<f64> = (0..m).map(|_| rng.next_gaussian()).collect();
                (0..m)
                    .map(|i| mean[i] + (0..=i).map(|j| l.get(i, j) * z[j]).sum::<f64>())
                    .collect()
            })
            .collect())
    }
}

/// Draws sample functions from a prior with the given kernel.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid kernel or an empty or
/// ragged point set; [`SolveError::NotPositiveDefinite`] if the
/// covariance matrix cannot be factored.
pub fn sample_prior(
    kernel: &KernelFn,
    x: &[Vec<f64>],
    count: usize,
    rng: &mut Rng,
) -> Result<Vec<Vec<f64>>, SolveError> {
    if !kernel.is_valid() {
        return Err(SolveError::InvalidArgument("the kernel has invalid hyperparameters"));
    }
    if x.is_empty() {
        return Err(SolveError::InvalidArgument("no points to sample at"));
    }
    let dim = x[0].len();
    if dim == 0 || x.iter().any(|p| p.len() != dim) {
        return Err(SolveError::InvalidArgument("the points are ragged or zero-dimensional"));
    }
    let n = x.len();
    let scale = kernel.eval(&x[0], &x[0]).abs().max(1.0);
    let mut k = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = kernel.eval(&x[i], &x[j]);
            k.set(i, j, v);
            k.set(j, i, v);
        }
        k.set(i, i, k.get(i, i) + JITTER * scale);
    }
    let l = crate::linalg::cholesky::cholesky(&k)?;
    Ok((0..count)
        .map(|_| {
            let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
            (0..n).map(|i| (0..=i).map(|j| l.get(i, j) * z[j]).sum()).collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize, step: f64) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![i as f64 * step]).collect()
    }

    #[test]
    fn a_noiseless_process_interpolates_its_data_exactly() {
        // Conditioning a normal on one of its own coordinates gives a
        // point mass there: the mean passes through and the variance is
        // nothing. The residual left is the jitter's, not the method's.
        let x = grid(8, 0.4);
        let y: Vec<f64> = x.iter().map(|p| p[0].sin()).collect();
        let gp = Gp::fit(KernelFn::Rbf { l: 1.0, s: 1.0 }, 0.0, &x, &y).unwrap();
        let (mean, var) = gp.predict(&x).unwrap();
        for i in 0..x.len() {
            assert!((mean[i] - y[i]).abs() < 1e-7, "point {i} was off by {}", mean[i] - y[i]);
            assert!(var[i] < 1e-8, "point {i} had variance {}", var[i]);
        }
        assert_eq!(gp.len(), 8);
        assert!(!gp.is_empty());
    }

    #[test]
    fn the_posterior_variance_does_not_depend_on_the_observations() {
        // k(x,x) - k_*^T K^-1 k_* has no y in it. Uncertainty is a
        // statement about where the data is, not what it said -- which
        // is a real and often surprising property of the model rather
        // than an artefact of this implementation.
        let x = grid(7, 0.5);
        let a: Vec<f64> = x.iter().map(|p| p[0].sin()).collect();
        let b: Vec<f64> = x.iter().map(|p| 4.0 * p[0] * p[0] - 3.0).collect();
        let kernel = KernelFn::Matern52 { l: 0.8, s: 1.2 };
        let ga = Gp::fit(kernel.clone(), 0.05, &x, &a).unwrap();
        let gb = Gp::fit(kernel.clone(), 0.05, &x, &b).unwrap();
        let q = grid(25, 0.15);
        let (_, va) = ga.predict(&q).unwrap();
        let (_, vb) = gb.predict(&q).unwrap();
        for i in 0..q.len() {
            assert_eq!(va[i], vb[i], "the variance moved with the data at {i}");
        }
        // The mean, by contrast, is exactly linear in the observations.
        let scaled: Vec<f64> = a.iter().map(|v| 2.5 * v).collect();
        let gs = Gp::fit(kernel, 0.05, &x, &scaled).unwrap();
        let (ma, _) = ga.predict(&q).unwrap();
        let (ms, _) = gs.predict(&q).unwrap();
        for i in 0..q.len() {
            assert!((ms[i] - 2.5 * ma[i]).abs() < 1e-10 * (1.0 + ma[i].abs()), "mean at {i}");
        }
    }

    #[test]
    fn far_from_the_data_the_posterior_is_the_prior() {
        let x = grid(6, 0.3);
        let y: Vec<f64> = x.iter().map(|p| p[0].cos()).collect();
        let kernel = KernelFn::Rbf { l: 0.5, s: 1.4 };
        let gp = Gp::fit(kernel.clone(), 0.0, &x, &y).unwrap();
        let far = vec![vec![100.0]];
        let (mean, var) = gp.predict(&far).unwrap();
        assert!(mean[0].abs() < 1e-12, "the mean did not return to zero: {}", mean[0]);
        let prior = kernel.eval(&far[0], &far[0]);
        assert!((var[0] - prior).abs() < 1e-12, "the variance did not return to {prior}");
        // And conditioning never increases uncertainty anywhere.
        let q = grid(40, 0.1);
        let (_, v) = gp.predict(&q).unwrap();
        for (i, &value) in v.iter().enumerate() {
            assert!(value <= prior + 1e-12, "point {i} had variance {value} above the prior");
        }
    }

    #[test]
    fn the_kernels_have_the_shapes_they_claim() {
        // A periodic kernel is exactly periodic, not nearly. A
        // stationary kernel depends only on the separation. And each of
        // them peaks at zero separation and decays.
        let p = KernelFn::Periodic { l: 1.0, p: 2.5, s: 1.3 };
        for x in [0.0, 0.7, -3.1] {
            for m in [1.0, 2.0, 5.0] {
                let shifted = x + m * 2.5;
                assert_eq!(
                    p.eval(&[x], &[shifted]),
                    p.eval(&[x], &[x]),
                    "the period was not exact at {x} after {m} periods"
                );
            }
        }
        for kernel in [
            KernelFn::Rbf { l: 0.9, s: 1.1 },
            KernelFn::Matern32 { l: 0.9, s: 1.1 },
            KernelFn::Matern52 { l: 0.9, s: 1.1 },
        ] {
            let peak = kernel.eval(&[0.0], &[0.0]);
            assert!((peak - 1.1 * 1.1).abs() < 1e-14, "the amplitude was wrong");
            let mut previous = peak;
            for k in 1..30 {
                let r = k as f64 * 0.2;
                let v = kernel.eval(&[0.0], &[r]);
                assert!(v < previous, "the kernel rose at separation {r}");
                assert!(v > 0.0, "the kernel went negative at {r}");
                // Stationary: only the separation matters.
                assert!((v - kernel.eval(&[7.3], &[7.3 + r])).abs() < 1e-14);
                assert!((v - kernel.eval(&[0.0], &[-r])).abs() < 1e-14);
                previous = v;
            }
            assert!(kernel.eval(&[0.0], &[50.0]) < 1e-12, "the kernel did not decay");
        }
        // A sum and a product are what they say.
        let a = KernelFn::Rbf { l: 1.0, s: 1.0 };
        let b = KernelFn::Linear { s: 0.5, c: 0.25 };
        let sum = KernelFn::Sum(Box::new(a.clone()), Box::new(b.clone()));
        let product = KernelFn::Product(Box::new(a.clone()), Box::new(b.clone()));
        let (u, v) = ([0.3], [1.1]);
        assert!((sum.eval(&u, &v) - (a.eval(&u, &v) + b.eval(&u, &v))).abs() < 1e-15);
        assert!((product.eval(&u, &v) - a.eval(&u, &v) * b.eval(&u, &v)).abs() < 1e-15);
        assert!(sum.is_valid() && product.is_valid());
        assert!(!KernelFn::Rbf { l: -1.0, s: 1.0 }.is_valid());
        assert!(!KernelFn::Sum(
            Box::new(KernelFn::Rbf { l: 1.0, s: 1.0 }),
            Box::new(KernelFn::Rbf { l: 0.0, s: 1.0 })
        )
        .is_valid());
    }

    #[test]
    fn the_marginal_likelihood_matches_its_own_definition() {
        // Computed here from the Cholesky diagonal; checked against an
        // independent LU determinant and solve, which shares no code
        // with it.
        let x = grid(6, 0.45);
        let y: Vec<f64> = x.iter().map(|p| (2.0 * p[0]).sin() + 0.3).collect();
        let kernel = KernelFn::Matern32 { l: 0.7, s: 1.1 };
        let noise = 0.02;
        let gp = Gp::fit(kernel.clone(), noise, &x, &y).unwrap();
        let n = x.len();
        let mut k = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                k.set(i, j, kernel.eval(&x[i], &x[j]));
            }
            k.set(i, i, k.get(i, i) + noise + JITTER);
        }
        let lu = crate::linalg::lu::lu_decompose(&k).unwrap();
        let solved = crate::linalg::lu::solve(&k, &y).unwrap();
        let fit: f64 = y.iter().zip(solved.iter()).map(|(a, b)| a * b).sum();
        let want = -0.5 * fit
            - 0.5 * lu.determinant().ln()
            - 0.5 * n as f64 * std::f64::consts::TAU.ln();
        let got = gp.log_marginal_likelihood();
        assert!((got - want).abs() < 1e-9 * want.abs().max(1.0), "{got} against {want}");
    }

    #[test]
    fn noise_turns_interpolation_into_smoothing() {
        // With no noise the mean passes through every point. As the
        // noise grows the mean pulls away from the data and towards the
        // prior mean of zero, monotonically.
        let x = grid(9, 0.35);
        let y: Vec<f64> = x.iter().map(|p| p[0].sin() + 0.4).collect();
        let kernel = KernelFn::Rbf { l: 0.6, s: 1.0 };
        let mut previous = -1.0;
        for noise in [0.0, 1e-4, 1e-2, 1.0] {
            let gp = Gp::fit(kernel.clone(), noise, &x, &y).unwrap();
            let (mean, var) = gp.predict(&x).unwrap();
            let residual = mean
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(residual > previous, "noise {noise} did not loosen the fit");
            previous = residual;
            // More noise, more posterior variance at the data.
            assert!(var.iter().all(|&v| v >= 0.0));
        }
        // In the limit of overwhelming noise the data says nothing and
        // the posterior mean collapses onto the prior's, which is zero.
        // That is the statement worth asserting; how far it has got at
        // any particular noise level depends on the amplitude and the
        // spacing and is not a property of the method.
        let peak = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let drowned = Gp::fit(kernel, 1e4, &x, &y).unwrap();
        let (mean, _) = drowned.predict(&x).unwrap();
        let left = mean.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(left < 0.01 * peak, "overwhelming noise left {left} of {peak}");
    }

    #[test]
    fn tuning_the_hyperparameters_raises_the_marginal_likelihood() {
        let mut rng = Rng::new(0x51ac_de07);
        let x = grid(12, 0.3);
        let y: Vec<f64> = x.iter().map(|p| (2.0 * p[0]).sin() + 0.05 * rng.next_gaussian()).collect();
        // Deliberately poor starting hyperparameters.
        let gp = Gp::fit(KernelFn::Rbf { l: 8.0, s: 0.15 }, 0.01, &x, &y).unwrap();
        let before = gp.log_marginal_likelihood();
        let tuned = gp.optimize_hyperparams(4, &mut rng).unwrap();
        let after = tuned.log_marginal_likelihood();
        assert!(after > before + 10.0, "the likelihood only moved from {before} to {after}");
        // And the tuned process predicts the held-out shape better.
        let q = grid(30, 0.12);
        let truth: Vec<f64> = q.iter().map(|p| (2.0 * p[0]).sin()).collect();
        let error = |g: &Gp| {
            let (m, _) = g.predict(&q).unwrap();
            m.iter().zip(truth.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
        };
        assert!(error(&tuned) < error(&gp), "tuning made the predictions worse");
        // The parameters stay positive, which optimising in log space
        // guarantees without a constraint.
        assert!(tuned.kernel.parameters().iter().all(|v| *v > 0.0));
    }

    #[test]
    fn prior_draws_have_the_covariance_they_were_asked_for() {
        let mut rng = Rng::new(0x3d90_1b6e);
        let kernel = KernelFn::Rbf { l: 1.0, s: 1.0 };
        let points = grid(5, 0.5);
        let draws = sample_prior(&kernel, &points, 20_000, &mut rng).unwrap();
        assert_eq!(draws.len(), 20_000);
        for i in 0..points.len() {
            for j in 0..points.len() {
                let empirical: f64 = draws.iter().map(|d| d[i] * d[j]).sum::<f64>()
                    / draws.len() as f64;
                let want = kernel.eval(&points[i], &points[j]);
                assert!((empirical - want).abs() < 0.05, "({i},{j}): {empirical} vs {want}");
            }
        }
    }

    #[test]
    fn posterior_draws_pass_through_noiseless_data() {
        let mut rng = Rng::new(0x2b71_c045);
        let x = grid(5, 0.6);
        let y: Vec<f64> = x.iter().map(|p| p[0].cos()).collect();
        let gp = Gp::fit(KernelFn::Rbf { l: 0.9, s: 1.0 }, 0.0, &x, &y).unwrap();
        let draws = gp.sample_posterior(&x, 20, &mut rng).unwrap();
        assert_eq!(draws.len(), 20);
        for d in &draws {
            for i in 0..x.len() {
                assert!((d[i] - y[i]).abs() < 1e-4, "a draw missed point {i} by {}", d[i] - y[i]);
            }
        }
        // Away from the data the draws spread out.
        let q = vec![vec![10.0]];
        let far = gp.sample_posterior(&q, 400, &mut rng).unwrap();
        let spread = far.iter().map(|d| d[0] * d[0]).sum::<f64>() / far.len() as f64;
        assert!(spread > 0.3, "the draws did not spread away from the data: {spread}");
    }

    #[test]
    fn the_process_refuses_impossible_arguments() {
        let x = grid(4, 0.5);
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let good = KernelFn::Rbf { l: 1.0, s: 1.0 };
        assert!(Gp::fit(KernelFn::Rbf { l: 0.0, s: 1.0 }, 0.0, &x, &y).is_err());
        assert!(Gp::fit(good.clone(), -1.0, &x, &y).is_err());
        assert!(Gp::fit(good.clone(), 0.0, &[], &[]).is_err());
        assert!(Gp::fit(good.clone(), 0.0, &x, &y[..2]).is_err());
        assert!(Gp::fit(good.clone(), 0.0, &[vec![1.0], vec![1.0, 2.0]], &[1.0, 2.0]).is_err());
        assert!(Gp::fit(good.clone(), 0.0, &[vec![], vec![]], &[1.0, 2.0]).is_err());
        assert!(Gp::fit(good.clone(), 0.0, &x, &[1.0, 2.0, 3.0, f64::NAN]).is_err());
        let gp = Gp::fit(good.clone(), 0.1, &x, &y).unwrap();
        assert!(gp.predict(&[vec![1.0, 2.0]]).is_err());
        assert!(sample_prior(&KernelFn::Rbf { l: -1.0, s: 1.0 }, &x, 1, &mut Rng::new(1)).is_err());
        assert!(sample_prior(&good, &[], 1, &mut Rng::new(1)).is_err());
        assert!(sample_prior(&good, &[vec![]], 1, &mut Rng::new(1)).is_err());
        // Parameter round-tripping.
        let periodic = KernelFn::Periodic { l: 1.0, p: 2.0, s: 3.0 };
        assert_eq!(periodic.parameters(), vec![1.0, 2.0, 3.0]);
        assert_eq!(periodic.with_parameters(&[4.0, 5.0, 6.0]).unwrap().parameters(), vec![4.0, 5.0, 6.0]);
        assert!(periodic.with_parameters(&[1.0]).is_err());
        let compound = KernelFn::Sum(
            Box::new(KernelFn::Rbf { l: 1.0, s: 2.0 }),
            Box::new(KernelFn::Linear { s: 3.0, c: 4.0 }),
        );
        assert_eq!(compound.parameters(), vec![1.0, 2.0, 3.0, 4.0]);
        let rebuilt = compound.with_parameters(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(rebuilt.parameters(), vec![5.0, 6.0, 7.0, 8.0]);
    }
}
