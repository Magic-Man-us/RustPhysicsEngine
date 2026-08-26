//! Convex optimisation: gradient methods, quasi-Newton methods, proximal
//! splitting, and constrained solvers.
//!
//! Convexity buys one thing, and it is decisive: every local minimum is
//! global. That removes the question the methods in
//! [`crate::optimization::metaheuristics`] spend all their effort on -- where
//! else to look -- and replaces it with a purely local question, how fast to
//! get downhill. Everything here is an answer to that.
//!
//! The answers differ in what they know about curvature. Gradient descent
//! knows nothing and pays for it: on a quadratic its error contracts by
//! `(k-1)/(k+1)` per step, so a condition number of a thousand costs a
//! thousand-fold more iterations than a condition number of one. Conjugate
//! gradients build a set of mutually conjugate directions and finish an
//! `n`-dimensional quadratic in at most `n` steps exactly. Newton's method
//! uses the Hessian outright and lands on a quadratic's minimum in a single
//! step. Quasi-Newton methods sit in between, accumulating an approximation
//! to the Hessian from the gradients they have already paid for.
//!
//! Those are not asymptotic claims but exact ones, and the tests check them
//! as such: Newton in one step, conjugate gradients in `n`, and every method
//! against the closed-form minimiser `-Q^-1 c` of the quadratic it was given.
//!
//! The proximal half of the module handles objectives that are convex but not
//! differentiable -- an L1 penalty, a constraint set -- by splitting them into
//! a smooth part, handled by a gradient step, and a simple part, handled by
//! its proximal operator. The reason that works is that the awkward part is
//! usually simple in isolation: the proximal operator of an L1 penalty is
//! soft thresholding, of a box is clamping, and of a simplex is a sorted
//! shift. Each is a projection or near-projection with a closed form, so the
//! non-smoothness costs almost nothing.

use crate::error::GeomError;
use crate::linalg::cholesky::{cholesky, cholesky_solve};
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// Steps shorter than this are treated as zero.
const STEP_TOL: f64 = 1e-14;

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Euclidean norm.
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

// ---------------------------------------------------------------------------
// Line search
// ---------------------------------------------------------------------------

/// Backtracking line search satisfying the Armijo sufficient-decrease
/// condition.
///
/// Halves the step until `f(x + t d) <= f(x) + c t g . d`. The condition is
/// what stops a long step that reduces the objective by less than the
/// gradient promised, which is how a descent method diverges on a curved
/// function despite every step going downhill.
///
/// # Panics
/// Panics unless the direction is a descent direction and `c` lies in
/// `(0, 1)`.
#[must_use]
pub fn backtracking(
    f: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    direction: &[f64],
    gradient: &[f64],
    c: f64,
    max_halvings: usize,
) -> f64 {
    assert!(c > 0.0 && c < 1.0, "backtracking requires c in (0, 1)");
    let slope = dot(gradient, direction);
    assert!(slope <= 0.0, "backtracking requires a descent direction");
    let base = f(x);
    let mut t = 1.0f64;
    for _ in 0..max_halvings {
        let trial: Vec<f64> = x.iter().zip(direction).map(|(a, d)| a + t * d).collect();
        if f(&trial) <= base + c * t * slope {
            return t;
        }
        t *= 0.5;
    }
    t
}

/// A line search satisfying the strong Wolfe conditions.
///
/// Armijo alone allows arbitrarily *short* steps, which stalls a quasi-Newton
/// method: the curvature information it accumulates comes from the difference
/// between successive gradients, and a step too short to change the gradient
/// carries none. The second Wolfe condition,
/// `|g(x + t d) . d| <= c2 |g(x) . d|`, rules that out by demanding the slope
/// actually flatten. Together they are what makes the BFGS update
/// well defined.
///
/// # Panics
/// Panics unless `0 < c1 < c2 < 1` and the direction descends.
#[must_use]
pub fn line_search_wolfe(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    direction: &[f64],
    c1: f64,
    c2: f64,
) -> f64 {
    assert!(c1 > 0.0 && c1 < c2 && c2 < 1.0, "the Wolfe constants must satisfy 0 < c1 < c2 < 1");
    let base = f(x);
    let slope = dot(&grad(x), direction);
    assert!(slope <= 0.0, "line_search_wolfe requires a descent direction");

    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    let mut t = 1.0f64;
    for _ in 0..80 {
        let trial: Vec<f64> = x.iter().zip(direction).map(|(a, d)| a + t * d).collect();
        if f(&trial) > base + c1 * t * slope {
            // Overshot the sufficient decrease: shrink.
            hi = t;
        } else if dot(&grad(&trial), direction) < c2 * slope {
            // Decreased enough but the slope is still steep: lengthen.
            lo = t;
        } else {
            return t;
        }
        t = if hi.is_finite() { 0.5 * (lo + hi) } else { 2.0 * lo.max(STEP_TOL) };
        if hi - lo < STEP_TOL {
            break;
        }
    }
    t
}

/// An exact line search, by root-finding on the directional derivative.
///
/// The minimiser of `phi(t) = f(x + t d)` is where `phi'(t) = g(x + t d) . d`
/// vanishes. Since `phi'(0) < 0` for a descent direction, all that is needed
/// is a `t` where the slope has turned non-negative; bisection then locates
/// the root to machine precision. On a quadratic `phi'` is affine, so the
/// answer is exact to rounding.
///
/// Returns `None` when no such bracket exists within a doubling cap, which
/// means the function is unbounded below along the direction -- there is no
/// minimiser to find, and the caller should use an inexact search instead.
///
/// # Panics
/// Panics unless the direction descends.
#[must_use]
pub fn exact_line_search(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    direction: &[f64],
) -> Option<f64> {
    let slope_at = |t: f64| -> f64 {
        let trial: Vec<f64> = x.iter().zip(direction).map(|(a, d)| a + t * d).collect();
        dot(&grad(&trial), direction)
    };
    let slope0 = slope_at(0.0);
    assert!(slope0 <= 0.0, "exact_line_search requires a descent direction");
    if slope0 == 0.0 {
        return Some(0.0);
    }

    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mut bracketed = false;
    for _ in 0..60 {
        if slope_at(hi) >= 0.0 {
            bracketed = true;
            break;
        }
        lo = hi;
        hi *= 2.0;
    }
    if !bracketed {
        return None;
    }

    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if slope_at(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// First-order methods
// ---------------------------------------------------------------------------

/// Nesterov's accelerated gradient method.
///
/// Evaluates the gradient at an extrapolated point rather than the current
/// one, which is the whole difference from heavy-ball momentum: the method
/// gets to see where the momentum is taking it before committing. That
/// changes the convergence rate on a smooth convex function from `O(1/k)` to
/// `O(1/k^2)`, which is optimal for a method that only ever sees gradients.
///
/// # Panics
/// Panics if the learning rate is not positive.
#[must_use]
pub fn nesterov(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    learning_rate: f64,
    momentum: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(learning_rate > 0.0, "nesterov requires a positive learning rate");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut velocity = vec![0.0; n];
    for _ in 0..iterations {
        // Look ahead along the current velocity before measuring the slope.
        let ahead: Vec<f64> =
            x.iter().zip(&velocity).map(|(a, v)| a + momentum * v).collect();
        let g = grad(&ahead);
        for i in 0..n {
            velocity[i] = momentum * velocity[i] - learning_rate * g[i];
            x[i] += velocity[i];
        }
    }
    x
}

/// Adagrad: scale each coordinate's step by the inverse root of its
/// accumulated squared gradient.
///
/// Coordinates with consistently large gradients get short steps and rare
/// coordinates get long ones, which is what makes it suit sparse problems.
/// The accumulator only grows, so the effective learning rate decays
/// monotonically to zero -- helpful for convergence, fatal if the problem
/// needs to keep moving, which is what [`rmsprop`] fixes.
///
/// # Panics
/// Panics if the learning rate is not positive.
#[must_use]
pub fn adagrad(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    learning_rate: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(learning_rate > 0.0, "adagrad requires a positive learning rate");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut accumulated = vec![0.0f64; n];
    for _ in 0..iterations {
        let g = grad(&x);
        for i in 0..n {
            accumulated[i] += g[i] * g[i];
            x[i] -= learning_rate * g[i] / (accumulated[i].sqrt() + 1e-12);
        }
    }
    x
}

/// RMSProp: Adagrad with an exponentially weighted accumulator.
///
/// Forgetting old gradients keeps the effective learning rate from decaying
/// to zero, so the method can keep making progress indefinitely.
///
/// # Panics
/// Panics unless the learning rate is positive and `decay` lies in `[0, 1)`.
#[must_use]
pub fn rmsprop(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    learning_rate: f64,
    decay: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(learning_rate > 0.0, "rmsprop requires a positive learning rate");
    assert!((0.0..1.0).contains(&decay), "rmsprop requires decay in [0, 1)");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut average = vec![0.0f64; n];
    for _ in 0..iterations {
        let g = grad(&x);
        for i in 0..n {
            average[i] = decay * average[i] + (1.0 - decay) * g[i] * g[i];
            x[i] -= learning_rate * g[i] / (average[i].sqrt() + 1e-12);
        }
    }
    x
}

/// AdamW: Adam with the weight decay applied to the parameters directly
/// rather than folded into the gradient.
///
/// The distinction matters because Adam divides the gradient by its own
/// running scale. A decay term added to the gradient gets divided too, so its
/// strength ends up depending on how large the other gradients happen to be;
/// applied to the parameters it does not. That is the entire content of the
/// change, and it is why the two behave differently at the same nominal decay.
///
/// # Panics
/// Panics unless the learning rate is positive and both moment decays lie in
/// `[0, 1)`.
#[must_use]
pub fn adamw(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    learning_rate: f64,
    weight_decay: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(learning_rate > 0.0, "adamw requires a positive learning rate");
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    for step in 1..=iterations {
        let g = grad(&x);
        let t = step as f64;
        for i in 0..n {
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * g[i];
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * g[i] * g[i];
            let m_hat = m[i] / (1.0 - BETA1.powf(t));
            let v_hat = v[i] / (1.0 - BETA2.powf(t));
            // The decay is applied here, outside the adaptive scaling.
            x[i] -= learning_rate * (m_hat / (v_hat.sqrt() + 1e-8) + weight_decay * x[i]);
        }
    }
    x
}

/// Subgradient descent for a convex objective that is not differentiable.
///
/// A subgradient is not a descent direction -- moving along it can increase
/// the objective, which is why the best value seen has to be tracked
/// separately rather than read off the last iterate. With a step size going
/// to zero but summing to infinity the method converges, at `O(1/sqrt(k))`:
/// far worse than the smooth case, and the price of giving up
/// differentiability.
///
/// Returns the best point found.
///
/// # Panics
/// Panics if the initial step is not positive.
#[must_use]
pub fn subgradient_method(
    f: &dyn Fn(&[f64]) -> f64,
    subgradient: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    initial_step: f64,
    iterations: usize,
) -> (Vec<f64>, f64) {
    assert!(initial_step > 0.0, "subgradient_method requires a positive step");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut best = x.clone();
    let mut best_value = f(&x);
    for k in 1..=iterations {
        let g = subgradient(&x);
        // A step like a / sqrt(k): square-summable would converge too fast to
        // reach a distant optimum, and constant would not converge at all.
        let step = initial_step / (k as f64).sqrt();
        for i in 0..n {
            x[i] -= step * g[i];
        }
        let value = f(&x);
        if value < best_value {
            best_value = value;
            best = x.clone();
        }
    }
    (best, best_value)
}

// ---------------------------------------------------------------------------
// Second-order and quasi-Newton
// ---------------------------------------------------------------------------

/// Newton's method in several variables.
///
/// Solves `H d = -g` for the step. On a quadratic the Hessian is exact and
/// the model is the function itself, so a single full step lands on the
/// minimum -- an exact statement, not an asymptotic one, and the sharpest
/// distinction between this and every first-order method here.
///
/// Falls back to the gradient direction where the Hessian is not positive
/// definite, since the Newton step then points uphill.
///
/// # Errors
/// Returns an error if the starting point is empty.
pub fn newton_method_nd(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    hess: &dyn Fn(&[f64]) -> Matrix,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, f64), GeomError> {
    if x0.is_empty() {
        return Err(GeomError::InvalidArgument("newton_method_nd requires variables"));
    }
    let mut x = x0.to_vec();
    for _ in 0..max_iter {
        let g = grad(&x);
        if norm(&g) < tol {
            break;
        }
        let h = hess(&x);
        let negated: Vec<f64> = g.iter().map(|v| -v).collect();
        // Cholesky succeeds exactly when the Hessian is positive definite,
        // which is also exactly when the Newton step descends.
        let direction = match cholesky(&h).and_then(|l| cholesky_solve(&l, &negated)) {
            Ok(d) => d,
            Err(_) => negated,
        };
        let t = backtracking(f, &x, &direction, &g, 1e-4, 60);
        for i in 0..x.len() {
            x[i] += t * direction[i];
        }
    }
    let value = f(&x);
    Ok((x, value))
}

/// BFGS with a Wolfe line search.
///
/// Maintains an approximation to the *inverse* Hessian, updated from the
/// change in gradient across each step, so a Newton-like direction costs one
/// matrix-vector product and no solve. The update preserves positive
/// definiteness whenever the curvature condition `y . s > 0` holds, which the
/// Wolfe line search guarantees -- the two are designed together, and pairing
/// BFGS with a plain Armijo search is a classic way to make it fail.
///
/// # Errors
/// Returns an error if the starting point is empty.
pub fn bfgs(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, f64), GeomError> {
    let n = x0.len();
    if n == 0 {
        return Err(GeomError::InvalidArgument("bfgs requires variables"));
    }
    let mut x = x0.to_vec();
    let mut inverse = Matrix::identity(n);
    let mut g = grad(&x);

    for _ in 0..max_iter {
        if norm(&g) < tol {
            break;
        }
        let direction: Vec<f64> =
            (0..n).map(|i| -(0..n).map(|j| inverse.get(i, j) * g[j]).sum::<f64>()).collect();
        if dot(&g, &direction) >= 0.0 {
            // The approximation has lost definiteness; restart from steepest
            // descent rather than stepping uphill.
            inverse = Matrix::identity(n);
            continue;
        }
        let t = line_search_wolfe(f, grad, &x, &direction, 1e-4, 0.9);
        let s: Vec<f64> = direction.iter().map(|d| t * d).collect();
        if norm(&s) < STEP_TOL {
            break;
        }
        let next: Vec<f64> = x.iter().zip(&s).map(|(a, d)| a + d).collect();
        let next_g = grad(&next);
        let y: Vec<f64> = next_g.iter().zip(&g).map(|(a, b)| a - b).collect();

        let sy = dot(&s, &y);
        if sy > 1e-12 {
            // The Sherman-Morrison form of the inverse update.
            let rho = 1.0 / sy;
            let hy: Vec<f64> =
                (0..n).map(|i| (0..n).map(|j| inverse.get(i, j) * y[j]).sum::<f64>()).collect();
            let yhy = dot(&y, &hy);
            let mut updated = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let value = inverse.get(i, j) - rho * (s[i] * hy[j] + hy[i] * s[j])
                        + rho * rho * (yhy + sy) * s[i] * s[j];
                    updated.set(i, j, value);
                }
            }
            inverse = updated;
        }
        x = next;
        g = next_g;
    }
    let value = f(&x);
    Ok((x, value))
}

/// Limited-memory BFGS.
///
/// Stores the last `m` pairs of step and gradient change instead of a full
/// matrix, and reconstructs the search direction by a two-loop recursion.
/// Memory drops from `n^2` to `mn`, which is what makes the method usable
/// where `n` runs to millions and a dense inverse Hessian could not be stored
/// at all, let alone factored.
///
/// # Errors
/// Returns an error if the starting point is empty or `m` is zero.
pub fn lbfgs(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    m: usize,
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, f64), GeomError> {
    let n = x0.len();
    if n == 0 || m == 0 {
        return Err(GeomError::InvalidArgument("lbfgs requires variables and memory"));
    }
    let mut x = x0.to_vec();
    let mut g = grad(&x);
    let mut history: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> =
        std::collections::VecDeque::new();

    for _ in 0..max_iter {
        if norm(&g) < tol {
            break;
        }
        // Two-loop recursion: apply the stored curvature pairs backwards,
        // scale, then apply them forwards.
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let alpha = rho * dot(s, &q);
            for i in 0..n {
                q[i] -= alpha * y[i];
            }
            alphas.push(alpha);
        }
        // Scale by the most recent curvature, which is the usual initial
        // Hessian estimate and matters far more than it looks.
        if let Some((s, y, _)) = history.back() {
            let scale = dot(s, y) / dot(y, y).max(1e-300);
            for entry in q.iter_mut() {
                *entry *= scale;
            }
        }
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot(y, &q);
            for i in 0..n {
                q[i] += (alpha - beta) * s[i];
            }
        }
        let direction: Vec<f64> = q.iter().map(|v| -v).collect();
        if dot(&g, &direction) >= 0.0 {
            history.clear();
            continue;
        }

        let t = line_search_wolfe(f, grad, &x, &direction, 1e-4, 0.9);
        let s: Vec<f64> = direction.iter().map(|d| t * d).collect();
        if norm(&s) < STEP_TOL {
            break;
        }
        let next: Vec<f64> = x.iter().zip(&s).map(|(a, d)| a + d).collect();
        let next_g = grad(&next);
        let y: Vec<f64> = next_g.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            history.push_back((s, y, 1.0 / sy));
            if history.len() > m {
                history.pop_front();
            }
        }
        x = next;
        g = next_g;
    }
    let value = f(&x);
    Ok((x, value))
}

/// Nonlinear conjugate gradients with the Polak-Ribiere update.
///
/// On a quadratic the directions produced are mutually conjugate, so the
/// method reaches the exact minimum in at most `n` steps -- an exact finite
/// termination, not a rate. Away from a quadratic that guarantee lapses,
/// and the restart when `beta` goes negative is what keeps the directions
/// descending regardless.
///
/// # Errors
/// Returns an error if the starting point is empty.
pub fn conjugate_gradient_nonlinear(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, f64), GeomError> {
    let n = x0.len();
    if n == 0 {
        return Err(GeomError::InvalidArgument("conjugate_gradient_nonlinear requires variables"));
    }
    let mut x = x0.to_vec();
    let mut g = grad(&x);
    let mut direction: Vec<f64> = g.iter().map(|v| -v).collect();

    for _ in 0..max_iter {
        if norm(&g) < tol {
            break;
        }
        if dot(&g, &direction) >= 0.0 {
            // An inexact line search can leave the Polak-Ribiere direction
            // pointing uphill even after beta is clipped at zero, because the
            // clipping argument assumes conjugacy that only an exact search
            // delivers. Steepest descent always descends, so restart there.
            direction = g.iter().map(|v| -v).collect();
        }
        // Conjugacy -- and with it the finite termination on a quadratic --
        // is a property of the *exact* minimiser along each direction. Fall
        // back to Wolfe only where the exact search cannot bracket, which
        // means the function is not convex along this line.
        let t = exact_line_search(grad, &x, &direction)
            .unwrap_or_else(|| line_search_wolfe(f, grad, &x, &direction, 1e-4, 0.1));
        for i in 0..n {
            x[i] += t * direction[i];
        }
        let next_g = grad(&x);
        // Polak-Ribiere, clipped at zero: a negative beta means the new
        // direction would not descend, and the fix is to restart.
        let beta = (dot(&next_g, &next_g) - dot(&next_g, &g)) / dot(&g, &g).max(1e-300);
        let beta = beta.max(0.0);
        for i in 0..n {
            direction[i] = -next_g[i] + beta * direction[i];
        }
        g = next_g;
    }
    let value = f(&x);
    Ok((x, value))
}

/// A trust-region method with the dogleg step.
///
/// Instead of choosing a direction and then a length, this chooses a radius
/// first and takes the best step inside it. The dogleg path runs from the
/// Cauchy point -- the minimiser along the steepest-descent direction -- to
/// the full Newton step, and the step taken is where that path leaves the
/// trust region. The radius grows when the quadratic model predicted the
/// actual decrease well and shrinks when it did not, so the method regulates
/// its own trust in the model.
///
/// # Errors
/// Returns an error if the starting point is empty or the radius is not
/// positive.
pub fn trust_region_dogleg(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    hess: &dyn Fn(&[f64]) -> Matrix,
    x0: &[f64],
    initial_radius: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, f64), GeomError> {
    let n = x0.len();
    if n == 0 {
        return Err(GeomError::InvalidArgument("trust_region_dogleg requires variables"));
    }
    if !(initial_radius > 0.0) {
        return Err(GeomError::InvalidArgument("trust_region_dogleg requires a positive radius"));
    }
    let mut x = x0.to_vec();
    let mut radius = initial_radius;
    let max_radius = initial_radius * 1e6;

    for _ in 0..max_iter {
        let g = grad(&x);
        if norm(&g) < 1e-12 {
            break;
        }
        let h = hess(&x);
        let hg: Vec<f64> =
            (0..n).map(|i| (0..n).map(|j| h.get(i, j) * g[j]).sum::<f64>()).collect();
        let ghg = dot(&g, &hg);

        // The Cauchy point: as far along -g as the model keeps improving.
        let cauchy: Vec<f64> = if ghg <= 0.0 {
            let scale = radius / norm(&g).max(1e-300);
            g.iter().map(|v| -scale * v).collect()
        } else {
            let t = (dot(&g, &g) / ghg).min(radius / norm(&g).max(1e-300));
            g.iter().map(|v| -t * v).collect()
        };

        let negated: Vec<f64> = g.iter().map(|v| -v).collect();
        let newton = cholesky(&h).and_then(|l| cholesky_solve(&l, &negated)).ok();

        let step = match newton {
            Some(full) if norm(&full) <= radius => full,
            Some(full) => {
                // Walk the dogleg from the Cauchy point toward the Newton
                // step until it touches the boundary.
                let diff: Vec<f64> = full.iter().zip(&cauchy).map(|(a, b)| a - b).collect();
                let a = dot(&diff, &diff);
                let b = 2.0 * dot(&cauchy, &diff);
                let c = dot(&cauchy, &cauchy) - radius * radius;
                let disc = (b * b - 4.0 * a * c).max(0.0).sqrt();
                let tau = if a > 1e-300 { ((-b + disc) / (2.0 * a)).clamp(0.0, 1.0) } else { 0.0 };
                cauchy.iter().zip(&diff).map(|(p, d)| p + tau * d).collect()
            }
            None => cauchy,
        };

        // Compare the model's predicted decrease against the real one.
        let hs: Vec<f64> =
            (0..n).map(|i| (0..n).map(|j| h.get(i, j) * step[j]).sum::<f64>()).collect();
        let predicted = -(dot(&g, &step) + 0.5 * dot(&step, &hs));
        let candidate: Vec<f64> = x.iter().zip(&step).map(|(a, d)| a + d).collect();
        let actual = f(&x) - f(&candidate);
        let ratio = if predicted.abs() < 1e-300 { 1.0 } else { actual / predicted };

        if ratio < 0.25 {
            radius *= 0.25;
        } else if ratio > 0.75 && (norm(&step) - radius).abs() < 1e-10 {
            radius = (2.0 * radius).min(max_radius);
        }
        if ratio > 0.0 {
            x = candidate;
        }
        if radius < 1e-14 {
            break;
        }
    }
    let value = f(&x);
    Ok((x, value))
}

// ---------------------------------------------------------------------------
// Proximal operators
// ---------------------------------------------------------------------------

/// The proximal operator of `t ||x||_1`: soft thresholding.
///
/// `prox(v) = sign(v) max(|v| - t, 0)`, which is the exact minimiser of
/// `||x - v||^2 / 2 + t ||x||_1`. It is what makes L1 penalties produce
/// genuinely zero coefficients rather than merely small ones -- the operator
/// maps a whole interval to exactly zero, which no smooth penalty does.
#[must_use]
pub fn prox_l1(v: &[f64], t: f64) -> Vec<f64> {
    v.iter().map(|x| x.signum() * (x.abs() - t).max(0.0)).collect()
}

/// The proximal operator of `t ||x||_2` (the norm, not its square): block
/// soft thresholding.
///
/// Shrinks the whole vector toward zero and sets it to exactly zero once its
/// norm falls below `t`. Unlike [`prox_l1`] it acts on the vector as a unit,
/// which is what group-sparse penalties need.
#[must_use]
pub fn prox_l2(v: &[f64], t: f64) -> Vec<f64> {
    let n = norm(v);
    if n <= t {
        return vec![0.0; v.len()];
    }
    let scale = 1.0 - t / n;
    v.iter().map(|x| scale * x).collect()
}

/// The proximal operator of a box constraint: clamping.
///
/// The proximal operator of an indicator function is the projection onto the
/// set, and for a box that is coordinatewise clamping.
#[must_use]
pub fn prox_box(v: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    v.iter().map(|x| x.clamp(lo, hi)).collect()
}

/// Euclidean projection onto the probability simplex.
///
/// Sort, find the threshold at which the shifted positive parts sum to one,
/// and subtract it. The result is the closest point of the simplex, which is
/// not simply the clamped-and-renormalised vector -- that is a common
/// substitute and it is a different point.
///
/// # Panics
/// Panics if the vector is empty.
#[must_use]
pub fn prox_simplex(v: &[f64]) -> Vec<f64> {
    assert!(!v.is_empty(), "prox_simplex requires a non-empty vector");
    let n = v.len();
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Walk down the sorted vector keeping the last threshold that still
    // leaves the entry above it positive. That threshold is the one whose
    // shifted positive parts sum to exactly one.
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for k in 0..n {
        cumulative += sorted[k];
        let candidate = (cumulative - 1.0) / (k + 1) as f64;
        if sorted[k] - candidate > 0.0 {
            theta = candidate;
        }
    }
    v.iter().map(|x| (x - theta).max(0.0)).collect()
}

// ---------------------------------------------------------------------------
// Proximal gradient methods
// ---------------------------------------------------------------------------

/// Proximal gradient descent, also called ISTA: a gradient step on the smooth
/// part followed by the proximal operator of the rest.
///
/// The whole point is that the non-smooth part never needs a gradient. It
/// only has to have a proximal operator that can be evaluated, and for the
/// penalties that matter -- L1, group norms, indicator functions -- that
/// operator is a closed form.
///
/// Converges at `O(1/k)`.
///
/// # Panics
/// Panics if the step size is not positive.
#[must_use]
pub fn proximal_gradient(
    smooth_grad: &dyn Fn(&[f64]) -> Vec<f64>,
    prox: &dyn Fn(&[f64], f64) -> Vec<f64>,
    x0: &[f64],
    step: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(step > 0.0, "proximal_gradient requires a positive step");
    let mut x = x0.to_vec();
    for _ in 0..iterations {
        let g = smooth_grad(&x);
        let stepped: Vec<f64> = x.iter().zip(&g).map(|(a, d)| a - step * d).collect();
        x = prox(&stepped, step);
    }
    x
}

/// FISTA: proximal gradient descent with Nesterov's extrapolation.
///
/// The same two operations per iteration as [`proximal_gradient`], applied at
/// an extrapolated point, which improves the rate from `O(1/k)` to `O(1/k^2)`
/// for no extra cost per step. The momentum sequence
/// `t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2` is what makes the accelerated
/// bound come out; an arbitrary momentum does not.
///
/// # Panics
/// Panics if the step size is not positive.
#[must_use]
pub fn fista(
    smooth_grad: &dyn Fn(&[f64]) -> Vec<f64>,
    prox: &dyn Fn(&[f64], f64) -> Vec<f64>,
    x0: &[f64],
    step: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(step > 0.0, "fista requires a positive step");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut y = x0.to_vec();
    let mut t = 1.0f64;
    for _ in 0..iterations {
        let g = smooth_grad(&y);
        let stepped: Vec<f64> = y.iter().zip(&g).map(|(a, d)| a - step * d).collect();
        let next = prox(&stepped, step);
        let next_t = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let factor = (t - 1.0) / next_t;
        y = (0..n).map(|i| next[i] + factor * (next[i] - x[i])).collect();
        x = next;
        t = next_t;
    }
    x
}

/// Projected gradient descent for a constrained smooth problem.
///
/// Take a gradient step, then project back onto the feasible set. Correct
/// whenever the set is convex and the projection is available; the projection
/// is what makes or breaks it, since for most sets it is itself an
/// optimisation problem.
///
/// # Panics
/// Panics if the step size is not positive.
#[must_use]
pub fn projected_gradient(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    project: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    step: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(step > 0.0, "projected_gradient requires a positive step");
    let mut x = project(x0);
    for _ in 0..iterations {
        let g = grad(&x);
        let stepped: Vec<f64> = x.iter().zip(&g).map(|(a, d)| a - step * d).collect();
        x = project(&stepped);
    }
    x
}

/// The Frank-Wolfe method, also called conditional gradient.
///
/// Instead of projecting, it minimises a linear approximation over the
/// feasible set and moves toward that vertex. The iterate stays feasible
/// automatically as a convex combination of feasible points, so no projection
/// is ever needed -- which is the reason to use it when a linear minimisation
/// over the set is cheap and a projection is not.
///
/// `linear_oracle` returns the minimiser of a linear function over the set.
///
/// # Panics
/// Panics if the starting point is empty.
#[must_use]
pub fn frank_wolfe(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    linear_oracle: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    iterations: usize,
) -> Vec<f64> {
    assert!(!x0.is_empty(), "frank_wolfe requires variables");
    let n = x0.len();
    let mut x = x0.to_vec();
    for k in 0..iterations {
        let g = grad(&x);
        let vertex = linear_oracle(&g);
        // The classic 2/(k+2) schedule keeps every iterate a convex
        // combination of the starting point and the vertices visited.
        let gamma = 2.0 / (k as f64 + 2.0);
        for i in 0..n {
            x[i] += gamma * (vertex[i] - x[i]);
        }
    }
    x
}

/// Mirror descent on the probability simplex, with the entropy mirror map.
///
/// The multiplicative update `x_i <- x_i exp(-t g_i)` followed by
/// renormalisation. Because the geometry matches the constraint set, the
/// dependence on dimension is `sqrt(log n)` rather than the `sqrt(n)` a
/// Euclidean projected gradient pays -- a large difference when the simplex
/// is over thousands of outcomes.
///
/// # Panics
/// Panics if the starting point is empty or the step is not positive.
#[must_use]
pub fn mirror_descent_simplex(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    step: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(!x0.is_empty(), "mirror_descent_simplex requires variables");
    assert!(step > 0.0, "mirror_descent_simplex requires a positive step");
    let mut x: Vec<f64> = {
        let total: f64 = x0.iter().map(|v| v.max(0.0)).sum();
        if total > 0.0 {
            x0.iter().map(|v| v.max(0.0) / total).collect()
        } else {
            vec![1.0 / x0.len() as f64; x0.len()]
        }
    };
    for _ in 0..iterations {
        let g = grad(&x);
        // Subtract the smallest exponent before exponentiating, so a large
        // gradient cannot overflow the update.
        let shift = g.iter().copied().fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> =
            x.iter().zip(&g).map(|(v, d)| v * (-step * (d - shift)).exp()).collect();
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) || !total.is_finite() {
            break;
        }
        x = weights.iter().map(|w| w / total).collect();
    }
    x
}

// ---------------------------------------------------------------------------
// Regularised regression
// ---------------------------------------------------------------------------

/// Ridge regression in closed form: solve `(A'A + lambda I) x = A'b`.
///
/// The one regularised regression with an exact answer, because the penalty
/// is smooth and quadratic like the loss. The added `lambda I` is also what
/// makes the system solvable when `A'A` is singular -- ridge regression
/// regularises the numerics as much as the statistics.
///
/// # Errors
/// Returns an error on a shape mismatch, a negative penalty, or a system that
/// is singular even after regularisation.
pub fn ridge_closed_form(a: &Matrix, b: &[f64], lambda: f64) -> Result<Vec<f64>, GeomError> {
    if a.rows != b.len() {
        return Err(GeomError::InvalidArgument("ridge_closed_form: shape mismatch"));
    }
    if lambda < 0.0 {
        return Err(GeomError::InvalidArgument("ridge_closed_form requires lambda >= 0"));
    }
    let k = a.cols;
    let mut normal = Matrix::zeros(k, k);
    let mut rhs = vec![0.0; k];
    for i in 0..k {
        for j in i..k {
            let v: f64 = (0..a.rows).map(|r| a.get(r, i) * a.get(r, j)).sum();
            normal.set(i, j, v);
            normal.set(j, i, v);
        }
        normal.set(i, i, normal.get(i, i) + lambda);
        rhs[i] = (0..a.rows).map(|r| a.get(r, i) * b[r]).sum();
    }
    let l = cholesky(&normal)
        .map_err(|_| GeomError::Degenerate("ridge_closed_form: the system is singular"))?;
    cholesky_solve(&l, &rhs)
        .map_err(|_| GeomError::Degenerate("ridge_closed_form: the solve failed"))
}

/// Lasso by cyclic coordinate descent.
///
/// Each coordinate is minimised exactly with the others held fixed, and that
/// one-dimensional problem has the soft-threshold closed form. Coordinate
/// descent works here precisely because the non-smooth part is *separable*:
/// the L1 penalty splits across coordinates, so a coordinatewise minimum is a
/// genuine minimum. On a non-separable penalty the same loop can stall at a
/// point that is optimal in every single direction and not optimal at all.
///
/// Minimises `||A x - b||^2 / (2 n) + lambda ||x||_1`.
///
/// # Errors
/// Returns an error on a shape mismatch or a negative penalty.
pub fn lasso_coordinate_descent(
    a: &Matrix,
    b: &[f64],
    lambda: f64,
    iterations: usize,
) -> Result<Vec<f64>, GeomError> {
    if a.rows != b.len() {
        return Err(GeomError::InvalidArgument("lasso_coordinate_descent: shape mismatch"));
    }
    if lambda < 0.0 {
        return Err(GeomError::InvalidArgument("lasso requires lambda >= 0"));
    }
    let (n, k) = (a.rows, a.cols);
    let scale = n as f64;
    let mut x = vec![0.0; k];
    let mut residual: Vec<f64> = b.to_vec();

    let column_norm: Vec<f64> =
        (0..k).map(|j| (0..n).map(|r| a.get(r, j) * a.get(r, j)).sum::<f64>()).collect();

    for _ in 0..iterations {
        for j in 0..k {
            if column_norm[j] <= 0.0 {
                continue;
            }
            // Add this coordinate's contribution back before re-minimising.
            for r in 0..n {
                residual[r] += a.get(r, j) * x[j];
            }
            let rho: f64 = (0..n).map(|r| a.get(r, j) * residual[r]).sum::<f64>() / scale;
            let denominator = column_norm[j] / scale;
            let updated = rho.signum() * (rho.abs() - lambda).max(0.0) / denominator;
            x[j] = updated;
            for r in 0..n {
                residual[r] -= a.get(r, j) * x[j];
            }
        }
    }
    Ok(x)
}

/// The lasso by the alternating direction method of multipliers.
///
/// Splits the objective into the smooth least-squares part and the L1 part
/// with a copy of the variable, then alternates: a ridge solve, a soft
/// threshold, and a dual update. The factorisation of the ridge system does
/// not change between iterations, so it can be computed once -- which is what
/// makes ADMM cheap here despite doing a linear solve every step.
///
/// Solves the same problem as [`lasso_coordinate_descent`] and must agree
/// with it.
///
/// # Errors
/// Returns an error on a shape mismatch, a non-positive penalty parameter, or
/// a singular system.
pub fn admm_lasso(
    a: &Matrix,
    b: &[f64],
    lambda: f64,
    rho: f64,
    iterations: usize,
) -> Result<Vec<f64>, GeomError> {
    if a.rows != b.len() {
        return Err(GeomError::InvalidArgument("admm_lasso: shape mismatch"));
    }
    if !(rho > 0.0) || lambda < 0.0 {
        return Err(GeomError::InvalidArgument("admm_lasso requires rho > 0 and lambda >= 0"));
    }
    let (n, k) = (a.rows, a.cols);
    let scale = n as f64;

    // (A'A / n + rho I) is fixed across iterations, so factor it once.
    let mut normal = Matrix::zeros(k, k);
    let mut atb = vec![0.0; k];
    for i in 0..k {
        for j in i..k {
            let v: f64 = (0..n).map(|r| a.get(r, i) * a.get(r, j)).sum::<f64>() / scale;
            normal.set(i, j, v);
            normal.set(j, i, v);
        }
        normal.set(i, i, normal.get(i, i) + rho);
        atb[i] = (0..n).map(|r| a.get(r, i) * b[r]).sum::<f64>() / scale;
    }
    let factor = cholesky(&normal)
        .map_err(|_| GeomError::Degenerate("admm_lasso: the system is singular"))?;

    let mut x = vec![0.0; k];
    let mut z = vec![0.0; k];
    let mut u = vec![0.0; k];
    for _ in 0..iterations {
        let rhs: Vec<f64> = (0..k).map(|j| atb[j] + rho * (z[j] - u[j])).collect();
        x = cholesky_solve(&factor, &rhs)
            .map_err(|_| GeomError::Degenerate("admm_lasso: the solve failed"))?;
        let shifted: Vec<f64> = (0..k).map(|j| x[j] + u[j]).collect();
        z = prox_l1(&shifted, lambda / rho);
        for j in 0..k {
            u[j] += x[j] - z[j];
        }
    }
    // Return the thresholded copy: it is the one that is exactly sparse.
    Ok(z)
}

/// A generic two-block ADMM.
///
/// Minimises `f(x) + g(z)` subject to `x = z`, given only the proximal
/// operator of each part. The two halves never need to be handled together,
/// which is the point: a problem that is hard as a whole is often two easy
/// problems joined by a constraint.
///
/// # Panics
/// Panics if `rho` is not positive or the starting point is empty.
#[must_use]
pub fn admm_generic(
    prox_f: &dyn Fn(&[f64], f64) -> Vec<f64>,
    prox_g: &dyn Fn(&[f64], f64) -> Vec<f64>,
    x0: &[f64],
    rho: f64,
    iterations: usize,
) -> Vec<f64> {
    assert!(rho > 0.0, "admm_generic requires rho > 0");
    assert!(!x0.is_empty(), "admm_generic requires variables");
    let k = x0.len();
    let mut z = x0.to_vec();
    let mut u = vec![0.0; k];
    for _ in 0..iterations {
        let a: Vec<f64> = (0..k).map(|j| z[j] - u[j]).collect();
        let x = prox_f(&a, 1.0 / rho);
        let b: Vec<f64> = (0..k).map(|j| x[j] + u[j]).collect();
        z = prox_g(&b, 1.0 / rho);
        for j in 0..k {
            u[j] += x[j] - z[j];
        }
    }
    z
}

/// Elastic net regression: an L1 and an L2 penalty together.
///
/// The L1 part selects variables and the L2 part keeps correlated ones
/// together. Pure lasso picks arbitrarily among a group of correlated
/// predictors and zeroes the rest, which is unstable under resampling; the
/// ridge term removes that arbitrariness. At `l1 = 0` it is ridge and at
/// `l2 = 0` it is the lasso, and the tests check both limits.
///
/// # Errors
/// Returns an error on a shape mismatch or a negative penalty.
pub fn elastic_net(
    a: &Matrix,
    b: &[f64],
    l1: f64,
    l2: f64,
    iterations: usize,
) -> Result<Vec<f64>, GeomError> {
    if a.rows != b.len() {
        return Err(GeomError::InvalidArgument("elastic_net: shape mismatch"));
    }
    if l1 < 0.0 || l2 < 0.0 {
        return Err(GeomError::InvalidArgument("elastic_net requires non-negative penalties"));
    }
    if l1 == 0.0 {
        // With no L1 term the problem is smooth and has a closed form; the
        // scaling matches the coordinate-descent objective below.
        return ridge_closed_form(a, b, l2 * a.rows as f64);
    }
    let (n, k) = (a.rows, a.cols);
    let scale = n as f64;
    let mut x = vec![0.0; k];
    let mut residual: Vec<f64> = b.to_vec();
    let column_norm: Vec<f64> =
        (0..k).map(|j| (0..n).map(|r| a.get(r, j) * a.get(r, j)).sum::<f64>()).collect();

    for _ in 0..iterations {
        for j in 0..k {
            if column_norm[j] <= 0.0 {
                continue;
            }
            for r in 0..n {
                residual[r] += a.get(r, j) * x[j];
            }
            let rho: f64 = (0..n).map(|r| a.get(r, j) * residual[r]).sum::<f64>() / scale;
            // The ridge term simply enlarges the denominator.
            let denominator = column_norm[j] / scale + l2;
            x[j] = rho.signum() * (rho.abs() - l1).max(0.0) / denominator;
            for r in 0..n {
                residual[r] -= a.get(r, j) * x[j];
            }
        }
    }
    Ok(x)
}

/// L2-penalised logistic regression, fitted by Newton's method.
///
/// The penalised log-likelihood is strictly concave for any positive penalty,
/// so the maximum is unique and Newton's method converges quadratically to
/// it. Without the penalty, perfectly separable data has no finite maximiser
/// at all -- the coefficients run to infinity as the fitted probabilities
/// approach zero and one -- which is a property of the data rather than a
/// failure of the solver, and the penalty is what makes the problem
/// well posed.
///
/// `y` holds zeros and ones. Returns the coefficients.
///
/// # Errors
/// Returns an error on a shape mismatch, a label outside `{0, 1}`, or a
/// non-positive penalty.
pub fn logistic_regression_fit(
    x: &Matrix,
    y: &[f64],
    lambda: f64,
    iterations: usize,
) -> Result<Vec<f64>, GeomError> {
    if x.rows != y.len() {
        return Err(GeomError::InvalidArgument("logistic_regression_fit: shape mismatch"));
    }
    if y.iter().any(|v| *v != 0.0 && *v != 1.0) {
        return Err(GeomError::InvalidArgument("logistic labels must be zero or one"));
    }
    if !(lambda > 0.0) {
        return Err(GeomError::InvalidArgument(
            "logistic_regression_fit requires a positive penalty; without one a separable \
             sample has no finite maximiser",
        ));
    }
    let (n, k) = (x.rows, x.cols);
    let mut beta = vec![0.0; k];

    for _ in 0..iterations {
        let mut gradient = vec![0.0; k];
        let mut hessian = Matrix::zeros(k, k);
        for r in 0..n {
            let z: f64 = (0..k).map(|j| x.get(r, j) * beta[j]).sum();
            let p = 1.0 / (1.0 + (-z).exp());
            let w = (p * (1.0 - p)).max(1e-12);
            for i in 0..k {
                gradient[i] += x.get(r, i) * (p - y[r]);
                for j in i..k {
                    let v = hessian.get(i, j) + w * x.get(r, i) * x.get(r, j);
                    hessian.set(i, j, v);
                    hessian.set(j, i, v);
                }
            }
        }
        for i in 0..k {
            gradient[i] += lambda * beta[i];
            hessian.set(i, i, hessian.get(i, i) + lambda);
        }
        if norm(&gradient) < 1e-12 {
            break;
        }
        let negated: Vec<f64> = gradient.iter().map(|v| -v).collect();
        let Ok(step) = cholesky(&hessian).and_then(|l| cholesky_solve(&l, &negated)) else {
            break;
        };
        for i in 0..k {
            beta[i] += step[i];
        }
    }
    Ok(beta)
}

// ---------------------------------------------------------------------------
// Constrained optimisation
// ---------------------------------------------------------------------------

/// A convex quadratic program with equality constraints, by the active-set
/// idea applied to the equalities alone.
///
/// Minimises `x'Qx/2 + c'x` subject to `Ax = b`. With only equalities the
/// active set is fixed, so the whole problem is one KKT linear system:
/// stationarity and feasibility stacked together. The solution satisfies
/// `Qx + c + A'y = 0` exactly, which is what the tests check rather than
/// merely that the objective looks small.
///
/// # Errors
/// Returns an error on a shape mismatch or a singular KKT system.
pub fn quadratic_program_active_set(
    q: &Matrix,
    c: &[f64],
    a: &Matrix,
    b: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), GeomError> {
    let n = c.len();
    let m = b.len();
    if !q.is_square() || q.rows != n || a.cols != n || a.rows != m {
        return Err(GeomError::InvalidArgument("quadratic_program_active_set: shape mismatch"));
    }
    // The KKT system: [Q A'; A 0] [x; y] = [-c; b].
    let size = n + m;
    let mut kkt = Matrix::zeros(size, size);
    let mut rhs = vec![0.0; size];
    for i in 0..n {
        for j in 0..n {
            kkt.set(i, j, q.get(i, j));
        }
        for r in 0..m {
            kkt.set(i, n + r, a.get(r, i));
            kkt.set(n + r, i, a.get(r, i));
        }
        rhs[i] = -c[i];
    }
    rhs[n..n + m].copy_from_slice(b);
    // The KKT matrix is symmetric but indefinite, so an LU rather than a
    // Cholesky factorisation is required.
    let solution = crate::linalg::lu::solve(&kkt, &rhs)
        .map_err(|_| GeomError::Degenerate("quadratic_program_active_set: singular KKT system"))?;
    Ok((solution[..n].to_vec(), solution[n..].to_vec()))
}

/// The norm of the Karush-Kuhn-Tucker residual at a candidate point.
///
/// Stacks the stationarity condition `grad f + sum y_i grad c_i` and the
/// feasibility conditions `c_i(x) = 0`. Zero exactly at a constrained
/// stationary point, which makes it the natural way to check a constrained
/// solver: it tests the conditions the answer must satisfy rather than
/// comparing against another solver that could share the same mistake.
#[must_use]
pub fn kkt_residual(
    objective_gradient: &[f64],
    constraint_values: &[f64],
    constraint_gradients: &[Vec<f64>],
    multipliers: &[f64],
) -> f64 {
    let n = objective_gradient.len();
    let mut total = 0.0;
    for i in 0..n {
        let mut stationarity = objective_gradient[i];
        for (k, g) in constraint_gradients.iter().enumerate() {
            if i < g.len() && k < multipliers.len() {
                stationarity += multipliers[k] * g[i];
            }
        }
        total += stationarity * stationarity;
    }
    for v in constraint_values {
        total += v * v;
    }
    total.sqrt()
}

/// The quadratic penalty method for equality-constrained minimisation.
///
/// Minimises `f(x) + mu ||c(x)||^2 / 2` for a sequence of growing `mu`. It is
/// the simplest constrained method and it has a real defect: the constraint is
/// only satisfied in the limit `mu -> infinity`, and the subproblem's
/// condition number grows with `mu`, so accuracy and conditioning pull in
/// opposite directions. [`augmented_lagrangian`] removes exactly that
/// trade-off.
///
/// # Panics
/// Panics if the starting penalty is not positive.
#[must_use]
pub fn penalty_method(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    constraints: &dyn Fn(&[f64]) -> Vec<f64>,
    constraint_gradients: &dyn Fn(&[f64]) -> Vec<Vec<f64>>,
    x0: &[f64],
    initial_penalty: f64,
    outer: usize,
    inner: usize,
    step: f64,
) -> Vec<f64> {
    assert!(initial_penalty > 0.0, "penalty_method requires a positive penalty");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut mu = initial_penalty;
    for _ in 0..outer {
        for _ in 0..inner {
            let g = grad(&x);
            let c = constraints(&x);
            let cg = constraint_gradients(&x);
            let mut total = g;
            for (k, gradient) in cg.iter().enumerate() {
                let weight = mu * c.get(k).copied().unwrap_or(0.0);
                for i in 0..n.min(gradient.len()) {
                    total[i] += weight * gradient[i];
                }
            }
            for i in 0..n {
                x[i] -= step * total[i];
            }
        }
        mu *= 10.0;
    }
    x
}

/// The augmented Lagrangian method, also called the method of multipliers.
///
/// Adds an explicit multiplier estimate to the quadratic penalty, and updates
/// it by `y <- y + mu c(x)` after each inner solve. That update is what lets
/// the constraint be satisfied exactly at a *finite* penalty: the multiplier
/// absorbs the work the penalty would otherwise have to do by growing without
/// bound, so the subproblems stay well conditioned.
///
/// Returns the point and the final multipliers.
///
/// # Panics
/// Panics if the penalty or step is not positive.
#[must_use]
pub fn augmented_lagrangian(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    constraints: &dyn Fn(&[f64]) -> Vec<f64>,
    constraint_gradients: &dyn Fn(&[f64]) -> Vec<Vec<f64>>,
    x0: &[f64],
    penalty: f64,
    outer: usize,
    inner: usize,
    step: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(penalty > 0.0 && step > 0.0, "augmented_lagrangian requires positive parameters");
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut multipliers = vec![0.0; constraints(x0).len()];
    let mut mu = penalty;
    let mut previous_violation = f64::INFINITY;

    for _ in 0..outer {
        for _ in 0..inner {
            let g = grad(&x);
            let c = constraints(&x);
            let cg = constraint_gradients(&x);
            let mut total = g;
            for (k, gradient) in cg.iter().enumerate() {
                let weight = multipliers.get(k).copied().unwrap_or(0.0)
                    + mu * c.get(k).copied().unwrap_or(0.0);
                for i in 0..n.min(gradient.len()) {
                    total[i] += weight * gradient[i];
                }
            }
            for i in 0..n {
                x[i] -= step * total[i];
            }
        }
        // The multiplier update: this is what the plain penalty method lacks.
        let c = constraints(&x);
        for (k, entry) in multipliers.iter_mut().enumerate() {
            *entry += mu * c.get(k).copied().unwrap_or(0.0);
        }
        // Raise the penalty only when the multiplier update failed to pull the
        // violation down. Doubling it unconditionally would defeat the point:
        // the inner problem's curvature grows with `mu`, so a fixed inner step
        // that was stable at the start diverges a handful of doublings later,
        // and the method would be no better conditioned than the plain penalty
        // it replaces.
        let violation = norm(&c);
        if violation > 0.25 * previous_violation {
            mu *= 2.0;
        }
        previous_violation = violation;
    }
    (x, multipliers)
}

/// Dual ascent for an equality-constrained problem.
///
/// Alternates minimising the Lagrangian over `x` with a gradient step on the
/// multipliers, whose gradient is the constraint violation itself. It
/// converges only under strong assumptions -- strict convexity of the
/// objective, chiefly -- which is precisely the gap that the augmented
/// Lagrangian and ADMM close by adding a penalty term.
///
/// `minimise_lagrangian` returns the minimiser of `f(x) + y . c(x)` for the
/// given multipliers.
///
/// # Panics
/// Panics if the step is not positive.
#[must_use]
pub fn dual_ascent(
    minimise_lagrangian: &dyn Fn(&[f64]) -> Vec<f64>,
    constraints: &dyn Fn(&[f64]) -> Vec<f64>,
    multipliers0: &[f64],
    step: f64,
    iterations: usize,
) -> (Vec<f64>, Vec<f64>) {
    assert!(step > 0.0, "dual_ascent requires a positive step");
    let mut y = multipliers0.to_vec();
    let mut x = minimise_lagrangian(&y);
    for _ in 0..iterations {
        x = minimise_lagrangian(&y);
        let c = constraints(&x);
        for (k, entry) in y.iter_mut().enumerate() {
            // Ascent, not descent: the dual is being maximised.
            *entry += step * c.get(k).copied().unwrap_or(0.0);
        }
    }
    (x, y)
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Tests convexity numerically by sampling the midpoint inequality.
///
/// A convex function satisfies `f((a+b)/2) <= (f(a) + f(b)) / 2` for every
/// pair. Sampling can only ever *refute* convexity, never establish it: a
/// single violating pair is a proof of non-convexity, while a million
/// satisfying pairs prove nothing about the pairs not tried. The return value
/// should be read accordingly -- `false` is a fact and `true` is an absence
/// of evidence.
///
/// # Panics
/// Panics if the bounds are empty or `trials` is zero.
#[must_use]
pub fn convexity_check_numeric(
    f: &dyn Fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    trials: usize,
    rng: &mut Rng,
) -> bool {
    assert!(!bounds.is_empty(), "convexity_check_numeric requires bounds");
    assert!(trials > 0, "convexity_check_numeric requires trials");
    for _ in 0..trials {
        let a: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
        let b: Vec<f64> = bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect();
        let mid: Vec<f64> = a.iter().zip(&b).map(|(p, q)| 0.5 * (p + q)).collect();
        let chord = 0.5 * (f(&a) + f(&b));
        if f(&mid) > chord + 1e-9 * (1.0 + chord.abs()) {
            return false;
        }
    }
    true
}

/// Iterations that gradient descent and conjugate gradients need on a
/// two-dimensional quadratic of the given condition number.
///
/// Returns `(gradient descent, conjugate gradients)`. The contrast is the
/// whole point: gradient descent's error contracts by `(k-1)/(k+1)` per
/// step, so its count grows linearly in the condition number, while
/// conjugate gradients terminate in at most `n` steps whatever the
/// conditioning. At a condition number of a thousand that is hundreds of
/// iterations against two.
///
/// # Panics
/// Panics if the condition number is below one.
#[must_use]
pub fn condition_number_effect_demo(kappa: f64) -> (usize, usize) {
    assert!(kappa >= 1.0, "the condition number must be at least one");
    // Minimise (x^2 + kappa y^2) / 2, whose Hessian is diag(1, kappa).
    let start = [1.0f64, 1.0f64];
    let tol = 1e-8;

    // Gradient descent at the optimal fixed step 2 / (L + m).
    let step = 2.0 / (1.0 + kappa);
    let mut x = start;
    let mut gd = 0usize;
    while (x[0] * x[0] + kappa * x[1] * x[1]) / 2.0 > tol && gd < 1_000_000 {
        x[0] -= step * x[0];
        x[1] -= step * kappa * x[1];
        gd += 1;
    }

    // Linear conjugate gradients on the same system.
    let mut y = start;
    let mut r = [-y[0], -kappa * y[1]];
    let mut p = r;
    let mut cg = 0usize;
    while r[0] * r[0] + r[1] * r[1] > tol * tol && cg < 100 {
        let ap = [p[0], kappa * p[1]];
        let denominator = p[0] * ap[0] + p[1] * ap[1];
        if denominator.abs() < 1e-300 {
            break;
        }
        let alpha = (r[0] * r[0] + r[1] * r[1]) / denominator;
        let old = r[0] * r[0] + r[1] * r[1];
        y[0] += alpha * p[0];
        y[1] += alpha * p[1];
        r[0] -= alpha * ap[0];
        r[1] -= alpha * ap[1];
        let beta = (r[0] * r[0] + r[1] * r[1]) / old;
        p[0] = r[0] + beta * p[0];
        p[1] = r[1] + beta * p[1];
        cg += 1;
    }
    (gd, cg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    /// A strongly convex quadratic `x'Qx/2 + c'x` and everything about it.
    struct Quadratic {
        q: Matrix,
        c: Vec<f64>,
    }

    impl Quadratic {
        fn new(diagonal: &[f64], coupling: f64) -> Self {
            let n = diagonal.len();
            let mut q = Matrix::zeros(n, n);
            for i in 0..n {
                q.set(i, i, diagonal[i]);
                if i + 1 < n {
                    q.set(i, i + 1, coupling);
                    q.set(i + 1, i, coupling);
                }
            }
            Self { q, c: (0..n).map(|i| 1.0 + i as f64).collect() }
        }
        fn value(&self, x: &[f64]) -> f64 {
            let n = x.len();
            let mut acc = 0.0;
            for i in 0..n {
                acc += self.c[i] * x[i];
                for j in 0..n {
                    acc += 0.5 * self.q.get(i, j) * x[i] * x[j];
                }
            }
            acc
        }
        fn gradient(&self, x: &[f64]) -> Vec<f64> {
            let n = x.len();
            (0..n)
                .map(|i| self.c[i] + (0..n).map(|j| self.q.get(i, j) * x[j]).sum::<f64>())
                .collect()
        }
        /// The exact minimiser, solving `Q x = -c`.
        fn minimiser(&self) -> Vec<f64> {
            let negated: Vec<f64> = self.c.iter().map(|v| -v).collect();
            let l = cholesky(&self.q).expect("the test quadratic is positive definite");
            cholesky_solve(&l, &negated).expect("solvable")
        }
    }

    // -----------------------------------------------------------------
    // The exact properties, which are exact rather than asymptotic
    // -----------------------------------------------------------------

    #[test]
    fn newton_lands_on_a_quadratic_minimum_in_a_single_step() {
        // The sharpest distinction in the module: the quadratic model Newton
        // builds *is* the function, so one full step is exact. Every
        // first-order method here needs an unbounded number.
        let quadratic = Quadratic::new(&[3.0, 5.0, 2.0], 0.5);
        let exact = quadratic.minimiser();
        let f = |x: &[f64]| quadratic.value(x);
        let g = |x: &[f64]| quadratic.gradient(x);
        let h = |_: &[f64]| quadratic.q.clone();

        let (x, value) = newton_method_nd(&f, &g, &h, &[10.0, -8.0, 4.0], 1e-12, 1).unwrap();
        for (a, b) in x.iter().zip(&exact) {
            assert!((a - b).abs() < 1e-9, "one step gave {x:?}, exact is {exact:?}");
        }
        assert!(close(value, quadratic.value(&exact), 1e-9));
        // The gradient really is zero there.
        assert!(norm(&quadratic.gradient(&x)) < 1e-9);
    }

    #[test]
    fn conjugate_gradients_finish_a_quadratic_in_at_most_n_steps() {
        // Finite termination, not a rate: the directions are mutually
        // conjugate, so after n of them the whole space has been searched.
        for n in 2..=5usize {
            let diagonal: Vec<f64> = (0..n).map(|i| 1.0 + 3.0 * i as f64).collect();
            let quadratic = Quadratic::new(&diagonal, 0.4);
            let exact = quadratic.minimiser();
            let f = |x: &[f64]| quadratic.value(x);
            let g = |x: &[f64]| quadratic.gradient(x);
            let start = vec![5.0; n];

            let (x, _) = conjugate_gradient_nonlinear(&f, &g, &start, 1e-14, n).unwrap();
            let error: f64 =
                x.iter().zip(&exact).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            assert!(
                error < 1e-6,
                "n = {n}: after {n} steps the error is still {error} ({x:?} against {exact:?})"
            );
        }
    }

    #[test]
    fn every_method_reaches_the_same_closed_form_minimiser() {
        // One quadratic, six methods, one answer known in closed form. Nothing
        // here is compared against another solver's output.
        let quadratic = Quadratic::new(&[4.0, 7.0, 3.0, 5.0], 1.0);
        let exact = quadratic.minimiser();
        let target = quadratic.value(&exact);
        let f = |x: &[f64]| quadratic.value(x);
        let g = |x: &[f64]| quadratic.gradient(x);
        let h = |_: &[f64]| quadratic.q.clone();
        let start = vec![6.0, -3.0, 2.0, 1.0];

        let results: Vec<(&str, Vec<f64>)> = vec![
            ("newton", newton_method_nd(&f, &g, &h, &start, 1e-12, 50).unwrap().0),
            ("bfgs", bfgs(&f, &g, &start, 1e-10, 500).unwrap().0),
            ("lbfgs", lbfgs(&f, &g, &start, 5, 1e-10, 500).unwrap().0),
            ("cg", conjugate_gradient_nonlinear(&f, &g, &start, 1e-12, 200).unwrap().0),
            ("dogleg", trust_region_dogleg(&f, &g, &h, &start, 1.0, 200).unwrap().0),
            ("nesterov", nesterov(&g, &start, 0.05, 0.9, 20_000)),
        ];
        for (name, x) in &results {
            for (a, b) in x.iter().zip(&exact) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "{name} gave {x:?}, the exact minimiser is {exact:?}"
                );
            }
            assert!(close(quadratic.value(x), target, 1e-8), "{name}'s value is off");
        }
    }

    #[test]
    fn the_conditioning_penalty_is_paid_by_gradient_descent_and_not_by_conjugate_gradients() {
        // Gradient descent's iteration count grows with the condition number;
        // conjugate gradients terminate in at most n whatever it is.
        let mut previous = 0usize;
        for kappa in [1.0f64, 10.0, 100.0, 1000.0] {
            let (gd, cg) = condition_number_effect_demo(kappa);
            assert!(cg <= 2, "conjugate gradients took {cg} steps in two dimensions");
            assert!(gd >= previous, "gradient descent got faster as conditioning worsened");
            previous = gd;
        }
        let (easy, _) = condition_number_effect_demo(1.0);
        let (hard, _) = condition_number_effect_demo(1000.0);
        assert!(
            hard > 20 * easy.max(1),
            "a thousandfold conditioning cost only {hard} against {easy} iterations"
        );
    }

    // -----------------------------------------------------------------
    // Line search
    // -----------------------------------------------------------------

    #[test]
    fn the_wolfe_search_returns_a_step_satisfying_both_conditions() {
        // The conditions are checkable directly, which is better than
        // checking that the method using them happens to converge.
        let quadratic = Quadratic::new(&[2.0, 9.0], 0.3);
        let f = |x: &[f64]| quadratic.value(x);
        let g = |x: &[f64]| quadratic.gradient(x);
        let (c1, c2) = (1e-4, 0.9);

        for start in [vec![3.0, 4.0], vec![-2.0, 1.0], vec![0.5, -6.0]] {
            let gradient = g(&start);
            let direction: Vec<f64> = gradient.iter().map(|v| -v).collect();
            let t = line_search_wolfe(&f, &g, &start, &direction, c1, c2);
            assert!(t > 0.0 && t.is_finite(), "the step is {t}");

            let moved: Vec<f64> = start.iter().zip(&direction).map(|(a, d)| a + t * d).collect();
            let slope = dot(&gradient, &direction);
            // Armijo: enough decrease for the step taken.
            assert!(
                f(&moved) <= f(&start) + c1 * t * slope + 1e-12,
                "the sufficient-decrease condition failed at t = {t}"
            );
            // Curvature: the slope has genuinely flattened.
            assert!(
                dot(&g(&moved), &direction) >= c2 * slope - 1e-12,
                "the curvature condition failed at t = {t}"
            );
        }
    }

    #[test]
    fn the_exact_search_lands_on_the_closed_form_step_and_declines_an_unbounded_line() {
        // Along a direction d from x, the minimiser of a quadratic is at
        // t* = -(g . d) / (d' Q d). That is checkable in closed form, so the
        // search is measured against arithmetic rather than against itself.
        let quadratic = Quadratic::new(&[2.0, 9.0, 4.0], 0.7);
        let g = |x: &[f64]| quadratic.gradient(x);

        for start in [vec![3.0, 4.0, -1.0], vec![-2.0, 1.0, 5.0], vec![0.5, -6.0, 0.0]] {
            for direction in [
                g(&start).iter().map(|v| -v).collect::<Vec<f64>>(),
                vec![-1.0, 0.0, 0.0],
                vec![0.2, -0.9, 0.4],
            ] {
                let gradient = g(&start);
                if dot(&gradient, &direction) > 0.0 {
                    continue;
                }
                let curvature: f64 = (0..3)
                    .map(|i| {
                        (0..3)
                            .map(|j| direction[i] * quadratic.q.get(i, j) * direction[j])
                            .sum::<f64>()
                    })
                    .sum();
                let expected = -dot(&gradient, &direction) / curvature;

                let t = exact_line_search(&g, &start, &direction).expect("bounded below");
                assert!(
                    (t - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                    "the exact step is {expected} but the search returned {t}"
                );
                // The defining property, stated directly: the slope vanishes.
                let moved: Vec<f64> =
                    start.iter().zip(&direction).map(|(a, d)| a + t * d).collect();
                assert!(
                    dot(&g(&moved), &direction).abs() < 1e-9,
                    "the directional derivative at the returned step is not zero"
                );
            }
        }

        // A line along which the function falls forever has no minimiser to
        // find, and the search says so rather than returning a step.
        let linear = |_: &[f64]| vec![1.0, 0.0];
        assert!(exact_line_search(&linear, &[0.0, 0.0], &[-1.0, 0.0]).is_none());
        // A direction that is already stationary returns a zero step.
        assert_eq!(exact_line_search(&linear, &[0.0, 0.0], &[0.0, -1.0]), Some(0.0));
    }

    #[test]
    fn backtracking_returns_a_step_meeting_armijo_and_refuses_an_ascent_direction() {
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[1] * x[1];
        let g = |x: &[f64]| vec![2.0 * x[0], 6.0 * x[1]];
        let x = [2.0, 1.0];
        let gradient = g(&x);
        let direction: Vec<f64> = gradient.iter().map(|v| -v).collect();
        let t = backtracking(&f, &x, &direction, &gradient, 1e-4, 60);
        let moved: Vec<f64> = x.iter().zip(&direction).map(|(a, d)| a + t * d).collect();
        assert!(f(&moved) <= f(&x) + 1e-4 * t * dot(&gradient, &direction) + 1e-12);
        assert!(t > 0.0 && t <= 1.0);

        // An ascent direction is a programming error, not a slow step.
        let uphill: Vec<f64> = gradient.clone();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backtracking(&f, &x, &uphill, &gradient, 1e-4, 10)
        }))
        .is_err());
    }

    // -----------------------------------------------------------------
    // Proximal operators
    // -----------------------------------------------------------------

    #[test]
    fn each_proximal_operator_minimises_the_problem_that_defines_it() {
        // A proximal operator is defined as the minimiser of
        // ||x - v||^2 / 2 + t h(x). Checking that directly, against a fine
        // grid, is stronger than checking the formula was transcribed right.
        let v = [1.4f64, -0.3, 0.05, -2.2];
        let t = 0.5f64;

        let l1 = prox_l1(&v, t);
        let objective_l1 = |x: &[f64]| -> f64 {
            0.5 * x.iter().zip(&v).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
                + t * x.iter().map(|a| a.abs()).sum::<f64>()
        };
        // Separable, so each coordinate can be swept independently.
        for i in 0..v.len() {
            for k in -400..=400 {
                let mut trial = l1.clone();
                trial[i] = k as f64 * 0.01;
                assert!(
                    objective_l1(&trial) >= objective_l1(&l1) - 1e-12,
                    "prox_l1 is beaten at coordinate {i}"
                );
            }
        }
        // Values below the threshold go to exactly zero, which is the whole
        // reason L1 penalties select variables.
        assert_eq!(l1[2], 0.0, "a small coordinate did not vanish");
        assert!(l1[0] > 0.0 && l1[3] < 0.0, "the signs were not preserved");

        // The block operator zeroes the whole vector at once.
        let small = [0.1f64, 0.1];
        assert_eq!(prox_l2(&small, 1.0), vec![0.0, 0.0]);
        let large = prox_l2(&[3.0, 4.0], 1.0);
        assert!(close(norm(&large), 4.0, 1e-12), "the norm went to {}", norm(&large));
        // It shrinks without rotating.
        assert!(close(large[0] / large[1], 3.0 / 4.0, 1e-12));

        // A box projection is clamping.
        assert_eq!(prox_box(&[-2.0, 0.5, 9.0], -1.0, 1.0), vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn the_simplex_projection_is_the_closest_point_of_the_simplex() {
        let cases: Vec<Vec<f64>> = vec![
            vec![0.5, 0.4, 0.3],
            vec![-1.0, 2.0, 0.0],
            vec![3.0, 3.0, 3.0],
            vec![0.2, 0.3, 0.5],
            vec![-5.0, -5.0, 8.0],
        ];
        for v in cases {
            let p = prox_simplex(&v);
            assert!(close(p.iter().sum::<f64>(), 1.0, 1e-12), "{p:?} does not sum to one");
            assert!(p.iter().all(|x| *x >= -1e-12), "{p:?} has a negative entry");

            // Nothing on the simplex is closer, checked against a grid.
            let distance = |q: &[f64]| -> f64 {
                q.iter().zip(&v).map(|(a, b)| (a - b) * (a - b)).sum()
            };
            let steps = 200;
            for i in 0..=steps {
                for j in 0..=steps - i {
                    let a = i as f64 / steps as f64;
                    let b = j as f64 / steps as f64;
                    let candidate = [a, b, 1.0 - a - b];
                    assert!(
                        distance(&candidate) >= distance(&p) - 1e-9,
                        "{candidate:?} beats the projection {p:?} of {v:?}"
                    );
                }
            }
            // A point already on the simplex is left alone.
            if (v.iter().sum::<f64>() - 1.0).abs() < 1e-12 && v.iter().all(|x| *x >= 0.0) {
                for (a, b) in p.iter().zip(&v) {
                    assert!((a - b).abs() < 1e-12, "an interior point moved");
                }
            }
        }
        // Clamping and renormalising is a *different* point, which is why the
        // sorted shift is needed.
        let v = [3.0f64, 1.0, 0.0];
        let naive_total: f64 = v.iter().map(|x| x.max(0.0)).sum();
        let naive: Vec<f64> = v.iter().map(|x| x.max(0.0) / naive_total).collect();
        let exact = prox_simplex(&v);
        assert!(
            naive.iter().zip(&exact).any(|(a, b)| (a - b).abs() > 1e-6),
            "the naive projection happened to agree, so the test proves nothing"
        );
    }

    // -----------------------------------------------------------------
    // Proximal gradient methods
    // -----------------------------------------------------------------

    #[test]
    fn acceleration_earns_its_name_on_the_same_problem() {
        // FISTA and ISTA take the same two operations per step; the only
        // difference is where the gradient is evaluated. The rate should
        // differ visibly.
        let a = Matrix::from_rows(&[
            &[1.0, 0.2, 0.0],
            &[0.2, 1.0, 0.3],
            &[0.0, 0.3, 1.0],
            &[0.5, 0.1, 0.4],
        ])
        .unwrap();
        let b = [1.0f64, 2.0, 0.5, 1.2];
        let lambda = 0.05f64;
        let smooth_grad = |x: &[f64]| -> Vec<f64> {
            let residual: Vec<f64> = (0..4)
                .map(|r| (0..3).map(|j| a.get(r, j) * x[j]).sum::<f64>() - b[r])
                .collect();
            (0..3).map(|j| (0..4).map(|r| a.get(r, j) * residual[r]).sum()).collect()
        };
        let objective = |x: &[f64]| -> f64 {
            let residual: f64 = (0..4)
                .map(|r| ((0..3).map(|j| a.get(r, j) * x[j]).sum::<f64>() - b[r]).powi(2))
                .sum();
            0.5 * residual + lambda * x.iter().map(|v| v.abs()).sum::<f64>()
        };
        let prox = |v: &[f64], t: f64| prox_l1(v, lambda * t);
        let step = 0.3f64;
        let start = vec![0.0; 3];

        // The same iteration count, and FISTA should be closer.
        let ista = proximal_gradient(&smooth_grad, &prox, &start, step, 40);
        let fast = fista(&smooth_grad, &prox, &start, step, 40);
        let settled = fista(&smooth_grad, &prox, &start, step, 20_000);
        let target = objective(&settled);
        assert!(
            objective(&fast) - target < objective(&ista) - target,
            "FISTA ({}) was not closer than ISTA ({}) to {target}",
            objective(&fast),
            objective(&ista)
        );
        // Both converge to the same place given enough steps.
        let slow = proximal_gradient(&smooth_grad, &prox, &start, step, 20_000);
        for (a, b) in slow.iter().zip(&settled) {
            assert!((a - b).abs() < 1e-6, "the two limits differ: {slow:?} against {settled:?}");
        }
    }

    #[test]
    fn the_constrained_methods_keep_their_iterates_feasible() {
        // Projected gradient onto a box, and Frank-Wolfe over a simplex,
        // which never projects at all.
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 5.0), 2.0 * (x[1] + 3.0)];
        let project = |x: &[f64]| prox_box(x, -1.0, 1.0);
        let x = projected_gradient(&grad, &project, &[0.0, 0.0], 0.1, 500);
        assert!(x.iter().all(|v| *v >= -1.0 - 1e-12 && *v <= 1.0 + 1e-12), "{x:?} left the box");
        // The unconstrained optimum is outside, so the answer is on the face.
        assert!(close(x[0], 1.0, 1e-9) && close(x[1], -1.0, 1e-9), "expected a corner, got {x:?}");

        // Frank-Wolfe over the simplex: the oracle returns the vertex with the
        // most negative gradient entry.
        let quadratic_grad = |x: &[f64]| -> Vec<f64> {
            vec![2.0 * (x[0] - 0.7), 2.0 * (x[1] - 0.2), 2.0 * (x[2] - 0.1)]
        };
        let oracle = |g: &[f64]| -> Vec<f64> {
            let best = (0..g.len())
                .min_by(|&a, &b| g[a].partial_cmp(&g[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0);
            (0..g.len()).map(|i| f64::from(u8::from(i == best))).collect()
        };
        let start = vec![1.0 / 3.0; 3];
        let fw = frank_wolfe(&quadratic_grad, &oracle, &start, 2000);
        assert!(close(fw.iter().sum::<f64>(), 1.0, 1e-9), "{fw:?} left the simplex");
        assert!(fw.iter().all(|v| *v >= -1e-9), "{fw:?} has a negative entry");
        // The target is already on the simplex, so it should be reached.
        for (a, b) in fw.iter().zip(&[0.7, 0.2, 0.1]) {
            assert!((a - b).abs() < 0.02, "Frank-Wolfe reached {fw:?}");
        }

        // Mirror descent stays on the simplex by construction.
        let md = mirror_descent_simplex(&quadratic_grad, &start, 0.5, 2000);
        assert!(close(md.iter().sum::<f64>(), 1.0, 1e-9), "{md:?} left the simplex");
        assert!(md.iter().all(|v| *v > 0.0), "{md:?} left the interior");
    }

    // -----------------------------------------------------------------
    // Regularised regression
    // -----------------------------------------------------------------

    #[test]
    fn ridge_satisfies_its_own_normal_equations_exactly() {
        let a = Matrix::from_rows(&[
            &[1.0, 0.5],
            &[0.3, 1.0],
            &[0.7, 0.2],
            &[1.5, 1.1],
        ])
        .unwrap();
        let b = [1.0f64, 2.0, 0.5, 3.0];
        for lambda in [0.0f64, 0.1, 1.0, 10.0] {
            let x = ridge_closed_form(&a, &b, lambda).unwrap();
            // (A'A + lambda I) x = A'b, checked entry by entry.
            for i in 0..2 {
                let lhs: f64 = (0..2)
                    .map(|j| {
                        let ata: f64 = (0..4).map(|r| a.get(r, i) * a.get(r, j)).sum();
                        (ata + if i == j { lambda } else { 0.0 }) * x[j]
                    })
                    .sum();
                let rhs: f64 = (0..4).map(|r| a.get(r, i) * b[r]).sum();
                assert!(
                    (lhs - rhs).abs() < 1e-9,
                    "lambda = {lambda}, row {i}: {lhs} against {rhs}"
                );
            }
        }
        // More penalty means a smaller solution, always.
        let mut previous = f64::INFINITY;
        for lambda in [0.01f64, 0.1, 1.0, 10.0, 100.0] {
            let n = norm(&ridge_closed_form(&a, &b, lambda).unwrap());
            assert!(n < previous, "the penalty did not shrink the fit at {lambda}");
            previous = n;
        }
        assert!(ridge_closed_form(&a, &[1.0], 0.1).is_err());
        assert!(ridge_closed_form(&a, &b, -1.0).is_err());
    }

    #[test]
    fn two_lasso_solvers_agree_and_produce_genuine_zeros() {
        // Coordinate descent and ADMM solve the same problem by entirely
        // different routes: one sweeps coordinates with a closed form, the
        // other alternates a linear solve with a threshold.
        let n = 40usize;
        let mut a = Matrix::zeros(n, 5);
        let mut b = vec![0.0; n];
        for r in 0..n {
            let t = r as f64 / n as f64;
            a.set(r, 0, 1.0);
            a.set(r, 1, t);
            a.set(r, 2, t * t);
            // Two columns that carry no signal at all.
            a.set(r, 3, (t * 13.0).sin());
            a.set(r, 4, (t * 29.0).cos());
            b[r] = 2.0 + 3.0 * t;
        }
        let lambda = 0.05f64;
        let cd = lasso_coordinate_descent(&a, &b, lambda, 2000).unwrap();
        let admm = admm_lasso(&a, &b, lambda, 1.0, 4000).unwrap();

        for (i, (x, y)) in cd.iter().zip(&admm).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "coordinate {i}: coordinate descent {x} against ADMM {y}"
            );
        }
        // The penalty produces exact zeros, which no smooth penalty does.
        let zeros = admm.iter().filter(|v| **v == 0.0).count();
        assert!(zeros >= 1, "the lasso produced no exact zeros: {admm:?}");
        // And a larger penalty produces more of them.
        let heavier = admm_lasso(&a, &b, 0.5, 1.0, 4000).unwrap();
        let more = heavier.iter().filter(|v| **v == 0.0).count();
        assert!(more >= zeros, "a heavier penalty did not zero more coefficients");

        assert!(lasso_coordinate_descent(&a, &[1.0], 0.1, 10).is_err());
        assert!(admm_lasso(&a, &b, 0.1, 0.0, 10).is_err());
    }

    #[test]
    fn the_elastic_net_reduces_to_its_two_endpoints() {
        let n = 30usize;
        let mut a = Matrix::zeros(n, 3);
        let mut b = vec![0.0; n];
        for r in 0..n {
            let t = r as f64 / n as f64;
            a.set(r, 0, 1.0);
            a.set(r, 1, t);
            a.set(r, 2, t * t);
            b[r] = 1.0 + 2.0 * t - 0.5 * t * t;
        }
        // With no L1 term it is ridge.
        let net = elastic_net(&a, &b, 0.0, 0.1, 5000).unwrap();
        let ridge = ridge_closed_form(&a, &b, 0.1 * n as f64).unwrap();
        for (x, y) in net.iter().zip(&ridge) {
            assert!((x - y).abs() < 1e-8, "elastic net {net:?} against ridge {ridge:?}");
        }
        // With no L2 term it is the lasso.
        let net = elastic_net(&a, &b, 0.05, 0.0, 5000).unwrap();
        let lasso = lasso_coordinate_descent(&a, &b, 0.05, 5000).unwrap();
        for (x, y) in net.iter().zip(&lasso) {
            assert!((x - y).abs() < 1e-8, "elastic net {net:?} against lasso {lasso:?}");
        }
        assert!(elastic_net(&a, &b, -1.0, 0.0, 10).is_err());
    }

    #[test]
    fn logistic_regression_drives_its_penalised_gradient_to_zero() {
        // The condition the fit is defined by, checked directly.
        let n = 60usize;
        let mut x = Matrix::zeros(n, 3);
        let mut y = vec![0.0; n];
        for r in 0..n {
            let t = r as f64 / n as f64 * 6.0 - 3.0;
            x.set(r, 0, 1.0);
            x.set(r, 1, t);
            x.set(r, 2, (t * 0.7).sin());
            y[r] = f64::from(u8::from(t + 0.3 * (t * 0.7).sin() > 0.2));
        }
        let lambda = 0.5f64;
        let beta = logistic_regression_fit(&x, &y, lambda, 100).unwrap();

        let mut gradient = vec![0.0; 3];
        for r in 0..n {
            let z: f64 = (0..3).map(|j| x.get(r, j) * beta[j]).sum();
            let p = 1.0 / (1.0 + (-z).exp());
            for i in 0..3 {
                gradient[i] += x.get(r, i) * (p - y[r]);
            }
        }
        for i in 0..3 {
            gradient[i] += lambda * beta[i];
        }
        assert!(norm(&gradient) < 1e-8, "the penalised gradient is {gradient:?}");

        // The fit separates the classes it was given.
        let correct = (0..n)
            .filter(|&r| {
                let z: f64 = (0..3).map(|j| x.get(r, j) * beta[j]).sum();
                f64::from(u8::from(z > 0.0)) == y[r]
            })
            .count();
        assert!(correct >= n - 2, "only {correct} of {n} were classified correctly");

        // A separable sample has no finite maximiser without a penalty, so
        // zero is refused rather than silently diverging.
        assert!(logistic_regression_fit(&x, &y, 0.0, 10).is_err());
        assert!(logistic_regression_fit(&x, &[0.5; 60], 0.1, 10).is_err());
        assert!(logistic_regression_fit(&x, &[1.0], 0.1, 10).is_err());
    }

    // -----------------------------------------------------------------
    // Constrained problems
    // -----------------------------------------------------------------

    #[test]
    fn the_quadratic_program_satisfies_its_kkt_conditions_exactly() {
        // Minimise x'x/2 - c'x subject to the coordinates summing to one.
        let q = Matrix::identity(3);
        let c = [-1.0f64, -2.0, -3.0];
        let a = Matrix::from_rows(&[&[1.0, 1.0, 1.0]]).unwrap();
        let b = [1.0f64];
        let (x, y) = quadratic_program_active_set(&q, &c, &a, &b).unwrap();

        // Feasibility.
        assert!(close(x.iter().sum::<f64>(), 1.0, 1e-9), "{x:?} is infeasible");
        // Stationarity: Q x + c + A' y = 0.
        for i in 0..3 {
            let stationarity: f64 = (0..3).map(|j| q.get(i, j) * x[j]).sum::<f64>() + c[i] + y[0];
            assert!(stationarity.abs() < 1e-9, "coordinate {i} is off by {stationarity}");
        }
        // And via the residual helper, which is what a caller would use.
        let gradient: Vec<f64> =
            (0..3).map(|i| (0..3).map(|j| q.get(i, j) * x[j]).sum::<f64>() + c[i]).collect();
        let residual = kkt_residual(
            &gradient,
            &[x.iter().sum::<f64>() - 1.0],
            &[vec![1.0, 1.0, 1.0]],
            &y,
        );
        assert!(residual < 1e-9, "the KKT residual is {residual}");

        // A point that is merely feasible has a large residual.
        let feasible = [1.0f64, 0.0, 0.0];
        let bad_gradient: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| q.get(i, j) * feasible[j]).sum::<f64>() + c[i])
            .collect();
        assert!(
            kkt_residual(&bad_gradient, &[0.0], &[vec![1.0, 1.0, 1.0]], &y) > 0.5,
            "a non-optimal feasible point had a small residual"
        );
        assert!(quadratic_program_active_set(&q, &c, &a, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn the_multiplier_update_is_what_lets_the_constraint_be_met_exactly() {
        // Minimise x^2 + y^2 subject to x + y = 2. The optimum is (1, 1) with
        // multiplier -2.
        let grad = |v: &[f64]| vec![2.0 * v[0], 2.0 * v[1]];
        let constraints = |v: &[f64]| vec![v[0] + v[1] - 2.0];
        let constraint_gradients = |_: &[f64]| vec![vec![1.0, 1.0]];

        let (x, y) = augmented_lagrangian(
            &grad,
            &constraints,
            &constraint_gradients,
            &[0.0, 0.0],
            1.0,
            12,
            400,
            0.05,
        );
        assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6, "got {x:?}");
        assert!((y[0] - (-2.0)).abs() < 1e-4, "the multiplier is {}", y[0]);
        assert!(constraints(&x)[0].abs() < 1e-7, "the constraint is violated by {}", constraints(&x)[0]);

        // The plain penalty method cannot do the same at any finite penalty,
        // and the gap is available in closed form. Minimising
        // x^2 + y^2 + (mu/2)(x + y - 2)^2 gives x = y = mu / (1 + mu), so the
        // violation is exactly -2 / (1 + mu): it vanishes only as mu grows
        // without bound. Four outer rounds at a tenfold increase end at
        // mu = 1000. The inner step has to be small enough for the curvature
        // that penalty brings -- the Hessian's top eigenvalue is 2 + 2 mu --
        // which is the ill conditioning the multiplier method avoids.
        let p = penalty_method(
            &grad,
            &constraints,
            &constraint_gradients,
            &[0.0, 0.0],
            1.0,
            4,
            40_000,
            4e-4,
        );
        let expected = -2.0 / 1001.0;
        assert!(
            (constraints(&p)[0] - expected).abs() < 1e-6,
            "the penalty method left {} where the closed form says {expected}",
            constraints(&p)[0]
        );
        assert!(
            constraints(&p)[0].abs() > 1000.0 * constraints(&x)[0].abs(),
            "the penalty method matched the multiplier method: {} against {}",
            constraints(&p)[0].abs(),
            constraints(&x)[0].abs()
        );
    }

    #[test]
    fn dual_ascent_recovers_the_same_answer_where_it_applies() {
        // Minimise x^2 + y^2 subject to x + y = 2, with the inner problem
        // solved exactly: the Lagrangian minimiser is (-y/2, -y/2).
        let minimise = |y: &[f64]| vec![-y[0] / 2.0, -y[0] / 2.0];
        let constraints = |v: &[f64]| vec![v[0] + v[1] - 2.0];
        let (x, y) = dual_ascent(&minimise, &constraints, &[0.0], 0.5, 500);
        assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6, "got {x:?}");
        assert!((y[0] - (-2.0)).abs() < 1e-6, "the multiplier is {}", y[0]);
        assert!(constraints(&x)[0].abs() < 1e-6);
    }

    #[test]
    fn the_generic_admm_splits_a_problem_into_two_easy_halves() {
        // Least squares subject to a box, split as a smooth part and an
        // indicator. Neither half is hard alone.
        let target = [3.0f64, -2.0, 0.4];
        let prox_f = |v: &[f64], t: f64| -> Vec<f64> {
            // The proximal operator of ||x - target||^2 / 2.
            v.iter().zip(&target).map(|(a, b)| (a + t * b) / (1.0 + t)).collect()
        };
        let prox_g = |v: &[f64], _: f64| prox_box(v, -1.0, 1.0);
        let x = admm_generic(&prox_f, &prox_g, &[0.0, 0.0, 0.0], 1.0, 500);
        // The answer is the target clamped into the box.
        assert!(close(x[0], 1.0, 1e-6) && close(x[1], -1.0, 1e-6), "got {x:?}");
        assert!(close(x[2], 0.4, 1e-6), "got {x:?}");
        assert!(x.iter().all(|v| *v >= -1.0 - 1e-9 && *v <= 1.0 + 1e-9));
    }

    // -----------------------------------------------------------------
    // First-order variants and diagnostics
    // -----------------------------------------------------------------

    #[test]
    fn the_adaptive_methods_all_reach_the_same_minimum() {
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + 5.0 * (x[1] + 1.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 2.0), 10.0 * (x[1] + 1.0)];
        let start = [0.0f64, 0.0];

        let results = [
            ("nesterov", nesterov(&g, &start, 0.02, 0.9, 5000)),
            ("adagrad", adagrad(&g, &start, 0.5, 20_000)),
            ("rmsprop", rmsprop(&g, &start, 0.01, 0.9, 20_000)),
            ("adamw", adamw(&g, &start, 0.01, 0.0, 20_000)),
        ];
        for (name, x) in &results {
            assert!((x[0] - 2.0).abs() < 1e-3, "{name} gave {x:?}");
            assert!((x[1] + 1.0).abs() < 1e-3, "{name} gave {x:?}");
            assert!(f(x) < 1e-5, "{name}'s value is {}", f(x));
        }

        // AdamW's decay pulls the answer toward the origin, which is the
        // point of it; the same nominal decay inside Adam's scaling would not
        // act the same way.
        let decayed = adamw(&g, &start, 0.01, 0.5, 20_000);
        assert!(
            decayed[0].abs() < 2.0 && decayed[0] > 0.0,
            "weight decay did not shrink the fit: {decayed:?}"
        );
    }

    #[test]
    fn the_subgradient_method_handles_a_kink_that_stops_a_gradient_method() {
        // Minimise |x - 3| + |x + 1|, which is flat between the kinks and has
        // no gradient at either.
        let f = |x: &[f64]| (x[0] - 3.0).abs() + (x[0] + 1.0).abs();
        let subgradient = |x: &[f64]| -> Vec<f64> {
            let a = if x[0] > 3.0 { 1.0 } else { -1.0 };
            let b = if x[0] > -1.0 { 1.0 } else { -1.0 };
            vec![a + b]
        };
        let (x, value) = subgradient_method(&f, &subgradient, &[10.0], 1.0, 5000);
        assert!(close(value, 4.0, 1e-3), "the minimum is 4, got {value} at {x:?}");
        assert!(x[0] >= -1.5 && x[0] <= 3.5, "the answer {x:?} is outside the flat region");
        // The best-so-far is tracked because a subgradient step can go uphill.
        assert!(value <= f(&[10.0]), "the method returned worse than its start");
    }

    #[test]
    fn the_convexity_check_refutes_but_does_not_prove() {
        let mut rng = Rng::new(0x_C0E0_0001);
        let bounds = vec![(-3.0, 3.0); 2];
        // Convex: no violating pair exists, so sampling finds none.
        let convex = |x: &[f64]| x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1];
        assert!(convexity_check_numeric(&convex, &bounds, 400, &mut rng));
        let absolute = |x: &[f64]| x[0].abs() + x[1].abs();
        assert!(convexity_check_numeric(&absolute, &bounds, 400, &mut rng));

        // Non-convex: a violating pair exists and sampling should find one.
        let wavy = |x: &[f64]| (x[0] * 3.0).sin() + (x[1] * 3.0).sin();
        assert!(!convexity_check_numeric(&wavy, &bounds, 400, &mut rng));
        let concave = |x: &[f64]| -(x[0] * x[0]) - x[1] * x[1];
        assert!(!convexity_check_numeric(&concave, &bounds, 400, &mut rng));
    }

    #[test]
    fn the_solvers_refuse_degenerate_input() {
        let f = |x: &[f64]| x[0] * x[0];
        let g = |x: &[f64]| vec![2.0 * x[0]];
        let h = |_: &[f64]| Matrix::identity(1);
        assert!(newton_method_nd(&f, &g, &h, &[], 1e-9, 10).is_err());
        assert!(bfgs(&f, &g, &[], 1e-9, 10).is_err());
        assert!(lbfgs(&f, &g, &[1.0], 0, 1e-9, 10).is_err());
        assert!(conjugate_gradient_nonlinear(&f, &g, &[], 1e-9, 10).is_err());
        assert!(trust_region_dogleg(&f, &g, &h, &[1.0], 0.0, 10).is_err());
        assert!(trust_region_dogleg(&f, &g, &h, &[], 1.0, 10).is_err());
    }
}
