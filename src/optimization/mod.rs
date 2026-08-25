// Numerical optimization algorithms: 1D search, gradient-based, derivative-free,
// and linear/nonlinear least-squares fitting.

pub mod least_squares;
pub mod convex;
pub mod game_theory;
pub mod integer;
pub mod lp;
pub mod metaheuristics;
pub mod network;

pub use least_squares::{
    fit_exponential_decay, fit_gaussian_peak, levenberg_marquardt, LmResult,
};

// ---------------------------------------------------------------------------
// Named constants — no magic numbers
// ---------------------------------------------------------------------------

const PHI: f64 = 1.618_033_988_749_895; // golden ratio
const RESPHI: f64 = 2.0 - PHI; // 1 - 1/φ  ≈ 0.381966…

const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPSILON: f64 = 1e-8;

const NM_ALPHA: f64 = 1.0; // reflection
const NM_GAMMA: f64 = 2.0; // expansion
const NM_RHO: f64 = 0.5; // contraction
const NM_SIGMA: f64 = 0.5; // shrink

const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;

// ---------------------------------------------------------------------------
// 1-D Optimization
// ---------------------------------------------------------------------------

/// Golden-section search for the minimum of `f` on `[a, b]`.
///
/// Returns the x value that minimizes f within tolerance `tol`.
#[must_use]
pub fn golden_section_min(
    f: &dyn Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    let mut x1 = a + RESPHI * (b - a);
    let mut x2 = b - RESPHI * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + RESPHI * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - RESPHI * (b - a);
            f2 = f(x2);
        }
    }
    (a + b) * 0.5
}

/// Brent's method for 1-D minimization, combining golden-section search with
/// parabolic interpolation.
#[must_use]
pub fn brent_min(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    let (mut lo, mut hi) = if a < b { (a, b) } else { (b, a) };
    let mut x = lo + RESPHI * (hi - lo);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;

    let mut d = 0.0_f64; // last step size
    let mut e = 0.0_f64; // step before last

    for _ in 0..max_iter {
        let mid = 0.5 * (lo + hi);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        if (x - mid).abs() <= tol2 - 0.5 * (hi - lo) {
            break;
        }

        // Try parabolic interpolation
        let mut use_golden = true;
        if (e).abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            if p.abs() < (0.5 * q * e).abs() && p > q * (lo - x) && p < q * (hi - x) {
                e = d;
                d = p / q;
                let u = x + d;
                if (u - lo) < tol2 || (hi - u) < tol2 {
                    d = if x < mid { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            e = if x < mid { hi - x } else { lo - x };
            d = RESPHI * e;
        }

        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + tol1.copysign(d)
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                hi = x;
            } else {
                lo = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || (w - x).abs() < f64::EPSILON {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < f64::EPSILON || (v - w).abs() < f64::EPSILON {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Central-difference numerical gradient of a scalar function of n variables.
#[must_use]
pub fn numerical_gradient_vec(f: &dyn Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut xp = x.to_vec();

    for i in 0..n {
        let orig = xp[i];
        xp[i] = orig + h;
        let fp = f(&xp);
        xp[i] = orig - h;
        let fm = f(&xp);
        grad[i] = (fp - fm) / (2.0 * h);
        xp[i] = orig;
    }
    grad
}

// ---------------------------------------------------------------------------
// Gradient-Based (multi-dimensional)
// ---------------------------------------------------------------------------

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|xi| xi * xi).sum::<f64>().sqrt()
}

/// Vanilla gradient descent: x ← x − α∇f.
#[must_use]
pub fn gradient_descent(
    _f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    learning_rate: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let mut x = x0.to_vec();

    for _ in 0..max_iter {
        let g = grad(&x);
        if vec_norm(&g) < tol {
            break;
        }
        for (xi, gi) in x.iter_mut().zip(g.iter()) {
            *xi -= learning_rate * gi;
        }
    }
    x
}

/// Gradient descent with momentum: v ← μv − α∇f, x ← x + v.
#[must_use]
pub fn gradient_descent_momentum(
    _f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    learning_rate: f64,
    momentum: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut v = vec![0.0; n];

    for _ in 0..max_iter {
        let g = grad(&x);
        if vec_norm(&g) < tol {
            break;
        }
        for i in 0..n {
            v[i] = momentum * v[i] - learning_rate * g[i];
            x[i] += v[i];
        }
    }
    x
}

/// Adam optimizer (β1=0.9, β2=0.999, ε=1e-8).
#[must_use]
pub fn adam(
    _f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    learning_rate: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0; n]; // first moment
    let mut v = vec![0.0; n]; // second moment

    for t in 1..=max_iter {
        let g = grad(&x);
        if vec_norm(&g) < tol {
            break;
        }
        let t_f = t as f64;
        for i in 0..n {
            m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * g[i];
            v[i] = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * g[i] * g[i];
            let m_hat = m[i] / (1.0 - ADAM_BETA1.powf(t_f));
            let v_hat = v[i] / (1.0 - ADAM_BETA2.powf(t_f));
            x[i] -= learning_rate * m_hat / (v_hat.sqrt() + ADAM_EPSILON);
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Derivative-Free
// ---------------------------------------------------------------------------

/// Nelder-Mead simplex algorithm for unconstrained minimization.
#[must_use]
pub fn nelder_mead(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    step: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = x0.len();
    let nv = n + 1; // number of simplex vertices

    // Build initial simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(nv);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut vertex = x0.to_vec();
        vertex[i] += step;
        simplex.push(vertex);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..max_iter {
        // Sort vertices by function value
        let mut order: Vec<usize> = (0..nv).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_fvals: Vec<f64> = order.iter().map(|&i| fvals[i]).collect();

        simplex = sorted_simplex;
        fvals = sorted_fvals;

        // Convergence check: spread of function values
        let f_spread = fvals[nv - 1] - fvals[0];
        if f_spread < tol {
            break;
        }

        // Centroid of all but worst
        let mut centroid = vec![0.0; n];
        for vertex in simplex.iter().take(n) {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += vertex[j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        // Reflection
        let worst = &simplex[nv - 1];
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + NM_ALPHA * (centroid[j] - worst[j]))
            .collect();
        let fr = f(&reflected);

        if fr < fvals[0] {
            // Try expansion
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + NM_GAMMA * (reflected[j] - centroid[j]))
                .collect();
            let fe = f(&expanded);
            if fe < fr {
                simplex[nv - 1] = expanded;
                fvals[nv - 1] = fe;
            } else {
                simplex[nv - 1] = reflected;
                fvals[nv - 1] = fr;
            }
        } else if fr < fvals[nv - 2] {
            simplex[nv - 1] = reflected;
            fvals[nv - 1] = fr;
        } else {
            // Contraction
            let base = if fr < fvals[nv - 1] {
                &reflected
            } else {
                &simplex[nv - 1]
            };
            let f_base = if fr < fvals[nv - 1] {
                fr
            } else {
                fvals[nv - 1]
            };
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + NM_RHO * (base[j] - centroid[j]))
                .collect();
            let fc = f(&contracted);
            if fc < f_base {
                simplex[nv - 1] = contracted;
                fvals[nv - 1] = fc;
            } else {
                // Shrink towards best
                let best = simplex[0].clone();
                for i in 1..nv {
                    for j in 0..n {
                        simplex[i][j] = best[j] + NM_SIGMA * (simplex[i][j] - best[j]);
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    // Return best vertex
    let best_idx = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    simplex[best_idx].clone()
}

/// Simple LCG pseudo-random number generator (deterministic, seeded from x0).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn from_seed(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        // Map to [0, 1)
        (self.state >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Simulated annealing for unconstrained minimization.
///
/// T = temp_initial × cooling_rate^iter. Accepts uphill moves with probability
/// exp(−ΔE / T). LCG seeded deterministically from `x0`.
#[must_use]
pub fn simulated_annealing(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    temp_initial: f64,
    cooling_rate: f64,
    step_size: f64,
    max_iter: usize,
) -> Vec<f64> {
    // Derive seed from x0 values
    let seed: u64 = x0
        .iter()
        .fold(0u64, |acc, &v| acc.wrapping_add(v.to_bits()));
    let mut rng = Lcg::from_seed(seed);

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut best_x = x.clone();
    let mut best_fx = fx;

    for iter in 0..max_iter {
        let temp = temp_initial * cooling_rate.powi(iter as i32);
        if temp < 1e-15 {
            break;
        }

        // Generate candidate by perturbing each dimension
        let candidate: Vec<f64> = x
            .iter()
            .map(|&xi| xi + step_size * (2.0 * rng.next_f64() - 1.0))
            .collect();
        let fc = f(&candidate);
        let delta = fc - fx;

        if delta < 0.0 || rng.next_f64() < (-delta / temp).exp() {
            x = candidate;
            fx = fc;
        }

        if fx < best_fx {
            best_x = x.clone();
            best_fx = fx;
        }
    }
    best_x
}

// ---------------------------------------------------------------------------
// Linear Least Squares
// ---------------------------------------------------------------------------

/// Ordinary linear regression: minimizes ‖a0 + a1·x − y‖₂ via
/// Householder-QR least squares (`linalg::qr::least_squares`).
///
/// Returns `(slope, intercept)`.
#[must_use]
pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "x and y must have equal length");
    assert!(x.len() >= 2, "linear_regression requires at least 2 points");
    let n = x.len();
    let mut a = crate::linalg::Matrix::zeros(n, 2);
    for (i, &xi) in x.iter().enumerate() {
        a.set(i, 0, 1.0);
        a.set(i, 1, xi);
    }
    match crate::linalg::qr::least_squares(&a, y) {
        Ok(c) => (c[1], c[0]),
        Err(_) => panic!("linear_regression: x-values have zero variance"),
    }
}

/// Fit a polynomial of the given degree to (x, y) data by QR least
/// squares on the Vandermonde matrix, falling back to normal equations
/// with Gaussian elimination when the system is rank deficient.
///
/// Returns coefficients `[a0, a1, …, a_degree]` such that
/// ŷ = a0 + a1·x + a2·x² + …
#[must_use]
pub fn polynomial_fit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    assert_eq!(x.len(), y.len(), "x and y must have equal length");
    let n = x.len();
    let m = degree + 1;

    if n >= m {
        let mut a = crate::linalg::Matrix::zeros(n, m);
        for (k, &xk) in x.iter().enumerate() {
            let mut p = 1.0;
            for j in 0..m {
                a.set(k, j, p);
                p *= xk;
            }
        }
        if let Ok(c) = crate::linalg::qr::least_squares(&a, y) {
            return c;
        }
    }

    // Rank-deficient or underdetermined input: legacy normal-equations
    // path, which returns finite (if non-unique) coefficients.
    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];

    for k in 0..n {
        let mut xi_pow = vec![1.0_f64; m];
        for j in 1..m {
            xi_pow[j] = xi_pow[j - 1] * x[k];
        }
        for i in 0..m {
            aty[i] += xi_pow[i] * y[k];
            for j in 0..m {
                ata[i][j] += xi_pow[i] * xi_pow[j];
            }
        }
    }

    gaussian_eliminate(&mut ata, &mut aty)
}

fn gaussian_eliminate(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let m = b.len();

    for col in 0..m {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..m {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let diag = a[col][col];
        if diag.abs() < 1e-15 {
            continue; // near-singular; skip
        }

        for row in (col + 1)..m {
            let factor = a[row][col] / diag;
            for j in col..m {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = b[i];
        for j in (i + 1)..m {
            s -= a[i][j] * x[j];
        }
        if a[i][i].abs() > 1e-15 {
            x[i] = s / a[i][i];
        }
    }
    x
}

/// Coefficient of determination R² = 1 − SS_res / SS_tot.
#[must_use]
pub fn r_squared(y_actual: &[f64], y_predicted: &[f64]) -> f64 {
    assert_eq!(
        y_actual.len(),
        y_predicted.len(),
        "y_actual and y_predicted must have equal length"
    );
    let n = y_actual.len() as f64;
    let mean_y: f64 = y_actual.iter().sum::<f64>() / n;

    let ss_res: f64 = y_actual
        .iter()
        .zip(y_predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    let ss_tot: f64 = y_actual.iter().map(|a| (a - mean_y).powi(2)).sum();

    if ss_tot < f64::EPSILON {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const LOOSE_TOL: f64 = 1e-3;
    const TIGHT_TOL: f64 = 1e-6;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -- 1-D optimization ---------------------------------------------------

    #[test]
    fn golden_section_finds_parabola_minimum() {
        let f = |x: f64| (x - 3.0).powi(2);
        let xmin = golden_section_min(&f, 0.0, 5.0, 1e-9, 1000);
        assert!(approx(xmin, 3.0, TIGHT_TOL), "got {xmin}");
    }

    #[test]
    fn brent_finds_parabola_minimum() {
        let f = |x: f64| (x - 3.0).powi(2);
        let xmin = brent_min(&f, 0.0, 5.0, 1e-12, 1000);
        assert!(approx(xmin, 3.0, TIGHT_TOL), "got {xmin}");
    }

    // -- Gradient-based -----------------------------------------------------

    #[test]
    fn gradient_descent_on_quadratic_bowl() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let result = gradient_descent(&f, &grad, &[5.0, -3.0], 0.1, 1e-8, 10_000);
        let (rx, ry) = (result[0], result[1]);
        assert!(approx(rx, 0.0, TIGHT_TOL), "x={rx}");
        assert!(approx(ry, 0.0, TIGHT_TOL), "y={ry}");
    }

    #[test]
    fn adam_converges_on_quadratic() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let result = adam(&f, &grad, &[5.0, -3.0], 0.1, 1e-6, 10_000);
        let (ax, ay) = (result[0], result[1]);
        assert!(approx(ax, 0.0, 1e-3), "x={ax}");
        assert!(approx(ay, 0.0, 1e-3), "y={ay}");
    }

    // -- Derivative-free ----------------------------------------------------

    #[test]
    fn nelder_mead_on_rosenbrock() {
        let rosenbrock =
            |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let result = nelder_mead(&rosenbrock, &[-1.0, 1.0], 0.5, 1e-12, 100_000);
        let (rx, ry) = (result[0], result[1]);
        assert!(
            approx(rx, 1.0, LOOSE_TOL) && approx(ry, 1.0, LOOSE_TOL),
            "got ({rx}, {ry})",
        );
    }

    // -- Linear least squares -----------------------------------------------

    #[test]
    fn linear_regression_exact_line() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let (m, b) = linear_regression(&x, &y);
        assert!(approx(m, 2.0, TIGHT_TOL), "slope={m}");
        assert!(approx(b, 1.0, TIGHT_TOL), "intercept={b}");
    }

    #[test]
    fn polynomial_fit_quadratic() {
        let x: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 + 2.0 * xi + 0.5 * xi * xi).collect();
        let coeffs = polynomial_fit(&x, &y, 2);
        let (a0, a1, a2) = (coeffs[0], coeffs[1], coeffs[2]);
        assert!(approx(a0, 3.0, TIGHT_TOL), "a0={a0}");
        assert!(approx(a1, 2.0, TIGHT_TOL), "a1={a1}");
        assert!(approx(a2, 0.5, TIGHT_TOL), "a2={a2}");
    }

    #[test]
    fn r_squared_perfect_fit() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = actual.clone();
        let r2 = r_squared(&actual, &predicted);
        assert!(approx(r2, 1.0, TIGHT_TOL), "R²={r2}");
    }

    #[test]
    fn numerical_gradient_matches_analytical() {
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let point = [3.0, 4.0];
        let ng = numerical_gradient_vec(&f, &point, 1e-7);
        // Analytical: [2x, 4y] = [6, 16]
        let (dfdx, dfdy) = (ng[0], ng[1]);
        assert!(approx(dfdx, 6.0, LOOSE_TOL), "dfdx={dfdx}");
        assert!(approx(dfdy, 16.0, LOOSE_TOL), "dfdy={dfdy}");
    }

    #[test]
    fn gradient_descent_momentum_on_quadratic() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let result = gradient_descent_momentum(&f, &grad, &[5.0, -3.0], 0.01, 0.9, 1e-8, 10_000);
        let (mx, my) = (result[0], result[1]);
        assert!(approx(mx, 0.0, LOOSE_TOL), "x={mx}");
        assert!(approx(my, 0.0, LOOSE_TOL), "y={my}");
    }

    #[test]
    fn simulated_annealing_finds_approximate_minimum() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let result = simulated_annealing(&f, &[0.0, 0.0], 10.0, 0.9999, 0.1, 100_000);
        let (rx, ry) = (result[0], result[1]);
        assert!(
            approx(rx, 3.0, 0.5) && approx(ry, -2.0, 0.5),
            "SA should find approximate minimum, got ({rx}, {ry})",
        );
    }

    #[test]
    fn brent_finds_asymmetric_minimum() {
        let f = |x: f64| (x - 7.0).powi(2) + 0.01 * (x - 7.0).powi(3);
        let xmin = brent_min(&f, 0.0, 15.0, 1e-12, 1000);
        assert!(approx(xmin, 7.0, LOOSE_TOL), "got {xmin}");
    }

    #[test]
    fn nelder_mead_on_steep_quadratic() {
        let f = |x: &[f64]| 100.0 * x[0] * x[0] + x[1] * x[1];
        let result = nelder_mead(&f, &[5.0, 5.0], 1.0, 1e-12, 100_000);
        let (sx, sy) = (result[0], result[1]);
        assert!(approx(sx, 0.0, LOOSE_TOL), "x={sx}");
        assert!(approx(sy, 0.0, LOOSE_TOL), "y={sy}");
    }

    #[test]
    fn r_squared_constant_actual() {
        let actual = vec![3.0, 3.0, 3.0, 3.0];
        let predicted = vec![3.0, 3.0, 3.0, 3.0];
        let r2 = r_squared(&actual, &predicted);
        assert!(approx(r2, 1.0, TIGHT_TOL), "R²={r2}");
    }

    #[test]
    fn simulated_annealing_cold_temperature() {
        let f = |x: &[f64]| x[0] * x[0];
        let result = simulated_annealing(&f, &[1.0], 1e-20, 0.99, 0.1, 100);
        assert!(result[0].is_finite());
    }

    #[test]
    fn polynomial_fit_linear() {
        let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 + 2.0 * xi).collect();
        let coeffs = polynomial_fit(&x, &y, 1);
        let (c0, c1) = (coeffs[0], coeffs[1]);
        assert!(approx(c0, 3.0, TIGHT_TOL), "a0={c0}");
        assert!(approx(c1, 2.0, TIGHT_TOL), "a1={c1}");
    }

    #[test]
    fn nelder_mead_triggers_shrink() {
        // Rastrigin-like function with many local features that force shrink steps
        let f = |x: &[f64]| {
            let a = 10.0;
            a * 2.0
                + (x[0] * x[0] - a * (2.0 * std::f64::consts::PI * x[0]).cos())
                + (x[1] * x[1] - a * (2.0 * std::f64::consts::PI * x[1]).cos())
        };
        let result = nelder_mead(&f, &[3.0, 4.0], 5.0, 1e-14, 50_000);
        assert!(result[0].is_finite() && result[1].is_finite());
    }

    #[test]
    fn polynomial_fit_overdetermined_near_singular() {
        // All x values identical: creates a near-singular Vandermonde matrix
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = vec![2.0, 2.1, 1.9, 2.0];
        let coeffs = polynomial_fit(&x, &y, 2);
        assert!(coeffs.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn brent_min_reversed_bounds() {
        let f = |x: f64| (x - 2.0).powi(2);
        let xmin = brent_min(&f, 5.0, 0.0, 1e-12, 1000);
        assert!(approx(xmin, 2.0, LOOSE_TOL), "got {xmin}");
    }
}
