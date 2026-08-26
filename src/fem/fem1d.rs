//! One-dimensional finite elements for `-(p u')' + q u = f`.
//!
//! # The weak form
//!
//! The strong form asks for a function whose second derivative satisfies
//! the equation pointwise. Multiplying by a test function `v` that
//! vanishes wherever `u` is prescribed, integrating over the interval and
//! integrating the second-derivative term by parts gives
//!
//! ```text
//! a(u, v) = integral p u' v' + q u v dx = integral f v dx = L(v)
//! ```
//!
//! for every admissible `v`. Two things happened in that line. The
//! solution now needs only one derivative rather than two, so a
//! discontinuous `p` -- a layered material -- is admissible instead of
//! fatal. And the boundary term `[p u' v]` that integration by parts
//! produced is where flux conditions enter: prescribe nothing and the
//! method silently imposes zero flux, which is why Neumann conditions are
//! called *natural* and Dirichlet conditions, which have to be built into
//! the space, are called *essential*.
//!
//! # Why the answer is the best one available
//!
//! Galerkin's method asks for the identity to hold not for every `v` but
//! for every `v` in a finite dimensional subspace, and looks for `u_h` in
//! that same subspace. Subtracting the two statements gives Galerkin
//! orthogonality, `a(u - u_h, v_h) = 0` for every `v_h` in the space: the
//! error is `a`-orthogonal to everything representable. When `a` is
//! symmetric and positive definite it is an inner product, orthogonality
//! of the error is exactly the characterisation of an orthogonal
//! projection, and so
//!
//! ```text
//! ||u - u_h||_a  <=  ||u - v_h||_a   for every v_h in the space
//! ```
//!
//! with a constant of one. The finite element solution is not merely a
//! good approximation in the energy norm; it is *the* best one. Nothing in
//! a finite difference scheme corresponds to this. It is checked directly
//! against the nodal interpolant in the property tests.
//!
//! Equivalently, `u_h` minimises the energy `J(v) = a(v,v)/2 - L(v)` over
//! the space -- the Ritz view -- which is why refining a mesh can only
//! lower the computed energy: the coarse space sits inside the fine one.
//!
//! # A variable coefficient is averaged, not sampled
//!
//! Linear elements have a constant derivative on each element, so the
//! quadrature in the stiffness term integrates `p` against a constant and
//! reproduces its element *average* exactly. That has a consequence worth
//! knowing: the discrete bilinear form still agrees with the true one on
//! the element space itself, so `u_h` is the exact `a`-orthogonal
//! projection of the true solution rather than an approximation of one,
//! and the Pythagoras identity
//!
//! ```text
//! ||u - v_h||_a^2 = ||u - u_h||_a^2 + ||u_h - v_h||_a^2
//! ```
//!
//! holds to rounding for every `v_h` in the space. It is not an accident
//! of a smooth `p`: a `p` that jumps *within* an element is averaged the
//! same way, which is the sense in which a finite element method handles a
//! discontinuous coefficient gracefully rather than exactly.
//!
//! # Nodal exactness, and its limits
//!
//! For the pure Poisson problem `-u'' = f` with Dirichlet data, the linear
//! element solution is exact *at the nodes*, to machine precision, on any
//! mesh. The Green's function of `-d^2/dx^2` is piecewise linear with its
//! kink at the source point, so for a mesh node it lies in the element
//! space itself; pairing it against the orthogonal error gives
//! `(u - u_h)(x_i) = 0`. This is a property of the operator, not a lucky
//! cancellation, and it fails the moment either ingredient goes:
//!
//! - a variable `p` makes the Green's function piecewise `int dx/p`,
//!   which is not piecewise linear, and nodal exactness disappears;
//! - a reaction term `q` does the same;
//! - for quadratic elements the piecewise linear Green's function of a
//!   *vertex* is still in the space, so vertices stay exact, but the one
//!   belonging to a midside node kinks in the middle of an element and is
//!   not. Quadratic elements are exact at element vertices and merely
//!   third-order accurate at the midsides.
//!
//! Nodal exactness also needs the load `integral f phi_i` integrated
//! exactly. Assembly here uses five-point Gauss-Legendre per element,
//! exact through degree nine, so it holds to rounding for polynomial data
//! and to quadrature error otherwise.
//!
//! # Sign conventions
//!
//! Flux conditions are stated with the *outward* normal, so the same
//! [`Bc::Neumann`] value means the same physical thing at both ends:
//! `p du/dn = g`, which is `-p u'(a) = g` on the left and `p u'(b) = g` on
//! the right. [`Bc::Robin`] is `p du/dn + alpha u = g` in the same
//! convention, and keeps the stiffness matrix symmetric.

use crate::error::SolveError;

/// Five-point Gauss-Legendre abscissae on the reference interval
/// `[-1, 1]`, exact for polynomials through degree nine.
const GAUSS_X: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683_1,
    0.0,
    0.538_469_310_105_683_1,
    0.906_179_845_938_664,
];

/// Weights matching [`GAUSS_X`].
const GAUSS_W: [f64; 5] = [
    0.236_926_885_056_189_1,
    0.478_628_670_499_366_5,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_5,
    0.236_926_885_056_189_1,
];

/// A boundary condition at one end of the interval.
///
/// Flux conditions use the outward normal, so a given value means the
/// same physical thing at either end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bc {
    /// `u = value` at the endpoint. Essential: built into the space.
    Dirichlet(f64),
    /// `p du/dn = value` with `n` outward. Natural: enters through the
    /// boundary term, and a value of zero is what the weak form imposes
    /// on its own if nothing is said.
    Neumann(f64),
    /// `p du/dn + alpha u = g`, outward normal. Positive `alpha` adds to
    /// the diagonal and so keeps the problem coercive even with no
    /// Dirichlet end anywhere.
    Robin { alpha: f64, g: f64 },
}

/// A finite element solution, with the mesh it lives on.
///
/// The solver functions return bare nodal values to match the shape of
/// the rest of the crate; wrapping them here is what makes it possible to
/// ask for the value *between* nodes, which is what an error norm needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Fem1dSolution {
    /// Left end of the interval.
    pub a: f64,
    /// Right end of the interval.
    pub b: f64,
    /// Polynomial degree of the elements: 1 or 2.
    pub degree: usize,
    /// Nodal values, `degree * elements + 1` of them, evenly spaced.
    pub values: Vec<f64>,
}

impl Fem1dSolution {
    /// Wraps nodal values from one of the solvers.
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidArgument`] if the degree is not 1 or 2, the
    /// interval is empty, or the value count is not `degree * k + 1` for
    /// some positive `k`.
    pub fn new(a: f64, b: f64, degree: usize, values: Vec<f64>) -> Result<Self, SolveError> {
        if !(degree == 1 || degree == 2) {
            return Err(SolveError::InvalidArgument("degree must be 1 or 2"));
        }
        if !(a.is_finite() && b.is_finite()) || b <= a {
            return Err(SolveError::InvalidArgument("need a finite interval with a < b"));
        }
        if values.len() < degree + 1 || !(values.len() - 1).is_multiple_of(degree) {
            return Err(SolveError::InvalidArgument("value count does not match the degree"));
        }
        Ok(Self { a, b, degree, values })
    }

    /// The number of elements the mesh has.
    pub fn elements(&self) -> usize {
        (self.values.len() - 1) / self.degree
    }

    /// The mesh spacing, meaning the element width rather than the node
    /// spacing -- for quadratic elements the nodes sit twice as close.
    pub fn h(&self) -> f64 {
        (self.b - self.a) / self.elements() as f64
    }

    /// The coordinates of the nodes.
    pub fn nodes(&self) -> Vec<f64> {
        let step = (self.b - self.a) / (self.values.len() - 1) as f64;
        (0..self.values.len()).map(|i| self.a + i as f64 * step).collect()
    }

    /// Locates `x` in the mesh, returning the element index and the
    /// reference coordinate `xi` in `[-1, 1]`.
    fn locate(&self, x: f64) -> (usize, f64) {
        let ne = self.elements();
        let h = self.h();
        let raw = ((x - self.a) / h).floor();
        // Clamping rather than refusing: a quadrature point can land a
        // rounding error outside the interval, and the polynomial on the
        // end element is the honest continuation there.
        let e = if raw < 0.0 {
            0
        } else if raw >= ne as f64 {
            ne - 1
        } else {
            raw as usize
        };
        let left = self.a + e as f64 * h;
        (e, 2.0 * (x - left) / h - 1.0)
    }

    /// Evaluates the piecewise polynomial at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let (e, xi) = self.locate(x);
        let base = e * self.degree;
        shape(self.degree, xi)
            .iter()
            .enumerate()
            .map(|(k, n)| n * self.values[base + k])
            .sum()
    }

    /// Evaluates the derivative at `x`.
    ///
    /// The derivative jumps at element boundaries -- a finite element
    /// solution is continuous but not smooth -- so the value returned
    /// there is the one from the element `x` was located in.
    pub fn eval_derivative(&self, x: f64) -> f64 {
        let (e, xi) = self.locate(x);
        let base = e * self.degree;
        let scale = 2.0 / self.h();
        shape_derivative(self.degree, xi)
            .iter()
            .enumerate()
            .map(|(k, d)| d * scale * self.values[base + k])
            .sum()
    }
}

/// Lagrange shape functions on the reference element `[-1, 1]`.
fn shape(degree: usize, xi: f64) -> Vec<f64> {
    if degree == 1 {
        vec![0.5 * (1.0 - xi), 0.5 * (1.0 + xi)]
    } else {
        vec![0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)]
    }
}

/// Their derivatives with respect to the reference coordinate.
fn shape_derivative(degree: usize, xi: f64) -> Vec<f64> {
    if degree == 1 {
        vec![-0.5, 0.5]
    } else {
        vec![xi - 0.5, -2.0 * xi, xi + 0.5]
    }
}

/// A symmetric banded matrix in upper storage: `data[i * w + k]` holds
/// `A[i][i + k]` for `k` up to the half-bandwidth.
struct Banded {
    n: usize,
    half: usize,
    data: Vec<f64>,
}

impl Banded {
    fn new(n: usize, half: usize) -> Self {
        Self { n, half, data: vec![0.0; n * (half + 1)] }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        if hi - lo > self.half {
            0.0
        } else {
            self.data[lo * (self.half + 1) + (hi - lo)]
        }
    }

    fn add(&mut self, i: usize, j: usize, v: f64) {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        self.data[lo * (self.half + 1) + (hi - lo)] += v;
    }

    fn set(&mut self, i: usize, j: usize, v: f64) {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        self.data[lo * (self.half + 1) + (hi - lo)] = v;
    }

    /// `A` times the all-ones vector, row by row.
    fn row_sums(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let lo = i.saturating_sub(self.half);
                let hi = (i + self.half).min(self.n - 1);
                (lo..=hi).map(|j| self.get(i, j)).sum()
            })
            .collect()
    }

    /// The largest diagonal entry in magnitude, used to scale the
    /// singularity tests.
    fn diagonal_scale(&self) -> f64 {
        (0..self.n).map(|i| self.get(i, i).abs()).fold(0.0, f64::max)
    }

    /// Solves `A x = rhs` by an `L D L^T` factorisation without pivoting.
    ///
    /// No pivoting is needed while the problem is coercive -- `p > 0` and
    /// `q >= 0` make the matrix positive definite, where the
    /// factorisation is unconditionally stable. A reaction term negative
    /// enough to push an eigenvalue through zero (a Helmholtz problem
    /// tuned to a resonance) is a genuinely singular operator, and it is
    /// reported as such rather than pivoted around.
    fn ldl_solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SolveError> {
        let n = self.n;
        let m = self.half;
        let scale = self.diagonal_scale().max(f64::MIN_POSITIVE);
        // l[i * m + (r - 1)] holds L[i][i - r].
        let mut l = vec![0.0; n * m];
        let mut d = vec![0.0; n];
        let at = |l: &[f64], i: usize, k: usize| -> f64 {
            if i == k {
                1.0
            } else if i > k && i - k <= m {
                l[i * m + (i - k - 1)]
            } else {
                0.0
            }
        };
        for j in 0..n {
            let mut dj = self.get(j, j);
            for k in j.saturating_sub(m)..j {
                let ljk = at(&l, j, k);
                dj -= ljk * ljk * d[k];
            }
            if dj.abs() <= 1e-13 * scale {
                return Err(SolveError::Singular);
            }
            d[j] = dj;
            for i in (j + 1)..(j + m + 1).min(n) {
                let mut s = self.get(i, j);
                for k in i.saturating_sub(m)..j {
                    s -= at(&l, i, k) * at(&l, j, k) * d[k];
                }
                l[i * m + (i - j - 1)] = s / dj;
            }
        }
        // Forward, diagonal, back.
        let mut y = rhs.to_vec();
        for i in 0..n {
            for k in i.saturating_sub(m)..i {
                y[i] -= at(&l, i, k) * y[k];
            }
        }
        for i in 0..n {
            y[i] /= d[i];
        }
        for i in (0..n).rev() {
            for k in (i + 1)..(i + m + 1).min(n) {
                y[i] -= at(&l, k, i) * y[k];
            }
        }
        Ok(y)
    }
}

/// Assembles and solves `-(p u')' + q u = f` with elements of the given
/// degree.
fn solve_degree(
    p: &dyn Fn(f64) -> f64,
    q: &dyn Fn(f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    bc: (Bc, Bc),
    n: usize,
    degree: usize,
) -> Result<Vec<f64>, SolveError> {
    if n == 0 {
        return Err(SolveError::InvalidArgument("need at least one element"));
    }
    if !(a.is_finite() && b.is_finite()) || b <= a {
        return Err(SolveError::InvalidArgument("need a finite interval with a < b"));
    }
    let h = (b - a) / n as f64;
    let nodes = degree * n + 1;
    let mut mat = Banded::new(nodes, degree);
    let mut rhs = vec![0.0; nodes];

    for e in 0..n {
        let left = a + e as f64 * h;
        let base = e * degree;
        for (&xi, &w) in GAUSS_X.iter().zip(GAUSS_W.iter()) {
            let x = left + 0.5 * (xi + 1.0) * h;
            let pv = p(x);
            let qv = q(x);
            let fv = f(x);
            if !(pv.is_finite() && qv.is_finite() && fv.is_finite()) {
                return Err(SolveError::InvalidArgument("coefficients must be finite"));
            }
            if pv <= 0.0 {
                return Err(SolveError::InvalidArgument("p must be positive"));
            }
            let sh = shape(degree, xi);
            let dsh = shape_derivative(degree, xi);
            // dx = (h/2) dxi and d/dx = (2/h) d/dxi, so the stiffness
            // term carries 2/h and the mass and load terms carry h/2.
            let stiff = 2.0 * w * pv / h;
            let mass = 0.5 * w * qv * h;
            let load = 0.5 * w * fv * h;
            for j in 0..=degree {
                rhs[base + j] += load * sh[j];
                for k in j..=degree {
                    mat.add(base + j, base + k, stiff * dsh[j] * dsh[k] + mass * sh[j] * sh[k]);
                }
            }
        }
    }

    // Natural and Robin conditions enter through the boundary term the
    // integration by parts left behind, with the outward normal on both
    // ends.
    for (end, cond) in [(0usize, bc.0), (nodes - 1, bc.1)] {
        match cond {
            Bc::Dirichlet(_) => {}
            Bc::Neumann(g) => {
                if !g.is_finite() {
                    return Err(SolveError::InvalidArgument("boundary data must be finite"));
                }
                rhs[end] += g;
            }
            Bc::Robin { alpha, g } => {
                if !(alpha.is_finite() && g.is_finite()) {
                    return Err(SolveError::InvalidArgument("boundary data must be finite"));
                }
                mat.add(end, end, alpha);
                rhs[end] += g;
            }
        }
    }

    // The constant function is in the kernel exactly when every row of
    // the assembled matrix sums to zero, which is what a pure-Neumann
    // problem with no reaction term gives. Detecting it here rather than
    // in the factorisation names the cause instead of reporting a small
    // pivot, and it is an exact test rather than a threshold on
    // conditioning.
    let has_dirichlet = matches!(bc.0, Bc::Dirichlet(_)) || matches!(bc.1, Bc::Dirichlet(_));
    if !has_dirichlet {
        let scale = mat.diagonal_scale().max(f64::MIN_POSITIVE);
        if mat.row_sums().iter().all(|s| s.abs() <= 1e-12 * scale) {
            return Err(SolveError::Singular);
        }
    }

    // Dirichlet data is eliminated symmetrically: the known value is
    // moved to the right-hand side of every equation that saw it, and
    // then its own row and column are replaced by the identity. Zeroing
    // the row alone would work but would destroy the symmetry that makes
    // the factorisation stable.
    for (end, cond) in [(0usize, bc.0), (nodes - 1, bc.1)] {
        if let Bc::Dirichlet(g) = cond {
            if !g.is_finite() {
                return Err(SolveError::InvalidArgument("boundary data must be finite"));
            }
            let lo = end.saturating_sub(degree);
            let hi = (end + degree).min(nodes - 1);
            for j in lo..=hi {
                if j != end {
                    rhs[j] -= mat.get(j, end) * g;
                    mat.set(j, end, 0.0);
                }
            }
            mat.set(end, end, 1.0);
            rhs[end] = g;
        }
    }

    mat.ldl_solve(&rhs)
}

/// Solves `-u'' = f` with linear elements on a uniform mesh of `n`
/// elements, returning the `n + 1` nodal values.
///
/// With exact load integration this is nodally exact -- see the module
/// documentation for why that is a property of the Laplacian rather than
/// of the discretisation.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an empty mesh, a degenerate
/// interval, or non-finite data; [`SolveError::Singular`] when both ends
/// carry a pure flux condition, which leaves the solution undetermined up
/// to an additive constant.
pub fn fem_1d_poisson(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    bc: (Bc, Bc),
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    solve_degree(&|_| 1.0, &|_| 0.0, f, a, b, bc, n, 1)
}

/// Solves `-(p u')' + q u = f` with linear elements, returning the
/// `n + 1` nodal values.
///
/// # Errors
///
/// As [`fem_1d_poisson`], and additionally
/// [`SolveError::InvalidArgument`] if `p` is not positive at a quadrature
/// point. A negative `q` large enough to make the operator indefinite is
/// reported as [`SolveError::Singular`].
pub fn fem_1d_general(
    p: &dyn Fn(f64) -> f64,
    q: &dyn Fn(f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    bc: (Bc, Bc),
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    solve_degree(p, q, f, a, b, bc, n, 1)
}

/// Solves `-(p u')' + q u = f` with quadratic elements, returning the
/// `2n + 1` nodal values: element vertices at the even indices and
/// midsides at the odd ones.
///
/// # Errors
///
/// As [`fem_1d_general`].
pub fn fem_1d_quadratic(
    p: &dyn Fn(f64) -> f64,
    q: &dyn Fn(f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    bc: (Bc, Bc),
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    solve_degree(p, q, f, a, b, bc, n, 2)
}

/// Integrates `g` element by element with five-point Gauss.
///
/// Splitting at the element boundaries is what makes this accurate: the
/// integrand involves the finite element derivative, which is
/// discontinuous there, and a global rule would be integrating across a
/// jump.
fn integrate_by_element(u_h: &Fem1dSolution, g: &dyn Fn(f64) -> f64) -> f64 {
    let h = u_h.h();
    let mut total = 0.0;
    for e in 0..u_h.elements() {
        let left = u_h.a + e as f64 * h;
        for (&xi, &w) in GAUSS_X.iter().zip(GAUSS_W.iter()) {
            let x = left + 0.5 * (xi + 1.0) * h;
            total += 0.5 * w * h * g(x);
        }
    }
    total
}

/// The `L2` norm of the error against an exact solution.
pub fn fem_1d_error_l2(u_h: &Fem1dSolution, u_exact: &dyn Fn(f64) -> f64) -> f64 {
    integrate_by_element(u_h, &|x| {
        let e = u_exact(x) - u_h.eval(x);
        e * e
    })
    .max(0.0)
    .sqrt()
}

/// The `H1` seminorm of the error: the `L2` norm of the derivative
/// difference alone.
///
/// For the Poisson problem this is the energy norm, up to the factor the
/// coefficient `p` contributes, and so it is the norm in which the finite
/// element solution is the best approximation available.
pub fn fem_1d_error_h1_seminorm(u_h: &Fem1dSolution, du_exact: &dyn Fn(f64) -> f64) -> f64 {
    integrate_by_element(u_h, &|x| {
        let e = du_exact(x) - u_h.eval_derivative(x);
        e * e
    })
    .max(0.0)
    .sqrt()
}

/// The full `H1` norm of the error, `sqrt(L2^2 + seminorm^2)`.
pub fn fem_1d_error_h1(
    u_h: &Fem1dSolution,
    u_exact: &dyn Fn(f64) -> f64,
    du_exact: &dyn Fn(f64) -> f64,
) -> f64 {
    let l2 = fem_1d_error_l2(u_h, u_exact);
    let semi = fem_1d_error_h1_seminorm(u_h, du_exact);
    l2.hypot(semi)
}

/// The observed order of convergence: the least-squares slope of
/// `ln(error)` against `ln(h)`.
///
/// A method converging as `C h^k` returns `k`. Fitting all the points
/// rather than taking the ratio of the last two is deliberate -- a single
/// ratio is a difference of two noisy logarithms and inherits the noise
/// of both.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] unless there are at least two pairs of
/// matching length, all strictly positive and finite, with at least two
/// distinct spacings.
pub fn convergence_rate(errors: &[f64], hs: &[f64]) -> Result<f64, SolveError> {
    if errors.len() != hs.len() {
        return Err(SolveError::DimensionMismatch { expected: errors.len(), got: hs.len() });
    }
    if errors.len() < 2 {
        return Err(SolveError::InvalidArgument("need at least two refinements"));
    }
    if errors.iter().chain(hs.iter()).any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(SolveError::InvalidArgument("errors and spacings must be positive"));
    }
    let n = errors.len() as f64;
    let lx: Vec<f64> = hs.iter().map(|h| h.ln()).collect();
    let ly: Vec<f64> = errors.iter().map(|e| e.ln()).collect();
    let mx = lx.iter().sum::<f64>() / n;
    let my = ly.iter().sum::<f64>() / n;
    let sxx: f64 = lx.iter().map(|x| (x - mx) * (x - mx)).sum();
    let sxy: f64 = lx.iter().zip(ly.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
    if sxx <= 0.0 {
        return Err(SolveError::InvalidArgument("need at least two distinct spacings"));
    }
    Ok(sxy / sxx)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    fn wrap(a: f64, b: f64, degree: usize, v: Vec<f64>) -> Fem1dSolution {
        Fem1dSolution::new(a, b, degree, v).unwrap()
    }

    #[test]
    fn linear_elements_are_nodally_exact_for_poisson() {
        // The Green's function of -d^2/dx^2 at a mesh node is piecewise
        // linear with its kink there, so it lies in the element space,
        // and pairing it against the orthogonal error kills the error at
        // that node. Nothing about h enters, so a three-element mesh is
        // exact at its nodes too.
        //
        // A polynomial load is used because nodal exactness needs the
        // load functional integrated exactly, and five-point Gauss is
        // exact only through degree nine.
        let u = |x: f64| 2.0 * x - x * x - x * x * x;
        for n in [3, 7, 40] {
            let v = fem_1d_poisson(
                &|x: f64| 2.0 + 6.0 * x,
                0.0,
                1.0,
                (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)),
                n,
            )
            .unwrap();
            for (i, got) in v.iter().enumerate() {
                let x = i as f64 / n as f64;
                assert!(
                    (got - u(x)).abs() < 1e-14,
                    "node {i} of {n} was off by {}",
                    got - u(x)
                );
            }
        }
    }

    #[test]
    fn nodal_exactness_degrades_only_by_the_load_quadrature() {
        // With a transcendental load the Galerkin argument still holds
        // exactly; what is no longer exact is the right-hand side. The
        // residual nodal error is therefore the quadrature error of the
        // five-point rule on that element, which falls off as h^11 and
        // is nothing like the h^2 of the solution itself. Measuring the
        // rate is what distinguishes the two: an assembly error would
        // show up as second order.
        let mut errors = Vec::new();
        let hs = [1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0];
        for n in [3usize, 4, 5] {
            let v = fem_1d_poisson(
                &|x: f64| PI * PI * (PI * x).sin(),
                0.0,
                1.0,
                (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)),
                n,
            )
            .unwrap();
            let worst = v
                .iter()
                .enumerate()
                .map(|(i, got)| {
                    let x = i as f64 / n as f64;
                    (got - (PI * x).sin()).abs()
                })
                .fold(0.0, f64::max);
            assert!(worst < 1e-10, "{n} elements were off by {worst}");
            errors.push(worst);
        }
        let rate = convergence_rate(&errors, &hs).unwrap();
        assert!(rate > 8.0, "nodal error fell off only as h^{rate}, not as the quadrature does");
    }

    #[test]
    fn the_patch_test_passes_at_both_degrees() {
        // A solution already inside the element space must come back
        // untouched. This is the oldest finite element check there is,
        // and it catches an assembly sign error immediately.
        let linear = fem_1d_poisson(&|_| 0.0, 0.0, 2.0, (Bc::Dirichlet(2.0), Bc::Dirichlet(8.0)), 5)
            .unwrap();
        for (i, got) in linear.iter().enumerate() {
            let x = 2.0 * i as f64 / 5.0;
            assert!((got - (2.0 + 3.0 * x)).abs() < 1e-12);
        }
        // x^2 has -u'' = -2.
        let quad = fem_1d_quadratic(
            &|_| 1.0,
            &|_| 0.0,
            &|_| -2.0,
            0.0,
            1.0,
            (Bc::Dirichlet(0.0), Bc::Dirichlet(1.0)),
            4,
        )
        .unwrap();
        for (i, got) in quad.iter().enumerate() {
            let x = i as f64 / 8.0;
            assert!((got - x * x).abs() < 1e-12, "midside {i} was off by {}", got - x * x);
        }
    }

    #[test]
    fn a_flux_condition_is_imposed_with_the_outward_normal() {
        // -u'' = 0 with u(0) = 0 and u'(1) = 3 is u = 3x. On the right
        // the outward normal is +x, so the prescribed outward flux is
        // the derivative itself.
        let right = fem_1d_poisson(&|_| 0.0, 0.0, 1.0, (Bc::Dirichlet(0.0), Bc::Neumann(3.0)), 6)
            .unwrap();
        assert!((right[6] - 3.0).abs() < 1e-12, "got {}", right[6]);
        // On the left the outward normal is -x, so the same value means
        // u'(0) = -3, and with u(1) = 0 the solution is 3 - 3x.
        let left = fem_1d_poisson(&|_| 0.0, 0.0, 1.0, (Bc::Neumann(3.0), Bc::Dirichlet(0.0)), 6)
            .unwrap();
        assert!((left[0] - 3.0).abs() < 1e-12, "got {}", left[0]);
    }

    #[test]
    fn a_pure_flux_problem_is_reported_as_singular() {
        // Both ends natural and no reaction term leaves the constant
        // function in the kernel: the solution is determined only up to
        // an additive constant, whatever the data.
        let e = fem_1d_poisson(&|_| 1.0, 0.0, 1.0, (Bc::Neumann(0.5), Bc::Neumann(-0.5)), 8);
        assert_eq!(e, Err(SolveError::Singular));
        // A reaction term removes the constant from the kernel and the
        // same boundary data becomes solvable.
        assert!(fem_1d_general(
            &|_| 1.0,
            &|_| 1.0,
            &|_| 1.0,
            0.0,
            1.0,
            (Bc::Neumann(0.5), Bc::Neumann(-0.5)),
            8
        )
        .is_ok());
        // So does a Robin end.
        assert!(fem_1d_poisson(
            &|_| 1.0,
            0.0,
            1.0,
            (Bc::Neumann(0.0), Bc::Robin { alpha: 2.0, g: 1.0 }),
            8
        )
        .is_ok());
    }

    #[test]
    fn a_robin_end_reproduces_its_own_algebra() {
        // -u'' = 0 with u(0) = 0 is u = c x, and u'(1) + alpha u(1) = g
        // fixes c = g / (1 + alpha).
        for alpha in [0.25, 1.0, 40.0] {
            let g = 2.0;
            let v = fem_1d_poisson(
                &|_| 0.0,
                0.0,
                1.0,
                (Bc::Dirichlet(0.0), Bc::Robin { alpha, g }),
                5,
            )
            .unwrap();
            let expect = g / (1.0 + alpha);
            assert!((v[5] - expect).abs() < 1e-12, "alpha {alpha}: got {} want {expect}", v[5]);
        }
    }

    #[test]
    fn a_variable_coefficient_converges_at_second_order() {
        // -( (1+x) u' )' = -u' - (1+x) u'' with u = sin(pi x).
        let p = |x: f64| 1.0 + x;
        let f = |x: f64| -PI * (PI * x).cos() + (1.0 + x) * PI * PI * (PI * x).sin();
        let u = |x: f64| (PI * x).sin();
        let mut errors = Vec::new();
        let mut hs = Vec::new();
        for n in [10, 20, 40, 80] {
            let v = fem_1d_general(
                &p,
                &|_| 0.0,
                &f,
                0.0,
                1.0,
                (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)),
                n,
            )
            .unwrap();
            errors.push(fem_1d_error_l2(&wrap(0.0, 1.0, 1, v), &u));
            hs.push(1.0 / n as f64);
        }
        let rate = convergence_rate(&errors, &hs).unwrap();
        assert!((rate - 2.0).abs() < 0.05, "L2 rate was {rate}");
    }

    #[test]
    fn quadratic_elements_are_exact_at_vertices_but_not_at_midsides() {
        // The vertex Green's function is piecewise linear and so lies in
        // the quadratic space; the midside one kinks in the middle of an
        // element and does not.
        let n = 8;
        let v = fem_1d_quadratic(
            &|_| 1.0,
            &|_| 0.0,
            &|x: f64| PI * PI * (PI * x).sin(),
            0.0,
            1.0,
            (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)),
            n,
        )
        .unwrap();
        let err = |i: usize| {
            let x = i as f64 / (2 * n) as f64;
            (v[i] - (PI * x).sin()).abs()
        };
        let vertex = (0..=2 * n).step_by(2).map(err).fold(0.0, f64::max);
        let midside = (1..2 * n).step_by(2).map(err).fold(0.0, f64::max);
        assert!(vertex < 1e-13, "vertices were off by {vertex}");
        assert!(midside > 1e3 * vertex, "midsides were as exact as the vertices");
    }

    #[test]
    fn quadratic_elements_converge_one_order_faster() {
        let u = |x: f64| (PI * x).sin();
        let du = |x: f64| PI * (PI * x).cos();
        let f = |x: f64| PI * PI * (PI * x).sin();
        let (mut l2, mut h1, mut hs) = (Vec::new(), Vec::new(), Vec::new());
        for n in [4, 8, 16, 32] {
            let v = fem_1d_quadratic(
                &|_| 1.0,
                &|_| 0.0,
                &f,
                0.0,
                1.0,
                (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)),
                n,
            )
            .unwrap();
            let s = wrap(0.0, 1.0, 2, v);
            l2.push(fem_1d_error_l2(&s, &u));
            h1.push(fem_1d_error_h1_seminorm(&s, &du));
            hs.push(1.0 / n as f64);
        }
        let rl2 = convergence_rate(&l2, &hs).unwrap();
        let rh1 = convergence_rate(&h1, &hs).unwrap();
        assert!((rl2 - 3.0).abs() < 0.05, "P2 L2 rate was {rl2}");
        assert!((rh1 - 2.0).abs() < 0.05, "P2 H1 rate was {rh1}");
    }

    #[test]
    fn the_convergence_rate_recovers_an_exact_power_law() {
        let hs: Vec<f64> = (1..6).map(|k| 0.5f64.powi(k)).collect();
        for k in [1.0, 2.0, 3.5] {
            let errors: Vec<f64> = hs.iter().map(|h| 7.0 * h.powf(k)).collect();
            let got = convergence_rate(&errors, &hs).unwrap();
            assert!((got - k).abs() < 1e-10, "wanted {k}, got {got}");
        }
        assert!(convergence_rate(&[1.0], &[1.0]).is_err());
        assert!(convergence_rate(&[1.0, 2.0], &[1.0]).is_err());
        assert!(convergence_rate(&[1.0, 0.0], &[1.0, 0.5]).is_err());
        assert!(convergence_rate(&[1.0, 2.0], &[0.5, 0.5]).is_err());
    }

    #[test]
    fn the_solution_wrapper_interpolates_and_differentiates() {
        let s = wrap(0.0, 1.0, 1, vec![0.0, 1.0, 4.0]);
        assert!((s.eval(0.25) - 0.5).abs() < 1e-14);
        assert!((s.eval_derivative(0.75) - 6.0).abs() < 1e-14);
        assert_eq!(s.elements(), 2);
        assert!((s.h() - 0.5).abs() < 1e-15);
        assert_eq!(s.nodes().len(), 3);
        // A quadratic through (0,0), (0.5,0.25), (1,1) is x^2.
        let q = wrap(0.0, 1.0, 2, vec![0.0, 0.25, 1.0]);
        assert!((q.eval(0.3) - 0.09).abs() < 1e-14);
        assert!((q.eval_derivative(0.3) - 0.6).abs() < 1e-14);
        assert!(Fem1dSolution::new(0.0, 1.0, 3, vec![0.0; 4]).is_err());
        assert!(Fem1dSolution::new(1.0, 0.0, 1, vec![0.0; 4]).is_err());
        assert!(Fem1dSolution::new(0.0, 1.0, 2, vec![0.0; 4]).is_err());
    }

    #[test]
    fn bad_arguments_are_refused() {
        let ok = (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0));
        assert!(fem_1d_poisson(&|_| 1.0, 0.0, 1.0, ok, 0).is_err());
        assert!(fem_1d_poisson(&|_| 1.0, 1.0, 1.0, ok, 4).is_err());
        assert!(fem_1d_poisson(&|_| f64::NAN, 0.0, 1.0, ok, 4).is_err());
        assert!(fem_1d_general(&|_| -1.0, &|_| 0.0, &|_| 1.0, 0.0, 1.0, ok, 4).is_err());
        assert!(fem_1d_poisson(&|_| 1.0, 0.0, 1.0, (Bc::Dirichlet(f64::NAN), ok.1), 4).is_err());
        assert!(
            fem_1d_poisson(&|_| 1.0, 0.0, 1.0, (Bc::Neumann(f64::INFINITY), ok.1), 4).is_err()
        );
    }

    #[test]
    fn a_reaction_term_is_assembled_with_the_right_sign() {
        // -u'' + u = f with u = e^x gives f = 0 exactly, so the solver
        // must reproduce the exponential from its boundary values alone.
        let n = 60;
        let v = fem_1d_general(
            &|_| 1.0,
            &|_| 1.0,
            &|_| 0.0,
            0.0,
            1.0,
            (Bc::Dirichlet(1.0), Bc::Dirichlet(std::f64::consts::E)),
            n,
        )
        .unwrap();
        let s = wrap(0.0, 1.0, 1, v);
        let err = fem_1d_error_l2(&s, &|x| x.exp());
        assert!(err < 1e-4, "L2 error was {err}");
        // A sign flip on the mass matrix would give sinh-like growth
        // instead; check the interior value directly.
        assert!((s.eval(0.5) - 0.5f64.exp()).abs() < 1e-4);
    }
}
