//! Solvers for the Schrodinger equation, stationary and time dependent.
//!
//! The stationary problem is an eigenvalue problem and the time-dependent one
//! is an initial value problem, and the two want different numerics. For the
//! first, discretising the Hamiltonian gives a symmetric matrix whose
//! eigenvalues converge to the true spectrum from below at second order in
//! the grid; for the second, what matters is not local accuracy but
//! *unitarity*, because an integrator that loses norm loses probability and
//! one that gains it manufactures particles from nothing. Both methods
//! offered here are unitary by construction rather than by accident: the
//! split-operator method applies exponentials of Hermitian operators, and
//! Crank-Nicolson applies a Cayley transform, which is unitary for any step
//! size at all.
//!
//! Everything takes `hbar` and the mass explicitly, so `hbar = m = 1` is
//! available for the cases with exact answers.

use crate::error::GeomError;
use crate::fractals::Complex;
use crate::linalg::matrix::Matrix;
use crate::quantum::wavefunction::{harmonic_oscillator_eigenstate, Wavefunction1D};
use crate::transforms::fft::{fft, ifft};

fn scale(z: Complex, k: f64) -> Complex {
    Complex::new(z.re * k, z.im * k)
}

fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

// ---------------------------------------------------------------------------
// The stationary equation
// ---------------------------------------------------------------------------

/// The lowest `n_states` bound states on a grid, by second-order finite
/// differences with hard walls at the ends.
///
/// Returns the energies in ascending order and the matching normalised
/// eigenvectors. The discrete Laplacian is tridiagonal and symmetric, so the
/// eigenproblem is solved directly rather than iteratively.
///
/// The walls matter: this solves the problem on `[x_0, x_{n-1}]` with the
/// wavefunction pinned to zero just outside, so a state that has not decayed
/// by the edge of the grid is being confined by the box rather than by the
/// potential, and its energy is wrong. The error is `O(dx^2)` and one-sided:
/// the discrete Laplacian underestimates curvature, so the computed energies
/// sit below the true ones.
///
/// # Errors
/// Returns an error for an empty potential, a non-positive spacing, mass or
/// `hbar`, or if the eigensolver fails.
pub fn tise_solve_fd(
    v: &[f64],
    dx: f64,
    mass: f64,
    hbar: f64,
    n_states: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), GeomError> {
    let n = v.len();
    if n < 3 {
        return Err(GeomError::InvalidArgument("tise_solve_fd needs at least three points"));
    }
    if !(dx > 0.0) || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("tise_solve_fd requires positive parameters"));
    }
    if n_states == 0 || n_states > n {
        return Err(GeomError::InvalidArgument("tise_solve_fd: bad state count"));
    }
    let kinetic = hbar * hbar / (2.0 * mass * dx * dx);
    let diag: Vec<f64> = v.iter().map(|vi| 2.0 * kinetic + vi).collect();
    let off = vec![-kinetic; n - 1];

    let values = lowest_eigenvalues(&diag, &off, n_states);
    let mut states = Vec::with_capacity(n_states);
    for (i, &lambda) in values.iter().enumerate() {
        let mut column = inverse_iteration(&diag, &off, lambda, &states[..i])
            .ok_or(GeomError::Degenerate("inverse iteration failed to separate a state"))?;
        let norm = (column.iter().map(|c| c * c).sum::<f64>() * dx).sqrt();
        if norm > 0.0 {
            for c in &mut column {
                *c /= norm;
            }
        }
        // Fix the sign so the first substantial component is positive, which
        // makes successive calls comparable.
        if let Some(&first) = column.iter().find(|c| c.abs() > 1e-9) {
            if first < 0.0 {
                for c in &mut column {
                    *c = -*c;
                }
            }
        }
        states.push(column);
    }
    Ok((values, states))
}

/// The number of eigenvalues of a symmetric tridiagonal matrix strictly below
/// `sigma`, from the Sturm sequence.
///
/// The count of negative pivots in the `LDL'` factorisation of `T - sigma I`
/// equals the number of eigenvalues below `sigma`, by Sylvester's law of
/// inertia. That single fact turns the eigenvalue problem into a search: the
/// count is a step function of `sigma` with a jump at each eigenvalue, so
/// bisecting on it isolates any level by index without touching the others.
fn sturm_count(diag: &[f64], off_squared: &[f64], sigma: f64) -> usize {
    let tiny = 1e-300;
    let mut count = 0usize;
    let mut pivot = diag[0] - sigma;
    for i in 0..diag.len() {
        if i > 0 {
            pivot = diag[i] - sigma - off_squared[i - 1] / pivot;
        }
        // A zero pivot has to be nudged -- the next step divides by it -- and
        // the nudge has to happen *before* the sign is read, not after. Sign
        // first and nudge second miscounts every eigenvalue the shift lands
        // on exactly, which for a free particle on a uniform grid is half the
        // spectrum at once: every diagonal entry is equal, so the shift sits
        // on a zero pivot at every other step.
        if pivot.abs() < tiny {
            pivot = -tiny;
        }
        if pivot < 0.0 {
            count += 1;
        }
    }
    count
}

/// The `k` lowest eigenvalues of a symmetric tridiagonal matrix, ascending.
///
/// Bisection on the Sturm count. Costs `O(n k)` per bisection step and needs
/// no eigenvectors, against the `O(n^3)` of computing the whole spectrum with
/// its full orthogonal factor -- which for a grid of a few thousand points is
/// the difference between a second and several minutes.
fn lowest_eigenvalues(diag: &[f64], off: &[f64], k: usize) -> Vec<f64> {
    let n = diag.len();
    let off_squared: Vec<f64> = off.iter().map(|b| b * b).collect();
    // Gershgorin discs bound the whole spectrum.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let radius = (if i > 0 { off[i - 1].abs() } else { 0.0 })
            + (if i + 1 < n { off[i].abs() } else { 0.0 });
        lo = lo.min(diag[i] - radius);
        hi = hi.max(diag[i] + radius);
    }
    let span = (hi - lo).max(1.0);
    lo -= 1e-9 * span;
    hi += 1e-9 * span;

    (0..k)
        .map(|index| {
            let (mut a, mut b) = (lo, hi);
            for _ in 0..200 {
                let mid = 0.5 * (a + b);
                if sturm_count(diag, &off_squared, mid) > index {
                    b = mid;
                } else {
                    a = mid;
                }
                if b - a <= 1e-15 * span {
                    break;
                }
            }
            0.5 * (a + b)
        })
        .collect()
}

/// Solves `(T - shift I) x = rhs` for a symmetric tridiagonal `T`, with
/// partial pivoting.
///
/// Pivoting introduces a second superdiagonal, which is why the band is three
/// wide on the way out. Plain Thomas would be shorter and would divide by a
/// pivot that inverse iteration deliberately drives to zero.
fn tridiagonal_shifted_solve(
    diag: &[f64],
    off: &[f64],
    shift: f64,
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = diag.len();
    let mut d: Vec<f64> = diag.iter().map(|a| a - shift).collect();
    let mut sub: Vec<f64> = off.to_vec();
    let mut sup: Vec<f64> = off.to_vec();
    let mut sup2 = vec![0.0f64; n];
    let mut r = rhs.to_vec();

    for i in 0..n - 1 {
        if sub[i].abs() > d[i].abs() {
            // Swap rows i and i + 1. Row i gains an entry two columns along.
            let (old_d, old_sup) = (d[i], sup[i]);
            d[i] = sub[i];
            sup[i] = d[i + 1];
            sup2[i] = if i + 1 < n - 1 { sup[i + 1] } else { 0.0 };
            sub[i] = old_d;
            d[i + 1] = old_sup;
            if i + 1 < n - 1 {
                sup[i + 1] = 0.0;
            }
            r.swap(i, i + 1);
        }
        if d[i] == 0.0 {
            return None;
        }
        let factor = sub[i] / d[i];
        d[i + 1] -= factor * sup[i];
        if i + 1 < n - 1 {
            sup[i + 1] -= factor * sup2[i];
        }
        r[i + 1] -= factor * r[i];
    }
    if d[n - 1] == 0.0 {
        return None;
    }

    let mut x = vec![0.0f64; n];
    x[n - 1] = r[n - 1] / d[n - 1];
    if n >= 2 {
        x[n - 2] = (r[n - 2] - sup[n - 2] * x[n - 1]) / d[n - 2];
    }
    for i in (0..n.saturating_sub(2)).rev() {
        x[i] = (r[i] - sup[i] * x[i + 1] - sup2[i] * x[i + 2]) / d[i];
    }
    if x.iter().any(|c| !c.is_finite()) {
        return None;
    }
    Some(x)
}

/// The eigenvector for a known eigenvalue, by inverse iteration.
///
/// Solving `(T - lambda I) x = b` amplifies whichever component of `b` lies
/// along the eigenvector by `1 / (mu - lambda)`, so an accurate eigenvalue
/// makes one solve almost sufficient. The near-singularity that would worry a
/// linear solver is the whole mechanism here.
///
/// `already` holds the eigenvectors found so far; each iterate is
/// orthogonalised against them, which is what keeps a nearly degenerate pair
/// -- a double well's ground doublet, say -- from collapsing onto the same
/// vector.
fn inverse_iteration(
    diag: &[f64],
    off: &[f64],
    lambda: f64,
    already: &[Vec<f64>],
) -> Option<Vec<f64>> {
    let n = diag.len();
    // A deterministic starting vector with a component along essentially
    // anything: an equal one would be orthogonal to every odd state.
    let mut x: Vec<f64> = (0..n)
        .map(|k| ((k as f64 * 0.7548776662466927).fract() - 0.5) * 2.0)
        .collect();
    let magnitude = diag.iter().fold(0.0f64, |acc, a| acc.max(a.abs())).max(1.0);

    for attempt in 0..4 {
        // Nudge the shift on a retry, in case it landed on an exact pivot.
        let shift = lambda + f64::from(attempt) * 1e-11 * magnitude;
        let mut converged = false;
        for _ in 0..3 {
            orthogonalise(&mut x, already);
            let normalised = normalise(&mut x);
            if !normalised {
                return None;
            }
            let Some(next) = tridiagonal_shifted_solve(diag, off, shift, &x) else {
                break;
            };
            x = next;
            converged = true;
        }
        if converged {
            orthogonalise(&mut x, already);
            if normalise(&mut x) {
                return Some(x);
            }
        }
        x = (0..n).map(|k| ((k as f64 * 0.3819660112501051).fract() - 0.5) * 2.0).collect();
    }
    None
}

fn orthogonalise(x: &mut [f64], already: &[Vec<f64>]) {
    for previous in already {
        let projection: f64 = x.iter().zip(previous).map(|(a, b)| a * b).sum();
        let square: f64 = previous.iter().map(|b| b * b).sum();
        if square > 0.0 {
            let factor = projection / square;
            for (a, b) in x.iter_mut().zip(previous) {
                *a -= factor * b;
            }
        }
    }
}

fn normalise(x: &mut [f64]) -> bool {
    let norm = x.iter().map(|c| c * c).sum::<f64>().sqrt();
    if !(norm > 0.0) || !norm.is_finite() {
        return false;
    }
    for c in x.iter_mut() {
        *c /= norm;
    }
    true
}

/// Bound-state energies by Numerov shooting with node counting.
///
/// Integrates from both ends toward a matching point and looks for the energy
/// at which the logarithmic derivatives agree. Node counting is what makes
/// the search reliable: the number of zeros of the solution is a monotone
/// function of the trial energy, so it says *which* state a bracket contains
/// and turns a search over a continuum into a bisection per state.
///
/// Numerov itself is worth the extra terms: it integrates `y'' = f y` to
/// fourth order using only three points, because the equation's lack of a
/// first-derivative term lets the `O(h^4)` error be absorbed into the
/// coefficients.
///
/// Returns `(energy, wavefunction)` for each of the lowest `n_states` levels
/// found inside `e_range`.
///
/// # Errors
/// Returns an error for a degenerate grid or an inverted energy range.
pub fn tise_solve_numerov(
    v: &dyn Fn(f64) -> f64,
    x_range: (f64, f64),
    n: usize,
    e_range: (f64, f64),
    mass: f64,
    hbar: f64,
    n_states: usize,
) -> Result<Vec<(f64, Vec<f64>)>, GeomError> {
    let (x_lo, x_hi) = x_range;
    let (e_lo, e_hi) = e_range;
    if n < 5 || !(x_hi > x_lo) || !(e_hi > e_lo) {
        return Err(GeomError::InvalidArgument("tise_solve_numerov: bad ranges"));
    }
    if !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("tise_solve_numerov requires positive constants"));
    }
    let h = (x_hi - x_lo) / (n - 1) as f64;
    let factor = 2.0 * mass / (hbar * hbar);

    // One outward Numerov sweep at a trial energy, returning the solution and
    // the number of nodes it has.
    let sweep = |energy: f64| -> (Vec<f64>, usize) {
        let g: Vec<f64> = (0..n).map(|k| factor * (energy - v(x_lo + k as f64 * h))).collect();
        let mut y = vec![0.0; n];
        y[0] = 0.0;
        y[1] = 1e-8;
        let c = h * h / 12.0;
        for k in 1..n - 1 {
            let numerator = 2.0 * (1.0 - 5.0 * c * g[k]) * y[k] - (1.0 + c * g[k - 1]) * y[k - 1];
            y[k + 1] = numerator / (1.0 + c * g[k + 1]);
        }
        let nodes = (1..n - 1).filter(|&k| y[k] * y[k + 1] < 0.0).count();
        (y, nodes)
    };

    let mut found = Vec::new();
    for level in 0..n_states {
        // Bisect on the energy for which the sweep first has `level + 1`
        // nodes: node count is non-decreasing in energy, so the boundary is
        // where the state sits.
        let (mut lo, mut hi) = (e_lo, e_hi);
        let (_, nodes_hi) = sweep(hi);
        if nodes_hi <= level {
            break;
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let (_, nodes) = sweep(mid);
            if nodes > level {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo < 1e-13 * (1.0 + hi.abs()) {
                break;
            }
        }
        let energy = 0.5 * (lo + hi);
        let (mut y, _) = sweep(energy);
        let norm = (y.iter().map(|c| c * c).sum::<f64>() * h).sqrt();
        if norm > 0.0 {
            for c in &mut y {
                *c /= norm;
            }
        }
        found.push((energy, y));
    }
    Ok(found)
}

/// Which basis to expand the Hamiltonian in.
#[derive(Debug, Clone, Copy)]
pub enum Basis {
    /// Harmonic oscillator eigenstates of the given mass and frequency.
    HarmonicOscillator {
        /// The reference oscillator's mass.
        mass: f64,
        /// The reference oscillator's frequency.
        omega: f64,
    },
    /// Particle-in-a-box states on `[0, length]`.
    Box {
        /// The width of the box.
        length: f64,
    },
}

/// Bound states by expanding the Hamiltonian in a fixed basis and
/// diagonalising.
///
/// Rayleigh-Ritz: the energies are upper bounds on the eigenvalues of the
/// same Hamiltonian, and they fall monotonically as the basis grows. The
/// bound is against the *discretised* operator -- the same tridiagonal
/// [`tise_solve_fd`] uses -- not against the continuum, since a truncated
/// basis cannot bound what the grid has already changed.
///
/// The basis is orthonormalised on the grid before use, and that is not
/// tidiness. Sampling a basis at finitely many points and cutting it off at
/// the ends leaves it non-orthogonal, so `H c = E c` is the wrong problem;
/// the right one is `H c = E S c` with the overlap matrix `S`. Solving the
/// former with a non-orthonormal basis breaks the bound in the worst way --
/// it returns energies *below* the true ones, which looks like a better
/// answer rather than a wrong one.
///
/// Returns the energies in ascending order and the coefficient matrix in the
/// orthonormalised basis, whose column `i` holds the expansion of state `i`.
///
/// # Errors
/// Returns an error for an empty basis, a degenerate grid, an eigensolver
/// failure, or a basis that collapses to nothing on this grid.
pub fn tise_solve_matrix_basis(
    v: &[f64],
    dx: f64,
    x0: f64,
    basis: Basis,
    n_basis: usize,
    mass: f64,
    hbar: f64,
) -> Result<(Vec<f64>, Matrix), GeomError> {
    if n_basis == 0 || v.len() < 3 {
        return Err(GeomError::InvalidArgument("tise_solve_matrix_basis: bad size"));
    }
    if !(dx > 0.0) || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("tise_solve_matrix_basis: bad constants"));
    }
    let n = v.len();
    let x = |k: usize| x0 + k as f64 * dx;

    // The basis functions and their own kinetic-plus-reference energies.
    let phi: Vec<Vec<f64>> = (0..n_basis)
        .map(|i| match basis {
            Basis::HarmonicOscillator { mass: bm, omega } => (0..n)
                .map(|k| harmonic_oscillator_eigenstate(i, x(k), bm, omega, hbar))
                .collect(),
            Basis::Box { length } => (0..n)
                .map(|k| {
                    let xi = x(k);
                    if xi <= 0.0 || xi >= length {
                        0.0
                    } else {
                        (2.0 / length).sqrt()
                            * ((i + 1) as f64 * std::f64::consts::PI * xi / length).sin()
                    }
                })
                .collect(),
            })
        .collect();

    // Orthonormalise on the grid, discarding anything the sampling has made
    // linearly dependent.
    let mut orthonormal: Vec<Vec<f64>> = Vec::with_capacity(n_basis);
    for candidate in &phi {
        let mut f = candidate.clone();
        for previous in &orthonormal {
            let projection: f64 =
                f.iter().zip(previous).map(|(a, b)| a * b).sum::<f64>() * dx;
            for (a, b) in f.iter_mut().zip(previous) {
                *a -= projection * b;
            }
        }
        let norm = (f.iter().map(|a| a * a).sum::<f64>() * dx).sqrt();
        if norm > 1e-8 {
            for a in &mut f {
                *a /= norm;
            }
            orthonormal.push(f);
        }
    }
    let size = orthonormal.len();
    if size == 0 {
        return Err(GeomError::Degenerate("the basis vanishes on this grid"));
    }

    // The same discrete Hamiltonian the finite-difference solver uses, so the
    // two are bounds on one operator rather than on two different ones.
    let kinetic = hbar * hbar / (2.0 * mass * dx * dx);
    let apply = |f: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|k| {
                let mut acc = (2.0 * kinetic + v[k]) * f[k];
                if k > 0 {
                    acc -= kinetic * f[k - 1];
                }
                if k + 1 < n {
                    acc -= kinetic * f[k + 1];
                }
                acc
            })
            .collect()
    };
    let applied: Vec<Vec<f64>> = orthonormal.iter().map(|f| apply(f)).collect();

    let mut h = Matrix::zeros(size, size);
    for i in 0..size {
        for j in i..size {
            let element: f64 =
                orthonormal[i].iter().zip(&applied[j]).map(|(a, b)| a * b).sum::<f64>() * dx;
            h.set(i, j, element);
            h.set(j, i, element);
        }
    }
    let decomposition = crate::linalg::eigen::eigen_symmetric(&h, 1e-12, 200)
        .map_err(|_| GeomError::Degenerate("the basis eigenproblem failed"))?;
    // The Jacobi solver sorts descending; bound states are wanted from the
    // ground state up, so both the values and the matching columns reverse.
    let values: Vec<f64> = decomposition.values.iter().rev().copied().collect();
    let vectors = Matrix::from_fn(size, size, |r, c| {
        decomposition.vectors.get(r, size - 1 - c)
    });
    Ok((values, vectors))
}

// ---------------------------------------------------------------------------
// Time evolution
// ---------------------------------------------------------------------------

/// Advances a wavefunction by the split-operator method.
///
/// Strang splitting: a half step of the potential, a full step of the kinetic
/// term in momentum space, and another half step of the potential. Each
/// factor is the exponential of a Hermitian operator and so is exactly
/// unitary, which is why the norm is conserved to rounding however large the
/// step is. What the step size controls is the *commutator* error between the
/// two -- second order for Strang against first for the naive ordering -- so
/// too large a step gives a wrong answer of exactly the right length.
///
/// # Errors
/// Returns an error for a mismatched potential, a non-power-of-two grid, or a
/// non-positive mass.
pub fn tdse_split_operator(
    psi: &mut Wavefunction1D,
    v: &[f64],
    dt: f64,
    steps: usize,
    mass: f64,
    hbar: f64,
) -> Result<(), GeomError> {
    if v.len() != psi.len() {
        return Err(GeomError::InvalidArgument("the potential has the wrong length"));
    }
    if !psi.len().is_power_of_two() {
        return Err(GeomError::InvalidArgument("split_operator needs a power-of-two grid"));
    }
    if !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("split_operator requires positive constants"));
    }
    let k = psi.wavenumbers();
    let half: Vec<Complex> = v.iter().map(|vi| cis(-vi * dt / (2.0 * hbar))).collect();
    let kinetic: Vec<Complex> =
        k.iter().map(|ki| cis(-hbar * ki * ki * dt / (2.0 * mass))).collect();

    for _ in 0..steps {
        for (z, factor) in psi.psi.iter_mut().zip(&half) {
            *z = *z * *factor;
        }
        let mut spectrum = fft(&psi.psi);
        for (z, factor) in spectrum.iter_mut().zip(&kinetic) {
            *z = *z * *factor;
        }
        psi.psi = ifft(&spectrum);
        for (z, factor) in psi.psi.iter_mut().zip(&half) {
            *z = *z * *factor;
        }
    }
    Ok(())
}

/// Solves a complex tridiagonal system by the Thomas algorithm.
fn complex_thomas(
    lower: &[Complex],
    diag: &[Complex],
    upper: &[Complex],
    rhs: &[Complex],
) -> Option<Vec<Complex>> {
    let n = diag.len();
    let mut c = vec![Complex::new(0.0, 0.0); n];
    let mut d = vec![Complex::new(0.0, 0.0); n];
    let divide = |a: Complex, b: Complex| -> Option<Complex> {
        let denominator = b.norm_sq();
        if denominator < 1e-300 {
            return None;
        }
        let numerator = a * b.conjugate();
        Some(scale(numerator, 1.0 / denominator))
    };
    c[0] = divide(upper[0], diag[0])?;
    d[0] = divide(rhs[0], diag[0])?;
    for i in 1..n {
        let pivot = diag[i] - lower[i - 1] * c[i - 1];
        if i + 1 < n {
            c[i] = divide(upper[i], pivot)?;
        }
        d[i] = divide(rhs[i] - lower[i - 1] * d[i - 1], pivot)?;
    }
    let mut x = vec![Complex::new(0.0, 0.0); n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    Some(x)
}

/// Advances a wavefunction by Crank-Nicolson.
///
/// Applies `(1 + i H dt / 2 hbar)^{-1} (1 - i H dt / 2 hbar)`, the Cayley
/// transform of the Hamiltonian. For Hermitian `H` that is exactly unitary at
/// every step size -- not approximately, and not only in the small-step limit
/// -- which is the reason to prefer it to an explicit scheme here. An explicit
/// Euler step on the same equation has modulus strictly greater than one for
/// every non-zero step and blows up.
///
/// Unlike the split-operator method this needs no FFT, so it works on any
/// grid length, and it imposes hard walls at the ends rather than periodicity.
///
/// # Errors
/// Returns an error for a mismatched potential, a non-positive mass, or a
/// singular system.
pub fn tdse_crank_nicolson(
    psi: &mut Wavefunction1D,
    v: &[f64],
    dt: f64,
    steps: usize,
    mass: f64,
    hbar: f64,
) -> Result<(), GeomError> {
    let n = psi.len();
    if v.len() != n {
        return Err(GeomError::InvalidArgument("the potential has the wrong length"));
    }
    if n < 3 || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("crank_nicolson requires positive constants"));
    }
    let dx = psi.dx;
    let kinetic = hbar * hbar / (2.0 * mass * dx * dx);
    // H = tridiag(-kinetic, 2 kinetic + v, -kinetic).
    let alpha = dt / (2.0 * hbar);
    let lower: Vec<Complex> = vec![Complex::new(0.0, alpha * -kinetic); n - 1];
    let upper: Vec<Complex> = vec![Complex::new(0.0, alpha * -kinetic); n - 1];
    let diag: Vec<Complex> =
        v.iter().map(|vi| Complex::new(1.0, alpha * (2.0 * kinetic + vi))).collect();

    for _ in 0..steps {
        // The right-hand side applies (1 - i H dt / 2 hbar).
        let mut rhs = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            let mut acc = scale(psi.psi[i], 2.0 * kinetic + v[i]);
            if i > 0 {
                acc = acc - scale(psi.psi[i - 1], kinetic);
            }
            if i + 1 < n {
                acc = acc - scale(psi.psi[i + 1], kinetic);
            }
            // psi - i alpha H psi.
            rhs[i] = psi.psi[i] - Complex::new(0.0, alpha) * acc;
        }
        psi.psi = complex_thomas(&lower, &diag, &upper, &rhs)
            .ok_or(GeomError::Degenerate("the Crank-Nicolson system is singular"))?;
    }
    Ok(())
}

/// Adds an imaginary absorbing layer of the given width and strength to the
/// two ends of a complex potential.
///
/// A wavepacket that reaches the edge of a periodic grid wraps around and
/// interferes with itself, which looks exactly like physics and is not. An
/// absorbing layer removes the outgoing amplitude instead. The profile has to
/// turn on smoothly -- a sudden absorber reflects, which is the problem it
/// was added to solve -- so the strength here rises quadratically.
///
/// Returns the imaginary part to be subtracted from the Hamiltonian.
///
/// # Errors
/// Returns an error if the two layers would overlap or the strength is
/// negative.
pub fn absorbing_boundary_cap(
    n: usize,
    width: usize,
    strength: f64,
) -> Result<Vec<f64>, GeomError> {
    if width == 0 || 2 * width >= n {
        return Err(GeomError::InvalidArgument("the absorbing layers must fit and not meet"));
    }
    if strength < 0.0 {
        return Err(GeomError::InvalidArgument("the absorber strength must be non-negative"));
    }
    let mut cap = vec![0.0; n];
    for k in 0..width {
        let depth = (width - k) as f64 / width as f64;
        cap[k] = strength * depth * depth;
        cap[n - 1 - k] = strength * depth * depth;
    }
    Ok(cap)
}

/// Applies one step of an absorbing layer to a wavefunction, damping the
/// amplitude by `exp(-cap dt / hbar)`.
///
/// # Errors
/// Returns an error if the layer has the wrong length.
pub fn apply_absorber(
    psi: &mut Wavefunction1D,
    cap: &[f64],
    dt: f64,
    hbar: f64,
) -> Result<(), GeomError> {
    if cap.len() != psi.len() {
        return Err(GeomError::InvalidArgument("the absorber has the wrong length"));
    }
    for (z, c) in psi.psi.iter_mut().zip(cap) {
        *z = scale(*z, (-c * dt / hbar).exp());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scattering and tunnelling
// ---------------------------------------------------------------------------

/// The transmission probability through an arbitrary piecewise-constant
/// barrier, by the transfer matrix method.
///
/// Each slice contributes a two-by-two matrix relating the amplitudes on its
/// two sides, and the product of them all relates the incoming wave to the
/// outgoing one. The method is exact for a piecewise-constant potential, so
/// its only error is the piecewise-constant approximation itself -- which
/// means a smooth barrier converges as the slices are refined, and a genuinely
/// rectangular one is exact at any resolution.
///
/// Below the barrier the wavenumber is imaginary and the same algebra
/// continues to work, which is where tunnelling comes from: the exponentially
/// decaying solution inside is not zero at the far side.
///
/// # Errors
/// Returns an error for an empty barrier, a non-positive width, mass or
/// `hbar`, or a non-positive energy.
pub fn transmission_coefficient(
    v: &[f64],
    dx: f64,
    energy: f64,
    mass: f64,
    hbar: f64,
) -> Result<f64, GeomError> {
    if v.is_empty() || !(dx > 0.0) || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("transmission_coefficient: bad parameters"));
    }
    if !(energy > 0.0) {
        return Err(GeomError::InvalidArgument("the incident energy must be positive"));
    }
    let factor = 2.0 * mass / (hbar * hbar);
    // The wavenumber in a region, as a complex number so that the classically
    // forbidden case needs no separate branch.
    let wavenumber = |potential: f64| -> Complex {
        let squared = factor * (energy - potential);
        if squared >= 0.0 {
            Complex::new(squared.sqrt(), 0.0)
        } else {
            Complex::new(0.0, (-squared).sqrt())
        }
    };

    let divide = |a: Complex, b: Complex| -> Complex {
        let denominator = b.norm_sq();
        if denominator < 1e-300 {
            return Complex::new(0.0, 0.0);
        }
        scale(a * b.conjugate(), 1.0 / denominator)
    };

    // Free on both sides, at zero potential.
    let outside = wavenumber(0.0);
    // Start with the identity and multiply in each interface and slab.
    let mut m = [[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]];
    let multiply = |a: [[Complex; 2]; 2], b: [[Complex; 2]; 2]| -> [[Complex; 2]; 2] {
        [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]
    };

    let mut previous = outside;
    for &potential in v {
        let k = wavenumber(potential);
        // Interface from `previous` to `k`.
        let ratio = divide(previous, k);
        let half = Complex::new(0.5, 0.0);
        let interface = [
            [half * (Complex::new(1.0, 0.0) + ratio), half * (Complex::new(1.0, 0.0) - ratio)],
            [half * (Complex::new(1.0, 0.0) - ratio), half * (Complex::new(1.0, 0.0) + ratio)],
        ];
        m = multiply(m, interface);
        // Propagation across the slab.
        let phase = k * Complex::new(0.0, dx);
        let forward = complex_exp(phase);
        let backward = complex_exp(scale(phase, -1.0));
        let slab = [
            [forward, Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), backward],
        ];
        m = multiply(m, slab);
        previous = k;
    }
    // The final interface back to free space.
    let ratio = divide(previous, outside);
    let half = Complex::new(0.5, 0.0);
    let interface = [
        [half * (Complex::new(1.0, 0.0) + ratio), half * (Complex::new(1.0, 0.0) - ratio)],
        [half * (Complex::new(1.0, 0.0) - ratio), half * (Complex::new(1.0, 0.0) + ratio)],
    ];
    m = multiply(m, interface);

    let denominator = m[0][0].norm_sq();
    if denominator < 1e-300 {
        return Ok(0.0);
    }
    Ok(1.0 / denominator)
}

fn complex_exp(z: Complex) -> Complex {
    scale(cis(z.im), z.re.exp())
}

/// The exact transmission probability through a rectangular barrier.
///
/// Three regimes in one formula. Below the barrier the transmission falls
/// exponentially with width, which is tunnelling; above it the transmission
/// oscillates and returns to one at the resonances where the barrier is a
/// whole number of half-wavelengths, which is the Ramsauer-Townsend effect
/// and has no classical counterpart at all -- classically, anything above the
/// barrier passes with certainty at every energy.
///
/// # Errors
/// Returns an error for a non-positive width, mass, `hbar` or energy.
pub fn tunneling_rectangular_exact(
    v0: f64,
    width: f64,
    energy: f64,
    mass: f64,
    hbar: f64,
) -> Result<f64, GeomError> {
    if !(width > 0.0) || !(mass > 0.0) || !(hbar > 0.0) || !(energy > 0.0) {
        return Err(GeomError::InvalidArgument("tunneling_rectangular_exact: bad parameters"));
    }
    if v0 == 0.0 {
        return Ok(1.0);
    }
    let factor = 2.0 * mass / (hbar * hbar);
    if energy < v0 {
        let kappa = (factor * (v0 - energy)).sqrt();
        let sinh = (kappa * width).sinh();
        Ok(1.0 / (1.0 + v0 * v0 * sinh * sinh / (4.0 * energy * (v0 - energy))))
    } else if energy > v0 {
        let k = (factor * (energy - v0)).sqrt();
        let sin = (k * width).sin();
        Ok(1.0 / (1.0 + v0 * v0 * sin * sin / (4.0 * energy * (energy - v0))))
    } else {
        // Exactly at the barrier top the limit of either branch.
        let k0 = (factor * energy).sqrt();
        Ok(1.0 / (1.0 + k0 * k0 * width * width / 4.0))
    }
}

/// The WKB tunnelling probability through a barrier between two turning
/// points.
///
/// `exp(-2 integral kappa dx)` over the classically forbidden region. It is
/// the leading exponential only: the prefactor is missing, so it is accurate
/// for a thick barrier and wrong by a factor of order one for a thin one. It
/// also diverges from the truth near the barrier top, where the turning
/// points merge and the approximation's own assumption -- that the wavelength
/// varies slowly -- fails exactly where it matters.
///
/// # Errors
/// Returns an error for an inverted interval or non-positive constants.
pub fn wkb_tunneling(
    v: &dyn Fn(f64) -> f64,
    energy: f64,
    turning_points: (f64, f64),
    mass: f64,
    hbar: f64,
    samples: usize,
) -> Result<f64, GeomError> {
    let (a, b) = turning_points;
    if !(b > a) || samples == 0 || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("wkb_tunneling: bad parameters"));
    }
    let h = (b - a) / samples as f64;
    let integral: f64 = (0..samples)
        .map(|k| {
            let x = a + (k as f64 + 0.5) * h;
            let gap = v(x) - energy;
            if gap > 0.0 {
                (2.0 * mass * gap).sqrt() / hbar
            } else {
                0.0
            }
        })
        .sum::<f64>()
        * h;
    Ok((-2.0 * integral).exp())
}

/// The Bohr-Sommerfeld energy of the `n`-th level: the energy at which the
/// action enclosed by the classical orbit is `(n + 1/2) 2 pi hbar`.
///
/// The half is the Maslov correction, one quarter of a cycle for each of the
/// two turning points. Without it the harmonic oscillator comes out with no
/// zero-point energy; with it the WKB spectrum of the oscillator is *exact*
/// at every level, which is a coincidence of the quadratic potential and not
/// a general property.
///
/// # Errors
/// Returns an error if no bracketing energy is found in `e_range`.
pub fn wkb_quantization(
    v: &dyn Fn(f64) -> f64,
    n: usize,
    e_range: (f64, f64),
    x_range: (f64, f64),
    mass: f64,
    hbar: f64,
    samples: usize,
) -> Result<f64, GeomError> {
    let (e_lo, e_hi) = e_range;
    let (x_lo, x_hi) = x_range;
    if !(e_hi > e_lo) || !(x_hi > x_lo) || samples == 0 || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("wkb_quantization: bad parameters"));
    }
    let target = (n as f64 + 0.5) * std::f64::consts::PI * hbar;
    // The action integral over the classically allowed region.
    let action = |energy: f64| -> f64 {
        let h = (x_hi - x_lo) / samples as f64;
        (0..samples)
            .map(|k| {
                let x = x_lo + (k as f64 + 0.5) * h;
                let gap = energy - v(x);
                if gap > 0.0 {
                    (2.0 * mass * gap).sqrt()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            * h
    };
    if action(e_lo) > target || action(e_hi) < target {
        return Err(GeomError::Degenerate("wkb_quantization: the level is outside the range"));
    }
    let (mut lo, mut hi) = (e_lo, e_hi);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if action(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-13 * (1.0 + hi.abs()) {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// The reflection probability at a potential step of height `v0`.
///
/// Non-zero even when the particle has more than enough energy to pass, which
/// has no classical analogue: a classical particle rolling over a downward
/// step always continues. Reflection here comes from the impedance mismatch
/// between the two wavenumbers, exactly as for light at a glass surface.
///
/// # Errors
/// Returns an error for a non-positive energy.
pub fn reflection_step_potential(v0: f64, energy: f64) -> Result<f64, GeomError> {
    if !(energy > 0.0) {
        return Err(GeomError::InvalidArgument("the incident energy must be positive"));
    }
    if energy <= v0 {
        // Total reflection: the transmitted wave is evanescent.
        return Ok(1.0);
    }
    let k1 = energy.sqrt();
    let k2 = (energy - v0).sqrt();
    let amplitude = (k1 - k2) / (k1 + k2);
    Ok(amplitude * amplitude)
}

/// The energy splitting of the lowest doublet in a symmetric double well.
///
/// The two lowest states are the symmetric and antisymmetric combinations of
/// the states localised in each well, and their energies differ by an amount
/// exponentially small in the barrier. A particle prepared in one well
/// oscillates to the other with period `2 pi hbar / splitting`, so the
/// splitting *is* the tunnelling rate -- a static spectral quantity carrying
/// entirely dynamical information.
///
/// # Errors
/// Returns an error if the finite-difference solve fails.
pub fn double_well_splitting(
    v: &[f64],
    dx: f64,
    mass: f64,
    hbar: f64,
) -> Result<f64, GeomError> {
    let (energies, _) = tise_solve_fd(v, dx, mass, hbar, 2)?;
    Ok(energies[1] - energies[0])
}

// ---------------------------------------------------------------------------
// Perturbation theory and the variational method
// ---------------------------------------------------------------------------

/// First-order energy shifts: the expectation of the perturbation in each
/// unperturbed state.
///
/// The whole of first order is a diagonal matrix element, which is why the
/// first-order shift of a state with a symmetry the perturbation breaks is so
/// often zero -- the integrand is odd. The Stark effect in hydrogen's ground
/// state is the standard case: no linear shift, because the ground state has
/// no permanent dipole.
///
/// # Errors
/// Returns an error if a state has the wrong length.
pub fn perturbation_theory_1st(
    states: &[Vec<f64>],
    perturbation: &[f64],
    dx: f64,
) -> Result<Vec<f64>, GeomError> {
    if states.is_empty() || !(dx > 0.0) {
        return Err(GeomError::InvalidArgument("perturbation_theory_1st: bad input"));
    }
    if states.iter().any(|s| s.len() != perturbation.len()) {
        return Err(GeomError::InvalidArgument("a state has the wrong length"));
    }
    Ok(states
        .iter()
        .map(|s| s.iter().zip(perturbation).map(|(c, p)| c * c * p).sum::<f64>() * dx)
        .collect())
}

/// Second-order energy shifts.
///
/// A sum over the other states of `|<m|V|n>|^2 / (E_n - E_m)`. The sign is
/// forced for the ground state: every other state lies above it, so every
/// term is negative and the ground state is always pushed *down* by a
/// perturbation at second order, whatever the perturbation is. That is
/// level repulsion, and it is why avoided crossings avoid.
///
/// # Errors
/// Returns an error on a length mismatch or degenerate levels, which
/// non-degenerate perturbation theory cannot treat.
pub fn perturbation_theory_2nd(
    states: &[Vec<f64>],
    energies: &[f64],
    perturbation: &[f64],
    dx: f64,
) -> Result<Vec<f64>, GeomError> {
    let n = states.len();
    if n == 0 || energies.len() != n || !(dx > 0.0) {
        return Err(GeomError::InvalidArgument("perturbation_theory_2nd: bad input"));
    }
    if states.iter().any(|s| s.len() != perturbation.len()) {
        return Err(GeomError::InvalidArgument("a state has the wrong length"));
    }
    let element = |i: usize, j: usize| -> f64 {
        states[i]
            .iter()
            .zip(&states[j])
            .zip(perturbation)
            .map(|((a, b), p)| a * b * p)
            .sum::<f64>()
            * dx
    };
    let mut out = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let gap = energies[i] - energies[j];
            if gap.abs() < 1e-12 {
                return Err(GeomError::Degenerate(
                    "non-degenerate perturbation theory needs distinct levels",
                ));
            }
            let v_ij = element(i, j);
            out[i] += v_ij * v_ij / gap;
        }
    }
    Ok(out)
}

/// The linear Stark shift of a hydrogen level in atomic units.
///
/// Zero for `n = 1` and `3 n (n_1 - n_2) / 2` times the field for the excited
/// levels, whose degeneracy the field lifts. The ground state's vanishing
/// first-order shift is the general rule -- a non-degenerate state with
/// definite parity has no permanent dipole -- and hydrogen's excited levels
/// are the exception because their accidental degeneracy mixes opposite
/// parities.
///
/// `parabolic_difference` is `n_1 - n_2` in the parabolic quantum numbers.
///
/// # Errors
/// Returns an error for `n = 0` or an out-of-range parabolic difference.
pub fn stark_shift_perturbative(
    field: f64,
    n: usize,
    parabolic_difference: i32,
) -> Result<f64, GeomError> {
    if n == 0 {
        return Err(GeomError::InvalidArgument("hydrogen levels are indexed from one"));
    }
    if parabolic_difference.unsigned_abs() as usize >= n && n > 1 {
        return Err(GeomError::InvalidArgument("the parabolic difference is out of range"));
    }
    if n == 1 {
        return Ok(0.0);
    }
    Ok(1.5 * n as f64 * f64::from(parabolic_difference) * field)
}

/// The variational ground state: minimises the expected energy of a trial
/// wavefunction over its parameters.
///
/// The bound is one-sided and it is exact: `<H>` over *any* normalisable
/// trial state is at least the true ground energy, because expanding the
/// trial state in eigenstates writes `<H>` as a weighted average of
/// eigenvalues. So a variational calculation can never accidentally report
/// too low an energy, and the only way to be wrong is to be too high.
///
/// Returns the minimised energy and the parameters that achieve it.
///
/// # Errors
/// Returns an error for an empty grid or parameter vector.
pub fn variational_ground_state(
    v: &[f64],
    dx: f64,
    x0: f64,
    trial: &dyn Fn(f64, &[f64]) -> f64,
    params0: &[f64],
    mass: f64,
    hbar: f64,
) -> Result<(f64, Vec<f64>), GeomError> {
    if v.len() < 3 || params0.is_empty() || !(dx > 0.0) || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("variational_ground_state: bad input"));
    }
    let n = v.len();
    let expectation = |params: &[f64]| -> f64 {
        let psi: Vec<f64> = (0..n).map(|k| trial(x0 + k as f64 * dx, params)).collect();
        let norm: f64 = psi.iter().map(|c| c * c).sum::<f64>() * dx;
        if norm <= 0.0 || !norm.is_finite() {
            return f64::INFINITY;
        }
        let mut total = 0.0;
        for k in 0..n {
            let second = if k == 0 || k + 1 == n {
                0.0
            } else {
                (psi[k + 1] - 2.0 * psi[k] + psi[k - 1]) / (dx * dx)
            };
            total += psi[k] * (-hbar * hbar / (2.0 * mass) * second + v[k] * psi[k]);
        }
        let energy = total * dx / norm;
        if energy.is_finite() {
            energy
        } else {
            f64::INFINITY
        }
    };
    let best = crate::optimization::nelder_mead(&expectation, params0, 0.2, 1e-12, 20_000);
    Ok((expectation(&best), best))
}

/// The ground state by propagation in imaginary time.
///
/// Replacing `t` with `-i tau` turns the oscillating phases `exp(-i E t)`
/// into decaying exponentials `exp(-E tau)`, so every excited component dies
/// faster than the ground state and what survives, renormalised, is the
/// ground state. The convergence rate is set by the gap `E_1 - E_0`, which
/// makes the method slow precisely for the nearly degenerate systems where
/// the answer is most delicate.
///
/// Returns the ground energy and the normalised state.
///
/// # Errors
/// Returns an error for a mismatched grid or non-positive constants.
pub fn imaginary_time_propagation(
    v: &[f64],
    dx: f64,
    dtau: f64,
    steps: usize,
    mass: f64,
    hbar: f64,
) -> Result<(f64, Vec<f64>), GeomError> {
    let n = v.len();
    if n < 3 || !(dx > 0.0) || !(dtau > 0.0) || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("imaginary_time_propagation: bad input"));
    }
    // Start from something with a component along the ground state. A
    // deliberately lopsided profile, so the test is not handed the answer.
    let mut psi: Vec<f64> = (0..n)
        .map(|k| {
            let t = k as f64 / (n - 1) as f64;
            (std::f64::consts::PI * t).sin() * (1.0 + 0.3 * (3.0 * t).cos())
        })
        .collect();
    let kinetic = hbar * hbar / (2.0 * mass * dx * dx);

    let normalise = |psi: &mut Vec<f64>| {
        let norm = (psi.iter().map(|c| c * c).sum::<f64>() * dx).sqrt();
        if norm > 0.0 {
            for c in psi.iter_mut() {
                *c /= norm;
            }
        }
    };
    normalise(&mut psi);

    for _ in 0..steps {
        // An explicit step of psi <- psi - dtau H psi / hbar, renormalised.
        let previous = psi.clone();
        for k in 0..n {
            let mut applied = (2.0 * kinetic + v[k]) * previous[k];
            if k > 0 {
                applied -= kinetic * previous[k - 1];
            }
            if k + 1 < n {
                applied -= kinetic * previous[k + 1];
            }
            psi[k] = previous[k] - dtau * applied / hbar;
        }
        normalise(&mut psi);
    }

    // The energy of whatever it converged to.
    let mut total = 0.0;
    for k in 0..n {
        let mut applied = (2.0 * kinetic + v[k]) * psi[k];
        if k > 0 {
            applied -= kinetic * psi[k - 1];
        }
        if k + 1 < n {
            applied -= kinetic * psi[k + 1];
        }
        total += psi[k] * applied;
    }
    Ok((total * dx, psi))
}

// ---------------------------------------------------------------------------
// Dynamics: theorems and models
// ---------------------------------------------------------------------------

/// The largest discrepancy in Ehrenfest's theorem along a trajectory.
///
/// `d<p>/dt = -<dV/dx>`: the expectations obey Newton's second law exactly,
/// with the force *averaged over the packet* rather than evaluated at its
/// centre. Those two differ as soon as the potential is not quadratic, which
/// is the precise sense in which a quantum particle is not a classical one --
/// and the reason a wavepacket in a harmonic well follows the classical orbit
/// forever while one in any other well does not.
///
/// # Errors
/// Returns an error for fewer than three snapshots or a mismatched potential.
pub fn ehrenfest_check(
    snapshots: &[Wavefunction1D],
    v: &[f64],
    dt: f64,
    hbar: f64,
    mass: f64,
) -> Result<f64, GeomError> {
    if snapshots.len() < 3 || !(dt > 0.0) || !(mass > 0.0) {
        return Err(GeomError::InvalidArgument("ehrenfest_check needs a trajectory"));
    }
    let n = snapshots[0].len();
    if v.len() != n {
        return Err(GeomError::InvalidArgument("the potential has the wrong length"));
    }
    let dx = snapshots[0].dx;
    // The force at each grid point, by central differences.
    let force: Vec<f64> = (0..n)
        .map(|k| {
            if k == 0 || k + 1 == n {
                0.0
            } else {
                -(v[k + 1] - v[k - 1]) / (2.0 * dx)
            }
        })
        .collect();

    let mut worst: f64 = 0.0;
    for i in 1..snapshots.len() - 1 {
        let before = hbar * snapshots[i - 1].expectation_k()?;
        let after = hbar * snapshots[i + 1].expectation_k()?;
        let rate = (after - before) / (2.0 * dt);

        let density = snapshots[i].probability_density();
        let weight: f64 = density.iter().sum::<f64>() * dx;
        if weight <= 0.0 {
            continue;
        }
        // Skip the outermost points, where the one-sided force is zero.
        let expected: f64 = (1..n - 1).map(|k| density[k] * force[k]).sum::<f64>() * dx / weight;
        worst = worst.max((rate - expected).abs());
    }
    let _ = mass;
    Ok(worst)
}

/// Scatters a wavepacket off a potential and returns the transmitted and
/// reflected probabilities.
///
/// The packet carries a spread of momenta, so what comes back is the
/// transmission averaged over that spread rather than the value at the mean
/// momentum. A narrow packet in position is broad in momentum, so the sharper
/// the incident pulse the more the measured coefficient is smeared -- the
/// uncertainty relation showing up as an experimental resolution limit.
///
/// # Errors
/// Returns an error for a mismatched grid or non-positive constants.
pub fn wavepacket_scattering(
    v: &[f64],
    dx: f64,
    x0: f64,
    barrier_centre: f64,
    k0: f64,
    sigma: f64,
    start: f64,
    dt: f64,
    steps: usize,
    mass: f64,
    hbar: f64,
) -> Result<(f64, f64), GeomError> {
    let n = v.len();
    if !n.is_power_of_two() || n < 8 {
        return Err(GeomError::InvalidArgument("wavepacket_scattering needs a power-of-two grid"));
    }
    let mut psi = Wavefunction1D::gaussian_packet(start, k0, sigma, dx, x0, n)?;
    tdse_split_operator(&mut psi, v, dt, steps, mass, hbar)?;

    let density = psi.probability_density();
    let total: f64 = density.iter().sum();
    if total <= 0.0 {
        return Ok((0.0, 0.0));
    }
    let split = ((barrier_centre - x0) / dx).round().clamp(0.0, (n - 1) as f64) as usize;
    let reflected: f64 = density[..split].iter().sum::<f64>() / total;
    let transmitted: f64 = density[split..].iter().sum::<f64>() / total;
    Ok((transmitted, reflected))
}

/// One-dimensional Gross-Pitaevskii evolution by split-step.
///
/// The condensate's mean field adds a term `g |psi|^2` to the potential, so
/// the equation is nonlinear and superposition fails. With `g < 0` the
/// attraction can balance dispersion exactly and the result is a bright
/// soliton that propagates without spreading -- which a free packet never
/// does, and which is the clearest signature that the nonlinearity is really
/// there.
///
/// # Errors
/// Returns an error for a mismatched grid or non-positive constants.
pub fn gross_pitaevskii_1d(
    psi: &mut Wavefunction1D,
    v: &[f64],
    g: f64,
    dt: f64,
    steps: usize,
    mass: f64,
    hbar: f64,
) -> Result<(), GeomError> {
    let n = psi.len();
    if v.len() != n {
        return Err(GeomError::InvalidArgument("the potential has the wrong length"));
    }
    if !n.is_power_of_two() || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("gross_pitaevskii_1d: bad grid"));
    }
    let k = psi.wavenumbers();
    let kinetic: Vec<Complex> =
        k.iter().map(|ki| cis(-hbar * ki * ki * dt / (2.0 * mass))).collect();

    for _ in 0..steps {
        // The nonlinear half step uses the current density, which is what
        // makes this a *step* rather than an exact factorisation.
        for (z, vi) in psi.psi.iter_mut().zip(v) {
            let local = vi + g * z.norm_sq();
            *z = *z * cis(-local * dt / (2.0 * hbar));
        }
        let mut spectrum = fft(&psi.psi);
        for (z, factor) in spectrum.iter_mut().zip(&kinetic) {
            *z = *z * *factor;
        }
        psi.psi = ifft(&spectrum);
        for (z, vi) in psi.psi.iter_mut().zip(v) {
            let local = vi + g * z.norm_sq();
            *z = *z * cis(-local * dt / (2.0 * hbar));
        }
    }
    Ok(())
}

/// The exact bright soliton of the one-dimensional Gross-Pitaevskii equation
/// with `g < 0`, moving at speed `velocity`.
///
/// `psi = sqrt(n0) sech((x - v t) / xi) exp(i(...))`. Its shape is preserved
/// exactly for all time, which is what "soliton" means and what distinguishes
/// it from an ordinary travelling wave.
///
/// # Panics
/// Panics unless the amplitude and healing length are positive.
#[must_use]
pub fn soliton_bright_exact(
    x: f64,
    t: f64,
    amplitude: f64,
    width: f64,
    velocity: f64,
    mass: f64,
    hbar: f64,
) -> Complex {
    assert!(amplitude > 0.0 && width > 0.0, "the soliton needs a positive amplitude and width");
    let envelope = amplitude / ((x - velocity * t) / width).cosh();
    // The phase carries the motion and the chemical potential.
    let mu = -hbar * hbar / (2.0 * mass * width * width);
    let phase = mass * velocity * x / hbar
        - (0.5 * mass * velocity * velocity + mu) * t / hbar;
    scale(cis(phase), envelope)
}

/// The revival time of a particle in a box: the period after which every
/// phase returns to its start.
///
/// The energies are `n^2` times a constant, so all the relative phases are
/// commensurate and the state reassembles exactly -- which is special to this
/// spectrum. At rational fractions of the revival time the state is a finite
/// superposition of displaced copies of itself, and plotting the density
/// against space and time produces the interference lattice known as a
/// quantum carpet.
///
/// # Panics
/// Panics unless the width, mass and `hbar` are positive.
#[must_use]
pub fn revival_time(length: f64, mass: f64, hbar: f64) -> f64 {
    assert!(length > 0.0 && mass > 0.0 && hbar > 0.0, "revival_time needs positive parameters");
    4.0 * mass * length * length / (std::f64::consts::PI * hbar)
}

/// The probability density of a box state at a sequence of times, one row per
/// time.
///
/// `coefficients` gives the amplitude of each eigenstate, indexed from the
/// ground state.
///
/// # Errors
/// Returns an error for an empty expansion or grid.
pub fn quantum_carpet(
    length: f64,
    coefficients: &[Complex],
    times: &[f64],
    points: usize,
    mass: f64,
    hbar: f64,
) -> Result<Vec<Vec<f64>>, GeomError> {
    if coefficients.is_empty() || points < 2 || !(length > 0.0) {
        return Err(GeomError::InvalidArgument("quantum_carpet: bad input"));
    }
    if !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("quantum_carpet: bad constants"));
    }
    let energy = |n: usize| {
        let k = (n + 1) as f64 * std::f64::consts::PI / length;
        hbar * hbar * k * k / (2.0 * mass)
    };
    Ok(times
        .iter()
        .map(|&t| {
            (0..points)
                .map(|p| {
                    let x = length * p as f64 / (points - 1) as f64;
                    let mut acc = Complex::new(0.0, 0.0);
                    for (n, c) in coefficients.iter().enumerate() {
                        let shape = (2.0 / length).sqrt()
                            * ((n + 1) as f64 * std::f64::consts::PI * x / length).sin();
                        acc = acc + scale(*c * cis(-energy(n) * t / hbar), shape);
                    }
                    acc.norm_sq()
                })
                .collect()
        })
        .collect())
}

/// The survival probability of a state under repeated projective measurement.
///
/// With `measurements` checks spread over a total time `t`, the survival
/// probability is `(1 - (t / measurements)^2 / tau^2)^measurements`, which
/// tends to one as the measurements are made more often. That is the quantum
/// Zeno effect, and it turns on the *quadratic* short-time behaviour of the
/// survival probability: an exponential decay law would give the same answer
/// however often it was interrupted.
///
/// # Errors
/// Returns an error for a non-positive Zeno time or no measurements.
pub fn zeno_survival(t: f64, tau: f64, measurements: usize) -> Result<f64, GeomError> {
    if !(tau > 0.0) || measurements == 0 {
        return Err(GeomError::InvalidArgument("zeno_survival: bad parameters"));
    }
    let interval = t / measurements as f64;
    let single = (1.0 - (interval / tau).powi(2)).max(0.0);
    Ok(single.powi(measurements as i32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::wavefunction::{
        harmonic_oscillator_energy, infinite_well_energy, Wavefunction1D,
    };

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// A harmonic potential on a grid, with `hbar = m = omega = 1`.
    fn oscillator_grid(n: usize, reach: f64) -> (Vec<f64>, f64, f64) {
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let v = (0..n).map(|k| 0.5 * (x0 + k as f64 * dx).powi(2)).collect();
        (v, dx, x0)
    }

    // -----------------------------------------------------------------
    // The stationary equation
    // -----------------------------------------------------------------

    #[test]
    fn finite_differences_recover_the_infinite_well_spectrum_and_converge_from_below() {
        // E_n = n^2 pi^2 hbar^2 / 2 m L^2 exactly. The discrete Laplacian
        // understates curvature, so every computed level must sit *below* the
        // true one -- a one-sided error, which is a sharper check than a
        // symmetric tolerance would be.
        let l = 1.0f64;
        for n in [400usize, 800, 1600] {
            // Interior points only: the walls are the boundary condition.
            let dx = l / (n + 1) as f64;
            let v = vec![0.0; n];
            let (energies, states) = tise_solve_fd(&v, dx, 1.0, 1.0, 5).unwrap();
            for level in 1..=5usize {
                let exact = infinite_well_energy(level, l, 1.0, 1.0);
                let got = energies[level - 1];
                assert!(got < exact, "level {level} came out at {got}, above the exact {exact}");
                assert!(
                    (got - exact).abs() / exact < 4.0 / (n as f64) * level as f64,
                    "level {level} at n = {n} is {got} against {exact}"
                );
            }
            // The states are normalised and have the right node count.
            for (level, state) in states.iter().enumerate() {
                let norm: f64 = state.iter().map(|c| c * c).sum::<f64>() * dx;
                assert!(close(norm, 1.0, 1e-9), "state {level} has norm {norm}");
                let nodes = (0..state.len() - 1).filter(|&k| state[k] * state[k + 1] < 0.0).count();
                assert_eq!(nodes, level, "state {level} should have {level} interior nodes");
            }
        }
        // The error really does shrink with the grid, quadratically.
        let coarse = {
            let dx = l / 401.0;
            tise_solve_fd(&vec![0.0; 400], dx, 1.0, 1.0, 1).unwrap().0[0]
        };
        let fine = {
            let dx = l / 801.0;
            tise_solve_fd(&vec![0.0; 800], dx, 1.0, 1.0, 1).unwrap().0[0]
        };
        let exact = infinite_well_energy(1, l, 1.0, 1.0);
        let ratio = (coarse - exact).abs() / (fine - exact).abs();
        assert!((3.5..4.6).contains(&ratio), "the convergence ratio is {ratio}, not near four");
    }

    #[test]
    fn finite_differences_recover_the_oscillator_ladder() {
        // Equally spaced levels at (n + 1/2) hbar omega, which is what makes
        // the oscillator the model of everything near a minimum.
        let (v, dx, _) = oscillator_grid(2001, 12.0);
        let (energies, states) = tise_solve_fd(&v, dx, 1.0, 1.0, 8).unwrap();
        for n in 0..8usize {
            let exact = harmonic_oscillator_energy(n, 1.0, 1.0);
            // The discretisation error is O(dx^2) and grows with the level,
            // since a higher state oscillates faster on the same grid.
            assert!(
                close(energies[n], exact, 1e-3),
                "level {n} is {} against {exact}",
                energies[n]
            );
        }
        // The spacings are equal to each other, not merely near the formula.
        for n in 1..7usize {
            let gap = energies[n + 1] - energies[n];
            assert!(close(gap, energies[1] - energies[0], 1e-3), "gap {n} is {gap}");
        }
        // Parity alternates: the ground state is even, the first odd.
        let middle = states[0].len() / 2;
        for (n, state) in states.iter().enumerate().take(6) {
            for offset in [40usize, 120, 300] {
                let left = state[middle - offset];
                let right = state[middle + offset];
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                assert!(
                    (left - sign * right).abs() < 1e-6,
                    "state {n} has the wrong parity at offset {offset}"
                );
            }
        }
    }

    #[test]
    fn numerov_agrees_with_finite_differences_and_with_the_closed_forms() {
        // A fourth-order method against a second-order one and against the
        // exact answer: if all three agree, none of them is being flattered
        // by the others' errors.
        let potential = |x: f64| 0.5 * x * x;
        let found =
            tise_solve_numerov(&potential, (-10.0, 10.0), 4001, (0.0, 12.0), 1.0, 1.0, 6).unwrap();
        assert_eq!(found.len(), 6);
        for (n, (energy, state)) in found.iter().enumerate() {
            let exact = harmonic_oscillator_energy(n, 1.0, 1.0);
            assert!(close(*energy, exact, 1e-6), "level {n} is {energy} against {exact}");
            let norm: f64 = state.iter().map(|c| c * c).sum::<f64>() * (20.0 / 4000.0);
            assert!(close(norm, 1.0, 1e-6), "state {n} has norm {norm}");
        }

        // The infinite well, where Numerov is exact up to the shooting
        // tolerance because the solution is a sine.
        let flat = |_: f64| 0.0f64;
        let found = tise_solve_numerov(&flat, (0.0, 1.0), 2001, (0.1, 200.0), 1.0, 1.0, 4).unwrap();
        for (n, (energy, _)) in found.iter().enumerate() {
            let exact = infinite_well_energy(n + 1, 1.0, 1.0, 1.0);
            assert!(
                (energy - exact).abs() / exact < 1e-8,
                "well level {} is {energy} against {exact}",
                n + 1
            );
        }
    }

    #[test]
    fn the_basis_expansion_bounds_the_energies_from_above_as_the_variational_principle_demands() {
        // Truncating a basis can only raise the computed energies, and adding
        // functions can only lower them. That monotonicity is the content of
        // the variational principle and nothing in the code enforces it.
        let n = 1201usize;
        let (v, dx, x0) = oscillator_grid(n, 10.0);
        let basis = Basis::HarmonicOscillator { mass: 1.0, omega: 0.7 };

        // The bound is against the discretised Hamiltonian, which is what a
        // truncated basis on this grid can bound. Comparing to the continuum
        // formula would be comparing two different operators.
        let (reference, _) = tise_solve_fd(&v, dx, 1.0, 1.0, 4).unwrap();
        let mut previous = [f64::INFINITY; 4];
        for size in [6usize, 10, 16, 24] {
            let (energies, coefficients) =
                tise_solve_matrix_basis(&v, dx, x0, basis, size, 1.0, 1.0).unwrap();
            assert_eq!(coefficients.rows, size);
            for level in 0..4usize {
                assert!(
                    energies[level] >= reference[level] - 1e-9,
                    "level {level} came out at {}, below the operator's own {}",
                    energies[level],
                    reference[level]
                );
                assert!(
                    energies[level] <= previous[level] + 1e-9,
                    "level {level} rose from {} to {} as the basis grew",
                    previous[level],
                    energies[level]
                );
                previous[level] = energies[level];
            }
        }
        // At a large enough basis it is accurate, not merely bounded.
        let (energies, _) =
            tise_solve_matrix_basis(&v, dx, x0, basis, 30, 1.0, 1.0).unwrap();
        for level in 0..4usize {
            assert!(
                close(energies[level], harmonic_oscillator_energy(level, 1.0, 1.0), 5e-3),
                "level {level} is {}",
                energies[level]
            );
        }

        // The box basis on the box problem is exact at any size, since the
        // basis functions are the eigenstates.
        let n = 801usize;
        let l = 1.0f64;
        let dx = l / (n - 1) as f64;
        let (energies, _) =
            tise_solve_matrix_basis(&vec![0.0; n], dx, 0.0, Basis::Box { length: l }, 5, 1.0, 1.0)
                .unwrap();
        for level in 1..=4usize {
            let exact = infinite_well_energy(level, l, 1.0, 1.0);
            assert!(
                (energies[level - 1] - exact).abs() / exact < 5e-3,
                "box level {level} is {} against {exact}",
                energies[level - 1]
            );
        }
    }

    // -----------------------------------------------------------------
    // Time evolution
    // -----------------------------------------------------------------

    #[test]
    fn the_split_operator_is_unitary_and_reproduces_free_spreading() {
        // Unitarity to rounding, whatever the step: the factors are
        // exponentials of Hermitian operators, so the norm cannot drift.
        let n = 1024usize;
        let dx = 40.0 / n as f64;
        let v = vec![0.0; n];
        for dt in [0.001f64, 0.01, 0.1, 1.0] {
            let mut psi = Wavefunction1D::gaussian_packet(0.0, 2.0, 1.0, dx, -20.0, n).unwrap();
            tdse_split_operator(&mut psi, &v, dt, 20, 1.0, 1.0).unwrap();
            assert!(
                close(psi.norm(), 1.0, 1e-12),
                "at dt = {dt} the norm became {}",
                psi.norm()
            );
        }

        // Against the exact free propagator, which is what the spectral
        // kinetic step is: for a free particle the splitting error vanishes
        // because there is nothing to split.
        let mut stepped = Wavefunction1D::gaussian_packet(0.0, 2.0, 1.0, dx, -20.0, n).unwrap();
        let exact = stepped.propagate_free(2.0, 1.0, 1.0).unwrap();
        tdse_split_operator(&mut stepped, &v, 0.02, 100, 1.0, 1.0).unwrap();
        for (a, b) in stepped.psi.iter().zip(&exact.psi) {
            assert!((a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10);
        }

        // In a harmonic well the energy is conserved, which the splitting
        // does not guarantee for free and is the real test of the method.
        let (harmonic, hdx, hx0) = oscillator_grid(n, 20.0);
        let mut psi = Wavefunction1D::gaussian_packet(2.0, 0.0, 1.0, hdx, hx0, n).unwrap();
        let initial = psi.energy(&harmonic, 1.0, 1.0).unwrap();
        tdse_split_operator(&mut psi, &harmonic, 0.002, 3000, 1.0, 1.0).unwrap();
        let final_energy = psi.energy(&harmonic, 1.0, 1.0).unwrap();
        assert!(
            close(final_energy, initial, 1e-6),
            "the energy drifted from {initial} to {final_energy}"
        );
        // Three thousand round trips through the FFT accumulate a random walk
        // of rounding error, so the norm holds to about 1e-12 rather than to
        // machine precision -- still a drift with no secular direction.
        assert!(close(psi.norm(), 1.0, 1e-10), "the norm became {}", psi.norm());
    }

    #[test]
    fn a_coherent_state_orbits_the_harmonic_well_without_changing_shape() {
        // A Gaussian of exactly the ground-state width, displaced, is a
        // coherent state: its centre follows the classical orbit and its
        // shape is rigid. Anything else spreads, so this is a sharp check on
        // the propagator rather than a qualitative one.
        let n = 1024usize;
        let reach = 20.0f64;
        let (v, dx, x0) = oscillator_grid(n, reach);
        let displacement = 3.0f64;
        let mut psi = Wavefunction1D::gaussian_packet(
            displacement,
            0.0,
            1.0 / 2.0f64.sqrt(),
            dx,
            x0,
            n,
        )
        .unwrap();
        let width0 = psi.variance_x().sqrt();

        let period = 2.0 * std::f64::consts::PI;
        let dt = period / 4000.0;
        for quarter in 1..=4usize {
            tdse_split_operator(&mut psi, &v, dt, 1000, 1.0, 1.0).unwrap();
            let expected = displacement * (quarter as f64 * std::f64::consts::PI / 2.0).cos();
            assert!(
                close(psi.expectation_x(), expected, 5e-3),
                "after a quarter {quarter} the centre is at {}, not {expected}",
                psi.expectation_x()
            );
            assert!(
                close(psi.variance_x().sqrt(), width0, 5e-4),
                "the width changed to {}",
                psi.variance_x().sqrt()
            );
        }
    }

    #[test]
    fn crank_nicolson_is_unitary_and_agrees_with_the_split_operator() {
        // Two entirely different discretisations of the same equation: one
        // spectral and periodic, one tridiagonal with walls. Where the packet
        // is far from the boundary they must give the same answer.
        let n = 1024usize;
        let dx = 60.0 / n as f64;
        let x0 = -30.0f64;
        let v: Vec<f64> = (0..n).map(|k| 0.5 * (x0 + k as f64 * dx).powi(2) * 0.05).collect();

        let start = Wavefunction1D::gaussian_packet(-4.0, 1.0, 1.5, dx, x0, n).unwrap();
        let mut cn = start.clone();
        tdse_crank_nicolson(&mut cn, &v, 0.002, 500, 1.0, 1.0).unwrap();
        assert!(close(cn.norm(), 1.0, 1e-10), "Crank-Nicolson lost norm: {}", cn.norm());

        let mut split = start.clone();
        tdse_split_operator(&mut split, &v, 0.002, 500, 1.0, 1.0).unwrap();
        let mut worst: f64 = 0.0;
        for (a, b) in cn.psi.iter().zip(&split.psi) {
            worst = worst.max((a.re - b.re).abs()).max((a.im - b.im).abs());
        }
        assert!(worst < 2e-3, "the two propagators differ by {worst}");

        // Unitarity holds at absurd step sizes too, which is the property
        // that distinguishes the Cayley transform from an explicit scheme.
        let mut brutal = start.clone();
        tdse_crank_nicolson(&mut brutal, &v, 5.0, 20, 1.0, 1.0).unwrap();
        assert!(
            close(brutal.norm(), 1.0, 1e-9),
            "at dt = 5 the norm became {}",
            brutal.norm()
        );
    }

    #[test]
    fn the_absorbing_layer_removes_outgoing_amplitude_without_reflecting_it() {
        // Without an absorber a packet that reaches the edge of a periodic
        // grid comes back around and interferes with itself. The absorber
        // must remove it -- and must not bounce it, which is what a sudden
        // one would do.
        let n = 1024usize;
        let dx = 60.0 / n as f64;
        let x0 = -30.0f64;
        let v = vec![0.0; n];
        let cap = absorbing_boundary_cap(n, 200, 4.0).unwrap();
        assert!(cap[0] > 0.0 && cap[n - 1] > 0.0);
        assert!(cap[n / 2] == 0.0, "the middle of the grid must be untouched");
        assert!(cap[..200].windows(2).all(|w| w[0] >= w[1]), "the profile must rise inward");

        let mut psi = Wavefunction1D::gaussian_packet(0.0, 4.0, 1.5, dx, x0, n).unwrap();
        // At k = 4 the packet moves four units per unit time and has some
        // eighteen to cover before it meets the layer, so it needs a good
        // deal longer than that to be swallowed.
        let dt = 0.002;
        for _ in 0..6000 {
            tdse_split_operator(&mut psi, &v, dt, 1, 1.0, 1.0).unwrap();
            apply_absorber(&mut psi, &cap, dt, 1.0).unwrap();
        }
        // Almost everything has left.
        assert!(psi.norm() < 0.05, "the absorber left a norm of {}", psi.norm());
        // And what little remains is not sitting in the interior as a
        // reflection would be.
        let interior: f64 =
            psi.probability_density()[300..700].iter().sum::<f64>() * dx;
        assert!(interior < 1e-4, "a reflection of weight {interior} came back");

        assert!(absorbing_boundary_cap(10, 0, 1.0).is_err());
        assert!(absorbing_boundary_cap(10, 5, 1.0).is_err());
        assert!(absorbing_boundary_cap(10, 2, -1.0).is_err());
        assert!(apply_absorber(&mut psi, &[0.0; 4], 0.1, 1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Scattering
    // -----------------------------------------------------------------

    #[test]
    fn the_transfer_matrix_reproduces_the_rectangular_barrier_in_closed_form() {
        // The transfer matrix is exact for a piecewise-constant potential, so
        // on a rectangular barrier it must match the textbook formula to
        // rounding whatever the resolution.
        let (v0, width) = (5.0f64, 1.0f64);
        for slices in [50usize, 200, 800] {
            let dx = width / slices as f64;
            let v = vec![v0; slices];
            for energy in [0.5f64, 1.0, 2.5, 4.9, 5.5, 8.0, 20.0] {
                let numeric = transmission_coefficient(&v, dx, energy, 1.0, 1.0).unwrap();
                let exact = tunneling_rectangular_exact(v0, width, energy, 1.0, 1.0).unwrap();
                assert!(
                    close(numeric, exact, 1e-9),
                    "at E = {energy} with {slices} slices: {numeric} against {exact}"
                );
                assert!((0.0..=1.0).contains(&numeric), "the probability is {numeric}");
            }
        }

        // Tunnelling falls exponentially with the barrier width.
        let mut previous = 1.0;
        for width in [0.5f64, 1.0, 1.5, 2.0, 3.0] {
            let t = tunneling_rectangular_exact(5.0, width, 1.0, 1.0, 1.0).unwrap();
            assert!(t < previous, "transmission rose with width at {width}");
            previous = t;
        }
        // The decay rate is 2 kappa, but doubling the width does *not* square
        // the transmission: for a thick barrier
        // T ~ 16 E (V0 - E) / V0^2 * exp(-2 kappa d), and squaring squares the
        // prefactor as well. The ratio T(2d) / T(d)^2 is therefore
        // V0^2 / 16 E (V0 - E), which for these numbers is exactly 25/64.
        let (v0, energy) = (5.0f64, 1.0f64);
        let thick = tunneling_rectangular_exact(v0, 4.0, energy, 1.0, 1.0).unwrap();
        let twice = tunneling_rectangular_exact(v0, 8.0, energy, 1.0, 1.0).unwrap();
        let prefactor = v0 * v0 / (16.0 * energy * (v0 - energy));
        assert!(
            (twice / (thick * thick) / prefactor - 1.0).abs() < 1e-3,
            "the exponential law fails: {} against {prefactor}",
            twice / (thick * thick)
        );
    }

    #[test]
    fn a_barrier_is_perfectly_transparent_at_its_resonances() {
        // Above the barrier the transmission returns to exactly one whenever
        // the barrier holds a whole number of half-wavelengths. Classically
        // the barrier is invisible at every energy above it; quantum
        // mechanically it is invisible only at these.
        let (v0, width) = (2.0f64, 3.0f64);
        for m in 1..=5usize {
            // k width = m pi, with k^2 = 2 m (E - V0) at hbar = m = 1.
            let k = m as f64 * std::f64::consts::PI / width;
            let energy = v0 + k * k / 2.0;
            let t = tunneling_rectangular_exact(v0, width, energy, 1.0, 1.0).unwrap();
            assert!(close(t, 1.0, 1e-9), "resonance {m} transmits {t}");

            // Halfway between resonances it is not transparent.
            let k_off = (m as f64 + 0.5) * std::f64::consts::PI / width;
            let off = tunneling_rectangular_exact(v0, width, v0 + k_off * k_off / 2.0, 1.0, 1.0)
                .unwrap();
            assert!(off < 0.999, "between resonances the transmission is {off}");
        }

        // A step reflects even when the particle has energy to spare.
        assert!(close(reflection_step_potential(0.0, 1.0).unwrap(), 0.0, 1e-12));
        assert!(close(reflection_step_potential(1.0, 0.5).unwrap(), 1.0, 1e-12));
        let over = reflection_step_potential(1.0, 2.0).unwrap();
        let expected = {
            let (k1, k2) = (2.0f64.sqrt(), 1.0f64);
            ((k1 - k2) / (k1 + k2)).powi(2)
        };
        assert!(close(over, expected, 1e-12), "the step reflects {over}, not {expected}");
        assert!(over > 0.0, "a classical particle would not reflect at all");
        assert!(reflection_step_potential(1.0, 0.0).is_err());
    }

    #[test]
    fn wkb_gets_the_exponent_of_a_thick_barrier_and_the_oscillator_spectrum_exactly() {
        // The tunnelling approximation is the exponential only, so it is
        // compared against the exact result's exponential rather than against
        // the exact result.
        let v0 = 5.0f64;
        let barrier = |x: f64| if (0.0..3.0).contains(&x) { v0 } else { 0.0 };
        let energy = 1.0f64;
        let approximate = wkb_tunneling(&barrier, energy, (0.0, 3.0), 1.0, 1.0, 20_000).unwrap();
        let kappa = (2.0 * (v0 - energy)).sqrt();
        assert!(
            close(approximate, (-2.0 * kappa * 3.0).exp(), 1e-9),
            "the WKB factor is {approximate}"
        );
        // The exact answer differs by a prefactor of order one, which is
        // exactly the accuracy claimed.
        let exact = tunneling_rectangular_exact(v0, 3.0, energy, 1.0, 1.0).unwrap();
        let ratio = exact / approximate;
        assert!(
            (1.0..40.0).contains(&ratio),
            "WKB should be right to a factor of order one, got {ratio}"
        );

        // Bohr-Sommerfeld on the harmonic oscillator is exact at every level,
        // Maslov correction included -- a coincidence of the quadratic well.
        let potential = |x: f64| 0.5 * x * x;
        for n in 0..6usize {
            let energy =
                wkb_quantization(&potential, n, (0.01, 20.0), (-15.0, 15.0), 1.0, 1.0, 20_000)
                    .unwrap();
            let exact = harmonic_oscillator_energy(n, 1.0, 1.0);
            assert!(
                (energy - exact).abs() / exact < 2e-4,
                "level {n} is {energy} against {exact}"
            );
        }
        assert!(wkb_quantization(&potential, 0, (10.0, 20.0), (-15.0, 15.0), 1.0, 1.0, 100).is_err());
        assert!(wkb_tunneling(&barrier, 1.0, (3.0, 0.0), 1.0, 1.0, 100).is_err());
    }

    #[test]
    fn a_wavepacket_scatters_at_roughly_the_rate_the_plane_wave_result_predicts() {
        // The packet carries a spread of momenta, so its transmission is the
        // plane-wave curve *averaged* over that spread -- not its value at
        // the mean momentum. Near a barrier top the curve is steep enough
        // that the two differ by several per cent, so comparing against the
        // value at the mean would either fail or need a tolerance loose
        // enough to hide a genuine error. The average is computed here from
        // the transfer matrix, which makes this a quantitative check of the
        // propagator against an independent method.
        let n = 2048usize;
        let dx = 120.0 / n as f64;
        let x0 = -60.0f64;
        let (v0, width) = (2.0f64, 1.0f64);
        let v: Vec<f64> = (0..n)
            .map(|k| {
                let x = x0 + k as f64 * dx;
                if x.abs() < width / 2.0 {
                    v0
                } else {
                    0.0
                }
            })
            .collect();
        // The grid is periodic, so the run has to stop before the
        // transmitted part wraps around and is counted as reflected. At
        // k = 2.2 it covers 55 units in the time allowed and the grid has 60
        // to the right of the barrier.
        let sigma = 3.0f64;
        for k0 in [1.6f64, 2.2] {
            let (transmitted, reflected) = wavepacket_scattering(
                &v, dx, x0, 0.0, k0, sigma, -20.0, 0.01, 2500, 1.0, 1.0,
            )
            .unwrap();
            assert!(
                close(transmitted + reflected, 1.0, 1e-9),
                "probability is not conserved: {transmitted} + {reflected}"
            );

            // The packet's momentum distribution is Gaussian with width
            // 1 / 2 sigma; average the plane-wave transmission over it.
            let spread = 1.0 / (2.0 * sigma);
            let steps = 4000usize;
            let (lo, hi) = (k0 - 6.0 * spread, k0 + 6.0 * spread);
            let h = (hi - lo) / steps as f64;
            let mut weight_total = 0.0;
            let mut weighted = 0.0;
            for j in 0..steps {
                let k = lo + (j as f64 + 0.5) * h;
                if k <= 0.0 {
                    continue;
                }
                let w = (-(k - k0) * (k - k0) / (2.0 * spread * spread)).exp();
                let t = tunneling_rectangular_exact(v0, width, k * k / 2.0, 1.0, 1.0).unwrap();
                weight_total += w;
                weighted += w * t;
            }
            let predicted = weighted / weight_total;
            assert!(
                (transmitted - predicted).abs() < 0.02,
                "at k = {k0} the packet transmitted {transmitted} against the averaged {predicted}"
            );

        }

        // That the averaging matters at all is worth establishing separately,
        // and it is pure arithmetic -- no propagation needed. Sitting exactly
        // on a resonance is the sharpest case: the plane wave transmits with
        // certainty, and any spread of momenta at all pulls the packet's
        // average below one, because the resonance is a maximum and every
        // neighbouring momentum does worse. Just above the barrier top, by
        // contrast, the curve is nearly straight and averaging changes
        // almost nothing -- so a test placed there would prove little.
        let k0 = (2.0f64 * (2.0 + std::f64::consts::PI * std::f64::consts::PI / 2.0)).sqrt();
        assert!(
            close(tunneling_rectangular_exact(v0, width, k0 * k0 / 2.0, 1.0, 1.0).unwrap(), 1.0, 1e-9),
            "the chosen momentum is not a resonance"
        );
        let spread = 1.0 / (2.0 * 0.5);
        let steps = 4000usize;
        let (lo, hi) = (k0 - 6.0 * spread, k0 + 6.0 * spread);
        let h = (hi - lo) / steps as f64;
        let (mut weight_total, mut weighted) = (0.0f64, 0.0f64);
        for j in 0..steps {
            let k = lo + (j as f64 + 0.5) * h;
            if k <= 0.0 {
                continue;
            }
            let w = (-(k - k0) * (k - k0) / (2.0 * spread * spread)).exp();
            weight_total += w;
            weighted += w * tunneling_rectangular_exact(v0, width, k * k / 2.0, 1.0, 1.0).unwrap();
        }
        let averaged = weighted / weight_total;
        let at_mean = tunneling_rectangular_exact(v0, width, k0 * k0 / 2.0, 1.0, 1.0).unwrap();
        assert!(
            at_mean - averaged > 0.05,
            "a spread of momenta must lose the resonance: {averaged} against {at_mean}"
        );
    }

    #[test]
    fn a_double_well_splits_its_ground_doublet_by_less_the_higher_the_barrier() {
        // The splitting is exponentially small in the barrier, so raising it
        // must shrink the splitting sharply -- and the two states must be the
        // symmetric and antisymmetric combinations, which is checked by
        // parity rather than assumed.
        let n = 2001usize;
        let reach = 6.0f64;
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let mut previous = f64::INFINITY;
        let mut splittings = Vec::new();
        for barrier in [2.0f64, 4.0, 8.0, 16.0] {
            let v: Vec<f64> = (0..n)
                .map(|k| {
                    let x = x0 + k as f64 * dx;
                    barrier * (x * x - 2.0) * (x * x - 2.0) / 4.0
                })
                .collect();
            let splitting = double_well_splitting(&v, dx, 1.0, 1.0).unwrap();
            assert!(splitting > 0.0, "the doublet did not split at all");
            assert!(
                splitting < previous,
                "raising the barrier to {barrier} widened the splitting to {splitting}"
            );
            previous = splitting;
            splittings.push(splitting);

            let (_, states) = tise_solve_fd(&v, dx, 1.0, 1.0, 2).unwrap();
            let middle = n / 2;
            for (level, state) in states.iter().enumerate() {
                let sign = if level % 2 == 0 { 1.0 } else { -1.0 };
                for offset in [200usize, 400, 600] {
                    assert!(
                        (state[middle - offset] - sign * state[middle + offset]).abs() < 1e-6,
                        "state {level} has the wrong parity"
                    );
                }
            }
        }
        // The fall is exponential, not merely monotone: each doubling of the
        // barrier costs more than a factor of three, and the tunnelling
        // integral grows as the square root of the barrier height, so the
        // ratio itself grows.
        for pair in splittings.windows(2) {
            assert!(
                pair[0] / pair[1] > 2.0,
                "doubling the barrier only reduced the splitting from {} to {}",
                pair[0],
                pair[1]
            );
        }
        // The ratio itself grows, which is what "exponential in the barrier"
        // means: the tunnelling integral scales as its square root, so each
        // doubling costs more than the last.
        assert!(
            splittings[2] / splittings[3] > splittings[0] / splittings[1],
            "the fall is not accelerating: {splittings:?}"
        );

        // Pushed far enough, the doublet becomes numerically degenerate, and
        // that is the case the eigenvector routine has to work for: inverse
        // iteration at two shifts a whisker apart converges to whichever
        // combination the starting vector happened to favour, so without
        // orthogonalisation against the state already found the second one
        // comes back a copy of the first.
        let v: Vec<f64> = (0..n)
            .map(|k| {
                let x = x0 + k as f64 * dx;
                400.0 * (x * x - 2.0) * (x * x - 2.0) / 4.0
            })
            .collect();
        let splitting = double_well_splitting(&v, dx, 1.0, 1.0).unwrap();
        assert!(
            splitting < 1e-6,
            "the doublet should be all but degenerate here, not split by {splitting}"
        );
        let (_, states) = tise_solve_fd(&v, dx, 1.0, 1.0, 2).unwrap();
        let overlap: f64 =
            states[0].iter().zip(&states[1]).map(|(a, b)| a * b).sum::<f64>() * dx;
        assert!(
            overlap.abs() < 1e-6,
            "the two nearly degenerate states came back parallel, overlapping by {overlap}"
        );
        // Parity is *not* recoverable here, and claiming it would be wrong:
        // once the splitting drops below the numerical resolution, every
        // combination of the two states is an eigenvector to within tolerance
        // and the algorithm is free to return an arbitrary rotation inside
        // the doublet. What does still hold is that both come back as genuine
        // eigenvectors spanning it, so that is what is checked.
        let (energies, _) = tise_solve_fd(&v, dx, 1.0, 1.0, 2).unwrap();
        let kinetic = 1.0 / (2.0 * dx * dx);
        let middle = n / 2;
        for (level, state) in states.iter().enumerate() {
            let mut residual: f64 = 0.0;
            for k in 0..n {
                let mut applied = (2.0 * kinetic + v[k]) * state[k];
                if k > 0 {
                    applied -= kinetic * state[k - 1];
                }
                if k + 1 < n {
                    applied -= kinetic * state[k + 1];
                }
                residual = residual.max((applied - energies[level] * state[k]).abs());
            }
            assert!(
                residual < 1e-6 * (1.0 + energies[level].abs()),
                "state {level} leaves a residual of {residual}"
            );
            // Nothing sits on top of the barrier.
            assert!(
                state[middle].abs() < 1e-6,
                "state {level} has amplitude {} at the barrier top",
                state[middle]
            );
        }
        assert!(
            splittings[0] / splittings[3] > 100.0,
            "an eightfold barrier should cost orders of magnitude: {splittings:?}"
        );
    }

    // -----------------------------------------------------------------
    // Perturbation theory and the variational method
    // -----------------------------------------------------------------

    #[test]
    fn perturbation_theory_matches_a_directly_solved_shift_for_a_small_perturbation() {
        // The test of a series is whether it converges to the thing it
        // expands. Solving the perturbed problem exactly and comparing is the
        // only honest check, and the second-order term must improve on the
        // first.
        let n = 2001usize;
        let reach = 10.0f64;
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let v: Vec<f64> = (0..n).map(|k| 0.5 * (x0 + k as f64 * dx).powi(2)).collect();
        let (energies, states) = tise_solve_fd(&v, dx, 1.0, 1.0, 12).unwrap();

        // A quartic perturbation, whose exact first-order shift for the
        // ground state is 3 lambda / 4.
        let lambda = 0.02f64;
        let perturbation: Vec<f64> =
            (0..n).map(|k| lambda * (x0 + k as f64 * dx).powi(4)).collect();
        let first = perturbation_theory_1st(&states, &perturbation, dx).unwrap();
        assert!(close(first[0], 0.75 * lambda, 1e-5), "the first-order shift is {}", first[0]);
        assert!(close(first[1], 3.75 * lambda, 1e-4), "the first excited shift is {}", first[1]);

        let second = perturbation_theory_2nd(&states, &energies, &perturbation, dx).unwrap();
        assert!(second[0] < 0.0, "the ground state must be pushed down at second order");

        let perturbed: Vec<f64> = v.iter().zip(&perturbation).map(|(a, b)| a + b).collect();
        let (exact, _) = tise_solve_fd(&perturbed, dx, 1.0, 1.0, 3).unwrap();
        let true_shift = exact[0] - energies[0];
        let one_term = (energies[0] + first[0] - exact[0]).abs();
        let two_terms = (energies[0] + first[0] + second[0] - exact[0]).abs();
        assert!(
            two_terms < one_term,
            "second order made it worse: {two_terms} against {one_term}"
        );
        assert!(
            two_terms < 0.02 * true_shift.abs(),
            "two terms leave an error of {two_terms} on a shift of {true_shift}"
        );

        // Degenerate levels are refused rather than divided by zero.
        assert!(
            perturbation_theory_2nd(&states, &vec![1.0; states.len()], &perturbation, dx).is_err()
        );
        assert!(perturbation_theory_1st(&[], &perturbation, dx).is_err());
        assert!(perturbation_theory_1st(&states, &[0.0; 3], dx).is_err());
    }

    #[test]
    fn the_stark_shift_vanishes_for_the_ground_state_and_grows_with_the_level() {
        assert!(close(stark_shift_perturbative(0.01, 1, 0).unwrap(), 0.0, 1e-15));
        // n = 2 splits into three: shifts of -3F, 0, +3F in atomic units.
        assert!(close(stark_shift_perturbative(0.01, 2, 1).unwrap(), 0.03, 1e-12));
        assert!(close(stark_shift_perturbative(0.01, 2, -1).unwrap(), -0.03, 1e-12));
        assert!(close(stark_shift_perturbative(0.01, 2, 0).unwrap(), 0.0, 1e-15));
        // The spread grows as n^2: 3 n (n - 1) F across the manifold.
        let n3 = stark_shift_perturbative(0.01, 3, 2).unwrap();
        let n2 = stark_shift_perturbative(0.01, 2, 1).unwrap();
        assert!(n3 / n2 > 2.9 && n3 / n2 < 3.1, "the ratio is {}", n3 / n2);
        assert!(stark_shift_perturbative(0.01, 0, 0).is_err());
        assert!(stark_shift_perturbative(0.01, 2, 5).is_err());
    }

    #[test]
    fn the_variational_energy_is_an_upper_bound_that_the_right_trial_state_saturates() {
        // A Gaussian trial state contains the oscillator's exact ground state,
        // so the minimum must be exactly hbar omega / 2. On a quartic well it
        // does not, and the answer must then be strictly above the true
        // ground energy -- never below, which is the theorem.
        let n = 1601usize;
        let reach = 8.0f64;
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let trial = |x: f64, p: &[f64]| (-p[0].abs() * x * x).exp();

        let harmonic: Vec<f64> = (0..n).map(|k| 0.5 * (x0 + k as f64 * dx).powi(2)).collect();
        let (energy, params) =
            variational_ground_state(&harmonic, dx, x0, &trial, &[0.9], 1.0, 1.0).unwrap();
        assert!(close(energy, 0.5, 1e-5), "the variational energy is {energy}");
        // The optimum is at exp(-x^2 / 2), so the parameter is one half.
        assert!(close(params[0].abs(), 0.5, 1e-3), "the parameter came out {}", params[0]);

        let quartic: Vec<f64> = (0..n).map(|k| 0.25 * (x0 + k as f64 * dx).powi(4)).collect();
        let (bound, _) =
            variational_ground_state(&quartic, dx, x0, &trial, &[0.7], 1.0, 1.0).unwrap();
        let (exact, _) = tise_solve_fd(&quartic, dx, 1.0, 1.0, 1).unwrap();
        assert!(
            bound >= exact[0] - 1e-6,
            "the variational bound {bound} fell below the true energy {}",
            exact[0]
        );
        assert!(bound < exact[0] + 0.02, "the Gaussian should be a decent trial state: {bound}");
        assert!(variational_ground_state(&harmonic, dx, x0, &trial, &[], 1.0, 1.0).is_err());
    }

    #[test]
    fn imaginary_time_finds_the_same_ground_state_the_eigensolver_does() {
        // Two unrelated routes to the same object: one diagonalises, the
        // other lets the excited components decay away.
        let n = 201usize;
        let reach = 6.0f64;
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let v: Vec<f64> = (0..n).map(|k| 0.5 * (x0 + k as f64 * dx).powi(2)).collect();

        let (energy, state) = imaginary_time_propagation(&v, dx, 1e-3, 20_000, 1.0, 1.0).unwrap();
        let (exact, states) = tise_solve_fd(&v, dx, 1.0, 1.0, 1).unwrap();
        assert!(close(energy, exact[0], 1e-6), "imaginary time gives {energy}, the solver {}", exact[0]);

        let overlap: f64 =
            state.iter().zip(&states[0]).map(|(a, b)| a * b).sum::<f64>() * dx;
        assert!(
            close(overlap.abs(), 1.0, 1e-5),
            "the two ground states overlap by {overlap}, not one"
        );
        // And it is normalised and nodeless, as a ground state must be.
        let norm: f64 = state.iter().map(|c| c * c).sum::<f64>() * dx;
        assert!(close(norm, 1.0, 1e-9));
        let interior = &state[5..n - 5];
        let nodes = (0..interior.len() - 1)
            .filter(|&k| interior[k] * interior[k + 1] < 0.0)
            .count();
        assert_eq!(nodes, 0, "the ground state should have no nodes");
        assert!(imaginary_time_propagation(&v, dx, -1.0, 10, 1.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Dynamical theorems
    // -----------------------------------------------------------------

    #[test]
    fn ehrenfest_holds_and_the_harmonic_case_is_the_one_that_is_exact() {
        // In a quadratic potential the average force equals the force at the
        // average, so the expectations follow the classical orbit exactly.
        // In a quartic one they do not, and the discrepancy is real physics
        // rather than numerical error.
        let n = 1024usize;
        let reach = 20.0f64;
        let (harmonic, dx, x0) = oscillator_grid(n, reach);
        let dt = 0.005f64;
        let mut psi = Wavefunction1D::gaussian_packet(2.0, 0.0, 1.0, dx, x0, n).unwrap();
        let mut snapshots = vec![psi.clone()];
        for _ in 0..40 {
            tdse_split_operator(&mut psi, &harmonic, dt, 1, 1.0, 1.0).unwrap();
            snapshots.push(psi.clone());
        }
        let worst = ehrenfest_check(&snapshots, &harmonic, dt, 1.0, 1.0).unwrap();
        // The residual is the central difference's own O(dt^2) error, not a
        // failure of the theorem, so the check is that it falls as dt^2 --
        // which an actual violation would not.
        assert!(worst < 1e-4, "Ehrenfest fails in a harmonic well by {worst}");
        let mut finer = Wavefunction1D::gaussian_packet(2.0, 0.0, 1.0, dx, x0, n).unwrap();
        let small = dt / 2.0;
        let mut fine_snapshots = vec![finer.clone()];
        for _ in 0..40 {
            tdse_split_operator(&mut finer, &harmonic, small, 1, 1.0, 1.0).unwrap();
            fine_snapshots.push(finer.clone());
        }
        let refined = ehrenfest_check(&fine_snapshots, &harmonic, small, 1.0, 1.0).unwrap();
        let ratio = worst / refined;
        assert!(
            (3.0..5.0).contains(&ratio),
            "the residual fell by {ratio}, not the fourfold of a second-order error"
        );

        // The centre follows the classical orbit, which is what the theorem
        // amounts to here.
        let elapsed = 40.0 * dt;
        assert!(
            close(psi.expectation_x(), 2.0 * elapsed.cos(), 1e-4),
            "the centre is at {}, not {}",
            psi.expectation_x(),
            2.0 * elapsed.cos()
        );

        // A quartic well: the theorem still holds -- it is exact for any
        // potential -- but the classical orbit no longer describes the centre.
        let quartic: Vec<f64> =
            (0..n).map(|k| 0.02 * (x0 + k as f64 * dx).powi(4)).collect();
        let mut psi = Wavefunction1D::gaussian_packet(2.0, 0.0, 1.0, dx, x0, n).unwrap();
        let mut snapshots = vec![psi.clone()];
        for _ in 0..40 {
            tdse_split_operator(&mut psi, &quartic, dt, 1, 1.0, 1.0).unwrap();
            snapshots.push(psi.clone());
        }
        let worst = ehrenfest_check(&snapshots, &quartic, dt, 1.0, 1.0).unwrap();
        assert!(worst < 1e-3, "Ehrenfest fails in a quartic well by {worst}");

        // But the average force differs from the force at the average, which
        // is exactly what makes the quartic case non-classical.
        let density = snapshots[20].probability_density();
        let weight: f64 = density.iter().sum();
        let mean_x: f64 =
            density.iter().enumerate().map(|(k, p)| p * (x0 + k as f64 * dx)).sum::<f64>() / weight;
        let mean_force: f64 = density
            .iter()
            .enumerate()
            .map(|(k, p)| p * -0.08 * (x0 + k as f64 * dx).powi(3))
            .sum::<f64>()
            / weight;
        let force_at_mean = -0.08 * mean_x.powi(3);
        assert!(
            (mean_force - force_at_mean).abs() > 1e-3,
            "the two forces agree to {}, so the quartic case is not being tested",
            (mean_force - force_at_mean).abs()
        );
        assert!(ehrenfest_check(&snapshots[..2], &quartic, dt, 1.0, 1.0).is_err());
    }

    #[test]
    fn a_bright_soliton_propagates_without_spreading_and_a_free_packet_does_not() {
        // The whole point of the nonlinearity, tested by comparison: the same
        // initial profile evolved with and without the interaction.
        let n = 2048usize;
        let dx = 60.0 / n as f64;
        let x0 = -30.0f64;
        let v = vec![0.0; n];
        // The exact soliton of the attractive equation: g = -1 and an
        // amplitude fixed by the width.
        let width = 1.5f64;
        let amplitude = 1.0 / width;
        let g = -1.0f64;
        let psi0: Vec<Complex> = (0..n)
            .map(|k| soliton_bright_exact(x0 + k as f64 * dx, 0.0, amplitude, width, 0.0, 1.0, 1.0))
            .collect();
        let mut soliton = Wavefunction1D::new(psi0.clone(), dx, x0).unwrap();
        let initial_width = soliton.variance_x().sqrt();
        gross_pitaevskii_1d(&mut soliton, &v, g, 0.0025, 2000, 1.0, 1.0).unwrap();
        let after = soliton.variance_x().sqrt();
        assert!(
            close(after, initial_width, 5e-3),
            "the soliton spread from {initial_width} to {after}"
        );

        // The same profile with no interaction spreads visibly.
        let mut free = Wavefunction1D::new(psi0, dx, x0).unwrap();
        gross_pitaevskii_1d(&mut free, &v, 0.0, 0.0025, 2000, 1.0, 1.0).unwrap();
        let spread = free.variance_x().sqrt();
        assert!(
            spread > initial_width * 1.5,
            "without the nonlinearity it should spread: {spread} against {initial_width}"
        );

        // The norm is conserved even though the equation is nonlinear, since
        // the nonlinear term is still a real potential.
        assert!(close(soliton.norm(), free.norm(), 1e-9));
        assert!(gross_pitaevskii_1d(&mut free, &[0.0; 3], g, 0.01, 1, 1.0, 1.0).is_err());
    }

    #[test]
    fn a_box_state_revives_exactly_at_the_revival_time() {
        // The n^2 spectrum makes every relative phase commensurate, so the
        // state reassembles perfectly. At half the revival time it reassembles
        // mirrored, which is the other half of the same fact.
        let l = 1.0f64;
        let revival = revival_time(l, 1.0, 1.0);
        let coefficients: Vec<Complex> = (0..8)
            .map(|n| Complex::new(1.0 / ((n + 1) as f64), 0.0))
            .collect();
        let carpet = quantum_carpet(
            l,
            &coefficients,
            &[0.0, revival / 2.0, revival, 0.137 * revival],
            201,
            1.0,
            1.0,
        )
        .unwrap();
        assert_eq!(carpet.len(), 4);

        for (a, b) in carpet[0].iter().zip(&carpet[2]) {
            assert!((a - b).abs() < 1e-9, "the state did not revive: {a} against {b}");
        }
        // The half-revival is the mirror image about the centre of the box.
        let points = carpet[0].len();
        for (k, value) in carpet[1].iter().enumerate() {
            let mirrored = carpet[0][points - 1 - k];
            assert!(
                (value - mirrored).abs() < 1e-9,
                "the half revival is not a mirror at point {k}"
            );
        }
        // At a generic time it is neither.
        let generic: f64 = carpet[3]
            .iter()
            .zip(&carpet[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(generic > 1e-3, "the state should differ at a generic time: {generic}");

        assert!(quantum_carpet(l, &[], &[0.0], 10, 1.0, 1.0).is_err());
        assert!(quantum_carpet(l, &coefficients, &[0.0], 1, 1.0, 1.0).is_err());
    }

    #[test]
    fn watching_a_state_stops_it_decaying() {
        // The Zeno effect turns entirely on the quadratic short-time
        // behaviour: with the same total time, more measurements means more
        // survival, and the limit is one.
        let (t, tau) = (0.5f64, 1.0f64);
        let mut previous = 0.0;
        for measurements in [1usize, 2, 5, 20, 100, 1000] {
            let survival = zeno_survival(t, tau, measurements).unwrap();
            assert!(
                survival > previous,
                "with {measurements} checks the survival fell to {survival}"
            );
            assert!((0.0..=1.0).contains(&survival));
            previous = survival;
        }
        assert!(previous > 0.99, "frequent measurement should nearly freeze it: {previous}");
        // A single check reproduces the quadratic law exactly.
        assert!(close(zeno_survival(0.3, 1.0, 1).unwrap(), 1.0 - 0.09, 1e-12));
        assert!(zeno_survival(1.0, 0.0, 5).is_err());
        assert!(zeno_survival(1.0, 1.0, 0).is_err());
    }

    #[test]
    fn the_solvers_refuse_degenerate_input() {
        assert!(tise_solve_fd(&[1.0, 2.0], 0.1, 1.0, 1.0, 1).is_err());
        assert!(tise_solve_fd(&[1.0; 5], 0.0, 1.0, 1.0, 1).is_err());
        assert!(tise_solve_fd(&[1.0; 5], 0.1, 0.0, 1.0, 1).is_err());
        assert!(tise_solve_fd(&[1.0; 5], 0.1, 1.0, 1.0, 0).is_err());
        assert!(tise_solve_fd(&[1.0; 5], 0.1, 1.0, 1.0, 9).is_err());
        assert!(tise_solve_numerov(&|_| 0.0, (1.0, 0.0), 100, (0.0, 1.0), 1.0, 1.0, 1).is_err());
        assert!(tise_solve_numerov(&|_| 0.0, (0.0, 1.0), 3, (0.0, 1.0), 1.0, 1.0, 1).is_err());
        assert!(tise_solve_numerov(&|_| 0.0, (0.0, 1.0), 100, (1.0, 0.0), 1.0, 1.0, 1).is_err());
        assert!(
            tise_solve_matrix_basis(&[1.0; 5], 0.1, 0.0, Basis::Box { length: 1.0 }, 0, 1.0, 1.0)
                .is_err()
        );
        assert!(transmission_coefficient(&[], 0.1, 1.0, 1.0, 1.0).is_err());
        assert!(transmission_coefficient(&[1.0], 0.1, 0.0, 1.0, 1.0).is_err());
        assert!(tunneling_rectangular_exact(1.0, 0.0, 1.0, 1.0, 1.0).is_err());
        // A zero barrier transmits everything, at any energy.
        assert!(close(tunneling_rectangular_exact(0.0, 2.0, 3.0, 1.0, 1.0).unwrap(), 1.0, 1e-15));
        // Exactly at the barrier top the two branches agree in the limit.
        let top = tunneling_rectangular_exact(2.0, 1.0, 2.0, 1.0, 1.0).unwrap();
        let just_below = tunneling_rectangular_exact(2.0, 1.0, 2.0 - 1e-7, 1.0, 1.0).unwrap();
        let just_above = tunneling_rectangular_exact(2.0, 1.0, 2.0 + 1e-7, 1.0, 1.0).unwrap();
        assert!(close(top, just_below, 1e-5) && close(top, just_above, 1e-5));

        let mut psi = Wavefunction1D::plane_wave(1.0, 0.1, 0.0, 32).unwrap();
        assert!(tdse_split_operator(&mut psi, &[0.0; 4], 0.1, 1, 1.0, 1.0).is_err());
        assert!(tdse_split_operator(&mut psi, &[0.0; 32], 0.1, 1, 0.0, 1.0).is_err());
        assert!(tdse_crank_nicolson(&mut psi, &[0.0; 4], 0.1, 1, 1.0, 1.0).is_err());
        assert!(tdse_crank_nicolson(&mut psi, &[0.0; 32], 0.1, 1, -1.0, 1.0).is_err());
        let mut odd = Wavefunction1D::plane_wave(1.0, 0.1, 0.0, 30).unwrap();
        assert!(tdse_split_operator(&mut odd, &[0.0; 30], 0.1, 1, 1.0, 1.0).is_err());
        // Crank-Nicolson has no such restriction.
        assert!(tdse_crank_nicolson(&mut odd, &[0.0; 30], 0.01, 1, 1.0, 1.0).is_ok());
        assert!(wavepacket_scattering(&[0.0; 30], 0.1, 0.0, 0.0, 1.0, 1.0, -1.0, 0.01, 1, 1.0, 1.0)
            .is_err());
    }

    #[test]
    #[should_panic(expected = "positive amplitude and width")]
    fn the_soliton_rejects_a_zero_width() {
        let _ = soliton_bright_exact(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "positive parameters")]
    fn the_revival_time_rejects_a_zero_box() {
        let _ = revival_time(0.0, 1.0, 1.0);
    }
}
