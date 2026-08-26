//! Spin operators, quantum magnets, and magnetic resonance.
//!
//! Two quite different things live here. The first is many-body: a chain of
//! coupled spins has a Hilbert space of dimension `2^n`, so exact
//! diagonalisation stops at a dozen or so sites and everything past that is a
//! matter of finding the small part of the space that matters. Lanczos does
//! that for the ground state, and the reason it works is that the extreme
//! eigenvalues of a large sparse matrix converge in a Krylov space of
//! dimension far smaller than the matrix.
//!
//! The second is single-spin dynamics -- Larmor precession, Rabi flopping,
//! echoes -- which is a two-level problem with closed-form answers and is
//! interesting for the opposite reason: the classical Bloch equations
//! describe it exactly, so it is where quantum mechanics is least mysterious.
//!
//! Spin-1/2 operators are `sigma / 2` throughout, and `hbar = 1` unless a
//! function takes it explicitly.

use crate::error::GeomError;
use crate::fractals::Complex;
use crate::linalg::matrix::Matrix;
use crate::linalg::tridiagonal::eigen_symmetric_tridiagonal;
use crate::monte_carlo::Rng;

const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
const ONE: Complex = Complex { re: 1.0, im: 0.0 };

fn scale(z: Complex, k: f64) -> Complex {
    Complex::new(z.re * k, z.im * k)
}

// ---------------------------------------------------------------------------
// Spin operators
// ---------------------------------------------------------------------------

/// The three Pauli matrices, in the order `X`, `Y`, `Z`.
#[must_use]
pub fn pauli_matrices() -> [Vec<Vec<Complex>>; 3] {
    [
        vec![vec![ZERO, ONE], vec![ONE, ZERO]],
        vec![
            vec![ZERO, Complex::new(0.0, -1.0)],
            vec![Complex::new(0.0, 1.0), ZERO],
        ],
        vec![vec![ONE, ZERO], vec![ZERO, Complex::new(-1.0, 0.0)]],
    ]
}

/// The spin operators `(Sx, Sy, Sz)` for any spin `s`, as
/// `(2s + 1)`-dimensional matrices.
///
/// Built from the ladder operators, whose matrix elements
/// `sqrt(s(s+1) - m(m+1))` are what make the representation finite: the
/// coefficient vanishes exactly at the top of the ladder, so raising the
/// highest state gives zero rather than escaping the space. That single fact
/// is why angular momentum is quantised.
///
/// # Errors
/// Returns an error unless `2s` is a non-negative integer no larger than 20.
pub fn spin_operators(
    s: f64,
) -> Result<(Vec<Vec<Complex>>, Vec<Vec<Complex>>, Vec<Vec<Complex>>), GeomError> {
    let twice = (2.0 * s).round();
    if twice < 0.0 || twice > 20.0 || (2.0 * s - twice).abs() > 1e-9 {
        return Err(GeomError::InvalidArgument("2s must be a small non-negative integer"));
    }
    let dim = twice as usize + 1;
    // Basis ordered from m = s down to m = -s.
    let m_of = |index: usize| s - index as f64;

    let mut sz = vec![vec![ZERO; dim]; dim];
    let mut plus = vec![vec![ZERO; dim]; dim];
    for i in 0..dim {
        sz[i][i] = Complex::new(m_of(i), 0.0);
        if i > 0 {
            // S+ raises m by one, taking basis index i to i - 1.
            let m = m_of(i);
            let element = (s * (s + 1.0) - m * (m + 1.0)).max(0.0).sqrt();
            plus[i - 1][i] = Complex::new(element, 0.0);
        }
    }
    let minus: Vec<Vec<Complex>> = (0..dim)
        .map(|i| (0..dim).map(|j| plus[j][i].conjugate()).collect())
        .collect();
    let sx: Vec<Vec<Complex>> = (0..dim)
        .map(|i| (0..dim).map(|j| scale(plus[i][j] + minus[i][j], 0.5)).collect())
        .collect();
    let sy: Vec<Vec<Complex>> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let difference = plus[i][j] - minus[i][j];
                    // Divide by 2i, which is multiplying by -i/2.
                    Complex::new(difference.im * 0.5, -difference.re * 0.5)
                })
                .collect()
        })
        .collect();
    Ok((sx, sy, sz))
}

/// A spin coherent state: the state pointing along `(theta, phi)`.
///
/// The closest a spin gets to a classical arrow. Its uncertainty is the
/// minimum the algebra allows, and it becomes classical as `s` grows -- the
/// relative uncertainty falls as `1 / sqrt(s)`, which is why a macroscopic
/// magnet has a definite direction and a single electron does not.
///
/// # Errors
/// Returns an error for an invalid spin.
pub fn spin_coherent_state(s: f64, theta: f64, phi: f64) -> Result<Vec<Complex>, GeomError> {
    let twice = (2.0 * s).round();
    if twice < 0.0 || twice > 20.0 || (2.0 * s - twice).abs() > 1e-9 {
        return Err(GeomError::InvalidArgument("2s must be a small non-negative integer"));
    }
    let dim = twice as usize + 1;
    let n = twice as usize;
    // Binomial coefficients, in logarithms to keep large s finite.
    let mut log_binomial = vec![0.0f64; dim];
    for k in 1..dim {
        log_binomial[k] = log_binomial[k - 1] + ((n - k + 1) as f64).ln() - (k as f64).ln();
    }
    let (c, sn) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    Ok((0..dim)
        .map(|k| {
            // k counts steps down from m = s.
            let magnitude = (0.5 * log_binomial[k]).exp()
                * c.powi((n - k) as i32)
                * sn.powi(k as i32);
            let angle = -phi * (s - k as f64);
            Complex::new(magnitude * angle.cos(), magnitude * angle.sin())
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Dense Hamiltonians for spin-1/2 chains
// ---------------------------------------------------------------------------

/// An XXZ spin-1/2 chain in a longitudinal field.
///
/// `H = sum_i [ j (Sx Sx + Sy Sy) + jz Sz Sz ] - h sum_i Sz`, with the spin
/// operators equal to half the Pauli matrices.
///
/// Setting `j == jz` gives the isotropic Heisenberg model; `j == 0` gives the
/// classical Ising chain; and `jz == 0` gives the XX model, which is free
/// fermions in disguise.
#[derive(Debug, Clone, Copy)]
pub struct SpinChain {
    /// Number of sites.
    pub n: usize,
    /// The transverse exchange coupling.
    pub j: f64,
    /// The longitudinal exchange coupling.
    pub jz: f64,
    /// The longitudinal field.
    pub h_field: f64,
    /// Whether the last site couples back to the first.
    pub periodic: bool,
}

impl SpinChain {
    /// A chain, checking the site count.
    ///
    /// # Errors
    /// Returns an error for fewer than two sites or more than sixteen.
    pub fn new(n: usize, j: f64, jz: f64, h_field: f64, periodic: bool) -> Result<Self, GeomError> {
        if !(2..=16).contains(&n) {
            return Err(GeomError::InvalidArgument("a chain needs 2 to 16 sites"));
        }
        Ok(Self { n, j, jz, h_field, periodic })
    }

    /// The bonds of the chain.
    fn bonds(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = (0..self.n - 1).map(|i| (i, i + 1)).collect();
        if self.periodic && self.n > 2 {
            out.push((self.n - 1, 0));
        } else if self.periodic {
            // Two sites with periodic boundaries carry the bond twice, which
            // is the honest reading of the ring and not a special case.
            out.push((1, 0));
        }
        out
    }

    /// Applies the Hamiltonian to a state vector.
    ///
    /// This is the primitive everything else uses. Nothing is stored: each
    /// term is applied on the fly, so the cost is `O(n 2^n)` in time and
    /// `O(2^n)` in memory rather than the `O(4^n)` a stored matrix would
    /// need. That difference is the whole reason a sixteen-site chain is
    /// reachable and a stored one is not.
    ///
    /// # Errors
    /// Returns an error if the vector has the wrong length.
    pub fn apply(&self, v: &[Complex]) -> Result<Vec<Complex>, GeomError> {
        let size = 1usize << self.n;
        if v.len() != size {
            return Err(GeomError::InvalidArgument("the vector has the wrong length"));
        }
        let mut out = vec![ZERO; size];
        for (index, amplitude) in v.iter().enumerate() {
            if amplitude.re == 0.0 && amplitude.im == 0.0 {
                continue;
            }
            // The field and the Ising term are diagonal.
            let mut diagonal = 0.0;
            for i in 0..self.n {
                let spin = if index >> i & 1 == 0 { 0.5 } else { -0.5 };
                diagonal -= self.h_field * spin;
            }
            for &(a, b) in &self.bonds() {
                let sa = if index >> a & 1 == 0 { 0.5 } else { -0.5 };
                let sb = if index >> b & 1 == 0 { 0.5 } else { -0.5 };
                diagonal += self.jz * sa * sb;
            }
            out[index] = out[index] + scale(*amplitude, diagonal);

            // The flip-flop term exchanges an up-down pair.
            for &(a, b) in &self.bonds() {
                let up_a = index >> a & 1 == 0;
                let up_b = index >> b & 1 == 0;
                if up_a != up_b {
                    let flipped = index ^ (1 << a) ^ (1 << b);
                    // (Sx Sx + Sy Sy) = (S+ S- + S- S+) / 2, which is 1/2 on
                    // an antialigned pair.
                    out[flipped] = out[flipped] + scale(*amplitude, self.j * 0.5);
                }
            }
        }
        Ok(out)
    }

    /// The Hamiltonian as a dense real symmetric matrix.
    ///
    /// The XXZ Hamiltonian in this basis has no imaginary part -- the `Sy Sy`
    /// term's factors of `i` cancel against each other -- so it is stored
    /// real, which halves the eigensolver's work.
    ///
    /// # Errors
    /// Returns an error above ten sites, where the matrix stops being worth
    /// forming.
    pub fn hamiltonian_dense(&self) -> Result<Matrix, GeomError> {
        if self.n > 10 {
            return Err(GeomError::InvalidArgument("a dense Hamiltonian stops at ten sites"));
        }
        let size = 1usize << self.n;
        let mut m = Matrix::zeros(size, size);
        for column in 0..size {
            let mut basis = vec![ZERO; size];
            basis[column] = ONE;
            let image = self.apply(&basis)?;
            for (row, z) in image.iter().enumerate() {
                if z.im.abs() > 1e-12 {
                    return Err(GeomError::Degenerate("the Hamiltonian is not real in this basis"));
                }
                m.set(row, column, z.re);
            }
        }
        Ok(m)
    }

    /// The full spectrum, ascending.
    ///
    /// # Errors
    /// Returns an error above ten sites or if the eigensolver fails.
    pub fn spectrum_small(&self) -> Result<Vec<f64>, GeomError> {
        let m = self.hamiltonian_dense()?;
        let decomposition = crate::linalg::eigen::eigen_symmetric(&m, 1e-12, 300)
            .map_err(|_| GeomError::Degenerate("the spin eigenproblem failed"))?;
        let mut values = decomposition.values.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(values)
    }

    /// The ground state by Lanczos, returning the energy and the vector.
    ///
    /// # Errors
    /// Returns an error if the iteration fails to build a Krylov space.
    pub fn ground_state_lanczos(
        &self,
        iterations: usize,
        rng: &mut Rng,
    ) -> Result<(f64, Vec<Complex>), GeomError> {
        let size = 1usize << self.n;
        let matvec = |v: &[Complex]| self.apply(v).unwrap_or_else(|_| vec![ZERO; v.len()]);
        let (values, vectors) = lanczos(&matvec, size, iterations, rng)?;
        Ok((values[0], vectors[0].clone()))
    }

    /// The total magnetisation per site, `<Sz> / n`.
    ///
    /// # Errors
    /// Returns an error if the state has the wrong length.
    pub fn magnetization(&self, state: &[Complex]) -> Result<f64, GeomError> {
        let size = 1usize << self.n;
        if state.len() != size {
            return Err(GeomError::InvalidArgument("the state has the wrong length"));
        }
        let weight: f64 = state.iter().map(|z| z.norm_sq()).sum();
        if weight <= 0.0 {
            return Ok(0.0);
        }
        let total: f64 = state
            .iter()
            .enumerate()
            .map(|(index, z)| {
                let m: f64 = (0..self.n)
                    .map(|i| if index >> i & 1 == 0 { 0.5 } else { -0.5 })
                    .sum();
                z.norm_sq() * m
            })
            .sum();
        Ok(total / weight / self.n as f64)
    }

    /// The spin-spin correlation `<Sz_i Sz_j>`.
    ///
    /// # Errors
    /// Returns an error for a bad site index or state length.
    pub fn correlation(&self, state: &[Complex], i: usize, j: usize) -> Result<f64, GeomError> {
        let size = 1usize << self.n;
        if state.len() != size {
            return Err(GeomError::InvalidArgument("the state has the wrong length"));
        }
        if i >= self.n || j >= self.n {
            return Err(GeomError::InvalidArgument("the site index is out of range"));
        }
        let weight: f64 = state.iter().map(|z| z.norm_sq()).sum();
        if weight <= 0.0 {
            return Ok(0.0);
        }
        let total: f64 = state
            .iter()
            .enumerate()
            .map(|(index, z)| {
                let a = if index >> i & 1 == 0 { 0.5 } else { -0.5 };
                let b = if index >> j & 1 == 0 { 0.5 } else { -0.5 };
                z.norm_sq() * a * b
            })
            .sum();
        Ok(total / weight)
    }

    /// The static structure factor at wavevector `k`.
    ///
    /// The Fourier transform of the correlations, and what a neutron
    /// scattering experiment measures. A peak at `k = pi` is
    /// antiferromagnetic order; a peak at zero is ferromagnetic.
    ///
    /// # Errors
    /// Returns an error if the state has the wrong length.
    pub fn structure_factor(&self, state: &[Complex], k: f64) -> Result<f64, GeomError> {
        let mut total = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                let phase = k * (i as f64 - j as f64);
                total += self.correlation(state, i, j)? * phase.cos();
            }
        }
        Ok(total / self.n as f64)
    }

    /// The entanglement entropy of the first `cut` sites, in bits.
    ///
    /// # Errors
    /// Returns an error for a bad cut or state length.
    pub fn entanglement_entropy_cut(&self, state: &[Complex], cut: usize) -> Result<f64, GeomError> {
        if cut == 0 || cut >= self.n {
            return Err(GeomError::InvalidArgument("the cut must split the chain"));
        }
        let size = 1usize << self.n;
        if state.len() != size {
            return Err(GeomError::InvalidArgument("the state has the wrong length"));
        }
        let left = 1usize << cut;
        let right = 1usize << (self.n - cut);
        // The reduced density matrix of the left block.
        let mut rho = vec![vec![ZERO; left]; left];
        for r in 0..right {
            for a in 0..left {
                for b in 0..left {
                    let ia = a | (r << cut);
                    let ib = b | (r << cut);
                    rho[a][b] = rho[a][b] + state[ia] * state[ib].conjugate();
                }
            }
        }
        let norm: f64 = (0..left).map(|a| rho[a][a].re).sum();
        if norm <= 0.0 {
            return Ok(0.0);
        }
        for row in &mut rho {
            for z in row.iter_mut() {
                *z = scale(*z, 1.0 / norm);
            }
        }
        let values = hermitian_eigenvalues(&rho)?;
        Ok(values.iter().filter(|v| **v > 1e-12).map(|v| -v * v.log2()).sum())
    }

    /// Evolves a state under the chain's Hamiltonian for a time `t`, by
    /// repeated Krylov steps.
    ///
    /// Each step builds a small Krylov space and exponentiates the
    /// tridiagonal projection exactly, which is why the method is stable at
    /// step sizes that would defeat a Taylor series -- the projection is
    /// Hermitian, so its exponential is unitary whatever the step.
    ///
    /// # Errors
    /// Returns an error for a bad state, a non-positive step, or a Krylov
    /// breakdown.
    pub fn time_evolve_krylov(
        &self,
        state: &[Complex],
        t: f64,
        steps: usize,
    ) -> Result<Vec<Complex>, GeomError> {
        let size = 1usize << self.n;
        if state.len() != size {
            return Err(GeomError::InvalidArgument("the state has the wrong length"));
        }
        if steps == 0 {
            return Err(GeomError::InvalidArgument("the step count must be positive"));
        }
        let dt = t / steps as f64;
        let mut current = state.to_vec();
        let depth = 12usize.min(size);
        for _ in 0..steps {
            current = krylov_step(&|v| self.apply(v).unwrap_or_else(|_| vec![ZERO; v.len()]),
                                  &current, dt, depth)?;
        }
        Ok(current)
    }
}

/// The eigenvalues of a small Hermitian matrix, via the real symmetric
/// embedding, which doubles each eigenvalue.
fn hermitian_eigenvalues(m: &[Vec<Complex>]) -> Result<Vec<f64>, GeomError> {
    let n = m.len();
    if n == 0 {
        return Err(GeomError::InvalidArgument("the matrix is empty"));
    }
    let mut embedded = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            embedded.set(i, j, m[i][j].re);
            embedded.set(i + n, j + n, m[i][j].re);
            embedded.set(i, j + n, -m[i][j].im);
            embedded.set(i + n, j, m[i][j].im);
        }
    }
    let decomposition = crate::linalg::eigen::eigen_symmetric(&embedded, 1e-13, 300)
        .map_err(|_| GeomError::Degenerate("the Hermitian eigenproblem failed"))?;
    Ok(decomposition.values.iter().step_by(2).copied().collect())
}

fn inner(a: &[Complex], b: &[Complex]) -> Complex {
    a.iter().zip(b).fold(ZERO, |acc, (x, y)| acc + x.conjugate() * *y)
}

fn norm_of(v: &[Complex]) -> f64 {
    v.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt()
}

/// The Lanczos algorithm: the extreme eigenvalues of a large Hermitian
/// operator given only its action on a vector.
///
/// Builds an orthonormal basis of the Krylov space and projects the operator
/// onto it, giving a tridiagonal matrix whose extreme eigenvalues converge
/// quickly to the operator's. The reorthogonalisation is not optional in
/// floating point: the Lanczos vectors lose orthogonality as soon as an
/// eigenvalue converges, and without it the algorithm reports spurious
/// duplicate eigenvalues -- a failure that looks like physics, since
/// degeneracies are physically meaningful.
///
/// Returns the Ritz values ascending and the matching Ritz vectors.
///
/// # Errors
/// Returns an error for a bad dimension or iteration count, or if the Krylov
/// space collapses immediately.
pub fn lanczos(
    matvec: &dyn Fn(&[Complex]) -> Vec<Complex>,
    dim: usize,
    iterations: usize,
    rng: &mut Rng,
) -> Result<(Vec<f64>, Vec<Vec<Complex>>), GeomError> {
    if dim == 0 || iterations == 0 {
        return Err(GeomError::InvalidArgument("lanczos: bad dimensions"));
    }
    let m = iterations.min(dim);
    let mut basis: Vec<Vec<Complex>> = Vec::with_capacity(m);
    let mut start: Vec<Complex> = (0..dim)
        .map(|_| Complex::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
        .collect();
    let magnitude = norm_of(&start);
    if magnitude <= 0.0 {
        return Err(GeomError::Degenerate("the starting vector vanished"));
    }
    for z in &mut start {
        *z = scale(*z, 1.0 / magnitude);
    }
    basis.push(start);

    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);
    for k in 0..m {
        let mut w = matvec(&basis[k]);
        let a = inner(&basis[k], &w).re;
        alpha.push(a);
        // Full reorthogonalisation against everything built so far.
        for previous in &basis {
            let projection = inner(previous, &w);
            for (target, source) in w.iter_mut().zip(previous) {
                *target = *target - projection * *source;
            }
        }
        let b = norm_of(&w);
        if b < 1e-12 || k + 1 == m {
            break;
        }
        beta.push(b);
        for z in &mut w {
            *z = scale(*z, 1.0 / b);
        }
        basis.push(w);
    }

    let size = alpha.len();
    let off = beta[..size.saturating_sub(1)].to_vec();
    let (values, vectors) = eigen_symmetric_tridiagonal(&alpha, &off)
        .map_err(|_| GeomError::Degenerate("the tridiagonal projection failed"))?;

    // Ritz vectors: combinations of the Lanczos basis.
    // `eigen_symmetric_tridiagonal` returns each eigenvector as a *row*, so
    // component k of eigenvector i is `vectors[i][k]`. Reading it the other
    // way round costs nothing in the eigenvalues -- they come back correct --
    // and silently produces Ritz vectors that are not eigenvectors at all.
    let ritz: Vec<Vec<Complex>> = (0..size)
        .map(|i| {
            let mut v = vec![ZERO; dim];
            for (k, b) in basis.iter().enumerate().take(size) {
                let weight = vectors[i][k];
                for (target, source) in v.iter_mut().zip(b) {
                    *target = *target + scale(*source, weight);
                }
            }
            let magnitude = norm_of(&v);
            if magnitude > 0.0 {
                for z in &mut v {
                    *z = scale(*z, 1.0 / magnitude);
                }
            }
            v
        })
        .collect();
    Ok((values, ritz))
}

/// One Krylov time step: `exp(-i H dt)` applied to a vector, via the
/// tridiagonal projection.
fn krylov_step(
    matvec: &dyn Fn(&[Complex]) -> Vec<Complex>,
    v: &[Complex],
    dt: f64,
    depth: usize,
) -> Result<Vec<Complex>, GeomError> {
    let magnitude = norm_of(v);
    if magnitude <= 0.0 {
        return Ok(v.to_vec());
    }
    let mut basis: Vec<Vec<Complex>> = vec![v.iter().map(|z| scale(*z, 1.0 / magnitude)).collect()];
    let mut alpha: Vec<f64> = Vec::new();
    let mut beta: Vec<f64> = Vec::new();
    for k in 0..depth {
        let mut w = matvec(&basis[k]);
        alpha.push(inner(&basis[k], &w).re);
        for previous in &basis {
            let projection = inner(previous, &w);
            for (target, source) in w.iter_mut().zip(previous) {
                *target = *target - projection * *source;
            }
        }
        let b = norm_of(&w);
        if b < 1e-12 || k + 1 == depth {
            break;
        }
        beta.push(b);
        for z in &mut w {
            *z = scale(*z, 1.0 / b);
        }
        basis.push(w);
    }
    let size = alpha.len();
    let off = beta[..size.saturating_sub(1)].to_vec();
    let (values, vectors) = eigen_symmetric_tridiagonal(&alpha, &off)
        .map_err(|_| GeomError::Degenerate("the Krylov projection failed"))?;

    // exp(-i T dt) applied to the first basis vector, in the Ritz basis.
    let mut coefficients = vec![ZERO; size];
    for i in 0..size {
        let overlap = vectors[i][0];
        let phase = Complex::new((-values[i] * dt).cos(), (-values[i] * dt).sin());
        for (k, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient = *coefficient + scale(phase, overlap * vectors[i][k]);
        }
    }
    let mut out = vec![ZERO; v.len()];
    for (k, b) in basis.iter().enumerate().take(size) {
        for (target, source) in out.iter_mut().zip(b) {
            *target = *target + coefficients[k] * *source;
        }
    }
    for z in &mut out {
        *z = scale(*z, magnitude);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Exactly solvable cases
// ---------------------------------------------------------------------------

/// The spectrum of two Heisenberg-coupled spin-1/2 particles: a singlet and a
/// triplet.
///
/// `S1 . S2 = (S^2 - S1^2 - S2^2) / 2`, so the energy depends only on the
/// total spin: `-3/4` for the singlet and `+1/4` for the threefold triplet,
/// times the coupling. The whole of chemical bonding in a two-electron
/// molecule is this splitting.
#[must_use]
pub fn heisenberg_2site_exact(j: f64) -> Vec<f64> {
    vec![-0.75 * j, 0.25 * j, 0.25 * j, 0.25 * j]
}

/// The transverse-field Ising chain as a dense matrix.
///
/// `H = -sum_i sigma^z_i sigma^z_{i+1} - g sum_i sigma^x_i`, in Pauli
/// matrices rather than spin operators, which is the convention the exact
/// solution below uses.
///
/// # Errors
/// Returns an error outside two to ten sites.
pub fn ising_transverse_field_dense(n: usize, g: f64, periodic: bool) -> Result<Matrix, GeomError> {
    if !(2..=10).contains(&n) {
        return Err(GeomError::InvalidArgument("the chain must have 2 to 10 sites"));
    }
    let size = 1usize << n;
    let mut m = Matrix::zeros(size, size);
    let mut bonds: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    if periodic {
        bonds.push((n - 1, 0));
    }
    for index in 0..size {
        // The Ising term is diagonal in the z basis.
        let mut diagonal = 0.0;
        for &(a, b) in &bonds {
            let sa = if index >> a & 1 == 0 { 1.0 } else { -1.0 };
            let sb = if index >> b & 1 == 0 { 1.0 } else { -1.0 };
            diagonal -= sa * sb;
        }
        m.set(index, index, diagonal);
        // The field flips one spin at a time.
        for i in 0..n {
            let flipped = index ^ (1 << i);
            m.set(flipped, index, m.get(flipped, index) - g);
        }
    }
    Ok(m)
}

/// Applies the transverse-field Ising Hamiltonian to a state vector.
///
/// Matrix free, so the cost is `O(n 2^n)` rather than the `O(4^n)` of forming
/// the matrix -- which at ten sites is the difference between a megabyte and
/// a gigabyte, and between a Jacobi diagonalisation that finishes and one
/// that does not.
///
/// # Errors
/// Returns an error for a bad site count or vector length.
pub fn ising_transverse_field_apply(
    n: usize,
    g: f64,
    periodic: bool,
    v: &[Complex],
) -> Result<Vec<Complex>, GeomError> {
    if !(2..=20).contains(&n) {
        return Err(GeomError::InvalidArgument("the chain must have 2 to 20 sites"));
    }
    let size = 1usize << n;
    if v.len() != size {
        return Err(GeomError::InvalidArgument("the vector has the wrong length"));
    }
    let mut bonds: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    if periodic {
        bonds.push((n - 1, 0));
    }
    let mut out = vec![ZERO; size];
    for (index, amplitude) in v.iter().enumerate() {
        if amplitude.re == 0.0 && amplitude.im == 0.0 {
            continue;
        }
        let mut diagonal = 0.0;
        for &(a, b) in &bonds {
            let sa = if index >> a & 1 == 0 { 1.0 } else { -1.0 };
            let sb = if index >> b & 1 == 0 { 1.0 } else { -1.0 };
            diagonal -= sa * sb;
        }
        out[index] = out[index] + scale(*amplitude, diagonal);
        for i in 0..n {
            let flipped = index ^ (1 << i);
            out[flipped] = out[flipped] - scale(*amplitude, g);
        }
    }
    Ok(out)
}

/// The exact ground energy of the periodic transverse-field Ising chain, from
/// the Jordan-Wigner solution.
///
/// The chain maps to free fermions, so the ground energy is a sum of
/// single-particle energies: `-sum_k sqrt(1 + g^2 - 2 g cos k)` over the
/// antiperiodic momenta `(2m + 1) pi / n`. That the interacting spin model
/// is secretly free is what makes it the standard testbed for quantum phase
/// transitions -- the critical point at `g = 1` is exactly known.
///
/// # Errors
/// Returns an error for fewer than two sites.
pub fn ising_transverse_field_exact(n: usize, g: f64) -> Result<f64, GeomError> {
    if n < 2 {
        return Err(GeomError::InvalidArgument("the chain must have at least two sites"));
    }
    let total: f64 = (0..n)
        .map(|m| {
            let k = (2 * m + 1) as f64 * std::f64::consts::PI / n as f64;
            (1.0 + g * g - 2.0 * g * k.cos()).max(0.0).sqrt()
        })
        .sum();
    Ok(-total)
}

/// The critical transverse field of the Ising chain, where the gap closes.
#[must_use]
pub fn itf_critical_point() -> f64 {
    1.0
}

/// The magnon dispersion of a ferromagnetic Heisenberg chain.
///
/// `2 j s (1 - cos(k a))`, which vanishes as `k^2` at long wavelength. The
/// quadratic -- rather than linear -- dispersion is the signature of a
/// ferromagnet's broken symmetry, and it is why a ferromagnet's low-
/// temperature heat capacity goes as `T^(3/2)` while an antiferromagnet's
/// goes as `T^3`.
#[must_use]
pub fn magnon_dispersion(j: f64, k: f64, s: f64, a: f64) -> f64 {
    2.0 * j * s * (1.0 - (k * a).cos())
}

// ---------------------------------------------------------------------------
// Single-spin dynamics
// ---------------------------------------------------------------------------

/// The Larmor precession angle after a time `t` in a field of magnitude `b`.
///
/// The precession rate depends on the field and the gyromagnetic ratio and
/// not at all on the angle, which is why a spin precesses at a fixed
/// frequency however it is tipped.
#[must_use]
pub fn larmor_frequency(b: f64, gamma: f64) -> f64 {
    gamma * b
}

/// The magnetisation vector after Larmor precession about the `z` axis.
///
/// The sense is the one the Bloch equation `dM/dt = gamma M x B` gives: for a
/// positive gyromagnetic ratio and a field along `+z`, the vector turns
/// *clockwise* seen from `+z`, so the angular velocity is `-gamma B`. Half
/// the sign conventions in the literature differ, and the two disagree on
/// everything that depends on the direction of a rotation.
#[must_use]
pub fn larmor_precession(m0: (f64, f64, f64), b: f64, gamma: f64, t: f64) -> (f64, f64, f64) {
    let angle = -larmor_frequency(b, gamma) * t;
    (
        m0.0 * angle.cos() - m0.1 * angle.sin(),
        m0.0 * angle.sin() + m0.1 * angle.cos(),
        m0.2,
    )
}

/// The excited-state probability of a driven two-level system: Rabi's
/// formula.
///
/// `(omega^2 / Omega^2) sin^2(Omega t / 2)` with the generalised frequency
/// `Omega = sqrt(omega^2 + delta^2)`. Off resonance the oscillation is faster
/// and shallower, and the peak probability falls as the detuning grows --
/// which is why a driven transition is a filter as well as a rotation.
///
/// # Errors
/// Returns an error if the drive and detuning are both zero.
pub fn rabi_oscillation(rabi: f64, detuning: f64, t: f64) -> Result<f64, GeomError> {
    let generalised = (rabi * rabi + detuning * detuning).sqrt();
    if generalised <= 0.0 {
        return Err(GeomError::InvalidArgument("the drive and detuning cannot both vanish"));
    }
    Ok((rabi * rabi / (generalised * generalised)) * (generalised * t / 2.0).sin().powi(2))
}

/// Ramsey fringes: the signal after two pulses separated by a free evolution.
///
/// The fringe spacing measures the detuning, and the envelope's decay
/// measures `T2*` -- the *inhomogeneous* dephasing time, which includes
/// static field variations that a spin echo can undo. That distinction is the
/// point of the technique.
#[must_use]
pub fn ramsey_fringes(detuning: f64, free_time: f64, t2_star: f64) -> f64 {
    let envelope = if t2_star > 0.0 { (-free_time / t2_star).exp() } else { 1.0 };
    0.5 * (1.0 + envelope * (detuning * free_time).cos())
}

/// The spin echo amplitude at time `t` after a refocusing pulse at `t / 2`.
///
/// The echo removes static dephasing -- every spin that ran fast now runs
/// slow for an equal time -- so what survives decays at the true `T2` rather
/// than the much shorter `T2*`. The difference between them is entirely
/// reversible dephasing, which is why the echo can recover a signal that
/// looked lost.
#[must_use]
pub fn spin_echo_sim(t: f64, t2: f64, _t2_star: f64) -> f64 {
    if t2 <= 0.0 {
        return 0.0;
    }
    (-t / t2).exp()
}

/// Integrates the Bloch equations for a magnetisation in a time-dependent
/// field.
///
/// `dM/dt = gamma M x B - (Mx, My) / T2 - (Mz - M0) / T1`. The two relaxation
/// times are independent parameters and `T2 <= 2 T1` always, since the
/// transverse components cannot survive the longitudinal decay.
///
/// # Errors
/// Returns an error for non-positive times or steps.
pub fn bloch_equations(
    m0: (f64, f64, f64),
    field: &dyn Fn(f64) -> (f64, f64, f64),
    gamma: f64,
    t1: f64,
    t2: f64,
    equilibrium: f64,
    t_end: f64,
    dt: f64,
) -> Result<Vec<(f64, f64, f64)>, GeomError> {
    if !(t1 > 0.0) || !(t2 > 0.0) || !(dt > 0.0) || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("bloch_equations requires positive times"));
    }
    let derivative = |t: f64, m: (f64, f64, f64)| -> (f64, f64, f64) {
        let b = field(t);
        let cross = (
            m.1 * b.2 - m.2 * b.1,
            m.2 * b.0 - m.0 * b.2,
            m.0 * b.1 - m.1 * b.0,
        );
        (
            gamma * cross.0 - m.0 / t2,
            gamma * cross.1 - m.1 / t2,
            gamma * cross.2 - (m.2 - equilibrium) / t1,
        )
    };
    let steps = (t_end / dt).ceil() as usize;
    let mut m = m0;
    let mut out = vec![m];
    let mut t = 0.0;
    for _ in 0..steps {
        // Fourth-order Runge-Kutta: precession is a rotation, and a
        // first-order method turns it into a spiral.
        let k1 = derivative(t, m);
        let k2 = derivative(t + dt / 2.0, add(m, k1, dt / 2.0));
        let k3 = derivative(t + dt / 2.0, add(m, k2, dt / 2.0));
        let k4 = derivative(t + dt, add(m, k3, dt));
        m = (
            m.0 + dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
            m.1 + dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
            m.2 + dt / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2),
        );
        t += dt;
        out.push(m);
    }
    Ok(out)
}

fn add(m: (f64, f64, f64), k: (f64, f64, f64), h: f64) -> (f64, f64, f64) {
    (m.0 + h * k.0, m.1 + h * k.1, m.2 + h * k.2)
}

/// A free induction decay: the sum of decaying sinusoids one per chemical
/// environment, sampled at `rate`.
///
/// The Fourier transform of this is the spectrum, which is how nuclear
/// magnetic resonance actually works: the signal is measured in time and the
/// chemistry is read in frequency.
///
/// # Errors
/// Returns an error for mismatched lists or a non-positive rate.
pub fn nmr_fid(
    frequencies: &[f64],
    decay_times: &[f64],
    samples: usize,
    rate: f64,
) -> Result<Vec<f64>, GeomError> {
    if frequencies.is_empty() || frequencies.len() != decay_times.len() {
        return Err(GeomError::InvalidArgument("nmr_fid: mismatched input"));
    }
    if !(rate > 0.0) || samples == 0 {
        return Err(GeomError::InvalidArgument("nmr_fid: bad sampling"));
    }
    if decay_times.iter().any(|t| !(*t > 0.0)) {
        return Err(GeomError::InvalidArgument("the decay times must be positive"));
    }
    Ok((0..samples)
        .map(|k| {
            let t = k as f64 / rate;
            frequencies
                .iter()
                .zip(decay_times)
                .map(|(f, t2)| (-t / t2).exp() * (2.0 * std::f64::consts::PI * f * t).cos())
                .sum()
        })
        .collect())
}

/// The Zeeman energy shift of a level in a magnetic field.
///
/// # Panics
/// Never; the arithmetic is a product.
#[must_use]
pub fn zeeman_splitting(b: f64, g_factor: f64, m_j: f64) -> f64 {
    // The Bohr magneton in joules per tesla.
    const BOHR_MAGNETON: f64 = 9.274_010_078_3e-24;
    g_factor * BOHR_MAGNETON * b * m_j
}

/// The hydrogen hyperfine transition frequency in hertz: the 21 centimetre
/// line.
///
/// The transition is forbidden to first order and has a mean lifetime of some
/// ten million years, so no laboratory sample of hydrogen would ever show it.
/// The galaxy has enough hydrogen that it is the brightest line in radio
/// astronomy.
#[must_use]
pub fn hyperfine_hydrogen_21cm() -> f64 {
    1_420_405_751.768
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn multiply(a: &[Vec<Complex>], b: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
        let n = a.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (0..n).fold(ZERO, |acc, k| acc + a[i][k] * b[k][j]))
                    .collect()
            })
            .collect()
    }

    fn subtract(a: &[Vec<Complex>], b: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
        a.iter()
            .zip(b)
            .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| *x - *y).collect())
            .collect()
    }

    fn matrix_close(a: &[Vec<Complex>], b: &[Vec<Complex>], tol: f64) -> bool {
        a.iter().zip(b).all(|(ra, rb)| {
            ra.iter()
                .zip(rb)
                .all(|(x, y)| (x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol)
        })
    }

    // -----------------------------------------------------------------
    // Spin operators
    // -----------------------------------------------------------------

    #[test]
    fn the_spin_operators_satisfy_the_angular_momentum_algebra() {
        // [Sx, Sy] = i Sz and its cyclic partners define angular momentum;
        // everything else about spin follows from them, so they are what to
        // test rather than any particular matrix element.
        for &s in &[0.5f64, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let (sx, sy, sz) = spin_operators(s).unwrap();
            let dim = sx.len();
            assert_eq!(dim, (2.0 * s) as usize + 1);

            let commutator = |a: &[Vec<Complex>], b: &[Vec<Complex>]| -> Vec<Vec<Complex>> {
                subtract(&multiply(a, b), &multiply(b, a))
            };
            let i_times = |m: &[Vec<Complex>]| -> Vec<Vec<Complex>> {
                m.iter()
                    .map(|row| row.iter().map(|z| Complex::new(0.0, 1.0) * *z).collect())
                    .collect()
            };
            assert!(
                matrix_close(&commutator(&sx, &sy), &i_times(&sz), 1e-10),
                "[Sx, Sy] != i Sz at s = {s}"
            );
            assert!(
                matrix_close(&commutator(&sy, &sz), &i_times(&sx), 1e-10),
                "[Sy, Sz] != i Sx at s = {s}"
            );
            assert!(
                matrix_close(&commutator(&sz, &sx), &i_times(&sy), 1e-10),
                "[Sz, Sx] != i Sy at s = {s}"
            );

            // S^2 = s(s + 1) times the identity, which is what makes s a
            // good quantum number.
            let square = (0..dim)
                .map(|i| {
                    (0..dim)
                        .map(|j| {
                            multiply(&sx, &sx)[i][j]
                                + multiply(&sy, &sy)[i][j]
                                + multiply(&sz, &sz)[i][j]
                        })
                        .collect::<Vec<Complex>>()
                })
                .collect::<Vec<_>>();
            for i in 0..dim {
                for j in 0..dim {
                    let expected = if i == j { s * (s + 1.0) } else { 0.0 };
                    assert!(
                        close(square[i][j].re, expected, 1e-9) && close(square[i][j].im, 0.0, 1e-10),
                        "S^2 is wrong at ({i}, {j}) for s = {s}"
                    );
                }
            }
            // Each operator is Hermitian and traceless.
            for m in [&sx, &sy, &sz] {
                let trace = (0..dim).fold(ZERO, |acc, i| acc + m[i][i]);
                assert!(close(trace.re, 0.0, 1e-10) && close(trace.im, 0.0, 1e-10));
                for i in 0..dim {
                    for j in 0..dim {
                        assert!(
                            close(m[i][j].re, m[j][i].re, 1e-12)
                                && close(m[i][j].im, -m[j][i].im, 1e-12),
                            "the operator is not Hermitian"
                        );
                    }
                }
            }
        }
        // Spin one half is exactly half the Pauli matrices.
        let (sx, sy, sz) = spin_operators(0.5).unwrap();
        let pauli = pauli_matrices();
        for (operator, sigma) in [(&sx, &pauli[0]), (&sy, &pauli[1]), (&sz, &pauli[2])] {
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        close(operator[i][j].re, sigma[i][j].re / 2.0, 1e-12)
                            && close(operator[i][j].im, sigma[i][j].im / 2.0, 1e-12)
                    );
                }
            }
        }
        assert!(spin_operators(0.3).is_err());
        assert!(spin_operators(-1.0).is_err());
        assert!(spin_operators(50.0).is_err());
    }

    #[test]
    fn a_spin_coherent_state_points_where_it_was_asked_to() {
        // The expectation of the spin vector must be `s` times the unit
        // vector in the given direction, exactly -- that is what makes it the
        // closest thing to a classical arrow.
        for &s in &[0.5f64, 1.0, 2.0, 4.0] {
            let (sx, sy, sz) = spin_operators(s).unwrap();
            for &(theta, phi) in &[
                (0.0f64, 0.0f64),
                (std::f64::consts::FRAC_PI_2, 0.0),
                (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                (0.7, 1.9),
                (2.4, -0.6),
                (std::f64::consts::PI, 0.3),
            ] {
                let state = spin_coherent_state(s, theta, phi).unwrap();
                let norm: f64 = state.iter().map(|z| z.norm_sq()).sum();
                assert!(close(norm, 1.0, 1e-9), "the state has norm {norm}");

                let expectation = |m: &[Vec<Complex>]| -> f64 {
                    let mut total = ZERO;
                    for i in 0..state.len() {
                        for j in 0..state.len() {
                            total = total + state[i].conjugate() * m[i][j] * state[j];
                        }
                    }
                    total.re
                };
                let (x, y, z) = (expectation(&sx), expectation(&sy), expectation(&sz));
                assert!(
                    close(x, s * theta.sin() * phi.cos(), 1e-8)
                        && close(y, s * theta.sin() * phi.sin(), 1e-8)
                        && close(z, s * theta.cos(), 1e-8),
                    "s = {s} at ({theta}, {phi}): got ({x}, {y}, {z})"
                );
                // Its length is exactly s: a coherent state is as classical
                // as the algebra permits.
                assert!(close(x.hypot(y).hypot(z), s, 1e-8));
            }
        }
        assert!(spin_coherent_state(0.25, 0.0, 0.0).is_err());
    }

    // -----------------------------------------------------------------
    // Spin chains
    // -----------------------------------------------------------------

    #[test]
    fn two_heisenberg_spins_split_into_a_singlet_and_a_triplet() {
        // The exactly solvable case, and the one every larger calculation
        // should reduce to.
        for j in [1.0f64, -1.0, 2.5] {
            let chain = SpinChain::new(2, j, j, 0.0, false).unwrap();
            let mut spectrum = chain.spectrum_small().unwrap();
            let mut expected = heisenberg_2site_exact(j);
            spectrum.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            expected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for (got, want) in spectrum.iter().zip(&expected) {
                assert!(close(*got, *want, 1e-9), "got {spectrum:?}, expected {expected:?}");
            }
        }
        // The antiferromagnetic ground state is the singlet, whose
        // magnetisation is zero and whose correlation is -1/4.
        let chain = SpinChain::new(2, 1.0, 1.0, 0.0, false).unwrap();
        let mut rng = Rng::new(0x_5911_0001);
        let (energy, state) = chain.ground_state_lanczos(20, &mut rng).unwrap();
        assert!(close(energy, -0.75, 1e-9), "the ground energy is {energy}");
        assert!(close(chain.magnetization(&state).unwrap(), 0.0, 1e-9));
        assert!(
            close(chain.correlation(&state, 0, 1).unwrap(), -0.25, 1e-9),
            "the correlation is {}",
            chain.correlation(&state, 0, 1).unwrap()
        );
        // And it is maximally entangled: one bit across the only cut.
        assert!(
            close(chain.entanglement_entropy_cut(&state, 1).unwrap(), 1.0, 1e-8),
            "the entropy is {}",
            chain.entanglement_entropy_cut(&state, 1).unwrap()
        );
    }

    #[test]
    fn lanczos_reproduces_the_dense_ground_state_on_every_chain_it_is_given() {
        // Two methods with nothing in common but the Hamiltonian: one forms
        // the matrix and diagonalises it, the other never stores it.
        let mut rng = Rng::new(0x_5911_0002);
        // Six sites is where a Jacobi sweep on a 2^n square matrix stops
        // being cheap; past that the residual check below stands on its own.
        for n in 2..=6usize {
            for (j, jz, h) in [(1.0, 1.0, 0.0), (1.0, 0.5, 0.3), (0.0, 1.0, 0.0), (1.0, -0.7, -0.4)]
            {
                let chain = SpinChain::new(n, j, jz, h, n > 2).unwrap();
                let spectrum = chain.spectrum_small().unwrap();
                let (energy, state) = chain.ground_state_lanczos(60, &mut rng).unwrap();
                assert!(
                    close(energy, spectrum[0], 1e-8),
                    "n = {n}, ({j}, {jz}, {h}): Lanczos gives {energy}, dense {}",
                    spectrum[0]
                );
                // The Lanczos vector is a genuine eigenvector: its residual
                // vanishes, which needs no reference at all.
                let applied = chain.apply(&state).unwrap();
                let mut residual: f64 = 0.0;
                for (a, b) in applied.iter().zip(&state) {
                    let difference = *a - scale(*b, energy);
                    residual = residual.max(difference.norm());
                }
                assert!(residual < 1e-7, "the residual is {residual}");
                let norm: f64 = state.iter().map(|z| z.norm_sq()).sum();
                assert!(close(norm, 1.0, 1e-9));
            }
        }
    }

    #[test]
    fn the_transverse_field_ising_chain_matches_its_free_fermion_solution() {
        // The Jordan-Wigner result is exact, so this is a check on the
        // formula's conventions as much as on the diagonalisation -- and a
        // convention error would show as a constant factor, which no
        // tolerance would hide.
        let mut rng = Rng::new(0x_5911_00FF);
        for n in [2usize, 4, 6] {
            for g in [0.0f64, 0.3, 0.7, 1.0, 1.5, 3.0] {
                // Full diagonalisation, where the matrix is small enough to
                // form: this checks the whole spectrum's floor, not just what
                // an iterative method converges to.
                let dense = ising_transverse_field_dense(n, g, true).unwrap();
                let decomposition =
                    crate::linalg::eigen::eigen_symmetric(&dense, 1e-12, 400).unwrap();
                let numerical = decomposition
                    .values
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let exact = ising_transverse_field_exact(n, g).unwrap();
                assert!(
                    close(numerical, exact, 1e-8),
                    "n = {n}, g = {g}: dense gives {numerical}, free fermions {exact}"
                );
            }
        }
        for n in [8usize, 10, 12] {
            for g in [0.4f64, 1.0, 2.0] {
                // Above six sites the Jacobi sweep is cubic in a matrix with
                // a million entries, so the ground state comes from Lanczos
                // on the matrix-free operator instead.
                let matvec =
                    |v: &[Complex]| ising_transverse_field_apply(n, g, true, v).unwrap();
                let (values, vectors) = lanczos(&matvec, 1usize << n, 80, &mut rng).unwrap();
                let exact = ising_transverse_field_exact(n, g).unwrap();
                assert!(
                    close(values[0], exact, 1e-7),
                    "n = {n}, g = {g}: Lanczos gives {}, free fermions {exact}",
                    values[0]
                );
                // The Lanczos vector really is an eigenvector.
                let applied = matvec(&vectors[0]);
                let mut residual: f64 = 0.0;
                for (a, b) in applied.iter().zip(&vectors[0]) {
                    residual = residual.max((*a - scale(*b, values[0])).norm());
                }
                assert!(residual < 1e-6, "n = {n}, g = {g}: the residual is {residual}");
            }
        }
        // At zero field the ground state is the two aligned configurations,
        // energy -n; at large field it is -n g.
        assert!(close(ising_transverse_field_exact(8, 0.0).unwrap(), -8.0, 1e-9));
        let strong = ising_transverse_field_exact(8, 100.0).unwrap();
        assert!(close(strong, -800.0, 0.1), "the strong-field limit is {strong}");
        assert!(close(itf_critical_point(), 1.0, 1e-15));
        assert!(ising_transverse_field_exact(1, 1.0).is_err());
        assert!(ising_transverse_field_dense(1, 1.0, true).is_err());
        assert!(ising_transverse_field_dense(11, 1.0, true).is_err());
    }

    #[test]
    fn the_field_polarises_the_chain_and_the_coupling_resists() {
        // A physical check rather than an algebraic one: raising the
        // longitudinal field must raise the magnetisation monotonically to
        // saturation, and an antiferromagnetic coupling must make that
        // harder than a ferromagnetic one.
        let mut rng = Rng::new(0x_5911_0003);
        let n = 6usize;
        let mut previous = -1.0;
        for h in [0.0f64, 0.5, 1.0, 2.0, 4.0, 10.0] {
            let chain = SpinChain::new(n, 1.0, 1.0, h, false).unwrap();
            let (_, state) = chain.ground_state_lanczos(60, &mut rng).unwrap();
            let m = chain.magnetization(&state).unwrap();
            assert!(m >= previous - 1e-6, "the magnetisation fell from {previous} to {m} at h = {h}");
            assert!((-0.5 - 1e-9..=0.5 + 1e-9).contains(&m), "the magnetisation is {m}");
            previous = m;
        }
        assert!(close(previous, 0.5, 1e-6), "a strong field should saturate: {previous}");

        // At zero field the antiferromagnet has alternating correlations and
        // the ferromagnet does not.
        let antiferro = SpinChain::new(n, 1.0, 1.0, 0.0, false).unwrap();
        let (_, state) = antiferro.ground_state_lanczos(80, &mut rng).unwrap();
        assert!(
            antiferro.correlation(&state, 0, 1).unwrap() < 0.0,
            "neighbours should anticorrelate"
        );
        assert!(
            antiferro.correlation(&state, 0, 2).unwrap() > 0.0,
            "next neighbours should correlate"
        );
        // Which shows up as a peak in the structure factor at k = pi.
        let at_pi = antiferro.structure_factor(&state, std::f64::consts::PI).unwrap();
        let at_zero = antiferro.structure_factor(&state, 0.0).unwrap();
        assert!(
            at_pi > 5.0 * at_zero.abs().max(1e-6),
            "the antiferromagnetic peak is {at_pi} against {at_zero} at k = 0"
        );
    }

    #[test]
    fn krylov_evolution_is_unitary_and_conserves_the_energy() {
        // Time evolution under a Hermitian Hamiltonian preserves both, and a
        // Krylov step does so by construction rather than approximately.
        let mut rng = Rng::new(0x_5911_0004);
        let chain = SpinChain::new(6, 1.0, 0.6, 0.2, true).unwrap();
        let size = 1usize << 6;
        let mut state: Vec<Complex> = (0..size)
            .map(|_| Complex::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
            .collect();
        let magnitude: f64 = state.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
        for z in &mut state {
            *z = scale(*z, 1.0 / magnitude);
        }
        let energy_of = |v: &[Complex]| -> f64 {
            let applied = chain.apply(v).unwrap();
            inner(v, &applied).re / v.iter().map(|z| z.norm_sq()).sum::<f64>()
        };
        let initial = energy_of(&state);

        for t in [0.1f64, 1.0, 5.0] {
            let moved = chain.time_evolve_krylov(&state, t, 40).unwrap();
            let norm: f64 = moved.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
            assert!(close(norm, 1.0, 1e-9), "at t = {t} the norm is {norm}");
            assert!(
                close(energy_of(&moved), initial, 1e-8),
                "at t = {t} the energy moved from {initial} to {}",
                energy_of(&moved)
            );
        }

        // An eigenstate only picks up a phase, so its density is unchanged.
        let (energy, ground) = chain.ground_state_lanczos(60, &mut rng).unwrap();
        let moved = chain.time_evolve_krylov(&ground, 3.0, 60).unwrap();
        for (a, b) in moved.iter().zip(&ground) {
            assert!(close(a.norm(), b.norm(), 1e-7), "an eigenstate changed shape");
        }
        // And the phase is exactly exp(-i E t).
        let overlap = inner(&ground, &moved);
        let expected = Complex::new((-energy * 3.0).cos(), (-energy * 3.0).sin());
        assert!(
            close(overlap.re, expected.re, 1e-6) && close(overlap.im, expected.im, 1e-6),
            "the phase is {overlap:?}, expected {expected:?}"
        );

        assert!(chain.time_evolve_krylov(&state, 1.0, 0).is_err());
        assert!(chain.time_evolve_krylov(&[ZERO; 4], 1.0, 5).is_err());
    }

    #[test]
    fn only_the_critical_chain_keeps_entangling_as_it_grows() {
        // The distinguishing property of a critical point is not the *value*
        // of the entanglement entropy but its *scaling*: at criticality the
        // half-chain entropy grows as (c / 6) log L with c = 1/2, and away
        // from it the entropy saturates at a constant set by the correlation
        // length. A test comparing values at one size would get this exactly
        // backwards, because deep in the ordered phase the ground state is
        // the symmetry-broken cat and carries a full bit across every cut --
        // more than the critical chain of the same size, and none of it from
        // correlations.
        let mut rng = Rng::new(0x_5911_0005);
        let sizes = [4usize, 6, 8, 10, 12];
        let entropy_curve = |g: f64, rng: &mut Rng| -> Vec<f64> {
            sizes
                .iter()
                .map(|&n| {
                    let matvec =
                        |v: &[Complex]| ising_transverse_field_apply(n, g, false, v).unwrap();
                    let (_, vectors) = lanczos(&matvec, 1usize << n, 70, rng).unwrap();
                    let chain = SpinChain::new(n, 0.0, 0.0, 0.0, false).unwrap();
                    chain.entanglement_entropy_cut(&vectors[0], n / 2).unwrap()
                })
                .collect()
        };

        // Ordered: a single bit, from the two-fold degeneracy, and flat.
        let ordered = entropy_curve(0.2, &mut rng);
        for value in &ordered {
            assert!(close(*value, 1.0, 0.01), "the ordered phase gives {ordered:?}");
        }

        // Disordered: nearly a product state, and flat.
        let disordered = entropy_curve(3.0, &mut rng);
        assert!(disordered[0] < 0.1, "the disordered phase gives {disordered:?}");
        assert!(
            (disordered[4] - disordered[2]).abs() < 0.005,
            "the disordered entropy did not saturate: {disordered:?}"
        );

        // Critical: still climbing at every size, with the slope the
        // conformal field theory predicts. The Ising chain has central
        // charge one half, so the half-chain entropy rises by c / 6 = 1/12
        // of a bit per doubling of the block.
        let critical = entropy_curve(1.0, &mut rng);
        assert!(
            critical.windows(2).all(|w| w[1] > w[0] + 1e-3),
            "the critical entropy stopped growing: {critical:?}"
        );
        let slope = (critical[4] - critical[2])
            / ((sizes[4] as f64 / 2.0).log2() - (sizes[2] as f64 / 2.0).log2());
        assert!(
            (slope - 1.0 / 12.0).abs() < 0.25 / 12.0,
            "the critical slope is {slope} bits per doubling, not near {}",
            1.0 / 12.0
        );
        // And the off-critical curves have essentially no slope at all.
        for (name, curve) in [("ordered", &ordered), ("disordered", &disordered)] {
            let flat = (curve[4] - curve[2])
                / ((sizes[4] as f64 / 2.0).log2() - (sizes[2] as f64 / 2.0).log2());
            assert!(
                flat.abs() < slope / 4.0,
                "the {name} phase has slope {flat} against the critical {slope}"
            );
        }
    }

    #[test]
    fn magnons_disperse_quadratically_at_long_wavelength() {
        // The gapless quadratic mode is the ferromagnet's Goldstone boson,
        // and the quadratic -- rather than linear -- form is what makes a
        // ferromagnet different from an antiferromagnet.
        let (j, s, a) = (1.0f64, 0.5f64, 1.0f64);
        assert!(close(magnon_dispersion(j, 0.0, s, a), 0.0, 1e-15));
        for k in [0.01f64, 0.02, 0.04] {
            let energy = magnon_dispersion(j, k, s, a);
            let quadratic = j * s * k * k;
            assert!(
                (energy - quadratic).abs() < 1e-3 * quadratic,
                "at k = {k} the dispersion is {energy}, the quadratic form {quadratic}"
            );
        }
        // Halving the wavevector quarters the energy.
        let ratio = magnon_dispersion(j, 0.02, s, a) / magnon_dispersion(j, 0.01, s, a);
        assert!(close(ratio, 4.0, 1e-3), "the ratio is {ratio}");
        // The band top is at the zone boundary, where the cosine is minus
        // one and the dispersion reaches 4 j s -- twice the coefficient, not
        // equal to it.
        assert!(close(magnon_dispersion(j, std::f64::consts::PI, s, a), 4.0 * j * s, 1e-12));
    }

    // -----------------------------------------------------------------
    // Magnetic resonance
    // -----------------------------------------------------------------

    #[test]
    fn larmor_precession_turns_at_the_rate_the_field_sets() {
        // The rate depends on the field and not on the angle, and the z
        // component never moves.
        let (b, gamma) = (2.5f64, 1.7f64);
        let omega = larmor_frequency(b, gamma);
        assert!(close(omega, 4.25, 1e-12));
        for m0 in [(1.0f64, 0.0f64, 0.0f64), (0.6, -0.8, 0.0), (0.3, 0.4, 0.5)] {
            let period = 2.0 * std::f64::consts::PI / omega;
            let after = larmor_precession(m0, b, gamma, period);
            assert!(
                close(after.0, m0.0, 1e-9) && close(after.1, m0.1, 1e-9),
                "a full period should return it: {after:?} against {m0:?}"
            );
            // A quarter turn clockwise, matching the Bloch equation's sense.
            let quarter = larmor_precession(m0, b, gamma, period / 4.0);
            assert!(
                close(quarter.0, m0.1, 1e-9) && close(quarter.1, -m0.0, 1e-9),
                "a quarter period gave {quarter:?} from {m0:?}"
            );
            // The length and the z component are conserved.
            assert!(close(quarter.2, m0.2, 1e-15));
            let before = m0.0.hypot(m0.1).hypot(m0.2);
            assert!(close(quarter.0.hypot(quarter.1).hypot(quarter.2), before, 1e-12));
        }
    }

    #[test]
    fn rabi_flopping_is_complete_on_resonance_and_partial_off_it() {
        // On resonance the population reaches one; the peak falls as the
        // detuning grows, and the oscillation speeds up. Both halves are
        // exact, so both are checked against the closed form rather than
        // eyeballed.
        let rabi = 2.0f64;
        for t in [0.0f64, 0.3, 1.1, 2.7] {
            let expected = (rabi * t / 2.0).sin().powi(2);
            assert!(
                close(rabi_oscillation(rabi, 0.0, t).unwrap(), expected, 1e-12),
                "on resonance at t = {t}"
            );
        }
        // The pi pulse.
        let pi_pulse = std::f64::consts::PI / rabi;
        assert!(close(rabi_oscillation(rabi, 0.0, pi_pulse).unwrap(), 1.0, 1e-12));
        // The pi over two pulse leaves half.
        assert!(close(rabi_oscillation(rabi, 0.0, pi_pulse / 2.0).unwrap(), 0.5, 1e-12));

        let mut previous_peak = 1.0;
        for detuning in [0.0f64, 1.0, 2.0, 5.0, 20.0] {
            let generalised = (rabi * rabi + detuning * detuning).sqrt();
            let peak_time = std::f64::consts::PI / generalised;
            let peak = rabi_oscillation(rabi, detuning, peak_time).unwrap();
            let expected = rabi * rabi / (generalised * generalised);
            assert!(close(peak, expected, 1e-12), "at detuning {detuning} the peak is {peak}");
            assert!(peak <= previous_peak + 1e-12, "the peak rose with the detuning");
            previous_peak = peak;
            // Never outside [0, 1].
            for t in [0.1f64, 0.9, 3.3, 8.8] {
                let p = rabi_oscillation(rabi, detuning, t).unwrap();
                assert!((0.0..=1.0).contains(&p), "the probability is {p}");
            }
        }
        assert!(previous_peak < 0.02, "a large detuning should nearly forbid the transition");
        assert!(rabi_oscillation(0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn ramsey_fringes_measure_the_detuning_and_the_echo_beats_the_dephasing() {
        // The fringe period is the detuning's reciprocal, and the envelope
        // decays at T2*. The echo, by construction, decays at the longer T2
        // instead -- which is the whole reason to apply one.
        let detuning = 3.0f64;
        let t2_star = 2.0f64;
        let period = 2.0 * std::f64::consts::PI / detuning;
        for k in 0..5usize {
            let t = k as f64 * period;
            let expected = 0.5 * (1.0 + (-t / t2_star).exp());
            assert!(
                close(ramsey_fringes(detuning, t, t2_star), expected, 1e-12),
                "the fringe maximum at t = {t} is wrong"
            );
            let trough = t + period / 2.0;
            let expected = 0.5 * (1.0 - (-trough / t2_star).exp());
            assert!(close(ramsey_fringes(detuning, trough, t2_star), expected, 1e-12));
        }
        // Contrast falls monotonically.
        let contrast = |t: f64| {
            ramsey_fringes(detuning, t, t2_star) - ramsey_fringes(detuning, t + period / 2.0, t2_star)
        };
        let mut previous = f64::INFINITY;
        for k in 0..6usize {
            let value = contrast(k as f64 * period);
            assert!(value < previous, "the contrast rose at k = {k}");
            previous = value;
        }

        // The echo outlives the Ramsey envelope whenever T2 exceeds T2*.
        let t2 = 20.0f64;
        for t in [1.0f64, 4.0, 10.0] {
            assert!(
                spin_echo_sim(t, t2, t2_star) > (-t / t2_star).exp(),
                "the echo should beat the free decay at t = {t}"
            );
        }
        assert!(close(spin_echo_sim(0.0, t2, t2_star), 1.0, 1e-15));
        assert!(close(spin_echo_sim(1.0, 0.0, t2_star), 0.0, 1e-15));
    }

    #[test]
    fn the_bloch_equations_relax_at_the_times_they_are_given() {
        // With no field the transverse components decay at T2 and the
        // longitudinal one approaches equilibrium at T1, both exponentially
        // and both checkable against the closed form.
        let (t1, t2) = (4.0f64, 1.5f64);
        let trajectory = bloch_equations(
            (1.0, 0.0, 0.0),
            &|_| (0.0, 0.0, 0.0),
            1.0,
            t1,
            t2,
            1.0,
            8.0,
            0.001,
        )
        .unwrap();
        for (k, m) in trajectory.iter().enumerate().step_by(200) {
            let t = k as f64 * 0.001;
            assert!(
                close(m.0, (-t / t2).exp(), 1e-6),
                "at t = {t} the transverse component is {}, not {}",
                m.0,
                (-t / t2).exp()
            );
            assert!(
                close(m.2, 1.0 - (-t / t1).exp(), 1e-6),
                "at t = {t} the longitudinal component is {}",
                m.2
            );
        }

        // With a field and no relaxation, the vector precesses and keeps its
        // length -- which a first-order integrator would not manage.
        let precessing = bloch_equations(
            (1.0, 0.0, 0.0),
            &|_| (0.0, 0.0, 2.0),
            1.0,
            1e9,
            1e9,
            0.0,
            10.0,
            0.001,
        )
        .unwrap();
        for m in precessing.iter().step_by(500) {
            assert!(
                close(m.0.hypot(m.1).hypot(m.2), 1.0, 1e-6),
                "the length drifted to {}",
                m.0.hypot(m.1).hypot(m.2)
            );
        }
        // And in the same clockwise sense as the closed form above.
        let last = precessing.last().unwrap();
        let angle = -2.0 * 10.0f64;
        assert!(
            close(last.0, angle.cos(), 1e-4) && close(last.1, angle.sin(), 1e-4),
            "the ODE ended at {last:?}, the closed form at ({}, {})",
            angle.cos(),
            angle.sin()
        );
        let closed = larmor_precession((1.0, 0.0, 0.0), 2.0, 1.0, 10.0);
        assert!(close(last.0, closed.0, 1e-4) && close(last.1, closed.1, 1e-4));

        assert!(bloch_equations((1.0, 0.0, 0.0), &|_| (0.0, 0.0, 0.0), 1.0, 0.0, 1.0, 1.0, 1.0, 0.1).is_err());
        assert!(bloch_equations((1.0, 0.0, 0.0), &|_| (0.0, 0.0, 0.0), 1.0, 1.0, 1.0, 1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn the_free_induction_decay_carries_its_frequencies_into_the_spectrum() {
        // The signal is measured in time and read in frequency, so the test
        // takes the transform and looks for the peaks it was given.
        let frequencies = [40.0f64, 110.0];
        // Short enough that the record captures the whole decay: at 0.4
        // seconds the signal is still a seventh of its start after a full
        // second, which is a truncated record rather than a decayed one.
        let decays = [0.12f64, 0.12];
        let rate = 1024.0f64;
        let samples = 1024usize;
        let signal = nmr_fid(&frequencies, &decays, samples, rate).unwrap();
        assert_eq!(signal.len(), samples);
        // It starts at the number of components and decays away.
        assert!(close(signal[0], 2.0, 1e-12));
        assert!(signal[samples - 1].abs() < 1e-3, "the tail is {}", signal[samples - 1]);

        let spectrum: Vec<f64> = {
            let input: Vec<Complex> =
                signal.iter().map(|v| Complex::new(*v, 0.0)).collect();
            crate::transforms::fft::fft(&input)
                .iter()
                .take(samples / 2)
                .map(|z| z.norm())
                .collect()
        };
        for f in &frequencies {
            let bin = (f * samples as f64 / rate).round() as usize;
            let local = spectrum[bin];
            // The peak dominates its neighbourhood.
            let away = spectrum[bin + 30];
            assert!(
                local > 8.0 * away,
                "the peak at {f} hertz is {local} against {away} thirty bins away"
            );
        }
        assert!(nmr_fid(&[], &[], 10, 100.0).is_err());
        assert!(nmr_fid(&[1.0], &[1.0, 2.0], 10, 100.0).is_err());
        assert!(nmr_fid(&[1.0], &[0.0], 10, 100.0).is_err());
        assert!(nmr_fid(&[1.0], &[1.0], 0, 100.0).is_err());
    }

    #[test]
    fn the_zeeman_shift_is_linear_and_the_hyperfine_line_is_where_it_should_be() {
        // The shift is linear in the field and in m_j, and it vanishes for
        // m_j = 0 -- which is why the anomalous Zeeman pattern has an
        // unshifted central line.
        let g = 2.002_319;
        assert!(close(zeeman_splitting(1.0, g, 0.0), 0.0, 1e-30));
        let one = zeeman_splitting(1.0, g, 0.5);
        assert!(close(zeeman_splitting(2.0, g, 0.5), 2.0 * one, 1e-30));
        assert!(close(zeeman_splitting(1.0, g, -0.5), -one, 1e-30));
        // The electron spin resonance frequency at one tesla is about 28 GHz.
        let frequency = 2.0 * one / 6.626_070_15e-34;
        assert!(
            (frequency - 28.0e9).abs() < 0.5e9,
            "electron spin resonance at one tesla is {frequency} hertz"
        );

        // The 21 centimetre line, checked by its wavelength rather than
        // restated.
        let wavelength = 299_792_458.0 / hyperfine_hydrogen_21cm();
        assert!(
            close(wavelength, 0.2110611405, 1e-9),
            "the wavelength is {wavelength} metres"
        );
    }

    #[test]
    fn the_solvers_refuse_degenerate_input() {
        assert!(SpinChain::new(1, 1.0, 1.0, 0.0, false).is_err());
        assert!(SpinChain::new(17, 1.0, 1.0, 0.0, false).is_err());
        let chain = SpinChain::new(4, 1.0, 1.0, 0.0, false).unwrap();
        assert!(chain.apply(&[ZERO; 3]).is_err());
        assert!(chain.magnetization(&[ZERO; 3]).is_err());
        assert!(chain.correlation(&[ZERO; 16], 9, 0).is_err());
        assert!(chain.entanglement_entropy_cut(&[ZERO; 16], 0).is_err());
        assert!(chain.entanglement_entropy_cut(&[ZERO; 16], 4).is_err());
        assert!(SpinChain::new(12, 1.0, 1.0, 0.0, false).unwrap().hamiltonian_dense().is_err());
        let mut rng = Rng::new(7);
        assert!(lanczos(&|v| v.to_vec(), 0, 5, &mut rng).is_err());
        assert!(lanczos(&|v| v.to_vec(), 8, 0, &mut rng).is_err());
        // A zero state has no magnetisation to report rather than a division
        // by zero.
        assert_eq!(chain.magnetization(&[ZERO; 16]).unwrap(), 0.0);
        assert_eq!(chain.correlation(&[ZERO; 16], 0, 1).unwrap(), 0.0);
    }
}
