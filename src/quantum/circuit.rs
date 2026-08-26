//! A state-vector quantum circuit simulator, with density matrices and noise
//! channels.
//!
//! The representation is the whole story. An `n`-qubit pure state is a vector
//! of `2^n` complex amplitudes, so the memory doubles with each qubit: thirty
//! qubits is sixteen gigabytes and there is no cleverness that avoids it for
//! a general state. That exponential is not a limitation of this
//! implementation but the reason quantum computers are interesting, and it is
//! why everything here is capped at a couple of dozen qubits.
//!
//! Applying a one-qubit gate does *not* cost `2^n x 2^n` work. The gate acts
//! on one tensor factor, so the amplitudes split into `2^(n-1)` independent
//! pairs and each pair gets a two-by-two multiply: `O(2^n)` in total. Building
//! the full unitary and multiplying would be `O(4^n)` and is offered only for
//! small circuits, where seeing the matrix is the point.
//!
//! Qubit `q` is bit `q` of the amplitude index, so `|q_2 q_1 q_0>` has index
//! `4 q_2 + 2 q_1 + q_0`. The opposite convention is equally common and the
//! two disagree on every multi-qubit gate, so it is stated here rather than
//! left to be inferred.

use crate::error::GeomError;
use crate::fractals::Complex;
use crate::monte_carlo::Rng;

/// Tolerance for unitarity and trace checks.
const QUANTUM_TOL: f64 = 1e-10;

/// The largest qubit count this module will allocate for.
const MAX_QUBITS: usize = 26;

const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
const ONE: Complex = Complex { re: 1.0, im: 0.0 };

fn scale(z: Complex, k: f64) -> Complex {
    Complex::new(z.re * k, z.im * k)
}

fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

/// A pure state of `n` qubits, as `2^n` amplitudes.
#[derive(Debug, Clone)]
pub struct QState {
    /// The number of qubits.
    pub n: usize,
    /// The amplitudes, indexed so that qubit `q` is bit `q`.
    pub amps: Vec<Complex>,
}

impl QState {
    /// The all-zeros computational basis state.
    ///
    /// # Errors
    /// Returns an error for zero qubits or more than `MAX_QUBITS` (26).
    pub fn zero(n: usize) -> Result<Self, GeomError> {
        Self::basis(n, 0)
    }

    /// A computational basis state.
    ///
    /// # Errors
    /// Returns an error for a bad qubit count or an out-of-range index.
    pub fn basis(n: usize, index: u64) -> Result<Self, GeomError> {
        if n == 0 || n > MAX_QUBITS {
            return Err(GeomError::InvalidArgument("the qubit count is out of range"));
        }
        let size = 1usize << n;
        if index as usize >= size {
            return Err(GeomError::InvalidArgument("the basis index is out of range"));
        }
        let mut amps = vec![ZERO; size];
        amps[index as usize] = ONE;
        Ok(Self { n, amps })
    }

    /// A state from explicit amplitudes, normalised on the way in.
    ///
    /// # Errors
    /// Returns an error unless the length is a power of two in range, and the
    /// amplitudes are not all zero.
    pub fn from_amps(amps: Vec<Complex>) -> Result<Self, GeomError> {
        if !amps.len().is_power_of_two() {
            return Err(GeomError::InvalidArgument("the amplitude count must be a power of two"));
        }
        let n = amps.len().trailing_zeros() as usize;
        if n == 0 || n > MAX_QUBITS {
            return Err(GeomError::InvalidArgument("the qubit count is out of range"));
        }
        let mut state = Self { n, amps };
        if state.norm() <= 0.0 {
            return Err(GeomError::InvalidArgument("the state is identically zero"));
        }
        state.normalize();
        Ok(state)
    }

    /// The equal superposition over every basis state.
    ///
    /// # Errors
    /// Returns an error for a bad qubit count.
    pub fn plus_all(n: usize) -> Result<Self, GeomError> {
        if n == 0 || n > MAX_QUBITS {
            return Err(GeomError::InvalidArgument("the qubit count is out of range"));
        }
        let size = 1usize << n;
        let amplitude = 1.0 / (size as f64).sqrt();
        Ok(Self { n, amps: vec![Complex::new(amplitude, 0.0); size] })
    }

    /// The number of amplitudes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.amps.len()
    }

    /// Always false: a state always has at least one qubit.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The Euclidean norm.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.amps.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt()
    }

    /// Rescales to unit norm, leaving a zero state alone.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 0.0 {
            let inverse = 1.0 / n;
            for z in &mut self.amps {
                *z = scale(*z, inverse);
            }
        }
    }

    /// The probability of a basis outcome.
    #[must_use]
    pub fn probability(&self, index: u64) -> f64 {
        self.amps.get(index as usize).map_or(0.0, |z| z.norm_sq())
    }

    /// Every outcome probability.
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(|z| z.norm_sq()).collect()
    }

    /// Samples one measurement of every qubit, returning the outcome as bits.
    pub fn measure_all(&self, rng: &mut Rng) -> u64 {
        let target = rng.next_f64() * self.amps.iter().map(|z| z.norm_sq()).sum::<f64>();
        let mut running = 0.0;
        for (index, z) in self.amps.iter().enumerate() {
            running += z.norm_sq();
            if running >= target {
                return index as u64;
            }
        }
        (self.len() - 1) as u64
    }

    /// Measures one qubit, returning the outcome and the collapsed state.
    ///
    /// The collapse is the projection onto the observed outcome, renormalised.
    /// Note what survives: the *other* qubits keep whatever correlations they
    /// had with this one, which is why measuring half of a Bell pair
    /// determines the other half.
    ///
    /// # Errors
    /// Returns an error if the qubit index is out of range.
    pub fn measure_qubit(&self, q: usize, rng: &mut Rng) -> Result<(bool, Self), GeomError> {
        if q >= self.n {
            return Err(GeomError::InvalidArgument("the qubit index is out of range"));
        }
        let mask = 1usize << q;
        let one_weight: f64 = self
            .amps
            .iter()
            .enumerate()
            .filter(|(i, _)| i & mask != 0)
            .map(|(_, z)| z.norm_sq())
            .sum();
        let outcome = rng.next_f64() < one_weight;
        let weight = if outcome { one_weight } else { 1.0 - one_weight };
        if weight <= 0.0 {
            return Err(GeomError::Degenerate("the measured outcome has zero probability"));
        }
        let inverse = 1.0 / weight.sqrt();
        let amps = self
            .amps
            .iter()
            .enumerate()
            .map(|(i, z)| if (i & mask != 0) == outcome { scale(*z, inverse) } else { ZERO })
            .collect();
        Ok((outcome, Self { n: self.n, amps }))
    }

    /// Repeated measurement, returning `(outcome, count)` pairs sorted by
    /// outcome.
    pub fn sample_counts(&self, shots: usize, rng: &mut Rng) -> Vec<(u64, u64)> {
        let mut counts = std::collections::BTreeMap::new();
        for _ in 0..shots {
            *counts.entry(self.measure_all(rng)).or_insert(0u64) += 1;
        }
        counts.into_iter().collect()
    }

    /// The expectation of `Z` on one qubit.
    ///
    /// # Errors
    /// Returns an error if the qubit index is out of range.
    pub fn expectation_z(&self, q: usize) -> Result<f64, GeomError> {
        if q >= self.n {
            return Err(GeomError::InvalidArgument("the qubit index is out of range"));
        }
        let mask = 1usize << q;
        Ok(self
            .amps
            .iter()
            .enumerate()
            .map(|(i, z)| if i & mask == 0 { z.norm_sq() } else { -z.norm_sq() })
            .sum())
    }

    /// The expectation of a Pauli string such as `"XIZY"`, whose leftmost
    /// character is the highest-numbered qubit.
    ///
    /// Measuring a Pauli string is the primitive every variational algorithm
    /// is built on, because any Hermitian operator decomposes into them.
    ///
    /// # Errors
    /// Returns an error if the string has the wrong length or an unknown
    /// character.
    pub fn expectation_pauli_string(&self, pauli: &str) -> Result<f64, GeomError> {
        if pauli.len() != self.n {
            return Err(GeomError::InvalidArgument("the Pauli string has the wrong length"));
        }
        let mut rotated = self.clone();
        // Rotate X and Y into the Z basis, then read off the parity.
        for (position, symbol) in pauli.chars().enumerate() {
            let q = self.n - 1 - position;
            match symbol {
                'I' | 'Z' => {}
                'X' => rotated.apply_single(q, &Gate::h())?,
                'Y' => {
                    rotated.apply_single(q, &Gate::sdg())?;
                    rotated.apply_single(q, &Gate::h())?;
                }
                _ => return Err(GeomError::InvalidArgument("unknown Pauli symbol")),
            }
        }
        let acting: Vec<usize> = pauli
            .chars()
            .enumerate()
            .filter(|(_, c)| *c != 'I')
            .map(|(position, _)| self.n - 1 - position)
            .collect();
        Ok(rotated
            .amps
            .iter()
            .enumerate()
            .map(|(i, z)| {
                let parity = acting.iter().filter(|&&q| i >> q & 1 == 1).count();
                if parity % 2 == 0 {
                    z.norm_sq()
                } else {
                    -z.norm_sq()
                }
            })
            .sum())
    }

    /// The inner product `<self | other>`.
    ///
    /// # Errors
    /// Returns an error if the two states have different sizes.
    pub fn inner(&self, other: &Self) -> Result<Complex, GeomError> {
        if self.n != other.n {
            return Err(GeomError::InvalidArgument("the states have different sizes"));
        }
        Ok(self
            .amps
            .iter()
            .zip(&other.amps)
            .fold(ZERO, |acc, (a, b)| acc + a.conjugate() * *b))
    }

    /// The fidelity `|<self | other>|^2`.
    ///
    /// # Errors
    /// Returns an error if the two states have different sizes.
    pub fn fidelity(&self, other: &Self) -> Result<f64, GeomError> {
        Ok(self.inner(other)?.norm_sq())
    }

    /// Applies a one-qubit gate in place.
    ///
    /// # Errors
    /// Returns an error if the qubit index is out of range.
    pub fn apply_single(&mut self, q: usize, gate: &Gate) -> Result<(), GeomError> {
        if q >= self.n {
            return Err(GeomError::InvalidArgument("the qubit index is out of range"));
        }
        let mask = 1usize << q;
        for i in 0..self.len() {
            if i & mask != 0 {
                continue;
            }
            let (a, b) = (self.amps[i], self.amps[i | mask]);
            self.amps[i] = gate.matrix[0][0] * a + gate.matrix[0][1] * b;
            self.amps[i | mask] = gate.matrix[1][0] * a + gate.matrix[1][1] * b;
        }
        Ok(())
    }

    /// Applies a one-qubit gate conditioned on a control qubit.
    ///
    /// # Errors
    /// Returns an error if either index is out of range, or they coincide.
    pub fn apply_controlled(
        &mut self,
        control: usize,
        target: usize,
        gate: &Gate,
    ) -> Result<(), GeomError> {
        if control >= self.n || target >= self.n {
            return Err(GeomError::InvalidArgument("the qubit index is out of range"));
        }
        if control == target {
            return Err(GeomError::InvalidArgument("a gate cannot control itself"));
        }
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        for i in 0..self.len() {
            if i & target_mask != 0 || i & control_mask == 0 {
                continue;
            }
            let (a, b) = (self.amps[i], self.amps[i | target_mask]);
            self.amps[i] = gate.matrix[0][0] * a + gate.matrix[0][1] * b;
            self.amps[i | target_mask] = gate.matrix[1][0] * a + gate.matrix[1][1] * b;
        }
        Ok(())
    }

    /// The Toffoli gate.
    ///
    /// # Errors
    /// Returns an error if any index is out of range or two coincide.
    pub fn apply_ccx(&mut self, a: usize, b: usize, target: usize) -> Result<(), GeomError> {
        if a >= self.n || b >= self.n || target >= self.n {
            return Err(GeomError::InvalidArgument("the qubit index is out of range"));
        }
        if a == b || a == target || b == target {
            return Err(GeomError::InvalidArgument("the Toffoli qubits must be distinct"));
        }
        let controls = (1usize << a) | (1usize << b);
        let mask = 1usize << target;
        for i in 0..self.len() {
            if i & controls == controls && i & mask == 0 {
                self.amps.swap(i, i | mask);
            }
        }
        Ok(())
    }

    /// Exchanges two qubits.
    ///
    /// # Errors
    /// Returns an error if an index is out of range.
    pub fn apply_swap(&mut self, a: usize, b: usize) -> Result<(), GeomError> {
        if a >= self.n || b >= self.n {
            return Err(GeomError::InvalidArgument("the qubit index is out of range"));
        }
        if a == b {
            return Ok(());
        }
        let (ma, mb) = (1usize << a, 1usize << b);
        for i in 0..self.len() {
            if i & ma != 0 && i & mb == 0 {
                self.amps.swap(i, (i & !ma) | mb);
            }
        }
        Ok(())
    }

    /// The reduced density matrix over the kept qubits, tracing out the rest.
    ///
    /// # Errors
    /// Returns an error for a repeated or out-of-range index, or an empty
    /// selection.
    pub fn reduced_density_matrix(&self, keep: &[usize]) -> Result<Vec<Vec<Complex>>, GeomError> {
        if keep.is_empty() || keep.len() > self.n {
            return Err(GeomError::InvalidArgument("the kept set is the wrong size"));
        }
        let mut seen = vec![false; self.n];
        for &q in keep {
            if q >= self.n || seen[q] {
                return Err(GeomError::InvalidArgument("the kept qubits must be distinct"));
            }
            seen[q] = true;
        }
        let traced: Vec<usize> = (0..self.n).filter(|q| !seen[*q]).collect();
        let kept_size = 1usize << keep.len();
        let traced_size = 1usize << traced.len();

        let assemble = |kept_index: usize, traced_index: usize| -> usize {
            let mut full = 0usize;
            for (bit, &q) in keep.iter().enumerate() {
                if kept_index >> bit & 1 == 1 {
                    full |= 1 << q;
                }
            }
            for (bit, &q) in traced.iter().enumerate() {
                if traced_index >> bit & 1 == 1 {
                    full |= 1 << q;
                }
            }
            full
        };

        let mut rho = vec![vec![ZERO; kept_size]; kept_size];
        for t in 0..traced_size {
            for r in 0..kept_size {
                for c in 0..kept_size {
                    let a = self.amps[assemble(r, t)];
                    let b = self.amps[assemble(c, t)];
                    rho[r][c] = rho[r][c] + a * b.conjugate();
                }
            }
        }
        Ok(rho)
    }

    /// The Schmidt coefficients across a bipartition: the square roots of the
    /// reduced density matrix's eigenvalues, descending.
    ///
    /// # Errors
    /// Returns an error for a bad partition or an eigensolver failure.
    pub fn schmidt_coefficients(&self, partition: &[usize]) -> Result<Vec<f64>, GeomError> {
        let rho = self.reduced_density_matrix(partition)?;
        let mut values = hermitian_eigenvalues(&rho)?;
        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Ok(values.iter().map(|v| v.max(0.0).sqrt()).collect())
    }

    /// The entanglement entropy across a bipartition, in bits.
    ///
    /// Zero exactly when the state factorises across the cut, and maximal --
    /// one bit per qubit of the smaller side -- for a maximally entangled
    /// state. It is symmetric between the two sides, which is not obvious and
    /// is the reason it is a property of the *cut* rather than of either
    /// piece.
    ///
    /// # Errors
    /// Returns an error for a bad partition.
    pub fn entanglement_entropy(&self, partition: &[usize]) -> Result<f64, GeomError> {
        let rho = self.reduced_density_matrix(partition)?;
        let values = hermitian_eigenvalues(&rho)?;
        Ok(values
            .iter()
            .filter(|v| **v > 1e-12)
            .map(|v| -v * v.log2())
            .sum())
    }

    /// The Bloch vector of one qubit, as `(x, y, z)`.
    ///
    /// Its length is one exactly when that qubit is in a pure state, so it
    /// shortens as the qubit becomes entangled with the others -- the
    /// geometric statement of monogamy.
    ///
    /// # Errors
    /// Returns an error if the qubit index is out of range.
    pub fn bloch_vector(&self, q: usize) -> Result<(f64, f64, f64), GeomError> {
        let rho = self.reduced_density_matrix(&[q])?;
        Ok((
            2.0 * rho[0][1].re,
            -2.0 * rho[0][1].im,
            rho[0][0].re - rho[1][1].re,
        ))
    }
}

/// The eigenvalues of a small Hermitian matrix, via the real symmetric
/// embedding `[[Re, -Im], [Im, Re]]`.
///
/// That embedding doubles every eigenvalue, so each appears twice and the
/// duplicates are dropped. It is the standard way to reach a complex
/// Hermitian spectrum with a real symmetric solver.
fn hermitian_eigenvalues(m: &[Vec<Complex>]) -> Result<Vec<f64>, GeomError> {
    let n = m.len();
    if n == 0 || m.iter().any(|row| row.len() != n) {
        return Err(GeomError::InvalidArgument("the matrix is not square"));
    }
    let mut embedded = crate::linalg::matrix::Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            embedded.set(i, j, m[i][j].re);
            embedded.set(i + n, j + n, m[i][j].re);
            embedded.set(i, j + n, -m[i][j].im);
            embedded.set(i + n, j, m[i][j].im);
        }
    }
    let decomposition = crate::linalg::eigen::eigen_symmetric(&embedded, 1e-13, 200)
        .map_err(|_| GeomError::Degenerate("the density matrix eigenproblem failed"))?;
    // Descending, so every pair sits together; take one of each.
    Ok(decomposition.values.iter().step_by(2).copied().collect())
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

/// A one-qubit gate: a two-by-two unitary.
#[derive(Debug, Clone, Copy)]
pub struct Gate {
    /// The matrix, row major.
    pub matrix: [[Complex; 2]; 2],
}

impl Gate {
    /// Builds a gate from a matrix, checking unitarity.
    ///
    /// # Errors
    /// Returns an error unless the matrix is unitary to tolerance. The check
    /// is worth having: a gate that is merely close to unitary leaks or
    /// creates probability at every application, and the drift is invisible
    /// until the norm has moved far enough to notice.
    pub fn from_matrix(matrix: [[Complex; 2]; 2]) -> Result<Self, GeomError> {
        let gate = Self { matrix };
        if !gate.is_unitary(QUANTUM_TOL) {
            return Err(GeomError::InvalidArgument("the gate matrix is not unitary"));
        }
        Ok(gate)
    }

    /// Whether `U^dagger U` is the identity to the given tolerance.
    #[must_use]
    pub fn is_unitary(&self, tol: f64) -> bool {
        for i in 0..2 {
            for j in 0..2 {
                let entry = (0..2)
                    .fold(ZERO, |acc, k| acc + self.matrix[k][i].conjugate() * self.matrix[k][j]);
                let expected = if i == j { 1.0 } else { 0.0 };
                if (entry.re - expected).abs() > tol || entry.im.abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// The adjoint, which is also the inverse.
    #[must_use]
    pub fn dagger(&self) -> Self {
        Self {
            matrix: [
                [self.matrix[0][0].conjugate(), self.matrix[1][0].conjugate()],
                [self.matrix[0][1].conjugate(), self.matrix[1][1].conjugate()],
            ],
        }
    }

    /// The identity.
    #[must_use]
    pub fn identity() -> Self {
        Self { matrix: [[ONE, ZERO], [ZERO, ONE]] }
    }
    /// The Pauli X, or bit flip.
    #[must_use]
    pub fn x() -> Self {
        Self { matrix: [[ZERO, ONE], [ONE, ZERO]] }
    }
    /// The Pauli Y.
    #[must_use]
    pub fn y() -> Self {
        Self {
            matrix: [
                [ZERO, Complex::new(0.0, -1.0)],
                [Complex::new(0.0, 1.0), ZERO],
            ],
        }
    }
    /// The Pauli Z, or phase flip.
    #[must_use]
    pub fn z() -> Self {
        Self { matrix: [[ONE, ZERO], [ZERO, Complex::new(-1.0, 0.0)]] }
    }
    /// The Hadamard.
    #[must_use]
    pub fn h() -> Self {
        let a = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        Self { matrix: [[a, a], [a, scale(a, -1.0)]] }
    }
    /// The phase gate `S`.
    #[must_use]
    pub fn s() -> Self {
        Self { matrix: [[ONE, ZERO], [ZERO, Complex::new(0.0, 1.0)]] }
    }
    /// The inverse of `S`.
    #[must_use]
    pub fn sdg() -> Self {
        Self { matrix: [[ONE, ZERO], [ZERO, Complex::new(0.0, -1.0)]] }
    }
    /// The `T` gate, an eighth turn about `Z`.
    #[must_use]
    pub fn t() -> Self {
        Self { matrix: [[ONE, ZERO], [ZERO, cis(std::f64::consts::FRAC_PI_4)]] }
    }
    /// The inverse of `T`.
    #[must_use]
    pub fn tdg() -> Self {
        Self { matrix: [[ONE, ZERO], [ZERO, cis(-std::f64::consts::FRAC_PI_4)]] }
    }
    /// A rotation about `X`.
    #[must_use]
    pub fn rx(theta: f64) -> Self {
        let c = Complex::new((theta / 2.0).cos(), 0.0);
        let s = Complex::new(0.0, -(theta / 2.0).sin());
        Self { matrix: [[c, s], [s, c]] }
    }
    /// A rotation about `Y`.
    #[must_use]
    pub fn ry(theta: f64) -> Self {
        let c = Complex::new((theta / 2.0).cos(), 0.0);
        let s = Complex::new((theta / 2.0).sin(), 0.0);
        Self { matrix: [[c, scale(s, -1.0)], [s, c]] }
    }
    /// A rotation about `Z`.
    #[must_use]
    pub fn rz(theta: f64) -> Self {
        Self { matrix: [[cis(-theta / 2.0), ZERO], [ZERO, cis(theta / 2.0)]] }
    }
    /// A relative phase on the one state.
    #[must_use]
    pub fn phase(phi: f64) -> Self {
        Self { matrix: [[ONE, ZERO], [ZERO, cis(phi)]] }
    }
    /// The general one-qubit gate.
    ///
    /// Every one-qubit unitary is this up to a global phase, which is the
    /// content of the Euler decomposition: three real parameters, because the
    /// group is three dimensional once the phase is quotiented out.
    #[must_use]
    pub fn u3(theta: f64, phi: f64, lambda: f64) -> Self {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        Self {
            matrix: [
                [Complex::new(c, 0.0), scale(cis(lambda), -s)],
                [scale(cis(phi), s), scale(cis(phi + lambda), c)],
            ],
        }
    }
    /// The square root of `X`.
    #[must_use]
    pub fn sqrt_x() -> Self {
        let half = Complex::new(0.5, 0.5);
        let other = Complex::new(0.5, -0.5);
        Self { matrix: [[half, other], [other, half]] }
    }
}

// ---------------------------------------------------------------------------
// Circuits
// ---------------------------------------------------------------------------

/// One instruction in a circuit.
#[derive(Debug, Clone)]
pub enum Op {
    /// A one-qubit gate on the given wire.
    Single(usize, Gate),
    /// A controlled one-qubit gate.
    Controlled(usize, usize, Gate),
    /// A Toffoli.
    CCX(usize, usize, usize),
    /// A swap.
    Swap(usize, usize),
    /// A visual separator with no effect.
    Barrier,
}

/// A sequence of operations on a fixed number of qubits.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// The number of qubits.
    pub n: usize,
    /// The operations, in order.
    pub ops: Vec<Op>,
}

impl Circuit {
    /// An empty circuit.
    ///
    /// # Errors
    /// Returns an error for a bad qubit count.
    pub fn new(n: usize) -> Result<Self, GeomError> {
        if n == 0 || n > MAX_QUBITS {
            return Err(GeomError::InvalidArgument("the qubit count is out of range"));
        }
        Ok(Self { n, ops: Vec::new() })
    }

    /// Appends a one-qubit gate.
    pub fn gate(&mut self, q: usize, gate: Gate) -> &mut Self {
        self.ops.push(Op::Single(q, gate));
        self
    }
    /// Appends an X.
    pub fn x(&mut self, q: usize) -> &mut Self {
        self.gate(q, Gate::x())
    }
    /// Appends a Y.
    pub fn y(&mut self, q: usize) -> &mut Self {
        self.gate(q, Gate::y())
    }
    /// Appends a Z.
    pub fn z(&mut self, q: usize) -> &mut Self {
        self.gate(q, Gate::z())
    }
    /// Appends a Hadamard.
    pub fn h(&mut self, q: usize) -> &mut Self {
        self.gate(q, Gate::h())
    }
    /// Appends an X rotation.
    pub fn rx(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(q, Gate::rx(theta))
    }
    /// Appends a Y rotation.
    pub fn ry(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(q, Gate::ry(theta))
    }
    /// Appends a Z rotation.
    pub fn rz(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(q, Gate::rz(theta))
    }
    /// Appends a phase.
    pub fn phase(&mut self, q: usize, phi: f64) -> &mut Self {
        self.gate(q, Gate::phase(phi))
    }
    /// Appends a controlled NOT.
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.ops.push(Op::Controlled(control, target, Gate::x()));
        self
    }
    /// Appends a controlled Z.
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.ops.push(Op::Controlled(control, target, Gate::z()));
        self
    }
    /// Appends a controlled phase.
    pub fn cphase(&mut self, control: usize, target: usize, phi: f64) -> &mut Self {
        self.ops.push(Op::Controlled(control, target, Gate::phase(phi)));
        self
    }
    /// Appends a Toffoli.
    pub fn ccx(&mut self, a: usize, b: usize, target: usize) -> &mut Self {
        self.ops.push(Op::CCX(a, b, target));
        self
    }
    /// Appends a swap.
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.ops.push(Op::Swap(a, b));
        self
    }
    /// Appends a barrier.
    pub fn barrier(&mut self) -> &mut Self {
        self.ops.push(Op::Barrier);
        self
    }

    /// Appends another circuit's operations.
    ///
    /// # Errors
    /// Returns an error if the widths disagree.
    pub fn append(&mut self, other: &Self) -> Result<&mut Self, GeomError> {
        if other.n != self.n {
            return Err(GeomError::InvalidArgument("the circuits have different widths"));
        }
        self.ops.extend(other.ops.iter().cloned());
        Ok(self)
    }

    /// The inverse circuit: every gate adjointed, in reverse order.
    ///
    /// Reversing without adjointing, or adjointing without reversing, is the
    /// classic error and gives the identity only for circuits of self-inverse
    /// gates -- which is most textbook examples, so it survives casual
    /// testing.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let ops = self
            .ops
            .iter()
            .rev()
            .map(|op| match op {
                Op::Single(q, g) => Op::Single(*q, g.dagger()),
                Op::Controlled(c, t, g) => Op::Controlled(*c, *t, g.dagger()),
                Op::CCX(a, b, t) => Op::CCX(*a, *b, *t),
                Op::Swap(a, b) => Op::Swap(*a, *b),
                Op::Barrier => Op::Barrier,
            })
            .collect();
        Self { n: self.n, ops }
    }

    /// The number of gates, ignoring barriers.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.ops.iter().filter(|op| !matches!(op, Op::Barrier)).count()
    }

    /// The circuit depth: the number of layers when gates on disjoint qubits
    /// are packed together.
    ///
    /// Depth rather than gate count is what sets the runtime on hardware,
    /// because gates on disjoint qubits run at once, and it is what a
    /// coherence time has to be compared against.
    #[must_use]
    pub fn depth(&self) -> usize {
        let mut layer = vec![0usize; self.n];
        for op in &self.ops {
            let touched: Vec<usize> = match op {
                Op::Single(q, _) => vec![*q],
                Op::Controlled(c, t, _) | Op::Swap(c, t) => vec![*c, *t],
                Op::CCX(a, b, t) => vec![*a, *b, *t],
                Op::Barrier => continue,
            };
            let next = touched.iter().map(|&q| layer[q]).max().unwrap_or(0) + 1;
            for &q in &touched {
                layer[q] = next;
            }
        }
        layer.into_iter().max().unwrap_or(0)
    }

    /// Runs the circuit on a state.
    ///
    /// # Errors
    /// Returns an error if the state has the wrong width or an operation
    /// names a bad qubit.
    pub fn run(&self, initial: &QState) -> Result<QState, GeomError> {
        if initial.n != self.n {
            return Err(GeomError::InvalidArgument("the state has the wrong width"));
        }
        let mut state = initial.clone();
        for op in &self.ops {
            match op {
                Op::Single(q, g) => state.apply_single(*q, g)?,
                Op::Controlled(c, t, g) => state.apply_controlled(*c, *t, g)?,
                Op::CCX(a, b, t) => state.apply_ccx(*a, *b, *t)?,
                Op::Swap(a, b) => state.apply_swap(*a, *b)?,
                Op::Barrier => {}
            }
        }
        Ok(state)
    }

    /// Runs from the all-zeros state and samples measurements.
    ///
    /// # Errors
    /// Returns an error if the circuit cannot run.
    pub fn run_shots(&self, shots: usize, rng: &mut Rng) -> Result<Vec<(u64, u64)>, GeomError> {
        let state = self.run(&QState::zero(self.n)?)?;
        Ok(state.sample_counts(shots, rng))
    }

    /// The full unitary, for small circuits.
    ///
    /// Costs `4^n` amplitudes, so it is capped at ten qubits. It is built by
    /// running the circuit on each basis state in turn, which makes each
    /// column the image of one basis vector -- the definition of the matrix.
    ///
    /// # Errors
    /// Returns an error above ten qubits, or if the circuit cannot run.
    pub fn unitary_small(&self) -> Result<Vec<Vec<Complex>>, GeomError> {
        if self.n > 10 {
            return Err(GeomError::InvalidArgument("unitary_small is capped at ten qubits"));
        }
        let size = 1usize << self.n;
        let mut columns = vec![vec![ZERO; size]; size];
        for column in 0..size {
            let out = self.run(&QState::basis(self.n, column as u64)?)?;
            for (row, z) in out.amps.iter().enumerate() {
                columns[row][column] = *z;
            }
        }
        Ok(columns)
    }

    /// A compact textual form, one line per operation.
    #[must_use]
    pub fn to_qasm_lite(&self) -> String {
        let mut out = format!("qubits {}\n", self.n);
        for op in &self.ops {
            match op {
                Op::Single(q, g) => out.push_str(&format!("u {} {}\n", q, gate_name(g))),
                Op::Controlled(c, t, g) => {
                    out.push_str(&format!("c{} {} {}\n", gate_name(g), c, t));
                }
                Op::CCX(a, b, t) => out.push_str(&format!("ccx {a} {b} {t}\n")),
                Op::Swap(a, b) => out.push_str(&format!("swap {a} {b}\n")),
                Op::Barrier => out.push_str("barrier\n"),
            }
        }
        out
    }

    /// An ASCII diagram, one row per qubit.
    #[must_use]
    pub fn draw_ascii(&self) -> String {
        let mut rows: Vec<String> = (0..self.n).map(|q| format!("q{q}: ")).collect();
        for op in &self.ops {
            let labels: Vec<(usize, String)> = match op {
                Op::Single(q, g) => vec![(*q, format!("-{}-", gate_name(g)))],
                Op::Controlled(c, t, g) => {
                    vec![(*c, "-*-".into()), (*t, format!("-{}-", gate_name(g)))]
                }
                Op::CCX(a, b, t) => {
                    vec![(*a, "-*-".into()), (*b, "-*-".into()), (*t, "-X-".into())]
                }
                Op::Swap(a, b) => vec![(*a, "-x-".into()), (*b, "-x-".into())],
                Op::Barrier => (0..self.n).map(|q| (q, "-|-".into())).collect(),
            };
            let width = labels.iter().map(|(_, s)| s.len()).max().unwrap_or(3);
            for q in 0..self.n {
                let piece = labels
                    .iter()
                    .find(|(target, _)| *target == q)
                    .map_or_else(|| "-".repeat(width), |(_, s)| s.clone());
                rows[q].push_str(&format!("{piece:-<width$}"));
            }
        }
        rows.join("\n")
    }
}

fn gate_name(g: &Gate) -> String {
    for (name, candidate) in [
        ("I", Gate::identity()),
        ("X", Gate::x()),
        ("Y", Gate::y()),
        ("Z", Gate::z()),
        ("H", Gate::h()),
        ("S", Gate::s()),
        ("SD", Gate::sdg()),
        ("T", Gate::t()),
        ("TD", Gate::tdg()),
        ("SX", Gate::sqrt_x()),
    ] {
        let same = (0..2).all(|i| {
            (0..2).all(|j| {
                (g.matrix[i][j].re - candidate.matrix[i][j].re).abs() < 1e-12
                    && (g.matrix[i][j].im - candidate.matrix[i][j].im).abs() < 1e-12
            })
        });
        if same {
            return name.into();
        }
    }
    "U".into()
}

// ---------------------------------------------------------------------------
// Density matrices and noise
// ---------------------------------------------------------------------------

/// A mixed state of `n` qubits.
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    /// The number of qubits.
    pub n: usize,
    /// The matrix, row major.
    pub rho: Vec<Vec<Complex>>,
}

impl DensityMatrix {
    /// The density matrix of a pure state.
    #[must_use]
    pub fn from_state(state: &QState) -> Self {
        let size = state.len();
        let mut rho = vec![vec![ZERO; size]; size];
        for i in 0..size {
            for j in 0..size {
                rho[i][j] = state.amps[i] * state.amps[j].conjugate();
            }
        }
        Self { n: state.n, rho }
    }

    /// A classical mixture of states.
    ///
    /// The distinction from a superposition is the whole of the difference
    /// between quantum and classical uncertainty: a mixture of `|0>` and
    /// `|1>` is diagonal and behaves like a coin, while their superposition
    /// has off-diagonal terms and interferes.
    ///
    /// # Errors
    /// Returns an error for mismatched lengths, differing widths, negative
    /// weights, or weights that do not sum to one.
    pub fn from_mixture(states: &[QState], weights: &[f64]) -> Result<Self, GeomError> {
        if states.is_empty() || states.len() != weights.len() {
            return Err(GeomError::InvalidArgument("from_mixture: mismatched input"));
        }
        if states.iter().any(|s| s.n != states[0].n) {
            return Err(GeomError::InvalidArgument("the states have different widths"));
        }
        if weights.iter().any(|w| *w < 0.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > QUANTUM_TOL
        {
            return Err(GeomError::InvalidArgument("the weights must be a distribution"));
        }
        let size = states[0].len();
        let mut rho = vec![vec![ZERO; size]; size];
        for (state, &w) in states.iter().zip(weights) {
            for i in 0..size {
                for j in 0..size {
                    rho[i][j] = rho[i][j] + scale(state.amps[i] * state.amps[j].conjugate(), w);
                }
            }
        }
        Ok(Self { n: states[0].n, rho })
    }

    /// The trace.
    #[must_use]
    pub fn trace(&self) -> Complex {
        (0..self.rho.len()).fold(ZERO, |acc, i| acc + self.rho[i][i])
    }

    /// The purity `tr(rho^2)`: one for a pure state, `1 / d` for the maximally
    /// mixed one.
    #[must_use]
    pub fn purity(&self) -> f64 {
        let size = self.rho.len();
        let mut total = 0.0;
        for i in 0..size {
            for j in 0..size {
                total += (self.rho[i][j] * self.rho[j][i]).re;
            }
        }
        total
    }

    /// The von Neumann entropy in bits.
    ///
    /// # Errors
    /// Returns an error if the eigenproblem fails.
    pub fn von_neumann_entropy(&self) -> Result<f64, GeomError> {
        let values = hermitian_eigenvalues(&self.rho)?;
        Ok(values.iter().filter(|v| **v > 1e-12).map(|v| -v * v.log2()).sum())
    }

    /// Whether the matrix is Hermitian, unit trace, and positive
    /// semi-definite -- the three conditions that make it a state.
    #[must_use]
    pub fn is_valid(&self, tol: f64) -> bool {
        let size = self.rho.len();
        let trace = self.trace();
        if (trace.re - 1.0).abs() > tol || trace.im.abs() > tol {
            return false;
        }
        for i in 0..size {
            for j in 0..size {
                let a = self.rho[i][j];
                let b = self.rho[j][i].conjugate();
                if (a.re - b.re).abs() > tol || (a.im - b.im).abs() > tol {
                    return false;
                }
            }
        }
        hermitian_eigenvalues(&self.rho)
            .map(|values| values.iter().all(|v| *v > -tol))
            .unwrap_or(false)
    }

    /// Applies a one-qubit gate by conjugation.
    ///
    /// # Errors
    /// Returns an error if the qubit index is out of range.
    pub fn apply_gate(&mut self, q: usize, gate: &Gate) -> Result<(), GeomError> {
        if q >= self.n {
            return Err(GeomError::InvalidArgument("the qubit index is out of range"));
        }
        let full = lift_single(self.n, q, gate);
        self.rho = conjugate(&full, &self.rho);
        Ok(())
    }

    /// Applies a quantum channel given by its Kraus operators.
    ///
    /// The Kraus form is what makes noise tractable: any physical evolution of
    /// an open system, however complicated the environment, is
    /// `sum_k K_k rho K_k^dagger` for some finite set of operators satisfying
    /// `sum_k K_k^dagger K_k = I`. That completeness condition is exactly
    /// trace preservation, which is why a channel cannot lose probability.
    ///
    /// # Errors
    /// Returns an error for the wrong operator size or a set that is not
    /// trace preserving.
    pub fn apply_channel(&mut self, kraus: &[Vec<Vec<Complex>>]) -> Result<(), GeomError> {
        let size = self.rho.len();
        if kraus.is_empty() || kraus.iter().any(|k| k.len() != size || k.iter().any(|r| r.len() != size)) {
            return Err(GeomError::InvalidArgument("the Kraus operators are the wrong size"));
        }
        if !is_trace_preserving(kraus, QUANTUM_TOL) {
            return Err(GeomError::InvalidArgument("the channel is not trace preserving"));
        }
        let mut out = vec![vec![ZERO; size]; size];
        for k in kraus {
            let piece = conjugate(k, &self.rho);
            for i in 0..size {
                for j in 0..size {
                    out[i][j] = out[i][j] + piece[i][j];
                }
            }
        }
        self.rho = out;
        Ok(())
    }

    /// Traces out every qubit but the kept ones.
    ///
    /// # Errors
    /// Returns an error for a repeated or out-of-range index.
    pub fn partial_trace(&self, keep: &[usize]) -> Result<Self, GeomError> {
        if keep.is_empty() || keep.len() > self.n {
            return Err(GeomError::InvalidArgument("the kept set is the wrong size"));
        }
        let mut seen = vec![false; self.n];
        for &q in keep {
            if q >= self.n || seen[q] {
                return Err(GeomError::InvalidArgument("the kept qubits must be distinct"));
            }
            seen[q] = true;
        }
        let traced: Vec<usize> = (0..self.n).filter(|q| !seen[*q]).collect();
        let kept_size = 1usize << keep.len();
        let traced_size = 1usize << traced.len();
        let assemble = |kept_index: usize, traced_index: usize| -> usize {
            let mut full = 0usize;
            for (bit, &q) in keep.iter().enumerate() {
                if kept_index >> bit & 1 == 1 {
                    full |= 1 << q;
                }
            }
            for (bit, &q) in traced.iter().enumerate() {
                if traced_index >> bit & 1 == 1 {
                    full |= 1 << q;
                }
            }
            full
        };
        let mut out = vec![vec![ZERO; kept_size]; kept_size];
        for t in 0..traced_size {
            for r in 0..kept_size {
                for c in 0..kept_size {
                    out[r][c] = out[r][c] + self.rho[assemble(r, t)][assemble(c, t)];
                }
            }
        }
        Ok(Self { n: keep.len(), rho: out })
    }
}

/// Whether a set of Kraus operators sums to the identity under
/// `sum_k K^dagger K`.
fn is_trace_preserving(kraus: &[Vec<Vec<Complex>>], tol: f64) -> bool {
    let size = kraus[0].len();
    let mut total = vec![vec![ZERO; size]; size];
    for k in kraus {
        for i in 0..size {
            for j in 0..size {
                let entry = (0..size).fold(ZERO, |acc, r| acc + k[r][i].conjugate() * k[r][j]);
                total[i][j] = total[i][j] + entry;
            }
        }
    }
    for i in 0..size {
        for j in 0..size {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (total[i][j].re - expected).abs() > tol || total[i][j].im.abs() > tol {
                return false;
            }
        }
    }
    true
}

fn conjugate(m: &[Vec<Complex>], rho: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
    let size = rho.len();
    let mut left = vec![vec![ZERO; size]; size];
    for i in 0..size {
        for j in 0..size {
            left[i][j] = (0..size).fold(ZERO, |acc, k| acc + m[i][k] * rho[k][j]);
        }
    }
    let mut out = vec![vec![ZERO; size]; size];
    for i in 0..size {
        for j in 0..size {
            out[i][j] = (0..size).fold(ZERO, |acc, k| acc + left[i][k] * m[j][k].conjugate());
        }
    }
    out
}

/// Embeds a one-qubit gate into the full `2^n` space.
fn lift_single(n: usize, q: usize, gate: &Gate) -> Vec<Vec<Complex>> {
    let size = 1usize << n;
    let mask = 1usize << q;
    let mut out = vec![vec![ZERO; size]; size];
    for i in 0..size {
        for j in 0..size {
            if i & !mask != j & !mask {
                continue;
            }
            let row = usize::from(i & mask != 0);
            let column = usize::from(j & mask != 0);
            out[i][j] = gate.matrix[row][column];
        }
    }
    out
}

fn from_rows(rows: [[Complex; 2]; 2]) -> Vec<Vec<Complex>> {
    vec![rows[0].to_vec(), rows[1].to_vec()]
}

/// The depolarising channel: with probability `p`, replace the qubit by the
/// maximally mixed state.
///
/// The one channel that treats every direction alike, so it shrinks the Bloch
/// vector uniformly toward the origin without rotating it.
///
/// # Errors
/// Returns an error unless `p` is a probability.
pub fn depolarizing_channel(p: f64) -> Result<Vec<Vec<Vec<Complex>>>, GeomError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(GeomError::InvalidArgument("the error rate must be a probability"));
    }
    let keep = (1.0 - 3.0 * p / 4.0).max(0.0).sqrt();
    let each = (p / 4.0).sqrt();
    Ok(vec![
        from_rows([[scale(ONE, keep), ZERO], [ZERO, scale(ONE, keep)]]),
        from_rows([[ZERO, scale(ONE, each)], [scale(ONE, each), ZERO]]),
        from_rows([
            [ZERO, Complex::new(0.0, -each)],
            [Complex::new(0.0, each), ZERO],
        ]),
        from_rows([[scale(ONE, each), ZERO], [ZERO, scale(ONE, -each)]]),
    ])
}

/// Amplitude damping: a qubit decaying from `|1>` to `|0>` with probability
/// `gamma`.
///
/// Models spontaneous emission, and unlike the symmetric channels it has a
/// fixed point that is not the maximally mixed state: everything ends up in
/// the ground state. That asymmetry is why `T_1` and `T_2` are different
/// numbers.
///
/// # Errors
/// Returns an error unless `gamma` is a probability.
pub fn amplitude_damping(gamma: f64) -> Result<Vec<Vec<Vec<Complex>>>, GeomError> {
    if !(0.0..=1.0).contains(&gamma) {
        return Err(GeomError::InvalidArgument("gamma must be a probability"));
    }
    Ok(vec![
        from_rows([[ONE, ZERO], [ZERO, scale(ONE, (1.0 - gamma).sqrt())]]),
        from_rows([[ZERO, scale(ONE, gamma.sqrt())], [ZERO, ZERO]]),
    ])
}

/// Phase damping: coherence lost without any energy exchange.
///
/// The off-diagonal terms shrink and the populations do not move at all, so
/// the Bloch vector flattens onto the `z` axis. It is the purely quantum kind
/// of noise -- there is no classical process it corresponds to.
///
/// # Errors
/// Returns an error unless `gamma` is a probability.
pub fn phase_damping(gamma: f64) -> Result<Vec<Vec<Vec<Complex>>>, GeomError> {
    if !(0.0..=1.0).contains(&gamma) {
        return Err(GeomError::InvalidArgument("gamma must be a probability"));
    }
    Ok(vec![
        from_rows([[ONE, ZERO], [ZERO, scale(ONE, (1.0 - gamma).sqrt())]]),
        from_rows([[ZERO, ZERO], [ZERO, scale(ONE, gamma.sqrt())]]),
    ])
}

/// The bit-flip channel.
///
/// # Errors
/// Returns an error unless `p` is a probability.
pub fn bit_flip(p: f64) -> Result<Vec<Vec<Vec<Complex>>>, GeomError> {
    pauli_channel(p, Gate::x())
}

/// The phase-flip channel.
///
/// # Errors
/// Returns an error unless `p` is a probability.
pub fn phase_flip(p: f64) -> Result<Vec<Vec<Vec<Complex>>>, GeomError> {
    pauli_channel(p, Gate::z())
}

fn pauli_channel(p: f64, gate: Gate) -> Result<Vec<Vec<Vec<Complex>>>, GeomError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(GeomError::InvalidArgument("the error rate must be a probability"));
    }
    let keep = (1.0 - p).sqrt();
    let flip = p.sqrt();
    Ok(vec![
        from_rows([[scale(ONE, keep), ZERO], [ZERO, scale(ONE, keep)]]),
        from_rows([
            [scale(gate.matrix[0][0], flip), scale(gate.matrix[0][1], flip)],
            [scale(gate.matrix[1][0], flip), scale(gate.matrix[1][1], flip)],
        ]),
    ])
}

// ---------------------------------------------------------------------------
// Standard states and demonstrations
// ---------------------------------------------------------------------------

/// One of the four Bell states, indexed zero to three.
///
/// # Errors
/// Returns an error for an index above three.
pub fn bell_state(which: u8) -> Result<QState, GeomError> {
    if which > 3 {
        return Err(GeomError::InvalidArgument("there are four Bell states"));
    }
    let mut circuit = Circuit::new(2)?;
    if which & 2 != 0 {
        circuit.x(1);
    }
    if which & 1 != 0 {
        circuit.x(0);
    }
    circuit.h(1).cx(1, 0);
    circuit.run(&QState::zero(2)?)
}

/// The `n`-qubit GHZ state.
///
/// Maximally entangled and maximally fragile: losing one qubit leaves the
/// rest in a classical mixture with no entanglement at all, which is what
/// distinguishes it from the W state.
///
/// # Errors
/// Returns an error for fewer than two qubits or more than the cap.
pub fn ghz(n: usize) -> Result<QState, GeomError> {
    if n < 2 {
        return Err(GeomError::InvalidArgument("a GHZ state needs at least two qubits"));
    }
    let mut circuit = Circuit::new(n)?;
    circuit.h(0);
    for q in 1..n {
        circuit.cx(0, q);
    }
    circuit.run(&QState::zero(n)?)
}

/// The `n`-qubit W state: one excitation shared equally.
///
/// The complement of GHZ. Losing a qubit leaves the others still entangled,
/// so the two are inequivalent under local operations -- there is no way to
/// turn one into the other without communication, even probabilistically.
///
/// # Errors
/// Returns an error for fewer than two qubits or more than the cap.
pub fn w_state(n: usize) -> Result<QState, GeomError> {
    if !(2..=MAX_QUBITS).contains(&n) {
        return Err(GeomError::InvalidArgument("a W state needs two qubits or more"));
    }
    let amplitude = 1.0 / (n as f64).sqrt();
    let mut amps = vec![ZERO; 1usize << n];
    for q in 0..n {
        amps[1usize << q] = Complex::new(amplitude, 0.0);
    }
    Ok(QState { n, amps })
}

/// A Haar-random pure state.
///
/// Built from independent complex Gaussians, which is the standard trick:
/// normalising a Gaussian vector gives the uniform measure on the sphere, so
/// this really is Haar random and not merely "random looking".
///
/// # Errors
/// Returns an error for a bad qubit count.
pub fn random_state(n: usize, rng: &mut Rng) -> Result<QState, GeomError> {
    if n == 0 || n > MAX_QUBITS {
        return Err(GeomError::InvalidArgument("the qubit count is out of range"));
    }
    let size = 1usize << n;
    let mut amps = Vec::with_capacity(size);
    for _ in 0..size {
        // Box-Muller, for a pair of standard normals.
        let u1 = rng.next_f64().max(1e-300);
        let u2 = rng.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        amps.push(Complex::new(radius * angle.cos(), radius * angle.sin()));
    }
    QState::from_amps(amps)
}

/// The CHSH correlation for a two-qubit state at four measurement angles.
///
/// `S = E(a, b) - E(a, b') + E(a', b) + E(a', b')`. Any local hidden variable
/// model obeys `|S| <= 2`; quantum mechanics reaches `2 sqrt 2` on a Bell
/// state, and no theory obeying no-signalling can exceed `4`. The gap between
/// two and `2 sqrt 2` is the whole experimental content of Bell's theorem.
///
/// # Errors
/// Returns an error unless the state has two qubits.
pub fn chsh_value(state: &QState, angles: (f64, f64, f64, f64)) -> Result<f64, GeomError> {
    if state.n != 2 {
        return Err(GeomError::InvalidArgument("CHSH is a two-qubit quantity"));
    }
    let (a, a_prime, b, b_prime) = angles;
    // The correlation of spin measurements along two axes in the x-z plane.
    let correlate = |theta_a: f64, theta_b: f64| -> Result<f64, GeomError> {
        let mut rotated = state.clone();
        rotated.apply_single(1, &Gate::ry(-theta_a))?;
        rotated.apply_single(0, &Gate::ry(-theta_b))?;
        rotated.expectation_pauli_string("ZZ")
    };
    Ok(correlate(a, b)? - correlate(a, b_prime)? + correlate(a_prime, b)?
        + correlate(a_prime, b_prime)?)
}

/// The angles that maximise CHSH on a Bell state, as
/// `(a, a', b, b')` in radians.
#[must_use]
pub fn chsh_optimal_angles() -> (f64, f64, f64, f64) {
    let q = std::f64::consts::FRAC_PI_4;
    (0.0, 2.0 * q, q, 3.0 * q)
}

/// Teleports a one-qubit state and returns the input and output Bloch
/// vectors.
///
/// The protocol consumes one Bell pair and two classical bits, and it moves
/// the state exactly -- not a copy, since the sender's qubit is destroyed by
/// the measurement, which is what keeps no-cloning intact. Without the
/// classical bits the receiver holds the maximally mixed state, so nothing
/// travels faster than light either.
///
/// # Errors
/// Returns an error if the simulation fails.
pub fn quantum_teleportation_demo(
    theta: f64,
    phi: f64,
    rng: &mut Rng,
) -> Result<((f64, f64, f64), (f64, f64, f64)), GeomError> {
    // Qubit 2 carries the message, qubits 1 and 0 the entangled pair.
    let mut state = QState::zero(3)?;
    state.apply_single(2, &Gate::u3(theta, phi, 0.0))?;
    let input = state.bloch_vector(2)?;

    state.apply_single(1, &Gate::h())?;
    state.apply_controlled(1, 0, &Gate::x())?;
    // Bell measurement on the message and the sender's half.
    state.apply_controlled(2, 1, &Gate::x())?;
    state.apply_single(2, &Gate::h())?;
    let (bit1, state) = state.measure_qubit(1, rng)?;
    let (bit2, mut state) = state.measure_qubit(2, rng)?;
    // The classical correction.
    if bit1 {
        state.apply_single(0, &Gate::x())?;
    }
    if bit2 {
        state.apply_single(0, &Gate::z())?;
    }
    let output = state.bloch_vector(0)?;
    Ok((input, output))
}

/// Superdense coding: two classical bits carried by one qubit, given a
/// shared Bell pair.
///
/// Returns the decoded bits, which must equal the encoded ones. The
/// bookkeeping is exact -- one qubit plus prior entanglement carries two
/// bits, and without the entanglement it carries one, which is Holevo's
/// bound.
///
/// # Errors
/// Returns an error if the simulation fails.
pub fn superdense_coding_demo(bits: (bool, bool)) -> Result<(bool, bool), GeomError> {
    let mut state = bell_state(0)?;
    // The sender acts only on their own qubit, number one.
    if bits.1 {
        state.apply_single(1, &Gate::x())?;
    }
    if bits.0 {
        state.apply_single(1, &Gate::z())?;
    }
    // The receiver undoes the entangling circuit and reads both qubits.
    state.apply_controlled(1, 0, &Gate::x())?;
    state.apply_single(1, &Gate::h())?;
    let outcome = state
        .probabilities()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0);
    Ok((outcome & 2 != 0, outcome & 1 != 0))
}

/// The best fidelity an approximate universal cloner can achieve: `5 / 6`.
///
/// Exact cloning is impossible because it is not linear, and the optimal
/// approximation is bounded by this number, which is a theorem rather than an
/// engineering limit.
#[must_use]
pub fn no_cloning_fidelity_bound() -> f64 {
    5.0 / 6.0
}

/// Decomposes a Hermitian matrix on one or two qubits into Pauli terms.
///
/// The Pauli strings form an orthogonal basis under the Hilbert-Schmidt inner
/// product, so each coefficient is just `tr(P H) / d` -- no linear solve
/// needed. That orthogonality is what makes measuring a Hamiltonian on
/// hardware possible at all.
///
/// # Errors
/// Returns an error unless the matrix is square with side two or four.
pub fn pauli_decompose(h: &[Vec<Complex>]) -> Result<Vec<(String, f64)>, GeomError> {
    let size = h.len();
    if (size != 2 && size != 4) || h.iter().any(|row| row.len() != size) {
        return Err(GeomError::InvalidArgument("pauli_decompose handles one or two qubits"));
    }
    let qubits = size.trailing_zeros() as usize;
    let symbols = ['I', 'X', 'Y', 'Z'];
    let single = |c: char| -> Gate {
        match c {
            'X' => Gate::x(),
            'Y' => Gate::y(),
            'Z' => Gate::z(),
            _ => Gate::identity(),
        }
    };
    let mut out = Vec::new();
    let combinations = 4usize.pow(qubits as u32);
    for code in 0..combinations {
        let name: String = (0..qubits)
            .rev()
            .map(|k| symbols[(code >> (2 * k)) & 3])
            .collect();
        // Build the tensor product and take the trace against h.
        let mut trace = ZERO;
        for i in 0..size {
            for j in 0..size {
                let mut entry = ONE;
                for k in 0..qubits {
                    let gate = single(symbols[(code >> (2 * (qubits - 1 - k))) & 3]);
                    let row = (i >> (qubits - 1 - k)) & 1;
                    let column = (j >> (qubits - 1 - k)) & 1;
                    entry = entry * gate.matrix[row][column];
                }
                trace = trace + entry * h[j][i];
            }
        }
        let coefficient = trace.re / size as f64;
        if coefficient.abs() > 1e-12 {
            out.push((name, coefficient));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn matrix_close(a: &[Vec<Complex>], b: &[Vec<Complex>], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(ra, rb)| {
                ra.len() == rb.len()
                    && ra.iter().zip(rb).all(|(x, y)| {
                        (x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol
                    })
            })
    }

    fn identity_matrix(size: usize) -> Vec<Vec<Complex>> {
        (0..size)
            .map(|i| (0..size).map(|j| if i == j { ONE } else { ZERO }).collect())
            .collect()
    }

    // -----------------------------------------------------------------
    // Gates
    // -----------------------------------------------------------------

    #[test]
    fn every_named_gate_is_unitary_and_its_own_stated_inverse() {
        // Unitarity is checkable directly from the matrix, so it is checked
        // rather than assumed for every gate the module offers.
        let named: Vec<(&str, Gate)> = vec![
            ("I", Gate::identity()),
            ("X", Gate::x()),
            ("Y", Gate::y()),
            ("Z", Gate::z()),
            ("H", Gate::h()),
            ("S", Gate::s()),
            ("Sdg", Gate::sdg()),
            ("T", Gate::t()),
            ("Tdg", Gate::tdg()),
            ("sqrtX", Gate::sqrt_x()),
            ("rx", Gate::rx(0.7)),
            ("ry", Gate::ry(-1.3)),
            ("rz", Gate::rz(2.2)),
            ("phase", Gate::phase(0.4)),
            ("u3", Gate::u3(0.6, 1.1, -0.3)),
        ];
        for (name, gate) in &named {
            assert!(gate.is_unitary(1e-12), "{name} is not unitary");
            // The adjoint undoes it.
            let mut state = QState::from_amps(vec![
                Complex::new(0.6, 0.2),
                Complex::new(-0.3, 0.7),
            ])
            .unwrap();
            let original = state.clone();
            state.apply_single(0, gate).unwrap();
            state.apply_single(0, &gate.dagger()).unwrap();
            for (a, b) in state.amps.iter().zip(&original.amps) {
                assert!(
                    (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                    "{name} followed by its adjoint is not the identity"
                );
            }
        }
        // The named pairs really are inverses of each other.
        for (a, b) in [(Gate::s(), Gate::sdg()), (Gate::t(), Gate::tdg())] {
            for i in 0..2 {
                for j in 0..2 {
                    let entry =
                        (0..2).fold(ZERO, |acc, k| acc + a.matrix[i][k] * b.matrix[k][j]);
                    let expected = f64::from(i == j);
                    assert!(close(entry.re, expected, 1e-12) && close(entry.im, 0.0, 1e-12));
                }
            }
        }
        // Non-unitary matrices are refused.
        assert!(Gate::from_matrix([[ONE, ONE], [ZERO, ONE]]).is_err());
        assert!(Gate::from_matrix([[scale(ONE, 2.0), ZERO], [ZERO, ONE]]).is_err());
        assert!(Gate::from_matrix(Gate::h().matrix).is_ok());
    }

    #[test]
    fn the_pauli_algebra_holds_as_the_gates_are_defined() {
        // X Y = i Z and the cyclic relatives, plus each Pauli squaring to the
        // identity. These are the relations the gates are *for*, and a sign
        // error in Y would pass a unitarity check and fail here.
        let multiply = |a: &Gate, b: &Gate| -> [[Complex; 2]; 2] {
            let mut out = [[ZERO; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    out[i][j] = (0..2).fold(ZERO, |acc, k| acc + a.matrix[i][k] * b.matrix[k][j]);
                }
            }
            out
        };
        let i_times = |g: &Gate| -> [[Complex; 2]; 2] {
            let mut out = [[ZERO; 2]; 2];
            for r in 0..2 {
                for c in 0..2 {
                    out[r][c] = Complex::new(0.0, 1.0) * g.matrix[r][c];
                }
            }
            out
        };
        let same = |a: &[[Complex; 2]; 2], b: &[[Complex; 2]; 2]| -> bool {
            (0..2).all(|i| {
                (0..2).all(|j| {
                    (a[i][j].re - b[i][j].re).abs() < 1e-12
                        && (a[i][j].im - b[i][j].im).abs() < 1e-12
                })
            })
        };
        assert!(same(&multiply(&Gate::x(), &Gate::y()), &i_times(&Gate::z())), "X Y != i Z");
        assert!(same(&multiply(&Gate::y(), &Gate::z()), &i_times(&Gate::x())), "Y Z != i X");
        assert!(same(&multiply(&Gate::z(), &Gate::x()), &i_times(&Gate::y())), "Z X != i Y");
        for g in [Gate::x(), Gate::y(), Gate::z(), Gate::h()] {
            assert!(same(&multiply(&g, &g), &Gate::identity().matrix), "a Pauli did not square to I");
        }
        // S^2 = Z and T^2 = S.
        assert!(same(&multiply(&Gate::s(), &Gate::s()), &Gate::z().matrix));
        assert!(same(&multiply(&Gate::t(), &Gate::t()), &Gate::s().matrix));
        // sqrt(X)^2 = X.
        assert!(same(&multiply(&Gate::sqrt_x(), &Gate::sqrt_x()), &Gate::x().matrix));
        // A rotation by 2 pi is minus the identity, not the identity: the
        // spinor sign that takes 4 pi to undo.
        let full = Gate::rx(2.0 * std::f64::consts::PI);
        assert!(close(full.matrix[0][0].re, -1.0, 1e-12), "rx(2 pi) is {:?}", full.matrix[0][0]);
        let double = Gate::rx(4.0 * std::f64::consts::PI);
        assert!(close(double.matrix[0][0].re, 1.0, 1e-12));
    }

    // -----------------------------------------------------------------
    // States
    // -----------------------------------------------------------------

    #[test]
    fn a_hadamard_on_each_qubit_makes_the_uniform_superposition() {
        for n in 1..=5usize {
            let mut circuit = Circuit::new(n).unwrap();
            for q in 0..n {
                circuit.h(q);
            }
            let state = circuit.run(&QState::zero(n).unwrap()).unwrap();
            let expected = 1.0 / (1usize << n) as f64;
            for p in state.probabilities() {
                assert!(close(p, expected, 1e-12), "an outcome has probability {p}");
            }
            // And it agrees with the direct constructor.
            let direct = QState::plus_all(n).unwrap();
            assert!(close(state.fidelity(&direct).unwrap(), 1.0, 1e-12));

            // Applying it twice returns the start exactly.
            let back = circuit.run(&state).unwrap();
            assert!(close(back.probability(0), 1.0, 1e-12), "H twice is not the identity");
        }
    }

    #[test]
    fn measurement_probabilities_match_the_amplitudes_and_the_collapse_is_consistent() {
        // Sampling is the one place a simulator can be subtly wrong without
        // any state being wrong, so the empirical frequencies are checked
        // against the amplitudes they came from.
        let mut rng = Rng::new(0x_9E11_0001);
        let mut state = QState::zero(2).unwrap();
        state.apply_single(0, &Gate::ry(1.1)).unwrap();
        state.apply_single(1, &Gate::ry(0.4)).unwrap();
        let expected = state.probabilities();

        let shots = 200_000usize;
        let counts = state.sample_counts(shots, &mut rng);
        for (outcome, count) in &counts {
            let observed = *count as f64 / shots as f64;
            let target = expected[*outcome as usize];
            assert!(
                (observed - target).abs() < 4.0 / (shots as f64).sqrt(),
                "outcome {outcome} came up {observed} against {target}"
            );
        }
        assert_eq!(counts.iter().map(|(_, c)| c).sum::<u64>(), shots as u64);

        // Collapsing a qubit and re-measuring it gives the same answer.
        let (bit, collapsed) = state.measure_qubit(0, &mut rng).unwrap();
        assert!(close(collapsed.norm(), 1.0, 1e-12));
        for _ in 0..20 {
            let (again, _) = collapsed.measure_qubit(0, &mut rng).unwrap();
            assert_eq!(again, bit, "a collapsed qubit changed its mind");
        }
        // The Z expectation is exactly plus or minus one afterwards.
        assert!(close(collapsed.expectation_z(0).unwrap().abs(), 1.0, 1e-12));
    }

    #[test]
    fn measuring_one_half_of_a_bell_pair_determines_the_other() {
        // The correlation is perfect and it is not a shared random bit: the
        // CHSH test below shows the same state violating the classical bound.
        let mut rng = Rng::new(0x_9E11_0002);
        for _ in 0..200 {
            let state = bell_state(0).unwrap();
            let (first, collapsed) = state.measure_qubit(0, &mut rng).unwrap();
            let (second, _) = collapsed.measure_qubit(1, &mut rng).unwrap();
            assert_eq!(first, second, "the Bell pair disagreed with itself");
        }
        // Before measurement each qubit alone is maximally mixed, which is
        // what makes the correlation impossible to see locally.
        let state = bell_state(0).unwrap();
        for q in 0..2 {
            let (x, y, z) = state.bloch_vector(q).unwrap();
            assert!(
                x.hypot(y).hypot(z) < 1e-12,
                "qubit {q} has a Bloch vector of length {}",
                x.hypot(y).hypot(z)
            );
        }
    }

    #[test]
    fn entanglement_entropy_separates_the_states_it_is_meant_to() {
        // A product state has none, a Bell state has exactly one bit, and GHZ
        // has one bit across any cut. The W state is the interesting case:
        // it also has entropy across a single-qubit cut, but unlike GHZ it
        // keeps entanglement after a qubit is lost.
        let mut product = QState::zero(2).unwrap();
        product.apply_single(0, &Gate::ry(0.9)).unwrap();
        product.apply_single(1, &Gate::rx(1.4)).unwrap();
        assert!(
            close(product.entanglement_entropy(&[0]).unwrap(), 0.0, 1e-9),
            "a product state has entropy {}",
            product.entanglement_entropy(&[0]).unwrap()
        );

        for which in 0..4u8 {
            let bell = bell_state(which).unwrap();
            assert!(
                close(bell.entanglement_entropy(&[0]).unwrap(), 1.0, 1e-9),
                "Bell state {which} has entropy {}",
                bell.entanglement_entropy(&[0]).unwrap()
            );
            // Symmetric across the cut, which is a theorem rather than a
            // property of the implementation.
            assert!(close(
                bell.entanglement_entropy(&[0]).unwrap(),
                bell.entanglement_entropy(&[1]).unwrap(),
                1e-9
            ));
        }

        for n in 2..=4usize {
            let state = ghz(n).unwrap();
            assert!(
                close(state.entanglement_entropy(&[0]).unwrap(), 1.0, 1e-9),
                "GHZ({n}) has entropy {}",
                state.entanglement_entropy(&[0]).unwrap()
            );
        }
        // Losing a qubit of GHZ leaves nothing; losing one of W does not.
        let ghz3 = ghz(3).unwrap();
        let rest = ghz3.reduced_density_matrix(&[0, 1]).unwrap();
        let mixed = DensityMatrix { n: 2, rho: rest };
        assert!(close(mixed.purity(), 0.5, 1e-9), "GHZ's pair has purity {}", mixed.purity());
        let w = w_state(3).unwrap();
        assert!(
            w.entanglement_entropy(&[0]).unwrap() > 0.9,
            "W(3) should be entangled across a single cut"
        );
        let w_pair = DensityMatrix { n: 2, rho: w.reduced_density_matrix(&[0, 1]).unwrap() };
        assert!(
            w_pair.von_neumann_entropy().unwrap() > 0.9,
            "the remaining W pair should still be mixed"
        );
    }

    #[test]
    fn the_schmidt_coefficients_reproduce_the_entropy_they_encode() {
        // Two routes to the same number, one through eigenvalues and one
        // through the coefficients. They must agree, and the coefficients
        // must be normalised.
        let mut rng = Rng::new(0x_9E11_0003);
        for _ in 0..40 {
            let state = random_state(4, &mut rng).unwrap();
            let coefficients = state.schmidt_coefficients(&[0, 1]).unwrap();
            let total: f64 = coefficients.iter().map(|c| c * c).sum();
            assert!(close(total, 1.0, 1e-8), "the coefficients square to {total}");
            assert!(
                coefficients.windows(2).all(|w| w[0] >= w[1] - 1e-12),
                "the coefficients are not descending"
            );
            let from_schmidt: f64 = coefficients
                .iter()
                .filter(|c| **c > 1e-8)
                .map(|c| -(c * c) * (c * c).log2())
                .sum();
            let direct = state.entanglement_entropy(&[0, 1]).unwrap();
            assert!(
                close(from_schmidt, direct, 1e-7),
                "the two entropies are {from_schmidt} and {direct}"
            );
        }
        // A Bell state has exactly two equal coefficients.
        let bell = bell_state(0).unwrap();
        let coefficients = bell.schmidt_coefficients(&[0]).unwrap();
        assert_eq!(coefficients.len(), 2);
        for c in &coefficients {
            assert!(close(*c, std::f64::consts::FRAC_1_SQRT_2, 1e-9), "a coefficient is {c}");
        }
    }

    #[test]
    fn the_bloch_vector_has_unit_length_exactly_when_the_qubit_is_unentangled() {
        let mut rng = Rng::new(0x_9E11_0004);
        for _ in 0..200 {
            // A single qubit is always pure.
            let single = random_state(1, &mut rng).unwrap();
            let (x, y, z) = single.bloch_vector(0).unwrap();
            assert!(
                close(x.hypot(y).hypot(z), 1.0, 1e-9),
                "a pure qubit has Bloch length {}",
                x.hypot(y).hypot(z)
            );
            // The z component is the Z expectation, by definition.
            assert!(close(z, single.expectation_z(0).unwrap(), 1e-12));

            // A qubit of a random larger state is generally mixed.
            let bigger = random_state(3, &mut rng).unwrap();
            let (x, y, z) = bigger.bloch_vector(1).unwrap();
            let length = x.hypot(y).hypot(z);
            assert!(length <= 1.0 + 1e-9, "the Bloch length is {length}");
        }
        // Known vectors for the axis states.
        let mut plus = QState::zero(1).unwrap();
        plus.apply_single(0, &Gate::h()).unwrap();
        let (x, y, z) = plus.bloch_vector(0).unwrap();
        assert!(close(x, 1.0, 1e-12) && close(y, 0.0, 1e-12) && close(z, 0.0, 1e-12));
        let mut plus_i = QState::zero(1).unwrap();
        plus_i.apply_single(0, &Gate::h()).unwrap();
        plus_i.apply_single(0, &Gate::s()).unwrap();
        let (x, y, z) = plus_i.bloch_vector(0).unwrap();
        assert!(close(x, 0.0, 1e-12) && close(y, 1.0, 1e-12) && close(z, 0.0, 1e-12));
    }

    #[test]
    fn pauli_string_expectations_agree_with_the_matrices_they_name() {
        // Building the operator explicitly and taking <psi|P|psi> is the
        // definition; the routine computes it by basis rotation instead, and
        // the two must agree on every string.
        let mut rng = Rng::new(0x_9E11_0005);
        let symbols = ['I', 'X', 'Y', 'Z'];
        for _ in 0..60 {
            let state = random_state(3, &mut rng).unwrap();
            for code in 0..64usize {
                let name: String = (0..3).rev().map(|k| symbols[(code >> (2 * k)) & 3]).collect();
                let reported = state.expectation_pauli_string(&name).unwrap();

                // The explicit route: apply the tensor product to the state.
                let mut applied = state.clone();
                for (position, symbol) in name.chars().enumerate() {
                    let q = 3 - 1 - position;
                    match symbol {
                        'X' => applied.apply_single(q, &Gate::x()).unwrap(),
                        'Y' => applied.apply_single(q, &Gate::y()).unwrap(),
                        'Z' => applied.apply_single(q, &Gate::z()).unwrap(),
                        _ => {}
                    }
                }
                let direct = state.inner(&applied).unwrap();
                assert!(close(direct.im, 0.0, 1e-9), "{name} has an imaginary expectation");
                assert!(
                    close(reported, direct.re, 1e-9),
                    "{name}: {reported} against {}",
                    direct.re
                );
            }
        }
        // The identity string is always one.
        let state = random_state(2, &mut rng).unwrap();
        assert!(close(state.expectation_pauli_string("II").unwrap(), 1.0, 1e-12));
        assert!(state.expectation_pauli_string("XYZ").is_err());
        assert!(state.expectation_pauli_string("XQ").is_err());
    }

    // -----------------------------------------------------------------
    // Circuits
    // -----------------------------------------------------------------

    #[test]
    fn a_circuit_followed_by_its_inverse_is_the_identity() {
        // The test that catches the reverse-without-adjoint error, which
        // survives any circuit built only from self-inverse gates -- so the
        // circuit here deliberately uses ones that are not.
        let mut circuit = Circuit::new(3).unwrap();
        circuit
            .h(0)
            .t(1)
            .cx(0, 1)
            .ry(2, 0.7)
            .ccx(0, 1, 2)
            .rz(0, -1.1)
            .swap(1, 2)
            .phase(1, 0.35)
            .cx(2, 0);
        let mut round_trip = circuit.clone();
        round_trip.append(&circuit.inverse()).unwrap();

        let unitary = round_trip.unitary_small().unwrap();
        assert!(
            matrix_close(&unitary, &identity_matrix(8), 1e-12),
            "the round trip is not the identity"
        );

        // Reversing without adjointing is not, which is what makes the test
        // worth running.
        let mut naive = circuit.clone();
        let reversed = Circuit { n: 3, ops: circuit.ops.iter().rev().cloned().collect() };
        naive.append(&reversed).unwrap();
        assert!(
            !matrix_close(&naive.unitary_small().unwrap(), &identity_matrix(8), 1e-9),
            "reversing alone happened to work, so the test proves nothing"
        );
        assert!(circuit.append(&Circuit::new(2).unwrap()).is_err());
    }

    trait TGate {
        fn t(&mut self, q: usize) -> &mut Self;
    }
    impl TGate for Circuit {
        fn t(&mut self, q: usize) -> &mut Self {
            self.gate(q, Gate::t())
        }
    }

    #[test]
    fn the_unitary_is_unitary_and_matches_running_the_circuit() {
        // The matrix is built column by column from basis states, so agreeing
        // with a run on a *superposition* is a real check on linearity.
        let mut rng = Rng::new(0x_9E11_0006);
        let mut circuit = Circuit::new(3).unwrap();
        circuit.h(0).cx(0, 1).ry(2, 1.2).ccx(1, 2, 0).cz(0, 2).swap(0, 1);
        let unitary = circuit.unitary_small().unwrap();

        // U^dagger U = I.
        let size = 8usize;
        for i in 0..size {
            for j in 0..size {
                let entry =
                    (0..size).fold(ZERO, |acc, k| acc + unitary[k][i].conjugate() * unitary[k][j]);
                let expected = f64::from(i == j);
                assert!(
                    close(entry.re, expected, 1e-12) && close(entry.im, 0.0, 1e-12),
                    "the columns are not orthonormal at ({i}, {j})"
                );
            }
        }

        for _ in 0..30 {
            let state = random_state(3, &mut rng).unwrap();
            let run = circuit.run(&state).unwrap();
            for row in 0..size {
                let expected =
                    (0..size).fold(ZERO, |acc, k| acc + unitary[row][k] * state.amps[k]);
                assert!(
                    (run.amps[row].re - expected.re).abs() < 1e-12
                        && (run.amps[row].im - expected.im).abs() < 1e-12,
                    "the matrix and the run disagree at row {row}"
                );
            }
        }
        assert!(Circuit::new(11).unwrap().unitary_small().is_err());
    }

    #[test]
    fn depth_counts_layers_and_gate_count_counts_gates() {
        // Three gates on disjoint qubits are one layer; three on the same
        // qubit are three. Depth is what a coherence time is compared with,
        // so the distinction matters.
        let mut wide = Circuit::new(3).unwrap();
        wide.h(0).h(1).h(2);
        assert_eq!(wide.depth(), 1, "disjoint gates should share a layer");
        assert_eq!(wide.gate_count(), 3);

        let mut deep = Circuit::new(3).unwrap();
        deep.h(0).x(0).z(0);
        assert_eq!(deep.depth(), 3, "gates on one qubit cannot share a layer");

        // A two-qubit gate blocks both its wires.
        let mut mixed = Circuit::new(3).unwrap();
        mixed.h(0).h(2).cx(0, 1).h(2);
        assert_eq!(mixed.depth(), 2);
        assert_eq!(mixed.gate_count(), 4);

        // Barriers are free.
        mixed.barrier();
        assert_eq!(mixed.depth(), 2);
        assert_eq!(mixed.gate_count(), 4);
        assert_eq!(Circuit::new(2).unwrap().depth(), 0);
    }

    #[test]
    fn the_text_and_diagram_forms_describe_the_circuit_they_came_from() {
        let mut circuit = Circuit::new(3).unwrap();
        circuit.h(0).cx(0, 1).ccx(0, 1, 2).swap(1, 2).barrier().rx(2, 0.3);
        let text = circuit.to_qasm_lite();
        assert!(text.starts_with("qubits 3\n"));
        assert!(text.contains("u 0 H"), "{text}");
        assert!(text.contains("cX 0 1"), "{text}");
        assert!(text.contains("ccx 0 1 2"), "{text}");
        assert!(text.contains("swap 1 2"), "{text}");
        assert!(text.contains("barrier"), "{text}");
        // An unnamed gate falls back to U rather than lying about itself.
        assert!(text.contains("u 2 U"), "{text}");

        let drawing = circuit.draw_ascii();
        assert_eq!(drawing.lines().count(), 3);
        assert!(drawing.lines().next().unwrap().starts_with("q0: "));
        // Every row is the same width, or the diagram does not line up.
        let widths: Vec<usize> = drawing.lines().map(str::len).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "the rows are ragged: {widths:?}");
    }

    // -----------------------------------------------------------------
    // Entanglement and non-locality
    // -----------------------------------------------------------------

    #[test]
    fn the_bell_state_violates_the_chsh_bound_and_a_product_state_does_not() {
        // The number that separates quantum mechanics from every local hidden
        // variable theory. Two square root two is not approached, it is hit
        // exactly, and no product state gets past two.
        let angles = chsh_optimal_angles();
        let bell = bell_state(0).unwrap();
        let value = chsh_value(&bell, angles).unwrap();
        assert!(
            close(value.abs(), 2.0 * 2.0f64.sqrt(), 1e-9),
            "the Bell state gives {value}, not 2 sqrt 2"
        );

        let mut rng = Rng::new(0x_9E11_0007);
        for _ in 0..300 {
            // Any product of two one-qubit states obeys the classical bound.
            let a = random_state(1, &mut rng).unwrap();
            let b = random_state(1, &mut rng).unwrap();
            let mut amps = vec![ZERO; 4];
            for i in 0..2 {
                for j in 0..2 {
                    amps[2 * i + j] = a.amps[i] * b.amps[j];
                }
            }
            let product = QState::from_amps(amps).unwrap();
            let random_angles = (
                rng.next_f64() * std::f64::consts::TAU,
                rng.next_f64() * std::f64::consts::TAU,
                rng.next_f64() * std::f64::consts::TAU,
                rng.next_f64() * std::f64::consts::TAU,
            );
            for angles in [angles, random_angles] {
                let value = chsh_value(&product, angles).unwrap();
                assert!(
                    value.abs() <= 2.0 + 1e-9,
                    "a product state reached {value}, above the classical bound"
                );
            }
        }
        // And nothing reaches the algebraic maximum of four.
        for _ in 0..200 {
            let state = random_state(2, &mut rng).unwrap();
            let random_angles = (
                rng.next_f64() * std::f64::consts::TAU,
                rng.next_f64() * std::f64::consts::TAU,
                rng.next_f64() * std::f64::consts::TAU,
                rng.next_f64() * std::f64::consts::TAU,
            );
            let value = chsh_value(&state, random_angles).unwrap();
            assert!(
                value.abs() <= 2.0 * 2.0f64.sqrt() + 1e-9,
                "a state reached {value}, above Tsirelson's bound"
            );
        }
        assert!(chsh_value(&ghz(3).unwrap(), angles).is_err());
    }

    #[test]
    fn teleportation_moves_the_state_exactly_whatever_it_was() {
        // The output Bloch vector must equal the input's, for every input and
        // every measurement outcome -- which is the point: the protocol
        // works without knowing what was sent.
        let mut rng = Rng::new(0x_9E11_0008);
        for _ in 0..200 {
            let theta = rng.next_f64() * std::f64::consts::PI;
            let phi = rng.next_f64() * std::f64::consts::TAU;
            let (input, output) = quantum_teleportation_demo(theta, phi, &mut rng).unwrap();
            assert!(
                (input.0 - output.0).abs() < 1e-10
                    && (input.1 - output.1).abs() < 1e-10
                    && (input.2 - output.2).abs() < 1e-10,
                "sent {input:?} and received {output:?}"
            );
            // And it really was a non-trivial state.
            let length = input.0.hypot(input.1).hypot(input.2);
            assert!(close(length, 1.0, 1e-9), "the input is not pure: length {length}");
        }
    }

    #[test]
    fn superdense_coding_carries_two_bits_on_one_qubit() {
        for bits in [(false, false), (false, true), (true, false), (true, true)] {
            let decoded = superdense_coding_demo(bits).unwrap();
            assert_eq!(decoded, bits, "sent {bits:?} and received {decoded:?}");
        }
        assert!(close(no_cloning_fidelity_bound(), 5.0 / 6.0, 1e-15));
    }

    // -----------------------------------------------------------------
    // Density matrices and channels
    // -----------------------------------------------------------------

    #[test]
    fn a_pure_state_has_purity_one_and_a_mixture_has_less() {
        let mut rng = Rng::new(0x_9E11_0009);
        for _ in 0..60 {
            let state = random_state(2, &mut rng).unwrap();
            let rho = DensityMatrix::from_state(&state);
            assert!(rho.is_valid(1e-9), "a pure state's density matrix is invalid");
            assert!(close(rho.purity(), 1.0, 1e-9), "purity is {}", rho.purity());
            assert!(
                close(rho.von_neumann_entropy().unwrap(), 0.0, 1e-8),
                "a pure state has entropy {}",
                rho.von_neumann_entropy().unwrap()
            );
        }

        // The maximally mixed state of n qubits has purity 1 / 2^n and
        // entropy n bits, both exactly.
        for n in 1..=3usize {
            let size = 1usize << n;
            let states: Vec<QState> =
                (0..size).map(|i| QState::basis(n, i as u64).unwrap()).collect();
            let weights = vec![1.0 / size as f64; size];
            let rho = DensityMatrix::from_mixture(&states, &weights).unwrap();
            assert!(rho.is_valid(1e-9));
            assert!(
                close(rho.purity(), 1.0 / size as f64, 1e-9),
                "purity is {}",
                rho.purity()
            );
            assert!(
                close(rho.von_neumann_entropy().unwrap(), n as f64, 1e-8),
                "entropy is {}",
                rho.von_neumann_entropy().unwrap()
            );
        }

        // A mixture of two non-orthogonal states is still a valid state and
        // still less pure than either.
        let mut a = QState::zero(1).unwrap();
        a.apply_single(0, &Gate::h()).unwrap();
        let b = QState::zero(1).unwrap();
        let rho = DensityMatrix::from_mixture(&[a, b], &[0.3, 0.7]).unwrap();
        assert!(rho.is_valid(1e-9));
        assert!(rho.purity() < 1.0 && rho.purity() > 0.5, "purity is {}", rho.purity());

        assert!(DensityMatrix::from_mixture(&[], &[]).is_err());
        assert!(DensityMatrix::from_mixture(
            &[QState::zero(1).unwrap()],
            &[0.5]
        )
        .is_err());
    }

    #[test]
    fn every_channel_preserves_the_trace_and_moves_the_bloch_vector_as_advertised() {
        // Trace preservation is the condition that makes a map physical, and
        // it is checkable directly. What each channel does to the Bloch
        // vector is what distinguishes them, and that is checked too.
        let mut source = QState::zero(1).unwrap();
        source.apply_single(0, &Gate::ry(0.9)).unwrap();
        source.apply_single(0, &Gate::rz(0.5)).unwrap();
        let start = DensityMatrix::from_state(&source);
        let bloch = |rho: &DensityMatrix| -> (f64, f64, f64) {
            (
                2.0 * rho.rho[0][1].re,
                -2.0 * rho.rho[0][1].im,
                rho.rho[0][0].re - rho.rho[1][1].re,
            )
        };
        let (x0, y0, z0) = bloch(&start);

        for p in [0.0f64, 0.1, 0.35, 1.0] {
            for (name, kraus) in [
                ("depolarizing", depolarizing_channel(p).unwrap()),
                ("amplitude", amplitude_damping(p).unwrap()),
                ("phase", phase_damping(p).unwrap()),
                ("bitflip", bit_flip(p).unwrap()),
                ("phaseflip", phase_flip(p).unwrap()),
            ] {
                assert!(
                    is_trace_preserving(&kraus, 1e-12),
                    "{name} at p = {p} is not trace preserving"
                );
                let mut rho = start.clone();
                rho.apply_channel(&kraus).unwrap();
                assert!(rho.is_valid(1e-9), "{name} at p = {p} produced an invalid state");
                assert!(
                    rho.purity() <= start.purity() + 1e-9,
                    "{name} at p = {p} raised the purity to {}",
                    rho.purity()
                );

                let (x, y, z) = bloch(&rho);
                match name {
                    // Depolarising shrinks every component by the same factor.
                    "depolarizing" if p > 0.0 && x0.abs() > 1e-9 => {
                        let factor = x / x0;
                        assert!(
                            close(y / y0, factor, 1e-9) && close(z / z0, factor, 1e-9),
                            "depolarising was not isotropic: {}, {}, {}",
                            x / x0,
                            y / y0,
                            z / z0
                        );
                    }
                    // Phase damping leaves z alone and shrinks x and y.
                    "phase" => {
                        assert!(close(z, z0, 1e-12), "phase damping moved z to {z}");
                        assert!(x.abs() <= x0.abs() + 1e-12 && y.abs() <= y0.abs() + 1e-12);
                    }
                    // Bit flip leaves x alone.
                    "bitflip" => assert!(close(x, x0, 1e-12), "the bit flip moved x to {x}"),
                    // Phase flip leaves z alone.
                    "phaseflip" => assert!(close(z, z0, 1e-12), "the phase flip moved z to {z}"),
                    _ => {}
                }
            }
        }
        // Full amplitude damping sends everything to the ground state.
        let mut decayed = start.clone();
        decayed.apply_channel(&amplitude_damping(1.0).unwrap()).unwrap();
        assert!(close(decayed.rho[0][0].re, 1.0, 1e-12), "the decayed state is {:?}", decayed.rho);
        // Full depolarisation gives the maximally mixed state.
        let mut wrecked = start.clone();
        wrecked.apply_channel(&depolarizing_channel(1.0).unwrap()).unwrap();
        assert!(close(wrecked.purity(), 0.5, 1e-9), "purity is {}", wrecked.purity());

        assert!(depolarizing_channel(-0.1).is_err());
        assert!(amplitude_damping(1.5).is_err());
        assert!(phase_damping(-1.0).is_err());
        assert!(bit_flip(2.0).is_err());
        assert!(phase_flip(-0.5).is_err());
    }

    #[test]
    fn the_partial_trace_agrees_with_the_state_vector_route() {
        // The reduced density matrix can be got either from the amplitudes or
        // from the full density matrix, and the two must coincide.
        let mut rng = Rng::new(0x_9E11_000A);
        for _ in 0..40 {
            let state = random_state(3, &mut rng).unwrap();
            let full = DensityMatrix::from_state(&state);
            for keep in [vec![0usize], vec![1], vec![0, 2], vec![1, 2]] {
                let from_state = state.reduced_density_matrix(&keep).unwrap();
                let from_rho = full.partial_trace(&keep).unwrap();
                assert!(
                    matrix_close(&from_state, &from_rho.rho, 1e-12),
                    "the two partial traces disagree on {keep:?}"
                );
                assert!(from_rho.is_valid(1e-9), "the reduced state is invalid");
            }
        }
        let state = random_state(2, &mut rng).unwrap();
        assert!(state.reduced_density_matrix(&[]).is_err());
        assert!(state.reduced_density_matrix(&[0, 0]).is_err());
        assert!(state.reduced_density_matrix(&[5]).is_err());
    }

    #[test]
    fn pauli_decomposition_reconstructs_the_matrix_it_came_from() {
        // The coefficients are only meaningful if summing the terms back
        // returns the original operator, so that is the test.
        let cases: Vec<Vec<Vec<Complex>>> = vec![
            vec![
                vec![Complex::new(1.5, 0.0), Complex::new(0.3, -0.7)],
                vec![Complex::new(0.3, 0.7), Complex::new(-0.4, 0.0)],
            ],
            lift_single(2, 0, &Gate::z()),
            (0..4)
                .map(|i| {
                    (0..4)
                        .map(|j| Complex::new(((i * 4 + j) % 5) as f64 - 2.0, 0.0))
                        .collect()
                })
                .collect(),
        ];
        for h in &cases {
            // Force Hermiticity, since only Hermitian operators decompose
            // into real Pauli coefficients.
            let size = h.len();
            let hermitian: Vec<Vec<Complex>> = (0..size)
                .map(|i| {
                    (0..size)
                        .map(|j| scale(h[i][j] + h[j][i].conjugate(), 0.5))
                        .collect()
                })
                .collect();
            let terms = pauli_decompose(&hermitian).unwrap();
            let qubits = size.trailing_zeros() as usize;

            let mut rebuilt = vec![vec![ZERO; size]; size];
            for (name, coefficient) in &terms {
                for i in 0..size {
                    for j in 0..size {
                        let mut entry = ONE;
                        for (k, symbol) in name.chars().enumerate() {
                            let gate = match symbol {
                                'X' => Gate::x(),
                                'Y' => Gate::y(),
                                'Z' => Gate::z(),
                                _ => Gate::identity(),
                            };
                            let row = (i >> (qubits - 1 - k)) & 1;
                            let column = (j >> (qubits - 1 - k)) & 1;
                            entry = entry * gate.matrix[row][column];
                        }
                        rebuilt[i][j] = rebuilt[i][j] + scale(entry, *coefficient);
                    }
                }
            }
            assert!(
                matrix_close(&rebuilt, &hermitian, 1e-9),
                "the decomposition does not rebuild the matrix"
            );
        }
        // A known case: Z on the low qubit of two.
        let terms = pauli_decompose(&lift_single(2, 0, &Gate::z())).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].0, "IZ");
        assert!(close(terms[0].1, 1.0, 1e-12));
        assert!(pauli_decompose(&vec![vec![ONE; 3]; 3]).is_err());
    }

    #[test]
    fn the_constructors_refuse_degenerate_input() {
        assert!(QState::zero(0).is_err());
        assert!(QState::zero(MAX_QUBITS + 1).is_err());
        assert!(QState::basis(2, 4).is_err());
        assert!(QState::from_amps(vec![ONE; 3]).is_err());
        assert!(QState::from_amps(vec![ZERO; 4]).is_err());
        assert!(QState::plus_all(0).is_err());
        assert!(Circuit::new(0).is_err());
        assert!(bell_state(4).is_err());
        assert!(ghz(1).is_err());
        assert!(w_state(1).is_err());
        assert!(random_state(0, &mut Rng::new(1)).is_err());

        let mut state = QState::zero(2).unwrap();
        assert!(state.apply_single(2, &Gate::x()).is_err());
        assert!(state.apply_controlled(0, 0, &Gate::x()).is_err());
        assert!(state.apply_controlled(0, 5, &Gate::x()).is_err());
        assert!(state.apply_ccx(0, 1, 1).is_err());
        assert!(state.apply_swap(0, 9).is_err());
        assert!(state.measure_qubit(7, &mut Rng::new(2)).is_err());
        assert!(state.expectation_z(9).is_err());
        assert!(state.inner(&QState::zero(3).unwrap()).is_err());
        // Swapping a qubit with itself is a no-op rather than an error.
        assert!(state.apply_swap(1, 1).is_ok());

        let mut rho = DensityMatrix::from_state(&state);
        assert!(rho.apply_gate(4, &Gate::x()).is_err());
        assert!(rho.apply_channel(&[]).is_err());
        // A set of operators that is not trace preserving is not a channel.
        assert!(rho
            .apply_channel(&[from_rows([[ONE, ZERO], [ZERO, ZERO]])])
            .is_err());
        assert!(rho.partial_trace(&[0, 0]).is_err());
    }
}
