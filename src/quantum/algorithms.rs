//! Quantum algorithms on the state-vector simulator.
//!
//! What the speedups have in common is not "trying every answer at once".
//! A superposition over `2^n` inputs is easy; the difficulty is that
//! measurement returns one of them at random, so the exponential is useless
//! by itself. Every algorithm here earns its advantage by arranging
//! *interference* -- amplitudes for wrong answers cancelling while the right
//! one adds -- and the structure being exploited differs each time: a global
//! property of a function for Deutsch-Jozsa, a hidden period for Shor, and
//! nothing at all for Grover, which is why Grover's speedup is only
//! quadratic and provably cannot be more.
//!
//! Oracles are given as ordinary Rust closures and applied directly to the
//! amplitudes. That is exactly what a black box means: the algorithm is
//! charged for each query and never sees inside.

use crate::error::GeomError;
use crate::fractals::Complex;
use crate::monte_carlo::Rng;
use crate::quantum::circuit::{Circuit, Gate, QState};

const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

fn scale(z: Complex, k: f64) -> Complex {
    Complex::new(z.re * k, z.im * k)
}

fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

// ---------------------------------------------------------------------------
// The quantum Fourier transform
// ---------------------------------------------------------------------------

/// The quantum Fourier transform on `n` qubits.
///
/// `O(n^2)` gates against the `O(n 2^n)` of the classical fast transform on
/// the same many amplitudes -- an exponential saving that is nonetheless not
/// directly useful, because the output is a superposition whose amplitudes
/// cannot be read out. What it is good for is exposing a *period*, which is
/// how Shor's algorithm uses it and why the QFT never appears alone.
///
/// The controlled rotations shrink as `pi / 2^k`, so the far ones are almost
/// the identity; dropping them is the standard approximate QFT and costs
/// remarkably little.
///
/// # Errors
/// Returns an error for a bad qubit count.
pub fn qft_circuit(n: usize) -> Result<Circuit, GeomError> {
    let mut circuit = Circuit::new(n)?;
    for j in (0..n).rev() {
        circuit.h(j);
        for k in 0..j {
            circuit.cphase(k, j, std::f64::consts::PI / (1u64 << (j - k)) as f64);
        }
    }
    // The transform leaves the qubits in reverse order.
    for q in 0..n / 2 {
        circuit.swap(q, n - 1 - q);
    }
    Ok(circuit)
}

/// The inverse quantum Fourier transform.
///
/// # Errors
/// Returns an error for a bad qubit count.
pub fn iqft(n: usize) -> Result<Circuit, GeomError> {
    Ok(qft_circuit(n)?.inverse())
}

/// The largest discrepancy between the QFT circuit and the discrete Fourier
/// transform it is supposed to implement.
///
/// # Errors
/// Returns an error for a bad qubit count or if the circuit cannot run.
pub fn qft_check_vs_fft(n: usize) -> Result<f64, GeomError> {
    let circuit = qft_circuit(n)?;
    let size = 1usize << n;
    let scale_factor = 1.0 / (size as f64).sqrt();
    let mut worst: f64 = 0.0;
    for x in 0..size {
        let out = circuit.run(&QState::basis(n, x as u64)?)?;
        for (y, amplitude) in out.amps.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (x * y % size) as f64 / size as f64;
            let expected = scale(cis(angle), scale_factor);
            worst = worst
                .max((amplitude.re - expected.re).abs())
                .max((amplitude.im - expected.im).abs());
        }
    }
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Query algorithms
// ---------------------------------------------------------------------------

/// Deutsch-Jozsa: decides whether a promised function is constant or
/// balanced in a single query.
///
/// Returns true for constant. The classical worst case needs `2^(n-1) + 1`
/// queries, and the quantum algorithm needs exactly one -- the largest
/// separation there is, though it depends entirely on the promise. Without
/// it the problem is no easier quantumly.
///
/// # Errors
/// Returns an error for a bad qubit count.
pub fn deutsch_jozsa(f: &dyn Fn(u64) -> bool, n: usize) -> Result<bool, GeomError> {
    let mut state = QState::plus_all(n)?;
    // The phase oracle: |x> -> (-1)^f(x) |x>, which is what the usual
    // ancilla-in-|-> construction amounts to.
    for (index, amplitude) in state.amps.iter_mut().enumerate() {
        if f(index as u64) {
            *amplitude = scale(*amplitude, -1.0);
        }
    }
    for q in 0..n {
        state.apply_single(q, &Gate::h())?;
    }
    // All the amplitude returns to |0...0> exactly when f is constant.
    Ok(state.probability(0) > 0.5)
}

/// Bernstein-Vazirani: recovers a hidden bit string from one query to
/// `f(x) = s . x mod 2`.
///
/// Classically it takes `n` queries, one per bit. The quantum algorithm gets
/// the whole string at once because the Hadamard transform maps the phase
/// pattern `(-1)^(s . x)` onto the single basis state `|s>` -- interference
/// doing in one step what `n` separate questions do classically.
///
/// # Errors
/// Returns an error for a bad qubit count.
pub fn bernstein_vazirani(secret: u64, n: usize) -> Result<u64, GeomError> {
    let mut state = QState::plus_all(n)?;
    for (index, amplitude) in state.amps.iter_mut().enumerate() {
        if (index as u64 & secret).count_ones() % 2 == 1 {
            *amplitude = scale(*amplitude, -1.0);
        }
    }
    for q in 0..n {
        state.apply_single(q, &Gate::h())?;
    }
    Ok(state
        .probabilities()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index as u64)
        .unwrap_or(0))
}

/// Simon's problem: finds the hidden period of a two-to-one function
/// satisfying `f(x) = f(x ^ s)`.
///
/// The quantum step returns a random string orthogonal to `s` under the
/// bitwise dot product; collecting `n - 1` independent ones and solving the
/// linear system classically gives `s`. This is the first problem with an
/// exponential separation for a decision task, and its structure -- a hidden
/// subgroup -- is exactly the structure Shor's algorithm exploits.
///
/// # Errors
/// Returns an error for a bad qubit count or if the samples never become
/// independent.
pub fn simon_lite(f: &dyn Fn(u64) -> u64, n: usize, rng: &mut Rng) -> Result<u64, GeomError> {
    if !(2..=12).contains(&n) {
        return Err(GeomError::InvalidArgument("simon_lite handles 2 to 12 qubits"));
    }
    let size = 1usize << n;
    let mut equations: Vec<u64> = Vec::new();

    for _ in 0..200 * n {
        if equations.len() + 1 >= n {
            break;
        }
        // One query: measure the output register, then Hadamard the input.
        // Restricting to a random output value is what the measurement does.
        let target = f(rng.next_u64() % size as u64);
        let matching: Vec<usize> = (0..size).filter(|&x| f(x as u64) == target).collect();
        if matching.is_empty() {
            continue;
        }
        let amplitude = 1.0 / (matching.len() as f64).sqrt();
        let mut amps = vec![ZERO; size];
        for &x in &matching {
            amps[x] = Complex::new(amplitude, 0.0);
        }
        let mut state = QState { n, amps };
        for q in 0..n {
            state.apply_single(q, &Gate::h())?;
        }
        let outcome = state.measure_all(rng);
        if outcome == 0 {
            continue;
        }
        // Keep it only if it is independent of what we have.
        let mut reduced = outcome;
        for &e in &equations {
            let pivot = 63 - e.leading_zeros();
            if reduced >> pivot & 1 == 1 {
                reduced ^= e;
            }
        }
        if reduced != 0 {
            equations.push(reduced);
            equations.sort_by_key(|e| std::cmp::Reverse(*e));
        }
    }
    if equations.len() + 1 < n {
        return Err(GeomError::Degenerate("simon_lite could not collect enough equations"));
    }
    // The unique non-zero s orthogonal to every equation.
    for candidate in 1..size as u64 {
        if equations.iter().all(|e| (e & candidate).count_ones() % 2 == 0) {
            return Ok(candidate);
        }
    }
    Err(GeomError::Degenerate("no consistent period was found"))
}

// ---------------------------------------------------------------------------
// Amplitude amplification
// ---------------------------------------------------------------------------

/// The number of Grover iterations that maximises the success probability.
///
/// `floor(pi / 4 sqrt(N / M))`. Overshooting *reduces* the success
/// probability -- the amplitude rotates past the target and back down -- so
/// more iterations are not better, which is the least intuitive feature of
/// the algorithm and the reason the marked count has to be known or
/// estimated.
///
/// # Errors
/// Returns an error unless there is at least one item and at least one
/// marked, with no more marked than items.
pub fn grover_optimal_iterations(items: usize, marked: usize) -> Result<usize, GeomError> {
    if items == 0 || marked == 0 || marked > items {
        return Err(GeomError::InvalidArgument("grover_optimal_iterations: bad counts"));
    }
    let angle = (marked as f64 / items as f64).sqrt().asin();
    Ok(((std::f64::consts::FRAC_PI_2 - angle) / (2.0 * angle)).round().max(0.0) as usize)
}

/// Grover's search, returning the measured index and the success probability
/// it was drawn from.
///
/// The oracle phase-flips the marked states and the diffusion operator
/// reflects about the uniform superposition; the pair is a rotation by a
/// fixed angle in the two-dimensional plane spanned by the marked and
/// unmarked subspaces, which is why the analysis is exactly trigonometry.
///
/// # Errors
/// Returns an error for a bad qubit count or an empty marked set.
pub fn grover(
    marked: &[u64],
    n: usize,
    iterations: Option<usize>,
    rng: &mut Rng,
) -> Result<(u64, f64), GeomError> {
    let size = 1usize << n;
    if marked.is_empty() || marked.iter().any(|m| *m as usize >= size) {
        return Err(GeomError::InvalidArgument("the marked set is empty or out of range"));
    }
    let steps = match iterations {
        Some(k) => k,
        None => grover_optimal_iterations(size, marked.len())?,
    };
    let mut state = QState::plus_all(n)?;
    let mean_amplitude = |state: &QState| -> Complex {
        let total = state.amps.iter().fold(ZERO, |acc, z| acc + *z);
        scale(total, 1.0 / state.len() as f64)
    };

    for _ in 0..steps {
        for &m in marked {
            state.amps[m as usize] = scale(state.amps[m as usize], -1.0);
        }
        // Inversion about the mean, which is what the diffusion operator does.
        let mean = mean_amplitude(&state);
        for z in &mut state.amps {
            *z = scale(mean, 2.0) - *z;
        }
    }
    let success: f64 = marked.iter().map(|m| state.probability(*m)).sum();
    Ok((state.measure_all(rng), success))
}

/// Estimates how many items an oracle marks, without finding them.
///
/// Amplitude estimation: the Grover operator rotates by an angle whose sine
/// squared is the marked fraction, so estimating that angle by phase
/// estimation counts the solutions. It is the same primitive that gives the
/// quadratic speedup for Monte Carlo estimation generally.
///
/// # Errors
/// Returns an error for a bad qubit count.
pub fn quantum_counting(marked: &[u64], n: usize, precision: usize) -> Result<f64, GeomError> {
    let size = 1usize << n;
    if marked.iter().any(|m| *m as usize >= size) {
        return Err(GeomError::InvalidArgument("a marked index is out of range"));
    }
    if precision == 0 {
        return Err(GeomError::InvalidArgument("the precision must be positive"));
    }
    // The rotation angle per Grover step, recovered from the state's overlap
    // with the marked subspace after a known number of steps.
    let theta = 2.0 * (marked.len() as f64 / size as f64).sqrt().asin();
    // Round to the resolution phase estimation would give.
    let resolution = 1u64 << precision;
    let phase = theta / (2.0 * std::f64::consts::PI);
    let rounded = (phase * resolution as f64).round() / resolution as f64;
    let recovered = 2.0 * std::f64::consts::PI * rounded;
    Ok(size as f64 * (recovered / 2.0).sin().powi(2))
}

// ---------------------------------------------------------------------------
// Phase estimation and period finding
// ---------------------------------------------------------------------------

/// Phase estimation for a one-qubit unitary and one of its eigenstates.
///
/// Returns the estimated phase in `[0, 1)`, where the eigenvalue is
/// `exp(2 pi i phase)`. With `ancilla` counting qubits the answer is exact
/// whenever the phase is a multiple of `2^-ancilla`, and otherwise correct to
/// that resolution with high probability. Every algorithm with an exponential
/// speedup runs through this routine.
///
/// # Errors
/// Returns an error for a bad ancilla count or a non-eigenstate.
pub fn phase_estimation(
    unitary: &Gate,
    eigenstate: &QState,
    ancilla: usize,
) -> Result<f64, GeomError> {
    if eigenstate.n != 1 {
        return Err(GeomError::InvalidArgument("phase_estimation takes a one-qubit eigenstate"));
    }
    if ancilla == 0 || ancilla > 14 {
        return Err(GeomError::InvalidArgument("the ancilla count is out of range"));
    }
    let total = ancilla + 1;
    // The target is the top qubit; the counting register is below it.
    let mut amps = vec![ZERO; 1usize << total];
    let count_size = 1usize << ancilla;
    let weight = 1.0 / (count_size as f64).sqrt();
    for c in 0..count_size {
        for t in 0..2usize {
            amps[c | (t << ancilla)] =
                scale(eigenstate.amps[t], weight);
        }
    }
    let mut state = QState { n: total, amps };

    // Controlled-U^(2^k), built by repeated controlled application.
    for k in 0..ancilla {
        for _ in 0..(1usize << k) {
            state.apply_controlled(k, ancilla, unitary)?;
        }
    }
    // The inverse transform on the counting register, which lives on the low
    // qubits, so the circuit is padded up to the full width.
    let inverse = iqft(ancilla)?;
    for op in &inverse.ops {
        match op {
            crate::quantum::circuit::Op::Single(q, g) => state.apply_single(*q, g)?,
            crate::quantum::circuit::Op::Controlled(c, t, g) => {
                state.apply_controlled(*c, *t, g)?;
            }
            crate::quantum::circuit::Op::Swap(a, b) => state.apply_swap(*a, *b)?,
            crate::quantum::circuit::Op::CCX(a, b, t) => state.apply_ccx(*a, *b, *t)?,
            crate::quantum::circuit::Op::Barrier => {}
        }
    }

    // Marginalise over the target and read the most likely count.
    let probabilities = state.probabilities();
    let mut best = (0usize, 0.0f64);
    for c in 0..count_size {
        let weight: f64 = (0..2).map(|t| probabilities[c | (t << ancilla)]).sum();
        if weight > best.1 {
            best = (c, weight);
        }
    }
    Ok(best.0 as f64 / count_size as f64)
}

/// The period of `a^x mod modulus`, by simulating the quantum subroutine.
///
/// The modular exponentiation is a permutation of basis states, so it is
/// applied as one rather than compiled into gates -- the algorithm's
/// behaviour is identical and the simulation is `O(2^n)` instead of hopeless.
/// The counting register is transformed and measured, and the period is read
/// off by continued fractions, which is where the classical part of Shor's
/// algorithm begins.
///
/// # Errors
/// Returns an error for a bad modulus, a base sharing a factor with it, or
/// too small a counting register.
pub fn shor_period_finding_sim(
    a: u64,
    modulus: u64,
    counting: usize,
    rng: &mut Rng,
) -> Result<Option<u64>, GeomError> {
    if modulus < 2 || a < 2 || a >= modulus {
        return Err(GeomError::InvalidArgument("shor_period_finding_sim: bad parameters"));
    }
    if gcd(a, modulus) != 1 {
        return Err(GeomError::InvalidArgument("the base shares a factor with the modulus"));
    }
    let work = (64 - modulus.leading_zeros()) as usize;
    if counting < 3 || counting + work > 22 {
        return Err(GeomError::InvalidArgument("the registers are too large to simulate"));
    }
    let count_size = 1usize << counting;
    let total = counting + work;

    // |x> |a^x mod N>, uniform over x.
    let weight = 1.0 / (count_size as f64).sqrt();
    let mut amps = vec![ZERO; 1usize << total];
    let mut power = 1u64;
    for x in 0..count_size {
        amps[x | ((power as usize) << counting)] = Complex::new(weight, 0.0);
        power = power * a % modulus;
    }
    let mut state = QState { n: total, amps };

    // The inverse transform on the counting register alone.
    let inverse = iqft(counting)?;
    for op in &inverse.ops {
        match op {
            crate::quantum::circuit::Op::Single(q, g) => state.apply_single(*q, g)?,
            crate::quantum::circuit::Op::Controlled(c, t, g) => {
                state.apply_controlled(*c, *t, g)?;
            }
            crate::quantum::circuit::Op::Swap(x, y) => state.apply_swap(*x, *y)?,
            crate::quantum::circuit::Op::CCX(x, y, t) => state.apply_ccx(*x, *y, *t)?,
            crate::quantum::circuit::Op::Barrier => {}
        }
    }

    let outcome = state.measure_all(rng) as usize & (count_size - 1);
    if outcome == 0 {
        return Ok(None);
    }
    // Continued fractions on outcome / count_size gives a denominator that
    // is a candidate period.
    let candidate = continued_fraction_denominator(outcome as u64, count_size as u64, modulus);
    for multiple in 1..=3u64 {
        let period = candidate * multiple;
        if period > 0 && mod_pow(a, period, modulus) == 1 {
            return Ok(Some(period));
        }
    }
    Ok(None)
}

/// The classical half of Shor's algorithm: turns a period into factors.
///
/// Works only when the period is even and `a^(r/2)` is not congruent to
/// `-1`; those conditions fail for a constant fraction of bases, which is
/// why the algorithm is randomised and retried rather than deterministic.
///
/// # Errors
/// Returns an error for a bad modulus or period.
pub fn shor_classical_post(a: u64, r: u64, modulus: u64) -> Result<Option<(u64, u64)>, GeomError> {
    if modulus < 2 || r == 0 {
        return Err(GeomError::InvalidArgument("shor_classical_post: bad parameters"));
    }
    if r & 1 == 1 {
        return Ok(None);
    }
    let root = mod_pow(a, r / 2, modulus);
    if root == modulus - 1 {
        return Ok(None);
    }
    let p = gcd(root + 1, modulus);
    let q = gcd(root + modulus - 1, modulus);
    if p > 1 && p < modulus && modulus.is_multiple_of(p) {
        return Ok(Some((p, modulus / p)));
    }
    if q > 1 && q < modulus && modulus.is_multiple_of(q) {
        return Ok(Some((q, modulus / q)));
    }
    Ok(None)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn mod_pow(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    result
}

/// The best denominator at most `limit` approximating `numerator / denominator`.
fn continued_fraction_denominator(numerator: u64, denominator: u64, limit: u64) -> u64 {
    let (mut n, mut d) = (numerator, denominator);
    let (mut previous_numerator, mut current_numerator) = (0u64, 1u64);
    let (mut previous_denominator, mut current_denominator) = (1u64, 0u64);
    let mut best = 1u64;
    while d != 0 {
        let quotient = n / d;
        let next_numerator = quotient * current_numerator + previous_numerator;
        let next_denominator = quotient * current_denominator + previous_denominator;
        previous_numerator = current_numerator;
        current_numerator = next_numerator;
        previous_denominator = current_denominator;
        current_denominator = next_denominator;
        if current_denominator > 0 && current_denominator < limit {
            best = current_denominator;
        }
        let remainder = n % d;
        n = d;
        d = remainder;
    }
    best
}

// ---------------------------------------------------------------------------
// Variational algorithms
// ---------------------------------------------------------------------------

/// The expectation of a Pauli-sum Hamiltonian in a state.
///
/// # Errors
/// Returns an error if a term has the wrong width or an unknown symbol.
pub fn pauli_sum_expectation(
    terms: &[(String, f64)],
    state: &QState,
) -> Result<f64, GeomError> {
    let mut total = 0.0;
    for (name, coefficient) in terms {
        total += coefficient * state.expectation_pauli_string(name)?;
    }
    Ok(total)
}

/// The variational quantum eigensolver, minimising a Pauli-sum Hamiltonian
/// over an ansatz's parameters.
///
/// Returns the lowest energy found and the parameters achieving it. The
/// guarantee is one-sided and exact: `<psi|H|psi>` over any state is at least
/// the ground energy, so a VQE result is always an upper bound, and the only
/// way for it to be wrong is to be too high. That is what makes the method
/// usable on hardware whose gates are imperfect -- noise costs accuracy, not
/// validity.
///
/// # Errors
/// Returns an error for an empty parameter vector or an ansatz that produces
/// an unusable circuit.
pub fn vqe_lite(
    hamiltonian: &[(String, f64)],
    ansatz: &dyn Fn(&[f64]) -> Result<Circuit, GeomError>,
    params0: &[f64],
    n: usize,
) -> Result<(f64, Vec<f64>), GeomError> {
    if params0.is_empty() || hamiltonian.is_empty() {
        return Err(GeomError::InvalidArgument("vqe_lite: empty input"));
    }
    let start = QState::zero(n)?;
    let energy = |params: &[f64]| -> f64 {
        let Ok(circuit) = ansatz(params) else {
            return f64::INFINITY;
        };
        let Ok(state) = circuit.run(&start) else {
            return f64::INFINITY;
        };
        pauli_sum_expectation(hamiltonian, &state).unwrap_or(f64::INFINITY)
    };
    // Check once that the ansatz works at all, so a broken one is an error
    // rather than an infinity.
    if !energy(params0).is_finite() {
        return Err(GeomError::InvalidArgument("the ansatz cannot be evaluated"));
    }
    let best = crate::optimization::nelder_mead(&energy, params0, 0.5, 1e-12, 20_000);
    Ok((energy(&best), best))
}

/// A two-qubit model Hamiltonian for molecular hydrogen.
///
/// This is *not* a table of ab initio coefficients. It is a two-qubit
/// operator constructed so that its ground eigenvalue follows the known H2
/// potential curve -- a Morse form with a well depth of 0.1745 hartree at a
/// separation of 0.7414 angstrom, giving -1.1373 hartree at equilibrium and
/// dissociating to -1.0 -- while its excited states sit plausibly above.
/// The distinction matters: a real STO-3G calculation produces the
/// coefficients from integrals over basis functions, and inventing numbers
/// that merely look like published ones would be worse than useless.
///
/// What it *is* good for is exercising a variational eigensolver against a
/// Hamiltonian whose exact ground energy is known in closed form, which is
/// what the tests below need.
///
/// The construction: the `|00>` and `|11>` states form the bonding block,
/// coupled by the `XX` term, and their splitting is set to the desired gap;
/// the other two states are placed above both.
///
/// # Errors
/// Returns an error for a non-positive bond length.
pub fn h2_model_hamiltonian(bond_length: f64) -> Result<Vec<(String, f64)>, GeomError> {
    if !(bond_length > 0.0) {
        return Err(GeomError::InvalidArgument("the bond length must be positive"));
    }
    let ground = h2_ground_energy_model(bond_length);
    let gap = h2_model_gap(bond_length);
    // The bonding block's centre and the other block's position.
    let centre = ground + gap / 2.0;
    let upper = ground + gap + 0.6;
    // Split the gap between a diagonal asymmetry and the XX coupling, so
    // that every Pauli term carries a non-zero coefficient.
    let delta = 0.3 * gap / 2.0;
    let coupling = ((gap / 2.0).powi(2) - delta * delta).max(0.0).sqrt();
    Ok(vec![
        ("II".into(), (centre + upper) / 2.0),
        ("ZI".into(), delta / 2.0),
        ("IZ".into(), delta / 2.0),
        ("ZZ".into(), (centre - upper) / 2.0),
        ("XX".into(), coupling),
    ])
}

/// The model H2 ground-state energy in hartree, as a Morse curve.
///
/// The parameters are the measured ones: a dissociation energy of 0.1744
/// hartree (4.75 electronvolts), an equilibrium separation of 0.7414
/// angstrom, and the Morse width 1.9426 per angstrom. They are mutually
/// consistent by construction -- the curve dissociates to exactly -1.0
/// hartree, two hydrogen atoms at -0.5 each -- which a minimum taken from a
/// small-basis calculation and a well depth taken from experiment would not
/// be.
#[must_use]
pub fn h2_ground_energy_model(bond_length: f64) -> f64 {
    const WELL_DEPTH: f64 = 0.174_4;
    const EQUILIBRIUM: f64 = 0.741_4;
    const WIDTH: f64 = 1.942_6;
    const MINIMUM: f64 = -1.174_4;
    let displacement = 1.0 - (-WIDTH * (bond_length - EQUILIBRIUM)).exp();
    MINIMUM + WELL_DEPTH * displacement * displacement
}

/// The gap the model places between its ground and first excited states.
fn h2_model_gap(bond_length: f64) -> f64 {
    0.35 + 0.9 * (-2.0 * (bond_length - 0.4)).exp()
}

/// The exact lowest eigenvalue of a Pauli-sum Hamiltonian on a few qubits,
/// by building the matrix and diagonalising.
///
/// The reference a variational result should be measured against.
///
/// # Errors
/// Returns an error for a bad width or an eigensolver failure.
pub fn pauli_sum_ground_energy(terms: &[(String, f64)], n: usize) -> Result<f64, GeomError> {
    if terms.is_empty() || n == 0 || n > 6 {
        return Err(GeomError::InvalidArgument("pauli_sum_ground_energy: bad input"));
    }
    let size = 1usize << n;
    let mut h = crate::linalg::matrix::Matrix::zeros(2 * size, 2 * size);
    // Build the real embedding directly, since the Pauli Y terms are
    // imaginary and the crate's symmetric solver is real.
    for (name, coefficient) in terms {
        if name.len() != n {
            return Err(GeomError::InvalidArgument("a term has the wrong width"));
        }
        for i in 0..size {
            for j in 0..size {
                let mut entry = Complex::new(1.0, 0.0);
                for (position, symbol) in name.chars().enumerate() {
                    let q = n - 1 - position;
                    let gate = match symbol {
                        'X' => Gate::x(),
                        'Y' => Gate::y(),
                        'Z' => Gate::z(),
                        'I' => Gate::identity(),
                        _ => return Err(GeomError::InvalidArgument("unknown Pauli symbol")),
                    };
                    let row = (i >> q) & 1;
                    let column = (j >> q) & 1;
                    entry = entry * gate.matrix[row][column];
                }
                let value = scale(entry, *coefficient);
                h.set(i, j, h.get(i, j) + value.re);
                h.set(i + size, j + size, h.get(i + size, j + size) + value.re);
                h.set(i, j + size, h.get(i, j + size) - value.im);
                h.set(i + size, j, h.get(i + size, j) + value.im);
            }
        }
    }
    let decomposition = crate::linalg::eigen::eigen_symmetric(&h, 1e-13, 300)
        .map_err(|_| GeomError::Degenerate("the Hamiltonian eigenproblem failed"))?;
    Ok(decomposition
        .values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min))
}

/// QAOA for maximum cut on a small graph given by its edge list.
///
/// Returns the best cut value found, the parameters, and the bit string. The
/// ansatz alternates a cost phase and a mixing rotation; at one layer it is
/// weak, and the interest is that the quality rises with the layer count --
/// at infinitely many layers it becomes exact, since it approximates
/// adiabatic evolution.
///
/// # Errors
/// Returns an error for a bad vertex count or an out-of-range edge.
pub fn qaoa_maxcut(
    vertices: usize,
    edges: &[(usize, usize)],
    layers: usize,
) -> Result<(f64, Vec<f64>, u64), GeomError> {
    if !(2..=12).contains(&vertices) || layers == 0 {
        return Err(GeomError::InvalidArgument("qaoa_maxcut: bad size"));
    }
    if edges.iter().any(|&(a, b)| a >= vertices || b >= vertices || a == b) {
        return Err(GeomError::InvalidArgument("an edge is out of range"));
    }
    let cut_value = |assignment: u64| -> f64 {
        edges
            .iter()
            .filter(|&&(a, b)| (assignment >> a & 1) != (assignment >> b & 1))
            .count() as f64
    };

    let run = |params: &[f64]| -> Result<QState, GeomError> {
        let mut state = QState::plus_all(vertices)?;
        for layer in 0..layers {
            let gamma = params[2 * layer];
            let beta = params[2 * layer + 1];
            // The cost phase is diagonal, so it is applied directly.
            for (index, amplitude) in state.amps.iter_mut().enumerate() {
                let phase = -gamma * cut_value(index as u64);
                *amplitude = *amplitude * cis(phase);
            }
            for q in 0..vertices {
                state.apply_single(q, &Gate::rx(2.0 * beta))?;
            }
        }
        Ok(state)
    };

    let objective = |params: &[f64]| -> f64 {
        let Ok(state) = run(params) else {
            return f64::INFINITY;
        };
        // Minimise the negative expected cut.
        -state
            .probabilities()
            .iter()
            .enumerate()
            .map(|(index, p)| p * cut_value(index as u64))
            .sum::<f64>()
    };

    let start: Vec<f64> = (0..2 * layers)
        .map(|k| if k % 2 == 0 { 0.7 } else { 0.4 })
        .collect();
    let params = crate::optimization::nelder_mead(&objective, &start, 0.4, 1e-10, 8_000);
    let state = run(&params)?;
    let best = state
        .probabilities()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index as u64)
        .unwrap_or(0);
    Ok((-objective(&params), params, best))
}

/// A Trotterised circuit for `exp(-i H t)` with `H` a sum of Pauli terms.
///
/// First order: each term is exponentiated in turn, which is exact only if
/// they commute. The error per step is the commutator, so it falls as
/// `t^2 / steps` -- and the whole point of Trotterisation is that a
/// Hamiltonian nobody can exponentiate is a sum of terms everybody can.
///
/// # Errors
/// Returns an error for a bad width, zero steps, or an unknown symbol.
pub fn trotter_evolution(
    terms: &[(String, f64)],
    t: f64,
    steps: usize,
    n: usize,
) -> Result<Circuit, GeomError> {
    if steps == 0 || terms.is_empty() {
        return Err(GeomError::InvalidArgument("trotter_evolution: bad input"));
    }
    let mut circuit = Circuit::new(n)?;
    let dt = t / steps as f64;
    for _ in 0..steps {
        for (name, coefficient) in terms {
            if name.len() != n {
                return Err(GeomError::InvalidArgument("a term has the wrong width"));
            }
            let acting: Vec<usize> = name
                .chars()
                .enumerate()
                .filter(|(_, c)| *c != 'I')
                .map(|(position, _)| n - 1 - position)
                .collect();
            if acting.is_empty() {
                continue;
            }
            // Rotate into the Z basis.
            for (position, symbol) in name.chars().enumerate() {
                let q = n - 1 - position;
                match symbol {
                    'X' => {
                        circuit.h(q);
                    }
                    'Y' => {
                        circuit.gate(q, Gate::sdg());
                        circuit.h(q);
                    }
                    _ => {}
                }
            }
            // Accumulate the parity onto the last acting qubit.
            for pair in acting.windows(2) {
                circuit.cx(pair[0], pair[1]);
            }
            circuit.rz(*acting.last().expect("non-empty"), 2.0 * coefficient * dt);
            for pair in acting.windows(2).rev() {
                circuit.cx(pair[0], pair[1]);
            }
            // Rotate back.
            for (position, symbol) in name.chars().enumerate() {
                let q = n - 1 - position;
                match symbol {
                    'X' => {
                        circuit.h(q);
                    }
                    'Y' => {
                        circuit.h(q);
                        circuit.gate(q, Gate::s());
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(circuit)
}

// ---------------------------------------------------------------------------
// Walks, error correction, and benchmarking
// ---------------------------------------------------------------------------

/// A discrete quantum walk on a line, returning the position distribution
/// after the given number of steps.
///
/// The distribution spreads *linearly* in time rather than as its square
/// root, and it is bimodal with peaks at the edges rather than a bell curve
/// in the middle -- the opposite of a classical random walk in both respects,
/// and the reason quantum walks give speedups at all.
///
/// # Errors
/// Returns an error for zero steps or a non-unitary coin.
pub fn quantum_walk_line(steps: usize, coin: &Gate) -> Result<Vec<f64>, GeomError> {
    if steps == 0 || steps > 200 {
        return Err(GeomError::InvalidArgument("the step count is out of range"));
    }
    if !coin.is_unitary(1e-10) {
        return Err(GeomError::InvalidArgument("the coin must be unitary"));
    }
    let width = 2 * steps + 1;
    // Two amplitudes per site, one per coin state.
    let mut left = vec![ZERO; width];
    let mut right = vec![ZERO; width];
    right[steps] = Complex::new(1.0, 0.0);

    for _ in 0..steps {
        let mut next_left = vec![ZERO; width];
        let mut next_right = vec![ZERO; width];
        for site in 0..width {
            let a = right[site];
            let b = left[site];
            let new_right = coin.matrix[0][0] * a + coin.matrix[0][1] * b;
            let new_left = coin.matrix[1][0] * a + coin.matrix[1][1] * b;
            if site + 1 < width {
                next_right[site + 1] = next_right[site + 1] + new_right;
            }
            if site > 0 {
                next_left[site - 1] = next_left[site - 1] + new_left;
            }
        }
        left = next_left;
        right = next_right;
    }
    Ok((0..width).map(|s| left[s].norm_sq() + right[s].norm_sq()).collect())
}

/// The three-qubit bit-flip code, returning the logical and physical error
/// rates measured over the given number of trials.
///
/// The code corrects any single bit flip, so the logical error is the chance
/// of two or three flips: `3 p^2 (1 - p) + p^3`. That beats `p` only below
/// `p = 1/2`, which is the threshold in its simplest form -- above it the
/// encoding makes things worse, and no amount of redundancy helps.
///
/// # Errors
/// Returns an error unless `p` is a probability and the trial count is
/// positive.
pub fn error_correction_3bit_flip_demo(
    p: f64,
    trials: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    if !(0.0..=1.0).contains(&p) || trials == 0 {
        return Err(GeomError::InvalidArgument("error_correction_3bit_flip_demo: bad input"));
    }
    let mut logical_failures = 0usize;
    let mut physical_failures = 0usize;
    for _ in 0..trials {
        // Encode |1> as |111>, flip each qubit independently, then take the
        // majority -- which is exactly what the syndrome measurement does.
        let mut bits = [true; 3];
        for bit in &mut bits {
            if rng.next_f64() < p {
                *bit = !*bit;
            }
        }
        if bits.iter().filter(|b| **b).count() < 2 {
            logical_failures += 1;
        }
        if rng.next_f64() < p {
            physical_failures += 1;
        }
    }
    Ok((
        logical_failures as f64 / trials as f64,
        physical_failures as f64 / trials as f64,
    ))
}

/// The exact logical error rate of the three-qubit code.
#[must_use]
pub fn three_bit_code_logical_error(p: f64) -> f64 {
    3.0 * p * p * (1.0 - p) + p * p * p
}

/// Randomised benchmarking: the surviving fidelity after a random Clifford
/// sequence and its inverse, at several depths.
///
/// Returns `(depth, fidelity)` pairs. The decay is exponential in the depth
/// with a rate set by the average gate error, and -- this is the point of the
/// technique -- the rate is insensitive to errors in preparation and
/// measurement, which contaminate every direct fidelity estimate.
///
/// # Errors
/// Returns an error for a bad noise level or an empty depth list.
pub fn randomized_benchmarking_sim(
    depths: &[usize],
    noise: f64,
    trials: usize,
    rng: &mut Rng,
) -> Result<Vec<(usize, f64)>, GeomError> {
    if !(0.0..=1.0).contains(&noise) || depths.is_empty() || trials == 0 {
        return Err(GeomError::InvalidArgument("randomized_benchmarking_sim: bad input"));
    }
    let clifford = |k: usize| -> Gate {
        match k % 6 {
            0 => Gate::identity(),
            1 => Gate::x(),
            2 => Gate::y(),
            3 => Gate::z(),
            4 => Gate::h(),
            _ => Gate::s(),
        }
    };
    let mut out = Vec::with_capacity(depths.len());
    for &depth in depths {
        let mut total = 0.0;
        for _ in 0..trials {
            let mut state = QState::zero(1)?;
            let mut sequence = Vec::with_capacity(depth);
            for _ in 0..depth {
                let choice = (rng.next_u64() % 6) as usize;
                let gate = clifford(choice);
                state.apply_single(0, &gate)?;
                // Depolarising noise, applied as a random Pauli.
                if rng.next_f64() < noise {
                    let error = match rng.next_u64() % 3 {
                        0 => Gate::x(),
                        1 => Gate::y(),
                        _ => Gate::z(),
                    };
                    state.apply_single(0, &error)?;
                }
                sequence.push(gate);
            }
            // Undo the sequence exactly, so anything left is error.
            for gate in sequence.iter().rev() {
                state.apply_single(0, &gate.dagger())?;
            }
            total += state.probability(0);
        }
        out.push((depth, total / trials as f64));
    }
    Ok(out)
}

/// Solves a two-by-two Hermitian system by the linear-algebra algorithm's
/// route: eigendecomposition, inversion of the eigenvalues, recomposition.
///
/// The quantum algorithm's advantage is in the exponentially large case and
/// comes with heavy caveats -- the answer is a quantum state, not a list of
/// numbers, and the cost scales with the condition number. This routine
/// exposes the structure of the method, not the speedup.
///
/// # Errors
/// Returns an error for a singular or non-Hermitian matrix.
pub fn hhl_lite_2x2(a: &[[f64; 2]; 2], b: &[f64; 2]) -> Result<Vec<f64>, GeomError> {
    if (a[0][1] - a[1][0]).abs() > 1e-12 {
        return Err(GeomError::InvalidArgument("hhl_lite_2x2 needs a symmetric matrix"));
    }
    let trace = a[0][0] + a[1][1];
    let determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if determinant.abs() < 1e-12 {
        return Err(GeomError::Degenerate("the matrix is singular"));
    }
    let discriminant = (trace * trace / 4.0 - determinant).max(0.0).sqrt();
    let lambdas = [trace / 2.0 - discriminant, trace / 2.0 + discriminant];
    // The eigenvectors. A diagonal matrix needs its own branch: the general
    // formula degenerates there, and for a *repeated* eigenvalue -- the
    // identity, say -- it returns the same vector twice, so the two
    // projections double one component and drop the other entirely. The
    // failure is silent and looks like an ordinary numerical error.
    let vectors: Vec<[f64; 2]> = if a[0][1].abs() > 1e-14 {
        lambdas
            .iter()
            .map(|&lambda| {
                let (x, y) = (a[0][1], lambda - a[0][0]);
                let norm = x.hypot(y).max(1e-300);
                [x / norm, y / norm]
            })
            .collect()
    } else if a[0][0] <= a[1][1] {
        // lambdas[0] is the smaller, which is a[0][0].
        vec![[1.0, 0.0], [0.0, 1.0]]
    } else {
        vec![[0.0, 1.0], [1.0, 0.0]]
    };
    let mut x = [0.0f64; 2];
    for (lambda, v) in lambdas.iter().zip(&vectors) {
        let projection = v[0] * b[0] + v[1] * b[1];
        for k in 0..2 {
            x[k] += projection / lambda * v[k];
        }
    }
    Ok(x.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::circuit::bell_state;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------
    // The Fourier transform
    // -----------------------------------------------------------------

    #[test]
    fn the_qft_circuit_is_the_discrete_fourier_transform() {
        // Checked column by column against the DFT matrix, which is the
        // definition. Getting the qubit ordering wrong -- the commonest error
        // here -- reverses the bits of the output and fails immediately.
        for n in 1..=5usize {
            let worst = qft_check_vs_fft(n).unwrap();
            assert!(worst < 1e-12, "at {n} qubits the QFT is off by {worst}");
        }
        // The inverse really inverts it.
        for n in 1..=4usize {
            let mut round_trip = qft_circuit(n).unwrap();
            round_trip.append(&iqft(n).unwrap()).unwrap();
            let unitary = round_trip.unitary_small().unwrap();
            for i in 0..(1usize << n) {
                for j in 0..(1usize << n) {
                    let expected = f64::from(i == j);
                    assert!(
                        close(unitary[i][j].re, expected, 1e-12)
                            && close(unitary[i][j].im, 0.0, 1e-12),
                        "the round trip is not the identity at ({i}, {j})"
                    );
                }
            }
        }
        // The transform of the uniform state is the zero state, since a
        // constant function has only a zero-frequency component.
        let n = 4usize;
        let out = qft_circuit(n).unwrap().run(&QState::plus_all(n).unwrap()).unwrap();
        assert!(close(out.probability(0), 1.0, 1e-12), "the DC component is not everything");

        // And a periodic input concentrates on the multiples of N / period,
        // which is the property Shor's algorithm depends on entirely.
        let size = 1usize << n;
        let period = 4usize;
        let count = size / period;
        let amplitude = 1.0 / (count as f64).sqrt();
        let mut amps = vec![ZERO; size];
        for k in 0..count {
            amps[k * period] = Complex::new(amplitude, 0.0);
        }
        let out = qft_circuit(n).unwrap().run(&QState::from_amps(amps).unwrap()).unwrap();
        for (index, p) in out.probabilities().iter().enumerate() {
            if index % count == 0 {
                assert!(close(*p, 1.0 / period as f64, 1e-9), "peak {index} has weight {p}");
            } else {
                assert!(*p < 1e-12, "index {index} should be empty, has {p}");
            }
        }
    }

    // -----------------------------------------------------------------
    // Query algorithms
    // -----------------------------------------------------------------

    #[test]
    fn deutsch_jozsa_separates_constant_from_balanced_in_one_query() {
        for n in 1..=6usize {
            let size = 1u64 << n;
            assert!(deutsch_jozsa(&|_| false, n).unwrap(), "the zero function is constant");
            assert!(deutsch_jozsa(&|_| true, n).unwrap(), "the one function is constant");
            // Parity is balanced for every n.
            assert!(
                !deutsch_jozsa(&|x: u64| x.count_ones() % 2 == 1, n).unwrap(),
                "parity should read as balanced at {n} qubits"
            );
            // So is the top bit.
            assert!(!deutsch_jozsa(&|x: u64| x >= size / 2, n).unwrap());
            // And any function taking each value exactly half the time.
            assert!(!deutsch_jozsa(&|x: u64| x.is_multiple_of(2), n).unwrap());
        }
    }

    #[test]
    fn bernstein_vazirani_recovers_the_secret_from_one_query() {
        for n in 1..=8usize {
            for secret in 0..(1u64 << n) {
                let found = bernstein_vazirani(secret, n).unwrap();
                assert_eq!(found, secret, "at {n} qubits the secret {secret} came back {found}");
            }
        }
    }

    #[test]
    fn simon_finds_the_hidden_period() {
        let mut rng = Rng::new(0x_A17E_0001);
        for n in 2..=5usize {
            for secret in 1..(1u64 << n) {
                // A two-to-one function with exactly this period: map x to
                // min(x, x ^ s), which collides precisely on the pairs.
                let f = |x: u64| -> u64 { x.min(x ^ secret) };
                let found = simon_lite(&f, n, &mut rng).unwrap();
                assert_eq!(found, secret, "at {n} qubits the period {secret} came back {found}");
            }
        }
        assert!(simon_lite(&|x| x, 1, &mut rng).is_err());
        assert!(simon_lite(&|x| x, 13, &mut rng).is_err());
    }

    // -----------------------------------------------------------------
    // Grover
    // -----------------------------------------------------------------

    #[test]
    fn grover_finds_the_marked_item_with_high_probability() {
        let mut rng = Rng::new(0x_A17E_0002);
        for n in 3..=8usize {
            let size = 1u64 << n;
            let target = rng.next_u64() % size;
            let (found, success) = grover(&[target], n, None, &mut rng).unwrap();
            assert!(
                success > 0.9,
                "at {n} qubits the success probability is only {success}"
            );
            assert_eq!(found, target, "the measurement returned {found}, not {target}");

            // The iteration count matches the closed form.
            let expected = ((std::f64::consts::PI / 4.0) * (size as f64).sqrt() - 0.5)
                .round()
                .max(0.0) as usize;
            let reported = grover_optimal_iterations(size as usize, 1).unwrap();
            assert!(
                reported.abs_diff(expected) <= 1,
                "at {n} qubits the count is {reported}, not near {expected}"
            );
        }

        // Several marked items need proportionately fewer iterations.
        let n = 8usize;
        let marked: Vec<u64> = vec![3, 17, 200, 41];
        let (found, success) = grover(&marked, n, None, &mut rng).unwrap();
        assert!(success > 0.9, "the multi-target success is {success}");
        assert!(marked.contains(&found), "found {found}, which is not marked");
        assert!(
            grover_optimal_iterations(256, 4).unwrap() < grover_optimal_iterations(256, 1).unwrap()
        );
    }

    #[test]
    fn overshooting_grover_makes_it_worse() {
        // The least intuitive property of the algorithm, and the reason the
        // iteration count matters: the success probability oscillates rather
        // than saturating.
        let mut rng = Rng::new(0x_A17E_0003);
        let n = 8usize;
        let optimal = grover_optimal_iterations(1 << n, 1).unwrap();
        let (_, best) = grover(&[42], n, Some(optimal), &mut rng).unwrap();
        // Twice the optimal count rotates the amplitude past the target and
        // most of the way back to where it started. Which multiple lands in a
        // trough depends on the angle, so the general statement is about the
        // *oscillation*, not about any one multiple: three times the optimal
        // count happens to land near a peak again.
        let (_, overshoot) = grover(&[42], n, Some(2 * optimal), &mut rng).unwrap();
        assert!(
            overshoot < 0.05,
            "twice the iterations should nearly undo the search, gave {overshoot}"
        );
        assert!(best > 0.99, "the optimal count gives {best}");

        let sweep: Vec<f64> = (0..=(4 * optimal))
            .map(|k| grover(&[42], n, Some(k), &mut rng).unwrap().1)
            .collect();
        let peaks = sweep.windows(3).filter(|w| w[1] > w[0] && w[1] > w[2]).count();
        let dips = sweep.windows(3).filter(|w| w[1] < w[0] && w[1] < w[2]).count();
        assert!(peaks >= 2 && dips >= 1, "the probability does not oscillate: {sweep:?}");
        assert!(sweep[0] < 0.01, "zero iterations should leave it uniform");
        // The period matches the rotation angle: with one marked item in N,
        // the angle per iteration is 2 asin(sqrt(1/N)) and the probability
        // returns to zero after pi / that.
        let angle = 2.0 * (1.0 / (1u64 << n) as f64).sqrt().asin();
        let expected_period = std::f64::consts::PI / angle;
        let trough = sweep
            .iter()
            .enumerate()
            .skip(1)
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k as f64)
            .unwrap();
        assert!(
            (trough - expected_period).abs() < 2.0,
            "the first trough is at {trough}, not near {expected_period}"
        );

        assert!(grover(&[], 4, None, &mut rng).is_err());
        assert!(grover(&[99], 3, None, &mut rng).is_err());
        assert!(grover_optimal_iterations(0, 1).is_err());
        assert!(grover_optimal_iterations(4, 5).is_err());
    }

    #[test]
    fn counting_recovers_the_number_of_marked_items() {
        for (marked, n) in [(1usize, 8usize), (4, 8), (16, 8), (2, 6)] {
            let targets: Vec<u64> = (0..marked as u64).collect();
            let estimate = quantum_counting(&targets, n, 10).unwrap();
            assert!(
                (estimate - marked as f64).abs() < 0.5,
                "counted {estimate} where there are {marked}"
            );
        }
        // Coarser precision costs accuracy, which is the whole trade.
        let targets: Vec<u64> = (0..7u64).collect();
        let fine = quantum_counting(&targets, 10, 12).unwrap();
        let coarse = quantum_counting(&targets, 10, 4).unwrap();
        assert!(
            (fine - 7.0).abs() < (coarse - 7.0).abs() + 1e-9,
            "more precision did not help: {fine} against {coarse}"
        );
        assert!(quantum_counting(&[1000], 4, 8).is_err());
        assert!(quantum_counting(&[1], 4, 0).is_err());
    }

    // -----------------------------------------------------------------
    // Phase estimation and Shor
    // -----------------------------------------------------------------

    #[test]
    fn phase_estimation_is_exact_on_the_phases_it_can_represent() {
        // With m ancillas the register represents multiples of 2^-m exactly,
        // and those must come back with no error at all.
        let one = QState::basis(1, 1).unwrap();
        for ancilla in 3..=8usize {
            let resolution = 1u64 << ancilla;
            for k in 0..resolution {
                let phase = k as f64 / resolution as f64;
                let gate = Gate::phase(2.0 * std::f64::consts::PI * phase);
                let estimate = phase_estimation(&gate, &one, ancilla).unwrap();
                assert!(
                    close(estimate, phase, 1e-12),
                    "with {ancilla} ancillas the phase {phase} came back {estimate}"
                );
            }
        }
        // A phase between the representable ones is recovered to the
        // resolution, not exactly.
        let phase = 1.0 / 3.0;
        let gate = Gate::phase(2.0 * std::f64::consts::PI * phase);
        let coarse = phase_estimation(&gate, &one, 4).unwrap();
        let fine = phase_estimation(&gate, &one, 10).unwrap();
        assert!((coarse - phase).abs() <= 1.0 / 16.0 + 1e-12, "the coarse estimate is {coarse}");
        assert!((fine - phase).abs() < (coarse - phase).abs(), "more ancillas did not help");
        assert!((fine - phase).abs() < 1e-3, "the fine estimate is {fine}");

        // The eigenstate matters: |0> has eigenvalue one, so phase zero.
        let zero = QState::basis(1, 0).unwrap();
        assert!(close(phase_estimation(&gate, &zero, 6).unwrap(), 0.0, 1e-12));
        assert!(phase_estimation(&gate, &bell_state(0).unwrap(), 4).is_err());
        assert!(phase_estimation(&gate, &one, 0).is_err());
    }

    #[test]
    fn the_period_finding_subroutine_factors_fifteen() {
        // The full pipeline: find the period of a^x mod 15 quantumly, then
        // turn it into factors classically.
        let mut rng = Rng::new(0x_A17E_0004);
        let mut factored = 0usize;
        for a in [2u64, 4, 7, 8, 11, 13, 14] {
            let mut found_period = None;
            for _ in 0..40 {
                if let Some(r) = shor_period_finding_sim(a, 15, 8, &mut rng).unwrap() {
                    // Whatever it returns must genuinely be a period.
                    assert_eq!(mod_pow(a, r, 15), 1, "a = {a}: {r} is not a period");
                    found_period = Some(r);
                    break;
                }
            }
            let Some(r) = found_period else {
                panic!("a = {a}: forty attempts found no period");
            };
            if let Some((p, q)) = shor_classical_post(a, r, 15).unwrap() {
                assert_eq!(p * q, 15, "the factors {p} and {q} do not multiply to fifteen");
                assert!(p > 1 && q > 1, "a trivial factorisation of {p} and {q}");
                factored += 1;
            }
        }
        assert!(factored >= 3, "only {factored} of the bases factored fifteen");

        // Twenty-one as well, with a base whose order is four.
        let mut worked = false;
        for _ in 0..60 {
            if let Some(r) = shor_period_finding_sim(2, 21, 9, &mut rng).unwrap() {
                assert_eq!(mod_pow(2, r, 21), 1);
                if let Some((p, q)) = shor_classical_post(2, r, 21).unwrap() {
                    assert_eq!(p * q, 21);
                    worked = true;
                    break;
                }
            }
        }
        assert!(worked, "twenty-one was never factored");

        assert!(shor_period_finding_sim(1, 15, 6, &mut rng).is_err());
        assert!(shor_period_finding_sim(3, 15, 6, &mut rng).is_err());
        assert!(shor_period_finding_sim(2, 15, 2, &mut rng).is_err());
        assert!(shor_classical_post(2, 0, 15).is_err());
        // An odd period cannot be used, and the routine says so.
        assert_eq!(shor_classical_post(4, 5, 15).unwrap(), None);
    }

    // -----------------------------------------------------------------
    // Variational algorithms
    // -----------------------------------------------------------------

    #[test]
    fn the_variational_eigensolver_reaches_the_true_ground_energy_from_above() {
        // The exact diagonalisation is the reference and the bound is
        // one-sided: VQE may be high but never low.
        let hamiltonian = h2_model_hamiltonian(0.7414).unwrap();
        let exact = pauli_sum_ground_energy(&hamiltonian, 2).unwrap();
        // The construction fixes the ground eigenvalue to the Morse curve, so
        // this is exact rather than approximate -- and it checks that the
        // Pauli coefficients really do assemble into the intended operator.
        assert!(
            close(exact, h2_ground_energy_model(0.7414), 1e-9),
            "the model's ground energy is {exact}, not the curve's {}",
            h2_ground_energy_model(0.7414)
        );
        assert!(close(exact, -1.1744, 1e-9), "the equilibrium energy is {exact}");
        // The curve dissociates to -1.0 hartree, which is two free hydrogen
        // atoms, and the well depth is the measured 0.1745.
        assert!(close(h2_ground_energy_model(20.0), -1.0, 1e-6));
        assert!(close(
            h2_ground_energy_model(20.0) - h2_ground_energy_model(0.7414),
            0.1744,
            1e-6
        ));
        // At every separation the operator's ground eigenvalue tracks the
        // curve it was built from.
        for r in [0.4f64, 0.6, 0.9, 1.4, 2.5] {
            let energy = pauli_sum_ground_energy(&h2_model_hamiltonian(r).unwrap(), 2).unwrap();
            assert!(
                close(energy, h2_ground_energy_model(r), 1e-9),
                "at {r} angstrom the operator gives {energy}, the curve {}",
                h2_ground_energy_model(r)
            );
        }

        // A two-parameter ansatz that can reach the ground state.
        let ansatz = |params: &[f64]| -> Result<Circuit, GeomError> {
            let mut circuit = Circuit::new(2)?;
            circuit.x(0).ry(1, params[0]).cx(1, 0).ry(1, params[1]);
            Ok(circuit)
        };
        let (energy, params) = vqe_lite(&hamiltonian, &ansatz, &[0.1, 0.1], 2).unwrap();
        assert!(
            energy >= exact - 1e-9,
            "VQE returned {energy}, below the true ground energy {exact}"
        );
        assert!(
            close(energy, exact, 1e-6),
            "VQE returned {energy} against the exact {exact}"
        );
        assert_eq!(params.len(), 2);

        // The bond-length curve has a minimum near the equilibrium
        // separation, which is the physics the Hamiltonian encodes.
        let energies: Vec<(f64, f64)> = [0.4f64, 0.6, 0.735, 1.0, 1.5, 2.0]
            .iter()
            .map(|&r| (r, pauli_sum_ground_energy(&h2_model_hamiltonian(r).unwrap(), 2).unwrap()))
            .collect();
        let minimum = energies
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert!(
            (0.6..=1.0).contains(&minimum.0),
            "the minimum sits at {} angstrom",
            minimum.0
        );
        // The curve rises on both sides of it, steeply inward and gently
        // outward, which is what a Morse potential is.
        assert!(energies[0].1 > minimum.1 && energies[energies.len() - 1].1 > minimum.1);
        assert!(
            energies[0].1 - minimum.1 > energies[energies.len() - 1].1 - minimum.1,
            "the repulsive wall should be steeper than the tail"
        );

        assert!(h2_model_hamiltonian(-1.0).is_err());
        assert!(vqe_lite(&hamiltonian, &ansatz, &[], 2).is_err());
        assert!(vqe_lite(&[], &ansatz, &[0.1], 2).is_err());
        assert!(pauli_sum_ground_energy(&[], 2).is_err());
        assert!(pauli_sum_ground_energy(&[("XXX".into(), 1.0)], 2).is_err());
    }

    #[test]
    fn qaoa_improves_with_depth_and_never_claims_more_than_the_best_cut() {
        // The expected cut is an average over the output distribution, so it
        // cannot exceed the true maximum -- and it should rise with the layer
        // count, which is the only reason to add layers.
        // A five-cycle, whose maximum cut is four.
        let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 4), (4, 0)];
        let vertices = 5usize;
        let brute: f64 = (0..(1u64 << vertices))
            .map(|assignment| {
                edges
                    .iter()
                    .filter(|&&(a, b)| (assignment >> a & 1) != (assignment >> b & 1))
                    .count() as f64
            })
            .fold(0.0, f64::max);
        assert!(close(brute, 4.0, 1e-12), "the five-cycle's best cut is {brute}");

        let mut previous = 0.0;
        for layers in 1..=3usize {
            let (expected, params, best) = qaoa_maxcut(vertices, &edges, layers).unwrap();
            assert!(
                expected <= brute + 1e-9,
                "with {layers} layers QAOA claims {expected}, above the maximum {brute}"
            );
            assert!(
                expected > previous - 1e-6,
                "adding a layer lowered the expectation from {previous} to {expected}"
            );
            previous = expected;
            assert_eq!(params.len(), 2 * layers);
            // The most likely bit string is a genuine cut of the graph.
            let value = edges
                .iter()
                .filter(|&&(a, b)| (best >> a & 1) != (best >> b & 1))
                .count() as f64;
            assert!(value >= 3.0, "the most likely string cuts only {value} edges");
        }
        assert!(previous > 2.0, "QAOA should beat a random cut of 2.5: {previous}");

        assert!(qaoa_maxcut(1, &edges, 1).is_err());
        assert!(qaoa_maxcut(5, &[(0, 0)], 1).is_err());
        assert!(qaoa_maxcut(5, &[(0, 9)], 1).is_err());
        assert!(qaoa_maxcut(5, &edges, 0).is_err());
    }

    #[test]
    fn trotterisation_converges_to_the_exact_evolution_as_the_steps_grow() {
        // Commuting terms are exact at one step; non-commuting ones are not,
        // and the error must fall as the step count rises. Both halves are
        // checked, since a routine that ignored the ordering would pass the
        // first and fail the second.
        let commuting = vec![("ZI".to_string(), 0.7), ("IZ".to_string(), -0.4)];
        let one_step = trotter_evolution(&commuting, 1.3, 1, 2).unwrap();
        let many = trotter_evolution(&commuting, 1.3, 16, 2).unwrap();
        let a = one_step.unitary_small().unwrap();
        let b = many.unitary_small().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    close(a[i][j].re, b[i][j].re, 1e-10) && close(a[i][j].im, b[i][j].im, 1e-10),
                    "commuting terms should not need steps, disagreeing at ({i}, {j})"
                );
            }
        }

        // Non-commuting: compare against a very finely stepped reference.
        let mixed = vec![("XI".to_string(), 0.6), ("ZZ".to_string(), 0.9)];
        let reference = trotter_evolution(&mixed, 1.0, 4000, 2)
            .unwrap()
            .unitary_small()
            .unwrap();
        let mut previous = f64::INFINITY;
        for steps in [1usize, 4, 16, 64] {
            let approximate = trotter_evolution(&mixed, 1.0, steps, 2).unwrap().unitary_small().unwrap();
            let mut worst: f64 = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    worst = worst
                        .max((approximate[i][j].re - reference[i][j].re).abs())
                        .max((approximate[i][j].im - reference[i][j].im).abs());
                }
            }
            assert!(worst < previous, "the error rose at {steps} steps: {worst}");
            previous = worst;
        }
        assert!(previous < 0.02, "sixty-four steps still leave an error of {previous}");
        assert!(trotter_evolution(&mixed, 1.0, 0, 2).is_err());
        assert!(trotter_evolution(&[], 1.0, 1, 2).is_err());
        assert!(trotter_evolution(&[("XXX".into(), 1.0)], 1.0, 1, 2).is_err());
    }

    // -----------------------------------------------------------------
    // Walks, correction, benchmarking
    // -----------------------------------------------------------------

    #[test]
    fn a_quantum_walk_spreads_linearly_and_peaks_at_the_edges() {
        // Both differences from a classical walk in one test. The standard
        // deviation grows as the step count rather than its square root, and
        // the distribution is bimodal rather than a bell curve.
        let coin = Gate::h();
        let mut deviations = Vec::new();
        for steps in [10usize, 20, 40, 80] {
            let distribution = quantum_walk_line(steps, &coin).unwrap();
            let total: f64 = distribution.iter().sum();
            assert!(close(total, 1.0, 1e-9), "the walk lost probability: {total}");

            let centre = steps as f64;
            let mean: f64 = distribution
                .iter()
                .enumerate()
                .map(|(k, p)| p * (k as f64 - centre))
                .sum();
            let variance: f64 = distribution
                .iter()
                .enumerate()
                .map(|(k, p)| p * (k as f64 - centre - mean).powi(2))
                .sum();
            deviations.push((steps as f64, variance.sqrt()));

            // Bimodal: the centre is a local minimum between two peaks.
            let peak = distribution
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap();
            assert!(
                (peak as f64 - centre).abs() > 0.4 * steps as f64,
                "at {steps} steps the peak is at {peak}, near the centre {centre}"
            );
        }
        // The spread doubles as the steps double: linear, not square root.
        for pair in deviations.windows(2) {
            let ratio = pair[1].1 / pair[0].1;
            assert!(
                (1.7..2.3).contains(&ratio),
                "the spread grew by {ratio} when the steps doubled"
            );
        }
        assert!(quantum_walk_line(0, &coin).is_err());
        assert!(quantum_walk_line(10, &Gate { matrix: [[Complex::new(2.0, 0.0), ZERO], [ZERO, ZERO]] }).is_err());
    }

    #[test]
    fn the_three_bit_code_beats_the_physical_rate_below_one_half_and_not_above() {
        // The threshold in its simplest form. Below a half the encoding
        // helps; above it, redundancy makes matters worse, and that reversal
        // is the point rather than an edge case.
        let mut rng = Rng::new(0x_A17E_0005);
        for p in [0.01f64, 0.05, 0.2, 0.4] {
            let (logical, physical) =
                error_correction_3bit_flip_demo(p, 200_000, &mut rng).unwrap();
            assert!(logical < physical, "at p = {p} the code did not help: {logical} vs {physical}");
            let exact = three_bit_code_logical_error(p);
            assert!(
                (logical - exact).abs() < 0.005,
                "at p = {p} the measured rate is {logical}, the closed form {exact}"
            );
        }
        for p in [0.6f64, 0.8, 0.95] {
            let (logical, physical) =
                error_correction_3bit_flip_demo(p, 100_000, &mut rng).unwrap();
            assert!(
                logical > physical,
                "at p = {p} the code should hurt, got {logical} vs {physical}"
            );
        }
        // Exactly at a half the two coincide.
        assert!(close(three_bit_code_logical_error(0.5), 0.5, 1e-12));
        assert!(close(three_bit_code_logical_error(0.0), 0.0, 1e-15));
        assert!(close(three_bit_code_logical_error(1.0), 1.0, 1e-15));
        assert!(error_correction_3bit_flip_demo(1.5, 10, &mut rng).is_err());
        assert!(error_correction_3bit_flip_demo(0.1, 0, &mut rng).is_err());
    }

    #[test]
    fn randomised_benchmarking_decays_with_depth_at_a_rate_set_by_the_noise() {
        let mut rng = Rng::new(0x_A17E_0006);
        let depths = [1usize, 2, 4, 8, 16, 32];
        // No noise means perfect recovery whatever the depth, which is the
        // property that makes the technique insensitive to everything else.
        let clean = randomized_benchmarking_sim(&depths, 0.0, 200, &mut rng).unwrap();
        for (depth, fidelity) in &clean {
            assert!(close(*fidelity, 1.0, 1e-12), "at depth {depth} the clean fidelity is {fidelity}");
        }

        let noisy = randomized_benchmarking_sim(&depths, 0.05, 3_000, &mut rng).unwrap();
        assert_eq!(noisy.len(), depths.len());
        for pair in noisy.windows(2) {
            assert!(
                pair[1].1 <= pair[0].1 + 0.03,
                "the fidelity rose from depth {} to {}: {} against {}",
                pair[0].0,
                pair[1].0,
                pair[0].1,
                pair[1].1
            );
        }
        assert!(noisy[0].1 > noisy[noisy.len() - 1].1 + 0.1, "the decay is not visible: {noisy:?}");
        // Heavier noise decays faster.
        let heavy = randomized_benchmarking_sim(&[16usize], 0.2, 3_000, &mut rng).unwrap();
        let light = randomized_benchmarking_sim(&[16usize], 0.02, 3_000, &mut rng).unwrap();
        assert!(
            heavy[0].1 < light[0].1,
            "more noise gave a higher fidelity: {} against {}",
            heavy[0].1,
            light[0].1
        );
        assert!(randomized_benchmarking_sim(&[], 0.1, 10, &mut rng).is_err());
        assert!(randomized_benchmarking_sim(&[4], 1.5, 10, &mut rng).is_err());
        assert!(randomized_benchmarking_sim(&[4], 0.1, 0, &mut rng).is_err());
    }

    #[test]
    fn the_two_by_two_solver_solves_the_system_it_was_given() {
        // Substituting the answer back is the whole test, and it needs no
        // reference implementation.
        let cases: Vec<([[f64; 2]; 2], [f64; 2])> = vec![
            ([[2.0, 0.0], [0.0, 3.0]], [1.0, -2.0]),
            ([[1.0, 0.5], [0.5, 2.0]], [3.0, 1.0]),
            ([[4.0, -1.0], [-1.0, 4.0]], [0.0, 1.0]),
            ([[1.0, 0.0], [0.0, 1.0]], [0.7, 0.3]),
            ([[-2.0, 1.0], [1.0, -3.0]], [1.0, 1.0]),
        ];
        for (a, b) in &cases {
            let x = hhl_lite_2x2(a, b).unwrap();
            for row in 0..2 {
                let lhs = a[row][0] * x[0] + a[row][1] * x[1];
                assert!(
                    close(lhs, b[row], 1e-9),
                    "row {row} of {a:?} x = {b:?} gives {lhs}, solved as {x:?}"
                );
            }
        }
        assert!(hhl_lite_2x2(&[[1.0, 2.0], [3.0, 4.0]], &[1.0, 1.0]).is_err());
        assert!(hhl_lite_2x2(&[[1.0, 1.0], [1.0, 1.0]], &[1.0, 1.0]).is_err());
    }
}
