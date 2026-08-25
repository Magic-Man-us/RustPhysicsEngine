//! Properties of the circuit simulator and the algorithms built on it.
//!
//! A simulator has an unusually strong specification. Every gate is unitary,
//! so probability is conserved exactly and every circuit is invertible
//! exactly; the reduced states of a pure state have equal entropy across a
//! cut whichever side is traced out; and each algorithm has a promise that
//! either holds on a given instance or does not. None of these is
//! statistical, so they are checked on random instances and demanded to hold
//! every time.

use rust_physics_engine::fractals::Complex;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::quantum::algorithms::{
    bernstein_vazirani, deutsch_jozsa, grover, grover_optimal_iterations, hhl_lite_2x2, iqft,
    pauli_sum_expectation, pauli_sum_ground_energy, phase_estimation, qft_check_vs_fft,
    qft_circuit, quantum_walk_line, simon_lite, three_bit_code_logical_error,
    trotter_evolution,
};
use rust_physics_engine::quantum::circuit::{
    amplitude_damping, bell_state, bit_flip, depolarizing_channel, ghz, pauli_decompose,
    phase_damping, phase_flip, random_state, w_state, Circuit, DensityMatrix, Gate, QState,
};

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

fn spread(rng: &mut Rng, half_width: f64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * half_width
}

/// A random circuit of the given width and length, drawn from a gate set that
/// includes non-self-inverse gates -- the ones a reverse-without-adjoint bug
/// would survive.
fn random_circuit(rng: &mut Rng, n: usize, gates: usize) -> Circuit {
    let mut circuit = Circuit::new(n).unwrap();
    for _ in 0..gates {
        match pick(rng, 8) {
            0 => {
                circuit.h(pick(rng, n));
            }
            1 => {
                circuit.gate(pick(rng, n), Gate::t());
            }
            2 => {
                circuit.rx(pick(rng, n), spread(rng, 3.0));
            }
            3 => {
                circuit.ry(pick(rng, n), spread(rng, 3.0));
            }
            4 => {
                circuit.rz(pick(rng, n), spread(rng, 3.0));
            }
            5 if n >= 2 => {
                let a = pick(rng, n);
                let b = (a + 1 + pick(rng, n - 1)) % n;
                circuit.cx(a, b);
            }
            6 if n >= 2 => {
                let a = pick(rng, n);
                let b = (a + 1 + pick(rng, n - 1)) % n;
                circuit.cphase(a, b, spread(rng, 3.0));
            }
            7 if n >= 3 => {
                let a = pick(rng, n);
                let b = (a + 1 + pick(rng, n - 1)) % n;
                let c = (0..n).find(|&q| q != a && q != b).unwrap();
                circuit.ccx(a, b, c);
            }
            _ => {
                circuit.x(pick(rng, n));
            }
        }
    }
    circuit
}

// ---------------------------------------------------------------------------
// Simulator invariants
// ---------------------------------------------------------------------------

#[test]
fn prop_every_circuit_preserves_the_norm_and_is_exactly_invertible() {
    // Unitarity means the norm never moves and the inverse circuit returns
    // the original amplitudes -- not approximately, but to rounding, since
    // every step is a rotation.
    let mut rng = Rng::new(0x_C111_0001);
    for _ in 0..200 {
        let n = 1 + pick(&mut rng, 4);
        let gates = 5 + pick(&mut rng, 25);
        let circuit = random_circuit(&mut rng, n, gates);
        let start = random_state(n, &mut rng).unwrap();

        let out = circuit.run(&start).unwrap();
        assert!(
            (out.norm() - 1.0).abs() < 1e-12,
            "the norm became {} after {} gates",
            out.norm(),
            circuit.gate_count()
        );

        let back = circuit.inverse().run(&out).unwrap();
        for (a, b) in back.amps.iter().zip(&start.amps) {
            assert!(
                (a.re - b.re).abs() < 1e-11 && (a.im - b.im).abs() < 1e-11,
                "the inverse did not restore the state"
            );
        }
        // The probabilities are a distribution.
        let total: f64 = out.probabilities().iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "the probabilities sum to {total}");
        assert!(out.probabilities().iter().all(|p| *p >= 0.0));
    }
}

#[test]
fn prop_the_matrix_and_the_simulation_agree_on_every_random_circuit() {
    // Two routes through the same circuit: gate by gate on the amplitudes,
    // and once through the assembled unitary. They exercise entirely
    // different index arithmetic and must give the same answer.
    let mut rng = Rng::new(0x_C111_0002);
    for _ in 0..80 {
        let n = 1 + pick(&mut rng, 3);
        let size = 1usize << n;
        let gates = 4 + pick(&mut rng, 12);
        let circuit = random_circuit(&mut rng, n, gates);
        let unitary = circuit.unitary_small().unwrap();

        // The matrix is unitary.
        for i in 0..size {
            for j in 0..size {
                let entry = (0..size).fold(Complex::new(0.0, 0.0), |acc, k| {
                    acc + unitary[k][i].conjugate() * unitary[k][j]
                });
                let expected = f64::from(i == j);
                assert!(
                    (entry.re - expected).abs() < 1e-11 && entry.im.abs() < 1e-11,
                    "the columns are not orthonormal at ({i}, {j})"
                );
            }
        }

        let state = random_state(n, &mut rng).unwrap();
        let simulated = circuit.run(&state).unwrap();
        for row in 0..size {
            let expected = (0..size)
                .fold(Complex::new(0.0, 0.0), |acc, k| acc + unitary[row][k] * state.amps[k]);
            assert!(
                (simulated.amps[row].re - expected.re).abs() < 1e-11
                    && (simulated.amps[row].im - expected.im).abs() < 1e-11,
                "the matrix and the run disagree at row {row}"
            );
        }
    }
}

#[test]
fn prop_entanglement_entropy_is_symmetric_across_every_cut() {
    // A pure state's two reduced states have the same spectrum whichever side
    // is traced out, so the entropy is a property of the cut. That is a real
    // theorem and it is easy to violate with an indexing error in the partial
    // trace, which is why it is worth checking on random states.
    let mut rng = Rng::new(0x_C111_0003);
    for _ in 0..120 {
        let n = 2 + pick(&mut rng, 3);
        let state = random_state(n, &mut rng).unwrap();
        // A random non-trivial subset.
        let mut left: Vec<usize> = Vec::new();
        for q in 0..n {
            if rng.next_f64() < 0.5 {
                left.push(q);
            }
        }
        if left.is_empty() || left.len() == n {
            continue;
        }
        let right: Vec<usize> = (0..n).filter(|q| !left.contains(q)).collect();

        let a = state.entanglement_entropy(&left).unwrap();
        let b = state.entanglement_entropy(&right).unwrap();
        assert!(
            (a - b).abs() < 1e-7,
            "the cut {left:?} gives {a} and its complement {b}"
        );
        // Bounded by the smaller side's qubit count.
        let bound = left.len().min(right.len()) as f64;
        assert!(a <= bound + 1e-7, "the entropy {a} exceeds {bound} bits");
        assert!(a >= -1e-9, "the entropy is negative: {a}");

        // The reduced state is a valid density matrix.
        let rho = DensityMatrix {
            n: left.len(),
            rho: state.reduced_density_matrix(&left).unwrap(),
        };
        assert!(rho.is_valid(1e-8), "the reduced state is not a state");
        // Purity and entropy agree on which states are pure.
        if a < 1e-8 {
            assert!((rho.purity() - 1.0).abs() < 1e-6, "zero entropy but purity {}", rho.purity());
        } else {
            assert!(rho.purity() < 1.0 - 1e-9, "positive entropy but purity {}", rho.purity());
        }
    }
}

#[test]
fn prop_a_product_state_has_no_entanglement_however_it_is_built() {
    // The converse of the previous test: states assembled as tensor products
    // must show exactly zero across the cut that separates the factors, and
    // their Bloch vectors must have unit length.
    let mut rng = Rng::new(0x_C111_0004);
    for _ in 0..150 {
        let n = 2 + pick(&mut rng, 3);
        let mut circuit = Circuit::new(n).unwrap();
        // Only one-qubit gates, so the state cannot entangle.
        for q in 0..n {
            circuit.ry(q, spread(&mut rng, 3.0));
            circuit.rz(q, spread(&mut rng, 3.0));
            circuit.gate(q, Gate::t());
        }
        let state = circuit.run(&QState::zero(n).unwrap()).unwrap();
        for q in 0..n {
            assert!(
                state.entanglement_entropy(&[q]).unwrap() < 1e-9,
                "a product state has entropy {}",
                state.entanglement_entropy(&[q]).unwrap()
            );
            let (x, y, z) = state.bloch_vector(q).unwrap();
            assert!(
                (x.hypot(y).hypot(z) - 1.0).abs() < 1e-9,
                "an unentangled qubit has Bloch length {}",
                x.hypot(y).hypot(z)
            );
        }
    }
}

#[test]
fn prop_pauli_expectations_are_bounded_and_reconstruct_the_state() {
    // Every Pauli expectation lies in [-1, 1] because the operators square to
    // the identity. On one qubit the three of them are the Bloch vector, and
    // their squares sum to at most one -- with equality exactly for a pure
    // qubit.
    let mut rng = Rng::new(0x_C111_0005);
    for _ in 0..200 {
        let single = random_state(1, &mut rng).unwrap();
        let x = single.expectation_pauli_string("X").unwrap();
        let y = single.expectation_pauli_string("Y").unwrap();
        let z = single.expectation_pauli_string("Z").unwrap();
        for value in [x, y, z] {
            assert!((-1.0..=1.0).contains(&value), "an expectation is {value}");
        }
        assert!(
            (x * x + y * y + z * z - 1.0).abs() < 1e-9,
            "a pure qubit's Bloch vector has length squared {}",
            x * x + y * y + z * z
        );
        let (bx, by, bz) = single.bloch_vector(0).unwrap();
        assert!((bx - x).abs() < 1e-12 && (by - y).abs() < 1e-12 && (bz - z).abs() < 1e-12);

        // On more qubits the bound still holds for every string.
        let n = 2 + pick(&mut rng, 2);
        let state = random_state(n, &mut rng).unwrap();
        let symbols = ['I', 'X', 'Y', 'Z'];
        for _ in 0..20 {
            let name: String = (0..n).map(|_| symbols[pick(&mut rng, 4)]).collect();
            let value = state.expectation_pauli_string(&name).unwrap();
            assert!((-1.0 - 1e-12..=1.0 + 1e-12).contains(&value), "{name} gives {value}");
        }
    }
}

#[test]
fn prop_every_channel_is_trace_preserving_and_never_increases_purity() {
    // A channel is physical exactly when it preserves the trace, and a
    // unital or damping channel cannot make a state purer than it was. Both
    // are exact statements about the output.
    let mut rng = Rng::new(0x_C111_0006);
    for _ in 0..200 {
        let state = random_state(1, &mut rng).unwrap();
        let start = DensityMatrix::from_state(&state);
        let p = rng.next_f64();
        for (name, kraus) in [
            ("depolarizing", depolarizing_channel(p).unwrap()),
            ("amplitude", amplitude_damping(p).unwrap()),
            ("phase", phase_damping(p).unwrap()),
            ("bitflip", bit_flip(p).unwrap()),
            ("phaseflip", phase_flip(p).unwrap()),
        ] {
            let mut rho = start.clone();
            rho.apply_channel(&kraus).unwrap();
            let trace = rho.trace();
            assert!(
                (trace.re - 1.0).abs() < 1e-12 && trace.im.abs() < 1e-12,
                "{name} at p = {p} left trace {trace:?}"
            );
            assert!(rho.is_valid(1e-9), "{name} at p = {p} produced an invalid state");
            assert!(
                rho.purity() <= start.purity() + 1e-9,
                "{name} at p = {p} raised the purity to {}",
                rho.purity()
            );
            assert!(
                rho.von_neumann_entropy().unwrap() >= -1e-9,
                "{name} gave a negative entropy"
            );
            // Applying it twice is still a state, so the channel composes.
            let mut again = rho.clone();
            again.apply_channel(&kraus).unwrap();
            assert!(again.is_valid(1e-9), "{name} does not compose");
        }
    }
}

#[test]
fn prop_pauli_decomposition_is_exact_and_its_coefficients_are_real() {
    // Every Hermitian matrix has a real Pauli expansion, and summing the
    // terms must give the matrix back. The coefficients are inner products in
    // an orthogonal basis, so nothing is approximate here.
    let mut rng = Rng::new(0x_C111_0007);
    for _ in 0..150 {
        let n = 1 + pick(&mut rng, 2);
        let size = 1usize << n;
        // A random Hermitian matrix.
        let raw: Vec<Vec<Complex>> = (0..size)
            .map(|_| {
                (0..size)
                    .map(|_| Complex::new(spread(&mut rng, 2.0), spread(&mut rng, 2.0)))
                    .collect()
            })
            .collect();
        let h: Vec<Vec<Complex>> = (0..size)
            .map(|i| {
                (0..size)
                    .map(|j| {
                        let s = raw[i][j] + raw[j][i].conjugate();
                        Complex::new(s.re * 0.5, s.im * 0.5)
                    })
                    .collect()
            })
            .collect();

        let terms = pauli_decompose(&h).unwrap();
        let mut rebuilt = vec![vec![Complex::new(0.0, 0.0); size]; size];
        for (name, coefficient) in &terms {
            for i in 0..size {
                for j in 0..size {
                    let mut entry = Complex::new(1.0, 0.0);
                    for (k, symbol) in name.chars().enumerate() {
                        let gate = match symbol {
                            'X' => Gate::x(),
                            'Y' => Gate::y(),
                            'Z' => Gate::z(),
                            _ => Gate::identity(),
                        };
                        let row = (i >> (n - 1 - k)) & 1;
                        let column = (j >> (n - 1 - k)) & 1;
                        entry = entry * gate.matrix[row][column];
                    }
                    rebuilt[i][j] = rebuilt[i][j]
                        + Complex::new(entry.re * coefficient, entry.im * coefficient);
                }
            }
        }
        for i in 0..size {
            for j in 0..size {
                assert!(
                    (rebuilt[i][j].re - h[i][j].re).abs() < 1e-9
                        && (rebuilt[i][j].im - h[i][j].im).abs() < 1e-9,
                    "the decomposition does not rebuild entry ({i}, {j})"
                );
            }
        }

        // And the expectation of the sum matches the expectation of the
        // matrix, taken directly.
        let state = random_state(n, &mut rng).unwrap();
        let from_terms = pauli_sum_expectation(&terms, &state).unwrap();
        let mut direct = 0.0;
        for i in 0..size {
            for j in 0..size {
                let contribution = state.amps[i].conjugate() * h[i][j] * state.amps[j];
                direct += contribution.re;
            }
        }
        assert!(
            (from_terms - direct).abs() < 1e-9,
            "the two expectations are {from_terms} and {direct}"
        );
    }
}

// ---------------------------------------------------------------------------
// Algorithms
// ---------------------------------------------------------------------------

#[test]
fn prop_the_qft_matches_the_dft_at_every_width_and_inverts_itself() {
    for n in 1..=6usize {
        assert!(
            qft_check_vs_fft(n).unwrap() < 1e-11,
            "at {n} qubits the QFT is off by {}",
            qft_check_vs_fft(n).unwrap()
        );
        let mut round_trip = qft_circuit(n).unwrap();
        round_trip.append(&iqft(n).unwrap()).unwrap();
        let mut rng = Rng::new(0x_C111_0008 + n as u64);
        for _ in 0..20 {
            let state = random_state(n, &mut rng).unwrap();
            let back = round_trip.run(&state).unwrap();
            for (a, b) in back.amps.iter().zip(&state.amps) {
                assert!(
                    (a.re - b.re).abs() < 1e-11 && (a.im - b.im).abs() < 1e-11,
                    "the QFT round trip moved a state at {n} qubits"
                );
            }
        }
    }
}

#[test]
fn prop_deutsch_jozsa_and_bernstein_vazirani_answer_correctly_on_random_promises() {
    // Every balanced function must read as balanced and every constant one as
    // constant, with no exceptions -- the algorithm is not probabilistic.
    let mut rng = Rng::new(0x_C111_0009);
    for n in 1..=6usize {
        let size = 1u64 << n;
        for _ in 0..30 {
            // A random balanced function: shuffle half the inputs to true.
            let mut values: Vec<bool> = (0..size).map(|i| i < size / 2).collect();
            for i in (1..values.len()).rev() {
                values.swap(i, pick(&mut rng, i + 1));
            }
            let balanced = |x: u64| values[x as usize];
            assert!(
                !deutsch_jozsa(&balanced, n).unwrap(),
                "a balanced function at {n} qubits read as constant"
            );
        }
        assert!(deutsch_jozsa(&|_| true, n).unwrap());
        assert!(deutsch_jozsa(&|_| false, n).unwrap());

        // Bernstein-Vazirani is exact for every secret.
        for _ in 0..40 {
            let secret = rng.next_u64() % size;
            assert_eq!(bernstein_vazirani(secret, n).unwrap(), secret);
        }
    }
}

#[test]
fn prop_simon_recovers_every_period_it_is_given() {
    let mut rng = Rng::new(0x_C111_000A);
    for n in 2..=5usize {
        for _ in 0..12 {
            let secret = 1 + rng.next_u64() % ((1u64 << n) - 1);
            let f = |x: u64| -> u64 { x.min(x ^ secret) };
            let found = simon_lite(&f, n, &mut rng).unwrap();
            assert_eq!(found, secret, "at {n} qubits the period {secret} came back {found}");
        }
    }
}

#[test]
fn prop_grover_succeeds_with_high_probability_for_every_marked_set() {
    // The success probability at the optimal iteration count is
    // sin^2((2k + 1) theta), which for a small marked fraction is close to
    // one. That is exact trigonometry, so it is checked against the formula
    // rather than against a threshold alone.
    let mut rng = Rng::new(0x_C111_000B);
    for n in 3..=8usize {
        let size = 1usize << n;
        for _ in 0..20 {
            let count = 1 + pick(&mut rng, 4.min(size / 4));
            let mut marked: Vec<u64> = Vec::new();
            while marked.len() < count {
                let candidate = rng.next_u64() % size as u64;
                if !marked.contains(&candidate) {
                    marked.push(candidate);
                }
            }
            let iterations = grover_optimal_iterations(size, marked.len()).unwrap();
            let (_, success) = grover(&marked, n, None, &mut rng).unwrap();

            // The measurement is a draw, not a guarantee: at a few marked
            // items out of sixteen the optimal count still leaves several per
            // cent on the unmarked states. So the frequency is checked
            // against the reported probability rather than the single draw
            // being demanded to succeed.
            let trials = 400usize;
            let hits = (0..trials)
                .filter(|_| {
                    let (drawn, _) = grover(&marked, n, None, &mut rng).unwrap();
                    marked.contains(&drawn)
                })
                .count();
            let observed = hits as f64 / trials as f64;
            assert!(
                (observed - success).abs() < 5.0 / (trials as f64).sqrt(),
                "at {n} qubits the marked states came up {observed} against the stated {success}"
            );

            let theta = (marked.len() as f64 / size as f64).sqrt().asin();
            let predicted = ((2 * iterations + 1) as f64 * theta).sin().powi(2);
            assert!(
                (success - predicted).abs() < 1e-9,
                "the success probability is {success}, the formula gives {predicted}"
            );
            assert!(success > 0.8, "at {n} qubits with {count} marked, success is {success}");
        }
    }
}

#[test]
fn prop_phase_estimation_is_exact_on_representable_phases_and_close_otherwise() {
    let one = QState::basis(1, 1).unwrap();
    let mut rng = Rng::new(0x_C111_000C);
    for ancilla in 3..=8usize {
        let resolution = 1u64 << ancilla;
        for _ in 0..20 {
            let k = rng.next_u64() % resolution;
            let phase = k as f64 / resolution as f64;
            let gate = Gate::phase(2.0 * std::f64::consts::PI * phase);
            let estimate = phase_estimation(&gate, &one, ancilla).unwrap();
            assert!(
                (estimate - phase).abs() < 1e-11,
                "the representable phase {phase} came back {estimate}"
            );
        }
        for _ in 0..20 {
            let phase = rng.next_f64();
            let gate = Gate::phase(2.0 * std::f64::consts::PI * phase);
            let estimate = phase_estimation(&gate, &one, ancilla).unwrap();
            // Correct to within one step of the register, allowing for the
            // wrap at one.
            let error = (estimate - phase).abs().min(1.0 - (estimate - phase).abs());
            assert!(
                error <= 1.0 / resolution as f64 + 1e-9,
                "with {ancilla} ancillas the phase {phase} came back {estimate}"
            );
        }
    }
}

#[test]
fn prop_trotterisation_is_unitary_and_converges_with_the_step_count() {
    // Whatever the terms and however coarse the stepping, the circuit is
    // still unitary -- Trotter error changes the answer, not its normalisation.
    let mut rng = Rng::new(0x_C111_000D);
    let symbols = ['I', 'X', 'Y', 'Z'];
    for _ in 0..60 {
        let n = 1 + pick(&mut rng, 2);
        let count = 1 + pick(&mut rng, 3);
        let terms: Vec<(String, f64)> = (0..count)
            .map(|_| {
                let name: String = (0..n).map(|_| symbols[pick(&mut rng, 4)]).collect();
                (name, spread(&mut rng, 1.5))
            })
            .collect();
        let t = spread(&mut rng, 2.0);

        let reference = trotter_evolution(&terms, t, 2000, n)
            .unwrap()
            .unitary_small()
            .unwrap();
        let size = 1usize << n;
        let mut previous = f64::INFINITY;
        for steps in [1usize, 8, 64] {
            let circuit = trotter_evolution(&terms, t, steps, n).unwrap();
            let unitary = circuit.unitary_small().unwrap();
            // Unitary at every step count.
            for i in 0..size {
                for j in 0..size {
                    let entry = (0..size).fold(Complex::new(0.0, 0.0), |acc, k| {
                        acc + unitary[k][i].conjugate() * unitary[k][j]
                    });
                    let expected = f64::from(i == j);
                    assert!(
                        (entry.re - expected).abs() < 1e-10 && entry.im.abs() < 1e-10,
                        "the Trotter circuit is not unitary at {steps} steps"
                    );
                }
            }
            let mut worst: f64 = 0.0;
            for i in 0..size {
                for j in 0..size {
                    worst = worst
                        .max((unitary[i][j].re - reference[i][j].re).abs())
                        .max((unitary[i][j].im - reference[i][j].im).abs());
                }
            }
            assert!(
                worst <= previous + 1e-9,
                "the error rose from {previous} to {worst} at {steps} steps"
            );
            previous = worst;
        }
    }
}

#[test]
fn prop_the_two_by_two_solver_satisfies_the_system_it_solves() {
    // Substitution is the certificate, and it needs no reference solver.
    let mut rng = Rng::new(0x_C111_000E);
    let mut solved = 0usize;
    for _ in 0..500 {
        let d0 = spread(&mut rng, 4.0);
        let d1 = spread(&mut rng, 4.0);
        let off = spread(&mut rng, 3.0);
        let a = [[d0, off], [off, d1]];
        let b = [spread(&mut rng, 3.0), spread(&mut rng, 3.0)];
        let Ok(x) = hhl_lite_2x2(&a, &b) else {
            continue;
        };
        solved += 1;
        let determinant = d0 * d1 - off * off;
        let magnitude = x[0].abs().max(x[1].abs()).max(1.0);
        for row in 0..2 {
            let lhs = a[row][0] * x[0] + a[row][1] * x[1];
            assert!(
                (lhs - b[row]).abs() < 1e-6 * magnitude / determinant.abs().clamp(1e-3, 1.0),
                "row {row} of {a:?} x = {b:?} gives {lhs}, solved as {x:?}"
            );
        }
    }
    assert!(solved > 400, "only {solved} systems were solvable");
    // A non-symmetric matrix is refused rather than silently symmetrised.
    assert!(hhl_lite_2x2(&[[1.0, 2.0], [3.0, 1.0]], &[1.0, 1.0]).is_err());
}

#[test]
fn prop_the_quantum_walk_conserves_probability_for_every_coin() {
    // The coin only has to be unitary; the walk is then unitary too, whatever
    // bias the coin has. A biased coin shifts the distribution without
    // leaking any of it.
    let mut rng = Rng::new(0x_C111_000F);
    for _ in 0..80 {
        let coin = Gate::u3(spread(&mut rng, 3.0), spread(&mut rng, 3.0), spread(&mut rng, 3.0));
        let steps = 5 + pick(&mut rng, 40);
        let distribution = quantum_walk_line(steps, &coin).unwrap();
        assert_eq!(distribution.len(), 2 * steps + 1);
        let total: f64 = distribution.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "the walk lost probability: {total}");
        assert!(distribution.iter().all(|p| *p >= 0.0));
        // The walker cannot outrun one site per step.
        assert!(distribution[0] >= 0.0 && distribution[2 * steps] >= 0.0);
        // Parity: the walker moves exactly one site per step from the middle
        // of the array, so after any number of steps it sits on an even
        // index -- the parity of the *offset* matches the step count, and the
        // start is itself at index `steps`.
        for (site, p) in distribution.iter().enumerate() {
            if site % 2 == 1 {
                assert!(*p < 1e-15, "site {site} is occupied against parity: {p}");
            }
        }
        assert!(distribution.iter().step_by(2).sum::<f64>() > 0.999);
    }
}

#[test]
fn prop_the_three_bit_code_helps_below_a_half_and_hurts_above_it() {
    // The threshold, as a closed form rather than a simulation: the logical
    // rate crosses the physical one at exactly one half, and nowhere else.
    for k in 1..500 {
        let p = k as f64 / 1000.0;
        assert!(
            three_bit_code_logical_error(p) < p,
            "at p = {p} the code should help"
        );
    }
    for k in 501..1000 {
        let p = k as f64 / 1000.0;
        assert!(
            three_bit_code_logical_error(p) > p,
            "at p = {p} the code should hurt"
        );
    }
    assert!((three_bit_code_logical_error(0.5) - 0.5).abs() < 1e-12);
    // And it is monotone, as more physical error can only mean more logical.
    let mut previous = -1.0;
    for k in 0..=1000 {
        let value = three_bit_code_logical_error(k as f64 / 1000.0);
        assert!(value >= previous - 1e-12, "the logical rate fell at p = {}", k as f64 / 1000.0);
        previous = value;
    }
}

#[test]
fn prop_ground_energies_bound_every_expectation_of_the_same_hamiltonian() {
    // The variational principle again, used as a test: no state's expectation
    // may fall below the lowest eigenvalue, on any Pauli-sum Hamiltonian.
    let mut rng = Rng::new(0x_C111_0010);
    let symbols = ['I', 'X', 'Y', 'Z'];
    for _ in 0..80 {
        let n = 1 + pick(&mut rng, 2);
        let count = 1 + pick(&mut rng, 4);
        let terms: Vec<(String, f64)> = (0..count)
            .map(|_| {
                let name: String = (0..n).map(|_| symbols[pick(&mut rng, 4)]).collect();
                (name, spread(&mut rng, 2.0))
            })
            .collect();
        let ground = pauli_sum_ground_energy(&terms, n).unwrap();
        for _ in 0..30 {
            let state = random_state(n, &mut rng).unwrap();
            let energy = pauli_sum_expectation(&terms, &state).unwrap();
            assert!(
                energy >= ground - 1e-8,
                "a state reached {energy}, below the ground energy {ground}"
            );
        }
        // And the standard entangled states are ordinary states too.
        if n == 2 {
            for candidate in [bell_state(0).unwrap(), ghz(2).unwrap(), w_state(2).unwrap()] {
                let energy = pauli_sum_expectation(&terms, &candidate).unwrap();
                assert!(energy >= ground - 1e-8, "a named state reached {energy}");
            }
        }
    }
}
