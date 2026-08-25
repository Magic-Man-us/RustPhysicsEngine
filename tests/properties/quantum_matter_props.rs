//! Properties of the spin and solid-state modules.
//!
//! Both halves are unusually well specified. A spin operator set is defined
//! by its commutators, and any implementation either satisfies them or is
//! wrong; a Lanczos ground state is certified by its own residual, needing no
//! reference; a tight-binding chain's spectrum is a closed form; and the
//! occupation functions have exact symmetries and limits. So these tests
//! check identities on random instances rather than comparing against stored
//! numbers.

use rust_physics_engine::fractals::Complex;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::quantum::solid_state::{
    bcs_gap_equation, bose_einstein, conductance_landauer, debye_heat_capacity,
    density_of_states_1d_free, density_of_states_2d_free, density_of_states_3d_free,
    dos_from_bands, effective_mass_from_band, einstein_heat_capacity, fermi_dirac,
    graphene_dispersion, kronig_penney, phonon_dispersion_1d_diatomic,
    phonon_dispersion_1d_monatomic, ssh_edge_states, ssh_model, ssh_winding_number,
    tight_binding_1d, tight_binding_band_1d,
};
use rust_physics_engine::quantum::spin::{
    ising_transverse_field_apply, ising_transverse_field_exact, lanczos, larmor_precession,
    magnon_dispersion, rabi_oscillation, spin_coherent_state, spin_operators, SpinChain,
};

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

fn spread(rng: &mut Rng, half_width: f64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * half_width
}

fn matmul(a: &[Vec<Complex>], b: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
    let n = a.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (0..n).fold(Complex::new(0.0, 0.0), |acc, k| acc + a[i][k] * b[k][j])
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Spin
// ---------------------------------------------------------------------------

#[test]
fn prop_the_spin_algebra_holds_at_every_representation() {
    // The commutators define angular momentum; a matrix set that satisfies
    // them is a representation and one that does not is not, whatever else it
    // gets right.
    for twice in 1..=16usize {
        let s = twice as f64 / 2.0;
        let (sx, sy, sz) = spin_operators(s).unwrap();
        let dim = twice + 1;
        let commutator = |a: &[Vec<Complex>], b: &[Vec<Complex>]| -> Vec<Vec<Complex>> {
            let ab = matmul(a, b);
            let ba = matmul(b, a);
            (0..dim)
                .map(|i| (0..dim).map(|j| ab[i][j] - ba[i][j]).collect())
                .collect()
        };
        for (a, b, c) in [(&sx, &sy, &sz), (&sy, &sz, &sx), (&sz, &sx, &sy)] {
            let bracket = commutator(a, b);
            for i in 0..dim {
                for j in 0..dim {
                    let expected = Complex::new(0.0, 1.0) * c[i][j];
                    assert!(
                        (bracket[i][j].re - expected.re).abs() < 1e-9
                            && (bracket[i][j].im - expected.im).abs() < 1e-9,
                        "the algebra fails at s = {s}, entry ({i}, {j})"
                    );
                }
            }
        }
        // The Casimir is s(s + 1) on every state.
        let square = {
            let xx = matmul(&sx, &sx);
            let yy = matmul(&sy, &sy);
            let zz = matmul(&sz, &sz);
            (0..dim)
                .map(|i| (0..dim).map(|j| xx[i][j] + yy[i][j] + zz[i][j]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { s * (s + 1.0) } else { 0.0 };
                assert!(
                    (square[i][j].re - expected).abs() < 1e-8 && square[i][j].im.abs() < 1e-9,
                    "the Casimir is wrong at s = {s}"
                );
            }
        }
    }
}

#[test]
fn prop_a_coherent_state_points_along_its_own_angles() {
    // The expectation is exactly `s` times the unit vector asked for, at any
    // spin and any direction -- which is the defining property, and the one
    // an error in the binomial weights would break.
    let mut rng = Rng::new(0x_5A11_0001);
    for _ in 0..300 {
        let twice = 1 + pick(&mut rng, 12);
        let s = twice as f64 / 2.0;
        let theta = rng.next_f64() * std::f64::consts::PI;
        let phi = spread(&mut rng, std::f64::consts::PI);
        let state = spin_coherent_state(s, theta, phi).unwrap();
        let (sx, sy, sz) = spin_operators(s).unwrap();

        let norm: f64 = state.iter().map(|z| z.norm_sq()).sum();
        assert!((norm - 1.0).abs() < 1e-9, "the state has norm {norm}");

        let expectation = |m: &[Vec<Complex>]| -> f64 {
            let mut total = Complex::new(0.0, 0.0);
            for i in 0..state.len() {
                for j in 0..state.len() {
                    total = total + state[i].conjugate() * m[i][j] * state[j];
                }
            }
            total.re
        };
        let (x, y, z) = (expectation(&sx), expectation(&sy), expectation(&sz));
        assert!(
            (x - s * theta.sin() * phi.cos()).abs() < 1e-7
                && (y - s * theta.sin() * phi.sin()).abs() < 1e-7
                && (z - s * theta.cos()).abs() < 1e-7,
            "s = {s} at ({theta}, {phi}) points at ({x}, {y}, {z})"
        );
        assert!((x.hypot(y).hypot(z) - s).abs() < 1e-7, "the length is not s");
    }
}

#[test]
fn prop_lanczos_returns_certified_eigenpairs_on_random_chains() {
    // The residual is the certificate. It needs no dense diagonalisation, so
    // it works at chain lengths where one would be impossible, and it cannot
    // be satisfied by an accidentally plausible answer.
    let mut rng = Rng::new(0x_5A11_0002);
    for _ in 0..40 {
        let n = 4 + pick(&mut rng, 5);
        let chain = SpinChain::new(
            n,
            spread(&mut rng, 2.0),
            spread(&mut rng, 2.0),
            spread(&mut rng, 1.0),
            rng.next_f64() < 0.5,
        )
        .unwrap();
        let (energy, state) = chain.ground_state_lanczos(70, &mut rng).unwrap();

        let norm: f64 = state.iter().map(|z| z.norm_sq()).sum();
        assert!((norm - 1.0).abs() < 1e-9, "the state has norm {norm}");
        let applied = chain.apply(&state).unwrap();
        let mut residual: f64 = 0.0;
        for (a, b) in applied.iter().zip(&state) {
            residual = residual.max((*a - Complex::new(b.re * energy, b.im * energy)).norm());
        }
        assert!(residual < 1e-6, "the residual is {residual} at n = {n}");

        // The Rayleigh quotient of any other state is at least the reported
        // energy -- the variational principle, used to certify a minimum.
        for _ in 0..10 {
            let trial: Vec<Complex> = (0..state.len())
                .map(|_| Complex::new(spread(&mut rng, 1.0), spread(&mut rng, 1.0)))
                .collect();
            let weight: f64 = trial.iter().map(|z| z.norm_sq()).sum();
            if weight <= 0.0 {
                continue;
            }
            let image = chain.apply(&trial).unwrap();
            let quotient: f64 = trial
                .iter()
                .zip(&image)
                .map(|(a, b)| (a.conjugate() * *b).re)
                .sum::<f64>()
                / weight;
            assert!(
                quotient >= energy - 1e-7,
                "a trial state reached {quotient}, below the ground energy {energy}"
            );
        }

        // The entanglement entropy of any cut is between zero and the
        // smaller side's size.
        for cut in 1..n {
            let entropy = chain.entanglement_entropy_cut(&state, cut).unwrap();
            assert!(entropy >= -1e-9, "a negative entropy at cut {cut}");
            assert!(
                entropy <= cut.min(n - cut) as f64 + 1e-7,
                "the entropy {entropy} exceeds the cut's capacity"
            );
        }
        // The magnetisation is a spin per site, and the self-correlation is
        // one quarter whatever the state.
        let m = chain.magnetization(&state).unwrap();
        assert!((-0.5 - 1e-9..=0.5 + 1e-9).contains(&m), "the magnetisation is {m}");
        for i in 0..n {
            assert!(
                (chain.correlation(&state, i, i).unwrap() - 0.25).abs() < 1e-9,
                "a spin does not correlate with itself"
            );
        }
    }
}

#[test]
fn prop_krylov_evolution_is_unitary_at_every_step_size() {
    // Unitarity is what the Krylov step buys, and it holds whatever the step:
    // the projection is Hermitian, so its exponential is a rotation.
    let mut rng = Rng::new(0x_5A11_0003);
    for _ in 0..30 {
        let n = 4 + pick(&mut rng, 3);
        let chain = SpinChain::new(
            n,
            spread(&mut rng, 2.0),
            spread(&mut rng, 2.0),
            spread(&mut rng, 1.0),
            false,
        )
        .unwrap();
        let size = 1usize << n;
        let mut state: Vec<Complex> = (0..size)
            .map(|_| Complex::new(spread(&mut rng, 1.0), spread(&mut rng, 1.0)))
            .collect();
        let magnitude: f64 = state.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
        for z in &mut state {
            *z = Complex::new(z.re / magnitude, z.im / magnitude);
        }
        let energy_of = |v: &[Complex]| -> f64 {
            let applied = chain.apply(v).unwrap();
            v.iter()
                .zip(&applied)
                .map(|(a, b)| (a.conjugate() * *b).re)
                .sum::<f64>()
                / v.iter().map(|z| z.norm_sq()).sum::<f64>()
        };
        let initial = energy_of(&state);

        for (t, steps) in [(0.05f64, 1usize), (1.0, 10), (7.0, 30)] {
            let moved = chain.time_evolve_krylov(&state, t, steps).unwrap();
            let norm: f64 = moved.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-8, "at t = {t} the norm is {norm}");
            assert!(
                (energy_of(&moved) - initial).abs() < 1e-6 * (1.0 + initial.abs()),
                "at t = {t} the energy moved from {initial} to {}",
                energy_of(&moved)
            );
        }
    }
}

#[test]
fn prop_the_ising_chain_matches_its_free_fermion_energy_at_every_field() {
    // The Jordan-Wigner solution is exact, so agreement is demanded at every
    // field including the critical one, where the gap closes and an
    // iterative method has the hardest time.
    let mut rng = Rng::new(0x_5A11_0004);
    for n in [4usize, 6, 8, 10] {
        for k in 0..12usize {
            let g = 0.05 + 0.3 * k as f64;
            let matvec = |v: &[Complex]| ising_transverse_field_apply(n, g, true, v).unwrap();
            let (values, _) = lanczos(&matvec, 1usize << n, 80, &mut rng).unwrap();
            let exact = ising_transverse_field_exact(n, g).unwrap();
            assert!(
                (values[0] - exact).abs() < 1e-6 * (1.0 + exact.abs()),
                "n = {n}, g = {g}: {} against {exact}",
                values[0]
            );
            // The ground energy falls as the field rises, since the field
            // term can only help.
            if k > 0 {
                let previous = ising_transverse_field_exact(n, 0.05 + 0.3 * (k - 1) as f64).unwrap();
                assert!(exact < previous, "the energy rose at g = {g}");
            }
        }
    }
}

#[test]
fn prop_rabi_and_larmor_obey_their_closed_forms() {
    // Both are exact, so they are checked against arithmetic rather than
    // against a threshold: the Rabi probability is bounded and periodic, and
    // Larmor precession is a rotation.
    let mut rng = Rng::new(0x_5A11_0005);
    for _ in 0..500 {
        let rabi = 0.1 + rng.next_f64() * 5.0;
        let detuning = spread(&mut rng, 5.0);
        let t = rng.next_f64() * 20.0;
        let p = rabi_oscillation(rabi, detuning, t).unwrap();
        assert!((0.0..=1.0).contains(&p), "the probability is {p}");
        let generalised = (rabi * rabi + detuning * detuning).sqrt();
        let peak = rabi * rabi / (generalised * generalised);
        assert!(p <= peak + 1e-12, "the probability {p} exceeds its own ceiling {peak}");
        // Periodic in the generalised frequency.
        let period = 2.0 * std::f64::consts::PI / generalised;
        assert!(
            (rabi_oscillation(rabi, detuning, t + period).unwrap() - p).abs() < 1e-9,
            "the oscillation is not periodic"
        );
        // Zero at every whole multiple of the period.
        assert!(rabi_oscillation(rabi, detuning, period).unwrap() < 1e-9);

        // Larmor: a rotation, so the length and the z component survive.
        let m0 = (spread(&mut rng, 1.0), spread(&mut rng, 1.0), spread(&mut rng, 1.0));
        let b = spread(&mut rng, 3.0);
        let gamma = spread(&mut rng, 3.0);
        let after = larmor_precession(m0, b, gamma, t);
        let before_length = m0.0.hypot(m0.1).hypot(m0.2);
        assert!(
            (after.0.hypot(after.1).hypot(after.2) - before_length).abs() < 1e-9,
            "the precession changed the length"
        );
        assert!((after.2 - m0.2).abs() < 1e-15, "the z component moved");
        // Composing two rotations is one rotation through the summed time.
        let composed = larmor_precession(after, b, gamma, 1.3);
        let direct = larmor_precession(m0, b, gamma, t + 1.3);
        assert!(
            (composed.0 - direct.0).abs() < 1e-8 && (composed.1 - direct.1).abs() < 1e-8,
            "the rotations do not compose"
        );
    }
}

// ---------------------------------------------------------------------------
// Solid state
// ---------------------------------------------------------------------------

#[test]
fn prop_the_tight_binding_chain_matches_its_closed_form_at_every_length() {
    let mut rng = Rng::new(0x_5A11_0006);
    for _ in 0..80 {
        let n = 2 + pick(&mut rng, 60);
        let t = spread(&mut rng, 3.0);
        if t.abs() < 1e-6 {
            continue;
        }
        let (energies, vectors) = tight_binding_1d(t, &vec![0.0; n], false).unwrap();
        let mut expected: Vec<f64> = (1..=n)
            .map(|m| -2.0 * t * (m as f64 * std::f64::consts::PI / (n + 1) as f64).cos())
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (got, want) in energies.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-8, "n = {n}, t = {t}: {got} against {want}");
        }
        // Orthonormal eigenvectors, and each really an eigenvector.
        for i in 0..n.min(6) {
            let norm: f64 = vectors[i].iter().map(|c| c * c).sum();
            assert!((norm - 1.0).abs() < 1e-9, "eigenvector {i} has norm {norm}");
            for k in 0..n {
                let mut applied = 0.0;
                if k > 0 {
                    applied -= t * vectors[i][k - 1];
                }
                if k + 1 < n {
                    applied -= t * vectors[i][k + 1];
                }
                assert!(
                    (applied - energies[i] * vectors[i][k]).abs() < 1e-7,
                    "eigenvector {i} fails at site {k}"
                );
            }
        }
        // Every level lies inside the infinite chain's band.
        for e in &energies {
            assert!(e.abs() <= 2.0 * t.abs() + 1e-9, "a level of {e} escapes the band");
        }
        // The band function reproduces the extremes.
        assert!(
            (tight_binding_band_1d(0.0, t, 1.0) + 2.0 * t).abs() < 1e-12,
            "the band bottom is wrong"
        );
    }
}

#[test]
fn prop_the_ssh_edge_states_follow_the_winding_number() {
    // Bulk-boundary correspondence on random couplings: the invariant is a
    // function of two numbers and the edge count comes from a spectrum, and
    // they must agree every time.
    let mut rng = Rng::new(0x_5A11_0007);
    let mut topological = 0usize;
    let mut trivial = 0usize;
    for _ in 0..150 {
        let t1 = 0.2 + rng.next_f64() * 2.0;
        let t2 = 0.2 + rng.next_f64() * 2.0;
        if (t1 - t2).abs() < 0.15 {
            // Too near the transition for a finite chain to resolve.
            continue;
        }
        let cells = 25 + pick(&mut rng, 25);
        let winding = ssh_winding_number(t1, t2);
        let states = ssh_edge_states(cells, t1, t2).unwrap();
        assert_eq!(
            states,
            2 * winding as usize,
            "t1 = {t1}, t2 = {t2}: winding {winding} but {states} edge states"
        );
        if winding == 1 {
            topological += 1;
        } else {
            trivial += 1;
        }

        // The spectrum is symmetric about zero, since the chain is bipartite.
        let (energies, _) = ssh_model(cells, t1, t2).unwrap();
        for (low, high) in energies.iter().zip(energies.iter().rev()) {
            assert!((low + high).abs() < 1e-8, "the spectrum is not symmetric");
        }
    }
    assert!(topological > 30 && trivial > 30, "one phase was barely sampled");
}

#[test]
fn prop_the_occupation_functions_keep_their_bounds_and_symmetries() {
    let mut rng = Rng::new(0x_5A11_0008);
    for _ in 0..500 {
        let mu = spread(&mut rng, 1e-19);
        let temperature = 1.0 + rng.next_f64() * 3000.0;
        let energy = mu + spread(&mut rng, 5e-20);

        let f = fermi_dirac(energy, mu, temperature).unwrap();
        assert!((0.0..=1.0).contains(&f), "the occupation is {f}");
        let mirrored = fermi_dirac(2.0 * mu - energy, mu, temperature).unwrap();
        assert!((f + mirrored - 1.0).abs() < 1e-12, "the function is not antisymmetric");
        // Monotone decreasing in energy.
        let higher = fermi_dirac(energy + 1e-21, mu, temperature).unwrap();
        assert!(higher <= f + 1e-15, "the occupation rose with energy");

        // Bosons exceed fermions at the same energy above mu. The
        // difference is 2 / (exp(2x) - 1), which falls off twice as fast as
        // either occupation, so far out in the tail the two are equal to
        // every bit a double has -- and demanding strict inequality there
        // would be demanding precision that does not exist.
        if energy > mu {
            const BOLTZMANN: f64 = 1.380_649e-23;
            let x = (energy - mu) / (BOLTZMANN * temperature);
            let b = bose_einstein(energy, mu, temperature).unwrap();
            assert!(b > 0.0, "the boson occupation is {b}");
            assert!(b >= f, "bosons should not be outnumbered: {b} against {f}");
            if x < 20.0 {
                assert!(b > f, "bosons should outnumber fermions at x = {x}");
                // And the difference is exactly what the algebra says.
                let predicted = 2.0 / ((2.0 * x).exp() - 1.0);
                assert!(
                    (b - f - predicted).abs() < 1e-9 * predicted.max(1e-12),
                    "the gap is {} against the closed form {predicted}",
                    b - f
                );
            }
        }
    }
}

#[test]
fn prop_the_heat_capacities_are_monotone_and_share_a_classical_limit() {
    // Both models rise from zero to three k per atom, monotonically. They
    // differ only in how they approach zero, which is the point of having
    // both.
    let mut rng = Rng::new(0x_5A11_0009);
    const BOLTZMANN: f64 = 1.380_649e-23;
    for _ in 0..80 {
        let theta = 50.0 + rng.next_f64() * 800.0;
        let mut previous_debye = -1.0;
        let mut previous_einstein = -1.0;
        for k in 0..30usize {
            let t = 0.02 * theta * (k + 1) as f64;
            let debye = debye_heat_capacity(t, theta).unwrap();
            let einstein = einstein_heat_capacity(t, theta).unwrap();
            assert!(debye > previous_debye - 1e-30, "Debye fell at T = {t}");
            assert!(einstein > previous_einstein - 1e-30, "Einstein fell at T = {t}");
            assert!(debye <= 3.0 * BOLTZMANN * 1.001, "Debye exceeds Dulong-Petit: {debye}");
            assert!(einstein <= 3.0 * BOLTZMANN * 1.001, "Einstein exceeds it: {einstein}");
            previous_debye = debye;
            previous_einstein = einstein;
        }
        // Both reach the classical value.
        assert!(
            (debye_heat_capacity(100.0 * theta, theta).unwrap() / (3.0 * BOLTZMANN) - 1.0).abs()
                < 0.01
        );
        assert!(
            (einstein_heat_capacity(100.0 * theta, theta).unwrap() / (3.0 * BOLTZMANN) - 1.0).abs()
                < 0.01
        );
        // And Einstein is always the smaller at low temperature, because a
        // single frequency leaves nothing cheap to excite.
        let low = 0.1 * theta;
        assert!(
            einstein_heat_capacity(low, theta).unwrap() < debye_heat_capacity(low, theta).unwrap(),
            "Einstein should fall faster at T = {low}"
        );
    }
}

#[test]
fn prop_the_densities_of_states_scale_as_their_dimension_dictates() {
    let mut rng = Rng::new(0x_5A11_000A);
    for _ in 0..300 {
        let mass = 0.1 + rng.next_f64() * 5.0;
        let hbar = 0.5 + rng.next_f64() * 2.0;
        let energy = 0.01 + rng.next_f64() * 10.0;
        let quadrupled = 4.0 * energy;

        let one = density_of_states_1d_free(energy, mass, hbar).unwrap();
        let one_up = density_of_states_1d_free(quadrupled, mass, hbar).unwrap();
        assert!((one / one_up - 2.0).abs() < 1e-9, "the 1D scaling is {}", one / one_up);

        let two = density_of_states_2d_free(energy, mass, hbar).unwrap();
        let two_up = density_of_states_2d_free(quadrupled, mass, hbar).unwrap();
        assert!((two - two_up).abs() < 1e-15 * two, "the 2D density is not constant");

        let three = density_of_states_3d_free(energy, mass, hbar).unwrap();
        let three_up = density_of_states_3d_free(quadrupled, mass, hbar).unwrap();
        assert!((three_up / three - 2.0).abs() < 1e-9, "the 3D scaling is wrong");

        // All positive, and all zero below the band bottom.
        assert!(one > 0.0 && two > 0.0 && three > 0.0);
        assert_eq!(density_of_states_1d_free(-energy, mass, hbar).unwrap(), 0.0);
        assert_eq!(density_of_states_3d_free(-energy, mass, hbar).unwrap(), 0.0);
    }

    // A broadened level set integrates to its own count, whatever the levels.
    let mut rng = Rng::new(0x_5A11_000B);
    for _ in 0..100 {
        let count = 1 + pick(&mut rng, 12);
        let levels: Vec<f64> = (0..count).map(|_| spread(&mut rng, 5.0)).collect();
        let sigma = 0.05 + rng.next_f64() * 0.3;
        let curve = dos_from_bands(&levels, sigma, 6000).unwrap();
        let h = curve[1].0 - curve[0].0;
        let total: f64 = curve.iter().map(|(_, d)| d).sum::<f64>() * h;
        assert!(
            (total - count as f64).abs() < 0.01 * count as f64,
            "the density integrates to {total}, not {count}"
        );
        assert!(curve.iter().all(|(_, d)| *d >= 0.0));
    }
}

#[test]
fn prop_phonon_branches_stay_ordered_and_real_across_the_zone() {
    let mut rng = Rng::new(0x_5A11_000C);
    for _ in 0..300 {
        let spring = 0.1 + rng.next_f64() * 10.0;
        let m1 = 0.1 + rng.next_f64() * 5.0;
        let m2 = 0.1 + rng.next_f64() * 5.0;
        let a = 0.5 + rng.next_f64();
        for step in 0..40usize {
            let k = std::f64::consts::PI / (2.0 * a) * step as f64 / 39.0;
            let (acoustic, optical) = phonon_dispersion_1d_diatomic(k, spring, m1, m2, a);
            assert!(acoustic.is_finite() && optical.is_finite());
            assert!(acoustic >= 0.0 && optical >= 0.0, "a negative frequency at k = {k}");
            assert!(acoustic <= optical + 1e-12, "the branches crossed at k = {k}");
            // The monatomic chain is the equal-mass limit.
            let mono = phonon_dispersion_1d_monatomic(k, spring, m1, a);
            assert!(mono >= 0.0 && mono.is_finite());
        }
        // The acoustic branch starts at zero and the optical does not.
        let (acoustic0, optical0) = phonon_dispersion_1d_diatomic(0.0, spring, m1, m2, a);
        assert!(acoustic0 < 1e-9, "the acoustic branch starts at {acoustic0}");
        assert!(optical0 > 1e-6, "the optical branch starts at {optical0}");
        // Magnons, meanwhile, are quadratic at long wavelength.
        let j = 0.1 + rng.next_f64() * 3.0;
        let small = magnon_dispersion(j, 0.001, 0.5, a);
        let doubled = magnon_dispersion(j, 0.002, 0.5, a);
        assert!(
            (doubled / small - 4.0).abs() < 1e-3,
            "the magnon dispersion is not quadratic: {}",
            doubled / small
        );
    }
}

#[test]
fn prop_the_kronig_penney_function_and_graphene_bands_stay_within_their_bounds() {
    let mut rng = Rng::new(0x_5A11_000D);
    let mut allowed = 0usize;
    let mut forbidden = 0usize;
    for _ in 0..600 {
        let v0 = rng.next_f64() * 30.0;
        let a = 0.3 + rng.next_f64() * 2.0;
        let b = 0.1 + rng.next_f64();
        let energy = 0.05 + rng.next_f64() * 40.0;
        let value = kronig_penney(v0, a, b, energy, 1.0, 1.0).unwrap();
        assert!(value.is_finite(), "the dispersion function is {value}");
        if value.abs() <= 1.0 {
            allowed += 1;
        } else {
            forbidden += 1;
        }
    }
    assert!(allowed > 50 && forbidden > 50, "one regime was barely sampled");

    // Graphene: the bands are symmetric about zero everywhere, and bounded
    // by three times the hopping.
    for _ in 0..500 {
        let kx = spread(&mut rng, 4.0);
        let ky = spread(&mut rng, 4.0);
        let t = 0.5 + rng.next_f64() * 3.0;
        let (lower, upper) = graphene_dispersion(kx, ky, t);
        assert!((lower + upper).abs() < 1e-12, "the bands are not symmetric");
        assert!(upper <= 3.0 * t + 1e-9, "the band reaches {upper}, above 3t");
        assert!(upper >= -1e-12, "the upper band went negative");
    }
}

#[test]
fn prop_the_derived_quantities_have_the_signs_and_limits_they_claim() {
    let mut rng = Rng::new(0x_5A11_000E);
    for _ in 0..300 {
        // The effective mass of a cosine band is positive at the bottom and
        // negative at the top, whatever the parameters.
        let t = 0.1 + rng.next_f64() * 3.0;
        let a = 0.5 + rng.next_f64() * 2.0;
        let band = |k: f64| tight_binding_band_1d(k, t, a);
        let bottom = effective_mass_from_band(&band, 0.0, 1e-4).unwrap();
        let top = effective_mass_from_band(&band, std::f64::consts::PI / a, 1e-4).unwrap();
        assert!(bottom > 0.0, "the band-bottom mass is {bottom}");
        assert!(top < 0.0, "the band-top mass is {top}");
        assert!(
            (bottom + top).abs() < 1e-6 * bottom.abs(),
            "the two masses should be equal and opposite"
        );

        // The superconducting gap falls monotonically to zero at Tc.
        let tc = 1.0 + rng.next_f64() * 100.0;
        let mut previous = 1.1;
        for k in 1..20usize {
            let gap = bcs_gap_equation(tc * k as f64 / 20.0, tc).unwrap();
            assert!((0.0..=1.0).contains(&gap), "the gap is {gap}");
            assert!(gap < previous, "the gap rose");
            previous = gap;
        }
        assert_eq!(bcs_gap_equation(tc, tc).unwrap(), 0.0);

        // Landauer conductance is additive and bounded by the channel count.
        let channels = 1 + pick(&mut rng, 8);
        let transmissions: Vec<f64> = (0..channels).map(|_| rng.next_f64()).collect();
        let g = conductance_landauer(&transmissions).unwrap();
        let perfect = conductance_landauer(&vec![1.0; channels]).unwrap();
        assert!(g <= perfect + 1e-20, "the conductance exceeds the ballistic limit");
        assert!(g >= 0.0);
    }
}
