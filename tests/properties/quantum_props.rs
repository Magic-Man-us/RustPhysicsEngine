//! Properties of the quantum modules.
//!
//! Quantum mechanics is unusually well supplied with exact statements that
//! hold for *every* state rather than typically: the norm is conserved by any
//! unitary evolution, the uncertainty product never falls below `hbar / 2`,
//! eigenstates of a Hermitian operator with different eigenvalues are
//! orthogonal, and the Hamiltonian's expectation over any trial state is at
//! least its lowest eigenvalue. Each of those is checked here on randomly
//! generated potentials and states, where a hand-picked example could not
//! rule out a coincidence.

use rust_physics_engine::fractals::Complex;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::quantum::schrodinger::{
    imaginary_time_propagation, tdse_crank_nicolson, tdse_split_operator, tise_solve_fd,
    transmission_coefficient, tunneling_rectangular_exact,
};
use rust_physics_engine::quantum::wavefunction::{
    coherent_state, hermite_polynomial, hydrogen_radial, laguerre_associated, Wavefunction1D,
};

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

fn spread(rng: &mut Rng, half_width: f64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * half_width
}

/// A random smooth confining potential on a grid: a quadratic floor plus a
/// few bumps, so that bound states exist and the shape is never the same
/// twice.
fn random_potential(rng: &mut Rng, n: usize, dx: f64, x0: f64) -> Vec<f64> {
    let curvature = 0.2 + rng.next_f64();
    let bumps: Vec<(f64, f64, f64)> = (0..3)
        .map(|_| (spread(rng, 6.0), spread(rng, 4.0), 0.5 + rng.next_f64() * 2.0))
        .collect();
    (0..n)
        .map(|k| {
            let x = x0 + k as f64 * dx;
            let mut value = 0.5 * curvature * x * x;
            for &(centre, height, width) in &bumps {
                value += height * (-(x - centre) * (x - centre) / (2.0 * width * width)).exp();
            }
            value
        })
        .collect()
}

/// A random normalised complex state on the grid.
fn random_state(rng: &mut Rng, n: usize, dx: f64, x0: f64) -> Wavefunction1D {
    let centre = spread(rng, 4.0);
    let width = 0.6 + rng.next_f64() * 2.0;
    let k0 = spread(rng, 3.0);
    let psi: Vec<Complex> = (0..n)
        .map(|k| {
            let x = x0 + k as f64 * dx;
            let envelope = (-(x - centre) * (x - centre) / (4.0 * width * width)).exp()
                * (1.0 + 0.4 * (x * 1.7).sin());
            let phase = k0 * x + 0.3 * (x * 0.9).cos();
            Complex::new(envelope * phase.cos(), envelope * phase.sin())
        })
        .collect();
    let mut state = Wavefunction1D::new(psi, dx, x0).unwrap();
    state.normalize();
    state
}

// ---------------------------------------------------------------------------
// Wavefunctions
// ---------------------------------------------------------------------------

#[test]
fn prop_the_uncertainty_product_never_falls_below_half_hbar() {
    // Robertson's bound, on states that are deliberately not Gaussian. The
    // interesting direction is the lower one: a numerical scheme that
    // computed either variance wrongly would show up as a violation, and the
    // bound is exact rather than asymptotic.
    let mut rng = Rng::new(0x_5C11_0001);
    let n = 1024usize;
    let dx = 40.0 / n as f64;
    let x0 = -20.0f64;
    let mut saturating = 0usize;
    for _ in 0..300 {
        let state = random_state(&mut rng, n, dx, x0);
        for hbar in [0.5f64, 1.0, 2.5] {
            let product = state.uncertainty_product(hbar).unwrap();
            assert!(
                product >= hbar / 2.0 - 1e-6,
                "the product is {product}, below hbar / 2 = {}",
                hbar / 2.0
            );
            if product < hbar * 0.55 {
                saturating += 1;
            }
        }
    }
    // Some draws come close to the bound, so the assertion is not trivially
    // satisfied by every state being far from it.
    assert!(saturating > 5, "only {saturating} states came near the bound");
}

#[test]
fn prop_free_evolution_is_unitary_and_reversible() {
    // Unitarity means two things that are worth checking separately: the norm
    // is preserved, and so is every overlap. The second is stronger and is
    // what makes probabilities meaningful.
    let mut rng = Rng::new(0x_5C11_0002);
    let n = 512usize;
    let dx = 40.0 / n as f64;
    let x0 = -20.0f64;
    for _ in 0..150 {
        let a = random_state(&mut rng, n, dx, x0);
        let b = random_state(&mut rng, n, dx, x0);
        let t = spread(&mut rng, 3.0);
        let mass = 0.5 + rng.next_f64() * 2.0;
        let hbar = 0.5 + rng.next_f64();

        let a_t = a.propagate_free(t, hbar, mass).unwrap();
        let b_t = b.propagate_free(t, hbar, mass).unwrap();
        assert!((a_t.norm() - a.norm()).abs() < 1e-12, "the norm changed to {}", a_t.norm());

        let before = a.overlap(&b).unwrap();
        let after = a_t.overlap(&b_t).unwrap();
        assert!(
            (before.re - after.re).abs() < 1e-10 && (before.im - after.im).abs() < 1e-10,
            "the overlap moved from {before:?} to {after:?}"
        );

        // Running the clock backwards returns the original state exactly.
        let back = a_t.propagate_free(-t, hbar, mass).unwrap();
        for (p, q) in back.psi.iter().zip(&a.psi) {
            assert!((p.re - q.re).abs() < 1e-10 && (p.im - q.im).abs() < 1e-10);
        }

        // And the momentum distribution is untouched, since the free
        // Hamiltonian is a function of momentum alone.
        assert!((a_t.expectation_k().unwrap() - a.expectation_k().unwrap()).abs() < 1e-9);
        assert!((a_t.variance_k().unwrap() - a.variance_k().unwrap()).abs() < 1e-9);
    }
}

#[test]
fn prop_the_orthogonal_polynomials_satisfy_their_differential_equations() {
    // The recurrences are checked against the equations that define the
    // polynomials, evaluated by finite differences. Nothing in the
    // implementation knows about the differential equation, so this is an
    // independent characterisation rather than a restatement.
    let mut rng = Rng::new(0x_5C11_0003);
    let h = 1e-4f64;
    for _ in 0..400 {
        let n = pick(&mut rng, 10);
        let x = spread(&mut rng, 2.5);
        // Hermite: y'' - 2 x y' + 2 n y = 0.
        let y = |t: f64| hermite_polynomial(n, t);
        let first = (y(x + h) - y(x - h)) / (2.0 * h);
        let second = (y(x + h) - 2.0 * y(x) + y(x - h)) / (h * h);
        let residual = second - 2.0 * x * first + 2.0 * n as f64 * y(x);
        let magnitude = second.abs().max(1.0);
        assert!(
            residual.abs() < 1e-4 * magnitude,
            "H_{n} fails its equation at {x}: residual {residual}"
        );

        // Laguerre: x y'' + (k + 1 - x) y' + n y = 0.
        let k = pick(&mut rng, 4) as f64;
        let x = 0.2 + rng.next_f64() * 5.0;
        let y = |t: f64| laguerre_associated(n, k, t);
        let first = (y(x + h) - y(x - h)) / (2.0 * h);
        let second = (y(x + h) - 2.0 * y(x) + y(x - h)) / (h * h);
        let residual = x * second + (k + 1.0 - x) * first + n as f64 * y(x);
        let magnitude = (x * second).abs().max(1.0);
        assert!(
            residual.abs() < 1e-4 * magnitude,
            "L_{n}^{k} fails its equation at {x}: residual {residual}"
        );
    }
}

#[test]
fn prop_coherent_states_are_normalised_and_poissonian_at_every_amplitude() {
    // The photon distribution is Poisson with mean |alpha|^2, so mean and
    // variance coincide -- an equality that a wrong normalisation or a
    // mishandled factorial would break immediately.
    let mut rng = Rng::new(0x_5C11_0004);
    for _ in 0..200 {
        let magnitude = rng.next_f64() * 4.0;
        let phase = spread(&mut rng, std::f64::consts::PI);
        let alpha = Complex::new(magnitude * phase.cos(), magnitude * phase.sin());
        let coefficients = coherent_state(alpha, 160).unwrap();
        let weights: Vec<f64> = coefficients.iter().map(|z| z.norm_sq()).collect();

        let total: f64 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-8, "the state has norm {total}");
        let mean: f64 = weights.iter().enumerate().map(|(k, w)| k as f64 * w).sum();
        let second: f64 = weights.iter().enumerate().map(|(k, w)| (k * k) as f64 * w).sum();
        let expected = magnitude * magnitude;
        assert!((mean - expected).abs() < 1e-6, "the mean is {mean}, not {expected}");
        assert!(
            (second - mean * mean - expected).abs() < 1e-5,
            "Poisson requires variance = mean, got {}",
            second - mean * mean
        );
        // The phase never touches the statistics.
        let plain = coherent_state(Complex::new(magnitude, 0.0), 160).unwrap();
        for (a, b) in coefficients.iter().zip(&plain) {
            assert!((a.norm() - b.norm()).abs() < 1e-12);
        }
    }
}

#[test]
fn prop_the_hydrogen_radial_states_are_orthonormal_within_each_l() {
    // Orthogonality between different n at the same l is forced by
    // hermiticity, and normalisation is put in by hand -- so checking both
    // tests the normalisation constant and the Laguerre recurrence together.
    let steps = 200_000usize;
    for l in 0..3usize {
        for n in (l + 1)..=(l + 4) {
            for m in (l + 1)..=(l + 4) {
                let reach = 40.0 * (n.max(m)) as f64;
                let h = reach / steps as f64;
                let integral: f64 = (0..steps)
                    .map(|k| {
                        let r = (k as f64 + 0.5) * h;
                        hydrogen_radial(n, l, r, 1.0) * hydrogen_radial(m, l, r, 1.0) * r * r
                    })
                    .sum::<f64>()
                    * h;
                let expected = f64::from(n == m);
                assert!(
                    (integral - expected).abs() < 2e-4,
                    "<{n},{l}|{m},{l}> is {integral}, not {expected}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The Schrodinger solvers
// ---------------------------------------------------------------------------

#[test]
fn prop_the_computed_states_really_are_eigenstates() {
    // The certificate for an eigenpair is the residual `H psi - E psi`, which
    // needs no reference answer at all. Checking it on random potentials is
    // the strongest statement available about the solver.
    let mut rng = Rng::new(0x_5C11_0005);
    for _ in 0..60 {
        let n = 300 + pick(&mut rng, 200);
        let reach = 8.0f64;
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let v = random_potential(&mut rng, n, dx, x0);
        let mass = 0.5 + rng.next_f64();
        let hbar = 0.5 + rng.next_f64();
        let (energies, states) = tise_solve_fd(&v, dx, mass, hbar, 4).unwrap();

        let kinetic = hbar * hbar / (2.0 * mass * dx * dx);
        for (level, psi) in states.iter().enumerate() {
            let norm: f64 = psi.iter().map(|c| c * c).sum::<f64>() * dx;
            assert!((norm - 1.0).abs() < 1e-9, "state {level} has norm {norm}");

            let mut residual: f64 = 0.0;
            for k in 0..n {
                let mut applied = (2.0 * kinetic + v[k]) * psi[k];
                if k > 0 {
                    applied -= kinetic * psi[k - 1];
                }
                if k + 1 < n {
                    applied -= kinetic * psi[k + 1];
                }
                residual = residual.max((applied - energies[level] * psi[k]).abs());
            }
            assert!(
                residual < 1e-6 * (1.0 + energies[level].abs()),
                "state {level} leaves a residual of {residual}"
            );
        }

        // Ascending, and orthogonal to each other.
        assert!(energies.windows(2).all(|w| w[0] <= w[1] + 1e-12), "{energies:?} is not ascending");
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                let overlap: f64 =
                    states[i].iter().zip(&states[j]).map(|(a, b)| a * b).sum::<f64>() * dx;
                assert!(overlap.abs() < 1e-6, "states {i} and {j} overlap by {overlap}");
            }
        }
        // The node count identifies the level, which is a theorem about
        // one-dimensional Sturm-Liouville problems and not an accident.
        for (level, psi) in states.iter().enumerate() {
            let nodes = (0..n - 1).filter(|&k| psi[k] * psi[k + 1] < 0.0).count();
            assert_eq!(nodes, level, "state {level} has {nodes} nodes");
        }
    }
}

#[test]
fn prop_no_trial_state_beats_the_computed_ground_energy() {
    // The variational principle, used as a test rather than as a method: the
    // Rayleigh quotient over *any* state is at least the lowest eigenvalue.
    // A solver that reported too low a ground energy would be caught by a
    // random trial state, and nothing else would catch it.
    let mut rng = Rng::new(0x_5C11_0006);
    for _ in 0..80 {
        let n = 256usize;
        let reach = 8.0f64;
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let v = random_potential(&mut rng, n, dx, x0);
        let (energies, _) = tise_solve_fd(&v, dx, 1.0, 1.0, 1).unwrap();
        let kinetic = 1.0 / (2.0 * dx * dx);

        for _ in 0..20 {
            let trial: Vec<f64> = (0..n).map(|_| spread(&mut rng, 1.0)).collect();
            let norm: f64 = trial.iter().map(|c| c * c).sum();
            if norm <= 0.0 {
                continue;
            }
            let mut quotient = 0.0;
            for k in 0..n {
                let mut applied = (2.0 * kinetic + v[k]) * trial[k];
                if k > 0 {
                    applied -= kinetic * trial[k - 1];
                }
                if k + 1 < n {
                    applied -= kinetic * trial[k + 1];
                }
                quotient += trial[k] * applied;
            }
            let rayleigh = quotient / norm;
            assert!(
                rayleigh >= energies[0] - 1e-6,
                "a trial state reached {rayleigh}, below the reported ground energy {}",
                energies[0]
            );
        }
    }
}

#[test]
fn prop_both_propagators_conserve_norm_on_random_potentials() {
    // Unitarity is the one thing a time-dependent solver must never lose, and
    // the two here achieve it by different means -- exponentials of Hermitian
    // operators, and a Cayley transform -- so they fail differently and are
    // worth checking on the same problems.
    let mut rng = Rng::new(0x_5C11_0007);
    let n = 256usize;
    let reach = 12.0f64;
    let dx = 2.0 * reach / n as f64;
    let x0 = -reach;
    for _ in 0..60 {
        let v = random_potential(&mut rng, n, dx, x0);
        let start = random_state(&mut rng, n, dx, x0);
        let dt = 0.001 + rng.next_f64() * 0.02;

        let mut split = start.clone();
        tdse_split_operator(&mut split, &v, dt, 200, 1.0, 1.0).unwrap();
        assert!(
            (split.norm() - 1.0).abs() < 1e-11,
            "the split operator left a norm of {}",
            split.norm()
        );

        let mut cayley = start.clone();
        tdse_crank_nicolson(&mut cayley, &v, dt, 200, 1.0, 1.0).unwrap();
        assert!(
            (cayley.norm() - 1.0).abs() < 1e-9,
            "Crank-Nicolson left a norm of {}",
            cayley.norm()
        );

        // The split operator also conserves energy on a static potential,
        // which the splitting does not give for free.
        let before = start.energy(&v, 1.0, 1.0).unwrap();
        let after = split.energy(&v, 1.0, 1.0).unwrap();
        assert!(
            (after - before).abs() < 1e-3 * (1.0 + before.abs()),
            "the energy moved from {before} to {after}"
        );
    }
}

#[test]
fn prop_imaginary_time_reaches_the_same_ground_state_the_eigensolver_does() {
    // Two unrelated algorithms on the same random potentials. Neither is a
    // reference implementation of the other, so agreement is evidence and
    // disagreement is a bug in one of them.
    let mut rng = Rng::new(0x_5C11_0008);
    for _ in 0..40 {
        let n = 201usize;
        let reach = 6.0f64;
        let dx = 2.0 * reach / (n - 1) as f64;
        let x0 = -reach;
        let v = random_potential(&mut rng, n, dx, x0);
        let (energies, states) = tise_solve_fd(&v, dx, 1.0, 1.0, 2).unwrap();
        let gap = energies[1] - energies[0];
        if gap < 0.05 {
            // A tiny gap makes imaginary time arbitrarily slow, which is a
            // known property of the method rather than a failure of it.
            continue;
        }
        let (energy, state) =
            imaginary_time_propagation(&v, dx, 5e-4, 60_000, 1.0, 1.0).unwrap();
        assert!(
            (energy - energies[0]).abs() < 1e-4 * (1.0 + energies[0].abs()),
            "imaginary time gives {energy} against {}",
            energies[0]
        );
        let overlap: f64 =
            state.iter().zip(&states[0]).map(|(a, b)| a * b).sum::<f64>() * dx;
        assert!(
            (overlap.abs() - 1.0).abs() < 1e-3,
            "the two ground states overlap by {overlap}"
        );
    }
}

#[test]
fn prop_the_transfer_matrix_matches_the_closed_form_on_every_rectangular_barrier() {
    // The transfer matrix is exact for a piecewise-constant potential, so on
    // a rectangle it is not an approximation to the closed form -- it is the
    // same number by a different route, at any resolution.
    let mut rng = Rng::new(0x_5C11_0009);
    let mut below = 0usize;
    let mut above = 0usize;
    for _ in 0..400 {
        let v0 = spread(&mut rng, 8.0);
        let width = 0.2 + rng.next_f64() * 3.0;
        let energy = 0.05 + rng.next_f64() * 12.0;
        let slices = 20 + pick(&mut rng, 200);
        let dx = width / slices as f64;
        let v = vec![v0; slices];

        let numeric = transmission_coefficient(&v, dx, energy, 1.0, 1.0).unwrap();
        let exact = tunneling_rectangular_exact(v0, width, energy, 1.0, 1.0).unwrap();
        assert!(
            (numeric - exact).abs() < 1e-8 * (1.0 + exact),
            "V0 = {v0}, width = {width}, E = {energy}: {numeric} against {exact}"
        );
        assert!((0.0..=1.0 + 1e-12).contains(&numeric), "the probability is {numeric}");
        if energy < v0 {
            below += 1;
            assert!(numeric < 1.0, "tunnelling should be imperfect");
        } else {
            above += 1;
        }
    }
    assert!(below > 40 && above > 40, "the two regimes were not both exercised: {below}, {above}");
}

#[test]
fn prop_a_barrier_and_a_well_of_the_same_depth_transmit_differently() {
    // A negative V0 is a well rather than a barrier, and there the
    // transmission has resonances at every energy -- the same algebra, an
    // entirely different phenomenon. A routine that took the absolute value
    // somewhere would give the same answer for both.
    let mut rng = Rng::new(0x_5C11_000A);
    let mut resonances = 0usize;
    for _ in 0..300 {
        let depth = 1.0 + rng.next_f64() * 6.0;
        let width = 0.5 + rng.next_f64() * 3.0;
        let energy = 0.1 + rng.next_f64() * 5.0;
        let barrier = tunneling_rectangular_exact(depth, width, energy, 1.0, 1.0).unwrap();
        let well = tunneling_rectangular_exact(-depth, width, energy, 1.0, 1.0).unwrap();
        assert!((0.0..=1.0 + 1e-12).contains(&well));
        if energy < depth {
            assert!(
                well > barrier,
                "a well should transmit better than a barrier: {well} against {barrier}"
            );
        }
        if well > 0.999 {
            resonances += 1;
        }
    }
    assert!(resonances > 5, "no well resonances arose, so the regime is untested");
}
