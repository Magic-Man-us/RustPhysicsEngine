//! Properties of the chemical kinetics module.
//!
//! Reaction networks come with invariants that hold on every trajectory
//! whatever the rate constants: matter is neither created nor destroyed,
//! concentrations never go negative, and the deterministic and stochastic
//! descriptions agree in the mean for a network whose propensities are
//! linear. Alongside those, every fit here inverts a closed form, so on data
//! generated from the model it must return the parameters that generated it
//! -- which is checkable on random parameters rather than on one worked
//! example.

use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::statistical_mechanics::kinetics::{
    autocatalysis_ignition, avrami_fit, buffer_henderson_hasselbalch, butler_volmer,
    chain_reaction_criticality, cottrell_current, debye_huckel_activity, enzyme_inhibition,
    equilibrium_composition, eyring, gillespie_ssa, hill_equation, hill_fit, jmak_avrami,
    kinetic_isotope_effect_estimate, kramers_rate_check, mass_action_rates, michaelis_menten,
    mm_fit, nernst, nucleation_barrier, nucleation_rate_cnt, ph_from_equilibria, rate_equations,
    stoichiometry_matrix, tau_leaping, temperature_jump_relaxation,
    transition_state_theory_rate, Inhibition, Reaction,
};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random network in which every reaction moves the same number of
/// molecules in as out, so the total count is conserved on every trajectory.
///
/// Generating the conservation law rather than asserting one particular
/// network's is what makes the check general: the integrator has no way to
/// know which combination is conserved, so getting it right on an arbitrary
/// network is evidence about the integrator rather than about the fixture.
fn balanced_network(rng: &mut Rng, species: usize, count: usize) -> (Vec<Reaction>, Vec<f64>) {
    let mut reactions = Vec::with_capacity(count);
    let mut k = Vec::with_capacity(count);
    while reactions.len() < count {
        let molecularity = 1 + pick(rng, 2);
        let mut reactants: Vec<(usize, u32)> = Vec::new();
        let mut products: Vec<(usize, u32)> = Vec::new();
        for _ in 0..molecularity {
            let s = pick(rng, species);
            match reactants.iter_mut().find(|(t, _)| *t == s) {
                Some(entry) => entry.1 += 1,
                None => reactants.push((s, 1)),
            }
        }
        for _ in 0..molecularity {
            let s = pick(rng, species);
            match products.iter_mut().find(|(t, _)| *t == s) {
                Some(entry) => entry.1 += 1,
                None => products.push((s, 1)),
            }
        }
        // A reaction whose products match its reactants does nothing, and
        // would make the "did anything happen" checks vacuous.
        let mut left = reactants.clone();
        let mut right = products.clone();
        left.sort_unstable();
        right.sort_unstable();
        if left == right {
            continue;
        }
        reactions.push(Reaction::new(&reactants, &products));
        k.push(0.05 + rng.next_f64() * 2.0);
    }
    (reactions, k)
}

// ---------------------------------------------------------------------------
// Networks
// ---------------------------------------------------------------------------

#[test]
fn prop_the_integrator_conserves_matter_and_keeps_concentrations_positive() {
    let mut rng = Rng::new(0x_C0DE_9001);
    for trial in 0..12 {
        let species = 3 + trial % 3;
        let (reactions, k) = balanced_network(&mut rng, species, 2 + trial % 4);
        let stoich = stoichiometry_matrix(&reactions, species).unwrap();
        let c0: Vec<f64> = (0..species).map(|_| 0.1 + rng.next_f64()).collect();
        let total: f64 = c0.iter().sum();
        let rates = |c: &[f64]| mass_action_rates(&reactions, &k, c).unwrap();
        let trace = rate_equations(&stoich, &rates, &c0, 5.0, 1e-8).unwrap();
        assert!(trace.len() > 2, "the run produced {} steps", trace.len());
        for (t, c) in &trace {
            assert!(
                close(c.iter().sum::<f64>(), total, 1e-5 * total),
                "at t = {t} the total is {} against {total}",
                c.iter().sum::<f64>()
            );
            assert!(c.iter().all(|v| *v >= -1e-12), "a concentration went negative at t = {t}");
            assert!(c.iter().all(|v| v.is_finite()), "the run blew up at t = {t}");
        }
        // Time advances and reaches the end.
        for pair in trace.windows(2) {
            assert!(pair[1].0 > pair[0].0, "time did not advance");
        }
        assert!(close(trace.last().unwrap().0, 5.0, 1e-9));
        // And something actually happened.
        let moved: f64 = (0..species)
            .map(|i| (trace.last().unwrap().1[i] - c0[i]).abs())
            .fold(0.0, f64::max);
        assert!(moved > 1e-6, "nothing reacted at all");
    }
}

#[test]
fn prop_the_stochastic_algorithms_conserve_the_same_count_exactly() {
    // Exactly, not approximately: the counts are integers and every event
    // applies a balanced net change, so the total cannot drift even by
    // rounding. Both algorithms are checked, since tau-leaping applies many
    // events at once and is the one that could get it wrong.
    let mut rng = Rng::new(0x_C0DE_9002);
    for trial in 0..10 {
        let species = 3 + trial % 3;
        let (reactions, k) = balanced_network(&mut rng, species, 2 + trial % 3);
        let x0: Vec<u64> = (0..species).map(|_| 20 + pick(&mut rng, 200) as u64).collect();
        let total: u64 = x0.iter().sum();
        let exact = gillespie_ssa(&reactions, &k, &x0, 1.0, 200_000, &mut rng).unwrap();
        for (t, x) in &exact {
            assert_eq!(x.iter().sum::<u64>(), total, "Gillespie lost a molecule at t = {t}");
        }
        assert!(exact.len() > 1, "no event fired at all");
        let leapt = tau_leaping(&reactions, &k, &x0, 1.0, 0.01, &mut rng).unwrap();
        for (t, x) in &leapt {
            assert_eq!(x.iter().sum::<u64>(), total, "tau-leaping lost a molecule at t = {t}");
        }
        // Neither runs past its end time.
        assert!(exact.last().unwrap().0 <= 1.0 + 1e-12);
        assert!(leapt.last().unwrap().0 <= 1.0 + 1e-12);
    }
}

#[test]
fn prop_the_gillespie_mean_tracks_the_rate_equations_for_a_linear_network() {
    // Where the propensities are linear in the counts the master equation
    // and the rate equations agree exactly in the mean -- the expectation of
    // a linear function is the function of the expectation -- so this is a
    // comparison with no approximation in it, only sampling error.
    let mut rng = Rng::new(0x_C0DE_9003);
    for trial in 0..5 {
        let species = 3 + trial % 2;
        // Unimolecular reactions only, so every propensity is linear.
        let reactions: Vec<Reaction> = (0..species)
            .map(|i| Reaction::new(&[(i, 1)], &[((i + 1) % species, 1)]))
            .collect();
        let k: Vec<f64> = (0..species).map(|_| 0.3 + rng.next_f64() * 1.5).collect();
        let start = 400u64;
        let mut x0 = vec![0u64; species];
        x0[0] = start;
        let t_end = 1.5;

        let runs = 800;
        let mut totals = vec![0.0f64; species];
        for _ in 0..runs {
            let trace = gillespie_ssa(&reactions, &k, &x0, t_end, 200_000, &mut rng).unwrap();
            for (i, v) in trace.last().unwrap().1.iter().enumerate() {
                totals[i] += *v as f64;
            }
        }
        let stoich = stoichiometry_matrix(&reactions, species).unwrap();
        let c0: Vec<f64> = x0.iter().map(|v| *v as f64).collect();
        let rates = |c: &[f64]| mass_action_rates(&reactions, &k, c).unwrap();
        let ode = rate_equations(&stoich, &rates, &c0, t_end, 1e-10).unwrap();
        let deterministic = &ode.last().unwrap().1;
        for i in 0..species {
            let mean = totals[i] / f64::from(runs);
            let scale = deterministic[i].max(5.0);
            assert!(
                close(mean, deterministic[i], 0.08 * scale),
                "species {i}: the mean is {mean} against the ODE's {}",
                deterministic[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fits
// ---------------------------------------------------------------------------

#[test]
fn prop_the_saturation_fits_invert_their_own_models() {
    // Exact data, so the fits must be exact: any discrepancy is a defect in
    // the fit and not sampling error. Across random parameters, so a fit
    // that happened to work at one scale could not pass.
    let mut rng = Rng::new(0x_C0DE_9010);
    for _ in 0..25 {
        let vmax = 0.2 + rng.next_f64() * 20.0;
        let km = 0.02 + rng.next_f64() * 8.0;
        let s: Vec<f64> = (1..=14).map(|j| km * f64::from(j) * 0.35).collect();
        let v: Vec<f64> = s.iter().map(|x| michaelis_menten(*x, vmax, km)).collect();
        let (fit_vmax, fit_km) = mm_fit(&s, &v).unwrap();
        assert!(close(fit_vmax, vmax, 1e-5 * vmax), "vmax {vmax} returned {fit_vmax}");
        assert!(close(fit_km, km, 1e-5 * km), "km {km} returned {fit_km}");
        // The double-reciprocal line carries the same constants.
        let (_, slope, intercept) = lineweaver_burk_line(&s, &v);
        assert!(close(intercept, 1.0 / vmax, 1e-6 / vmax));
        assert!(close(slope, km / vmax, 1e-6 * km / vmax));

        // And the Hill fit, with a cooperativity of its own.
        let n = 0.6 + rng.next_f64() * 3.0;
        let hv: Vec<f64> = s.iter().map(|x| hill_equation(*x, vmax, km, n)).collect();
        let (hf_vmax, hf_k, hf_n) = hill_fit(&s, &hv).unwrap();
        assert!(close(hf_n, n, 0.03 * n), "the exponent {n} returned {hf_n}");
        assert!(close(hf_vmax, vmax, 0.03 * vmax), "Hill vmax {vmax} returned {hf_vmax}");
        assert!(close(hf_k, km, 0.03 * km), "Hill k {km} returned {hf_k}");
    }
}

fn lineweaver_burk_line(s: &[f64], v: &[f64]) -> (Vec<(f64, f64)>, f64, f64) {
    rust_physics_engine::statistical_mechanics::kinetics::lineweaver_burk(s, v).unwrap()
}

#[test]
fn prop_the_avrami_fit_inverts_its_own_transformation() {
    let mut rng = Rng::new(0x_C0DE_9011);
    for _ in 0..30 {
        let n = 0.5 + rng.next_f64() * 4.0;
        let k = 0.05 + rng.next_f64() * 5.0;
        let times: Vec<f64> = (1..=20).map(|j| f64::from(j) * 0.12 / k).collect();
        let fraction: Vec<f64> = times.iter().map(|t| jmak_avrami(*t, k, n)).collect();
        let (fit_k, fit_n) = avrami_fit(&times, &fraction).unwrap();
        assert!(close(fit_n, n, 1e-6 * n), "the exponent {n} returned {fit_n}");
        assert!(close(fit_k, k, 1e-6 * k), "the rate {k} returned {fit_k}");
        // The curve is a distribution function: monotone from zero to one.
        let mut previous = 0.0;
        for x in &fraction {
            assert!(*x >= previous - 1e-15 && *x <= 1.0, "the fraction left [0, 1]");
            previous = *x;
        }
        assert!(close(jmak_avrami(0.0, k, n), 0.0, 1e-15));
        // Passing 1 - 1/e at t = 1/k whatever the exponent.
        assert!(close(jmak_avrami(1.0 / k, k, n), 1.0 - (-1.0f64).exp(), 1e-12));
    }
}

#[test]
fn prop_inhibition_reduces_to_the_uninhibited_rate_and_moves_the_right_constant() {
    // Every mechanism must vanish as the inhibitor does, and each must move
    // the constant it is named for by exactly the factor 1 + i/ki. Checked
    // by refitting rather than by inspecting the formula, so the test is
    // about the observable behaviour.
    let mut rng = Rng::new(0x_C0DE_9012);
    for _ in 0..15 {
        let vmax = 0.5 + rng.next_f64() * 10.0;
        let km = 0.1 + rng.next_f64() * 5.0;
        let ki = 0.1 + rng.next_f64() * 5.0;
        let i = rng.next_f64() * 10.0;
        let alpha = 1.0 + i / ki;
        let s: Vec<f64> = (1..=14).map(|j| km * f64::from(j) * 0.35).collect();
        for kind in [Inhibition::Competitive, Inhibition::Uncompetitive, Inhibition::NonCompetitive]
        {
            for x in &s {
                assert!(close(
                    enzyme_inhibition(*x, 0.0, vmax, km, ki, kind).unwrap(),
                    michaelis_menten(*x, vmax, km),
                    1e-12 * vmax
                ));
                // An inhibitor can only slow the reaction, never speed it.
                assert!(
                    enzyme_inhibition(*x, i, vmax, km, ki, kind).unwrap()
                        <= michaelis_menten(*x, vmax, km) + 1e-12
                );
            }
            let v: Vec<f64> = s
                .iter()
                .map(|x| enzyme_inhibition(*x, i, vmax, km, ki, kind).unwrap())
                .collect();
            let (fit_vmax, fit_km) = mm_fit(&s, &v).unwrap();
            let (want_vmax, want_km) = match kind {
                Inhibition::Competitive => (vmax, km * alpha),
                Inhibition::Uncompetitive => (vmax / alpha, km / alpha),
                Inhibition::NonCompetitive => (vmax / alpha, km),
            };
            assert!(
                close(fit_vmax, want_vmax, 1e-3 * want_vmax),
                "{kind:?}: vmax fitted to {fit_vmax} against {want_vmax}"
            );
            assert!(
                close(fit_km, want_km, 1e-3 * want_km),
                "{kind:?}: km fitted to {fit_km} against {want_km}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Equilibrium and acid-base
// ---------------------------------------------------------------------------

#[test]
fn prop_the_equilibrium_composition_satisfies_both_of_its_conditions() {
    let mut rng = Rng::new(0x_C0DE_9020);
    let mut stoich = rust_physics_engine::linalg::Matrix::zeros(3, 1);
    stoich.set(0, 0, -1.0);
    stoich.set(1, 0, -1.0);
    stoich.set(2, 0, 1.0);
    for _ in 0..40 {
        let k = 10f64.powf(rng.next_f64() * 8.0 - 4.0);
        let a_total = 0.05 + rng.next_f64() * 5.0;
        let b_total = 0.05 + rng.next_f64() * 5.0;
        let totals = vec![
            (vec![1.0, 0.0, 1.0], a_total),
            (vec![0.0, 1.0, 1.0], b_total),
        ];
        let c = equilibrium_composition(&stoich, &[k], &totals).unwrap();
        assert!(c.iter().all(|v| *v > 0.0), "a concentration is not positive");
        assert!(
            close(c[2] / (c[0] * c[1]), k, 1e-5 * k),
            "the quotient is {} against {k}",
            c[2] / (c[0] * c[1])
        );
        assert!(close(c[0] + c[2], a_total, 1e-8 * a_total));
        assert!(close(c[1] + c[2], b_total, 1e-8 * b_total));
        // The product cannot exceed the scarcer reagent.
        assert!(c[2] <= a_total.min(b_total) + 1e-9);
    }
}

#[test]
fn prop_the_ph_solver_returns_a_root_of_the_charge_balance() {
    // Self-consistency rather than comparison: whatever pH comes back, the
    // charge balance must change sign across it. That is checkable on any
    // input at all, including the dilute and strong cases where every
    // textbook approximation fails.
    let mut rng = Rng::new(0x_C0DE_9021);
    const KW: f64 = 1e-14;
    for _ in 0..200 {
        let acids: Vec<(f64, f64)> = (0..1 + pick(&mut rng, 3))
            .map(|_| {
                (
                    rng.next_f64() * 12.0,
                    10f64.powf(rng.next_f64() * 8.0 - 8.0),
                )
            })
            .collect();
        let base = if rng.next_f64() < 0.5 {
            0.0
        } else {
            10f64.powf(rng.next_f64() * 8.0 - 8.0)
        };
        let Ok(ph) = ph_from_equilibria(&acids, base) else {
            continue;
        };
        assert!((-1.0..=15.0).contains(&ph), "the pH came back as {ph}");
        let balance = |h: f64| -> f64 {
            let mut total = KW / h - h - base;
            for (pka, c) in &acids {
                let ka = 10f64.powf(-pka);
                total += c * ka / (ka + h);
            }
            total
        };
        // Bracketing: the balance is monotone decreasing in [H+], so it is
        // positive just below the root's [H+] and negative just above.
        let low = balance(10f64.powf(-(ph + 1e-6)));
        let high = balance(10f64.powf(-(ph - 1e-6)));
        assert!(
            low >= -1e-18 && high <= 1e-18,
            "the balance does not change sign across pH {ph}: {low} and {high}"
        );
        // Adding base can only raise the pH.
        if let Ok(more) = ph_from_equilibria(&acids, base + 1e-3) {
            assert!(more >= ph - 1e-9, "adding base lowered the pH from {ph} to {more}");
        }
        // And adding acid can only lower it.
        let mut stronger = acids.clone();
        stronger.push((2.0, 1e-3));
        if let Ok(less) = ph_from_equilibria(&stronger, base) {
            assert!(less <= ph + 1e-9, "adding acid raised the pH from {ph} to {less}");
        }
    }
}

#[test]
fn prop_henderson_hasselbalch_is_a_logarithm_of_its_ratio() {
    let mut rng = Rng::new(0x_C0DE_9022);
    for _ in 0..200 {
        let pka = rng.next_f64() * 14.0;
        let ratio = 10f64.powf(rng.next_f64() * 6.0 - 3.0);
        let ph = buffer_henderson_hasselbalch(pka, ratio).unwrap();
        assert!(close(ph, pka + ratio.log10(), 1e-12));
        // Tenfold more base is one unit up, always.
        assert!(close(
            buffer_henderson_hasselbalch(pka, 10.0 * ratio).unwrap(),
            ph + 1.0,
            1e-12
        ));
        // Equal amounts give the pKa exactly.
        assert!(close(buffer_henderson_hasselbalch(pka, 1.0).unwrap(), pka, 1e-15));
    }
}

// ---------------------------------------------------------------------------
// Rate theory and electrochemistry
// ---------------------------------------------------------------------------

#[test]
fn prop_the_rate_theories_have_the_scalings_they_claim() {
    let mut rng = Rng::new(0x_C0DE_9030);
    for _ in 0..100 {
        let t = 200.0 + rng.next_f64() * 400.0;
        let dh = rng.next_f64() * 120_000.0;
        let ds = rng.next_f64() * 200.0 - 100.0;
        let rate = eyring(dh, ds, t).unwrap();
        assert!(rate > 0.0 && rate.is_finite());
        // An extra RT ln 10 of enthalpy costs exactly one decade.
        let decade =
            eyring(dh + std::f64::consts::LN_10 * 8.314_462_618 * t, ds, t).unwrap();
        assert!(close(decade * 10.0 / rate, 1.0, 1e-9), "an RT ln 10 did not cost a decade");
        // Entropy enters as a pure multiplier.
        assert!(close(
            eyring(dh, ds + 8.314_462_618, t).unwrap() / rate,
            std::f64::consts::E,
            1e-9
        ));
        // A higher barrier is always slower.
        assert!(eyring(dh + 1_000.0, ds, t).unwrap() < rate);

        // Transition-state theory is linear in the transmission coefficient
        // and bounded above by it.
        let dg = rng.next_f64() * 120_000.0;
        let full = transition_state_theory_rate(dg, t, 1.0).unwrap();
        let kappa = rng.next_f64();
        assert!(close(
            transition_state_theory_rate(dg, t, kappa).unwrap(),
            kappa * full,
            1e-9 * full
        ));

        // The Kramers factor is at most one, falls with friction, and
        // depends only on the ratio of friction to barrier frequency.
        let gamma = rng.next_f64() * 40.0;
        let omega = 0.1 + rng.next_f64() * 5.0;
        let factor = kramers_rate_check(gamma, omega).unwrap();
        assert!(factor > 0.0 && factor <= 1.0 + 1e-12, "the factor is {factor}");
        assert!(kramers_rate_check(gamma + 1.0, omega).unwrap() <= factor);
        let scale = 1.0 + rng.next_f64() * 9.0;
        assert!(close(
            kramers_rate_check(gamma * scale, omega * scale).unwrap(),
            factor,
            1e-12
        ));
    }
}

#[test]
fn prop_the_isotope_effect_and_the_relaxation_time_invert_their_definitions() {
    let mut rng = Rng::new(0x_C0DE_9031);
    for _ in 0..200 {
        let t = 150.0 + rng.next_f64() * 500.0;
        let heavy = 500.0 + rng.next_f64() * 2_000.0;
        let light = heavy + rng.next_f64() * 1_500.0;
        let effect = kinetic_isotope_effect_estimate(light, heavy, t).unwrap();
        assert!(effect >= 1.0 - 1e-12, "a normal effect came out below one: {effect}");
        // Swapping the isotopes inverts the ratio exactly.
        assert!(close(
            kinetic_isotope_effect_estimate(heavy, light, t).unwrap(),
            1.0 / effect,
            1e-9 / effect.max(1.0)
        ));
        // Heating always shrinks it toward one.
        assert!(kinetic_isotope_effect_estimate(light, heavy, 2.0 * t).unwrap() <= effect);

        // The relaxation rate is the sum, so it is symmetric in the two
        // constants and shorter than either one alone.
        let kf = 0.01 + rng.next_f64() * 10.0;
        let kr = 0.01 + rng.next_f64() * 10.0;
        let tau = temperature_jump_relaxation(kf, kr).unwrap();
        assert!(close(tau, temperature_jump_relaxation(kr, kf).unwrap(), 1e-15));
        assert!(tau < 1.0 / kf && tau < 1.0 / kr);
        assert!(close(1.0 / tau, kf + kr, 1e-9 * (kf + kr)));
    }
}

#[test]
fn prop_the_electrochemical_relations_scale_as_their_formulas_do() {
    let mut rng = Rng::new(0x_C0DE_9032);
    for _ in 0..150 {
        let t = 250.0 + rng.next_f64() * 200.0;
        let z = 1.0 + f64::from(pick(&mut rng, 3) as u32);
        let ratio = 10f64.powf(rng.next_f64() * 6.0 - 3.0);
        let e0 = rng.next_f64() * 2.0 - 1.0;
        let e = nernst(e0, z, ratio, t).unwrap();
        // A decade in the quotient is one Nernst slope, and the offset is
        // the standard potential exactly.
        let decade = nernst(e0, z, 10.0 * ratio, t).unwrap();
        let slope = std::f64::consts::LN_10 * 8.314_462_618 * t / (z * 96_485.0);
        assert!(close(e - decade, slope, 1e-9 * slope));
        assert!(close(nernst(e0, z, 1.0, t).unwrap(), e0, 1e-12));
        // The standard potential is a pure offset.
        assert!(close(nernst(e0 + 0.3, z, ratio, t).unwrap(), e + 0.3, 1e-12));

        // Butler-Volmer: zero at equilibrium, linear in the exchange
        // current, and of the same sign as the overpotential.
        let i0 = 10f64.powf(rng.next_f64() * 6.0 - 8.0);
        let alpha = 0.1 + rng.next_f64() * 0.8;
        assert!(close(butler_volmer(i0, alpha, 0.0, z, t).unwrap(), 0.0, 1e-20));
        let eta = rng.next_f64() * 0.2 - 0.1;
        let current = butler_volmer(i0, alpha, eta, z, t).unwrap();
        assert!(
            current * eta >= -1e-30,
            "the current opposes the overpotential: {current} at {eta}"
        );
        assert!(close(
            butler_volmer(3.0 * i0, alpha, eta, z, t).unwrap(),
            3.0 * current,
            1e-9 * current.abs().max(1e-30)
        ));

        // Cottrell: inverse square root in time, linear in everything else.
        let area = 0.001 + rng.next_f64();
        let conc = 1e-5 + rng.next_f64() * 0.01;
        let diffusivity = 1e-10 + rng.next_f64() * 1e-8;
        let base = cottrell_current(z, area, conc, diffusivity, 1.0).unwrap();
        assert!(close(
            cottrell_current(z, area, conc, diffusivity, 9.0).unwrap() * 3.0,
            base,
            1e-9 * base
        ));
        assert!(close(
            cottrell_current(z, 2.5 * area, conc, diffusivity, 1.0).unwrap(),
            2.5 * base,
            1e-9 * base
        ));

        // Debye-Huckel: at most one, and log gamma scaling as z squared.
        let ionic = rng.next_f64() * 0.2;
        let single = debye_huckel_activity(1.0, ionic).unwrap();
        assert!(single > 0.0 && single <= 1.0 + 1e-15);
        assert!(close(debye_huckel_activity(3.0, ionic).unwrap(), single.powi(9), 1e-9));
        assert!(close(debye_huckel_activity(-2.0, ionic).unwrap(), single.powi(4), 1e-9));
    }
}

#[test]
fn prop_the_threshold_quantities_are_sharp_where_they_should_be() {
    let mut rng = Rng::new(0x_C0DE_9033);
    for _ in 0..150 {
        // The branching ratio crosses one exactly at equality.
        let k_term = 0.01 + rng.next_f64() * 10.0;
        assert!(close(chain_reaction_criticality(k_term, k_term).unwrap(), 1.0, 1e-15));
        assert!(chain_reaction_criticality(k_term * 0.999, k_term).unwrap() < 1.0);
        assert!(chain_reaction_criticality(k_term * 1.001, k_term).unwrap() > 1.0);

        // Autocatalytic ignition: logarithmic in the seed, inverse in the
        // rate, and zero once the product already leads.
        let a0 = 0.1 + rng.next_f64() * 5.0;
        let b0 = a0 * 10f64.powf(-1.0 - rng.next_f64() * 6.0);
        let k = 0.05 + rng.next_f64() * 5.0;
        let time = autocatalysis_ignition(a0, b0, k).unwrap();
        assert!(time > 0.0);
        assert!(close(
            autocatalysis_ignition(a0, b0, 2.0 * k).unwrap() * 2.0,
            time,
            1e-9 * time
        ));
        assert!(autocatalysis_ignition(a0, b0 * 0.1, k).unwrap() > time);
        assert!(close(autocatalysis_ignition(a0, a0 * 1.5, k).unwrap(), 0.0, 1e-15));

        // The nucleation barrier is inverse square in the driving force and
        // cubic in the surface tension, so the rate is exponentially
        // sensitive to both.
        let sigma = 0.005 + rng.next_f64() * 0.1;
        let density = 1e28 * (0.5 + rng.next_f64());
        let drive = 1e-21 * (0.5 + rng.next_f64() * 3.0);
        let barrier = nucleation_barrier(sigma, density, drive).unwrap();
        assert!(barrier > 0.0 && barrier.is_finite());
        assert!(close(
            nucleation_barrier(sigma, density, 2.0 * drive).unwrap() * 4.0,
            barrier,
            1e-9 * barrier
        ));
        assert!(close(
            nucleation_barrier(2.0 * sigma, density, drive).unwrap(),
            8.0 * barrier,
            1e-9 * barrier
        ));
        let t = 250.0 + rng.next_f64() * 200.0;
        let rate = nucleation_rate_cnt(barrier, 1e35, t).unwrap();
        assert!((0.0..=1e35 + 1.0).contains(&rate));
        assert!(nucleation_rate_cnt(barrier * 1.001, 1e35, t).unwrap() <= rate);
    }
}
