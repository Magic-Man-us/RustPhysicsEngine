//! Properties of the population dynamics and genetics module.
//!
//! The growth laws here are closed-form solutions of differential
//! equations, so on any parameters at all they must satisfy the equation
//! they solve -- a check that needs no reference curve. The genetic results
//! are exact in a different sense: a Wright-Fisher frequency is a
//! martingale, a Moran fixation probability has a closed form, the Price
//! equation is an identity, and Hardy-Weinberg proportions are arithmetic.
//! Those hold at every parameter, so they are the right things to check on
//! random ones.

use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::biophysics::population::{
    allee_effect_ode, balanced_polymorphism, beverton_holt, coalescent_time_expected,
    coalescent_tmrca_expected, coexistence_condition, competition_lv, euler_lotka_solve,
    fixation_probability_moran, fst, genetic_drift_heterozygosity, gompertz, hardy_weinberg,
    hw_chi_square_test, kin_selection_hamilton, leslie_growth_rate, leslie_matrix,
    logistic_growth, lotka_volterra, metapopulation_levins, moran_process,
    mutation_selection_balance, nucleotide_diversity, price_equation_decompose, richards,
    ricker_map, segregating_sites, selection_one_locus, stable_age_distribution, watterson_theta,
    wright_fisher, Competition,
};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

// ---------------------------------------------------------------------------
// Growth
// ---------------------------------------------------------------------------

#[test]
fn prop_the_growth_laws_solve_the_equations_they_claim_to() {
    // Each is an analytic solution, so a central difference of the formula
    // must match the right-hand side of its own differential equation at any
    // parameters. That is a check on the algebra rather than on a
    // remembered curve, and it is the strongest thing available here.
    let mut rng = Rng::new(0x0B10_A001);
    let h = 1e-6;
    for _ in 0..60 {
        let r = 0.05 + rng.next_f64() * 2.0;
        let k = 1.0 + rng.next_f64() * 1_000.0;
        let n0 = 0.01 + rng.next_f64() * k * 0.5;
        let nu = 0.2 + rng.next_f64() * 3.0;
        for step in 1..=10 {
            let t = f64::from(step) * 0.3 / r;
            // Logistic.
            let n = logistic_growth(r, k, n0, t).unwrap();
            let d = (logistic_growth(r, k, n0, t + h).unwrap()
                - logistic_growth(r, k, n0, t - h).unwrap())
                / (2.0 * h);
            let want = r * n * (1.0 - n / k);
            assert!(close(d, want, 1e-4 * want.abs().max(1.0)), "logistic: {d} against {want}");

            // Gompertz.
            let g = gompertz(r, k, n0, t).unwrap();
            let dg = (gompertz(r, k, n0, t + h).unwrap() - gompertz(r, k, n0, t - h).unwrap())
                / (2.0 * h);
            let want_g = r * g * (k / g).ln();
            assert!(
                close(dg, want_g, 1e-4 * want_g.abs().max(1.0)),
                "Gompertz: {dg} against {want_g}"
            );

            // Richards, whose equation carries the r/nu.
            let x = richards(r, k, nu, n0, t).unwrap();
            let dx = (richards(r, k, nu, n0, t + h).unwrap()
                - richards(r, k, nu, n0, t - h).unwrap())
                / (2.0 * h);
            let want_x = (r / nu) * x * (1.0 - (x / k).powf(nu));
            assert!(
                close(dx, want_x, 1e-3 * want_x.abs().max(1.0)),
                "Richards at nu = {nu}: {dx} against {want_x}"
            );
        }
        // All three start where told and end at capacity, monotonically.
        for value in [
            logistic_growth(r, k, n0, 0.0).unwrap(),
            gompertz(r, k, n0, 0.0).unwrap(),
            richards(r, k, nu, n0, 0.0).unwrap(),
        ] {
            assert!(close(value, n0, 1e-8 * n0.max(1.0)), "a curve started at {value}, not {n0}");
        }
        for value in [
            logistic_growth(r, k, n0, 500.0 / r).unwrap(),
            gompertz(r, k, n0, 500.0 / r).unwrap(),
            richards(r, k, nu, n0, 500.0 / r).unwrap(),
        ] {
            assert!(close(value, k, 1e-5 * k), "a curve ended at {value}, not {k}");
        }
        // Richards at nu = 1 is exactly the logistic.
        for step in 0..8 {
            let t = f64::from(step) * 0.5 / r;
            assert!(close(
                richards(r, k, 1.0, n0, t).unwrap(),
                logistic_growth(r, k, n0, t).unwrap(),
                1e-8 * k
            ));
        }
    }
}

#[test]
fn prop_the_allee_threshold_decides_extinction_from_either_side() {
    let mut rng = Rng::new(0x0B10_A002);
    for _ in 0..25 {
        let r = 0.2 + rng.next_f64() * 2.0;
        let a = 1.0 + rng.next_f64() * 40.0;
        let k = a * (1.5 + rng.next_f64() * 8.0);
        let horizon = 300.0 / r;
        let below = allee_effect_ode(r, a, k, a * 0.9, horizon).unwrap();
        assert!(
            below.last().unwrap().1 < 1e-3 * a,
            "below the threshold it reached {}",
            below.last().unwrap().1
        );
        let above = allee_effect_ode(r, a, k, a * 1.1, horizon).unwrap();
        assert!(
            close(above.last().unwrap().1, k, 1e-3 * k),
            "above the threshold it reached {} rather than {k}",
            above.last().unwrap().1
        );
        for (_, n) in &below {
            assert!(*n >= -1e-9 && n.is_finite());
        }
    }
}

#[test]
fn prop_the_metapopulation_equilibrium_is_one_minus_the_rate_ratio() {
    let mut rng = Rng::new(0x0B10_A003);
    for _ in 0..40 {
        let c = 0.05 + rng.next_f64() * 2.0;
        let e = 0.05 + rng.next_f64() * 2.0;
        let p0 = 0.05 + rng.next_f64() * 0.9;
        let trace = metapopulation_levins(c, e, p0, 2_000.0 / c.min(e)).unwrap();
        let end = trace.last().unwrap().1;
        if c > e * 1.02 {
            assert!(
                close(end, 1.0 - e / c, 1e-4),
                "c = {c}, e = {e} settled at {end} rather than {}",
                1.0 - e / c
            );
        } else if e > c * 1.02 {
            assert!(end < 1e-3, "c = {c}, e = {e} persisted at {end}");
        }
        for (_, p) in &trace {
            assert!((-1e-9..=1.0 + 1e-9).contains(p), "occupancy left [0, 1]: {p}");
        }
    }
}

// ---------------------------------------------------------------------------
// Interacting species
// ---------------------------------------------------------------------------

#[test]
fn prop_the_lotka_volterra_invariant_is_conserved_on_every_orbit() {
    let mut rng = Rng::new(0x0B10_A010);
    for _ in 0..20 {
        let alpha = 0.3 + rng.next_f64() * 1.5;
        let beta = 0.2 + rng.next_f64();
        let delta = 0.2 + rng.next_f64();
        let gamma = 0.3 + rng.next_f64() * 1.5;
        // Start near the fixed point, so the orbit stays in a range the
        // integrator resolves well.
        let x0 = (gamma / delta) * (0.5 + rng.next_f64());
        let y0 = (alpha / beta) * (0.5 + rng.next_f64());
        let (trace, invariant) =
            lotka_volterra(alpha, beta, delta, gamma, x0, y0, 40.0).unwrap();
        let first = invariant[0];
        for v in &invariant {
            assert!(
                close(*v, first, 1e-4 * first.abs().max(1.0)),
                "the invariant drifted from {first} to {v}"
            );
        }
        for (_, x, y) in &trace {
            assert!(*x > 0.0 && *y > 0.0 && x.is_finite() && y.is_finite());
        }
        // Starting exactly at the fixed point leaves it there.
        let (fixed, _) =
            lotka_volterra(alpha, beta, delta, gamma, gamma / delta, alpha / beta, 40.0).unwrap();
        for (_, x, y) in &fixed {
            assert!(close(*x, gamma / delta, 1e-6 * gamma / delta));
            assert!(close(*y, alpha / beta, 1e-6 * alpha / beta));
        }
    }
}

#[test]
fn prop_the_competition_criterion_predicts_the_integrated_outcome() {
    // The criterion is algebra and the integration is not, so agreement at
    // random parameters is evidence about both.
    let mut rng = Rng::new(0x0B10_A011);
    for _ in 0..40 {
        let k1 = 20.0 + rng.next_f64() * 200.0;
        let k2 = 20.0 + rng.next_f64() * 200.0;
        let a12 = rng.next_f64() * 2.5;
        let a21 = rng.next_f64() * 2.5;
        // Skip anything near a boundary, where a finite run cannot decide.
        if (a12 - k1 / k2).abs() < 0.08 || (a21 - k2 / k1).abs() < 0.08 {
            continue;
        }
        let verdict = coexistence_condition(k1, k2, a12, a21).unwrap();
        let trace = competition_lv(0.8, 0.9, k1, k2, a12, a21, k1 * 0.2, k2 * 0.2, 3_000.0)
            .unwrap();
        let (_, n1, n2) = *trace.last().unwrap();
        match verdict {
            Competition::Coexistence => {
                let denominator = 1.0 - a12 * a21;
                let want1 = (k1 - a12 * k2) / denominator;
                let want2 = (k2 - a21 * k1) / denominator;
                assert!(close(n1, want1, 1e-3 * want1.max(1.0)), "N1 is {n1} against {want1}");
                assert!(close(n2, want2, 1e-3 * want2.max(1.0)), "N2 is {n2} against {want2}");
            }
            Competition::FirstExcludes => {
                assert!(n2 < 1e-3 * k2 && close(n1, k1, 1e-3 * k1), "gave {n1} and {n2}");
            }
            Competition::SecondExcludes => {
                assert!(n1 < 1e-3 * k1 && close(n2, k2, 1e-3 * k2), "gave {n1} and {n2}");
            }
            Competition::FounderControl => {
                assert!(
                    (n1 < 1e-3 * k1) != (n2 < 1e-3 * k2),
                    "founder control left both at {n1} and {n2}"
                );
            }
        }
        for (_, a, b) in &trace {
            assert!(*a >= -1e-9 && *b >= -1e-9 && a.is_finite() && b.is_finite());
        }
    }
}

// ---------------------------------------------------------------------------
// Age structure and maps
// ---------------------------------------------------------------------------

#[test]
fn prop_the_two_routes_to_the_growth_rate_agree() {
    // The matrix eigenvalue and the Euler-Lotka root are computed by
    // entirely different means, and the stable distribution must be a true
    // eigenvector of the matrix. All three are checked on random life
    // tables.
    let mut rng = Rng::new(0x0B10_A020);
    for _ in 0..40 {
        let classes = 2 + pick(&mut rng, 5);
        // Fecundity spread over at least two classes, so the matrix is
        // primitive and the iteration settles.
        let fecundity: Vec<f64> = (0..classes)
            .map(|i| if i == 0 { 0.0 } else { rng.next_f64() * 3.0 })
            .collect();
        if fecundity.iter().filter(|f| **f > 0.05).count() < 2 {
            continue;
        }
        let survival: Vec<f64> = (0..classes - 1).map(|_| 0.2 + rng.next_f64() * 0.75).collect();
        let l = leslie_matrix(&fecundity, &survival).unwrap();
        let Ok((lambda, distribution)) = leslie_growth_rate(&l) else {
            continue;
        };
        assert!(lambda > 0.0 && lambda.is_finite());
        assert!(close(distribution.iter().sum::<f64>(), 1.0, 1e-12));
        assert_eq!(stable_age_distribution(&l).unwrap(), distribution);

        // An eigenvector, exactly.
        for i in 0..classes {
            let applied: f64 = (0..classes).map(|j| l.get(i, j) * distribution[j]).sum();
            assert!(
                close(applied, lambda * distribution[i], 1e-7 * lambda),
                "class {i} is not an eigenvector component"
            );
        }

        // The life table gives the same lambda.
        let mut lx = Vec::with_capacity(classes);
        let mut running = 1.0;
        for i in 0..classes {
            if i > 0 {
                running *= survival[i - 1];
            }
            lx.push(running);
        }
        let r = euler_lotka_solve(&lx, &fecundity).unwrap();
        assert!(close(r, lambda, 1e-5 * lambda), "Euler-Lotka gives {r} against {lambda}");

        // Scaling every fecundity by c scales the net reproductive rate and
        // moves lambda the same way, monotonically.
        let doubled: Vec<f64> = fecundity.iter().map(|f| f * 2.0).collect();
        assert!(euler_lotka_solve(&lx, &doubled).unwrap() > r);
    }
}

#[test]
fn prop_beverton_holt_matches_its_closed_form_and_never_overshoots() {
    let mut rng = Rng::new(0x0B10_A021);
    for _ in 0..40 {
        let ratio = 1.05 + rng.next_f64() * 9.0;
        let k = 1.0 + rng.next_f64() * 500.0;
        let n0 = 0.01 + rng.next_f64() * k * 3.0;
        let trace = beverton_holt(ratio, k, n0, 30).unwrap();
        for (t, n) in trace.iter().enumerate() {
            let expected = k * n0 / (n0 + (k - n0) * ratio.powi(-(t as i32)));
            assert!(
                close(*n, expected, 1e-8 * expected.abs().max(1.0)),
                "R = {ratio}, t = {t}: {n} against {expected}"
            );
            assert!(*n >= 0.0 && n.is_finite());
        }
        // Monotone and never past K: compensating density dependence has no
        // route to chaos at any R, unlike Ricker.
        for pair in trace.windows(2) {
            if n0 < k {
                assert!(pair[1] >= pair[0] - 1e-9 && pair[1] <= k + 1e-6);
            } else {
                assert!(pair[1] <= pair[0] + 1e-9 && pair[1] >= k - 1e-6);
            }
        }
    }
    // The Ricker map, by contrast, overshoots from above capacity whenever
    // the growth rate is appreciable.
    let overshoot = ricker_map(2.0, 1.0, 3.0, 1).unwrap();
    assert!(overshoot[1] < 1.0, "Ricker did not overshoot downward");
}

// ---------------------------------------------------------------------------
// Genetics
// ---------------------------------------------------------------------------

#[test]
fn prop_wright_fisher_preserves_its_expected_frequency() {
    // A martingale: the expected frequency is unchanged by any number of
    // generations, at any population size and any starting frequency. It is
    // also the fixation probability, since the only absorbing states are
    // zero and one.
    let mut rng = Rng::new(0x0B10_A030);
    for trial in 0..12 {
        let n = 8 + (trial as u64) * 5;
        let p0 = 0.15 + (trial % 5) as f64 * 0.15;
        let runs = 3_000;
        let mut sum = 0.0;
        let mut fixed = 0;
        for _ in 0..runs {
            let trace = wright_fisher(n, p0, 40, &mut rng).unwrap();
            assert_eq!(trace.len(), 41);
            for p in &trace {
                assert!((0.0..=1.0).contains(p), "a frequency left [0, 1]: {p}");
                // Frequencies are multiples of 1/(2N), always.
                let copies = 2.0 * n as f64;
                assert!(close(p * copies, (p * copies).round(), 1e-9));
            }
            sum += *trace.last().unwrap();
            let long = wright_fisher(n, p0, 40 * n as usize, &mut rng).unwrap();
            let end = *long.last().unwrap();
            if end >= 1.0 - 1e-12 {
                fixed += 1;
            }
        }
        let mean = sum / f64::from(runs);
        let started = (p0 * 2.0 * n as f64).round() / (2.0 * n as f64);
        assert!(
            close(mean, started, 0.03),
            "at N = {n}, p0 = {started} the mean drifted to {mean}"
        );
        let fixation = f64::from(fixed) / f64::from(runs);
        assert!(
            close(fixation, started, 0.04),
            "at N = {n}, p0 = {started} the fixation rate is {fixation}"
        );
    }
}

#[test]
fn prop_the_moran_formula_matches_its_own_simulation() {
    let mut rng = Rng::new(0x0B10_A031);
    for trial in 0..10 {
        let n = 10 + (trial as u64) * 3;
        let i0 = 1 + (trial as u64) % (n - 1);
        let fitness = [0.6f64, 1.0, 1.4, 2.2][trial % 4];
        let predicted = fixation_probability_moran(n, i0, fitness).unwrap();
        assert!((0.0..=1.0).contains(&predicted));
        let runs = 4_000;
        let mut fixed = 0;
        for _ in 0..runs {
            let (won, steps) = moran_process(n, i0, fitness, &mut rng).unwrap();
            assert!(steps > 0, "an unresolved population took no steps");
            if won {
                fixed += 1;
            }
        }
        let observed = f64::from(fixed) / f64::from(runs);
        assert!(
            close(observed, predicted, 0.03),
            "n = {n}, i0 = {i0}, r = {fitness}: {observed} against {predicted}"
        );
        // Monotone in the starting count and in fitness.
        if i0 + 1 < n {
            assert!(fixation_probability_moran(n, i0 + 1, fitness).unwrap() > predicted);
        }
        assert!(fixation_probability_moran(n, i0, fitness * 1.2).unwrap() > predicted);
    }
    // Neutral drift fixes with probability equal to the starting frequency,
    // exactly, at every size.
    for n in [3u64, 11, 64, 500] {
        for i in 0..=n {
            assert!(close(
                fixation_probability_moran(n, i, 1.0).unwrap(),
                i as f64 / n as f64,
                1e-12
            ));
        }
    }
}

#[test]
fn prop_heterozygosity_decays_geometrically_in_the_gene_copy_count() {
    let mut rng = Rng::new(0x0B10_A032);
    for _ in 0..100 {
        let n = 1 + (rng.next_f64() * 500.0) as u64;
        let h0 = rng.next_f64();
        let t = rng.next_f64() * 200.0;
        let h = genetic_drift_heterozygosity(n, h0, t).unwrap();
        assert!(h >= 0.0 && h <= h0 + 1e-15, "heterozygosity {h} is outside [0, {h0}]");
        // Each generation multiplies by exactly 1 - 1/(2N).
        let next = genetic_drift_heterozygosity(n, h0, t + 1.0).unwrap();
        assert!(close(next, h * (1.0 - 1.0 / (2.0 * n as f64)), 1e-12 * h0.max(1e-12)));
        // A larger population loses less.
        if n < 400 {
            assert!(genetic_drift_heterozygosity(n * 2, h0, t).unwrap() >= h);
        }
        assert!(close(genetic_drift_heterozygosity(n, h0, 0.0).unwrap(), h0, 1e-15));
    }
}

#[test]
fn prop_hardy_weinberg_is_arithmetic_and_its_test_is_calibrated() {
    let mut rng = Rng::new(0x0B10_A033);
    for _ in 0..200 {
        let p = rng.next_f64();
        let (aa, ab, bb) = hardy_weinberg(p).unwrap();
        assert!(close(aa + ab + bb, 1.0, 1e-15));
        assert!(close(aa + ab / 2.0, p, 1e-12), "the allele frequency is not recovered");
        assert!(ab <= 0.5 + 1e-15, "heterozygosity {ab} exceeds a half");
        // Exact proportions give a statistic of zero at any sample size.
        if p > 0.02 && p < 0.98 {
            let total = 200.0 + rng.next_f64() * 5_000.0;
            let result = hw_chi_square_test([aa * total, ab * total, bb * total]).unwrap();
            assert!(
                close(result.statistic, 0.0, 1e-8),
                "exact proportions gave a statistic of {}",
                result.statistic
            );
            assert!(close(result.df, 1.0, 1e-15));
            assert!(result.p_value > 0.99);
            // Removing every heterozygote is rejected outright.
            let inbred = hw_chi_square_test([p * total, 0.0, (1.0 - p) * total]).unwrap();
            assert!(
                inbred.p_value < 1e-6,
                "no heterozygotes passed at p = {}",
                inbred.p_value
            );
        }
    }
}

#[test]
fn prop_selection_converges_to_the_equilibrium_its_fitnesses_imply() {
    let mut rng = Rng::new(0x0B10_A034);
    for _ in 0..40 {
        // Heterozygote advantage: an interior equilibrium reached from both
        // sides, at a point with a closed form.
        let low = 0.1 + rng.next_f64() * 0.7;
        let other = 0.1 + rng.next_f64() * 0.7;
        let w = [low, 1.0, other];
        let star = balanced_polymorphism(w).unwrap();
        assert!((0.0..1.0).contains(&star), "the equilibrium is {star}");
        for &p0 in &[0.02f64, 0.5, 0.98] {
            let trace = selection_one_locus(p0, w, 20_000).unwrap();
            assert!(
                close(*trace.last().unwrap(), star, 1e-5),
                "from {p0} it settled at {} rather than {star}",
                trace.last().unwrap()
            );
            for p in &trace {
                assert!((0.0..=1.0).contains(p), "a frequency left [0, 1]: {p}");
            }
        }
        // Directional selection fixes the fitter allele, monotonically.
        let directional = [1.0, 0.5 + rng.next_f64() * 0.4, 0.1 + rng.next_f64() * 0.3];
        if directional[1] > directional[2] {
            let trace = selection_one_locus(0.1, directional, 20_000).unwrap();
            assert!(close(*trace.last().unwrap(), 1.0, 1e-5));
            for pair in trace.windows(2) {
                assert!(pair[1] >= pair[0] - 1e-15, "the frequency fell");
            }
        }
        // Neutrality changes nothing at all.
        let start = rng.next_f64();
        let flat = selection_one_locus(start, [1.0; 3], 200).unwrap();
        assert!(flat.iter().all(|p| close(*p, start, 1e-12)));
    }
}

#[test]
fn prop_the_mutation_selection_balance_scales_as_its_two_regimes_say() {
    let mut rng = Rng::new(0x0B10_A035);
    for _ in 0..200 {
        let mu = 10f64.powf(-8.0 + rng.next_f64() * 4.0);
        let s = 0.001 + rng.next_f64() * 0.5;
        // Fully recessive: sqrt(mu/s), so quadrupling mu doubles it.
        let recessive = mutation_selection_balance(mu, s, 0.0).unwrap();
        assert!(close(recessive, (mu / s).sqrt().min(1.0), 1e-12));
        if 4.0 * mu < s {
            assert!(close(
                mutation_selection_balance(4.0 * mu, s, 0.0).unwrap(),
                (2.0 * recessive).min(1.0),
                1e-9
            ));
        }
        // With dominance: mu/(h s), linear in mu.
        let h = 0.05 + rng.next_f64() * 0.9;
        if h * s > mu {
            let partial = mutation_selection_balance(mu, s, h).unwrap();
            assert!(close(partial, (mu / (h * s)).min(1.0), 1e-12));
            assert!(partial <= recessive + 1e-12, "dominance made the allele commoner");
            if 2.0 * mu < h * s {
                assert!(close(
                    mutation_selection_balance(2.0 * mu, s, h).unwrap(),
                    (2.0 * partial).min(1.0),
                    1e-9
                ));
            }
        }
        assert!((0.0..=1.0).contains(&recessive));
    }
}

#[test]
fn prop_the_price_equation_is_an_identity_at_any_numbers() {
    // It assumes nothing about inheritance or fitness, so the two terms sum
    // to the change in the mean trait for any population whatever. That is
    // what makes it worth having, and it is exactly checkable.
    let mut rng = Rng::new(0x0B10_A036);
    for _ in 0..400 {
        let n = 2 + pick(&mut rng, 20);
        let z: Vec<f64> = (0..n).map(|_| rng.next_f64() * 40.0 - 20.0).collect();
        let w: Vec<f64> = (0..n).map(|_| rng.next_f64() * 5.0).collect();
        let offspring: Vec<f64> = (0..n).map(|k| z[k] + rng.next_f64() * 4.0 - 2.0).collect();
        let mean_w: f64 = w.iter().sum::<f64>() / n as f64;
        if mean_w < 1e-6 {
            continue;
        }
        let (selection, transmission) = price_equation_decompose(&z, &w, &offspring).unwrap();
        let parent_mean: f64 = z.iter().sum::<f64>() / n as f64;
        let offspring_mean: f64 =
            (0..n).map(|k| w[k] * offspring[k]).sum::<f64>() / (n as f64 * mean_w);
        assert!(
            close(selection + transmission, offspring_mean - parent_mean, 1e-8),
            "the terms sum to {} against a change of {}",
            selection + transmission,
            offspring_mean - parent_mean
        );
        // Faithful transmission puts everything in the selection term.
        let (_, faithful) = price_equation_decompose(&z, &w, &z).unwrap();
        assert!(close(faithful, 0.0, 1e-12), "faithful inheritance transmitted {faithful}");
        // Equal fitness puts everything in the transmission term.
        let flat = vec![1.0; n];
        let (neutral, _) = price_equation_decompose(&z, &flat, &offspring).unwrap();
        assert!(close(neutral, 0.0, 1e-12), "equal fitness selected {neutral}");
    }
    // Hamilton's rule is just the comparison it claims to be.
    let mut rng = Rng::new(0x0B10_A037);
    for _ in 0..500 {
        let r = rng.next_f64();
        let b = rng.next_f64() * 10.0;
        let c = rng.next_f64() * 10.0;
        assert_eq!(kin_selection_hamilton(r, b, c).unwrap(), r * b > c);
    }
}

// ---------------------------------------------------------------------------
// Coalescent and diversity
// ---------------------------------------------------------------------------

#[test]
fn prop_the_coalescent_expectations_sum_and_bound_as_the_theory_says() {
    let mut rng = Rng::new(0x0B10_A040);
    for _ in 0..100 {
        let n = 1 + (rng.next_f64() * 10_000.0) as u64;
        let samples = 2 + (rng.next_f64() * 200.0) as u64;
        // The tree height is the sum of the intervals, exactly.
        let summed: f64 = (2..=samples)
            .map(|k| coalescent_time_expected(n, k).unwrap())
            .sum();
        let height = coalescent_tmrca_expected(n, samples).unwrap();
        assert!(
            close(summed, height, 1e-9 * height),
            "the intervals sum to {summed} against a height of {height}"
        );
        // Bounded by 4N however large the sample.
        assert!(height < 4.0 * n as f64, "the height {height} exceeds 4N");
        // The deepest interval is at least half the height.
        let t2 = coalescent_time_expected(n, 2).unwrap();
        assert!(t2 >= 0.5 * height - 1e-9, "T_2 is {t2} of a height of {height}");
        // Intervals shorten as lineages accumulate.
        if samples > 3 {
            assert!(
                coalescent_time_expected(n, samples).unwrap()
                    < coalescent_time_expected(n, samples - 1).unwrap()
            );
        }
    }
}

#[test]
fn prop_the_diversity_statistics_count_what_they_are_defined_from() {
    // Built from alignments whose statistics can be derived directly, so the
    // check is a definition rather than a reference value.
    let mut rng = Rng::new(0x0B10_A041);
    for _ in 0..60 {
        let n = 2 + pick(&mut rng, 8);
        let length = 4 + pick(&mut rng, 30);
        let sequences: Vec<Vec<u8>> = (0..n)
            .map(|_| (0..length).map(|_| if rng.next_f64() < 0.5 { b'A' } else { b'T' }).collect())
            .collect();
        let s = segregating_sites(&sequences).unwrap();
        assert!(s <= length, "more segregating sites than sites");
        // Counted directly.
        let direct = (0..length)
            .filter(|k| sequences.iter().any(|seq| seq[*k] != sequences[0][*k]))
            .count();
        assert_eq!(s, direct);
        // Nucleotide diversity is the mean pairwise distance, counted by
        // hand over all pairs.
        let pi = nucleotide_diversity(&sequences).unwrap();
        let mut total = 0.0;
        let mut pairs = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += (0..length).filter(|k| sequences[i][*k] != sequences[j][*k]).count()
                    as f64;
                pairs += 1.0;
            }
        }
        assert!(close(pi, total / pairs, 1e-12));
        assert!(pi <= s as f64 + 1e-12, "pi exceeds the segregating site count");
        // Watterson inverts its own normalisation exactly.
        let a: f64 = (1..n as u64).map(|i| 1.0 / i as f64).sum();
        assert!(close(watterson_theta(s as f64, n as u64).unwrap(), s as f64 / a, 1e-12));
        assert!(close(watterson_theta(7.0 * a, n as u64).unwrap(), 7.0, 1e-9));
    }
}

#[test]
fn prop_fst_is_a_fraction_that_vanishes_only_without_structure() {
    let mut rng = Rng::new(0x0B10_A042);
    for _ in 0..400 {
        let count = 2 + pick(&mut rng, 8);
        let freqs: Vec<f64> = (0..count).map(|_| rng.next_f64()).collect();
        let Ok(value) = fst(&freqs) else {
            continue;
        };
        assert!((0.0..=1.0).contains(&value), "Fst left [0, 1]: {value}");
        // Identical subpopulations have none.
        let mean: f64 = freqs.iter().sum::<f64>() / count as f64;
        if mean > 0.02 && mean < 0.98 {
            let uniform = vec![mean; count];
            assert!(close(fst(&uniform).unwrap(), 0.0, 1e-12));
        }
        // Fixed differences give one.
        let split: Vec<f64> = (0..count).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        if count >= 2 {
            assert!(close(fst(&split).unwrap(), 1.0, 1e-12));
        }
        // It measures variance in frequency: spreading the same mean
        // further apart can only raise it.
        let d = 0.1 * rng.next_f64();
        let tight = fst(&[0.5 - d, 0.5 + d]).unwrap();
        let wide = fst(&[0.5 - 2.0 * d, 0.5 + 2.0 * d]).unwrap();
        assert!(wide >= tight - 1e-12, "a wider spread gave a smaller Fst");
    }
}
