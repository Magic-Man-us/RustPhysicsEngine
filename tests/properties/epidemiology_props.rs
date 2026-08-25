//! Properties of the epidemiology module.
//!
//! Compartment models carry an invariant that holds on every trajectory
//! whatever the parameters -- the population is conserved and no compartment
//! goes negative -- and a threshold that is exact rather than approximate:
//! an epidemic grows if and only if the effective reproduction number
//! exceeds one. The estimators here invert closed forms, so on data
//! generated from a renewal process they must return the reproduction
//! number that generated it, at any value and any serial interval.

use rust_physics_engine::graph::Graph;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::biophysics::epidemiology::{
    age_structured, effective_r_estimate, epidemic_threshold_network,
    extinction_probability_epidemic, final_size_equation, herd_immunity_threshold, msir,
    network_sir, r0_age_structured, r0_sir, seir, seirs, serial_interval_fit, sir,
    sir_stochastic_gillespie, sir_with_demography, sir_with_vaccination, sirs, sis, two_strain,
    wallinga_teunis, EpidemicSample,
};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// Every compartment sums to one and none goes negative.
fn assert_well_formed(trace: &[EpidemicSample], label: &str) {
    assert!(trace.len() > 2, "{label} produced {} samples", trace.len());
    for sample in trace {
        assert!(
            close(sample.total(), 1.0, 1e-6),
            "{label} at t = {} sums to {}",
            sample.t,
            sample.total()
        );
        for value in [sample.s, sample.e, sample.i, sample.r] {
            assert!(value >= -1e-8, "{label} went negative: {value}");
            assert!(value <= 1.0 + 1e-8, "{label} exceeded the population: {value}");
        }
        assert!(sample.t.is_finite());
    }
    for pair in trace.windows(2) {
        assert!(pair[1].t > pair[0].t, "{label} did not advance in time");
    }
}

// ---------------------------------------------------------------------------
// Compartment models
// ---------------------------------------------------------------------------

#[test]
fn prop_every_model_conserves_its_population_at_any_parameters() {
    let mut rng = Rng::new(0x0B10_9001);
    for _ in 0..40 {
        let beta = rng.next_f64() * 2.0;
        let gamma = 0.02 + rng.next_f64() * 1.5;
        let sigma = 0.02 + rng.next_f64() * 2.0;
        let omega = rng.next_f64() * 0.3;
        let mu = rng.next_f64() * 0.05;
        let i0 = 10f64.powf(-6.0 + rng.next_f64() * 4.0);
        let s0 = 1.0 - i0;
        let t_end = 20.0 + rng.next_f64() * 200.0;
        assert_well_formed(&sir(beta, gamma, s0, i0, t_end).unwrap(), "SIR");
        assert_well_formed(&sis(beta, gamma, s0, i0, t_end).unwrap(), "SIS");
        assert_well_formed(&sirs(beta, gamma, omega, s0, i0, t_end).unwrap(), "SIRS");
        assert_well_formed(&seir(beta, sigma, gamma, s0, 0.0, i0, t_end).unwrap(), "SEIR");
        assert_well_formed(
            &seirs(beta, sigma, gamma, omega, s0, 0.0, i0, t_end).unwrap(),
            "SEIRS",
        );
        assert_well_formed(
            &sir_with_demography(beta, gamma, mu, s0, i0, t_end).unwrap(),
            "SIR with demography",
        );
        // And the models that report their own tuples.
        let m0 = rng.next_f64() * 0.3;
        for (t, a, b, c, d) in msir(beta, gamma, 0.1, m0, 1.0 - m0 - i0, i0, t_end).unwrap() {
            assert!(close(a + b + c + d, 1.0, 1e-6), "MSIR at t = {t} sums to {}", a + b + c + d);
            assert!([a, b, c, d].iter().all(|v| *v >= -1e-8));
        }
        let split = i0 * 0.5;
        for (t, a, b, c, d) in
            two_strain(beta, gamma, beta * 0.7, gamma, 1.0 - i0, split, split, t_end).unwrap()
        {
            assert!(
                close(a + b + c + d, 1.0, 1e-6),
                "two-strain at t = {t} sums to {}",
                a + b + c + d
            );
            assert!([a, b, c, d].iter().all(|v| *v >= -1e-8));
        }
    }
}

#[test]
fn prop_the_epidemic_grows_exactly_when_the_effective_reproduction_number_exceeds_one() {
    // The threshold is sharp, not gradual: the initial growth rate is
    // gamma (R0 S0 - 1), so its sign is decided by that product alone. It is
    // checked from both sides across random parameters, including the
    // partially immune populations where R0 alone would give the wrong
    // answer.
    let mut rng = Rng::new(0x0B10_9002);
    for _ in 0..60 {
        let gamma = 0.05 + rng.next_f64();
        let r0 = 0.2 + rng.next_f64() * 5.0;
        let beta = r0 * gamma;
        let s0 = 0.1 + rng.next_f64() * 0.9;
        let i0 = 1e-7;
        if (r0 * s0 - 1.0).abs() < 0.05 {
            // Too close to the threshold for a finite run to decide.
            continue;
        }
        let trace = sir(beta, gamma, s0.min(1.0 - i0), i0, 5.0 / gamma).unwrap();
        let peak = trace.iter().map(|x| x.i).fold(0.0f64, f64::max);
        if r0 * s0 > 1.0 {
            assert!(peak > 1.5 * i0, "R0 S0 = {} but the epidemic did not grow", r0 * s0);
        } else {
            assert!(
                peak <= i0 * 1.000_001,
                "R0 S0 = {} but the epidemic grew to {peak}",
                r0 * s0
            );
        }
        // The susceptible fraction only ever falls, and the removed only
        // rises -- true of SIR at every parameter.
        for pair in trace.windows(2) {
            assert!(pair[1].s <= pair[0].s + 1e-12, "susceptibles increased");
            assert!(pair[1].r >= pair[0].r - 1e-12, "removed decreased");
        }
    }
}

#[test]
fn prop_the_final_size_solves_its_own_equation_and_overshoots_herd_immunity() {
    let mut rng = Rng::new(0x0B10_9003);
    for _ in 0..300 {
        let r0 = 1.0 + rng.next_f64() * 30.0;
        let z = final_size_equation(r0).unwrap();
        assert!(z > 0.0 && z < 1.0, "at R0 = {r0} the final size is {z}");
        assert!(
            close(1.0 - z, (-r0 * z).exp(), 1e-10),
            "at R0 = {r0} the root {z} does not satisfy 1 - z = exp(-R0 z)"
        );
        // Always beyond the herd immunity threshold: the people already
        // infectious when it is reached go on infecting.
        let threshold = herd_immunity_threshold(r0).unwrap();
        assert!(z > threshold, "at R0 = {r0} the final size {z} undershot {threshold}");
        // Both rise with R0.
        let higher = final_size_equation(r0 + 0.1).unwrap();
        assert!(higher > z);
        assert!(herd_immunity_threshold(r0 + 0.1).unwrap() > threshold);
        // And R0 itself is just the ratio.
        let gamma = 0.05 + rng.next_f64();
        assert!(close(r0_sir(r0 * gamma, gamma).unwrap(), r0, 1e-9 * r0));
    }
    // Below threshold there is no epidemic and extinction is certain.
    for step in 0..50 {
        let r0 = f64::from(step) * 0.02;
        assert!(close(final_size_equation(r0).unwrap(), 0.0, 1e-12));
        assert!(close(extinction_probability_epidemic(r0, 3).unwrap(), 1.0, 1e-15));
    }
}

#[test]
fn prop_the_integrated_final_size_matches_the_implicit_solution() {
    // The integration and the transcendental equation are entirely separate
    // routes to the same number: one steps the differential equations, the
    // other solves an algebraic relation derived from them. Agreement across
    // random parameters is evidence about both.
    let mut rng = Rng::new(0x0B10_9004);
    for _ in 0..25 {
        let gamma = 0.1 + rng.next_f64() * 0.5;
        let r0 = 1.3 + rng.next_f64() * 4.0;
        let beta = r0 * gamma;
        let i0 = 1e-7;
        // Long enough for the epidemic to finish: the growth rate is
        // gamma (R0 - 1) and it must climb seven decades.
        let horizon = 200.0 / gamma + 40.0 / (gamma * (r0 - 1.0));
        let trace = sir(beta, gamma, 1.0 - i0, i0, horizon).unwrap();
        assert!(
            trace.last().unwrap().i < 1e-9,
            "at R0 = {r0} the epidemic had not finished"
        );
        let ever = 1.0 - trace.last().unwrap().s;
        let predicted = final_size_equation(r0).unwrap();
        assert!(
            close(ever, predicted, 3e-3),
            "at R0 = {r0} the integration gives {ever} against {predicted}"
        );
    }
}

#[test]
fn prop_vaccination_is_equivalent_to_removing_susceptibles() {
    // Which is the reason the threshold coverage is the herd immunity
    // threshold: vaccinating a fraction is the same system as starting with
    // that fraction already immune.
    let mut rng = Rng::new(0x0B10_9005);
    for _ in 0..25 {
        let gamma = 0.1 + rng.next_f64() * 0.5;
        let r0 = 0.5 + rng.next_f64() * 5.0;
        let beta = r0 * gamma;
        let coverage = rng.next_f64() * 0.9;
        let i0 = 1e-6;
        let vaccinated = sir_with_vaccination(beta, gamma, coverage, i0, 300.0 / gamma).unwrap();
        let equivalent = sir(beta, gamma, 1.0 - coverage - i0, i0, 300.0 / gamma).unwrap();
        assert_eq!(vaccinated.len(), equivalent.len());
        for (a, b) in vaccinated.iter().zip(&equivalent) {
            assert!(close(a.s, b.s, 1e-12) && close(a.i, b.i, 1e-12) && close(a.r, b.r, 1e-12));
        }
        assert_well_formed(&vaccinated, "vaccinated SIR");
        // Above the threshold coverage nothing takes off.
        if r0 > 1.0 {
            let threshold = herd_immunity_threshold(r0).unwrap();
            let protected =
                sir_with_vaccination(beta, gamma, (threshold + 0.02).min(0.999), i0, 300.0 / gamma)
                    .unwrap();
            let peak = protected.iter().map(|x| x.i).fold(0.0f64, f64::max);
            assert!(peak <= i0 * 1.000_001, "an epidemic ran above herd immunity: {peak}");
        }
    }
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

#[test]
fn prop_the_network_threshold_is_the_reciprocal_spectral_radius() {
    // Checked against a direct power iteration on the adjacency matrix, an
    // independent route to the same eigenvalue, over random graphs.
    let mut rng = Rng::new(0x0B10_9010);
    for trial in 0..20 {
        let n = 6 + trial % 10;
        let mut g = Graph::new(n, false);
        let mut edges = 0;
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.next_f64() < 0.35 {
                    g.add_edge(u, v, 1.0);
                    edges += 1;
                }
            }
        }
        if edges == 0 {
            assert!(epidemic_threshold_network(&g).is_err());
            continue;
        }
        let threshold = epidemic_threshold_network(&g).unwrap();
        assert!(threshold > 0.0 && threshold.is_finite());
        // Power iteration on the adjacency matrix, shifted to be positive
        // so the dominant eigenvalue is the one with the largest modulus.
        let shift = n as f64;
        let mut v = vec![1.0 / n as f64; n];
        let mut lambda = 0.0;
        for _ in 0..20_000 {
            let mut next = vec![0.0f64; n];
            for u in 0..n {
                next[u] += shift * v[u];
                for (w, _) in &g.adj[u] {
                    next[u] += v[*w];
                }
            }
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm <= 0.0 || norm.is_nan() {
                break;
            }
            v = next.iter().map(|x| x / norm).collect();
            lambda = norm;
        }
        let radius = lambda - shift;
        assert!(
            close(1.0 / threshold, radius, 1e-5 * radius.max(1.0)),
            "the threshold implies a radius of {} against {radius}",
            1.0 / threshold
        );
        // The threshold never exceeds the reciprocal of the maximum degree,
        // since the spectral radius is at least that.
        let max_degree = (0..n).map(|u| g.adj[u].len()).max().unwrap_or(0) as f64;
        if max_degree > 0.0 {
            assert!(
                threshold <= 1.0 / max_degree.sqrt() + 1e-9,
                "the threshold {threshold} exceeds the max-degree bound"
            );
        }
    }
}

#[test]
fn prop_a_network_epidemic_stays_inside_its_component() {
    // Whatever the transmission rate. The well-mixed model has no way to
    // express this, and it is the sharpest thing structure buys.
    let mut rng = Rng::new(0x0B10_9011);
    for trial in 0..15 {
        let per_part = 5 + trial % 4;
        let parts = 2 + trial % 3;
        let n = per_part * parts;
        let mut g = Graph::new(n, false);
        for part in 0..parts {
            let base = part * per_part;
            for u in base..(base + per_part) {
                for v in (u + 1)..(base + per_part) {
                    g.add_edge(u, v, 1.0);
                }
            }
        }
        let start = pick(&mut rng, n);
        let trace = network_sir(&g, 500.0, 0.01, start, &mut rng).unwrap();
        let (s, i, r) = *trace.last().unwrap();
        assert_eq!(i, 0, "the epidemic did not finish");
        assert_eq!(s + i + r, n, "the network model lost a node");
        assert!(
            r <= per_part,
            "the epidemic reached {r} of {n} nodes, beyond its component of {per_part}"
        );
        // Every step keeps the counts consistent, and the removed never
        // decreases.
        for pair in trace.windows(2) {
            assert_eq!(pair[1].0 + pair[1].1 + pair[1].2, n);
            assert!(pair[1].2 >= pair[0].2, "the removed count fell");
            assert!(pair[1].0 <= pair[0].0, "the susceptible count rose");
        }
    }
}

#[test]
fn prop_the_age_structured_model_conserves_every_group() {
    let mut rng = Rng::new(0x0B10_9012);
    for trial in 0..15 {
        let groups = 2 + trial % 3;
        let raw: Vec<f64> = (0..groups).map(|_| 0.2 + rng.next_f64()).collect();
        let total: f64 = raw.iter().sum();
        let sizes: Vec<f64> = raw.iter().map(|x| x / total).collect();
        let contact: Vec<Vec<f64>> = (0..groups)
            .map(|_| (0..groups).map(|_| rng.next_f64() * 2.0).collect())
            .collect();
        let gamma = 0.1 + rng.next_f64();
        let i0: Vec<f64> = sizes.iter().map(|s| s * 1e-5).collect();
        let trace = age_structured(&contact, &sizes, gamma, &i0, 60.0).unwrap();
        for (t, s, i, r) in &trace {
            for a in 0..groups {
                assert!(
                    close(s[a] + i[a] + r[a], sizes[a], 1e-6),
                    "group {a} at t = {t} sums to {} against {}",
                    s[a] + i[a] + r[a],
                    sizes[a]
                );
                assert!(s[a] >= -1e-9 && i[a] >= -1e-9 && r[a] >= -1e-9);
            }
        }
        // R0 is a non-negative eigenvalue, and it scales inversely with the
        // recovery rate -- doubling gamma halves it, exactly.
        let r0 = r0_age_structured(&contact, &sizes, gamma).unwrap();
        assert!(r0 >= 0.0 && r0.is_finite());
        let halved = r0_age_structured(&contact, &sizes, 2.0 * gamma).unwrap();
        assert!(close(halved * 2.0, r0, 1e-6 * r0.max(1e-12)));
        // And it scales linearly with the contact matrix.
        let doubled: Vec<Vec<f64>> =
            contact.iter().map(|row| row.iter().map(|c| c * 3.0).collect()).collect();
        assert!(close(
            r0_age_structured(&doubled, &sizes, gamma).unwrap(),
            3.0 * r0,
            1e-6 * r0.max(1e-12)
        ));
    }
}

// ---------------------------------------------------------------------------
// Stochastic
// ---------------------------------------------------------------------------

#[test]
fn prop_the_stochastic_epidemic_is_a_consistent_jump_process() {
    let mut rng = Rng::new(0x0B10_9020);
    for trial in 0..25 {
        let n = 200 + (trial as u64) * 137;
        let gamma = 0.1 + rng.next_f64();
        let r0 = 0.3 + rng.next_f64() * 4.0;
        let i0 = 1 + (pick(&mut rng, 5) as u64);
        let trace = sir_stochastic_gillespie(r0 * gamma, gamma, n, i0, 500.0, &mut rng).unwrap();
        assert!(!trace.is_empty());
        assert_eq!(trace[0], (0.0, n - i0, i0, 0));
        for (t, s, i, r) in &trace {
            assert_eq!(s + i + r, n, "an individual was lost at t = {t}");
            assert!(t.is_finite() && *t >= 0.0);
        }
        for pair in trace.windows(2) {
            assert!(pair[1].0 >= pair[0].0, "time went backwards");
            assert!(pair[1].1 <= pair[0].1, "susceptibles increased");
            assert!(pair[1].3 >= pair[0].3, "removed decreased");
            // Exactly one event per step: either an infection or a recovery.
            let infected = pair[0].1 - pair[1].1;
            let removed = pair[1].3 - pair[0].3;
            assert!(
                (infected == 1 && removed == 0) || (infected == 0 && removed == 1),
                "a step moved {infected} infections and {removed} recoveries"
            );
        }
        // It ends either at the horizon or with no infectives left.
        let (last_t, _, last_i, _) = *trace.last().unwrap();
        assert!(last_i == 0 || last_t <= 500.0);
    }
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

/// A renewal process with a chosen reproduction number and serial interval.
fn renewal(r: f64, weights: &[f64], days: usize) -> Vec<f64> {
    let mut incidence = vec![0.0; days];
    incidence[0] = 100.0;
    for day in 1..days {
        let force: f64 = weights
            .iter()
            .enumerate()
            .filter(|(s, _)| day > *s)
            .map(|(s, w)| w * incidence[day - s - 1])
            .sum();
        incidence[day] = r * force;
    }
    incidence
}

#[test]
fn prop_the_effective_r_estimate_inverts_the_renewal_process() {
    // Generated from a known reproduction number and serial interval, so the
    // answer is known by construction. Both are randomised, because the
    // whole point of the method is that it separates them -- a growth rate
    // alone cannot.
    let mut rng = Rng::new(0x0B10_9030);
    for _ in 0..40 {
        let length = 2 + pick(&mut rng, 6);
        let raw: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();
        let mass: f64 = raw.iter().sum();
        if mass <= 0.0 || mass.is_nan() {
            continue;
        }
        let weights: Vec<f64> = raw.iter().map(|x| x / mass).collect();
        let r = 0.3 + rng.next_f64() * 2.5;
        let days = 120;
        let incidence = renewal(r, &weights, days);
        if !incidence.iter().all(|c| c.is_finite() && *c >= 0.0) {
            continue;
        }
        let estimate = effective_r_estimate(&incidence, &weights, 7).unwrap();
        for day in (days - 30)..days {
            assert!(
                close(estimate[day], r, 0.02 * r),
                "at R = {r} day {day} the estimate is {}",
                estimate[day]
            );
        }
        assert!(estimate[..7].iter().all(|v| v.is_nan()), "an estimate appeared before the window");
        // Unnormalised weights give the same answer: the method normalises.
        let scaled: Vec<f64> = weights.iter().map(|w| w * 7.3).collect();
        let again = effective_r_estimate(&incidence, &scaled, 7).unwrap();
        assert!(close(again[days - 1], estimate[days - 1], 1e-9 * r));
    }
}

#[test]
fn prop_wallinga_teunis_recovers_the_same_number_away_from_the_record_ends() {
    let mut rng = Rng::new(0x0B10_9031);
    for _ in 0..20 {
        let weights = vec![0.25f64, 0.35, 0.25, 0.15];
        let r = 0.5 + rng.next_f64() * 2.0;
        let days = 100;
        let incidence = renewal(r, &weights, days);
        if !incidence.iter().all(|c| c.is_finite()) {
            continue;
        }
        let wt = wallinga_teunis(&incidence, &weights).unwrap();
        for day in 30..60 {
            assert!(
                close(wt[day], r, 0.05 * r),
                "at R = {r} day {day} Wallinga-Teunis gives {}",
                wt[day]
            );
        }
        // The end effect is structural: the last day has no future in the
        // record to have infected anyone in.
        assert!(wt[days - 1] < 0.5 * r, "no end effect: {} at the last day", wt[days - 1]);
        assert!(wt.iter().take(days - 5).all(|v| v.is_finite() && *v >= 0.0));
    }
}

#[test]
fn prop_the_serial_interval_fit_reproduces_the_sample_moments_exactly() {
    // Method of moments: the fitted gamma must have the sample's own mean
    // and variance, which is an identity rather than an approximation and
    // holds on any positive sample at all.
    let mut rng = Rng::new(0x0B10_9032);
    for _ in 0..100 {
        let count = 3 + pick(&mut rng, 40);
        let scale = 0.1 + rng.next_f64() * 10.0;
        let sample: Vec<f64> = (0..count).map(|_| 0.01 + rng.next_f64() * scale).collect();
        let Ok((shape, fitted_scale)) = serial_interval_fit(&sample) else {
            continue;
        };
        assert!(shape > 0.0 && fitted_scale > 0.0);
        let n = sample.len() as f64;
        let mean: f64 = sample.iter().sum::<f64>() / n;
        let variance: f64 =
            sample.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        assert!(
            close(shape * fitted_scale, mean, 1e-9 * mean),
            "the fitted mean is {} against {mean}",
            shape * fitted_scale
        );
        assert!(
            close(shape * fitted_scale * fitted_scale, variance, 1e-9 * variance),
            "the fitted variance is {} against {variance}",
            shape * fitted_scale * fitted_scale
        );
        // Scaling every observation scales the scale and leaves the shape.
        let stretched: Vec<f64> = sample.iter().map(|x| x * 3.0).collect();
        let (again_shape, again_scale) = serial_interval_fit(&stretched).unwrap();
        assert!(close(again_shape, shape, 1e-8 * shape));
        assert!(close(again_scale, 3.0 * fitted_scale, 1e-8 * fitted_scale));
    }
}
