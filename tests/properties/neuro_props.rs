//! Properties of the computational neuroscience module.
//!
//! Two things make these checkable rather than merely plausible. Several
//! of the models have an exact answer to compare against: the leaky
//! integrator's firing rate, the drift-diffusion process's accuracy, the
//! passive cable's hyperbolic cosine, and the von Mises fit's own
//! generating parameters. And the conductance-based models have
//! invariants that no trajectory may violate whatever the stimulus -- a
//! gating variable is a probability, an activity is a fraction, a spike
//! train is ordered -- which is what a randomised sweep over stimuli can
//! actually test.

use rust_physics_engine::biophysics::neuro::{
    adex, alpha_synapse, cable_equation_1d, cv_isi, ddm_analytic_accuracy, fano_factor,
    fitzhugh_nagumo_neuron, hh_steady_state, hodgkin_huxley, hopfield_energy, hopfield_recall,
    hopfield_store, interspike_intervals, izhikevich, izhikevich_network, izhikevich_presets,
    length_constant, lif_fi_exact, lif_neuron, morris_lecar, poisson_spike_train, psth,
    raster_data, reaction_time_ddm, spike_times, spike_triggered_average, stdp_train, stdp_window,
    synapse_exp, tuning_curve_fit_von_mises, wilson_cowan, MorrisLecar, HH_E_K, HH_E_NA,
    HH_V_REST,
};
use rust_physics_engine::monte_carlo::Rng;

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

#[test]
fn prop_the_gating_variables_stay_probabilities_under_any_stimulus() {
    // m, h and n are fractions of channels open. No stimulus -- steady,
    // pulsed, oscillating or reversed -- may take one outside [0, 1].
    //
    // The voltage is a different matter. Between the reversal potentials
    // is where the *ionic* currents can put it; an injected current is
    // outside that accounting and a hyperpolarising one drives the
    // membrane below E_K without anything being wrong. So the reversal
    // bound is asserted only where it applies, for a drive that never goes
    // negative.
    let mut rng = Rng::new(0x0E0E_4001);
    for trial in 0..12 {
        let amplitude = 40.0 * rng.next_f64();
        let depolarising = trial % 2 == 0;
        let offset = if depolarising { amplitude } else { -20.0 + amplitude };
        let frequency = 0.02 + 0.3 * rng.next_f64();
        let stimulus = move |t: f64| offset + amplitude * (frequency * t).sin();
        let trace = hodgkin_huxley(&stimulus, 100.0, 0.01).unwrap();
        for row in &trace {
            for gate in [row.2, row.3, row.4] {
                assert!((0.0..=1.0).contains(&gate), "a gate reached {gate}");
            }
            assert!(row.1.is_finite() && (-200.0..200.0).contains(&row.1));
            if depolarising {
                assert!(
                    row.1 > HH_E_K - 1e-9,
                    "a depolarising drive still pushed the voltage to {}",
                    row.1
                );
                assert!(row.1 < HH_E_NA + 60.0, "the voltage reached {}", row.1);
            }
        }
        let voltage = trace.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>();
        let times = spike_times(&voltage, 0.0);
        assert!(times.windows(2).all(|p| p[1] > p[0]), "spike times are not ordered");
    }
}

#[test]
fn prop_a_hyperpolarisation_the_step_cannot_follow_is_reported_not_returned() {
    // beta_m grows exponentially as the membrane hyperpolarises, so below
    // about -25 uA/cm^2 a fixed step of 0.01 ms is no longer stable. The
    // failure mode that matters is a trace of plausible-looking numbers
    // that are wrong; what comes back instead is an error.
    for current in [-30.0f64, -60.0, -200.0] {
        assert!(
            hodgkin_huxley(&move |_| current, 100.0, 0.01).is_err(),
            "{current} uA/cm^2 returned a trace"
        );
    }
    // Depolarising currents of any size integrate cleanly and simply stop
    // firing -- depolarisation block, not a numerical failure.
    for current in [40.0f64, 200.0, 500.0] {
        let trace = hodgkin_huxley(&move |_| current, 100.0, 0.01).unwrap();
        assert!(trace.iter().all(|r| r.1.is_finite()));
        assert!(trace.iter().all(|r| (0.0..=1.0).contains(&r.2)));
    }
    // Mild hyperpolarisation is fine, and drives the voltage below E_K.
    let cooled = hodgkin_huxley(&|_| -15.0, 100.0, 0.01).unwrap();
    assert!(cooled.last().unwrap().1 < HH_E_K);
    assert!(cooled.last().unwrap().1 < HH_V_REST);
}

#[test]
fn prop_the_steady_state_gates_are_a_fixed_point_of_the_dynamics() {
    // Started at their steady state with no current, the gates should not
    // move. Away from the resting potential they should move toward it.
    let (m, h, n) = hh_steady_state(HH_V_REST);
    for gate in [m, h, n] {
        assert!((0.0..=1.0).contains(&gate));
    }
    let trace = hodgkin_huxley(&|_| 0.0, 30.0, 0.01).unwrap();
    let last = trace.last().unwrap();
    assert!((last.2 - m).abs() < 1e-3);
    assert!((last.3 - h).abs() < 1e-3);
    assert!((last.4 - n).abs() < 1e-3);
    // Activation rises with depolarisation and inactivation falls: the
    // two curves cross, and that overlap is what limits the sodium window
    // current.
    let mut previous = hh_steady_state(-100.0);
    for v in [-80.0, -60.0, -40.0, -20.0, 0.0, 20.0] {
        let now = hh_steady_state(v);
        assert!(now.0 > previous.0, "m did not rise at {v}");
        assert!(now.1 < previous.1, "h did not fall at {v}");
        assert!(now.2 > previous.2, "n did not rise at {v}");
        previous = now;
    }
}

#[test]
fn prop_every_spiking_model_reports_ordered_times_and_bounded_voltages() {
    let mut rng = Rng::new(0x0E0E_4002);
    for _ in 0..10 {
        let current = 2.0 + 30.0 * rng.next_f64();
        let (_, p) = izhikevich_presets()[pick(&mut rng, 5)];
        let run = izhikevich(p[0], p[1], p[2], p[3], current, 300.0, 0.25).unwrap();
        assert!(run.iter().all(|r| r.1 <= 30.0 + 1e-9 && r.1.is_finite()));
        assert!(run.windows(2).all(|w| w[1].0 > w[0].0), "the clock did not advance");
        let times = spike_times(&run, 20.0);
        assert!(times.windows(2).all(|p| p[1] > p[0]));
        assert!(times.iter().all(|t| (0.0..=300.0).contains(t)));
        if times.len() > 2 {
            assert!(cv_isi(&times).unwrap() >= 0.0);
            assert!(interspike_intervals(&times).unwrap().iter().all(|d| *d > 0.0));
        }
    }
    for _ in 0..6 {
        let current = 40.0 + 120.0 * rng.next_f64();
        let run = morris_lecar(&MorrisLecar::hopf(), current, -60.0, 0.0, 500.0, 0.05).unwrap();
        assert!(run.iter().all(|r| r.1.is_finite() && (0.0..=1.0).contains(&r.2)));
    }
}

#[test]
fn prop_a_stronger_current_never_lowers_the_firing_rate() {
    // Monotonicity in the drive is the one thing every model here shares.
    let mut rng = Rng::new(0x0E0E_4003);
    let (_, p) = izhikevich_presets()[0];
    let mut previous = 0usize;
    for step in 0..8 {
        let current = 4.0 + 3.0 * step as f64;
        let run = izhikevich(p[0], p[1], p[2], p[3], current, 400.0, 0.25).unwrap();
        let count = spike_times(&run, 20.0).len();
        assert!(count >= previous, "the rate fell from {previous} to {count} at I={current}");
        previous = count;
    }
    // And the leaky integrator's exact curve is monotone at any parameters.
    for _ in 0..30 {
        let tau = 1.0 + 30.0 * rng.next_f64();
        let refractory = 5.0 * rng.next_f64();
        let mut last = 0.0;
        for current in [1.01f64, 1.1, 1.5, 3.0, 10.0, 1000.0] {
            let rate = lif_fi_exact(current, tau, 1.0, 0.0, refractory).unwrap();
            assert!(rate > last, "the exact curve fell at I={current}");
            if refractory > 0.0 {
                assert!(rate < 1.0 / refractory);
            }
            last = rate;
        }
    }
}

#[test]
fn prop_the_simulated_leaky_integrator_tracks_its_closed_form() {
    // The interval is the time an exponential takes to reach threshold,
    // which has an exact answer; the simulation may differ only by its
    // step size.
    let mut rng = Rng::new(0x0E0E_4004);
    for _ in 0..12 {
        let tau = 5.0 + 15.0 * rng.next_f64();
        let current = 1.1 + 8.0 * rng.next_f64();
        let refractory = 4.0 * rng.next_f64();
        let span = 6000.0;
        let spikes =
            lif_neuron(current, tau, 1.0, 0.0, refractory, 0.0, span, 0.002, &mut rng).unwrap();
        assert!(spikes.windows(2).all(|p| p[1] > p[0]));
        assert!(spikes.iter().all(|t| (0.0..=span).contains(t)));
        let simulated = spikes.len() as f64 / span;
        let exact = lif_fi_exact(current, tau, 1.0, 0.0, refractory).unwrap();
        assert!(
            (simulated - exact).abs() < 0.02 * exact,
            "tau {tau}, I {current}: {simulated} against {exact}"
        );
        // Without noise every interval is the same one.
        let intervals = interspike_intervals(&spikes).unwrap();
        if intervals.len() > 3 {
            assert!(cv_isi(&spikes).unwrap() < 0.02, "a noiseless train was irregular");
        }
    }
}

#[test]
fn prop_a_poisson_train_is_ordered_and_as_irregular_as_it_should_be() {
    let mut rng = Rng::new(0x0E0E_4005);
    for _ in 0..8 {
        let rate = 0.01 + 0.2 * rng.next_f64();
        let span = 50_000.0 / rate.max(0.02);
        let train = poisson_spike_train(rate, span, &mut rng).unwrap();
        assert!(train.windows(2).all(|p| p[1] > p[0]));
        assert!(train.iter().all(|t| (0.0..span).contains(t)));
        let observed = train.len() as f64 / span;
        assert!((observed - rate).abs() < 0.06 * rate, "rate {observed} against {rate}");
        let cv = cv_isi(&train).unwrap();
        assert!((cv - 1.0).abs() < 0.08, "the coefficient of variation was {cv}");
        // Rescaling time cannot change an irregularity measure.
        let stretched: Vec<f64> = train.iter().map(|t| 7.0 * t).collect();
        assert!((cv_isi(&stretched).unwrap() - cv).abs() < 1e-9);
    }
}

#[test]
fn prop_a_histogram_integrates_back_to_the_spikes_it_binned() {
    // Whatever the bin width, the rate curve times the width sums to the
    // mean spike count per trial. A PSTH that forgot a divisor would not
    // survive being asked at three widths.
    let mut rng = Rng::new(0x0E0E_4006);
    for _ in 0..10 {
        let span = 100.0 + 400.0 * rng.next_f64();
        let rate = 0.02 + 0.2 * rng.next_f64();
        let trials: Vec<Vec<f64>> =
            (0..60).map(|_| poisson_spike_train(rate, span, &mut rng).unwrap()).collect();
        let counted = trials.iter().map(Vec::len).sum::<usize>() as f64 / trials.len() as f64;
        for divisions in [4usize, 17, 53] {
            let bin = span / divisions as f64;
            let histogram = psth(&trials, bin, span).unwrap();
            assert!(histogram.iter().all(|r| *r >= 0.0));
            let integral: f64 = histogram.iter().map(|r| r * bin).sum();
            assert!((integral - counted).abs() < 1e-9, "{integral} against {counted}");
        }
        // The raster holds every spike exactly once.
        let raster = raster_data(&trials);
        assert_eq!(raster.len(), trials.iter().map(Vec::len).sum::<usize>());
        assert!(raster.windows(2).all(|p| p[1].0 >= p[0].0));
    }
}

#[test]
fn prop_the_spike_triggered_average_lies_between_the_stimulus_extremes() {
    // It is a mean of stimulus samples, so it cannot leave their range --
    // and with the spikes independent of the stimulus it sits near the
    // stimulus mean rather than anywhere interesting.
    let mut rng = Rng::new(0x0E0E_4007);
    for _ in 0..15 {
        let length = 3000 + pick(&mut rng, 3000);
        let stimulus: Vec<f64> = (0..length).map(|_| rng.next_gaussian()).collect();
        let low = stimulus.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        let high = stimulus.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let window = 3 + pick(&mut rng, 12);
        let spikes: Vec<f64> =
            (0..600).map(|_| (window + pick(&mut rng, length - window)) as f64).collect();
        let mut sorted = spikes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let average = spike_triggered_average(&stimulus, 1.0, &sorted, window).unwrap();
        assert_eq!(average.len(), window);
        for value in &average {
            assert!((low..=high).contains(value), "the average left the stimulus range");
            assert!(value.abs() < 0.3, "unrelated spikes gave a feature of {value}");
        }
    }
}

#[test]
fn prop_the_von_mises_fit_recovers_the_curve_it_was_given() {
    // Noiseless data with a positive rate everywhere: the linearised fit
    // is exact, whatever the parameters or the sampling of the circle.
    let mut rng = Rng::new(0x0E0E_4008);
    for _ in 0..40 {
        let preferred = -std::f64::consts::PI + std::f64::consts::TAU * rng.next_f64();
        let kappa = 0.05 + 6.0 * rng.next_f64();
        let amplitude = 0.1 + 20.0 * rng.next_f64();
        let count = 5 + pick(&mut rng, 20);
        let angles: Vec<f64> = (0..count)
            .map(|k| -std::f64::consts::PI + k as f64 * std::f64::consts::TAU / count as f64)
            .collect();
        let rates: Vec<f64> =
            angles.iter().map(|a| amplitude * (kappa * (a - preferred).cos()).exp()).collect();
        let (mu, k, amp) = tuning_curve_fit_von_mises(&angles, &rates).unwrap();
        let offset = (mu - preferred).sin().atan2((mu - preferred).cos()).abs();
        assert!(offset < 1e-7, "preferred {mu} against {preferred}");
        assert!((k - kappa).abs() < 1e-7 * kappa.max(1.0));
        assert!((amp - amplitude).abs() < 1e-7 * amplitude);
        assert!(mu > -std::f64::consts::PI - 1e-12 && mu <= std::f64::consts::PI + 1e-12);
    }
}

#[test]
fn prop_synaptic_conductances_are_positive_bounded_and_additive() {
    let mut rng = Rng::new(0x0E0E_4009);
    for _ in 0..25 {
        let tau = 0.5 + 20.0 * rng.next_f64();
        let g_max = 0.05 + 2.0 * rng.next_f64();
        let mut train: Vec<f64> = (0..8).map(|_| 50.0 * rng.next_f64()).collect();
        train.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for step in 0..40 {
            let t = step as f64 * 2.0;
            let exponential = synapse_exp(g_max, tau, &train, t).unwrap();
            let alpha = alpha_synapse(g_max, tau, &train, t).unwrap();
            assert!(exponential >= 0.0 && alpha >= 0.0);
            // One spike can never exceed g_max in either form.
            assert!(alpha_synapse(g_max, tau, &[0.0], t).unwrap() <= g_max + 1e-12);
            assert!(synapse_exp(g_max, tau, &[0.0], t).unwrap() <= g_max + 1e-12);
            // And the train is the sum of its spikes.
            let apart: f64 = train.iter().map(|s| synapse_exp(g_max, tau, &[*s], t).unwrap()).sum();
            assert!((exponential - apart).abs() < 1e-9);
            let alpha_apart: f64 =
                train.iter().map(|s| alpha_synapse(g_max, tau, &[*s], t).unwrap()).sum();
            assert!((alpha - alpha_apart).abs() < 1e-9);
        }
    }
}

#[test]
fn prop_the_plasticity_window_keeps_its_sign_and_its_bound() {
    // Potentiation on one side, depression on the other, and neither ever
    // larger than its amplitude.
    let mut rng = Rng::new(0x0E0E_400A);
    for _ in 0..30 {
        let a_plus = 0.001 + 0.05 * rng.next_f64();
        let a_minus = 0.001 + 0.05 * rng.next_f64();
        let tau_plus = 1.0 + 40.0 * rng.next_f64();
        let tau_minus = 1.0 + 40.0 * rng.next_f64();
        for step in 1..60 {
            let delta = step as f64 * 2.0;
            let up = stdp_window(delta, a_plus, a_minus, tau_plus, tau_minus).unwrap();
            let down = stdp_window(-delta, a_plus, a_minus, tau_plus, tau_minus).unwrap();
            assert!(up > 0.0 && up <= a_plus);
            assert!(down < 0.0 && down >= -a_minus);
        }
        assert_eq!(stdp_window(0.0, a_plus, a_minus, tau_plus, tau_minus).unwrap(), 0.0);
        // A train's change is bounded by the worst case over its pairs.
        let pre: Vec<f64> = (0..10).map(|k| k as f64 * 12.0).collect();
        let post: Vec<f64> = pre.iter().map(|t| t + 4.0).collect();
        let total = stdp_train(&pre, &post, a_plus, a_minus, tau_plus, tau_minus).unwrap();
        assert!(total.abs() <= 100.0 * a_plus.max(a_minus) + 1e-12);
    }
}

#[test]
fn prop_hopfield_recall_descends_the_energy_to_a_fixed_point() {
    // The Lyapunov property, which is what sequential updates buy and
    // simultaneous ones do not: every sweep lowers the energy, and what it
    // reaches does not move again.
    let mut rng = Rng::new(0x0E0E_400B);
    for _ in 0..12 {
        let n = 30 + pick(&mut rng, 50);
        let stored = 1 + pick(&mut rng, 6);
        let patterns: Vec<Vec<i8>> = (0..stored)
            .map(|_| (0..n).map(|_| if rng.next_f64() < 0.5 { -1i8 } else { 1 }).collect())
            .collect();
        let w = hopfield_store(&patterns).unwrap();
        for i in 0..n {
            assert!(w.get(i, i).abs() < 1e-15);
            for j in 0..n {
                assert!((w.get(i, j) - w.get(j, i)).abs() < 1e-15);
            }
        }
        for _ in 0..5 {
            let probe: Vec<i8> =
                (0..n).map(|_| if rng.next_f64() < 0.5 { -1i8 } else { 1 }).collect();
            let mut energy = hopfield_energy(&w, &probe).unwrap();
            let mut state = probe;
            for _ in 0..25 {
                let next = hopfield_recall(&w, &state, 1).unwrap();
                let now = hopfield_energy(&w, &next).unwrap();
                assert!(now <= energy + 1e-9, "a sweep raised the energy from {energy} to {now}");
                energy = now;
                state = next;
            }
            // Settled: another sweep changes nothing.
            assert_eq!(hopfield_recall(&w, &state, 1).unwrap(), state);
            // And the mirror state has the same energy, always.
            let mirrored: Vec<i8> = state.iter().map(|s| -s).collect();
            assert!((hopfield_energy(&w, &mirrored).unwrap() - energy).abs() < 1e-9);
        }
    }
}

#[test]
fn prop_wilson_cowan_activities_never_leave_the_unit_interval() {
    // They are fractions of a population, so the logistic response has to
    // keep them in range for every set of couplings, not just tame ones.
    let mut rng = Rng::new(0x0E0E_400C);
    for _ in 0..25 {
        let draw = |rng: &mut Rng| -20.0 + 40.0 * rng.next_f64();
        let run = wilson_cowan(
            draw(&mut rng),
            draw(&mut rng),
            draw(&mut rng),
            draw(&mut rng),
            draw(&mut rng) * 0.2,
            draw(&mut rng) * 0.2,
            0.5 + 2.0 * rng.next_f64(),
            0.5 + 2.0 * rng.next_f64(),
            0.5 + 2.0 * rng.next_f64(),
            -2.0 + 8.0 * rng.next_f64(),
            rng.next_f64(),
            rng.next_f64(),
            80.0,
            0.005,
        )
        .unwrap();
        for row in &run {
            assert!((0.0..=1.0).contains(&row.1), "E reached {}", row.1);
            assert!((0.0..=1.0).contains(&row.2), "I reached {}", row.2);
        }
    }
}

#[test]
fn prop_the_cable_falls_off_monotonically_and_matches_the_closed_form() {
    // The discretisation is checked against the analytic sealed-end
    // solution at whatever length, length constant and resolution.
    let mut rng = Rng::new(0x0E0E_400D);
    for _ in 0..25 {
        let lambda = 0.05 + 2.0 * rng.next_f64();
        let length = lambda * (0.3 + 5.0 * rng.next_f64());
        let injected = 0.1 + 5.0 * rng.next_f64();
        let points = 201 + 2 * pick(&mut rng, 400);
        let v = cable_equation_1d(length, lambda, injected, points).unwrap();
        assert_eq!(v.len(), points);
        assert!((v[0] - injected).abs() < 1e-12);
        assert!(v.windows(2).all(|p| p[1] <= p[0] + 1e-9), "the profile rose along the cable");
        assert!(v.iter().all(|x| *x > 0.0 && x.is_finite()));
        let scale = (length / lambda).cosh();
        for (index, value) in v.iter().enumerate() {
            let x = length * index as f64 / (points - 1) as f64;
            let analytic = injected * ((length - x) / lambda).cosh() / scale;
            assert!(
                (value - analytic).abs() < 1e-3 * injected,
                "at x={x} the solution gave {value} not {analytic}"
            );
        }
    }
}

#[test]
fn prop_the_length_constant_scales_as_the_square_root_it_is_built_from() {
    let mut rng = Rng::new(0x0E0E_400E);
    for _ in 0..40 {
        let r_m = 100.0 + 100_000.0 * rng.next_f64();
        let r_i = 10.0 + 500.0 * rng.next_f64();
        let diameter = 1e-5 + 1e-2 * rng.next_f64();
        let base = length_constant(r_m, r_i, diameter).unwrap();
        assert!(base > 0.0 && base.is_finite());
        let factor = 1.5 + 8.0 * rng.next_f64();
        assert!(
            (length_constant(r_m * factor, r_i, diameter).unwrap() / base - factor.sqrt()).abs()
                < 1e-9 * factor
        );
        assert!(
            (length_constant(r_m, r_i, diameter * factor).unwrap() / base - factor.sqrt()).abs()
                < 1e-9 * factor
        );
        assert!(
            (length_constant(r_m, r_i * factor, diameter).unwrap() / base
                - 1.0 / factor.sqrt())
            .abs()
                < 1e-9
        );
    }
}

#[test]
fn prop_simulated_accuracy_matches_the_gamblers_ruin_formula() {
    let mut rng = Rng::new(0x0E0E_400F);
    for _ in 0..8 {
        let drift = -1.5 + 3.0 * rng.next_f64();
        let threshold = 0.5 + 1.0 * rng.next_f64();
        let noise = 0.6 + 1.0 * rng.next_f64();
        let trials = 3000;
        let runs = reaction_time_ddm(drift, threshold, noise, 0.001, trials, &mut rng).unwrap();
        assert_eq!(runs.len(), trials);
        assert!(runs.iter().all(|r| r.0 > 0.0 && r.0.is_finite()));
        let observed = runs.iter().filter(|r| r.1).count() as f64 / trials as f64;
        let exact = ddm_analytic_accuracy(drift, threshold, noise).unwrap();
        let error = (exact * (1.0 - exact) / trials as f64).sqrt();
        assert!(
            (observed - exact).abs() < 4.0 * error + 0.02,
            "drift {drift}, bound {threshold}, noise {noise}: {observed} against {exact}"
        );
        // The formula itself is a probability and reflects with the drift.
        assert!((0.0..=1.0).contains(&exact));
        assert!(
            (exact + ddm_analytic_accuracy(-drift, threshold, noise).unwrap() - 1.0).abs() < 1e-12
        );
    }
}

#[test]
fn prop_the_network_reports_spikes_inside_its_own_bounds() {
    let mut rng = Rng::new(0x0E0E_4010);
    for _ in 0..6 {
        let excitatory = 20 + pick(&mut rng, 60);
        let inhibitory = 5 + pick(&mut rng, 20);
        let span = 100.0 + 200.0 * rng.next_f64();
        let spikes = izhikevich_network(excitatory, inhibitory, span, &mut rng).unwrap();
        let neurons = excitatory + inhibitory;
        assert!(spikes.iter().all(|s| s.1 < neurons));
        assert!(spikes.iter().all(|s| (0.0..span).contains(&s.0)));
        assert!(spikes.windows(2).all(|p| p[1].0 >= p[0].0), "the spikes are out of order");
        // A neuron cannot spike twice in the same millisecond.
        for pair in spikes.windows(2) {
            assert!(pair[0] != pair[1], "a neuron fired twice at the same instant");
        }
        let rate = 1000.0 * spikes.len() as f64 / (neurons as f64 * span);
        assert!(rate < 500.0, "the network ran away to {rate} Hz per neuron");
    }
}

#[test]
fn prop_adaptation_can_only_slow_a_train_down() {
    // Adding either adaptation current to an AdEx neuron subtracts from
    // its drive, so it can never fire more than the unadapting one.
    let mut rng = Rng::new(0x0E0E_4011);
    for _ in 0..8 {
        let current = 400.0 + 400.0 * rng.next_f64();
        let count = |a: f64, b: f64| {
            let run =
                adex(200.0, 10.0, -70.0, 2.0, -50.0, 100.0, a, b, -58.0, current, 400.0, 0.05)
                    .unwrap();
            let trace: Vec<(f64, f64)> = run.iter().map(|r| (r.0, r.1)).collect();
            spike_times(&trace, -32.0).len()
        };
        let plain = count(0.0, 0.0);
        assert!(plain > 5, "the unadapting neuron only fired {plain} times");
        for (a, b) in [(0.0, 30.0), (2.0, 0.0), (2.0, 30.0)] {
            let adapted = count(a, b);
            assert!(adapted <= plain, "adaptation raised the count from {plain} to {adapted}");
        }
    }
}

#[test]
fn prop_fitzhugh_nagumo_settles_or_cycles_but_never_diverges() {
    let mut rng = Rng::new(0x0E0E_4012);
    for _ in 0..25 {
        let a = 0.5 + 0.4 * rng.next_f64();
        let b = 0.4 + 0.6 * rng.next_f64();
        let tau = 5.0 + 20.0 * rng.next_f64();
        let current = -1.0 + 2.5 * rng.next_f64();
        let run = fitzhugh_nagumo_neuron(a, b, tau, current, -2.0 + 4.0 * rng.next_f64(),
            -1.0 + 2.0 * rng.next_f64(), 400.0, 0.05)
            .unwrap();
        // The cubic bounds the excursion: the vector field points inward
        // well before |v| = 4, so no parameter set here can run away.
        for row in &run {
            assert!(row.1.abs() < 6.0, "v reached {}", row.1);
            assert!(row.2.abs() < 6.0, "w reached {}", row.2);
        }
        let tail: Vec<f64> = run.iter().filter(|r| r.0 > 300.0).map(|r| r.1).collect();
        let swing = tail.iter().fold(f64::NEG_INFINITY, |x, y| x.max(*y))
            - tail.iter().fold(f64::INFINITY, |x, y| x.min(*y));
        assert!(swing.is_finite());
    }
}

#[test]
fn prop_the_count_statistics_reject_what_they_cannot_measure() {
    let mut rng = Rng::new(0x0E0E_4013);
    for _ in 0..20 {
        let n = 3 + pick(&mut rng, 20);
        let counts: Vec<u64> = (0..n).map(|_| pick(&mut rng, 20) as u64).collect();
        match fano_factor(&counts) {
            Ok(f) => {
                assert!(f >= 0.0 && f.is_finite());
                assert!(counts.iter().any(|c| *c > 0));
            }
            Err(_) => assert!(counts.iter().all(|c| *c == 0)),
        }
        let mut train: Vec<f64> = (0..n).map(|_| 100.0 * rng.next_f64()).collect();
        train.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let intervals = interspike_intervals(&train).unwrap();
        assert_eq!(intervals.len(), n - 1);
        assert!(intervals.iter().all(|d| *d >= 0.0));
        // Reversing the train is not a train.
        if n > 2 && train[0] < train[n - 1] {
            let mut reversed = train.clone();
            reversed.reverse();
            assert!(interspike_intervals(&reversed).is_err());
        }
    }
}

