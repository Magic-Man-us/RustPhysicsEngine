//! Properties of the molecular dynamics module.
//!
//! Molecular dynamics is unusually well supplied with exact statements that
//! hold configuration by configuration rather than on average, and they are
//! the ones worth checking on random instances: the total force is the
//! gradient of the total energy, the internal forces cancel, the equations
//! of motion are invariant under translation and under relabelling, and --
//! the strongest of them -- the integrator is exactly reversible, so
//! running a trajectory backwards returns it to where it began. None of
//! these depend on a thermostat having converged or a run being long
//! enough.

use rust_physics_engine::math::Vec3;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::statistical_mechanics::md::{
    collision_rate, energy_drift, ewald_sum_energy_lite, green_kubo_viscosity_lite,
    jarzynski_free_energy, lj_phase_point, mean_free_path, umbrella_sampling_pmf,
    virial_coefficient_b2, MdSample, MdSystem, Potential,
};
use std::sync::Arc;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn spread(rng: &mut Rng, half_width: f64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * half_width
}

/// A jittered lattice, so the configuration is neither symmetric nor so
/// close-packed that the forces overflow.
fn scattered(rng: &mut Rng, cells: usize, density: f64, jitter: f64) -> MdSystem {
    let mut system = MdSystem::lattice_fcc(cells, density, 1.0, 1.0, 1.0, rng).unwrap();
    for k in 0..system.pos.len() {
        let step = Vec3::new(spread(rng, jitter), spread(rng, jitter), spread(rng, jitter));
        system.pos[k] = system.wrap(system.pos[k] + step);
        system.unwrapped[k] = system.unwrapped[k] + step;
    }
    // Writing to `pos` leaves the integrator's cached forces stale.
    system.refresh_forces();
    system
}

// ---------------------------------------------------------------------------
// Forces
// ---------------------------------------------------------------------------

#[test]
fn prop_the_total_force_is_the_gradient_of_the_total_energy() {
    // Not the pair law against its own derivative -- that is a check on one
    // formula -- but the *system's* force against a finite difference of the
    // *system's* energy. It exercises the pair traversal, the minimum image
    // and the cutoff shift at once, and it is the invariant every conserved
    // quantity in the module rests on.
    let mut rng = Rng::new(0x011D_9001);
    for trial in 0..6 {
        let system = scattered(&mut rng, 2 + trial % 2, 0.5 + 0.1 * (trial % 3) as f64, 0.15);
        let forces = system.forces();
        let h = 1e-6;
        for _ in 0..8 {
            let k = ((u128::from(rng.next_u64()) * system.len() as u128) >> 64) as usize;
            for axis in 0..3 {
                let bump = |v: f64| match axis {
                    0 => Vec3::new(v, 0.0, 0.0),
                    1 => Vec3::new(0.0, v, 0.0),
                    _ => Vec3::new(0.0, 0.0, v),
                };
                let mut up = system.clone();
                up.pos[k] = up.wrap(up.pos[k] + bump(h));
                let mut down = system.clone();
                down.pos[k] = down.wrap(down.pos[k] + bump(-h));
                let numeric =
                    -(up.potential_energy() - down.potential_energy()) / (2.0 * h);
                let analytic = match axis {
                    0 => forces[k].x,
                    1 => forces[k].y,
                    _ => forces[k].z,
                };
                let scale = analytic.abs().max(numeric.abs()).max(1.0);
                assert!(
                    close(analytic, numeric, 2e-3 * scale),
                    "particle {k} axis {axis}: force {analytic} against gradient {numeric}"
                );
            }
        }
    }
}

#[test]
fn prop_the_internal_forces_cancel_on_every_configuration() {
    let mut rng = Rng::new(0x011D_9002);
    for trial in 0..12 {
        let system = scattered(&mut rng, 2 + trial % 3, 0.4 + 0.08 * (trial % 6) as f64, 0.2);
        let forces = system.forces();
        let total = forces.iter().fold(Vec3::new(0.0, 0.0, 0.0), |a, f| a + *f);
        let magnitude: f64 = forces.iter().map(Vec3::magnitude).sum();
        assert!(
            close(total.magnitude(), 0.0, 1e-9 * magnitude.max(1.0)),
            "the net force is {} against a total magnitude of {magnitude}",
            total.magnitude()
        );
    }
}

#[test]
fn prop_translating_the_box_changes_nothing() {
    // Homogeneity of space, and on a periodic box it is exact rather than
    // asymptotic: a rigid shift of every particle is the same configuration.
    // An implementation that measured a displacement from the box origin
    // rather than between particles would fail here and nowhere else.
    let mut rng = Rng::new(0x011D_9003);
    for trial in 0..8 {
        let system = scattered(&mut rng, 2 + trial % 2, 0.6, 0.15);
        let energy = system.potential_energy();
        let forces = system.forces();
        let shift = Vec3::new(spread(&mut rng, 20.0), spread(&mut rng, 20.0), spread(&mut rng, 20.0));
        let mut moved = system.clone();
        for k in 0..moved.len() {
            moved.pos[k] = moved.wrap(moved.pos[k] + shift);
        }
        assert!(
            close(moved.potential_energy(), energy, 1e-8 * energy.abs().max(1.0)),
            "a rigid shift moved the energy from {energy} to {}",
            moved.potential_energy()
        );
        let shifted_forces = moved.forces();
        for k in 0..system.len() {
            assert!(
                close((shifted_forces[k] - forces[k]).magnitude(), 0.0, 1e-8 * forces[k].magnitude().max(1.0)),
                "the force on particle {k} changed under a rigid shift"
            );
        }
    }
}

#[test]
fn prop_relabelling_the_particles_permutes_the_forces() {
    // The particles are indistinguishable, so the answer cannot depend on
    // the order they are stored in -- which is exactly what a cell list,
    // whose traversal order *does* depend on it, could break.
    let mut rng = Rng::new(0x011D_9004);
    for trial in 0..6 {
        let system = scattered(&mut rng, 2 + trial % 2, 0.7, 0.15);
        let forces = system.forces();
        let n = system.len();
        // A random permutation by Fisher-Yates.
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = ((u128::from(rng.next_u64()) * (i + 1) as u128) >> 64) as usize;
            order.swap(i, j);
        }
        let mut shuffled = system.clone();
        for (new, &old) in order.iter().enumerate() {
            shuffled.pos[new] = system.pos[old];
            shuffled.unwrapped[new] = system.unwrapped[old];
            shuffled.vel[new] = system.vel[old];
        }
        assert!(close(
            shuffled.potential_energy(),
            system.potential_energy(),
            1e-9 * system.potential_energy().abs().max(1.0)
        ));
        let shuffled_forces = shuffled.forces();
        for (new, &old) in order.iter().enumerate() {
            assert!(
                close((shuffled_forces[new] - forces[old]).magnitude(), 0.0, 1e-9),
                "relabelling {old} to {new} changed its force"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The integrator
// ---------------------------------------------------------------------------

#[test]
fn prop_velocity_verlet_is_exactly_reversible() {
    // The strongest statement available about this integrator, and the one
    // that distinguishes it from every dissipative scheme: run forward,
    // reverse the velocities, run the same number of steps, and the system
    // is back where it started -- not approximately, but to rounding. The
    // equations of motion are time-symmetric and velocity Verlet respects
    // that exactly, which is the structural reason its energy error stays
    // bounded.
    let mut rng = Rng::new(0x011D_9010);
    for trial in 0..4 {
        let mut system = scattered(&mut rng, 2, 0.5 + 0.1 * trial as f64, 0.1);
        let start_pos = system.pos.clone();
        let start_vel = system.vel.clone();
        let dt = 0.002;
        let steps = 300;
        for _ in 0..steps {
            system.step_velocity_verlet(dt);
        }
        // Having gone somewhere: otherwise the test would pass on a system
        // that never moved.
        let travelled: f64 = (0..system.len())
            .map(|k| system.minimum_image(system.pos[k] - start_pos[k]).magnitude())
            .sum::<f64>()
            / system.len() as f64;
        assert!(travelled > 0.05, "the system barely moved: {travelled}");

        for v in &mut system.vel {
            *v = -*v;
        }
        // The cached force is a function of position alone, so reversing
        // the velocities alone is the whole of time reversal.
        for _ in 0..steps {
            system.step_velocity_verlet(dt);
        }
        for k in 0..system.len() {
            let back = system.minimum_image(system.pos[k] - start_pos[k]).magnitude();
            assert!(
                back < 1e-7,
                "particle {k} came back {back} away from where it started"
            );
            let speed = (system.vel[k] + start_vel[k]).magnitude();
            assert!(speed < 1e-7, "particle {k}'s reversed velocity is off by {speed}");
        }
    }
}

#[test]
fn prop_an_isolated_run_conserves_its_energy_and_its_momentum() {
    let mut rng = Rng::new(0x011D_9011);
    for trial in 0..4 {
        let mut system = scattered(&mut rng, 2, 0.45 + 0.08 * trial as f64, 0.12);
        system.remove_drift();
        let momentum = system.total_momentum();
        let samples = system.run_nve(1_500, 0.002).unwrap();
        assert!(energy_drift(&samples).unwrap() < 1e-4);
        let after = system.total_momentum();
        assert!(close((after - momentum).magnitude(), 0.0, 1e-9));
        // The reported total really is the sum of its parts.
        for s in &samples {
            assert!(close(s.total, s.kinetic + s.potential, 1e-9 * s.total.abs().max(1.0)));
            assert!(s.kinetic >= 0.0);
            assert!(s.temperature >= 0.0);
        }
        // And the time advances by exactly the step.
        for pair in samples.windows(2) {
            assert!(close(pair[1].time - pair[0].time, 0.002, 1e-12));
        }
    }
}

#[test]
fn prop_rescaling_hits_the_temperature_it_is_given() {
    let mut rng = Rng::new(0x011D_9012);
    for trial in 0..10 {
        let mut system = scattered(&mut rng, 2, 0.6, 0.1);
        let target = 0.1 + 0.4 * (trial % 7) as f64;
        system.rescale_to_temperature(target);
        assert!(
            close(system.temperature(), target, 1e-9 * target),
            "rescaling to {target} gave {}",
            system.temperature()
        );
        // Rescaling changes no direction, only magnitudes.
        let before: Vec<Vec3> = system.vel.clone();
        system.rescale_to_temperature(2.0 * target);
        for k in 0..system.len() {
            if before[k].magnitude() > 1e-12 {
                let ratio = system.vel[k].magnitude() / before[k].magnitude();
                assert!(close(ratio, 2f64.sqrt(), 1e-9));
                let cosine = system.vel[k].dot(&before[k])
                    / (system.vel[k].magnitude() * before[k].magnitude());
                assert!(close(cosine, 1.0, 1e-9), "the direction turned");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry and measurement
// ---------------------------------------------------------------------------

#[test]
fn prop_the_minimum_image_is_the_shortest_equivalent_displacement() {
    let mut rng = Rng::new(0x011D_9020);
    let system = MdSystem::new(
        vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
        vec![Vec3::new(0.0, 0.0, 0.0); 2],
        vec![1.0; 2],
        Vec3::new(7.0, 11.0, 13.0),
        true,
        Potential::LennardJones { eps: 1.0, sigma: 1.0 },
        3.0,
    )
    .unwrap();
    let edges = [7.0f64, 11.0, 13.0];
    for _ in 0..2_000 {
        let raw = Vec3::new(spread(&mut rng, 60.0), spread(&mut rng, 60.0), spread(&mut rng, 60.0));
        let folded = system.minimum_image(raw);
        for (c, l) in [(folded.x, edges[0]), (folded.y, edges[1]), (folded.z, edges[2])] {
            assert!(c.abs() <= 0.5 * l + 1e-9, "the component {c} exceeds half of {l}");
        }
        // It differs from the original by whole box lengths, so it is the
        // same point of the torus...
        for (a, b, l) in [
            (raw.x, folded.x, edges[0]),
            (raw.y, folded.y, edges[1]),
            (raw.z, folded.z, edges[2]),
        ] {
            let images = (a - b) / l;
            assert!(close(images, images.round(), 1e-9));
        }
        // ...and it is idempotent, since it is already the shortest.
        let again = system.minimum_image(folded);
        assert!(close((again - folded).magnitude(), 0.0, 1e-12));
    }
}

#[test]
fn prop_the_radial_distribution_counts_the_neighbours_that_are_there() {
    // An identity, not an approximation: integrating 4 pi rho g r^2 out to
    // r_max gives the mean neighbour count within r_max by construction, so
    // it holds on any configuration whatever and catches a normalisation
    // error immediately.
    let mut rng = Rng::new(0x011D_9021);
    for trial in 0..6 {
        let system = scattered(&mut rng, 2 + trial % 2, 0.3 + 0.2 * (trial % 4) as f64, 0.3);
        let r_max = 0.45 * system.box_size.x;
        let bins = 40 + trial * 7;
        let g = system.rdf(bins, r_max).unwrap();
        let width = r_max / bins as f64;
        let density = system.len() as f64 / system.volume();
        let integral: f64 = g
            .iter()
            .enumerate()
            .map(|(k, v)| {
                let lo = k as f64 * width;
                let hi = lo + width;
                v * 4.0 / 3.0 * std::f64::consts::PI * (hi * hi * hi - lo * lo * lo)
            })
            .sum::<f64>()
            * density;
        let mut counted = 0usize;
        for i in 0..system.len() {
            for j in 0..system.len() {
                if i != j && system.minimum_image(system.pos[i] - system.pos[j]).magnitude() < r_max
                {
                    counted += 1;
                }
            }
        }
        let expected = counted as f64 / system.len() as f64;
        assert!(
            close(integral, expected, 1e-9 * expected.max(1.0)),
            "the integral gives {integral} against {expected} counted"
        );
        assert!(g.iter().all(|v| *v >= 0.0), "a negative g(r)");
    }
}

#[test]
fn prop_the_structure_factor_tends_to_one_at_large_wavenumber() {
    // True of every configuration: the phases decorrelate, the Debye sum
    // averages to nothing, and only the self term survives. It is the check
    // on the normalisation, and it needs no reference structure.
    let mut rng = Rng::new(0x011D_9022);
    for trial in 0..5 {
        let system = scattered(&mut rng, 2 + trial % 2, 0.5, 0.3);
        let far: Vec<f64> = (0..12).map(|k| 80.0 + f64::from(k) * 13.0).collect();
        let s = system.structure_factor(&far).unwrap();
        let mean: f64 = s.iter().sum::<f64>() / s.len() as f64;
        assert!(close(mean, 1.0, 0.1), "the large-k mean is {mean}");
    }
}

#[test]
fn prop_the_displacement_measures_agree_with_their_own_definitions() {
    // Built from random walks rather than a simulation, so the identities
    // are exact: the lag-zero displacement is zero, the lag-zero
    // autocorrelation is one, and free flight is ballistic at every lag.
    let mut rng = Rng::new(0x011D_9023);
    for _ in 0..6 {
        let count = 25;
        let frames = 24;
        let dt = 0.05;
        let velocities: Vec<Vec3> = (0..count)
            .map(|_| Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian()))
            .collect();
        let traj: Vec<Vec<Vec3>> = (0..frames)
            .map(|t| velocities.iter().map(|v| *v * (t as f64 * dt)).collect())
            .collect();
        let msd = MdSystem::msd(&traj).unwrap();
        assert_eq!(msd.len(), frames);
        assert!(close(msd[0], 0.0, 1e-15));
        let mean_v2: f64 =
            velocities.iter().map(Vec3::magnitude_squared).sum::<f64>() / count as f64;
        for lag in 1..frames {
            let t = lag as f64 * dt;
            assert!(close(msd[lag], mean_v2 * t * t, 1e-8 * mean_v2 * t * t));
        }
        // And a displacement measure must never decrease with lag for
        // straight-line motion.
        for pair in msd.windows(2) {
            assert!(pair[1] >= pair[0] - 1e-12);
        }
        let vel_traj = vec![velocities.clone(); frames];
        let vacf = MdSystem::vacf(&vel_traj).unwrap();
        assert!(close(vacf[0], 1.0, 1e-12));
        assert!(vacf.iter().all(|c| close(*c, 1.0, 1e-12)));
    }
}

// ---------------------------------------------------------------------------
// Reference quantities
// ---------------------------------------------------------------------------

#[test]
fn prop_the_hard_sphere_virial_is_its_own_closed_form() {
    // B2 = 2 pi d^3 / 3 at every temperature, so the quadrature can be
    // checked rather than trusted -- and across diameters, so a hard-coded
    // constant could not pass.
    let mut rng = Rng::new(0x011D_9030);
    for _ in 0..10 {
        let d = 0.3 + rng.next_f64() * 2.0;
        let hard = Potential::Custom(Arc::new(move |r: f64| if r < d { (1e6, 0.0) } else { (0.0, 0.0) }));
        let expected = 2.0 * std::f64::consts::PI * d * d * d / 3.0;
        for _ in 0..3 {
            let t = 0.2 + rng.next_f64() * 5.0;
            let b2 = virial_coefficient_b2(&hard, t, d * 3.0, 60_000).unwrap();
            assert!(
                close(b2, expected, 2e-3 * expected),
                "a sphere of diameter {d} at T = {t} gives {b2} against {expected}"
            );
        }
    }
}

#[test]
fn prop_the_ewald_energy_is_independent_of_the_splitting_parameter() {
    // Alpha divides the sum between real and reciprocal space and is no
    // part of the physics, so the total must not move with it. This catches
    // a dropped self-energy or a swapped erf and erfc without needing any
    // reference value -- those errors are alpha-dependent by construction.
    let mut rng = Rng::new(0x011D_9031);
    for trial in 0..6 {
        let count = 4 + trial;
        let side = 4.0 + rng.next_f64() * 2.0;
        let pos: Vec<Vec3> = (0..count)
            .map(|_| {
                Vec3::new(
                    rng.next_f64() * side,
                    rng.next_f64() * side,
                    rng.next_f64() * side,
                )
            })
            .collect();
        let mut charges: Vec<f64> = (0..count - 1).map(|_| spread(&mut rng, 1.0)).collect();
        let balance = -charges.iter().sum::<f64>();
        charges.push(balance);
        let reference = ewald_sum_energy_lite(&charges, &pos, side, 6.0 / side, 12).unwrap();
        for step in 0..4 {
            let alpha = (4.0 + f64::from(step)) / side;
            let other = ewald_sum_energy_lite(&charges, &pos, side, alpha, 14).unwrap();
            assert!(
                close(other, reference, 2e-3 * reference.abs().max(1.0)),
                "alpha {alpha} gives {other} against {reference}"
            );
        }
    }
}

#[test]
fn prop_wham_recovers_whatever_profile_it_is_shown() {
    // The histograms are built exactly from a chosen profile and the
    // windows' own biases, so the inversion has no statistical error to
    // hide behind: WHAM must return that profile up to a constant, for any
    // profile at all.
    let mut rng = Rng::new(0x011D_9032);
    for trial in 0..5 {
        let temperature = 0.5 + rng.next_f64();
        let k = 8.0 + rng.next_f64() * 8.0;
        let bins = 50;
        let bin_lo = -2.5;
        let bin_width = 0.1;
        let x = |b: usize| bin_lo + (b as f64 + 0.5) * bin_width;
        // A random quartic, so no two trials invert the same shape.
        let (a, b, c) = (
            0.5 + rng.next_f64() * 2.0,
            spread(&mut rng, 3.0),
            spread(&mut rng, 1.0),
        );
        let truth = move |v: f64| a * v * v * v * v + b * v * v + c * v;
        let centers: Vec<f64> = (0..11).map(|w| -2.0 + 0.4 * f64::from(w)).collect();
        let histograms: Vec<Vec<f64>> = centers
            .iter()
            .map(|centre| {
                let raw: Vec<f64> = (0..bins)
                    .map(|bin| {
                        let v = x(bin);
                        let bias = 0.5 * k * (v - centre) * (v - centre);
                        (-(truth(v) + bias) / temperature).exp()
                    })
                    .collect();
                let total: f64 = raw.iter().sum();
                raw.into_iter().map(|p| p / total * 50_000.0).collect()
            })
            .collect();
        let pmf =
            umbrella_sampling_pmf(&histograms, &centers, k, bin_lo, bin_width, temperature).unwrap();
        let true_curve: Vec<f64> = (0..bins).map(|bin| truth(x(bin))).collect();
        let inside: Vec<usize> = (0..bins).filter(|bin| x(*bin).abs() <= 2.0).collect();
        // Both are defined up to a constant, so compare after removing the
        // mean over the region the windows actually cover.
        let mean_pmf: f64 =
            inside.iter().map(|bin| pmf[*bin]).sum::<f64>() / inside.len() as f64;
        let mean_true: f64 =
            inside.iter().map(|bin| true_curve[*bin]).sum::<f64>() / inside.len() as f64;
        let scale = inside
            .iter()
            .map(|bin| (true_curve[*bin] - mean_true).abs())
            .fold(0.0, f64::max)
            .max(1.0);
        for bin in &inside {
            assert!(
                close(pmf[*bin] - mean_pmf, true_curve[*bin] - mean_true, 0.02 * scale),
                "trial {trial} at x = {}: {} against {}",
                x(*bin),
                pmf[*bin] - mean_pmf,
                true_curve[*bin] - mean_true
            );
        }
        assert!(pmf.iter().filter(|v| v.is_finite()).all(|v| *v >= -1e-9), "a negative PMF");
    }
}

#[test]
fn prop_the_jarzynski_estimate_never_exceeds_the_mean_work() {
    // Jensen's inequality, which in this setting *is* the second law:
    // the exponential average sits at or below the arithmetic one, with
    // equality only when every pull cost the same.
    let mut rng = Rng::new(0x011D_9033);
    for trial in 0..12 {
        let temperature = 0.2 + rng.next_f64() * 2.0;
        let width = 2.0 * (trial % 4) as f64;
        let centre = spread(&mut rng, 5.0);
        let work: Vec<f64> = (0..500).map(|_| centre + width * rng.next_gaussian()).collect();
        let mean: f64 = work.iter().sum::<f64>() / work.len() as f64;
        let estimate = jarzynski_free_energy(&work, temperature).unwrap();
        assert!(estimate <= mean + 1e-9, "the estimate {estimate} exceeds the mean {mean}");
        if width == 0.0 {
            assert!(close(estimate, centre, 1e-9), "identical pulls gave {estimate}");
        } else {
            assert!(estimate < mean, "a spread of {width} produced no gap at all");
        }
        // Shifting every work value shifts the estimate by the same amount:
        // the free energy has an origin, and the estimator must respect it.
        let shifted: Vec<f64> = work.iter().map(|w| w + 3.5).collect();
        assert!(close(
            jarzynski_free_energy(&shifted, temperature).unwrap(),
            estimate + 3.5,
            1e-6 * (1.0 + estimate.abs())
        ));
    }
}

#[test]
fn prop_the_kinetic_theory_relations_are_reciprocal() {
    let mut rng = Rng::new(0x011D_9034);
    for _ in 0..200 {
        let density = 0.01 + rng.next_f64() * 40.0;
        let sigma = 0.01 + rng.next_f64() * 5.0;
        let speed = 0.05 + rng.next_f64() * 10.0;
        let lambda = mean_free_path(density, sigma).unwrap();
        let rate = collision_rate(density, sigma, speed).unwrap();
        // One mean free path per collision, by definition.
        assert!(close(rate * lambda, speed, 1e-9 * speed));
        // And the path is inversely proportional to both its arguments.
        let denser = mean_free_path(2.0 * density, sigma).unwrap();
        assert!(close(denser * 2.0, lambda, 1e-9 * lambda));
        let bigger = mean_free_path(density, 3.0 * sigma).unwrap();
        assert!(close(bigger * 3.0, lambda, 1e-9 * lambda));
    }
}

#[test]
fn prop_energy_drift_is_linear_in_the_trend_and_blind_to_the_offset() {
    // The measure is a fitted slope times the span over the mean, so on a
    // pure trend it has a closed form, and an oscillation of a given size
    // must stay a small correction beside a trend much larger than it.
    let mut rng = Rng::new(0x011D_9035);
    for _ in 0..20 {
        let base = 50.0 + rng.next_f64() * 100.0;
        let slope = rng.next_f64() * 0.5;
        let phase = rng.next_f64() * 6.0;
        // On a pure trend the answer is closed form, so it can be checked
        // exactly rather than compared.
        let clean = |trend: f64| -> Vec<MdSample> {
            (0..300)
                .map(|k| {
                    let t = k as f64 * 0.01;
                    let e = base + trend * t;
                    MdSample {
                        time: t,
                        kinetic: e,
                        potential: 0.0,
                        total: e,
                        temperature: 1.0,
                        pressure: 0.0,
                    }
                })
                .collect()
        };
        let span = 299.0 * 0.01;
        let mean = base + slope * span / 2.0;
        let exact = (slope * span / mean).abs();
        let single = energy_drift(&clean(slope)).unwrap();
        assert!(
            close(single, exact, 1e-9 * exact.max(1e-12)),
            "a pure trend read {single} against the closed form {exact}"
        );
        // Sign does not matter: drift is a magnitude.
        assert!(close(
            energy_drift(&clean(-slope)).unwrap(),
            (slope * span / (base - slope * span / 2.0)).abs(),
            1e-9
        ));

        // A wobble is not a trend. It is not *invisible* to a straight-line
        // fit -- an oscillation that stops part way through a cycle leaves a
        // residual slope of order twice its amplitude over the span, which
        // is a limitation of the measure and not a defect. So the comparison
        // is made where it means something: against a trend whose total rise
        // is twenty times the wobble's amplitude, the wobble must be a small
        // correction. Drawing the amplitude independently of the trend
        // would compare a rise of 0.15 against an amplitude of 2, which
        // would prove nothing either way.
        let rise = 0.5 + rng.next_f64();
        let amplitude = rise / 20.0;
        let ripple = |trend: f64| -> Vec<MdSample> {
            (0..300)
                .map(|k| {
                    let t = k as f64 * 0.01;
                    let e = base + trend * t + amplitude * (7.0 * t + phase).sin();
                    MdSample {
                        time: t,
                        kinetic: e,
                        potential: 0.0,
                        total: e,
                        temperature: 1.0,
                        pressure: 0.0,
                    }
                })
                .collect()
        };
        let wobble_only = energy_drift(&ripple(0.0)).unwrap();
        let with_trend = energy_drift(&ripple(rise / span)).unwrap();
        assert!(
            wobble_only < 0.15 * with_trend,
            "the wobble alone read {wobble_only} against {with_trend} with a trend"
        );
    }
}

#[test]
fn prop_the_green_kubo_estimate_scales_with_its_own_prefactor() {
    // The volume and temperature enter as a plain prefactor, so the
    // estimate must scale exactly with them however noisy the correlation
    // underneath is. That separates a prefactor error from a sampling one,
    // which a comparison against a reference value cannot.
    let mut rng = Rng::new(0x011D_9036);
    let dt = 0.01;
    for _ in 0..6 {
        let tau = 0.2 + rng.next_f64();
        let decay = (-dt / tau).exp();
        let noise = (1.0 - decay * decay).sqrt();
        let mut x = rng.next_gaussian();
        let series: Vec<f64> = (0..4_000)
            .map(|_| {
                x = x * decay + noise * rng.next_gaussian();
                x
            })
            .collect();
        let base = green_kubo_viscosity_lite(&series, dt, 2.0, 1.0).unwrap();
        assert!(base > 0.0);
        assert!(close(
            green_kubo_viscosity_lite(&series, dt, 6.0, 1.0).unwrap(),
            3.0 * base,
            1e-9 * base
        ));
        assert!(close(
            green_kubo_viscosity_lite(&series, dt, 2.0, 4.0).unwrap(),
            base / 4.0,
            1e-9 * base
        ));
        // Scaling the stress scales the estimate quadratically, since the
        // correlation is a product of two of them.
        let louder: Vec<f64> = series.iter().map(|s| s * 3.0).collect();
        assert!(close(
            green_kubo_viscosity_lite(&louder, dt, 2.0, 1.0).unwrap(),
            9.0 * base,
            1e-8 * base
        ));
    }
}

#[test]
fn prop_the_phase_classification_is_total_and_stable() {
    // Every physical point gets a label, and no point on the interior of a
    // region changes label under a small perturbation -- a classifier with
    // an unreachable branch or an inverted comparison would show up as a
    // gap or as an island.
    let mut rng = Rng::new(0x011D_9037);
    let known = [
        "solid",
        "liquid",
        "gas",
        "gas-liquid coexistence",
        "supercritical fluid",
        "fluid",
    ];
    let mut seen: Vec<&str> = Vec::new();
    for _ in 0..4_000 {
        let t = rng.next_f64() * 3.0;
        let rho = rng.next_f64() * 1.2;
        let label = lj_phase_point(t, rho);
        assert!(known.contains(&label), "the classifier returned {label}");
        if !seen.contains(&label) {
            seen.push(label);
        }
        // Unphysical input is refused rather than guessed at.
        assert_eq!(lj_phase_point(-t - 0.1, rho), "unphysical");
        assert_eq!(lj_phase_point(t, -rho - 0.1), "unphysical");
    }
    for label in known {
        assert!(seen.contains(&label), "the region {label} is unreachable");
    }
}
