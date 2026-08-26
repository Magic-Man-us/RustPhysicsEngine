//! Properties of Lambert's problem and the orbital manoeuvres.
//!
//! Lambert's solver has one property that dominates all others: the orbit
//! it returns must actually connect the two points in the stated time.
//! That is checkable against a propagator which shares none of its code,
//! and it subsumes every internal consistency check -- a solver that got
//! the Stumpff functions, the bracket or the Lagrange coefficients wrong
//! would fail it.
//!
//! The manoeuvre formulas are mostly scaling laws and inequalities, and
//! those are testable across randomised parameters in a way that a single
//! worked example is not.

use rust_physics_engine::astrophysics::kepler::{
    orbit_period, propagate_kepler, state_from_elements, vis_viva,
};
use rust_physics_engine::astrophysics::lambert::{
    lambert_universal, porkchop_data, stumpff_c, stumpff_s, Ephemeris,
};
use rust_physics_engine::astrophysics::maneuvers::{
    combined_maneuver, gravity_assist_deflection, ground_track, j2_raan_drift,
    oberth_effect_dv, patched_conic_escape, sphere_of_influence, sun_synchronous_inclination,
};
use rust_physics_engine::astrophysics::orbital_elements::OrbitalElements;
use rust_physics_engine::math::Vec3;
use rust_physics_engine::monte_carlo::Rng;

const MU: f64 = 398_600.441_8;
const RE: f64 = 6378.137;
const J2: f64 = 1.082_626_68e-3;
const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn distance(a: Vec3, b: Vec3) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

fn random_elements(rng: &mut Rng) -> OrbitalElements {
    OrbitalElements {
        semi_major_axis: 8000.0 + 30000.0 * rng.next_f64(),
        eccentricity: 0.7 * rng.next_f64(),
        inclination: 0.1 + 2.9 * rng.next_f64(),
        longitude_ascending_node: TAU * rng.next_f64(),
        argument_periapsis: TAU * rng.next_f64(),
        true_anomaly: TAU * rng.next_f64(),
    }
}

#[test]
fn prop_the_stumpff_functions_match_their_defining_series() {
    // The implementation switches between series and closed forms, and
    // between three branches by sign. The series is the definition, so
    // agreeing with it everywhere is the check that covers all of them.
    let mut rng = Rng::new(0x0A57_4001);
    for _ in 0..2000 {
        let z = -60.0 + 100.0 * rng.next_f64();
        let series = |first: f64, gap: usize| {
            let mut term = first;
            let mut total = term;
            for k in 1..60 {
                term *= -z / ((2 * k + gap) as f64 * (2 * k + gap + 1) as f64);
                total += term;
            }
            total
        };
        let c = series(0.5, 1);
        let s = series(1.0 / 6.0, 2);
        assert!(
            (stumpff_c(z) - c).abs() < 1e-9 * c.abs().max(1.0),
            "C({z}) was {} against {c}",
            stumpff_c(z)
        );
        assert!(
            (stumpff_s(z) - s).abs() < 1e-9 * s.abs().max(1.0),
            "S({z}) was {} against {s}",
            stumpff_s(z)
        );
        // Both are positive everywhere on the real line, which is what
        // lets the universal formulation take square roots of them.
        assert!(stumpff_c(z) > 0.0 && stumpff_s(z) > 0.0);
    }
}

#[test]
fn prop_a_lambert_transfer_actually_connects_its_two_points() {
    // The definition, checked against a propagator that shares no code
    // with the solver. Everything internal to the solver is downstream of
    // this.
    let mut rng = Rng::new(0x0A57_4002);
    let mut solved = 0usize;
    for _ in 0..400 {
        let elements = random_elements(&mut rng);
        let (r_a, v_a) = state_from_elements(&elements, MU).unwrap();
        let period = orbit_period(elements.semi_major_axis, MU).unwrap();
        let tof = period * (0.03 + 0.6 * rng.next_f64());
        let (r_b, _) = propagate_kepler(r_a, v_a, tof, MU).unwrap();
        let prograde = r_a.cross(&v_a).z >= 0.0;
        let Ok((depart, arrive)) = lambert_universal(r_a, r_b, tof, MU, prograde) else {
            continue;
        };
        solved += 1;
        // Fly the answer and land on the target.
        let (landed, landed_v) = propagate_kepler(r_a, depart, tof, MU).unwrap();
        assert!(
            distance(landed, r_b) < 1e-6 * r_b.magnitude(),
            "the transfer missed by {} km",
            distance(landed, r_b)
        );
        assert!(distance(landed_v, arrive) < 1e-6 * arrive.magnitude());
        // The transfer stays in the plane the two radii span.
        let normal = r_a.cross(&r_b);
        if normal.magnitude() > 1e-6 * r_a.magnitude() * r_b.magnitude() {
            let unit = normal.normalized();
            assert!(
                depart.dot(&unit).abs() < 1e-8 * depart.magnitude(),
                "the departure velocity left the transfer plane"
            );
        }
    }
    assert!(solved > 380, "only {solved} of 400 geometries were solved");
}

#[test]
fn prop_lambert_recovers_the_velocities_that_generated_the_arc() {
    // A stronger statement than connecting the points: the arc came from
    // a known orbit, and there is only one zero-revolution transfer with
    // that geometry and duration, so the solver must find that orbit.
    let mut rng = Rng::new(0x0A57_4003);
    for _ in 0..300 {
        let elements = random_elements(&mut rng);
        let (r_a, v_a) = state_from_elements(&elements, MU).unwrap();
        let period = orbit_period(elements.semi_major_axis, MU).unwrap();
        let tof = period * (0.05 + 0.5 * rng.next_f64());
        let (r_b, v_b) = propagate_kepler(r_a, v_a, tof, MU).unwrap();
        let prograde = r_a.cross(&v_a).z >= 0.0;
        let Ok((depart, arrive)) = lambert_universal(r_a, r_b, tof, MU, prograde) else {
            continue;
        };
        // A part in 1e7 rather than 1e8: the Lagrange form is
        // ill-conditioned as the transfer angle approaches pi, where
        // `r2 - f r1` is a difference of two nearly equal vectors. Over
        // three thousand draws the worst residual was 1.2e-8, at a
        // transfer angle of 179.99 degrees.
        assert!(
            distance(depart, v_a) < 1e-7 * v_a.magnitude(),
            "the departure velocity was off by {}",
            distance(depart, v_a)
        );
        assert!(distance(arrive, v_b) < 1e-7 * v_b.magnitude());
        // And the transfer's own energy matches the generating orbit's.
        let energy = 0.5 * depart.magnitude_squared() - MU / r_a.magnitude();
        assert!((energy + MU / (2.0 * elements.semi_major_axis)).abs() < 1e-6 * energy.abs());
    }
}

#[test]
fn prop_reflecting_the_problem_reflects_the_answer() {
    // Gravity is central, so reflection through the equatorial plane is a
    // symmetry of the whole two-body problem. The transfer angle is
    // unchanged by it -- and so is the z component of `r_a x r_b`, which
    // is what decides the direction -- so the *same* prograde flag gives
    // the mirrored solution.
    let mut rng = Rng::new(0x0A57_4004);
    let mut checked = 0usize;
    for _ in 0..300 {
        let draw = |rng: &mut Rng| {
            Vec3::new(
                -20000.0 + 40000.0 * rng.next_f64(),
                -20000.0 + 40000.0 * rng.next_f64(),
                -10000.0 + 20000.0 * rng.next_f64(),
            )
        };
        let r_a = draw(&mut rng);
        let r_b = draw(&mut rng);
        if r_a.magnitude() < 7000.0 || r_b.magnitude() < 7000.0 {
            continue;
        }
        let tof = 2000.0 + 30000.0 * rng.next_f64();
        let prograde = rng.next_f64() < 0.5;
        let Ok((depart, arrive)) = lambert_universal(r_a, r_b, tof, MU, prograde) else {
            continue;
        };
        let flip = |v: Vec3| Vec3::new(v.x, v.y, -v.z);
        let Ok((mirrored, mirrored_arrive)) =
            lambert_universal(flip(r_a), flip(r_b), tof, MU, prograde)
        else {
            continue;
        };
        checked += 1;
        assert!(
            distance(flip(mirrored), depart) < 1e-8 * depart.magnitude(),
            "the mirrored departure differed: {depart:?} against {mirrored:?}"
        );
        assert!(
            distance(flip(mirrored_arrive), arrive) < 1e-8 * arrive.magnitude(),
            "the mirrored arrival differed"
        );
    }
    assert!(checked > 150, "only {checked} pairs were comparable");
}

#[test]
fn prop_a_porkchop_cell_is_the_excess_speed_squared_of_its_own_transfer() {
    // Each entry must be reproducible from the Lambert solution it came
    // from, which pins the definition rather than leaving it a plausible
    // number.
    let mut rng = Rng::new(0x0A57_4005);
    for _ in 0..30 {
        let departures: Vec<Ephemeris> = (0..4)
            .map(|i| {
                let elements = random_elements(&mut rng);
                let (r, v) = state_from_elements(&elements, MU).unwrap();
                (i as f64 * 500.0, r, v)
            })
            .collect();
        let arrivals: Vec<Ephemeris> = (0..4)
            .map(|j| {
                let elements = random_elements(&mut rng);
                let (r, v) = state_from_elements(&elements, MU).unwrap();
                (4000.0 + j as f64 * 2500.0, r, v)
            })
            .collect();
        let grid = porkchop_data(&departures, &arrivals, MU, true).unwrap();
        assert_eq!(grid.len(), departures.len());
        for (i, row) in grid.iter().enumerate() {
            assert_eq!(row.len(), arrivals.len());
            for (j, cell) in row.iter().enumerate() {
                let tof = arrivals[j].0 - departures[i].0;
                let direct = lambert_universal(departures[i].1, arrivals[j].1, tof, MU, true);
                match (cell, direct) {
                    (Some(c3), Ok((depart, _))) => {
                        let excess = Vec3::new(
                            depart.x - departures[i].2.x,
                            depart.y - departures[i].2.y,
                            depart.z - departures[i].2.z,
                        );
                        assert!(
                            (c3 - excess.magnitude_squared()).abs() < 1e-9 * c3.max(1.0),
                            "cell {i},{j} was {c3} against {}",
                            excess.magnitude_squared()
                        );
                        assert!(*c3 >= 0.0);
                    }
                    (None, Err(_)) => {}
                    (a, b) => panic!("cell {i},{j} disagreed with the solver: {a:?}, {}", b.is_ok()),
                }
            }
        }
    }
}

#[test]
fn prop_a_combined_burn_never_costs_more_than_two_separate_ones() {
    // The triangle inequality on velocity, which holds for every pair of
    // speeds and every angle between them.
    let mut rng = Rng::new(0x0A57_4006);
    for _ in 0..1000 {
        let v1 = 0.1 + 12.0 * rng.next_f64();
        let v2 = 0.1 + 12.0 * rng.next_f64();
        let angle = PI * rng.next_f64();
        let together = combined_maneuver(v1, v2, angle).unwrap();
        // Any decomposition into a speed change and a rotation.
        let speed_then_turn = (v2 - v1).abs() + 2.0 * v2 * (0.5 * angle).sin();
        let turn_then_speed = 2.0 * v1 * (0.5 * angle).sin() + (v2 - v1).abs();
        assert!(together <= speed_then_turn + 1e-12);
        assert!(together <= turn_then_speed + 1e-12);
        // Bounded below by the speed change and above by the sum.
        assert!(together >= (v2 - v1).abs() - 1e-12);
        assert!(together <= v1 + v2 + 1e-12);
        // Symmetric in the two speeds, and even in the angle.
        assert!((together - combined_maneuver(v2, v1, angle).unwrap()).abs() < 1e-12);
        assert!((together - combined_maneuver(v1, v2, -angle).unwrap()).abs() < 1e-12);
        // Monotone in the angle, which is what makes a plane change dear.
        if angle < PI - 0.01 {
            assert!(combined_maneuver(v1, v2, angle + 0.01).unwrap() > together);
        }
    }
}

#[test]
fn prop_the_manoeuvre_formulas_obey_their_scaling_laws() {
    let mut rng = Rng::new(0x0A57_4007);
    for _ in 0..500 {
        let distance_to = 1e6 + 1e9 * rng.next_f64();
        let ratio = 1e-9 + 1e-3 * rng.next_f64();
        let primary = 1e24 * (1.0 + rng.next_f64());
        let body = primary * ratio;
        let soi = sphere_of_influence(distance_to, body, primary).unwrap();
        assert!(soi > 0.0 && soi < distance_to, "the sphere reached {soi} of {distance_to}");
        // Linear in the separation, two-fifths power in the mass ratio.
        let factor = 0.2 + 8.0 * rng.next_f64();
        assert!(
            (sphere_of_influence(distance_to * factor, body, primary).unwrap() / soi - factor)
                .abs()
                < 1e-10 * factor
        );
        let heavier = sphere_of_influence(distance_to, body * 2.0, primary).unwrap();
        assert!((heavier / soi - 2.0f64.powf(0.4)).abs() < 1e-10);
        // It always exceeds the radius at which the pulls balance.
        assert!(soi > distance_to * ratio.sqrt());

        // Escaping costs less than escaping with speed to spare, and both
        // are below the escape speed itself.
        let radius = 6500.0 + 40000.0 * rng.next_f64();
        let circular = (MU / radius).sqrt();
        let bare = patched_conic_escape(radius, MU, 0.0).unwrap();
        assert!((bare / circular - (2.0f64.sqrt() - 1.0)).abs() < 1e-12);
        let excess = 10.0 * rng.next_f64();
        let with_excess = patched_conic_escape(radius, MU, excess).unwrap();
        assert!(with_excess >= bare);
        // The burn buys more than it costs, which is the Oberth effect.
        assert!(with_excess - bare <= excess + 1e-12, "the excess cost more than it is worth");
    }
}

#[test]
fn prop_a_flyby_turns_between_nothing_and_a_reversal() {
    let mut rng = Rng::new(0x0A57_4008);
    for _ in 0..500 {
        let mu = 1e4 + 1e8 * rng.next_f64();
        let periapsis = 1000.0 + 100_000.0 * rng.next_f64();
        let excess = 0.1 + 30.0 * rng.next_f64();
        let turn = gravity_assist_deflection(excess, periapsis, mu).unwrap();
        assert!(turn > 0.0 && turn < PI, "the turn was {turn}");
        // Faster is straighter, closer is sharper: both monotone.
        assert!(gravity_assist_deflection(excess * 1.5, periapsis, mu).unwrap() < turn);
        assert!(gravity_assist_deflection(excess, periapsis * 0.7, mu).unwrap() > turn);
        assert!(gravity_assist_deflection(excess, periapsis, mu * 1.5).unwrap() > turn);
        // The turn depends only on the combination r_p v^2 / mu.
        let scaled = gravity_assist_deflection(excess * 2.0, periapsis * 0.25, mu).unwrap();
        assert!((scaled - turn).abs() < 1e-9, "the invariant combination moved the turn");
    }
}

#[test]
fn prop_a_burn_gains_energy_at_the_rate_the_oberth_effect_says() {
    // To first order the gain is `v dv`, so the same delta-v is worth
    // more where the craft is already fast. The check is against the
    // exact expression, whose second-order term is `dv^2/2`.
    let mut rng = Rng::new(0x0A57_4009);
    for _ in 0..500 {
        let a = 8000.0 + 40000.0 * rng.next_f64();
        let e = 0.8 * rng.next_f64();
        let periapsis = a * (1.0 - e);
        let apoapsis = a * (1.0 + e);
        let fast = vis_viva(periapsis, a, MU).unwrap();
        let slow = vis_viva(apoapsis, a, MU).unwrap();
        assert!(fast >= slow);
        let before = -MU / (2.0 * a);
        let burn = 0.01 + 1.0 * rng.next_f64();
        for (speed, radius) in [(fast, periapsis), (slow, apoapsis)] {
            let after = oberth_effect_dv(speed, burn, radius, MU).unwrap();
            let gain = after - before;
            assert!(
                (gain - (speed * burn + 0.5 * burn * burn)).abs() < 1e-8 * gain.abs().max(1.0),
                "the gain was {gain} against v dv + dv^2/2"
            );
            assert!(gain > 0.0);
        }
        // And periapsis is the better place, by the speed ratio.
        if e > 0.05 {
            let low = oberth_effect_dv(fast, burn, periapsis, MU).unwrap() - before;
            let high = oberth_effect_dv(slow, burn, apoapsis, MU).unwrap() - before;
            assert!(low > high, "apoapsis was the better place to burn");
        }
    }
}

#[test]
fn prop_the_nodal_drift_follows_its_cosine_and_the_sun_synchronous_inverse_works() {
    let mut rng = Rng::new(0x0A57_400A);
    for _ in 0..400 {
        let a = RE + 200.0 + 2000.0 * rng.next_f64();
        let e = 0.2 * rng.next_f64();
        let inclination = PI * rng.next_f64();
        let drift = j2_raan_drift(a, e, inclination, J2, RE, MU).unwrap();
        assert!(drift.is_finite());
        // The sign is the cosine's, reversed.
        assert_eq!(drift < 0.0, inclination < PI / 2.0 - 1e-12);
        // It scales exactly with the cosine at fixed geometry.
        let other = 0.1 + 2.9 * rng.next_f64();
        let scaled = j2_raan_drift(a, e, other, J2, RE, MU).unwrap();
        assert!(
            (scaled * inclination.cos() - drift * other.cos()).abs()
                < 1e-18 + 1e-9 * drift.abs()
        );
        // And linearly with J2.
        assert!(
            (j2_raan_drift(a, e, inclination, 2.0 * J2, RE, MU).unwrap() - 2.0 * drift).abs()
                < 1e-9 * drift.abs()
        );

        // The inverse: solving for the inclination that gives a drift
        // returns one that does.
        if drift.abs() > 1e-12 {
            let recovered = sun_synchronous_inclination(a, e, J2, RE, MU, drift).unwrap();
            let check = j2_raan_drift(a, e, recovered, J2, RE, MU).unwrap();
            assert!(
                (check - drift).abs() < 1e-9 * drift.abs(),
                "the inverse gave {check} against {drift}"
            );
        }
    }
}

#[test]
fn prop_a_ground_track_is_bounded_by_its_inclination() {
    // The latitude cannot exceed the orbit's inclination, and for a
    // prograde orbit below ninety degrees it reaches it exactly. That
    // bound is why a polar orbit is needed to see the poles.
    let mut rng = Rng::new(0x0A57_400B);
    for _ in 0..60 {
        let inclination = 0.05 + (PI - 0.1) * rng.next_f64();
        let elements = OrbitalElements {
            semi_major_axis: RE + 300.0 + 2000.0 * rng.next_f64(),
            eccentricity: 0.1 * rng.next_f64(),
            inclination,
            longitude_ascending_node: TAU * rng.next_f64(),
            argument_periapsis: TAU * rng.next_f64(),
            true_anomaly: TAU * rng.next_f64(),
        };
        let (r0, v0) = state_from_elements(&elements, MU).unwrap();
        let period = orbit_period(elements.semi_major_axis, MU).unwrap();
        let track = ground_track(r0, v0, MU, TAU / 86_164.0, 1.5 * period, 800).unwrap();
        assert_eq!(track.len(), 800);
        let reach = if inclination > PI / 2.0 { PI - inclination } else { inclination };
        let highest = track.iter().map(|(_, lat)| lat.abs()).fold(0.0f64, f64::max);
        assert!(
            highest <= reach + 1e-9,
            "it reached {} on a {} degree orbit",
            highest.to_degrees(),
            inclination.to_degrees()
        );
        assert!((highest - reach).abs() < 5e-3, "it only reached {}", highest.to_degrees());
        for (longitude, latitude) in &track {
            assert!((-PI..=PI).contains(longitude), "a longitude was {longitude}");
            assert!(latitude.abs() <= PI / 2.0 + 1e-12);
        }
    }
}

