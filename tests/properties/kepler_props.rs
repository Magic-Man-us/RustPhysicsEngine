//! Properties of the Kepler and two-body propagation module.
//!
//! Orbital mechanics is unusually rich in exact invariants, and they fall
//! into three kinds that check different things.
//!
//! *Inverse pairs.* Kepler's equation is transcendental one way and
//! trivial the other, so solving it and reading it forward must compose to
//! the identity. The same holds for the three anomalies, and for the
//! conversion between elements and state vectors.
//!
//! *Conserved quantities.* Energy, the angular momentum vector, and the
//! eccentricity vector are constants of the two-body motion. A propagator
//! that drifts in any of them has a defect that no plausibility check on
//! the trajectory would reveal -- a wrong sign in the Lagrange
//! coefficients still produces a smooth curve through space.
//!
//! *Group structure.* Propagation by `t1` then `t2` must equal
//! propagation by `t1 + t2`, and propagating by `-t` must undo `t`.
//! Two-body motion is a one-parameter group, and a propagator that is not
//! is wrong somewhere.

use rust_physics_engine::astrophysics::kepler::{
    eccentric_from_true, kepler_solve_elliptic, kepler_solve_hyperbolic, mean_from_eccentric,
    orbit_period, propagate_kepler, state_from_elements, true_from_eccentric, vis_viva,
};
use rust_physics_engine::astrophysics::orbital_elements::OrbitalElements;
use rust_physics_engine::math::Vec3;
use rust_physics_engine::monte_carlo::Rng;

/// Earth's gravitational parameter, km^3/s^2.
const MU: f64 = 398_600.441_8;
const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn angle_gap(a: f64, b: f64) -> f64 {
    (a - b + PI).rem_euclid(TAU) - PI
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

/// A random bound orbit, kept away from the inclinations where the element
/// set itself is degenerate rather than the conversion being wrong.
fn random_elements(rng: &mut Rng, max_eccentricity: f64) -> OrbitalElements {
    OrbitalElements {
        semi_major_axis: 7000.0 + 40000.0 * rng.next_f64(),
        eccentricity: max_eccentricity * rng.next_f64(),
        inclination: 0.05 + (PI - 0.1) * rng.next_f64(),
        longitude_ascending_node: TAU * rng.next_f64(),
        argument_periapsis: TAU * rng.next_f64(),
        true_anomaly: TAU * rng.next_f64(),
    }
}

#[test]
fn prop_keplers_equation_is_solved_at_every_eccentricity_below_one() {
    // Substituting the answer back is the only check that does not trust
    // the solver's own machinery. Eccentricities to 0.9999 are included
    // because that is where a poor seed leaves the basin.
    let mut rng = Rng::new(0x0A57_2001);
    for _ in 0..400 {
        let e = rng.next_f64().powi(3) * 0.9999;
        for _ in 0..20 {
            let m = TAU * rng.next_f64();
            let anomaly = kepler_solve_elliptic(m, e, 1e-14).unwrap();
            let residual = anomaly - e * anomaly.sin() - m;
            assert!(residual.abs() < 1e-12, "at e={e}, M={m} the residual was {residual}");
            assert!((0.0..=TAU).contains(&anomaly));
            // Reading the equation forward returns the mean anomaly.
            let back = mean_from_eccentric(anomaly, e).unwrap();
            assert!(angle_gap(back, m).abs() < 1e-11);
        }
    }
}

#[test]
fn prop_the_solver_is_monotone_in_the_mean_anomaly() {
    // E is a strictly increasing function of M for any eccentricity below
    // one, since dM/dE = 1 - e cos E is positive. A solver that lands in
    // the wrong basin breaks the ordering even where the residual is
    // small.
    let mut rng = Rng::new(0x0A57_2002);
    for _ in 0..200 {
        let e = 0.999 * rng.next_f64();
        let mut previous = -1.0;
        for k in 0..200 {
            let m = TAU * k as f64 / 200.0;
            let anomaly = kepler_solve_elliptic(m, e, 1e-14).unwrap();
            assert!(anomaly >= previous - 1e-12, "at e={e} the anomaly fell at M={m}");
            previous = anomaly;
        }
    }
}

#[test]
fn prop_the_three_anomalies_form_a_cycle_that_closes() {
    let mut rng = Rng::new(0x0A57_2003);
    for _ in 0..400 {
        let e = 0.99 * rng.next_f64();
        for _ in 0..10 {
            let nu = TAU * rng.next_f64();
            let eccentric = eccentric_from_true(nu, e).unwrap();
            let mean = mean_from_eccentric(eccentric, e).unwrap();
            let solved = kepler_solve_elliptic(mean, e, 1e-14).unwrap();
            let round = true_from_eccentric(solved, e).unwrap();
            assert!(angle_gap(round, nu).abs() < 1e-9, "nu={nu} at e={e} came back {round}");
            // Each conversion inverts its partner on its own.
            assert!(angle_gap(true_from_eccentric(eccentric, e).unwrap(), nu).abs() < 1e-11);
            assert!(angle_gap(solved, eccentric).abs() < 1e-10);
            // All three land in the same half of the orbit.
            assert_eq!(nu < PI, eccentric < PI, "the quadrant was lost");
            assert_eq!(nu < PI, mean < PI, "the quadrant was lost");
        }
    }
}

#[test]
fn prop_the_hyperbolic_equation_is_solved_and_is_odd() {
    let mut rng = Rng::new(0x0A57_2004);
    for _ in 0..300 {
        let e = 1.0005 + 20.0 * rng.next_f64();
        for _ in 0..15 {
            let m = -60.0 + 120.0 * rng.next_f64();
            let h = kepler_solve_hyperbolic(m, e, 1e-13).unwrap();
            let residual = e * h.sinh() - h - m;
            assert!(
                residual.abs() < 1e-10 * (1.0 + m.abs()),
                "at e={e}, M={m} the residual was {residual}"
            );
            assert!(h.is_finite());
            // Odd in the mean anomaly.
            let mirrored = kepler_solve_hyperbolic(-m, e, 1e-13).unwrap();
            assert!((h + mirrored).abs() < 1e-9 * (1.0 + h.abs()));
            // And the sign follows the mean anomaly's.
            assert_eq!(m > 0.0, h > 0.0, "the sign was lost at M={m}");
        }
    }
}

#[test]
fn prop_elements_and_state_vectors_invert_each_other() {
    let mut rng = Rng::new(0x0A57_2005);
    for _ in 0..400 {
        let elements = random_elements(&mut rng, 0.9);
        let (r, v) = state_from_elements(&elements, MU).unwrap();
        assert!(r.magnitude().is_finite() && v.magnitude().is_finite());
        let recovered = OrbitalElements::from_state_vectors(r, v, MU);
        assert!(
            (recovered.semi_major_axis - elements.semi_major_axis).abs()
                < 1e-7 * elements.semi_major_axis
        );
        assert!((recovered.eccentricity - elements.eccentricity).abs() < 1e-8);
        assert!((recovered.inclination - elements.inclination).abs() < 1e-8);
        assert!(
            angle_gap(recovered.longitude_ascending_node, elements.longitude_ascending_node).abs()
                < 1e-6
        );
        assert!(angle_gap(recovered.argument_periapsis, elements.argument_periapsis).abs() < 1e-6);
        assert!(angle_gap(recovered.true_anomaly, elements.true_anomaly).abs() < 1e-6);
        // The state itself round trips more tightly than the angles do,
        // since the angles can trade against each other.
        let (r2, v2) = state_from_elements(&recovered, MU).unwrap();
        assert!(distance(r2, r) < 1e-7 * r.magnitude());
        assert!(distance(v2, v) < 1e-7 * v.magnitude());
    }
}

#[test]
fn prop_the_state_satisfies_the_conic_equation_and_vis_viva() {
    // Three independent statements about the same construction: the
    // radius from the conic, the angular momentum from the semi-latus
    // rectum, and the speed from the energy.
    let mut rng = Rng::new(0x0A57_2006);
    for _ in 0..400 {
        let elements = random_elements(&mut rng, 0.95);
        let (a, e) = (elements.semi_major_axis, elements.eccentricity);
        let (r, v) = state_from_elements(&elements, MU).unwrap();
        let p = a * (1.0 - e * e);
        let radius = p / (1.0 + e * elements.true_anomaly.cos());
        assert!((r.magnitude() - radius).abs() < 1e-8 * radius);
        // Between periapsis and apoapsis, always.
        assert!(r.magnitude() >= a * (1.0 - e) - 1e-8 * a);
        assert!(r.magnitude() <= a * (1.0 + e) + 1e-8 * a);
        let h = r.cross(&v);
        assert!((h.magnitude() - (MU * p).sqrt()).abs() < 1e-7 * (MU * p).sqrt());
        let speed = vis_viva(r.magnitude(), a, MU).unwrap();
        assert!((v.magnitude() - speed).abs() < 1e-7 * speed);
        // The eccentricity vector points at periapsis and has length e.
        let ecc = Vec3::new(
            v.magnitude_squared() / MU * r.x - r.dot(&v) / MU * v.x - r.x / r.magnitude(),
            v.magnitude_squared() / MU * r.y - r.dot(&v) / MU * v.y - r.y / r.magnitude(),
            v.magnitude_squared() / MU * r.z - r.dot(&v) / MU * v.z - r.z / r.magnitude(),
        );
        assert!((ecc.magnitude() - e).abs() < 1e-8, "the eccentricity vector had length {}", ecc.magnitude());
    }
}

#[test]
fn prop_propagation_conserves_what_the_two_body_problem_conserves() {
    // Energy, the angular momentum *vector*, and the eccentricity vector.
    // A sign error in the Lagrange coefficients leaves a smooth curve
    // through space and destroys all three.
    let mut rng = Rng::new(0x0A57_2007);
    for _ in 0..200 {
        let elements = random_elements(&mut rng, 0.85);
        let (r0, v0) = state_from_elements(&elements, MU).unwrap();
        let period = orbit_period(elements.semi_major_axis, MU).unwrap();
        let energy0 = 0.5 * v0.magnitude_squared() - MU / r0.magnitude();
        let h0 = r0.cross(&v0);
        for _ in 0..6 {
            let dt = (-3.0 + 6.0 * rng.next_f64()) * period;
            let (r, v) = propagate_kepler(r0, v0, dt, MU).unwrap();
            let energy = 0.5 * v.magnitude_squared() - MU / r.magnitude();
            assert!(
                (energy - energy0).abs() < 1e-9 * energy0.abs(),
                "energy went from {energy0} to {energy}"
            );
            let h = r.cross(&v);
            assert!(distance(h, h0) < 1e-8 * h0.magnitude(), "the angular momentum moved");
            // The eccentricity vector is fixed too, which pins the
            // orientation of the orbit within its plane.
            let ecc = |r: Vec3, v: Vec3| {
                let s = v.magnitude_squared() / MU;
                let d = r.dot(&v) / MU;
                Vec3::new(
                    s * r.x - d * v.x - r.x / r.magnitude(),
                    s * r.y - d * v.y - r.y / r.magnitude(),
                    s * r.z - d * v.z - r.z / r.magnitude(),
                )
            };
            assert!(distance(ecc(r, v), ecc(r0, v0)) < 1e-7, "the periapsis direction moved");
        }
    }
}

#[test]
fn prop_propagation_is_a_one_parameter_group() {
    // Composition and reversal. Two-body motion is a flow, so propagating
    // by t1 then t2 is propagating by t1 + t2, and -t undoes t.
    let mut rng = Rng::new(0x0A57_2008);
    for _ in 0..200 {
        let elements = random_elements(&mut rng, 0.8);
        let (r0, v0) = state_from_elements(&elements, MU).unwrap();
        let period = orbit_period(elements.semi_major_axis, MU).unwrap();
        let t1 = (-1.0 + 2.0 * rng.next_f64()) * period;
        let t2 = (-1.0 + 2.0 * rng.next_f64()) * period;
        let (ra, va) = propagate_kepler(r0, v0, t1, MU).unwrap();
        let (rb, vb) = propagate_kepler(ra, va, t2, MU).unwrap();
        let (rc, vc) = propagate_kepler(r0, v0, t1 + t2, MU).unwrap();
        assert!(distance(rb, rc) < 1e-6 * r0.magnitude(), "composition failed");
        assert!(distance(vb, vc) < 1e-6 * v0.magnitude());
        let (rd, vd) = propagate_kepler(ra, va, -t1, MU).unwrap();
        assert!(distance(rd, r0) < 1e-7 * r0.magnitude(), "reversal failed");
        assert!(distance(vd, v0) < 1e-7 * v0.magnitude());
    }
}

#[test]
fn prop_a_whole_number_of_periods_changes_nothing() {
    let mut rng = Rng::new(0x0A57_2009);
    for _ in 0..200 {
        let elements = random_elements(&mut rng, 0.85);
        let (r0, v0) = state_from_elements(&elements, MU).unwrap();
        let period = orbit_period(elements.semi_major_axis, MU).unwrap();
        for turns in [1.0f64, 2.0, 5.0, -1.0, -3.0] {
            let (r, v) = propagate_kepler(r0, v0, turns * period, MU).unwrap();
            assert!(
                distance(r, r0) < 1e-7 * r0.magnitude(),
                "after {turns} turns it was {} km away",
                distance(r, r0)
            );
            assert!(distance(v, v0) < 1e-7 * v0.magnitude());
        }
    }
}

#[test]
fn prop_propagating_agrees_with_advancing_the_mean_anomaly() {
    // Two entirely different routes: the Lagrange coefficients, and going
    // out to elements, adding n dt to the mean anomaly, and coming back.
    let mut rng = Rng::new(0x0A57_200A);
    for _ in 0..250 {
        let elements = random_elements(&mut rng, 0.75);
        let (a, e) = (elements.semi_major_axis, elements.eccentricity);
        let (r0, v0) = state_from_elements(&elements, MU).unwrap();
        let dt = orbit_period(a, MU).unwrap() * (-1.5 + 3.0 * rng.next_f64());
        let (r_prop, v_prop) = propagate_kepler(r0, v0, dt, MU).unwrap();

        let mean0 =
            mean_from_eccentric(eccentric_from_true(elements.true_anomaly, e).unwrap(), e).unwrap();
        let n = (MU / (a * a * a)).sqrt();
        let advanced =
            kepler_solve_elliptic((mean0 + n * dt).rem_euclid(TAU), e, 1e-14).unwrap();
        let moved =
            OrbitalElements { true_anomaly: true_from_eccentric(advanced, e).unwrap(), ..elements };
        let (r_el, v_el) = state_from_elements(&moved, MU).unwrap();
        assert!(
            distance(r_prop, r_el) < 1e-6 * r0.magnitude(),
            "the two routes differ by {} km",
            distance(r_prop, r_el)
        );
        assert!(distance(v_prop, v_el) < 1e-6 * v0.magnitude());
    }
}

#[test]
fn prop_an_unbound_trajectory_keeps_its_energy_and_never_returns() {
    // The hyperbolic branch has its own Lagrange coefficients, obtained by
    // substituting E = i H, which flips the sign of two of them.
    let mut rng = Rng::new(0x0A57_200B);
    for _ in 0..150 {
        let radius = 7000.0 + 20000.0 * rng.next_f64();
        let escape = (2.0 * MU / radius).sqrt();
        let speed = escape * (1.05 + 2.0 * rng.next_f64());
        // A little radial velocity as well, so the state is not
        // artificially at periapsis.
        let angle = 0.6 * rng.next_f64();
        let r0 = Vec3::new(radius, 0.0, 0.0);
        let v0 = Vec3::new(speed * angle.sin(), speed * angle.cos(), 0.0);
        let energy0 = 0.5 * speed * speed - MU / radius;
        assert!(energy0 > 0.0);
        let h0 = r0.cross(&v0);
        let mut previous = radius;
        for dt in [1.0f64, 50.0, 500.0, 5000.0, 50_000.0] {
            let (r, v) = propagate_kepler(r0, v0, dt, MU).unwrap();
            let energy = 0.5 * v.magnitude_squared() - MU / r.magnitude();
            assert!(
                (energy - energy0).abs() < 1e-9 * energy0,
                "at dt={dt} the energy went from {energy0} to {energy}"
            );
            assert!(distance(r.cross(&v), h0) < 1e-8 * h0.magnitude());
            assert!(r.magnitude() > previous, "an unbound trajectory turned around");
            previous = r.magnitude();
            // It composes and reverses like any other.
            let (back, back_v) = propagate_kepler(r, v, -dt, MU).unwrap();
            assert!(distance(back, r0) < 1e-6 * radius);
            assert!(distance(back_v, v0) < 1e-6 * speed);
        }
        // And its speed tends to the hyperbolic excess.
        let a = -MU / (2.0 * energy0);
        let excess = (MU / -a).sqrt();
        let (_, far) = propagate_kepler(r0, v0, 5_000_000.0, MU).unwrap();
        assert!(
            (far.magnitude() - excess).abs() < 0.05 * excess,
            "far out it was doing {} against an excess of {excess}",
            far.magnitude()
        );
    }
}

#[test]
fn prop_vis_viva_is_the_energy_equation_rearranged() {
    // One formula for all three conics. The check is that it agrees with
    // the energy it came from, at every radius the orbit reaches.
    let mut rng = Rng::new(0x0A57_200C);
    for _ in 0..400 {
        let a = 7000.0 + 40000.0 * rng.next_f64();
        let e = 0.95 * rng.next_f64();
        for k in 0..10 {
            let radius = a * (1.0 - e) + a * 2.0 * e * k as f64 / 9.0;
            let speed = vis_viva(radius, a, MU).unwrap();
            let energy = 0.5 * speed * speed - MU / radius;
            assert!(
                (energy + MU / (2.0 * a)).abs() < 1e-9 * (MU / (2.0 * a)),
                "the energy came out {energy} against {}",
                -MU / (2.0 * a)
            );
        }
        // Escape speed is the parabolic limit, and a hyperbola beats it.
        let radius = a * (1.0 - e);
        let escape = (2.0 * MU / radius).sqrt();
        assert!((vis_viva(radius, f64::INFINITY, MU).unwrap() - escape).abs() < 1e-9 * escape);
        assert!(vis_viva(radius, -a, MU).unwrap() > escape);
        assert!(vis_viva(radius, a, MU).unwrap() < escape);
        // The boundary is twice the semi-major axis, where the kinetic
        // energy runs out -- not apoapsis, which the formula knows
        // nothing about.
        assert!(vis_viva(2.0 * a * 0.999, a, MU).is_ok());
        assert!(vis_viva(2.0 * a * 1.001, a, MU).is_err());
        // Between apoapsis and 2a it still answers, with the speed a body
        // of that energy would have rather than one anything reaches.
        if e > 0.05 {
            assert!(vis_viva(a * (1.0 + e) * 1.02, a, MU).is_ok());
        }
    }
}

#[test]
fn prop_the_period_scales_as_the_three_halves_power_of_the_axis() {
    let mut rng = Rng::new(0x0A57_200D);
    for _ in 0..300 {
        let a = 1000.0 + 100_000.0 * rng.next_f64();
        let mu = 1e4 + 1e6 * rng.next_f64();
        let base = orbit_period(a, mu).unwrap();
        let factor = 0.1 + 20.0 * rng.next_f64();
        assert!(
            (orbit_period(a * factor, mu).unwrap() / base - factor.powf(1.5)).abs()
                < 1e-10 * factor.powf(1.5)
        );
        // And inversely with the square root of the gravitational
        // parameter.
        assert!(
            (orbit_period(a, mu * factor).unwrap() / base - 1.0 / factor.sqrt()).abs() < 1e-10
        );
        // A propagation over one period returns the orbit, which ties the
        // period to the propagator rather than leaving it a formula.
        let elements = OrbitalElements {
            semi_major_axis: a,
            eccentricity: 0.5 * rng.next_f64(),
            inclination: 0.3,
            longitude_ascending_node: 1.0,
            argument_periapsis: 2.0,
            true_anomaly: TAU * rng.next_f64(),
        };
        let (r0, v0) = state_from_elements(&elements, mu).unwrap();
        let (r, _) = propagate_kepler(r0, v0, base, mu).unwrap();
        assert!(distance(r, r0) < 1e-7 * r0.magnitude());
    }
}
