//! Lambert's problem: the orbit connecting two positions in a given time.
//!
//! # The problem and why it is hard
//!
//! Given where a spacecraft is, where it must be, and how long it has to
//! get there, find the transfer orbit. Stated that way it sounds like
//! [`crate::astrophysics::kepler::propagate_kepler`] run backwards, but it
//! is a genuinely different problem: propagation is an initial-value
//! problem with one answer, and Lambert's is a *boundary*-value problem
//! whose answer need not be unique.
//!
//! It is not, however, a problem of existence. Within a single revolution
//! a transfer exists for every positive flight time: making the trip
//! faster costs more energy without limit, and the minimum-energy
//! transfer is a particular duration rather than a floor on one. What
//! *does* fail is a degenerate geometry -- a transfer angle of zero or
//! exactly `pi`, where the two radii do not determine a plane and
//! infinitely many orbits connect the points.
//!
//! Only the zero-revolution solution is computed here, which is the one
//! interplanetary trajectory design starts from. Multi-revolution
//! transfers exist for longer flight times and are a separate search, with
//! two branches per revolution count; they are not attempted rather than
//! approximated.
//!
//! # The universal-variable formulation
//!
//! Every conic is covered by one iteration, on a variable `z` that is
//! positive for an ellipse, negative for a hyperbola and zero for a
//! parabola. The Stumpff functions `C(z)` and `S(z)` carry the difference,
//! and their series expansions near zero are what keep the parabolic case
//! from losing precision to cancellation -- the closed forms are `0/0`
//! there.

use crate::error::GeomError;
use crate::math::Vec3;

/// The Stumpff function `C(z)`.
///
/// `(1 - cos sqrt(z))/z` for positive `z` and the hyperbolic analogue for
/// negative, both of which are `0/0` at the origin. The series
/// `1/2 - z/24 + z^2/720 - ...` is used near zero, where the closed forms
/// lose their leading digits to cancellation, and the positive branch is
/// evaluated as `2 sin^2(sqrt(z)/2)/z` so that it stays accurate at the
/// other end of the range as well.
#[must_use]
pub fn stumpff_c(z: f64) -> f64 {
    if z.abs() < 0.1 {
        // Six terms carry it to a part in 1e-17 over this range.
        let mut term = 0.5;
        let mut total = term;
        for k in 1..8 {
            term *= -z / ((2 * k + 1) as f64 * (2 * k + 2) as f64);
            total += term;
        }
        return total;
    }
    if z > 0.0 {
        // The half-angle form `2 sin^2(u/2)` rather than `1 - cos u`. The
        // two are identical in exact arithmetic and not in floating point:
        // near `z = 4 pi^2`, which is the single-revolution boundary the
        // Lambert bracket runs up to, `cos u` is within an ulp of one and
        // the subtraction keeps no digits at all -- it can return exactly
        // zero, or negative. The sine is small there instead of large, so
        // squaring it loses nothing.
        let root = z.sqrt();
        2.0 * (0.5 * root).sin().powi(2) / z
    } else {
        let root = (-z).sqrt();
        (root.cosh() - 1.0) / (-z)
    }
}

/// The Stumpff function `S(z)`.
///
/// `(sqrt(z) - sin sqrt(z))/z^(3/2)` for positive `z`, with the series
/// `1/6 - z/120 + z^2/5040 - ...` near the origin for the same reason as
/// [`stumpff_c`] -- and worse, since the numerator there is a difference
/// of two nearly equal quantities that agree to three orders.
#[must_use]
pub fn stumpff_s(z: f64) -> f64 {
    if z.abs() < 0.1 {
        let mut term = 1.0 / 6.0;
        let mut total = term;
        for k in 1..8 {
            term *= -z / ((2 * k + 2) as f64 * (2 * k + 3) as f64);
            total += term;
        }
        return total;
    }
    if z > 0.0 {
        let root = z.sqrt();
        (root - root.sin()) / (z * root)
    } else {
        let root = (-z).sqrt();
        (root.sinh() - root) / (root * root * root)
    }
}

/// Solves Lambert's problem by universal variables, returning the
/// departure and arrival velocities.
///
/// `prograde` selects the transfer direction: true takes the short way
/// round in the sense of increasing right ascension, false the long way.
/// The two are genuinely different orbits with different flight paths and
/// different costs, and which one is wanted is not deducible from the
/// endpoints -- the transfer angle is `theta` one way and `2 pi - theta`
/// the other.
///
/// The iteration is bisection on `z`. Bisection rather than Newton because
/// the flight time is monotone in `z`, so bisection cannot fail, and the
/// derivative a Newton step needs is itself delicate near the parabolic
/// point.
///
/// Accuracy degrades as the transfer angle approaches `pi`. The
/// velocities are recovered as `(r2 - f r1)/g`, and near a half turn
/// `f` approaches one with `r2` near `-r1`, so the numerator is a
/// difference of nearly equal vectors. Over three thousand randomised
/// geometries the worst departure velocity was off by a part in 1e8, and
/// that case had a transfer angle of 179.99 degrees. Exactly `pi` is
/// refused; the approach to it is merely imprecise.
///
/// # Errors
/// Returns an error for a non-positive gravitational parameter or flight
/// time, a position at the origin, or a transfer angle of zero or exactly
/// `pi`, where the plane is undefined and infinitely many orbits connect
/// the points.
pub fn lambert_universal(
    r1: Vec3,
    r2: Vec3,
    tof: f64,
    mu: f64,
    prograde: bool,
) -> Result<(Vec3, Vec3), GeomError> {
    if !(mu > 0.0) || !(tof > 0.0) || !mu.is_finite() || !tof.is_finite() {
        return Err(GeomError::InvalidArgument("lambert_universal: bad time or parameter"));
    }
    let (m1, m2) = (r1.magnitude(), r2.magnitude());
    if !(m1 > 0.0) || !(m2 > 0.0) || !m1.is_finite() || !m2.is_finite() {
        return Err(GeomError::InvalidArgument("lambert_universal: a position is degenerate"));
    }
    let cos_theta = (r1.dot(&r2) / (m1 * m2)).clamp(-1.0, 1.0);
    let cross = r1.cross(&r2);
    // The transfer plane is the one containing both radii. Which way round
    // it is travelled is the caller's choice, and it changes the orbit.
    let mut theta = cos_theta.acos();
    let direct = cross.z >= 0.0;
    if prograde != direct {
        theta = std::f64::consts::TAU - theta;
    }
    let sin_theta = theta.sin();
    if sin_theta.abs() < 1e-12 {
        return Err(GeomError::Degenerate(
            "the transfer angle is zero or pi: the plane is undefined and the solution is not unique",
        ));
    }
    let a_coefficient = sin_theta * (m1 * m2 / (1.0 - cos_theta)).sqrt();
    if !a_coefficient.is_finite() || a_coefficient.abs() < 1e-12 {
        return Err(GeomError::Degenerate("the transfer geometry is degenerate"));
    }

    // y(z), the chord parameter, and the flight time it implies.
    let y_of = |z: f64| -> f64 {
        let c = stumpff_c(z);
        m1 + m2 + a_coefficient * (z * stumpff_s(z) - 1.0) / c.sqrt()
    };
    let time_of = |z: f64| -> Option<f64> {
        let y = y_of(z);
        if y < 0.0 {
            return None;
        }
        let c = stumpff_c(z);
        let x = (y / c).sqrt();
        let t = (x * x * x * stumpff_s(z) + a_coefficient * y.sqrt()) / mu.sqrt();
        t.is_finite().then_some(t)
    };

    // Bracket. The flight time increases with z, and the useful range is
    // bounded above by the single-revolution boundary at 4 pi^2, where the
    // transfer closes on itself and the time diverges. Below, the walk
    // stops at the first z that is either fast enough or has no positive
    // chord at all -- the latter is a valid lower bracket, since the
    // bisection treats a missing solution as "too fast" and moves up.
    let ceiling = 4.0 * std::f64::consts::PI * std::f64::consts::PI;
    let mut high = ceiling - 1e-8;
    let mut low = -1.0;
    let mut bracketed = false;
    for _ in 0..200 {
        match time_of(low) {
            Some(t) if t <= tof => {
                bracketed = true;
                break;
            }
            None => {
                bracketed = true;
                break;
            }
            Some(_) => low *= 2.0,
        }
        if low < -1e14 {
            break;
        }
    }
    if !bracketed {
        return Err(GeomError::Degenerate("no transfer is fast enough for that flight time"));
    }
    let Some(long) = time_of(high) else {
        return Err(GeomError::Degenerate("the slow end of the bracket has no transfer"));
    };
    if long < tof {
        return Err(GeomError::Degenerate(
            "that flight time exceeds what a single-revolution transfer can take",
        ));
    }

    let mut z = 0.5 * (low + high);
    for _ in 0..300 {
        match time_of(z) {
            Some(t) if t < tof => low = z,
            Some(_) => high = z,
            None => low = z,
        }
        z = 0.5 * (low + high);
        // Tight, because the velocities are read off `y(z)` and the
        // sensitivity `dy/dz` carries any slack in z straight into them:
        // stopping at 1e-13 leaves the recovered departure velocity off
        // by a part in 1e8, which is visible against a propagator.
        if high - low < 1e-15 * (1.0 + z.abs()) {
            break;
        }
    }
    let y = y_of(z);
    if !(y > 0.0) {
        return Err(GeomError::Degenerate("the converged transfer has no positive chord"));
    }
    // Lagrange coefficients read straight off the converged geometry.
    let f = 1.0 - y / m1;
    let g = a_coefficient * (y / mu).sqrt();
    let g_dot = 1.0 - y / m2;
    if !(g.abs() > 0.0) || !g.is_finite() {
        return Err(GeomError::Degenerate("the transfer's g coefficient vanished"));
    }
    let v1 = Vec3::new((r2.x - f * r1.x) / g, (r2.y - f * r1.y) / g, (r2.z - f * r1.z) / g);
    let v2 = Vec3::new(
        (g_dot * r2.x - r1.x) / g,
        (g_dot * r2.y - r1.y) / g,
        (g_dot * r2.z - r1.z) / g,
    );
    if !v1.magnitude().is_finite() || !v2.magnitude().is_finite() {
        return Err(GeomError::Degenerate("the transfer velocities are not finite"));
    }
    Ok((v1, v2))
}

/// One body's state at one epoch: `(epoch, position, velocity)`.
pub type Ephemeris = (f64, Vec3, Vec3);

/// A porkchop grid of departure characteristic energies.
///
/// Entry `[i][j]` is the departure `C3 = v_infinity^2` for leaving
/// `departures[i]` and arriving at `arrivals[j]`, with the flight time
/// taken as the difference of their epochs. `None` marks a pair with no
/// transfer: a non-positive flight time, a degenerate geometry, or a
/// duration outside what one revolution allows.
///
/// `C3` rather than delta-v because it is what a launch vehicle's
/// performance is quoted against: the energy left over after escaping,
/// which is what the upper stage must supply. The characteristic ridges
/// and islands of a real porkchop plot come from the two branches of the
/// transfer -- Type I below a half revolution and Type II above -- meeting
/// where the transfer angle passes `pi` and the solution degenerates.
///
/// # Errors
/// Returns an error for an empty grid, a non-positive gravitational
/// parameter, or more than a million cells.
pub fn porkchop_data(
    departures: &[Ephemeris],
    arrivals: &[Ephemeris],
    mu: f64,
    prograde: bool,
) -> Result<Vec<Vec<Option<f64>>>, GeomError> {
    if departures.is_empty() || arrivals.is_empty() || !(mu > 0.0) || !mu.is_finite() {
        return Err(GeomError::InvalidArgument("porkchop_data: bad grid or parameter"));
    }
    if departures.len().saturating_mul(arrivals.len()) > 1_000_000 {
        return Err(GeomError::InvalidArgument("porkchop_data: that grid is too large"));
    }
    Ok(departures
        .iter()
        .map(|(t0, r0, v0)| {
            arrivals
                .iter()
                .map(|(t1, r1, _)| {
                    let tof = t1 - t0;
                    if !(tof > 0.0) {
                        return None;
                    }
                    let (depart, _) = lambert_universal(*r0, *r1, tof, mu, prograde).ok()?;
                    let excess = Vec3::new(depart.x - v0.x, depart.y - v0.y, depart.z - v0.z);
                    Some(excess.magnitude_squared())
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astrophysics::kepler::{orbit_period, propagate_kepler, state_from_elements};
    use crate::astrophysics::orbital_elements::OrbitalElements;
    use crate::monte_carlo::Rng;

    const MU: f64 = 398_600.441_8;
    const TAU: f64 = std::f64::consts::TAU;
    const PI: f64 = std::f64::consts::PI;

    fn distance(a: Vec3, b: Vec3) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
    }

    #[test]
    fn the_stumpff_functions_are_continuous_where_the_series_takes_over() {
        // The switch from series to closed form at |z| = 0.1 must not be
        // visible. A jump there would put a kink in the flight time and
        // send the bisection to the wrong root. The step across the
        // boundary is 2e-12, so the slope contributes under 1e-13 and
        // anything larger is a genuine discontinuity.
        for boundary in [0.1f64, -0.1] {
            let inside = stumpff_c(boundary - boundary.signum() * 1e-12);
            let outside = stumpff_c(boundary + boundary.signum() * 1e-12);
            assert!(
                (inside - outside).abs() < 1e-12,
                "C jumped at {boundary}: {inside} against {outside}"
            );
            let inside = stumpff_s(boundary - boundary.signum() * 1e-12);
            let outside = stumpff_s(boundary + boundary.signum() * 1e-12);
            assert!(
                (inside - outside).abs() < 1e-12,
                "S jumped at {boundary}: {inside} against {outside}"
            );
        }
        // The values at the origin are the limits, exactly.
        assert!((stumpff_c(0.0) - 0.5).abs() < 1e-17);
        assert!((stumpff_s(0.0) - 1.0 / 6.0).abs() < 1e-17);
        // And both agree with their defining series far from it, where
        // the implementation uses the closed forms instead.
        for z in [-40.0f64, -5.0, -0.5, 0.5, 5.0, 39.0] {
            let series = |first: f64, step: fn(usize) -> f64| {
                let mut term = first;
                let mut total = term;
                for k in 1..40 {
                    term *= -z / step(k);
                    total += term;
                }
                total
            };
            let c = series(0.5, |k| (2 * k + 1) as f64 * (2 * k + 2) as f64);
            assert!(
                (stumpff_c(z) - c).abs() < 1e-10 * stumpff_c(z).abs().max(1.0),
                "C({z}) was {} against the series' {c}",
                stumpff_c(z)
            );
            let sn = series(1.0 / 6.0, |k| (2 * k + 2) as f64 * (2 * k + 3) as f64);
            assert!(
                (stumpff_s(z) - sn).abs() < 1e-10 * stumpff_s(z).abs().max(1.0),
                "S({z}) was {} against the series' {sn}",
                stumpff_s(z)
            );
        }
    }

    #[test]
    fn the_positive_branch_survives_the_single_revolution_boundary() {
        // At z = 4 pi^2 the cosine is within an ulp of one, and `1 - cos u`
        // keeps no digits: it can return zero or go negative, which makes
        // the flight time infinite and the bracket unusable. The half-angle
        // form stays positive and accurate.
        let ceiling = 4.0 * PI * PI;
        for offset in [1e-4f64, 1e-6, 1e-8, 1e-10] {
            let c = stumpff_c(ceiling - offset);
            assert!(c > 0.0 && c.is_finite(), "C was {c} at {offset} below the boundary");
            // It vanishes quadratically in the distance from the boundary.
            let expected = offset * offset / (128.0 * PI.powi(4));
            assert!(
                (c / expected - 1.0).abs() < 1e-3,
                "C was {c} against the expected {expected}"
            );
        }
        assert!(stumpff_s(ceiling - 1e-8) > 0.0);
    }

    #[test]
    fn lambert_reproduces_the_textbook_transfer() {
        // Vallado's example 7-5: two positions an hour and a quarter
        // apart, with published velocities.
        let r1 = Vec3::new(15945.34, 0.0, 0.0);
        let r2 = Vec3::new(12_214.838_99, 10_249.467_31, 0.0);
        let (v1, v2) = lambert_universal(r1, r2, 76.0 * 60.0, MU, true).unwrap();
        assert!((v1.x - 2.058_913).abs() < 1e-5, "v1.x was {}", v1.x);
        assert!((v1.y - 2.915_965).abs() < 1e-5, "v1.y was {}", v1.y);
        assert!(v1.z.abs() < 1e-12, "the transfer left the plane");
        assert!((v2.x - -3.451_569).abs() < 1e-4, "v2.x was {}", v2.x);
        assert!((v2.y - 0.910_301).abs() < 1e-4, "v2.y was {}", v2.y);
        assert!(v2.z.abs() < 1e-12);
    }

    #[test]
    fn a_lambert_solution_propagates_to_the_target_it_was_solved_for() {
        // The definition, checked by an independent propagator: fly the
        // departure velocity for the flight time and land on the arrival
        // position. Nothing in the solver knows about propagate_kepler.
        let mut rng = Rng::new(0x0A57_3001);
        for _ in 0..300 {
            let elements = OrbitalElements {
                semi_major_axis: 8000.0 + 30000.0 * rng.next_f64(),
                eccentricity: 0.7 * rng.next_f64(),
                inclination: 0.1 + 2.9 * rng.next_f64(),
                longitude_ascending_node: TAU * rng.next_f64(),
                argument_periapsis: TAU * rng.next_f64(),
                true_anomaly: TAU * rng.next_f64(),
            };
            let (r_a, v_a) = state_from_elements(&elements, MU).unwrap();
            let period = orbit_period(elements.semi_major_axis, MU).unwrap();
            let tof = period * (0.05 + 0.5 * rng.next_f64());
            let (r_b, v_b) = propagate_kepler(r_a, v_a, tof, MU).unwrap();
            let prograde = r_a.cross(&v_a).z >= 0.0;
            let (s1, s2) = lambert_universal(r_a, r_b, tof, MU, prograde).unwrap();
            // It recovers the very velocities that generated the arc.
            assert!(
                distance(s1, v_a) < 1e-9 * v_a.magnitude(),
                "the departure velocity was off by {}",
                distance(s1, v_a)
            );
            assert!(distance(s2, v_b) < 1e-9 * v_b.magnitude());
            // And flying the solution lands on the target.
            let (landed, _) = propagate_kepler(r_a, s1, tof, MU).unwrap();
            assert!(distance(landed, r_b) < 1e-7 * r_b.magnitude());
        }
    }

    #[test]
    fn there_is_a_cheapest_flight_time_and_it_is_not_at_either_end() {
        // Both hurrying and dawdling cost speed. Rushing needs a
        // hyperbola; taking too long needs a large slow ellipse that
        // arrives from the wrong direction. In between sits the
        // minimum-energy transfer, and the minimum being *interior* is
        // the whole reason it has a name.
        let r_a = Vec3::new(10000.0, 0.0, 0.0);
        let r_b = Vec3::new(0.0, 12000.0, 0.0);
        let times: Vec<f64> =
            (0..60).map(|k| 300.0 * (1.0 + 0.12f64).powi(k)).take_while(|t| *t < 3e5).collect();
        let speeds: Vec<f64> = times
            .iter()
            .map(|t| lambert_universal(r_a, r_b, *t, MU, true).unwrap().0.magnitude())
            .collect();
        let cheapest = speeds
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        assert!(cheapest > 0 && cheapest < speeds.len() - 1, "the minimum was at an end");
        // Falling before it and rising after.
        for pair in speeds[..=cheapest].windows(2) {
            assert!(pair[1] <= pair[0], "the cost rose before the minimum");
        }
        for pair in speeds[cheapest..].windows(2) {
            assert!(pair[1] >= pair[0], "the cost fell after the minimum");
        }

        // And the fast end really does leave the ellipse.
        let (fast, _) = lambert_universal(r_a, r_b, 200.0, MU, true).unwrap();
        assert!(0.5 * fast.magnitude_squared() - MU / r_a.magnitude() > 0.0, "not hyperbolic");
        let (slow, _) = lambert_universal(r_a, r_b, 20000.0, MU, true).unwrap();
        assert!(0.5 * slow.magnitude_squared() - MU / r_a.magnitude() < 0.0, "not elliptic");
    }

    #[test]
    fn going_the_long_way_round_is_a_different_orbit() {
        // The two radii do not determine the transfer: which side of the
        // central body it passes is the caller's choice, and the two
        // orbits differ in energy and in the direction of travel.
        let r_a = Vec3::new(10000.0, 0.0, 0.0);
        let r_b = Vec3::new(0.0, 15000.0, 0.0);
        let (short, _) = lambert_universal(r_a, r_b, 5000.0, MU, true).unwrap();
        let (long, _) = lambert_universal(r_a, r_b, 5000.0, MU, false).unwrap();
        assert!(
            distance(short, long) > 0.01,
            "the two directions gave the same velocity: {short:?} and {long:?}"
        );
        // The prograde solution circulates the way the cross product says.
        assert!(r_a.cross(&short).z > 0.0, "the prograde transfer went the wrong way");
        assert!(r_a.cross(&long).z < 0.0, "the retrograde transfer went the wrong way");
    }

    #[test]
    fn a_degenerate_geometry_is_refused_rather_than_guessed_at() {
        let r = Vec3::new(10000.0, 0.0, 0.0);
        // The same point: the transfer angle is zero and every orbit
        // through it qualifies.
        assert!(lambert_universal(r, r, 1000.0, MU, true).is_err());
        // Exactly opposite: the plane is undefined, since the two radii
        // are collinear and any plane containing the line will do.
        let opposite = Vec3::new(-12000.0, 0.0, 0.0);
        assert!(lambert_universal(r, opposite, 5000.0, MU, true).is_err());
        // A hair off it works again.
        let nearly = Vec3::new(-12000.0, 1.0, 0.0);
        assert!(lambert_universal(r, nearly, 5000.0, MU, true).is_ok());

        assert!(lambert_universal(r, Vec3::new(0.0, 12000.0, 0.0), 0.0, MU, true).is_err());
        assert!(lambert_universal(r, Vec3::new(0.0, 12000.0, 0.0), -100.0, MU, true).is_err());
        assert!(lambert_universal(r, Vec3::new(0.0, 12000.0, 0.0), 1000.0, 0.0, true).is_err());
        let origin = Vec3::new(0.0, 0.0, 0.0);
        assert!(lambert_universal(origin, r, 1000.0, MU, true).is_err());
    }

    #[test]
    fn a_porkchop_grid_is_cheapest_near_the_transfer_it_wants() {
        // Every cell is a departure energy; the pairs with no positive
        // flight time are absent rather than zero, and the grid has a
        // minimum where the geometry suits the duration.
        let departures: Vec<Ephemeris> = (0..5)
            .map(|i| (i as f64 * 600.0, Vec3::new(10000.0, 0.0, 0.0), Vec3::new(0.0, 6.3, 0.0)))
            .collect();
        let arrivals: Vec<Ephemeris> = (0..5)
            .map(|j| {
                (3000.0 + j as f64 * 900.0, Vec3::new(0.0, 15000.0, 0.0), Vec3::new(-5.1, 0.0, 0.0))
            })
            .collect();
        let grid = porkchop_data(&departures, &arrivals, MU, true).unwrap();
        assert_eq!(grid.len(), 5);
        assert!(grid.iter().all(|row| row.len() == 5));
        let mut best = f64::INFINITY;
        for row in &grid {
            for c3 in row.iter().flatten() {
                assert!(*c3 >= 0.0 && c3.is_finite(), "a characteristic energy was {c3}");
                best = best.min(*c3);
            }
        }
        assert!(best.is_finite() && best < 5.0, "the best departure cost {best}");

        // A grid whose arrivals all precede its departures has no cells.
        let backwards: Vec<Ephemeris> =
            vec![(0.0, Vec3::new(0.0, 15000.0, 0.0), Vec3::new(-5.1, 0.0, 0.0))];
        let empty = porkchop_data(&departures, &backwards, MU, true).unwrap();
        assert!(empty.iter().all(|row| row.iter().all(Option::is_none)));

        assert!(porkchop_data(&[], &arrivals, MU, true).is_err());
        assert!(porkchop_data(&departures, &[], MU, true).is_err());
        assert!(porkchop_data(&departures, &arrivals, 0.0, true).is_err());
    }
}
