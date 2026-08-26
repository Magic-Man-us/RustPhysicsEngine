//! Orbital manoeuvres: combined burns, patched conics, gravity assists
//! and the perturbation that dominates low orbits.
//!
//! # What lives elsewhere
//!
//! The impulsive transfers themselves are already in
//! [`crate::propulsion`]: `hohmann_delta_v`, `hohmann_transfer_time`,
//! `bi_elliptic_delta_v`, `delta_v_plane_change`, `tsiolkovsky_delta_v`
//! and `delta_v_staged`. The Roche limit is in
//! [`crate::astrophysics::tidal`] and the Hill radius in
//! [`crate::astrophysics::lagrange`]. This module adds what those do not
//! cover, and reuses rather than repeats them.
//!
//! # Why delta-v is the currency
//!
//! Every manoeuvre here is priced in velocity change rather than in fuel,
//! because the conversion between them is exponential: Tsiolkovsky's
//! equation says the mass ratio is `e^(dv/v_e)`, so a mission's delta-v
//! budget is a linear quantity that adds up while its mass is not. Ten
//! per cent more delta-v is not ten per cent more spacecraft.
//!
//! The other consequence is the Oberth effect. A burn's *energy* gain is
//! `v dv`, proportional to the speed you already have, so the same
//! delta-v spent deep in a gravity well buys far more energy than the
//! same delta-v spent far from it. That is why escape burns are made at
//! periapsis and why a flyby is worth planning around.

use crate::error::GeomError;
use crate::math::Vec3;

/// The delta-v of a combined speed change and plane change, by the law of
/// cosines.
///
/// `sqrt(v1^2 + v2^2 - 2 v1 v2 cos(di))`. Doing both at once is always
/// cheaper than doing them one after the other, because the two vector
/// changes partly cancel -- the triangle inequality, applied to velocity.
/// The saving is largest when the plane change is large, which is why an
/// inclination change is combined with an apoapsis burn wherever the
/// mission allows.
///
/// # Errors
/// Returns an error for a negative speed or a non-finite input.
pub fn combined_maneuver(v1: f64, v2: f64, plane_change: f64) -> Result<f64, GeomError> {
    if v1 < 0.0 || v2 < 0.0 || ![v1, v2, plane_change].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("combined_maneuver: bad speeds or angle"));
    }
    Ok((v1 * v1 + v2 * v2 - 2.0 * v1 * v2 * plane_change.cos()).max(0.0).sqrt())
}

/// The radius of a body's sphere of influence:
/// `a (m_body / m_primary)^(2/5)`.
///
/// Inside it the body's gravity dominates the primary's for the purposes
/// of a patched-conic approximation, and outside it does not. The
/// two-fifths power is not the equal-force radius, which would be a
/// square root: it comes from comparing the *perturbing* accelerations
/// rather than the direct ones, and it is the boundary at which
/// switching which body you orbit makes the smaller error.
///
/// The sphere is a fiction. Gravity has no boundary, and a real
/// trajectory feels both bodies throughout; the patched conic is an
/// approximation whose error is largest exactly at the crossing, where
/// the neglected body's pull is at its relative peak.
///
/// # Errors
/// Returns an error for a non-positive distance or mass, or a body more
/// massive than its primary.
pub fn sphere_of_influence(
    distance: f64,
    body_mass: f64,
    primary_mass: f64,
) -> Result<f64, GeomError> {
    if !(distance > 0.0) || !(body_mass > 0.0) || !(primary_mass > 0.0) {
        return Err(GeomError::InvalidArgument("sphere_of_influence: bad distance or mass"));
    }
    if !distance.is_finite() || !body_mass.is_finite() || !primary_mass.is_finite() {
        return Err(GeomError::InvalidArgument("sphere_of_influence: an input is not finite"));
    }
    if body_mass >= primary_mass {
        return Err(GeomError::InvalidArgument(
            "the body is not lighter than its primary, so it has no sphere of influence within it",
        ));
    }
    Ok(distance * (body_mass / primary_mass).powf(0.4))
}

/// The delta-v to leave a circular parking orbit on a hyperbola with the
/// given excess speed: `sqrt(v_infinity^2 + 2 mu / r) - sqrt(mu / r)`.
///
/// The first term is the speed needed at radius `r` to arrive at infinity
/// still moving at `v_infinity`; the second is what a circular orbit
/// already provides. The gap is small compared with either, which is the
/// Oberth effect in its most practical form: escaping from low orbit
/// costs about 0.41 of the circular speed, and the deeper the parking
/// orbit the smaller that fraction becomes.
///
/// # Errors
/// Returns an error for a non-positive radius or gravitational parameter,
/// a negative excess speed, or a non-finite input.
pub fn patched_conic_escape(
    parking_radius: f64,
    mu: f64,
    v_infinity: f64,
) -> Result<f64, GeomError> {
    if !(parking_radius > 0.0) || !(mu > 0.0) || v_infinity < 0.0 {
        return Err(GeomError::InvalidArgument("patched_conic_escape: bad parameters"));
    }
    if ![parking_radius, mu, v_infinity].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("patched_conic_escape: an input is not finite"));
    }
    let circular = (mu / parking_radius).sqrt();
    Ok((v_infinity * v_infinity + 2.0 * mu / parking_radius).sqrt() - circular)
}

/// The turn angle of a hyperbolic flyby:
/// `2 arcsin(1 / (1 + r_p v_inf^2 / mu))`.
///
/// A gravity assist changes the direction of the excess velocity, not its
/// magnitude -- in the *planet's* frame the spacecraft arrives and leaves
/// at the same speed. The gain is in the sun's frame, where rotating the
/// excess velocity vector adds or subtracts from the planet's orbital
/// motion, and the planet loses exactly as much momentum as the
/// spacecraft gains.
///
/// The turn is largest for a slow approach and a close pass. A fast
/// spacecraft is barely deflected, which is why an assist buys less the
/// more energy you already have.
///
/// # Errors
/// Returns an error for a non-positive periapsis, gravitational parameter
/// or excess speed, or a non-finite input.
pub fn gravity_assist_deflection(
    v_infinity: f64,
    periapsis: f64,
    mu: f64,
) -> Result<f64, GeomError> {
    if !(v_infinity > 0.0) || !(periapsis > 0.0) || !(mu > 0.0) {
        return Err(GeomError::InvalidArgument("gravity_assist_deflection: bad parameters"));
    }
    if ![v_infinity, periapsis, mu].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("gravity_assist_deflection: an input is not finite"));
    }
    // The eccentricity of the flyby hyperbola.
    let e = 1.0 + periapsis * v_infinity * v_infinity / mu;
    Ok(2.0 * (1.0 / e).asin())
}

/// The speed after an impulsive burn of `delta_v` made at radius `r`,
/// through the energy it buys.
///
/// The point of the function is the comparison it makes possible: the same
/// delta-v spent at two radii leaves the craft with different energies,
/// and the difference is `v dv` -- large where `v` is large, which is deep
/// in the well. Burning at periapsis rather than apoapsis can double the
/// escape energy for the same fuel.
///
/// # Errors
/// Returns an error for a non-positive radius or gravitational parameter,
/// a negative speed, or a non-finite input.
pub fn oberth_effect_dv(
    speed: f64,
    delta_v: f64,
    radius: f64,
    mu: f64,
) -> Result<f64, GeomError> {
    if speed < 0.0 || !(radius > 0.0) || !(mu > 0.0) {
        return Err(GeomError::InvalidArgument("oberth_effect_dv: bad parameters"));
    }
    if ![speed, delta_v, radius, mu].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("oberth_effect_dv: an input is not finite"));
    }
    let after = speed + delta_v;
    if after < 0.0 {
        return Err(GeomError::Degenerate("the burn reverses the motion past a standstill"));
    }
    Ok(0.5 * after * after - mu / radius)
}

/// The nodal regression rate from the Earth's oblateness, in radians per
/// second.
///
/// `-3/2 n J2 (R/p)^2 cos(i)`, with `p = a(1 - e^2)` and `n` the mean
/// motion. The `cos i` is what makes the whole thing useful: the drift is
/// westward for a prograde orbit, zero at exactly ninety degrees, and
/// eastward beyond it. A retrograde orbit at the right inclination
/// therefore drifts eastward at precisely the rate the Earth goes round
/// the sun -- see [`sun_synchronous_inclination`].
///
/// J2 dominates every other perturbation in low orbit by three orders of
/// magnitude, which is why a first-order treatment of it is worth more
/// than a careful treatment of anything else.
///
/// # Errors
/// Returns an error for a non-positive semi-major axis, body radius or
/// gravitational parameter, an eccentricity outside `[0, 1)`, or a
/// non-finite input.
pub fn j2_raan_drift(
    a: f64,
    e: f64,
    inclination: f64,
    j2: f64,
    body_radius: f64,
    mu: f64,
) -> Result<f64, GeomError> {
    if !(a > 0.0) || !(body_radius > 0.0) || !(mu > 0.0) || !(0.0..1.0).contains(&e) {
        return Err(GeomError::InvalidArgument("j2_raan_drift: bad orbit or body"));
    }
    if ![a, e, inclination, j2, body_radius, mu].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("j2_raan_drift: an input is not finite"));
    }
    let p = a * (1.0 - e * e);
    let n = (mu / (a * a * a)).sqrt();
    Ok(-1.5 * n * j2 * (body_radius / p).powi(2) * inclination.cos())
}

/// The inclination at which J2 makes an orbit sun-synchronous.
///
/// The node must drift eastward by one turn a year, which is
/// `1.991e-7 rad/s`. Solving [`j2_raan_drift`] for the inclination gives
/// a value just past ninety degrees -- about 98 degrees for a low orbit --
/// and it must be retrograde, since a prograde orbit's node drifts the
/// wrong way.
///
/// The orbit is sun-synchronous in the sense that it crosses the equator
/// at the same local solar time every pass, which is what makes imaging
/// comparable between days. It says nothing about lighting at high
/// latitudes, where the geometry differs.
///
/// # Errors
/// Returns an error for a non-positive semi-major axis, body radius or
/// gravitational parameter, an eccentricity outside `[0, 1)`, or an orbit
/// for which no inclination gives the required drift -- which happens
/// when the orbit is too high for J2 to turn it fast enough.
pub fn sun_synchronous_inclination(
    a: f64,
    e: f64,
    j2: f64,
    body_radius: f64,
    mu: f64,
    drift_per_second: f64,
) -> Result<f64, GeomError> {
    if !(a > 0.0) || !(body_radius > 0.0) || !(mu > 0.0) || !(0.0..1.0).contains(&e) {
        return Err(GeomError::InvalidArgument("sun_synchronous_inclination: bad orbit or body"));
    }
    if ![a, e, j2, body_radius, mu, drift_per_second].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("sun_synchronous_inclination: bad input"));
    }
    let p = a * (1.0 - e * e);
    let n = (mu / (a * a * a)).sqrt();
    let coefficient = -1.5 * n * j2 * (body_radius / p).powi(2);
    if coefficient == 0.0 {
        return Err(GeomError::Degenerate("without oblateness there is no nodal drift to use"));
    }
    let cosine = drift_per_second / coefficient;
    if !(-1.0..=1.0).contains(&cosine) {
        return Err(GeomError::Degenerate(
            "no inclination gives that drift: the orbit is too high for J2 to turn it",
        ));
    }
    Ok(cosine.acos())
}

/// The ground track of an orbit: `(longitude, latitude)` in radians at
/// each sample, accounting for the body turning underneath.
///
/// The longitude drift per orbit is what makes a track a spiral rather
/// than a closed curve: the body turns by `rotation_rate * period` while
/// the orbit plane stays put, so each pass crosses the equator further
/// west. A track closes only when the period is a rational fraction of
/// the rotation, which is what a repeat-ground-track orbit is designed
/// for.
///
/// The latitude never exceeds the inclination, and reaches it exactly
/// twice per orbit. That bound is the reason a polar orbit is needed to
/// see the poles at all.
///
/// # Errors
/// Returns an error for a bad state, a non-positive gravitational
/// parameter, no samples, more than a million, or a propagation failure.
pub fn ground_track(
    r0: Vec3,
    v0: Vec3,
    mu: f64,
    rotation_rate: f64,
    duration: f64,
    samples: usize,
) -> Result<Vec<(f64, f64)>, GeomError> {
    if !(mu > 0.0) || !mu.is_finite() || !rotation_rate.is_finite() || !(duration > 0.0) {
        return Err(GeomError::InvalidArgument("ground_track: bad parameters"));
    }
    if samples == 0 || samples > 1_000_000 {
        return Err(GeomError::InvalidArgument("ground_track: bad sample count"));
    }
    let mut out = Vec::with_capacity(samples);
    for step in 0..samples {
        let t = duration * step as f64 / (samples - 1).max(1) as f64;
        let (r, _) = crate::astrophysics::kepler::propagate_kepler(r0, v0, t, mu)?;
        let magnitude = r.magnitude();
        if !(magnitude > 0.0) {
            return Err(GeomError::Degenerate("the track passed through the centre"));
        }
        let latitude = (r.z / magnitude).clamp(-1.0, 1.0).asin();
        // Subtract the body's rotation to get the longitude beneath.
        let inertial = r.y.atan2(r.x);
        let longitude = wrap_pi(inertial - rotation_rate * t);
        out.push((longitude, latitude));
    }
    Ok(out)
}

/// Wraps an angle to `(-pi, pi]`.
fn wrap_pi(angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let wrapped = (angle + std::f64::consts::PI).rem_euclid(tau) - std::f64::consts::PI;
    if wrapped <= -std::f64::consts::PI {
        wrapped + tau
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astrophysics::kepler::{orbit_period, state_from_elements, vis_viva};
    use crate::astrophysics::orbital_elements::OrbitalElements;

    const MU: f64 = 398_600.441_8;
    /// Earth's equatorial radius, km.
    const RE: f64 = 6378.137;
    /// Earth's second zonal harmonic.
    const J2: f64 = 1.082_626_68e-3;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;

    #[test]
    fn doing_both_at_once_beats_doing_them_one_after_the_other() {
        // The triangle inequality applied to velocity: changing speed and
        // direction together is never dearer than in sequence, and the
        // saving grows with the plane change.
        for (v1, v2) in [(7.7f64, 7.7f64), (3.07, 1.6), (10.0, 4.0)] {
            for degrees in [1.0f64, 10.0, 30.0, 60.0, 90.0, 150.0] {
                let angle = degrees.to_radians();
                let together = combined_maneuver(v1, v2, angle).unwrap();
                // In sequence: change speed first, then rotate at the new
                // speed. How the saving varies with the angle depends on
                // which order the separate burns are taken in, so only the
                // inequality is asserted -- that one is a theorem.
                let separate = (v2 - v1).abs() + 2.0 * v2 * (0.5 * angle).sin();
                assert!(
                    together <= separate + 1e-12,
                    "at {degrees} degrees the combined burn cost {together} against {separate}"
                );
            }
            // At an equal speed the combined burn is a pure rotation, and
            // then the two orderings agree exactly.
            if (v1 - v2).abs() < 1e-12 {
                let angle = 0.7;
                assert!(
                    (combined_maneuver(v1, v2, angle).unwrap() - 2.0 * v2 * (0.5 * angle).sin())
                        .abs()
                        < 1e-12
                );
            }
            // With no plane change it is just the speed difference.
            assert!((combined_maneuver(v1, v2, 0.0).unwrap() - (v2 - v1).abs()).abs() < 1e-12);
            // Reversing direction entirely costs the sum.
            assert!((combined_maneuver(v1, v2, PI).unwrap() - (v1 + v2)).abs() < 1e-12);
        }
        assert!(combined_maneuver(-1.0, 5.0, 0.1).is_err());
        assert!(combined_maneuver(1.0, f64::NAN, 0.1).is_err());
    }

    #[test]
    fn a_sphere_of_influence_scales_as_the_two_fifths_power_of_the_mass_ratio() {
        // Not the square root, which is where the forces balance. The
        // two-fifths comes from comparing perturbing accelerations, and it
        // puts the boundary much closer in.
        let earth_mass = 5.972e24;
        let sun_mass = 1.989e30;
        let au = 1.495_978_707e8;
        let soi = sphere_of_influence(au, earth_mass, sun_mass).unwrap();
        // The textbook value is about 924,000 km.
        assert!((soi - 924_000.0).abs() < 15_000.0, "it came out at {soi} km");
        // The sphere of influence reaches well *beyond* where the two
        // pulls balance, which is at the square-root radius rather than
        // the two-fifths one. A smaller exponent on a ratio below one
        // gives a larger answer, and the difference is a factor of three
        // and a half: 259,000 km against 924,000. What decides the
        // patched-conic boundary is the ratio of *perturbing*
        // accelerations, not of direct ones.
        let balance = au * (earth_mass / sun_mass).sqrt();
        assert!((balance - 259_000.0).abs() < 5_000.0, "the balance radius was {balance}");
        assert!(soi > 3.0 * balance, "the sphere of influence was only {soi}");

        // The scaling law itself.
        let base = sphere_of_influence(1.0, 1.0, 1000.0).unwrap();
        for factor in [4.0f64, 32.0, 100.0] {
            let scaled = sphere_of_influence(1.0, factor, 1000.0).unwrap();
            assert!(
                (scaled / base - factor.powf(0.4)).abs() < 1e-12,
                "scaling the mass by {factor} scaled the radius by {}",
                scaled / base
            );
        }
        // And it is linear in the distance.
        assert!(
            (sphere_of_influence(7.0, 1.0, 1000.0).unwrap() / base - 7.0).abs() < 1e-12
        );
        assert!(sphere_of_influence(au, sun_mass, earth_mass).is_err());
        assert!(sphere_of_influence(0.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn escaping_from_low_orbit_costs_a_fraction_of_the_speed_already_there() {
        // The classic 0.4142: escape speed is sqrt(2) times circular, so
        // leaving with nothing to spare costs sqrt(2) - 1 of it. That is
        // the Oberth effect stated as a number.
        let radius = 6678.0;
        let circular = (MU / radius).sqrt();
        let bare = patched_conic_escape(radius, MU, 0.0).unwrap();
        assert!(
            (bare / circular - (2.0f64.sqrt() - 1.0)).abs() < 1e-12,
            "the escape burn was {} of circular speed",
            bare / circular
        );
        // Arriving somewhere still moving costs more, but sub-linearly:
        // three km/s of excess costs well under three km/s of burn.
        let with_excess = patched_conic_escape(radius, MU, 3.0).unwrap();
        assert!(with_excess > bare);
        assert!(
            with_excess - bare < 1.0,
            "three km/s of excess cost {} extra",
            with_excess - bare
        );
        // And the deeper the parking orbit, the better the bargain.
        let high = patched_conic_escape(42_164.0, MU, 3.0).unwrap();
        let low = patched_conic_escape(6678.0, MU, 3.0).unwrap();
        let high_share = (high - patched_conic_escape(42_164.0, MU, 0.0).unwrap()) / 3.0;
        let low_share = (low - bare) / 3.0;
        assert!(low_share < high_share, "the low orbit was not the better place to burn");
        assert!(patched_conic_escape(0.0, MU, 1.0).is_err());
        assert!(patched_conic_escape(7000.0, MU, -1.0).is_err());
    }

    #[test]
    fn a_flyby_turns_more_the_slower_and_closer_it_is() {
        // The deflection depends on the flyby hyperbola's eccentricity,
        // which rises with both the speed and the miss distance. A fast
        // spacecraft is barely bent, which is why an assist buys less the
        // more energy you already have.
        let mu_jupiter = 1.266_865_34e8;
        let radius = 71_492.0;
        let mut previous = PI;
        for excess in [1.0f64, 3.0, 6.0, 12.0, 30.0] {
            let turn = gravity_assist_deflection(excess, 2.0 * radius, mu_jupiter).unwrap();
            assert!(turn > 0.0 && turn < PI, "the turn was {turn}");
            assert!(turn < previous, "a faster pass turned further at {excess} km/s");
            previous = turn;
        }
        let mut previous = 0.0;
        for altitude in [20.0f64, 5.0, 2.0, 1.1] {
            let turn = gravity_assist_deflection(6.0, altitude * radius, mu_jupiter).unwrap();
            assert!(turn > previous, "a closer pass turned less at {altitude} radii");
            previous = turn;
        }
        // A grazing pass by a heavy planet can reverse the approach
        // almost entirely.
        let extreme = gravity_assist_deflection(0.5, 1.01 * radius, mu_jupiter).unwrap();
        assert!(extreme > 2.5, "the extreme flyby only turned {extreme} rad");
        assert!(gravity_assist_deflection(0.0, radius, mu_jupiter).is_err());
        assert!(gravity_assist_deflection(6.0, 0.0, mu_jupiter).is_err());
    }

    #[test]
    fn the_same_burn_buys_more_energy_where_the_craft_is_already_fast() {
        // The Oberth effect in its cleanest form: energy gain is v dv, so
        // spending the same delta-v at periapsis rather than apoapsis of
        // the same ellipse is worth several times as much.
        let (a, e) = (24_000.0f64, 0.7f64);
        let periapsis = a * (1.0 - e);
        let apoapsis = a * (1.0 + e);
        let fast = vis_viva(periapsis, a, MU).unwrap();
        let slow = vis_viva(apoapsis, a, MU).unwrap();
        let before = -MU / (2.0 * a);
        let burn = 0.5;
        let at_periapsis = oberth_effect_dv(fast, burn, periapsis, MU).unwrap() - before;
        let at_apoapsis = oberth_effect_dv(slow, burn, apoapsis, MU).unwrap() - before;
        assert!(at_periapsis > 0.0 && at_apoapsis > 0.0);
        assert!(
            at_periapsis > 4.0 * at_apoapsis,
            "periapsis bought {at_periapsis} against apoapsis' {at_apoapsis}"
        );
        // To first order the gain is exactly v dv.
        assert!((at_periapsis - fast * burn - 0.5 * burn * burn).abs() < 1e-9);
        // Burning nothing changes nothing.
        assert!((oberth_effect_dv(fast, 0.0, periapsis, MU).unwrap() - before).abs() < 1e-9);
        assert!(oberth_effect_dv(1.0, -2.0, 7000.0, MU).is_err());
        assert!(oberth_effect_dv(1.0, 1.0, 0.0, MU).is_err());
    }

    #[test]
    fn the_nodal_drift_is_westward_prograde_and_vanishes_at_the_pole() {
        // The cos i factor is the whole content: negative below ninety
        // degrees, zero at it, positive above.
        let a = RE + 500.0;
        let prograde = j2_raan_drift(a, 0.0, 45f64.to_radians(), J2, RE, MU).unwrap();
        assert!(prograde < 0.0, "a prograde orbit drifted eastward: {prograde}");
        let polar = j2_raan_drift(a, 0.0, PI / 2.0, J2, RE, MU).unwrap();
        assert!(polar.abs() < 1e-18, "a polar orbit drifted by {polar}");
        let retrograde = j2_raan_drift(a, 0.0, 120f64.to_radians(), J2, RE, MU).unwrap();
        assert!(retrograde > 0.0, "a retrograde orbit drifted westward: {retrograde}");
        // Equatorial is the fastest drift there is, at each altitude.
        let equatorial = j2_raan_drift(a, 0.0, 0.0, J2, RE, MU).unwrap();
        assert!(equatorial < prograde, "the equatorial drift was not the largest");
        // A 500 km orbit at 45 degrees drifts about five degrees a day.
        let per_day = prograde * 86_400.0;
        assert!(
            (per_day.to_degrees() + 5.4).abs() < 0.2,
            "it drifted {} degrees a day",
            per_day.to_degrees()
        );
        // Higher orbits drift more slowly, since J2 falls off fast.
        let high = j2_raan_drift(RE + 20_000.0, 0.0, 45f64.to_radians(), J2, RE, MU).unwrap();
        assert!(high.abs() < 0.1 * prograde.abs());
        assert!(j2_raan_drift(a, 1.0, 0.5, J2, RE, MU).is_err());
        assert!(j2_raan_drift(0.0, 0.0, 0.5, J2, RE, MU).is_err());
    }

    #[test]
    fn a_sun_synchronous_orbit_is_retrograde_and_near_ninety_eight_degrees() {
        // One turn a year eastward: 1.991e-7 rad/s. Prograde orbits drift
        // the wrong way, so the answer must be past ninety degrees.
        let yearly = TAU / 365.242_19 / 86_400.0;
        for altitude in [400.0f64, 600.0, 800.0] {
            let a = RE + altitude;
            let inclination =
                sun_synchronous_inclination(a, 0.0, J2, RE, MU, yearly).unwrap();
            let degrees = inclination.to_degrees();
            assert!(
                (97.0..100.5).contains(&degrees),
                "at {altitude} km it came out at {degrees} degrees"
            );
            // And it really does produce the drift asked for.
            let drift = j2_raan_drift(a, 0.0, inclination, J2, RE, MU).unwrap();
            assert!(
                (drift - yearly).abs() < 1e-15,
                "the drift was {drift} against the required {yearly}"
            );
        }
        // Higher orbits need more inclination, since J2 has less grip.
        let low = sun_synchronous_inclination(RE + 300.0, 0.0, J2, RE, MU, yearly).unwrap();
        let high = sun_synchronous_inclination(RE + 1200.0, 0.0, J2, RE, MU, yearly).unwrap();
        assert!(high > low, "the higher orbit needed less inclination");
        // Far enough out and no inclination will do.
        assert!(sun_synchronous_inclination(RE + 40_000.0, 0.0, J2, RE, MU, yearly).is_err());
        assert!(sun_synchronous_inclination(RE + 500.0, 0.0, 0.0, RE, MU, yearly).is_err());
    }

    #[test]
    fn a_ground_track_stays_within_its_inclination_and_walks_west() {
        // Two facts a track cannot escape: the latitude is bounded by the
        // inclination, and each pass crosses the equator further west by
        // the angle the body turned during one orbit.
        let inclination = 51.6f64.to_radians();
        let elements = OrbitalElements {
            semi_major_axis: RE + 400.0,
            eccentricity: 0.0,
            inclination,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let (r0, v0) = state_from_elements(&elements, MU).unwrap();
        let period = orbit_period(elements.semi_major_axis, MU).unwrap();
        let rotation = TAU / 86_164.0;
        // Three periods, not two: the run starts exactly on the
        // ascending node, so that crossing has no descending sample
        // before it to be detected by.
        let track = ground_track(r0, v0, MU, rotation, 3.0 * period, 3000).unwrap();
        assert_eq!(track.len(), 3000);
        let highest = track.iter().map(|(_, lat)| lat.abs()).fold(0.0f64, f64::max);
        assert!(
            highest <= inclination + 1e-9,
            "the track reached {} degrees on a {} degree orbit",
            highest.to_degrees(),
            inclination.to_degrees()
        );
        // And it gets there: the bound is attained, twice per orbit.
        assert!(
            (highest - inclination).abs() < 1e-3,
            "the track only reached {} degrees",
            highest.to_degrees()
        );
        for (longitude, latitude) in &track {
            assert!((-PI..=PI).contains(longitude));
            assert!(latitude.abs() <= PI / 2.0);
        }
        // Successive ascending nodes are one rotation-per-period apart.
        let expected_walk = rotation * period;
        let nodes: Vec<f64> = track
            .windows(2)
            .filter(|pair| pair[0].1 < 0.0 && pair[1].1 >= 0.0)
            .map(|pair| pair[1].0)
            .collect();
        assert!(nodes.len() >= 2, "the track did not cross the equator twice");
        let walked = wrap_pi(nodes[0] - nodes[1]);
        assert!(
            (walked - expected_walk).abs() < 0.02,
            "it walked {walked} rad against the expected {expected_walk}"
        );
        assert!(walked > 0.0, "the track did not move westward");

        assert!(ground_track(r0, v0, MU, rotation, 0.0, 100).is_err());
        assert!(ground_track(r0, v0, MU, rotation, 1000.0, 0).is_err());
        assert!(ground_track(r0, v0, 0.0, rotation, 1000.0, 100).is_err());
    }
}
