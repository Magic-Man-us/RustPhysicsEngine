//! Kepler's equation, anomaly conversions and two-body propagation.
//!
//! # Three anomalies and why there are three
//!
//! An orbit's position is described by an angle, and three different
//! angles are useful for different things. *True anomaly* is the physical
//! angle from periapsis to the body, seen from the focus -- it is what a
//! telescope measures and what converts directly to a position. *Mean
//! anomaly* advances uniformly in time, `M = n (t - t_p)`, so it is what
//! a clock gives. *Eccentric anomaly* is the intermediate angle on the
//! circumscribing circle that connects the two, and it exists because no
//! closed form connects the other two directly.
//!
//! Kepler's equation `M = E - e sin E` is the link, and it is
//! transcendental. Everything in orbital mechanics that looks like "where
//! will it be at time t" bottoms out in solving it, which is why five
//! centuries of work have gone into doing so quickly.
//!
//! # What is not here
//!
//! [`crate::astrophysics::orbital_elements`] already provides the element
//! set, the state-to-elements conversion and the geometric quantities
//! read off an orbit; this module adds the time dependence and the
//! inverse conversion, and does not repeat them.

use crate::astrophysics::orbital_elements::OrbitalElements;
use crate::error::GeomError;
use crate::math::Vec3;

/// Wraps an angle to `[0, 2 pi)`.
fn wrap_two_pi(angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let wrapped = angle % tau;
    if wrapped < 0.0 {
        wrapped + tau
    } else {
        wrapped
    }
}

/// Solves `M = E - e sin E` for the eccentric anomaly.
///
/// Newton's method from a seed that keeps it in the basin: for nearly
/// circular orbits `M` itself is already close, and for high
/// eccentricities the standard `M + e sin M` correction is not -- near
/// periapsis at `e = 0.99` the function is almost flat in `E` and a naive
/// seed sends the first step far outside `[0, 2 pi)`. The seed here is
/// Danby's, which is chosen to converge for every eccentricity below one.
///
/// Returns the anomaly in `[0, 2 pi]`.
///
/// # Errors
/// Returns an error for an eccentricity outside `[0, 1)`, a non-finite
/// mean anomaly or tolerance, a non-positive tolerance, or an iteration
/// that fails to converge.
pub fn kepler_solve_elliptic(mean_anomaly: f64, e: f64, tol: f64) -> Result<f64, GeomError> {
    if !(0.0..1.0).contains(&e) || !mean_anomaly.is_finite() || !(tol > 0.0) || !tol.is_finite() {
        return Err(GeomError::InvalidArgument("kepler_solve_elliptic: bad parameters"));
    }
    let m = wrap_two_pi(mean_anomaly);
    if e == 0.0 {
        return Ok(m);
    }
    // Danby's seed: exact at e = 0 and inside the basin of attraction for
    // every eccentricity below one.
    let mut anomaly = m + 0.85 * e * if m > std::f64::consts::PI { -1.0 } else { 1.0 };
    for _ in 0..100 {
        let (sin, cos) = anomaly.sin_cos();
        let residual = anomaly - e * sin - m;
        if residual.abs() < tol {
            // Clamped, not wrapped. The root lies in the same revolution
            // as the mean anomaly, so Newton leaves it inside the range
            // up to rounding -- and wrapping a converged root that
            // undershot zero by an ulp would return it as a full turn.
            return Ok(anomaly.clamp(0.0, std::f64::consts::TAU));
        }
        let slope = 1.0 - e * cos;
        if slope.abs() < 1e-14 {
            // Flat where the derivative vanishes; nudge rather than divide.
            anomaly += 0.1;
            continue;
        }
        anomaly -= residual / slope;
    }
    Err(GeomError::Degenerate("Kepler's equation did not converge"))
}

/// Solves the hyperbolic Kepler equation `M = e sinh H - H`.
///
/// The hyperbolic form has no periodicity to wrap, and `sinh` grows
/// exponentially, so a poor seed overflows rather than merely converging
/// slowly. The seed here is logarithmic for large `M`, which is where the
/// solution actually lives.
///
/// # Errors
/// Returns an error for an eccentricity at or below one, a non-finite
/// mean anomaly or tolerance, a non-positive tolerance, or an iteration
/// that fails to converge.
pub fn kepler_solve_hyperbolic(mean_anomaly: f64, e: f64, tol: f64) -> Result<f64, GeomError> {
    if !(e > 1.0) || !mean_anomaly.is_finite() || !(tol > 0.0) || !tol.is_finite() {
        return Err(GeomError::InvalidArgument("kepler_solve_hyperbolic: bad parameters"));
    }
    let sign = if mean_anomaly < 0.0 { -1.0 } else { 1.0 };
    let m = mean_anomaly.abs();
    if m == 0.0 {
        return Ok(0.0);
    }
    // `asinh(M/e)` inverts the leading term of `e sinh H = M + H` and is
    // bounded everywhere. The textbook small-M seed `M/(e-1)` is not: at
    // an eccentricity of 1.001 it puts the first guess at four hundred,
    // where `cosh` overflows and the iteration has no derivative left.
    let mut anomaly = (m / e).asinh();
    for _ in 0..200 {
        let residual = e * anomaly.sinh() - anomaly - m;
        if residual.abs() < tol * (1.0 + m) {
            return Ok(sign * anomaly);
        }
        let slope = e * anomaly.cosh() - 1.0;
        if !(slope.abs() > 1e-300) || !slope.is_finite() {
            return Err(GeomError::Degenerate("the hyperbolic iteration lost its derivative"));
        }
        let step = residual / slope;
        // A full Newton step can overshoot into the exponential's tail
        // and overflow; halving keeps it in range.
        anomaly -= if step.abs() > 1.0 { step.signum() * 1.0 } else { step };
    }
    Err(GeomError::Degenerate("the hyperbolic Kepler equation did not converge"))
}

/// The true anomaly corresponding to an eccentric anomaly.
///
/// `tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2)`, evaluated through `atan2` so
/// it stays correct across all four quadrants rather than losing a half
/// turn where the tangent wraps.
///
/// # Errors
/// Returns an error for an eccentricity outside `[0, 1)` or a non-finite
/// anomaly.
pub fn true_from_eccentric(eccentric: f64, e: f64) -> Result<f64, GeomError> {
    if !(0.0..1.0).contains(&e) || !eccentric.is_finite() {
        return Err(GeomError::InvalidArgument("true_from_eccentric: bad parameters"));
    }
    let (sin, cos) = eccentric.sin_cos();
    let factor = (1.0 - e * e).sqrt();
    Ok(wrap_two_pi((factor * sin).atan2(cos - e)))
}

/// The eccentric anomaly corresponding to a true anomaly.
///
/// # Errors
/// As [`true_from_eccentric`].
pub fn eccentric_from_true(true_anomaly: f64, e: f64) -> Result<f64, GeomError> {
    if !(0.0..1.0).contains(&e) || !true_anomaly.is_finite() {
        return Err(GeomError::InvalidArgument("eccentric_from_true: bad parameters"));
    }
    let (sin, cos) = true_anomaly.sin_cos();
    let factor = (1.0 - e * e).sqrt();
    Ok(wrap_two_pi((factor * sin).atan2(cos + e)))
}

/// The mean anomaly corresponding to an eccentric anomaly: Kepler's
/// equation read forwards, which needs no solving at all.
///
/// # Errors
/// As [`true_from_eccentric`].
pub fn mean_from_eccentric(eccentric: f64, e: f64) -> Result<f64, GeomError> {
    if !(0.0..1.0).contains(&e) || !eccentric.is_finite() {
        return Err(GeomError::InvalidArgument("mean_from_eccentric: bad parameters"));
    }
    Ok(wrap_two_pi(eccentric - e * eccentric.sin()))
}

/// The orbital period `2 pi sqrt(a^3 / mu)`.
///
/// # Errors
/// Returns an error for a non-positive semi-major axis or gravitational
/// parameter, which is to say for an unbound orbit, where there is no
/// period.
pub fn orbit_period(a: f64, mu: f64) -> Result<f64, GeomError> {
    if !(a > 0.0) || !(mu > 0.0) || !a.is_finite() || !mu.is_finite() {
        return Err(GeomError::InvalidArgument("orbit_period: an unbound orbit has no period"));
    }
    Ok(std::f64::consts::TAU * (a * a * a / mu).sqrt())
}

/// The vis-viva speed at radius `r` on an orbit of semi-major axis `a`:
/// `sqrt(mu (2/r - 1/a))`.
///
/// The equation is conservation of energy rearranged, and it holds for
/// every conic: a positive `a` for an ellipse, negative for a hyperbola,
/// and the parabolic limit `1/a = 0` giving escape speed. That one formula
/// covers all three is the reason it is the workhorse of manoeuvre
/// planning.
///
/// The formula knows about energy, not about geometry: it returns a speed
/// for any radius up to `2a`, which for a bound orbit reaches past
/// apoapsis at `a(1+e)`. Radii between the two are not on the orbit and
/// the number returned there is the speed a body of that energy *would*
/// have, not one anything reaches. Beyond `2a` the kinetic energy would be
/// negative and there is no answer at all.
///
/// # Errors
/// Returns an error for a non-positive radius or gravitational parameter,
/// a NaN input, or a radius beyond `2a` on a bound orbit, where the speed
/// would be imaginary.
pub fn vis_viva(r: f64, a: f64, mu: f64) -> Result<f64, GeomError> {
    // An infinite semi-major axis is the parabolic case, where `1/a` is
    // zero and the formula gives escape speed. It is a legitimate input,
    // not a malformed one.
    if !(r > 0.0) || !(mu > 0.0) || !r.is_finite() || !mu.is_finite() || a.is_nan() {
        return Err(GeomError::InvalidArgument("vis_viva: bad radius or gravitational parameter"));
    }
    let squared = mu * (2.0 / r - 1.0 / a);
    if squared < 0.0 {
        return Err(GeomError::Degenerate(
            "that radius is beyond twice the semi-major axis: the speed would be imaginary",
        ));
    }
    Ok(squared.sqrt())
}

/// The state vectors implied by a set of elements: the inverse of
/// [`OrbitalElements::from_state_vectors`].
///
/// The position and velocity are built in the perifocal frame, where the
/// orbit is a plane conic with periapsis along the x axis, and then
/// rotated into the reference frame by the three Euler angles. Doing it
/// this way rather than by direct formulae is what keeps the retrograde
/// and equatorial cases right: the rotation is the same in every case,
/// and only the angles differ.
///
/// # Errors
/// Returns an error for a non-positive gravitational parameter, a
/// non-finite element, a negative eccentricity, or a semi-latus rectum
/// that comes out non-positive -- which happens for a degenerate orbit
/// with no extent.
pub fn state_from_elements(
    elements: &OrbitalElements,
    mu: f64,
) -> Result<(Vec3, Vec3), GeomError> {
    let el = *elements;
    if !(mu > 0.0) || !mu.is_finite() || el.eccentricity < 0.0 {
        return Err(GeomError::InvalidArgument("state_from_elements: bad parameters"));
    }
    if ![
        el.semi_major_axis,
        el.eccentricity,
        el.inclination,
        el.longitude_ascending_node,
        el.argument_periapsis,
        el.true_anomaly,
    ]
    .iter()
    .all(|x| x.is_finite())
    {
        return Err(GeomError::InvalidArgument("an orbital element is not finite"));
    }
    // The semi-latus rectum is what makes one formula serve every conic.
    let p = el.semi_major_axis * (1.0 - el.eccentricity * el.eccentricity);
    if !(p > 0.0) {
        return Err(GeomError::Degenerate("the orbit has no positive semi-latus rectum"));
    }
    let (sin_nu, cos_nu) = el.true_anomaly.sin_cos();
    let radius = p / (1.0 + el.eccentricity * cos_nu);
    let speed = (mu / p).sqrt();
    // Perifocal frame: periapsis along x, motion counter-clockwise.
    let r_pf = Vec3::new(radius * cos_nu, radius * sin_nu, 0.0);
    let v_pf = Vec3::new(-speed * sin_nu, speed * (el.eccentricity + cos_nu), 0.0);
    Ok((
        rotate_to_frame(r_pf, &el),
        rotate_to_frame(v_pf, &el),
    ))
}

/// Rotates a perifocal vector into the reference frame by the three
/// Euler angles: argument of periapsis, inclination, then node.
fn rotate_to_frame(v: Vec3, el: &OrbitalElements) -> Vec3 {
    let (sw, cw) = el.argument_periapsis.sin_cos();
    let (si, ci) = el.inclination.sin_cos();
    let (so, co) = el.longitude_ascending_node.sin_cos();
    // Rotate by the argument of periapsis about z.
    let x1 = v.x * cw - v.y * sw;
    let y1 = v.x * sw + v.y * cw;
    let z1 = v.z;
    // Then by the inclination about x.
    let x2 = x1;
    let y2 = y1 * ci - z1 * si;
    let z2 = y1 * si + z1 * ci;
    // Then by the node about z.
    Vec3::new(x2 * co - y2 * so, x2 * so + y2 * co, z2)
}

/// Propagates a two-body state forward by `dt` using Lagrange's f and g
/// functions.
///
/// The trick is that the new position is a *linear combination of the old
/// position and velocity*: `r = f r0 + g v0`, with `f` and `g` scalars
/// depending only on the change in eccentric anomaly. The orbit plane is
/// therefore preserved exactly by construction, whatever the arithmetic
/// does -- which is why this is used in preference to integrating the
/// equations of motion when the two-body assumption holds.
///
/// Elliptic and hyperbolic orbits are handled by their own anomaly
/// solvers. A parabolic orbit -- eccentricity exactly one -- has neither
/// and is refused rather than approximated.
///
/// # Errors
/// Returns an error for a non-positive gravitational parameter, a
/// non-finite input, a degenerate or parabolic orbit, or an anomaly
/// solver that does not converge.
pub fn propagate_kepler(
    r0: Vec3,
    v0: Vec3,
    dt: f64,
    mu: f64,
) -> Result<(Vec3, Vec3), GeomError> {
    if !(mu > 0.0) || !mu.is_finite() || !dt.is_finite() {
        return Err(GeomError::InvalidArgument("propagate_kepler: bad time or parameter"));
    }
    let r_mag = r0.magnitude();
    let v_mag = v0.magnitude();
    if !(r_mag > 0.0) || !r_mag.is_finite() || !v_mag.is_finite() {
        return Err(GeomError::InvalidArgument("propagate_kepler: bad state"));
    }
    if dt == 0.0 {
        return Ok((r0, v0));
    }
    let energy = 0.5 * v_mag * v_mag - mu / r_mag;
    let radial = r0.dot(&v0);
    if energy.abs() < 1e-14 * mu / r_mag {
        return Err(GeomError::Degenerate(
            "a parabolic orbit has neither an elliptic nor a hyperbolic anomaly",
        ));
    }
    if energy < 0.0 {
        let a = -mu / (2.0 * energy);
        let n = (mu / (a * a * a)).sqrt();
        // The change in eccentric anomaly satisfies a Kepler-like
        // equation in its own right, with the initial radius and radial
        // velocity carrying the starting point.
        let sigma = radial / mu.sqrt();
        let target = n * dt;
        let residual = |de: f64| {
            let (sin, cos) = de.sin_cos();
            de + sigma / a.sqrt() * (1.0 - cos) - (1.0 - r_mag / a) * sin - target
        };
        let slope = |de: f64| {
            let (sin, cos) = de.sin_cos();
            1.0 + sigma / a.sqrt() * sin - (1.0 - r_mag / a) * cos
        };
        let mut de = target;
        let mut converged = false;
        for _ in 0..200 {
            let value = residual(de);
            if value.abs() < 1e-13 * (1.0 + target.abs()) {
                converged = true;
                break;
            }
            let derivative = slope(de);
            if derivative.abs() < 1e-14 {
                de += 0.1;
                continue;
            }
            de -= value / derivative;
        }
        if !converged {
            return Err(GeomError::Degenerate("the propagation did not converge"));
        }
        let (sin, cos) = de.sin_cos();
        let f = 1.0 - a / r_mag * (1.0 - cos);
        let g = dt + (sin - de) / n;
        let r = Vec3::new(
            f * r0.x + g * v0.x,
            f * r0.y + g * v0.y,
            f * r0.z + g * v0.z,
        );
        let r_new = r.magnitude();
        if !(r_new > 0.0) {
            return Err(GeomError::Degenerate("the propagated radius collapsed"));
        }
        let f_dot = -(mu * a).sqrt() / (r_new * r_mag) * sin;
        let g_dot = 1.0 - a / r_new * (1.0 - cos);
        let v = Vec3::new(
            f_dot * r0.x + g_dot * v0.x,
            f_dot * r0.y + g_dot * v0.y,
            f_dot * r0.z + g_dot * v0.z,
        );
        return Ok((r, v));
    }
    // Hyperbolic: the same construction with hyperbolic functions.
    let a = -mu / (2.0 * energy);
    let sigma = radial / mu.sqrt();
    let scale = (-a).sqrt();
    let target = dt * (mu / (-a * a * a)).sqrt();
    let residual = |dh: f64| {
        -(1.0 - r_mag / a) * dh.sinh() + sigma / scale * (dh.cosh() - 1.0) + dh - target
    };
    let slope =
        |dh: f64| -(1.0 - r_mag / a) * dh.cosh() + sigma / scale * dh.sinh() + 1.0;
    let mut dh = target.clamp(-5.0, 5.0);
    let mut converged = false;
    for _ in 0..300 {
        let value = residual(dh);
        if value.abs() < 1e-12 * (1.0 + target.abs()) {
            converged = true;
            break;
        }
        let derivative = slope(dh);
        if !(derivative.abs() > 1e-300) || !derivative.is_finite() {
            return Err(GeomError::Degenerate("the hyperbolic propagation lost its derivative"));
        }
        let step = value / derivative;
        dh -= if step.abs() > 1.0 { step.signum() } else { step };
    }
    if !converged {
        return Err(GeomError::Degenerate("the hyperbolic propagation did not converge"));
    }
    // Substituting E = i H and sqrt(a) = i sqrt(-a) into the elliptic f
    // and g flips the sign of both correction terms: the imaginary units
    // cancel in `f` and `g_dot` but not in `g` or `f_dot`.
    let f = 1.0 - a / r_mag * (1.0 - dh.cosh());
    let g = dt + (dh.sinh() - dh) / (mu / (-a * a * a)).sqrt();
    let r = Vec3::new(f * r0.x + g * v0.x, f * r0.y + g * v0.y, f * r0.z + g * v0.z);
    let r_new = r.magnitude();
    if !(r_new > 0.0) {
        return Err(GeomError::Degenerate("the propagated radius collapsed"));
    }
    let f_dot = (mu * -a).sqrt() / (r_new * r_mag) * dh.sinh();
    let g_dot = 1.0 - a / r_new * (1.0 - dh.cosh());
    let v = Vec3::new(
        f_dot * r0.x + g_dot * v0.x,
        f_dot * r0.y + g_dot * v0.y,
        f_dot * r0.z + g_dot * v0.z,
    );
    Ok((r, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    /// Earth's gravitational parameter, km^3/s^2.
    const MU: f64 = 398_600.441_8;
    const TAU: f64 = std::f64::consts::TAU;
    const PI: f64 = std::f64::consts::PI;

    /// The signed difference between two angles, in `(-pi, pi]`.
    fn angle_gap(a: f64, b: f64) -> f64 {
        (a - b + PI).rem_euclid(TAU) - PI
    }

    fn distance(a: Vec3, b: Vec3) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
    }

    #[test]
    fn kepler_solve_returns_an_anomaly_that_satisfies_the_equation() {
        // The equation is transcendental, so the only check that means
        // anything is substituting the answer back. Held across
        // eccentricities up to 0.999, where the function is nearly flat
        // near periapsis and a poor seed diverges.
        for e in [0.0f64, 0.1, 0.5, 0.9, 0.99, 0.999] {
            for k in 0..500 {
                let m = TAU * k as f64 / 500.0;
                let anomaly = kepler_solve_elliptic(m, e, 1e-14).unwrap();
                let residual = anomaly - e * anomaly.sin() - m;
                assert!(
                    residual.abs() < 1e-13,
                    "at e={e}, M={m} the residual was {residual}"
                );
                assert!((0.0..TAU).contains(&anomaly), "the anomaly left its range: {anomaly}");
            }
        }
        // A circle has no equation to solve: the anomalies coincide.
        for k in 0..100 {
            let m = TAU * k as f64 / 100.0;
            assert!((kepler_solve_elliptic(m, 0.0, 1e-15).unwrap() - m).abs() < 1e-15);
        }
        // Periapsis and apoapsis are fixed points whatever the shape.
        for e in [0.0f64, 0.3, 0.95] {
            assert!(kepler_solve_elliptic(0.0, e, 1e-15).unwrap().abs() < 1e-12);
            assert!((kepler_solve_elliptic(PI, e, 1e-15).unwrap() - PI).abs() < 1e-12);
        }
        assert!(kepler_solve_elliptic(1.0, 1.0, 1e-12).is_err());
        assert!(kepler_solve_elliptic(1.0, -0.1, 1e-12).is_err());
        assert!(kepler_solve_elliptic(1.0, 0.5, 0.0).is_err());
        assert!(kepler_solve_elliptic(f64::NAN, 0.5, 1e-12).is_err());
    }

    #[test]
    fn the_three_anomalies_convert_back_and_forth_without_losing_a_turn() {
        // The tangent half-angle formula loses a half turn in two of four
        // quadrants unless it goes through atan2, which is what this
        // catches.
        for e in [0.0f64, 0.3, 0.8, 0.97] {
            for k in 0..400 {
                let nu = TAU * k as f64 / 400.0;
                let eccentric = eccentric_from_true(nu, e).unwrap();
                let back = true_from_eccentric(eccentric, e).unwrap();
                assert!(
                    angle_gap(back, nu).abs() < 1e-12,
                    "at e={e}, nu={nu} it came back as {back}"
                );
                // And the whole chain nu -> E -> M -> E -> nu.
                let mean = mean_from_eccentric(eccentric, e).unwrap();
                let solved = kepler_solve_elliptic(mean, e, 1e-14).unwrap();
                assert!(angle_gap(solved, eccentric).abs() < 1e-11);
                let round = true_from_eccentric(solved, e).unwrap();
                assert!(angle_gap(round, nu).abs() < 1e-10);
            }
            // At periapsis and apoapsis all three agree exactly.
            assert!(eccentric_from_true(0.0, e).unwrap().abs() < 1e-15);
            assert!(mean_from_eccentric(0.0, e).unwrap().abs() < 1e-15);
            assert!((eccentric_from_true(PI, e).unwrap() - PI).abs() < 1e-14);
            assert!((mean_from_eccentric(PI, e).unwrap() - PI).abs() < 1e-14);
        }
        // On a circle all three are the same angle.
        for k in 0..50 {
            let nu = TAU * k as f64 / 50.0;
            assert!(angle_gap(eccentric_from_true(nu, 0.0).unwrap(), nu).abs() < 1e-15);
            assert!(angle_gap(mean_from_eccentric(nu, 0.0).unwrap(), nu).abs() < 1e-15);
        }
        assert!(true_from_eccentric(1.0, 1.0).is_err());
        assert!(eccentric_from_true(1.0, 1.5).is_err());
        assert!(mean_from_eccentric(f64::INFINITY, 0.5).is_err());
    }

    #[test]
    fn between_periapsis_and_apoapsis_the_true_anomaly_runs_ahead_of_the_mean() {
        // Kepler's second law in one inequality: the body moves fastest
        // near periapsis, so it covers more true angle than uniform time
        // would suggest. The gap is zero at both ends and largest in
        // between, and it grows with eccentricity.
        for e in [0.1f64, 0.5, 0.9] {
            let mut largest = 0.0f64;
            for k in 1..200 {
                let nu = PI * k as f64 / 200.0;
                let mean = mean_from_eccentric(eccentric_from_true(nu, e).unwrap(), e).unwrap();
                assert!(nu > mean, "at e={e}, nu={nu} the mean anomaly {mean} was not behind");
                largest = largest.max(nu - mean);
            }
            // And on the way back the mean runs ahead instead.
            for k in 1..200 {
                let nu = PI + PI * k as f64 / 200.0;
                let mean = mean_from_eccentric(eccentric_from_true(nu, e).unwrap(), e).unwrap();
                assert!(nu < mean, "past apoapsis at e={e} the mean anomaly did not lead");
            }
            assert!(largest > 0.5 * e, "the lead was only {largest} at e={e}");
        }
    }

    #[test]
    fn the_hyperbolic_equation_is_solved_and_is_odd_in_its_argument() {
        for e in [1.001f64, 1.1, 2.0, 10.0] {
            for k in 0..100 {
                let m = -40.0 + 80.0 * k as f64 / 100.0;
                let h = kepler_solve_hyperbolic(m, e, 1e-13).unwrap();
                let residual = e * h.sinh() - h - m;
                assert!(
                    residual.abs() < 1e-11 * (1.0 + m.abs()),
                    "at e={e}, M={m} the residual was {residual}"
                );
            }
            // Zero maps to zero, and the equation is odd.
            assert!(kepler_solve_hyperbolic(0.0, e, 1e-14).unwrap().abs() < 1e-14);
            for m in [0.5f64, 3.0, 20.0] {
                let forward = kepler_solve_hyperbolic(m, e, 1e-13).unwrap();
                let backward = kepler_solve_hyperbolic(-m, e, 1e-13).unwrap();
                assert!((forward + backward).abs() < 1e-11, "{forward} against {backward}");
            }
        }
        assert!(kepler_solve_hyperbolic(1.0, 1.0, 1e-12).is_err());
        assert!(kepler_solve_hyperbolic(1.0, 0.5, 1e-12).is_err());
    }

    #[test]
    fn a_circular_equatorial_orbit_has_the_state_a_schoolbook_would_give() {
        let radius = 7000.0;
        let elements = OrbitalElements {
            semi_major_axis: radius,
            eccentricity: 0.0,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let (r, v) = state_from_elements(&elements, MU).unwrap();
        assert!((r.x - radius).abs() < 1e-9 && r.y.abs() < 1e-9 && r.z.abs() < 1e-9);
        let speed = (MU / radius).sqrt();
        assert!(v.x.abs() < 1e-12 && (v.y - speed).abs() < 1e-9 && v.z.abs() < 1e-12);
        // Position and velocity are perpendicular on a circle, everywhere.
        for k in 0..20 {
            let moved = OrbitalElements { true_anomaly: TAU * k as f64 / 20.0, ..elements };
            let (r, v) = state_from_elements(&moved, MU).unwrap();
            assert!((r.magnitude() - radius).abs() < 1e-9);
            assert!((v.magnitude() - speed).abs() < 1e-9);
            assert!(r.dot(&v).abs() < 1e-8 * radius * speed, "they were not perpendicular");
        }
    }

    #[test]
    fn the_state_matches_the_geometry_the_elements_describe() {
        // Angular momentum sqrt(mu p), the radius from the conic equation,
        // and the flight-path angle from the eccentricity. Each is an
        // independent statement about the same construction.
        let mut rng = Rng::new(0x0A57_1001);
        for _ in 0..300 {
            let a = 7000.0 + 30000.0 * rng.next_f64();
            let e = 0.9 * rng.next_f64();
            let nu = TAU * rng.next_f64();
            let elements = OrbitalElements {
                semi_major_axis: a,
                eccentricity: e,
                inclination: PI * rng.next_f64(),
                longitude_ascending_node: TAU * rng.next_f64(),
                argument_periapsis: TAU * rng.next_f64(),
                true_anomaly: nu,
            };
            let (r, v) = state_from_elements(&elements, MU).unwrap();
            let p = a * (1.0 - e * e);
            // The conic equation.
            let expected = p / (1.0 + e * nu.cos());
            assert!((r.magnitude() - expected).abs() < 1e-9 * expected);
            // Angular momentum.
            let h = r.cross(&v);
            assert!((h.magnitude() - (MU * p).sqrt()).abs() < 1e-8 * (MU * p).sqrt());
            // Energy, through vis-viva.
            let speed = vis_viva(r.magnitude(), a, MU).unwrap();
            assert!((v.magnitude() - speed).abs() < 1e-8 * speed);
            // The plane contains the position and the velocity, and the
            // inclination is the angle its normal makes with z.
            let inclination = (h.z / h.magnitude()).acos();
            assert!((inclination - elements.inclination).abs() < 1e-9);
        }
    }

    #[test]
    fn elements_and_state_are_inverse_to_each_other() {
        let mut rng = Rng::new(0x0A57_1002);
        for _ in 0..400 {
            let elements = OrbitalElements {
                semi_major_axis: 7000.0 + 30000.0 * rng.next_f64(),
                eccentricity: 0.9 * rng.next_f64(),
                // Away from zero and pi, where the node is undefined and
                // the element set itself is degenerate rather than the
                // conversion being wrong.
                inclination: 0.1 + (PI - 0.2) * rng.next_f64(),
                longitude_ascending_node: TAU * rng.next_f64(),
                argument_periapsis: TAU * rng.next_f64(),
                true_anomaly: TAU * rng.next_f64(),
            };
            let (r, v) = state_from_elements(&elements, MU).unwrap();
            let recovered = OrbitalElements::from_state_vectors(r, v, MU);
            assert!(
                (recovered.semi_major_axis - elements.semi_major_axis).abs()
                    < 1e-8 * elements.semi_major_axis
            );
            assert!((recovered.eccentricity - elements.eccentricity).abs() < 1e-9);
            assert!((recovered.inclination - elements.inclination).abs() < 1e-9);
            assert!(
                angle_gap(recovered.longitude_ascending_node, elements.longitude_ascending_node)
                    .abs()
                    < 1e-8
            );
            assert!(
                angle_gap(recovered.argument_periapsis, elements.argument_periapsis).abs() < 1e-7
            );
            assert!(angle_gap(recovered.true_anomaly, elements.true_anomaly).abs() < 1e-7);
            // And the state round trips through the elements.
            let (r2, v2) = state_from_elements(&recovered, MU).unwrap();
            assert!(distance(r2, r) < 1e-8 * r.magnitude());
            assert!(distance(v2, v) < 1e-8 * v.magnitude());
        }
        assert!(state_from_elements(&OrbitalElements {
            semi_major_axis: 7000.0,
            eccentricity: 1.0,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        }, MU).is_err());
    }

    #[test]
    fn a_full_period_of_propagation_returns_the_orbit_to_where_it_started() {
        let mut rng = Rng::new(0x0A57_1003);
        for _ in 0..200 {
            let a = 7000.0 + 30000.0 * rng.next_f64();
            let elements = OrbitalElements {
                semi_major_axis: a,
                eccentricity: 0.85 * rng.next_f64(),
                inclination: PI * rng.next_f64(),
                longitude_ascending_node: TAU * rng.next_f64(),
                argument_periapsis: TAU * rng.next_f64(),
                true_anomaly: TAU * rng.next_f64(),
            };
            let (r0, v0) = state_from_elements(&elements, MU).unwrap();
            let period = orbit_period(a, MU).unwrap();
            let (r1, v1) = propagate_kepler(r0, v0, period, MU).unwrap();
            assert!(
                distance(r1, r0) < 1e-9 * r0.magnitude(),
                "after one period it was {} km away",
                distance(r1, r0)
            );
            assert!(distance(v1, v0) < 1e-9 * v0.magnitude());
            // Three periods too, since the error would compound.
            let (r3, _) = propagate_kepler(r0, v0, 3.0 * period, MU).unwrap();
            assert!(distance(r3, r0) < 1e-8 * r0.magnitude());
        }
    }

    #[test]
    fn propagation_conserves_energy_and_angular_momentum_exactly() {
        // The f and g construction writes the new position as a linear
        // combination of the old position and velocity, so the orbit plane
        // is preserved by construction. Energy and the magnitude of the
        // angular momentum are not, and they are what a sign error in the
        // Lagrange coefficients destroys.
        let mut rng = Rng::new(0x0A57_1004);
        for _ in 0..150 {
            let a = 7000.0 + 30000.0 * rng.next_f64();
            let elements = OrbitalElements {
                semi_major_axis: a,
                eccentricity: 0.8 * rng.next_f64(),
                inclination: PI * rng.next_f64(),
                longitude_ascending_node: TAU * rng.next_f64(),
                argument_periapsis: TAU * rng.next_f64(),
                true_anomaly: TAU * rng.next_f64(),
            };
            let (r0, v0) = state_from_elements(&elements, MU).unwrap();
            let energy0 = 0.5 * v0.magnitude_squared() - MU / r0.magnitude();
            let h0 = r0.cross(&v0);
            let period = orbit_period(a, MU).unwrap();
            for fraction in [0.05f64, 0.37, 0.5, 0.83, 2.6] {
                let (r, v) = propagate_kepler(r0, v0, fraction * period, MU).unwrap();
                let energy = 0.5 * v.magnitude_squared() - MU / r.magnitude();
                assert!(
                    (energy - energy0).abs() < 1e-10 * energy0.abs(),
                    "energy drifted to {energy} from {energy0}"
                );
                let h = r.cross(&v);
                assert!((h.magnitude() - h0.magnitude()).abs() < 1e-10 * h0.magnitude());
                // The plane is preserved exactly, not merely closely.
                assert!(
                    distance(h.normalized(), h0.normalized()) < 1e-10,
                    "the orbit plane moved"
                );
            }
        }
    }

    #[test]
    fn propagation_composes_and_runs_backwards() {
        // Going forward twice is going forward once by the sum, and going
        // back undoes going forward. Both follow from the two-body
        // problem being time-reversible, and neither is built in.
        let mut rng = Rng::new(0x0A57_1005);
        for _ in 0..150 {
            let a = 8000.0 + 20000.0 * rng.next_f64();
            let elements = OrbitalElements {
                semi_major_axis: a,
                eccentricity: 0.7 * rng.next_f64(),
                inclination: PI * rng.next_f64(),
                longitude_ascending_node: TAU * rng.next_f64(),
                argument_periapsis: TAU * rng.next_f64(),
                true_anomaly: TAU * rng.next_f64(),
            };
            let (r0, v0) = state_from_elements(&elements, MU).unwrap();
            let period = orbit_period(a, MU).unwrap();
            let (t1, t2) = (0.19 * period, 0.44 * period);
            let (ra, va) = propagate_kepler(r0, v0, t1, MU).unwrap();
            let (rb, vb) = propagate_kepler(ra, va, t2, MU).unwrap();
            let (rc, vc) = propagate_kepler(r0, v0, t1 + t2, MU).unwrap();
            assert!(distance(rb, rc) < 1e-8 * r0.magnitude(), "composition failed");
            assert!(distance(vb, vc) < 1e-8 * v0.magnitude());
            // And back again.
            let (rd, vd) = propagate_kepler(rc, vc, -(t1 + t2), MU).unwrap();
            assert!(distance(rd, r0) < 1e-8 * r0.magnitude(), "reversal failed");
            assert!(distance(vd, v0) < 1e-8 * v0.magnitude());
            // Nothing at all happens in no time.
            let (re, ve) = propagate_kepler(r0, v0, 0.0, MU).unwrap();
            assert!(distance(re, r0) < 1e-15 && distance(ve, v0) < 1e-15);
        }
    }

    #[test]
    fn propagation_agrees_with_advancing_the_mean_anomaly_by_hand() {
        // Two independent routes to the same state: the Lagrange
        // coefficients, and converting to elements, adding n dt to the
        // mean anomaly, and converting back.
        let mut rng = Rng::new(0x0A57_1006);
        for _ in 0..200 {
            let a = 7000.0 + 20000.0 * rng.next_f64();
            let e = 0.7 * rng.next_f64();
            let elements = OrbitalElements {
                semi_major_axis: a,
                eccentricity: e,
                inclination: 0.2 + 2.5 * rng.next_f64(),
                longitude_ascending_node: TAU * rng.next_f64(),
                argument_periapsis: TAU * rng.next_f64(),
                true_anomaly: TAU * rng.next_f64(),
            };
            let (r0, v0) = state_from_elements(&elements, MU).unwrap();
            let dt = orbit_period(a, MU).unwrap() * (0.05 + 0.9 * rng.next_f64());

            let (r_prop, v_prop) = propagate_kepler(r0, v0, dt, MU).unwrap();

            let mean0 = mean_from_eccentric(
                eccentric_from_true(elements.true_anomaly, e).unwrap(),
                e,
            )
            .unwrap();
            let n = (MU / (a * a * a)).sqrt();
            let advanced = kepler_solve_elliptic(mean0 + n * dt, e, 1e-14).unwrap();
            let moved = OrbitalElements {
                true_anomaly: true_from_eccentric(advanced, e).unwrap(),
                ..elements
            };
            let (r_el, v_el) = state_from_elements(&moved, MU).unwrap();
            assert!(
                distance(r_prop, r_el) < 1e-7 * r0.magnitude(),
                "the two routes differ by {} km",
                distance(r_prop, r_el)
            );
            assert!(distance(v_prop, v_el) < 1e-7 * v0.magnitude());
        }
    }

    #[test]
    fn an_unbound_orbit_propagates_without_losing_its_energy() {
        // The hyperbolic branch of the Lagrange coefficients comes from
        // substituting E = i H, which flips the sign of two of the four.
        // Getting either wrong leaves the trajectory looking plausible
        // while the energy drifts by half its value over an hour.
        let r0 = Vec3::new(7000.0, 0.0, 0.0);
        for speed in [12.0f64, 15.0, 25.0] {
            let v0 = Vec3::new(0.0, speed, 0.0);
            let energy0 = 0.5 * speed * speed - MU / 7000.0;
            assert!(energy0 > 0.0, "the test orbit is not unbound at {speed} km/s");
            let h0 = r0.cross(&v0);
            let mut previous = 7000.0;
            for dt in [10.0f64, 100.0, 1000.0, 5000.0, 20000.0] {
                let (r, v) = propagate_kepler(r0, v0, dt, MU).unwrap();
                let energy = 0.5 * v.magnitude_squared() - MU / r.magnitude();
                assert!(
                    (energy - energy0).abs() < 1e-10 * energy0,
                    "at dt={dt} the energy went from {energy0} to {energy}"
                );
                let h = r.cross(&v);
                assert!((h.magnitude() - h0.magnitude()).abs() < 1e-9 * h0.magnitude());
                // It recedes, and never comes back.
                assert!(r.magnitude() > previous, "the trajectory turned around");
                previous = r.magnitude();
            }
            // And it reverses like any other two-body trajectory.
            let (r, v) = propagate_kepler(r0, v0, 3000.0, MU).unwrap();
            let (back, back_v) = propagate_kepler(r, v, -3000.0, MU).unwrap();
            assert!(distance(back, r0) < 1e-7 * 7000.0);
            assert!(distance(back_v, v0) < 1e-7 * speed);
        }
    }

    #[test]
    fn a_parabolic_orbit_is_refused_rather_than_forced_into_the_wrong_branch() {
        // Escape speed exactly: neither an ellipse nor a hyperbola, and
        // neither anomaly exists. Approximating it with either would give
        // a plausible trajectory that is not the right one.
        let r0 = Vec3::new(7000.0, 0.0, 0.0);
        let escape = (2.0 * MU / 7000.0).sqrt();
        let v0 = Vec3::new(0.0, escape, 0.0);
        assert!(propagate_kepler(r0, v0, 100.0, MU).is_err());
        // A hair either side works.
        assert!(propagate_kepler(r0, Vec3::new(0.0, escape * 0.999, 0.0), 100.0, MU).is_ok());
        assert!(propagate_kepler(r0, Vec3::new(0.0, escape * 1.001, 0.0), 100.0, MU).is_ok());
        assert!(propagate_kepler(r0, v0, 100.0, 0.0).is_err());
        assert!(propagate_kepler(Vec3::new(0.0, 0.0, 0.0), v0, 100.0, MU).is_err());
        assert!(propagate_kepler(r0, v0, f64::NAN, MU).is_err());
    }

    #[test]
    fn the_period_follows_keplers_third_law_and_vis_viva_covers_every_conic() {
        // T^2 proportional to a^3, which is the law itself.
        let base = orbit_period(7000.0, MU).unwrap();
        for factor in [2.0f64, 4.0, 10.0] {
            let scaled = orbit_period(7000.0 * factor, MU).unwrap();
            assert!(
                (scaled / base - factor.powf(1.5)).abs() < 1e-12,
                "scaling a by {factor} scaled T by {}",
                scaled / base
            );
        }
        // A low Earth orbit takes about ninety minutes.
        let leo = orbit_period(6778.0, MU).unwrap();
        assert!((leo / 60.0 - 92.6).abs() < 0.5, "it came out at {} minutes", leo / 60.0);
        // Geostationary is a sidereal day.
        let geo = orbit_period(42_164.0, MU).unwrap();
        assert!((geo - 86_164.0).abs() < 20.0, "it came out at {geo} seconds");

        // Vis-viva: circular, escape, and hyperbolic excess.
        let r = 7000.0;
        assert!((vis_viva(r, r, MU).unwrap() - (MU / r).sqrt()).abs() < 1e-12);
        // The parabolic limit, 1/a = 0, is escape speed.
        assert!((vis_viva(r, f64::INFINITY, MU).unwrap() - (2.0 * MU / r).sqrt()).abs() < 1e-12);
        // A hyperbola has a negative semi-major axis and speed above escape.
        assert!(vis_viva(r, -20000.0, MU).unwrap() > (2.0 * MU / r).sqrt());
        // Beyond apoapsis there is no orbit to be on.
        assert!(vis_viva(30000.0, 10000.0, MU).is_err());
        assert!(vis_viva(0.0, 10000.0, MU).is_err());
        assert!(orbit_period(-7000.0, MU).is_err());
        assert!(orbit_period(7000.0, 0.0).is_err());
    }
}
