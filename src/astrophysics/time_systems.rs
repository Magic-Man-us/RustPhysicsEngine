//! Astronomical time: Julian dates and sidereal time.
//!
//! # Why a day is not a day
//!
//! The Earth turns once on its axis in 23h 56m 04s -- a *sidereal* day --
//! and takes the extra four minutes to face the sun again, because it has
//! moved along its orbit in the meantime. A solar day is therefore longer
//! than a rotation, by almost exactly one part in 366. Everything about
//! pointing a telescope, predicting a satellite pass or reading a ground
//! track depends on keeping the two apart.
//!
//! Sidereal time is the hour angle of the vernal equinox, which is to say
//! how far the Earth has turned relative to the stars. Greenwich mean
//! sidereal time is that quantity at longitude zero, and adding the
//! observer's longitude gives the local value. Right ascension is measured
//! from the same origin, so an object is due south exactly when the local
//! sidereal time equals its right ascension -- which is the whole reason
//! the quantity exists.
//!
//! # What is approximated here
//!
//! `UT1` and `UTC` are treated as the same thing. They differ by up to
//! 0.9 seconds, which is 0.0037 degrees of rotation -- irrelevant for
//! anything in this module and decisive for geodesy. The `TT`/`UTC`
//! offset from leap seconds is likewise ignored; the sun and planet
//! positions here are low-precision approximations for which it does not
//! matter.

use crate::error::GeomError;

/// The Julian date of the J2000.0 epoch: noon on 1 January 2000, TT.
pub const J2000: f64 = 2_451_545.0;

/// Days in a Julian century, which is what the polynomial series are
/// expressed in.
pub const JULIAN_CENTURY: f64 = 36_525.0;

/// The Julian date of a Gregorian calendar moment.
///
/// Uses the standard Fliegel-Van Flandern arithmetic, shifting January
/// and February into the previous year so the leap-day irregularity falls
/// at the end. The count begins at noon, not midnight -- a convention
/// from before electric light, kept because it puts a single night's
/// observations inside one Julian day.
///
/// Proleptic Gregorian throughout: dates before the 1582 reform are given
/// the Gregorian rule rather than the Julian one, which is what almost
/// every astronomical application wants and is not what a historian
/// wants.
///
/// # Errors
/// Returns an error for a month outside 1..=12, a day outside 1..=31, a
/// time component out of range, or a non-finite second.
pub fn julian_date(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: f64,
) -> Result<f64, GeomError> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(GeomError::InvalidArgument("julian_date: bad month or day"));
    }
    if hour > 23 || minute > 59 || !(0.0..61.0).contains(&second) || !second.is_finite() {
        return Err(GeomError::InvalidArgument("julian_date: bad time of day"));
    }
    // January and February become months 13 and 14 of the previous year,
    // which puts the leap day at the end of the arithmetic year.
    let (y, m) = if month <= 2 { (year - 1, month + 12) } else { (year, month) };
    let a = y.div_euclid(100);
    let b = 2 - a + a.div_euclid(4);
    let days = (365.25 * f64::from(y + 4716)).floor()
        + (30.6001 * f64::from(m + 1)).floor()
        + f64::from(day)
        + f64::from(b)
        - 1524.5;
    let fraction = (f64::from(hour) + f64::from(minute) / 60.0 + second / 3600.0) / 24.0;
    Ok(days + fraction)
}

/// The Gregorian calendar moment of a Julian date, as
/// `(year, month, day, hour, minute, second)`.
///
/// The inverse of [`julian_date`], proleptic Gregorian throughout to keep
/// it so, and exact to the limits of the representation: a Julian date near the present carries about 2.5
/// million days, so a double resolves it to some 20 microseconds. That is
/// why serious work splits the date into an integer part and a fraction,
/// which this does not.
///
/// # Errors
/// Returns an error for a non-finite Julian date or one outside the range
/// the arithmetic covers.
pub fn jd_to_calendar(jd: f64) -> Result<(i32, u32, u32, u32, u32, f64), GeomError> {
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("jd_to_calendar: the date is out of range"));
    }
    let shifted = jd + 0.5;
    let z = shifted.floor();
    let fraction = shifted - z;
    // Proleptic Gregorian throughout, matching `julian_date`. The
    // textbook form of this algorithm switches to the Julian calendar
    // below JD 2299161 -- the 1582 reform -- which is what a historian
    // wants and makes the two functions stop inverting each other:
    // 1 January -4712 went out as JD 38 and came back as 8 February.
    let alpha = ((z - 1_867_216.25) / 36_524.25).floor();
    let a = z + 1.0 + alpha - (alpha / 4.0).floor();
    let b = a + 1524.0;
    let c = ((b - 122.1) / 365.25).floor();
    let d = (365.25 * c).floor();
    let e = ((b - d) / 30.6001).floor();
    let day_with_fraction = b - d - (30.6001 * e).floor() + fraction;
    let day = day_with_fraction.floor();
    let month = if e < 14.0 { e - 1.0 } else { e - 13.0 };
    let year = if month > 2.0 { c - 4716.0 } else { c - 4715.0 };

    let mut seconds = (day_with_fraction - day) * 86_400.0;
    // Rounding can put the second at exactly 60; carry it rather than
    // reporting a time that does not exist.
    let hour = (seconds / 3600.0).floor();
    seconds -= hour * 3600.0;
    let minute = (seconds / 60.0).floor();
    seconds -= minute * 60.0;
    if !(0.0..2e6).contains(&year) && !(-2e6..2e6).contains(&year) {
        return Err(GeomError::Degenerate("the year is out of range"));
    }
    Ok((
        year as i32,
        month as u32,
        day as u32,
        hour as u32,
        minute as u32,
        seconds,
    ))
}

/// Greenwich mean sidereal time in radians, from a Julian date.
///
/// The IAU 1982 polynomial in Julian centuries from J2000. The linear
/// coefficient, `8_640_184.812_866` seconds per century, is the whole
/// content: divided by the century's 36525 days it says the Earth gains
/// about 236.6 seconds of sidereal time per solar day, which is the four
/// minutes by which the stars rise earlier each night.
///
/// "Mean" means the equinox is the smoothly precessing one, without
/// nutation. Apparent sidereal time adds the equation of the equinoxes,
/// up to about a second of time, which matters for pointing a large
/// telescope and not for anything here.
///
/// # Errors
/// Returns an error for a non-finite or out-of-range Julian date.
pub fn gmst(jd: f64) -> Result<f64, GeomError> {
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("gmst: the date is out of range"));
    }
    let t = (jd - J2000) / JULIAN_CENTURY;
    // Seconds of sidereal time at 0h UT, plus the day's own rotation.
    let seconds = 67_310.548_41
        + (876_600.0 * 3600.0 + 8_640_184.812_866) * t
        + 0.093_104 * t * t
        - 6.2e-6 * t * t * t;
    let turns = seconds / 86_400.0;
    Ok(wrap_two_pi(turns.fract() * std::f64::consts::TAU))
}

/// Local mean sidereal time: Greenwich's plus the observer's longitude.
///
/// East longitude is positive. The result is what an object's right
/// ascension must equal for it to be due south, which is what makes it
/// the natural clock for an observatory.
///
/// # Errors
/// As [`gmst`], plus a non-finite longitude.
pub fn local_sidereal(jd: f64, longitude: f64) -> Result<f64, GeomError> {
    if !longitude.is_finite() {
        return Err(GeomError::InvalidArgument("local_sidereal: the longitude is not finite"));
    }
    Ok(wrap_two_pi(gmst(jd)? + longitude))
}

/// The Julian date of a two-line element set's epoch field.
///
/// TLEs carry the epoch as `YYDDD.DDDDDDDD`: a two-digit year and the
/// fractional day of that year. The two-digit year is resolved by the
/// convention the format itself uses -- 57 through 99 mean the twentieth
/// century and 00 through 56 the twenty-first, chosen because Sputnik
/// went up in 1957 and nothing older has a TLE.
///
/// # Errors
/// Returns an error for a non-finite epoch, a year outside 0..=99, or a
/// day of year outside `[1, 367)`.
pub fn tle_epoch_to_jd(epoch: f64) -> Result<f64, GeomError> {
    if !epoch.is_finite() || !(0.0..100_000.0).contains(&epoch) {
        return Err(GeomError::InvalidArgument("tle_epoch_to_jd: the epoch is out of range"));
    }
    let two_digit = (epoch / 1000.0).floor();
    let day_of_year = epoch - two_digit * 1000.0;
    if !(1.0..367.0).contains(&day_of_year) {
        return Err(GeomError::InvalidArgument("tle_epoch_to_jd: bad day of year"));
    }
    let year = if two_digit < 57.0 { 2000.0 + two_digit } else { 1900.0 + two_digit };
    // Day one is 1 January, so the offset is from 31 December of the year
    // before.
    let start = julian_date(year as i32 - 1, 12, 31, 0, 0, 0.0)?;
    Ok(start + day_of_year)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const TAU: f64 = std::f64::consts::TAU;

    #[test]
    fn the_julian_date_of_j2000_is_the_number_the_epoch_is_defined_by() {
        // Noon on 1 January 2000 is 2451545.0 exactly, by definition. It
        // is the anchor every polynomial in this module counts from.
        assert!((julian_date(2000, 1, 1, 12, 0, 0.0).unwrap() - J2000).abs() < 1e-9);
        // Midnight is half a day earlier, since the count starts at noon.
        assert!((julian_date(2000, 1, 1, 0, 0, 0.0).unwrap() - (J2000 - 0.5)).abs() < 1e-9);
        // Two other fixed points from the literature.
        assert!((julian_date(1600, 1, 1, 0, 0, 0.0).unwrap() - 2_305_447.5).abs() < 1e-9);
        assert!(
            (julian_date(1957, 10, 4, 19, 28, 34.0).unwrap() - 2_436_116.311_5).abs() < 1e-3,
            "Sputnik's launch came out at {}",
            julian_date(1957, 10, 4, 19, 28, 34.0).unwrap()
        );
    }

    #[test]
    fn one_day_of_calendar_is_one_of_julian_date() {
        // Across a leap day, a century that is not a leap year, one that
        // is, and a year end.
        for (y, m, d) in [
            (2024, 2, 28),
            (2023, 2, 28),
            (1900, 2, 28),
            (2000, 2, 28),
            (1999, 12, 31),
            (2026, 8, 25),
        ] {
            let today = julian_date(y, m, d, 0, 0, 0.0).unwrap();
            let tomorrow = jd_to_calendar(today + 1.0).unwrap();
            let back = julian_date(tomorrow.0, tomorrow.1, tomorrow.2, 0, 0, 0.0).unwrap();
            assert!((back - today - 1.0).abs() < 1e-9, "{y}-{m}-{d} plus a day went wrong");
        }
        // 1900 was not a leap year and 2000 was, which is the Gregorian
        // rule's whole content.
        let feb28 = julian_date(1900, 2, 28, 0, 0, 0.0).unwrap();
        assert_eq!(jd_to_calendar(feb28 + 1.0).unwrap().1, 3, "1900 had a 29 February");
        let feb28 = julian_date(2000, 2, 28, 0, 0, 0.0).unwrap();
        assert_eq!(jd_to_calendar(feb28 + 1.0).unwrap().2, 29, "2000 had no 29 February");
    }

    #[test]
    fn the_calendar_and_the_julian_date_invert_each_other() {
        // Including well before the 1582 reform: both functions are
        // proleptic Gregorian, and a version of the inverse that switched
        // to the Julian calendar there would send 1 January -4712 back as
        // 8 February.
        for (y, m, d, h, mi, s) in [
            (2000, 1, 1, 12, 0, 0.0f64),
            (1999, 12, 31, 23, 59, 59.0),
            (2024, 2, 29, 6, 30, 15.5),
            (1900, 3, 1, 0, 0, 0.0),
            (1582, 10, 15, 0, 0, 0.0),
            (1000, 6, 15, 18, 45, 30.0),
            (-4712, 1, 1, 12, 0, 0.0),
        ] {
            let jd = julian_date(y, m, d, h, mi, s).unwrap();
            let (ry, rm, rd, rh, rmi, rs) = jd_to_calendar(jd).unwrap();
            assert_eq!((ry, rm, rd, rh, rmi), (y, m, d, h, mi), "the date {y}-{m}-{d} came back wrong");
            assert!((rs - s).abs() < 1e-3, "the seconds came back as {rs} not {s}");
        }
    }

    #[test]
    fn the_julian_date_runs_forward_with_the_clock() {
        let mut previous = f64::NEG_INFINITY;
        for (y, m, d, h) in [
            (1999, 12, 31, 23),
            (2000, 1, 1, 0),
            (2000, 1, 1, 11),
            (2000, 1, 1, 12),
            (2000, 3, 1, 0),
            (2001, 1, 1, 0),
        ] {
            let jd = julian_date(y, m, d, h, 0, 0.0).unwrap();
            assert!(jd > previous, "{y}-{m}-{d} {h}h did not follow its predecessor");
            previous = jd;
        }
        // An hour is a twenty-fourth and a minute a 1440th -- but only to
        // the precision a double has left. A modern Julian date carries
        // about 2.46 million days, so an ulp is 5e-10 of a day, or 40
        // microseconds. That is the resolution limit the module documents,
        // and it is why serious work splits the date into an integer part
        // and a fraction.
        let base = julian_date(2026, 5, 5, 0, 0, 0.0).unwrap();
        let ulp = f64::EPSILON * base;
        assert!(ulp > 1e-10 && ulp < 1e-9, "an ulp at this date is {ulp} days");
        assert!((julian_date(2026, 5, 5, 1, 0, 0.0).unwrap() - base - 1.0 / 24.0).abs() < 4.0 * ulp);
        assert!(
            (julian_date(2026, 5, 5, 0, 1, 0.0).unwrap() - base - 1.0 / 1440.0).abs() < 4.0 * ulp
        );
        // Near the epoch itself, where the magnitude is smaller, the same
        // arithmetic is no better -- the count is what limits it, not the
        // operation.
        let early = julian_date(1, 1, 1, 0, 0, 0.0).unwrap();
        assert!((julian_date(1, 1, 1, 1, 0, 0.0).unwrap() - early - 1.0 / 24.0).abs() < 1e-9);
        assert!(julian_date(2026, 13, 1, 0, 0, 0.0).is_err());
        assert!(julian_date(2026, 0, 1, 0, 0, 0.0).is_err());
        assert!(julian_date(2026, 1, 32, 0, 0, 0.0).is_err());
        assert!(julian_date(2026, 1, 1, 24, 0, 0.0).is_err());
        assert!(julian_date(2026, 1, 1, 0, 60, 0.0).is_err());
        assert!(jd_to_calendar(f64::NAN).is_err());
    }

    #[test]
    fn sidereal_time_gains_four_minutes_on_the_clock_every_day() {
        // The Earth turns once relative to the stars in less time than it
        // takes to face the sun again, and the difference is what makes a
        // sidereal day 86164.09 seconds rather than 86400.
        let a = gmst(J2000).unwrap();
        let b = gmst(J2000 + 1.0).unwrap();
        let gained = (b - a).rem_euclid(TAU);
        let seconds = gained * 86_400.0 / TAU;
        assert!((seconds - 236.5554).abs() < 0.01, "it gained {seconds} seconds");
        // Which makes the sidereal day this long.
        let sidereal_day = 86_400.0 * TAU / (TAU + gained);
        assert!((sidereal_day - 86_164.09).abs() < 0.02, "the sidereal day was {sidereal_day} s");
        // The published value at J2000 itself.
        let hours = a * 12.0 / std::f64::consts::PI;
        assert!((hours - 18.697_374_558).abs() < 1e-5, "GMST at J2000 was {hours} h");
        // And it stays inside one turn.
        for offset in [-40_000.0f64, -1.0, 0.0, 0.37, 1000.0, 40_000.0] {
            let value = gmst(J2000 + offset).unwrap();
            assert!((0.0..TAU).contains(&value), "GMST left its range at {offset}");
        }
        assert!(gmst(f64::INFINITY).is_err());
    }

    #[test]
    fn local_sidereal_time_is_greenwichs_plus_the_longitude() {
        for jd in [J2000, J2000 + 1234.5, J2000 - 9876.25] {
            let greenwich = gmst(jd).unwrap();
            assert!((local_sidereal(jd, 0.0).unwrap() - greenwich).abs() < 1e-15);
            for longitude in [-3.0f64, -0.5, 0.5, 3.0] {
                let local = local_sidereal(jd, longitude).unwrap();
                let expected = (greenwich + longitude).rem_euclid(TAU);
                assert!((local - expected).abs() < 1e-12, "{local} against {expected}");
                assert!((0.0..TAU).contains(&local));
            }
        }
        // Fifteen degrees of longitude is an hour of sidereal time, which
        // is where time zones came from.
        let hour = local_sidereal(J2000, 15f64.to_radians()).unwrap() - gmst(J2000).unwrap();
        assert!((hour * 12.0 / std::f64::consts::PI - 1.0).abs() < 1e-12);
        assert!(local_sidereal(J2000, f64::NAN).is_err());
    }

    #[test]
    fn a_tle_epoch_resolves_its_two_digit_year_the_way_the_format_does() {
        // 57 through 99 are the twentieth century and 00 through 56 the
        // twenty-first, because nothing older than Sputnik has a TLE.
        let (y, m, d, ..) = jd_to_calendar(tle_epoch_to_jd(24_001.0).unwrap()).unwrap();
        assert_eq!((y, m, d), (2024, 1, 1), "24001 should be 1 January 2024");
        let (y, m, d, ..) = jd_to_calendar(tle_epoch_to_jd(98_001.0).unwrap()).unwrap();
        assert_eq!((y, m, d), (1998, 1, 1), "98001 should be 1 January 1998");
        let (y, ..) = jd_to_calendar(tle_epoch_to_jd(56_001.0).unwrap()).unwrap();
        assert_eq!(y, 2056);
        let (y, ..) = jd_to_calendar(tle_epoch_to_jd(57_001.0).unwrap()).unwrap();
        assert_eq!(y, 1957);
        // Day one is 1 January at midnight. The epoch here is 00001.0:
        // year 00, day 1.0.
        assert!(
            (tle_epoch_to_jd(1.0).unwrap() - julian_date(2000, 1, 1, 0, 0, 0.0).unwrap()).abs()
                < 1e-9
        );
        // And the fraction is a fraction of a day.
        let noon = tle_epoch_to_jd(24_001.5).unwrap();
        assert_eq!(jd_to_calendar(noon).unwrap().3, 12, "the half-day was not noon");
        // Day 366 exists in a leap year and is refused past 366.
        assert!(tle_epoch_to_jd(24_366.0).is_ok());
        assert!(tle_epoch_to_jd(24_367.0).is_err());
        assert!(tle_epoch_to_jd(24_000.5).is_err(), "there is no day zero");
        assert!(tle_epoch_to_jd(-1.0).is_err());
    }
}
