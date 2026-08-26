//! Properties of astronomical time and coordinates.
//!
//! Almost everything here is an exact statement. The calendar and the
//! Julian date are inverses; each pair of coordinate frames is a rotation
//! and so invertible; sidereal time advances at a fixed rate. Those hold
//! for every input and are checkable as identities rather than as
//! tolerances.
//!
//! The ephemerides are the exception and are treated differently. They are
//! truncated series with no exact answer to compare against, so what is
//! tested is the *envelope*: the Sun's declination never leaves the
//! obliquity's band, a planet's distance stays between its own perihelion
//! and aphelion, and the Moon's latitude is bounded by its orbital
//! inclination. Those are consequences of the orbits rather than of the
//! series, so they hold however many terms are kept.

use rust_physics_engine::astrophysics::coords::{
    ecliptic_to_equatorial, equatorial_to_ecliptic, equatorial_to_horizontal,
    horizontal_to_equatorial, mean_obliquity, moon_position_approx,
    planet_position_low_precision, precession_approx, rise_set_times, sun_position_approx, Planet,
};
use rust_physics_engine::astrophysics::time_systems::{
    gmst, jd_to_calendar, julian_date, local_sidereal, tle_epoch_to_jd, J2000,
};
use rust_physics_engine::monte_carlo::Rng;

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;
const HALF: f64 = std::f64::consts::FRAC_PI_2;

fn angle_gap(a: f64, b: f64) -> f64 {
    (a - b + PI).rem_euclid(TAU) - PI
}

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random valid calendar moment.
fn random_date(rng: &mut Rng) -> (i32, u32, u32, u32, u32, f64) {
    let year = -3000 + pick(rng, 7000) as i32;
    let month = 1 + pick(rng, 12) as u32;
    let longest = match month {
        2 => {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    (
        year,
        month,
        1 + pick(rng, longest) as u32,
        pick(rng, 24) as u32,
        pick(rng, 60) as u32,
        60.0 * rng.next_f64(),
    )
}

#[test]
fn prop_the_calendar_and_the_julian_date_are_inverses() {
    // Over four millennia, every month, and every day each month has --
    // including the leap days the Gregorian rule gives and withholds.
    let mut rng = Rng::new(0x0A57_5001);
    for _ in 0..4000 {
        let (year, month, day, hour, minute, second) = random_date(&mut rng);
        let jd = julian_date(year, month, day, hour, minute, second).unwrap();
        let (ry, rm, rd, rh, rmi, rs) = jd_to_calendar(jd).unwrap();
        assert_eq!(
            (ry, rm, rd, rh, rmi),
            (year, month, day, hour, minute),
            "{year}-{month}-{day} {hour}:{minute} came back wrong"
        );
        // The seconds are limited by the magnitude of the count, not by
        // the algorithm: an ulp at a modern date is 40 microseconds.
        assert!((rs - second).abs() < 1e-3, "the seconds came back {rs} not {second}");
    }
}

#[test]
fn prop_the_julian_date_is_strictly_increasing_in_time() {
    let mut rng = Rng::new(0x0A57_5002);
    for _ in 0..2000 {
        let (year, month, day, hour, minute, _) = random_date(&mut rng);
        let earlier = julian_date(year, month, day, hour, minute, 10.0).unwrap();
        let later = julian_date(year, month, day, hour, minute, 40.0).unwrap();
        assert!(later > earlier, "thirty seconds went backwards");
        // A day later is exactly one more, to the precision available.
        let ulp = f64::EPSILON * earlier.abs().max(1.0);
        let tomorrow = jd_to_calendar(earlier + 1.0).unwrap();
        let rebuilt =
            julian_date(tomorrow.0, tomorrow.1, tomorrow.2, tomorrow.3, tomorrow.4, 10.0).unwrap();
        assert!(
            (rebuilt - earlier - 1.0).abs() < 10.0 * ulp + 1e-6,
            "a day later was {} away",
            rebuilt - earlier
        );
    }
}

#[test]
fn prop_sidereal_time_advances_at_a_constant_rate() {
    // Greenwich mean sidereal time is very nearly linear in the Julian
    // date: the quadratic and cubic terms are tiny over any span that
    // matters. So the gain over any whole number of days is that number
    // times the daily gain, to within the higher-order terms.
    let mut rng = Rng::new(0x0A57_5003);
    let daily = {
        let a = gmst(J2000).unwrap();
        let b = gmst(J2000 + 1.0).unwrap();
        (b - a).rem_euclid(TAU)
    };
    for _ in 0..500 {
        let start = J2000 + (-20_000.0 + 40_000.0 * rng.next_f64());
        let days = 1.0 + pick(&mut rng, 500) as f64;
        let gained = angle_gap(gmst(start + days).unwrap(), gmst(start).unwrap());
        let predicted = angle_gap(days * daily, 0.0);
        assert!(
            angle_gap(gained, predicted).abs() < 2e-4,
            "over {days} days it gained {gained} against {predicted}"
        );
        // And it never leaves its range.
        assert!((0.0..TAU).contains(&gmst(start).unwrap()));
    }
}

#[test]
fn prop_local_sidereal_time_is_greenwichs_shifted_by_the_longitude() {
    let mut rng = Rng::new(0x0A57_5004);
    for _ in 0..1000 {
        let jd = J2000 + (-30_000.0 + 60_000.0 * rng.next_f64());
        let longitude = -PI + TAU * rng.next_f64();
        let local = local_sidereal(jd, longitude).unwrap();
        assert!((0.0..TAU).contains(&local));
        assert!(angle_gap(local, gmst(jd).unwrap() + longitude).abs() < 1e-12);
        // Shifting by a full turn of longitude changes nothing.
        assert!(
            angle_gap(local_sidereal(jd, longitude + TAU).unwrap(), local).abs() < 1e-12
        );
    }
}

#[test]
fn prop_every_coordinate_conversion_inverts_exactly() {
    // Each is a rotation of the sphere, so each has an exact inverse and
    // the composition is the identity for every position and every
    // observer.
    let mut rng = Rng::new(0x0A57_5005);
    for _ in 0..3000 {
        let ra = TAU * rng.next_f64();
        let dec = -HALF + PI * rng.next_f64();
        let latitude = -HALF + PI * rng.next_f64();
        let lst = TAU * rng.next_f64();
        let (azimuth, altitude) = equatorial_to_horizontal(ra, dec, latitude, lst).unwrap();
        assert!((0.0..TAU).contains(&azimuth));
        assert!((-HALF..=HALF).contains(&altitude));
        let (back_ra, back_dec) =
            horizontal_to_equatorial(azimuth, altitude, latitude, lst).unwrap();
        assert!(
            angle_gap(back_ra, ra).abs() < 1e-9,
            "the right ascension came back {back_ra} not {ra}"
        );
        assert!((back_dec - dec).abs() < 1e-12);

        let obliquity = 0.7 * rng.next_f64();
        let (era, edec) = ecliptic_to_equatorial(ra, dec, obliquity).unwrap();
        let (back_long, back_lat) = equatorial_to_ecliptic(era, edec, obliquity).unwrap();
        assert!(angle_gap(back_long, ra).abs() < 1e-9);
        assert!((back_lat - dec).abs() < 1e-12);
        // With no obliquity the ecliptic and equatorial frames coincide.
        let (same_ra, same_dec) = ecliptic_to_equatorial(ra, dec, 0.0).unwrap();
        assert!(angle_gap(same_ra, ra).abs() < 1e-12 && (same_dec - dec).abs() < 1e-12);
    }
}

#[test]
fn prop_the_horizontal_conversion_preserves_angular_separation() {
    // A rotation cannot change the angle between two directions. That is
    // a much stronger statement than the round trip, because it would
    // catch a conversion that inverted its own mistake.
    let mut rng = Rng::new(0x0A57_5006);
    for _ in 0..1000 {
        let latitude = -HALF + PI * rng.next_f64();
        let lst = TAU * rng.next_f64();
        let mut positions = Vec::new();
        for _ in 0..4 {
            let ra = TAU * rng.next_f64();
            let dec = -HALF + PI * rng.next_f64();
            let (azimuth, altitude) = equatorial_to_horizontal(ra, dec, latitude, lst).unwrap();
            positions.push(((ra, dec), (azimuth, altitude)));
        }
        let separation = |a: (f64, f64), b: (f64, f64)| {
            (a.1.sin() * b.1.sin() + a.1.cos() * b.1.cos() * (a.0 - b.0).cos())
                .clamp(-1.0, 1.0)
                .acos()
        };
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let before = separation(positions[i].0, positions[j].0);
                let after = separation(positions[i].1, positions[j].1);
                assert!(
                    (before - after).abs() < 1e-9,
                    "the separation went from {before} to {after}"
                );
            }
        }
    }
}

#[test]
fn prop_precession_is_a_small_linear_drift_that_vanishes_at_its_epoch() {
    let mut rng = Rng::new(0x0A57_5007);
    for _ in 0..1000 {
        let ra = TAU * rng.next_f64();
        // Away from the poles, where the tangent in the linear form
        // diverges and the approximation stops being one.
        let dec = -1.2 + 2.4 * rng.next_f64();
        // At its own epoch it does nothing at all.
        let (same_ra, same_dec) = precession_approx(ra, dec, J2000).unwrap();
        assert!(angle_gap(same_ra, ra).abs() < 1e-15 && (same_dec - dec).abs() < 1e-15);

        // And the shift is linear in the elapsed time.
        let years = 1.0 + 80.0 * rng.next_f64();
        let (moved, moved_dec) = precession_approx(ra, dec, J2000 + years * 365.25).unwrap();
        let (double, double_dec) =
            precession_approx(ra, dec, J2000 + 2.0 * years * 365.25).unwrap();
        let first = angle_gap(moved, ra);
        let second = angle_gap(double, ra);
        assert!(
            (second / first - 2.0).abs() < 1e-9,
            "doubling the span scaled the shift by {}",
            second / first
        );
        assert!(((double_dec - dec) / (moved_dec - dec) - 2.0).abs() < 1e-6);
        // Over a century it is under a degree and a half, which is what
        // makes the linear form usable at all.
        let (century, _) = precession_approx(ra, dec, J2000 + 36_525.0).unwrap();
        assert!(angle_gap(century, ra).abs() < 0.05, "it moved {} rad", angle_gap(century, ra));
    }
}

#[test]
fn prop_the_sun_stays_inside_the_band_the_obliquity_allows() {
    // The declination cannot exceed the obliquity, because the Sun is on
    // the ecliptic by definition. And the distance stays between the
    // perihelion and aphelion of an orbit with eccentricity 0.0167.
    let mut rng = Rng::new(0x0A57_5008);
    let mut extreme = 0.0f64;
    for _ in 0..4000 {
        let jd = J2000 + (-36_500.0 + 73_000.0 * rng.next_f64());
        let (ra, dec, distance) = sun_position_approx(jd).unwrap();
        let obliquity = mean_obliquity(jd).unwrap();
        assert!((0.0..TAU).contains(&ra));
        assert!(
            dec.abs() <= obliquity + 1e-6,
            "the declination reached {} against an obliquity of {}",
            dec.to_degrees(),
            obliquity.to_degrees()
        );
        extreme = extreme.max(dec.abs());
        assert!((0.98..1.02).contains(&distance), "the distance was {distance} AU");
    }
    // And it attains the bound, twice a year.
    assert!(
        (extreme - mean_obliquity(J2000).unwrap()).abs() < 1e-3,
        "it only reached {} degrees",
        extreme.to_degrees()
    );
}

#[test]
fn prop_the_sun_advances_through_the_zodiac_once_a_year() {
    // Its ecliptic longitude gains a full turn in a tropical year, and
    // never runs backwards -- the Sun has no retrograde motion, unlike
    // every planet seen from the Earth.
    let mut previous = None;
    let mut total = 0.0;
    for step in 0..36_500 {
        let jd = J2000 + step as f64 * 0.1;
        let (ra, dec, _) = sun_position_approx(jd).unwrap();
        let (longitude, latitude) =
            equatorial_to_ecliptic(ra, dec, mean_obliquity(jd).unwrap()).unwrap();
        // The Sun is on the ecliptic, so its latitude is nothing.
        assert!(latitude.abs() < 1e-9, "the Sun was {} off the ecliptic", latitude.to_degrees());
        if let Some(before) = previous {
            let step = angle_gap(longitude, before);
            assert!(step > 0.0, "the Sun moved backwards");
            total += step;
        }
        previous = Some(longitude);
    }
    // 3649.9 days of sampling is 9.993 tropical years.
    let turns = total / TAU;
    assert!((turns - 9.993).abs() < 0.02, "it made {turns} turns in ten years");
}

#[test]
fn prop_the_moon_keeps_inside_its_own_orbit() {
    // A truncated series has no exact answer to check, but the envelope
    // is a property of the orbit rather than of the series: the distance
    // between perigee and apogee, and the ecliptic latitude bounded by
    // the five-degree inclination.
    let mut rng = Rng::new(0x0A57_5009);
    for _ in 0..4000 {
        let jd = J2000 + (-3650.0 + 7300.0 * rng.next_f64());
        let (ra, dec, distance) = moon_position_approx(jd).unwrap();
        assert!((0.0..TAU).contains(&ra));
        assert!((-HALF..=HALF).contains(&dec));
        assert!(
            (350_000.0..410_000.0).contains(&distance),
            "the Moon was {distance} km away"
        );
        let (_, latitude) = equatorial_to_ecliptic(ra, dec, mean_obliquity(jd).unwrap()).unwrap();
        assert!(
            latitude.to_degrees().abs() < 6.0,
            "it was {} degrees off the ecliptic",
            latitude.to_degrees()
        );
        // The declination can exceed the obliquity, because the orbit is
        // tilted to the ecliptic as well -- which is why the Moon rides
        // higher some years than others.
        assert!(dec.to_degrees().abs() < 29.0);
    }
}

#[test]
fn prop_each_planet_stays_between_its_perihelion_and_aphelion() {
    let bounds = [
        (Planet::Mercury, 0.305, 0.470),
        (Planet::Venus, 0.716, 0.730),
        (Planet::Earth, 0.980, 1.020),
        (Planet::Mars, 1.375, 1.670),
        (Planet::Jupiter, 4.93, 5.48),
        (Planet::Saturn, 8.99, 10.10),
        (Planet::Uranus, 18.2, 20.2),
        (Planet::Neptune, 29.7, 30.4),
    ];
    let mut rng = Rng::new(0x0A57_500A);
    for _ in 0..600 {
        // Inside the Standish elements' stated window of 1800 to 2050.
        let jd = julian_date(1800 + pick(&mut rng, 250) as i32, 1, 1, 0, 0, 0.0).unwrap()
            + 365.0 * rng.next_f64();
        let mut previous = 0.0;
        for (planet, near, far) in bounds {
            let (longitude, latitude, distance) =
                planet_position_low_precision(planet, jd).unwrap();
            assert!((0.0..TAU).contains(&longitude));
            assert!(
                (near..=far).contains(&distance),
                "{planet:?} was at {distance} AU, outside [{near}, {far}]"
            );
            assert!(distance > previous, "{planet:?} was inside the previous planet");
            previous = distance;
            assert!(latitude.to_degrees().abs() < 7.5);
        }
    }
}

#[test]
fn prop_a_planets_longitude_advances_once_per_its_own_year() {
    // Heliocentric motion is never retrograde -- that is a geocentric
    // illusion -- so the longitude increases monotonically, and it makes
    // one turn per orbital period.
    for (planet, years) in [
        (Planet::Mercury, 0.2408f64),
        (Planet::Venus, 0.6152),
        (Planet::Earth, 1.0),
        (Planet::Mars, 1.8808),
        (Planet::Jupiter, 11.862),
    ] {
        let span = years * 365.25;
        let steps = 400;
        let mut previous = None;
        let mut total = 0.0;
        for step in 0..=steps {
            let jd = J2000 + span * step as f64 / steps as f64;
            let (longitude, _, _) = planet_position_low_precision(planet, jd).unwrap();
            if let Some(before) = previous {
                let advance = angle_gap(longitude, before);
                assert!(advance > 0.0, "{planet:?} moved backwards");
                total += advance;
            }
            previous = Some(longitude);
        }
        assert!(
            (total / TAU - 1.0).abs() < 0.01,
            "{planet:?} made {} turns in one of its years",
            total / TAU
        );
    }
}

#[test]
fn prop_rise_and_set_bracket_the_transit_or_the_body_is_circumpolar() {
    // Either a body crosses the horizon twice a day with its transit in
    // between, or it never crosses at all -- and which of the two is
    // decided by the latitude and the declination alone.
    let mut rng = Rng::new(0x0A57_500B);
    let mut rose = 0usize;
    let mut circumpolar = 0usize;
    for _ in 0..2000 {
        let latitude = -HALF + PI * rng.next_f64();
        let declination = -1.4 + 2.8 * rng.next_f64();
        let ra = TAU * rng.next_f64();
        let longitude = -PI + TAU * rng.next_f64();
        let jd = J2000 + 10_000.0 * rng.next_f64();
        let result = rise_set_times(ra, declination, latitude, longitude, jd, 0.0).unwrap();
        // The condition for a crossing, from the same spherical triangle.
        let cos_h = (-latitude.sin() * declination.sin()) / (latitude.cos() * declination.cos());
        match result {
            Some((rise, set)) => {
                rose += 1;
                assert!((-1.0..=1.0).contains(&cos_h), "it rose where it should not have");
                // Both fall within a day of the date asked for.
                assert!((rise - jd.floor()).abs() < 2.0 && (set - jd.floor()).abs() < 2.0);
                // At each, the body is on the horizon.
                for moment in [rise, set] {
                    let lst = local_sidereal(moment, longitude).unwrap();
                    let (_, altitude) =
                        equatorial_to_horizontal(ra, declination, latitude, lst).unwrap();
                    assert!(
                        altitude.abs() < 2e-3,
                        "at the crossing the altitude was {} degrees",
                        altitude.to_degrees()
                    );
                }
            }
            None => {
                circumpolar += 1;
                assert!(!(-1.0..=1.0).contains(&cos_h), "it should have risen");
            }
        }
    }
    assert!(rose > 1000 && circumpolar > 100, "{rose} rose and {circumpolar} were circumpolar");
}

#[test]
fn prop_a_tle_epoch_lands_in_the_year_its_two_digits_name() {
    let mut rng = Rng::new(0x0A57_500C);
    for _ in 0..2000 {
        let two_digit = pick(&mut rng, 100);
        let day = 1.0 + 364.0 * rng.next_f64();
        let epoch = two_digit as f64 * 1000.0 + day;
        let jd = tle_epoch_to_jd(epoch).unwrap();
        let (year, ..) = jd_to_calendar(jd).unwrap();
        let expected = if two_digit < 57 { 2000 + two_digit } else { 1900 + two_digit } as i32;
        assert_eq!(year, expected, "epoch {epoch} decoded to {year}");
        // The fractional day is the fraction of the day.
        let start = julian_date(expected, 1, 1, 0, 0, 0.0).unwrap();
        assert!(
            (jd - start - (day - 1.0)).abs() < 1e-6,
            "the day of year did not line up"
        );
    }
}
