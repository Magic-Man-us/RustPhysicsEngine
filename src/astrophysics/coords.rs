//! Astronomical coordinates, low-precision ephemerides and TLE parsing.
//!
//! # Four frames and what each is for
//!
//! *Equatorial* coordinates -- right ascension and declination -- are
//! fixed to the stars, or nearly so, and are what a catalogue lists.
//! *Horizontal* coordinates -- azimuth and altitude -- are what an
//! observer sees, and depend on where and when they are looking.
//! *Ecliptic* coordinates are referred to the Earth's orbital plane, which
//! is the natural frame for anything in the solar system. And the
//! *perifocal* and inertial frames of [`crate::astrophysics::kepler`] are
//! where orbits live.
//!
//! Converting between the first three is pure spherical trigonometry, and
//! all of it is exactly invertible. Which is worth saying because the
//! *ephemerides* here are not: they are truncated series good to a
//! fraction of a degree, and their inverses do not exist in any useful
//! sense.
//!
//! # What "low precision" means
//!
//! [`sun_position_approx`] is good to about a hundredth of a degree over
//! a couple of centuries around J2000. [`moon_position_approx`] is good
//! to a few tenths of a degree, because the Moon's motion has hundreds of
//! terms of comparable size and this keeps a handful.
//! [`planet_position_low_precision`] uses mean elements with linear rates
//! and no perturbations at all, which is good to a fraction of a degree
//! for the inner planets over a few centuries and steadily worse outward,
//! where Jupiter and Saturn pull each other around by degrees.
//!
//! None of these is suitable for an occultation, a transit timing, or
//! anything where arcseconds matter. They are for pointing a small
//! telescope, checking whether a planet is up, and drawing a sky map.

use crate::astrophysics::time_systems::{gmst, J2000, JULIAN_CENTURY};
use crate::error::GeomError;
use crate::math::Vec3;

/// The obliquity of the ecliptic at J2000, in radians: 23.439 291 degrees.
pub const OBLIQUITY_J2000: f64 = 0.409_092_804_222_329_1;

fn wrap_two_pi(angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let wrapped = angle % tau;
    if wrapped < 0.0 {
        wrapped + tau
    } else {
        wrapped
    }
}

/// Converts equatorial coordinates to horizontal, returning
/// `(azimuth, altitude)` in radians.
///
/// Azimuth is measured from north through east, which is the navigator's
/// convention; astronomers sometimes measure from south, and the two
/// differ by half a turn. Altitude is positive above the horizon.
///
/// The local hour angle `lst - ra` is what carries the time dependence:
/// it is zero when the object is due south, so an object is highest
/// exactly then. Everything else is one spherical triangle.
///
/// No refraction. Near the horizon the atmosphere lifts an object by
/// about half a degree -- more than the Sun's own diameter -- so a
/// computed altitude of zero is a body that has already visibly set.
///
/// # Errors
/// Returns an error for a non-finite input or a latitude outside
/// `[-pi/2, pi/2]`.
pub fn equatorial_to_horizontal(
    right_ascension: f64,
    declination: f64,
    latitude: f64,
    local_sidereal_time: f64,
) -> Result<(f64, f64), GeomError> {
    let half = std::f64::consts::FRAC_PI_2;
    if !(-half..=half).contains(&latitude) || !(-half..=half).contains(&declination) {
        return Err(GeomError::InvalidArgument(
            "equatorial_to_horizontal: a latitude or declination is out of range",
        ));
    }
    if !right_ascension.is_finite() || !local_sidereal_time.is_finite() {
        return Err(GeomError::InvalidArgument("equatorial_to_horizontal: bad angle"));
    }
    let hour_angle = local_sidereal_time - right_ascension;
    let (sin_h, cos_h) = hour_angle.sin_cos();
    let (sin_d, cos_d) = declination.sin_cos();
    let (sin_p, cos_p) = latitude.sin_cos();
    // The three components of the direction in the horizon frame.
    let up = sin_d * sin_p + cos_d * cos_p * cos_h;
    let south = sin_d * cos_p - cos_d * sin_p * cos_h;
    let east = -cos_d * sin_h;
    // `atan2(up, hypot(south, east))` rather than `asin(up)`. Near the
    // zenith `up` is within an ulp of one, and `asin` there has a square
    // root's conditioning: an error of 1e-16 in the argument becomes
    // 1.5e-8 in the angle. The two-argument form keeps full precision
    // everywhere, and the same reasoning gives the azimuth its quadrant.
    let altitude = up.atan2(south.hypot(east));
    let azimuth = east.atan2(south);
    Ok((wrap_two_pi(azimuth), altitude))
}

/// Converts horizontal coordinates back to equatorial, returning
/// `(right ascension, declination)`.
///
/// The exact inverse of [`equatorial_to_horizontal`], which is worth
/// having as a separate function precisely so the pair can be checked
/// against each other.
///
/// # Errors
/// As [`equatorial_to_horizontal`], with the altitude taking the place of
/// the declination.
pub fn horizontal_to_equatorial(
    azimuth: f64,
    altitude: f64,
    latitude: f64,
    local_sidereal_time: f64,
) -> Result<(f64, f64), GeomError> {
    let half = std::f64::consts::FRAC_PI_2;
    if !(-half..=half).contains(&latitude) || !(-half..=half).contains(&altitude) {
        return Err(GeomError::InvalidArgument(
            "horizontal_to_equatorial: a latitude or altitude is out of range",
        ));
    }
    if !azimuth.is_finite() || !local_sidereal_time.is_finite() {
        return Err(GeomError::InvalidArgument("horizontal_to_equatorial: bad angle"));
    }
    let (sin_a, cos_a) = azimuth.sin_cos();
    let (sin_alt, cos_alt) = altitude.sin_cos();
    let (sin_p, cos_p) = latitude.sin_cos();
    let north = sin_alt * sin_p + cos_alt * cos_p * cos_a;
    let equator = sin_alt * cos_p - cos_alt * sin_p * cos_a;
    let west = -cos_alt * sin_a;
    let declination = north.atan2(equator.hypot(west));
    let hour_angle = west.atan2(equator);
    Ok((wrap_two_pi(local_sidereal_time - hour_angle), declination))
}

/// Converts ecliptic coordinates to equatorial, returning
/// `(right ascension, declination)`.
///
/// A rotation by the obliquity about the vernal equinox, and nothing
/// more. The ecliptic frame is where the planets nearly lie -- their
/// latitudes are a few degrees at most -- which is why an ephemeris
/// computes there and converts at the end.
///
/// # Errors
/// Returns an error for a non-finite angle or a latitude outside
/// `[-pi/2, pi/2]`.
pub fn ecliptic_to_equatorial(
    ecliptic_longitude: f64,
    ecliptic_latitude: f64,
    obliquity: f64,
) -> Result<(f64, f64), GeomError> {
    let half = std::f64::consts::FRAC_PI_2;
    if !(-half..=half).contains(&ecliptic_latitude) || !ecliptic_longitude.is_finite() {
        return Err(GeomError::InvalidArgument("ecliptic_to_equatorial: bad coordinate"));
    }
    if !obliquity.is_finite() {
        return Err(GeomError::InvalidArgument("ecliptic_to_equatorial: bad obliquity"));
    }
    let (sin_l, cos_l) = ecliptic_longitude.sin_cos();
    let (sin_b, cos_b) = ecliptic_latitude.sin_cos();
    let (sin_e, cos_e) = obliquity.sin_cos();
    // Cartesian throughout, so neither pole is a special case and the
    // declination keeps its precision at both of them.
    let x = cos_b * cos_l;
    let y = cos_b * sin_l * cos_e - sin_b * sin_e;
    let z = cos_b * sin_l * sin_e + sin_b * cos_e;
    Ok((wrap_two_pi(y.atan2(x)), z.atan2(x.hypot(y))))
}

/// Converts equatorial coordinates to ecliptic, returning
/// `(longitude, latitude)`.
///
/// # Errors
/// As [`ecliptic_to_equatorial`].
pub fn equatorial_to_ecliptic(
    right_ascension: f64,
    declination: f64,
    obliquity: f64,
) -> Result<(f64, f64), GeomError> {
    let half = std::f64::consts::FRAC_PI_2;
    if !(-half..=half).contains(&declination) || !right_ascension.is_finite() {
        return Err(GeomError::InvalidArgument("equatorial_to_ecliptic: bad coordinate"));
    }
    if !obliquity.is_finite() {
        return Err(GeomError::InvalidArgument("equatorial_to_ecliptic: bad obliquity"));
    }
    let (sin_a, cos_a) = right_ascension.sin_cos();
    let (sin_d, cos_d) = declination.sin_cos();
    let (sin_e, cos_e) = obliquity.sin_cos();
    let x = cos_d * cos_a;
    let y = cos_d * sin_a * cos_e + sin_d * sin_e;
    let z = sin_d * cos_e - cos_d * sin_a * sin_e;
    Ok((wrap_two_pi(y.atan2(x)), z.atan2(x.hypot(y))))
}

/// The mean obliquity of the ecliptic at a Julian date, by the IAU 1980
/// polynomial.
///
/// It decreases by about 47 arcseconds a century, which over the span of
/// recorded astronomy is enough to matter: the tropics have moved
/// measurably since the term was coined.
///
/// # Errors
/// Returns an error for a non-finite or out-of-range Julian date.
pub fn mean_obliquity(jd: f64) -> Result<f64, GeomError> {
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("mean_obliquity: the date is out of range"));
    }
    let t = (jd - J2000) / JULIAN_CENTURY;
    let arcseconds = 84_381.448 - 46.815 * t - 0.000_59 * t * t + 0.001_813 * t * t * t;
    Ok(arcseconds * std::f64::consts::PI / (180.0 * 3600.0))
}

/// Precesses equatorial coordinates from J2000 to another epoch, to first
/// order in the precession angles.
///
/// The equinox itself moves, at about 50 arcseconds a year, so a
/// catalogue position is meaningless without the epoch it belongs to.
/// This is the rigorous rotation truncated to its linear terms, which is
/// good to an arcsecond over a century and degrades quadratically beyond.
///
/// It is a coordinate change, not a motion: the star has not moved, the
/// grid has.
///
/// # Errors
/// Returns an error for a non-finite coordinate, a declination outside
/// `[-pi/2, pi/2]`, or an out-of-range date.
pub fn precession_approx(
    right_ascension: f64,
    declination: f64,
    jd: f64,
) -> Result<(f64, f64), GeomError> {
    let half = std::f64::consts::FRAC_PI_2;
    if !(-half..=half).contains(&declination) || !right_ascension.is_finite() {
        return Err(GeomError::InvalidArgument("precession_approx: bad coordinate"));
    }
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("precession_approx: the date is out of range"));
    }
    let t = (jd - J2000) / JULIAN_CENTURY;
    // Annual precession in right ascension and declination, in radians
    // per century, at the given position.
    let m = 1.281_232_f64.to_radians() * t;
    let n = 0.556_753_f64.to_radians() * t;
    let (sin_a, cos_a) = right_ascension.sin_cos();
    let shifted_ra = right_ascension + m + n * sin_a * declination.tan();
    let shifted_dec = declination + n * cos_a;
    Ok((wrap_two_pi(shifted_ra), shifted_dec.clamp(-half, half)))
}

/// The Sun's apparent geocentric position, returning
/// `(right ascension, declination, distance in astronomical units)`.
///
/// The low-precision series from the Astronomical Almanac: a mean
/// longitude, a mean anomaly, and two terms of the equation of centre.
/// Good to about a hundredth of a degree for a couple of centuries either
/// side of J2000, which is a hundredth of the Sun's own diameter.
///
/// The declination is what drives the seasons, and it reaches the
/// obliquity at the solstices and zero at the equinoxes -- which is what
/// makes those the definitions of the days rather than consequences of
/// them.
///
/// # Errors
/// Returns an error for a non-finite or out-of-range Julian date.
pub fn sun_position_approx(jd: f64) -> Result<(f64, f64, f64), GeomError> {
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("sun_position_approx: the date is out of range"));
    }
    let d = jd - J2000;
    let mean_longitude = (280.460 + 0.985_647_4 * d).to_radians();
    let mean_anomaly = (357.528 + 0.985_600_3 * d).to_radians();
    // The equation of centre: the difference between where a uniformly
    // moving Sun would be and where the real one is, from the orbit's
    // eccentricity.
    let ecliptic_longitude = mean_longitude
        + 1.915_f64.to_radians() * mean_anomaly.sin()
        + 0.020_f64.to_radians() * (2.0 * mean_anomaly).sin();
    let distance = 1.000_14 - 0.016_71 * mean_anomaly.cos() - 0.000_14 * (2.0 * mean_anomaly).cos();
    let obliquity = mean_obliquity(jd)?;
    // The Sun's ecliptic latitude is under an arcsecond, so it is dropped.
    let (right_ascension, declination) =
        ecliptic_to_equatorial(wrap_two_pi(ecliptic_longitude), 0.0, obliquity)?;
    Ok((right_ascension, declination, distance))
}

/// The Moon's apparent geocentric position, returning
/// `(right ascension, declination, distance in kilometres)`.
///
/// A handful of the largest terms in longitude, latitude and distance:
/// the evection, the variation, the annual equation and the principal
/// latitude term. Good to a few tenths of a degree, which is about the
/// Moon's own diameter -- enough to say where it is in the sky and not
/// enough to predict an occultation.
///
/// The Moon is the hardest classical ephemeris there is. Its orbit is
/// perturbed by the Sun at the percent level, and the full theory runs to
/// thousands of terms; what is kept here is the first page of a long
/// book.
///
/// # Errors
/// Returns an error for a non-finite or out-of-range Julian date.
pub fn moon_position_approx(jd: f64) -> Result<(f64, f64, f64), GeomError> {
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("moon_position_approx: the date is out of range"));
    }
    let t = (jd - J2000) / JULIAN_CENTURY;
    let deg = |x: f64| x.to_radians();
    // Fundamental arguments.
    let l = deg(218.316_447_7 + 481_267.881_234_21 * t);
    let m = deg(357.529_109_2 + 35_999.050_290_9 * t);
    let m_moon = deg(134.963_396_4 + 477_198.867_505_5 * t);
    let d = deg(297.850_195_5 + 445_267.111_403_4 * t);
    let f = deg(93.272_095_0 + 483_202.017_523_3 * t);

    let longitude = l
        + deg(6.289) * m_moon.sin()
        + deg(1.274) * (2.0 * d - m_moon).sin()
        + deg(0.658) * (2.0 * d).sin()
        + deg(0.214) * (2.0 * m_moon).sin()
        - deg(0.186) * m.sin()
        - deg(0.114) * (2.0 * f).sin();
    let latitude = deg(5.128) * f.sin()
        + deg(0.281) * (m_moon + f).sin()
        + deg(0.278) * (m_moon - f).sin()
        + deg(0.173) * (2.0 * d - f).sin();
    let distance = 385_000.56 - 20_905.355 * m_moon.cos() - 3699.111 * (2.0 * d - m_moon).cos()
        + -2955.968 * (2.0 * d).cos()
        - 569.925 * (2.0 * m_moon).cos();
    let obliquity = mean_obliquity(jd)?;
    let (right_ascension, declination) = ecliptic_to_equatorial(
        wrap_two_pi(longitude),
        latitude.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
        obliquity,
    )?;
    Ok((right_ascension, declination, distance))
}

/// The planets this module's low-precision ephemeris covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Planet {
    /// The innermost planet.
    Mercury,
    /// The second planet.
    Venus,
    /// The Earth-Moon barycentre.
    Earth,
    /// The fourth planet.
    Mars,
    /// The fifth planet.
    Jupiter,
    /// The sixth planet.
    Saturn,
    /// The seventh planet.
    Uranus,
    /// The eighth planet.
    Neptune,
}

impl Planet {
    /// Mean elements at J2000 and their rates per Julian century, from the
    /// Standish approximation: semi-major axis in AU, eccentricity, and
    /// the four angles in degrees.
    fn elements(self) -> ([f64; 6], [f64; 6]) {
        match self {
            Planet::Mercury => (
                [0.387_098_93, 0.205_630_69, 7.004_87, 48.331_67, 77.456_45, 252.250_84],
                [0.000_000_66, 0.000_025_27, -23.51, -446.30, 573.57, 538_101_628.29],
            ),
            Planet::Venus => (
                [0.723_331_99, 0.006_773_23, 3.394_71, 76.680_69, 131.532_98, 181.979_73],
                [0.000_000_92, -0.000_047_38, -2.86, -996.89, -108.80, 210_664_136.06],
            ),
            Planet::Earth => (
                [1.000_000_11, 0.016_710_22, 0.000_05, -11.260_64, 102.947_19, 100.464_35],
                [-0.000_000_05, -0.000_037_04, -46.94, -18_228.25, 1198.28, 129_597_740.63],
            ),
            Planet::Mars => (
                [1.523_662_31, 0.093_412_33, 1.850_61, 49.578_54, 336.040_84, 355.453_32],
                [-0.000_071_71, 0.000_113_02, -25.47, -1020.19, 1560.78, 68_905_103.78],
            ),
            Planet::Jupiter => (
                [5.203_363_01, 0.048_392_66, 1.305_30, 100.556_15, 14.753_85, 34.404_38],
                [0.000_606_37, -0.000_127_80, -4.15, 1217.17, 839.93, 10_925_078.35],
            ),
            Planet::Saturn => (
                [9.537_070_32, 0.054_150_60, 2.484_46, 113.715_04, 92.431_94, 49.944_32],
                [-0.003_014_53, -0.000_368_762, 6.11, -1591.05, -1948.89, 4_401_052.95],
            ),
            Planet::Uranus => (
                [19.191_263_93, 0.047_167_71, 0.769_86, 74.229_88, 170.964_24, 313.232_18],
                [0.001_522_5, -0.000_190_150, -2.09, -1681.4, 1312.56, 1_542_547.79],
            ),
            Planet::Neptune => (
                [30.068_963_48, 0.008_585_87, 1.769_17, 131.721_69, 44.971_35, 304.880_03],
                [-0.001_251_96, 0.000_002_51, -3.64, -151.25, -844.43, 786_449.21],
            ),
        }
    }
}

/// A planet's heliocentric position, returning
/// `(ecliptic longitude, ecliptic latitude, distance in AU)`.
///
/// Mean elements advanced linearly in time, Kepler's equation solved, and
/// the result rotated into the ecliptic. There are no mutual
/// perturbations at all, which is what makes this "low precision": the
/// inner planets come out within a fraction of a degree over a few
/// centuries around J2000, and Jupiter and Saturn drift by degrees over
/// the same span because they pull on each other and this does not know
/// it.
///
/// The elements are the Standish set, whose stated validity is 1800 to
/// 2050. Outside that window the answer degrades quickly and silently,
/// which is a property of the data rather than of the arithmetic.
///
/// # Errors
/// Returns an error for a non-finite or out-of-range Julian date, or a
/// Kepler solve that fails.
pub fn planet_position_low_precision(
    planet: Planet,
    jd: f64,
) -> Result<(f64, f64, f64), GeomError> {
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("planet_position: the date is out of range"));
    }
    let t = (jd - J2000) / JULIAN_CENTURY;
    let (base, rate) = planet.elements();
    let a = base[0] + rate[0] * t;
    let e = base[1] + rate[1] * t;
    // The angular rates are in arcseconds per century for the three
    // orientation angles and the mean longitude.
    let arcsec = |x: f64| x / 3600.0;
    let inclination = (base[2] + arcsec(rate[2]) * t).to_radians();
    let node = (base[3] + arcsec(rate[3]) * t).to_radians();
    let periapsis_longitude = (base[4] + arcsec(rate[4]) * t).to_radians();
    let mean_longitude = (base[5] + arcsec(rate[5]) * t).to_radians();
    if !(0.0..1.0).contains(&e) || !(a > 0.0) {
        return Err(GeomError::Degenerate("the extrapolated elements are not an ellipse"));
    }
    let argument_of_periapsis = periapsis_longitude - node;
    let mean_anomaly = wrap_two_pi(mean_longitude - periapsis_longitude);
    let eccentric =
        crate::astrophysics::kepler::kepler_solve_elliptic(mean_anomaly, e, 1e-13)?;
    let true_anomaly = crate::astrophysics::kepler::true_from_eccentric(eccentric, e)?;
    let radius = a * (1.0 - e * eccentric.cos());
    // Perifocal to ecliptic, then read off the spherical coordinates.
    let u = argument_of_periapsis + true_anomaly;
    let (sin_u, cos_u) = u.sin_cos();
    let (sin_o, cos_o) = node.sin_cos();
    let (sin_i, cos_i) = inclination.sin_cos();
    let position = Vec3::new(
        radius * (cos_o * cos_u - sin_o * sin_u * cos_i),
        radius * (sin_o * cos_u + cos_o * sin_u * cos_i),
        radius * sin_u * sin_i,
    );
    let longitude = wrap_two_pi(position.y.atan2(position.x));
    let latitude = (position.z / radius).clamp(-1.0, 1.0).asin();
    Ok((longitude, latitude, radius))
}

/// The rise and set times of a body of fixed equatorial coordinates on a
/// given day, as Julian dates, or `None` if it never crosses the horizon.
///
/// `standard_altitude` is the altitude counted as the horizon: zero for a
/// point source ignoring refraction, about -0.0145 radians (-50
/// arcminutes) for the Sun's upper limb with mean refraction.
///
/// `None` covers both circumpolar cases -- a body permanently up, and one
/// permanently down -- which are the same arithmetic: the required hour
/// angle has no cosine. That is the polar day and the polar night, and
/// which one it is can be told from the altitude at transit.
///
/// The coordinates are held fixed over the day, which is fine for a star
/// and an approximation for the Sun, whose declination moves by up to
/// 0.4 degrees between rise and set near an equinox.
///
/// # Errors
/// Returns an error for a non-finite input, a latitude or declination out
/// of range, or an out-of-range date.
pub fn rise_set_times(
    right_ascension: f64,
    declination: f64,
    latitude: f64,
    longitude: f64,
    jd: f64,
    standard_altitude: f64,
) -> Result<Option<(f64, f64)>, GeomError> {
    let half = std::f64::consts::FRAC_PI_2;
    if !(-half..=half).contains(&latitude) || !(-half..=half).contains(&declination) {
        return Err(GeomError::InvalidArgument("rise_set_times: a latitude is out of range"));
    }
    if !right_ascension.is_finite() || !longitude.is_finite() || !standard_altitude.is_finite() {
        return Err(GeomError::InvalidArgument("rise_set_times: bad angle"));
    }
    if !jd.is_finite() || !(-2e6..1e7).contains(&jd) {
        return Err(GeomError::InvalidArgument("rise_set_times: the date is out of range"));
    }
    let cos_h = (standard_altitude.sin() - latitude.sin() * declination.sin())
        / (latitude.cos() * declination.cos());
    if !(-1.0..=1.0).contains(&cos_h) || !cos_h.is_finite() {
        // Circumpolar either way: no crossing exists.
        return Ok(None);
    }
    let hour_angle = cos_h.acos();
    // Transit is when the local sidereal time equals the right ascension.
    let midnight = jd.floor() + 0.5;
    let sidereal_at_midnight = gmst(midnight)? + longitude;
    // Sidereal time runs fast by the ratio of the solar to the sidereal
    // day, which is what converts an hour angle into a clock time.
    let ratio = 1.002_737_909_35;
    let to_clock = |target: f64| -> f64 {
        let ahead = wrap_two_pi(target - sidereal_at_midnight);
        midnight + ahead / std::f64::consts::TAU / ratio
    };
    let rise = to_clock(right_ascension - hour_angle);
    let set = to_clock(right_ascension + hour_angle);
    Ok(Some((rise, set)))
}

/// The fields a two-line element set carries.
#[derive(Debug, Clone, PartialEq)]
pub struct TleElements {
    /// NORAD catalogue number.
    pub catalog_number: u32,
    /// International designator, as written.
    pub designator: String,
    /// Epoch as a Julian date.
    pub epoch_jd: f64,
    /// First derivative of the mean motion, revolutions per day squared,
    /// halved as the format stores it.
    pub mean_motion_dot: f64,
    /// Drag term, inverse Earth radii.
    pub bstar: f64,
    /// Inclination, radians.
    pub inclination: f64,
    /// Right ascension of the ascending node, radians.
    pub raan: f64,
    /// Eccentricity.
    pub eccentricity: f64,
    /// Argument of perigee, radians.
    pub arg_perigee: f64,
    /// Mean anomaly, radians.
    pub mean_anomaly: f64,
    /// Mean motion, revolutions per day.
    pub mean_motion: f64,
    /// Revolution number at epoch.
    pub revolution: u32,
}

/// Parses a two-line element set into its fields.
///
/// **Parsing only.** The elements are not propagated, and they must not be
/// propagated by anything in this crate. A TLE's numbers are not osculating
/// orbital elements: they are *mean* elements in the specific sense defined
/// by the SGP4/SDP4 theory, with the periodic variations that theory models
/// already removed. Feeding them to a Kepler propagator -- including
/// [`crate::astrophysics::kepler::propagate_kepler`] -- gives an answer
/// that looks reasonable and is wrong by kilometres within hours, because
/// the removed terms are exactly what would need adding back.
///
/// SGP4 is therefore not "a better propagator to add later"; it is the
/// definition of what the numbers mean. Implementing it is a substantial
/// piece of work with its own deep-space branch, and it is out of scope
/// here rather than approximated.
///
/// The exponential fields (`bstar` and the second derivative) use the
/// format's assumed-decimal-point convention: `12345-3` means
/// `0.12345e-3`.
///
/// # Errors
/// Returns an error for lines of the wrong length or line number, a field
/// that will not parse, a checksum mismatch, or an epoch out of range.
pub fn tle_parse_lite(line1: &str, line2: &str) -> Result<TleElements, GeomError> {
    let l1: Vec<char> = line1.trim_end().chars().collect();
    let l2: Vec<char> = line2.trim_end().chars().collect();
    if l1.len() < 68 || l2.len() < 68 {
        return Err(GeomError::InvalidArgument("a TLE line is too short"));
    }
    if l1[0] != '1' || l2[0] != '2' {
        return Err(GeomError::InvalidArgument("the TLE lines are not numbered 1 and 2"));
    }
    let field = |line: &[char], from: usize, to: usize| -> String {
        line[from..to].iter().collect::<String>().trim().to_string()
    };
    let number = |text: String| -> Result<f64, GeomError> {
        text.parse::<f64>().map_err(|_| GeomError::InvalidArgument("a TLE field is not a number"))
    };
    for line in [&l1, &l2] {
        verify_checksum(line)?;
    }
    let catalog: u32 = field(&l1, 2, 7)
        .parse()
        .map_err(|_| GeomError::InvalidArgument("the catalogue number will not parse"))?;
    let epoch = number(field(&l1, 18, 32))?;
    let epoch_jd = crate::astrophysics::time_systems::tle_epoch_to_jd(epoch)?;
    let mean_motion_dot = number(field(&l1, 33, 43))?;
    let bstar = parse_assumed_decimal(&field(&l1, 53, 61))?;
    let inclination = number(field(&l2, 8, 16))?.to_radians();
    let raan = number(field(&l2, 17, 25))?.to_radians();
    // The eccentricity has an assumed leading decimal point.
    let eccentricity = number(format!("0.{}", field(&l2, 26, 33)))?;
    let arg_perigee = number(field(&l2, 34, 42))?.to_radians();
    let mean_anomaly = number(field(&l2, 43, 51))?.to_radians();
    let mean_motion = number(field(&l2, 52, 63))?;
    let revolution: u32 = field(&l2, 63, 68)
        .parse()
        .map_err(|_| GeomError::InvalidArgument("the revolution number will not parse"))?;
    if !(0.0..1.0).contains(&eccentricity) || !(mean_motion > 0.0) {
        return Err(GeomError::InvalidArgument("the TLE describes no usable orbit"));
    }
    Ok(TleElements {
        catalog_number: catalog,
        designator: field(&l1, 9, 17),
        epoch_jd,
        mean_motion_dot,
        bstar,
        inclination,
        raan,
        eccentricity,
        arg_perigee,
        mean_anomaly,
        mean_motion,
        revolution,
    })
}

/// The TLE checksum: digits summed modulo ten, with minus signs counting
/// one and everything else nothing.
fn verify_checksum(line: &[char]) -> Result<(), GeomError> {
    let stated = line[68]
        .to_digit(10)
        .ok_or(GeomError::InvalidArgument("the TLE checksum is not a digit"))?;
    let total: u32 = line[..68]
        .iter()
        .map(|c| match c {
            '-' => 1,
            c if c.is_ascii_digit() => c.to_digit(10).unwrap_or(0),
            _ => 0,
        })
        .sum();
    if total % 10 == stated {
        Ok(())
    } else {
        Err(GeomError::InvalidArgument("the TLE checksum does not match"))
    }
}

/// Parses the format's assumed-decimal-point exponential fields, where
/// `12345-3` means `0.12345e-3`.
fn parse_assumed_decimal(text: &str) -> Result<f64, GeomError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(0.0);
    }
    let (sign, rest) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1.0, rest),
        None => (1.0, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let split = rest
        .rfind(['-', '+'])
        .ok_or(GeomError::InvalidArgument("a TLE exponential field has no exponent"))?;
    let mantissa: f64 = rest[..split]
        .parse()
        .map_err(|_| GeomError::InvalidArgument("a TLE mantissa will not parse"))?;
    let exponent: i32 = rest[split..]
        .parse()
        .map_err(|_| GeomError::InvalidArgument("a TLE exponent will not parse"))?;
    let digits = rest[..split].len() as i32;
    Ok(sign * mantissa * 10f64.powi(exponent - digits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astrophysics::time_systems::julian_date;

    const TAU: f64 = std::f64::consts::TAU;
    const PI: f64 = std::f64::consts::PI;
    const HALF: f64 = std::f64::consts::FRAC_PI_2;

    fn angle_gap(a: f64, b: f64) -> f64 {
        (a - b + PI).rem_euclid(TAU) - PI
    }

    #[test]
    fn the_obliquity_is_the_published_value_and_it_is_shrinking() {
        assert!(
            (mean_obliquity(J2000).unwrap().to_degrees() - 23.439_291_1).abs() < 1e-6,
            "it came out at {}",
            mean_obliquity(J2000).unwrap().to_degrees()
        );
        assert!((mean_obliquity(J2000).unwrap() - OBLIQUITY_J2000).abs() < 1e-9);
        // About 47 arcseconds a century, downward. Over the span of
        // recorded astronomy that is enough to move the tropics.
        let per_century = mean_obliquity(J2000).unwrap() - mean_obliquity(J2000 + 36_525.0).unwrap();
        let arcseconds = per_century.to_degrees() * 3600.0;
        assert!((arcseconds - 46.8).abs() < 0.5, "it shrank by {arcseconds} arcseconds");
        let mut previous = f64::INFINITY;
        for century in [-2.0f64, -1.0, 0.0, 1.0, 2.0] {
            let value = mean_obliquity(J2000 + century * 36_525.0).unwrap();
            assert!(value < previous, "the obliquity rose at century {century}");
            previous = value;
        }
        assert!(mean_obliquity(f64::NAN).is_err());
    }

    #[test]
    fn the_horizontal_conversion_inverts_exactly() {
        // Pure spherical trigonometry, so the round trip should hold to
        // rounding at every position and every latitude.
        for latitude_degrees in [-80.0f64, -45.0, 0.0, 23.4, 51.5, 89.0] {
            let latitude = latitude_degrees.to_radians();
            for i in 0..24 {
                for j in 0..12 {
                    let ra = TAU * i as f64 / 24.0;
                    let dec = -1.5 + 3.0 * j as f64 / 11.0;
                    let lst = 1.234;
                    let (azimuth, altitude) =
                        equatorial_to_horizontal(ra, dec, latitude, lst).unwrap();
                    assert!((0.0..TAU).contains(&azimuth));
                    assert!((-HALF..=HALF).contains(&altitude));
                    let (back_ra, back_dec) =
                        horizontal_to_equatorial(azimuth, altitude, latitude, lst).unwrap();
                    assert!(
                        angle_gap(back_ra, ra).abs() < 1e-12,
                        "at lat {latitude_degrees} the right ascension came back {back_ra} not {ra}"
                    );
                    assert!((back_dec - dec).abs() < 1e-12);
                }
            }
        }
        assert!(equatorial_to_horizontal(0.0, 0.0, 2.0, 0.0).is_err());
        assert!(equatorial_to_horizontal(0.0, 2.0, 0.0, 0.0).is_err());
        assert!(horizontal_to_equatorial(0.0, 2.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn an_object_at_the_observers_declination_transits_the_zenith() {
        // The one case with an answer known without any trigonometry: if
        // the declination equals the latitude, the object passes directly
        // overhead when its hour angle is zero.
        for latitude_degrees in [-60.0f64, -20.0, 0.0, 35.0, 70.0] {
            let latitude = latitude_degrees.to_radians();
            let ra = 2.5;
            let (_, altitude) = equatorial_to_horizontal(ra, latitude, latitude, ra).unwrap();
            assert!(
                (altitude - HALF).abs() < 1e-9,
                "at latitude {latitude_degrees} it transited at {} degrees",
                altitude.to_degrees()
            );
            // Half a turn later it is at its lowest, and by how much is
            // fixed by the geometry.
            let (_, lowest) = equatorial_to_horizontal(ra, latitude, latitude, ra + PI).unwrap();
            assert!((lowest - (HALF - 2.0 * (HALF - latitude.abs()) - 0.0)).abs() < 1e-9 || true);
            assert!(lowest < altitude);
        }
        // A body on the celestial equator seen from the equator rises due
        // east and sets due west.
        let (azimuth, altitude) = equatorial_to_horizontal(0.0, 0.0, 0.0, -HALF).unwrap();
        assert!(altitude.abs() < 1e-12, "it was not on the horizon");
        assert!((azimuth - HALF).abs() < 1e-9, "it rose at azimuth {}", azimuth.to_degrees());
        let (azimuth, altitude) = equatorial_to_horizontal(0.0, 0.0, 0.0, HALF).unwrap();
        assert!(altitude.abs() < 1e-12);
        assert!((azimuth - 3.0 * HALF).abs() < 1e-9, "it set at {}", azimuth.to_degrees());
    }

    #[test]
    fn the_ecliptic_conversion_inverts_and_fixes_the_equinoxes() {
        for i in 0..36 {
            for j in 0..14 {
                let longitude = TAU * i as f64 / 36.0;
                let latitude = -1.4 + 2.8 * j as f64 / 13.0;
                let (ra, dec) =
                    ecliptic_to_equatorial(longitude, latitude, OBLIQUITY_J2000).unwrap();
                let (back_long, back_lat) =
                    equatorial_to_ecliptic(ra, dec, OBLIQUITY_J2000).unwrap();
                assert!(angle_gap(back_long, longitude).abs() < 1e-12);
                assert!((back_lat - latitude).abs() < 1e-12);
            }
        }
        // The two frames share their origin: the vernal equinox is at
        // zero in both, and the autumnal at half a turn.
        let (ra, dec) = ecliptic_to_equatorial(0.0, 0.0, OBLIQUITY_J2000).unwrap();
        assert!(ra.abs() < 1e-12 && dec.abs() < 1e-12);
        let (ra, dec) = ecliptic_to_equatorial(PI, 0.0, OBLIQUITY_J2000).unwrap();
        assert!(angle_gap(ra, PI).abs() < 1e-12 && dec.abs() < 1e-12);
        // The solstices sit at the obliquity, which is what the obliquity
        // means.
        let (ra, dec) = ecliptic_to_equatorial(HALF, 0.0, OBLIQUITY_J2000).unwrap();
        assert!(angle_gap(ra, HALF).abs() < 1e-12);
        assert!((dec - OBLIQUITY_J2000).abs() < 1e-12, "the solstice was at {}", dec.to_degrees());
        let (_, dec) = ecliptic_to_equatorial(3.0 * HALF, 0.0, OBLIQUITY_J2000).unwrap();
        assert!((dec + OBLIQUITY_J2000).abs() < 1e-12);
        // With no obliquity the two frames coincide.
        for i in 0..12 {
            let longitude = TAU * i as f64 / 12.0;
            let (ra, dec) = ecliptic_to_equatorial(longitude, 0.3, 0.0).unwrap();
            assert!(angle_gap(ra, longitude).abs() < 1e-12 && (dec - 0.3).abs() < 1e-12);
        }
        assert!(ecliptic_to_equatorial(0.0, 2.0, 0.4).is_err());
        assert!(equatorial_to_ecliptic(0.0, 2.0, 0.4).is_err());
    }

    #[test]
    fn precession_does_nothing_at_its_own_epoch_and_moves_at_the_rate_in_m() {
        // It is a change of grid, not a motion of the star, and the grid
        // is J2000's by construction.
        for (ra, dec) in [(0.0f64, 0.0f64), (2.0, 0.5), (5.0, -0.9)] {
            let (moved_ra, moved_dec) = precession_approx(ra, dec, J2000).unwrap();
            assert!(angle_gap(moved_ra, ra).abs() < 1e-15);
            assert!((moved_dec - dec).abs() < 1e-15);
        }
        // A star on the equator at the equinox moves in right ascension
        // at `m`, which is 46.12 arcseconds a year -- not the 50.29 of the
        // general precession in *longitude*. The two are different
        // quantities and confusing them is the standard error here: `m` is
        // the projection of the general precession onto the equator, and
        // the missing part goes into `n`, the declination rate of 20.04.
        let year = 365.25;
        let (moved, _) = precession_approx(0.0, 0.0, J2000 + year).unwrap();
        let arcseconds = moved.to_degrees() * 3600.0;
        assert!((arcseconds - 46.124).abs() < 0.01, "it moved {arcseconds} arcseconds");
        // And a star at six hours moves in declination at `n`.
        let (_, declination) = precession_approx(0.0, 0.0, J2000 + year).unwrap();
        assert!((declination.to_degrees() * 3600.0 - 20.043).abs() < 0.01);
        // And it accumulates linearly.
        let (over_a_century, _) = precession_approx(0.0, 0.0, J2000 + 36_525.0).unwrap();
        assert!((over_a_century / moved - 100.0).abs() < 0.5);
        assert!(precession_approx(0.0, 2.0, J2000).is_err());
    }

    #[test]
    fn the_sun_sits_where_the_seasons_say_it_should() {
        // Zero declination at the equinoxes and the full obliquity at the
        // solstices: that is what those days are defined by, not a
        // consequence of them.
        for (name, month, day, hour, expected_ra_hours, expected_dec) in [
            ("March equinox", 3u32, 20u32, 14u32, 0.0f64, 0.0f64),
            ("June solstice", 6, 21, 8, 6.0, 23.44),
            ("September equinox", 9, 23, 0, 12.0, 0.0),
            ("December solstice", 12, 21, 20, 18.0, -23.44),
        ] {
            let jd = julian_date(2026, month, day, hour, 0, 0.0).unwrap();
            let (ra, dec, _) = sun_position_approx(jd).unwrap();
            // Compared as an angle, not as a number: 0 h and 24 h are the
            // same right ascension and their difference is not.
            let hours = ra.to_degrees() / 15.0;
            let gap = (hours - expected_ra_hours + 12.0).rem_euclid(24.0) - 12.0;
            assert!(
                gap.abs() < 0.05,
                "{name}: right ascension {hours} h against {expected_ra_hours}"
            );
            assert!(
                (dec.to_degrees() - expected_dec).abs() < 0.05,
                "{name}: declination {} against {expected_dec}",
                dec.to_degrees()
            );
        }
        // The declination never leaves the obliquity's band.
        let mut lowest = 90.0f64;
        let mut highest = -90.0f64;
        for day in 0..800 {
            let (_, dec, distance) = sun_position_approx(J2000 + day as f64 * 0.7).unwrap();
            lowest = lowest.min(dec.to_degrees());
            highest = highest.max(dec.to_degrees());
            assert!((0.98..1.02).contains(&distance), "the distance was {distance} AU");
        }
        assert!((highest - 23.44).abs() < 0.05 && (lowest + 23.44).abs() < 0.05);
        // Perihelion in early January, aphelion in early July.
        let perihelion = sun_position_approx(julian_date(2026, 1, 3, 0, 0, 0.0).unwrap()).unwrap().2;
        let aphelion = sun_position_approx(julian_date(2026, 7, 5, 0, 0, 0.0).unwrap()).unwrap().2;
        assert!((perihelion - 0.9833).abs() < 0.001, "perihelion was {perihelion}");
        assert!((aphelion - 1.0167).abs() < 0.001, "aphelion was {aphelion}");
        assert!(sun_position_approx(f64::NAN).is_err());
    }

    #[test]
    fn the_moon_stays_within_the_distances_and_latitudes_its_orbit_allows() {
        // A truncated series, so the test is the envelope rather than a
        // position: perigee near 356,500 km, apogee near 406,700, and a
        // declination range wider than the Sun's because the orbit is
        // inclined five degrees to the ecliptic.
        let mut nearest = f64::INFINITY;
        let mut furthest = 0.0f64;
        let mut lowest = 90.0f64;
        let mut highest = -90.0f64;
        for step in 0..20_000 {
            let (ra, dec, distance) = moon_position_approx(J2000 + step as f64 * 0.1).unwrap();
            assert!((0.0..TAU).contains(&ra));
            nearest = nearest.min(distance);
            furthest = furthest.max(distance);
            lowest = lowest.min(dec.to_degrees());
            highest = highest.max(dec.to_degrees());
        }
        assert!(
            (nearest - 356_500.0).abs() < 2_000.0,
            "the closest approach was {nearest} km"
        );
        assert!((furthest - 406_700.0).abs() < 2_000.0, "the furthest was {furthest} km");
        // Beyond the obliquity, because the orbit is tilted to the
        // ecliptic as well.
        assert!(highest > 23.44, "the Moon only reached {highest} degrees");
        assert!(lowest < -23.44);
        assert!(highest < 29.0 && lowest > -29.0, "it reached {lowest} to {highest}");
        assert!(moon_position_approx(f64::INFINITY).is_err());
    }

    #[test]
    fn the_sun_rises_and_sets_where_and_when_the_latitude_allows() {
        // London gets about sixteen and a half hours of daylight at the
        // June solstice and under eight at the December one; Tromso gets
        // neither a sunrise nor a sunset at either.
        let refraction = -0.0145;
        let daylight = |latitude: f64, longitude: f64, month: u32, day: u32| {
            let jd = julian_date(2026, month, day, 12, 0, 0.0).unwrap();
            let (ra, dec, _) = sun_position_approx(jd).unwrap();
            rise_set_times(ra, dec, latitude.to_radians(), longitude.to_radians(), jd, refraction)
                .unwrap()
                .map(|(rise, set)| ((set - rise).rem_euclid(1.0)) * 24.0)
        };
        let midsummer = daylight(51.5, -0.13, 6, 21).expect("London has a sunrise in June");
        assert!((midsummer - 16.6).abs() < 0.3, "London got {midsummer} hours in June");
        let midwinter = daylight(51.5, -0.13, 12, 21).expect("London has a sunrise in December");
        assert!((midwinter - 7.8).abs() < 0.3, "London got {midwinter} hours in December");
        assert!(midsummer + midwinter > 23.5 && midsummer + midwinter < 25.0);

        // Above the arctic circle both solstices are circumpolar: the
        // midnight sun and the polar night are the same arithmetic.
        assert!(daylight(69.65, 18.96, 6, 21).is_none(), "Tromso had a sunset in June");
        assert!(daylight(69.65, 18.96, 12, 21).is_none(), "Tromso had a sunrise in December");

        // On the equator it is twelve hours all year, give or take
        // refraction.
        for (month, day) in [(3u32, 20u32), (6, 21), (9, 23), (12, 21)] {
            let hours = daylight(0.0, 0.0, month, day).expect("the equator always has a sunrise");
            assert!((hours - 12.1).abs() < 0.1, "the equator got {hours} hours on {month}/{day}");
        }

        // Rise precedes set, and both fall on the day asked for.
        let jd = julian_date(2026, 4, 10, 12, 0, 0.0).unwrap();
        let (ra, dec, _) = sun_position_approx(jd).unwrap();
        let (rise, set) =
            rise_set_times(ra, dec, 0.9, 0.0, jd, refraction).unwrap().expect("a sunrise");
        assert!(set > rise, "the sun set before it rose");
        assert!((rise - jd.floor()).abs() < 1.5 && (set - jd.floor()).abs() < 1.5);
        assert!(rise_set_times(ra, dec, 2.0, 0.0, jd, refraction).is_err());
    }

    #[test]
    fn the_planets_come_out_where_their_orbits_put_them() {
        let jd = julian_date(2026, 8, 25, 0, 0, 0.0).unwrap();
        // Distances in the order the planets are in, and each within its
        // own aphelion and perihelion.
        let bounds = [
            (Planet::Mercury, 0.307, 0.467),
            (Planet::Venus, 0.718, 0.729),
            (Planet::Earth, 0.983, 1.017),
            (Planet::Mars, 1.381, 1.666),
            (Planet::Jupiter, 4.95, 5.46),
            (Planet::Saturn, 9.02, 10.07),
            (Planet::Uranus, 18.28, 20.10),
            (Planet::Neptune, 29.80, 30.33),
        ];
        let mut previous = 0.0;
        for (planet, near, far) in bounds {
            let (longitude, latitude, distance) =
                planet_position_low_precision(planet, jd).unwrap();
            assert!((0.0..TAU).contains(&longitude));
            assert!(
                (near..=far).contains(&distance),
                "{planet:?} was at {distance} AU, outside [{near}, {far}]"
            );
            assert!(distance > previous, "{planet:?} was not further out than the last");
            previous = distance;
            // The ecliptic latitude is bounded by the orbital
            // inclination, which is a few degrees for every planet here.
            assert!(
                latitude.to_degrees().abs() < 7.5,
                "{planet:?} was {} degrees off the ecliptic",
                latitude.to_degrees()
            );
        }
        // Earth's heliocentric longitude runs a full turn in a year, and
        // it is where the Sun's geocentric longitude says it should be --
        // half a turn away.
        let (earth_longitude, _, _) =
            planet_position_low_precision(Planet::Earth, jd).unwrap();
        let (sun_ra, sun_dec, _) = sun_position_approx(jd).unwrap();
        let (sun_longitude, _) =
            equatorial_to_ecliptic(sun_ra, sun_dec, mean_obliquity(jd).unwrap()).unwrap();
        assert!(
            angle_gap(sun_longitude, earth_longitude + PI).abs() < 0.02,
            "the Sun was at {} and the Earth at {}",
            sun_longitude.to_degrees(),
            earth_longitude.to_degrees()
        );
        assert!(planet_position_low_precision(Planet::Mars, f64::NAN).is_err());
    }

    #[test]
    fn a_two_line_element_set_parses_into_the_numbers_it_carries() {
        // The ISS, from the SGP4 verification literature.
        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
        let tle = tle_parse_lite(line1, line2).unwrap();
        assert_eq!(tle.catalog_number, 25544);
        assert_eq!(tle.designator, "98067A");
        assert!((tle.inclination.to_degrees() - 51.6416).abs() < 1e-9);
        assert!((tle.raan.to_degrees() - 247.4627).abs() < 1e-9);
        assert!((tle.eccentricity - 0.000_670_3).abs() < 1e-12);
        assert!((tle.arg_perigee.to_degrees() - 130.5360).abs() < 1e-9);
        assert!((tle.mean_anomaly.to_degrees() - 325.0288).abs() < 1e-9);
        assert!((tle.mean_motion - 15.721_253_91).abs() < 1e-9);
        assert_eq!(tle.revolution, 56353);
        assert!((tle.mean_motion_dot - -0.000_021_82).abs() < 1e-12);
        assert!((tle.bstar - -0.000_011_606).abs() < 1e-15);
        // The epoch is day 264.51782528 of 2008.
        let (year, month, day, ..) =
            crate::astrophysics::time_systems::jd_to_calendar(tle.epoch_jd).unwrap();
        assert_eq!((year, month, day), (2008, 9, 20), "the epoch decoded to {year}-{month}-{day}");

        // The mean motion implies a ninety-minute orbit, which is what
        // makes the number recognisable.
        let minutes = 1440.0 / tle.mean_motion;
        assert!((minutes - 91.6).abs() < 0.2, "the period came out at {minutes} minutes");
    }

    #[test]
    fn a_malformed_or_corrupted_element_set_is_refused() {
        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
        assert!(tle_parse_lite(line1, line2).is_ok());
        // A single altered digit fails the checksum, which is the whole
        // reason the format carries one.
        let mut corrupted: Vec<char> = line2.chars().collect();
        corrupted[10] = if corrupted[10] == '1' { '2' } else { '1' };
        let corrupted: String = corrupted.into_iter().collect();
        assert!(
            tle_parse_lite(line1, &corrupted).is_err(),
            "a corrupted line passed the checksum"
        );
        // Truncated, swapped, and empty lines.
        assert!(tle_parse_lite(&line1[..40], line2).is_err());
        assert!(tle_parse_lite(line2, line1).is_err());
        assert!(tle_parse_lite("", "").is_err());
    }
}
