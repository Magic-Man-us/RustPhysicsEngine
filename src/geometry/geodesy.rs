//! Geodesy on a reference ellipsoid.
//!
//! Vincenty's inverse and direct formulae (Vincenty, "Direct and
//! inverse solutions of geodesics on the ellipsoid", Survey Review
//! 1975) plus geodetic ↔ ECEF ↔ ENU coordinate conversions. Angles are
//! radians; distances and heights are meters.

use crate::error::SolveError;
use crate::math::Vec3;

/// Reference ellipsoid: semi-major axis a (m) and flattening f.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipsoid {
    pub a: f64,
    pub f: f64,
}

impl Ellipsoid {
    /// WGS-84: a = 6378137 m, f = 1/298.257223563.
    pub const WGS84: Ellipsoid = Ellipsoid { a: 6_378_137.0, f: 1.0 / 298.257_223_563 };

    /// Semi-minor axis b = a(1 − f).
    #[must_use]
    pub fn b(&self) -> f64 {
        self.a * (1.0 - self.f)
    }

    /// First eccentricity squared e² = f(2 − f).
    #[must_use]
    pub fn e_sq(&self) -> f64 {
        self.f * (2.0 - self.f)
    }
}

const VINCENTY_TOL: f64 = 1e-12;
const VINCENTY_MAX_ITER: usize = 200;

/// Vincenty inverse problem: geodesic distance (m) and forward/reverse
/// azimuths (rad) between two geodetic points.
///
/// Returns `NoConvergence` for the nearly antipodal cases where
/// Vincenty's lambda iteration fails, and distance 0 with azimuth 0
/// for coincident points.
pub fn vincenty_inverse(
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    e: &Ellipsoid,
) -> Result<(f64, f64, f64), SolveError> {
    let a = e.a;
    let b = e.b();
    let f = e.f;
    let l = lon2 - lon1;
    let u1 = ((1.0 - f) * lat1.tan()).atan();
    let u2 = ((1.0 - f) * lat2.tan()).atan();
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();

    let mut lambda = l;
    let mut iter = 0;
    let (sigma, sin_sigma, cos_sigma, cos_sq_alpha, cos_2sigma_m) = loop {
        let (sin_lambda, cos_lambda) = lambda.sin_cos();
        let sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();
        if sin_sigma == 0.0 {
            return Ok((0.0, 0.0, 0.0)); // coincident points
        }
        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        let sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
        let cos_2sigma_m = if cos_sq_alpha == 0.0 {
            0.0 // equatorial geodesic
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        };
        let c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));
        let lambda_prev = lambda;
        lambda = l
            + (1.0 - c)
                * f
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));
        iter += 1;
        if (lambda - lambda_prev).abs() < VINCENTY_TOL {
            break (sigma, sin_sigma, cos_sigma, cos_sq_alpha, cos_2sigma_m);
        }
        if iter >= VINCENTY_MAX_ITER {
            return Err(SolveError::NoConvergence {
                iters: iter,
                residual: (lambda - lambda_prev).abs(),
            });
        }
    };

    let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);
    let big_a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
    let delta_sigma = big_b
        * sin_sigma
        * (cos_2sigma_m
            + big_b / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                    - big_b / 6.0
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                        * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));
    let dist = b * big_a * (sigma - delta_sigma);

    let (sin_lambda, cos_lambda) = lambda.sin_cos();
    let az1 = (cos_u2 * sin_lambda).atan2(cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda);
    let az2 = (cos_u1 * sin_lambda).atan2(-sin_u1 * cos_u2 + cos_u1 * sin_u2 * cos_lambda);
    Ok((dist, az1, az2))
}

/// Vincenty direct problem: destination (lat2, lon2) and final azimuth
/// after traveling `dist` meters from (lat1, lon1) on initial azimuth
/// `az1`.
#[must_use]
pub fn vincenty_direct(
    lat1: f64,
    lon1: f64,
    az1: f64,
    dist: f64,
    e: &Ellipsoid,
) -> (f64, f64, f64) {
    let a = e.a;
    let b = e.b();
    let f = e.f;
    let (sin_az, cos_az) = az1.sin_cos();
    let tan_u1 = (1.0 - f) * lat1.tan();
    let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
    let sin_u1 = tan_u1 * cos_u1;
    let sigma1 = tan_u1.atan2(cos_az);
    let sin_alpha = cos_u1 * sin_az;
    let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
    let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);
    let big_a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let mut sigma = dist / (b * big_a);
    let mut sin_sigma;
    let mut cos_sigma;
    let mut cos_2sigma_m = (2.0 * sigma1 + sigma).cos();
    for _ in 0..VINCENTY_MAX_ITER {
        cos_2sigma_m = (2.0 * sigma1 + sigma).cos();
        let (s, c) = sigma.sin_cos();
        sin_sigma = s;
        cos_sigma = c;
        let delta_sigma = big_b
            * sin_sigma
            * (cos_2sigma_m
                + big_b / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                        - big_b / 6.0
                            * cos_2sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));
        let sigma_new = dist / (b * big_a) + delta_sigma;
        if (sigma_new - sigma).abs() < VINCENTY_TOL {
            sigma = sigma_new;
            break;
        }
        sigma = sigma_new;
    }
    let (s, c) = sigma.sin_cos();
    sin_sigma = s;
    cos_sigma = c;

    let tmp = sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_az;
    let lat2 = (sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_az)
        .atan2((1.0 - f) * (sin_alpha * sin_alpha + tmp * tmp).sqrt());
    let lambda = (sin_sigma * sin_az).atan2(cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_az);
    let c_coef = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));
    let l = lambda
        - (1.0 - c_coef)
            * f
            * sin_alpha
            * (sigma
                + c_coef * sin_sigma
                    * (cos_2sigma_m
                        + c_coef * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));
    let lon2 = lon1 + l;
    let az2 = sin_alpha.atan2(-tmp);
    (lat2, lon2, az2)
}

/// Geodetic (lat, lon, height) → Earth-centered Earth-fixed Cartesian.
#[must_use]
pub fn geodetic_to_ecef(lat: f64, lon: f64, h: f64, e: &Ellipsoid) -> Vec3 {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let n = e.a / (1.0 - e.e_sq() * sin_lat * sin_lat).sqrt();
    Vec3::new(
        (n + h) * cos_lat * cos_lon,
        (n + h) * cos_lat * sin_lon,
        (n * (1.0 - e.e_sq()) + h) * sin_lat,
    )
}

/// ECEF Cartesian → geodetic (lat, lon, height) by fixed-point
/// iteration on the latitude (converges to sub-millimeter in a few
/// steps).
#[must_use]
pub fn ecef_to_geodetic(p: Vec3, e: &Ellipsoid) -> (f64, f64, f64) {
    let lon = p.y.atan2(p.x);
    let rho = (p.x * p.x + p.y * p.y).sqrt();
    let e_sq = e.e_sq();
    if rho < 1e-9 {
        // On the polar axis.
        let lat = if p.z >= 0.0 { std::f64::consts::FRAC_PI_2 } else { -std::f64::consts::FRAC_PI_2 };
        let h = p.z.abs() - e.b();
        return (lat, lon, h);
    }
    let mut lat = (p.z / (rho * (1.0 - e_sq))).atan();
    let mut h = 0.0;
    for _ in 0..20 {
        let sin_lat = lat.sin();
        let n = e.a / (1.0 - e_sq * sin_lat * sin_lat).sqrt();
        h = rho / lat.cos() - n;
        let lat_new = (p.z / (rho * (1.0 - e_sq * n / (n + h)))).atan();
        if (lat_new - lat).abs() < 1e-14 {
            lat = lat_new;
            break;
        }
        lat = lat_new;
    }
    (lat, lon, h)
}

/// ECEF point → local East-North-Up coordinates relative to the given
/// geodetic reference.
#[must_use]
pub fn ecef_to_enu(p: Vec3, ref_lat: f64, ref_lon: f64, ref_h: f64, e: &Ellipsoid) -> Vec3 {
    let origin = geodetic_to_ecef(ref_lat, ref_lon, ref_h, e);
    let dx = p.x - origin.x;
    let dy = p.y - origin.y;
    let dz = p.z - origin.z;
    let (sin_lat, cos_lat) = ref_lat.sin_cos();
    let (sin_lon, cos_lon) = ref_lon.sin_cos();
    Vec3::new(
        -sin_lon * dx + cos_lon * dy,
        -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz,
        cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    fn deg(d: f64) -> f64 {
        d * PI / 180.0
    }

    #[test]
    fn test_vincenty_flinders_to_buninyong() {
        // Vincenty's classic test line (Flinders Peak -> Buninyong,
        // Australia): s = 54972.271 m.
        let lat1 = -deg(37.0 + 57.0 / 60.0 + 3.72030 / 3600.0);
        let lon1 = deg(144.0 + 25.0 / 60.0 + 29.52440 / 3600.0);
        let lat2 = -deg(37.0 + 39.0 / 60.0 + 10.15610 / 3600.0);
        let lon2 = deg(143.0 + 55.0 / 60.0 + 35.38390 / 3600.0);
        let (d, _, _) = vincenty_inverse(lat1, lon1, lat2, lon2, &Ellipsoid::WGS84).unwrap();
        assert!((d - 54_972.271).abs() < 0.01, "distance {d}");
    }

    #[test]
    fn test_vincenty_direct_inverse_roundtrip() {
        let e = Ellipsoid::WGS84;
        let (lat1, lon1) = (deg(48.8566), deg(2.3522)); // Paris
        let (lat2, lon2) = (deg(40.7128), -deg(74.0060)); // New York
        let (d, az1, _) = vincenty_inverse(lat1, lon1, lat2, lon2, &e).unwrap();
        let (rlat, rlon, _) = vincenty_direct(lat1, lon1, az1, d, &e);
        // Recovered point within 1e-6 m of the target.
        let (miss, _, _) = vincenty_inverse(rlat, rlon, lat2, lon2, &e).unwrap();
        assert!(miss < 1e-6, "roundtrip miss {miss} m");
    }

    #[test]
    fn test_vincenty_sphere_matches_great_circle() {
        // f = 0 reduces the geodesic to a great circle.
        let sphere = Ellipsoid { a: 6_371_000.0, f: 0.0 };
        let (lat1, lon1) = (deg(10.0), deg(20.0));
        let (lat2, lon2) = (deg(-30.0), deg(80.0));
        let (d, _, _) = vincenty_inverse(lat1, lon1, lat2, lon2, &sphere).unwrap();
        let gc = crate::geometry::great_circle_distance(6_371_000.0, lat1, lon1, lat2, lon2);
        assert!((d - gc).abs() < 1e-3, "vincenty {d} vs great circle {gc}");
    }

    #[test]
    fn test_vincenty_coincident_points() {
        let e = Ellipsoid::WGS84;
        let (d, az1, az2) = vincenty_inverse(deg(45.0), deg(9.0), deg(45.0), deg(9.0), &e).unwrap();
        assert_eq!((d, az1, az2), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_ecef_roundtrip() {
        let e = Ellipsoid::WGS84;
        for &(lat, lon, h) in &[
            (deg(0.0), deg(0.0), 0.0),
            (deg(45.0), deg(-120.0), 1500.0),
            (deg(-80.0), deg(170.0), -50.0),
            (deg(89.9), deg(10.0), 3000.0),
        ] {
            let p = geodetic_to_ecef(lat, lon, h, &e);
            let (rlat, rlon, rh) = ecef_to_geodetic(p, &e);
            assert!((rlat - lat).abs() < 1e-10, "lat {lat}");
            assert!((rlon - lon).abs() < 1e-12, "lon {lon}");
            assert!((rh - h).abs() < 1e-4, "h {h}: got {rh}");
        }
    }

    #[test]
    fn test_ecef_equator_prime_meridian() {
        let e = Ellipsoid::WGS84;
        let p = geodetic_to_ecef(0.0, 0.0, 0.0, &e);
        assert!((p.x - e.a).abs() < 1e-6 && p.y.abs() < 1e-6 && p.z.abs() < 1e-6);
    }

    #[test]
    fn test_enu_axes() {
        let e = Ellipsoid::WGS84;
        let (lat0, lon0, h0) = (deg(40.0), deg(-75.0), 100.0);
        // A point slightly above the reference is pure Up.
        let above = geodetic_to_ecef(lat0, lon0, h0 + 50.0, &e);
        let enu = ecef_to_enu(above, lat0, lon0, h0, &e);
        assert!(enu.x.abs() < 1e-6 && enu.y.abs() < 1e-6);
        assert!((enu.z - 50.0).abs() < 1e-6);
        // A point slightly north maps mostly to +North.
        let north = geodetic_to_ecef(lat0 + 1e-5, lon0, h0, &e);
        let enu_n = ecef_to_enu(north, lat0, lon0, h0, &e);
        assert!(enu_n.y > 1.0 && enu_n.x.abs() < 1e-3);
    }
}
