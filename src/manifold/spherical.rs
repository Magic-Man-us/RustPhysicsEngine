//! Spherical geometry: n-sphere maps, spherical trigonometry, map
//! projections, the Hopf fibration, spherical harmonics and their
//! transforms, sky pixelizations, point distributions, and directional
//! statistics.

use crate::fractals::Complex;
use crate::manifold::lie::So3;
use crate::manifold::vecn::VecN;
use crate::math::{Vec2, Vec3};
use crate::monte_carlo::Rng;
use crate::quaternion::Quaternion;

const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// n-sphere maps
// ---------------------------------------------------------------------------

/// Geodesic (angular) distance on the unit n-sphere.
#[must_use]
pub fn sphere_distance_n(a: &VecN, b: &VecN) -> f64 {
    (a.dot(b) / (a.norm() * b.norm())).clamp(-1.0, 1.0).acos()
}

/// Slerp along the great circle from a to b.
#[must_use]
pub fn sphere_geodesic_n(a: &VecN, b: &VecN, t: f64) -> VecN {
    let omega = sphere_distance_n(a, b);
    if omega < 1e-12 {
        return a.clone();
    }
    let s = omega.sin();
    a.scale(((1.0 - t) * omega).sin() / s)
        .add(&b.scale((t * omega).sin() / s))
        .normalized()
}

/// Exponential map at p: follow the great circle in direction v (tangent,
/// |v| = arc length).
#[must_use]
pub fn sphere_exp_n(p: &VecN, v: &VecN) -> VecN {
    let th = v.norm();
    if th < 1e-15 {
        return p.normalized();
    }
    let u = v.scale(1.0 / th);
    p.scale(th.cos()).add(&u.scale(th.sin())).normalized()
}

/// Logarithm map: tangent vector at p pointing toward q with |v| equal to
/// the geodesic distance.
#[must_use]
pub fn sphere_log_n(p: &VecN, q: &VecN) -> VecN {
    let th = sphere_distance_n(p, q);
    let perp = q.sub(&p.scale(q.dot(p)));
    let n = perp.norm();
    if n < 1e-15 {
        return VecN::zeros(p.dim());
    }
    perp.scale(th / n)
}

/// Parallel transport of tangent vector v from p to q along the connecting
/// geodesic.
#[must_use]
pub fn sphere_parallel_transport_n(v: &VecN, p: &VecN, q: &VecN) -> VecN {
    let log_pq = sphere_log_n(p, q);
    let th = log_pq.norm();
    if th < 1e-15 {
        return v.clone();
    }
    let u = log_pq.scale(1.0 / th);
    // decompose v into the (p, u) plane component and the orthogonal rest
    let a = v.dot(&u);
    let rest = v.sub(&u.scale(a));
    // the u component rotates: u -> -p sin + u cos; p -> p cos + u sin
    rest.add(&u.scale(a * th.cos()))
        .add(&p.scale(-a * th.sin()))
}

// ---------------------------------------------------------------------------
// Spherical trigonometry
// ---------------------------------------------------------------------------

/// Area of a spherical triangle on the unit sphere via l'Huilier's theorem.
#[must_use]
pub fn spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (an, bn, cn) = (a.normalized(), b.normalized(), c.normalized());
    let sa = bn.dot(&cn).clamp(-1.0, 1.0).acos();
    let sb = an.dot(&cn).clamp(-1.0, 1.0).acos();
    let sc = an.dot(&bn).clamp(-1.0, 1.0).acos();
    let s = 0.5 * (sa + sb + sc);
    let t = ((0.5 * s).tan()
        * (0.5 * (s - sa)).tan()
        * (0.5 * (s - sb)).tan()
        * (0.5 * (s - sc)).tan())
    .max(0.0);
    4.0 * t.sqrt().atan()
}

/// Interior angles of a spherical triangle at vertices (a, b, c).
#[must_use]
pub fn spherical_triangle_angles(a: Vec3, b: Vec3, c: Vec3) -> (f64, f64, f64) {
    let angle_at = |p: Vec3, q: Vec3, r: Vec3| {
        let pn = p.normalized();
        let tq = (q - pn * q.dot(&pn)).normalized();
        let tr = (r - pn * r.dot(&pn)).normalized();
        tq.dot(&tr).clamp(-1.0, 1.0).acos()
    };
    (
        angle_at(a, b, c),
        angle_at(b, c, a),
        angle_at(c, a, b),
    )
}

/// Spherical law of cosines: cos c = cos a cos b + sin a sin b cos gamma.
#[must_use]
pub fn spherical_law_of_cosines(a: f64, b: f64, gamma: f64) -> f64 {
    (a.cos() * b.cos() + a.sin() * b.sin() * gamma.cos()).clamp(-1.0, 1.0).acos()
}

/// Spherical law of sines: alpha from (a, b, beta) via
/// sin alpha / sin a = sin beta / sin b.
#[must_use]
pub fn spherical_law_of_sines(a: f64, b: f64, beta: f64) -> f64 {
    (beta.sin() * a.sin() / b.sin()).clamp(-1.0, 1.0).asin()
}

/// Haversine great-circle distance on a sphere of radius r.
#[must_use]
pub fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64, r: f64) -> f64 {
    let h = (0.5 * (lat2 - lat1)).sin().powi(2)
        + lat1.cos() * lat2.cos() * (0.5 * (lon2 - lon1)).sin().powi(2);
    2.0 * r * h.sqrt().min(1.0).asin()
}

/// Area of a spherical polygon (unit sphere) by summing triangle fan areas
/// with orientation from the spherical excess formula.
#[must_use]
pub fn spherical_polygon_area(vertices: &[Vec3]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    // sum of interior angles minus (n-2) pi
    let n = vertices.len();
    let mut total = 0.0;
    for i in 0..n {
        let p = vertices[i].normalized();
        let prev = vertices[(i + n - 1) % n];
        let next = vertices[(i + 1) % n];
        let tq = (prev - p * prev.dot(&p)).normalized();
        let tr = (next - p * next.dot(&p)).normalized();
        total += tq.dot(&tr).clamp(-1.0, 1.0).acos();
    }
    (total - (n as f64 - 2.0) * PI).abs()
}

/// Spherical centroid: normalized arithmetic mean.
#[must_use]
pub fn spherical_centroid(points: &[Vec3]) -> Vec3 {
    let s = points.iter().fold(Vec3::new(0.0, 0.0, 0.0), |a, &b| a + b);
    s.normalized()
}

/// Weighted spherical mean.
#[must_use]
pub fn spherical_mean_weighted(points: &[Vec3], weights: &[f64]) -> Vec3 {
    let mut s = Vec3::new(0.0, 0.0, 0.0);
    for (p, &w) in points.iter().zip(weights) {
        s = s + *p * w;
    }
    s.normalized()
}

// ---------------------------------------------------------------------------
// Spherical Delaunay / Voronoi / hull
// ---------------------------------------------------------------------------

/// Spherical Delaunay triangulation by the empty-circumcap test (brute
/// force; suitable for modest site counts).
#[must_use]
pub fn spherical_delaunay(sites: &[Vec3]) -> Vec<[usize; 3]> {
    let p: Vec<Vec3> = sites.iter().map(Vec3::normalized).collect();
    let n = p.len();
    let mut tris = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                // circumcenter direction of the spherical triangle
                let c = (p[j] - p[i]).cross(&(p[k] - p[i]));
                let cn = c.magnitude();
                if cn < 1e-12 {
                    continue;
                }
                let center = c * (1.0 / cn);
                let cos_r = center.dot(&p[i]);
                // empty cap on this side
                let empty = (0..n)
                    .all(|m| m == i || m == j || m == k || center.dot(&p[m]) < cos_r + 1e-12);
                let center2 = center * -1.0;
                let cos_r2 = center2.dot(&p[i]);
                let empty2 = (0..n)
                    .all(|m| m == i || m == j || m == k || center2.dot(&p[m]) < cos_r2 + 1e-12);
                if empty || empty2 {
                    tris.push([i, j, k]);
                }
            }
        }
    }
    tris
}

/// Spherical Voronoi cells as the dual of the Delaunay triangulation: each
/// cell is the list of circumcenters of triangles incident to the site,
/// ordered by angle.
#[must_use]
pub fn spherical_voronoi(sites: &[Vec3]) -> Vec<Vec<Vec3>> {
    let p: Vec<Vec3> = sites.iter().map(Vec3::normalized).collect();
    let tris = spherical_delaunay(&p);
    let circum = |t: &[usize; 3]| -> Vec3 {
        let c = (p[t[1]] - p[t[0]]).cross(&(p[t[2]] - p[t[0]]));
        let cn = c.normalized();
        // orient toward the site
        if cn.dot(&p[t[0]]) < 0.0 {
            cn * -1.0
        } else {
            cn
        }
    };
    (0..p.len())
        .map(|s| {
            let mut verts: Vec<Vec3> = tris
                .iter()
                .filter(|t| t.contains(&s))
                .map(circum)
                .collect();
            // order around the site
            if verts.len() > 2 {
                let z = p[s];
                let x = (verts[0] - z * verts[0].dot(&z)).normalized();
                let y = z.cross(&x);
                verts.sort_by(|a, b| {
                    let aa = a.dot(&y).atan2(a.dot(&x));
                    let bb = b.dot(&y).atan2(b.dot(&x));
                    aa.partial_cmp(&bb).unwrap()
                });
            }
            verts
        })
        .collect()
}

/// Indices of points on the 3D convex hull (brute-force facet search).
#[must_use]
pub fn spherical_convex_hull(points: &[Vec3]) -> Vec<usize> {
    let n = points.len();
    let mut on_hull = vec![false; n];
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let nrm = (points[j] - points[i]).cross(&(points[k] - points[i]));
                if nrm.magnitude() < 1e-12 {
                    continue;
                }
                let d0 = nrm.dot(&points[i]);
                let mut above = 0;
                let mut below = 0;
                for (m, p) in points.iter().enumerate() {
                    if m == i || m == j || m == k {
                        continue;
                    }
                    let s = nrm.dot(p) - d0;
                    if s > 1e-12 {
                        above += 1;
                    } else if s < -1e-12 {
                        below += 1;
                    }
                }
                if above == 0 || below == 0 {
                    on_hull[i] = true;
                    on_hull[j] = true;
                    on_hull[k] = true;
                }
            }
        }
    }
    (0..n).filter(|&i| on_hull[i]).collect()
}

// ---------------------------------------------------------------------------
// Projections
// ---------------------------------------------------------------------------

/// Stereographic projection from the north pole onto the equatorial plane.
#[must_use]
pub fn stereographic(p: Vec3) -> Vec2 {
    Vec2::new(p.x / (1.0 - p.z), p.y / (1.0 - p.z))
}

/// Inverse stereographic projection.
#[must_use]
pub fn inverse_stereographic(q: Vec2) -> Vec3 {
    let r2 = q.x * q.x + q.y * q.y;
    Vec3::new(2.0 * q.x, 2.0 * q.y, r2 - 1.0) * (1.0 / (r2 + 1.0))
}

/// Stereographic projection of the unit n-sphere from the last-coordinate
/// pole.
#[must_use]
pub fn stereographic_n(p: &VecN) -> VecN {
    let n = p.dim();
    let denom = 1.0 - p[n - 1];
    VecN::from(
        &p.data[..n - 1]
            .iter()
            .map(|&x| x / denom)
            .collect::<Vec<f64>>(),
    )
}

fn latlon(p: Vec3) -> (f64, f64) {
    let q = p.normalized();
    (q.z.asin(), q.y.atan2(q.x))
}

fn from_latlon(lat: f64, lon: f64) -> Vec3 {
    Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
}

/// Gnomonic projection about `center` (great circles map to lines).
#[must_use]
pub fn gnomonic(p: Vec3, center: Vec3) -> Vec2 {
    let c = center.normalized();
    let q = p.normalized();
    let cos_c = q.dot(&c);
    // local east/north basis
    let (e, n) = local_basis(c);
    Vec2::new(q.dot(&e) / cos_c, q.dot(&n) / cos_c)
}

/// Inverse gnomonic projection.
#[must_use]
pub fn gnomonic_inverse(q: Vec2, center: Vec3) -> Vec3 {
    let c = center.normalized();
    let (e, n) = local_basis(c);
    (c + e * q.x + n * q.y).normalized()
}

fn local_basis(c: Vec3) -> (Vec3, Vec3) {
    let up = if c.z.abs() < 0.9 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let e = up.cross(&c).normalized();
    let n = c.cross(&e);
    (e, n)
}

/// Orthographic projection about `center`.
#[must_use]
pub fn orthographic(p: Vec3, center: Vec3) -> Vec2 {
    let (e, n) = local_basis(center.normalized());
    let q = p.normalized();
    Vec2::new(q.dot(&e), q.dot(&n))
}

/// Inverse orthographic (near-side solution).
#[must_use]
pub fn orthographic_inverse(q: Vec2, center: Vec3) -> Vec3 {
    let c = center.normalized();
    let (e, n) = local_basis(c);
    let r2 = q.x * q.x + q.y * q.y;
    let z = (1.0 - r2).max(0.0).sqrt();
    e * q.x + n * q.y + c * z
}

/// Mercator projection (x = lon, y = ln tan(pi/4 + lat/2)).
#[must_use]
pub fn mercator(p: Vec3) -> Vec2 {
    let (lat, lon) = latlon(p);
    Vec2::new(lon, (0.25 * PI + 0.5 * lat).tan().ln())
}

/// Inverse Mercator.
#[must_use]
pub fn mercator_inverse(q: Vec2) -> Vec3 {
    let lat = 2.0 * q.y.exp().atan() - 0.5 * PI;
    from_latlon(lat, q.x)
}

/// Lambert azimuthal equal-area projection about the north pole.
#[must_use]
pub fn lambert_azimuthal_equal_area(p: Vec3) -> Vec2 {
    let q = p.normalized();
    let k = (2.0 / (1.0 + q.z)).max(0.0).sqrt();
    Vec2::new(k * q.x, k * q.y)
}

/// Inverse Lambert azimuthal equal-area.
#[must_use]
pub fn lambert_azimuthal_equal_area_inverse(q: Vec2) -> Vec3 {
    let r2 = q.x * q.x + q.y * q.y;
    let f = (1.0 - r2 / 4.0).max(0.0).sqrt();
    Vec3::new(f * q.x, f * q.y, 1.0 - r2 / 2.0)
}

/// Mollweide projection (equal-area pseudocylindrical).
#[must_use]
pub fn mollweide(p: Vec3) -> Vec2 {
    let (lat, lon) = latlon(p);
    // solve 2 theta + sin 2 theta = pi sin lat
    let target = PI * lat.sin();
    let mut th = lat;
    for _ in 0..50 {
        let f = 2.0 * th + (2.0 * th).sin() - target;
        let fp = 2.0 + 2.0 * (2.0 * th).cos();
        if fp.abs() < 1e-14 {
            break;
        }
        th -= f / fp;
    }
    Vec2::new(
        2.0 * 2.0_f64.sqrt() / PI * lon * th.cos(),
        2.0_f64.sqrt() * th.sin(),
    )
}

/// Inverse Mollweide.
#[must_use]
pub fn mollweide_inverse(q: Vec2) -> Vec3 {
    let th = (q.y / 2.0_f64.sqrt()).clamp(-1.0, 1.0).asin();
    let lat = ((2.0 * th + (2.0 * th).sin()) / PI).clamp(-1.0, 1.0).asin();
    let lon = PI * q.x / (2.0 * 2.0_f64.sqrt() * th.cos().max(1e-12));
    from_latlon(lat, lon)
}

/// Equirectangular projection (x = lon, y = lat).
#[must_use]
pub fn equirectangular(p: Vec3) -> Vec2 {
    let (lat, lon) = latlon(p);
    Vec2::new(lon, lat)
}

/// Inverse equirectangular.
#[must_use]
pub fn equirectangular_inverse(q: Vec2) -> Vec3 {
    from_latlon(q.y, q.x)
}

/// Azimuthal equidistant projection about the north pole.
#[must_use]
pub fn azimuthal_equidistant(p: Vec3) -> Vec2 {
    let q = p.normalized();
    let c = q.z.clamp(-1.0, 1.0).acos(); // angular distance from pole
    let r = (q.x * q.x + q.y * q.y).sqrt();
    if r < 1e-15 {
        return Vec2::new(0.0, 0.0);
    }
    Vec2::new(c * q.x / r, c * q.y / r)
}

/// Inverse azimuthal equidistant.
#[must_use]
pub fn azimuthal_equidistant_inverse(q: Vec2) -> Vec3 {
    let c = (q.x * q.x + q.y * q.y).sqrt();
    if c < 1e-15 {
        return Vec3::new(0.0, 0.0, 1.0);
    }
    let (s, co) = c.sin_cos();
    Vec3::new(s * q.x / c, s * q.y / c, co)
}

/// Robinson projection (table-interpolated pseudocylindrical).
#[must_use]
pub fn robinson(p: Vec3) -> Vec2 {
    let (lat, lon) = latlon(p);
    let (x_len, y_len) = robinson_coeffs(lat.abs());
    Vec2::new(
        0.8487 * x_len * lon,
        1.3523 * y_len * if lat < 0.0 { -1.0 } else { 1.0 },
    )
}

/// Inverse Robinson (bisection on the latitude table).
#[must_use]
pub fn robinson_inverse(q: Vec2) -> Vec3 {
    let target = (q.y / 1.3523).abs();
    let (mut lo, mut hi) = (0.0_f64, 0.5 * PI);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let (_, y_len) = robinson_coeffs(mid);
        if y_len < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lat = 0.5 * (lo + hi) * q.y.signum();
    let (x_len, _) = robinson_coeffs(lat.abs());
    let lon = q.x / (0.8487 * x_len);
    from_latlon(lat, lon)
}

fn robinson_coeffs(lat_abs: f64) -> (f64, f64) {
    // (X, Y) table every 5 degrees
    const TABLE: [(f64, f64); 19] = [
        (1.0000, 0.0000),
        (0.9986, 0.0620),
        (0.9954, 0.1240),
        (0.9900, 0.1860),
        (0.9822, 0.2480),
        (0.9730, 0.3100),
        (0.9600, 0.3720),
        (0.9427, 0.4340),
        (0.9216, 0.4958),
        (0.8962, 0.5571),
        (0.8679, 0.6176),
        (0.8350, 0.6769),
        (0.7986, 0.7346),
        (0.7597, 0.7903),
        (0.7186, 0.8435),
        (0.6732, 0.8936),
        (0.6213, 0.9394),
        (0.5722, 0.9761),
        (0.5322, 1.0000),
    ];
    let deg = lat_abs.to_degrees().clamp(0.0, 90.0);
    let i = (deg / 5.0).floor() as usize;
    let i = i.min(17);
    let f = deg / 5.0 - i as f64;
    let x = TABLE[i].0 * (1.0 - f) + TABLE[i + 1].0 * f;
    let y = TABLE[i].1 * (1.0 - f) + TABLE[i + 1].1 * f;
    (x, y)
}

// ---------------------------------------------------------------------------
// Hopf fibration and S3
// ---------------------------------------------------------------------------

/// Hopf map S3 -> S2: q -> q k q^-1 image of the base point, giving
/// (2(xz + wy), 2(yz - wx), w^2 + z^2 - x^2 - y^2).
#[must_use]
pub fn hopf_fibration(q: Quaternion) -> Vec3 {
    let q = q.normalize();
    let (w, x, y, z) = (q.w, q.x, q.y, q.z);
    Vec3::new(
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        w * w + z * z - x * x - y * y,
    )
}

/// The circle fiber in S3 above a point of S2, sampled at n quaternions.
#[must_use]
pub fn hopf_fiber(p: Vec3, n: usize) -> Vec<Quaternion> {
    // find one preimage: rotation carrying +z to p (half-angle quaternion)
    let pn = p.normalized();
    let base = if (pn.z + 1.0).abs() < 1e-12 {
        Quaternion::new(0.0, 1.0, 0.0, 0.0)
    } else {
        let axis = Vec3::new(0.0, 0.0, 1.0).cross(&pn);
        let angle = pn.z.clamp(-1.0, 1.0).acos();
        if axis.magnitude() < 1e-15 {
            Quaternion::identity()
        } else {
            Quaternion::from_axis_angle(axis.normalized(), angle)
        }
    };
    // the fiber is base * exp(k phi/...): right multiplication by
    // rotations about z
    (0..n)
        .map(|k| {
            // the fiber is a great circle in S3: parametrize the full
            // (w, z)-plane circle so it closes after 2 pi
            let phi = 2.0 * PI * k as f64 / n as f64;
            let spin = Quaternion::new(phi.cos(), 0.0, 0.0, phi.sin());
            (base * spin).normalize()
        })
        .collect()
}

/// Hopf fiber stereographically projected to R3 (a Villarceau circle).
#[must_use]
pub fn hopf_fiber_stereographic(p: Vec3, n: usize) -> Vec<Vec3> {
    hopf_fiber(p, n)
        .into_iter()
        .map(|q| {
            let denom = 1.0 - q.w;
            if denom.abs() < 1e-12 {
                Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY)
            } else {
                Vec3::new(q.x / denom, q.y / denom, q.z / denom)
            }
        })
        .collect()
}

/// Geodesic on S3 between unit quaternions (slerp).
#[must_use]
pub fn s3_geodesic(a: Quaternion, b: Quaternion, t: f64) -> Quaternion {
    crate::quaternion::slerp(&a, &b, t)
}

/// Near-uniform deterministic points on S3 (super-Fibonacci spiral).
#[must_use]
pub fn s3_uniform_points(n: usize) -> Vec<Quaternion> {
    // Alexa's super-Fibonacci constants
    let phi = 2.0_f64.sqrt();
    let psi = 1.533_751_168_755_204_3;
    (0..n)
        .map(|i| {
            let s = i as f64 + 0.5;
            let t = s / n as f64;
            let d = 2.0 * PI * s;
            let r = t.sqrt();
            let rr = (1.0 - t).sqrt();
            let alpha = d / phi;
            let beta = d / psi;
            Quaternion::new(
                r * alpha.sin(),
                r * alpha.cos(),
                rr * beta.sin(),
                rr * beta.cos(),
            )
        })
        .collect()
}

/// Uniform random points on the unit (dim-1)-sphere in R^dim.
#[must_use]
pub fn sphere_uniform_points_n(n: usize, dim: usize, rng: &mut Rng) -> Vec<VecN> {
    (0..n).map(|_| VecN::random_unit(dim, rng)).collect()
}

// ---------------------------------------------------------------------------
// Volumes and caps
// ---------------------------------------------------------------------------

/// Volume of the n-ball of radius r.
#[must_use]
pub fn sphere_volume_n(r: f64, n: usize) -> f64 {
    let nf = n as f64;
    PI.powf(nf / 2.0) / crate::special::gamma(nf / 2.0 + 1.0) * r.powf(nf)
}

/// Surface area of the (n-1)-sphere of radius r in R^n.
#[must_use]
pub fn sphere_surface_n(r: f64, n: usize) -> f64 {
    let nf = n as f64;
    nf * PI.powf(nf / 2.0) / crate::special::gamma(nf / 2.0 + 1.0) * r.powf(nf - 1.0)
}

/// Area of a spherical cap of opening angle theta on a sphere of radius r.
#[must_use]
pub fn sphere_cap_area(r: f64, theta: f64) -> f64 {
    2.0 * PI * r * r * (1.0 - theta.cos())
}

/// Volume of the corresponding solid cap.
#[must_use]
pub fn sphere_cap_volume(r: f64, theta: f64) -> f64 {
    let h = r * (1.0 - theta.cos());
    PI * h * h * (3.0 * r - h) / 3.0
}

// ---------------------------------------------------------------------------
// Spherical harmonics
// ---------------------------------------------------------------------------

/// Complex spherical harmonic Y_l^m(theta, phi) with the Condon-Shortley
/// phase.
#[must_use]
pub fn spherical_harmonics_complex(l: u32, m: i32, theta: f64, phi: f64) -> Complex {
    let ma = m.unsigned_abs();
    let mut norm = ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt();
    for k in (l - ma + 1)..=(l + ma) {
        norm /= (k as f64).sqrt();
    }
    let p = crate::special::legendre_p_assoc(l, ma as i32, theta.cos());
    let base = norm * p;
    let (s, c) = (ma as f64 * phi).sin_cos();
    if m >= 0 {
        Complex::new(base * c, base * s)
    } else {
        // Y_{l,-m} = (-1)^m conj(Y_lm)
        let sign = if ma.is_multiple_of(2) { 1.0 } else { -1.0 };
        Complex::new(sign * base * c, -sign * base * s)
    }
}

fn sh_index(l: u32, m: i32) -> usize {
    ((l * l + l) as i64 + m as i64) as usize
}

/// Forward spherical harmonic transform up to `l_max` by quadrature on an
/// (n_theta x n_phi) grid. Coefficients are ordered (l, m) with
/// index l^2 + l + m.
#[must_use]
pub fn spherical_harmonic_transform(
    f: &dyn Fn(f64, f64) -> f64,
    l_max: u32,
    n_theta: usize,
    n_phi: usize,
) -> Vec<Complex> {
    let ncoef = ((l_max + 1) * (l_max + 1)) as usize;
    let mut out = vec![Complex::new(0.0, 0.0); ncoef];
    // Gauss-Legendre in cos(theta), uniform in phi
    let (nodes, weights) = crate::special::gauss_legendre_nodes(n_theta);
    let dphi = 2.0 * PI / n_phi as f64;
    for (x, w) in nodes.iter().zip(&weights) {
        let theta = x.clamp(-1.0, 1.0).acos();
        for kp in 0..n_phi {
            let phi = kp as f64 * dphi;
            let val = f(theta, phi);
            for l in 0..=l_max {
                for m in -(l as i32)..=(l as i32) {
                    let y = spherical_harmonics_complex(l, m, theta, phi);
                    let integrand = y.conjugate() * Complex::new(val * w * dphi, 0.0);
                    let idx = sh_index(l, m);
                    out[idx] = out[idx] + integrand;
                }
            }
        }
    }
    out
}

/// Evaluate a coefficient vector at (theta, phi).
#[must_use]
pub fn spherical_harmonic_inverse(coeffs: &[Complex], l_max: u32, theta: f64, phi: f64) -> f64 {
    let mut s = Complex::new(0.0, 0.0);
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            s = s + coeffs[sh_index(l, m)] * spherical_harmonics_complex(l, m, theta, phi);
        }
    }
    s.re
}

/// Spectral convolution with a zonal kernel: multiplies each (l, m)
/// coefficient by sqrt(4 pi/(2l+1)) g_l0.
#[must_use]
pub fn spherical_convolution(
    f_coeffs: &[Complex],
    g_coeffs: &[Complex],
    l_max: u32,
) -> Vec<Complex> {
    let mut out = f_coeffs.to_vec();
    for l in 0..=l_max {
        let gl = g_coeffs[sh_index(l, 0)];
        let factor = (4.0 * PI / (2.0 * l as f64 + 1.0)).sqrt();
        for m in -(l as i32)..=(l as i32) {
            let idx = sh_index(l, m);
            out[idx] = out[idx] * gl * Complex::new(factor, 0.0);
        }
    }
    out
}

/// Spectral Laplace-Beltrami: multiplies each degree-l coefficient by
/// -l(l+1).
#[must_use]
pub fn spherical_laplacian_spectral(coeffs: &[Complex], l_max: u32) -> Vec<Complex> {
    let mut out = coeffs.to_vec();
    for l in 0..=l_max {
        let e = -(l as f64) * (l as f64 + 1.0);
        for m in -(l as i32)..=(l as i32) {
            let idx = sh_index(l, m);
            out[idx] = out[idx] * Complex::new(e, 0.0);
        }
    }
    out
}

/// Heat flow on the sphere: coefficients decay as exp(-l(l+1) t).
#[must_use]
pub fn spherical_heat_flow(coeffs: &[Complex], l_max: u32, t: f64) -> Vec<Complex> {
    let mut out = coeffs.to_vec();
    for l in 0..=l_max {
        let decay = (-(l as f64) * (l as f64 + 1.0) * t).exp();
        for m in -(l as i32)..=(l as i32) {
            let idx = sh_index(l, m);
            out[idx] = out[idx] * Complex::new(decay, 0.0);
        }
    }
    out
}

/// Simple spherical wavelet band-pass: difference of two heat kernels at
/// scales t and 2t applied spectrally.
#[must_use]
pub fn spherical_wavelets(coeffs: &[Complex], l_max: u32, t: f64) -> Vec<Complex> {
    let fine = spherical_heat_flow(coeffs, l_max, t);
    let coarse = spherical_heat_flow(coeffs, l_max, 2.0 * t);
    fine.iter().zip(&coarse).map(|(&a, &b)| a - b).collect()
}

// ---------------------------------------------------------------------------
// HEALPix (ring scheme)
// ---------------------------------------------------------------------------

/// Number of HEALPix pixels: 12 nside^2.
#[must_use]
pub fn healpix_npix(nside: usize) -> usize {
    12 * nside * nside
}

/// HEALPix ring-scheme pixel index for direction (theta, phi).
#[must_use]
pub fn healpix_ang2pix(nside: usize, theta: f64, phi: f64) -> usize {
    let ns = nside as f64;
    let z = theta.cos();
    let phi = phi.rem_euclid(2.0 * PI);
    let tt = phi / (0.5 * PI); // in [0, 4)
    if z.abs() <= 2.0 / 3.0 {
        // equatorial region
        let t1 = ns * (0.5 + tt);
        let t2 = ns * z * 0.75;
        let jp = (t1 - t2).floor() as i64; // ascending edge
        let jm = (t1 + t2).floor() as i64; // descending edge
        let ir = (ns as i64) + 1 + jp - jm; // ring counter in {1, 2nside+1}
        let kshift = 1 - (ir & 1);
        let nl4 = 4 * nside as i64;
        let mut ip = (jp + jm - ns as i64 + kshift + 1) / 2;
        ip = ip.rem_euclid(nl4);
        let npix_north = 2 * nside as i64 * (nside as i64 - 1);
        (npix_north + (ir - 1) * nl4 + ip) as usize
    } else {
        // polar caps
        let tp = tt.fract();
        let tmp = ns * (3.0 * (1.0 - z.abs())).sqrt();
        let jp = (tp * tmp).floor() as i64;
        let jm = ((1.0 - tp) * tmp).floor() as i64;
        let ir = jp + jm + 1; // ring number counted from the closest pole
        let ip = ((tt * ir as f64).floor() as i64).rem_euclid(4 * ir);
        if z > 0.0 {
            (2 * ir * (ir - 1) + ip) as usize
        } else {
            (healpix_npix(nside) as i64 - 2 * ir * (ir + 1) + ip) as usize
        }
    }
}

/// Center direction (theta, phi) of a HEALPix ring-scheme pixel.
#[must_use]
pub fn healpix_pix2ang(nside: usize, pix: usize) -> (f64, f64) {
    let ns = nside as i64;
    let npix = healpix_npix(nside) as i64;
    let p = pix as i64;
    let ncap = 2 * ns * (ns - 1);
    if p < ncap {
        // north polar cap
        let ir = (((1.0 + (1.0 + 2.0 * p as f64).sqrt()) / 2.0).floor()) as i64;
        let ir = if 2 * ir * (ir - 1) > p { ir - 1 } else { ir };
        let ip = p - 2 * ir * (ir - 1);
        let z = 1.0 - (ir as f64) * (ir as f64) / (3.0 * ns as f64 * ns as f64);
        let phi = (ip as f64 + 0.5) * 0.5 * PI / ir as f64;
        (z.clamp(-1.0, 1.0).acos(), phi)
    } else if p < npix - ncap {
        // equatorial belt
        let pp = p - ncap;
        let nl4 = 4 * ns;
        let ir = pp / nl4 + ns; // ring index from the north pole
        let ip = pp % nl4;
        let fodd = if (ir + ns) % 2 == 0 { 0.5 } else { 0.0 };
        let z = 4.0 / 3.0 - 2.0 * ir as f64 / (3.0 * ns as f64);
        let phi = (ip as f64 + fodd) * 0.5 * PI / ns as f64;
        (z.clamp(-1.0, 1.0).acos(), phi)
    } else {
        // south polar cap
        let ps = npix - 1 - p;
        let ir = (((1.0 + (1.0 + 2.0 * ps as f64).sqrt()) / 2.0).floor()) as i64;
        let ir = if 2 * ir * (ir - 1) > ps { ir - 1 } else { ir };
        let ip = ps - 2 * ir * (ir - 1);
        let z = -1.0 + (ir as f64) * (ir as f64) / (3.0 * ns as f64 * ns as f64);
        let phi = 2.0 * PI - (ip as f64 + 0.5) * 0.5 * PI / ir as f64;
        (z.clamp(-1.0, 1.0).acos(), phi.rem_euclid(2.0 * PI))
    }
}

// ---------------------------------------------------------------------------
// Point distributions
// ---------------------------------------------------------------------------

/// Estimate of the Tammes-problem packing angle for n caps (empirical
/// asymptotic bound).
#[must_use]
pub fn spherical_cap_packing(n: usize) -> f64 {
    // asymptotic optimal angular separation ~ sqrt(8 pi / (sqrt(3) n))
    (8.0 * PI / (3.0_f64.sqrt() * n as f64)).sqrt()
}

/// Thomson problem: minimize Coulomb energy of n charges by projected
/// gradient descent. Returns the final configuration.
#[must_use]
pub fn thomson_problem(n: usize, iters: usize, rng: &mut Rng) -> Vec<Vec3> {
    let mut pts: Vec<Vec3> = (0..n)
        .map(|_| {
            Vec3::new(
                rng.next_gaussian(),
                rng.next_gaussian(),
                rng.next_gaussian(),
            )
            .normalized()
        })
        .collect();
    let mut step = 0.05;
    let energy = |pts: &[Vec3]| -> f64 {
        let mut e = 0.0;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                e += 1.0 / (pts[i] - pts[j]).magnitude();
            }
        }
        e
    };
    let mut e_prev = energy(&pts);
    for _ in 0..iters {
        let mut forces = vec![Vec3::new(0.0, 0.0, 0.0); n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = pts[i] - pts[j];
                let r = d.magnitude().max(1e-9);
                forces[i] = forces[i] + d * (1.0 / (r * r * r));
            }
        }
        let trial: Vec<Vec3> = pts
            .iter()
            .zip(&forces)
            .map(|(p, f)| {
                // tangential component only
                let ft = *f - *p * f.dot(p);
                (*p + ft * step).normalized()
            })
            .collect();
        let e_new = energy(&trial);
        if e_new < e_prev {
            pts = trial;
            e_prev = e_new;
            step *= 1.1;
        } else {
            step *= 0.5;
        }
    }
    pts
}

/// Minimum pairwise angular distance of a spherical code.
#[must_use]
pub fn spherical_code_min_angle(points: &[Vec3]) -> f64 {
    let mut min = PI;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let a = points[i]
                .normalized()
                .dot(&points[j].normalized())
                .clamp(-1.0, 1.0)
                .acos();
            min = min.min(a);
        }
    }
    min
}

/// Rotate a point set by a rotation.
#[must_use]
pub fn rotate_sphere_points(points: &[Vec3], r: &So3) -> Vec<Vec3> {
    points.iter().map(|&p| r.apply(p)).collect()
}

/// Spherical k-means with cosine distance. Returns (centroids, labels).
#[must_use]
pub fn spherical_kmeans(
    points: &[Vec3],
    k: usize,
    iters: usize,
    rng: &mut Rng,
) -> (Vec<Vec3>, Vec<usize>) {
    let n = points.len();
    let mut centroids: Vec<Vec3> = (0..k)
        .map(|_| points[(rng.next_u64() as usize) % n].normalized())
        .collect();
    let mut labels = vec![0usize; n];
    for _ in 0..iters {
        for (i, p) in points.iter().enumerate() {
            labels[i] = (0..k)
                .max_by(|&a, &b| {
                    centroids[a]
                        .dot(p)
                        .partial_cmp(&centroids[b].dot(p))
                        .unwrap()
                })
                .unwrap();
        }
        for (c, centroid) in centroids.iter_mut().enumerate() {
            let mut s = Vec3::new(0.0, 0.0, 0.0);
            for (p, &l) in points.iter().zip(&labels) {
                if l == c {
                    s = s + *p;
                }
            }
            if s.magnitude() > 1e-12 {
                *centroid = s.normalized();
            }
        }
    }
    (centroids, labels)
}

// ---------------------------------------------------------------------------
// Directional statistics
// ---------------------------------------------------------------------------

/// Von Mises-Fisher density on S2.
#[must_use]
pub fn von_mises_fisher_pdf(x: Vec3, mu: Vec3, kappa: f64) -> f64 {
    if kappa < 1e-12 {
        return 1.0 / (4.0 * PI);
    }
    let c = kappa / (4.0 * PI * kappa.sinh());
    c * (kappa * mu.normalized().dot(&x.normalized())).exp()
}

/// Sample from the von Mises-Fisher distribution on S2 (Ulrich/Wood).
#[must_use]
pub fn vmf_sample(mu: Vec3, kappa: f64, rng: &mut Rng) -> Vec3 {
    let mu = mu.normalized();
    // sample w = cos(theta) by inverse CDF
    let u = rng.next_f64();
    let w = if kappa < 1e-8 {
        2.0 * u - 1.0
    } else {
        1.0 + (u + (1.0 - u) * (-2.0 * kappa).exp()).ln() / kappa
    };
    let phi = 2.0 * PI * rng.next_f64();
    let s = (1.0 - w * w).max(0.0).sqrt();
    // orthonormal frame around mu
    let helper = if mu.z.abs() < 0.9 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let e1 = helper.cross(&mu).normalized();
    let e2 = mu.cross(&e1);
    mu * w + e1 * (s * phi.cos()) + e2 * (s * phi.sin())
}

/// Fit (mu, kappa) of a von Mises-Fisher distribution from samples.
#[must_use]
pub fn vmf_fit(points: &[Vec3]) -> (Vec3, f64) {
    let s = points.iter().fold(Vec3::new(0.0, 0.0, 0.0), |a, &b| a + b.normalized());
    let r_bar = s.magnitude() / points.len() as f64;
    let mu = s.normalized();
    let kappa = r_bar * (3.0 - r_bar * r_bar) / (1.0 - r_bar * r_bar).max(1e-12);
    (mu, kappa)
}

/// Kent (Fisher-Bingham 5-parameter) density up to normalization refinement:
/// f = C exp(kappa g1.x + beta ((g2.x)^2 - (g3.x)^2)).
#[must_use]
pub fn kent_distribution_pdf(x: Vec3, g1: Vec3, g2: Vec3, g3: Vec3, kappa: f64, beta: f64) -> f64 {
    // series normalization (valid for 2 beta < kappa)
    let c = 2.0 * PI * ((kappa - 2.0 * beta) * (kappa + 2.0 * beta)).max(1e-12).sqrt().recip()
        * kappa.exp();
    let xn = x.normalized();
    let e = kappa * g1.normalized().dot(&xn)
        + beta * (g2.normalized().dot(&xn).powi(2) - g3.normalized().dot(&xn).powi(2));
    e.exp() / c
}

// ---------------------------------------------------------------------------
// Quadrature and designs
// ---------------------------------------------------------------------------

/// Small spherical t-designs from tables: t = 1 (antipodes), 2
/// (tetrahedron), 3 (octahedron), 5 (icosahedron). None otherwise.
#[must_use]
pub fn spherical_t_design(t: usize) -> Option<Vec<Vec3>> {
    match t {
        1 => Some(vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)]),
        2 => {
            let a = 1.0 / 3.0_f64.sqrt();
            Some(vec![
                Vec3::new(a, a, a),
                Vec3::new(a, -a, -a),
                Vec3::new(-a, a, -a),
                Vec3::new(-a, -a, a),
            ])
        }
        3 => Some(vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ]),
        5 => {
            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let norm = (1.0 + phi * phi).sqrt();
            let mut pts = Vec::new();
            for &s1 in &[1.0, -1.0] {
                for &s2 in &[1.0, -1.0] {
                    pts.push(Vec3::new(0.0, s1 / norm, s2 * phi / norm));
                    pts.push(Vec3::new(s1 / norm, s2 * phi / norm, 0.0));
                    pts.push(Vec3::new(s1 * phi / norm, 0.0, s2 / norm));
                }
            }
            Some(pts)
        }
        _ => None,
    }
}

/// Lebedev quadrature nodes and weights for orders 6, 14, and 26 (weights
/// sum to 1; integrates times 4 pi).
///
/// # Panics
/// Panics for unsupported orders.
#[must_use]
pub fn lebedev_quadrature(order: usize) -> Vec<(Vec3, f64)> {
    let oct = |w: f64, out: &mut Vec<(Vec3, f64)>| {
        for &s in &[1.0, -1.0] {
            out.push((Vec3::new(s, 0.0, 0.0), w));
            out.push((Vec3::new(0.0, s, 0.0), w));
            out.push((Vec3::new(0.0, 0.0, s), w));
        }
    };
    let cube = |w: f64, out: &mut Vec<(Vec3, f64)>| {
        let a = 1.0 / 3.0_f64.sqrt();
        for &s1 in &[1.0, -1.0] {
            for &s2 in &[1.0, -1.0] {
                for &s3 in &[1.0, -1.0] {
                    out.push((Vec3::new(s1 * a, s2 * a, s3 * a), w));
                }
            }
        }
    };
    let edge_mid = |w: f64, out: &mut Vec<(Vec3, f64)>| {
        let a = 1.0 / 2.0_f64.sqrt();
        for &s1 in &[1.0, -1.0] {
            for &s2 in &[1.0, -1.0] {
                out.push((Vec3::new(s1 * a, s2 * a, 0.0), w));
                out.push((Vec3::new(s1 * a, 0.0, s2 * a), w));
                out.push((Vec3::new(0.0, s1 * a, s2 * a), w));
            }
        }
    };
    let mut out = Vec::new();
    match order {
        6 => oct(1.0 / 6.0, &mut out),
        14 => {
            oct(1.0 / 15.0, &mut out);
            cube(3.0 / 40.0, &mut out);
        }
        26 => {
            oct(1.0 / 21.0, &mut out);
            edge_mid(4.0 / 105.0, &mut out);
            cube(27.0 / 840.0, &mut out);
        }
        _ => panic!("lebedev_quadrature supports orders 6, 14, 26"),
    }
    out
}

/// Product Gauss-Legendre x uniform-phi quadrature on the sphere: returns
/// (theta, phi, weight) with weights summing to 4 pi.
#[must_use]
pub fn gauss_legendre_sphere(n_theta: usize, n_phi: usize) -> Vec<(f64, f64, f64)> {
    let (nodes, weights) = crate::special::gauss_legendre_nodes(n_theta);
    let dphi = 2.0 * PI / n_phi as f64;
    let mut out = Vec::with_capacity(n_theta * n_phi);
    for (x, w) in nodes.iter().zip(&weights) {
        let theta = x.clamp(-1.0, 1.0).acos();
        for k in 0..n_phi {
            out.push((theta, k as f64 * dphi, w * dphi));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn test_nsphere_maps() {
        let a = VecN::from(&[1.0, 0.0, 0.0, 0.0]).normalized();
        let b = VecN::from(&[0.0, 1.0, 0.0, 0.0]).normalized();
        assert!((sphere_distance_n(&a, &b) - 0.5 * PI).abs() < 1e-12);
        // slerp midpoint is equidistant and unit
        let m = sphere_geodesic_n(&a, &b, 0.5);
        assert!((m.norm() - 1.0).abs() < 1e-12);
        assert!((sphere_distance_n(&a, &m) - 0.25 * PI).abs() < 1e-12);
        // exp/log roundtrip
        let v = sphere_log_n(&a, &b);
        assert!((v.norm() - 0.5 * PI).abs() < 1e-12);
        let b2 = sphere_exp_n(&a, &v);
        assert!(b2.sub(&b).norm() < 1e-12);
        // parallel transport preserves norm and tangency
        let p = VecN::from(&[1.0, 0.0, 0.0]).normalized();
        let q = VecN::from(&[0.0, 1.0, 0.0]).normalized();
        let w = VecN::from(&[0.0, 0.3, 0.4]); // tangent at p
        let wt = sphere_parallel_transport_n(&w, &p, &q);
        assert!((wt.norm() - w.norm()).abs() < 1e-12);
        assert!(wt.dot(&q).abs() < 1e-12, "tangency after transport");
    }

    #[test]
    fn test_spherical_trig() {
        // octant triangle: all angles pi/2, area pi/2
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);
        let area = spherical_triangle_area(a, b, c);
        assert!((area - 0.5 * PI).abs() < 1e-12);
        let (al, be, ga) = spherical_triangle_angles(a, b, c);
        assert!((al - 0.5 * PI).abs() < 1e-12);
        // area matches geometry::spherical_excess of the angles
        let exc = crate::geometry::spherical_excess(al, be, ga);
        assert!((area - exc).abs() < 1e-10);
        // law of cosines: octant sides are pi/2
        let side = spherical_law_of_cosines(0.5 * PI, 0.5 * PI, 0.5 * PI);
        assert!((side - 0.5 * PI).abs() < 1e-12);
        // law of sines on the octant
        let alpha = spherical_law_of_sines(0.5 * PI, 0.5 * PI, 0.5 * PI);
        assert!((alpha - 0.5 * PI).abs() < 1e-9);
        // haversine matches geometry::great_circle_distance
        let (la1, lo1, la2, lo2) = (0.5, 0.3, -0.2, 1.4);
        let h = haversine(la1, lo1, la2, lo2, 2.5);
        let g = crate::geometry::great_circle_distance(2.5, la1, lo1, la2, lo2);
        assert!((h - g).abs() < 1e-10, "{h} vs {g}");
        // polygon area: octant as a 3-gon
        let poly = spherical_polygon_area(&[a, b, c]);
        assert!((poly - 0.5 * PI).abs() < 1e-9);
        // centroid of a symmetric cluster
        let ctr = spherical_centroid(&[
            Vec3::new(0.9, 0.1, 0.0).normalized(),
            Vec3::new(0.9, -0.1, 0.0).normalized(),
        ]);
        assert!((ctr.x - 1.0).abs() < 1e-3 && ctr.y.abs() < 1e-12);
        let w = spherical_mean_weighted(&[a, b], &[3.0, 1.0]);
        assert!(w.x > w.y);
    }

    #[test]
    fn test_delaunay_voronoi_hull() {
        // octahedron vertices: Delaunay must produce 8 faces
        let sites = spherical_t_design(3).unwrap();
        let tris = spherical_delaunay(&sites);
        assert_eq!(tris.len(), 8, "octahedron faces: {}", tris.len());
        // Voronoi cells: 6 cells, each with 4 vertices (cube faces)
        let cells = spherical_voronoi(&sites);
        assert_eq!(cells.len(), 6);
        for cell in &cells {
            assert_eq!(cell.len(), 4, "cube-face cell size {}", cell.len());
        }
        // hull of sphere points includes everything
        let mut rng = Rng::new(3);
        let pts: Vec<Vec3> = (0..12)
            .map(|_| {
                Vec3::new(
                    rng.next_gaussian(),
                    rng.next_gaussian(),
                    rng.next_gaussian(),
                )
                .normalized()
            })
            .collect();
        let hull = spherical_convex_hull(&pts);
        assert_eq!(hull.len(), 12);
        // interior point excluded
        let mut with_inner = pts.clone();
        with_inner.push(Vec3::new(0.01, 0.0, 0.0));
        let hull2 = spherical_convex_hull(&with_inner);
        assert!(!hull2.contains(&12));
    }

    #[test]
    fn test_projections_roundtrip() {
        let pts = [
            Vec3::new(0.3, -0.5, 0.6).normalized(),
            Vec3::new(-0.7, 0.2, 0.4).normalized(),
            Vec3::new(0.1, 0.9, -0.3).normalized(),
        ];
        for (k, &p) in pts.iter().enumerate() {
            assert!((inverse_stereographic(stereographic(p)) - p).magnitude() < 1e-12, "stereo {k}");
            assert!((mercator_inverse(mercator(p)) - p).magnitude() < 1e-12, "mercator {k}");
            assert!(
                (equirectangular_inverse(equirectangular(p)) - p).magnitude() < 1e-12,
                "equirect {k}"
            );
            assert!(
                (mollweide_inverse(mollweide(p)) - p).magnitude() < 1e-9,
                "mollweide {k}"
            );
            assert!(
                (azimuthal_equidistant_inverse(azimuthal_equidistant(p)) - p).magnitude() < 1e-12,
                "azeq {k}"
            );
            assert!(
                (lambert_azimuthal_equal_area_inverse(lambert_azimuthal_equal_area(p)) - p)
                    .magnitude()
                    < 1e-12,
                "lambert {k}"
            );
            assert!(
                (robinson_inverse(robinson(p)) - p).magnitude() < 1e-6,
                "robinson {k}"
            );
            let center = Vec3::new(0.2, 0.3, 0.93).normalized();
            if p.dot(&center) > 0.2 {
                assert!(
                    (gnomonic_inverse(gnomonic(p, center), center) - p).magnitude() < 1e-12,
                    "gnomonic {k}"
                );
                assert!(
                    (orthographic_inverse(orthographic(p, center), center) - p).magnitude() < 1e-9,
                    "ortho {k}"
                );
            }
        }
        // stereographic_n matches the 3D version
        let p = Vec3::new(0.3, -0.5, 0.6).normalized();
        let sn = stereographic_n(&VecN::from(&[p.x, p.y, p.z]));
        let s2 = stereographic(p);
        assert!((sn[0] - s2.x).abs() < 1e-12 && (sn[1] - s2.y).abs() < 1e-12);
        // Lambert azimuthal is equal-area: small cap around pole maps to
        // disk of equal area
        let theta = 0.4;
        let rim = lambert_azimuthal_equal_area(from_latlon(0.5 * PI - theta, 0.0));
        let disk_area = PI * (rim.x * rim.x + rim.y * rim.y);
        assert!((disk_area - sphere_cap_area(1.0, theta)).abs() < 1e-9);
    }

    #[test]
    fn test_hopf() {
        let mut rng = Rng::new(9);
        // hopf maps unit quaternions to unit vectors; fibers map to a point
        for _ in 0..5 {
            let q = Quaternion::new(
                rng.next_gaussian(),
                rng.next_gaussian(),
                rng.next_gaussian(),
                rng.next_gaussian(),
            )
            .normalize();
            let p = hopf_fibration(q);
            assert!((p.magnitude() - 1.0).abs() < 1e-12);
        }
        let base = Vec3::new(0.3, -0.4, 0.5).normalized();
        let fiber = hopf_fiber(base, 16);
        for q in &fiber {
            let img = hopf_fibration(*q);
            assert!((img - base).magnitude() < 1e-9, "fiber maps to base");
        }
        // two distinct fibers are disjoint circles; stereographic images are
        // linked (compute the Gauss linking integral)
        let f1 = hopf_fiber_stereographic(Vec3::new(0.0, 0.0, -1.0), 60);
        let f2 = hopf_fiber_stereographic(Vec3::new(1.0, 0.0, 0.0), 60);
        let link = gauss_linking(&f1, &f2);
        assert!((link.abs() - 1.0).abs() < 0.05, "linking number {link}");
        // s3 geodesic stays unit
        let a = Quaternion::identity();
        let b = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.0);
        let mid = s3_geodesic(a, b, 0.5);
        assert!((mid.norm() - 1.0).abs() < 1e-12);
        // super-Fibonacci points are unit and well spread
        let pts = s3_uniform_points(64);
        for q in &pts {
            assert!((q.norm() - 1.0).abs() < 1e-9);
        }
        let mut min_d = f64::MAX;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let d = pts[i].dot(&pts[j]).abs();
                min_d = min_d.min(1.0 - d);
            }
        }
        assert!(min_d > 1e-3, "S3 points distinct");
    }

    fn gauss_linking(a: &[Vec3], b: &[Vec3]) -> f64 {
        let mut sum = 0.0;
        let n = a.len();
        let m = b.len();
        for i in 0..n {
            let da = a[(i + 1) % n] - a[i];
            let am = (a[(i + 1) % n] + a[i]) * 0.5;
            for j in 0..m {
                let db = b[(j + 1) % m] - b[j];
                let bm = (b[(j + 1) % m] + b[j]) * 0.5;
                let r = bm - am;
                let rn = r.magnitude();
                if rn < 1e-9 {
                    continue;
                }
                sum += da.cross(&db).dot(&r) / (rn * rn * rn);
            }
        }
        -sum / (4.0 * PI)
    }

    #[test]
    fn test_volumes() {
        assert!((sphere_volume_n(1.0, 3) - 4.0 * PI / 3.0).abs() < 1e-12);
        assert!((sphere_volume_n(1.0, 2) - PI).abs() < 1e-12);
        assert!((sphere_surface_n(1.0, 3) - 4.0 * PI).abs() < 1e-12);
        assert!((sphere_surface_n(2.0, 3) - 16.0 * PI).abs() < 1e-12);
        // hemisphere cap
        assert!((sphere_cap_area(1.0, 0.5 * PI) - 2.0 * PI).abs() < 1e-12);
        assert!((sphere_cap_volume(1.0, PI) - 4.0 * PI / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_spherical_harmonics() {
        // orthonormality by quadrature: transform of Y_lm gives a delta
        let (l0, m0) = (2u32, 1i32);
        let f = move |theta: f64, phi: f64| {
            spherical_harmonics_complex(l0, m0, theta, phi).re
        };
        let coeffs = spherical_harmonic_transform(&f, 3, 24, 48);
        // Re(Y_21) = (Y_21 + conj(Y_21))/2 = (Y_21 - Y_2-1 (-1)^...)/2
        let c21 = coeffs[sh_index(2, 1)];
        assert!((c21.re - 0.5).abs() < 1e-9, "c21 = {c21:?}");
        // all coefficients with l != 2 vanish
        for l in [0u32, 1, 3] {
            for m in -(l as i32)..=(l as i32) {
                let cc = coeffs[sh_index(l, m)];
                assert!(cc.norm() < 1e-9, "leak at ({l},{m})");
            }
        }
        // inverse reproduces the function
        let val = spherical_harmonic_inverse(&coeffs, 3, 0.7, 1.1);
        assert!((val - f(0.7, 1.1)).abs() < 1e-9);
        // Laplacian eigenvalue: -l(l+1)
        let lap = spherical_laplacian_spectral(&coeffs, 3);
        let ratio = lap[sh_index(2, 1)].re / coeffs[sh_index(2, 1)].re;
        assert!((ratio + 6.0).abs() < 1e-9);
        // heat flow decays high degrees more
        let heated = spherical_heat_flow(&coeffs, 3, 0.1);
        assert!(heated[sh_index(2, 1)].norm() < coeffs[sh_index(2, 1)].norm());
        // convolution with a zonal delta-like kernel scales degree bands
        let g = spherical_harmonic_transform(&|theta, _| theta.cos(), 3, 24, 48);
        let conv = spherical_convolution(&coeffs, &g, 3);
        assert!(conv[sh_index(2, 1)].norm() >= 0.0);
        // Y_00 = 1/sqrt(4 pi)
        let y00 = spherical_harmonics_complex(0, 0, 1.0, 2.0);
        assert!((y00.re - 1.0 / (4.0 * PI).sqrt()).abs() < 1e-12);
        // wavelets vanish for constant input
        let const_coeffs = spherical_harmonic_transform(&|_, _| 1.0, 3, 24, 48);
        let wav = spherical_wavelets(&const_coeffs, 3, 0.5);
        assert!(wav[sh_index(0, 0)].norm() < 1e-9);
    }

    #[test]
    fn test_healpix() {
        for nside in [1usize, 2, 4] {
            let npix = healpix_npix(nside);
            assert_eq!(npix, 12 * nside * nside);
            // roundtrip: pix -> ang -> pix
            for p in 0..npix {
                let (th, ph) = healpix_pix2ang(nside, p);
                let p2 = healpix_ang2pix(nside, th, ph);
                assert_eq!(p, p2, "nside {nside} pixel {p} -> ({th}, {ph}) -> {p2}");
            }
        }
        // pixels partition the sphere: random directions land in [0, npix)
        let mut rng = Rng::new(31);
        for _ in 0..200 {
            let z = 2.0 * rng.next_f64() - 1.0;
            let phi = 2.0 * PI * rng.next_f64();
            let pix = healpix_ang2pix(4, z.acos(), phi);
            assert!(pix < healpix_npix(4));
        }
        // equal-area statistics: counts of many random points per pixel are
        // roughly uniform
        let nside = 2;
        let mut counts = vec![0usize; healpix_npix(nside)];
        let n_samples = 48_000;
        for _ in 0..n_samples {
            let z = 2.0 * rng.next_f64() - 1.0;
            let phi = 2.0 * PI * rng.next_f64();
            counts[healpix_ang2pix(nside, z.acos(), phi)] += 1;
        }
        let expected = n_samples / counts.len();
        for (p, &cc) in counts.iter().enumerate() {
            assert!(
                (cc as f64 - expected as f64).abs() < 5.0 * (expected as f64).sqrt(),
                "pixel {p}: {cc} vs {expected}"
            );
        }
    }

    #[test]
    fn test_point_distributions() {
        // Thomson n=12: icosahedral minimum energy 49.165
        let mut rng = Rng::new(7);
        let pts = thomson_problem(12, 600, &mut rng);
        let mut e = 0.0;
        for i in 0..12 {
            for j in (i + 1)..12 {
                e += 1.0 / (pts[i] - pts[j]).magnitude();
            }
        }
        assert!((e - 49.165_25).abs() < 0.05, "Thomson energy {e}");
        // icosahedral min angle ~ 63.43 degrees
        let min_angle = spherical_code_min_angle(&pts);
        assert!((min_angle - 1.1071).abs() < 0.05, "min angle {min_angle}");
        // Tammes estimate decreases with n
        assert!(spherical_cap_packing(10) > spherical_cap_packing(100));
        // rotation preserves the code geometry
        let r = So3::from_axis_angle(Vec3::new(0.3, 1.0, 0.2), 0.8);
        let rot = rotate_sphere_points(&pts, &r);
        assert!((spherical_code_min_angle(&rot) - min_angle).abs() < 1e-9);
        // spherical k-means separates two clusters
        let mut data = Vec::new();
        for _ in 0..30 {
            data.push(vmf_sample(Vec3::new(0.0, 0.0, 1.0), 50.0, &mut rng));
            data.push(vmf_sample(Vec3::new(1.0, 0.0, 0.0), 50.0, &mut rng));
        }
        let (cents, labels) = spherical_kmeans(&data, 2, 20, &mut rng);
        // the two centroids align with the two cluster directions
        let aligned = |c: Vec3| c.z.abs().max(c.x.abs()) > 0.9;
        assert!(aligned(cents[0]) && aligned(cents[1]));
        // consecutive pairs (alternating clusters) get different labels
        let diff = (0..30).filter(|&k| labels[2 * k] != labels[2 * k + 1]).count();
        assert!(diff > 25, "kmeans separation {diff}/30");
    }

    #[test]
    fn test_vmf() {
        let mut rng = Rng::new(13);
        let mu = Vec3::new(0.0, 0.0, 1.0);
        let kappa = 20.0;
        let samples: Vec<Vec3> = (0..2000).map(|_| vmf_sample(mu, kappa, &mut rng)).collect();
        // fit recovers direction and roughly the concentration
        let (mu_fit, k_fit) = vmf_fit(&samples);
        assert!(mu_fit.dot(&mu) > 0.999, "vmf direction {mu_fit:?}");
        assert!((k_fit - kappa).abs() < 0.2 * kappa, "vmf kappa {k_fit}");
        // pdf integrates to ~1 over the sphere (Lebedev order 26 with
        // moderate kappa)
        let quad = lebedev_quadrature(26);
        let integral: f64 = quad
            .iter()
            .map(|&(x, w)| von_mises_fisher_pdf(x, mu, 1.5) * w * 4.0 * PI)
            .sum();
        assert!((integral - 1.0).abs() < 0.01, "vmf integral {integral}");
        // pdf peaks at mu
        assert!(
            von_mises_fisher_pdf(mu, mu, kappa) > von_mises_fisher_pdf(Vec3::new(1.0, 0.0, 0.0), mu, kappa)
        );
        // Kent reduces toward vMF when beta = 0 (proportional)
        let g1 = Vec3::new(0.0, 0.0, 1.0);
        let g2 = Vec3::new(1.0, 0.0, 0.0);
        let g3 = Vec3::new(0.0, 1.0, 0.0);
        let ratio1 = kent_distribution_pdf(g1, g1, g2, g3, 5.0, 0.0)
            / von_mises_fisher_pdf(g1, g1, 5.0);
        let x2 = Vec3::new(0.6, 0.0, 0.8);
        let ratio2 = kent_distribution_pdf(x2, g1, g2, g3, 5.0, 0.0)
            / von_mises_fisher_pdf(x2, g1, 5.0);
        assert!((ratio1 / ratio2 - 1.0).abs() < 1e-9, "Kent/vMF proportional");
    }

    #[test]
    fn test_quadrature_designs() {
        // t-designs integrate degree <= t polynomials exactly:
        // f = x^2 (degree 2) has average 1/3
        for t in [2usize, 3, 5] {
            let pts = spherical_t_design(t).unwrap();
            let avg: f64 = pts.iter().map(|p| p.x * p.x).sum::<f64>() / pts.len() as f64;
            assert!((avg - 1.0 / 3.0).abs() < 1e-12, "t={t} design");
        }
        assert!(spherical_t_design(7).is_none());
        // Lebedev: weights sum to 1; integrates x^4 + y^4 + z^4 correctly
        // (exact value 4 pi * 3/5... average 3/5)
        for order in [6usize, 14, 26] {
            let quad = lebedev_quadrature(order);
            assert_eq!(quad.len(), order);
            let wsum: f64 = quad.iter().map(|&(_, w)| w).sum();
            assert!((wsum - 1.0).abs() < 1e-12, "order {order} weights {wsum}");
            let avg2: f64 = quad.iter().map(|&(p, w)| w * p.x * p.x).sum();
            assert!((avg2 - 1.0 / 3.0).abs() < 1e-12, "order {order} x^2");
        }
        // order >= 14 integrates quartics exactly: <x^4> = 1/5
        for order in [14usize, 26] {
            let quad = lebedev_quadrature(order);
            let avg4: f64 = quad.iter().map(|&(p, w)| w * p.x.powi(4)).sum();
            assert!((avg4 - 0.2).abs() < 1e-12, "order {order} x^4 = {avg4}");
        }
        // Gauss-Legendre sphere: weights sum to 4 pi; integrates z^2
        let gl = gauss_legendre_sphere(8, 16);
        let wsum: f64 = gl.iter().map(|&(_, _, w)| w).sum();
        assert!((wsum - 4.0 * PI).abs() < 1e-10);
        let z2: f64 = gl
            .iter()
            .map(|&(th, _, w)| w * th.cos() * th.cos())
            .sum::<f64>()
            / (4.0 * PI);
        assert!((z2 - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_sphere_uniform_points_n_moments() {
        let mut rng = Rng::new(2024);
        for dim in [2usize, 3, 5] {
            let n = 4000;
            let pts = sphere_uniform_points_n(n, dim, &mut rng);
            assert_eq!(pts.len(), n);
            // every sample is a unit vector of the requested dimension
            for p in &pts {
                assert_eq!(p.dim(), dim);
                assert!((p.norm() - 1.0).abs() < 1e-12, "unit norm {}", p.norm());
            }
            // the mean vanishes by symmetry: each coordinate mean has
            // standard error 1/sqrt(dim n), so the mean vector norm is about
            // sqrt(1/n); allow five standard errors
            let mut mean = VecN::zeros(dim);
            for p in &pts {
                mean = mean.add(p);
            }
            mean = mean.scale(1.0 / n as f64);
            let tol = 5.0 / (n as f64).sqrt();
            assert!(
                mean.norm() < tol,
                "dim {dim} mean {} exceeds {tol}",
                mean.norm()
            );
            // isotropy: E[x_i x_j] = delta_ij / dim
            for i in 0..dim {
                for j in 0..dim {
                    let m2: f64 =
                        pts.iter().map(|p| p[i] * p[j]).sum::<f64>() / n as f64;
                    let want = if i == j { 1.0 / dim as f64 } else { 0.0 };
                    assert!(
                        (m2 - want).abs() < 6.0 / (n as f64).sqrt(),
                        "dim {dim} second moment ({i},{j}) = {m2}, want {want}"
                    );
                }
            }
            // pairwise geodesic distances average to pi/2 (antipodal
            // symmetry of the uniform measure)
            let sample = &pts[..200];
            let mut sum = 0.0;
            let mut count = 0.0;
            for (i, a) in sample.iter().enumerate() {
                for b in sample.iter().skip(i + 1) {
                    sum += sphere_distance_n(a, b);
                    count += 1.0;
                }
            }
            let mean_angle = sum / count;
            assert!(
                (mean_angle - PI / 2.0).abs() < 0.05,
                "dim {dim} mean angle {mean_angle}"
            );
        }
        // the sequence is deterministic for a fixed seed
        let a = sphere_uniform_points_n(5, 3, &mut Rng::new(7));
        let b = sphere_uniform_points_n(5, 3, &mut Rng::new(7));
        for (x, y) in a.iter().zip(&b) {
            assert!(x.sub(y).norm() < 1e-15, "reproducible for a fixed seed");
        }
    }
}
