use crate::math::Vec3;
use crate::math::constants::PI;

#[derive(Debug, Clone, Copy)]
pub struct OrbitalElements {
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub inclination: f64,
    pub longitude_ascending_node: f64,
    pub argument_periapsis: f64,
    pub true_anomaly: f64,
}

impl OrbitalElements {
    /// Converts Cartesian state vectors (position, velocity) to Keplerian orbital elements for gravitational parameter μ.
    pub fn from_state_vectors(position: Vec3, velocity: Vec3, mu: f64) -> Self {
        let r = position.magnitude();
        let v = velocity.magnitude();

        let h_vec = specific_angular_momentum(position, velocity);
        let energy = specific_orbital_energy(mu, r, v);
        let e_vec = eccentricity_vector(position, velocity, mu);
        let ecc = e_vec.magnitude();

        let sma = if energy.abs() > 1e-20 {
            -mu / (2.0 * energy)
        } else {
            f64::INFINITY
        };

        let inc = inclination(h_vec);
        let loan = longitude_of_ascending_node(h_vec);
        let aop = argument_of_periapsis(h_vec, e_vec);
        let ta = true_anomaly(position, velocity, mu);

        Self {
            semi_major_axis: sma,
            eccentricity: ecc,
            inclination: inc,
            longitude_ascending_node: loan,
            argument_periapsis: aop,
            true_anomaly: ta,
        }
    }

    /// Returns true if the orbit is gravitationally bound: e < 1.
    pub fn is_bound(&self) -> bool {
        self.eccentricity < 1.0
    }

    /// Computes the orbital period: T = 2π√(a³/μ).
    pub fn period(&self, mu: f64) -> f64 {
        assert!(mu > 0.0, "gravitational parameter must be positive");
        2.0 * PI * (self.semi_major_axis.powi(3) / mu).sqrt()
    }

    /// Returns the periapsis distance: r_p = a(1 - e).
    pub fn periapsis(&self) -> f64 {
        periapsis(self.semi_major_axis, self.eccentricity)
    }

    /// Returns the apoapsis distance if bound (e < 1): r_a = a(1 + e). Returns None for unbound orbits.
    pub fn apoapsis(&self) -> Option<f64> {
        if self.eccentricity < 1.0 {
            Some(apoapsis(self.semi_major_axis, self.eccentricity))
        } else {
            None
        }
    }
}

/// Computes specific orbital energy: ε = v²/2 - μ/r.
pub fn specific_orbital_energy(mu: f64, r: f64, v: f64) -> f64 {
    assert!(r > 0.0, "orbital radius must be positive");
    0.5 * v * v - mu / r
}

/// Computes specific angular momentum vector: h = r × v.
pub fn specific_angular_momentum(position: Vec3, velocity: Vec3) -> Vec3 {
    position.cross(&velocity)
}

/// Computes the eccentricity vector: e = (v × h)/μ - r̂, pointing toward periapsis.
pub fn eccentricity_vector(position: Vec3, velocity: Vec3, mu: f64) -> Vec3 {
    let r = position.magnitude();
    assert!(r > 0.0, "position magnitude must be positive");
    assert!(mu != 0.0, "gravitational parameter must be non-zero");
    let h = position.cross(&velocity);
    let v_cross_h = velocity.cross(&h);
    let r_hat = position * (1.0 / r);
    v_cross_h * (1.0 / mu) - r_hat
}

/// Computes the orbital eccentricity as the magnitude of the eccentricity vector: e = |e_vec|.
pub fn eccentricity(position: Vec3, velocity: Vec3, mu: f64) -> f64 {
    eccentricity_vector(position, velocity, mu).magnitude()
}

/// Computes the semi-major axis from the vis-viva relation: a = -μ/(2ε). Returns infinity for parabolic orbits.
pub fn semi_major_axis(mu: f64, energy: f64) -> f64 {
    if energy.abs() < 1e-20 {
        return f64::INFINITY;
    }
    -mu / (2.0 * energy)
}

/// Computes the semi-minor axis: b = a√(1 - e²).
pub fn semi_minor_axis(semi_major: f64, ecc: f64) -> f64 {
    semi_major * (1.0 - ecc * ecc).max(0.0).sqrt()
}

/// Computes the periapsis distance: r_p = a(1 - e).
pub fn periapsis(semi_major: f64, ecc: f64) -> f64 {
    semi_major * (1.0 - ecc)
}

/// Computes the apoapsis distance: r_a = a(1 + e).
pub fn apoapsis(semi_major: f64, ecc: f64) -> f64 {
    semi_major * (1.0 + ecc)
}

/// Computes the true anomaly ν from state vectors: ν = acos(e · r / (|e||r|)), adjusted for quadrant.
pub fn true_anomaly(position: Vec3, velocity: Vec3, mu: f64) -> f64 {
    let e_vec = eccentricity_vector(position, velocity, mu);
    let ecc = e_vec.magnitude();
    if ecc < 1e-12 {
        return 0.0;
    }
    let r = position.magnitude();
    assert!(r > 0.0, "position magnitude must be positive");
    let cos_nu = e_vec.dot(&position) / (ecc * r);
    let cos_clamped = cos_nu.clamp(-1.0, 1.0);
    let nu = cos_clamped.acos();

    if position.dot(&velocity) < 0.0 {
        2.0 * PI - nu
    } else {
        nu
    }
}

/// Computes the orbital inclination: i = acos(h_z / |h|).
pub fn inclination(angular_momentum: Vec3) -> f64 {
    let h = angular_momentum.magnitude();
    if h < 1e-20 {
        return 0.0;
    }
    (angular_momentum.z / h).clamp(-1.0, 1.0).acos()
}

/// Computes the longitude of the ascending node Ω from the nodal vector n = (-h_y, h_x, 0).
pub fn longitude_of_ascending_node(angular_momentum: Vec3) -> f64 {
    let n = Vec3::new(-angular_momentum.y, angular_momentum.x, 0.0);
    let n_mag = n.magnitude();
    if n_mag < 1e-20 {
        return 0.0;
    }
    let omega = (n.x / n_mag).clamp(-1.0, 1.0).acos();
    if n.y < 0.0 {
        2.0 * PI - omega
    } else {
        omega
    }
}

/// Computes the argument of periapsis ω: ω = acos(n · e / (|n||e|)), adjusted for quadrant.
pub fn argument_of_periapsis(angular_momentum: Vec3, ecc_vec: Vec3) -> f64 {
    let n = Vec3::new(-angular_momentum.y, angular_momentum.x, 0.0);
    let n_mag = n.magnitude();
    let e_mag = ecc_vec.magnitude();
    if n_mag < 1e-20 || e_mag < 1e-20 {
        return 0.0;
    }
    let cos_w = n.dot(&ecc_vec) / (n_mag * e_mag);
    let w = cos_w.clamp(-1.0, 1.0).acos();
    if ecc_vec.z < 0.0 {
        2.0 * PI - w
    } else {
        w
    }
}

/// Generates 3D points along an elliptical orbit using the conic section r = p/(1 + e cos θ).
pub fn orbit_points_ellipse(elements: &OrbitalElements, _mu: f64, num_points: usize) -> Vec<Vec3> {
    if elements.eccentricity >= 1.0 || num_points == 0 {
        return Vec::new();
    }

    let a = elements.semi_major_axis;
    let ecc = elements.eccentricity;
    let p = a * (1.0 - ecc * ecc);

    let (frame_x, frame_y) = orbital_frame(elements);

    let mut points = Vec::with_capacity(num_points + 1);
    for i in 0..=num_points {
        let theta = (i as f64 / num_points as f64) * 2.0 * PI;
        let r = p / (1.0 + ecc * theta.cos());
        let x = r * theta.cos();
        let y = r * theta.sin();
        points.push(frame_x * x + frame_y * y);
    }
    points
}

/// Generates 3D points along a hyperbolic orbit trajectory using r = p/(1 + e cos θ) with θ bounded by the asymptotes.
pub fn orbit_points_hyperbola(elements: &OrbitalElements, _mu: f64, num_points: usize) -> Vec<Vec3> {
    if elements.eccentricity <= 1.0 || num_points == 0 {
        return Vec::new();
    }

    let a = elements.semi_major_axis.abs();
    let ecc = elements.eccentricity;
    let p = a * (ecc * ecc - 1.0);
    let theta_max = (-1.0 / ecc).acos() * 0.95;

    let (frame_x, frame_y) = orbital_frame(elements);

    let mut points = Vec::with_capacity(num_points + 1);
    for i in 0..=num_points {
        let theta = -theta_max + (2.0 * theta_max * i as f64) / num_points as f64;
        let denom = 1.0 + ecc * theta.cos();
        if denom <= 0.0 { continue; }
        let r = p / denom;
        let x = r * theta.cos();
        let y = r * theta.sin();
        points.push(frame_x * x + frame_y * y);
    }
    points
}

fn orbital_frame(elements: &OrbitalElements) -> (Vec3, Vec3) {
    let i = elements.inclination;
    let omega = elements.longitude_ascending_node;
    let w = elements.argument_periapsis;

    let cos_o = omega.cos();
    let sin_o = omega.sin();
    let cos_i = i.cos();
    let sin_i = i.sin();
    let cos_w = w.cos();
    let sin_w = w.sin();

    let px = cos_o * cos_w - sin_o * sin_w * cos_i;
    let py = sin_o * cos_w + cos_o * sin_w * cos_i;
    let pz = sin_w * sin_i;

    let qx = -cos_o * sin_w - sin_o * cos_w * cos_i;
    let qy = -sin_o * sin_w + cos_o * cos_w * cos_i;
    let qz = cos_w * sin_i;

    (Vec3::new(px, py, pz), Vec3::new(qx, qy, qz))
}

/// Returns true if the specific orbital energy indicates a bound orbit: ε < 0.
pub fn is_bound(energy: f64) -> bool {
    energy < 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::G;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_circular_orbit() {
        let mu = G * 1.989e30; // Sun
        let r = 1.496e11; // 1 AU
        let v = (mu / r).sqrt(); // circular velocity

        let pos = Vec3::new(r, 0.0, 0.0);
        let vel = Vec3::new(0.0, v, 0.0);

        let elements = OrbitalElements::from_state_vectors(pos, vel, mu);
        assert!(approx(elements.eccentricity, 0.0, 1e-6), "ecc = {}", elements.eccentricity);
        assert!(approx(elements.semi_major_axis, r, r * 1e-6));
        assert!(elements.is_bound());
    }

    #[test]
    fn test_periapsis_apoapsis() {
        assert!(approx(periapsis(10.0, 0.5), 5.0, 1e-9));
        assert!(approx(apoapsis(10.0, 0.5), 15.0, 1e-9));
    }

    #[test]
    fn test_semi_minor() {
        let b = semi_minor_axis(10.0, 0.0);
        assert!(approx(b, 10.0, 1e-9));
    }

    #[test]
    fn test_ellipse_points_close() {
        let elements = OrbitalElements {
            semi_major_axis: 1.0,
            eccentricity: 0.5,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let pts = orbit_points_ellipse(&elements, 1.0, 100);
        assert_eq!(pts.len(), 101);
        let first = pts.first().unwrap();
        let last = pts.last().unwrap();
        assert!((*first - *last).magnitude() < 1e-10, "Ellipse should close");
    }

    #[test]
    fn test_specific_orbital_energy_bound_orbit() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let v = (mu / r).sqrt();
        let energy = specific_orbital_energy(mu, r, v);
        assert!(energy < 0.0, "Circular orbit should be bound, got energy = {energy}");
    }

    #[test]
    fn test_specific_orbital_energy_unbound() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let v = (3.0 * mu / r).sqrt();
        let energy = specific_orbital_energy(mu, r, v);
        assert!(energy > 0.0, "Hyperbolic orbit should be unbound, got energy = {energy}");
    }

    #[test]
    fn test_specific_angular_momentum_circular() {
        let r = 1.0e11;
        let v = 3.0e4;
        let pos = Vec3::new(r, 0.0, 0.0);
        let vel = Vec3::new(0.0, v, 0.0);
        let h = specific_angular_momentum(pos, vel);
        assert!(approx(h.z, r * v, 1e-3), "h.z = {}, expected {}", h.z, r * v);
        assert!(approx(h.x, 0.0, 1e-3));
        assert!(approx(h.y, 0.0, 1e-3));
    }

    #[test]
    fn test_eccentricity_vector_circular() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let v = (mu / r).sqrt();
        let pos = Vec3::new(r, 0.0, 0.0);
        let vel = Vec3::new(0.0, v, 0.0);
        let e_vec = eccentricity_vector(pos, vel, mu);
        assert!(e_vec.magnitude() < 1e-6, "Circular orbit eccentricity vector should be ~0, got {}", e_vec.magnitude());
    }

    #[test]
    fn test_eccentricity_circular() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let v = (mu / r).sqrt();
        let pos = Vec3::new(r, 0.0, 0.0);
        let vel = Vec3::new(0.0, v, 0.0);
        let ecc = eccentricity(pos, vel, mu);
        assert!(ecc < 1e-6, "Circular orbit eccentricity should be ~0, got {ecc}");
    }

    #[test]
    fn test_semi_major_axis_from_energy() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let v = (mu / r).sqrt();
        let energy = specific_orbital_energy(mu, r, v);
        let a = semi_major_axis(mu, energy);
        assert!(approx(a, r, r * 1e-6), "Semi-major axis should equal r for circular orbit, got {a} vs {r}");
    }

    #[test]
    fn test_semi_major_axis_parabolic() {
        let a = semi_major_axis(1.0, 0.0);
        assert!(a.is_infinite(), "Parabolic orbit should have infinite semi-major axis");
    }

    #[test]
    fn test_inclination_equatorial() {
        let h = Vec3::new(0.0, 0.0, 1.0);
        let inc = inclination(h);
        assert!(approx(inc, 0.0, 1e-12), "Equatorial orbit inclination should be 0, got {inc}");
    }

    #[test]
    fn test_inclination_polar() {
        let h = Vec3::new(1.0, 0.0, 0.0);
        let inc = inclination(h);
        assert!(approx(inc, PI / 2.0, 1e-12), "Polar orbit inclination should be pi/2, got {inc}");
    }

    #[test]
    fn test_longitude_of_ascending_node_reference() {
        let h = Vec3::new(0.0, 0.0, 1.0);
        let loan = longitude_of_ascending_node(h);
        assert!(approx(loan, 0.0, 1e-12), "Equatorial orbit should have LOAN=0, got {loan}");
    }

    #[test]
    fn test_longitude_of_ascending_node_inclined() {
        let h = Vec3::new(0.0, -1.0, 1.0);
        let loan = longitude_of_ascending_node(h);
        // n = (-h_y, h_x, 0) = (1, 0, 0), so omega = acos(1) = 0
        assert!(approx(loan, 0.0, 1e-12), "LOAN should be 0 for n along x-axis, got {loan}");
    }

    #[test]
    fn test_argument_of_periapsis_zero_ecc() {
        let h = Vec3::new(0.0, 0.0, 1.0);
        let e_vec = Vec3::ZERO;
        let aop = argument_of_periapsis(h, e_vec);
        assert!(approx(aop, 0.0, 1e-12), "Zero eccentricity should give aop=0, got {aop}");
    }

    #[test]
    fn test_argument_of_periapsis_known() {
        let h = Vec3::new(0.0, -1.0, 1.0);
        // n = (1, 0, 0), e = (1, 0, 0) => aop = acos(1) = 0
        let e_vec = Vec3::new(1.0, 0.0, 0.0);
        let aop = argument_of_periapsis(h, e_vec);
        assert!(approx(aop, 0.0, 1e-12), "aop should be 0 when n and e are aligned, got {aop}");
    }

    #[test]
    fn test_true_anomaly_at_periapsis() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let v = (mu / r).sqrt();
        let pos = Vec3::new(r, 0.0, 0.0);
        let vel = Vec3::new(0.0, v, 0.0);
        let ta = true_anomaly(pos, vel, mu);
        // For circular orbit eccentricity ~0, true_anomaly returns 0
        assert!(ta < 0.01 || ta > 2.0 * PI - 0.01, "True anomaly at periapsis should be ~0, got {ta}");
    }

    #[test]
    fn test_period_circular() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let elements = OrbitalElements {
            semi_major_axis: r,
            eccentricity: 0.0,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let t = elements.period(mu);
        // Earth's orbital period ~3.156e7 s (1 year)
        let year_s = 3.156e7;
        let rel_err = (t - year_s).abs() / year_s;
        assert!(rel_err < 0.01, "Period should be ~1 year, got {t} s, error {rel_err}");
    }

    #[test]
    fn test_orbit_points_hyperbola_generates_points() {
        let elements = OrbitalElements {
            semi_major_axis: -1.0,
            eccentricity: 2.0,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let pts = orbit_points_hyperbola(&elements, 1.0, 50);
        assert!(!pts.is_empty(), "Hyperbolic orbit should generate points");
        assert!(pts.len() <= 51);
    }

    #[test]
    fn test_orbit_points_hyperbola_rejects_elliptical() {
        let elements = OrbitalElements {
            semi_major_axis: 1.0,
            eccentricity: 0.5,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let pts = orbit_points_hyperbola(&elements, 1.0, 50);
        assert!(pts.is_empty(), "Should return empty for e <= 1.0");
    }

    #[test]
    fn test_is_bound_free_function() {
        assert!(is_bound(-1.0e10));
        assert!(!is_bound(0.0));
        assert!(!is_bound(1.0e10));
    }

    #[test]
    fn test_orbital_elements_periapsis_method() {
        // ISS-like orbit: a = 6_781 km, e = 0.0001
        let elements = OrbitalElements {
            semi_major_axis: 6_781_000.0,
            eccentricity: 0.0001,
            inclination: 0.9006, // 51.6 deg
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let rp = elements.periapsis();
        let expected = 6_781_000.0 * (1.0 - 0.0001);
        assert!(approx(rp, expected, 1.0), "periapsis = {rp}, expected {expected}");
    }

    #[test]
    fn test_orbital_elements_apoapsis_bound() {
        // Moon orbit: a = 384_400 km, e = 0.0549
        let elements = OrbitalElements {
            semi_major_axis: 384_400_000.0,
            eccentricity: 0.0549,
            inclination: 0.0898, // 5.145 deg
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let ra = elements.apoapsis();
        assert!(ra.is_some());
        let expected = 384_400_000.0 * (1.0 + 0.0549);
        assert!(approx(ra.unwrap(), expected, 1.0));
    }

    #[test]
    fn test_orbital_elements_apoapsis_unbound() {
        let elements = OrbitalElements {
            semi_major_axis: -1_000_000.0,
            eccentricity: 1.5,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        assert!(elements.apoapsis().is_none());
    }

    #[test]
    fn test_true_anomaly_eccentric_orbit() {
        // Elliptical orbit at periapsis: position along +x, velocity along +y
        // with speed > circular velocity to create eccentricity
        let mu = G * 5.972e24; // Earth
        let r = 7_000_000.0; // 7000 km
        let v_circ = (mu / r).sqrt();
        let v = v_circ * 1.2; // faster than circular -> eccentric orbit

        let pos = Vec3::new(r, 0.0, 0.0);
        let vel = Vec3::new(0.0, v, 0.0);
        let ta = true_anomaly(pos, vel, mu);
        // At periapsis with positive radial velocity component = 0, true anomaly should be 0
        assert!(ta < 0.1, "True anomaly at periapsis should be near 0, got {ta}");
    }

    #[test]
    fn test_true_anomaly_past_periapsis_negative_rdot() {
        // Moving toward periapsis: position.dot(velocity) < 0
        // Place body at +x, velocity mostly -x (approaching) and +y
        let mu = G * 5.972e24;
        let r = 7_000_000.0;
        let v_circ = (mu / r).sqrt();

        let pos = Vec3::new(r, 0.0, 0.0);
        // Give it a large inward radial component so r_dot < 0
        let vel = Vec3::new(-v_circ * 0.5, v_circ * 0.8, 0.0);
        let ta = true_anomaly(pos, vel, mu);
        // With r_dot < 0, true anomaly should be in (PI, 2*PI)
        assert!(ta > PI, "True anomaly should be > PI when approaching periapsis, got {ta}");
    }

    #[test]
    fn test_inclination_zero_angular_momentum() {
        let h = Vec3::new(0.0, 0.0, 0.0);
        let inc = inclination(h);
        assert!(approx(inc, 0.0, 1e-12), "Zero h should give inclination 0, got {inc}");
    }

    #[test]
    fn test_longitude_of_ascending_node_negative_ny() {
        // h = (1, 0, 1) => n = (-h_y, h_x, 0) = (0, 1, 0)
        // n.y = 1 > 0, omega = acos(0/1) = PI/2
        // We need n.y < 0: h = (-1, 0, 1) => n = (0, -1, 0), n.y = -1 < 0
        // omega = acos(0/1) = PI/2, result = 2*PI - PI/2 = 3*PI/2
        let h = Vec3::new(-1.0, 0.0, 1.0);
        let loan = longitude_of_ascending_node(h);
        let expected = 2.0 * PI - (PI / 2.0);
        assert!(approx(loan, expected, 1e-12), "LOAN should be 3*PI/2, got {loan}");
    }

    #[test]
    fn test_argument_of_periapsis_negative_ez() {
        // Need ecc_vec.z < 0 and non-zero n_mag and e_mag
        // h = (0, -1, 1) => n = (1, 0, 0)
        let h = Vec3::new(0.0, -1.0, 1.0);
        let e_vec = Vec3::new(0.0, 0.0, -1.0);
        let aop = argument_of_periapsis(h, e_vec);
        // cos_w = n.dot(e) / (|n|*|e|) = 0, so w = PI/2
        // ecc_vec.z < 0, so result = 2*PI - PI/2 = 3*PI/2
        let expected = 2.0 * PI - (PI / 2.0);
        assert!(approx(aop, expected, 1e-12), "aop should be 3*PI/2, got {aop}");
    }

    #[test]
    fn test_orbit_points_ellipse_rejects_hyperbolic() {
        let elements = OrbitalElements {
            semi_major_axis: 1.0,
            eccentricity: 1.5,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let pts = orbit_points_ellipse(&elements, 1.0, 100);
        assert!(pts.is_empty(), "Should return empty for e >= 1.0");
    }

    #[test]
    fn test_orbit_points_ellipse_zero_points() {
        let elements = OrbitalElements {
            semi_major_axis: 1.0,
            eccentricity: 0.5,
            inclination: 0.0,
            longitude_ascending_node: 0.0,
            argument_periapsis: 0.0,
            true_anomaly: 0.0,
        };
        let pts = orbit_points_ellipse(&elements, 1.0, 0);
        assert!(pts.is_empty(), "Should return empty for num_points == 0");
    }

    #[test]
    fn test_from_state_vectors_parabolic() {
        // Parabolic orbit: energy = 0, v = sqrt(2*mu/r) (escape velocity)
        let mu = G * 5.972e24;
        let r = 7_000_000.0;
        let v_esc = (2.0 * mu / r).sqrt();

        let pos = Vec3::new(r, 0.0, 0.0);
        let vel = Vec3::new(0.0, v_esc, 0.0);
        let elements = OrbitalElements::from_state_vectors(pos, vel, mu);
        assert!(elements.semi_major_axis.is_infinite() || elements.semi_major_axis.abs() > 1e20,
            "Parabolic orbit should have infinite or very large semi-major axis, got {}", elements.semi_major_axis);
    }
}
