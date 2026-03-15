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

    pub fn is_bound(&self) -> bool {
        self.eccentricity < 1.0
    }

    pub fn period(&self, mu: f64) -> f64 {
        2.0 * PI * (self.semi_major_axis.powi(3) / mu).sqrt()
    }

    pub fn periapsis(&self) -> f64 {
        periapsis(self.semi_major_axis, self.eccentricity)
    }

    pub fn apoapsis(&self) -> Option<f64> {
        if self.eccentricity < 1.0 {
            Some(apoapsis(self.semi_major_axis, self.eccentricity))
        } else {
            None
        }
    }
}

pub fn specific_orbital_energy(mu: f64, r: f64, v: f64) -> f64 {
    0.5 * v * v - mu / r
}

pub fn specific_angular_momentum(position: Vec3, velocity: Vec3) -> Vec3 {
    position.cross(&velocity)
}

pub fn eccentricity_vector(position: Vec3, velocity: Vec3, mu: f64) -> Vec3 {
    let r = position.magnitude();
    let h = position.cross(&velocity);
    let v_cross_h = velocity.cross(&h);
    let r_hat = position * (1.0 / r);
    v_cross_h * (1.0 / mu) - r_hat
}

pub fn eccentricity(position: Vec3, velocity: Vec3, mu: f64) -> f64 {
    eccentricity_vector(position, velocity, mu).magnitude()
}

pub fn semi_major_axis(mu: f64, energy: f64) -> f64 {
    if energy.abs() < 1e-20 {
        return f64::INFINITY;
    }
    -mu / (2.0 * energy)
}

pub fn semi_minor_axis(semi_major: f64, ecc: f64) -> f64 {
    semi_major * (1.0 - ecc * ecc).max(0.0).sqrt()
}

pub fn periapsis(semi_major: f64, ecc: f64) -> f64 {
    semi_major * (1.0 - ecc)
}

pub fn apoapsis(semi_major: f64, ecc: f64) -> f64 {
    semi_major * (1.0 + ecc)
}

pub fn true_anomaly(position: Vec3, velocity: Vec3, mu: f64) -> f64 {
    let e_vec = eccentricity_vector(position, velocity, mu);
    let ecc = e_vec.magnitude();
    if ecc < 1e-12 {
        return 0.0;
    }
    let r = position.magnitude();
    let cos_nu = e_vec.dot(&position) / (ecc * r);
    let cos_clamped = cos_nu.clamp(-1.0, 1.0);
    let nu = cos_clamped.acos();

    if position.dot(&velocity) < 0.0 {
        2.0 * PI - nu
    } else {
        nu
    }
}

pub fn inclination(angular_momentum: Vec3) -> f64 {
    let h = angular_momentum.magnitude();
    if h < 1e-20 {
        return 0.0;
    }
    (angular_momentum.z / h).clamp(-1.0, 1.0).acos()
}

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
}
