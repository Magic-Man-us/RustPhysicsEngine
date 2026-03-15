use crate::math::Vec3;
use crate::math::constants::G;

pub const DEFAULT_DT: f64 = 0.0001;
pub const DEFAULT_SOFTENING: f64 = 0.05;

#[derive(Debug, Clone)]
pub struct Body {
    pub id: u32,
    pub mass: f64,
    pub radius: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
}

impl Body {
    pub fn new(id: u32, mass: f64, radius: f64, position: Vec3, velocity: Vec3) -> Self {
        Self {
            id,
            mass,
            radius,
            position,
            velocity,
            acceleration: Vec3::ZERO,
        }
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.magnitude_squared()
    }
}

pub struct NBodySystem {
    pub bodies: Vec<Body>,
    pub dt: f64,
    pub softening: f64,
    pub time: f64,
}

impl NBodySystem {
    pub fn new(bodies: Vec<Body>, dt: f64, softening: f64) -> Self {
        let mut system = Self {
            bodies,
            dt,
            softening,
            time: 0.0,
        };
        init_accelerations(&mut system.bodies, system.softening);
        system
    }

    pub fn step(&mut self) {
        step_verlet(&mut self.bodies, self.dt, self.softening);
        self.time += self.dt;
    }

    pub fn total_energy(&self) -> f64 {
        total_energy(&self.bodies, self.softening)
    }

    pub fn center_of_mass(&self) -> Vec3 {
        center_of_mass(&self.bodies)
    }

    pub fn total_momentum(&self) -> Vec3 {
        total_momentum(&self.bodies)
    }
}

pub fn compute_acceleration(bodies: &[Body], idx: usize, softening: f64) -> Vec3 {
    let bi = &bodies[idx];
    let soft_sq = softening * softening;
    let mut ax = 0.0_f64;
    let mut ay = 0.0_f64;
    let mut az = 0.0_f64;

    for (j, bj) in bodies.iter().enumerate() {
        if j == idx {
            continue;
        }
        let dx = bj.position.x - bi.position.x;
        let dy = bj.position.y - bi.position.y;
        let dz = bj.position.z - bi.position.z;

        let dist_sq = dx * dx + dy * dy + dz * dz + soft_sq;
        let inv_dist3 = 1.0 / (dist_sq * dist_sq.sqrt());

        let f = G * bj.mass * inv_dist3;
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }

    Vec3::new(ax, ay, az)
}

pub fn init_accelerations(bodies: &mut [Body], softening: f64) {
    let accels: Vec<Vec3> = (0..bodies.len())
        .map(|i| compute_acceleration(bodies, i, softening))
        .collect();
    for (i, acc) in accels.into_iter().enumerate() {
        bodies[i].acceleration = acc;
    }
}

pub fn step_verlet(bodies: &mut [Body], dt: f64, softening: f64) {
    let n = bodies.len();
    let half_dt = 0.5 * dt;

    // Half-kick + drift
    for i in 0..n {
        let acc = bodies[i].acceleration;
        bodies[i].velocity = bodies[i].velocity + acc * half_dt;
        let vel = bodies[i].velocity;
        bodies[i].position = bodies[i].position + vel * dt;
    }

    // Recompute accelerations at new positions
    let new_accels: Vec<Vec3> = (0..n)
        .map(|i| compute_acceleration(bodies, i, softening))
        .collect();

    // Second half-kick
    for i in 0..n {
        bodies[i].velocity = bodies[i].velocity + new_accels[i] * half_dt;
        bodies[i].acceleration = new_accels[i];
    }
}

pub fn kinetic_energy(bodies: &[Body]) -> f64 {
    bodies.iter().map(|b| b.kinetic_energy()).sum()
}

pub fn potential_energy(bodies: &[Body], softening: f64) -> f64 {
    let soft_sq = softening * softening;
    let mut pe = 0.0;
    for (i, bi) in bodies.iter().enumerate() {
        for bj in bodies.iter().skip(i + 1) {
            let r_sq = (bi.position - bj.position).magnitude_squared() + soft_sq;
            pe -= G * bi.mass * bj.mass / r_sq.sqrt();
        }
    }
    pe
}

pub fn total_energy(bodies: &[Body], softening: f64) -> f64 {
    kinetic_energy(bodies) + potential_energy(bodies, softening)
}

pub fn center_of_mass(bodies: &[Body]) -> Vec3 {
    let mut total_mass = 0.0;
    let mut com = Vec3::ZERO;
    for b in bodies {
        com = com + b.position * b.mass;
        total_mass += b.mass;
    }
    if total_mass > 0.0 {
        com * (1.0 / total_mass)
    } else {
        Vec3::ZERO
    }
}

pub fn total_momentum(bodies: &[Body]) -> Vec3 {
    let mut p = Vec3::ZERO;
    for b in bodies {
        p = p + b.velocity * b.mass;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_two_body_energy_conservation() {
        let mut bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)),
            Body::new(1, 1.0e24, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::new(0.0, 3.0e4, 0.0)),
        ];
        init_accelerations(&mut bodies, DEFAULT_SOFTENING);
        let e0 = total_energy(&bodies, DEFAULT_SOFTENING);

        for _ in 0..100 {
            step_verlet(&mut bodies, DEFAULT_DT, DEFAULT_SOFTENING);
        }

        let e1 = total_energy(&bodies, DEFAULT_SOFTENING);
        let rel_err = ((e1 - e0) / e0).abs();
        assert!(rel_err < 1e-4, "Energy not conserved: relative error = {rel_err}");
    }

    #[test]
    fn test_momentum_conservation() {
        let mut bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)),
            Body::new(1, 1.0e30, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::new(-10.0, 0.0, 0.0)),
        ];
        init_accelerations(&mut bodies, DEFAULT_SOFTENING);
        let p0 = total_momentum(&bodies);

        for _ in 0..100 {
            step_verlet(&mut bodies, DEFAULT_DT, DEFAULT_SOFTENING);
        }

        let p1 = total_momentum(&bodies);
        assert!(approx(p0.x, p1.x, 1e-6), "Momentum x not conserved");
        assert!(approx(p0.y, p1.y, 1e-6), "Momentum y not conserved");
        assert!(approx(p0.z, p1.z, 1e-6), "Momentum z not conserved");
    }

    #[test]
    fn test_center_of_mass() {
        let bodies = vec![
            Body::new(0, 2.0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 2.0, 1.0, Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let com = center_of_mass(&bodies);
        assert!(approx(com.x, 2.0, 1e-9));
        assert!(approx(com.y, 0.0, 1e-9));
    }
}
