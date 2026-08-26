//! Direct N-body gravitational simulation.
//!
//! Velocity Verlet integration, chosen because it is symplectic: it
//! conserves a nearby "shadow" energy exactly rather than drifting, so
//! orbits stay closed over long integrations where Runge-Kutta of the same
//! order would spiral.
//!
//! Softening replaces `1/r²` with `1/(r² + ε²)` to keep close encounters
//! from producing unbounded accelerations, at the cost of biasing the
//! force at short range. Includes energy and momentum diagnostics, and
//! system generators.
//!
//! Cost is O(N²) per step. For large N use [`crate::astrophysics::octree`].

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
    /// Creates a new body with the given properties and zero initial acceleration.
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

    /// Computes kinetic energy of this body: KE = ½mv².
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
    /// Initializes an N-body system with the given timestep and gravitational softening length, computing initial accelerations.
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

    /// Advances the simulation by one timestep using the velocity Verlet integrator.
    pub fn step(&mut self) {
        step_verlet(&mut self.bodies, self.dt, self.softening);
        self.time += self.dt;
    }

    /// Computes the total mechanical energy (kinetic + potential) of the system.
    pub fn total_energy(&self) -> f64 {
        total_energy(&self.bodies, self.softening)
    }

    /// Computes the mass-weighted center of mass of all bodies.
    pub fn center_of_mass(&self) -> Vec3 {
        center_of_mass(&self.bodies)
    }

    /// Computes the total linear momentum of the system: p = Σ(m_i × v_i).
    pub fn total_momentum(&self) -> Vec3 {
        total_momentum(&self.bodies)
    }
}

/// Computes gravitational acceleration on body `idx` via direct O(N) pairwise summation with Plummer softening.
pub fn compute_acceleration(bodies: &[Body], idx: usize, softening: f64) -> Vec3 {
    assert!(softening > 0.0, "softening length must be positive to avoid division by zero");
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

/// Initializes acceleration vectors for all bodies by computing pairwise gravitational interactions.
pub fn init_accelerations(bodies: &mut [Body], softening: f64) {
    let accels: Vec<Vec3> = (0..bodies.len())
        .map(|i| compute_acceleration(bodies, i, softening))
        .collect();
    for (i, acc) in accels.into_iter().enumerate() {
        bodies[i].acceleration = acc;
    }
}

/// Performs one velocity Verlet integration step (kick-drift-kick) by
/// delegating to the generic symplectic integrator
/// `numerical::ode::symplectic::velocity_verlet` over the flattened
/// phase-space state.
pub fn step_verlet(bodies: &mut [Body], dt: f64, softening: f64) {
    let n = bodies.len();
    let mut x = Vec::with_capacity(3 * n);
    let mut v = Vec::with_capacity(3 * n);
    for b in bodies.iter() {
        x.extend_from_slice(&[b.position.x, b.position.y, b.position.z]);
        v.extend_from_slice(&[b.velocity.x, b.velocity.y, b.velocity.z]);
    }

    let template: Vec<Body> = bodies.to_vec();
    let acc = move |xs: &[f64]| -> Vec<f64> {
        let mut temp = template.clone();
        for (i, b) in temp.iter_mut().enumerate() {
            b.position = Vec3::new(xs[3 * i], xs[3 * i + 1], xs[3 * i + 2]);
        }
        let mut out = Vec::with_capacity(xs.len());
        for i in 0..temp.len() {
            let a = compute_acceleration(&temp, i, softening);
            out.extend_from_slice(&[a.x, a.y, a.z]);
        }
        out
    };
    crate::numerical::ode::symplectic::velocity_verlet(&acc, &mut x, &mut v, dt);

    for (i, b) in bodies.iter_mut().enumerate() {
        b.position = Vec3::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        b.velocity = Vec3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
    }
    // Refresh the cached accelerations at the new positions.
    init_accelerations(bodies, softening);
}

/// Computes the total kinetic energy of all bodies: KE = Σ ½m_i v_i².
pub fn kinetic_energy(bodies: &[Body]) -> f64 {
    bodies.iter().map(|b| b.kinetic_energy()).sum()
}

/// Computes the total gravitational potential energy: PE = -Σ G m_i m_j / r_ij (with Plummer softening).
pub fn potential_energy(bodies: &[Body], softening: f64) -> f64 {
    assert!(softening > 0.0, "softening length must be positive to avoid division by zero");
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

/// Computes the total mechanical energy as the sum of kinetic and potential energy.
pub fn total_energy(bodies: &[Body], softening: f64) -> f64 {
    kinetic_energy(bodies) + potential_energy(bodies, softening)
}

/// Computes the center of mass position: R_cm = Σ(m_i r_i) / Σ(m_i).
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

/// Computes the total linear momentum: p = Σ(m_i v_i).
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

    #[test]
    fn test_compute_acceleration_two_body() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e24, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::ZERO),
        ];
        let acc0 = compute_acceleration(&bodies, 0, DEFAULT_SOFTENING);
        let acc1 = compute_acceleration(&bodies, 1, DEFAULT_SOFTENING);
        // Body 0 should be pulled toward body 1 (positive x)
        assert!(acc0.x > 0.0, "Body 0 should accelerate toward body 1");
        // Body 1 should be pulled toward body 0 (negative x)
        assert!(acc1.x < 0.0, "Body 1 should accelerate toward body 0");
    }

    #[test]
    fn test_compute_acceleration_symmetry() {
        let m = 1.0e30;
        let bodies = vec![
            Body::new(0, m, 1.0, Vec3::new(-1.0e10, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, m, 1.0, Vec3::new(1.0e10, 0.0, 0.0), Vec3::ZERO),
        ];
        let acc0 = compute_acceleration(&bodies, 0, DEFAULT_SOFTENING);
        let acc1 = compute_acceleration(&bodies, 1, DEFAULT_SOFTENING);
        assert!(approx(acc0.x, -acc1.x, 1e-20), "Equal masses should have equal and opposite accelerations");
        assert!(approx(acc0.y, 0.0, 1e-20));
        assert!(approx(acc0.z, 0.0, 1e-20));
    }

    #[test]
    fn test_kinetic_energy_stationary() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::ZERO, Vec3::ZERO),
        ];
        assert!(approx(kinetic_energy(&bodies), 0.0, 1e-20));
    }

    #[test]
    fn test_kinetic_energy_moving() {
        let m = 2.0;
        let v = 3.0;
        let bodies = vec![
            Body::new(0, m, 1.0, Vec3::ZERO, Vec3::new(v, 0.0, 0.0)),
        ];
        let ke = kinetic_energy(&bodies);
        assert!(approx(ke, 0.5 * m * v * v, 1e-12), "KE should be 0.5*m*v^2 = {}, got {ke}", 0.5 * m * v * v);
    }

    #[test]
    fn test_potential_energy_negative() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e30, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::ZERO),
        ];
        let pe = potential_energy(&bodies, DEFAULT_SOFTENING);
        assert!(pe < 0.0, "Gravitational PE should be negative, got {pe}");
    }

    #[test]
    fn test_potential_energy_closer_is_more_negative() {
        let bodies_close = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e30, 1.0, Vec3::new(1.0e10, 0.0, 0.0), Vec3::ZERO),
        ];
        let bodies_far = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e30, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::ZERO),
        ];
        let pe_close = potential_energy(&bodies_close, DEFAULT_SOFTENING);
        let pe_far = potential_energy(&bodies_far, DEFAULT_SOFTENING);
        assert!(pe_close < pe_far, "Closer bodies should have more negative PE");
    }

    #[test]
    fn test_nbody_system_step_advances_time() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e24, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::new(0.0, 3.0e4, 0.0)),
        ];
        let mut system = NBodySystem::new(bodies, DEFAULT_DT, DEFAULT_SOFTENING);
        assert!(approx(system.time, 0.0, 1e-20));
        system.step();
        assert!(approx(system.time, DEFAULT_DT, 1e-20), "Time should advance by dt after one step");
        system.step();
        assert!(approx(system.time, 2.0 * DEFAULT_DT, 1e-20));
    }

    #[test]
    fn test_nbody_system_step_conserves_energy() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e24, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::new(0.0, 3.0e4, 0.0)),
        ];
        let mut system = NBodySystem::new(bodies, DEFAULT_DT, DEFAULT_SOFTENING);
        let e0 = system.total_energy();
        for _ in 0..100 {
            system.step();
        }
        let e1 = system.total_energy();
        let rel_err = ((e1 - e0) / e0).abs();
        assert!(rel_err < 1e-4, "Energy not conserved via NBodySystem::step: relative error = {rel_err}");
    }

    #[test]
    fn test_nbody_system_center_of_mass() {
        let bodies = vec![
            Body::new(0, 2.0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 2.0, 1.0, Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let system = NBodySystem::new(bodies, DEFAULT_DT, DEFAULT_SOFTENING);
        let com = system.center_of_mass();
        assert!(approx(com.x, 2.0, 1e-9));
    }

    #[test]
    fn test_nbody_system_total_momentum() {
        let bodies = vec![
            Body::new(0, 1.0, 1.0, Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)),
            Body::new(1, 2.0, 1.0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(-1.5, 0.0, 0.0)),
        ];
        let system = NBodySystem::new(bodies, DEFAULT_DT, DEFAULT_SOFTENING);
        let p = system.total_momentum();
        assert!(approx(p.x, 0.0, 1e-9));
    }

    #[test]
    fn test_center_of_mass_zero_mass() {
        let bodies = vec![
            Body::new(0, 0.0, 1.0, Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let com = center_of_mass(&bodies);
        assert!(approx(com.x, 0.0, 1e-12));
    }
}
