use crate::math::Vec3;

// ── Kinematics ──

/// Position after uniform acceleration: x = x0 + v0*t + 0.5*a*t^2
pub fn displacement(initial_velocity: f64, acceleration: f64, time: f64) -> f64 {
    initial_velocity * time + 0.5 * acceleration * time * time
}

/// Velocity after uniform acceleration: v = v0 + a*t
pub fn velocity(initial_velocity: f64, acceleration: f64, time: f64) -> f64 {
    initial_velocity + acceleration * time
}

/// Velocity squared: v^2 = v0^2 + 2*a*d
pub fn velocity_squared(initial_velocity: f64, acceleration: f64, displacement: f64) -> f64 {
    initial_velocity * initial_velocity + 2.0 * acceleration * displacement
}

/// 3D position under constant acceleration.
pub fn position_3d(pos: Vec3, vel: Vec3, acc: Vec3, t: f64) -> Vec3 {
    pos + vel * t + acc * (0.5 * t * t)
}

/// 3D velocity under constant acceleration.
pub fn velocity_3d(vel: Vec3, acc: Vec3, t: f64) -> Vec3 {
    vel + acc * t
}

// ── Projectile Motion ──

/// Range of a projectile on flat ground: R = v^2 * sin(2θ) / g
pub fn projectile_range(speed: f64, angle_rad: f64, g: f64) -> f64 {
    speed * speed * (2.0 * angle_rad).sin() / g
}

/// Maximum height of a projectile: H = v^2 * sin^2(θ) / (2g)
pub fn projectile_max_height(speed: f64, angle_rad: f64, g: f64) -> f64 {
    let sin_a = angle_rad.sin();
    speed * speed * sin_a * sin_a / (2.0 * g)
}

/// Time of flight for a projectile on flat ground: T = 2v*sin(θ) / g
pub fn projectile_time_of_flight(speed: f64, angle_rad: f64, g: f64) -> f64 {
    2.0 * speed * angle_rad.sin() / g
}

// ── Newton's Laws ──

/// Force = mass * acceleration (Newton's second law)
pub fn force(mass: f64, acceleration: f64) -> f64 {
    mass * acceleration
}

/// F = ma as vectors
pub fn force_3d(mass: f64, acceleration: Vec3) -> Vec3 {
    acceleration * mass
}

/// Acceleration from force: a = F / m
pub fn acceleration(force: f64, mass: f64) -> f64 {
    force / mass
}

/// Weight: W = m * g
pub fn weight(mass: f64, g: f64) -> f64 {
    mass * g
}

// ── Momentum ──

/// Linear momentum: p = m * v
pub fn momentum(mass: f64, velocity: f64) -> f64 {
    mass * velocity
}

/// 3D momentum.
pub fn momentum_3d(mass: f64, velocity: Vec3) -> Vec3 {
    velocity * mass
}

/// Impulse: J = F * Δt
pub fn impulse(force: f64, delta_t: f64) -> f64 {
    force * delta_t
}

// ── Collisions ──

/// Final velocities after a 1D elastic collision between two masses.
/// Returns (v1_final, v2_final).
pub fn elastic_collision_1d(m1: f64, v1: f64, m2: f64, v2: f64) -> (f64, f64) {
    let total = m1 + m2;
    let v1f = ((m1 - m2) * v1 + 2.0 * m2 * v2) / total;
    let v2f = ((m2 - m1) * v2 + 2.0 * m1 * v1) / total;
    (v1f, v2f)
}

/// Final velocity after a perfectly inelastic collision (objects stick together).
pub fn inelastic_collision_1d(m1: f64, v1: f64, m2: f64, v2: f64) -> f64 {
    (m1 * v1 + m2 * v2) / (m1 + m2)
}

/// Coefficient of restitution: e = -(v1f - v2f) / (v1i - v2i)
pub fn coefficient_of_restitution(v1i: f64, v2i: f64, v1f: f64, v2f: f64) -> f64 {
    -(v1f - v2f) / (v1i - v2i)
}

// ── Energy ──

/// Kinetic energy: KE = 0.5 * m * v^2
pub fn kinetic_energy(mass: f64, speed: f64) -> f64 {
    0.5 * mass * speed * speed
}

/// Gravitational potential energy: PE = m * g * h
pub fn potential_energy_gravity(mass: f64, g: f64, height: f64) -> f64 {
    mass * g * height
}

/// Elastic potential energy: PE = 0.5 * k * x^2
pub fn potential_energy_spring(spring_constant: f64, displacement: f64) -> f64 {
    0.5 * spring_constant * displacement * displacement
}

/// Work: W = F * d * cos(θ)
pub fn work(force: f64, displacement: f64, angle_rad: f64) -> f64 {
    force * displacement * angle_rad.cos()
}

/// Power: P = W / t
pub fn power(work: f64, time: f64) -> f64 {
    work / time
}

/// Power (instantaneous): P = F * v
pub fn power_instantaneous(force: f64, velocity: f64) -> f64 {
    force * velocity
}

// ── Rotation ──

/// Angular velocity: ω = Δθ / Δt
pub fn angular_velocity(delta_theta: f64, delta_t: f64) -> f64 {
    delta_theta / delta_t
}

/// Angular acceleration: α = Δω / Δt
pub fn angular_acceleration(delta_omega: f64, delta_t: f64) -> f64 {
    delta_omega / delta_t
}

/// Torque: τ = r * F * sin(θ)
pub fn torque(radius: f64, force: f64, angle_rad: f64) -> f64 {
    radius * force * angle_rad.sin()
}

/// Torque as cross product: τ = r × F
pub fn torque_3d(r: Vec3, f: Vec3) -> Vec3 {
    r.cross(&f)
}

/// Moment of inertia of a point mass: I = m * r^2
pub fn moment_of_inertia_point(mass: f64, radius: f64) -> f64 {
    mass * radius * radius
}

/// Moment of inertia of a solid sphere: I = (2/5) * m * r^2
pub fn moment_of_inertia_solid_sphere(mass: f64, radius: f64) -> f64 {
    0.4 * mass * radius * radius
}

/// Moment of inertia of a solid cylinder about its axis: I = (1/2) * m * r^2
pub fn moment_of_inertia_solid_cylinder(mass: f64, radius: f64) -> f64 {
    0.5 * mass * radius * radius
}

/// Rotational kinetic energy: KE = 0.5 * I * ω^2
pub fn rotational_kinetic_energy(moment_of_inertia: f64, angular_velocity: f64) -> f64 {
    0.5 * moment_of_inertia * angular_velocity * angular_velocity
}

/// Angular momentum: L = I * ω
pub fn angular_momentum(moment_of_inertia: f64, angular_velocity: f64) -> f64 {
    moment_of_inertia * angular_velocity
}

/// Centripetal acceleration: a = v^2 / r
pub fn centripetal_acceleration(speed: f64, radius: f64) -> f64 {
    speed * speed / radius
}

/// Centripetal force: F = m * v^2 / r
pub fn centripetal_force(mass: f64, speed: f64, radius: f64) -> f64 {
    mass * speed * speed / radius
}

// ── Friction ──

/// Friction force: f = μ * N
pub fn friction_force(coefficient: f64, normal_force: f64) -> f64 {
    coefficient * normal_force
}

// ── Simple Harmonic Motion ──

/// Period of a mass-spring system: T = 2π * sqrt(m / k)
pub fn shm_period_spring(mass: f64, spring_constant: f64) -> f64 {
    2.0 * std::f64::consts::PI * (mass / spring_constant).sqrt()
}

/// Period of a simple pendulum: T = 2π * sqrt(L / g)
pub fn shm_period_pendulum(length: f64, g: f64) -> f64 {
    2.0 * std::f64::consts::PI * (length / g).sqrt()
}

/// Position of SHM: x(t) = A * cos(ωt + φ)
pub fn shm_position(amplitude: f64, angular_freq: f64, time: f64, phase: f64) -> f64 {
    amplitude * (angular_freq * time + phase).cos()
}

/// Velocity of SHM: v(t) = -A * ω * sin(ωt + φ)
pub fn shm_velocity(amplitude: f64, angular_freq: f64, time: f64, phase: f64) -> f64 {
    -amplitude * angular_freq * (angular_freq * time + phase).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_displacement() {
        // Object starting at rest, a=2, t=3 → d = 0 + 0.5*2*9 = 9
        assert!(approx(displacement(0.0, 2.0, 3.0), 9.0));
    }

    #[test]
    fn test_velocity() {
        assert!(approx(velocity(10.0, -9.8, 1.0), 0.2));
    }

    #[test]
    fn test_force() {
        assert!(approx(force(5.0, 3.0), 15.0));
    }

    #[test]
    fn test_elastic_collision() {
        let (v1f, v2f) = elastic_collision_1d(1.0, 2.0, 1.0, 0.0);
        assert!(approx(v1f, 0.0));
        assert!(approx(v2f, 2.0));
    }

    #[test]
    fn test_inelastic_collision() {
        let vf = inelastic_collision_1d(2.0, 3.0, 1.0, 0.0);
        assert!(approx(vf, 2.0));
    }

    #[test]
    fn test_kinetic_energy() {
        assert!(approx(kinetic_energy(2.0, 3.0), 9.0));
    }

    #[test]
    fn test_work() {
        assert!(approx(work(10.0, 5.0, 0.0), 50.0));
    }

    #[test]
    fn test_centripetal_force() {
        assert!(approx(centripetal_force(2.0, 3.0, 1.5), 12.0));
    }

    #[test]
    fn test_projectile_range() {
        let r = projectile_range(20.0, std::f64::consts::PI / 4.0, 9.8);
        assert!(approx(r, 40.816326530612244));
    }

    #[test]
    fn test_shm_period_pendulum() {
        let t = shm_period_pendulum(1.0, 9.8);
        assert!((t - 2.006).abs() < 0.01);
    }

    #[test]
    fn test_torque() {
        assert!(approx(torque(2.0, 5.0, std::f64::consts::PI / 2.0), 10.0));
    }

    #[test]
    fn test_momentum_conservation() {
        let p_before = momentum(2.0, 3.0) + momentum(1.0, -1.0);
        let vf = inelastic_collision_1d(2.0, 3.0, 1.0, -1.0);
        let p_after = momentum(3.0, vf);
        assert!(approx(p_before, p_after));
    }
}
