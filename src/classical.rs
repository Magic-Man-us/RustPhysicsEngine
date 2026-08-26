//! Newtonian mechanics: kinematics, dynamics, and the harmonic
//! oscillator.
//!
//! Linear and rotational motion under constant acceleration, forces and
//! momentum, work, energy and power, collisions in one dimension from
//! perfectly elastic to perfectly inelastic, moments of inertia for the
//! standard bodies, and circular motion.
//!
//! The oscillator section runs from the undamped period through the
//! damped response -- damping ratio, logarithmic decrement, quality factor
//! -- to the driven steady state and its resonance, and ends with the
//! normal frequencies of two coupled oscillators. For the same problem
//! solved numerically, or with more than two masses, see [`crate::resonance`].

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
    assert!(g > 0.0, "gravitational acceleration must be positive");
    speed * speed * (2.0 * angle_rad).sin() / g
}

/// Maximum height of a projectile: H = v^2 * sin^2(θ) / (2g)
pub fn projectile_max_height(speed: f64, angle_rad: f64, g: f64) -> f64 {
    assert!(g > 0.0, "gravitational acceleration must be positive");
    let sin_a = angle_rad.sin();
    speed * speed * sin_a * sin_a / (2.0 * g)
}

/// Time of flight for a projectile on flat ground: T = 2v*sin(θ) / g
pub fn projectile_time_of_flight(speed: f64, angle_rad: f64, g: f64) -> f64 {
    assert!(g > 0.0, "gravitational acceleration must be positive");
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
    assert!(mass > 0.0, "mass must be positive");
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
    assert!(m1 > 0.0, "m1 must be positive");
    assert!(m2 > 0.0, "m2 must be positive");
    let total = m1 + m2;
    let v1f = ((m1 - m2) * v1 + 2.0 * m2 * v2) / total;
    let v2f = ((m2 - m1) * v2 + 2.0 * m1 * v1) / total;
    (v1f, v2f)
}

/// Final velocity after a perfectly inelastic collision (objects stick together).
pub fn inelastic_collision_1d(m1: f64, v1: f64, m2: f64, v2: f64) -> f64 {
    assert!(m1 > 0.0, "m1 must be positive");
    assert!(m2 > 0.0, "m2 must be positive");
    (m1 * v1 + m2 * v2) / (m1 + m2)
}

/// Coefficient of restitution: e = -(v1f - v2f) / (v1i - v2i)
pub fn coefficient_of_restitution(v1i: f64, v2i: f64, v1f: f64, v2f: f64) -> f64 {
    assert!(v1i != v2i, "initial velocities must differ");
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
    assert!(time > 0.0, "time must be positive");
    work / time
}

/// Power (instantaneous): P = F * v
pub fn power_instantaneous(force: f64, velocity: f64) -> f64 {
    force * velocity
}

// ── Rotation ──

/// Angular velocity: ω = Δθ / Δt
pub fn angular_velocity(delta_theta: f64, delta_t: f64) -> f64 {
    assert!(delta_t != 0.0, "time interval must not be zero");
    delta_theta / delta_t
}

/// Angular acceleration: α = Δω / Δt
pub fn angular_acceleration(delta_omega: f64, delta_t: f64) -> f64 {
    assert!(delta_t != 0.0, "time interval must not be zero");
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
    assert!(radius > 0.0, "radius must be positive");
    speed * speed / radius
}

/// Centripetal force: F = m * v^2 / r
pub fn centripetal_force(mass: f64, speed: f64, radius: f64) -> f64 {
    assert!(radius > 0.0, "radius must be positive");
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
    assert!(spring_constant > 0.0, "spring constant must be positive");
    2.0 * std::f64::consts::PI * (mass / spring_constant).sqrt()
}

/// Period of a simple pendulum: T = 2π * sqrt(L / g)
pub fn shm_period_pendulum(length: f64, g: f64) -> f64 {
    assert!(g > 0.0, "gravitational acceleration must be positive");
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

// ── Damped Oscillations ──

/// Damped angular frequency: ωd = ω₀√(1 - ζ²), returns 0 if overdamped (ζ ≥ 1)
pub fn damped_frequency(natural_freq: f64, damping_ratio: f64) -> f64 {
    if damping_ratio >= 1.0 {
        return 0.0;
    }
    natural_freq * (1.0 - damping_ratio * damping_ratio).sqrt()
}

/// Damped amplitude: A(t) = A₀ × e^(-γt)
pub fn damped_amplitude(initial_amplitude: f64, damping_coeff: f64, time: f64) -> f64 {
    initial_amplitude * (-damping_coeff * time).exp()
}

/// Damped oscillation position: x(t) = A₀e^(-γt)cos(ωdt + φ)
pub fn damped_position(
    amplitude: f64,
    damping_coeff: f64,
    angular_freq: f64,
    time: f64,
    phase: f64,
) -> f64 {
    amplitude * (-damping_coeff * time).exp() * (angular_freq * time + phase).cos()
}

/// Damping ratio: ζ = c / (2√(mk))
pub fn damping_ratio(damping_coeff: f64, mass: f64, spring_constant: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    assert!(spring_constant > 0.0, "spring constant must be positive");
    damping_coeff / (2.0 * (mass * spring_constant).sqrt())
}

/// Critical damping coefficient: c_crit = 2√(mk)
pub fn critical_damping(mass: f64, spring_constant: f64) -> f64 {
    2.0 * (mass * spring_constant).sqrt()
}

/// Logarithmic decrement: δ = 2πζ / √(1 - ζ²)
pub fn logarithmic_decrement(damping_ratio: f64) -> f64 {
    assert!((0.0..1.0).contains(&damping_ratio), "damping ratio must be in [0, 1)");
    2.0 * std::f64::consts::PI * damping_ratio / (1.0 - damping_ratio * damping_ratio).sqrt()
}

/// Quality factor: Q = 1 / (2ζ)
pub fn quality_factor(damping_ratio: f64) -> f64 {
    assert!(damping_ratio > 0.0, "damping ratio must be positive");
    1.0 / (2.0 * damping_ratio)
}

/// Decay time constant: τ = 1/γ (time for amplitude to drop to 1/e)
pub fn decay_time(damping_coeff: f64) -> f64 {
    assert!(damping_coeff > 0.0, "damping coefficient must be positive");
    1.0 / damping_coeff
}

// ── Driven (Forced) Oscillations ──

/// Driven oscillation amplitude: A = f₀ / √((ω₀²-ω²)² + (2γω)²)
/// where f₀ = F₀/m (driving force per unit mass)
pub fn driven_amplitude(f0: f64, omega: f64, omega0: f64, gamma: f64) -> f64 {
    let delta = omega0 * omega0 - omega * omega;
    let denom = (delta * delta + (2.0 * gamma * omega).powi(2)).sqrt();
    f0 / denom
}

/// Phase lag of driven oscillation: φ = atan2(2γω, ω₀²-ω²)
pub fn driven_phase(omega: f64, omega0: f64, gamma: f64) -> f64 {
    (2.0 * gamma * omega).atan2(omega0 * omega0 - omega * omega)
}

/// Resonance frequency: ωr = √(ω₀² - 2γ²), returns 0 if overdamped
pub fn resonance_frequency(natural_freq: f64, damping_coeff: f64) -> f64 {
    let sq = natural_freq * natural_freq - 2.0 * damping_coeff * damping_coeff;
    if sq <= 0.0 {
        return 0.0;
    }
    sq.sqrt()
}

/// Peak amplitude at resonance: A_max = f₀ / (2γ√(ω₀² - γ²))
pub fn resonance_amplitude(f0: f64, omega0: f64, gamma: f64) -> f64 {
    assert!(gamma > 0.0, "damping coefficient must be positive");
    let denom_sq = omega0 * omega0 - gamma * gamma;
    if denom_sq <= 0.0 {
        return f64::INFINITY;
    }
    f0 / (2.0 * gamma * denom_sq.sqrt())
}

// ── Coupled Oscillators ──

/// Normal-mode frequencies of two identical masses coupled by a spring:
/// ω₁ = √(k/m), ω₂ = √((k + 2k_c)/m)
pub fn coupled_normal_frequencies(k: f64, k_coupling: f64, m: f64) -> (f64, f64) {
    assert!(m > 0.0, "mass must be positive");
    let omega1 = (k / m).sqrt();
    let omega2 = ((k + 2.0 * k_coupling) / m).sqrt();
    (omega1, omega2)
}

/// Beat frequency of coupled oscillators: f_beat = |f1 - f2|
pub fn beat_frequency_coupled(freq1: f64, freq2: f64) -> f64 {
    (freq1 - freq2).abs()
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
        // p_before = 2*3 + 1*(-1) = 5.0
        // vf = (6 - 1) / 3 = 5/3, p_after = 3 * 5/3 = 5.0
        let vf = inelastic_collision_1d(2.0, 3.0, 1.0, -1.0);
        assert!(approx(momentum(2.0, 3.0) + momentum(1.0, -1.0), 5.0));
        assert!(approx(momentum(3.0, vf), 5.0));
    }

    // ── Damped Oscillation Tests ──

    #[test]
    fn test_damped_frequency_underdamped() {
        // ω₀ = 10, ζ = 0.1 → ωd = 10√(1-0.01) = 10√0.99
        let wd = damped_frequency(10.0, 0.1);
        assert!(approx(wd, 9.949874));
    }

    #[test]
    fn test_damped_frequency_overdamped() {
        assert!(approx(damped_frequency(10.0, 1.0), 0.0));
        assert!(approx(damped_frequency(10.0, 2.0), 0.0));
    }

    #[test]
    fn test_damped_amplitude() {
        // A₀ = 5.0, γ = 0.5, t = 2 → 5e^(-1)
        let a = damped_amplitude(5.0, 0.5, 2.0);
        assert!(approx(a, 1.839397));
    }

    #[test]
    fn test_damped_position() {
        // At t=0, phase=0: x = A₀ * 1 * cos(0) = A₀
        assert!(approx(damped_position(3.0, 0.5, 10.0, 0.0, 0.0), 3.0));
    }

    #[test]
    fn test_damping_ratio() {
        // c = 4, m = 1, k = 16 → ζ = 4 / (2√16) = 4/8 = 0.5
        assert!(approx(damping_ratio(4.0, 1.0, 16.0), 0.5));
    }

    #[test]
    fn test_critical_damping() {
        // m = 1, k = 100 → c_crit = 2√100 = 20
        assert!(approx(critical_damping(1.0, 100.0), 20.0));
    }

    #[test]
    fn test_logarithmic_decrement() {
        // ζ = 0.1 → δ = 2π(0.1)/√(1-0.01)
        let d = logarithmic_decrement(0.1);
        assert!(approx(d, 0.631484));
    }

    #[test]
    fn test_quality_factor() {
        // ζ = 0.05 → Q = 1/(2*0.05) = 10
        assert!(approx(quality_factor(0.05), 10.0));
    }

    #[test]
    fn test_decay_time() {
        // γ = 0.25 → τ = 4.0
        assert!(approx(decay_time(0.25), 4.0));
    }

    // ── Driven Oscillation Tests ──

    #[test]
    fn test_driven_amplitude_at_resonance() {
        // At ω = ω₀: A = f₀ / (2γω₀)
        let a = driven_amplitude(10.0, 5.0, 5.0, 0.5);
        assert!(approx(a, 2.0));
    }

    #[test]
    fn test_driven_amplitude_off_resonance() {
        // f₀=1, ω=1, ω₀=3, γ=0.5 → denom = √((9-1)² + (2*0.5*1)²) = √(64+1) = √65
        let a = driven_amplitude(1.0, 1.0, 3.0, 0.5);
        assert!(approx(a, 0.124035));
    }

    #[test]
    fn test_driven_phase_at_resonance() {
        // At ω = ω₀: φ = atan2(2γω, 0) = π/2
        let p = driven_phase(5.0, 5.0, 0.5);
        assert!(approx(p, std::f64::consts::PI / 2.0));
    }

    #[test]
    fn test_driven_phase_low_freq() {
        // ω → 0: φ = atan2(0, ω₀²) = 0
        let p = driven_phase(0.0, 5.0, 0.5);
        assert!(approx(p, 0.0));
    }

    #[test]
    fn test_resonance_frequency() {
        // ω₀ = 10, γ = 1 → ωr = √(100 - 2) = √98
        let wr = resonance_frequency(10.0, 1.0);
        assert!(approx(wr, 9.899495));
    }

    #[test]
    fn test_resonance_frequency_overdamped() {
        // ω₀ = 1, γ = 1 → ω₀² - 2γ² = 1-2 = -1 → returns 0
        assert!(approx(resonance_frequency(1.0, 1.0), 0.0));
    }

    #[test]
    fn test_resonance_amplitude() {
        // f₀ = 10, ω₀ = 5, γ = 0.5 → 10/(2×0.5×√(25-0.25)) ≈ 2.010076
        assert!(approx(resonance_amplitude(10.0, 5.0, 0.5), 2.010076));
    }

    // ── Coupled Oscillator Tests ──

    #[test]
    fn test_coupled_normal_frequencies() {
        // k=4, k_c=3, m=1 → ω₁ = 2, ω₂ = √(4+6) = √10
        let (w1, w2) = coupled_normal_frequencies(4.0, 3.0, 1.0);
        assert!(approx(w1, 2.0));
        assert!(approx(w2, 3.162278));
    }

    #[test]
    fn test_beat_frequency_coupled() {
        assert!(approx(beat_frequency_coupled(5.0, 3.0), 2.0));
        assert!(approx(beat_frequency_coupled(3.0, 5.0), 2.0));
    }

    // ── Kinematics ──

    #[test]
    fn test_velocity_squared() {
        // v0=3, a=2, d=4 → v^2 = 9 + 16 = 25
        assert!(approx(velocity_squared(3.0, 2.0, 4.0), 25.0));
    }

    #[test]
    fn test_position_3d() {
        let pos = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
        let vel = Vec3 { x: 2.0, y: 0.0, z: 0.0 };
        let acc = Vec3 { x: 0.0, y: -10.0, z: 0.0 };
        let result = position_3d(pos, vel, acc, 1.0);
        assert!(approx(result.x, 3.0));
        assert!(approx(result.y, -5.0));
        assert!(approx(result.z, 0.0));
    }

    #[test]
    fn test_velocity_3d() {
        let vel = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let acc = Vec3 { x: 0.0, y: -9.8, z: 0.0 };
        let result = velocity_3d(vel, acc, 2.0);
        assert!(approx(result.x, 1.0));
        assert!(approx(result.y, -17.6));
        assert!(approx(result.z, 3.0));
    }

    // ── Projectile Motion ──

    #[test]
    fn test_projectile_max_height() {
        // v=20, θ=π/4, g=9.8 → H = 400 * 0.5 / (2*9.8) = 200/19.6 ≈ 10.204
        let h = projectile_max_height(20.0, std::f64::consts::PI / 4.0, 9.8);
        assert!(approx(h, 10.204081632653061));
    }

    #[test]
    fn test_projectile_time_of_flight() {
        // v=20, θ=π/6, g=9.8 → T = 2*20*sin(π/6)/9.8 = 2*20*0.5/9.8 ≈ 2.0408
        let t = projectile_time_of_flight(20.0, std::f64::consts::PI / 6.0, 9.8);
        assert!(approx(t, 2.040816));
    }

    // ── Newton's Laws ──

    #[test]
    fn test_force_3d() {
        let acc = Vec3 { x: 1.0, y: -2.0, z: 3.0 };
        let result = force_3d(5.0, acc);
        assert!(approx(result.x, 5.0));
        assert!(approx(result.y, -10.0));
        assert!(approx(result.z, 15.0));
    }

    #[test]
    fn test_acceleration() {
        // F=20, m=4 → a=5
        assert!(approx(acceleration(20.0, 4.0), 5.0));
    }

    #[test]
    fn test_weight() {
        // m=10, g=9.8 → W=98
        assert!(approx(weight(10.0, 9.8), 98.0));
    }

    // ── Momentum ──

    #[test]
    fn test_momentum_3d() {
        let vel = Vec3 { x: 3.0, y: -1.0, z: 2.0 };
        let result = momentum_3d(4.0, vel);
        assert!(approx(result.x, 12.0));
        assert!(approx(result.y, -4.0));
        assert!(approx(result.z, 8.0));
    }

    #[test]
    fn test_impulse() {
        // F=100, Δt=0.05 → J=5
        assert!(approx(impulse(100.0, 0.05), 5.0));
    }

    // ── Collisions ──

    #[test]
    fn test_coefficient_of_restitution() {
        // Perfectly elastic: e = -(0 - 2) / (2 - 0) = 1
        assert!(approx(coefficient_of_restitution(2.0, 0.0, 0.0, 2.0), 1.0));
        // Perfectly inelastic: both move at same speed → e = 0
        assert!(approx(coefficient_of_restitution(3.0, 0.0, 1.0, 1.0), 0.0));
    }

    // ── Energy ──

    #[test]
    fn test_potential_energy_gravity() {
        // m=5, g=9.8, h=10 → PE=490
        assert!(approx(potential_energy_gravity(5.0, 9.8, 10.0), 490.0));
    }

    #[test]
    fn test_potential_energy_spring() {
        // k=200, x=0.1 → PE=0.5*200*0.01=1.0
        assert!(approx(potential_energy_spring(200.0, 0.1), 1.0));
    }

    #[test]
    fn test_power() {
        // W=500, t=10 → P=50
        assert!(approx(power(500.0, 10.0), 50.0));
    }

    #[test]
    fn test_power_instantaneous() {
        // F=20, v=3 → P=60
        assert!(approx(power_instantaneous(20.0, 3.0), 60.0));
    }

    // ── Rotation ──

    #[test]
    fn test_angular_velocity() {
        // Δθ=2π, Δt=1 → ω=2π
        assert!(approx(angular_velocity(2.0 * std::f64::consts::PI, 1.0), std::f64::consts::TAU));
    }

    #[test]
    fn test_angular_acceleration() {
        // Δω=10, Δt=2 → α=5
        assert!(approx(angular_acceleration(10.0, 2.0), 5.0));
    }

    #[test]
    fn test_torque_3d() {
        let r = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
        let f = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
        let result = torque_3d(r, f);
        // r × f = (0, 0, 1)
        assert!(approx(result.x, 0.0));
        assert!(approx(result.y, 0.0));
        assert!(approx(result.z, 1.0));
    }

    #[test]
    fn test_moment_of_inertia_point() {
        // m=2, r=3 → I=18
        assert!(approx(moment_of_inertia_point(2.0, 3.0), 18.0));
    }

    #[test]
    fn test_moment_of_inertia_solid_sphere() {
        // I = (2/5)*m*r^2 = 0.4*10*4 = 16
        assert!(approx(moment_of_inertia_solid_sphere(10.0, 2.0), 16.0));
    }

    #[test]
    fn test_moment_of_inertia_solid_cylinder() {
        // I = 0.5*m*r^2 = 0.5*10*4 = 20
        assert!(approx(moment_of_inertia_solid_cylinder(10.0, 2.0), 20.0));
    }

    #[test]
    fn test_rotational_kinetic_energy() {
        // KE = 0.5*I*ω^2 = 0.5*4*9 = 18
        assert!(approx(rotational_kinetic_energy(4.0, 3.0), 18.0));
    }

    #[test]
    fn test_angular_momentum() {
        // L = I*ω = 5*3 = 15
        assert!(approx(angular_momentum(5.0, 3.0), 15.0));
    }

    #[test]
    fn test_centripetal_acceleration() {
        // a = v^2/r = 9/3 = 3
        assert!(approx(centripetal_acceleration(3.0, 3.0), 3.0));
    }

    // ── Friction ──

    #[test]
    fn test_friction_force() {
        // μ=0.3, N=100 → f=30
        assert!(approx(friction_force(0.3, 100.0), 30.0));
    }

    // ── SHM ──

    #[test]
    fn test_shm_period_spring() {
        // T = 2π√(m/k) = 2π√(1/4) = 2π*0.5 = π
        assert!(approx(shm_period_spring(1.0, 4.0), std::f64::consts::PI));
    }

    #[test]
    fn test_shm_position() {
        // x(0) = A*cos(φ) with φ=0 → x = A
        assert!(approx(shm_position(5.0, 2.0, 0.0, 0.0), 5.0));
        // x(t) = 5*cos(2*π/4 + 0) = 5*cos(π/2) = 0
        assert!(approx(shm_position(5.0, 2.0, std::f64::consts::PI / 4.0, 0.0), 0.0));
    }

    #[test]
    fn test_shm_velocity() {
        // v(0) = -A*ω*sin(φ) with φ=0 → v = 0
        assert!(approx(shm_velocity(5.0, 2.0, 0.0, 0.0), 0.0));
        // v at t=π/(2ω) with φ=0: v = -A*ω*sin(π/2) = -A*ω
        assert!(approx(
            shm_velocity(5.0, 2.0, std::f64::consts::PI / 4.0, 0.0),
            -10.0
        ));
    }

    #[test]
    fn test_resonance_amplitude_overdamped() {
        let a = resonance_amplitude(1.0, 1.0, 2.0);
        assert!(a.is_infinite());
    }
}
