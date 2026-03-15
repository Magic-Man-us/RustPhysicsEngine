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
    damping_coeff / (2.0 * (mass * spring_constant).sqrt())
}

/// Critical damping coefficient: c_crit = 2√(mk)
pub fn critical_damping(mass: f64, spring_constant: f64) -> f64 {
    2.0 * (mass * spring_constant).sqrt()
}

/// Logarithmic decrement: δ = 2πζ / √(1 - ζ²)
pub fn logarithmic_decrement(damping_ratio: f64) -> f64 {
    2.0 * std::f64::consts::PI * damping_ratio / (1.0 - damping_ratio * damping_ratio).sqrt()
}

/// Quality factor: Q = 1 / (2ζ)
pub fn quality_factor(damping_ratio: f64) -> f64 {
    1.0 / (2.0 * damping_ratio)
}

/// Decay time constant: τ = 1/γ (time for amplitude to drop to 1/e)
pub fn decay_time(damping_coeff: f64) -> f64 {
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
        let p_before = momentum(2.0, 3.0) + momentum(1.0, -1.0);
        let vf = inelastic_collision_1d(2.0, 3.0, 1.0, -1.0);
        let p_after = momentum(3.0, vf);
        assert!(approx(p_before, p_after));
    }

    // ── Damped Oscillation Tests ──

    #[test]
    fn test_damped_frequency_underdamped() {
        // ω₀ = 10, ζ = 0.1 → ωd = 10√(1-0.01) = 10√0.99
        let wd = damped_frequency(10.0, 0.1);
        assert!(approx(wd, 10.0 * (0.99_f64).sqrt()));
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
        assert!(approx(a, 5.0 * (-1.0_f64).exp()));
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
        let expected = 2.0 * std::f64::consts::PI * 0.1 / (0.99_f64).sqrt();
        assert!(approx(d, expected));
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
        assert!(approx(a, 10.0 / (2.0 * 0.5 * 5.0)));
    }

    #[test]
    fn test_driven_amplitude_off_resonance() {
        // f₀=1, ω=1, ω₀=3, γ=0.5 → denom = √((9-1)² + (2*0.5*1)²) = √(64+1) = √65
        let a = driven_amplitude(1.0, 1.0, 3.0, 0.5);
        assert!(approx(a, 1.0 / 65.0_f64.sqrt()));
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
        assert!(approx(wr, 98.0_f64.sqrt()));
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
        assert!(approx(w2, 10.0_f64.sqrt()));
    }

    #[test]
    fn test_beat_frequency_coupled() {
        assert!(approx(beat_frequency_coupled(5.0, 3.0), 2.0));
        assert!(approx(beat_frequency_coupled(3.0, 5.0), 2.0));
    }
}
