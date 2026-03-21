use crate::math::Vec3;

const DEFAULT_GRAVITY_Y: f64 = -9.81;
const MIN_DISTANCE: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec3,
    pub previous_position: Vec3,
    pub acceleration: Vec3,
    pub mass: f64,
    pub pinned: bool,
}

impl Particle {
    /// Create a free particle at the given position with the given mass.
    pub fn new(position: Vec3, mass: f64) -> Self {
        assert!(mass > 0.0, "particle mass must be positive");
        Self {
            position,
            previous_position: position,
            acceleration: Vec3::ZERO,
            mass,
            pinned: false,
        }
    }

    /// Create a pinned (immovable) particle at the given position.
    pub fn new_pinned(position: Vec3) -> Self {
        Self {
            position,
            previous_position: position,
            acceleration: Vec3::ZERO,
            mass: 1.0,
            pinned: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Spring {
    pub particle_a: usize,
    pub particle_b: usize,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
}

impl Spring {
    /// Create a spring connecting particles a and b with given rest length, stiffness, and damping.
    pub fn new(a: usize, b: usize, rest_length: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            particle_a: a,
            particle_b: b,
            rest_length,
            stiffness,
            damping,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MassSpringSystem {
    pub particles: Vec<Particle>,
    pub springs: Vec<Spring>,
    pub gravity: Vec3,
    pub time: f64,
}

impl MassSpringSystem {
    /// Create an empty mass-spring system with the given gravity vector.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            gravity,
            time: 0.0,
        }
    }

    /// Add a particle to the system, returning its index.
    pub fn add_particle(&mut self, p: Particle) -> usize {
        let idx = self.particles.len();
        self.particles.push(p);
        idx
    }

    /// Add a spring constraint to the system.
    pub fn add_spring(&mut self, s: Spring) {
        self.springs.push(s);
    }

    /// Verlet integration with Hooke's law spring forces and viscous damping.
    ///
    /// F_spring = -k (|dx| - L0) * dx_hat
    /// F_damp   = -d (v_rel . dx_hat) * dx_hat
    /// F_grav   = m * g
    ///
    /// Verlet update: new_pos = 2*pos - prev_pos + a*dt^2
    pub fn step_verlet(&mut self, dt: f64) {
        let n = self.particles.len();

        // Reset accelerations to zero
        for p in &mut self.particles {
            p.acceleration = Vec3::ZERO;
        }

        // Accumulate spring forces (Hooke's law: F = -k(|dx| - L0) * dx_hat)
        for spring in &self.springs {
            let pos_a = self.particles[spring.particle_a].position;
            let pos_b = self.particles[spring.particle_b].position;
            let delta = pos_b - pos_a;
            let dist = delta.magnitude();

            if dist < MIN_DISTANCE {
                continue;
            }

            let dir = delta * (1.0 / dist);
            let stretch = dist - spring.rest_length;

            // Hooke's law: force on A toward B when stretched
            let spring_force = dir * (spring.stiffness * stretch);

            // Damping: estimate velocity from Verlet positions
            if dt > 0.0 {
                let vel_a = (pos_a - self.particles[spring.particle_a].previous_position) * (1.0 / dt);
                let vel_b = (pos_b - self.particles[spring.particle_b].previous_position) * (1.0 / dt);
                let v_rel = vel_b - vel_a;
                let damp_scalar = spring.damping * v_rel.dot(&dir);
                let damp_force = dir * damp_scalar;

                let total_force = spring_force + damp_force;
                let mass_a = self.particles[spring.particle_a].mass;
                let mass_b = self.particles[spring.particle_b].mass;
                self.particles[spring.particle_a].acceleration =
                    self.particles[spring.particle_a].acceleration + total_force * (1.0 / mass_a);
                self.particles[spring.particle_b].acceleration =
                    self.particles[spring.particle_b].acceleration - total_force * (1.0 / mass_b);
            } else {
                let mass_a = self.particles[spring.particle_a].mass;
                let mass_b = self.particles[spring.particle_b].mass;
                self.particles[spring.particle_a].acceleration =
                    self.particles[spring.particle_a].acceleration + spring_force * (1.0 / mass_a);
                self.particles[spring.particle_b].acceleration =
                    self.particles[spring.particle_b].acceleration - spring_force * (1.0 / mass_b);
            }
        }

        // Add gravity: a += g (since F = m*g and a = F/m = g)
        for p in &mut self.particles {
            if !p.pinned {
                p.acceleration = p.acceleration + self.gravity;
            }
        }

        // Verlet integration
        let dt_sq = dt * dt;
        for i in 0..n {
            if self.particles[i].pinned {
                continue;
            }
            let pos = self.particles[i].position;
            let prev = self.particles[i].previous_position;
            let acc = self.particles[i].acceleration;
            let new_pos = pos * 2.0 - prev + acc * dt_sq;
            self.particles[i].previous_position = pos;
            self.particles[i].position = new_pos;
        }

        // Reset accelerations
        for p in &mut self.particles {
            p.acceleration = Vec3::ZERO;
        }

        self.time += dt;
    }

    /// Position-based dynamics (Jakobsen method).
    ///
    /// Performs a Verlet step with only gravity (no spring forces), then
    /// iteratively projects particle positions to satisfy distance constraints.
    pub fn step_with_constraints(&mut self, dt: f64, iterations: usize) {
        // Reset accelerations and apply only gravity
        for p in &mut self.particles {
            p.acceleration = if p.pinned { Vec3::ZERO } else { self.gravity };
        }

        // Verlet integration (gravity only, no spring forces)
        let dt_sq = dt * dt;
        let n = self.particles.len();
        for i in 0..n {
            if self.particles[i].pinned {
                continue;
            }
            let pos = self.particles[i].position;
            let prev = self.particles[i].previous_position;
            let acc = self.particles[i].acceleration;
            let new_pos = pos * 2.0 - prev + acc * dt_sq;
            self.particles[i].previous_position = pos;
            self.particles[i].position = new_pos;
        }

        // Constraint satisfaction loop
        for _ in 0..iterations {
            for s_idx in 0..self.springs.len() {
                let a = self.springs[s_idx].particle_a;
                let b = self.springs[s_idx].particle_b;
                let rest = self.springs[s_idx].rest_length;

                let delta = self.particles[b].position - self.particles[a].position;
                let dist = delta.magnitude();

                if dist < MIN_DISTANCE {
                    continue;
                }

                let diff = (dist - rest) / dist;
                let correction = delta * (0.5 * diff);

                let a_pinned = self.particles[a].pinned;
                let b_pinned = self.particles[b].pinned;

                match (a_pinned, b_pinned) {
                    (false, false) => {
                        self.particles[a].position = self.particles[a].position + correction;
                        self.particles[b].position = self.particles[b].position - correction;
                    }
                    (true, false) => {
                        self.particles[b].position =
                            self.particles[b].position - correction * 2.0;
                    }
                    (false, true) => {
                        self.particles[a].position =
                            self.particles[a].position + correction * 2.0;
                    }
                    (true, true) => {}
                }
            }
        }

        // Reset accelerations
        for p in &mut self.particles {
            p.acceleration = Vec3::ZERO;
        }

        self.time += dt;
    }

    /// Total mechanical energy: KE + PE_gravity + PE_spring.
    ///
    /// KE = 0.5 * m * v^2 (velocity estimated from Verlet)
    /// PE_gravity = m * (g . pos)  (dot product to handle arbitrary gravity direction)
    /// PE_spring = 0.5 * k * (|dx| - L0)^2
    pub fn total_energy(&self, dt: f64) -> f64 {
        let mut energy = 0.0;

        // Kinetic + gravitational potential
        for p in &self.particles {
            if p.pinned {
                continue;
            }
            let vel = if dt > 0.0 {
                (p.position - p.previous_position) * (1.0 / dt)
            } else {
                Vec3::ZERO
            };
            let ke = 0.5 * p.mass * vel.magnitude_squared();
            // PE_gravity = -m * (g . pos) so that lower = less PE with g=(0,-9.81,0)
            // Convention: PE = m*g*h. With g as a vector pointing down and pos.y as height,
            // PE_grav = -m * (g . pos) gives positive PE for positive height.
            let pe_grav = -p.mass * self.gravity.dot(&p.position);
            energy += ke + pe_grav;
        }

        // Spring potential energy
        for spring in &self.springs {
            let delta = self.particles[spring.particle_b].position
                - self.particles[spring.particle_a].position;
            let stretch = delta.magnitude() - spring.rest_length;
            energy += 0.5 * spring.stiffness * stretch * stretch;
        }

        energy
    }

    /// Total linear momentum: sum of m * v for all non-pinned particles.
    pub fn total_momentum(&self, dt: f64) -> Vec3 {
        let mut momentum = Vec3::ZERO;
        for p in &self.particles {
            if p.pinned {
                continue;
            }
            let vel = if dt > 0.0 {
                (p.position - p.previous_position) * (1.0 / dt)
            } else {
                Vec3::ZERO
            };
            momentum = momentum + vel * p.mass;
        }
        momentum
    }
}

/// Creates a rectangular cloth grid with structural and shear springs.
///
/// Particles are laid out in the XY plane. The top row (y = (height-1)*spacing)
/// is pinned. Gravity points in -Y.
///
/// Structural springs: horizontal and vertical neighbors (rest_length = spacing).
/// Shear springs: diagonal neighbors (rest_length = spacing * sqrt(2)).
pub fn create_cloth_grid(
    width: usize,
    height: usize,
    spacing: f64,
    mass: f64,
    stiffness: f64,
    damping: f64,
) -> MassSpringSystem {
    assert!(mass > 0.0, "particle mass must be positive");
    assert!(spacing > 0.0, "grid spacing must be positive");
    let mut system = MassSpringSystem::new(Vec3::new(0.0, DEFAULT_GRAVITY_Y, 0.0));

    // Create particles in row-major order: index = row * width + col
    for row in 0..height {
        for col in 0..width {
            let x = col as f64 * spacing;
            let y = (height - 1 - row) as f64 * spacing;
            let pos = Vec3::new(x, y, 0.0);

            let is_top_row = row == 0;
            let particle = if is_top_row {
                Particle::new_pinned(pos)
            } else {
                Particle::new(pos, mass)
            };
            system.add_particle(particle);
        }
    }

    let idx = |row: usize, col: usize| -> usize { row * width + col };
    let diag_rest = spacing * std::f64::consts::SQRT_2;

    for row in 0..height {
        for col in 0..width {
            // Horizontal (right neighbor)
            if col + 1 < width {
                system.add_spring(Spring::new(
                    idx(row, col),
                    idx(row, col + 1),
                    spacing,
                    stiffness,
                    damping,
                ));
            }
            // Vertical (down neighbor)
            if row + 1 < height {
                system.add_spring(Spring::new(
                    idx(row, col),
                    idx(row + 1, col),
                    spacing,
                    stiffness,
                    damping,
                ));
            }
            // Diagonal: down-right
            if row + 1 < height && col + 1 < width {
                system.add_spring(Spring::new(
                    idx(row, col),
                    idx(row + 1, col + 1),
                    diag_rest,
                    stiffness,
                    damping,
                ));
            }
            // Diagonal: down-left
            if row + 1 < height && col > 0 {
                system.add_spring(Spring::new(
                    idx(row, col),
                    idx(row + 1, col - 1),
                    diag_rest,
                    stiffness,
                    damping,
                ));
            }
        }
    }

    system
}

/// Creates a 1D rope (linear chain of particles connected by springs).
///
/// The first particle is pinned. Gravity points in -Y.
pub fn create_rope(
    n_particles: usize,
    spacing: f64,
    mass: f64,
    stiffness: f64,
    damping: f64,
) -> MassSpringSystem {
    assert!(mass > 0.0, "particle mass must be positive");
    assert!(spacing > 0.0, "rope spacing must be positive");
    let mut system = MassSpringSystem::new(Vec3::new(0.0, DEFAULT_GRAVITY_Y, 0.0));

    for i in 0..n_particles {
        let pos = Vec3::new(i as f64 * spacing, 0.0, 0.0);
        let particle = if i == 0 {
            Particle::new_pinned(pos)
        } else {
            Particle::new(pos, mass)
        };
        system.add_particle(particle);
    }

    for i in 0..n_particles.saturating_sub(1) {
        system.add_spring(Spring::new(i, i + 1, spacing, stiffness, damping));
    }

    system
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-6;
    const ENERGY_TOLERANCE: f64 = 0.02;
    const PERIOD_TOLERANCE: f64 = 0.05;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_relative(a: f64, b: f64, rel_tol: f64) -> bool {
        let denom = b.abs().max(1e-15);
        ((a - b) / denom).abs() < rel_tol
    }

    #[test]
    fn test_single_spring_oscillation_period() {
        // Two equal masses connected by a spring, free in space (no gravity).
        // Reduced mass mu = m/2. Angular frequency omega = sqrt(k/mu) = sqrt(2k/m).
        // Period T = 2*pi / omega.
        let k = 100.0;
        let m = 1.0;
        let initial_stretch = 0.5;
        let rest_length = 2.0;

        let mut system = MassSpringSystem::new(Vec3::ZERO);
        let mut p0 = Particle::new(Vec3::new(0.0, 0.0, 0.0), m);
        let mut p1 = Particle::new(Vec3::new(rest_length + initial_stretch, 0.0, 0.0), m);

        // Give them zero initial velocity by setting prev_pos = pos
        p0.previous_position = p0.position;
        p1.previous_position = p1.position;

        system.add_particle(p0);
        system.add_particle(p1);
        system.add_spring(Spring::new(0, 1, rest_length, k, 0.0));

        // Reduced mass for two equal masses: mu = m/2
        let reduced_mass = m / 2.0;
        let omega = (k / reduced_mass).sqrt();
        let expected_period = 2.0 * PI / omega;

        // Simulate and find the period by detecting when the spring returns
        // to its initial stretched state (one full oscillation of relative coordinate).
        let dt = 0.0001;
        let initial_dist = rest_length + initial_stretch;
        let mut prev_dist = initial_dist;
        let mut crossings: Vec<f64> = Vec::new();

        let max_steps = (expected_period * 3.0 / dt) as usize;
        for step in 1..max_steps {
            system.step_verlet(dt);
            let dist = (system.particles[1].position - system.particles[0].position).magnitude();

            // Detect downward zero-crossing through rest_length + initial_stretch
            // (i.e., returning to max extension from above)
            if prev_dist >= initial_dist && dist < initial_dist && step > 10 {
                // Linear interpolation for crossing time
                let frac = (prev_dist - initial_dist) / (prev_dist - dist);
                crossings.push((step as f64 - 1.0 + frac) * dt);
            }
            prev_dist = dist;
        }

        let num_crossings = crossings.len();
        assert!(
            num_crossings >= 2,
            "expected at least 2 crossings, got {num_crossings}",
        );
        let measured_period = crossings[1] - crossings[0];
        assert!(
            approx_relative(measured_period, expected_period, PERIOD_TOLERANCE),
            "measured period {measured_period:.6} vs expected {expected_period:.6}"
        );
    }

    #[test]
    fn test_pinned_particle_does_not_move() {
        let mut system = MassSpringSystem::new(Vec3::new(0.0, DEFAULT_GRAVITY_Y, 0.0));
        let pinned_pos = Vec3::new(1.0, 5.0, 3.0);
        let pinned = Particle::new_pinned(pinned_pos);
        system.add_particle(pinned);
        system.add_particle(Particle::new(Vec3::new(1.0, 4.0, 3.0), 1.0));
        system.add_spring(Spring::new(0, 1, 1.0, 50.0, 0.0));

        for _ in 0..1000 {
            system.step_verlet(0.001);
        }

        let actual_pos = system.particles[0].position;
        assert!(
            approx(actual_pos.x, pinned_pos.x, TOLERANCE)
                && approx(actual_pos.y, pinned_pos.y, TOLERANCE)
                && approx(actual_pos.z, pinned_pos.z, TOLERANCE),
            "pinned particle moved from {pinned_pos:?} to {actual_pos:?}",
        );
    }

    #[test]
    fn test_rope_hanging_under_gravity() {
        // A rope with 3 particles: first pinned at origin, two hanging.
        // Under gravity the rope should hang downward.
        let n = 3;
        let spacing = 1.0;
        let mass = 1.0;
        let stiffness = 1000.0;
        let damping = 5.0;
        let mut rope = create_rope(n, spacing, mass, stiffness, damping);

        // Simulate until settled
        let dt = 0.001;
        for _ in 0..20000 {
            rope.step_verlet(dt);
        }

        // The bottom particle should be below the pinned one
        let y0 = rope.particles[0].position.y;
        let y1 = rope.particles[1].position.y;
        let y2 = rope.particles[2].position.y;
        assert!(
            y2 < y0,
            "bottom particle y={y2:.4} should be below pinned y={y0:.4}",
        );

        // Each successive particle should be lower
        assert!(
            y1 > y2,
            "particle 1 y={y1:.4} should be above particle 2 y={y2:.4}",
        );

        // Check that bottom particle is roughly at the right position.
        // With stiff springs and damping, the rope should be nearly vertical.
        // Total weight below particle 0 is 2*m*g, each spring carries weight of mass below it.
        // Spring 0-1 carries 2*m*g, spring 1-2 carries 1*m*g.
        // Extension of spring i: F_i / k.
        // Total drop ≈ spacing + 2*m*g/k + spacing + 1*m*g/k
        let g = 9.81;
        let expected_drop = 2.0 * spacing + (2.0 * mass * g + 1.0 * mass * g) / stiffness;
        let actual_drop = rope.particles[0].position.y - rope.particles[2].position.y;
        assert!(
            approx_relative(actual_drop, expected_drop, 0.6),
            "rope drop {actual_drop:.4} vs expected {expected_drop:.4}"
        );
    }

    #[test]
    fn test_energy_conservation_no_damping() {
        // Two masses on a spring in zero gravity, no damping.
        // Total energy should be conserved.
        let k = 50.0;
        let m = 2.0;
        let rest = 1.0;
        let stretch = 0.3;

        let mut system = MassSpringSystem::new(Vec3::ZERO);
        system.add_particle(Particle::new(Vec3::new(0.0, 0.0, 0.0), m));
        system.add_particle(Particle::new(Vec3::new(rest + stretch, 0.0, 0.0), m));
        system.add_spring(Spring::new(0, 1, rest, k, 0.0));

        let dt = 0.0001;
        let initial_energy = system.total_energy(dt);

        for _ in 0..50000 {
            system.step_verlet(dt);
        }

        let final_energy = system.total_energy(dt);
        assert!(
            approx_relative(final_energy, initial_energy, ENERGY_TOLERANCE),
            "energy not conserved: initial={initial_energy:.6}, final={final_energy:.6}"
        );
    }

    #[test]
    fn test_constraint_solver_maintains_lengths() {
        // Create a rope and run with constraint solver.
        // After settling, spring lengths should be close to rest length.
        let n = 5;
        let spacing = 1.0;
        let stiffness = 100.0;

        let mut system = create_rope(n, spacing, 1.0, stiffness, 1.0);

        let dt = 0.01;
        let constraint_iterations = 10;
        for _ in 0..2000 {
            system.step_with_constraints(dt, constraint_iterations);
        }

        let max_deviation: f64 = system
            .springs
            .iter()
            .map(|s| {
                let d = (system.particles[s.particle_b].position
                    - system.particles[s.particle_a].position)
                    .magnitude();
                (d - s.rest_length).abs()
            })
            .fold(0.0, f64::max);

        assert!(
            max_deviation < 0.05,
            "constraint solver max length deviation: {max_deviation:.6}"
        );

        // Also run the same scenario with force-based for comparison
        let mut force_system = create_rope(n, spacing, 1.0, stiffness, 1.0);
        for _ in 0..2000 {
            force_system.step_verlet(dt);
        }

        let force_max_deviation: f64 = force_system
            .springs
            .iter()
            .map(|s| {
                let d = (force_system.particles[s.particle_b].position
                    - force_system.particles[s.particle_a].position)
                    .magnitude();
                (d - s.rest_length).abs()
            })
            .fold(0.0, f64::max);

        // Constraint solver should do at least as well (usually much better)
        assert!(
            max_deviation <= force_max_deviation + 0.01,
            "constraint solver ({max_deviation:.6}) should maintain lengths \
             at least as well as force-based ({force_max_deviation:.6})"
        );
    }

    #[test]
    fn test_cloth_grid_particle_and_spring_counts() {
        let w = 5;
        let h = 4;
        let cloth = create_cloth_grid(w, h, 1.0, 1.0, 100.0, 0.5);

        let expected_particles = w * h;
        assert_eq!(cloth.particles.len(), expected_particles);

        // Structural horizontal: h * (w-1)
        let horizontal = h * (w - 1);
        // Structural vertical: (h-1) * w
        let vertical = (h - 1) * w;
        // Shear down-right: (h-1) * (w-1)
        let diag_right = (h - 1) * (w - 1);
        // Shear down-left: (h-1) * (w-1)
        let diag_left = (h - 1) * (w - 1);
        let expected_springs = horizontal + vertical + diag_right + diag_left;
        let actual_springs = cloth.springs.len();
        assert_eq!(
            actual_springs,
            expected_springs,
            "expected {expected_springs} springs, got {actual_springs}",
        );

        // Top row should be pinned
        for col in 0..w {
            assert!(
                cloth.particles[col].pinned,
                "top row particle {col} should be pinned"
            );
        }

        // Other rows should not be pinned
        for row in 1..h {
            for col in 0..w {
                assert!(
                    !cloth.particles[row * w + col].pinned,
                    "particle at row={row}, col={col} should not be pinned"
                );
            }
        }
    }

    #[test]
    fn test_rope_structure() {
        let n = 10;
        let rope = create_rope(n, 0.5, 1.0, 100.0, 0.1);

        assert_eq!(rope.particles.len(), n);
        assert_eq!(rope.springs.len(), n - 1);
        assert!(rope.particles[0].pinned);
        for i in 1..n {
            assert!(!rope.particles[i].pinned);
        }
    }

    #[test]
    fn test_total_momentum_no_external_forces() {
        // Two free masses with a spring, no gravity. Momentum should be conserved.
        let k = 80.0;
        let m = 1.5;
        let rest = 1.0;

        let mut system = MassSpringSystem::new(Vec3::ZERO);

        let mut p0 = Particle::new(Vec3::new(0.0, 0.0, 0.0), m);
        let mut p1 = Particle::new(Vec3::new(rest + 0.2, 0.0, 0.0), m);
        // Give initial velocity to p0 via offset previous_position
        let dt = 0.0001;
        let v0 = Vec3::new(1.0, 0.0, 0.0);
        p0.previous_position = p0.position - v0 * dt;
        p1.previous_position = p1.position;

        system.add_particle(p0);
        system.add_particle(p1);
        system.add_spring(Spring::new(0, 1, rest, k, 0.0));

        let initial_momentum = system.total_momentum(dt);

        for _ in 0..10000 {
            system.step_verlet(dt);
        }

        let final_momentum = system.total_momentum(dt);

        assert!(
            approx(initial_momentum.x, final_momentum.x, 0.05),
            "momentum x: initial={:.6}, final={:.6}",
            initial_momentum.x,
            final_momentum.x
        );
        assert!(
            approx(initial_momentum.y, final_momentum.y, 0.05),
            "momentum y: initial={:.6}, final={:.6}",
            initial_momentum.y,
            final_momentum.y
        );
    }

    #[test]
    fn test_step_verlet_coincident_particles_skips_force() {
        // Two particles at the exact same position: dist < MIN_DISTANCE branch
        let mut system = MassSpringSystem::new(Vec3::ZERO);
        let origin = Vec3::new(1.0, 2.0, 3.0);
        system.add_particle(Particle::new(origin, 1.0));
        system.add_particle(Particle::new(origin, 1.0));
        system.add_spring(Spring::new(0, 1, 1.0, 100.0, 0.5));

        let pos_before_a = system.particles[0].position;
        let pos_before_b = system.particles[1].position;

        system.step_verlet(0.01);

        // With zero gravity and coincident particles, spring force is skipped,
        // so positions stay the same (Verlet: new = 2*pos - prev + 0 = pos)
        assert!(approx(system.particles[0].position.x, pos_before_a.x, TOLERANCE));
        assert!(approx(system.particles[0].position.y, pos_before_a.y, TOLERANCE));
        assert!(approx(system.particles[1].position.x, pos_before_b.x, TOLERANCE));
        assert!(approx(system.particles[1].position.y, pos_before_b.y, TOLERANCE));
    }

    #[test]
    fn test_step_verlet_zero_dt_applies_spring_force_without_damping() {
        // dt=0 branch: spring force applied without damping calculation
        let mut system = MassSpringSystem::new(Vec3::ZERO);
        let rest = 1.0;
        let stretch = 0.5;
        let k = 100.0;

        system.add_particle(Particle::new(Vec3::new(0.0, 0.0, 0.0), 1.0));
        system.add_particle(Particle::new(Vec3::new(rest + stretch, 0.0, 0.0), 1.0));
        system.add_spring(Spring::new(0, 1, rest, k, 10.0));

        // With dt=0, Verlet update is pos*2 - prev + acc*0 = pos (no movement),
        // but the spring force accumulation still runs through the dt<=0 branch
        system.step_verlet(0.0);

        // Particles should not move (dt^2 = 0)
        assert!(approx(system.particles[0].position.x, 0.0, TOLERANCE));
        assert!(approx(system.particles[1].position.x, rest + stretch, TOLERANCE));
    }

    #[test]
    fn test_constraint_solver_free_a_pinned_b() {
        // Exercises the (false, true) match arm: particle A is free, B is pinned
        let mut system = MassSpringSystem::new(Vec3::ZERO);

        // A is free, B is pinned -- spring connects them
        system.add_particle(Particle::new(Vec3::new(0.0, 0.0, 0.0), 1.0));
        system.add_particle(Particle::new_pinned(Vec3::new(3.0, 0.0, 0.0)));
        system.add_spring(Spring::new(0, 1, 1.0, 100.0, 0.0));

        // Run constraint solver: A should be pulled toward B since rest < dist
        system.step_with_constraints(0.01, 20);

        // A should have moved closer to B (toward x=3), getting correction*2.0
        let ax = system.particles[0].position.x;
        assert!(
            ax > 0.0,
            "free particle A should move toward pinned B, got x={ax:.4}",
        );
        // B should remain pinned
        let bx = system.particles[1].position.x;
        assert!(
            approx(bx, 3.0, TOLERANCE),
            "pinned particle B should not move, got x={bx:.4}",
        );
    }

    #[test]
    fn test_constraint_solver_both_pinned() {
        // Exercises the (true, true) match arm: both particles are pinned
        let mut system = MassSpringSystem::new(Vec3::ZERO);

        let pos_a = Vec3::new(0.0, 0.0, 0.0);
        let pos_b = Vec3::new(5.0, 0.0, 0.0);
        system.add_particle(Particle::new_pinned(pos_a));
        system.add_particle(Particle::new_pinned(pos_b));
        system.add_spring(Spring::new(0, 1, 1.0, 100.0, 0.0));

        system.step_with_constraints(0.01, 10);

        // Neither particle should move
        assert!(approx(system.particles[0].position.x, pos_a.x, TOLERANCE));
        assert!(approx(system.particles[0].position.y, pos_a.y, TOLERANCE));
        assert!(approx(system.particles[1].position.x, pos_b.x, TOLERANCE));
        assert!(approx(system.particles[1].position.y, pos_b.y, TOLERANCE));
    }

    #[test]
    fn test_constraint_solver_coincident_particles_skips() {
        // Exercises the dist < MIN_DISTANCE continue in constraint loop
        let mut system = MassSpringSystem::new(Vec3::ZERO);

        let pos = Vec3::new(1.0, 1.0, 1.0);
        system.add_particle(Particle::new(pos, 1.0));
        system.add_particle(Particle::new(pos, 1.0));
        system.add_spring(Spring::new(0, 1, 1.0, 100.0, 0.0));

        // Should not panic even with coincident particles
        system.step_with_constraints(0.0, 5);

        // Particles remain at the same position (dt=0 means no Verlet movement)
        assert!(approx(system.particles[0].position.x, pos.x, TOLERANCE));
        assert!(approx(system.particles[1].position.x, pos.x, TOLERANCE));
    }

    #[test]
    fn test_total_energy_with_pinned_particles() {
        // Exercises the pinned-particle skip in total_energy
        let mut system = MassSpringSystem::new(Vec3::new(0.0, -9.81, 0.0));

        system.add_particle(Particle::new_pinned(Vec3::new(0.0, 5.0, 0.0)));
        system.add_particle(Particle::new(Vec3::new(0.0, 3.0, 0.0), 2.0));
        system.add_spring(Spring::new(0, 1, 1.0, 50.0, 0.0));

        let dt = 0.01;
        let energy = system.total_energy(dt);

        // Energy should come only from the free particle and the spring.
        // Pinned particle contributes nothing to KE or PE_grav.
        // Free particle: vel=0 (pos==prev_pos), so KE=0.
        // PE_grav = -m * g.dot(pos) = -2.0 * (-9.81*3.0) = 58.86
        // Spring stretch = 2.0 - 1.0 = 1.0, PE_spring = 0.5 * 50 * 1.0 = 25.0
        let expected = 58.86 + 25.0;
        assert!(
            approx(energy, expected, 0.01),
            "energy {energy:.4} vs expected {expected:.4}"
        );
    }

    #[test]
    fn test_total_energy_zero_dt() {
        // Exercises the dt==0 branch in total_energy (velocity becomes ZERO)
        let mut system = MassSpringSystem::new(Vec3::ZERO);

        let mut p = Particle::new(Vec3::new(1.0, 0.0, 0.0), 1.0);
        // Offset previous position to simulate velocity -- but dt=0 should ignore it
        p.previous_position = Vec3::new(0.0, 0.0, 0.0);
        system.add_particle(p);
        system.add_spring(Spring::new(0, 0, 0.0, 0.0, 0.0));

        let energy = system.total_energy(0.0);

        // With dt=0, velocity is ZERO, so KE=0. No gravity, so PE_grav=0.
        // Spring from particle 0 to itself: delta=0, stretch=0, PE_spring=0.
        assert!(
            approx(energy, 0.0, TOLERANCE),
            "energy with dt=0 should be 0, got {energy:.6}"
        );
    }

    #[test]
    fn test_total_momentum_with_pinned_particles() {
        // Exercises the pinned-particle skip in total_momentum
        let mut system = MassSpringSystem::new(Vec3::ZERO);

        // Pinned particle with offset previous_position (should be excluded)
        let mut pinned = Particle::new_pinned(Vec3::new(0.0, 0.0, 0.0));
        pinned.previous_position = Vec3::new(-1.0, 0.0, 0.0);
        system.add_particle(pinned);

        // Free particle with known velocity
        let dt = 0.01;
        let mut free = Particle::new(Vec3::new(5.0, 0.0, 0.0), 2.0);
        free.previous_position = Vec3::new(4.9, 0.0, 0.0);
        system.add_particle(free);

        let momentum = system.total_momentum(dt);

        // Only the free particle contributes: v = (5.0 - 4.9)/0.01 = 10.0
        // momentum = m*v = 2.0 * 10.0 = 20.0
        assert!(
            approx(momentum.x, 20.0, TOLERANCE),
            "momentum x should be 20.0, got {:.6}",
            momentum.x
        );
        assert!(
            approx(momentum.y, 0.0, TOLERANCE),
            "momentum y should be 0, got {:.6}",
            momentum.y
        );
    }

    #[test]
    fn test_total_momentum_zero_dt() {
        // Exercises the dt==0 branch in total_momentum (velocity becomes ZERO)
        let mut system = MassSpringSystem::new(Vec3::ZERO);

        let mut p = Particle::new(Vec3::new(5.0, 0.0, 0.0), 3.0);
        p.previous_position = Vec3::new(0.0, 0.0, 0.0);
        system.add_particle(p);

        let momentum = system.total_momentum(0.0);

        assert!(
            approx(momentum.x, 0.0, TOLERANCE),
            "momentum with dt=0 should be 0, got {:.6}",
            momentum.x
        );
    }
}
