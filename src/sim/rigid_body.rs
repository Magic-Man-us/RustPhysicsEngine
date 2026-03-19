use crate::math::Vec3;
use crate::quaternion::Quaternion;

const HALF: f64 = 0.5;

// Moment-of-inertia fractions for standard shapes
const SPHERE_INERTIA_FACTOR: f64 = 2.0 / 5.0;
const CYLINDER_AXIAL_FACTOR: f64 = 0.5;
const BOX_INERTIA_DENOM: f64 = 12.0;

pub struct RigidBody {
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f64,

    pub orientation: Quaternion,
    pub angular_velocity: Vec3,
    pub inertia: [f64; 3],

    force: Vec3,
    torque: Vec3,
}

impl RigidBody {
    #[must_use]
    pub fn new(mass: f64, inertia: [f64; 3]) -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            mass,
            orientation: Quaternion::identity(),
            angular_velocity: Vec3::ZERO,
            inertia,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
        }
    }

    #[must_use]
    pub fn new_sphere(mass: f64, radius: f64) -> Self {
        let i = SPHERE_INERTIA_FACTOR * mass * radius * radius;
        Self::new(mass, [i, i, i])
    }

    #[must_use]
    pub fn new_box(mass: f64, wx: f64, wy: f64, wz: f64) -> Self {
        let ix = mass * (wy * wy + wz * wz) / BOX_INERTIA_DENOM;
        let iy = mass * (wx * wx + wz * wz) / BOX_INERTIA_DENOM;
        let iz = mass * (wx * wx + wy * wy) / BOX_INERTIA_DENOM;
        Self::new(mass, [ix, iy, iz])
    }

    #[must_use]
    pub fn new_cylinder(mass: f64, radius: f64, height: f64) -> Self {
        let i_axial = CYLINDER_AXIAL_FACTOR * mass * radius * radius;
        let i_perp = mass * (3.0 * radius * radius + height * height) / BOX_INERTIA_DENOM;
        Self::new(mass, [i_perp, i_perp, i_axial])
    }

    pub fn apply_force(&mut self, force: Vec3) {
        self.force = self.force + force;
    }

    pub fn apply_force_at_point(&mut self, force: Vec3, point: Vec3) {
        self.force = self.force + force;
        let r = point - self.position;
        self.torque = self.torque + r.cross(&force);
    }

    pub fn apply_torque(&mut self, torque: Vec3) {
        self.torque = self.torque + torque;
    }

    pub fn clear_forces(&mut self) {
        self.force = Vec3::ZERO;
        self.torque = Vec3::ZERO;
    }

    /// Symplectic Euler integration with full Euler rotation equations.
    pub fn step(&mut self, dt: f64) {
        // --- Linear dynamics: a = F/m, v += a*dt, pos += v*dt ---
        let linear_accel = self.force * (1.0 / self.mass);
        self.velocity = self.velocity + linear_accel * dt;
        self.position = self.position + self.velocity * dt;

        // --- Angular dynamics (Euler's rotation equations in body frame) ---
        let q_inv = self.orientation.conjugate();

        // Transform world-frame angular velocity to body frame
        let omega_body = q_inv.rotate_vec(self.angular_velocity);

        // Transform world-frame torque to body frame
        let torque_body = q_inv.rotate_vec(self.torque);

        // I*omega in body frame (diagonal inertia tensor)
        let i_omega = Vec3::new(
            self.inertia[0] * omega_body.x,
            self.inertia[1] * omega_body.y,
            self.inertia[2] * omega_body.z,
        );

        // Euler equations: tau_body = I*alpha + omega x (I*omega)
        // => alpha_body = I^-1 * (tau_body - omega x (I*omega))
        let gyroscopic = omega_body.cross(&i_omega);
        let alpha_body = Vec3::new(
            (torque_body.x - gyroscopic.x) / self.inertia[0],
            (torque_body.y - gyroscopic.y) / self.inertia[1],
            (torque_body.z - gyroscopic.z) / self.inertia[2],
        );

        // Transform angular acceleration back to world frame
        let alpha_world = self.orientation.rotate_vec(alpha_body);

        // Update angular velocity in world frame
        self.angular_velocity = self.angular_velocity + alpha_world * dt;

        // Update quaternion: q += 0.5 * dt * Quaternion(0, omega) * q
        let omega_quat = Quaternion::new(
            0.0,
            self.angular_velocity.x,
            self.angular_velocity.y,
            self.angular_velocity.z,
        );
        self.orientation = (self.orientation + omega_quat * self.orientation * (HALF * dt)).normalize();

        self.clear_forces();
    }

    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let translational = HALF * self.mass * self.velocity.magnitude_squared();

        let omega_body = self.orientation.conjugate().rotate_vec(self.angular_velocity);
        let rotational = HALF * (
            self.inertia[0] * omega_body.x * omega_body.x
            + self.inertia[1] * omega_body.y * omega_body.y
            + self.inertia[2] * omega_body.z * omega_body.z
        );

        translational + rotational
    }

    #[must_use]
    pub fn angular_momentum(&self) -> Vec3 {
        let omega_body = self.orientation.conjugate().rotate_vec(self.angular_velocity);
        let l_body = Vec3::new(
            self.inertia[0] * omega_body.x,
            self.inertia[1] * omega_body.y,
            self.inertia[2] * omega_body.z,
        );
        self.orientation.rotate_vec(l_body)
    }

    #[must_use]
    pub fn local_to_world(&self, local_point: Vec3) -> Vec3 {
        self.position + self.orientation.rotate_vec(local_point)
    }

    #[must_use]
    pub fn world_to_local(&self, world_point: Vec3) -> Vec3 {
        self.orientation.conjugate().rotate_vec(world_point - self.position)
    }

    #[must_use]
    pub fn velocity_at_point(&self, world_point: Vec3) -> Vec3 {
        let r = world_point - self.position;
        self.velocity + self.angular_velocity.cross(&r)
    }
}

pub struct RigidBodySystem {
    pub bodies: Vec<RigidBody>,
    pub gravity: Vec3,
    pub time: f64,
}

impl RigidBodySystem {
    #[must_use]
    pub fn new(gravity: Vec3) -> Self {
        Self {
            bodies: Vec::new(),
            gravity,
            time: 0.0,
        }
    }

    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    pub fn step(&mut self, dt: f64) {
        for body in &mut self.bodies {
            let gravity_force = self.gravity * body.mass;
            body.apply_force(gravity_force);
            body.step(dt);
        }
        self.time += dt;
    }

    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.bodies.iter().map(|b| {
            let ke = b.kinetic_energy();
            // Gravitational PE = m * g · position (works for arbitrary gravity direction)
            let pe = -b.mass * self.gravity.dot(&b.position);
            ke + pe
        }).sum()
    }

    #[must_use]
    pub fn total_momentum(&self) -> Vec3 {
        self.bodies.iter().fold(Vec3::ZERO, |acc, b| {
            acc + b.velocity * b.mass
        })
    }
}

// --- Sphere-sphere collision ---

#[must_use]
pub fn sphere_sphere_collision(
    a: &RigidBody,
    radius_a: f64,
    b: &RigidBody,
    radius_b: f64,
) -> Option<(Vec3, f64)> {
    let diff = b.position - a.position;
    let dist = diff.magnitude();
    let min_dist = radius_a + radius_b;
    if dist >= min_dist || dist == 0.0 {
        return None;
    }
    let normal = diff * (1.0 / dist);
    let penetration = min_dist - dist;
    Some((normal, penetration))
}

pub fn resolve_collision(
    a: &mut RigidBody,
    b: &mut RigidBody,
    normal: Vec3,
    restitution: f64,
) {
    let v_rel = a.velocity - b.velocity;
    let v_rel_normal = v_rel.dot(&normal);

    // Bodies separating — no impulse needed
    if v_rel_normal > 0.0 {
        return;
    }

    let inv_mass_sum = 1.0 / a.mass + 1.0 / b.mass;
    let j = -(1.0 + restitution) * v_rel_normal / inv_mass_sum;

    a.velocity = a.velocity + normal * (j / a.mass);
    b.velocity = b.velocity - normal * (j / b.mass);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-6;
    const FINE_DT: f64 = 1e-5;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_rel(a: f64, b: f64, rel_tol: f64) -> bool {
        let denom = a.abs().max(b.abs()).max(1e-15);
        (a - b).abs() / denom < rel_tol
    }

    fn vec3_approx(a: Vec3, b: Vec3, tol: f64) -> bool {
        (a.x - b.x).abs() < tol && (a.y - b.y).abs() < tol && (a.z - b.z).abs() < tol
    }

    // --- Free fall under gravity ---

    #[test]
    fn free_fall_position_matches_half_gt_squared() {
        let g = 9.80665;
        let mut sys = RigidBodySystem::new(Vec3::new(0.0, -g, 0.0));
        sys.add_body(RigidBody::new_sphere(1.0, 0.5));

        let total_time = 2.0;
        let steps = (total_time / FINE_DT) as usize;
        for _ in 0..steps {
            sys.step(FINE_DT);
        }

        let expected_y = -HALF * g * total_time * total_time;
        let actual_y = sys.bodies[0].position.y;
        assert!(
            approx_rel(actual_y, expected_y, 1e-3),
            "expected y ~ {expected_y}, got {actual_y}"
        );
    }

    // --- Torque-free symmetric body: angular velocity constant ---

    #[test]
    fn torque_free_symmetric_body_constant_angular_velocity() {
        let mut body = RigidBody::new_sphere(2.0, 1.0);
        let omega_0 = Vec3::new(1.0, 2.0, 3.0);
        body.angular_velocity = omega_0;

        let steps = 10_000;
        for _ in 0..steps {
            body.step(1e-4);
        }

        assert!(
            vec3_approx(body.angular_velocity, omega_0, 1e-3),
            "omega should stay constant for symmetric body, got {:?}",
            body.angular_velocity
        );
    }

    // --- Sphere collision: momentum conserved ---

    #[test]
    fn sphere_collision_momentum_conserved() {
        let mut a = RigidBody::new_sphere(2.0, 1.0);
        a.velocity = Vec3::new(3.0, 0.0, 0.0);

        let mut b = RigidBody::new_sphere(1.0, 1.0);
        b.position = Vec3::new(5.0, 0.0, 0.0);
        b.velocity = Vec3::new(-1.0, 0.0, 0.0);

        let p_before = a.velocity * a.mass + b.velocity * b.mass;

        let normal = Vec3::new(1.0, 0.0, 0.0);
        resolve_collision(&mut a, &mut b, normal, 0.7);

        let p_after = a.velocity * a.mass + b.velocity * b.mass;
        assert!(
            vec3_approx(p_before, p_after, TOLERANCE),
            "momentum not conserved: before={p_before:?}, after={p_after:?}"
        );
    }

    // --- Energy conserved for elastic collision ---

    #[test]
    fn elastic_collision_energy_conserved() {
        let mut a = RigidBody::new_sphere(3.0, 1.0);
        a.velocity = Vec3::new(4.0, 0.0, 0.0);

        let mut b = RigidBody::new_sphere(2.0, 1.0);
        b.position = Vec3::new(5.0, 0.0, 0.0);
        b.velocity = Vec3::new(-2.0, 0.0, 0.0);

        let ke_before = a.kinetic_energy() + b.kinetic_energy();

        let normal = Vec3::new(1.0, 0.0, 0.0);
        resolve_collision(&mut a, &mut b, normal, 1.0);

        let ke_after = a.kinetic_energy() + b.kinetic_energy();
        assert!(
            approx_rel(ke_before, ke_after, 1e-9),
            "KE not conserved in elastic collision: before={ke_before}, after={ke_after}"
        );
    }

    // --- Quaternion orientation: 90-degree rotation transforms points ---

    #[test]
    fn quaternion_90deg_rotation_transforms_point() {
        let mut body = RigidBody::new_sphere(1.0, 1.0);
        body.orientation = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI / 2.0);

        let local = Vec3::new(1.0, 0.0, 0.0);
        let world = body.local_to_world(local);

        assert!(
            vec3_approx(world, Vec3::new(0.0, 1.0, 0.0), TOLERANCE),
            "90-deg z rotation should map (1,0,0) to (0,1,0), got {world:?}"
        );

        let back = body.world_to_local(world);
        assert!(
            vec3_approx(back, local, TOLERANCE),
            "round-trip local->world->local failed: got {back:?}"
        );
    }

    // --- velocity_at_point for spinning body matches omega x r ---

    #[test]
    fn velocity_at_point_matches_omega_cross_r() {
        let mut body = RigidBody::new_sphere(1.0, 1.0);
        body.angular_velocity = Vec3::new(0.0, 0.0, 5.0);
        body.velocity = Vec3::new(1.0, 0.0, 0.0);

        let point = Vec3::new(2.0, 0.0, 0.0);
        let v_point = body.velocity_at_point(point);

        let r = point - body.position;
        let expected = body.velocity + body.angular_velocity.cross(&r);

        assert!(
            vec3_approx(v_point, expected, TOLERANCE),
            "velocity_at_point mismatch: got {v_point:?}, expected {expected:?}"
        );
    }

    // --- Collision detection ---

    #[test]
    fn sphere_collision_detection_overlap() {
        let a = RigidBody::new_sphere(1.0, 1.0);
        let mut b = RigidBody::new_sphere(1.0, 1.0);
        b.position = Vec3::new(1.5, 0.0, 0.0);

        let result = sphere_sphere_collision(&a, 1.0, &b, 1.0);
        assert!(result.is_some());
        let (normal, penetration) = result.unwrap();
        assert!(approx(penetration, 0.5));
        assert!(vec3_approx(normal, Vec3::new(1.0, 0.0, 0.0), TOLERANCE));
    }

    #[test]
    fn sphere_collision_detection_no_overlap() {
        let a = RigidBody::new_sphere(1.0, 1.0);
        let mut b = RigidBody::new_sphere(1.0, 1.0);
        b.position = Vec3::new(3.0, 0.0, 0.0);

        let result = sphere_sphere_collision(&a, 1.0, &b, 1.0);
        assert!(result.is_none());
    }

    // --- Angular momentum conservation (torque-free) ---

    #[test]
    fn torque_free_angular_momentum_conserved() {
        let mut body = RigidBody::new(5.0, [2.0, 3.0, 4.0]);
        body.angular_velocity = Vec3::new(1.0, 0.5, 0.3);

        let l_initial = body.angular_momentum();

        for _ in 0..50_000 {
            body.step(1e-4);
        }

        let l_final = body.angular_momentum();
        assert!(
            vec3_approx(l_initial, l_final, 1e-2),
            "angular momentum not conserved: initial={l_initial:?}, final={l_final:?}"
        );
    }

    // --- Constructor inertia values ---

    #[test]
    fn sphere_inertia_correct() {
        let body = RigidBody::new_sphere(10.0, 2.0);
        let expected = SPHERE_INERTIA_FACTOR * 10.0 * 4.0;
        assert!(approx(body.inertia[0], expected));
        assert!(approx(body.inertia[1], expected));
        assert!(approx(body.inertia[2], expected));
    }

    #[test]
    fn box_inertia_correct() {
        let body = RigidBody::new_box(12.0, 2.0, 3.0, 4.0);
        let ix = 12.0 * (9.0 + 16.0) / BOX_INERTIA_DENOM;
        let iy = 12.0 * (4.0 + 16.0) / BOX_INERTIA_DENOM;
        let iz = 12.0 * (4.0 + 9.0) / BOX_INERTIA_DENOM;
        assert!(approx(body.inertia[0], ix));
        assert!(approx(body.inertia[1], iy));
        assert!(approx(body.inertia[2], iz));
    }

    #[test]
    fn cylinder_inertia_correct() {
        let body = RigidBody::new_cylinder(6.0, 2.0, 5.0);
        let i_axial = CYLINDER_AXIAL_FACTOR * 6.0 * 4.0;
        let i_perp = 6.0 * (12.0 + 25.0) / BOX_INERTIA_DENOM;
        assert!(approx(body.inertia[2], i_axial));
        assert!(approx(body.inertia[0], i_perp));
        assert!(approx(body.inertia[1], i_perp));
    }

    // --- System-level energy conservation (no gravity, elastic) ---

    #[test]
    fn system_total_energy_conserved_no_gravity() {
        let mut sys = RigidBodySystem::new(Vec3::ZERO);

        let mut a = RigidBody::new_sphere(2.0, 1.0);
        a.velocity = Vec3::new(3.0, 0.0, 0.0);
        a.angular_velocity = Vec3::new(0.0, 0.0, 1.0);

        let mut b = RigidBody::new_sphere(2.0, 1.0);
        b.position = Vec3::new(10.0, 0.0, 0.0);
        b.velocity = Vec3::new(-3.0, 0.0, 0.0);

        let e_initial = a.kinetic_energy() + b.kinetic_energy();
        sys.add_body(a);
        sys.add_body(b);

        // Run until they get close enough to collide
        for _ in 0..100_000 {
            sys.step(1e-4);
        }

        // No collision applied, so total energy is just sum of KE
        let e_final = sys.total_energy();
        assert!(
            approx_rel(e_initial, e_final, 1e-6),
            "energy drifted without collision: initial={e_initial}, final={e_final}"
        );
    }

    // --- Apply force at point generates correct torque ---

    #[test]
    fn apply_force_at_point_generates_torque() {
        let mut body = RigidBody::new_sphere(1.0, 1.0);
        let force = Vec3::new(0.0, 1.0, 0.0);
        let point = Vec3::new(1.0, 0.0, 0.0);

        body.apply_force_at_point(force, point);

        // tau = r x F = (1,0,0) x (0,1,0) = (0,0,1)
        body.step(1.0);

        // After one second, angular velocity should be along z
        assert!(
            body.angular_velocity.z.abs() > TOLERANCE,
            "expected non-zero angular velocity around z"
        );
    }
}
