use crate::math::Vec3;
use crate::math::constants::G;
use crate::astrophysics::nbody::Body;

pub const BH_THETA: f64 = 0.5;
pub const BH_CROSSOVER: usize = 800;

const EMPTY: u32 = u32::MAX;

struct Node {
    com: Vec3,
    mass: f64,
    center: Vec3,
    half_size: f64,
    children: [u32; 8],
    body_idx: u32,
    count: u32,
}

impl Node {
    fn new(center: Vec3, half_size: f64) -> Self {
        Self {
            com: Vec3::ZERO,
            mass: 0.0,
            center,
            half_size,
            children: [EMPTY; 8],
            body_idx: EMPTY,
            count: 0,
        }
    }

    fn octant(&self, pos: &Vec3) -> usize {
        let mut idx = 0;
        if pos.x >= self.center.x { idx |= 1; }
        if pos.y >= self.center.y { idx |= 2; }
        if pos.z >= self.center.z { idx |= 4; }
        idx
    }

    fn child_center(&self, octant: usize) -> Vec3 {
        let q = self.half_size * 0.5;
        Vec3::new(
            self.center.x + if octant & 1 != 0 { q } else { -q },
            self.center.y + if octant & 2 != 0 { q } else { -q },
            self.center.z + if octant & 4 != 0 { q } else { -q },
        )
    }
}

pub struct Octree {
    nodes: Vec<Node>,
    root: u32,
}

impl Octree {
    /// Builds a Barnes-Hut octree from a set of bodies, computing bounding box and center-of-mass hierarchy.
    pub fn build(bodies: &[Body]) -> Self {
        if bodies.is_empty() {
            return Self { nodes: Vec::new(), root: EMPTY };
        }

        let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
        let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);
        for b in bodies {
            min_x = min_x.min(b.position.x);
            min_y = min_y.min(b.position.y);
            min_z = min_z.min(b.position.z);
            max_x = max_x.max(b.position.x);
            max_y = max_y.max(b.position.y);
            max_z = max_z.max(b.position.z);
        }

        let cx = (min_x + max_x) * 0.5;
        let cy = (min_y + max_y) * 0.5;
        let cz = (min_z + max_z) * 0.5;
        let half = ((max_x - min_x).max(max_y - min_y).max(max_z - min_z)) * 0.5 + 1e-6;

        let mut tree = Self {
            nodes: Vec::with_capacity(bodies.len() * 2),
            root: 0,
        };
        tree.nodes.push(Node::new(Vec3::new(cx, cy, cz), half));

        for (i, b) in bodies.iter().enumerate() {
            tree.insert(0, i as u32, b.position, b.mass);
        }

        tree
    }

    fn insert(&mut self, node_idx: u32, body_idx: u32, pos: Vec3, mass: f64) {
        let ni = node_idx as usize;

        // Update center of mass
        let total = self.nodes[ni].mass + mass;
        if total > 0.0 {
            let old_com = self.nodes[ni].com;
            self.nodes[ni].com = old_com * (self.nodes[ni].mass / total) + pos * (mass / total);
        }
        self.nodes[ni].mass = total;
        self.nodes[ni].count += 1;

        if self.nodes[ni].count == 1 {
            self.nodes[ni].body_idx = body_idx;
            return;
        }

        // If this was a leaf, push existing body down
        if self.nodes[ni].body_idx != EMPTY {
            let old_idx = self.nodes[ni].body_idx;
            let old_com = self.nodes[ni].com;
            // We need to reconstruct old position from com before this insert
            // Simpler: store and re-insert
            let old_mass = total - mass;
            let old_pos = if old_mass > 0.0 {
                (old_com * total - pos * mass) * (1.0 / old_mass)
            } else {
                old_com
            };
            self.nodes[ni].body_idx = EMPTY;
            self.push_down(node_idx, old_idx, old_pos, old_mass);
        }

        self.push_down(node_idx, body_idx, pos, mass);
    }

    fn push_down(&mut self, node_idx: u32, body_idx: u32, pos: Vec3, mass: f64) {
        let ni = node_idx as usize;
        let oct = self.nodes[ni].octant(&pos);

        if self.nodes[ni].children[oct] == EMPTY {
            let child_center = self.nodes[ni].child_center(oct);
            let child_half = self.nodes[ni].half_size * 0.5;
            let child_idx = self.nodes.len() as u32;
            self.nodes.push(Node::new(child_center, child_half));
            self.nodes[ni].children[oct] = child_idx;
        }

        let child = self.nodes[ni].children[oct];
        self.insert(child, body_idx, pos, mass);
    }

    /// Computes gravitational acceleration on body `idx` using the Barnes-Hut tree walk with opening angle θ.
    pub fn compute_acceleration(&self, bodies: &[Body], idx: usize, theta: f64, softening: f64) -> Vec3 {
        if self.root == EMPTY {
            return Vec3::ZERO;
        }
        let pos = bodies[idx].position;
        let theta_sq = theta * theta;
        let soft_sq = softening * softening;
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut az = 0.0_f64;
        self.walk(self.root, idx as u32, &pos, theta_sq, soft_sq, &mut ax, &mut ay, &mut az);
        Vec3::new(ax, ay, az)
    }

    fn walk(
        &self, node_idx: u32, body_idx: u32, pos: &Vec3,
        theta_sq: f64, soft_sq: f64,
        ax: &mut f64, ay: &mut f64, az: &mut f64,
    ) {
        let node = &self.nodes[node_idx as usize];

        if node.count == 0 || node.mass <= 0.0 { return; }
        if node.count == 1 && node.body_idx == body_idx { return; }

        let dx = node.com.x - pos.x;
        let dy = node.com.y - pos.y;
        let dz = node.com.z - pos.z;
        let dist_sq = dx * dx + dy * dy + dz * dz;

        let side = node.half_size * 2.0;
        let side_sq = side * side;

        if node.count == 1 || (dist_sq > 0.0 && side_sq < theta_sq * dist_sq) {
            let r_sq = dist_sq + soft_sq;
            let inv_dist3 = 1.0 / (r_sq * r_sq.sqrt());
            let f = G * node.mass * inv_dist3;
            *ax += f * dx;
            *ay += f * dy;
            *az += f * dz;
            return;
        }

        for &child in &node.children {
            if child != EMPTY {
                self.walk(child, body_idx, pos, theta_sq, soft_sq, ax, ay, az);
            }
        }
    }
}

/// Computes accelerations for all bodies, using direct summation below the crossover threshold or Barnes-Hut above it.
pub fn compute_all_accelerations(bodies: &[Body], theta: f64, softening: f64) -> Vec<Vec3> {
    if bodies.len() < BH_CROSSOVER {
        return (0..bodies.len())
            .map(|i| crate::astrophysics::nbody::compute_acceleration(bodies, i, softening))
            .collect();
    }

    let tree = Octree::build(bodies);
    (0..bodies.len())
        .map(|i| tree.compute_acceleration(bodies, i, theta, softening))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astrophysics::nbody;

    #[test]
    fn test_octree_vs_direct() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e24, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::ZERO),
            Body::new(2, 1.0e24, 1.0, Vec3::new(0.0, 1.0e11, 0.0), Vec3::ZERO),
        ];
        let softening = 0.05;

        let direct: Vec<Vec3> = (0..bodies.len())
            .map(|i| nbody::compute_acceleration(&bodies, i, softening))
            .collect();

        let tree = Octree::build(&bodies);
        let bh: Vec<Vec3> = (0..bodies.len())
            .map(|i| tree.compute_acceleration(&bodies, i, 0.0, softening))
            .collect();

        for i in 0..bodies.len() {
            let err = (direct[i] - bh[i]).magnitude() / direct[i].magnitude();
            assert!(err < 1e-10, "Body {i}: BH theta=0 should match direct, error = {err}");
        }
    }

    #[test]
    fn test_octree_builds_without_panic() {
        let mut bodies = Vec::new();
        for i in 0..100 {
            let angle = i as f64 * 0.1;
            bodies.push(Body::new(
                i, 1.0e20, 1.0,
                Vec3::new(angle.cos() * 1e10, angle.sin() * 1e10, 0.0),
                Vec3::ZERO,
            ));
        }
        let tree = Octree::build(&bodies);
        let acc = tree.compute_acceleration(&bodies, 0, BH_THETA, 0.05);
        assert!(acc.magnitude() > 0.0);
    }

    #[test]
    fn test_compute_all_accelerations_matches_direct() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Body::new(1, 1.0e24, 1.0, Vec3::new(1.0e11, 0.0, 0.0), Vec3::ZERO),
            Body::new(2, 1.0e24, 1.0, Vec3::new(0.0, 1.0e11, 0.0), Vec3::ZERO),
        ];
        let softening = 0.05;
        // Below BH_CROSSOVER, compute_all_accelerations uses direct summation
        let all_accs = compute_all_accelerations(&bodies, BH_THETA, softening);
        assert_eq!(all_accs.len(), bodies.len());
        for i in 0..bodies.len() {
            let direct = nbody::compute_acceleration(&bodies, i, softening);
            let err = (all_accs[i] - direct).magnitude();
            assert!(err < 1e-20, "Body {i}: compute_all_accelerations should match direct, error = {err}");
        }
    }

    #[test]
    fn test_compute_all_accelerations_empty() {
        let bodies: Vec<Body> = Vec::new();
        let accs = compute_all_accelerations(&bodies, BH_THETA, 0.05);
        assert!(accs.is_empty());
    }

    #[test]
    fn test_octree_build_empty() {
        let bodies: Vec<Body> = Vec::new();
        let tree = Octree::build(&bodies);
        let acc = tree.compute_acceleration(&[], 0, BH_THETA, 0.05);
        assert!((acc.x.abs() + acc.y.abs() + acc.z.abs()) < 1e-20);
    }

    #[test]
    fn test_octree_two_bodies_at_same_position() {
        let bodies = vec![
            Body::new(0, 1.0e30, 1.0, Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO),
            Body::new(1, 1.0e30, 1.0, Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO),
        ];
        let tree = Octree::build(&bodies);
        let acc = tree.compute_acceleration(&bodies, 0, BH_THETA, 0.05);
        assert!(acc.magnitude().is_finite());
    }

    #[test]
    fn test_octree_zero_mass_body_then_insert() {
        // Body 0 has mass=0 at (1,1,1). When body 1 (also mass=0) is inserted
        // into the same leaf, old_mass = 0, triggering the old_com fallback.
        // The com stays ZERO (update skipped since total=0), so old_pos = ZERO
        // which maps to octant 0. Body 1 at (1,1,1) maps to octant 7,
        // so they split into different children and avoid infinite recursion.
        let bodies = vec![
            Body::new(0, 0.0, 1.0, Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO),
            Body::new(1, 0.0, 1.0, Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO),
            Body::new(2, 1.0e30, 1.0, Vec3::new(-1.0, -1.0, -1.0), Vec3::ZERO),
        ];
        let tree = Octree::build(&bodies);
        let acc = tree.compute_acceleration(&bodies, 2, BH_THETA, 0.05);
        assert!(acc.magnitude().is_finite());
    }

    #[test]
    fn test_compute_all_accelerations_above_crossover() {
        let mut bodies = Vec::new();
        for i in 0..BH_CROSSOVER + 10 {
            let angle = i as f64 * 0.05;
            bodies.push(Body::new(
                i as u32, 1.0e20, 1.0,
                Vec3::new(angle.cos() * 1e10, angle.sin() * 1e10, 0.0),
                Vec3::ZERO,
            ));
        }
        let accs = compute_all_accelerations(&bodies, BH_THETA, 0.05);
        assert_eq!(accs.len(), bodies.len());
        assert!(accs[0].magnitude() > 0.0);
    }
}
