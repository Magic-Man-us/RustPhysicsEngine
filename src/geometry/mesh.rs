//! Minimal indexed triangle mesh with ray intersection, backfilling the
//! Part 2 `Mesh` type consumed by acoustics ray tracing and display
//! helpers.

use crate::math::Vec3;

/// Indexed triangle mesh. `materials[i]` is a per-triangle material index
/// (into caller-owned tables such as absorption coefficients).
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[usize; 3]>,
    pub materials: Vec<usize>,
}

/// A ray/mesh intersection.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub t: f64,
    pub point: Vec3,
    pub normal: Vec3,
    pub triangle: usize,
}

impl Mesh {
    /// Empty mesh.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Axis-aligned box (e.g. a shoebox room) spanning from the origin to
    /// `size`, with inward-facing triangles and material indices 0..5 in
    /// the wall order -x, +x, -y, +y, -z, +z.
    #[must_use]
    pub fn box_room(size: Vec3) -> Self {
        let (sx, sy, sz) = (size.x, size.y, size.z);
        let v = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
        let vertices = vec![
            v(0.0, 0.0, 0.0),
            v(sx, 0.0, 0.0),
            v(0.0, sy, 0.0),
            v(sx, sy, 0.0),
            v(0.0, 0.0, sz),
            v(sx, 0.0, sz),
            v(0.0, sy, sz),
            v(sx, sy, sz),
        ];
        // Two triangles per face, material index per wall pair.
        let faces: [([usize; 4], usize); 6] = [
            ([0, 2, 6, 4], 0), // -x
            ([1, 5, 7, 3], 1), // +x
            ([0, 4, 5, 1], 2), // -y
            ([2, 3, 7, 6], 3), // +y
            ([0, 1, 3, 2], 4), // -z
            ([4, 6, 7, 5], 5), // +z
        ];
        let mut triangles = Vec::new();
        let mut materials = Vec::new();
        for (quad, m) in faces {
            triangles.push([quad[0], quad[1], quad[2]]);
            triangles.push([quad[0], quad[2], quad[3]]);
            materials.push(m);
            materials.push(m);
        }
        Self { vertices, triangles, materials }
    }

    /// Geometric normal of triangle `i` (right-hand rule, unnormalized
    /// winding as stored).
    #[must_use]
    pub fn triangle_normal(&self, i: usize) -> Vec3 {
        let [a, b, c] = self.triangles[i];
        let ab = self.vertices[b] - self.vertices[a];
        let ac = self.vertices[c] - self.vertices[a];
        ab.cross(&ac).normalized()
    }

    /// Nearest ray intersection (Möller-Trumbore) with t > `t_min`.
    #[must_use]
    pub fn intersect_ray(&self, origin: Vec3, dir: Vec3, t_min: f64) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        for (i, tri) in self.triangles.iter().enumerate() {
            let [ia, ib, ic] = *tri;
            let a = self.vertices[ia];
            let e1 = self.vertices[ib] - a;
            let e2 = self.vertices[ic] - a;
            let p = dir.cross(&e2);
            let det = e1.dot(&p);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let s = origin - a;
            let u = s.dot(&p) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(&e1);
            let v = dir.dot(&q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = e2.dot(&q) * inv;
            if t > t_min && best.as_ref().is_none_or(|h| t < h.t) {
                let normal = self.triangle_normal(i);
                best = Some(RayHit {
                    t,
                    point: origin + dir * t,
                    normal,
                    triangle: i,
                });
            }
        }
        best
    }

    /// True if the straight segment between two points is unobstructed.
    #[must_use]
    pub fn segment_clear(&self, a: Vec3, b: Vec3) -> bool {
        let d = b - a;
        let len = d.magnitude();
        if len < 1e-12 {
            return true;
        }
        let dir = d * (1.0 / len);
        match self.intersect_ray(a, dir, 1e-9) {
            Some(h) => h.t >= len - 1e-9,
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_room_intersection() {
        let room = Mesh::box_room(Vec3::new(4.0, 3.0, 2.5));
        assert_eq!(room.triangles.len(), 12);
        let origin = Vec3::new(2.0, 1.5, 1.25);
        // Ray toward +x wall hits at distance 2.
        let hit = room.intersect_ray(origin, Vec3::new(1.0, 0.0, 0.0), 1e-9).unwrap();
        assert!((hit.t - 2.0).abs() < 1e-12);
        assert_eq!(room.materials[hit.triangle], 1);
        // Inward normal points back toward the room interior.
        assert!(hit.normal.x < 0.0);
        // Segment between two interior points is clear; segment to a
        // point beyond the wall is not.
        assert!(room.segment_clear(origin, Vec3::new(3.9, 1.0, 1.0)));
        let outside = Vec3::new(6.0, 1.5, 1.25);
        assert!(!room.segment_clear(origin, outside));
    }
}
