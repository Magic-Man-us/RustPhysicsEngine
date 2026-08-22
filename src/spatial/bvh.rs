//! Bounding volume hierarchy over axis-aligned boxes.
//!
//! Built top-down with binned surface-area-heuristic splits (12 bins;
//! Wald 2007), falling back to a median split when SAH finds no gain.
//! Leaves store index ranges into a permutation of the input.

use crate::math::Vec3;
use crate::spatial::distance::{closest_point_aabb, closest_point_triangle};
use crate::spatial::intersect::{aabb_aabb, ray_aabb, ray_triangle, RayHit};
use crate::spatial::primitives::{Aabb, Ray, Sphere, Triangle};

const SAH_BINS: usize = 12;
const LEAF_SIZE: usize = 4;

#[derive(Debug, Clone)]
struct BvhNode {
    aabb: Aabb,
    /// Child node indices, or `usize::MAX` marking a leaf.
    left: usize,
    right: usize,
    /// Leaf range into `indices`.
    start: usize,
    count: usize,
}

impl BvhNode {
    fn is_leaf(&self) -> bool {
        self.left == usize::MAX
    }
}

/// Binary BVH; query methods return primitive indices into the
/// original input slice.
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<BvhNode>,
    indices: Vec<usize>,
    bounds: Vec<Aabb>,
}

fn centroid(b: &Aabb) -> Vec3 {
    b.center()
}

impl Bvh {
    /// Builds over one AABB per primitive.
    ///
    /// # Panics
    /// Panics on empty input.
    #[must_use]
    pub fn build(bounds: &[Aabb]) -> Self {
        assert!(!bounds.is_empty(), "Bvh::build requires primitives");
        let mut bvh = Self {
            nodes: Vec::with_capacity(2 * bounds.len()),
            indices: (0..bounds.len()).collect(),
            bounds: bounds.to_vec(),
        };
        let root_range = (0usize, bounds.len());
        bvh.nodes.push(BvhNode {
            aabb: bvh.range_bounds(root_range.0, root_range.1),
            left: usize::MAX,
            right: usize::MAX,
            start: root_range.0,
            count: root_range.1,
        });
        bvh.split(0);
        bvh
    }

    /// Convenience: builds over triangle bounds.
    ///
    /// # Panics
    /// Panics on empty input.
    #[must_use]
    pub fn build_triangles(tris: &[Triangle]) -> Self {
        let bounds: Vec<Aabb> = tris
            .iter()
            .map(|t| Aabb::from_points(&[t.a, t.b, t.c]))
            .collect();
        Self::build(&bounds)
    }

    fn range_bounds(&self, start: usize, count: usize) -> Aabb {
        let mut aabb = self.bounds[self.indices[start]];
        for &i in &self.indices[start + 1..start + count] {
            aabb = aabb.union(&self.bounds[i]);
        }
        aabb
    }

    fn split(&mut self, node: usize) {
        let (start, count) = (self.nodes[node].start, self.nodes[node].count);
        if count <= LEAF_SIZE {
            return;
        }
        let parent_aabb = self.nodes[node].aabb;
        let ext = parent_aabb.max - parent_aabb.min;
        // Longest axis for binning.
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };
        let axis_min = [parent_aabb.min.x, parent_aabb.min.y, parent_aabb.min.z][axis];
        let axis_ext = [ext.x, ext.y, ext.z][axis];
        if axis_ext <= 0.0 {
            return; // all centroids coincide
        }
        let coord = |b: &Aabb| {
            let c = centroid(b);
            [c.x, c.y, c.z][axis]
        };

        // Bin primitives by centroid.
        let mut bin_counts = [0usize; SAH_BINS];
        let mut bin_bounds: [Option<Aabb>; SAH_BINS] = [None; SAH_BINS];
        for &i in &self.indices[start..start + count] {
            let t = ((coord(&self.bounds[i]) - axis_min) / axis_ext * SAH_BINS as f64) as usize;
            let bin = t.min(SAH_BINS - 1);
            bin_counts[bin] += 1;
            bin_bounds[bin] = Some(match bin_bounds[bin] {
                Some(b) => b.union(&self.bounds[i]),
                None => self.bounds[i],
            });
        }
        // Evaluate the SAH cost of each of the 11 split planes.
        let mut best_cost = f64::INFINITY;
        let mut best_split = usize::MAX;
        for split in 1..SAH_BINS {
            let (mut lb, mut rb): (Option<Aabb>, Option<Aabb>) = (None, None);
            let (mut lc, mut rc) = (0usize, 0usize);
            for b in 0..split {
                if let Some(bb) = bin_bounds[b] {
                    lb = Some(lb.map_or(bb, |x| x.union(&bb)));
                    lc += bin_counts[b];
                }
            }
            for b in split..SAH_BINS {
                if let Some(bb) = bin_bounds[b] {
                    rb = Some(rb.map_or(bb, |x| x.union(&bb)));
                    rc += bin_counts[b];
                }
            }
            if lc == 0 || rc == 0 {
                continue;
            }
            let cost = lb.map_or(0.0, |b| b.surface_area()) * lc as f64
                + rb.map_or(0.0, |b| b.surface_area()) * rc as f64;
            if cost < best_cost {
                best_cost = cost;
                best_split = split;
            }
        }
        // Partition indices (SAH plane, or median fallback).
        let mid = if best_split != usize::MAX {
            let threshold = axis_min + axis_ext * best_split as f64 / SAH_BINS as f64;
            let slice = &mut self.indices[start..start + count];
            let mut i = 0;
            let mut j = count;
            while i < j {
                if coord(&self.bounds[slice[i]]) < threshold {
                    i += 1;
                } else {
                    j -= 1;
                    slice.swap(i, j);
                }
            }
            if i == 0 || i == count {
                count / 2 // degenerate partition; use median
            } else {
                i
            }
        } else {
            count / 2
        };
        if mid == 0 || mid == count {
            return;
        }
        if best_split == usize::MAX || mid == count / 2 {
            // Median split needs sorted order along the axis.
            self.indices[start..start + count]
                .sort_by(|&a, &b| {
                    coord(&self.bounds[a])
                        .partial_cmp(&coord(&self.bounds[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
        }

        let left = self.nodes.len();
        self.nodes.push(BvhNode {
            aabb: self.range_bounds(start, mid),
            left: usize::MAX,
            right: usize::MAX,
            start,
            count: mid,
        });
        let right = self.nodes.len();
        self.nodes.push(BvhNode {
            aabb: self.range_bounds(start + mid, count - mid),
            left: usize::MAX,
            right: usize::MAX,
            start: start + mid,
            count: count - mid,
        });
        self.nodes[node].left = left;
        self.nodes[node].right = right;
        self.nodes[node].count = 0;
        self.split(left);
        self.split(right);
    }

    fn collect<F: Fn(&Aabb) -> bool>(&self, pred: &F) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if !pred(&node.aabb) {
                continue;
            }
            if node.is_leaf() {
                for &i in &self.indices[node.start..node.start + node.count] {
                    if pred(&self.bounds[i]) {
                        out.push(i);
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        out
    }

    /// Primitive indices whose bounds a ray may hit within `max_t`.
    #[must_use]
    pub fn query_ray(&self, r: &Ray, max_t: f64) -> Vec<usize> {
        self.collect(&|b: &Aabb| matches!(ray_aabb(r, b), Some((enter, _)) if enter <= max_t))
    }

    /// Primitive indices whose bounds overlap the box.
    #[must_use]
    pub fn query_aabb(&self, b: &Aabb) -> Vec<usize> {
        self.collect(&|n: &Aabb| aabb_aabb(n, b))
    }

    /// Primitive indices whose bounds overlap the sphere.
    #[must_use]
    pub fn query_sphere(&self, s: &Sphere) -> Vec<usize> {
        self.collect(&|n: &Aabb| crate::spatial::intersect::sphere_aabb(s, n))
    }

    /// Nearest ray-triangle hit over the indexed triangles.
    #[must_use]
    pub fn closest_hit(&self, r: &Ray, tris: &[Triangle]) -> Option<(usize, RayHit)> {
        let mut best: Option<(usize, RayHit)> = None;
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            match ray_aabb(r, &node.aabb) {
                None => continue,
                Some((enter, _)) => {
                    if let Some((_, ref hit)) = best {
                        if enter > hit.t {
                            continue;
                        }
                    }
                }
            }
            if node.is_leaf() {
                for &i in &self.indices[node.start..node.start + node.count] {
                    if let Some((hit, _)) = ray_triangle(r, &tris[i], false) {
                        if best.as_ref().is_none_or(|(_, b)| hit.t < b.t) {
                            best = Some((i, hit));
                        }
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        best
    }

    /// Closest point on any indexed triangle: (index, point, distance).
    #[must_use]
    pub fn closest_point(&self, p: Vec3, tris: &[Triangle]) -> (usize, Vec3, f64) {
        let mut best = (usize::MAX, Vec3::ZERO, f64::INFINITY);
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if closest_point_aabb(p, &node.aabb).distance_to(&p) >= best.2 {
                continue;
            }
            if node.is_leaf() {
                for &i in &self.indices[node.start..node.start + node.count] {
                    let q = closest_point_triangle(p, &tris[i]);
                    let d = q.distance_to(&p);
                    if d < best.2 {
                        best = (i, q, d);
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        best
    }

    /// Broadphase: all unordered primitive pairs (i < j) whose bounds
    /// overlap.
    #[must_use]
    pub fn self_overlaps(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        while let Some((a, b)) = stack.pop() {
            let (na, nb) = (&self.nodes[a], &self.nodes[b]);
            if !aabb_aabb(&na.aabb, &nb.aabb) {
                continue;
            }
            match (na.is_leaf(), nb.is_leaf()) {
                (true, true) => {
                    for &i in &self.indices[na.start..na.start + na.count] {
                        for &j in &self.indices[nb.start..nb.start + nb.count] {
                            if i != j && aabb_aabb(&self.bounds[i], &self.bounds[j]) {
                                out.push((i.min(j), i.max(j)));
                            }
                        }
                    }
                }
                (false, true) => {
                    stack.push((na.left, b));
                    stack.push((na.right, b));
                }
                (true, false) => {
                    stack.push((a, nb.left));
                    stack.push((a, nb.right));
                }
                (false, false) => {
                    if a == b {
                        stack.push((na.left, na.left));
                        stack.push((na.left, na.right));
                        stack.push((na.right, na.right));
                    } else {
                        stack.push((na.left, nb.left));
                        stack.push((na.left, nb.right));
                        stack.push((na.right, nb.left));
                        stack.push((na.right, nb.right));
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Updates leaf bounds in place (same topology) and recomputes
    /// internal boxes bottom-up.
    ///
    /// # Panics
    /// Panics unless `bounds.len()` matches the build-time count.
    pub fn refit(&mut self, bounds: &[Aabb]) {
        assert!(bounds.len() == self.bounds.len(), "refit requires the same primitive count");
        self.bounds.copy_from_slice(bounds);
        // Children always have larger indices than their parent, so a
        // reverse sweep sees children first.
        for n in (0..self.nodes.len()).rev() {
            if self.nodes[n].is_leaf() {
                self.nodes[n].aabb =
                    self.range_bounds(self.nodes[n].start, self.nodes[n].count);
            } else {
                let (l, r) = (self.nodes[n].left, self.nodes[n].right);
                self.nodes[n].aabb = self.nodes[l].aabb.union(&self.nodes[r].aabb);
            }
        }
    }

    /// Maximum node depth (root = 1).
    #[must_use]
    pub fn depth(&self) -> usize {
        fn walk(bvh: &Bvh, n: usize) -> usize {
            let node = &bvh.nodes[n];
            if node.is_leaf() {
                1
            } else {
                1 + walk(bvh, node.left).max(walk(bvh, node.right))
            }
        }
        walk(self, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn random_boxes(rng: &mut Rng, n: usize) -> Vec<Aabb> {
        (0..n)
            .map(|_| {
                let c = Vec3::new(
                    rng.next_f64() * 20.0 - 10.0,
                    rng.next_f64() * 20.0 - 10.0,
                    rng.next_f64() * 20.0 - 10.0,
                );
                let h = Vec3::new(
                    0.1 + rng.next_f64(),
                    0.1 + rng.next_f64(),
                    0.1 + rng.next_f64(),
                );
                Aabb { min: c - h, max: c + h }
            })
            .collect()
    }

    #[test]
    fn test_query_aabb_matches_brute_force() {
        let mut rng = Rng::new(301);
        let boxes = random_boxes(&mut rng, 500);
        let bvh = Bvh::build(&boxes);
        for _ in 0..30 {
            let probe = random_boxes(&mut rng, 1)[0];
            let mut fast = bvh.query_aabb(&probe);
            fast.sort_unstable();
            let brute: Vec<usize> = (0..boxes.len())
                .filter(|&i| aabb_aabb(&boxes[i], &probe))
                .collect();
            assert_eq!(fast, brute);
        }
    }

    #[test]
    fn test_query_ray_superset_of_hits() {
        let mut rng = Rng::new(302);
        let boxes = random_boxes(&mut rng, 300);
        let bvh = Bvh::build(&boxes);
        let r = Ray::new(Vec3::new(-50.0, 0.0, 0.0), Vec3::new(1.0, 0.01, 0.02));
        let mut fast = bvh.query_ray(&r, f64::INFINITY);
        fast.sort_unstable();
        let brute: Vec<usize> = (0..boxes.len())
            .filter(|&i| ray_aabb(&r, &boxes[i]).is_some())
            .collect();
        assert_eq!(fast, brute);
    }

    #[test]
    fn test_closest_hit_and_point_on_triangles() {
        let tris = vec![
            Triangle {
                a: Vec3::new(0.0, -1.0, -1.0),
                b: Vec3::new(0.0, 1.0, -1.0),
                c: Vec3::new(0.0, 0.0, 1.0),
            },
            Triangle {
                a: Vec3::new(3.0, -1.0, -1.0),
                b: Vec3::new(3.0, 1.0, -1.0),
                c: Vec3::new(3.0, 0.0, 1.0),
            },
        ];
        let bvh = Bvh::build_triangles(&tris);
        let r = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let (idx, hit) = bvh.closest_hit(&r, &tris).unwrap();
        assert_eq!(idx, 0);
        assert!((hit.t - 5.0).abs() < 1e-12);

        let (pi, point, d) = bvh.closest_point(Vec3::new(2.9, 0.0, 0.0), &tris);
        assert_eq!(pi, 1);
        assert!((d - 0.1).abs() < 1e-12);
        assert!((point.x - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_self_overlaps_matches_brute_force() {
        let mut rng = Rng::new(303);
        let boxes = random_boxes(&mut rng, 200);
        let bvh = Bvh::build(&boxes);
        let fast = bvh.self_overlaps();
        let mut brute = Vec::new();
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                if aabb_aabb(&boxes[i], &boxes[j]) {
                    brute.push((i, j));
                }
            }
        }
        assert_eq!(fast, brute);
    }

    #[test]
    fn test_refit_and_depth() {
        let mut rng = Rng::new(304);
        let boxes = random_boxes(&mut rng, 300);
        let mut bvh = Bvh::build(&boxes);
        assert!(bvh.depth() > 3, "depth {} too shallow for 300 prims", bvh.depth());
        // Shift every box and refit: queries stay exact.
        let shifted: Vec<Aabb> = boxes
            .iter()
            .map(|b| Aabb { min: b.min + Vec3::new(5.0, 0.0, 0.0), max: b.max + Vec3::new(5.0, 0.0, 0.0) })
            .collect();
        bvh.refit(&shifted);
        let probe = Aabb { min: Vec3::new(-2.0, -2.0, -2.0), max: Vec3::new(8.0, 8.0, 8.0) };
        let mut fast = bvh.query_aabb(&probe);
        fast.sort_unstable();
        let brute: Vec<usize> = (0..shifted.len())
            .filter(|&i| aabb_aabb(&shifted[i], &probe))
            .collect();
        assert_eq!(fast, brute);
    }
}
