//! k-d trees (3-D and 2-D) with median splits, plus a uniform spatial
//! hash for broadphase neighbor queries.
//!
//! Reference: Bentley 1975; Friedman, Bentley & Finkel 1977 (nearest
//! neighbor search with bounds pruning).

use std::collections::HashMap;

use crate::math::{Vec2, Vec3};

#[derive(Debug, Clone)]
struct KdNode {
    point: usize,
    left: usize,  // usize::MAX = none
    right: usize, // usize::MAX = none
    axis: usize,
}

const NONE: usize = usize::MAX;

macro_rules! kd_impl {
    ($name:ident, $vec:ty, $dims:expr, $get:expr) => {
        /// Median-split k-d tree over points.
        #[derive(Debug, Clone)]
        pub struct $name {
            points: Vec<$vec>,
            nodes: Vec<KdNode>,
            root: usize,
        }

        impl $name {
            /// Builds by recursive median split (O(n log² n)).
            ///
            /// # Panics
            /// Panics on empty input.
            #[must_use]
            pub fn build(points: &[$vec]) -> Self {
                assert!(!points.is_empty(), "KdTree::build requires points");
                let mut tree = Self {
                    points: points.to_vec(),
                    nodes: Vec::with_capacity(points.len()),
                    root: NONE,
                };
                let mut idx: Vec<usize> = (0..points.len()).collect();
                tree.root = tree.build_rec(&mut idx, 0);
                tree
            }

            fn build_rec(&mut self, idx: &mut [usize], depth: usize) -> usize {
                if idx.is_empty() {
                    return NONE;
                }
                let axis = depth % $dims;
                idx.sort_by(|&a, &b| {
                    let ga: fn(&$vec, usize) -> f64 = $get;
                    ga(&self.points[a], axis)
                        .partial_cmp(&ga(&self.points[b], axis))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mid = idx.len() / 2;
                let point = idx[mid];
                let node_idx = self.nodes.len();
                self.nodes.push(KdNode { point, left: NONE, right: NONE, axis });
                let (lo, rest) = idx.split_at_mut(mid);
                let hi = &mut rest[1..];
                let left = self.build_rec(lo, depth + 1);
                let right = self.build_rec(hi, depth + 1);
                self.nodes[node_idx].left = left;
                self.nodes[node_idx].right = right;
                node_idx
            }

            fn dist(&self, i: usize, p: &$vec) -> f64 {
                let ga: fn(&$vec, usize) -> f64 = $get;
                (0..$dims)
                    .map(|d| {
                        let delta = ga(&self.points[i], d) - ga(p, d);
                        delta * delta
                    })
                    .sum::<f64>()
                    .sqrt()
            }

            fn search<F: FnMut(usize, f64) -> f64>(&self, p: &$vec, visit: &mut F) {
                // `visit` returns the current pruning radius.
                fn rec<F: FnMut(usize, f64) -> f64>(
                    tree: &$name,
                    n: usize,
                    p: &$vec,
                    visit: &mut F,
                    mut radius: f64,
                ) -> f64 {
                    if n == NONE {
                        return radius;
                    }
                    let node = &tree.nodes[n];
                    let d = tree.dist(node.point, p);
                    radius = visit(node.point, d);
                    let ga: fn(&$vec, usize) -> f64 = $get;
                    let delta = ga(p, node.axis) - ga(&tree.points[node.point], node.axis);
                    let (near, far) = if delta < 0.0 {
                        (node.left, node.right)
                    } else {
                        (node.right, node.left)
                    };
                    radius = rec(tree, near, p, visit, radius);
                    if delta.abs() <= radius {
                        radius = rec(tree, far, p, visit, radius);
                    }
                    radius
                }
                rec(self, self.root, p, visit, f64::INFINITY);
            }

            /// Nearest stored point: (index, distance).
            #[must_use]
            pub fn nearest(&self, p: $vec) -> Option<(usize, f64)> {
                let mut best: Option<(usize, f64)> = None;
                self.search(&p, &mut |i, d| {
                    if best.is_none() || d < best.unwrap().1 {
                        best = Some((i, d));
                    }
                    best.map_or(f64::INFINITY, |(_, bd)| bd)
                });
                best
            }

            /// The k nearest points sorted by ascending distance.
            #[must_use]
            pub fn k_nearest(&self, p: $vec, k: usize) -> Vec<(usize, f64)> {
                if k == 0 {
                    return Vec::new();
                }
                let mut heap: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
                self.search(&p, &mut |i, d| {
                    heap.push((i, d));
                    heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    heap.truncate(k);
                    if heap.len() == k {
                        heap[k - 1].1
                    } else {
                        f64::INFINITY
                    }
                });
                heap
            }

            /// All points within radius r, unsorted: (index, distance).
            #[must_use]
            pub fn within_radius(&self, p: $vec, r: f64) -> Vec<(usize, f64)> {
                let mut out = Vec::new();
                self.search(&p, &mut |i, d| {
                    if d <= r {
                        out.push((i, d));
                    }
                    r
                });
                out
            }

            /// All unordered pairs (i < j) within distance r.
            #[must_use]
            pub fn all_pairs_within(&self, r: f64) -> Vec<(usize, usize)> {
                let mut out = Vec::new();
                for i in 0..self.points.len() {
                    for (j, _) in self.within_radius(self.points[i], r) {
                        if i < j {
                            out.push((i, j));
                        }
                    }
                }
                out.sort_unstable();
                out
            }
        }
    };
}

fn get3(v: &Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn get2(v: &Vec2, axis: usize) -> f64 {
    if axis == 0 {
        v.x
    } else {
        v.y
    }
}

kd_impl!(KdTree, Vec3, 3, get3);
kd_impl!(KdTree2, Vec2, 2, get2);

/// Uniform-grid spatial hash for 3-D points; O(1) insert, sphere
/// queries visit only overlapping cells.
#[derive(Debug, Clone, Default)]
pub struct SpatialHash {
    pub cell: f64,
    pub map: HashMap<(i64, i64, i64), Vec<usize>>,
    positions: HashMap<usize, Vec3>,
}

impl SpatialHash {
    /// # Panics
    /// Panics unless cell > 0.
    #[must_use]
    pub fn new(cell: f64) -> Self {
        assert!(cell > 0.0, "SpatialHash requires cell > 0");
        Self { cell, map: HashMap::new(), positions: HashMap::new() }
    }

    fn key(&self, p: Vec3) -> (i64, i64, i64) {
        (
            (p.x / self.cell).floor() as i64,
            (p.y / self.cell).floor() as i64,
            (p.z / self.cell).floor() as i64,
        )
    }

    /// Registers item i at position p.
    pub fn insert(&mut self, i: usize, p: Vec3) {
        self.map.entry(self.key(p)).or_default().push(i);
        self.positions.insert(i, p);
    }

    /// Indices of items within r of p (exact, sorted).
    #[must_use]
    pub fn query_sphere(&self, p: Vec3, r: f64) -> Vec<usize> {
        let lo = self.key(p - Vec3::new(r, r, r));
        let hi = self.key(p + Vec3::new(r, r, r));
        let mut out = Vec::new();
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    if let Some(items) = self.map.get(&(x, y, z)) {
                        for &i in items {
                            if self.positions[&i].distance_to(&p) <= r {
                                out.push(i);
                            }
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Removes all items.
    pub fn clear(&mut self) {
        self.map.clear();
        self.positions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn random_points3(rng: &mut Rng, n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|_| {
                Vec3::new(
                    rng.next_f64() * 10.0 - 5.0,
                    rng.next_f64() * 10.0 - 5.0,
                    rng.next_f64() * 10.0 - 5.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_nearest_matches_brute_force() {
        let mut rng = Rng::new(321);
        let pts = random_points3(&mut rng, 1000);
        let tree = KdTree::build(&pts);
        for _ in 0..50 {
            let q = random_points3(&mut rng, 1)[0];
            let (i, d) = tree.nearest(q).unwrap();
            let (bi, bd) = pts
                .iter()
                .enumerate()
                .map(|(j, p)| (j, p.distance_to(&q)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            assert!((d - bd).abs() < 1e-12, "kd {d} vs brute {bd}");
            assert!(d <= pts[bi].distance_to(&q) + 1e-12);
            let _ = i;
        }
    }

    #[test]
    fn test_k_nearest_sorted_and_correct() {
        let mut rng = Rng::new(322);
        let pts = random_points3(&mut rng, 500);
        let tree = KdTree::build(&pts);
        for _ in 0..20 {
            let q = random_points3(&mut rng, 1)[0];
            let k = 7;
            let knn = tree.k_nearest(q, k);
            assert_eq!(knn.len(), k);
            for w in knn.windows(2) {
                assert!(w[0].1 <= w[1].1, "k_nearest not ascending");
            }
            let mut brute: Vec<f64> = pts.iter().map(|p| p.distance_to(&q)).collect();
            brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, &(_, d)) in knn.iter().enumerate() {
                assert!((d - brute[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_k_nearest_index_set_and_edge_cases() {
        let mut rng = Rng::new(327);
        let pts = random_points3(&mut rng, 300);
        let tree = KdTree::build(&pts);
        for _ in 0..20 {
            let q = random_points3(&mut rng, 1)[0];
            let k = 5;
            let knn = tree.k_nearest(q, k);
            // The returned indices are exactly the k brute-force nearest.
            let mut order: Vec<usize> = (0..pts.len()).collect();
            order.sort_by(|&a, &b| {
                pts[a].distance_to(&q).partial_cmp(&pts[b].distance_to(&q)).unwrap()
            });
            let mut brute: Vec<usize> = order[..k].to_vec();
            brute.sort_unstable();
            let mut fast: Vec<usize> = knn.iter().map(|&(i, _)| i).collect();
            fast.sort_unstable();
            assert_eq!(fast, brute);
            // Each reported distance is the true distance to that point,
            // and no excluded point is nearer than the k-th.
            let kth = knn[k - 1].1;
            for &(i, d) in &knn {
                assert!((d - pts[i].distance_to(&q)).abs() < 1e-12);
            }
            for i in 0..pts.len() {
                if !fast.contains(&i) {
                    assert!(pts[i].distance_to(&q) >= kth - 1e-12);
                }
            }
            // k = 1 agrees with `nearest`.
            let one = tree.k_nearest(q, 1);
            assert_eq!(one.len(), 1);
            assert!((one[0].1 - tree.nearest(q).unwrap().1).abs() < 1e-12);
        }
        // k = 0 returns nothing; k ≥ n returns every point, sorted.
        let q = random_points3(&mut rng, 1)[0];
        assert!(tree.k_nearest(q, 0).is_empty());
        let all = tree.k_nearest(q, pts.len() + 10);
        assert_eq!(all.len(), pts.len());
        let mut idx: Vec<usize> = all.iter().map(|&(i, _)| i).collect();
        idx.sort_unstable();
        assert_eq!(idx, (0..pts.len()).collect::<Vec<_>>());
        for w in all.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_within_radius_distances_and_degenerate_radii() {
        let mut rng = Rng::new(328);
        let pts = random_points3(&mut rng, 300);
        let tree = KdTree::build(&pts);
        for _ in 0..15 {
            let q = random_points3(&mut rng, 1)[0];
            let r = 0.5 + rng.next_f64() * 2.0;
            let hits = tree.within_radius(q, r);
            // Reported distances are true distances and inside the ball.
            for &(i, d) in &hits {
                assert!((d - pts[i].distance_to(&q)).abs() < 1e-12);
                assert!(d <= r);
            }
            // Nothing inside the ball is missed.
            let inside: Vec<usize> =
                (0..pts.len()).filter(|&i| pts[i].distance_to(&q) <= r).collect();
            let mut got: Vec<usize> = hits.iter().map(|&(i, _)| i).collect();
            got.sort_unstable();
            assert_eq!(got, inside);
            // Monotone in r: a larger radius is a superset.
            let wide: Vec<usize> =
                tree.within_radius(q, r * 2.0).into_iter().map(|(i, _)| i).collect();
            assert!(got.iter().all(|i| wide.contains(i)));
        }
        // Zero radius centered on a stored point finds that point only
        // (plus any exact duplicates), at distance 0.
        let hits = tree.within_radius(pts[42], 0.0);
        assert!(hits.iter().any(|&(i, _)| i == 42));
        for &(i, d) in &hits {
            assert_eq!(d, 0.0);
            assert_eq!(pts[i], pts[42]);
        }
        // A radius beyond the point cloud's diameter returns everything.
        let mut every: Vec<usize> =
            tree.within_radius(Vec3::ZERO, 1e6).into_iter().map(|(i, _)| i).collect();
        every.sort_unstable();
        assert_eq!(every, (0..pts.len()).collect::<Vec<_>>());
    }

    #[test]
    fn test_all_pairs_within_ordering_and_limits() {
        let mut rng = Rng::new(329);
        let pts = random_points3(&mut rng, 120);
        let tree = KdTree::build(&pts);
        for &r in &[0.5_f64, 1.5, 3.0] {
            let pairs = tree.all_pairs_within(r);
            let mut brute = Vec::new();
            for i in 0..pts.len() {
                for j in (i + 1)..pts.len() {
                    if pts[i].distance_to(&pts[j]) <= r {
                        brute.push((i, j));
                    }
                }
            }
            assert_eq!(pairs, brute, "r = {r}");
            // Output is strictly increasing (sorted, no duplicates) and
            // every pair is reported with i < j.
            for w in pairs.windows(2) {
                assert!(w[0] < w[1], "pairs not strictly sorted");
            }
            assert!(pairs.iter().all(|&(i, j)| i < j));
        }
        // Zero radius over distinct points yields no pairs; a radius past
        // the diameter yields all n(n−1)/2 of them.
        assert!(tree.all_pairs_within(0.0).is_empty());
        let n = pts.len();
        assert_eq!(tree.all_pairs_within(1e6).len(), n * (n - 1) / 2);
    }

    #[test]
    fn test_within_radius_matches_brute_force() {
        let mut rng = Rng::new(323);
        let pts = random_points3(&mut rng, 800);
        let tree = KdTree::build(&pts);
        for _ in 0..20 {
            let q = random_points3(&mut rng, 1)[0];
            let r = rng.next_f64() * 2.0;
            let mut fast: Vec<usize> =
                tree.within_radius(q, r).into_iter().map(|(i, _)| i).collect();
            fast.sort_unstable();
            let brute: Vec<usize> = (0..pts.len())
                .filter(|&i| pts[i].distance_to(&q) <= r)
                .collect();
            assert_eq!(fast, brute);
        }
    }

    #[test]
    fn test_all_pairs_within_matches_brute_force() {
        let mut rng = Rng::new(324);
        let pts = random_points3(&mut rng, 200);
        let tree = KdTree::build(&pts);
        let r = 1.0;
        let fast = tree.all_pairs_within(r);
        let mut brute = Vec::new();
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                if pts[i].distance_to(&pts[j]) <= r {
                    brute.push((i, j));
                }
            }
        }
        assert_eq!(fast, brute);
    }

    #[test]
    fn test_kdtree2() {
        let mut rng = Rng::new(325);
        let pts: Vec<Vec2> = (0..500)
            .map(|_| Vec2::new(rng.next_f64() * 10.0 - 5.0, rng.next_f64() * 10.0 - 5.0))
            .collect();
        let tree = KdTree2::build(&pts);
        for _ in 0..20 {
            let q = Vec2::new(rng.next_f64() * 10.0 - 5.0, rng.next_f64() * 10.0 - 5.0);
            let (_, d) = tree.nearest(q).unwrap();
            let bd = pts
                .iter()
                .map(|p| p.distance_to(&q))
                .fold(f64::INFINITY, f64::min);
            assert!((d - bd).abs() < 1e-12);
        }
    }

    #[test]
    fn test_spatial_hash_matches_brute_force() {
        let mut rng = Rng::new(326);
        let pts = random_points3(&mut rng, 600);
        let mut hash = SpatialHash::new(0.75);
        for (i, &p) in pts.iter().enumerate() {
            hash.insert(i, p);
        }
        for _ in 0..30 {
            let q = random_points3(&mut rng, 1)[0];
            let r = rng.next_f64() * 2.0;
            let fast = hash.query_sphere(q, r);
            let brute: Vec<usize> = (0..pts.len())
                .filter(|&i| pts[i].distance_to(&q) <= r)
                .collect();
            assert_eq!(fast, brute);
        }
        hash.clear();
        assert!(hash.query_sphere(Vec3::ZERO, 100.0).is_empty());
    }
}
