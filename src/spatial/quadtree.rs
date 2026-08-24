//! Point quadtree with bucket capacity and depth limit.

use crate::math::Vec2;
use crate::spatial::intersect::rect_rect;
use crate::spatial::primitives::{Circle, Rect};

#[derive(Debug, Clone)]
struct QtNode<T: Copy> {
    bounds: Rect,
    items: Vec<(Vec2, T)>,
    /// Index of the first of four children, or `usize::MAX` for leaves.
    children: usize,
    depth: usize,
}

/// Quadtree over 2-D points carrying a payload per point.
#[derive(Debug, Clone)]
pub struct Quadtree<T: Copy> {
    nodes: Vec<QtNode<T>>,
    capacity: usize,
    max_depth: usize,
    len: usize,
}

impl<T: Copy> Quadtree<T> {
    /// # Panics
    /// Panics unless capacity ≥ 1.
    #[must_use]
    pub fn new(bounds: Rect, capacity: usize, max_depth: usize) -> Self {
        assert!(capacity >= 1, "Quadtree requires capacity >= 1");
        Self {
            nodes: vec![QtNode { bounds, items: Vec::new(), children: usize::MAX, depth: 0 }],
            capacity,
            max_depth,
            len: 0,
        }
    }

    fn subdivide(&mut self, n: usize) {
        let b = self.nodes[n].bounds;
        let c = b.center();
        let depth = self.nodes[n].depth + 1;
        let first = self.nodes.len();
        let quads = [
            Rect { min: b.min, max: c },
            Rect { min: Vec2::new(c.x, b.min.y), max: Vec2::new(b.max.x, c.y) },
            Rect { min: Vec2::new(b.min.x, c.y), max: Vec2::new(c.x, b.max.y) },
            Rect { min: c, max: b.max },
        ];
        for q in quads {
            self.nodes.push(QtNode { bounds: q, items: Vec::new(), children: usize::MAX, depth });
        }
        self.nodes[n].children = first;
        // Push existing items down.
        let items = std::mem::take(&mut self.nodes[n].items);
        for (p, item) in items {
            let child = first + Self::quadrant(&c, p);
            self.nodes[child].items.push((p, item));
        }
    }

    fn quadrant(center: &Vec2, p: Vec2) -> usize {
        (if p.x >= center.x { 1 } else { 0 }) + (if p.y >= center.y { 2 } else { 0 })
    }

    /// Inserts a point; returns false when it lies outside the root
    /// bounds.
    pub fn insert(&mut self, p: Vec2, item: T) -> bool {
        if !self.nodes[0].bounds.contains_point(p) {
            return false;
        }
        let mut n = 0usize;
        loop {
            if self.nodes[n].children != usize::MAX {
                let c = self.nodes[n].bounds.center();
                n = self.nodes[n].children + Self::quadrant(&c, p);
                continue;
            }
            if self.nodes[n].items.len() < self.capacity || self.nodes[n].depth >= self.max_depth {
                self.nodes[n].items.push((p, item));
                self.len += 1;
                return true;
            }
            self.subdivide(n);
        }
    }

    fn query<F: Fn(&Rect) -> bool, G: Fn(Vec2) -> bool>(
        &self,
        node_pred: F,
        point_pred: G,
    ) -> Vec<(Vec2, T)> {
        let mut out = Vec::new();
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if !node_pred(&node.bounds) {
                continue;
            }
            for &(p, item) in &node.items {
                if point_pred(p) {
                    out.push((p, item));
                }
            }
            if node.children != usize::MAX {
                for k in 0..4 {
                    stack.push(node.children + k);
                }
            }
        }
        out
    }

    /// All points inside the rectangle (closed).
    #[must_use]
    pub fn query_rect(&self, r: &Rect) -> Vec<(Vec2, T)> {
        self.query(|b| rect_rect(b, r), |p| r.contains_point(p))
    }

    /// All points inside the circle (closed).
    #[must_use]
    pub fn query_circle(&self, c: &Circle) -> Vec<(Vec2, T)> {
        let bounds = Rect {
            min: c.center - Vec2::new(c.radius, c.radius),
            max: c.center + Vec2::new(c.radius, c.radius),
        };
        self.query(
            |b| rect_rect(b, &bounds),
            |p| p.distance_to(&c.center) <= c.radius,
        )
    }

    /// Nearest stored point to p with its payload and distance.
    #[must_use]
    pub fn nearest(&self, p: Vec2) -> Option<(Vec2, T, f64)> {
        let mut best: Option<(Vec2, T, f64)> = None;
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            // Prune by rectangle distance.
            let b = &node.bounds;
            let dx = (b.min.x - p.x).max(p.x - b.max.x).max(0.0);
            let dy = (b.min.y - p.y).max(p.y - b.max.y).max(0.0);
            let rect_dist = (dx * dx + dy * dy).sqrt();
            if let Some((_, _, bd)) = best {
                if rect_dist >= bd {
                    continue;
                }
            }
            for &(q, item) in &node.items {
                let d = q.distance_to(&p);
                if best.is_none() || d < best.unwrap().2 {
                    best = Some((q, item, d));
                }
            }
            if node.children != usize::MAX {
                for k in 0..4 {
                    stack.push(node.children + k);
                }
            }
        }
        best
    }

    /// Number of stored points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no points are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn build_random(rng: &mut Rng, n: usize) -> (Quadtree<usize>, Vec<Vec2>) {
        let bounds = Rect { min: Vec2::new(-10.0, -10.0), max: Vec2::new(10.0, 10.0) };
        let mut qt = Quadtree::new(bounds, 8, 12);
        let pts: Vec<Vec2> = (0..n)
            .map(|_| Vec2::new(rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 20.0 - 10.0))
            .collect();
        for (i, &p) in pts.iter().enumerate() {
            assert!(qt.insert(p, i));
        }
        (qt, pts)
    }

    #[test]
    fn test_insert_bounds_and_len() {
        let bounds = Rect { min: Vec2::ZERO, max: Vec2::new(1.0, 1.0) };
        let mut qt = Quadtree::new(bounds, 2, 8);
        assert!(qt.insert(Vec2::new(0.5, 0.5), 1));
        assert!(!qt.insert(Vec2::new(2.0, 0.5), 2));
        assert_eq!(qt.len(), 1);
        assert!(!qt.is_empty());
    }

    #[test]
    fn test_query_rect_matches_brute_force() {
        let mut rng = Rng::new(311);
        let (qt, pts) = build_random(&mut rng, 1000);
        for _ in 0..20 {
            let a = Vec2::new(rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 20.0 - 10.0);
            let b = a + Vec2::new(rng.next_f64() * 5.0, rng.next_f64() * 5.0);
            let r = Rect { min: a, max: b };
            let mut fast: Vec<usize> = qt.query_rect(&r).into_iter().map(|(_, i)| i).collect();
            fast.sort_unstable();
            let brute: Vec<usize> =
                (0..pts.len()).filter(|&i| r.contains_point(pts[i])).collect();
            assert_eq!(fast, brute);
        }
    }

    #[test]
    fn test_query_circle_and_nearest_match_brute_force() {
        let mut rng = Rng::new(312);
        let (qt, pts) = build_random(&mut rng, 1000);
        for _ in 0..20 {
            let c = Circle {
                center: Vec2::new(rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 20.0 - 10.0),
                radius: rng.next_f64() * 4.0,
            };
            let mut fast: Vec<usize> = qt.query_circle(&c).into_iter().map(|(_, i)| i).collect();
            fast.sort_unstable();
            let brute: Vec<usize> = (0..pts.len())
                .filter(|&i| pts[i].distance_to(&c.center) <= c.radius)
                .collect();
            assert_eq!(fast, brute);

            let (_, _, nd) = qt.nearest(c.center).unwrap();
            let brute_nd = pts
                .iter()
                .map(|p| p.distance_to(&c.center))
                .fold(f64::INFINITY, f64::min);
            assert!((nd - brute_nd).abs() < 1e-12);
        }
    }
}
