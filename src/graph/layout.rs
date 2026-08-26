//! Graph drawing: where to put the vertices.
//!
//! Two families. The *metric* layouts -- Kamada-Kawai, stress majorization,
//! Fruchterman-Reingold, spectral -- treat drawing as optimisation: pick a
//! target distance for every pair, usually the number of edges between them,
//! and place the points so the drawn distances match. What they optimise is
//! stated exactly, so what they achieve can be measured, which is why
//! [`layout_stress`] is public.
//!
//! The *structural* layouts -- circular, shell, Reingold-Tilford, Sugiyama --
//! draw a shape the graph already has. They are not approximating anything,
//! and their output satisfies exact statements: a tree drawn by
//! Reingold-Tilford has every parent centred over its children and no two
//! subtrees overlapping, and a layered drawing of an acyclic graph has every
//! arc pointing downward.
//!
//! Planarity sits apart from both: [`planarity_test`] answers whether a
//! crossing-free drawing exists at all, and [`planar_embedding_small`]
//! produces the combinatorial structure of one.

use crate::graph::core::Graph;
use crate::manifold::vecn::VecN;
use crate::math::Vec2;
use crate::monte_carlo::Rng;
use std::collections::{BTreeSet, VecDeque};
use std::f64::consts::TAU;

/// Hop distances between every pair, by breadth-first search from each
/// vertex.
///
/// Edge weights are deliberately ignored: a drawing is laid out by graph
/// structure, and a weight of a thousand on one edge should not stretch the
/// picture by a factor of a thousand. Pairs in different components are given
/// one more than the largest finite distance, which is the usual convention
/// and keeps the stress function finite.
#[must_use]
pub fn hop_distances(g: &Graph) -> Vec<Vec<f64>> {
    let n = g.n;
    let mut d = vec![vec![f64::INFINITY; n]; n];
    for s in 0..n {
        d[s][s] = 0.0;
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            for &(v, _) in &g.adj[u] {
                if d[s][v].is_infinite() {
                    d[s][v] = d[s][u] + 1.0;
                    q.push_back(v);
                }
            }
        }
    }
    let finite = d
        .iter()
        .flatten()
        .copied()
        .filter(|x| x.is_finite())
        .fold(0.0f64, f64::max);
    for row in &mut d {
        for x in row.iter_mut() {
            if x.is_infinite() {
                *x = finite + 1.0;
            }
        }
    }
    d
}

/// The stress of a two-dimensional drawing: the weighted squared mismatch
/// between drawn and graph distance, `sum_{i<j} (|p_i - p_j| - d_ij)^2 /
/// d_ij^2`.
///
/// This is the quantity [`kamada_kawai`] and [`stress_majorization`] exist to
/// reduce, so it is the honest way to compare two drawings of the same graph.
/// The `1/d^2` weighting is Kamada and Kawai's: without it the distant pairs,
/// of which there are many more, drown out the local structure the eye reads
/// first.
///
/// # Panics
/// Panics unless there is one position per vertex.
#[must_use]
pub fn layout_stress(g: &Graph, positions: &[Vec2]) -> f64 {
    assert_eq!(positions.len(), g.n, "one position per vertex is required");
    let pts: Vec<VecN> = positions.iter().map(|p| VecN::from(&[p.x, p.y])).collect();
    stress_nd(g, &pts)
}

/// The same stress functional for a drawing in any number of dimensions.
///
/// # Panics
/// Panics unless there is one position per vertex.
#[must_use]
pub fn stress_nd(g: &Graph, positions: &[VecN]) -> f64 {
    assert_eq!(positions.len(), g.n, "one position per vertex is required");
    let d = hop_distances(g);
    let mut total = 0.0;
    for i in 0..g.n {
        for j in i + 1..g.n {
            if d[i][j] <= 0.0 {
                continue;
            }
            let drawn = positions[i].sub(&positions[j]).norm();
            let e = drawn - d[i][j];
            total += e * e / (d[i][j] * d[i][j]);
        }
    }
    total
}

/// `n` points equally spaced around the unit circle, starting at `(1, 0)` and
/// going anticlockwise.
///
/// The one layout with no free parameters and nothing to converge. Every
/// vertex is visible and no two coincide, which is why it is the standard
/// starting point for the iterative layouts here.
#[must_use]
pub fn circular_layout(n: usize) -> Vec<Vec2> {
    (0..n)
        .map(|i| {
            let a = TAU * i as f64 / n.max(1) as f64;
            Vec2::new(a.cos(), a.sin())
        })
        .collect()
}

/// Concentric circles, one per shell, in the order given.
///
/// A shell holding a single vertex is drawn at the centre; every other shell
/// `k` goes on the circle of radius `k + 1`, its members equally spaced.
/// Useful when the grouping is already known -- levels of a hierarchy, orbits
/// of a symmetry, distance classes from a root.
///
/// # Panics
/// Panics unless the shells partition `0..g.n`.
#[must_use]
pub fn shell_layout(g: &Graph, shells: &[Vec<usize>]) -> Vec<Vec2> {
    let mut seen = vec![false; g.n];
    let mut count = 0;
    for s in shells {
        for &v in s {
            assert!(v < g.n, "vertex {v} is outside 0..{}", g.n);
            assert!(!seen[v], "vertex {v} appears in two shells");
            seen[v] = true;
            count += 1;
        }
    }
    assert_eq!(count, g.n, "the shells must cover every vertex");
    let mut pos = vec![Vec2::ZERO; g.n];
    for (k, shell) in shells.iter().enumerate() {
        let r = if shell.len() == 1 && k == 0 { 0.0 } else { k as f64 + 1.0 };
        for (i, &v) in shell.iter().enumerate() {
            let a = TAU * i as f64 / shell.len().max(1) as f64;
            pos[v] = Vec2::new(r * a.cos(), r * a.sin());
        }
    }
    pos
}

/// Spectral layout: the two Laplacian eigenvectors just above the constant
/// one, used as coordinates.
///
/// The constant vector is the Laplacian's zero eigenvector and carries no
/// information, so the drawing starts at the next two. Those minimise
/// `sum_edges |p_u - p_v|^2` subject to being centred and orthonormal, which
/// is to say they are the drawing that makes edges as short as possible
/// without collapsing everything to a point. Coordinates come out on the
/// scale of a unit vector; scale them for display.
///
/// # Panics
/// Panics if the graph is directed, or has fewer than three vertices.
#[must_use]
pub fn spectral_layout(g: &Graph) -> Vec<Vec2> {
    assert!(!g.directed, "the Laplacian here is for undirected graphs");
    assert!(g.n >= 3, "a spectral layout needs at least three vertices");
    let l = crate::graph::spectral::laplacian_matrix(g);
    let e = crate::linalg::eigen::eigen_symmetric(&l, 1e-12, 200)
        .expect("Jacobi converges on a symmetric matrix");
    // Descending order, so the two smallest above the constant one sit at
    // columns n - 2 and n - 3.
    let mut out = Vec::with_capacity(g.n);
    for v in 0..g.n {
        out.push(Vec2::new(e.vectors.get(v, g.n - 2), e.vectors.get(v, g.n - 3)));
    }
    out
}

/// Kamada-Kawai layout: move one vertex at a time to the position that best
/// matches its graph distances to everything else.
///
/// The energy is [`layout_stress`]. Each round picks the vertex whose
/// gradient is largest and takes a Newton step on its two coordinates, which
/// converges quadratically near the solution. The step is accepted only if
/// the energy actually falls, so the sequence of drawings is monotone: the
/// result is never worse than the circular layout it starts from. Newton on a
/// non-convex energy will otherwise happily step uphill.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn kamada_kawai(g: &Graph, iters: usize) -> Vec<Vec2> {
    assert!(!g.directed, "layouts here are for undirected graphs");
    let n = g.n;
    if n < 2 {
        return vec![Vec2::ZERO; n];
    }
    let d = hop_distances(g);
    let scale = d.iter().flatten().copied().fold(1.0f64, f64::max);
    let mut p: Vec<Vec2> = circular_layout(n).into_iter().map(|q| Vec2::new(q.x * scale, q.y * scale)).collect();

    // The spring constant and rest length for the pair (i, j).
    let k = |i: usize, j: usize| 1.0 / (d[i][j] * d[i][j]);
    let energy = |p: &[Vec2]| -> f64 {
        let mut e = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let diff = p[i].distance_to(&p[j]) - d[i][j];
                e += 0.5 * k(i, j) * diff * diff;
            }
        }
        e
    };
    let grad = |p: &[Vec2], m: usize| -> (f64, f64, f64, f64, f64) {
        let (mut gx, mut gy, mut hxx, mut hxy, mut hyy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            if i == m {
                continue;
            }
            let dx = p[m].x - p[i].x;
            let dy = p[m].y - p[i].y;
            let r2 = dx * dx + dy * dy;
            if r2 <= 0.0 {
                continue;
            }
            let r = r2.sqrt();
            let r3 = r2 * r;
            let kk = k(i, m);
            let l = d[i][m];
            gx += kk * (dx - l * dx / r);
            gy += kk * (dy - l * dy / r);
            hxx += kk * (1.0 - l * dy * dy / r3);
            hxy += kk * (l * dx * dy / r3);
            hyy += kk * (1.0 - l * dx * dx / r3);
        }
        (gx, gy, hxx, hxy, hyy)
    };

    let mut current = energy(&p);
    for _ in 0..iters {
        let Some(m) = (0..n).max_by(|&a, &b| {
            let ga = grad(&p, a);
            let gb = grad(&p, b);
            (ga.0 * ga.0 + ga.1 * ga.1).total_cmp(&(gb.0 * gb.0 + gb.1 * gb.1))
        }) else {
            break;
        };
        let (gx, gy, hxx, hxy, hyy) = grad(&p, m);
        if gx * gx + gy * gy < 1e-18 {
            break;
        }
        let det = hxx * hyy - hxy * hxy;
        // A Newton step where the Hessian is usable, the steepest descent
        // direction where it is not.
        let (mut sx, mut sy) = if det.abs() > 1e-12 {
            ((-gx * hyy + gy * hxy) / det, (-gy * hxx + gx * hxy) / det)
        } else {
            (-gx, -gy)
        };
        let saved = p[m];
        let mut accepted = false;
        for _ in 0..30 {
            p[m] = Vec2::new(saved.x + sx, saved.y + sy);
            let next = energy(&p);
            if next < current {
                current = next;
                accepted = true;
                break;
            }
            sx *= 0.5;
            sy *= 0.5;
        }
        if !accepted {
            p[m] = saved;
        }
    }
    p
}

/// Stress majorization (SMACOF) in `dim` dimensions.
///
/// Each round replaces the stress by a quadratic that touches it at the
/// current drawing and lies above it everywhere else, then jumps to that
/// quadratic's minimum. Because the surrogate is an upper bound, the true
/// stress cannot rise -- which is the whole point, and the reason this is
/// preferred to gradient descent on the same objective: there is no step size
/// to tune and no way to overshoot.
///
/// The starting drawing is classical scaling, the closed-form embedding that
/// best reproduces the *squared* distances. Majorization only ever descends,
/// so where it starts decides which local minimum it reaches; starting from
/// the classical solution makes the result deterministic and already close.
///
/// # Panics
/// Panics if the graph is directed, or `dim` is zero.
#[must_use]
pub fn stress_majorization(g: &Graph, dim: usize, iters: usize) -> Vec<VecN> {
    assert!(!g.directed, "layouts here are for undirected graphs");
    assert!(dim > 0, "a drawing needs at least one dimension");
    let n = g.n;
    let d = hop_distances(g);
    let mut p = classical_scaling(&d, dim);
    for _ in 0..iters {
        let mut next = Vec::with_capacity(n);
        for i in 0..n {
            let mut acc = VecN::zeros(dim);
            let mut wsum = 0.0;
            for j in 0..n {
                if i == j || d[i][j] <= 0.0 {
                    continue;
                }
                let w = 1.0 / (d[i][j] * d[i][j]);
                let diff = p[i].sub(&p[j]);
                let r = diff.norm();
                // Where two points coincide the direction is undefined; the
                // majorizing bound holds for any unit vector, so leave them
                // where they are rather than inventing one.
                let term = if r > 1e-12 {
                    p[j].add(&diff.scale(d[i][j] / r))
                } else {
                    p[j].clone()
                };
                acc = acc.add(&term.scale(w));
                wsum += w;
            }
            next.push(if wsum > 0.0 { acc.scale(1.0 / wsum) } else { p[i].clone() });
        }
        p = next;
    }
    p
}

/// Classical scaling: the `dim` coordinates that best reproduce the squared
/// distances, from the eigenvectors of the double-centred squared-distance
/// matrix.
fn classical_scaling(d: &[Vec<f64>], dim: usize) -> Vec<VecN> {
    let n = d.len();
    if n == 0 {
        return Vec::new();
    }
    // B = -1/2 J D^2 J, whose eigenvectors scaled by the root of their
    // eigenvalues are the coordinates.
    let sq: Vec<Vec<f64>> = d.iter().map(|r| r.iter().map(|x| x * x).collect()).collect();
    let row: Vec<f64> = sq.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
    let grand: f64 = row.iter().sum::<f64>() / n as f64;
    let mut b = crate::linalg::matrix::Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            b.set(i, j, -0.5 * (sq[i][j] - row[i] - row[j] + grand));
        }
    }
    let e = crate::linalg::eigen::eigen_symmetric(&b, 1e-12, 200)
        .expect("Jacobi converges on a symmetric matrix");
    (0..n)
        .map(|v| {
            let coords: Vec<f64> = (0..dim)
                .map(|k| {
                    if k >= n {
                        return 0.0;
                    }
                    // Descending order, so the leading eigenvectors come
                    // first. A negative eigenvalue means that direction is
                    // not realisable in Euclidean space and contributes
                    // nothing.
                    let lambda = e.values[k].max(0.0);
                    e.vectors.get(v, k) * lambda.sqrt()
                })
                .collect();
            VecN::from(&coords)
        })
        .collect()
}

/// Fruchterman-Reingold: vertices repel like charges, edges pull like
/// springs, and the whole thing cools.
///
/// Repulsion is `k^2 / r` between every pair and attraction is `r^2 / k`
/// along every edge, for the ideal separation `k = sqrt(area / n)`. The two
/// balance at `r = k`, which is what sets the scale of the drawing. The
/// temperature caps how far any vertex may move in one round and falls
/// linearly to zero, so the layout freezes rather than oscillating -- the
/// method is a heuristic with no monotonicity guarantee, and the cooling is
/// what stands in for one.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn fruchterman_reingold(g: &Graph, iters: usize, rng: &mut Rng) -> Vec<Vec2> {
    assert!(!g.directed, "layouts here are for undirected graphs");
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }
    let side = (n as f64).sqrt();
    let k = side / (n as f64).sqrt();
    let mut p: Vec<Vec2> = (0..n)
        .map(|_| Vec2::new(side * (rng.next_f64() - 0.5), side * (rng.next_f64() - 0.5)))
        .collect();
    let edges: Vec<(usize, usize)> =
        g.edges().iter().filter(|&&(u, v, _)| u != v).map(|&(u, v, _)| (u, v)).collect();
    let mut temp = side / 10.0;
    let cool = temp / (iters.max(1) as f64);
    for _ in 0..iters {
        let mut disp = vec![Vec2::ZERO; n];
        for i in 0..n {
            for j in i + 1..n {
                let mut delta = p[i] - p[j];
                let mut r = delta.magnitude();
                if r < 1e-9 {
                    // Two vertices exactly on top of each other have no
                    // direction to separate along; nudge them apart.
                    delta = Vec2::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5);
                    r = delta.magnitude().max(1e-9);
                }
                let force = k * k / r;
                let unit = delta.normalized();
                disp[i] = disp[i] + unit * force;
                disp[j] = disp[j] - unit * force;
            }
        }
        for &(u, v) in &edges {
            let delta = p[u] - p[v];
            let r = delta.magnitude().max(1e-9);
            let force = r * r / k;
            let unit = delta.normalized();
            disp[u] = disp[u] - unit * force;
            disp[v] = disp[v] + unit * force;
        }
        for i in 0..n {
            let r = disp[i].magnitude();
            if r > 1e-12 {
                let step = r.min(temp);
                p[i] = p[i] + disp[i].normalized() * step;
            }
        }
        temp = (temp - cool).max(0.0);
    }
    p
}

// ---------------------------------------------------------------------------
// Structural layouts
// ---------------------------------------------------------------------------

/// The children of each vertex, rooted at `root`, in ascending index order.
fn rooted_children(g: &Graph, root: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = g.n;
    let mut children = vec![Vec::new(); n];
    let mut depth = vec![0usize; n];
    let mut seen = vec![false; n];
    seen[root] = true;
    let mut q = VecDeque::from([root]);
    while let Some(u) = q.pop_front() {
        let mut next: Vec<usize> =
            g.adj[u].iter().map(|&(v, _)| v).filter(|&v| !seen[v]).collect();
        next.sort_unstable();
        next.dedup();
        for v in next {
            seen[v] = true;
            depth[v] = depth[u] + 1;
            children[u].push(v);
            q.push_back(v);
        }
    }
    (children, depth)
}

/// The relative drawing of one subtree: offsets from its own root, and the
/// leftmost and rightmost occupied position at each depth below it.
struct Subtree {
    offset: Vec<(usize, f64)>,
    left: Vec<f64>,
    right: Vec<f64>,
}

/// Reingold-Tilford tree layout.
///
/// Depth sets the vertical position and the horizontal one is chosen so that
/// three things hold at once: no two subtrees overlap, every parent sits at
/// the midpoint of its first and last child, and the drawing is as narrow as
/// those two allow. The third is what the algorithm is for -- centring a
/// parent over its children is easy, and doing it while packing sibling
/// subtrees as tightly as their outlines permit is not.
///
/// Packing works on *contours*: the leftmost and rightmost position each
/// subtree occupies at every depth. Two siblings are pushed apart by the
/// largest overlap between the right contour of everything placed so far and
/// the left contour of the newcomer, so subtrees interlock where their shapes
/// leave room.
///
/// The root is at `(0, 0)` and depth `k` at `y = -k`, so the tree hangs
/// downward. Vertices unreachable from the root keep the origin.
///
/// # Panics
/// Panics if the graph is directed, `root` is out of range, or the graph has
/// a cycle reachable from the root -- the layout is defined on trees.
#[must_use]
pub fn tree_layout_reingold_tilford(g: &Graph, root: usize) -> Vec<Vec2> {
    assert!(!g.directed, "the tree layout is for undirected trees");
    assert!(root < g.n, "root {root} is outside 0..{}", g.n);
    let (children, depth) = rooted_children(g, root);
    // Reachable vertices must span a tree: one edge fewer than vertices.
    let reachable: usize = 1 + children.iter().map(Vec::len).sum::<usize>();
    let inside: BTreeSet<usize> = {
        let mut s = BTreeSet::from([root]);
        let mut stack = vec![root];
        while let Some(u) = stack.pop() {
            for &c in &children[u] {
                s.insert(c);
                stack.push(c);
            }
        }
        s
    };
    let spanned = g
        .edges()
        .iter()
        .filter(|&&(u, v, _)| u != v && inside.contains(&u) && inside.contains(&v))
        .map(|&(u, v, _)| (u.min(v), u.max(v)))
        .collect::<BTreeSet<_>>()
        .len();
    assert_eq!(spanned, reachable - 1, "the component of the root is not a tree");

    let sub = tree_pack(root, &children, 1.0);
    let mut pos = vec![Vec2::ZERO; g.n];
    for (v, x) in sub.offset {
        pos[v] = Vec2::new(x, -(depth[v] as f64));
    }
    pos
}

fn tree_pack(v: usize, children: &[Vec<usize>], sep: f64) -> Subtree {
    if children[v].is_empty() {
        return Subtree { offset: vec![(v, 0.0)], left: vec![0.0], right: vec![0.0] };
    }
    let parts: Vec<Subtree> = children[v].iter().map(|&c| tree_pack(c, children, sep)).collect();
    // Place each child as far left as its contour allows against everything
    // already placed.
    let mut place = Vec::with_capacity(parts.len());
    let mut right_so_far: Vec<f64> = Vec::new();
    for part in &parts {
        let mut at = if place.is_empty() { 0.0 } else { f64::NEG_INFINITY };
        for t in 0..right_so_far.len().min(part.left.len()) {
            at = at.max(right_so_far[t] - part.left[t] + sep);
        }
        if !at.is_finite() {
            // No shared depth, so nothing to clear: sit beside the previous
            // child at the separation distance.
            at = right_so_far.first().copied().unwrap_or(0.0) + sep;
        }
        place.push(at);
        for t in 0..part.right.len() {
            let x = at + part.right[t];
            if t < right_so_far.len() {
                right_so_far[t] = right_so_far[t].max(x);
            } else {
                right_so_far.push(x);
            }
        }
    }
    // Centre the parent over its first and last child.
    let centre = (place[0] + place[place.len() - 1]) / 2.0;
    let mut offset: Vec<(usize, f64)> = vec![(v, 0.0)];
    let mut left: Vec<f64> = vec![0.0];
    let mut right: Vec<f64> = vec![0.0];
    for (part, at) in parts.iter().zip(&place) {
        let shift = at - centre;
        for &(w, x) in &part.offset {
            offset.push((w, x + shift));
        }
        for t in 0..part.left.len() {
            let (lo, hi) = (part.left[t] + shift, part.right[t] + shift);
            if t + 1 < left.len() {
                left[t + 1] = left[t + 1].min(lo);
                right[t + 1] = right[t + 1].max(hi);
            } else {
                left.push(lo);
                right.push(hi);
            }
        }
    }
    Subtree { offset, left, right }
}

/// Sugiyama layered drawing of a directed acyclic graph.
///
/// Layer `k` holds the vertices whose longest incoming path has `k` arcs, so
/// every arc goes from a strictly lower layer to a higher one and the drawing
/// reads in one direction. Within a layer the order is fixed by repeated
/// barycentre sweeps: put each vertex at the average position of its
/// neighbours in the adjacent layer, sort, and repeat, alternating direction.
/// That is Sugiyama's crossing-reduction heuristic; minimising crossings
/// exactly is NP-hard even for two layers.
///
/// Returns `x` as the position within the layer and `y` as minus the layer,
/// so the arcs point downward.
///
/// # Panics
/// Panics unless the graph is directed and acyclic.
#[must_use]
pub fn sugiyama_layered(dag: &Graph) -> Vec<Vec2> {
    assert!(dag.directed, "a layered drawing is for directed graphs");
    let order = dag.topological_sort().expect("a layered drawing needs an acyclic graph");
    let n = dag.n;
    let mut layer = vec![0usize; n];
    for &u in &order {
        for &(v, _) in &dag.adj[u] {
            layer[v] = layer[v].max(layer[u] + 1);
        }
    }
    let depth = layer.iter().copied().max().map_or(0, |m| m + 1);
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); depth];
    for v in 0..n {
        rows[layer[v]].push(v);
    }
    // Position within the layer, which is what the sweeps permute.
    let mut at = vec![0usize; n];
    for row in &rows {
        for (i, &v) in row.iter().enumerate() {
            at[v] = i;
        }
    }
    let mut down = vec![Vec::new(); n];
    let mut up = vec![Vec::new(); n];
    for u in 0..n {
        for &(v, _) in &dag.adj[u] {
            if u != v {
                down[u].push(v);
                up[v].push(u);
            }
        }
    }
    for round in 0..8 {
        let forward = round % 2 == 0;
        let sweep: Vec<usize> = if forward { (1..depth).collect() } else { (0..depth.saturating_sub(1)).rev().collect() };
        for k in sweep {
            let refs = if forward { &up } else { &down };
            let mut row = rows[k].clone();
            row.sort_by(|&a, &b| {
                let bary = |v: usize| -> f64 {
                    let ns = &refs[v];
                    if ns.is_empty() {
                        at[v] as f64
                    } else {
                        ns.iter().map(|&w| at[w] as f64).sum::<f64>() / ns.len() as f64
                    }
                };
                bary(a).total_cmp(&bary(b)).then_with(|| a.cmp(&b))
            });
            for (i, &v) in row.iter().enumerate() {
                at[v] = i;
            }
            rows[k] = row;
        }
    }
    (0..n).map(|v| Vec2::new(at[v] as f64, -(layer[v] as f64))).collect()
}

// ---------------------------------------------------------------------------
// Crossings and planarity
// ---------------------------------------------------------------------------

/// Whether the open segments `a-b` and `c-d` cross at an interior point of
/// both.
fn segments_cross(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let side = |p: Vec2, q: Vec2, r: Vec2| (q - p).cross(&(r - p));
    let (d1, d2) = (side(c, d, a), side(c, d, b));
    let (d3, d4) = (side(a, b, c), side(a, b, d));
    // Strict signs on both: a shared endpoint or a touching endpoint gives a
    // zero and is not a crossing.
    ((d1 > 0.0) != (d2 > 0.0)) && d1 != 0.0 && d2 != 0.0
        && ((d3 > 0.0) != (d4 > 0.0)) && d3 != 0.0 && d4 != 0.0
}

/// The number of edge crossings in the straight-line drawing given by
/// `layout`.
///
/// An upper bound on the graph's crossing number, and only that: the crossing
/// number is the minimum over all drawings, and a graph's best drawing need
/// not even be straight-line for a general graph. Edges sharing an endpoint
/// are never counted, and neither is a touching that is not a proper
/// crossing.
///
/// # Panics
/// Panics unless there is one position per vertex.
#[must_use]
pub fn crossing_number_estimate(g: &Graph, layout: &[Vec2]) -> usize {
    assert_eq!(layout.len(), g.n, "one position per vertex is required");
    let edges: Vec<(usize, usize)> =
        g.edges().iter().filter(|&&(u, v, _)| u != v).map(|&(u, v, _)| (u, v)).collect();
    let mut count = 0;
    for i in 0..edges.len() {
        for j in i + 1..edges.len() {
            let (a, b) = edges[i];
            let (c, d) = edges[j];
            if a == c || a == d || b == c || b == d {
                continue;
            }
            if segments_cross(layout[a], layout[b], layout[c], layout[d]) {
                count += 1;
            }
        }
    }
    count
}

/// The edge sets of the biconnected components, each a maximal subgraph with
/// no cut vertex.
///
/// A graph is planar exactly when every block is, which is what makes this
/// the right decomposition to plan a planarity test around: the blocks meet
/// only at single vertices, and a drawing of each can be rotated and scaled
/// into place around those without interfering.
///
/// Self-loops are dropped and parallel edges collapsed, so each returned
/// block lists distinct simple edges.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn biconnected_components(g: &Graph) -> Vec<Vec<(usize, usize)>> {
    assert!(!g.directed, "blocks are defined for undirected graphs");
    let n = g.n;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, v, _) in g.edges() {
        if u != v {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut num = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut timer = 0usize;
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut out: Vec<Vec<(usize, usize)>> = Vec::new();
    for s in 0..n {
        if num[s] != usize::MAX {
            continue;
        }
        // Iterative depth-first search, carrying the cursor into each
        // vertex's adjacency so the traversal can be resumed.
        let mut frames: Vec<(usize, usize, usize)> = vec![(s, usize::MAX, 0)];
        num[s] = timer;
        low[s] = timer;
        timer += 1;
        while let Some(&mut (u, parent, ref mut i)) = frames.last_mut() {
            if *i < adj[u].len() {
                let v = adj[u][*i];
                *i += 1;
                if num[v] == usize::MAX {
                    stack.push((u, v));
                    num[v] = timer;
                    low[v] = timer;
                    timer += 1;
                    frames.push((v, u, 0));
                } else if v != parent && num[v] < num[u] {
                    stack.push((u, v));
                    low[u] = low[u].min(num[v]);
                }
            } else {
                frames.pop();
                if let Some(&mut (p, _, _)) = frames.last_mut() {
                    low[p] = low[p].min(low[u]);
                    if low[u] >= num[p] {
                        // p is a cut vertex (or the root): everything pushed
                        // since the edge into u forms one block.
                        let mut block = Vec::new();
                        while let Some(&(a, b)) = stack.last() {
                            if num[a] >= num[u] || (a, b) == (p, u) {
                                block.push((a, b));
                                stack.pop();
                                if (a, b) == (p, u) {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        if !block.is_empty() {
                            out.push(block);
                        }
                    }
                }
            }
        }
    }
    out
}

/// A planar embedding of a biconnected graph, as its faces: each face is the
/// cyclic sequence of vertices bounding it. `None` if the graph is not
/// planar.
///
/// By Demoucron's path-addition method. Start with any cycle, which divides
/// the plane into two faces, and grow: the parts of the graph not yet drawn
/// -- its *fragments* -- each attach to the drawn part at a set of vertices,
/// and a fragment can only go inside a face that contains all of them. If
/// some fragment fits nowhere, the graph is not planar. If a fragment fits in
/// exactly one face it is forced, so it is drawn first; otherwise any choice
/// will do, and that is the theorem the method rests on. Drawing a path of a
/// fragment across a face splits that face in two, and the process repeats
/// until every edge is drawn.
///
/// The outer face is among those returned; which one it is depends on the
/// starting cycle, since on the sphere no face is distinguished.
///
/// # Panics
/// Panics if the graph is directed, has a self-loop, has fewer than three
/// vertices, or is not biconnected. Use [`planarity_test`] for a graph that
/// is any of those: it decomposes into blocks first.
#[must_use]
pub fn planar_embedding_small(g: &Graph) -> Option<Vec<Vec<usize>>> {
    assert!(!g.directed, "planarity here is for undirected graphs");
    assert!(!g.edges().iter().any(|&(u, v, _)| u == v), "a self-loop has no place in a face");
    assert!(g.n >= 3, "an embedding needs at least three vertices");
    let blocks = biconnected_components(g);
    assert!(blocks.len() == 1, "the graph must be biconnected; it has {} blocks", blocks.len());
    let edges: BTreeSet<(usize, usize)> = g
        .edges()
        .iter()
        .map(|&(u, v, _)| (u.min(v), u.max(v)))
        .collect();
    embed_biconnected(g.n, &edges)
}

/// Neighbour lists from a canonical edge set.
fn neighbors_of(n: usize, edges: &BTreeSet<(usize, usize)>) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    adj
}

/// Any cycle in a graph with minimum degree two, as a vertex sequence.
fn find_cycle(n: usize, adj: &[Vec<usize>]) -> Option<Vec<usize>> {
    let mut parent = vec![usize::MAX; n];
    let mut seen = vec![false; n];
    for s in 0..n {
        if seen[s] || adj[s].is_empty() {
            continue;
        }
        seen[s] = true;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    parent[v] = u;
                    stack.push(v);
                } else if parent[u] != v {
                    // A non-tree edge closes a cycle: walk both ends up to
                    // their meeting point.
                    let mut a = vec![u];
                    let mut x = u;
                    while parent[x] != usize::MAX {
                        x = parent[x];
                        a.push(x);
                    }
                    let mut b = vec![v];
                    let mut y = v;
                    while parent[y] != usize::MAX {
                        y = parent[y];
                        b.push(y);
                    }
                    let common = a.iter().position(|p| b.contains(p))?;
                    let meet = a[common];
                    let cut = b.iter().position(|&p| p == meet)?;
                    let mut cycle: Vec<usize> = a[..=common].to_vec();
                    for &w in b[..cut].iter().rev() {
                        cycle.push(w);
                    }
                    if cycle.len() >= 3 {
                        return Some(cycle);
                    }
                }
            }
        }
    }
    None
}

/// Demoucron on a biconnected simple graph given as `n` and its edge set.
fn embed_biconnected(n: usize, edges: &BTreeSet<(usize, usize)>) -> Option<Vec<Vec<usize>>> {
    if edges.is_empty() {
        return Some(Vec::new());
    }
    // Euler's bounds reject the dense cases outright, and cheaply.
    let m = edges.len();
    if n >= 3 && m > 3 * n - 6 {
        return None;
    }
    let adj = neighbors_of(n, edges);
    let cycle = find_cycle(n, &adj)?;
    let mut faces: Vec<Vec<usize>> = {
        let mut back = cycle.clone();
        back.reverse();
        vec![cycle.clone(), back]
    };
    let mut drawn: BTreeSet<(usize, usize)> = BTreeSet::new();
    for w in 0..cycle.len() {
        let (a, b) = (cycle[w], cycle[(w + 1) % cycle.len()]);
        drawn.insert((a.min(b), a.max(b)));
    }
    let mut on_face = vec![false; n];
    for &v in &cycle {
        on_face[v] = true;
    }

    while drawn.len() < edges.len() {
        // Fragments: single undrawn edges between drawn vertices, and the
        // connected pieces of everything not yet drawn at all.
        let mut fragments: Vec<(Vec<usize>, Vec<usize>)> = Vec::new(); // (attachments, interior)
        for &(u, v) in edges.difference(&drawn) {
            if on_face[u] && on_face[v] {
                fragments.push((vec![u, v], Vec::new()));
            }
        }
        let mut visited = vec![false; n];
        for s in 0..n {
            if on_face[s] || visited[s] {
                continue;
            }
            let mut interior = Vec::new();
            let mut attach = BTreeSet::new();
            let mut stack = vec![s];
            visited[s] = true;
            while let Some(u) = stack.pop() {
                interior.push(u);
                for &w in &adj[u] {
                    if on_face[w] {
                        attach.insert(w);
                    } else if !visited[w] {
                        visited[w] = true;
                        stack.push(w);
                    }
                }
            }
            fragments.push((attach.into_iter().collect(), interior));
        }
        if fragments.is_empty() {
            break;
        }
        // Which faces can hold each fragment: those containing every
        // attachment.
        let admissible: Vec<Vec<usize>> = fragments
            .iter()
            .map(|(att, _)| {
                (0..faces.len())
                    .filter(|&f| att.iter().all(|a| faces[f].contains(a)))
                    .collect()
            })
            .collect();
        if admissible.iter().any(Vec::is_empty) {
            return None;
        }
        // A fragment with one admissible face is forced, so settle it first.
        let choice = admissible
            .iter()
            .position(|f| f.len() == 1)
            .unwrap_or(0);
        let face = admissible[choice][0];
        let (att, interior) = &fragments[choice];
        let path = fragment_path(att, interior, &adj, &drawn, &on_face)?;

        // Split the chosen face along the path.
        let f = &faces[face];
        let i = f.iter().position(|&x| x == path[0])?;
        let j = f.iter().position(|&x| x == path[path.len() - 1])?;
        let arc = |from: usize, to: usize| -> Vec<usize> {
            let mut out = vec![f[from]];
            let mut k = from;
            while k != to {
                k = (k + 1) % f.len();
                out.push(f[k]);
            }
            out
        };
        let inner: Vec<usize> = path[1..path.len() - 1].to_vec();
        let mut f1 = arc(i, j);
        f1.extend(inner.iter().rev());
        let mut f2 = arc(j, i);
        f2.extend(inner.iter());
        faces.swap_remove(face);
        faces.push(f1);
        faces.push(f2);

        for w in 0..path.len() - 1 {
            let (a, b) = (path[w], path[w + 1]);
            drawn.insert((a.min(b), a.max(b)));
            on_face[a] = true;
            on_face[b] = true;
        }
    }
    Some(faces)
}

/// A path across a fragment: from one attachment, through its interior, to
/// another. A fragment that is a single undrawn edge is already such a path.
fn fragment_path(
    att: &[usize],
    interior: &[usize],
    adj: &[Vec<usize>],
    drawn: &BTreeSet<(usize, usize)>,
    on_face: &[bool],
) -> Option<Vec<usize>> {
    if interior.is_empty() {
        return Some(vec![att[0], att[1]]);
    }
    let start = att[0];
    let inside: BTreeSet<usize> = interior.iter().copied().collect();
    // Breadth-first from the start's neighbours in the fragment, stopping at
    // the first interior vertex that reaches a different attachment.
    let mut parent = std::collections::BTreeMap::new();
    let mut q = VecDeque::new();
    for &w in &adj[start] {
        if inside.contains(&w) && !parent.contains_key(&w) {
            parent.insert(w, start);
            q.push_back(w);
        }
    }
    while let Some(u) = q.pop_front() {
        for &w in &adj[u] {
            if on_face[w] && w != start && !drawn.contains(&(u.min(w), u.max(w))) {
                // Walk back to the start and hand over the whole path.
                let mut path = vec![w, u];
                let mut x = u;
                while let Some(&p) = parent.get(&x) {
                    path.push(p);
                    if p == start {
                        break;
                    }
                    x = p;
                }
                path.reverse();
                return Some(path);
            }
            if inside.contains(&w) && !parent.contains_key(&w) {
                parent.insert(w, u);
                q.push_back(w);
            }
        }
    }
    None
}

/// Whether the graph can be drawn in the plane with no edge crossings.
///
/// Exact, not an estimate. Parallel edges and self-loops are ignored, since
/// neither can make a drawable graph undrawable, and the graph is split into
/// its blocks: planarity holds for the whole exactly when it holds for each,
/// and each block is biconnected, which is what
/// [`planar_embedding_small`] needs.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn planarity_test(g: &Graph) -> bool {
    assert!(!g.directed, "planarity here is for undirected graphs");
    for block in biconnected_components(g) {
        let mut verts: Vec<usize> = block.iter().flat_map(|&(u, v)| [u, v]).collect();
        verts.sort_unstable();
        verts.dedup();
        if verts.len() < 3 {
            continue;
        }
        // Renumber the block into 0..k so the embedding works on a compact
        // range, and drop parallel edges on the way.
        let index = |v: usize| verts.binary_search(&v).expect("v is in the block");
        let edges: BTreeSet<(usize, usize)> = block
            .iter()
            .filter(|&&(u, v)| u != v)
            .map(|&(u, v)| {
                let (a, b) = (index(u), index(v));
                (a.min(b), a.max(b))
            })
            .collect();
        if embed_biconnected(verts.len(), &edges).is_none() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::spectral::laplacian_matrix;
    use crate::linalg::eigen::eigen_symmetric;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * a.abs().max(b.abs()).max(1.0)
    }

    fn random_graph(n: usize, p: f64, rng: &mut Rng) -> Graph {
        let mut g = Graph::new(n, false);
        for u in 0..n {
            for v in u + 1..n {
                if rng.next_f64() < p {
                    g.add_edge(u, v, 1.0);
                }
            }
        }
        g
    }

    fn cycle_graph(n: usize) -> Graph {
        let mut g = Graph::new(n, false);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n, 1.0);
        }
        g
    }

    fn complete_graph(n: usize) -> Graph {
        let mut g = Graph::new(n, false);
        for i in 0..n {
            for j in i + 1..n {
                g.add_edge(i, j, 1.0);
            }
        }
        g
    }

    fn path_graph(n: usize) -> Graph {
        let mut g = Graph::new(n, false);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1, 1.0);
        }
        g
    }

    fn complete_bipartite(a: usize, b: usize) -> Graph {
        let mut g = Graph::new(a + b, false);
        for i in 0..a {
            for j in 0..b {
                g.add_edge(i, a + j, 1.0);
            }
        }
        g
    }

    fn petersen_graph() -> Graph {
        let mut g = Graph::new(10, false);
        for i in 0..5 {
            g.add_edge(i, (i + 1) % 5, 1.0);
            g.add_edge(i, 5 + i, 1.0);
            g.add_edge(5 + i, 5 + (i + 2) % 5, 1.0);
        }
        g
    }

    fn grid_graph(w: usize, h: usize) -> Graph {
        let mut g = Graph::new(w * h, false);
        for r in 0..h {
            for c in 0..w {
                if c + 1 < w {
                    g.add_edge(r * w + c, r * w + c + 1, 1.0);
                }
                if r + 1 < h {
                    g.add_edge(r * w + c, (r + 1) * w + c, 1.0);
                }
            }
        }
        g
    }

    /// A binary tree on `n` vertices with the usual heap indexing.
    fn binary_tree(n: usize) -> Graph {
        let mut g = Graph::new(n, false);
        for v in 1..n {
            g.add_edge((v - 1) / 2, v, 1.0);
        }
        g
    }

    /// Subdividing every edge of `g` once. Planarity is preserved by
    /// subdivision, which makes this the sharpest way to probe a planarity
    /// test: the subdivided K5 has no K5 subgraph at all.
    fn subdivide(g: &Graph) -> Graph {
        let edges = g.edges();
        let mut h = Graph::new(g.n + edges.len(), false);
        for (i, &(u, v, _)) in edges.iter().enumerate() {
            let mid = g.n + i;
            h.add_edge(u, mid, 1.0);
            h.add_edge(mid, v, 1.0);
        }
        h
    }

    /// The circular layout is a regular polygon: unit radius, equal angular
    /// steps, and no two points together.
    #[test]
    fn circular_layout_is_a_regular_polygon() {
        for n in 1..=20usize {
            let p = circular_layout(n);
            assert_eq!(p.len(), n);
            for q in &p {
                assert!(close(q.magnitude(), 1.0), "not on the unit circle");
            }
            for i in 0..n {
                for j in i + 1..n {
                    assert!(p[i].distance_to(&p[j]) > 1e-9, "two vertices coincide");
                }
                // Consecutive points are one chord apart, the same chord all
                // the way round.
                let step = p[i].distance_to(&p[(i + 1) % n]);
                let first = p[0].distance_to(&p[1 % n]);
                assert!(close(step, first), "the steps are not equal");
            }
            // A cycle drawn on a circle in cycle order has no crossings at
            // all, which is the reason to draw it that way.
            if n >= 3 {
                assert_eq!(crossing_number_estimate(&cycle_graph(n), &p), 0);
            }
        }
    }

    /// Every shell sits on its own circle, and the members of a shell are
    /// spread evenly round it.
    #[test]
    fn shell_layout_places_each_shell_on_its_circle() {
        let g = Graph::new(10, false);
        let shells = vec![vec![0], vec![1, 2, 3], vec![4, 5, 6, 7, 8, 9]];
        let p = shell_layout(&g, &shells);
        assert!(close(p[0].magnitude(), 0.0), "a lone first shell is the centre");
        for &v in &shells[1] {
            assert!(close(p[v].magnitude(), 2.0));
        }
        for &v in &shells[2] {
            assert!(close(p[v].magnitude(), 3.0));
        }
        // Equal spacing within a shell: consecutive members are one chord
        // apart on their circle.
        for shell in &shells[1..] {
            let first = p[shell[0]].distance_to(&p[shell[1]]);
            for i in 0..shell.len() {
                let step = p[shell[i]].distance_to(&p[shell[(i + 1) % shell.len()]]);
                assert!(close(step, first));
            }
        }
        // A shell of more than one vertex is not put at the centre.
        let two = shell_layout(&g, &[vec![0, 1], (2..10).collect()]);
        assert!(close(two[0].magnitude(), 1.0));
    }

    /// The spectral layout's coordinates must actually be Laplacian
    /// eigenvectors: the two just above the constant one.
    #[test]
    fn spectral_layout_uses_laplacian_eigenvectors() {
        let mut rng = Rng::new(0x_5AEC);
        for _ in 0..60 {
            let n = 3 + pick(&mut rng, 8);
            let g = random_graph(n, 0.3 + 0.5 * rng.next_f64(), &mut rng);
            if !g.is_connected() {
                continue;
            }
            let p = spectral_layout(&g);
            let l = laplacian_matrix(&g);
            let spectrum = {
                let e = eigen_symmetric(&l, 1e-12, 200).expect("symmetric");
                let mut v = e.values;
                v.reverse();
                v
            };
            for (k, coord) in [(1usize, 0usize), (2, 1)] {
                let x: Vec<f64> = p.iter().map(|q| if coord == 0 { q.x } else { q.y }).collect();
                let lambda = spectrum[k];
                // L x = lambda x, entry by entry.
                for i in 0..n {
                    let lx: f64 = (0..n).map(|j| l.get(i, j) * x[j]).sum();
                    assert!(
                        (lx - lambda * x[i]).abs() < 1e-6,
                        "coordinate {coord} is not the eigenvector for {lambda}"
                    );
                }
                // Orthogonal to the constant vector, so the drawing is
                // centred rather than translated off somewhere.
                assert!(x.iter().sum::<f64>().abs() < 1e-6, "not centred");
                assert!((x.iter().map(|a| a * a).sum::<f64>() - 1.0).abs() < 1e-6);
            }
            // And to each other.
            let dot: f64 = p.iter().map(|q| q.x * q.y).sum();
            assert!(dot.abs() < 1e-6, "the two axes are not orthogonal");
        }
    }

    /// Majorization's guarantee: the stress never rises, at any round, on any
    /// graph. That is the property the method exists for, and nothing weaker
    /// distinguishes it from gradient descent.
    #[test]
    fn stress_majorization_never_increases_stress() {
        let mut rng = Rng::new(0x_5735);
        for _ in 0..60 {
            let n = 2 + pick(&mut rng, 10);
            let g = random_graph(n, 0.25 + 0.5 * rng.next_f64(), &mut rng);
            let mut previous = f64::INFINITY;
            for iters in 0..12 {
                let p = stress_majorization(&g, 2, iters);
                assert_eq!(p.len(), n);
                let s = stress_nd(&g, &p);
                assert!(s.is_finite(), "the stress went non-finite");
                assert!(
                    s <= previous + 1e-9,
                    "stress rose from {previous} to {s} at round {iters}"
                );
                previous = s;
            }
        }
        // A path is exactly realisable on a line, so one dimension is enough
        // and the stress must fall essentially to zero.
        let p = stress_majorization(&path_graph(8), 1, 400);
        assert!(stress_nd(&path_graph(8), &p) < 1e-6, "a path did not lay out on a line");
        // Consecutive vertices one apart, in order.
        let xs: Vec<f64> = p.iter().map(|q| q.data[0]).collect();
        for i in 0..7 {
            assert!(close((xs[i + 1] - xs[i]).abs(), 1.0));
        }
        // A cycle needs two dimensions, and gets them: the drawing should be
        // close to a regular polygon, with every edge the same length.
        let c = cycle_graph(9);
        let q = stress_majorization(&c, 2, 400);
        let len = |i: usize, j: usize| q[i].sub(&q[j]).norm();
        let first = len(0, 1);
        for i in 0..9 {
            assert!(
                (len(i, (i + 1) % 9) - first).abs() < 1e-3,
                "the cycle is not laid out symmetrically"
            );
        }
    }

    /// Kamada-Kawai must not do worse than the drawing it starts from, and
    /// on graphs a circle draws badly it must do markedly better.
    #[test]
    fn kamada_kawai_improves_on_its_starting_drawing() {
        let mut rng = Rng::new(0x_4A4A);
        for _ in 0..40 {
            let n = 2 + pick(&mut rng, 9);
            let g = random_graph(n, 0.25 + 0.5 * rng.next_f64(), &mut rng);
            let d = hop_distances(&g);
            let scale = d.iter().flatten().copied().fold(1.0f64, f64::max);
            let start: Vec<Vec2> = circular_layout(n)
                .into_iter()
                .map(|q| Vec2::new(q.x * scale, q.y * scale))
                .collect();
            let before = layout_stress(&g, &start);
            let after = layout_stress(&g, &kamada_kawai(&g, 200));
            assert!(after <= before + 1e-9, "stress rose from {before} to {after}");
        }
        // A path drawn on a circle is badly wrong and the method must fix it.
        let g = path_graph(9);
        let d = hop_distances(&g);
        let scale = d.iter().flatten().copied().fold(1.0f64, f64::max);
        let start: Vec<Vec2> =
            circular_layout(9).into_iter().map(|q| Vec2::new(q.x * scale, q.y * scale)).collect();
        let p = kamada_kawai(&g, 400);
        assert!(layout_stress(&g, &p) < 0.25 * layout_stress(&g, &start));
        // A path drawn well has no crossings.
        assert_eq!(crossing_number_estimate(&g, &p), 0);
    }

    /// Fruchterman-Reingold has no monotonicity to check, so what is checked
    /// is what it does promise: a bounded drawing with no two vertices on top
    /// of each other, and edges pulled to roughly the ideal length.
    #[test]
    fn fruchterman_reingold_separates_and_settles() {
        let mut rng = Rng::new(0x_F1EE);
        for _ in 0..40 {
            let n = 2 + pick(&mut rng, 10);
            let g = random_graph(n, 0.25 + 0.4 * rng.next_f64(), &mut rng);
            let p = fruchterman_reingold(&g, 300, &mut rng);
            assert_eq!(p.len(), n);
            for q in &p {
                assert!(q.x.is_finite() && q.y.is_finite(), "the layout diverged");
            }
            for i in 0..n {
                for j in i + 1..n {
                    assert!(p[i].distance_to(&p[j]) > 1e-6, "vertices {i} and {j} coincide");
                }
            }
        }
        // On a single edge the two vertices settle near the ideal separation
        // k = sqrt(area / n), where repulsion and attraction balance.
        let mut e = Graph::new(2, false);
        e.add_edge(0, 1, 1.0);
        let p = fruchterman_reingold(&e, 2000, &mut rng);
        let k = (2.0f64).sqrt() / (2.0f64).sqrt();
        assert!(
            (p[0].distance_to(&p[1]) - k).abs() < 0.35 * k,
            "the spring did not settle near its rest length"
        );
    }

    /// The three statements a Reingold-Tilford drawing is defined by: depth
    /// gives the height, no two subtrees overlap, and a parent is centred
    /// over its outermost children.
    #[test]
    fn tree_layout_has_its_defining_properties() {
        let mut rng = Rng::new(0x_77EE);
        for _ in 0..80 {
            let n = 1 + pick(&mut rng, 24);
            // A random rooted tree: every vertex after the first attaches to
            // an earlier one.
            let mut g = Graph::new(n, false);
            for v in 1..n {
                g.add_edge(pick(&mut rng, v), v, 1.0);
            }
            let p = tree_layout_reingold_tilford(&g, 0);
            let (children, depth) = rooted_children(&g, 0);

            for v in 0..n {
                assert!(close(p[v].y, -(depth[v] as f64)), "depth is not the height");
            }
            // Nothing at the same depth is closer than the separation.
            for u in 0..n {
                for v in u + 1..n {
                    if depth[u] == depth[v] {
                        assert!(
                            (p[u].x - p[v].x).abs() >= 1.0 - 1e-9,
                            "vertices {u} and {v} overlap at depth {}",
                            depth[u]
                        );
                    }
                }
            }
            // Each parent sits midway between its first and last child.
            for v in 0..n {
                if let (Some(&first), Some(&last)) = (children[v].first(), children[v].last()) {
                    assert!(
                        close(p[v].x, (p[first].x + p[last].x) / 2.0),
                        "vertex {v} is not centred over its children"
                    );
                }
            }
            // Children keep their order left to right.
            for v in 0..n {
                for w in children[v].windows(2) {
                    assert!(p[w[0]].x < p[w[1]].x, "children of {v} are out of order");
                }
            }
            // A tree drawing has no crossings; that is the point of it.
            assert_eq!(crossing_number_estimate(&g, &p), 0, "the tree drawing crosses itself");
        }

        // A complete binary tree is symmetric about its root.
        let t = binary_tree(15);
        let p = tree_layout_reingold_tilford(&t, 0);
        assert!(close(p[0].x, 0.0));
        for (l, r) in [(1usize, 2usize), (3, 6), (4, 5), (7, 14), (8, 13)] {
            assert!(close(p[l].x, -p[r].x), "the drawing is not symmetric at ({l}, {r})");
        }
        // The bottom row is packed at exactly the separation, so a complete
        // tree is as narrow as it can be.
        let bottom: Vec<f64> = (7..15).map(|v| p[v].x).collect();
        for w in bottom.windows(2) {
            assert!(close(w[1] - w[0], 1.0), "the leaves are not packed tightly");
        }
    }

    /// A layered drawing must have every arc pointing downward, and the layer
    /// of a vertex must be the length of the longest path reaching it.
    #[test]
    fn sugiyama_layers_point_downward() {
        let mut rng = Rng::new(0x_5461);
        for _ in 0..80 {
            let n = 1 + pick(&mut rng, 12);
            let mut g = Graph::new(n, true);
            for u in 0..n {
                for v in u + 1..n {
                    if rng.next_f64() < 0.3 {
                        g.add_edge(u, v, 1.0);
                    }
                }
            }
            let p = sugiyama_layered(&g);
            // Longest incoming path, computed independently.
            let mut want = vec![0usize; n];
            for v in 0..n {
                for u in 0..n {
                    if g.adj[u].iter().any(|&(w, _)| w == v) {
                        want[v] = want[v].max(want[u] + 1);
                    }
                }
            }
            for v in 0..n {
                assert!(close(p[v].y, -(want[v] as f64)), "vertex {v} is on the wrong layer");
            }
            for u in 0..n {
                for &(v, _) in &g.adj[u] {
                    assert!(p[u].y > p[v].y, "the arc {u} -> {v} does not point downward");
                }
            }
            // Within a layer the positions are 0, 1, ... with no repeats.
            let depth = want.iter().copied().max().unwrap_or(0) + 1;
            for k in 0..depth {
                let mut xs: Vec<i64> = (0..n)
                    .filter(|&v| want[v] == k)
                    .map(|v| p[v].x as i64)
                    .collect();
                let size = xs.len();
                xs.sort_unstable();
                assert_eq!(xs, (0..size as i64).collect::<Vec<_>>(), "layer {k} is not a row");
            }
        }
    }

    /// Crossing counts against the one family where the answer is a formula:
    /// a complete graph drawn with its vertices in convex position has one
    /// crossing for every four of them, since each four in convex position
    /// contribute exactly one.
    #[test]
    fn crossing_count_matches_the_convex_position_formula() {
        for n in 3..=9usize {
            let p = circular_layout(n);
            let want = n * (n - 1) * (n - 2) * (n - 3) / 24;
            assert_eq!(
                crossing_number_estimate(&complete_graph(n), &p),
                want,
                "K_{n} in convex position"
            );
        }
        // Sharing an endpoint is not a crossing, however the drawing looks.
        let mut star = Graph::new(5, false);
        for v in 1..5 {
            star.add_edge(0, v, 1.0);
        }
        let p = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.0, -1.0),
        ];
        assert_eq!(crossing_number_estimate(&star, &p), 0);
        // A four-cycle drawn as a bow tie crosses once; drawn as a square, not
        // at all. The graph is the same both times.
        let c4 = cycle_graph(4);
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        assert_eq!(crossing_number_estimate(&c4, &square), 0);
        let bow = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];
        assert_eq!(crossing_number_estimate(&c4, &bow), 1);
    }

    /// The blocks must partition the edges, and a block is a single edge
    /// exactly when that edge is a bridge.
    #[test]
    fn biconnected_components_partition_the_edges() {
        let mut rng = Rng::new(0x_B10C);
        for _ in 0..120 {
            let n = 1 + pick(&mut rng, 12);
            let g = random_graph(n, 0.15 + 0.4 * rng.next_f64(), &mut rng);
            let blocks = biconnected_components(&g);
            let mut seen: Vec<(usize, usize)> = blocks
                .iter()
                .flatten()
                .map(|&(u, v)| (u.min(v), u.max(v)))
                .collect();
            let total = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), total, "an edge is in two blocks");
            let mut want: Vec<(usize, usize)> =
                g.edges().iter().map(|&(u, v, _)| (u.min(v), u.max(v))).collect();
            want.sort_unstable();
            want.dedup();
            assert_eq!(seen, want, "the blocks do not cover the edges");

            let bridges: BTreeSet<(usize, usize)> =
                g.bridges().iter().map(|&(u, v)| (u.min(v), u.max(v))).collect();
            for block in &blocks {
                let is_bridge = block.len() == 1;
                let e = (block[0].0.min(block[0].1), block[0].0.max(block[0].1));
                if is_bridge {
                    assert!(bridges.contains(&e), "a lone block is not a bridge: {e:?}");
                }
            }
            for b in &bridges {
                assert!(
                    blocks.iter().any(|bl| bl.len() == 1
                        && (bl[0].0.min(bl[0].1), bl[0].0.max(bl[0].1)) == *b),
                    "the bridge {b:?} is not its own block"
                );
            }
        }
    }

    /// Planarity decided from the definition: does some cyclic ordering of
    /// the edges around each vertex give a surface of genus zero?
    ///
    /// Tracing the faces of a rotation system and reading Euler's formula
    /// backwards gives the genus of the surface that rotation embeds the
    /// graph in, and planar means genus zero. Enumerating every rotation
    /// system is hopeless beyond a handful of vertices, which is why
    /// Demoucron exists -- but it shares no reasoning with Demoucron at all,
    /// which is what makes it worth checking against.
    ///
    /// `None` when the enumeration would be too large to run.
    fn planar_by_rotation_enumeration(g: &Graph, budget: u64) -> Option<bool> {
        let n = g.n;
        let adj: Vec<Vec<usize>> = (0..n)
            .map(|u| {
                let mut ns: Vec<usize> =
                    g.adj[u].iter().map(|&(v, _)| v).filter(|&v| v != u).collect();
                ns.sort_unstable();
                ns.dedup();
                ns
            })
            .collect();
        for comp in g.connected_components() {
            if comp.len() < 3 {
                continue;
            }
            // Work in local indices so the face walk is array lookups.
            let mut local_of = vec![usize::MAX; n];
            for (i, &v) in comp.iter().enumerate() {
                local_of[v] = i;
            }
            let local: Vec<Vec<usize>> = comp
                .iter()
                .map(|&u| adj[u].iter().map(|&v| local_of[v]).filter(|&v| v != usize::MAX).collect())
                .collect();
            let e: usize = local.iter().map(Vec::len).sum::<usize>() / 2;
            let want = e as i64 - comp.len() as i64 + 2;
            // The rotations to try at each vertex: the neighbours after the
            // first, in every order. Fixing the first breaks the cyclic
            // symmetry, which would otherwise multiply the work by the degree.
            let choices: Vec<Vec<Vec<usize>>> = local
                .iter()
                .map(|ns| {
                    if ns.len() <= 2 {
                        vec![ns.clone()]
                    } else {
                        crate::discrete::combinatorics::permutations_iter(&ns[1..])
                            .map(|rest| {
                                let mut r = vec![ns[0]];
                                r.extend(rest);
                                r
                            })
                            .collect()
                    }
                })
                .collect();
            let total: u64 = choices.iter().map(|c| c.len() as u64).product();
            if total > budget {
                return None;
            }
            // Where each neighbour sits in each candidate rotation, so the
            // walk never searches.
            let where_in: Vec<Vec<Vec<usize>>> = choices
                .iter()
                .map(|opts| {
                    opts.iter()
                        .map(|r| {
                            let mut w = vec![usize::MAX; comp.len()];
                            for (k, &x) in r.iter().enumerate() {
                                w[x] = k;
                            }
                            w
                        })
                        .collect()
                })
                .collect();
            let mut index = vec![0usize; comp.len()];
            let mut found = false;
            loop {
                if trace_faces(&choices, &where_in, &index) == want {
                    found = true;
                    break;
                }
                let mut k = 0;
                while k < comp.len() {
                    index[k] += 1;
                    if index[k] < choices[k].len() {
                        break;
                    }
                    index[k] = 0;
                    k += 1;
                }
                if k == comp.len() {
                    break;
                }
            }
            if !found {
                return Some(false);
            }
        }
        Some(true)
    }

    /// The number of faces one rotation system traces out.
    fn trace_faces(
        choices: &[Vec<Vec<usize>>],
        where_in: &[Vec<Vec<usize>>],
        index: &[usize],
    ) -> i64 {
        let n = index.len();
        let rot: Vec<&Vec<usize>> = (0..n).map(|i| &choices[i][index[i]]).collect();
        let pos: Vec<&Vec<usize>> = (0..n).map(|i| &where_in[i][index[i]]).collect();
        let mut seen: Vec<Vec<bool>> = rot.iter().map(|r| vec![false; r.len()]).collect();
        let mut faces = 0i64;
        for u in 0..n {
            for slot in 0..rot[u].len() {
                if seen[u][slot] {
                    continue;
                }
                faces += 1;
                // Walk the face: arriving at `b` from `a`, leave along the
                // neighbour that follows `a` in `b`'s rotation.
                let (mut a, mut k) = (u, slot);
                loop {
                    if seen[a][k] {
                        break;
                    }
                    seen[a][k] = true;
                    let b = rot[a][k];
                    let j = pos[b][a];
                    k = (j + 1) % rot[b].len();
                    a = b;
                }
            }
        }
        faces
    }

    /// Demoucron against exhaustive enumeration of rotation systems: two
    /// algorithms with nothing in common beyond the answer they compute.
    #[test]
    fn planarity_agrees_with_rotation_system_enumeration() {
        let mut rng = Rng::new(0x_9074);
        let mut checked = 0;
        for _ in 0..1500 {
            let n = 1 + pick(&mut rng, 8);
            let g = random_graph(n, 0.15 + 0.55 * rng.next_f64(), &mut rng);
            let Some(want) = planar_by_rotation_enumeration(&g, 60_000) else {
                continue;
            };
            checked += 1;
            assert_eq!(
                planarity_test(&g),
                want,
                "Demoucron and the genus computation disagree on a graph of {n} vertices: {:?}",
                g.edges().iter().map(|&(u, v, _)| (u, v)).collect::<Vec<_>>()
            );
        }
        assert!(checked > 250, "only {checked} graphs were small enough to cross-check");
        // And on the graphs the whole subject is about.
        for g in [complete_graph(5), complete_bipartite(3, 3), complete_graph(4), petersen_graph()]
        {
            if let Some(want) = planar_by_rotation_enumeration(&g, 100_000) {
                assert_eq!(planarity_test(&g), want);
            }
        }
    }

    /// Planarity against the graphs it is defined by, and against the
    /// properties any correct test must have.
    #[test]
    fn planarity_matches_the_known_families() {
        // Kuratowski's two, and the graphs built from them.
        assert!(!planarity_test(&complete_graph(5)));
        assert!(!planarity_test(&complete_bipartite(3, 3)));
        assert!(!planarity_test(&complete_graph(6)));
        assert!(!planarity_test(&petersen_graph()));
        // Subdivision preserves planarity in both directions, which is the
        // whole content of Kuratowski's theorem: the subdivided K5 has no K5
        // in it as a subgraph at all.
        assert!(!planarity_test(&subdivide(&complete_graph(5))));
        assert!(!planarity_test(&subdivide(&complete_bipartite(3, 3))));
        assert!(!planarity_test(&subdivide(&subdivide(&complete_bipartite(3, 3)))));

        // Planar families.
        assert!(planarity_test(&complete_graph(4)));
        assert!(planarity_test(&complete_bipartite(2, 5)));
        assert!(planarity_test(&Graph::new(6, false)));
        for n in 3..=10 {
            assert!(planarity_test(&cycle_graph(n)), "C_{n}");
            assert!(planarity_test(&path_graph(n)));
            assert!(planarity_test(&binary_tree(n)));
        }
        for (w, h) in [(2usize, 2usize), (3, 3), (4, 5), (2, 7)] {
            assert!(planarity_test(&grid_graph(w, h)), "the {w} by {h} grid");
            assert!(planarity_test(&subdivide(&grid_graph(w, h))));
        }
        // The wheel and the prism, both planar, both three-connected.
        let mut wheel = cycle_graph(7);
        let mut w8 = Graph::new(8, false);
        for (u, v, _) in wheel.edges() {
            w8.add_edge(u, v, 1.0);
        }
        for v in 0..7 {
            w8.add_edge(7, v, 1.0);
        }
        wheel = w8;
        assert!(planarity_test(&wheel));
        let mut prism = Graph::new(6, false);
        for i in 0..3 {
            prism.add_edge(i, (i + 1) % 3, 1.0);
            prism.add_edge(3 + i, 3 + (i + 1) % 3, 1.0);
            prism.add_edge(i, 3 + i, 1.0);
        }
        assert!(planarity_test(&prism));

        // The boundary: removing any single edge from either Kuratowski graph
        // makes it planar, so the test must not be rejecting them for some
        // coarser reason.
        for base in [complete_graph(5), complete_bipartite(3, 3)] {
            let edges = base.edges();
            for skip in 0..edges.len() {
                let mut h = Graph::new(base.n, false);
                for (i, &(u, v, _)) in edges.iter().enumerate() {
                    if i != skip {
                        h.add_edge(u, v, 1.0);
                    }
                }
                assert!(planarity_test(&h), "K minus an edge should be planar");
            }
        }

        // Random graphs: planarity is closed under taking subgraphs, and a
        // graph over Euler's bound cannot be planar.
        let mut rng = Rng::new(0x_71A4);
        for _ in 0..200 {
            let n = 1 + pick(&mut rng, 10);
            let g = random_graph(n, 0.1 + 0.5 * rng.next_f64(), &mut rng);
            let planar = planarity_test(&g);
            let m = g.edge_count();
            if n >= 3 && m > 3 * n - 6 {
                assert!(!planar, "over Euler's bound but reported planar");
            }
            if planar {
                // Every edge-deleted subgraph is planar too.
                let edges = g.edges();
                for skip in 0..edges.len() {
                    let mut h = Graph::new(n, false);
                    for (i, &(u, v, _)) in edges.iter().enumerate() {
                        if i != skip {
                            h.add_edge(u, v, 1.0);
                        }
                    }
                    assert!(planarity_test(&h), "a subgraph of a planar graph is not planar");
                }
                // And so is the subdivision.
                assert!(planarity_test(&subdivide(&g)));
            } else {
                // A non-planar graph stays non-planar when subdivided.
                assert!(!planarity_test(&subdivide(&g)));
            }
        }
    }

    /// The embedding must be a genuine one: Euler's formula, every edge on
    /// exactly two faces, and every face a closed walk in the graph.
    #[test]
    fn planar_embedding_satisfies_eulers_formula() {
        let mut prism = Graph::new(6, false);
        for i in 0..3 {
            prism.add_edge(i, (i + 1) % 3, 1.0);
            prism.add_edge(3 + i, 3 + (i + 1) % 3, 1.0);
            prism.add_edge(i, 3 + i, 1.0);
        }
        let mut cases = vec![
            complete_graph(3),
            complete_graph(4),
            cycle_graph(6),
            grid_graph(3, 3),
            grid_graph(2, 4),
            complete_bipartite(2, 4),
            prism,
        ];
        let mut rng = Rng::new(0x_E01E);
        let mut extra = 0;
        while extra < 60 {
            let n = 3 + pick(&mut rng, 8);
            let g = random_graph(n, 0.2 + 0.4 * rng.next_f64(), &mut rng);
            if biconnected_components(&g).len() == 1 && g.is_connected() && planarity_test(&g) {
                cases.push(g);
                extra += 1;
            }
        }
        for g in cases {
            let faces = planar_embedding_small(&g).expect("these are all planar");
            let v = g.n;
            let e = g.edge_count();
            assert_eq!(
                v as i64 - e as i64 + faces.len() as i64,
                2,
                "Euler's formula fails: V {v}, E {e}, F {}",
                faces.len()
            );
            let adj: Vec<BTreeSet<usize>> = (0..v)
                .map(|u| g.adj[u].iter().map(|&(w, _)| w).collect())
                .collect();
            let mut border: std::collections::BTreeMap<(usize, usize), usize> =
                std::collections::BTreeMap::new();
            for f in &faces {
                assert!(f.len() >= 3, "a face of a simple graph has at least three sides");
                for i in 0..f.len() {
                    let (a, b) = (f[i], f[(i + 1) % f.len()]);
                    assert!(adj[a].contains(&b), "the face uses the non-edge ({a}, {b})");
                    *border.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
            assert_eq!(border.len(), e, "not every edge is on a face");
            for (edge, times) in border {
                assert_eq!(times, 2, "the edge {edge:?} borders {times} faces, not two");
            }
        }
        // A non-planar graph has no embedding to return.
        assert!(planar_embedding_small(&complete_graph(5)).is_none());
        assert!(planar_embedding_small(&complete_bipartite(3, 3)).is_none());
    }
}
