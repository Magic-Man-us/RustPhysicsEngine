//! Shortest paths, spanning trees, and tours.
//!
//! Distances are `f64` and an unreachable vertex is `f64::INFINITY`, so the
//! results compose without an `Option` at every step. Predecessor arrays use
//! `None` for the source and for unreachable vertices alike; the distance
//! distinguishes the two.

use crate::exact::bigint::BigInt;
use crate::graph::core::Graph;
use crate::linalg::matrix::Matrix;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Reported when a graph reachable from the source contains a negative cycle,
/// which makes "shortest" meaningless there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeCycle {
    /// A vertex known to lie on or downstream of the negative cycle.
    pub witness: usize,
}

impl std::fmt::Display for NegativeCycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "negative cycle reachable at vertex {}", self.witness)
    }
}

impl std::error::Error for NegativeCycle {}

/// A heap entry ordered by ascending key.
///
/// `BinaryHeap` is a max-heap and `f64` is not `Ord`, so this wraps both
/// problems: the comparison is reversed, and the key is compared with
/// `total_cmp`.
///
/// `partial_cmp(..).unwrap_or(Equal)` is the obvious thing to write here and
/// is wrong: it makes a NaN key compare equal to every other key, which is not
/// transitive, so `Ord`'s contract is broken and the heap can return items out
/// of order -- a NaN pushed among 1, 2 and 3 came back second. `total_cmp` is
/// a genuine total order on every `f64`, and puts a positive NaN above
/// infinity, so a NaN key settles last instead of corrupting the search.
#[derive(PartialEq)]
struct MinKey(f64, usize);

impl Eq for MinKey {}

impl Ord for MinKey {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then_with(|| other.1.cmp(&self.1))
    }
}

impl PartialOrd for MinKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Rebuilds the path to `t` from a predecessor array, or `None` if `t` was
/// never reached.
fn rebuild(prev: &[Option<usize>], s: usize, t: usize) -> Option<Vec<usize>> {
    if s == t {
        return Some(vec![s]);
    }
    let mut path = vec![t];
    let mut cur = t;
    while let Some(p) = prev[cur] {
        path.push(p);
        cur = p;
        if cur == s {
            path.reverse();
            return Some(path);
        }
    }
    None
}

/// Single-source shortest paths with non-negative weights, by Dijkstra.
///
/// Returns the distances and the predecessor array. Unreached vertices have
/// distance `f64::INFINITY` and no predecessor.
///
/// # Panics
/// Panics if any weight is negative, where the algorithm is simply wrong
/// rather than merely slow -- use [`bellman_ford`] instead.
#[must_use]
pub fn dijkstra(g: &Graph, s: usize) -> (Vec<f64>, Vec<Option<usize>>) {
    assert!(
        g.edges().iter().all(|&(_, _, w)| w >= 0.0),
        "dijkstra needs non-negative weights"
    );
    let mut dist = vec![f64::INFINITY; g.n];
    let mut prev = vec![None; g.n];
    let mut heap = BinaryHeap::new();
    dist[s] = 0.0;
    heap.push(MinKey(0.0, s));
    while let Some(MinKey(d, v)) = heap.pop() {
        // Lazy deletion: a stale entry is one whose key is worse than the
        // settled distance.
        if d > dist[v] {
            continue;
        }
        for &(w, weight) in &g.adj[v] {
            let cand = d + weight;
            if cand < dist[w] {
                dist[w] = cand;
                prev[w] = Some(v);
                heap.push(MinKey(cand, w));
            }
        }
    }
    (dist, prev)
}

/// The shortest path from `s` to `t` and its length, or `None` if `t` is
/// unreachable.
#[must_use]
pub fn dijkstra_target(g: &Graph, s: usize, t: usize) -> Option<(f64, Vec<usize>)> {
    let (dist, prev) = dijkstra(g, s);
    if !dist[t].is_finite() {
        return None;
    }
    Some((dist[t], rebuild(&prev, s, t)?))
}

/// Single-source shortest paths allowing negative weights, by Bellman-Ford.
///
/// # Errors
/// Returns [`NegativeCycle`] when a cycle of negative total weight is
/// reachable from `s`, which is detected by one relaxation pass beyond the
/// `n - 1` that suffice when none exists.
pub fn bellman_ford(
    g: &Graph,
    s: usize,
) -> Result<(Vec<f64>, Vec<Option<usize>>), NegativeCycle> {
    let mut dist = vec![f64::INFINITY; g.n];
    let mut prev = vec![None; g.n];
    dist[s] = 0.0;
    // Every arc, in both directions for an undirected graph.
    let arcs = directed_arcs(g);
    for _ in 1..g.n.max(1) {
        let mut changed = false;
        for &(u, v, w) in &arcs {
            if dist[u].is_finite() && dist[u] + w < dist[v] {
                dist[v] = dist[u] + w;
                prev[v] = Some(u);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for &(u, v, w) in &arcs {
        if dist[u].is_finite() && dist[u] + w < dist[v] {
            return Err(NegativeCycle { witness: v });
        }
    }
    Ok((dist, prev))
}

/// Every arc as `(tail, head, weight)`, with an undirected edge appearing in
/// both directions.
fn directed_arcs(g: &Graph) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for u in 0..g.n {
        for &(v, w) in &g.adj[u] {
            out.push((u, v, w));
        }
    }
    out
}

/// All-pairs shortest paths by Floyd-Warshall, `O(n^3)`.
///
/// Entry `(i, j)` is the distance, `f64::INFINITY` when unreachable. Negative
/// cycles are not detected here; a negative diagonal entry in the result is
/// the sign of one.
#[must_use]
pub fn floyd_warshall(g: &Graph) -> Matrix {
    let n = g.n;
    let mut d = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d.set(i, j, if i == j { 0.0 } else { f64::INFINITY });
        }
    }
    for (u, v, w) in directed_arcs(g) {
        if w < d.get(u, v) {
            d.set(u, v, w);
        }
    }
    for k in 0..n {
        for i in 0..n {
            let dik = d.get(i, k);
            if !dik.is_finite() {
                continue;
            }
            for j in 0..n {
                let cand = dik + d.get(k, j);
                if cand < d.get(i, j) {
                    d.set(i, j, cand);
                }
            }
        }
    }
    d
}

/// All-pairs shortest paths by Johnson's algorithm: a Bellman-Ford pass from a
/// virtual source supplies potentials that make every weight non-negative,
/// then one Dijkstra per vertex.
///
/// Faster than Floyd-Warshall on sparse graphs, and unlike plain Dijkstra it
/// tolerates negative weights.
///
/// # Errors
/// Returns [`NegativeCycle`] if the graph contains one.
pub fn johnson(g: &Graph) -> Result<Matrix, NegativeCycle> {
    let n = g.n;
    // The virtual source reaches every vertex at zero cost, so its
    // Bellman-Ford distances are valid potentials for the whole graph.
    let mut aug = Graph::new(n + 1, true);
    for (u, v, w) in directed_arcs(g) {
        aug.add_edge(u, v, w);
    }
    for v in 0..n {
        aug.add_edge(n, v, 0.0);
    }
    let (h, _) = bellman_ford(&aug, n)?;

    // Reweight: w'(u,v) = w(u,v) + h(u) - h(v) >= 0 by the triangle
    // inequality, and shortest paths are preserved because the potentials
    // telescope along any path.
    let mut rew = Graph::new(n, true);
    for (u, v, w) in directed_arcs(g) {
        rew.add_edge(u, v, (w + h[u] - h[v]).max(0.0));
    }
    let mut out = Matrix::zeros(n, n);
    for u in 0..n {
        let (d, _) = dijkstra(&rew, u);
        for v in 0..n {
            let real = if d[v].is_finite() {
                d[v] - h[u] + h[v]
            } else {
                f64::INFINITY
            };
            out.set(u, v, real);
        }
    }
    Ok(out)
}

/// A* search with the heuristic `h`.
///
/// Returns the path and its true length, or `None` if `t` is unreachable. The
/// result is optimal exactly when `h` is admissible -- never overestimating
/// the remaining distance -- and the search is efficient when `h` is also
/// consistent. An inadmissible heuristic still terminates but may return a
/// suboptimal path, which is the caller's trade to make.
///
/// # Panics
/// Panics if any weight is negative.
pub fn a_star(
    g: &Graph,
    s: usize,
    t: usize,
    h: &dyn Fn(usize) -> f64,
) -> Option<(f64, Vec<usize>)> {
    assert!(
        g.edges().iter().all(|&(_, _, w)| w >= 0.0),
        "a_star needs non-negative weights"
    );
    let mut dist = vec![f64::INFINITY; g.n];
    let mut prev = vec![None; g.n];
    let mut heap = BinaryHeap::new();
    dist[s] = 0.0;
    heap.push(MinKey(h(s), s));
    while let Some(MinKey(f, v)) = heap.pop() {
        if v == t {
            return Some((dist[t], rebuild(&prev, s, t)?));
        }
        if f > dist[v] + h(v) {
            continue;
        }
        for &(w, weight) in &g.adj[v] {
            let cand = dist[v] + weight;
            if cand < dist[w] {
                dist[w] = cand;
                prev[w] = Some(v);
                heap.push(MinKey(cand + h(w), w));
            }
        }
    }
    None
}

/// Dijkstra from both ends at once, alternating between them.
///
/// Both searches settle vertices; the answer is the best path through any
/// vertex either has reached, and the search stops once the two settled
/// radii sum to at least the best path found. On a graph where the reachable
/// set grows with the radius, this settles roughly the square root of the
/// vertices a one-sided search would.
///
/// # Panics
/// Panics if any weight is negative.
#[must_use]
pub fn bidirectional_dijkstra(g: &Graph, s: usize, t: usize) -> Option<(f64, Vec<usize>)> {
    assert!(
        g.edges().iter().all(|&(_, _, w)| w >= 0.0),
        "bidirectional_dijkstra needs non-negative weights"
    );
    if s == t {
        return Some((0.0, vec![s]));
    }
    let rev = g.reverse();
    let mut df = vec![f64::INFINITY; g.n];
    let mut db = vec![f64::INFINITY; g.n];
    let mut pf: Vec<Option<usize>> = vec![None; g.n];
    let mut pb: Vec<Option<usize>> = vec![None; g.n];
    let mut hf = BinaryHeap::new();
    let mut hb = BinaryHeap::new();
    df[s] = 0.0;
    db[t] = 0.0;
    hf.push(MinKey(0.0, s));
    hb.push(MinKey(0.0, t));
    let mut best = f64::INFINITY;
    let mut meet = usize::MAX;
    let (mut rf, mut rb) = (0.0f64, 0.0f64);

    while !hf.is_empty() || !hb.is_empty() {
        // Stop once no unsettled path can beat what has been found.
        if rf + rb >= best {
            break;
        }
        // Expand whichever side has the smaller frontier radius.
        let forward = match (hf.peek(), hb.peek()) {
            (Some(a), Some(b)) => a.0 <= b.0,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let (heap, dist, other, prev, adj, radius) = if forward {
            (&mut hf, &mut df, &db, &mut pf, &g.adj, &mut rf)
        } else {
            (&mut hb, &mut db, &df, &mut pb, &rev.adj, &mut rb)
        };
        let Some(MinKey(d, v)) = heap.pop() else { break };
        if d > dist[v] {
            continue;
        }
        *radius = d;
        if other[v].is_finite() && d + other[v] < best {
            best = d + other[v];
            meet = v;
        }
        for &(w, weight) in &adj[v] {
            let cand = d + weight;
            if cand < dist[w] {
                dist[w] = cand;
                prev[w] = Some(v);
                heap.push(MinKey(cand, w));
            }
        }
    }
    if meet == usize::MAX {
        return None;
    }
    // Splice: the forward half ends at the meeting point and the backward half
    // starts there, so drop the duplicate.
    let mut path = rebuild(&pf, s, meet)?;
    let back = rebuild(&pb, t, meet)?;
    path.extend(back.into_iter().rev().skip(1));
    Some((best, path))
}

/// The `k` shortest loopless paths from `s` to `t`, by Yen's algorithm.
///
/// Returns them in increasing length, and may return fewer than `k` when
/// fewer exist. Each candidate is found by forcing a shared prefix with an
/// already-accepted path and forbidding the arc it took next, which is what
/// keeps the results distinct and loopless.
#[must_use]
pub fn k_shortest_paths_yen(g: &Graph, s: usize, t: usize, k: usize) -> Vec<(f64, Vec<usize>)> {
    let mut accepted: Vec<(f64, Vec<usize>)> = Vec::new();
    let Some(first) = dijkstra_target(g, s, t) else {
        return accepted;
    };
    accepted.push(first);
    let mut candidates: Vec<(f64, Vec<usize>)> = Vec::new();

    while accepted.len() < k {
        let last = accepted.last().unwrap().1.clone();
        for i in 0..last.len().saturating_sub(1) {
            let spur = last[i];
            let root = &last[..=i];
            // Remove the arcs that would repeat an accepted path's next step,
            // and the root's own interior vertices, which keeps the spur
            // loopless.
            let mut banned_arcs: Vec<(usize, usize)> = Vec::new();
            for (_, p) in &accepted {
                if p.len() > i + 1 && p[..=i] == *root {
                    banned_arcs.push((p[i], p[i + 1]));
                }
            }
            let banned_vertices: Vec<usize> = root[..i].to_vec();
            let mut sub = Graph::new(g.n, true);
            for (u, v, w) in directed_arcs(g) {
                if banned_vertices.contains(&u) || banned_vertices.contains(&v) {
                    continue;
                }
                if banned_arcs.contains(&(u, v)) {
                    continue;
                }
                sub.add_edge(u, v, w);
            }
            let Some((spur_cost, spur_path)) = dijkstra_target(&sub, spur, t) else {
                continue;
            };
            let root_cost: f64 = root
                .windows(2)
                .map(|w| arc_weight(g, w[0], w[1]).unwrap_or(f64::INFINITY))
                .sum();
            let mut full = root[..i].to_vec();
            full.extend(spur_path);
            let total = root_cost + spur_cost;
            if !accepted.iter().any(|(_, p)| *p == full)
                && !candidates.iter().any(|(_, p)| *p == full)
            {
                candidates.push((total, full));
            }
        }
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        accepted.push(candidates.remove(0));
    }
    accepted
}

/// The least weight among the arcs from `u` to `v`, if any.
fn arc_weight(g: &Graph, u: usize, v: usize) -> Option<f64> {
    g.adj[u]
        .iter()
        .filter(|&&(t, _)| t == v)
        .map(|&(_, w)| w)
        .min_by(|a: &f64, b: &f64| a.total_cmp(b))
}

/// The widest path: the one whose narrowest edge is as wide as possible.
///
/// Also called the bottleneck shortest path or the maximum capacity path.
/// Dijkstra with `min` in place of `+` and `max` in place of `min`, which is
/// valid because `min` is monotone in the same way `+` is.
#[must_use]
pub fn widest_path(g: &Graph, s: usize, t: usize) -> Option<(f64, Vec<usize>)> {
    let mut width = vec![f64::NEG_INFINITY; g.n];
    let mut prev: Vec<Option<usize>> = vec![None; g.n];
    let mut heap = BinaryHeap::new();
    width[s] = f64::INFINITY;
    // MinKey orders ascending, so negate to pop the widest first.
    heap.push(MinKey(f64::NEG_INFINITY, s));
    while let Some(MinKey(negw, v)) = heap.pop() {
        if -negw < width[v] {
            continue;
        }
        if v == t {
            break;
        }
        for &(w, cap) in &g.adj[v] {
            let cand = width[v].min(cap);
            if cand > width[w] {
                width[w] = cand;
                prev[w] = Some(v);
                heap.push(MinKey(-cand, w));
            }
        }
    }
    if width[t] == f64::NEG_INFINITY {
        return None;
    }
    Some((width[t], rebuild(&prev, s, t)?))
}

/// The minimax path: the one whose widest edge is as narrow as possible.
///
/// The dual of [`widest_path`], and the path a minimum spanning tree gives
/// between any two vertices.
#[must_use]
pub fn minimax_path(g: &Graph, s: usize, t: usize) -> Option<(f64, Vec<usize>)> {
    let mut bottleneck = vec![f64::INFINITY; g.n];
    let mut prev: Vec<Option<usize>> = vec![None; g.n];
    let mut heap = BinaryHeap::new();
    bottleneck[s] = f64::NEG_INFINITY;
    heap.push(MinKey(f64::NEG_INFINITY, s));
    while let Some(MinKey(d, v)) = heap.pop() {
        if d > bottleneck[v] {
            continue;
        }
        if v == t {
            break;
        }
        for &(w, cost) in &g.adj[v] {
            let cand = bottleneck[v].max(cost);
            if cand < bottleneck[w] {
                bottleneck[w] = cand;
                prev[w] = Some(v);
                heap.push(MinKey(cand, w));
            }
        }
    }
    if !bottleneck[t].is_finite() && bottleneck[t] > 0.0 {
        return None;
    }
    if bottleneck[t] == f64::INFINITY {
        return None;
    }
    Some((bottleneck[t].max(0.0), rebuild(&prev, s, t)?))
}

/// Shortest distances from `s` in a DAG, by relaxing in topological order.
///
/// Linear time and correct with negative weights, neither of which Dijkstra
/// manages.
///
/// # Panics
/// Panics if the graph is not a DAG.
#[must_use]
pub fn dag_shortest(g: &Graph, s: usize) -> Vec<f64> {
    dag_extreme(g, s, true)
}

/// Longest distances from `s` in a DAG.
///
/// Longest path is NP-hard in general but linear on a DAG, since the
/// topological order removes any need to revisit.
///
/// # Panics
/// Panics if the graph is not a DAG.
#[must_use]
pub fn dag_longest(g: &Graph, s: usize) -> Vec<f64> {
    dag_extreme(g, s, false)
}

fn dag_extreme(g: &Graph, s: usize, shortest: bool) -> Vec<f64> {
    let order = g.topological_sort().expect("dag_shortest needs a DAG");
    let unreached = if shortest {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    let mut dist = vec![unreached; g.n];
    dist[s] = 0.0;
    for &v in &order {
        if dist[v] == unreached {
            continue;
        }
        for &(w, weight) in &g.adj[v] {
            let cand = dist[v] + weight;
            let better = if shortest {
                cand < dist[w]
            } else {
                cand > dist[w]
            };
            if better {
                dist[w] = cand;
            }
        }
    }
    dist
}

/// The number of distinct directed paths from `s` to `t` in a DAG.
///
/// Exact, because the count grows exponentially: a grid DAG of side `n` has
/// `C(2n, n)` paths, past `u64` before `n = 34`.
///
/// # Panics
/// Panics if the graph is not a DAG.
#[must_use]
pub fn count_paths_dag(g: &Graph, s: usize, t: usize) -> BigInt {
    let order = g.topological_sort().expect("count_paths_dag needs a DAG");
    let mut count = vec![BigInt::zero(); g.n];
    count[s] = BigInt::one();
    for &v in &order {
        if count[v].is_zero() {
            continue;
        }
        let here = count[v].clone();
        for &(w, _) in &g.adj[v] {
            count[w] = count[w].add(&here);
        }
    }
    count[t].clone()
}

/// The reachability matrix: `[i][j]` is true when `j` is reachable from `i`.
///
/// Every vertex reaches itself.
#[must_use]
pub fn transitive_closure(g: &Graph) -> Vec<Vec<bool>> {
    let n = g.n;
    let mut r = vec![vec![false; n]; n];
    for (i, row) in r.iter_mut().enumerate() {
        row[i] = true;
    }
    for (u, v, _) in directed_arcs(g) {
        r[u][v] = true;
    }
    // Warshall.
    for k in 0..n {
        for i in 0..n {
            if r[i][k] {
                for j in 0..n {
                    if r[k][j] {
                        r[i][j] = true;
                    }
                }
            }
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Spanning trees
// ---------------------------------------------------------------------------

/// A minimum spanning forest by Kruskal's algorithm: sort the edges, accept
/// each one that joins two different components.
///
/// Returns the total weight and the edges, each with `u < v`. On a
/// disconnected graph this is a spanning forest, and the edge count is
/// `n - components` rather than `n - 1`.
#[must_use]
pub fn minimum_spanning_tree_kruskal(g: &Graph) -> (f64, Vec<(usize, usize)>) {
    let mut edges: Vec<(f64, usize, usize)> = g
        .edges()
        .into_iter()
        .filter(|&(u, v, _)| u != v)
        .map(|(u, v, w)| (w, u.min(v), u.max(v)))
        .collect();
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut ds = crate::discrete::disjoint_set::DisjointSet::new(g.n);
    let mut total = 0.0;
    let mut chosen = Vec::new();
    for (w, u, v) in edges {
        if ds.union(u, v) {
            total += w;
            chosen.push((u, v));
        }
    }
    (total, chosen)
}

/// A minimum spanning forest by Prim's algorithm: grow a tree from each
/// unvisited vertex, always taking the cheapest edge leaving it.
///
/// Returns the same weight as Kruskal on any graph, though possibly a
/// different tree when weights tie.
#[must_use]
pub fn minimum_spanning_tree_prim(g: &Graph) -> (f64, Vec<(usize, usize)>) {
    let mut in_tree = vec![false; g.n];
    let mut total = 0.0;
    let mut chosen = Vec::new();
    for root in 0..g.n {
        if in_tree[root] {
            continue;
        }
        let mut heap = BinaryHeap::new();
        in_tree[root] = true;
        for &(w, weight) in &g.adj[root] {
            heap.push((MinKey(weight, w), root));
        }
        while let Some((MinKey(weight, v), from)) = heap.pop() {
            if in_tree[v] {
                continue;
            }
            in_tree[v] = true;
            total += weight;
            chosen.push((from.min(v), from.max(v)));
            for &(w, next) in &g.adj[v] {
                if !in_tree[w] {
                    heap.push((MinKey(next, w), v));
                }
            }
        }
    }
    (total, chosen)
}

/// A minimum spanning forest by Boruvka's algorithm: every component picks its
/// own cheapest outgoing edge, and all of them are added at once.
///
/// Halves the component count per round, so `O(log n)` rounds suffice. Ties
/// are broken by edge index, which is what stops two components from each
/// picking the other's edge and forming a cycle.
#[must_use]
pub fn minimum_spanning_tree_boruvka(g: &Graph) -> (f64, Vec<(usize, usize)>) {
    let edges: Vec<(usize, usize, f64)> = g
        .edges()
        .into_iter()
        .filter(|&(u, v, _)| u != v)
        .map(|(u, v, w)| (u.min(v), u.max(v), w))
        .collect();
    let mut ds = crate::discrete::disjoint_set::DisjointSet::new(g.n);
    let mut total = 0.0;
    let mut chosen = Vec::new();
    loop {
        // Cheapest outgoing edge per component, by (weight, index).
        let mut best: Vec<Option<usize>> = vec![None; g.n];
        for (i, &(u, v, w)) in edges.iter().enumerate() {
            let (a, b) = (ds.find(u), ds.find(v));
            if a == b {
                continue;
            }
            for root in [a, b] {
                let better = match best[root] {
                    None => true,
                    Some(j) => (w, i) < (edges[j].2, j),
                };
                if better {
                    best[root] = Some(i);
                }
            }
        }
        let mut added = false;
        for root in 0..g.n {
            if let Some(i) = best[root] {
                let (u, v, w) = edges[i];
                if ds.union(u, v) {
                    total += w;
                    chosen.push((u, v));
                    added = true;
                }
            }
        }
        if !added {
            break;
        }
    }
    (total, chosen)
}

/// The second-best spanning tree: the cheapest spanning tree that differs from
/// the minimum one in at least one edge.
///
/// Found by swapping: for each non-tree edge, adding it creates one cycle, and
/// removing the heaviest tree edge on that cycle gives the cheapest tree
/// containing it. The best such swap is the answer.
///
/// Returns `None` when the graph is disconnected or has no non-tree edge, so
/// no second tree exists.
#[must_use]
pub fn second_best_mst(g: &Graph) -> Option<(f64, Vec<(usize, usize)>)> {
    let (base_cost, tree) = minimum_spanning_tree_kruskal(g);
    if tree.len() + 1 != g.n {
        return None;
    }
    // The tree as a graph, so the path between any two vertices is unique.
    let mut t = Graph::new(g.n, false);
    for &(u, v) in &tree {
        t.add_edge(u, v, arc_weight(g, u, v).unwrap_or(0.0));
    }
    let mut best: Option<(f64, (usize, usize), (usize, usize))> = None;
    for (u, v, w) in g.edges() {
        let (a, b) = (u.min(v), u.max(v));
        if u == v || tree.contains(&(a, b)) {
            continue;
        }
        // The heaviest edge on the unique tree path between the endpoints.
        let Some(path) = tree_path(&t, u, v) else {
            continue;
        };
        let Some((hu, hv, hw)) = path
            .windows(2)
            .map(|p| (p[0], p[1], arc_weight(&t, p[0], p[1]).unwrap_or(0.0)))
            .max_by(|x, y| x.2.total_cmp(&y.2))
        else {
            continue;
        };
        let delta = w - hw;
        if best.as_ref().is_none_or(|(d, _, _)| delta < *d) {
            best = Some((delta, (hu.min(hv), hu.max(hv)), (a, b)));
        }
    }
    let (delta, drop, add) = best?;
    let mut edges: Vec<(usize, usize)> = tree.into_iter().filter(|&e| e != drop).collect();
    edges.push(add);
    edges.sort_unstable();
    Some((base_cost + delta, edges))
}

/// The unique path between two vertices of a tree.
fn tree_path(t: &Graph, s: usize, e: usize) -> Option<Vec<usize>> {
    let mut prev: Vec<Option<usize>> = vec![None; t.n];
    let mut seen = vec![false; t.n];
    seen[s] = true;
    let mut queue = std::collections::VecDeque::from(vec![s]);
    while let Some(v) = queue.pop_front() {
        for &(w, _) in &t.adj[v] {
            if !seen[w] {
                seen[w] = true;
                prev[w] = Some(v);
                queue.push_back(w);
            }
        }
    }
    rebuild(&prev, s, e)
}

/// A minimum Steiner tree spanning the given terminals, by Dreyfus-Wagner.
///
/// Returns the weight and the edges. The tree may use non-terminal vertices,
/// which is what separates the problem from a spanning tree. Costs
/// `O(3^t n + 2^t n^2)` for `t` terminals, so the terminal count is what has
/// to stay small, not the graph.
///
/// # Panics
/// Panics if there are more than 12 terminals, or a terminal is out of range.
#[must_use]
pub fn steiner_tree_small(g: &Graph, terminals: &[usize]) -> (f64, Vec<(usize, usize)>) {
    assert!(terminals.len() <= 12, "steiner_tree_small needs at most 12 terminals");
    assert!(terminals.iter().all(|&t| t < g.n), "terminal out of range");
    let t = terminals.len();
    if t <= 1 {
        return (0.0, Vec::new());
    }
    let apsp = floyd_warshall(g);
    let full = 1usize << t;
    // dp[mask][v] is the cheapest tree spanning the terminals in mask plus v.
    let mut dp = vec![vec![f64::INFINITY; g.n]; full];
    for (i, &term) in terminals.iter().enumerate() {
        for v in 0..g.n {
            dp[1 << i][v] = apsp.get(term, v);
        }
    }
    for mask in 1..full {
        if mask.count_ones() < 2 {
            continue;
        }
        for v in 0..g.n {
            // Split the terminal set in two and join the two trees at v.
            let mut sub = (mask - 1) & mask;
            while sub > 0 {
                let other = mask ^ sub;
                if sub < other {
                    let cand = dp[sub][v] + dp[other][v];
                    if cand < dp[mask][v] {
                        dp[mask][v] = cand;
                    }
                }
                sub = (sub - 1) & mask;
            }
        }
        // Then allow moving the join point anywhere.
        for v in 0..g.n {
            for u in 0..g.n {
                let cand = dp[mask][u] + apsp.get(u, v);
                if cand < dp[mask][v] {
                    dp[mask][v] = cand;
                }
            }
        }
    }
    let cost = (0..g.n).fold(f64::INFINITY, |a, v| a.min(dp[full - 1][v]));
    if !cost.is_finite() {
        return (f64::INFINITY, Vec::new());
    }
    // Recover a witness by taking the metric closure over the terminals and
    // expanding its minimum spanning tree back into graph edges. That is the
    // 2-approximate construction, so it is only used to report a concrete edge
    // set; the returned weight is the exact optimum from the table above.
    let mut edges = Vec::new();
    let mut closure = Graph::new(t, false);
    for i in 0..t {
        for j in i + 1..t {
            closure.add_edge(i, j, apsp.get(terminals[i], terminals[j]));
        }
    }
    let (_, mst) = minimum_spanning_tree_kruskal(&closure);
    for (i, j) in mst {
        if let Some(p) = shortest_path_edges(g, terminals[i], terminals[j]) {
            edges.extend(p);
        }
    }
    edges.sort_unstable();
    edges.dedup();
    (cost, edges)
}

fn shortest_path_edges(g: &Graph, s: usize, t: usize) -> Option<Vec<(usize, usize)>> {
    let (_, path) = dijkstra_target(g, s, t)?;
    Some(
        path.windows(2)
            .map(|w| (w[0].min(w[1]), w[0].max(w[1])))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Tours
// ---------------------------------------------------------------------------

/// The exact optimal travelling salesman tour, by Held-Karp.
///
/// Returns the tour length and the tour as a vertex sequence starting and
/// ending at 0, with the final return implied rather than repeated. Costs
/// `O(2^n n^2)` time and `O(2^n n)` memory.
///
/// # Panics
/// Panics if `dist` is not square, or has more than 20 rows.
#[must_use]
pub fn traveling_salesman_exact(dist: &Matrix) -> (f64, Vec<usize>) {
    assert_eq!(dist.rows, dist.cols, "the distance matrix must be square");
    assert!(dist.rows <= 20, "Held-Karp needs at most 20 cities");
    let n = dist.rows;
    if n <= 1 {
        return (0.0, (0..n).collect());
    }
    // dp[mask][j]: the cheapest path from 0 through exactly the cities in mask
    // (which excludes 0) ending at j.
    let sub = 1usize << (n - 1);
    let mut dp = vec![vec![f64::INFINITY; n - 1]; sub];
    let mut parent = vec![vec![usize::MAX; n - 1]; sub];
    for j in 0..n - 1 {
        dp[1 << j][j] = dist.get(0, j + 1);
    }
    for mask in 1..sub {
        for j in 0..n - 1 {
            if mask >> j & 1 == 0 || !dp[mask][j].is_finite() {
                continue;
            }
            let base = dp[mask][j];
            for k in 0..n - 1 {
                if mask >> k & 1 == 1 {
                    continue;
                }
                let cand = base + dist.get(j + 1, k + 1);
                let next = mask | 1 << k;
                if cand < dp[next][k] {
                    dp[next][k] = cand;
                    parent[next][k] = j;
                }
            }
        }
    }
    let full = sub - 1;
    let mut best = f64::INFINITY;
    let mut last = 0usize;
    for j in 0..n - 1 {
        let cand = dp[full][j] + dist.get(j + 1, 0);
        if cand < best {
            best = cand;
            last = j;
        }
    }
    let mut tour = Vec::with_capacity(n);
    let mut mask = full;
    let mut j = last;
    while j != usize::MAX {
        tour.push(j + 1);
        let p = parent[mask][j];
        mask ^= 1 << j;
        j = p;
    }
    tour.push(0);
    tour.reverse();
    (best, tour)
}

/// A nearest-neighbour tour: repeatedly walk to the closest unvisited city.
///
/// Fast and usually poor: on a metric instance it can be a logarithmic factor
/// worse than optimal, so it is a starting point for [`tsp_2opt`] rather than
/// an answer.
///
/// # Panics
/// Panics if `dist` is not square.
#[must_use]
pub fn tsp_nearest_neighbor(dist: &Matrix) -> (f64, Vec<usize>) {
    assert_eq!(dist.rows, dist.cols, "the distance matrix must be square");
    let n = dist.rows;
    if n == 0 {
        return (0.0, Vec::new());
    }
    let mut seen = vec![false; n];
    let mut tour = vec![0usize];
    seen[0] = true;
    let mut total = 0.0;
    let mut cur = 0usize;
    for _ in 1..n {
        let mut best = f64::INFINITY;
        let mut pick = usize::MAX;
        for v in 0..n {
            if !seen[v] && dist.get(cur, v) < best {
                best = dist.get(cur, v);
                pick = v;
            }
        }
        seen[pick] = true;
        tour.push(pick);
        total += best;
        cur = pick;
    }
    total += dist.get(cur, 0);
    (total, tour)
}

/// The length of a closed tour under `dist`.
#[must_use]
pub fn tour_length(dist: &Matrix, tour: &[usize]) -> f64 {
    if tour.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..tour.len() {
        total += dist.get(tour[i], tour[(i + 1) % tour.len()]);
    }
    total
}

/// 2-opt local search: repeatedly reverse a tour segment when that shortens
/// the tour, until no single reversal helps.
///
/// The result is 2-optimal, not optimal. On a symmetric instance a reversal
/// changes only the two edges at its ends, which is what makes each move an
/// `O(1)` decision.
///
/// # Panics
/// Panics if `dist` is not square, or `tour` is not a permutation of its rows.
#[must_use]
pub fn tsp_2opt(dist: &Matrix, tour: &[usize]) -> (f64, Vec<usize>) {
    assert_eq!(dist.rows, dist.cols, "the distance matrix must be square");
    assert!(
        crate::discrete::combinatorics::is_permutation(tour) && tour.len() == dist.rows,
        "the tour must be a permutation of the cities"
    );
    let n = tour.len();
    let mut t = tour.to_vec();
    if n < 4 {
        return (tour_length(dist, &t), t);
    }
    loop {
        let mut improved = false;
        for i in 0..n - 1 {
            for j in i + 2..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                let (a, b) = (t[i], t[i + 1]);
                let (c, d) = (t[j], t[(j + 1) % n]);
                // Reversing t[i+1..=j] replaces (a,b) and (c,d) by (a,c),(b,d).
                let delta = dist.get(a, c) + dist.get(b, d) - dist.get(a, b) - dist.get(c, d);
                if delta < -1e-12 {
                    t[i + 1..=j].reverse();
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    (tour_length(dist, &t), t)
}

/// Or-opt local search: relocate a run of one, two or three consecutive cities
/// elsewhere in the tour, in either orientation, while that shortens it.
///
/// Complements 2-opt, which can only reverse: a run that belongs elsewhere
/// entirely is a move 2-opt cannot make in one step.
///
/// # Panics
/// Panics if `dist` is not square, or `tour` is not a permutation of its rows.
#[must_use]
pub fn tsp_or_opt(dist: &Matrix, tour: &[usize]) -> (f64, Vec<usize>) {
    assert_eq!(dist.rows, dist.cols, "the distance matrix must be square");
    assert!(
        crate::discrete::combinatorics::is_permutation(tour) && tour.len() == dist.rows,
        "the tour must be a permutation of the cities"
    );
    let n = tour.len();
    let mut t = tour.to_vec();
    if n < 5 {
        return (tour_length(dist, &t), t);
    }
    let mut best = tour_length(dist, &t);
    loop {
        let mut improved = false;
        'outer: for len in 1..=3usize {
            for start in 0..n {
                if start + len > n {
                    continue;
                }
                let segment: Vec<usize> = t[start..start + len].to_vec();
                let mut rest: Vec<usize> = t.clone();
                rest.drain(start..start + len);
                for pos in 0..=rest.len() {
                    for reversed in [false, true] {
                        let mut cand = rest.clone();
                        let mut seg = segment.clone();
                        if reversed {
                            seg.reverse();
                        }
                        for (k, v) in seg.into_iter().enumerate() {
                            cand.insert(pos + k, v);
                        }
                        let len_c = tour_length(dist, &cand);
                        if len_c < best - 1e-12 {
                            best = len_c;
                            t = cand;
                            improved = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    (best, t)
}

/// Christofides' tour, which is within a factor of 1.5 of optimal on a metric
/// instance.
///
/// Takes a minimum spanning tree, adds a minimum-weight perfect matching on
/// the odd-degree vertices to make every degree even, walks the resulting
/// Eulerian circuit, and shortcuts repeats. The matching here is exact by
/// brute force over pairings, which is affordable because a tree has few
/// odd-degree vertices on the instances this is used for, and is refused
/// beyond sixteen of them rather than silently degrading to a greedy one.
///
/// Returns `None` when the odd set is too large for the exact matching.
///
/// # Panics
/// Panics if `dist` is not square or is not symmetric, since the guarantee
/// needs a metric.
#[must_use]
pub fn tsp_christofides(dist: &Matrix) -> Option<(f64, Vec<usize>)> {
    assert_eq!(dist.rows, dist.cols, "the distance matrix must be square");
    let n = dist.rows;
    assert!(
        (0..n).all(|i| (0..n).all(|j| (dist.get(i, j) - dist.get(j, i)).abs() < 1e-12)),
        "Christofides needs a symmetric distance matrix"
    );
    if n <= 2 {
        return Some((tour_length(dist, &(0..n).collect::<Vec<_>>()), (0..n).collect()));
    }
    let mut g = Graph::new(n, false);
    for i in 0..n {
        for j in i + 1..n {
            g.add_edge(i, j, dist.get(i, j));
        }
    }
    let (_, mst) = minimum_spanning_tree_kruskal(&g);
    let mut deg = vec![0usize; n];
    for &(u, v) in &mst {
        deg[u] += 1;
        deg[v] += 1;
    }
    let odd: Vec<usize> = (0..n).filter(|&v| !deg[v].is_multiple_of(2)).collect();
    if odd.len() > 16 {
        return None;
    }
    let matching = min_weight_perfect_matching_brute(dist, &odd);

    // The multigraph of tree edges plus matching edges has every degree even,
    // so it has an Eulerian circuit.
    let mut multi = Graph::new(n, false);
    for &(u, v) in &mst {
        multi.add_edge(u, v, dist.get(u, v));
    }
    for &(u, v) in &matching {
        multi.add_edge(u, v, dist.get(u, v));
    }
    let circuit = multi.eulerian_circuit()?;
    // Shortcut: keep the first occurrence of each vertex. The triangle
    // inequality is what makes skipping never cost more.
    let mut seen = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    for v in circuit {
        if !seen[v] {
            seen[v] = true;
            tour.push(v);
        }
    }
    Some((tour_length(dist, &tour), tour))
}

/// A minimum-weight perfect matching on an even-sized vertex set, by
/// recursion over pairings.
fn min_weight_perfect_matching_brute(dist: &Matrix, vs: &[usize]) -> Vec<(usize, usize)> {
    let k = vs.len();
    if k == 0 {
        return Vec::new();
    }
    let full = 1usize << k;
    let mut dp = vec![f64::INFINITY; full];
    let mut choice = vec![(usize::MAX, usize::MAX); full];
    dp[0] = 0.0;
    for mask in 0..full {
        if !dp[mask].is_finite() {
            continue;
        }
        let Some(i) = (0..k).find(|&i| mask >> i & 1 == 0) else {
            continue;
        };
        for j in i + 1..k {
            if mask >> j & 1 == 1 {
                continue;
            }
            let next = mask | 1 << i | 1 << j;
            let cand = dp[mask] + dist.get(vs[i], vs[j]);
            if cand < dp[next] {
                dp[next] = cand;
                choice[next] = (i, j);
            }
        }
    }
    let mut out = Vec::new();
    let mut mask = full - 1;
    while mask != 0 {
        let (i, j) = choice[mask];
        if i == usize::MAX {
            break;
        }
        out.push((vs[i].min(vs[j]), vs[i].max(vs[j])));
        mask ^= 1 << i | 1 << j;
    }
    out
}

/// A shortest closed walk crossing every edge at least once: the Chinese
/// postman problem.
///
/// Returns the walk's total weight and the vertex sequence. When every degree
/// is already even the answer is an Eulerian circuit and costs exactly the
/// total edge weight; otherwise the odd-degree vertices are paired up by a
/// minimum-weight perfect matching over shortest paths, and those paths are
/// duplicated. Returns `None` when the edges span more than one component, so
/// that no single closed walk can cross them all, or when the odd set is too
/// large for the exact matching. An edgeless graph has nothing to cross, so it
/// returns the empty route rather than failing on being disconnected.
///
/// # Panics
/// Panics if the graph is directed, where the construction differs.
#[must_use]
pub fn chinese_postman(g: &Graph) -> Option<(f64, Vec<usize>)> {
    assert!(!g.directed, "chinese_postman here is for undirected graphs");
    let total: f64 = g.edges().iter().map(|&(_, _, w)| w).sum();
    if g.edge_count() == 0 {
        return Some((0.0, vec![0]));
    }
    if !g.is_connected() {
        return None;
    }
    let odd: Vec<usize> = (0..g.n).filter(|&v| !g.degree(v).is_multiple_of(2)).collect();
    if odd.is_empty() {
        let circuit = g.eulerian_circuit()?;
        return Some((total, circuit));
    }
    if odd.len() > 16 {
        return None;
    }
    let apsp = floyd_warshall(g);
    let matching = min_weight_perfect_matching_brute(&apsp, &odd);
    let extra: f64 = matching.iter().map(|&(u, v)| apsp.get(u, v)).sum();

    // Duplicate the matched shortest paths; the augmented graph then has every
    // degree even and its Eulerian circuit is the postman's route.
    let mut aug = g.clone();
    for &(u, v) in &matching {
        if let Some(edges) = shortest_path_edges(g, u, v) {
            for (a, b) in edges {
                aug.add_edge(a, b, arc_weight(g, a, b).unwrap_or(0.0));
            }
        }
    }
    let circuit = aug.eulerian_circuit()?;
    Some((total + extra, circuit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::{
        complete_graph, cycle_graph, grid_2d, path_graph, petersen_graph, star_graph,
    };
    use crate::monte_carlo::Rng;

    fn random_weighted(n: usize, p: f64, directed: bool, rng: &mut Rng) -> Graph {
        let mut g = Graph::new(n, directed);
        for u in 0..n {
            let start = if directed { 0 } else { u + 1 };
            for v in start..n {
                if u != v && rng.next_f64() < p {
                    g.add_edge(u, v, 1.0 + 9.0 * rng.next_f64());
                }
            }
        }
        g
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * a.abs().max(b.abs()).max(1.0) || (!a.is_finite() && !b.is_finite())
    }

    /// The weight of a path, or infinity if any step is not an edge.
    fn path_weight(g: &Graph, path: &[usize]) -> f64 {
        path.windows(2)
            .map(|w| arc_weight(g, w[0], w[1]).unwrap_or(f64::INFINITY))
            .sum()
    }

    // -----------------------------------------------------------------------
    // Shortest paths
    // -----------------------------------------------------------------------

    /// The roadmap's headline property: the four shortest-path algorithms must
    /// agree on random graphs, and the paths they report must have the lengths
    /// they claim.
    #[test]
    fn all_shortest_path_algorithms_agree() {
        let mut rng = Rng::new(0x5EED);
        for directed in [false, true] {
            for n in 1..=9usize {
                for _ in 0..15 {
                    let g = random_weighted(n, 0.4, directed, &mut rng);
                    let fw = floyd_warshall(&g);
                    let jn = johnson(&g).expect("no negative weights here");
                    for s in 0..n {
                        let (dj, prev) = dijkstra(&g, s);
                        let (bf, _) = bellman_ford(&g, s).expect("no negative cycle");
                        for t in 0..n {
                            assert!(close(dj[t], bf[t]), "dijkstra vs bellman-ford {s}->{t}");
                            assert!(close(dj[t], fw.get(s, t)), "dijkstra vs floyd {s}->{t}");
                            assert!(close(dj[t], jn.get(s, t)), "dijkstra vs johnson {s}->{t}");
                            // The reported path really has that weight.
                            if dj[t].is_finite() {
                                let path = rebuild(&prev, s, t).expect("a path exists");
                                assert_eq!(path[0], s);
                                assert_eq!(*path.last().unwrap(), t);
                                assert!(
                                    close(path_weight(&g, &path), dj[t]),
                                    "path weight disagrees at {s}->{t}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Bellman-Ford must handle negative weights that Dijkstra cannot, and
    /// must report a negative cycle rather than looping.
    #[test]
    fn bellman_ford_handles_negative_weights_and_cycles() {
        // A negative edge that Dijkstra would get wrong: the direct arc looks
        // best until the negative detour is taken.
        let g = Graph::from_edges(
            4,
            &[(0, 1, 1.0), (0, 2, 5.0), (1, 3, 4.0), (3, 2, -3.0)],
            true,
        );
        let (d, _) = bellman_ford(&g, 0).expect("no cycle");
        assert!(close(d[2], 2.0), "expected 1 + 4 - 3 = 2, got {}", d[2]);
        assert!(close(d[3], 5.0));
        // Johnson agrees, since it reweights rather than assuming positivity.
        let j = johnson(&g).expect("no cycle");
        assert!(close(j.get(0, 2), 2.0));
        assert!(close(j.get(0, 3), 5.0));
        // And so does Floyd-Warshall.
        let f = floyd_warshall(&g);
        assert!(close(f.get(0, 2), 2.0));

        // A negative cycle is reported, not looped on.
        let bad = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, -3.0), (2, 0, 1.0)], true);
        assert!(bellman_ford(&bad, 0).is_err());
        assert!(johnson(&bad).is_err());
        // A negative cycle unreachable from the source is not an error there,
        // but Johnson's virtual source reaches everything, so it is for
        // Johnson.
        let mut split = Graph::new(5, true);
        split.add_edge(0, 1, 1.0);
        for (u, v, w) in [(2, 3, 1.0), (3, 4, -3.0), (4, 2, 1.0)] {
            split.add_edge(u, v, w);
        }
        assert!(
            bellman_ford(&split, 0).is_ok(),
            "the cycle is unreachable from 0"
        );
        assert!(johnson(&split).is_err());
    }

    /// A* with an admissible heuristic must return the same length as
    /// Dijkstra, and the zero heuristic makes it Dijkstra exactly.
    #[test]
    fn a_star_matches_dijkstra_with_admissible_heuristics() {
        let mut rng = Rng::new(0xA57A2);
        for n in 2..=9usize {
            for _ in 0..15 {
                let g = random_weighted(n, 0.5, false, &mut rng);
                for s in 0..n {
                    let (d, _) = dijkstra(&g, s);
                    for t in 0..n {
                        // The zero heuristic is trivially admissible.
                        match a_star(&g, s, t, &|_| 0.0) {
                            Some((len, path)) => {
                                assert!(close(len, d[t]), "A* {s}->{t}");
                                assert!(close(path_weight(&g, &path), len));
                            }
                            None => assert!(!d[t].is_finite()),
                        }
                        // The exact remaining distance is also admissible, and
                        // is the strongest such heuristic.
                        let (dt, _) = dijkstra(&g.reverse(), t);
                        let perfect =
                            a_star(&g, s, t, &|v| if dt[v].is_finite() { dt[v] } else { 0.0 });
                        match perfect {
                            Some((len, _)) => assert!(close(len, d[t]), "perfect h {s}->{t}"),
                            None => assert!(!d[t].is_finite()),
                        }
                    }
                }
            }
        }
        // On a grid the Manhattan distance is admissible and speeds the search
        // without changing the answer.
        let w = 6usize;
        let g = grid_2d(w, 6);
        let h = |v: usize| {
            let (x, y) = (v % w, v / w);
            let (tx, ty) = (35 % w, 35 / w);
            (x as f64 - tx as f64).abs() + (y as f64 - ty as f64).abs()
        };
        let (len, path) = a_star(&g, 0, 35, &h).expect("the grid is connected");
        assert!(close(len, 10.0), "Manhattan distance from a corner is 10");
        assert_eq!(path.len(), 11);
    }

    /// The bidirectional search must return the same length as the one-sided
    /// one, and a valid path.
    #[test]
    fn bidirectional_dijkstra_matches_dijkstra() {
        let mut rng = Rng::new(0xB1D1);
        for directed in [false, true] {
            for n in 1..=9usize {
                for _ in 0..15 {
                    let g = random_weighted(n, 0.45, directed, &mut rng);
                    for s in 0..n {
                        let (d, _) = dijkstra(&g, s);
                        for t in 0..n {
                            match bidirectional_dijkstra(&g, s, t) {
                                Some((len, path)) => {
                                    assert!(
                                        close(len, d[t]),
                                        "n={n} {s}->{t}: {len} vs {}",
                                        d[t]
                                    );
                                    assert_eq!(path[0], s);
                                    assert_eq!(*path.last().unwrap(), t);
                                    assert!(
                                        close(path_weight(&g, &path), len),
                                        "spliced path weight disagrees"
                                    );
                                }
                                None => {
                                    assert!(!d[t].is_finite(), "missed a reachable {s}->{t}")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Yen's k shortest paths must be loopless, distinct, in increasing order,
    /// and start with the true shortest path.
    #[test]
    fn yen_returns_increasing_distinct_loopless_paths() {
        let mut rng = Rng::new(0x7E71);
        for n in 2..=8usize {
            for _ in 0..10 {
                let g = random_weighted(n, 0.5, true, &mut rng);
                for s in 0..n {
                    for t in 0..n {
                        if s == t {
                            continue;
                        }
                        let ks = k_shortest_paths_yen(&g, s, t, 4);
                        let best = dijkstra_target(&g, s, t);
                        match (&best, ks.first()) {
                            (Some((bl, _)), Some((kl, _))) => assert!(close(*bl, *kl)),
                            (None, None) => {}
                            _ => panic!("Yen and Dijkstra disagree on reachability"),
                        }
                        for (len, path) in &ks {
                            assert_eq!(path[0], s);
                            assert_eq!(*path.last().unwrap(), t);
                            assert!(
                                close(path_weight(&g, path), *len),
                                "claimed length is wrong"
                            );
                            // Loopless.
                            let mut sorted = path.clone();
                            sorted.sort_unstable();
                            sorted.dedup();
                            assert_eq!(sorted.len(), path.len(), "path repeats a vertex");
                        }
                        // Increasing and distinct.
                        for w in ks.windows(2) {
                            assert!(w[0].0 <= w[1].0 + 1e-12, "not in increasing order");
                            assert_ne!(w[0].1, w[1].1);
                        }
                    }
                }
            }
        }
        // On a graph with exactly three s-t paths, asking for five gives three.
        let g = Graph::from_edges(
            4,
            &[
                (0, 1, 1.0),
                (0, 2, 2.0),
                (1, 3, 5.0),
                (2, 3, 3.0),
                (0, 3, 9.0),
            ],
            true,
        );
        let ks = k_shortest_paths_yen(&g, 0, 3, 5);
        assert_eq!(ks.len(), 3);
        assert!(close(ks[0].0, 5.0), "0-2-3 costs 5");
        assert!(close(ks[1].0, 6.0), "0-1-3 costs 6");
        assert!(close(ks[2].0, 9.0), "the direct arc costs 9");
    }

    /// The widest path's width must equal the best bottleneck found by brute
    /// force over all simple paths, and likewise for the minimax path.
    #[test]
    fn widest_and_minimax_paths_match_brute_force() {
        let mut rng = Rng::new(0x21DE);
        for n in 2..=7usize {
            for _ in 0..15 {
                let g = random_weighted(n, 0.5, false, &mut rng);
                for s in 0..n {
                    for t in 0..n {
                        if s == t {
                            continue;
                        }
                        let paths = all_simple_paths(&g, s, t);
                        let widest = paths
                            .iter()
                            .map(|p| {
                                p.windows(2)
                                    .map(|w| arc_weight(&g, w[0], w[1]).unwrap())
                                    .fold(f64::INFINITY, f64::min)
                            })
                            .fold(f64::NEG_INFINITY, f64::max);
                        let narrowest = paths
                            .iter()
                            .map(|p| {
                                p.windows(2)
                                    .map(|w| arc_weight(&g, w[0], w[1]).unwrap())
                                    .fold(f64::NEG_INFINITY, f64::max)
                            })
                            .fold(f64::INFINITY, f64::min);
                        match widest_path(&g, s, t) {
                            Some((w, path)) => {
                                assert!(close(w, widest), "widest {s}->{t}: {w} vs {widest}");
                                let actual = path
                                    .windows(2)
                                    .map(|x| arc_weight(&g, x[0], x[1]).unwrap())
                                    .fold(f64::INFINITY, f64::min);
                                assert!(close(actual, w), "reported path is not that wide");
                            }
                            None => assert!(paths.is_empty()),
                        }
                        match minimax_path(&g, s, t) {
                            Some((w, path)) => {
                                assert!(close(w, narrowest), "minimax {s}->{t}");
                                let actual = path
                                    .windows(2)
                                    .map(|x| arc_weight(&g, x[0], x[1]).unwrap())
                                    .fold(f64::NEG_INFINITY, f64::max);
                                assert!(close(actual, w));
                            }
                            None => assert!(paths.is_empty()),
                        }
                    }
                }
            }
        }
    }

    /// Every simple path from s to t, by depth-first enumeration.
    fn all_simple_paths(g: &Graph, s: usize, t: usize) -> Vec<Vec<usize>> {
        fn go(
            g: &Graph,
            cur: usize,
            t: usize,
            on_path: &mut Vec<bool>,
            path: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
        ) {
            if cur == t {
                out.push(path.clone());
                return;
            }
            for &(w, _) in &g.adj[cur] {
                if !on_path[w] {
                    on_path[w] = true;
                    path.push(w);
                    go(g, w, t, on_path, path, out);
                    path.pop();
                    on_path[w] = false;
                }
            }
        }
        let mut on_path = vec![false; g.n];
        on_path[s] = true;
        let mut path = vec![s];
        let mut out = Vec::new();
        go(g, s, t, &mut on_path, &mut path, &mut out);
        out
    }

    /// The minimax path's bottleneck must equal the largest edge on the
    /// minimum spanning tree path, which is a theorem about spanning trees and
    /// so an independent check of both.
    #[test]
    fn minimax_path_matches_the_mst_path() {
        let mut rng = Rng::new(0x1157);
        for n in 2..=8usize {
            for _ in 0..15 {
                let mut g = Graph::new(n, false);
                for u in 0..n {
                    for v in u + 1..n {
                        g.add_edge(u, v, 1.0 + 9.0 * rng.next_f64());
                    }
                }
                let (_, mst) = minimum_spanning_tree_kruskal(&g);
                let mut t = Graph::new(n, false);
                for &(u, v) in &mst {
                    t.add_edge(u, v, arc_weight(&g, u, v).unwrap());
                }
                for s in 0..n {
                    for e in 0..n {
                        if s == e {
                            continue;
                        }
                        let (bottleneck, _) = minimax_path(&g, s, e).expect("connected");
                        let tp = tree_path(&t, s, e).expect("the tree is connected");
                        let on_tree = tp
                            .windows(2)
                            .map(|w| arc_weight(&t, w[0], w[1]).unwrap())
                            .fold(f64::NEG_INFINITY, f64::max);
                        assert!(
                            close(bottleneck, on_tree),
                            "n = {n}, {s}->{e}: {bottleneck} vs {on_tree}"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // DAGs
    // -----------------------------------------------------------------------

    /// A monotone grid DAG from one corner to the other, of the given side.
    fn grid_dag(side: usize) -> Graph {
        let mut g = Graph::new(side * side, true);
        for y in 0..side {
            for x in 0..side {
                let v = y * side + x;
                if x + 1 < side {
                    g.add_edge(v, v + 1, 1.0);
                }
                if y + 1 < side {
                    g.add_edge(v, v + side, 1.0);
                }
            }
        }
        g
    }

    #[test]
    fn dag_distances_and_path_counts_are_exact() {
        // A grid DAG with only right and down moves has C(2n, n) paths from
        // one corner to the other.
        for n in 1..=9usize {
            let side = n + 1;
            let g = grid_dag(side);
            assert!(g.is_dag());
            let paths = count_paths_dag(&g, 0, side * side - 1);
            assert_eq!(paths, BigInt::binomial(2 * n as u64, n as u64), "n = {n}");
            // Every monotone route has the same length, so shortest = longest.
            let s = dag_shortest(&g, 0);
            let l = dag_longest(&g, 0);
            assert!(close(s[side * side - 1], 2.0 * n as f64));
            assert!(close(l[side * side - 1], 2.0 * n as f64));
        }
        // 34 steps a side already passes u64, so an integer counter would wrap.
        let big = grid_dag(35);
        let c = count_paths_dag(&big, 0, 35 * 35 - 1);
        assert_eq!(c, BigInt::binomial(68, 34));
        assert!(c > BigInt::from_str_radix(&u64::MAX.to_string(), 10).unwrap());

        // Shortest and longest genuinely differ when the weights do.
        let g = Graph::from_edges(
            4,
            &[(0, 1, 1.0), (0, 2, 5.0), (1, 3, 1.0), (2, 3, 1.0)],
            true,
        );
        assert!(close(dag_shortest(&g, 0)[3], 2.0));
        assert!(close(dag_longest(&g, 0)[3], 6.0));
        // Shortest on a DAG matches Bellman-Ford, negative weights included --
        // which Dijkstra could not do.
        let neg = Graph::from_edges(
            4,
            &[(0, 1, 1.0), (0, 2, 5.0), (1, 3, -4.0), (2, 3, 1.0)],
            true,
        );
        let (bf, _) = bellman_ford(&neg, 0).unwrap();
        let ds = dag_shortest(&neg, 0);
        for v in 0..4 {
            assert!(close(ds[v], bf[v]), "vertex {v}");
        }
    }

    #[test]
    fn transitive_closure_matches_reachability() {
        let mut rng = Rng::new(0xC105);
        for directed in [false, true] {
            for n in 1..=9usize {
                let g = random_weighted(n, 0.25, directed, &mut rng);
                let r = transitive_closure(&g);
                for s in 0..n {
                    let bfs = g.bfs(s);
                    for t in 0..n {
                        assert_eq!(r[s][t], bfs[t].is_some(), "({s}, {t})");
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Spanning trees
    // -----------------------------------------------------------------------

    /// The three MST algorithms must agree on weight, and each result must be
    /// an acyclic spanning forest.
    #[test]
    fn all_mst_algorithms_agree() {
        let mut rng = Rng::new(0x5A7);
        for n in 1..=10usize {
            for _ in 0..20 {
                let g = random_weighted(n, 0.4, false, &mut rng);
                let (wk, ek) = minimum_spanning_tree_kruskal(&g);
                let (wp, ep) = minimum_spanning_tree_prim(&g);
                let (wb, eb) = minimum_spanning_tree_boruvka(&g);
                assert!(close(wk, wp), "kruskal {wk} vs prim {wp} at n = {n}");
                assert!(close(wk, wb), "kruskal {wk} vs boruvka {wb} at n = {n}");
                let components = g.connected_components().len();
                for (name, edges) in [("kruskal", &ek), ("prim", &ep), ("boruvka", &eb)] {
                    assert_eq!(edges.len(), n - components, "{name} edge count");
                    let mut ds = crate::discrete::disjoint_set::DisjointSet::new(n);
                    for &(u, v) in edges {
                        assert!(ds.union(u, v), "{name} produced a cycle");
                    }
                    assert_eq!(ds.count(), components, "{name} does not span");
                }
                // Minimal: no other spanning tree is cheaper, checked exactly
                // for small n by enumerating every spanning tree.
                if n <= 6 && components == 1 {
                    let best = brute_force_mst_weight(&g);
                    assert!(close(wk, best), "n = {n}: {wk} vs brute force {best}");
                }
                // The weight is the sum of the chosen edges.
                let sum: f64 = ek.iter().map(|&(u, v)| arc_weight(&g, u, v).unwrap()).sum();
                assert!(close(wk, sum));
            }
        }
    }

    /// The cheapest spanning tree, by enumerating every edge subset of the
    /// right size and keeping the acyclic spanning ones.
    fn brute_force_mst_weight(g: &Graph) -> f64 {
        let edges: Vec<(usize, usize, f64)> =
            g.edges().into_iter().filter(|&(u, v, _)| u != v).collect();
        let mut best = f64::INFINITY;
        for combo in crate::discrete::combinatorics::combinations_iter(edges.len(), g.n - 1) {
            let mut ds = crate::discrete::disjoint_set::DisjointSet::new(g.n);
            let mut ok = true;
            let mut total = 0.0;
            for &i in &combo {
                let (u, v, w) = edges[i];
                if !ds.union(u, v) {
                    ok = false;
                    break;
                }
                total += w;
            }
            if ok && ds.count() == 1 {
                best = best.min(total);
            }
        }
        best
    }

    /// The second-best spanning tree must be a valid spanning tree, differ
    /// from the best, cost at least as much, and be the cheapest such.
    #[test]
    fn second_best_mst_is_the_next_cheapest_tree() {
        let mut rng = Rng::new(0x2D0);
        for n in 3..=6usize {
            for _ in 0..25 {
                let mut g = Graph::new(n, false);
                for u in 0..n {
                    for v in u + 1..n {
                        g.add_edge(u, v, 1.0 + 9.0 * rng.next_f64());
                    }
                }
                let (best, tree) = minimum_spanning_tree_kruskal(&g);
                let (second, other) =
                    second_best_mst(&g).expect("a complete graph has a second tree");
                assert!(second >= best - 1e-9, "second {second} is below best {best}");
                let mut t1 = tree.clone();
                t1.sort_unstable();
                assert_ne!(t1, other, "the second tree is the same tree");
                assert_eq!(other.len(), n - 1);
                // Valid: acyclic and spanning, with the weight claimed.
                let mut ds = crate::discrete::disjoint_set::DisjointSet::new(n);
                let mut total = 0.0;
                for &(u, v) in &other {
                    assert!(ds.union(u, v));
                    total += arc_weight(&g, u, v).unwrap();
                }
                assert_eq!(ds.count(), 1);
                assert!(close(total, second), "reported weight is wrong");
                // Cheapest such: brute force over every other spanning tree.
                let cheapest_other = brute_force_second_best(&g, &t1);
                assert!(
                    close(second, cheapest_other),
                    "n = {n}: {second} vs {cheapest_other}"
                );
            }
        }
        // A tree has no second spanning tree, nor does a disconnected graph.
        assert!(second_best_mst(&path_graph(5)).is_none());
        assert!(second_best_mst(&Graph::new(4, false)).is_none());
    }

    fn brute_force_second_best(g: &Graph, best_tree: &[(usize, usize)]) -> f64 {
        let edges: Vec<(usize, usize, f64)> = g.edges();
        let mut best = f64::INFINITY;
        for combo in crate::discrete::combinatorics::combinations_iter(edges.len(), g.n - 1) {
            let mut ds = crate::discrete::disjoint_set::DisjointSet::new(g.n);
            let mut ok = true;
            let mut total = 0.0;
            let mut set: Vec<(usize, usize)> = Vec::new();
            for &i in &combo {
                let (u, v, w) = edges[i];
                if !ds.union(u, v) {
                    ok = false;
                    break;
                }
                total += w;
                set.push((u.min(v), u.max(v)));
            }
            set.sort_unstable();
            if ok && ds.count() == 1 && set != best_tree {
                best = best.min(total);
            }
        }
        best
    }

    /// The Steiner tree weight must equal a brute-force optimum, and the
    /// reported edges must actually connect every terminal.
    #[test]
    fn steiner_tree_matches_brute_force() {
        let mut rng = Rng::new(0x57E1);
        for n in 3..=7usize {
            for _ in 0..10 {
                let mut g = Graph::new(n, false);
                for u in 0..n {
                    for v in u + 1..n {
                        g.add_edge(u, v, 1.0 + 9.0 * rng.next_f64());
                    }
                }
                for t in 2..=n.min(4) {
                    let terminals: Vec<usize> = (0..t).collect();
                    let (cost, edges) = steiner_tree_small(&g, &terminals);
                    let brute = brute_steiner(&g, &terminals);
                    assert!(close(cost, brute), "n = {n}, t = {t}: {cost} vs {brute}");
                    let mut ds = crate::discrete::disjoint_set::DisjointSet::new(n);
                    for &(u, v) in &edges {
                        ds.union(u, v);
                    }
                    for &term in &terminals {
                        assert!(
                            ds.connected(terminals[0], term),
                            "terminal {term} is cut off"
                        );
                    }
                }
            }
        }
        // Fewer than two terminals costs nothing.
        assert_eq!(steiner_tree_small(&complete_graph(5), &[]).0, 0.0);
        assert_eq!(steiner_tree_small(&complete_graph(5), &[2]).0, 0.0);
    }

    /// The cheapest connected subgraph containing every terminal, by trying
    /// every subset of Steiner points and spanning it.
    fn brute_steiner(g: &Graph, terminals: &[usize]) -> f64 {
        let others: Vec<usize> = (0..g.n).filter(|v| !terminals.contains(v)).collect();
        let mut best = f64::INFINITY;
        for extra in 0..=others.len() {
            for combo in crate::discrete::combinatorics::combinations_iter(others.len(), extra) {
                let mut vs: Vec<usize> = terminals.to_vec();
                vs.extend(combo.iter().map(|&i| others[i]));
                vs.sort_unstable();
                let sub = g.subgraph(&vs);
                if !sub.is_connected() {
                    continue;
                }
                best = best.min(minimum_spanning_tree_kruskal(&sub).0);
            }
        }
        best
    }

    // -----------------------------------------------------------------------
    // Tours
    // -----------------------------------------------------------------------

    /// Points in the plane, so the triangle inequality holds exactly.
    fn random_metric(n: usize, rng: &mut Rng) -> Matrix {
        let pts: Vec<(f64, f64)> = (0..n).map(|_| (rng.next_f64(), rng.next_f64())).collect();
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let (dx, dy) = (pts[i].0 - pts[j].0, pts[i].1 - pts[j].1);
                m.set(i, j, (dx * dx + dy * dy).sqrt());
            }
        }
        m
    }

    /// Held-Karp must match brute force over every tour.
    #[test]
    fn held_karp_matches_brute_force() {
        let mut rng = Rng::new(0x7595);
        for n in 1..=8usize {
            for _ in 0..10 {
                let d = random_metric(n, &mut rng);
                let (cost, tour) = traveling_salesman_exact(&d);
                assert_eq!(tour.len(), n);
                assert!(crate::discrete::combinatorics::is_permutation(&tour));
                if n > 1 {
                    assert_eq!(tour[0], 0, "the tour must start at 0");
                }
                assert!(close(cost, tour_length(&d, &tour)), "claimed cost is wrong");
                if n >= 2 {
                    let rest: Vec<usize> = (1..n).collect();
                    let best = crate::discrete::combinatorics::permutations_iter(&rest)
                        .map(|p| {
                            let mut t = vec![0usize];
                            t.extend(p);
                            tour_length(&d, &t)
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert!(close(cost, best), "n = {n}: {cost} vs {best}");
                }
            }
        }
        // A degenerate instance: every distance equal.
        let mut d = Matrix::zeros(5, 5);
        for i in 0..5 {
            for j in 0..5 {
                d.set(i, j, if i == j { 0.0 } else { 1.0 });
            }
        }
        assert!(close(traveling_salesman_exact(&d).0, 5.0));
    }

    /// The heuristics must never worsen the tour they are given, must return
    /// valid tours, and must never beat the exact optimum.
    #[test]
    fn tsp_heuristics_only_improve() {
        let mut rng = Rng::new(0x2077);
        for n in 5..=9usize {
            for _ in 0..8 {
                let d = random_metric(n, &mut rng);
                let (nn_cost, nn_tour) = tsp_nearest_neighbor(&d);
                assert!(crate::discrete::combinatorics::is_permutation(&nn_tour));
                assert!(close(nn_cost, tour_length(&d, &nn_tour)));

                let (two_cost, two_tour) = tsp_2opt(&d, &nn_tour);
                assert!(crate::discrete::combinatorics::is_permutation(&two_tour));
                assert!(close(two_cost, tour_length(&d, &two_tour)));
                assert!(two_cost <= nn_cost + 1e-9, "2-opt made it worse");

                let (or_cost, or_tour) = tsp_or_opt(&d, &two_tour);
                assert!(crate::discrete::combinatorics::is_permutation(&or_tour));
                assert!(close(or_cost, tour_length(&d, &or_tour)));
                assert!(or_cost <= two_cost + 1e-9, "or-opt made it worse");

                let (opt, _) = traveling_salesman_exact(&d);
                assert!(or_cost >= opt - 1e-9, "a heuristic beat the optimum");

                // 2-opt really is 2-optimal: no single reversal helps.
                for i in 0..n - 1 {
                    for j in i + 2..n {
                        if i == 0 && j == n - 1 {
                            continue;
                        }
                        let mut t = two_tour.clone();
                        t[i + 1..=j].reverse();
                        assert!(
                            tour_length(&d, &t) >= two_cost - 1e-9,
                            "a reversal at ({i}, {j}) still improves"
                        );
                    }
                }
            }
        }
    }

    /// Christofides' guarantee: on a metric instance the tour is within 1.5
    /// times the optimum.
    #[test]
    fn christofides_is_within_three_halves_of_optimal() {
        let mut rng = Rng::new(0xC471);
        for n in 3..=10usize {
            for _ in 0..15 {
                let d = random_metric(n, &mut rng);
                let (cost, tour) = tsp_christofides(&d).expect("the odd set is small here");
                assert_eq!(tour.len(), n, "the tour must visit every city");
                assert!(crate::discrete::combinatorics::is_permutation(&tour));
                assert!(close(cost, tour_length(&d, &tour)), "claimed cost is wrong");
                let (opt, _) = traveling_salesman_exact(&d);
                assert!(cost <= 1.5 * opt + 1e-9, "n = {n}: {cost} exceeds 1.5 x {opt}");
                assert!(cost >= opt - 1e-9);
            }
        }
    }

    /// The Chinese postman route must cross every edge, cost at least the
    /// total edge weight, and equal it exactly when every degree is even.
    #[test]
    fn chinese_postman_covers_every_edge() {
        // Even degrees: the route is an Eulerian circuit and costs the total.
        for g in [cycle_graph(6), complete_graph(5), complete_graph(7)] {
            let total: f64 = g.edges().iter().map(|&(_, _, w)| w).sum();
            let (cost, walk) = chinese_postman(&g).expect("connected with even degrees");
            assert!(close(cost, total), "even-degree cost should be the total");
            check_covers(&g, &walk);
        }
        // Odd degrees force a repeat, so the cost strictly exceeds the total.
        for g in [path_graph(4), star_graph(5), petersen_graph()] {
            let total: f64 = g.edges().iter().map(|&(_, _, w)| w).sum();
            let (cost, walk) = chinese_postman(&g).expect("connected");
            assert!(cost > total + 1e-9, "odd degrees must force a repeat");
            check_covers(&g, &walk);
        }
        // On a tree every edge is a bridge, so the route has to come back
        // along each one: the cost is exactly twice the total edge weight.
        for g in [path_graph(4), path_graph(7), star_graph(6)] {
            let total: f64 = g.edges().iter().map(|&(_, _, w)| w).sum();
            let (cost, _) = chinese_postman(&g).unwrap();
            assert!(close(cost, 2.0 * total), "tree cost {cost} is not 2 x {total}");
        }
        // Edges in two components: no single closed walk crosses them all.
        let split = Graph::from_edges(4, &[(0, 1, 1.0), (2, 3, 1.0)], false);
        assert!(chinese_postman(&split).is_none());
        // An edgeless graph is disconnected but has nothing to cross, so the
        // empty route is the answer rather than a failure.
        assert_eq!(chinese_postman(&Graph::new(4, false)), Some((0.0, vec![0])));
    }

    fn check_covers(g: &Graph, walk: &[usize]) {
        assert_eq!(walk[0], *walk.last().unwrap(), "the route must close");
        let mut used: std::collections::BTreeSet<(usize, usize)> =
            std::collections::BTreeSet::new();
        for w in walk.windows(2) {
            assert!(
                g.adj[w[0]].iter().any(|&(t, _)| t == w[1]),
                "step {w:?} is not an edge"
            );
            used.insert((w[0].min(w[1]), w[0].max(w[1])));
        }
        for (u, v, _) in g.edges() {
            assert!(
                used.contains(&(u.min(v), u.max(v))),
                "edge ({u}, {v}) is never crossed"
            );
        }
    }

    /// The heap key must order ascending and never let a NaN come out ahead of
    /// a real key, which is what stops a NaN weight from corrupting a search.
    #[test]
    fn min_key_orders_ascending_and_sinks_nan() {
        let mut heap = BinaryHeap::new();
        for x in [3.0, 1.0, f64::NAN, 2.0, f64::INFINITY] {
            heap.push(MinKey(x, 0));
        }
        let mut popped = Vec::new();
        while let Some(MinKey(x, _)) = heap.pop() {
            popped.push(x);
        }
        assert_eq!(popped[0], 1.0);
        assert_eq!(popped[1], 2.0);
        assert_eq!(popped[2], 3.0);
        assert!(popped[3].is_infinite() || popped[3].is_nan());
        assert!(popped.iter().any(|x| x.is_nan()));
    }
}
