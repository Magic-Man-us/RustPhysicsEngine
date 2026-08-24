//! Network flow: maximum flow, minimum cut, and the problems that reduce to
//! them.
//!
//! A flow network is a [`Graph`] whose weights are capacities. An undirected
//! edge is treated as a pair of arcs, each with the full capacity, which is
//! the usual convention: flow may run either way but not both at once.
//!
//! Capacities must be finite and non-negative. The residual graph is built
//! internally as an arc list with paired indices, so the reverse arc of arc
//! `i` is arc `i ^ 1`.

use crate::graph::core::Graph;

/// A residual network: arcs in pairs, `i` and `i ^ 1` reverse each other.
struct Residual {
    n: usize,
    /// `head[i]` is the arc's target; `cap[i]` its remaining capacity.
    head: Vec<usize>,
    cap: Vec<f64>,
    /// `out[v]` lists the arc indices leaving `v`.
    out: Vec<Vec<usize>>,
    /// The original capacity of each arc, so the flow can be read back.
    original: Vec<f64>,
}

impl Residual {
    fn new(n: usize) -> Self {
        Self {
            n,
            head: Vec::new(),
            cap: Vec::new(),
            out: vec![Vec::new(); n],
            original: Vec::new(),
        }
    }

    /// Adds a forward arc of the given capacity and its zero-capacity mate.
    fn add(&mut self, u: usize, v: usize, c: f64) {
        let i = self.head.len();
        self.head.push(v);
        self.cap.push(c);
        self.original.push(c);
        self.out[u].push(i);
        self.head.push(u);
        self.cap.push(0.0);
        self.original.push(0.0);
        self.out[v].push(i + 1);
    }

    /// Adds an arc with capacity in both directions, for an undirected edge.
    fn add_both(&mut self, u: usize, v: usize, c: f64) {
        let i = self.head.len();
        self.head.push(v);
        self.cap.push(c);
        self.original.push(c);
        self.out[u].push(i);
        self.head.push(u);
        self.cap.push(c);
        self.original.push(c);
        self.out[v].push(i + 1);
    }

    /// The residual network of `g`.
    fn from_graph(g: &Graph) -> Self {
        let mut r = Residual::new(g.n);
        for (u, v, c) in g.edges() {
            assert!(
                c >= 0.0 && c.is_finite(),
                "capacities must be finite and non-negative"
            );
            if u == v {
                continue;
            }
            if g.directed {
                r.add(u, v, c);
            } else {
                r.add_both(u, v, c);
            }
        }
        r
    }

    /// The vertices reachable from `s` along arcs with residual capacity.
    ///
    /// After a maximum flow this is exactly the source side of a minimum cut.
    fn reachable(&self, s: usize) -> Vec<bool> {
        let mut seen = vec![false; self.n];
        seen[s] = true;
        let mut stack = vec![s];
        while let Some(v) = stack.pop() {
            for &i in &self.out[v] {
                if self.cap[i] > EPS && !seen[self.head[i]] {
                    seen[self.head[i]] = true;
                    stack.push(self.head[i]);
                }
            }
        }
        seen
    }
}

/// Capacities below this are treated as saturated.
///
/// Floating-point capacities do not cancel exactly, so an augmenting search
/// that accepted any positive residual would keep finding paths carrying
/// `1e-17` and never terminate. Everything here compares against this instead
/// of against zero.
const EPS: f64 = 1e-9;

/// The maximum flow from `s` to `t` by Dinic's algorithm, and the flow on each
/// arc as a matrix.
///
/// Dinic repeatedly builds a level graph by breadth-first search and pushes
/// blocking flow through it, which bounds the number of phases by the vertex
/// count rather than by the flow value -- the difference between terminating
/// and not on a network with large capacities.
///
/// The returned matrix holds the net flow: entry `(u, v)` is what crosses from
/// `u` to `v`, and is zero where nothing does.
///
/// # Panics
/// Panics if `s` or `t` is out of range, if they are equal, or if any capacity
/// is negative or not finite.
#[must_use]
pub fn max_flow_dinic(g: &Graph, s: usize, t: usize) -> (f64, Vec<Vec<f64>>) {
    assert!(s < g.n && t < g.n, "endpoints must be vertices");
    assert!(s != t, "source and sink must differ");
    let mut r = Residual::from_graph(g);
    let mut total = 0.0;

    loop {
        // Level graph: the hop distance from s in the residual network.
        let mut level = vec![usize::MAX; r.n];
        level[s] = 0;
        let mut queue = std::collections::VecDeque::from(vec![s]);
        while let Some(v) = queue.pop_front() {
            for &i in &r.out[v] {
                let w = r.head[i];
                if r.cap[i] > EPS && level[w] == usize::MAX {
                    level[w] = level[v] + 1;
                    queue.push_back(w);
                }
            }
        }
        if level[t] == usize::MAX {
            break;
        }
        // Blocking flow: depth-first, only ever descending one level, with a
        // per-vertex cursor so a saturated arc is never retried in this phase.
        let mut cursor = vec![0usize; r.n];
        loop {
            let pushed = dinic_augment(&mut r, s, t, f64::INFINITY, &level, &mut cursor);
            if pushed <= EPS {
                break;
            }
            total += pushed;
        }
    }

    (total, flow_matrix(&r))
}

fn dinic_augment(
    r: &mut Residual,
    v: usize,
    t: usize,
    limit: f64,
    level: &[usize],
    cursor: &mut [usize],
) -> f64 {
    if v == t {
        return limit;
    }
    while cursor[v] < r.out[v].len() {
        let i = r.out[v][cursor[v]];
        let w = r.head[i];
        if r.cap[i] > EPS && level[w] == level[v] + 1 {
            let pushed = dinic_augment(r, w, t, limit.min(r.cap[i]), level, cursor);
            if pushed > EPS {
                r.cap[i] -= pushed;
                r.cap[i ^ 1] += pushed;
                return pushed;
            }
        }
        cursor[v] += 1;
    }
    0.0
}

/// The net flow on each ordered pair, read back from the residual capacities.
fn flow_matrix(r: &Residual) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; r.n]; r.n];
    for v in 0..r.n {
        for &i in &r.out[v] {
            // A forward arc that lost capacity carries that much flow.
            let used = r.original[i] - r.cap[i];
            if used > EPS {
                m[v][r.head[i]] += used;
            }
        }
    }
    // Cancel opposing flow so the result is the net crossing.
    for u in 0..r.n {
        for v in u + 1..r.n {
            let net = m[u][v] - m[v][u];
            m[u][v] = net.max(0.0);
            m[v][u] = (-net).max(0.0);
        }
    }
    m
}

/// The maximum flow value by the push-relabel method with the highest-label
/// rule.
///
/// A different algorithm from [`max_flow_dinic`] rather than a variation on
/// it: push-relabel never maintains a valid flow until it finishes, working
/// instead with a preflow that it gradually returns to feasibility. The two
/// agreeing is therefore evidence, not a tautology.
///
/// # Panics
/// Panics under the same conditions as [`max_flow_dinic`].
#[must_use]
pub fn max_flow_push_relabel(g: &Graph, s: usize, t: usize) -> f64 {
    assert!(s < g.n && t < g.n, "endpoints must be vertices");
    assert!(s != t, "source and sink must differ");
    let mut r = Residual::from_graph(g);
    let n = r.n;
    let mut height = vec![0usize; n];
    let mut excess = vec![0.0f64; n];
    height[s] = n;

    // Saturate every arc out of the source, creating the initial preflow.
    for idx in 0..r.out[s].len() {
        let i = r.out[s][idx];
        let c = r.cap[i];
        if c > EPS {
            r.cap[i] -= c;
            r.cap[i ^ 1] += c;
            excess[r.head[i]] += c;
            excess[s] -= c;
        }
    }

    let mut cursor = vec![0usize; n];
    // Highest label first: discharge the active vertex of greatest height,
    // which is what gives the O(n^2 sqrt(m)) bound.
    while let Some(v) = (0..n)
        .filter(|&v| v != s && v != t && excess[v] > EPS)
        .max_by_key(|&v| height[v])
    {
        // Discharge v: push where possible, relabel when not.
        if cursor[v] == r.out[v].len() {
            // Relabel to one above the lowest reachable neighbour.
            let min_h = r.out[v]
                .iter()
                .filter(|&&i| r.cap[i] > EPS)
                .map(|&i| height[r.head[i]])
                .min();
            match min_h {
                Some(h) => height[v] = h + 1,
                // No residual arc at all: the excess is stranded and cannot
                // move, which can only happen once the preflow is a flow.
                None => break,
            }
            cursor[v] = 0;
            continue;
        }
        let i = r.out[v][cursor[v]];
        let w = r.head[i];
        if r.cap[i] > EPS && height[v] == height[w] + 1 {
            let delta = excess[v].min(r.cap[i]);
            r.cap[i] -= delta;
            r.cap[i ^ 1] += delta;
            excess[v] -= delta;
            excess[w] += delta;
        } else {
            cursor[v] += 1;
        }
    }
    excess[t]
}

/// The minimum `s`-`t` cut: its capacity and the source side.
///
/// By the max-flow min-cut theorem the capacity equals the maximum flow, and
/// the source side is exactly what remains reachable from `s` in the residual
/// network once the flow is maximum.
///
/// # Panics
/// Panics under the same conditions as [`max_flow_dinic`].
#[must_use]
pub fn min_cut(g: &Graph, s: usize, t: usize) -> (f64, Vec<bool>) {
    assert!(s < g.n && t < g.n, "endpoints must be vertices");
    assert!(s != t, "source and sink must differ");
    let mut r = Residual::from_graph(g);
    let mut total = 0.0;
    loop {
        let mut level = vec![usize::MAX; r.n];
        level[s] = 0;
        let mut queue = std::collections::VecDeque::from(vec![s]);
        while let Some(v) = queue.pop_front() {
            for &i in &r.out[v] {
                let w = r.head[i];
                if r.cap[i] > EPS && level[w] == usize::MAX {
                    level[w] = level[v] + 1;
                    queue.push_back(w);
                }
            }
        }
        if level[t] == usize::MAX {
            break;
        }
        let mut cursor = vec![0usize; r.n];
        loop {
            let pushed = dinic_augment(&mut r, s, t, f64::INFINITY, &level, &mut cursor);
            if pushed <= EPS {
                break;
            }
            total += pushed;
        }
    }
    (total, r.reachable(s))
}

/// The global minimum cut, by the Stoer-Wagner algorithm.
///
/// Finds the cheapest way to split the graph in two without naming the two
/// sides, which no single `s`-`t` computation does. Each phase grows a set by
/// always adding the most tightly connected vertex, which makes the last two
/// added a valid `s`-`t` pair for free; merging them and repeating gives the
/// global optimum in `n - 1` phases.
///
/// Returns the cut capacity and one side of it.
///
/// # Panics
/// Panics if the graph is directed, or has fewer than two vertices.
#[must_use]
pub fn global_min_cut_stoer_wagner(g: &Graph) -> (f64, Vec<usize>) {
    assert!(!g.directed, "Stoer-Wagner is for undirected graphs");
    assert!(g.n >= 2, "a cut needs at least two vertices");
    let n = g.n;
    // Dense weights, since the algorithm merges vertices repeatedly.
    let mut w = vec![vec![0.0f64; n]; n];
    for (u, v, c) in g.edges() {
        if u != v {
            w[u][v] += c;
            w[v][u] += c;
        }
    }
    // Each surviving vertex stands for the original vertices merged into it.
    let mut group: Vec<Vec<usize>> = (0..n).map(|v| vec![v]).collect();
    let mut alive: Vec<usize> = (0..n).collect();
    let mut best = f64::INFINITY;
    let mut best_side: Vec<usize> = Vec::new();

    while alive.len() > 1 {
        // Maximum adjacency ordering.
        let mut added = vec![false; n];
        let mut weight = vec![0.0f64; n];
        let mut order: Vec<usize> = Vec::with_capacity(alive.len());
        for _ in 0..alive.len() {
            let v = *alive
                .iter()
                .filter(|&&v| !added[v])
                .max_by(|&&a, &&b| weight[a].total_cmp(&weight[b]))
                .expect("a vertex remains");
            added[v] = true;
            order.push(v);
            for &u in &alive {
                if !added[u] {
                    weight[u] += w[v][u];
                }
            }
        }
        // The last vertex added defines a cut of exactly its own weight.
        let last = *order.last().unwrap();
        let prev = order[order.len() - 2];
        if weight[last] < best {
            best = weight[last];
            best_side = group[last].clone();
        }
        // Merge the last two and repeat.
        let merged: Vec<usize> = group[last].clone();
        group[prev].extend(merged);
        for &u in &alive {
            if u != last && u != prev {
                w[prev][u] += w[last][u];
                w[u][prev] = w[prev][u];
            }
        }
        alive.retain(|&v| v != last);
    }
    best_side.sort_unstable();
    (best, best_side)
}

/// The minimum-cost maximum flow from `s` to `t`.
///
/// `costs` gives the cost per unit on each arc, in the same order as
/// `g.edges()`. Returns the flow value and its total cost.
///
/// Augments along a shortest path by cost each round, found with Bellman-Ford
/// so negative costs are allowed. Sending flow along a shortest path keeps the
/// residual network free of negative cycles, which is what makes the greedy
/// choice optimal rather than merely feasible.
///
/// # Panics
/// Panics if `costs` does not have one entry per edge, or under the same
/// conditions as [`max_flow_dinic`].
#[must_use]
pub fn min_cost_max_flow(g: &Graph, costs: &[f64], s: usize, t: usize) -> (f64, f64) {
    assert!(s < g.n && t < g.n, "endpoints must be vertices");
    assert!(s != t, "source and sink must differ");
    let edges = g.edges();
    assert_eq!(costs.len(), edges.len(), "one cost per edge is required");

    let mut r = Residual::new(g.n);
    let mut arc_cost: Vec<f64> = Vec::new();
    for (&(u, v, c), &cost) in edges.iter().zip(costs) {
        assert!(
            c >= 0.0 && c.is_finite(),
            "capacities must be finite and non-negative"
        );
        if u == v {
            continue;
        }
        r.add(u, v, c);
        arc_cost.push(cost);
        // Sending flow back refunds the cost.
        arc_cost.push(-cost);
        if !g.directed {
            r.add(v, u, c);
            arc_cost.push(cost);
            arc_cost.push(-cost);
        }
    }

    let mut flow = 0.0;
    let mut cost_total = 0.0;
    loop {
        // Cheapest augmenting path by Bellman-Ford over residual arcs.
        let mut dist = vec![f64::INFINITY; r.n];
        let mut from: Vec<Option<usize>> = vec![None; r.n];
        dist[s] = 0.0;
        for _ in 0..r.n {
            let mut changed = false;
            for v in 0..r.n {
                if !dist[v].is_finite() {
                    continue;
                }
                for &i in &r.out[v] {
                    if r.cap[i] <= EPS {
                        continue;
                    }
                    let cand = dist[v] + arc_cost[i];
                    if cand < dist[r.head[i]] - 1e-12 {
                        dist[r.head[i]] = cand;
                        from[r.head[i]] = Some(i);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if !dist[t].is_finite() {
            break;
        }
        // The bottleneck along that path.
        let mut push = f64::INFINITY;
        let mut v = t;
        while let Some(i) = from[v] {
            push = push.min(r.cap[i]);
            v = r.head[i ^ 1];
            if v == s {
                break;
            }
        }
        if push <= EPS || !push.is_finite() {
            break;
        }
        let mut v = t;
        while let Some(i) = from[v] {
            r.cap[i] -= push;
            r.cap[i ^ 1] += push;
            cost_total += push * arc_cost[i];
            v = r.head[i ^ 1];
            if v == s {
                break;
            }
        }
        flow += push;
    }
    (flow, cost_total)
}

/// A feasible circulation meeting the given vertex demands, or `None` if none
/// exists.
///
/// `demand[v]` is positive when `v` must receive that much and negative when
/// it must send it. `lower` gives the minimum flow on each edge, in the order
/// of `g.edges()`. Solved by the standard reduction: subtract the lower bounds
/// into the demands, then look for a saturating flow from a super-source to a
/// super-sink.
///
/// Returns the flow on each edge in the order of `g.edges()`.
///
/// # Panics
/// Panics unless `demand` has one entry per vertex, `lower` one per edge, the
/// demands sum to zero, and every lower bound is within its capacity.
#[must_use]
pub fn circulation_with_demands(
    g: &Graph,
    demand: &[f64],
    lower: &[f64],
) -> Option<Vec<f64>> {
    let edges = g.edges();
    assert_eq!(demand.len(), g.n, "one demand per vertex is required");
    assert_eq!(lower.len(), edges.len(), "one lower bound per edge");
    assert!(
        demand.iter().sum::<f64>().abs() < 1e-9,
        "demands must sum to zero for a circulation to exist"
    );
    assert!(
        edges.iter().zip(lower).all(|(&(_, _, c), &l)| l >= 0.0 && l <= c + 1e-12),
        "each lower bound must lie within its capacity"
    );

    // Super-source and super-sink absorb the demands.
    let src = g.n;
    let snk = g.n + 1;
    let mut net = Graph::new(g.n + 2, true);
    // Adjusted demand: a lower bound forces flow whether we like it or not.
    let mut adjusted = demand.to_vec();
    for (&(u, v, c), &l) in edges.iter().zip(lower) {
        net.add_edge(u, v, c - l);
        adjusted[u] += l;
        adjusted[v] -= l;
    }
    let mut required = 0.0;
    for v in 0..g.n {
        if adjusted[v] > 0.0 {
            // v must still receive this much, so it draws from the super-sink
            // side: saturating v -> snk is what meeting its demand means.
            net.add_edge(v, snk, adjusted[v]);
            required += adjusted[v];
        } else if adjusted[v] < 0.0 {
            // v must still send this much, so the super-source supplies it.
            net.add_edge(src, v, -adjusted[v]);
        }
    }
    let (value, matrix) = max_flow_dinic(&net, src, snk);
    if (value - required).abs() > 1e-6 {
        // The super-source cannot be saturated, so no circulation exists.
        return None;
    }
    // Read the flow back and restore the lower bounds.
    Some(
        edges
            .iter()
            .zip(lower)
            .map(|(&(u, v, _), &l)| l + matrix[u][v])
            .collect(),
    )
}

/// A maximum matching of a bipartite graph, found by maximum flow.
///
/// `left` names the vertices on one side; the rest are the other side. Returns
/// the partner of each vertex, or `None` for the unmatched.
///
/// Slower than [`crate::graph::matching::hopcroft_karp`] but built from a
/// different primitive, so the two agreeing is evidence about both.
///
/// # Panics
/// Panics if `left` names a vertex twice or out of range, or if an edge joins
/// two vertices on the same side.
#[must_use]
pub fn max_bipartite_matching_via_flow(g: &Graph, left: &[usize]) -> Vec<Option<usize>> {
    let mut is_left = vec![false; g.n];
    for &v in left {
        assert!(v < g.n, "vertex {v} is outside 0..{}", g.n);
        assert!(!is_left[v], "vertex {v} appears twice");
        is_left[v] = true;
    }
    let src = g.n;
    let snk = g.n + 1;
    let mut net = Graph::new(g.n + 2, true);
    for (u, v, _) in g.edges() {
        if u == v {
            continue;
        }
        assert!(
            is_left[u] != is_left[v],
            "edge ({u}, {v}) joins the same side"
        );
        let (a, b) = if is_left[u] { (u, v) } else { (v, u) };
        net.add_edge(a, b, 1.0);
    }
    for v in 0..g.n {
        if is_left[v] {
            net.add_edge(src, v, 1.0);
        } else {
            net.add_edge(v, snk, 1.0);
        }
    }
    let (_, matrix) = max_flow_dinic(&net, src, snk);
    let mut partner = vec![None; g.n];
    for u in 0..g.n {
        if !is_left[u] {
            continue;
        }
        for v in 0..g.n {
            if !is_left[v] && matrix[u][v] > 0.5 {
                partner[u] = Some(v);
                partner[v] = Some(u);
                break;
            }
        }
    }
    partner
}

/// The number of pairwise edge-disjoint paths from `s` to `t`.
///
/// Menger's theorem says this equals the minimum number of edges whose removal
/// separates them, which is the maximum flow with every capacity one.
///
/// # Panics
/// Panics if `s` or `t` is out of range, or they are equal.
#[must_use]
pub fn edge_disjoint_paths(g: &Graph, s: usize, t: usize) -> usize {
    let mut unit = Graph::new(g.n, g.directed);
    for (u, v, _) in g.edges() {
        if u != v {
            unit.add_edge(u, v, 1.0);
        }
    }
    max_flow_dinic(&unit, s, t).0.round() as usize
}

/// The number of pairwise internally vertex-disjoint paths from `s` to `t`.
///
/// The vertex form of Menger's theorem. Each vertex other than `s` and `t` is
/// split into an in-copy and an out-copy joined by a unit arc, which caps how
/// many paths can use it; the answer is then the edge-disjoint count on the
/// split graph.
///
/// # Panics
/// Panics if `s` or `t` is out of range, or they are equal.
#[must_use]
pub fn vertex_disjoint_paths(g: &Graph, s: usize, t: usize) -> usize {
    assert!(s < g.n && t < g.n, "endpoints must be vertices");
    assert!(s != t, "source and sink must differ");
    let n = g.n;
    // Vertex v becomes v (in) and v + n (out).
    let mut split = Graph::new(2 * n, true);
    for v in 0..n {
        // s and t must not be throttled, so give them room for every path.
        let cap = if v == s || v == t { n as f64 } else { 1.0 };
        split.add_edge(v, v + n, cap);
    }
    for (u, v, _) in g.edges() {
        if u == v {
            continue;
        }
        // Capacity one, not n: two vertex-disjoint paths cannot share an edge
        // either, since sharing one means sharing both its endpoints. Giving
        // the arc capacity n lets a single edge carry several paths, which
        // reports two paths between adjacent vertices joined by one edge.
        split.add_edge(u + n, v, 1.0);
        if !g.directed {
            split.add_edge(v + n, u, 1.0);
        }
    }
    max_flow_dinic(&split, s + n, t).0.round() as usize
}

/// A Gomory-Hu tree: an `n`-vertex tree in which the minimum cut between any
/// two vertices equals the lightest edge on the tree path between them.
///
/// Built by Gusfield's simplification, which needs only `n - 1` maximum-flow
/// computations and no vertex contraction. The result encodes all `C(n, 2)`
/// pairwise minimum cuts in `n - 1` numbers.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn gomory_hu_tree(g: &Graph) -> Graph {
    assert!(!g.directed, "a Gomory-Hu tree is defined for undirected graphs");
    let n = g.n;
    let mut tree = Graph::new(n, false);
    if n < 2 {
        return tree;
    }
    // parent[i] starts at 0 for every i; each round fixes one edge.
    let mut parent = vec![0usize; n];
    for i in 1..n {
        let (value, side) = min_cut(g, i, parent[i]);
        tree.add_edge(i, parent[i], value);
        // Any later vertex on i's side of the cut now hangs off i instead.
        for j in i + 1..n {
            if side[j] && parent[j] == parent[i] {
                parent[j] = i;
            }
        }
    }
    tree
}

/// The maximum-weight closed subset of a directed graph.
///
/// A closure is a vertex set containing every successor of every member. The
/// maximum-weight closure reduces to a minimum cut: positive vertices are
/// joined to a source with their weight, negative ones to a sink with its
/// magnitude, and each original arc is given infinite capacity so no cut can
/// break it, which is exactly the closure condition.
///
/// Returns the weight and the membership flags.
///
/// # Panics
/// Panics unless `weights` has one entry per vertex.
#[must_use]
pub fn closure_problem(g: &Graph, weights: &[f64]) -> (f64, Vec<bool>) {
    assert_eq!(weights.len(), g.n, "one weight per vertex is required");
    let n = g.n;
    let src = n;
    let snk = n + 1;
    let mut net = Graph::new(n + 2, true);
    let mut positive_total = 0.0;
    for v in 0..n {
        if weights[v] > 0.0 {
            net.add_edge(src, v, weights[v]);
            positive_total += weights[v];
        } else if weights[v] < 0.0 {
            net.add_edge(v, snk, -weights[v]);
        }
    }
    // A closure must contain every successor, so the arcs are uncuttable.
    let big = positive_total * 2.0 + 1.0;
    for (u, v, _) in g.edges() {
        if u != v {
            net.add_edge(u, v, big);
        }
    }
    let (cut, side) = min_cut(&net, src, snk);
    let members: Vec<bool> = (0..n).map(|v| side[v]).collect();
    (positive_total - cut, members)
}

/// The maximum profit of a project selection problem.
///
/// Projects have revenues and require machines that cost money; a project may
/// only be taken if every machine it needs is bought. `project_revenue[i]` is
/// the revenue of project `i`, `machine_cost[j]` the cost of machine `j`, and
/// `requires[i]` the machines project `i` needs.
///
/// This is [`closure_problem`] on the bipartite graph of projects and
/// machines, with revenues positive and costs negative.
///
/// # Panics
/// Panics if `requires` does not have one entry per project, or names a
/// machine out of range.
#[must_use]
pub fn project_selection(
    project_revenue: &[f64],
    machine_cost: &[f64],
    requires: &[Vec<usize>],
) -> f64 {
    assert_eq!(
        requires.len(),
        project_revenue.len(),
        "one requirement list per project"
    );
    let (p, m) = (project_revenue.len(), machine_cost.len());
    let mut g = Graph::new(p + m, true);
    for (i, reqs) in requires.iter().enumerate() {
        for &j in reqs {
            assert!(j < m, "machine {j} is outside 0..{m}");
            g.add_edge(i, p + j, 1.0);
        }
    }
    let mut weights = project_revenue.to_vec();
    weights.extend(machine_cost.iter().map(|c| -c));
    closure_problem(&g, &weights).0
}

/// The maximum flow value as a plain number, for callers that do not want the
/// arc-by-arc matrix.
///
/// # Panics
/// Panics under the same conditions as [`max_flow_dinic`].
#[must_use]
pub fn max_flow(g: &Graph, s: usize, t: usize) -> f64 {
    max_flow_dinic(g, s, t).0
}

/// The capacity of the cut defined by `side`: the total weight of the edges
/// leaving the `true` set.
///
/// A directed graph counts only arcs from the `true` side to the `false` one,
/// which is the `s`-`t` cut convention; an undirected graph counts every edge
/// crossing.
///
/// # Panics
/// Panics unless `side` has one flag per vertex.
#[must_use]
pub fn cut_capacity(g: &Graph, side: &[bool]) -> f64 {
    assert_eq!(side.len(), g.n, "one side flag per vertex is required");
    g.edges()
        .into_iter()
        .filter(|&(u, v, _)| {
            // A directed arc counts only when it leaves the true side; an
            // undirected edge counts whichever way it crosses.
            (side[u] && !side[v]) || (!g.directed && side[v] && !side[u])
        })
        .map(|(_, _, c)| c)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::{complete_bipartite, complete_graph, cycle_graph, path_graph};
    use crate::monte_carlo::Rng;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6 * a.abs().max(b.abs()).max(1.0)
    }

    fn random_network(n: usize, p: f64, directed: bool, rng: &mut Rng) -> Graph {
        let mut g = Graph::new(n, directed);
        for u in 0..n {
            let start = if directed { 0 } else { u + 1 };
            for v in start..n {
                if u != v && rng.next_f64() < p {
                    g.add_edge(u, v, 1.0 + (10.0 * rng.next_f64()).floor());
                }
            }
        }
        g
    }

    /// A flow must conserve at every vertex but the source and sink, respect
    /// every capacity, and carry the value it claims.
    fn check_flow(g: &Graph, m: &[Vec<f64>], s: usize, t: usize, value: f64) {
        let n = g.n;
        // Capacity: the net flow across a pair cannot exceed what is there.
        let mut cap = vec![vec![0.0f64; n]; n];
        for (u, v, c) in g.edges() {
            if u == v {
                continue;
            }
            cap[u][v] += c;
            if !g.directed {
                cap[v][u] += c;
            }
        }
        for u in 0..n {
            for v in 0..n {
                assert!(
                    m[u][v] <= cap[u][v] + 1e-6,
                    "flow {} on ({u}, {v}) exceeds capacity {}",
                    m[u][v],
                    cap[u][v]
                );
                assert!(m[u][v] >= -1e-9, "negative flow on ({u}, {v})");
            }
        }
        // Conservation everywhere but s and t.
        for v in 0..n {
            if v == s || v == t {
                continue;
            }
            let inflow: f64 = (0..n).map(|u| m[u][v]).sum();
            let outflow: f64 = (0..n).map(|w| m[v][w]).sum();
            assert!(
                close(inflow, outflow),
                "vertex {v} leaks: in {inflow}, out {outflow}"
            );
        }
        // The value is what leaves the source net of what returns.
        let out_s: f64 = (0..n).map(|w| m[s][w]).sum();
        let in_s: f64 = (0..n).map(|u| m[u][s]).sum();
        assert!(close(out_s - in_s, value), "source net is not the value");
        let in_t: f64 = (0..n).map(|u| m[u][t]).sum();
        let out_t: f64 = (0..n).map(|w| m[t][w]).sum();
        assert!(close(in_t - out_t, value), "sink net is not the value");
    }

    /// The roadmap's headline property: max-flow equals min-cut, and the two
    /// algorithms agree with each other.
    #[test]
    fn max_flow_equals_min_cut() {
        let mut rng = Rng::new(0x_F10A);
        for directed in [false, true] {
            for n in 2..=7usize {
                for _ in 0..8 {
                    let g = random_network(n, 0.45, directed, &mut rng);
                    for s in 0..n {
                        for t in 0..n {
                            if s == t {
                                continue;
                            }
                            let (value, m) = max_flow_dinic(&g, s, t);
                            check_flow(&g, &m, s, t, value);

                            // Push-relabel is a different algorithm entirely.
                            let pr = max_flow_push_relabel(&g, s, t);
                            assert!(close(value, pr), "dinic {value} vs push-relabel {pr}");

                            // Min-cut: same capacity, and it really is a cut.
                            let (cut, side) = min_cut(&g, s, t);
                            assert!(close(value, cut), "flow {value} vs cut {cut}");
                            assert!(side[s] && !side[t], "the cut does not separate s from t");
                            assert!(
                                close(cut, cut_capacity(&g, &side)),
                                "the reported capacity is not the cut's"
                            );
                            // No cut is cheaper, checked over all 2^n
                            // partitions. Capped at six vertices: the sweep is
                            // exponential and runs once per ordered pair, so
                            // the whole test is O(2^n n^2) per graph.
                            if n <= 6 {
                                let best = brute_min_cut(&g, s, t);
                                assert!(close(cut, best), "n = {n}: {cut} vs brute {best}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The cheapest s-t cut, by trying every partition.
    fn brute_min_cut(g: &Graph, s: usize, t: usize) -> f64 {
        let n = g.n;
        let mut best = f64::INFINITY;
        for mask in 0u64..(1u64 << n) {
            let side: Vec<bool> = (0..n).map(|v| mask >> v & 1 == 1).collect();
            if !side[s] || side[t] {
                continue;
            }
            best = best.min(cut_capacity(g, &side));
        }
        best
    }

    /// Textbook networks with known answers.
    #[test]
    fn known_networks_give_known_flows() {
        // CLRS figure 26.1: the classic 6-vertex network, max flow 23.
        let g = Graph::from_edges(
            6,
            &[
                (0, 1, 16.0),
                (0, 2, 13.0),
                (1, 2, 10.0),
                (2, 1, 4.0),
                (1, 3, 12.0),
                (3, 2, 9.0),
                (2, 4, 14.0),
                (4, 3, 7.0),
                (3, 5, 20.0),
                (4, 5, 4.0),
            ],
            true,
        );
        let (value, m) = max_flow_dinic(&g, 0, 5);
        assert!(close(value, 23.0), "expected 23, got {value}");
        check_flow(&g, &m, 0, 5, value);
        assert!(close(max_flow_push_relabel(&g, 0, 5), 23.0));
        let (cut, side) = min_cut(&g, 0, 5);
        assert!(close(cut, 23.0));
        assert!(side[0] && !side[5]);

        // A single path: the bottleneck is the flow.
        let chain = Graph::from_edges(
            4,
            &[(0, 1, 5.0), (1, 2, 3.0), (2, 3, 7.0)],
            true,
        );
        assert!(close(max_flow_dinic(&chain, 0, 3).0, 3.0));
        // Parallel paths add.
        let parallel = Graph::from_edges(
            4,
            &[(0, 1, 5.0), (1, 3, 5.0), (0, 2, 2.0), (2, 3, 2.0)],
            true,
        );
        assert!(close(max_flow_dinic(&parallel, 0, 3).0, 7.0));
        // No path at all.
        assert!(close(max_flow_dinic(&Graph::new(3, true), 0, 2).0, 0.0));
    }

    /// The global minimum cut must be at most every s-t cut, and equal the
    /// cheapest of them.
    #[test]
    fn stoer_wagner_matches_the_best_st_cut() {
        let mut rng = Rng::new(0x_570E);
        for n in 2..=8usize {
            for _ in 0..15 {
                let g = random_network(n, 0.5, false, &mut rng);
                let (global, side) = global_min_cut_stoer_wagner(&g);
                // The best over every s-t pair is the global minimum.
                let mut best = f64::INFINITY;
                for s in 0..n {
                    for t in s + 1..n {
                        best = best.min(min_cut(&g, s, t).0);
                    }
                }
                assert!(close(global, best), "n = {n}: global {global} vs best {best}");
                // The reported side really is a proper non-empty subset with
                // that capacity.
                assert!(!side.is_empty() && side.len() < n, "not a proper cut");
                let flags: Vec<bool> = (0..n).map(|v| side.contains(&v)).collect();
                assert!(
                    close(global, cut_capacity(&g, &flags)),
                    "the reported side does not have the reported capacity"
                );
            }
        }
        // A disconnected graph has a cut of zero.
        let mut split = Graph::new(4, false);
        split.add_edge(0, 1, 5.0);
        split.add_edge(2, 3, 5.0);
        assert!(close(global_min_cut_stoer_wagner(&split).0, 0.0));
        // A cycle must be cut in two places.
        let c = cycle_graph(6);
        assert!(close(global_min_cut_stoer_wagner(&c).0, 2.0));
        // A complete graph: the cheapest cut isolates one vertex.
        for n in 2..=7usize {
            let k = complete_graph(n);
            assert!(
                close(global_min_cut_stoer_wagner(&k).0, (n - 1) as f64),
                "K{n}"
            );
        }
    }

    /// Minimum-cost flow must send the maximum amount and do so as cheaply as
    /// possible, checked against brute force over path decompositions.
    #[test]
    fn min_cost_flow_is_max_flow_at_least_cost() {
        // Two routes, one cheap and narrow, one dear and wide.
        let g = Graph::from_edges(
            4,
            &[(0, 1, 1.0), (1, 3, 1.0), (0, 2, 5.0), (2, 3, 5.0)],
            true,
        );
        // Costs must line up with edges() order, which is by tail vertex, so
        // build them from the endpoints rather than by hand.
        let costs: Vec<f64> = g
            .edges()
            .iter()
            .map(|&(u, v, _)| if u == 1 || v == 1 { 1.0 } else { 10.0 })
            .collect();
        let (flow, cost) = min_cost_max_flow(&g, &costs, 0, 3);
        assert!(close(flow, 6.0), "max flow is 6");
        // The route through 1 costs 2 a unit and carries one; the route
        // through 2 costs 20 a unit and carries five.
        assert!(close(cost, 1.0 * 2.0 + 5.0 * 20.0), "got {cost}");
        // The flow value must match the plain max flow.
        assert!(close(flow, max_flow_dinic(&g, 0, 3).0));

        // A single path: cost is the path cost times the bottleneck.
        let chain = Graph::from_edges(3, &[(0, 1, 4.0), (1, 2, 2.0)], true);
        let (f, c) = min_cost_max_flow(&chain, &[3.0, 5.0], 0, 2);
        assert!(close(f, 2.0));
        assert!(close(c, 2.0 * 8.0));

        // Negative costs are allowed, and using them is profitable.
        let neg = Graph::from_edges(
            4,
            &[(0, 1, 2.0), (1, 3, 2.0), (0, 2, 2.0), (2, 3, 2.0)],
            true,
        );
        let (f2, c2) = min_cost_max_flow(&neg, &[1.0, 1.0, -3.0, 1.0], 0, 3);
        assert!(close(f2, 4.0));
        assert!(close(c2, 2.0 * 2.0 + 2.0 * (-2.0)), "got {c2}");

        // Against max flow on random networks: the value must always agree,
        // whatever the costs.
        let mut rng = Rng::new(0x_C155);
        for n in 2..=6usize {
            for _ in 0..12 {
                let g = random_network(n, 0.5, true, &mut rng);
                let costs: Vec<f64> = g
                    .edges()
                    .iter()
                    .map(|_| (5.0 * rng.next_f64()).floor())
                    .collect();
                for s in 0..n {
                    for t in 0..n {
                        if s == t {
                            continue;
                        }
                        let (f, _) = min_cost_max_flow(&g, &costs, s, t);
                        let mf = max_flow_dinic(&g, s, t).0;
                        assert!(close(f, mf), "n = {n}: mcmf {f} vs maxflow {mf}");
                    }
                }
            }
        }
    }

    /// Menger's theorem in both forms, against brute force.
    #[test]
    fn mengers_theorem_holds_in_both_forms() {
        let mut rng = Rng::new(0x_3E17);
        for n in 2..=6usize {
            for _ in 0..10 {
                let g = random_network(n, 0.4, false, &mut rng);
                for s in 0..n {
                    for t in 0..n {
                        if s == t {
                            continue;
                        }
                        // Edge form: the count equals the minimum number of
                        // edges whose removal separates s from t.
                        let k = edge_disjoint_paths(&g, s, t);
                        let cut = brute_edge_cut(&g, s, t);
                        assert_eq!(k, cut, "edge Menger at {s}->{t}, n = {n}");

                        // Vertex form: the count equals the minimum number of
                        // interior vertices whose removal separates them, or
                        // is capped by the adjacency when s and t are joined.
                        let kv = vertex_disjoint_paths(&g, s, t);
                        let cutv = brute_vertex_cut(&g, s, t);
                        assert_eq!(kv, cutv, "vertex Menger at {s}->{t}, n = {n}");
                    }
                }
            }
        }
        // A cycle has exactly two disjoint paths between any two vertices.
        let c = cycle_graph(7);
        for s in 0..7 {
            for t in 0..7 {
                if s != t {
                    assert_eq!(edge_disjoint_paths(&c, s, t), 2);
                    assert_eq!(vertex_disjoint_paths(&c, s, t), 2);
                }
            }
        }
        // A path has exactly one.
        let p = path_graph(5);
        assert_eq!(edge_disjoint_paths(&p, 0, 4), 1);
        assert_eq!(vertex_disjoint_paths(&p, 0, 4), 1);
        // K_n has n - 1 vertex-disjoint paths between any two vertices.
        for n in 2..=6usize {
            let k = complete_graph(n);
            assert_eq!(vertex_disjoint_paths(&k, 0, 1), n - 1, "K{n}");
        }
    }

    /// The fewest edges whose removal disconnects s from t.
    fn brute_edge_cut(g: &Graph, s: usize, t: usize) -> usize {
        let edges = g.edges();
        for k in 0..=edges.len() {
            for combo in crate::discrete::combinatorics::combinations_iter(edges.len(), k) {
                let mut h = Graph::new(g.n, g.directed);
                for (i, &(u, v, _)) in edges.iter().enumerate() {
                    if !combo.contains(&i) && u != v {
                        h.add_edge(u, v, 1.0);
                    }
                }
                if h.bfs(s)[t].is_none() {
                    return k;
                }
            }
        }
        edges.len()
    }

    /// The largest set of internally vertex-disjoint s-t paths, found by
    /// enumerating every simple path and packing them.
    ///
    /// A vertex cut is the wrong reference here: when s and t are adjacent no
    /// set of interior vertices separates them at all, so the minimum-cut
    /// formulation of Menger's theorem simply does not apply to that case.
    /// Counting the paths directly does.
    fn brute_vertex_cut(g: &Graph, s: usize, t: usize) -> usize {
        let paths = all_simple_paths(g, s, t);
        // Each path is described by its interior vertex set and, for a direct
        // s-t hop, by the edge itself; two paths may share neither.
        let mut best = 0usize;
        for k in (1..=paths.len()).rev() {
            if k <= best {
                break;
            }
            let mut feasible = false;
            for combo in crate::discrete::combinatorics::combinations_iter(paths.len(), k) {
                let mut used = vec![false; g.n];
                let mut direct = 0usize;
                let mut ok = true;
                for &i in &combo {
                    let p = &paths[i];
                    if p.len() == 2 {
                        // The direct edge: only as many as there are copies.
                        direct += 1;
                        continue;
                    }
                    for &v in &p[1..p.len() - 1] {
                        if used[v] {
                            ok = false;
                            break;
                        }
                        used[v] = true;
                    }
                    if !ok {
                        break;
                    }
                }
                let copies = g
                    .edges()
                    .iter()
                    .filter(|&&(u, v, _)| (u == s && v == t) || (u == t && v == s))
                    .count();
                if ok && direct <= copies {
                    feasible = true;
                    break;
                }
            }
            if feasible {
                best = k;
                break;
            }
        }
        best
    }

    /// Every simple path from s to t.
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
            for idx in 0..g.adj[cur].len() {
                let w = g.adj[cur][idx].0;
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
        // Distinct vertex sequences only; parallel edges do not make new ones.
        out.sort();
        out.dedup();
        out
    }

    /// The Gomory-Hu tree must encode every pairwise minimum cut as the
    /// lightest edge on its tree path.
    #[test]
    fn gomory_hu_encodes_every_pairwise_cut() {
        let mut rng = Rng::new(0x_6017);
        for n in 2..=7usize {
            for _ in 0..12 {
                let g = random_network(n, 0.55, false, &mut rng);
                if !g.is_connected() {
                    continue;
                }
                let tree = gomory_hu_tree(&g);
                assert_eq!(tree.n, n);
                assert_eq!(tree.edge_count(), n - 1, "not a tree");
                assert!(tree.is_tree());
                for s in 0..n {
                    for t in s + 1..n {
                        let direct = min_cut(&g, s, t).0;
                        let on_tree = lightest_on_tree_path(&tree, s, t);
                        assert!(
                            close(direct, on_tree),
                            "n = {n}, {s}-{t}: cut {direct} vs tree {on_tree}"
                        );
                    }
                }
            }
        }
    }

    /// The lightest edge on the unique tree path between two vertices.
    fn lightest_on_tree_path(t: &Graph, s: usize, e: usize) -> f64 {
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
        let mut best = f64::INFINITY;
        let mut cur = e;
        while let Some(p) = prev[cur] {
            let w = t.adj[cur]
                .iter()
                .filter(|&&(x, _)| x == p)
                .map(|&(_, w)| w)
                .fold(f64::INFINITY, f64::min);
            best = best.min(w);
            cur = p;
            if cur == s {
                break;
            }
        }
        best
    }

    /// A circulation must respect every bound and meet every demand.
    #[test]
    fn circulation_respects_bounds_and_demands() {
        // A feasible instance: 0 sends two units to 2 via 1.
        let g = Graph::from_edges(3, &[(0, 1, 3.0), (1, 2, 3.0)], true);
        let demand = vec![-2.0, 0.0, 2.0];
        let lower = vec![0.0, 0.0];
        let f = circulation_with_demands(&g, &demand, &lower).expect("feasible");
        assert!(close(f[0], 2.0) && close(f[1], 2.0), "got {f:?}");

        // A lower bound that forces three of the four units along the
        // two-hop route, leaving one for the direct edge. Keyed by endpoints,
        // since edges() orders by tail vertex rather than by insertion.
        let g2 = Graph::from_edges(3, &[(0, 1, 5.0), (1, 2, 5.0), (0, 2, 5.0)], true);
        let e2 = g2.edges();
        let lower2: Vec<f64> = e2
            .iter()
            .map(|&(u, v, _)| if (u, v) == (0, 2) { 0.0 } else { 3.0 })
            .collect();
        let f2 = circulation_with_demands(&g2, &[-4.0, 0.0, 4.0], &lower2).expect("feasible");
        for (i, &(u, v, c)) in e2.iter().enumerate() {
            assert!(f2[i] >= lower2[i] - 1e-9, "({u}, {v}) below its lower bound");
            assert!(f2[i] <= c + 1e-9, "({u}, {v}) exceeds its capacity");
        }
        // Conservation, with the demands as the imbalance at each vertex.
        for v in 0..3 {
            let inflow: f64 = e2
                .iter()
                .enumerate()
                .filter(|(_, &(_, b, _))| b == v)
                .map(|(i, _)| f2[i])
                .sum();
            let outflow: f64 = e2
                .iter()
                .enumerate()
                .filter(|(_, &(a, _, _))| a == v)
                .map(|(i, _)| f2[i])
                .sum();
            let want = [-4.0, 0.0, 4.0][v];
            assert!(
                close(inflow - outflow, want),
                "vertex {v}: net {} but demand {want}",
                inflow - outflow
            );
        }

        // Raising the lower bound on the direct edge too makes it infeasible:
        // six units are forced out of a vertex that only supplies four.
        let all_three: Vec<f64> = vec![3.0; 3];
        assert!(circulation_with_demands(&g2, &[-4.0, 0.0, 4.0], &all_three).is_none());

        // Infeasible: the demand exceeds what the network can carry.
        assert!(circulation_with_demands(&g, &[-5.0, 0.0, 5.0], &[0.0, 0.0]).is_none());
        // Infeasible: a lower bound above what conservation allows.
        let g3 = Graph::from_edges(2, &[(0, 1, 2.0)], true);
        assert!(circulation_with_demands(&g3, &[0.0, 0.0], &[1.0]).is_none());
    }

    /// The closure problem and its project-selection specialisation.
    #[test]
    fn closure_and_project_selection_match_brute_force() {
        let mut rng = Rng::new(0x_C105);
        for n in 1..=8usize {
            for _ in 0..15 {
                let mut g = Graph::new(n, true);
                for u in 0..n {
                    for v in 0..n {
                        if u != v && rng.next_f64() < 0.25 {
                            g.add_edge(u, v, 1.0);
                        }
                    }
                }
                let weights: Vec<f64> =
                    (0..n).map(|_| (20.0 * rng.next_f64() - 10.0).round()).collect();
                let (best, members) = closure_problem(&g, &weights);
                // The reported set is genuinely closed.
                for (u, v, _) in g.edges() {
                    if members[u] {
                        assert!(members[v], "closure omits successor {v} of {u}");
                    }
                }
                let got: f64 = (0..n).filter(|&v| members[v]).map(|v| weights[v]).sum();
                assert!(close(got, best), "reported weight {best} vs actual {got}");
                // Optimal, checked over every subset.
                let mut brute = f64::NEG_INFINITY;
                for mask in 0u64..(1u64 << n) {
                    let inside: Vec<bool> = (0..n).map(|v| mask >> v & 1 == 1).collect();
                    if g.edges().iter().any(|&(u, v, _)| inside[u] && !inside[v]) {
                        continue;
                    }
                    let w: f64 = (0..n).filter(|&v| inside[v]).map(|v| weights[v]).sum();
                    brute = brute.max(w);
                }
                assert!(close(best, brute), "n = {n}: {best} vs brute {brute}");
            }
        }

        // Project selection: two projects sharing a machine.
        let profit = project_selection(&[10.0, 10.0], &[15.0], &[vec![0], vec![0]]);
        assert!(close(profit, 5.0), "20 revenue minus one 15 machine, got {profit}");
        // A project that does not pay for its own machine is declined.
        let none = project_selection(&[5.0], &[15.0], &[vec![0]]);
        assert!(close(none, 0.0), "got {none}");
        // No requirements: take every profitable project.
        let free = project_selection(&[3.0, -1.0, 4.0], &[], &[vec![], vec![], vec![]]);
        assert!(close(free, 7.0), "got {free}");
    }

    /// Bipartite matching by flow must have the size Konig's theorem predicts
    /// and must actually be a matching.
    #[test]
    fn bipartite_matching_via_flow_is_valid_and_maximum() {
        let mut rng = Rng::new(0x_B1A4);
        for l in 1..=5usize {
            for r in 1..=5usize {
                for _ in 0..15 {
                    let n = l + r;
                    let mut g = Graph::new(n, false);
                    for a in 0..l {
                        for b in 0..r {
                            if rng.next_f64() < 0.5 {
                                g.add_edge(a, l + b, 1.0);
                            }
                        }
                    }
                    let left: Vec<usize> = (0..l).collect();
                    let m = max_bipartite_matching_via_flow(&g, &left);
                    // Symmetric, and every matched pair is an edge.
                    for v in 0..n {
                        if let Some(w) = m[v] {
                            assert_eq!(m[w], Some(v), "not symmetric at {v}");
                            assert!(
                                g.adj[v].iter().any(|&(x, _)| x == w),
                                "matched a non-edge"
                            );
                        }
                    }
                    let size = m.iter().filter(|x| x.is_some()).count() / 2;
                    // Maximum, by brute force over subsets of edges.
                    let brute = brute_max_matching(&g);
                    assert_eq!(size, brute, "l = {l}, r = {r}");
                }
            }
        }
        // A complete bipartite graph matches the smaller side entirely.
        for m in 1..=4usize {
            for n in 1..=4usize {
                let g = complete_bipartite(m, n);
                let left: Vec<usize> = (0..m).collect();
                let matching = max_bipartite_matching_via_flow(&g, &left);
                let size = matching.iter().filter(|x| x.is_some()).count() / 2;
                assert_eq!(size, m.min(n), "K_{{{m},{n}}}");
            }
        }
    }

    /// The largest matching, by trying every set of pairwise disjoint edges.
    fn brute_max_matching(g: &Graph) -> usize {
        let edges: Vec<(usize, usize)> = g
            .edges()
            .into_iter()
            .filter(|&(u, v, _)| u != v)
            .map(|(u, v, _)| (u, v))
            .collect();
        let mut best = 0usize;
        for k in (1..=edges.len()).rev() {
            if k <= best {
                break;
            }
            for combo in crate::discrete::combinatorics::combinations_iter(edges.len(), k) {
                let mut used = vec![false; g.n];
                let mut ok = true;
                for &i in &combo {
                    let (u, v) = edges[i];
                    if used[u] || used[v] {
                        ok = false;
                        break;
                    }
                    used[u] = true;
                    used[v] = true;
                }
                if ok {
                    best = best.max(k);
                    break;
                }
            }
        }
        best
    }
}
