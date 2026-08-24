//! Colouring, cliques, independent sets, and covers.
//!
//! Almost everything here is NP-hard in general, so the module is split
//! deliberately between two kinds of routine. The heuristics -- greedy
//! colouring, Welsh-Powell, the two-approximation for vertex cover, the
//! greedy dominating set -- run on any graph and come with a stated
//! guarantee, usually a bound relative to a structural parameter rather than
//! to the optimum. The exact routines carry `_small` or `_exact` in their
//! names and are honest about the size they can take: they enumerate, and
//! the cost is exponential.
//!
//! The exception is Vizing's edge colouring, which is exact-ish for free:
//! the theorem says `Delta` or `Delta + 1` colours always suffice, and the
//! Misra-Gries construction reaches `Delta + 1` in polynomial time. Which of
//! the two a given graph needs is itself NP-hard to decide.

use crate::exact::polynomial::PolyQ;
use crate::exact::rational::Rational;
use crate::graph::core::Graph;
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

/// The vertex order a greedy colouring walks.
///
/// Greedy colouring gives every vertex the smallest colour none of its
/// already-coloured neighbours holds. The order is the whole algorithm: some
/// order always produces an optimal colouring, and finding it is the hard
/// part, so these are the standard heuristics for choosing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Vertex index order. No guarantee beyond `Delta + 1`.
    Natural,
    /// Descending degree. The order Welsh-Powell prescribes.
    LargestFirst,
    /// Degeneracy order: repeatedly strip a minimum-degree vertex and colour
    /// in the reverse of the removal order. Uses at most `d + 1` colours for
    /// the degeneracy `d`, which is never worse than `Delta + 1` and is often
    /// much better -- on a planar graph it gives six.
    SmallestLast,
    /// Dynamic: always colour the uncoloured vertex whose neighbours already
    /// show the most distinct colours, breaking ties by degree. Exact on
    /// bipartite graphs and on cycles, unlike any static order.
    Dsatur,
}

/// Neighbour sets, with self-loops and parallel edges collapsed.
///
/// Colouring is a property of the simple graph underneath: a parallel edge
/// constrains nothing a single edge does not, and a self-loop makes proper
/// colouring impossible rather than harder, so it is dropped and documented
/// rather than silently changing every answer in the module.
fn simple_neighbors(g: &Graph) -> Vec<BTreeSet<usize>> {
    let mut adj = vec![BTreeSet::new(); g.n];
    for u in 0..g.n {
        for &(v, _) in &g.adj[u] {
            if u != v {
                adj[u].insert(v);
                adj[v].insert(u);
            }
        }
    }
    adj
}

/// The smallest non-negative integer not in `used`.
fn mex(used: &BTreeSet<usize>) -> usize {
    (0..).find(|c| !used.contains(c)).expect("the naturals are unbounded")
}

/// A degeneracy order: the reverse of repeatedly removing a vertex of
/// minimum degree in what remains.
fn degeneracy_order(adj: &[BTreeSet<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut deg: Vec<usize> = adj.iter().map(BTreeSet::len).collect();
    let mut gone = vec![false; n];
    let mut removal = Vec::with_capacity(n);
    for _ in 0..n {
        let v = (0..n)
            .filter(|&v| !gone[v])
            .min_by_key(|&v| (deg[v], v))
            .expect("one vertex remains");
        gone[v] = true;
        removal.push(v);
        for &w in &adj[v] {
            if !gone[w] {
                deg[w] -= 1;
            }
        }
    }
    removal.reverse();
    removal
}

/// Greedy colouring in the given vertex order.
///
/// Returns one colour per vertex, numbered from zero. Every order yields a
/// proper colouring; the count of colours is what varies, and
/// [`Order::SmallestLast`] and [`Order::Dsatur`] carry the guarantees worth
/// having. Self-loops are ignored, since no colouring can respect one.
#[must_use]
pub fn greedy_coloring(g: &Graph, order: Order) -> Vec<usize> {
    let n = g.n;
    let adj = simple_neighbors(g);
    let mut color = vec![usize::MAX; n];
    if order == Order::Dsatur {
        // Saturation is the count of distinct colours already on a vertex's
        // neighbours, and it changes after every assignment, so the order
        // cannot be precomputed.
        let mut seen: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for _ in 0..n {
            let v = (0..n)
                .filter(|&v| color[v] == usize::MAX)
                .max_by_key(|&v| (seen[v].len(), adj[v].len(), usize::MAX - v))
                .expect("one vertex is uncoloured");
            let c = mex(&seen[v]);
            color[v] = c;
            for &w in &adj[v] {
                seen[w].insert(c);
            }
        }
        return color;
    }
    let sequence: Vec<usize> = match order {
        Order::Natural => (0..n).collect(),
        Order::LargestFirst => {
            let mut s: Vec<usize> = (0..n).collect();
            s.sort_by_key(|&v| (std::cmp::Reverse(adj[v].len()), v));
            s
        }
        Order::SmallestLast => degeneracy_order(&adj),
        Order::Dsatur => unreachable!("handled above"),
    };
    for v in sequence {
        let used: BTreeSet<usize> =
            adj[v].iter().map(|&w| color[w]).filter(|&c| c != usize::MAX).collect();
        color[v] = mex(&used);
    }
    color
}

/// The number of distinct colours a colouring uses.
#[must_use]
pub fn color_count(coloring: &[usize]) -> usize {
    coloring.iter().collect::<BTreeSet<_>>().len()
}

/// Whether a colouring gives no edge two ends of the same colour.
///
/// A self-loop always fails, which is the correct answer: a graph with one
/// has no proper colouring at all.
#[must_use]
pub fn is_proper_coloring(g: &Graph, coloring: &[usize]) -> bool {
    coloring.len() == g.n && g.edges().iter().all(|&(u, v, _)| coloring[u] != coloring[v])
}

/// Welsh-Powell colouring: sort by descending degree, then fill one colour
/// class at a time by sweeping the list.
///
/// This is the same colouring [`greedy_coloring`] produces under
/// [`Order::LargestFirst`], and for the same reason: a vertex takes colour
/// `c` in the sweep exactly when every earlier class held a neighbour of it,
/// which is the greedy rule stated the other way round. The procedure is
/// kept in its own form because the bound it is quoted with --
/// `max_i min(d_i + 1, i)` over the sorted degrees -- is a statement about
/// the sweep.
#[must_use]
pub fn welsh_powell(g: &Graph) -> Vec<usize> {
    let n = g.n;
    let adj = simple_neighbors(g);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&v| (std::cmp::Reverse(adj[v].len()), v));
    let mut color = vec![usize::MAX; n];
    let mut c = 0;
    let mut left = n;
    while left > 0 {
        let mut class: Vec<usize> = Vec::new();
        for &v in &order {
            if color[v] == usize::MAX && class.iter().all(|&w| !adj[v].contains(&w)) {
                color[v] = c;
                class.push(v);
                left -= 1;
            }
        }
        c += 1;
    }
    color
}

/// The Welsh-Powell bound on the number of colours: `max_i min(d_i + 1, i)`
/// over the degrees sorted descending, indexed from one.
#[must_use]
pub fn welsh_powell_bound(g: &Graph) -> usize {
    let adj = simple_neighbors(g);
    let mut degrees: Vec<usize> = adj.iter().map(BTreeSet::len).collect();
    degrees.sort_unstable_by(|a, b| b.cmp(a));
    degrees
        .iter()
        .enumerate()
        .map(|(i, &d)| (d + 1).min(i + 1))
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Exact colouring
// ---------------------------------------------------------------------------

/// Backtracking search for a proper `k`-colouring, with the deadline checked
/// as it goes. `None` means either infeasible or out of time; the caller
/// gets no way to tell those apart, which is what a time-limited search is.
fn color_search(
    adj: &[BTreeSet<usize>],
    k: usize,
    deadline: Option<Instant>,
) -> Option<Vec<usize>> {
    let n = adj.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if k == 0 {
        return None;
    }
    // Domains as bitmasks over the k colours, so propagation is a mask AND.
    let full: u64 = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
    let mut domain = vec![full; n];
    let mut color = vec![usize::MAX; n];
    // Symmetry breaking: colours are interchangeable, so the first vertex may
    // as well take colour zero, and no vertex may open a colour more than one
    // above the highest already in use. Without this the search re-derives
    // every relabelling of the same colouring, k! of them.
    let mut steps: u64 = 0;
    fn go(
        adj: &[BTreeSet<usize>],
        k: usize,
        domain: &mut Vec<u64>,
        color: &mut Vec<usize>,
        left: usize,
        highest: usize,
        deadline: Option<Instant>,
        steps: &mut u64,
    ) -> bool {
        if left == 0 {
            return true;
        }
        *steps += 1;
        // Checking the clock costs a syscall, so do it once every so often.
        if (*steps).is_multiple_of(4096) {
            if let Some(t) = deadline {
                if Instant::now() >= t {
                    return false;
                }
            }
        }
        // Most-constrained variable first: the fewest remaining colours.
        let v = (0..adj.len())
            .filter(|&v| color[v] == usize::MAX)
            .min_by_key(|&v| (domain[v].count_ones(), std::cmp::Reverse(adj[v].len()), v))
            .expect("some vertex is uncoloured");
        if domain[v] == 0 {
            return false;
        }
        let cap = (highest + 2).min(k);
        for c in 0..cap {
            if domain[v] & (1 << c) == 0 {
                continue;
            }
            color[v] = c;
            let mut undone: Vec<usize> = Vec::new();
            let mut dead = false;
            for &w in &adj[v] {
                if color[w] == usize::MAX && domain[w] & (1 << c) != 0 {
                    domain[w] &= !(1u64 << c);
                    undone.push(w);
                    if domain[w] == 0 {
                        dead = true;
                    }
                }
            }
            if !dead
                && go(
                    adj,
                    k,
                    domain,
                    color,
                    left - 1,
                    highest.max(c),
                    deadline,
                    steps,
                )
            {
                return true;
            }
            for w in undone {
                domain[w] |= 1u64 << c;
            }
            color[v] = usize::MAX;
        }
        false
    }
    if go(adj, k, &mut domain, &mut color, n, 0, deadline, &mut steps) {
        Some(color)
    } else {
        None
    }
}

/// A proper `k`-colouring found by constraint propagation, or `None` if the
/// search proves there is none or runs out of time.
///
/// The search is the SAT solver's shape rather than its machinery: colour
/// domains as bitmasks, unit propagation by masking a colour out of every
/// neighbour, most-constrained-variable branching, and the symmetry break
/// that stops the search from rediscovering every relabelling of a colouring
/// it has already rejected.
///
/// `time_limit` is a wall-clock budget. A zero limit returns `None` without
/// searching. Because timeout and infeasibility both return `None`, use
/// [`chromatic_number_exact_small`] when the distinction matters.
#[must_use]
pub fn is_k_colorable_sat_style(g: &Graph, k: usize, time_limit: Duration) -> Option<Vec<usize>> {
    if g.edges().iter().any(|&(u, v, _)| u == v) {
        return None;
    }
    if time_limit.is_zero() {
        return None;
    }
    let adj = simple_neighbors(g);
    color_search(&adj, k, Instant::now().checked_add(time_limit))
}

/// The chromatic number, by exhaustive search. Intended for `n <= 20`.
///
/// Bracketed first: a greedy clique gives a lower bound, since a clique of
/// size `q` needs `q` colours, and DSATUR gives an upper bound. Then each `k`
/// in between is decided exactly. On a graph the bracket already pins -- and
/// it often does -- no search runs at all.
///
/// # Panics
/// Panics on a self-loop, which admits no proper colouring.
#[must_use]
pub fn chromatic_number_exact_small(g: &Graph) -> usize {
    assert!(
        !g.edges().iter().any(|&(u, v, _)| u == v),
        "a self-loop admits no proper colouring"
    );
    let n = g.n;
    if n == 0 {
        return 0;
    }
    let adj = simple_neighbors(g);
    if adj.iter().all(BTreeSet::is_empty) {
        return 1;
    }
    let lower = max_clique_bron_kerbosch(g).len().max(2);
    let upper = color_count(&greedy_coloring(g, Order::Dsatur));
    for k in lower..upper {
        if color_search(&adj, k, None).is_some() {
            return k;
        }
    }
    upper
}

/// The chromatic polynomial, exactly, by deletion-contraction. For `n <= 12`.
///
/// `P(G, x)` counts the proper colourings of `G` with `x` colours, and the
/// recursion is `P(G) = P(G - e) - P(G / e)`: colourings of `G - e` either
/// give `e`'s ends different colours, which is a colouring of `G`, or the
/// same colour, which is a colouring of the contraction. Both branches
/// shrink the graph -- deletion loses an edge, contraction loses a vertex --
/// so the recursion terminates on the edgeless graph, whose polynomial is
/// `x^n`. Memoised on the canonical edge set, which is what makes it
/// tractable at all: the two branches meet again constantly.
///
/// # Panics
/// Panics on a self-loop. Contraction can create one only from a parallel
/// edge, which is collapsed first.
#[must_use]
pub fn chromatic_polynomial_small(g: &Graph) -> PolyQ {
    assert!(
        !g.edges().iter().any(|&(u, v, _)| u == v),
        "a self-loop admits no proper colouring"
    );
    let adj = simple_neighbors(g);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for u in 0..g.n {
        for &v in &adj[u] {
            if u < v {
                edges.push((u, v));
            }
        }
    }
    let mut memo: BTreeMap<(usize, Vec<(usize, usize)>), PolyQ> = BTreeMap::new();
    chromatic_rec(g.n, &edges, &mut memo)
}

/// `x^n`, the polynomial of the edgeless graph on `n` vertices.
fn x_pow(n: usize) -> PolyQ {
    let mut c = vec![Rational::zero(); n + 1];
    c[n] = Rational::from_i64(1, 1);
    PolyQ::new(c)
}

fn chromatic_rec(
    n: usize,
    edges: &[(usize, usize)],
    memo: &mut BTreeMap<(usize, Vec<(usize, usize)>), PolyQ>,
) -> PolyQ {
    if edges.is_empty() {
        return x_pow(n);
    }
    // A complete graph closes the recursion in one step, with the falling
    // factorial x(x-1)...(x-n+1): the vertices must all differ, so each takes
    // one fewer choice than the last.
    if edges.len() == n * (n - 1) / 2 {
        let mut p = PolyQ::from_i64s(&[1]);
        for i in 0..n {
            p = p.mul(&PolyQ::from_i64s(&[-(i as i64), 1]));
        }
        return p;
    }
    let key = (n, edges.to_vec());
    if let Some(p) = memo.get(&key) {
        return p.clone();
    }
    let (a, b) = edges[0];
    let deleted: Vec<(usize, usize)> = edges[1..].to_vec();
    // Contract b into a and renumber the survivors down, collapsing the
    // parallel edges the merge creates. A self-loop cannot survive: the only
    // candidate is (a, b) itself, and that is the edge being contracted.
    let map = |v: usize| -> usize {
        let t = if v == b { a } else { v };
        if t > b {
            t - 1
        } else {
            t
        }
    };
    let mut contracted: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(u, v) in &deleted {
        let (p, q) = (map(u), map(v));
        if p != q {
            contracted.insert((p.min(q), p.max(q)));
        }
    }
    let contracted: Vec<(usize, usize)> = contracted.into_iter().collect();
    let result = chromatic_rec(n, &deleted, memo).sub(&chromatic_rec(n - 1, &contracted, memo));
    memo.insert(key, result.clone());
    result
}

// ---------------------------------------------------------------------------
// Edge colouring
// ---------------------------------------------------------------------------

/// Vizing edge colouring by the Misra-Gries construction: one colour per
/// edge, no two edges sharing a vertex alike, in at most `Delta + 1` colours.
///
/// Vizing's theorem says every simple graph needs `Delta` or `Delta + 1`, and
/// this reaches the upper end constructively. Each edge is coloured by
/// building a *fan* around one endpoint -- a run of neighbours where each
/// one's edge colour is free at the previous, so the whole run can shift
/// down by one -- then either rotating the fan to slide a free colour into
/// place, or first inverting a two-colour alternating path to make one free.
/// The alternating path is the part that makes the bound work: it repairs
/// the one obstruction rotation alone cannot, and it does so without
/// disturbing any other vertex, since every interior vertex of the path
/// simply exchanges its `c` for its `d`.
///
/// Returns one colour per entry of `g.edges()`, in that order.
///
/// # Panics
/// Panics if the graph is directed, or is not simple. A self-loop cannot be
/// coloured at all, and a parallel edge takes the bound outside Vizing's
/// theorem into Shannon's `Delta + mu`.
#[must_use]
pub fn edge_coloring_vizing(g: &Graph) -> Vec<usize> {
    assert!(!g.directed, "edge colouring here is for undirected graphs");
    let edges = g.edges();
    assert!(!edges.iter().any(|&(u, v, _)| u == v), "a self-loop cannot be edge-coloured");
    let mut seen = BTreeSet::new();
    for &(u, v, _) in &edges {
        assert!(seen.insert((u.min(v), u.max(v))), "Vizing's bound is for simple graphs");
    }
    let n = g.n;
    let m = edges.len();
    let mut count = vec![0usize; n];
    for &(u, v, _) in &edges {
        count[u] += 1;
        count[v] += 1;
    }
    let k = count.iter().copied().max().unwrap_or(0) + 1;

    let mut color: Vec<Option<usize>> = vec![None; m];
    // `inc[v][c]` is the edge at `v` carrying colour `c`. The incidence table
    // is what makes "is colour c free at v" a lookup rather than a scan.
    let mut inc: Vec<Vec<Option<usize>>> = vec![vec![None; k]; n];
    let far = |e: usize, v: usize| -> usize {
        let (a, b, _) = edges[e];
        if a == v {
            b
        } else {
            a
        }
    };

    for e0 in 0..m {
        let (u, x0, _) = edges[e0];
        // The fan, as edges at `u` together with their far ends. Carrying the
        // edge indices rather than only the vertices keeps the rotation exact.
        let mut fan_e = vec![e0];
        let mut fan_v = vec![x0];
        loop {
            let last = *fan_v.last().expect("the fan starts non-empty");
            let mut grew = false;
            for c in 0..k {
                if inc[last][c].is_some() {
                    continue;
                }
                // `c` is free at the fan's end; does `u` wear it, on an edge
                // to a vertex the fan has not already taken?
                if let Some(f) = inc[u][c] {
                    let y = far(f, u);
                    if !fan_v.contains(&y) {
                        fan_e.push(f);
                        fan_v.push(y);
                        grew = true;
                        break;
                    }
                }
            }
            if !grew {
                break;
            }
        }

        let free_at = |inc: &[Vec<Option<usize>>], v: usize| -> usize {
            (0..k).find(|&c| inc[v][c].is_none()).expect("Delta + 1 colours leave one free")
        };
        let c = free_at(&inc, u);
        let d = free_at(&inc, *fan_v.last().expect("non-empty"));

        // Invert the maximal c/d-alternating path leaving `u`. `c` is free at
        // `u`, so the path starts on a `d`-edge, and inverting it frees `d`
        // at `u`.
        if c != d {
            let mut at = u;
            let mut want = d;
            let mut path: Vec<usize> = Vec::new();
            while let Some(f) = inc[at][want] {
                path.push(f);
                at = far(f, at);
                want = if want == c { d } else { c };
            }
            // Clear the whole path from the incidence table before writing
            // any of it back. Two consecutive path edges meet at a vertex and
            // exchange colours there, so interleaving the clear and the write
            // has the second edge erase the entry the first has just made.
            for &f in &path {
                let (a, b, _) = edges[f];
                let old = color[f].expect("a path edge is coloured");
                inc[a][old] = None;
                inc[b][old] = None;
            }
            for &f in &path {
                let (a, b, _) = edges[f];
                let old = color[f].expect("a path edge is coloured");
                let new = if old == c { d } else { c };
                color[f] = Some(new);
                inc[a][new] = Some(f);
                inc[b][new] = Some(f);
            }
        }

        // The inversion can have broken the fan further along: it recolours
        // one edge at `u`, and it can occupy at a fan vertex the very colour
        // the fan property needed free there. So re-establish how far the
        // fan still reaches, and take the last vertex within that prefix
        // where `d` is free. Misra and Gries show one exists.
        let mut reach = 0usize;
        while reach + 1 < fan_v.len() {
            let Some(next) = color[fan_e[reach + 1]] else { break };
            if inc[fan_v[reach]][next].is_some() {
                break;
            }
            reach += 1;
        }
        let w = (0..=reach)
            .rev()
            .find(|&i| inc[fan_v[i]][d].is_none())
            .expect("Misra-Gries guarantees a fan vertex where d is free");

        // Rotate the prefix: each edge takes the next one's colour, which the
        // fan invariant says is free at its own far end, and the last takes
        // `d`. Read the colours before clearing any, or the shift reads what
        // it has already written.
        let moving: Vec<usize> = (1..=w).map(|i| color[fan_e[i]].expect("coloured")).collect();
        for i in 0..=w {
            let f = fan_e[i];
            if let Some(old) = color[f] {
                let (a, b, _) = edges[f];
                inc[a][old] = None;
                inc[b][old] = None;
                color[f] = None;
            }
        }
        for i in 0..=w {
            let f = fan_e[i];
            let new = if i < w { moving[i] } else { d };
            let (a, b, _) = edges[f];
            debug_assert!(inc[a][new].is_none() && inc[b][new].is_none(), "rotation collided");
            color[f] = Some(new);
            inc[a][new] = Some(f);
            inc[b][new] = Some(f);
        }
    }
    color.into_iter().map(|c| c.expect("every edge is coloured")).collect()
}

/// Whether an edge colouring gives no two edges sharing a vertex the same
/// colour.
#[must_use]
pub fn is_proper_edge_coloring(g: &Graph, coloring: &[usize]) -> bool {
    let edges = g.edges();
    if coloring.len() != edges.len() {
        return false;
    }
    let mut at: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); g.n];
    for (i, &(u, v, _)) in edges.iter().enumerate() {
        if u == v || !at[u].insert(coloring[i]) || !at[v].insert(coloring[i]) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Interval and map colouring
// ---------------------------------------------------------------------------

/// Optimal colouring of an interval graph, given the intervals themselves.
///
/// Two intervals conflict when they overlap, and the greedy sweep in order of
/// left endpoint is optimal here -- unlike on a general graph -- because at
/// the moment an interval opens, every interval it will ever conflict with
/// that came earlier is still open. So the colours in use are exactly the
/// current overlap, and the total is the maximum overlap, which is a lower
/// bound for any colouring. Half-open intervals: touching at an endpoint is
/// not an overlap, and an interval whose ends coincide is empty, meets
/// nothing, and shares the first colour.
///
/// Returns one colour per interval, in the input order.
///
/// # Panics
/// Panics if an interval has its end before its start, or is not finite.
#[must_use]
pub fn interval_graph_coloring(intervals: &[(f64, f64)]) -> Vec<usize> {
    for &(a, b) in intervals {
        assert!(a.is_finite() && b.is_finite(), "intervals must be finite");
        assert!(a <= b, "an interval must not end before it starts");
    }
    let mut order: Vec<usize> = (0..intervals.len()).collect();
    order.sort_by(|&i, &j| {
        intervals[i]
            .0
            .total_cmp(&intervals[j].0)
            .then_with(|| intervals[i].1.total_cmp(&intervals[j].1))
            .then_with(|| i.cmp(&j))
    });
    let mut color = vec![usize::MAX; intervals.len()];
    // (end, colour) for each interval still open, kept sorted by end.
    let mut open: Vec<(f64, usize)> = Vec::new();
    let mut free: BTreeSet<usize> = BTreeSet::new();
    let mut next = 0usize;
    for &i in &order {
        let (start, end) = intervals[i];
        if start == end {
            // Half-open, so this interval is empty: it meets nothing, and
            // giving it a colour of its own would push the total past the
            // maximum overlap, which is the one thing the sweep guarantees.
            color[i] = 0;
            continue;
        }
        open.retain(|&(e, c)| {
            if e <= start {
                free.insert(c);
                false
            } else {
                true
            }
        });
        let c = if let Some(&c) = free.iter().next() {
            free.remove(&c);
            c
        } else {
            next += 1;
            next - 1
        };
        color[i] = c;
        open.push((end, c));
    }
    color
}

/// A proper `k`-colouring of a graph given by adjacency lists, or `None`.
///
/// The classic map-colouring formulation: regions and the regions they
/// border. Straight chronological backtracking with forward checking, which
/// is what the four-colour problem was posed as long before it was a theorem.
///
/// # Panics
/// Panics if an adjacency list names a region outside the range.
#[must_use]
pub fn map_coloring_backtrack(adjacency: &[Vec<usize>], k: usize) -> Option<Vec<usize>> {
    let n = adjacency.len();
    let mut adj = vec![BTreeSet::new(); n];
    for (u, list) in adjacency.iter().enumerate() {
        for &v in list {
            assert!(v < n, "region {v} is outside 0..{n}");
            if u != v {
                adj[u].insert(v);
                adj[v].insert(u);
            }
        }
    }
    if adjacency.iter().enumerate().any(|(u, l)| l.contains(&u)) {
        return None;
    }
    color_search(&adj, k, None)
}

// ---------------------------------------------------------------------------
// Cliques, independent sets, covers
// ---------------------------------------------------------------------------

/// Neighbour bitmasks, for the enumeration routines. `None` above 64
/// vertices, which is far past where they are usable anyway.
fn masks(g: &Graph) -> Option<Vec<u64>> {
    if g.n > 64 {
        return None;
    }
    let mut m = vec![0u64; g.n];
    for u in 0..g.n {
        for &(v, _) in &g.adj[u] {
            if u != v {
                m[u] |= 1 << v;
                m[v] |= 1 << u;
            }
        }
    }
    Some(m)
}

fn bits(mut x: u64) -> Vec<usize> {
    let mut out = Vec::with_capacity(x.count_ones() as usize);
    while x != 0 {
        let i = x.trailing_zeros() as usize;
        out.push(i);
        x &= x - 1;
    }
    out
}

/// Every maximal clique, by Bron-Kerbosch with pivoting.
///
/// A clique is maximal when no vertex can be added; the algorithm grows one
/// while maintaining the candidates that could still join (`p`) and those
/// already ruled out (`x`), and reports when both are empty. The pivot is the
/// speedup: choosing a vertex `q` from `p | x` with the most neighbours in
/// `p`, and branching only on `p` minus `q`'s neighbourhood, skips the
/// branches that could only ever rediscover a clique through `q`.
///
/// # Panics
/// Panics above 64 vertices. The output can be exponential in the input --
/// a graph on `3j` vertices can have `3^j` maximal cliques -- so this is for
/// small graphs by construction.
#[must_use]
pub fn all_maximal_cliques(g: &Graph) -> Vec<Vec<usize>> {
    let adj = masks(g).expect("clique enumeration is for at most 64 vertices");
    let n = g.n;
    let mut out = Vec::new();
    let all: u64 = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    bron_kerbosch(0, all, 0, &adj, &mut out);
    out.sort();
    out
}

fn bron_kerbosch(r: u64, p: u64, x: u64, adj: &[u64], out: &mut Vec<Vec<usize>>) {
    if p == 0 && x == 0 {
        out.push(bits(r));
        return;
    }
    let pivot = bits(p | x)
        .into_iter()
        .max_by_key(|&q| (adj[q] & p).count_ones())
        .expect("p | x is non-empty");
    let mut p = p;
    let mut x = x;
    for v in bits(p & !adj[pivot]) {
        bron_kerbosch(r | (1 << v), p & adj[v], x & adj[v], adj, out);
        p &= !(1u64 << v);
        x |= 1 << v;
    }
}

/// A maximum clique: the largest set of mutually adjacent vertices.
///
/// Every maximum clique is maximal, so enumerating the maximal ones and
/// taking the largest is exact. Ties go to the lexicographically first.
///
/// # Panics
/// Panics above 64 vertices.
#[must_use]
pub fn max_clique_bron_kerbosch(g: &Graph) -> Vec<usize> {
    all_maximal_cliques(g)
        .into_iter()
        .max_by_key(|c| c.len())
        .unwrap_or_default()
}

/// A maximal independent set, greedily: repeatedly take a vertex of minimum
/// remaining degree and discard its neighbours.
///
/// Minimum degree first is the right greed here: taking the vertex that
/// eliminates the fewest others leaves the most room for the rest. The
/// result is guaranteed maximal -- nothing can be added -- and at least
/// `sum_v 1/(d_v + 1)` in size by the Caro-Wei bound, but not maximum.
#[must_use]
pub fn independent_set_greedy(g: &Graph) -> Vec<usize> {
    let adj = simple_neighbors(g);
    let n = g.n;
    let mut alive = vec![true; n];
    let mut out = Vec::new();
    loop {
        let pick = (0..n)
            .filter(|&v| alive[v])
            .min_by_key(|&v| (adj[v].iter().filter(|&&w| alive[w]).count(), v));
        let Some(v) = pick else { break };
        out.push(v);
        alive[v] = false;
        for &w in &adj[v] {
            alive[w] = false;
        }
    }
    out.sort_unstable();
    out
}

/// A maximum independent set, exactly, via the complement.
///
/// An independent set in `G` is a clique in the complement of `G` and the
/// other way round, so this is the clique enumeration with the edges flipped.
///
/// # Panics
/// Panics above 64 vertices.
#[must_use]
pub fn max_independent_set_small(g: &Graph) -> Vec<usize> {
    max_clique_bron_kerbosch(&g.complement())
}

/// A vertex cover within a factor of two of the smallest, by taking both ends
/// of a maximal matching.
///
/// The matching's edges are disjoint, so any cover must contain at least one
/// end of each, giving `opt >= |M|`; taking both ends gives `2|M| <= 2 opt`.
/// The bound comes free with the construction and holds on every graph, which
/// is more than the best known algorithm can say about doing better.
#[must_use]
pub fn vertex_cover_2approx(g: &Graph) -> Vec<usize> {
    let mut covered = vec![false; g.n];
    let mut out = BTreeSet::new();
    for (u, v, _) in g.edges() {
        if u == v || covered[u] || covered[v] {
            continue;
        }
        covered[u] = true;
        covered[v] = true;
        out.insert(u);
        out.insert(v);
    }
    out.into_iter().collect()
}

/// A minimum vertex cover, exactly, as the complement of a maximum
/// independent set.
///
/// Gallai's identity: a set covers every edge exactly when its complement
/// spans none, so the two problems are the same problem read twice, and
/// `tau + alpha = n`.
///
/// # Panics
/// Panics above 64 vertices, or on a self-loop, whose vertex every cover must
/// contain and which the complement identity does not account for.
#[must_use]
pub fn vertex_cover_exact_small(g: &Graph) -> Vec<usize> {
    assert!(
        !g.edges().iter().any(|&(u, v, _)| u == v),
        "the independent set identity does not hold with self-loops"
    );
    let keep: BTreeSet<usize> = max_independent_set_small(g).into_iter().collect();
    (0..g.n).filter(|v| !keep.contains(v)).collect()
}

/// A dominating set, greedily: every vertex is in it or next to it.
///
/// Set cover in disguise, with each vertex offering its closed neighbourhood,
/// so the greedy choice of whichever vertex newly dominates the most inherits
/// set cover's `ln(n) + 1` guarantee -- and its hardness, since matching that
/// factor in polynomial time would collapse the same complexity assumption.
#[must_use]
pub fn dominating_set_greedy(g: &Graph) -> Vec<usize> {
    let adj = simple_neighbors(g);
    let n = g.n;
    let mut done = vec![false; n];
    let mut left = n;
    let mut out = Vec::new();
    while left > 0 {
        let gain = |v: usize| -> usize {
            usize::from(!done[v]) + adj[v].iter().filter(|&&w| !done[w]).count()
        };
        let v = (0..n)
            .max_by_key(|&v| (gain(v), std::cmp::Reverse(v)))
            .expect("n > 0 while vertices remain");
        out.push(v);
        for w in std::iter::once(v).chain(adj[v].iter().copied()) {
            if !done[w] {
                done[w] = true;
                left -= 1;
            }
        }
    }
    out.sort_unstable();
    out
}

/// A feedback arc set: arcs whose removal leaves a directed acyclic graph.
///
/// By the Eades-Lin-Smyth ordering. It builds a linear order by repeatedly
/// taking sinks from the right, sources from the left, and otherwise the
/// vertex whose out-degree most exceeds its in-degree; the arcs pointing
/// backwards in that order are the answer. Removing them must leave a DAG,
/// since a linear order that every remaining arc respects is a topological
/// order. The count is within `m/2 - n/6` of the total, which is the bound
/// the heuristic is quoted for.
///
/// Self-loops are always returned: no ordering can place a vertex before
/// itself.
///
/// # Panics
/// Panics if the graph is undirected.
#[must_use]
pub fn feedback_arc_set_greedy(g: &Graph) -> Vec<(usize, usize)> {
    assert!(g.directed, "a feedback arc set is for directed graphs");
    let n = g.n;
    let mut alive = vec![true; n];
    let mut out_deg = vec![0usize; n];
    let mut in_deg = vec![0usize; n];
    let mut arcs: Vec<(usize, usize)> = Vec::new();
    for u in 0..n {
        for &(v, _) in &g.adj[u] {
            arcs.push((u, v));
            if u != v {
                out_deg[u] += 1;
                in_deg[v] += 1;
            }
        }
    }
    let mut left: Vec<usize> = Vec::new();
    let mut right: Vec<usize> = Vec::new();
    let mut remaining = n;
    // Removing a vertex from the working graph means discounting its arcs
    // from the degrees of whatever is still alive.
    let drop = |v: usize,
                    alive: &mut Vec<bool>,
                    out_deg: &mut Vec<usize>,
                    in_deg: &mut Vec<usize>| {
        alive[v] = false;
        for &(a, b) in &arcs {
            if a == b {
                continue;
            }
            if a == v && alive[b] {
                in_deg[b] -= 1;
            }
            if b == v && alive[a] {
                out_deg[a] -= 1;
            }
        }
    };
    while remaining > 0 {
        loop {
            let sink = (0..n).find(|&v| alive[v] && out_deg[v] == 0);
            let Some(v) = sink else { break };
            drop(v, &mut alive, &mut out_deg, &mut in_deg);
            right.push(v);
            remaining -= 1;
        }
        loop {
            let source = (0..n).find(|&v| alive[v] && in_deg[v] == 0);
            let Some(v) = source else { break };
            drop(v, &mut alive, &mut out_deg, &mut in_deg);
            left.push(v);
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        let v = (0..n)
            .filter(|&v| alive[v])
            .max_by_key(|&v| (out_deg[v] as i64 - in_deg[v] as i64, std::cmp::Reverse(v)))
            .expect("a vertex remains");
        drop(v, &mut alive, &mut out_deg, &mut in_deg);
        left.push(v);
        remaining -= 1;
    }
    right.reverse();
    left.extend(right);
    let mut rank = vec![0usize; n];
    for (i, &v) in left.iter().enumerate() {
        rank[v] = i;
    }
    arcs.into_iter().filter(|&(u, v)| rank[u] >= rank[v]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    /// A value in `0..n` from the high bits: `% n` reads the low bits of the
    /// linear congruential generator, where bit `b` has period `2^(b+1)`.
    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
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

    fn max_degree(g: &Graph) -> usize {
        (0..g.n).map(|v| simple_neighbors(g)[v].len()).max().unwrap_or(0)
    }

    /// The degeneracy: the largest `k` for which every subgraph has a vertex
    /// of degree at most `k`, computed the way the definition reads.
    fn degeneracy(g: &Graph) -> usize {
        let adj = simple_neighbors(g);
        let n = g.n;
        let mut alive = vec![true; n];
        let mut best = 0;
        for _ in 0..n {
            let v = (0..n)
                .filter(|&v| alive[v])
                .min_by_key(|&v| adj[v].iter().filter(|&&w| alive[w]).count())
                .expect("a vertex remains");
            best = best.max(adj[v].iter().filter(|&&w| alive[w]).count());
            alive[v] = false;
        }
        best
    }

    /// Proper colourings with `k` colours, counted by exhaustion.
    fn count_colorings(g: &Graph, k: usize) -> i64 {
        let n = g.n;
        if k == 0 {
            return i64::from(n == 0);
        }
        let edges: Vec<(usize, usize)> =
            g.edges().iter().map(|&(u, v, _)| (u, v)).collect();
        let mut assign = vec![0usize; n];
        let mut total = 0i64;
        let mut i = 0usize;
        // Odometer over k^n assignments.
        loop {
            if i == n {
                if edges.iter().all(|&(u, v)| assign[u] != assign[v]) {
                    total += 1;
                }
                i = n;
                // advance
                let mut j = n;
                loop {
                    if j == 0 {
                        return total;
                    }
                    j -= 1;
                    assign[j] += 1;
                    if assign[j] < k {
                        break;
                    }
                    assign[j] = 0;
                }
                continue;
            }
            i += 1;
        }
    }

    /// Every order produces a proper colouring, and each carries the bound it
    /// is chosen for: `Delta + 1` in general, degeneracy plus one for the
    /// smallest-last order, and exactness on graphs where DSATUR is known to
    /// be exact.
    #[test]
    fn greedy_orders_are_proper_and_meet_their_bounds() {
        let mut rng = Rng::new(0x_C010);
        let orders = [Order::Natural, Order::LargestFirst, Order::SmallestLast, Order::Dsatur];
        for _ in 0..300 {
            let n = 1 + pick(&mut rng, 12);
            let g = random_graph(n, 0.15 + 0.6 * rng.next_f64(), &mut rng);
            let d = max_degree(&g);
            let deg = degeneracy(&g);
            for order in orders {
                let c = greedy_coloring(&g, order);
                assert!(is_proper_coloring(&g, &c), "{order:?} is not proper");
                assert!(color_count(&c) <= d + 1, "{order:?} exceeded Delta + 1");
                if order == Order::SmallestLast {
                    assert!(
                        color_count(&c) <= deg + 1,
                        "the degeneracy order used more than degeneracy + 1 colours"
                    );
                }
                // No heuristic can beat the optimum.
                assert!(color_count(&c) >= chromatic_number_exact_small(&g));
            }
        }

        // DSATUR is exact on bipartite graphs and on cycles; that is the
        // property it is chosen for and no static order has it.
        for g in [complete_bipartite(3, 4), path_graph(7), cycle_graph(8), cycle_graph(9)] {
            let c = greedy_coloring(&g, Order::Dsatur);
            assert!(is_proper_coloring(&g, &c));
            assert_eq!(
                color_count(&c),
                chromatic_number_exact_small(&g),
                "DSATUR was not exact"
            );
        }
        // The natural order is not: the crown graph on 2k vertices, indexed so
        // that i and i + k are the non-edge, forces k colours out of a graph
        // that needs two.
        let mut crown = Graph::new(8, false);
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    crown.add_edge(i, 4 + j, 1.0);
                }
            }
        }
        let mut relabelled = Graph::new(8, false);
        for (u, v, _) in crown.edges() {
            // Interleave the sides so the natural order alternates between them.
            let f = |x: usize| if x < 4 { 2 * x } else { 2 * (x - 4) + 1 };
            relabelled.add_edge(f(u), f(v), 1.0);
        }
        assert_eq!(chromatic_number_exact_small(&relabelled), 2);
        assert_eq!(color_count(&greedy_coloring(&relabelled, Order::Natural)), 4);
        assert_eq!(color_count(&greedy_coloring(&relabelled, Order::Dsatur)), 2);
    }

    /// Welsh-Powell's colour-class sweep and largest-first greedy are the same
    /// procedure written twice: a vertex enters class `c` exactly when every
    /// earlier class already holds a neighbour of it, which is the greedy rule.
    #[test]
    fn welsh_powell_is_largest_first_greedy() {
        let mut rng = Rng::new(0x_57EE);
        for _ in 0..300 {
            let n = 1 + pick(&mut rng, 14);
            let g = random_graph(n, 0.1 + 0.7 * rng.next_f64(), &mut rng);
            let wp = welsh_powell(&g);
            let greedy = greedy_coloring(&g, Order::LargestFirst);
            assert_eq!(wp, greedy, "the two forms disagree");
            assert!(is_proper_coloring(&g, &wp));
            assert!(
                color_count(&wp) <= welsh_powell_bound(&g),
                "the Welsh-Powell bound was exceeded"
            );
        }
    }

    /// The chromatic number against the closed forms it is known by, and
    /// against exhaustive search on small graphs.
    #[test]
    fn chromatic_number_matches_closed_forms_and_brute_force() {
        assert_eq!(chromatic_number_exact_small(&Graph::new(0, false)), 0);
        assert_eq!(chromatic_number_exact_small(&Graph::new(5, false)), 1);
        for n in 1..=7 {
            assert_eq!(chromatic_number_exact_small(&complete_graph(n)), n, "K_{n}");
        }
        for n in 3..=9 {
            let want = if n % 2 == 0 { 2 } else { 3 };
            assert_eq!(chromatic_number_exact_small(&cycle_graph(n)), want, "C_{n}");
        }
        assert_eq!(chromatic_number_exact_small(&complete_bipartite(3, 4)), 2);
        // The Petersen graph is 3-chromatic: it has odd cycles, so not two,
        // and an explicit 3-colouring exists.
        assert_eq!(chromatic_number_exact_small(&petersen_graph()), 3);

        let mut rng = Rng::new(0x_C480);
        for _ in 0..120 {
            let n = 1 + pick(&mut rng, 8);
            let g = random_graph(n, 0.2 + 0.6 * rng.next_f64(), &mut rng);
            let chi = chromatic_number_exact_small(&g);
            // Exactly: no fewer colours suffice, and that many do.
            assert!(count_colorings(&g, chi) > 0, "chi colours do not suffice");
            if chi > 0 {
                assert_eq!(count_colorings(&g, chi - 1), 0, "fewer colours would do");
            }
            // Sandwiched by the clique number below and Delta + 1 above.
            assert!(chi >= max_clique_bron_kerbosch(&g).len());
            assert!(chi <= max_degree(&g) + 1);
        }
    }

    /// The chromatic polynomial must count what it says it counts: its value
    /// at every integer `k` is the number of proper `k`-colourings.
    #[test]
    fn chromatic_polynomial_counts_proper_colorings() {
        let mut rng = Rng::new(0x_C407);
        for _ in 0..60 {
            let n = 1 + pick(&mut rng, 7);
            let g = random_graph(n, 0.2 + 0.6 * rng.next_f64(), &mut rng);
            let p = chromatic_polynomial_small(&g);
            for k in 0..=5usize {
                let want = Rational::from_i64(count_colorings(&g, k), 1);
                let got = p.eval(&Rational::from_i64(k as i64, 1));
                assert_eq!(got, want, "P({k}) is wrong on a graph of {n} vertices");
            }
            // Degree is n, it is monic, and the next coefficient is minus the
            // edge count -- all three read straight off deletion-contraction.
            assert_eq!(p.degree(), n);
            assert_eq!(p.leading(), Rational::from_i64(1, 1));
            if n >= 1 {
                let m = g.edge_count() as i64;
                assert_eq!(p.c[n - 1], Rational::from_i64(-m, 1), "the x^(n-1) coefficient");
            }
            // The chromatic number is the least k with a positive value.
            let chi = (0..=n)
                .find(|&k| p.eval(&Rational::from_i64(k as i64, 1)) != Rational::zero())
                .expect("n colours always suffice");
            assert_eq!(chi, chromatic_number_exact_small(&g));
        }

        // The roadmap's landmark: P(C_5, 3) = 30.
        let c5 = chromatic_polynomial_small(&cycle_graph(5));
        assert_eq!(c5.eval(&Rational::from_i64(3, 1)), Rational::from_i64(30, 1));
        // And the cycle's closed form, (x-1)^n + (-1)^n (x-1).
        for n in 3..=8usize {
            let p = chromatic_polynomial_small(&cycle_graph(n));
            for k in 0..=6i64 {
                let want = (k - 1).pow(n as u32) + if n % 2 == 0 { k - 1 } else { -(k - 1) };
                assert_eq!(
                    p.eval(&Rational::from_i64(k, 1)),
                    Rational::from_i64(want, 1),
                    "C_{n} at {k}"
                );
            }
        }
        // A tree on n vertices always has x(x-1)^(n-1), whatever its shape.
        for n in 1..=8usize {
            let p = chromatic_polynomial_small(&path_graph(n));
            for k in 0..=5i64 {
                let want = k * (k - 1).pow(n as u32 - 1);
                assert_eq!(p.eval(&Rational::from_i64(k, 1)), Rational::from_i64(want, 1));
            }
        }
        // A complete graph gives the falling factorial.
        for n in 1..=6usize {
            let p = chromatic_polynomial_small(&complete_graph(n));
            for k in 0..=7i64 {
                let want: i64 = (0..n as i64).map(|i| k - i).product();
                assert_eq!(p.eval(&Rational::from_i64(k, 1)), Rational::from_i64(want, 1));
            }
        }
    }

    /// Vizing's theorem, constructively: at most `Delta + 1` colours, no two
    /// edges at a vertex alike, and never fewer than `Delta`, which is a
    /// lower bound for any edge colouring at all.
    #[test]
    fn vizing_edge_coloring_is_proper_and_within_the_bound() {
        let mut rng = Rng::new(0x_1712);
        for _ in 0..400 {
            let n = 1 + pick(&mut rng, 22);
            let g = random_graph(n, 0.05 + 0.9 * rng.next_f64(), &mut rng);
            let c = edge_coloring_vizing(&g);
            assert!(is_proper_edge_coloring(&g, &c), "not a proper edge colouring");
            let d = max_degree(&g);
            let used = color_count(&c);
            assert!(used <= d + 1, "used {used} colours, Vizing allows {}", d + 1);
            if g.edge_count() > 0 {
                assert!(used >= d, "a vertex of degree {d} needs {d} colours");
            }
        }
        // Konig: a bipartite graph is class one, so exactly Delta suffices --
        // and the construction must at least not exceed it by more than one.
        for g in [complete_bipartite(3, 3), complete_bipartite(2, 5), path_graph(9)] {
            let c = edge_coloring_vizing(&g);
            assert!(is_proper_edge_coloring(&g, &c));
            assert!(color_count(&c) <= max_degree(&g) + 1);
        }
        // An odd cycle is class two: Delta is two but three colours are
        // needed, since the edges of an odd cycle cannot be split into two
        // matchings.
        for n in [3usize, 5, 7, 9] {
            let g = cycle_graph(n);
            let c = edge_coloring_vizing(&g);
            assert!(is_proper_edge_coloring(&g, &c));
            assert_eq!(color_count(&c), 3, "an odd cycle needs Delta + 1");
        }
        // The Petersen graph is the standard class-two cubic example.
        let p = petersen_graph();
        let c = edge_coloring_vizing(&p);
        assert!(is_proper_edge_coloring(&p, &c));
        assert!(color_count(&c) <= 4);
    }

    /// An interval graph's greedy sweep is optimal, and the optimum is the
    /// maximum number of intervals covering any single point.
    #[test]
    fn interval_coloring_uses_exactly_the_maximum_overlap() {
        let mut rng = Rng::new(0x_147E);
        for _ in 0..300 {
            let k = 1 + pick(&mut rng, 20);
            let iv: Vec<(f64, f64)> = (0..k)
                .map(|_| {
                    let a = (20.0 * rng.next_f64()).floor();
                    (a, a + (6.0 * rng.next_f64()).floor())
                })
                .collect();
            let c = interval_graph_coloring(&iv);
            assert_eq!(c.len(), k);
            // Overlapping intervals get different colours. Two half-open
            // intervals meet exactly when the later start precedes the
            // earlier end -- the usual two-comparison form of that assumes
            // both are non-empty, which these are not.
            for i in 0..k {
                for j in i + 1..k {
                    if iv[i].0.max(iv[j].0) < iv[i].1.min(iv[j].1) {
                        assert_ne!(c[i], c[j], "overlapping intervals {i} and {j} share a colour");
                    }
                }
            }
            // The count is the maximum overlap, which no colouring can beat:
            // intervals through a common point are pairwise adjacent.
            let mut worst = 0usize;
            for &(a, _) in &iv {
                worst = worst.max(iv.iter().filter(|&&(x, y)| x <= a && a < y).count());
            }
            assert_eq!(color_count(&c), worst.max(1), "not the maximum overlap");
        }
        // Degenerate: a point interval overlaps nothing, so every interval of
        // zero width can share one colour.
        let c = interval_graph_coloring(&[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]);
        assert_eq!(color_count(&c), 1);
    }

    /// Map colouring against the graph the map describes: same problem, so
    /// the same answer for every k.
    #[test]
    fn map_coloring_agrees_with_the_chromatic_number() {
        let mut rng = Rng::new(0x_4A70);
        for _ in 0..120 {
            let n = 1 + pick(&mut rng, 8);
            let g = random_graph(n, 0.2 + 0.6 * rng.next_f64(), &mut rng);
            let adjacency: Vec<Vec<usize>> =
                (0..n).map(|v| simple_neighbors(&g)[v].iter().copied().collect()).collect();
            let chi = chromatic_number_exact_small(&g);
            for k in 0..=n {
                let got = map_coloring_backtrack(&adjacency, k);
                assert_eq!(got.is_some(), k >= chi, "k = {k} against chi = {chi}");
                if let Some(c) = got {
                    assert!(is_proper_coloring(&g, &c));
                    assert!(c.iter().all(|&x| x < k));
                }
            }
        }
        // A region bordering itself has no colouring, whatever k is.
        assert!(map_coloring_backtrack(&[vec![0]], 5).is_none());
        // K4 needs four; three will not do however long it searches.
        let k4: Vec<Vec<usize>> = (0..4).map(|v| (0..4).filter(|&w| w != v).collect()).collect();
        assert!(map_coloring_backtrack(&k4, 3).is_none());
        assert!(map_coloring_backtrack(&k4, 4).is_some());
    }

    /// The time-limited search must agree with the exact one when given time,
    /// and must decline immediately when given none.
    #[test]
    fn k_colorable_search_agrees_with_the_exact_answer() {
        let mut rng = Rng::new(0x_5A70);
        let generous = Duration::from_secs(30);
        for _ in 0..80 {
            let n = 1 + pick(&mut rng, 9);
            let g = random_graph(n, 0.2 + 0.6 * rng.next_f64(), &mut rng);
            let chi = chromatic_number_exact_small(&g);
            for k in 0..=n {
                match is_k_colorable_sat_style(&g, k, generous) {
                    Some(c) => {
                        assert!(k >= chi, "claimed {k} colours where {chi} are needed");
                        assert!(is_proper_coloring(&g, &c));
                        assert!(c.iter().all(|&x| x < k));
                    }
                    None => assert!(k < chi, "found no {k}-colouring but chi is {chi}"),
                }
            }
        }
        // No budget, no answer -- even for a graph that trivially has one.
        assert!(is_k_colorable_sat_style(&complete_graph(2), 9, Duration::ZERO).is_none());
        // A self-loop is never colourable.
        let mut loopy = Graph::new(2, false);
        loopy.add_edge(0, 0, 1.0);
        assert!(is_k_colorable_sat_style(&loopy, 9, generous).is_none());
    }

    /// Cliques and independent sets are the same objects seen through the
    /// complement, and covers are the complements of independent sets.
    #[test]
    fn cliques_covers_and_independent_sets_are_dual() {
        let mut rng = Rng::new(0x_C119);
        for _ in 0..200 {
            let n = 1 + pick(&mut rng, 10);
            let g = random_graph(n, 0.2 + 0.6 * rng.next_f64(), &mut rng);
            let adj = simple_neighbors(&g);
            let cliques = all_maximal_cliques(&g);
            // Each is a clique, and maximal: nothing outside is adjacent to
            // all of it.
            for c in &cliques {
                for i in 0..c.len() {
                    for j in i + 1..c.len() {
                        assert!(adj[c[i]].contains(&c[j]), "not a clique");
                    }
                }
                assert!(
                    (0..n)
                        .filter(|v| !c.contains(v))
                        .all(|v| !c.iter().all(|&w| adj[v].contains(&w))),
                    "a vertex could be added, so it is not maximal"
                );
            }
            // Every clique of the graph sits inside one of them, which is what
            // makes taking the largest exact.
            let best = max_clique_bron_kerbosch(&g);
            assert_eq!(best.len(), cliques.iter().map(Vec::len).max().unwrap_or(0));
            let brute = (0..1u64 << n)
                .filter(|&s| {
                    let vs = bits(s);
                    (0..vs.len())
                        .all(|i| (i + 1..vs.len()).all(|j| adj[vs[i]].contains(&vs[j])))
                })
                .map(|s| s.count_ones() as usize)
                .max()
                .unwrap_or(0);
            assert_eq!(best.len(), brute, "not a maximum clique");

            // Gallai: alpha + tau = n.
            let alpha = max_independent_set_small(&g);
            let tau = vertex_cover_exact_small(&g);
            assert_eq!(alpha.len() + tau.len(), n, "alpha + tau is not n");
            for i in 0..alpha.len() {
                for j in i + 1..alpha.len() {
                    assert!(!adj[alpha[i]].contains(&alpha[j]), "not independent");
                }
            }
            for (u, v, _) in g.edges() {
                assert!(tau.contains(&u) || tau.contains(&v), "an edge is uncovered");
            }
            // A clique in the complement is an independent set here.
            assert_eq!(alpha.len(), max_clique_bron_kerbosch(&g.complement()).len());

            // The greedy independent set is maximal and never beats the
            // maximum, and its size clears the Caro-Wei bound.
            let greedy = independent_set_greedy(&g);
            for i in 0..greedy.len() {
                for j in i + 1..greedy.len() {
                    assert!(!adj[greedy[i]].contains(&greedy[j]));
                }
            }
            assert!(
                (0..n)
                    .filter(|v| !greedy.contains(v))
                    .all(|v| greedy.iter().any(|&w| adj[v].contains(&w))),
                "the greedy set is not maximal"
            );
            assert!(greedy.len() <= alpha.len());
            let caro_wei: f64 = (0..n).map(|v| 1.0 / (adj[v].len() as f64 + 1.0)).sum();
            assert!(
                greedy.len() as f64 >= caro_wei - 1e-9,
                "below the Caro-Wei bound of {caro_wei}"
            );

            // The two-approximation covers every edge and is within twice the
            // optimum, both of which follow from its maximal matching.
            let approx = vertex_cover_2approx(&g);
            for (u, v, _) in g.edges() {
                assert!(approx.contains(&u) || approx.contains(&v), "an edge is uncovered");
            }
            assert!(approx.len() <= 2 * tau.len(), "worse than twice the optimum");
            assert!(approx.len() >= tau.len());
        }
    }

    /// A dominating set has to dominate, and the greedy choice must not be
    /// beaten by more than set cover's logarithmic factor.
    #[test]
    fn dominating_set_dominates() {
        let mut rng = Rng::new(0x_D011);
        for _ in 0..200 {
            let n = 1 + pick(&mut rng, 11);
            let g = random_graph(n, 0.15 + 0.5 * rng.next_f64(), &mut rng);
            let adj = simple_neighbors(&g);
            let d = dominating_set_greedy(&g);
            for v in 0..n {
                assert!(
                    d.contains(&v) || adj[v].iter().any(|w| d.contains(w)),
                    "vertex {v} is not dominated"
                );
            }
            // Against the exact optimum by exhaustion.
            let best = (0..1u64 << n)
                .filter(|&s| {
                    (0..n).all(|v| {
                        s & (1 << v) != 0 || adj[v].iter().any(|&w| s & (1 << w) != 0)
                    })
                })
                .map(|s| s.count_ones() as usize)
                .min()
                .expect("all of V dominates");
            assert!(d.len() >= best);
            let bound = best as f64 * ((n as f64).ln() + 1.0);
            assert!(d.len() as f64 <= bound + 1e-9, "greedy {} vs bound {bound}", d.len());
        }
        // A star: the centre alone dominates it.
        let mut star = Graph::new(7, false);
        for v in 1..7 {
            star.add_edge(0, v, 1.0);
        }
        assert_eq!(dominating_set_greedy(&star), vec![0]);
        // No edges: every vertex must be in the set.
        assert_eq!(dominating_set_greedy(&Graph::new(4, false)), vec![0, 1, 2, 3]);
    }

    /// Removing a feedback arc set must leave a directed acyclic graph, and
    /// on a graph that is already acyclic the set must be empty.
    #[test]
    fn feedback_arc_set_leaves_a_dag() {
        let mut rng = Rng::new(0x_FA55);
        for _ in 0..200 {
            let n = 1 + pick(&mut rng, 10);
            let mut g = Graph::new(n, true);
            for u in 0..n {
                for v in 0..n {
                    if u != v && rng.next_f64() < 0.25 {
                        g.add_edge(u, v, 1.0);
                    }
                }
            }
            let fas = feedback_arc_set_greedy(&g);
            let mut h = Graph::new(n, true);
            let mut left: Vec<(usize, usize)> = Vec::new();
            for u in 0..n {
                for &(v, _) in &g.adj[u] {
                    left.push((u, v));
                }
            }
            for &a in &fas {
                let i = left.iter().position(|&x| x == a).expect("the arc is in the graph");
                left.remove(i);
            }
            for &(u, v) in &left {
                h.add_edge(u, v, 1.0);
            }
            assert!(h.is_dag(), "removing the set left a cycle");
            // Nothing removed unnecessarily on an already acyclic graph.
            let mut dag = Graph::new(n, true);
            for u in 0..n {
                for v in u + 1..n {
                    if rng.next_f64() < 0.4 {
                        dag.add_edge(u, v, 1.0);
                    }
                }
            }
            assert!(dag.is_dag());
            assert!(feedback_arc_set_greedy(&dag).is_empty(), "cut arcs from a DAG");
        }
        // A directed triangle needs exactly one arc removed.
        let mut tri = Graph::new(3, true);
        tri.add_edge(0, 1, 1.0);
        tri.add_edge(1, 2, 1.0);
        tri.add_edge(2, 0, 1.0);
        assert_eq!(feedback_arc_set_greedy(&tri).len(), 1);
        // A self-loop is a cycle no ordering can break.
        let mut sl = Graph::new(1, true);
        sl.add_edge(0, 0, 1.0);
        assert_eq!(feedback_arc_set_greedy(&sl), vec![(0, 0)]);
    }
}
