//! Matchings: bipartite, general, weighted, and stable.
//!
//! A matching is a set of edges no two of which share a vertex. It is returned
//! as a partner array: `m[v]` is the vertex matched to `v`, or `None` when `v`
//! is unmatched. That form is symmetric by construction, so `m[m[v]] == v`
//! whenever `m[v]` is `Some`.

use crate::graph::core::Graph;
use crate::linalg::matrix::Matrix;

/// A maximum matching of a bipartite graph, by Hopcroft-Karp.
///
/// The left side is `0..left_n` and the right side `0..right_n`, numbered
/// separately; `edges` gives `(left, right)` pairs. The returned array is
/// indexed by left vertex and holds the right vertex matched to it.
///
/// Hopcroft-Karp augments along a maximal set of shortest augmenting paths at
/// once rather than one at a time, which bounds the number of phases by
/// `sqrt(V)` instead of `V`.
///
/// # Panics
/// Panics if an edge names a vertex outside its side.
#[must_use]
pub fn hopcroft_karp(left_n: usize, right_n: usize, edges: &[(usize, usize)]) -> Vec<Option<usize>> {
    let mut adj = vec![Vec::new(); left_n];
    for &(l, r) in edges {
        assert!(l < left_n, "left vertex {l} is outside 0..{left_n}");
        assert!(r < right_n, "right vertex {r} is outside 0..{right_n}");
        adj[l].push(r);
    }
    let mut match_l: Vec<Option<usize>> = vec![None; left_n];
    let mut match_r: Vec<Option<usize>> = vec![None; right_n];

    loop {
        // Phase one: layer the free left vertices by breadth-first search,
        // stopping at the first level that reaches a free right vertex.
        let mut dist = vec![usize::MAX; left_n];
        let mut queue = std::collections::VecDeque::new();
        for l in 0..left_n {
            if match_l[l].is_none() {
                dist[l] = 0;
                queue.push_back(l);
            }
        }
        let mut found = false;
        while let Some(l) = queue.pop_front() {
            for &r in &adj[l] {
                match match_r[r] {
                    None => found = true,
                    Some(next) if dist[next] == usize::MAX => {
                        dist[next] = dist[l] + 1;
                        queue.push_back(next);
                    }
                    Some(_) => {}
                }
            }
        }
        if !found {
            break;
        }
        // Phase two: augment along vertex-disjoint shortest paths.
        for l in 0..left_n {
            if match_l[l].is_none() {
                hk_augment(l, &adj, &mut match_l, &mut match_r, &mut dist);
            }
        }
    }
    match_l
}

fn hk_augment(
    l: usize,
    adj: &[Vec<usize>],
    match_l: &mut [Option<usize>],
    match_r: &mut [Option<usize>],
    dist: &mut [usize],
) -> bool {
    for idx in 0..adj[l].len() {
        let r = adj[l][idx];
        let ok = match match_r[r] {
            None => true,
            // Only descend one layer, which keeps the paths shortest.
            Some(next) => dist[next] == dist[l] + 1 && hk_augment(next, adj, match_l, match_r, dist),
        };
        if ok {
            match_l[l] = Some(r);
            match_r[r] = Some(l);
            return true;
        }
    }
    // Mark l dead for this phase so it is not retried.
    dist[l] = usize::MAX;
    false
}

/// The minimum-cost perfect assignment, by the Hungarian algorithm in its
/// `O(n^3)` shortest-augmenting-path form.
///
/// `cost` must be square. Returns the total cost and the column assigned to
/// each row.
///
/// The algorithm maintains dual potentials that keep every reduced cost
/// non-negative, so each augmenting search is a Dijkstra rather than a
/// Bellman-Ford; that is what turns the naive `O(n^4)` into `O(n^3)`.
///
/// # Panics
/// Panics if `cost` is not square or contains a non-finite entry.
#[must_use]
pub fn hungarian(cost: &Matrix) -> (f64, Vec<usize>) {
    assert_eq!(cost.rows, cost.cols, "the cost matrix must be square");
    assert!(
        cost.data.iter().all(|x| x.is_finite()),
        "costs must be finite"
    );
    // No zero-size guard: Matrix::zeros rejects a zero dimension, so an empty
    // cost matrix cannot be constructed and the branch would be dead.
    let n = cost.rows;
    // One-based internally, with index 0 as the sentinel for "unassigned".
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    // p[j] is the row assigned to column j; way[j] the column it came from.
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut min_v = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                // Reduced cost of putting row i0 in column j.
                let cur = cost.get(i0 - 1, j - 1) - u[i0] - v[j];
                if cur < min_v[j] {
                    min_v[j] = cur;
                    way[j] = j0;
                }
                if min_v[j] < delta {
                    delta = min_v[j];
                    j1 = j;
                }
            }
            // Shift the potentials so the tight edges stay tight.
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Walk the alternating path back, reassigning as we go.
        while j0 != 0 {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }

    let mut assignment = vec![0usize; n];
    for j in 1..=n {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    let total = (0..n).map(|i| cost.get(i, assignment[i])).sum();
    (total, assignment)
}

/// The minimum-cost assignment by the auction algorithm.
///
/// Rows bid for columns, raising each column's price by at least `eps` to win
/// it. The final assignment is within `n * eps` of optimal, so a small `eps`
/// buys accuracy at the cost of more rounds. Scaling `eps` down geometrically
/// -- which this does -- reaches the exact optimum for integer costs and a
/// very good one otherwise.
///
/// Returns the total cost and the column assigned to each row.
///
/// # Panics
/// Panics if `cost` is not square, contains a non-finite entry, or `eps` is
/// not positive.
#[must_use]
pub fn auction_assignment(cost: &Matrix, eps: f64) -> (f64, Vec<usize>) {
    assert_eq!(cost.rows, cost.cols, "the cost matrix must be square");
    assert!(eps > 0.0, "eps must be positive");
    assert!(
        cost.data.iter().all(|x| x.is_finite()),
        "costs must be finite"
    );
    let n = cost.rows;
    // Auction maximises, so work with negated costs.
    let value = |i: usize, j: usize| -cost.get(i, j);
    let mut price = vec![0.0f64; n];
    let mut owner: Vec<Option<usize>> = vec![None; n];
    let mut assignment: Vec<Option<usize>> = vec![None; n];

    // Epsilon scaling: start coarse and refine, which avoids the price wars a
    // single small epsilon causes.
    //
    // The prices carry over between rounds and only the assignment is torn
    // down. That is the whole mechanism: each round starts from the previous
    // round's near-equilibrium prices and needs few bids to settle. Resetting
    // the prices as well makes the last round a fresh auction at the smallest
    // epsilon, which is the slowest variant there is and takes on the order of
    // n^2 * (cost range) / eps bids.
    let mut e = (n as f64).max(1.0);
    while e >= eps {
        owner.iter_mut().for_each(|o| *o = None);
        assignment.iter_mut().for_each(|a| *a = None);
        let mut guard = 0usize;
        let cap = 1_000 * n * n + 1_000;
        while assignment.iter().any(Option::is_none) && guard < cap {
            guard += 1;
            let i = assignment.iter().position(Option::is_none).unwrap();
            // Best and second-best net value for this row.
            let mut best_j = 0usize;
            let mut best = f64::NEG_INFINITY;
            let mut second = f64::NEG_INFINITY;
            for j in 0..n {
                let net = value(i, j) - price[j];
                if net > best {
                    second = best;
                    best = net;
                    best_j = j;
                } else if net > second {
                    second = net;
                }
            }
            // Bid up by the margin plus epsilon, so progress is guaranteed.
            price[best_j] += best - second + e;
            if let Some(prev) = owner[best_j] {
                assignment[prev] = None;
            }
            owner[best_j] = Some(i);
            assignment[i] = Some(best_j);
        }
        e /= 4.0;
    }
    let out: Vec<usize> = assignment.into_iter().map(|a| a.unwrap_or(0)).collect();
    let total = (0..n).map(|i| cost.get(i, out[i])).sum();
    (total, out)
}

/// A maximum matching of a general graph, by Edmonds' blossom algorithm.
///
/// The bipartite algorithms fail on odd cycles: an augmenting search can enter
/// one and come back out at the same vertex with the wrong parity. Edmonds'
/// insight is to contract each such cycle -- a blossom -- to a single vertex,
/// search the contracted graph, and lift the result back.
///
/// The lifting is the part that is easy to get wrong. Contracting is not
/// enough: when a blossom forms, the parent pointers of every vertex on the
/// odd cycle have to be rewired so that a later augmenting path can be traced
/// back *through* the blossom the long way round. Without that rewiring the
/// traceback leaves the tree by the wrong edge and produces an asymmetric
/// pairing. `mark_blossom_path` below is what does it.
///
/// Returns the partner array over all vertices.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn blossom_max_matching(g: &Graph) -> Vec<Option<usize>> {
    assert!(!g.directed, "a matching is defined on an undirected graph");
    let n = g.n;
    let mut adj = vec![Vec::new(); n];
    for (u, v, _) in g.edges() {
        if u != v {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    // NONE stands in for "unmatched" or "no parent" so the index arithmetic
    // stays direct; the Option form is rebuilt at the end.
    const NONE: usize = usize::MAX;
    let mut mate = vec![NONE; n];

    // Greedy start: every edge taken now is one augmentation not needed.
    for u in 0..n {
        if mate[u] == NONE {
            if let Some(&v) = adj[u].iter().find(|&&v| mate[v] == NONE) {
                mate[u] = v;
                mate[v] = u;
            }
        }
    }

    let mut parent = vec![NONE; n];
    let mut base: Vec<usize> = (0..n).collect();
    let mut outer = vec![false; n];

    for root in 0..n {
        if mate[root] != NONE {
            continue;
        }
        // Grow an alternating tree from root.
        outer.iter_mut().for_each(|x| *x = false);
        parent.iter_mut().for_each(|x| *x = NONE);
        for (i, b) in base.iter_mut().enumerate() {
            *b = i;
        }
        outer[root] = true;
        let mut queue = std::collections::VecDeque::from(vec![root]);
        let mut found = NONE;

        'search: while let Some(v) = queue.pop_front() {
            for idx in 0..adj[v].len() {
                let to = adj[v][idx];
                if base[v] == base[to] || mate[v] == to {
                    continue;
                }
                if to == root || (mate[to] != NONE && parent[mate[to]] != NONE) {
                    // An odd cycle closes here: contract it.
                    let curbase = blossom_base(&base, &mate, &parent, v, to);
                    let mut in_blossom = vec![false; n];
                    mark_blossom_path(&base, &mate, &mut parent, &mut in_blossom, v, curbase, to);
                    mark_blossom_path(&base, &mate, &mut parent, &mut in_blossom, to, curbase, v);
                    for i in 0..n {
                        if in_blossom[base[i]] {
                            base[i] = curbase;
                            if !outer[i] {
                                outer[i] = true;
                                queue.push_back(i);
                            }
                        }
                    }
                } else if parent[to] == NONE {
                    parent[to] = v;
                    if mate[to] == NONE {
                        // An augmenting path, ending at an unmatched vertex.
                        found = to;
                        break 'search;
                    }
                    outer[mate[to]] = true;
                    queue.push_back(mate[to]);
                }
            }
        }

        // Flip the path, which grows the matching by one.
        let mut u = found;
        while u != NONE {
            let pv = parent[u];
            let ppv = mate[pv];
            mate[u] = pv;
            mate[pv] = u;
            u = ppv;
        }
    }
    mate
        .into_iter()
        .map(|x| if x == NONE { None } else { Some(x) })
        .collect()
}

/// The base of the blossom formed by joining `u` and `v`: their lowest common
/// ancestor in the alternating tree, taken over bases rather than vertices.
fn blossom_base(
    base: &[usize],
    mate: &[usize],
    parent: &[usize],
    mut u: usize,
    mut v: usize,
) -> usize {
    const NONE: usize = usize::MAX;
    let mut seen = vec![false; base.len()];
    // Climb from u, marking every base on the way to the root.
    loop {
        u = base[u];
        seen[u] = true;
        if mate[u] == NONE {
            break;
        }
        u = parent[mate[u]];
    }
    // Climb from v until a marked base repeats: that is the meeting point.
    loop {
        v = base[v];
        if seen[v] {
            return v;
        }
        v = parent[mate[v]];
    }
}

/// Walks from `v` up to the blossom base `b`, marking the bases it passes and
/// rewiring the parent pointers to point back along the cycle.
///
/// The rewiring is the lifting step. After it, tracing parents from any vertex
/// of the blossom reaches the base by an even-length alternating walk, which
/// is what makes a later augmenting path through the blossom valid.
fn mark_blossom_path(
    base: &[usize],
    mate: &[usize],
    parent: &mut [usize],
    in_blossom: &mut [bool],
    mut v: usize,
    b: usize,
    mut child: usize,
) {
    while base[v] != b {
        in_blossom[base[v]] = true;
        in_blossom[base[mate[v]]] = true;
        parent[v] = child;
        child = mate[v];
        v = parent[mate[v]];
    }
}

/// A stable marriage by the Gale-Shapley algorithm.
///
/// `prefs_a[i]` ranks every member of the other side in decreasing preference,
/// and likewise `prefs_b`. Returns, for each member of side A, the member of
/// side B they are matched to.
///
/// The result is the A-optimal stable matching: every proposer gets the best
/// partner they could have in any stable matching, and every receiver the
/// worst. That asymmetry is a property of the algorithm, not an artefact.
///
/// # Panics
/// Panics unless both preference lists are complete permutations of the other
/// side, and the two sides are the same size.
#[must_use]
pub fn stable_marriage(prefs_a: &[Vec<usize>], prefs_b: &[Vec<usize>]) -> Vec<usize> {
    let n = prefs_a.len();
    assert_eq!(prefs_b.len(), n, "both sides must be the same size");
    for p in prefs_a.iter().chain(prefs_b.iter()) {
        assert!(
            crate::discrete::combinatorics::is_permutation(p) && p.len() == n,
            "each preference list must rank every member of the other side"
        );
    }
    // rank_b[j][i] is how highly j rates i; smaller is better.
    let mut rank_b = vec![vec![0usize; n]; n];
    for (j, p) in prefs_b.iter().enumerate() {
        for (r, &i) in p.iter().enumerate() {
            rank_b[j][i] = r;
        }
    }
    let mut next_proposal = vec![0usize; n];
    let mut partner_b: Vec<Option<usize>> = vec![None; n];
    let mut free: Vec<usize> = (0..n).rev().collect();

    while let Some(i) = free.pop() {
        let j = prefs_a[i][next_proposal[i]];
        next_proposal[i] += 1;
        match partner_b[j] {
            None => partner_b[j] = Some(i),
            Some(k) if rank_b[j][i] < rank_b[j][k] => {
                // j prefers the new proposer, so k goes back to the pool.
                partner_b[j] = Some(i);
                free.push(k);
            }
            Some(_) => free.push(i),
        }
    }
    let mut out = vec![0usize; n];
    for (j, p) in partner_b.iter().enumerate() {
        out[p.expect("every receiver ends matched")] = j;
    }
    out
}

/// A stable roommates matching, or `None` when none exists.
///
/// Unlike stable marriage, this is a single pool with no sides, and a stable
/// matching need not exist at all -- the smallest counterexample has four
/// people. Irving's algorithm: a proposal phase, then repeated elimination of
/// rotations.
///
/// `prefs[i]` ranks the other `n - 1` people in decreasing preference.
///
/// # Panics
/// Panics unless `n` is even and each list ranks exactly the other people.
#[must_use]
pub fn stable_roommates(prefs: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = prefs.len();
    assert!(n.is_multiple_of(2), "a roommates instance needs an even size");
    for (i, p) in prefs.iter().enumerate() {
        assert_eq!(p.len(), n - 1, "each list must rank the other {} people", n - 1);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), n - 1, "person {i} has a repeated preference");
        assert!(!p.contains(&i), "person {i} cannot rank themselves");
    }
    // rank[i][j] is how highly i rates j.
    let mut rank = vec![vec![usize::MAX; n]; n];
    for (i, p) in prefs.iter().enumerate() {
        for (r, &j) in p.iter().enumerate() {
            rank[i][j] = r;
        }
    }
    // Working lists, shortened as pairs are ruled out.
    let mut list: Vec<Vec<usize>> = prefs.to_vec();

    // Phase one: proposals, as in Gale-Shapley but with one pool.
    let mut held: Vec<Option<usize>> = vec![None; n];
    let mut next = vec![0usize; n];
    let mut free: Vec<usize> = (0..n).rev().collect();
    while let Some(i) = free.pop() {
        loop {
            if next[i] >= list[i].len() {
                return None;
            }
            let j = list[i][next[i]];
            next[i] += 1;
            match held[j] {
                None => {
                    held[j] = Some(i);
                    break;
                }
                Some(k) if rank[j][i] < rank[j][k] => {
                    held[j] = Some(i);
                    free.push(k);
                    break;
                }
                Some(_) => {}
            }
        }
    }
    if held.iter().any(Option::is_none) {
        return None;
    }
    // Trim: j rejects everyone it rates below its holder, and symmetrically.
    for j in 0..n {
        let h = held[j].unwrap();
        let cutoff = rank[j][h];
        list[j].retain(|&x| rank[j][x] <= cutoff);
    }
    for i in 0..n {
        let keep: Vec<usize> = list[i]
            .iter()
            .copied()
            .filter(|&j| list[j].contains(&i))
            .collect();
        list[i] = keep;
    }

    // Phase two: eliminate rotations until every list has one entry.
    loop {
        if list.iter().any(Vec::is_empty) {
            return None;
        }
        let Some(start) = (0..n).find(|&i| list[i].len() > 1) else {
            break;
        };
        // Find a rotation: alternate "second choice" and "last holder".
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut seen = vec![usize::MAX; n];
        let mut p = start;
        let mut step = 0usize;
        loop {
            if seen[p] != usize::MAX {
                // The cycle closes at the first repeat.
                let cut = seen[p];
                xs.drain(..cut);
                ys.drain(..cut);
                break;
            }
            seen[p] = step;
            step += 1;
            if list[p].len() < 2 {
                return None;
            }
            let q = list[p][1];
            xs.push(p);
            ys.push(q);
            p = *list[q].last().expect("the list is non-empty");
        }
        // Remove the rotation by breaking the pair (x_{i+1}, y_i).
        //
        // The comparison has to be strict. x_{i+1} is *defined* as the last
        // entry of y_i's list, so rejecting everyone y_i rates strictly worse
        // than x_{i+1} rejects nobody: the lists never shrink, no rotation is
        // ever consumed, and the outer loop spins forever. x_{i+1} itself is
        // the entry that must go, together with everyone below it.
        for k in 0..xs.len() {
            let y = ys[k];
            let x_next = xs[(k + 1) % xs.len()];
            let cutoff = rank[y][x_next];
            let doomed: Vec<usize> = list[y]
                .iter()
                .copied()
                .filter(|&z| rank[y][z] >= cutoff)
                .collect();
            list[y].retain(|&z| rank[y][z] < cutoff);
            for z in doomed {
                list[z].retain(|&w| w != y);
            }
        }
    }
    let out: Vec<usize> = (0..n).map(|i| list[i][0]).collect();
    // A valid matching must pair people up symmetrically.
    if (0..n).any(|i| out[out[i]] != i) {
        return None;
    }
    Some(out)
}

/// A minimum vertex cover of a bipartite graph, by Konig's theorem.
///
/// Konig's theorem says the minimum vertex cover of a bipartite graph has
/// exactly the size of its maximum matching, and names the cover: start an
/// alternating search from the unmatched left vertices, then take the left
/// vertices *not* reached together with the right vertices that are.
///
/// `left` names one side; `matching` is a partner array over all vertices.
///
/// # Panics
/// Panics if `matching` is not symmetric, or `left` names a vertex twice.
#[must_use]
pub fn konig_vertex_cover(g: &Graph, left: &[usize], matching: &[Option<usize>]) -> Vec<usize> {
    let n = g.n;
    let mut is_left = vec![false; n];
    for &v in left {
        assert!(v < n, "vertex {v} is outside 0..{n}");
        assert!(!is_left[v], "vertex {v} appears twice");
        is_left[v] = true;
    }
    for v in 0..n {
        if let Some(w) = matching[v] {
            assert_eq!(matching[w], Some(v), "the matching is not symmetric");
        }
    }
    let mut adj = vec![Vec::new(); n];
    for (u, v, _) in g.edges() {
        if u != v {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    // Alternating search from the unmatched left vertices: unmatched edges
    // going right, matched edges coming back left.
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = (0..n)
        .filter(|&v| is_left[v] && matching[v].is_none())
        .collect();
    for &v in &stack {
        seen[v] = true;
    }
    while let Some(v) = stack.pop() {
        if is_left[v] {
            for &w in &adj[v] {
                if matching[v] != Some(w) && !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        } else if let Some(w) = matching[v] {
            if !seen[w] {
                seen[w] = true;
                stack.push(w);
            }
        }
    }
    (0..n)
        .filter(|&v| if is_left[v] { !seen[v] } else { seen[v] })
        .collect()
}

/// Checks Hall's condition on a bipartite graph.
///
/// Hall's theorem says a matching saturating the left side exists exactly when
/// every subset of the left has at least as many distinct neighbours as it has
/// members. Returns `Ok(())` when it holds, or the smallest violating subset
/// found.
///
/// The violating set is not searched for over all `2^|L|` subsets: by Konig's
/// theorem the deficiency equals `|L|` minus the maximum matching, and the
/// unreached left vertices of the alternating search form a violating set.
///
/// # Errors
/// Returns the violating subset of `left` when the condition fails.
pub fn hall_condition_check(g: &Graph, left: &[usize]) -> Result<(), Vec<usize>> {
    let n = g.n;
    let mut is_left = vec![false; n];
    for &v in left {
        is_left[v] = true;
    }
    let right: Vec<usize> = (0..n).filter(|&v| !is_left[v]).collect();
    let mut index_l = vec![usize::MAX; n];
    let mut index_r = vec![usize::MAX; n];
    for (i, &v) in left.iter().enumerate() {
        index_l[v] = i;
    }
    for (i, &v) in right.iter().enumerate() {
        index_r[v] = i;
    }
    let mut edges = Vec::new();
    for (u, v, _) in g.edges() {
        if u == v {
            continue;
        }
        let (a, b) = if is_left[u] { (u, v) } else { (v, u) };
        if index_l[a] != usize::MAX && index_r[b] != usize::MAX {
            edges.push((index_l[a], index_r[b]));
        }
    }
    let m = hopcroft_karp(left.len(), right.len(), &edges);
    if m.iter().all(Option::is_some) {
        return Ok(());
    }
    // Alternating search from an unmatched left vertex reaches a set whose
    // neighbourhood is too small.
    let mut adj_l = vec![Vec::new(); left.len()];
    let mut match_r: Vec<Option<usize>> = vec![None; right.len()];
    for &(a, b) in &edges {
        adj_l[a].push(b);
    }
    for (a, &b) in m.iter().enumerate() {
        if let Some(b) = b {
            match_r[b] = Some(a);
        }
    }
    let mut seen_l = vec![false; left.len()];
    let mut seen_r = vec![false; right.len()];
    let mut stack: Vec<usize> = (0..left.len()).filter(|&a| m[a].is_none()).collect();
    for &a in &stack {
        seen_l[a] = true;
    }
    while let Some(a) = stack.pop() {
        for &b in &adj_l[a] {
            if !seen_r[b] {
                seen_r[b] = true;
                if let Some(c) = match_r[b] {
                    if !seen_l[c] {
                        seen_l[c] = true;
                        stack.push(c);
                    }
                }
            }
        }
    }
    Err((0..left.len())
        .filter(|&a| seen_l[a])
        .map(|a| left[a])
        .collect())
}

/// The maximum-weight bipartite matching, allowing an unbalanced graph and
/// leaving a vertex unmatched when that pays better.
///
/// `weights` is a left-by-right matrix. Reduces to the Hungarian algorithm by
/// padding to a square and negating, with the padding entries at zero so an
/// unprofitable match is never forced.
///
/// Returns the total weight and the partner of each left vertex.
#[must_use]
pub fn maximum_weight_bipartite(weights: &Matrix) -> (f64, Vec<Option<usize>>) {
    let (rows, cols) = (weights.rows, weights.cols);
    let n = rows.max(cols);
    // Hungarian minimises, so negate. A padded or unprofitable pair costs
    // zero, which is what makes leaving a vertex unmatched an option.
    let mut cost = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let w = if i < rows && j < cols {
                weights.get(i, j).max(0.0)
            } else {
                0.0
            };
            cost.set(i, j, -w);
        }
    }
    let (_, assignment) = hungarian(&cost);
    let mut partner = vec![None; rows];
    let mut total = 0.0;
    for i in 0..rows {
        let j = assignment[i];
        if j < cols && weights.get(i, j) > 0.0 {
            partner[i] = Some(j);
            total += weights.get(i, j);
        }
    }
    (total, partner)
}

/// The number of edges in a partner array.
#[must_use]
pub fn matching_size(m: &[Option<usize>]) -> usize {
    m.iter().filter(|x| x.is_some()).count() / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::combinatorics::{combinations_iter, permutations_iter};
    use crate::graph::core::{complete_bipartite, complete_graph, cycle_graph, petersen_graph};
    use crate::graph::flow::max_bipartite_matching_via_flow;
    use crate::monte_carlo::Rng;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6 * a.abs().max(b.abs()).max(1.0)
    }

    /// A random bipartite graph as an edge list plus the whole-graph form.
    fn random_bipartite(l: usize, r: usize, p: f64, rng: &mut Rng) -> (Vec<(usize, usize)>, Graph) {
        let mut edges = Vec::new();
        let mut g = Graph::new(l + r, false);
        for a in 0..l {
            for b in 0..r {
                if rng.next_f64() < p {
                    edges.push((a, b));
                    g.add_edge(a, l + b, 1.0);
                }
            }
        }
        (edges, g)
    }

    /// The largest matching of a bipartite edge list, by brute force.
    fn brute_bipartite(l: usize, edges: &[(usize, usize)]) -> usize {
        let mut best = 0usize;
        for k in (1..=edges.len()).rev() {
            if k <= best {
                break;
            }
            for combo in combinations_iter(edges.len(), k) {
                let mut used_l = vec![false; l];
                let mut used_r = std::collections::BTreeSet::new();
                let mut ok = true;
                for &i in &combo {
                    let (a, b) = edges[i];
                    if used_l[a] || !used_r.insert(b) {
                        ok = false;
                        break;
                    }
                    used_l[a] = true;
                }
                if ok {
                    best = k;
                    break;
                }
            }
            if best == k {
                break;
            }
        }
        best
    }

    /// Hopcroft-Karp must find a maximum matching, and it must be a matching.
    #[test]
    fn hopcroft_karp_is_maximum() {
        let mut rng = Rng::new(0x_4C41);
        for l in 1..=5usize {
            for r in 1..=5usize {
                for _ in 0..20 {
                    let (edges, g) = random_bipartite(l, r, 0.45, &mut rng);
                    let m = hopcroft_karp(l, r, &edges);
                    // Valid: each matched pair is an edge, no right vertex twice.
                    let mut seen_r = std::collections::BTreeSet::new();
                    for (a, &b) in m.iter().enumerate() {
                        if let Some(b) = b {
                            assert!(edges.contains(&(a, b)), "matched a non-edge");
                            assert!(seen_r.insert(b), "right vertex {b} matched twice");
                        }
                    }
                    let size = m.iter().filter(|x| x.is_some()).count();
                    assert_eq!(size, brute_bipartite(l, &edges), "l = {l}, r = {r}");
                    // And the flow-based routine agrees.
                    let left: Vec<usize> = (0..l).collect();
                    let via_flow = max_bipartite_matching_via_flow(&g, &left);
                    let flow_size = via_flow.iter().filter(|x| x.is_some()).count() / 2;
                    assert_eq!(size, flow_size, "Hopcroft-Karp vs flow");
                }
            }
        }
        // K_{m,n} matches the smaller side entirely.
        for l in 1..=5usize {
            for r in 1..=5usize {
                let edges: Vec<(usize, usize)> =
                    (0..l).flat_map(|a| (0..r).map(move |b| (a, b))).collect();
                let m = hopcroft_karp(l, r, &edges);
                assert_eq!(m.iter().filter(|x| x.is_some()).count(), l.min(r));
            }
        }
        // No edges, no matching.
        assert_eq!(hopcroft_karp(3, 3, &[]), vec![None, None, None]);
    }

    /// Konig's theorem: the minimum vertex cover equals the maximum matching,
    /// and the reported cover really covers every edge.
    #[test]
    fn konig_cover_matches_the_matching_size() {
        let mut rng = Rng::new(0x_C061);
        for l in 1..=5usize {
            for r in 1..=5usize {
                for _ in 0..15 {
                    let (edges, g) = random_bipartite(l, r, 0.45, &mut rng);
                    let left: Vec<usize> = (0..l).collect();
                    let m = max_bipartite_matching_via_flow(&g, &left);
                    let size = matching_size(&m);
                    let cover = konig_vertex_cover(&g, &left, &m);
                    assert_eq!(cover.len(), size, "Konig: cover {} vs matching {size}", cover.len());
                    // Every edge is covered.
                    for (u, v, _) in g.edges() {
                        assert!(
                            cover.contains(&u) || cover.contains(&v),
                            "edge ({u}, {v}) is uncovered"
                        );
                    }
                    // Minimum, by brute force over subsets.
                    let n = l + r;
                    let mut best = n;
                    for k in 0..=n {
                        let mut found = false;
                        for combo in combinations_iter(n, k) {
                            if g.edges()
                                .iter()
                                .all(|&(u, v, _)| combo.contains(&u) || combo.contains(&v))
                            {
                                found = true;
                                break;
                            }
                        }
                        if found {
                            best = k;
                            break;
                        }
                    }
                    assert_eq!(cover.len(), best, "cover is not minimum");
                    let _ = edges;
                }
            }
        }
    }

    /// Hall's theorem: a saturating matching exists exactly when no left
    /// subset outgrows its neighbourhood, and the reported violating set
    /// really violates.
    #[test]
    fn hall_condition_matches_subset_search() {
        let mut rng = Rng::new(0x_4A11);
        for l in 1..=5usize {
            for r in 1..=5usize {
                for _ in 0..15 {
                    let (_, g) = random_bipartite(l, r, 0.4, &mut rng);
                    let left: Vec<usize> = (0..l).collect();
                    // Brute force: does some subset outgrow its neighbourhood?
                    let mut violator: Option<Vec<usize>> = None;
                    for k in 1..=l {
                        for combo in combinations_iter(l, k) {
                            let mut nbrs = std::collections::BTreeSet::new();
                            for &a in &combo {
                                for &(w, _) in &g.adj[a] {
                                    nbrs.insert(w);
                                }
                            }
                            if nbrs.len() < combo.len() {
                                violator = Some(combo.clone());
                                break;
                            }
                        }
                        if violator.is_some() {
                            break;
                        }
                    }
                    match hall_condition_check(&g, &left) {
                        Ok(()) => {
                            assert!(violator.is_none(), "condition passed but {violator:?} violates");
                            // A saturating matching must then exist.
                            let m = max_bipartite_matching_via_flow(&g, &left);
                            assert_eq!(matching_size(&m), l, "Hall holds but no saturation");
                        }
                        Err(set) => {
                            assert!(violator.is_some(), "condition failed but none violates");
                            // The returned set must genuinely violate.
                            let mut nbrs = std::collections::BTreeSet::new();
                            for &a in &set {
                                for &(w, _) in &g.adj[a] {
                                    nbrs.insert(w);
                                }
                            }
                            assert!(
                                nbrs.len() < set.len(),
                                "reported set {set:?} has {} neighbours, not fewer than {}",
                                nbrs.len(),
                                set.len()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The Hungarian algorithm must find the cheapest assignment, checked
    /// against every permutation.
    #[test]
    fn hungarian_matches_brute_force() {
        let mut rng = Rng::new(0x_4055);
        for n in 1..=7usize {
            for _ in 0..15 {
                let mut cost = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        cost.set(i, j, (20.0 * rng.next_f64() - 5.0).round());
                    }
                }
                let (total, assign) = hungarian(&cost);
                assert!(
                    crate::discrete::combinatorics::is_permutation(&assign),
                    "the assignment is not a permutation: {assign:?}"
                );
                let actual: f64 = (0..n).map(|i| cost.get(i, assign[i])).sum();
                assert!(close(total, actual), "reported {total} but costs {actual}");
                let best = permutations_iter(&(0..n).collect::<Vec<_>>())
                    .map(|p| (0..n).map(|i| cost.get(i, p[i])).sum::<f64>())
                    .fold(f64::INFINITY, f64::min);
                assert!(close(total, best), "n = {n}: {total} vs brute {best}");
            }
        }
        // The identity is optimal when the diagonal is cheapest.
        let mut c = Matrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                c.set(i, j, if i == j { 0.0 } else { 1.0 });
            }
        }
        let (t, a) = hungarian(&c);
        assert!(close(t, 0.0));
        assert_eq!(a, vec![0, 1, 2, 3]);
        // A single cell is the smallest matrix there is; Matrix::zeros rejects
        // a zero dimension, so there is no empty case to check.
        let one = Matrix { rows: 1, cols: 1, data: vec![7.0] };
        assert_eq!(hungarian(&one), (7.0, vec![0]));
    }

    /// The auction algorithm must reach the same total as the Hungarian one,
    /// which is the only claim the epsilon scaling makes.
    #[test]
    fn auction_reaches_the_hungarian_optimum() {
        let mut rng = Rng::new(0x_A0C7);
        for n in 1..=6usize {
            for _ in 0..12 {
                let mut cost = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        cost.set(i, j, (20.0 * rng.next_f64()).round());
                    }
                }
                let (opt, _) = hungarian(&cost);
                let (got, assign) = auction_assignment(&cost, 1e-3);
                assert!(
                    crate::discrete::combinatorics::is_permutation(&assign),
                    "auction produced {assign:?}"
                );
                let actual: f64 = (0..n).map(|i| cost.get(i, assign[i])).sum();
                assert!(close(got, actual), "reported {got} but costs {actual}");
                // Within n * eps of optimal is the guarantee; with scaling on
                // integer costs it reaches the optimum exactly.
                assert!(
                    got <= opt + n as f64 * 1e-3 + 1e-9,
                    "n = {n}: auction {got} exceeds optimum {opt}"
                );
                assert!(got >= opt - 1e-9, "auction beat the optimum");
            }
        }
    }

    /// Blossom matching on general graphs, checked against brute force -- the
    /// odd cycles bipartite algorithms cannot handle are exactly the point.
    #[test]
    fn blossom_matches_brute_force_on_general_graphs() {
        let mut rng = Rng::new(0x_B105);
        for n in 1..=8usize {
            for _ in 0..20 {
                let mut g = Graph::new(n, false);
                for u in 0..n {
                    for v in u + 1..n {
                        if rng.next_f64() < 0.4 {
                            g.add_edge(u, v, 1.0);
                        }
                    }
                }
                let m = blossom_max_matching(&g);
                // Valid: symmetric, and every pair is an edge.
                for v in 0..n {
                    if let Some(w) = m[v] {
                        assert_eq!(m[w], Some(v), "not symmetric at {v}");
                        assert!(
                            g.adj[v].iter().any(|&(x, _)| x == w),
                            "matched a non-edge ({v}, {w})"
                        );
                    }
                }
                assert_eq!(matching_size(&m), brute_general(&g), "n = {n}");
            }
        }
        // The roadmap's case: the Petersen graph has a perfect matching of
        // size five, which no bipartite algorithm could find -- its girth is
        // five, so it is full of odd cycles.
        let p = petersen_graph();
        let m = blossom_max_matching(&p);
        assert_eq!(matching_size(&m), 5, "Petersen has a perfect matching");
        assert!(m.iter().all(Option::is_some), "every vertex must be matched");

        // An odd cycle: the maximum matching leaves exactly one vertex out.
        for n in [3usize, 5, 7, 9] {
            let c = cycle_graph(n);
            assert_eq!(matching_size(&blossom_max_matching(&c)), n / 2, "C{n}");
        }
        // An even cycle is perfectly matchable.
        for n in [4usize, 6, 8] {
            assert_eq!(matching_size(&blossom_max_matching(&cycle_graph(n))), n / 2);
        }
        // K_n: floor(n/2).
        for n in 1..=8usize {
            assert_eq!(
                matching_size(&blossom_max_matching(&complete_graph(n))),
                n / 2,
                "K{n}"
            );
        }
        // A triangle with a pendant: the blossom must be contracted to find
        // the size-two matching, which a naive search misses.
        let tri = Graph::from_edges(
            4,
            &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0)],
            false,
        );
        assert_eq!(matching_size(&blossom_max_matching(&tri)), 2);
    }

    /// The largest matching of a general graph, by brute force over edge sets.
    fn brute_general(g: &Graph) -> usize {
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
            for combo in combinations_iter(edges.len(), k) {
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
                    best = k;
                    break;
                }
            }
            if best == k {
                break;
            }
        }
        best
    }

    /// Gale-Shapley must produce a stable matching, and specifically the
    /// A-optimal one.
    #[test]
    fn gale_shapley_is_stable_and_a_optimal() {
        let mut rng = Rng::new(0x_6A1E);
        for n in 1..=5usize {
            for _ in 0..25 {
                let prefs_a: Vec<Vec<usize>> = (0..n)
                    .map(|_| crate::discrete::combinatorics::random_permutation(n, &mut rng))
                    .collect();
                let prefs_b: Vec<Vec<usize>> = (0..n)
                    .map(|_| crate::discrete::combinatorics::random_permutation(n, &mut rng))
                    .collect();
                let m = stable_marriage(&prefs_a, &prefs_b);
                assert!(
                    crate::discrete::combinatorics::is_permutation(&m),
                    "not a perfect matching: {m:?}"
                );
                assert!(is_stable(&prefs_a, &prefs_b, &m), "unstable: {m:?}");

                // A-optimal: no other stable matching gives any proposer a
                // partner they prefer.
                for other in permutations_iter(&(0..n).collect::<Vec<_>>()) {
                    if !is_stable(&prefs_a, &prefs_b, &other) {
                        continue;
                    }
                    for i in 0..n {
                        let rank_got = prefs_a[i].iter().position(|&x| x == m[i]).unwrap();
                        let rank_other = prefs_a[i].iter().position(|&x| x == other[i]).unwrap();
                        assert!(
                            rank_got <= rank_other,
                            "proposer {i} could do better in {other:?}"
                        );
                    }
                }
            }
        }
    }

    /// A matching is stable when no pair would both rather have each other.
    fn is_stable(prefs_a: &[Vec<usize>], prefs_b: &[Vec<usize>], m: &[usize]) -> bool {
        let n = m.len();
        // partner_b[j] is the A-side member matched to j.
        let mut partner_b = vec![0usize; n];
        for (i, &j) in m.iter().enumerate() {
            partner_b[j] = i;
        }
        let rank = |p: &[usize], x: usize| p.iter().position(|&y| y == x).unwrap();
        for i in 0..n {
            for j in 0..n {
                if j == m[i] {
                    continue;
                }
                let i_prefers = rank(&prefs_a[i], j) < rank(&prefs_a[i], m[i]);
                let j_prefers = rank(&prefs_b[j], i) < rank(&prefs_b[j], partner_b[j]);
                if i_prefers && j_prefers {
                    return false;
                }
            }
        }
        true
    }

    /// Stable roommates: when a matching is returned it must be stable, and
    /// when none is returned none must exist.
    #[test]
    fn stable_roommates_agrees_with_exhaustive_search() {
        let mut rng = Rng::new(0x_2001);
        for n in [2usize, 4, 6] {
            for _ in 0..40 {
                // Each person ranks the others in a random order.
                let prefs: Vec<Vec<usize>> = (0..n)
                    .map(|i| {
                        let others: Vec<usize> = (0..n).filter(|&x| x != i).collect();
                        let perm = crate::discrete::combinatorics::random_permutation(
                            others.len(),
                            &mut rng,
                        );
                        perm.into_iter().map(|k| others[k]).collect()
                    })
                    .collect();
                let brute = brute_roommates(&prefs);
                match stable_roommates(&prefs) {
                    Some(m) => {
                        assert!(m.iter().enumerate().all(|(i, &j)| m[j] == i), "not a pairing");
                        assert!(roommates_stable(&prefs, &m), "returned an unstable pairing");
                        assert!(brute.is_some(), "found one where exhaustive search found none");
                    }
                    None => assert!(
                        brute.is_none(),
                        "reported none but {brute:?} is stable"
                    ),
                }
            }
        }
        // The classic four-person instance with no stable matching: everyone
        // ranks the same person last, creating a rotation that never settles.
        let none = vec![
            vec![1, 2, 3],
            vec![2, 0, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ];
        assert!(brute_roommates(&none).is_none(), "the instance must be unsolvable");
        assert!(stable_roommates(&none).is_none());
        // Two people have exactly one, trivially stable, pairing.
        assert_eq!(stable_roommates(&[vec![1], vec![0]]), Some(vec![1, 0]));
    }

    /// Any stable roommates pairing, by trying every perfect matching.
    fn brute_roommates(prefs: &[Vec<usize>]) -> Option<Vec<usize>> {
        let n = prefs.len();
        for perm in permutations_iter(&(0..n).collect::<Vec<_>>()) {
            if (0..n).any(|i| perm[perm[i]] != i || perm[i] == i) {
                continue;
            }
            if roommates_stable(prefs, &perm) {
                return Some(perm);
            }
        }
        None
    }

    /// A pairing is stable when no two people would both rather swap.
    fn roommates_stable(prefs: &[Vec<usize>], m: &[usize]) -> bool {
        let n = m.len();
        let rank = |i: usize, x: usize| prefs[i].iter().position(|&y| y == x).unwrap();
        for i in 0..n {
            for j in 0..n {
                if i == j || m[i] == j {
                    continue;
                }
                if rank(i, j) < rank(i, m[i]) && rank(j, i) < rank(j, m[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// The maximum-weight bipartite matching must beat every other matching,
    /// and must leave a vertex unmatched when that pays better.
    #[test]
    fn maximum_weight_bipartite_beats_every_alternative() {
        let mut rng = Rng::new(0x_1471);
        for rows in 1..=5usize {
            for cols in 1..=5usize {
                for _ in 0..12 {
                    let mut w = Matrix::zeros(rows, cols);
                    for i in 0..rows {
                        for j in 0..cols {
                            // Mostly positive with some zeros, so leaving a
                            // vertex unmatched is sometimes right.
                            let v = (12.0 * rng.next_f64() - 4.0).round();
                            w.set(i, j, v.max(0.0));
                        }
                    }
                    let (total, partner) = maximum_weight_bipartite(&w);
                    // Valid: no column used twice, every pair positive.
                    let mut seen = std::collections::BTreeSet::new();
                    let mut actual = 0.0;
                    for (i, &p) in partner.iter().enumerate() {
                        if let Some(j) = p {
                            assert!(seen.insert(j), "column {j} used twice");
                            assert!(w.get(i, j) > 0.0, "matched a zero-weight pair");
                            actual += w.get(i, j);
                        }
                    }
                    assert!(close(total, actual), "reported {total} but sums to {actual}");
                    // Maximal, by brute force over injections.
                    let best = brute_weighted(&w);
                    assert!(close(total, best), "{rows}x{cols}: {total} vs brute {best}");
                }
            }
        }
        // All-zero weights: nothing worth matching.
        let (t, p) = maximum_weight_bipartite(&Matrix::zeros(3, 3));
        assert!(close(t, 0.0));
        assert!(p.iter().all(Option::is_none));
        // A single cell: taken when positive, declined when not.
        assert_eq!(
            maximum_weight_bipartite(&Matrix { rows: 1, cols: 1, data: vec![3.0] }),
            (3.0, vec![Some(0)])
        );
        assert_eq!(
            maximum_weight_bipartite(&Matrix { rows: 1, cols: 1, data: vec![0.0] }),
            (0.0, vec![None])
        );
    }

    /// The best total weight, over every partial injection of rows to columns.
    fn brute_weighted(w: &Matrix) -> f64 {
        let (rows, cols) = (w.rows, w.cols);
        let mut best = 0.0f64;
        // Choose which rows to match, then how.
        for k in 0..=rows.min(cols) {
            for row_set in combinations_iter(rows, k) {
                for col_set in combinations_iter(cols, k) {
                    for perm in permutations_iter(&(0..k).collect::<Vec<_>>()) {
                        let total: f64 = (0..k)
                            .map(|idx| w.get(row_set[idx], col_set[perm[idx]]))
                            .sum();
                        best = best.max(total);
                    }
                }
            }
        }
        best
    }

    #[test]
    fn matching_size_counts_pairs() {
        assert_eq!(matching_size(&[None, None]), 0);
        assert_eq!(matching_size(&[Some(1), Some(0)]), 1);
        assert_eq!(matching_size(&[Some(1), Some(0), Some(3), Some(2)]), 2);
        assert_eq!(matching_size(&[Some(1), Some(0), None, None]), 1);
        // Agrees with the flow-based count on a complete bipartite graph.
        let g = complete_bipartite(3, 4);
        let left: Vec<usize> = (0..3).collect();
        assert_eq!(matching_size(&max_bipartite_matching_via_flow(&g, &left)), 3);
    }
}
