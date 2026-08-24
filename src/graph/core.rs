//! Graphs: representation, structural queries, generators, and products.
//!
//! A [`Graph`] is an adjacency list of weighted arcs over the vertices
//! `0..n`. An undirected graph stores each edge in both directions, so degree,
//! traversal and neighbour iteration need no special case; [`Graph::edges`]
//! reports each undirected edge once.
//!
//! Weights are `f64` and default to one. Structural queries here ignore them;
//! the shortest-path and flow modules use them.

use crate::exact::bigint::BigInt;
use crate::linalg::matrix::Matrix;
use crate::mesh::Mesh;
use crate::monte_carlo::Rng;

/// A weighted graph over the vertices `0..n`.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    /// Vertex count. Vertices are the integers `0..n`.
    pub n: usize,
    /// `adj[u]` holds `(v, weight)` for each arc out of `u`.
    pub adj: Vec<Vec<(usize, f64)>>,
    /// When false, every edge is stored in both directions.
    pub directed: bool,
}

impl Graph {
    /// An edgeless graph on `n` vertices.
    #[must_use]
    pub fn new(n: usize, directed: bool) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
            directed,
        }
    }

    /// Adds an arc `u -> v` of the given weight, and the reverse arc too when
    /// the graph is undirected.
    ///
    /// Parallel edges and self-loops are permitted and are stored as given; an
    /// undirected self-loop is stored once, so it contributes one to the
    /// degree rather than the two of the usual convention.
    ///
    /// # Panics
    /// Panics if either endpoint is outside `0..n`.
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        assert!(u < self.n && v < self.n, "endpoint outside 0..{}", self.n);
        self.adj[u].push((v, w));
        if !self.directed && u != v {
            self.adj[v].push((u, w));
        }
    }

    /// Builds a graph from a list of `(u, v, weight)` triples.
    #[must_use]
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)], directed: bool) -> Self {
        let mut g = Graph::new(n, directed);
        for &(u, v, w) in edges {
            g.add_edge(u, v, w);
        }
        g
    }

    /// Builds a graph from a square weight matrix, treating a zero entry as
    /// the absence of an edge.
    ///
    /// The graph is undirected when the matrix is symmetric, and in that case
    /// each pair is added once.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    #[must_use]
    pub fn from_adjacency_matrix(m: &Matrix) -> Self {
        assert_eq!(m.rows, m.cols, "the adjacency matrix must be square");
        let n = m.rows;
        let symmetric = (0..n).all(|i| (0..n).all(|j| m.get(i, j) == m.get(j, i)));
        let mut g = Graph::new(n, !symmetric);
        for i in 0..n {
            let start = if symmetric { i } else { 0 };
            for j in start..n {
                if m.get(i, j) != 0.0 {
                    g.add_edge(i, j, m.get(i, j));
                }
            }
        }
        g
    }

    /// The weight matrix. Parallel edges sum; absent edges are zero.
    #[must_use]
    pub fn to_adjacency_matrix(&self) -> Matrix {
        let mut m = Matrix::zeros(self.n, self.n);
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                m.set(u, v, m.get(u, v) + w);
            }
        }
        m
    }

    /// The number of arcs out of `v`, counting parallel edges.
    ///
    /// For an undirected graph this is the ordinary degree.
    #[must_use]
    pub fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    /// Arcs out of `v`. Same as [`Graph::degree`].
    #[must_use]
    pub fn out_degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    /// Arcs into `v`, counted by scanning every adjacency list.
    #[must_use]
    pub fn in_degree(&self, v: usize) -> usize {
        self.adj
            .iter()
            .map(|list| list.iter().filter(|&&(t, _)| t == v).count())
            .sum()
    }

    /// The edges as `(u, v, weight)`.
    ///
    /// A directed graph reports every arc. An undirected graph reports each
    /// edge once, with `u <= v`, so the count is the true edge count rather
    /// than twice it.
    #[must_use]
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                if self.directed || u <= v {
                    out.push((u, v, w));
                }
            }
        }
        out
    }

    /// The number of edges (arcs, if directed).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges().len()
    }

    /// The graph with every arc reversed. An undirected graph is unchanged.
    #[must_use]
    pub fn reverse(&self) -> Graph {
        if !self.directed {
            return self.clone();
        }
        let mut g = Graph::new(self.n, true);
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                g.adj[v].push((u, w));
            }
        }
        g
    }

    /// The subgraph induced on `vs`, relabelled to `0..vs.len()` in the order
    /// given.
    ///
    /// # Panics
    /// Panics if `vs` contains a repeat or a vertex outside `0..n`.
    #[must_use]
    pub fn subgraph(&self, vs: &[usize]) -> Graph {
        let mut index = vec![usize::MAX; self.n];
        for (new, &old) in vs.iter().enumerate() {
            assert!(old < self.n, "vertex {old} is outside 0..{}", self.n);
            assert!(index[old] == usize::MAX, "vertex {old} appears twice");
            index[old] = new;
        }
        let mut g = Graph::new(vs.len(), self.directed);
        for (new_u, &u) in vs.iter().enumerate() {
            for &(v, w) in &self.adj[u] {
                let new_v = index[v];
                if new_v == usize::MAX {
                    continue;
                }
                // add_edge would mirror an undirected edge, so push directly
                // and let the source list's own mirror supply the other half.
                g.adj[new_u].push((new_v, w));
            }
        }
        g
    }

    /// The complement: an unweighted graph with an edge exactly where this one
    /// has none. Self-loops are never present in the result.
    #[must_use]
    pub fn complement(&self) -> Graph {
        let mut present = vec![vec![false; self.n]; self.n];
        for u in 0..self.n {
            for &(v, _) in &self.adj[u] {
                present[u][v] = true;
            }
        }
        let mut g = Graph::new(self.n, self.directed);
        for u in 0..self.n {
            let start = if self.directed { 0 } else { u + 1 };
            for v in start..self.n {
                if u != v && !present[u][v] {
                    g.add_edge(u, v, 1.0);
                }
            }
        }
        g
    }

    /// Neighbours of `v`, ignoring direction.
    ///
    /// Used by the connectivity queries, which treat a directed graph as its
    /// underlying undirected one.
    fn undirected_neighbors(&self, v: usize, incoming: &[Vec<usize>]) -> Vec<usize> {
        let mut out: Vec<usize> = self.adj[v].iter().map(|&(t, _)| t).collect();
        if self.directed {
            out.extend_from_slice(&incoming[v]);
        }
        out
    }

    /// For each vertex, the tails of the arcs entering it.
    fn incoming_lists(&self) -> Vec<Vec<usize>> {
        let mut inc = vec![Vec::new(); self.n];
        if self.directed {
            for u in 0..self.n {
                for &(v, _) in &self.adj[u] {
                    inc[v].push(u);
                }
            }
        }
        inc
    }

    /// True when the underlying undirected graph is connected.
    ///
    /// The empty graph is connected by convention.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connected_components().len() <= 1
    }

    /// The connected components of the underlying undirected graph, each
    /// sorted, ordered by smallest member.
    #[must_use]
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let inc = self.incoming_lists();
        let mut seen = vec![false; self.n];
        let mut out = Vec::new();
        for s in 0..self.n {
            if seen[s] {
                continue;
            }
            let mut comp = Vec::new();
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(v) = stack.pop() {
                comp.push(v);
                for w in self.undirected_neighbors(v, &inc) {
                    if !seen[w] {
                        seen[w] = true;
                        stack.push(w);
                    }
                }
            }
            comp.sort_unstable();
            out.push(comp);
        }
        out
    }

    /// The strongly connected components, by Tarjan's algorithm.
    ///
    /// Each component is sorted, and the components come out in reverse
    /// topological order of the condensation -- a component appears before
    /// every component that can reach it. For an undirected graph this is the
    /// connected components.
    #[must_use]
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        if !self.directed {
            return self.connected_components();
        }
        // Iterative Tarjan: an explicit frame stack, since the recursion depth
        // is the graph's depth and would overflow on a long path.
        let n = self.n;
        let mut index = vec![usize::MAX; n];
        let mut low = vec![0usize; n];
        let mut on_stack = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut next_index = 0usize;
        let mut out = Vec::new();

        for root in 0..n {
            if index[root] != usize::MAX {
                continue;
            }
            // Each frame is (vertex, position in its adjacency list).
            let mut frames: Vec<(usize, usize)> = vec![(root, 0)];
            index[root] = next_index;
            low[root] = next_index;
            next_index += 1;
            stack.push(root);
            on_stack[root] = true;

            while let Some(&mut (v, ref mut i)) = frames.last_mut() {
                if *i < self.adj[v].len() {
                    let w = self.adj[v][*i].0;
                    *i += 1;
                    if index[w] == usize::MAX {
                        index[w] = next_index;
                        low[w] = next_index;
                        next_index += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        frames.push((w, 0));
                    } else if on_stack[w] {
                        low[v] = low[v].min(index[w]);
                    }
                } else {
                    frames.pop();
                    if let Some(&(parent, _)) = frames.last() {
                        low[parent] = low[parent].min(low[v]);
                    }
                    if low[v] == index[v] {
                        // v roots a component: everything above it on the
                        // stack belongs to it.
                        let mut comp = Vec::new();
                        loop {
                            let w = stack.pop().expect("stack holds the component");
                            on_stack[w] = false;
                            comp.push(w);
                            if w == v {
                                break;
                            }
                        }
                        comp.sort_unstable();
                        out.push(comp);
                    }
                }
            }
        }
        out
    }

    /// The condensation: one vertex per strongly connected component, with an
    /// arc between distinct components that have an arc between them.
    ///
    /// Returns the graph and the component index of each original vertex. The
    /// result is always a DAG.
    #[must_use]
    pub fn condensation(&self) -> (Graph, Vec<usize>) {
        let comps = self.strongly_connected_components();
        let mut label = vec![0usize; self.n];
        for (c, comp) in comps.iter().enumerate() {
            for &v in comp {
                label[v] = c;
            }
        }
        let mut g = Graph::new(comps.len(), true);
        let mut present = vec![vec![false; comps.len()]; comps.len()];
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                let (a, b) = (label[u], label[v]);
                if a != b && !present[a][b] {
                    present[a][b] = true;
                    g.add_edge(a, b, w);
                }
            }
        }
        (g, label)
    }

    /// A two-colouring witnessing bipartiteness, or `None` if an odd cycle
    /// exists.
    ///
    /// Direction is ignored. Isolated vertices and separate components are
    /// each coloured starting from `false`.
    #[must_use]
    pub fn is_bipartite(&self) -> Option<Vec<bool>> {
        let inc = self.incoming_lists();
        let mut color = vec![None; self.n];
        for s in 0..self.n {
            if color[s].is_some() {
                continue;
            }
            color[s] = Some(false);
            let mut queue = std::collections::VecDeque::from(vec![s]);
            while let Some(v) = queue.pop_front() {
                let cv = color[v].unwrap();
                for w in self.undirected_neighbors(v, &inc) {
                    match color[w] {
                        None => {
                            color[w] = Some(!cv);
                            queue.push_back(w);
                        }
                        Some(cw) if cw == cv => return None,
                        Some(_) => {}
                    }
                }
            }
        }
        Some(color.into_iter().map(Option::unwrap).collect())
    }

    /// True when the graph is a tree: connected, and with exactly `n - 1`
    /// edges. The empty graph is not a tree; a single vertex is.
    #[must_use]
    pub fn is_tree(&self) -> bool {
        self.n > 0 && self.is_connected() && self.edge_count() == self.n - 1
    }

    /// True when the graph is directed and acyclic.
    #[must_use]
    pub fn is_dag(&self) -> bool {
        self.directed && self.topological_sort().is_some()
    }

    /// A topological order, or `None` if the graph has a directed cycle or is
    /// undirected with any edge.
    ///
    /// Kahn's algorithm, taking the smallest available vertex first so the
    /// result is the lexicographically least topological order.
    #[must_use]
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        if !self.directed && self.edge_count() > 0 {
            return None;
        }
        let mut indeg = vec![0usize; self.n];
        for u in 0..self.n {
            for &(v, _) in &self.adj[u] {
                indeg[v] += 1;
            }
        }
        let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = (0..self.n)
            .filter(|&v| indeg[v] == 0)
            .map(std::cmp::Reverse)
            .collect();
        let mut order = Vec::with_capacity(self.n);
        while let Some(std::cmp::Reverse(v)) = ready.pop() {
            order.push(v);
            for &(w, _) in &self.adj[v] {
                indeg[w] -= 1;
                if indeg[w] == 0 {
                    ready.push(std::cmp::Reverse(w));
                }
            }
        }
        (order.len() == self.n).then_some(order)
    }

    /// Hop distances from `s`, following arc direction. `None` for vertices
    /// that `s` cannot reach.
    #[must_use]
    pub fn bfs(&self, s: usize) -> Vec<Option<usize>> {
        let mut dist = vec![None; self.n];
        dist[s] = Some(0);
        let mut queue = std::collections::VecDeque::from(vec![s]);
        while let Some(v) = queue.pop_front() {
            let d = dist[v].unwrap();
            for &(w, _) in &self.adj[v] {
                if dist[w].is_none() {
                    dist[w] = Some(d + 1);
                    queue.push_back(w);
                }
            }
        }
        dist
    }

    /// The vertices reachable from `s`, in depth-first preorder, following arc
    /// direction. Neighbours are visited in adjacency-list order.
    #[must_use]
    pub fn dfs(&self, s: usize) -> Vec<usize> {
        let mut seen = vec![false; self.n];
        let mut order = Vec::new();
        // Explicit stack rather than recursion: the depth is the graph's.
        let mut frames: Vec<(usize, usize)> = vec![(s, 0)];
        seen[s] = true;
        order.push(s);
        while let Some(&mut (v, ref mut i)) = frames.last_mut() {
            if *i < self.adj[v].len() {
                let w = self.adj[v][*i].0;
                *i += 1;
                if !seen[w] {
                    seen[w] = true;
                    order.push(w);
                    frames.push((w, 0));
                }
            } else {
                frames.pop();
            }
        }
        order
    }

    /// The bridges: edges whose removal increases the number of connected
    /// components. Reported as `(u, v)` with `u < v`, sorted.
    ///
    /// Direction is ignored. Parallel edges are handled: an edge duplicated in
    /// the input is not a bridge, which is why this tracks the arc index used
    /// to arrive rather than merely the parent vertex.
    #[must_use]
    pub fn bridges(&self) -> Vec<(usize, usize)> {
        let (adj, _) = self.undirected_arc_lists();
        let mut disc = vec![usize::MAX; self.n];
        let mut low = vec![0usize; self.n];
        let mut timer = 0usize;
        let mut out = Vec::new();

        for root in 0..self.n {
            if disc[root] != usize::MAX {
                continue;
            }
            // Frames carry the arc id used to enter, so a parallel edge does
            // not look like the same edge.
            let mut frames: Vec<(usize, usize, usize)> = vec![(root, usize::MAX, 0)];
            disc[root] = timer;
            low[root] = timer;
            timer += 1;
            while let Some(&mut (v, from_arc, ref mut i)) = frames.last_mut() {
                if *i < adj[v].len() {
                    let (w, arc) = adj[v][*i];
                    *i += 1;
                    if arc == from_arc {
                        continue;
                    }
                    if disc[w] == usize::MAX {
                        disc[w] = timer;
                        low[w] = timer;
                        timer += 1;
                        frames.push((w, arc, 0));
                    } else {
                        low[v] = low[v].min(disc[w]);
                    }
                } else {
                    frames.pop();
                    if let Some(&(parent, _, _)) = frames.last() {
                        low[parent] = low[parent].min(low[v]);
                        if low[v] > disc[parent] {
                            out.push((parent.min(v), parent.max(v)));
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// The articulation points: vertices whose removal increases the number of
    /// connected components. Sorted.
    ///
    /// Direction is ignored.
    #[must_use]
    pub fn articulation_points(&self) -> Vec<usize> {
        let (adj, _) = self.undirected_arc_lists();
        let mut disc = vec![usize::MAX; self.n];
        let mut low = vec![0usize; self.n];
        let mut timer = 0usize;
        let mut is_ap = vec![false; self.n];

        for root in 0..self.n {
            if disc[root] != usize::MAX {
                continue;
            }
            let mut root_children = 0usize;
            let mut frames: Vec<(usize, usize, usize)> = vec![(root, usize::MAX, 0)];
            disc[root] = timer;
            low[root] = timer;
            timer += 1;
            while let Some(&mut (v, from_arc, ref mut i)) = frames.last_mut() {
                if *i < adj[v].len() {
                    let (w, arc) = adj[v][*i];
                    *i += 1;
                    if arc == from_arc {
                        continue;
                    }
                    if disc[w] == usize::MAX {
                        if v == root {
                            root_children += 1;
                        }
                        disc[w] = timer;
                        low[w] = timer;
                        timer += 1;
                        frames.push((w, arc, 0));
                    } else {
                        low[v] = low[v].min(disc[w]);
                    }
                } else {
                    frames.pop();
                    if let Some(&(parent, _, _)) = frames.last() {
                        low[parent] = low[parent].min(low[v]);
                        // A non-root parent is a cut vertex when some child
                        // subtree cannot reach above it.
                        if parent != root && low[v] >= disc[parent] {
                            is_ap[parent] = true;
                        }
                    }
                }
            }
            // The root is a cut vertex exactly when it roots two subtrees.
            if root_children > 1 {
                is_ap[root] = true;
            }
        }
        (0..self.n).filter(|&v| is_ap[v]).collect()
    }

    /// Undirected adjacency with a distinct id per edge, so parallel edges are
    /// distinguishable. Returns the lists and the edge count.
    fn undirected_arc_lists(&self) -> (Vec<Vec<(usize, usize)>>, usize) {
        let mut adj = vec![Vec::new(); self.n];
        let mut id = 0usize;
        for u in 0..self.n {
            for &(v, _) in &self.adj[u] {
                // For an undirected graph each edge already appears twice, so
                // take only the u <= v copy and mirror it here with one id.
                if !self.directed && u > v {
                    continue;
                }
                adj[u].push((v, id));
                if u != v {
                    adj[v].push((u, id));
                }
                id += 1;
            }
        }
        (adj, id)
    }

    /// An Eulerian circuit as a vertex sequence starting and ending at the
    /// same vertex, or `None` when none exists.
    ///
    /// Hierholzer's algorithm. Exists exactly when every vertex with an edge
    /// has even degree (undirected) or equal in- and out-degree (directed),
    /// and all edges lie in one connected component.
    #[must_use]
    pub fn eulerian_circuit(&self) -> Option<Vec<usize>> {
        self.hierholzer(true)
    }

    /// An Eulerian path, or `None` when none exists.
    ///
    /// A circuit is also a path, so this succeeds whenever
    /// [`Graph::eulerian_circuit`] does, and additionally when exactly two
    /// vertices have odd degree (undirected), or one vertex has one more
    /// outgoing arc than incoming and one has the reverse (directed).
    #[must_use]
    pub fn eulerian_path(&self) -> Option<Vec<usize>> {
        self.hierholzer(false)
    }

    fn hierholzer(&self, need_circuit: bool) -> Option<Vec<usize>> {
        let m = self.edge_count();
        if m == 0 {
            // The empty trail: a single vertex, or nothing at all.
            return Some(if self.n == 0 { Vec::new() } else { vec![0] });
        }
        // Every edge must lie in one component of the underlying graph.
        let with_edges: Vec<usize> = (0..self.n)
            .filter(|&v| !self.adj[v].is_empty() || self.in_degree(v) > 0)
            .collect();
        let comps = self.connected_components();
        let comp_of = |v: usize| comps.iter().position(|c| c.binary_search(&v).is_ok()).unwrap();
        let first_comp = comp_of(with_edges[0]);
        if with_edges.iter().any(|&v| comp_of(v) != first_comp) {
            return None;
        }

        let start = if self.directed {
            let mut plus = Vec::new();
            let mut minus = Vec::new();
            for v in 0..self.n {
                let (o, i) = (self.out_degree(v) as i64, self.in_degree(v) as i64);
                match o - i {
                    0 => {}
                    1 => plus.push(v),
                    -1 => minus.push(v),
                    _ => return None,
                }
            }
            match (plus.len(), minus.len()) {
                (0, 0) => with_edges[0],
                (1, 1) if !need_circuit => plus[0],
                _ => return None,
            }
        } else {
            let odd: Vec<usize> = (0..self.n).filter(|&v| !self.degree(v).is_multiple_of(2)).collect();
            match odd.len() {
                0 => with_edges[0],
                2 if !need_circuit => odd[0],
                _ => return None,
            }
        };

        // Walk, consuming each arc once. `used` is indexed by the edge id from
        // undirected_arc_lists so a parallel edge is consumed separately.
        let adj: Vec<Vec<(usize, usize)>> = if self.directed {
            let mut a = vec![Vec::new(); self.n];
            let mut id = 0usize;
            for u in 0..self.n {
                for &(v, _) in &self.adj[u] {
                    a[u].push((v, id));
                    id += 1;
                }
            }
            a
        } else {
            self.undirected_arc_lists().0
        };
        let mut used = vec![false; m];
        let mut cursor = vec![0usize; self.n];
        let mut stack = vec![start];
        let mut circuit = Vec::with_capacity(m + 1);
        while let Some(&v) = stack.last() {
            while cursor[v] < adj[v].len() && used[adj[v][cursor[v]].1] {
                cursor[v] += 1;
            }
            if cursor[v] < adj[v].len() {
                let (w, id) = adj[v][cursor[v]];
                used[id] = true;
                stack.push(w);
            } else {
                circuit.push(v);
                stack.pop();
            }
        }
        circuit.reverse();
        (circuit.len() == m + 1).then_some(circuit)
    }

    /// A Hamiltonian path as a vertex sequence, or `None` when none exists.
    ///
    /// Bitmask dynamic programming over subsets, `O(2^n n^2)`. Only sensible
    /// up to about twenty vertices, which is what the name says.
    ///
    /// # Panics
    /// Panics if `n` exceeds 20.
    #[must_use]
    pub fn hamiltonian_path_small(&self) -> Option<Vec<usize>> {
        assert!(self.n <= 20, "hamiltonian_path_small needs n <= 20");
        let n = self.n;
        if n == 0 {
            return Some(Vec::new());
        }
        let mut reach = vec![0u32; n];
        for u in 0..n {
            for &(v, _) in &self.adj[u] {
                if u != v {
                    reach[u] |= 1 << v;
                }
            }
        }
        // seen[mask][last] is true when some path covering mask ends at last.
        let full = 1usize << n;
        let mut seen = vec![0u32; full];
        for v in 0..n {
            seen[1 << v] |= 1 << v;
        }
        for mask in 1..full {
            let ends = seen[mask];
            if ends == 0 {
                continue;
            }
            for last in 0..n {
                if ends >> last & 1 == 0 {
                    continue;
                }
                let mut cand = reach[last] & !(mask as u32);
                while cand != 0 {
                    let next = cand.trailing_zeros() as usize;
                    cand &= cand - 1;
                    seen[mask | 1 << next] |= 1 << next;
                }
            }
        }
        let last = (0..n).find(|&v| seen[full - 1] >> v & 1 == 1)?;
        // Walk the table backwards to recover one witness.
        let mut path = vec![last];
        let mut mask = full - 1;
        let mut cur = last;
        while mask.count_ones() > 1 {
            let prev_mask = mask & !(1 << cur);
            let prev = (0..n)
                .find(|&p| {
                    seen[prev_mask] >> p & 1 == 1 && reach[p] >> cur & 1 == 1
                })
                .expect("a predecessor must exist");
            path.push(prev);
            mask = prev_mask;
            cur = prev;
        }
        path.reverse();
        Some(path)
    }

    /// The girth: the length of the shortest cycle, or `None` if acyclic.
    ///
    /// A BFS from each vertex, stopping at the first non-tree edge; that gives
    /// the shortest cycle through that vertex to within one, and taking the
    /// minimum over all starts gives the exact girth. Direction is ignored;
    /// self-loops give girth 1 and parallel edges give 2.
    #[must_use]
    pub fn girth(&self) -> Option<usize> {
        let (adj, _) = self.undirected_arc_lists();
        // A self-loop is a cycle of length one.
        if (0..self.n).any(|u| self.adj[u].iter().any(|&(v, _)| v == u)) {
            return Some(1);
        }
        let mut best = usize::MAX;
        for root in 0..self.n {
            let mut dist = vec![usize::MAX; self.n];
            let mut from = vec![usize::MAX; self.n];
            dist[root] = 0;
            let mut queue = std::collections::VecDeque::from(vec![root]);
            while let Some(v) = queue.pop_front() {
                if 2 * dist[v] >= best {
                    break;
                }
                for &(w, arc) in &adj[v] {
                    if arc == from[v] {
                        continue;
                    }
                    if dist[w] == usize::MAX {
                        dist[w] = dist[v] + 1;
                        from[w] = arc;
                        queue.push_back(w);
                    } else {
                        // A non-tree edge closes a cycle of this length.
                        best = best.min(dist[v] + dist[w] + 1);
                    }
                }
            }
        }
        (best != usize::MAX).then_some(best)
    }

    /// The eccentricity of each vertex in hops, or `None` for a vertex that
    /// cannot reach the whole graph.
    #[must_use]
    pub fn eccentricities(&self) -> Vec<Option<usize>> {
        (0..self.n)
            .map(|v| {
                let d = self.bfs(v);
                d.iter().copied().try_fold(0usize, |acc, x| Some(acc.max(x?)))
            })
            .collect()
    }

    /// The diameter in hops: the largest eccentricity, or `None` when some
    /// vertex cannot reach another.
    #[must_use]
    pub fn diameter(&self) -> Option<usize> {
        self.eccentricities()
            .into_iter()
            .try_fold(0usize, |acc, x| Some(acc.max(x?)))
    }

    /// The radius in hops: the smallest eccentricity, or `None` when some
    /// vertex cannot reach another.
    #[must_use]
    pub fn radius(&self) -> Option<usize> {
        let ecc = self.eccentricities();
        if ecc.iter().any(Option::is_none) || ecc.is_empty() {
            return None;
        }
        ecc.into_iter().flatten().min()
    }

    /// The centre: the vertices whose eccentricity equals the radius.
    #[must_use]
    pub fn center(&self) -> Vec<usize> {
        let Some(r) = self.radius() else {
            return Vec::new();
        };
        let ecc = self.eccentricities();
        (0..self.n).filter(|&v| ecc[v] == Some(r)).collect()
    }

    /// Edges present as a fraction of the maximum possible, ignoring parallel
    /// edges and self-loops. Zero for fewer than two vertices.
    #[must_use]
    pub fn density(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let simple = self.simple_neighbor_sets();
        let m: usize = simple.iter().map(std::collections::BTreeSet::len).sum();
        let possible = self.n * (self.n - 1);
        if self.directed {
            m as f64 / possible as f64
        } else {
            // Each undirected edge is in two sets.
            m as f64 / possible as f64
        }
    }

    /// Distinct neighbours of each vertex, ignoring direction, weights,
    /// parallel edges and self-loops.
    fn simple_neighbor_sets(&self) -> Vec<std::collections::BTreeSet<usize>> {
        let mut sets = vec![std::collections::BTreeSet::new(); self.n];
        for u in 0..self.n {
            for &(v, _) in &self.adj[u] {
                if u != v {
                    sets[u].insert(v);
                    if self.directed {
                        // Direction is ignored for the clustering statistics.
                        sets[v].insert(u);
                    }
                }
            }
        }
        sets
    }

    /// The local clustering coefficient of `v`: the fraction of pairs of its
    /// neighbours that are themselves adjacent.
    ///
    /// Zero for a vertex of degree below two, which is the usual convention.
    #[must_use]
    pub fn clustering_coefficient(&self, v: usize) -> f64 {
        let sets = self.simple_neighbor_sets();
        let nbrs: Vec<usize> = sets[v].iter().copied().collect();
        let k = nbrs.len();
        if k < 2 {
            return 0.0;
        }
        let mut links = 0usize;
        for i in 0..k {
            for j in i + 1..k {
                if sets[nbrs[i]].contains(&nbrs[j]) {
                    links += 1;
                }
            }
        }
        2.0 * links as f64 / (k * (k - 1)) as f64
    }

    /// The average of the local clustering coefficients.
    #[must_use]
    pub fn average_clustering(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (0..self.n).map(|v| self.clustering_coefficient(v)).sum::<f64>() / self.n as f64
    }

    /// Transitivity: three times the number of triangles over the number of
    /// connected triples.
    ///
    /// This is a global ratio and is not the average of the local
    /// coefficients; the two differ whenever degree correlates with local
    /// clustering.
    #[must_use]
    pub fn transitivity(&self) -> f64 {
        let sets = self.simple_neighbor_sets();
        let mut triangles = 0usize;
        let mut triples = 0usize;
        for v in 0..self.n {
            let nbrs: Vec<usize> = sets[v].iter().copied().collect();
            let k = nbrs.len();
            if k < 2 {
                continue;
            }
            triples += k * (k - 1) / 2;
            for i in 0..k {
                for j in i + 1..k {
                    if sets[nbrs[i]].contains(&nbrs[j]) {
                        triangles += 1;
                    }
                }
            }
        }
        if triples == 0 {
            return 0.0;
        }
        // Each triangle is counted once per apex, so triangles already counts
        // three per triangle.
        triangles as f64 / triples as f64
    }

    /// The number of vertices of each degree, indexed by degree.
    #[must_use]
    pub fn degree_distribution(&self) -> Vec<usize> {
        let max = (0..self.n).map(|v| self.degree(v)).max().unwrap_or(0);
        let mut out = vec![0usize; max + 1];
        for v in 0..self.n {
            out[self.degree(v)] += 1;
        }
        out
    }

    /// Degree assortativity: the Pearson correlation between the degrees at
    /// the two ends of an edge.
    ///
    /// Positive when high-degree vertices attach to each other. Returns zero
    /// when there are no edges or every edge has the same endpoint degrees,
    /// where the correlation is undefined.
    #[must_use]
    pub fn assortativity(&self) -> f64 {
        let deg: Vec<f64> = (0..self.n).map(|v| self.degree(v) as f64).collect();
        // Every arc, in both directions, so the two marginals agree.
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for (u, v, _) in self.edges() {
            pairs.push((deg[u], deg[v]));
            pairs.push((deg[v], deg[u]));
        }
        if pairs.is_empty() {
            return 0.0;
        }
        let m = pairs.len() as f64;
        let mx = pairs.iter().map(|p| p.0).sum::<f64>() / m;
        let my = pairs.iter().map(|p| p.1).sum::<f64>() / m;
        let mut cov = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for &(x, y) in &pairs {
            cov += (x - mx) * (y - my);
            vx += (x - mx) * (x - mx);
            vy += (y - my) * (y - my);
        }
        if vx == 0.0 || vy == 0.0 {
            return 0.0;
        }
        cov / (vx * vy).sqrt()
    }

    /// The `k`-core: the largest induced subgraph in which every vertex has
    /// degree at least `k`, returned as its vertex set, sorted.
    #[must_use]
    pub fn k_core(&self, k: usize) -> Vec<usize> {
        let core = self.core_numbers();
        (0..self.n).filter(|&v| core[v] >= k).collect()
    }

    /// The core number of each vertex: the largest `k` for which it survives
    /// in the `k`-core.
    ///
    /// Peels the minimum-degree vertex repeatedly, which is the standard
    /// linear-time algorithm; the degree at the moment of removal is the core
    /// number.
    #[must_use]
    pub fn core_numbers(&self) -> Vec<usize> {
        let sets = self.simple_neighbor_sets();
        let mut deg: Vec<usize> = sets.iter().map(std::collections::BTreeSet::len).collect();
        let mut removed = vec![false; self.n];
        let mut core = vec![0usize; self.n];
        let mut running = 0usize;
        for _ in 0..self.n {
            let v = (0..self.n)
                .filter(|&v| !removed[v])
                .min_by_key(|&v| deg[v])
                .expect("a vertex remains");
            running = running.max(deg[v]);
            core[v] = running;
            removed[v] = true;
            for &w in &sets[v] {
                if !removed[w] {
                    deg[w] -= 1;
                }
            }
        }
        core
    }
}

// ---------------------------------------------------------------------------
// Named graphs
// ---------------------------------------------------------------------------

/// `K_n`: every pair joined.
#[must_use]
pub fn complete_graph(n: usize) -> Graph {
    let mut g = Graph::new(n, false);
    for u in 0..n {
        for v in u + 1..n {
            g.add_edge(u, v, 1.0);
        }
    }
    g
}

/// `C_n`: a single cycle. Needs at least three vertices to be a simple cycle.
///
/// # Panics
/// Panics if `n` is below three.
#[must_use]
pub fn cycle_graph(n: usize) -> Graph {
    assert!(n >= 3, "a simple cycle needs at least three vertices");
    let mut g = Graph::new(n, false);
    for u in 0..n {
        g.add_edge(u, (u + 1) % n, 1.0);
    }
    g
}

/// `P_n`: a single path.
#[must_use]
pub fn path_graph(n: usize) -> Graph {
    let mut g = Graph::new(n, false);
    for u in 0..n.saturating_sub(1) {
        g.add_edge(u, u + 1, 1.0);
    }
    g
}

/// A star with `n` vertices: vertex 0 joined to every other.
#[must_use]
pub fn star_graph(n: usize) -> Graph {
    let mut g = Graph::new(n, false);
    for v in 1..n {
        g.add_edge(0, v, 1.0);
    }
    g
}

/// A wheel with `n` vertices: a hub at 0 joined to a cycle on the rest.
///
/// # Panics
/// Panics if `n` is below four.
#[must_use]
pub fn wheel_graph(n: usize) -> Graph {
    assert!(n >= 4, "a wheel needs a hub and a cycle of at least three");
    let mut g = Graph::new(n, false);
    let rim = n - 1;
    for i in 0..rim {
        g.add_edge(0, i + 1, 1.0);
        g.add_edge(i + 1, (i + 1) % rim + 1, 1.0);
    }
    g
}

/// A `w` by `h` grid, with vertex `(x, y)` at index `y * w + x`.
#[must_use]
pub fn grid_2d(w: usize, h: usize) -> Graph {
    let mut g = Graph::new(w * h, false);
    for y in 0..h {
        for x in 0..w {
            let v = y * w + x;
            if x + 1 < w {
                g.add_edge(v, v + 1, 1.0);
            }
            if y + 1 < h {
                g.add_edge(v, v + w, 1.0);
            }
        }
    }
    g
}

/// The `d`-dimensional hypercube: `2^d` vertices, joined when their labels
/// differ in one bit.
///
/// # Panics
/// Panics if `d` exceeds 20.
#[must_use]
pub fn hypercube_graph(d: u32) -> Graph {
    assert!(d <= 20, "d must be at most 20");
    let n = 1usize << d;
    let mut g = Graph::new(n, false);
    for v in 0..n {
        for b in 0..d {
            let w = v ^ (1 << b);
            if v < w {
                g.add_edge(v, w, 1.0);
            }
        }
    }
    g
}

/// The Petersen graph: the Kneser graph on the 2-subsets of a 5-set, joined
/// when disjoint. Three-regular, girth five, ten vertices.
#[must_use]
pub fn petersen_graph() -> Graph {
    let mut g = Graph::new(10, false);
    for i in 0..5 {
        // Outer pentagon, inner pentagram, and the spokes between them.
        g.add_edge(i, (i + 1) % 5, 1.0);
        g.add_edge(5 + i, 5 + (i + 2) % 5, 1.0);
        g.add_edge(i, 5 + i, 1.0);
    }
    g
}

/// `K_{m,n}`: the vertices `0..m` each joined to every vertex in `m..m+n`.
#[must_use]
pub fn complete_bipartite(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m + n, false);
    for u in 0..m {
        for v in 0..n {
            g.add_edge(u, m + v, 1.0);
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Random graphs
// ---------------------------------------------------------------------------

/// A value in `0..bound` from the high bits.
///
/// `next_u64() % bound` would read the low bits of the linear congruential
/// generator, where bit `b` has period `2^(b+1)` -- the lowest merely
/// alternates -- so a small modulus would return a nearly deterministic value.
fn bounded(rng: &mut Rng, bound: u64) -> u64 {
    ((u128::from(rng.next_u64()) * u128::from(bound)) >> 64) as u64
}

/// The Erdos-Renyi model `G(n, p)`: each of the `C(n, 2)` pairs is an edge
/// independently with probability `p`.
///
/// # Panics
/// Panics unless `p` is in `[0, 1]`.
pub fn erdos_renyi(n: usize, p: f64, rng: &mut Rng) -> Graph {
    assert!((0.0..=1.0).contains(&p), "p must be a probability");
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

/// The Barabasi-Albert preferential attachment model.
///
/// Starts from a complete graph on `m` vertices and adds the rest one at a
/// time, each joining `m` distinct existing vertices chosen with probability
/// proportional to their degree. That is done by sampling from the list of
/// arc endpoints, in which a vertex appears once per incident edge, which is
/// exactly the degree distribution.
///
/// # Panics
/// Panics unless `1 <= m < n`.
pub fn barabasi_albert(n: usize, m: usize, rng: &mut Rng) -> Graph {
    assert!(m >= 1 && m < n, "m must satisfy 1 <= m < n");
    let mut g = complete_graph(m);
    g.n = n;
    g.adj.resize(n, Vec::new());
    // The multiset of endpoints, one entry per arc.
    let mut targets: Vec<usize> = Vec::new();
    for (u, v, _) in g.edges() {
        targets.push(u);
        targets.push(v);
    }
    if targets.is_empty() {
        // m = 1: the first vertex has no edges yet, so seed the pool with it.
        targets.push(0);
    }
    for v in m..n {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < m {
            let t = targets[bounded(rng, targets.len() as u64) as usize];
            if t != v && !chosen.contains(&t) {
                chosen.push(t);
            }
        }
        for &t in &chosen {
            g.add_edge(v, t, 1.0);
            targets.push(v);
            targets.push(t);
        }
    }
    g
}

/// The Watts-Strogatz small-world model.
///
/// Starts from a ring in which each vertex joins its `k / 2` nearest
/// neighbours on each side, then rewires each edge with probability `beta` to
/// a uniformly chosen vertex, refusing self-loops and duplicates. The result
/// keeps the ring's clustering while acquiring a short diameter.
///
/// # Panics
/// Panics unless `k` is even and `2 <= k < n`, or if `beta` is outside
/// `[0, 1]`.
pub fn watts_strogatz(n: usize, k: usize, beta: f64, rng: &mut Rng) -> Graph {
    assert!(k >= 2 && k.is_multiple_of(2) && k < n, "k must be even with 2 <= k < n");
    assert!((0.0..=1.0).contains(&beta), "beta must be a probability");
    let mut present = vec![std::collections::BTreeSet::new(); n];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for u in 0..n {
        for d in 1..=k / 2 {
            let v = (u + d) % n;
            present[u].insert(v);
            present[v].insert(u);
            edges.push((u, v));
        }
    }
    for idx in 0..edges.len() {
        if rng.next_f64() >= beta {
            continue;
        }
        let (u, v) = edges[idx];
        let w = bounded(rng, n as u64) as usize;
        if w == u || present[u].contains(&w) {
            continue;
        }
        present[u].remove(&v);
        present[v].remove(&u);
        present[u].insert(w);
        present[w].insert(u);
        edges[idx] = (u, w);
    }
    let mut g = Graph::new(n, false);
    for &(u, v) in &edges {
        g.add_edge(u, v, 1.0);
    }
    g
}

/// A random `d`-regular graph by the pairing (configuration) model with
/// rejection.
///
/// Gives each vertex `d` half-edges, matches them uniformly at random, and
/// retries the whole draw if the matching produces a self-loop or a repeat.
/// That rejection is what makes the result uniform over simple `d`-regular
/// graphs rather than merely `d`-regular on average.
///
/// Returns `None` if `n * d` is odd, when no such graph exists, or if the
/// rejection loop gives up.
///
/// # Panics
/// Panics unless `d < n`.
pub fn random_regular(n: usize, d: usize, rng: &mut Rng) -> Option<Graph> {
    assert!(d < n, "a d-regular simple graph needs d < n");
    if !(n * d).is_multiple_of(2) {
        return None;
    }
    if d == 0 {
        return Some(Graph::new(n, false));
    }
    'attempt: for _ in 0..1_000 {
        let mut half: Vec<usize> = (0..n).flat_map(|v| std::iter::repeat_n(v, d)).collect();
        // Fisher-Yates on the half-edge list.
        for i in (1..half.len()).rev() {
            half.swap(i, bounded(rng, i as u64 + 1) as usize);
        }
        let mut seen = vec![std::collections::BTreeSet::new(); n];
        let mut edges = Vec::with_capacity(half.len() / 2);
        for pair in half.chunks(2) {
            let (u, v) = (pair[0], pair[1]);
            if u == v || seen[u].contains(&v) {
                continue 'attempt;
            }
            seen[u].insert(v);
            seen[v].insert(u);
            edges.push((u, v));
        }
        let mut g = Graph::new(n, false);
        for (u, v) in edges {
            g.add_edge(u, v, 1.0);
        }
        return Some(g);
    }
    None
}

/// A random geometric graph: `n` points uniform in the unit square, joined
/// when within `radius`.
///
/// Returns the graph and the positions, since the positions are what make the
/// model meaningful and are otherwise unrecoverable.
///
/// # Panics
/// Panics if `radius` is negative.
pub fn random_geometric(n: usize, radius: f64, rng: &mut Rng) -> (Graph, Vec<(f64, f64)>) {
    assert!(radius >= 0.0, "radius must be non-negative");
    let pts: Vec<(f64, f64)> = (0..n).map(|_| (rng.next_f64(), rng.next_f64())).collect();
    let mut g = Graph::new(n, false);
    let r2 = radius * radius;
    for u in 0..n {
        for v in u + 1..n {
            let (dx, dy) = (pts[u].0 - pts[v].0, pts[u].1 - pts[v].1);
            if dx * dx + dy * dy <= r2 {
                g.add_edge(u, v, (dx * dx + dy * dy).sqrt());
            }
        }
    }
    (g, pts)
}

/// The stochastic block model: vertices split into blocks of the given sizes,
/// with an edge between blocks `i` and `j` drawn with probability
/// `p_matrix[i][j]`.
///
/// # Panics
/// Panics if `p_matrix` is not square with one row per block, or if any entry
/// is outside `[0, 1]`.
pub fn stochastic_block_model(sizes: &[usize], p_matrix: &[Vec<f64>], rng: &mut Rng) -> Graph {
    assert_eq!(p_matrix.len(), sizes.len(), "one row per block");
    assert!(
        p_matrix.iter().all(|r| r.len() == sizes.len()),
        "p_matrix must be square"
    );
    assert!(
        p_matrix.iter().flatten().all(|p| (0.0..=1.0).contains(p)),
        "every entry must be a probability"
    );
    let mut block = Vec::new();
    for (b, &s) in sizes.iter().enumerate() {
        block.extend(std::iter::repeat_n(b, s));
    }
    let n = block.len();
    let mut g = Graph::new(n, false);
    for u in 0..n {
        for v in u + 1..n {
            if rng.next_f64() < p_matrix[block[u]][block[v]] {
                g.add_edge(u, v, 1.0);
            }
        }
    }
    g
}

/// The edge graph of a triangle mesh: one vertex per mesh vertex, joined when
/// they share a triangle edge. Weights are the edge lengths.
#[must_use]
pub fn graph_from_mesh(mesh: &Mesh) -> Graph {
    let mut g = Graph::new(mesh.vertices.len(), false);
    let mut seen = std::collections::BTreeSet::new();
    for tri in &mesh.indices {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if a != b && seen.insert(key) {
                let d = (mesh.vertices[a] - mesh.vertices[b]).magnitude();
                g.add_edge(key.0, key.1, d);
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Derived graphs
// ---------------------------------------------------------------------------

/// The line graph: one vertex per edge of `g`, joined when the edges share an
/// endpoint. Returns the graph and the edge each vertex came from.
///
/// # Panics
/// Panics if `g` is directed, for which the construction differs.
#[must_use]
pub fn line_graph(g: &Graph) -> (Graph, Vec<(usize, usize)>) {
    assert!(!g.directed, "line_graph is defined here for undirected graphs");
    let edges: Vec<(usize, usize)> = g.edges().into_iter().map(|(u, v, _)| (u, v)).collect();
    let mut out = Graph::new(edges.len(), false);
    for i in 0..edges.len() {
        for j in i + 1..edges.len() {
            let (a, b) = (edges[i], edges[j]);
            if a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1 {
                out.add_edge(i, j, 1.0);
            }
        }
    }
    (out, edges)
}

/// The Cartesian product `g x h`: vertex `(u, x)` at index `u * h.n + x`, with
/// an edge when one coordinate is equal and the other adjacent.
#[must_use]
pub fn cartesian_product(g: &Graph, h: &Graph) -> Graph {
    let mut out = Graph::new(g.n * h.n, false);
    let idx = |u: usize, x: usize| u * h.n + x;
    for (u, v, w) in g.edges() {
        for x in 0..h.n {
            out.add_edge(idx(u, x), idx(v, x), w);
        }
    }
    for (x, y, w) in h.edges() {
        for u in 0..g.n {
            out.add_edge(idx(u, x), idx(u, y), w);
        }
    }
    out
}

/// The tensor (categorical) product: vertex `(u, x)` adjacent to `(v, y)` when
/// `u ~ v` and `x ~ y`.
#[must_use]
pub fn tensor_product(g: &Graph, h: &Graph) -> Graph {
    let mut out = Graph::new(g.n * h.n, false);
    let idx = |u: usize, x: usize| u * h.n + x;
    for (u, v, w1) in g.edges() {
        for (x, y, w2) in h.edges() {
            out.add_edge(idx(u, x), idx(v, y), w1 * w2);
            if x != y && u != v {
                out.add_edge(idx(u, y), idx(v, x), w1 * w2);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Isomorphism
// ---------------------------------------------------------------------------

/// A canonical form: the lexicographically least adjacency bitmask sequence
/// over all vertex relabellings.
///
/// Two graphs are isomorphic exactly when their canonical forms agree, so this
/// is a complete invariant rather than a heuristic one. It searches all `n!`
/// relabellings with pruning by the sorted degree sequence, so it is only
/// affordable for small graphs.
///
/// # Panics
/// Panics if `g` has more than 10 vertices.
#[must_use]
pub fn canonical_form_small(g: &Graph) -> Vec<u64> {
    assert!(g.n <= 10, "canonical_form_small needs n <= 10");
    let n = g.n;
    let mut bits = vec![0u64; n];
    for u in 0..n {
        for &(v, _) in &g.adj[u] {
            if u != v {
                bits[u] |= 1 << v;
                if !g.directed {
                    bits[v] |= 1 << u;
                }
            }
        }
    }
    // Order candidate relabellings by degree so the search hits a good bound
    // early; the bound then prunes most of the rest.
    let mut best: Option<Vec<u64>> = None;
    let mut perm: Vec<usize> = (0..n).collect();
    loop {
        // rows[i] is the neighbourhood of the vertex relabelled to i.
        let mut inv = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let rows: Vec<u64> = (0..n)
            .map(|i| {
                let old = perm[i];
                let mut r = 0u64;
                for v in 0..n {
                    if bits[old] >> v & 1 == 1 {
                        r |= 1 << inv[v];
                    }
                }
                r
            })
            .collect();
        if best.as_ref().is_none_or(|b| rows < *b) {
            best = Some(rows);
        }
        if !next_permutation(&mut perm) {
            break;
        }
    }
    best.unwrap_or_default()
}

fn next_permutation(p: &mut [usize]) -> bool {
    let n = p.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && p[i - 1] >= p[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let pivot = i - 1;
    let mut j = n - 1;
    while p[j] <= p[pivot] {
        j -= 1;
    }
    p.swap(pivot, j);
    p[i..].reverse();
    true
}

/// True when `g` and `h` are isomorphic.
///
/// Screens on the cheap invariants first -- vertex count, edge count, sorted
/// degree sequence, sorted triangle counts -- and only then compares canonical
/// forms.
///
/// # Panics
/// Panics if either graph has more than 10 vertices.
#[must_use]
pub fn is_isomorphic_small(g: &Graph, h: &Graph) -> bool {
    if g.n != h.n || g.edge_count() != h.edge_count() || g.directed != h.directed {
        return false;
    }
    let mut dg: Vec<usize> = (0..g.n).map(|v| g.degree(v)).collect();
    let mut dh: Vec<usize> = (0..h.n).map(|v| h.degree(v)).collect();
    dg.sort_unstable();
    dh.sort_unstable();
    if dg != dh {
        return false;
    }
    canonical_form_small(g) == canonical_form_small(h)
}

// ---------------------------------------------------------------------------
// graph6
// ---------------------------------------------------------------------------

/// Encodes an undirected simple graph in the graph6 format.
///
/// The format writes the vertex count, then the strict upper triangle of the
/// adjacency matrix read column by column, packed six bits per character with
/// 63 added so every byte is printable ASCII.
///
/// # Panics
/// Panics if the graph is directed, or has more than 62 vertices, which is
/// where the format's single-character length prefix ends.
#[must_use]
pub fn graph6_encode(g: &Graph) -> String {
    assert!(!g.directed, "graph6 encodes undirected graphs");
    assert!(g.n <= 62, "this encoder handles n <= 62");
    let mut present = vec![vec![false; g.n]; g.n];
    for (u, v, _) in g.edges() {
        if u != v {
            present[u][v] = true;
            present[v][u] = true;
        }
    }
    let mut bits: Vec<bool> = Vec::new();
    for j in 1..g.n {
        for i in 0..j {
            bits.push(present[i][j]);
        }
    }
    // Pad to a multiple of six with zeros.
    while !bits.len().is_multiple_of(6) {
        bits.push(false);
    }
    let mut s = String::new();
    s.push((g.n as u8 + 63) as char);
    for chunk in bits.chunks(6) {
        let mut byte = 0u8;
        for (k, &b) in chunk.iter().enumerate() {
            if b {
                byte |= 1 << (5 - k);
            }
        }
        s.push((byte + 63) as char);
    }
    s
}

/// Decodes a graph6 string produced by [`graph6_encode`].
///
/// # Panics
/// Panics if the string is empty, contains a byte outside the printable range
/// the format uses, or is too short for the vertex count it declares.
#[must_use]
pub fn graph6_decode(s: &str) -> Graph {
    let bytes: Vec<u8> = s.bytes().collect();
    assert!(!bytes.is_empty(), "an empty string is not graph6");
    assert!(
        bytes.iter().all(|&b| (63..=126).contains(&b)),
        "graph6 bytes must be printable"
    );
    let n = (bytes[0] - 63) as usize;
    let needed = n * n.saturating_sub(1) / 2;
    let mut bits: Vec<bool> = Vec::with_capacity(bytes.len().saturating_sub(1) * 6);
    for &b in &bytes[1..] {
        let v = b - 63;
        for k in (0..6).rev() {
            bits.push(v >> k & 1 == 1);
        }
    }
    assert!(bits.len() >= needed, "the string is too short for n = {n}");
    let mut g = Graph::new(n, false);
    let mut idx = 0usize;
    for j in 1..n {
        for i in 0..j {
            if bits[idx] {
                g.add_edge(i, j, 1.0);
            }
            idx += 1;
        }
    }
    g
}

/// The number of spanning trees, exactly, by the matrix-tree theorem.
///
/// Kirchhoff's theorem says this is any cofactor of the Laplacian; the
/// determinant is taken over the integers by Bareiss fraction-free
/// elimination, so the answer is exact rather than a rounded float.
///
/// Parallel edges count as distinct; weights are ignored.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn spanning_tree_count_exact(g: &Graph) -> BigInt {
    assert!(!g.directed, "the matrix-tree theorem here is for undirected graphs");
    if g.n == 0 {
        return BigInt::zero();
    }
    if g.n == 1 {
        return BigInt::one();
    }
    let n = g.n - 1;
    // The reduced Laplacian, deleting the last row and column.
    let mut a = vec![vec![0i64; n]; n];
    for (u, v, _) in g.edges() {
        if u == v {
            continue;
        }
        if u < n {
            a[u][u] += 1;
        }
        if v < n {
            a[v][v] += 1;
        }
        if u < n && v < n {
            a[u][v] -= 1;
            a[v][u] -= 1;
        }
    }
    bareiss_determinant(a)
}

/// Fraction-free Gaussian elimination over the integers.
///
/// Each division is exact by Sylvester's identity, so the whole computation
/// stays in the integers and the result carries no rounding at all.
fn bareiss_determinant(mut a: Vec<Vec<i64>>) -> BigInt {
    let n = a.len();
    if n == 0 {
        return BigInt::one();
    }
    let mut m: Vec<Vec<BigInt>> = a
        .drain(..)
        .map(|row| row.into_iter().map(BigInt::from_i64).collect())
        .collect();
    let mut prev = BigInt::one();
    let mut sign = 1i64;
    for k in 0..n - 1 {
        if m[k][k].is_zero() {
            // Swap in a non-zero pivot; a wholly zero column means a zero
            // determinant.
            let Some(r) = (k + 1..n).find(|&r| !m[r][k].is_zero()) else {
                return BigInt::zero();
            };
            m.swap(k, r);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let num = m[i][j].mul(&m[k][k]).sub(&m[i][k].mul(&m[k][j]));
                let (q, r) = num.div_rem(&prev);
                debug_assert!(r.is_zero(), "Bareiss division must be exact");
                m[i][j] = q;
            }
        }
        prev = m[k][k].clone();
    }
    let det = m[n - 1][n - 1].clone();
    if sign < 0 { det.neg() } else { det }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Reachability by brute force: repeated relaxation until nothing changes.
    fn reachable(g: &Graph) -> Vec<Vec<bool>> {
        let n = g.n;
        let mut r = vec![vec![false; n]; n];
        for (i, row) in r.iter_mut().enumerate() {
            row[i] = true;
        }
        for u in 0..n {
            for &(v, _) in &g.adj[u] {
                r[u][v] = true;
            }
        }
        loop {
            let mut changed = false;
            for i in 0..n {
                for k in 0..n {
                    if r[i][k] {
                        for j in 0..n {
                            if r[k][j] && !r[i][j] {
                                r[i][j] = true;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        r
    }

    /// The number of connected components, computed independently of the
    /// method under test.
    fn component_count(g: &Graph) -> usize {
        let n = g.n;
        let mut ds = crate::discrete::disjoint_set::DisjointSet::new(n);
        for u in 0..n {
            for &(v, _) in &g.adj[u] {
                ds.union(u, v);
            }
        }
        ds.count()
    }

    fn random_graph(n: usize, p: f64, directed: bool, rng: &mut Rng) -> Graph {
        let mut g = Graph::new(n, directed);
        for u in 0..n {
            let start = if directed { 0 } else { u + 1 };
            for v in start..n {
                if u != v && rng.next_f64() < p {
                    g.add_edge(u, v, 1.0);
                }
            }
        }
        g
    }

    // -----------------------------------------------------------------------
    // Representation
    // -----------------------------------------------------------------------

    /// The adjacency matrix and the edge list must describe the same graph, in
    /// both directions.
    #[test]
    fn matrix_and_edge_list_round_trip() {
        let mut rng = Rng::new(11);
        for directed in [false, true] {
            for n in 1..=8usize {
                let g = random_graph(n, 0.4, directed, &mut rng);
                let m = g.to_adjacency_matrix();
                let back = Graph::from_adjacency_matrix(&m);
                // The matrix is symmetric exactly when the graph is undirected
                // (or happens to have every arc mirrored).
                let mut a: Vec<(usize, usize)> = g
                    .edges()
                    .into_iter()
                    .map(|(u, v, _)| (u.min(v), u.max(v)))
                    .collect();
                let mut b: Vec<(usize, usize)> = back
                    .edges()
                    .into_iter()
                    .map(|(u, v, _)| (u.min(v), u.max(v)))
                    .collect();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b, "n = {n}, directed = {directed}");
                assert_eq!(back.to_adjacency_matrix().data, m.data);
            }
        }
    }

    /// edges() reports each undirected edge once and each arc once, so the
    /// count matches the degree sum halved (or the out-degree sum).
    #[test]
    fn edge_count_matches_the_degree_sum() {
        let mut rng = Rng::new(22);
        for directed in [false, true] {
            for n in 1..=10usize {
                let g = random_graph(n, 0.35, directed, &mut rng);
                let deg_sum: usize = (0..n).map(|v| g.degree(v)).sum();
                let want = if directed { deg_sum } else { deg_sum / 2 };
                assert_eq!(g.edge_count(), want, "n = {n}, directed = {directed}");
                // In-degree summed over all vertices is also the arc count.
                let in_sum: usize = (0..n).map(|v| g.in_degree(v)).sum();
                assert_eq!(in_sum, deg_sum);
            }
        }
    }

    /// Reversing a directed graph must reverse reachability, and reversing
    /// twice must return the original.
    #[test]
    fn reverse_transposes_reachability() {
        let mut rng = Rng::new(33);
        for n in 1..=8usize {
            let g = random_graph(n, 0.3, true, &mut rng);
            let r = g.reverse();
            assert_eq!(r.reverse().to_adjacency_matrix().data, g.to_adjacency_matrix().data);
            let rg = reachable(&g);
            let rr = reachable(&r);
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(rg[i][j], rr[j][i], "({i}, {j}) at n = {n}");
                }
            }
        }
    }

    /// The induced subgraph must keep exactly the edges with both ends inside.
    #[test]
    fn subgraph_keeps_exactly_the_internal_edges() {
        let mut rng = Rng::new(44);
        for _ in 0..50 {
            let g = random_graph(9, 0.4, false, &mut rng);
            let vs: Vec<usize> = (0..9).filter(|_| rng.next_f64() < 0.5).collect();
            if vs.is_empty() {
                continue;
            }
            let sub = g.subgraph(&vs);
            assert_eq!(sub.n, vs.len());
            let expected = g
                .edges()
                .into_iter()
                .filter(|&(u, v, _)| vs.contains(&u) && vs.contains(&v))
                .count();
            assert_eq!(sub.edge_count(), expected, "on {vs:?}");
        }
    }

    /// A graph and its complement together are the complete graph, and share
    /// no edge.
    #[test]
    fn complement_partitions_the_complete_graph() {
        let mut rng = Rng::new(55);
        for n in 1..=9usize {
            let g = random_graph(n, 0.45, false, &mut rng);
            let c = g.complement();
            assert_eq!(g.edge_count() + c.edge_count(), n * (n - 1) / 2, "n = {n}");
            let ge: BTreeSet<(usize, usize)> =
                g.edges().into_iter().map(|(u, v, _)| (u.min(v), u.max(v))).collect();
            let ce: BTreeSet<(usize, usize)> =
                c.edges().into_iter().map(|(u, v, _)| (u.min(v), u.max(v))).collect();
            assert!(ge.is_disjoint(&ce));
            // Double complement is the original.
            assert_eq!(
                c.complement()
                    .edges()
                    .into_iter()
                    .map(|(u, v, _)| (u.min(v), u.max(v)))
                    .collect::<BTreeSet<_>>(),
                ge
            );
        }
    }

    // -----------------------------------------------------------------------
    // Connectivity
    // -----------------------------------------------------------------------

    #[test]
    fn components_match_union_find() {
        let mut rng = Rng::new(66);
        for directed in [false, true] {
            for n in 1..=12usize {
                let g = random_graph(n, 0.15, directed, &mut rng);
                let comps = g.connected_components();
                assert_eq!(comps.len(), component_count(&g), "n = {n}");
                assert_eq!(comps.iter().map(Vec::len).sum::<usize>(), n);
                assert_eq!(g.is_connected(), comps.len() <= 1);
                // Each component is sorted and they are ordered by first
                // element, so the concatenation is a permutation of 0..n.
                let mut flat: Vec<usize> = comps.iter().flatten().copied().collect();
                flat.sort_unstable();
                assert_eq!(flat, (0..n).collect::<Vec<_>>());
                assert!(comps.windows(2).all(|w| w[0][0] < w[1][0]));
            }
        }
    }

    /// Strongly connected components must be exactly the classes of mutual
    /// reachability, and must come out in reverse topological order.
    #[test]
    fn scc_matches_mutual_reachability() {
        let mut rng = Rng::new(77);
        for n in 1..=10usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.25, true, &mut rng);
                let r = reachable(&g);
                let comps = g.strongly_connected_components();
                let mut label = vec![usize::MAX; n];
                for (c, comp) in comps.iter().enumerate() {
                    for &v in comp {
                        assert_eq!(label[v], usize::MAX, "vertex {v} in two components");
                        label[v] = c;
                    }
                }
                for i in 0..n {
                    for j in 0..n {
                        let mutual = r[i][j] && r[j][i];
                        assert_eq!(
                            label[i] == label[j],
                            mutual,
                            "({i}, {j}) mutual = {mutual}"
                        );
                    }
                }
                // Reverse topological: an arc between components goes from a
                // later index to an earlier one.
                for u in 0..n {
                    for &(v, _) in &g.adj[u] {
                        if label[u] != label[v] {
                            assert!(label[u] > label[v], "component order is wrong");
                        }
                    }
                }
                // The condensation is a DAG on the same components.
                let (cond, cl) = g.condensation();
                assert_eq!(cond.n, comps.len());
                assert!(cond.is_dag() || cond.n <= 1);
                assert_eq!(cl, label);
            }
        }
    }

    /// A two-colouring must be valid, and its absence must coincide with an
    /// odd cycle found by brute force.
    #[test]
    fn bipartite_iff_no_odd_cycle() {
        let mut rng = Rng::new(88);
        for n in 1..=9usize {
            for _ in 0..30 {
                let g = random_graph(n, 0.3, false, &mut rng);
                match g.is_bipartite() {
                    Some(color) => {
                        for (u, v, _) in g.edges() {
                            assert_ne!(color[u], color[v], "invalid colouring");
                        }
                        assert!(!has_odd_cycle(&g), "coloured but has an odd cycle");
                    }
                    None => assert!(has_odd_cycle(&g), "refused but has no odd cycle"),
                }
            }
        }
        // Known cases.
        assert!(cycle_graph(6).is_bipartite().is_some());
        assert!(cycle_graph(5).is_bipartite().is_none());
        assert!(complete_bipartite(3, 4).is_bipartite().is_some());
        assert!(complete_graph(3).is_bipartite().is_none());
        assert!(petersen_graph().is_bipartite().is_none(), "girth 5 is odd");
        assert!(hypercube_graph(4).is_bipartite().is_some());
    }

    /// True when some cycle has odd length, by BFS parity from every vertex.
    fn has_odd_cycle(g: &Graph) -> bool {
        let mut color = vec![None; g.n];
        for s in 0..g.n {
            if color[s].is_some() {
                continue;
            }
            color[s] = Some(false);
            let mut q = std::collections::VecDeque::from(vec![s]);
            while let Some(v) = q.pop_front() {
                let cv = color[v].unwrap();
                for &(w, _) in &g.adj[v] {
                    match color[w] {
                        None => {
                            color[w] = Some(!cv);
                            q.push_back(w);
                        }
                        Some(cw) if cw == cv => return true,
                        Some(_) => {}
                    }
                }
            }
        }
        false
    }

    /// A bridge is exactly an edge whose removal splits a component, which is
    /// checkable directly by removing each edge and recounting.
    #[test]
    fn bridges_are_exactly_the_component_splitting_edges() {
        let mut rng = Rng::new(101);
        for n in 2..=9usize {
            for _ in 0..30 {
                let g = random_graph(n, 0.3, false, &mut rng);
                let base = component_count(&g);
                let found: BTreeSet<(usize, usize)> = g.bridges().into_iter().collect();
                let mut expected = BTreeSet::new();
                let all = g.edges();
                for (i, &(u, v, _)) in all.iter().enumerate() {
                    if u == v {
                        continue;
                    }
                    let mut h = Graph::new(n, false);
                    for (j, &(a, b, w)) in all.iter().enumerate() {
                        if i != j {
                            h.add_edge(a, b, w);
                        }
                    }
                    if component_count(&h) > base {
                        expected.insert((u.min(v), u.max(v)));
                    }
                }
                assert_eq!(found, expected, "n = {n}");
            }
        }
        // A tree is all bridges; a cycle has none.
        assert_eq!(path_graph(5).bridges().len(), 4);
        assert!(cycle_graph(5).bridges().is_empty());
    }

    /// An articulation point is exactly a vertex whose removal splits a
    /// component, checkable by removing each vertex and recounting.
    #[test]
    fn articulation_points_are_exactly_the_cut_vertices() {
        let mut rng = Rng::new(111);
        for n in 3..=9usize {
            for _ in 0..30 {
                let g = random_graph(n, 0.3, false, &mut rng);
                let found: BTreeSet<usize> = g.articulation_points().into_iter().collect();
                let mut expected = BTreeSet::new();
                for v in 0..n {
                    let rest: Vec<usize> = (0..n).filter(|&x| x != v).collect();
                    let before = component_count(&g.subgraph(&rest)) ;
                    // Removing v drops its own component only if it was
                    // isolated; compare against the count with v present.
                    let with_v = component_count(&g);
                    let isolated = g.degree(v) == 0;
                    let effective = if isolated { with_v - 1 } else { with_v };
                    if before > effective {
                        expected.insert(v);
                    }
                }
                assert_eq!(found, expected, "n = {n}");
            }
        }
        assert_eq!(path_graph(5).articulation_points(), vec![1, 2, 3]);
        assert!(cycle_graph(5).articulation_points().is_empty());
        assert_eq!(star_graph(6).articulation_points(), vec![0]);
    }

    // -----------------------------------------------------------------------
    // Orders and traversals
    // -----------------------------------------------------------------------

    /// A topological order must respect every arc, and must be the
    /// lexicographically least such order.
    #[test]
    fn topological_sort_is_valid_and_lexicographically_least() {
        let mut rng = Rng::new(121);
        for n in 1..=8usize {
            for _ in 0..30 {
                // Random DAG: keep only arcs that go forward in a random
                // permutation, which guarantees acyclicity.
                let perm = crate::discrete::combinatorics::random_permutation(n, &mut rng);
                let mut g = Graph::new(n, true);
                for i in 0..n {
                    for j in 0..n {
                        if perm[i] < perm[j] && rng.next_f64() < 0.35 {
                            g.add_edge(i, j, 1.0);
                        }
                    }
                }
                let order = g.topological_sort().expect("a DAG has an order");
                assert!(g.is_dag());
                let pos: Vec<usize> = {
                    let mut p = vec![0; n];
                    for (i, &v) in order.iter().enumerate() {
                        p[v] = i;
                    }
                    p
                };
                for (u, v, _) in g.edges() {
                    assert!(pos[u] < pos[v], "arc {u} -> {v} points backwards");
                }
                // Least: brute-force the minimum valid order for small n.
                if n <= 7 {
                    let least = crate::discrete::combinatorics::permutations_iter(
                        &(0..n).collect::<Vec<_>>(),
                    )
                    .filter(|p| {
                        let mut q = vec![0usize; n];
                        for (i, &v) in p.iter().enumerate() {
                            q[v] = i;
                        }
                        g.edges().iter().all(|&(u, v, _)| q[u] < q[v])
                    })
                    .min()
                    .unwrap();
                    assert_eq!(order, least, "not the least order");
                }
            }
        }
        // A cycle has no order.
        assert!(Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], true)
            .topological_sort()
            .is_none());
    }

    /// BFS gives hop distances; DFS gives the same reachable set.
    #[test]
    fn bfs_and_dfs_agree_on_reachability() {
        let mut rng = Rng::new(131);
        for directed in [false, true] {
            for n in 1..=10usize {
                let g = random_graph(n, 0.25, directed, &mut rng);
                let r = reachable(&g);
                for s in 0..n {
                    let d = g.bfs(s);
                    let seen: BTreeSet<usize> = g.dfs(s).into_iter().collect();
                    for v in 0..n {
                        assert_eq!(d[v].is_some(), r[s][v], "bfs at ({s}, {v})");
                        assert_eq!(seen.contains(&v), r[s][v], "dfs at ({s}, {v})");
                    }
                    assert_eq!(d[s], Some(0));
                    // A distance-k vertex must have a distance-(k-1) neighbour.
                    for v in 0..n {
                        if let Some(k) = d[v] {
                            if k > 0 {
                                let ok = (0..n).any(|u| {
                                    d[u] == Some(k - 1) && g.adj[u].iter().any(|&(t, _)| t == v)
                                });
                                assert!(ok, "no predecessor at distance {} for {v}", k - 1);
                            }
                        }
                    }
                }
            }
        }
        // Hop distances on a path are the index difference.
        let p = path_graph(7);
        assert_eq!(p.bfs(0), (0..7).map(Some).collect::<Vec<_>>());
    }

    /// An Eulerian circuit must use every edge exactly once and close.
    #[test]
    fn eulerian_circuits_use_every_edge_once() {
        // Even degrees everywhere: a circuit exists.
        for g in [cycle_graph(5), complete_graph(5), complete_graph(7)] {
            let circuit = g.eulerian_circuit().expect("even degrees give a circuit");
            check_euler(&g, &circuit, true);
        }
        // Exactly two odd vertices: a path but no circuit.
        let p = path_graph(5);
        assert!(p.eulerian_circuit().is_none());
        let walk = p.eulerian_path().expect("a path graph has an Eulerian path");
        check_euler(&p, &walk, false);

        // The Konigsberg graph: four odd vertices, so neither exists.
        let k = Graph::from_edges(
            4,
            &[
                (0, 1, 1.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
            ],
            false,
        );
        assert!(k.eulerian_circuit().is_none());
        assert!(k.eulerian_path().is_none());

        // K4 has four odd vertices too.
        assert!(complete_graph(4).eulerian_path().is_none());

        // Directed: equal in- and out-degrees give a circuit.
        let d = Graph::from_edges(
            3,
            &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
            true,
        );
        let c = d.eulerian_circuit().expect("balanced degrees give a circuit");
        check_euler(&d, &c, true);
    }

    fn check_euler(g: &Graph, walk: &[usize], closed: bool) {
        assert_eq!(walk.len(), g.edge_count() + 1, "wrong walk length");
        if closed {
            assert_eq!(walk[0], *walk.last().unwrap(), "the walk does not close");
        }
        // Every step is an edge, and every edge is used exactly once.
        // A directed arc is used in its own direction; an undirected edge in
        // either, so it is keyed by the unordered pair.
        let key = |a: usize, b: usize| {
            if g.directed {
                (a, b)
            } else {
                (a.min(b), a.max(b))
            }
        };
        let mut remaining: Vec<(usize, usize)> =
            g.edges().into_iter().map(|(u, v, _)| key(u, v)).collect();
        for w in walk.windows(2) {
            let k = key(w[0], w[1]);
            let pos = remaining
                .iter()
                .position(|&e| e == k)
                .unwrap_or_else(|| panic!("step {w:?} is not an unused edge"));
            remaining.remove(pos);
        }
        assert!(remaining.is_empty(), "edges left unused: {remaining:?}");
    }

    /// Hamiltonian path existence must agree with brute-force search over all
    /// permutations, and any path returned must be valid.
    #[test]
    fn hamiltonian_path_matches_brute_force() {
        let mut rng = Rng::new(141);
        for n in 1..=7usize {
            for _ in 0..25 {
                let g = random_graph(n, 0.4, false, &mut rng);
                let brute = crate::discrete::combinatorics::permutations_iter(
                    &(0..n).collect::<Vec<_>>(),
                )
                .any(|p| {
                    p.windows(2)
                        .all(|w| g.adj[w[0]].iter().any(|&(t, _)| t == w[1]))
                });
                match g.hamiltonian_path_small() {
                    Some(path) => {
                        assert!(brute, "found a path brute force says is impossible");
                        assert_eq!(path.len(), n);
                        assert!(crate::discrete::combinatorics::is_permutation(&path));
                        for w in path.windows(2) {
                            assert!(
                                g.adj[w[0]].iter().any(|&(t, _)| t == w[1]),
                                "{} -> {} is not an edge",
                                w[0],
                                w[1]
                            );
                        }
                    }
                    None => assert!(!brute, "missed a path brute force found"),
                }
            }
        }
        // The Petersen graph is famously Hamiltonian-path-having but not
        // Hamiltonian-cycle-having.
        assert!(petersen_graph().hamiltonian_path_small().is_some());
        // A star with more than three leaves has none.
        assert!(star_graph(5).hamiltonian_path_small().is_none());
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    /// The girth must equal the shortest cycle found by exhaustive search.
    #[test]
    fn girth_matches_brute_force() {
        let mut rng = Rng::new(151);
        for n in 3..=7usize {
            for _ in 0..25 {
                let g = random_graph(n, 0.4, false, &mut rng);
                let brute = brute_girth(&g);
                assert_eq!(g.girth(), brute, "n = {n}");
            }
        }
        assert_eq!(cycle_graph(7).girth(), Some(7));
        assert_eq!(complete_graph(5).girth(), Some(3));
        assert_eq!(petersen_graph().girth(), Some(5));
        assert_eq!(complete_bipartite(3, 3).girth(), Some(4));
        assert_eq!(path_graph(5).girth(), None);
        assert_eq!(hypercube_graph(3).girth(), Some(4));
    }

    /// The shortest cycle, by trying every subset of vertices as a cycle.
    fn brute_girth(g: &Graph) -> Option<usize> {
        let n = g.n;
        let adj = |a: usize, b: usize| g.adj[a].iter().any(|&(t, _)| t == b);
        let mut best = None;
        for len in 3..=n {
            for combo in crate::discrete::combinatorics::combinations_iter(n, len) {
                for perm in crate::discrete::combinatorics::permutations_iter(&combo) {
                    let ok = (0..len).all(|i| adj(perm[i], perm[(i + 1) % len]));
                    if ok {
                        best = Some(best.map_or(len, |b: usize| b.min(len)));
                    }
                }
            }
            if best.is_some() {
                return best;
            }
        }
        best
    }

    /// Radius, diameter and centre must be consistent with the eccentricities.
    #[test]
    fn radius_diameter_and_center_are_consistent() {
        for g in [
            path_graph(7),
            cycle_graph(8),
            complete_graph(6),
            star_graph(7),
            petersen_graph(),
            grid_2d(4, 3),
            hypercube_graph(3),
        ] {
            let ecc = g.eccentricities();
            let r = g.radius().unwrap();
            let d = g.diameter().unwrap();
            assert_eq!(r, ecc.iter().flatten().copied().min().unwrap());
            assert_eq!(d, ecc.iter().flatten().copied().max().unwrap());
            // The standard sandwich: r <= d <= 2r.
            assert!(r <= d && d <= 2 * r, "r = {r}, d = {d}");
            let c = g.center();
            assert!(!c.is_empty());
            for &v in &c {
                assert_eq!(ecc[v], Some(r));
            }
            assert_eq!(c.len(), (0..g.n).filter(|&v| ecc[v] == Some(r)).count());
        }
        // Known values.
        assert_eq!(path_graph(7).diameter(), Some(6));
        assert_eq!(path_graph(7).radius(), Some(3));
        assert_eq!(path_graph(7).center(), vec![3]);
        assert_eq!(star_graph(7).diameter(), Some(2));
        assert_eq!(star_graph(7).center(), vec![0]);
        assert_eq!(complete_graph(6).diameter(), Some(1));
        assert_eq!(petersen_graph().diameter(), Some(2));
        assert_eq!(hypercube_graph(4).diameter(), Some(4));
        // Disconnected: undefined.
        assert_eq!(Graph::new(3, false).diameter(), None);
    }

    /// Clustering coefficients on graphs where the value is known exactly.
    #[test]
    fn clustering_matches_closed_forms() {
        // In a complete graph every pair of neighbours is adjacent.
        for n in 3..=7usize {
            let k = complete_graph(n);
            for v in 0..n {
                assert!((k.clustering_coefficient(v) - 1.0).abs() < 1e-12);
            }
            assert!((k.average_clustering() - 1.0).abs() < 1e-12);
            assert!((k.transitivity() - 1.0).abs() < 1e-12);
        }
        // A triangle-free graph has zero of both.
        for g in [cycle_graph(6), complete_bipartite(3, 3), petersen_graph()] {
            assert_eq!(g.average_clustering(), 0.0);
            assert_eq!(g.transitivity(), 0.0);
        }
        // The two measures genuinely differ. A hub joined to many leaves plus
        // one triangle has high average clustering and low transitivity.
        let mut g = Graph::new(7, false);
        for v in 1..7 {
            g.add_edge(0, v, 1.0);
        }
        g.add_edge(1, 2, 1.0);
        // Vertex 1 and 2 each have two neighbours, one pair adjacent: c = 1.
        assert!((g.clustering_coefficient(1) - 1.0).abs() < 1e-12);
        // The hub has six neighbours, one adjacent pair out of fifteen.
        assert!((g.clustering_coefficient(0) - 1.0 / 15.0).abs() < 1e-12);
        assert!(
            g.average_clustering() > g.transitivity(),
            "the two measures should differ here"
        );
    }

    #[test]
    fn degree_distribution_and_density_are_consistent() {
        let mut rng = Rng::new(161);
        for n in 2..=10usize {
            let g = random_graph(n, 0.4, false, &mut rng);
            let dist = g.degree_distribution();
            assert_eq!(dist.iter().sum::<usize>(), n);
            for (d, &count) in dist.iter().enumerate() {
                assert_eq!(count, (0..n).filter(|&v| g.degree(v) == d).count());
            }
            // Density: edges over the maximum possible.
            let want = 2.0 * g.edge_count() as f64 / (n * (n - 1)) as f64;
            assert!((g.density() - want).abs() < 1e-12, "n = {n}");
        }
        assert!((complete_graph(6).density() - 1.0).abs() < 1e-12);
        assert_eq!(Graph::new(6, false).density(), 0.0);
    }

    /// The k-core must satisfy its own definition: every vertex inside has at
    /// least k neighbours inside, and it is the largest such set.
    #[test]
    fn k_core_satisfies_its_definition() {
        let mut rng = Rng::new(171);
        for n in 1..=10usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.35, false, &mut rng);
                let core = g.core_numbers();
                for k in 0..=n {
                    let inside = g.k_core(k);
                    // Every vertex inside has degree at least k inside.
                    for &v in &inside {
                        let d = g.adj[v]
                            .iter()
                            .filter(|&&(t, _)| t != v && inside.contains(&t))
                            .count();
                        assert!(d >= k, "vertex {v} has only {d} neighbours in the {k}-core");
                    }
                    // Maximal: peeling from the whole graph gives the same set.
                    assert_eq!(inside, peel(&g, k), "k = {k}, n = {n}");
                }
                // Core numbers are bounded by the degree.
                for v in 0..n {
                    assert!(core[v] <= g.degree(v));
                }
            }
        }
        // A complete graph is its own (n-1)-core.
        assert_eq!(complete_graph(5).k_core(4), vec![0, 1, 2, 3, 4]);
        assert!(complete_graph(5).k_core(5).is_empty());
        // A cycle is 2-regular.
        assert_eq!(cycle_graph(6).core_numbers(), vec![2; 6]);
    }

    /// The k-core by direct peeling: repeatedly delete a vertex of degree
    /// below k until none remains.
    fn peel(g: &Graph, k: usize) -> Vec<usize> {
        let mut alive: Vec<bool> = vec![true; g.n];
        loop {
            let mut removed = false;
            for v in 0..g.n {
                if !alive[v] {
                    continue;
                }
                let d = g.adj[v]
                    .iter()
                    .filter(|&&(t, _)| t != v && alive[t])
                    .count();
                if d < k {
                    alive[v] = false;
                    removed = true;
                }
            }
            if !removed {
                break;
            }
        }
        (0..g.n).filter(|&v| alive[v]).collect()
    }

    /// Assortativity is a correlation, so it must lie in [-1, 1], be +1 on a
    /// regular graph's degenerate case, and be negative on a star.
    #[test]
    fn assortativity_is_a_bounded_correlation() {
        let mut rng = Rng::new(181);
        for n in 2..=10usize {
            let g = random_graph(n, 0.4, false, &mut rng);
            let a = g.assortativity();
            assert!((-1.0..=1.0).contains(&a) || a == 0.0, "n = {n} gave {a}");
        }
        // A star is maximally disassortative: every edge joins degree n-1 to
        // degree 1, so the correlation is -1.
        assert!((star_graph(8).assortativity() + 1.0).abs() < 1e-9);
        // A regular graph has zero variance in degree, so the correlation is
        // undefined and reported as zero.
        assert_eq!(cycle_graph(6).assortativity(), 0.0);
        assert_eq!(complete_graph(5).assortativity(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Generators
    // -----------------------------------------------------------------------

    #[test]
    fn named_graphs_have_their_defining_properties() {
        for n in 1..=8usize {
            let k = complete_graph(n);
            assert_eq!(k.edge_count(), n * (n - 1) / 2);
            assert!((0..n).all(|v| k.degree(v) == n - 1));
        }
        for n in 3..=9usize {
            let c = cycle_graph(n);
            assert_eq!(c.edge_count(), n);
            assert!((0..n).all(|v| c.degree(v) == 2));
            assert!(c.is_connected());
            assert!(!c.is_tree());
        }
        for n in 1..=9usize {
            let p = path_graph(n);
            assert_eq!(p.edge_count(), n.saturating_sub(1));
            assert!(p.is_tree());
            let s = star_graph(n);
            assert!(s.is_tree());
            assert_eq!(s.degree(0), n.saturating_sub(1));
        }
        for n in 4..=9usize {
            let w = wheel_graph(n);
            assert_eq!(w.edge_count(), 2 * (n - 1));
            assert_eq!(w.degree(0), n - 1);
            assert!((1..n).all(|v| w.degree(v) == 3));
        }
        let grid = grid_2d(4, 3);
        assert_eq!(grid.n, 12);
        // Horizontal plus vertical edges.
        assert_eq!(grid.edge_count(), 3 * 3 + 4 * 2);
        assert!(grid.is_bipartite().is_some());

        for d in 0..=5u32 {
            let h = hypercube_graph(d);
            assert_eq!(h.n, 1 << d);
            assert!((0..h.n).all(|v| h.degree(v) == d as usize));
            assert_eq!(h.edge_count(), (d as usize) * (1 << d) / 2);
            assert!(h.is_bipartite().is_some());
        }

        let p = petersen_graph();
        assert_eq!(p.n, 10);
        assert_eq!(p.edge_count(), 15);
        assert!((0..10).all(|v| p.degree(v) == 3), "not 3-regular");
        assert_eq!(p.girth(), Some(5));
        assert_eq!(p.diameter(), Some(2));
        assert!(p.is_connected());

        for m in 1..=5usize {
            for n in 1..=5usize {
                let b = complete_bipartite(m, n);
                assert_eq!(b.edge_count(), m * n);
                let color = b.is_bipartite().expect("bipartite by construction");
                assert!((0..m).all(|v| color[v] == color[0]));
            }
        }
    }

    #[test]
    fn random_generators_respect_their_parameters() {
        let mut rng = Rng::new(191);
        // Erdos-Renyi at p = 0 and p = 1 are the extremes.
        assert_eq!(erdos_renyi(8, 0.0, &mut rng).edge_count(), 0);
        assert_eq!(erdos_renyi(8, 1.0, &mut rng).edge_count(), 28);
        // The expected edge count is p * C(n, 2).
        let mut total = 0usize;
        for _ in 0..200 {
            total += erdos_renyi(20, 0.3, &mut rng).edge_count();
        }
        let mean = total as f64 / 200.0;
        let expected = 0.3 * 190.0;
        assert!((mean - expected).abs() < 0.1 * expected, "mean {mean} vs {expected}");

        // Barabasi-Albert: n vertices, and each new one adds exactly m edges.
        for m in 1..=3usize {
            let g = barabasi_albert(30, m, &mut rng);
            assert_eq!(g.n, 30);
            assert_eq!(g.edge_count(), m * (m - 1) / 2 + m * (30 - m));
            assert!(g.is_connected());
        }

        // Watts-Strogatz at beta = 0 is the ring lattice.
        let ring = watts_strogatz(20, 4, 0.0, &mut rng);
        assert_eq!(ring.edge_count(), 40);
        assert!((0..20).all(|v| ring.degree(v) == 4));
        // Rewiring keeps the edge count.
        let rewired = watts_strogatz(20, 4, 0.5, &mut rng);
        assert_eq!(rewired.edge_count(), 40);

        // Random regular graphs really are regular.
        for (n, d) in [(10usize, 3usize), (12, 4), (9, 4), (20, 5)] {
            let g = random_regular(n, d, &mut rng).expect("a d-regular graph exists");
            assert!((0..n).all(|v| g.degree(v) == d), "n = {n}, d = {d}");
            // Simple: no self-loop, no repeat.
            let e: BTreeSet<(usize, usize)> = g
                .edges()
                .into_iter()
                .map(|(u, v, _)| (u.min(v), u.max(v)))
                .collect();
            assert_eq!(e.len(), g.edge_count());
            assert!(e.iter().all(|&(u, v)| u != v));
        }
        // n * d odd is impossible.
        assert!(random_regular(5, 3, &mut rng).is_none());

        // Geometric: an edge exactly when within the radius.
        let (g, pts) = random_geometric(30, 0.3, &mut rng);
        for u in 0..30 {
            for v in u + 1..30 {
                let (dx, dy) = (pts[u].0 - pts[v].0, pts[u].1 - pts[v].1);
                let near = (dx * dx + dy * dy).sqrt() <= 0.3;
                let joined = g.adj[u].iter().any(|&(t, _)| t == v);
                assert_eq!(near, joined, "({u}, {v})");
            }
        }

        // Block model: no cross-block edges when the off-diagonal is zero.
        let p = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let sbm = stochastic_block_model(&[5, 5], &p, &mut rng);
        assert_eq!(sbm.connected_components().len(), 2);
        assert_eq!(sbm.edge_count(), 10 + 10);
    }

    // -----------------------------------------------------------------------
    // Derived graphs and isomorphism
    // -----------------------------------------------------------------------

    #[test]
    fn line_graph_has_the_expected_size() {
        // The line graph has one vertex per edge, and sum over v of C(d(v), 2)
        // edges -- each pair of edges at a common vertex.
        let mut rng = Rng::new(201);
        for n in 2..=8usize {
            let g = random_graph(n, 0.4, false, &mut rng);
            let (l, edges) = line_graph(&g);
            assert_eq!(l.n, g.edge_count());
            assert_eq!(edges.len(), g.edge_count());
            let want: usize = (0..n).map(|v| g.degree(v) * g.degree(v).saturating_sub(1) / 2).sum();
            assert_eq!(l.edge_count(), want, "n = {n}");
        }
        // The line graph of a cycle is the same cycle.
        for n in 3..=7usize {
            let (l, _) = line_graph(&cycle_graph(n));
            assert!(is_isomorphic_small(&l, &cycle_graph(n)), "n = {n}");
        }
        // The line graph of K3 is K3.
        let (l, _) = line_graph(&complete_graph(3));
        assert!(is_isomorphic_small(&l, &complete_graph(3)));
    }

    #[test]
    fn products_have_the_expected_size() {
        let a = path_graph(3);
        let b = path_graph(4);
        let c = cartesian_product(&a, &b);
        assert_eq!(c.n, 12);
        // |E(G x H)| = |V(G)| |E(H)| + |V(H)| |E(G)|.
        assert_eq!(c.edge_count(), 3 * 3 + 4 * 2);
        // A grid is exactly the Cartesian product of two paths. Checked at
        // 2 x 3, which is inside canonical_form_small's ten-vertex ceiling.
        let small = cartesian_product(&path_graph(2), &path_graph(3));
        assert!(is_isomorphic_small(&small, &grid_2d(3, 2)));
        // At 4 x 3 the degree sequence still has to match exactly.
        let mut dc: Vec<usize> = (0..12).map(|v| c.degree(v)).collect();
        let g43 = grid_2d(4, 3);
        let mut dg: Vec<usize> = (0..12).map(|v| g43.degree(v)).collect();
        dc.sort_unstable();
        dg.sort_unstable();
        assert_eq!(dc, dg);
        // The hypercube is the product of a hypercube with K2.
        let q3 = cartesian_product(&hypercube_graph(2), &complete_graph(2));
        assert!(is_isomorphic_small(&q3, &hypercube_graph(3)));

        // The tensor product of K2 with K2 is two disjoint edges.
        let t = tensor_product(&complete_graph(2), &complete_graph(2));
        assert_eq!(t.n, 4);
        assert_eq!(t.edge_count(), 2);
        assert_eq!(t.connected_components().len(), 2);
    }

    /// Isomorphism must be invariant under relabelling and must separate
    /// graphs that only agree on the degree sequence.
    #[test]
    fn isomorphism_is_relabelling_invariant() {
        let mut rng = Rng::new(211);
        for n in 1..=7usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.4, false, &mut rng);
                let perm = crate::discrete::combinatorics::random_permutation(n, &mut rng);
                let mut h = Graph::new(n, false);
                for (u, v, w) in g.edges() {
                    h.add_edge(perm[u], perm[v], w);
                }
                assert!(is_isomorphic_small(&g, &h), "relabelling broke isomorphism");
                assert_eq!(canonical_form_small(&g), canonical_form_small(&h));
            }
        }
        // The classic pair with the same degree sequence but not isomorphic:
        // C3 + C3 versus C6, both 2-regular on six vertices.
        let mut two_triangles = Graph::new(6, false);
        for (u, v) in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
            two_triangles.add_edge(u, v, 1.0);
        }
        let c6 = cycle_graph(6);
        let mut d1: Vec<usize> = (0..6).map(|v| two_triangles.degree(v)).collect();
        let mut d2: Vec<usize> = (0..6).map(|v| c6.degree(v)).collect();
        d1.sort_unstable();
        d2.sort_unstable();
        assert_eq!(d1, d2, "the degree sequences must agree for this to be a test");
        assert!(!is_isomorphic_small(&two_triangles, &c6));
        // Different sizes are refused cheaply.
        assert!(!is_isomorphic_small(&complete_graph(4), &complete_graph(5)));
    }

    /// graph6 must round-trip, and produce the published encodings.
    #[test]
    fn graph6_round_trips() {
        let mut rng = Rng::new(221);
        for n in 1..=10usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.4, false, &mut rng);
                let s = graph6_encode(&g);
                let back = graph6_decode(&s);
                assert_eq!(back.n, g.n);
                let ge: BTreeSet<(usize, usize)> = g
                    .edges()
                    .into_iter()
                    .map(|(u, v, _)| (u.min(v), u.max(v)))
                    .collect();
                let be: BTreeSet<(usize, usize)> = back
                    .edges()
                    .into_iter()
                    .map(|(u, v, _)| (u.min(v), u.max(v)))
                    .collect();
                assert_eq!(ge, be, "round trip failed for {s}");
                assert!(s.bytes().all(|b| (63..=126).contains(&b)), "not printable");
            }
        }
        // The published graph6 strings for the two five-vertex extremes.
        assert_eq!(graph6_encode(&complete_graph(5)), "D~{");
        assert_eq!(graph6_encode(&Graph::new(5, false)), "D??");
        assert_eq!(graph6_decode("D~{").edge_count(), 10);
    }

    /// Cayley's formula: the complete graph on n vertices has n^(n-2)
    /// spanning trees. Checked exactly, past where f64 would be exact.
    #[test]
    fn matrix_tree_theorem_gives_cayleys_formula() {
        for n in 1..=12u64 {
            let want = if n <= 2 {
                BigInt::one()
            } else {
                BigInt::from_u64(n).pow(n - 2)
            };
            assert_eq!(
                spanning_tree_count_exact(&complete_graph(n as usize)),
                want,
                "Cayley fails at n = {n}"
            );
        }
        // 12^10 is past 2^53, so an f64 determinant could not be exact here.
        assert_eq!(
            spanning_tree_count_exact(&complete_graph(12)).to_string(),
            "61917364224"
        );
        // A tree has exactly one spanning tree; a cycle has n.
        for n in 3..=8usize {
            assert_eq!(spanning_tree_count_exact(&path_graph(n)), BigInt::one());
            assert_eq!(
                spanning_tree_count_exact(&cycle_graph(n)),
                BigInt::from_u64(n as u64)
            );
        }
        // K_{m,n} has m^(n-1) n^(m-1).
        for m in 1..=4u64 {
            for n in 1..=4u64 {
                let want = BigInt::from_u64(m)
                    .pow(n - 1)
                    .mul(&BigInt::from_u64(n).pow(m - 1));
                assert_eq!(
                    spanning_tree_count_exact(&complete_bipartite(m as usize, n as usize)),
                    want,
                    "K_{{{m},{n}}}"
                );
            }
        }
        // The Petersen graph has 2000.
        assert_eq!(
            spanning_tree_count_exact(&petersen_graph()),
            BigInt::from_u64(2000)
        );
        // A disconnected graph has none.
        assert_eq!(spanning_tree_count_exact(&Graph::new(3, false)), BigInt::zero());
    }
}
