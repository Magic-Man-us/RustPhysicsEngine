//! Spectral graph theory: Laplacians, centralities, resistances, and
//! community detection.
//!
//! The Laplacian `L = D - A` is the object almost everything here rests on.
//! It is symmetric positive semi-definite for an undirected graph, its
//! smallest eigenvalue is always zero with the all-ones eigenvector, and the
//! multiplicity of that zero is the number of connected components. The
//! second-smallest eigenvalue -- the algebraic connectivity -- measures how
//! hard the graph is to cut, and its eigenvector orders the vertices in a way
//! that separates the graph well.
//!
//! Weights are treated as edge multiplicities where that makes sense
//! (Laplacian, resistance, random walks) and ignored where it does not
//! (the combinatorial centralities, which count edges).

use crate::exact::bigint::BigInt;
use crate::graph::core::Graph;
use crate::linalg::eigen::eigen_symmetric;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// Tolerance for treating an eigenvalue as zero.
///
/// The Jacobi solver leaves the exact zero of a Laplacian at around 1e-14
/// relative to the largest eigenvalue, so a fixed absolute threshold would be
/// wrong on a graph with large weights; everything here scales by the spectral
/// radius.
const EIG_TOL: f64 = 1e-9;

/// The combinatorial Laplacian `L = D - A`.
///
/// The degree is the weighted degree, so `L` has row sums of exactly zero and
/// the all-ones vector is always in its kernel. Self-loops contribute to
/// neither the degree nor the adjacency, since they cancel.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn laplacian_matrix(g: &Graph) -> Matrix {
    assert!(!g.directed, "the Laplacian here is for undirected graphs");
    let n = g.n;
    let mut l = Matrix::zeros(n, n);
    for (u, v, w) in g.edges() {
        if u == v {
            continue;
        }
        l.set(u, u, l.get(u, u) + w);
        l.set(v, v, l.get(v, v) + w);
        l.set(u, v, l.get(u, v) - w);
        l.set(v, u, l.get(v, u) - w);
    }
    l
}

/// The weighted degree of each vertex, ignoring self-loops.
#[must_use]
pub fn weighted_degrees(g: &Graph) -> Vec<f64> {
    let mut d = vec![0.0; g.n];
    for (u, v, w) in g.edges() {
        if u == v {
            continue;
        }
        d[u] += w;
        if !g.directed {
            d[v] += w;
        }
    }
    d
}

/// The symmetric normalized Laplacian `I - D^(-1/2) A D^(-1/2)`.
///
/// Its spectrum lies in `[0, 2]` whatever the graph, which is what makes it
/// the right object for comparing graphs of different sizes and densities.
/// The upper end is reached exactly on a bipartite component. An isolated
/// vertex has no degree to normalize by and is given a diagonal of zero.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn normalized_laplacian(g: &Graph) -> Matrix {
    assert!(!g.directed, "the Laplacian here is for undirected graphs");
    let n = g.n;
    let deg = weighted_degrees(g);
    let mut m = Matrix::zeros(n, n);
    for v in 0..n {
        if deg[v] > 0.0 {
            m.set(v, v, 1.0);
        }
    }
    for (u, v, w) in g.edges() {
        if u == v || deg[u] <= 0.0 || deg[v] <= 0.0 {
            continue;
        }
        let s = w / (deg[u] * deg[v]).sqrt();
        m.set(u, v, m.get(u, v) - s);
        m.set(v, u, m.get(v, u) - s);
    }
    m
}

/// The adjacency eigenvalues, ascending.
///
/// # Panics
/// Panics if the graph is directed, or the solver fails to converge.
#[must_use]
pub fn adjacency_spectrum(g: &Graph) -> Vec<f64> {
    assert!(!g.directed, "the adjacency spectrum here is for undirected graphs");
    let mut a = g.to_adjacency_matrix();
    // to_adjacency_matrix keeps a self-loop on the diagonal; the spectrum is
    // conventionally taken of the simple adjacency, so drop them.
    for v in 0..g.n {
        a.set(v, v, 0.0);
    }
    ascending_eigenvalues(&a)
}

/// The Laplacian eigenvalues, ascending. The first is always zero.
///
/// # Panics
/// Panics if the graph is directed, or the solver fails to converge.
#[must_use]
pub fn laplacian_spectrum(g: &Graph) -> Vec<f64> {
    ascending_eigenvalues(&laplacian_matrix(g))
}

/// The normalized Laplacian eigenvalues, ascending. All lie in `[0, 2]`.
///
/// # Panics
/// Panics if the graph is directed, or the solver fails to converge.
#[must_use]
pub fn normalized_laplacian_spectrum(g: &Graph) -> Vec<f64> {
    ascending_eigenvalues(&normalized_laplacian(g))
}

/// Eigenvalues of a symmetric matrix in ascending order.
///
/// `eigen_symmetric` returns them descending, which is the wrong end for
/// spectral graph theory: the interesting eigenvalues of a Laplacian are the
/// smallest.
fn ascending_eigenvalues(m: &Matrix) -> Vec<f64> {
    let e = eigen_symmetric(m, 1e-12, 200).expect("Jacobi converges on a symmetric matrix");
    let mut v = e.values;
    v.reverse();
    v
}

/// The algebraic connectivity: the second-smallest Laplacian eigenvalue.
///
/// Zero exactly when the graph is disconnected, and larger the harder the
/// graph is to cut. Returns zero for fewer than two vertices.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn algebraic_connectivity(g: &Graph) -> f64 {
    if g.n < 2 {
        return 0.0;
    }
    let s = laplacian_spectrum(g);
    // Clamp: the exact zero comes back as a tiny negative or positive value,
    // and a negative algebraic connectivity is meaningless.
    s[1].max(0.0)
}

/// The Fiedler vector: the Laplacian eigenvector for the second-smallest
/// eigenvalue.
///
/// Its sign pattern is the classic spectral bisection, and its ordering is a
/// good one-dimensional embedding of the graph. Normalized to unit length,
/// with the sign fixed so the first non-zero entry is positive -- an
/// eigenvector is only defined up to sign, and leaving that free would make
/// the output unreproducible.
///
/// # Panics
/// Panics if the graph is directed, or has fewer than two vertices.
#[must_use]
pub fn fiedler_vector(g: &Graph) -> Vec<f64> {
    assert!(g.n >= 2, "the Fiedler vector needs at least two vertices");
    let l = laplacian_matrix(g);
    let e = eigen_symmetric(&l, 1e-12, 200).expect("Jacobi converges on a symmetric matrix");
    // Descending order, so the second-smallest is at index n - 2.
    let col = g.n - 2;
    let mut v: Vec<f64> = (0..g.n).map(|r| e.vectors.get(r, col)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    if let Some(&first) = v.iter().find(|x| x.abs() > EIG_TOL) {
        if first < 0.0 {
            v.iter_mut().for_each(|x| *x = -*x);
        }
    }
    v
}

/// Spectral bisection: split the vertices by the sign of the Fiedler vector.
///
/// # Panics
/// Panics if the graph is directed, or has fewer than two vertices.
#[must_use]
pub fn spectral_bisection(g: &Graph) -> Vec<bool> {
    fiedler_vector(g).into_iter().map(|x| x >= 0.0).collect()
}

/// Spectral clustering into `k` groups.
///
/// Embeds each vertex in the `k` lowest Laplacian eigenvectors and runs
/// k-means there. The embedding is what does the work: in it, vertices that
/// are hard to separate by cutting edges sit close together, so a distance
/// clustering in that space corresponds to a good cut in the graph.
///
/// # Panics
/// Panics if the graph is directed, `k` is zero, or `k` exceeds the vertex
/// count.
#[must_use]
pub fn spectral_clustering(g: &Graph, k: usize, rng: &mut Rng) -> Vec<usize> {
    assert!(k > 0 && k <= g.n, "k must satisfy 1 <= k <= n");
    if k == 1 {
        return vec![0; g.n];
    }
    let l = laplacian_matrix(g);
    let e = eigen_symmetric(&l, 1e-12, 200).expect("Jacobi converges on a symmetric matrix");
    // The k smallest eigenvectors are the last k columns.
    let coords: Vec<Vec<f64>> = (0..g.n)
        .map(|v| (0..k).map(|j| e.vectors.get(v, g.n - 1 - j)).collect())
        .collect();
    kmeans(&coords, k, rng)
}

/// Lloyd's algorithm on the given points, seeded by k-means++.
fn kmeans(points: &[Vec<f64>], k: usize, rng: &mut Rng) -> Vec<usize> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = points[0].len();
    let dist2 = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    };
    // k-means++ seeding: each new centre is drawn with probability
    // proportional to its squared distance from the nearest chosen one, which
    // is what keeps Lloyd's from starting with two centres on top of each
    // other.
    let first = ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize;
    let mut centres: Vec<Vec<f64>> = vec![points[first].clone()];
    while centres.len() < k {
        let d: Vec<f64> = points
            .iter()
            .map(|p| centres.iter().map(|c| dist2(p, c)).fold(f64::INFINITY, f64::min))
            .collect();
        let total: f64 = d.iter().sum();
        let pick = if total <= 0.0 {
            ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
        } else {
            let target = rng.next_f64() * total;
            let mut acc = 0.0;
            let mut idx = n - 1;
            for (i, &x) in d.iter().enumerate() {
                acc += x;
                if acc >= target {
                    idx = i;
                    break;
                }
            }
            idx
        };
        centres.push(points[pick].clone());
    }

    let mut label = vec![0usize; n];
    for _ in 0..100 {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let best = (0..k)
                .min_by(|&a, &b| dist2(p, &centres[a]).total_cmp(&dist2(p, &centres[b])))
                .unwrap();
            if label[i] != best {
                label[i] = best;
                changed = true;
            }
        }
        // Recentre. An empty cluster keeps its old centre rather than moving
        // to the origin, which would drag it into the middle of the data.
        for c in 0..k {
            let members: Vec<&Vec<f64>> =
                (0..n).filter(|&i| label[i] == c).map(|i| &points[i]).collect();
            if members.is_empty() {
                continue;
            }
            for j in 0..dim {
                centres[c][j] = members.iter().map(|p| p[j]).sum::<f64>() / members.len() as f64;
            }
        }
        if !changed {
            break;
        }
    }
    label
}

/// The number of spanning trees, by Kirchhoff's matrix-tree theorem.
///
/// Any cofactor of the Laplacian gives the count; this uses the product of
/// the non-zero Laplacian eigenvalues divided by `n`, which is the same
/// number and needs no pivoting. Returns zero for a disconnected graph.
///
/// The result is a float and is only exact while the count stays inside 53
/// bits; [`crate::graph::core::spanning_tree_count_exact`] does it over the
/// integers.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn number_spanning_trees(g: &Graph) -> f64 {
    if g.n == 0 {
        return 0.0;
    }
    if g.n == 1 {
        return 1.0;
    }
    let s = laplacian_spectrum(g);
    let scale = s.last().copied().unwrap_or(1.0).max(1.0);
    if s[1] <= EIG_TOL * scale {
        return 0.0;
    }
    s[1..].iter().product::<f64>() / g.n as f64
}

/// The number of spanning trees, exactly.
///
/// Re-exported from [`crate::graph::core::spanning_tree_count_exact`] so the
/// spectral module offers both the float and the exact form side by side.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn number_spanning_trees_exact(g: &Graph) -> BigInt {
    crate::graph::core::spanning_tree_count_exact(g)
}

// ---------------------------------------------------------------------------
// Centrality
// ---------------------------------------------------------------------------

/// PageRank with the given damping factor.
///
/// The rank vector is the stationary distribution of a random surfer who
/// follows an out-link with probability `damping` and teleports uniformly
/// otherwise. A vertex with no out-links would leak probability, so its mass
/// is redistributed uniformly -- without that the result would not sum to one.
///
/// Returns a distribution summing to one.
///
/// # Panics
/// Panics unless `damping` is in `[0, 1)` and `tol` is positive.
#[must_use]
pub fn pagerank(g: &Graph, damping: f64, tol: f64) -> Vec<f64> {
    assert!((0.0..1.0).contains(&damping), "damping must be in [0, 1)");
    assert!(tol > 0.0, "tol must be positive");
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }
    let out: Vec<f64> = (0..n)
        .map(|v| g.adj[v].iter().filter(|&&(t, _)| t != v).map(|&(_, w)| w).sum())
        .collect();
    let mut rank = vec![1.0 / n as f64; n];
    for _ in 0..1_000 {
        let mut next = vec![(1.0 - damping) / n as f64; n];
        // Dangling mass: a vertex with no outgoing weight sends its rank
        // everywhere rather than nowhere.
        let dangling: f64 = (0..n).filter(|&v| out[v] <= 0.0).map(|v| rank[v]).sum();
        for x in next.iter_mut() {
            *x += damping * dangling / n as f64;
        }
        for u in 0..n {
            if out[u] <= 0.0 {
                continue;
            }
            for &(v, w) in &g.adj[u] {
                if v != u {
                    next[v] += damping * rank[u] * w / out[u];
                }
            }
        }
        let delta: f64 = (0..n).map(|v| (next[v] - rank[v]).abs()).sum();
        rank = next;
        if delta < tol {
            break;
        }
    }
    rank
}

/// HITS: the hub and authority scores.
///
/// A good authority is pointed to by good hubs and a good hub points to good
/// authorities, which is a mutual recurrence solved by alternating updates.
/// Both vectors are normalized to unit length.
///
/// # Panics
/// Panics unless `tol` is positive.
#[must_use]
pub fn hits(g: &Graph, tol: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(tol > 0.0, "tol must be positive");
    let n = g.n;
    let mut hub = vec![1.0; n];
    let mut auth = vec![1.0; n];
    for _ in 0..1_000 {
        let mut new_auth = vec![0.0; n];
        for u in 0..n {
            for &(v, w) in &g.adj[u] {
                new_auth[v] += hub[u] * w;
            }
        }
        let mut new_hub = vec![0.0; n];
        for u in 0..n {
            for &(v, w) in &g.adj[u] {
                new_hub[u] += new_auth[v] * w;
            }
        }
        normalize(&mut new_auth);
        normalize(&mut new_hub);
        let delta: f64 = (0..n)
            .map(|v| (new_auth[v] - auth[v]).abs() + (new_hub[v] - hub[v]).abs())
            .sum();
        auth = new_auth;
        hub = new_hub;
        if delta < tol {
            break;
        }
    }
    (hub, auth)
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Eigenvector centrality: the principal eigenvector of the adjacency matrix.
///
/// A vertex is important when its neighbours are, which is exactly the
/// eigenvector equation. Found by power iteration; the result is
/// non-negative by Perron-Frobenius and is normalized to unit length.
///
/// # Panics
/// Panics unless `tol` is positive.
#[must_use]
pub fn eigenvector_centrality(g: &Graph, tol: f64) -> Vec<f64> {
    assert!(tol > 0.0, "tol must be positive");
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }
    // Power iteration on `A` alone does not converge on a bipartite graph:
    // its spectrum is symmetric about zero, so the two extreme eigenvalues
    // tie in magnitude and the iterate flips between their eigenvectors
    // forever. Iterating on `A + cI` for a Gershgorin bound `c` moves the
    // whole spectrum into `[0, 2c]`, which breaks the tie without moving a
    // single eigenvector, so the limit is still the principal eigenvector
    // of `A`. The floor of one keeps an edgeless graph off a zero iterate.
    let mut shift = 1.0f64;
    for u in 0..n {
        shift = shift.max(g.adj[u].iter().map(|&(_, w)| w.abs()).sum::<f64>());
    }
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..10_000 {
        // `adj` holds both directions of an undirected edge, so accumulating
        // into the far end once per stored arc builds `A x` exactly.
        let mut next: Vec<f64> = x.iter().map(|v| v * shift).collect();
        for u in 0..n {
            for &(v, w) in &g.adj[u] {
                next[v] += x[u] * w;
            }
        }
        normalize(&mut next);
        let delta: f64 = (0..n).map(|v| (next[v] - x[v]).abs()).sum();
        x = next;
        if delta < tol {
            break;
        }
    }
    x
}

/// Katz centrality: the attenuated count of walks reaching each vertex.
///
/// `x = (I - alpha A)^-1 * 1 - 1`, summed over walk lengths with each step
/// weighted by `alpha`. Converges only when `alpha` is below the reciprocal
/// of the largest adjacency eigenvalue, which is the caller's responsibility;
/// beyond that the walk count diverges and so does the series.
///
/// # Panics
/// Panics unless `alpha` is positive.
#[must_use]
pub fn katz_centrality(g: &Graph, alpha: f64) -> Vec<f64> {
    assert!(alpha > 0.0, "alpha must be positive");
    let n = g.n;
    let mut x = vec![0.0; n];
    for _ in 0..10_000 {
        let mut next = vec![1.0; n];
        for u in 0..n {
            for &(v, w) in &g.adj[u] {
                next[v] += alpha * x[u] * w;
            }
        }
        let delta: f64 = (0..n).map(|v| (next[v] - x[v]).abs()).sum();
        x = next;
        if delta < 1e-12 {
            break;
        }
    }
    x
}

/// Betweenness centrality, by Brandes' algorithm.
///
/// The number of shortest paths through each vertex, summed over all source
/// and target pairs and normalized by how many shortest paths there are.
/// Brandes computes it in `O(VE)` by accumulating dependencies backwards
/// along one shortest-path DAG per source, rather than enumerating the
/// quadratically many pairs.
///
/// Counts hops rather than weights. An undirected graph counts each unordered
/// pair once, so the values are halved.
#[must_use]
pub fn betweenness_centrality(g: &Graph) -> Vec<f64> {
    let n = g.n;
    let mut score = vec![0.0; n];
    for s in 0..n {
        // Forward pass: BFS building the shortest-path DAG.
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n];
        let mut dist = vec![usize::MAX; n];
        let mut order: Vec<usize> = Vec::new();
        sigma[s] = 1.0;
        dist[s] = 0;
        let mut queue = std::collections::VecDeque::from(vec![s]);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            for &(w, _) in &g.adj[v] {
                if dist[w] == usize::MAX {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    preds[w].push(v);
                }
            }
        }
        // Backward pass: accumulate dependencies from the far end inwards.
        let mut delta = vec![0.0f64; n];
        for &w in order.iter().rev() {
            for &v in &preds[w] {
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w]);
            }
            if w != s {
                score[w] += delta[w];
            }
        }
    }
    if !g.directed {
        score.iter_mut().for_each(|x| *x /= 2.0);
    }
    score
}

/// Closeness centrality: the reciprocal of the mean hop distance to every
/// reachable vertex, scaled by the fraction reachable.
///
/// The scaling is what makes the value comparable across components: without
/// it, a vertex in a small tight component would outrank one in a large
/// well-connected component.
#[must_use]
pub fn closeness_centrality(g: &Graph) -> Vec<f64> {
    let n = g.n;
    (0..n)
        .map(|v| {
            let d = g.bfs(v);
            // Distance zero is the vertex itself, which is not a target.
            let reached: Vec<usize> = d.iter().filter_map(|x| *x).filter(|&x| x > 0).collect();
            let total: usize = reached.iter().sum();
            if total == 0 {
                return 0.0;
            }
            let r = reached.len() as f64;
            (r / total as f64) * (r / (n as f64 - 1.0))
        })
        .collect()
}

/// Harmonic centrality: the sum of reciprocal distances.
///
/// Unlike closeness this needs no special case for a disconnected graph -- an
/// unreachable vertex contributes `1/infinity = 0` -- which is why it is
/// preferred when the graph may not be connected.
#[must_use]
pub fn harmonic_centrality(g: &Graph) -> Vec<f64> {
    (0..g.n)
        .map(|v| {
            g.bfs(v)
                .iter()
                .enumerate()
                .filter(|&(u, _)| u != v)
                .filter_map(|(_, d)| d.map(|x| 1.0 / x as f64))
                .sum()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Resistance and random walks
// ---------------------------------------------------------------------------

/// The Moore-Penrose pseudoinverse of the Laplacian.
///
/// The Laplacian is singular by construction, so the ordinary inverse does not
/// exist. The pseudoinverse is formed by inverting the non-zero eigenvalues
/// and leaving the kernel alone, which is what makes the resistance formulas
/// below well defined.
fn laplacian_pseudoinverse(g: &Graph) -> Matrix {
    let n = g.n;
    let l = laplacian_matrix(g);
    let e = eigen_symmetric(&l, 1e-12, 200).expect("Jacobi converges on a symmetric matrix");
    let scale = e.values.iter().fold(0.0f64, |a, &b| a.max(b.abs())).max(1.0);
    let mut p = Matrix::zeros(n, n);
    for k in 0..n {
        let lambda = e.values[k];
        if lambda.abs() <= EIG_TOL * scale {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                let add = e.vectors.get(i, k) * e.vectors.get(j, k) / lambda;
                p.set(i, j, p.get(i, j) + add);
            }
        }
    }
    p
}

/// The effective resistance between two vertices, treating each edge as a
/// conductance equal to its weight.
///
/// `R(u,v) = L+(u,u) + L+(v,v) - 2 L+(u,v)` for the Laplacian pseudoinverse.
/// Infinite when the two lie in different components.
///
/// # Panics
/// Panics if the graph is directed, or an endpoint is out of range.
#[must_use]
pub fn effective_resistance(g: &Graph, u: usize, v: usize) -> f64 {
    assert!(u < g.n && v < g.n, "endpoints must be vertices");
    if u == v {
        return 0.0;
    }
    if g.bfs(u)[v].is_none() {
        return f64::INFINITY;
    }
    let p = laplacian_pseudoinverse(g);
    p.get(u, u) + p.get(v, v) - 2.0 * p.get(u, v)
}

/// The effective resistance between every pair.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn resistance_matrix(g: &Graph) -> Matrix {
    let n = g.n;
    let p = laplacian_pseudoinverse(g);
    let mut r = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let value = if g.bfs(i)[j].is_none() {
                f64::INFINITY
            } else {
                p.get(i, i) + p.get(j, j) - 2.0 * p.get(i, j)
            };
            r.set(i, j, value);
        }
    }
    r
}

/// The commute time between two vertices: the expected number of steps for a
/// random walk to go from `u` to `v` and back.
///
/// Equal to `2m * R(u,v)` for total edge weight `m`, which is the theorem
/// that makes effective resistance a graph distance rather than merely an
/// analogy.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn commute_time(g: &Graph, u: usize, v: usize) -> f64 {
    let total: f64 = g.edges().iter().filter(|&&(a, b, _)| a != b).map(|&(_, _, w)| w).sum();
    2.0 * total * effective_resistance(g, u, v)
}

/// The stationary distribution of a simple random walk.
///
/// On a connected undirected graph this is the degree distribution: the walk
/// spends time at a vertex in proportion to its weighted degree.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn random_walk_stationary(g: &Graph) -> Vec<f64> {
    assert!(!g.directed, "this stationary form is for undirected graphs");
    let deg = weighted_degrees(g);
    let total: f64 = deg.iter().sum();
    if total <= 0.0 {
        return vec![0.0; g.n];
    }
    deg.into_iter().map(|d| d / total).collect()
}

/// An estimate of the mixing time: how many steps until the walk is within
/// `eps` of stationary in total variation.
///
/// Bounded by `log(1/(eps * pi_min)) / (1 - lambda2)` for the second-largest
/// transition eigenvalue in magnitude, which relates mixing to the spectral
/// gap. Infinite when the graph is disconnected or bipartite, where the walk
/// does not converge at all.
///
/// # Panics
/// Panics if the graph is directed, or `eps` is not in `(0, 1)`.
#[must_use]
pub fn mixing_time_estimate(g: &Graph, eps: f64) -> f64 {
    assert!((0.0..1.0).contains(&eps) && eps > 0.0, "eps must be in (0, 1)");
    let pi = random_walk_stationary(g);
    let pi_min = pi.iter().copied().fold(f64::INFINITY, f64::min);
    if pi_min <= 0.0 {
        return f64::INFINITY;
    }
    // The transition eigenvalues are 1 - mu for the normalized Laplacian's mu.
    // Drop exactly one zero: that is the Perron eigenvalue, transition
    // eigenvalue one, which carries the stationary distribution and is
    // supposed to stay. What governs convergence is the largest magnitude
    // among the rest. A second eigenvalue of magnitude one means the walk
    // never settles -- from another mu = 0, so a second component, or from
    // mu = 2, so a bipartite component whose walk changes side every step.
    let mut s = normalized_laplacian_spectrum(g);
    if s.is_empty() {
        return f64::INFINITY;
    }
    s.remove(0);
    let second = s.iter().map(|&mu| (1.0 - mu).abs()).fold(0.0f64, f64::max);
    if second >= 1.0 - EIG_TOL {
        return f64::INFINITY;
    }
    (1.0 / (eps * pi_min)).ln() / (1.0 - second)
}

/// The Cheeger bounds on the graph's conductance.
///
/// Cheeger's inequality brackets the conductance `h` between `mu/2` and
/// `sqrt(2 mu)` for the second-smallest normalized Laplacian eigenvalue `mu`.
/// Returns `(lower, upper)`.
///
/// # Panics
/// Panics if the graph is directed, or has fewer than two vertices.
#[must_use]
pub fn cheeger_bound(g: &Graph) -> (f64, f64) {
    assert!(g.n >= 2, "conductance needs at least two vertices");
    let s = normalized_laplacian_spectrum(g);
    let mu = s[1].max(0.0);
    (mu / 2.0, (2.0 * mu).sqrt())
}

/// True when the graph's spectral gap is at least `target_gap`.
///
/// The gap is what makes an expander an expander: a large gap forces every
/// cut to be expensive, by Cheeger's inequality.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn expander_check(g: &Graph, target_gap: f64) -> bool {
    g.n >= 2 && algebraic_connectivity(g) >= target_gap
}

/// The graph energy: the sum of the absolute adjacency eigenvalues.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn graph_energy(g: &Graph) -> f64 {
    adjacency_spectrum(g).iter().map(|x| x.abs()).sum()
}

/// The Estrada index: the sum of `exp(lambda)` over the adjacency spectrum.
///
/// Equal to the trace of `exp(A)`, which counts closed walks with each length
/// weighted by the reciprocal of its factorial.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn estrada_index(g: &Graph) -> f64 {
    adjacency_spectrum(g).iter().map(|x| x.exp()).sum()
}

/// True when two graphs have the same adjacency spectrum to within `tol`.
///
/// Isomorphic graphs are always isospectral; the converse is false, which is
/// what makes the spectrum a cheap but incomplete invariant.
///
/// # Panics
/// Panics if either graph is directed.
#[must_use]
pub fn isospectral_check(g: &Graph, h: &Graph, tol: f64) -> bool {
    if g.n != h.n {
        return false;
    }
    let a = adjacency_spectrum(g);
    let b = adjacency_spectrum(h);
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
}

// ---------------------------------------------------------------------------
// Communities
// ---------------------------------------------------------------------------

/// Degrees as modularity counts them, where a self-loop contributes twice
/// because both of its ends are at the same vertex.
///
/// `weighted_degrees` drops loops, which is right for the Laplacian: a loop
/// adds the same amount to `D` and to `A`, so it cancels in `L = D - A`.
/// Modularity has no such cancellation, and Louvain's contraction turns each
/// community's internal edges into exactly one loop -- so a degree sum that
/// ignored loops would shrink at every level of the recursion and the gain
/// formula would be comparing against the wrong total edge weight.
fn modularity_degrees(g: &Graph) -> Vec<f64> {
    let mut d = vec![0.0; g.n];
    for (u, v, w) in g.edges() {
        d[u] += w;
        d[v] += w;
    }
    d
}

/// Newman's modularity of a vertex partition.
///
/// The fraction of edge weight inside communities, minus what that fraction
/// would be if the same degrees were wired at random. Positive means the
/// partition captures more structure than chance; the maximum over all
/// partitions is what community detection tries to find.
///
/// # Panics
/// Panics if the graph is directed, or `communities` does not have one label
/// per vertex.
#[must_use]
pub fn modularity(g: &Graph, communities: &[usize]) -> f64 {
    assert!(!g.directed, "modularity here is for undirected graphs");
    assert_eq!(communities.len(), g.n, "one label per vertex is required");
    let deg = modularity_degrees(g);
    let two_m: f64 = deg.iter().sum();
    if two_m <= 0.0 {
        return 0.0;
    }
    let mut inside = 0.0;
    for (u, v, w) in g.edges() {
        if communities[u] != communities[v] {
            continue;
        }
        // Each internal edge contributes twice to the 2m normalisation. A
        // self-loop is internal by definition and counts on the same footing.
        inside += 2.0 * w;
    }
    let labels: std::collections::BTreeSet<usize> = communities.iter().copied().collect();
    let expected: f64 = labels
        .iter()
        .map(|&c| {
            let d: f64 = (0..g.n).filter(|&v| communities[v] == c).map(|v| deg[v]).sum();
            (d / two_m) * (d / two_m)
        })
        .sum();
    inside / two_m - expected
}

/// Community detection by the Louvain method.
///
/// Two phases repeated: move each vertex to whichever neighbouring community
/// most improves modularity, then contract each community to a single vertex
/// and repeat on the smaller graph. The contraction is what lets it find
/// structure at several scales rather than only among immediate neighbours.
///
/// Labels are renumbered from zero in order of first appearance.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn community_louvain(g: &Graph, rng: &mut Rng) -> Vec<usize> {
    assert!(!g.directed, "Louvain here is for undirected graphs");
    let n = g.n;
    if n == 0 {
        return Vec::new();
    }
    // Each original vertex's current community, and the working graph.
    let mut assignment: Vec<usize> = (0..n).collect();
    let mut work = g.clone();

    for _ in 0..20 {
        let m = work.n;
        let deg = modularity_degrees(&work);
        let two_m: f64 = deg.iter().sum();
        if two_m <= 0.0 {
            break;
        }
        let mut comm: Vec<usize> = (0..m).collect();
        // Total degree of each community.
        let mut ctot: Vec<f64> = deg.clone();
        let mut improved = false;

        for _ in 0..20 {
            let mut moved = false;
            let order = crate::discrete::combinatorics::random_permutation(m, rng);
            for &v in &order {
                let old = comm[v];
                ctot[old] -= deg[v];
                // Weight from v into each neighbouring community.
                let mut links: std::collections::BTreeMap<usize, f64> =
                    std::collections::BTreeMap::new();
                for &(w, weight) in &work.adj[v] {
                    if w != v {
                        *links.entry(comm[w]).or_insert(0.0) += weight;
                    }
                }
                // The modularity gain of joining c is k_in - k_v * tot_c / 2m.
                let mut best = old;
                let mut best_gain = links.get(&old).copied().unwrap_or(0.0)
                    - deg[v] * ctot[old] / two_m;
                for (&c, &k_in) in &links {
                    let gain = k_in - deg[v] * ctot[c] / two_m;
                    if gain > best_gain + 1e-12 {
                        best_gain = gain;
                        best = c;
                    }
                }
                ctot[best] += deg[v];
                if best != old {
                    comm[v] = best;
                    moved = true;
                    improved = true;
                }
            }
            if !moved {
                break;
            }
        }
        if !improved {
            break;
        }
        // Renumber the communities and push the labels down to the originals.
        let mut relabel: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for &c in &comm {
            let next = relabel.len();
            relabel.entry(c).or_insert(next);
        }
        let compact: Vec<usize> = comm.iter().map(|c| relabel[c]).collect();
        for a in assignment.iter_mut() {
            *a = compact[*a];
        }
        // Contract.
        let mut next_graph = Graph::new(relabel.len(), false);
        let mut merged: std::collections::BTreeMap<(usize, usize), f64> =
            std::collections::BTreeMap::new();
        for (u, v, w) in work.edges() {
            let (a, b) = (compact[u], compact[v]);
            *merged.entry((a.min(b), a.max(b))).or_insert(0.0) += w;
        }
        for ((a, b), w) in merged {
            next_graph.add_edge(a, b, w);
        }
        if next_graph.n == work.n {
            break;
        }
        work = next_graph;
    }
    renumber(&assignment)
}

/// Community detection by label propagation.
///
/// Each vertex repeatedly adopts the label carried by the greatest weight
/// among its neighbours, ties broken at random. Near-linear and parameter-
/// free, but the outcome depends on the visiting order, which is why the
/// generator is a parameter rather than fixed.
///
/// # Panics
/// Panics if the graph is directed.
#[must_use]
pub fn label_propagation(g: &Graph, rng: &mut Rng) -> Vec<usize> {
    assert!(!g.directed, "label propagation here is for undirected graphs");
    let n = g.n;
    let mut label: Vec<usize> = (0..n).collect();
    for _ in 0..100 {
        let mut changed = false;
        for &v in &crate::discrete::combinatorics::random_permutation(n, rng) {
            let mut weight: std::collections::BTreeMap<usize, f64> =
                std::collections::BTreeMap::new();
            for &(w, x) in &g.adj[v] {
                if w != v {
                    *weight.entry(label[w]).or_insert(0.0) += x;
                }
            }
            if weight.is_empty() {
                continue;
            }
            let best = weight
                .values()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let tied: Vec<usize> = weight
                .iter()
                .filter(|(_, &w)| (w - best).abs() < 1e-12)
                .map(|(&l, _)| l)
                .collect();
            let pick = tied[((u128::from(rng.next_u64()) * tied.len() as u128) >> 64) as usize];
            if pick != label[v] {
                label[v] = pick;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    renumber(&label)
}

/// Renumbers labels from zero in order of first appearance, so the output
/// depends on the partition rather than on which internal labels survived.
fn renumber(labels: &[usize]) -> Vec<usize> {
    let mut map: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    labels
        .iter()
        .map(|&l| {
            let next = map.len();
            *map.entry(l).or_insert(next)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::{
        complete_bipartite, complete_graph, cycle_graph, hypercube_graph, path_graph,
        petersen_graph, star_graph,
    };

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-7 * a.abs().max(b.abs()).max(1.0)
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

    // -----------------------------------------------------------------------
    // Laplacians
    // -----------------------------------------------------------------------

    /// The defining structure of a Laplacian: symmetric, zero row sums, the
    /// all-ones vector in its kernel, and positive semi-definite.
    #[test]
    fn laplacian_has_its_defining_structure() {
        let mut rng = Rng::new(0x_1A81);
        for n in 1..=9usize {
            for _ in 0..15 {
                let g = random_graph(n, 0.4, &mut rng);
                let l = laplacian_matrix(&g);
                for i in 0..n {
                    // Symmetric.
                    for j in 0..n {
                        assert!(close(l.get(i, j), l.get(j, i)), "not symmetric");
                    }
                    // Row sums vanish, so L * 1 = 0.
                    let row: f64 = (0..n).map(|j| l.get(i, j)).sum();
                    assert!(row.abs() < 1e-9, "row {i} sums to {row}");
                    // The diagonal is the degree and the off-diagonal the
                    // negated adjacency.
                    assert!(close(l.get(i, i), g.degree(i) as f64));
                }
                // Positive semi-definite: every eigenvalue is non-negative.
                let s = laplacian_spectrum(&g);
                assert!(s.iter().all(|&x| x > -1e-9), "negative eigenvalue in {s:?}");
                // The smallest is exactly zero.
                assert!(s[0].abs() < 1e-9, "smallest eigenvalue is {}", s[0]);
                // The trace is twice the edge count.
                let trace: f64 = s.iter().sum();
                assert!(close(trace, 2.0 * g.edge_count() as f64), "trace is wrong");
            }
        }
    }

    /// The multiplicity of the zero Laplacian eigenvalue is the number of
    /// connected components. This is the theorem the whole module rests on.
    #[test]
    fn zero_eigenvalue_multiplicity_is_the_component_count() {
        let mut rng = Rng::new(0x_C0AF);
        for n in 1..=9usize {
            for _ in 0..25 {
                let g = random_graph(n, 0.25, &mut rng);
                let s = laplacian_spectrum(&g);
                let scale = s.last().copied().unwrap_or(1.0).max(1.0);
                let zeros = s.iter().filter(|&&x| x.abs() <= 1e-9 * scale).count();
                assert_eq!(
                    zeros,
                    g.connected_components().len(),
                    "n = {n}, spectrum {s:?}"
                );
                // Equivalently: the algebraic connectivity vanishes exactly
                // when the graph is disconnected.
                let connected = g.is_connected();
                assert_eq!(
                    algebraic_connectivity(&g) > 1e-9 * scale,
                    connected && n >= 2,
                    "connectivity disagrees at n = {n}"
                );
            }
        }
    }

    /// Closed-form Laplacian spectra for the named families.
    #[test]
    fn laplacian_spectra_match_their_closed_forms() {
        // K_n: eigenvalue 0 once and n with multiplicity n - 1.
        for n in 2..=8usize {
            let s = laplacian_spectrum(&complete_graph(n));
            assert!(s[0].abs() < 1e-9);
            for &x in &s[1..] {
                assert!(close(x, n as f64), "K{n} gave {x}");
            }
            assert!(close(algebraic_connectivity(&complete_graph(n)), n as f64));
        }
        // C_n: 2 - 2 cos(2 pi k / n).
        for n in 3..=9usize {
            let s = laplacian_spectrum(&cycle_graph(n));
            let mut want: Vec<f64> = (0..n)
                .map(|k| 2.0 - 2.0 * (std::f64::consts::TAU * k as f64 / n as f64).cos())
                .collect();
            want.sort_by(f64::total_cmp);
            for (a, b) in s.iter().zip(&want) {
                assert!(close(*a, *b), "C{n}: {a} vs {b}");
            }
        }
        // P_n: 2 - 2 cos(pi k / n).
        for n in 2..=9usize {
            let s = laplacian_spectrum(&path_graph(n));
            let mut want: Vec<f64> = (0..n)
                .map(|k| 2.0 - 2.0 * (std::f64::consts::PI * k as f64 / n as f64).cos())
                .collect();
            want.sort_by(f64::total_cmp);
            for (a, b) in s.iter().zip(&want) {
                assert!(close(*a, *b), "P{n}: {a} vs {b}");
            }
        }
        // The star K_{1,n-1}: 0, 1 with multiplicity n - 2, and n.
        for n in 3..=8usize {
            let s = laplacian_spectrum(&star_graph(n));
            assert!(s[0].abs() < 1e-9);
            for &x in &s[1..n - 1] {
                assert!(close(x, 1.0), "star {n} gave {x}");
            }
            assert!(close(s[n - 1], n as f64));
        }
        // The hypercube Q_d: 2k with multiplicity C(d, k).
        for d in 1..=4u32 {
            let s = laplacian_spectrum(&hypercube_graph(d));
            for k in 0..=d {
                let want = 2.0 * k as f64;
                let count = s.iter().filter(|&&x| close(x, want)).count();
                let expect = crate::discrete::combinatorics::binomial_u64(
                    u64::from(d),
                    u64::from(k),
                )
                .unwrap() as usize;
                assert_eq!(count, expect, "Q{d} eigenvalue {want}");
            }
        }
    }

    /// The normalized Laplacian's spectrum lies in [0, 2], and reaches 2
    /// exactly on a bipartite graph.
    #[test]
    fn normalized_spectrum_is_bounded_and_detects_bipartiteness() {
        let mut rng = Rng::new(0x_0A1F);
        for n in 2..=9usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.4, &mut rng);
                if g.edge_count() == 0 {
                    continue;
                }
                let s = normalized_laplacian_spectrum(&g);
                for &x in &s {
                    assert!(
                        (-1e-9..=2.0 + 1e-9).contains(&x),
                        "eigenvalue {x} is outside [0, 2]"
                    );
                }
                // Eigenvalue 2 appears exactly when some component with an
                // edge is bipartite.
                let has_two = s.iter().any(|&x| (x - 2.0).abs() < 1e-7);
                let bipartite_component = g
                    .connected_components()
                    .iter()
                    .filter(|c| c.len() > 1)
                    .any(|c| g.subgraph(c).is_bipartite().is_some());
                assert_eq!(has_two, bipartite_component, "n = {n}, spectrum {s:?}");
            }
        }
        // Known: a complete bipartite graph has 2 in its spectrum.
        for (m, n) in [(2usize, 3usize), (3, 3), (1, 4)] {
            let s = normalized_laplacian_spectrum(&complete_bipartite(m, n));
            assert!(s.iter().any(|&x| (x - 2.0).abs() < 1e-7), "K_{{{m},{n}}}");
        }
        // An odd cycle is not bipartite and so falls short of 2.
        let s = normalized_laplacian_spectrum(&cycle_graph(5));
        assert!(s.iter().all(|&x| x < 2.0 - 1e-6), "C5 should not reach 2");
    }

    /// The Fiedler vector must be a genuine eigenvector for the algebraic
    /// connectivity, orthogonal to the all-ones vector.
    #[test]
    fn fiedler_vector_is_the_second_eigenvector() {
        let mut rng = Rng::new(0x_F1ED);
        for n in 2..=9usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.45, &mut rng);
                if !g.is_connected() {
                    continue;
                }
                let v = fiedler_vector(&g);
                let l = laplacian_matrix(&g);
                let lambda = algebraic_connectivity(&g);
                // L v = lambda v, entry by entry.
                for i in 0..n {
                    let lv: f64 = (0..n).map(|j| l.get(i, j) * v[j]).sum();
                    assert!(
                        (lv - lambda * v[i]).abs() < 1e-6,
                        "not an eigenvector at {i}: {lv} vs {}",
                        lambda * v[i]
                    );
                }
                // Unit length and orthogonal to the constant vector.
                let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(close(norm, 1.0), "not normalized: {norm}");
                let dot: f64 = v.iter().sum();
                assert!(dot.abs() < 1e-6, "not orthogonal to ones: {dot}");
                // The sign convention is deterministic.
                assert_eq!(fiedler_vector(&g), v, "not reproducible");
                // Bisection splits into two non-empty parts on a path or
                // cycle, where the Fiedler vector genuinely changes sign.
            }
        }
        // On a path the Fiedler vector is monotone, which is what makes it a
        // good linear ordering.
        let v = fiedler_vector(&path_graph(8));
        let increasing = v.windows(2).all(|w| w[0] <= w[1] + 1e-9);
        let decreasing = v.windows(2).all(|w| w[0] >= w[1] - 1e-9);
        assert!(increasing || decreasing, "not monotone on a path: {v:?}");
        // On a barbell the bisection separates the two halves.
        let mut barbell = Graph::new(8, false);
        for (u, v) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            barbell.add_edge(u, v, 1.0);
        }
        for (u, v) in [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)] {
            barbell.add_edge(u, v, 1.0);
        }
        barbell.add_edge(3, 4, 1.0);
        let side = spectral_bisection(&barbell);
        assert!(
            (0..4).all(|i| side[i] == side[0]) && (4..8).all(|i| side[i] == side[4]),
            "bisection did not find the barbell's halves: {side:?}"
        );
        assert_ne!(side[0], side[4], "both halves on the same side");
    }

    /// Spectral clustering must recover a planted partition.
    #[test]
    fn spectral_clustering_recovers_planted_blocks() {
        let mut rng = Rng::new(0x_C1A5);
        // Three cliques joined by single edges: the blocks are unambiguous.
        let mut g = Graph::new(12, false);
        for b in 0..3 {
            for i in 0..4 {
                for j in i + 1..4 {
                    g.add_edge(b * 4 + i, b * 4 + j, 1.0);
                }
            }
        }
        g.add_edge(3, 4, 1.0);
        g.add_edge(7, 8, 1.0);
        let labels = spectral_clustering(&g, 3, &mut rng);
        for b in 0..3 {
            let first = labels[b * 4];
            for i in 1..4 {
                assert_eq!(labels[b * 4 + i], first, "block {b} was split");
            }
        }
        let distinct: std::collections::BTreeSet<usize> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 3, "the three blocks were not separated");
        // k = 1 puts everything together; k = n is one per vertex.
        assert_eq!(spectral_clustering(&g, 1, &mut rng), vec![0; 12]);
        let all = spectral_clustering(&g, 12, &mut rng);
        assert_eq!(all.len(), 12);
    }

    /// The matrix-tree theorem via eigenvalues must agree with the exact
    /// integer determinant.
    #[test]
    fn spectral_spanning_tree_count_matches_the_exact_one() {
        let mut rng = Rng::new(0x_71EE);
        for n in 1..=8usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.5, &mut rng);
                let exact = number_spanning_trees_exact(&g);
                let approx = number_spanning_trees(&g);
                assert!(
                    (approx - exact.to_f64()).abs() < 1e-6 * exact.to_f64().max(1.0),
                    "n = {n}: spectral {approx} vs exact {exact}"
                );
            }
        }
        // Cayley's formula through the spectral route.
        for n in 2..=9u64 {
            let want = (n as f64).powi(n as i32 - 2);
            assert!(
                close(number_spanning_trees(&complete_graph(n as usize)), want),
                "K{n}"
            );
        }
        assert!(close(number_spanning_trees(&petersen_graph()), 2000.0));
        assert!(close(number_spanning_trees(&path_graph(6)), 1.0));
        assert!(close(number_spanning_trees(&cycle_graph(7)), 7.0));
        // Disconnected: none.
        assert_eq!(number_spanning_trees(&Graph::new(4, false)), 0.0);
    }

    // -----------------------------------------------------------------------
    // Centrality
    // -----------------------------------------------------------------------

    /// PageRank must sum to one, and must agree with the stationary
    /// distribution of the walk it describes.
    #[test]
    fn pagerank_sums_to_one_and_is_stationary() {
        let mut rng = Rng::new(0x_9A6E);
        for n in 1..=9usize {
            for _ in 0..12 {
                let mut g = Graph::new(n, true);
                for u in 0..n {
                    for v in 0..n {
                        if u != v && rng.next_f64() < 0.3 {
                            g.add_edge(u, v, 1.0);
                        }
                    }
                }
                let r = pagerank(&g, 0.85, 1e-12);
                let total: f64 = r.iter().sum();
                assert!(close(total, 1.0), "n = {n}: sums to {total}");
                assert!(r.iter().all(|&x| x >= 0.0), "negative rank");

                // Stationary: applying the operator once does not move it.
                let out: Vec<f64> = (0..n)
                    .map(|v| g.adj[v].iter().filter(|&&(t, _)| t != v).count() as f64)
                    .collect();
                let dangling: f64 = (0..n).filter(|&v| out[v] <= 0.0).map(|v| r[v]).sum();
                let mut next = vec![0.15 / n as f64 + 0.85 * dangling / n as f64; n];
                for u in 0..n {
                    if out[u] <= 0.0 {
                        continue;
                    }
                    for &(v, _) in &g.adj[u] {
                        if v != u {
                            next[v] += 0.85 * r[u] / out[u];
                        }
                    }
                }
                for v in 0..n {
                    assert!((next[v] - r[v]).abs() < 1e-6, "not stationary at {v}");
                }
            }
        }
        // On an undirected regular graph every vertex ranks equally.
        for g in [cycle_graph(7), complete_graph(6), petersen_graph()] {
            let r = pagerank(&g, 0.85, 1e-14);
            let first = r[0];
            for (v, &x) in r.iter().enumerate() {
                assert!(close(x, first), "vertex {v} differs on a regular graph");
            }
        }
        // Damping zero is the uniform distribution.
        let r = pagerank(&path_graph(5), 0.0, 1e-14);
        assert!(r.iter().all(|&x| close(x, 0.2)));
    }

    /// The centralities must match their closed forms on graphs where those
    /// are known.
    #[test]
    fn centralities_match_closed_forms_on_named_graphs() {
        // The roadmap's case: the betweenness of a star's centre is
        // (n-1)(n-2)/2, and every leaf is zero.
        for n in 3..=9usize {
            let b = betweenness_centrality(&star_graph(n));
            let want = (n - 1) as f64 * (n - 2) as f64 / 2.0;
            assert!(close(b[0], want), "star {n} centre: {} vs {want}", b[0]);
            for (v, &x) in b.iter().enumerate().skip(1) {
                assert!(x.abs() < 1e-9, "leaf {v} has betweenness {x}");
            }
        }
        // A complete graph has no betweenness at all: every pair is adjacent.
        for n in 2..=7usize {
            let b = betweenness_centrality(&complete_graph(n));
            assert!(b.iter().all(|&x| x.abs() < 1e-9), "K{n} has betweenness");
        }
        // On a path the interior vertex at position i lies on i * (n-1-i)
        // shortest paths.
        for n in 2..=8usize {
            let b = betweenness_centrality(&path_graph(n));
            for i in 0..n {
                let want = (i * (n - 1 - i)) as f64;
                assert!(close(b[i], want), "P{n} at {i}: {} vs {want}", b[i]);
            }
        }
        // Closeness and harmonic on a complete graph: every distance is one.
        for n in 2..=7usize {
            let c = closeness_centrality(&complete_graph(n));
            assert!(c.iter().all(|&x| close(x, 1.0)), "K{n} closeness");
            let h = harmonic_centrality(&complete_graph(n));
            assert!(h.iter().all(|&x| close(x, (n - 1) as f64)), "K{n} harmonic");
        }
        // Harmonic centrality handles disconnection without a special case.
        let mut split = Graph::new(4, false);
        split.add_edge(0, 1, 1.0);
        split.add_edge(2, 3, 1.0);
        let h = harmonic_centrality(&split);
        assert!(h.iter().all(|&x| close(x, 1.0)), "got {h:?}");
        // Eigenvector centrality is uniform on a regular graph.
        for g in [cycle_graph(6), complete_graph(5), petersen_graph()] {
            let e = eigenvector_centrality(&g, 1e-12);
            let first = e[0];
            assert!(e.iter().all(|&x| close(x, first)), "regular graph is not uniform");
            let norm: f64 = e.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(close(norm, 1.0));
        }
    }

    /// Eigenvector centrality must satisfy the eigenvector equation it is
    /// named for.
    #[test]
    fn eigenvector_centrality_solves_its_equation() {
        let mut rng = Rng::new(0x_E16E);
        for n in 2..=8usize {
            for _ in 0..15 {
                let g = random_graph(n, 0.5, &mut rng);
                if !g.is_connected() {
                    continue;
                }
                let x = eigenvector_centrality(&g, 1e-14);
                // A x = lambda x for the Rayleigh quotient lambda.
                let a = g.to_adjacency_matrix();
                let ax: Vec<f64> = (0..n)
                    .map(|i| (0..n).map(|j| a.get(i, j) * x[j]).sum())
                    .collect();
                let lambda: f64 = (0..n).map(|i| x[i] * ax[i]).sum();
                for i in 0..n {
                    assert!(
                        (ax[i] - lambda * x[i]).abs() < 1e-5,
                        "not an eigenvector at {i}"
                    );
                }
                // Perron-Frobenius: non-negative, and lambda is the largest
                // adjacency eigenvalue.
                assert!(x.iter().all(|&v| v >= -1e-9), "negative entry");
                let top = *adjacency_spectrum(&g).last().unwrap();
                assert!(close(lambda, top), "lambda {lambda} vs top {top}");
            }
        }
    }

    /// HITS must produce unit vectors whose mutual recurrence holds.
    #[test]
    fn hits_scores_satisfy_their_recurrence() {
        let mut rng = Rng::new(0x_417);
        for n in 2..=8usize {
            for _ in 0..10 {
                let mut g = Graph::new(n, true);
                for u in 0..n {
                    for v in 0..n {
                        if u != v && rng.next_f64() < 0.4 {
                            g.add_edge(u, v, 1.0);
                        }
                    }
                }
                if g.edge_count() == 0 {
                    continue;
                }
                let (hub, auth) = hits(&g, 1e-14);
                for x in [&hub, &auth] {
                    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                    assert!(close(norm, 1.0) || norm < 1e-12, "not normalized: {norm}");
                    assert!(x.iter().all(|&v| v >= -1e-9), "negative score");
                }
                // A vertex with no in-links has no authority; with no
                // out-links, no hub score.
                for v in 0..n {
                    if g.in_degree(v) == 0 {
                        assert!(auth[v].abs() < 1e-9, "vertex {v} has authority from nothing");
                    }
                    if g.out_degree(v) == 0 {
                        assert!(hub[v].abs() < 1e-9, "vertex {v} hubs nothing");
                    }
                }
            }
        }
        // A pure hub-authority pair: 0 and 1 point at 2 and 3.
        let g = Graph::from_edges(
            4,
            &[(0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (1, 3, 1.0)],
            true,
        );
        let (hub, auth) = hits(&g, 1e-14);
        assert!(hub[0] > 0.5 && hub[1] > 0.5, "0 and 1 should be hubs");
        assert!(hub[2].abs() < 1e-9 && hub[3].abs() < 1e-9);
        assert!(auth[2] > 0.5 && auth[3] > 0.5, "2 and 3 should be authorities");
        assert!(auth[0].abs() < 1e-9 && auth[1].abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Resistance
    // -----------------------------------------------------------------------

    /// Effective resistance must behave like a resistor network: series adds,
    /// parallel halves, and it is a metric.
    #[test]
    fn effective_resistance_behaves_like_a_circuit() {
        // A path of unit resistors in series.
        for n in 2..=8usize {
            let r = effective_resistance(&path_graph(n), 0, n - 1);
            assert!(close(r, (n - 1) as f64), "P{n}: {r}");
        }
        // Two vertices joined by k parallel unit edges: resistance 1/k.
        for k in 1..=5usize {
            let mut g = Graph::new(2, false);
            for _ in 0..k {
                g.add_edge(0, 1, 1.0);
            }
            let r = effective_resistance(&g, 0, 1);
            assert!(close(r, 1.0 / k as f64), "{k} parallel: {r}");
        }
        // A cycle: the two arcs are in parallel, so R = a(n-a)/n.
        for n in 3..=8usize {
            let c = cycle_graph(n);
            for a in 1..n {
                let want = (a * (n - a)) as f64 / n as f64;
                let r = effective_resistance(&c, 0, a);
                assert!(close(r, want), "C{n} at {a}: {r} vs {want}");
            }
        }
        // K_n: every pair has resistance 2/n.
        for n in 2..=8usize {
            let k = complete_graph(n);
            for u in 0..n {
                for v in u + 1..n {
                    let r = effective_resistance(&k, u, v);
                    assert!(close(r, 2.0 / n as f64), "K{n}: {r}");
                }
            }
        }
        // Zero to itself, infinite across components.
        assert_eq!(effective_resistance(&path_graph(4), 2, 2), 0.0);
        let mut split = Graph::new(4, false);
        split.add_edge(0, 1, 1.0);
        split.add_edge(2, 3, 1.0);
        assert!(effective_resistance(&split, 0, 2).is_infinite());

        // A metric: symmetric and satisfying the triangle inequality.
        let mut rng = Rng::new(0x_2E51);
        for n in 2..=7usize {
            for _ in 0..15 {
                let g = random_graph(n, 0.6, &mut rng);
                if !g.is_connected() {
                    continue;
                }
                let m = resistance_matrix(&g);
                for i in 0..n {
                    assert!(m.get(i, i).abs() < 1e-9);
                    for j in 0..n {
                        assert!(close(m.get(i, j), m.get(j, i)), "not symmetric");
                        assert!(m.get(i, j) >= -1e-9, "negative resistance");
                        for k in 0..n {
                            assert!(
                                m.get(i, k) <= m.get(i, j) + m.get(j, k) + 1e-6,
                                "triangle inequality fails at ({i}, {j}, {k})"
                            );
                        }
                    }
                }
                // Foster's theorem: the resistances over the edges sum to
                // n - 1. That is a global identity no single pair can fake.
                let foster: f64 = g.edges().iter().map(|&(u, v, _)| m.get(u, v)).sum();
                assert!(
                    close(foster, (n - 1) as f64),
                    "Foster's theorem: {foster} vs {}",
                    n - 1
                );
            }
        }
    }

    /// Commute time is 2m times the effective resistance, and matches a
    /// direct absorbing-walk computation on small graphs.
    #[test]
    fn commute_time_matches_the_resistance_identity() {
        for g in [path_graph(5), cycle_graph(6), complete_graph(5), petersen_graph()] {
            let m: f64 = g.edges().iter().map(|&(_, _, w)| w).sum();
            for u in 0..g.n {
                for v in 0..g.n {
                    let c = commute_time(&g, u, v);
                    let r = effective_resistance(&g, u, v);
                    assert!(close(c, 2.0 * m * r), "commute {c} vs 2mR {}", 2.0 * m * r);
                }
            }
        }
        // On a path the commute time between the ends is 2 * (n-1)^2 for the
        // n - 1 unit edges.
        for n in 2..=7usize {
            let c = commute_time(&path_graph(n), 0, n - 1);
            let want = 2.0 * (n - 1) as f64 * (n - 1) as f64;
            assert!(close(c, want), "P{n}: {c} vs {want}");
        }
    }

    /// The stationary distribution is the degree distribution, and really is
    /// stationary under the walk.
    #[test]
    fn random_walk_stationary_is_the_degree_distribution() {
        let mut rng = Rng::new(0x_2A1C);
        for n in 2..=9usize {
            for _ in 0..15 {
                let g = random_graph(n, 0.5, &mut rng);
                if g.edge_count() == 0 {
                    continue;
                }
                let pi = random_walk_stationary(&g);
                assert!(close(pi.iter().sum::<f64>(), 1.0), "does not sum to one");
                // pi P = pi, where P(u, v) = w(u,v) / deg(u).
                let deg = weighted_degrees(&g);
                let mut next = vec![0.0; n];
                for u in 0..n {
                    if deg[u] <= 0.0 {
                        continue;
                    }
                    for &(v, w) in &g.adj[u] {
                        if u != v {
                            next[v] += pi[u] * w / deg[u];
                        }
                    }
                }
                for v in 0..n {
                    if deg[v] > 0.0 {
                        assert!((next[v] - pi[v]).abs() < 1e-9, "not stationary at {v}");
                    }
                }
            }
        }
        // Regular graphs are uniform.
        for g in [cycle_graph(8), complete_graph(6), petersen_graph()] {
            let pi = random_walk_stationary(&g);
            assert!(pi.iter().all(|&x| close(x, 1.0 / g.n as f64)));
        }
    }

    /// Cheeger's inequality must bracket the true conductance.
    #[test]
    fn cheeger_bounds_bracket_the_true_conductance() {
        let mut rng = Rng::new(0x_C4EE);
        for n in 2..=7usize {
            for _ in 0..15 {
                let g = random_graph(n, 0.5, &mut rng);
                if !g.is_connected() || g.edge_count() == 0 {
                    continue;
                }
                let (lo, hi) = cheeger_bound(&g);
                let h = brute_conductance(&g);
                assert!(
                    h >= lo - 1e-7 && h <= hi + 1e-7,
                    "n = {n}: conductance {h} outside [{lo}, {hi}]"
                );
            }
        }
        // A complete graph is a good expander; a path is not.
        assert!(expander_check(&complete_graph(8), 4.0));
        assert!(!expander_check(&path_graph(8), 1.0));
        assert!(!expander_check(&Graph::new(4, false), 0.1), "disconnected");
    }

    /// The exact conductance, minimised over every vertex subset.
    fn brute_conductance(g: &Graph) -> f64 {
        let n = g.n;
        let deg = weighted_degrees(g);
        let total: f64 = deg.iter().sum();
        let mut best = f64::INFINITY;
        for mask in 1u64..(1u64 << n) - 1 {
            let inside: Vec<bool> = (0..n).map(|v| mask >> v & 1 == 1).collect();
            let vol: f64 = (0..n).filter(|&v| inside[v]).map(|v| deg[v]).sum();
            let vol_c = total - vol;
            if vol <= 0.0 || vol_c <= 0.0 {
                continue;
            }
            let cut: f64 = g
                .edges()
                .iter()
                .filter(|&&(u, v, _)| inside[u] != inside[v])
                .map(|&(_, _, w)| w)
                .sum();
            best = best.min(cut / vol.min(vol_c));
        }
        best
    }

    /// Mixing time is finite on a connected non-bipartite graph and infinite
    /// where the walk does not converge.
    #[test]
    fn mixing_time_is_finite_exactly_when_the_walk_converges() {
        // Connected and non-bipartite: finite.
        for g in [complete_graph(6), cycle_graph(7), petersen_graph()] {
            let t = mixing_time_estimate(&g, 0.01);
            assert!(t.is_finite() && t > 0.0, "expected finite, got {t}");
        }
        // Bipartite: the walk alternates sides forever and never mixes.
        for g in [cycle_graph(6), path_graph(5), complete_bipartite(3, 3)] {
            assert!(
                mixing_time_estimate(&g, 0.01).is_infinite(),
                "a bipartite walk should not mix"
            );
        }
        // Disconnected: an isolated vertex has zero stationary mass.
        let mut split = Graph::new(4, false);
        split.add_edge(0, 1, 1.0);
        assert!(mixing_time_estimate(&split, 0.01).is_infinite());
        // A better-connected graph mixes faster.
        let fast = mixing_time_estimate(&complete_graph(10), 0.01);
        let slow = mixing_time_estimate(&cycle_graph(11), 0.01);
        assert!(fast < slow, "K10 ({fast}) should mix faster than C11 ({slow})");
    }

    /// Spectral invariants: energy, the Estrada index, and isospectrality.
    #[test]
    fn spectral_invariants_match_their_definitions() {
        // K_n: eigenvalues n-1 once and -1 with multiplicity n-1, so the
        // energy is 2(n-1).
        for n in 2..=8usize {
            let e = graph_energy(&complete_graph(n));
            assert!(close(e, 2.0 * (n - 1) as f64), "K{n} energy: {e}");
        }
        // The adjacency trace is zero, so the eigenvalues sum to zero.
        let mut rng = Rng::new(0x_5AEC);
        for n in 1..=8usize {
            for _ in 0..15 {
                let g = random_graph(n, 0.5, &mut rng);
                let s = adjacency_spectrum(&g);
                assert!(s.iter().sum::<f64>().abs() < 1e-9, "trace is not zero");
                // The sum of squares is twice the edge count, since it is the
                // trace of A^2 and counts closed walks of length two.
                let sq: f64 = s.iter().map(|x| x * x).sum();
                assert!(close(sq, 2.0 * g.edge_count() as f64), "trace of A^2");
                // The Estrada index is the trace of exp(A), which is at least
                // n by the arithmetic-geometric mean.
                let est = estrada_index(&g);
                assert!(est >= n as f64 - 1e-9, "Estrada {est} below {n}");
            }
        }
        // Isomorphic graphs are isospectral.
        let mut rng = Rng::new(0x_1505);
        for n in 1..=7usize {
            let g = random_graph(n, 0.5, &mut rng);
            let perm = crate::discrete::combinatorics::random_permutation(n, &mut rng);
            let mut h = Graph::new(n, false);
            for (u, v, w) in g.edges() {
                h.add_edge(perm[u], perm[v], w);
            }
            assert!(isospectral_check(&g, &h, 1e-9), "relabelling changed the spectrum");
        }
        // The classic cospectral pair that is not isomorphic: K_{1,4} and
        // C4 plus an isolated vertex both have spectrum {-2, 0, 0, 0, 2}.
        let star = star_graph(5);
        let mut c4_plus = Graph::new(5, false);
        for (u, v) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            c4_plus.add_edge(u, v, 1.0);
        }
        assert!(
            isospectral_check(&star, &c4_plus, 1e-9),
            "the classic cospectral pair should match"
        );
        assert!(
            !crate::graph::core::is_isomorphic_small(&star, &c4_plus),
            "but they are not isomorphic"
        );
        // Different sizes are refused.
        assert!(!isospectral_check(&complete_graph(3), &complete_graph(4), 1e-9));
    }

    // -----------------------------------------------------------------------
    // Communities
    // -----------------------------------------------------------------------

    /// Modularity must match its definition computed directly, and be zero for
    /// the single-community partition.
    #[test]
    fn modularity_matches_its_definition() {
        let mut rng = Rng::new(0x_10D);
        for n in 2..=8usize {
            for _ in 0..20 {
                let g = random_graph(n, 0.5, &mut rng);
                if g.edge_count() == 0 {
                    continue;
                }
                let labels: Vec<usize> = (0..n)
                    .map(|_| ((u128::from(rng.next_u64()) * 3) >> 64) as usize)
                    .collect();
                let q = modularity(&g, &labels);
                // Direct: sum over pairs of (A_ij - k_i k_j / 2m) / 2m.
                let a = g.to_adjacency_matrix();
                let deg = weighted_degrees(&g);
                let two_m: f64 = deg.iter().sum();
                let mut want = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        if labels[i] == labels[j] {
                            want += a.get(i, j) - deg[i] * deg[j] / two_m;
                        }
                    }
                }
                want /= two_m;
                assert!(close(q, want), "n = {n}: {q} vs {want}");
                // Bounded above by one.
                assert!(q <= 1.0 + 1e-9, "modularity {q} exceeds one");
            }
        }
        // Everything in one community: exactly zero.
        for g in [complete_graph(6), cycle_graph(7), petersen_graph()] {
            assert!(close(modularity(&g, &vec![0; g.n]), 0.0));
        }
    }

    /// Modularity has to count a self-loop, and count it twice, or Louvain's
    /// contraction step loses the edge weight it has just folded away.
    ///
    /// Contracting a partition into one vertex per community, with each
    /// community's internal weight becoming a loop, must leave modularity
    /// unchanged: it is the same sum of the same terms, regrouped. That
    /// identity is what makes the recursion legitimate, and it fails outright
    /// if loops are dropped from either the degrees or the internal weight.
    #[test]
    fn modularity_survives_contracting_a_partition() {
        let mut rng = Rng::new(0x_C047);
        for _ in 0..80 {
            let n = 2 + ((u128::from(rng.next_u64()) * 8) >> 64) as usize;
            let g = random_graph(n, 0.4, &mut rng);
            if g.edge_count() == 0 {
                continue;
            }
            let k = 1 + ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize;
            let labels: Vec<usize> = (0..n)
                .map(|_| ((u128::from(rng.next_u64()) * k as u128) >> 64) as usize)
                .collect();
            let before = modularity(&g, &labels);

            // Contract, exactly as Louvain does.
            let compact = renumber(&labels);
            let c = compact.iter().copied().max().unwrap() + 1;
            let mut merged: std::collections::BTreeMap<(usize, usize), f64> =
                std::collections::BTreeMap::new();
            for (u, v, w) in g.edges() {
                let (a, b) = (compact[u], compact[v]);
                *merged.entry((a.min(b), a.max(b))).or_insert(0.0) += w;
            }
            let mut h = Graph::new(c, false);
            for ((a, b), w) in merged {
                h.add_edge(a, b, w);
            }
            let after = modularity(&h, &(0..c).collect::<Vec<_>>());
            assert!(
                close(before, after),
                "contraction changed modularity: {before} to {after}"
            );
        }

        // And the plain statement the above rests on: a loop is internal
        // weight and it counts twice, so adding one to a community raises
        // that community's share of the total.
        let mut g = Graph::new(2, false);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 0, 3.0);
        // Degrees are 2*3 + 1 = 7 and 1, so 2m = 8; both vertices in one
        // community gives inside = 2*(1 + 3) = 8 and expected = 1.
        assert!(close(modularity(&g, &[0, 0]), 0.0));
        // Split them: only the loop is internal, so inside = 6 of 8, against
        // (7/8)^2 + (1/8)^2 of expected weight.
        let split = 6.0 / 8.0 - ((7.0 / 8.0f64).powi(2) + (1.0 / 8.0f64).powi(2));
        assert!(close(modularity(&g, &[0, 1]), split), "loop weight is not counted");
    }

    /// Louvain and label propagation must both find the planted partition of
    /// a graph with obvious communities, and Louvain must beat the trivial
    /// partition on modularity.
    #[test]
    fn community_detection_recovers_a_planted_partition() {
        let mut rng = Rng::new(0x_10AF);
        // Four cliques of five, joined by one edge each.
        let mut g = Graph::new(20, false);
        for b in 0..4 {
            for i in 0..5 {
                for j in i + 1..5 {
                    g.add_edge(b * 5 + i, b * 5 + j, 1.0);
                }
            }
        }
        for b in 0..3 {
            g.add_edge(b * 5 + 4, (b + 1) * 5, 1.0);
        }

        let louvain = community_louvain(&g, &mut rng);
        for b in 0..4 {
            let first = louvain[b * 5];
            for i in 1..5 {
                assert_eq!(louvain[b * 5 + i], first, "Louvain split block {b}");
            }
        }
        let q = modularity(&g, &louvain);
        assert!(q > 0.6, "Louvain modularity {q} is too low for a planted partition");
        assert!(q > modularity(&g, &[0; 20]), "worse than one community");

        let lp = label_propagation(&g, &mut rng);
        for b in 0..4 {
            let first = lp[b * 5];
            for i in 1..5 {
                assert_eq!(lp[b * 5 + i], first, "label propagation split block {b}");
            }
        }
        assert!(modularity(&g, &lp) > 0.6);

        // Labels are renumbered from zero with no gaps.
        for labels in [&louvain, &lp] {
            let distinct: std::collections::BTreeSet<usize> = labels.iter().copied().collect();
            assert_eq!(
                distinct,
                (0..distinct.len()).collect::<std::collections::BTreeSet<_>>(),
                "labels are not compact"
            );
        }

        // On a complete graph there is no community structure to find, so a
        // single community is optimal.
        let k = complete_graph(8);
        let single = community_louvain(&k, &mut rng);
        assert!(
            modularity(&k, &single) <= 1e-9,
            "K8 should have no positive-modularity partition"
        );
    }
}
