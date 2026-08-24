//! Properties for `graph::flow` and `graph::matching`.
//!
//! Each property pits an algorithm against either a different algorithm for
//! the same quantity or the definition the quantity is given by.

use rust_physics_engine::discrete::combinatorics::{combinations_iter, is_permutation};
use rust_physics_engine::graph::core::Graph;
use rust_physics_engine::graph::flow::{
    cut_capacity, edge_disjoint_paths, global_min_cut_stoer_wagner,
    max_bipartite_matching_via_flow, max_flow_dinic, max_flow_push_relabel,
    min_cost_max_flow, min_cut, vertex_disjoint_paths,
};
use rust_physics_engine::graph::matching::{
    blossom_max_matching, hopcroft_karp, hungarian, konig_vertex_cover, matching_size,
    maximum_weight_bipartite, stable_marriage,
};
use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;

/// A value in `0..n` from the high bits: `% n` reads the low bits of the
/// linear congruential generator, where bit `b` has period `2^(b+1)`.
fn pick(rng: &mut Rng, n: u64) -> u64 {
    ((u128::from(rng.next_u64()) * u128::from(n)) >> 64) as u64
}

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

/// The roadmap's headline property: max-flow equals min-cut, on random
/// networks, with two independent flow algorithms agreeing.
#[test]
fn prop_max_flow_equals_min_cut() {
    let mut rng = Rng::new(0x_F10A);
    for _ in 0..120 {
        let n = 2 + pick(&mut rng, 8) as usize;
        let directed = rng.next_f64() < 0.5;
        let g = random_network(n, 0.25 + 0.5 * rng.next_f64(), directed, &mut rng);
        let s = pick(&mut rng, n as u64) as usize;
        let mut t = pick(&mut rng, n as u64) as usize;
        if s == t {
            t = (t + 1) % n;
        }
        let (value, m) = max_flow_dinic(&g, s, t);
        assert!(
            close(value, max_flow_push_relabel(&g, s, t)),
            "dinic and push-relabel disagree"
        );
        let (cut, side) = min_cut(&g, s, t);
        assert!(close(value, cut), "flow {value} vs cut {cut}");
        assert!(side[s] && !side[t], "the cut does not separate s from t");
        assert!(close(cut, cut_capacity(&g, &side)));

        // The flow conserves at every interior vertex and respects capacity.
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
                assert!(m[u][v] <= cap[u][v] + 1e-6, "capacity violated on ({u}, {v})");
                assert!(m[u][v] >= -1e-9);
            }
        }
        for v in 0..n {
            if v == s || v == t {
                continue;
            }
            let inn: f64 = (0..n).map(|u| m[u][v]).sum();
            let out: f64 = (0..n).map(|w| m[v][w]).sum();
            assert!(close(inn, out), "vertex {v} leaks");
        }
        // Min-cost max-flow moves the same amount whatever the costs are.
        let costs: Vec<f64> = g.edges().iter().map(|_| (9.0 * rng.next_f64()).floor()).collect();
        let (mcmf, _) = min_cost_max_flow(&g, &costs, s, t);
        assert!(close(mcmf, value), "mcmf {mcmf} vs max flow {value}");
    }
}

/// The global minimum cut must equal the best over all s-t pairs.
#[test]
fn prop_stoer_wagner_is_the_global_minimum() {
    let mut rng = Rng::new(0x_570E);
    for _ in 0..60 {
        let n = 2 + pick(&mut rng, 7) as usize;
        let g = random_network(n, 0.3 + 0.5 * rng.next_f64(), false, &mut rng);
        let (global, side) = global_min_cut_stoer_wagner(&g);
        let mut best = f64::INFINITY;
        for s in 0..n {
            for t in s + 1..n {
                best = best.min(min_cut(&g, s, t).0);
            }
        }
        assert!(close(global, best), "global {global} vs best s-t {best}");
        assert!(!side.is_empty() && side.len() < n, "not a proper cut");
        let flags: Vec<bool> = (0..n).map(|v| side.contains(&v)).collect();
        assert!(close(global, cut_capacity(&g, &flags)));
    }
}

/// Menger's theorem in the edge form, against direct removal search.
#[test]
fn prop_edge_menger_matches_removal() {
    let mut rng = Rng::new(0x_3E17);
    for _ in 0..40 {
        let n = 2 + pick(&mut rng, 5) as usize;
        let g = random_network(n, 0.3 + 0.3 * rng.next_f64(), false, &mut rng);
        for s in 0..n {
            for t in 0..n {
                if s == t {
                    continue;
                }
                let k = edge_disjoint_paths(&g, s, t);
                // The fewest edges whose removal separates them.
                let edges = g.edges();
                let mut cut = edges.len();
                'outer: for r in 0..=edges.len() {
                    for combo in combinations_iter(edges.len(), r) {
                        let mut h = Graph::new(n, false);
                        for (i, &(u, v, _)) in edges.iter().enumerate() {
                            if !combo.contains(&i) && u != v {
                                h.add_edge(u, v, 1.0);
                            }
                        }
                        if h.bfs(s)[t].is_none() {
                            cut = r;
                            break 'outer;
                        }
                    }
                }
                assert_eq!(k, cut, "edge Menger at {s}->{t}");
                // The vertex form never exceeds the edge form.
                assert!(vertex_disjoint_paths(&g, s, t) <= k);
            }
        }
    }
}

/// Hopcroft-Karp, the flow reduction, and blossom must all agree on the size
/// of a maximum matching of a bipartite graph -- three different algorithms
/// for the same number.
#[test]
fn prop_three_matching_algorithms_agree_on_bipartite() {
    let mut rng = Rng::new(0x_4C41);
    for _ in 0..200 {
        let l = 1 + pick(&mut rng, 6) as usize;
        let r = 1 + pick(&mut rng, 6) as usize;
        let mut edges = Vec::new();
        let mut g = Graph::new(l + r, false);
        for a in 0..l {
            for b in 0..r {
                if rng.next_f64() < 0.4 {
                    edges.push((a, b));
                    g.add_edge(a, l + b, 1.0);
                }
            }
        }
        let hk = hopcroft_karp(l, r, &edges);
        let hk_size = hk.iter().filter(|x| x.is_some()).count();
        let left: Vec<usize> = (0..l).collect();
        let flow_size = matching_size(&max_bipartite_matching_via_flow(&g, &left));
        let blossom_size = matching_size(&blossom_max_matching(&g));
        assert_eq!(hk_size, flow_size, "Hopcroft-Karp vs flow");
        assert_eq!(hk_size, blossom_size, "Hopcroft-Karp vs blossom");

        // Konig: the minimum vertex cover has exactly that size.
        let m = max_bipartite_matching_via_flow(&g, &left);
        let cover = konig_vertex_cover(&g, &left, &m);
        assert_eq!(cover.len(), hk_size, "Konig's theorem");
        for (u, v, _) in g.edges() {
            assert!(cover.contains(&u) || cover.contains(&v), "edge uncovered");
        }
    }
}

/// Blossom matching must be a valid matching and maximum, checked on general
/// graphs where the odd cycles are the whole difficulty.
#[test]
fn prop_blossom_is_valid_and_maximum() {
    let mut rng = Rng::new(0x_B105);
    for _ in 0..120 {
        let n = 1 + pick(&mut rng, 9) as usize;
        let g = random_network(n, 0.2 + 0.4 * rng.next_f64(), false, &mut rng);
        let m = blossom_max_matching(&g);
        for v in 0..n {
            if let Some(w) = m[v] {
                assert_eq!(m[w], Some(v), "asymmetric at {v}");
                assert!(g.adj[v].iter().any(|&(x, _)| x == w), "matched a non-edge");
            }
        }
        // Maximum: by Berge's lemma, a matching is maximum exactly when no
        // augmenting path exists. Checking that directly is a different
        // statement from the algorithm's own search.
        assert!(
            !has_augmenting_path(&g, &m),
            "an augmenting path remains, so the matching is not maximum"
        );
    }
}

/// Berge's lemma test: is there a path between two unmatched vertices whose
/// edges alternate out of and into the matching?
fn has_augmenting_path(g: &Graph, m: &[Option<usize>]) -> bool {
    let n = g.n;
    for start in 0..n {
        if m[start].is_some() {
            continue;
        }
        // Depth-first over alternating walks, tracking the parity of the step.
        let mut stack = vec![(start, false, vec![start])];
        while let Some((v, need_matched, path)) = stack.pop() {
            for &(w, _) in &g.adj[v] {
                if path.contains(&w) {
                    continue;
                }
                let is_matched = m[v] == Some(w);
                if is_matched != need_matched {
                    continue;
                }
                if !need_matched && m[w].is_none() && w != start {
                    return true;
                }
                let mut next = path.clone();
                next.push(w);
                stack.push((w, !need_matched, next));
            }
        }
    }
    false
}

/// The Hungarian algorithm must find an optimal assignment, and the weighted
/// bipartite routine built on it must never lose to any alternative.
#[test]
fn prop_assignment_is_optimal() {
    let mut rng = Rng::new(0x_4055);
    for _ in 0..80 {
        let n = 1 + pick(&mut rng, 6) as usize;
        let mut cost = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                cost.set(i, j, (20.0 * rng.next_f64() - 5.0).round());
            }
        }
        let (total, assign) = hungarian(&cost);
        assert!(is_permutation(&assign), "not a permutation: {assign:?}");
        let actual: f64 = (0..n).map(|i| cost.get(i, assign[i])).sum();
        assert!(close(total, actual), "reported {total} but costs {actual}");
        let best = rust_physics_engine::discrete::combinatorics::permutations_iter(
            &(0..n).collect::<Vec<_>>(),
        )
        .map(|p| (0..n).map(|i| cost.get(i, p[i])).sum::<f64>())
        .fold(f64::INFINITY, f64::min);
        assert!(close(total, best), "n = {n}: {total} vs brute {best}");

        // Weighted bipartite: never worse than any single valid matching.
        let mut w = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                w.set(i, j, (12.0 * rng.next_f64() - 4.0).round().max(0.0));
            }
        }
        let (got, partner) = maximum_weight_bipartite(&w);
        let mut seen = std::collections::BTreeSet::new();
        let mut sum = 0.0;
        for (i, &p) in partner.iter().enumerate() {
            if let Some(j) = p {
                assert!(seen.insert(j), "column {j} used twice");
                sum += w.get(i, j);
            }
        }
        assert!(close(got, sum), "reported {got} but sums to {sum}");
        for p in rust_physics_engine::discrete::combinatorics::permutations_iter(
            &(0..n).collect::<Vec<_>>(),
        ) {
            let alt: f64 = (0..n).map(|i| w.get(i, p[i])).sum();
            assert!(got >= alt - 1e-9, "an alternative matching scores {alt} > {got}");
        }
    }
}

/// Gale-Shapley must always produce a stable matching.
#[test]
fn prop_gale_shapley_is_always_stable() {
    let mut rng = Rng::new(0x_6A1E);
    for _ in 0..200 {
        let n = 1 + pick(&mut rng, 7) as usize;
        let prefs_a: Vec<Vec<usize>> = (0..n)
            .map(|_| rust_physics_engine::discrete::combinatorics::random_permutation(n, &mut rng))
            .collect();
        let prefs_b: Vec<Vec<usize>> = (0..n)
            .map(|_| rust_physics_engine::discrete::combinatorics::random_permutation(n, &mut rng))
            .collect();
        let m = stable_marriage(&prefs_a, &prefs_b);
        assert!(is_permutation(&m), "not a perfect matching");
        // No blocking pair: nobody prefers someone who also prefers them.
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
                let blocking = rank(&prefs_a[i], j) < rank(&prefs_a[i], m[i])
                    && rank(&prefs_b[j], i) < rank(&prefs_b[j], partner_b[j]);
                assert!(!blocking, "({i}, {j}) is a blocking pair");
            }
        }
    }
}
