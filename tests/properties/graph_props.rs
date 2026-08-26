//! Properties for `graph::core` and `graph::paths`.
//!
//! Randomized cross-checks between algorithms that must agree, and between an
//! algorithm and the definition it implements.

use rust_physics_engine::discrete::disjoint_set::DisjointSet;
use rust_physics_engine::exact::bigint::BigInt;
use rust_physics_engine::graph::core::{
    complete_bipartite, complete_graph, cycle_graph, hypercube_graph, is_isomorphic_small,
    path_graph, spanning_tree_count_exact, Graph,
};
use rust_physics_engine::graph::paths::{
    bellman_ford, bidirectional_dijkstra, chinese_postman, dijkstra, floyd_warshall, johnson,
    minimum_spanning_tree_boruvka, minimum_spanning_tree_kruskal, minimum_spanning_tree_prim,
    tour_length, traveling_salesman_exact, transitive_closure, tsp_2opt, tsp_christofides,
    tsp_nearest_neighbor,
};
use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;

/// A value in `0..n` from the high bits: `% n` reads the low bits of the
/// linear congruential generator, where bit `b` has period `2^(b+1)`.
fn pick(rng: &mut Rng, n: u64) -> u64 {
    ((u128::from(rng.next_u64()) * u128::from(n)) >> 64) as u64
}

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

/// The roadmap's headline property: Dijkstra, Bellman-Ford, Floyd-Warshall and
/// Johnson must agree on every pair of every random graph.
#[test]
fn prop_shortest_path_algorithms_agree() {
    let mut rng = Rng::new(0x_5A07);
    for _ in 0..120 {
        let n = 1 + pick(&mut rng, 10) as usize;
        let directed = rng.next_f64() < 0.5;
        let g = random_weighted(n, 0.2 + 0.5 * rng.next_f64(), directed, &mut rng);
        let fw = floyd_warshall(&g);
        let jn = johnson(&g).expect("weights are positive here");
        for s in 0..n {
            let (dj, _) = dijkstra(&g, s);
            let (bf, _) = bellman_ford(&g, s).expect("no negative cycle");
            for t in 0..n {
                assert!(close(dj[t], bf[t]), "dijkstra vs bellman-ford {s}->{t}");
                assert!(close(dj[t], fw.get(s, t)), "dijkstra vs floyd {s}->{t}");
                assert!(close(dj[t], jn.get(s, t)), "dijkstra vs johnson {s}->{t}");
                // Bidirectional search must agree too.
                match bidirectional_dijkstra(&g, s, t) {
                    Some((len, _)) => assert!(close(len, dj[t]), "bidirectional {s}->{t}"),
                    None => assert!(!dj[t].is_finite()),
                }
            }
            // The triangle inequality holds for shortest paths by definition.
            for t in 0..n {
                for u in 0..n {
                    if dj[t].is_finite() && fw.get(t, u).is_finite() {
                        assert!(
                            dj[u] <= dj[t] + fw.get(t, u) + 1e-9,
                            "triangle inequality violated at {s}, {t}, {u}"
                        );
                    }
                }
            }
        }
    }
}

/// The three minimum spanning tree algorithms must agree on weight, and each
/// result must be an acyclic spanning forest.
#[test]
fn prop_mst_algorithms_agree_and_produce_forests() {
    let mut rng = Rng::new(0x_1457);
    for _ in 0..200 {
        let n = 1 + pick(&mut rng, 14) as usize;
        let g = random_weighted(n, 0.15 + 0.5 * rng.next_f64(), false, &mut rng);
        let (wk, ek) = minimum_spanning_tree_kruskal(&g);
        let (wp, ep) = minimum_spanning_tree_prim(&g);
        let (wb, eb) = minimum_spanning_tree_boruvka(&g);
        assert!(close(wk, wp), "kruskal {wk} vs prim {wp}");
        assert!(close(wk, wb), "kruskal {wk} vs boruvka {wb}");
        let components = g.connected_components().len();
        for (name, edges) in [("kruskal", &ek), ("prim", &ep), ("boruvka", &eb)] {
            assert_eq!(edges.len(), n - components, "{name} edge count");
            let mut ds = DisjointSet::new(n);
            for &(u, v) in edges {
                assert!(ds.union(u, v), "{name} produced a cycle");
            }
            assert_eq!(ds.count(), components, "{name} does not span");
        }
        // The cut property: every MST edge is the cheapest across some cut it
        // defines, so removing it and reconnecting cannot be cheaper.
        for &(u, v) in &ek {
            let w = g
                .adj[u]
                .iter()
                .filter(|&&(t, _)| t == v)
                .map(|&(_, w)| w)
                .fold(f64::INFINITY, f64::min);
            // The side of the cut reachable without this edge.
            let mut ds = DisjointSet::new(n);
            for &(a, b) in ek.iter().filter(|&&e| e != (u, v)) {
                ds.union(a, b);
            }
            for (a, b, aw) in g.edges() {
                if ds.connected(a, u) != ds.connected(b, u) {
                    assert!(aw >= w - 1e-9, "a cheaper edge crosses the same cut");
                }
            }
        }
    }
}

/// Strongly connected components must be exactly the classes of mutual
/// reachability.
#[test]
fn prop_scc_matches_mutual_reachability() {
    let mut rng = Rng::new(0x_05CC);
    for _ in 0..200 {
        let n = 1 + pick(&mut rng, 12) as usize;
        let g = random_weighted(n, 0.1 + 0.3 * rng.next_f64(), true, &mut rng);
        let r = transitive_closure(&g);
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
                assert_eq!(label[i] == label[j], r[i][j] && r[j][i], "({i}, {j})");
            }
        }
        // The condensation is always a DAG.
        let (cond, cl) = g.condensation();
        assert_eq!(cl, label);
        assert!(cond.n <= 1 || cond.is_dag());
    }
}

/// Bridges and articulation points must match direct removal-and-recount.
#[test]
fn prop_bridges_and_cut_vertices_match_removal() {
    let mut rng = Rng::new(0x_B21D);
    for _ in 0..120 {
        let n = 2 + pick(&mut rng, 8) as usize;
        let g = random_weighted(n, 0.2 + 0.3 * rng.next_f64(), false, &mut rng);
        let base = g.connected_components().len();

        let found: Vec<(usize, usize)> = g.bridges();
        let all = g.edges();
        let mut expected = Vec::new();
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
            if h.connected_components().len() > base {
                expected.push((u.min(v), u.max(v)));
            }
        }
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(found, expected, "bridges disagree at n = {n}");

        let cuts = g.articulation_points();
        for v in 0..n {
            let rest: Vec<usize> = (0..n).filter(|&x| x != v).collect();
            let after = g.subgraph(&rest).connected_components().len();
            let before = if g.degree(v) == 0 { base - 1 } else { base };
            assert_eq!(
                cuts.contains(&v),
                after > before,
                "cut vertex disagreement at {v}"
            );
        }
    }
}

/// Held-Karp must be optimal: no local-search tour can beat it, and every
/// heuristic must land between it and its own starting tour.
#[test]
fn prop_held_karp_bounds_the_heuristics() {
    let mut rng = Rng::new(0x_7595);
    for _ in 0..60 {
        let n = 3 + pick(&mut rng, 7) as usize;
        let d = random_metric(n, &mut rng);
        let (opt, tour) = traveling_salesman_exact(&d);
        assert!(close(opt, tour_length(&d, &tour)));

        let (nn, nn_tour) = tsp_nearest_neighbor(&d);
        assert!(nn >= opt - 1e-9, "nearest neighbour beat the optimum");
        let (two, _) = tsp_2opt(&d, &nn_tour);
        assert!(two >= opt - 1e-9, "2-opt beat the optimum");
        assert!(two <= nn + 1e-9, "2-opt made it worse");

        // Christofides' 1.5 guarantee holds on a metric instance.
        let (ch, ch_tour) = tsp_christofides(&d).expect("small odd set");
        assert_eq!(ch_tour.len(), n);
        assert!(ch >= opt - 1e-9, "Christofides beat the optimum");
        assert!(ch <= 1.5 * opt + 1e-9, "Christofides {ch} exceeds 1.5 x {opt}");
    }
}

/// Kirchhoff's matrix-tree theorem against direct enumeration of spanning
/// trees on small graphs, and against the closed forms on the named families.
#[test]
fn prop_matrix_tree_matches_enumeration() {
    let mut rng = Rng::new(0x_C417);
    for _ in 0..80 {
        let n = 1 + pick(&mut rng, 6) as usize;
        let g = random_weighted(n, 0.3 + 0.5 * rng.next_f64(), false, &mut rng);
        let exact = spanning_tree_count_exact(&g);
        // Enumerate: every edge subset of size n - 1 that is acyclic and
        // spanning is a spanning tree.
        let edges: Vec<(usize, usize, f64)> =
            g.edges().into_iter().filter(|&(u, v, _)| u != v).collect();
        let mut brute = 0u64;
        if n >= 1 && edges.len() >= n.saturating_sub(1) {
            for combo in rust_physics_engine::discrete::combinatorics::combinations_iter(
                edges.len(),
                n - 1,
            ) {
                let mut ds = DisjointSet::new(n);
                let mut ok = true;
                for &i in &combo {
                    let (u, v, _) = edges[i];
                    if !ds.union(u, v) {
                        ok = false;
                        break;
                    }
                }
                if ok && ds.count() == 1 {
                    brute += 1;
                }
            }
        }
        if n == 1 {
            brute = 1;
        }
        assert_eq!(exact, BigInt::from_u64(brute), "n = {n}");
    }
    // Cayley's formula, past where an f64 determinant would stay exact.
    for n in 1..=14u64 {
        let want = if n <= 2 {
            BigInt::one()
        } else {
            BigInt::from_u64(n).pow(n - 2)
        };
        assert_eq!(spanning_tree_count_exact(&complete_graph(n as usize)), want);
    }
    // K_{m,n} has m^(n-1) n^(m-1).
    for m in 1..=5u64 {
        for n in 1..=5u64 {
            let want = BigInt::from_u64(m)
                .pow(n - 1)
                .mul(&BigInt::from_u64(n).pow(m - 1));
            assert_eq!(
                spanning_tree_count_exact(&complete_bipartite(m as usize, n as usize)),
                want
            );
        }
    }
    // A tree has one; a cycle has n.
    for n in 3..=10usize {
        assert_eq!(spanning_tree_count_exact(&path_graph(n)), BigInt::one());
        assert_eq!(
            spanning_tree_count_exact(&cycle_graph(n)),
            BigInt::from_u64(n as u64)
        );
    }
}

/// The Chinese postman route must cross every edge and cost at least the total
/// edge weight, with equality exactly when every degree is even.
#[test]
fn prop_chinese_postman_covers_and_bounds() {
    let mut rng = Rng::new(0x_CB05);
    for _ in 0..80 {
        let n = 2 + pick(&mut rng, 7) as usize;
        let g = random_weighted(n, 0.4 + 0.4 * rng.next_f64(), false, &mut rng);
        if !g.is_connected() || g.edge_count() == 0 {
            continue;
        }
        let total: f64 = g.edges().iter().map(|&(_, _, w)| w).sum();
        let Some((cost, walk)) = chinese_postman(&g) else {
            continue;
        };
        assert!(cost >= total - 1e-9, "cost is below the total edge weight");
        let all_even = (0..n).all(|v| g.degree(v).is_multiple_of(2));
        if all_even {
            assert!(close(cost, total), "even degrees should cost exactly the total");
        } else {
            assert!(cost > total + 1e-9, "odd degrees must force a repeat");
        }
        // Closed, and every step is an edge, and every edge is crossed.
        assert_eq!(walk[0], *walk.last().unwrap());
        let mut used = std::collections::BTreeSet::new();
        for w in walk.windows(2) {
            assert!(g.adj[w[0]].iter().any(|&(t, _)| t == w[1]));
            used.insert((w[0].min(w[1]), w[0].max(w[1])));
        }
        for (u, v, _) in g.edges() {
            assert!(used.contains(&(u.min(v), u.max(v))), "edge ({u}, {v}) missed");
        }
    }
}

/// Isomorphism must be invariant under relabelling and must separate graphs
/// that agree only on the cheap invariants.
#[test]
fn prop_isomorphism_survives_relabelling() {
    let mut rng = Rng::new(0x_0150);
    for _ in 0..150 {
        let n = 1 + pick(&mut rng, 8) as usize;
        let g = random_weighted(n, 0.2 + 0.5 * rng.next_f64(), false, &mut rng);
        let perm = rust_physics_engine::discrete::combinatorics::random_permutation(n, &mut rng);
        let mut h = Graph::new(n, false);
        for (u, v, w) in g.edges() {
            h.add_edge(perm[u], perm[v], w);
        }
        assert!(is_isomorphic_small(&g, &h), "relabelling broke isomorphism");
        // Isomorphic graphs share every structural invariant.
        assert_eq!(g.edge_count(), h.edge_count());
        assert_eq!(g.girth(), h.girth());
        assert_eq!(g.diameter(), h.diameter());
        assert_eq!(g.connected_components().len(), h.connected_components().len());
        assert_eq!(spanning_tree_count_exact(&g), spanning_tree_count_exact(&h));
        assert!((g.transitivity() - h.transitivity()).abs() < 1e-12);
        let mut ck: Vec<usize> = g.core_numbers();
        let mut ch: Vec<usize> = h.core_numbers();
        ck.sort_unstable();
        ch.sort_unstable();
        assert_eq!(ck, ch);
    }
    // The hypercube is the Cartesian product of smaller ones, which is a
    // structural claim rather than a relabelling. Capped at d = 2 so the
    // product has eight vertices, inside canonical_form_small's ceiling of ten.
    for d in 1..=2u32 {
        let prod = rust_physics_engine::graph::core::cartesian_product(
            &hypercube_graph(d),
            &complete_graph(2),
        );
        assert!(is_isomorphic_small(&prod, &hypercube_graph(d + 1)), "d = {d}");
    }
}
