//! Properties tying `graph::spectral`, `graph::coloring` and `graph::layout`
//! to each other and to `graph::matching`.
//!
//! Each module's own tests check it against its definition. These check the
//! theorems that connect the modules, which no one of them can check alone:
//! Koenig between matchings and covers, Hoffman and Wilf between the
//! adjacency spectrum and the chromatic number, and the four colour theorem
//! between planarity and colouring.

use rust_physics_engine::graph::coloring::{
    chromatic_number_exact_small, color_count, greedy_coloring, is_proper_coloring,
    max_clique_bron_kerbosch, max_independent_set_small, vertex_cover_exact_small, Order,
};
use rust_physics_engine::graph::core::Graph;
use rust_physics_engine::graph::layout::{
    circular_layout, crossing_number_estimate, fruchterman_reingold, kamada_kawai,
    planarity_test, spectral_layout, tree_layout_reingold_tilford,
};
use rust_physics_engine::graph::matching::blossom_max_matching;
use rust_physics_engine::graph::spectral::adjacency_spectrum;
use rust_physics_engine::monte_carlo::Rng;

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

/// Koenig's theorem: on a bipartite graph the largest matching and the
/// smallest vertex cover are the same size.
///
/// One is computed by the blossom algorithm in `graph::matching`, the other
/// through maximum independent sets in `graph::coloring`. Neither knows about
/// the other, and the theorem says they must agree -- but only on bipartite
/// graphs, so the test also checks that the gap really does open up on an odd
/// cycle, where the matching is one short of the cover.
#[test]
fn prop_koenig_links_matching_and_cover() {
    let mut rng = Rng::new(0x_C047);
    let mut bipartite_seen = 0;
    for _ in 0..300 {
        let n = 1 + pick(&mut rng, 10);
        let g = random_graph(n, 0.15 + 0.5 * rng.next_f64(), &mut rng);
        let matching = blossom_max_matching(&g);
        let size = matching.iter().filter(|x| x.is_some()).count() / 2;
        let cover = vertex_cover_exact_small(&g);
        // Every matching edge needs its own cover vertex, bipartite or not.
        assert!(size <= cover.len(), "a matching of {size} under a cover of {}", cover.len());
        if g.is_bipartite().is_some() {
            bipartite_seen += 1;
            assert_eq!(size, cover.len(), "Koenig's theorem fails on a bipartite graph");
        }
    }
    assert!(bipartite_seen > 30, "only {bipartite_seen} bipartite graphs were drawn");

    // An odd cycle is the smallest witness that the theorem needs
    // bipartiteness: the matching misses one vertex and the cover needs one
    // more than the matching has edges.
    for n in [3usize, 5, 7, 9, 11] {
        let mut c = Graph::new(n, false);
        for i in 0..n {
            c.add_edge(i, (i + 1) % n, 1.0);
        }
        let m = blossom_max_matching(&c);
        let size = m.iter().filter(|x| x.is_some()).count() / 2;
        assert_eq!(size, n / 2, "a maximum matching of C_{n}");
        assert_eq!(vertex_cover_exact_small(&c).len(), n / 2 + 1, "a minimum cover of C_{n}");
    }
}

/// Hoffman below and Wilf above: the adjacency spectrum brackets the
/// chromatic number.
///
/// Hoffman's bound is `chi >= 1 - lambda_max / lambda_min` and Wilf's is
/// `chi <= 1 + lambda_max`. Both are statements about eigenvalues of a matrix
/// constraining a purely combinatorial quantity, and they are computed here
/// by completely separate machinery -- Jacobi rotations on one side, an
/// exhaustive colouring search on the other.
#[test]
fn prop_spectral_bounds_bracket_the_chromatic_number() {
    let mut rng = Rng::new(0x_40FF);
    for _ in 0..200 {
        let n = 2 + pick(&mut rng, 8);
        let g = random_graph(n, 0.2 + 0.6 * rng.next_f64(), &mut rng);
        if g.edge_count() == 0 {
            continue;
        }
        let chi = chromatic_number_exact_small(&g) as f64;
        let spectrum = adjacency_spectrum(&g);
        let hi = *spectrum.last().expect("non-empty");
        let lo = spectrum[0];
        assert!(chi <= 1.0 + hi + 1e-6, "Wilf: chi {chi} over 1 + {hi}");
        // A graph with an edge has a negative eigenvalue, since the trace is
        // zero and the spectrum is not.
        assert!(lo < -1e-9, "a graph with an edge has a negative eigenvalue");
        let hoffman = 1.0 - hi / lo;
        assert!(chi >= hoffman - 1e-6, "Hoffman: chi {chi} under {hoffman}");
    }
    // Both bounds are tight on a complete graph, where chi = n, lambda_max =
    // n - 1 and lambda_min = -1.
    for n in 2..=7usize {
        let mut k = Graph::new(n, false);
        for i in 0..n {
            for j in i + 1..n {
                k.add_edge(i, j, 1.0);
            }
        }
        let s = adjacency_spectrum(&k);
        let chi = chromatic_number_exact_small(&k) as f64;
        assert!((chi - (1.0 + s[n - 1])).abs() < 1e-6, "Wilf is not tight on K_{n}");
        assert!((chi - (1.0 - s[n - 1] / s[0])).abs() < 1e-6, "Hoffman is not tight on K_{n}");
    }
}

/// The four colour theorem, checked against a planarity test that knows
/// nothing about colouring.
///
/// Every graph `graph::layout` calls planar must be four-colourable, and the
/// degeneracy of a planar graph is at most five, so the smallest-last greedy
/// order must never need a sixth colour. Both are properties of the planar
/// graphs alone, so a planarity test that said yes too often would be caught
/// here rather than in its own module.
#[test]
fn prop_planar_graphs_are_four_colorable() {
    let mut rng = Rng::new(0x_4C01);
    let mut planar_seen = 0;
    let mut needed_four = 0;
    for _ in 0..400 {
        let n = 1 + pick(&mut rng, 10);
        let g = random_graph(n, 0.1 + 0.45 * rng.next_f64(), &mut rng);
        if !planarity_test(&g) {
            continue;
        }
        planar_seen += 1;
        let chi = chromatic_number_exact_small(&g);
        assert!(chi <= 4, "a planar graph needed {chi} colours");
        if chi == 4 {
            needed_four += 1;
        }
        // Planar graphs have degeneracy at most five, so the degeneracy order
        // never opens a sixth colour.
        let c = greedy_coloring(&g, Order::SmallestLast);
        assert!(is_proper_coloring(&g, &c));
        assert!(color_count(&c) <= 6, "the degeneracy order used {} colours", color_count(&c));
        // Euler's bound holds for every planar graph with three vertices or
        // more, which is the other half of what makes six work.
        if n >= 3 {
            assert!(g.edge_count() <= 3 * n - 6, "over Euler's bound but called planar");
        }
    }
    assert!(planar_seen > 100, "only {planar_seen} planar graphs were drawn");
    assert!(needed_four > 0, "no drawn planar graph actually needed four colours");
}

/// Colour classes are independent sets and cliques force colours, so the
/// chromatic number is squeezed from both directions by quantities computed
/// by an entirely different search.
#[test]
fn prop_clique_and_independence_bound_the_chromatic_number() {
    let mut rng = Rng::new(0x_C119);
    for _ in 0..200 {
        let n = 1 + pick(&mut rng, 10);
        let g = random_graph(n, 0.2 + 0.6 * rng.next_f64(), &mut rng);
        let chi = chromatic_number_exact_small(&g);
        let omega = max_clique_bron_kerbosch(&g).len();
        let alpha = max_independent_set_small(&g).len();
        // A clique needs a colour per vertex.
        assert!(chi >= omega, "chi {chi} under the clique number {omega}");
        // Each colour class is independent, so chi classes cover at most
        // chi * alpha vertices.
        assert!(chi * alpha >= n, "chi {chi} times alpha {alpha} under n = {n}");
        // And chi is never more than n.
        assert!(chi <= n);
    }
}

/// A drawing of a non-planar graph must cross, whatever the algorithm that
/// produced it.
///
/// This is the one direction of planarity that can be checked against a
/// drawing: if `planarity_test` says no crossing-free drawing exists, then
/// every straight-line drawing any of the layout routines produces must have
/// at least one crossing. A planarity test that wrongly said no would sail
/// through its own module's tests and fail here.
#[test]
fn prop_nonplanar_graphs_cross_in_every_drawing() {
    let mut rng = Rng::new(0x_C205);
    let mut checked = 0;
    for _ in 0..200 {
        let n = 3 + pick(&mut rng, 8);
        let g = random_graph(n, 0.3 + 0.5 * rng.next_f64(), &mut rng);
        if planarity_test(&g) {
            continue;
        }
        checked += 1;
        let drawings = [
            circular_layout(n),
            kamada_kawai(&g, 200),
            fruchterman_reingold(&g, 200, &mut rng),
            spectral_layout(&g),
        ];
        for (i, d) in drawings.iter().enumerate() {
            assert!(
                crossing_number_estimate(&g, d) >= 1,
                "drawing {i} of a non-planar graph has no crossings"
            );
        }
    }
    assert!(checked > 50, "only {checked} non-planar graphs were drawn");

    // The other direction where it can be had exactly: a tree is planar, and
    // the tree layout draws it without a single crossing.
    for _ in 0..100 {
        let n = 1 + pick(&mut rng, 20);
        let mut t = Graph::new(n, false);
        for v in 1..n {
            t.add_edge(pick(&mut rng, v), v, 1.0);
        }
        assert!(planarity_test(&t), "a tree is planar");
        assert_eq!(crossing_number_estimate(&t, &tree_layout_reingold_tilford(&t, 0)), 0);
    }
}
