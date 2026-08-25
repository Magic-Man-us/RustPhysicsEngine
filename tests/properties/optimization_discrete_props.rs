//! Properties tying `optimization::integer` and `optimization::network` to
//! the rest of the crate.
//!
//! Two kinds of check dominate. Where a problem has both a combinatorial
//! algorithm and a linear programming formulation -- shortest path, maximum
//! flow -- the two must agree, and they share no code at all: one is a
//! priority-queue sweep or an augmenting-path search, the other a simplex
//! method over a node-arc incidence matrix. Agreement is evidence for both.
//!
//! Where a problem is solved by a greedy rule with a proven ratio, the ratio
//! is checked against an exact answer rather than the greedy result being
//! checked for plausibility. A bound nobody tests against an optimum is not a
//! guarantee, it is a hope.

use rust_physics_engine::graph::core::Graph;
use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::optimization::integer::{
    bin_packing_exact_small, bin_packing_ffd, bin_packing_lower_bound, branch_and_bound,
    edit_distance, knapsack_01, knapsack_branch_bound, longest_common_subsequence,
    longest_increasing_subsequence, set_cover_exact_small, set_cover_greedy, subset_sum,
    subset_sum_count,
};
use rust_physics_engine::optimization::lp::{simplex, LpProblem};
use rust_physics_engine::optimization::network::{
    critical_path_method, lpt_makespan, max_flow_lp_check, shortest_path_lp_check,
};

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random directed graph with positive arc weights.
fn random_digraph(n: usize, density: f64, rng: &mut Rng) -> Graph {
    let mut g = Graph::new(n, true);
    for u in 0..n {
        for v in 0..n {
            if u != v && rng.next_f64() < density {
                g.add_edge(u, v, (rng.next_f64() * 9.0).round() + 1.0);
            }
        }
    }
    g
}

#[test]
fn prop_the_shortest_path_program_and_dijkstra_never_disagree() {
    // Totally unimodular constraints mean the relaxation is integral, which is
    // why a general-purpose linear program can answer a combinatorial question
    // exactly. The two methods share nothing but the graph.
    let mut rng = Rng::new(0x_5407_2001);
    let mut compared = 0usize;
    for _ in 0..60 {
        let n = 4 + pick(&mut rng, 5);
        let g = random_digraph(n, 0.45, &mut rng);
        let (distances, _) = rust_physics_engine::graph::paths::dijkstra(&g, 0);
        for t in 1..n {
            let lp = shortest_path_lp_check(&g, 0, t).unwrap();
            match (distances[t].is_finite(), lp) {
                (true, Some(value)) => {
                    compared += 1;
                    assert!(
                        (value - distances[t]).abs() < 1e-6,
                        "to {t}: the program gave {value}, Dijkstra {}",
                        distances[t]
                    );
                }
                (false, None) => {}
                (reachable, other) => panic!(
                    "disagreed on reachability to {t}: Dijkstra {reachable}, program {other:?}"
                ),
            }
        }
    }
    assert!(compared > 100, "only {compared} pairs were comparable");
}

#[test]
fn prop_the_max_flow_program_and_the_augmenting_path_search_never_disagree() {
    let mut rng = Rng::new(0x_F108_2002);
    for _ in 0..50 {
        let n = 4 + pick(&mut rng, 4);
        let g = random_digraph(n, 0.5, &mut rng);
        let combinatorial = rust_physics_engine::graph::flow::max_flow(&g, 0, n - 1);
        let lp = max_flow_lp_check(&g, 0, n - 1).unwrap().unwrap_or(0.0);
        assert!(
            (lp - combinatorial).abs() < 1e-6,
            "the program gave {lp}, the augmenting-path method {combinatorial}"
        );
        // Both are bounded by the capacity leaving the source.
        let out: f64 = g.adj[0].iter().map(|&(_, w)| w).sum();
        assert!(combinatorial <= out + 1e-9, "the flow exceeds the source's capacity");
    }
}

#[test]
fn prop_the_knapsack_table_and_search_tree_always_agree() {
    // Two exact methods with nothing in common: one fills a table over
    // capacities, the other prunes a binary search tree with a fractional
    // bound. Any disagreement is a bug in one of them.
    let mut rng = Rng::new(0x_C0FF_2003);
    for _ in 0..400 {
        let n = 1 + pick(&mut rng, 16);
        let values: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 50)).collect();
        let weights: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 25)).collect();
        let capacity = 1 + (rng.next_u64() % 80);

        let (table_value, table_pick) = knapsack_01(&values, &weights, capacity);
        let (tree_value, tree_pick) = knapsack_branch_bound(&values, &weights, capacity);
        assert_eq!(table_value, tree_value, "the two knapsack methods disagreed");

        // Both selections fit and are worth what was claimed.
        for (label, picks) in [("table", &table_pick), ("tree", &tree_pick)] {
            let w: u64 = picks.iter().enumerate().filter(|(_, &t)| t).map(|(i, _)| weights[i]).sum();
            let v: u64 = picks.iter().enumerate().filter(|(_, &t)| t).map(|(i, _)| values[i]).sum();
            assert!(w <= capacity, "{label} overfilled the sack");
            assert_eq!(v, table_value, "{label}'s selection is worth {v}");
        }
    }
}

#[test]
fn prop_branch_and_bound_never_beats_its_own_relaxation() {
    // The bound that makes the method terminate: no integer point can be
    // better than the best fractional one, since the integer points are a
    // subset of the fractional region.
    let mut rng = Rng::new(0x_B4B0_2004);
    let mut checked = 0usize;
    for _ in 0..150 {
        let n = 2 + pick(&mut rng, 3);
        let m = 1 + pick(&mut rng, 3);
        let mut a = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a.set(i, j, (rng.next_f64() * 4.0).round() + 1.0);
            }
        }
        let b: Vec<f64> = (0..m).map(|_| (rng.next_f64() * 20.0).round() + 4.0).collect();
        let c: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 9.0).round() + 1.0).collect();
        let mut p = LpProblem::new(c, a, b, true).unwrap();
        for j in 0..n {
            p.bounds[j] = (0.0, 12.0);
        }
        let vars: Vec<usize> = (0..n).collect();

        let Some((x, value)) = branch_and_bound(&p, &vars, 200_000).unwrap() else { continue };
        checked += 1;
        assert!(p.is_feasible(&x, 1e-6), "the integer answer is infeasible");
        assert!(
            x.iter().all(|v| (v - v.round()).abs() < 1e-6),
            "a variable came back fractional: {x:?}"
        );
        let relaxed = simplex(&p).unwrap().objective().unwrap();
        assert!(
            value <= relaxed + 1e-6,
            "the integer optimum {value} beat its relaxation {relaxed}"
        );
        // And the integer point is genuinely achievable in the relaxation.
        assert!((p.objective_at(&x) - value).abs() < 1e-9);
    }
    assert!(checked > 100, "only {checked} of 150 programs had an integer optimum");
}

#[test]
fn prop_greedy_packing_and_covering_stay_inside_their_proven_ratios() {
    let mut rng = Rng::new(0x_B1CE_2005);
    for _ in 0..120 {
        // Bin packing: first-fit-decreasing within 11/9 OPT + 6/9.
        let n = 1 + pick(&mut rng, 9);
        let sizes: Vec<f64> = (0..n).map(|_| rng.next_f64() * 0.75 + 0.05).collect();
        let greedy = bin_packing_ffd(&sizes, 1.0);
        let exact = bin_packing_exact_small(&sizes, 1.0);
        let bound = bin_packing_lower_bound(&sizes, 1.0);

        let mut seen = vec![0usize; n];
        for bin in &greedy {
            let load: f64 = bin.iter().map(|&i| sizes[i]).sum();
            assert!(load <= 1.0 + 1e-9, "a bin holds {load}");
            for &i in bin {
                seen[i] += 1;
            }
        }
        assert!(seen.iter().all(|&k| k == 1), "an item was lost or duplicated");
        assert!(exact.len() >= bound, "the exact packing beat the volume bound");
        assert!(greedy.len() >= exact.len(), "greedy beat the optimum");
        let guarantee = 11.0 / 9.0 * exact.len() as f64 + 6.0 / 9.0;
        assert!(
            greedy.len() as f64 <= guarantee + 1e-9,
            "{} bins exceeds the guarantee {guarantee} against {}",
            greedy.len(),
            exact.len()
        );

        // Set cover: greedy within H_n of optimal.
        let universe = 3 + pick(&mut rng, 7);
        let sets: Vec<Vec<usize>> = (0..2 + pick(&mut rng, 7))
            .map(|_| (0..universe).filter(|_| rng.next_f64() < 0.45).collect())
            .collect();
        match (set_cover_greedy(universe, &sets), set_cover_exact_small(universe, &sets)) {
            (Some(g), Some(e)) => {
                let mut covered = vec![false; universe];
                for &i in &g {
                    for &v in &sets[i] {
                        if v < universe {
                            covered[v] = true;
                        }
                    }
                }
                assert!(covered.iter().all(|&c| c), "the greedy cover is incomplete");
                let harmonic: f64 = (1..=universe).map(|k| 1.0 / k as f64).sum();
                assert!(
                    g.len() as f64 <= harmonic * e.len() as f64 + 1e-9,
                    "{} sets exceeds H_n times {}",
                    g.len(),
                    e.len()
                );
            }
            (None, None) => {}
            _ => panic!("greedy and exact disagreed on whether a cover exists"),
        }
    }
}

#[test]
fn prop_longest_processing_time_stays_inside_its_ratio() {
    // The bound is 4/3 - 1/(3m), and it is tight, so it is worth checking
    // against an exact answer rather than assuming.
    let mut rng = Rng::new(0x_1B70_2006);
    for _ in 0..100 {
        let machines = 2 + pick(&mut rng, 3);
        let n = machines + pick(&mut rng, 5);
        if n > 8 {
            continue;
        }
        let jobs: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 9.0).round() + 1.0).collect();
        let (makespan, assignment) = lpt_makespan(&jobs, machines);

        // The reported makespan really is the busiest machine's load.
        let loads: Vec<f64> = (0..machines)
            .map(|m| {
                jobs.iter().enumerate().filter(|(i, _)| assignment[*i] == m).map(|(_, &d)| d).sum()
            })
            .collect();
        assert!((loads.iter().copied().fold(0.0f64, f64::max) - makespan).abs() < 1e-9);
        // No machine is left with work it was not assigned.
        assert!((loads.iter().sum::<f64>() - jobs.iter().sum::<f64>()).abs() < 1e-9);

        // Exhaustive assignment.
        let mut best = f64::INFINITY;
        let mut counter = vec![0usize; n];
        loop {
            let mut load = vec![0.0f64; machines];
            for (i, &m) in counter.iter().enumerate() {
                load[m] += jobs[i];
            }
            best = best.min(load.iter().copied().fold(0.0f64, f64::max));
            let mut k = 0usize;
            while k < n {
                counter[k] += 1;
                if counter[k] < machines {
                    break;
                }
                counter[k] = 0;
                k += 1;
            }
            if k == n {
                break;
            }
        }
        let ratio = 4.0 / 3.0 - 1.0 / (3.0 * machines as f64);
        assert!(makespan >= best - 1e-9, "greedy beat the optimum");
        assert!(
            makespan <= ratio * best + 1e-9,
            "makespan {makespan} exceeds {ratio} times {best}"
        );
    }
}

#[test]
fn prop_the_critical_path_bounds_every_schedule() {
    // No schedule can finish before the longest chain of dependencies, however
    // many resources are available -- which is the reason to compute it.
    let mut rng = Rng::new(0x_C97A_2007);
    for _ in 0..120 {
        let n = 2 + pick(&mut rng, 8);
        let tasks: Vec<(f64, Vec<usize>)> = (0..n)
            .map(|i| {
                let preds: Vec<usize> = (0..i).filter(|_| rng.next_f64() < 0.35).collect();
                ((rng.next_f64() * 9.0).round() + 1.0, preds)
            })
            .collect();
        let (duration, critical, times) = critical_path_method(&tasks).unwrap();

        // Independently: the longest chain ending at each task.
        let mut longest = vec![0.0f64; n];
        for i in 0..n {
            longest[i] =
                tasks[i].0 + tasks[i].1.iter().map(|&p| longest[p]).fold(0.0f64, f64::max);
        }
        let expected = longest.iter().copied().fold(0.0f64, f64::max);
        assert!((duration - expected).abs() < 1e-9, "{duration} against {expected}");

        // Every precedence is respected, and slack is zero exactly on the
        // critical path.
        for i in 0..n {
            for &p in &tasks[i].1 {
                assert!(times[p].early_finish <= times[i].early_start + 1e-9);
                assert!(times[i].late_start + 1e-9 >= times[p].late_finish);
            }
            assert_eq!(critical.contains(&i), times[i].slack().abs() < 1e-9);
            assert!(times[i].slack() >= -1e-9, "negative slack at task {i}");
        }
        assert!(!critical.is_empty(), "no task is critical");
    }
}

#[test]
fn prop_the_dynamic_programming_classics_return_what_they_claim() {
    let mut rng = Rng::new(0x_D9C1_2008);
    for _ in 0..200 {
        // Subset sum: the reported subset sums to the target, and the count
        // matches an independent enumeration.
        let n = 1 + pick(&mut rng, 12);
        let xs: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 20)).collect();
        let target = rng.next_u64() % 50;
        let mut brute = 0u64;
        for mask in 0u32..(1u32 << n) {
            let s: u64 = (0..n).filter(|k| mask & (1 << k) != 0).map(|k| xs[k]).sum();
            if s == target {
                brute += 1;
            }
        }
        assert_eq!(subset_sum_count(&xs, target).to_string(), brute.to_string());
        match subset_sum(&xs, target) {
            Some(indices) => {
                assert!(brute > 0, "found a subset where none exists");
                assert_eq!(indices.iter().map(|&i| xs[i]).sum::<u64>(), target);
            }
            None => assert_eq!(brute, 0, "missed an existing subset"),
        }

        // Longest increasing subsequence: increasing, and of maximal length.
        let sequence: Vec<f64> =
            (0..1 + pick(&mut rng, 25)).map(|_| (rng.next_f64() * 15.0).round()).collect();
        let lis = longest_increasing_subsequence(&sequence);
        assert!(lis.windows(2).all(|w| w[0] < w[1] && sequence[w[0]] < sequence[w[1]]));
        let m = sequence.len();
        let mut best = vec![1usize; m];
        for i in 1..m {
            for j in 0..i {
                if sequence[j] < sequence[i] && best[j] + 1 > best[i] {
                    best[i] = best[j] + 1;
                }
            }
        }
        assert_eq!(lis.len(), *best.iter().max().unwrap_or(&0));

        // Edit distance is a metric, and the common subsequence is common.
        let word = |rng: &mut Rng, k: usize| -> Vec<u8> {
            (0..k).map(|_| b'a' + (rng.next_u64() % 3) as u8).collect()
        };
        let (ka, kb, kc) = (pick(&mut rng, 8), pick(&mut rng, 8), pick(&mut rng, 8));
        let (a, b, c) = (word(&mut rng, ka), word(&mut rng, kb), word(&mut rng, kc));
        assert_eq!(edit_distance(&a, &b), edit_distance(&b, &a));
        assert!(edit_distance(&a, &b) <= edit_distance(&a, &c) + edit_distance(&c, &b));
        let lcs = longest_common_subsequence(&a, &b);
        let is_sub = |s: &[u8], whole: &[u8]| {
            let mut it = whole.iter();
            s.iter().all(|ch| it.any(|w| w == ch))
        };
        assert!(is_sub(&lcs, &a) && is_sub(&lcs, &b), "the subsequence is not common");
        // A common subsequence of length k implies the distance is at most
        // the leftover on each side.
        assert!(edit_distance(&a, &b) <= a.len() + b.len() - 2 * lcs.len());
    }
}
