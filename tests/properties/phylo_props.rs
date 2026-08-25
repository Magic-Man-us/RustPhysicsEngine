//! Properties of the phylogenetics module.
//!
//! A tree is a structure with strong internal redundancy, and that is what
//! these properties exploit. The patristic distances a tree induces are not
//! free numbers: they obey the four-point condition, and a method given
//! distances that came from a tree must give that tree back. Newick text is
//! a lossless encoding, so a round trip is an identity. Parsimony and
//! likelihood each have a bound that no data can violate. And the shape
//! statistics have a null distribution the simulator is supposed to
//! reproduce, which turns "does the simulator sample the right process"
//! into an arithmetic check.

use rust_physics_engine::biophysics::phylo::{
    birth_death_tree, distance_matrix_jc69, gamma_statistic, likelihood_jc69,
    lineage_through_time, neighbor_joining, parsimony_fitch, upgma, DistanceMethod, PhyloTree,
};
use rust_physics_engine::biophysics::phylo::bootstrap_trees;
use rust_physics_engine::linalg::Matrix;
use rust_physics_engine::monte_carlo::Rng;

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random birth-death tree, which is the module's own source of shapes.
fn random_tree(tips: usize, rng: &mut Rng) -> PhyloTree {
    birth_death_tree(1.0, 0.4, tips, rng).expect("a supercritical process reaches its target")
}

/// The patristic distance matrix of a tree, with its leaf labels.
fn patristic(tree: &PhyloTree) -> (Matrix, Vec<String>) {
    let leaves = tree.leaves();
    let mut out = Matrix::zeros(leaves.len(), leaves.len());
    for (i, a) in leaves.iter().enumerate() {
        for (j, b) in leaves.iter().enumerate() {
            out.set(i, j, tree.distance(*a, *b).expect("leaf indices are in range"));
        }
    }
    (out, leaves.iter().map(|k| tree.labels[*k].clone()).collect())
}

fn random_dna(width: usize, rng: &mut Rng) -> Vec<u8> {
    (0..width).map(|_| b"ACGT"[pick(rng, 4)]).collect()
}

fn mutate(seq: &[u8], rate: f64, rng: &mut Rng) -> Vec<u8> {
    seq.iter()
        .map(|base| if rng.next_f64() < rate { b"ACGT"[pick(rng, 4)] } else { *base })
        .collect()
}

#[test]
fn prop_patristic_distances_obey_the_four_point_condition() {
    // The defining property of a tree metric: of the three ways to pair up
    // four leaves, two of the summed distances are equal and the third is
    // no larger. It holds for every tree, whatever its shape or rooting.
    let mut rng = Rng::new(0x0B10_4001);
    for _ in 0..40 {
        let tree = random_tree(8, &mut rng);
        let leaves = tree.leaves();
        for _ in 0..20 {
            let mut chosen: Vec<usize> = Vec::new();
            while chosen.len() < 4 {
                let candidate = leaves[pick(&mut rng, leaves.len())];
                if !chosen.contains(&candidate) {
                    chosen.push(candidate);
                }
            }
            let d = |a: usize, b: usize| tree.distance(chosen[a], chosen[b]).unwrap();
            let mut sums = [d(0, 1) + d(2, 3), d(0, 2) + d(1, 3), d(0, 3) + d(1, 2)];
            sums.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let scale = sums[2].max(1.0);
            assert!(
                (sums[2] - sums[1]).abs() < 1e-9 * scale,
                "the two larger pairings differ: {sums:?}"
            );
            assert!(sums[0] <= sums[1] + 1e-9 * scale, "the smallest pairing is not smallest");
        }
    }
}

#[test]
fn prop_patristic_distances_are_a_metric() {
    let mut rng = Rng::new(0x0B10_4002);
    for _ in 0..30 {
        let tree = random_tree(7, &mut rng);
        let leaves = tree.leaves();
        for a in &leaves {
            assert!(tree.distance(*a, *a).unwrap().abs() < 1e-12);
            for b in &leaves {
                let ab = tree.distance(*a, *b).unwrap();
                assert!((ab - tree.distance(*b, *a).unwrap()).abs() < 1e-12);
                assert!(ab >= 0.0);
                assert_eq!(a == b, ab < 1e-12);
                for c in &leaves {
                    let detour = tree.distance(*a, *c).unwrap() + tree.distance(*c, *b).unwrap();
                    assert!(ab <= detour + 1e-9, "going via a third leaf was shorter");
                }
            }
        }
    }
}

#[test]
fn prop_newick_is_a_lossless_encoding() {
    // Text out, text in, and the tree that comes back describes the same
    // clades with the same lengths.
    let mut rng = Rng::new(0x0B10_4003);
    for tips in [3usize, 5, 9, 16] {
        for _ in 0..10 {
            let tree = random_tree(tips, &mut rng);
            let text = tree.to_newick();
            let again = PhyloTree::from_newick(&text).unwrap();
            assert_eq!(again.to_newick(), text, "a second pass changed the text");
            assert_eq!(again.robinson_foulds(&tree).unwrap(), 0);
            assert!((again.total_length() - tree.total_length()).abs() < 1e-9);
            assert!((again.height() - tree.height()).abs() < 1e-9);
            // Newick preserves the tree, not the node numbering, so the
            // distances are compared by label rather than by index.
            let leaves = tree.leaves();
            for a in &leaves {
                for b in &leaves {
                    let here = tree.distance(*a, *b).unwrap();
                    let find = |name: &String| {
                        again
                            .leaves()
                            .into_iter()
                            .find(|k| again.labels[*k] == *name)
                            .expect("the label survived")
                    };
                    let there = again
                        .distance(find(&tree.labels[*a]), find(&tree.labels[*b]))
                        .unwrap();
                    assert!((here - there).abs() < 1e-9, "{here} became {there}");
                }
            }
        }
    }
}

#[test]
fn prop_neighbour_joining_inverts_the_patristic_map() {
    // Neighbour joining is consistent: on distances that came from a tree
    // it returns that tree's distances, to rounding. This is the strongest
    // statement anyone makes about a distance method and it is exact.
    let mut rng = Rng::new(0x0B10_4004);
    for tips in [4usize, 6, 9, 14] {
        for _ in 0..8 {
            let tree = random_tree(tips, &mut rng);
            let (dist, labels) = patristic(&tree);
            let built = neighbor_joining(&dist, &labels).unwrap();
            let (again, order) = patristic(&built);
            assert_eq!(order, labels, "the leaves came back in a different order");
            let scale = dist.data.iter().fold(0.0f64, |a, b| a.max(*b)).max(1.0);
            for i in 0..dist.rows {
                for j in 0..dist.cols {
                    assert!(
                        (again.get(i, j) - dist.get(i, j)).abs() < 1e-9 * scale,
                        "distance {i},{j}: {} against {}",
                        again.get(i, j),
                        dist.get(i, j)
                    );
                }
            }
            // The same tree, so the same unrooted branches.
            assert_eq!(built.bipartitions(), tree.bipartitions());
        }
    }
}

#[test]
fn prop_upgma_inverts_the_patristic_map_when_the_clock_holds() {
    // Birth-death trees are ultrametric, which is exactly the condition
    // under which UPGMA's assumption is true -- and there it is exact too.
    let mut rng = Rng::new(0x0B10_4005);
    for tips in [3usize, 6, 11] {
        for _ in 0..8 {
            let tree = random_tree(tips, &mut rng);
            let (dist, labels) = patristic(&tree);
            let built = upgma(&dist, &labels).unwrap();
            assert_eq!(built.robinson_foulds(&tree).unwrap(), 0);
            let (again, _) = patristic(&built);
            let scale = dist.data.iter().fold(0.0f64, |a, b| a.max(*b)).max(1.0);
            for i in 0..dist.rows {
                for j in 0..dist.cols {
                    assert!((again.get(i, j) - dist.get(i, j)).abs() < 1e-9 * scale);
                }
            }
        }
    }
}

#[test]
fn prop_upgma_always_returns_an_ultrametric_tree() {
    // Whatever goes in. That is the assumption made visible: a clocklike
    // answer is not evidence of a clock.
    let mut rng = Rng::new(0x0B10_4006);
    for size in 2usize..8 {
        for _ in 0..20 {
            let mut dist = Matrix::zeros(size, size);
            for i in 0..size {
                for j in (i + 1)..size {
                    let d = 0.01 + rng.next_f64();
                    dist.set(i, j, d);
                    dist.set(j, i, d);
                }
            }
            let labels: Vec<String> = (0..size).map(|k| format!("t{k}")).collect();
            let tree = upgma(&dist, &labels).unwrap();
            assert!(tree.is_ultrametric(1e-9 * tree.height().max(1.0)));
            assert_eq!(tree.leaves().len(), size);
            assert!(tree.branch_length.iter().all(|b| *b >= 0.0));
        }
    }
}

#[test]
fn prop_a_reconstructed_tree_is_binary_ultrametric_and_the_size_asked_for() {
    let mut rng = Rng::new(0x0B10_4007);
    for tips in [3usize, 5, 12, 30] {
        for mu in [0.0, 0.5, 0.9] {
            let tree = birth_death_tree(1.0, mu, tips, &mut rng).unwrap();
            assert_eq!(tree.leaves().len(), tips);
            assert!(tree.is_binary());
            assert!(tree.height() > 0.0);
            assert!(tree.is_ultrametric(1e-9 * tree.height()));
            assert_eq!(tree.len(), 2 * tips - 1, "a binary tree has 2n - 1 nodes");
            assert!(tree.branch_length.iter().all(|b| b.is_finite() && *b >= 0.0));
        }
    }
}

#[test]
fn prop_parsimony_sits_between_its_two_bounds() {
    // At least one change per extra state, and never more than one per
    // branch. Both are theorems, so no character can escape them.
    let mut rng = Rng::new(0x0B10_4008);
    for _ in 0..30 {
        let tips = 4 + pick(&mut rng, 9);
        let tree = random_tree(tips, &mut rng);
        for _ in 0..20 {
            let characters: Vec<u8> = (0..tips).map(|_| b"ACGT"[pick(&mut rng, 4)]).collect();
            let mut distinct = characters.clone();
            distinct.sort_unstable();
            distinct.dedup();
            let score = parsimony_fitch(&tree, &characters).unwrap();
            assert!(score >= distinct.len() as u64 - 1, "{score} changes for {distinct:?}");
            assert!((score as usize) < tree.len(), "more changes than branches");
        }
    }
}

#[test]
fn prop_parsimony_is_blind_to_which_state_is_which() {
    // Fitch counts changes, so renaming the states cannot change the count.
    // A method that treated one state as ancestral would fail this.
    let mut rng = Rng::new(0x0B10_4009);
    for _ in 0..40 {
        let tips = 4 + pick(&mut rng, 8);
        let tree = random_tree(tips, &mut rng);
        let characters: Vec<u8> = (0..tips).map(|_| b"ACGT"[pick(&mut rng, 4)]).collect();
        let score = parsimony_fitch(&tree, &characters).unwrap();
        // Any permutation of the alphabet.
        let mut alphabet = *b"ACGT";
        for i in (1..4).rev() {
            alphabet.swap(i, pick(&mut rng, i + 1));
        }
        let renamed: Vec<u8> = characters
            .iter()
            .map(|c| alphabet[b"ACGT".iter().position(|b| b == c).unwrap()])
            .collect();
        assert_eq!(parsimony_fitch(&tree, &renamed).unwrap(), score);
    }
}

#[test]
fn prop_the_log_likelihood_is_negative_and_adds_over_sites() {
    // Sites are independent under the model, so the alignment's
    // log-likelihood is the sum of its columns'. A pruning pass that leaked
    // state between sites would not survive this.
    let mut rng = Rng::new(0x0B10_400A);
    for _ in 0..20 {
        let tips = 3 + pick(&mut rng, 6);
        let tree = random_tree(tips, &mut rng);
        let width = 6;
        let seqs: Vec<Vec<u8>> = (0..tips).map(|_| random_dna(width, &mut rng)).collect();
        let whole = likelihood_jc69(&tree, &seqs).unwrap();
        let mut summed = 0.0;
        for site in 0..width {
            let column: Vec<Vec<u8>> = seqs.iter().map(|s| vec![s[site]]).collect();
            summed += likelihood_jc69(&tree, &column).unwrap();
        }
        assert!((whole - summed).abs() < 1e-9, "{whole} against {summed}");
        // A likelihood is a probability, so its log cannot be positive.
        assert!(whole < 0.0);
        // And no column can be likelier than a certain event.
        assert!(whole <= 0.0 + 1e-12);
    }
}

#[test]
fn prop_a_constant_site_is_likelier_the_shorter_the_tree() {
    // Substitution destroys agreement, so an alignment where every tip
    // carries the same base gets less likely as the branches grow -- and
    // tends to a quarter, the chance the root drew that base at all.
    let mut rng = Rng::new(0x0B10_400B);
    for _ in 0..15 {
        let tips = 3 + pick(&mut rng, 5);
        let base = random_tree(tips, &mut rng);
        let seqs: Vec<Vec<u8>> = vec![b"A".to_vec(); tips];
        let mut previous = f64::INFINITY;
        for scale in [0.05, 0.2, 0.5, 1.0, 2.0] {
            let stretched = PhyloTree::new(
                base.parent.clone(),
                base.branch_length.iter().map(|b| b * scale).collect(),
                base.labels.clone(),
            )
            .unwrap();
            let value = likelihood_jc69(&stretched, &seqs).unwrap();
            assert!(value < previous, "stretching the tree raised the likelihood");
            previous = value;
            assert!(value.exp() <= 0.25 + 1e-12, "a constant site beat the root's own draw");
        }
    }
}

#[test]
fn prop_bootstrap_support_is_a_fraction_of_replicates() {
    // Whatever the data, support is a count over a count: in the unit
    // interval, one entry per branch of the reference, and a multiple of
    // 1 / replicates.
    let mut rng = Rng::new(0x0B10_400C);
    let replicates = 25;
    for _ in 0..8 {
        let tips = 4 + pick(&mut rng, 4);
        let root = random_dna(200, &mut rng);
        let seqs: Vec<Vec<u8>> =
            (0..tips).map(|_| mutate(&root, 0.02 + 0.1 * rng.next_f64(), &mut rng)).collect();
        let labels: Vec<String> = (0..tips).map(|k| format!("t{k}")).collect();
        for method in [DistanceMethod::Upgma, DistanceMethod::NeighborJoining] {
            let (tree, support) =
                bootstrap_trees(&seqs, &labels, replicates, method, &mut rng).unwrap();
            assert_eq!(support.len(), tree.bipartitions().len());
            for value in &support {
                assert!((0.0..=1.0).contains(value), "support of {value}");
                let ticks = value * replicates as f64;
                assert!((ticks - ticks.round()).abs() < 1e-9, "support is not a count");
            }
        }
    }
}

#[test]
fn prop_the_lineage_curve_is_a_staircase_ending_at_the_tips() {
    let mut rng = Rng::new(0x0B10_400D);
    for tips in [3usize, 7, 15, 26] {
        for _ in 0..6 {
            let tree = random_tree(tips, &mut rng);
            let curve = lineage_through_time(&tree).unwrap();
            assert_eq!(curve[0], (0.0, 1));
            assert_eq!(curve.last().unwrap().1, tips);
            assert!((curve.last().unwrap().0 - tree.height()).abs() < 1e-12);
            for pair in curve.windows(2) {
                assert!(pair[1].0 >= pair[0].0 - 1e-12);
                assert!(pair[1].1 >= pair[0].1);
                assert!(pair[1].1 - pair[0].1 <= 1, "a binary tree adds one lineage at a time");
            }
            // Each branching happens at the depth of its node.
            let branchings: usize = curve.windows(2).filter(|p| p[1].1 > p[0].1).count();
            assert_eq!(branchings, tips - 1);
        }
    }
}

#[test]
fn prop_gamma_reproduces_its_null_distribution_on_pure_birth_trees() {
    // The simulator and the statistic are checked against each other: if
    // the trees are Yule trees and the formula is Pybus and Harvey's, the
    // values must look standard normal. Either being wrong breaks this.
    let mut rng = Rng::new(0x0B10_400E);
    for tips in [15usize, 40] {
        let replicates = 250;
        let values: Vec<f64> = (0..replicates)
            .map(|_| {
                gamma_statistic(&birth_death_tree(1.7, 0.0, tips, &mut rng).unwrap()).unwrap()
            })
            .collect();
        let mean: f64 = values.iter().sum::<f64>() / replicates as f64;
        let variance: f64 =
            values.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / replicates as f64;
        let standard_error = variance.sqrt() / (replicates as f64).sqrt();
        assert!(mean.abs() < 4.0 * standard_error, "{tips} tips centred gamma at {mean}");
        assert!((variance.sqrt() - 1.0).abs() < 0.2, "{tips} tips gave spread {}", variance.sqrt());
        // The rate should not enter: gamma is scale free in time.
        let scaled: Vec<f64> = (0..40)
            .map(|_| {
                gamma_statistic(&birth_death_tree(0.1, 0.0, tips, &mut rng).unwrap()).unwrap()
            })
            .collect();
        let slow: f64 = scaled.iter().sum::<f64>() / scaled.len() as f64;
        assert!(slow.abs() < 0.8, "a slower process shifted gamma to {slow}");
    }
}

#[test]
fn prop_gamma_is_invariant_under_rescaling_time() {
    // Multiplying every branch by a constant changes no proportion, so a
    // statistic about the *placement* of branchings cannot move.
    let mut rng = Rng::new(0x0B10_400F);
    for _ in 0..25 {
        let tips = 5 + pick(&mut rng, 20);
        let tree = random_tree(tips, &mut rng);
        let reference = gamma_statistic(&tree).unwrap();
        for scale in [0.001, 0.5, 3.0, 1000.0] {
            let stretched = PhyloTree::new(
                tree.parent.clone(),
                tree.branch_length.iter().map(|b| b * scale).collect(),
                tree.labels.clone(),
            )
            .unwrap();
            assert!(
                (gamma_statistic(&stretched).unwrap() - reference).abs() < 1e-8,
                "scaling by {scale} moved gamma"
            );
        }
    }
}

#[test]
fn prop_the_jukes_cantor_matrix_is_a_valid_distance_matrix() {
    // Symmetric, zero on the diagonal, positive off it, and always at or
    // above the raw proportion of differences.
    let mut rng = Rng::new(0x0B10_4010);
    for _ in 0..30 {
        let taxa = 3 + pick(&mut rng, 5);
        let root = random_dna(150, &mut rng);
        let seqs: Vec<Vec<u8>> =
            (0..taxa).map(|_| mutate(&root, 0.02 + 0.15 * rng.next_f64(), &mut rng)).collect();
        let Ok(dist) = distance_matrix_jc69(&seqs) else { continue };
        for i in 0..taxa {
            assert!(dist.get(i, i).abs() < 1e-15);
            for j in 0..taxa {
                assert!((dist.get(i, j) - dist.get(j, i)).abs() < 1e-15);
                assert!(dist.get(i, j) >= 0.0 && dist.get(i, j).is_finite());
                let raw = rust_physics_engine::biophysics::seq_align::p_distance(
                    &seqs[i], &seqs[j],
                )
                .unwrap();
                assert!(
                    dist.get(i, j) >= raw - 1e-12,
                    "the correction shrank {raw} to {}",
                    dist.get(i, j)
                );
            }
        }
    }
}
