//! Properties of the tree, forest and boosting module.
//!
//! *Exact identities.* Gini impurity is exactly zero for a pure node and
//! exactly `1 - 1/k` for `k` classes in equal proportion; entropy is
//! exactly `ln k`. Both are maximised by the uniform distribution, and
//! both vanish only for a pure node.
//!
//! *Bounds on what the model can express, not on how often it is right.*
//! A tree of depth `d` has at most `2^d` leaves and can therefore name
//! at most `2^d` distinct classes. That is a statement about the
//! hypothesis class and holds for every dataset; an accuracy bound
//! would not, because how well a stump does depends on how the class
//! sizes happen to fall.
//!
//! *Invariance.* Splits are decided by the order of a column's values,
//! not their magnitudes, so any increasing affine rescaling of any
//! feature leaves the tree computing the same function. This
//! distinguishes trees sharply from every distance-based method in this
//! crate, and it is asserted directly rather than described.
//!
//! *Conservation.* Feature importances are nonnegative and sum to
//! exactly the total weighted impurity the tree removed, so no credit
//! is created or lost in dividing it among the columns.
//!
//! *Monotonicity.* Boosting under squared loss walks its training loss
//! down at every round, and a tree of depth zero cannot split so it must
//! change nothing at all.

use rust_physics_engine::learn::tree::{
    decision_tree_fit, entropy, feature_importance, forest_predict, gbm_predict, gini,
    gradient_boosting_lite, random_forest_fit, regression_tree_fit, tree_predict,
    tree_predict_value, Tree, TreeNode,
};
use rust_physics_engine::monte_carlo::Rng;

fn design(rng: &mut Rng, n: usize, dim: usize) -> Vec<Vec<f64>> {
    (0..n).map(|_| (0..dim).map(|_| 4.0 * rng.next_f64() - 2.0).collect()).collect()
}

/// Labels from axis-aligned regions, which a tree can represent exactly.
fn regions(x: &[Vec<f64>]) -> Vec<usize> {
    x.iter().map(|p| usize::from(p[0] > 0.0) + 2 * usize::from(p[p.len() - 1] > 0.0)).collect()
}

/// How many distinct classes a tree ever predicts on a dataset.
fn named(tree: &Tree, x: &[Vec<f64>]) -> usize {
    x.iter()
        .map(|p| tree_predict(tree, p).unwrap())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

#[test]
fn prop_the_impurity_measures_are_exact_at_their_extremes() {
    let mut rng = Rng::new(0x2f80_c1a4);
    for _ in 0..40 {
        let k = 2 + (rng.below(7)) as usize;
        // Pure: exactly zero, both of them.
        let mut pure = vec![0usize; k];
        pure[(rng.below(k as u64)) as usize] = 1 + (rng.below(20)) as usize;
        assert_eq!(gini(&pure), 0.0);
        assert_eq!(entropy(&pure), 0.0);
        // Uniform: exactly 1 - 1/k and exactly ln k.
        let count = 1 + (rng.below(15)) as usize;
        let uniform = vec![count; k];
        assert!((gini(&uniform) - (1.0 - 1.0 / k as f64)).abs() < 1e-14);
        assert!((entropy(&uniform) - (k as f64).ln()).abs() < 1e-14);
        // Nothing beats uniform, and nothing is negative.
        let arbitrary: Vec<usize> = (0..k).map(|_| (rng.below(30)) as usize).collect();
        if arbitrary.iter().sum::<usize>() > 0 {
            assert!(gini(&arbitrary) <= gini(&uniform) + 1e-14);
            assert!(entropy(&arbitrary) <= entropy(&uniform) + 1e-14);
            assert!(gini(&arbitrary) >= 0.0);
            assert!(entropy(&arbitrary) >= 0.0);
            // Impurity is zero exactly when the node is pure.
            let nonzero = arbitrary.iter().filter(|&&c| c > 0).count();
            assert_eq!(gini(&arbitrary) == 0.0, nonzero == 1);
            assert_eq!(entropy(&arbitrary) == 0.0, nonzero == 1);
        }
        // Both are blind to the order of the counts and to scaling them
        // all by the same factor -- they see proportions.
        let mut shuffled = arbitrary.clone();
        shuffled.reverse();
        assert!((gini(&shuffled) - gini(&arbitrary)).abs() < 1e-14);
        let doubled: Vec<usize> = arbitrary.iter().map(|c| c * 2).collect();
        if doubled.iter().sum::<usize>() > 0 {
            assert!((gini(&doubled) - gini(&arbitrary)).abs() < 1e-14);
            assert!((entropy(&doubled) - entropy(&arbitrary)).abs() < 1e-14);
        }
    }
}

#[test]
fn prop_depth_bounds_what_a_tree_can_say() {
    // At most 2^d leaves, so at most 2^d distinct predictions -- a
    // statement about the model that holds whatever the data is.
    let mut rng = Rng::new(0x51d3_708b);
    for _ in 0..20 {
        let n = 30 + (rng.below(50)) as usize;
        let dim = 2 + (rng.below(3)) as usize;
        let x = design(&mut rng, n, dim);
        let y = regions(&x);
        let mut previous = 0.0;
        for depth in 1..=5usize {
            let tree = decision_tree_fit(&x, &y, depth, 1).unwrap();
            assert!(
                named(&tree, &x) <= 1usize << depth,
                "depth {depth} named more classes than it has leaves"
            );
            // Deeper is never less accurate on the training set, since
            // a deeper tree can reproduce a shallower one.
            let accuracy = x
                .iter()
                .enumerate()
                .filter(|(i, p)| tree_predict(&tree, p).unwrap() == y[*i])
                .count() as f64
                / n as f64;
            assert!(accuracy >= previous - 1e-12, "depth {depth} was less accurate");
            previous = accuracy;
        }
        // Unlimited depth separates everything separable. These labels
        // are a function of the features, so that is everything.
        let full = decision_tree_fit(&x, &y, usize::MAX, 1).unwrap();
        for (i, p) in x.iter().enumerate() {
            assert_eq!(tree_predict(&full, p).unwrap(), y[i], "point {i}");
        }
    }
}

#[test]
fn prop_a_tree_is_invariant_to_rescaling_any_feature() {
    // Increasing affine maps of a column leave the order of its values
    // alone, and the order is all a threshold sees. No distance-based
    // method in this crate can say the same.
    let mut rng = Rng::new(0x7ac0_1e36);
    for _ in 0..25 {
        let n = 25 + (rng.below(40)) as usize;
        let dim = 2 + (rng.below(3)) as usize;
        let x = design(&mut rng, n, dim);
        let y = regions(&x);
        let base = decision_tree_fit(&x, &y, 5, 2).unwrap();
        let column = (rng.below(dim as u64)) as usize;
        let factor = 0.001 + 1000.0 * rng.next_f64();
        let shift = 20.0 * rng.next_gaussian();
        let scaled: Vec<Vec<f64>> = x
            .iter()
            .map(|p| {
                let mut q = p.clone();
                q[column] = q[column] * factor + shift;
                q
            })
            .collect();
        let other = decision_tree_fit(&scaled, &y, 5, 2).unwrap();
        assert_eq!(base.nodes.len(), other.nodes.len(), "the structure changed");
        for (i, p) in x.iter().enumerate() {
            assert_eq!(
                tree_predict(&base, p).unwrap(),
                tree_predict(&other, &scaled[i]).unwrap(),
                "rescaling column {column} by {factor} moved point {i}"
            );
        }
        // Importances move with the column but keep their values.
        let a = feature_importance(&base);
        let b = feature_importance(&other);
        for j in 0..dim {
            assert!((a[j] - b[j]).abs() < 1e-9, "importance of column {j} changed");
        }
    }
}

#[test]
fn prop_importances_are_nonnegative_and_conserve_the_total() {
    let mut rng = Rng::new(0x0d47_29e1);
    for _ in 0..25 {
        let n = 30 + (rng.below(40)) as usize;
        let dim = 2 + (rng.below(4)) as usize;
        let x = design(&mut rng, n, dim);
        let y = regions(&x);
        let tree = decision_tree_fit(&x, &y, 6, 2).unwrap();
        let importance = feature_importance(&tree);
        assert_eq!(importance.len(), dim);
        assert!(importance.iter().all(|&v| v >= 0.0));
        // Every split really did reduce impurity: refusing is always
        // available, so a split with no gain is never taken.
        let root = match tree.nodes[0] {
            TreeNode::Split { samples, .. } => samples as f64,
            TreeNode::Leaf { .. } => continue,
        };
        let mut total = 0.0;
        for node in &tree.nodes {
            if let TreeNode::Split { decrease, samples, feature, .. } = node {
                assert!(*decrease > 0.0, "a split with decrease {decrease} was taken");
                assert!(*feature < dim);
                total += *samples as f64 / root * decrease;
            }
        }
        assert!(
            (importance.iter().sum::<f64>() - total).abs() < 1e-10 * total.max(1.0),
            "the importances did not sum to the total decrease"
        );
        // A column of pure noise earns less than the columns the labels
        // are actually built from.
        let padded: Vec<Vec<f64>> =
            x.iter().map(|p| { let mut q = p.clone(); q.push(rng.next_gaussian()); q }).collect();
        let wider = decision_tree_fit(&padded, &y, 3, 5).unwrap();
        let scores = feature_importance(&wider);
        let real = scores[0].max(scores[dim - 1]);
        assert!(scores[dim] <= real, "noise outranked every real feature");
    }
}

#[test]
fn prop_a_forest_votes_and_leaves_a_third_of_the_data_out() {
    let mut rng = Rng::new(0x63b1_4c07);
    for _ in 0..12 {
        let n = 40 + (rng.below(40)) as usize;
        let dim = 2 + (rng.below(3)) as usize;
        let x = design(&mut rng, n, dim);
        let y = regions(&x);
        let trees = 8 + (rng.below(12)) as usize;
        let forest = random_forest_fit(&x, &y, trees, 8, 1, 0, &mut rng).unwrap();
        assert_eq!(forest.trees.len(), trees);
        assert_eq!(forest.out_of_bag.len(), trees);
        for bag in &forest.out_of_bag {
            // A bootstrap of n draws misses each row with probability
            // (1 - 1/n)^n, which tends to 1/e. With these sample sizes
            // the fraction sits near a third; the band is generous
            // because the count is binomial.
            let fraction = bag.len() as f64 / n as f64;
            assert!((0.15..0.55).contains(&fraction), "out-of-bag fraction {fraction}");
            assert!(bag.iter().all(|&i| i < n));
        }
        // The vote is a real label and does not depend on tree order.
        let mut shuffled = forest.clone();
        shuffled.trees.reverse();
        for p in x.iter().take(10) {
            let vote = forest_predict(&forest, p).unwrap();
            assert!(y.contains(&vote), "a label nobody had was voted for");
            assert_eq!(vote, forest_predict(&shuffled, p).unwrap(), "order changed the vote");
        }
    }
}

#[test]
fn prop_boosting_walks_its_loss_down_and_a_stump_of_no_depth_does_nothing() {
    let mut rng = Rng::new(0x18ba_5d92);
    for _ in 0..15 {
        let n = 30 + (rng.below(40)) as usize;
        let dim = 1 + (rng.below(3)) as usize;
        let x = design(&mut rng, n, dim);
        let y: Vec<f64> = x.iter().map(|p| p[0].sin() + 0.1 * rng.next_gaussian()).collect();
        let rate = 0.05 + 0.9 * rng.next_f64();
        let rounds = 5 + (rng.below(20)) as usize;
        let model = gradient_boosting_lite(&x, &y, rounds, rate, 2).unwrap();
        assert_eq!(model.loss_history.len(), rounds + 1);
        for w in model.loss_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "the loss rose from {} to {}", w[0], w[1]);
        }
        // The first entry is the variance of the targets, since the
        // model starts by predicting their mean.
        let mean = y.iter().sum::<f64>() / n as f64;
        let variance = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        assert!((model.loss_history[0] - variance).abs() < 1e-10);
        assert!((model.base - mean).abs() < 1e-12);
        // Prediction agrees with the fit's own bookkeeping.
        let recomputed: f64 = x
            .iter()
            .zip(&y)
            .map(|(p, t)| {
                let e = gbm_predict(&model, p).unwrap() - t;
                e * e
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            (recomputed - model.loss_history[rounds]).abs() < 1e-9,
            "predict disagreed with the recorded loss"
        );
        // A tree of depth zero cannot split, so every round adds a
        // constant zero and the loss cannot move at all.
        let inert = gradient_boosting_lite(&x, &y, 4, 1.0, 0).unwrap();
        for w in inert.loss_history.windows(2) {
            assert!((w[1] - w[0]).abs() < 1e-12, "a zero-depth round changed the loss");
        }
    }
}

#[test]
fn prop_a_regression_tree_predicts_within_the_range_it_was_given() {
    // Every leaf is a mean of training targets, so no prediction can
    // leave their range. This is the same statement as "a tree never
    // extrapolates", and it is what makes trees useless for trends and
    // safe against runaway outputs.
    let mut rng = Rng::new(0x4ec7_1130);
    for _ in 0..25 {
        let n = 20 + (rng.below(40)) as usize;
        let dim = 1 + (rng.below(3)) as usize;
        let x = design(&mut rng, n, dim);
        let y: Vec<f64> = (0..n).map(|_| 10.0 * rng.next_gaussian()).collect();
        let tree = regression_tree_fit(&x, &y, 4, 2).unwrap();
        let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Inside the training set, and far outside it.
        for p in x.iter() {
            let got = tree_predict_value(&tree, p).unwrap();
            assert!(got >= lo - 1e-12 && got <= hi + 1e-12, "prediction {got} left [{lo}, {hi}]");
        }
        for _ in 0..10 {
            let far: Vec<f64> = (0..dim).map(|_| 1e6 * rng.next_gaussian()).collect();
            let got = tree_predict_value(&tree, &far).unwrap();
            assert!(
                got >= lo - 1e-12 && got <= hi + 1e-12,
                "a point far outside the data predicted {got}"
            );
        }
        // Grown without limit and with singleton leaves it reproduces
        // every target exactly, provided no two rows coincide.
        let full = regression_tree_fit(&x, &y, usize::MAX, 1).unwrap();
        let distinct: std::collections::BTreeSet<Vec<u64>> =
            x.iter().map(|p| p.iter().map(|v| v.to_bits()).collect()).collect();
        if distinct.len() == n {
            for (i, p) in x.iter().enumerate() {
                assert!(
                    (tree_predict_value(&full, p).unwrap() - y[i]).abs() < 1e-9,
                    "point {i} was not reproduced"
                );
            }
        }
    }
}
