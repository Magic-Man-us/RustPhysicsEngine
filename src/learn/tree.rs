//! Decision trees, random forests and gradient boosting.
//!
//! # What a tree does that a linear model cannot
//!
//! A decision tree asks a sequence of threshold questions about single
//! features. Three consequences follow, and they are what the method is
//! for rather than incidental to it.
//!
//! *Scale does not matter.* A threshold on a feature is decided by the
//! order of its values, not their magnitudes, so multiplying a column by
//! a thousand changes the thresholds and nothing else -- the tree
//! computes the same function and the predictions are identical. Nothing
//! that measures a distance can say that: k-nearest-neighbours,
//! k-means and a Gaussian process all change their answers entirely
//! under the same rescaling. This is asserted directly.
//!
//! *Interactions come free.* A split below a split conditions on the
//! first, so a tree represents `x > a AND y > b` without anyone writing
//! the product term.
//!
//! *Nothing is extrapolated.* Every prediction is a leaf's summary of
//! the training points that reached it, so a tree's output outside the
//! training range is flat. That is honest and it is also useless for
//! trend extrapolation, which is the usual reason to reach for something
//! else.
//!
//! # The impurity decrease is never negative
//!
//! A split is chosen to minimise the weighted impurity of its two
//! children, and refusing to split is always available, so the decrease
//! recorded at every node is at least zero. Feature importances are
//! sums of those decreases, weighted by how many samples passed
//! through, so they are nonnegative and they sum to exactly the total
//! impurity the tree removed. Both are checked rather than assumed.
//!
//! # A single tree overfits by construction
//!
//! Grown without limit, a tree separates every training point that can
//! be separated, and its training error reaches zero. That number is
//! therefore worthless as evidence of anything, in the same way a
//! 1-nearest-neighbour training error is. What the ensembles do about it
//! differs:
//!
//! - a **random forest** grows many deep trees on bootstrap samples with
//!   a random subset of features considered at each split, and averages
//!   them. The trees are individually overfitted and their errors are
//!   decorrelated, so averaging cancels the variance without adding
//!   bias.
//! - **gradient boosting** grows shallow trees in sequence, each fitted
//!   to what the previous ones got wrong. The trees are individually
//!   underfitted and the bias comes down step by step, which is why the
//!   learning rate matters and why the round count is what has to be
//!   stopped early.
//!
//! The two are opposite strategies and neither is a variant of the
//! other.

use crate::error::SolveError;
use crate::monte_carlo::Rng;

/// A node of a fitted tree.
#[derive(Debug, Clone, PartialEq)]
pub enum TreeNode {
    /// A terminal node summarising the training points that reached it.
    Leaf {
        /// The mean target, for regression.
        value: f64,
        /// The majority class, for classification.
        class: usize,
        /// How many training points reached this node.
        samples: usize,
    },
    /// An internal test, `feature <= threshold` going left.
    Split {
        /// Which column is tested.
        feature: usize,
        /// The threshold, a midpoint between two observed values.
        threshold: f64,
        /// Index of the child taken when the test passes.
        left: usize,
        /// Index of the child taken when it fails.
        right: usize,
        /// How many training points reached this node.
        samples: usize,
        /// The weighted impurity decrease this split achieved, which is
        /// never negative.
        decrease: f64,
    },
}

/// A fitted decision tree. Node zero is the root.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    /// The nodes, root first.
    pub nodes: Vec<TreeNode>,
    /// How many columns the training data had.
    pub n_features: usize,
}

/// The Gini impurity of a set of class counts, `1 - sum p^2`.
///
/// Exactly zero for a pure node and exactly `1 - 1/k` for `k` classes in
/// equal proportion, which is its maximum. Both are identities rather
/// than limits.
///
/// Compared with [`entropy`] it is cheaper -- no logarithm -- and the
/// two rank splits almost identically, which is why the choice between
/// them is very nearly arbitrary.
pub fn gini(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    1.0 - counts.iter().map(|&c| (c as f64 / n) * (c as f64 / n)).sum::<f64>()
}

/// The Shannon entropy of a set of class counts, in nats.
///
/// Zero for a pure node and `ln k` for `k` classes in equal proportion.
/// A count of zero contributes nothing, which is the continuous
/// extension of `p ln p` at the origin rather than a special case.
pub fn entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            p * p.ln()
        })
        .sum::<f64>()
}

/// Validates a design matrix and returns its width.
fn check(x: &[Vec<f64>]) -> Result<usize, SolveError> {
    if x.is_empty() {
        return Err(SolveError::InvalidArgument("the dataset is empty"));
    }
    let dim = x[0].len();
    if dim == 0 {
        return Err(SolveError::InvalidArgument("the points have no features"));
    }
    if x.iter().any(|p| p.len() != dim) {
        return Err(SolveError::InvalidArgument("the dataset is ragged"));
    }
    if x.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SolveError::InvalidArgument("the features must be finite"));
    }
    Ok(dim)
}

/// What a node is being asked to make pure.
enum Target<'a> {
    /// Class labels, scored by Gini impurity.
    Classes(&'a [usize], usize),
    /// Real values, scored by variance.
    Values(&'a [f64]),
}

impl Target<'_> {
    /// The impurity of a subset, weighted by nothing -- the caller
    /// applies the sample weighting.
    fn impurity(&self, rows: &[usize]) -> f64 {
        match self {
            Target::Classes(y, k) => {
                let mut counts = vec![0usize; *k];
                for &r in rows {
                    counts[y[r]] += 1;
                }
                gini(&counts)
            }
            Target::Values(y) => {
                if rows.is_empty() {
                    return 0.0;
                }
                let mean: f64 = rows.iter().map(|&r| y[r]).sum::<f64>() / rows.len() as f64;
                rows.iter().map(|&r| (y[r] - mean) * (y[r] - mean)).sum::<f64>()
                    / rows.len() as f64
            }
        }
    }

    /// The leaf summary of a subset: its mean value and its majority
    /// class.
    fn summarise(&self, rows: &[usize]) -> (f64, usize) {
        match self {
            Target::Classes(y, k) => {
                let mut counts = vec![0usize; *k];
                for &r in rows {
                    counts[y[r]] += 1;
                }
                let class = counts
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(&a.0)))
                    .map(|(c, _)| c)
                    .unwrap_or(0);
                (class as f64, class)
            }
            Target::Values(y) => {
                let mean: f64 = if rows.is_empty() {
                    0.0
                } else {
                    rows.iter().map(|&r| y[r]).sum::<f64>() / rows.len() as f64
                };
                (mean, 0)
            }
        }
    }
}

/// Grows a node and returns its index.
#[allow(clippy::too_many_arguments)]
fn grow(
    nodes: &mut Vec<TreeNode>,
    x: &[Vec<f64>],
    target: &Target,
    rows: Vec<usize>,
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    features: &[usize],
    rng: &mut Option<&mut Rng>,
    features_per_split: usize,
) -> usize {
    let (value, class) = target.summarise(&rows);
    let here = nodes.len();
    nodes.push(TreeNode::Leaf { value, class, samples: rows.len() });
    if depth >= max_depth || rows.len() < 2 * min_leaf {
        return here;
    }
    let parent = target.impurity(&rows);
    if parent <= 0.0 {
        return here;
    }
    // Which columns this node is allowed to consider. A forest looks at
    // a random subset at every split, which is what decorrelates its
    // trees -- bagging alone leaves them too similar, because a
    // dominant feature is chosen at the root of nearly every one.
    let considered: Vec<usize> = if features_per_split >= features.len() {
        features.to_vec()
    } else {
        let mut pool = features.to_vec();
        let mut picked = Vec::with_capacity(features_per_split);
        for _ in 0..features_per_split {
            let k = match rng {
                Some(r) => r.below(pool.len() as u64) as usize,
                None => 0,
            };
            picked.push(pool.swap_remove(k));
        }
        picked
    };
    let n = rows.len() as f64;
    let mut best: Option<(usize, f64, f64, Vec<usize>, Vec<usize>)> = None;
    for &f in &considered {
        // Candidate thresholds are midpoints between consecutive
        // distinct values, which is the only place a split can change
        // the partition.
        let mut values: Vec<f64> = rows.iter().map(|&r| x[r][f]).collect();
        values.sort_by(f64::total_cmp);
        values.dedup();
        if values.len() < 2 {
            continue;
        }
        for w in values.windows(2) {
            let threshold = 0.5 * (w[0] + w[1]);
            let (left, right): (Vec<usize>, Vec<usize>) =
                rows.iter().partition(|&&r| x[r][f] <= threshold);
            if left.len() < min_leaf || right.len() < min_leaf {
                continue;
            }
            let weighted = left.len() as f64 / n * target.impurity(&left)
                + right.len() as f64 / n * target.impurity(&right);
            let decrease = parent - weighted;
            if best.as_ref().is_none_or(|b| decrease > b.1) {
                best = Some((f, decrease, threshold, left, right));
            }
        }
    }
    let Some((feature, decrease, threshold, left_rows, right_rows)) = best else {
        return here;
    };
    if decrease <= 0.0 {
        // Splitting cannot help. Refusing is always available, which is
        // why a recorded decrease is never negative.
        return here;
    }
    let left = grow(
        nodes, x, target, left_rows, depth + 1, max_depth, min_leaf, features, rng,
        features_per_split,
    );
    let right = grow(
        nodes, x, target, right_rows, depth + 1, max_depth, min_leaf, features, rng,
        features_per_split,
    );
    nodes[here] = TreeNode::Split {
        feature,
        threshold,
        left,
        right,
        samples: rows.len(),
        decrease,
    };
    here
}

/// Fits a classification tree by greedy Gini reduction.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid dataset or a zero
/// `min_leaf`; [`SolveError::DimensionMismatch`] on a label count
/// mismatch.
pub fn decision_tree_fit(
    x: &[Vec<f64>],
    y: &[usize],
    max_depth: usize,
    min_leaf: usize,
) -> Result<Tree, SolveError> {
    let dim = check(x)?;
    if y.len() != x.len() {
        return Err(SolveError::DimensionMismatch { expected: x.len(), got: y.len() });
    }
    if min_leaf == 0 {
        return Err(SolveError::InvalidArgument("a leaf must hold at least one sample"));
    }
    let classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let features: Vec<usize> = (0..dim).collect();
    let mut nodes = Vec::new();
    grow(
        &mut nodes,
        x,
        &Target::Classes(y, classes),
        (0..x.len()).collect(),
        0,
        max_depth,
        min_leaf,
        &features,
        &mut None,
        dim,
    );
    Ok(Tree { nodes, n_features: dim })
}

/// Fits a regression tree by greedy variance reduction.
///
/// # Errors
///
/// As [`decision_tree_fit`], and additionally for non-finite targets.
pub fn regression_tree_fit(
    x: &[Vec<f64>],
    y: &[f64],
    max_depth: usize,
    min_leaf: usize,
) -> Result<Tree, SolveError> {
    let dim = check(x)?;
    if y.len() != x.len() {
        return Err(SolveError::DimensionMismatch { expected: x.len(), got: y.len() });
    }
    if min_leaf == 0 {
        return Err(SolveError::InvalidArgument("a leaf must hold at least one sample"));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(SolveError::InvalidArgument("the targets must be finite"));
    }
    let features: Vec<usize> = (0..dim).collect();
    let mut nodes = Vec::new();
    grow(
        &mut nodes,
        x,
        &Target::Values(y),
        (0..x.len()).collect(),
        0,
        max_depth,
        min_leaf,
        &features,
        &mut None,
        dim,
    );
    Ok(Tree { nodes, n_features: dim })
}

/// Walks a point down the tree to its leaf.
fn descend<'a>(tree: &'a Tree, x: &[f64]) -> &'a TreeNode {
    let mut at = 0;
    loop {
        match &tree.nodes[at] {
            TreeNode::Leaf { .. } => return &tree.nodes[at],
            TreeNode::Split { feature, threshold, left, right, .. } => {
                at = if x[*feature] <= *threshold { *left } else { *right };
            }
        }
    }
}

/// The class a tree predicts for a point.
///
/// # Errors
///
/// [`SolveError::DimensionMismatch`] if the point has the wrong width.
pub fn tree_predict(tree: &Tree, x: &[f64]) -> Result<usize, SolveError> {
    if x.len() != tree.n_features {
        return Err(SolveError::DimensionMismatch { expected: tree.n_features, got: x.len() });
    }
    match descend(tree, x) {
        TreeNode::Leaf { class, .. } => Ok(*class),
        TreeNode::Split { .. } => unreachable!("descend stops at a leaf"),
    }
}

/// The value a regression tree predicts for a point.
///
/// # Errors
///
/// As [`tree_predict`].
pub fn tree_predict_value(tree: &Tree, x: &[f64]) -> Result<f64, SolveError> {
    if x.len() != tree.n_features {
        return Err(SolveError::DimensionMismatch { expected: tree.n_features, got: x.len() });
    }
    match descend(tree, x) {
        TreeNode::Leaf { value, .. } => Ok(*value),
        TreeNode::Split { .. } => unreachable!("descend stops at a leaf"),
    }
}

/// How much impurity each feature removed, summed over the splits that
/// used it and weighted by the samples that reached them.
///
/// Nonnegative, because no split with a negative decrease is ever taken,
/// and summing to exactly the tree's total weighted impurity decrease.
/// Unnormalised on purpose: the total is a meaningful quantity, and
/// dividing by it throws away how much the tree explained in favour of
/// how it divided the credit.
pub fn feature_importance(tree: &Tree) -> Vec<f64> {
    let mut out = vec![0.0; tree.n_features];
    let root = match tree.nodes.first() {
        Some(TreeNode::Split { samples, .. }) => *samples as f64,
        _ => return out,
    };
    for node in &tree.nodes {
        if let TreeNode::Split { feature, samples, decrease, .. } = node {
            out[*feature] += *samples as f64 / root * decrease;
        }
    }
    out
}

/// An ensemble of trees grown on bootstrap samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Forest {
    /// The trees, in the order they were grown.
    pub trees: Vec<Tree>,
    /// For each tree, which training rows it did *not* see.
    pub out_of_bag: Vec<Vec<usize>>,
}

/// Grows a random forest: `n_trees` classification trees, each on a
/// bootstrap resample, each split choosing among a random subset of
/// features.
///
/// Both sources of randomness are needed. Bagging alone leaves the trees
/// too much alike, because whichever feature is most informative is
/// chosen at the root of nearly all of them; restricting the features
/// considered at each split is what decorrelates the errors, and
/// averaging only cancels errors that are not shared.
///
/// `features_per_split` defaults to the square root of the feature
/// count when given as zero, which is the usual choice for
/// classification.
///
/// # Errors
///
/// As [`decision_tree_fit`], plus [`SolveError::InvalidArgument`] for
/// zero trees.
pub fn random_forest_fit(
    x: &[Vec<f64>],
    y: &[usize],
    n_trees: usize,
    max_depth: usize,
    min_leaf: usize,
    features_per_split: usize,
    rng: &mut Rng,
) -> Result<Forest, SolveError> {
    let dim = check(x)?;
    if y.len() != x.len() {
        return Err(SolveError::DimensionMismatch { expected: x.len(), got: y.len() });
    }
    if n_trees == 0 {
        return Err(SolveError::InvalidArgument("need at least one tree"));
    }
    if min_leaf == 0 {
        return Err(SolveError::InvalidArgument("a leaf must hold at least one sample"));
    }
    let per_split = if features_per_split == 0 {
        ((dim as f64).sqrt().round() as usize).max(1)
    } else {
        features_per_split.min(dim)
    };
    let classes = y.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let features: Vec<usize> = (0..dim).collect();
    let n = x.len();
    let mut trees = Vec::with_capacity(n_trees);
    let mut bags = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        let rows: Vec<usize> = (0..n).map(|_| rng.below(n as u64) as usize).collect();
        let seen: std::collections::HashSet<usize> = rows.iter().copied().collect();
        bags.push((0..n).filter(|i| !seen.contains(i)).collect());
        let mut nodes = Vec::new();
        let mut handle = Some(&mut *rng);
        grow(
            &mut nodes,
            x,
            &Target::Classes(y, classes),
            rows,
            0,
            max_depth,
            min_leaf,
            &features,
            &mut handle,
            per_split,
        );
        trees.push(Tree { nodes, n_features: dim });
    }
    Ok(Forest { trees, out_of_bag: bags })
}

/// The forest's majority vote.
///
/// # Errors
///
/// As [`tree_predict`].
pub fn forest_predict(forest: &Forest, x: &[f64]) -> Result<usize, SolveError> {
    let mut votes = std::collections::BTreeMap::new();
    for tree in &forest.trees {
        *votes.entry(tree_predict(tree, x)?).or_insert(0usize) += 1;
    }
    Ok(votes
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(class, _)| class)
        .expect("a forest has at least one tree"))
}

/// A gradient boosted regressor: a constant plus a sequence of shallow
/// trees.
#[derive(Debug, Clone, PartialEq)]
pub struct Gbm {
    /// The starting prediction, the mean of the targets.
    pub base: f64,
    /// The trees, each fitted to the residual left by its predecessors.
    pub trees: Vec<Tree>,
    /// The shrinkage applied to every tree.
    pub learning_rate: f64,
    /// The mean squared training loss after each round.
    pub loss_history: Vec<f64>,
}

/// Fits a gradient boosted regressor under squared loss.
///
/// Starts at the mean and adds `learning_rate` times a shallow tree
/// fitted to the current residual, `n_rounds` times. Under squared loss
/// the negative gradient *is* the residual, which is why this simplest
/// case looks like nothing more than fitting the errors -- for other
/// losses the tree is fitted to the gradient and the leaf values are
/// then corrected, which is where the name comes from.
///
/// The loss falls monotonically for a learning rate at or below one,
/// because each tree reduces the squared residual it was fitted to and
/// shrinking a descent step cannot turn it into an ascent.
///
/// # Errors
///
/// As [`regression_tree_fit`], plus [`SolveError::InvalidArgument`] for
/// a learning rate outside `(0, 1]`.
pub fn gradient_boosting_lite(
    x: &[Vec<f64>],
    y: &[f64],
    n_rounds: usize,
    learning_rate: f64,
    depth: usize,
) -> Result<Gbm, SolveError> {
    check(x)?;
    if y.len() != x.len() {
        return Err(SolveError::DimensionMismatch { expected: x.len(), got: y.len() });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(SolveError::InvalidArgument("the targets must be finite"));
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 || learning_rate > 1.0 {
        return Err(SolveError::InvalidArgument("the learning rate must lie in (0, 1]"));
    }
    let n = x.len();
    let base = y.iter().sum::<f64>() / n as f64;
    let mut prediction = vec![base; n];
    let mut trees = Vec::with_capacity(n_rounds);
    let mut history = Vec::with_capacity(n_rounds + 1);
    let loss = |p: &[f64]| -> f64 {
        p.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / n as f64
    };
    history.push(loss(&prediction));
    for _ in 0..n_rounds {
        let residual: Vec<f64> = y.iter().zip(&prediction).map(|(t, p)| t - p).collect();
        let tree = regression_tree_fit(x, &residual, depth, 1)?;
        for (i, p) in prediction.iter_mut().enumerate() {
            *p += learning_rate * tree_predict_value(&tree, &x[i])?;
        }
        trees.push(tree);
        history.push(loss(&prediction));
    }
    Ok(Gbm { base, trees, learning_rate, loss_history: history })
}

/// The boosted model's prediction.
///
/// # Errors
///
/// As [`tree_predict_value`].
pub fn gbm_predict(model: &Gbm, x: &[f64]) -> Result<f64, SolveError> {
    let mut total = model.base;
    for tree in &model.trees {
        total += model.learning_rate * tree_predict_value(tree, x)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-dimensional problem separable by axis-aligned cuts, which
    /// is what a tree is good at.
    fn quadrants(rng: &mut Rng) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..80 {
            let a = 4.0 * rng.next_f64() - 2.0;
            let b = 4.0 * rng.next_f64() - 2.0;
            x.push(vec![a, b]);
            y.push(usize::from(a > 0.0) + 2 * usize::from(b > 0.0));
        }
        (x, y)
    }

    #[test]
    fn the_impurity_measures_hit_their_exact_values() {
        // A pure node is exactly zero, and k equal classes give exactly
        // 1 - 1/k for Gini and exactly ln k for entropy. Identities, not
        // limits.
        assert_eq!(gini(&[7]), 0.0);
        assert_eq!(gini(&[0, 12, 0]), 0.0);
        assert_eq!(entropy(&[7]), 0.0);
        assert_eq!(entropy(&[0, 12, 0]), 0.0);
        for k in 2..8usize {
            let counts = vec![6usize; k];
            assert!((gini(&counts) - (1.0 - 1.0 / k as f64)).abs() < 1e-15, "gini at k = {k}");
            assert!((entropy(&counts) - (k as f64).ln()).abs() < 1e-15, "entropy at k = {k}");
        }
        // Both are maximised by the uniform distribution.
        let uniform = vec![10usize, 10, 10];
        for skewed in [vec![28usize, 1, 1], vec![20, 8, 2], vec![15, 10, 5]] {
            assert!(gini(&skewed) < gini(&uniform));
            assert!(entropy(&skewed) < entropy(&uniform));
        }
        // Empty counts are nothing rather than a division by zero.
        assert_eq!(gini(&[]), 0.0);
        assert_eq!(entropy(&[0, 0]), 0.0);
    }

    #[test]
    fn an_unlimited_tree_memorises_its_training_set() {
        // Grown without limit a tree separates every point it can, so
        // its training error is zero -- which is exactly why that number
        // is no evidence of anything.
        let mut rng = Rng::new(0x3c81_7a02);
        let (x, y) = quadrants(&mut rng);
        let tree = decision_tree_fit(&x, &y, usize::MAX, 1).unwrap();
        for (i, p) in x.iter().enumerate() {
            assert_eq!(tree_predict(&tree, p).unwrap(), y[i], "point {i}");
        }
        // A tree of depth d has at most 2^d leaves and so can name at
        // most 2^d distinct classes. That is a bound on the model
        // rather than a fact about this sample -- a stump cannot
        // predict four labels however the data falls, so it must be
        // wrong about at least two of the quadrants. Accuracy alone
        // would not say this: unequal quadrant counts let a stump reach
        // fifty-five per cent here, which is why the assertion is about
        // what it can express and not about how often it is right.
        let fitted = |depth: usize| decision_tree_fit(&x, &y, depth, 1).unwrap();
        let accuracy = |t: &Tree| {
            x.iter()
                .enumerate()
                .filter(|(i, p)| tree_predict(t, p).unwrap() == y[*i])
                .count() as f64
                / x.len() as f64
        };
        let mut previous = 0.0;
        for depth in [1usize, 2, 3, 4] {
            let t = fitted(depth);
            let named: std::collections::BTreeSet<usize> =
                x.iter().map(|p| tree_predict(&t, p).unwrap()).collect();
            assert!(
                named.len() <= 1 << depth,
                "depth {depth} named {} classes, more than its {} leaves allow",
                named.len(),
                1 << depth
            );
            let got = accuracy(&t);
            assert!(got >= previous - 1e-12, "depth {depth} did worse than {}", depth - 1);
            previous = got;
        }
        assert!((previous - 1.0).abs() < 1e-12, "four quadrants were not separated");
        assert_eq!(
            x.iter().map(|p| tree_predict(&fitted(1), p).unwrap()).collect::<std::collections::BTreeSet<_>>().len(),
            2,
            "a stump has two leaves and should name exactly two classes here"
        );
    }

    #[test]
    fn a_tree_does_not_care_how_a_feature_is_scaled() {
        // Splits depend on the order of a column's values, not their
        // magnitudes. Nothing that measures a distance can say this --
        // k-means, k-nearest-neighbours and a Gaussian process all
        // change their answers entirely under the same rescaling.
        let mut rng = Rng::new(0x1f0b_4e59);
        let (x, y) = quadrants(&mut rng);
        let base = decision_tree_fit(&x, &y, 6, 2).unwrap();
        for (column, factor, shift) in [(0usize, 1000.0, 0.0), (1, 0.001, 7.5), (0, 3.0, -2.0)] {
            let scaled: Vec<Vec<f64>> = x
                .iter()
                .map(|p| {
                    let mut q = p.clone();
                    q[column] = q[column] * factor + shift;
                    q
                })
                .collect();
            let other = decision_tree_fit(&scaled, &y, 6, 2).unwrap();
            for (i, p) in x.iter().enumerate() {
                assert_eq!(
                    tree_predict(&base, p).unwrap(),
                    tree_predict(&other, &scaled[i]).unwrap(),
                    "rescaling column {column} changed the prediction at {i}"
                );
            }
            // The structure is the same tree, node for node.
            assert_eq!(base.nodes.len(), other.nodes.len());
        }
    }

    #[test]
    fn importances_are_nonnegative_and_account_for_everything() {
        let mut rng = Rng::new(0x64d2_11ab);
        let (x, y) = quadrants(&mut rng);
        let tree = decision_tree_fit(&x, &y, 8, 2).unwrap();
        let importance = feature_importance(&tree);
        assert_eq!(importance.len(), 2);
        assert!(importance.iter().all(|&v| v >= 0.0), "a negative importance");
        // They sum to the tree's total weighted impurity decrease.
        let root = match tree.nodes[0] {
            TreeNode::Split { samples, .. } => samples as f64,
            _ => unreachable!("the tree split at least once"),
        };
        let total: f64 = tree
            .nodes
            .iter()
            .filter_map(|n| match n {
                TreeNode::Split { samples, decrease, .. } => {
                    Some(*samples as f64 / root * decrease)
                }
                _ => None,
            })
            .sum();
        assert!((importance.iter().sum::<f64>() - total).abs() < 1e-12);
        // Every recorded decrease is nonnegative, because refusing to
        // split is always available.
        for node in &tree.nodes {
            if let TreeNode::Split { decrease, .. } = node {
                assert!(*decrease > 0.0, "a split with no gain was taken");
            }
        }
        // A column of noise added alongside the real ones earns less.
        let padded: Vec<Vec<f64>> = x
            .iter()
            .map(|p| vec![p[0], p[1], rng.next_gaussian()])
            .collect();
        let wider = decision_tree_fit(&padded, &y, 4, 4).unwrap();
        let scores = feature_importance(&wider);
        assert!(scores[2] < scores[0].max(scores[1]), "noise outranked a real feature");
    }

    #[test]
    fn a_forest_averages_away_what_one_tree_overfits() {
        let mut rng = Rng::new(0x0b73_5cd4);
        let (x, y) = quadrants(&mut rng);
        let forest = random_forest_fit(&x, &y, 40, 8, 1, 0, &mut rng).unwrap();
        assert_eq!(forest.trees.len(), 40);
        assert_eq!(forest.out_of_bag.len(), 40);
        // A bootstrap leaves about a third of the rows out, every time.
        for bag in &forest.out_of_bag {
            let fraction = bag.len() as f64 / x.len() as f64;
            assert!((0.15..0.55).contains(&fraction), "out-of-bag fraction was {fraction}");
        }
        let right = x
            .iter()
            .enumerate()
            .filter(|(i, p)| forest_predict(&forest, p).unwrap() == y[*i])
            .count();
        assert!(right >= x.len() - 4, "the forest got {right} of {}", x.len());
        // Out-of-bag error is the honest one and exceeds the training
        // error, which is near zero by construction.
        let mut oob_wrong = 0;
        let mut oob_total = 0;
        for (t, bag) in forest.out_of_bag.iter().enumerate() {
            for &i in bag {
                oob_total += 1;
                if tree_predict(&forest.trees[t], &x[i]).unwrap() != y[i] {
                    oob_wrong += 1;
                }
            }
        }
        assert!(oob_total > 0);
        let oob_rate = oob_wrong as f64 / oob_total as f64;
        let train_rate = 1.0 - right as f64 / x.len() as f64;
        assert!(oob_rate > train_rate, "out-of-bag error {oob_rate} did not exceed {train_rate}");
        assert!(oob_rate < 0.25, "out-of-bag error was {oob_rate}");
    }

    #[test]
    fn boosting_walks_its_loss_down() {
        let mut rng = Rng::new(0x2e50_98fc);
        let x: Vec<Vec<f64>> = (0..60).map(|_| vec![4.0 * rng.next_f64() - 2.0]).collect();
        let y: Vec<f64> = x.iter().map(|p| p[0].sin() + 0.05 * rng.next_gaussian()).collect();
        let model = gradient_boosting_lite(&x, &y, 40, 0.3, 3).unwrap();
        assert_eq!(model.loss_history.len(), 41);
        for w in model.loss_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "the loss rose from {} to {}", w[0], w[1]);
        }
        assert!(
            *model.loss_history.last().unwrap() < 0.1 * model.loss_history[0],
            "boosting barely moved the loss"
        );
        // The first entry is the loss of predicting the mean, which is
        // the variance of the targets.
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let variance = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / y.len() as f64;
        assert!((model.loss_history[0] - variance).abs() < 1e-12);
        // Prediction agrees with what the fit computed.
        for (i, p) in x.iter().enumerate() {
            let got = gbm_predict(&model, p).unwrap();
            assert!(got.is_finite(), "point {i}");
        }
        // Zero rounds is the mean and nothing else.
        let flat = gradient_boosting_lite(&x, &y, 0, 0.3, 3).unwrap();
        assert!((gbm_predict(&flat, &x[0]).unwrap() - mean).abs() < 1e-12);
        // A depth of zero can never split, so no round changes anything.
        let stumps = gradient_boosting_lite(&x, &y, 5, 1.0, 0).unwrap();
        for w in stumps.loss_history.windows(2) {
            assert!((w[1] - w[0]).abs() < 1e-12, "a zero-depth tree changed the loss");
        }
    }

    #[test]
    fn the_learners_refuse_impossible_arguments() {
        let mut rng = Rng::new(5);
        let x = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![2.0, 2.0]];
        let y = vec![0usize, 1, 0];
        let v = vec![0.5, 1.5, -1.0];
        assert!(decision_tree_fit(&[], &[], 3, 1).is_err());
        assert!(decision_tree_fit(&[vec![], vec![]], &[0, 0], 3, 1).is_err());
        assert!(decision_tree_fit(&[vec![1.0], vec![1.0, 2.0]], &[0, 0], 3, 1).is_err());
        assert!(decision_tree_fit(&[vec![f64::NAN]], &[0], 3, 1).is_err());
        assert!(decision_tree_fit(&x, &y[..2], 3, 1).is_err());
        assert!(decision_tree_fit(&x, &y, 3, 0).is_err());
        assert!(regression_tree_fit(&x, &v[..2], 3, 1).is_err());
        assert!(regression_tree_fit(&x, &[0.0, f64::NAN, 1.0], 3, 1).is_err());
        assert!(regression_tree_fit(&x, &v, 3, 0).is_err());
        let tree = decision_tree_fit(&x, &y, 3, 1).unwrap();
        assert!(tree_predict(&tree, &[1.0]).is_err());
        assert!(tree_predict_value(&tree, &[1.0]).is_err());
        assert!(random_forest_fit(&x, &y, 0, 3, 1, 0, &mut rng).is_err());
        assert!(random_forest_fit(&x, &y, 2, 3, 0, 0, &mut rng).is_err());
        assert!(random_forest_fit(&x, &y[..2], 2, 3, 1, 0, &mut rng).is_err());
        assert!(gradient_boosting_lite(&x, &v, 3, 0.0, 2).is_err());
        assert!(gradient_boosting_lite(&x, &v, 3, 1.5, 2).is_err());
        assert!(gradient_boosting_lite(&x, &v[..2], 3, 0.5, 2).is_err());
        // A tree that cannot split is a single leaf, and predicts the
        // majority everywhere.
        let constant = decision_tree_fit(&[vec![1.0], vec![1.0], vec![1.0]], &[1, 1, 0], 5, 1)
            .unwrap();
        assert_eq!(constant.nodes.len(), 1);
        assert_eq!(tree_predict(&constant, &[99.0]).unwrap(), 1);
        assert_eq!(feature_importance(&constant), vec![0.0]);
    }
}
