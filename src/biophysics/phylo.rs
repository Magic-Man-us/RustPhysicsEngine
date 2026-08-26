//! Phylogenetics: trees, the distance and character methods that build them,
//! and the statistics read off them.
//!
//! # What a tree is here
//!
//! [`PhyloTree`] stores a parent index and a branch length per node, with
//! leaves first and internal nodes after. That representation makes the two
//! operations everything else needs -- walking to the root, and finding a
//! common ancestor -- direct, at the cost of making "children of" a search.
//! Trees in this module are rooted; an unrooted method such as neighbour
//! joining produces a tree whose root is an artefact of the construction and
//! carries no meaning, which is noted where it matters.
//!
//! # Distances are not times
//!
//! A branch length is a number of substitutions per site, not an elapsed
//! time, and converting between them needs a rate that no method here
//! estimates. UPGMA is the exception and it is an *assumption* rather than
//! an inference: it produces an ultrametric tree, in which every leaf is
//! equidistant from the root, which is true only under a strict molecular
//! clock. Neighbour joining makes no such assumption, and the difference
//! shows immediately on data where rates vary between lineages.

use crate::error::GeomError;
use crate::linalg::Matrix;
use crate::monte_carlo::Rng;

/// A rooted phylogenetic tree.
///
/// Nodes `0..leaf_count` are leaves; the rest are internal. The root is the
/// unique node whose parent is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhyloTree {
    /// The parent of each node, or `None` for the root.
    pub parent: Vec<Option<usize>>,
    /// The length of the branch above each node. The root's is zero.
    pub branch_length: Vec<f64>,
    /// Labels, empty for unnamed internal nodes.
    pub labels: Vec<String>,
}

impl PhyloTree {
    /// A tree from its arrays, checked for consistency.
    ///
    /// # Errors
    /// Returns an error for mismatched lengths, a negative branch, no root
    /// or more than one, a parent index out of range, or a cycle.
    pub fn new(
        parent: Vec<Option<usize>>,
        branch_length: Vec<f64>,
        labels: Vec<String>,
    ) -> Result<Self, GeomError> {
        let n = parent.len();
        if n == 0 || branch_length.len() != n || labels.len() != n {
            return Err(GeomError::InvalidArgument("PhyloTree: mismatched arrays"));
        }
        if branch_length.iter().any(|b| *b < 0.0 || !b.is_finite()) {
            return Err(GeomError::InvalidArgument("a branch length is negative or not finite"));
        }
        if parent.iter().flatten().any(|p| *p >= n) {
            return Err(GeomError::InvalidArgument("a parent index is out of range"));
        }
        if parent.iter().filter(|p| p.is_none()).count() != 1 {
            return Err(GeomError::InvalidArgument("a tree has exactly one root"));
        }
        // Every node must reach the root in at most n steps, which rules out
        // a cycle without a separate traversal.
        for start in 0..n {
            let mut node = start;
            let mut steps = 0;
            while let Some(next) = parent[node] {
                node = next;
                steps += 1;
                if steps > n {
                    return Err(GeomError::InvalidArgument("the parent links contain a cycle"));
                }
            }
        }
        Ok(Self { parent, branch_length, labels })
    }

    /// The number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Whether the tree has no nodes. Never true for a constructed tree.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// The root.
    #[must_use]
    pub fn root(&self) -> usize {
        self.parent.iter().position(std::option::Option::is_none).expect("a tree has a root")
    }

    /// The children of a node, in index order.
    #[must_use]
    pub fn children(&self, node: usize) -> Vec<usize> {
        (0..self.len()).filter(|k| self.parent[*k] == Some(node)).collect()
    }

    /// The leaves: nodes with no children, in index order.
    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        let mut has_child = vec![false; self.len()];
        for p in self.parent.iter().flatten() {
            has_child[*p] = true;
        }
        (0..self.len()).filter(|k| !has_child[*k]).collect()
    }

    /// Whether every internal node has exactly two children.
    ///
    /// A tree that is not binary has an unresolved node -- a polytomy --
    /// which usually means the data could not distinguish the orders, not
    /// that three lineages truly diverged at once.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        let leaves = self.leaves();
        (0..self.len())
            .filter(|k| !leaves.contains(k))
            .all(|k| self.children(k).len() == 2)
    }

    /// The path from a node to the root, inclusive of both.
    #[must_use]
    pub fn path_to_root(&self, mut node: usize) -> Vec<usize> {
        let mut path = vec![node];
        while let Some(next) = self.parent[node] {
            path.push(next);
            node = next;
        }
        path
    }

    /// The distance from a node to the root, summing branch lengths.
    #[must_use]
    pub fn depth(&self, node: usize) -> f64 {
        let mut total = 0.0;
        let mut current = node;
        while let Some(next) = self.parent[current] {
            total += self.branch_length[current];
            current = next;
        }
        total
    }

    /// The greatest root-to-leaf distance.
    #[must_use]
    pub fn height(&self) -> f64 {
        self.leaves().into_iter().map(|leaf| self.depth(leaf)).fold(0.0, f64::max)
    }

    /// The sum of every branch length.
    #[must_use]
    pub fn total_length(&self) -> f64 {
        let root = self.root();
        (0..self.len()).filter(|k| *k != root).map(|k| self.branch_length[k]).sum()
    }

    /// The most recent common ancestor of two nodes.
    ///
    /// # Errors
    /// Returns an error for a node index out of range.
    pub fn mrca(&self, a: usize, b: usize) -> Result<usize, GeomError> {
        if a >= self.len() || b >= self.len() {
            return Err(GeomError::InvalidArgument("a node index is out of range"));
        }
        let path = self.path_to_root(a);
        let mut node = b;
        loop {
            if path.contains(&node) {
                return Ok(node);
            }
            match self.parent[node] {
                Some(next) => node = next,
                None => return Ok(self.root()),
            }
        }
    }

    /// The patristic distance: the path length between two nodes through
    /// their common ancestor.
    ///
    /// # Errors
    /// Returns an error for a node index out of range.
    pub fn distance(&self, a: usize, b: usize) -> Result<f64, GeomError> {
        let ancestor = self.mrca(a, b)?;
        Ok(self.depth(a) + self.depth(b) - 2.0 * self.depth(ancestor))
    }

    /// Whether the tree is ultrametric: every leaf the same distance from
    /// the root.
    ///
    /// True under a strict molecular clock and rarely otherwise. UPGMA
    /// *imposes* it; neighbour joining does not.
    #[must_use]
    pub fn is_ultrametric(&self, tolerance: f64) -> bool {
        let depths: Vec<f64> = self.leaves().into_iter().map(|leaf| self.depth(leaf)).collect();
        match depths.first() {
            None => true,
            Some(first) => depths.iter().all(|d| (d - first).abs() <= tolerance),
        }
    }

    /// The set of leaf labels below each internal node: the tree's splits.
    ///
    /// Two trees describe the same topology exactly when they induce the
    /// same splits, which is what [`PhyloTree::robinson_foulds`] compares.
    #[must_use]
    pub fn splits(&self) -> Vec<Vec<String>> {
        let leaves = self.leaves();
        let root = self.root();
        let mut out = Vec::new();
        for node in 0..self.len() {
            if node == root || leaves.contains(&node) {
                continue;
            }
            let mut below: Vec<String> = leaves
                .iter()
                .filter(|leaf| self.path_to_root(**leaf).contains(&node))
                .map(|leaf| self.labels[*leaf].clone())
                .collect();
            below.sort();
            // A split covering every leaf is the trivial one and carries no
            // information about the topology.
            if below.len() > 1 && below.len() < leaves.len() {
                out.push(below);
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// The tree's splits as *unrooted* bipartitions.
    ///
    /// Each internal node divides the leaves in two, and on an unrooted
    /// tree neither side is "below" the other -- `{A,B}` and `{C,D}` on a
    /// four-taxon tree name the same branch. Each bipartition is therefore
    /// reported by its smaller side, with ties broken alphabetically, so
    /// the two descriptions collapse to one. Bipartitions with fewer than
    /// two leaves on a side are trivial and omitted.
    ///
    /// This is what to compare when the rooting is an artefact of the
    /// method, as it is for [`neighbor_joining`], and what bootstrap
    /// support is conventionally reported on.
    #[must_use]
    pub fn bipartitions(&self) -> Vec<Vec<String>> {
        let leaves = self.leaves();
        let mut all: Vec<String> = leaves.iter().map(|k| self.labels[*k].clone()).collect();
        all.sort();
        let root = self.root();
        let mut out = Vec::new();
        for node in 0..self.len() {
            if node == root {
                continue;
            }
            let mut side: Vec<String> = leaves
                .iter()
                .filter(|leaf| self.path_to_root(**leaf).contains(&node))
                .map(|leaf| self.labels[*leaf].clone())
                .collect();
            side.sort();
            let mut other: Vec<String> = all.iter().filter(|l| !side.contains(l)).cloned().collect();
            other.sort();
            if side.len() < 2 || other.len() < 2 {
                continue;
            }
            out.push(if (side.len(), &side) <= (other.len(), &other) { side } else { other });
        }
        out.sort();
        out.dedup();
        out
    }

    /// The Robinson-Foulds distance: the number of splits present in one
    /// tree and not the other.
    ///
    /// Splits here are rooted clades, so two trees that differ only in
    /// where the root sits score above zero. Compare
    /// [`PhyloTree::bipartitions`] instead when the rooting carries no
    /// meaning.
    ///
    /// A topological measure that ignores branch lengths entirely, which is
    /// both its use and its weakness -- two trees can differ by one badly
    /// placed leaf and score the maximum, so the raw number is hard to
    /// interpret without normalising by the possible total.
    ///
    /// # Errors
    /// Returns an error if the two trees do not have the same leaf labels.
    pub fn robinson_foulds(&self, other: &PhyloTree) -> Result<usize, GeomError> {
        let mut mine: Vec<String> =
            self.leaves().into_iter().map(|k| self.labels[k].clone()).collect();
        let mut theirs: Vec<String> =
            other.leaves().into_iter().map(|k| other.labels[k].clone()).collect();
        mine.sort();
        theirs.sort();
        if mine != theirs {
            return Err(GeomError::InvalidArgument("the trees have different leaf sets"));
        }
        let a = self.splits();
        let b = other.splits();
        let only_a = a.iter().filter(|s| !b.contains(s)).count();
        let only_b = b.iter().filter(|s| !a.contains(s)).count();
        Ok(only_a + only_b)
    }

    /// The tree in Newick format, with branch lengths.
    #[must_use]
    pub fn to_newick(&self) -> String {
        fn render(tree: &PhyloTree, node: usize, root: usize) -> String {
            let children = tree.children(node);
            let body = if children.is_empty() {
                tree.labels[node].clone()
            } else {
                let inner: Vec<String> =
                    children.into_iter().map(|c| render(tree, c, root)).collect();
                format!("({}){}", inner.join(","), tree.labels[node])
            };
            if node == root {
                body
            } else {
                format!("{body}:{}", tree.branch_length[node])
            }
        }
        format!("{};", render(self, self.root(), self.root()))
    }

    /// Parses a Newick string.
    ///
    /// Accepts the common subset: nested parentheses, optional labels, and
    /// optional `:length` suffixes, terminated by a semicolon.
    ///
    /// # Errors
    /// Returns an error for unbalanced parentheses, a missing semicolon, a
    /// malformed branch length, or an empty tree.
    pub fn from_newick(text: &str) -> Result<Self, GeomError> {
        let trimmed = text.trim();
        let body = trimmed
            .strip_suffix(';')
            .ok_or(GeomError::InvalidArgument("a Newick string ends with a semicolon"))?;
        if body.trim().is_empty() {
            return Err(GeomError::InvalidArgument("the tree is empty"));
        }
        let bytes: Vec<char> = body.chars().collect();
        let mut parent: Vec<Option<usize>> = Vec::new();
        let mut branch_length: Vec<f64> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        let mut position = 0usize;
        let root = parse_node(&bytes, &mut position, &mut parent, &mut branch_length, &mut labels)?;
        // Skip trailing whitespace.
        while position < bytes.len() && bytes[position].is_whitespace() {
            position += 1;
        }
        if position != bytes.len() {
            return Err(GeomError::InvalidArgument("trailing characters after the tree"));
        }
        parent[root] = None;
        branch_length[root] = 0.0;
        // Reorder so leaves come first, which is the invariant the rest of
        // the module relies on.
        let temporary = PhyloTree { parent, branch_length, labels };
        Ok(reorder_leaves_first(&temporary))
    }
}

/// Parses one node and its subtree, appending to the arrays and returning
/// its index.
fn parse_node(
    text: &[char],
    position: &mut usize,
    parent: &mut Vec<Option<usize>>,
    branch_length: &mut Vec<f64>,
    labels: &mut Vec<String>,
) -> Result<usize, GeomError> {
    while *position < text.len() && text[*position].is_whitespace() {
        *position += 1;
    }
    if *position >= text.len() {
        return Err(GeomError::InvalidArgument("the Newick string ended early"));
    }
    let mut children = Vec::new();
    if text[*position] == '(' {
        *position += 1;
        loop {
            let child =
                parse_node(text, position, parent, branch_length, labels)?;
            children.push(child);
            while *position < text.len() && text[*position].is_whitespace() {
                *position += 1;
            }
            match text.get(*position) {
                Some(',') => *position += 1,
                Some(')') => {
                    *position += 1;
                    break;
                }
                _ => return Err(GeomError::InvalidArgument("unbalanced parentheses")),
            }
        }
        if children.len() < 2 {
            return Err(GeomError::InvalidArgument("an internal node needs two children"));
        }
    }
    // The label, then an optional branch length.
    let start = *position;
    while *position < text.len()
        && !matches!(text[*position], ',' | ')' | ':' | '(')
    {
        *position += 1;
    }
    let label: String = text[start..*position].iter().collect::<String>().trim().to_string();
    let mut length = 0.0;
    if text.get(*position) == Some(&':') {
        *position += 1;
        let number_start = *position;
        while *position < text.len() && !matches!(text[*position], ',' | ')') {
            *position += 1;
        }
        let raw: String = text[number_start..*position].iter().collect();
        length = raw
            .trim()
            .parse::<f64>()
            .map_err(|_| GeomError::InvalidArgument("a branch length is not a number"))?;
        if length < 0.0 || !length.is_finite() {
            return Err(GeomError::InvalidArgument("a branch length is negative or not finite"));
        }
    }
    let index = parent.len();
    parent.push(None);
    branch_length.push(length);
    labels.push(label);
    for child in children {
        parent[child] = Some(index);
    }
    Ok(index)
}

/// Renumbers a tree so leaves occupy the low indices.
fn reorder_leaves_first(tree: &PhyloTree) -> PhyloTree {
    let leaves = tree.leaves();
    let mut order: Vec<usize> = leaves.clone();
    order.extend((0..tree.len()).filter(|k| !leaves.contains(k)));
    let mut position = vec![0usize; tree.len()];
    for (new, old) in order.iter().enumerate() {
        position[*old] = new;
    }
    PhyloTree {
        parent: order.iter().map(|old| tree.parent[*old].map(|p| position[p])).collect(),
        branch_length: order.iter().map(|old| tree.branch_length[*old]).collect(),
        labels: order.iter().map(|old| tree.labels[*old].clone()).collect(),
    }
}

// ---------------------------------------------------------------------------
// Distance methods
// ---------------------------------------------------------------------------

/// Which distance method a bootstrap replicate should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMethod {
    /// Average linkage clustering, which imposes a molecular clock.
    Upgma,
    /// Neighbour joining, which does not.
    NeighborJoining,
}

/// Checks a distance matrix and returns its size.
fn check_distances(dist: &Matrix, labels: &[String], least: usize) -> Result<usize, GeomError> {
    let n = dist.rows;
    if dist.cols != n || labels.len() != n {
        return Err(GeomError::InvalidArgument(
            "the distance matrix must be square and match the labels",
        ));
    }
    if n < least {
        return Err(GeomError::InvalidArgument("too few taxa for this method"));
    }
    for i in 0..n {
        if dist.get(i, i) != 0.0 {
            return Err(GeomError::InvalidArgument("a distance from a taxon to itself is not zero"));
        }
        for j in 0..n {
            let d = dist.get(i, j);
            if d < 0.0 || !d.is_finite() {
                return Err(GeomError::InvalidArgument("a distance is negative or not finite"));
            }
            if (d - dist.get(j, i)).abs() > 1e-9 * d.abs().max(1.0) {
                return Err(GeomError::InvalidArgument("the distance matrix is not symmetric"));
            }
        }
    }
    Ok(n)
}

/// UPGMA: unweighted pair group method with arithmetic mean.
///
/// Repeatedly joins the two closest clusters and places their common
/// ancestor at half their distance, so every leaf ends up the same distance
/// from the root. That ultrametricity is *assumed*, not measured: UPGMA
/// returns a clocklike tree whether or not the data are clocklike, and on
/// data where one lineage evolves faster it will place that lineage's
/// long branch too close to the root -- the classic long-branch artefact.
/// Use [`neighbor_joining`] unless a clock is justified.
///
/// The distance between merged clusters is the mean over all pairs of
/// members, which is what makes the merge heights non-decreasing and the
/// result a valid ultrametric tree.
///
/// # Errors
/// Returns an error for a non-square, asymmetric, negative or non-finite
/// matrix, a label count that disagrees with it, or fewer than two taxa.
pub fn upgma(dist: &Matrix, labels: &[String]) -> Result<PhyloTree, GeomError> {
    let n = check_distances(dist, labels, 2)?;
    let total_nodes = 2 * n - 1;
    let mut parent: Vec<Option<usize>> = vec![None; total_nodes];
    let mut branch_length = vec![0.0; total_nodes];
    let mut names: Vec<String> = labels.to_vec();
    names.resize(total_nodes, String::new());

    // active[k] is the node index of cluster k; size and height track it.
    let mut active: Vec<usize> = (0..n).collect();
    let mut size: Vec<f64> = vec![1.0; n];
    let mut height: Vec<f64> = vec![0.0; n];
    let mut d: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| dist.get(i, j)).collect()).collect();
    let mut next_node = n;

    while active.len() > 1 {
        let (mut bi, mut bj, mut best) = (0usize, 1usize, f64::INFINITY);
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                if d[i][j] < best {
                    best = d[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }
        let new_height = 0.5 * best;
        let node = next_node;
        next_node += 1;
        for side in [bi, bj] {
            parent[active[side]] = Some(node);
            // Average linkage cannot invert, so this subtraction is
            // non-negative in exact arithmetic; the clamp guards rounding.
            branch_length[active[side]] = (new_height - height[side]).max(0.0);
        }
        let merged_size = size[bi] + size[bj];
        let updated: Vec<f64> = (0..active.len())
            .filter(|k| *k != bi && *k != bj)
            .map(|k| (size[bi] * d[bi][k] + size[bj] * d[bj][k]) / merged_size)
            .collect();
        let keep: Vec<usize> = (0..active.len()).filter(|k| *k != bi && *k != bj).collect();
        active = keep.iter().map(|k| active[*k]).collect();
        size = keep.iter().map(|k| size[*k]).collect();
        height = keep.iter().map(|k| height[*k]).collect();
        let mut shrunk: Vec<Vec<f64>> =
            keep.iter().map(|a| keep.iter().map(|b| d[*a][*b]).collect()).collect();
        for (row, value) in shrunk.iter_mut().zip(updated.iter()) {
            row.push(*value);
        }
        let mut last = updated;
        last.push(0.0);
        shrunk.push(last);
        d = shrunk;
        active.push(node);
        size.push(merged_size);
        height.push(new_height);
    }
    PhyloTree::new(parent, branch_length, names)
}

/// Saitou and Nei's neighbour joining.
///
/// Joins the pair minimising `Q(i,j) = (n-2) d(i,j) - r_i - r_j`, where
/// `r_i` is the row sum, rather than the pair that is simply closest. The
/// correction is what makes the method consistent without a clock: two
/// taxa can be close together merely because both evolve slowly, and `Q`
/// discounts exactly that. Given an additive matrix the method recovers
/// the true tree exactly.
///
/// The result is an **unrooted** tree returned in rooted form: the final
/// node has three children and is a placeholder, not an inferred ancestor.
/// Do not read [`PhyloTree::height`] or [`PhyloTree::depth`] off it as
/// times, and expect [`PhyloTree::is_binary`] to be false at that node.
///
/// Non-additive data can imply a negative branch. Since a negative length
/// has no meaning as a number of substitutions, it is clamped to zero --
/// the standard remedy, and a sign that the data do not fit a tree.
///
/// # Errors
/// Returns an error for a malformed matrix (see [`upgma`]) or fewer than
/// three taxa.
pub fn neighbor_joining(dist: &Matrix, labels: &[String]) -> Result<PhyloTree, GeomError> {
    let n = check_distances(dist, labels, 3)?;
    let total_nodes = 2 * n - 2;
    let mut parent: Vec<Option<usize>> = vec![None; total_nodes];
    let mut branch_length = vec![0.0; total_nodes];
    let mut names: Vec<String> = labels.to_vec();
    names.resize(total_nodes, String::new());

    let mut active: Vec<usize> = (0..n).collect();
    let mut d: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| dist.get(i, j)).collect()).collect();
    let mut next_node = n;

    while active.len() > 3 {
        let m = active.len();
        let row: Vec<f64> = (0..m).map(|i| (0..m).map(|j| d[i][j]).sum()).collect();
        let (mut bi, mut bj, mut best) = (0usize, 1usize, f64::INFINITY);
        for i in 0..m {
            for j in (i + 1)..m {
                let q = (m as f64 - 2.0) * d[i][j] - row[i] - row[j];
                if q < best {
                    best = q;
                    bi = i;
                    bj = j;
                }
            }
        }
        let node = next_node;
        next_node += 1;
        let to_i = 0.5 * d[bi][bj] + (row[bi] - row[bj]) / (2.0 * (m as f64 - 2.0));
        let to_j = d[bi][bj] - to_i;
        parent[active[bi]] = Some(node);
        parent[active[bj]] = Some(node);
        branch_length[active[bi]] = to_i.max(0.0);
        branch_length[active[bj]] = to_j.max(0.0);

        let keep: Vec<usize> = (0..m).filter(|k| *k != bi && *k != bj).collect();
        let updated: Vec<f64> =
            keep.iter().map(|k| (0.5 * (d[bi][*k] + d[bj][*k] - d[bi][bj])).max(0.0)).collect();
        active = keep.iter().map(|k| active[*k]).collect();
        let mut shrunk: Vec<Vec<f64>> =
            keep.iter().map(|a| keep.iter().map(|b| d[*a][*b]).collect()).collect();
        for (r, value) in shrunk.iter_mut().zip(updated.iter()) {
            r.push(*value);
        }
        let mut last = updated;
        last.push(0.0);
        shrunk.push(last);
        d = shrunk;
        active.push(node);
    }

    // Three clusters remain. Their branches to the common node are the
    // unique lengths consistent with the three pairwise distances.
    let node = next_node;
    let (a, b, c) = (active[0], active[1], active[2]);
    let arms = [
        0.5 * (d[0][1] + d[0][2] - d[1][2]),
        0.5 * (d[0][1] + d[1][2] - d[0][2]),
        0.5 * (d[0][2] + d[1][2] - d[0][1]),
    ];
    for (child, arm) in [a, b, c].into_iter().zip(arms) {
        parent[child] = Some(node);
        branch_length[child] = arm.max(0.0);
    }
    PhyloTree::new(parent, branch_length, names)
}

/// The matrix of Jukes-Cantor corrected pairwise distances.
///
/// Sites where either sequence is not one of A, C, G, T are skipped for
/// that pair, so different pairs may rest on different numbers of sites.
///
/// # Errors
/// Returns an error for fewer than two sequences, sequences of differing
/// or zero length, a pair with no comparable site, or a pair whose observed
/// difference has saturated at three quarters, where the correction gives
/// no finite answer.
pub fn distance_matrix_jc69(seqs: &[Vec<u8>]) -> Result<Matrix, GeomError> {
    let n = seqs.len();
    if n < 2 {
        return Err(GeomError::InvalidArgument("a distance matrix needs at least two sequences"));
    }
    let width = seqs[0].len();
    if width == 0 || seqs.iter().any(|s| s.len() != width) {
        return Err(GeomError::InvalidArgument("the sequences must be aligned and non-empty"));
    }
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut compared = 0usize;
            let mut differing = 0usize;
            for site in 0..width {
                let (a, b) = (base_index(seqs[i][site]), base_index(seqs[j][site]));
                if let (Some(a), Some(b)) = (a, b) {
                    compared += 1;
                    if a != b {
                        differing += 1;
                    }
                }
            }
            if compared == 0 {
                return Err(GeomError::InvalidArgument("a pair of sequences shares no usable site"));
            }
            let p = differing as f64 / compared as f64;
            let d = crate::biophysics::seq_align::jukes_cantor_distance(p)?;
            out.set(i, j, d);
            out.set(j, i, d);
        }
    }
    Ok(out)
}

/// A, C, G or T as 0..4, case-insensitively; anything else is missing.
fn base_index(byte: u8) -> Option<usize> {
    match byte.to_ascii_uppercase() {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' | b'U' => Some(3),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Character methods
// ---------------------------------------------------------------------------

/// Nodes ordered children before parents.
fn postorder(tree: &PhyloTree) -> Vec<usize> {
    let mut level = vec![0usize; tree.len()];
    for node in 0..tree.len() {
        level[node] = tree.path_to_root(node).len();
    }
    let mut order: Vec<usize> = (0..tree.len()).collect();
    order.sort_by(|a, b| level[*b].cmp(&level[*a]));
    order
}

/// Fitch's parsimony score: the fewest character changes the tree needs.
///
/// `characters` holds one state per leaf, in the order [`PhyloTree::leaves`]
/// returns them. Working from the tips down, each node takes the
/// intersection of its children's state sets, or -- when that is empty --
/// their union at the cost of one change.
///
/// The score counts changes, not their positions: a site can be explained
/// by several equally parsimonious assignments, and parsimony picks none of
/// them. It is also biased when rates vary a lot between branches, where it
/// can be positively misled (long-branch attraction) into preferring the
/// wrong topology however much data you add.
///
/// # Errors
/// Returns an error if the character count differs from the leaf count or
/// more than 32 distinct states appear.
pub fn parsimony_fitch(tree: &PhyloTree, characters: &[u8]) -> Result<u64, GeomError> {
    let leaves = tree.leaves();
    if characters.len() != leaves.len() {
        return Err(GeomError::InvalidArgument("one character per leaf is required"));
    }
    let mut alphabet: Vec<u8> = characters.to_vec();
    alphabet.sort_unstable();
    alphabet.dedup();
    if alphabet.len() > 32 {
        return Err(GeomError::InvalidArgument("parsimony_fitch handles at most 32 states"));
    }
    let mut sets = vec![0u32; tree.len()];
    for (slot, state) in leaves.iter().zip(characters.iter()) {
        let bit = alphabet.iter().position(|s| s == state).expect("state is in the alphabet");
        sets[*slot] = 1u32 << bit;
    }
    let mut changes = 0u64;
    for node in postorder(tree) {
        let children = tree.children(node);
        if children.is_empty() {
            continue;
        }
        let shared = children.iter().fold(u32::MAX, |acc, c| acc & sets[*c]);
        if shared == 0 {
            sets[node] = children.iter().fold(0u32, |acc, c| acc | sets[*c]);
            changes += 1;
        } else {
            sets[node] = shared;
        }
    }
    Ok(changes)
}

/// The log-likelihood of an alignment on a tree under Jukes-Cantor, by
/// Felsenstein's pruning algorithm.
///
/// `seqs` holds one aligned sequence per leaf, in the order
/// [`PhyloTree::leaves`] returns them, and branch lengths are expected
/// substitutions per site. Under JC69 a branch of length `t` leaves a site
/// unchanged with probability `1/4 + 3/4 e^(-4t/3)` and sends it to each
/// other base with `1/4 - 1/4 e^(-4t/3)`; pruning sums over every ancestral
/// assignment in one pass up the tree rather than enumerating `4^nodes` of
/// them.
///
/// The result is a *log* likelihood because the likelihood itself
/// underflows: a thousand sites each contributing a factor near `0.25`
/// gives a number around `1e-600`, which is not representable.
///
/// Sites where a leaf carries an ambiguous or missing base contribute a
/// factor of one from that leaf -- the site still informs the others.
///
/// # Errors
/// Returns an error if the sequence count differs from the leaf count, the
/// sequences are empty or of differing length.
pub fn likelihood_jc69(tree: &PhyloTree, seqs: &[Vec<u8>]) -> Result<f64, GeomError> {
    let leaves = tree.leaves();
    if seqs.len() != leaves.len() {
        return Err(GeomError::InvalidArgument("one sequence per leaf is required"));
    }
    let width = seqs.first().map_or(0, Vec::len);
    if width == 0 || seqs.iter().any(|s| s.len() != width) {
        return Err(GeomError::InvalidArgument("the sequences must be aligned and non-empty"));
    }
    let order = postorder(tree);
    let root = tree.root();
    let mut total = 0.0;
    let mut partial = vec![[0.0f64; 4]; tree.len()];
    for site in 0..width {
        for (slot, seq) in leaves.iter().zip(seqs.iter()) {
            partial[*slot] = match base_index(seq[site]) {
                Some(base) => {
                    let mut row = [0.0; 4];
                    row[base] = 1.0;
                    row
                }
                None => [1.0; 4],
            };
        }
        for node in &order {
            let children = tree.children(*node);
            if children.is_empty() {
                continue;
            }
            let mut row = [1.0f64; 4];
            for child in children {
                let t = tree.branch_length[child];
                let decay = (-4.0 * t / 3.0).exp();
                let same = 0.25 + 0.75 * decay;
                let other = 0.25 - 0.25 * decay;
                let sum: f64 = partial[child].iter().sum();
                for (from, slot) in row.iter_mut().enumerate() {
                    // sum over the child's states: `same` for a match and
                    // `other` for the three alternatives.
                    *slot *= other * (sum - partial[child][from]) + same * partial[child][from];
                }
            }
            partial[*node] = row;
        }
        let site_likelihood: f64 = partial[root].iter().map(|p| 0.25 * p).sum();
        if !(site_likelihood > 0.0) {
            return Err(GeomError::Degenerate("a site has zero likelihood on this tree"));
        }
        total += site_likelihood.ln();
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/// Bootstrap support for the splits of a distance tree.
///
/// Builds a reference tree from the whole alignment, then resamples the
/// *columns* with replacement `replicates` times, rebuilds, and reports the
/// fraction of replicates recovering each branch of the reference. The
/// returned vector is aligned with `reference.bipartitions()`.
///
/// Branches are compared as unrooted bipartitions rather than rooted
/// clades. Neighbour joining's root is an artefact, so two replicates that
/// found the same tree can report a clade and its complement; treating
/// those as different answers would understate support for no reason.
///
/// Columns are the sampling unit because sites are what the model treats as
/// independent draws; resampling taxa instead would answer a different
/// question. High support means the signal is spread across the alignment
/// rather than resting on a handful of sites -- it is not a probability
/// that the split is true, and a consistently wrong method will support a
/// wrong split at 100%.
///
/// Replicates whose resampled alignment yields no usable distance matrix
/// (a saturated pair, say) are skipped, and the divisor counts only those
/// that succeeded.
///
/// # Errors
/// Returns an error for fewer than three sequences, unaligned or empty
/// sequences, a label count that disagrees, zero replicates, or a whole
/// alignment that yields no tree.
pub fn bootstrap_trees(
    seqs: &[Vec<u8>],
    labels: &[String],
    replicates: usize,
    method: DistanceMethod,
    rng: &mut Rng,
) -> Result<(PhyloTree, Vec<f64>), GeomError> {
    if seqs.len() < 3 || labels.len() != seqs.len() {
        return Err(GeomError::InvalidArgument("bootstrap_trees needs at least three taxa"));
    }
    let width = seqs[0].len();
    if width == 0 || seqs.iter().any(|s| s.len() != width) {
        return Err(GeomError::InvalidArgument("the sequences must be aligned and non-empty"));
    }
    if replicates == 0 {
        return Err(GeomError::InvalidArgument("at least one replicate is required"));
    }
    let build = |data: &[Vec<u8>]| -> Result<PhyloTree, GeomError> {
        let d = distance_matrix_jc69(data)?;
        match method {
            DistanceMethod::Upgma => upgma(&d, labels),
            DistanceMethod::NeighborJoining => neighbor_joining(&d, labels),
        }
    };
    let reference = build(seqs)?;
    let target = reference.bipartitions();
    let mut hits = vec![0usize; target.len()];
    let mut succeeded = 0usize;
    for _ in 0..replicates {
        let columns: Vec<usize> =
            (0..width).map(|_| (rng.next_f64() * width as f64) as usize % width).collect();
        let resampled: Vec<Vec<u8>> =
            seqs.iter().map(|s| columns.iter().map(|c| s[*c]).collect()).collect();
        let Ok(tree) = build(&resampled) else { continue };
        succeeded += 1;
        let found = tree.bipartitions();
        for (slot, split) in hits.iter_mut().zip(target.iter()) {
            if found.contains(split) {
                *slot += 1;
            }
        }
    }
    if succeeded == 0 {
        return Err(GeomError::Degenerate("no bootstrap replicate produced a tree"));
    }
    let support = hits.iter().map(|h| *h as f64 / succeeded as f64).collect();
    Ok((reference, support))
}

// ---------------------------------------------------------------------------
// Simulation and tree shape
// ---------------------------------------------------------------------------

/// An exponential waiting time at the given rate.
fn exponential(rate: f64, rng: &mut Rng) -> f64 {
    -(1.0 - rng.next_f64()).ln() / rate
}

/// A birth-death tree, pruned to the lineages that survive.
///
/// Runs the forward process -- each lineage speciating at rate `lambda` and
/// dying at rate `mu` -- until `n_leaves` lineages are alive at once, then
/// removes the extinct ones and suppresses the resulting single-child
/// nodes. What comes back is the *reconstructed* tree, the only one a
/// phylogeny of living species could ever show.
///
/// That pruning is why extinction leaves a signature rather than
/// disappearing. Near the present, lineages have not yet had time to die,
/// so the reconstructed tree grows at the full rate `lambda` there while
/// deeper down it grows at `lambda - mu`. The surviving tree therefore
/// looks as though speciation accelerated toward the present -- the "pull
/// of the present", which shows up as a positive [`gamma_statistic`] and an
/// upturn in the [`lineage_through_time`] curve.
///
/// The tree is stopped at the first event *after* the target count is
/// reached, so the interval during which `n_leaves` lineages coexist has a
/// length rather than collapsing to zero.
///
/// The tree is ultrametric by construction -- every tip sits at the same
/// stopping time.
///
/// # Errors
/// Returns an error for a non-positive `lambda`, a negative or non-finite
/// `mu`, `mu >= lambda`, fewer than three leaves, or if every attempt died
/// out before reaching the target.
pub fn birth_death_tree(
    lambda: f64,
    mu: f64,
    n_leaves: usize,
    rng: &mut Rng,
) -> Result<PhyloTree, GeomError> {
    if !(lambda > 0.0) || mu < 0.0 || !mu.is_finite() {
        return Err(GeomError::InvalidArgument("birth_death_tree: bad rates"));
    }
    if mu >= lambda {
        return Err(GeomError::InvalidArgument(
            "a critical or subcritical process rarely reaches a target size",
        ));
    }
    if !(3..=10_000).contains(&n_leaves) {
        return Err(GeomError::InvalidArgument("birth_death_tree: bad leaf count"));
    }
    for _ in 0..64 {
        if let Some(tree) = attempt_birth_death(lambda, mu, n_leaves, rng) {
            return Ok(tree);
        }
    }
    Err(GeomError::Degenerate("every birth-death attempt went extinct"))
}

/// One forward run, or `None` if the process died out.
fn attempt_birth_death(
    lambda: f64,
    mu: f64,
    n_leaves: usize,
    rng: &mut Rng,
) -> Option<PhyloTree> {
    let mut parent: Vec<Option<usize>> = vec![None];
    let mut born = vec![0.0f64];
    let mut ended: Vec<Option<f64>> = vec![None];
    let mut alive = vec![0usize];
    let mut now = 0.0;
    while alive.len() < n_leaves {
        let rate = (lambda + mu) * alive.len() as f64;
        now += exponential(rate, rng);
        let victim = (rng.next_f64() * alive.len() as f64) as usize % alive.len();
        let node = alive[victim];
        ended[node] = Some(now);
        if rng.next_f64() < lambda / (lambda + mu) {
            alive.swap_remove(victim);
            for _ in 0..2 {
                parent.push(Some(node));
                born.push(now);
                ended.push(None);
                alive.push(parent.len() - 1);
            }
        } else {
            alive.swap_remove(victim);
            if alive.is_empty() {
                return None;
            }
        }
    }
    // Stop at the next event time rather than at the instant the target
    // was reached. Stopping on the branching itself would leave the tree's
    // last internode interval exactly zero, which biases every shape
    // statistic read off it -- the gamma statistic by about +sqrt(3/n).
    let stop = now + exponential((lambda + mu) * n_leaves as f64, rng);
    let extant: Vec<bool> = (0..parent.len()).map(|k| ended[k].is_none()).collect();
    let length: Vec<f64> =
        (0..parent.len()).map(|k| ended[k].unwrap_or(stop) - born[k]).collect();
    Some(prune_extinct(&parent, &length, &extant))
}

/// Drops extinct tips and suppresses the single-child nodes left behind.
fn prune_extinct(parent: &[Option<usize>], length: &[f64], extant: &[bool]) -> PhyloTree {
    let n = parent.len();
    // A node survives if it is extant or has a surviving descendant. Nodes
    // are created after their parents, so one reverse pass suffices.
    let mut keep = extant.to_vec();
    for node in (0..n).rev() {
        if keep[node] {
            if let Some(p) = parent[node] {
                keep[p] = true;
            }
        }
    }
    // Walk each kept node up to its nearest kept ancestor with more than
    // one kept child, accumulating the branch lengths passed through.
    let kept_children: Vec<usize> = (0..n)
        .map(|node| (0..n).filter(|k| keep[*k] && parent[*k] == Some(node)).count())
        .collect();
    let significant = |node: usize| keep[node] && (kept_children[node] != 1);
    let survivors: Vec<usize> = (0..n).filter(|k| significant(*k)).collect();
    let mut slot = vec![usize::MAX; n];
    for (new, old) in survivors.iter().enumerate() {
        slot[*old] = new;
    }
    let mut new_parent = vec![None; survivors.len()];
    let mut new_length = vec![0.0; survivors.len()];
    let labels = vec![String::new(); survivors.len()];
    for (new, old) in survivors.iter().enumerate() {
        let mut walked = length[*old];
        let mut current = parent[*old];
        while let Some(node) = current {
            if significant(node) {
                new_parent[new] = Some(slot[node]);
                break;
            }
            walked += length[node];
            current = parent[node];
        }
        if new_parent[new].is_some() {
            new_length[new] = walked;
        }
    }
    let tree = PhyloTree { parent: new_parent, branch_length: new_length, labels };
    let ordered = reorder_leaves_first(&tree);
    let leaves = ordered.leaves();
    let mut named = ordered;
    for (position, leaf) in leaves.iter().enumerate() {
        named.labels[*leaf] = format!("t{position}");
    }
    named
}

/// The waiting times between successive branching events, `g[k]` being the
/// interval during which the tree had `k + 2` lineages.
fn internode_intervals(tree: &PhyloTree) -> Result<Vec<f64>, GeomError> {
    let leaves = tree.leaves();
    let n = leaves.len();
    if n < 3 {
        return Err(GeomError::InvalidArgument("tree shape statistics need at least three tips"));
    }
    let mut events: Vec<f64> = (0..tree.len())
        .filter(|k| !tree.children(*k).is_empty())
        .flat_map(|k| {
            // A polytomy of c children is c - 1 simultaneous branchings.
            let extra = tree.children(k).len() - 1;
            std::iter::repeat_n(tree.depth(k), extra)
        })
        .collect();
    if events.len() + 1 != n {
        return Err(GeomError::InvalidArgument("the tree's branchings do not match its tips"));
    }
    events.sort_by(|a, b| a.partial_cmp(b).expect("finite depths"));
    let height = tree.height();
    let mut g = Vec::with_capacity(n - 1);
    for k in 1..events.len() {
        g.push(events[k] - events[k - 1]);
    }
    g.push(height - events[events.len() - 1]);
    Ok(g)
}

/// Pybus and Harvey's gamma statistic.
///
/// Standard normal under a constant-rate pure-birth process, so it is a
/// direct test of that null: negative gamma means the internal branching
/// events sit closer to the root than a constant rate predicts -- an early
/// burst, or a diversification rate that slowed -- and positive gamma means
/// they crowd toward the present.
///
/// Extinction pushes gamma *positive* on a reconstructed tree even at a
/// constant rate: recent lineages have not yet had time to die, so nodes
/// crowd toward the present. A positive value is therefore not by itself
/// evidence of an accelerating rate. The bias runs the other way from the
/// slowdown test, which is why a significantly negative gamma is taken as
/// conservative evidence of a slowdown.
///
/// The statistic reads times off the tree, so it is meaningful only for an
/// ultrametric one; a tree with unequal tip depths is rejected rather than
/// silently misread.
///
/// # Errors
/// Returns an error for fewer than three tips, a tree that is not
/// ultrametric to `1e-8` relative, or one of zero height.
pub fn gamma_statistic(tree: &PhyloTree) -> Result<f64, GeomError> {
    let height = tree.height();
    if !(height > 0.0) {
        return Err(GeomError::Degenerate("the tree has no height"));
    }
    if !tree.is_ultrametric(1e-8 * height) {
        return Err(GeomError::InvalidArgument("the gamma statistic needs an ultrametric tree"));
    }
    let g = internode_intervals(tree)?;
    let n = g.len() + 1;
    // T = sum over k of k * g[k], with k running from 2 to n.
    let weighted: Vec<f64> = g.iter().enumerate().map(|(i, v)| (i as f64 + 2.0) * v).collect();
    let total: f64 = weighted.iter().sum();
    if !(total > 0.0) {
        return Err(GeomError::Degenerate("the tree has no length"));
    }
    let mut running = 0.0;
    let mut inner = 0.0;
    for value in weighted.iter().take(n - 2) {
        running += value;
        inner += running;
    }
    let mean = inner / (n as f64 - 2.0);
    Ok((mean - total / 2.0) / (total * (1.0 / (12.0 * (n as f64 - 2.0))).sqrt()))
}

/// The lineage-through-time curve: `(time, lineage count)` at the root, at
/// every branching, and at the present.
///
/// Time is measured from the root. Plotted with a log count axis, a
/// constant-rate pure-birth tree gives a straight line of slope `lambda`,
/// which is what makes the curve's departures readable: a bend downward
/// toward the tips is a slowdown, and the upturn near the present on a tree
/// with extinction is the pull of the present rather than a real burst.
///
/// For a tree whose tips are not all at the same depth, the final point
/// uses the deepest tip and the count there is the leaf total.
///
/// # Errors
/// Returns an error for a tree with fewer than two tips.
pub fn lineage_through_time(tree: &PhyloTree) -> Result<Vec<(f64, usize)>, GeomError> {
    let leaves = tree.leaves();
    if leaves.len() < 2 {
        return Err(GeomError::InvalidArgument("a lineage curve needs at least two tips"));
    }
    let mut events: Vec<(f64, usize)> = (0..tree.len())
        .filter(|k| !tree.children(*k).is_empty())
        .map(|k| (tree.depth(k), tree.children(k).len() - 1))
        .collect();
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite depths"));
    let mut out = vec![(0.0, 1usize)];
    let mut count = 1usize;
    for (time, added) in events {
        count += added;
        out.push((time, count));
    }
    out.push((tree.height(), count));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The patristic distance matrix of a tree, with its leaf labels.
    fn patristic(tree: &PhyloTree) -> (Matrix, Vec<String>) {
        let leaves = tree.leaves();
        let mut out = Matrix::zeros(leaves.len(), leaves.len());
        for (i, a) in leaves.iter().enumerate() {
            for (j, b) in leaves.iter().enumerate() {
                out.set(i, j, tree.distance(*a, *b).unwrap());
            }
        }
        (out, leaves.iter().map(|k| tree.labels[*k].clone()).collect())
    }

    /// The labels below each internal node, as a sorted set of sets.
    fn split_set(tree: &PhyloTree) -> Vec<Vec<String>> {
        tree.splits()
    }

    fn labelled(tree: &PhyloTree, name: &str) -> usize {
        tree.labels.iter().position(|l| l == name).unwrap_or_else(|| panic!("no leaf {name}"))
    }

    #[test]
    fn a_newick_string_survives_a_round_trip_through_the_tree_and_back() {
        let text = "((A:1.5,B:0.5):2,(C:1,D:1):2.5);";
        let tree = PhyloTree::from_newick(text).unwrap();
        let again = PhyloTree::from_newick(&tree.to_newick()).unwrap();
        assert_eq!(split_set(&tree), split_set(&again));
        assert!((tree.total_length() - again.total_length()).abs() < 1e-12);
        for name in ["A", "B", "C", "D"] {
            let here = labelled(&tree, name);
            let there = labelled(&again, name);
            assert!((tree.depth(here) - again.depth(there)).abs() < 1e-12);
        }
    }

    #[test]
    fn the_distances_on_a_hand_written_tree_are_the_ones_the_string_says() {
        // ((A:1,B:1):2,C:3): A and B meet one unit up, and each reaches the
        // root through two more.
        let tree = PhyloTree::from_newick("((A:1,B:1):2,C:3);").unwrap();
        let (a, b, c) = (labelled(&tree, "A"), labelled(&tree, "B"), labelled(&tree, "C"));
        assert!((tree.distance(a, b).unwrap() - 2.0).abs() < 1e-12);
        assert!((tree.distance(a, c).unwrap() - 6.0).abs() < 1e-12);
        assert!((tree.distance(b, c).unwrap() - 6.0).abs() < 1e-12);
        assert!((tree.height() - 3.0).abs() < 1e-12);
        assert!((tree.total_length() - 7.0).abs() < 1e-12);
        // A and B's ancestor is the internal node, not the root.
        assert_ne!(tree.mrca(a, b).unwrap(), tree.root());
        assert_eq!(tree.mrca(a, c).unwrap(), tree.root());
        assert!(tree.is_binary());
    }

    #[test]
    fn a_malformed_newick_string_is_refused_rather_than_guessed_at() {
        for bad in ["", "(A:1,B:1)", "((A:1,B:1);", "(A:1,B:x);", "(A:1,B:-2);", "A:1,B:1;"] {
            assert!(PhyloTree::from_newick(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn robinson_foulds_is_zero_against_itself_and_positive_once_two_leaves_swap() {
        let one = PhyloTree::from_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let two = PhyloTree::from_newick("((A:1,C:1):1,(B:1,D:1):1);").unwrap();
        assert_eq!(one.robinson_foulds(&one).unwrap(), 0);
        assert_eq!(two.robinson_foulds(&two).unwrap(), 0);
        assert!(one.robinson_foulds(&two).unwrap() > 0);
        assert_eq!(one.robinson_foulds(&two).unwrap(), two.robinson_foulds(&one).unwrap());
        // Branch lengths do not enter: stretching every branch changes no
        // split, so the distance stays zero.
        let stretched = PhyloTree::from_newick("((A:9,B:0.1):4,(C:2,D:7):3);").unwrap();
        assert_eq!(one.robinson_foulds(&stretched).unwrap(), 0);
    }

    #[test]
    fn robinson_foulds_refuses_trees_that_do_not_describe_the_same_leaves() {
        let one = PhyloTree::from_newick("((A:1,B:1):1,C:1);").unwrap();
        let two = PhyloTree::from_newick("((A:1,B:1):1,D:1);").unwrap();
        assert!(one.robinson_foulds(&two).is_err());
    }

    #[test]
    fn upgma_reconstructs_an_ultrametric_matrix_exactly() {
        // Every tip one unit from the root, so the clock UPGMA assumes is
        // the clock the data were built under.
        let truth = PhyloTree::from_newick("(((A:0.2,B:0.2):0.3,C:0.5):0.5,(D:0.4,E:0.4):0.6);")
            .unwrap();
        assert!(truth.is_ultrametric(1e-12));
        let (dist, labels) = patristic(&truth);
        let built = upgma(&dist, &labels).unwrap();
        assert_eq!(built.robinson_foulds(&truth).unwrap(), 0);
        let (again, _) = patristic(&built);
        for i in 0..dist.rows {
            for j in 0..dist.cols {
                assert!(
                    (again.get(i, j) - dist.get(i, j)).abs() < 1e-12,
                    "distance {i},{j} came back as {} not {}",
                    again.get(i, j),
                    dist.get(i, j)
                );
            }
        }
    }

    #[test]
    fn upgma_returns_a_clocklike_tree_even_when_the_data_are_not_clocklike() {
        // The assumption is in the method, not the data: the output is
        // ultrametric whatever went in.
        let truth =
            PhyloTree::from_newick("((A:0.4,B:0.02):0.05,(C:0.4,D:0.02):0.05);").unwrap();
        assert!(!truth.is_ultrametric(1e-3));
        let (dist, labels) = patristic(&truth);
        let built = upgma(&dist, &labels).unwrap();
        assert!(built.is_ultrametric(1e-9));
    }

    #[test]
    fn unequal_rates_fool_upgma_where_neighbour_joining_holds() {
        // A and C evolve twenty times faster than their sisters. The two
        // slow tips B and D are then the closest pair in the matrix even
        // though they are not relatives, and average linkage joins them.
        let truth =
            PhyloTree::from_newick("((A:0.4,B:0.02):0.05,(C:0.4,D:0.02):0.05);").unwrap();
        let (dist, labels) = patristic(&truth);

        let clustered = upgma(&dist, &labels).unwrap();
        let wrong: Vec<String> = vec!["B".into(), "D".into()];
        assert!(
            clustered.splits().contains(&wrong),
            "UPGMA was expected to group the two slow tips, gave {:?}",
            clustered.splits()
        );

        let joined = neighbor_joining(&dist, &labels).unwrap();
        let right: Vec<String> = vec!["A".into(), "B".into()];
        assert!(
            joined.splits().contains(&right),
            "neighbour joining lost the true clade, gave {:?}",
            joined.splits()
        );
    }

    #[test]
    fn neighbour_joining_reproduces_an_additive_matrix_to_rounding() {
        // The theorem: given distances that came from a tree, neighbour
        // joining returns that tree's distances exactly.
        let truth = PhyloTree::from_newick(
            "(((A:0.1,B:0.7):0.2,C:0.05):0.3,((D:0.4,E:0.02):0.15,F:0.6):0.25);",
        )
        .unwrap();
        let (dist, labels) = patristic(&truth);
        let built = neighbor_joining(&dist, &labels).unwrap();
        let (again, order) = patristic(&built);
        assert_eq!(order, labels);
        for i in 0..dist.rows {
            for j in 0..dist.cols {
                assert!(
                    (again.get(i, j) - dist.get(i, j)).abs() < 1e-9,
                    "distance {i},{j} came back as {} not {}",
                    again.get(i, j),
                    dist.get(i, j)
                );
            }
        }
    }

    #[test]
    fn the_neighbour_joining_root_is_a_trifurcation_and_not_an_ancestor() {
        let truth = PhyloTree::from_newick("(((A:0.1,B:0.7):0.2,C:0.05):0.3,D:0.6);").unwrap();
        let (dist, labels) = patristic(&truth);
        let built = neighbor_joining(&dist, &labels).unwrap();
        assert_eq!(built.children(built.root()).len(), 3);
        assert!(!built.is_binary(), "an unrooted tree should not claim to be resolved");
    }

    #[test]
    fn the_distance_methods_refuse_a_matrix_that_is_not_one() {
        let labels: Vec<String> = ["A", "B", "C"].iter().map(|s| (*s).to_string()).collect();
        let mut good = Matrix::zeros(3, 3);
        for (i, j, d) in [(0, 1, 0.3), (0, 2, 0.5), (1, 2, 0.4)] {
            good.set(i, j, d);
            good.set(j, i, d);
        }
        assert!(upgma(&good, &labels).is_ok());
        assert!(neighbor_joining(&good, &labels).is_ok());

        let mut asymmetric = good.clone();
        asymmetric.set(0, 1, 0.9);
        assert!(upgma(&asymmetric, &labels).is_err());

        let mut negative = good.clone();
        negative.set(0, 1, -0.1);
        negative.set(1, 0, -0.1);
        assert!(neighbor_joining(&negative, &labels).is_err());

        let mut self_distance = good.clone();
        self_distance.set(2, 2, 0.1);
        assert!(upgma(&self_distance, &labels).is_err());

        assert!(upgma(&good, &labels[..2]).is_err());
        // Neighbour joining needs three taxa; two carry no topology.
        let two: Vec<String> = labels[..2].to_vec();
        let mut pair = Matrix::zeros(2, 2);
        pair.set(0, 1, 0.3);
        pair.set(1, 0, 0.3);
        assert!(neighbor_joining(&pair, &two).is_err());
        assert!(upgma(&pair, &two).is_ok());
    }

    #[test]
    fn the_jukes_cantor_matrix_corrects_upward_from_the_raw_difference() {
        let seqs: Vec<Vec<u8>> = vec![
            b"ACGTACGTACGTACGTACGT".to_vec(),
            b"ACGTACGTACGTACGTAAAA".to_vec(),
            b"TGCATGCATGCTACGTACGT".to_vec(),
        ];
        let d = distance_matrix_jc69(&seqs).unwrap();
        for i in 0..3 {
            assert!((d.get(i, i)).abs() < 1e-15);
            for j in 0..3 {
                assert!((d.get(i, j) - d.get(j, i)).abs() < 1e-15);
            }
        }
        // Sequence 0 differs from 1 at three of twenty sites and from 2 at
        // eleven; the correction inflates both, and more so the larger one.
        let raw01 = crate::biophysics::seq_align::p_distance(&seqs[0], &seqs[1]).unwrap();
        let raw02 = crate::biophysics::seq_align::p_distance(&seqs[0], &seqs[2]).unwrap();
        assert!(d.get(0, 1) > raw01);
        assert!(d.get(0, 2) > raw02);
        assert!(d.get(0, 2) - raw02 > d.get(0, 1) - raw01);
    }

    #[test]
    fn a_saturated_pair_is_reported_rather_than_given_an_infinite_distance() {
        let seqs: Vec<Vec<u8>> =
            vec![b"AAAAAAAA".to_vec(), b"TTTTTTTT".to_vec(), b"AAAAAAAA".to_vec()];
        assert!(distance_matrix_jc69(&seqs).is_err());
    }

    #[test]
    fn a_character_that_never_varies_costs_nothing_and_one_that_alternates_costs_the_most() {
        let tree = PhyloTree::from_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let order: Vec<String> = tree.leaves().iter().map(|k| tree.labels[*k].clone()).collect();
        let at = |name: &str| order.iter().position(|l| l == name).unwrap();

        let constant = vec![b'A'; 4];
        assert_eq!(parsimony_fitch(&tree, &constant).unwrap(), 0);

        // A character shared by one true clade needs a single change on the
        // branch leading to it.
        let mut clade = vec![b'A'; 4];
        clade[at("C")] = b'G';
        clade[at("D")] = b'G';
        assert_eq!(parsimony_fitch(&tree, &clade).unwrap(), 1);

        // The same two states arranged across the clades cannot be
        // explained by one change on this topology.
        let mut crossing = vec![b'A'; 4];
        crossing[at("B")] = b'G';
        crossing[at("C")] = b'G';
        assert_eq!(parsimony_fitch(&tree, &crossing).unwrap(), 2);

        // Four distinct states need three changes however they are placed.
        assert_eq!(parsimony_fitch(&tree, b"ACGT").unwrap(), 3);
    }

    #[test]
    fn parsimony_never_costs_less_than_the_number_of_extra_states() {
        // The floor is a theorem: k states need at least k - 1 changes on
        // any tree, since each change introduces at most one new state.
        let tree = PhyloTree::from_newick("(((A:1,B:1):1,C:1):1,(D:1,(E:1,F:1):1):1);").unwrap();
        let mut rng = Rng::new(0x0B10_2001);
        let alphabet = b"ACGT";
        for _ in 0..200 {
            let characters: Vec<u8> = (0..6)
                .map(|_| alphabet[(rng.next_f64() * 4.0) as usize % 4])
                .collect();
            let mut distinct = characters.clone();
            distinct.sort_unstable();
            distinct.dedup();
            let score = parsimony_fitch(&tree, &characters).unwrap();
            assert!(
                score >= distinct.len() as u64 - 1,
                "{characters:?} scored {score} with {} states",
                distinct.len()
            );
            // And never more than one change per branch above a leaf.
            assert!(score <= 6);
        }
    }

    #[test]
    fn parsimony_prefers_the_topology_the_characters_were_built_on() {
        // Characters generated to agree with ((A,B),(C,D)) score lower on
        // that tree than on either of the two alternatives. This is the
        // whole method in one assertion.
        let truth = PhyloTree::from_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let alt1 = PhyloTree::from_newick("((A:1,C:1):1,(B:1,D:1):1);").unwrap();
        let alt2 = PhyloTree::from_newick("((A:1,D:1):1,(B:1,C:1):1);").unwrap();
        let names = ["A", "B", "C", "D"];
        let order = |tree: &PhyloTree| -> Vec<usize> {
            let leaves = tree.leaves();
            names
                .iter()
                .map(|n| leaves.iter().position(|k| tree.labels[*k] == **n).unwrap())
                .collect()
        };
        let (o0, o1, o2) = (order(&truth), order(&alt1), order(&alt2));
        let permute = |c: &[u8; 4], o: &[usize]| -> Vec<u8> {
            let mut out = vec![0u8; 4];
            for (name, slot) in o.iter().enumerate() {
                out[*slot] = c[name];
            }
            out
        };
        // Each character splits AB from CD.
        let characters: [[u8; 4]; 3] = [*b"AAGG", *b"CCTT", *b"GGAA"];
        let score = |tree: &PhyloTree, o: &[usize]| -> u64 {
            characters.iter().map(|c| parsimony_fitch(tree, &permute(c, o)).unwrap()).sum()
        };
        assert_eq!(score(&truth, &o0), 3);
        assert_eq!(score(&alt1, &o1), 6);
        assert_eq!(score(&alt2, &o2), 6);
    }

    #[test]
    fn parsimony_refuses_a_character_vector_that_does_not_match_the_leaves() {
        let tree = PhyloTree::from_newick("((A:1,B:1):1,C:1);").unwrap();
        assert!(parsimony_fitch(&tree, b"AC").is_err());
        assert!(parsimony_fitch(&tree, b"ACGT").is_err());
        assert!(parsimony_fitch(&tree, b"ACG").is_ok());
    }

    /// The likelihood of one site by enumerating every ancestral state.
    fn brute_force_site(tree: &PhyloTree, bases: &[usize]) -> f64 {
        let leaves = tree.leaves();
        let internal: Vec<usize> =
            (0..tree.len()).filter(|k| !leaves.contains(k)).collect();
        let mut state = vec![0usize; tree.len()];
        for (leaf, base) in leaves.iter().zip(bases.iter()) {
            state[*leaf] = *base;
        }
        let mut total = 0.0;
        for code in 0..4usize.pow(internal.len() as u32) {
            let mut rest = code;
            for node in &internal {
                state[*node] = rest % 4;
                rest /= 4;
            }
            let mut product = 0.25;
            for node in 0..tree.len() {
                let Some(parent) = tree.parent[node] else { continue };
                let t = tree.branch_length[node];
                let decay = (-4.0 * t / 3.0f64).exp();
                product *= if state[node] == state[parent] {
                    0.25 + 0.75 * decay
                } else {
                    0.25 - 0.25 * decay
                };
            }
            total += product;
        }
        total
    }

    #[test]
    fn pruning_agrees_with_enumerating_every_ancestral_state() {
        // Felsenstein's algorithm is a rearrangement of the sum over
        // 4^internal assignments, so the two must agree to rounding.
        let tree = PhyloTree::from_newick("(((A:0.1,B:0.3):0.2,C:0.05):0.15,(D:0.4,E:0.2):0.1);")
            .unwrap();
        let mut rng = Rng::new(0x0B10_2002);
        for _ in 0..20 {
            let bases: Vec<usize> = (0..5).map(|_| (rng.next_f64() * 4.0) as usize % 4).collect();
            let letters: Vec<Vec<u8>> =
                bases.iter().map(|b| vec![b"ACGT"[*b]]).collect();
            let pruned = likelihood_jc69(&tree, &letters).unwrap().exp();
            let direct = brute_force_site(&tree, &bases);
            assert!(
                (pruned - direct).abs() < 1e-12 * direct,
                "pruning gave {pruned} where enumeration gives {direct}"
            );
        }
    }

    #[test]
    fn the_maximum_likelihood_branch_length_of_a_pair_is_the_jukes_cantor_distance() {
        // For two sequences the closed-form estimate and the likelihood
        // peak are the same number, which is what makes the correction a
        // maximum-likelihood one rather than a rule of thumb.
        let mut rng = Rng::new(0x0B10_2003);
        let width = 400;
        let a: Vec<u8> = (0..width).map(|_| b"ACGT"[(rng.next_f64() * 4.0) as usize % 4]).collect();
        let mut b = a.clone();
        for slot in b.iter_mut() {
            if rng.next_f64() < 0.2 {
                *slot = b"ACGT"[(rng.next_f64() * 4.0) as usize % 4];
            }
        }
        let p = crate::biophysics::seq_align::p_distance(&a, &b).unwrap();
        let closed = crate::biophysics::seq_align::jukes_cantor_distance(p).unwrap();

        let at = |t: f64| {
            let tree = PhyloTree::from_newick(&format!("(A:{t},B:0.0);")).unwrap();
            let order = tree.leaves();
            let seqs: Vec<Vec<u8>> = order
                .iter()
                .map(|k| if tree.labels[*k] == "A" { a.clone() } else { b.clone() })
                .collect();
            likelihood_jc69(&tree, &seqs).unwrap()
        };
        // The scan starts above zero: a tree of no length cannot explain
        // sequences that differ, and reports that rather than a likelihood.
        let mut best = (f64::NEG_INFINITY, 0.0);
        let mut step = 0.0005;
        while step < 1.5 {
            let value = at(step);
            if value > best.0 {
                best = (value, step);
            }
            step += 0.0005;
        }
        assert!(
            (best.1 - closed).abs() < 2e-3,
            "the likelihood peaks at {} where the closed form gives {closed}",
            best.1
        );
    }

    #[test]
    fn very_long_branches_wash_the_tree_out_to_independent_uniform_tips() {
        // Once every branch is long the leaves are independent draws from
        // the equilibrium, so a site's likelihood tends to (1/4)^tips.
        let tree = PhyloTree::from_newick("((A:60,B:60):60,(C:60,D:60):60);").unwrap();
        let seqs: Vec<Vec<u8>> = vec![b"A".to_vec(), b"C".to_vec(), b"G".to_vec(), b"T".to_vec()];
        let value = likelihood_jc69(&tree, &seqs).unwrap().exp();
        assert!((value - 0.25f64.powi(4)).abs() < 1e-12, "washed out to {value}");
    }

    #[test]
    fn a_tree_of_zero_length_only_explains_identical_tips() {
        let tree = PhyloTree::from_newick("((A:0,B:0):0,C:0);").unwrap();
        let same: Vec<Vec<u8>> = vec![b"A".to_vec(); 3];
        assert!((likelihood_jc69(&tree, &same).unwrap() - 0.25f64.ln()).abs() < 1e-12);
        let differing: Vec<Vec<u8>> = vec![b"A".to_vec(), b"C".to_vec(), b"A".to_vec()];
        assert!(likelihood_jc69(&tree, &differing).is_err());
    }

    #[test]
    fn an_ambiguous_base_costs_nothing_while_the_rest_of_the_site_still_counts() {
        let tree = PhyloTree::from_newick("((A:0.2,B:0.2):0.1,C:0.3);").unwrap();
        let known: Vec<Vec<u8>> = vec![b"AC".to_vec(), b"AC".to_vec(), b"AC".to_vec()];
        let masked: Vec<Vec<u8>> = vec![b"AN".to_vec(), b"AC".to_vec(), b"AC".to_vec()];
        let dropped: Vec<Vec<u8>> = vec![b"A".to_vec(), b"A".to_vec(), b"A".to_vec()];
        let with_mask = likelihood_jc69(&tree, &masked).unwrap();
        // The masked leaf contributes a factor of one, so the second site
        // still carries information from the other two tips.
        assert!(with_mask > likelihood_jc69(&tree, &known).unwrap());
        assert!(with_mask < likelihood_jc69(&tree, &dropped).unwrap());
    }

    #[test]
    fn the_likelihood_refuses_an_alignment_that_does_not_fit_the_tree() {
        let tree = PhyloTree::from_newick("((A:0.2,B:0.2):0.1,C:0.3);").unwrap();
        assert!(likelihood_jc69(&tree, &[b"AC".to_vec(), b"AC".to_vec()]).is_err());
        assert!(likelihood_jc69(&tree, &[b"AC".to_vec(), b"A".to_vec(), b"AC".to_vec()]).is_err());
        assert!(likelihood_jc69(&tree, &[vec![], vec![], vec![]]).is_err());
    }
    /// A random DNA sequence.
    fn random_dna(width: usize, rng: &mut Rng) -> Vec<u8> {
        (0..width).map(|_| b"ACGT"[(rng.next_f64() * 4.0) as usize % 4]).collect()
    }

    /// A copy of `seq` with each site replaced at probability `rate`.
    fn mutate(seq: &[u8], rate: f64, rng: &mut Rng) -> Vec<u8> {
        seq.iter()
            .map(|base| {
                if rng.next_f64() < rate {
                    b"ACGT"[(rng.next_f64() * 4.0) as usize % 4]
                } else {
                    *base
                }
            })
            .collect()
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn a_split_the_whole_alignment_agrees_on_gets_full_bootstrap_support() {
        // Two tight pairs, far apart. Every column carries the same story,
        // so resampling columns cannot change the answer.
        let mut rng = Rng::new(0x0B10_3101);
        let root = random_dna(400, &mut rng);
        let left = mutate(&root, 0.25, &mut rng);
        let right = mutate(&root, 0.25, &mut rng);
        let seqs = vec![
            mutate(&left, 0.01, &mut rng),
            mutate(&left, 0.01, &mut rng),
            mutate(&right, 0.01, &mut rng),
            mutate(&right, 0.01, &mut rng),
        ];
        let labels = names(&["A", "B", "C", "D"]);
        let (tree, support) =
            bootstrap_trees(&seqs, &labels, 60, DistanceMethod::NeighborJoining, &mut rng).unwrap();
        assert_eq!(support.len(), tree.bipartitions().len());
        assert!(tree.bipartitions().contains(&names(&["A", "B"])), "lost the true clade");
        for (split, value) in tree.bipartitions().iter().zip(support.iter()) {
            assert!((0.0..=1.0).contains(value));
            assert!(*value > 0.95, "split {split:?} supported at only {value}");
        }
    }

    #[test]
    fn a_star_alignment_with_no_clade_structure_gets_weak_support() {
        // Four sequences equally diverged from one ancestor and from each
        // other. Whatever split the reference tree happens to pick, the
        // replicates should keep disagreeing with it.
        let mut rng = Rng::new(0x0B10_3102);
        let root = random_dna(300, &mut rng);
        let seqs: Vec<Vec<u8>> = (0..4).map(|_| mutate(&root, 0.10, &mut rng)).collect();
        let labels = names(&["A", "B", "C", "D"]);
        let (_, support) =
            bootstrap_trees(&seqs, &labels, 100, DistanceMethod::NeighborJoining, &mut rng).unwrap();
        assert!(!support.is_empty());
        for value in &support {
            assert!(*value < 0.9, "an arbitrary split was supported at {value}");
        }
    }

    #[test]
    fn bootstrap_support_stays_a_fraction_and_refuses_malformed_input() {
        let mut rng = Rng::new(0x0B10_3103);
        let root = random_dna(120, &mut rng);
        let seqs: Vec<Vec<u8>> = (0..4).map(|_| mutate(&root, 0.05, &mut rng)).collect();
        let labels = names(&["A", "B", "C", "D"]);
        for method in [DistanceMethod::Upgma, DistanceMethod::NeighborJoining] {
            let (_, support) = bootstrap_trees(&seqs, &labels, 20, method, &mut rng).unwrap();
            assert!(support.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        assert!(bootstrap_trees(&seqs, &labels, 0, DistanceMethod::Upgma, &mut rng).is_err());
        assert!(
            bootstrap_trees(&seqs, &names(&["A", "B", "C"]), 5, DistanceMethod::Upgma, &mut rng)
                .is_err()
        );
        assert!(
            bootstrap_trees(&seqs[..2], &names(&["A", "B"]), 5, DistanceMethod::Upgma, &mut rng)
                .is_err()
        );
        let ragged = vec![seqs[0].clone(), seqs[1][..50].to_vec(), seqs[2].clone(), seqs[3].clone()];
        assert!(bootstrap_trees(&ragged, &labels, 5, DistanceMethod::Upgma, &mut rng).is_err());
    }

    #[test]
    fn a_birth_death_tree_has_the_size_and_shape_it_was_asked_for() {
        let mut rng = Rng::new(0x0B10_3104);
        for (lambda, mu, tips) in [(1.0, 0.0, 12), (1.5, 0.7, 20), (2.0, 1.0, 8)] {
            for _ in 0..8 {
                let tree = birth_death_tree(lambda, mu, tips, &mut rng).unwrap();
                assert_eq!(tree.leaves().len(), tips);
                // Pruning the extinct lineages leaves no unbranched node,
                // so every internal node still has two children.
                assert!(tree.is_binary(), "{}", tree.to_newick());
                // Every tip sits at the stopping time.
                assert!(tree.is_ultrametric(1e-9 * tree.height()), "{}", tree.to_newick());
                assert!(tree.height() > 0.0);
                let mut labels: Vec<String> =
                    tree.leaves().iter().map(|k| tree.labels[*k].clone()).collect();
                labels.sort();
                labels.dedup();
                assert_eq!(labels.len(), tips, "tip labels are not distinct");
                // And it survives a round trip through Newick.
                let again = PhyloTree::from_newick(&tree.to_newick()).unwrap();
                assert_eq!(again.robinson_foulds(&tree).unwrap(), 0);
            }
        }
    }

    #[test]
    fn birth_death_refuses_rates_that_would_not_reach_the_target() {
        let mut rng = Rng::new(0x0B10_3105);
        assert!(birth_death_tree(0.0, 0.0, 10, &mut rng).is_err());
        assert!(birth_death_tree(-1.0, 0.0, 10, &mut rng).is_err());
        assert!(birth_death_tree(1.0, -0.1, 10, &mut rng).is_err());
        // A critical process reaches any target only by luck and a
        // supercritical death rate never does.
        assert!(birth_death_tree(1.0, 1.0, 10, &mut rng).is_err());
        assert!(birth_death_tree(1.0, 2.0, 10, &mut rng).is_err());
        assert!(birth_death_tree(1.0, 0.0, 2, &mut rng).is_err());
    }

    #[test]
    fn gamma_is_standard_normal_on_pure_birth_trees() {
        // The point of the statistic: under the constant-rate pure-birth
        // null it has mean zero and unit variance, so a value can be read
        // as a number of standard deviations without further calibration.
        let mut rng = Rng::new(0x0B10_3106);
        let replicates = 300;
        let values: Vec<f64> = (0..replicates)
            .map(|_| gamma_statistic(&birth_death_tree(1.0, 0.0, 30, &mut rng).unwrap()).unwrap())
            .collect();
        let mean: f64 = values.iter().sum::<f64>() / replicates as f64;
        let variance: f64 =
            values.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / replicates as f64;
        let standard_error = variance.sqrt() / (replicates as f64).sqrt();
        assert!(mean.abs() < 4.0 * standard_error, "gamma centred at {mean}, not zero");
        assert!((variance.sqrt() - 1.0).abs() < 0.15, "gamma has spread {}", variance.sqrt());
    }

    #[test]
    fn extinction_pushes_gamma_up_rather_than_down() {
        // The pull of the present: recent lineages have not had time to
        // die, so the reconstructed tree's nodes crowd toward the tips.
        let mut rng = Rng::new(0x0B10_3107);
        let sample = |mu: f64, rng: &mut Rng| -> f64 {
            let values: Vec<f64> = (0..120)
                .map(|_| gamma_statistic(&birth_death_tree(1.0, mu, 30, rng).unwrap()).unwrap())
                .collect();
            values.iter().sum::<f64>() / values.len() as f64
        };
        let pure = sample(0.0, &mut rng);
        let dying = sample(0.7, &mut rng);
        assert!(dying > pure + 0.5, "extinction moved gamma from {pure} to {dying}");
        assert!(dying > 0.0);
    }

    #[test]
    fn gamma_reads_the_position_of_the_branchings_and_not_their_number() {
        // Two trees with the same tips and the same height, differing only
        // in whether the branchings sit near the root or near the present.
        let early =
            PhyloTree::from_newick("((((A:2.97,B:2.97):0.01,C:2.98):0.01,D:2.99):0.01,E:3.0);")
                .unwrap();
        let late =
            PhyloTree::from_newick("((((A:0.03,B:0.03):0.01,C:0.04):0.01,D:0.05):2.95,E:3.0);")
                .unwrap();
        assert!(early.is_ultrametric(1e-12));
        assert!(late.is_ultrametric(1e-12));
        let (a, b) = (gamma_statistic(&early).unwrap(), gamma_statistic(&late).unwrap());
        assert!(a < -1.5, "a root-heavy tree gave gamma {a}");
        assert!(b > 1.5, "a tip-heavy tree gave gamma {b}");
    }

    #[test]
    fn gamma_refuses_a_tree_whose_tips_are_not_contemporaneous() {
        // Branch lengths that are not times make the statistic meaningless,
        // so it is refused rather than computed on the wrong quantity.
        let ragged = PhyloTree::from_newick("((A:0.4,B:0.02):0.05,(C:0.4,D:0.02):0.05);").unwrap();
        assert!(gamma_statistic(&ragged).is_err());
        let tiny = PhyloTree::from_newick("(A:1,B:1);").unwrap();
        assert!(gamma_statistic(&tiny).is_err());
        let flat = PhyloTree::from_newick("((A:0,B:0):0,C:0);").unwrap();
        assert!(gamma_statistic(&flat).is_err());
    }

    #[test]
    fn the_lineage_curve_starts_at_one_climbs_and_ends_at_the_tip_count() {
        let mut rng = Rng::new(0x0B10_3108);
        let tree = birth_death_tree(1.0, 0.3, 25, &mut rng).unwrap();
        let curve = lineage_through_time(&tree).unwrap();
        assert_eq!(curve[0], (0.0, 1));
        assert_eq!(curve.last().unwrap().1, 25);
        assert!((curve.last().unwrap().0 - tree.height()).abs() < 1e-12);
        for pair in curve.windows(2) {
            assert!(pair[1].0 >= pair[0].0 - 1e-12, "time went backwards");
            assert!(pair[1].1 >= pair[0].1, "lineages were lost");
        }
        // Every branching appears exactly once, so the count rises by the
        // number of internal nodes.
        let internal = (0..tree.len()).filter(|k| !tree.children(*k).is_empty()).count();
        assert_eq!(curve.len(), internal + 2);
        let lone = PhyloTree::new(vec![None], vec![0.0], vec!["A".to_string()]).unwrap();
        assert!(lineage_through_time(&lone).is_err());
    }

    #[test]
    fn the_lineage_curve_of_a_yule_tree_grows_at_the_speciation_rate() {
        // Under pure birth the wait from k lineages to k + 1 is exponential
        // with rate k * lambda, so the time at which the k-th lineage
        // appears averages (H_(k-1) - 1) / lambda after the root.
        let lambda = 2.0;
        let tips = 24;
        let replicates = 150;
        let mut rng = Rng::new(0x0B10_3109);
        let mut arrival = vec![0.0f64; tips + 1];
        for _ in 0..replicates {
            let tree = birth_death_tree(lambda, 0.0, tips, &mut rng).unwrap();
            let curve = lineage_through_time(&tree).unwrap();
            // The curve's final point repeats the last branching's count
            // at the stopping time, so only the first sighting of each
            // count is an arrival.
            let mut seen = vec![false; tips + 1];
            for (time, count) in curve {
                if count <= tips && !seen[count] {
                    seen[count] = true;
                    arrival[count] += time / replicates as f64;
                }
            }
        }
        for k in [6usize, 12, 24] {
            let harmonic: f64 = (1..k).map(|j| 1.0 / j as f64).sum();
            let expected = (harmonic - 1.0) / lambda;
            assert!(
                (arrival[k] - expected).abs() < 0.1 * expected.max(0.1),
                "the {k}th lineage arrived at {} where theory says {expected}",
                arrival[k]
            );
        }
    }

    #[test]
    fn re_rooting_changes_the_clades_but_not_the_bipartitions() {
        // The same unrooted tree written with two different roots. Rooted
        // clades disagree; the branches themselves do not.
        let rooted_between = PhyloTree::from_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let rooted_on_d = PhyloTree::from_newick("(((A:1,B:1):1,C:1):0.5,D:1.5);").unwrap();
        assert_ne!(rooted_between.splits(), rooted_on_d.splits());
        assert_eq!(rooted_between.bipartitions(), rooted_on_d.bipartitions());
        assert_eq!(rooted_between.bipartitions(), vec![names(&["A", "B"])]);
        assert!(rooted_between.robinson_foulds(&rooted_on_d).unwrap() > 0);
    }

    #[test]
    fn a_bipartition_needs_two_leaves_on_each_side_to_say_anything() {
        // A three-taxon tree has no informative branch however it is drawn.
        let tree = PhyloTree::from_newick("((A:1,B:1):1,C:2);").unwrap();
        assert!(tree.bipartitions().is_empty());
        assert_eq!(tree.splits(), vec![names(&["A", "B"])]);
        // Five taxa: two informative branches, and each is reported once.
        let bigger = PhyloTree::from_newick("(((A:1,B:1):1,C:2):1,(D:1,E:1):2);").unwrap();
        let parts = bigger.bipartitions();
        assert_eq!(parts.len(), 2);
        assert!(parts.contains(&names(&["A", "B"])));
        assert!(parts.contains(&names(&["D", "E"])));
    }

    #[test]
    fn parsimony_refuses_an_alphabet_it_cannot_hold_in_a_word() {
        // A caterpillar of 33 tips: s0 and s1 are sisters, and each later
        // tip attaches one rung further down.
        let tips = 33usize;
        let mut text = "(s0:1,s1:1)".to_string();
        for i in 2..tips {
            text = format!("({text}:1,s{i}:1)");
        }
        text.push(';');
        let tree = PhyloTree::from_newick(&text).unwrap();
        assert_eq!(tree.leaves().len(), tips);
        let state = |name: &str, states: &[u8]| -> u8 {
            let index: usize = name[1..].parse().unwrap();
            states[index]
        };

        // Thirty-two states, with the two sisters sharing one: the floor of
        // thirty-one changes is reachable and Fitch reaches it.
        let shared: Vec<u8> = (0..tips as u8).map(|k| k.saturating_sub(1)).collect();
        let characters: Vec<u8> =
            tree.leaves().iter().map(|k| state(&tree.labels[*k], &shared)).collect();
        assert_eq!(parsimony_fitch(&tree, &characters).unwrap(), 31);

        // Thirty-three distinct states will not fit in the state word.
        let all: Vec<u8> = (0..tips as u8).collect();
        let distinct: Vec<u8> =
            tree.leaves().iter().map(|k| state(&tree.labels[*k], &all)).collect();
        assert!(parsimony_fitch(&tree, &distinct).is_err());
    }
}
