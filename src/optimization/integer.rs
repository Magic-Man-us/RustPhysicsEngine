//! Integer programming, dynamic programming, and combinatorial search.
//!
//! Adding "and the answer must be a whole number" to a linear program changes
//! its character completely. The feasible region stops being a convex
//! polyhedron and becomes a scatter of lattice points inside one, so the
//! guarantee that made linear programming easy -- that an optimum sits at a
//! vertex, reachable by local moves -- is gone. What remains is the
//! relaxation: drop the integrality, solve the linear program, and use its
//! value as a bound on what any integer solution could achieve. Branch and
//! bound is that observation applied recursively, and the bound is the only
//! reason it terminates before enumerating everything.
//!
//! Most problems here have that flavour. A few do not, and those are the
//! dynamic programming classics: when a problem decomposes into overlapping
//! subproblems whose optimal solutions compose, the exponential search
//! collapses to a table and the answer is exact in polynomial time. Knapsack,
//! edit distance and the rest are here because the boundary between the two
//! situations is worth being able to see -- the 0/1 knapsack is
//! NP-hard and yet has a pseudo-polynomial table, which is not a
//! contradiction but a statement about what "polynomial" is measured against.
//!
//! Where an exact method is impractical the module gives a greedy one with
//! its proven ratio: first-fit-decreasing bin packing within `11/9` of
//! optimal, greedy set cover within `H_n`, longest-processing-time
//! scheduling within `4/3 - 1/(3m)`. Those ratios are worst-case guarantees
//! rather than typical behaviour, and the tests check the guarantee holds
//! against an exact answer on small instances rather than checking the greedy
//! answer is merely plausible.

use crate::error::GeomError;
use crate::exact::bigint::BigInt;
use crate::optimization::lp::{simplex, Cmp, LpProblem, LpResult};

/// Values within this of an integer are treated as integral.
const INTEGRALITY_TOL: f64 = 1e-7;

// ---------------------------------------------------------------------------
// Branch and bound over the linear relaxation
// ---------------------------------------------------------------------------

/// Solves a mixed-integer linear program by branch and bound.
///
/// Solves the linear relaxation; if the named variables all came out integral
/// the answer is optimal, and otherwise one fractional variable is chosen and
/// the problem split into the branch where it is rounded down and the branch
/// where it is rounded up. The relaxation's value bounds every integer
/// solution below it, so a branch whose relaxation is already worse than the
/// best integer solution found can be discarded whole -- which is the entire
/// content of the method, and the reason it beats enumeration.
///
/// `node_limit` caps the search. Returns `None` if the problem is infeasible
/// over the integers, or if the limit is reached before any integer solution
/// is found.
///
/// # Errors
/// Returns an error if a named variable is out of range, or the underlying
/// linear program is malformed.
pub fn branch_and_bound(
    p: &LpProblem,
    integer_vars: &[usize],
    node_limit: usize,
) -> Result<Option<(Vec<f64>, f64)>, GeomError> {
    p.validate()?;
    if integer_vars.iter().any(|&j| j >= p.n()) {
        return Err(GeomError::InvalidArgument("branch_and_bound: variable index out of range"));
    }
    // A maximisation improves upward and a minimisation downward; carrying the
    // sign lets one comparison serve both.
    let better = |a: f64, b: f64| if p.maximize { a > b } else { a < b };

    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut stack = vec![p.clone()];
    let mut nodes = 0usize;

    while let Some(node) = stack.pop() {
        nodes += 1;
        if nodes > node_limit {
            break;
        }
        let LpResult::Optimal { x, objective, .. } = simplex(&node)? else {
            // Infeasible or unbounded: nothing below this node to find.
            continue;
        };
        // Bound: this branch cannot beat what is already in hand.
        if let Some((_, incumbent)) = &best {
            if !better(objective, *incumbent) {
                continue;
            }
        }

        let fractional = integer_vars
            .iter()
            .copied()
            .find(|&j| (x[j] - x[j].round()).abs() > INTEGRALITY_TOL);
        let Some(j) = fractional else {
            // Every named variable is integral, so this is a candidate.
            let rounded: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(k, &v)| if integer_vars.contains(&k) { v.round() } else { v })
                .collect();
            let value = node.objective_at(&rounded);
            if best.as_ref().is_none_or(|(_, incumbent)| better(value, *incumbent)) {
                best = Some((rounded, value));
            }
            continue;
        };

        // Branch: the fractional value lies strictly between two integers, so
        // no integer solution is lost by excluding the gap between them.
        let floor = x[j].floor();
        for (bound, side) in [(floor, Cmp::Le), (floor + 1.0, Cmp::Ge)] {
            let mut child = node.clone();
            let (lo, hi) = child.bounds[j];
            match side {
                Cmp::Le => {
                    if bound < lo - INTEGRALITY_TOL {
                        continue;
                    }
                    child.bounds[j] = (lo, hi.min(bound));
                }
                Cmp::Ge => {
                    if bound > hi + INTEGRALITY_TOL {
                        continue;
                    }
                    child.bounds[j] = (lo.max(bound), hi);
                }
                Cmp::Eq => unreachable!("branching uses only inequalities"),
            }
            if child.bounds[j].0 <= child.bounds[j].1 + INTEGRALITY_TOL {
                stack.push(child);
            }
        }
    }
    Ok(best)
}

/// Adds Chvatal-Gomory rounding cuts to a linear program.
///
/// A cut is only worth the name if it is valid: satisfied by every integer
/// point of the feasible region, while removing part of the fractional
/// relaxation. The rounding cut earns that as follows. Scale a `<=` row by
/// some `lambda > 0`, so `lambda a . x <= lambda b` still holds. Rounding each
/// coefficient down can only lower the left-hand side when `x >= 0`, so
/// `floor(lambda a) . x <= lambda b`. But the left-hand side is now an integer
/// combination of integers, hence an integer, so it is bounded by the floor of
/// the right:
///
/// ```text
/// sum_j floor(lambda a_j) x_j <= floor(lambda b)
/// ```
///
/// Every non-negative integer point survives that, and a fractional one need
/// not. Multipliers are tried at the reciprocals of the row's own
/// coefficients and at a few small fractions, and a cut is kept only when the
/// current relaxation optimum actually violates it.
///
/// # Errors
/// Returns an error unless every variable is integer and non-negative, which
/// is what the rounding argument requires, or if the relaxation has no
/// optimum.
pub fn gomory_cuts(
    p: &LpProblem,
    integer_vars: &[usize],
    max_cuts: usize,
) -> Result<LpProblem, GeomError> {
    p.validate()?;
    if integer_vars.len() != p.n() || (0..p.n()).any(|j| !integer_vars.contains(&j)) {
        return Err(GeomError::InvalidArgument(
            "gomory_cuts requires every variable to be an integer variable",
        ));
    }
    if p.bounds.iter().any(|&(lo, _)| lo < 0.0) {
        return Err(GeomError::InvalidArgument(
            "gomory_cuts requires non-negative variables; the rounding step needs it",
        ));
    }

    let mut out = p.clone();
    for _ in 0..max_cuts {
        let LpResult::Optimal { x, .. } = simplex(&out)? else {
            break;
        };
        if x.iter().all(|v| (v - v.round()).abs() <= INTEGRALITY_TOL) {
            // Nothing fractional left to cut off.
            break;
        }

        let mut best: Option<(Vec<f64>, f64, f64)> = None;
        for i in 0..out.m() {
            // Orient the row as `<=`; an equality serves in both directions.
            let orientations: &[f64] = match out.constraint_types[i] {
                Cmp::Le => &[1.0],
                Cmp::Ge => &[-1.0],
                Cmp::Eq => &[1.0, -1.0],
            };
            for &sign in orientations {
                let row: Vec<f64> = (0..out.n()).map(|j| sign * out.a.get(i, j)).collect();
                let rhs = sign * out.b[i];

                let mut multipliers: Vec<f64> = vec![0.5, 1.0 / 3.0, 2.0 / 3.0, 0.25];
                for &v in &row {
                    if v.abs() > 1e-9 {
                        multipliers.push(1.0 / v.abs());
                    }
                }
                for lambda in multipliers {
                    if !(lambda > 0.0) || !lambda.is_finite() {
                        continue;
                    }
                    let cut: Vec<f64> = row.iter().map(|&v| (lambda * v).floor()).collect();
                    let bound = (lambda * rhs).floor();
                    if cut.iter().all(|&v| v == 0.0) {
                        continue;
                    }
                    let lhs: f64 = cut.iter().zip(&x).map(|(a, b)| a * b).sum();
                    let violation = lhs - bound;
                    if violation > 1e-6
                        && best.as_ref().is_none_or(|(_, _, v)| violation > *v)
                    {
                        best = Some((cut, bound, violation));
                    }
                }
            }
        }

        let Some((cut, bound, _)) = best else { break };
        // Append the cut as a new row.
        let (m, n) = (out.m(), out.n());
        let mut a = crate::linalg::matrix::Matrix::zeros(m + 1, n);
        for i in 0..m {
            for j in 0..n {
                a.set(i, j, out.a.get(i, j));
            }
        }
        for (j, &v) in cut.iter().enumerate() {
            a.set(m, j, v);
        }
        out.a = a;
        out.b.push(bound);
        out.constraint_types.push(Cmp::Le);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The knapsack family
// ---------------------------------------------------------------------------

/// The 0/1 knapsack by dynamic programming: each item taken at most once.
///
/// Returns the best value and which items to take. The table is
/// `O(n * capacity)`, which is polynomial in the *value* of the capacity but
/// exponential in the number of digits it takes to write it down -- the
/// problem is NP-hard, and the table is pseudo-polynomial rather than a
/// contradiction of that.
///
/// # Panics
/// Panics unless the value and weight lists have the same length.
#[must_use]
pub fn knapsack_01(values: &[u64], weights: &[u64], capacity: u64) -> (u64, Vec<bool>) {
    assert!(values.len() == weights.len(), "knapsack_01 needs one weight per value");
    let n = values.len();
    let cap = capacity as usize;
    let mut table = vec![vec![0u64; cap + 1]; n + 1];
    for i in 1..=n {
        let w = weights[i - 1] as usize;
        for c in 0..=cap {
            table[i][c] = table[i - 1][c];
            if w <= c {
                let with = table[i - 1][c - w] + values[i - 1];
                if with > table[i][c] {
                    table[i][c] = with;
                }
            }
        }
    }
    // Walk back through the table: an item was taken exactly where the value
    // differs from the row above.
    let mut chosen = vec![false; n];
    let mut c = cap;
    for i in (1..=n).rev() {
        if table[i][c] != table[i - 1][c] {
            chosen[i - 1] = true;
            c -= weights[i - 1] as usize;
        }
    }
    (table[n][cap], chosen)
}

/// The unbounded knapsack: each item available without limit.
///
/// Returns the best value and how many of each item to take. A one-dimensional
/// table suffices, because an item may be reused within the same pass.
///
/// # Panics
/// Panics unless the lists match in length and every weight is positive.
#[must_use]
pub fn knapsack_unbounded(values: &[u64], weights: &[u64], capacity: u64) -> (u64, Vec<u64>) {
    assert!(values.len() == weights.len(), "knapsack_unbounded needs one weight per value");
    assert!(weights.iter().all(|&w| w > 0), "knapsack_unbounded requires positive weights");
    let cap = capacity as usize;
    let mut best = vec![0u64; cap + 1];
    let mut taken = vec![usize::MAX; cap + 1];
    for c in 1..=cap {
        for (i, (&v, &w)) in values.iter().zip(weights).enumerate() {
            let w = w as usize;
            if w <= c && best[c - w] + v > best[c] {
                best[c] = best[c - w] + v;
                taken[c] = i;
            }
        }
    }
    let mut counts = vec![0u64; values.len()];
    let mut c = cap;
    while c > 0 && taken[c] != usize::MAX {
        let i = taken[c];
        counts[i] += 1;
        c -= weights[i] as usize;
    }
    (best[cap], counts)
}

/// The bounded knapsack: each item available up to its own limit.
///
/// Expanded by binary splitting -- an item with a limit of `k` becomes items
/// of multiplicity `1, 2, 4, ...` summing to `k` -- so any count up to the
/// limit is expressible and the 0/1 solver applies. That costs
/// `O(log k)` copies rather than the `k` a naive expansion would need.
///
/// # Panics
/// Panics unless all three lists match in length.
#[must_use]
pub fn knapsack_bounded(
    values: &[u64],
    weights: &[u64],
    limits: &[u64],
    capacity: u64,
) -> (u64, Vec<u64>) {
    assert!(
        values.len() == weights.len() && values.len() == limits.len(),
        "knapsack_bounded needs one weight and limit per value"
    );
    let mut expanded_values = Vec::new();
    let mut expanded_weights = Vec::new();
    let mut origin = Vec::new();
    let mut multiplicity = Vec::new();
    for (i, ((&v, &w), &limit)) in values.iter().zip(weights).zip(limits).enumerate() {
        let mut remaining = limit;
        let mut piece = 1u64;
        while remaining > 0 {
            let take = piece.min(remaining);
            expanded_values.push(v * take);
            expanded_weights.push(w * take);
            origin.push(i);
            multiplicity.push(take);
            remaining -= take;
            piece *= 2;
        }
    }
    let (best, chosen) = knapsack_01(&expanded_values, &expanded_weights, capacity);
    let mut counts = vec![0u64; values.len()];
    for (k, &taken) in chosen.iter().enumerate() {
        if taken {
            counts[origin[k]] += multiplicity[k];
        }
    }
    (best, counts)
}

/// The multiple knapsack: several bins, each item into at most one.
///
/// Solved greedily by value density with a first-fit placement, which is not
/// exact -- the problem is NP-hard even with two bins -- so the result is a
/// lower bound on the optimum. Returns the total value and the bin each item
/// went into, `None` for an item left out.
///
/// # Panics
/// Panics unless the lists match in length.
#[must_use]
pub fn knapsack_multiple(
    values: &[u64],
    weights: &[u64],
    capacities: &[u64],
) -> (u64, Vec<Option<usize>>) {
    assert!(values.len() == weights.len(), "knapsack_multiple needs one weight per value");
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Densest first: the classic greedy order for a knapsack.
    order.sort_by(|&a, &b| {
        let da = values[a] as f64 / weights[a].max(1) as f64;
        let db = values[b] as f64 / weights[b].max(1) as f64;
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut remaining: Vec<u64> = capacities.to_vec();
    let mut placement = vec![None; n];
    let mut total = 0u64;
    for &i in &order {
        if let Some(bin) = remaining.iter().position(|&r| r >= weights[i]) {
            remaining[bin] -= weights[i];
            placement[i] = Some(bin);
            total += values[i];
        }
    }
    (total, placement)
}

/// The 0/1 knapsack by branch and bound over the fractional relaxation.
///
/// The relaxation of a knapsack is solved by taking items in density order
/// and splitting the last one, which gives a bound in linear time once the
/// items are sorted. Nodes whose bound cannot beat the incumbent are pruned.
///
/// Exact, and must agree with [`knapsack_01`] on every instance -- one walks a
/// table and the other a search tree, so their agreement is a real check on
/// both.
///
/// # Panics
/// Panics unless the lists match in length.
#[must_use]
pub fn knapsack_branch_bound(values: &[u64], weights: &[u64], capacity: u64) -> (u64, Vec<bool>) {
    assert!(values.len() == weights.len(), "knapsack_branch_bound needs one weight per value");
    let n = values.len();
    if n == 0 {
        return (0, Vec::new());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let da = values[a] as f64 / weights[a].max(1) as f64;
        let db = values[b] as f64 / weights[b].max(1) as f64;
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    // The fractional optimum from position `k` onward, given `room` left.
    let bound = |k: usize, room: u64, value: u64| -> f64 {
        let mut left = room;
        let mut total = value as f64;
        for &i in &order[k..] {
            if weights[i] <= left {
                left -= weights[i];
                total += values[i] as f64;
            } else {
                // The relaxation may split this one item, and nothing after
                // it can add more than that fraction is worth.
                total += values[i] as f64 * left as f64 / weights[i].max(1) as f64;
                break;
            }
        }
        total
    };

    let mut best_value = 0u64;
    let mut best_take = vec![false; n];
    let mut take = vec![false; n];

    // An explicit stack of (depth, remaining capacity, value so far, whether
    // the parent's decision has been undone) rather than recursion.
    struct Frame {
        depth: usize,
        room: u64,
        value: u64,
        branch: u8,
    }
    let mut stack = vec![Frame { depth: 0, room: capacity, value: 0, branch: 0 }];
    while let Some(frame) = stack.last_mut() {
        let Frame { depth, room, value, branch } = *frame;
        if depth == n || branch == 2 {
            if value > best_value && depth == n {
                best_value = value;
                best_take.copy_from_slice(&take);
            }
            if depth == n && branch == 0 {
                // Nothing to undo at a leaf.
            }
            stack.pop();
            if let Some(parent) = stack.last() {
                let i = order[parent.depth];
                take[i] = false;
            }
            continue;
        }
        frame.branch += 1;
        let i = order[depth];
        let (next_room, next_value, feasible) = if branch == 0 {
            // Take the item, if it fits.
            (room.checked_sub(weights[i]), value + values[i], weights[i] <= room)
        } else {
            (Some(room), value, true)
        };
        if !feasible {
            continue;
        }
        let room_left = next_room.unwrap_or(0);
        if bound(depth + 1, room_left, next_value) <= best_value as f64 {
            continue;
        }
        take[i] = branch == 0;
        if next_value > best_value {
            best_value = next_value;
            best_take.copy_from_slice(&take);
        }
        stack.push(Frame { depth: depth + 1, room: room_left, value: next_value, branch: 0 });
    }
    (best_value, best_take)
}

// ---------------------------------------------------------------------------
// Subset sum and partition
// ---------------------------------------------------------------------------

/// Indices of a subset summing exactly to `target`, if one exists.
///
/// # Panics
/// Panics if the values are large enough that the table would not fit.
#[must_use]
pub fn subset_sum(xs: &[u64], target: u64) -> Option<Vec<usize>> {
    let t = target as usize;
    let n = xs.len();
    let mut reachable = vec![vec![false; t + 1]; n + 1];
    for row in reachable.iter_mut() {
        row[0] = true;
    }
    for i in 1..=n {
        let v = xs[i - 1] as usize;
        for s in 0..=t {
            reachable[i][s] = reachable[i - 1][s] || (v <= s && reachable[i - 1][s - v]);
        }
    }
    if !reachable[n][t] {
        return None;
    }
    let mut chosen = Vec::new();
    let mut s = t;
    for i in (1..=n).rev() {
        let v = xs[i - 1] as usize;
        if !reachable[i - 1][s] {
            chosen.push(i - 1);
            s -= v;
        }
    }
    chosen.reverse();
    Some(chosen)
}

/// How many subsets sum exactly to `target`.
///
/// Counted as a [`BigInt`], since the number of subsets of an `n`-element set
/// is `2^n` and the count routinely overflows a machine word well before the
/// table does.
#[must_use]
pub fn subset_sum_count(xs: &[u64], target: u64) -> BigInt {
    let t = target as usize;
    let mut counts = vec![BigInt::zero(); t + 1];
    counts[0] = BigInt::one();
    for &v in xs {
        let v = v as usize;
        if v > t {
            continue;
        }
        // Descending, so each item is counted once per subset.
        for s in (v..=t).rev() {
            let carried = counts[s - v].clone();
            counts[s] = counts[s].add(&carried);
        }
    }
    counts[t].clone()
}

/// Splits the values into two groups whose totals are as close as possible.
///
/// Returns the difference and the membership flags. The problem is
/// NP-hard in general and solved here by the subset-sum table over half the
/// total, which is exact and pseudo-polynomial.
#[must_use]
pub fn partition_min_diff(xs: &[u64]) -> (u64, Vec<bool>) {
    let total: u64 = xs.iter().sum();
    let half = (total / 2) as usize;
    let n = xs.len();
    let mut reachable = vec![vec![false; half + 1]; n + 1];
    for row in reachable.iter_mut() {
        row[0] = true;
    }
    for i in 1..=n {
        let v = xs[i - 1] as usize;
        for s in 0..=half {
            reachable[i][s] = reachable[i - 1][s] || (v <= s && reachable[i - 1][s - v]);
        }
    }
    // The largest reachable total at or below half minimises the gap.
    let best = (0..=half).rev().find(|&s| reachable[n][s]).unwrap_or(0);
    let mut flags = vec![false; n];
    let mut s = best;
    for i in (1..=n).rev() {
        let v = xs[i - 1] as usize;
        if !reachable[i - 1][s] {
            flags[i - 1] = true;
            s -= v;
        }
    }
    (total - 2 * best as u64, flags)
}

// ---------------------------------------------------------------------------
// Packing and covering
// ---------------------------------------------------------------------------

/// Bin packing by first-fit-decreasing: sort the items large to small and put
/// each into the first bin it fits.
///
/// Returns the item indices in each bin. The rule uses at most
/// `11/9 OPT + 6/9` bins, a bound that is tight -- so the tests check the
/// guarantee against an exact answer rather than checking the result merely
/// looks reasonable.
///
/// # Panics
/// Panics if any item exceeds the bin capacity, which makes packing
/// impossible rather than merely hard.
#[must_use]
pub fn bin_packing_ffd(sizes: &[f64], capacity: f64) -> Vec<Vec<usize>> {
    assert!(capacity > 0.0, "bin_packing_ffd requires a positive capacity");
    assert!(
        sizes.iter().all(|&s| s <= capacity + 1e-12 && s >= 0.0),
        "every item must fit in an empty bin"
    );
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| sizes[b].partial_cmp(&sizes[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut bins: Vec<Vec<usize>> = Vec::new();
    let mut room: Vec<f64> = Vec::new();
    for &i in &order {
        match room.iter().position(|&r| r >= sizes[i] - 1e-12) {
            Some(b) => {
                room[b] -= sizes[i];
                bins[b].push(i);
            }
            None => {
                room.push(capacity - sizes[i]);
                bins.push(vec![i]);
            }
        }
    }
    bins
}

/// The fewest bins any packing could use: the total size divided by the
/// capacity, rounded up.
///
/// A valid lower bound because a bin holds at most `capacity`, so no packing
/// can use fewer. It is not always attainable -- three items of size 0.4 need
/// two bins though their total is 1.2 -- which is exactly why it is a bound
/// and not an answer.
#[must_use]
pub fn bin_packing_lower_bound(sizes: &[f64], capacity: f64) -> usize {
    assert!(capacity > 0.0, "bin_packing_lower_bound requires a positive capacity");
    let total: f64 = sizes.iter().sum();
    (total / capacity).ceil().max(0.0) as usize
}

/// The exact minimum number of bins, by trying each count in turn.
///
/// Exponential, and meant for the small instances the tests use to check the
/// first-fit-decreasing guarantee. Returns the packing.
///
/// # Panics
/// Panics under the same conditions as [`bin_packing_ffd`].
#[must_use]
pub fn bin_packing_exact_small(sizes: &[f64], capacity: f64) -> Vec<Vec<usize>> {
    assert!(capacity > 0.0, "bin_packing_exact_small requires a positive capacity");
    assert!(sizes.len() <= 12, "bin_packing_exact_small is for instances of at most twelve items");
    let n = sizes.len();
    if n == 0 {
        return Vec::new();
    }
    let lower = bin_packing_lower_bound(sizes, capacity).max(1);
    for count in lower..=n {
        let mut assignment = vec![usize::MAX; n];
        let mut room = vec![capacity; count];
        if pack(sizes, 0, &mut assignment, &mut room) {
            let mut bins = vec![Vec::new(); count];
            for (i, &b) in assignment.iter().enumerate() {
                bins[b].push(i);
            }
            return bins;
        }
    }
    // Every item in its own bin always works.
    (0..n).map(|i| vec![i]).collect()
}

/// Depth-first placement for [`bin_packing_exact_small`].
fn pack(sizes: &[f64], i: usize, assignment: &mut Vec<usize>, room: &mut Vec<f64>) -> bool {
    if i == sizes.len() {
        return true;
    }
    for b in 0..room.len() {
        if room[b] >= sizes[i] - 1e-12 {
            room[b] -= sizes[i];
            assignment[i] = b;
            if pack(sizes, i + 1, assignment, room) {
                return true;
            }
            room[b] += sizes[i];
            assignment[i] = usize::MAX;
        }
    }
    false
}

/// Greedy set cover: repeatedly take the set covering the most of what is
/// still uncovered.
///
/// Returns the indices of the chosen sets, or `None` if the sets do not cover
/// the universe at all. Greedy uses at most `H_n` times the optimal number of
/// sets, where `H_n` is the `n`-th harmonic number, and no polynomial
/// algorithm does asymptotically better unless P equals NP -- so this is not
/// a placeholder for something better.
#[must_use]
pub fn set_cover_greedy(universe_n: usize, sets: &[Vec<usize>]) -> Option<Vec<usize>> {
    let mut covered = vec![false; universe_n];
    let mut chosen = Vec::new();
    let mut remaining = universe_n;
    while remaining > 0 {
        let best = (0..sets.len())
            .filter(|i| !chosen.contains(i))
            .max_by_key(|&i| sets[i].iter().filter(|&&e| e < universe_n && !covered[e]).count());
        let best = best?;
        let gain = sets[best].iter().filter(|&&e| e < universe_n && !covered[e]).count();
        if gain == 0 {
            return None;
        }
        for &e in &sets[best] {
            if e < universe_n && !covered[e] {
                covered[e] = true;
                remaining -= 1;
            }
        }
        chosen.push(best);
    }
    Some(chosen)
}

/// The exact minimum set cover, by trying every subset of the sets in order of
/// size.
///
/// For the small instances that make the greedy ratio checkable.
///
/// # Panics
/// Panics if there are more than 20 sets, where the enumeration stops being
/// reasonable.
#[must_use]
pub fn set_cover_exact_small(universe_n: usize, sets: &[Vec<usize>]) -> Option<Vec<usize>> {
    assert!(sets.len() <= 20, "set_cover_exact_small is for at most twenty sets");
    let m = sets.len();
    let masks: Vec<u64> = sets
        .iter()
        .map(|s| s.iter().filter(|&&e| e < universe_n).fold(0u64, |acc, &e| acc | (1 << e)))
        .collect();
    let full = if universe_n >= 64 { u64::MAX } else { (1u64 << universe_n) - 1 };

    for size in 0..=m {
        for combination in 0u32..(1u32 << m) {
            if combination.count_ones() as usize != size {
                continue;
            }
            let mut union = 0u64;
            for (i, &mask) in masks.iter().enumerate() {
                if combination & (1 << i) != 0 {
                    union |= mask;
                }
            }
            if union == full {
                return Some((0..m).filter(|&i| combination & (1 << i) != 0).collect());
            }
        }
    }
    None
}

/// Uncapacitated facility location, solved greedily.
///
/// `open_costs[i]` is the fixed cost of opening facility `i` and
/// `serve_costs[(i, j)]` the cost of serving client `j` from it. Facilities
/// are opened one at a time, each time the one whose opening cost plus
/// improved service most reduces the total.
///
/// Returns the total cost and which facilities to open.
///
/// # Panics
/// Panics unless the shapes agree and there is at least one facility.
#[must_use]
pub fn facility_location_greedy(
    open_costs: &[f64],
    serve_costs: &crate::linalg::matrix::Matrix,
) -> (f64, Vec<bool>) {
    let m = open_costs.len();
    assert!(m > 0 && serve_costs.rows == m, "facility_location_greedy: shape mismatch");
    let n = serve_costs.cols;

    let mut open = vec![false; m];
    let mut best_serve = vec![f64::INFINITY; n];
    let mut total = f64::INFINITY;

    loop {
        let mut improvement: Option<(f64, usize, Vec<f64>)> = None;
        for i in 0..m {
            if open[i] {
                continue;
            }
            let candidate: Vec<f64> =
                (0..n).map(|j| best_serve[j].min(serve_costs.get(i, j))).collect();
            let cost: f64 = open_costs[i]
                + candidate.iter().sum::<f64>()
                + (0..m).filter(|&k| open[k]).map(|k| open_costs[k]).sum::<f64>();
            if cost < total && improvement.as_ref().is_none_or(|(best, _, _)| cost < *best) {
                improvement = Some((cost, i, candidate));
            }
        }
        let Some((cost, i, serve)) = improvement else { break };
        open[i] = true;
        best_serve = serve;
        total = cost;
    }
    (total, open)
}

/// The cutting stock problem by column generation, relaxed.
///
/// Each cutting pattern is a column of the linear program, and there are far
/// too many to write down, so patterns are generated on demand: solve the
/// relaxation over the patterns in hand, read the dual prices off it, and ask
/// which single new pattern would be most profitable at those prices. That
/// subproblem is an unbounded knapsack, and when its best pattern is not
/// profitable the relaxation is optimal over *all* patterns without ever
/// having enumerated them.
///
/// Returns the relaxed number of stock lengths needed, which lower-bounds the
/// integer answer.
///
/// # Errors
/// Returns an error if the inputs disagree in length, or a piece is longer
/// than the stock.
pub fn cutting_stock_column_generation(
    demand: &[u64],
    lengths: &[u64],
    stock_length: u64,
    max_rounds: usize,
) -> Result<f64, GeomError> {
    if demand.len() != lengths.len() || demand.is_empty() {
        return Err(GeomError::InvalidArgument("cutting_stock: one demand per length"));
    }
    if lengths.iter().any(|&l| l == 0 || l > stock_length) {
        return Err(GeomError::InvalidArgument("cutting_stock: a piece does not fit the stock"));
    }
    let n = lengths.len();

    // Start with the trivial patterns: one length repeated as often as fits.
    let mut patterns: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut p = vec![0.0; n];
            p[i] = (stock_length / lengths[i]) as f64;
            p
        })
        .collect();

    let mut value = f64::INFINITY;
    for _ in 0..max_rounds {
        // min sum(x) s.t. each length's demand is met by the patterns used.
        let mut a = crate::linalg::matrix::Matrix::zeros(n, patterns.len());
        for (k, pattern) in patterns.iter().enumerate() {
            for (i, &count) in pattern.iter().enumerate() {
                a.set(i, k, count);
            }
        }
        let p = LpProblem {
            c: vec![1.0; patterns.len()],
            a,
            b: demand.iter().map(|&d| d as f64).collect(),
            constraint_types: vec![Cmp::Ge; n],
            bounds: vec![(0.0, f64::INFINITY); patterns.len()],
            maximize: false,
        };
        let LpResult::Optimal { objective, duals, .. } = simplex(&p)? else {
            return Err(GeomError::Degenerate("cutting_stock: the relaxation has no optimum"));
        };
        value = objective;

        // Pricing: the most profitable new pattern is an unbounded knapsack
        // with the duals as values and the piece lengths as weights.
        let scale = 10_000.0;
        let values: Vec<u64> = duals.iter().map(|&d| (d.max(0.0) * scale) as u64).collect();
        let (best, counts) = knapsack_unbounded(&values, lengths, stock_length);
        if best as f64 / scale <= 1.0 + 1e-6 {
            // No pattern pays for the stock length it consumes: optimal.
            break;
        }
        let column: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        if patterns.contains(&column) {
            break;
        }
        patterns.push(column);
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Dynamic programming classics
// ---------------------------------------------------------------------------

/// The fewest coins summing to `amount`, or `None` if no combination does.
///
/// Returns how many of each denomination. Greedy is wrong for general
/// denominations -- with coins 1, 3, 4 and an amount of 6, greedy takes
/// 4 + 1 + 1 while two threes do it -- so this is a table, not a loop.
#[must_use]
pub fn coin_change_min(coins: &[u64], amount: u64) -> Option<Vec<u64>> {
    let target = amount as usize;
    let mut best = vec![usize::MAX; target + 1];
    let mut used = vec![usize::MAX; target + 1];
    best[0] = 0;
    for s in 1..=target {
        for (i, &c) in coins.iter().enumerate() {
            let c = c as usize;
            if c > 0 && c <= s && best[s - c] != usize::MAX && best[s - c] + 1 < best[s] {
                best[s] = best[s - c] + 1;
                used[s] = i;
            }
        }
    }
    if best[target] == usize::MAX {
        return None;
    }
    let mut counts = vec![0u64; coins.len()];
    let mut s = target;
    while s > 0 {
        let i = used[s];
        counts[i] += 1;
        s -= coins[i] as usize;
    }
    Some(counts)
}

/// How many combinations of coins sum to `amount`, order disregarded.
///
/// Iterating coins in the outer loop is what makes this count combinations
/// rather than permutations: each coin is considered once for the whole table,
/// so `1 + 2` and `2 + 1` are never both counted.
#[must_use]
pub fn coin_change_count(coins: &[u64], amount: u64) -> BigInt {
    let target = amount as usize;
    let mut ways = vec![BigInt::zero(); target + 1];
    ways[0] = BigInt::one();
    for &c in coins {
        let c = c as usize;
        if c == 0 {
            continue;
        }
        for s in c..=target {
            let carried = ways[s - c].clone();
            ways[s] = ways[s].add(&carried);
        }
    }
    ways[target].clone()
}

/// Indices of a longest strictly increasing subsequence, in `O(n log n)`.
///
/// The trick is to keep, for each length, the smallest value that can end a
/// subsequence of that length. That list is sorted by construction, so the
/// position each new element belongs at is a binary search rather than a scan
/// -- which is what turns the quadratic table into an `n log n` sweep.
#[must_use]
pub fn longest_increasing_subsequence(x: &[f64]) -> Vec<usize> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    // `tails[k]` is the index of the smallest tail of an increasing
    // subsequence of length `k + 1`.
    let mut tails: Vec<usize> = Vec::new();
    let mut previous = vec![usize::MAX; n];
    for i in 0..n {
        let mut lo = 0usize;
        let mut hi = tails.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x[tails[mid]] < x[i] {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > 0 {
            previous[i] = tails[lo - 1];
        }
        if lo == tails.len() {
            tails.push(i);
        } else {
            tails[lo] = i;
        }
    }
    let mut out = Vec::with_capacity(tails.len());
    let mut k = *tails.last().unwrap_or(&0);
    while k != usize::MAX {
        out.push(k);
        k = previous[k];
    }
    out.reverse();
    out
}

/// One edit in a transformation from one sequence to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOp {
    /// Both sequences agree here; `a` index, `b` index.
    Keep(usize, usize),
    /// Replace `a[i]` with `b[j]`.
    Substitute(usize, usize),
    /// Remove `a[i]`.
    Delete(usize),
    /// Insert `b[j]`.
    Insert(usize),
}

/// The Levenshtein distance: the fewest single-symbol insertions, deletions
/// and substitutions turning `a` into `b`.
///
/// It is a metric on sequences -- symmetric, zero only between equal
/// sequences, and obeying the triangle inequality -- which is what makes it
/// usable for clustering and nearest-neighbour search rather than merely a
/// similarity score.
#[must_use]
pub fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut previous: Vec<usize> = (0..=m).collect();
    let mut current = vec![0usize; m + 1];
    for i in 1..=n {
        current[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            current[j] = (previous[j] + 1).min(current[j - 1] + 1).min(previous[j - 1] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[m]
}

/// The edits themselves, in order, from a full table.
///
/// Applying them to `a` reproduces `b`, and their count of non-`Keep`
/// operations is exactly [`edit_distance`].
#[must_use]
pub fn edit_distance_ops(a: &[u8], b: &[u8]) -> Vec<EditOp> {
    let (n, m) = (a.len(), b.len());
    let mut table = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in table.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        table[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            table[i][j] =
                (table[i - 1][j] + 1).min(table[i][j - 1] + 1).min(table[i - 1][j - 1] + cost);
        }
    }

    let mut ops = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            if table[i][j] == table[i - 1][j - 1] + cost {
                ops.push(if cost == 0 {
                    EditOp::Keep(i - 1, j - 1)
                } else {
                    EditOp::Substitute(i - 1, j - 1)
                });
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && table[i][j] == table[i - 1][j] + 1 {
            ops.push(EditOp::Delete(i - 1));
            i -= 1;
            continue;
        }
        ops.push(EditOp::Insert(j - 1));
        j -= 1;
    }
    ops.reverse();
    ops
}

/// A longest common subsequence of two sequences.
#[must_use]
pub fn longest_common_subsequence(a: &[u8], b: &[u8]) -> Vec<u8> {
    let (n, m) = (a.len(), b.len());
    let mut table = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            table[i][j] = if a[i - 1] == b[j - 1] {
                table[i - 1][j - 1] + 1
            } else {
                table[i - 1][j].max(table[i][j - 1])
            };
        }
    }
    let mut out = Vec::with_capacity(table[n][m]);
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            out.push(a[i - 1]);
            i -= 1;
            j -= 1;
        } else if table[i - 1][j] >= table[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    out.reverse();
    out
}

/// The cheapest way to parenthesise a chain of matrix multiplications.
///
/// `dims` holds the shared dimensions: matrix `k` is `dims[k]` by
/// `dims[k + 1]`. Returns the scalar multiplication count and the
/// parenthesisation as a string.
///
/// The order matters enormously -- multiplying a `1x100`, `100x1` and `1x100`
/// chain costs 200 one way and 20,000 the other -- and the number of
/// parenthesisations is Catalan, so the table is what makes it tractable.
///
/// # Panics
/// Panics unless there are at least two dimensions.
#[must_use]
pub fn matrix_chain_order(dims: &[usize]) -> (u64, String) {
    assert!(dims.len() >= 2, "matrix_chain_order needs at least one matrix");
    let n = dims.len() - 1;
    let mut cost = vec![vec![0u64; n]; n];
    let mut split = vec![vec![0usize; n]; n];
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            cost[i][j] = u64::MAX;
            for k in i..j {
                let c = cost[i][k]
                    + cost[k + 1][j]
                    + (dims[i] * dims[k + 1] * dims[j + 1]) as u64;
                if c < cost[i][j] {
                    cost[i][j] = c;
                    split[i][j] = k;
                }
            }
        }
    }
    fn render(split: &[Vec<usize>], i: usize, j: usize, out: &mut String) {
        if i == j {
            out.push_str(&format!("A{i}"));
            return;
        }
        out.push('(');
        render(split, i, split[i][j], out);
        render(split, split[i][j] + 1, j, out);
        out.push(')');
    }
    let mut rendered = String::new();
    render(&split, 0, n - 1, &mut rendered);
    (cost[0][n - 1], rendered)
}

/// The most valuable way to cut a rod of length `n` into pieces.
///
/// `prices[k]` is what a piece of length `k + 1` sells for. Returns the value
/// and the piece lengths.
#[must_use]
pub fn rod_cutting(prices: &[u64], n: usize) -> (u64, Vec<usize>) {
    let mut best = vec![0u64; n + 1];
    let mut first = vec![0usize; n + 1];
    for length in 1..=n {
        for (k, &price) in prices.iter().enumerate() {
            let piece = k + 1;
            if piece <= length && best[length - piece] + price > best[length] {
                best[length] = best[length - piece] + price;
                first[length] = piece;
            }
        }
    }
    let mut pieces = Vec::new();
    let mut length = n;
    while length > 0 && first[length] > 0 {
        pieces.push(first[length]);
        length -= first[length];
    }
    (best[n], pieces)
}

/// The fewest drops that always determine the critical floor, with `eggs`
/// eggs and `floors` floors.
///
/// The classic answer for two eggs and a hundred floors is fourteen: drop
/// from 14, then 27, then 39, and so on, each interval one shorter than the
/// last so the worst case stays flat.
#[must_use]
pub fn egg_drop(eggs: usize, floors: usize) -> u64 {
    if eggs == 0 || floors == 0 {
        return 0;
    }
    // `reach[e]` is how many floors `e` eggs can cover in the drops so far.
    let mut reach = vec![0u64; eggs + 1];
    let mut drops = 0u64;
    while (reach[eggs] as usize) < floors {
        drops += 1;
        for e in (1..=eggs).rev() {
            // A drop either breaks the egg -- covering what one fewer egg
            // covers below -- or it does not, covering the same eggs above.
            reach[e] = reach[e] + reach[e - 1] + 1;
        }
    }
    drops
}

/// The expected search cost of the optimal binary search tree over keys with
/// the given access frequencies.
///
/// Frequencies are taken in key order. The optimum is not the balanced tree:
/// a key accessed far more often than the rest belongs near the root even if
/// that unbalances everything else.
#[must_use]
pub fn optimal_bst(frequencies: &[f64]) -> f64 {
    let n = frequencies.len();
    if n == 0 {
        return 0.0;
    }
    let mut prefix = vec![0.0; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + frequencies[i];
    }
    let sum = |i: usize, j: usize| prefix[j + 1] - prefix[i];

    let mut cost = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        cost[i][i] = frequencies[i];
    }
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            cost[i][j] = f64::INFINITY;
            for r in i..=j {
                let left = if r > i { cost[i][r - 1] } else { 0.0 };
                let right = if r < j { cost[r + 1][j] } else { 0.0 };
                // Every key in the range gains one level of depth whichever
                // root is chosen, which is where the `sum` term comes from.
                let c = left + right + sum(i, j);
                if c < cost[i][j] {
                    cost[i][j] = c;
                }
            }
        }
    }
    cost[0][n - 1]
}

/// The least-cost state path through a trellis.
///
/// `transition[(a, b)]` is the cost of moving from state `a` to state `b`, and
/// `emission[(s, t)]` the cost of state `s` at time `t`. Returns the best path.
///
/// The same recursion as the probabilistic Viterbi algorithm in
/// `stochastic::hmm`, stated in costs rather than log-probabilities -- which
/// is the more general form, since any additive path cost works.
///
/// # Errors
/// Returns an error if the matrices disagree in shape or there are no steps.
pub fn viterbi_generic(
    transition: &crate::linalg::matrix::Matrix,
    emission: &crate::linalg::matrix::Matrix,
) -> Result<Vec<usize>, GeomError> {
    let s = transition.rows;
    if !transition.is_square() || emission.rows != s || emission.cols == 0 {
        return Err(GeomError::InvalidArgument("viterbi_generic: shape mismatch"));
    }
    let t = emission.cols;
    let mut cost = vec![vec![f64::INFINITY; s]; t];
    let mut from = vec![vec![0usize; s]; t];
    for i in 0..s {
        cost[0][i] = emission.get(i, 0);
    }
    for step in 1..t {
        for j in 0..s {
            for i in 0..s {
                let c = cost[step - 1][i] + transition.get(i, j) + emission.get(j, step);
                if c < cost[step][j] {
                    cost[step][j] = c;
                    from[step][j] = i;
                }
            }
        }
    }
    let mut best = 0usize;
    for i in 1..s {
        if cost[t - 1][i] < cost[t - 1][best] {
            best = i;
        }
    }
    let mut path = vec![0usize; t];
    path[t - 1] = best;
    for step in (1..t).rev() {
        path[step - 1] = from[step][path[step]];
    }
    Ok(path)
}

// ---------------------------------------------------------------------------
// Exact cover and constraint search
// ---------------------------------------------------------------------------

/// Solves an exact cover problem: choose rows so that every column is covered
/// exactly once.
///
/// Implemented as Knuth's Algorithm X with the column-selection heuristic that
/// makes dancing links effective -- always branch on the column with the
/// fewest remaining options, which fails fast and keeps the search tree
/// narrow. The doubly linked list of the classic implementation is replaced
/// here by bitmask bookkeeping, which is the same algorithm with the same
/// search order for the column counts this module needs.
///
/// Returns the chosen row indices, or `None` if no exact cover exists.
///
/// # Errors
/// Returns an error for a ragged matrix or more than 64 columns.
pub fn exact_cover_dlx(matrix: &[Vec<bool>]) -> Result<Option<Vec<usize>>, GeomError> {
    if matrix.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let cols = matrix[0].len();
    if matrix.iter().any(|r| r.len() != cols) {
        return Err(GeomError::InvalidArgument("exact_cover_dlx: ragged matrix"));
    }
    if cols > 64 {
        return Err(GeomError::InvalidArgument("exact_cover_dlx: at most 64 columns"));
    }
    let rows: Vec<u64> = matrix
        .iter()
        .map(|r| r.iter().enumerate().filter(|(_, &v)| v).fold(0u64, |acc, (j, _)| acc | (1 << j)))
        .collect();
    let full = if cols == 64 { u64::MAX } else { (1u64 << cols) - 1 };

    let mut chosen = Vec::new();
    let mut used = vec![false; rows.len()];
    if cover(&rows, full, 0, &mut used, &mut chosen) {
        chosen.sort_unstable();
        Ok(Some(chosen))
    } else {
        Ok(None)
    }
}

/// Algorithm X's recursion: cover `remaining` using unused rows.
fn cover(
    rows: &[u64],
    remaining: u64,
    covered: u64,
    used: &mut Vec<bool>,
    chosen: &mut Vec<usize>,
) -> bool {
    if covered == remaining {
        return true;
    }
    // Branch on the uncovered column with the fewest candidate rows: the
    // heuristic that makes the search narrow rather than merely correct.
    let mut best_column = usize::MAX;
    let mut best_count = usize::MAX;
    for j in 0..64 {
        let bit = 1u64 << j;
        if bit > remaining {
            break;
        }
        if remaining & bit == 0 || covered & bit != 0 {
            continue;
        }
        let count = rows
            .iter()
            .enumerate()
            .filter(|(i, &r)| !used[*i] && r & bit != 0 && r & covered == 0)
            .count();
        if count < best_count {
            best_count = count;
            best_column = j;
        }
        if count == 0 {
            // A column no remaining row can cover: this branch is dead.
            return false;
        }
    }
    if best_column == usize::MAX {
        return covered == remaining;
    }

    let bit = 1u64 << best_column;
    for i in 0..rows.len() {
        if used[i] || rows[i] & bit == 0 || rows[i] & covered != 0 {
            continue;
        }
        used[i] = true;
        chosen.push(i);
        if cover(rows, remaining, covered | rows[i], used, chosen) {
            return true;
        }
        chosen.pop();
        used[i] = false;
    }
    false
}

/// Solves a Sudoku grid, with zero marking a blank.
///
/// Bitmask backtracking with the same fewest-options-first heuristic as
/// [`exact_cover_dlx`]: fill the cell with the fewest legal digits, which
/// collapses most puzzles without any search at all.
///
/// Returns `None` if the puzzle has no solution.
#[must_use]
pub fn sudoku_solve(grid: &[[u8; 9]; 9]) -> Option<[[u8; 9]; 9]> {
    let mut cells = *grid;
    // Row, column and box occupancy as nine-bit masks.
    let (mut rows, mut cols, mut boxes) = ([0u16; 9], [0u16; 9], [0u16; 9]);
    for r in 0..9 {
        for c in 0..9 {
            let v = cells[r][c];
            if v == 0 {
                continue;
            }
            if !(1..=9).contains(&v) {
                return None;
            }
            let bit = 1u16 << (v - 1);
            let b = (r / 3) * 3 + c / 3;
            if rows[r] & bit != 0 || cols[c] & bit != 0 || boxes[b] & bit != 0 {
                return None;
            }
            rows[r] |= bit;
            cols[c] |= bit;
            boxes[b] |= bit;
        }
    }
    if fill(&mut cells, &mut rows, &mut cols, &mut boxes) {
        Some(cells)
    } else {
        None
    }
}

/// Backtracking step for [`sudoku_solve`].
fn fill(
    cells: &mut [[u8; 9]; 9],
    rows: &mut [u16; 9],
    cols: &mut [u16; 9],
    boxes: &mut [u16; 9],
) -> bool {
    let mut target: Option<(usize, usize, u16, u32)> = None;
    for r in 0..9 {
        for c in 0..9 {
            if cells[r][c] != 0 {
                continue;
            }
            let b = (r / 3) * 3 + c / 3;
            let available = !(rows[r] | cols[c] | boxes[b]) & 0x1FF;
            let count = available.count_ones();
            if count == 0 {
                return false;
            }
            if target.is_none_or(|(_, _, _, best)| count < best) {
                target = Some((r, c, available, count));
            }
        }
    }
    let Some((r, c, available, _)) = target else { return true };

    let b = (r / 3) * 3 + c / 3;
    let mut options = available;
    while options != 0 {
        let bit = options.isolate_lowest_one();
        options ^= bit;
        let digit = bit.trailing_zeros() as u8 + 1;
        cells[r][c] = digit;
        rows[r] |= bit;
        cols[c] |= bit;
        boxes[b] |= bit;
        if fill(cells, rows, cols, boxes) {
            return true;
        }
        cells[r][c] = 0;
        rows[r] ^= bit;
        cols[c] ^= bit;
        boxes[b] ^= bit;
    }
    false
}

/// Every placement of `n` non-attacking queens, each as the column of the
/// queen in each row.
///
/// # Panics
/// Panics if `n` exceeds 12, where the count runs into the hundreds of
/// thousands and the list stops being a sensible return value.
#[must_use]
pub fn n_queens(n: usize) -> Vec<Vec<usize>> {
    assert!(n <= 12, "n_queens is for boards up to twelve squares wide");
    let mut solutions = Vec::new();
    let mut placement = Vec::with_capacity(n);
    queens(n, 0, 0, 0, &mut placement, &mut solutions, false);
    solutions
}

/// How many placements of `n` non-attacking queens exist.
///
/// The sequence begins 1, 0, 0, 2, 10, 4, 40, 92 for boards one to eight wide
/// -- there is no solution on a three-square board, and a six-square board has
/// fewer than a five-square one, which is the usual surprise.
///
/// # Panics
/// Panics if `n` exceeds 14.
#[must_use]
pub fn n_queens_count(n: usize) -> u64 {
    assert!(n <= 14, "n_queens_count is for boards up to fourteen squares wide");
    let mut solutions = Vec::new();
    let mut placement = Vec::with_capacity(n);
    queens(n, 0, 0, 0, &mut placement, &mut solutions, true) as u64
}

/// Bitmask backtracking over the columns and the two diagonals.
fn queens(
    n: usize,
    cols: u32,
    left: u32,
    right: u32,
    placement: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
    count_only: bool,
) -> usize {
    if placement.len() == n {
        if !count_only {
            out.push(placement.clone());
        }
        return 1;
    }
    let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
    // A queen attacks along its column and both diagonals; shifting the
    // diagonal masks by one each row is what advances them.
    let mut available = !(cols | left | right) & mask;
    let mut found = 0usize;
    while available != 0 {
        let bit = available.isolate_lowest_one();
        available ^= bit;
        placement.push(bit.trailing_zeros() as usize);
        found += queens(
            n,
            cols | bit,
            (left | bit) << 1,
            (right | bit) >> 1,
            placement,
            out,
            count_only,
        );
        placement.pop();
    }
    found
}

/// Arc consistency by AC-3: prunes values that cannot participate in any
/// solution.
///
/// `domains[i]` is a bitmask of the values variable `i` may take, and
/// `constraints` lists pairs `(i, j)` that must differ. Repeatedly removes any
/// value in one domain with no support in a neighbour's, until nothing
/// changes.
///
/// Returns the reduced domains, or `None` if some domain empties -- which
/// proves the constraints unsatisfiable without any search. AC-3 never removes
/// a value that appears in a solution, so the reduced domains are a sound
/// simplification rather than a heuristic.
#[must_use]
pub fn constraint_propagation_ac3(
    domains: &[u64],
    constraints: &[(usize, usize)],
) -> Option<Vec<u64>> {
    let mut d = domains.to_vec();
    let n = d.len();
    if constraints.iter().any(|&(a, b)| a >= n || b >= n) {
        return None;
    }
    // The queue holds directed arcs; an arc is re-queued when the domain at
    // its far end shrinks, since that may remove support elsewhere.
    let mut queue: Vec<(usize, usize)> = Vec::new();
    for &(a, b) in constraints {
        queue.push((a, b));
        queue.push((b, a));
    }

    while let Some((a, b)) = queue.pop() {
        let mut revised = false;
        let mut values = d[a];
        while values != 0 {
            let bit = values.isolate_lowest_one();
            values ^= bit;
            // With a not-equal constraint the only unsupported case is a
            // neighbour pinned to this very value.
            if d[b] == bit {
                d[a] &= !bit;
                revised = true;
            }
        }
        if d[a] == 0 {
            return None;
        }
        if revised {
            for &(x, y) in constraints {
                if y == a && x != b {
                    queue.push((x, y));
                }
                if x == a && y != b {
                    queue.push((y, x));
                }
            }
        }
    }
    Some(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix::Matrix;
    use crate::monte_carlo::Rng;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    /// Every 0/1 knapsack answer by exhaustive enumeration.
    fn knapsack_brute(values: &[u64], weights: &[u64], capacity: u64) -> u64 {
        let n = values.len();
        let mut best = 0u64;
        for mask in 0u32..(1u32 << n) {
            let (mut w, mut v) = (0u64, 0u64);
            for i in 0..n {
                if mask & (1 << i) != 0 {
                    w += weights[i];
                    v += values[i];
                }
            }
            if w <= capacity && v > best {
                best = v;
            }
        }
        best
    }

    // -----------------------------------------------------------------
    // Branch and bound
    // -----------------------------------------------------------------

    #[test]
    fn branch_and_bound_matches_exhaustive_integer_search() {
        // Small integer programs, solved twice: once by branch and bound over
        // the relaxation and once by trying every lattice point in the box.
        let mut rng = Rng::new(0xB4B0_0001);
        let mut compared = 0usize;
        for _ in 0..120 {
            let n = 2 + pick(&mut rng, 2);
            let m = 1 + pick(&mut rng, 3);
            let mut a = Matrix::zeros(m, n);
            for i in 0..m {
                for j in 0..n {
                    a.set(i, j, (rng.next_f64() * 4.0).round() + 1.0);
                }
            }
            let b: Vec<f64> = (0..m).map(|_| (rng.next_f64() * 15.0).round() + 3.0).collect();
            let c: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 8.0).round() + 1.0).collect();
            let mut p = LpProblem::new(c.clone(), a.clone(), b.clone(), true).unwrap();
            // Bound every variable so the exhaustive search is finite.
            for j in 0..n {
                p.bounds[j] = (0.0, 8.0);
            }
            let integer_vars: Vec<usize> = (0..n).collect();

            let Some((x, value)) = branch_and_bound(&p, &integer_vars, 100_000).unwrap() else {
                continue;
            };
            compared += 1;
            assert!(p.is_feasible(&x, 1e-6), "branch and bound returned {x:?}, not feasible");
            assert!(
                x.iter().all(|v| (v - v.round()).abs() < 1e-6),
                "a variable came back fractional: {x:?}"
            );

            // Exhaustive: every point of the integer box.
            let mut best = f64::NEG_INFINITY;
            let mut counter = vec![0usize; n];
            loop {
                let point: Vec<f64> = counter.iter().map(|&k| k as f64).collect();
                if p.is_feasible(&point, 1e-9) {
                    best = best.max(p.objective_at(&point));
                }
                let mut k = 0usize;
                while k < n {
                    counter[k] += 1;
                    if counter[k] <= 8 {
                        break;
                    }
                    counter[k] = 0;
                    k += 1;
                }
                if k == n {
                    break;
                }
            }
            assert!(
                (value - best).abs() < 1e-6,
                "branch and bound gave {value}, exhaustive search {best}"
            );

            // The relaxation bounds the integer optimum from above.
            if let Some(relaxed) = simplex(&p).unwrap().objective() {
                assert!(
                    value <= relaxed + 1e-6,
                    "the integer optimum {value} beat its own relaxation {relaxed}"
                );
            }
        }
        assert!(compared > 80, "only {compared} of 120 programs were comparable");
    }

    #[test]
    fn branch_and_bound_reports_an_integer_infeasibility() {
        // 2x = 1 has a rational solution and no integer one.
        let p = LpProblem {
            c: vec![1.0],
            a: Matrix::from_rows(&[&[2.0]]).unwrap(),
            b: vec![1.0],
            constraint_types: vec![Cmp::Eq],
            bounds: vec![(0.0, 10.0)],
            maximize: true,
        };
        assert!(simplex(&p).unwrap().objective().is_some(), "the relaxation should be feasible");
        assert_eq!(branch_and_bound(&p, &[0], 10_000).unwrap(), None);
        assert!(branch_and_bound(&p, &[9], 10).is_err());
    }

    #[test]
    fn gomory_cuts_never_remove_an_integer_point() {
        // A cut is valid only if every integer-feasible point survives it.
        let mut rng = Rng::new(0x0060_0001);
        for _ in 0..40 {
            let n = 2usize;
            let m = 2usize;
            let mut a = Matrix::zeros(m, n);
            for i in 0..m {
                for j in 0..n {
                    a.set(i, j, (rng.next_f64() * 4.0).round() + 1.0);
                }
            }
            let b: Vec<f64> = (0..m).map(|_| (rng.next_f64() * 12.0).round() + 4.0).collect();
            let c: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 6.0).round() + 1.0).collect();
            let mut p = LpProblem::new(c, a, b, true).unwrap();
            for j in 0..n {
                p.bounds[j] = (0.0, 10.0);
            }

            let Some((integer_point, integer_best)) =
                branch_and_bound(&p, &[0, 1], 50_000).unwrap()
            else {
                continue;
            };
            let cut = gomory_cuts(&p, &[0, 1], 6).unwrap();

            // Validity: every integer-feasible point still satisfies every
            // added row. This is the property that makes it a cut rather than
            // a branch, and the one that was wrong before.
            for a in 0..=10u32 {
                for b in 0..=10u32 {
                    let point = [f64::from(a), f64::from(b)];
                    if p.is_feasible(&point, 1e-9) {
                        assert!(
                            cut.is_feasible(&point, 1e-6),
                            "the cuts removed the integer point {point:?}"
                        );
                    }
                }
            }
            assert!(
                cut.is_feasible(&integer_point, 1e-6),
                "the cuts removed the integer optimum {integer_point:?}"
            );

            // Tightening: the relaxation is no looser than before, and still
            // bounds the integer optimum.
            let before = simplex(&p).unwrap().objective().unwrap_or(f64::INFINITY);
            if let Some(after) = simplex(&cut).unwrap().objective() {
                assert!(after <= before + 1e-7, "the cut loosened the relaxation");
                assert!(
                    after >= integer_best - 1e-7,
                    "the tightened bound {after} fell below the integer optimum {integer_best}"
                );
            }
        }
    }

    #[test]
    fn gomory_cuts_refuse_the_problems_the_rounding_argument_does_not_cover() {
        let p = LpProblem::new(
            vec![1.0, 1.0],
            Matrix::from_rows(&[&[2.0, 3.0]]).unwrap(),
            vec![7.0],
            true,
        )
        .unwrap();
        // The rounding step needs every variable integral and non-negative.
        assert!(gomory_cuts(&p, &[0], 3).is_err(), "a continuous variable should be refused");
        let mut negative = p.clone();
        negative.bounds[0] = (f64::NEG_INFINITY, f64::INFINITY);
        assert!(
            gomory_cuts(&negative, &[0, 1], 3).is_err(),
            "a free variable should be refused"
        );
        // An already-integral relaxation optimum gives nothing to cut.
        let integral = LpProblem::new(
            vec![1.0],
            Matrix::from_rows(&[&[1.0]]).unwrap(),
            vec![4.0],
            true,
        )
        .unwrap();
        let same = gomory_cuts(&integral, &[0], 3).unwrap();
        assert_eq!(same.m(), integral.m(), "a cut was added where none was needed");
    }

    // -----------------------------------------------------------------
    // Knapsack
    // -----------------------------------------------------------------

    #[test]
    fn the_knapsack_table_and_the_search_tree_agree_with_brute_force() {
        // Three independent methods on the same instances: a table, a search
        // tree with a relaxation bound, and enumeration.
        let mut rng = Rng::new(0xC0FF_0001);
        for _ in 0..200 {
            let n = 1 + pick(&mut rng, 12);
            let values: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 40)).collect();
            let weights: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 20)).collect();
            let capacity = 5 + (rng.next_u64() % 60);

            let (dp_value, chosen) = knapsack_01(&values, &weights, capacity);
            let (bb_value, bb_chosen) = knapsack_branch_bound(&values, &weights, capacity);
            let brute = knapsack_brute(&values, &weights, capacity);

            assert_eq!(dp_value, brute, "the table disagreed with brute force");
            assert_eq!(bb_value, brute, "branch and bound disagreed with brute force");

            // Each reported selection actually achieves its value and fits.
            for (label, picks) in [("table", &chosen), ("branch and bound", &bb_chosen)] {
                let w: u64 =
                    picks.iter().enumerate().filter(|(_, &t)| t).map(|(i, _)| weights[i]).sum();
                let v: u64 =
                    picks.iter().enumerate().filter(|(_, &t)| t).map(|(i, _)| values[i]).sum();
                assert!(w <= capacity, "{label} overfilled the sack: {w} > {capacity}");
                assert_eq!(v, brute, "{label}'s selection is worth {v}, not {brute}");
            }
        }
    }

    #[test]
    fn the_knapsack_variants_order_themselves_the_way_the_rules_imply() {
        let values = [10u64, 30, 25, 50];
        let weights = [5u64, 10, 6, 20];
        let capacity = 30u64;

        let (once, _) = knapsack_01(&values, &weights, capacity);
        let (limited, counts) =
            knapsack_bounded(&values, &weights, &[1, 1, 1, 1], capacity);
        let (unlimited, repeats) = knapsack_unbounded(&values, &weights, capacity);

        // A bounded knapsack with every limit at one is exactly the 0/1 case.
        assert_eq!(limited, once, "bounded at one disagreed with 0/1");
        assert!(counts.iter().all(|&c| c <= 1), "a limit of one was exceeded: {counts:?}");
        // Allowing repeats can only help.
        assert!(unlimited >= once, "unbounded {unlimited} fell below 0/1 {once}");
        // And the reported repeats fit and are worth what was claimed.
        let w: u64 = repeats.iter().zip(&weights).map(|(&c, &w)| c * w).sum();
        let v: u64 = repeats.iter().zip(&values).map(|(&c, &v)| c * v).sum();
        assert!(w <= capacity, "the unbounded pack overfilled: {w}");
        assert_eq!(v, unlimited);

        // Raising a limit can only help, never hurt.
        let mut previous = 0u64;
        for limit in 1..=5u64 {
            let (value, _) = knapsack_bounded(&values, &weights, &[limit; 4], capacity);
            assert!(value >= previous, "raising the limit to {limit} reduced the value");
            previous = value;
        }
        assert_eq!(previous, unlimited, "a high enough limit should reach the unbounded answer");

        // Several bins hold at least as much as one of the same size.
        let (multi, placement) = knapsack_multiple(&values, &weights, &[15, 15]);
        assert!(multi > 0);
        for (i, spot) in placement.iter().enumerate() {
            if let Some(bin) = spot {
                assert!(*bin < 2, "item {i} went into bin {bin}");
            }
        }
        for bin in 0..2 {
            let load: u64 = placement
                .iter()
                .enumerate()
                .filter(|(_, s)| **s == Some(bin))
                .map(|(i, _)| weights[i])
                .sum();
            assert!(load <= 15, "bin {bin} holds {load}");
        }
    }

    // -----------------------------------------------------------------
    // Subset sum and partition
    // -----------------------------------------------------------------

    #[test]
    fn subset_sum_finds_a_subset_and_counts_them_all() {
        let mut rng = Rng::new(0x5085_0001);
        for _ in 0..150 {
            let n = 1 + pick(&mut rng, 12);
            let xs: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 25)).collect();
            let target = rng.next_u64() % 60;

            // Brute force: every subset.
            let mut brute_count = 0u64;
            let mut brute_found = false;
            for mask in 0u32..(1u32 << n) {
                let s: u64 = (0..n).filter(|i| mask & (1 << i) != 0).map(|i| xs[i]).sum();
                if s == target {
                    brute_count += 1;
                    brute_found = true;
                }
            }

            match subset_sum(&xs, target) {
                Some(indices) => {
                    assert!(brute_found, "a subset was found where none exists");
                    let s: u64 = indices.iter().map(|&i| xs[i]).sum();
                    assert_eq!(s, target, "the reported subset sums to {s}, not {target}");
                    // Indices are distinct and in range.
                    let mut sorted = indices.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    assert_eq!(sorted.len(), indices.len(), "repeated index in {indices:?}");
                }
                None => assert!(!brute_found, "a subset exists but none was found"),
            }
            assert_eq!(
                subset_sum_count(&xs, target).to_string(),
                brute_count.to_string(),
                "the count disagreed with brute force"
            );
        }
    }

    #[test]
    fn the_partition_split_is_as_even_as_any_split_can_be() {
        let mut rng = Rng::new(0x9A27_0001);
        for _ in 0..120 {
            let n = 1 + pick(&mut rng, 12);
            let xs: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 30)).collect();
            let total: u64 = xs.iter().sum();

            let (difference, flags) = partition_min_diff(&xs);
            let left: u64 =
                flags.iter().enumerate().filter(|(_, &f)| f).map(|(i, _)| xs[i]).sum();
            let right = total - left;
            assert_eq!(
                left.abs_diff(right),
                difference,
                "the reported flags give a gap of {}, not {difference}",
                left.abs_diff(right)
            );

            // No split does better.
            let mut best = u64::MAX;
            for mask in 0u32..(1u32 << n) {
                let s: u64 = (0..n).filter(|i| mask & (1 << i) != 0).map(|i| xs[i]).sum();
                best = best.min(s.abs_diff(total - s));
            }
            assert_eq!(difference, best, "a more even split exists");
        }
    }

    // -----------------------------------------------------------------
    // Packing and covering
    // -----------------------------------------------------------------

    #[test]
    fn first_fit_decreasing_packs_validly_and_within_its_proven_ratio() {
        let mut rng = Rng::new(0x00B1_0001);
        for _ in 0..80 {
            let n = 1 + pick(&mut rng, 10);
            let sizes: Vec<f64> = (0..n).map(|_| rng.next_f64() * 0.7 + 0.05).collect();
            let capacity = 1.0f64;

            let bins = bin_packing_ffd(&sizes, capacity);
            // Every item is placed exactly once and no bin overflows.
            let mut seen = vec![0usize; n];
            for bin in &bins {
                let load: f64 = bin.iter().map(|&i| sizes[i]).sum();
                assert!(load <= capacity + 1e-9, "a bin holds {load}");
                for &i in bin {
                    seen[i] += 1;
                }
            }
            assert!(seen.iter().all(|&k| k == 1), "an item was lost or duplicated: {seen:?}");

            let lower = bin_packing_lower_bound(&sizes, capacity);
            assert!(bins.len() >= lower, "{} bins is below the bound {lower}", bins.len());

            let exact = bin_packing_exact_small(&sizes, capacity);
            assert!(exact.len() >= lower, "the exact packing beat the lower bound");
            assert!(exact.len() <= bins.len(), "the exact packing used more bins than greedy");
            // The first-fit-decreasing guarantee: at most 11/9 OPT + 6/9.
            let guarantee = 11.0 / 9.0 * exact.len() as f64 + 6.0 / 9.0;
            assert!(
                bins.len() as f64 <= guarantee + 1e-9,
                "{} bins exceeds the guarantee {guarantee} against an optimum of {}",
                bins.len(),
                exact.len()
            );
        }
        // The bound is not always attainable: three items of size 0.4 total
        // 1.2, so the bound says two, and two is indeed enough here.
        assert_eq!(bin_packing_lower_bound(&[0.4, 0.4, 0.4], 1.0), 2);
        assert_eq!(bin_packing_exact_small(&[0.4, 0.4, 0.4], 1.0).len(), 2);
        // Two items of 0.4 share a bin, so four of them still need only two.
        assert_eq!(bin_packing_exact_small(&[0.4; 4], 1.0).len(), 2);
        // Three items of 0.6 total 1.8, so the volume bound says two, but no
        // two of them share a bin and three are needed -- the bound is a
        // bound, not an answer.
        assert_eq!(bin_packing_lower_bound(&[0.6; 3], 1.0), 2);
        assert_eq!(bin_packing_exact_small(&[0.6; 3], 1.0).len(), 3);
        assert!(bin_packing_ffd(&[], 1.0).is_empty());
    }

    #[test]
    fn greedy_set_cover_covers_everything_within_its_harmonic_ratio() {
        let mut rng = Rng::new(0x5E7C_0001);
        for _ in 0..80 {
            let universe = 3 + pick(&mut rng, 8);
            let count = 2 + pick(&mut rng, 8);
            let sets: Vec<Vec<usize>> = (0..count)
                .map(|_| {
                    (0..universe).filter(|_| rng.next_f64() < 0.45).collect::<Vec<usize>>()
                })
                .collect();

            let greedy = set_cover_greedy(universe, &sets);
            let exact = set_cover_exact_small(universe, &sets);
            match (&greedy, &exact) {
                (Some(g), Some(e)) => {
                    // The greedy choice really does cover the universe.
                    let mut covered = vec![false; universe];
                    for &i in g {
                        for &v in &sets[i] {
                            if v < universe {
                                covered[v] = true;
                            }
                        }
                    }
                    assert!(covered.iter().all(|&c| c), "the greedy cover misses an element");
                    // Greedy is within H_n of optimal.
                    let harmonic: f64 = (1..=universe).map(|k| 1.0 / k as f64).sum();
                    assert!(
                        g.len() as f64 <= harmonic * e.len() as f64 + 1e-9,
                        "{} sets exceeds H_n * {} = {}",
                        g.len(),
                        e.len(),
                        harmonic * e.len() as f64
                    );
                    assert!(g.len() >= e.len(), "greedy beat the exact minimum");
                }
                (None, None) => {}
                _ => panic!("greedy and exact disagreed on whether a cover exists"),
            }
        }
        // A universe no set reaches has no cover at all.
        assert_eq!(set_cover_greedy(3, &[vec![0], vec![1]]), None);
        assert_eq!(set_cover_exact_small(3, &[vec![0], vec![1]]), None);
    }

    #[test]
    fn facility_location_opens_a_set_that_serves_every_client() {
        let serve = Matrix::from_rows(&[
            &[1.0, 9.0, 9.0],
            &[9.0, 1.0, 9.0],
            &[2.0, 2.0, 2.0],
        ])
        .unwrap();
        // One central facility is dear to open but cheap to serve from; the
        // two specialists are the reverse.
        let (total, open) = facility_location_greedy(&[1.0, 1.0, 3.0], &serve);
        assert!(open.iter().any(|&o| o), "no facility was opened");
        assert!(total.is_finite() && total > 0.0);

        // The reported total matches serving every client from its cheapest
        // open facility, plus the opening costs.
        let opening: f64 =
            open.iter().enumerate().filter(|(_, &o)| o).map(|(i, _)| [1.0, 1.0, 3.0][i]).sum();
        let serving: f64 = (0..3)
            .map(|j| {
                (0..3)
                    .filter(|&i| open[i])
                    .map(|i| serve.get(i, j))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum();
        assert!(
            (total - opening - serving).abs() < 1e-9,
            "reported {total} against {opening} + {serving}"
        );
        // Greedy must do at least as well as the best single facility, which
        // is the central one at 3 to open and 2 per client.
        let open_costs = [1.0f64, 1.0, 3.0];
        let best_single = (0..3)
            .map(|i| open_costs[i] + (0..3).map(|j| serve.get(i, j)).sum::<f64>())
            .fold(f64::INFINITY, f64::min);
        assert!((best_single - 9.0).abs() < 1e-9, "the best single facility costs {best_single}");
        assert!(total <= best_single + 1e-9, "greedy {total} lost to a single facility");
    }

    #[test]
    fn column_generation_bounds_the_cutting_stock_problem() {
        // Stock of length 100, pieces of 45, 36 and 31, demanded 97, 610, 395.
        // The relaxation's value lower-bounds any integer packing, and must
        // beat the trivial bound of total length over stock length.
        let value =
            cutting_stock_column_generation(&[97, 610, 395], &[45, 36, 31], 100, 40).unwrap();
        let total_length = 97 * 45 + 610 * 36 + 395 * 31;
        let trivial = total_length as f64 / 100.0;
        assert!(value >= trivial - 1e-6, "the relaxation {value} fell below the bound {trivial}");
        assert!(value.is_finite() && value > 0.0);
        // No packing can do better than the relaxation, and the naive
        // one-piece-per-pattern answer is worse.
        let naive = 97.0 / 2.0 + 610.0 / 2.0 + 395.0 / 3.0;
        assert!(value <= naive + 1e-6, "column generation {value} lost to the naive {naive}");

        assert!(cutting_stock_column_generation(&[1], &[1, 2], 10, 5).is_err());
        assert!(cutting_stock_column_generation(&[1], &[200], 100, 5).is_err());
        assert!(cutting_stock_column_generation(&[], &[], 100, 5).is_err());
    }

    // -----------------------------------------------------------------
    // Dynamic programming classics
    // -----------------------------------------------------------------

    #[test]
    fn coin_change_is_minimal_where_greedy_is_not() {
        // The canonical counterexample: with 1, 3, 4 and a target of six,
        // greedy takes 4 + 1 + 1 while two threes suffice.
        let counts = coin_change_min(&[1, 3, 4], 6).unwrap();
        assert_eq!(counts.iter().sum::<u64>(), 2, "expected two coins, got {counts:?}");
        let paid: u64 = counts.iter().zip([1u64, 3, 4]).map(|(&c, v)| c * v).sum();
        assert_eq!(paid, 6);

        let mut rng = Rng::new(0x0C01_0001);
        for _ in 0..120 {
            let k = 1 + pick(&mut rng, 4);
            let coins: Vec<u64> = (0..k).map(|_| 1 + (rng.next_u64() % 12)).collect();
            let amount = rng.next_u64() % 40;
            match coin_change_min(&coins, amount) {
                Some(counts) => {
                    let paid: u64 = counts.iter().zip(&coins).map(|(&c, &v)| c * v).sum();
                    assert_eq!(paid, amount, "the coins pay {paid}, not {amount}");
                    // Minimality against an independent table.
                    let mut best = vec![u64::MAX; amount as usize + 1];
                    best[0] = 0;
                    for s in 1..=amount as usize {
                        for &c in &coins {
                            let c = c as usize;
                            if c <= s && best[s - c] != u64::MAX {
                                best[s] = best[s].min(best[s - c] + 1);
                            }
                        }
                    }
                    assert_eq!(counts.iter().sum::<u64>(), best[amount as usize]);
                }
                None => {
                    // Nothing reaches the amount; check by the same table.
                    let mut reachable = vec![false; amount as usize + 1];
                    reachable[0] = true;
                    for s in 1..=amount as usize {
                        reachable[s] = coins
                            .iter()
                            .any(|&c| c as usize <= s && reachable[s - c as usize]);
                    }
                    assert!(!reachable[amount as usize], "a combination exists");
                }
            }
        }
        // Combinations, not permutations: 1 + 2 and 2 + 1 are one way.
        assert_eq!(coin_change_count(&[1, 2], 3).to_string(), "2");
        assert_eq!(coin_change_count(&[1, 2, 5], 11).to_string(), "11");
        assert_eq!(coin_change_count(&[2], 3).to_string(), "0");
    }

    #[test]
    fn the_longest_increasing_subsequence_is_increasing_and_longest() {
        let mut rng = Rng::new(0x0011_0001);
        for _ in 0..150 {
            let n = pick(&mut rng, 40);
            let x: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 20.0).round()).collect();
            let indices = longest_increasing_subsequence(&x);

            if n == 0 {
                assert!(indices.is_empty());
                continue;
            }
            // Indices ascend and so do the values they name.
            assert!(indices.windows(2).all(|w| w[0] < w[1]), "indices out of order");
            assert!(
                indices.windows(2).all(|w| x[w[0]] < x[w[1]]),
                "the subsequence is not increasing"
            );

            // Length against a quadratic table.
            let mut best = vec![1usize; n];
            for i in 1..n {
                for j in 0..i {
                    if x[j] < x[i] && best[j] + 1 > best[i] {
                        best[i] = best[j] + 1;
                    }
                }
            }
            assert_eq!(indices.len(), *best.iter().max().unwrap_or(&0));
        }
    }

    #[test]
    fn edit_distance_is_a_metric_and_its_operations_reproduce_the_target() {
        let mut rng = Rng::new(0x00ED_0001);
        let word = |rng: &mut Rng, n: usize| -> Vec<u8> {
            (0..n).map(|_| b'a' + (rng.next_u64() % 4) as u8).collect()
        };
        for _ in 0..150 {
            let (la, lb, lc) = (pick(&mut rng, 9), pick(&mut rng, 9), pick(&mut rng, 9));
            let a = word(&mut rng, la);
            let b = word(&mut rng, lb);
            let c = word(&mut rng, lc);
            let d = edit_distance(&a, &b);

            assert_eq!(d, edit_distance(&b, &a), "the distance is not symmetric");
            assert_eq!(edit_distance(&a, &a), 0, "a sequence differs from itself");
            if d == 0 {
                assert_eq!(a, b, "distinct sequences at distance zero");
            }
            assert!(
                d <= edit_distance(&a, &c) + edit_distance(&c, &b),
                "the triangle inequality failed"
            );
            assert!(d <= a.len().max(b.len()), "the distance exceeds the longer length");
            assert!(d >= a.len().abs_diff(b.len()), "the distance is below the length gap");

            // Replaying the operations turns a into b, and their non-Keep
            // count is exactly the distance.
            let ops = edit_distance_ops(&a, &b);
            let mut rebuilt = Vec::new();
            for op in &ops {
                match *op {
                    EditOp::Keep(i, _) => rebuilt.push(a[i]),
                    EditOp::Substitute(_, j) | EditOp::Insert(j) => rebuilt.push(b[j]),
                    EditOp::Delete(_) => {}
                }
            }
            assert_eq!(rebuilt, b, "replaying the edits did not reproduce b");
            let cost = ops.iter().filter(|o| !matches!(o, EditOp::Keep(_, _))).count();
            assert_eq!(cost, d, "the operation list costs {cost}, not {d}");
        }
    }

    #[test]
    fn the_common_subsequence_is_common_and_longest() {
        let mut rng = Rng::new(0x01C5_0001);
        for _ in 0..150 {
            let a: Vec<u8> =
                (0..pick(&mut rng, 12)).map(|_| b'a' + (rng.next_u64() % 4) as u8).collect();
            let b: Vec<u8> =
                (0..pick(&mut rng, 12)).map(|_| b'a' + (rng.next_u64() % 4) as u8).collect();
            let lcs = longest_common_subsequence(&a, &b);

            // It really is a subsequence of both.
            let is_sub = |s: &[u8], whole: &[u8]| -> bool {
                let mut it = whole.iter();
                s.iter().all(|c| it.any(|w| w == c))
            };
            assert!(is_sub(&lcs, &a), "{lcs:?} is not a subsequence of {a:?}");
            assert!(is_sub(&lcs, &b), "{lcs:?} is not a subsequence of {b:?}");

            // Length against a table.
            let (n, m) = (a.len(), b.len());
            let mut table = vec![vec![0usize; m + 1]; n + 1];
            for i in 1..=n {
                for j in 1..=m {
                    table[i][j] = if a[i - 1] == b[j - 1] {
                        table[i - 1][j - 1] + 1
                    } else {
                        table[i - 1][j].max(table[i][j - 1])
                    };
                }
            }
            assert_eq!(lcs.len(), table[n][m]);
        }
    }

    #[test]
    fn the_matrix_chain_order_beats_every_parenthesisation() {
        // The textbook chain: 40x20, 20x30, 30x10, 10x30 costs 26000.
        let (cost, order) = matrix_chain_order(&[40, 20, 30, 10, 30]);
        assert_eq!(cost, 26_000, "got {cost} with {order}");
        assert!(order.starts_with('('), "the rendering is not parenthesised: {order}");

        // The order matters enormously: 1x100, 100x1, 1x100.
        let (cheap, _) = matrix_chain_order(&[1, 100, 1, 100]);
        assert_eq!(cheap, 200, "the cheap order costs {cheap}");

        // Against exhaustive splitting on small chains.
        fn brute(dims: &[usize], i: usize, j: usize) -> u64 {
            if i == j {
                return 0;
            }
            (i..j)
                .map(|k| {
                    brute(dims, i, k)
                        + brute(dims, k + 1, j)
                        + (dims[i] * dims[k + 1] * dims[j + 1]) as u64
                })
                .min()
                .unwrap_or(0)
        }
        let mut rng = Rng::new(0x003A_0001);
        for _ in 0..60 {
            let k = 2 + pick(&mut rng, 5);
            let dims: Vec<usize> = (0..=k).map(|_| 1 + pick(&mut rng, 30)).collect();
            let (table, _) = matrix_chain_order(&dims);
            assert_eq!(table, brute(&dims, 0, k - 1), "the table lost to brute force");
        }
        // One matrix costs nothing.
        assert_eq!(matrix_chain_order(&[3, 4]).0, 0);
    }

    #[test]
    fn rod_cutting_returns_pieces_that_add_up_and_pay_out() {
        let prices = [1u64, 5, 8, 9, 10, 17, 17, 20];
        // The textbook answer for a rod of eight with these prices is 22.
        let (value, pieces) = rod_cutting(&prices, 8);
        assert_eq!(value, 22, "got {value} from {pieces:?}");
        assert_eq!(pieces.iter().sum::<usize>(), 8, "the pieces are {pieces:?}");
        let paid: u64 = pieces.iter().map(|&p| prices[p - 1]).sum();
        assert_eq!(paid, value);

        // Longer rods are worth at least as much.
        let mut previous = 0u64;
        for n in 0..=8 {
            let (v, p) = rod_cutting(&prices, n);
            assert!(v >= previous, "a longer rod was worth less at n = {n}");
            assert_eq!(p.iter().sum::<usize>(), n, "pieces {p:?} do not total {n}");
            previous = v;
        }
    }

    #[test]
    fn egg_drop_reproduces_its_known_values() {
        // The classic: two eggs and a hundred floors take fourteen drops.
        assert_eq!(egg_drop(2, 100), 14);
        // One egg has to be dropped from every floor in turn.
        assert_eq!(egg_drop(1, 37), 37);
        // Enough eggs and it is a binary search.
        assert_eq!(egg_drop(20, 1000), 10);
        assert_eq!(egg_drop(0, 5), 0);
        assert_eq!(egg_drop(3, 0), 0);
        // More eggs never need more drops; more floors never need fewer.
        for floors in [10usize, 50, 200] {
            let mut previous = u64::MAX;
            for eggs in 1..=8 {
                let d = egg_drop(eggs, floors);
                assert!(d <= previous, "an extra egg cost more drops at {eggs}");
                previous = d;
            }
        }
        for eggs in [1usize, 2, 4] {
            let mut previous = 0u64;
            for floors in [1usize, 10, 100, 500] {
                let d = egg_drop(eggs, floors);
                assert!(d >= previous, "more floors needed fewer drops");
                previous = d;
            }
        }
    }

    #[test]
    fn the_optimal_search_tree_beats_every_arrangement() {
        // Against brute force over every root choice, recursively.
        fn brute(freq: &[f64], i: usize, j: usize) -> f64 {
            if i > j {
                return 0.0;
            }
            let sum: f64 = freq[i..=j].iter().sum();
            (i..=j)
                .map(|r| {
                    let left = if r > i { brute(freq, i, r - 1) } else { 0.0 };
                    let right = if r < j { brute(freq, r + 1, j) } else { 0.0 };
                    left + right + sum
                })
                .fold(f64::INFINITY, f64::min)
        }
        let mut rng = Rng::new(0x0B57_0001);
        for _ in 0..40 {
            let n = 1 + pick(&mut rng, 6);
            let freq: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 10.0).round() + 1.0).collect();
            let table = optimal_bst(&freq);
            let exact = brute(&freq, 0, n - 1);
            assert!(
                (table - exact).abs() < 1e-9,
                "the table gave {table}, brute force {exact}"
            );
        }
        assert_eq!(optimal_bst(&[]), 0.0);
        // A single key is one comparison, weighted by its frequency.
        assert!((optimal_bst(&[0.7]) - 0.7).abs() < 1e-12);
        // A skewed distribution is cheaper than a flat one of the same total.
        let flat = optimal_bst(&[1.0, 1.0, 1.0, 1.0]);
        let skewed = optimal_bst(&[3.7, 0.1, 0.1, 0.1]);
        assert!(skewed < flat, "skewed {skewed} was not cheaper than flat {flat}");
    }

    #[test]
    fn the_trellis_path_is_the_cheapest_of_them_all() {
        let mut rng = Rng::new(0x1727_0001);
        for _ in 0..60 {
            let s = 2 + pick(&mut rng, 3);
            let t = 2 + pick(&mut rng, 4);
            let mut transition = Matrix::zeros(s, s);
            let mut emission = Matrix::zeros(s, t);
            for i in 0..s {
                for j in 0..s {
                    transition.set(i, j, (rng.next_f64() * 9.0).round());
                }
                for k in 0..t {
                    emission.set(i, k, (rng.next_f64() * 9.0).round());
                }
            }
            let path = viterbi_generic(&transition, &emission).unwrap();
            assert_eq!(path.len(), t);

            let cost = |p: &[usize]| -> f64 {
                let mut acc = emission.get(p[0], 0);
                for k in 1..t {
                    acc += transition.get(p[k - 1], p[k]) + emission.get(p[k], k);
                }
                acc
            };
            // Every path, enumerated.
            let mut best = f64::INFINITY;
            let mut counter = vec![0usize; t];
            loop {
                best = best.min(cost(&counter));
                let mut k = 0usize;
                while k < t {
                    counter[k] += 1;
                    if counter[k] < s {
                        break;
                    }
                    counter[k] = 0;
                    k += 1;
                }
                if k == t {
                    break;
                }
            }
            assert!(
                (cost(&path) - best).abs() < 1e-9,
                "the trellis path costs {}, the best is {best}",
                cost(&path)
            );
        }
        assert!(viterbi_generic(&Matrix::zeros(2, 3), &Matrix::zeros(2, 2)).is_err());
        assert!(viterbi_generic(&Matrix::zeros(2, 2), &Matrix::zeros(3, 2)).is_err());
    }

    // -----------------------------------------------------------------
    // Exact cover and constraint search
    // -----------------------------------------------------------------

    #[test]
    fn exact_cover_partitions_the_columns() {
        // Knuth's own example, which has the unique solution {0, 3, 4}.
        let matrix = vec![
            vec![true, false, false, true, false, false, true],
            vec![true, false, false, true, false, false, false],
            vec![false, false, false, true, true, false, true],
            vec![false, false, true, false, true, true, false],
            vec![false, true, true, false, false, true, true],
            vec![false, true, false, false, false, false, true],
        ];
        let chosen = exact_cover_dlx(&matrix).unwrap().expect("a cover exists");
        // Every column is covered exactly once, which is the definition.
        for col in 0..7 {
            let hits = chosen.iter().filter(|&&r| matrix[r][col]).count();
            assert_eq!(hits, 1, "column {col} is covered {hits} times by {chosen:?}");
        }

        // No cover: two rows both claiming the same single column.
        assert_eq!(exact_cover_dlx(&[vec![true, false], vec![true, false]]).unwrap(), None);
        // A cover of nothing is the empty selection.
        assert_eq!(exact_cover_dlx(&[]).unwrap(), Some(Vec::new()));
        assert!(exact_cover_dlx(&[vec![true], vec![true, false]]).is_err());
        assert!(exact_cover_dlx(&[vec![false; 65]]).is_err());
    }

    #[test]
    fn a_solved_sudoku_is_valid_and_keeps_its_clues() {
        let puzzle = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ];
        let solved = sudoku_solve(&puzzle).expect("this puzzle has a solution");

        for r in 0..9 {
            for c in 0..9 {
                assert!((1..=9).contains(&solved[r][c]), "cell ({r}, {c}) is {}", solved[r][c]);
                if puzzle[r][c] != 0 {
                    assert_eq!(solved[r][c], puzzle[r][c], "clue at ({r}, {c}) was changed");
                }
            }
        }
        // Each row, column and box holds all nine digits.
        for k in 0..9 {
            let row: Vec<u8> = (0..9).map(|c| solved[k][c]).collect();
            let col: Vec<u8> = (0..9).map(|r| solved[r][k]).collect();
            let boxed: Vec<u8> = (0..9)
                .map(|i| solved[(k / 3) * 3 + i / 3][(k % 3) * 3 + i % 3])
                .collect();
            for group in [row, col, boxed] {
                let mut sorted = group.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (1..=9).collect::<Vec<u8>>(), "a group repeats: {group:?}");
            }
        }

        // An already-contradictory grid is rejected without search.
        let mut broken = puzzle;
        broken[0][2] = 5;
        assert_eq!(sudoku_solve(&broken), None);
        let mut invalid = puzzle;
        invalid[0][2] = 10;
        assert_eq!(sudoku_solve(&invalid), None);
        // An empty grid has many solutions, and one is returned.
        assert!(sudoku_solve(&[[0u8; 9]; 9]).is_some());
    }

    #[test]
    fn n_queens_places_them_legally_and_counts_them_correctly() {
        // The sequence for boards one to ten wide.
        let known = [1u64, 0, 0, 2, 10, 4, 40, 92, 352, 724];
        for (n, &expected) in known.iter().enumerate() {
            assert_eq!(n_queens_count(n + 1), expected, "board {} has {expected}", n + 1);
        }
        // A six-square board has fewer solutions than a five-square one, which
        // is the standard surprise.
        assert!(known[5] < known[4]);

        for n in 1..=8usize {
            let solutions = n_queens(n);
            assert_eq!(solutions.len() as u64, n_queens_count(n));
            for placement in &solutions {
                assert_eq!(placement.len(), n);
                for i in 0..n {
                    assert!(placement[i] < n, "a queen left the board");
                    for j in i + 1..n {
                        assert_ne!(placement[i], placement[j], "two queens share a column");
                        assert_ne!(
                            placement[i].abs_diff(placement[j]),
                            j - i,
                            "two queens share a diagonal"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn arc_consistency_prunes_soundly_and_detects_contradictions() {
        // Three variables that must all differ, one pinned to a single value.
        let domains = [0b001u64, 0b011, 0b111];
        let constraints = [(0usize, 1usize), (1, 2), (0, 2)];
        let reduced = constraint_propagation_ac3(&domains, &constraints).unwrap();
        // Variable 0 is pinned, so 1 loses that value and is pinned in turn,
        // and 2 loses both.
        assert_eq!(reduced[0], 0b001);
        assert_eq!(reduced[1], 0b010, "variable 1 came out {:#b}", reduced[1]);
        assert_eq!(reduced[2], 0b100, "variable 2 came out {:#b}", reduced[2]);

        // Never removes a value that appears in a solution: check by
        // enumeration over the original domains.
        let mut rng = Rng::new(0x0AC3_0001);
        for _ in 0..200 {
            let n = 2 + pick(&mut rng, 3);
            let domains: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() % 15)).collect();
            let pairs: Vec<(usize, usize)> = (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .filter(|_| rng.next_f64() < 0.7)
                .collect();

            // Every assignment satisfying the constraints, under the original
            // domains.
            let mut solutions: Vec<Vec<u64>> = Vec::new();
            let mut assignment = vec![0u64; n];
            fn search(
                k: usize,
                domains: &[u64],
                pairs: &[(usize, usize)],
                assignment: &mut Vec<u64>,
                out: &mut Vec<Vec<u64>>,
            ) {
                if k == domains.len() {
                    out.push(assignment.clone());
                    return;
                }
                let mut values = domains[k];
                while values != 0 {
                    let bit = values.isolate_lowest_one();
                    values ^= bit;
                    assignment[k] = bit;
                    if pairs
                        .iter()
                        .all(|&(a, b)| a > k || b > k || assignment[a] != assignment[b])
                    {
                        search(k + 1, domains, pairs, assignment, out);
                    }
                }
                assignment[k] = 0;
            }
            search(0, &domains, &pairs, &mut assignment, &mut solutions);

            match constraint_propagation_ac3(&domains, &pairs) {
                Some(reduced) => {
                    // Soundness: every solution survives the pruning.
                    for solution in &solutions {
                        for (k, &v) in solution.iter().enumerate() {
                            assert!(
                                reduced[k] & v != 0,
                                "AC-3 pruned a value that appears in a solution"
                            );
                        }
                    }
                    // And it only ever removes.
                    for k in 0..n {
                        assert_eq!(reduced[k] & !domains[k], 0, "AC-3 added a value");
                    }
                }
                None => assert!(
                    solutions.is_empty(),
                    "AC-3 declared a contradiction where solutions exist"
                ),
            }
        }
        // An out-of-range constraint is refused rather than indexed.
        assert_eq!(constraint_propagation_ac3(&[1, 2], &[(0, 5)]), None);
    }
}
