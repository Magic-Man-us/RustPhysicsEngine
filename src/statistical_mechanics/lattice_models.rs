//! Lattice models: percolation, walks, growth, and avalanches.
//!
//! These are the systems where critical behaviour appears without any
//! Hamiltonian or temperature at all. Percolation has a sharp threshold and a
//! divergent cluster size, self-avoiding walks have a non-trivial exponent
//! that mean-field theory gets wrong, and a growing interface roughens with
//! exponents shared by systems that have nothing physically in common. That
//! last fact -- universality -- is what makes the subject more than a
//! collection of models: the exponents depend on dimension and symmetry, and
//! on essentially nothing else.
//!
//! Everything here is on a square lattice unless said otherwise, and the
//! random routines take the crate's deterministic generator so a run can be
//! repeated exactly.

use crate::discrete::disjoint_set::DisjointSet;
use crate::error::GeomError;
use crate::monte_carlo::Rng;

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

// ---------------------------------------------------------------------------
// Percolation
// ---------------------------------------------------------------------------

/// Site percolation on a square lattice: occupy each site with probability
/// `p` and report whether an occupied cluster spans top to bottom.
///
/// Returns the grid and whether it spans. The transition is sharp only in the
/// infinite lattice; on a finite one the spanning probability rises smoothly
/// through the threshold over a width that shrinks as the lattice grows,
/// which is finite-size scaling in its simplest visible form.
///
/// # Errors
/// Returns an error for a bad lattice size or a probability outside `[0, 1]`.
pub fn percolation_site(
    n: usize,
    p: f64,
    rng: &mut Rng,
) -> Result<(Vec<bool>, bool), GeomError> {
    if !(2..=2048).contains(&n) {
        return Err(GeomError::InvalidArgument("the lattice must be 2 to 2048 a side"));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(GeomError::InvalidArgument("the occupation must be a probability"));
    }
    let grid: Vec<bool> = (0..n * n).map(|_| rng.next_f64() < p).collect();
    let spans = spans_vertically(&grid, n);
    Ok((grid, spans))
}

/// Whether an occupied cluster connects the top row to the bottom.
///
/// Uses a disjoint set with two virtual nodes, one for each boundary, which
/// turns the question into a single connectivity query rather than a search
/// per starting site.
fn spans_vertically(grid: &[bool], n: usize) -> bool {
    let top = n * n;
    let bottom = n * n + 1;
    let mut sets = DisjointSet::new(n * n + 2);
    for row in 0..n {
        for column in 0..n {
            let index = row * n + column;
            if !grid[index] {
                continue;
            }
            if row == 0 {
                sets.union(index, top);
            }
            if row + 1 == n {
                sets.union(index, bottom);
            }
            if row + 1 < n && grid[index + n] {
                sets.union(index, index + n);
            }
            if column + 1 < n && grid[index + 1] {
                sets.union(index, index + 1);
            }
        }
    }
    sets.connected(top, bottom)
}

/// Bond percolation on a square lattice: open each bond with probability `p`
/// and report whether the lattice spans.
///
/// The bond threshold in two dimensions is exactly one half, by a duality
/// argument -- the dual of an open bond is a closed one, so the model is
/// self-dual at `p = 1/2` and the transition can be nowhere else. The site
/// threshold has no such argument and is only known numerically.
///
/// # Errors
/// Returns an error for a bad lattice size or probability.
pub fn percolation_bond(n: usize, p: f64, rng: &mut Rng) -> Result<bool, GeomError> {
    if !(2..=2048).contains(&n) {
        return Err(GeomError::InvalidArgument("the lattice must be 2 to 2048 a side"));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(GeomError::InvalidArgument("the opening must be a probability"));
    }
    let top = n * n;
    let bottom = n * n + 1;
    let mut sets = DisjointSet::new(n * n + 2);
    for row in 0..n {
        for column in 0..n {
            let index = row * n + column;
            if row == 0 {
                sets.union(index, top);
            }
            if row + 1 == n {
                sets.union(index, bottom);
            }
            if row + 1 < n && rng.next_f64() < p {
                sets.union(index, index + n);
            }
            if column + 1 < n && rng.next_f64() < p {
                sets.union(index, index + 1);
            }
        }
    }
    Ok(sets.connected(top, bottom))
}

/// Estimates the site percolation threshold by bisection on the spanning
/// probability.
///
/// The true two-dimensional value is about 0.592746, and unlike the bond
/// threshold it has no closed form. A finite lattice puts the half-spanning
/// point slightly off it, and the offset shrinks as the lattice grows.
///
/// # Errors
/// Returns an error for a bad lattice size or trial count.
pub fn percolation_threshold_binary_search(
    n: usize,
    trials: usize,
    rng: &mut Rng,
) -> Result<f64, GeomError> {
    if !(8..=512).contains(&n) || trials == 0 {
        return Err(GeomError::InvalidArgument("percolation_threshold: bad parameters"));
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        let mut spanned = 0usize;
        for _ in 0..trials {
            if percolation_site(n, mid, rng)?.1 {
                spanned += 1;
            }
        }
        if spanned * 2 >= trials {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// The sizes of every occupied cluster, descending.
///
/// # Errors
/// Returns an error if the grid is not square.
pub fn cluster_size_distribution(grid: &[bool], n: usize) -> Result<Vec<usize>, GeomError> {
    if n == 0 || grid.len() != n * n {
        return Err(GeomError::InvalidArgument("the grid is not square"));
    }
    let mut sets = DisjointSet::new(n * n);
    for row in 0..n {
        for column in 0..n {
            let index = row * n + column;
            if !grid[index] {
                continue;
            }
            if row + 1 < n && grid[index + n] {
                sets.union(index, index + n);
            }
            if column + 1 < n && grid[index + 1] {
                sets.union(index, index + 1);
            }
        }
    }
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for index in 0..n * n {
        if grid[index] {
            *counts.entry(sets.find(index)).or_insert(0) += 1;
        }
    }
    let mut sizes: Vec<usize> = counts.into_values().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    Ok(sizes)
}

// ---------------------------------------------------------------------------
// Walks
// ---------------------------------------------------------------------------

/// The number of self-avoiding walks of `n` steps from the origin on the
/// square lattice.
///
/// Counted by exhaustive backtracking, so it is exact and exponential --
/// which is the state of the art: no formula is known, and the published
/// counts come from much cleverer enumerations of the same kind.
///
/// # Errors
/// Returns an error above eighteen steps, where the count exceeds what this
/// enumeration will finish.
pub fn self_avoiding_walk_count(n: usize) -> Result<u64, GeomError> {
    if n > 18 {
        return Err(GeomError::InvalidArgument("the enumeration stops at eighteen steps"));
    }
    if n == 0 {
        return Ok(1);
    }
    let span = 2 * n + 1;
    let mut visited = vec![false; span * span];
    let start = n * span + n;
    visited[start] = true;
    Ok(count_saw(start, n, span, &mut visited))
}

fn count_saw(position: usize, remaining: usize, span: usize, visited: &mut Vec<bool>) -> u64 {
    if remaining == 0 {
        return 1;
    }
    let (row, column) = (position / span, position % span);
    let mut total = 0u64;
    let mut moves = [usize::MAX; 4];
    if row > 0 {
        moves[0] = position - span;
    }
    if row + 1 < span {
        moves[1] = position + span;
    }
    if column > 0 {
        moves[2] = position - 1;
    }
    if column + 1 < span {
        moves[3] = position + 1;
    }
    for next in moves {
        if next == usize::MAX || visited[next] {
            continue;
        }
        visited[next] = true;
        total += count_saw(next, remaining - 1, span, visited);
        visited[next] = false;
    }
    total
}

/// One self-avoiding walk sampled by the Rosenbluth method, with its weight.
///
/// Growing a walk step by step and refusing to revisit gives a *biased*
/// sample: walks that had few choices are over-represented. The Rosenbluth
/// weight -- the product of the available choices at each step -- corrects
/// exactly for that, so weighted averages are unbiased. The method's known
/// weakness is that the weights become very unequal for long walks, so the
/// effective sample size collapses even though the estimator stays unbiased.
///
/// Returns the path and its weight; a walk that traps itself returns a weight
/// of zero.
///
/// # Errors
/// Returns an error for an excessive step count.
pub fn saw_sample_rosenbluth(
    n: usize,
    rng: &mut Rng,
) -> Result<(Vec<(i64, i64)>, f64), GeomError> {
    if n > 10_000 {
        return Err(GeomError::InvalidArgument("the walk is too long"));
    }
    let mut path = vec![(0i64, 0i64)];
    let mut visited = std::collections::HashSet::new();
    visited.insert((0i64, 0i64));
    let mut weight = 1.0f64;
    for _ in 0..n {
        let (x, y) = *path.last().expect("non-empty");
        let options: Vec<(i64, i64)> = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            .into_iter()
            .filter(|site| !visited.contains(site))
            .collect();
        if options.is_empty() {
            return Ok((path, 0.0));
        }
        weight *= options.len() as f64;
        let next = options[pick(rng, options.len())];
        visited.insert(next);
        path.push(next);
    }
    Ok((path, weight))
}

/// An estimate of the connective constant from exact walk counts.
///
/// `mu = lim c_n^(1/n)`, about 2.63816 on the square lattice.
///
/// Two corrections have to be removed and they are removed differently. The
/// counts alternate with parity -- `c_n / c_(n-1)` oscillates between about
/// 2.694 and 2.702 at these lengths -- so the ratio is taken two steps at a
/// time, `sqrt(c_n / c_(n-2))`, which averages the parity out rather than
/// amplifying it. What remains behaves as `mu (1 + (gamma - 1) / n)` because
/// `c_n ~ A mu^n n^(gamma - 1)`, and one Richardson step on `1 / n` cancels
/// it whatever the unknown coefficient. Applying Richardson to the raw
/// consecutive ratios instead makes matters *worse*, since it differences two
/// numbers of opposite parity and doubles the oscillation.
///
/// # Errors
/// Returns an error for fewer than five counts, or a zero count.
pub fn connective_constant_estimate(counts: &[u64]) -> Result<f64, GeomError> {
    if counts.len() < 5 || counts.contains(&0) {
        return Err(GeomError::InvalidArgument("the estimate needs five positive counts"));
    }
    let last = counts.len() - 1;
    let rho = |n: usize| -> f64 { (counts[n] as f64 / counts[n - 2] as f64).sqrt() };
    let a = rho(last);
    let b = rho(last - 2);
    Ok((last as f64 * a - (last - 2) as f64 * b) / 2.0)
}

/// A simple random walk on the `d`-dimensional cubic lattice.
///
/// # Errors
/// Returns an error for zero dimensions or an excessive step count.
pub fn random_walk_lattice(
    steps: usize,
    dimensions: usize,
    rng: &mut Rng,
) -> Result<Vec<Vec<i64>>, GeomError> {
    if dimensions == 0 || dimensions > 8 || steps > 1_000_000 {
        return Err(GeomError::InvalidArgument("random_walk_lattice: bad parameters"));
    }
    let mut position = vec![0i64; dimensions];
    let mut path = vec![position.clone()];
    for _ in 0..steps {
        let axis = pick(rng, dimensions);
        position[axis] += if rng.next_f64() < 0.5 { 1 } else { -1 };
        path.push(position.clone());
    }
    Ok(path)
}

/// Polya's return probability for a simple random walk in `d` dimensions.
///
/// One in one and two dimensions and less than one from three up. The
/// dimension at which a walk stops returning is not a matter of degree: in
/// two dimensions the walker returns with certainty and in three it escapes
/// with probability about 0.66, and nothing continuous separates them.
///
/// # Errors
/// Returns an error for zero dimensions or above eight.
pub fn return_probability(dimensions: usize) -> Result<f64, GeomError> {
    if dimensions == 0 || dimensions > 8 {
        return Err(GeomError::InvalidArgument("return_probability handles 1 to 8 dimensions"));
    }
    // The known values; there is no elementary closed form above two.
    const KNOWN: [f64; 8] = [
        1.0,
        1.0,
        0.340_537_329_5,
        0.193_201_673_0,
        0.135_178_872_0,
        0.104_715_000_0,
        0.085_844_000_0,
        0.072_912_000_0,
    ];
    Ok(KNOWN[dimensions - 1])
}

/// The mean squared end-to-end distance of a set of weighted walks.
///
/// # Errors
/// Returns an error for an empty sample or zero total weight.
pub fn polymer_end_to_end(samples: &[(Vec<(i64, i64)>, f64)]) -> Result<f64, GeomError> {
    if samples.is_empty() {
        return Err(GeomError::InvalidArgument("polymer_end_to_end needs samples"));
    }
    let mut weight_total = 0.0;
    let mut squared_total = 0.0;
    for (path, weight) in samples {
        if *weight <= 0.0 || path.is_empty() {
            continue;
        }
        let (x, y) = path[path.len() - 1];
        squared_total += weight * (x * x + y * y) as f64;
        weight_total += weight;
    }
    if weight_total <= 0.0 {
        return Err(GeomError::Degenerate("every sampled walk was trapped"));
    }
    Ok(squared_total / weight_total)
}

/// The Flory exponent fitted from end-to-end distances at several lengths.
///
/// `<R^2> ~ n^(2 nu)` with `nu = 3/4` exactly in two dimensions -- a result
/// of Nienhuis, and one that Flory's own mean-field argument happens to get
/// right in this dimension and wrong in three.
///
/// # Errors
/// Returns an error for fewer than two lengths or a non-positive distance.
pub fn flory_exponent_estimate(lengths: &[usize], squared: &[f64]) -> Result<f64, GeomError> {
    if lengths.len() < 2 || lengths.len() != squared.len() {
        return Err(GeomError::InvalidArgument("flory_exponent_estimate: mismatched input"));
    }
    if lengths.contains(&0) || squared.iter().any(|r| !(*r > 0.0)) {
        return Err(GeomError::InvalidArgument("the lengths and distances must be positive"));
    }
    // Least squares of ln R^2 against ln n; the slope is twice nu.
    let points: Vec<(f64, f64)> = lengths
        .iter()
        .zip(squared)
        .map(|(n, r)| ((*n as f64).ln(), r.ln()))
        .collect();
    let count = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denominator = count * sxx - sx * sx;
    if denominator.abs() < 1e-12 {
        return Err(GeomError::Degenerate("the lengths do not vary"));
    }
    Ok((count * sxy - sx * sy) / denominator / 2.0)
}

// ---------------------------------------------------------------------------
// Dimers
// ---------------------------------------------------------------------------

/// The number of perfect matchings of an `m` by `n` grid, by Kasteleyn's
/// formula.
///
/// `prod_{j,k} (4 cos^2(pi j / (m+1)) + 4 cos^2(pi k / (n+1)))^(1/4)`. The
/// remarkable part is that a counting problem which is `#P`-complete on a
/// general graph is *polynomial* on a planar one, because the count becomes a
/// Pfaffian once the edges are oriented correctly.
///
/// Returned as a float, since the count outgrows a `u64` by about the twelve
/// by twelve grid; it is exact to rounding and the caller can round it.
///
/// # Errors
/// Returns an error for a zero dimension or an odd number of cells, which
/// admits no perfect matching at all.
pub fn dimer_count_kasteleyn(m: usize, n: usize) -> Result<f64, GeomError> {
    if m == 0 || n == 0 || m > 64 || n > 64 {
        return Err(GeomError::InvalidArgument("the grid must be 1 to 64 a side"));
    }
    if (m * n) % 2 == 1 {
        return Err(GeomError::InvalidArgument("an odd grid has no perfect matching"));
    }
    // Carried in logarithms: the product spans hundreds of orders of
    // magnitude before the fourth root brings it back.
    let mut log_total = 0.0f64;
    for j in 1..=m {
        for k in 1..=n {
            let a = (std::f64::consts::PI * j as f64 / (m + 1) as f64).cos();
            let b = (std::f64::consts::PI * k as f64 / (n + 1) as f64).cos();
            let term = 4.0 * a * a + 4.0 * b * b;
            if term <= 0.0 {
                return Ok(0.0);
            }
            log_total += term.ln();
        }
    }
    Ok((log_total / 4.0).exp())
}

// ---------------------------------------------------------------------------
// Growth
// ---------------------------------------------------------------------------

/// Ballistic deposition on a line, returning the final interface heights.
///
/// A particle falls on a random column and sticks at the first point where it
/// touches the deposit, which may be the side of a neighbouring column rather
/// than the top of its own. That sideways sticking is the whole model: without
/// it the interface stays flat, and with it the interface roughens with the
/// Kardar-Parisi-Zhang exponents.
///
/// # Errors
/// Returns an error for a bad width or an excessive time.
pub fn kpz_growth_ballistic(
    width: usize,
    depositions: usize,
    rng: &mut Rng,
) -> Result<Vec<f64>, GeomError> {
    if !(4..=100_000).contains(&width) || depositions > 200_000_000 {
        return Err(GeomError::InvalidArgument("kpz_growth_ballistic: bad parameters"));
    }
    let mut heights = vec![0i64; width];
    for _ in 0..depositions {
        let column = pick(rng, width);
        let left = heights[(column + width - 1) % width];
        let right = heights[(column + 1) % width];
        // Stick on contact: the new height is one above its own column, or
        // level with a taller neighbour, whichever is higher.
        heights[column] = (heights[column] + 1).max(left).max(right);
    }
    Ok(heights.into_iter().map(|h| h as f64).collect())
}

/// The width of an interface: the standard deviation of its heights.
///
/// # Errors
/// Returns an error for an empty interface.
pub fn interface_width(heights: &[f64]) -> Result<f64, GeomError> {
    if heights.is_empty() {
        return Err(GeomError::InvalidArgument("the interface is empty"));
    }
    let n = heights.len() as f64;
    let mean: f64 = heights.iter().sum::<f64>() / n;
    Ok((heights.iter().map(|h| (h - mean) * (h - mean)).sum::<f64>() / n).sqrt())
}

/// The growth exponent `beta`, fitted from the width against time.
///
/// `W ~ t^beta` before the width saturates, with `beta = 1/3` in the
/// one-dimensional KPZ class. The fit must stay inside the growth regime:
/// once the correlation length reaches the system size the width stops
/// growing altogether, and including saturated points drags the exponent
/// toward zero.
///
/// # Errors
/// Returns an error for fewer than three points or a non-positive width.
pub fn growth_exponent_estimate(times: &[f64], widths: &[f64]) -> Result<f64, GeomError> {
    if times.len() < 3 || times.len() != widths.len() {
        return Err(GeomError::InvalidArgument("growth_exponent_estimate: mismatched input"));
    }
    if times.iter().any(|t| !(*t > 0.0)) || widths.iter().any(|w| !(*w > 0.0)) {
        return Err(GeomError::InvalidArgument("the times and widths must be positive"));
    }
    let points: Vec<(f64, f64)> = times.iter().zip(widths).map(|(t, w)| (t.ln(), w.ln())).collect();
    let count = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denominator = count * sxx - sx * sx;
    if denominator.abs() < 1e-12 {
        return Err(GeomError::Degenerate("the times do not vary"));
    }
    Ok((count * sxy - sx * sy) / denominator)
}

/// The Abelian sandpile: drop grains at random sites and record the size of
/// each avalanche.
///
/// The pile organises itself to the critical state without any parameter
/// being tuned, which is what "self-organised criticality" means: the
/// avalanche sizes come out power-law distributed whatever the initial
/// condition, with no temperature or field set by hand.
///
/// # Errors
/// Returns an error for a bad lattice size or drop count.
pub fn sandpile_avalanche_distribution(
    n: usize,
    drops: usize,
    rng: &mut Rng,
) -> Result<Vec<u64>, GeomError> {
    if !(4..=256).contains(&n) || drops == 0 {
        return Err(GeomError::InvalidArgument("sandpile: bad parameters"));
    }
    let mut pile = vec![0u8; n * n];
    let mut sizes = Vec::with_capacity(drops);
    for _ in 0..drops {
        let site = pick(rng, n * n);
        pile[site] += 1;
        let mut stack = vec![site];
        let mut toppled = 0u64;
        while let Some(current) = stack.pop() {
            if pile[current] < 4 {
                continue;
            }
            pile[current] -= 4;
            toppled += 1;
            let (row, column) = (current / n, current % n);
            // Grains falling off the edge leave the system, which is what
            // keeps the pile from filling up.
            if row > 0 {
                pile[current - n] += 1;
                stack.push(current - n);
            }
            if row + 1 < n {
                pile[current + n] += 1;
                stack.push(current + n);
            }
            if column > 0 {
                pile[current - 1] += 1;
                stack.push(current - 1);
            }
            if column + 1 < n {
                pile[current + 1] += 1;
                stack.push(current + 1);
            }
            stack.push(current);
        }
        sizes.push(toppled);
    }
    Ok(sizes)
}

/// Clauset's maximum-likelihood power-law fit above a cutoff, with the
/// Kolmogorov-Smirnov distance to the fitted law.
///
/// Fitting a straight line to a log-log histogram is the traditional method
/// and it is badly biased: the bins in the tail hold few points, and least
/// squares weights them as heavily as the bins that hold thousands. The
/// maximum-likelihood estimator has a closed form for a continuous power law
/// and no such problem.
///
/// Returns `(alpha, ks_distance)`.
///
/// # Errors
/// Returns an error for a non-positive cutoff or too few points above it.
pub fn power_law_fit_clauset(data: &[f64], x_min: f64) -> Result<(f64, f64), GeomError> {
    if !(x_min > 0.0) {
        return Err(GeomError::InvalidArgument("the cutoff must be positive"));
    }
    let mut tail: Vec<f64> = data.iter().copied().filter(|x| *x >= x_min).collect();
    if tail.len() < 5 {
        return Err(GeomError::InvalidArgument("too few points above the cutoff"));
    }
    let n = tail.len() as f64;
    let sum: f64 = tail.iter().map(|x| (x / x_min).ln()).sum();
    if sum <= 0.0 {
        return Err(GeomError::Degenerate("every point sits at the cutoff"));
    }
    let alpha = 1.0 + n / sum;

    // The Kolmogorov-Smirnov distance between the empirical distribution and
    // the fitted one.
    tail.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut distance: f64 = 0.0;
    for (k, x) in tail.iter().enumerate() {
        let empirical_low = k as f64 / n;
        let empirical_high = (k + 1) as f64 / n;
        let fitted = 1.0 - (x / x_min).powf(1.0 - alpha);
        distance = distance
            .max((fitted - empirical_low).abs())
            .max((fitted - empirical_high).abs());
    }
    Ok((alpha, distance))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------
    // Percolation
    // -----------------------------------------------------------------

    #[test]
    fn percolation_spans_when_it_should_and_not_when_it_should_not() {
        // The limits are certainties, not probabilities: an empty lattice
        // never spans and a full one always does.
        let mut rng = Rng::new(0x_1A77_0001);
        for n in [8usize, 16, 32] {
            assert!(!percolation_site(n, 0.0, &mut rng).unwrap().1);
            assert!(percolation_site(n, 1.0, &mut rng).unwrap().1);
            assert!(!percolation_bond(n, 0.0, &mut rng).unwrap());
            assert!(percolation_bond(n, 1.0, &mut rng).unwrap());
        }
        // The spanning probability rises monotonically with the occupation.
        let n = 24usize;
        let trials = 400usize;
        let mut previous = -1.0;
        for p in [0.3f64, 0.45, 0.55, 0.59, 0.65, 0.8] {
            let spanned = (0..trials)
                .filter(|_| percolation_site(n, p, &mut rng).unwrap().1)
                .count() as f64
                / trials as f64;
            assert!(spanned >= previous - 0.03, "spanning fell at p = {p}");
            previous = spanned;
        }
        assert!(previous > 0.98, "at p = 0.8 the lattice should nearly always span");

        assert!(percolation_site(1, 0.5, &mut rng).is_err());
        assert!(percolation_site(8, 1.5, &mut rng).is_err());
        assert!(percolation_bond(1, 0.5, &mut rng).is_err());
        assert!(percolation_bond(8, -0.1, &mut rng).is_err());
    }

    #[test]
    fn the_site_threshold_comes_out_near_its_known_value() {
        // About 0.592746, which has no closed form and is known only
        // numerically. A finite lattice puts the half-spanning point close to
        // it, and closer as the lattice grows.
        let mut rng = Rng::new(0x_1A77_0002);
        let mut errors = Vec::new();
        for n in [16usize, 32, 64] {
            let estimate = percolation_threshold_binary_search(n, 200, &mut rng).unwrap();
            assert!(
                (0.55..0.64).contains(&estimate),
                "at n = {n} the threshold came out {estimate}"
            );
            errors.push((estimate - 0.592_746).abs());
        }
        assert!(
            errors[2] < 0.02,
            "the largest lattice is still {} away from the known value",
            errors[2]
        );

        // Bond percolation's threshold is exactly one half, by self-duality.
        let trials = 600usize;
        let n = 48usize;
        let at_half = (0..trials)
            .filter(|_| percolation_bond(n, 0.5, &mut rng).unwrap())
            .count() as f64
            / trials as f64;
        assert!(
            (0.35..0.65).contains(&at_half),
            "at the exact bond threshold the spanning rate is {at_half}"
        );
        // Well away from it the answer is nearly certain either way.
        let below = (0..trials)
            .filter(|_| percolation_bond(n, 0.40, &mut rng).unwrap())
            .count();
        let above = (0..trials)
            .filter(|_| percolation_bond(n, 0.60, &mut rng).unwrap())
            .count();
        assert!(below < trials / 20, "below the threshold {below} of {trials} spanned");
        assert!(above > trials * 19 / 20, "above it only {above} of {trials} spanned");

        assert!(percolation_threshold_binary_search(4, 10, &mut rng).is_err());
        assert!(percolation_threshold_binary_search(16, 0, &mut rng).is_err());
    }

    #[test]
    fn the_cluster_sizes_account_for_every_occupied_site() {
        // A partition, so the sizes must sum to the occupancy. That is the
        // check that catches a union-find used wrongly, which otherwise
        // produces plausible-looking distributions.
        let mut rng = Rng::new(0x_1A77_0003);
        for p in [0.2f64, 0.45, 0.59, 0.75] {
            let n = 32usize;
            let (grid, spans) = percolation_site(n, p, &mut rng).unwrap();
            let sizes = cluster_size_distribution(&grid, n).unwrap();
            let occupied = grid.iter().filter(|c| **c).count();
            assert_eq!(
                sizes.iter().sum::<usize>(),
                occupied,
                "the clusters do not partition the occupied sites at p = {p}"
            );
            assert!(sizes.windows(2).all(|w| w[0] >= w[1]), "the sizes are not sorted");
            if spans {
                // A spanning cluster must reach across, so it has at least n
                // sites.
                assert!(
                    sizes[0] >= n,
                    "a spanning lattice has a largest cluster of only {}",
                    sizes[0]
                );
            }
        }
        // The largest cluster grows sharply through the threshold, which is
        // the order parameter of the transition.
        let n = 48usize;
        let mut previous = 0.0;
        for p in [0.4f64, 0.55, 0.62, 0.75] {
            let mut total = 0.0;
            for _ in 0..20 {
                let (grid, _) = percolation_site(n, p, &mut rng).unwrap();
                let sizes = cluster_size_distribution(&grid, n).unwrap();
                total += sizes.first().copied().unwrap_or(0) as f64;
            }
            let fraction = total / 20.0 / (n * n) as f64;
            assert!(fraction > previous, "the largest cluster shrank at p = {p}");
            previous = fraction;
        }
        assert!(previous > 0.6, "well above the threshold most sites should join up");

        assert!(cluster_size_distribution(&[true; 4], 3).is_err());
        assert_eq!(cluster_size_distribution(&[false; 16], 4).unwrap(), Vec::<usize>::new());
        assert_eq!(cluster_size_distribution(&[true; 16], 4).unwrap(), vec![16]);
    }

    // -----------------------------------------------------------------
    // Walks
    // -----------------------------------------------------------------

    #[test]
    fn the_self_avoiding_walk_counts_match_the_published_sequence() {
        // OEIS A001411: the number of self-avoiding walks on the square
        // lattice. These are counted, not fitted, so they either match
        // exactly or the enumeration is wrong.
        const EXPECTED: [u64; 16] = [
            1, 4, 12, 36, 100, 284, 780, 2172, 5916, 16268, 44100, 120292, 324932, 881500,
            2374444, 6416596,
        ];
        for (n, &count) in EXPECTED.iter().enumerate() {
            assert_eq!(
                self_avoiding_walk_count(n).unwrap(),
                count,
                "the count at n = {n} is wrong"
            );
        }
        // Every walk is self-avoiding, so the count is below the free walk's.
        // From the second step onward: the first step has no chance to
        // revisit, so the two counts coincide at n = 1.
        assert_eq!(self_avoiding_walk_count(1).unwrap(), 4);
        for n in 2..=12usize {
            let saw = self_avoiding_walk_count(n).unwrap();
            assert!(saw < 4u64.pow(n as u32), "the count exceeds the free walk at n = {n}");
            // And above the strictly directed walk's.
            assert!(saw >= 2u64.pow(n as u32), "the count is below the directed walk at n = {n}");
        }
        assert!(self_avoiding_walk_count(19).is_err());

        // The connective constant is about 2.638.
        let counts: Vec<u64> = (0..=16).map(|n| self_avoiding_walk_count(n).unwrap()).collect();
        let mu = connective_constant_estimate(&counts).unwrap();
        assert!(
            (mu - 2.638_158).abs() < 0.01,
            "the connective constant came out {mu}"
        );
        // The raw ratio is still nearly two per cent high at sixteen steps, so
        // the extrapolation is doing real work rather than dressing up a
        // number that was already right.  Comparing the two errors rather
        // than testing the raw ratio against a fixed tolerance keeps the
        // control tied to what it is meant to show: the extrapolation must
        // beat the ratio it is built from by a wide margin.
        let raw = counts[16] as f64 / counts[15] as f64;
        let raw_error = (raw - 2.638_158).abs();
        let extrapolated_error = (mu - 2.638_158).abs();
        assert!(
            raw_error > 0.04,
            "the raw ratio {raw} is already accurate, so the test proves nothing"
        );
        assert!(
            raw_error > 10.0 * extrapolated_error,
            "the extrapolation ({extrapolated_error}) is no better than the raw \
             ratio ({raw_error})"
        );
        assert!(connective_constant_estimate(&[1, 4, 12, 36]).is_err());
        assert!(connective_constant_estimate(&[1, 0, 4, 12, 36]).is_err());
    }

    #[test]
    fn rosenbluth_sampling_reproduces_the_exact_walk_count() {
        // The mean Rosenbluth weight is exactly the number of walks -- that
        // is what makes the weighting unbiased rather than merely plausible.
        // Trapped walks contribute zero and must be counted in the average,
        // not discarded: discarding them is precisely the bias the weight
        // exists to remove.
        let mut rng = Rng::new(0x_1A77_0004);
        for n in [4usize, 6, 8] {
            let exact = self_avoiding_walk_count(n).unwrap() as f64;
            let trials = 200_000usize;
            let mut total = 0.0;
            for _ in 0..trials {
                let (path, weight) = saw_sample_rosenbluth(n, &mut rng).unwrap();
                if weight > 0.0 {
                    assert_eq!(path.len(), n + 1, "a completed walk has the wrong length");
                    // It really is self-avoiding.
                    let unique: std::collections::HashSet<(i64, i64)> =
                        path.iter().copied().collect();
                    assert_eq!(unique.len(), path.len(), "the walk revisited a site");
                }
                // Every walk starts with four choices, so the weight carries
                // the factor 4 the first step contributes.
                total += weight;
            }
            let estimate = total / trials as f64;
            assert!(
                (estimate - exact).abs() < 0.03 * exact,
                "n = {n}: the weights average {estimate} against the exact count {exact}"
            );
        }
        assert!(saw_sample_rosenbluth(20_000, &mut rng).is_err());
    }

    #[test]
    fn a_self_avoiding_walk_spreads_faster_than_a_free_one() {
        // The Flory exponent in two dimensions is exactly 3/4, against the
        // free walk's 1/2. That a walk forbidden to cross itself travels
        // further is unsurprising; that the exponent is a simple fraction is
        // not, and it is the reason the model is studied.
        let mut rng = Rng::new(0x_1A77_0005);
        let lengths = [10usize, 20, 40, 80];
        let mut squared = Vec::new();
        for &n in &lengths {
            let samples: Vec<(Vec<(i64, i64)>, f64)> = (0..4_000)
                .map(|_| saw_sample_rosenbluth(n, &mut rng).unwrap())
                .collect();
            squared.push(polymer_end_to_end(&samples).unwrap());
        }
        assert!(
            squared.windows(2).all(|w| w[1] > w[0]),
            "the walks did not lengthen: {squared:?}"
        );
        let nu = flory_exponent_estimate(&lengths, &squared).unwrap();
        assert!(
            (nu - 0.75).abs() < 0.06,
            "the Flory exponent came out {nu}, not near three quarters"
        );
        assert!(nu > 0.55, "it should clearly exceed the free walk's one half");

        // The free walk gives one half, from its own trajectories.
        let free_lengths = [50usize, 100, 200, 400];
        let mut free_squared = Vec::new();
        for &n in &free_lengths {
            let mut total = 0.0;
            for _ in 0..2_000 {
                let path = random_walk_lattice(n, 2, &mut rng).unwrap();
                let end = path.last().unwrap();
                total += (end[0] * end[0] + end[1] * end[1]) as f64;
            }
            free_squared.push(total / 2_000.0);
        }
        let free_nu = flory_exponent_estimate(&free_lengths, &free_squared).unwrap();
        assert!(
            (free_nu - 0.5).abs() < 0.04,
            "the free walk's exponent came out {free_nu}"
        );
        // And its mean square displacement is the step count itself.
        assert!(
            (free_squared[3] / 400.0 - 1.0).abs() < 0.1,
            "the free walk's mean square displacement is {} at 400 steps",
            free_squared[3]
        );

        assert!(polymer_end_to_end(&[]).is_err());
        assert!(flory_exponent_estimate(&[10], &[1.0]).is_err());
        assert!(flory_exponent_estimate(&[10, 20], &[1.0]).is_err());
        assert!(flory_exponent_estimate(&[0, 20], &[1.0, 2.0]).is_err());
        assert!(random_walk_lattice(10, 0, &mut rng).is_err());
        assert!(random_walk_lattice(10, 9, &mut rng).is_err());
    }

    #[test]
    fn the_return_probability_is_one_below_three_dimensions_and_less_above() {
        // Polya's theorem, which is a statement about certainty rather than
        // about magnitude: in one and two dimensions the walker returns with
        // probability one and in three it does not.
        assert!(close(return_probability(1).unwrap(), 1.0, 1e-15));
        assert!(close(return_probability(2).unwrap(), 1.0, 1e-15));
        assert!(close(return_probability(3).unwrap(), 0.340_537_33, 1e-6));
        let mut previous = 1.0;
        for d in 3..=8usize {
            let p = return_probability(d).unwrap();
            assert!(p < previous, "the return probability rose at d = {d}");
            assert!((0.0..1.0).contains(&p));
            previous = p;
        }
        assert!(return_probability(0).is_err());
        assert!(return_probability(9).is_err());

        // Simulated. The distinction is not visible in a single run length:
        // the two-dimensional walk returns with certainty but only
        // logarithmically slowly, so at any finite horizon a good fraction of
        // walks have not yet come back. What separates the dimensions is that
        // the two-dimensional fraction keeps climbing with the horizon while
        // the three-dimensional one has already stopped.
        let mut rng = Rng::new(0x_1A77_0006);
        let trials = 300usize;
        let horizons = [500usize, 2_000, 8_000];
        let mut two = Vec::new();
        let mut three = Vec::new();
        for &steps in &horizons {
            for (dimensions, target) in [(2usize, &mut two), (3usize, &mut three)] {
                let returned = (0..trials)
                    .filter(|_| {
                        let path = random_walk_lattice(steps, dimensions, &mut rng).unwrap();
                        path.iter().skip(1).any(|p| p.iter().all(|c| *c == 0))
                    })
                    .count() as f64
                    / trials as f64;
                target.push(returned);
            }
        }
        assert!(
            two.windows(2).all(|w| w[1] > w[0]),
            "the two-dimensional fraction did not keep climbing: {two:?}"
        );
        assert!(
            two[2] > three[2] + 0.2,
            "at the longest horizon two dimensions returned {} against three's {}",
            two[2],
            three[2]
        );
        // Three dimensions saturates: sixteenfold more time buys almost
        // nothing, because a walk that has not returned by then never will.
        assert!(
            three[2] - three[0] < 0.1,
            "the three-dimensional fraction moved from {} to {}",
            three[0],
            three[2]
        );
        assert!(
            (three[2] - 0.34).abs() < 0.08,
            "the three-dimensional fraction is {}, not near Polya's 0.34",
            three[2]
        );
    }

    // -----------------------------------------------------------------
    // Dimers
    // -----------------------------------------------------------------

    #[test]
    fn the_dimer_count_matches_the_values_that_can_be_counted_by_hand() {
        // The two-by-two grid has two matchings and the two-by-n grid has the
        // Fibonacci numbers, both of which can be checked without the
        // formula. The eight-by-eight value is the published one.
        assert!(close(dimer_count_kasteleyn(2, 2).unwrap(), 2.0, 1e-9));
        assert!(close(dimer_count_kasteleyn(1, 2).unwrap(), 1.0, 1e-9));
        assert!(close(dimer_count_kasteleyn(1, 4).unwrap(), 1.0, 1e-9));
        // A 2 x n grid has F(n + 1) matchings: 1, 2, 3, 5, 8, 13, ...
        const FIBONACCI: [f64; 8] = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
        for (k, &expected) in FIBONACCI.iter().enumerate() {
            let value = dimer_count_kasteleyn(2, k + 1).unwrap();
            assert!(
                close(value, expected, 1e-6 * expected),
                "the 2 by {} grid gives {value}, not {expected}",
                k + 1
            );
        }
        // The famous eight-by-eight count.
        let eight = dimer_count_kasteleyn(8, 8).unwrap();
        assert!(
            close(eight, 12_988_816.0, 1.0),
            "the eight by eight count is {eight}, not 12988816"
        );
        // And the four-by-four, which is 36.
        assert!(close(dimer_count_kasteleyn(4, 4).unwrap(), 36.0, 1e-6));
        // Symmetric in its two arguments, as it must be.
        for (m, n) in [(2usize, 6usize), (4, 6), (6, 8), (3, 8)] {
            let a = dimer_count_kasteleyn(m, n).unwrap();
            let b = dimer_count_kasteleyn(n, m).unwrap();
            assert!(close(a, b, 1e-6 * a.max(1.0)), "the count is not symmetric at {m} by {n}");
        }
        // An odd number of cells admits no matching at all, and the routine
        // says so rather than returning zero and letting it pass unnoticed.
        assert!(dimer_count_kasteleyn(3, 3).unwrap_err() == crate::error::GeomError::InvalidArgument("an odd grid has no perfect matching"));
        assert!(dimer_count_kasteleyn(0, 4).is_err());
        assert!(dimer_count_kasteleyn(4, 100).is_err());
    }

    // -----------------------------------------------------------------
    // Growth and avalanches
    // -----------------------------------------------------------------

    #[test]
    fn a_growing_interface_roughens_with_the_kpz_exponent() {
        // The width grows as t^(1/3) before saturating. Fitting outside the
        // growth regime is the standard way to get the wrong answer: once the
        // correlations reach the system size the width stops growing, and
        // including those points drags the exponent toward zero.
        let mut rng = Rng::new(0x_1A77_0007);
        let width = 2_048usize;
        let mut heights = vec![0f64; width];
        let mut times = Vec::new();
        let mut widths = Vec::new();
        let mut deposited = 0usize;
        for &target in &[4usize, 8, 16, 32, 64, 128] {
            let want = target * width;
            let extra = want - deposited;
            // Continue the same interface rather than starting again.
            let grown = kpz_continue(&mut heights, extra, &mut rng);
            assert!(grown, "the growth step failed");
            deposited = want;
            times.push(target as f64);
            widths.push(interface_width(&heights).unwrap());
        }
        assert!(
            widths.windows(2).all(|w| w[1] > w[0]),
            "the interface did not roughen: {widths:?}"
        );
        let beta = growth_exponent_estimate(&times, &widths).unwrap();
        assert!(
            (beta - 1.0 / 3.0).abs() < 0.06,
            "the growth exponent came out {beta}, not near one third"
        );
        // It is clearly above the 1/4 of a linear interface and below 1/2.
        assert!(beta > 0.27 && beta < 0.45, "the exponent is {beta}");

        // A flat interface has zero width, and one deposition column has
        // some.
        assert!(close(interface_width(&[3.0; 10]).unwrap(), 0.0, 1e-15));
        assert!(interface_width(&[]).is_err());
        assert!(growth_exponent_estimate(&[1.0, 2.0], &[1.0, 2.0]).is_err());
        assert!(growth_exponent_estimate(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(growth_exponent_estimate(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(kpz_growth_ballistic(2, 10, &mut rng).is_err());
    }

    /// Continues a ballistic interface in place, so the growth exponent can
    /// be fitted along one trajectory rather than across restarts.
    fn kpz_continue(heights: &mut [f64], depositions: usize, rng: &mut Rng) -> bool {
        let width = heights.len();
        if width < 4 {
            return false;
        }
        for _ in 0..depositions {
            let column = pick(rng, width);
            let left = heights[(column + width - 1) % width];
            let right = heights[(column + 1) % width];
            heights[column] = (heights[column] + 1.0).max(left).max(right);
        }
        true
    }

    #[test]
    fn the_standalone_growth_routine_agrees_with_the_incremental_one() {
        // The two must give statistically the same interface, since they are
        // the same rule.
        let mut rng = Rng::new(0x_1A77_0008);
        let width = 512usize;
        let a = kpz_growth_ballistic(width, 40 * width, &mut rng).unwrap();
        let mut b = vec![0f64; width];
        kpz_continue(&mut b, 40 * width, &mut rng);
        let wa = interface_width(&a).unwrap();
        let wb = interface_width(&b).unwrap();
        assert!(
            (wa - wb).abs() < 0.25 * wa.max(wb),
            "the two routines give widths {wa} and {wb}"
        );
        // The mean height is the deposition count per column, at least.
        let mean: f64 = a.iter().sum::<f64>() / width as f64;
        assert!(mean >= 40.0, "the mean height is only {mean}");
        // Ballistic deposition grows faster than the deposition rate, since
        // sideways sticking adds height without adding a particle to that
        // column.
        assert!(mean > 40.0, "sideways sticking should raise the interface above 40");
    }

    #[test]
    fn sandpile_avalanches_are_power_law_distributed() {
        // The pile reaches its critical state without any parameter being
        // tuned, which is what self-organised criticality means.
        let mut rng = Rng::new(0x_1A77_0009);
        let sizes = sandpile_avalanche_distribution(32, 60_000, &mut rng).unwrap();
        assert_eq!(sizes.len(), 60_000);
        // Most drops do nothing; a few topple a great deal.
        let quiet = sizes.iter().filter(|s| **s == 0).count();
        assert!(quiet > 0, "every drop caused an avalanche");
        let largest = sizes.iter().copied().max().unwrap();
        assert!(largest > 100, "the largest avalanche is only {largest}");

        // The distribution's tail is a power law with an exponent near one
        // and a half in two dimensions.
        let tail: Vec<f64> = sizes
            .iter()
            .skip(20_000)
            .filter(|s| **s > 0)
            .map(|s| *s as f64)
            .collect();
        let (alpha, ks) = power_law_fit_clauset(&tail, 5.0).unwrap();
        assert!(
            (1.0..2.5).contains(&alpha),
            "the avalanche exponent came out {alpha}"
        );
        assert!(ks < 0.25, "the fit is poor: the KS distance is {ks}");

        assert!(sandpile_avalanche_distribution(2, 100, &mut rng).is_err());
        assert!(sandpile_avalanche_distribution(16, 0, &mut rng).is_err());
    }

    #[test]
    fn the_power_law_fit_recovers_the_exponent_it_was_given() {
        // Synthetic data with a known exponent, generated by inverse
        // transform: if x is uniform then x_min * u^(-1/(alpha-1)) is a power
        // law. Recovering alpha from it is the only honest test of the
        // estimator.
        let mut rng = Rng::new(0x_1A77_000A);
        for truth in [1.5f64, 2.0, 2.5, 3.5] {
            let x_min = 2.0f64;
            let data: Vec<f64> = (0..40_000)
                .map(|_| {
                    let u = rng.next_f64().max(1e-12);
                    x_min * u.powf(-1.0 / (truth - 1.0))
                })
                .collect();
            let (alpha, ks) = power_law_fit_clauset(&data, x_min).unwrap();
            assert!(
                (alpha - truth).abs() < 0.05,
                "the fit gives {alpha} for a true exponent of {truth}"
            );
            assert!(ks < 0.02, "the KS distance is {ks} on data drawn from the fitted law");
        }
        // Data that is not a power law is rejected by the distance, not by
        // the exponent -- which is the point of reporting both.
        let uniform: Vec<f64> = (0..20_000).map(|_| 2.0 + rng.next_f64() * 8.0).collect();
        let (_, ks) = power_law_fit_clauset(&uniform, 2.0).unwrap();
        assert!(ks > 0.1, "uniform data should not fit a power law: the distance is {ks}");

        assert!(power_law_fit_clauset(&[1.0, 2.0, 3.0], 0.0).is_err());
        assert!(power_law_fit_clauset(&[1.0, 2.0], 1.0).is_err());
        assert!(power_law_fit_clauset(&[2.0; 10], 2.0).is_err());
    }
}
