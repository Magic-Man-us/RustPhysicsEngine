//! Clustering, mixture models and nearest neighbours.
//!
//! # Clustering has no ground truth, so the tests need invariants
//!
//! Nothing here has a right answer to compare against. What it has
//! instead is a supply of exact statements, and those are what the tests
//! use:
//!
//! *Lloyd's algorithm cannot go uphill.* Each half of a k-means
//! iteration -- reassigning points to their nearest centre, then moving
//! each centre to its cluster's mean -- minimises the same objective
//! over one of its two arguments, so the inertia is non-increasing and
//! the algorithm terminates in finitely many steps. There are finitely
//! many assignments and none repeats.
//!
//! *Expectation-maximisation cannot go downhill.* The same argument in
//! the other direction: each step maximises a lower bound that touches
//! the log-likelihood at the current parameters, so the likelihood
//! climbs monotonically. Both are asserted step by step rather than end
//! to end, because a monotone sequence is a much sharper claim than an
//! improved endpoint.
//!
//! *A label is not a name.* Cluster indices are arbitrary, so every
//! comparison between two clusterings has to be invariant under
//! relabelling either of them. [`adjusted_rand_index`] is, exactly, and
//! it is corrected for chance so that two independent random partitions
//! score about zero rather than about a half.
//!
//! # Where the guarantees stop, and why that is worth saying
//!
//! Single and complete linkage produce merge heights that never
//! decrease, so their dendrograms can be drawn without crossings.
//! *Centroid linkage does not.* Merging two clusters moves their centre
//! to somewhere between them, which can be closer to a third cluster
//! than either original was, and the dendrogram then contains an
//! inversion. That is a property of the method, not a bug in it, and
//! [`Linkage::Centroid`] is documented and tested as inverting rather
//! than quietly producing dendrograms nobody should draw.
//!
//! DBSCAN's core points are determined by the data alone and do not
//! depend on the order it arrives in. Its *border* points can: a point
//! within reach of two clusters joins whichever claimed it first. That
//! asymmetry is in the algorithm as Ester and colleagues defined it, and
//! pretending otherwise would mean inventing a tie-break and calling it
//! DBSCAN.

use crate::error::SolveError;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// Squared Euclidean distance.
fn distance_squared(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euclidean distance.
fn distance(a: &[f64], b: &[f64]) -> f64 {
    distance_squared(a, b).sqrt()
}

/// Checks that a dataset is non-empty, rectangular and finite, and
/// returns its dimension.
fn check_data(data: &[Vec<f64>]) -> Result<usize, SolveError> {
    if data.is_empty() {
        return Err(SolveError::InvalidArgument("the dataset is empty"));
    }
    let dim = data[0].len();
    if dim == 0 {
        return Err(SolveError::InvalidArgument("the points have no coordinates"));
    }
    if data.iter().any(|p| p.len() != dim) {
        return Err(SolveError::InvalidArgument("the dataset is ragged"));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SolveError::InvalidArgument("the data must be finite"));
    }
    Ok(dim)
}

/// The outcome of a k-means run.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeans {
    /// Cluster centres.
    pub centroids: Vec<Vec<f64>>,
    /// The cluster each point was assigned to.
    pub labels: Vec<usize>,
    /// The inertia after each iteration, which is non-increasing.
    pub inertia_history: Vec<f64>,
    /// How many iterations ran before the assignment stopped changing.
    pub iterations: usize,
}

impl KMeans {
    /// The final within-cluster sum of squared distances.
    pub fn inertia(&self) -> f64 {
        *self.inertia_history.last().expect("there is always one iteration")
    }
}

/// Chooses `k` starting centres by the k-means++ rule: the first
/// uniformly at random, each subsequent one with probability
/// proportional to its squared distance from the nearest centre already
/// chosen.
///
/// The rule matters. Uniform initialisation regularly puts two centres
/// in the same dense region and leaves another region unclaimed, and
/// Lloyd's algorithm cannot repair that -- it is a local method and the
/// bad split is a local optimum. The `D^2` weighting makes the expected
/// final inertia within a logarithmic factor of the best possible,
/// which is the only approximation guarantee k-means has.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid dataset, `k == 0`, or
/// more centres than points.
pub fn kmeans_pp_init(
    data: &[Vec<f64>],
    k: usize,
    rng: &mut Rng,
) -> Result<Vec<Vec<f64>>, SolveError> {
    check_data(data)?;
    if k == 0 {
        return Err(SolveError::InvalidArgument("need at least one cluster"));
    }
    if k > data.len() {
        return Err(SolveError::InvalidArgument("more clusters than points"));
    }
    let n = data.len();
    let first = (rng.next_u64() % n as u64) as usize;
    let mut centres = vec![data[first].clone()];
    let mut best = vec![0.0; n];
    for (i, p) in data.iter().enumerate() {
        best[i] = distance_squared(p, &centres[0]);
    }
    while centres.len() < k {
        let total: f64 = best.iter().sum();
        let pick = if total > 0.0 {
            // Weighted by D^2. With every remaining point coincident
            // with a centre the total is zero and there is nothing to
            // weight by, so fall back to a uniform draw rather than
            // dividing by nothing.
            let target = rng.next_f64() * total;
            let mut running = 0.0;
            let mut chosen = n - 1;
            for (i, w) in best.iter().enumerate() {
                running += w;
                if running >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            (rng.next_u64() % n as u64) as usize
        };
        centres.push(data[pick].clone());
        let latest = centres.last().expect("just pushed");
        for (i, p) in data.iter().enumerate() {
            best[i] = best[i].min(distance_squared(p, latest));
        }
    }
    Ok(centres)
}

/// How many times [`kmeans`] restarts before keeping the best.
///
/// Lloyd's algorithm converges to a local optimum, and on data with
/// well-separated groups an unlucky k-means++ draw can still leave two
/// centres inside one group and one centre spanning two others. That is
/// a stable configuration -- no single point wants to move -- and no
/// number of iterations escapes it. Restarting and keeping the lowest
/// inertia is the only remedy, and it is what every practical
/// implementation does. On the three-blob data in the tests about one
/// single run in two hundred lands on an optimum with forty times the
/// inertia of the best -- rare enough to be missed by a quick look and
/// common enough to matter.
const RESTARTS: usize = 10;

/// Lloyd's algorithm, restarted [`RESTARTS`] times from independent
/// k-means++ starts, keeping the run with the lowest inertia.
///
/// The result carries the winning run's inertia history, which is
/// non-increasing within that run -- see the module note. Use
/// [`kmeans_once`] to observe a single trajectory.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid dataset, `k == 0`,
/// more clusters than points, or zero iterations.
pub fn kmeans(
    data: &[Vec<f64>],
    k: usize,
    iters: usize,
    rng: &mut Rng,
) -> Result<KMeans, SolveError> {
    let mut best: Option<KMeans> = None;
    for _ in 0..RESTARTS {
        let run = kmeans_once(data, k, iters, rng)?;
        if best.as_ref().is_none_or(|b| run.inertia() < b.inertia()) {
            best = Some(run);
        }
    }
    Ok(best.expect("at least one restart"))
}

/// A single run of Lloyd's algorithm from one k-means++ start.
///
/// Runs until the assignment stops changing or `iters` iterations have
/// passed. The inertia after each iteration is recorded, and it is
/// non-increasing by construction.
///
/// An empty cluster is refilled with the point currently furthest from
/// its own centre. Leaving it empty would silently return fewer clusters
/// than were asked for, and the mean of no points is not a number.
///
/// # Errors
///
/// As [`kmeans`].
pub fn kmeans_once(
    data: &[Vec<f64>],
    k: usize,
    iters: usize,
    rng: &mut Rng,
) -> Result<KMeans, SolveError> {
    let dim = check_data(data)?;
    if iters == 0 {
        return Err(SolveError::InvalidArgument("need at least one iteration"));
    }
    let mut centroids = kmeans_pp_init(data, k, rng)?;
    let n = data.len();
    let mut labels = vec![0usize; n];
    let mut history = Vec::with_capacity(iters);
    let mut used = 0;
    for step in 0..iters {
        used = step + 1;
        // Assignment: each point to its nearest centre. This minimises
        // the inertia over the labels with the centres held fixed.
        let mut changed = false;
        for (i, p) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (c, centre) in centroids.iter().enumerate() {
                let d = distance_squared(p, centre);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if labels[i] != best {
                changed = true;
            }
            labels[i] = best;
        }
        // Update: each centre to its cluster's mean, which minimises the
        // inertia over the centres with the labels held fixed.
        let mut sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in data.iter().enumerate() {
            counts[labels[i]] += 1;
            for j in 0..dim {
                sums[labels[i]][j] += p[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] / counts[c] as f64;
                }
            }
        }
        // Refill any empty cluster with the worst-served point.
        for c in 0..k {
            if counts[c] == 0 {
                let (worst, _) = data
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i, distance_squared(p, &centroids[labels[i]])))
                    .fold((0usize, -1.0), |acc, x| if x.1 > acc.1 { x } else { acc });
                centroids[c] = data[worst].clone();
                counts[labels[worst]] -= 1;
                counts[c] = 1;
                labels[worst] = c;
                changed = true;
            }
        }
        let inertia: f64 = data
            .iter()
            .enumerate()
            .map(|(i, p)| distance_squared(p, &centroids[labels[i]]))
            .sum();
        history.push(inertia);
        if !changed && step > 0 {
            break;
        }
    }
    Ok(KMeans { centroids, labels, inertia_history: history, iterations: used })
}

/// The final inertia for each cluster count in `k_range`, for plotting
/// an elbow.
///
/// Inertia falls monotonically with `k` in expectation and reaches zero
/// when every point is its own cluster, so the number alone says
/// nothing -- the elbow is where the fall stops being worth the extra
/// cluster, and that is a judgement rather than a computation. The
/// function returns the curve and declines to pick a point on it.
///
/// # Errors
///
/// As [`kmeans`], or [`SolveError::InvalidArgument`] for an empty range.
pub fn elbow_data(
    data: &[Vec<f64>],
    k_range: &[usize],
    iters: usize,
    rng: &mut Rng,
) -> Result<Vec<(usize, f64)>, SolveError> {
    if k_range.is_empty() {
        return Err(SolveError::InvalidArgument("no cluster counts to try"));
    }
    let mut out = Vec::with_capacity(k_range.len());
    for &k in k_range {
        out.push((k, kmeans(data, k, iters, rng)?.inertia()));
    }
    Ok(out)
}

/// Density-based clustering. Returns a label per point, with `-1` for
/// noise.
///
/// A point is a *core* point if at least `min_pts` points (itself
/// included) lie within `eps`. Clusters are the connected components of
/// the core points, plus the non-core points within `eps` of one.
///
/// Core points are determined by the data alone. Border points are not:
/// one within reach of two clusters joins whichever reaches it first,
/// which depends on the order the points arrive in. That is in the
/// algorithm as defined, not an artefact here -- see the module note.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid dataset, a
/// non-positive `eps`, or `min_pts == 0`.
pub fn dbscan(data: &[Vec<f64>], eps: f64, min_pts: usize) -> Result<Vec<i32>, SolveError> {
    check_data(data)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(SolveError::InvalidArgument("eps must be positive"));
    }
    if min_pts == 0 {
        return Err(SolveError::InvalidArgument("min_pts must be positive"));
    }
    let n = data.len();
    let neighbours: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| distance(&data[i], &data[j]) <= eps).collect())
        .collect();
    let core: Vec<bool> = neighbours.iter().map(|v| v.len() >= min_pts).collect();
    let mut labels = vec![-1i32; n];
    let mut next = 0i32;
    for i in 0..n {
        if !core[i] || labels[i] != -1 {
            continue;
        }
        // Breadth-first over the core points reachable from here.
        let cluster = next;
        next += 1;
        labels[i] = cluster;
        let mut queue = vec![i];
        while let Some(p) = queue.pop() {
            if !core[p] {
                continue;
            }
            for &q in &neighbours[p] {
                if labels[q] == -1 {
                    labels[q] = cluster;
                    if core[q] {
                        queue.push(q);
                    }
                }
            }
        }
    }
    Ok(labels)
}

/// How the distance between two merged clusters is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// The nearest pair. Produces long straggling clusters -- the
    /// chaining effect -- and merge heights that never decrease.
    Single,
    /// The furthest pair. Produces compact clusters, and merge heights
    /// that never decrease.
    Complete,
    /// The mean over all cross pairs. Also monotone.
    Average,
    /// The distance between the clusters' centroids. **Not** monotone:
    /// merging two clusters puts their centre between them, which can be
    /// nearer a third cluster than either original was, and the
    /// dendrogram then inverts.
    Centroid,
}

/// Agglomerative clustering, returning the merges in order as
/// `(left, right, height)`.
///
/// Cluster indices below `n` are the original points; the merge at step
/// `t` creates cluster `n + t`. Heights are Euclidean distances under
/// the chosen linkage.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid dataset or fewer than
/// two points.
pub fn hierarchical_agglomerative(
    data: &[Vec<f64>],
    linkage: Linkage,
) -> Result<Vec<(usize, usize, f64)>, SolveError> {
    check_data(data)?;
    let n = data.len();
    if n < 2 {
        return Err(SolveError::InvalidArgument("need at least two points to merge"));
    }
    // Centroid linkage's update rule is only valid on squared
    // distances, so it works in that space throughout and the height is
    // square-rooted on the way out.
    let squared = linkage == Linkage::Centroid;
    let mut d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            d[i][j] =
                if squared { distance_squared(&data[i], &data[j]) } else { distance(&data[i], &data[j]) };
        }
    }
    let mut active: Vec<usize> = (0..n).collect();
    let mut size = vec![1usize; n];
    let mut name: Vec<usize> = (0..n).collect();
    let mut merges = Vec::with_capacity(n - 1);
    for step in 0..n - 1 {
        // The closest surviving pair.
        let (mut bi, mut bj, mut best) = (0usize, 1usize, f64::INFINITY);
        for a in 0..active.len() {
            for b in (a + 1)..active.len() {
                let v = d[active[a]][active[b]];
                if v < best {
                    best = v;
                    bi = a;
                    bj = b;
                }
            }
        }
        let (i, j) = (active[bi], active[bj]);
        merges.push((name[i], name[j], if squared { best.max(0.0).sqrt() } else { best }));
        // Lance-Williams update of every other cluster's distance to
        // the new one, written into slot i.
        let (ni, nj) = (size[i] as f64, size[j] as f64);
        for &k in &active {
            if k == i || k == j {
                continue;
            }
            let (dki, dkj, dij) = (d[k][i], d[k][j], d[i][j]);
            let updated = match linkage {
                Linkage::Single => dki.min(dkj),
                Linkage::Complete => dki.max(dkj),
                Linkage::Average => (ni * dki + nj * dkj) / (ni + nj),
                Linkage::Centroid => {
                    (ni * dki + nj * dkj) / (ni + nj) - ni * nj * dij / ((ni + nj) * (ni + nj))
                }
            };
            d[k][i] = updated;
            d[i][k] = updated;
        }
        size[i] += size[j];
        name[i] = n + step;
        active.remove(bj);
    }
    Ok(merges)
}

/// Cuts a dendrogram into `k` clusters, returning a label per original
/// point.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] if `k` is zero or exceeds the point
/// count, or if the merge list is not `n - 1` long.
pub fn dendrogram_cut(
    merges: &[(usize, usize, f64)],
    n: usize,
    k: usize,
) -> Result<Vec<usize>, SolveError> {
    if n == 0 || k == 0 || k > n {
        return Err(SolveError::InvalidArgument("the cluster count must lie in 1..=n"));
    }
    if merges.len() + 1 != n {
        return Err(SolveError::DimensionMismatch { expected: n - 1, got: merges.len() + 1 });
    }
    // Apply the first n - k merges and read off the components.
    let mut parent: Vec<usize> = (0..2 * n - 1).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (t, &(a, b, _)) in merges.iter().take(n - k).enumerate() {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        parent[ra] = n + t;
        parent[rb] = n + t;
    }
    let mut seen = std::collections::HashMap::new();
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let root = find(&mut parent, i);
        let next = seen.len();
        labels.push(*seen.entry(root).or_insert(next));
    }
    Ok(labels)
}

/// A fitted Gaussian mixture.
#[derive(Debug, Clone, PartialEq)]
pub struct Gmm {
    /// Mixing weights, summing to one.
    pub weights: Vec<f64>,
    /// Component means.
    pub means: Vec<Vec<f64>>,
    /// Component covariance matrices.
    pub covariances: Vec<Matrix>,
    /// The log-likelihood after each iteration, which is non-decreasing.
    pub log_likelihood_history: Vec<f64>,
}

impl Gmm {
    /// The final log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        *self.log_likelihood_history.last().expect("there is always one iteration")
    }
}

/// A small multiple of the data scale added to each covariance
/// diagonal, so that a component collapsing onto a single point does
/// not produce a singular covariance and an infinite likelihood.
///
/// That collapse is not hypothetical: the likelihood of a Gaussian
/// mixture is unbounded above, and a component sitting exactly on one
/// point with vanishing variance achieves infinity. Every practical
/// implementation regularises, and saying so is better than presenting
/// the maximum as if it existed.
const COVARIANCE_FLOOR: f64 = 1e-6;

/// Fits a Gaussian mixture by expectation-maximisation.
///
/// Each step increases the log-likelihood, which is recorded so that the
/// monotonicity can be checked rather than assumed.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid dataset, `k == 0`,
/// more components than points, or zero iterations;
/// [`SolveError::NotPositiveDefinite`] if a covariance cannot be
/// factored even with the floor applied.
pub fn gaussian_mixture_em(
    data: &[Vec<f64>],
    k: usize,
    iters: usize,
    rng: &mut Rng,
) -> Result<Gmm, SolveError> {
    let dim = check_data(data)?;
    if k == 0 {
        return Err(SolveError::InvalidArgument("need at least one component"));
    }
    if k > data.len() {
        return Err(SolveError::InvalidArgument("more components than points"));
    }
    if iters == 0 {
        return Err(SolveError::InvalidArgument("need at least one iteration"));
    }
    let n = data.len();
    // Start from k-means, which is what everyone does: EM from a random
    // start regularly stalls with a component owning nothing.
    let start = kmeans(data, k, 50, rng)?;
    let mut means = start.centroids;
    // The overall spread, kept only as the unit for the covariance
    // floor and as a fallback for a component that starts with one
    // point.
    let scale: f64 = {
        let mut total = 0.0;
        for j in 0..dim {
            let mean: f64 = data.iter().map(|p| p[j]).sum::<f64>() / n as f64;
            total += data.iter().map(|p| (p[j] - mean) * (p[j] - mean)).sum::<f64>() / n as f64;
        }
        (total / dim as f64).max(1e-12)
    };
    // Weights and covariances from the k-means partition, not from the
    // data as a whole. Starting every covariance at the global spread
    // is the obvious thing and it is wrong: a component wide enough to
    // cover the entire dataset claims responsibility for every point
    // almost equally, so the first maximisation drags all the means
    // back towards the global mean and throws away the initialisation
    // that was just computed. Seeding from the within-cluster scatter
    // keeps the components where k-means put them.
    let mut weights = vec![0.0; k];
    let mut covariances = Vec::with_capacity(k);
    for c in 0..k {
        let members: Vec<&Vec<f64>> = data
            .iter()
            .zip(start.labels.iter())
            .filter(|(_, &l)| l == c)
            .map(|(p, _)| p)
            .collect();
        weights[c] = members.len() as f64 / n as f64;
        let mut cov = Matrix::zeros(dim, dim);
        if members.len() > 1 {
            for p in &members {
                for a in 0..dim {
                    for b in 0..dim {
                        let v = cov.get(a, b)
                            + (p[a] - means[c][a]) * (p[b] - means[c][b]);
                        cov.set(a, b, v);
                    }
                }
            }
            for a in 0..dim {
                for b in 0..dim {
                    cov.set(a, b, cov.get(a, b) / members.len() as f64);
                }
            }
        } else {
            for a in 0..dim {
                cov.set(a, a, scale);
            }
        }
        for a in 0..dim {
            cov.set(a, a, cov.get(a, a) + COVARIANCE_FLOOR * scale);
        }
        covariances.push(cov);
    }
    let mut history = Vec::with_capacity(iters);
    let mut responsibility = vec![vec![0.0; k]; n];
    for _ in 0..iters {
        // Expectation: the posterior over components for each point,
        // computed through a Cholesky so that the quadratic form and
        // the determinant come from the same factorisation.
        let mut factors = Vec::with_capacity(k);
        for c in 0..k {
            factors.push(crate::linalg::cholesky::cholesky(&covariances[c])?);
        }
        let mut total_log = 0.0;
        for (i, p) in data.iter().enumerate() {
            let mut logs = Vec::with_capacity(k);
            for c in 0..k {
                let l = &factors[c];
                let diff: Vec<f64> = p.iter().zip(&means[c]).map(|(a, b)| a - b).collect();
                // Forward substitution gives L^-1 (x - mu); its squared
                // norm is the Mahalanobis distance.
                let mut v = vec![0.0; dim];
                for r in 0..dim {
                    let mut acc = diff[r];
                    for s in 0..r {
                        acc -= l.get(r, s) * v[s];
                    }
                    v[r] = acc / l.get(r, r);
                }
                let quad: f64 = v.iter().map(|x| x * x).sum();
                let log_det: f64 = (0..dim).map(|r| l.get(r, r).ln()).sum::<f64>() * 2.0;
                logs.push(
                    weights[c].max(1e-300).ln()
                        - 0.5 * quad
                        - 0.5 * log_det
                        - 0.5 * dim as f64 * std::f64::consts::TAU.ln(),
                );
            }
            // Log-sum-exp, for the same reason softmax subtracts its
            // maximum: a component the point is far from underflows to
            // zero and takes the whole sum with it.
            let peak = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logs.iter().map(|v| (v - peak).exp()).sum();
            let log_p = peak + sum.ln();
            total_log += log_p;
            for c in 0..k {
                responsibility[i][c] = (logs[c] - log_p).exp();
            }
        }
        history.push(total_log);
        // Maximisation.
        for c in 0..k {
            let mass: f64 = (0..n).map(|i| responsibility[i][c]).sum();
            let safe = mass.max(1e-300);
            weights[c] = mass / n as f64;
            for j in 0..dim {
                means[c][j] =
                    (0..n).map(|i| responsibility[i][c] * data[i][j]).sum::<f64>() / safe;
            }
            let mut cov = Matrix::zeros(dim, dim);
            for i in 0..n {
                let r = responsibility[i][c];
                for a in 0..dim {
                    for b in 0..dim {
                        let v = cov.get(a, b)
                            + r * (data[i][a] - means[c][a]) * (data[i][b] - means[c][b]);
                        cov.set(a, b, v);
                    }
                }
            }
            for a in 0..dim {
                for b in 0..dim {
                    cov.set(a, b, cov.get(a, b) / safe);
                }
                cov.set(a, a, cov.get(a, a) + COVARIANCE_FLOOR * scale);
            }
            covariances[c] = cov;
        }
    }
    Ok(Gmm { weights, means, covariances, log_likelihood_history: history })
}

/// The mean silhouette over all points, in `[-1, 1]`.
///
/// A point's silhouette compares the mean distance to its own cluster
/// against the mean distance to the nearest other cluster. One means
/// the clusters are tight and far apart; zero means the point sits on a
/// boundary; negative means it is closer to another cluster than its
/// own. A point alone in its cluster scores zero by convention -- there
/// is no within-cluster distance to compute, and calling it a perfect
/// one would reward splitting every point off.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid dataset or fewer than
/// two distinct labels; [`SolveError::DimensionMismatch`] on a length
/// mismatch.
pub fn silhouette_score(data: &[Vec<f64>], labels: &[usize]) -> Result<f64, SolveError> {
    check_data(data)?;
    if labels.len() != data.len() {
        return Err(SolveError::DimensionMismatch { expected: data.len(), got: labels.len() });
    }
    let clusters = labels.iter().copied().collect::<std::collections::BTreeSet<_>>();
    if clusters.len() < 2 {
        return Err(SolveError::InvalidArgument("silhouette needs at least two clusters"));
    }
    let n = data.len();
    let mut total = 0.0;
    for i in 0..n {
        let own = labels[i];
        let mut sums = std::collections::BTreeMap::new();
        let mut counts = std::collections::BTreeMap::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            *sums.entry(labels[j]).or_insert(0.0) += distance(&data[i], &data[j]);
            *counts.entry(labels[j]).or_insert(0usize) += 1;
        }
        let a = match counts.get(&own) {
            Some(&c) if c > 0 => sums[&own] / c as f64,
            // A singleton cluster: no within-cluster distance exists.
            _ => {
                continue;
            }
        };
        let b = clusters
            .iter()
            .filter(|&&c| c != own)
            .filter_map(|c| counts.get(c).map(|&m| sums[c] / m as f64))
            .fold(f64::INFINITY, f64::min);
        if !b.is_finite() {
            continue;
        }
        let denominator = a.max(b);
        if denominator > 0.0 {
            total += (b - a) / denominator;
        }
    }
    Ok(total / n as f64)
}

/// The adjusted Rand index between two partitions.
///
/// Counts the pairs of points the two partitions agree about, then
/// subtracts what agreement would be expected by chance from partitions
/// with the same cluster sizes. Identical partitions score exactly one;
/// independent random ones score about zero, and may score below it.
///
/// The correction is what makes the number usable. The unadjusted Rand
/// index of two random partitions of many points into a few clusters is
/// close to one, because most pairs are in different clusters under both
/// and that counts as agreement.
///
/// Invariant under relabelling either partition, which is the minimum a
/// comparison between clusterings has to satisfy: a cluster index is not
/// a name.
///
/// # Errors
///
/// [`SolveError::DimensionMismatch`] if the two have different lengths;
/// [`SolveError::InvalidArgument`] if they are empty.
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> Result<f64, SolveError> {
    if a.len() != b.len() {
        return Err(SolveError::DimensionMismatch { expected: a.len(), got: b.len() });
    }
    if a.is_empty() {
        return Err(SolveError::InvalidArgument("the partitions are empty"));
    }
    let n = a.len() as f64;
    let choose2 = |x: f64| x * (x - 1.0) / 2.0;
    let mut joint = std::collections::HashMap::new();
    let mut left = std::collections::HashMap::new();
    let mut right = std::collections::HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *joint.entry((x, y)).or_insert(0.0) += 1.0;
        *left.entry(x).or_insert(0.0) += 1.0;
        *right.entry(y).or_insert(0.0) += 1.0;
    }
    let index: f64 = joint.values().map(|&v| choose2(v)).sum();
    let sum_a: f64 = left.values().map(|&v| choose2(v)).sum();
    let sum_b: f64 = right.values().map(|&v| choose2(v)).sum();
    let expected = sum_a * sum_b / choose2(n);
    let maximum = 0.5 * (sum_a + sum_b);
    if (maximum - expected).abs() < 1e-300 {
        // Both partitions are trivial in the same way -- all singletons,
        // or all one cluster. There are no pairs to disagree about, so
        // they agree perfectly.
        return Ok(1.0);
    }
    Ok((index - expected) / (maximum - expected))
}

/// The Davies-Bouldin index: the mean over clusters of the worst ratio
/// of within-cluster spread to between-cluster separation.
///
/// Lower is better, and zero is unattainable. Unlike the silhouette it
/// is unbounded above, and unlike the silhouette it uses only the
/// centroids, so it is cheap and it is blind to cluster shape.
///
/// # Errors
///
/// As [`silhouette_score`].
pub fn davies_bouldin(data: &[Vec<f64>], labels: &[usize]) -> Result<f64, SolveError> {
    let dim = check_data(data)?;
    if labels.len() != data.len() {
        return Err(SolveError::DimensionMismatch { expected: data.len(), got: labels.len() });
    }
    let ids: Vec<usize> = labels.iter().copied().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    if ids.len() < 2 {
        return Err(SolveError::InvalidArgument("Davies-Bouldin needs at least two clusters"));
    }
    let mut centroids = Vec::with_capacity(ids.len());
    let mut spreads = Vec::with_capacity(ids.len());
    for &c in &ids {
        let members: Vec<&Vec<f64>> =
            data.iter().zip(labels).filter(|(_, &l)| l == c).map(|(p, _)| p).collect();
        let mut centre = vec![0.0; dim];
        for p in &members {
            for j in 0..dim {
                centre[j] += p[j];
            }
        }
        for v in centre.iter_mut() {
            *v /= members.len() as f64;
        }
        let spread =
            members.iter().map(|p| distance(p, &centre)).sum::<f64>() / members.len() as f64;
        centroids.push(centre);
        spreads.push(spread);
    }
    let mut total = 0.0;
    for i in 0..ids.len() {
        let mut worst = 0.0f64;
        for j in 0..ids.len() {
            if i == j {
                continue;
            }
            let separation = distance(&centroids[i], &centroids[j]);
            if separation > 0.0 {
                worst = worst.max((spreads[i] + spreads[j]) / separation);
            } else {
                worst = f64::INFINITY;
            }
        }
        total += worst;
    }
    Ok(total / ids.len() as f64)
}

/// The indices of the `k` nearest training points to `x`, nearest first.
fn nearest(train: &[Vec<f64>], x: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<(usize, f64)> =
        train.iter().enumerate().map(|(i, p)| (i, distance_squared(p, x))).collect();
    order.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    order.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Classifies `x` by a majority vote of its `k` nearest neighbours.
///
/// Ties are broken towards the smaller label, which is arbitrary but
/// deterministic; an even `k` on a two-class problem can produce them,
/// which is the usual reason to prefer an odd one.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an invalid or empty training set,
/// `k == 0`, or more neighbours than points;
/// [`SolveError::DimensionMismatch`] on a label count or query
/// dimension mismatch.
pub fn knn_classify(
    train: &[Vec<f64>],
    labels: &[usize],
    x: &[f64],
    k: usize,
) -> Result<usize, SolveError> {
    let dim = check_data(train)?;
    if labels.len() != train.len() {
        return Err(SolveError::DimensionMismatch { expected: train.len(), got: labels.len() });
    }
    if x.len() != dim {
        return Err(SolveError::DimensionMismatch { expected: dim, got: x.len() });
    }
    if k == 0 || k > train.len() {
        return Err(SolveError::InvalidArgument("k must lie in 1..=len"));
    }
    let mut votes = std::collections::BTreeMap::new();
    for i in nearest(train, x, k) {
        *votes.entry(labels[i]).or_insert(0usize) += 1;
    }
    Ok(votes
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(label, _)| label)
        .expect("k is at least one"))
}

/// Predicts a value for `x` as the mean of its `k` nearest neighbours'
/// targets.
///
/// # Errors
///
/// As [`knn_classify`].
pub fn knn_regress(
    train: &[Vec<f64>],
    targets: &[f64],
    x: &[f64],
    k: usize,
) -> Result<f64, SolveError> {
    let dim = check_data(train)?;
    if targets.len() != train.len() {
        return Err(SolveError::DimensionMismatch { expected: train.len(), got: targets.len() });
    }
    if x.len() != dim {
        return Err(SolveError::DimensionMismatch { expected: dim, got: x.len() });
    }
    if k == 0 || k > train.len() {
        return Err(SolveError::InvalidArgument("k must lie in 1..=len"));
    }
    let picked = nearest(train, x, k);
    Ok(picked.iter().map(|&i| targets[i]).sum::<f64>() / k as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three well-separated blobs.
    fn blobs(rng: &mut Rng) -> (Vec<Vec<f64>>, Vec<usize>) {
        let centres = [[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]];
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for (label, c) in centres.iter().enumerate() {
            for _ in 0..25 {
                data.push(vec![c[0] + 0.4 * rng.next_gaussian(), c[1] + 0.4 * rng.next_gaussian()]);
                truth.push(label);
            }
        }
        (data, truth)
    }

    #[test]
    fn lloyds_algorithm_never_goes_uphill() {
        // Both halves of an iteration minimise the same objective over
        // one of its arguments, so the inertia is non-increasing. It is
        // a property of the algorithm, not of the data, so it holds for
        // any k on any input.
        let mut rng = Rng::new(0x2a71_0c93);
        let (data, _) = blobs(&mut rng);
        for k in [1usize, 2, 3, 5, 8] {
            let run = kmeans(&data, k, 100, &mut rng).unwrap();
            for w in run.inertia_history.windows(2) {
                assert!(w[1] <= w[0] + 1e-9, "the inertia rose from {} to {}", w[0], w[1]);
            }
            assert!(run.iterations <= 100);
            assert_eq!(run.centroids.len(), k);
            assert_eq!(run.labels.len(), data.len());
            assert!(run.labels.iter().all(|&l| l < k));
            // Every cluster is used: an empty one would mean fewer
            // clusters were returned than asked for.
            for c in 0..k {
                assert!(run.labels.contains(&c), "cluster {c} of {k} was empty");
            }
        }
        // More clusters cannot fit worse.
        let mut previous = f64::INFINITY;
        for (_, inertia) in elbow_data(&data, &[1, 2, 3, 4, 6], 100, &mut rng).unwrap() {
            assert!(inertia <= previous + 1e-6, "inertia rose with k");
            previous = inertia;
        }
    }

    #[test]
    fn k_means_recovers_well_separated_blobs() {
        let mut rng = Rng::new(0x51f0_2b74);
        let (data, truth) = blobs(&mut rng);
        let run = kmeans(&data, 3, 100, &mut rng).unwrap();
        // The labels are arbitrary, so compare through an index that
        // does not care what they are called.
        let agreement = adjusted_rand_index(&run.labels, &truth).unwrap();
        assert!(agreement > 0.95, "the clustering agreed only {agreement} with the truth");
        assert!(silhouette_score(&data, &run.labels).unwrap() > 0.7);
        assert!(davies_bouldin(&data, &run.labels).unwrap() < 0.5);
    }

    #[test]
    fn the_adjusted_index_is_one_for_agreement_and_blind_to_names() {
        let a = vec![0usize, 0, 1, 1, 2, 2, 2];
        assert!((adjusted_rand_index(&a, &a).unwrap() - 1.0).abs() < 1e-12);
        // Relabelling either side changes nothing at all.
        let renamed: Vec<usize> = a.iter().map(|&x| (x + 2) % 3).collect();
        assert!((adjusted_rand_index(&a, &renamed).unwrap() - 1.0).abs() < 1e-12);
        let swapped: Vec<usize> = a.iter().map(|&x| if x == 0 { 1 } else if x == 1 { 0 } else { x }).collect();
        assert!((adjusted_rand_index(&a, &swapped).unwrap() - 1.0).abs() < 1e-12);
        // A partition that splits one cluster scores below one but well
        // above zero.
        let split = vec![0usize, 3, 1, 1, 2, 2, 2];
        let partial = adjusted_rand_index(&a, &split).unwrap();
        assert!(partial < 1.0 && partial > 0.3, "a near miss scored {partial}");
        // Trivial partitions: all in one cluster, or all singletons.
        let one = vec![0usize; 7];
        let singles: Vec<usize> = (0..7).collect();
        assert_eq!(adjusted_rand_index(&one, &one).unwrap(), 1.0);
        assert_eq!(adjusted_rand_index(&singles, &singles).unwrap(), 1.0);
        assert!(adjusted_rand_index(&a, &[0, 1]).is_err());
        assert!(adjusted_rand_index(&[], &[]).is_err());
    }

    #[test]
    fn dbscan_finds_density_and_calls_the_rest_noise() {
        // Two dense blobs and one far outlier. The outlier has no
        // neighbours, so it is noise however the clusters come out.
        let mut data = Vec::new();
        for i in 0..12 {
            let t = i as f64 * 0.1;
            data.push(vec![t, 0.0]);
            data.push(vec![t + 10.0, 0.0]);
        }
        data.push(vec![100.0, 100.0]);
        let labels = dbscan(&data, 0.25, 3).unwrap();
        assert_eq!(labels[labels.len() - 1], -1, "the outlier was not called noise");
        let found: std::collections::BTreeSet<i32> =
            labels.iter().copied().filter(|&l| l >= 0).collect();
        assert_eq!(found.len(), 2, "found {} clusters, wanted 2", found.len());
        // The two lines are separated by ten, so nothing joins them.
        assert_ne!(labels[0], labels[1]);
        // Raising min_pts past the neighbourhood size turns everything
        // into noise; lowering eps below the spacing does too.
        assert!(dbscan(&data, 0.25, 40).unwrap().iter().all(|&l| l == -1));
        assert!(dbscan(&data, 0.05, 3).unwrap().iter().all(|&l| l == -1));
        assert!(dbscan(&data, -1.0, 3).is_err());
        assert!(dbscan(&data, 0.5, 0).is_err());
    }

    #[test]
    fn single_and_complete_linkage_never_invert_but_centroid_can() {
        // Monotone merge heights are what let a dendrogram be drawn
        // without crossings, and centroid linkage does not have them.
        // The classic inversion needs three points in a near-equilateral
        // arrangement: merging the closest pair moves their centre
        // towards the third, which then joins at a smaller height.
        let triangle =
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.8]];
        for linkage in [Linkage::Single, Linkage::Complete, Linkage::Average] {
            let merges = hierarchical_agglomerative(&triangle, linkage).unwrap();
            for w in merges.windows(2) {
                assert!(w[1].2 >= w[0].2 - 1e-12, "{linkage:?} inverted");
            }
        }
        let centroid = hierarchical_agglomerative(&triangle, Linkage::Centroid).unwrap();
        assert!(
            centroid[1].2 < centroid[0].2,
            "centroid linkage did not invert on the arrangement built to make it: {centroid:?}"
        );
        // The merge list has the right shape whatever the linkage.
        let mut rng = Rng::new(0x11b0_3d67);
        let (data, truth) = blobs(&mut rng);
        for linkage in [Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Centroid] {
            let merges = hierarchical_agglomerative(&data, linkage).unwrap();
            assert_eq!(merges.len(), data.len() - 1);
            assert!(merges.iter().all(|m| m.2 >= 0.0));
            let cut = dendrogram_cut(&merges, data.len(), 3).unwrap();
            assert_eq!(cut.len(), data.len());
            assert_eq!(cut.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), 3);
            if linkage == Linkage::Complete || linkage == Linkage::Average {
                let agreement = adjusted_rand_index(&cut, &truth).unwrap();
                assert!(agreement > 0.9, "{linkage:?} agreed only {agreement}");
            }
        }
        assert!(hierarchical_agglomerative(&[vec![1.0]], Linkage::Single).is_err());
    }

    #[test]
    fn the_dendrogram_cut_gives_the_counts_it_is_asked_for() {
        let mut rng = Rng::new(0x77c2_10ea);
        let (data, _) = blobs(&mut rng);
        let merges = hierarchical_agglomerative(&data, Linkage::Complete).unwrap();
        for k in [1usize, 2, 5, 20, data.len()] {
            let cut = dendrogram_cut(&merges, data.len(), k).unwrap();
            let distinct = cut.iter().copied().collect::<std::collections::BTreeSet<_>>();
            assert_eq!(distinct.len(), k, "cutting for {k} gave {}", distinct.len());
        }
        assert!(dendrogram_cut(&merges, data.len(), 0).is_err());
        assert!(dendrogram_cut(&merges, data.len(), data.len() + 1).is_err());
        assert!(dendrogram_cut(&merges, data.len() + 1, 2).is_err());
    }

    /// Three blobs that overlap, so that soft assignment differs from
    /// hard and expectation-maximisation has something to do. On
    /// well-separated blobs it starts at the answer and its likelihood
    /// history is flat, which makes a monotonicity test vacuous.
    fn overlapping(rng: &mut Rng) -> (Vec<Vec<f64>>, Vec<usize>) {
        let centres = [[0.0, 0.0], [3.0, 0.0], [1.5, 2.6]];
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for (label, c) in centres.iter().enumerate() {
            for _ in 0..40 {
                data.push(vec![c[0] + 1.1 * rng.next_gaussian(), c[1] + 1.1 * rng.next_gaussian()]);
                truth.push(label);
            }
        }
        (data, truth)
    }

    #[test]
    fn a_restart_finds_what_a_single_run_can_miss() {
        // Lloyd's algorithm is local. On these blobs a single run lands
        // on a badly wrong optimum about once in two hundred, and no
        // number of iterations escapes it because no single point wants
        // to move. The restarted version is the minimum over runs, so it
        // is never worse than any of them.
        let mut rng = Rng::new(0x2b0c_7741);
        let (data, _) = blobs(&mut rng);
        // Best-of-ten is a minimum over its own draws, so it beats a
        // single run on average rather than every time. Averages are
        // what is compared.
        let mut single = 0.0;
        let mut restarted = 0.0;
        for _ in 0..20 {
            let one = kmeans_once(&data, 3, 50, &mut rng).unwrap();
            for w in one.inertia_history.windows(2) {
                assert!(w[1] <= w[0] + 1e-9, "a single run went uphill");
            }
            single += one.inertia();
            restarted += kmeans(&data, 3, 50, &mut rng).unwrap().inertia();
        }
        assert!(restarted <= single, "restarting did not help: {restarted} vs {single}");
    }

    #[test]
    fn expectation_maximisation_never_goes_downhill() {
        let mut rng = Rng::new(0x4a03_9c15);
        let (data, truth) = overlapping(&mut rng);
        let fit = gaussian_mixture_em(&data, 3, 40, &mut rng).unwrap();
        // The run has to do something, or the monotonicity below is
        // being asserted about a constant sequence.
        assert!(
            fit.log_likelihood() > fit.log_likelihood_history[0] + 1.0,
            "the likelihood never moved: {:?}",
            fit.log_likelihood_history
        );
        for w in fit.log_likelihood_history.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "the likelihood fell from {} to {}", w[0], w[1]);
        }
        assert!((fit.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(fit.weights.iter().all(|&w| w >= 0.0));
        assert_eq!(fit.means.len(), 3);
        assert_eq!(fit.covariances.len(), 3);
        // Each component should own roughly a third of the points. The
        // blobs overlap, so "roughly" is the honest word -- a point in
        // the overlap genuinely belongs to both.
        for w in &fit.weights {
            assert!((w - 1.0 / 3.0).abs() < 0.15, "a component took {w}");
        }
        // Hard assignment from the fit still recovers most of the
        // structure, though not all of it: overlapping blobs are not
        // separable and a perfect score would mean the data was easier
        // than it looks.
        let assigned: Vec<usize> = data
            .iter()
            .map(|p| {
                (0..3)
                    .min_by(|&a, &b| {
                        distance(p, &fit.means[a]).total_cmp(&distance(p, &fit.means[b]))
                    })
                    .unwrap()
            })
            .collect();
        let agreement = adjusted_rand_index(&assigned, &truth).unwrap();
        assert!(agreement > 0.3, "the mixture recovered only {agreement} of the structure");
        // And on separated blobs it recovers them completely.
        let (clean, clean_truth) = blobs(&mut rng);
        let sharp = gaussian_mixture_em(&clean, 3, 40, &mut rng).unwrap();
        let hard: Vec<usize> = clean
            .iter()
            .map(|p| {
                (0..3)
                    .min_by(|&a, &b| {
                        distance(p, &sharp.means[a]).total_cmp(&distance(p, &sharp.means[b]))
                    })
                    .unwrap()
            })
            .collect();
        assert!(adjusted_rand_index(&hard, &clean_truth).unwrap() > 0.95);
    }

    #[test]
    fn the_silhouette_is_bounded_and_rewards_separation() {
        let mut rng = Rng::new(0x6f18_2c40);
        let (data, truth) = blobs(&mut rng);
        let good = silhouette_score(&data, &truth).unwrap();
        assert!((-1.0..=1.0).contains(&good), "out of range at {good}");
        assert!(good > 0.7, "well-separated blobs scored only {good}");
        // A deliberately wrong clustering scores far worse.
        let scrambled: Vec<usize> = (0..data.len()).map(|i| i % 3).collect();
        let bad = silhouette_score(&data, &scrambled).unwrap();
        assert!((-1.0..=1.0).contains(&bad));
        assert!(bad < 0.1, "a scrambled clustering scored {bad}");
        assert!(good > bad);
        // Davies-Bouldin runs the other way: lower is better.
        assert!(davies_bouldin(&data, &truth).unwrap() < davies_bouldin(&data, &scrambled).unwrap());
        assert!(silhouette_score(&data, &vec![0; data.len()]).is_err());
        assert!(silhouette_score(&data, &truth[..3]).is_err());
        assert!(davies_bouldin(&data, &vec![0; data.len()]).is_err());
    }

    #[test]
    fn one_nearest_neighbour_reproduces_its_training_set() {
        // With k = 1 the nearest point to a training point is itself,
        // at distance zero, so the label comes back exactly. This is
        // also the reason a 1-NN training error of zero says nothing.
        let mut rng = Rng::new(0x0c94_71fe);
        let (data, truth) = blobs(&mut rng);
        for (i, p) in data.iter().enumerate() {
            assert_eq!(knn_classify(&data, &truth, p, 1).unwrap(), truth[i], "point {i}");
        }
        let targets: Vec<f64> = data.iter().map(|p| p[0] * 2.0 - p[1]).collect();
        for (i, p) in data.iter().enumerate() {
            let got = knn_regress(&data, &targets, p, 1).unwrap();
            assert!((got - targets[i]).abs() < 1e-12, "point {i}");
        }
        // Larger k smooths: the regression of a constant is that
        // constant whatever k is.
        let flat = vec![5.0; data.len()];
        for k in [1usize, 3, 10] {
            assert!((knn_regress(&data, &flat, &[1.0, 1.0], k).unwrap() - 5.0).abs() < 1e-12);
        }
        assert!(knn_classify(&data, &truth, &[0.0, 0.0], 0).is_err());
        assert!(knn_classify(&data, &truth, &[0.0, 0.0], data.len() + 1).is_err());
        assert!(knn_classify(&data, &truth, &[0.0], 1).is_err());
        assert!(knn_classify(&data, &truth[..2], &[0.0, 0.0], 1).is_err());
        assert!(knn_regress(&data, &targets[..2], &[0.0, 0.0], 1).is_err());
        assert!(knn_regress(&data, &targets, &[0.0], 1).is_err());
    }

    #[test]
    fn the_clusterers_refuse_impossible_arguments() {
        let mut rng = Rng::new(3);
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        assert!(kmeans(&[], 1, 10, &mut rng).is_err());
        assert!(kmeans(&[vec![], vec![]], 1, 10, &mut rng).is_err());
        assert!(kmeans(&[vec![1.0], vec![1.0, 2.0]], 1, 10, &mut rng).is_err());
        assert!(kmeans(&[vec![f64::NAN]], 1, 10, &mut rng).is_err());
        assert!(kmeans(&data, 0, 10, &mut rng).is_err());
        assert!(kmeans(&data, 9, 10, &mut rng).is_err());
        assert!(kmeans(&data, 2, 0, &mut rng).is_err());
        assert!(kmeans_pp_init(&data, 0, &mut rng).is_err());
        assert!(kmeans_pp_init(&data, 9, &mut rng).is_err());
        assert!(elbow_data(&data, &[], 10, &mut rng).is_err());
        assert!(gaussian_mixture_em(&data, 0, 5, &mut rng).is_err());
        assert!(gaussian_mixture_em(&data, 9, 5, &mut rng).is_err());
        assert!(gaussian_mixture_em(&data, 2, 0, &mut rng).is_err());
        // Coincident points make k-means++ fall back to a uniform draw
        // rather than dividing by a zero total weight.
        let identical = vec![vec![1.0, 1.0]; 5];
        let centres = kmeans_pp_init(&identical, 3, &mut rng).unwrap();
        assert_eq!(centres.len(), 3);
        assert!(kmeans(&identical, 2, 10, &mut rng).is_ok());
    }
}
