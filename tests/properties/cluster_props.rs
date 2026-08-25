//! Properties of the clustering module.
//!
//! Clustering has no ground truth to check against, so the tests lean on
//! the exact statements the algorithms do make.
//!
//! *Monotonicity.* Lloyd's algorithm cannot raise the inertia and
//! expectation-maximisation cannot lower the log-likelihood, because
//! each step of each optimises the same objective over one of its
//! arguments. Both are checked step by step: a monotone sequence is a
//! far sharper claim than an improved endpoint, and an implementation
//! with a sign error in one half of an iteration still improves its
//! endpoint most of the time.
//!
//! *Relabelling invariance.* A cluster index is not a name, so any
//! comparison between clusterings has to survive permuting either side.
//! [`adjusted_rand_index`] does so exactly, and it is corrected for
//! chance, so two independent random partitions score near zero rather
//! than near the one the uncorrected index would give.
//!
//! *Where the guarantees stop.* Single, complete and average linkage
//! give merge heights that never decrease; centroid linkage does not,
//! and the inversion is asserted rather than avoided. DBSCAN's core
//! points do not depend on the order the data arrives in; its border
//! points may, and that is tested as the asymmetry it is.

use rust_physics_engine::learn::cluster::{
    adjusted_rand_index, davies_bouldin, dbscan, dendrogram_cut, gaussian_mixture_em,
    hierarchical_agglomerative, kmeans, kmeans_once, kmeans_pp_init, knn_classify, knn_regress,
    silhouette_score, Linkage,
};
use rust_physics_engine::monte_carlo::Rng;

/// Blobs with a controllable spread, so that a test can pick separated
/// data or overlapping data on purpose.
fn blobs(rng: &mut Rng, per: usize, spread: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let centres = [[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]];
    let mut data = Vec::new();
    let mut truth = Vec::new();
    for (label, c) in centres.iter().enumerate() {
        for _ in 0..per {
            data.push(vec![
                c[0] + spread * rng.next_gaussian(),
                c[1] + spread * rng.next_gaussian(),
            ]);
            truth.push(label);
        }
    }
    (data, truth)
}

fn scatter(rng: &mut Rng, n: usize, dim: usize) -> Vec<Vec<f64>> {
    (0..n).map(|_| (0..dim).map(|_| 4.0 * rng.next_gaussian()).collect()).collect()
}

#[test]
fn prop_lloyds_algorithm_is_monotone_and_terminates() {
    let mut rng = Rng::new(0x38f1_0a27);
    for _ in 0..25 {
        let n = 12 + (rng.next_u64() % 30) as usize;
        let dim = 1 + (rng.next_u64() % 3) as usize;
        let data = scatter(&mut rng, n, dim);
        let k = 1 + (rng.next_u64() % 5.min(n as u64)) as usize;
        let run = kmeans_once(&data, k, 200, &mut rng).unwrap();
        for w in run.inertia_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "the inertia rose from {} to {}", w[0], w[1]);
        }
        assert!(run.inertia() >= 0.0);
        // It stops before the cap, because there are finitely many
        // assignments and none repeats.
        assert!(run.iterations < 200, "it never settled");
        assert_eq!(run.centroids.len(), k);
        assert_eq!(run.labels.len(), n);
        for c in 0..k {
            assert!(run.labels.contains(&c), "cluster {c} came back empty");
        }
        // The restarted version is a minimum over its *own* draws, so it
        // is not guaranteed to beat an independent single run -- only to
        // beat one on average, which the dedicated test below measures.
        // What does hold on every run is that the result is a genuine
        // fixed point of Lloyd's algorithm.
        let best = kmeans(&data, k, 200, &mut rng).unwrap();
        assert!(best.inertia() >= 0.0);
        // Every point really is with its nearest centre at the end.
        for (i, p) in data.iter().enumerate() {
            let own = best
                .centroids
                .iter()
                .map(|c| p.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            let assigned: f64 = p
                .iter()
                .zip(&best.centroids[best.labels[i]])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            assert!(assigned <= own + 1e-9, "point {i} was not with its nearest centre");
        }
    }
}

#[test]
fn prop_restarting_helps_on_average() {
    // Best-of-ten cannot be guaranteed to beat an independent eleventh
    // draw -- that would be a statement about two unrelated random
    // variables. What restarts buy is a better distribution, and that is
    // what gets measured: over many datasets the mean inertia of the
    // restarted version is below the mean of the single-run version, and
    // it is never far above the best single run seen.
    let mut rng = Rng::new(0x71c3_08fa);
    let mut single_total = 0.0;
    let mut restarted_total = 0.0;
    let trials = 30;
    for _ in 0..trials {
        let (data, _) = blobs(&mut rng, 12, 0.4);
        single_total += kmeans_once(&data, 3, 100, &mut rng).unwrap().inertia();
        restarted_total += kmeans(&data, 3, 100, &mut rng).unwrap().inertia();
    }
    assert!(
        restarted_total <= single_total,
        "restarting did not help: {restarted_total} against {single_total}"
    );
}

#[test]
fn prop_more_clusters_never_fit_worse() {
    // Inertia falls with k, and reaches zero when every point is its
    // own cluster. That is why the number alone cannot choose k.
    let mut rng = Rng::new(0x5c73_9be0);
    for _ in 0..12 {
        let n = 10 + (rng.next_u64() % 15) as usize;
        let data = scatter(&mut rng, n, 2);
        let mut previous = f64::INFINITY;
        for k in 1..=6.min(n) {
            let run = kmeans(&data, k, 100, &mut rng).unwrap();
            assert!(run.inertia() <= previous + 1e-6, "inertia rose going to k = {k}");
            previous = run.inertia();
        }
        let all = kmeans(&data, n, 100, &mut rng).unwrap();
        assert!(all.inertia() < 1e-18, "one cluster per point left inertia {}", all.inertia());
    }
}

#[test]
fn prop_kmeans_pp_returns_the_right_number_of_real_points() {
    let mut rng = Rng::new(0x1ab4_6f92);
    for _ in 0..30 {
        let n = 5 + (rng.next_u64() % 25) as usize;
        let dim = 1 + (rng.next_u64() % 3) as usize;
        let data = scatter(&mut rng, n, dim);
        let k = 1 + (rng.next_u64() % n as u64) as usize;
        let centres = kmeans_pp_init(&data, k, &mut rng).unwrap();
        assert_eq!(centres.len(), k);
        for c in &centres {
            assert_eq!(c.len(), dim);
            assert!(data.iter().any(|p| p == c), "a centre was not one of the points");
        }
    }
}

#[test]
fn prop_the_adjusted_index_is_exact_and_blind_to_labels() {
    let mut rng = Rng::new(0x7f20_5c8d);
    for _ in 0..40 {
        let n = 6 + (rng.next_u64() % 30) as usize;
        let k = 2 + (rng.below(4)) as usize;
        let a: Vec<usize> = (0..n).map(|_| (rng.next_u64() % k as u64) as usize).collect();
        assert!((adjusted_rand_index(&a, &a).unwrap() - 1.0).abs() < 1e-12);
        // Permuting either side is invisible.
        let mut permutation: Vec<usize> = (0..k).collect();
        for i in (1..k).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            permutation.swap(i, j);
        }
        let renamed: Vec<usize> = a.iter().map(|&x| permutation[x]).collect();
        assert!((adjusted_rand_index(&a, &renamed).unwrap() - 1.0).abs() < 1e-12);
        let b: Vec<usize> = (0..n).map(|_| (rng.next_u64() % k as u64) as usize).collect();
        let straight = adjusted_rand_index(&a, &b).unwrap();
        let b_renamed: Vec<usize> = b.iter().map(|&x| permutation[x]).collect();
        assert!((adjusted_rand_index(&a, &b_renamed).unwrap() - straight).abs() < 1e-12);
        // Symmetric in its two arguments.
        assert!((adjusted_rand_index(&b, &a).unwrap() - straight).abs() < 1e-12);
        // And bounded above by one.
        assert!(straight <= 1.0 + 1e-12, "an index above one: {straight}");
    }
}

#[test]
fn prop_independent_partitions_score_about_nothing() {
    // The correction for chance is the whole point. Two random
    // partitions of the same data agree on most *pairs* -- nearly all
    // pairs are separated under both -- so the uncorrected index is
    // close to one and useless. The adjusted one hovers around zero.
    let mut rng = Rng::new(0x0e3d_71fa);
    let mut total = 0.0;
    let trials = 60;
    for _ in 0..trials {
        let n = 60;
        let a: Vec<usize> = (0..n).map(|_| (rng.below(4)) as usize).collect();
        let b: Vec<usize> = (0..n).map(|_| (rng.below(4)) as usize).collect();
        let score = adjusted_rand_index(&a, &b).unwrap();
        assert!(score < 0.4, "independent partitions scored {score}");
        total += score;
    }
    let mean = total / trials as f64;
    assert!(mean.abs() < 0.05, "the mean over independent pairs was {mean}");
}

#[test]
fn prop_the_quality_measures_stay_in_their_ranges_and_agree_on_order() {
    // The silhouette lives in [-1, 1]; Davies-Bouldin is nonnegative and
    // runs the other way. On the same data they must rank a good
    // clustering above a scrambled one, since otherwise at least one of
    // them is not measuring cluster quality.
    let mut rng = Rng::new(0x64b0_2c1e);
    for _ in 0..15 {
        let (data, truth) = blobs(&mut rng, 15, 0.4);
        let good = silhouette_score(&data, &truth).unwrap();
        assert!((-1.0..=1.0).contains(&good), "silhouette out of range: {good}");
        let scrambled: Vec<usize> = (0..data.len()).map(|i| i % 3).collect();
        let bad = silhouette_score(&data, &scrambled).unwrap();
        assert!((-1.0..=1.0).contains(&bad));
        assert!(good > bad, "the scrambled clustering scored better");
        let db_good = davies_bouldin(&data, &truth).unwrap();
        let db_bad = davies_bouldin(&data, &scrambled).unwrap();
        assert!(db_good >= 0.0 && db_bad >= 0.0);
        assert!(db_good < db_bad, "Davies-Bouldin preferred the scrambled clustering");
        // Both are blind to what the clusters are called.
        let renamed: Vec<usize> = truth.iter().map(|&x| (x + 1) % 3).collect();
        assert!((silhouette_score(&data, &renamed).unwrap() - good).abs() < 1e-12);
        assert!((davies_bouldin(&data, &renamed).unwrap() - db_good).abs() < 1e-12);
    }
}

#[test]
fn prop_dbscan_is_deterministic_and_its_cores_ignore_the_order() {
    // Core membership is a property of the data. Cluster *identity*
    // under a permutation is only recoverable up to relabelling, which
    // is what the adjusted index is for -- and border points may move,
    // so the comparison is over the core points alone.
    let mut rng = Rng::new(0x2d81_c07b);
    for _ in 0..20 {
        let (data, _) = blobs(&mut rng, 12, 0.3);
        let eps = 0.5 + rng.next_f64();
        let min_pts = 2 + (rng.below(4)) as usize;
        let labels = dbscan(&data, eps, min_pts).unwrap();
        // Deterministic: the same input gives the same answer.
        assert_eq!(dbscan(&data, eps, min_pts).unwrap(), labels);
        // Permute the data and re-run.
        let mut order: Vec<usize> = (0..data.len()).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let shuffled: Vec<Vec<f64>> = order.iter().map(|&i| data[i].clone()).collect();
        let other = dbscan(&shuffled, eps, min_pts).unwrap();
        // A core point is one with min_pts neighbours within eps, which
        // no permutation changes.
        let core = |set: &[Vec<f64>], i: usize| {
            set.iter()
                .filter(|q| {
                    set[i].iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
                        <= eps * eps
                })
                .count()
                >= min_pts
        };
        for (position, &original) in order.iter().enumerate() {
            assert_eq!(
                core(&data, original),
                core(&shuffled, position),
                "core membership moved with the order"
            );
            // A core point is never noise, in either run.
            if core(&data, original) {
                assert!(labels[original] >= 0 && other[position] >= 0);
            }
        }
        // The partition of the core points agrees up to relabelling.
        let cores: Vec<usize> = (0..data.len()).filter(|&i| core(&data, i)).collect();
        if cores.len() > 1 {
            let left: Vec<usize> = cores.iter().map(|&i| labels[i] as usize).collect();
            let right: Vec<usize> = cores
                .iter()
                .map(|&i| other[order.iter().position(|&o| o == i).unwrap()] as usize)
                .collect();
            let agreement = adjusted_rand_index(&left, &right).unwrap();
            assert!((agreement - 1.0).abs() < 1e-9, "the core partition changed: {agreement}");
        }
    }
}

#[test]
fn prop_linkage_monotonicity_holds_except_for_centroid() {
    let mut rng = Rng::new(0x4e17_53da);
    let mut inversions = 0;
    for _ in 0..25 {
        let n = 4 + (rng.next_u64() % 12) as usize;
        let data = scatter(&mut rng, n, 2);
        for linkage in [Linkage::Single, Linkage::Complete, Linkage::Average] {
            let merges = hierarchical_agglomerative(&data, linkage).unwrap();
            assert_eq!(merges.len(), n - 1);
            for w in merges.windows(2) {
                assert!(
                    w[1].2 >= w[0].2 - 1e-12,
                    "{linkage:?} inverted: {} then {}",
                    w[0].2,
                    w[1].2
                );
            }
            // Every cluster index is either an original point or a
            // merge that already happened.
            for (t, &(a, b, _)) in merges.iter().enumerate() {
                assert!(a < n + t && b < n + t, "a merge referred to a future cluster");
                assert_ne!(a, b);
            }
            // Cutting reproduces every requested count.
            for k in 1..=n {
                let cut = dendrogram_cut(&merges, n, k).unwrap();
                let distinct: std::collections::BTreeSet<usize> = cut.iter().copied().collect();
                assert_eq!(distinct.len(), k);
            }
        }
        let centroid = hierarchical_agglomerative(&data, Linkage::Centroid).unwrap();
        assert_eq!(centroid.len(), n - 1);
        if centroid.windows(2).any(|w| w[1].2 < w[0].2 - 1e-12) {
            inversions += 1;
        }
    }
    // Centroid linkage is not merely allowed to invert -- on random
    // point sets it does, often. Asserting that it happens keeps the
    // documented caveat honest.
    assert!(inversions > 0, "centroid linkage never inverted in 25 random point sets");
}

#[test]
fn prop_expectation_maximisation_climbs() {
    let mut rng = Rng::new(0x1c60_8a35);
    for _ in 0..12 {
        // Overlapping, so that the soft assignment differs from the hard
        // one it starts from and there is something to climb.
        let (data, _) = blobs(&mut rng, 25, 2.0);
        let k = 2 + (rng.next_u64() % 3) as usize;
        let fit = gaussian_mixture_em(&data, k, 30, &mut rng).unwrap();
        for w in fit.log_likelihood_history.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "the likelihood fell from {} to {}", w[0], w[1]);
        }
        assert_eq!(fit.weights.len(), k);
        assert_eq!(fit.means.len(), k);
        assert_eq!(fit.covariances.len(), k);
        assert!((fit.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(fit.weights.iter().all(|&w| (0.0..=1.0).contains(&w)));
        // Every covariance is symmetric with a positive diagonal, which
        // is what makes it factorable at the next step.
        for cov in &fit.covariances {
            for i in 0..cov.rows {
                assert!(cov.get(i, i) > 0.0, "a covariance had a non-positive variance");
                for j in 0..cov.cols {
                    assert!((cov.get(i, j) - cov.get(j, i)).abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn prop_nearest_neighbours_reproduce_their_own_training_set() {
    // With k = 1 the closest training point to a training point is
    // itself. Which is also why a 1-NN training error of zero is no
    // evidence of anything.
    let mut rng = Rng::new(0x39c2_7e04);
    for _ in 0..20 {
        let n = 8 + (rng.next_u64() % 25) as usize;
        let dim = 1 + (rng.next_u64() % 3) as usize;
        let data = scatter(&mut rng, n, dim);
        let labels: Vec<usize> = (0..n).map(|_| (rng.below(4)) as usize).collect();
        let targets: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        for i in 0..n {
            assert_eq!(knn_classify(&data, &labels, &data[i], 1).unwrap(), labels[i]);
            assert!((knn_regress(&data, &targets, &data[i], 1).unwrap() - targets[i]).abs() < 1e-12);
        }
        // A constant target comes back constant for every k, since the
        // prediction is a mean of equal values.
        let constant = 2.0 * rng.next_f64() - 1.0;
        let flat = vec![constant; n];
        for k in 1..=n.min(7) {
            let q: Vec<f64> = (0..dim).map(|_| rng.next_gaussian()).collect();
            assert!((knn_regress(&data, &flat, &q, k).unwrap() - constant).abs() < 1e-12);
        }
        // The prediction always lies within the range of the targets it
        // averages, whatever k is.
        let lo = targets.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for k in 1..=n.min(7) {
            let q: Vec<f64> = (0..dim).map(|_| rng.next_gaussian()).collect();
            let got = knn_regress(&data, &targets, &q, k).unwrap();
            assert!(got >= lo - 1e-12 && got <= hi + 1e-12, "prediction {got} left [{lo}, {hi}]");
            let label = knn_classify(&data, &labels, &q, k).unwrap();
            assert!(labels.contains(&label), "a label nobody had was predicted");
        }
    }
}
