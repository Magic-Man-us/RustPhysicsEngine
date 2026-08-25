//! Properties of the statistical-mechanics modules.
//!
//! Lattice statistical mechanics is unusually rich in exact statements, and
//! they are exact for structural reasons rather than numerical ones: a
//! percolation cluster grows monotonically as sites are added, a winding
//! number around a torus sums to zero, a dimer count on a two-row strip obeys
//! the Fibonacci recurrence, a global spin flip is a symmetry of the
//! zero-field Hamiltonian. None of those depend on how well a sampler has
//! converged. Where a test does have to lean on sampling, it is checked
//! against an exact enumeration of the same system rather than against a
//! remembered number.

use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::statistical_mechanics::ising::{
    binder_crossing, canonical_from_dos, fluctuation_dissipation_check, ising_1d_exact,
    ising_tc_exact, onsager_magnetization, partition_function_exact_small, potts_tc_exact,
    thermodynamics_exact_small, Ising2D, IsingStats, Potts2D, XyModel2D,
};
use rust_physics_engine::statistical_mechanics::lattice_models::{
    cluster_size_distribution, connective_constant_estimate, dimer_count_kasteleyn,
    flory_exponent_estimate, growth_exponent_estimate, interface_width, percolation_site,
    polymer_end_to_end, power_law_fit_clauset, random_walk_lattice, return_probability,
    saw_sample_rosenbluth, self_avoiding_walk_count,
};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ---------------------------------------------------------------------------
// Percolation
// ---------------------------------------------------------------------------

#[test]
fn prop_percolation_is_monotone_in_the_occupation_under_a_common_random_number() {
    // The generator draws one uniform per site, so replaying the same seed at
    // two occupations couples the two lattices exactly: a site occupied at
    // the lower probability is occupied at the higher one. Adding sites can
    // only join clusters, never split them, so spanning is monotone -- and
    // under this coupling that is a statement about *each pair of lattices*,
    // not about an average over many. A sampler that got the comparison
    // backwards, or that redrew rather than thresholded, would fail here on
    // essentially every seed.
    for seed in 0..12u64 {
        let n = 24;
        let mut previous_grid: Option<Vec<bool>> = None;
        let mut previous_spans = false;
        for step in 0..=10 {
            let p = f64::from(step) / 10.0;
            let mut rng = Rng::new(0x_5EED_0000 + seed);
            let (grid, spans) = percolation_site(n, p, &mut rng).unwrap();
            if let Some(before) = &previous_grid {
                for (index, (old, new)) in before.iter().zip(&grid).enumerate() {
                    assert!(
                        !*old || *new,
                        "site {index} lost its occupation as p rose to {p}"
                    );
                }
                assert!(
                    !previous_spans || spans,
                    "spanning was lost as p rose to {p} on seed {seed}"
                );
            }
            previous_grid = Some(grid);
            previous_spans = spans;
        }
    }
}

#[test]
fn prop_the_extreme_occupations_settle_percolation_outright() {
    // No randomness survives at either end: an empty lattice cannot span and
    // a full one must.
    let mut rng = Rng::new(0x_5EED_0001);
    for n in [4usize, 9, 16, 33] {
        let (empty, spans_empty) = percolation_site(n, 0.0, &mut rng).unwrap();
        assert!(empty.iter().all(|s| !*s));
        assert!(!spans_empty, "an empty lattice spanned at n = {n}");
        let (full, spans_full) = percolation_site(n, 1.0, &mut rng).unwrap();
        assert!(full.iter().all(|s| *s));
        assert!(spans_full, "a full lattice failed to span at n = {n}");
        assert_eq!(cluster_size_distribution(&full, n).unwrap(), vec![n * n]);
        assert!(cluster_size_distribution(&empty, n).unwrap().is_empty());
    }
}

#[test]
fn prop_the_cluster_decomposition_partitions_the_occupied_sites() {
    // The clusters are a partition, so their sizes sum to the occupied count
    // however the lattice came out; and a spanning cluster has to reach from
    // one boundary to the other, which takes at least n sites.
    let mut rng = Rng::new(0x_5EED_0002);
    for trial in 0..40 {
        let n = 12 + trial % 9;
        let p = 0.1 + 0.02 * (trial % 40) as f64;
        let (grid, spans) = percolation_site(n, p, &mut rng).unwrap();
        let occupied = grid.iter().filter(|s| **s).count();
        let sizes = cluster_size_distribution(&grid, n).unwrap();
        assert_eq!(sizes.iter().sum::<usize>(), occupied, "the sizes do not partition");
        assert!(sizes.iter().all(|s| *s >= 1), "an empty cluster was reported");
        for pair in sizes.windows(2) {
            assert!(pair[0] >= pair[1], "the sizes are not descending");
        }
        if spans {
            assert!(
                sizes.first().copied().unwrap_or(0) >= n,
                "a spanning cluster of fewer than {n} sites at p = {p}"
            );
        }
    }
    assert!(cluster_size_distribution(&[true, false, true], 2).is_err());
}

// ---------------------------------------------------------------------------
// Walks
// ---------------------------------------------------------------------------

#[test]
fn prop_the_walk_counts_sit_between_their_two_elementary_bounds() {
    // A self-avoiding walk is in particular non-reversing, giving at most
    // four choices then three; and it is at least as free as a walk confined
    // to the two increasing directions, which can never self-intersect. Both
    // bounds are combinatorial, not empirical.
    assert_eq!(self_avoiding_walk_count(0).unwrap(), 1);
    for n in 1..=15usize {
        let c = self_avoiding_walk_count(n).unwrap();
        assert!(
            c <= 4 * 3u64.pow(n as u32 - 1),
            "the count at n = {n} exceeds the non-reversing bound"
        );
        assert!(c >= 2u64.pow(n as u32), "the count at n = {n} is below the directed walk");
    }
    // Submultiplicativity: a walk of n + m steps splits into a walk of n and
    // a translate of a walk of m, and not every such pair is self-avoiding as
    // a whole. This is the inequality that makes the connective constant
    // exist at all, by Fekete's lemma.
    for n in 1..=8usize {
        for m in 1..=8usize {
            let joint = self_avoiding_walk_count(n + m).unwrap();
            let split = self_avoiding_walk_count(n).unwrap() * self_avoiding_walk_count(m).unwrap();
            assert!(joint <= split, "submultiplicativity fails at n = {n}, m = {m}");
        }
    }
}

#[test]
fn prop_the_connective_constant_estimate_improves_on_the_ratio_it_is_built_from() {
    // Checked at several truncations rather than one: an extrapolation that
    // happened to land well at a single length would be a coincidence, while
    // one that beats the raw ratio at every length is doing arithmetic.
    let counts: Vec<u64> = (0..=18).map(|n| self_avoiding_walk_count(n).unwrap()).collect();
    const TRUE_MU: f64 = 2.638_158;
    for last in [10usize, 12, 14, 16, 18] {
        let mu = connective_constant_estimate(&counts[..=last]).unwrap();
        let raw = counts[last] as f64 / counts[last - 1] as f64;
        assert!(
            (mu - TRUE_MU).abs() < (raw - TRUE_MU).abs(),
            "at n = {last} the extrapolation {mu} is no better than the ratio {raw}"
        );
        assert!((mu - TRUE_MU).abs() < 0.02, "the estimate at n = {last} came out {mu}");
    }
    // And the estimate settles as more terms are added, rather than drifting.
    let short = connective_constant_estimate(&counts[..=12]).unwrap();
    let long = connective_constant_estimate(&counts).unwrap();
    assert!(
        (long - TRUE_MU).abs() <= (short - TRUE_MU).abs(),
        "adding terms made the estimate worse: {short} then {long}"
    );
}

#[test]
fn prop_every_rosenbluth_sample_is_a_walk_and_its_weight_is_the_product_of_its_choices() {
    // The weight is only unbiased if it is exactly the number of options
    // taken at each step, so it must factor into integers between one and
    // four -- and the path must actually be self-avoiding, which is the
    // constraint the weighting exists to enforce.
    let mut rng = Rng::new(0x_5EED_0010);
    for _ in 0..200 {
        let n = 12;
        let (path, weight) = saw_sample_rosenbluth(n, &mut rng).unwrap();
        assert!(weight >= 0.0);
        let mut seen = std::collections::HashSet::new();
        for site in &path {
            assert!(seen.insert(*site), "the path revisits {site:?}");
        }
        for pair in path.windows(2) {
            let d = (pair[1].0 - pair[0].0).abs() + (pair[1].1 - pair[0].1).abs();
            assert_eq!(d, 1, "a step of length {d} is not a lattice move");
        }
        if weight > 0.0 {
            assert_eq!(path.len(), n + 1, "a surviving walk is short");
            // The weight is a product of n integers in 1..=4, so its
            // logarithm base-4 bounds it and it is an exact integer.
            assert!(weight <= 4.0 * 3f64.powi(n as i32 - 1) + 0.5);
            assert!(weight >= 1.0);
            assert!(close(weight, weight.round(), 1e-6), "the weight {weight} is not an integer");
        } else {
            assert!(path.len() <= n, "a trapped walk reached full length");
        }
    }
}

#[test]
fn prop_rosenbluth_weighting_recovers_the_exact_count_and_discarding_traps_does_not() {
    // The mean weight *is* the walk count -- that identity is the whole
    // justification for the method. The second half is the negative control:
    // dropping the trapped walks instead of counting them as zero inflates
    // the estimate, and the inflation grows with length, which is exactly the
    // bias the weighting was introduced to remove.
    for n in [6usize, 8, 10, 12] {
        let mut rng = Rng::new(0x_5EED_0011 + n as u64);
        let exact = self_avoiding_walk_count(n).unwrap() as f64;
        let samples = 40_000;
        let mut total = 0.0;
        let mut survivors = 0usize;
        let mut survivor_total = 0.0;
        for _ in 0..samples {
            let (_, weight) = saw_sample_rosenbluth(n, &mut rng).unwrap();
            total += weight;
            if weight > 0.0 {
                survivors += 1;
                survivor_total += weight;
            }
        }
        let unbiased = total / samples as f64;
        assert!(
            (unbiased / exact - 1.0).abs() < 0.05,
            "at n = {n} the weighted mean {unbiased} misses the exact count {exact}"
        );
        if survivors < samples {
            let discarded = survivor_total / survivors as f64;
            assert!(
                discarded > unbiased,
                "at n = {n} discarding traps did not inflate the estimate"
            );
        }
    }
}

#[test]
fn prop_a_lattice_walk_moves_one_axis_at_a_time() {
    for dimensions in 1..=6usize {
        let mut rng = Rng::new(0x_5EED_0020 + dimensions as u64);
        let steps = 500;
        let path = random_walk_lattice(steps, dimensions, &mut rng).unwrap();
        assert_eq!(path.len(), steps + 1);
        assert!(path[0].iter().all(|c| *c == 0));
        for pair in path.windows(2) {
            let moved: Vec<usize> = (0..dimensions).filter(|k| pair[0][*k] != pair[1][*k]).collect();
            assert_eq!(moved.len(), 1, "a step changed {} coordinates", moved.len());
            assert_eq!((pair[1][moved[0]] - pair[0][moved[0]]).abs(), 1);
        }
        // Parity: after k steps the coordinate sum has the parity of k, since
        // every step changes it by one. A walk cannot be at the origin after
        // an odd number of steps, which is why the return probability is a
        // statement about even times.
        for (k, position) in path.iter().enumerate() {
            let sum: i64 = position.iter().sum();
            assert_eq!(sum.rem_euclid(2), (k as i64).rem_euclid(2));
        }
    }
    assert!(random_walk_lattice(10, 0, &mut Rng::new(1)).is_err());
}

#[test]
fn prop_the_return_probability_is_certain_below_three_dimensions_and_falls_above() {
    for d in 1..=2usize {
        assert!(close(return_probability(d).unwrap(), 1.0, 1e-12), "Polya fails at d = {d}");
    }
    let mut previous = 1.0;
    for d in 3..=8usize {
        let p = return_probability(d).unwrap();
        assert!(p > 0.0 && p < 1.0, "the return probability at d = {d} is {p}");
        assert!(p < previous, "the return probability rose from {previous} to {p} at d = {d}");
        previous = p;
    }
    assert!(return_probability(0).is_err());
    assert!(return_probability(9).is_err());
}

#[test]
fn prop_the_flory_fit_inverts_its_own_power_law_exactly() {
    // On data that is exactly a power law the least-squares fit in
    // logarithms is exact, so any discrepancy here is a defect in the fit and
    // not sampling error. Checked across exponents so a hard-coded three
    // quarters could not pass.
    for tenths in 1..=12i32 {
        let nu = f64::from(tenths) / 10.0;
        let amplitude = 0.3 + 0.7 * f64::from(tenths);
        let lengths: Vec<usize> = vec![8, 16, 32, 64, 128, 256];
        let squared: Vec<f64> = lengths
            .iter()
            .map(|n| amplitude * (*n as f64).powf(2.0 * nu))
            .collect();
        let fitted = flory_exponent_estimate(&lengths, &squared).unwrap();
        assert!(close(fitted, nu, 1e-10), "the fit returned {fitted} for nu = {nu}");
    }
    assert!(flory_exponent_estimate(&[8], &[1.0]).is_err());
    assert!(flory_exponent_estimate(&[8, 16], &[1.0, 0.0]).is_err());
    assert!(flory_exponent_estimate(&[8, 8], &[1.0, 1.0]).is_err());
}

#[test]
fn prop_the_polymer_average_is_the_weighted_one() {
    // A weighted mean must reproduce a constant exactly whatever the weights,
    // and must move with the weights otherwise -- an implementation that
    // ignored them would pass the first check and fail the second.
    // The weights are chosen so the weighted and flat means differ: with
    // 1, 3, 2 they both come to 388/6, and the negative control below could
    // not have failed.
    let paths: Vec<(Vec<(i64, i64)>, f64)> = vec![
        (vec![(0, 0), (3, 4)], 1.0),
        (vec![(0, 0), (5, 0)], 1.0),
        (vec![(0, 0), (0, 12)], 4.0),
    ];
    let expected = (1.0 * 25.0 + 1.0 * 25.0 + 4.0 * 144.0) / 6.0;
    assert!(close(polymer_end_to_end(&paths).unwrap(), expected, 1e-9));
    let uniform: Vec<(Vec<(i64, i64)>, f64)> =
        paths.iter().map(|(p, _)| (p.clone(), 1.0)).collect();
    let flat = polymer_end_to_end(&uniform).unwrap();
    assert!(close(flat, (25.0 + 25.0 + 144.0) / 3.0, 1e-9));
    assert!(!close(flat, expected, 1e-6), "the weights made no difference");
    // Trapped walks carry zero weight and drop out rather than dragging the
    // mean to zero.
    let mut with_trap = paths.clone();
    with_trap.push((vec![(0, 0)], 0.0));
    assert!(close(polymer_end_to_end(&with_trap).unwrap(), expected, 1e-9));
    assert!(polymer_end_to_end(&[]).is_err());
    assert!(polymer_end_to_end(&[(vec![(0, 0)], 0.0)]).is_err());
}

// ---------------------------------------------------------------------------
// Dimers
// ---------------------------------------------------------------------------

#[test]
fn prop_the_dimer_count_obeys_the_strip_recurrence_and_the_grids_symmetry() {
    // A two-row strip's matchings satisfy the Fibonacci recurrence -- the
    // rightmost column is either covered by one vertical dimer or by two
    // horizontals -- and the grid does not care which side is called m.
    // Kasteleyn's product formula has no visible connection to either fact,
    // which is what makes them worth checking.
    let strip: Vec<f64> = (1..=14).map(|n| dimer_count_kasteleyn(2, n).unwrap()).collect();
    assert!(close(strip[0], 1.0, 1e-6));
    assert!(close(strip[1], 2.0, 1e-6));
    for k in 2..strip.len() {
        assert!(
            close(strip[k], strip[k - 1] + strip[k - 2], 1e-6 * strip[k]),
            "the strip count {} at n = {} breaks the recurrence",
            strip[k],
            k + 1
        );
    }
    for m in 1..=8usize {
        for n in 1..=8usize {
            if (m * n) % 2 == 1 {
                assert!(dimer_count_kasteleyn(m, n).is_err());
                continue;
            }
            let a = dimer_count_kasteleyn(m, n).unwrap();
            let b = dimer_count_kasteleyn(n, m).unwrap();
            assert!(close(a, b, 1e-6 * a.max(1.0)), "the count is not symmetric at {m} by {n}");
            assert!(a >= 1.0 - 1e-9, "the count at {m} by {n} is {a}");
            assert!(close(a, a.round(), 1e-5 * a.max(1.0)), "the count {a} is not an integer");
        }
    }
    // A single row admits exactly one matching, however long it is.
    for n in (2..=16).step_by(2) {
        assert!(close(dimer_count_kasteleyn(1, n).unwrap(), 1.0, 1e-6));
    }
    assert!(dimer_count_kasteleyn(0, 4).is_err());
    assert!(dimer_count_kasteleyn(65, 4).is_err());
}

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

#[test]
fn prop_the_interface_width_ignores_the_mean_height_and_scales_with_the_relief() {
    // The width is a standard deviation, so shifting the whole interface
    // leaves it alone and stretching it multiplies it. A flat interface has
    // no width at all -- and it is a real distinction, since a mean height
    // that grew with time would otherwise be mistaken for roughening.
    let mut rng = Rng::new(0x_5EED_0030);
    for _ in 0..30 {
        let heights: Vec<f64> = (0..64).map(|_| rng.next_f64() * 10.0).collect();
        let base = interface_width(&heights).unwrap();
        let shift = rng.next_f64() * 1000.0 - 500.0;
        let shifted: Vec<f64> = heights.iter().map(|h| h + shift).collect();
        assert!(close(interface_width(&shifted).unwrap(), base, 1e-9));
        let scale = 0.25 + rng.next_f64() * 4.0;
        let scaled: Vec<f64> = heights.iter().map(|h| h * scale).collect();
        assert!(close(interface_width(&scaled).unwrap(), base * scale, 1e-9 * (1.0 + base * scale)));
    }
    assert!(close(interface_width(&[7.5; 40]).unwrap(), 0.0, 1e-12));
    assert!(interface_width(&[]).is_err());
}

#[test]
fn prop_the_growth_exponent_fit_inverts_its_own_power_law_exactly() {
    for hundredths in 5..=60i32 {
        let beta = f64::from(hundredths) / 100.0;
        let times: Vec<f64> = vec![1.0, 3.0, 10.0, 30.0, 100.0, 300.0];
        let widths: Vec<f64> = times.iter().map(|t| 0.7 * t.powf(beta)).collect();
        let fitted = growth_exponent_estimate(&times, &widths).unwrap();
        assert!(close(fitted, beta, 1e-10), "the fit returned {fitted} for beta = {beta}");
    }
    // A saturated interface has exponent zero, which is the failure mode the
    // documentation warns about: including saturated points drags the fit
    // down rather than reporting anything about the growth regime.
    let times: Vec<f64> = vec![1.0, 10.0, 100.0, 1000.0];
    assert!(close(
        growth_exponent_estimate(&times, &[4.0, 4.0, 4.0, 4.0]).unwrap(),
        0.0,
        1e-12
    ));
    assert!(growth_exponent_estimate(&[1.0, 2.0], &[1.0, 2.0]).is_err());
    assert!(growth_exponent_estimate(&[1.0, 2.0, 3.0], &[1.0, 0.0, 3.0]).is_err());
}

// ---------------------------------------------------------------------------
// Heavy tails
// ---------------------------------------------------------------------------

#[test]
fn prop_the_power_law_fit_is_scale_free_and_recovers_a_sampled_exponent() {
    // A power law has no scale, so measuring the data and the cutoff in
    // different units must not move the exponent or the distance. And on data
    // drawn from the fitted family by inverse transform, the maximum
    // likelihood estimate has to come back to the exponent that generated it.
    let mut rng = Rng::new(0x_5EED_0040);
    for tenths in 15..=40i32 {
        let alpha = f64::from(tenths) / 10.0;
        let x_min = 2.0;
        let data: Vec<f64> = (0..40_000)
            .map(|_| {
                let u = 1.0 - rng.next_f64();
                x_min * u.powf(-1.0 / (alpha - 1.0))
            })
            .collect();
        let (fitted, distance) = power_law_fit_clauset(&data, x_min).unwrap();
        assert!(
            (fitted - alpha).abs() < 0.05 * (alpha - 1.0),
            "the fit returned {fitted} for alpha = {alpha}"
        );
        assert!(distance < 0.02, "the fitted law sits {distance} from its own sample");
        let unit = 1_000.0;
        let rescaled: Vec<f64> = data.iter().map(|x| x * unit).collect();
        let (again, distance_again) = power_law_fit_clauset(&rescaled, x_min * unit).unwrap();
        assert!(close(again, fitted, 1e-9), "changing units moved the exponent");
        assert!(close(distance_again, distance, 1e-9), "changing units moved the distance");
    }
}

#[test]
fn prop_the_power_law_distance_separates_a_power_law_from_an_exponential() {
    // The distance has to be a test and not a formality: a sample that is not
    // a power law must be reported as far from one, whatever exponent the
    // estimator settles on. Without this the fit would pass every input.
    let mut rng = Rng::new(0x_5EED_0041);
    let x_min = 2.0;
    let power: Vec<f64> = (0..20_000)
        .map(|_| x_min * (1.0 - rng.next_f64()).powf(-1.0 / 1.5))
        .collect();
    let exponential: Vec<f64> = (0..20_000)
        .map(|_| x_min - 1.5 * (1.0 - rng.next_f64()).ln())
        .collect();
    let (_, good) = power_law_fit_clauset(&power, x_min).unwrap();
    let (_, bad) = power_law_fit_clauset(&exponential, x_min).unwrap();
    assert!(
        bad > 6.0 * good,
        "the exponential sample scored {bad} against the power law's {good}"
    );
    assert!(power_law_fit_clauset(&power, 0.0).is_err());
    assert!(power_law_fit_clauset(&[3.0, 4.0], 1.0).is_err());
    assert!(power_law_fit_clauset(&[1.0; 20], 1.0).is_err());
}

// ---------------------------------------------------------------------------
// Ising: symmetries and exact references
// ---------------------------------------------------------------------------

/// The energy of a 4 by 4 periodic zero-field Ising lattice from a bitmask,
/// matching [`Ising2D::energy`] bond for bond.
fn ising_4x4_energy(state: u64, j: f64) -> f64 {
    let spin = |site: usize| -> f64 {
        if state >> site & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    };
    let mut bonds = 0.0;
    for row in 0..4usize {
        for column in 0..4usize {
            let here = spin(row * 4 + column);
            bonds += here * spin(((row + 1) % 4) * 4 + column);
            bonds += here * spin(row * 4 + (column + 1) % 4);
        }
    }
    -j * bonds
}

#[test]
fn prop_a_global_spin_flip_is_a_symmetry_of_the_zero_field_lattice() {
    // Z2 symmetry is the reason the magnetisation of a finite lattice
    // averages to zero and the reason the absolute value has to be measured
    // instead. It is exact configuration by configuration, not on average.
    let mut rng = Rng::new(0x_5EED_0050);
    for trial in 0..30 {
        let n = 4 + trial % 7;
        let j = 0.4 + rng.next_f64();
        let periodic = trial % 2 == 0;
        let lattice = Ising2D::random(n, j, 0.0, 0.6, periodic, &mut rng).unwrap();
        let mut flipped = lattice.clone();
        for spin in &mut flipped.spins {
            *spin = -*spin;
        }
        assert!(close(flipped.energy(), lattice.energy(), 1e-9 * (1.0 + lattice.energy().abs())));
        assert!(close(flipped.magnetization(), -lattice.magnetization(), 1e-12));
        // A field breaks it, and must.
        let mut fielded = lattice.clone();
        fielded.h = 0.35;
        let mut fielded_flip = flipped.clone();
        fielded_flip.h = 0.35;
        if lattice.magnetization() != 0.0 {
            assert!(
                !close(fielded.energy(), fielded_flip.energy(), 1e-6),
                "the field failed to break the symmetry"
            );
        }
    }
}

#[test]
fn prop_the_cold_lattice_sits_at_the_ground_state_energy() {
    // Every bond is satisfied, so the energy per site is exactly -2j - h on a
    // torus, and no configuration is lower.
    let mut rng = Rng::new(0x_5EED_0051);
    for n in [4usize, 5, 8, 11] {
        for &j in &[0.5f64, 1.0, 2.5] {
            for &h in &[0.0f64, 0.3] {
                let cold = Ising2D::cold(n, j, h, 0.5, true).unwrap();
                assert!(close(cold.magnetization(), 1.0, 1e-12));
                assert!(
                    close(cold.energy_per_site(), -2.0 * j - h, 1e-12),
                    "the cold energy at n = {n}, j = {j}, h = {h} is {}",
                    cold.energy_per_site()
                );
                let random = Ising2D::random(n, j, h, 0.5, true, &mut rng).unwrap();
                assert!(
                    random.energy() >= cold.energy() - 1e-9,
                    "a random configuration fell below the ground state"
                );
            }
        }
    }
    assert!(Ising2D::cold(1, 1.0, 0.0, 1.0, true).is_err());
}

#[test]
fn prop_the_exact_enumeration_matches_the_closed_form_of_a_free_two_level_system() {
    // n independent two-level sites: Z = (1 + e^-beta)^n exactly, the mean
    // energy is n / (1 + e^beta), and the entropy is n times the binary
    // entropy of that occupation. Nothing here is an approximation, so this
    // pins the enumeration, its free energy and its entropy at once -- and a
    // shifted-sum implementation that lost the shift would fail on the cold
    // end where the shift matters.
    for sites in 1..=12usize {
        for &beta in &[0.05f64, 0.5, 1.0, 4.0, 25.0] {
            let energy = |state: u64| -> f64 { f64::from(state.count_ones()) };
            let z = partition_function_exact_small(&energy, sites, beta).unwrap();
            let expected_z = (1.0 + (-beta).exp()).powi(sites as i32);
            assert!(
                close(z / expected_z, 1.0, 1e-9),
                "Z at {sites} sites, beta {beta} is {z} against {expected_z}"
            );
            let (mean, entropy) = thermodynamics_exact_small(&energy, sites, beta).unwrap();
            let p = 1.0 / (1.0 + beta.exp());
            assert!(close(mean, sites as f64 * p, 1e-9 * (1.0 + mean.abs())));
            let binary = if p > 0.0 && p < 1.0 {
                -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
            } else {
                0.0
            };
            assert!(
                close(entropy, sites as f64 * binary, 1e-7 * (1.0 + entropy.abs())),
                "the entropy at {sites} sites, beta {beta} is {entropy}"
            );
            assert!(entropy >= -1e-9, "a negative entropy {entropy}");
            assert!(entropy <= sites as f64 * std::f64::consts::LN_2 + 1e-9);
        }
    }
    let energy = |_: u64| -> f64 { 0.0 };
    assert!(partition_function_exact_small(&energy, 0, 1.0).is_err());
    assert!(partition_function_exact_small(&energy, 25, 1.0).is_err());
    assert!(thermodynamics_exact_small(&energy, 4, 0.0).is_err());
}

#[test]
fn prop_the_density_of_states_reproduces_the_enumeration_at_every_temperature() {
    // One density of states, every temperature: that is the claim Wang-Landau
    // rests on, and it is checked here against an exact enumeration of the
    // same system rather than against a sampler. The heat capacity is checked
    // against a finite difference of the mean energy, which is an independent
    // route to it -- the fluctuation formula and the derivative agree only if
    // both are right.
    let sites = 12usize;
    let mut log_g = vec![f64::NEG_INFINITY; sites + 1];
    for k in 0..=sites {
        // ln C(12, k), summed rather than divided so nothing overflows.
        let mut total = 0.0;
        for i in 0..k {
            total += ((sites - i) as f64).ln() - ((i + 1) as f64).ln();
        }
        log_g[k] = total;
    }
    for &beta in &[0.1f64, 0.5, 1.0, 2.0, 5.0] {
        let energy = |state: u64| -> f64 { f64::from(state.count_ones()) };
        let (mean, capacity) = canonical_from_dos(&log_g, 0.0, 1.0, beta).unwrap();
        let (exact_mean, _) = thermodynamics_exact_small(&energy, sites, beta).unwrap();
        assert!(
            close(mean, exact_mean, 1e-8 * (1.0 + exact_mean.abs())),
            "the density gives {mean} against the enumeration's {exact_mean} at beta {beta}"
        );
        // C = -beta^2 dE/dbeta.
        let d = 1e-4;
        let up = thermodynamics_exact_small(&energy, sites, beta + d).unwrap().0;
        let down = thermodynamics_exact_small(&energy, sites, beta - d).unwrap().0;
        let derivative = -beta * beta * (up - down) / (2.0 * d);
        assert!(
            close(capacity, derivative, 1e-4 * (1.0 + capacity.abs())),
            "the fluctuation capacity {capacity} misses the derivative {derivative}"
        );
    }
    assert!(canonical_from_dos(&[], 0.0, 1.0, 1.0).is_err());
    assert!(canonical_from_dos(&log_g, 0.0, 1.0, 0.0).is_err());
    assert!(canonical_from_dos(&[f64::NEG_INFINITY; 4], 0.0, 1.0, 1.0).is_err());
}

#[test]
fn prop_the_sampler_reproduces_an_exactly_enumerated_lattice() {
    // Sixteen spins can be summed over exactly, so the Monte Carlo mean has a
    // reference that owes nothing to a remembered number. Both updates are
    // checked against it: Metropolis and Wolff sample the same distribution
    // and must agree with the enumeration and with each other.
    let n = 4usize;
    let j = 1.0;
    for &beta in &[0.15f64, 0.3, 0.5] {
        let energy = |state: u64| -> f64 { ising_4x4_energy(state, j) };
        let (exact_total, _) = thermodynamics_exact_small(&energy, n * n, beta).unwrap();
        let exact = exact_total / (n * n) as f64;
        for use_wolff in [false, true] {
            let mut rng = Rng::new(0x_5EED_0060 + u64::from(use_wolff));
            let mut lattice = Ising2D::random(n, j, 0.0, beta, true, &mut rng).unwrap();
            let stats = lattice.sample(60_000, 2_000, 5, use_wolff, &mut rng).unwrap();
            assert!(
                close(stats.e_mean, exact, 0.02),
                "at beta {beta} the sampler (wolff = {use_wolff}) gives {} against {exact}",
                stats.e_mean
            );
            // Fluctuation-dissipation on the sampler's own output: the heat
            // capacity it reports must be the variance it measured.
            let mismatch = fluctuation_dissipation_check(&stats, beta, n * n).unwrap();
            assert!(mismatch < 1e-9, "the reported capacity is not the measured variance");
            assert!(stats.m_abs >= stats.m_mean.abs() - 1e-12);
            assert!(stats.e_var >= 0.0 && stats.susceptibility >= 0.0);
            assert!(stats.binder_cumulant <= 2.0 / 3.0 + 1e-9);
        }
    }
}

#[test]
fn prop_the_fluctuation_dissipation_check_reports_a_real_discrepancy() {
    // The check is only worth calling if it can fail, so it is fed a
    // deliberately inconsistent pair.
    let mut stats = IsingStats {
        e_mean: -1.5,
        e_var: 0.25,
        m_mean: 0.1,
        m_abs: 0.4,
        susceptibility: 1.0,
        heat_capacity: 0.0,
        binder_cumulant: 0.3,
        samples: 100,
    };
    let beta = 0.4;
    let sites = 64usize;
    stats.heat_capacity = beta * beta * sites as f64 * stats.e_var;
    assert!(close(fluctuation_dissipation_check(&stats, beta, sites).unwrap(), 0.0, 1e-12));
    stats.heat_capacity *= 2.0;
    assert!(close(fluctuation_dissipation_check(&stats, beta, sites).unwrap(), 0.5, 1e-12));
    assert!(fluctuation_dissipation_check(&stats, 0.0, sites).is_err());
    assert!(fluctuation_dissipation_check(&stats, beta, 0).is_err());
}

#[test]
fn prop_the_onsager_magnetisation_switches_on_exactly_at_the_critical_point() {
    // The transition is not a gradual crossover in the exact solution: the
    // magnetisation is identically zero above the critical temperature and
    // rises with the eighth-root singularity below it. Both halves are
    // checked, along with the monotonicity in between.
    let tc = ising_tc_exact();
    assert!(close(tc, 2.0 / (1.0 + 2f64.sqrt()).ln(), 1e-12));
    assert!(close(potts_tc_exact(2).unwrap(), tc / 2.0, 1e-12));
    for &j in &[0.5f64, 1.0, 2.0] {
        let beta_c = 1.0 / (tc * j);
        for k in 1..=20 {
            let hot = beta_c * (1.0 - 0.02 * f64::from(k));
            assert!(
                close(onsager_magnetization(hot, j).unwrap(), 0.0, 1e-15),
                "a magnetisation above the critical temperature at j = {j}"
            );
        }
        let mut previous = 0.0;
        for k in 1..=20 {
            let cold = beta_c * (1.0 + 0.02 * f64::from(k));
            let m = onsager_magnetization(cold, j).unwrap();
            assert!(m > previous, "the magnetisation fell from {previous} to {m}");
            assert!(m <= 1.0 + 1e-12);
            previous = m;
        }
        assert!(close(onsager_magnetization(20.0 / j, j).unwrap(), 1.0, 1e-9));
    }
    assert!(onsager_magnetization(1.0, 0.0).is_err());
    assert!(potts_tc_exact(1).is_err());
}

#[test]
fn prop_the_potts_critical_point_rises_with_the_state_count() {
    // 1 / ln(1 + sqrt q) falls as q grows: more states cost more entropy to
    // order, so ordering survives only to a lower temperature.
    let mut previous = f64::INFINITY;
    for q in 2..=200u8 {
        let tc = potts_tc_exact(q).unwrap();
        assert!(tc > 0.0);
        assert!(tc < previous, "the Potts critical temperature rose at q = {q}");
        assert!(
            close(tc, 1.0 / (1.0 + f64::from(q).sqrt()).ln(), 1e-12),
            "the closed form is wrong at q = {q}"
        );
        previous = tc;
    }
}

#[test]
fn prop_the_one_dimensional_chain_has_no_transition() {
    // The free energy is analytic and the magnetisation vanishes with the
    // field at every positive temperature -- Ising's own result. The
    // zero-field free energy is -ln(2 cosh(beta j)) / beta exactly.
    for &j in &[0.5f64, 1.0, 3.0] {
        for k in 1..=40 {
            let beta = 0.05 * f64::from(k);
            let (f, m) = ising_1d_exact(beta, j, 0.0).unwrap();
            assert!(close(m, 0.0, 1e-12), "a spontaneous magnetisation at beta = {beta}");
            let expected = -(2.0 * (beta * j).cosh()).ln() / beta;
            assert!(close(f, expected, 1e-9 * (1.0 + f.abs())), "the free energy is {f}");
            // The magnetisation is odd in the field and saturates.
            let (_, up) = ising_1d_exact(beta, j, 0.4).unwrap();
            let (_, down) = ising_1d_exact(beta, j, -0.4).unwrap();
            assert!(close(up, -down, 1e-12));
            assert!(up > 0.0 && up < 1.0);
            // Saturation is set by beta * h rather than by h alone, so the
            // field has to be scaled with the temperature to reach it.
            let (_, strong) = ising_1d_exact(beta, j, 100.0 / beta).unwrap();
            assert!(close(strong, 1.0, 1e-12), "the chain saturated only to {strong}");
            // And it climbs there monotonically.
            let mut previous = 0.0;
            for step in 1..=12 {
                let (_, m) = ising_1d_exact(beta, j, f64::from(step) * 0.25 / beta).unwrap();
                assert!(m > previous, "the magnetisation fell from {previous} to {m}");
                previous = m;
            }
        }
    }
    assert!(ising_1d_exact(0.0, 1.0, 0.0).is_err());
}

#[test]
fn prop_the_binder_crossing_recovers_a_crossing_it_was_given() {
    // Straight lines with a common intersection: the crossing is exact, so
    // the estimate has to be too, whatever the slopes.
    let temperatures: Vec<f64> = (0..11).map(|k| 1.0 + 0.2 * f64::from(k)).collect();
    for tenths in 1..=18i32 {
        let star = 1.05 + f64::from(tenths) * 0.1;
        let curves: Vec<Vec<f64>> = [0.3f64, 0.8, 1.7]
            .iter()
            .map(|slope| temperatures.iter().map(|t| 0.5 + slope * (t - star)).collect())
            .collect();
        let found = binder_crossing(&temperatures, &curves).unwrap();
        assert!(close(found, star, 1e-9), "the crossing at {star} was reported as {found}");
    }
    assert!(binder_crossing(&temperatures, &[vec![0.0; 11]]).is_err());
    assert!(binder_crossing(&temperatures, &[vec![0.0; 11], vec![1.0; 3]]).is_err());
}

// ---------------------------------------------------------------------------
// Potts and XY
// ---------------------------------------------------------------------------

#[test]
fn prop_the_potts_order_parameter_is_calibrated_at_both_ends() {
    // Zero when every state is equally common and one when a single state
    // takes the lattice, with nothing outside that range in between. The
    // normalisation (q * largest - 1) / (q - 1) is what makes different q
    // comparable, so an implementation that dropped it would still look
    // plausible on q = 2 alone.
    let mut rng = Rng::new(0x_5EED_0070);
    for q in 2..=8u8 {
        let n = 12usize;
        let mut model = Potts2D::random(q, n, 1.0, 0.5, &mut rng).unwrap();
        assert!(model.order_parameter() >= -1e-12);
        assert!(model.order_parameter() <= 1.0 + 1e-12);
        model.states.fill(0);
        assert!(close(model.order_parameter(), 1.0, 1e-12));
        assert!(close(model.energy(), -2.0 * (n * n) as f64, 1e-9));
        // An exactly even split over q states scores zero.
        if (n * n).is_multiple_of(q as usize) {
            for (index, state) in model.states.iter_mut().enumerate() {
                *state = (index % q as usize) as u8;
            }
            assert!(close(model.order_parameter(), 0.0, 1e-12));
        }
    }
    assert!(Potts2D::random(1, 8, 1.0, 0.5, &mut rng).is_err());
    assert!(Potts2D::random(3, 1, 1.0, 0.5, &mut rng).is_err());
    assert!(Potts2D::random(3, 8, 1.0, 0.0, &mut rng).is_err());
}

#[test]
fn prop_the_xy_energy_is_invariant_under_a_global_rotation() {
    // The XY model's symmetry is continuous, and it is the reason the model
    // has no ordered phase in two dimensions at all: the energy depends only
    // on angle differences, so a uniform twist costs nothing. Checked on
    // sampled configurations rather than a special one.
    let mut rng = Rng::new(0x_5EED_0080);
    for _ in 0..20 {
        let n = 8usize;
        let model = XyModel2D::random(n, 1.0, 0.8, &mut rng).unwrap();
        let base = model.energy();
        let twist = rng.next_f64() * std::f64::consts::TAU;
        let mut rotated = model.clone();
        for angle in &mut rotated.theta {
            *angle += twist;
        }
        assert!(
            close(rotated.energy(), base, 1e-8 * (1.0 + base.abs())),
            "a global rotation moved the energy from {base} to {}",
            rotated.energy()
        );
        // A uniform configuration is the ground state, at -2 j per site.
        let mut uniform = model.clone();
        uniform.theta.fill(twist);
        assert!(close(uniform.energy(), -2.0 * (n * n) as f64, 1e-9));
        assert!(base >= uniform.energy() - 1e-9);
        assert_eq!(uniform.vortex_count(), (0, 0));
    }
    assert!(XyModel2D::random(2, 1.0, 1.0, &mut rng).is_err());
    assert!(XyModel2D::random(8, 1.0, 0.0, &mut rng).is_err());
}

#[test]
fn prop_the_total_vorticity_of_a_torus_vanishes() {
    // Every plaquette's winding is an integer and their sum is zero, because
    // each bond is traversed once in each direction. It is a topological
    // identity: it holds on a random configuration, on a thermalised one, and
    // after any update whatever, so vortices can only be created in pairs.
    let mut rng = Rng::new(0x_5EED_0081);
    for trial in 0..12 {
        let n = 8 + trial % 5;
        let beta = 0.3 + 0.2 * (trial % 6) as f64;
        let mut model = XyModel2D::random(n, 1.0, beta, &mut rng).unwrap();
        for stage in 0..3 {
            if stage > 0 {
                for _ in 0..20 {
                    model.metropolis_sweep(&mut rng, 1.2);
                }
            }
            let mut total = 0i32;
            for row in 0..model.n {
                for column in 0..model.n {
                    let v = model.plaquette_vorticity(row, column);
                    assert!(v.abs() <= 2, "a winding of {v} on one plaquette");
                    total += v;
                }
            }
            assert_eq!(total, 0, "the total vorticity is {total} at stage {stage}");
            let (positive, negative) = model.vortex_count();
            // With every winding a single unit, the counts balance as well as
            // the sum; a stray double-winding would show up here.
            assert_eq!(
                positive, negative,
                "{positive} vortices against {negative} antivortices"
            );
        }
    }
}
