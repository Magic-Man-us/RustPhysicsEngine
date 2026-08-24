//! Properties for `discrete::combinatorics`, `partitions`, `sequences` and
//! `disjoint_set`.
//!
//! These are randomized cross-checks between two independent routes to the
//! same value, rather than comparisons against stored tables.

use rust_physics_engine::discrete::combinatorics::{
    binomial_u64, catalan, catalan_mod, is_permutation, nth_permutation, permutation_compose,
    permutation_cycle_type, permutation_index, permutation_inverse, permutation_order,
    permutation_sign, permutation_to_cycles, random_permutation, stirling_second,
    twelvefold_way,
};
use rust_physics_engine::discrete::disjoint_set::DisjointSet;
use rust_physics_engine::discrete::partitions::{
    durfee_square, hook_lengths, partition_conjugate, partition_count, partitions_iter,
    rsk_correspondence, standard_tableaux_count,
};
use rust_physics_engine::discrete::sequences::{
    fibonacci_mod, find_linear_recurrence, linear_recurrence, linear_recurrence_mod,
    pisano_period,
};
use rust_physics_engine::exact::bigint::BigInt;
use rust_physics_engine::exact::rational::Rational;
use rust_physics_engine::monte_carlo::Rng;

/// A value in `0..n` from the high bits, since the generator is a linear
/// congruential one whose low bits have very short periods.
fn pick(rng: &mut Rng, n: u64) -> u64 {
    ((u128::from(rng.next_u64()) * u128::from(n)) >> 64) as u64
}

/// The sign is a homomorphism onto {+1, -1}, and the order is the least power
/// giving the identity, for random permutations rather than a fixed few.
#[test]
fn prop_permutation_group_laws() {
    let mut rng = Rng::new(0xC0FFEE);
    for _ in 0..300 {
        let n = 1 + pick(&mut rng, 9) as usize;
        let a = random_permutation(n, &mut rng);
        let b = random_permutation(n, &mut rng);
        assert!(is_permutation(&a) && is_permutation(&b));

        let ab = permutation_compose(&a, &b);
        assert_eq!(permutation_sign(&ab), permutation_sign(&a) * permutation_sign(&b));

        let inv = permutation_inverse(&a);
        let id: Vec<usize> = (0..n).collect();
        assert_eq!(permutation_compose(&a, &inv), id);
        assert_eq!(permutation_compose(&inv, &a), id);
        // The inverse has the same cycle type and the same order.
        assert_eq!(permutation_cycle_type(&a), permutation_cycle_type(&inv));
        assert_eq!(permutation_order(&a), permutation_order(&inv));

        // Conjugation preserves the cycle type.
        let g = random_permutation(n, &mut rng);
        let gi = permutation_inverse(&g);
        let conj = permutation_compose(&permutation_compose(&g, &a), &gi);
        assert_eq!(permutation_cycle_type(&conj), permutation_cycle_type(&a));

        // The order really is the least such power.
        let order = permutation_order(&a).to_i64().unwrap() as usize;
        let mut acc = id.clone();
        for step in 1..=order {
            acc = permutation_compose(&acc, &a);
            if step < order {
                assert_ne!(acc, id, "returned to the identity at step {step}");
            }
        }
        assert_eq!(acc, id);

        // The cycle lengths partition n.
        assert_eq!(permutation_cycle_type(&a).iter().sum::<usize>(), n);
        assert_eq!(permutation_to_cycles(&a).len(), permutation_cycle_type(&a).len());
    }
}

/// nth_permutation and permutation_index invert each other at random indices,
/// including ones far too large to reach by enumeration.
#[test]
fn prop_factoradic_index_round_trips() {
    let mut rng = Rng::new(7_654_321);
    for _ in 0..300 {
        let n = 1 + pick(&mut rng, 12) as usize;
        let total = BigInt::factorial(n as u64);
        // Build a uniform index below n! from 64 random bits per limb.
        let idx = BigInt::random_below(&total, &mut rng);
        let p = nth_permutation(n, &idx);
        assert!(is_permutation(&p));
        assert_eq!(permutation_index(&p), idx);
    }
    // The order is respected: a larger index gives a lexicographically larger
    // permutation.
    let mut rng = Rng::new(99);
    for _ in 0..200 {
        let n = 2 + pick(&mut rng, 8) as usize;
        let total = BigInt::factorial(n as u64);
        let i = BigInt::random_below(&total, &mut rng);
        let j = BigInt::random_below(&total, &mut rng);
        let (pi, pj) = (nth_permutation(n, &i), nth_permutation(n, &j));
        assert_eq!(i < j, pi < pj, "order not preserved at {i} vs {j}");
    }
}

/// binomial_u64 agrees with the arbitrary-precision value whenever it reports
/// a result, and reports none exactly when the value does not fit.
#[test]
fn prop_binomial_u64_agrees_with_bigint() {
    let mut rng = Rng::new(31_337);
    let max = BigInt::from_str_radix(&u64::MAX.to_string(), 10).unwrap();
    for _ in 0..2_000 {
        let n = pick(&mut rng, 200);
        let k = pick(&mut rng, n + 1);
        let exact = BigInt::binomial(n, k);
        match binomial_u64(n, k) {
            Some(v) => {
                assert_eq!(BigInt::from_u64(v), exact, "C({n}, {k})");
            }
            None => assert!(exact > max, "C({n}, {k}) fits but was refused"),
        }
    }
}

/// The modular Catalan number agrees with reducing the exact one, for moduli
/// that share factors with n + 1 as well as coprime ones.
#[test]
fn prop_catalan_mod_agrees_with_exact() {
    let mut rng = Rng::new(2_718_281);
    for _ in 0..200 {
        let n = pick(&mut rng, 60);
        let m = 1 + pick(&mut rng, 10_000);
        let exact = catalan(n)
            .rem_euclid(&BigInt::from_u64(m))
            .to_i64()
            .unwrap() as u64;
        assert_eq!(catalan_mod(n, m), exact, "C({n}) mod {m}");
    }
}

/// Conjugation is an involution that preserves the sum and swaps the length
/// with the largest part, on random partitions.
#[test]
fn prop_partition_conjugation_is_an_involution() {
    let mut rng = Rng::new(161_803);
    for _ in 0..200 {
        let n = 1 + pick(&mut rng, 40);
        // Sample a partition by taking a random one from the enumeration.
        let all: Vec<Vec<u64>> = partitions_iter(n).collect();
        assert_eq!(
            BigInt::from_u64(all.len() as u64),
            partition_count(n),
            "enumeration count disagrees at n = {n}"
        );
        let p = &all[pick(&mut rng, all.len() as u64) as usize];
        let c = partition_conjugate(p);
        assert_eq!(c.iter().sum::<u64>(), n);
        assert_eq!(partition_conjugate(&c), *p);
        assert_eq!(c.len() as u64, p[0]);
        assert_eq!(p.len() as u64, c[0]);
        assert_eq!(durfee_square(p), durfee_square(&c));
        // Hook lengths transpose with the diagram.
        let hp = hook_lengths(p);
        let hc = hook_lengths(&c);
        for (i, row) in hp.iter().enumerate() {
            for (j, &h) in row.iter().enumerate() {
                assert_eq!(h, hc[j][i], "hook ({i}, {j}) of {p:?}");
            }
        }
        // The tableau count is conjugation-invariant.
        assert_eq!(standard_tableaux_count(p), standard_tableaux_count(&c));
    }
}

/// RSK on a random permutation: matching shapes, and the first row length is
/// the longest increasing subsequence (Schensted).
#[test]
fn prop_rsk_shape_is_schensted() {
    let mut rng = Rng::new(1_414_213);
    for _ in 0..300 {
        let n = 1 + pick(&mut rng, 30) as usize;
        let perm = random_permutation(n, &mut rng);
        let (p, q) = rsk_correspondence(&perm);
        let shape_p: Vec<usize> = p.iter().map(Vec::len).collect();
        let shape_q: Vec<usize> = q.iter().map(Vec::len).collect();
        assert_eq!(shape_p, shape_q);
        assert!(shape_p.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(shape_p.iter().sum::<usize>(), n);

        // Longest increasing and decreasing subsequences by O(n^2) DP.
        let mut inc = vec![1usize; n];
        let mut dec = vec![1usize; n];
        for i in 0..n {
            for j in 0..i {
                if perm[j] < perm[i] {
                    inc[i] = inc[i].max(inc[j] + 1);
                }
                if perm[j] > perm[i] {
                    dec[i] = dec[i].max(dec[j] + 1);
                }
            }
        }
        assert_eq!(p[0].len(), *inc.iter().max().unwrap(), "{perm:?}");
        assert_eq!(p.len(), *dec.iter().max().unwrap(), "{perm:?}");

        // Schuetzenberger: RSK of the inverse swaps the tableaux.
        let (pi, qi) = rsk_correspondence(&permutation_inverse(&perm));
        assert_eq!(pi, q);
        assert_eq!(qi, p);
    }
}

/// Berlekamp-Massey recovers a recurrence that regenerates its input, for
/// randomly generated integer recurrences.
#[test]
fn prop_berlekamp_massey_regenerates() {
    let mut rng = Rng::new(577_215);
    for _ in 0..200 {
        let order = 1 + pick(&mut rng, 4) as usize;
        let coeffs: Vec<i64> = (0..order)
            .map(|_| pick(&mut rng, 11) as i64 - 5)
            .collect();
        let init: Vec<i64> = (0..order)
            .map(|_| pick(&mut rng, 11) as i64 - 5)
            .collect();
        // A trailing zero coefficient makes the true order smaller, which is
        // fine: Berlekamp-Massey finds the minimal one, not the stated one.
        let terms: Vec<Rational> = (0..4 * order as u64 + 4)
            .map(|i| Rational::from_int(linear_recurrence(&init, &coeffs, i)))
            .collect();
        let Some(found) = find_linear_recurrence(&terms) else {
            // Only an all-zero sequence with too few terms can fail here.
            assert!(terms.iter().all(Rational::is_zero) || terms.len() < 2);
            continue;
        };
        assert!(found.len() <= order, "order {} exceeds {order}", found.len());
        for i in found.len()..terms.len() {
            let mut acc = Rational::zero();
            for (j, cj) in found.iter().enumerate() {
                acc = acc.add(&cj.mul(&terms[i - 1 - j]));
            }
            assert_eq!(acc, terms[i], "regeneration fails at {i} for {coeffs:?}");
        }
    }
}

/// The matrix-power recurrence agrees with direct iteration, and the fast
/// doubling Fibonacci agrees with both.
#[test]
fn prop_linear_recurrence_mod_agrees_with_iteration() {
    let mut rng = Rng::new(1_729);
    for _ in 0..300 {
        let order = 1 + pick(&mut rng, 3) as usize;
        let coeffs: Vec<i64> = (0..order).map(|_| pick(&mut rng, 9) as i64 - 4).collect();
        let init: Vec<i64> = (0..order).map(|_| pick(&mut rng, 9) as i64 - 4).collect();
        let n = pick(&mut rng, 120);
        let m = 1 + pick(&mut rng, 1_000_000);
        let direct = linear_recurrence(&init, &coeffs, n)
            .rem_euclid(&BigInt::from_u64(m))
            .to_i64()
            .unwrap() as u64;
        assert_eq!(
            linear_recurrence_mod(&init, &coeffs, n, m),
            direct,
            "init {init:?}, coeffs {coeffs:?}, n = {n}, m = {m}"
        );
    }
    // Fast doubling against the same engine, and periodicity at a huge index.
    let mut rng = Rng::new(4_669);
    for _ in 0..200 {
        let n = pick(&mut rng, 500);
        let m = 1 + pick(&mut rng, 5_000);
        let direct = linear_recurrence(&[0, 1], &[1, 1], n)
            .rem_euclid(&BigInt::from_u64(m))
            .to_i64()
            .unwrap() as u64;
        assert_eq!(fibonacci_mod(n, m), direct, "F({n}) mod {m}");
        let p = pisano_period(m);
        assert_eq!(fibonacci_mod(n + p, m), direct, "period {p} for m = {m}");
    }
}

/// Union-find agrees with the equivalence relation the same unions generate.
#[test]
fn prop_disjoint_set_matches_transitive_closure() {
    let mut rng = Rng::new(0x00D1_5C0D);
    for _ in 0..40 {
        let n = 2 + pick(&mut rng, 30) as usize;
        let mut ds = DisjointSet::new(n);
        let mut reach = vec![vec![false; n]; n];
        for (i, row) in reach.iter_mut().enumerate() {
            row[i] = true;
        }
        let mut merges = 0usize;
        for _ in 0..2 * n {
            let a = pick(&mut rng, n as u64) as usize;
            let b = pick(&mut rng, n as u64) as usize;
            if ds.union(a, b) {
                merges += 1;
            }
            let ca: Vec<usize> = (0..n).filter(|&i| reach[i][a]).collect();
            let cb: Vec<usize> = (0..n).filter(|&i| reach[i][b]).collect();
            for &i in &ca {
                for &j in &cb {
                    reach[i][j] = true;
                    reach[j][i] = true;
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                assert_eq!(ds.connected(i, j), reach[i][j], "({i}, {j}) of {n}");
            }
        }
        assert_eq!(ds.count(), n - merges);
        let sets = ds.sets();
        assert_eq!(sets.len(), ds.count());
        assert_eq!(sets.iter().map(Vec::len).sum::<usize>(), n);
    }
}

/// The twelvefold way's surjection column agrees with inclusion-exclusion over
/// the boxes left empty, which is a different derivation from the Stirling
/// recurrence the implementation uses.
#[test]
fn prop_twelvefold_surjections_by_inclusion_exclusion() {
    let mut rng = Rng::new(8_675_309);
    for _ in 0..200 {
        let n = pick(&mut rng, 15);
        let k = pick(&mut rng, 12);
        // Surjections from n labelled balls onto k labelled boxes.
        let mut by_ie = BigInt::zero();
        for j in 0..=k {
            let term = BigInt::binomial(k, j).mul(&BigInt::from_u64(k - j).pow(n));
            if j % 2 == 0 {
                by_ie = by_ie.add(&term);
            } else {
                by_ie = by_ie.sub(&term);
            }
        }
        assert_eq!(
            twelvefold_way(n, k, None, Some(true), true, true),
            by_ie,
            "surjections from {n} onto {k}"
        );
        // And k! S(n, k) is the same thing.
        assert_eq!(BigInt::factorial(k).mul(&stirling_second(n, k)), by_ie);
    }
}
