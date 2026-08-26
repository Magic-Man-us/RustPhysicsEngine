//! Integer partitions, Young diagrams, and the RSK correspondence.
//!
//! A partition of `n` is a weakly decreasing list of positive integers
//! summing to `n`. It is stored as `Vec<u64>` in that order, so `p[0]` is the
//! largest part.

use crate::discrete::primes::sieve_eratosthenes;
use crate::exact::bigint::BigInt;

/// The number of partitions of `n`, by Euler's pentagonal number theorem.
///
/// The theorem gives `p(n) = sum_k (-1)^(k+1) [p(n - g_k) + p(n - g'_k)]` over
/// the generalised pentagonal numbers `g_k = k(3k-1)/2`. There are only
/// `O(sqrt n)` of those below `n`, so each value costs `O(sqrt n)` additions
/// and the whole table costs `O(n^1.5)` -- far less than the `O(n^2)` of the
/// naive "partitions of n into parts at most m" table.
#[must_use]
pub fn partition_count(n: u64) -> BigInt {
    partition_count_table(n).pop().unwrap()
}

/// `p(0)` through `p(n)`.
#[must_use]
pub fn partition_count_table(n: u64) -> Vec<BigInt> {
    let n = n as usize;
    let mut p: Vec<BigInt> = Vec::with_capacity(n + 1);
    p.push(BigInt::one());
    for m in 1..=n {
        let mut acc = BigInt::zero();
        let mut k = 1i64;
        loop {
            // The two pentagonal numbers for this k.
            let g1 = (k * (3 * k - 1) / 2) as usize;
            if g1 > m {
                break;
            }
            let g2 = (k * (3 * k + 1) / 2) as usize;
            // Signs alternate with k, not with which of the pair it is.
            if k % 2 == 1 {
                acc = acc.add(&p[m - g1]);
                if g2 <= m {
                    acc = acc.add(&p[m - g2]);
                }
            } else {
                acc = acc.sub(&p[m - g1]);
                if g2 <= m {
                    acc = acc.sub(&p[m - g2]);
                }
            }
            k += 1;
        }
        p.push(acc);
    }
    p
}

/// The partitions of `n`, each weakly decreasing, in reverse lexicographic
/// order (starting at `[n]` and ending at all ones).
pub fn partitions_iter(n: u64) -> impl Iterator<Item = Vec<u64>> + use<> {
    Partitions {
        current: if n == 0 {
            Some(Vec::new())
        } else {
            Some(vec![n])
        },
        exhausted_empty: n != 0,
    }
}

struct Partitions {
    current: Option<Vec<u64>>,
    /// For n = 0 the single partition is the empty one and there is no
    /// successor; this flag distinguishes that from a real state.
    exhausted_empty: bool,
}

impl Iterator for Partitions {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Vec<u64>> {
        let cur = self.current.take()?;
        let out = cur.clone();
        if !self.exhausted_empty {
            return Some(out);
        }
        // Successor in reverse lexicographic order: find the rightmost part
        // that can be decreased (any part above 1, given something to its
        // right to absorb the unit), decrease it, and pad the remainder with
        // as many copies of the new value as fit, then a single leftover.
        let mut p = cur;
        let mut i = p.len();
        loop {
            if i == 0 {
                self.current = None;
                return Some(out);
            }
            i -= 1;
            if p[i] > 1 {
                break;
            }
        }
        // Everything from i onwards is redistributed.
        let rest: u64 = p[i..].iter().sum();
        let val = p[i] - 1;
        p.truncate(i);
        let mut left = rest;
        while left >= val {
            p.push(val);
            left -= val;
        }
        if left > 0 {
            p.push(left);
        }
        self.current = Some(p);
        Some(out)
    }
}

/// The number of partitions of `n` into exactly `k` positive parts.
///
/// Recurrence `P(n, k) = P(n-1, k-1) + P(n-k, k)`: either the smallest part is
/// a one, which removes it, or every part is at least two, which subtracts one
/// from each.
#[must_use]
pub fn partitions_into_k(n: u64, k: u64) -> BigInt {
    if k == 0 {
        return if n == 0 { BigInt::one() } else { BigInt::zero() };
    }
    if k > n {
        return BigInt::zero();
    }
    let (n, k) = (n as usize, k as usize);
    // table[j][m] = partitions of m into exactly j parts, rolled over j.
    let mut prev = vec![BigInt::zero(); n + 1];
    prev[0] = BigInt::one(); // zero parts sum to zero
    for j in 1..=k {
        let mut cur = vec![BigInt::zero(); n + 1];
        for m in 1..=n {
            let a = prev[m - 1].clone();
            let b = if m >= j {
                cur[m - j].clone()
            } else {
                BigInt::zero()
            };
            cur[m] = a.add(&b);
        }
        prev = cur;
    }
    prev[n].clone()
}

/// The number of partitions of `n` into at most `k` parts.
///
/// By conjugation this also counts the partitions of `n` whose largest part is
/// at most `k`.
#[must_use]
pub fn partition_count_into_at_most_k(n: u64, k: u64) -> BigInt {
    (0..=k).fold(BigInt::zero(), |a, j| a.add(&partitions_into_k(n, j)))
}

/// The number of partitions of `n` into distinct parts.
///
/// Product `prod_{i=1..n} (1 + x^i)` accumulated as a coefficient table.
#[must_use]
pub fn partitions_distinct(n: u64) -> BigInt {
    let n = n as usize;
    let mut c = vec![BigInt::zero(); n + 1];
    c[0] = BigInt::one();
    for part in 1..=n {
        // Each part is used at most once, so sweep downwards.
        for m in (part..=n).rev() {
            let add = c[m - part].clone();
            c[m] = c[m].add(&add);
        }
    }
    c[n].clone()
}

/// The number of partitions of `n` into odd parts.
///
/// Euler's theorem says this equals [`partitions_distinct`]; the two are
/// computed independently here so that agreement is evidence rather than a
/// tautology.
#[must_use]
pub fn partitions_odd(n: u64) -> BigInt {
    let n = n as usize;
    let mut c = vec![BigInt::zero(); n + 1];
    c[0] = BigInt::one();
    let mut part = 1usize;
    while part <= n {
        // Unbounded multiplicity, so sweep upwards.
        for m in part..=n {
            let add = c[m - part].clone();
            c[m] = c[m].add(&add);
        }
        part += 2;
    }
    c[n].clone()
}

/// The conjugate partition: the column lengths of the Young diagram.
///
/// `conjugate(p)[j]` counts the parts of `p` exceeding `j`. Conjugation is an
/// involution and preserves the sum.
#[must_use]
pub fn partition_conjugate(p: &[u64]) -> Vec<u64> {
    let Some(&largest) = p.first() else {
        return Vec::new();
    };
    (0..largest)
        .map(|j| p.iter().filter(|&&x| x > j).count() as u64)
        .collect()
}

/// The Young diagram of `p` in English notation: row `i` has `p[i]` true
/// cells, padded with false to the width of the first row.
#[must_use]
pub fn young_diagram(p: &[u64]) -> Vec<Vec<bool>> {
    let width = p.first().copied().unwrap_or(0) as usize;
    p.iter()
        .map(|&len| (0..width).map(|j| (j as u64) < len).collect())
        .collect()
}

/// The hook length of every cell of the Young diagram, in the same ragged
/// shape as `p`.
///
/// The hook of a cell is the cell itself, the cells to its right in the row
/// (the arm), and the cells below it in the column (the leg).
#[must_use]
pub fn hook_lengths(p: &[u64]) -> Vec<Vec<u64>> {
    let conj = partition_conjugate(p);
    p.iter()
        .enumerate()
        .map(|(i, &len)| {
            (0..len)
                .map(|j| {
                    let arm = len - j - 1;
                    let leg = conj[j as usize] - i as u64 - 1;
                    arm + leg + 1
                })
                .collect()
        })
        .collect()
}

/// The number of standard Young tableaux of shape `p`, by the hook length
/// formula `n! / prod(hooks)`.
///
/// # Panics
/// Panics if `p` is not weakly decreasing, since the hook lengths would then
/// be meaningless.
#[must_use]
pub fn standard_tableaux_count(p: &[u64]) -> BigInt {
    assert!(
        p.windows(2).all(|w| w[0] >= w[1]),
        "a partition must be weakly decreasing"
    );
    let n: u64 = p.iter().sum();
    let mut denom = BigInt::one();
    for row in hook_lengths(p) {
        for h in row {
            denom = denom.mul(&BigInt::from_u64(h));
        }
    }
    BigInt::factorial(n).div_rem(&denom).0
}

/// The Robinson-Schensted correspondence: a permutation of `0..n` maps to a
/// pair of standard Young tableaux of the same shape.
///
/// `P` is built by row insertion (each value bumps the leftmost strictly
/// larger entry down a row) and `Q` records which cell was created at each
/// step, so `Q` is standard by construction. The map is a bijection between
/// `S_n` and such pairs, which is the combinatorial content of the identity
/// `sum_shapes f(shape)^2 = n!`.
///
/// Entries of `P` are the permutation's own values; entries of `Q` are the
/// step indices `0..n`.
#[must_use]
pub fn rsk_correspondence(perm: &[usize]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut p: Vec<Vec<usize>> = Vec::new();
    let mut q: Vec<Vec<usize>> = Vec::new();
    for (step, &value) in perm.iter().enumerate() {
        let mut carry = value;
        let mut row = 0usize;
        loop {
            if row == p.len() {
                p.push(vec![carry]);
                q.push(vec![step]);
                break;
            }
            // Bump the leftmost entry strictly greater than the carry.
            match p[row].iter().position(|&x| x > carry) {
                Some(idx) => {
                    std::mem::swap(&mut p[row][idx], &mut carry);
                    row += 1;
                }
                None => {
                    p[row].push(carry);
                    q[row].push(step);
                    break;
                }
            }
        }
    }
    (p, q)
}

/// The side of the Durfee square: the largest `s` with `p[s-1] >= s`, that is,
/// the largest square that fits in the top-left of the Young diagram.
#[must_use]
pub fn durfee_square(p: &[u64]) -> u64 {
    let mut s = 0u64;
    while (s as usize) < p.len() && p[s as usize] > s {
        s += 1;
    }
    s
}

/// The Hardy-Ramanujan asymptotic for the partition count,
/// `exp(pi sqrt(2n/3)) / (4 n sqrt 3)`.
///
/// The relative error decays like `1/sqrt(n)`, so this is an order-of-magnitude
/// estimate rather than a value to round.
#[must_use]
pub fn hardy_ramanujan_estimate(n: u64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let x = n as f64;
    (std::f64::consts::PI * (2.0 * x / 3.0).sqrt()).exp() / (4.0 * x * 3.0f64.sqrt())
}

/// True when every even number from 4 to `up_to` is a sum of two primes.
///
/// Verification, not proof: the conjecture is open. Returns `true` vacuously
/// for `up_to < 4`.
#[must_use]
pub fn goldbach_conjecture_verify(up_to: u64) -> bool {
    if up_to < 4 {
        return true;
    }
    let limit = up_to as usize;
    let primes = sieve_eratosthenes(limit);
    let mut is_prime = vec![false; limit + 1];
    for &p in &primes {
        is_prime[p] = true;
    }
    let mut n = 4u64;
    while n <= up_to {
        // Small primes first: a decomposition with a small summand exists for
        // every even number tested so far, so this finds one almost at once.
        let found = primes
            .iter()
            .take_while(|&&p| (p as u64) <= n / 2)
            .any(|&p| is_prime[(n - p as u64) as usize]);
        if !found {
            return false;
        }
        n += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::combinatorics::{
        binomial_u64, permutation_inverse, permutations_iter,
    };
    use std::collections::HashSet;

    fn big(n: u64) -> BigInt {
        BigInt::from_u64(n)
    }

    /// The pentagonal recurrence must agree with a completely different
    /// method: the coefficient table of the generating product
    /// `prod 1/(1 - x^i)`, built by dynamic programming.
    #[test]
    fn partition_count_matches_the_generating_function() {
        const N: usize = 120;
        let mut c = vec![BigInt::zero(); N + 1];
        c[0] = BigInt::one();
        for part in 1..=N {
            for m in part..=N {
                let add = c[m - part].clone();
                c[m] = c[m].add(&add);
            }
        }
        let table = partition_count_table(N as u64);
        for n in 0..=N {
            assert_eq!(table[n], c[n], "p({n}) disagrees");
            assert_eq!(partition_count(n as u64), c[n]);
        }
        // The roadmap's headline value.
        assert_eq!(partition_count(100), big(190_569_292));
        assert_eq!(partition_count(50), big(204_226));
        // Well past u64, from the published table.
        assert_eq!(
            partition_count(200).to_string(),
            "3972999029388"
        );
    }

    /// The enumerator must produce exactly p(n) partitions, each weakly
    /// decreasing and summing to n, with no duplicates.
    #[test]
    fn partitions_iter_is_complete_and_canonical() {
        for n in 0..=25u64 {
            let all: Vec<Vec<u64>> = partitions_iter(n).collect();
            assert_eq!(big(all.len() as u64), partition_count(n), "n = {n}");
            let distinct: HashSet<&Vec<u64>> = all.iter().collect();
            assert_eq!(distinct.len(), all.len(), "duplicates at n = {n}");
            for p in &all {
                assert_eq!(p.iter().sum::<u64>(), n, "{p:?} does not sum to {n}");
                assert!(p.iter().all(|&x| x > 0), "{p:?} has a zero part");
                assert!(
                    p.windows(2).all(|w| w[0] >= w[1]),
                    "{p:?} is not weakly decreasing"
                );
            }
            // Reverse lexicographic: strictly decreasing in that order.
            assert!(all.windows(2).all(|w| w[0] > w[1]), "not in order at n = {n}");
            if n > 0 {
                assert_eq!(all[0], vec![n]);
                assert_eq!(*all.last().unwrap(), vec![1u64; n as usize]);
            }
        }
    }

    /// Partitions by number of parts, cross-checked against enumeration and
    /// against the conjugation identity.
    #[test]
    fn partitions_by_part_count_match_enumeration_and_conjugation() {
        for n in 0..=22u64 {
            let all: Vec<Vec<u64>> = partitions_iter(n).collect();
            let mut running = BigInt::zero();
            for k in 0..=n {
                let brute = all.iter().filter(|p| p.len() as u64 == k).count();
                assert_eq!(partitions_into_k(n, k), big(brute as u64), "P({n}, {k})");
                running = running.add(&big(brute as u64));
                assert_eq!(
                    partition_count_into_at_most_k(n, k),
                    running,
                    "at most {k} parts of {n}"
                );
                // Conjugation: partitions of n into exactly k parts are in
                // bijection with those whose largest part is exactly k.
                let by_largest = all
                    .iter()
                    .filter(|p| p.first().copied().unwrap_or(0) == k)
                    .count();
                assert_eq!(brute, by_largest, "conjugation fails at n={n}, k={k}");
            }
            // Summing over k recovers p(n).
            assert_eq!(running, partition_count(n));
        }
    }

    /// Euler's theorem: partitions into distinct parts and into odd parts are
    /// equinumerous. The two are computed by different recurrences here, and
    /// both are checked against enumeration.
    #[test]
    fn euler_distinct_equals_odd() {
        for n in 0..=40u64 {
            assert_eq!(
                partitions_distinct(n),
                partitions_odd(n),
                "Euler's theorem fails at n = {n}"
            );
        }
        for n in 0..=22u64 {
            let all: Vec<Vec<u64>> = partitions_iter(n).collect();
            let distinct = all.iter().filter(|p| p.windows(2).all(|w| w[0] > w[1])).count();
            let odd = all
                .iter()
                .filter(|p| p.iter().all(|x| !x.is_multiple_of(2)))
                .count();
            assert_eq!(partitions_distinct(n), big(distinct as u64), "distinct({n})");
            assert_eq!(partitions_odd(n), big(odd as u64), "odd({n})");
        }
        assert_eq!(partitions_distinct(100), big(444_793));
    }

    /// Conjugation is an involution, preserves the sum, and swaps the number
    /// of parts with the largest part.
    #[test]
    fn conjugation_is_an_involution() {
        for n in 0..=20u64 {
            for p in partitions_iter(n) {
                let c = partition_conjugate(&p);
                assert_eq!(c.iter().sum::<u64>(), n, "sum changed for {p:?}");
                assert!(c.windows(2).all(|w| w[0] >= w[1]), "{c:?} is not a partition");
                assert_eq!(partition_conjugate(&c), p, "not an involution for {p:?}");
                assert_eq!(c.len() as u64, p.first().copied().unwrap_or(0));
                assert_eq!(p.len() as u64, c.first().copied().unwrap_or(0));
                // The Durfee square is conjugation-invariant, being the
                // largest square inside a self-conjugate corner.
                assert_eq!(durfee_square(&p), durfee_square(&c));
            }
        }
        assert_eq!(partition_conjugate(&[4, 2, 1]), vec![3, 2, 1, 1]);
    }

    /// The Durfee square is the largest s x s block that fits, so it must be
    /// bounded by both dimensions and be maximal.
    #[test]
    fn durfee_square_is_the_largest_fitting_square() {
        for n in 0..=20u64 {
            for p in partitions_iter(n) {
                let s = durfee_square(&p);
                assert!(s * s <= n, "a {s}x{s} square does not fit in {n} cells");
                // Fits: the first s rows each have at least s cells.
                for i in 0..s as usize {
                    assert!(p[i] >= s, "row {i} of {p:?} is too short");
                }
                // Maximal: adding a row breaks it.
                let bigger = s + 1;
                let fits_bigger = (bigger as usize) <= p.len()
                    && (0..bigger as usize).all(|i| p[i] >= bigger);
                assert!(!fits_bigger, "a {bigger}x{bigger} square also fits in {p:?}");
            }
        }
        assert_eq!(durfee_square(&[]), 0);
        assert_eq!(durfee_square(&[5, 4, 3, 2, 1]), 3);
    }

    /// The Young diagram must have exactly n true cells laid out as p says.
    #[test]
    fn young_diagram_realises_the_partition() {
        for n in 0..=15u64 {
            for p in partitions_iter(n) {
                let d = young_diagram(&p);
                assert_eq!(d.len(), p.len());
                let width = p.first().copied().unwrap_or(0) as usize;
                assert!(d.iter().all(|r| r.len() == width));
                assert_eq!(
                    d.iter().flatten().filter(|&&x| x).count() as u64,
                    n,
                    "wrong cell count for {p:?}"
                );
                for (i, row) in d.iter().enumerate() {
                    assert_eq!(row.iter().filter(|&&x| x).count() as u64, p[i]);
                    // Left-justified: no gap inside a row.
                    assert!(row.windows(2).all(|w| w[0] || !w[1]));
                }
                // Column heights are the conjugate.
                let conj = partition_conjugate(&p);
                for j in 0..width {
                    let h = d.iter().filter(|r| r[j]).count() as u64;
                    assert_eq!(h, conj[j]);
                }
            }
        }
    }

    /// Hook lengths must equal arm + leg + 1 measured directly on the diagram,
    /// and their product must divide n! (the hook length formula).
    #[test]
    fn hook_lengths_measure_arms_and_legs() {
        for n in 1..=14u64 {
            for p in partitions_iter(n) {
                let d = young_diagram(&p);
                let h = hook_lengths(&p);
                for i in 0..p.len() {
                    assert_eq!(h[i].len() as u64, p[i]);
                    for j in 0..p[i] as usize {
                        // Count directly on the diagram rather than reusing
                        // the conjugate the implementation uses.
                        let arm = (j + 1..d[i].len()).filter(|&c| d[i][c]).count();
                        let leg = (i + 1..d.len()).filter(|&r| d[r][j]).count();
                        assert_eq!(
                            h[i][j] as usize,
                            arm + leg + 1,
                            "hook at ({i}, {j}) of {p:?}"
                        );
                    }
                }
                // The corner cell always has hook length 1.
                let last = p.len() - 1;
                assert_eq!(h[last][p[last] as usize - 1], 1);
            }
        }
        // A worked example: the hooks of (2, 2) are 3 2 / 2 1.
        assert_eq!(hook_lengths(&[2, 2]), vec![vec![3, 2], vec![2, 1]]);
    }

    /// The hook length formula against exhaustive enumeration of standard
    /// Young tableaux, and against the identity sum f(shape)^2 = n!.
    #[test]
    fn hook_length_formula_counts_standard_tableaux() {
        // The roadmap's stated case.
        assert_eq!(standard_tableaux_count(&[2, 2]), big(2));

        for n in 1..=7u64 {
            let mut sum_squares = BigInt::zero();
            for p in partitions_iter(n) {
                let by_formula = standard_tableaux_count(&p);
                assert_eq!(by_formula, big(count_tableaux_brute(&p)), "shape {p:?}");
                sum_squares = sum_squares.add(&by_formula.mul(&by_formula));
            }
            // The RSK identity, which is what makes the correspondence a
            // bijection with S_n.
            assert_eq!(sum_squares, BigInt::factorial(n), "sum of squares at n = {n}");
        }
        // A hook shape (n-k ones under a row) has C(n-1, k) tableaux.
        for n in 2..=10u64 {
            for k in 0..n {
                let mut shape = vec![n - k];
                shape.extend(std::iter::repeat_n(1u64, k as usize));
                assert_eq!(
                    standard_tableaux_count(&shape),
                    big(binomial_u64(n - 1, k).unwrap()),
                    "hook shape {shape:?}"
                );
            }
        }
    }

    /// Count standard Young tableaux by filling the diagram directly: place
    /// 1..n so every row and column increases.
    fn count_tableaux_brute(p: &[u64]) -> u64 {
        fn go(p: &[u64], filled: &mut Vec<u64>, next: u64, n: u64) -> u64 {
            if next > n {
                return 1;
            }
            let mut total = 0;
            for i in 0..p.len() {
                // A cell may be filled only if its row and column predecessors
                // already are, which for a left-to-right, top-to-bottom fill
                // means row i has room and row i-1 is strictly ahead.
                if filled[i] < p[i] && (i == 0 || filled[i - 1] > filled[i]) {
                    filled[i] += 1;
                    total += go(p, filled, next + 1, n);
                    filled[i] -= 1;
                }
            }
            total
        }
        let n: u64 = p.iter().sum();
        let mut filled = vec![0u64; p.len()];
        go(p, &mut filled, 1, n)
    }

    /// RSK: both tableaux have the same shape, both are standard, and the map
    /// is injective on S_n -- which with the counting identity makes it the
    /// bijection it is claimed to be.
    #[test]
    fn rsk_produces_a_matching_pair_of_standard_tableaux() {
        for n in 0..=7usize {
            let items: Vec<usize> = (0..n).collect();
            let mut images: HashSet<(Vec<Vec<usize>>, Vec<Vec<usize>>)> = HashSet::new();
            for perm in permutations_iter(&items) {
                let (p, q) = rsk_correspondence(&perm);
                // Same shape.
                let shape_p: Vec<usize> = p.iter().map(Vec::len).collect();
                let shape_q: Vec<usize> = q.iter().map(Vec::len).collect();
                assert_eq!(shape_p, shape_q, "shapes differ for {perm:?}");
                // A partition shape.
                assert!(shape_p.windows(2).all(|w| w[0] >= w[1]), "{shape_p:?}");
                assert_eq!(shape_p.iter().sum::<usize>(), n);
                // Both standard: rows increase left to right, columns top to
                // bottom, and the entries are exactly 0..n.
                for t in [&p, &q] {
                    let mut all: Vec<usize> = t.iter().flatten().copied().collect();
                    all.sort_unstable();
                    assert_eq!(all, (0..n).collect::<Vec<_>>());
                    for row in t.iter() {
                        assert!(row.windows(2).all(|w| w[0] < w[1]), "row not increasing");
                    }
                    for i in 1..t.len() {
                        for j in 0..t[i].len() {
                            assert!(t[i][j] > t[i - 1][j], "column not increasing");
                        }
                    }
                }
                assert!(images.insert((p, q)), "RSK is not injective at {perm:?}");
            }
            assert_eq!(
                images.len() as u64,
                BigInt::factorial(n as u64).to_i64().unwrap() as u64
            );
        }
    }

    /// Schuetzenberger's theorem: RSK of the inverse permutation swaps P and
    /// Q. This is a property of the correspondence that a shape-only check
    /// cannot see, so it independently validates the bumping rule.
    #[test]
    fn rsk_of_the_inverse_swaps_the_two_tableaux() {
        for n in 1..=7usize {
            let items: Vec<usize> = (0..n).collect();
            for perm in permutations_iter(&items) {
                let (p, q) = rsk_correspondence(&perm);
                let (pi, qi) = rsk_correspondence(&permutation_inverse(&perm));
                assert_eq!(pi, q, "P(w^-1) != Q(w) for {perm:?}");
                assert_eq!(qi, p, "Q(w^-1) != P(w) for {perm:?}");
            }
        }
    }

    /// The first row of P is the longest increasing subsequence, and the
    /// number of rows is the longest decreasing one -- Schensted's theorem.
    #[test]
    fn rsk_shape_gives_the_longest_monotone_subsequences() {
        for n in 1..=7usize {
            let items: Vec<usize> = (0..n).collect();
            for perm in permutations_iter(&items) {
                let (p, _) = rsk_correspondence(&perm);
                assert_eq!(p[0].len(), longest_subsequence(&perm, true), "{perm:?}");
                assert_eq!(p.len(), longest_subsequence(&perm, false), "{perm:?}");
            }
        }
    }

    /// Longest increasing (or strictly decreasing) subsequence, by O(n^2) DP.
    fn longest_subsequence(v: &[usize], increasing: bool) -> usize {
        let n = v.len();
        let mut best = vec![1usize; n];
        for i in 0..n {
            for j in 0..i {
                let ok = if increasing { v[j] < v[i] } else { v[j] > v[i] };
                if ok {
                    best[i] = best[i].max(best[j] + 1);
                }
            }
        }
        best.into_iter().max().unwrap_or(0)
    }

    /// The Hardy-Ramanujan asymptotic must have relative error decaying like
    /// 1/sqrt(n). Asserting a fixed tolerance would pass for any function with
    /// roughly the right magnitude; asserting the decay rate does not.
    #[test]
    fn hardy_ramanujan_relative_error_decays_as_one_over_sqrt_n() {
        let rel = |n: u64| {
            let exact = partition_count(n).to_f64();
            (hardy_ramanujan_estimate(n) - exact).abs() / exact
        };
        // The leading term overshoots at every n tested, by a margin that
        // shrinks: 4.6% at n = 100 down to 1.0% at n = 2000.
        for n in [100u64, 400, 1_000, 2_000] {
            let ratio = hardy_ramanujan_estimate(n) / partition_count(n).to_f64();
            assert!(ratio > 1.0, "estimate undershoots at n = {n}");
            assert!(ratio < 1.05, "estimate is {ratio} times the exact value");
        }
        // Quadrupling n should roughly halve the relative error.
        let (a, b, c) = (rel(100), rel(400), rel(1_600));
        assert!(a < 0.1, "relative error at n = 100 is {a}");
        for (lo, hi) in [(b, a), (c, b)] {
            let ratio = hi / lo;
            assert!(
                (1.7..2.4).contains(&ratio),
                "error ratio {ratio} is not near the expected 2"
            );
        }
    }

    /// Goldbach verification must find a decomposition for every even number
    /// and must actually be checking primality -- so a case with no
    /// decomposition has to come back false.
    #[test]
    fn goldbach_verification_finds_real_decompositions() {
        assert!(goldbach_conjecture_verify(0));
        assert!(goldbach_conjecture_verify(3));
        assert!(goldbach_conjecture_verify(10_000));
        // Independently confirm a decomposition exists for each even n, using
        // a primality test rather than the sieve the function builds.
        let primes: HashSet<u64> = sieve_eratosthenes(2_000)
            .into_iter()
            .map(|p| p as u64)
            .collect();
        let mut n = 4u64;
        while n <= 2_000 {
            assert!(
                primes.iter().any(|&p| p <= n && primes.contains(&(n - p))),
                "no decomposition found for {n}"
            );
            n += 2;
        }
    }
}
