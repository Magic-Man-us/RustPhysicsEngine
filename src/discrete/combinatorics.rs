//! Counting, enumeration, and the permutation group.
//!
//! Three kinds of function live here. Counting functions return a `BigInt`
//! whenever the value outgrows 64 bits, which is almost immediately -- the
//! Bell numbers pass `u64::MAX` at n = 25 and the Catalan numbers at n = 33.
//! Enumeration functions return iterators that generate one object at a time
//! rather than materialising the whole family. The permutation functions
//! treat a `&[usize]` as the one-line form of a bijection on `0..n`, so
//! `p[i]` is the image of `i`.

use crate::discrete::number_theory::{divisors, euler_phi, multiplicative_order};
use crate::discrete::partitions::{partition_count_into_at_most_k, partitions_into_k};
use crate::exact::bigint::BigInt;
use crate::exact::polynomial::PolyQ;
use crate::exact::rational::Rational;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

// ---------------------------------------------------------------------------
// Binomials and multinomials
// ---------------------------------------------------------------------------

/// `C(n, k)` in `u64`, exactly when the result fits, otherwise `None`.
///
/// Multiplies and divides alternately so the running value is always an exact
/// binomial coefficient and therefore an integer: after step `i` the value is
/// `C(n, i + 1)`.
///
/// The running product before the division is `C(n, i+1) * (i+1)`, which is up
/// to `k` times the answer, so doing this in `u64` would report overflow for
/// results that fit. It runs in `u128` instead and tests the *coefficient*
/// against `u64::MAX`. Since `k` is folded to `min(k, n-k)`, the coefficient
/// only increases along the loop, so passing the bound once is final.
#[must_use]
pub fn binomial_u64(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc.checked_mul(u128::from(n - i))?;
        acc /= u128::from(i + 1);
        if acc > u128::from(u64::MAX) {
            return None;
        }
    }
    Some(acc as u64)
}

/// `C(n, k) mod p` for prime `p`, by Lucas's theorem.
///
/// Lucas reduces the coefficient to a product of coefficients of the base-`p`
/// digits, each of which is below `p` and so computable directly. A digit of
/// `k` exceeding the matching digit of `n` makes the whole product zero.
///
/// # Panics
/// Panics if `p` is zero or one. The result is only correct for prime `p`.
#[must_use]
pub fn binomial_mod_p(mut n: u64, mut k: u64, p: u64) -> u64 {
    assert!(p > 1, "modulus must exceed 1");
    let mut acc: u64 = 1;
    while n > 0 || k > 0 {
        let (nd, kd) = (n % p, k % p);
        if kd > nd {
            return 0;
        }
        // Both digits are below p, so this small binomial fits and is then
        // reduced; p may be large, so reduce through u128.
        acc = ((acc as u128 * small_binomial_mod(nd, kd, p) as u128) % p as u128) as u64;
        n /= p;
        k /= p;
    }
    acc % p
}

/// `C(n, k) mod p` for `n < p` prime, via factorials and Fermat inversion.
fn small_binomial_mod(n: u64, k: u64, p: u64) -> u64 {
    let mut num: u128 = 1;
    let mut den: u128 = 1;
    for i in 0..k {
        num = num * ((n - i) % p) as u128 % p as u128;
        den = den * ((i + 1) % p) as u128 % p as u128;
    }
    let inv = crate::discrete::number_theory::mod_pow_u64(den as u64, p - 2, p);
    (num * inv as u128 % p as u128) as u64
}

/// The multinomial `(sum ks)! / prod(ks!)`.
///
/// Built as a product of binomials rather than a ratio of factorials, so
/// every intermediate is itself an integer count.
#[must_use]
pub fn multinomial(ks: &[u64]) -> BigInt {
    let mut acc = BigInt::one();
    let mut running = 0u64;
    for &k in ks {
        running += k;
        acc = acc.mul(&BigInt::binomial(running, k));
    }
    acc
}

/// The falling factorial `n * (n-1) * ... * (n-k+1)`, or `None` on overflow.
#[must_use]
pub fn permutations_count(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    let mut acc: u64 = 1;
    for i in 0..k {
        acc = acc.checked_mul(n - i)?;
    }
    Some(acc)
}

// ---------------------------------------------------------------------------
// Permutation enumeration
// ---------------------------------------------------------------------------

/// All permutations of `items`, by Heap's algorithm.
///
/// Heap's algorithm reaches each of the `n!` arrangements with a single
/// transposition per step, so generating the whole family costs `O(n!)` swaps
/// rather than `O(n * n!)` copies -- the copies here are only to hand out
/// owned results. The order is Heap's, not lexicographic.
pub fn permutations_iter(items: &[usize]) -> impl Iterator<Item = Vec<usize>> + use<> {
    HeapPermutations {
        state: items.to_vec(),
        // c[i] is Heap's counter for level i.
        counters: vec![0usize; items.len()],
        level: 0,
        emitted_first: false,
        done: false,
    }
}

struct HeapPermutations {
    state: Vec<usize>,
    counters: Vec<usize>,
    level: usize,
    /// The starting arrangement is emitted before any swap. An empty slice
    /// therefore yields exactly one item, the empty permutation, which is the
    /// 0! = 1 the counting identities expect.
    emitted_first: bool,
    done: bool,
}

impl Iterator for HeapPermutations {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done {
            return None;
        }
        if !self.emitted_first {
            self.emitted_first = true;
            return Some(self.state.clone());
        }
        let n = self.state.len();
        while self.level < n {
            if self.counters[self.level] < self.level {
                // Heap's swap rule: parity of the level decides the partner.
                if self.level.is_multiple_of(2) {
                    self.state.swap(0, self.level);
                } else {
                    self.state.swap(self.counters[self.level], self.level);
                }
                self.counters[self.level] += 1;
                self.level = 0;
                return Some(self.state.clone());
            }
            self.counters[self.level] = 0;
            self.level += 1;
        }
        self.done = true;
        None
    }
}

/// Advances `p` to the next permutation in lexicographic order in place.
///
/// Returns `false` when `p` is already the last (descending) arrangement, in
/// which case `p` is left untouched. This is the standard pivot-and-reverse
/// step: find the rightmost ascent, swap its left element with the smallest
/// larger element to its right, then reverse the now-descending suffix.
pub fn permutations_lex_next(p: &mut [usize]) -> bool {
    let n = p.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && p[i - 1] >= p[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let pivot = i - 1;
    let mut j = n - 1;
    while p[j] <= p[pivot] {
        j -= 1;
    }
    p.swap(pivot, j);
    p[i..].reverse();
    true
}

/// The permutation of `0..n_items` at the given lexicographic `index`, by the
/// factorial number system.
///
/// Digit `i` of the factoradic expansion says how many of the still-unused
/// symbols to skip, which is exactly what selecting the `index`-th
/// lexicographic arrangement does.
///
/// # Panics
/// Panics if `index` is negative or at least `n_items!`.
#[must_use]
pub fn nth_permutation(n_items: usize, index: &BigInt) -> Vec<usize> {
    assert!(!index.is_negative(), "index must be non-negative");
    let total = BigInt::factorial(n_items as u64);
    assert!(*index < total, "index must be below n_items!");
    let mut rest = index.clone();
    let mut pool: Vec<usize> = (0..n_items).collect();
    let mut out = Vec::with_capacity(n_items);
    for j in 0..n_items {
        // Each of the (n - 1 - j)! arrangements of the suffix shares a first
        // symbol, so the quotient by that factorial is the position in the
        // remaining pool and the remainder indexes within the suffix.
        let block = BigInt::factorial((n_items - 1 - j) as u64);
        let (q, r) = rest.div_rem(&block);
        let pick = q.to_i64().unwrap() as usize;
        out.push(pool.remove(pick));
        rest = r;
    }
    out
}

/// The lexicographic index of `p` among the permutations of its own symbols.
///
/// Inverse of [`nth_permutation`]: counts, at each position, how many unused
/// symbols are smaller than the one chosen, and weights that by the factorial
/// of the remaining length.
#[must_use]
pub fn permutation_index(p: &[usize]) -> BigInt {
    let n = p.len();
    let mut idx = BigInt::zero();
    for i in 0..n {
        let smaller = p[i + 1..].iter().filter(|&&x| x < p[i]).count();
        let weight = BigInt::factorial((n - 1 - i) as u64);
        idx = idx.add(&BigInt::from_u64(smaller as u64).mul(&weight));
    }
    idx
}

/// The `k`-subsets of `0..n`, each sorted ascending, in lexicographic order.
pub fn combinations_iter(n: usize, k: usize) -> impl Iterator<Item = Vec<usize>> + use<> {
    Combinations {
        n,
        k,
        current: if k <= n { Some((0..k).collect()) } else { None },
    }
}

struct Combinations {
    n: usize,
    k: usize,
    current: Option<Vec<usize>>,
}

impl Iterator for Combinations {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let cur = self.current.take()?;
        let out = cur.clone();
        // Advance: find the rightmost element still below its ceiling.
        let mut next = cur;
        let mut i = self.k;
        loop {
            if i == 0 {
                self.current = None;
                return Some(out);
            }
            i -= 1;
            if next[i] != i + self.n - self.k {
                break;
            }
        }
        next[i] += 1;
        for j in i + 1..self.k {
            next[j] = next[j - 1] + 1;
        }
        self.current = Some(next);
        Some(out)
    }
}

/// The `k`-multisets over `0..n`, each non-decreasing, in lexicographic order.
///
/// Same shape as [`combinations_iter`] with the strict ceiling relaxed:
/// entries may repeat, so position `j` is capped at `n - 1` rather than at
/// `j + n - k`.
pub fn combinations_with_replacement_iter(
    n: usize,
    k: usize,
) -> impl Iterator<Item = Vec<usize>> + use<> {
    MultiCombinations {
        n,
        k,
        current: if k == 0 {
            Some(Vec::new())
        } else if n == 0 {
            None
        } else {
            Some(vec![0usize; k])
        },
    }
}

struct MultiCombinations {
    n: usize,
    k: usize,
    current: Option<Vec<usize>>,
}

impl Iterator for MultiCombinations {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let cur = self.current.take()?;
        let out = cur.clone();
        let mut next = cur;
        let mut i = self.k;
        loop {
            if i == 0 {
                self.current = None;
                return Some(out);
            }
            i -= 1;
            if next[i] != self.n - 1 {
                break;
            }
        }
        next[i] += 1;
        for j in i + 1..self.k {
            next[j] = next[i];
        }
        self.current = Some(next);
        Some(out)
    }
}

/// The `2^n_bits` reflected binary Gray codes in order.
///
/// `g(i) = i XOR (i >> 1)`, whose consecutive values differ in exactly one
/// bit.
///
/// # Panics
/// Panics if `n_bits` exceeds 63.
pub fn gray_code_iter(n_bits: u32) -> impl Iterator<Item = u64> + use<> {
    assert!(n_bits <= 63, "n_bits must be at most 63");
    (0u64..(1u64 << n_bits)).map(|i| i ^ (i >> 1))
}

/// The `2^n` subsets of `0..n` as bitmasks, in increasing numeric order.
///
/// # Panics
/// Panics if `n` exceeds 63.
pub fn subsets_iter(n: u32) -> impl Iterator<Item = u64> + use<> {
    assert!(n <= 63, "n must be at most 63");
    0u64..(1u64 << n)
}

// ---------------------------------------------------------------------------
// Derangements and random permutations
// ---------------------------------------------------------------------------

/// The number of permutations of `n` symbols with no fixed point.
///
/// Uses the recurrence `D(n) = (n-1) (D(n-1) + D(n-2))`, which is exact in
/// integers, rather than the alternating factorial sum, which alternates in
/// sign and would need cancellation.
#[must_use]
pub fn derangements_count(n: u64) -> BigInt {
    if n == 0 {
        return BigInt::one();
    }
    if n == 1 {
        return BigInt::zero();
    }
    let mut prev = BigInt::one(); // D(0)
    let mut cur = BigInt::zero(); // D(1)
    for i in 2..=n {
        let next = BigInt::from_u64(i - 1).mul(&cur.add(&prev));
        prev = cur;
        cur = next;
    }
    cur
}

/// True when `p` is a permutation with no fixed point.
#[must_use]
pub fn is_derangement(p: &[usize]) -> bool {
    is_permutation(p) && p.iter().enumerate().all(|(i, &x)| i != x)
}

/// True when `p` is a bijection on `0..p.len()`.
#[must_use]
pub fn is_permutation(p: &[usize]) -> bool {
    let n = p.len();
    let mut seen = vec![false; n];
    for &x in p {
        if x >= n || seen[x] {
            return false;
        }
        seen[x] = true;
    }
    true
}

/// A uniformly random permutation of `0..n`, by Fisher-Yates.
///
/// Each step picks uniformly from the untouched suffix, which gives every one
/// of the `n!` arrangements the same probability.
pub fn random_permutation(n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, bounded(rng, i as u64 + 1) as usize);
    }
    p
}

/// A value in `0..bound`, taken from the high bits of the generator.
///
/// `next_u64() % bound` would be wrong here. The generator is a linear
/// congruential one modulo `2^64`, and bit `b` of such a sequence has period
/// `2^(b+1)`: the lowest bit merely alternates, the next cycles with period
/// four, and so on. A small modulus reads exactly those bits, so shuffling a
/// six-element array that way produces a handful of arrangements on repeat
/// rather than a sample of the 720. Multiplying by the bound and keeping the
/// top half of the 128-bit product reads the high bits instead, which carry
/// the full period. The residual bias is at most `bound / 2^64`.
fn bounded(rng: &mut Rng, bound: u64) -> u64 {
    ((u128::from(rng.next_u64()) * u128::from(bound)) >> 64) as u64
}

/// A uniformly random derangement of `0..n`, by rejection.
///
/// The density of derangements tends to `1/e`, so the expected number of
/// draws is about 2.72 regardless of `n` -- rejection is the cheap method
/// here, not a fallback. Returns the empty permutation for `n = 0` and panics
/// for `n = 1`, which has no derangement.
///
/// # Panics
/// Panics if `n` is 1.
pub fn random_derangement(n: usize, rng: &mut Rng) -> Vec<usize> {
    assert!(n != 1, "no derangement of a single symbol exists");
    loop {
        let p = random_permutation(n, rng);
        if is_derangement(&p) {
            return p;
        }
    }
}

// ---------------------------------------------------------------------------
// The permutation group
// ---------------------------------------------------------------------------

/// The composition `a` after `b`: `(a . b)(i) = a[b[i]]`.
///
/// # Panics
/// Panics if the two permutations have different lengths.
#[must_use]
pub fn permutation_compose(a: &[usize], b: &[usize]) -> Vec<usize> {
    assert_eq!(a.len(), b.len(), "permutations must have equal length");
    b.iter().map(|&i| a[i]).collect()
}

/// The inverse permutation.
#[must_use]
pub fn permutation_inverse(p: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; p.len()];
    for (i, &x) in p.iter().enumerate() {
        inv[x] = i;
    }
    inv
}

/// The cycle lengths of `p`, sorted descending.
///
/// This is the conjugacy class invariant: two permutations are conjugate in
/// the symmetric group exactly when their cycle types agree. Fixed points
/// count as cycles of length one, so the entries sum to `p.len()`.
#[must_use]
pub fn permutation_cycle_type(p: &[usize]) -> Vec<usize> {
    let mut lens: Vec<usize> = permutation_to_cycles(p).iter().map(Vec::len).collect();
    lens.sort_unstable_by(|a, b| b.cmp(a));
    lens
}

/// The order of `p` in the symmetric group: the lcm of its cycle lengths.
///
/// Returns a `BigInt` because the maximum order over `S_n` (Landau's
/// function) passes `u64::MAX` well before `n = 130`.
#[must_use]
pub fn permutation_order(p: &[usize]) -> BigInt {
    let mut acc = BigInt::one();
    for len in permutation_cycle_type(p) {
        acc = acc.lcm(&BigInt::from_u64(len as u64));
    }
    acc
}

/// The sign of `p`: `+1` for an even permutation, `-1` for an odd one.
///
/// A cycle of length `L` is a product of `L - 1` transpositions, so the sign
/// is `(-1)^(n - number of cycles)`.
#[must_use]
pub fn permutation_sign(p: &[usize]) -> i8 {
    let cycles = permutation_to_cycles(p).len();
    if (p.len() - cycles).is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// The disjoint cycles of `p`, each starting at its smallest element, ordered
/// by that element. Fixed points appear as one-element cycles.
#[must_use]
pub fn permutation_to_cycles(p: &[usize]) -> Vec<Vec<usize>> {
    let n = p.len();
    let mut seen = vec![false; n];
    let mut cycles = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut x = start;
        while !seen[x] {
            seen[x] = true;
            cycle.push(x);
            x = p[x];
        }
        cycles.push(cycle);
    }
    cycles
}

/// The permutation of `0..n` with the given disjoint cycles.
///
/// Symbols not mentioned are fixed. Each cycle maps every element to the next
/// one listed and the last back to the first.
///
/// # Panics
/// Panics if a symbol is at least `n` or appears in two cycles.
#[must_use]
pub fn permutation_from_cycles(n: usize, cycles: &[Vec<usize>]) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    let mut used = vec![false; n];
    for cycle in cycles {
        for (k, &x) in cycle.iter().enumerate() {
            assert!(x < n, "symbol {x} is outside 0..{n}");
            assert!(!used[x], "symbol {x} appears in two cycles");
            used[x] = true;
            p[x] = cycle[(k + 1) % cycle.len()];
        }
    }
    p
}

/// The permutation matrix `P` with `P[p[j], j] = 1`.
///
/// With this convention `P` applied to a coordinate vector moves the entry at
/// `j` to `p[j]`, so `permutation_matrix(compose(a, b))` is the product of the
/// two matrices in the same order.
#[must_use]
pub fn permutation_matrix(p: &[usize]) -> Matrix {
    let n = p.len();
    let mut m = Matrix::zeros(n, n);
    for (j, &i) in p.iter().enumerate() {
        m.set(i, j, 1.0);
    }
    m
}

// ---------------------------------------------------------------------------
// The classical counting numbers
// ---------------------------------------------------------------------------

/// Unsigned Stirling numbers of the first kind: the number of permutations of
/// `n` symbols with exactly `k` cycles.
///
/// Recurrence `c(n, k) = c(n-1, k-1) + (n-1) c(n-1, k)`: the new symbol is
/// either its own cycle or inserted after one of the `n-1` existing symbols.
#[must_use]
pub fn stirling_first(n: u64, k: u64) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    let (n, k) = (n as usize, k as usize);
    let mut row = vec![BigInt::zero(); k + 1];
    row[0] = BigInt::one(); // c(0, 0) = 1
    for i in 1..=n {
        let mut next = vec![BigInt::zero(); k + 1];
        for j in (1..=k.min(i)).rev() {
            next[j] = row[j - 1].add(&BigInt::from_u64((i - 1) as u64).mul(&row[j]));
        }
        row = next;
    }
    row[k].clone()
}

/// Stirling numbers of the second kind: the number of ways to partition `n`
/// labelled objects into exactly `k` non-empty unlabelled blocks.
///
/// Recurrence `S(n, k) = S(n-1, k-1) + k S(n-1, k)`: the new object either
/// opens a block of its own or joins one of the `k` existing ones.
#[must_use]
pub fn stirling_second(n: u64, k: u64) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    let (n, k) = (n as usize, k as usize);
    let mut row = vec![BigInt::zero(); k + 1];
    row[0] = BigInt::one();
    for i in 1..=n {
        let mut next = vec![BigInt::zero(); k + 1];
        for j in (1..=k.min(i)).rev() {
            next[j] = row[j - 1].add(&BigInt::from_u64(j as u64).mul(&row[j]));
        }
        row = next;
    }
    row[k].clone()
}

/// The `n`-th Bell number: the number of partitions of an `n`-element set.
#[must_use]
pub fn bell_number(n: u64) -> BigInt {
    bell_triangle(n).last().unwrap()[0].clone()
}

/// The first `n + 1` rows of the Bell (Peirce) triangle.
///
/// Row 0 is `[1]`; each later row starts with the last entry of the previous
/// row and each subsequent entry is the sum of its left neighbour and the
/// entry above that neighbour. Row `i` begins with the `i`-th Bell number.
#[must_use]
pub fn bell_triangle(n: u64) -> Vec<Vec<BigInt>> {
    let n = n as usize;
    let mut rows: Vec<Vec<BigInt>> = vec![vec![BigInt::one()]];
    for i in 1..=n {
        let prev = &rows[i - 1];
        let mut row = vec![prev[prev.len() - 1].clone()];
        for j in 1..=prev.len() {
            let v = row[j - 1].add(&prev[j - 1]);
            row.push(v);
        }
        rows.push(row);
    }
    rows
}

/// The `n`-th Catalan number, `C(2n, n) / (n + 1)`.
#[must_use]
pub fn catalan(n: u64) -> BigInt {
    BigInt::binomial(2 * n, n)
        .div_rem(&BigInt::from_u64(n + 1))
        .0
}

/// The `n`-th Catalan number modulo `m`, for any `m`.
///
/// Uses the convolution recurrence `C(n+1) = sum_i C(i) C(n-i)` rather than
/// the closed form. The closed form needs a division by `n + 1`, which has no
/// modular meaning when `n + 1` shares a factor with `m`; the convolution is
/// pure addition and multiplication and so is valid for every modulus.
/// Costs `O(n^2)`.
///
/// # Panics
/// Panics if `m` is zero.
#[must_use]
pub fn catalan_mod(n: u64, m: u64) -> u64 {
    assert!(m > 0, "modulus must be positive");
    let n = n as usize;
    let mut c = vec![0u64; n + 1];
    c[0] = 1 % m;
    for i in 1..=n {
        let mut acc: u128 = 0;
        for j in 0..i {
            acc += c[j] as u128 * c[i - 1 - j] as u128 % m as u128;
        }
        c[i] = (acc % m as u128) as u64;
    }
    c[n]
}

/// Eulerian number `A(n, k)`: permutations of `n` symbols with exactly `k`
/// ascents.
///
/// Recurrence `A(n, k) = (k+1) A(n-1, k) + (n-k) A(n-1, k-1)`.
#[must_use]
pub fn eulerian_number(n: u64, k: u64) -> BigInt {
    if n == 0 {
        return if k == 0 { BigInt::one() } else { BigInt::zero() };
    }
    if k >= n {
        return BigInt::zero();
    }
    let (n, k) = (n as usize, k as usize);
    let mut row = vec![BigInt::zero(); k + 1];
    row[0] = BigInt::one(); // A(1, 0) = 1
    for i in 2..=n {
        let mut next = vec![BigInt::zero(); k + 1];
        for j in 0..=k.min(i - 1) {
            let a = BigInt::from_u64((j + 1) as u64).mul(&row[j]);
            let b = if j == 0 {
                BigInt::zero()
            } else {
                BigInt::from_u64((i - j) as u64).mul(&row[j - 1])
            };
            next[j] = a.add(&b);
        }
        row = next;
    }
    row[k].clone()
}

/// Narayana number `N(n, k) = C(n, k) C(n, k-1) / n`, the number of Dyck paths
/// of semilength `n` with exactly `k` peaks. Defined for `1 <= k <= n`.
#[must_use]
pub fn narayana(n: u64, k: u64) -> BigInt {
    if n == 0 || k == 0 || k > n {
        return BigInt::zero();
    }
    BigInt::binomial(n, k)
        .mul(&BigInt::binomial(n, k - 1))
        .div_rem(&BigInt::from_u64(n))
        .0
}

/// The `n`-th Motzkin number: lattice paths from `(0,0)` to `(n,0)` with steps
/// up, down and level that never dip below the axis.
///
/// Recurrence `M(n+1) = M(n) + sum_i M(i) M(n-1-i)`.
#[must_use]
pub fn motzkin(n: u64) -> BigInt {
    let n = n as usize;
    let mut m = vec![BigInt::zero(); n + 1];
    m[0] = BigInt::one();
    for i in 1..=n {
        let mut acc = m[i - 1].clone();
        for j in 0..i.saturating_sub(1) {
            acc = acc.add(&m[j].mul(&m[i - 2 - j]));
        }
        m[i] = acc;
    }
    m[n].clone()
}

/// The `n`-th large Schroeder number: lattice paths from `(0,0)` to `(n,n)`
/// with steps east, north and diagonal that stay weakly below the diagonal.
///
/// Recurrence `3(2n-1) S(n-1) = (n+1) S(n) + (n-2) S(n-2)`, rearranged; done
/// here by the equivalent convolution `S(n) = S(n-1) + sum_i S(i) S(n-1-i)`.
#[must_use]
pub fn schroeder(n: u64) -> BigInt {
    let n = n as usize;
    let mut s = vec![BigInt::zero(); n + 1];
    s[0] = BigInt::one();
    for i in 1..=n {
        let mut acc = s[i - 1].clone();
        for j in 0..i {
            acc = acc.add(&s[j].mul(&s[i - 1 - j]));
        }
        s[i] = acc;
    }
    s[n].clone()
}

/// The Delannoy number `D(m, n)`: lattice paths from `(0,0)` to `(m,n)` with
/// east, north and diagonal steps.
#[must_use]
pub fn delannoy(m: u64, n: u64) -> BigInt {
    let (m, n) = (m as usize, n as usize);
    let mut row = vec![BigInt::one(); n + 1];
    for _ in 1..=m {
        let mut next = vec![BigInt::one(); n + 1];
        for j in 1..=n {
            // East, north, and diagonal predecessors.
            next[j] = next[j - 1].add(&row[j]).add(&row[j - 1]);
        }
        row = next;
    }
    row[n].clone()
}

/// The unsigned Lah number `L(n, k) = C(n-1, k-1) n! / k!`: the number of ways
/// to partition `n` labelled objects into `k` non-empty ordered lists.
#[must_use]
pub fn lah_number(n: u64, k: u64) -> BigInt {
    if n == 0 && k == 0 {
        return BigInt::one();
    }
    if k == 0 || k > n {
        return BigInt::zero();
    }
    BigInt::binomial(n - 1, k - 1)
        .mul(&BigInt::factorial(n))
        .div_rem(&BigInt::factorial(k))
        .0
}

/// The ballot number: the number of ways to count `p` votes for A and `q` for
/// B so that A is never behind.
///
/// Equal to `C(p+q, q) (p - q + 1) / (p + 1)`; zero when `q > p`.
#[must_use]
pub fn ballot_number(p: u64, q: u64) -> BigInt {
    if q > p {
        return BigInt::zero();
    }
    BigInt::binomial(p + q, q)
        .mul(&BigInt::from_u64(p - q + 1))
        .div_rem(&BigInt::from_u64(p + 1))
        .0
}

// ---------------------------------------------------------------------------
// Structured enumeration
// ---------------------------------------------------------------------------

/// The Dyck paths of semilength `n`, as step vectors of `2n` booleans where
/// `true` is an up step.
///
/// Every prefix has at least as many up steps as down steps and the whole path
/// balances, so there are `catalan(n)` of them. Generated in lexicographic
/// order with `false < true`.
pub fn dyck_paths_iter(n: usize) -> impl Iterator<Item = Vec<bool>> + use<> {
    DyckPaths {
        n,
        stack: vec![(Vec::new(), 0usize, 0usize)],
    }
}

struct DyckPaths {
    n: usize,
    /// Partial path, up steps used, down steps used.
    stack: Vec<(Vec<bool>, usize, usize)>,
}

impl Iterator for DyckPaths {
    type Item = Vec<bool>;

    fn next(&mut self) -> Option<Vec<bool>> {
        while let Some((path, up, down)) = self.stack.pop() {
            if up == self.n && down == self.n {
                return Some(path);
            }
            // Pushed in reverse so `true` (up) is explored first, which makes
            // the emitted order lexicographic with false < true reversed --
            // see the caller-visible ordering note above.
            if up < self.n {
                let mut p = path.clone();
                p.push(true);
                self.stack.push((p, up + 1, down));
            }
            if down < up {
                let mut p = path.clone();
                p.push(false);
                self.stack.push((p, up, down + 1));
            }
        }
        None
    }
}

/// The set partitions of `0..n`, as restricted growth strings.
///
/// Entry `i` of the string is the index of the block containing `i`. The
/// restriction is that a string starts at 0 and never jumps by more than one
/// above the running maximum, which makes the correspondence with partitions
/// exactly one-to-one -- block indices are forced to appear in order of their
/// smallest element, so relabelling the blocks cannot produce a duplicate.
/// There are `bell_number(n)` of them.
pub fn set_partitions_iter(n: usize) -> impl Iterator<Item = Vec<usize>> + use<> {
    RestrictedGrowth {
        n,
        a: vec![0usize; n],
        done: false,
        first: true,
    }
}

struct RestrictedGrowth {
    n: usize,
    /// The string itself; `a[i]` is the block index of element `i`.
    a: Vec<usize>,
    done: bool,
    first: bool,
}

impl Iterator for RestrictedGrowth {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done {
            return None;
        }
        if self.first {
            self.first = false;
            // All zeros: everything in one block. Also the sole answer for
            // n = 0, where the string is empty.
            return Some(self.a.clone());
        }
        if self.n == 0 {
            self.done = true;
            return None;
        }
        // Increment the rightmost position that can still grow. Position j may
        // hold anything up to one more than the largest index used before it,
        // so it can grow exactly when it is not already at that ceiling.
        let mut j = self.n - 1;
        loop {
            if j == 0 {
                self.done = true;
                return None;
            }
            let prefix_max = self.a[..j].iter().copied().max().unwrap();
            if self.a[j] <= prefix_max {
                self.a[j] += 1;
                // Everything to the right restarts at its own minimum.
                for t in j + 1..self.n {
                    self.a[t] = 0;
                }
                return Some(self.a.clone());
            }
            j -= 1;
        }
    }
}

/// The compositions of `n`: the ordered tuples of positive integers summing to
/// `n`. There are `2^(n-1)` for `n >= 1`, and one (the empty tuple) for `n = 0`.
///
/// Generated from the `n - 1` gap positions: a composition is exactly a choice
/// of which of the `n - 1` gaps between `n` units to cut.
pub fn compositions_iter(n: u64) -> impl Iterator<Item = Vec<u64>> + use<> {
    let gaps = n.saturating_sub(1) as u32;
    let total: u64 = if n == 0 { 1 } else { 1u64 << gaps };
    (0..total).map(move |mask| {
        if n == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut run = 1u64;
        for g in 0..gaps {
            if mask >> g & 1 == 1 {
                out.push(run);
                run = 1;
            } else {
                run += 1;
            }
        }
        out.push(run);
        out
    })
}

// ---------------------------------------------------------------------------
// Burnside and Polya
// ---------------------------------------------------------------------------

/// The number of necklaces: `k`-colourings of `n` beads in a cycle, counted up
/// to rotation.
///
/// Burnside over the cyclic group: the rotation by `j` fixes a colouring
/// exactly when the colouring is constant on the `gcd(j, n)` orbits, so the
/// count is `(1/n) sum_{d | n} phi(d) k^(n/d)`.
#[must_use]
pub fn necklaces_count(n: u64, k: u64) -> BigInt {
    if n == 0 {
        return BigInt::one();
    }
    let mut acc = BigInt::zero();
    for d in divisors(n) {
        acc = acc.add(&BigInt::from_u64(euler_phi(d)).mul(&BigInt::from_u64(k).pow(n / d)));
    }
    acc.div_rem(&BigInt::from_u64(n)).0
}

/// The number of bracelets: `k`-colourings of `n` beads in a cycle, counted up
/// to rotation *and* reflection.
///
/// Burnside over the dihedral group. The reflections contribute
/// `k^((n+1)/2)` each for odd `n`, and for even `n` split into `n/2` axes
/// through two beads (`k^(n/2 + 1)`) and `n/2` axes through two gaps
/// (`k^(n/2)`).
#[must_use]
pub fn bracelets_count(n: u64, k: u64) -> BigInt {
    if n == 0 {
        return BigInt::one();
    }
    let rotations = necklaces_count(n, k).mul(&BigInt::from_u64(n));
    let kb = BigInt::from_u64(k);
    let reflections = if n % 2 == 1 {
        BigInt::from_u64(n).mul(&kb.pow(n / 2 + 1))
    } else {
        BigInt::from_u64(n / 2).mul(&kb.pow(n / 2 + 1).add(&kb.pow(n / 2)))
    };
    rotations
        .add(&reflections)
        .div_rem(&BigInt::from_u64(2 * n))
        .0
}

/// Burnside's lemma: the number of orbits is the average number of points
/// fixed by a group element.
///
/// Takes one fixed-point count per group element, so the slice length is the
/// group order.
///
/// # Panics
/// Panics on an empty slice, and if the average is not an integer -- which
/// cannot happen for a genuine group action, so a non-zero remainder means the
/// caller's counts are not a group's.
#[must_use]
pub fn burnside_orbit_count(group_element_fixed_counts: &[BigInt]) -> BigInt {
    assert!(
        !group_element_fixed_counts.is_empty(),
        "the group must be non-empty"
    );
    let order = BigInt::from_u64(group_element_fixed_counts.len() as u64);
    let sum = group_element_fixed_counts
        .iter()
        .fold(BigInt::zero(), |a, b| a.add(b));
    let (q, r) = sum.div_rem(&order);
    assert!(
        r.is_zero(),
        "the fixed-point counts do not average to an integer"
    );
    q
}

/// Polya enumeration: the number of colourings with `colors` colours, given a
/// cycle index.
///
/// The cycle index of a group acting on `n` points is a polynomial in `n`
/// variables `a_1..a_n`. Polya's theorem with unweighted colours substitutes
/// the same value -- the number of colours -- for every variable, and the
/// result of that substitution is a polynomial in one variable. That single
/// variable form is what [`cycle_index_cyclic`], [`cycle_index_dihedral`] and
/// [`cycle_index_symmetric`] return and what this function evaluates, so the
/// specialisation happens once at construction rather than at every call.
///
/// # Panics
/// Panics if the value at `colors` is not an integer, which cannot happen for
/// a cycle index of a genuine group.
#[must_use]
pub fn polya_enumeration(cycle_index: &PolyQ, colors: u64) -> BigInt {
    let v = cycle_index.eval(&Rational::from_int(BigInt::from_u64(colors)));
    assert!(v.is_integer(), "a cycle index must take integer values");
    v.floor()
}

/// The cycle index of the cyclic group `C_n` acting on `n` points, with every
/// variable already set to the colour count: `(1/n) sum_{d | n} phi(d) x^(n/d)`.
#[must_use]
pub fn cycle_index_cyclic(n: u64) -> PolyQ {
    if n == 0 {
        return PolyQ::from_i64s(&[1]);
    }
    let mut c = vec![Rational::zero(); n as usize + 1];
    for d in divisors(n) {
        let e = (n / d) as usize;
        c[e] = c[e].add(&Rational::from_i64(euler_phi(d) as i64, n as i64));
    }
    PolyQ::new(c)
}

/// The cycle index of the dihedral group `D_n` acting on `n` points, with
/// every variable set to the colour count.
///
/// Half the cyclic index plus the reflection average.
#[must_use]
pub fn cycle_index_dihedral(n: u64) -> PolyQ {
    if n == 0 {
        return PolyQ::from_i64s(&[1]);
    }
    let rot = cycle_index_cyclic(n).mul_scalar(&Rational::from_i64(1, 2));
    let mut c = vec![Rational::zero(); n as usize + 2];
    if n % 2 == 1 {
        // n reflections, each with (n+1)/2 cycles.
        let e = (n / 2 + 1) as usize;
        c[e] = c[e].add(&Rational::from_i64(1, 2));
    } else {
        // n/2 through opposite beads, n/2 through opposite gaps.
        let e1 = (n / 2 + 1) as usize;
        let e2 = (n / 2) as usize;
        c[e1] = c[e1].add(&Rational::from_i64(1, 4));
        c[e2] = c[e2].add(&Rational::from_i64(1, 4));
    }
    rot.add(&PolyQ::new(c))
}

/// The cycle index of the symmetric group `S_n` acting on `n` points, with
/// every variable set to the colour count.
///
/// Averaging over all of `S_n` collapses to the rising factorial
/// `x (x+1) ... (x+n-1) / n!`, which is `C(x + n - 1, n)` -- the count of
/// `n`-multisets, exactly what "colourings up to any relabelling of the
/// points" means.
#[must_use]
pub fn cycle_index_symmetric(n: u64) -> PolyQ {
    let mut p = PolyQ::from_i64s(&[1]);
    for i in 0..n {
        // Multiply by (x + i).
        p = p.mul(&PolyQ::from_i64s(&[i as i64, 1]));
    }
    p.div_scalar(&Rational::from_int(BigInt::factorial(n)))
        .expect("n! is non-zero")
}

/// Inclusion-exclusion over `n` sets.
///
/// `sizes(s)` must return the size of the intersection of the sets indexed by
/// the sorted, non-empty slice `s`. Returns the size of the union. Costs
/// `2^n - 1` calls.
///
/// # Panics
/// Panics if `n` exceeds 63.
pub fn inclusion_exclusion(sizes: &dyn Fn(&[usize]) -> BigInt, n: usize) -> BigInt {
    assert!(n <= 63, "n must be at most 63");
    let mut total = BigInt::zero();
    for mask in 1u64..(1u64 << n) {
        let subset: Vec<usize> = (0..n).filter(|&i| mask >> i & 1 == 1).collect();
        let term = sizes(&subset);
        if !subset.len().is_multiple_of(2) {
            total = total.add(&term);
        } else {
            total = total.sub(&term);
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Named puzzles and constructions
// ---------------------------------------------------------------------------

/// The guaranteed occupancy of the fullest box: `ceil(items / boxes)`.
///
/// The pigeonhole principle in its quantitative form -- some box holds at
/// least this many, and a balanced distribution shows the bound is attained.
///
/// # Panics
/// Panics if `boxes` is zero.
#[must_use]
pub fn pigeonhole_min_overlap(items: u64, boxes: u64) -> u64 {
    assert!(boxes > 0, "there must be at least one box");
    items.div_ceil(boxes)
}

/// The Ramsey number `R(s, t)` when it is known exactly, otherwise `None`.
///
/// Only nine non-trivial values are known; everything beyond `R(4,5) = 25`
/// and the `R(3, t)` ladder is open, so this returns `None` rather than a
/// bound.
#[must_use]
pub fn ramsey_known(s: u64, t: u64) -> Option<u64> {
    let (a, b) = if s <= t { (s, t) } else { (t, s) };
    match (a, b) {
        (0, _) => Some(0),
        (1, _) => Some(1),
        // A monochromatic edge or a b-clique in the other colour: R(2, b) = b.
        (2, _) => Some(b),
        (3, 3) => Some(6),
        (3, 4) => Some(9),
        (3, 5) => Some(14),
        (3, 6) => Some(18),
        (3, 7) => Some(23),
        (3, 8) => Some(28),
        (3, 9) => Some(36),
        (4, 4) => Some(18),
        (4, 5) => Some(25),
        _ => None,
    }
}

/// True when every row and every column of `sq` is a permutation of `0..n`.
#[must_use]
pub fn is_latin_square(sq: &[Vec<usize>]) -> bool {
    let n = sq.len();
    if sq.iter().any(|r| r.len() != n) {
        return false;
    }
    for row in sq {
        if !is_permutation(row) {
            return false;
        }
    }
    for c in 0..n {
        let col: Vec<usize> = (0..n).map(|r| sq[r][c]).collect();
        if !is_permutation(&col) {
            return false;
        }
    }
    true
}

/// A random Latin square of order `n`.
///
/// Built from the cyclic square `(i + j) mod n` by applying an independent
/// random permutation to the rows, to the columns, and to the symbols. Each
/// of those three operations preserves the Latin property, so the result is
/// always valid. It samples the isotopy class of the cyclic square rather
/// than all Latin squares uniformly, which the caller should not assume
/// otherwise.
pub fn latin_square_random(n: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
    let rp = random_permutation(n, rng);
    let cp = random_permutation(n, rng);
    let sp = random_permutation(n, rng);
    (0..n)
        .map(|i| (0..n).map(|j| sp[(rp[i] + cp[j]) % n]).collect())
        .collect()
}

/// A magic square of order `n`, or `None` for `n = 2`, which has none.
///
/// Three constructions by residue: the Siamese method for odd `n`, the
/// complement pattern for `n` divisible by four, and Strachey's LUX method for
/// `n` congruent to 2 mod 4. Entries are `1..=n^2` and every row, column and
/// both diagonals sum to `n(n^2+1)/2`.
#[must_use]
pub fn magic_square(n: usize) -> Option<Vec<Vec<u64>>> {
    match n {
        0 => Some(Vec::new()),
        2 => None,
        _ if !n.is_multiple_of(2) => Some(magic_odd(n)),
        _ if n.is_multiple_of(4) => Some(magic_doubly_even(n)),
        _ => Some(magic_singly_even(n)),
    }
}

/// Siamese method: start at the top middle, step up-right, drop down on a
/// collision or a wrap.
fn magic_odd(n: usize) -> Vec<Vec<u64>> {
    let mut sq = vec![vec![0u64; n]; n];
    let (mut r, mut c) = (0usize, n / 2);
    for v in 1..=(n * n) as u64 {
        sq[r][c] = v;
        let nr = (r + n - 1) % n;
        let nc = (c + 1) % n;
        if sq[nr][nc] == 0 {
            r = nr;
            c = nc;
        } else {
            r = (r + 1) % n;
        }
    }
    sq
}

/// Doubly even: fill 1..n^2 in reading order, then complement the cells whose
/// row and column both lie in the same half of their 4-block.
fn magic_doubly_even(n: usize) -> Vec<Vec<u64>> {
    let mut sq = vec![vec![0u64; n]; n];
    let total = (n * n) as u64;
    for r in 0..n {
        for c in 0..n {
            let v = (r * n + c) as u64 + 1;
            let keep = (r % 4 == 0 || r % 4 == 3) == (c % 4 == 0 || c % 4 == 3);
            sq[r][c] = if keep { total + 1 - v } else { v };
        }
    }
    sq
}

/// Strachey's LUX method for n = 4m + 2: build the odd square of order
/// `2m + 1`, expand each cell to a 2x2 block offset by `4 (cell - 1)`, and
/// choose the block's internal pattern from the L/U/X rows, with the L and U
/// of the middle row swapped in the central column.
fn magic_singly_even(n: usize) -> Vec<Vec<u64>> {
    let m = (n - 2) / 4;
    let half = 2 * m + 1;
    let odd = magic_odd(half);
    // Row bands: m + 1 rows of L, one of U, then m - 1 of X.
    let mut kind = vec![b'L'; half];
    kind[m + 1] = b'U';
    for k in kind.iter_mut().skip(m + 2) {
        *k = b'X';
    }
    let mut sq = vec![vec![0u64; n]; n];
    for i in 0..half {
        for j in 0..half {
            let mut k = kind[i];
            // The one exception: swap L and U in the middle row's centre.
            if i == m && j == m {
                k = b'U';
            } else if i == m + 1 && j == m {
                k = b'L';
            }
            let base = 4 * (odd[i][j] - 1);
            // Offsets within the 2x2 block, reading (0,0) (0,1) (1,0) (1,1).
            let off: [u64; 4] = match k {
                b'L' => [4, 1, 2, 3],
                b'U' => [1, 4, 2, 3],
                _ => [1, 4, 3, 2],
            };
            sq[2 * i][2 * j] = base + off[0];
            sq[2 * i][2 * j + 1] = base + off[1];
            sq[2 * i + 1][2 * j] = base + off[2];
            sq[2 * i + 1][2 * j + 1] = base + off[3];
        }
    }
    sq
}

/// A de Bruijn sequence `B(k, n)`: a cyclic sequence of length `k^n` over the
/// alphabet `0..k` in which every `n`-tuple appears exactly once.
///
/// Built by the Frank-Kessler-Maiorana algorithm, which concatenates the
/// Lyndon words over the alphabet whose length divides `n`, in lexicographic
/// order.
///
/// # Panics
/// Panics if `k` is zero or `n` is zero.
#[must_use]
pub fn de_bruijn_sequence(k: usize, n: usize) -> Vec<usize> {
    assert!(k > 0 && n > 0, "k and n must be positive");
    let mut out = Vec::new();
    // a holds the current pre-necklace; index 0 is a sentinel.
    let mut a = vec![0usize; k * n + 1];
    fn db(t: usize, p: usize, k: usize, n: usize, a: &mut Vec<usize>, out: &mut Vec<usize>) {
        if t > n {
            // A necklace is emitted only when its period divides n.
            if n.is_multiple_of(p) {
                out.extend_from_slice(&a[1..=p]);
            }
        } else {
            a[t] = a[t - p];
            db(t + 1, p, k, n, a, out);
            for j in a[t - p] + 1..k {
                a[t] = j;
                db(t + 1, t, k, n, a, out);
            }
        }
    }
    db(1, 1, k, n, &mut a, &mut out);
    out
}

/// The number of perfect shuffles that restore a deck of `n_cards`.
///
/// An out-shuffle keeps the top and bottom cards fixed and permutes the rest
/// by doubling their position modulo `n_cards - 1`, so its order is the
/// multiplicative order of 2 there. An in-shuffle moves every card, doubling
/// position modulo `n_cards + 1`.
///
/// # Panics
/// Panics if `n_cards` is odd or below two: a perfect shuffle needs two equal
/// halves.
#[must_use]
pub fn perfect_shuffles_order(n_cards: u64, out: bool) -> u64 {
    assert!(
        n_cards >= 2 && n_cards.is_multiple_of(2),
        "a perfect shuffle needs an even deck of at least two cards"
    );
    let m = if out { n_cards - 1 } else { n_cards + 1 };
    if m == 1 {
        return 1;
    }
    multiplicative_order(2, m).expect("2 is a unit modulo an odd modulus")
}

/// The survivor of the Josephus problem: `n` people in a circle, every `k`-th
/// eliminated, returned as a zero-based position.
///
/// Recurrence `J(1) = 0`, `J(i) = (J(i-1) + k) mod i`: after the first
/// elimination the problem is the same one on `i - 1` people with the origin
/// shifted by `k`.
///
/// # Panics
/// Panics if `n` or `k` is zero.
#[must_use]
pub fn josephus(n: usize, k: usize) -> usize {
    assert!(n > 0 && k > 0, "n and k must be positive");
    let mut pos = 0usize;
    for i in 2..=n {
        pos = (pos + k) % i;
    }
    pos
}

/// The moves solving the Tower of Hanoi for `n` discs, as `(from, to)` pegs.
///
/// Exactly `2^n - 1` moves, the known minimum.
///
/// # Panics
/// Panics if `from` and `to` are equal or either is outside `0..3`.
#[must_use]
pub fn tower_of_hanoi_moves(n: u32, from: u8, to: u8) -> Vec<(u8, u8)> {
    assert!(from < 3 && to < 3, "pegs are numbered 0, 1, 2");
    assert!(from != to, "source and destination must differ");
    let mut out = Vec::new();
    fn go(n: u32, from: u8, to: u8, out: &mut Vec<(u8, u8)>) {
        if n == 0 {
            return;
        }
        let via = 3 - from - to;
        go(n - 1, from, via, out);
        out.push((from, to));
        go(n - 1, via, to, out);
    }
    go(n, from, to, &mut out);
    out
}

/// The twelvefold way: `n` balls into `k` boxes under the six combinations of
/// distinguishability and the three restrictions.
///
/// A restriction applies when its argument is `Some(true)`; `Some(false)` and
/// `None` both mean "no restriction", so `Some(false)` does not ask for a
/// map that fails to be injective.
#[must_use]
pub fn twelvefold_way(
    n: u64,
    k: u64,
    injective: Option<bool>,
    surjective: Option<bool>,
    distinguishable_balls: bool,
    distinguishable_boxes: bool,
) -> BigInt {
    let inj = injective == Some(true);
    let sur = surjective == Some(true);
    let one_if = |c: bool| if c { BigInt::one() } else { BigInt::zero() };

    match (distinguishable_balls, distinguishable_boxes, inj, sur) {
        // Bijections: only possible when n == k.
        (true, true, true, true) => {
            if n == k {
                BigInt::factorial(n)
            } else {
                BigInt::zero()
            }
        }
        (_, _, true, true) => one_if(n == k),

        // Distinguishable balls, distinguishable boxes: arbitrary functions,
        // injections, surjections.
        (true, true, false, false) => BigInt::from_u64(k).pow(n),
        (true, true, true, false) => {
            if n > k {
                BigInt::zero()
            } else {
                // Falling factorial k (k-1) ... (k-n+1).
                (0..n).fold(BigInt::one(), |a, i| a.mul(&BigInt::from_u64(k - i)))
            }
        }
        (true, true, false, true) => BigInt::factorial(k).mul(&stirling_second(n, k)),

        // Indistinguishable balls, distinguishable boxes: multisets.
        (false, true, false, false) => {
            if k == 0 {
                one_if(n == 0)
            } else {
                BigInt::binomial(n + k - 1, n)
            }
        }
        (false, true, true, false) => BigInt::binomial(k, n),
        (false, true, false, true) => {
            if n < k {
                BigInt::zero()
            } else if k == 0 {
                one_if(n == 0)
            } else {
                BigInt::binomial(n - 1, n - k)
            }
        }

        // Distinguishable balls, indistinguishable boxes: set partitions into
        // at most k, exactly k, or (injective) one ball per box.
        (true, false, false, false) => {
            (0..=k).fold(BigInt::zero(), |a, j| a.add(&stirling_second(n, j)))
        }
        (true, false, true, false) => one_if(n <= k),
        (true, false, false, true) => stirling_second(n, k),

        // Indistinguishable both: integer partitions.
        (false, false, false, false) => partition_count_into_at_most_k(n, k),
        (false, false, true, false) => one_if(n <= k),
        (false, false, false, true) => partitions_into_k(n, k),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn big(n: u64) -> BigInt {
        BigInt::from_u64(n)
    }

    /// Pascal's rule and the row sum, which together pin every binomial.
    #[test]
    fn binomials_satisfy_pascal_and_row_sums() {
        for n in 1..=30u64 {
            let mut row_sum = 0u64;
            for k in 0..=n {
                let c = binomial_u64(n, k).unwrap();
                let left = if k == 0 {
                    0
                } else {
                    binomial_u64(n - 1, k - 1).unwrap()
                };
                let right = binomial_u64(n - 1, k).unwrap();
                assert_eq!(c, left + right, "Pascal fails at C({n}, {k})");
                row_sum += c;
                // Symmetry.
                assert_eq!(c, binomial_u64(n, n - k).unwrap());
                // Agreement with the BigInt implementation.
                assert_eq!(BigInt::binomial(n, k), big(c));
            }
            assert_eq!(row_sum, 1u64 << n, "row {n} does not sum to 2^n");
        }
        assert_eq!(binomial_u64(5, 9), Some(0));
    }

    /// `None` must mean "does not fit in u64" and nothing else. The exact
    /// value from BigInt decides, so this catches both a premature overflow
    /// report and a wrapped value returned as if it were exact.
    #[test]
    fn binomial_overflow_is_reported_exactly_at_the_boundary() {
        let max = BigInt::from_str_radix(&u64::MAX.to_string(), 10).unwrap();
        let mut saw_fit = false;
        let mut saw_overflow = false;
        for n in 0..=80u64 {
            for k in 0..=n {
                let exact = BigInt::binomial(n, k);
                let fits = exact <= max;
                match binomial_u64(n, k) {
                    Some(v) => {
                        assert!(fits, "C({n}, {k}) does not fit but was returned");
                        assert_eq!(big(v), exact, "C({n}, {k}) is wrong");
                        saw_fit = true;
                    }
                    None => {
                        assert!(!fits, "C({n}, {k}) fits but overflow was reported");
                        saw_overflow = true;
                    }
                }
            }
        }
        // Both outcomes occur in the range, so neither branch is vacuous.
        assert!(saw_fit && saw_overflow);

        // The intermediate is up to k times the answer, which is the case a
        // u64 accumulator gets wrong. C(62, 31) needs 63 bits, and computing
        // it multiplies through a value above 2^68.
        assert_eq!(binomial_u64(62, 31), Some(465_428_353_255_261_088));
        assert_eq!(big(binomial_u64(62, 31).unwrap()), BigInt::binomial(62, 31));
    }

    /// Lucas's theorem against direct computation of the binomial mod p.
    #[test]
    fn lucas_matches_direct_reduction() {
        for &p in &[2u64, 3, 5, 7, 13, 101] {
            for n in 0..60u64 {
                for k in 0..=n {
                    let direct = BigInt::binomial(n, k)
                        .div_rem(&big(p))
                        .1
                        .to_i64()
                        .unwrap() as u64;
                    assert_eq!(
                        binomial_mod_p(n, k, p),
                        direct,
                        "Lucas disagrees at C({n}, {k}) mod {p}"
                    );
                }
            }
        }
        // A case far beyond direct computation: Kummer's theorem says
        // C(n, k) is odd exactly when k's binary digits are a submask of n's.
        for n in 0..256u64 {
            for k in 0..=n {
                assert_eq!(binomial_mod_p(n, k, 2), u64::from(n & k == k));
            }
        }
    }

    /// The multinomial counts the distinct arrangements of a multiset, which
    /// is checkable by enumeration for small cases.
    #[test]
    fn multinomial_counts_multiset_arrangements() {
        for ks in [
            vec![1u64, 1, 1],
            vec![2, 1],
            vec![2, 2],
            vec![3, 1, 1],
            vec![2, 2, 1],
        ] {
            // Build the multiset and count distinct orderings by brute force.
            let mut items = Vec::new();
            for (sym, &count) in ks.iter().enumerate() {
                for _ in 0..count {
                    items.push(sym);
                }
            }
            let distinct: HashSet<Vec<usize>> = permutations_iter(&items).collect();
            assert_eq!(
                multinomial(&ks),
                big(distinct.len() as u64),
                "multinomial disagrees for {ks:?}"
            );
        }
        // And the identity multinomial(k, n-k) == C(n, k).
        for n in 0..=20u64 {
            for k in 0..=n {
                assert_eq!(multinomial(&[k, n - k]), BigInt::binomial(n, k));
            }
        }
    }

    #[test]
    fn falling_factorial_counts_injections() {
        for n in 0..=8u64 {
            for k in 0..=n {
                // Injections from k labelled balls into n boxes.
                let by_formula = permutations_count(n, k).unwrap();
                let by_twelvefold =
                    twelvefold_way(k, n, Some(true), None, true, true);
                assert_eq!(big(by_formula), by_twelvefold);
            }
        }
        assert_eq!(permutations_count(21, 21), None);
        assert_eq!(permutations_count(20, 20), Some(2_432_902_008_176_640_000));
    }

    // -----------------------------------------------------------------------
    // Enumeration
    // -----------------------------------------------------------------------

    /// Heap's algorithm must produce every permutation exactly once, and
    /// consecutive outputs must differ by exactly one transposition -- that
    /// second property is what distinguishes Heap's from any other generator.
    #[test]
    fn heap_permutations_are_complete_and_adjacent_by_one_swap() {
        for n in 0..=6usize {
            let items: Vec<usize> = (0..n).collect();
            let all: Vec<Vec<usize>> = permutations_iter(&items).collect();
            assert_eq!(all.len() as u64, BigInt::factorial(n as u64).to_i64().unwrap() as u64);
            let distinct: HashSet<&Vec<usize>> = all.iter().collect();
            assert_eq!(distinct.len(), all.len(), "n = {n} has duplicates");
            for w in all.windows(2) {
                let diffs = (0..n).filter(|&i| w[0][i] != w[1][i]).count();
                assert_eq!(diffs, 2, "consecutive outputs differ in {diffs} places");
            }
        }
        // 0! = 1: the empty permutation, not nothing at all.
        assert_eq!(permutations_iter(&[]).collect::<Vec<_>>(), vec![Vec::new()]);
    }

    /// The lexicographic successor must walk the sorted order exactly, so
    /// repeatedly applying it from the identity enumerates n! permutations in
    /// increasing order and then stops.
    #[test]
    fn lex_successor_walks_sorted_order() {
        for n in 1..=6usize {
            let mut p: Vec<usize> = (0..n).collect();
            let mut seen = vec![p.clone()];
            while permutations_lex_next(&mut p) {
                assert!(*seen.last().unwrap() < p, "order is not increasing");
                seen.push(p.clone());
            }
            assert_eq!(seen.len() as u64, BigInt::factorial(n as u64).to_i64().unwrap() as u64);
            // The final state is the descending arrangement and is unchanged
            // by the failed call.
            assert_eq!(p, (0..n).rev().collect::<Vec<_>>());
            // The enumeration is exactly the sorted set of all permutations.
            let mut sorted = seen.clone();
            sorted.sort();
            assert_eq!(seen, sorted);
        }
    }

    /// nth_permutation and permutation_index are mutually inverse, and agree
    /// with the lexicographic walk.
    #[test]
    fn factoradic_indexing_inverts_the_lex_order() {
        for n in 0..=6usize {
            let total = BigInt::factorial(n as u64);
            let mut walk: Vec<usize> = (0..n).collect();
            let mut i = 0u64;
            loop {
                let idx = big(i);
                let p = nth_permutation(n, &idx);
                assert_eq!(p, walk, "nth_permutation disagrees with the walk");
                assert_eq!(permutation_index(&p), idx, "index is not the inverse");
                i += 1;
                if big(i) >= total || !permutations_lex_next(&mut walk) {
                    break;
                }
            }
            assert_eq!(big(i), total);
        }
        // A case well past what enumeration could reach: index 10! - 1 must be
        // the descending permutation.
        let last = BigInt::factorial(10).sub(&BigInt::one());
        assert_eq!(nth_permutation(10, &last), (0..10).rev().collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "below n_items!")]
    fn nth_permutation_rejects_an_out_of_range_index() {
        let _ = nth_permutation(4, &big(24));
    }

    /// Combinations: complete, distinct, sorted within and between, and the
    /// right number of them.
    #[test]
    fn combinations_are_complete_and_lexicographic() {
        for n in 0..=7usize {
            for k in 0..=n {
                let all: Vec<Vec<usize>> = combinations_iter(n, k).collect();
                assert_eq!(all.len() as u64, binomial_u64(n as u64, k as u64).unwrap());
                let distinct: HashSet<&Vec<usize>> = all.iter().collect();
                assert_eq!(distinct.len(), all.len());
                for c in &all {
                    assert_eq!(c.len(), k);
                    assert!(c.windows(2).all(|w| w[0] < w[1]), "not ascending: {c:?}");
                    assert!(c.iter().all(|&x| x < n));
                }
                assert!(all.windows(2).all(|w| w[0] < w[1]), "not lexicographic");
            }
        }
        // k > n yields nothing.
        assert_eq!(combinations_iter(3, 4).count(), 0);
    }

    /// Multisets: count must be C(n + k - 1, k), the stars-and-bars value.
    #[test]
    fn multisets_match_stars_and_bars() {
        for n in 0..=6usize {
            for k in 0..=6usize {
                let all: Vec<Vec<usize>> = combinations_with_replacement_iter(n, k).collect();
                let expected = if k == 0 {
                    1
                } else if n == 0 {
                    0
                } else {
                    binomial_u64((n + k - 1) as u64, k as u64).unwrap()
                };
                assert_eq!(all.len() as u64, expected, "n = {n}, k = {k}");
                let distinct: HashSet<&Vec<usize>> = all.iter().collect();
                assert_eq!(distinct.len(), all.len());
                for c in &all {
                    assert!(c.windows(2).all(|w| w[0] <= w[1]), "not sorted: {c:?}");
                }
                assert!(all.windows(2).all(|w| w[0] < w[1]), "not lexicographic");
            }
        }
    }

    /// The defining property of a Gray code: consecutive values differ in
    /// exactly one bit, and the whole cycle covers every value once.
    #[test]
    fn gray_code_changes_one_bit_at_a_time() {
        for bits in 1..=12u32 {
            let all: Vec<u64> = gray_code_iter(bits).collect();
            assert_eq!(all.len(), 1usize << bits);
            let distinct: HashSet<&u64> = all.iter().collect();
            assert_eq!(distinct.len(), all.len(), "not a permutation of 0..2^n");
            for w in all.windows(2) {
                assert_eq!((w[0] ^ w[1]).count_ones(), 1);
            }
            // Cyclic: the wrap-around step is also a single bit.
            assert_eq!((all[0] ^ all[all.len() - 1]).count_ones(), 1);
        }
    }

    #[test]
    fn subsets_enumerate_the_power_set() {
        for n in 0..=10u32 {
            let all: Vec<u64> = subsets_iter(n).collect();
            assert_eq!(all.len(), 1usize << n);
            // Every popcount class has C(n, k) members.
            for k in 0..=n {
                let count = all.iter().filter(|&&m| m.count_ones() == k).count();
                assert_eq!(count as u64, binomial_u64(n as u64, k as u64).unwrap());
            }
        }
    }

    /// Dyck paths: the count is Catalan, every prefix is balanced, and the
    /// peak distribution is the Narayana triangle.
    #[test]
    fn dyck_paths_are_catalan_and_narayana_by_peaks() {
        for n in 0..=8usize {
            let all: Vec<Vec<bool>> = dyck_paths_iter(n).collect();
            assert_eq!(big(all.len() as u64), catalan(n as u64), "n = {n}");
            let distinct: HashSet<&Vec<bool>> = all.iter().collect();
            assert_eq!(distinct.len(), all.len());
            for path in &all {
                assert_eq!(path.len(), 2 * n);
                let mut height = 0i64;
                for &up in path {
                    height += if up { 1 } else { -1 };
                    assert!(height >= 0, "path dips below the axis");
                }
                assert_eq!(height, 0, "path does not return to the axis");
            }
            // Peaks are the "up then down" positions.
            for k in 1..=n {
                let with_k = all
                    .iter()
                    .filter(|p| p.windows(2).filter(|w| w[0] && !w[1]).count() == k)
                    .count();
                assert_eq!(
                    big(with_k as u64),
                    narayana(n as u64, k as u64),
                    "Narayana disagrees at n = {n}, k = {k}"
                );
            }
        }
    }

    /// Restricted growth strings are in bijection with set partitions, so the
    /// count is Bell and each string satisfies the growth restriction.
    #[test]
    fn set_partitions_are_bell_many_and_restricted() {
        for n in 0..=8usize {
            let all: Vec<Vec<usize>> = set_partitions_iter(n).collect();
            assert_eq!(big(all.len() as u64), bell_number(n as u64), "n = {n}");
            let distinct: HashSet<&Vec<usize>> = all.iter().collect();
            assert_eq!(distinct.len(), all.len(), "duplicates at n = {n}");
            for s in &all {
                assert_eq!(s.len(), n);
                let mut running_max = 0usize;
                for (i, &b) in s.iter().enumerate() {
                    if i == 0 {
                        assert_eq!(b, 0);
                    }
                    assert!(b <= running_max, "growth restriction violated: {s:?}");
                    running_max = running_max.max(b + 1);
                }
            }
            // Blocks-per-partition distribution must be Stirling second kind.
            for k in 0..=n {
                let with_k = all
                    .iter()
                    .filter(|s| s.iter().copied().max().map_or(0, |m| m + 1) == k)
                    .count();
                assert_eq!(
                    big(with_k as u64),
                    stirling_second(n as u64, k as u64),
                    "S({n}, {k}) disagrees with enumeration"
                );
            }
        }
    }

    /// Compositions: 2^(n-1) of them, all parts positive, all summing to n,
    /// and the count with exactly k parts is C(n-1, k-1).
    #[test]
    fn compositions_are_complete_and_binomial_by_length() {
        for n in 0..=10u64 {
            let all: Vec<Vec<u64>> = compositions_iter(n).collect();
            let expected = if n == 0 { 1 } else { 1u64 << (n - 1) };
            assert_eq!(all.len() as u64, expected, "n = {n}");
            let distinct: HashSet<&Vec<u64>> = all.iter().collect();
            assert_eq!(distinct.len(), all.len());
            for c in &all {
                assert_eq!(c.iter().sum::<u64>(), n);
                assert!(c.iter().all(|&x| x > 0));
            }
            for k in 1..=n {
                let with_k = all.iter().filter(|c| c.len() as u64 == k).count();
                assert_eq!(with_k as u64, binomial_u64(n - 1, k - 1).unwrap());
            }
        }
    }

    // -----------------------------------------------------------------------
    // The permutation group
    // -----------------------------------------------------------------------

    /// Group axioms on S_5, checked exhaustively: associativity, identity,
    /// inverses, and closure.
    #[test]
    fn permutations_form_a_group_under_composition() {
        let items: Vec<usize> = (0..4).collect();
        let all: Vec<Vec<usize>> = permutations_iter(&items).collect();
        let id: Vec<usize> = (0..4).collect();
        for a in &all {
            assert_eq!(permutation_compose(a, &id), *a);
            assert_eq!(permutation_compose(&id, a), *a);
            let inv = permutation_inverse(a);
            assert_eq!(permutation_compose(a, &inv), id);
            assert_eq!(permutation_compose(&inv, a), id);
            for b in &all {
                let ab = permutation_compose(a, b);
                assert!(is_permutation(&ab), "not closed");
                for c in &all {
                    assert_eq!(
                        permutation_compose(&permutation_compose(a, b), c),
                        permutation_compose(a, &permutation_compose(b, c)),
                        "associativity fails"
                    );
                }
                // The sign is a homomorphism to {+1, -1}.
                assert_eq!(
                    permutation_sign(&ab),
                    permutation_sign(a) * permutation_sign(b)
                );
            }
        }
    }

    /// The order is the least k with p^k = identity -- verified by actually
    /// composing p with itself that many times.
    #[test]
    fn order_is_the_least_power_giving_the_identity() {
        let items: Vec<usize> = (0..6).collect();
        let id: Vec<usize> = (0..6).collect();
        for p in permutations_iter(&items) {
            let order = permutation_order(&p).to_i64().unwrap() as usize;
            let mut acc = id.clone();
            for step in 1..=order {
                acc = permutation_compose(&acc, &p);
                if step < order {
                    assert_ne!(acc, id, "p^{step} is already the identity");
                }
            }
            assert_eq!(acc, id, "p^order is not the identity");
        }
        // Landau's function: the largest order in S_n. g(6) = 6, g(7) = 12.
        let max6 = permutations_iter(&(0..6).collect::<Vec<_>>())
            .map(|p| permutation_order(&p).to_i64().unwrap())
            .max()
            .unwrap();
        assert_eq!(max6, 6);
        let max7 = permutations_iter(&(0..7).collect::<Vec<_>>())
            .map(|p| permutation_order(&p).to_i64().unwrap())
            .max()
            .unwrap();
        assert_eq!(max7, 12);
    }

    /// The cycle type is the conjugacy invariant: two permutations share a
    /// cycle type exactly when some g conjugates one into the other.
    #[test]
    fn cycle_type_is_exactly_the_conjugacy_invariant() {
        let items: Vec<usize> = (0..4).collect();
        let all: Vec<Vec<usize>> = permutations_iter(&items).collect();
        for a in &all {
            assert_eq!(permutation_cycle_type(a).iter().sum::<usize>(), 4);
            for b in &all {
                let conjugate_exists = all.iter().any(|g| {
                    let gi = permutation_inverse(g);
                    permutation_compose(&permutation_compose(g, a), &gi) == *b
                });
                assert_eq!(
                    permutation_cycle_type(a) == permutation_cycle_type(b),
                    conjugate_exists,
                    "cycle type does not match conjugacy for {a:?} and {b:?}"
                );
            }
        }
        // The number of permutations with k cycles is the unsigned Stirling
        // number of the first kind.
        for n in 1..=6u64 {
            let items: Vec<usize> = (0..n as usize).collect();
            for k in 1..=n {
                let count = permutations_iter(&items)
                    .filter(|p| permutation_to_cycles(p).len() as u64 == k)
                    .count();
                assert_eq!(big(count as u64), stirling_first(n, k), "c({n}, {k})");
            }
        }
    }

    /// to_cycles and from_cycles are mutually inverse.
    #[test]
    fn cycle_notation_round_trips() {
        for n in 0..=6usize {
            let items: Vec<usize> = (0..n).collect();
            for p in permutations_iter(&items) {
                let cycles = permutation_to_cycles(&p);
                assert_eq!(permutation_from_cycles(n, &cycles), p);
                // Each cycle starts at its own smallest element.
                for c in &cycles {
                    assert_eq!(c[0], *c.iter().min().unwrap());
                }
                // Cycles are ordered by that element.
                assert!(cycles.windows(2).all(|w| w[0][0] < w[1][0]));
            }
        }
        // Fixed points may be omitted from the input.
        assert_eq!(
            permutation_from_cycles(5, &[vec![1, 3]]),
            vec![0, 3, 2, 1, 4]
        );
    }

    /// The permutation matrix is orthogonal, its determinant is the sign, and
    /// the matrix of a composition is the product of the matrices.
    #[test]
    fn permutation_matrix_is_a_faithful_representation() {
        let items: Vec<usize> = (0..4).collect();
        let all: Vec<Vec<usize>> = permutations_iter(&items).collect();
        for a in &all {
            let ma = permutation_matrix(a);
            // Orthogonal: M^T M = I.
            let mt = ma.transpose();
            let prod = mt.mul(&ma).unwrap();
            for r in 0..4 {
                for c in 0..4 {
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!((prod.get(r, c) - want).abs() < 1e-12);
                }
            }
            let det = crate::linalg::lu::lu_decompose(&ma).unwrap().determinant();
            assert!((det - f64::from(permutation_sign(a))).abs() < 1e-12);
            for b in &all {
                let mab = permutation_matrix(&permutation_compose(a, b));
                let mprod = permutation_matrix(a).mul(&permutation_matrix(b)).unwrap();
                for r in 0..4 {
                    for c in 0..4 {
                        assert!(
                            (mab.get(r, c) - mprod.get(r, c)).abs() < 1e-12,
                            "matrix homomorphism fails"
                        );
                    }
                }
            }
        }
    }

    /// Derangement count against exhaustive enumeration, plus the n!/e
    /// asymptotic, plus the rejection sampler's output validity.
    #[test]
    fn derangements_count_matches_enumeration() {
        for n in 0..=8usize {
            let items: Vec<usize> = (0..n).collect();
            let brute = if n == 0 {
                1
            } else {
                permutations_iter(&items).filter(|p| is_derangement(p)).count()
            };
            assert_eq!(derangements_count(n as u64), big(brute as u64), "n = {n}");
        }
        // D(n) is the nearest integer to n!/e for n >= 1.
        for n in 1..=15u64 {
            let approx = BigInt::factorial(n).to_f64() / std::f64::consts::E;
            let exact = derangements_count(n).to_f64();
            assert!((exact - approx).abs() <= 0.5, "n = {n}");
        }
        let mut rng = Rng::new(2024);
        for n in [2usize, 3, 5, 9] {
            for _ in 0..50 {
                let d = random_derangement(n, &mut rng);
                assert!(is_derangement(&d), "sampler produced {d:?}");
            }
        }
    }

    /// Fisher-Yates must produce permutations, and over many draws every
    /// symbol must land in every position -- a shuffle that never moves a
    /// symbol past some point would still pass a validity-only check.
    #[test]
    fn fisher_yates_reaches_every_position() {
        let mut rng = Rng::new(11);
        const N: usize = 6;
        let mut hits = [[0u32; N]; N];
        for _ in 0..20_000 {
            let p = random_permutation(N, &mut rng);
            assert!(is_permutation(&p));
            for (i, &x) in p.iter().enumerate() {
                hits[i][x] += 1;
            }
        }
        // Uniform would put 20000/6 = 3333 in each cell; allow a wide band and
        // still catch any structural bias.
        for row in &hits {
            for &c in row {
                assert!((2500..4200).contains(&c), "cell count {c} is far from uniform");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Counting numbers
    // -----------------------------------------------------------------------

    /// The Stirling numbers must satisfy the identity connecting them to
    /// ordinary powers, and the first kind must expand the rising factorial.
    #[test]
    fn stirling_numbers_satisfy_their_defining_identities() {
        // x^n = sum_k S(n, k) * falling(x, k), tested at integer x.
        for n in 0..=8u64 {
            for x in 0..=10u64 {
                let lhs = big(x).pow(n);
                let mut rhs = BigInt::zero();
                for k in 0..=n {
                    let falling = (0..k).fold(BigInt::one(), |a, i| {
                        if x >= i {
                            a.mul(&big(x - i))
                        } else {
                            BigInt::zero()
                        }
                    });
                    rhs = rhs.add(&stirling_second(n, k).mul(&falling));
                }
                assert_eq!(lhs, rhs, "n = {n}, x = {x}");
            }
        }
        // rising(x, n) = sum_k c(n, k) x^k for the unsigned first kind.
        for n in 0..=8u64 {
            for x in 1..=8u64 {
                let rising = (0..n).fold(BigInt::one(), |a, i| a.mul(&big(x + i)));
                let mut rhs = BigInt::zero();
                for k in 0..=n {
                    rhs = rhs.add(&stirling_first(n, k).mul(&big(x).pow(k)));
                }
                assert_eq!(rising, rhs, "n = {n}, x = {x}");
            }
        }
        // Row sum of the first kind is n!.
        for n in 0..=12u64 {
            let sum = (0..=n).fold(BigInt::zero(), |a, k| a.add(&stirling_first(n, k)));
            assert_eq!(sum, BigInt::factorial(n));
        }
    }

    /// Bell numbers by three independent routes: the triangle, the Stirling
    /// row sum, and the Bell recurrence with binomial weights.
    #[test]
    fn bell_numbers_agree_across_three_derivations() {
        let mut prior: Vec<BigInt> = Vec::new();
        for n in 0..=25u64 {
            let b = bell_number(n);
            let by_stirling = (0..=n).fold(BigInt::zero(), |a, k| a.add(&stirling_second(n, k)));
            assert_eq!(b, by_stirling, "Stirling row sum disagrees at n = {n}");
            if n > 0 {
                // B(n) = sum_k C(n-1, k) B(k).
                let by_recurrence = (0..n).fold(BigInt::zero(), |a, k| {
                    a.add(&BigInt::binomial(n - 1, k).mul(&prior[k as usize]))
                });
                assert_eq!(b, by_recurrence, "recurrence disagrees at n = {n}");
            }
            prior.push(b);
        }
        // Published values, including one past u64.
        assert_eq!(bell_number(10), big(115_975));
        assert_eq!(
            bell_number(25).to_string(),
            "4638590332229999353"
        );
        assert_eq!(
            bell_number(30).to_string(),
            "846749014511809332450147"
        );
    }

    /// Catalan numbers by formula, by the Segner recurrence, and modulo a
    /// composite where the closed form has no modular meaning.
    #[test]
    fn catalan_numbers_agree_with_the_segner_recurrence() {
        let mut c: Vec<BigInt> = vec![BigInt::one()];
        for n in 1..=40u64 {
            let by_recurrence = (0..n).fold(BigInt::zero(), |a, i| {
                a.add(&c[i as usize].mul(&c[(n - 1 - i) as usize]))
            });
            assert_eq!(catalan(n), by_recurrence, "n = {n}");
            c.push(catalan(n));
        }
        assert_eq!(catalan(10), big(16_796));
        // The modular version must agree with reducing the exact value, for
        // moduli sharing factors with n + 1 as well as coprime ones.
        for &m in &[2u64, 6, 10, 12, 1_000_000_007] {
            for n in 0..=40u64 {
                let exact = catalan(n).div_rem(&big(m)).1.to_i64().unwrap() as u64;
                assert_eq!(catalan_mod(n, m), exact, "C({n}) mod {m}");
            }
        }
    }

    /// Eulerian numbers count ascents, verified by enumerating permutations.
    #[test]
    fn eulerian_numbers_count_ascents() {
        for n in 1..=7u64 {
            let items: Vec<usize> = (0..n as usize).collect();
            let mut by_ascents = vec![0u64; n as usize];
            for p in permutations_iter(&items) {
                let ascents = p.windows(2).filter(|w| w[0] < w[1]).count();
                by_ascents[ascents] += 1;
            }
            for k in 0..n {
                assert_eq!(
                    eulerian_number(n, k),
                    big(by_ascents[k as usize]),
                    "A({n}, {k})"
                );
            }
            // Row sum is n!, and the row is a palindrome.
            let sum = (0..n).fold(BigInt::zero(), |a, k| a.add(&eulerian_number(n, k)));
            assert_eq!(sum, BigInt::factorial(n));
            for k in 0..n {
                assert_eq!(eulerian_number(n, k), eulerian_number(n, n - 1 - k));
            }
        }
    }

    /// The lattice-path numbers, each checked against a direct path count on
    /// a grid rather than against a table of values.
    #[test]
    fn lattice_path_numbers_match_direct_path_counts() {
        // Motzkin: paths with up, down, level steps staying at or above zero.
        for n in 0..=10usize {
            // dp[h] = number of ways to be at height h after i steps.
            let mut dp = vec![BigInt::zero(); n + 2];
            dp[0] = BigInt::one();
            for _ in 0..n {
                let mut next = vec![BigInt::zero(); n + 2];
                for h in 0..=n {
                    if dp[h].is_zero() {
                        continue;
                    }
                    next[h] = next[h].add(&dp[h]); // level
                    next[h + 1] = next[h + 1].add(&dp[h]); // up
                    if h > 0 {
                        next[h - 1] = next[h - 1].add(&dp[h]); // down
                    }
                }
                dp = next;
            }
            assert_eq!(motzkin(n as u64), dp[0], "motzkin({n})");
        }

        // Delannoy: paths with east, north, diagonal steps, counted by a
        // straightforward grid fill (the implementation rolls one row, so this
        // is an independent layout).
        for m in 0..=6usize {
            for n in 0..=6usize {
                let mut grid = vec![vec![BigInt::zero(); n + 1]; m + 1];
                for (i, row) in grid.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell = if i == 0 || j == 0 {
                            BigInt::one()
                        } else {
                            BigInt::zero()
                        };
                    }
                }
                for i in 1..=m {
                    for j in 1..=n {
                        grid[i][j] = grid[i - 1][j]
                            .add(&grid[i][j - 1])
                            .add(&grid[i - 1][j - 1]);
                    }
                }
                assert_eq!(delannoy(m as u64, n as u64), grid[m][n], "D({m}, {n})");
            }
        }
        // The central Delannoy numbers are a published sequence.
        assert_eq!(delannoy(3, 3), big(63));
        assert_eq!(delannoy(6, 6), big(8_989));

        // Schroeder: the large Schroeder numbers start 1, 2, 6, 22, 90, 394.
        let expected = [1u64, 2, 6, 22, 90, 394, 1_806, 8_558, 41_586];
        for (n, &want) in expected.iter().enumerate() {
            assert_eq!(schroeder(n as u64), big(want), "schroeder({n})");
        }
        // And S(n) = D(n, n) - D(n+1, n-1) is a known identity.
        for n in 1..=6u64 {
            assert_eq!(
                schroeder(n),
                delannoy(n, n).sub(&delannoy(n + 1, n - 1)),
                "Schroeder-Delannoy identity at n = {n}"
            );
        }
    }

    /// Lah numbers count ordered set partitions into lists, which is checkable
    /// by their connection identity to the two kinds of Stirling number.
    #[test]
    fn lah_numbers_connect_the_two_stirling_kinds() {
        // L(n, k) = sum_j c(n, j) S(j, k) for the unsigned first kind.
        for n in 0..=8u64 {
            for k in 0..=n {
                let rhs = (0..=n).fold(BigInt::zero(), |a, j| {
                    a.add(&stirling_first(n, j).mul(&stirling_second(j, k)))
                });
                assert_eq!(lah_number(n, k), rhs, "L({n}, {k})");
            }
            // Row sum with k >= 1 gives the number of "sets of lists".
            let sum = (0..=n).fold(BigInt::zero(), |a, k| a.add(&lah_number(n, k)));
            assert!(!sum.is_zero());
        }
        assert_eq!(lah_number(4, 2), big(36));
    }

    /// The ballot problem, checked by enumerating every vote sequence.
    #[test]
    fn ballot_numbers_count_never_behind_sequences() {
        for p in 0..=8u64 {
            for q in 0..=p {
                // A sequence is a bit string with p ones (A) and q zeros (B).
                let n = (p + q) as usize;
                let mut good = 0u64;
                for mask in subsets_iter(n as u32) {
                    if mask.count_ones() as u64 != p {
                        continue;
                    }
                    let mut lead = 0i64;
                    let mut ok = true;
                    for i in 0..n {
                        lead += if mask >> i & 1 == 1 { 1 } else { -1 };
                        if lead < 0 {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        good += 1;
                    }
                }
                assert_eq!(ballot_number(p, q), big(good), "ballot({p}, {q})");
            }
        }
        // Ballot(n, n) is the n-th Catalan number.
        for n in 0..=10u64 {
            assert_eq!(ballot_number(n, n), catalan(n));
        }
    }

    // -----------------------------------------------------------------------
    // Burnside, Polya, inclusion-exclusion
    // -----------------------------------------------------------------------

    /// Necklaces and bracelets against brute-force orbit counting under the
    /// cyclic and dihedral group actions.
    #[test]
    fn necklace_and_bracelet_counts_match_brute_force_orbits() {
        for n in 1..=8usize {
            for k in 1..=4usize {
                let total = k.pow(n as u32);
                // Canonical form under rotation.
                let mut rot_orbits: HashSet<Vec<usize>> = HashSet::new();
                let mut dih_orbits: HashSet<Vec<usize>> = HashSet::new();
                for code in 0..total {
                    let mut beads = Vec::with_capacity(n);
                    let mut c = code;
                    for _ in 0..n {
                        beads.push(c % k);
                        c /= k;
                    }
                    let rotations: Vec<Vec<usize>> = (0..n)
                        .map(|s| (0..n).map(|i| beads[(i + s) % n]).collect())
                        .collect();
                    let mut reflections: Vec<Vec<usize>> = rotations
                        .iter()
                        .map(|r| r.iter().rev().copied().collect())
                        .collect();
                    rot_orbits.insert(rotations.iter().min().unwrap().clone());
                    reflections.extend(rotations.iter().cloned());
                    dih_orbits.insert(reflections.iter().min().unwrap().clone());
                }
                assert_eq!(
                    necklaces_count(n as u64, k as u64),
                    big(rot_orbits.len() as u64),
                    "necklaces({n}, {k})"
                );
                assert_eq!(
                    bracelets_count(n as u64, k as u64),
                    big(dih_orbits.len() as u64),
                    "bracelets({n}, {k})"
                );
            }
        }
        // Bracelets never exceed necklaces, and match when reflection adds
        // nothing new (n <= 2).
        for k in 1..=5u64 {
            assert_eq!(bracelets_count(1, k), necklaces_count(1, k));
            assert_eq!(bracelets_count(2, k), necklaces_count(2, k));
            for n in 3..=8u64 {
                assert!(bracelets_count(n, k) <= necklaces_count(n, k));
            }
        }
    }

    /// Burnside's lemma applied to an explicit group action, cross-checked by
    /// counting the orbits directly.
    #[test]
    fn burnside_averages_fixed_points_to_orbits() {
        // The rotation group of a 6-bead cycle acting on 3-colourings.
        let (n, k) = (6usize, 3usize);
        let fixed: Vec<BigInt> = (0..n)
            .map(|s| {
                let mut count = 0u64;
                for code in 0..k.pow(n as u32) {
                    let mut beads = Vec::with_capacity(n);
                    let mut c = code;
                    for _ in 0..n {
                        beads.push(c % k);
                        c /= k;
                    }
                    if (0..n).all(|i| beads[i] == beads[(i + s) % n]) {
                        count += 1;
                    }
                }
                big(count)
            })
            .collect();
        assert_eq!(
            burnside_orbit_count(&fixed),
            necklaces_count(n as u64, k as u64)
        );
    }

    /// The cycle indices, evaluated at a colour count, must reproduce the
    /// combinatorial counts they encode.
    #[test]
    fn cycle_indices_reproduce_their_orbit_counts() {
        for n in 1..=8u64 {
            let cyc = cycle_index_cyclic(n);
            let dih = cycle_index_dihedral(n);
            let sym = cycle_index_symmetric(n);
            for k in 1..=6u64 {
                assert_eq!(
                    polya_enumeration(&cyc, k),
                    necklaces_count(n, k),
                    "cyclic index at n = {n}, k = {k}"
                );
                assert_eq!(
                    polya_enumeration(&dih, k),
                    bracelets_count(n, k),
                    "dihedral index at n = {n}, k = {k}"
                );
                // S_n orbits of colourings are multisets of size n from k.
                assert_eq!(
                    polya_enumeration(&sym, k),
                    BigInt::binomial(k + n - 1, n),
                    "symmetric index at n = {n}, k = {k}"
                );
            }
            // The value at one colour is always one, for any group.
            for ci in [&cyc, &dih, &sym] {
                assert_eq!(polya_enumeration(ci, 1), BigInt::one());
            }
        }
    }

    /// Inclusion-exclusion applied to divisibility classes must reproduce
    /// Euler's totient, and applied to arbitrary explicit sets must reproduce
    /// the union size counted directly.
    #[test]
    fn inclusion_exclusion_recovers_totient_and_explicit_unions() {
        // Numbers in 1..=n divisible by at least one of the distinct primes.
        for &(n, ref primes) in &[
            (30u64, vec![2u64, 3, 5]),
            (100, vec![2, 5]),
            (210, vec![2, 3, 5, 7]),
        ] {
            let union = inclusion_exclusion(
                &|s: &[usize]| {
                    let d: u64 = s.iter().map(|&i| primes[i]).product();
                    big(n / d)
                },
                primes.len(),
            );
            let coprime = big(n).sub(&union).to_i64().unwrap() as u64;
            assert_eq!(coprime, euler_phi(n), "totient of {n}");
        }

        // Explicit sets: the union size must match direct counting.
        let sets: Vec<HashSet<u64>> = vec![
            (0..20).filter(|x| x % 2 == 0).collect(),
            (0..20).filter(|x| x % 3 == 0).collect(),
            (0..20).filter(|x| x % 5 == 0).collect(),
            (7..13).collect(),
        ];
        let direct: HashSet<u64> = sets.iter().flatten().copied().collect();
        let by_ie = inclusion_exclusion(
            &|s: &[usize]| {
                let mut it = s.iter().map(|&i| &sets[i]);
                let first = it.next().unwrap().clone();
                let inter = it.fold(first, |acc, other| {
                    acc.intersection(other).copied().collect()
                });
                big(inter.len() as u64)
            },
            sets.len(),
        );
        assert_eq!(by_ie, big(direct.len() as u64));
    }

    // -----------------------------------------------------------------------
    // Puzzles and constructions
    // -----------------------------------------------------------------------

    #[test]
    fn pigeonhole_bound_is_attained_and_tight() {
        for items in 0..=40u64 {
            for boxes in 1..=10u64 {
                let bound = pigeonhole_min_overlap(items, boxes);
                // Attained: the balanced distribution has a box this full.
                let balanced_max = items / boxes + u64::from(!items.is_multiple_of(boxes));
                assert_eq!(bound, balanced_max);
                // Guaranteed: no distribution keeps every box below it.
                assert!(bound * boxes >= items);
                if bound > 0 {
                    assert!((bound - 1) * boxes < items);
                }
            }
        }
    }

    #[test]
    fn ramsey_values_are_symmetric_and_only_the_known_ones() {
        assert_eq!(ramsey_known(3, 3), Some(6));
        assert_eq!(ramsey_known(4, 4), Some(18));
        assert_eq!(ramsey_known(4, 5), Some(25));
        assert_eq!(ramsey_known(5, 5), None, "R(5,5) is not known");
        assert_eq!(ramsey_known(3, 10), None);
        for s in 0..=6u64 {
            for t in 0..=10u64 {
                assert_eq!(ramsey_known(s, t), ramsey_known(t, s));
            }
        }
        // R(2, t) = t follows from the pigeonhole argument, so it must hold
        // for every t rather than being tabulated.
        for t in 2..=50u64 {
            assert_eq!(ramsey_known(2, t), Some(t));
        }
    }

    #[test]
    fn latin_squares_are_valid_and_the_validator_rejects_near_misses() {
        let mut rng = Rng::new(5);
        for n in 1..=9usize {
            for _ in 0..20 {
                let sq = latin_square_random(n, &mut rng);
                assert!(is_latin_square(&sq), "invalid square of order {n}: {sq:?}");
            }
        }
        // A square that is row-valid but not column-valid must be rejected.
        let bad = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        assert!(!is_latin_square(&bad));
        // Ragged input is rejected.
        assert!(!is_latin_square(&[vec![0, 1], vec![1]]));
        // A symbol out of range is rejected.
        assert!(!is_latin_square(&[vec![0, 3], vec![3, 0]]));
    }

    /// Magic squares across all three residue classes, checked against the
    /// magic constant on every row, column and both diagonals, with entries
    /// forming exactly 1..=n^2.
    #[test]
    fn magic_squares_are_magic_in_all_three_constructions() {
        assert_eq!(magic_square(2), None);
        for n in [1usize, 3, 5, 7, 9, 11, 4, 8, 12, 16, 6, 10, 14] {
            let sq = magic_square(n).unwrap();
            assert_eq!(sq.len(), n);
            let magic = (n as u64) * ((n as u64) * (n as u64) + 1) / 2;
            let mut seen: Vec<u64> = sq.iter().flatten().copied().collect();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (1..=(n as u64 * n as u64)).collect::<Vec<_>>(),
                "order {n} does not use 1..=n^2 exactly once"
            );
            for (i, row) in sq.iter().enumerate() {
                assert_eq!(row.iter().sum::<u64>(), magic, "row {i} of order {n}");
            }
            for c in 0..n {
                let s: u64 = (0..n).map(|r| sq[r][c]).sum();
                assert_eq!(s, magic, "column {c} of order {n}");
            }
            let d1: u64 = (0..n).map(|i| sq[i][i]).sum();
            let d2: u64 = (0..n).map(|i| sq[i][n - 1 - i]).sum();
            assert_eq!(d1, magic, "main diagonal of order {n}");
            assert_eq!(d2, magic, "anti-diagonal of order {n}");
        }
    }

    /// A de Bruijn sequence must contain every n-tuple exactly once when read
    /// cyclically.
    #[test]
    fn de_bruijn_contains_every_tuple_once() {
        for k in 2..=4usize {
            for n in 1..=4usize {
                let seq = de_bruijn_sequence(k, n);
                assert_eq!(seq.len(), k.pow(n as u32), "length for B({k}, {n})");
                assert!(seq.iter().all(|&x| x < k));
                let mut seen: HashSet<Vec<usize>> = HashSet::new();
                for i in 0..seq.len() {
                    let window: Vec<usize> =
                        (0..n).map(|j| seq[(i + j) % seq.len()]).collect();
                    assert!(seen.insert(window.clone()), "{window:?} appears twice");
                }
                assert_eq!(seen.len(), k.pow(n as u32));
            }
        }
    }

    /// The shuffle order, verified by actually shuffling a deck that many
    /// times and checking it returns -- and that it does not return sooner.
    #[test]
    fn perfect_shuffle_order_is_the_true_period() {
        for n in (2..=40u64).step_by(2) {
            for out in [true, false] {
                let order = perfect_shuffles_order(n, out);
                let mut deck: Vec<u64> = (0..n).collect();
                let identity = deck.clone();
                for step in 1..=order {
                    deck = riffle(&deck, out);
                    if step < order {
                        assert_ne!(deck, identity, "deck returns early at step {step}");
                    }
                }
                assert_eq!(deck, identity, "deck does not return after {order} shuffles");
            }
        }
        // The classical result for a 52-card deck.
        assert_eq!(perfect_shuffles_order(52, true), 8);
        assert_eq!(perfect_shuffles_order(52, false), 52);
    }

    /// One perfect riffle: split in half and interleave. An out-shuffle keeps
    /// the original top card on top; an in-shuffle buries it.
    fn riffle(deck: &[u64], out: bool) -> Vec<u64> {
        let h = deck.len() / 2;
        let (top, bottom) = deck.split_at(h);
        let mut result = Vec::with_capacity(deck.len());
        for i in 0..h {
            if out {
                result.push(top[i]);
                result.push(bottom[i]);
            } else {
                result.push(bottom[i]);
                result.push(top[i]);
            }
        }
        result
    }

    /// Josephus against a direct simulation of the elimination circle.
    #[test]
    fn josephus_matches_direct_elimination() {
        for n in 1..=60usize {
            for k in 1..=8usize {
                let mut circle: Vec<usize> = (0..n).collect();
                let mut idx = 0usize;
                while circle.len() > 1 {
                    idx = (idx + k - 1) % circle.len();
                    circle.remove(idx);
                }
                assert_eq!(josephus(n, k), circle[0], "n = {n}, k = {k}");
            }
        }
        // The k = 2 closed form: J(n) = 2 * (n - 2^floor(log2 n)).
        for n in 1..=1000usize {
            let l = n - (1usize << (usize::BITS - 1 - n.leading_zeros()));
            assert_eq!(josephus(n, 2), 2 * l);
        }
    }

    /// Hanoi: the move list must be legal (never a larger disc on a smaller),
    /// must move every disc to the target, and must have minimal length.
    #[test]
    fn hanoi_moves_are_legal_minimal_and_complete() {
        for n in 0..=10u32 {
            let moves = tower_of_hanoi_moves(n, 0, 2);
            assert_eq!(moves.len(), (1usize << n) - 1, "not minimal for n = {n}");
            // Simulate. Each peg is a stack with the largest disc at the base.
            let mut pegs: [Vec<u32>; 3] = [(1..=n).rev().collect(), Vec::new(), Vec::new()];
            for &(from, to) in &moves {
                let disc = pegs[from as usize].pop().expect("moved from an empty peg");
                if let Some(&top) = pegs[to as usize].last() {
                    assert!(disc < top, "placed disc {disc} on smaller disc {top}");
                }
                pegs[to as usize].push(disc);
            }
            assert!(pegs[0].is_empty() && pegs[1].is_empty());
            assert_eq!(pegs[2], (1..=n).rev().collect::<Vec<_>>());
        }
    }

    /// The twelvefold way: each of the twelve entries checked against
    /// exhaustive enumeration of the maps themselves.
    #[test]
    fn twelvefold_entries_match_exhaustive_enumeration() {
        for n in 0..=5u64 {
            for k in 0..=5u64 {
                // Enumerate every function from n balls to k boxes.
                let mut all_maps: Vec<Vec<usize>> = Vec::new();
                if k > 0 || n == 0 {
                    let mut stack = vec![Vec::new()];
                    while let Some(m) = stack.pop() {
                        if m.len() as u64 == n {
                            all_maps.push(m);
                            continue;
                        }
                        for b in 0..k as usize {
                            let mut next = m.clone();
                            next.push(b);
                            stack.push(next);
                        }
                    }
                }

                for &(inj, sur) in &[
                    (None, None),
                    (Some(true), None),
                    (None, Some(true)),
                    (Some(true), Some(true)),
                ] {
                    let want_inj = inj == Some(true);
                    let want_sur = sur == Some(true);
                    let valid: Vec<&Vec<usize>> = all_maps
                        .iter()
                        .filter(|m| {
                            let mut used = vec![0usize; k as usize];
                            for &b in m.iter() {
                                used[b] += 1;
                            }
                            (!want_inj || used.iter().all(|&c| c <= 1))
                                && (!want_sur || used.iter().all(|&c| c >= 1))
                        })
                        .collect();

                    // Distinguishable balls, distinguishable boxes.
                    assert_eq!(
                        twelvefold_way(n, k, inj, sur, true, true),
                        big(valid.len() as u64),
                        "dd n={n} k={k} inj={want_inj} sur={want_sur}"
                    );

                    // Indistinguishable balls: identify maps with the same
                    // multiplicity vector.
                    let by_counts: HashSet<Vec<usize>> = valid
                        .iter()
                        .map(|m| {
                            let mut used = vec![0usize; k as usize];
                            for &b in m.iter() {
                                used[b] += 1;
                            }
                            used
                        })
                        .collect();
                    assert_eq!(
                        twelvefold_way(n, k, inj, sur, false, true),
                        big(by_counts.len() as u64),
                        "id n={n} k={k} inj={want_inj} sur={want_sur}"
                    );

                    // Indistinguishable boxes: identify maps up to relabelling
                    // the boxes, i.e. by the sorted block-size partition of the
                    // induced set partition.
                    let by_blocks: HashSet<Vec<Vec<usize>>> = valid
                        .iter()
                        .map(|m| {
                            let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); k as usize];
                            for (ball, &b) in m.iter().enumerate() {
                                blocks[b].push(ball);
                            }
                            blocks.retain(|b| !b.is_empty());
                            blocks.sort();
                            blocks
                        })
                        .collect();
                    assert_eq!(
                        twelvefold_way(n, k, inj, sur, true, false),
                        big(by_blocks.len() as u64),
                        "di n={n} k={k} inj={want_inj} sur={want_sur}"
                    );

                    // Both indistinguishable: the multiset of block sizes.
                    let by_sizes: HashSet<Vec<usize>> = valid
                        .iter()
                        .map(|m| {
                            let mut used = vec![0usize; k as usize];
                            for &b in m.iter() {
                                used[b] += 1;
                            }
                            used.retain(|&c| c > 0);
                            used.sort_unstable_by(|a, b| b.cmp(a));
                            used
                        })
                        .collect();
                    assert_eq!(
                        twelvefold_way(n, k, inj, sur, false, false),
                        big(by_sizes.len() as u64),
                        "ii n={n} k={k} inj={want_inj} sur={want_sur}"
                    );
                }
            }
        }
    }

    /// Some/false and None must be indistinguishable, as documented.
    #[test]
    fn twelvefold_treats_some_false_as_no_restriction() {
        for n in 0..=4u64 {
            for k in 0..=4u64 {
                for &db in &[true, false] {
                    for &dx in &[true, false] {
                        assert_eq!(
                            twelvefold_way(n, k, Some(false), None, db, dx),
                            twelvefold_way(n, k, None, None, db, dx)
                        );
                        assert_eq!(
                            twelvefold_way(n, k, None, Some(false), db, dx),
                            twelvefold_way(n, k, None, None, db, dx)
                        );
                    }
                }
            }
        }
    }
}
