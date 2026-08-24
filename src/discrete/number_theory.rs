//! Elementary and analytic number theory.
//!
//! Divisibility and the Euclidean algorithm, modular arithmetic and the
//! Chinese remainder theorem, the classical arithmetic functions (`phi`,
//! `mu`, `sigma_k`, Carmichael's `lambda`) together with their sieves,
//! multiplicative order, discrete logarithms, quadratic residues, and a
//! collection of Diophantine and digit problems.
//!
//! Factorization comes from [`crate::discrete::primes`]; nothing here
//! re-implements it.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::discrete::primes::{factorize, is_prime_u64};
use crate::exact::bigint::BigInt;
use crate::exact::rational::Rational;

// ---------------------------------------------------------------------
// internal helpers
// ---------------------------------------------------------------------

/// Modular multiplication through `u128`, exact for every `u64` modulus.
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64
}

/// Extended Euclidean algorithm on `i128`, returning `(g, x, y)` with
/// `a*x + b*y == g` and `g >= 0`.
fn extended_gcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i128, 0i128);
    let (mut old_t, mut t) = (0i128, 1i128);
    while r != 0 {
        let q = old_r / r;
        let nr = old_r - q * r;
        old_r = r;
        r = nr;
        let ns = old_s - q * s;
        old_s = s;
        s = ns;
        let nt = old_t - q * t;
        old_t = t;
        t = nt;
    }
    if old_r < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

/// Modular inverse on `u128` operands, used by the CRT combiner.
fn mod_inverse_u128(a: u128, m: u128) -> Option<u128> {
    if m == 0 {
        return None;
    }
    if m == 1 {
        return Some(0);
    }
    let (g, x, _) = extended_gcd_i128(a as i128 % m as i128, m as i128);
    if g != 1 {
        return None;
    }
    Some(x.rem_euclid(m as i128) as u128)
}

/// Greatest common divisor on `u128`.
fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Smallest `a <= b` with `a*a + b*b == n`, by direct search and without
/// the factorization test that [`sum_of_two_squares`] applies first.
fn two_squares_search(n: u64) -> Option<(u64, u64)> {
    let amax = (n / 2).isqrt();
    for a in 0..=amax {
        let r = n - a * a;
        let b = r.isqrt();
        if b * b == r {
            return Some((a, b));
        }
    }
    None
}

/// Baby-step giant-step inside a subgroup of known order.
///
/// Returns the least `x` in `[0, ord)` with `g^x == h (mod m)`.
fn bsgs_bounded(g: u64, h: u64, m: u64, ord: u64) -> Option<u64> {
    if m == 1 {
        return Some(0);
    }
    let n = ord.isqrt() + 1;
    let mut table: HashMap<u64, u64> = HashMap::new();
    let mut cur = 1 % m;
    for j in 0..n {
        table.entry(cur).or_insert(j);
        cur = mul_mod(cur, g, m);
    }
    let step = mod_inverse_u64(mod_pow_u64(g, n, m), m)?;
    let mut y = h % m;
    for i in 0..=n {
        if let Some(&j) = table.get(&y) {
            let x = i * n + j;
            if x < ord {
                return Some(x);
            }
        }
        y = mul_mod(y, step, m);
    }
    None
}

// ---------------------------------------------------------------------
// divisibility and modular arithmetic
// ---------------------------------------------------------------------

/// Greatest common divisor, by the binary (Stein) algorithm.
///
/// `gcd(0, n) == n`, so `gcd(0, 0) == 0`.
#[must_use]
pub fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            break;
        }
    }
    a << shift
}

/// Least common multiple; zero whenever either argument is zero.
///
/// # Panics
/// Panics if the least common multiple does not fit in a `u64`.
#[must_use]
pub fn lcm_u64(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd_u64(a, b)).checked_mul(b).expect("least common multiple overflows u64")
}

/// Extended Euclidean algorithm: `(g, x, y)` with `a*x + b*y == g` and
/// `g == gcd(|a|, |b|) >= 0`.
///
/// # Panics
/// Panics on `a == i64::MIN` or `b == i64::MIN`, whose negation is not
/// representable.
#[must_use]
pub fn extended_gcd_i64(a: i64, b: i64) -> (i64, i64, i64) {
    assert!(a != i64::MIN && b != i64::MIN, "i64::MIN has no representable negation");
    let (g, x, y) = extended_gcd_i128(i128::from(a), i128::from(b));
    (g as i64, x as i64, y as i64)
}

/// Modular exponentiation `base^exp mod m`.
///
/// Shares the implementation in [`crate::discrete::primes::mod_pow_u64`].
#[must_use]
pub fn mod_pow_u64(base: u64, exp: u64, m: u64) -> u64 {
    crate::discrete::primes::mod_pow_u64(base, exp, m)
}

/// The inverse of `a` modulo `m`, or `None` when `gcd(a, m) != 1`.
///
/// The residue is returned in `[0, m)`; the modulus `0` has no residues
/// and yields `None`, while modulus `1` yields `0`.
#[must_use]
pub fn mod_inverse_u64(a: u64, m: u64) -> Option<u64> {
    if m == 0 {
        return None;
    }
    if m == 1 {
        return Some(0);
    }
    let (g, x, _) = extended_gcd_i128(i128::from(a % m), i128::from(m));
    if g != 1 {
        return None;
    }
    Some(x.rem_euclid(i128::from(m)) as u64)
}

/// Chinese remainder theorem for general (not necessarily coprime)
/// moduli.
///
/// Takes `(remainder, modulus)` pairs and returns the unique class
/// `(r, m)` with `m == lcm` of the moduli and `r` in `[0, m)` satisfying
/// every congruence. Returns `None` when the system is inconsistent,
/// when any modulus is zero, or when the combined modulus overflows a
/// `u64`. An empty system is solved by `(0, 1)`.
#[must_use]
pub fn crt(residues: &[(u64, u64)]) -> Option<(u64, u64)> {
    let mut r0: u128 = 0;
    let mut m0: u128 = 1;
    for &(r, m) in residues {
        if m == 0 {
            return None;
        }
        let m1 = u128::from(m);
        let r1 = u128::from(r) % m1;
        let g = gcd_u128(m0, m1);
        let diff = r1.abs_diff(r0);
        if !diff.is_multiple_of(g) {
            return None;
        }
        let lcm = m0 / g * m1;
        if lcm > u128::from(u64::MAX) {
            return None;
        }
        let m1g = m1 / g;
        // Solve r0 + m0*t == r1 (mod m1), i.e. (m0/g)*t == (r1-r0)/g (mod m1/g).
        let t = if m1g == 1 {
            0
        } else {
            let inv = mod_inverse_u128((m0 / g) % m1g, m1g)?;
            let d = if r1 >= r0 {
                (diff / g) % m1g
            } else {
                (m1g - (diff / g) % m1g) % m1g
            };
            d * inv % m1g
        };
        r0 = (r0 + m0 * t) % lcm;
        m0 = lcm;
    }
    Some((r0 as u64, m0 as u64))
}

// ---------------------------------------------------------------------
// arithmetic functions
// ---------------------------------------------------------------------

/// Euler's totient: the count of integers in `[1, n]` coprime to `n`.
///
/// `euler_phi(0)` is defined as `0`.
#[must_use]
pub fn euler_phi(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    for (p, _) in factorize(n) {
        result = result / p * (p - 1);
    }
    result
}

/// `euler_phi` for every index up to `n`, by a sieve.
///
/// Entry `i` of the returned vector is `euler_phi(i)`, so its length is
/// `n + 1`.
#[must_use]
pub fn phi_sieve(n: usize) -> Vec<u64> {
    let mut phi: Vec<u64> = (0..=n as u64).collect();
    for i in 2..=n {
        if phi[i] == i as u64 {
            // i is prime: apply the (1 - 1/i) factor to all its multiples.
            let mut j = i;
            while j <= n {
                phi[j] -= phi[j] / i as u64;
                j += i;
            }
        }
    }
    phi[0] = 0;
    phi
}

/// The Moebius function: `0` when `n` is not squarefree, otherwise
/// `(-1)^k` for `k` distinct prime factors.
///
/// `mobius(0)` is defined as `0` and `mobius(1) == 1`.
#[must_use]
pub fn mobius(n: u64) -> i8 {
    if n == 0 {
        return 0;
    }
    let f = factorize(n);
    if f.iter().any(|&(_, e)| e > 1) {
        return 0;
    }
    if f.len().is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// `mobius` for every index up to `n`, by a linear sieve.
///
/// Entry `i` of the returned vector is `mobius(i)`, so its length is
/// `n + 1`.
#[must_use]
pub fn mobius_sieve(n: usize) -> Vec<i8> {
    let mut mu = vec![0i8; n + 1];
    if n >= 1 {
        mu[1] = 1;
    }
    let mut composite = vec![false; n + 1];
    let mut primes: Vec<usize> = Vec::new();
    for i in 2..=n {
        if !composite[i] {
            primes.push(i);
            mu[i] = -1;
        }
        for &p in &primes {
            if i * p > n {
                break;
            }
            composite[i * p] = true;
            if i.is_multiple_of(p) {
                mu[i * p] = 0;
                break;
            }
            mu[i * p] = -mu[i];
        }
    }
    mu
}

/// Every divisor of `n`, ascending. Empty for `n == 0`.
#[must_use]
pub fn divisors(n: u64) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }
    let mut small = Vec::new();
    let mut large = Vec::new();
    let mut d = 1u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            small.push(d);
            if d != n / d {
                large.push(n / d);
            }
        }
        d += 1;
    }
    large.reverse();
    small.extend(large);
    small
}

/// The number of divisors, `sigma_0(n)`. Zero for `n == 0`.
#[must_use]
pub fn divisor_count(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    factorize(n).iter().map(|&(_, e)| u64::from(e) + 1).product()
}

/// The sum of divisors, `sigma_1(n)`. Zero for `n == 0`.
///
/// # Panics
/// Panics if the sum does not fit in a `u64`.
#[must_use]
pub fn divisor_sum(n: u64) -> u64 {
    sigma_k(n, 1)
}

/// The divisor power sum `sigma_k(n) = sum_{d | n} d^k`.
///
/// `k == 0` counts divisors. Zero for `n == 0`.
///
/// # Panics
/// Panics if the sum does not fit in a `u64`.
#[must_use]
pub fn sigma_k(n: u64, k: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    if k == 0 {
        return divisor_count(n);
    }
    let mut total: u128 = 1;
    for (p, e) in factorize(n) {
        let pk = u128::from(p).checked_pow(k).expect("sigma_k overflows");
        let mut power: u128 = 1;
        let mut term: u128 = 1;
        for _ in 0..e {
            power = power.checked_mul(pk).expect("sigma_k overflows");
            term = term.checked_add(power).expect("sigma_k overflows");
        }
        total = total.checked_mul(term).expect("sigma_k overflows");
    }
    u64::try_from(total).expect("sigma_k overflows u64")
}

/// Whether `n` equals the sum of its proper divisors.
///
/// # Panics
/// Panics if the divisor sum does not fit in a `u64`.
#[must_use]
pub fn is_perfect(n: u64) -> bool {
    n > 0 && u128::from(divisor_sum(n)) == 2 * u128::from(n)
}

/// Whether the proper divisors of `n` sum to more than `n`.
///
/// # Panics
/// Panics if the divisor sum does not fit in a `u64`.
#[must_use]
pub fn is_abundant(n: u64) -> bool {
    n > 0 && u128::from(divisor_sum(n)) > 2 * u128::from(n)
}

/// Whether the proper divisors of `n` sum to less than `n`.
///
/// # Panics
/// Panics if the divisor sum does not fit in a `u64`.
#[must_use]
pub fn is_deficient(n: u64) -> bool {
    n > 0 && u128::from(divisor_sum(n)) < 2 * u128::from(n)
}

/// All amicable pairs `(a, b)` with `a < b <= limit`.
///
/// A pair is amicable when each number is the sum of the other's proper
/// divisors. Aliquot sums are built by one `O(limit log limit)` sieve.
#[must_use]
pub fn amicable_pairs(limit: u64) -> Vec<(u64, u64)> {
    let lim = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut aliquot = vec![0u64; lim + 1];
    for d in 1..=lim / 2 {
        let mut m = 2 * d;
        while m <= lim {
            aliquot[m] += d as u64;
            m += d;
        }
    }
    let mut out = Vec::new();
    for a in 2..=lim {
        let b = aliquot[a];
        if b > a as u64 && b <= limit && aliquot[b as usize] == a as u64 {
            out.push((a as u64, b));
        }
    }
    out
}

// ---------------------------------------------------------------------
// multiplicative order, primitive roots, discrete logarithms
// ---------------------------------------------------------------------

/// The least `k > 0` with `a^k == 1 (mod n)`, or `None` when `a` and `n`
/// are not coprime.
///
/// The trivial group modulo `1` gives `Some(1)`.
#[must_use]
pub fn multiplicative_order(a: u64, n: u64) -> Option<u64> {
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(1);
    }
    let a = a % n;
    if gcd_u64(a, n) != 1 {
        return None;
    }
    let mut ord = carmichael_lambda(n);
    for (p, e) in factorize(ord) {
        for _ in 0..e {
            if ord.is_multiple_of(p) && mod_pow_u64(a, ord / p, n) == 1 {
                ord /= p;
            } else {
                break;
            }
        }
    }
    Some(ord)
}

/// The least primitive root modulo the prime `p`, or `None` when `p` is
/// not prime.
///
/// A primitive root generates the whole multiplicative group, so its
/// order is `p - 1`.
#[must_use]
pub fn primitive_root(p: u64) -> Option<u64> {
    if !is_prime_u64(p) {
        return None;
    }
    if p == 2 {
        return Some(1);
    }
    let phi = p - 1;
    let qs: Vec<u64> = factorize(phi).into_iter().map(|(q, _)| q).collect();
    (2..p).find(|&g| qs.iter().all(|&q| mod_pow_u64(g, phi / q, p) != 1))
}

/// Every primitive root modulo the prime `p`, ascending.
///
/// There are `euler_phi(p - 1)` of them; the list is empty when `p` is
/// not prime.
#[must_use]
pub fn all_primitive_roots(p: u64) -> Vec<u64> {
    let Some(g) = primitive_root(p) else {
        return Vec::new();
    };
    if p == 2 {
        return vec![1];
    }
    let phi = p - 1;
    let mut out: Vec<u64> = (1..phi)
        .filter(|&k| gcd_u64(k, phi) == 1)
        .map(|k| mod_pow_u64(g, k, p))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Discrete logarithm by baby-step giant-step: the least `x >= 0` with
/// `base^x == target (mod modulus)`, or `None` when none exists.
///
/// The modulus is arbitrary — a leading reduction strips the common
/// factors of `base` and `modulus` before the classical coprime search,
/// so `base` need not be invertible. Time and memory are both
/// `O(sqrt(modulus))`.
#[must_use]
pub fn discrete_log_bsgs(base: u64, target: u64, modulus: u64) -> Option<u64> {
    if modulus == 0 {
        return None;
    }
    if modulus == 1 {
        return Some(0);
    }
    let mut a = base % modulus;
    let mut b = target % modulus;
    let mut m = modulus;
    let mut k = 1u64 % m;
    let mut add = 0u64;
    loop {
        let g = gcd_u64(a, m);
        if g == 1 {
            break;
        }
        if b == k {
            return Some(add);
        }
        if !b.is_multiple_of(g) {
            return None;
        }
        b /= g;
        m /= g;
        a %= m;
        // g divides the unreduced base, and base/g agrees with a/g modulo
        // the shrunken m, so the cofactor can be taken from base directly.
        k = mul_mod(k % m, (base / g) % m, m);
        add += 1;
        if m == 1 {
            return Some(add);
        }
    }
    if b == k {
        return Some(add);
    }
    // Solve a^x * k == b (mod m) with gcd(a, m) == 1.
    let n = m.isqrt() + 1;
    let an = mod_pow_u64(a, n, m);
    let mut table: HashMap<u64, u64> = HashMap::new();
    let mut cur = b;
    for q in 0..=n {
        table.insert(cur, q);
        cur = mul_mod(cur, a, m);
    }
    let mut cur = k;
    for p in 1..=n {
        cur = mul_mod(cur, an, m);
        if let Some(&q) = table.get(&cur) {
            return Some(n * p - q + add);
        }
    }
    None
}

/// Discrete logarithm modulo a prime by the Pohlig-Hellman reduction.
///
/// `factorization` is the factorization of the order of `base` — for a
/// primitive root, that of `p - 1`, as produced by
/// [`crate::discrete::primes::factorize`]. The logarithm is recovered in
/// each prime-power subgroup and glued by the CRT, which costs
/// `O(sum e_i (log n + sqrt(q_i)))` instead of `O(sqrt(p))`.
///
/// Returns `None` when `p` is not an odd prime, when the factorization
/// does not describe the order of `base`, or when no logarithm exists.
#[must_use]
pub fn discrete_log_pohlig_hellman(
    base: u64,
    target: u64,
    p: u64,
    factorization: &[(u64, u32)],
) -> Option<u64> {
    if p < 3 || !is_prime_u64(p) || factorization.is_empty() {
        return None;
    }
    let base = base % p;
    let target = target % p;
    let mut order: u64 = 1;
    for &(q, e) in factorization {
        order = order.checked_mul(q.checked_pow(e)?)?;
    }
    if mod_pow_u64(base, order, p) != 1 {
        return None;
    }
    let mut congruences = Vec::with_capacity(factorization.len());
    for &(q, e) in factorization {
        let qe = q.pow(e);
        let cofactor = order / qe;
        let g1 = mod_pow_u64(base, cofactor, p);
        let h1 = mod_pow_u64(target, cofactor, p);
        let gamma = mod_pow_u64(g1, qe / q, p);
        let g1inv = mod_inverse_u64(g1, p)?;
        let mut x = 0u64;
        let mut qk = 1u64;
        for _ in 0..e {
            let shifted = mul_mod(h1, mod_pow_u64(g1inv, x, p), p);
            let hk = mod_pow_u64(shifted, qe / (qk * q), p);
            let d = bsgs_bounded(gamma, hk, p, q)?;
            x += d * qk;
            qk *= q;
        }
        congruences.push((x % qe, qe));
    }
    let (r, _) = crt(&congruences)?;
    if mod_pow_u64(base, r, p) == target {
        Some(r)
    } else {
        None
    }
}

// ---------------------------------------------------------------------
// quadratic residues
// ---------------------------------------------------------------------

/// The Legendre symbol `(a/p)`: `0` when `p` divides `a`, `1` when `a`
/// is a nonzero quadratic residue, `-1` otherwise.
///
/// # Panics
/// Panics unless `p` is an odd prime.
#[must_use]
pub fn legendre_symbol(a: i64, p: u64) -> i8 {
    assert!(p > 2 && !p.is_multiple_of(2) && is_prime_u64(p), "Legendre symbol needs an odd prime");
    let am = i128::from(a).rem_euclid(i128::from(p)) as u64;
    if am == 0 {
        return 0;
    }
    if mod_pow_u64(am, (p - 1) / 2, p) == 1 {
        1
    } else {
        -1
    }
}

/// The Jacobi symbol `(a/n)` for odd `n > 0`, by reciprocity.
///
/// Equal to the Legendre symbol when `n` is prime. A value of `1` for
/// composite `n` does not imply that `a` is a residue.
///
/// # Panics
/// Panics if `n` is even or zero.
#[must_use]
pub fn jacobi_symbol(a: i64, n: u64) -> i8 {
    assert!(n > 0 && !n.is_multiple_of(2), "Jacobi symbol needs an odd positive modulus");
    let mut a = i128::from(a).rem_euclid(i128::from(n)) as u64;
    let mut n = n;
    let mut result: i8 = 1;
    while a != 0 {
        while a.is_multiple_of(2) {
            a /= 2;
            let r = n % 8;
            if r == 3 || r == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 {
        result
    } else {
        0
    }
}

/// A square root of `a` modulo the prime `p` by Tonelli-Shanks, or
/// `None` when `a` is a non-residue.
///
/// The smaller of the two roots is returned, so the result is always in
/// `[0, p/2]`.
///
/// # Panics
/// Panics unless `p` is prime.
#[must_use]
pub fn tonelli_shanks(a: u64, p: u64) -> Option<u64> {
    assert!(is_prime_u64(p), "Tonelli-Shanks needs a prime modulus");
    if p == 2 {
        return Some(a % 2);
    }
    let a = a % p;
    if a == 0 {
        return Some(0);
    }
    if mod_pow_u64(a, (p - 1) / 2, p) != 1 {
        return None;
    }
    if p % 4 == 3 {
        let r = mod_pow_u64(a, (p + 1) / 4, p);
        return Some(r.min(p - r));
    }
    // p - 1 = q * 2^s with q odd.
    let mut q = p - 1;
    let mut s = 0u32;
    while q.is_multiple_of(2) {
        q /= 2;
        s += 1;
    }
    let mut z = 2u64;
    while mod_pow_u64(z, (p - 1) / 2, p) == 1 {
        z += 1;
    }
    let mut m = s;
    let mut c = mod_pow_u64(z, q, p);
    let mut t = mod_pow_u64(a, q, p);
    let mut r = mod_pow_u64(a, q.div_ceil(2), p);
    while t != 1 {
        let mut i = 0u32;
        let mut t2 = t;
        while t2 != 1 {
            t2 = mul_mod(t2, t2, p);
            i += 1;
            if i == m {
                return None;
            }
        }
        let b = mod_pow_u64(c, 1u64 << (m - i - 1), p);
        m = i;
        c = mul_mod(b, b, p);
        t = mul_mod(t, c, p);
        r = mul_mod(r, b, p);
    }
    Some(r.min(p - r))
}

/// The nonzero quadratic residues modulo the odd prime `p`, ascending.
///
/// There are exactly `(p - 1) / 2` of them. The list is empty when `p`
/// is not an odd prime.
#[must_use]
pub fn quadratic_residues(p: u64) -> Vec<u64> {
    if p < 3 || !is_prime_u64(p) {
        return Vec::new();
    }
    let mut out: Vec<u64> = (1..=(p - 1) / 2).map(|x| mul_mod(x, x, p)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

// ---------------------------------------------------------------------
// Carmichael
// ---------------------------------------------------------------------

/// Carmichael's `lambda(n)`: the exponent of the group of units modulo
/// `n`, that is the least `k` with `a^k == 1 (mod n)` for every `a`
/// coprime to `n`.
///
/// Always a divisor of `euler_phi(n)`. `lambda(0)` is defined as `0`.
#[must_use]
pub fn carmichael_lambda(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut result = 1u64;
    for (p, e) in factorize(n) {
        let term = if p == 2 {
            match e {
                1 => 1,
                2 => 2,
                _ => 1u64 << (e - 2),
            }
        } else {
            p.pow(e - 1) * (p - 1)
        };
        result = lcm_u64(result, term);
    }
    result
}

/// Whether `n` is a Carmichael number: composite, yet `a^(n-1) == 1
/// (mod n)` for every `a` coprime to `n`.
///
/// Decided by Korselt's criterion — `n` odd, squarefree, and `p - 1`
/// divides `n - 1` for every prime `p` dividing `n`.
#[must_use]
pub fn is_carmichael(n: u64) -> bool {
    if n < 3 || n.is_multiple_of(2) || is_prime_u64(n) {
        return false;
    }
    let f = factorize(n);
    f.len() >= 2 && f.iter().all(|&(p, e)| e == 1 && (n - 1).is_multiple_of(p - 1))
}

// ---------------------------------------------------------------------
// digits
// ---------------------------------------------------------------------

/// The digits of `n` in the given base, least significant first.
fn digits_of(n: u64, base: u32) -> Vec<u64> {
    assert!(base >= 2, "base must be at least 2");
    let b = u64::from(base);
    if n == 0 {
        return vec![0];
    }
    let mut n = n;
    let mut out = Vec::new();
    while n > 0 {
        out.push(n % b);
        n /= b;
    }
    out
}

/// The sum of the digits of `n` written in `base`.
///
/// # Panics
/// Panics if `base < 2`.
#[must_use]
pub fn digit_sum(n: u64, base: u32) -> u64 {
    digits_of(n, base).iter().sum()
}

/// The digital root: repeated digit sums until a single digit remains.
///
/// Equal to `1 + (n - 1) mod (base - 1)` for positive `n`, which is the
/// closed form used here.
///
/// # Panics
/// Panics if `base < 2`.
#[must_use]
pub fn digital_root(n: u64, base: u32) -> u64 {
    assert!(base >= 2, "base must be at least 2");
    if n == 0 {
        return 0;
    }
    1 + (n - 1) % (u64::from(base) - 1)
}

/// Whether the digits of `n` in `base` read the same both ways.
///
/// # Panics
/// Panics if `base < 2`.
#[must_use]
pub fn is_palindrome(n: u64, base: u32) -> bool {
    let d = digits_of(n, base);
    let len = d.len();
    (0..len / 2).all(|i| d[i] == d[len - 1 - i])
}

/// `n` with its digits in `base` reversed.
///
/// # Panics
/// Panics if `base < 2`, or if the reversed value overflows a `u64`.
#[must_use]
pub fn reverse_digits(n: u64, base: u32) -> u64 {
    let b = u64::from(base);
    let mut out = 0u64;
    // digits_of is little-endian, so consuming it in order builds the
    // reversal directly.
    for d in digits_of(n, base) {
        out = out.checked_mul(b).and_then(|v| v.checked_add(d)).expect("reversal overflows u64");
    }
    out
}

// ---------------------------------------------------------------------
// iterated maps
// ---------------------------------------------------------------------

/// One step of the happy-number map: the sum of the squares of the
/// decimal digits.
fn happy_step(n: u64) -> u64 {
    digits_of(n, 10).iter().map(|d| d * d).sum()
}

/// Whether iterating the sum of squared decimal digits reaches `1`.
///
/// Cycle detection is by Floyd's algorithm; `0` is not happy.
#[must_use]
pub fn happy_number(n: u64) -> bool {
    if n == 0 {
        return false;
    }
    let mut slow = n;
    let mut fast = n;
    loop {
        slow = happy_step(slow);
        fast = happy_step(happy_step(fast));
        if slow == fast {
            return slow == 1;
        }
    }
}

/// The Collatz trajectory of `n`, from `n` down to the terminal `1`.
///
/// Empty for `n == 0`.
///
/// # Panics
/// Panics if some `3x + 1` step overflows a `u64`.
#[must_use]
pub fn collatz_trajectory(n: u64) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }
    let mut x = n;
    let mut out = vec![x];
    while x != 1 {
        x = if x.is_multiple_of(2) {
            x / 2
        } else {
            x.checked_mul(3).and_then(|v| v.checked_add(1)).expect("Collatz step overflows u64")
        };
        out.push(x);
    }
    out
}

/// The total stopping time: the number of Collatz steps from `n` to `1`.
///
/// Zero for `n == 0` and `n == 1`.
///
/// # Panics
/// Panics if some `3x + 1` step overflows a `u64`.
#[must_use]
pub fn collatz_stopping_time(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut steps = 0u64;
    while x != 1 {
        x = if x.is_multiple_of(2) {
            x / 2
        } else {
            x.checked_mul(3).and_then(|v| v.checked_add(1)).expect("Collatz step overflows u64")
        };
        steps += 1;
    }
    steps
}

// ---------------------------------------------------------------------
// sums of squares and Pythagorean triples
// ---------------------------------------------------------------------

/// A representation `n = a^2 + b^2` with `a <= b`, or `None` when none
/// exists.
///
/// By Fermat's two-square theorem a representation exists exactly when
/// every prime `p == 3 (mod 4)` divides `n` to an even power; that test
/// runs first, so non-representable inputs cost only a factorization.
#[must_use]
pub fn sum_of_two_squares(n: u64) -> Option<(u64, u64)> {
    if n == 0 {
        return Some((0, 0));
    }
    for (p, e) in factorize(n) {
        if p % 4 == 3 && !e.is_multiple_of(2) {
            return None;
        }
    }
    two_squares_search(n)
}

/// A representation `n = a^2 + b^2 + c^2 + d^2` with the parts
/// ascending.
///
/// Lagrange's four-square theorem guarantees one exists for every `n`.
/// The search fixes the largest part first, which leaves a small
/// remainder for the inner two-square search.
///
/// # Panics
/// Panics if no representation is found, which would contradict
/// Lagrange's theorem.
#[must_use]
pub fn sum_of_four_squares(n: u64) -> (u64, u64, u64, u64) {
    if n == 0 {
        return (0, 0, 0, 0);
    }
    for a in (0..=n.isqrt()).rev() {
        let r = n - a * a;
        for b in (0..=r.isqrt()).rev() {
            let s = r - b * b;
            if let Some((c, d)) = two_squares_search(s) {
                let mut parts = [a, b, c, d];
                parts.sort_unstable();
                return (parts[0], parts[1], parts[2], parts[3]);
            }
        }
    }
    unreachable!("Lagrange's four-square theorem guarantees a representation")
}

/// Every primitive Pythagorean triple `(a, b, c)` with `a < b < c` and
/// hypotenuse `c <= limit`, ascending.
///
/// Generated by the Berggren ternary tree rooted at `(3, 4, 5)`: every
/// primitive triple is reached exactly once, so no gcd filtering or
/// deduplication is needed.
#[must_use]
pub fn pythagorean_triples_primitive(limit: u64) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    if limit < 5 {
        return out;
    }
    let mut stack: Vec<(i64, i64, i64)> = vec![(3, 4, 5)];
    while let Some((a, b, c)) = stack.pop() {
        out.push((a.min(b) as u64, a.max(b) as u64, c as u64));
        let children = [
            (a - 2 * b + 2 * c, 2 * a - b + 2 * c, 2 * a - 2 * b + 3 * c),
            (a + 2 * b + 2 * c, 2 * a + b + 2 * c, 2 * a + 2 * b + 3 * c),
            (-a + 2 * b + 2 * c, -2 * a + b + 2 * c, -2 * a + 2 * b + 3 * c),
        ];
        for child in children {
            if child.2 as u64 <= limit {
                stack.push(child);
            }
        }
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------
// Gaussian integers
// ---------------------------------------------------------------------

/// Exact division in `Z[i]`, or `None` when the quotient is not a
/// Gaussian integer.
fn gauss_div_exact(z: (i64, i64), d: (i64, i64)) -> Option<(i64, i64)> {
    let (x, y) = (i128::from(z.0), i128::from(z.1));
    let (c, e) = (i128::from(d.0), i128::from(d.1));
    let norm = c * c + e * e;
    if norm == 0 {
        return None;
    }
    let re = x * c + y * e;
    let im = y * c - x * e;
    if re % norm != 0 || im % norm != 0 {
        return None;
    }
    Some(((re / norm) as i64, (im / norm) as i64))
}

/// Factor a Gaussian integer into Gaussian primes.
///
/// The product of the returned list reproduces the input exactly: a
/// leading unit (`-1`, `i` or `-i`) is included whenever one is needed,
/// and the empty list is returned for the input `1` and for `0`. Rational
/// primes `p == 3 (mod 4)` stay inert and appear as `(p, 0)`; `2` splits
/// as powers of `1 + i`; primes `p == 1 (mod 4)` split into the conjugate
/// pair coming from `p = a^2 + b^2`.
///
/// # Panics
/// Panics if the norm `re^2 + im^2` does not fit in a `u64`.
#[must_use]
pub fn gaussian_integer_factor(re: i64, im: i64) -> Vec<(i64, i64)> {
    if re == 0 && im == 0 {
        return Vec::new();
    }
    let norm = i128::from(re) * i128::from(re) + i128::from(im) * i128::from(im);
    let norm = u64::try_from(norm).expect("Gaussian norm overflows u64");
    let mut z = (re, im);
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (p, _) in factorize(norm) {
        let candidates: Vec<(i64, i64)> = if p == 2 {
            vec![(1, 1)]
        } else if p % 4 == 3 {
            vec![(p as i64, 0)]
        } else {
            let (a, b) = two_squares_search(p).expect("p == 1 (mod 4) is a sum of two squares");
            vec![(a as i64, b as i64), (a as i64, -(b as i64))]
        };
        for cand in candidates {
            while let Some(q) = gauss_div_exact(z, cand) {
                out.push(cand);
                z = q;
            }
        }
    }
    if z != (1, 0) {
        out.insert(0, z);
    }
    out
}

// ---------------------------------------------------------------------
// Diophantine problems
// ---------------------------------------------------------------------

/// The Frobenius number of a coin system: the largest amount that cannot
/// be paid exactly.
///
/// `None` when the coins share a common factor (infinitely many amounts
/// are then unreachable) or when the list holds no positive coin. A coin
/// of value `1` makes every non-negative amount payable and reports `0`.
/// Two coprime coins use the closed form `ab - a - b`; more coins use a
/// Dijkstra search over the residues of the smallest coin, so memory is
/// `O(min(coins))`.
#[must_use]
pub fn frobenius_number(coins: &[u64]) -> Option<u64> {
    let mut c: Vec<u64> = coins.iter().copied().filter(|&x| x > 0).collect();
    c.sort_unstable();
    c.dedup();
    if c.is_empty() {
        return None;
    }
    let g = c.iter().fold(0u64, |acc, &x| gcd_u64(acc, x));
    if g != 1 {
        return None;
    }
    if c[0] == 1 {
        return Some(0);
    }
    if c.len() == 2 {
        return Some(c[0] * c[1] - c[0] - c[1]);
    }
    let a = c[0];
    let size = usize::try_from(a).ok()?;
    let mut dist = vec![u64::MAX; size];
    dist[0] = 0;
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0, 0)));
    while let Some(Reverse((d, r))) = heap.pop() {
        if d > dist[r] {
            continue;
        }
        for &x in c.iter().skip(1) {
            let nd = d + x;
            let nr = (nd % a) as usize;
            if nd < dist[nr] {
                dist[nr] = nd;
                heap.push(Reverse((nd, nr)));
            }
        }
    }
    let worst = *dist.iter().max().expect("residue table is non-empty");
    Some(worst - a)
}

/// The greedy (Fibonacci-Sylvester) Egyptian-fraction expansion of a
/// positive rational: denominators `d` with `sum 1/d == r`.
///
/// Each step subtracts the largest unit fraction not exceeding the
/// remainder, which strictly reduces the numerator and therefore
/// terminates. An empty list is returned for `r <= 0`.
#[must_use]
pub fn egyptian_fractions_greedy(r: &Rational) -> Vec<BigInt> {
    let mut out = Vec::new();
    let zero = Rational::zero();
    let mut cur = r.clone();
    while cur > zero {
        let inv = cur.recip().expect("a positive rational has a reciprocal");
        let d = inv.ceil();
        let unit = Rational::new(BigInt::one(), d.clone()).expect("denominator is positive");
        cur = cur.sub(&unit);
        out.push(d);
    }
    out
}

/// The Zeckendorf representation of `n`: the unique set of
/// non-consecutive Fibonacci numbers summing to `n`, ascending.
///
/// Uses the Fibonacci numbers `1, 2, 3, 5, 8, ...`, each at most once.
/// Empty for `n == 0`.
#[must_use]
pub fn zeckendorf(n: u64) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }
    let mut fibs = vec![1u64, 2u64];
    loop {
        let len = fibs.len();
        let next = fibs[len - 1] + fibs[len - 2];
        if next > n {
            break;
        }
        fibs.push(next);
    }
    let mut rest = n;
    let mut out = Vec::new();
    for &f in fibs.iter().rev() {
        if f <= rest {
            out.push(f);
            rest -= f;
        }
    }
    out.reverse();
    out
}

/// The Lucas sequence `U_n(P, Q) mod m`, where `U_0 = 0`, `U_1 = 1` and
/// `U_n = P*U_{n-1} - Q*U_{n-2}`.
///
/// `U_n(1, -1)` is the Fibonacci sequence. Evaluated by the recurrence,
/// so the cost is linear in `n`. Returns `0` for `m <= 1`.
#[must_use]
pub fn lucas_sequence_u(p: i64, q: i64, n: u64, m: u64) -> u64 {
    if m <= 1 || n == 0 {
        return 0;
    }
    let pm = i128::from(p).rem_euclid(i128::from(m)) as u64;
    let qm = i128::from(q).rem_euclid(i128::from(m)) as u64;
    let mut u0 = 0u64;
    let mut u1 = 1 % m;
    for _ in 1..n {
        let next = (mul_mod(pm, u1, m) + m - mul_mod(qm, u0, m)) % m;
        u0 = u1;
        u1 = next;
    }
    u1
}

/// Every integer solution of `a*x^2 + b*y^2 == c`, ascending.
///
/// Only the definite case is enumerable: with `a > 0`, `b > 0` and
/// `c >= 0` the solution set is finite and is returned in full. An
/// indefinite form (a Pell-type equation) has infinitely many solutions,
/// so an empty list is returned there instead.
#[must_use]
pub fn quadratic_diophantine_solve(a: i64, b: i64, c: i64) -> Vec<(i64, i64)> {
    if a <= 0 || b <= 0 || c < 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let xmax = ((c / a) as u64).isqrt() as i64;
    for x in -xmax..=xmax {
        let rem = c - a * x * x;
        if rem < 0 || rem % b != 0 {
            continue;
        }
        let t = (rem / b) as u64;
        let y = t.isqrt() as i64;
        if (y * y) as u64 == t {
            out.push((x, y));
            if y != 0 {
                out.push((x, -y));
            }
        }
    }
    out.sort_unstable();
    out
}

/// Solve `a*x + b*y == c` over the integers.
///
/// Returns `(x0, y0, dx, dy)`: a particular solution together with the
/// homogeneous step, so that `(x0 + t*dx, y0 + t*dy)` is a solution for
/// every integer `t` and every solution has this form. `None` when
/// `gcd(a, b)` does not divide `c`, when both coefficients are zero, or
/// when the particular solution overflows an `i64`.
#[must_use]
pub fn linear_diophantine(a: i64, b: i64, c: i64) -> Option<(i64, i64, i64, i64)> {
    if a == 0 && b == 0 {
        return None;
    }
    let (g, x, y) = extended_gcd_i64(a, b);
    if c % g != 0 {
        return None;
    }
    let k = c / g;
    let x0 = x.checked_mul(k)?;
    let y0 = y.checked_mul(k)?;
    Some((x0, y0, b / g, -(a / g)))
}

// ---------------------------------------------------------------------
// Stern-Brocot, Farey, Dirichlet
// ---------------------------------------------------------------------

/// The `n`-th positive rational in breadth-first order on the
/// Stern-Brocot tree, counting the root `1/1` as `n == 1`.
///
/// The bits of `n` below its leading bit spell the descent: `0` goes
/// left, `1` goes right, and each node is the mediant of its bounding
/// ancestors. Every positive rational appears exactly once, already in
/// lowest terms.
///
/// # Panics
/// Panics if `n == 0`.
#[must_use]
pub fn stern_brocot_nth(n: u64) -> Rational {
    assert!(n > 0, "Stern-Brocot indexing starts at 1");
    // Bounds as (numerator, denominator); the right bound 1/0 is the
    // formal infinity, so they are kept as raw pairs rather than rationals.
    let mut lo = (BigInt::zero(), BigInt::one());
    let mut hi = (BigInt::one(), BigInt::zero());
    let mut cur = (BigInt::one(), BigInt::one());
    let leading = 63 - n.leading_zeros();
    for i in (0..leading).rev() {
        if (n >> i) & 1 == 0 {
            hi = cur.clone();
        } else {
            lo = cur.clone();
        }
        cur = (lo.0.add(&hi.0), lo.1.add(&hi.1));
    }
    Rational::new(cur.0, cur.1).expect("mediant denominators stay positive")
}

/// The next fraction after `a` in the Farey sequence of order `n`.
///
/// The successor `r/s` is the unique fraction with `s <= n` and
/// `r*q - p*s == 1` for `a = p/q`, found by solving `p*s == -1 (mod q)`
/// and taking the largest admissible `s`.
///
/// # Panics
/// Panics if `n == 0`, if `a` does not fit in `i64`, or if the
/// denominator of `a` exceeds `n`.
#[must_use]
pub fn farey_next(a: &Rational, n: u64) -> Rational {
    assert!(n > 0, "Farey order must be positive");
    let p = a.num.to_i64().expect("numerator must fit i64");
    let q = a.den.to_i64().expect("denominator must fit i64");
    let order = i64::try_from(n).expect("Farey order must fit i64");
    assert!(q <= order, "denominator exceeds the Farey order");
    if q == 1 {
        return Rational::from_i64(p * order + 1, order);
    }
    let inv = mod_inverse_u64(p.rem_euclid(q) as u64, q as u64)
        .expect("a reduced fraction has coprime parts");
    let s0 = ((q as u64 - inv % q as u64) % q as u64) as i64;
    let s = s0 + q * ((order - s0) / q);
    let r = (1 + p * s) / q;
    Rational::from_i64(r, s)
}

/// The Dirichlet convolution `(f * g)(n) = sum_{d | n} f(d) g(n/d)`.
///
/// Both slices are indexed by the argument, so element `i` holds the
/// value at `i` and element `0` is unused (it is zero on output). The
/// result has the length of the shorter input.
#[must_use]
pub fn dirichlet_convolution(f: &[i64], g: &[i64]) -> Vec<i64> {
    let len = f.len().min(g.len());
    let mut h = vec![0i64; len];
    for d in 1..len {
        if f[d] == 0 {
            continue;
        }
        let mut m = d;
        while m < len {
            h[m] += f[d] * g[m / d];
            m += d;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::rational::farey_sequence;
    use crate::monte_carlo::Rng;
    use std::collections::HashSet;

    // -- divisibility and modular arithmetic ---------------------------

    #[test]
    fn gcd_and_lcm_satisfy_the_product_identity() {
        let mut rng = Rng::new(0x00C0_FFEE);
        for _ in 0..300 {
            let a = rng.next_u64() % 1_000_000 + 1;
            let b = rng.next_u64() % 1_000_000 + 1;
            let g = gcd_u64(a, b);
            assert!(a.is_multiple_of(g) && b.is_multiple_of(g));
            assert_eq!(
                u128::from(g) * u128::from(lcm_u64(a, b)),
                u128::from(a) * u128::from(b)
            );
        }
        assert_eq!(gcd_u64(0, 7), 7);
        assert_eq!(gcd_u64(7, 0), 7);
        assert_eq!(gcd_u64(0, 0), 0);
        assert_eq!(lcm_u64(0, 7), 0);
        assert_eq!(lcm_u64(4, 6), 12);
    }

    #[test]
    fn extended_gcd_satisfies_bezout() {
        let mut rng = Rng::new(7);
        for _ in 0..300 {
            let a = (rng.next_u64() % 200_000) as i64 - 100_000;
            let b = (rng.next_u64() % 200_000) as i64 - 100_000;
            let (g, x, y) = extended_gcd_i64(a, b);
            assert_eq!(a * x + b * y, g, "Bezout identity for {a}, {b}");
            assert_eq!(g, gcd_u64(a.unsigned_abs(), b.unsigned_abs()) as i64);
            assert!(g >= 0);
        }
    }

    #[test]
    fn modular_inverse_round_trips() {
        for m in [2u64, 7, 10, 97, 1009, 65_537] {
            for a in 0..m.min(400) {
                match mod_inverse_u64(a, m) {
                    Some(inv) => {
                        assert_eq!(gcd_u64(a, m), 1);
                        assert_eq!(mul_mod(a, inv, m), 1 % m);
                        assert!(inv < m);
                    }
                    None => assert_ne!(gcd_u64(a, m), 1),
                }
            }
        }
        assert_eq!(mod_inverse_u64(3, 0), None);
        assert_eq!(mod_inverse_u64(3, 1), Some(0));
    }

    #[test]
    fn mod_pow_matches_repeated_multiplication() {
        for m in [1u64, 2, 13, 1000] {
            for b in 0..20u64 {
                for e in 0..12u64 {
                    let mut naive = 1 % m;
                    for _ in 0..e {
                        naive = naive * (b % m) % m;
                    }
                    assert_eq!(mod_pow_u64(b, e, m), naive);
                }
            }
        }
    }

    #[test]
    fn crt_result_satisfies_every_congruence() {
        let mut rng = Rng::new(11);
        let moduli = [3u64, 4, 5, 7, 8, 9, 11, 13, 16, 25];
        for _ in 0..400 {
            let x = rng.next_u64() % 100_000;
            let mut system = Vec::new();
            for _ in 0..3 {
                let m = moduli[(rng.next_u64() % moduli.len() as u64) as usize];
                system.push((x % m, m));
            }
            let (r, m) = crt(&system).expect("a system built from a witness is consistent");
            for &(ri, mi) in &system {
                assert_eq!(r % mi, ri, "congruence mod {mi} violated");
            }
            assert!(r < m);
            assert_eq!(x % m, r, "the class is the one the witness lies in");
            let lcm = system.iter().fold(1u64, |acc, &(_, mi)| lcm_u64(acc, mi));
            assert_eq!(m, lcm, "combined modulus is the lcm");
        }
    }

    #[test]
    fn crt_handles_non_coprime_moduli_and_contradictions() {
        let (r, m) = crt(&[(2, 6), (8, 15)]).expect("consistent overlap on the shared factor 3");
        assert_eq!((r, m), (8, 30));
        assert_eq!(crt(&[(1, 4), (2, 6)]), None, "1 mod 4 and 2 mod 6 disagree mod 2");
        assert_eq!(crt(&[(0, 0)]), None);
        assert_eq!(crt(&[]), Some((0, 1)));
        // Exhaustive agreement with brute force over pairs of small moduli,
        // coprime or not.
        for m1 in 2u64..12 {
            for m2 in 2u64..12 {
                for r1 in 0..m1 {
                    for r2 in 0..m2 {
                        let lcm = lcm_u64(m1, m2);
                        let brute = (0..lcm).find(|x| x % m1 == r1 && x % m2 == r2);
                        match (crt(&[(r1, m1), (r2, m2)]), brute) {
                            (Some((r, m)), Some(b)) => {
                                assert_eq!(r, b);
                                assert_eq!(m, lcm);
                            }
                            (None, None) => {}
                            (got, want) => {
                                panic!("crt disagrees for {r1} mod {m1}, {r2} mod {m2}: {got:?} vs {want:?}")
                            }
                        }
                    }
                }
            }
        }
    }

    // -- arithmetic functions ------------------------------------------

    #[test]
    fn phi_sieve_matches_euler_phi_over_a_full_range() {
        let n = 3000;
        let sieve = phi_sieve(n);
        assert_eq!(sieve.len(), n + 1);
        for i in 0..=n {
            assert_eq!(sieve[i], euler_phi(i as u64), "phi({i})");
        }
        // and euler_phi itself against the definition
        for i in 1..300u64 {
            let coprime = (1..=i).filter(|&k| gcd_u64(k, i) == 1).count() as u64;
            assert_eq!(coprime, euler_phi(i), "phi({i}) counts units");
        }
        assert_eq!(euler_phi(0), 0);
    }

    #[test]
    fn sum_of_phi_over_divisors_is_n() {
        for n in 1..=500u64 {
            let s: u64 = divisors(n).iter().map(|&d| euler_phi(d)).sum();
            assert_eq!(s, n, "sum_{{d|n}} phi(d) == n failed at {n}");
        }
    }

    #[test]
    fn mobius_sieve_matches_direct_and_sums_to_the_delta() {
        let n = 2000;
        let sieve = mobius_sieve(n);
        assert_eq!(sieve.len(), n + 1);
        for i in 0..=n {
            assert_eq!(sieve[i], mobius(i as u64), "mu({i})");
        }
        for n in 1..=500u64 {
            let s: i64 = divisors(n).iter().map(|&d| i64::from(mobius(d))).sum();
            assert_eq!(s, i64::from(n == 1), "sum_{{d|n}} mu(d) == [n == 1] failed at {n}");
        }
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(0), 0);
        assert_eq!(mobius(30), -1);
        assert_eq!(mobius(12), 0);
    }

    #[test]
    fn dirichlet_convolution_makes_mobius_and_phi_an_inverse_pair() {
        let n = 300usize;
        let one = vec![1i64; n + 1];
        let mu: Vec<i64> = (0..=n).map(|i| i64::from(mobius(i as u64))).collect();
        let phi: Vec<i64> = (0..=n).map(|i| euler_phi(i as u64) as i64).collect();
        let id: Vec<i64> = (0..=n).map(|i| i as i64).collect();

        // mu * 1 is the Dirichlet identity, so mu inverts the constant 1.
        let delta = dirichlet_convolution(&mu, &one);
        for k in 1..=n {
            assert_eq!(delta[k], i64::from(k == 1), "(mu * 1)({k})");
        }
        // phi * 1 == Id
        let recovered = dirichlet_convolution(&phi, &one);
        for k in 1..=n {
            assert_eq!(recovered[k], k as i64, "(phi * 1)({k})");
        }
        // and therefore phi == mu * Id
        let from_mobius = dirichlet_convolution(&mu, &id);
        for k in 1..=n {
            assert_eq!(from_mobius[k], phi[k], "(mu * Id)({k}) == phi({k})");
        }
        // convolution is commutative
        assert_eq!(dirichlet_convolution(&id, &mu), from_mobius);
        assert_eq!(dirichlet_convolution(&one, &one)[12], divisor_count(12) as i64);
    }

    #[test]
    fn divisor_functions_agree_with_explicit_enumeration() {
        for n in 1..=2000u64 {
            let d = divisors(n);
            assert!(d.windows(2).all(|w| w[0] < w[1]), "divisors ascend");
            assert!(d.iter().all(|&x| n.is_multiple_of(x)));
            assert_eq!(d[0], 1);
            assert_eq!(*d.last().unwrap(), n);
            assert_eq!(divisor_count(n), d.len() as u64);
            assert_eq!(divisor_sum(n), d.iter().sum::<u64>());
            assert_eq!(sigma_k(n, 0), d.len() as u64);
            assert_eq!(sigma_k(n, 1), divisor_sum(n));
            assert_eq!(sigma_k(n, 2), d.iter().map(|&x| x * x).sum::<u64>());
            assert_eq!(sigma_k(n, 3), d.iter().map(|&x| x.pow(3)).sum::<u64>());
        }
        assert!(divisors(0).is_empty());
        assert_eq!(divisor_count(0), 0);
        assert_eq!(divisor_sum(0), 0);
    }

    #[test]
    fn perfect_numbers_have_divisor_sum_twice_themselves() {
        let perfect: Vec<u64> = (1..10_000u64).filter(|&n| is_perfect(n)).collect();
        assert_eq!(perfect, vec![6, 28, 496, 8128]);
        for &p in &perfect {
            assert_eq!(divisor_sum(p), 2 * p);
        }
        for n in 1..2000u64 {
            let classes = [is_perfect(n), is_abundant(n), is_deficient(n)];
            assert_eq!(
                classes.iter().filter(|&&f| f).count(),
                1,
                "{n} must fall in exactly one class"
            );
        }
        assert!(is_abundant(12));
        assert!(is_deficient(8));
        assert!(!is_perfect(0));
    }

    #[test]
    fn amicable_pairs_below_ten_thousand_are_the_known_five() {
        let pairs = amicable_pairs(10_000);
        assert_eq!(
            pairs,
            vec![(220, 284), (1184, 1210), (2620, 2924), (5020, 5564), (6232, 6368)]
        );
        for (a, b) in pairs {
            assert_eq!(divisor_sum(a) - a, b, "aliquot sum of {a}");
            assert_eq!(divisor_sum(b) - b, a, "aliquot sum of {b}");
        }
        assert!(amicable_pairs(200).is_empty());
    }

    // -- order, primitive roots, discrete logarithms --------------------

    #[test]
    fn multiplicative_order_is_minimal_and_divides_lambda() {
        for n in 2..150u64 {
            let lambda = carmichael_lambda(n);
            for a in 1..n {
                match multiplicative_order(a, n) {
                    Some(ord) => {
                        assert_eq!(gcd_u64(a, n), 1);
                        assert_eq!(mod_pow_u64(a, ord, n), 1 % n);
                        assert!(
                            (1..ord).all(|k| mod_pow_u64(a, k, n) != 1),
                            "order of {a} mod {n} is not minimal"
                        );
                        assert!(lambda.is_multiple_of(ord));
                        assert!(euler_phi(n).is_multiple_of(ord), "Lagrange's theorem");
                    }
                    None => assert_ne!(gcd_u64(a, n), 1),
                }
            }
        }
        assert_eq!(multiplicative_order(5, 1), Some(1));
        assert_eq!(multiplicative_order(2, 4), None);
    }

    #[test]
    fn carmichael_lambda_is_the_exponent_of_the_unit_group() {
        for n in 1..300u64 {
            let lambda = carmichael_lambda(n);
            assert!(euler_phi(n).is_multiple_of(lambda), "lambda | phi failed at {n}");
            for a in 1..n {
                if gcd_u64(a, n) == 1 {
                    assert_eq!(mod_pow_u64(a, lambda, n), 1 % n, "a^lambda == 1 failed: {a} mod {n}");
                }
            }
            if n > 1 {
                assert!(
                    (1..n).any(|a| multiplicative_order(a, n) == Some(lambda)),
                    "the exponent {lambda} must be attained mod {n}"
                );
            }
        }
        assert_eq!(carmichael_lambda(0), 0);
        assert_eq!(carmichael_lambda(1), 1);
        assert_eq!(carmichael_lambda(8), 2);
        assert_eq!(carmichael_lambda(15), 4);
    }

    #[test]
    fn primitive_root_generates_the_whole_group() {
        for p in [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 97, 101, 1009] {
            let g = primitive_root(p).expect("every prime has a primitive root");
            assert_eq!(multiplicative_order(g, p), Some(p - 1), "full order p-1");
            let powers: HashSet<u64> = (0..p - 1).map(|k| mod_pow_u64(g, k, p)).collect();
            assert_eq!(powers.len() as u64, p - 1, "powers cover every unit mod {p}");
            assert!(!powers.contains(&0));
        }
        assert_eq!(primitive_root(2), Some(1));
        assert_eq!(primitive_root(9), None, "9 is not prime");
        assert_eq!(primitive_root(1), None);
    }

    #[test]
    fn all_primitive_roots_matches_brute_force() {
        for p in [3u64, 5, 7, 11, 13, 17, 19, 23, 31, 97, 101] {
            let roots = all_primitive_roots(p);
            assert_eq!(roots.len() as u64, euler_phi(p - 1), "there are phi(p-1) of them");
            assert!(roots.windows(2).all(|w| w[0] < w[1]));
            let brute: Vec<u64> =
                (1..p).filter(|&g| multiplicative_order(g, p) == Some(p - 1)).collect();
            assert_eq!(roots, brute);
        }
        assert_eq!(all_primitive_roots(2), vec![1]);
        assert!(all_primitive_roots(15).is_empty());
    }

    #[test]
    fn discrete_logs_agree_and_reproduce_the_target() {
        let mut rng = Rng::new(2024);
        for p in [101u64, 1009, 7919] {
            let g = primitive_root(p).unwrap();
            let factors = factorize(p - 1);
            for _ in 0..25 {
                let x = rng.next_u64() % (p - 1);
                let target = mod_pow_u64(g, x, p);
                let bsgs = discrete_log_bsgs(g, target, p).expect("a generator hits every unit");
                assert_eq!(mod_pow_u64(g, bsgs, p), target, "base^result == target");
                assert_eq!(bsgs, x, "the least logarithm of a generator power is the exponent");
                let ph = discrete_log_pohlig_hellman(g, target, p, &factors)
                    .expect("Pohlig-Hellman solves the same instance");
                assert_eq!(ph, bsgs, "both algorithms agree");
                assert_eq!(mod_pow_u64(g, ph, p), target);
            }
        }
        // A non-generator: solvable only inside its own subgroup.
        let p = 101u64;
        let h = mod_pow_u64(primitive_root(p).unwrap(), 4, p); // order 25
        assert_eq!(multiplicative_order(h, p), Some(25));
        let factors = factorize(25);
        for x in 0..25u64 {
            let target = mod_pow_u64(h, x, p);
            assert_eq!(discrete_log_pohlig_hellman(h, target, p, &factors), Some(x));
        }
        assert_eq!(discrete_log_pohlig_hellman(2, 3, 4, &[(3, 1)]), None, "4 is not prime");
    }

    #[test]
    fn bsgs_handles_composite_moduli_and_matches_brute_force() {
        for m in 2u64..25 {
            for base in 0..m {
                for target in 0..m {
                    let brute = (0..2 * m).find(|&x| mod_pow_u64(base, x, m) == target);
                    match discrete_log_bsgs(base, target, m) {
                        Some(x) => {
                            assert_eq!(mod_pow_u64(base, x, m), target, "{base}^{x} mod {m}");
                            assert_eq!(Some(x), brute, "least solution for {base}^x = {target} mod {m}");
                        }
                        None => {
                            assert_eq!(brute, None, "missed {base}^x = {target} mod {m}");
                        }
                    }
                }
            }
        }
        // Larger non-invertible bases.
        for m in [1024u64, 999, 1_000_000] {
            for base in [2u64, 6, 10] {
                for x in 0..15u64 {
                    let target = mod_pow_u64(base, x, m);
                    let got = discrete_log_bsgs(base, target, m).expect("x itself is a solution");
                    assert_eq!(mod_pow_u64(base, got, m), target);
                    assert!(got <= x);
                }
            }
        }
        assert_eq!(discrete_log_bsgs(3, 5, 1), Some(0));
        assert_eq!(discrete_log_bsgs(3, 5, 0), None);
    }

    // -- quadratic residues --------------------------------------------

    #[test]
    fn legendre_matches_enumeration_and_jacobi_agrees_on_primes() {
        for p in [3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 101] {
            let qr = quadratic_residues(p);
            assert_eq!(qr.len() as u64, (p - 1) / 2, "half the units are residues");
            assert!(qr.windows(2).all(|w| w[0] < w[1]));
            for a in 0..p {
                let expected = if a == 0 {
                    0
                } else if qr.contains(&a) {
                    1
                } else {
                    -1
                };
                assert_eq!(legendre_symbol(a as i64, p), expected, "({a}/{p})");
                assert_eq!(jacobi_symbol(a as i64, p), expected, "Jacobi == Legendre mod {p}");
            }
            for a in -20i64..20 {
                assert_eq!(legendre_symbol(a, p), legendre_symbol(a + 10 * p as i64, p));
            }
        }
        assert!(quadratic_residues(4).is_empty());
    }

    #[test]
    fn jacobi_is_multiplicative_and_obeys_reciprocity() {
        let mut rng = Rng::new(99);
        for _ in 0..300 {
            let n = 2 * (rng.next_u64() % 500) + 3;
            let a = (rng.next_u64() % 2000) as i64 - 1000;
            let b = (rng.next_u64() % 2000) as i64 - 1000;
            assert_eq!(
                jacobi_symbol(a * b, n),
                jacobi_symbol(a, n) * jacobi_symbol(b, n),
                "multiplicative in the numerator"
            );
            // definition: the product of Legendre symbols over the prime factors
            let mut expected: i8 = 1;
            for (p, e) in factorize(n) {
                for _ in 0..e {
                    expected *= legendre_symbol(a, p);
                }
            }
            assert_eq!(jacobi_symbol(a, n), expected, "({a}/{n}) by definition");
        }
        let mut rng = Rng::new(1234);
        let mut checked = 0;
        for _ in 0..400 {
            let m = 2 * (rng.next_u64() % 300) + 3;
            let n = 2 * (rng.next_u64() % 300) + 3;
            if gcd_u64(m, n) != 1 {
                continue;
            }
            let sign = if m % 4 == 3 && n % 4 == 3 { -1 } else { 1 };
            assert_eq!(
                jacobi_symbol(m as i64, n) * jacobi_symbol(n as i64, m),
                sign,
                "quadratic reciprocity for {m}, {n}"
            );
            checked += 1;
        }
        assert!(checked > 100, "reciprocity was exercised");
    }

    #[test]
    fn tonelli_shanks_returns_a_genuine_square_root() {
        for p in [3u64, 5, 7, 11, 13, 17, 29, 41, 97, 101, 1009, 10_007, 65_537] {
            let mut residues = 0u64;
            for a in 0..p.min(400) {
                match tonelli_shanks(a, p) {
                    Some(r) => {
                        assert_eq!(mul_mod(r, r, p), a % p, "sqrt({a}) mod {p} squared back");
                        assert!(r <= p / 2, "the smaller root is returned");
                        if a > 0 {
                            residues += 1;
                            assert_eq!(legendre_symbol(a as i64, p), 1);
                        }
                    }
                    None => assert_eq!(legendre_symbol(a as i64, p), -1, "{a} mod {p}"),
                }
            }
            assert!(residues > 0);
        }
        assert_eq!(tonelli_shanks(0, 2), Some(0));
        assert_eq!(tonelli_shanks(1, 2), Some(1));
        assert_eq!(tonelli_shanks(2, 3), None);
    }

    // -- Carmichael -----------------------------------------------------

    #[test]
    fn carmichael_numbers_below_ten_thousand() {
        let numbers: Vec<u64> = (1..10_000u64).filter(|&n| is_carmichael(n)).collect();
        assert_eq!(numbers, vec![561, 1105, 1729, 2465, 2821, 6601, 8911]);
        for &n in &numbers {
            assert!(!is_prime_u64(n), "{n} must be composite");
            assert!((n - 1).is_multiple_of(carmichael_lambda(n)), "lambda(n) | n-1");
            for a in 2..n.min(600) {
                if gcd_u64(a, n) == 1 {
                    assert_eq!(
                        mod_pow_u64(a, n - 1, n),
                        1,
                        "{n} is not a Fermat pseudoprime to base {a}"
                    );
                }
            }
        }
        // an ordinary composite has a Fermat witness
        assert!(!is_carmichael(15));
        assert!((2..15u64).any(|a| gcd_u64(a, 15) == 1 && mod_pow_u64(a, 14, 15) != 1));
        assert!(!is_carmichael(7), "primes are excluded");
    }

    // -- digits ---------------------------------------------------------

    #[test]
    fn digit_functions_agree_with_the_base_expansion() {
        assert_eq!(digit_sum(9875, 10), 29);
        assert_eq!(digit_sum(255, 16), 30);
        assert_eq!(digit_sum(0, 10), 0);
        assert_eq!(digital_root(9875, 10), 1 + (9875 - 1) % 9);
        assert_eq!(reverse_digits(1230, 10), 321);
        assert!(is_palindrome(12_321, 10));
        assert!(!is_palindrome(1231, 10));
        assert!(is_palindrome(0b1001_1001, 2));

        for n in 0..2000u64 {
            for base in [2u32, 3, 7, 10, 16] {
                let b = u64::from(base);
                let mut m = n;
                let mut sum = 0;
                while m > 0 {
                    sum += m % b;
                    m /= b;
                }
                assert_eq!(digit_sum(n, base), sum, "digit sum of {n} base {base}");
                // the digital root is the fixed point of iterated digit sums
                let mut r = n;
                while r >= b {
                    r = digit_sum(r, base);
                }
                assert_eq!(digital_root(n, base), r, "digital root of {n} base {base}");
                let rev = reverse_digits(n, base);
                assert_eq!(is_palindrome(n, base), rev == n);
                if !n.is_multiple_of(b) || n == 0 {
                    assert_eq!(reverse_digits(rev, base), n, "reversal is an involution");
                }
            }
        }
    }

    // -- iterated maps ---------------------------------------------------

    #[test]
    fn happy_numbers_match_the_known_list() {
        let happy: Vec<u64> = (1..=50u64).filter(|&n| happy_number(n)).collect();
        assert_eq!(happy, vec![1, 7, 10, 13, 19, 23, 28, 31, 32, 44, 49]);
        assert!(!happy_number(0));
        // happiness is invariant along the orbit
        for n in 1..500u64 {
            assert_eq!(happy_number(n), happy_number(happy_step(n)), "orbit invariance at {n}");
        }
        // the unhappy cycle is the classic 4 -> 16 -> 37 -> ... -> 4
        assert!(!happy_number(4));
        assert_eq!(happy_step(4), 16);
    }

    #[test]
    fn collatz_trajectories_reach_one() {
        for n in 1..2000u64 {
            let t = collatz_trajectory(n);
            assert_eq!(t[0], n);
            assert_eq!(*t.last().unwrap(), 1, "{n} reaches 1");
            assert_eq!(t.len() as u64, collatz_stopping_time(n) + 1);
            for w in t.windows(2) {
                let expected = if w[0].is_multiple_of(2) { w[0] / 2 } else { 3 * w[0] + 1 };
                assert_eq!(w[1], expected, "step from {}", w[0]);
            }
        }
        assert_eq!(collatz_stopping_time(27), 111);
        assert_eq!(collatz_stopping_time(1), 0);
        assert_eq!(collatz_trajectory(1), vec![1]);
        assert!(collatz_trajectory(0).is_empty());
    }

    // -- sums of squares --------------------------------------------------

    #[test]
    fn two_squares_exists_exactly_when_fermat_allows_it() {
        for n in 0..3000u64 {
            let allowed = factorize(n).iter().all(|&(p, e)| p % 4 != 3 || e.is_multiple_of(2));
            match sum_of_two_squares(n) {
                Some((a, b)) => {
                    assert!(allowed, "{n} should not be a sum of two squares");
                    assert!(a <= b);
                    assert_eq!(a * a + b * b, n, "{a}^2 + {b}^2 == {n}");
                }
                None => assert!(allowed.eq(&false), "missed a representation of {n}"),
            }
        }
        assert_eq!(sum_of_two_squares(25), Some((0, 5)));
        assert_eq!(sum_of_two_squares(3), None);
        assert_eq!(sum_of_two_squares(0), Some((0, 0)));
    }

    #[test]
    fn four_squares_always_exists() {
        for n in 0..1500u64 {
            let (a, b, c, d) = sum_of_four_squares(n);
            assert!(a <= b && b <= c && c <= d, "parts ascend");
            assert_eq!(a * a + b * b + c * c + d * d, n, "Lagrange decomposition of {n}");
        }
        for n in [99_991u64, 123_456, 1_000_003, 4_294_967_291] {
            let (a, b, c, d) = sum_of_four_squares(n);
            assert_eq!(a * a + b * b + c * c + d * d, n);
        }
    }

    #[test]
    fn primitive_pythagorean_triples_match_brute_force() {
        let limit = 120u64;
        let tree = pythagorean_triples_primitive(limit);
        let mut brute = Vec::new();
        for c in 1..=limit {
            for a in 1..c {
                for b in a + 1..c {
                    if a * a + b * b == c * c && gcd_u64(gcd_u64(a, b), c) == 1 {
                        brute.push((a, b, c));
                    }
                }
            }
        }
        brute.sort_unstable();
        assert_eq!(tree, brute, "the Berggren tree enumerates exactly the primitive triples");
        for &(a, b, c) in &tree {
            assert_eq!(a * a + b * b, c * c);
            assert_eq!(gcd_u64(gcd_u64(a, b), c), 1, "primitive");
            assert!(a < b && b < c);
        }
        let below_100 = brute.iter().filter(|t| t.2 <= 100).count();
        assert_eq!(pythagorean_triples_primitive(100).len(), below_100);
        assert!(tree.contains(&(3, 4, 5)) && tree.contains(&(20, 21, 29)));
        assert!(pythagorean_triples_primitive(4).is_empty());
    }

    // -- Gaussian integers -------------------------------------------------

    #[test]
    fn gaussian_factors_multiply_back_to_the_input() {
        let mut rng = Rng::new(555);
        for _ in 0..150 {
            let re = (rng.next_u64() % 200) as i64 - 100;
            let im = (rng.next_u64() % 200) as i64 - 100;
            if re == 0 && im == 0 {
                continue;
            }
            let factors = gaussian_integer_factor(re, im);
            let mut product = (1i64, 0i64);
            for &(a, b) in &factors {
                product = (product.0 * a - product.1 * b, product.0 * b + product.1 * a);
            }
            assert_eq!(product, (re, im), "factors of {re}+{im}i multiply back");
            for (i, &(a, b)) in factors.iter().enumerate() {
                let norm = (a * a + b * b) as u64;
                if i == 0 && norm == 1 {
                    continue; // leading unit
                }
                let split = is_prime_u64(norm);
                let inert = b == 0 && is_prime_u64(a.unsigned_abs()) && a.unsigned_abs() % 4 == 3;
                assert!(split || inert, "{a}+{b}i is not a Gaussian prime");
            }
        }
        assert!(gaussian_integer_factor(0, 0).is_empty());
        assert!(gaussian_integer_factor(1, 0).is_empty());
        // 2 = -i (1 + i)^2
        assert_eq!(gaussian_integer_factor(2, 0), vec![(0, -1), (1, 1), (1, 1)]);
        // 3 stays inert, 5 splits
        assert_eq!(gaussian_integer_factor(3, 0), vec![(3, 0)]);
        assert_eq!(gaussian_integer_factor(5, 0).len(), 2);
    }

    // -- Diophantine problems ---------------------------------------------

    #[test]
    fn frobenius_two_coins_matches_the_closed_form() {
        for a in 2..40u64 {
            for b in a + 1..40 {
                if gcd_u64(a, b) == 1 {
                    assert_eq!(
                        frobenius_number(&[a, b]),
                        Some(a * b - a - b),
                        "Chicken McNugget for {a}, {b}"
                    );
                } else {
                    assert_eq!(frobenius_number(&[a, b]), None);
                }
            }
        }
    }

    #[test]
    fn frobenius_search_matches_brute_force() {
        let sets: [&[u64]; 6] =
            [&[6, 9, 20], &[3, 5, 7], &[4, 7, 10], &[5, 8, 12], &[11, 13, 17], &[7, 11, 13, 18]];
        for coins in sets {
            let f = frobenius_number(coins).expect("coprime coin systems have a Frobenius number");
            let bound = (f + 200) as usize;
            let mut representable = vec![false; bound + 1];
            representable[0] = true;
            for v in 1..=bound {
                representable[v] =
                    coins.iter().any(|&c| v as u64 >= c && representable[v - c as usize]);
            }
            let largest = (0..=bound).filter(|&v| !representable[v]).max().unwrap_or(0) as u64;
            assert_eq!(f, largest, "Frobenius number of {coins:?}");
            assert!(!representable[f as usize]);
            assert!((f + 1..=f + 100).all(|v| representable[v as usize]), "everything above is payable");
        }
        assert_eq!(frobenius_number(&[6, 9, 20]), Some(43));
        assert_eq!(frobenius_number(&[4, 6]), None);
        assert_eq!(frobenius_number(&[1, 5]), Some(0));
        assert_eq!(frobenius_number(&[]), None);
    }

    #[test]
    fn egyptian_fractions_sum_back_exactly() {
        for (n, d) in [(3i64, 7i64), (5, 6), (2, 3), (4, 5), (5, 121), (7, 15), (9, 4)] {
            let r = Rational::from_i64(n, d);
            let terms = egyptian_fractions_greedy(&r);
            assert!(!terms.is_empty());
            let mut sum = Rational::zero();
            for t in &terms {
                let unit = Rational::new(BigInt::one(), t.clone()).expect("positive denominator");
                sum = sum.add(&unit);
            }
            assert_eq!(sum, r, "greedy expansion of {n}/{d} sums back");
            if n < d {
                assert!(
                    terms.windows(2).all(|w| w[0] < w[1]),
                    "proper fractions give strictly increasing denominators"
                );
            }
        }
        assert_eq!(egyptian_fractions_greedy(&Rational::from_i64(3, 7)).len(), 3);
        assert!(egyptian_fractions_greedy(&Rational::zero()).is_empty());
        assert!(egyptian_fractions_greedy(&Rational::from_i64(-1, 2)).is_empty());
    }

    #[test]
    fn zeckendorf_uses_non_consecutive_fibonacci_numbers() {
        let mut fibs = vec![1u64, 2];
        while *fibs.last().unwrap() < 10_000 {
            let k = fibs.len();
            fibs.push(fibs[k - 1] + fibs[k - 2]);
        }
        for n in 1..3000u64 {
            let z = zeckendorf(n);
            assert_eq!(z.iter().sum::<u64>(), n, "terms sum to {n}");
            assert!(z.iter().all(|f| fibs.contains(f)), "every term is a Fibonacci number");
            assert!(z.windows(2).all(|w| w[0] < w[1]));
            for w in z.windows(2) {
                let i = fibs.iter().position(|&f| f == w[0]).unwrap();
                let j = fibs.iter().position(|&f| f == w[1]).unwrap();
                assert!(j >= i + 2, "consecutive Fibonacci terms in the expansion of {n}");
            }
        }
        assert!(zeckendorf(0).is_empty());
        assert_eq!(zeckendorf(100), vec![3, 8, 89]);
    }

    #[test]
    fn lucas_sequence_matches_the_recurrence() {
        let m = 1_000_000_007u64;
        let mut fib = vec![0u64, 1];
        for i in 2..60 {
            let v = (fib[i - 1] + fib[i - 2]) % m;
            fib.push(v);
        }
        for n in 0..60u64 {
            assert_eq!(lucas_sequence_u(1, -1, n, m), fib[n as usize], "U_n(1,-1) is Fibonacci");
        }
        for (p, q) in [(3i64, 2i64), (5, -3), (-2, 4), (1, 1)] {
            let md = 97u64;
            let mut u: Vec<i128> = vec![0, 1];
            for i in 2..40 {
                let v = i128::from(p) * u[i - 1] - i128::from(q) * u[i - 2];
                u.push(v.rem_euclid(i128::from(md)));
            }
            for n in 0..40u64 {
                let want = u64::try_from(u[n as usize]).unwrap();
                assert_eq!(lucas_sequence_u(p, q, n, md), want, "U_{n}({p},{q}) mod {md}");
            }
        }
        assert_eq!(lucas_sequence_u(1, -1, 10, 1000), 55);
        assert_eq!(lucas_sequence_u(2, -1, 5, 1_000_000), 29, "Pell numbers");
        assert_eq!(lucas_sequence_u(1, -1, 10, 1), 0);
    }

    #[test]
    fn linear_diophantine_particular_plus_homogeneous() {
        let mut rng = Rng::new(31_337);
        for _ in 0..400 {
            let a = (rng.next_u64() % 200) as i64 - 100;
            let b = (rng.next_u64() % 200) as i64 - 100;
            let c = (rng.next_u64() % 400) as i64 - 200;
            match linear_diophantine(a, b, c) {
                Some((x0, y0, dx, dy)) => {
                    assert_eq!(a * x0 + b * y0, c, "particular solution of {a}x + {b}y = {c}");
                    for t in -4i64..=4 {
                        assert_eq!(
                            a * (x0 + t * dx) + b * (y0 + t * dy),
                            c,
                            "homogeneous step preserves the solution"
                        );
                    }
                    assert!(dx != 0 || dy != 0, "the step is non-trivial");
                }
                None => {
                    if a == 0 && b == 0 {
                        continue;
                    }
                    let g = gcd_u64(a.unsigned_abs(), b.unsigned_abs()) as i64;
                    assert_ne!(c % g, 0, "solvable systems must not be rejected");
                }
            }
        }
        assert_eq!(linear_diophantine(0, 0, 5), None);
        assert_eq!(linear_diophantine(6, 9, 5), None);
        let (x, y, dx, dy) = linear_diophantine(6, 9, 21).unwrap();
        assert_eq!(6 * x + 9 * y, 21);
        assert_eq!((dx, dy), (3, -2));
    }

    #[test]
    fn quadratic_diophantine_finds_every_solution() {
        for a in 1i64..5 {
            for b in 1i64..5 {
                for c in 0i64..60 {
                    let solutions = quadratic_diophantine_solve(a, b, c);
                    for &(x, y) in &solutions {
                        assert_eq!(a * x * x + b * y * y, c);
                    }
                    let mut brute = Vec::new();
                    for x in -10i64..=10 {
                        for y in -10i64..=10 {
                            if a * x * x + b * y * y == c {
                                brute.push((x, y));
                            }
                        }
                    }
                    brute.sort_unstable();
                    assert_eq!(solutions, brute, "{a}x^2 + {b}y^2 = {c}");
                }
            }
        }
        assert_eq!(quadratic_diophantine_solve(1, 1, 25).len(), 12);
        assert!(
            quadratic_diophantine_solve(1, -1, 5).is_empty(),
            "indefinite forms are not enumerated"
        );
    }

    // -- Stern-Brocot and Farey ---------------------------------------------

    #[test]
    fn stern_brocot_enumerates_the_positive_rationals() {
        let expected = [(1i64, 1i64), (1, 2), (2, 1), (1, 3), (2, 3), (3, 2), (3, 1)];
        for (i, &(n, d)) in expected.iter().enumerate() {
            assert_eq!(stern_brocot_nth(i as u64 + 1), Rational::from_i64(n, d), "index {}", i + 1);
        }
        let mut seen = HashSet::new();
        for n in 1..=511u64 {
            let r = stern_brocot_nth(n);
            assert!(r > Rational::zero(), "every entry is positive");
            assert_eq!(r.num.gcd(&r.den), BigInt::one(), "already in lowest terms");
            assert!(seen.insert(r.to_string()), "no rational appears twice");
        }
        for num in 1i64..=4 {
            for den in 1i64..=4 {
                let target = Rational::from_i64(num, den);
                assert!(
                    (1..=511u64).any(|n| stern_brocot_nth(n) == target),
                    "{num}/{den} must appear"
                );
            }
        }
    }

    #[test]
    fn farey_next_walks_the_farey_sequence() {
        for n in 1..=12u64 {
            let seq = farey_sequence(n);
            for w in seq.windows(2) {
                assert_eq!(farey_next(&w[0], n), w[1], "successor in F_{n}");
            }
        }
        for n in 2..=20u64 {
            let seq = farey_sequence(n);
            for a in &seq[..seq.len() - 1] {
                let b = farey_next(a, n);
                // neighbours in a Farey sequence satisfy r*q - p*s == 1
                let det = b.num.mul(&a.den).sub(&a.num.mul(&b.den));
                assert_eq!(det, BigInt::one(), "unimodular neighbours {a} and {b}");
                assert!(b > *a);
                assert!(b.den.to_i64().unwrap() as u64 <= n, "denominator within the order");
            }
        }
        assert_eq!(farey_next(&Rational::from_i64(1, 3), 5), Rational::from_i64(2, 5));
        assert_eq!(farey_next(&Rational::from_i64(0, 1), 5), Rational::from_i64(1, 5));
        assert_eq!(farey_next(&Rational::from_i64(1, 1), 5), Rational::from_i64(6, 5));
    }
}
