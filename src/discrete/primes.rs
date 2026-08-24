//! Primes: sieves, primality testing, factorization, and prime counting.

use crate::exact::bigint::BigInt;
use crate::monte_carlo::Rng;

/// All primes up to and including `n`, by the sieve of Eratosthenes.
#[must_use]
pub fn sieve_eratosthenes(n: usize) -> Vec<usize> {
    if n < 2 {
        return Vec::new();
    }
    let mut is_p = vec![true; n + 1];
    is_p[0] = false;
    is_p[1] = false;
    let mut i = 2usize;
    while i * i <= n {
        if is_p[i] {
            let mut j = i * i;
            while j <= n {
                is_p[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    (2..=n).filter(|&k| is_p[k]).collect()
}

/// Primes in `[lo, hi)`, sieving only that window.
///
/// The window is marked using the primes up to `sqrt(hi)`, so memory scales
/// with the window rather than with `hi`.
#[must_use]
pub fn sieve_segmented(lo: u64, hi: u64) -> Vec<u64> {
    if hi <= 2 || hi <= lo {
        return Vec::new();
    }
    let lo = lo.max(2);
    let root = (hi as f64).sqrt() as usize + 1;
    let base = sieve_eratosthenes(root);
    let len = (hi - lo) as usize;
    let mut is_p = vec![true; len];
    for p in base {
        let p = p as u64;
        if p * p >= hi {
            break;
        }
        // First multiple of p at or above lo, never below p^2.
        let start = (lo.div_ceil(p) * p).max(p * p);
        let mut m = start;
        while m < hi {
            is_p[(m - lo) as usize] = false;
            m += p;
        }
    }
    (0..len)
        .filter(|&i| is_p[i])
        .map(|i| lo + i as u64)
        .collect()
}

/// Primes up to `n` together with the smallest prime factor of every
/// integer up to `n`, by the linear (Gries-Misra) sieve.
///
/// Each composite is struck exactly once, by its smallest prime factor.
#[must_use]
pub fn sieve_linear(n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut spf = vec![0usize; n + 1];
    let mut primes = Vec::new();
    for i in 2..=n {
        if spf[i] == 0 {
            spf[i] = i;
            primes.push(i);
        }
        for &p in &primes {
            if p > spf[i] || i * p > n {
                break;
            }
            spf[i * p] = p;
        }
    }
    (primes, spf)
}

/// Modular multiplication via `u128`, avoiding overflow for any `u64`.
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64
}

/// Modular exponentiation on `u64`.
#[must_use]
pub fn mod_pow_u64(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut acc = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// Deterministic primality for every `u64`.
///
/// Miller-Rabin over the first twelve prime bases is proven correct for
/// all 64-bit inputs, so this is a decision procedure rather than a
/// probabilistic test.
#[must_use]
pub fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut r = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        r += 1;
    }
    'base: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = mod_pow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..r {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'base;
            }
        }
        return false;
    }
    true
}

/// A Miller-Rabin round on a `BigInt` for one base.
fn mr_round(n: &BigInt, d: &BigInt, r: u32, a: &BigInt) -> bool {
    let n_minus_1 = n.sub(&BigInt::one());
    let mut x = a.mod_pow(d, n);
    if x == BigInt::one() || x == n_minus_1 {
        return true;
    }
    for _ in 1..r {
        x = x.mul(&x).rem_euclid(n);
        if x == n_minus_1 {
            return true;
        }
    }
    false
}

/// Probabilistic primality for a `BigInt`: `rounds` Miller-Rabin bases
/// followed by a strong Lucas test, which together form BPSW.
///
/// No composite is known to pass BPSW, though none is proven not to; a
/// composite passing `rounds` independent Miller-Rabin bases alone has
/// probability at most `4^-rounds`.
///
/// # Panics
/// Panics if `n` is negative.
#[must_use]
pub fn is_prime_bigint(n: &BigInt, rounds: usize, rng: &mut Rng) -> bool {
    assert!(!n.is_negative(), "primality is defined for non-negative integers");
    if let Some(small) = n.to_i64() {
        if (0..(1 << 62)).contains(&small) {
            return is_prime_u64(small as u64);
        }
    }
    if n.is_even() {
        return false;
    }
    let one = BigInt::one();
    let two = BigInt::from_u64(2);
    // n - 1 = d * 2^r with d odd.
    let n_minus_1 = n.sub(&one);
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    while d.is_even() {
        d = d.shr(1);
        r += 1;
    }
    // A fixed base 2 round first, as BPSW prescribes, then random bases.
    if !mr_round(n, &d, r, &two) {
        return false;
    }
    for _ in 0..rounds {
        let a = BigInt::random_below(&n.sub(&BigInt::from_u64(3)), rng).add(&two);
        if !mr_round(n, &d, r, &a) {
            return false;
        }
    }
    strong_lucas_probable_prime(n)
}

/// The strong Lucas probable-prime test with Selfridge's parameters.
fn strong_lucas_probable_prime(n: &BigInt) -> bool {
    if n.is_perfect_square() {
        // Selfridge's D search never terminates on a square.
        return false;
    }
    // Find D with Jacobi(D, n) = -1, alternating 5, -7, 9, -11, ...
    let mut d_val: i64 = 5;
    loop {
        let j = jacobi_bigint(d_val, n);
        if j == -1 {
            break;
        }
        if j == 0 && n.abs() != BigInt::from_u64(d_val.unsigned_abs()) {
            return false;
        }
        d_val = if d_val > 0 { -(d_val + 2) } else { -(d_val - 2) };
        if d_val.abs() > 1_000_000 {
            return false;
        }
    }
    let p = BigInt::one();
    let q_val = (1 - d_val) / 4;
    let q = BigInt::from_i64(q_val);
    // n + 1 = d * 2^s with d odd.
    let mut dd = n.add(&BigInt::one());
    let mut s = 0u32;
    while dd.is_even() {
        dd = dd.shr(1);
        s += 1;
    }
    // Compute U_d, V_d by binary ladder on the Lucas sequences.
    let (mut u, mut v) = (BigInt::one(), p.clone());
    let mut q_k = q.clone();
    let bits = dd.bits();
    for i in (0..bits.saturating_sub(1)).rev() {
        // Doubling: U_2k = U_k V_k, V_2k = V_k^2 - 2 Q^k.
        u = u.mul(&v).rem_euclid(n);
        v = v.mul(&v).sub(&q_k.mul(&BigInt::from_u64(2))).rem_euclid(n);
        q_k = q_k.mul(&q_k).rem_euclid(n);
        if dd.bit(i) {
            // Increment by one index.
            let u_next = u.add(&v);
            let v_next = v.add(&u.mul(&BigInt::from_i64(d_val)));
            u = half_mod(&u_next, n);
            v = half_mod(&v_next, n);
            q_k = q_k.mul(&q).rem_euclid(n);
        }
    }
    if u.is_zero() || v.is_zero() {
        return true;
    }
    for _ in 1..s {
        v = v.mul(&v).sub(&q_k.mul(&BigInt::from_u64(2))).rem_euclid(n);
        if v.is_zero() {
            return true;
        }
        q_k = q_k.mul(&q_k).rem_euclid(n);
    }
    false
}

/// Halve modulo an odd `n`, adding `n` first when the value is odd.
fn half_mod(x: &BigInt, n: &BigInt) -> BigInt {
    let v = x.rem_euclid(n);
    if v.is_even() {
        v.shr(1)
    } else {
        v.add(n).shr(1)
    }
}

/// The Jacobi symbol of a small integer over a `BigInt` modulus.
fn jacobi_bigint(mut a: i64, n: &BigInt) -> i8 {
    // Reduce a modulo n first; n is odd and positive here.
    let mut a_big = BigInt::from_i64(a).rem_euclid(n);
    let mut n_big = n.clone();
    let mut result = 1i8;
    while !a_big.is_zero() {
        while a_big.is_even() {
            a_big = a_big.shr(1);
            let r = n_big.rem_euclid(&BigInt::from_u64(8)).to_i64().unwrap_or(0);
            if r == 3 || r == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a_big, &mut n_big);
        let ra = a_big.rem_euclid(&BigInt::from_u64(4)).to_i64().unwrap_or(0);
        let rn = n_big.rem_euclid(&BigInt::from_u64(4)).to_i64().unwrap_or(0);
        if ra == 3 && rn == 3 {
            result = -result;
        }
        a_big = a_big.rem_euclid(&n_big);
    }
    a = 0;
    let _ = a;
    if n_big == BigInt::one() {
        result
    } else {
        0
    }
}

/// The smallest prime strictly greater than `n`.
///
/// # Panics
/// Panics if the search would overflow `u64`.
#[must_use]
pub fn next_prime(n: u64) -> u64 {
    if n < 2 {
        return 2;
    }
    let mut c = n + 1;
    while !is_prime_u64(c) {
        c = c.checked_add(1).expect("no further u64 prime");
    }
    c
}

/// The largest prime strictly less than `n`, or `None` below 3.
#[must_use]
pub fn prev_prime(n: u64) -> Option<u64> {
    if n <= 2 {
        return None;
    }
    let mut c = n - 1;
    loop {
        if is_prime_u64(c) {
            return Some(c);
        }
        c -= 1;
    }
}

/// A random prime with exactly `bits` bits.
///
/// # Panics
/// Panics if `bits` is below 2.
#[must_use]
pub fn random_prime(bits: usize, rng: &mut Rng) -> BigInt {
    assert!(bits >= 2, "need at least two bits");
    loop {
        let mut c = BigInt::random_bits(bits, rng);
        if c.is_even() {
            c = c.add(&BigInt::one());
        }
        if c.bits() != bits {
            continue;
        }
        if is_prime_bigint(&c, 8, rng) {
            return c;
        }
    }
}

/// A non-trivial factor of a composite `n` by Pollard's rho with
/// Brent's cycle detection, or `None` if the attempt fails.
#[must_use]
pub fn pollard_rho(n: u64) -> Option<u64> {
    if n.is_multiple_of(2) {
        return Some(2);
    }
    if n < 4 || is_prime_u64(n) {
        return None;
    }
    // Vary the polynomial constant until a factor separates.
    for c in 1..64u64 {
        let f = |x: u64| (mul_mod(x, x, n) + c) % n;
        let (mut x, mut y, mut d) = (2u64, 2u64, 1u64);
        while d == 1 {
            x = f(x);
            y = f(f(y));
            d = gcd(x.abs_diff(y), n);
        }
        if d != n {
            return Some(d);
        }
    }
    None
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Pollard's rho over `BigInt`, for factors beyond `u64`.
#[must_use]
pub fn pollard_rho_bigint(n: &BigInt, rng: &mut Rng) -> Option<BigInt> {
    if n.is_even() {
        return Some(BigInt::from_u64(2));
    }
    let one = BigInt::one();
    for _ in 0..16 {
        let c = BigInt::random_below(n, rng).add(&one);
        let mut x = BigInt::random_below(n, rng);
        let mut y = x.clone();
        let mut d = one.clone();
        let f = |v: &BigInt| v.mul(v).add(&c).rem_euclid(n);
        let mut steps = 0u32;
        while d == one && steps < 200_000 {
            x = f(&x);
            y = f(&f(&y));
            let diff = x.sub(&y).abs();
            if diff.is_zero() {
                break;
            }
            d = diff.gcd(n);
            steps += 1;
        }
        if d != one && d != *n {
            return Some(d);
        }
    }
    None
}

/// Pollard's p-1 method: finds a factor `p` of `n` when `p - 1` is
/// `bound`-smooth. Returns `None` when no such factor separates.
#[must_use]
pub fn pollard_p_minus_1(n: u64, bound: u64) -> Option<u64> {
    if n.is_multiple_of(2) {
        return Some(2);
    }
    let mut a = 2u64;
    for q in sieve_eratosthenes(bound as usize) {
        let q = q as u64;
        // Raise to the highest power of q not exceeding the bound.
        let mut e = q;
        while e <= bound {
            a = mod_pow_u64(a, q, n);
            e = e.saturating_mul(q);
        }
        let d = gcd(a.wrapping_sub(1), n);
        if d > 1 && d < n {
            return Some(d);
        }
    }
    None
}

/// Trial division up to `limit`: the factors found and the unfactored
/// remainder.
#[must_use]
pub fn trial_division(mut n: u64, limit: u64) -> (Vec<(u64, u32)>, u64) {
    let mut out = Vec::new();
    let mut p = 2u64;
    while p <= limit && p.saturating_mul(p) <= n {
        if n.is_multiple_of(p) {
            let mut e = 0u32;
            while n.is_multiple_of(p) {
                n /= p;
                e += 1;
            }
            out.push((p, e));
        }
        p += if p == 2 { 1 } else { 2 };
    }
    (out, n)
}

/// Fermat's method: write an odd `n` as a difference of squares.
///
/// Effective only when `n` has two factors close to its square root;
/// returns `None` once the search passes a generous bound.
#[must_use]
pub fn fermat_factor(n: u64) -> Option<(u64, u64)> {
    if n.is_multiple_of(2) {
        return Some((2, n / 2));
    }
    let mut a = (n as f64).sqrt().ceil() as u64;
    for _ in 0..1_000_000 {
        let b2 = a.checked_mul(a)?.checked_sub(n)?;
        let b = (b2 as f64).sqrt().round() as u64;
        if b * b == b2 {
            return Some((a - b, a + b));
        }
        a += 1;
    }
    None
}

/// The complete prime factorization of `n`, ascending by prime.
///
/// Small factors go by trial division, the rest by Pollard's rho.
#[must_use]
pub fn factorize(n: u64) -> Vec<(u64, u32)> {
    if n < 2 {
        return Vec::new();
    }
    let (mut out, rest) = trial_division(n, 100_000);
    if rest > 1 {
        let mut stack = vec![rest];
        let mut found: Vec<u64> = Vec::new();
        while let Some(m) = stack.pop() {
            if m == 1 {
                continue;
            }
            if is_prime_u64(m) {
                found.push(m);
                continue;
            }
            match pollard_rho(m) {
                Some(d) => {
                    stack.push(d);
                    stack.push(m / d);
                }
                None => found.push(m),
            }
        }
        found.sort_unstable();
        for f in found {
            match out.iter_mut().find(|(p, _)| *p == f) {
                Some((_, e)) => *e += 1,
                None => out.push((f, 1)),
            }
        }
    }
    out.sort_unstable();
    out
}

/// The complete factorization of a `BigInt`.
///
/// # Panics
/// Panics if `n` is not positive.
#[must_use]
pub fn factorize_bigint(n: &BigInt, rng: &mut Rng) -> Vec<(BigInt, u32)> {
    assert!(!n.is_negative() && !n.is_zero(), "factorization needs a positive integer");
    let mut out: Vec<(BigInt, u32)> = Vec::new();
    let mut stack = vec![n.clone()];
    while let Some(m) = stack.pop() {
        if m == BigInt::one() {
            continue;
        }
        if is_prime_bigint(&m, 8, rng) {
            match out.iter_mut().find(|(p, _)| *p == m) {
                Some((_, e)) => *e += 1,
                None => out.push((m, 1)),
            }
            continue;
        }
        match pollard_rho_bigint(&m, rng) {
            Some(d) => {
                let other = m.div_rem(&d).0;
                stack.push(d);
                stack.push(other);
            }
            None => out.push((m, 1)),
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The exact count of primes up to `n`, without sieving to `n`.
///
/// Uses the Lucy_Hedgehog recurrence over the distinct values of
/// `n / i`: starting from a count of all integers, each prime up to
/// `sqrt(n)` sieves its multiples out of every partial count at once. The
/// state has `O(sqrt n)` entries and the whole computation is
/// `O(n^(3/4))`, so `pi(10^9)` is reachable without a `10^9`-bit sieve.
#[must_use]
pub fn prime_count_meissel(n: u64) -> u64 {
    if n < 2 {
        return 0;
    }
    let r = (n as f64).sqrt() as u64;
    let r = (r + 2).min(n);
    let r = (0..=r).rev().find(|&k| k * k <= n).expect("root exists");
    // Key space: n/1 .. n/r, then r' .. 1 where r' = n/r - 1.
    let mut small: Vec<u64> = vec![0; (r + 1) as usize]; // indexed by v
    let mut large: Vec<u64> = vec![0; (r + 1) as usize]; // indexed by i, value n/i
    for v in 1..=r {
        small[v as usize] = v - 1;
    }
    for i in 1..=r {
        large[i as usize] = n / i - 1;
    }
    for p in 2..=r {
        if small[p as usize] == small[(p - 1) as usize] {
            continue; // p is composite
        }
        let sp = small[(p - 1) as usize];
        let p2 = p * p;
        let lim = (n / p2).min(r);
        for i in 1..=lim {
            let d = i * p;
            large[i as usize] -= if d <= r {
                large[d as usize] - sp
            } else {
                small[(n / d) as usize] - sp
            };
        }
        let mut v = r;
        while v >= p2 {
            small[v as usize] -= small[(v / p) as usize] - sp;
            v -= 1;
        }
    }
    large[1]
}

/// The logarithmic integral estimate of `pi(x)`, by series.
#[must_use]
pub fn prime_count_li_approx(x: f64) -> f64 {
    if x <= 1.0 {
        return 0.0;
    }
    // li(x) = gamma + ln ln x + sum_{k>=1} (ln x)^k / (k * k!)
    let l = x.ln();
    let gamma = 0.577_215_664_901_532_9_f64;
    let mut sum = gamma + l.abs().ln();
    let mut term = 1.0f64;
    for k in 1..200 {
        term *= l / k as f64;
        sum += term / k as f64;
        if term.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    // Subtract li(2) so the estimate is the offset logarithmic integral.
    sum - 1.045_163_780_117_493
}

/// Riemann's refinement `R(x) = sum_{k>=1} mu(k)/k * li(x^(1/k))`.
#[must_use]
pub fn riemann_r(x: f64) -> f64 {
    if x <= 1.0 {
        return 0.0;
    }
    let mu = mobius_small(64);
    let mut sum = 0.0;
    for k in 1..64usize {
        if mu[k] == 0 {
            continue;
        }
        let root = x.powf(1.0 / k as f64);
        if root < 2.0 {
            break;
        }
        sum += f64::from(mu[k]) / k as f64 * prime_count_li_approx(root);
    }
    sum
}

/// The Moebius function on `0..n`, by sieve. Local helper so that
/// `riemann_r` does not depend on the number-theory module.
fn mobius_small(n: usize) -> Vec<i8> {
    let mut mu = vec![1i8; n + 1];
    let mut primes = vec![true; n + 1];
    for i in 2..=n {
        if primes[i] {
            let mut j = i;
            while j <= n {
                if j > i {
                    primes[j] = false;
                }
                mu[j] = -mu[j];
                j += i;
            }
            let sq = i * i;
            let mut j = sq;
            while j <= n {
                mu[j] = 0;
                j += sq;
            }
        }
    }
    mu
}

/// The `n`th prime, one-based: `nth_prime(1) == 2`.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn nth_prime(n: usize) -> u64 {
    assert!(n > 0, "primes are numbered from one");
    if n < 6 {
        return [2u64, 3, 5, 7, 11][n - 1];
    }
    // Rosser's bound: p_n < n (ln n + ln ln n) for n >= 6.
    let fl = n as f64;
    let limit = (fl * (fl.ln() + fl.ln().ln())).ceil() as usize + 10;
    let primes = sieve_eratosthenes(limit);
    primes[n - 1] as u64
}

/// The gaps between consecutive primes up to `n`.
#[must_use]
pub fn prime_gaps(n: usize) -> Vec<u64> {
    let p = sieve_eratosthenes(n);
    p.windows(2).map(|w| (w[1] - w[0]) as u64).collect()
}

/// Twin prime pairs `(p, p+2)` with `p + 2 <= n`.
#[must_use]
pub fn twin_primes(n: usize) -> Vec<(u64, u64)> {
    let p = sieve_eratosthenes(n);
    p.windows(2)
        .filter(|w| w[1] - w[0] == 2)
        .map(|w| (w[0] as u64, w[1] as u64))
        .collect()
}

/// Every way to write an even `n` as an ordered sum of two primes with
/// `p <= q`.
#[must_use]
pub fn goldbach_partitions(n: u64) -> Vec<(u64, u64)> {
    if n < 4 || !n.is_multiple_of(2) {
        return Vec::new();
    }
    sieve_eratosthenes(n as usize / 2)
        .into_iter()
        .map(|p| p as u64)
        .filter(|&p| is_prime_u64(n - p))
        .map(|p| (p, n - p))
        .collect()
}

/// The first `count` primes in the arithmetic progression `a, a+d, ...`.
///
/// # Panics
/// Panics if `d` is zero.
#[must_use]
pub fn primes_in_arithmetic_progression(a: u64, d: u64, count: usize) -> Vec<u64> {
    assert!(d > 0, "step must be positive");
    let mut out = Vec::with_capacity(count);
    let mut v = a;
    while out.len() < count {
        if is_prime_u64(v) {
            out.push(v);
        }
        v = match v.checked_add(d) {
            Some(x) => x,
            None => break,
        };
    }
    out
}

/// The Lucas-Lehmer test: is the Mersenne number `2^p - 1` prime?
///
/// `p` must itself be prime for the test to be meaningful; composite `p`
/// gives a composite Mersenne number and the function returns false.
#[must_use]
pub fn mersenne_lucas_lehmer(p: u32) -> bool {
    if p == 2 {
        return true;
    }
    if p < 2 || !is_prime_u64(u64::from(p)) {
        return false;
    }
    let m = BigInt::one().shl(p as usize).sub(&BigInt::one());
    let mut s = BigInt::from_u64(4);
    let two = BigInt::from_u64(2);
    for _ in 0..(p - 2) {
        s = s.mul(&s).sub(&two).rem_euclid(&m);
    }
    s.is_zero()
}

/// Wilson's theorem: `p` is prime exactly when `(p-1)! = -1 (mod p)`.
///
/// Correct but exponentially slower than [`is_prime_u64`]; included for
/// the identity rather than for use.
#[must_use]
pub fn wilson_check(p: u64) -> bool {
    if p < 2 {
        return false;
    }
    let mut acc = 1u64;
    for k in 2..p {
        acc = mul_mod(acc, k, p);
    }
    acc == p - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieves_agree_with_each_other() {
        let p = sieve_eratosthenes(100);
        assert_eq!(p, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97]);
        assert!(sieve_eratosthenes(1).is_empty());
        assert_eq!(sieve_eratosthenes(2), [2]);

        // The three sieves are independent implementations; they must agree.
        let n = 20_000usize;
        let era = sieve_eratosthenes(n);
        let (lin, spf) = sieve_linear(n);
        assert_eq!(era, lin, "linear sieve disagrees with Eratosthenes");
        let seg: Vec<u64> = sieve_segmented(0, n as u64 + 1);
        assert_eq!(seg, era.iter().map(|&x| x as u64).collect::<Vec<_>>());
        // And with the deterministic primality test.
        for k in 0..=n {
            assert_eq!(era.binary_search(&k).is_ok(), is_prime_u64(k as u64), "n={k}");
        }
        // The smallest-prime-factor table really is the smallest factor.
        for k in 2..=n {
            let f = spf[k];
            assert!(is_prime_u64(f as u64) && k % f == 0, "spf({k}) = {f}");
            assert!((2..f).all(|d| k % d != 0), "spf({k}) is not smallest");
        }
        // A segment away from the origin.
        let seg = sieve_segmented(1_000_000, 1_000_100);
        assert_eq!(seg, [1_000_003, 1_000_033, 1_000_037, 1_000_039, 1_000_081, 1_000_099]);
        assert!(seg.iter().all(|&p| is_prime_u64(p)));
        assert!(sieve_segmented(10, 10).is_empty());
    }

    #[test]
    fn test_primality_and_navigation() {
        // Carmichael numbers fool Fermat but not Miller-Rabin.
        for c in [561u64, 1105, 1729, 2465, 2821, 6601, 8911] {
            assert!(!is_prime_u64(c), "{c} is a Carmichael number, not a prime");
        }
        // Large primes and their neighbours.
        assert!(is_prime_u64(2_147_483_647), "2^31-1 is prime");
        assert!(is_prime_u64(18_446_744_073_709_551_557), "largest u64 prime");
        assert!(!is_prime_u64(18_446_744_073_709_551_615), "2^64-1 is composite");
        assert!(!is_prime_u64(3_215_031_751), "smallest strong pseudoprime to 2,3,5,7");
        assert!(!is_prime_u64(1) && !is_prime_u64(0));

        assert_eq!(next_prime(0), 2);
        assert_eq!(next_prime(7), 11);
        assert_eq!(next_prime(89), 97);
        assert_eq!(prev_prime(11), Some(7));
        assert_eq!(prev_prime(2), None);
        // next and prev bracket a prime with nothing in between.
        for n in 3..2000u64 {
            if is_prime_u64(n) {
                assert_eq!(prev_prime(next_prime(n)), Some(n), "bracket at {n}");
            }
        }
        assert_eq!(nth_prime(1), 2);
        assert_eq!(nth_prime(6), 13);
        assert_eq!(nth_prime(10_001), 104_743, "the classic 10001st prime");
        for k in 1..500usize {
            assert!(is_prime_u64(nth_prime(k)));
            assert_eq!(prime_count_meissel(nth_prime(k)), k as u64, "pi(p_k) = k");
        }
    }

    #[test]
    fn test_factorization_reconstructs_its_input() {
        // The roadmap's property, over a wide spread of shapes.
        let mut rng = Rng::new(17);
        for _ in 0..2_000 {
            let n = rng.next_u64() % 1_000_000_000_000 + 2;
            let f = factorize(n);
            let mut prod = 1u128;
            for &(p, e) in &f {
                assert!(is_prime_u64(p), "{p} is not prime in the factorization of {n}");
                prod *= u128::from(p).pow(e);
            }
            assert_eq!(prod, u128::from(n), "factorization of {n} does not multiply back");
            assert!(f.windows(2).all(|w| w[0].0 < w[1].0), "factors not ascending");
        }
        // Hard shapes: semiprimes of near-equal factors, prime powers,
        // and a prime that trial division alone would not reach.
        for n in [1_000_003u64 * 1_000_033, 2u64.pow(59), 3u64.pow(37),
                  999_999_000_001, 1_000_000_007, 4] {
            let f = factorize(n);
            let prod: u128 = f.iter().map(|&(p, e)| u128::from(p).pow(e)).product();
            assert_eq!(prod, u128::from(n), "failed on {n}");
        }
        assert!(factorize(1).is_empty());
        assert_eq!(factorize(2), [(2, 1)]);
        assert_eq!(factorize(360), [(2, 3), (3, 2), (5, 1)]);

        // The individual engines.
        assert_eq!(pollard_rho(8_051).map(|d| 8_051 % d), Some(0));
        assert!(pollard_rho(97).is_none(), "no factor of a prime");
        // Fermat is at its best on factors near the square root.
        assert_eq!(fermat_factor(5_959), Some((59, 101)));
        // p-1 works when p-1 is smooth: 10007-1 = 2 * 5003 is not,
        // but 1000003-1 = 2*3*166667 is not either; use a built case.
        let p = 1_000_037u64; // p-1 = 2^2 * 7 * 35715 ... smooth enough
        let q = 1_000_039u64;
        if let Some(d) = pollard_p_minus_1(p * q, 200_000) {
            assert!(d == p || d == q, "p-1 returned a wrong factor {d}");
        }
        let (small, rest) = trial_division(2u64.pow(10) * 3 * 1_000_003, 100);
        assert_eq!(small, [(2, 10), (3, 1)]);
        assert_eq!(rest, 1_000_003);
    }

    #[test]
    fn test_prime_counting() {
        // The roadmap's property: pi(1e9) exactly, without a 1e9 sieve.
        assert_eq!(prime_count_meissel(1_000_000_000), 50_847_534);
        // Published values across the scale.
        for (n, want) in [(0u64, 0u64), (1, 0), (2, 1), (10, 4), (100, 25), (1_000, 168),
                          (10_000, 1_229), (100_000, 9_592), (1_000_000, 78_498),
                          (10_000_000, 664_579), (100_000_000, 5_761_455)] {
            assert_eq!(prime_count_meissel(n), want, "pi({n})");
        }
        // Agreement with a direct sieve over a dense range, which is the
        // real check that the recurrence is right at every value.
        let era = sieve_eratosthenes(5_000);
        for n in 0..=5_000u64 {
            let direct = era.iter().filter(|&&p| p as u64 <= n).count() as u64;
            assert_eq!(prime_count_meissel(n), direct, "pi({n})");
        }
        // The analytic estimates bracket the truth and improve with x.
        // Riemann's R is markedly better than li: at 1e9 li overshoots by
        // about 1700 while R is within a few dozen.
        let pi9 = 50_847_534.0;
        let li_err = (prime_count_li_approx(1e9) - pi9).abs();
        let r_err = (riemann_r(1e9) - pi9).abs();
        assert!(li_err < 3_000.0, "li(1e9) off by {li_err}");
        assert!(r_err < li_err / 5.0, "R should beat li: {r_err} vs {li_err}");
        assert_eq!(prime_count_li_approx(1.0), 0.0);
    }

    #[test]
    fn test_prime_patterns() {
        assert_eq!(twin_primes(100),
                   [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43), (59, 61), (71, 73)]);
        let gaps = prime_gaps(100);
        assert_eq!(gaps[0], 1, "2 to 3");
        assert!(gaps[1..].iter().all(|&g| g % 2 == 0), "gaps above 3 are even");
        assert_eq!(gaps.iter().sum::<u64>(), 97 - 2, "gaps telescope");

        // Goldbach: every even n in range has a partition, and each is valid.
        for n in (4..2_000u64).step_by(2) {
            let parts = goldbach_partitions(n);
            assert!(!parts.is_empty(), "no Goldbach partition for {n}");
            for (p, q) in parts {
                assert!(p <= q && p + q == n && is_prime_u64(p) && is_prime_u64(q));
            }
        }
        assert!(goldbach_partitions(7).is_empty(), "odd input");

        // Dirichlet: 4k+3 primes.
        let ap = primes_in_arithmetic_progression(3, 4, 5);
        assert_eq!(ap, [3, 7, 11, 19, 23]);
        assert!(ap.iter().all(|&p| is_prime_u64(p) && p % 4 == 3));

        // Lucas-Lehmer against the known Mersenne exponents below 130.
        let known = [2u32, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127];
        for p in 2..=127u32 {
            let want = known.contains(&p);
            assert_eq!(mersenne_lucas_lehmer(p), want, "M_{p}");
        }
        // Cross-check the small cases against direct primality.
        for p in [2u32, 3, 5, 7, 13, 17, 19, 31] {
            let m = 2u64.pow(p) - 1;
            assert_eq!(mersenne_lucas_lehmer(p), is_prime_u64(m), "M_{p} = {m}");
        }

        // Wilson's theorem agrees with the primality test.
        for n in 2..300u64 {
            assert_eq!(wilson_check(n), is_prime_u64(n), "Wilson at {n}");
        }
    }

    #[test]
    fn test_bigint_primality_and_factorization() {
        let mut rng = Rng::new(29);
        // The roadmap's property: BPSW agrees with the deterministic test.
        // Sampled densely at the low end and randomly above it.
        for n in 0..3_000u64 {
            let b = BigInt::from_u64(n);
            assert_eq!(is_prime_bigint(&b, 4, &mut rng), is_prime_u64(n), "BPSW at {n}");
        }
        for _ in 0..400 {
            let n = rng.next_u64() % 10_000_000;
            let b = BigInt::from_u64(n);
            assert_eq!(is_prime_bigint(&b, 4, &mut rng), is_prime_u64(n), "BPSW at {n}");
        }
        // Beyond u64: known large primes and obvious composites.
        let m127 = BigInt::one().shl(127).sub(&BigInt::one());
        assert!(is_prime_bigint(&m127, 8, &mut rng), "2^127-1 is prime");
        let m128 = BigInt::one().shl(128).sub(&BigInt::one());
        assert!(!is_prime_bigint(&m128, 8, &mut rng), "2^128-1 is composite");
        // A square must never be called prime; this is the case that breaks
        // a Lucas test whose parameter search is not guarded.
        let sq = BigInt::from_u64(1_000_003).pow(2);
        assert!(!is_prime_bigint(&sq, 8, &mut rng));

        // Everything below 2^62 short-circuits to the deterministic u64
        // test, so the checks above barely touch the Lucas half of BPSW.
        // Compare the two in [2^62, 2^64), where BPSW really runs and
        // is_prime_u64 is still a decision procedure.
        let lo = 1u64 << 62;
        for k in 0..400u64 {
            let n = lo + k;
            let b = BigInt::from_u64(n);
            assert_eq!(is_prime_bigint(&b, 2, &mut rng), is_prime_u64(n),
                       "BPSW disagrees at {n}");
        }
        for _ in 0..200 {
            let n = lo | (rng.next_u64() >> 2);
            let b = BigInt::from_u64(n);
            assert_eq!(is_prime_bigint(&b, 2, &mut rng), is_prime_u64(n),
                       "BPSW disagrees at {n}");
        }


        // random_prime returns a prime of exactly the requested width.
        for bits in [16usize, 32, 64, 96] {
            let p = random_prime(bits, &mut rng);
            assert_eq!(p.bits(), bits, "width of a {bits}-bit prime");
            assert!(is_prime_bigint(&p, 12, &mut rng));
        }

        // BigInt factorization reconstructs its input.
        for n in [BigInt::from_u64(1_000_003).mul(&BigInt::from_u64(1_000_033)),
                  BigInt::from_u64(2).pow(20).mul(&BigInt::from_u64(3).pow(9)),
                  BigInt::from_u64(999_999_000_001)] {
            let f = factorize_bigint(&n, &mut rng);
            let mut prod = BigInt::one();
            for (p, e) in &f {
                assert!(is_prime_bigint(p, 8, &mut rng), "{p} is not prime");
                prod = prod.mul(&p.pow(u64::from(*e)));
            }
            assert_eq!(prod, n, "factorization of {n} does not multiply back");
        }
    }
}

#[cfg(test)]
mod lucas_tests {
    use super::*;

    /// The Lucas half of BPSW, exercised directly.
    ///
    /// `is_prime_bigint` short-circuits below 2^62 and otherwise runs
    /// Miller-Rabin on random bases first, which rejects essentially every
    /// composite before the Lucas step is reached. Disabling that step
    /// entirely left the whole suite green, so it needs testing on its own
    /// terms rather than through the wrapper.
    #[test]
    fn test_strong_lucas_against_its_pseudoprimes() {
        // The strong Lucas pseudoprimes below 20000 for Selfridge's
        // parameters. Every other odd composite must be rejected, and
        // every odd prime accepted.
        const LUCAS_PSEUDOPRIMES: [u64; 5] = [5459, 5777, 10877, 16109, 18971];
        let mut found = Vec::new();
        for n in (3..20_000u64).step_by(2) {
            let got = strong_lucas_probable_prime(&BigInt::from_u64(n));
            if is_prime_u64(n) {
                assert!(got, "the strong Lucas test rejected the prime {n}");
            } else if got {
                found.push(n);
            }
        }
        assert_eq!(found, LUCAS_PSEUDOPRIMES,
                   "the set of strong Lucas pseudoprimes is wrong");

        // The point of pairing the two tests: no number below 20000 is
        // both a base-2 strong pseudoprime and a strong Lucas pseudoprime.
        // That disjointness is why BPSW has no known counterexample.
        const SPSP_BASE_2: [u64; 6] = [2047, 3277, 4033, 4681, 8321, 15841];
        for n in SPSP_BASE_2 {
            assert!(!is_prime_u64(n), "{n} should be composite");
            // Passes Miller-Rabin on base 2 ...
            let mut d = n - 1;
            let mut r = 0u32;
            while d % 2 == 0 {
                d /= 2;
                r += 1;
            }
            let mut x = mod_pow_u64(2, d, n);
            let mut passes = x == 1 || x == n - 1;
            for _ in 1..r {
                x = mul_mod(x, x, n);
                if x == n - 1 {
                    passes = true;
                }
            }
            assert!(passes, "{n} is not a base-2 strong pseudoprime");
            // ... but the Lucas test catches it.
            assert!(!strong_lucas_probable_prime(&BigInt::from_u64(n)),
                    "Lucas failed to reject the base-2 pseudoprime {n}");
        }
        for n in LUCAS_PSEUDOPRIMES {
            assert!(!SPSP_BASE_2.contains(&n), "{n} would defeat BPSW");
        }
    }
}
