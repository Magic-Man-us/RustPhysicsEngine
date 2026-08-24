//! Integer sequences, linear recurrences, and generating functions.
//!
//! Two halves. The first recovers a sequence from an analytic or algebraic
//! description: Taylor coefficients from a function by Cauchy's integral,
//! and the minimal linear recurrence from a prefix by Berlekamp-Massey. The
//! second is the named sequences themselves.

use crate::discrete::number_theory::divisor_sum;
use crate::exact::bigint::BigInt;
use crate::exact::rational::Rational;
use crate::fractals::Complex;
use crate::transforms::fft::fft;

// ---------------------------------------------------------------------------
// Generating functions
// ---------------------------------------------------------------------------

/// The first `n` Taylor coefficients of `f` about the origin, by Cauchy's
/// integral evaluated on a circle of the given radius.
///
/// `a_k = (1 / 2 pi i) * contour integral of f(z) / z^(k+1)`. Sampling the
/// circle at `N` equally spaced points turns that into a discrete Fourier
/// transform, so all `N` coefficients come out of one FFT rather than `n`
/// separate quadratures.
///
/// The radius is the accuracy knob and the caller owns it: it must be inside
/// the disc of convergence, and the error in `a_k` scales like
/// `(radius / R)^N` for the true radius of convergence `R`. A radius near `R`
/// resolves high-order coefficients but amplifies the low-order ones by
/// `radius^-k`; a small radius does the reverse.
///
/// Returns the real parts, so this is for series with real coefficients.
///
/// # Panics
/// Panics if `n` is zero or `radius` is not positive.
#[must_use]
pub fn ogf_coefficients(f: &dyn Fn(Complex) -> Complex, n: usize, radius: f64) -> Vec<f64> {
    assert!(n > 0, "n must be positive");
    assert!(radius > 0.0, "radius must be positive");
    // Oversample to at least four times the requested order, rounded to a
    // power of two, so the aliasing term (radius/R)^N is pushed well down.
    let mut size = 1usize;
    while size < 4 * n {
        size <<= 1;
    }
    let samples: Vec<Complex> = (0..size)
        .map(|j| {
            let theta = std::f64::consts::TAU * j as f64 / size as f64;
            f(Complex::new(radius * theta.cos(), radius * theta.sin()))
        })
        .collect();
    let spectrum = fft(&samples);
    let mut out = Vec::with_capacity(n);
    let mut scale = 1.0 / size as f64;
    for k in 0..n {
        out.push(spectrum[k].re * scale);
        scale /= radius;
    }
    out
}

/// Converts exponential generating function coefficients to ordinary ones by
/// multiplying term `k` by `k!`.
///
/// The factorial overflows `f64` past `k = 170`, so the tail beyond that is
/// infinite rather than silently wrong.
#[must_use]
pub fn egf_to_ogf(coeffs: &[f64]) -> Vec<f64> {
    let mut fact = 1.0f64;
    coeffs
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            if k > 0 {
                fact *= k as f64;
            }
            c * fact
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Linear recurrences
// ---------------------------------------------------------------------------

/// The `n`-th term of the linear recurrence
/// `a_k = coeffs[0] a_{k-1} + coeffs[1] a_{k-2} + ...`, with `init` giving
/// `a_0 .. a_{order-1}`.
///
/// # Panics
/// Panics unless `init` and `coeffs` have the same non-zero length.
#[must_use]
pub fn linear_recurrence(init: &[i64], coeffs: &[i64], n: u64) -> BigInt {
    assert!(!init.is_empty(), "the recurrence needs an initial segment");
    assert_eq!(
        init.len(),
        coeffs.len(),
        "one coefficient per initial term is required"
    );
    let order = init.len();
    if (n as usize) < order {
        return BigInt::from_i64(init[n as usize]);
    }
    let mut window: Vec<BigInt> = init.iter().map(|&x| BigInt::from_i64(x)).collect();
    let cs: Vec<BigInt> = coeffs.iter().map(|&x| BigInt::from_i64(x)).collect();
    for _ in order..=n as usize {
        // window[order - 1] is the most recent term, so coefficient i pairs
        // with window[order - 1 - i].
        let mut next = BigInt::zero();
        for (i, c) in cs.iter().enumerate() {
            next = next.add(&c.mul(&window[order - 1 - i]));
        }
        window.remove(0);
        window.push(next);
    }
    window[order - 1].clone()
}

/// The `n`-th term of the same recurrence, modulo `m`, by matrix
/// exponentiation.
///
/// Costs `O(order^3 log n)` rather than `O(order * n)`, which is what makes an
/// index like `10^18` reachable.
///
/// # Panics
/// Panics unless `init` and `coeffs` have the same non-zero length, or if `m`
/// is zero.
#[must_use]
pub fn linear_recurrence_mod(init: &[i64], coeffs: &[i64], n: u64, m: u64) -> u64 {
    assert!(!init.is_empty(), "the recurrence needs an initial segment");
    assert_eq!(init.len(), coeffs.len(), "one coefficient per initial term");
    assert!(m > 0, "modulus must be positive");
    let k = init.len();
    let red = |x: i64| -> u64 { x.rem_euclid(m as i64) as u64 };
    if (n as usize) < k {
        return red(init[n as usize]);
    }
    // Companion matrix: the top row is the coefficients, with a shifted
    // identity beneath, so multiplying advances the window by one step.
    let mut base = vec![vec![0u64; k]; k];
    for j in 0..k {
        base[0][j] = red(coeffs[j]);
    }
    for i in 1..k {
        base[i][i - 1] = 1 % m;
    }
    let power = mat_pow_mod(&base, n - (k as u64 - 1), m);
    // The state vector holds a_{k-1} down to a_0.
    let mut acc = 0u128;
    for j in 0..k {
        acc += power[0][j] as u128 * red(init[k - 1 - j]) as u128 % m as u128;
    }
    (acc % m as u128) as u64
}

fn mat_mul_mod(a: &[Vec<u64>], b: &[Vec<u64>], m: u64) -> Vec<Vec<u64>> {
    let k = a.len();
    let mut out = vec![vec![0u64; k]; k];
    for i in 0..k {
        for l in 0..k {
            if a[i][l] == 0 {
                continue;
            }
            let av = a[i][l] as u128;
            for j in 0..k {
                out[i][j] = ((out[i][j] as u128 + av * b[l][j] as u128) % m as u128) as u64;
            }
        }
    }
    out
}

fn mat_pow_mod(a: &[Vec<u64>], mut e: u64, m: u64) -> Vec<Vec<u64>> {
    let k = a.len();
    let mut result = vec![vec![0u64; k]; k];
    for (i, row) in result.iter_mut().enumerate() {
        row[i] = 1 % m;
    }
    let mut base = a.to_vec();
    while e > 0 {
        if e & 1 == 1 {
            result = mat_mul_mod(&result, &base, m);
        }
        base = mat_mul_mod(&base, &base, m);
        e >>= 1;
    }
    result
}

/// The shortest linear recurrence generating `seq`, by Berlekamp-Massey over
/// the rationals.
///
/// Returns `c` with `a_n = c[0] a_{n-1} + c[1] a_{n-2} + ...`, or `None` when
/// the sequence is too short to determine one. A recurrence of order `L` is
/// only pinned down by `2L` terms, so a candidate found from fewer is a guess;
/// this reports `None` in that case rather than returning it. The empty vector
/// is returned for the all-zero sequence, whose recurrence has order zero.
#[must_use]
pub fn find_linear_recurrence(seq: &[Rational]) -> Option<Vec<Rational>> {
    let n = seq.len();
    if n == 0 {
        return None;
    }
    // c is the connection polynomial with c[0] = 1; b is the previous one.
    let mut c = vec![Rational::one()];
    let mut b = vec![Rational::one()];
    let mut l = 0usize;
    let mut shift = 1usize;
    let mut last_d = Rational::one();

    for i in 0..n {
        // Discrepancy: how far the current polynomial misses term i.
        let mut d = seq[i].clone();
        for j in 1..=l {
            d = d.add(&c[j].mul(&seq[i - j]));
        }
        if d.is_zero() {
            shift += 1;
            continue;
        }
        let scale = d.div(&last_d).expect("last_d is non-zero once set");
        // c <- c - scale * x^shift * b
        let mut next = c.clone();
        if next.len() < b.len() + shift {
            next.resize(b.len() + shift, Rational::zero());
        }
        for (j, bj) in b.iter().enumerate() {
            next[j + shift] = next[j + shift].sub(&scale.mul(bj));
        }
        if 2 * l <= i {
            b = c;
            l = i + 1 - l;
            last_d = d;
            shift = 1;
        } else {
            shift += 1;
        }
        c = next;
    }

    if 2 * l > n {
        return None;
    }
    // a_n = -c[1] a_{n-1} - c[2] a_{n-2} - ...
    Some((1..=l).map(|j| c[j].neg()).collect())
}

/// The connection polynomial of the shortest linear feedback shift register
/// generating `seq` over GF(2), returned as taps `t` with
/// `a_n = t[0] a_{n-1} XOR t[1] a_{n-2} XOR ...`.
///
/// Same algorithm as [`find_linear_recurrence`] with the field replaced by
/// GF(2), where every non-zero discrepancy is one and subtraction is XOR, so
/// there is no division to do.
#[must_use]
pub fn berlekamp_massey_gf2(seq: &[bool]) -> Vec<bool> {
    let n = seq.len();
    let mut c = vec![true];
    let mut b = vec![true];
    let mut l = 0usize;
    let mut shift = 1usize;

    for i in 0..n {
        let mut d = seq[i];
        for j in 1..=l {
            d ^= c[j] && seq[i - j];
        }
        if !d {
            shift += 1;
            continue;
        }
        let mut next = c.clone();
        if next.len() < b.len() + shift {
            next.resize(b.len() + shift, false);
        }
        for (j, &bj) in b.iter().enumerate() {
            next[j + shift] ^= bj;
        }
        if 2 * l <= i {
            b = c;
            l = i + 1 - l;
            shift = 1;
        } else {
            shift += 1;
        }
        c = next;
    }
    (1..=l).map(|j| c[j]).collect()
}

// ---------------------------------------------------------------------------
// Fibonacci and friends
// ---------------------------------------------------------------------------

/// `F(n) mod m`, by fast doubling.
///
/// The identities `F(2k) = F(k) (2 F(k+1) - F(k))` and
/// `F(2k+1) = F(k)^2 + F(k+1)^2` halve the index each step, so this is
/// `O(log n)` multiplications rather than `O(n)` additions.
///
/// # Panics
/// Panics if `m` is zero.
#[must_use]
pub fn fibonacci_mod(n: u64, m: u64) -> u64 {
    assert!(m > 0, "modulus must be positive");
    fn go(n: u64, m: u128) -> (u128, u128) {
        if n == 0 {
            return (0, 1 % m);
        }
        let (a, b) = go(n >> 1, m);
        // c = F(2k), d = F(2k+1); the doubled b may leave a negative
        // difference, so add a multiple of m before subtracting.
        let c = a * ((2 * b + m - a % m) % m) % m;
        let d = (a * a + b * b) % m;
        if n & 1 == 0 { (c, d) } else { (d, (c + d) % m) }
    }
    go(n, m as u128).0 as u64
}

/// The Pisano period: the period of the Fibonacci sequence modulo `m`.
///
/// Found by advancing until the pair `(0, 1)` recurs, which is the state that
/// starts the sequence, so the first recurrence is the full period.
///
/// # Panics
/// Panics if `m` is zero.
#[must_use]
pub fn pisano_period(m: u64) -> u64 {
    assert!(m > 0, "modulus must be positive");
    if m == 1 {
        return 1;
    }
    let (mut a, mut b) = (0u64, 1u64);
    let mut period = 0u64;
    loop {
        let next = (a + b) % m;
        a = b;
        b = next;
        period += 1;
        if a == 0 && b == 1 {
            return period;
        }
    }
}

/// The `n`-th Lucas number: `L(0) = 2`, `L(1) = 1`, `L(n) = L(n-1) + L(n-2)`.
#[must_use]
pub fn lucas(n: u64) -> BigInt {
    two_term(BigInt::from_u64(2), BigInt::one(), 1, 1, n)
}

/// The `n`-th Pell number: `P(0) = 0`, `P(1) = 1`, `P(n) = 2 P(n-1) + P(n-2)`.
#[must_use]
pub fn pell_number(n: u64) -> BigInt {
    two_term(BigInt::zero(), BigInt::one(), 2, 1, n)
}

/// The `n`-th Jacobsthal number: `J(0) = 0`, `J(1) = 1`,
/// `J(n) = J(n-1) + 2 J(n-2)`.
#[must_use]
pub fn jacobsthal(n: u64) -> BigInt {
    two_term(BigInt::zero(), BigInt::one(), 1, 2, n)
}

/// `a_n = p a_{n-1} + q a_{n-2}` from the given two seeds.
fn two_term(a0: BigInt, a1: BigInt, p: u64, q: u64, n: u64) -> BigInt {
    if n == 0 {
        return a0;
    }
    let (pb, qb) = (BigInt::from_u64(p), BigInt::from_u64(q));
    let (mut prev, mut cur) = (a0, a1);
    for _ in 1..n {
        let next = pb.mul(&cur).add(&qb.mul(&prev));
        prev = cur;
        cur = next;
    }
    cur
}

/// The `n`-th tribonacci number: `0, 0, 1, 1, 2, 4, 7, 13, ...`.
#[must_use]
pub fn tribonacci(n: u64) -> BigInt {
    let mut w = [BigInt::zero(), BigInt::zero(), BigInt::one()];
    if (n as usize) < 3 {
        return w[n as usize].clone();
    }
    for _ in 3..=n {
        let next = w[0].add(&w[1]).add(&w[2]);
        w = [w[1].clone(), w[2].clone(), next];
    }
    w[2].clone()
}

// ---------------------------------------------------------------------------
// Self-describing and digit sequences
// ---------------------------------------------------------------------------

/// The look-and-say sequence: each step reads the previous term aloud.
///
/// `"1"` becomes `"11"` (one 1), which becomes `"21"` (two 1s), and so on.
///
/// # Panics
/// Panics if `seed` is empty or contains a non-digit.
#[must_use]
pub fn look_and_say(seed: &str, iterations: usize) -> String {
    assert!(!seed.is_empty(), "the seed must be non-empty");
    assert!(
        seed.chars().all(|c| c.is_ascii_digit()),
        "the seed must be digits"
    );
    let mut cur: Vec<char> = seed.chars().collect();
    for _ in 0..iterations {
        let mut next = String::new();
        let mut i = 0usize;
        while i < cur.len() {
            let c = cur[i];
            let mut run = 0usize;
            while i < cur.len() && cur[i] == c {
                run += 1;
                i += 1;
            }
            next.push_str(&run.to_string());
            next.push(c);
        }
        cur = next.chars().collect();
    }
    cur.into_iter().collect()
}

/// Conway's constant, estimated from the growth of look-and-say lengths.
///
/// The true value 1.303577... is the unique real root above one of Conway's
/// degree-71 polynomial. Lengths grow at that rate asymptotically, but the
/// single-step ratio does not settle onto it smoothly: it is still swinging
/// between 1.3137 and 1.3510 at twenty iterations, so reading off one ratio
/// would be worse at twenty steps than at sixteen. The swing has period four,
/// so this takes the geometric mean across a four-step window instead, which
/// cancels most of it and reaches four digits by thirty iterations.
///
/// Fewer than four iterations are run as four, since the window needs them.
#[must_use]
pub fn conway_constant_estimate(iters: usize) -> f64 {
    const LAG: usize = 4;
    let mut cur = String::from("1");
    let mut lengths = vec![1usize];
    for _ in 0..iters.max(LAG) {
        cur = look_and_say(&cur, 1);
        lengths.push(cur.len());
    }
    let last = lengths.len() - 1;
    (lengths[last] as f64 / lengths[last - LAG] as f64).powf(1.0 / LAG as f64)
}

/// The `n`-th Thue-Morse bit: the parity of the number of ones in `n`.
#[must_use]
pub fn thue_morse(n: u64) -> bool {
    n.count_ones() % 2 == 1
}

/// The first `n` bits of the Thue-Morse sequence.
#[must_use]
pub fn thue_morse_sequence(n: usize) -> Vec<bool> {
    (0..n as u64).map(thue_morse).collect()
}

/// The first `n` terms of the Kolakoski sequence over `{1, 2}`.
///
/// The sequence is its own run-length encoding: it starts `1, 2, 2, 1, 1, 2`,
/// whose run lengths are `1, 2, 2, 1, 1, 2` again. Generated by reading the
/// sequence back as it is written -- term `k` says how long run `k` is.
#[must_use]
pub fn kolakoski(n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    // Three terms have to be seeded before the sequence can be read back: the
    // first run is a single 1, described by k(0), and the second is a pair of
    // 2s, described by k(1). Only from run 2 onwards does the reader trail far
    // enough behind the writer to stay inside what is already written.
    let mut out: Vec<u8> = vec![1, 2, 2];
    out.truncate(n);
    let mut reader = 2usize;
    let mut value = 1u8;
    while out.len() < n {
        let run = out[reader];
        for _ in 0..run {
            if out.len() == n {
                break;
            }
            out.push(value);
        }
        reader += 1;
        // Runs alternate between the two symbols.
        value = 3 - value;
    }
    out
}

/// The first `n` terms of Recaman's sequence.
///
/// `a(0) = 0`; each step subtracts the index if the result is positive and
/// has not appeared before, and otherwise adds it.
#[must_use]
pub fn recaman(n: usize) -> Vec<i64> {
    if n == 0 {
        return Vec::new();
    }
    let mut seen = std::collections::HashSet::new();
    seen.insert(0i64);
    let mut out = vec![0i64];
    for k in 1..n {
        let prev = out[k - 1];
        let back = prev - k as i64;
        let next = if back > 0 && !seen.contains(&back) {
            back
        } else {
            prev + k as i64
        };
        seen.insert(next);
        out.push(next);
    }
    out
}

/// The first `n` terms of the Ulam sequence starting `a, b`.
///
/// After the seeds, each term is the smallest integer larger than the last
/// that is the sum of two distinct earlier terms in exactly one way.
///
/// # Panics
/// Panics unless `0 < a < b`.
#[must_use]
pub fn ulam_sequence(a: u64, b: u64, n: usize) -> Vec<u64> {
    assert!(a > 0 && a < b, "the seeds must satisfy 0 < a < b");
    let mut seq = vec![a, b];
    seq.truncate(n);
    while seq.len() < n {
        let mut candidate = seq[seq.len() - 1] + 1;
        loop {
            // Count representations as a sum of two distinct earlier terms.
            let mut ways = 0usize;
            for i in 0..seq.len() {
                for j in i + 1..seq.len() {
                    if seq[i] + seq[j] == candidate {
                        ways += 1;
                        if ways > 1 {
                            break;
                        }
                    }
                }
                if ways > 1 {
                    break;
                }
            }
            if ways == 1 {
                seq.push(candidate);
                break;
            }
            candidate += 1;
        }
    }
    seq
}

/// The aliquot sequence from `n`: repeatedly replace a number by the sum of
/// its proper divisors.
///
/// Stops early at zero, which is terminal, and at a repeat, which means the
/// sequence has entered a cycle (a perfect number, an amicable pair, or a
/// longer sociable chain). The returned vector includes `n` itself and the
/// repeated value, so a cycle is visible in the output.
#[must_use]
pub fn aliquot_sequence(n: u64, max_steps: usize) -> Vec<u64> {
    let mut out = vec![n];
    let mut seen = std::collections::HashSet::new();
    seen.insert(n);
    let mut cur = n;
    for _ in 0..max_steps {
        if cur == 0 {
            break;
        }
        cur = divisor_sum(cur) - cur;
        out.push(cur);
        if cur == 0 || !seen.insert(cur) {
            break;
        }
    }
    out
}

/// The Ackermann function, for arguments whose value is representable.
///
/// `A(m, n)` is computed by the closed forms rather than the recursion, which
/// would not terminate in practice: `A(0,n) = n+1`, `A(1,n) = n+2`,
/// `A(2,n) = 2n+3`, `A(3,n) = 2^(n+3) - 3`, and `A(4,n)` is a tower of twos.
/// Returns `None` when the value cannot be built -- `A(4, 2)` already has
/// 19729 digits and `A(5, 0) = A(4, 1)` is the largest value below it that
/// this returns.
#[must_use]
pub fn ackermann_small(m: u64, n: u64) -> Option<BigInt> {
    match m {
        0 => Some(BigInt::from_u64(n + 1)),
        1 => Some(BigInt::from_u64(n + 2)),
        2 => Some(BigInt::from_u64(2 * n + 3)),
        3 => {
            // 2^(n+3) - 3, refused past a megabit of result.
            if n + 3 > 1_000_000 {
                return None;
            }
            Some(BigInt::one().shl((n + 3) as usize).sub(&BigInt::from_u64(3)))
        }
        4 => {
            // A(4, n) = 2^^(n+3) - 3, a tower of n+3 twos.
            if n > 1 {
                return None;
            }
            let mut v = BigInt::from_u64(2);
            for _ in 0..n + 2 {
                let e = v.to_i64()?;
                if e > 1_000_000 {
                    return None;
                }
                v = BigInt::one().shl(e as usize);
            }
            Some(v.sub(&BigInt::from_u64(3)))
        }
        // A(m, 0) = A(m-1, 1), which is the only reachable case above m = 4.
        _ if n == 0 => ackermann_small(m - 1, 1),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------

/// Names of the known sequences whose opening terms match `terms`.
///
/// Every candidate family is generated and compared term by term, so a name is
/// returned only on an exact match of the whole input. A linear recurrence
/// found by [`find_linear_recurrence`] is reported as well, which covers the
/// families not listed by name.
///
/// The result is a list because short prefixes are genuinely ambiguous:
/// `1, 1, 2` opens the Fibonacci numbers, the Catalan numbers, and the
/// partition counts alike.
#[must_use]
pub fn sequence_identify(terms: &[i64]) -> Vec<String> {
    let mut names = Vec::new();
    if terms.is_empty() {
        return names;
    }
    let n = terms.len();
    let want = |gen: &dyn Fn(u64) -> Option<BigInt>| -> bool {
        (0..n as u64).all(|i| gen(i).is_some_and(|v| v == BigInt::from_i64(terms[i as usize])))
    };

    let families: Vec<(&str, Box<dyn Fn(u64) -> Option<BigInt>>)> = vec![
        ("constant zero", Box::new(|_| Some(BigInt::zero()))),
        ("constant one", Box::new(|_| Some(BigInt::one()))),
        ("natural numbers", Box::new(|i| Some(BigInt::from_u64(i)))),
        ("positive integers", Box::new(|i| Some(BigInt::from_u64(i + 1)))),
        ("odd numbers", Box::new(|i| Some(BigInt::from_u64(2 * i + 1)))),
        ("even numbers", Box::new(|i| Some(BigInt::from_u64(2 * i)))),
        ("squares", Box::new(|i| Some(BigInt::from_u64(i * i)))),
        ("cubes", Box::new(|i| Some(BigInt::from_u64(i * i * i)))),
        (
            "triangular numbers",
            Box::new(|i| Some(BigInt::from_u64(i * (i + 1) / 2))),
        ),
        (
            "powers of two",
            Box::new(|i| Some(BigInt::from_u64(2).pow(i))),
        ),
        (
            "powers of three",
            Box::new(|i| Some(BigInt::from_u64(3).pow(i))),
        ),
        ("factorials", Box::new(|i| Some(BigInt::factorial(i)))),
        ("Fibonacci numbers", Box::new(|i| Some(BigInt::fibonacci(i)))),
        ("Lucas numbers", Box::new(|i| Some(lucas(i)))),
        ("Pell numbers", Box::new(|i| Some(pell_number(i)))),
        ("Jacobsthal numbers", Box::new(|i| Some(jacobsthal(i)))),
        ("tribonacci numbers", Box::new(|i| Some(tribonacci(i)))),
        (
            "Catalan numbers",
            Box::new(|i| Some(crate::discrete::combinatorics::catalan(i))),
        ),
        (
            "Motzkin numbers",
            Box::new(|i| Some(crate::discrete::combinatorics::motzkin(i))),
        ),
        (
            "large Schroeder numbers",
            Box::new(|i| Some(crate::discrete::combinatorics::schroeder(i))),
        ),
        (
            "Bell numbers",
            Box::new(|i| Some(crate::discrete::combinatorics::bell_number(i))),
        ),
        (
            "derangement counts",
            Box::new(|i| Some(crate::discrete::combinatorics::derangements_count(i))),
        ),
        (
            "central binomial coefficients",
            Box::new(|i| Some(BigInt::binomial(2 * i, i))),
        ),
        (
            "partition counts",
            Box::new(|i| Some(crate::discrete::partitions::partition_count(i))),
        ),
        (
            "partition counts into distinct parts",
            Box::new(|i| Some(crate::discrete::partitions::partitions_distinct(i))),
        ),
        (
            "primes",
            Box::new(|i| {
                let ps = crate::discrete::primes::sieve_eratosthenes(1000);
                ps.get(i as usize).map(|&p| BigInt::from_u64(p as u64))
            }),
        ),
        (
            "Thue-Morse sequence",
            Box::new(|i| Some(BigInt::from_u64(u64::from(thue_morse(i))))),
        ),
        (
            "Recaman's sequence",
            Box::new(|i| Some(BigInt::from_i64(recaman(i as usize + 1)[i as usize]))),
        ),
        (
            "Mersenne numbers",
            Box::new(|i| Some(BigInt::from_u64(2).pow(i).sub(&BigInt::one()))),
        ),
        (
            "Euler totients",
            Box::new(|i| {
                (i > 0).then(|| {
                    BigInt::from_u64(crate::discrete::number_theory::euler_phi(i))
                })
            }),
        ),
        (
            "divisor counts",
            Box::new(|i| {
                (i > 0).then(|| {
                    BigInt::from_u64(crate::discrete::number_theory::divisor_count(i))
                })
            }),
        ),
    ];

    for (name, gen) in &families {
        if want(gen.as_ref()) {
            names.push((*name).to_string());
        }
    }

    // Then the general case: any linear recurrence the prefix determines.
    let rationals: Vec<Rational> = terms
        .iter()
        .map(|&x| Rational::from_int(BigInt::from_i64(x)))
        .collect();
    if let Some(c) = find_linear_recurrence(&rationals) {
        if !c.is_empty() {
            let body = c
                .iter()
                .enumerate()
                .map(|(i, r)| format!("({}) a(n-{})", rational_text(r), i + 1))
                .collect::<Vec<_>>()
                .join(" + ");
            names.push(format!("linear recurrence a(n) = {body}"));
        }
    }
    names
}

fn rational_text(r: &Rational) -> String {
    if r.is_integer() {
        r.floor().to_string()
    } else {
        format!("{}/{}", r.num, r.den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::number_theory::gcd_u64;

    fn big(n: u64) -> BigInt {
        BigInt::from_u64(n)
    }

    fn rat(n: i64) -> Rational {
        Rational::from_i64(n, 1)
    }

    // -----------------------------------------------------------------------
    // Generating functions
    // -----------------------------------------------------------------------

    /// Cauchy extraction against Taylor coefficients known in closed form.
    #[test]
    fn ogf_recovers_known_taylor_coefficients() {
        // 1 / (1 - z) has every coefficient 1.
        let geom = ogf_coefficients(
            &|z| Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - z),
            12,
            0.5,
        );
        for (k, &c) in geom.iter().enumerate() {
            assert!((c - 1.0).abs() < 1e-9, "coefficient {k} is {c}");
        }

        // 1 / (1 - z - z^2) is the Fibonacci generating function.
        let fib = ogf_coefficients(
            &|z| {
                let d = Complex::new(1.0, 0.0) - z - z * z;
                Complex::new(1.0, 0.0) / d
            },
            15,
            0.4,
        );
        for (k, &c) in fib.iter().enumerate() {
            let want = BigInt::fibonacci(k as u64 + 1).to_f64();
            assert!(
                (c - want).abs() < 1e-6 * want.max(1.0),
                "Fibonacci coefficient {k} is {c}, expected {want}"
            );
        }

        // exp(z): coefficients 1/k!.
        let e = ogf_coefficients(
            &|z| {
                let m = z.re.exp();
                Complex::new(m * z.im.cos(), m * z.im.sin())
            },
            10,
            1.0,
        );
        let mut fact = 1.0f64;
        for (k, &c) in e.iter().enumerate() {
            if k > 0 {
                fact *= k as f64;
            }
            assert!((c - 1.0 / fact).abs() < 1e-10, "exp coefficient {k}");
        }
    }

    /// The error is aliasing, and for a function with known coefficients it
    /// can be predicted exactly rather than merely bounded.
    ///
    /// Sampling at `N` points folds coefficient `k + jN` into coefficient `k`
    /// with weight `radius^(jN)`. For `1/(1-z)`, whose coefficients are all
    /// one, that sum is `r^N / (1 - r^N)` for every `k` alike. Matching the
    /// measured error against that closed form tests the mechanism, not just
    /// the magnitude.
    #[test]
    fn ogf_error_is_exactly_the_predicted_aliasing() {
        for &r in &[0.5f64, 0.7, 0.9] {
            // ogf_coefficients oversamples to the next power of two above 4n.
            let n = 4usize;
            let mut size = 1usize;
            while size < 4 * n {
                size <<= 1;
            }
            let predicted = r.powi(size as i32) / (1.0 - r.powi(size as i32));
            let c = ogf_coefficients(
                &|z| Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - z),
                n,
                r,
            );
            for (k, &v) in c.iter().enumerate() {
                let err = v - 1.0;
                assert!(
                    (err - predicted).abs() < 1e-12,
                    "radius {r}, coefficient {k}: error {err} vs predicted {predicted}"
                );
            }
        }
        // And the error therefore falls as the radius does.
        let err_at = |r: f64| {
            ogf_coefficients(
                &|z| Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - z),
                4,
                r,
            )[0] - 1.0
        };
        assert!(err_at(0.5) < err_at(0.9));
        assert!(err_at(0.5) < 2e-5);
    }

    #[test]
    fn egf_to_ogf_multiplies_by_the_factorial() {
        // The EGF of the derangements is exp(-z)/(1-z); its coefficients times
        // k! are the derangement counts themselves.
        let egf: Vec<f64> = (0..10u64)
            .map(|k| {
                crate::discrete::combinatorics::derangements_count(k).to_f64()
                    / BigInt::factorial(k).to_f64()
            })
            .collect();
        let ogf = egf_to_ogf(&egf);
        for (k, &v) in ogf.iter().enumerate() {
            let want = crate::discrete::combinatorics::derangements_count(k as u64).to_f64();
            assert!((v - want).abs() < 1e-6 * want.max(1.0), "term {k}");
        }
        // The EGF of the constant one is exp(z), whose OGF terms are k!.
        let ones = vec![0.0; 0];
        assert!(egf_to_ogf(&ones).is_empty());
        let inv_fact: Vec<f64> = (0..12u64).map(|k| 1.0 / BigInt::factorial(k).to_f64()).collect();
        for (k, &v) in egf_to_ogf(&inv_fact).iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-12, "term {k} is {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Linear recurrences
    // -----------------------------------------------------------------------

    /// The recurrence engine against the closed-form sequences it should
    /// reproduce.
    #[test]
    fn linear_recurrence_reproduces_named_sequences() {
        for n in 0..60u64 {
            assert_eq!(linear_recurrence(&[0, 1], &[1, 1], n), BigInt::fibonacci(n));
            assert_eq!(linear_recurrence(&[2, 1], &[1, 1], n), lucas(n));
            assert_eq!(linear_recurrence(&[0, 1], &[2, 1], n), pell_number(n));
            assert_eq!(linear_recurrence(&[0, 1], &[1, 2], n), jacobsthal(n));
            assert_eq!(
                linear_recurrence(&[0, 0, 1], &[1, 1, 1], n),
                tribonacci(n)
            );
            // Powers of two as a first-order recurrence.
            assert_eq!(linear_recurrence(&[1], &[2], n), big(2).pow(n));
        }
        // Negative coefficients: a(n) = 2a(n-1) - a(n-2) is the arithmetic
        // progression through its first two terms.
        for n in 0..20u64 {
            assert_eq!(
                linear_recurrence(&[5, 8], &[2, -1], n),
                BigInt::from_i64(5 + 3 * n as i64)
            );
        }
    }

    /// The matrix-power version must agree with the direct iteration wherever
    /// both are affordable, and then reach an index the direct one cannot.
    #[test]
    fn matrix_power_recurrence_agrees_and_scales() {
        for &m in &[2u64, 7, 1_000, 1_000_000_007] {
            for n in 0..80u64 {
                let direct = linear_recurrence(&[0, 1], &[1, 1], n)
                    .rem_euclid(&big(m))
                    .to_i64()
                    .unwrap() as u64;
                assert_eq!(
                    linear_recurrence_mod(&[0, 1], &[1, 1], n, m),
                    direct,
                    "F({n}) mod {m}"
                );
                assert_eq!(fibonacci_mod(n, m), direct, "fast doubling F({n}) mod {m}");

                let trib = linear_recurrence(&[0, 0, 1], &[1, 1, 1], n)
                    .rem_euclid(&big(m))
                    .to_i64()
                    .unwrap() as u64;
                assert_eq!(linear_recurrence_mod(&[0, 0, 1], &[1, 1, 1], n, m), trib);
            }
        }
        // An index far out of reach of iteration. The Pisano period modulo
        // 1000 is 1500, so the value at 10^18 must equal the one at
        // 10^18 mod 1500.
        let p = pisano_period(1_000);
        assert_eq!(p, 1_500);
        let huge = 1_000_000_000_000_000_000u64;
        assert_eq!(
            fibonacci_mod(huge, 1_000),
            fibonacci_mod(huge % p, 1_000)
        );
        assert_eq!(
            linear_recurrence_mod(&[0, 1], &[1, 1], huge, 1_000),
            fibonacci_mod(huge, 1_000)
        );
    }

    /// Berlekamp-Massey must recover the generating recurrence, and the
    /// recovered one must actually regenerate the sequence.
    #[test]
    fn berlekamp_massey_recovers_the_generating_recurrence() {
        // The roadmap's case: Fibonacci from eight terms.
        let fib: Vec<Rational> = (0..8u64)
            .map(|i| Rational::from_int(BigInt::fibonacci(i)))
            .collect();
        let c = find_linear_recurrence(&fib).expect("a recurrence exists");
        assert_eq!(c, vec![rat(1), rat(1)], "Fibonacci recurrence not recovered");

        for (name, init, coeffs) in [
            ("Fibonacci", vec![0i64, 1], vec![1i64, 1]),
            ("Lucas", vec![2, 1], vec![1, 1]),
            ("Pell", vec![0, 1], vec![2, 1]),
            ("Jacobsthal", vec![0, 1], vec![1, 2]),
            ("tribonacci", vec![0, 0, 1], vec![1, 1, 1]),
            ("powers of two", vec![1], vec![2]),
            ("arithmetic", vec![5, 8], vec![2, -1]),
            ("order four", vec![1, 2, 3, 4], vec![1, 0, 0, 1]),
        ] {
            let terms: Vec<Rational> = (0..4 * coeffs.len() as u64)
                .map(|i| Rational::from_int(linear_recurrence(&init, &coeffs, i)))
                .collect();
            let found = find_linear_recurrence(&terms)
                .unwrap_or_else(|| panic!("no recurrence found for {name}"));
            assert!(
                found.len() <= coeffs.len(),
                "{name}: found order {} exceeds {}",
                found.len(),
                coeffs.len()
            );
            // Regenerate: whatever order was found, it must reproduce every
            // remaining term.
            for i in found.len()..terms.len() {
                let mut acc = Rational::zero();
                for (j, cj) in found.iter().enumerate() {
                    acc = acc.add(&cj.mul(&terms[i - 1 - j]));
                }
                assert_eq!(acc, terms[i], "{name}: regeneration fails at term {i}");
            }
        }

        // Rational coefficients, not just integer ones: a(n) = a(n-1)/2.
        let halving: Vec<Rational> = (0..10i64)
            .map(|k| Rational::from_i64(1, 1 << k))
            .collect();
        assert_eq!(
            find_linear_recurrence(&halving),
            Some(vec![Rational::from_i64(1, 2)])
        );

        // The all-zero sequence has the empty recurrence, order zero.
        let zeros = vec![Rational::zero(); 6];
        assert_eq!(find_linear_recurrence(&zeros), Some(Vec::new()));

        // Too few terms to pin an order-3 recurrence down: three terms cannot
        // determine it, so None rather than a guess.
        let short: Vec<Rational> = vec![rat(1), rat(2), rat(4)];
        let found = find_linear_recurrence(&short);
        // Three terms do determine an order-1 recurrence (doubling).
        assert_eq!(found, Some(vec![rat(2)]));
        let unpinnable: Vec<Rational> = vec![rat(0), rat(0), rat(1)];
        assert_eq!(
            find_linear_recurrence(&unpinnable),
            None,
            "an order-2 recurrence cannot be determined by three terms"
        );
    }

    /// The GF(2) version against a shift register run forward: the recovered
    /// taps must regenerate the whole output.
    #[test]
    fn berlekamp_massey_gf2_recovers_the_shift_register() {
        for taps in [
            vec![true, false, false, true],       // x^4 + x + 1, period 15
            vec![true, true],                     // x^2 + x + 1, period 3
            vec![false, false, true, false, true], // x^5 + x^3 + 1
            vec![true, true, false, false, true],
        ] {
            let k = taps.len();
            let mut state = vec![true; k];
            let mut out = state.clone();
            for i in k..8 * k {
                let mut bit = false;
                for (j, &t) in taps.iter().enumerate() {
                    bit ^= t && out[i - 1 - j];
                }
                out.push(bit);
                state.push(bit);
            }
            let found = berlekamp_massey_gf2(&out);
            assert!(found.len() <= k, "order {} exceeds {k}", found.len());
            for i in found.len()..out.len() {
                let mut bit = false;
                for (j, &t) in found.iter().enumerate() {
                    bit ^= t && out[i - 1 - j];
                }
                assert_eq!(bit, out[i], "taps {taps:?} fail to regenerate bit {i}");
            }
        }
        // The all-zero stream needs no taps at all.
        assert!(berlekamp_massey_gf2(&[false; 10]).is_empty());
        // A single one at the start needs a register, so the order is not zero.
        let mut impulse = vec![false; 10];
        impulse[0] = true;
        assert!(!berlekamp_massey_gf2(&impulse).is_empty());
    }

    // -----------------------------------------------------------------------
    // Named sequences
    // -----------------------------------------------------------------------

    /// The Pisano period must be a genuine period: the sequence must repeat
    /// with it, and not with anything shorter.
    #[test]
    fn pisano_period_is_the_true_minimal_period() {
        for m in 1..=200u64 {
            let p = pisano_period(m);
            // It is a period.
            for n in 0..3 * p.min(60) {
                assert_eq!(
                    fibonacci_mod(n, m),
                    fibonacci_mod(n + p, m),
                    "not a period for m = {m}"
                );
            }
            // It is minimal: no proper divisor of p is also a period.
            for d in 1..p {
                if !p.is_multiple_of(d) {
                    continue;
                }
                let repeats = (0..p).all(|n| fibonacci_mod(n, m) == fibonacci_mod(n + d, m));
                assert!(!repeats, "m = {m} repeats with {d}, shorter than {p}");
            }
        }
        // Published values.
        assert_eq!(pisano_period(10), 60);
        assert_eq!(pisano_period(2), 3);
        assert_eq!(pisano_period(3), 8);
        assert_eq!(pisano_period(5), 20);
        assert_eq!(pisano_period(1), 1);
        // Multiplicativity over coprime moduli.
        for a in 2..=15u64 {
            for b in 2..=15u64 {
                if gcd_u64(a, b) != 1 {
                    continue;
                }
                let want = pisano_period(a) * pisano_period(b)
                    / gcd_u64(pisano_period(a), pisano_period(b));
                assert_eq!(pisano_period(a * b), want, "pi({a}) and pi({b})");
            }
        }
    }

    /// The companion sequences against the identities that connect them to
    /// the Fibonacci numbers.
    #[test]
    fn companion_sequences_satisfy_their_identities() {
        for n in 1..=50u64 {
            // L(n) = F(n-1) + F(n+1).
            assert_eq!(
                lucas(n),
                BigInt::fibonacci(n - 1).add(&BigInt::fibonacci(n + 1)),
                "Lucas identity at n = {n}"
            );
            // F(2n) = F(n) L(n).
            assert_eq!(
                BigInt::fibonacci(2 * n),
                BigInt::fibonacci(n).mul(&lucas(n)),
                "doubling identity at n = {n}"
            );
        }
        // Jacobsthal has the closed form (2^n - (-1)^n)/3.
        for n in 0..=40u64 {
            let sign = if n.is_multiple_of(2) {
                BigInt::one()
            } else {
                BigInt::one().neg()
            };
            assert_eq!(
                jacobsthal(n),
                big(2).pow(n).sub(&sign).div_rem(&big(3)).0,
                "Jacobsthal closed form at n = {n}"
            );
        }
        // Pell numerators solve x^2 - 2y^2 = +-1 with the half-companion.
        for n in 1..=25u64 {
            let p = pell_number(n);
            let q = pell_number(n - 1);
            // The Pell-Lucas relation: (P(n) + P(n-1))^2 - 2 P(n)^2 = +-1.
            let h = p.add(&q);
            let lhs = h.mul(&h).sub(&big(2).mul(&p.mul(&p)));
            assert!(lhs == BigInt::one() || lhs == BigInt::one().neg(), "n = {n}");
        }
        // Opening terms, as published.
        let trib: Vec<String> = (0..12u64).map(|i| tribonacci(i).to_string()).collect();
        assert_eq!(
            trib,
            ["0", "0", "1", "1", "2", "4", "7", "13", "24", "44", "81", "149"]
        );
    }

    /// Look-and-say: each term must literally describe the previous one.
    #[test]
    fn look_and_say_describes_its_predecessor() {
        assert_eq!(look_and_say("1", 0), "1");
        assert_eq!(look_and_say("1", 1), "11");
        assert_eq!(look_and_say("1", 2), "21");
        assert_eq!(look_and_say("1", 3), "1211");
        assert_eq!(look_and_say("1", 4), "111221");
        assert_eq!(look_and_say("1", 5), "312211");

        // The defining property, checked by decoding rather than by table.
        let mut cur = String::from("1");
        for step in 1..=12 {
            let next = look_and_say(&cur, 1);
            // Decode: read pairs and expand.
            let bytes: Vec<char> = next.chars().collect();
            assert!(bytes.len().is_multiple_of(2), "step {step} is not pairs");
            let mut decoded = String::new();
            for pair in bytes.chunks(2) {
                let count = pair[0].to_digit(10).unwrap();
                for _ in 0..count {
                    decoded.push(pair[1]);
                }
            }
            assert_eq!(decoded, cur, "step {step} does not describe its predecessor");
            cur = next;
        }
    }

    /// Conway's constant, to the precision the ratio actually reaches.
    #[test]
    fn conway_constant_converges_to_its_known_value() {
        const TRUE_VALUE: f64 = 1.303_577_269_034_296;
        let err = |i: usize| (conway_constant_estimate(i) - TRUE_VALUE).abs();
        assert!(err(30) < err(10), "more iterations did not help");
        assert!(err(30) < 1e-3, "estimate at 30 iterations is off by {}", err(30));
        assert!(err(40) < 1e-3, "estimate at 40 iterations is off by {}", err(40));
        // Fewer iterations than the window still returns something usable.
        assert!(conway_constant_estimate(0) > 1.0);
    }

    /// Thue-Morse: the cube-free and self-similar properties, which no other
    /// binary sequence of this density has.
    #[test]
    fn thue_morse_is_self_similar_and_cube_free() {
        let t = thue_morse_sequence(2048);
        // Self-similarity: t(2n) = t(n) and t(2n+1) = 1 - t(n).
        for n in 0..1024 {
            assert_eq!(t[2 * n], t[n]);
            assert_eq!(t[2 * n + 1], !t[n]);
        }
        // The doubling map on blocks: the first 2^k terms, complemented,
        // are the next 2^k terms.
        for k in 0..10usize {
            let block = 1usize << k;
            for i in 0..block {
                assert_eq!(t[block + i], !t[i], "block {k}, offset {i}");
            }
        }
        // Cube-free: no block repeats three times in a row.
        for len in 1..=20usize {
            for start in 0..t.len() - 3 * len {
                let a = &t[start..start + len];
                let b = &t[start + len..start + 2 * len];
                let c = &t[start + 2 * len..start + 3 * len];
                assert!(!(a == b && b == c), "cube of length {len} at {start}");
            }
        }
        assert_eq!(
            thue_morse_sequence(8),
            vec![false, true, true, false, true, false, false, true]
        );
    }

    /// Kolakoski: the sequence must be its own run-length encoding.
    #[test]
    fn kolakoski_is_its_own_run_length_encoding() {
        let k = kolakoski(2000);
        assert!(k.iter().all(|&x| x == 1 || x == 2));
        // Compute the run lengths and compare against the sequence itself.
        let mut runs = Vec::new();
        let mut i = 0usize;
        while i < k.len() {
            let v = k[i];
            let mut len = 0u8;
            while i < k.len() && k[i] == v {
                len += 1;
                i += 1;
            }
            runs.push(len);
        }
        // Drop the last run, which may be truncated by the cut-off.
        runs.pop();
        for (i, &r) in runs.iter().enumerate() {
            assert_eq!(r, k[i], "run {i} has length {r} but the sequence says {}", k[i]);
        }
        assert_eq!(kolakoski(10), vec![1, 2, 2, 1, 1, 2, 1, 2, 2, 1]);
        assert!(kolakoski(0).is_empty());
        // The density of ones tends to 1/2, so a long prefix is close.
        let ones = k.iter().filter(|&&x| x == 1).count() as f64 / k.len() as f64;
        assert!((ones - 0.5).abs() < 0.02, "density of ones is {ones}");
    }

    /// Recaman's sequence by its defining rule, checked step by step.
    #[test]
    fn recaman_follows_its_rule() {
        let r = recaman(200);
        assert_eq!(&r[..10], &[0, 1, 3, 6, 2, 7, 13, 20, 12, 21]);
        let mut seen = std::collections::HashSet::new();
        seen.insert(0i64);
        for k in 1..r.len() {
            let back = r[k - 1] - k as i64;
            let expected = if back > 0 && !seen.contains(&back) {
                back
            } else {
                r[k - 1] + k as i64
            };
            assert_eq!(r[k], expected, "term {k}");
            seen.insert(r[k]);
        }
        assert!(r.iter().all(|&x| x >= 0));
    }

    /// The Ulam sequence by its definition: every term after the seeds has
    /// exactly one representation as a sum of two distinct earlier terms, and
    /// nothing skipped in between has one.
    #[test]
    fn ulam_terms_have_exactly_one_representation() {
        let u = ulam_sequence(1, 2, 30);
        assert_eq!(&u[..12], &[1, 2, 3, 4, 6, 8, 11, 13, 16, 18, 26, 28]);

        let ways = |seq: &[u64], target: u64| -> usize {
            let mut n = 0;
            for i in 0..seq.len() {
                for j in i + 1..seq.len() {
                    if seq[i] + seq[j] == target {
                        n += 1;
                    }
                }
            }
            n
        };
        for k in 2..u.len() {
            let prefix = &u[..k];
            assert_eq!(ways(prefix, u[k]), 1, "term {k} = {} is not unique", u[k]);
            // Nothing between the previous term and this one qualifies.
            for skipped in u[k - 1] + 1..u[k] {
                assert_ne!(ways(prefix, skipped), 1, "{skipped} was wrongly skipped");
            }
        }
        // The (1, 3) sequence is a different one.
        assert_eq!(&ulam_sequence(1, 3, 8), &[1, 3, 4, 5, 6, 8, 10, 12]);
    }

    /// Aliquot sequences: perfect numbers are fixed points, amicable pairs are
    /// two-cycles, and 138 is the classic long ascent.
    #[test]
    fn aliquot_sequence_finds_the_known_cycles() {
        // Perfect: s(6) = 6.
        assert_eq!(aliquot_sequence(6, 10), vec![6, 6]);
        assert_eq!(aliquot_sequence(28, 10), vec![28, 28]);
        // Amicable: 220 and 284.
        assert_eq!(aliquot_sequence(220, 10), vec![220, 284, 220]);
        // Prime: falls straight to 1 then 0.
        assert_eq!(aliquot_sequence(13, 10), vec![13, 1, 0]);
        // Sociable: the 5-cycle starting at 12496.
        let s = aliquot_sequence(12_496, 10);
        assert_eq!(s, vec![12_496, 14_288, 15_472, 14_536, 14_264, 12_496]);
        // Every step is the sum of proper divisors of the previous.
        for w in s.windows(2) {
            if w[0] == 0 {
                break;
            }
            assert_eq!(w[1], divisor_sum(w[0]) - w[0]);
        }
        // The step budget is respected.
        assert!(aliquot_sequence(138, 5).len() <= 6);
    }

    /// Ackermann against the recursive definition wherever the recursion is
    /// affordable, so the closed forms are checked rather than assumed.
    #[test]
    fn ackermann_matches_the_recursive_definition() {
        fn slow(m: u64, n: u64) -> u64 {
            if m == 0 {
                n + 1
            } else if n == 0 {
                slow(m - 1, 1)
            } else {
                slow(m - 1, slow(m, n - 1))
            }
        }
        for m in 0..=3u64 {
            let cap = if m == 3 { 6 } else { 10 };
            for n in 0..=cap {
                assert_eq!(
                    ackermann_small(m, n),
                    Some(big(slow(m, n))),
                    "A({m}, {n})"
                );
            }
        }
        // The published boundary values.
        assert_eq!(ackermann_small(4, 0), Some(big(13)));
        assert_eq!(ackermann_small(4, 1), Some(big(65_533)));
        assert_eq!(ackermann_small(5, 0), Some(big(65_533)));
        // A(4, 2) has 19729 digits; the tower cap refuses it rather than
        // trying to build it.
        assert_eq!(ackermann_small(4, 2), None);
        assert_eq!(ackermann_small(5, 1), None);
        // A(3, n) is exactly 2^(n+3) - 3 for a large n the recursion cannot
        // reach.
        assert_eq!(
            ackermann_small(3, 100),
            Some(BigInt::one().shl(103).sub(&big(3)))
        );
    }

    // -----------------------------------------------------------------------
    // Identification
    // -----------------------------------------------------------------------

    /// Identification must name the right family, and must not name a family
    /// the terms do not match.
    #[test]
    fn sequence_identify_names_the_right_families() {
        let has = |terms: &[i64], name: &str| sequence_identify(terms).iter().any(|s| s == name);

        assert!(has(&[0, 1, 1, 2, 3, 5, 8, 13, 21], "Fibonacci numbers"));
        assert!(has(&[2, 1, 3, 4, 7, 11, 18, 29], "Lucas numbers"));
        assert!(has(&[1, 1, 2, 5, 14, 42, 132, 429], "Catalan numbers"));
        assert!(has(&[1, 1, 2, 5, 15, 52, 203, 877], "Bell numbers"));
        assert!(has(&[1, 1, 2, 6, 24, 120, 720], "factorials"));
        assert!(has(&[1, 2, 4, 8, 16, 32, 64], "powers of two"));
        assert!(has(&[0, 1, 4, 9, 16, 25, 36], "squares"));
        assert!(has(&[2, 3, 5, 7, 11, 13, 17, 19], "primes"));
        assert!(has(&[0, 1, 3, 6, 10, 15, 21], "triangular numbers"));
        assert!(has(&[1, 1, 2, 3, 5, 7, 11, 15, 22], "partition counts"));
        assert!(has(&[1, 0, 1, 2, 9, 44, 265], "derangement counts"));
        assert!(has(&[1, 2, 6, 20, 70, 252], "central binomial coefficients"));
        assert!(has(&[0, 1, 2, 5, 12, 29, 70], "Pell numbers"));
        assert!(has(&[0, 1, 1, 3, 5, 11, 21, 43], "Jacobsthal numbers"));
        assert!(has(&[0, 1, 1, 0, 1, 0, 0, 1], "Thue-Morse sequence"));
        assert!(has(&[0, 1, 3, 6, 2, 7, 13, 20], "Recaman's sequence"));
        assert!(has(&[0, 1, 3, 7, 15, 31, 63], "Mersenne numbers"));

        // Negative controls: a family must not be named for terms that leave
        // it, however long the agreeing prefix.
        assert!(!has(&[0, 1, 1, 2, 3, 5, 8, 14], "Fibonacci numbers"));
        assert!(!has(&[1, 1, 2, 5, 14, 42, 133], "Catalan numbers"));
        assert!(!has(&[2, 3, 5, 7, 11, 13, 17, 21], "primes"));
        assert!(sequence_identify(&[]).is_empty());

        // The recurrence fallback covers a family with no name here.
        let names = sequence_identify(&[3, 5, 13, 31, 75, 181, 437]);
        assert!(
            names.iter().any(|s| s.starts_with("linear recurrence")),
            "no recurrence reported for a linear sequence: {names:?}"
        );

        // Short prefixes are genuinely ambiguous, and the result says so.
        let ambiguous = sequence_identify(&[1, 1, 2]);
        assert!(
            ambiguous.len() > 1,
            "1, 1, 2 should match several families, got {ambiguous:?}"
        );
    }

    /// A reported linear recurrence must actually regenerate the input, which
    /// is the only claim the string makes.
    #[test]
    fn reported_recurrences_regenerate_their_input() {
        for terms in [
            vec![0i64, 1, 1, 2, 3, 5, 8, 13],
            vec![1, 2, 4, 8, 16, 32],
            vec![3, 5, 13, 31, 75, 181, 437],
            vec![5, 8, 11, 14, 17, 20],
        ] {
            let rationals: Vec<Rational> = terms
                .iter()
                .map(|&x| Rational::from_int(BigInt::from_i64(x)))
                .collect();
            let c = find_linear_recurrence(&rationals)
                .unwrap_or_else(|| panic!("no recurrence for {terms:?}"));
            for i in c.len()..terms.len() {
                let mut acc = Rational::zero();
                for (j, cj) in c.iter().enumerate() {
                    acc = acc.add(&cj.mul(&rationals[i - 1 - j]));
                }
                assert_eq!(acc, rationals[i], "{terms:?} fails at term {i}");
            }
            assert!(
                sequence_identify(&terms)
                    .iter()
                    .any(|s| s.starts_with("linear recurrence")),
                "{terms:?} has a recurrence but none was reported"
            );
        }
    }
}
