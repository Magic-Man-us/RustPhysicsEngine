//! The arithmetic underneath public-key cryptography, for study rather than
//! for use.
//!
//! **None of this is safe to deploy.** Every routine here branches and
//! indexes on secret values, so the time it takes and the memory it touches
//! leak what it is working on; a modular exponentiation that skips a squaring
//! when a bit is zero tells anyone timing it how many bits are set. Real
//! implementations are written to take the same time and the same path
//! whatever the key, use blinding to break the correlation between input and
//! timing, and are audited for the dozen further side channels that remain.
//! Nothing here does any of that, and the key sizes the tests use are small
//! enough to factor over lunch.
//!
//! What it is for is seeing why the constructions work. RSA rests on the fact
//! that exponentiating by `e` and then by `d` returns you to where you
//! started whenever `ed = 1` modulo the group order -- so anyone who can
//! compute the group order can find `d`, and the security assumption is
//! exactly that factoring `n` is hard. Diffie-Hellman and elliptic curve
//! Diffie-Hellman rest on the same shape in a different group. Shamir's
//! scheme rests on a polynomial of degree `k - 1` being determined by `k`
//! points and by no fewer. Each of those is a theorem, and the tests here
//! check the theorem rather than the ciphertext.

use crate::exact::BigInt;
use crate::monte_carlo::Rng;

// ---------------------------------------------------------------------------
// RSA
// ---------------------------------------------------------------------------

/// Generates an RSA modulus and exponent pair: `(n, e, d)`.
///
/// Two primes of about `bits / 2` each are drawn, `n` is their product, and
/// `d` inverts `e` modulo the Carmichael function of `n` -- the exponent of
/// the multiplicative group, which is the least value that works and so gives
/// the smallest `d`. The public exponent is 65537, whose binary form has two
/// set bits and therefore encrypts in seventeen squarings.
///
/// # Panics
/// Panics unless `bits` is between 16 and 2048. Anything in that range is far
/// too small to protect anything.
#[must_use]
pub fn rsa_keygen(bits: usize, rng: &mut Rng) -> (BigInt, BigInt, BigInt) {
    assert!((16..=2048).contains(&bits), "bits must lie between 16 and 2048");
    let half = bits / 2;
    let e = BigInt::from_u64(65537);
    loop {
        let p = crate::discrete::primes::random_prime(half, rng);
        let q = crate::discrete::primes::random_prime(bits - half, rng);
        if p == q {
            continue;
        }
        let one = BigInt::one();
        let pm = p.sub(&one);
        let qm = q.sub(&one);
        // The Carmichael lambda: the least exponent that kills the whole
        // group, which is the lowest common multiple rather than the product.
        let lambda = pm.lcm(&qm);
        let Some(d) = e.mod_inverse(&lambda) else { continue };
        let n = p.mul(&q);
        if n.bits() < bits {
            continue;
        }
        return (n, e, d);
    }
}

/// Generates a key and keeps the primes, which the Chinese remainder form of
/// decryption needs.
///
/// # Panics
/// Panics unless `bits` is between 16 and 2048.
#[must_use]
pub fn rsa_keygen_with_primes(
    bits: usize,
    rng: &mut Rng,
) -> (BigInt, BigInt, BigInt, BigInt, BigInt) {
    assert!((16..=2048).contains(&bits), "bits must lie between 16 and 2048");
    let half = bits / 2;
    let e = BigInt::from_u64(65537);
    loop {
        let p = crate::discrete::primes::random_prime(half, rng);
        let q = crate::discrete::primes::random_prime(bits - half, rng);
        if p == q {
            continue;
        }
        let one = BigInt::one();
        let lambda = p.sub(&one).lcm(&q.sub(&one));
        let Some(d) = e.mod_inverse(&lambda) else { continue };
        let n = p.mul(&q);
        if n.bits() < bits {
            continue;
        }
        return (n, e, d, p, q);
    }
}

/// Textbook RSA encryption: `m^e` modulo `n`.
///
/// Deterministic, and therefore not a secure encryption scheme on its own --
/// the same message always gives the same ciphertext, so an attacker who can
/// guess the plaintext can confirm the guess. Real use pads the message with
/// randomness first.
#[must_use]
pub fn rsa_encrypt(m: &BigInt, e: &BigInt, n: &BigInt) -> BigInt {
    m.mod_pow(e, n)
}

/// Textbook RSA decryption: `c^d` modulo `n`.
#[must_use]
pub fn rsa_decrypt(c: &BigInt, d: &BigInt, n: &BigInt) -> BigInt {
    c.mod_pow(d, n)
}

/// Decryption through the Chinese remainder theorem, given the two primes.
///
/// Working modulo `p` and `q` separately and recombining costs about a
/// quarter of the work, since modular exponentiation is cubic in the operand
/// size and the operands are half as long. Every real implementation does
/// this, which is also why a fault during one of the two halves famously
/// reveals the factorisation.
///
/// # Panics
/// Panics if `p` and `q` are not coprime, so that the recombination has no
/// inverse.
#[must_use]
pub fn rsa_crt_decrypt(c: &BigInt, d: &BigInt, p: &BigInt, q: &BigInt) -> BigInt {
    let one = BigInt::one();
    let dp = d.rem_euclid(&p.sub(&one));
    let dq = d.rem_euclid(&q.sub(&one));
    let mp = c.rem_euclid(p).mod_pow(&dp, p);
    let mq = c.rem_euclid(q).mod_pow(&dq, q);
    let qinv = q.mod_inverse(p).expect("the primes must be coprime");
    // Garner's recombination: start from the residue modulo q and add the
    // multiple of q that fixes the residue modulo p.
    let h = qinv.mul(&mp.sub(&mq)).rem_euclid(p);
    mq.add(&h.mul(q))
}

// ---------------------------------------------------------------------------
// Diffie-Hellman
// ---------------------------------------------------------------------------

/// A Diffie-Hellman exchange in full: both parties' key pairs and the shared
/// secret they arrive at.
///
/// Returns `((a, A), (b, B), s)` where `A = g^a`, `B = g^b` and
/// `s = B^a = A^b`, all modulo `p`. The exchange works because
/// exponentiation commutes; it is secure only if recovering `a` from `g^a` is
/// hard, which needs `p` to be a large safe prime and `g` to generate a large
/// subgroup. Neither is checked here.
///
/// # Panics
/// Panics unless `p` is at least three.
#[must_use]
pub fn diffie_hellman_demo(
    p: &BigInt,
    g: &BigInt,
    rng: &mut Rng,
) -> ((BigInt, BigInt), (BigInt, BigInt), BigInt) {
    assert!(p.cmp_abs(&BigInt::from_u64(3)) != std::cmp::Ordering::Less, "the modulus is too small");
    let two = BigInt::from_u64(2);
    let bound = p.sub(&two);
    let a = BigInt::random_below(&bound, rng).add(&BigInt::one());
    let b = BigInt::random_below(&bound, rng).add(&BigInt::one());
    let big_a = g.mod_pow(&a, p);
    let big_b = g.mod_pow(&b, p);
    let s = big_b.mod_pow(&a, p);
    ((a, big_a), (b, big_b), s)
}

// ---------------------------------------------------------------------------
// Elliptic curves
// ---------------------------------------------------------------------------

/// A point on a short Weierstrass curve, or the point at infinity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcPoint {
    /// The identity of the group law.
    Infinity,
    /// An affine point.
    Affine(BigInt, BigInt),
}

/// A short Weierstrass curve `y^2 = x^3 + a x + b` over the prime field
/// `F_p`.
///
/// The points form a group under the chord-and-tangent construction: three
/// points on a line sum to the identity, so adding two points means drawing
/// the line through them, finding the third intersection, and reflecting it.
/// That the construction is associative is the one non-obvious fact, and it
/// is what makes the whole subject possible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcCurve {
    /// The linear coefficient.
    pub a: BigInt,
    /// The constant coefficient.
    pub b: BigInt,
    /// The field's characteristic.
    pub p: BigInt,
}

impl EcCurve {
    /// The curve with the given coefficients over `F_p`.
    ///
    /// # Panics
    /// Panics if the discriminant `4a^3 + 27b^2` vanishes, which means the
    /// curve is singular and its points do not form a group.
    #[must_use]
    pub fn new(a: BigInt, b: BigInt, p: BigInt) -> Self {
        let four = BigInt::from_u64(4);
        let twenty_seven = BigInt::from_u64(27);
        let disc = four
            .mul(&a.mod_pow(&BigInt::from_u64(3), &p))
            .add(&twenty_seven.mul(&b.mul(&b)))
            .rem_euclid(&p);
        assert!(!disc.is_zero(), "the curve is singular");
        EcCurve { a, b, p }
    }

    /// Whether a point satisfies the curve equation.
    #[must_use]
    pub fn is_on_curve(&self, pt: &EcPoint) -> bool {
        match pt {
            EcPoint::Infinity => true,
            EcPoint::Affine(x, y) => {
                let lhs = y.mul(y).rem_euclid(&self.p);
                let rhs = x
                    .mul(x)
                    .mul(x)
                    .add(&self.a.mul(x))
                    .add(&self.b)
                    .rem_euclid(&self.p);
                lhs == rhs
            }
        }
    }

    /// The additive inverse: the reflection in the `x` axis.
    #[must_use]
    pub fn negate(&self, pt: &EcPoint) -> EcPoint {
        match pt {
            EcPoint::Infinity => EcPoint::Infinity,
            EcPoint::Affine(x, y) => {
                let ny = self.p.sub(&y.rem_euclid(&self.p)).rem_euclid(&self.p);
                EcPoint::Affine(x.clone(), ny)
            }
        }
    }

    /// The group law.
    ///
    /// # Panics
    /// Panics if a required inverse does not exist, which cannot happen over
    /// a prime field with a non-singular curve.
    #[must_use]
    pub fn add(&self, p1: &EcPoint, p2: &EcPoint) -> EcPoint {
        match (p1, p2) {
            (EcPoint::Infinity, q) | (q, EcPoint::Infinity) => q.clone(),
            (EcPoint::Affine(x1, y1), EcPoint::Affine(x2, y2)) => {
                let (x1, y1) = (x1.rem_euclid(&self.p), y1.rem_euclid(&self.p));
                let (x2, y2) = (x2.rem_euclid(&self.p), y2.rem_euclid(&self.p));
                if x1 == x2 {
                    // Either the points are reflections, and the line through
                    // them is vertical, or they coincide and the chord
                    // becomes the tangent.
                    if y1 == y2 && !y1.is_zero() {
                        return self.double(p1);
                    }
                    return EcPoint::Infinity;
                }
                let num = y2.sub(&y1).rem_euclid(&self.p);
                let den = x2.sub(&x1).rem_euclid(&self.p);
                let slope = num
                    .mul(&den.mod_inverse(&self.p).expect("a non-zero residue is invertible"))
                    .rem_euclid(&self.p);
                self.third_intersection(&slope, &x1, &y1, &x2)
            }
        }
    }

    /// Doubling, which the chord construction degenerates to when the two
    /// points coincide and the line becomes the tangent.
    ///
    /// # Panics
    /// Panics if a required inverse does not exist.
    #[must_use]
    pub fn double(&self, pt: &EcPoint) -> EcPoint {
        match pt {
            EcPoint::Infinity => EcPoint::Infinity,
            EcPoint::Affine(x, y) => {
                let (x, y) = (x.rem_euclid(&self.p), y.rem_euclid(&self.p));
                if y.is_zero() {
                    // The tangent is vertical, so the point is its own
                    // inverse and doubling reaches infinity.
                    return EcPoint::Infinity;
                }
                let three = BigInt::from_u64(3);
                let two = BigInt::from_u64(2);
                let num = three.mul(&x.mul(&x)).add(&self.a).rem_euclid(&self.p);
                let den = two.mul(&y).rem_euclid(&self.p);
                let slope = num
                    .mul(&den.mod_inverse(&self.p).expect("a non-zero residue is invertible"))
                    .rem_euclid(&self.p);
                self.third_intersection(&slope, &x, &y, &x)
            }
        }
    }

    /// The third intersection of a line of the given slope, reflected.
    fn third_intersection(&self, slope: &BigInt, x1: &BigInt, y1: &BigInt, x2: &BigInt) -> EcPoint {
        let x3 = slope.mul(slope).sub(x1).sub(x2).rem_euclid(&self.p);
        let y3 = slope.mul(&x1.sub(&x3)).sub(y1).rem_euclid(&self.p);
        EcPoint::Affine(x3, y3)
    }

    /// Repeated addition, by the double-and-add ladder.
    ///
    /// The exponentiation of the additive group, and the operation whose
    /// difficulty to invert -- recovering `k` from `k P` -- everything
    /// elliptic-curve rests on.
    #[must_use]
    pub fn scalar_mul(&self, k: &BigInt, pt: &EcPoint) -> EcPoint {
        if k.is_zero() {
            return EcPoint::Infinity;
        }
        let (k, pt) = if k.is_negative() {
            (k.neg(), self.negate(pt))
        } else {
            (k.clone(), pt.clone())
        };
        let mut acc = EcPoint::Infinity;
        let mut base = pt;
        for i in 0..k.bits() {
            if k.bit(i) {
                acc = self.add(&acc, &base);
            }
            base = self.double(&base);
        }
        acc
    }

    /// Every affine point, for a curve small enough to enumerate.
    ///
    /// # Panics
    /// Panics if the field has more than a million elements.
    #[must_use]
    pub fn all_points(&self) -> Vec<EcPoint> {
        let p = self.p.to_i64().expect("a small prime") as u64;
        assert!(p <= 1_000_000, "enumeration is for small curves");
        // Which residues are squares, and one square root of each.
        let mut root: Vec<Option<u64>> = vec![None; p as usize];
        for y in 0..p {
            let sq = (u128::from(y) * u128::from(y) % u128::from(p)) as u64;
            if root[sq as usize].is_none() {
                root[sq as usize] = Some(y);
            }
        }
        let a = self.a.rem_euclid(&self.p).to_i64().expect("small") as u64;
        let b = self.b.rem_euclid(&self.p).to_i64().expect("small") as u64;
        let mut out = Vec::new();
        for x in 0..p {
            let x2 = u128::from(x) * u128::from(x) % u128::from(p);
            let rhs = ((x2 * u128::from(x) + u128::from(a) * u128::from(x) + u128::from(b))
                % u128::from(p)) as u64;
            if let Some(y) = root[rhs as usize] {
                out.push(EcPoint::Affine(BigInt::from_u64(x), BigInt::from_u64(y)));
                if y != 0 {
                    out.push(EcPoint::Affine(BigInt::from_u64(x), BigInt::from_u64(p - y)));
                }
            }
        }
        out
    }

    /// The group order, including the point at infinity, by enumeration.
    ///
    /// # Panics
    /// Panics if the field has more than a million elements.
    #[must_use]
    pub fn order_naive_small(&self) -> u64 {
        self.all_points().len() as u64 + 1
    }

    /// The order of a single point: the least positive `k` with `k P` at
    /// infinity.
    ///
    /// # Panics
    /// Panics if the field has more than a million elements, or the point is
    /// not on the curve.
    #[must_use]
    pub fn point_order_small(&self, pt: &EcPoint) -> u64 {
        assert!(self.is_on_curve(pt), "the point is not on the curve");
        if *pt == EcPoint::Infinity {
            return 1;
        }
        let bound = self.order_naive_small();
        let mut acc = pt.clone();
        for k in 1..=bound {
            if acc == EcPoint::Infinity {
                return k;
            }
            acc = self.add(&acc, pt);
        }
        unreachable!("Lagrange bounds the order by the group's")
    }

    /// A uniformly chosen affine point.
    ///
    /// # Panics
    /// Panics if the field has more than a million elements, or the curve has
    /// no affine points.
    #[must_use]
    pub fn random_point(&self, rng: &mut Rng) -> EcPoint {
        let pts = self.all_points();
        assert!(!pts.is_empty(), "the curve has no affine points");
        let i = ((u128::from(rng.next_u64()) * pts.len() as u128) >> 64) as usize;
        pts[i].clone()
    }

    /// The secp256k1 curve, `y^2 = x^3 + 7`, used by Bitcoin.
    ///
    /// # Panics
    /// Panics only if the built-in constants fail to parse.
    #[must_use]
    pub fn secp256k1() -> Self {
        let p = BigInt::from_str_radix(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
            16,
        )
        .expect("a valid constant");
        EcCurve { a: BigInt::zero(), b: BigInt::from_u64(7), p }
    }

    /// The generator of secp256k1, and its order.
    ///
    /// # Panics
    /// Panics only if the built-in constants fail to parse.
    #[must_use]
    pub fn secp256k1_generator() -> (EcPoint, BigInt) {
        let gx = BigInt::from_str_radix(
            "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
            16,
        )
        .expect("a valid constant");
        let gy = BigInt::from_str_radix(
            "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
            16,
        )
        .expect("a valid constant");
        let n = BigInt::from_str_radix(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            16,
        )
        .expect("a valid constant");
        (EcPoint::Affine(gx, gy), n)
    }

    /// The NIST P-256 curve.
    ///
    /// # Panics
    /// Panics only if the built-in constants fail to parse.
    #[must_use]
    pub fn p256() -> Self {
        let p = BigInt::from_str_radix(
            "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF",
            16,
        )
        .expect("a valid constant");
        let a = p.sub(&BigInt::from_u64(3));
        let b = BigInt::from_str_radix(
            "5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B",
            16,
        )
        .expect("a valid constant");
        EcCurve { a, b, p }
    }

    /// The generator of P-256, and its order.
    ///
    /// # Panics
    /// Panics only if the built-in constants fail to parse.
    #[must_use]
    pub fn p256_generator() -> (EcPoint, BigInt) {
        let gx = BigInt::from_str_radix(
            "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
            16,
        )
        .expect("a valid constant");
        let gy = BigInt::from_str_radix(
            "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            16,
        )
        .expect("a valid constant");
        let n = BigInt::from_str_radix(
            "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
            16,
        )
        .expect("a valid constant");
        (EcPoint::Affine(gx, gy), n)
    }
}

/// An elliptic curve Diffie-Hellman exchange in full.
///
/// Returns `((a, aG), (b, bG), s)`. The same construction as the
/// multiplicative version, in a group where the best known attack is
/// square-root time rather than sub-exponential -- which is why a 256-bit
/// curve stands against a 3072-bit modulus.
///
/// # Panics
/// Panics if the base point is not on the curve.
#[must_use]
pub fn ecdh_demo(
    curve: &EcCurve,
    g: &EcPoint,
    order: &BigInt,
    rng: &mut Rng,
) -> ((BigInt, EcPoint), (BigInt, EcPoint), EcPoint) {
    assert!(curve.is_on_curve(g), "the base point is not on the curve");
    let one = BigInt::one();
    let a = BigInt::random_below(&order.sub(&one), rng).add(&one);
    let b = BigInt::random_below(&order.sub(&one), rng).add(&one);
    let big_a = curve.scalar_mul(&a, g);
    let big_b = curve.scalar_mul(&b, g);
    let s = curve.scalar_mul(&a, &big_b);
    ((a, big_a), (b, big_b), s)
}

/// The number of points on a small curve, including infinity.
///
/// # Panics
/// Panics if the field has more than a million elements.
#[must_use]
pub fn ec_count_points_small(curve: &EcCurve) -> u64 {
    curve.order_naive_small()
}

/// Whether a point count satisfies Hasse's theorem.
///
/// The count lies within `2 sqrt(p)` of `p + 1`. That is a remarkably tight
/// bound -- the group is always about as large as the field, never a constant
/// factor away -- and it is what makes a curve's security predictable from
/// its field size alone.
#[must_use]
pub fn hasse_bound_check(count: u64, p: u64) -> bool {
    let expected = p as f64 + 1.0;
    (count as f64 - expected).abs() <= 2.0 * (p as f64).sqrt() + 1e-9
}

// ---------------------------------------------------------------------------
// Secret sharing
// ---------------------------------------------------------------------------

/// Splits a secret into `n` shares of which any `k` suffice.
///
/// The secret is the constant term of a random polynomial of degree `k - 1`
/// over `F_prime`, and a share is that polynomial's value at a non-zero
/// point. Any `k` points determine the polynomial by interpolation, and any
/// `k - 1` leave the constant term uniformly distributed -- so fewer than `k`
/// shares give not merely a hard problem but no information at all. That is
/// what makes the scheme *perfect*, and it is rare.
///
/// # Panics
/// Panics unless `1 <= k <= n`, `n` is below the prime, and the secret is a
/// non-negative residue below it.
#[must_use]
pub fn shamir_split(
    secret: &BigInt,
    k: usize,
    n: usize,
    prime: &BigInt,
    rng: &mut Rng,
) -> Vec<(u64, BigInt)> {
    assert!(k >= 1 && k <= n, "need 1 <= k <= n");
    assert!(!secret.is_negative(), "the secret must be a non-negative residue");
    assert!(secret.cmp_abs(prime) == std::cmp::Ordering::Less, "the secret must be below the prime");
    assert!(
        BigInt::from_u64(n as u64).cmp_abs(prime) == std::cmp::Ordering::Less,
        "there are more shares than the field has non-zero points"
    );
    let mut coeffs = vec![secret.clone()];
    for _ in 1..k {
        coeffs.push(BigInt::random_below(prime, rng));
    }
    (1..=n as u64)
        .map(|x| {
            let xb = BigInt::from_u64(x);
            // Horner, from the top coefficient down.
            let y = coeffs
                .iter()
                .rev()
                .fold(BigInt::zero(), |acc, c| acc.mul(&xb).add(c).rem_euclid(prime));
            (x, y)
        })
        .collect()
}

/// Recovers the secret from any `k` shares by Lagrange interpolation at zero.
///
/// # Panics
/// Panics on an empty share list, on a repeated abscissa, or if the modulus
/// is not prime enough for the required inverses to exist.
#[must_use]
pub fn shamir_reconstruct(shares: &[(u64, BigInt)], prime: &BigInt) -> BigInt {
    assert!(!shares.is_empty(), "reconstruction needs at least one share");
    let mut seen = std::collections::BTreeSet::new();
    for &(x, _) in shares {
        assert!(seen.insert(x), "a share is repeated");
    }
    let mut acc = BigInt::zero();
    for (i, (xi, yi)) in shares.iter().enumerate() {
        let mut num = BigInt::one();
        let mut den = BigInt::one();
        for (j, (xj, _)) in shares.iter().enumerate() {
            if i == j {
                continue;
            }
            // The basis polynomial evaluated at zero: a product of
            // (0 - x_j) / (x_i - x_j). The numerator's minus sign matters --
            // dropping it flips the answer's sign whenever the threshold is
            // even, so a two-of-n split reconstructs the negation.
            num = num.mul(&BigInt::from_u64(*xj).neg()).rem_euclid(prime);
            let d = BigInt::from_i64(*xi as i64 - *xj as i64).rem_euclid(prime);
            den = den.mul(&d).rem_euclid(prime);
        }
        let inv = den.mod_inverse(prime).expect("the modulus must be prime");
        acc = acc.add(&yi.mul(&num).mul(&inv)).rem_euclid(prime);
    }
    acc
}

// ---------------------------------------------------------------------------
// Stream ciphers and keystreams
// ---------------------------------------------------------------------------

/// Exclusive-or of the data with a repeating key.
///
/// With a key as long as the message, drawn uniformly and never reused, this
/// is the one cipher with a proof of perfect secrecy: the ciphertext is
/// independent of the plaintext, so an adversary with unlimited computation
/// learns nothing. With a short key repeated, it is a Vigenere cipher and
/// [`vigenere_break`] undoes it. The gap between those two is entirely the
/// key.
///
/// # Panics
/// Panics on an empty key.
#[must_use]
pub fn one_time_pad(data: &[u8], key: &[u8]) -> Vec<u8> {
    assert!(!key.is_empty(), "the key must not be empty");
    data.iter().enumerate().map(|(i, &b)| b ^ key[i % key.len()]).collect()
}

/// A Fibonacci linear feedback shift register: `n` output bits from a state
/// and a tap mask.
///
/// The new bit is the parity of the tapped positions, and the register shifts
/// right. The output is a linear recurrence over `GF(2)`, which is what makes
/// it fast, and also what makes it hopeless as a cipher on its own:
/// [`berlekamp_massey_attack`] recovers the whole register from twice its
/// length in output.
///
/// Tap bit zero, or the step map is not reversible and the register cannot
/// reach every state -- see [`lfsr_period`].
///
/// # Panics
/// Panics on a zero tap mask.
#[must_use]
pub fn lfsr(taps: u64, state: u64, n: usize) -> Vec<bool> {
    assert!(taps != 0, "a register with no taps produces nothing");
    let mut s = state;
    (0..n)
        .map(|_| {
            let out = s & 1 == 1;
            let feedback = (s & taps).count_ones() % 2;
            s = (s >> 1) | (u64::from(feedback) << 63);
            out
        })
        .collect()
}

/// The period of a shift register of the given width, by running it until it
/// repeats.
///
/// A width-`w` register has at most `2^w - 1` states before it must repeat,
/// and reaches that only for a *primitive* tap polynomial. The all-zero state
/// is absorbing, which is why the maximum is one short of the state count.
///
/// The step map is a bijection only when bit zero is tapped: without it, the
/// outgoing bit does not influence the feedback, two states share an image,
/// and the register runs into a cycle it can never leave and never started
/// on. Returns zero in that case, meaning the register never comes back.
#[must_use]
pub fn lfsr_period(taps: u64, width: u32) -> u64 {
    assert!((1..=24).contains(&width), "the width must lie between one and 24");
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let taps = taps & mask;
    let start = 1u64;
    let mut s = start;
    for k in 1..=(1u64 << width) {
        let feedback = (s & taps).count_ones() % 2;
        s = ((s >> 1) | (u64::from(feedback) << (width - 1))) & mask;
        if s == start {
            return k;
        }
    }
    0
}

/// Recovers the shortest linear recurrence a bit stream satisfies, as
/// `(length, taps)`.
///
/// The Berlekamp-Massey algorithm, over `GF(2)`. Given `2L` bits of output
/// from a register of length `L` it returns that register, which is why a
/// bare shift register is not a cipher: the keystream reveals the key
/// generator in time linear in its size.
#[must_use]
pub fn berlekamp_massey_attack(stream: &[bool]) -> (u64, u64) {
    let n = stream.len();
    let mut c = vec![false; n + 1];
    let mut b = vec![false; n + 1];
    c[0] = true;
    b[0] = true;
    let mut l = 0usize;
    let mut m = 1usize;
    for i in 0..n {
        // The discrepancy between the recurrence's prediction and the bit.
        let mut d = stream[i];
        for j in 1..=l {
            d ^= c[j] & stream[i - j];
        }
        if !d {
            m += 1;
        } else if 2 * l <= i {
            let t = c.clone();
            for j in 0..=n - m {
                c[j + m] ^= b[j];
            }
            l = i + 1 - l;
            b = t;
            m = 1;
        } else {
            for j in 0..=n - m {
                c[j + m] ^= b[j];
            }
            m += 1;
        }
    }
    let mut taps = 0u64;
    for j in 1..=l.min(64) {
        if c[j] {
            taps |= 1 << (j - 1);
        }
    }
    (l as u64, taps)
}

/// How close a hash comes to flipping half its output bits when one input bit
/// changes.
///
/// Returns the mean fraction of output bits that flip. A good hash sits at a
/// half: every output bit should be an unbiased, independent-looking function
/// of every input bit, so that no partial information about the input
/// survives. A value far from a half is a structural weakness a distinguisher
/// can be built from.
///
/// # Panics
/// Panics if `trials` is zero.
pub fn hash_avalanche_test(h: &dyn Fn(&[u8]) -> u64, trials: usize, rng: &mut Rng) -> f64 {
    assert!(trials > 0, "run at least one trial");
    let mut total = 0f64;
    let mut count = 0usize;
    for _ in 0..trials {
        let len = 1 + ((u128::from(rng.next_u64()) * 16) >> 64) as usize;
        let data: Vec<u8> = (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        let base = h(&data);
        for byte in 0..len {
            for bit in 0..8 {
                let mut flipped = data.clone();
                flipped[byte] ^= 1 << bit;
                total += f64::from((base ^ h(&flipped)).count_ones()) / 64.0;
                count += 1;
            }
        }
    }
    total / count as f64
}

/// The number of samples at which a collision becomes likely for an output of
/// `n_bits`.
///
/// About `2^(n/2)`, up to a constant: with `k` samples there are about
/// `k^2 / 2` pairs and each collides with probability `2^-n`, so the count of
/// collisions reaches one near the square root. It is why a 128-bit hash
/// offers 64 bits of collision resistance, not 128.
#[must_use]
pub fn birthday_bound(n_bits: u32) -> f64 {
    (PI_OVER_2 * (2.0f64).powi(n_bits as i32)).sqrt()
}

const PI_OVER_2: f64 = std::f64::consts::PI / 2.0;

// ---------------------------------------------------------------------------
// Classical cipher analysis
// ---------------------------------------------------------------------------

/// The frequency of each letter, ignoring everything else, as fractions
/// summing to one.
#[must_use]
pub fn frequency_analysis(text: &[u8]) -> [f64; 26] {
    let mut counts = [0f64; 26];
    let mut total = 0f64;
    for &b in text {
        let c = b.to_ascii_lowercase();
        if c.is_ascii_lowercase() {
            counts[(c - b'a') as usize] += 1.0;
            total += 1.0;
        }
    }
    if total > 0.0 {
        for c in &mut counts {
            *c /= total;
        }
    }
    counts
}

/// The index of coincidence: the chance that two letters drawn at random from
/// the text are the same.
///
/// About `0.066` for English and `0.038` for a uniform jumble. Because it is
/// unchanged by a substitution -- relabelling the letters does not change how
/// often two match -- it tells a monoalphabetic cipher from a polyalphabetic
/// one without any guess about the key, which is what makes it the first
/// measurement to take.
#[must_use]
pub fn index_of_coincidence(text: &[u8]) -> f64 {
    let mut counts = [0f64; 26];
    let mut n = 0f64;
    for &b in text {
        let c = b.to_ascii_lowercase();
        if c.is_ascii_lowercase() {
            counts[(c - b'a') as usize] += 1.0;
            n += 1.0;
        }
    }
    if n < 2.0 {
        return 0.0;
    }
    counts.iter().map(|&f| f * (f - 1.0)).sum::<f64>() / (n * (n - 1.0))
}

/// Candidate key lengths from repeated trigrams, as Kasiski proposed.
///
/// A trigram repeating in the ciphertext usually means the same plaintext
/// trigram met the same stretch of key, so the gap between the two is a
/// multiple of the key length. Returns the lengths that divide the most gaps,
/// best first.
#[must_use]
pub fn kasiski_examination(text: &[u8]) -> Vec<usize> {
    let letters: Vec<u8> = text
        .iter()
        .map(|b| b.to_ascii_lowercase())
        .filter(u8::is_ascii_lowercase)
        .collect();
    if letters.len() < 6 {
        return Vec::new();
    }
    let mut seen: std::collections::BTreeMap<[u8; 3], Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..letters.len() - 2 {
        seen.entry([letters[i], letters[i + 1], letters[i + 2]]).or_default().push(i);
    }
    let mut votes = vec![0usize; 32];
    for positions in seen.values() {
        for w in positions.windows(2) {
            let gap = w[1] - w[0];
            for (len, vote) in votes.iter_mut().enumerate().skip(2) {
                if gap.is_multiple_of(len) {
                    *vote += 1;
                }
            }
        }
    }
    let mut order: Vec<usize> = (2..votes.len()).filter(|&i| votes[i] > 0).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(votes[i]), i));
    order
}

/// The expected letter frequencies of English text.
const ENGLISH: [f64; 26] = [
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153,
    0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056,
    0.02758, 0.00978, 0.02360, 0.00150, 0.01974, 0.00074,
];

/// The Caesar shift that best matches English letter frequencies.
///
/// Scored by the dot product of the observed and expected distributions,
/// which is largest when the two line up -- the same statistic as chi-squared
/// scoring, with the arithmetic the other way up.
#[must_use]
pub fn caesar_break(text: &[u8]) -> u8 {
    let freq = frequency_analysis(text);
    (0..26u8)
        .max_by(|&s1, &s2| {
            let score = |s: u8| -> f64 {
                (0..26).map(|i| freq[(i + s as usize) % 26] * ENGLISH[i]).sum()
            };
            score(s1).total_cmp(&score(s2))
        })
        .expect("there are 26 shifts")
}

/// The most likely Vigenere key, searching lengths up to `max_key`.
///
/// The key length is chosen by the average index of coincidence of the
/// columns -- at the true length each column is a Caesar shift of English and
/// so looks like English, and at any other length the columns are jumbled --
/// and each column is then solved as its own Caesar shift.
///
/// # Panics
/// Panics if `max_key` is zero.
#[must_use]
pub fn vigenere_break(text: &[u8], max_key: usize) -> String {
    assert!(max_key > 0, "search at least one key length");
    let letters: Vec<u8> = text
        .iter()
        .map(|b| b.to_ascii_lowercase())
        .filter(u8::is_ascii_lowercase)
        .collect();
    if letters.is_empty() {
        return String::new();
    }
    let column_ioc = |len: usize| -> f64 {
        let mut total = 0.0;
        for c in 0..len {
            let column: Vec<u8> = letters.iter().skip(c).step_by(len).copied().collect();
            total += index_of_coincidence(&column);
        }
        total / len as f64
    };
    let best_len = (1..=max_key.min(letters.len()))
        .max_by(|&a, &b| column_ioc(a).total_cmp(&column_ioc(b)))
        .expect("at least one length");
    (0..best_len)
        .map(|c| {
            let column: Vec<u8> = letters.iter().skip(c).step_by(best_len).copied().collect();
            (b'a' + caesar_break(&column)) as char
        })
        .collect()
}

/// The permutation a perfect riffle shuffle applies to `n` cards.
///
/// An *out* shuffle keeps the top card on top; an *in* shuffle pushes it to
/// second. Eight out-shuffles restore a 52-card deck and 52 in-shuffles do,
/// which is the standard demonstration that a deterministic shuffle is no
/// shuffle at all.
///
/// # Panics
/// Panics unless `n` is positive and even.
#[must_use]
pub fn perfect_shuffle_permutation(n: usize, out: bool) -> Vec<usize> {
    assert!(n > 0 && n.is_multiple_of(2), "a riffle needs an even, positive deck");
    let half = n / 2;
    (0..n)
        .map(|i| {
            let (from_top, idx) = if out { (i % 2 == 0, i / 2) } else { (i % 2 == 1, i / 2) };
            if from_top {
                idx
            } else {
                half + idx
            }
        })
        .collect()
}

/// How many times a permutation must be applied before everything returns
/// home: the least common multiple of its cycle lengths.
///
/// # Panics
/// Panics unless the input is a permutation of `0..n`.
#[must_use]
pub fn permutation_cipher_period(perm: &[usize]) -> u64 {
    let n = perm.len();
    let mut seen = vec![false; n];
    for &x in perm {
        assert!(x < n && !seen[x], "the input is not a permutation");
        seen[x] = true;
    }
    let mut visited = vec![false; n];
    let mut period = 1u64;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut len = 0u64;
        let mut i = start;
        while !visited[i] {
            visited[i] = true;
            i = perm[i];
            len += 1;
        }
        period = num_lcm(period, len);
    }
    period
}

fn num_lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / num_gcd(a, b) * b
}

fn num_gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn big(n: u64) -> BigInt {
        BigInt::from_u64(n)
    }

    /// RSA works because exponentiating by `e` and then by `d` is the
    /// identity on the whole ring, which is the theorem rather than a
    /// property of any particular message.
    #[test]
    fn rsa_roundtrips_and_its_exponents_invert() {
        let mut rng = Rng::new(0x_45A1);
        for bits in [32usize, 48, 64] {
            let (n, e, d, p, q) = rsa_keygen_with_primes(bits, &mut rng);
            assert!(n.bits() >= bits, "the modulus came out too small");
            assert_eq!(p.mul(&q), n, "the primes do not multiply to the modulus");
            assert!(crate::discrete::primes::is_prime_bigint(&p, 20, &mut rng));
            assert!(crate::discrete::primes::is_prime_bigint(&q, 20, &mut rng));
            // The defining relation: e d is one modulo the group's exponent.
            let one = BigInt::one();
            let lambda = p.sub(&one).lcm(&q.sub(&one));
            assert_eq!(e.mul(&d).rem_euclid(&lambda), one, "e and d do not invert");

            for _ in 0..12 {
                let m = BigInt::random_below(&n, &mut rng);
                let c = rsa_encrypt(&m, &e, &n);
                assert_eq!(rsa_decrypt(&c, &d, &n), m, "the roundtrip failed");
                // The Chinese remainder route must agree exactly.
                assert_eq!(rsa_crt_decrypt(&c, &d, &p, &q), m, "the CRT route disagreed");
                // Encryption is a bijection on residues, so distinct messages
                // give distinct ciphertexts.
                let m2 = m.add(&one).rem_euclid(&n);
                if m2 != m {
                    assert_ne!(rsa_encrypt(&m2, &e, &n), c, "two messages collided");
                }
            }
            // Signing is the same operation with the exponents swapped.
            let m = BigInt::random_below(&n, &mut rng);
            let sig = rsa_decrypt(&m, &d, &n);
            assert_eq!(rsa_encrypt(&sig, &e, &n), m, "the signature did not verify");
        }
        assert!(std::panic::catch_unwind(|| {
            let _ = rsa_keygen(8, &mut Rng::new(1));
        })
        .is_err());
    }

    /// Diffie-Hellman: both sides reach the same value, and it is the one the
    /// exponents say it should be.
    #[test]
    fn diffie_hellman_agrees_on_both_sides() {
        let mut rng = Rng::new(0x_D1FE);
        // Safe primes, so the generator lands in a large subgroup.
        for &p64 in &[23u64, 47, 167, 359, 1439, 2027] {
            let p = big(p64);
            let g = big(5);
            for _ in 0..10 {
                let ((a, big_a), (b, big_b), s) = diffie_hellman_demo(&p, &g, &mut rng);
                assert_eq!(big_a, g.mod_pow(&a, &p));
                assert_eq!(big_b, g.mod_pow(&b, &p));
                // The point of the exchange: the two computations agree.
                assert_eq!(s, big_a.mod_pow(&b, &p), "the two sides disagree");
                assert_eq!(s, g.mod_pow(&a.mul(&b), &p), "the secret is not g^(ab)");
            }
        }
    }

    /// The curve group law is a group law: it has an identity, inverses, and
    /// -- the one non-obvious part -- it is associative.
    #[test]
    fn the_curve_group_law_is_a_group() {
        let mut rng = Rng::new(0x_EC97);
        for (a, b, p) in [(2u64, 3u64, 97u64), (0, 7, 199), (1, 1, 101), (3, 8, 13)] {
            let curve = EcCurve::new(big(a), big(b), big(p));
            let points = curve.all_points();
            assert!(!points.is_empty());
            for pt in &points {
                assert!(curve.is_on_curve(pt), "an enumerated point is off the curve");
            }
            let mut with_inf = points.clone();
            with_inf.push(EcPoint::Infinity);

            for pt in &with_inf {
                // Identity and inverse.
                assert_eq!(curve.add(pt, &EcPoint::Infinity), *pt);
                assert_eq!(curve.add(&EcPoint::Infinity, pt), *pt);
                assert_eq!(curve.add(pt, &curve.negate(pt)), EcPoint::Infinity);
                assert!(curve.is_on_curve(&curve.negate(pt)));
            }
            // Closure and commutativity across every pair.
            for x in &with_inf {
                for y in &with_inf {
                    let s = curve.add(x, y);
                    assert!(curve.is_on_curve(&s), "the sum left the curve");
                    assert_eq!(s, curve.add(y, x), "the group law is not commutative");
                }
            }
            // Associativity, on a sample -- the full triple loop says nothing
            // more and costs the cube of the group order.
            for _ in 0..300 {
                let x = &with_inf[pick(&mut rng, with_inf.len())];
                let y = &with_inf[pick(&mut rng, with_inf.len())];
                let z = &with_inf[pick(&mut rng, with_inf.len())];
                assert_eq!(
                    curve.add(&curve.add(x, y), z),
                    curve.add(x, &curve.add(y, z)),
                    "the group law is not associative"
                );
            }
            // Doubling agrees with adding a point to itself.
            for pt in &with_inf {
                assert_eq!(curve.double(pt), curve.add(pt, pt));
            }
            // A singular curve is refused.
        }
        // 4a^3 + 27b^2 = 0 modulo p makes the curve singular.
        assert!(std::panic::catch_unwind(|| EcCurve::new(big(0), big(0), big(97))).is_err());
    }

    /// Scalar multiplication is repeated addition, and Lagrange and Hasse
    /// both hold.
    #[test]
    fn scalar_multiplication_matches_repeated_addition() {
        let mut rng = Rng::new(0x_5CA1);
        for (a, b, p) in [(2u64, 3u64, 97u64), (0, 7, 199), (1, 1, 101), (2, 2, 1009)] {
            let curve = EcCurve::new(big(a), big(b), big(p));
            let order = curve.order_naive_small();
            assert!(hasse_bound_check(order, p), "Hasse's bound fails for {order} on {p}");
            assert_eq!(ec_count_points_small(&curve), order);

            for _ in 0..12 {
                let g = curve.random_point(&mut rng);
                // Repeated addition, against the ladder.
                let mut acc = EcPoint::Infinity;
                for k in 0..40u64 {
                    assert_eq!(
                        curve.scalar_mul(&big(k), &g),
                        acc,
                        "the ladder disagrees at {k}"
                    );
                    acc = curve.add(&acc, &g);
                }
                // Lagrange: the point's order divides the group's.
                let ord = curve.point_order_small(&g);
                assert!(order.is_multiple_of(ord), "{ord} does not divide {order}");
                assert_eq!(curve.scalar_mul(&big(ord), &g), EcPoint::Infinity);
                // And the whole group annihilates every point.
                assert_eq!(curve.scalar_mul(&big(order), &g), EcPoint::Infinity);
                // A negative scalar is the inverse of the positive one.
                let k = big(7);
                assert_eq!(
                    curve.scalar_mul(&k.neg(), &g),
                    curve.negate(&curve.scalar_mul(&k, &g))
                );
            }
        }
    }

    /// The standard curves are what they are documented to be: their
    /// generators lie on them and have the stated order.
    #[test]
    fn the_named_curves_have_their_published_generators() {
        for (name, curve, (g, n)) in [
            ("secp256k1", EcCurve::secp256k1(), EcCurve::secp256k1_generator()),
            ("P-256", EcCurve::p256(), EcCurve::p256_generator()),
        ] {
            assert!(curve.is_on_curve(&g), "{name}: the generator is not on the curve");
            // The order really is the order: n G is infinity and the
            // generator is not itself infinity.
            assert_ne!(g, EcPoint::Infinity);
            assert_eq!(curve.scalar_mul(&n, &g), EcPoint::Infinity, "{name}: n G is not infinity");
            // The order is prime, so no smaller multiple can vanish.
            let mut rng = Rng::new(0x_C127);
            assert!(
                crate::discrete::primes::is_prime_bigint(&n, 20, &mut rng),
                "{name}: the group order should be prime"
            );
            // Scalar multiplication is a homomorphism, checked on the real
            // curve rather than a toy one.
            let (a, b) = (big(123_456_789), big(987_654_321));
            let sum = curve.add(&curve.scalar_mul(&a, &g), &curve.scalar_mul(&b, &g));
            assert_eq!(curve.scalar_mul(&a.add(&b), &g), sum, "{name}: not a homomorphism");
        }
    }

    /// Elliptic curve Diffie-Hellman reaches the same point from both sides.
    #[test]
    fn ecdh_agrees_on_both_sides() {
        let mut rng = Rng::new(0x_ECD4);
        let curve = EcCurve::new(big(2), big(3), big(1009));
        let order = big(curve.order_naive_small());
        for _ in 0..20 {
            let g = curve.random_point(&mut rng);
            let ((a, big_a), (b, big_b), s) = ecdh_demo(&curve, &g, &order, &mut rng);
            assert_eq!(big_a, curve.scalar_mul(&a, &g));
            assert_eq!(big_b, curve.scalar_mul(&b, &g));
            assert_eq!(s, curve.scalar_mul(&b, &big_a), "the two sides disagree");
            assert!(curve.is_on_curve(&s));
        }
        // On a real curve too, where the arithmetic is the same and the
        // numbers are not.
        let (g, n) = EcCurve::secp256k1_generator();
        let curve = EcCurve::secp256k1();
        let ((_, big_a), (b, _), s) = ecdh_demo(&curve, &g, &n, &mut rng);
        assert_eq!(s, curve.scalar_mul(&b, &big_a));
    }

    /// Any `k` shares rebuild the secret and any `k - 1` determine nothing --
    /// which is the exact statement that makes the scheme perfect rather than
    /// merely hard.
    #[test]
    fn shamir_needs_exactly_k_shares() {
        let mut rng = Rng::new(0x_5A31);
        let prime = big(2_147_483_647);
        for k in 1..=5usize {
            for n in k..=7usize {
                for _ in 0..8 {
                    let secret = BigInt::random_below(&prime, &mut rng);
                    let shares = shamir_split(&secret, k, n, &prime, &mut rng);
                    assert_eq!(shares.len(), n);
                    // Any k of them work, whichever k.
                    for combo in
                        crate::discrete::combinatorics::combinations_iter(n, k)
                    {
                        let subset: Vec<(u64, BigInt)> =
                            combo.iter().map(|&i| shares[i].clone()).collect();
                        assert_eq!(
                            shamir_reconstruct(&subset, &prime),
                            secret,
                            "{k} of {n} shares failed to reconstruct"
                        );
                    }
                    // Fewer than k do not. They interpolate to *something*,
                    // and that something is almost never the secret; the
                    // point is that every value is equally consistent.
                    if k >= 2 {
                        let mut wrong = 0;
                        for combo in
                            crate::discrete::combinatorics::combinations_iter(n, k - 1)
                        {
                            let subset: Vec<(u64, BigInt)> =
                                combo.iter().map(|&i| shares[i].clone()).collect();
                            if shamir_reconstruct(&subset, &prime) != secret {
                                wrong += 1;
                            }
                        }
                        assert!(wrong > 0, "{} shares recovered a {k}-threshold secret", k - 1);
                    }
                }
            }
        }
        // A repeated share is refused rather than silently interpolated.
        let shares = shamir_split(&big(42), 2, 3, &prime, &mut rng);
        let dup = vec![shares[0].clone(), shares[0].clone()];
        assert!(std::panic::catch_unwind(move || shamir_reconstruct(&dup, &big(2_147_483_647)))
            .is_err());
    }

    /// The pad is its own inverse, and with a full-length key it hides
    /// everything: every plaintext is consistent with every ciphertext.
    #[test]
    fn the_pad_is_its_own_inverse_and_hides_everything() {
        let mut rng = Rng::new(0x_07AD);
        for _ in 0..200 {
            let n = 1 + pick(&mut rng, 64);
            let data: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let key: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let c = one_time_pad(&data, &key);
            assert_eq!(one_time_pad(&c, &key), data, "the pad is not an involution");
            // Perfect secrecy, constructively: for any target plaintext there
            // is a key turning this ciphertext into it, so the ciphertext
            // rules nothing out.
            let target: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let alt: Vec<u8> = c.iter().zip(&target).map(|(&x, &y)| x ^ y).collect();
            assert_eq!(one_time_pad(&c, &alt), target);
        }
        // Reusing a short key is a Vigenere cipher, and the difference of two
        // ciphertexts loses the key entirely -- which is why reuse is fatal.
        let key = b"key";
        let a = one_time_pad(b"attack at dawn!!", key);
        let b = one_time_pad(b"retreat at once!", key);
        let diff: Vec<u8> = a.iter().zip(&b).map(|(&x, &y)| x ^ y).collect();
        let plain: Vec<u8> = b"attack at dawn!!"
            .iter()
            .zip(b"retreat at once!")
            .map(|(&x, &y)| x ^ y)
            .collect();
        assert_eq!(diff, plain, "key reuse should cancel the key");
    }

    /// A shift register's output is a linear recurrence, and Berlekamp-Massey
    /// recovers the register from twice its length of output -- which is the
    /// reason a bare register is not a cipher.
    #[test]
    fn berlekamp_massey_recovers_the_register() {
        // Primitive polynomials, whose registers run through every non-zero
        // state before repeating.
        // Tap masks that reach every non-zero state. Bit zero is set in each,
        // without which the step map is not a bijection and the register
        // never returns to where it started.
        for (width, taps, period) in
            [(3u32, 0b011u64, 7u64), (4, 0b0011, 15), (5, 0b00101, 31), (7, 0b0000011, 127)]
        {
            assert_eq!(lfsr_period(taps, width), period, "width {width} has the wrong period");
            let stream = lfsr(taps, 1, 4 * width as usize);
            let (len, _) = berlekamp_massey_attack(&stream);
            assert!(
                len <= u64::from(width),
                "the recovered register is longer than the real one"
            );
            // The recovered recurrence predicts the rest of the stream, which
            // is the actual attack: everything after the observed prefix.
            let long = lfsr(taps, 1, 8 * width as usize);
            let (l, t) = berlekamp_massey_attack(&long[..4 * width as usize]);
            let l = l as usize;
            for i in l..long.len() {
                let predicted = (0..l)
                    .filter(|&j| t >> j & 1 == 1)
                    .fold(false, |acc, j| acc ^ long[i - 1 - j]);
                assert_eq!(predicted, long[i], "the recovered recurrence mispredicts at {i}");
            }
        }
        // A tap set that is reversible but not primitive falls short of the
        // maximum, and one that does not tap bit zero never returns at all.
        assert!((1..15).contains(&lfsr_period(0b0101, 4)));
        assert_eq!(lfsr_period(0b1010, 4), 0, "an irreversible register cannot return");
        // A stream of zeros needs no recurrence at all, and a single one
        // needs the shortest that can produce it.
        assert_eq!(berlekamp_massey_attack(&[false; 20]).0, 0);
        assert!(berlekamp_massey_attack(&[true, false, false, false]).0 >= 1);
    }

    /// The avalanche measurement distinguishes a mixing function from one
    /// that is not.
    #[test]
    fn the_avalanche_test_separates_good_mixing_from_bad() {
        let mut rng = Rng::new(0x_4A14);
        // A deliberately terrible hash: the first byte, zero-extended. One
        // input bit flip changes at most one output bit.
        let bad = |d: &[u8]| -> u64 { u64::from(d[0]) };
        let bad_score = hash_avalanche_test(&bad, 40, &mut rng);
        assert!(bad_score < 0.02, "a trivial hash scored {bad_score}");

        // A mixing hash in the SplitMix style.
        let good = |d: &[u8]| -> u64 {
            let mut h = 0xCBF2_9CE4_8422_2325u64;
            for &b in d {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100_0000_01B3);
                h ^= h >> 33;
                h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                h ^= h >> 33;
            }
            h
        };
        let good_score = hash_avalanche_test(&good, 40, &mut rng);
        assert!(
            (good_score - 0.5).abs() < 0.03,
            "a mixing hash should flip half the bits, not {good_score}"
        );
        // The birthday bound: collisions become likely near the square root.
        assert!((birthday_bound(64) / (2.0f64).powi(32) - (PI_OVER_2).sqrt()).abs() < 1e-9);
        assert!(birthday_bound(128) > birthday_bound(64));
        assert!(birthday_bound(256) / birthday_bound(128) > 1e19);
    }

    /// The classical analyses recover what they are supposed to from a sample
    /// of English.
    #[test]
    fn classical_cipher_analysis_recovers_its_keys() {
        // A passage long enough for the statistics to settle.
        let plain: Vec<u8> = std::iter::repeat_n(
            b"it is a truth universally acknowledged that a single man in possession \
              of a good fortune must be in want of a wife. however little known the \
              feelings or views of such a man may be on his first entering a \
              neighbourhood this truth is so well fixed in the minds of the surrounding \
              families that he is considered as the rightful property of some one or \
              other of their daughters. "
                .as_slice(),
            2,
        )
        .flatten()
        .copied()
        .collect();
        assert!(plain.len() > 500, "the sample is too short to break anything");

        // The index of coincidence separates English from a jumble.
        let ioc = index_of_coincidence(&plain);
        assert!((0.055..0.085).contains(&ioc), "English should sit near 0.066, not {ioc}");
        let mut rng = Rng::new(0x_1A55);
        let jumble: Vec<u8> = (0..2000).map(|_| b'a' + (pick(&mut rng, 26) as u8)).collect();
        let flat = index_of_coincidence(&jumble);
        assert!(flat < 0.05, "a uniform jumble should sit near 0.038, not {flat}");
        // Frequencies sum to one, and `e` is the commonest letter.
        let freq = frequency_analysis(&plain);
        assert!((freq.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let top = (0..26).max_by(|&i, &j| freq[i].total_cmp(&freq[j])).expect("26 letters");
        assert_eq!(top, (b'e' - b'a') as usize, "the commonest letter should be e");

        // Caesar, for every shift.
        for shift in 0..26u8 {
            let ct: Vec<u8> = plain
                .iter()
                .map(|&b| {
                    if b.is_ascii_lowercase() {
                        b'a' + (b - b'a' + shift) % 26
                    } else {
                        b
                    }
                })
                .collect();
            assert_eq!(caesar_break(&ct), shift, "the Caesar shift {shift} was not recovered");
        }

        // Vigenere, for several keys.
        for key in ["lemon", "cipher", "zebra", "wxyz"] {
            let letters: Vec<u8> =
                plain.iter().copied().filter(u8::is_ascii_lowercase).collect();
            let ct: Vec<u8> = letters
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    let k = key.as_bytes()[i % key.len()] - b'a';
                    b'a' + (b - b'a' + k) % 26
                })
                .collect();
            assert_eq!(vigenere_break(&ct, 12), key, "the key {key} was not recovered");
            // Kasiski should suggest the key length or a multiple of it.
            let suggestions = kasiski_examination(&ct);
            assert!(
                suggestions.iter().take(5).any(|&l| l.is_multiple_of(key.len())),
                "Kasiski's top suggestions {suggestions:?} miss {}",
                key.len()
            );
        }
    }

    /// The perfect shuffle is a permutation with the periods it is famous
    /// for.
    #[test]
    fn the_perfect_shuffle_has_its_known_periods() {
        // Eight out-shuffles restore a 52-card deck; 52 in-shuffles do.
        assert_eq!(permutation_cipher_period(&perfect_shuffle_permutation(52, true)), 8);
        assert_eq!(permutation_cipher_period(&perfect_shuffle_permutation(52, false)), 52);
        // Both are genuine permutations at every even size.
        for n in (2..=40).step_by(2) {
            for out in [true, false] {
                let p = perfect_shuffle_permutation(n, out);
                let mut seen = vec![false; n];
                for &x in &p {
                    assert!(x < n && !seen[x], "the shuffle is not a permutation at {n}");
                    seen[x] = true;
                }
                // The period really returns the deck to its start.
                let period = permutation_cipher_period(&p);
                let mut deck: Vec<usize> = (0..n).collect();
                for _ in 0..period {
                    deck = p.iter().map(|&i| deck[i]).collect();
                }
                assert_eq!(deck, (0..n).collect::<Vec<_>>(), "the period is wrong at {n}");
            }
        }
        // The identity has period one, and a transposition period two.
        assert_eq!(permutation_cipher_period(&[0, 1, 2, 3]), 1);
        assert_eq!(permutation_cipher_period(&[1, 0, 2, 3]), 2);
        assert_eq!(permutation_cipher_period(&[1, 2, 0, 4, 3]), 6);
        assert!(std::panic::catch_unwind(|| permutation_cipher_period(&[0, 0])).is_err());
        assert!(std::panic::catch_unwind(|| perfect_shuffle_permutation(5, true)).is_err());
    }
}
