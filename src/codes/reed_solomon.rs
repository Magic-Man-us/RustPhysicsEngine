//! Reed-Solomon and BCH codes over finite fields.
//!
//! Reed-Solomon works on symbols rather than bits, which is why it appears
//! wherever errors arrive in clumps: a scratch on a disc, a fading burst on a
//! radio link, a smudge across a printed barcode. A byte is wrong whether one
//! bit of it flipped or all eight, so a burst that would defeat a bit-level
//! code costs an `RS(255, 223)` codeword at most one of its sixteen
//! correctable symbols per byte touched.
//!
//! The construction is one idea. Fix a field, treat the message as the
//! coefficients of a polynomial, and multiply by a generator whose roots are
//! consecutive powers of a primitive element. A codeword is then exactly a
//! polynomial vanishing at those `n - k` points, so evaluating the received
//! word there gives zero if nothing went wrong and, if something did, a set
//! of *syndromes* that depend only on the errors. Berlekamp-Massey turns
//! those syndromes into a polynomial whose roots say where the errors are,
//! Chien search finds the roots, and Forney's formula says how large each
//! error was. Every step is field arithmetic; none of it looks at the
//! message.
//!
//! Because the generator has exactly `n - k` roots, the code meets the
//! Singleton bound with equality -- `d = n - k + 1`. Reed-Solomon codes are
//! the standard example of a maximum distance separable code, and there is no
//! slack anywhere in the parameters.

use std::fmt;

/// The field `GF(2^8)`, with logarithm and antilogarithm tables.
///
/// Multiplication in a field of characteristic two is not the processor's
/// multiplication, so it is done through logarithms: every non-zero element
/// is a power of a primitive element, and a product of powers adds their
/// exponents. The `exp` table is doubled in length so the sum of two
/// exponents never needs reducing modulo 255 at the point of use.
#[derive(Debug, Clone)]
pub struct Gf256 {
    /// `log[x]` is the exponent `e` with `alpha^e = x`, for `x` non-zero.
    pub log: [u8; 256],
    /// `exp[e]` is `alpha^e`, tabulated twice round.
    pub exp: [u8; 512],
}

impl Default for Gf256 {
    fn default() -> Self {
        Gf256::new(0x11D)
    }
}

impl Gf256 {
    /// The field defined by a primitive polynomial, given with its leading
    /// term: `0x11D` is `x^8 + x^4 + x^3 + x^2 + 1`, the polynomial used by
    /// CCSDS telemetry, QR codes and most of the rest of the world.
    ///
    /// # Panics
    /// Panics if the polynomial is not primitive, which shows up as the
    /// powers of `alpha` repeating before they have covered all 255 non-zero
    /// elements.
    #[must_use]
    pub fn new(prim_poly: u32) -> Self {
        let mut log = [0u8; 256];
        let mut exp = [0u8; 512];
        let mut x: u32 = 1;
        for e in 0..255 {
            exp[e] = x as u8;
            log[x as usize] = e as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= prim_poly;
            }
        }
        assert_eq!(x, 1, "the polynomial {prim_poly:#x} is not primitive");
        for e in 255..512 {
            exp[e] = exp[e - 255];
        }
        Gf256 { log, exp }
    }

    /// Addition, which in characteristic two is exclusive or and is its own
    /// inverse.
    #[must_use]
    pub fn add(a: u8, b: u8) -> u8 {
        a ^ b
    }

    /// Multiplication, by adding logarithms.
    #[must_use]
    pub fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        self.exp[usize::from(self.log[a as usize]) + usize::from(self.log[b as usize])]
    }

    /// Division.
    ///
    /// # Panics
    /// Panics on division by zero.
    #[must_use]
    pub fn div(&self, a: u8, b: u8) -> u8 {
        assert!(b != 0, "division by zero in GF(256)");
        if a == 0 {
            return 0;
        }
        let d = 255 + i32::from(self.log[a as usize]) - i32::from(self.log[b as usize]);
        self.exp[(d % 255) as usize]
    }

    /// The multiplicative inverse.
    ///
    /// # Panics
    /// Panics on zero, which has none.
    #[must_use]
    pub fn inv(&self, a: u8) -> u8 {
        assert!(a != 0, "zero has no inverse");
        self.exp[255 - usize::from(self.log[a as usize])]
    }

    /// A power, including negative exponents.
    #[must_use]
    pub fn pow(&self, a: u8, e: i32) -> u8 {
        if a == 0 {
            return u8::from(e == 0);
        }
        let l = (i32::from(self.log[a as usize]) * e).rem_euclid(255);
        self.exp[l as usize]
    }

    /// `alpha^e`, the `e`-th power of the primitive element.
    #[must_use]
    pub fn alpha(&self, e: i32) -> u8 {
        self.exp[e.rem_euclid(255) as usize]
    }

    /// Evaluates a polynomial, highest coefficient first, by Horner's rule.
    #[must_use]
    pub fn poly_eval(&self, poly: &[u8], x: u8) -> u8 {
        poly.iter().fold(0u8, |acc, &c| self.mul(acc, x) ^ c)
    }

    /// The product of two polynomials, highest coefficient first.
    #[must_use]
    pub fn poly_mul(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0u8; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            if x == 0 {
                continue;
            }
            for (j, &y) in b.iter().enumerate() {
                out[i + j] ^= self.mul(x, y);
            }
        }
        out
    }

    /// The remainder of `a` on division by `b`, both highest coefficient
    /// first.
    ///
    /// # Panics
    /// Panics if the divisor is zero or has a zero leading coefficient.
    #[must_use]
    pub fn poly_rem(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        assert!(!b.is_empty() && b[0] != 0, "the divisor must be monic-ish and non-zero");
        let mut r = a.to_vec();
        if r.len() < b.len() {
            return trim(&r);
        }
        for i in 0..=r.len() - b.len() {
            let c = r[i];
            if c == 0 {
                continue;
            }
            let factor = self.div(c, b[0]);
            for (j, &d) in b.iter().enumerate() {
                r[i + j] ^= self.mul(factor, d);
            }
        }
        trim(&r[r.len() - b.len() + 1..])
    }
}

/// Drops leading zero coefficients.
fn trim(p: &[u8]) -> Vec<u8> {
    let start = p.iter().position(|&c| c != 0).unwrap_or(p.len());
    p[start..].to_vec()
}

/// A prime field `GF(p)`, for the places a power of two is the wrong shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GfP {
    /// The characteristic, which must be prime.
    pub p: u64,
}

impl GfP {
    /// The field of integers modulo `p`.
    ///
    /// # Panics
    /// Panics unless `p` is prime.
    #[must_use]
    pub fn new(p: u64) -> Self {
        assert!(crate::discrete::primes::is_prime_u64(p), "{p} is not prime");
        GfP { p }
    }

    /// Addition modulo `p`.
    #[must_use]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        (a + b) % self.p
    }

    /// Subtraction modulo `p`.
    #[must_use]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        (a + self.p - b % self.p) % self.p
    }

    /// Multiplication modulo `p`, widened so it cannot overflow.
    #[must_use]
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        ((u128::from(a) * u128::from(b)) % u128::from(self.p)) as u64
    }

    /// A power by repeated squaring.
    #[must_use]
    pub fn pow(&self, a: u64, mut e: u64) -> u64 {
        let (mut base, mut acc) = (a % self.p, 1u64);
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul(acc, base);
            }
            base = self.mul(base, base);
            e >>= 1;
        }
        acc
    }

    /// The multiplicative inverse, by Fermat's little theorem.
    ///
    /// # Panics
    /// Panics on zero.
    #[must_use]
    pub fn inv(&self, a: u64) -> u64 {
        assert!(!a.is_multiple_of(self.p), "zero has no inverse");
        self.pow(a, self.p - 2)
    }
}

/// A general binary extension field `GF(2^m)`, elements held as bit patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gf2m {
    /// The extension degree.
    pub m: u32,
    /// The primitive polynomial, with its leading term.
    pub prim: u64,
}

impl Gf2m {
    /// The field of degree `m` defined by `prim`.
    ///
    /// # Panics
    /// Panics unless `m` is between one and sixteen and `prim` is primitive.
    #[must_use]
    pub fn new(m: u32, prim: u64) -> Self {
        assert!((1..=16).contains(&m), "the degree must be between one and sixteen");
        let f = Gf2m { m, prim };
        // Primitivity: the powers of x must run through every non-zero
        // element before returning to one.
        let n = (1u64 << m) - 1;
        let mut x = 1u64;
        for _ in 0..n - 1 {
            x = f.mul(x, 2);
            assert!(x != 1, "the polynomial {prim:#x} is not primitive");
        }
        assert_eq!(f.mul(x, 2), 1, "the polynomial {prim:#x} is not primitive");
        f
    }

    /// `GF(2^m)` with a primitive polynomial chosen for the degree.
    ///
    /// # Panics
    /// Panics unless `m` is between one and sixteen.
    #[must_use]
    pub fn with_degree(m: u32) -> Self {
        // A primitive polynomial for each degree, in the usual tabulated
        // choices.
        const PRIM: [u64; 17] = [
            0, 0x3, 0x7, 0xB, 0x13, 0x25, 0x43, 0x89, 0x11D, 0x211, 0x409, 0x805, 0x1053,
            0x201B, 0x4443, 0x8003, 0x1100B,
        ];
        assert!((1..=16).contains(&m), "the degree must be between one and sixteen");
        Gf2m::new(m, PRIM[m as usize])
    }

    /// The number of elements.
    #[must_use]
    pub fn order(&self) -> u64 {
        1 << self.m
    }

    /// Addition, which is exclusive or.
    #[must_use]
    pub fn add(a: u64, b: u64) -> u64 {
        a ^ b
    }

    /// Carry-less multiplication reduced by the primitive polynomial.
    #[must_use]
    pub fn mul(&self, mut a: u64, mut b: u64) -> u64 {
        let mut acc = 0u64;
        let top = 1u64 << self.m;
        while b != 0 {
            if b & 1 == 1 {
                acc ^= a;
            }
            b >>= 1;
            a <<= 1;
            if a & top != 0 {
                a ^= self.prim;
            }
        }
        acc
    }

    /// A power by repeated squaring.
    #[must_use]
    pub fn pow(&self, a: u64, mut e: u64) -> u64 {
        let (mut base, mut acc) = (a, 1u64);
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul(acc, base);
            }
            base = self.mul(base, base);
            e >>= 1;
        }
        acc
    }

    /// The multiplicative inverse, as `a^(2^m - 2)`.
    ///
    /// # Panics
    /// Panics on zero.
    #[must_use]
    pub fn inv(&self, a: u64) -> u64 {
        assert!(a != 0, "zero has no inverse");
        self.pow(a, self.order() - 2)
    }

    /// The absolute trace: `a + a^2 + a^4 + ... + a^(2^(m-1))`.
    ///
    /// Always zero or one, because it lands in the prime subfield -- it is
    /// fixed by squaring, and the only elements squaring fixes are the ones
    /// satisfying `x^2 = x`.
    #[must_use]
    pub fn trace(&self, a: u64) -> u64 {
        let mut t = a;
        let mut x = a;
        for _ in 1..self.m {
            x = self.mul(x, x);
            t ^= x;
        }
        t
    }

    /// Every element of the field, in increasing bit-pattern order.
    #[must_use]
    pub fn all_elements(&self) -> Vec<u64> {
        (0..self.order()).collect()
    }

    /// The minimal polynomial of `alpha^e` over `GF(2)`, coefficients from
    /// the constant term up.
    ///
    /// The conjugates of an element in characteristic two are its repeated
    /// squares, and the minimal polynomial is the product of `x - c` over
    /// that cyclotomic coset. Its coefficients land back in `GF(2)` because
    /// squaring permutes the conjugates and so fixes the product.
    #[must_use]
    pub fn minimal_polynomial(&self, e: u64) -> Vec<u64> {
        let n = self.order() - 1;
        let alpha = 2u64;
        let root = self.pow(alpha, e % n);
        if root == 0 {
            return vec![0, 1];
        }
        // The cyclotomic coset of e: e, 2e, 4e, ... modulo 2^m - 1.
        let mut coset = vec![e % n];
        let mut c = (2 * (e % n)) % n;
        while c != e % n {
            coset.push(c);
            c = (2 * c) % n;
        }
        // Multiply out (x - alpha^c) over the coset, constant term first.
        let mut poly = vec![1u64];
        for c in coset {
            let r = self.pow(alpha, c);
            let mut next = vec![0u64; poly.len() + 1];
            for (i, &p) in poly.iter().enumerate() {
                next[i] ^= self.mul(p, r);
                next[i + 1] ^= p;
            }
            poly = next;
        }
        poly
    }
}

/// Decoding failed: more errors than the code can correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooManyErrors;

impl fmt::Display for TooManyErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "more errors than the code can correct")
    }
}

impl std::error::Error for TooManyErrors {}

/// A Reed-Solomon code over `GF(256)`, systematic, with the parity symbols
/// appended.
#[derive(Debug, Clone)]
pub struct ReedSolomon {
    /// Codeword length in symbols, at most 255.
    pub n: usize,
    /// Message length in symbols.
    pub k: usize,
    gf: Gf256,
    gen_poly: Vec<u8>,
}

impl ReedSolomon {
    /// The code with the given length and dimension.
    ///
    /// The generator is `(x - alpha^1)(x - alpha^2) ... (x - alpha^(n-k))`,
    /// so a codeword vanishes at those `n - k` powers. That is the whole
    /// design: the parity symbols are chosen to make it so, and decoding
    /// starts by checking whether it still does.
    ///
    /// # Panics
    /// Panics unless `0 < k < n <= 255`.
    #[must_use]
    pub fn new(n: usize, k: usize) -> Self {
        assert!(k > 0 && k < n && n <= 255, "need 0 < k < n <= 255");
        let gf = Gf256::default();
        let mut gen_poly = vec![1u8];
        for i in 1..=(n - k) {
            gen_poly = gf.poly_mul(&gen_poly, &[1, gf.alpha(i as i32)]);
        }
        ReedSolomon { n, k, gf, gen_poly }
    }

    /// The number of symbol errors the code corrects, `(n - k) / 2`.
    #[must_use]
    pub fn correction_capacity(&self) -> usize {
        (self.n - self.k) / 2
    }

    /// The minimum distance, `n - k + 1`.
    ///
    /// Equal to the Singleton bound, which is what makes Reed-Solomon codes
    /// maximum distance separable.
    #[must_use]
    pub fn distance(&self) -> usize {
        self.n - self.k + 1
    }

    /// Encodes a message into a systematic codeword: the message unchanged,
    /// followed by `n - k` parity symbols.
    ///
    /// # Panics
    /// Panics unless the message has exactly `k` symbols.
    #[must_use]
    pub fn encode(&self, msg: &[u8]) -> Vec<u8> {
        assert_eq!(msg.len(), self.k, "the message must have exactly k symbols");
        // Shift the message up by n - k and take the remainder: subtracting
        // it leaves a multiple of the generator whose top k symbols are
        // still the message.
        let mut shifted = msg.to_vec();
        shifted.extend(std::iter::repeat_n(0u8, self.n - self.k));
        let rem = self.gf.poly_rem(&shifted, &self.gen_poly);
        let mut out = msg.to_vec();
        out.extend(std::iter::repeat_n(0u8, self.n - self.k - rem.len()));
        out.extend(rem);
        out
    }

    /// The syndromes of a received word: its value at each generator root.
    ///
    /// All zero exactly when the word is a codeword. Crucially they depend
    /// only on the error pattern, not on what was sent, since the transmitted
    /// polynomial contributes zero at every one of these points.
    #[must_use]
    pub fn syndromes(&self, recv: &[u8]) -> Vec<u8> {
        (1..=(self.n - self.k))
            .map(|i| self.gf.poly_eval(recv, self.gf.alpha(i as i32)))
            .collect()
    }

    /// Decodes a received word, returning the message and the number of
    /// symbols corrected.
    ///
    /// Berlekamp-Massey finds the shortest linear recurrence the syndromes
    /// satisfy; its characteristic polynomial is the error locator, whose
    /// roots are the reciprocals of the error positions. Chien search finds
    /// them by evaluating at every field element, and Forney's formula
    /// recovers each error's magnitude from the error evaluator polynomial.
    ///
    /// # Errors
    /// Returns [`TooManyErrors`] when the word is further than
    /// `(n - k) / 2` symbols from every codeword, which the decoder detects
    /// as a locator whose roots do not account for its own degree.
    ///
    /// # Panics
    /// Panics unless the word has exactly `n` symbols.
    pub fn decode(&self, recv: &[u8]) -> Result<(Vec<u8>, usize), TooManyErrors> {
        assert_eq!(recv.len(), self.n, "the word must have exactly n symbols");
        let word = self.correct(recv)?;
        let corrected = recv.iter().zip(&word.0).filter(|(a, b)| a != b).count();
        Ok((word.0[..self.k].to_vec(), corrected))
    }

    /// Decodes to the full corrected codeword rather than just the message.
    ///
    /// # Errors
    /// Returns [`TooManyErrors`] as [`decode`](Self::decode) does.
    ///
    /// # Panics
    /// Panics unless the word has exactly `n` symbols.
    pub fn correct(&self, recv: &[u8]) -> Result<(Vec<u8>, usize), TooManyErrors> {
        assert_eq!(recv.len(), self.n, "the word must have exactly n symbols");
        let syn = self.syndromes(recv);
        if syn.iter().all(|&s| s == 0) {
            return Ok((recv.to_vec(), 0));
        }
        let locator = self.berlekamp_massey(&syn);
        let degree = locator.len() - 1;
        if degree > self.correction_capacity() {
            return Err(TooManyErrors);
        }
        let positions = self.chien_search(&locator);
        if positions.len() != degree {
            return Err(TooManyErrors);
        }
        let fixed = self.forney(recv, &syn, &locator, &positions);
        // A successful correction lands on a codeword. Checking that rather
        // than trusting the algebra is what turns a miscorrection into a
        // reported failure.
        if !self.syndromes(&fixed).iter().all(|&s| s == 0) {
            return Err(TooManyErrors);
        }
        Ok((fixed, positions.len()))
    }

    /// The error locator polynomial, by the Berlekamp-Massey algorithm.
    ///
    /// Returned highest coefficient first, so its degree is the number of
    /// errors. The algorithm builds the shortest recurrence generating the
    /// syndromes, extending it only when the current one mispredicts, which
    /// is why it finds the *fewest* errors consistent with what was seen.
    fn berlekamp_massey(&self, syn: &[u8]) -> Vec<u8> {
        let gf = &self.gf;
        // Both polynomials are held constant term first here, and reversed
        // on the way out.
        let mut c = vec![1u8];
        let mut b = vec![1u8];
        let mut l = 0usize;
        let mut m = 1usize;
        let mut bb = 1u8;
        for i in 0..syn.len() {
            // The discrepancy: what the current recurrence predicts against
            // what the syndrome actually is.
            let mut d = syn[i];
            for j in 1..=l {
                d ^= gf.mul(c[j], syn[i - j]);
            }
            if d == 0 {
                m += 1;
            } else if 2 * l <= i {
                let t = c.clone();
                let scale = gf.div(d, bb);
                if c.len() < b.len() + m {
                    c.resize(b.len() + m, 0);
                }
                for (j, &x) in b.iter().enumerate() {
                    c[j + m] ^= gf.mul(scale, x);
                }
                l = i + 1 - l;
                b = t;
                bb = d;
                m = 1;
            } else {
                let scale = gf.div(d, bb);
                if c.len() < b.len() + m {
                    c.resize(b.len() + m, 0);
                }
                for (j, &x) in b.iter().enumerate() {
                    c[j + m] ^= gf.mul(scale, x);
                }
                m += 1;
            }
        }
        c.truncate(l + 1);
        c.reverse();
        trim(&c)
    }

    /// The locator value of a position: the power of `alpha` that symbol
    /// carries in the codeword polynomial.
    ///
    /// Symbol zero is the *leading* coefficient here, so position `j` carries
    /// `x^(n-1-j)` and its locator value is `alpha^(n-1-j)`. Getting this
    /// backwards is the classic way to build a decoder that verifies its own
    /// algebra and still corrects the wrong symbols.
    fn locator_value(&self, j: usize) -> u8 {
        self.gf.alpha((self.n - 1 - j) as i32)
    }

    /// The error positions, by evaluating the locator at every field element.
    ///
    /// The locator vanishes at the reciprocal of each error's locator value.
    /// Chien's contribution is that stepping from one element to the next is
    /// a multiplication per coefficient rather than a fresh evaluation.
    fn chien_search(&self, locator: &[u8]) -> Vec<usize> {
        (0..self.n)
            .filter(|&j| {
                let x = self.gf.inv(self.locator_value(j));
                self.gf.poly_eval(locator, x) == 0
            })
            .collect()
    }

    /// The corrected word, by Forney's formula for the error magnitudes.
    fn forney(&self, recv: &[u8], syn: &[u8], locator: &[u8], positions: &[usize]) -> Vec<u8> {
        let gf = &self.gf;
        // The syndrome polynomial, constant term first in the usual
        // convention, held here highest first to match poly_mul.
        let mut synpoly: Vec<u8> = syn.to_vec();
        synpoly.reverse();
        let mut omega = gf.poly_mul(&synpoly, locator);
        // Modulo x^(n-k): keep the low n - k coefficients.
        let keep = self.n - self.k;
        if omega.len() > keep {
            omega = omega[omega.len() - keep..].to_vec();
        }
        // The formal derivative of the locator. In characteristic two every
        // even-power term differentiates away, so this keeps the alternate
        // coefficients and nothing else.
        let mut deriv: Vec<u8> = Vec::new();
        let deg = locator.len() - 1;
        for (idx, &c) in locator.iter().enumerate() {
            let power = deg - idx;
            if power % 2 == 1 {
                deriv.push(c);
                deriv.push(0);
            }
        }
        if !deriv.is_empty() {
            deriv.pop();
        }
        let mut out = recv.to_vec();
        for &j in positions {
            let xi = gf.inv(self.locator_value(j));
            let num = gf.poly_eval(&omega, xi);
            let den = gf.poly_eval(&deriv, xi);
            if den == 0 {
                continue;
            }
            // Forney: the magnitude is X^(1-b) omega(X^-1) / lambda'(X^-1)
            // for a generator whose roots start at alpha^b. Here b is one, so
            // the leading factor is one and drops out; the sign the formula
            // carries drops out too, since this field has characteristic two.
            out[j] ^= gf.div(num, den);
        }
        out
    }

    /// Decodes a word with known erasure positions.
    ///
    /// An erasure -- a symbol known to be unreliable but whose correct value
    /// is unknown -- costs half what an error does, because its position is
    /// already known and only its magnitude has to be found. The code can
    /// handle any `e` errors and `f` erasures with `2e + f <= n - k`; this
    /// routine takes the pure-erasure case, `f <= n - k`.
    ///
    /// # Errors
    /// Returns [`TooManyErrors`] if there are more erasures than parity
    /// symbols, or if the result is not a codeword.
    ///
    /// # Panics
    /// Panics unless the word has `n` symbols and the positions are inside it.
    pub fn decode_erasures(
        &self,
        recv: &[u8],
        erasure_pos: &[usize],
    ) -> Result<Vec<u8>, TooManyErrors> {
        assert_eq!(recv.len(), self.n, "the word must have exactly n symbols");
        assert!(erasure_pos.iter().all(|&p| p < self.n), "an erasure is outside the word");
        if erasure_pos.len() > self.n - self.k {
            return Err(TooManyErrors);
        }
        let gf = &self.gf;
        // The erasure locator has a root at each known position, so its
        // degree is the erasure count and no search is needed.
        let mut locator = vec![1u8];
        for &p in erasure_pos {
            // A factor vanishing at the reciprocal of that position's
            // locator value, so the product has exactly the known roots.
            locator = gf.poly_mul(&locator, &[self.locator_value(p), 1]);
        }
        let syn = self.syndromes(recv);
        if syn.iter().all(|&s| s == 0) {
            return Ok(recv[..self.k].to_vec());
        }
        let fixed = self.forney(recv, &syn, &locator, erasure_pos);
        if !self.syndromes(&fixed).iter().all(|&s| s == 0) {
            return Err(TooManyErrors);
        }
        Ok(fixed[..self.k].to_vec())
    }
}

/// `RS(255, 223)`, the CCSDS telemetry standard: sixteen correctable symbol
/// errors in a 255-byte frame, used on essentially every deep space mission
/// since Voyager.
#[must_use]
pub fn rs_ccsds() -> ReedSolomon {
    ReedSolomon::new(255, 223)
}

/// The Reed-Solomon block a QR code of the given version uses at its lowest
/// error correction level.
///
/// # Panics
/// Panics unless the version is between one and four, the range tabulated
/// here.
#[must_use]
pub fn rs_qr_code(version: usize) -> ReedSolomon {
    // (total codewords, data codewords) for versions 1 to 4 at level L.
    const BLOCKS: [(usize, usize); 4] = [(26, 19), (44, 34), (70, 55), (100, 80)];
    assert!((1..=4).contains(&version), "versions one to four are tabulated");
    let (n, k) = BLOCKS[version - 1];
    ReedSolomon::new(n, k)
}

/// `RS(32, 28)`, the outer code of the cross-interleaved scheme on a compact
/// disc and its descendants.
#[must_use]
pub fn rs_dvd() -> ReedSolomon {
    ReedSolomon::new(32, 28)
}

/// A binary BCH code: cyclic, with a designed distance, over `GF(2^m)`.
///
/// The generator is the least common multiple of the minimal polynomials of
/// `alpha^1` through `alpha^(2t)`. Those `2t` consecutive roots force a
/// distance of at least `2t + 1` by the BCH bound, which is what "designed
/// distance" means -- the true distance can be larger, and often is.
#[derive(Debug, Clone)]
pub struct BchCode {
    /// The field degree, so the length is `2^m - 1`.
    pub m: u32,
    /// The designed error correction capability.
    pub t: usize,
    /// Block length, `2^m - 1`.
    pub n: usize,
    /// Dimension, `n` minus the generator's degree.
    pub k: usize,
    /// The generator polynomial over `GF(2)`, constant term first.
    pub generator: Vec<u8>,
}

impl BchCode {
    /// The binary BCH code of length `2^m - 1` correcting `t` errors.
    ///
    /// # Panics
    /// Panics unless `m` is between three and ten and the designed distance
    /// leaves a positive dimension.
    #[must_use]
    pub fn new(m: u32, t: usize) -> Self {
        assert!((3..=10).contains(&m), "the degree must be between three and ten");
        let n = (1usize << m) - 1;
        let f = Gf2m::with_degree(m);
        // The union of the cyclotomic cosets of 1 through 2t, as one product
        // of distinct minimal polynomials.
        let mut used = vec![false; n];
        let mut gen = vec![1u8];
        for i in 1..=2 * t {
            if used[i % n] {
                continue;
            }
            // Mark the whole coset so its minimal polynomial is used once.
            let mut c = i % n;
            loop {
                used[c] = true;
                c = (2 * c) % n;
                if c == i % n {
                    break;
                }
            }
            let min = f.minimal_polynomial(i as u64);
            let min8: Vec<u8> = min.iter().map(|&x| x as u8).collect();
            gen = gf2_poly_mul(&gen, &min8);
        }
        let k = n.checked_sub(gen.len() - 1).expect("the generator fits");
        assert!(k > 0, "the designed distance leaves no dimension");
        BchCode { m, t, n, k, generator: gen }
    }

    /// Encodes `k` message bits into `n`, systematically.
    ///
    /// # Panics
    /// Panics unless the message has exactly `k` bits.
    #[must_use]
    pub fn encode(&self, msg: &[bool]) -> Vec<bool> {
        assert_eq!(msg.len(), self.k, "the message must have exactly k bits");
        // Message in the high positions, remainder in the low ones.
        let mut shifted = vec![0u8; self.n - self.k];
        shifted.extend(msg.iter().map(|&b| u8::from(b)));
        let rem = gf2_poly_rem(&shifted, &self.generator);
        let mut out: Vec<bool> = (0..self.n - self.k)
            .map(|i| rem.get(i).copied().unwrap_or(0) == 1)
            .collect();
        out.extend(msg.iter().copied());
        out
    }

    /// Decodes a received word by the same syndrome route Reed-Solomon uses,
    /// carried out in `GF(2^m)`.
    ///
    /// # Errors
    /// Returns [`TooManyErrors`] if the errors exceed the designed
    /// capability.
    ///
    /// # Panics
    /// Panics unless the word has exactly `n` bits.
    pub fn decode(&self, recv: &[bool]) -> Result<(Vec<bool>, usize), TooManyErrors> {
        assert_eq!(recv.len(), self.n, "the word must have exactly n bits");
        let f = Gf2m::with_degree(self.m);
        let syn: Vec<u64> = (1..=2 * self.t)
            .map(|i| {
                // Evaluate the received polynomial at alpha^i, with position
                // j carrying x^j.
                let mut acc = 0u64;
                for (j, &b) in recv.iter().enumerate() {
                    if b {
                        acc ^= f.pow(2, (i * j) as u64 % (self.n as u64));
                    }
                }
                acc
            })
            .collect();
        if syn.iter().all(|&s| s == 0) {
            return Ok((recv[self.n - self.k..].to_vec(), 0));
        }
        let locator = bch_berlekamp_massey(&f, &syn);
        let degree = locator.len() - 1;
        if degree > self.t {
            return Err(TooManyErrors);
        }
        // Chien search: a root at alpha^-j means position j is wrong.
        let mut fixed = recv.to_vec();
        let mut found = 0;
        for j in 0..self.n {
            let x = f.pow(2, ((self.n - j % self.n) % self.n) as u64);
            let mut acc = 0u64;
            for (i, &c) in locator.iter().enumerate() {
                acc ^= f.mul(c, f.pow(x, i as u64));
            }
            if acc == 0 {
                fixed[j] = !fixed[j];
                found += 1;
            }
        }
        if found != degree {
            return Err(TooManyErrors);
        }
        Ok((fixed[self.n - self.k..].to_vec(), found))
    }
}

/// Berlekamp-Massey over `GF(2^m)`, returning the locator constant term
/// first.
fn bch_berlekamp_massey(f: &Gf2m, syn: &[u64]) -> Vec<u64> {
    let mut c = vec![1u64];
    let mut b = vec![1u64];
    let mut l = 0usize;
    let mut m = 1usize;
    let mut bb = 1u64;
    for i in 0..syn.len() {
        let mut d = syn[i];
        for j in 1..=l {
            if j < c.len() {
                d ^= f.mul(c[j], syn[i - j]);
            }
        }
        if d == 0 {
            m += 1;
        } else {
            let scale = f.mul(d, f.inv(bb));
            let t = c.clone();
            if c.len() < b.len() + m {
                c.resize(b.len() + m, 0);
            }
            for (j, &x) in b.iter().enumerate() {
                c[j + m] ^= f.mul(scale, x);
            }
            if 2 * l <= i {
                l = i + 1 - l;
                b = t;
                bb = d;
                m = 1;
            } else {
                m += 1;
            }
        }
    }
    c.truncate(l + 1);
    while c.len() > 1 && *c.last().expect("non-empty") == 0 {
        c.pop();
    }
    c
}

/// Polynomial product over `GF(2)`, constant term first.
fn gf2_poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            out[i + j] ^= x & y;
        }
    }
    out
}

/// Polynomial remainder over `GF(2)`, constant term first.
fn gf2_poly_rem(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut r = a.to_vec();
    let bd = b.len() - 1;
    if r.len() <= bd {
        return r;
    }
    for i in (bd..r.len()).rev() {
        if r[i] == 1 {
            for (j, &c) in b.iter().enumerate() {
                r[i - bd + j] ^= c;
            }
        }
    }
    r.truncate(bd);
    r
}

/// The generator polynomials of every binary cyclic code of length `n`, as
/// the divisors of `x^n - 1` over `GF(2)`.
///
/// A cyclic code of length `n` is exactly an ideal in `GF(2)[x] / (x^n - 1)`,
/// and every such ideal is generated by a divisor of `x^n - 1`. So the
/// cyclic codes of a given length are in bijection with those divisors, and
/// listing them lists the codes. Returned constant term first.
///
/// # Panics
/// Panics unless `n` is odd and at most 31 -- an even `n` makes `x^n - 1`
/// non-squarefree in characteristic two, and the enumeration is exponential.
#[must_use]
pub fn cyclic_code_generators(n: usize) -> Vec<Vec<u8>> {
    assert!(n % 2 == 1 && n <= 31, "n must be odd and at most 31");
    // x^n - 1 factors into the minimal polynomials of the cyclotomic cosets,
    // and every divisor is a product of a subset of them.
    let mut cosets: Vec<Vec<usize>> = Vec::new();
    let mut seen = vec![false; n];
    for i in 0..n {
        if seen[i] {
            continue;
        }
        let mut c = i;
        let mut coset = Vec::new();
        loop {
            seen[c] = true;
            coset.push(c);
            c = (2 * c) % n;
            if c == i {
                break;
            }
        }
        cosets.push(coset);
    }
    // The minimal polynomial of each coset, over the smallest field holding
    // an n-th root of unity.
    let m = (1..=16u32)
        .find(|&m| ((1usize << m) - 1).is_multiple_of(n))
        .expect("some degree works");
    let f = Gf2m::with_degree(m);
    let step = ((1usize << m) - 1) / n;
    let factors: Vec<Vec<u8>> = cosets
        .iter()
        .map(|c| {
            let e = (c[0] * step) as u64;
            f.minimal_polynomial(e).iter().map(|&x| x as u8).collect()
        })
        .collect();
    let mut out = Vec::new();
    for mask in 0..1u32 << factors.len() {
        let mut g = vec![1u8];
        for (i, factor) in factors.iter().enumerate() {
            if mask & (1 << i) != 0 {
                g = gf2_poly_mul(&g, factor);
            }
        }
        out.push(g);
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn byte(rng: &mut Rng) -> u8 {
        (rng.next_u64() & 0xFF) as u8
    }

    /// Distinct positions in `0..n`.
    fn positions(rng: &mut Rng, n: usize, count: usize) -> Vec<usize> {
        let mut s = std::collections::BTreeSet::new();
        while s.len() < count {
            s.insert(pick(rng, n));
        }
        s.into_iter().collect()
    }

    /// `GF(256)` satisfies the field axioms, exhaustively where that is
    /// affordable and on the structure where it is not.
    #[test]
    fn gf256_is_a_field() {
        let gf = Gf256::default();
        // The powers of the primitive element run through every non-zero
        // element exactly once before returning. That is what "primitive"
        // means and what makes the logarithm table well defined.
        let mut seen = vec![false; 256];
        for e in 0..255 {
            let x = gf.alpha(e);
            assert!(x != 0, "a power of alpha is zero");
            assert!(!seen[x as usize], "alpha^{e} repeats an earlier power");
            seen[x as usize] = true;
        }
        assert_eq!(gf.alpha(255), 1, "the order of alpha is not 255");
        assert!(seen.iter().skip(1).all(|&b| b), "the powers miss a non-zero element");

        // log and exp invert each other.
        for x in 1..=255u8 {
            assert_eq!(gf.exp[gf.log[x as usize] as usize], x);
        }
        // Multiplication is commutative and has an identity; inverses exist.
        for a in 0..=255u8 {
            assert_eq!(gf.mul(a, 1), a);
            assert_eq!(gf.mul(a, 0), 0);
            if a != 0 {
                assert_eq!(gf.mul(a, gf.inv(a)), 1, "{a} has the wrong inverse");
                assert_eq!(gf.div(a, a), 1);
                assert_eq!(gf.pow(a, 255), 1, "Lagrange fails at {a}");
                assert_eq!(gf.pow(a, -1), gf.inv(a));
            }
            for b in 0..=255u8 {
                assert_eq!(gf.mul(a, b), gf.mul(b, a), "not commutative at ({a}, {b})");
                if b != 0 {
                    assert_eq!(gf.mul(gf.div(a, b), b), a, "division does not undo multiplication");
                }
            }
        }
        // Associativity and distributivity, on a sample: the exhaustive
        // triple loop is sixteen million products and says nothing more.
        let mut rng = Rng::new(0x_6F25);
        for _ in 0..20_000 {
            let (a, b, c) = (byte(&mut rng), byte(&mut rng), byte(&mut rng));
            assert_eq!(gf.mul(gf.mul(a, b), c), gf.mul(a, gf.mul(b, c)));
            assert_eq!(gf.mul(a, b ^ c), gf.mul(a, b) ^ gf.mul(a, c));
            assert_eq!(Gf256::add(a, b), a ^ b);
            assert_eq!(Gf256::add(Gf256::add(a, b), b), a, "addition is not its own inverse");
        }
        // A non-primitive polynomial is rejected rather than quietly giving a
        // broken table.
        assert!(std::panic::catch_unwind(|| Gf256::new(0x100)).is_err());
    }

    /// Polynomial arithmetic over the field: the remainder really is one, and
    /// evaluation is a ring homomorphism.
    #[test]
    fn gf256_polynomial_arithmetic_is_consistent() {
        let gf = Gf256::default();
        let mut rng = Rng::new(0x_0001);
        for _ in 0..500 {
            let da = 1 + pick(&mut rng, 12);
            let db = 1 + pick(&mut rng, 6);
            let a: Vec<u8> = (0..da).map(|_| byte(&mut rng)).collect();
            let mut b: Vec<u8> = (0..db).map(|_| byte(&mut rng)).collect();
            if b[0] == 0 {
                b[0] = 1;
            }
            // Evaluation commutes with multiplication.
            let prod = gf.poly_mul(&a, &b);
            for _ in 0..4 {
                let x = byte(&mut rng);
                assert_eq!(
                    gf.poly_eval(&prod, x),
                    gf.mul(gf.poly_eval(&a, x), gf.poly_eval(&b, x)),
                    "evaluation is not multiplicative"
                );
            }
            // a = q b + r with deg r < deg b, checked by rebuilding a from
            // the remainder: a - r must be divisible, so its remainder is
            // zero.
            let r = gf.poly_rem(&a, &b);
            assert!(r.len() < b.len(), "the remainder has too high a degree");
            let mut diff = a.clone();
            let off = diff.len() - r.len();
            for (i, &c) in r.iter().enumerate() {
                diff[off + i] ^= c;
            }
            assert!(gf.poly_rem(&diff, &b).is_empty(), "the remainder does not divide out");
        }
    }

    /// `GF(2^m)` for every degree it supports: a field, with a trace landing
    /// in `GF(2)` and minimal polynomials that vanish where they should.
    #[test]
    fn gf2m_is_a_field_with_a_binary_trace() {
        for m in 2..=8u32 {
            let f = Gf2m::with_degree(m);
            let n = f.order();
            assert_eq!(f.all_elements().len(), n as usize);
            // Two is the primitive element `x`, so its powers cover the
            // non-zero elements.
            let mut seen = vec![false; n as usize];
            let mut x = 1u64;
            for _ in 0..n - 1 {
                assert!(!seen[x as usize], "the powers of x repeat in GF(2^{m})");
                seen[x as usize] = true;
                x = f.mul(x, 2);
            }
            assert_eq!(x, 1, "x does not have order 2^{m} - 1");

            let mut zero_trace = 0;
            for a in 0..n {
                assert_eq!(f.mul(a, 1), a);
                assert_eq!(f.mul(a, 0), 0);
                if a != 0 {
                    assert_eq!(f.mul(a, f.inv(a)), 1, "{a} has the wrong inverse in GF(2^{m})");
                    assert_eq!(f.pow(a, n - 1), 1, "Lagrange fails at {a} in GF(2^{m})");
                }
                let t = f.trace(a);
                assert!(t == 0 || t == 1, "the trace of {a} is {t}, outside GF(2)");
                zero_trace += usize::from(t == 0);
                for b in 0..n.min(64) {
                    assert_eq!(f.mul(a, b), f.mul(b, a));
                    // The trace is additive, which is what makes it linear
                    // over the prime subfield.
                    assert_eq!(f.trace(a ^ b), f.trace(a) ^ f.trace(b));
                }
            }
            // A surjective linear map onto GF(2) splits the field in half.
            assert_eq!(zero_trace, (n / 2) as usize, "the trace is not balanced in GF(2^{m})");

            // Minimal polynomials: each vanishes at its own root, has binary
            // coefficients, and has degree dividing m.
            for e in 1..n {
                let p = f.minimal_polynomial(e);
                assert!(p.iter().all(|&c| c <= 1), "a minimal polynomial left GF(2)");
                let deg = p.len() - 1;
                assert!((m as u64).is_multiple_of(deg as u64), "degree {deg} does not divide {m}");
                let root = f.pow(2, e);
                let mut acc = 0u64;
                for (i, &c) in p.iter().enumerate() {
                    if c == 1 {
                        acc ^= f.pow(root, i as u64);
                    }
                }
                assert_eq!(acc, 0, "the minimal polynomial of alpha^{e} does not vanish there");
            }
        }
        assert!(std::panic::catch_unwind(|| Gf2m::new(4, 0x1F)).is_err());
    }

    /// A prime field, with inverses from Fermat's little theorem.
    #[test]
    fn gfp_is_a_field() {
        let mut rng = Rng::new(0x_9F97);
        for p in [2u64, 3, 7, 97, 65537, 1_000_000_007] {
            let f = GfP::new(p);
            for _ in 0..200 {
                let a = rng.next_u64() % p;
                let b = rng.next_u64() % p;
                let c = rng.next_u64() % p;
                assert_eq!(f.add(a, b), f.add(b, a));
                assert_eq!(f.mul(a, b), f.mul(b, a));
                assert_eq!(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)));
                assert_eq!(f.mul(a, f.add(b, c)), f.add(f.mul(a, b), f.mul(a, c)));
                assert_eq!(f.sub(f.add(a, b), b), a);
                if a != 0 {
                    assert_eq!(f.mul(a, f.inv(a)), 1, "{a} has the wrong inverse modulo {p}");
                    assert_eq!(f.pow(a, p - 1), 1, "Fermat fails at {a} modulo {p}");
                }
            }
        }
        assert!(std::panic::catch_unwind(|| GfP::new(91)).is_err());
    }

    /// The codes are systematic, their syndromes vanish exactly on codewords,
    /// and every error pattern within the capacity is corrected exactly.
    #[test]
    fn reed_solomon_corrects_up_to_its_capacity() {
        let mut rng = Rng::new(0x_5201);
        for (n, k) in [(15usize, 11usize), (31, 21), (32, 28), (63, 55), (100, 80), (255, 223)] {
            let rs = ReedSolomon::new(n, k);
            let t = rs.correction_capacity();
            assert_eq!(rs.distance(), n - k + 1, "not maximum distance separable");
            for _ in 0..8 {
                let msg: Vec<u8> = (0..k).map(|_| byte(&mut rng)).collect();
                let code = rs.encode(&msg);
                assert_eq!(code.len(), n);
                assert_eq!(&code[..k], &msg[..], "the encoding is not systematic");
                assert!(rs.syndromes(&code).iter().all(|&s| s == 0), "a codeword has a syndrome");
                assert_eq!(rs.decode(&code), Ok((msg.clone(), 0)));

                for errors in 1..=t {
                    let mut recv = code.clone();
                    let where_ = positions(&mut rng, n, errors);
                    for &i in &where_ {
                        // A non-zero change, so the error count is exact.
                        let mut delta = byte(&mut rng);
                        if delta == 0 {
                            delta = 1;
                        }
                        recv[i] ^= delta;
                    }
                    let (got, fixed) = rs
                        .decode(&recv)
                        .unwrap_or_else(|_| panic!("RS({n}, {k}) failed on {errors} errors"));
                    assert_eq!(got, msg, "RS({n}, {k}) mis-decoded {errors} errors");
                    assert_eq!(fixed, errors, "RS({n}, {k}) reported the wrong error count");
                }
            }
        }
    }

    /// The roadmap's headline: the CCSDS code corrects sixteen random byte
    /// errors in a 255-byte frame, over and over.
    #[test]
    fn ccsds_corrects_sixteen_byte_errors() {
        let rs = rs_ccsds();
        assert_eq!((rs.n, rs.k), (255, 223));
        assert_eq!(rs.correction_capacity(), 16);
        let mut rng = Rng::new(0x_CCD5);
        for _ in 0..40 {
            let msg: Vec<u8> = (0..223).map(|_| byte(&mut rng)).collect();
            let mut recv = rs.encode(&msg);
            for i in positions(&mut rng, 255, 16) {
                recv[i] = byte(&mut rng);
            }
            let (got, _) = rs.decode(&recv).expect("sixteen errors are within capacity");
            assert_eq!(got, msg);
        }
        // Its siblings have the parameters they are named for.
        assert_eq!((rs_dvd().n, rs_dvd().k), (32, 28));
        assert_eq!((rs_qr_code(1).n, rs_qr_code(1).k), (26, 19));
        assert_eq!((rs_qr_code(4).n, rs_qr_code(4).k), (100, 80));
    }

    /// What Reed-Solomon is actually deployed for: a burst of corrupted bits
    /// confined to a few symbols is one error per symbol, however many bits
    /// it flipped.
    #[test]
    fn a_burst_costs_one_error_per_symbol_it_touches() {
        let rs = ReedSolomon::new(255, 223);
        let t = rs.correction_capacity();
        let mut rng = Rng::new(0x_B025);
        for _ in 0..20 {
            let msg: Vec<u8> = (0..223).map(|_| byte(&mut rng)).collect();
            let code = rs.encode(&msg);
            // A contiguous run of 16 bytes, every bit of it inverted: 128
            // flipped bits, which no bit-level code of this rate could
            // survive, and 16 symbol errors, which this one corrects exactly.
            let start = pick(&mut rng, 255 - t);
            let mut recv = code.clone();
            let mut flipped_bits = 0;
            for i in start..start + t {
                flipped_bits += (recv[i] ^ 0xFF).count_ones() + recv[i].count_ones();
                recv[i] = !recv[i];
            }
            assert_eq!(flipped_bits, 8 * t as u32, "the burst did not invert every bit");
            let (got, fixed) = rs.decode(&recv).expect("a burst of t symbols is correctable");
            assert_eq!(got, msg);
            assert_eq!(fixed, t);
        }
    }

    /// Erasures cost half what errors do, and the code recovers from as many
    /// erasures as it has parity symbols -- which is the maximum distance
    /// separable property stated operationally: any `k` symbols determine
    /// the codeword.
    #[test]
    fn erasures_cost_half_an_error() {
        let mut rng = Rng::new(0x_E245);
        for (n, k) in [(15usize, 9usize), (31, 21), (63, 47), (255, 223)] {
            let rs = ReedSolomon::new(n, k);
            for _ in 0..8 {
                let msg: Vec<u8> = (0..k).map(|_| byte(&mut rng)).collect();
                let code = rs.encode(&msg);
                // Exactly n - k erasures: the most the code can take, and
                // twice what it could take as errors.
                let lost = positions(&mut rng, n, n - k);
                let mut recv = code.clone();
                for &i in &lost {
                    recv[i] = byte(&mut rng);
                }
                let got = rs
                    .decode_erasures(&recv, &lost)
                    .unwrap_or_else(|_| panic!("RS({n}, {k}) failed on {} erasures", n - k));
                assert_eq!(got, msg, "RS({n}, {k}) mis-decoded its erasures");
                // The same corruption without the position information is
                // beyond the error-correcting capacity, so it must not be
                // silently accepted as some other message.
                if n - k > 2 * rs.correction_capacity() || lost.len() > rs.correction_capacity() {
                    match rs.decode(&recv) {
                        Err(TooManyErrors) => {}
                        Ok((other, _)) => assert_ne!(
                            other, msg,
                            "blind decoding recovered more than the capacity allows"
                        ),
                    }
                }
            }
            // One erasure too many has no unique answer.
            let msg: Vec<u8> = (0..k).map(|_| byte(&mut rng)).collect();
            let recv = rs.encode(&msg);
            let too_many: Vec<usize> = (0..n - k + 1).collect();
            assert_eq!(rs.decode_erasures(&recv, &too_many), Err(TooManyErrors));
        }
    }

    /// Two codewords differ in at least `n - k + 1` places, which is the
    /// Singleton bound met with equality.
    #[test]
    fn reed_solomon_meets_the_singleton_bound() {
        let mut rng = Rng::new(0x_51E7);
        for (n, k) in [(15usize, 11usize), (15, 7), (31, 21), (63, 55)] {
            let rs = ReedSolomon::new(n, k);
            let want = n - k + 1;
            let mut seen_exactly = false;
            for _ in 0..300 {
                let a: Vec<u8> = (0..k).map(|_| byte(&mut rng)).collect();
                let mut b = a.clone();
                // Change one symbol, which gives the lightest difference the
                // code allows and so is where the bound is attained.
                let i = pick(&mut rng, k);
                let mut delta = byte(&mut rng);
                if delta == 0 {
                    delta = 1;
                }
                b[i] ^= delta;
                let (ca, cb) = (rs.encode(&a), rs.encode(&b));
                let dist = ca.iter().zip(&cb).filter(|(x, y)| x != y).count();
                assert!(dist >= want, "RS({n}, {k}) has two codewords {dist} apart");
                if dist == want {
                    seen_exactly = true;
                }
            }
            assert!(seen_exactly, "RS({n}, {k}) never attained its own distance");
        }
    }

    /// Past the capacity, the decoder must never return a word that is not a
    /// codeword: it either corrects to something, or says it cannot.
    #[test]
    fn beyond_the_capacity_the_decoder_fails_rather_than_lies() {
        let mut rng = Rng::new(0x_B340);
        let rs = ReedSolomon::new(31, 21);
        let t = rs.correction_capacity();
        let mut refused = 0;
        let mut miscorrected = 0;
        for _ in 0..400 {
            let msg: Vec<u8> = (0..21).map(|_| byte(&mut rng)).collect();
            let code = rs.encode(&msg);
            let mut recv = code.clone();
            let count = t + 1 + pick(&mut rng, 3);
            for i in positions(&mut rng, 31, count) {
                let mut delta = byte(&mut rng);
                if delta == 0 {
                    delta = 1;
                }
                recv[i] ^= delta;
            }
            match rs.correct(&recv) {
                Err(TooManyErrors) => refused += 1,
                Ok((word, _)) => {
                    assert!(
                        rs.syndromes(&word).iter().all(|&s| s == 0),
                        "the decoder returned a non-codeword"
                    );
                    if word != code {
                        miscorrected += 1;
                    }
                }
            }
        }
        assert!(refused > 0, "the decoder never refused an uncorrectable word");
        // Miscorrection is expected, not a defect: past the radius the
        // received word can genuinely be nearer some other codeword.
        assert!(refused + miscorrected > 300, "too many of these decoded as if clean");
    }

    /// BCH codes have their tabulated parameters, their generator divides
    /// `x^n - 1`, and they correct up to their designed capability.
    #[test]
    fn bch_codes_decode_to_their_designed_distance() {
        // (m, t) against the classical (n, k) table.
        let table = [
            (4u32, 1usize, 15usize, 11usize),
            (4, 2, 15, 7),
            (4, 3, 15, 5),
            (5, 1, 31, 26),
            (5, 2, 31, 21),
            (5, 3, 31, 16),
            (6, 1, 63, 57),
            (6, 2, 63, 51),
            (6, 3, 63, 45),
        ];
        let mut rng = Rng::new(0x_BC40);
        for (m, t, n, k) in table {
            let c = BchCode::new(m, t);
            assert_eq!((c.n, c.k), (n, k), "BCH({m}, {t}) has the wrong parameters");
            assert_eq!(c.generator.len() - 1, n - k, "the generator has the wrong degree");
            // A cyclic code's generator divides x^n - 1.
            let mut xn = vec![0u8; n + 1];
            xn[0] = 1;
            xn[n] = 1;
            assert!(
                gf2_poly_rem(&xn, &c.generator).iter().all(|&x| x == 0),
                "the BCH({m}, {t}) generator does not divide x^n - 1"
            );

            for _ in 0..6 {
                let msg: Vec<bool> = (0..k).map(|_| rng.next_u64() & 1 == 1).collect();
                let code = c.encode(&msg);
                assert_eq!(code.len(), n);
                assert_eq!(&code[n - k..], &msg[..], "the encoding is not systematic");
                assert_eq!(c.decode(&code), Ok((msg.clone(), 0)));
                for errors in 1..=t {
                    let mut recv = code.clone();
                    for i in positions(&mut rng, n, errors) {
                        recv[i] = !recv[i];
                    }
                    let (got, fixed) = c
                        .decode(&recv)
                        .unwrap_or_else(|_| panic!("BCH({m}, {t}) failed on {errors} errors"));
                    assert_eq!(got, msg, "BCH({m}, {t}) mis-decoded {errors} errors");
                    assert_eq!(fixed, errors);
                }
            }
        }
    }

    /// The cyclic codes of a given length are exactly the divisors of
    /// `x^n - 1`, so enumerating those enumerates the codes.
    #[test]
    fn cyclic_generators_are_the_divisors_of_x_n_minus_one() {
        for n in [3usize, 5, 7, 9, 15, 21, 31] {
            let gens = cyclic_code_generators(n);
            let mut xn = vec![0u8; n + 1];
            xn[0] = 1;
            xn[n] = 1;
            for g in &gens {
                assert!(
                    gf2_poly_rem(&xn, g).iter().all(|&x| x == 0),
                    "a returned generator of degree {} does not divide x^{n} - 1",
                    g.len() - 1
                );
            }
            // The count is two to the power of the number of cyclotomic
            // cosets, since a divisor is a choice of subset of the
            // irreducible factors.
            let mut seen = vec![false; n];
            let mut cosets = 0;
            for i in 0..n {
                if seen[i] {
                    continue;
                }
                cosets += 1;
                let mut c = i;
                loop {
                    seen[c] = true;
                    c = (2 * c) % n;
                    if c == i {
                        break;
                    }
                }
            }
            assert_eq!(gens.len(), 1 << cosets, "the wrong number of divisors for n = {n}");
            // The trivial ones are there: the whole space and the repetition
            // code.
            assert!(gens.contains(&vec![1u8]), "the generator 1 is missing");
            assert!(gens.contains(&xn[..n].iter().map(|_| 1u8).collect::<Vec<u8>>().to_vec())
                || gens.iter().any(|g| g.len() == n && g.iter().all(|&c| c == 1)),
                "the all-ones generator is missing");
        }
    }
}
