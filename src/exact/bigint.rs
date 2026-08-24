//! Arbitrary-precision signed integers.
//!
//! Magnitudes are little-endian vectors of `u64` limbs in base 2^64, held
//! in a canonical form: no trailing zero limbs, and the limb vector is
//! empty exactly when the value is zero. Every operation restores that
//! form, so equality is structural and `is_zero` is a length check.

use crate::error::GeomError;
use crate::monte_carlo::Rng;
use std::cmp::Ordering;

/// An arbitrary-precision signed integer.
///
/// `sign` is -1, 0 or +1, and is 0 if and only if `limbs` is empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigInt {
    pub sign: i8,
    pub limbs: Vec<u64>,
}

// ---------------------------------------------------------------------------
// limb-vector helpers (unsigned magnitudes, little-endian)
// ---------------------------------------------------------------------------

/// Drop trailing zero limbs so a magnitude has a unique representation.
fn trim(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

/// Compare two magnitudes.
fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    if a.len() != b.len() {
        return a.len().cmp(&b.len());
    }
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

/// Magnitude sum.
fn add_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(long.len() + 1);
    let mut carry = 0u64;
    for i in 0..long.len() {
        let (mut s, mut c) = long[i].overflowing_add(carry);
        carry = u64::from(c);
        if i < short.len() {
            let (s2, c2) = s.overflowing_add(short[i]);
            s = s2;
            c = c2;
            carry += u64::from(c);
        }
        out.push(s);
    }
    if carry != 0 {
        out.push(carry);
    }
    out
}

/// Magnitude difference, requiring `a >= b`.
fn sub_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    debug_assert!(cmp_mag(a, b) != Ordering::Less);
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0u64;
    for i in 0..a.len() {
        let bi = if i < b.len() { b[i] } else { 0 };
        let (d, b1) = a[i].overflowing_sub(bi);
        let (d, b2) = d.overflowing_sub(borrow);
        borrow = u64::from(b1) + u64::from(b2);
        out.push(d);
    }
    trim(&mut out);
    out
}

/// Schoolbook product, O(n*m).
fn mul_mag_school(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u64; a.len() + b.len()];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        let mut carry = 0u128;
        for (j, &bj) in b.iter().enumerate() {
            let cur = u128::from(out[i + j]) + u128::from(ai) * u128::from(bj) + carry;
            out[i + j] = cur as u64;
            carry = cur >> 64;
        }
        let mut k = i + b.len();
        while carry != 0 {
            let cur = u128::from(out[k]) + carry;
            out[k] = cur as u64;
            carry = cur >> 64;
            k += 1;
        }
    }
    trim(&mut out);
    out
}

/// Shift a magnitude left by whole limbs.
fn shl_limbs(a: &[u64], k: usize) -> Vec<u64> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u64; k];
    out.extend_from_slice(a);
    out
}

/// The Karatsuba threshold, in limbs, quoted by the roadmap.
const KARATSUBA_LIMBS: usize = 32;

/// Magnitude product: schoolbook for small inputs, Karatsuba above
/// [`KARATSUBA_LIMBS`].
fn mul_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    if a.len() < KARATSUBA_LIMBS || b.len() < KARATSUBA_LIMBS {
        return mul_mag_school(a, b);
    }
    // Split both operands at the same point so the recombination shifts
    // line up: a = a0 + a1 B^h, b = b0 + b1 B^h.
    let h = a.len().max(b.len()) / 2;
    let (a0, a1) = a.split_at(h.min(a.len()));
    let (b0, b1) = b.split_at(h.min(b.len()));
    let z0 = mul_mag(a0, b0);
    let z2 = mul_mag(a1, b1);
    // z1 = (a0+a1)(b0+b1) - z0 - z2
    let sa = add_mag(a0, a1);
    let sb = add_mag(b0, b1);
    let mut z1 = mul_mag(&sa, &sb);
    z1 = sub_mag(&z1, &z0);
    z1 = sub_mag(&z1, &z2);
    let mut out = add_mag(&z0, &shl_limbs(&z1, h));
    out = add_mag(&out, &shl_limbs(&z2, 2 * h));
    trim(&mut out);
    out
}

/// Shift a magnitude left by `bits`.
fn shl_bits(a: &[u64], bits: usize) -> Vec<u64> {
    if a.is_empty() {
        return Vec::new();
    }
    let limb = bits / 64;
    let off = bits % 64;
    let mut out = vec![0u64; limb];
    if off == 0 {
        out.extend_from_slice(a);
    } else {
        let mut carry = 0u64;
        for &x in a {
            out.push((x << off) | carry);
            carry = x >> (64 - off);
        }
        if carry != 0 {
            out.push(carry);
        }
    }
    trim(&mut out);
    out
}

/// Shift a magnitude right by `bits`, discarding the low bits.
fn shr_bits(a: &[u64], bits: usize) -> Vec<u64> {
    let limb = bits / 64;
    if limb >= a.len() {
        return Vec::new();
    }
    let off = bits % 64;
    let src = &a[limb..];
    let mut out = Vec::with_capacity(src.len());
    if off == 0 {
        out.extend_from_slice(src);
    } else {
        for i in 0..src.len() {
            let hi = if i + 1 < src.len() { src[i + 1] } else { 0 };
            out.push((src[i] >> off) | (hi << (64 - off)));
        }
    }
    trim(&mut out);
    out
}

/// Divide a magnitude by a single limb, returning (quotient, remainder).
fn divrem_small(a: &[u64], d: u64) -> (Vec<u64>, u64) {
    let mut q = vec![0u64; a.len()];
    let mut rem = 0u128;
    for i in (0..a.len()).rev() {
        let cur = (rem << 64) | u128::from(a[i]);
        q[i] = (cur / u128::from(d)) as u64;
        rem = cur % u128::from(d);
    }
    trim(&mut q);
    (q, rem as u64)
}

/// Knuth's Algorithm D: (quotient, remainder) of two magnitudes.
///
/// `v` must have at least two limbs; single-limb divisors go through
/// [`divrem_small`].
fn divrem_knuth(u: &[u64], v: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let n = v.len();
    let m = u.len() - n;
    // D1. Normalize so the divisor's top limb has its high bit set, which
    // is what bounds the quotient-digit estimate below.
    let shift = v[n - 1].leading_zeros() as usize;
    let vn = shl_bits(v, shift);
    let mut un = shl_bits(u, shift);
    un.resize(u.len() + 1, 0); // the extra limb Algorithm D indexes as u[j+n]
    let mut q = vec![0u64; m + 1];
    const B: u128 = 1u128 << 64;
    for j in (0..=m).rev() {
        // D3. Estimate the quotient digit from the top two limbs.
        let top = (u128::from(un[j + n]) << 64) | u128::from(un[j + n - 1]);
        let mut qhat = top / u128::from(vn[n - 1]);
        let mut rhat = top % u128::from(vn[n - 1]);
        while qhat >= B
            || qhat * u128::from(vn[n - 2]) > (rhat << 64) | u128::from(un[j + n - 2])
        {
            qhat -= 1;
            rhat += u128::from(vn[n - 1]);
            if rhat >= B {
                break;
            }
        }
        // D4. Multiply and subtract.
        let mut borrow = 0i128;
        let mut carry = 0u128;
        for i in 0..n {
            let p = qhat * u128::from(vn[i]) + carry;
            carry = p >> 64;
            let sub = i128::from(un[i + j]) - (p & (B - 1)) as i128 - borrow;
            if sub < 0 {
                un[i + j] = (sub + B as i128) as u64;
                borrow = 1;
            } else {
                un[i + j] = sub as u64;
                borrow = 0;
            }
        }
        let sub = i128::from(un[j + n]) - carry as i128 - borrow;
        if sub < 0 {
            un[j + n] = (sub + B as i128) as u64;
            borrow = 1;
        } else {
            un[j + n] = sub as u64;
            borrow = 0;
        }
        // D5/D6. The estimate was one too large: add the divisor back.
        if borrow != 0 {
            qhat -= 1;
            let mut carry = 0u64;
            for i in 0..n {
                let s = u128::from(un[i + j]) + u128::from(vn[i]) + u128::from(carry);
                un[i + j] = s as u64;
                carry = (s >> 64) as u64;
            }
            un[j + n] = un[j + n].wrapping_add(carry);
        }
        q[j] = qhat as u64;
    }
    trim(&mut q);
    // D8. Undo the normalizing shift on the remainder.
    let mut r = un[..n].to_vec();
    trim(&mut r);
    (q, shr_bits(&r, shift))
}

// ---------------------------------------------------------------------------
// construction and conversion
// ---------------------------------------------------------------------------

impl BigInt {
    /// Build from a sign and a magnitude, restoring the canonical form.
    fn from_parts(sign: i8, mut limbs: Vec<u64>) -> Self {
        trim(&mut limbs);
        if limbs.is_empty() {
            BigInt { sign: 0, limbs }
        } else {
            BigInt { sign, limbs }
        }
    }

    #[must_use]
    pub fn zero() -> Self {
        BigInt { sign: 0, limbs: Vec::new() }
    }

    #[must_use]
    pub fn one() -> Self {
        BigInt { sign: 1, limbs: vec![1] }
    }

    #[must_use]
    pub fn from_u64(n: u64) -> Self {
        if n == 0 {
            Self::zero()
        } else {
            BigInt { sign: 1, limbs: vec![n] }
        }
    }

    #[must_use]
    pub fn from_i64(n: i64) -> Self {
        match n.cmp(&0) {
            Ordering::Equal => Self::zero(),
            Ordering::Greater => BigInt { sign: 1, limbs: vec![n as u64] },
            // Negate through u64 so i64::MIN does not overflow.
            Ordering::Less => BigInt { sign: -1, limbs: vec![(n as i128).unsigned_abs() as u64] },
        }
    }

    /// Parse in `radix` (2..=36), accepting a leading `+` or `-` and
    /// either case of letter digit.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] for an unsupported radix, an
    /// empty digit string, or an out-of-range character.
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, GeomError> {
        if !(2..=36).contains(&radix) {
            return Err(GeomError::InvalidArgument("radix must be in 2..=36"));
        }
        let (sign, digits) = match s.strip_prefix('-') {
            Some(rest) => (-1i8, rest),
            None => (1i8, s.strip_prefix('+').unwrap_or(s)),
        };
        if digits.is_empty() {
            return Err(GeomError::InvalidArgument("empty integer literal"));
        }
        let mut limbs: Vec<u64> = Vec::new();
        for ch in digits.chars() {
            let d = ch
                .to_digit(radix)
                .ok_or(GeomError::InvalidArgument("digit out of range for radix"))?;
            // limbs = limbs * radix + d
            let mut carry = u128::from(d);
            for limb in &mut limbs {
                let cur = u128::from(*limb) * u128::from(radix) + carry;
                *limb = cur as u64;
                carry = cur >> 64;
            }
            while carry != 0 {
                limbs.push(carry as u64);
                carry >>= 64;
            }
        }
        Ok(Self::from_parts(sign, limbs))
    }

    /// Render in `radix` (2..=36) using lower-case letter digits.
    ///
    /// # Panics
    /// Panics if `radix` is outside 2..=36.
    #[must_use]
    pub fn to_string_radix(&self, radix: u32) -> String {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");
        if self.is_zero() {
            return "0".to_string();
        }
        let mut digits = Vec::new();
        let mut cur = self.limbs.clone();
        while !cur.is_empty() {
            let (q, r) = divrem_small(&cur, u64::from(radix));
            digits.push(std::char::from_digit(r as u32, radix).expect("digit in range"));
            cur = q;
        }
        let mut out = String::with_capacity(digits.len() + 1);
        if self.sign < 0 {
            out.push('-');
        }
        out.extend(digits.iter().rev());
        out
    }

    /// Nearest `f64`, saturating to infinity beyond the exponent range.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        let mut v = 0.0f64;
        for &limb in self.limbs.iter().rev() {
            v = v * 18_446_744_073_709_551_616.0 + limb as f64;
        }
        if self.sign < 0 {
            -v
        } else {
            v
        }
    }

    /// The value as an `i64`, or `None` if it does not fit.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        match self.limbs.len() {
            0 => Some(0),
            1 => {
                let m = self.limbs[0];
                if self.sign > 0 {
                    i64::try_from(m).ok()
                } else if m <= (i64::MAX as u64) + 1 {
                    // -(i64::MAX + 1) is exactly i64::MIN.
                    Some(-(m as i128) as i64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Number of bits in the magnitude; zero has zero bits.
    #[must_use]
    pub fn bits(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => self.limbs.len() * 64 - top.leading_zeros() as usize,
        }
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.sign == 0
    }

    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.sign < 0
    }

    #[must_use]
    pub fn is_even(&self) -> bool {
        self.limbs.first().is_none_or(|l| l % 2 == 0)
    }

    /// Compare magnitudes, ignoring sign.
    #[must_use]
    pub fn cmp_abs(&self, other: &BigInt) -> Ordering {
        cmp_mag(&self.limbs, &other.limbs)
    }

    #[must_use]
    pub fn abs(&self) -> Self {
        BigInt { sign: self.sign.abs(), limbs: self.limbs.clone() }
    }

    #[must_use]
    pub fn neg(&self) -> Self {
        BigInt { sign: -self.sign, limbs: self.limbs.clone() }
    }

    // -- arithmetic ---------------------------------------------------------

    #[must_use]
    pub fn add(&self, other: &BigInt) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        if self.sign == other.sign {
            Self::from_parts(self.sign, add_mag(&self.limbs, &other.limbs))
        } else {
            match cmp_mag(&self.limbs, &other.limbs) {
                Ordering::Equal => Self::zero(),
                Ordering::Greater => {
                    Self::from_parts(self.sign, sub_mag(&self.limbs, &other.limbs))
                }
                Ordering::Less => {
                    Self::from_parts(other.sign, sub_mag(&other.limbs, &self.limbs))
                }
            }
        }
    }

    #[must_use]
    pub fn sub(&self, other: &BigInt) -> Self {
        self.add(&other.neg())
    }

    #[must_use]
    pub fn mul(&self, other: &BigInt) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        Self::from_parts(self.sign * other.sign, mul_mag(&self.limbs, &other.limbs))
    }

    /// Truncated division: the quotient rounds toward zero and the
    /// remainder takes the sign of the dividend, matching Rust's `/` and
    /// `%` on primitive integers.
    ///
    /// # Panics
    /// Panics if `other` is zero.
    #[must_use]
    pub fn div_rem(&self, other: &BigInt) -> (Self, Self) {
        assert!(!other.is_zero(), "division by zero");
        if cmp_mag(&self.limbs, &other.limbs) == Ordering::Less {
            return (Self::zero(), self.clone());
        }
        let (q, r) = if other.limbs.len() == 1 {
            let (q, r) = divrem_small(&self.limbs, other.limbs[0]);
            (q, if r == 0 { Vec::new() } else { vec![r] })
        } else {
            divrem_knuth(&self.limbs, &other.limbs)
        };
        (
            Self::from_parts(self.sign * other.sign, q),
            Self::from_parts(self.sign, r),
        )
    }

    /// Euclidean remainder: always in `0..|m|`.
    ///
    /// # Panics
    /// Panics if `m` is zero.
    #[must_use]
    pub fn rem_euclid(&self, m: &BigInt) -> Self {
        let r = self.div_rem(m).1;
        if r.is_negative() {
            r.add(&m.abs())
        } else {
            r
        }
    }

    /// `self` raised to `e` by binary exponentiation.
    #[must_use]
    pub fn pow(&self, e: u64) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// Modular exponentiation by a 4-bit sliding window, reducing after
    /// every multiply. The result is the least non-negative residue.
    ///
    /// # Panics
    /// Panics if `m` is zero or `e` is negative.
    #[must_use]
    pub fn mod_pow(&self, e: &BigInt, m: &BigInt) -> Self {
        assert!(!m.is_zero(), "modulus must be non-zero");
        assert!(!e.is_negative(), "negative exponent");
        let m_abs = m.abs();
        if m_abs == Self::one() {
            return Self::zero();
        }
        if e.is_zero() {
            return Self::one();
        }
        let base = self.rem_euclid(&m_abs);
        // Odd powers base^1, base^3, .. base^15 for the window.
        const W: usize = 4;
        let base_sq = base.mul(&base).rem_euclid(&m_abs);
        let mut table = vec![base.clone()];
        for i in 1..(1 << (W - 1)) {
            table.push(table[i - 1].mul(&base_sq).rem_euclid(&m_abs));
        }
        let mut result = Self::one();
        let mut i = e.bits() as isize - 1;
        while i >= 0 {
            if !e.bit(i as usize) {
                result = result.mul(&result).rem_euclid(&m_abs);
                i -= 1;
                continue;
            }
            // Longest window ending in a set bit, at most W wide.
            let lo = (i - W as isize + 1).max(0);
            let mut l = lo;
            while !e.bit(l as usize) {
                l += 1;
            }
            let width = (i - l + 1) as usize;
            let mut idx = 0usize;
            for b in (l..=i).rev() {
                idx = (idx << 1) | usize::from(e.bit(b as usize));
            }
            for _ in 0..width {
                result = result.mul(&result).rem_euclid(&m_abs);
            }
            result = result.mul(&table[idx >> 1]).rem_euclid(&m_abs);
            i = l - 1;
        }
        result
    }

    /// Greatest common divisor, always non-negative. `gcd(0, 0)` is 0.
    #[must_use]
    pub fn gcd(&self, other: &BigInt) -> Self {
        let mut a = self.abs();
        let mut b = other.abs();
        while !b.is_zero() {
            let r = a.div_rem(&b).1;
            a = b;
            b = r.abs();
        }
        a
    }

    /// Least common multiple, always non-negative. Zero if either side is
    /// zero.
    #[must_use]
    pub fn lcm(&self, other: &BigInt) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let g = self.gcd(other);
        self.div_rem(&g).0.mul(other).abs()
    }

    /// Extended Euclid: returns `(g, x, y)` with `self*x + other*y == g`
    /// and `g == gcd(self, other) >= 0`.
    #[must_use]
    pub fn extended_gcd(&self, other: &BigInt) -> (Self, Self, Self) {
        let (mut old_r, mut r) = (self.clone(), other.clone());
        let (mut old_s, mut s) = (Self::one(), Self::zero());
        let (mut old_t, mut t) = (Self::zero(), Self::one());
        while !r.is_zero() {
            let q = old_r.div_rem(&r).0;
            let nr = old_r.sub(&q.mul(&r));
            old_r = std::mem::replace(&mut r, nr);
            let ns = old_s.sub(&q.mul(&s));
            old_s = std::mem::replace(&mut s, ns);
            let nt = old_t.sub(&q.mul(&t));
            old_t = std::mem::replace(&mut t, nt);
        }
        if old_r.is_negative() {
            // Keep the gcd non-negative by flipping the whole identity.
            (old_r.neg(), old_s.neg(), old_t.neg())
        } else {
            (old_r, old_s, old_t)
        }
    }

    /// Modular inverse, or `None` when `gcd(self, m) != 1`.
    #[must_use]
    pub fn mod_inverse(&self, m: &BigInt) -> Option<Self> {
        let m_abs = m.abs();
        if m_abs.is_zero() || m_abs == Self::one() {
            return None;
        }
        let (g, x, _) = self.rem_euclid(&m_abs).extended_gcd(&m_abs);
        if g != Self::one() {
            return None;
        }
        Some(x.rem_euclid(&m_abs))
    }
}

// ---------------------------------------------------------------------------
// bit manipulation, roots, randomness, combinatorics
// ---------------------------------------------------------------------------

impl BigInt {
    /// Shift left by `bits`, preserving sign.
    #[must_use]
    pub fn shl(&self, bits: usize) -> Self {
        Self::from_parts(self.sign, shl_bits(&self.limbs, bits))
    }

    /// Shift the magnitude right by `bits`, preserving sign. This
    /// truncates toward zero rather than flooring, so it matches
    /// `div_rem` by a power of two rather than an arithmetic shift.
    #[must_use]
    pub fn shr(&self, bits: usize) -> Self {
        Self::from_parts(self.sign, shr_bits(&self.limbs, bits))
    }

    /// Bit `i` of the magnitude, counting from the least significant.
    #[must_use]
    pub fn bit(&self, i: usize) -> bool {
        let limb = i / 64;
        limb < self.limbs.len() && (self.limbs[limb] >> (i % 64)) & 1 == 1
    }

    /// Set or clear bit `i` of the magnitude.
    pub fn set_bit(&mut self, i: usize, value: bool) {
        let limb = i / 64;
        if value {
            if limb >= self.limbs.len() {
                self.limbs.resize(limb + 1, 0);
            }
            self.limbs[limb] |= 1u64 << (i % 64);
            if self.sign == 0 {
                self.sign = 1;
            }
        } else if limb < self.limbs.len() {
            self.limbs[limb] &= !(1u64 << (i % 64));
            trim(&mut self.limbs);
            if self.limbs.is_empty() {
                self.sign = 0;
            }
        }
    }

    /// Bitwise AND of the magnitudes; the result takes `self`'s sign.
    #[must_use]
    pub fn and(&self, other: &BigInt) -> Self {
        let n = self.limbs.len().min(other.limbs.len());
        let limbs = (0..n).map(|i| self.limbs[i] & other.limbs[i]).collect();
        Self::from_parts(self.sign, limbs)
    }

    /// Bitwise OR of the magnitudes; the result takes `self`'s sign, or
    /// `other`'s when `self` is zero.
    #[must_use]
    pub fn or(&self, other: &BigInt) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let limbs = (0..n)
            .map(|i| {
                self.limbs.get(i).copied().unwrap_or(0) | other.limbs.get(i).copied().unwrap_or(0)
            })
            .collect();
        let sign = if self.sign != 0 { self.sign } else { other.sign };
        Self::from_parts(sign, limbs)
    }

    /// Bitwise XOR of the magnitudes; the result takes `self`'s sign, or
    /// `other`'s when `self` is zero.
    #[must_use]
    pub fn xor(&self, other: &BigInt) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let limbs = (0..n)
            .map(|i| {
                self.limbs.get(i).copied().unwrap_or(0) ^ other.limbs.get(i).copied().unwrap_or(0)
            })
            .collect();
        let sign = if self.sign != 0 { self.sign } else { other.sign };
        Self::from_parts(sign, limbs)
    }

    /// Integer square root: the largest `r` with `r*r <= self`.
    ///
    /// # Panics
    /// Panics if `self` is negative.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        assert!(!self.is_negative(), "sqrt of a negative BigInt");
        if self.is_zero() {
            return Self::zero();
        }
        if self.bits() <= 1 {
            return Self::one();
        }
        // Newton from a power-of-two overestimate: x_{k+1} = (x_k + n/x_k)/2
        // decreases monotonically to floor(sqrt(n)) from above.
        let mut x = Self::one().shl(self.bits() / 2 + 1);
        loop {
            let y = x.add(&self.div_rem(&x).0).shr(1);
            if y.cmp_abs(&x) != Ordering::Less {
                return x;
            }
            x = y;
        }
    }

    /// Integer `n`th root: the largest `r` with `r^n <= self`.
    ///
    /// # Panics
    /// Panics if `n` is zero, or if `self` is negative with even `n`.
    #[must_use]
    pub fn nth_root(&self, n: u32) -> Self {
        assert!(n > 0, "nth_root requires n > 0");
        assert!(!(self.is_negative() && n.is_multiple_of(2)), "even root of a negative BigInt");
        if self.is_zero() {
            return Self::zero();
        }
        if n == 1 {
            return self.clone();
        }
        let neg = self.is_negative();
        let a = self.abs();
        if a.bits() <= 1 {
            return if neg { Self::one().neg() } else { Self::one() };
        }
        let big_n = Self::from_u64(u64::from(n));
        let mut x = Self::one().shl(a.bits() / n as usize + 1);
        loop {
            // x_{k+1} = ((n-1) x_k + a / x_k^(n-1)) / n
            let pow = x.pow(u64::from(n - 1));
            let t = Self::from_u64(u64::from(n - 1))
                .mul(&x)
                .add(&a.div_rem(&pow).0)
                .div_rem(&big_n)
                .0;
            if t.cmp_abs(&x) != Ordering::Less {
                return if neg { x.neg() } else { x };
            }
            x = t;
        }
    }

    #[must_use]
    pub fn is_perfect_square(&self) -> bool {
        if self.is_negative() {
            return false;
        }
        let r = self.sqrt();
        r.mul(&r) == *self
    }

    /// A uniformly random non-negative integer with exactly `bits` bits of
    /// magnitude (the top bit is set), or zero when `bits` is zero.
    #[must_use]
    pub fn random_bits(bits: usize, rng: &mut Rng) -> Self {
        if bits == 0 {
            return Self::zero();
        }
        let n_limbs = bits.div_ceil(64);
        let mut limbs: Vec<u64> = (0..n_limbs).map(|_| rng.next_u64()).collect();
        let top_bits = bits - (n_limbs - 1) * 64;
        if top_bits < 64 {
            limbs[n_limbs - 1] &= (1u64 << top_bits) - 1;
        }
        // Pin the leading bit so the result has exactly `bits` bits.
        limbs[n_limbs - 1] |= 1u64 << (top_bits - 1);
        Self::from_parts(1, limbs)
    }

    /// A uniformly random integer in `0..bound` by rejection sampling.
    ///
    /// # Panics
    /// Panics if `bound` is not positive.
    #[must_use]
    pub fn random_below(bound: &BigInt, rng: &mut Rng) -> Self {
        assert!(bound.sign > 0, "bound must be positive");
        let bits = bound.bits();
        loop {
            // Sample the full bit width, top bit free, and reject high draws.
            let n_limbs = bits.div_ceil(64);
            let mut limbs: Vec<u64> = (0..n_limbs).map(|_| rng.next_u64()).collect();
            let top_bits = bits - (n_limbs - 1) * 64;
            if top_bits < 64 {
                limbs[n_limbs - 1] &= (1u64 << top_bits) - 1;
            }
            let candidate = Self::from_parts(1, limbs);
            if candidate.cmp_abs(bound) == Ordering::Less {
                return candidate;
            }
        }
    }

    /// `n!`.
    #[must_use]
    pub fn factorial(n: u64) -> Self {
        let mut acc = Self::one();
        for k in 2..=n {
            acc = acc.mul(&Self::from_u64(k));
        }
        acc
    }

    /// The binomial coefficient `n choose k`, zero when `k > n`.
    #[must_use]
    pub fn binomial(n: u64, k: u64) -> Self {
        if k > n {
            return Self::zero();
        }
        let k = k.min(n - k);
        let mut acc = Self::one();
        for i in 0..k {
            // acc = acc * (n - i) / (i + 1) stays exact: the product of any
            // i+1 consecutive integers is divisible by (i+1)!.
            acc = acc.mul(&Self::from_u64(n - i));
            acc = acc.div_rem(&Self::from_u64(i + 1)).0;
        }
        acc
    }

    /// The `n`th Fibonacci number by fast doubling, with `F(0) = 0`.
    #[must_use]
    pub fn fibonacci(n: u64) -> Self {
        fn go(n: u64) -> (BigInt, BigInt) {
            if n == 0 {
                return (BigInt::zero(), BigInt::one());
            }
            let (a, b) = go(n >> 1);
            // F(2k) = F(k) * (2 F(k+1) - F(k)), F(2k+1) = F(k)^2 + F(k+1)^2
            let two_b = b.shl(1);
            let c = a.mul(&two_b.sub(&a));
            let d = a.mul(&a).add(&b.mul(&b));
            if n & 1 == 0 {
                (c, d)
            } else {
                let e = c.add(&d);
                (d, e)
            }
        }
        go(n).0
    }
}

// ---------------------------------------------------------------------------
// trait impls
// ---------------------------------------------------------------------------

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        // Signs first; within a sign, magnitudes, reversed when negative.
        match self.sign.cmp(&other.sign) {
            Ordering::Equal => {
                let m = cmp_mag(&self.limbs, &other.limbs);
                if self.sign < 0 {
                    m.reverse()
                } else {
                    m
                }
            }
            other_order => other_order,
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Default for BigInt {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Display for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_radix(10))
    }
}

impl From<i64> for BigInt {
    fn from(n: i64) -> Self {
        Self::from_i64(n)
    }
}

impl From<u64> for BigInt {
    fn from(n: u64) -> Self {
        Self::from_u64(n)
    }
}

macro_rules! binop {
    ($trait:ident, $method:ident, $call:ident) => {
        impl std::ops::$trait for BigInt {
            type Output = BigInt;
            fn $method(self, rhs: BigInt) -> BigInt {
                BigInt::$call(&self, &rhs)
            }
        }
        impl<'a> std::ops::$trait<&'a BigInt> for &'a BigInt {
            type Output = BigInt;
            fn $method(self, rhs: &'a BigInt) -> BigInt {
                BigInt::$call(self, rhs)
            }
        }
    };
}

binop!(Add, add, add);
binop!(Sub, sub, sub);
binop!(Mul, mul, mul);

impl std::ops::Div for BigInt {
    type Output = BigInt;
    fn div(self, rhs: BigInt) -> BigInt {
        self.div_rem(&rhs).0
    }
}

impl std::ops::Rem for BigInt {
    type Output = BigInt;
    fn rem(self, rhs: BigInt) -> BigInt {
        self.div_rem(&rhs).1
    }
}

impl std::ops::Neg for BigInt {
    type Output = BigInt;
    fn neg(self) -> BigInt {
        BigInt::neg(&self)
    }
}

impl std::ops::Shl<usize> for BigInt {
    type Output = BigInt;
    fn shl(self, bits: usize) -> BigInt {
        BigInt::shl(&self, bits)
    }
}

impl std::ops::Shr<usize> for BigInt {
    type Output = BigInt;
    fn shr(self, bits: usize) -> BigInt {
        BigInt::shr(&self, bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> BigInt {
        BigInt::from_str_radix(s, 10).expect("decimal literal")
    }

    /// A magnitude of exactly `n` limbs, deterministic in the seed.
    fn rand_limbs(n: usize, rng: &mut Rng) -> BigInt {
        let mut limbs: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
        limbs[n - 1] |= 1 << 63; // force the full limb count
        BigInt::from_parts(1, limbs)
    }

    #[test]
    fn test_construction_and_string_roundtrip() {
        assert!(BigInt::zero().is_zero());
        assert_eq!(BigInt::zero().sign, 0);
        assert!(BigInt::zero().limbs.is_empty());
        assert_eq!(BigInt::from_i64(0), BigInt::zero());
        assert_eq!(BigInt::from_u64(0), BigInt::zero());
        // i64::MIN negates without overflow.
        let min = BigInt::from_i64(i64::MIN);
        assert_eq!(min.to_i64(), Some(i64::MIN));
        assert_eq!(min.to_string_radix(10), "-9223372036854775808");
        assert_eq!(BigInt::from_i64(i64::MAX).to_i64(), Some(i64::MAX));
        // One past i64::MAX no longer fits.
        assert_eq!(BigInt::from_u64(i64::MAX as u64 + 1).to_i64(), None);

        // Roundtrip through every supported radix, both signs.
        let mut rng = Rng::new(7);
        for radix in 2..=36u32 {
            for _ in 0..8 {
                let mut v = rand_limbs(3, &mut rng);
                if rng.next_u64().is_multiple_of(2) {
                    v = v.neg();
                }
                let s = v.to_string_radix(radix);
                let back = BigInt::from_str_radix(&s, radix).expect("roundtrip parse");
                assert_eq!(back, v, "radix {radix} roundtrip of {s}");
            }
        }
        assert_eq!(big("123456789").to_string_radix(36), "21i3v9");
        assert_eq!(BigInt::from_str_radix("21i3v9", 36).unwrap(), big("123456789"));
        assert_eq!(BigInt::from_str_radix("-ff", 16).unwrap(), BigInt::from_i64(-255));
        assert_eq!(BigInt::from_str_radix("+1010", 2).unwrap(), BigInt::from_i64(10));
        assert_eq!(BigInt::zero().to_string_radix(16), "0");

        // Rejected inputs.
        assert!(BigInt::from_str_radix("12", 1).is_err());
        assert!(BigInt::from_str_radix("12", 37).is_err());
        assert!(BigInt::from_str_radix("", 10).is_err());
        assert!(BigInt::from_str_radix("-", 10).is_err());
        assert!(BigInt::from_str_radix("2", 2).is_err());
    }

    #[test]
    fn test_ordering_and_bits() {
        // Ordering is by value, not by limb layout: negatives reverse.
        let mut v = [big("-300"), big("5"), big("-1"), BigInt::zero(), big("300"), big("-5")];
        v.sort();
        let got: Vec<String> = v.iter().map(BigInt::to_string_radix_10_for_test).collect();
        assert_eq!(got, ["-300", "-5", "-1", "0", "5", "300"]);
        assert!(big("-1000000000000000000000") < big("-999999999999999999999"));
        assert!(big("-1") < BigInt::zero() && BigInt::zero() < big("1"));
        // A longer magnitude with a smaller leading limb is still larger.
        assert!(BigInt::from_parts(1, vec![0, 1]) > BigInt::from_u64(u64::MAX));

        assert_eq!(BigInt::zero().bits(), 0);
        assert_eq!(BigInt::one().bits(), 1);
        assert_eq!(BigInt::from_u64(u64::MAX).bits(), 64);
        assert_eq!(BigInt::one().shl(255).bits(), 256);
        assert_eq!(big("115792089237316195423570985008687907853269984665640564039457584007913129639936").bits(), 257);
    }

    #[test]
    fn test_add_sub_mul_identities() {
        let mut rng = Rng::new(11);
        for _ in 0..40 {
            let a = {
                let x = rand_limbs(1 + (rng.next_u64() % 5) as usize, &mut rng);
                if rng.next_u64().is_multiple_of(2) { x.neg() } else { x }
            };
            let b = {
                let x = rand_limbs(1 + (rng.next_u64() % 5) as usize, &mut rng);
                if rng.next_u64().is_multiple_of(2) { x.neg() } else { x }
            };
            // Ring laws.
            assert_eq!(a.add(&b), b.add(&a), "add commutes");
            assert_eq!(a.mul(&b), b.mul(&a), "mul commutes");
            assert_eq!(a.add(&b).sub(&b), a, "sub inverts add");
            assert_eq!(a.sub(&a), BigInt::zero(), "a - a = 0");
            assert_eq!(a.add(&BigInt::zero()), a);
            assert_eq!(a.mul(&BigInt::one()), a);
            assert_eq!(a.mul(&BigInt::zero()), BigInt::zero());
            assert_eq!(a.neg().neg(), a);
            // Distributivity ties the three operations together.
            let c = rand_limbs(2, &mut rng);
            assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)), "distributive");
            // Sign rules.
            assert_eq!(a.neg().mul(&b), a.mul(&b).neg());
            assert_eq!(a.neg().mul(&b.neg()), a.mul(&b));
        }
        // Carry propagation across a full limb boundary.
        let max = BigInt::from_u64(u64::MAX);
        assert_eq!(max.add(&BigInt::one()), BigInt::from_parts(1, vec![0, 1]));
        assert_eq!(BigInt::from_parts(1, vec![0, 1]).sub(&BigInt::one()), max);
        assert_eq!(max.mul(&max), big("340282366920938463426481119284349108225"));
    }

    #[test]
    fn test_karatsuba_matches_schoolbook() {
        // The roadmap's property: above the crossover the two paths agree.
        let mut rng = Rng::new(2024);
        for _ in 0..6 {
            let a = rand_limbs(100, &mut rng);
            let b = rand_limbs(100, &mut rng);
            assert!(a.limbs.len() >= KARATSUBA_LIMBS && b.limbs.len() >= KARATSUBA_LIMBS);
            let fast = mul_mag(&a.limbs, &b.limbs);
            let slow = mul_mag_school(&a.limbs, &b.limbs);
            assert_eq!(fast, slow, "Karatsuba disagrees with schoolbook");
        }
        // Ragged shapes exercise the unequal-split branch.
        for (m, n) in [(40, 33), (100, 35), (64, 200), (33, 32)] {
            let a = rand_limbs(m, &mut rng);
            let b = rand_limbs(n, &mut rng);
            assert_eq!(mul_mag(&a.limbs, &b.limbs), mul_mag_school(&a.limbs, &b.limbs), "{m}x{n}");
        }
    }

    #[test]
    fn test_div_rem_inverts_mul() {
        let mut rng = Rng::new(99);
        for _ in 0..40 {
            let a = rand_limbs(1 + (rng.next_u64() % 6) as usize, &mut rng);
            let b = rand_limbs(1 + (rng.next_u64() % 3) as usize, &mut rng);
            // The roadmap's property: (a*b)/b == a exactly.
            assert_eq!(a.mul(&b).div_rem(&b).0, a, "(a*b)/b != a");
            assert!(a.mul(&b).div_rem(&b).1.is_zero());
            // Division identity a = q*b + r with |r| < |b|.
            let (q, r) = a.div_rem(&b);
            assert_eq!(q.mul(&b).add(&r), a, "a != q*b + r");
            assert_eq!(r.cmp_abs(&b), Ordering::Less, "remainder not reduced");
        }
        // Signs follow truncation, matching Rust's primitive operators.
        for (x, y) in [(17i64, 5i64), (-17, 5), (17, -5), (-17, -5)] {
            let (q, r) = BigInt::from_i64(x).div_rem(&BigInt::from_i64(y));
            assert_eq!(q.to_i64(), Some(x / y), "quotient of {x}/{y}");
            assert_eq!(r.to_i64(), Some(x % y), "remainder of {x}%{y}");
        }
        // Euclidean remainder is always non-negative.
        assert_eq!(BigInt::from_i64(-17).rem_euclid(&BigInt::from_i64(5)).to_i64(), Some(3));
        assert_eq!(BigInt::from_i64(-17).rem_euclid(&BigInt::from_i64(-5)).to_i64(), Some(3));
        // A dividend smaller than the divisor.
        let (q, r) = BigInt::from_u64(5).div_rem(&BigInt::from_u64(9));
        assert!(q.is_zero() && r == BigInt::from_u64(5));
        // The multi-limb path with a divisor whose top limb is already
        // normalized, and one that needs the full shift.
        for shift in [0usize, 1, 17, 63] {
            let d = BigInt::one().shl(64 + 63 - shift).add(&BigInt::from_u64(12345));
            let a = rand_limbs(8, &mut rng);
            let (q, r) = a.div_rem(&d);
            assert_eq!(q.mul(&d).add(&r), a, "reconstruct with shift {shift}");
            assert_eq!(r.cmp_abs(&d), Ordering::Less);
        }
    }

    /// Algorithm D's add-back correction fires only when the two-limb
    /// quotient estimate is still one too large after its own correction
    /// loop, which happens with probability about 2/2^64 on random input --
    /// random testing never reaches it. This case is constructed: with
    /// `v[n-2] == 0` the correction loop cannot run, so the lowest limb
    /// alone pushes the estimate past the true digit.
    #[test]
    fn test_div_rem_knuth_add_back_correction() {
        let u = big("57896044618658097711785492504343953926975274699741220483173719867314623479807");
        let v = big("3138550867693340381917894711603833208069624466305726808063");
        assert_eq!(v.limbs.len(), 3, "the trigger needs a three-limb divisor");
        assert_eq!(v.limbs[1], 0, "a zero middle limb disables the estimate correction");
        let (q, r) = u.div_rem(&v);
        assert_eq!(q.to_string_radix(10), "18446744073709551615");
        assert_eq!(r.to_string_radix(10), "3138550867693340381917894711603833208069624466305726808062");
        assert_eq!(q.mul(&v).add(&r), u, "reconstruction after the add-back");
        assert_eq!(r.cmp_abs(&v), Ordering::Less);
    }

    #[test]
    fn test_pow_modpow_and_gcd() {
        assert_eq!(BigInt::from_u64(2).pow(256).to_string_radix(10),
                   "115792089237316195423570985008687907853269984665640564039457584007913129639936");
        assert_eq!(BigInt::from_i64(-3).pow(3).to_i64(), Some(-27));
        assert_eq!(BigInt::from_i64(-3).pow(4).to_i64(), Some(81));
        assert_eq!(BigInt::from_u64(7).pow(0), BigInt::one());

        // mod_pow against the naive repeated-multiply definition.
        let mut rng = Rng::new(5);
        for _ in 0..25 {
            let m = BigInt::from_u64(1 + rng.next_u64() % 10_000);
            if m == BigInt::one() {
                continue;
            }
            let b = BigInt::from_u64(rng.next_u64() % 10_000);
            let e = rng.next_u64() % 60;
            let mut naive = BigInt::one();
            for _ in 0..e {
                naive = naive.mul(&b).rem_euclid(&m);
            }
            assert_eq!(b.mod_pow(&BigInt::from_u64(e), &m), naive, "{b}^{e} mod {m}");
        }
        // A window-crossing exponent, checked by Fermat's little theorem:
        // a^(p-1) = 1 (mod p) for prime p not dividing a.
        let p = big("170141183460469231731687303715884105727"); // 2^127 - 1, prime
        let a = big("123456789012345678901234567890");
        assert_eq!(a.mod_pow(&p.sub(&BigInt::one()), &p), BigInt::one());
        assert_eq!(a.mod_pow(&BigInt::zero(), &p), BigInt::one());
        assert_eq!(a.mod_pow(&BigInt::from_u64(5), &BigInt::one()), BigInt::zero());

        // gcd/lcm and the Bezout identity.
        assert_eq!(BigInt::from_u64(12).gcd(&BigInt::from_u64(18)).to_i64(), Some(6));
        assert_eq!(BigInt::from_i64(-12).gcd(&BigInt::from_u64(18)).to_i64(), Some(6));
        assert_eq!(BigInt::from_u64(12).lcm(&BigInt::from_u64(18)).to_i64(), Some(36));
        assert!(BigInt::zero().gcd(&BigInt::zero()).is_zero());
        assert_eq!(BigInt::zero().gcd(&BigInt::from_u64(7)).to_i64(), Some(7));
        for _ in 0..25 {
            let a = {
                let x = rand_limbs(1 + (rng.next_u64() % 3) as usize, &mut rng);
                if rng.next_u64().is_multiple_of(2) { x.neg() } else { x }
            };
            let b = rand_limbs(1 + (rng.next_u64() % 3) as usize, &mut rng);
            let (g, x, y) = a.extended_gcd(&b);
            // The roadmap's property: the Bezout identity holds exactly.
            assert_eq!(a.mul(&x).add(&b.mul(&y)), g, "Bezout failed");
            assert!(!g.is_negative(), "gcd must be non-negative");
            assert_eq!(g, a.gcd(&b), "extended_gcd disagrees with gcd");
            assert!(a.div_rem(&g).1.is_zero() && b.div_rem(&g).1.is_zero(), "g must divide both");
            // gcd * lcm = |a*b|
            assert_eq!(g.mul(&a.lcm(&b)), a.mul(&b).abs());
        }

        // Modular inverses.
        let m = BigInt::from_u64(1_000_000_007);
        for k in [2u64, 3, 999, 123_456_789] {
            let a = BigInt::from_u64(k);
            let inv = a.mod_inverse(&m).expect("coprime to a prime modulus");
            assert_eq!(a.mul(&inv).rem_euclid(&m), BigInt::one(), "inverse of {k}");
        }
        assert!(BigInt::from_u64(4).mod_inverse(&BigInt::from_u64(8)).is_none());
        assert!(BigInt::from_u64(3).mod_inverse(&BigInt::one()).is_none());
        // A negative residue still inverts.
        let inv = BigInt::from_i64(-3).mod_inverse(&m).unwrap();
        assert_eq!(BigInt::from_i64(-3).mul(&inv).rem_euclid(&m), BigInt::one());
    }

    #[test]
    fn test_bits_roots_and_shifts() {
        // Shifts agree with multiply and divide by powers of two.
        let mut rng = Rng::new(31);
        for _ in 0..20 {
            let a = rand_limbs(1 + (rng.next_u64() % 4) as usize, &mut rng);
            let k = (rng.next_u64() % 200) as usize;
            let two_k = BigInt::one().shl(k);
            assert_eq!(a.shl(k), a.mul(&two_k), "shl != mul by 2^k");
            assert_eq!(a.shr(k), a.div_rem(&two_k).0, "shr != div by 2^k");
            assert_eq!(a.shl(k).shr(k), a, "shl then shr must round-trip");
        }
        assert_eq!(BigInt::from_i64(-8).shl(2).to_i64(), Some(-32));
        assert_eq!(BigInt::from_i64(-8).shr(2).to_i64(), Some(-2));
        assert!(BigInt::from_u64(3).shr(9000).is_zero());

        // bit/set_bit are consistent with the shift-based definition.
        let mut v = BigInt::zero();
        for i in [0usize, 5, 63, 64, 65, 200] {
            v.set_bit(i, true);
            assert!(v.bit(i), "bit {i} should be set");
        }
        assert_eq!(v, [0usize, 5, 63, 64, 65, 200]
            .iter()
            .fold(BigInt::zero(), |acc, &i| acc.add(&BigInt::one().shl(i))));
        for i in [0usize, 5, 63, 64, 65, 200] {
            v.set_bit(i, false);
        }
        assert!(v.is_zero() && v.sign == 0, "clearing every bit must renormalize");
        assert!(!BigInt::zero().bit(0));

        // Bitwise ops act on magnitudes.
        let a = BigInt::from_u64(0b1100);
        let b = BigInt::from_u64(0b1010);
        assert_eq!(a.and(&b).to_i64(), Some(0b1000));
        assert_eq!(a.or(&b).to_i64(), Some(0b1110));
        assert_eq!(a.xor(&b).to_i64(), Some(0b0110));
        assert_eq!(a.xor(&a), BigInt::zero());
        assert_eq!(a.and(&BigInt::zero()), BigInt::zero());
        assert_eq!(a.or(&BigInt::zero()), a);
        // Multi-limb widths line up.
        let wide = BigInt::one().shl(130).or(&BigInt::one());
        assert!(wide.bit(130) && wide.bit(0) && !wide.bit(65));

        // Integer roots: exact squares, and the floor property elsewhere.
        assert_eq!(BigInt::zero().sqrt(), BigInt::zero());
        assert_eq!(BigInt::one().sqrt(), BigInt::one());
        assert_eq!(big("10000000000000000000000000000000000000000").sqrt(),
                   big("100000000000000000000"));
        for n in 0u64..200 {
            let r = BigInt::from_u64(n).sqrt();
            let r_u = r.to_i64().unwrap() as u64;
            assert!(r_u * r_u <= n && (r_u + 1) * (r_u + 1) > n, "isqrt({n})");
            assert_eq!(BigInt::from_u64(n).is_perfect_square(), r_u * r_u == n);
        }
        for _ in 0..10 {
            let a = rand_limbs(4, &mut rng);
            let r = a.sqrt();
            assert!(r.mul(&r) <= a, "sqrt overshoots");
            let r1 = r.add(&BigInt::one());
            assert!(r1.mul(&r1) > a, "sqrt undershoots");
            assert!(a.mul(&a).is_perfect_square());
        }
        assert!(!BigInt::from_i64(-4).is_perfect_square());

        // nth_root, including the odd-root-of-negative case.
        assert_eq!(BigInt::from_u64(1000).nth_root(3).to_i64(), Some(10));
        assert_eq!(BigInt::from_u64(1001).nth_root(3).to_i64(), Some(10));
        assert_eq!(BigInt::from_u64(999).nth_root(3).to_i64(), Some(9));
        assert_eq!(BigInt::from_i64(-27).nth_root(3).to_i64(), Some(-3));
        assert_eq!(BigInt::from_u64(81).nth_root(1).to_i64(), Some(81));
        assert_eq!(BigInt::from_u64(2).pow(100).nth_root(4), BigInt::from_u64(2).pow(25));
        for n in 2u32..7 {
            let a = rand_limbs(3, &mut rng);
            let r = a.nth_root(n);
            assert!(r.pow(u64::from(n)) <= a && r.add(&BigInt::one()).pow(u64::from(n)) > a,
                    "nth_root({n}) is not the floor");
        }
    }

    #[test]
    fn test_combinatorics_and_conversion() {
        // The roadmap's property: factorial(30) matches the known digits.
        assert_eq!(BigInt::factorial(30).to_string_radix(10),
                   "265252859812191058636308480000000");
        assert_eq!(BigInt::factorial(0), BigInt::one());
        assert_eq!(BigInt::factorial(1), BigInt::one());
        assert_eq!(BigInt::factorial(100).bits(), 525);
        // n! = n * (n-1)!
        for n in 1u64..20 {
            assert_eq!(BigInt::factorial(n), BigInt::factorial(n - 1).mul(&BigInt::from_u64(n)));
        }

        // Binomials: symmetry, Pascal's rule, a known large value, and the
        // factorial definition.
        assert_eq!(BigInt::binomial(60, 30).to_string_radix(10), "118264581564861424");
        assert_eq!(BigInt::binomial(5, 0), BigInt::one());
        assert_eq!(BigInt::binomial(5, 6), BigInt::zero());
        for n in 0u64..25 {
            let mut row_sum = BigInt::zero();
            for k in 0..=n {
                let c = BigInt::binomial(n, k);
                assert_eq!(c, BigInt::binomial(n, n - k), "C({n},{k}) symmetry");
                assert_eq!(
                    c,
                    BigInt::factorial(n)
                        .div_rem(&BigInt::factorial(k).mul(&BigInt::factorial(n - k)))
                        .0,
                    "C({n},{k}) against factorials"
                );
                if n > 0 && k > 0 && k < n {
                    assert_eq!(c, BigInt::binomial(n - 1, k - 1).add(&BigInt::binomial(n - 1, k)),
                               "Pascal at ({n},{k})");
                }
                row_sum = row_sum.add(&c);
            }
            assert_eq!(row_sum, BigInt::from_u64(2).pow(n), "row {n} sums to 2^n");
        }

        // Fibonacci by fast doubling: recurrence, a known value, and
        // Cassini's identity F(n-1)F(n+1) - F(n)^2 = (-1)^n.
        assert_eq!(BigInt::fibonacci(0), BigInt::zero());
        assert_eq!(BigInt::fibonacci(1), BigInt::one());
        assert_eq!(BigInt::fibonacci(100).to_string_radix(10), "354224848179261915075");
        for n in 2u64..60 {
            assert_eq!(BigInt::fibonacci(n),
                       BigInt::fibonacci(n - 1).add(&BigInt::fibonacci(n - 2)), "F({n})");
            let cassini = BigInt::fibonacci(n - 1)
                .mul(&BigInt::fibonacci(n + 1))
                .sub(&BigInt::fibonacci(n).mul(&BigInt::fibonacci(n)));
            let expect = if n % 2 == 0 { BigInt::one() } else { BigInt::one().neg() };
            assert_eq!(cassini, expect, "Cassini at {n}");
        }
        assert_eq!(BigInt::fibonacci(300).bits(), 208);

        // to_f64 tracks the exact value where f64 can represent it.
        assert!((BigInt::from_i64(-12345).to_f64() + 12345.0).abs() < 1e-9);
        assert!((BigInt::from_u64(2).pow(100).to_f64() - 2f64.powi(100)).abs() / 2f64.powi(100) < 1e-15);
        assert_eq!(BigInt::zero().to_f64(), 0.0);
        assert!(BigInt::from_u64(2).pow(4096).to_f64().is_infinite());
    }

    #[test]
    fn test_random_and_operators() {
        let mut rng = Rng::new(4242);
        for bits in [1usize, 7, 64, 65, 200] {
            for _ in 0..20 {
                let v = BigInt::random_bits(bits, &mut rng);
                assert_eq!(v.bits(), bits, "random_bits({bits}) width");
                assert!(!v.is_negative());
            }
        }
        assert!(BigInt::random_bits(0, &mut rng).is_zero());
        let bound = big("1000000000000000000000000");
        let mut seen_small = false;
        for _ in 0..200 {
            let v = BigInt::random_below(&bound, &mut rng);
            assert!(v < bound && !v.is_negative(), "sample out of range");
            if v < bound.div_rem(&BigInt::from_u64(2)).0 {
                seen_small = true;
            }
        }
        assert!(seen_small, "sampler never produced a value below the midpoint");

        // Operator sugar agrees with the named methods.
        let a = big("123456789012345678901234567890");
        let b = big("987654321098765432109876543210");
        assert_eq!(a.clone() + b.clone(), a.add(&b));
        assert_eq!(a.clone() - b.clone(), a.sub(&b));
        assert_eq!(a.clone() * b.clone(), a.mul(&b));
        assert_eq!(b.clone() / a.clone(), b.div_rem(&a).0);
        assert_eq!(b.clone() % a.clone(), b.div_rem(&a).1);
        assert_eq!(-a.clone(), a.neg());
        assert_eq!(a.clone() << 10, a.shl(10));
        assert_eq!(a.clone() >> 10, a.shr(10));
        assert_eq!(&a + &b, a.add(&b));
        assert_eq!(format!("{a}"), "123456789012345678901234567890");
        assert_eq!(BigInt::from(-5i64), BigInt::from_i64(-5));
        assert_eq!(BigInt::from(5u64), BigInt::from_u64(5));
        assert_eq!(BigInt::default(), BigInt::zero());
    }

    // Small helper so the ordering test can print values compactly.
    impl BigInt {
        fn to_string_radix_10_for_test(&self) -> String {
            self.to_string_radix(10)
        }
    }
}
