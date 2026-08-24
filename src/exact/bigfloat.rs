//! Arbitrary-precision binary floating point.
//!
//! A [`BigFloat`] is the exact dyadic rational `mantissa * 2^exponent`
//! together with a working `precision` measured in bits.
//!
//! # Canonical form
//!
//! Every value produced by this module is normalized: either the mantissa
//! is zero (and the exponent is zero), or the mantissa's magnitude has
//! *exactly* `precision` significant bits. Normalization is applied on
//! construction and after every operation, so `precision` is the true
//! working precision rather than an upper bound, and the leading bit of
//! the mantissa is always set.
//!
//! # Rounding
//!
//! All rounding is **round-to-nearest, ties-to-even** — the IEEE-754
//! default — applied exactly once per operation. `add`, `sub`, `mul`,
//! `div` and `sqrt` form the exact result (or an exact result plus a
//! sticky low bit that cannot change the rounding decision) and round it
//! once, so they are correctly rounded: the returned value is the closest
//! `precision`-bit dyadic to the true mathematical result. As a
//! consequence they reproduce IEEE-754 `f64` arithmetic bit for bit when
//! used at `precision = 53` on operands in the normal range.
//!
//! The transcendental functions ([`BigFloat::exp`], [`BigFloat::ln`],
//! [`BigFloat::sin`], [`BigFloat::cos`], [`BigFloat::atan`],
//! [`BigFloat::pow`]) and the constants ([`BigFloat::pi`],
//! [`BigFloat::e`], [`BigFloat::ln2`]) evaluate their series at 64 guard
//! bits above the target precision and round once at the end. They are
//! not *proved* correctly rounded (that would need the table-maker's
//! dilemma resolved), but the guard digits put the error far below one
//! ulp of the requested precision.
//!
//! Formulas: Gauss-Legendre AGM iteration for π (Brent 1976, Salamin
//! 1976), Machin's `π/4 = 4·atan(1/5) − atan(1/239)` as an independent
//! cross-check, `ln 2 = 2·atanh(1/3)`, exponential and circular Taylor
//! series after range reduction, and `ln x = 2^s · 2·atanh((m−1)/(m+1))`
//! after repeated square roots.

use crate::error::GeomError;
use crate::exact::bigint::BigInt;
use std::cmp::Ordering;

/// Guard bits carried above the target precision inside iterative
/// algorithms so the final rounding is not polluted by accumulated error.
const GUARD: usize = 64;

/// Smallest precision a `BigFloat` may carry.
const MIN_PRECISION: usize = 2;

/// Largest `|scale|` accepted by the range reductions in `exp`, `sin`
/// and `cos`; beyond this the reduction would need more working bits
/// than any caller could plausibly want.
const MAX_REDUCTION_SCALE: i64 = 1 << 20;

/// Largest decimal exponent accepted by [`BigFloat::from_str`].
const MAX_DECIMAL_EXP: u64 = 10_000;

// ---------------------------------------------------------------------------
// BigInt helpers
// ---------------------------------------------------------------------------

/// Is any bit strictly below position `k` set in the non-negative `a`?
fn has_set_bit_below(a: &BigInt, k: usize) -> bool {
    if k == 0 || a.is_zero() {
        return false;
    }
    if a.bits() <= k {
        return true; // every set bit lies below position k
    }
    a.shr(k).shl(k) != *a
}

/// Round the non-negative `a` down by `d` bits, to nearest with ties to
/// even.
fn round_shift_half_even(a: &BigInt, d: usize) -> BigInt {
    if d == 0 {
        return a.clone();
    }
    let q = a.shr(d);
    if !a.bit(d - 1) {
        return q;
    }
    if has_set_bit_below(a, d - 1) || q.bit(0) {
        q.add(&BigInt::one())
    } else {
        q
    }
}

/// Shift `m` right by `d` bits, forcing the low bit of the result when
/// anything was discarded.
///
/// The forced bit is the classical *sticky* bit: as long as at least
/// three guard bits sit below the eventual rounding position, replacing
/// the exact tail by an odd unit keeps the value strictly inside the same
/// rounding interval, so the later round-to-nearest decision is the one
/// the exact value would have produced.
fn shr_sticky(m: &BigInt, d: usize) -> BigInt {
    if d == 0 {
        return m.clone();
    }
    let neg = m.is_negative();
    let a = m.abs();
    let mut q = a.shr(d);
    if has_set_bit_below(&a, d) {
        q.set_bit(0, true);
    }
    if neg {
        q.neg()
    } else {
        q
    }
}

/// Force the low bit of `q`'s magnitude (see [`shr_sticky`]).
fn force_sticky(q: &BigInt) -> BigInt {
    let neg = q.is_negative();
    let mut a = q.abs();
    a.set_bit(0, true);
    if neg {
        a.neg()
    } else {
        a
    }
}

/// Rescale `m` from binary exponent `from` to binary exponent `to`,
/// exactly when shifting left and with a sticky bit when shifting right.
fn align_mantissa(m: &BigInt, from: i64, to: i64) -> BigInt {
    if from >= to {
        let d = usize::try_from(from - to).expect("alignment shift fits in usize");
        m.shl(d)
    } else {
        let d = usize::try_from(to - from).unwrap_or(usize::MAX);
        shr_sticky(m, d)
    }
}

// ---------------------------------------------------------------------------
// the type
// ---------------------------------------------------------------------------

/// An arbitrary-precision binary float: the exact value
/// `mantissa * 2^exponent`, carried at `precision` bits.
///
/// See the [module documentation](self) for the canonical form and the
/// rounding rules. Comparison, `PartialEq` and `Ord` are by *numeric
/// value*: two `BigFloat`s that represent the same number compare equal
/// even when their `precision` fields differ.
#[derive(Debug, Clone)]
pub struct BigFloat {
    /// Signed significand; its magnitude has exactly `precision` bits
    /// unless the value is zero.
    pub mantissa: BigInt,
    /// Binary exponent: the value is `mantissa * 2^exponent`.
    pub exponent: i64,
    /// Working precision in bits, at least 2.
    pub precision: usize,
}

impl BigFloat {
    /// The normalized value `mantissa * 2^exponent`, rounded to
    /// `precision` bits (nearest, ties to even).
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn new(mantissa: BigInt, exponent: i64, precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        if mantissa.is_zero() {
            return BigFloat { mantissa: BigInt::zero(), exponent: 0, precision };
        }
        let bits = mantissa.bits();
        if bits <= precision {
            let d = precision - bits;
            return BigFloat {
                mantissa: mantissa.shl(d),
                exponent: exponent - d as i64,
                precision,
            };
        }
        let drop = bits - precision;
        let neg = mantissa.is_negative();
        let mut q = round_shift_half_even(&mantissa.abs(), drop);
        let mut exponent = exponent + drop as i64;
        if q.bits() > precision {
            // The rounding carried into a new leading bit: q == 2^precision.
            q = q.shr(1);
            exponent += 1;
        }
        BigFloat {
            mantissa: if neg { q.neg() } else { q },
            exponent,
            precision,
        }
    }

    /// Zero at `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn zero(precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        BigFloat { mantissa: BigInt::zero(), exponent: 0, precision }
    }

    /// One at `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn one(precision: usize) -> Self {
        Self::new(BigInt::one(), 0, precision)
    }

    /// The integer `n` rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn from_i64(n: i64, precision: usize) -> Self {
        Self::new(BigInt::from_i64(n), 0, precision)
    }

    /// The integer `n` rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn from_bigint(n: &BigInt, precision: usize) -> Self {
        Self::new(n.clone(), 0, precision)
    }

    /// The exact value of `n`, at whatever precision that needs.
    fn exact_int(n: &BigInt) -> Self {
        Self::new(n.clone(), 0, n.bits().max(MIN_PRECISION))
    }

    /// The exact value of the finite `f64` `x`, rounded to `precision`
    /// bits (exact whenever `precision >= 53`, since every `f64` is a
    /// dyadic rational).
    ///
    /// # Panics
    /// Panics if `x` is infinite or NaN, or if `precision < 2`.
    #[must_use]
    pub fn from_f64(x: f64, precision: usize) -> Self {
        assert!(x.is_finite(), "BigFloat::from_f64 requires a finite value");
        if x == 0.0_f64 {
            return Self::zero(precision);
        }
        let bits = x.to_bits();
        let neg = (bits >> 63) == 1;
        let biased = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & 0x000f_ffff_ffff_ffff;
        let (mag, exponent) = if biased == 0 {
            (frac, -1074_i64)
        } else {
            (frac | (1_u64 << 52), biased - 1075)
        };
        let mantissa = BigInt::from_u64(mag);
        Self::new(if neg { mantissa.neg() } else { mantissa }, exponent, precision)
    }

    /// Parse a decimal literal such as `"-12.5"`, `"3.14159e-7"` or
    /// `"42"`, correctly rounded to `precision` bits.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] if the string is not a
    /// decimal literal, or if its decimal exponent exceeds ±10000.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    pub fn from_str(s: &str, precision: usize) -> Result<Self, GeomError> {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        let text = s.trim();
        if text.is_empty() {
            return Err(GeomError::InvalidArgument("empty numeric literal"));
        }
        let (negative, rest) = match text.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        let (mant_str, exp_str) = match rest.find(['e', 'E']) {
            Some(i) => (&rest[..i], &rest[i + 1..]),
            None => (rest, ""),
        };
        let exp10: i64 = if exp_str.is_empty() {
            0
        } else {
            exp_str
                .parse::<i64>()
                .map_err(|_| GeomError::InvalidArgument("invalid decimal exponent"))?
        };
        let (int_part, frac_part) = match mant_str.find('.') {
            Some(i) => (&mant_str[..i], &mant_str[i + 1..]),
            None => (mant_str, ""),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return Err(GeomError::InvalidArgument("no digits in numeric literal"));
        }
        if !int_part.bytes().all(|b| b.is_ascii_digit())
            || !frac_part.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(GeomError::InvalidArgument("non-digit in numeric literal"));
        }
        let digits = format!("{int_part}{frac_part}");
        let value = BigInt::from_str_radix(&digits, 10)?;
        if value.is_zero() {
            return Ok(Self::zero(precision));
        }
        let scale = exp10
            .checked_sub(frac_part.len() as i64)
            .ok_or(GeomError::InvalidArgument("decimal exponent out of range"))?;
        if scale.unsigned_abs() > MAX_DECIMAL_EXP {
            return Err(GeomError::InvalidArgument("decimal exponent out of range"));
        }
        let value = if negative { value.neg() } else { value };
        let ten = BigInt::from_u64(10);
        Ok(if scale >= 0 {
            Self::new(value.mul(&ten.pow(scale.unsigned_abs())), 0, precision)
        } else {
            let den = ten.pow(scale.unsigned_abs());
            Self::exact_int(&value).div_prec(&Self::exact_int(&den), precision)
        })
    }

    // -----------------------------------------------------------------
    // predicates and cheap transforms
    // -----------------------------------------------------------------

    /// Is this value zero?
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    /// Is this value strictly negative?
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.mantissa.is_negative()
    }

    /// Is this value strictly positive?
    #[must_use]
    pub fn is_positive(&self) -> bool {
        !self.mantissa.is_zero() && !self.mantissa.is_negative()
    }

    /// The sign: `-1`, `0` or `+1`.
    #[must_use]
    pub fn signum(&self) -> i8 {
        self.mantissa.sign
    }

    /// The negation, exactly (negation never rounds).
    #[must_use]
    pub fn neg(&self) -> Self {
        BigFloat {
            mantissa: self.mantissa.neg(),
            exponent: self.exponent,
            precision: self.precision,
        }
    }

    /// The magnitude, exactly.
    #[must_use]
    pub fn abs(&self) -> Self {
        BigFloat {
            mantissa: self.mantissa.abs(),
            exponent: self.exponent,
            precision: self.precision,
        }
    }

    /// Multiply by `2^k`, exactly (only the exponent moves).
    #[must_use]
    pub fn mul_pow2(&self, k: i64) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        BigFloat {
            mantissa: self.mantissa.clone(),
            exponent: self.exponent.saturating_add(k),
            precision: self.precision,
        }
    }

    /// The same value rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn round_to(&self, precision: usize) -> Self {
        if precision == self.precision {
            return self.clone();
        }
        Self::new(self.mantissa.clone(), self.exponent, precision)
    }

    /// Binary scale: the exponent of the leading bit, so that
    /// `2^scale <= |self| < 2^(scale+1)`. Meaningless for zero.
    fn scale(&self) -> i64 {
        self.exponent + self.mantissa.bits() as i64 - 1
    }

    /// The nearest integer (ties to even) as a [`BigInt`].
    ///
    /// # Panics
    /// Panics if the value is so large that the integer would not fit in
    /// memory (binary exponent above 2^32).
    fn round_to_bigint(&self) -> BigInt {
        if self.is_zero() {
            return BigInt::zero();
        }
        if self.exponent >= 0 {
            let e = usize::try_from(self.exponent).expect("exponent fits in usize");
            assert!(e < (1_usize << 32), "BigFloat is too large to convert to an integer");
            return self.mantissa.shl(e);
        }
        let d = usize::try_from(self.exponent.unsigned_abs()).unwrap_or(usize::MAX);
        let q = round_shift_half_even(&self.mantissa.abs(), d);
        if self.mantissa.is_negative() {
            q.neg()
        } else {
            q
        }
    }

    /// The exact integer value, or `None` when the value is not an
    /// integer (or is too large to materialise).
    fn as_exact_int(&self) -> Option<BigInt> {
        if self.is_zero() {
            return Some(BigInt::zero());
        }
        if self.exponent >= 0 {
            let e = usize::try_from(self.exponent).ok()?;
            if e > 4096 {
                return None;
            }
            return Some(self.mantissa.shl(e));
        }
        let d = usize::try_from(self.exponent.unsigned_abs()).ok()?;
        let q = self.mantissa.shr(d);
        if q.shl(d) == self.mantissa {
            Some(q)
        } else {
            None
        }
    }

    // -----------------------------------------------------------------
    // correctly rounded arithmetic
    // -----------------------------------------------------------------

    /// `self + other`, correctly rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn add_prec(&self, other: &Self, precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        if self.is_zero() {
            return other.round_to(precision);
        }
        if other.is_zero() {
            return self.round_to(precision);
        }
        // Align on the smaller exponent, but never below a floor that is
        // more than max(precision) + GUARD bits under the larger operand:
        // anything discarded there is far too small to survive rounding,
        // and it cannot participate in cancellation either, because the
        // operands then differ by more than a factor 2^(precision+GUARD).
        let top = self.scale().max(other.scale());
        let widest = precision.max(self.precision).max(other.precision) as i64;
        let floor = top.saturating_sub(widest).saturating_sub(GUARD as i64);
        let e = self.exponent.min(other.exponent).max(floor);
        let a = align_mantissa(&self.mantissa, self.exponent, e);
        let b = align_mantissa(&other.mantissa, other.exponent, e);
        Self::new(a.add(&b), e, precision)
    }

    /// `self - other`, correctly rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn sub_prec(&self, other: &Self, precision: usize) -> Self {
        self.add_prec(&other.neg(), precision)
    }

    /// `self * other`, correctly rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn mul_prec(&self, other: &Self, precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        if self.is_zero() || other.is_zero() {
            return Self::zero(precision);
        }
        Self::new(
            self.mantissa.mul(&other.mantissa),
            self.exponent.saturating_add(other.exponent),
            precision,
        )
    }

    /// `self / other`, correctly rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `other` is zero, or if `precision < 2`.
    #[must_use]
    pub fn div_prec(&self, other: &Self, precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        assert!(!other.is_zero(), "BigFloat division by zero");
        if self.is_zero() {
            return Self::zero(precision);
        }
        // Pre-scale the numerator so the quotient carries at least
        // precision + 4 bits; the remainder then only feeds the sticky bit.
        let want = (precision + 4) as i64;
        let shift = want + other.mantissa.bits() as i64 - self.mantissa.bits() as i64;
        let shift = usize::try_from(shift.max(0)).expect("shift fits in usize");
        let (q, r) = self.mantissa.shl(shift).div_rem(&other.mantissa);
        let q = if r.is_zero() { q } else { force_sticky(&q) };
        let exponent = self
            .exponent
            .saturating_sub(other.exponent)
            .saturating_sub(shift as i64);
        Self::new(q, exponent, precision)
    }

    /// The square root, correctly rounded to `precision` bits.
    ///
    /// # Panics
    /// Panics if `self` is negative, or if `precision < 2`.
    #[must_use]
    pub fn sqrt_prec(&self, precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        assert!(!self.is_negative(), "BigFloat::sqrt of a negative value");
        if self.is_zero() {
            return Self::zero(precision);
        }
        // Scale the mantissa up to at least 2*(precision+4) bits so the
        // integer square root carries precision+4 bits, and keep the
        // exponent even so halving it is exact.
        let want = 2 * (precision + 4);
        let mut shift = want.saturating_sub(self.mantissa.bits());
        if (self.exponent - shift as i64).rem_euclid(2) != 0 {
            shift += 1;
        }
        let scaled = self.mantissa.shl(shift);
        let root = scaled.sqrt();
        let exact = root.mul(&root) == scaled;
        let mantissa = if exact { root } else { force_sticky(&root) };
        Self::new(mantissa, (self.exponent - shift as i64) / 2, precision)
    }

    /// `self + other`, correctly rounded to the larger of the two
    /// operand precisions.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.add_prec(other, self.precision.max(other.precision))
    }

    /// `self - other`, correctly rounded to the larger of the two
    /// operand precisions.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.sub_prec(other, self.precision.max(other.precision))
    }

    /// `self * other`, correctly rounded to the larger of the two
    /// operand precisions.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        self.mul_prec(other, self.precision.max(other.precision))
    }

    /// `self / other`, correctly rounded to the larger of the two
    /// operand precisions.
    ///
    /// # Panics
    /// Panics if `other` is zero.
    #[must_use]
    pub fn div(&self, other: &Self) -> Self {
        self.div_prec(other, self.precision.max(other.precision))
    }

    /// The square root, correctly rounded to this value's precision.
    ///
    /// # Panics
    /// Panics if `self` is negative.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        self.sqrt_prec(self.precision)
    }

    // -----------------------------------------------------------------
    // conversions out
    // -----------------------------------------------------------------

    /// The nearest `f64` (ties to even), saturating to ±∞ on overflow
    /// and supporting the subnormal range.
    ///
    /// Exact whenever the value is representable, so
    /// `BigFloat::from_f64(x, p).to_f64() == x` for every finite `x` and
    /// every `p >= 53`.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0_f64;
        }
        let negative = self.mantissa.is_negative();
        let scale = self.scale();
        if scale > 1023 {
            return if negative { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        // Bits of significand available at this scale: 53 in the normal
        // range, fewer once the value falls below 2^-1022.
        let avail = 1075_i64 + scale;
        if avail <= 0 {
            // |value| <= 2^-1075, i.e. at most half the smallest subnormal.
            let mag = self.mantissa.abs();
            let is_half = scale == -1075 && mag.bits() > 0 && !has_set_bit_below(&mag, mag.bits() - 1);
            let out = if scale < -1075 || is_half {
                0.0_f64 // below, or exactly at, the tie that rounds to even
            } else {
                f64::from_bits(1)
            };
            return if negative { -out } else { out };
        }
        let target = usize::try_from(avail.min(53)).expect("positive and small");
        let rounded = self.round_to(target.max(MIN_PRECISION));
        let mut v = rounded.mantissa.to_f64();
        let mut e = rounded.exponent;
        // Scale in bounded steps. Going down, every intermediate is >= the
        // final value, so no intermediate rounds; going up, an overflow
        // saturates to infinity, which is what we want anyway.
        while e != 0 {
            let step = e.clamp(-512, 512);
            v *= f64::powi(2.0_f64, step as i32);
            e -= step;
        }
        v
    }

    /// `|self| * 10^digits`, either truncated or rounded (ties to even).
    fn scaled_magnitude(&self, digits: usize, round: bool) -> BigInt {
        if self.is_zero() {
            return BigInt::zero();
        }
        assert!(
            self.exponent.unsigned_abs() < (1_u64 << 32),
            "BigFloat exponent is too extreme to render as a decimal string"
        );
        let scaled = self.mantissa.abs().mul(&BigInt::from_u64(10).pow(digits as u64));
        if self.exponent >= 0 {
            scaled.shl(usize::try_from(self.exponent).expect("checked above"))
        } else {
            let d = usize::try_from(self.exponent.unsigned_abs()).expect("checked above");
            if round {
                round_shift_half_even(&scaled, d)
            } else {
                scaled.shr(d)
            }
        }
    }

    /// Fixed-point decimal rendering with `digits` digits after the point.
    fn digit_string(&self, digits: usize, round: bool) -> String {
        let mag = self.scaled_magnitude(digits, round).to_string_radix(10);
        let padded = if mag.len() <= digits {
            format!("{}{}", "0".repeat(digits + 1 - mag.len()), mag)
        } else {
            mag
        };
        let split = padded.len() - digits;
        let mut out = String::with_capacity(padded.len() + 2);
        if self.mantissa.is_negative() {
            out.push('-');
        }
        out.push_str(&padded[..split]);
        if digits > 0 {
            out.push('.');
            out.push_str(&padded[split..]);
        }
        out
    }

    /// Fixed-point decimal string with exactly `digits` digits after the
    /// decimal point, correctly rounded (ties to even).
    ///
    /// # Panics
    /// Panics if the binary exponent exceeds ±2^32, which would make the
    /// digit string astronomically long.
    #[must_use]
    pub fn to_string_decimal(&self, digits: usize) -> String {
        self.digit_string(digits, true)
    }

    // -----------------------------------------------------------------
    // constants
    // -----------------------------------------------------------------

    /// π to `precision` bits, by the Gauss-Legendre AGM iteration.
    ///
    /// The iteration doubles its correct digits each step, so it needs
    /// only `O(log precision)` square roots.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn pi(precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        let wp = precision + GUARD;
        let one = Self::one(wp);
        let mut a = one.clone();
        let mut b = one.div_prec(&Self::from_i64(2, wp).sqrt_prec(wp), wp);
        let mut t = one.mul_pow2(-2);
        let mut p = one.clone();
        let iters = 3 + (usize::BITS - wp.leading_zeros()) as usize;
        for _ in 0..iters {
            let a_next = a.add_prec(&b, wp).mul_pow2(-1);
            let b_next = a.mul_prec(&b, wp).sqrt_prec(wp);
            let d = a.sub_prec(&a_next, wp);
            t = t.sub_prec(&p.mul_prec(&d.mul_prec(&d, wp), wp), wp);
            let done = a_next == b_next;
            a = a_next;
            b = b_next;
            p = p.mul_pow2(1);
            if done {
                break;
            }
        }
        let s = a.add_prec(&b, wp);
        s.mul_prec(&s, wp).div_prec(&t.mul_pow2(2), wp).round_to(precision)
    }

    /// Euler's number `e` to `precision` bits, by summing `1/k!`.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn e(precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        let wp = precision + GUARD;
        let mut sum = Self::one(wp);
        let mut term = Self::one(wp);
        let mut k: u64 = 1;
        loop {
            term = term.div_prec(&Self::from_i64(k as i64, wp), wp);
            if term.is_zero() || term.scale() < sum.scale() - wp as i64 {
                break;
            }
            sum = sum.add_prec(&term, wp);
            k += 1;
        }
        sum.round_to(precision)
    }

    /// `ln 2` to `precision` bits, via `ln 2 = 2·atanh(1/3)`.
    ///
    /// # Panics
    /// Panics if `precision < 2`.
    #[must_use]
    pub fn ln2(precision: usize) -> Self {
        assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
        let wp = precision + GUARD;
        let third = Self::one(wp).div_prec(&Self::from_i64(3, wp), wp);
        odd_power_series(&third, wp, false).mul_pow2(1).round_to(precision)
    }

    // -----------------------------------------------------------------
    // elementary functions
    // -----------------------------------------------------------------

    /// `e^self`, evaluated with 64 guard bits.
    ///
    /// Reduces `self = k·ln 2 + r` with `|r| <= ln2/2`, halves `r` ten
    /// more times, sums the exponential series, then undoes both
    /// reductions.
    ///
    /// # Panics
    /// Panics if `|self|` is astronomically large (binary scale above
    /// 2^20), where the range reduction would need unreasonable
    /// precision.
    #[must_use]
    pub fn exp(&self) -> Self {
        let p = self.precision;
        if self.is_zero() {
            return Self::one(p);
        }
        let scale = self.scale();
        assert!(scale < MAX_REDUCTION_SCALE, "exp argument is too large to range-reduce");
        let wp = p + GUARD + usize::try_from(scale.max(0)).expect("bounded above");
        let ln2 = Self::ln2(wp);
        let k = self.div_prec(&ln2, wp).round_to_bigint();
        let r = self.sub_prec(&Self::from_bigint(&k, wp).mul_prec(&ln2, wp), wp);
        const HALVINGS: u32 = 10;
        let reduced = r.mul_pow2(-i64::from(HALVINGS));
        let mut sum = exp_series(&reduced, wp);
        for _ in 0..HALVINGS {
            sum = sum.mul_prec(&sum, wp);
        }
        let shift = k.to_i64().expect("exp argument is within range");
        sum.mul_pow2(shift).round_to(p)
    }

    /// The natural logarithm, evaluated with 64 guard bits.
    ///
    /// Splits `self = m·2^k` with `m` near 1, takes square roots until
    /// `|m - 1| <= 2^-8`, then sums `2·atanh((m-1)/(m+1))`.
    ///
    /// # Panics
    /// Panics if `self` is zero or negative.
    #[must_use]
    pub fn ln(&self) -> Self {
        assert!(self.is_positive(), "ln requires a strictly positive argument");
        let p = self.precision;
        let wp = p + GUARD;
        let lead = self.mantissa.bits() as i64 - 1;
        let mut k = self.exponent + lead;
        let mut m = BigFloat {
            mantissa: self.mantissa.clone(),
            exponent: -lead,
            precision: self.precision,
        }
        .round_to(wp);
        // m is in [1, 2); centre it on 1 to halve the number of roots.
        if m.to_f64() > 1.5_f64 {
            m = m.mul_pow2(-1);
            k += 1;
        }
        let one = Self::one(wp);
        let limit = one.mul_pow2(-8);
        let mut roots: u32 = 0;
        while roots < 64 {
            let diff = m.sub_prec(&one, wp).abs();
            if diff.is_zero() || diff <= limit {
                break;
            }
            m = m.sqrt_prec(wp);
            roots += 1;
        }
        let z = m
            .sub_prec(&one, wp)
            .div_prec(&m.add_prec(&one, wp), wp);
        let ln_m = odd_power_series(&z, wp, false).mul_pow2(1 + i64::from(roots));
        Self::from_i64(k, wp)
            .mul_prec(&Self::ln2(wp), wp)
            .add_prec(&ln_m, wp)
            .round_to(p)
    }

    /// The sine and cosine together, evaluated with 64 guard bits.
    ///
    /// Reduces the argument modulo `π/2` (carrying enough extra bits to
    /// absorb the cancellation), sums both Maclaurin series on the
    /// reduced argument, then applies the quadrant symmetry.
    ///
    /// # Panics
    /// Panics if `|self|` is astronomically large (binary scale above
    /// 2^20).
    #[must_use]
    pub fn sin_cos(&self) -> (Self, Self) {
        let p = self.precision;
        if self.is_zero() {
            return (Self::zero(p), Self::one(p));
        }
        let scale = self.scale();
        assert!(scale < MAX_REDUCTION_SCALE, "trigonometric argument is too large to reduce");
        let wp = p + GUARD + usize::try_from(scale.max(0)).expect("bounded above");
        let half_pi = Self::pi(wp).mul_pow2(-1);
        let k = self.div_prec(&half_pi, wp).round_to_bigint();
        let r = self.sub_prec(&Self::from_bigint(&k, wp).mul_prec(&half_pi, wp), wp);
        let (s, c) = sin_cos_series(&r, wp);
        let quadrant = k
            .rem_euclid(&BigInt::from_i64(4))
            .to_i64()
            .expect("remainder mod 4 fits in i64");
        let (sin, cos) = match quadrant {
            0 => (s, c),
            1 => (c, s.neg()),
            2 => (s.neg(), c.neg()),
            _ => (c.neg(), s),
        };
        (sin.round_to(p), cos.round_to(p))
    }

    /// The sine, evaluated with 64 guard bits.
    ///
    /// # Panics
    /// Panics if `|self|` is astronomically large (binary scale above 2^20).
    #[must_use]
    pub fn sin(&self) -> Self {
        self.sin_cos().0
    }

    /// The cosine, evaluated with 64 guard bits.
    ///
    /// # Panics
    /// Panics if `|self|` is astronomically large (binary scale above 2^20).
    #[must_use]
    pub fn cos(&self) -> Self {
        self.sin_cos().1
    }

    /// The arc tangent, evaluated with 64 guard bits.
    ///
    /// Uses `atan x = π/2 − atan(1/x)` for `|x| > 1`, then the halving
    /// identity `atan x = 2·atan(x / (1 + sqrt(1 + x²)))` until the
    /// argument is below `2^-8`, then the alternating series.
    #[must_use]
    pub fn atan(&self) -> Self {
        let p = self.precision;
        if self.is_zero() {
            return Self::zero(p);
        }
        let wp = p + GUARD;
        let negative = self.is_negative();
        let x = self.abs().round_to(wp);
        let one = Self::one(wp);
        let value = if x > one {
            Self::pi(wp)
                .mul_pow2(-1)
                .sub_prec(&atan_reduced(&one.div_prec(&x, wp), wp), wp)
        } else {
            atan_reduced(&x, wp)
        };
        let value = if negative { value.neg() } else { value };
        value.round_to(p)
    }

    /// `self^n` for an integer exponent, by binary exponentiation with
    /// guard bits.
    ///
    /// # Panics
    /// Panics if `self` is zero and `n` is negative.
    #[must_use]
    pub fn powi(&self, n: i64) -> Self {
        let p = self.precision;
        if n == 0 {
            return Self::one(p);
        }
        let magnitude = n.unsigned_abs();
        let extra = 2 * (64 - magnitude.leading_zeros()) as usize;
        let wp = p + GUARD + extra;
        let mut base = self.round_to(wp);
        let mut acc = Self::one(wp);
        let mut k = magnitude;
        while k > 0 {
            if !k.is_multiple_of(2) {
                acc = acc.mul_prec(&base, wp);
            }
            k >>= 1;
            if k > 0 {
                base = base.mul_prec(&base, wp);
            }
        }
        if n < 0 {
            acc = Self::one(wp).div_prec(&acc, wp);
        }
        acc.round_to(p)
    }

    /// `self^exponent`.
    ///
    /// Integer exponents (including negative ones) go through
    /// [`BigFloat::powi`] and so work for any sign of base; other
    /// exponents are evaluated as `exp(exponent · ln self)`.
    ///
    /// # Panics
    /// Panics if `exponent` is not an integer and `self` is not strictly
    /// positive, or if `self` is zero and the exponent is a negative
    /// integer.
    #[must_use]
    pub fn pow(&self, exponent: &Self) -> Self {
        if let Some(n) = exponent.as_exact_int().and_then(|i| i.to_i64()) {
            return self.powi(n);
        }
        assert!(self.is_positive(), "pow with a non-integer exponent requires a positive base");
        let p = self.precision.max(exponent.precision);
        let mut wp = p + GUARD;
        // Widen the working precision when the product is large, since
        // exp's range reduction loses bits proportional to its scale.
        let mut product = exponent.mul_prec(&self.round_to(wp).ln(), wp);
        if !product.is_zero() && product.scale() > 0 {
            wp += usize::try_from(product.scale()).expect("bounded by the argument");
            product = exponent.mul_prec(&self.round_to(wp).ln(), wp);
        }
        product.exp().round_to(p)
    }

    /// The arithmetic-geometric mean of `a` and `b`.
    ///
    /// Iterates `(a, b) -> ((a+b)/2, sqrt(a·b))`, which converges
    /// quadratically to the common limit.
    ///
    /// # Panics
    /// Panics if either argument is negative.
    #[must_use]
    pub fn agm(a: &Self, b: &Self) -> Self {
        assert!(!a.is_negative() && !b.is_negative(), "agm requires non-negative arguments");
        let p = a.precision.max(b.precision);
        if a.is_zero() || b.is_zero() {
            return Self::zero(p);
        }
        let wp = p + GUARD;
        let mut x = a.round_to(wp);
        let mut y = b.round_to(wp);
        for _ in 0..128 {
            if x == y {
                break;
            }
            let x_next = x.add_prec(&y, wp).mul_pow2(-1);
            let y_next = x.mul_prec(&y, wp).sqrt_prec(wp);
            x = x_next;
            y = y_next;
            let d = x.sub_prec(&y, wp).abs();
            if d.is_zero() || d.scale() < x.scale() - p as i64 - 8 {
                break;
            }
        }
        x.round_to(p)
    }
}

// ---------------------------------------------------------------------------
// series kernels
// ---------------------------------------------------------------------------

/// `Σ x^(2j+1)/(2j+1)`, alternating in sign when `alternating` is set.
///
/// With `alternating = false` this is `atanh x`, with `alternating = true`
/// it is `atan x`; both need `|x| < 1` and converge quickly once `|x|`
/// has been reduced well below 1.
fn odd_power_series(x: &BigFloat, wp: usize, alternating: bool) -> BigFloat {
    if x.is_zero() {
        return BigFloat::zero(wp);
    }
    let x2 = x.mul_prec(x, wp);
    let mut power = x.round_to(wp);
    let mut sum = power.clone();
    let mut k: u64 = 1;
    loop {
        power = power.mul_prec(&x2, wp);
        if power.is_zero() {
            break;
        }
        let mut term = power.div_prec(&BigFloat::from_i64((2 * k + 1) as i64, wp), wp);
        if alternating && !k.is_multiple_of(2) {
            term = term.neg();
        }
        if term.scale() < sum.scale() - wp as i64 {
            break;
        }
        sum = sum.add_prec(&term, wp);
        k += 1;
    }
    sum
}

/// `Σ x^n/n!` — the exponential series, for a well-reduced `|x| < 1`.
fn exp_series(x: &BigFloat, wp: usize) -> BigFloat {
    let mut sum = BigFloat::one(wp);
    let mut term = BigFloat::one(wp);
    let mut n: u64 = 1;
    loop {
        term = term
            .mul_prec(x, wp)
            .div_prec(&BigFloat::from_i64(n as i64, wp), wp);
        if term.is_zero() || term.scale() < sum.scale() - wp as i64 {
            break;
        }
        sum = sum.add_prec(&term, wp);
        n += 1;
    }
    sum
}

/// Maclaurin series for `(sin x, cos x)` on a reduced `|x| <= π/4`.
fn sin_cos_series(x: &BigFloat, wp: usize) -> (BigFloat, BigFloat) {
    let one = BigFloat::one(wp);
    if x.is_zero() {
        return (BigFloat::zero(wp), one);
    }
    let neg_x2 = x.mul_prec(x, wp).neg();
    let mut term = x.round_to(wp);
    let mut sin = term.clone();
    let mut n: u64 = 1;
    loop {
        term = term
            .mul_prec(&neg_x2, wp)
            .div_prec(&BigFloat::from_i64(((2 * n) * (2 * n + 1)) as i64, wp), wp);
        if term.is_zero() || term.scale() < sin.scale() - wp as i64 {
            break;
        }
        sin = sin.add_prec(&term, wp);
        n += 1;
    }
    let mut term = one.clone();
    let mut cos = one;
    let mut n: u64 = 1;
    loop {
        term = term
            .mul_prec(&neg_x2, wp)
            .div_prec(&BigFloat::from_i64(((2 * n - 1) * (2 * n)) as i64, wp), wp);
        if term.is_zero() || term.scale() < cos.scale() - wp as i64 {
            break;
        }
        cos = cos.add_prec(&term, wp);
        n += 1;
    }
    (sin, cos)
}

/// `atan x` for `0 <= x <= 1`, using the halving identity before the
/// series so that convergence is fast for every argument in range.
fn atan_reduced(x: &BigFloat, wp: usize) -> BigFloat {
    let one = BigFloat::one(wp);
    let limit = one.mul_pow2(-8);
    let mut y = x.round_to(wp);
    let mut halvings: u32 = 0;
    while halvings < 64 && y > limit {
        let d = one
            .add_prec(&y.mul_prec(&y, wp), wp)
            .sqrt_prec(wp)
            .add_prec(&one, wp);
        y = y.div_prec(&d, wp);
        halvings += 1;
    }
    odd_power_series(&y, wp, true).mul_pow2(i64::from(halvings))
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

impl BigFloat {
    /// Compare magnitudes; both operands must be non-zero.
    fn cmp_magnitude(&self, other: &Self) -> Ordering {
        let sa = self.scale();
        let sb = other.scale();
        if sa != sb {
            return sa.cmp(&sb);
        }
        // Equal scales: the mantissas differ in length by exactly the
        // difference of the exponents, so aligning is cheap.
        let ba = self.mantissa.bits();
        let bb = other.mantissa.bits();
        let a = self.mantissa.abs();
        let b = other.mantissa.abs();
        match ba.cmp(&bb) {
            Ordering::Less => a.shl(bb - ba).cmp_abs(&b),
            Ordering::Greater => a.cmp_abs(&b.shl(ba - bb)),
            Ordering::Equal => a.cmp_abs(&b),
        }
    }
}

impl PartialEq for BigFloat {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for BigFloat {}

impl Ord for BigFloat {
    /// Numeric comparison; the `precision` field plays no part.
    fn cmp(&self, other: &Self) -> Ordering {
        let sa = self.mantissa.sign;
        let sb = other.mantissa.sign;
        if sa != sb {
            return sa.cmp(&sb);
        }
        if sa == 0 {
            return Ordering::Equal;
        }
        let ord = self.cmp_magnitude(other);
        if sa > 0 {
            ord
        } else {
            ord.reverse()
        }
    }
}

impl PartialOrd for BigFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// free functions
// ---------------------------------------------------------------------------

/// Bits of working precision needed for `n` correct decimal digits, with
/// room to spare (log2(10) < 3.33, so 4 bits per digit plus 96 guard bits).
fn digits_to_bits(n: usize) -> usize {
    n.saturating_mul(4).saturating_add(96)
}

/// The decimal expansion of π truncated to `n_decimal` places, e.g.
/// `"3.14159"` for `n_decimal == 5`.
#[must_use]
pub fn pi_digits(n_decimal: usize) -> String {
    BigFloat::pi(digits_to_bits(n_decimal)).digit_string(n_decimal, false)
}

/// The decimal expansion of `e` truncated to `n` places.
#[must_use]
pub fn e_digits(n: usize) -> String {
    BigFloat::e(digits_to_bits(n)).digit_string(n, false)
}

/// The decimal expansion of `√2` truncated to `n` places.
#[must_use]
pub fn sqrt2_digits(n: usize) -> String {
    let prec = digits_to_bits(n);
    BigFloat::from_i64(2, prec).sqrt().digit_string(n, false)
}

/// π to `precision` bits from Machin's formula,
/// `π = 16·atan(1/5) − 4·atan(1/239)`.
///
/// This is deliberately a different algorithm from [`BigFloat::pi`] (a
/// linearly convergent arctangent series against a quadratically
/// convergent AGM iteration) so that the two can cross-check each other.
///
/// # Panics
/// Panics if `precision < 2`.
#[must_use]
pub fn machin_pi(precision: usize) -> BigFloat {
    assert!(precision >= MIN_PRECISION, "BigFloat precision must be at least 2 bits");
    let wp = precision + GUARD;
    let one = BigFloat::one(wp);
    let a = odd_power_series(&one.div_prec(&BigFloat::from_i64(5, wp), wp), wp, true);
    let b = odd_power_series(&one.div_prec(&BigFloat::from_i64(239, wp), wp), wp, true);
    a.mul_pow2(4).sub_prec(&b.mul_pow2(2), wp).round_to(precision)
}

/// The exact error of [`crate::core::compensated::sum_neumaier`] on `xs`.
///
/// Every `f64` is a dyadic rational, so the true sum `Σ xᵢ` is computed
/// exactly in `BigFloat` at a precision wide enough to hold every bit of
/// every operand. The return value is `sum_neumaier(xs) − Σ xᵢ`,
/// evaluated exactly and then rounded once to `f64`; it is exactly `0.0`
/// whenever the compensated sum is perfect.
///
/// # Panics
/// Panics if any element is infinite or NaN.
#[must_use]
pub fn compensated_to_bigfloat_check(xs: &[f64]) -> f64 {
    let approx = crate::core::compensated::sum_neumaier(xs);
    assert!(approx.is_finite(), "compensated_to_bigfloat_check requires a finite sum");
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut any = false;
    for &x in xs.iter().chain(std::iter::once(&approx)) {
        assert!(x.is_finite(), "compensated_to_bigfloat_check requires finite inputs");
        if x != 0.0_f64 {
            let s = BigFloat::from_f64(x, 53).scale();
            lo = lo.min(s);
            hi = hi.max(s);
            any = true;
        }
    }
    if !any {
        return 0.0_f64;
    }
    // Widest span of bits any partial sum can occupy, plus room for the
    // carries out of the top and the 53 bits below the smallest term.
    let span = usize::try_from(hi - lo).expect("finite f64 exponent span");
    let headroom = usize::BITS as usize - xs.len().leading_zeros() as usize;
    let prec = span + 53 + headroom + 8;
    let mut exact = BigFloat::zero(prec);
    for &x in xs {
        exact = exact.add_prec(&BigFloat::from_f64(x, prec), prec);
    }
    BigFloat::from_f64(approx, prec).sub_prec(&exact, prec).to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    // Published reference expansions, truncated (not rounded) to 100
    // decimal places.
    const PI_100: &str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679";
    const E_100: &str = "2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274";
    const SQRT2_100: &str = "1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727";
    const LN2_100: &str = "0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875";
    const AGM1SQRT2_100: &str = "1.1981402347355922074399224922803238782272126632156515582636749529464052141439156708358855564897933893";
    const GAUSS_100: &str = "0.8346268416740731862814297327990468089939930134903470024498273701036819927095264118696911603512753241";

    fn bf(s: &str, precision: usize) -> BigFloat {
        BigFloat::from_str(s, precision).expect("valid literal")
    }

    /// Assert `a` and `b` agree to at least `bits` relative bits.
    fn assert_close(a: &BigFloat, b: &BigFloat, bits: i64, what: &str) {
        let d = a.sub(b).abs();
        if d.is_zero() {
            return;
        }
        assert!(!b.is_zero(), "{what}: expected exactly zero, got {}", a.to_f64());
        let rel = d.scale() - b.abs().scale();
        assert!(rel <= -bits, "{what}: relative error 2^({rel}) exceeds 2^(-{bits})");
    }

    /// A random `f64` with a full 53-bit significand and a moderate
    /// exponent, so sums, products and quotients stay in the normal range.
    fn rand_f64(rng: &mut Rng) -> f64 {
        let m = (rng.next_u64() >> 11) | (1_u64 << 52);
        let e = (rng.next_u64() % 121) as i32 - 60;
        let v = (m as f64) * f64::powi(2.0_f64, e - 52);
        if rng.next_u64().is_multiple_of(2) {
            v
        } else {
            -v
        }
    }

    /// A random `BigFloat` with a full `precision`-bit significand.
    fn rand_bigfloat(rng: &mut Rng, precision: usize) -> BigFloat {
        let mut m = BigInt::random_bits(precision, rng);
        m.set_bit(precision - 1, true);
        if !rng.next_u64().is_multiple_of(2) {
            m = m.neg();
        }
        let e = (rng.next_u64() % 41) as i64 - 20;
        BigFloat::new(m, e, precision)
    }

    // -- canonical form -------------------------------------------------

    #[test]
    fn test_canonical_form_is_maintained() {
        let mut rng = Rng::new(0xC0FFEE);
        let mut values = vec![
            BigFloat::zero(53),
            BigFloat::one(53),
            BigFloat::from_f64(0.1, 53),
            bf("-12345.6789", 90),
            BigFloat::pi(120),
            BigFloat::e(120),
            BigFloat::ln2(120),
        ];
        for _ in 0..20 {
            let a = rand_bigfloat(&mut rng, 77);
            let b = rand_bigfloat(&mut rng, 40);
            values.push(a.add(&b));
            values.push(a.sub(&b));
            values.push(a.mul(&b));
            values.push(a.div(&b));
            values.push(a.abs().sqrt());
        }
        for v in &values {
            if v.is_zero() {
                assert_eq!(v.exponent, 0, "zero must have a zero exponent");
                assert_eq!(v.mantissa.bits(), 0);
            } else {
                assert_eq!(
                    v.mantissa.bits(),
                    v.precision,
                    "mantissa must carry exactly `precision` bits"
                );
            }
        }
    }

    #[test]
    fn test_from_f64_bit_pattern() {
        let one = BigFloat::from_f64(1.0_f64, 53);
        assert_eq!(one.mantissa, BigInt::one().shl(52));
        assert_eq!(one.exponent, -52);
        let half = BigFloat::from_f64(-0.5_f64, 8);
        assert_eq!(half.mantissa, BigInt::from_i64(-128));
        assert_eq!(half.exponent, -8);
        assert!(BigFloat::from_f64(0.0_f64, 53).is_zero());
    }

    // -- exactness on dyadic values -------------------------------------

    #[test]
    fn test_from_f64_roundtrip_is_exact() {
        let samples = [
            0.0_f64,
            1.0_f64,
            -1.0_f64,
            0.1_f64,
            -3.5_f64,
            1e300_f64,
            1e-300_f64,
            f64::MIN_POSITIVE,
            f64::MAX,
            std::f64::consts::PI,
            5e-324_f64, // smallest subnormal
            2.5e-320_f64,
        ];
        for &x in &samples {
            for prec in [53_usize, 64, 200] {
                let v = BigFloat::from_f64(x, prec);
                assert_eq!(v.to_f64(), x, "round trip of {x} at {prec} bits");
            }
        }
    }

    #[test]
    fn test_to_f64_overflow_and_underflow() {
        let big = BigFloat::one(53).mul_pow2(2000);
        assert_eq!(big.to_f64(), f64::INFINITY);
        assert_eq!(big.neg().to_f64(), f64::NEG_INFINITY);
        // 2^-1075 is exactly half the smallest subnormal: ties to even -> 0.
        let half_ulp = BigFloat::one(53).mul_pow2(-1075);
        assert_eq!(half_ulp.to_f64(), 0.0_f64);
        // Just above that tie rounds up to the smallest subnormal.
        let above = BigFloat::from_i64(3, 53).mul_pow2(-1076);
        assert_eq!(above.to_f64(), 5e-324_f64);
        assert_eq!(BigFloat::one(53).mul_pow2(-2000).to_f64(), 0.0_f64);
    }

    #[test]
    fn test_sum_of_f64s_is_exact_at_ample_precision() {
        let xs = [1.0_f64, 1e100_f64, 1.0_f64, -1e100_f64];
        let mut sum = BigFloat::zero(400);
        for &x in &xs {
            sum = sum.add_prec(&BigFloat::from_f64(x, 400), 400);
        }
        assert_eq!(sum.to_f64(), 2.0_f64);
        assert_eq!(sum, BigFloat::from_i64(2, 8));
    }

    // -- correct rounding ------------------------------------------------

    #[test]
    fn test_arithmetic_reproduces_ieee_double() {
        let mut rng = Rng::new(20240824);
        for _ in 0..300 {
            let a = rand_f64(&mut rng);
            let b = rand_f64(&mut rng);
            let (x, y) = (BigFloat::from_f64(a, 53), BigFloat::from_f64(b, 53));
            assert_eq!(x.add_prec(&y, 53).to_f64(), a + b, "add {a} + {b}");
            assert_eq!(x.sub_prec(&y, 53).to_f64(), a - b, "sub {a} - {b}");
            assert_eq!(x.mul_prec(&y, 53).to_f64(), a * b, "mul {a} * {b}");
            assert_eq!(x.div_prec(&y, 53).to_f64(), a / b, "div {a} / {b}");
            assert_eq!(x.abs().sqrt_prec(53).to_f64(), a.abs().sqrt(), "sqrt {a}");
        }
    }

    #[test]
    fn test_operations_are_correctly_rounded_against_higher_precision() {
        // Computing at p, and computing at p + 192 then rounding to p,
        // must agree: that is what "correctly rounded" means.
        let mut rng = Rng::new(0x5EED_1234);
        const P: usize = 64;
        const REF: usize = P + 192;
        for _ in 0..200 {
            let a = rand_bigfloat(&mut rng, P);
            let b = rand_bigfloat(&mut rng, P);
            assert_eq!(a.add_prec(&b, P), a.add_prec(&b, REF).round_to(P), "add");
            assert_eq!(a.sub_prec(&b, P), a.sub_prec(&b, REF).round_to(P), "sub");
            assert_eq!(a.mul_prec(&b, P), a.mul_prec(&b, REF).round_to(P), "mul");
            assert_eq!(a.div_prec(&b, P), a.div_prec(&b, REF).round_to(P), "div");
            let c = a.abs();
            assert_eq!(c.sqrt_prec(P), c.sqrt_prec(REF).round_to(P), "sqrt");
        }
    }

    #[test]
    fn test_multiplication_is_exact_when_precision_allows() {
        let mut rng = Rng::new(77);
        for _ in 0..50 {
            let a = rand_bigfloat(&mut rng, 64);
            let b = rand_bigfloat(&mut rng, 64);
            let exact = BigFloat::new(
                a.mantissa.mul(&b.mantissa),
                a.exponent + b.exponent,
                200,
            );
            assert_eq!(a.mul_prec(&b, 200), exact);
        }
    }

    #[test]
    fn test_round_to_uses_ties_to_even() {
        // 5 = 0b101 -> 2 bits: tie, round down to 0b10 -> 4.
        assert_eq!(BigFloat::from_i64(5, 8).round_to(2).to_f64(), 4.0_f64);
        // 7 = 0b111 -> 2 bits: tie, round up to 0b100 -> 8.
        assert_eq!(BigFloat::from_i64(7, 8).round_to(2).to_f64(), 8.0_f64);
        // 3 = 0b11 is exact at 2 bits.
        assert_eq!(BigFloat::from_i64(3, 8).round_to(2).to_f64(), 3.0_f64);
        // 13 = 0b1101 -> 2 bits: above the tie, rounds up to 0b100 -> 16.
        assert_eq!(BigFloat::from_i64(13, 8).round_to(2).to_f64(), 16.0_f64);
        assert_eq!(BigFloat::from_i64(-5, 8).round_to(2).to_f64(), -4.0_f64);
    }

    #[test]
    fn test_addition_survives_catastrophic_cancellation() {
        // 1 - (1 - 2^-600) must be exactly 2^-600, even though the target
        // precision is only 53 bits.
        let one = BigFloat::one(53);
        let almost = BigFloat::new(BigInt::one().shl(600).sub(&BigInt::one()), -600, 600);
        let d = one.sub_prec(&almost, 53);
        assert_eq!(d, BigFloat::one(53).mul_pow2(-600));
        // A tiny addend well below the ulp must round away entirely.
        let tiny = BigFloat::one(53).mul_pow2(-1000);
        assert_eq!(one.add_prec(&tiny, 53), one);
        // ... but it must still break a tie in the right direction.
        let tie = BigFloat::new(BigInt::from_i64(0b1_0000_0001), 0, 9);
        assert_eq!(tie.round_to(8).to_f64(), 256.0_f64);
        assert_eq!(tie.add_prec(&tiny, 8).to_f64(), 258.0_f64);
    }

    // -- square root -----------------------------------------------------

    #[test]
    fn test_sqrt_exact_and_inverse() {
        assert_eq!(BigFloat::from_i64(4, 53).sqrt(), BigFloat::from_i64(2, 53));
        assert_eq!(BigFloat::from_i64(1 << 40, 53).sqrt(), BigFloat::from_i64(1 << 20, 53));
        assert!(BigFloat::zero(53).sqrt().is_zero());
        let big = BigFloat::from_bigint(&BigInt::from_u64(10).pow(40), 200);
        assert_eq!(big.mul_prec(&big, 400).sqrt_prec(400), big);
        let mut rng = Rng::new(999);
        for _ in 0..40 {
            let x = rand_bigfloat(&mut rng, 160).abs();
            let r = x.sqrt();
            assert_close(&r.mul_prec(&r, 160), &x, 158, "sqrt(x)^2");
        }
    }

    // -- pi, e, ln2, sqrt2 ------------------------------------------------

    #[test]
    fn test_pi_digits_match_published_expansion() {
        assert_eq!(pi_digits(100), PI_100);
        assert_eq!(pi_digits(0), "3");
        assert_eq!(pi_digits(5), "3.14159");
        assert_eq!(pi_digits(30), PI_100[..32]);
    }

    #[test]
    fn test_e_and_sqrt2_digits_match_published_expansion() {
        assert_eq!(e_digits(100), E_100);
        assert_eq!(sqrt2_digits(100), SQRT2_100);
        assert_eq!(e_digits(15), E_100[..17]);
        assert_eq!(sqrt2_digits(15), SQRT2_100[..17]);
    }

    #[test]
    fn test_ln2_digits_match_published_expansion() {
        let l = BigFloat::ln2(digits_to_bits(100));
        assert_eq!(l.digit_string(100, false), LN2_100);
    }

    #[test]
    fn test_machin_pi_agrees_with_agm_pi() {
        // Two independent algorithms: an arctangent series and the
        // Gauss-Legendre AGM iteration.
        for prec in [64_usize, 256, 700] {
            let a = machin_pi(prec);
            let b = BigFloat::pi(prec);
            assert_close(&a, &b, prec as i64 - 4, "machin vs agm pi");
        }
        assert_eq!(machin_pi(700).round_to(650), BigFloat::pi(700).round_to(650));
    }

    #[test]
    fn test_pi_matches_std_f64() {
        assert_eq!(BigFloat::pi(53).to_f64(), std::f64::consts::PI);
        assert_eq!(machin_pi(53).to_f64(), std::f64::consts::PI);
        assert_eq!(BigFloat::e(53).to_f64(), std::f64::consts::E);
        assert_eq!(BigFloat::ln2(53).to_f64(), std::f64::consts::LN_2);
        assert_eq!(BigFloat::from_i64(2, 53).sqrt().to_f64(), std::f64::consts::SQRT_2);
    }

    // -- elementary functions --------------------------------------------

    #[test]
    fn test_exp_ln_are_mutual_inverses() {
        let prec = 240;
        for s in ["0.5", "1", "2", "7.25", "1234.5", "0.001", "1e-8", "1e12"] {
            let x = bf(s, prec);
            assert_close(&x.ln().exp(), &x, prec as i64 - 24, "exp(ln x)");
            assert_close(&x.exp().ln(), &x, prec as i64 - 24, "ln(exp x)");
        }
        assert!(BigFloat::one(120).ln().is_zero());
        assert_eq!(BigFloat::zero(120).exp(), BigFloat::one(120));
    }

    #[test]
    fn test_exp_and_ln_against_the_constants() {
        let prec = 300;
        assert_close(&BigFloat::one(prec).exp(), &BigFloat::e(prec), prec as i64 - 16, "exp(1)");
        assert_close(
            &BigFloat::from_i64(2, prec).ln(),
            &BigFloat::ln2(prec),
            prec as i64 - 16,
            "ln 2",
        );
        assert_close(
            &BigFloat::e(prec).ln(),
            &BigFloat::one(prec),
            prec as i64 - 16,
            "ln e",
        );
        // exp of a negative argument is the reciprocal.
        let x = bf("-3.75", prec);
        assert_close(
            &x.exp().mul(&x.neg().exp()),
            &BigFloat::one(prec),
            prec as i64 - 16,
            "exp(x)exp(-x)",
        );
    }

    #[test]
    fn test_exp_ln_match_f64() {
        let mut rng = Rng::new(4242);
        for _ in 0..40 {
            let x = rng.next_f64() * 20.0_f64 - 10.0_f64;
            let got = BigFloat::from_f64(x, 80).exp().to_f64();
            assert!((got - x.exp()).abs() <= x.exp().abs() * 1e-14_f64, "exp({x})");
            let y = rng.next_f64() * 100.0_f64 + 1e-3_f64;
            let got = BigFloat::from_f64(y, 80).ln().to_f64();
            assert!((got - y.ln()).abs() <= y.ln().abs() * 1e-14_f64 + 1e-14_f64, "ln({y})");
        }
    }

    #[test]
    fn test_pythagorean_identity_for_sin_cos() {
        let prec = 200;
        for s in ["0", "0.5", "1", "3", "-7.25", "1000", "123456.789"] {
            let x = bf(s, prec);
            let (sn, cs) = x.sin_cos();
            let id = sn.mul_prec(&sn, prec).add_prec(&cs.mul_prec(&cs, prec), prec);
            assert_close(&id, &BigFloat::one(prec), prec as i64 - 24, "sin^2 + cos^2");
            assert_eq!(sn, x.sin());
            assert_eq!(cs, x.cos());
        }
    }

    #[test]
    fn test_sin_cos_known_values() {
        let prec = 200;
        let pi = BigFloat::pi(prec);
        let half = bf("0.5", prec);
        assert_close(
            &pi.div_prec(&BigFloat::from_i64(6, prec), prec).sin(),
            &half,
            prec as i64 - 24,
            "sin(pi/6)",
        );
        assert_close(
            &pi.div_prec(&BigFloat::from_i64(3, prec), prec).cos(),
            &half,
            prec as i64 - 24,
            "cos(pi/3)",
        );
        assert_close(
            &pi.mul_pow2(-1).sin(),
            &BigFloat::one(prec),
            prec as i64 - 24,
            "sin(pi/2)",
        );
        assert_close(&pi.cos(), &BigFloat::one(prec).neg(), prec as i64 - 24, "cos(pi)");
        // sin(pi) is not exactly zero at finite precision, but it must be
        // smaller than 2^-(prec-8) relative to 1.
        assert!(pi.sin().abs().scale() < -(prec as i64) + 8, "sin(pi) must be tiny");
        // Angle-addition identity: sin(2x) = 2 sin x cos x.
        let x = bf("0.8", prec);
        let (s, c) = x.sin_cos();
        assert_close(
            &x.mul_pow2(1).sin(),
            &s.mul_prec(&c, prec).mul_pow2(1),
            prec as i64 - 24,
            "sin(2x)",
        );
    }

    #[test]
    fn test_sin_cos_match_f64() {
        let mut rng = Rng::new(31337);
        for _ in 0..40 {
            let x = (rng.next_f64() - 0.5_f64) * 200.0_f64;
            let v = BigFloat::from_f64(x, 90);
            assert!((v.sin().to_f64() - x.sin()).abs() < 1e-14_f64, "sin({x})");
            assert!((v.cos().to_f64() - x.cos()).abs() < 1e-14_f64, "cos({x})");
        }
    }

    #[test]
    fn test_atan_identities() {
        let prec = 240;
        let pi = BigFloat::pi(prec);
        assert_close(
            &BigFloat::one(prec).atan().mul_pow2(2),
            &pi,
            prec as i64 - 16,
            "4 atan(1) == pi",
        );
        assert!(BigFloat::zero(prec).atan().is_zero());
        for s in ["0.25", "1", "3", "97.5"] {
            let x = bf(s, prec);
            let sum = x.atan().add_prec(&BigFloat::one(prec).div_prec(&x, prec).atan(), prec);
            assert_close(&sum, &pi.mul_pow2(-1), prec as i64 - 16, "atan x + atan 1/x");
            // tan(atan x) == x, via sin/cos of the arctangent.
            let (s_, c_) = x.atan().sin_cos();
            assert_close(&s_.div_prec(&c_, prec), &x, prec as i64 - 24, "tan(atan x)");
        }
        assert_eq!(bf("-1", prec).atan(), BigFloat::one(prec).atan().neg());
    }

    #[test]
    fn test_atan_matches_f64() {
        let mut rng = Rng::new(2718);
        for _ in 0..40 {
            let x = (rng.next_f64() - 0.5_f64) * 40.0_f64;
            let got = BigFloat::from_f64(x, 90).atan().to_f64();
            assert!((got - x.atan()).abs() < 1e-15_f64, "atan({x})");
        }
    }

    // -- powers ------------------------------------------------------------

    #[test]
    fn test_integer_powers_are_exact() {
        assert_eq!(BigFloat::from_i64(2, 53).powi(10), BigFloat::from_i64(1024, 53));
        assert_eq!(BigFloat::from_i64(-3, 53).powi(3), BigFloat::from_i64(-27, 53));
        assert_eq!(BigFloat::from_i64(7, 53).powi(0), BigFloat::one(53));
        assert_eq!(
            BigFloat::from_i64(2, 53).powi(-3),
            bf("0.125", 53),
        );
        // 3^40 needs 64 bits and must come out exactly at that precision.
        let expected = BigFloat::from_bigint(&BigInt::from_u64(3).pow(40), 200);
        assert_eq!(BigFloat::from_i64(3, 200).powi(40), expected);
    }

    #[test]
    fn test_pow_general_exponent() {
        let prec = 200;
        let two = BigFloat::from_i64(2, prec);
        assert_close(
            &two.pow(&bf("0.5", prec)),
            &two.sqrt(),
            prec as i64 - 16,
            "2^0.5 == sqrt 2",
        );
        // x^y * x^-y == 1
        let x = bf("3.7", prec);
        let y = bf("2.25", prec);
        assert_close(
            &x.pow(&y).mul_prec(&x.pow(&y.neg()), prec),
            &BigFloat::one(prec),
            prec as i64 - 24,
            "x^y x^-y",
        );
        // (x^y)^(1/y) == x
        let inv = BigFloat::one(prec).div_prec(&y, prec);
        assert_close(&x.pow(&y).pow(&inv), &x, prec as i64 - 24, "(x^y)^(1/y)");
        // An integral BigFloat exponent takes the exact integer path.
        assert_eq!(x.pow(&BigFloat::from_i64(3, prec)), x.powi(3));
    }

    // -- AGM ---------------------------------------------------------------

    #[test]
    fn test_agm_properties_and_gauss_constant() {
        let prec = 400;
        let one = BigFloat::one(prec);
        let sqrt2 = BigFloat::from_i64(2, prec).sqrt();
        let x = bf("3.5", prec);
        assert_eq!(BigFloat::agm(&x, &x), x);
        assert!(BigFloat::agm(&x, &BigFloat::zero(prec)).is_zero());
        // Homogeneity: agm(c a, c b) = c agm(a, b).
        let c = bf("7.25", prec);
        let a = bf("1.5", prec);
        let b = bf("9.75", prec);
        assert_close(
            &BigFloat::agm(&a.mul_prec(&c, prec), &b.mul_prec(&c, prec)),
            &BigFloat::agm(&a, &b).mul_prec(&c, prec),
            prec as i64 - 16,
            "agm homogeneity",
        );
        // The mean is bracketed by its arguments.
        let m = BigFloat::agm(&a, &b);
        assert!(m > a && m < b);
        // Published digits of agm(1, sqrt 2) and of Gauss's constant.
        let g = BigFloat::agm(&one, &sqrt2);
        assert_eq!(g.round_to(digits_to_bits(100)).digit_string(100, false), AGM1SQRT2_100);
        let gauss = one.div_prec(&g, prec);
        assert_eq!(
            gauss.round_to(digits_to_bits(100)).digit_string(100, false),
            GAUSS_100
        );
    }

    // -- comparison ---------------------------------------------------------

    #[test]
    fn test_cmp_agrees_with_f64() {
        let mut rng = Rng::new(0xABCD);
        for _ in 0..300 {
            let a = rand_f64(&mut rng);
            let b = rand_f64(&mut rng);
            let x = BigFloat::from_f64(a, 53);
            let y = BigFloat::from_f64(b, 53);
            assert_eq!(x.cmp(&y), a.partial_cmp(&b).expect("finite"), "cmp {a} ? {b}");
        }
    }

    #[test]
    fn test_cmp_ignores_precision_and_orders_signs() {
        assert_eq!(BigFloat::one(53), BigFloat::one(400));
        assert_eq!(BigFloat::zero(53), BigFloat::zero(400));
        assert!(BigFloat::one(53).neg() < BigFloat::zero(9));
        assert!(BigFloat::zero(9) < BigFloat::one(53));
        // Same scale, different mantissa lengths.
        let a = BigFloat::new(BigInt::from_i64(3), -1, 2); // 1.5
        let b = BigFloat::new(BigInt::from_i64(13), -3, 4); // 1.625
        assert!(a < b);
        assert!(b.neg() < a.neg());
        // Different scales.
        assert!(BigFloat::one(8).mul_pow2(-100) < BigFloat::one(8).mul_pow2(100));
        let mut v = vec![bf("3", 60), bf("-1", 60), bf("0", 60), bf("2.5", 60)];
        v.sort();
        assert_eq!(
            v.iter().map(BigFloat::to_f64).collect::<Vec<_>>(),
            vec![-1.0_f64, 0.0_f64, 2.5_f64, 3.0_f64]
        );
    }

    // -- strings ------------------------------------------------------------

    #[test]
    fn test_from_str_is_correctly_rounded_like_the_rust_parser() {
        let samples = [
            "0.1",
            "0.2",
            "3.141592653589793",
            "2.718281828459045",
            "1e-5",
            "-7.25e3",
            "123456789.123456789",
            "0.30000000000000004",
            "9007199254740993",
            "1.7976931348623157e308",
            "2.5e-300",
            "+42",
            "0",
            "-0.0",
        ];
        for s in samples {
            let expected: f64 = s.parse().expect("valid f64 literal");
            assert_eq!(bf(s, 53).to_f64(), expected, "parsing {s}");
        }
    }

    #[test]
    fn test_from_str_rejects_garbage() {
        for s in ["", "   ", "abc", "1.2.3", "1e", "1e5x", "--1", "1e99999", ".", "1e-99999"] {
            assert!(BigFloat::from_str(s, 53).is_err(), "{s} must be rejected");
        }
        assert!(BigFloat::from_str(".5", 53).is_ok());
        assert!(BigFloat::from_str("5.", 53).is_ok());
    }

    #[test]
    fn test_to_string_decimal() {
        assert_eq!(bf("0.5", 53).to_string_decimal(3), "0.500");
        assert_eq!(bf("-2.25", 53).to_string_decimal(2), "-2.25");
        assert_eq!(BigFloat::zero(53).to_string_decimal(4), "0.0000");
        assert_eq!(bf("12345", 53).to_string_decimal(0), "12345");
        // Ties to even on the printed digit: 0.125 -> 0.12, 0.375 -> 0.38.
        assert_eq!(bf("0.125", 53).to_string_decimal(2), "0.12");
        assert_eq!(bf("0.375", 53).to_string_decimal(2), "0.38");
        // Carrying into a new integer digit.
        assert_eq!(bf("0.999", 60).to_string_decimal(2), "1.00");
        // A third correctly rounded to 40 places.
        let third = BigFloat::one(200).div_prec(&BigFloat::from_i64(3, 200), 200);
        assert_eq!(third.to_string_decimal(40), format!("0.{}", "3".repeat(40)));
        let two_thirds = third.mul_pow2(1);
        assert_eq!(two_thirds.to_string_decimal(10), "0.6666666667");
        assert_eq!(BigFloat::pi(200).to_string_decimal(20), "3.14159265358979323846");
    }

    // -- compensated summation cross-check -----------------------------------

    #[test]
    fn test_compensated_check_is_zero_when_the_sum_is_exact() {
        assert_eq!(compensated_to_bigfloat_check(&[]), 0.0_f64);
        assert_eq!(compensated_to_bigfloat_check(&[1.0, 2.0, 3.0]), 0.0_f64);
        // The classic catastrophic case: Neumaier gets it exactly right.
        assert_eq!(compensated_to_bigfloat_check(&[1.0, 1e100, 1.0, -1e100]), 0.0_f64);
        assert_eq!(compensated_to_bigfloat_check(&[0.1, 0.2, -0.3]), 0.0_f64);
    }

    #[test]
    fn test_compensated_check_bounds_the_neumaier_error() {
        let xs = vec![0.1_f64; 1000];
        let err = compensated_to_bigfloat_check(&xs);
        // The exact sum is 1000 * 0.1(binary) which is not 100 exactly;
        // Neumaier must land within a couple of ulps of it.
        assert!(err.abs() <= 4.0_f64 * f64::EPSILON * 100.0_f64, "neumaier error {err}");
        // The naive sum is much worse: check that the exact error of the
        // naive loop is strictly larger.
        let naive: f64 = xs.iter().sum();
        let mut exact = BigFloat::zero(200);
        for &x in &xs {
            exact = exact.add_prec(&BigFloat::from_f64(x, 200), 200);
        }
        let naive_err = BigFloat::from_f64(naive, 200).sub_prec(&exact, 200).to_f64();
        assert!(naive_err.abs() > err.abs(), "naive {naive_err} vs neumaier {err}");
        // Random data with a wide dynamic range.
        let mut rng = Rng::new(555);
        let mut data = Vec::new();
        for _ in 0..200 {
            data.push(rand_f64(&mut rng));
        }
        let e = compensated_to_bigfloat_check(&data);
        let magnitude = data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(e.abs() <= 8.0_f64 * f64::EPSILON * magnitude, "error {e} too large");
    }

    // -- panics ---------------------------------------------------------------

    #[test]
    #[should_panic(expected = "negative")]
    fn test_sqrt_of_negative_panics() {
        let _ = BigFloat::from_i64(-4, 53).sqrt();
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_division_by_zero_panics() {
        let _ = BigFloat::one(53).div(&BigFloat::zero(53));
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn test_ln_of_zero_panics() {
        let _ = BigFloat::zero(53).ln();
    }

    #[test]
    #[should_panic(expected = "at least 2 bits")]
    fn test_tiny_precision_panics() {
        let _ = BigFloat::one(1);
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn test_from_f64_rejects_nan() {
        let _ = BigFloat::from_f64(f64::NAN, 53);
    }
}
