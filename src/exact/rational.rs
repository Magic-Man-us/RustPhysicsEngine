//! Exact rational arithmetic over [`BigInt`].
//!
//! Every `Rational` is kept in lowest terms with a strictly positive
//! denominator, so equality is structural and there is exactly one
//! representation of each value. Zero is `0/1`.

use crate::error::GeomError;
use crate::exact::bigint::BigInt;
use crate::linalg::Matrix;
use std::cmp::Ordering;

/// An exact rational number, always reduced with `den > 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rational {
    pub num: BigInt,
    pub den: BigInt,
}

/// Multiply by `2^e` without overflowing intermediate `powi` calls, which
/// are limited to exponents inside the `f64` range.
fn scale_pow2(mut x: f64, mut e: i64) -> f64 {
    while e > 1000 {
        x *= 2f64.powi(1000);
        e -= 1000;
        if !x.is_finite() {
            return x;
        }
    }
    while e < -1000 {
        x *= 2f64.powi(-1000);
        e += 1000;
        if x == 0.0 {
            return x;
        }
    }
    x * 2f64.powi(e as i32)
}

impl Rational {
    /// Reduce a numerator and denominator into canonical form.
    fn reduced(mut num: BigInt, mut den: BigInt) -> Self {
        if den.is_negative() {
            num = num.neg();
            den = den.neg();
        }
        let g = num.gcd(&den);
        if g != BigInt::one() && !g.is_zero() {
            num = num.div_rem(&g).0;
            den = den.div_rem(&g).0;
        }
        if num.is_zero() {
            den = BigInt::one();
        }
        Rational { num, den }
    }

    /// The rational `n/d`, or `None` when `d` is zero.
    #[must_use]
    pub fn new(n: BigInt, d: BigInt) -> Option<Self> {
        if d.is_zero() {
            return None;
        }
        Some(Self::reduced(n, d))
    }

    /// The rational `n/d` from machine integers.
    ///
    /// # Panics
    /// Panics if `d` is zero.
    #[must_use]
    pub fn from_i64(n: i64, d: i64) -> Self {
        assert!(d != 0, "zero denominator");
        Self::reduced(BigInt::from_i64(n), BigInt::from_i64(d))
    }

    /// The integer `n` as a rational.
    #[must_use]
    pub fn from_int(n: BigInt) -> Self {
        Rational { num: n, den: BigInt::one() }
    }

    #[must_use]
    pub fn zero() -> Self {
        Rational { num: BigInt::zero(), den: BigInt::one() }
    }

    #[must_use]
    pub fn one() -> Self {
        Rational { num: BigInt::one(), den: BigInt::one() }
    }

    /// The exact value of an IEEE-754 double, or `None` for NaN and the
    /// infinities.
    ///
    /// Every finite `f64` is a dyadic rational `m * 2^e`, so the reduced
    /// denominator is always a power of two.
    #[must_use]
    pub fn from_f64_exact(x: f64) -> Option<Self> {
        if !x.is_finite() {
            return None;
        }
        if x == 0.0 {
            return Some(Self::zero());
        }
        let bits = x.to_bits();
        let sign = if bits >> 63 == 1 { -1i64 } else { 1i64 };
        let raw_exp = ((bits >> 52) & 0x7ff) as i64;
        let raw_frac = bits & 0x000f_ffff_ffff_ffff;
        // Subnormals have no implicit leading bit and a fixed exponent.
        let (mantissa, exp2) = if raw_exp == 0 {
            (raw_frac, -1074i64)
        } else {
            (raw_frac | (1 << 52), raw_exp - 1075)
        };
        let m = BigInt::from_u64(mantissa).mul(&BigInt::from_i64(sign));
        Some(if exp2 >= 0 {
            Self::from_int(m.shl(exp2 as usize))
        } else {
            Self::reduced(m, BigInt::one().shl((-exp2) as usize))
        })
    }

    /// The best rational approximation to `x` with denominator at most
    /// `max_den`, found by walking the continued fraction (equivalently,
    /// descending the Stern-Brocot tree).
    ///
    /// # Panics
    /// Panics if `max_den` is zero or `x` is not finite.
    #[must_use]
    pub fn from_f64_approx(x: f64, max_den: u64) -> Self {
        assert!(max_den > 0, "max_den must be positive");
        assert!(x.is_finite(), "x must be finite");
        best_rational_approximations(x, max_den)
            .pop()
            .unwrap_or_else(Self::zero)
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.num.is_negative()
    }

    #[must_use]
    pub fn is_integer(&self) -> bool {
        self.den == BigInt::one()
    }

    #[must_use]
    pub fn abs(&self) -> Self {
        Rational { num: self.num.abs(), den: self.den.clone() }
    }

    #[must_use]
    pub fn neg(&self) -> Self {
        Rational { num: self.num.neg(), den: self.den.clone() }
    }

    /// The reciprocal, or `None` for zero.
    #[must_use]
    pub fn recip(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(Self::reduced(self.den.clone(), self.num.clone()))
    }

    #[must_use]
    pub fn add(&self, other: &Rational) -> Self {
        // Divide by the gcd of the denominators first so the intermediate
        // products stay as small as the result allows.
        let g = self.den.gcd(&other.den);
        let a = self.den.div_rem(&g).0;
        let b = other.den.div_rem(&g).0;
        let num = self.num.mul(&b).add(&other.num.mul(&a));
        Self::reduced(num, a.mul(&other.den))
    }

    #[must_use]
    pub fn sub(&self, other: &Rational) -> Self {
        self.add(&other.neg())
    }

    #[must_use]
    pub fn mul(&self, other: &Rational) -> Self {
        Self::reduced(self.num.mul(&other.num), self.den.mul(&other.den))
    }

    /// Quotient, or `None` when `other` is zero.
    #[must_use]
    pub fn div(&self, other: &Rational) -> Option<Self> {
        if other.is_zero() {
            return None;
        }
        Some(Self::reduced(self.num.mul(&other.den), self.den.mul(&other.num)))
    }

    /// `self` raised to a signed integer power.
    ///
    /// # Panics
    /// Panics when raising zero to a negative power.
    #[must_use]
    pub fn pow(&self, e: i64) -> Self {
        if e >= 0 {
            let k = e.unsigned_abs();
            Self::reduced(self.num.pow(k), self.den.pow(k))
        } else {
            assert!(!self.is_zero(), "zero to a negative power");
            let k = e.unsigned_abs();
            Self::reduced(self.den.pow(k), self.num.pow(k))
        }
    }

    /// Nearest `f64`.
    ///
    /// When both parts are individually representable the quotient is a
    /// single correctly-rounded division. Otherwise -- a tiny value like
    /// `1e-300` has a denominator of about `2^1049`, far past the `f64`
    /// range even though the quotient is fine -- the numerator is scaled
    /// by a power of two first so the quotient itself lands in range, and
    /// the scale is undone afterwards.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let n = self.num.to_f64();
        let d = self.den.to_f64();
        if n.is_finite() && d.is_finite() && n != 0.0 && d != 0.0 {
            return n / d;
        }
        // Pick a shift that leaves the quotient about 64 bits wide.
        let bn = self.num.bits() as i64;
        let bd = self.den.bits() as i64;
        let s = 64 + bd - bn;
        let scaled = if s >= 0 {
            self.num.shl(s as usize)
        } else {
            self.num.shr((-s) as usize)
        };
        let q = scaled.div_rem(&self.den).0.to_f64();
        scale_pow2(q, -s)
    }

    /// Greatest integer not exceeding the value.
    #[must_use]
    pub fn floor(&self) -> BigInt {
        let (q, r) = self.num.div_rem(&self.den);
        if self.num.is_negative() && !r.is_zero() {
            q.sub(&BigInt::one())
        } else {
            q
        }
    }

    /// Least integer not below the value.
    #[must_use]
    pub fn ceil(&self) -> BigInt {
        let (q, r) = self.num.div_rem(&self.den);
        if !self.num.is_negative() && !r.is_zero() {
            q.add(&BigInt::one())
        } else {
            q
        }
    }

    /// Nearest integer, with halves rounded away from zero.
    #[must_use]
    pub fn round(&self) -> BigInt {
        let half = Rational::from_i64(1, 2);
        if self.is_negative() {
            self.sub(&half).ceil()
        } else {
            self.add(&half).floor()
        }
    }

    /// The fractional part `self - floor(self)`, always in `[0, 1)`.
    #[must_use]
    pub fn fract(&self) -> Self {
        self.sub(&Self::from_int(self.floor()))
    }

    /// The continued-fraction expansion `[a0; a1, a2, ...]`.
    ///
    /// The expansion is finite for every rational and, apart from the
    /// integer case, never ends in a 1, which makes it canonical.
    #[must_use]
    pub fn to_continued_fraction(&self) -> Vec<BigInt> {
        let mut out = Vec::new();
        let mut n = self.num.clone();
        let mut d = self.den.clone();
        while !d.is_zero() {
            let mut q = n.div_rem(&d).0;
            let mut r = n.sub(&q.mul(&d));
            // Floor rather than truncate, so partial quotients stay positive.
            if r.is_negative() {
                q = q.sub(&BigInt::one());
                r = r.add(&d);
            }
            out.push(q);
            n = d;
            d = r;
        }
        // Normalize a trailing 1: [.., a, 1] == [.., a+1].
        if out.len() > 1 && out.last() == Some(&BigInt::one()) {
            out.pop();
            let last = out.pop().expect("at least one term remains");
            out.push(last.add(&BigInt::one()));
        }
        out
    }

    /// Rebuild a rational from a continued fraction.
    ///
    /// # Errors
    /// Returns [`GeomError::Empty`] for an empty expansion, and
    /// [`GeomError::InvalidArgument`] if a non-leading term is not
    /// positive, which cannot arise from [`Self::to_continued_fraction`].
    pub fn from_continued_fraction(cf: &[BigInt]) -> Result<Self, GeomError> {
        if cf.is_empty() {
            return Err(GeomError::Empty);
        }
        if cf[1..].iter().any(|a| !a.is_negative() && a.is_zero() || a.is_negative()) {
            return Err(GeomError::InvalidArgument("continued fraction terms must be positive"));
        }
        let mut acc = Self::from_int(cf[cf.len() - 1].clone());
        for a in cf[..cf.len() - 1].iter().rev() {
            acc = Self::from_int(a.clone())
                .add(&acc.recip().ok_or(GeomError::InvalidArgument("zero term"))?);
        }
        Ok(acc)
    }

    /// The mediant `(a.num + b.num) / (a.den + b.den)`.
    ///
    /// The mediant of two fractions always lies strictly between them, the
    /// property the Stern-Brocot tree and Farey sequences are built on.
    #[must_use]
    pub fn mediant(a: &Rational, b: &Rational) -> Self {
        Self::reduced(a.num.add(&b.num), a.den.add(&b.den))
    }
}

// ---------------------------------------------------------------------------
// trait impls
// ---------------------------------------------------------------------------

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // Denominators are positive, so cross-multiplication preserves the
        // direction of the comparison.
        self.num.mul(&other.den).cmp(&other.num.mul(&self.den))
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Default for Rational {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl std::ops::Add for Rational {
    type Output = Rational;
    fn add(self, rhs: Rational) -> Rational {
        Rational::add(&self, &rhs)
    }
}

impl std::ops::Sub for Rational {
    type Output = Rational;
    fn sub(self, rhs: Rational) -> Rational {
        Rational::sub(&self, &rhs)
    }
}

impl std::ops::Mul for Rational {
    type Output = Rational;
    fn mul(self, rhs: Rational) -> Rational {
        Rational::mul(&self, &rhs)
    }
}

impl std::ops::Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational::neg(&self)
    }
}

// ---------------------------------------------------------------------------
// Farey sequences, Stern-Brocot, and approximation
// ---------------------------------------------------------------------------

/// The Farey sequence F_n: every reduced fraction in `[0, 1]` with
/// denominator at most `n`, in ascending order.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn farey_sequence(n: u64) -> Vec<Rational> {
    assert!(n > 0, "Farey order must be positive");
    // Standard next-term recurrence: from consecutive p/q < r/s,
    // the following term has denominator (n + q) / s * s - q.
    let mut out = vec![Rational::from_i64(0, 1)];
    let (mut p, mut q, mut r, mut s) = (0i128, 1i128, 1i128, n as i128);
    out.push(Rational::from_i64(r as i64, s as i64));
    while r < s {
        let k = (n as i128 + q) / s;
        let (np, nq) = (r, s);
        let (nr, ns) = (k * r - p, k * s - q);
        p = np;
        q = nq;
        r = nr;
        s = ns;
        out.push(Rational::from_i64(r as i64, s as i64));
    }
    out
}

/// The path from the Stern-Brocot root `1/1` down to `r`, as a sequence of
/// branch choices: `true` for the right (larger) child, `false` for the
/// left.
///
/// The root itself has an empty path. Only positive rationals have one.
///
/// # Panics
/// Panics unless `r` is strictly positive.
#[must_use]
pub fn stern_brocot_path(r: &Rational) -> Vec<bool> {
    assert!(!r.is_negative() && !r.is_zero(), "Stern-Brocot covers positive rationals");
    // Descend by repeatedly taking mediants; the continued fraction gives
    // the run lengths directly, but walking is clearer and just as exact.
    let mut lo = (BigInt::zero(), BigInt::one()); // 0/1
    let mut hi = (BigInt::one(), BigInt::zero()); // 1/0, the right sentinel
    let mut path = Vec::new();
    loop {
        let med = Rational::reduced(lo.0.add(&hi.0), lo.1.add(&hi.1));
        match r.cmp(&med) {
            Ordering::Equal => return path,
            Ordering::Greater => {
                path.push(true);
                lo = (med.num, med.den);
            }
            Ordering::Less => {
                path.push(false);
                hi = (med.num, med.den);
            }
        }
    }
}

/// Every continued-fraction convergent of `x` with denominator at most
/// `max_den`, in increasing order of denominator.
///
/// The last element is the best rational approximation to `x` under that
/// bound, in the strong sense that no fraction with a smaller denominator
/// is closer.
///
/// # Panics
/// Panics if `max_den` is zero or `x` is not finite.
#[must_use]
pub fn best_rational_approximations(x: f64, max_den: u64) -> Vec<Rational> {
    assert!(max_den > 0, "max_den must be positive");
    assert!(x.is_finite(), "x must be finite");
    let exact = Rational::from_f64_exact(x).expect("finite input");
    let cf = exact.to_continued_fraction();
    let mut out = Vec::new();
    // Convergents by the standard recurrence h_k = a_k h_{k-1} + h_{k-2}.
    let (mut h_prev, mut h) = (BigInt::zero(), BigInt::one());
    let (mut k_prev, mut k) = (BigInt::one(), BigInt::zero());
    let bound = BigInt::from_u64(max_den);
    for a in &cf {
        let h_next = a.mul(&h).add(&h_prev);
        let k_next = a.mul(&k).add(&k_prev);
        if k_next.abs() > bound {
            // A semiconvergent may still fit under the bound: the largest
            // t with k_{k-1} + t*k_k <= max_den, t at least half of a_k.
            let room = bound.sub(&k_prev);
            if !room.is_negative() && !k.is_zero() {
                let t = room.div_rem(&k).0;
                let half = a.div_rem(&BigInt::from_u64(2)).0;
                if !t.is_zero() && t >= half {
                    let hs = t.mul(&h).add(&h_prev);
                    let ks = t.mul(&k).add(&k_prev);
                    if let Some(cand) = Rational::new(hs, ks) {
                        // Keep it only if it genuinely beats the last convergent.
                        let better = out.last().is_none_or(|prev: &Rational| {
                            cand.sub(&exact).abs() < prev.sub(&exact).abs()
                        });
                        if better {
                            out.push(cand);
                        }
                    }
                }
            }
            break;
        }
        h_prev = std::mem::replace(&mut h, h_next);
        k_prev = std::mem::replace(&mut k, k_next);
        if !k.is_zero() {
            out.push(Rational::reduced(h.clone(), k.clone()));
        }
    }
    if out.is_empty() {
        out.push(Rational::zero());
    }
    out
}

// ---------------------------------------------------------------------------
// exact linear algebra
// ---------------------------------------------------------------------------

/// Bareiss fraction-free elimination on an integer augmented matrix,
/// in place. Returns the sign contributed by row swaps, or `None` if the
/// system is singular.
///
/// Every division in the inner loop is exact: that is the defining
/// property of the Bareiss recurrence, and it keeps the entries as small
/// as the minors allow instead of letting fractions grow.
fn bareiss(m: &mut [Vec<BigInt>]) -> Option<i8> {
    let n = m.len();
    if n == 0 {
        return Some(1);
    }
    let width = m[0].len();
    let mut sign = 1i8;
    let mut prev = BigInt::one();
    for k in 0..n.min(width) {
        if m[k][k].is_zero() {
            let pivot = (k + 1..n).find(|&i| !m[i][k].is_zero())?;
            m.swap(k, pivot);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..width {
                let t = m[i][j].mul(&m[k][k]).sub(&m[i][k].mul(&m[k][j]));
                let (q, r) = t.div_rem(&prev);
                debug_assert!(r.is_zero(), "Bareiss division must be exact");
                m[i][j] = q;
            }
            m[i][k] = BigInt::zero();
        }
        prev = m[k][k].clone();
    }
    Some(sign)
}

/// Solve `A x = b` exactly for rational data, by clearing denominators and
/// running Bareiss fraction-free elimination.
///
/// Returns `None` if the matrix is not square, the shapes disagree, or the
/// system is singular.
#[must_use]
pub fn solve_exact_rational(a: &[Vec<Rational>], b: &[Rational]) -> Option<Vec<Rational>> {
    let n = a.len();
    if n == 0 || b.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    // Scale each equation by the lcm of its denominators. Scaling a single
    // equation leaves the solution set untouched.
    let mut m: Vec<Vec<BigInt>> = Vec::with_capacity(n);
    for (row, rhs) in a.iter().zip(b) {
        let mut l = BigInt::one();
        for e in row.iter().chain(std::iter::once(rhs)) {
            l = l.lcm(&e.den);
        }
        let scale = |e: &Rational| e.num.mul(&l.div_rem(&e.den).0);
        let mut integer_row: Vec<BigInt> = row.iter().map(scale).collect();
        integer_row.push(scale(rhs));
        m.push(integer_row);
    }
    bareiss(&mut m)?;
    if m[n - 1][n - 1].is_zero() {
        return None;
    }
    // Back-substitute over the rationals; the triangular system is exact.
    let mut x = vec![Rational::zero(); n];
    for i in (0..n).rev() {
        let mut acc = Rational::from_int(m[i][n].clone());
        for j in i + 1..n {
            acc = acc.sub(&Rational::from_int(m[i][j].clone()).mul(&x[j]));
        }
        x[i] = acc.div(&Rational::from_int(m[i][i].clone()))?;
    }
    Some(x)
}

/// Solve `A x = b` exactly for an `f64` system.
///
/// Each coefficient is converted to the rational it exactly equals — every
/// finite `f64` is dyadic — so the result is the exact solution of the
/// system as stored. Where the `f64` inputs are themselves roundings of
/// intended values, use [`solve_exact_rational`] to keep those values
/// exact instead.
///
/// Returns `None` for a non-square or singular system, mismatched shapes,
/// or any non-finite entry.
#[must_use]
pub fn solve_exact(a: &Matrix, b: &[f64]) -> Option<Vec<Rational>> {
    if a.rows != a.cols || b.len() != a.rows {
        return None;
    }
    let n = a.rows;
    let mut ar: Vec<Vec<Rational>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            row.push(Rational::from_f64_exact(a.get(i, j))?);
        }
        ar.push(row);
    }
    let br: Option<Vec<Rational>> = b.iter().map(|&v| Rational::from_f64_exact(v)).collect();
    solve_exact_rational(&ar, &br?)
}

/// The exact determinant of a rational matrix, via Bareiss on the
/// denominator-cleared integer matrix.
///
/// # Panics
/// Panics if the matrix is not square.
#[must_use]
pub fn determinant_exact(a: &[Vec<Rational>]) -> Rational {
    let n = a.len();
    if n == 0 {
        return Rational::one();
    }
    assert!(a.iter().all(|row| row.len() == n), "determinant needs a square matrix");
    // Pull one scale factor per row out of the determinant: scaling row i
    // by l_i multiplies the determinant by l_i.
    let mut scale_prod = Rational::one();
    let mut m: Vec<Vec<BigInt>> = Vec::with_capacity(n);
    for row in a {
        let mut l = BigInt::one();
        for e in row {
            l = l.lcm(&e.den);
        }
        m.push(row.iter().map(|e| e.num.mul(&l.div_rem(&e.den).0)).collect());
        scale_prod = scale_prod.mul(&Rational::from_int(l));
    }
    let Some(sign) = bareiss(&mut m) else {
        return Rational::zero(); // a zero pivot column means a singular matrix
    };
    let det_int = Rational::from_int(m[n - 1][n - 1].clone());
    let signed = if sign < 0 { det_int.neg() } else { det_int };
    signed.div(&scale_prod).expect("row scales are non-zero")
}

/// The exact inverse of the `n x n` Hilbert matrix `H_ij = 1/(i+j+1)`.
///
/// Uses the closed form
/// `(-1)^(i+j) (i+j+1) C(n+i, n-j-1) C(n+j, n-i-1) C(i+j, i)^2`,
/// whose entries are all integers.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn hilbert_matrix_inverse_exact(n: usize) -> Vec<Vec<Rational>> {
    assert!(n > 0, "Hilbert order must be positive");
    let nn = n as u64;
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let (iu, ju) = (i as u64, j as u64);
                    let mut v = BigInt::from_u64(iu + ju + 1)
                        .mul(&BigInt::binomial(nn + iu, nn - ju - 1))
                        .mul(&BigInt::binomial(nn + ju, nn - iu - 1))
                        .mul(&BigInt::binomial(iu + ju, iu).pow(2));
                    if (i + j) % 2 == 1 {
                        v = v.neg();
                    }
                    Rational::from_int(v)
                })
                .collect()
        })
        .collect()
}

/// The `n x n` Hilbert matrix as exact rationals.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn hilbert_matrix_exact(n: usize) -> Vec<Vec<Rational>> {
    assert!(n > 0, "Hilbert order must be positive");
    (0..n)
        .map(|i| (0..n).map(|j| Rational::from_i64(1, (i + j + 1) as i64)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn r(n: i64, d: i64) -> Rational {
        Rational::from_i64(n, d)
    }

    fn mat_mul(a: &[Vec<Rational>], b: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
        let n = a.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (0..n).fold(Rational::zero(), |acc, k| acc.add(&a[i][k].mul(&b[k][j])))
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_canonical_form_and_ordering() {
        // Always reduced, denominator positive, one representation per value.
        assert_eq!(r(2, 4), r(1, 2));
        assert_eq!(r(-2, -4), r(1, 2));
        assert_eq!(r(2, -4), r(-1, 2));
        assert_eq!(r(1, -2).den, BigInt::one().add(&BigInt::one()));
        assert!(!r(1, -2).den.is_negative());
        assert_eq!(r(0, 5), Rational::zero());
        assert_eq!(r(0, -5).den, BigInt::one(), "zero normalizes to 0/1");
        assert!(Rational::new(BigInt::one(), BigInt::zero()).is_none());
        assert!(r(4, 2).is_integer() && !r(1, 2).is_integer());

        // Ordering is numeric, not lexicographic on the fields: 1/3 < 1/2
        // even though 3 > 2, and negatives sort below zero.
        let mut v = [r(1, 2), r(1, 3), r(-1, 2), Rational::zero(), r(2, 1), r(-5, 3)];
        v.sort();
        let got: Vec<String> = v.iter().map(ToString::to_string).collect();
        assert_eq!(got, ["-5/3", "-1/2", "0", "1/3", "1/2", "2"]);
        assert!(r(1, 3) < r(1, 2));
        assert!(r(-1, 3) > r(-1, 2));
        // Comparison agrees with f64 on values both can hold.
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let a = r((rng.next_u64() % 41) as i64 - 20, 1 + (rng.next_u64() % 12) as i64);
            let b = r((rng.next_u64() % 41) as i64 - 20, 1 + (rng.next_u64() % 12) as i64);
            assert_eq!(a.cmp(&b), a.to_f64().partial_cmp(&b.to_f64()).unwrap(), "{a} vs {b}");
        }
    }

    #[test]
    fn test_arithmetic_is_exact_over_many_operations() {
        // The roadmap's property: no drift over 1e4 operations. Denominators
        // are drawn from a fixed small set so the exact result stays cheap,
        // and the same terms are added then removed in a different order.
        let terms: Vec<Rational> = (0..50)
            .map(|k| r(1 + (k % 7), [2i64, 3, 5, 7, 11][(k % 5) as usize]))
            .collect();
        let mut acc = Rational::zero();
        let mut ops = 0usize;
        for _ in 0..100 {
            for t in &terms {
                acc = acc.add(t);
                ops += 1;
            }
        }
        for _ in 0..100 {
            for t in terms.iter().rev() {
                acc = acc.sub(t);
                ops += 1;
            }
        }
        assert!(ops >= 10_000, "expected at least 1e4 operations, ran {ops}");
        assert_eq!(acc, Rational::zero(), "exact arithmetic drifted after {ops} ops");

        // The same schedule in f64 does drift, which is the point.
        let mut f = 0.0f64;
        for _ in 0..100 {
            for t in &terms {
                f += t.to_f64();
            }
        }
        for _ in 0..100 {
            for t in terms.iter().rev() {
                f -= t.to_f64();
            }
        }
        assert!(f != 0.0, "the f64 control was expected to accumulate error");

        // Field laws on random values.
        let mut rng = Rng::new(17);
        for _ in 0..200 {
            let pick = |rng: &mut Rng| {
                r((rng.next_u64() % 41) as i64 - 20, 1 + (rng.next_u64() % 20) as i64)
            };
            let (a, b, c) = (pick(&mut rng), pick(&mut rng), pick(&mut rng));
            assert_eq!(a.add(&b), b.add(&a));
            assert_eq!(a.mul(&b), b.mul(&a));
            assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
            assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)), "distributive");
            assert_eq!(a.add(&b).sub(&b), a);
            if !b.is_zero() {
                assert_eq!(a.div(&b).unwrap().mul(&b), a, "div then mul must round-trip");
                assert_eq!(b.recip().unwrap().mul(&b), Rational::one());
            }
            assert_eq!(a.neg().neg(), a);
            assert_eq!(a.pow(3), a.mul(&a).mul(&a));
            assert_eq!(a.pow(0), Rational::one());
            if !a.is_zero() {
                assert_eq!(a.pow(-2), a.mul(&a).recip().unwrap());
            }
        }
        assert!(r(3, 4).div(&Rational::zero()).is_none());
        assert!(Rational::zero().recip().is_none());
    }

    #[test]
    fn test_f64_conversion_and_rounding() {
        // The roadmap's property: from_f64_exact(0.1) has a power-of-two
        // denominator, because every finite f64 is dyadic.
        let tenth = Rational::from_f64_exact(0.1).unwrap();
        assert!(tenth.den.bits() > 1);
        assert_eq!(tenth.den, BigInt::one().shl(tenth.den.bits() - 1), "denominator is 2^k");
        assert_ne!(tenth, r(1, 10), "0.1 is not exactly one tenth");
        assert_eq!(tenth.to_f64(), 0.1);
        // The exact value of the double nearest 0.1.
        assert_eq!(tenth.num.to_string_radix(10), "3602879701896397");
        assert_eq!(tenth.den.to_string_radix(10), "36028797018963968");

        // Round-trip for a spread of magnitudes, including subnormals.
        for x in [0.0, 1.0, -1.0, 0.5, 1e300, -1e-300, f64::MIN_POSITIVE, 5e-324, 12345.678] {
            let q = Rational::from_f64_exact(x).expect("finite");
            assert_eq!(q.to_f64(), x, "round-trip of {x}");
        }
        assert!(Rational::from_f64_exact(f64::NAN).is_none());
        assert!(Rational::from_f64_exact(f64::INFINITY).is_none());
        // Values beyond f64 range still convert back sensibly.
        let huge = Rational::new(BigInt::one().shl(4000), BigInt::one().shl(3000)).unwrap();
        assert!((huge.to_f64() - 2f64.powi(1000)).abs() / 2f64.powi(1000) < 1e-12);

        // floor/ceil/round/fract, including the negative cases where
        // truncation and flooring differ.
        assert_eq!(r(7, 2).floor().to_i64(), Some(3));
        assert_eq!(r(-7, 2).floor().to_i64(), Some(-4));
        assert_eq!(r(7, 2).ceil().to_i64(), Some(4));
        assert_eq!(r(-7, 2).ceil().to_i64(), Some(-3));
        assert_eq!(r(4, 2).floor().to_i64(), Some(2), "exact integers do not shift");
        assert_eq!(r(4, 2).ceil().to_i64(), Some(2));
        assert_eq!(r(7, 2).round().to_i64(), Some(4));
        assert_eq!(r(-7, 2).round().to_i64(), Some(-4), "halves round away from zero");
        assert_eq!(r(5, 3).round().to_i64(), Some(2));
        assert_eq!(r(-5, 3).round().to_i64(), Some(-2));
        let mut rng = Rng::new(23);
        for _ in 0..200 {
            let a = r((rng.next_u64() % 61) as i64 - 30, 1 + (rng.next_u64() % 9) as i64);
            let f = a.fract();
            assert!(!f.is_negative() && f < Rational::one(), "fract out of [0,1): {f}");
            assert_eq!(Rational::from_int(a.floor()).add(&f), a, "floor + fract must rebuild");
            assert!(Rational::from_int(a.floor()) <= a && a < Rational::from_int(a.floor().add(&BigInt::one())));
        }
    }

    #[test]
    fn test_continued_fractions_and_approximation() {
        // Round-trip, and the canonical form never ends in 1.
        let mut rng = Rng::new(29);
        for _ in 0..200 {
            let a = r((rng.next_u64() % 400) as i64 - 200, 1 + (rng.next_u64() % 200) as i64);
            let cf = a.to_continued_fraction();
            assert!(!cf.is_empty());
            assert!(cf[1..].iter().all(|t| !t.is_negative() && !t.is_zero()),
                    "tail terms must be positive for {a}: {cf:?}");
            if cf.len() > 1 {
                assert_ne!(cf[cf.len() - 1], BigInt::one(), "canonical form must not end in 1");
            }
            assert_eq!(Rational::from_continued_fraction(&cf).unwrap(), a, "round-trip of {a}");
        }
        // A worked example: 415/93 = [4; 2, 6, 7].
        let cf = r(415, 93).to_continued_fraction();
        let got: Vec<i64> = cf.iter().map(|t| t.to_i64().unwrap()).collect();
        assert_eq!(got, [4, 2, 6, 7]);
        assert_eq!(r(5, 1).to_continued_fraction().len(), 1);
        assert!(Rational::from_continued_fraction(&[]).is_err());
        assert!(Rational::from_continued_fraction(&[BigInt::one(), BigInt::zero()]).is_err());

        // Convergents of pi: the classic 3, 22/7, 333/106, 355/113.
        let approx = best_rational_approximations(std::f64::consts::PI, 1000);
        let shown: Vec<String> = approx.iter().map(ToString::to_string).collect();
        assert!(shown.contains(&"22/7".to_string()), "missing 22/7: {shown:?}");
        assert!(shown.contains(&"333/106".to_string()), "missing 333/106: {shown:?}");
        assert_eq!(approx.last().unwrap().to_string(), "355/113");
        // Each convergent is strictly better than the one before it.
        let target = Rational::from_f64_exact(std::f64::consts::PI).unwrap();
        for w in approx.windows(2) {
            assert!(w[1].sub(&target).abs() < w[0].sub(&target).abs(),
                    "convergents must improve: {} then {}", w[0], w[1]);
        }
        // No fraction with a smaller denominator beats the best one.
        let best = approx.last().unwrap().clone();
        let err = best.sub(&target).abs();
        let limit = best.den.to_i64().unwrap();
        for q in 1..limit {
            let p = (std::f64::consts::PI * q as f64).round() as i64;
            let cand = r(p, q);
            assert!(cand.sub(&target).abs() >= err, "{cand} beats the reported best {best}");
        }
        assert_eq!(Rational::from_f64_approx(0.5, 10), r(1, 2));
        assert_eq!(Rational::from_f64_approx(0.0, 10), Rational::zero());

        // Stern-Brocot: the path rebuilds the value, and the root is empty.
        assert!(stern_brocot_path(&Rational::one()).is_empty());
        for _ in 0..60 {
            let a = r(1 + (rng.next_u64() % 60) as i64, 1 + (rng.next_u64() % 60) as i64);
            let path = stern_brocot_path(&a);
            let (mut lo, mut hi) = ((BigInt::zero(), BigInt::one()), (BigInt::one(), BigInt::zero()));
            let mut cur = Rational::one();
            for &right in &path {
                let med = Rational::new(lo.0.add(&hi.0), lo.1.add(&hi.1)).unwrap();
                if right {
                    lo = (med.num.clone(), med.den.clone());
                } else {
                    hi = (med.num.clone(), med.den.clone());
                }
                cur = Rational::new(lo.0.add(&hi.0), lo.1.add(&hi.1)).unwrap();
            }
            assert_eq!(cur, a, "Stern-Brocot path does not rebuild {a}");
        }

        // Mediant lies strictly between its parents.
        for _ in 0..60 {
            let a = r((rng.next_u64() % 20) as i64, 1 + (rng.next_u64() % 20) as i64);
            let b = r((rng.next_u64() % 20) as i64, 1 + (rng.next_u64() % 20) as i64);
            if a == b {
                continue;
            }
            let (lo, hi) = if a < b { (&a, &b) } else { (&b, &a) };
            let m = Rational::mediant(lo, hi);
            assert!(*lo < m && m < *hi, "mediant of {lo} and {hi} was {m}");
        }
    }

    #[test]
    fn test_farey_sequence() {
        let f5 = farey_sequence(5);
        let shown: Vec<String> = f5.iter().map(ToString::to_string).collect();
        assert_eq!(shown, ["0", "1/5", "1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "1"]);
        for n in 1..=12u64 {
            let f = farey_sequence(n);
            // Ascending, in [0,1], reduced, and bounded denominators.
            assert!(f.windows(2).all(|w| w[0] < w[1]), "F_{n} is not ascending");
            assert_eq!(*f.first().unwrap(), Rational::zero());
            assert_eq!(*f.last().unwrap(), Rational::one());
            assert!(f.iter().all(|q| q.den.to_i64().unwrap() as u64 <= n));
            // The defining unimodular property: consecutive p/q, r/s have
            // q*r - p*s == 1.
            for w in f.windows(2) {
                let d = w[0].den.mul(&w[1].num).sub(&w[0].num.mul(&w[1].den));
                assert_eq!(d, BigInt::one(), "F_{n} neighbours are not unimodular");
                // Which forces each to be the mediant of its neighbours.
            }
            // Length is 1 + sum of Euler's totient up to n.
            let totient = |mut m: u64| {
                let mut result = m;
                let mut p = 2;
                while p * p <= m {
                    if m.is_multiple_of(p) {
                        while m.is_multiple_of(p) {
                            m /= p;
                        }
                        result -= result / p;
                    }
                    p += 1;
                }
                if m > 1 {
                    result -= result / m;
                }
                result
            };
            let expect = 1 + (1..=n).map(totient).sum::<u64>() as usize;
            assert_eq!(f.len(), expect, "|F_{n}| should be 1 + sum of totients");
        }
    }

    #[test]
    fn test_exact_linear_algebra_on_hilbert() {
        // The Hilbert inverse closed form is validated against the matrix
        // itself: H * H^-1 must be exactly the identity, with no tolerance.
        for n in 1..=8usize {
            let h = hilbert_matrix_exact(n);
            let hi = hilbert_matrix_inverse_exact(n);
            let prod = mat_mul(&h, &hi);
            for (i, row) in prod.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    let want = if i == j { Rational::one() } else { Rational::zero() };
                    assert_eq!(*v, want, "H*H^-1 at ({i},{j}) for n={n}");
                }
            }
            // Inverse entries are integers for the Hilbert matrix.
            assert!(hi.iter().flatten().all(Rational::is_integer));
            // det(H) * det(H^-1) == 1 exactly.
            let d = determinant_exact(&h);
            let di = determinant_exact(&hi);
            assert_eq!(d.mul(&di), Rational::one(), "determinants must be reciprocal at n={n}");
            assert!(!d.is_zero());
        }
        // Known values: det H_2 = 1/12, det H_3 = 1/2160.
        assert_eq!(determinant_exact(&hilbert_matrix_exact(2)), r(1, 12));
        assert_eq!(determinant_exact(&hilbert_matrix_exact(3)), r(1, 2160));
        assert_eq!(hilbert_matrix_inverse_exact(2)[0][0], r(4, 1));
        assert_eq!(hilbert_matrix_inverse_exact(2)[0][1], r(-6, 1));

        // The roadmap's property: solving the exact 8x8 Hilbert system
        // against each unit vector reproduces the known inverse columns.
        let n = 8;
        let h = hilbert_matrix_exact(n);
        let hi = hilbert_matrix_inverse_exact(n);
        for j in 0..n {
            let e: Vec<Rational> = (0..n)
                .map(|i| if i == j { Rational::one() } else { Rational::zero() })
                .collect();
            let x = solve_exact_rational(&h, &e).expect("Hilbert is invertible");
            for i in 0..n {
                assert_eq!(x[i], hi[i][j], "solve column {j} row {i}");
            }
        }

        // determinant_exact against cofactor expansion on small matrices,
        // and the multiplicative law det(AB) = det(A)det(B).
        let mut rng = Rng::new(41);
        let pick = |rng: &mut Rng| r((rng.next_u64() % 13) as i64 - 6, 1 + (rng.next_u64() % 5) as i64);
        for _ in 0..30 {
            let a: Vec<Vec<Rational>> = (0..3).map(|_| (0..3).map(|_| pick(&mut rng)).collect()).collect();
            let cof = a[0][0].mul(&a[1][1].mul(&a[2][2]).sub(&a[1][2].mul(&a[2][1])))
                .sub(&a[0][1].mul(&a[1][0].mul(&a[2][2]).sub(&a[1][2].mul(&a[2][0]))))
                .add(&a[0][2].mul(&a[1][0].mul(&a[2][1]).sub(&a[1][1].mul(&a[2][0]))));
            assert_eq!(determinant_exact(&a), cof, "3x3 determinant vs cofactor expansion");
            let b: Vec<Vec<Rational>> = (0..3).map(|_| (0..3).map(|_| pick(&mut rng)).collect()).collect();
            assert_eq!(
                determinant_exact(&mat_mul(&a, &b)),
                determinant_exact(&a).mul(&determinant_exact(&b)),
                "det(AB) = det(A)det(B)"
            );
        }
        // A singular matrix has zero determinant and no solution.
        let sing = vec![vec![r(1, 1), r(2, 1)], vec![r(2, 1), r(4, 1)]];
        assert_eq!(determinant_exact(&sing), Rational::zero());
        assert!(solve_exact_rational(&sing, &[r(1, 1), r(1, 1)]).is_none());
        assert_eq!(determinant_exact(&[]), Rational::one(), "empty product convention");

        // solve_exact over f64 input: exact for a dyadic system, and the
        // residual is exactly zero.
        let a = Matrix::from_fn(3, 3, |i, j| [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]][i][j]);
        let b = [8.0, -11.0, -3.0];
        let x = solve_exact(&a, &b).expect("non-singular");
        assert_eq!(x, vec![r(2, 1), r(3, 1), r(-1, 1)]);
        for i in 0..3 {
            let lhs = (0..3).fold(Rational::zero(), |acc, j| {
                acc.add(&Rational::from_f64_exact(a.get(i, j)).unwrap().mul(&x[j]))
            });
            assert_eq!(lhs, Rational::from_f64_exact(b[i]).unwrap(), "residual at row {i}");
        }
        assert!(solve_exact(&Matrix::zeros(2, 2), &[1.0, 1.0]).is_none());
        assert!(solve_exact(&Matrix::identity(2), &[1.0]).is_none(), "shape mismatch");
    }
}
