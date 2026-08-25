//! Dense univariate polynomials with `f64` coefficients ([`Poly`]) and with
//! exact rational coefficients ([`PolyQ`]).
//!
//! Both types store coefficients from lowest to highest degree, so `c[i]`
//! multiplies `x^i`, and both keep that vector trimmed: the last entry of a
//! non-empty coefficient vector is never zero. The zero polynomial is the
//! empty vector, and [`Poly::degree`] reports `0` for it (use
//! [`Poly::is_zero`] to tell the zero polynomial from a non-zero constant).
//!
//! [`Poly`] carries the numerical machinery -- root finding, Sturm
//! sequences, Chebyshev fitting, Pade approximants -- while [`PolyQ`] carries
//! the exact machinery: subresultant GCDs, content and primitive parts,
//! rational root factoring, and Eisenstein's criterion.

use crate::error::SolveError;
use crate::exact::bigint::BigInt;
use crate::exact::rational::Rational;
use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::numerical::polynomial_roots;
use crate::transforms::fft::{fft, ifft};

/// A polynomial with `f64` coefficients, ordered from the constant term up.
#[derive(Debug, Clone, PartialEq)]
pub struct Poly {
    /// Coefficients low to high: `c[i]` multiplies `x^i`. Trimmed, so the
    /// last entry is non-zero unless the vector is empty.
    pub c: Vec<f64>,
}

/// Drop trailing zero coefficients so the representation is canonical.
fn trim_f64(mut c: Vec<f64>) -> Vec<f64> {
    while c.last().is_some_and(|&x| x == 0.0) {
        c.pop();
    }
    c
}

impl Poly {
    /// The polynomial with the given coefficients, low degree first.
    ///
    /// Trailing zeros are dropped, so `new(vec![1.0, 0.0])` and
    /// `new(vec![1.0])` are equal.
    #[must_use]
    pub fn new(c: Vec<f64>) -> Self {
        Poly { c: trim_f64(c) }
    }

    /// The zero polynomial.
    #[must_use]
    pub fn zero() -> Self {
        Poly { c: Vec::new() }
    }

    /// The constant polynomial `a`.
    #[must_use]
    pub fn constant(a: f64) -> Self {
        Poly::new(vec![a])
    }

    /// The monomial `coeff * x^k`.
    #[must_use]
    pub fn monomial(k: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; k + 1];
        c[k] = coeff;
        Poly::new(c)
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }

    /// The degree, with the convention that the zero polynomial has degree
    /// `0` (pair with [`Poly::is_zero`] when that distinction matters).
    #[must_use]
    pub fn degree(&self) -> usize {
        self.c.len().saturating_sub(1)
    }

    /// The leading coefficient, or `0.0` for the zero polynomial.
    #[must_use]
    pub fn leading(&self) -> f64 {
        self.c.last().copied().unwrap_or(0.0)
    }

    /// Value at `x` by Horner's rule.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        self.c.iter().rev().fold(0.0_f64, |acc, &a| acc * x + a)
    }

    /// Value at a complex point by Horner's rule.
    #[must_use]
    pub fn eval_complex(&self, z: Complex) -> Complex {
        self.c
            .iter()
            .rev()
            .fold(Complex::new(0.0, 0.0), |acc, &a| acc * z + Complex::new(a, 0.0))
    }

    /// Sum of two polynomials.
    #[must_use]
    pub fn add(&self, other: &Poly) -> Self {
        let n = self.c.len().max(other.c.len());
        let mut c = vec![0.0; n];
        for (i, s) in self.c.iter().enumerate() {
            c[i] += s;
        }
        for (i, s) in other.c.iter().enumerate() {
            c[i] += s;
        }
        Poly::new(c)
    }

    /// Difference `self - other`.
    #[must_use]
    pub fn sub(&self, other: &Poly) -> Self {
        let n = self.c.len().max(other.c.len());
        let mut c = vec![0.0; n];
        for (i, s) in self.c.iter().enumerate() {
            c[i] += s;
        }
        for (i, s) in other.c.iter().enumerate() {
            c[i] -= s;
        }
        Poly::new(c)
    }

    /// Additive inverse.
    #[must_use]
    pub fn neg(&self) -> Self {
        Poly::new(self.c.iter().map(|&a| -a).collect())
    }

    /// Schoolbook product. See [`polynomial_multiply_fft`] for the
    /// `O(n log n)` alternative.
    #[must_use]
    pub fn mul(&self, other: &Poly) -> Self {
        if self.is_zero() || other.is_zero() {
            return Poly::zero();
        }
        let mut c = vec![0.0; self.c.len() + other.c.len() - 1];
        for (i, &a) in self.c.iter().enumerate() {
            for (j, &b) in other.c.iter().enumerate() {
                c[i + j] += a * b;
            }
        }
        Poly::new(c)
    }

    /// Every coefficient multiplied by `k`.
    #[must_use]
    pub fn mul_scalar(&self, k: f64) -> Self {
        Poly::new(self.c.iter().map(|&a| a * k).collect())
    }

    /// Quotient and remainder of `self / divisor`, satisfying
    /// `self == q * divisor + r` with `r` of lower degree than `divisor`.
    ///
    /// Returns `None` when `divisor` is the zero polynomial.
    #[must_use]
    pub fn div_rem(&self, divisor: &Poly) -> Option<(Self, Self)> {
        if divisor.is_zero() {
            return None;
        }
        let dd = divisor.degree();
        if self.is_zero() || self.c.len() < divisor.c.len() {
            return Some((Poly::zero(), self.clone()));
        }
        let dl = divisor.leading();
        let mut r = self.c.clone();
        let mut q = vec![0.0; self.c.len() - divisor.c.len() + 1];
        for i in (0..q.len()).rev() {
            let coef = r[i + dd] / dl;
            q[i] = coef;
            for j in 0..dd {
                r[i + j] -= coef * divisor.c[j];
            }
            // Exact by construction: forcing it avoids leaving rounding
            // dust in the slot that must cancel, which would otherwise
            // inflate the remainder's apparent degree.
            r[i + dd] = 0.0;
        }
        Some((Poly::new(q), Poly::new(r)))
    }

    /// Derivative `p'(x)`.
    #[must_use]
    pub fn derivative(&self) -> Self {
        if self.c.len() < 2 {
            return Poly::zero();
        }
        Poly::new(self.c.iter().enumerate().skip(1).map(|(i, &a)| a * i as f64).collect())
    }

    /// Antiderivative with constant term `c0`.
    #[must_use]
    pub fn integral(&self, c0: f64) -> Self {
        let mut c = Vec::with_capacity(self.c.len() + 1);
        c.push(c0);
        for (i, &a) in self.c.iter().enumerate() {
            c.push(a / (i as f64 + 1.0));
        }
        Poly::new(c)
    }

    /// Composition `self(inner(x))`, by Horner's rule in the inner
    /// polynomial.
    #[must_use]
    pub fn compose(&self, inner: &Poly) -> Self {
        let mut acc = Poly::zero();
        for &a in self.c.iter().rev() {
            acc = acc.mul(inner).add(&Poly::constant(a));
        }
        acc
    }

    /// The polynomial `p(k*x)`.
    #[must_use]
    pub fn scale_arg(&self, k: f64) -> Self {
        let mut p = 1.0;
        let mut c = self.c.clone();
        for (i, a) in c.iter_mut().enumerate() {
            if i > 0 {
                p *= k;
            }
            *a *= p;
        }
        Poly::new(c)
    }

    /// The polynomial `p(x + h)` (a Taylor shift, by repeated synthetic
    /// division).
    #[must_use]
    pub fn shift_arg(&self, h: f64) -> Self {
        let n = self.c.len();
        if n < 2 {
            return self.clone();
        }
        let mut b = self.c.clone();
        for i in 0..n {
            for j in (i..n - 1).rev() {
                b[j] += h * b[j + 1];
            }
        }
        Poly::new(b)
    }
}

/// Largest absolute coefficient, or `0.0` for the zero polynomial.
fn max_abs(p: &Poly) -> f64 {
    p.c.iter().fold(0.0_f64, |m, &a| m.max(a.abs()))
}

/// Zero out coefficients that are pure rounding dust next to `scale`, then
/// re-trim. Used by the tolerant `f64` GCD and by the Sturm chain, where a
/// remainder that is mathematically zero comes out as `~1e-16` noise.
fn chop(p: &Poly, tol: f64, scale: f64) -> Poly {
    let cut = tol * scale;
    Poly::new(p.c.iter().map(|&a| if a.abs() <= cut { 0.0 } else { a }).collect())
}

impl Poly {
    /// Monic greatest common divisor, computed by the Euclidean algorithm
    /// with a relative tolerance: a remainder whose coefficients are all
    /// below `tol` times the scale of the inputs is treated as zero.
    ///
    /// Returns the zero polynomial when both inputs are zero. `tol` around
    /// `1e-9` suits well-scaled polynomials of modest degree; exact input
    /// deserves [`PolyQ::gcd_exact`] instead.
    #[must_use]
    pub fn gcd(&self, other: &Poly, tol: f64) -> Self {
        let scale = max_abs(self).max(max_abs(other)).max(1.0);
        let mut a = chop(self, tol, scale);
        let mut b = chop(other, tol, scale);
        if a.is_zero() && b.is_zero() {
            return Poly::zero();
        }
        while !b.is_zero() {
            // Rescale to unit size each step so the tolerance keeps meaning
            // the same thing as the Euclidean remainders shrink or blow up.
            b = b.mul_scalar(1.0 / max_abs(&b));
            let (_, r) = a.div_rem(&b).expect("b is non-zero");
            a = b;
            b = chop(&r, tol, 1.0);
        }
        a.mul_scalar(1.0 / a.leading())
    }

    /// All complex roots, via Durand-Kerner
    /// ([`crate::numerical::polynomial_roots`]).
    ///
    /// # Errors
    /// Returns [`SolveError::InvalidArgument`] for a constant or zero
    /// polynomial and [`SolveError::NoConvergence`] if the iteration stalls.
    pub fn roots(&self) -> Result<Vec<Complex>, SolveError> {
        let high_first: Vec<f64> = self.c.iter().rev().copied().collect();
        polynomial_roots(&high_first)
    }

    /// The monic polynomial with exactly the given real roots,
    /// `prod (x - r_i)`.
    #[must_use]
    pub fn from_roots(roots: &[f64]) -> Self {
        let mut p = Poly::constant(1.0);
        for &r in roots {
            p = p.mul(&Poly::new(vec![-r, 1.0]));
        }
        p
    }

    /// The resultant `Res(self, other)`, as the determinant of the
    /// Sylvester matrix.
    ///
    /// It vanishes exactly when the two polynomials share a root (or when
    /// either is zero), and equals
    /// `lc(p)^deg(q) * prod_{p(a)=0} q(a)` otherwise.
    #[must_use]
    pub fn resultant(&self, other: &Poly) -> f64 {
        if self.is_zero() || other.is_zero() {
            return 0.0;
        }
        let m = self.degree();
        let n = other.degree();
        if m == 0 && n == 0 {
            return 1.0;
        }
        if m == 0 {
            return self.c[0].powi(n as i32);
        }
        if n == 0 {
            return other.c[0].powi(m as i32);
        }
        let size = m + n;
        let mut a = vec![vec![0.0; size]; size];
        for r in 0..n {
            for (k, &v) in self.c.iter().rev().enumerate() {
                a[r][r + k] = v;
            }
        }
        for r in 0..m {
            for (k, &v) in other.c.iter().rev().enumerate() {
                a[n + r][r + k] = v;
            }
        }
        determinant(&mut a)
    }

    /// The discriminant `(-1)^(n(n-1)/2) * Res(p, p') / lc(p)`.
    ///
    /// For `a x^2 + b x + c` this is `b^2 - 4ac`. It vanishes exactly when
    /// the polynomial has a repeated root. Degenerate inputs return `0.0`
    /// for the zero polynomial and non-zero constants (no roots to
    /// collide) and `1.0` for a linear polynomial, the usual conventions.
    #[must_use]
    pub fn discriminant(&self) -> f64 {
        let n = self.degree();
        if self.is_zero() || n == 0 {
            return 0.0;
        }
        if n == 1 {
            return 1.0;
        }
        let sign = if (n * (n - 1) / 2).is_multiple_of(2) { 1.0 } else { -1.0 };
        sign * self.resultant(&self.derivative()) / self.leading()
    }

    /// The canonical Sturm chain `p, p', -rem(p, p'), ...`, each member
    /// normalized to unit maximum coefficient so that long chains neither
    /// overflow nor underflow. Normalizing by a positive scalar leaves
    /// every sign, and therefore every root count, unchanged.
    ///
    /// The chain is empty for the zero polynomial.
    #[must_use]
    pub fn sturm_sequence(&self) -> Vec<Poly> {
        if self.is_zero() {
            return Vec::new();
        }
        let unit = |p: &Poly| {
            let m = max_abs(p);
            if m == 0.0 {
                Poly::zero()
            } else {
                p.mul_scalar(1.0 / m)
            }
        };
        let mut seq = vec![unit(self)];
        let d = self.derivative();
        if d.is_zero() {
            return seq;
        }
        seq.push(unit(&d));
        loop {
            let k = seq.len();
            let (_, r) = seq[k - 2].div_rem(&seq[k - 1]).expect("chain members are non-zero");
            let r = chop(&r, STURM_CHOP, 1.0);
            if r.is_zero() {
                return seq;
            }
            seq.push(unit(&r.neg()));
        }
    }

    /// Number of distinct real roots in the half-open interval `(a, b]`,
    /// by Sturm's theorem.
    ///
    /// Multiple roots are counted once. Returns `0` when `a >= b` or for
    /// the zero polynomial.
    ///
    /// # Panics
    /// Panics if `a` or `b` is not finite.
    #[must_use]
    pub fn count_real_roots(&self, a: f64, b: f64) -> usize {
        assert!(a.is_finite() && b.is_finite(), "count_real_roots needs finite bounds");
        if a >= b || self.is_zero() {
            return 0;
        }
        let seq = self.sturm_sequence();
        let va = sign_variations(&seq, a);
        let vb = sign_variations(&seq, b);
        va.saturating_sub(vb)
    }

    /// A Cauchy bound: every real root lies in `[-r, r]`.
    #[must_use]
    pub fn root_bound(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let lc = self.leading();
        1.0 + self.c.iter().take(self.c.len() - 1).fold(0.0_f64, |m, &a| m.max((a / lc).abs()))
    }

    /// Disjoint intervals, one per distinct real root, found by bisecting
    /// the Cauchy bound and counting with Sturm's theorem.
    ///
    /// Each returned `(a, b)` is a half-open interval `(a, b]` holding
    /// exactly one distinct real root; feed one to [`Poly::refine_root`].
    /// The intervals come out in increasing order.
    #[must_use]
    pub fn isolate_real_roots(&self) -> Vec<(f64, f64)> {
        if self.is_zero() || self.degree() == 0 {
            return Vec::new();
        }
        let seq = self.sturm_sequence();
        // Widen the bound slightly so a root sitting exactly on it is
        // strictly inside the half-open interval being counted.
        let bound = self.root_bound() * 1.0625;
        let count = |x: f64| sign_variations(&seq, x);
        let mut out = Vec::new();
        // Explicit stack of (lo, hi, roots in (lo, hi]).
        let total = count(-bound).saturating_sub(count(bound));
        let mut stack = vec![(-bound, bound, total)];
        while let Some((lo, hi, n)) = stack.pop() {
            if n == 0 {
                continue;
            }
            if n == 1 || hi - lo <= ISOLATE_MIN_WIDTH {
                out.push((lo, hi));
                continue;
            }
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                out.push((lo, hi));
                continue;
            }
            let vm = count(mid);
            let left = count(lo).saturating_sub(vm);
            let right = vm.saturating_sub(count(hi));
            // Push right first so the stack pops left-to-right.
            stack.push((mid, hi, right));
            stack.push((lo, mid, left));
        }
        out.sort_by(|x, y| x.0.partial_cmp(&y.0).expect("finite bounds"));
        out
    }

    /// Refine a root inside `interval` to an absolute width below `tol`.
    ///
    /// Bisection is used when the endpoints bracket a sign change (always
    /// true for a root of odd multiplicity); otherwise the midpoint is
    /// polished with Newton's method, which is what an even-multiplicity
    /// root needs. Returns the best estimate found.
    ///
    /// # Panics
    /// Panics if `tol` is not positive or the interval is not finite.
    #[must_use]
    pub fn refine_root(&self, interval: (f64, f64), tol: f64) -> f64 {
        let (mut lo, mut hi) = interval;
        assert!(tol > 0.0, "refine_root needs a positive tolerance");
        assert!(lo.is_finite() && hi.is_finite(), "refine_root needs a finite interval");
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        let flo = self.eval(lo);
        let fhi = self.eval(hi);
        if flo == 0.0 {
            return lo;
        }
        if fhi == 0.0 {
            return hi;
        }
        if flo * fhi < 0.0 {
            let mut flo = flo;
            while hi - lo > tol {
                let mid = 0.5 * (lo + hi);
                if mid <= lo || mid >= hi {
                    break;
                }
                let fm = self.eval(mid);
                if fm == 0.0 {
                    return mid;
                }
                if flo * fm < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            return 0.5 * (lo + hi);
        }
        let d = self.derivative();
        let mut x = 0.5 * (lo + hi);
        for _ in 0..NEWTON_STEPS {
            let fx = self.eval(x);
            let dx = d.eval(x);
            if dx == 0.0 {
                break;
            }
            let step = fx / dx;
            x -= step;
            if !x.is_finite() || x < lo || x > hi {
                return 0.5 * (lo + hi);
            }
            if step.abs() <= tol {
                break;
            }
        }
        x
    }
}

/// Coefficients below this fraction of the (unit-normalized) chain scale are
/// rounding dust, not signal, when building a Sturm sequence.
const STURM_CHOP: f64 = 1e-12;
/// Root isolation stops subdividing once an interval is this narrow.
const ISOLATE_MIN_WIDTH: f64 = 1e-12;
/// Newton polish steps for an even-multiplicity root.
const NEWTON_STEPS: usize = 200;

/// Sign variations of a Sturm chain evaluated at `x`, skipping zeros.
fn sign_variations(seq: &[Poly], x: f64) -> usize {
    let mut count = 0;
    let mut prev = 0.0_f64;
    for p in seq {
        let v = p.eval(x);
        if v == 0.0 {
            continue;
        }
        if prev != 0.0 && (v > 0.0) != (prev > 0.0) {
            count += 1;
        }
        prev = v;
    }
    count
}

/// Determinant by Gaussian elimination with partial pivoting; `a` is
/// consumed as scratch space.
fn determinant(a: &mut [Vec<f64>]) -> f64 {
    let n = a.len();
    let mut det = 1.0;
    for k in 0..n {
        let mut piv = k;
        for i in k + 1..n {
            if a[i][k].abs() > a[piv][k].abs() {
                piv = i;
            }
        }
        if a[piv][k] == 0.0 {
            return 0.0;
        }
        if piv != k {
            a.swap(piv, k);
            det = -det;
        }
        det *= a[k][k];
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            if f == 0.0 {
                continue;
            }
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
        }
    }
    det
}

/// Solve a dense square system in place by Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is (numerically) singular.
fn solve_linear(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = a.len();
    for k in 0..n {
        let mut piv = k;
        for i in k + 1..n {
            if a[i][k].abs() > a[piv][k].abs() {
                piv = i;
            }
        }
        if a[piv][k].abs() <= LINEAR_PIVOT_EPS {
            return None;
        }
        a.swap(piv, k);
        b.swap(piv, k);
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            if f == 0.0 {
                continue;
            }
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// A pivot at or below this magnitude means the Pade (or interpolation)
/// system is singular for practical purposes.
const LINEAR_PIVOT_EPS: f64 = 1e-300;

impl Poly {
    /// The unique interpolating polynomial through `(xs[i], ys[i])`, in the
    /// Lagrange form `sum_i y_i prod_{j != i} (x - x_j)/(x_i - x_j)`.
    ///
    /// # Panics
    /// Panics if the two slices differ in length or if any two nodes
    /// coincide.
    #[must_use]
    pub fn interpolate_lagrange(xs: &[f64], ys: &[f64]) -> Self {
        assert_eq!(xs.len(), ys.len(), "interpolate_lagrange needs matching lengths");
        let n = xs.len();
        let mut acc = Poly::zero();
        for i in 0..n {
            let mut term = Poly::constant(ys[i]);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = xs[i] - xs[j];
                assert!(d != 0.0, "interpolate_lagrange needs distinct nodes");
                term = term.mul(&Poly::new(vec![-xs[j] / d, 1.0 / d]));
            }
            acc = acc.add(&term);
        }
        acc
    }

    /// The same interpolating polynomial built from Newton's divided
    /// differences -- a different algorithm reaching the same answer.
    ///
    /// # Panics
    /// Panics if the two slices differ in length or if any two nodes
    /// coincide.
    #[must_use]
    pub fn interpolate_newton(xs: &[f64], ys: &[f64]) -> Self {
        assert_eq!(xs.len(), ys.len(), "interpolate_newton needs matching lengths");
        let n = xs.len();
        let mut dd = ys.to_vec();
        for k in 1..n {
            for i in (k..n).rev() {
                let d = xs[i] - xs[i - k];
                assert!(d != 0.0, "interpolate_newton needs distinct nodes");
                dd[i] = (dd[i] - dd[i - 1]) / d;
            }
        }
        let mut acc = Poly::zero();
        for i in (0..n).rev() {
            acc = acc.mul(&Poly::new(vec![-xs[i], 1.0])).add(&Poly::constant(dd[i]));
        }
        acc
    }

    /// Chebyshev coefficients of degree `n` for `f` on `[a, b]`, from the
    /// `n + 1` Chebyshev-Gauss nodes.
    ///
    /// The result is `c_0 .. c_n` for the expansion
    /// `f(x) ~ sum_k c_k T_k(t)`, `t = (2x - a - b)/(b - a)`; the usual
    /// halving of `c_0` is already applied, so [`Poly::chebyshev_eval`]
    /// consumes the coefficients directly. Use
    /// [`Poly::chebyshev_eval_on`] to evaluate in the original variable.
    ///
    /// # Panics
    /// Panics unless `a < b`.
    #[must_use]
    pub fn chebyshev_fit<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> Vec<f64> {
        assert!(a < b, "chebyshev_fit needs a < b");
        let m = n + 1;
        let mid = 0.5 * (a + b);
        let half = 0.5 * (b - a);
        let nodes: Vec<f64> = (0..m)
            .map(|j| (PI * (j as f64 + 0.5) / m as f64).cos())
            .collect();
        let vals: Vec<f64> = nodes.iter().map(|&t| f(mid + half * t)).collect();
        (0..m)
            .map(|k| {
                let s: f64 = (0..m)
                    .map(|j| vals[j] * (PI * k as f64 * (j as f64 + 0.5) / m as f64).cos())
                    .sum();
                let c = 2.0 * s / m as f64;
                if k == 0 {
                    0.5 * c
                } else {
                    c
                }
            })
            .collect()
    }

    /// Evaluate `sum_k c_k T_k(x)` on `[-1, 1]` by Clenshaw recurrence.
    #[must_use]
    pub fn chebyshev_eval(coeffs: &[f64], x: f64) -> f64 {
        let n = coeffs.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return coeffs[0];
        }
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        for k in (1..n).rev() {
            let b0 = 2.0 * x * b1 - b2 + coeffs[k];
            b2 = b1;
            b1 = b0;
        }
        coeffs[0] + x * b1 - b2
    }

    /// Evaluate a [`Poly::chebyshev_fit`] result at `x` in the original
    /// interval `[a, b]`.
    ///
    /// # Panics
    /// Panics unless `a < b`.
    #[must_use]
    pub fn chebyshev_eval_on(coeffs: &[f64], a: f64, b: f64, x: f64) -> f64 {
        assert!(a < b, "chebyshev_eval_on needs a < b");
        Self::chebyshev_eval(coeffs, (2.0 * x - a - b) / (b - a))
    }

    /// The Chebyshev polynomials `T_0 .. T_n` in the monomial basis.
    #[must_use]
    pub fn chebyshev_basis(n: usize) -> Vec<Poly> {
        let mut t = vec![Poly::constant(1.0)];
        if n == 0 {
            return t;
        }
        t.push(Poly::new(vec![0.0, 1.0]));
        for k in 2..=n {
            let next = t[k - 1].mul(&Poly::new(vec![0.0, 2.0])).sub(&t[k - 2]);
            t.push(next);
        }
        t
    }

    /// Coefficients of this polynomial in the Chebyshev basis on `[-1, 1]`,
    /// so that `p(x) = sum_k out[k] T_k(x)` exactly (up to rounding).
    ///
    /// The zero polynomial maps to an empty vector.
    #[must_use]
    pub fn to_chebyshev_basis(&self) -> Vec<f64> {
        if self.is_zero() {
            return Vec::new();
        }
        let n = self.degree();
        let t = Self::chebyshev_basis(n);
        // T_k is the only basis element of degree k, so peel from the top.
        let mut rem = self.clone();
        let mut out = vec![0.0; n + 1];
        for k in (0..=n).rev() {
            let coef = if rem.c.len() == k + 1 { rem.leading() / t[k].leading() } else { 0.0 };
            out[k] = coef;
            if coef != 0.0 {
                rem = rem.sub(&t[k].mul_scalar(coef));
            }
        }
        out
    }

    /// The monomial-basis polynomial `sum_k coeffs[k] T_k(x)`.
    #[must_use]
    pub fn from_chebyshev_basis(coeffs: &[f64]) -> Self {
        if coeffs.is_empty() {
            return Poly::zero();
        }
        let t = Self::chebyshev_basis(coeffs.len() - 1);
        let mut acc = Poly::zero();
        for (k, &c) in coeffs.iter().enumerate() {
            if c != 0.0 {
                acc = acc.add(&t[k].mul_scalar(c));
            }
        }
        acc
    }

    /// The Pade approximant `[m/n]` of the power series whose coefficients
    /// are `self.c`: the rational function `P/Q` with `deg P <= m`,
    /// `deg Q <= n`, `Q(0) = 1`, agreeing with the series through order
    /// `x^(m + n)`.
    ///
    /// Returns `None` when the series has fewer than `m + n + 1` terms or
    /// the defining linear system is singular (a degenerate Pade table
    /// entry).
    #[must_use]
    pub fn pade(&self, m: usize, n: usize) -> Option<(Poly, Poly)> {
        let a: Vec<f64> = (0..=m + n).map(|i| self.c.get(i).copied().unwrap_or(0.0)).collect();
        if self.c.len() < m + n + 1 {
            return None;
        }
        let mut q = vec![1.0; n + 1];
        if n > 0 {
            // sum_{j=1..n} q_j a_{m+k-j} = -a_{m+k}, k = 1..n
            let mut mat = vec![vec![0.0; n]; n];
            let mut rhs = vec![0.0; n];
            for k in 1..=n {
                for j in 1..=n {
                    let idx = m + k;
                    mat[k - 1][j - 1] = if idx >= j { a[idx - j] } else { 0.0 };
                }
                rhs[k - 1] = -a[m + k];
            }
            let sol = solve_linear(&mut mat, &mut rhs)?;
            q[1..=n].copy_from_slice(&sol);
        }
        let mut p = vec![0.0; m + 1];
        for (k, pk) in p.iter_mut().enumerate() {
            let mut s = 0.0;
            for j in 0..=k.min(n) {
                s += q[j] * a[k - j];
            }
            *pk = s;
        }
        Some((Poly::new(p), Poly::new(q)))
    }

    /// Wilkinson's polynomial `prod_{k=1}^{n} (x - k)`, the classic
    /// example of catastrophic root sensitivity in the monomial basis.
    #[must_use]
    pub fn wilkinson(n: usize) -> Self {
        let roots: Vec<f64> = (1..=n).map(|k| k as f64).collect();
        Poly::from_roots(&roots)
    }

    /// The `n`-th cyclotomic polynomial, exactly, from the defining
    /// identity `x^n - 1 = prod_{d | n} Phi_d(x)`.
    ///
    /// Its degree is Euler's totient of `n` and its coefficients are
    /// integers.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    #[must_use]
    pub fn cyclotomic(n: usize) -> PolyQ {
        assert!(n > 0, "cyclotomic needs n >= 1");
        // x^n - 1, divided by the cyclotomics of the proper divisors.
        let mut c = vec![Rational::zero(); n + 1];
        c[0] = Rational::from_i64(-1, 1);
        c[n] = Rational::one();
        let mut acc = PolyQ::new(c);
        for d in 1..n {
            if n.is_multiple_of(d) {
                let phi = Self::cyclotomic(d);
                let (q, r) = acc.div_rem(&phi).expect("cyclotomic factors are non-zero");
                debug_assert!(r.is_zero(), "cyclotomic division must be exact");
                acc = q;
            }
        }
        acc
    }

    /// Whether the polynomial has no repeated factor, i.e. `gcd(p, p')` is
    /// a non-zero constant within `tol`.
    #[must_use]
    pub fn is_squarefree(&self, tol: f64) -> bool {
        if self.is_zero() {
            return false;
        }
        if self.degree() == 0 {
            return true;
        }
        let g = self.gcd(&self.derivative(), tol);
        !g.is_zero() && g.degree() == 0
    }

    /// The squarefree part `p / gcd(p, p')`: the same roots, each with
    /// multiplicity one, normalized to monic.
    #[must_use]
    pub fn squarefree_part(&self, tol: f64) -> Self {
        if self.is_zero() {
            return Poly::zero();
        }
        let g = self.gcd(&self.derivative(), tol);
        if g.is_zero() || g.degree() == 0 {
            return self.mul_scalar(1.0 / self.leading());
        }
        let (q, _) = self.div_rem(&g).expect("gcd is non-zero");
        q.mul_scalar(1.0 / q.leading())
    }
}

/// A polynomial with exact rational coefficients, ordered from the constant
/// term up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyQ {
    /// Coefficients low to high: `c[i]` multiplies `x^i`. Trimmed, so the
    /// last entry is non-zero unless the vector is empty.
    pub c: Vec<Rational>,
}

/// Drop trailing zero coefficients so the representation is canonical.
fn trim_q(mut c: Vec<Rational>) -> Vec<Rational> {
    while c.last().is_some_and(Rational::is_zero) {
        c.pop();
    }
    c
}

impl PolyQ {
    /// The polynomial with the given exact coefficients, low degree first.
    #[must_use]
    pub fn new(c: Vec<Rational>) -> Self {
        PolyQ { c: trim_q(c) }
    }

    /// The zero polynomial.
    #[must_use]
    pub fn zero() -> Self {
        PolyQ { c: Vec::new() }
    }

    /// The constant polynomial `a`.
    #[must_use]
    pub fn constant(a: Rational) -> Self {
        PolyQ::new(vec![a])
    }

    /// Integer coefficients, low degree first.
    #[must_use]
    pub fn from_i64s(c: &[i64]) -> Self {
        PolyQ::new(c.iter().map(|&a| Rational::from_i64(a, 1)).collect())
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }

    /// The degree, with the convention that the zero polynomial has degree
    /// `0` (pair with [`PolyQ::is_zero`] when that distinction matters).
    #[must_use]
    pub fn degree(&self) -> usize {
        self.c.len().saturating_sub(1)
    }

    /// The leading coefficient, or zero for the zero polynomial.
    #[must_use]
    pub fn leading(&self) -> Rational {
        self.c.last().cloned().unwrap_or_else(Rational::zero)
    }

    /// The same polynomial with `f64` coefficients.
    #[must_use]
    pub fn to_poly(&self) -> Poly {
        Poly::new(self.c.iter().map(Rational::to_f64).collect())
    }

    /// Exact value at `x` by Horner's rule.
    #[must_use]
    pub fn eval(&self, x: &Rational) -> Rational {
        self.c.iter().rev().fold(Rational::zero(), |acc, a| acc.mul(x).add(a))
    }

    /// Exact sum.
    #[must_use]
    pub fn add(&self, other: &PolyQ) -> Self {
        let n = self.c.len().max(other.c.len());
        let mut c = vec![Rational::zero(); n];
        for (i, s) in self.c.iter().enumerate() {
            c[i] = c[i].add(s);
        }
        for (i, s) in other.c.iter().enumerate() {
            c[i] = c[i].add(s);
        }
        PolyQ::new(c)
    }

    /// Exact difference `self - other`.
    #[must_use]
    pub fn sub(&self, other: &PolyQ) -> Self {
        let n = self.c.len().max(other.c.len());
        let mut c = vec![Rational::zero(); n];
        for (i, s) in self.c.iter().enumerate() {
            c[i] = c[i].add(s);
        }
        for (i, s) in other.c.iter().enumerate() {
            c[i] = c[i].sub(s);
        }
        PolyQ::new(c)
    }

    /// Additive inverse.
    #[must_use]
    pub fn neg(&self) -> Self {
        PolyQ::new(self.c.iter().map(Rational::neg).collect())
    }

    /// Exact schoolbook product.
    #[must_use]
    pub fn mul(&self, other: &PolyQ) -> Self {
        if self.is_zero() || other.is_zero() {
            return PolyQ::zero();
        }
        let mut c = vec![Rational::zero(); self.c.len() + other.c.len() - 1];
        for (i, a) in self.c.iter().enumerate() {
            for (j, b) in other.c.iter().enumerate() {
                c[i + j] = c[i + j].add(&a.mul(b));
            }
        }
        PolyQ::new(c)
    }

    /// Every coefficient multiplied by `k`.
    #[must_use]
    pub fn mul_scalar(&self, k: &Rational) -> Self {
        PolyQ::new(self.c.iter().map(|a| a.mul(k)).collect())
    }

    /// Every coefficient divided by `k`, or `None` when `k` is zero.
    #[must_use]
    pub fn div_scalar(&self, k: &Rational) -> Option<Self> {
        if k.is_zero() {
            return None;
        }
        Some(PolyQ::new(self.c.iter().map(|a| a.div(k).expect("k is non-zero")).collect()))
    }

    /// The monic associate `p / lc(p)`; the zero polynomial maps to itself.
    #[must_use]
    pub fn monic(&self) -> Self {
        if self.is_zero() {
            return PolyQ::zero();
        }
        self.div_scalar(&self.leading()).expect("leading coefficient is non-zero")
    }

    /// Exact quotient and remainder, satisfying `self == q * divisor + r`
    /// with `r` of lower degree than `divisor`.
    ///
    /// Returns `None` when `divisor` is the zero polynomial.
    #[must_use]
    pub fn div_rem(&self, divisor: &PolyQ) -> Option<(Self, Self)> {
        if divisor.is_zero() {
            return None;
        }
        let dd = divisor.degree();
        if self.is_zero() || self.c.len() < divisor.c.len() {
            return Some((PolyQ::zero(), self.clone()));
        }
        let dl = divisor.leading();
        let mut r = self.c.clone();
        let mut q = vec![Rational::zero(); self.c.len() - divisor.c.len() + 1];
        for i in (0..q.len()).rev() {
            let coef = r[i + dd].div(&dl).expect("leading coefficient is non-zero");
            for j in 0..dd {
                r[i + j] = r[i + j].sub(&coef.mul(&divisor.c[j]));
            }
            r[i + dd] = Rational::zero();
            q[i] = coef;
        }
        Some((PolyQ::new(q), PolyQ::new(r)))
    }

    /// Exact derivative.
    #[must_use]
    pub fn derivative(&self) -> Self {
        if self.c.len() < 2 {
            return PolyQ::zero();
        }
        PolyQ::new(
            self.c
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, a)| a.mul(&Rational::from_i64(i as i64, 1)))
                .collect(),
        )
    }

    /// Exact antiderivative with constant term `c0`.
    #[must_use]
    pub fn integral(&self, c0: &Rational) -> Self {
        let mut c = Vec::with_capacity(self.c.len() + 1);
        c.push(c0.clone());
        for (i, a) in self.c.iter().enumerate() {
            c.push(a.div(&Rational::from_i64(i as i64 + 1, 1)).expect("non-zero divisor"));
        }
        PolyQ::new(c)
    }

    /// Exact composition `self(inner(x))`.
    #[must_use]
    pub fn compose(&self, inner: &PolyQ) -> Self {
        let mut acc = PolyQ::zero();
        for a in self.c.iter().rev() {
            acc = acc.mul(inner).add(&PolyQ::constant(a.clone()));
        }
        acc
    }

    /// The polynomial `p(k*x)`.
    #[must_use]
    pub fn scale_arg(&self, k: &Rational) -> Self {
        let mut p = Rational::one();
        let mut c = self.c.clone();
        for (i, a) in c.iter_mut().enumerate() {
            if i > 0 {
                p = p.mul(k);
            }
            *a = a.mul(&p);
        }
        PolyQ::new(c)
    }

    /// The polynomial `p(x + h)`, by repeated synthetic division.
    #[must_use]
    pub fn shift_arg(&self, h: &Rational) -> Self {
        let n = self.c.len();
        if n < 2 {
            return self.clone();
        }
        let mut b = self.c.clone();
        for i in 0..n {
            for j in (i..n - 1).rev() {
                b[j] = b[j].add(&h.mul(&b[j + 1]));
            }
        }
        PolyQ::new(b)
    }

    /// The monic polynomial with exactly the given rational roots.
    #[must_use]
    pub fn from_roots(roots: &[Rational]) -> Self {
        let mut p = PolyQ::constant(Rational::one());
        for r in roots {
            p = p.mul(&PolyQ::new(vec![r.neg(), Rational::one()]));
        }
        p
    }

    /// The content: the GCD of the numerators over the LCM of the
    /// denominators, signed like the leading coefficient.
    ///
    /// Dividing by it yields an integer-coefficient polynomial with
    /// positive leading coefficient and coefficient GCD one, so
    /// `content * primitive_part == self` exactly. The zero polynomial has
    /// content zero.
    #[must_use]
    pub fn content(&self) -> Rational {
        if self.is_zero() {
            return Rational::zero();
        }
        let mut num_g = BigInt::zero();
        let mut den_l = BigInt::one();
        for a in &self.c {
            num_g = num_g.gcd(&a.num.abs());
            den_l = den_l.lcm(&a.den);
        }
        let cont = Rational::new(num_g, den_l).expect("denominator LCM is non-zero");
        if self.leading().is_negative() {
            cont.neg()
        } else {
            cont
        }
    }

    /// `self / content`: integer coefficients with GCD one and a positive
    /// leading coefficient. The zero polynomial maps to itself.
    #[must_use]
    pub fn primitive_part(&self) -> Self {
        if self.is_zero() {
            return PolyQ::zero();
        }
        self.div_scalar(&self.content()).expect("content of a non-zero polynomial is non-zero")
    }

    /// Pseudo-division: the pair `(q, r)` with
    /// `lc(b)^(deg a - deg b + 1) * a == q * b + r` and `deg r < deg b`.
    ///
    /// Returns `None` when `b` is zero. When `deg a < deg b` the multiplier
    /// is one and the answer is `(0, a)`.
    #[must_use]
    pub fn pseudo_div(&self, b: &PolyQ) -> Option<(Self, Self)> {
        if b.is_zero() {
            return None;
        }
        if self.is_zero() || self.c.len() < b.c.len() {
            return Some((PolyQ::zero(), self.clone()));
        }
        let d = self.degree() - b.degree();
        let mult = b.leading().pow(d as i64 + 1);
        self.mul_scalar(&mult).div_rem(b)
    }

    /// Monic greatest common divisor, by the subresultant polynomial
    /// remainder sequence.
    ///
    /// The subresultant scaling keeps the intermediate coefficients from
    /// exploding the way a naive pseudo-remainder chain does, while every
    /// step stays exact. `gcd(0, 0)` is the zero polynomial.
    #[must_use]
    pub fn gcd_exact(&self, other: &PolyQ) -> Self {
        if self.is_zero() && other.is_zero() {
            return PolyQ::zero();
        }
        if self.is_zero() {
            return other.monic();
        }
        if other.is_zero() {
            return self.monic();
        }
        let (mut a, mut b) = if self.degree() >= other.degree() {
            (self.primitive_part(), other.primitive_part())
        } else {
            (other.primitive_part(), self.primitive_part())
        };
        let one = PolyQ::constant(Rational::one());
        if b.degree() == 0 {
            return one;
        }
        let mut g = Rational::one();
        let mut h = Rational::one();
        loop {
            let d = a.degree() - b.degree();
            let (_, r) = a.pseudo_div(&b).expect("b is non-zero");
            if r.is_zero() {
                return b.monic();
            }
            if r.degree() == 0 {
                return one;
            }
            let denom = g.mul(&h.pow(d as i64));
            let next = r.div_scalar(&denom).expect("subresultant divisor is non-zero");
            a = b;
            b = next;
            g = a.leading();
            h = g.pow(d as i64).mul(&h.pow(1 - d as i64));
        }
    }

    /// Whether the polynomial has no repeated factor, i.e. `gcd(p, p')` is
    /// a constant. The zero polynomial is not squarefree.
    #[must_use]
    pub fn is_squarefree(&self) -> bool {
        if self.is_zero() {
            return false;
        }
        if self.degree() == 0 {
            return true;
        }
        self.gcd_exact(&self.derivative()).degree() == 0
    }

    /// The monic squarefree part `p / gcd(p, p')`.
    #[must_use]
    pub fn squarefree_part(&self) -> Self {
        if self.is_zero() {
            return PolyQ::zero();
        }
        let g = self.gcd_exact(&self.derivative());
        if g.is_zero() || g.degree() == 0 {
            return self.monic();
        }
        let (q, _) = self.div_rem(&g).expect("gcd is non-zero");
        q.monic()
    }

    /// The exact resultant, as the determinant of the Sylvester matrix.
    #[must_use]
    pub fn resultant(&self, other: &PolyQ) -> Rational {
        if self.is_zero() || other.is_zero() {
            return Rational::zero();
        }
        let m = self.degree();
        let n = other.degree();
        if m == 0 && n == 0 {
            return Rational::one();
        }
        if m == 0 {
            return self.c[0].pow(n as i64);
        }
        if n == 0 {
            return other.c[0].pow(m as i64);
        }
        let size = m + n;
        let mut a = vec![vec![Rational::zero(); size]; size];
        for r in 0..n {
            for (k, v) in self.c.iter().rev().enumerate() {
                a[r][r + k] = v.clone();
            }
        }
        for r in 0..m {
            for (k, v) in other.c.iter().rev().enumerate() {
                a[n + r][r + k] = v.clone();
            }
        }
        crate::exact::rational::determinant_exact(&a)
    }

    /// The exact discriminant `(-1)^(n(n-1)/2) Res(p, p') / lc(p)`.
    ///
    /// Returns zero for the zero polynomial and for non-zero constants, and
    /// one for a linear polynomial.
    #[must_use]
    pub fn discriminant(&self) -> Rational {
        let n = self.degree();
        if self.is_zero() || n == 0 {
            return Rational::zero();
        }
        if n == 1 {
            return Rational::one();
        }
        let res = self.resultant(&self.derivative());
        let signed = if (n * (n - 1) / 2).is_multiple_of(2) { res } else { res.neg() };
        signed.div(&self.leading()).expect("leading coefficient is non-zero")
    }

    /// Every rational root with its multiplicity, in increasing order, by
    /// the rational root theorem.
    ///
    /// The polynomial is first made primitive (integer coefficients);
    /// candidates are `+-p/q` for `p` dividing the constant term and `q`
    /// the leading one, and each hit is divided out repeatedly to get its
    /// multiplicity. A zero root is handled separately.
    ///
    /// Divisor search is by trial division, so a constant or leading term
    /// whose magnitude exceeds `10^12` is out of reach and contributes no
    /// candidates.
    #[must_use]
    pub fn factor_rational_roots(&self) -> Vec<(Rational, usize)> {
        let mut out: Vec<(Rational, usize)> = Vec::new();
        if self.is_zero() {
            return out;
        }
        let mut work = self.clone();
        let mut zero_mult = 0;
        while !work.is_zero() && work.c[0].is_zero() {
            work = PolyQ::new(work.c[1..].to_vec());
            zero_mult += 1;
        }
        if zero_mult > 0 {
            out.push((Rational::zero(), zero_mult));
        }
        if work.degree() == 0 {
            return out;
        }
        let prim = work.primitive_part();
        let (ps, qs) = match (divisors(&prim.c[0].num), divisors(&prim.leading().num)) {
            (Some(ps), Some(qs)) => (ps, qs),
            _ => return out,
        };
        let mut cands: Vec<Rational> = Vec::new();
        for p in &ps {
            for q in &qs {
                let r = Rational::new(p.clone(), q.clone()).expect("divisors are non-zero");
                for cand in [r.clone(), r.neg()] {
                    if !cands.contains(&cand) {
                        cands.push(cand);
                    }
                }
            }
        }
        for cand in cands {
            if !work.eval(&cand).is_zero() {
                continue;
            }
            let lin = PolyQ::new(vec![cand.neg(), Rational::one()]);
            let mut m = 0;
            loop {
                let (q, r) = work.div_rem(&lin).expect("linear factor is non-zero");
                if !r.is_zero() {
                    break;
                }
                work = q;
                m += 1;
            }
            if m > 0 {
                out.push((cand, m));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Eisenstein's irreducibility criterion at the prime `p`, applied to
    /// the primitive part: `p` divides every coefficient but the leading
    /// one, and `p^2` does not divide the constant term.
    ///
    /// `true` proves irreducibility over the rationals; `false` proves
    /// nothing. `p` is assumed prime -- primality is not verified -- and a
    /// polynomial of degree below one always returns `false`.
    #[must_use]
    pub fn eisenstein_check(&self, p: &BigInt) -> bool {
        if self.is_zero() || self.degree() < 1 || p.is_negative() || *p <= BigInt::one() {
            return false;
        }
        let prim = self.primitive_part();
        let n = prim.degree();
        if prim.c[n].num.rem_euclid(p).is_zero() {
            return false;
        }
        for a in prim.c.iter().take(n) {
            if !a.num.rem_euclid(p).is_zero() {
                return false;
            }
        }
        let p2 = p.mul(p);
        !prim.c[0].num.rem_euclid(&p2).is_zero()
    }
}

/// Every positive divisor of `n`, by trial division. `None` for zero and for
/// magnitudes past [`DIVISOR_SEARCH_MAX`], where trial division stops being
/// affordable.
fn divisors(n: &BigInt) -> Option<Vec<BigInt>> {
    if n.is_zero() {
        return None;
    }
    let v = n.abs().to_i64()?;
    if v > DIVISOR_SEARCH_MAX {
        return None;
    }
    let mut out = Vec::new();
    let mut d = 1_i64;
    while d * d <= v {
        if v % d == 0 {
            out.push(BigInt::from_i64(d));
            if d != v / d {
                out.push(BigInt::from_i64(v / d));
            }
        }
        d += 1;
    }
    Some(out)
}

/// Trial division stays under a million steps below this bound.
const DIVISOR_SEARCH_MAX: i64 = 1_000_000_000_000;

/// Binomial coefficient `C(n, k)` as an `f64`, by the multiplicative
/// formula (exact for the sizes these polynomial routines reach).
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut acc = 1.0;
    for i in 0..k {
        acc = acc * (n - i) as f64 / (i + 1) as f64;
    }
    acc
}

/// Product of two polynomials through the FFT: transform, multiply
/// pointwise, transform back.
///
/// Mathematically identical to [`Poly::mul`], and asymptotically faster,
/// at the cost of rounding on the order of `eps * n * max|a| * max|b|`.
#[must_use]
pub fn polynomial_multiply_fft(a: &Poly, b: &Poly) -> Poly {
    if a.is_zero() || b.is_zero() {
        return Poly::zero();
    }
    let out_len = a.c.len() + b.c.len() - 1;
    let mut size = 1;
    while size < out_len {
        size <<= 1;
    }
    let embed = |p: &Poly| -> Vec<Complex> {
        (0..size)
            .map(|i| Complex::new(p.c.get(i).copied().unwrap_or(0.0), 0.0))
            .collect()
    };
    let fa = fft(&embed(a));
    let fb = fft(&embed(b));
    let prod: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&x, &y)| x * y).collect();
    let back = ifft(&prod);
    Poly::new(back.iter().take(out_len).map(|z| z.re).collect())
}

/// The Bernstein basis polynomial `B_{i,n}(t) = C(n, i) t^i (1 - t)^(n - i)`.
///
/// Returns `0.0` when `i > n`.
#[must_use]
pub fn bernstein_basis(n: usize, i: usize, t: f64) -> f64 {
    if i > n {
        return 0.0;
    }
    binom(n, i) * t.powi(i as i32) * (1.0 - t).powi((n - i) as i32)
}

/// Bernstein coefficients of `p` on `[a, b]`.
///
/// The returned `w` of length `deg(p) + 1` satisfies
/// `p(a + (b - a) t) = sum_i w[i] * bernstein_basis(n, i, t)` for all `t`,
/// which is the control polygon of `p` viewed as a Bezier curve.
///
/// # Panics
/// Panics unless `a < b`.
#[must_use]
pub fn to_bernstein(p: &Poly, a: f64, b: f64) -> Vec<f64> {
    assert!(a < b, "to_bernstein needs a < b");
    let q = p.shift_arg(a).scale_arg(b - a);
    let n = p.degree();
    (0..=n)
        .map(|i| {
            (0..=i)
                .map(|k| binom(i, k) / binom(n, k) * q.c.get(k).copied().unwrap_or(0.0))
                .sum()
        })
        .collect()
}

/// Elementary symmetric functions from power sums, by Newton's identities.
///
/// Given `p_1 .. p_n` in `power_sums`, returns `e_0 .. e_n` (so the result
/// is one longer, and starts at `e_0 = 1`), using
/// `k e_k = sum_{i=1}^{k} (-1)^(i-1) e_{k-i} p_i`.
#[must_use]
pub fn newton_identities(power_sums: &[f64]) -> Vec<f64> {
    let n = power_sums.len();
    let mut e = vec![0.0; n + 1];
    e[0] = 1.0;
    for k in 1..=n {
        let mut s = 0.0;
        for i in 1..=k {
            let sign = if i % 2 == 1 { 1.0 } else { -1.0 };
            s += sign * e[k - i] * power_sums[i - 1];
        }
        e[k] = s / k as f64;
    }
    e
}

/// Coefficients of the monic polynomial with the given complex roots, low
/// degree first (Vieta's formulas).
///
/// Entry `n - k` is `(-1)^k e_k`, the signed `k`-th elementary symmetric
/// function of the roots; the leading entry is `1`.
#[must_use]
pub fn vieta(roots: &[Complex]) -> Vec<Complex> {
    let mut c = vec![Complex::new(1.0, 0.0)];
    for &r in roots {
        let mut next = vec![Complex::new(0.0, 0.0); c.len() + 1];
        for (i, &ci) in c.iter().enumerate() {
            next[i + 1] = next[i + 1] + ci;
            next[i] = next[i] - ci * r;
        }
        c = next;
    }
    c
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    // Tolerance rationale. Every f64 assertion below runs on polynomials of
    // degree <= 12 with coefficients of order 1..10, so a single arithmetic
    // chain accumulates at most a few hundred roundings: ~1e-13 absolute on
    // quantities of order 1. The bounds are set an order or two above that
    // measured noise, never loose enough to hide a wrong algorithm.
    /// Coefficient-level agreement for exactly-equal expressions.
    const EXACT_TOL: f64 = 1e-9;
    /// Root locations recovered through an iterative solver.
    const ROOT_TOL: f64 = 1e-7;

    fn rq(n: i64, d: i64) -> Rational {
        Rational::from_i64(n, d)
    }

    fn max_coeff_diff(a: &Poly, b: &Poly) -> f64 {
        let n = a.c.len().max(b.c.len());
        (0..n).fold(0.0_f64, |m, i| {
            let x = a.c.get(i).copied().unwrap_or(0.0);
            let y = b.c.get(i).copied().unwrap_or(0.0);
            m.max((x - y).abs())
        })
    }

    fn rand_poly(rng: &mut Rng, deg: usize) -> Poly {
        let mut c: Vec<f64> = (0..=deg).map(|_| rng.next_f64().mul_add(10.0, -5.0)).collect();
        if c[deg].abs() < 0.5 {
            c[deg] = 1.0 + rng.next_f64();
        }
        Poly::new(c)
    }

    fn rand_polyq(rng: &mut Rng, deg: usize) -> PolyQ {
        let mut c: Vec<Rational> = (0..=deg)
            .map(|_| rq((rng.next_u64() % 21) as i64 - 10, 1 + (rng.next_u64() % 7) as i64))
            .collect();
        if c[deg].is_zero() {
            c[deg] = Rational::one();
        }
        PolyQ::new(c)
    }

    fn totient(n: usize) -> usize {
        (1..=n).filter(|k| gcd_usize(*k, n) == 1).count()
    }

    fn gcd_usize(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }

    #[test]
    fn ring_axioms_and_evaluation() {
        let mut rng = Rng::new(7);
        for _ in 0..40 {
            let a = rand_poly(&mut rng, 4);
            let b = rand_poly(&mut rng, 3);
            let c = rand_poly(&mut rng, 2);
            // Distributivity and commutativity of the polynomial ring.
            assert!(
                max_coeff_diff(&a.mul(&b.add(&c)), &a.mul(&b).add(&a.mul(&c))) < EXACT_TOL,
                "distributive law"
            );
            assert!(max_coeff_diff(&a.mul(&b), &b.mul(&a)) < EXACT_TOL, "commutativity");
            assert!(a.add(&a.neg()).is_zero(), "additive inverse trims to zero");
            // Degrees add, and evaluation is a ring homomorphism.
            assert_eq!(a.mul(&b).degree(), a.degree() + b.degree());
            let x = rng.next_f64().mul_add(4.0, -2.0);
            assert!((a.mul(&b).eval(x) - a.eval(x) * b.eval(x)).abs() < 1e-9, "eval of product");
            assert!((a.add(&b).eval(x) - (a.eval(x) + b.eval(x))).abs() < 1e-9, "eval of sum");
            // Horner over the reals agrees with the complex path.
            let z = a.eval_complex(Complex::new(x, 0.0));
            assert!((z.re - a.eval(x)).abs() < 1e-9 && z.im == 0.0, "eval_complex on the real axis");
        }
        assert!(Poly::zero().is_zero());
        assert_eq!(Poly::new(vec![1.0, 0.0, 0.0]).c, vec![1.0], "trailing zeros trimmed");
    }

    #[test]
    fn div_rem_reconstructs_the_dividend() {
        let mut rng = Rng::new(11);
        for _ in 0..60 {
            let d_a = 2 + (rng.next_u64() % 7) as usize;
            let a = rand_poly(&mut rng, d_a);
            let d_b = 1 + (rng.below(4)) as usize;
            let b = rand_poly(&mut rng, d_b);
            let (q, r) = a.div_rem(&b).expect("non-zero divisor");
            assert!(r.is_zero() || r.degree() < b.degree(), "remainder degree drops");
            let back = q.mul(&b).add(&r);
            let scale = a.c.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
            assert!(max_coeff_diff(&a, &back) < 1e-9 * scale, "a == q*b + r");
        }
        // Exact division leaves no remainder.
        let f = Poly::from_roots(&[1.0, -2.0, 3.5]);
        let g = Poly::from_roots(&[-2.0]);
        let (q, r) = f.div_rem(&g).expect("non-zero divisor");
        assert!(r.is_zero(), "exact division has zero remainder");
        assert!(max_coeff_diff(&q, &Poly::from_roots(&[1.0, 3.5])) < EXACT_TOL);
        assert!(Poly::constant(1.0).div_rem(&Poly::zero()).is_none(), "division by zero");
    }

    #[test]
    fn calculus_and_composition() {
        let mut rng = Rng::new(13);
        for _ in 0..30 {
            let p = rand_poly(&mut rng, 5);
            // The fundamental theorem of calculus, both directions.
            assert!(max_coeff_diff(&p.integral(3.25).derivative(), &p) < EXACT_TOL, "d/dx of integral");
            let back = p.derivative().integral(p.c[0]);
            assert!(max_coeff_diff(&back, &p) < EXACT_TOL, "integral of derivative up to the constant");
            // Product rule.
            let q = rand_poly(&mut rng, 3);
            let lhs = p.mul(&q).derivative();
            let rhs = p.derivative().mul(&q).add(&p.mul(&q.derivative()));
            assert!(max_coeff_diff(&lhs, &rhs) < 1e-8, "product rule");
            // Chain rule through compose.
            let r = rand_poly(&mut rng, 2);
            let d_comp = p.compose(&r).derivative();
            let chain = p.derivative().compose(&r).mul(&r.derivative());
            assert!(max_coeff_diff(&d_comp, &chain) < 1e-6, "chain rule");
        }
        // Composition is associative (and not commutative).
        let f = Poly::new(vec![1.0, 0.0, 2.0]);
        let g = Poly::new(vec![-1.0, 3.0]);
        let h = Poly::new(vec![2.0, 1.0, 1.0]);
        let left = f.compose(&g).compose(&h);
        let right = f.compose(&g.compose(&h));
        assert!(max_coeff_diff(&left, &right) < EXACT_TOL, "compose associativity");
        assert!(max_coeff_diff(&f.compose(&g), &g.compose(&f)) > 1.0, "compose is not commutative");
        // Argument scaling and shifting agree with direct evaluation.
        let p = Poly::new(vec![2.0, -1.0, 0.5, 3.0]);
        for i in 0..9 {
            let x = -2.0 + 0.5 * i as f64;
            assert!((p.scale_arg(1.5).eval(x) - p.eval(1.5 * x)).abs() < EXACT_TOL, "p(kx)");
            assert!((p.shift_arg(0.75).eval(x) - p.eval(x + 0.75)).abs() < EXACT_TOL, "p(x+h)");
        }
        assert!(max_coeff_diff(&p.shift_arg(2.0).shift_arg(-2.0), &p) < EXACT_TOL, "shift is invertible");
    }

    #[test]
    fn roots_from_roots_and_vieta() {
        let want = [-2.5_f64, 0.5, 1.0, 3.0];
        let p = Poly::from_roots(&want);
        assert_eq!(p.degree(), 4);
        assert!((p.leading() - 1.0).abs() < f64::EPSILON, "from_roots is monic");
        for &r in &want {
            assert!(p.eval(r).abs() < 1e-12, "constructed root {r} evaluates to zero");
        }
        let mut got: Vec<f64> = p
            .roots()
            .expect("degree 4 has roots")
            .iter()
            .map(|z| {
                assert!(z.im.abs() < ROOT_TOL, "these roots are real");
                z.re
            })
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < ROOT_TOL, "recovered {a} vs {b}");
        }
        // Vieta: the monic coefficients are the signed elementary symmetric
        // functions of the roots.
        let cx: Vec<Complex> = want.iter().map(|&r| Complex::new(r, 0.0)).collect();
        let v = vieta(&cx);
        assert_eq!(v.len(), 5);
        for (i, z) in v.iter().enumerate() {
            assert!((z.re - p.c[i]).abs() < EXACT_TOL && z.im.abs() < EXACT_TOL, "vieta coefficient {i}");
        }
        let e1: f64 = want.iter().sum();
        let e4: f64 = want.iter().product();
        assert!((p.c[3] + e1).abs() < EXACT_TOL, "sum of roots = -c_{{n-1}}");
        assert!((p.c[0] - e4).abs() < EXACT_TOL, "product of roots = (-1)^n c_0");
        // A complex-conjugate pair gives a real quadratic.
        let pair = vieta(&[Complex::new(1.0, 2.0), Complex::new(1.0, -2.0)]);
        assert!((pair[0].re - 5.0).abs() < EXACT_TOL && pair[0].im.abs() < EXACT_TOL);
        assert!((pair[1].re + 2.0).abs() < EXACT_TOL && pair[1].im.abs() < EXACT_TOL);
    }

    #[test]
    fn resultant_and_discriminant() {
        // Res(p, x - k) = (-1)^deg(p) p(k), an identity that pins both the
        // Sylvester layout and the determinant.
        let mut rng = Rng::new(17);
        for _ in 0..20 {
            let p = rand_poly(&mut rng, 4);
            for k in -2..=2 {
                let k = k as f64;
                let res = p.resultant(&Poly::new(vec![-k, 1.0]));
                let want = p.eval(k); // (-1)^4 = 1 for degree 4
                assert!((res - want).abs() < 1e-9 * (1.0 + want.abs()), "Res(p, x-k) = p(k)");
            }
        }
        // Zero exactly on a shared root, non-zero otherwise.
        let a = Poly::from_roots(&[1.0, 2.0, 3.0]);
        assert!(a.resultant(&Poly::from_roots(&[3.0, 4.0])).abs() < 1e-9, "shared root");
        let res = a.resultant(&Poly::from_roots(&[4.0, 5.0]));
        assert!((res - 144.0).abs() < 1e-9, "prod q(alpha) = 144, got {res}");
        assert_eq!(a.resultant(&Poly::zero()), 0.0);
        // Discriminant of a quadratic is b^2 - 4ac.
        for _ in 0..20 {
            let q = rand_poly(&mut rng, 2);
            let (c, b, a2) = (q.c[0], q.c[1], q.c[2]);
            let want = b * b - 4.0 * a2 * c;
            assert!((q.discriminant() - want).abs() < 1e-9 * (1.0 + want.abs()), "b^2-4ac");
        }
        // A repeated root kills the discriminant; distinct roots do not.
        assert!(Poly::from_roots(&[2.0, 2.0, 5.0]).discriminant().abs() < 1e-9, "repeated root");
        assert!(Poly::from_roots(&[1.0, 2.0, 3.0]).discriminant().abs() > 1.0, "distinct roots");
        assert!((Poly::new(vec![-3.0, 1.0]).discriminant() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sturm_counts_wilkinson_roots() {
        let w = Poly::wilkinson(10);
        assert_eq!(w.degree(), 10);
        assert!((w.leading() - 1.0).abs() < f64::EPSILON);
        assert!((w.c[0] - 3_628_800.0).abs() < 1e-6, "constant term is 10!");
        // The headline invariant: all ten roots are in (0.5, 10.5).
        assert_eq!(w.count_real_roots(0.5, 10.5), 10, "Sturm count on wilkinson(10)");
        // Sub-intervals partition the count.
        assert_eq!(w.count_real_roots(0.5, 5.5), 5);
        assert_eq!(w.count_real_roots(5.5, 10.5), 5);
        assert_eq!(w.count_real_roots(10.5, 20.0), 0, "no roots beyond 10");
        assert_eq!(w.count_real_roots(-20.0, 0.5), 0, "no non-positive roots");
        for k in 1..=10 {
            assert_eq!(w.count_real_roots(k as f64 - 0.5, k as f64 + 0.5), 1, "root {k} isolated");
        }
        // Multiple roots are counted once by Sturm's theorem.
        let m = Poly::from_roots(&[1.0, 1.0, 1.0, 4.0]);
        assert_eq!(m.count_real_roots(0.0, 5.0), 2, "distinct roots only");
        assert_eq!(m.count_real_roots(5.0, 0.0), 0, "empty interval");
        // Isolation plus refinement recovers 1..10.
        let iso = w.isolate_real_roots();
        assert_eq!(iso.len(), 10, "one interval per root");
        for (i, &iv) in iso.iter().enumerate() {
            let r = w.refine_root(iv, 1e-12);
            assert!((r - (i + 1) as f64).abs() < 1e-6, "refined root {r} vs {}", i + 1);
        }
        // An even-multiplicity root has no sign change; the Newton fallback
        // still lands on it.
        let sq = Poly::from_roots(&[2.0, 2.0]);
        assert!((sq.refine_root((1.0, 3.0), 1e-12) - 2.0).abs() < 1e-6, "double root");
        assert!(Poly::new(vec![1.0, 0.0, 1.0]).isolate_real_roots().is_empty(), "x^2+1 has no real root");
    }

    #[test]
    fn lagrange_and_newton_interpolation_agree() {
        let xs = [-2.0_f64, -0.5, 0.0, 1.0, 2.5, 4.0];
        let f = |x: f64| 2.0 * x * x * x - x * x + 0.5 * x - 7.0;
        let ys: Vec<f64> = xs.iter().map(|&x| f(x)).collect();
        let l = Poly::interpolate_lagrange(&xs, &ys);
        let n = Poly::interpolate_newton(&xs, &ys);
        // Two independent algorithms, one unique interpolant.
        assert!(max_coeff_diff(&l, &n) < 1e-10, "Lagrange == Newton");
        // Both reproduce the samples.
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert!((l.eval(x) - y).abs() < 1e-10, "Lagrange at node");
            assert!((n.eval(x) - y).abs() < 1e-10, "Newton at node");
        }
        // Interpolating a cubic at six nodes recovers that cubic exactly.
        let cubic = Poly::new(vec![-7.0, 0.5, -1.0, 2.0]);
        assert!(max_coeff_diff(&l, &cubic) < 1e-10, "unique interpolant of a cubic");
        // The two surplus degrees cancel to rounding, not to exact zero.
        for k in 4..l.c.len() {
            assert!(l.c[k].abs() < 1e-12, "surplus coefficient {k} cancels");
        }
        // Off-node agreement too.
        let mut rng = Rng::new(23);
        for _ in 0..20 {
            let x = rng.next_f64().mul_add(6.0, -2.0);
            assert!((l.eval(x) - f(x)).abs() < 1e-9);
        }
    }

    #[test]
    fn chebyshev_fit_and_basis() {
        // The roadmap's bar: degree 10 fit of exp on [-1, 1].
        let coeffs = Poly::chebyshev_fit(f64::exp, -1.0, 1.0, 10);
        assert_eq!(coeffs.len(), 11);
        let mut worst = 0.0_f64;
        for i in 0..=2000 {
            let x = -1.0 + 2.0 * i as f64 / 2000.0;
            worst = worst.max((Poly::chebyshev_eval(&coeffs, x) - x.exp()).abs());
        }
        assert!(worst < 1e-9, "max |cheb(exp) - exp| = {worst:e}");
        // Clenshaw agrees with a direct sum over the T_k.
        let basis = Poly::chebyshev_basis(10);
        for i in 0..=40 {
            let x = -1.0 + i as f64 / 20.0;
            let direct: f64 = coeffs.iter().zip(basis.iter()).map(|(&c, t)| c * t.eval(x)).sum();
            assert!((Poly::chebyshev_eval(&coeffs, x) - direct).abs() < 1e-12, "Clenshaw == direct sum");
        }
        // Fitting on a shifted interval works through the mapped evaluator.
        let on_ab = Poly::chebyshev_fit(f64::sin, 0.0, 3.0, 14);
        for i in 0..=60 {
            let x = 3.0 * i as f64 / 60.0;
            assert!((Poly::chebyshev_eval_on(&on_ab, 0.0, 3.0, x) - x.sin()).abs() < 1e-12, "sin on [0,3]");
        }
        // T_2 = 2x^2 - 1, so x^2 = (T_0 + T_2)/2.
        let x2 = Poly::new(vec![0.0, 0.0, 1.0]).to_chebyshev_basis();
        assert!((x2[0] - 0.5).abs() < 1e-15 && x2[1].abs() < 1e-15 && (x2[2] - 0.5).abs() < 1e-15);
        // Basis round trip.
        let mut rng = Rng::new(29);
        for _ in 0..20 {
            let p = rand_poly(&mut rng, 6);
            let back = Poly::from_chebyshev_basis(&p.to_chebyshev_basis());
            assert!(max_coeff_diff(&p, &back) < 1e-10, "monomial -> Chebyshev -> monomial");
            let x = rng.next_f64().mul_add(2.0, -1.0);
            let via_cheb = Poly::chebyshev_eval(&p.to_chebyshev_basis(), x);
            assert!((via_cheb - p.eval(x)).abs() < 1e-11, "same value in either basis");
        }
        assert!(Poly::zero().to_chebyshev_basis().is_empty());
    }

    #[test]
    fn pade_matches_the_series_to_order() {
        // exp's Taylor coefficients, enough for [3/3].
        let mut a = vec![1.0; 8];
        for k in 1..8 {
            a[k] = a[k - 1] / k as f64;
        }
        let series = Poly::new(a.clone());
        let (p, q) = series.pade(3, 3).expect("the [3/3] entry is non-degenerate");
        assert_eq!(p.degree(), 3);
        assert_eq!(q.degree(), 3);
        assert!((q.c[0] - 1.0).abs() < 1e-15, "Q is normalized to Q(0) = 1");
        // The classic [3/3] approximant of exp.
        let want_p = Poly::new(vec![1.0, 0.5, 0.1, 1.0 / 120.0]);
        let want_q = Poly::new(vec![1.0, -0.5, 0.1, -1.0 / 120.0]);
        assert!(max_coeff_diff(&p, &want_p) < 1e-12, "numerator");
        assert!(max_coeff_diff(&q, &want_q) < 1e-12, "denominator");
        // Defining property: series*Q - P vanishes through x^(m+n).
        let resid = series.mul(&q).sub(&p);
        for k in 0..=6 {
            let v = resid.c.get(k).copied().unwrap_or(0.0);
            assert!(v.abs() < 1e-12, "residual coefficient {k} = {v:e}");
        }
        assert!(resid.c.get(7).copied().unwrap_or(0.0).abs() > 1e-9, "order 7 does not vanish");
        // The error follows the theoretical order-(m+n+1) law
        // |P/Q - exp| ~ [m! n! / ((m+n)! (m+n+1)!)] x^7 e^x, whose constant
        // is 36/(720*5040) = 9.92e-6. Bracketing it within +-10% asserts the
        // approximation order itself, not just "small".
        let law = 36.0 / (720.0 * 5040.0);
        for i in 1..=10 {
            let x = 0.1 * i as f64;
            let err = (p.eval(x) / q.eval(x) - x.exp()).abs();
            let want = law * x.powi(7) * x.exp();
            assert!(err > 0.9 * want && err < 1.1 * want, "Pade error at {x} = {err:e}, law {want:e}");
        }
        // The [3/3] entry consumes a_0..a_6, so its fair rival is the
        // degree-6 truncation of the same series: at x = 1 the approximant
        // is an order of magnitude closer for the same information.
        let trunc = Poly::new(a[..7].to_vec());
        let pade_err = (p.eval(1.0) / q.eval(1.0) - 1.0_f64.exp()).abs();
        let trunc_err = (trunc.eval(1.0) - 1.0_f64.exp()).abs();
        assert!(pade_err < 0.2 * trunc_err, "Pade {pade_err:e} beats truncation {trunc_err:e}");
        assert!(Poly::new(vec![1.0, 1.0]).pade(3, 3).is_none(), "not enough series terms");
    }

    #[test]
    fn fft_multiplication_matches_schoolbook() {
        let mut rng = Rng::new(31);
        for _ in 0..25 {
            let d_a = 1 + (rng.next_u64() % 20) as usize;
            let a = rand_poly(&mut rng, d_a);
            let d_b = 1 + (rng.next_u64() % 20) as usize;
            let b = rand_poly(&mut rng, d_b);
            let want = a.mul(&b);
            let got = polynomial_multiply_fft(&a, &b);
            assert_eq!(got.degree(), want.degree(), "same degree");
            let scale = want.c.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
            assert!(max_coeff_diff(&got, &want) < 1e-10 * scale, "FFT product == schoolbook");
        }
        assert!(polynomial_multiply_fft(&Poly::zero(), &Poly::constant(2.0)).is_zero());
        // Degenerate transform lengths (1 and 2) still work.
        let cc = polynomial_multiply_fft(&Poly::constant(3.0), &Poly::constant(-2.0));
        assert!((cc.eval(0.0) + 6.0).abs() < 1e-12, "constant times constant");
        let cl = polynomial_multiply_fft(&Poly::constant(3.0), &Poly::new(vec![1.0, 1.0]));
        assert!(max_coeff_diff(&cl, &Poly::new(vec![3.0, 3.0])) < 1e-12, "constant times linear");
        // A power-of-two-crossing length still lands on the right degree.
        let ones = Poly::new(vec![1.0; 9]);
        let sq = polynomial_multiply_fft(&ones, &ones);
        assert_eq!(sq.degree(), 16);
        assert!((sq.c[8] - 9.0).abs() < 1e-10, "middle coefficient of the square counts pairs");
    }

    #[test]
    fn bernstein_basis_and_conversion() {
        // Partition of unity, and the endpoint interpolation property.
        for n in 0..6 {
            for i in 0..=20 {
                let t = i as f64 / 20.0;
                let s: f64 = (0..=n).map(|k| bernstein_basis(n, k, t)).sum();
                assert!((s - 1.0).abs() < 1e-12, "Bernstein partition of unity at degree {n}");
            }
            assert!((bernstein_basis(n, 0, 0.0) - 1.0).abs() < 1e-15);
            assert!((bernstein_basis(n, n, 1.0) - 1.0).abs() < 1e-15);
        }
        assert_eq!(bernstein_basis(2, 5, 0.5), 0.0, "index past the degree");
        // The control polygon reproduces the polynomial on [a, b].
        let mut rng = Rng::new(37);
        for _ in 0..15 {
            let p = rand_poly(&mut rng, 4);
            let (a, b) = (-1.5, 2.5);
            let w = to_bernstein(&p, a, b);
            assert_eq!(w.len(), p.degree() + 1);
            for i in 0..=20 {
                let t = i as f64 / 20.0;
                let bern: f64 = w.iter().enumerate().map(|(k, &wk)| wk * bernstein_basis(4, k, t)).sum();
                let want = p.eval(a + (b - a) * t);
                assert!((bern - want).abs() < 1e-9 * (1.0 + want.abs()), "Bernstein form at t = {t}");
            }
            // Endpoints are interpolated exactly.
            assert!((w[0] - p.eval(a)).abs() < 1e-9, "first control point");
            assert!((w[4] - p.eval(b)).abs() < 1e-9, "last control point");
        }
    }

    #[test]
    fn newton_identities_give_elementary_symmetric() {
        let roots = [1.0_f64, 2.0, 3.0, 4.0];
        let power_sums: Vec<f64> = (1..=4).map(|k| roots.iter().map(|r| r.powi(k)).sum()).collect();
        let e = newton_identities(&power_sums);
        assert_eq!(e, vec![1.0, 10.0, 35.0, 50.0, 24.0], "e_k of {{1,2,3,4}}");
        // Cross-check against the monic polynomial's coefficients:
        // c_{n-k} = (-1)^k e_k.
        let p = Poly::from_roots(&roots);
        for k in 0..=4 {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert!((p.c[4 - k] - sign * e[k]).abs() < 1e-9, "c_{{n-k}} = (-1)^k e_k at k = {k}");
        }
        // And against brute-force elementary symmetric sums on a second set.
        let r2 = [-1.5_f64, 0.25, 2.0, 3.5, -4.0];
        let ps: Vec<f64> = (1..=5).map(|k| r2.iter().map(|r| r.powi(k)).sum()).collect();
        let e2 = newton_identities(&ps);
        let n = r2.len();
        for k in 0..=n {
            // Sum over all k-subsets, by bitmask.
            let mut direct = 0.0;
            for mask in 0..(1u32 << n) {
                if mask.count_ones() as usize != k {
                    continue;
                }
                let mut prod = 1.0;
                for (i, &r) in r2.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        prod *= r;
                    }
                }
                direct += prod;
            }
            assert!((e2[k] - direct).abs() < 1e-9, "e_{k} = {direct} vs {}", e2[k]);
        }
    }

    #[test]
    fn gcd_and_squarefree_over_f64() {
        // A deliberate repeated factor.
        let p = Poly::from_roots(&[1.0, 1.0, 2.0]);
        assert!(!p.is_squarefree(1e-9), "(x-1)^2 (x-2) is not squarefree");
        let sf = p.squarefree_part(1e-9);
        assert_eq!(sf.degree(), 2, "the repeated factor drops to multiplicity one");
        assert!(max_coeff_diff(&sf, &Poly::from_roots(&[1.0, 2.0])) < 1e-7, "squarefree part");
        let simple = Poly::from_roots(&[1.0, 2.0, 3.0]);
        assert!(simple.is_squarefree(1e-9), "distinct roots");
        assert!(max_coeff_diff(&simple.squarefree_part(1e-9), &simple) < 1e-9, "already squarefree");
        // GCD of two products with a known common factor.
        let g = Poly::from_roots(&[0.5, -3.0]);
        let a = g.mul(&Poly::from_roots(&[2.0]));
        let b = g.mul(&Poly::from_roots(&[7.0]));
        let got = a.gcd(&b, 1e-9);
        assert!(max_coeff_diff(&got, &g) < 1e-7, "gcd recovers the common factor, got {got:?}");
        // Coprime inputs give a constant.
        let cop = Poly::from_roots(&[1.0, 2.0]).gcd(&Poly::from_roots(&[3.0, 4.0]), 1e-9);
        assert_eq!(cop.degree(), 0, "coprime gcd is constant");
        assert!(Poly::zero().gcd(&Poly::zero(), 1e-9).is_zero());
        assert!(!Poly::zero().is_squarefree(1e-9), "zero is not squarefree");
    }

    #[test]
    fn polyq_exact_algebra() {
        let mut rng = Rng::new(41);
        for _ in 0..30 {
            let d_a = 2 + (rng.next_u64() % 5) as usize;
            let a = rand_polyq(&mut rng, d_a);
            let d_b = 1 + (rng.next_u64() % 3) as usize;
            let b = rand_polyq(&mut rng, d_b);
            let (q, r) = a.div_rem(&b).expect("non-zero divisor");
            assert!(r.is_zero() || r.degree() < b.degree(), "remainder degree drops");
            // Exact, not approximate: structural equality of reduced fractions.
            assert_eq!(q.mul(&b).add(&r), a, "a == q*b + r exactly");
            // Pseudo-division identity.
            let (pq, pr) = a.pseudo_div(&b).expect("non-zero divisor");
            let d = if a.degree() >= b.degree() { a.degree() - b.degree() + 1 } else { 0 };
            let mult = b.leading().pow(d as i64);
            assert_eq!(pq.mul(&b).add(&pr), a.mul_scalar(&mult), "lc(b)^(d+1) a == q b + r");
            // Evaluation is a homomorphism, exactly.
            let x = rq((rng.next_u64() % 9) as i64 - 4, 1 + (rng.next_u64() % 3) as i64);
            assert_eq!(a.mul(&b).eval(&x), a.eval(&x).mul(&b.eval(&x)), "eval of product");
            // Calculus, exactly.
            assert_eq!(a.integral(&rq(5, 3)).derivative(), a, "d/dx of integral");
            assert_eq!(a.mul(&b).derivative(), a.derivative().mul(&b).add(&a.mul(&b.derivative())), "product rule");
            // Argument shifts, exactly.
            let h = rq(3, 2);
            assert_eq!(a.shift_arg(&h).shift_arg(&h.neg()), a, "shift is invertible");
            assert_eq!(a.shift_arg(&h).eval(&x), a.eval(&x.add(&h)), "p(x+h)");
            assert_eq!(a.scale_arg(&h).eval(&x), a.eval(&x.mul(&h)), "p(kx)");
            // The f64 shadow agrees.
            assert!(max_coeff_diff(&a.mul(&b).to_poly(), &a.to_poly().mul(&b.to_poly())) < 1e-9);
        }
        assert!(PolyQ::zero().div_rem(&PolyQ::zero()).is_none());
        assert_eq!(PolyQ::from_i64s(&[1, 0, 0]).c.len(), 1, "trailing zeros trimmed");
    }

    #[test]
    fn content_and_primitive_part() {
        // content * primitive_part == the original, and the primitive part
        // is an integer polynomial with unit content.
        let p = PolyQ::new(vec![rq(2, 3), rq(4, 9), rq(-2, 3)]);
        let cont = p.content();
        assert_eq!(cont, rq(-2, 9), "gcd(2,4,2)/lcm(3,9,3), signed by the leading term");
        let prim = p.primitive_part();
        assert_eq!(prim, PolyQ::from_i64s(&[-3, -2, 3]), "primitive part");
        assert_eq!(prim.mul_scalar(&cont), p, "content * primitive == original");
        assert!(!prim.leading().is_negative(), "primitive part has positive leading coefficient");
        assert_eq!(prim.content(), Rational::one(), "the primitive part is primitive");
        let mut rng = Rng::new(43);
        for _ in 0..30 {
            let d_a = 1 + (rng.next_u64() % 5) as usize;
            let a = rand_polyq(&mut rng, d_a);
            let c = a.content();
            let pp = a.primitive_part();
            assert_eq!(pp.mul_scalar(&c), a, "reconstruction");
            assert!(pp.c.iter().all(Rational::is_integer), "primitive part is integral");
            assert_eq!(pp.content(), Rational::one());
        }
        assert!(PolyQ::zero().content().is_zero());
    }

    #[test]
    fn subresultant_gcd_and_squarefree() {
        let g = PolyQ::from_i64s(&[-1, 2, 3]); // 3x^2 + 2x - 1
        let a = g.mul(&PolyQ::from_i64s(&[2, -1, 0, 1])); // times x^3 - x + 2
        let b = g.mul(&PolyQ::from_i64s(&[7, 5, 1])); // times x^2 + 5x + 7
        assert_eq!(a.gcd_exact(&b), g.monic(), "gcd recovers the planted factor");
        assert_eq!(b.gcd_exact(&a), g.monic(), "gcd is symmetric");
        // The cofactors are coprime.
        let one = PolyQ::constant(Rational::one());
        assert_eq!(
            PolyQ::from_i64s(&[2, -1, 0, 1]).gcd_exact(&PolyQ::from_i64s(&[7, 5, 1])),
            one,
            "coprime cofactors"
        );
        assert_eq!(a.gcd_exact(&PolyQ::zero()), a.monic(), "gcd with zero");
        assert!(PolyQ::zero().gcd_exact(&PolyQ::zero()).is_zero());
        // The gcd divides both inputs exactly.
        let d = a.gcd_exact(&b);
        assert!(a.div_rem(&d).expect("non-zero").1.is_zero(), "gcd divides a");
        assert!(b.div_rem(&d).expect("non-zero").1.is_zero(), "gcd divides b");
        // Squarefree detection, exactly.
        let rep = PolyQ::from_roots(&[rq(1, 2), rq(1, 2), rq(-3, 1)]);
        assert!(!rep.is_squarefree(), "planted double root");
        assert_eq!(rep.squarefree_part(), PolyQ::from_roots(&[rq(1, 2), rq(-3, 1)]), "squarefree part");
        let distinct = PolyQ::from_roots(&[rq(1, 2), rq(2, 1), rq(-3, 1)]);
        assert!(distinct.is_squarefree());
        assert_eq!(distinct.squarefree_part(), distinct.monic());
        assert!(!PolyQ::zero().is_squarefree());
        // x^2 + 1 is squarefree even though it has no rational root.
        assert!(PolyQ::from_i64s(&[1, 0, 1]).is_squarefree());
    }

    #[test]
    fn exact_resultant_and_discriminant() {
        // The exact discriminant of a quadratic, and agreement with the f64 path.
        let q = PolyQ::from_i64s(&[-5, 3, 2]); // 2x^2 + 3x - 5
        assert_eq!(q.discriminant(), rq(9 + 40, 1), "b^2 - 4ac");
        assert!((q.to_poly().discriminant() - 49.0).abs() < 1e-9, "f64 path agrees");
        // A shared factor forces a zero resultant, exactly.
        let g = PolyQ::from_i64s(&[1, 1]);
        let a = g.mul(&PolyQ::from_i64s(&[-2, 1]));
        let b = g.mul(&PolyQ::from_i64s(&[3, 1]));
        assert!(a.resultant(&b).is_zero(), "shared factor");
        assert!(!PolyQ::from_i64s(&[-2, 1]).resultant(&PolyQ::from_i64s(&[3, 1])).is_zero());
        // Res(p, x - k) = (-1)^deg(p) p(k), exactly.
        let p = PolyQ::from_i64s(&[2, -3, 0, 1]); // x^3 - 3x + 2, degree 3
        for k in -3..=3 {
            let kk = rq(k, 1);
            let res = p.resultant(&PolyQ::new(vec![kk.neg(), Rational::one()]));
            assert_eq!(res, p.eval(&kk).neg(), "(-1)^3 p(k) at k = {k}");
        }
        // A repeated root zeroes the discriminant exactly.
        assert!(PolyQ::from_roots(&[rq(1, 1), rq(1, 1), rq(4, 1)]).discriminant().is_zero());
    }

    #[test]
    fn cyclotomic_divisor_product_identity() {
        for n in 1..=24_usize {
            let phi = Poly::cyclotomic(n);
            assert_eq!(phi.degree(), totient(n), "deg Phi_{n} = totient({n})");
            assert!(phi.c.iter().all(Rational::is_integer), "Phi_{n} has integer coefficients");
            assert_eq!(phi.leading(), Rational::one(), "Phi_{n} is monic");
            // The defining identity: prod_{d | n} Phi_d = x^n - 1, exactly.
            let mut prod = PolyQ::constant(Rational::one());
            for d in 1..=n {
                if n % d == 0 {
                    prod = prod.mul(&Poly::cyclotomic(d));
                }
            }
            let mut want = vec![Rational::zero(); n + 1];
            want[0] = rq(-1, 1);
            want[n] = Rational::one();
            assert_eq!(prod, PolyQ::new(want), "prod over d|{n} of Phi_d = x^{n} - 1");
        }
        // Phi_p = 1 + x + ... + x^(p-1) for prime p.
        assert_eq!(Poly::cyclotomic(7), PolyQ::from_i64s(&[1, 1, 1, 1, 1, 1, 1]));
        assert_eq!(Poly::cyclotomic(1), PolyQ::from_i64s(&[-1, 1]));
        assert_eq!(Poly::cyclotomic(2), PolyQ::from_i64s(&[1, 1]));
        assert_eq!(Poly::cyclotomic(6), PolyQ::from_i64s(&[1, -1, 1]));
        // The classic surprise: Phi_105 is the first with a coefficient
        // outside {-1, 0, 1}.
        let p105 = Poly::cyclotomic(105);
        assert_eq!(p105.degree(), totient(105));
        assert_eq!(p105.c[7], rq(-2, 1), "Phi_105 has -2 at x^7");
    }

    #[test]
    fn rational_root_factoring() {
        // (2x - 1)^2 (x + 3) (x^2 + 1): rational roots 1/2 (twice) and -3.
        let p = PolyQ::from_i64s(&[-1, 2])
            .mul(&PolyQ::from_i64s(&[-1, 2]))
            .mul(&PolyQ::from_i64s(&[3, 1]))
            .mul(&PolyQ::from_i64s(&[1, 0, 1]));
        let roots = p.factor_rational_roots();
        assert_eq!(roots, vec![(rq(-3, 1), 1), (rq(1, 2), 2)], "rational roots with multiplicity");
        for (r, m) in &roots {
            assert!(p.eval(r).is_zero(), "{r} is a root");
            let lin = PolyQ::new(vec![r.neg(), Rational::one()]);
            let mut q = p.clone();
            for _ in 0..*m {
                let (nq, rem) = q.div_rem(&lin).expect("non-zero");
                assert!(rem.is_zero(), "divides {m} times");
                q = nq;
            }
            assert!(!q.div_rem(&lin).expect("non-zero").1.is_zero(), "and no more");
        }
        // A zero root is reported with its multiplicity.
        let with_zero = p.mul(&PolyQ::from_i64s(&[0, 0, 1]));
        let rz = with_zero.factor_rational_roots();
        assert_eq!(rz, vec![(rq(-3, 1), 1), (Rational::zero(), 2), (rq(1, 2), 2)], "x^2 factor included");
        // No rational roots at all.
        assert!(PolyQ::from_i64s(&[1, 0, 1]).factor_rational_roots().is_empty(), "x^2+1");
        assert!(PolyQ::from_i64s(&[-2, 0, 1]).factor_rational_roots().is_empty(), "x^2-2 (sqrt 2 is irrational)");
        assert!(PolyQ::zero().factor_rational_roots().is_empty());
        // Multiplicities sum to the degree when the polynomial splits.
        let split = PolyQ::from_roots(&[rq(1, 3), rq(1, 3), rq(1, 3), rq(-5, 2)]);
        let rs = split.factor_rational_roots();
        assert_eq!(rs.iter().map(|(_, m)| m).sum::<usize>(), 4, "fully split");
        assert_eq!(rs, vec![(rq(-5, 2), 1), (rq(1, 3), 3)]);
    }

    #[test]
    fn eisenstein_criterion() {
        let two = BigInt::from_i64(2);
        let three = BigInt::from_i64(3);
        // x^3 + 2x + 2 at p = 2: the roadmap's example.
        assert!(PolyQ::from_i64s(&[2, 2, 0, 1]).eisenstein_check(&two), "x^3+2x+2 at p=2");
        // Fails: p does not divide the middle coefficients.
        assert!(!PolyQ::from_i64s(&[1, 1, 0, 1]).eisenstein_check(&two), "x^3+x+1 at p=2");
        // Fails: p^2 divides the constant term.
        assert!(!PolyQ::from_i64s(&[4, 2, 0, 1]).eisenstein_check(&two), "p^2 | a_0");
        // Fails: p divides the leading coefficient.
        assert!(!PolyQ::from_i64s(&[2, 2, 0, 2]).eisenstein_check(&two), "p | a_n");
        // x^3 + 3x + 3 is Eisenstein at 3 but not at 2.
        let e3 = PolyQ::from_i64s(&[3, 3, 0, 1]);
        assert!(e3.eisenstein_check(&three));
        assert!(!e3.eisenstein_check(&two));
        // The check runs on the primitive part, so a common factor of p in
        // every coefficient does not manufacture a false positive:
        // 2x^3 + 2x + 2 is primitively x^3 + x + 1.
        assert!(!PolyQ::from_i64s(&[2, 2, 0, 2]).eisenstein_check(&two));
        // Degenerate inputs.
        assert!(!PolyQ::from_i64s(&[2]).eisenstein_check(&two), "constant");
        assert!(!PolyQ::zero().eisenstein_check(&two));
        assert!(!PolyQ::from_i64s(&[2, 2, 0, 1]).eisenstein_check(&BigInt::one()), "p must exceed 1");
        // Rational coefficients are cleared first: (1/2)(x^3 + 2x + 2).
        let scaled = PolyQ::from_i64s(&[2, 2, 0, 1]).mul_scalar(&rq(1, 2));
        assert!(scaled.eisenstein_check(&two), "scaling does not change irreducibility");
    }
}
