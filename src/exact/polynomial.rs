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
