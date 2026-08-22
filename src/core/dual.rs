//! Forward-mode automatic differentiation with dual numbers.
//!
//! A dual number x = re + ε·eps with ε² = 0 propagates exact first
//! derivatives through arithmetic: f(a + ε·a') = f(a) + ε·f'(a)·a'.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::linalg::Matrix;

/// Dual number: `re` carries the value, `eps` the derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub re: f64,
    pub eps: f64,
}

impl Dual {
    /// A variable seeded for differentiation: x + ε.
    #[must_use]
    pub fn variable(x: f64) -> Self {
        Self { re: x, eps: 1.0 }
    }

    /// A constant: c + 0·ε.
    #[must_use]
    pub fn constant(c: f64) -> Self {
        Self { re: c, eps: 0.0 }
    }

    /// sin(x): derivative cos(x).
    #[must_use]
    pub fn sin(self) -> Self {
        Self { re: self.re.sin(), eps: self.eps * self.re.cos() }
    }

    /// cos(x): derivative −sin(x).
    #[must_use]
    pub fn cos(self) -> Self {
        Self { re: self.re.cos(), eps: -self.eps * self.re.sin() }
    }

    /// tan(x): derivative 1/cos²(x).
    #[must_use]
    pub fn tan(self) -> Self {
        let c = self.re.cos();
        Self { re: self.re.tan(), eps: self.eps / (c * c) }
    }

    /// eˣ: derivative eˣ.
    #[must_use]
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        Self { re: e, eps: self.eps * e }
    }

    /// ln(x): derivative 1/x.
    #[must_use]
    pub fn ln(self) -> Self {
        Self { re: self.re.ln(), eps: self.eps / self.re }
    }

    /// √x: derivative 1/(2√x).
    #[must_use]
    pub fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        Self { re: s, eps: self.eps / (2.0 * s) }
    }

    /// x^p for constant real p: derivative p·x^(p−1).
    #[must_use]
    pub fn powf(self, p: f64) -> Self {
        Self { re: self.re.powf(p), eps: self.eps * p * self.re.powf(p - 1.0) }
    }

    /// xⁿ for integer n: derivative n·xⁿ⁻¹ (exact for polynomials).
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        Self { re: self.re.powi(n), eps: self.eps * n as f64 * self.re.powi(n - 1) }
    }

    /// |x|: derivative sign(x) (undefined at 0; returns 0 there).
    #[must_use]
    pub fn abs(self) -> Self {
        Self { re: self.re.abs(), eps: self.eps * self.re.signum() * if self.re == 0.0 { 0.0 } else { 1.0 } }
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual { re: self.re + rhs.re, eps: self.eps + rhs.eps }
    }
}

impl Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual { re: self.re - rhs.re, eps: self.eps - rhs.eps }
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual { re: self.re * rhs.re, eps: self.re * rhs.eps + self.eps * rhs.re }
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        Dual {
            re: self.re / rhs.re,
            eps: (self.eps * rhs.re - self.re * rhs.eps) / (rhs.re * rhs.re),
        }
    }
}

impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { re: -self.re, eps: -self.eps }
    }
}

/// Exact derivative f'(x) of a scalar function via forward-mode AD.
#[must_use]
pub fn derivative(f: impl Fn(Dual) -> Dual, x: f64) -> f64 {
    f(Dual::variable(x)).eps
}

/// Gradient ∇f(x) of a scalar field, one forward pass per component.
#[must_use]
pub fn gradient(f: impl Fn(&[Dual]) -> Dual, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut args: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
    for (i, g) in grad.iter_mut().enumerate() {
        args[i].eps = 1.0;
        *g = f(&args).eps;
        args[i].eps = 0.0;
    }
    grad
}

/// Jacobian matrix J[i][j] = ∂fᵢ/∂xⱼ of a vector-valued function.
///
/// # Panics
/// Panics if `f` returns an empty vector or `x` is empty.
#[must_use]
pub fn jacobian(f: impl Fn(&[Dual]) -> Vec<Dual>, x: &[f64]) -> Matrix {
    let n = x.len();
    assert!(n > 0, "jacobian requires at least one input");
    let mut args: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
    args[0].eps = 1.0;
    let first = f(&args);
    let m = first.len();
    assert!(m > 0, "jacobian requires at least one output");
    let mut j = Matrix::zeros(m, n);
    for (i, out) in first.iter().enumerate() {
        j.set(i, 0, out.eps);
    }
    args[0].eps = 0.0;
    for col in 1..n {
        args[col].eps = 1.0;
        let out = f(&args);
        assert!(out.len() == m, "jacobian: inconsistent output length");
        for (i, o) in out.iter().enumerate() {
            j.set(i, col, o.eps);
        }
        args[col].eps = 0.0;
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_polynomial_exact() {
        // f(x) = 3x^3 - 2x + 5, f'(x) = 9x^2 - 2
        let f = |x: Dual| Dual::constant(3.0) * x.powi(3) - Dual::constant(2.0) * x
            + Dual::constant(5.0);
        for &x in &[-2.0, 0.0, 1.5, 10.0] {
            assert_eq!(derivative(f, x), 9.0 * x * x - 2.0);
        }
    }

    #[test]
    fn test_transcendentals() {
        assert!(approx(derivative(|x| x.sin(), 1.0), 1.0_f64.cos(), 1e-15));
        assert!(approx(derivative(|x| x.cos(), 1.0), -(1.0_f64.sin()), 1e-15));
        assert!(approx(derivative(|x| x.exp(), 2.0), 2.0_f64.exp(), 1e-12));
        assert!(approx(derivative(|x| x.ln(), 3.0), 1.0 / 3.0, 1e-15));
        assert!(approx(derivative(|x| x.sqrt(), 4.0), 0.25, 1e-15));
        assert!(approx(derivative(|x| x.tan(), 0.5), 1.0 / 0.5_f64.cos().powi(2), 1e-13));
        assert!(approx(derivative(|x| x.powf(2.5), 2.0), 2.5 * 2.0_f64.powf(1.5), 1e-12));
    }

    #[test]
    fn test_quotient_and_chain_rule() {
        // f(x) = sin(x^2) / x, f'(x) = (2x^2 cos(x^2) - sin(x^2)) / x^2
        let f = |x: Dual| (x * x).sin() / x;
        let x = 1.3_f64;
        let expected = (2.0 * x * x * (x * x).cos() - (x * x).sin()) / (x * x);
        assert!(approx(derivative(f, x), expected, 1e-13));
    }

    #[test]
    fn test_abs_and_neg() {
        assert_eq!(derivative(|x| x.abs(), -2.0), -1.0);
        assert_eq!(derivative(|x| x.abs(), 2.0), 1.0);
        assert_eq!(derivative(|x| -x, 5.0), -1.0);
    }

    #[test]
    fn test_gradient() {
        // f(x, y) = x^2 y + y^3 → ∇f = (2xy, x^2 + 3y^2)
        let f = |v: &[Dual]| v[0] * v[0] * v[1] + v[1].powi(3);
        let g = gradient(f, &[2.0, 3.0]);
        assert!(approx(g[0], 12.0, 1e-13));
        assert!(approx(g[1], 31.0, 1e-13));
    }

    #[test]
    fn test_jacobian() {
        // f(x, y) = (x*y, x + y, sin(x)) → J = [[y, x], [1, 1], [cos x, 0]]
        let f = |v: &[Dual]| vec![v[0] * v[1], v[0] + v[1], v[0].sin()];
        let j = jacobian(f, &[1.0, 2.0]);
        assert_eq!(j.rows, 3);
        assert_eq!(j.cols, 2);
        assert!(approx(j.get(0, 0), 2.0, 1e-15));
        assert!(approx(j.get(0, 1), 1.0, 1e-15));
        assert!(approx(j.get(1, 0), 1.0, 1e-15));
        assert!(approx(j.get(1, 1), 1.0, 1e-15));
        assert!(approx(j.get(2, 0), 1.0_f64.cos(), 1e-15));
        assert!(approx(j.get(2, 1), 0.0, 1e-15));
    }
}
