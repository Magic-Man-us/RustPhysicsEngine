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

    /// arctan(x): derivative 1/(1+x²).
    #[must_use]
    pub fn atan(self) -> Self {
        Self { re: self.re.atan(), eps: self.eps / (1.0 + self.re * self.re) }
    }

    /// sinh(x): derivative cosh(x).
    #[must_use]
    pub fn sinh(self) -> Self {
        Self { re: self.re.sinh(), eps: self.eps * self.re.cosh() }
    }

    /// cosh(x): derivative sinh(x).
    #[must_use]
    pub fn cosh(self) -> Self {
        Self { re: self.re.cosh(), eps: self.eps * self.re.sinh() }
    }

    /// tanh(x): derivative 1/cosh²(x).
    #[must_use]
    pub fn tanh(self) -> Self {
        let c = self.re.cosh();
        Self { re: self.re.tanh(), eps: self.eps / (c * c) }
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
    #[test]
    fn test_hyperbolic_and_atan_derivatives() {
        // Each new rule against its closed form, plus the defining
        // identities cosh^2 - sinh^2 = 1 and tanh = sinh/cosh.
        for &x in &[-1.3_f64, -0.4, 0.0, 0.25, 1.7] {
            let d = Dual::variable(x);
            assert!((d.atan().eps - 1.0 / (1.0 + x * x)).abs() < 1e-12);
            assert!((d.sinh().eps - x.cosh()).abs() < 1e-12);
            assert!((d.cosh().eps - x.sinh()).abs() < 1e-12);
            assert!((d.tanh().eps - 1.0 / (x.cosh() * x.cosh())).abs() < 1e-12);
            let (sh, ch) = (d.sinh(), d.cosh());
            assert!((ch.re * ch.re - sh.re * sh.re - 1.0).abs() < 1e-12);
            assert!((d.tanh().re - sh.re / ch.re).abs() < 1e-12);
            // atan and tan invert one another, derivatives included.
            assert!((d.atan().re.tan() - x).abs() < 1e-12);
        }
    }

    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // Miri evaluates the float intrinsics with its own implementations, which
    // are allowed to differ from the host's in the last bits and which Miri
    // deliberately randomises within that slack. This test asserts an exact
    // value, so it fails under Miri for that reason and not because anything
    // is wrong; it still runs normally everywhere else.
    #[cfg_attr(miri, ignore = "Miri's float intrinsics are not bit-exact")]
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

    // Miri evaluates the float intrinsics with its own implementations, which
    // are allowed to differ from the host's in the last bits and which Miri
    // deliberately randomises within that slack. This test asserts an exact
    // value, so it fails under Miri for that reason and not because anything
    // is wrong; it still runs normally everywhere else.
    #[cfg_attr(miri, ignore = "Miri's float intrinsics are not bit-exact")]
    #[test]
    fn test_derivative_exact_on_polynomials() {
        // Dual arithmetic is exact on polynomials: no truncation error,
        // so equality (not a tolerance) is the right assertion.
        // p(x) = x^4 - 5x^2 + 7x - 3, p'(x) = 4x^3 - 10x + 7.
        let p = |x: Dual| {
            x.powi(4) - Dual::constant(5.0) * x.powi(2) + Dual::constant(7.0) * x
                - Dual::constant(3.0)
        };
        for &x in &[-3.0_f64, -1.0, 0.0, 0.5, 2.0, 6.0] {
            assert_eq!(derivative(p, x), 4.0 * x * x * x - 10.0 * x + 7.0, "p' at {x}");
        }
        // A constant function has zero derivative everywhere, exactly.
        assert_eq!(derivative(|_| Dual::constant(11.5), 4.25), 0.0);
        // The identity has derivative 1 everywhere, exactly.
        assert_eq!(derivative(|x| x, -8.75), 1.0);
        // Product rule on (x+1)(x+2) = x^2 + 3x + 2 -> 2x + 3.
        let q = |x: Dual| (x + Dual::constant(1.0)) * (x + Dual::constant(2.0));
        for &x in &[-2.0_f64, 0.0, 3.0] {
            assert_eq!(derivative(q, x), 2.0 * x + 3.0);
        }
    }

    #[test]
    fn test_gradient_of_quadratic_form_matches_analytic() {
        // f(x) = ½ xᵀ A x + bᵀ x with A symmetric: ∇f = A x + b.
        let a = [[2.0, -1.0, 0.5], [-1.0, 4.0, 1.5], [0.5, 1.5, 3.0]];
        let b = [0.7, -2.0, 1.25];
        let f = |v: &[Dual]| {
            let mut acc = Dual::constant(0.0);
            for (i, ai) in a.iter().enumerate() {
                for (j, &aij) in ai.iter().enumerate() {
                    acc = acc + Dual::constant(0.5 * aij) * v[i] * v[j];
                }
                acc = acc + Dual::constant(b[i]) * v[i];
            }
            acc
        };
        for x in [[1.0, 2.0, -1.0], [0.0, 0.0, 0.0], [-3.5, 0.25, 4.0]] {
            let g = gradient(f, &x);
            for i in 0..3 {
                let expected: f64 =
                    (0..3).map(|j| a[i][j] * x[j]).sum::<f64>() + b[i];
                assert!((g[i] - expected).abs() < 1e-12, "grad[{i}] {} vs {expected}", g[i]);
            }
        }
        // A linear functional has a constant gradient equal to its
        // coefficient vector, independent of the evaluation point.
        let lin = |v: &[Dual]| Dual::constant(3.0) * v[0] - Dual::constant(0.5) * v[1];
        assert_eq!(gradient(lin, &[0.0, 0.0]), vec![3.0, -0.5]);
        assert_eq!(gradient(lin, &[100.0, -7.0]), vec![3.0, -0.5]);
    }

    #[test]
    fn test_jacobian_of_linear_map_is_the_matrix() {
        // f(x) = M x has Jacobian M exactly, at every point.
        let m = [[1.0, -2.0, 3.0], [0.0, 4.0, -0.5]];
        let f = |v: &[Dual]| {
            m.iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .fold(Dual::constant(0.0), |acc, (j, &c)| {
                            acc + Dual::constant(c) * v[j]
                        })
                })
                .collect::<Vec<Dual>>()
        };
        for x in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-2.5, 9.0, 0.125]] {
            let j = jacobian(f, &x);
            assert_eq!(j.rows, 2);
            assert_eq!(j.cols, 3);
            for r in 0..2 {
                for c in 0..3 {
                    assert_eq!(j.get(r, c), m[r][c], "J[{r}][{c}] at {x:?}");
                }
            }
        }
        // Jacobian of the identity map is the identity matrix (exact).
        let id = |v: &[Dual]| v.to_vec();
        let j = jacobian(id, &[3.0, -1.0, 0.5, 2.0]);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(j.get(r, c), f64::from(u8::from(r == c)));
            }
        }
    }

    // Miri evaluates the float intrinsics with its own implementations, which
    // are allowed to differ from the host's in the last bits and which Miri
    // deliberately randomises within that slack. This test asserts an exact
    // value, so it fails under Miri for that reason and not because anything
    // is wrong; it still runs normally everywhere else.
    #[cfg_attr(miri, ignore = "Miri's float intrinsics are not bit-exact")]
    #[test]
    fn test_gradient_and_jacobian_agree_on_a_scalar_field() {
        // The Jacobian of a 1-output function is the gradient row.
        let f = |v: &[Dual]| v[0].sin() * v[1].exp() + v[2].powi(3);
        let x = [0.7, -0.4, 1.3];
        let g = gradient(f, &x);
        let j = jacobian(|v| vec![f(v)], &x);
        assert_eq!(j.rows, 1);
        for (c, &gi) in g.iter().enumerate() {
            assert!((j.get(0, c) - gi).abs() < 1e-15);
        }
        // Cross-check against analytic partials.
        assert!(approx(g[0], x[0].cos() * x[1].exp(), 1e-13));
        assert!(approx(g[1], x[0].sin() * x[1].exp(), 1e-13));
        assert!(approx(g[2], 3.0 * x[2] * x[2], 1e-13));
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
