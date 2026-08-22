//! Numerical integration (quadrature) rules.

use crate::core::compensated::sum_neumaier;

// 5-point Gauss-Legendre quadrature nodes on [-1, 1]
const GL5_NODES: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683,
    0.0,
    0.538_469_310_105_683,
    0.906_179_845_938_664,
];

// 5-point Gauss-Legendre quadrature weights
const GL5_WEIGHTS: [f64; 5] = [
    0.236_926_885_056_189_1,
    0.478_628_670_499_366_5,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_5,
    0.236_926_885_056_189_1,
];

/// Trapezoidal rule for numerical integration of f over [a, b] with n subintervals.
/// Sample values are accumulated with Neumaier compensated summation.
#[must_use]
pub fn trapezoid(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = n.max(1);
    let h = (b - a) / n as f64;
    let mut vals = Vec::with_capacity(n);
    vals.push(0.5 * (f(a) + f(b)));
    for i in 1..n {
        vals.push(f(a + i as f64 * h));
    }
    sum_neumaier(&vals) * h
}

/// Simpson's 1/3 rule for numerical integration of f over [a, b] with n subintervals.
/// If n is odd it is rounded up to the next even number.
#[must_use]
pub fn simpson(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n < 2 { 2 } else if n % 2 != 0 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coeff * f(a + i as f64 * h);
    }
    sum * h / 3.0
}

/// 5-point Gauss-Legendre quadrature of f over [a, b].
#[must_use]
pub fn gaussian_quadrature_5(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let half_width = (b - a) / 2.0;
    let midpoint = (a + b) / 2.0;
    let mut sum = 0.0;
    for i in 0..5 {
        let x = midpoint + half_width * GL5_NODES[i];
        sum += GL5_WEIGHTS[i] * f(x);
    }
    sum * half_width
}
