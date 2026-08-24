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
    let n = if n < 2 { 2 } else if !n.is_multiple_of(2) { n + 1 } else { n };
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

// ── Gauss-Kronrod and adaptive quadrature ───────────────────────────

use crate::error::SolveError;

/// Result of an error-estimating quadrature: the integral estimate, an
/// upper bound on its error, and the number of function evaluations.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadResult {
    pub value: f64,
    pub error: f64,
    pub evals: usize,
}

// Kronrod-15 abscissae (positive half; symmetric about 0).
const GK15_NODES: [f64; 8] = [
    0.991_455_371_120_813,
    0.949_107_912_342_759,
    0.864_864_423_359_769,
    0.741_531_185_599_394,
    0.586_087_235_467_691,
    0.405_845_151_377_397,
    0.207_784_955_007_898,
    0.0,
];
// Kronrod-15 weights matching GK15_NODES.
const GK15_WEIGHTS: [f64; 8] = [
    0.022_935_322_010_529,
    0.063_092_092_629_979,
    0.104_790_010_322_250,
    0.140_653_259_715_525,
    0.169_004_726_639_267,
    0.190_350_578_064_785,
    0.204_432_940_075_298,
    0.209_482_141_084_728,
];
// Embedded Gauss-7 weights (nodes 1, 3, 5, 7 of GK15_NODES).
const G7_WEIGHTS: [f64; 4] = [
    0.129_484_966_168_870,
    0.279_705_391_489_277,
    0.381_830_050_505_119,
    0.417_959_183_673_469,
];

/// 15-point Gauss-Kronrod quadrature of f over [a, b].
///
/// `value` is the K15 estimate; `error` is |K15 − G7|, the classical
/// (conservative) error bound from the embedded 7-point Gauss rule.
#[must_use]
pub fn gauss_kronrod_15(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> QuadResult {
    let half = (b - a) / 2.0;
    let mid = (a + b) / 2.0;
    let mut k15 = 0.0;
    let mut g7 = 0.0;
    for (i, (&x, &wk)) in GK15_NODES.iter().zip(GK15_WEIGHTS.iter()).enumerate() {
        let (fp, fm) = if x == 0.0 {
            let v = f(mid);
            (v, 0.0) // center node counted once
        } else {
            (f(mid + half * x), f(mid - half * x))
        };
        k15 += wk * (fp + fm);
        // Gauss-7 nodes sit at the odd Kronrod indices (center included
        // at i = 7, where fm is 0 so fp + fm counts it once).
        if i % 2 == 1 {
            g7 += G7_WEIGHTS[i / 2] * (fp + fm);
        }
    }
    QuadResult {
        value: k15 * half,
        error: ((k15 - g7) * half).abs(),
        evals: 15,
    }
}

fn adaptive_quad_rec(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    depth: usize,
    evals: &mut usize,
) -> Result<(f64, f64), SolveError> {
    let r = gauss_kronrod_15(f, a, b);
    *evals += r.evals;
    if r.error <= tol || !r.value.is_finite() {
        return Ok((r.value, r.error));
    }
    if depth == 0 {
        return Err(SolveError::NoConvergence { iters: *evals, residual: r.error });
    }
    let mid = (a + b) / 2.0;
    let (v1, e1) = adaptive_quad_rec(f, a, mid, tol / 2.0, depth - 1, evals)?;
    let (v2, e2) = adaptive_quad_rec(f, mid, b, tol / 2.0, depth - 1, evals)?;
    Ok((v1 + v2, e1 + e2))
}

/// Adaptive quadrature: recursive bisection with the GK15 rule until
/// each panel's error estimate is below its share of `tol`.
pub fn adaptive_quad(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> Result<QuadResult, SolveError> {
    if !(tol > 0.0) {
        return Err(SolveError::InvalidArgument("adaptive_quad requires tol > 0"));
    }
    let mut evals = 0usize;
    let (value, error) = adaptive_quad_rec(f, a, b, tol, max_depth, &mut evals)?;
    Ok(QuadResult { value, error, evals })
}

/// Romberg integration: trapezoid estimates at h, h/2, h/4, … with
/// Richardson extrapolation across the levels (NR §4.3). Converges when
/// two successive diagonal entries agree within `tol`.
pub fn romberg(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    max_levels: usize,
    tol: f64,
) -> Result<QuadResult, SolveError> {
    if max_levels == 0 || !(tol > 0.0) {
        return Err(SolveError::InvalidArgument("romberg requires max_levels > 0 and tol > 0"));
    }
    let mut r: Vec<Vec<f64>> = Vec::with_capacity(max_levels);
    let mut evals = 0usize;
    let mut h = b - a;
    let trap0 = 0.5 * h * (f(a) + f(b));
    evals += 2;
    r.push(vec![trap0]);
    for k in 1..max_levels {
        // Refine the trapezoid estimate with the midpoints of the
        // current panels: T_k = T_{k-1}/2 + (h/2)·Σ f(midpoints).
        let points = 1usize << (k - 1);
        let mut sum = 0.0;
        for i in 0..points {
            sum += f(a + h * (i as f64 + 0.5));
        }
        evals += points;
        h /= 2.0;
        let trap = 0.5 * r[k - 1][0] + h * sum;
        let mut row = vec![trap];
        for j in 1..=k {
            let factor = 4.0_f64.powi(j as i32);
            let val = (factor * row[j - 1] - r[k - 1][j - 1]) / (factor - 1.0);
            row.push(val);
        }
        let prev_best = *r[k - 1].last().unwrap();
        let best = *row.last().unwrap();
        r.push(row);
        if (best - prev_best).abs() <= tol {
            return Ok(QuadResult { value: best, error: (best - prev_best).abs(), evals });
        }
    }
    let best = *r.last().unwrap().last().unwrap();
    let prev = *r[r.len() - 2].last().unwrap();
    Err(SolveError::NoConvergence { iters: evals, residual: (best - prev).abs() })
}

/// One generalized Richardson extrapolation pass over estimates whose
/// step sizes shrink by `ratio` between entries and whose error expands
/// in powers of h^order (E = c₁·h^p + c₂·h^{2p} + …). Returns the
/// highest-order extrapolant.
///
/// # Panics
/// Panics if `estimates` is empty, `ratio <= 1`, or `order == 0`.
#[must_use]
pub fn richardson_extrapolate(estimates: &[f64], ratio: f64, order: u32) -> f64 {
    assert!(!estimates.is_empty(), "richardson_extrapolate requires estimates");
    assert!(ratio > 1.0, "richardson_extrapolate requires ratio > 1");
    assert!(order > 0, "richardson_extrapolate requires order > 0");
    let mut table = estimates.to_vec();
    let n = table.len();
    for level in 1..n {
        let factor = ratio.powi((order * level as u32) as i32);
        for i in (level..n).rev() {
            table[i] = (factor * table[i] - table[i - 1]) / (factor - 1.0);
        }
    }
    table[n - 1]
}

/// Integral of f over (−∞, ∞) via the substitution x = t/(1−t²),
/// dx = (1+t²)/(1−t²)² dt, mapped onto t ∈ (−1, 1) and evaluated with
/// [`adaptive_quad`]. Requires f to decay at infinity.
pub fn integrate_infinite(
    f: &dyn Fn(f64) -> f64,
    tol: f64,
) -> Result<QuadResult, SolveError> {
    let g = move |t: f64| {
        let one_minus = 1.0 - t * t;
        if one_minus <= 0.0 {
            return 0.0;
        }
        let x = t / one_minus;
        let jac = (1.0 + t * t) / (one_minus * one_minus);
        let v = f(x) * jac;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    };
    adaptive_quad(&g, -1.0, 1.0, tol, 60)
}
