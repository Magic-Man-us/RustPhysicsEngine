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

// ── Numerical Integration ───────────────────────────────────────────

/// Trapezoidal rule for numerical integration of f over [a, b] with n subintervals.
#[must_use]
pub fn trapezoid(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = n.max(1);
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
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

// ── Root Finding ────────────────────────────────────────────────────

/// Bisection method for finding a root of f in [a, b].
/// Returns `None` if f(a) and f(b) have the same sign.
#[must_use]
pub fn bisection(
    f: &dyn Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut fa = f(a);
    let fb = f(b);
    if fa * fb > 0.0 {
        return None;
    }
    for _ in 0..max_iter {
        let mid = (a + b) / 2.0;
        let fm = f(mid);
        if fm.abs() < tol || (b - a) / 2.0 < tol {
            return Some(mid);
        }
        if fa * fm < 0.0 {
            b = mid;
        } else {
            a = mid;
            fa = fm;
        }
    }
    Some((a + b) / 2.0)
}

/// Newton-Raphson method starting from x0.
/// Returns `None` if the derivative is zero or the method does not converge within max_iter.
#[must_use]
pub fn newton_raphson(
    f: &dyn Fn(f64) -> f64,
    df: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let dfx = df(x);
        if dfx.abs() < 1e-15 {
            return None;
        }
        let x_next = x - f(x) / dfx;
        if (x_next - x).abs() < tol {
            return Some(x_next);
        }
        x = x_next;
    }
    None
}

/// Secant method starting from two initial guesses x0 and x1.
/// Returns `None` if the method does not converge within max_iter.
#[must_use]
pub fn secant(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    x1: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut x_prev = x0;
    let mut x_curr = x1;
    let mut f_prev = f(x_prev);
    for _ in 0..max_iter {
        let f_curr = f(x_curr);
        let denom = f_curr - f_prev;
        if denom.abs() < 1e-15 {
            return None;
        }
        let x_next = x_curr - f_curr * (x_curr - x_prev) / denom;
        if (x_next - x_curr).abs() < tol {
            return Some(x_next);
        }
        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_next;
    }
    None
}

// ── ODE Solvers ─────────────────────────────────────────────────────

/// Single forward Euler step: y_next = y + dt * f(t, y).
#[must_use]
pub fn euler_step(f: &dyn Fn(f64, f64) -> f64, t: f64, y: f64, dt: f64) -> f64 {
    y + dt * f(t, y)
}

/// Single step of the classic 4th-order Runge-Kutta method.
#[must_use]
pub fn rk4_step(f: &dyn Fn(f64, f64) -> f64, t: f64, y: f64, dt: f64) -> f64 {
    let k1 = f(t, y);
    let k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1);
    let k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2);
    let k4 = f(t + dt, y + dt * k3);
    y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

/// Full RK4 integration of dy/dt = f(t, y) from t0 to t_end, returning (t, y) pairs.
#[must_use]
pub fn rk4_solve(
    f: &dyn Fn(f64, f64) -> f64,
    t0: f64,
    y0: f64,
    t_end: f64,
    dt: f64,
) -> Vec<(f64, f64)> {
    let mut t = t0;
    let mut y = y0;
    let mut results = vec![(t, y)];
    while t < t_end - dt * 0.5 {
        y = rk4_step(f, t, y, dt);
        t += dt;
        results.push((t, y));
    }
    results
}

/// Single RK4 step for a system of ODEs (vector state).
/// f(t, y) returns a Vec<f64> of derivatives matching the length of y.
#[must_use]
pub fn rk4_step_vec(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    t: f64,
    y: &[f64],
    dt: f64,
) -> Vec<f64> {
    let n = y.len();
    let k1 = f(t, y);

    let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k1[i]).collect();
    let k2 = f(t + 0.5 * dt, &y2);

    let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k2[i]).collect();
    let k3 = f(t + 0.5 * dt, &y3);

    let y4: Vec<f64> = (0..n).map(|i| y[i] + dt * k3[i]).collect();
    let k4 = f(t + dt, &y4);

    (0..n)
        .map(|i| y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

// ── Interpolation ───────────────────────────────────────────────────

/// Linear interpolation between a and b: a + t*(b - a).
#[must_use]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Piecewise linear interpolation in sorted (x_data, y_data).
/// Clamps to the endpoint values if x is outside the data range.
/// Panics if data slices are empty or mismatched in length.
#[must_use]
pub fn linear_interp(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    assert!(
        !x_data.is_empty() && x_data.len() == y_data.len(),
        "x_data and y_data must be non-empty and equal length"
    );
    let n = x_data.len();
    if n == 1 || x <= x_data[0] {
        return y_data[0];
    }
    if x >= x_data[n - 1] {
        return y_data[n - 1];
    }
    // Binary search for the interval
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x_data[mid] > x {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let t = (x - x_data[lo]) / (x_data[hi] - x_data[lo]);
    lerp(y_data[lo], y_data[hi], t)
}

/// Natural cubic spline interpolation for a single query point.
/// Falls back to linear interpolation if fewer than 4 data points.
/// Panics if data slices are empty or mismatched in length.
#[must_use]
pub fn cubic_interp(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    assert!(
        !x_data.is_empty() && x_data.len() == y_data.len(),
        "x_data and y_data must be non-empty and equal length"
    );
    let n = x_data.len();
    if n < 4 {
        return linear_interp(x_data, y_data, x);
    }

    // Build the tridiagonal system for natural cubic spline second derivatives (M).
    // Natural boundary: M[0] = M[n-1] = 0.
    let segments = n - 1;
    let mut h = vec![0.0; segments];
    for i in 0..segments {
        h[i] = x_data[i + 1] - x_data[i];
    }

    // Interior equations: h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*divided_diff
    // With M[0]=0 and M[n-1]=0, we solve for M[1..n-2] using Thomas algorithm.
    let interior = n - 2; // number of interior unknowns (always >= 2 since n >= 4)

    let mut diag = vec![0.0; interior];
    let mut upper = vec![0.0; interior];
    let mut lower = vec![0.0; interior];
    let mut rhs = vec![0.0; interior];

    for i in 0..interior {
        let idx = i + 1; // index into full arrays
        diag[i] = 2.0 * (h[idx - 1] + h[idx]);
        rhs[i] = 6.0
            * ((y_data[idx + 1] - y_data[idx]) / h[idx]
                - (y_data[idx] - y_data[idx - 1]) / h[idx - 1]);
        if i > 0 {
            lower[i] = h[idx - 1];
        }
        if i + 1 < interior {
            upper[i] = h[idx];
        }
    }

    // Thomas algorithm (tridiagonal solve)
    for i in 1..interior {
        let factor = lower[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    let mut m_interior = vec![0.0; interior];
    m_interior[interior - 1] = rhs[interior - 1] / diag[interior - 1];
    for i in (0..interior - 1).rev() {
        m_interior[i] = (rhs[i] - upper[i] * m_interior[i + 1]) / diag[i];
    }

    // Full M array with natural boundary conditions
    let mut m = vec![0.0; n];
    for i in 0..interior {
        m[i + 1] = m_interior[i];
    }

    // Find the segment containing x (clamp to endpoints)
    let x_clamped = x.clamp(x_data[0], x_data[n - 1]);
    let mut seg = 0;
    for i in 0..segments {
        if x_clamped <= x_data[i + 1] {
            seg = i;
            break;
        }
        seg = i;
    }

    // Evaluate the cubic polynomial on segment seg
    let dx_right = x_data[seg + 1] - x_clamped;
    let dx_left = x_clamped - x_data[seg];
    let hi = h[seg];

    m[seg] * dx_right.powi(3) / (6.0 * hi)
        + m[seg + 1] * dx_left.powi(3) / (6.0 * hi)
        + (y_data[seg] / hi - m[seg] * hi / 6.0) * dx_right
        + (y_data[seg + 1] / hi - m[seg + 1] * hi / 6.0) * dx_left
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    const INTEGRATION_TOL: f64 = 1e-6;
    const ROOT_TOL: f64 = 1e-10;
    const ODE_TOL: f64 = 1e-4;
    const MAX_ITER: usize = 1000;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── Integration tests: ∫sin(x)dx from 0 to π = 2.0 ────────────

    #[test]
    fn test_trapezoid_sin() {
        let result = trapezoid(&f64::sin, 0.0, PI, 10_000);
        assert!(
            approx_eq(result, 2.0, INTEGRATION_TOL),
            "trapezoid: expected 2.0, got {result}"
        );
    }

    #[test]
    fn test_simpson_sin() {
        let result = simpson(&f64::sin, 0.0, PI, 1_000);
        assert!(
            approx_eq(result, 2.0, INTEGRATION_TOL),
            "simpson: expected 2.0, got {result}"
        );
    }

    #[test]
    fn test_simpson_clamps_odd_n() {
        let result = simpson(&f64::sin, 0.0, PI, 999);
        assert!(
            approx_eq(result, 2.0, INTEGRATION_TOL),
            "simpson with odd n: expected 2.0, got {result}"
        );
    }

    #[test]
    fn test_gaussian_quadrature_sin() {
        let result = gaussian_quadrature_5(&f64::sin, 0.0, PI);
        assert!(
            approx_eq(result, 2.0, 1e-5),
            "gauss5: expected 2.0, got {result}"
        );
    }

    // ── Root finding tests: x² - 2 = 0, root = √2 ─────────────────

    #[test]
    fn test_bisection_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = bisection(&f, 1.0, 2.0, ROOT_TOL, MAX_ITER).unwrap();
        assert!(
            approx_eq(root, std::f64::consts::SQRT_2, ROOT_TOL),
            "bisection: expected √2, got {root}"
        );
    }

    #[test]
    fn test_bisection_same_sign_returns_none() {
        let f = |x: f64| x * x + 1.0;
        assert!(bisection(&f, 0.0, 1.0, ROOT_TOL, MAX_ITER).is_none());
    }

    #[test]
    fn test_newton_raphson_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let root = newton_raphson(&f, &df, 1.5, ROOT_TOL, MAX_ITER).unwrap();
        assert!(
            approx_eq(root, std::f64::consts::SQRT_2, ROOT_TOL),
            "newton: expected √2, got {root}"
        );
    }

    #[test]
    fn test_newton_raphson_zero_derivative() {
        assert!(newton_raphson(&|_: f64| 1.0, &|_: f64| 0.0, 1.0, ROOT_TOL, 10).is_none());
    }

    #[test]
    fn test_secant_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = secant(&f, 1.0, 2.0, ROOT_TOL, MAX_ITER).unwrap();
        assert!(
            approx_eq(root, std::f64::consts::SQRT_2, ROOT_TOL),
            "secant: expected √2, got {root}"
        );
    }

    // ── ODE tests: dy/dt = -y, y(0) = 1 → y(t) = e^(-t) ──────────

    #[test]
    fn test_euler_step() {
        let f = |_t: f64, y: f64| -y;
        let y1 = euler_step(&f, 0.0, 1.0, 0.01);
        assert!(
            approx_eq(y1, 0.99, 1e-12),
            "euler step: expected 0.99, got {y1}"
        );
    }

    #[test]
    fn test_rk4_step() {
        let f = |_t: f64, y: f64| -y;
        let dt = 0.1;
        let y1 = rk4_step(&f, 0.0, 1.0, dt);
        let expected = 0.9048374180359595;
        assert!(
            approx_eq(y1, expected, 1e-6),
            "rk4 step: expected {expected}, got {y1}"
        );
    }

    #[test]
    fn test_rk4_solve_exponential_decay() {
        let f = |_t: f64, y: f64| -y;
        let dt = 0.01;
        let results = rk4_solve(&f, 0.0, 1.0, 5.0, dt);
        for &(t, y) in &results {
            let exact = (-t).exp();
            assert!(
                approx_eq(y, exact, ODE_TOL),
                "rk4 solve at t={t}: expected {exact}, got {y}"
            );
        }
    }

    #[test]
    fn test_rk4_step_vec_system() {
        // Coupled system: harmonic oscillator x'' = -x
        // State: [x, v], derivatives: [v, -x]
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let y0 = vec![1.0, 0.0]; // x=1, v=0 → x(t) = cos(t)
        let dt = 0.01;
        let mut t = 0.0;
        let mut y = y0;
        let steps = 100;
        for _ in 0..steps {
            y = rk4_step_vec(&f, t, &y, dt);
            t += dt;
        }
        let exact_x = t.cos();
        let got_x = y[0];
        assert!(
            approx_eq(got_x, exact_x, 1e-6),
            "rk4 vec at t={t}: expected x={exact_x}, got {got_x}",
        );
    }

    // ── Interpolation tests ────────────────────────────────────────

    #[test]
    fn test_lerp() {
        assert!(approx_eq(lerp(0.0, 10.0, 0.5), 5.0, 1e-15));
        assert!(approx_eq(lerp(0.0, 10.0, 0.0), 0.0, 1e-15));
        assert!(approx_eq(lerp(0.0, 10.0, 1.0), 10.0, 1e-15));
    }

    #[test]
    fn test_linear_interp_exact_points() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 2.0, 4.0, 6.0];
        assert!(approx_eq(linear_interp(&xs, &ys, 1.5), 3.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 0.0), 0.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 3.0), 6.0, 1e-12));
    }

    #[test]
    fn test_linear_interp_clamping() {
        let xs = [1.0, 2.0, 3.0];
        let ys = [10.0, 20.0, 30.0];
        assert!(approx_eq(linear_interp(&xs, &ys, 0.0), 10.0, 1e-12));
        assert!(approx_eq(linear_interp(&xs, &ys, 5.0), 30.0, 1e-12));
    }

    #[test]
    fn test_cubic_interp_quadratic() {
        // y = x^2 sampled at 5 points; natural cubic spline is close but not exact
        let xs: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        for (&x, &expected) in [0.5, 1.5, 2.5, 3.5].iter().zip(&[0.25, 2.25, 6.25, 12.25]) {
            let got = cubic_interp(&xs, &ys, x);
            assert!(
                approx_eq(got, expected, 0.5),
                "cubic at x={x}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_cubic_interp_two_points_falls_back() {
        let xs = [0.0, 1.0];
        let ys = [0.0, 1.0];
        let got = cubic_interp(&xs, &ys, 0.5);
        let expected = 0.5;
        assert!(
            approx_eq(got, expected, 1e-12),
            "cubic 2-point fallback: expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_cubic_interp_few_points_falls_back() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 4.0];
        let got = cubic_interp(&xs, &ys, 0.5);
        let expected = 0.5;
        assert!(
            approx_eq(got, expected, 1e-12),
            "cubic fallback: expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_integration_polynomial() {
        // ∫(x^2)dx from 0 to 3 = 9
        let f = |x: f64| x * x;
        let trap = trapezoid(&f, 0.0, 3.0, 10_000);
        let simp = simpson(&f, 0.0, 3.0, 100);
        let gauss = gaussian_quadrature_5(&f, 0.0, 3.0);
        assert!(approx_eq(trap, 9.0, 1e-4), "trap x^2: got {trap}");
        assert!(approx_eq(simp, 9.0, 1e-10), "simpson x^2: got {simp}");
        assert!(approx_eq(gauss, 9.0, 1e-10), "gauss x^2: got {gauss}");
    }

    #[test]
    fn test_bisection_max_iter_reached() {
        let f = |x: f64| x.sin();
        let result = bisection(&f, 2.5, 3.8, 1e-20, 3);
        assert!(result.is_some());
    }

    #[test]
    fn test_newton_raphson_no_convergence() {
        // f(x) = x^2 + 1 has no real roots; Newton's method oscillates
        let f = |x: f64| x * x + 1.0;
        let df = |x: f64| 2.0 * x;
        let result = newton_raphson(&f, &df, 1.0, 1e-12, 100);
        assert!(result.is_none());
    }

    #[test]
    fn test_newton_raphson_max_iter() {
        let f = |x: f64| x.sin();
        let df = |x: f64| x.cos();
        let result = newton_raphson(&f, &df, 1.0, 1e-100, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_secant_no_convergence() {
        let f = |x: f64| x * x + 1.0;
        let result = secant(&f, 0.0, 0.0, 1e-12, 100);
        assert!(result.is_none());
    }

    #[test]
    fn test_secant_max_iter() {
        let f = |x: f64| x.sin();
        let result = secant(&f, 2.0, 3.5, 1e-100, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_cubic_spline_two_points() {
        let x_data = vec![0.0, 1.0];
        let y_data = vec![0.0, 2.0];
        let y = cubic_interp(&x_data, &y_data, 0.5);
        assert!(approx_eq(y, 1.0, 1e-12));
    }

    #[test]
    fn test_simpson_n_less_than_2() {
        let result = simpson(&|x: f64| x * x, 0.0, 1.0, 1);
        // n=1 gets rounded up to n=2; Simpson's rule is exact for polynomials up to degree 3
        assert!(approx_eq(result, 1.0 / 3.0, 1e-12));
    }
}
