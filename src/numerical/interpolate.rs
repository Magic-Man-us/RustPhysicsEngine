//! Interpolation routines.

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
    m[1..=interior].copy_from_slice(&m_interior[..interior]);

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

// ---------------------------------------------------------------------------
// Cubic splines, Catmull-Rom, B-splines, and Bezier evaluation
// ---------------------------------------------------------------------------

use crate::error::SolveError;
use crate::linalg::thomas_solve;
use crate::math::Vec3;

/// Piecewise cubic spline S_i(t) = a_i + b_i·Δ + c_i·Δ² + d_i·Δ³ with
/// Δ = t − x_i on segment i (Burden & Faires, *Numerical Analysis*,
/// §3.5). Built with the Thomas tridiagonal solve; C² across knots.
#[derive(Debug, Clone, PartialEq)]
pub struct CubicSpline {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline {
    fn build(
        x: &[f64],
        y: &[f64],
        clamped: Option<(f64, f64)>,
    ) -> Result<Self, SolveError> {
        if x.len() != y.len() {
            return Err(SolveError::DimensionMismatch { expected: x.len(), got: y.len() });
        }
        let n = x.len();
        if n < 3 {
            return Err(SolveError::InvalidArgument("CubicSpline requires at least 3 points"));
        }
        for w in x.windows(2) {
            if w[1] <= w[0] {
                return Err(SolveError::InvalidArgument(
                    "CubicSpline requires strictly increasing x",
                ));
            }
        }
        let segs = n - 1;
        let h: Vec<f64> = (0..segs).map(|i| x[i + 1] - x[i]).collect();
        let a = y.to_vec();

        // Tridiagonal system for the c coefficients (n unknowns).
        let mut sub = vec![0.0; n - 1];
        let mut diag = vec![0.0; n];
        let mut sup = vec![0.0; n - 1];
        let mut rhs = vec![0.0; n];
        match clamped {
            None => {
                // Natural: c_0 = c_{n-1} = 0.
                diag[0] = 1.0;
                diag[n - 1] = 1.0;
            }
            Some((dy0, dyn_)) => {
                diag[0] = 2.0 * h[0];
                sup[0] = h[0];
                rhs[0] = 3.0 * ((a[1] - a[0]) / h[0] - dy0);
                diag[n - 1] = 2.0 * h[segs - 1];
                sub[n - 2] = h[segs - 1];
                rhs[n - 1] = 3.0 * (dyn_ - (a[n - 1] - a[n - 2]) / h[segs - 1]);
            }
        }
        for i in 1..n - 1 {
            sub[i - 1] = h[i - 1];
            diag[i] = 2.0 * (h[i - 1] + h[i]);
            sup[i] = h[i];
            rhs[i] = 3.0 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1]);
        }
        // Natural boundary rows must not couple: zero their neighbors.
        if clamped.is_none() {
            sup[0] = 0.0;
            sub[n - 2] = 0.0;
        }
        let c = thomas_solve(&sub, &diag, &sup, &rhs)?;

        let mut b = vec![0.0; segs];
        let mut d = vec![0.0; segs];
        for i in 0..segs {
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }
        Ok(Self { x: x.to_vec(), a, b, c, d })
    }

    /// Natural spline: zero second derivative at both ends.
    pub fn natural(x: &[f64], y: &[f64]) -> Result<Self, SolveError> {
        Self::build(x, y, None)
    }

    /// Clamped spline with prescribed end slopes dy0 = S'(x₀) and
    /// dyn = S'(xₙ).
    pub fn clamped(x: &[f64], y: &[f64], dy0: f64, dyn_: f64) -> Result<Self, SolveError> {
        Self::build(x, y, Some((dy0, dyn_)))
    }

    /// Index of the segment containing t (end segments extrapolate).
    fn segment(&self, t: f64) -> usize {
        let n = self.x.len();
        if t <= self.x[0] {
            return 0;
        }
        if t >= self.x[n - 1] {
            return n - 2;
        }
        // Binary search for the last knot <= t.
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] <= t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Spline value S(t); outside the knot range the end polynomials
    /// extrapolate.
    #[must_use]
    pub fn eval(&self, t: f64) -> f64 {
        let i = self.segment(t);
        let dt = t - self.x[i];
        self.a[i] + dt * (self.b[i] + dt * (self.c[i] + dt * self.d[i]))
    }

    /// First derivative S'(t).
    #[must_use]
    pub fn derivative(&self, t: f64) -> f64 {
        let i = self.segment(t);
        let dt = t - self.x[i];
        self.b[i] + dt * (2.0 * self.c[i] + 3.0 * dt * self.d[i])
    }

    /// Definite integral ∫ₐᵇ S(t) dt, splitting at interior knots
    /// (segments are integrated in closed form).
    #[must_use]
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        if a == b {
            return 0.0;
        }
        if b < a {
            return -self.integrate(b, a);
        }
        let seg_integral = |i: usize, from: f64, to: f64| -> f64 {
            let prim = |t: f64| {
                let dt = t - self.x[i];
                dt * (self.a[i]
                    + dt * (self.b[i] / 2.0 + dt * (self.c[i] / 3.0 + dt * self.d[i] / 4.0)))
            };
            prim(to) - prim(from)
        };
        let mut total = 0.0;
        let mut t = a;
        while t < b {
            let i = self.segment(t);
            let seg_end = if i + 2 < self.x.len() { self.x[i + 1] } else { f64::INFINITY };
            let to = seg_end.min(b);
            total += seg_integral(i, t, to);
            if to >= b {
                break;
            }
            t = to;
        }
        total
    }
}

fn catmull_rom_weights(u: f64) -> [f64; 4] {
    // 0.5·[−u³+2u²−u, 3u³−5u²+2, −3u³+4u²+u, u³−u²]
    let u2 = u * u;
    let u3 = u2 * u;
    [
        0.5 * (-u3 + 2.0 * u2 - u),
        0.5 * (3.0 * u3 - 5.0 * u2 + 2.0),
        0.5 * (-3.0 * u3 + 4.0 * u2 + u),
        0.5 * (u3 - u2),
    ]
}

/// Uniform Catmull-Rom spline through `points`, parameterized so that
/// t = i lands exactly on points[i] (t ∈ [0, n−1]; endpoints use
/// duplicated boundary points).
///
/// # Panics
/// Panics unless there are at least 2 points and t is within range.
#[must_use]
pub fn catmull_rom(points: &[Vec3], t: f64) -> Vec3 {
    assert!(points.len() >= 2, "catmull_rom requires at least 2 points");
    let n = points.len();
    assert!(
        (0.0..=(n as f64 - 1.0)).contains(&t),
        "catmull_rom requires t in [0, n-1]"
    );
    let j = (t.floor() as usize).min(n - 2);
    let u = t - j as f64;
    let p0 = points[j.saturating_sub(1)];
    let p1 = points[j];
    let p2 = points[j + 1];
    let p3 = points[(j + 2).min(n - 1)];
    let w = catmull_rom_weights(u);
    Vec3::new(
        w[0] * p0.x + w[1] * p1.x + w[2] * p2.x + w[3] * p3.x,
        w[0] * p0.y + w[1] * p1.y + w[2] * p2.y + w[3] * p3.y,
        w[0] * p0.z + w[1] * p1.z + w[2] * p2.z + w[3] * p3.z,
    )
}

/// 2-D uniform Catmull-Rom spline (same parameterization as
/// [`catmull_rom`]).
///
/// # Panics
/// Panics unless there are at least 2 points and t is within range.
#[must_use]
pub fn catmull_rom_2d(points: &[(f64, f64)], t: f64) -> (f64, f64) {
    assert!(points.len() >= 2, "catmull_rom_2d requires at least 2 points");
    let n = points.len();
    assert!(
        (0.0..=(n as f64 - 1.0)).contains(&t),
        "catmull_rom_2d requires t in [0, n-1]"
    );
    let j = (t.floor() as usize).min(n - 2);
    let u = t - j as f64;
    let p0 = points[j.saturating_sub(1)];
    let p1 = points[j];
    let p2 = points[j + 1];
    let p3 = points[(j + 2).min(n - 1)];
    let w = catmull_rom_weights(u);
    (
        w[0] * p0.0 + w[1] * p1.0 + w[2] * p2.0 + w[3] * p3.0,
        w[0] * p0.1 + w[1] * p1.1 + w[2] * p2.1 + w[3] * p3.1,
    )
}

/// Clamped uniform B-spline curve evaluated by de Boor's algorithm
/// (de Boor, *A Practical Guide to Splines*).
#[derive(Debug, Clone, PartialEq)]
pub struct BSpline {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub control: Vec<Vec3>,
}

impl BSpline {
    /// Clamped uniform knot vector: the curve interpolates the first
    /// and last control points, with parameter domain [0, n − p].
    ///
    /// # Panics
    /// Panics unless degree ≥ 1 and there are more than `degree`
    /// control points.
    #[must_use]
    pub fn uniform(degree: usize, control: &[Vec3]) -> Self {
        assert!(degree >= 1, "BSpline requires degree >= 1");
        assert!(
            control.len() > degree,
            "BSpline requires more control points than the degree"
        );
        let n = control.len();
        let interior = n - degree; // interior spans
        let mut knots = Vec::with_capacity(n + degree + 1);
        knots.resize(knots.len() + degree + 1, 0.0);
        for i in 1..interior {
            knots.push(i as f64);
        }
        for _ in 0..=degree {
            knots.push(interior as f64);
        }
        Self { degree, knots, control: control.to_vec() }
    }

    /// Domain of the curve parameter.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        (
            self.knots[self.degree],
            self.knots[self.knots.len() - self.degree - 1],
        )
    }

    fn find_span(&self, u: f64) -> usize {
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let last = self.knots.len() - self.degree - 2;
        if u >= hi {
            return last;
        }
        let mut span = self.degree;
        while span < last && self.knots[span + 1] <= u {
            span += 1;
        }
        span
    }

    /// Point on the curve at parameter u (clamped to the domain).
    #[must_use]
    pub fn eval(&self, u: f64) -> Vec3 {
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let p = self.degree;
        let k = self.find_span(u);
        // de Boor recursion on the p+1 active control points.
        let mut d: Vec<Vec3> = (0..=p).map(|j| self.control[k - p + j]).collect();
        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = k - p + j;
                let denom = self.knots[i + p - r + 1] - self.knots[i];
                let alpha = if denom == 0.0 { 0.0 } else { (u - self.knots[i]) / denom };
                d[j] = Vec3::new(
                    (1.0 - alpha) * d[j - 1].x + alpha * d[j].x,
                    (1.0 - alpha) * d[j - 1].y + alpha * d[j].y,
                    (1.0 - alpha) * d[j - 1].z + alpha * d[j].z,
                );
            }
        }
        d[p]
    }

    /// Derivative curve value at u: the degree p−1 B-spline with
    /// control points p·(P_{i+1} − P_i)/(u_{i+p+1} − u_{i+1}).
    #[must_use]
    pub fn derivative(&self, u: f64) -> Vec3 {
        let p = self.degree;
        if p == 1 {
            // Piecewise linear: constant derivative per span.
            let k = self.find_span(u);
            let i = k; // span [u_k, u_{k+1}) uses P_{k-1}, P_k for p=1
            let denom = self.knots[i + 1] - self.knots[i];
            let (a, b) = (self.control[i - 1], self.control[i]);
            if denom == 0.0 {
                return Vec3::new(0.0, 0.0, 0.0);
            }
            return Vec3::new(
                (b.x - a.x) / denom,
                (b.y - a.y) / denom,
                (b.z - a.z) / denom,
            );
        }
        let n = self.control.len();
        let mut dcontrol = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let denom = self.knots[i + p + 1] - self.knots[i + 1];
            let scale = if denom == 0.0 { 0.0 } else { p as f64 / denom };
            dcontrol.push(Vec3::new(
                scale * (self.control[i + 1].x - self.control[i].x),
                scale * (self.control[i + 1].y - self.control[i].y),
                scale * (self.control[i + 1].z - self.control[i].z),
            ));
        }
        let dspline = BSpline {
            degree: p - 1,
            knots: self.knots[1..self.knots.len() - 1].to_vec(),
            control: dcontrol,
        };
        dspline.eval(u)
    }
}

/// Arbitrary-degree Bezier curve point by the de Casteljau algorithm.
///
/// # Panics
/// Panics if `control` is empty or t is outside [0, 1].
#[must_use]
pub fn de_casteljau(control: &[Vec3], t: f64) -> Vec3 {
    assert!(!control.is_empty(), "de_casteljau requires control points");
    assert!((0.0..=1.0).contains(&t), "de_casteljau requires t in [0, 1]");
    let mut pts = control.to_vec();
    let n = pts.len();
    for r in 1..n {
        for i in 0..n - r {
            pts[i] = Vec3::new(
                (1.0 - t) * pts[i].x + t * pts[i + 1].x,
                (1.0 - t) * pts[i].y + t * pts[i + 1].y,
                (1.0 - t) * pts[i].z + t * pts[i + 1].z,
            );
        }
    }
    pts[0]
}

#[cfg(test)]
mod spline_tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_spline_passes_through_knots() {
        let x = [0.0, 1.0, 2.5, 4.0, 5.0];
        let y = [1.0, 3.0, -1.0, 2.0, 0.0];
        let s = CubicSpline::natural(&x, &y).unwrap();
        for (xi, yi) in x.iter().zip(&y) {
            assert!(approx(s.eval(*xi), *yi, 1e-12));
        }
    }

    #[test]
    fn test_clamped_end_slopes() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 0.0, 1.0];
        let s = CubicSpline::clamped(&x, &y, 2.0, -1.5).unwrap();
        assert!(approx(s.derivative(0.0), 2.0, 1e-10));
        assert!(approx(s.derivative(3.0), -1.5, 1e-10));
    }

    #[test]
    fn test_natural_second_derivative_zero_at_ends() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 2.0, 1.0, 3.0, 2.0];
        let s = CubicSpline::natural(&x, &y).unwrap();
        // S'' ~ finite difference of S' at the boundary knots.
        let h = 1e-6;
        let spp0 = (s.derivative(0.0 + h) - s.derivative(0.0)) / h;
        let sppn = (s.derivative(4.0) - s.derivative(4.0 - h)) / h;
        assert!(spp0.abs() < 1e-4, "S''(0) = {spp0}");
        assert!(sppn.abs() < 1e-4, "S''(n) = {sppn}");
    }

    #[test]
    fn test_spline_integrate_matches_quadratic() {
        // A quadratic sampled densely is reproduced closely; check the
        // integral of x^2 on [0, 2] = 8/3.
        let x: Vec<f64> = (0..21).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * v).collect();
        let s = CubicSpline::natural(&x, &y).unwrap();
        assert!(approx(s.integrate(0.0, 2.0), 8.0 / 3.0, 1e-4));
        assert!(approx(s.integrate(2.0, 0.0), -8.0 / 3.0, 1e-4));
        assert_eq!(s.integrate(1.0, 1.0), 0.0);
    }

    #[test]
    fn test_spline_errors() {
        assert!(CubicSpline::natural(&[0.0, 1.0], &[0.0, 1.0]).is_err());
        assert!(CubicSpline::natural(&[0.0, 1.0, 1.0], &[0.0, 1.0, 2.0]).is_err());
        assert!(CubicSpline::natural(&[0.0, 1.0, 2.0], &[0.0, 1.0]).is_err());
    }

    #[test]
    fn test_catmull_rom_interpolates() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, -1.0),
            Vec3::new(3.0, 1.0, 0.5),
            Vec3::new(4.0, 4.0, 2.0),
        ];
        for (i, p) in pts.iter().enumerate() {
            let q = catmull_rom(&pts, i as f64);
            assert!(approx(q.x, p.x, 1e-12) && approx(q.y, p.y, 1e-12) && approx(q.z, p.z, 1e-12));
        }
        let (x, y) = catmull_rom_2d(&[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], 1.0);
        assert!(approx(x, 1.0, 1e-12) && approx(y, 1.0, 1e-12));
    }

    #[test]
    fn test_bspline_endpoint_interpolation() {
        let ctrl = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(3.0, 2.0, 1.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let s = BSpline::uniform(3, &ctrl);
        let (lo, hi) = s.domain();
        let start = s.eval(lo);
        let end = s.eval(hi);
        assert!(approx(start.x, 0.0, 1e-12) && approx(start.y, 0.0, 1e-12));
        assert!(approx(end.x, 4.0, 1e-12) && approx(end.y, 0.0, 1e-12));
    }

    #[test]
    fn test_bspline_degree3_matches_bezier_for_4_points() {
        // A clamped cubic B-spline over exactly 4 control points is a
        // Bezier curve; compare with de Casteljau.
        let ctrl = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
        ];
        let s = BSpline::uniform(3, &ctrl);
        let (lo, hi) = s.domain();
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let u = lo + t * (hi - lo);
            let a = s.eval(u);
            let b = de_casteljau(&ctrl, t);
            assert!(approx(a.x, b.x, 1e-12) && approx(a.y, b.y, 1e-12), "t = {t}");
        }
    }

    #[test]
    fn test_bspline_derivative_matches_fd() {
        let ctrl = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 1.0),
            Vec3::new(2.0, 2.0, -1.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(4.0, 1.0, 2.0),
        ];
        let s = BSpline::uniform(3, &ctrl);
        let (lo, hi) = s.domain();
        let h = 1e-6;
        for i in 1..10 {
            let u = lo + (hi - lo) * i as f64 / 10.0;
            let d = s.derivative(u);
            let p1 = s.eval(u + h);
            let p0 = s.eval(u - h);
            assert!(approx(d.x, (p1.x - p0.x) / (2.0 * h), 1e-4), "u = {u}");
            assert!(approx(d.y, (p1.y - p0.y) / (2.0 * h), 1e-4));
            assert!(approx(d.z, (p1.z - p0.z) / (2.0 * h), 1e-4));
        }
    }

    #[test]
    fn test_de_casteljau_line_and_endpoints() {
        let ctrl = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 4.0, 6.0)];
        let mid = de_casteljau(&ctrl, 0.5);
        assert!(approx(mid.x, 1.0, 1e-15) && approx(mid.y, 2.0, 1e-15) && approx(mid.z, 3.0, 1e-15));
        let start = de_casteljau(&ctrl, 0.0);
        assert!(approx(start.x, 0.0, 1e-15));
    }
}
