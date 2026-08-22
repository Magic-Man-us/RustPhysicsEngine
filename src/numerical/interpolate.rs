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
