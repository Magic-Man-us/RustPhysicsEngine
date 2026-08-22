//! Scalar root finding.

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
