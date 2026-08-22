//! Scalar and polynomial root finding.

use crate::error::SolveError;
use crate::fractals::Complex;

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

// ── Polynomial roots and Brent's method ─────────────────────────────

/// Evaluates a real polynomial by Horner's rule; `coeffs` are ordered
/// highest degree first.
///
/// # Panics
/// Panics if `coeffs` is empty.
#[must_use]
pub fn polynomial_eval(coeffs: &[f64], x: f64) -> f64 {
    assert!(!coeffs.is_empty(), "polynomial_eval requires coefficients");
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// Horner evaluation of a real-coefficient polynomial at a complex
/// point; `coeffs` are ordered highest degree first.
///
/// # Panics
/// Panics if `coeffs` is empty.
#[must_use]
pub fn polynomial_eval_complex(coeffs: &[f64], z: Complex) -> Complex {
    assert!(!coeffs.is_empty(), "polynomial_eval_complex requires coefficients");
    coeffs
        .iter()
        .fold(Complex::new(0.0, 0.0), |acc, &c| acc * z + Complex::new(c, 0.0))
}

const DK_MAX_ITER: usize = 1000;
const DK_TOL: f64 = 1e-13;

/// All complex roots of a real polynomial (coefficients highest degree
/// first) via the Durand-Kerner (Weierstrass) simultaneous iteration.
///
/// Leading zeros are stripped; the root count equals the degree.
/// Returns `InvalidArgument` for constant (degree-0) or all-zero input
/// and `NoConvergence` if the iteration stalls.
pub fn polynomial_roots(coeffs: &[f64]) -> Result<Vec<Complex>, SolveError> {
    let first_nonzero = coeffs.iter().position(|&c| c != 0.0);
    let coeffs = match first_nonzero {
        Some(i) => &coeffs[i..],
        None => return Err(SolveError::InvalidArgument("polynomial_roots requires a non-zero polynomial")),
    };
    let degree = coeffs.len() - 1;
    if degree == 0 {
        return Err(SolveError::InvalidArgument("polynomial_roots requires degree >= 1"));
    }
    // Normalize to a monic polynomial.
    let lead = coeffs[0];
    let monic: Vec<f64> = coeffs.iter().map(|&c| c / lead).collect();

    // Radius bound for the roots: 1 + max |a_i| (Cauchy bound).
    let radius = 1.0 + monic.iter().skip(1).fold(0.0_f64, |m, &c| m.max(c.abs()));

    // Initial guesses spread on a circle (the classical (0.4 + 0.9i)^k
    // seeds, scaled to the root bound so large roots are reachable).
    let seed = Complex::new(0.4, 0.9);
    let mut z: Vec<Complex> = Vec::with_capacity(degree);
    let mut acc = Complex::new(1.0, 0.0);
    for _ in 0..degree {
        acc = acc * seed;
        z.push(Complex::new(acc.re * radius.max(1.0), acc.im * radius.max(1.0)));
    }

    for _ in 0..DK_MAX_ITER {
        let mut max_step = 0.0_f64;
        for i in 0..degree {
            let p = polynomial_eval_complex(&monic, z[i]);
            let mut denom = Complex::new(1.0, 0.0);
            for j in 0..degree {
                if j != i {
                    denom = denom * (z[i] - z[j]);
                }
            }
            if denom.norm_sq() == 0.0 {
                // Coincident estimates: nudge apart and continue.
                z[i] = z[i] + Complex::new(1e-8, 1e-8);
                max_step = f64::MAX;
                continue;
            }
            let delta = p / denom;
            z[i] = z[i] - delta;
            max_step = max_step.max(delta.norm());
        }
        if max_step < DK_TOL * radius.max(1.0) {
            // Snap conjugate-symmetric roots' small imaginary parts to zero.
            for zi in z.iter_mut() {
                if zi.im.abs() < 1e-10 * (1.0 + zi.re.abs()) {
                    zi.im = 0.0;
                }
            }
            return Ok(z);
        }
    }
    Err(SolveError::NoConvergence { iters: DK_MAX_ITER, residual: f64::NAN })
}

const BRENT_EPS: f64 = f64::EPSILON;

/// Brent's method: bracketing root finder combining bisection, secant,
/// and inverse quadratic interpolation (Brent 1973; NR §9.3).
/// Superlinear convergence with guaranteed bracket retention.
///
/// Returns `InvalidArgument` unless f(a) and f(b) have opposite signs.
pub fn brent_root(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, SolveError> {
    if !(tol >= 0.0) {
        return Err(SolveError::InvalidArgument("brent_root requires tol >= 0"));
    }
    let (mut a, mut b) = (a, b);
    let mut fa = f(a);
    let mut fb = f(b);
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa * fb > 0.0 {
        return Err(SolveError::InvalidArgument("brent_root requires a sign change on [a, b]"));
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    for _ in 0..max_iter {
        if fb * fc > 0.0 {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * BRENT_EPS * b.abs() + 0.5 * tol;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 {
            return Ok(b);
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // Attempt inverse quadratic interpolation / secant.
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                let qq = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            let min1 = 3.0 * xm * q - (tol1 * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b += d;
        } else {
            b += if xm >= 0.0 { tol1 } else { -tol1 };
        }
        fb = f(b);
    }
    Err(SolveError::NoConvergence { iters: max_iter, residual: fb.abs() })
}
