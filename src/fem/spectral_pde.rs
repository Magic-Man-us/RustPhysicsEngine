//! Spectral methods: global basis functions instead of local ones.
//!
//! # What changes when the basis stops being local
//!
//! A finite element expands the solution in functions that are nonzero
//! on one or two cells. The matrix is sparse, and the accuracy is
//! whatever the polynomial degree gives -- `h^2`, `h^3`, a fixed power
//! of the mesh size no matter how smooth the answer is.
//!
//! A spectral method expands in functions that are nonzero everywhere
//! and smooth: complex exponentials on a periodic domain, Chebyshev
//! polynomials on an interval. The matrix becomes dense, and in exchange
//! the error stops obeying any fixed power of `N` at all. For an
//! analytic function it falls geometrically -- adding a few points
//! multiplies the error by a constant factor rather than reducing it by
//! a fixed order -- and for a function with `k` continuous derivatives
//! it falls as `N^-k`. The method is only as good as the solution is
//! smooth, and it is *exactly* as good as that. Both halves are measured
//! in the tests rather than asserted.
//!
//! # Two Poisson solvers that are not the same solver
//!
//! [`crate::transforms::fft::fft_poisson_2d`] already solves the
//! periodic Poisson problem with an FFT, but it is not a spectral
//! method. It divides by the eigenvalue of the *five-point* Laplacian,
//! `(2 cos kx + 2 cos ky - 4)/h^2`, which makes the discrete residual
//! vanish to rounding -- exactly what a pressure projection in a fluid
//! solver wants, since there the finite-difference divergence is the
//! thing that must be zero. Against the continuum it is second-order
//! accurate and no better.
//!
//! [`spectral_poisson_periodic`] divides by the true symbol `-k^2`. Its
//! discrete residual is not zero, and its error against the continuum
//! solution is nil for anything the grid can represent and geometrically
//! small otherwise. The two answers differ by `O(h^2)`, and which one is
//! wanted depends on whether the discrete operator or the differential
//! one is the thing being solved.
//!
//! # Chebyshev points cluster, and they have to
//!
//! Interpolating at equally spaced points on an interval diverges as the
//! degree grows, even for functions as tame as `1/(1+25x^2)` -- Runge's
//! phenomenon, and it is not a rounding problem but a property of the
//! Lebesgue constant, which grows like `2^N/(N log N)`. The Chebyshev
//! points `cos(j pi / N)` cluster towards the ends at a density that
//! makes the Lebesgue constant grow only logarithmically, which is what
//! makes high-degree interpolation usable at all.

use crate::error::SolveError;
use crate::fractals::Complex;
use crate::linalg::matrix::Matrix;

/// The `n + 1` Chebyshev-Gauss-Lobatto points on `[a, b]`.
///
/// Ordered descending on `[-1, 1]` -- `x_j = cos(j pi / n)` runs from `1`
/// to `-1` -- which is the convention Trefethen's differentiation matrix
/// assumes, and mapped affinely onto `[a, b]`. Getting the order
/// backwards flips the sign of every derivative, silently.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for `n == 0` or a degenerate
/// interval.
pub fn chebyshev_points(n: usize, a: f64, b: f64) -> Result<Vec<f64>, SolveError> {
    if n == 0 {
        return Err(SolveError::InvalidArgument("need at least one interval"));
    }
    if !(a.is_finite() && b.is_finite()) || b <= a {
        return Err(SolveError::InvalidArgument("need a finite interval with a < b"));
    }
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    Ok((0..=n)
        .map(|j| {
            let x = (std::f64::consts::PI * j as f64 / n as f64).cos();
            mid + half * x
        })
        .collect())
}

/// The Chebyshev differentiation matrix on `[a, b]`, `(n+1)` square.
///
/// Multiplying a vector of values at [`chebyshev_points`] by this matrix
/// gives the derivative of the degree-`n` polynomial through those
/// values, at the same points. For data that *is* a polynomial of degree
/// at most `n` the result is the exact derivative, to rounding, however
/// large `n` is.
///
/// The off-diagonal entries are Trefethen's
/// `(c_i / c_j) (-1)^{i+j} / (x_i - x_j)`, with `c` equal to two at the
/// ends and one inside. The diagonal is *not* set from its closed form
/// but as minus the sum of the rest of its row -- the negative sum trick.
/// The two agree analytically, and differ in floating point by
/// cancellation that grows with `n`; taking the sum makes the matrix
/// annihilate constants exactly instead of nearly, which matters because
/// the constant is the one thing every derivative operator must kill and
/// the error in it pollutes everything else.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for `n == 0` or a degenerate
/// interval.
pub fn cheb_diff_matrix(n: usize, a: f64, b: f64) -> Result<Matrix, SolveError> {
    let x = chebyshev_points(n, a, b)?;
    let c = |i: usize| if i == 0 || i == n { 2.0 } else { 1.0 };
    let mut d = Matrix::zeros(n + 1, n + 1);
    for i in 0..=n {
        for j in 0..=n {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                d.set(i, j, c(i) / c(j) * sign / (x[i] - x[j]));
            }
        }
    }
    for i in 0..=n {
        let row: f64 = (0..=n).filter(|&j| j != i).map(|j| d.get(i, j)).sum();
        d.set(i, i, -row);
    }
    Ok(d)
}

/// Differentiates values sampled at [`chebyshev_points`].
///
/// # Errors
///
/// [`SolveError::DimensionMismatch`] if the sample count is not
/// `n + 1` for the matrix's `n`.
pub fn cheb_differentiate(d: &Matrix, values: &[f64]) -> Result<Vec<f64>, SolveError> {
    d.mul_vec(values)
}

/// Solves `-(p u')' + q u = f` on `[a, b]` with Dirichlet ends by
/// Chebyshev collocation.
///
/// The operator is assembled as `-D diag(p) D + diag(q)` and the
/// equation is imposed at the interior collocation points, with the two
/// end rows replaced by the boundary conditions. Returns the `n + 1`
/// values at [`chebyshev_points`].
///
/// Only Dirichlet conditions are offered. A flux condition in a
/// collocation method means replacing an end row by a row of the
/// differentiation matrix, which works but changes the conditioning
/// enough to deserve its own treatment rather than a flag here.
///
/// The matrix is dense and the cost is `O(n^3)`, which is the trade the
/// method makes: far fewer unknowns for the same accuracy, each of them
/// coupled to all the others.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a degenerate interval, `n < 2`,
/// a non-positive `p`, or non-finite data; [`SolveError::Singular`] if
/// the collocation matrix is singular, which a reaction term negative
/// enough to hit an eigenvalue will do.
pub fn chebyshev_collocation_bvp(
    p: &dyn Fn(f64) -> f64,
    q: &dyn Fn(f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    bc: (f64, f64),
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    if n < 2 {
        return Err(SolveError::InvalidArgument("collocation needs at least two intervals"));
    }
    if !(bc.0.is_finite() && bc.1.is_finite()) {
        return Err(SolveError::InvalidArgument("boundary data must be finite"));
    }
    let x = chebyshev_points(n, a, b)?;
    let d = cheb_diff_matrix(n, a, b)?;
    let mut pv = Vec::with_capacity(n + 1);
    for &xi in &x {
        let v = p(xi);
        if !v.is_finite() || v <= 0.0 {
            return Err(SolveError::InvalidArgument("p must be positive and finite"));
        }
        pv.push(v);
    }
    // -D diag(p) D + diag(q), built directly rather than by three matrix
    // products, which is the same arithmetic without the temporaries.
    let mut m = Matrix::zeros(n + 1, n + 1);
    for i in 0..=n {
        for j in 0..=n {
            let s: f64 = (0..=n).map(|k| d.get(i, k) * pv[k] * d.get(k, j)).sum();
            m.set(i, j, -s);
        }
        let qv = q(x[i]);
        if !qv.is_finite() {
            return Err(SolveError::InvalidArgument("q must be finite"));
        }
        m.set(i, i, m.get(i, i) + qv);
    }
    let mut rhs = Vec::with_capacity(n + 1);
    for &xi in &x {
        let v = f(xi);
        if !v.is_finite() {
            return Err(SolveError::InvalidArgument("f must be finite"));
        }
        rhs.push(v);
    }
    // Row 0 is x = b and row n is x = a, because the points descend.
    for (row, value) in [(0usize, bc.1), (n, bc.0)] {
        for j in 0..=n {
            m.set(row, j, if j == row { 1.0 } else { 0.0 });
        }
        rhs[row] = value;
    }
    crate::linalg::lu::solve(&m, &rhs)
}

/// Solves `u'' = f` on a periodic interval of the given length, using
/// the true spectral symbol `-k^2`.
///
/// `f` is sampled at `n` equally spaced points starting at the left end;
/// the point at the right end is the same as the first and is not
/// included. The solution is fixed by taking it mean-free, which is the
/// only choice available: a periodic Poisson problem determines `u` only
/// up to a constant, and it has no solution at all unless `f` itself has
/// zero mean. A nonzero mean in the data is silently dropped -- the
/// alternative is refusing perfectly good data over a rounding-level
/// mean -- and [`spectral_poisson_periodic`] returns the solution of the
/// mean-free part.
///
/// Compare [`crate::transforms::fft::fft_poisson_2d`], which divides by
/// the five-point Laplacian's eigenvalue instead. See the module note:
/// they solve different problems and both are right.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for fewer than two samples, a
/// non-positive length, or non-finite data.
pub fn spectral_poisson_periodic(f: &[f64], length: f64) -> Result<Vec<f64>, SolveError> {
    let n = f.len();
    if n < 2 {
        return Err(SolveError::InvalidArgument("need at least two samples"));
    }
    if !length.is_finite() || length <= 0.0 {
        return Err(SolveError::InvalidArgument("the period must be positive"));
    }
    if f.iter().any(|v| !v.is_finite()) {
        return Err(SolveError::InvalidArgument("the source must be finite"));
    }
    let spec = crate::transforms::fft::fft_any(
        &f.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>(),
    );
    let mut out = vec![Complex::new(0.0, 0.0); n];
    for m in 1..n {
        // The signed frequency: modes past the halfway point are the
        // negative ones. Using the unsigned index would give the high
        // modes an enormous wavenumber and damp them into nothing.
        let signed = if m * 2 <= n { m as f64 } else { m as f64 - n as f64 };
        let k = std::f64::consts::TAU * signed / length;
        let lambda = -k * k;
        out[m] = Complex::new(spec[m].re / lambda, spec[m].im / lambda);
    }
    let inverse = crate::transforms::fft::ifft_any(&out);
    Ok(inverse.iter().map(|c| c.re).collect())
}

/// Differentiates a periodic sample twice with the spectral symbol,
/// which is the exact inverse of [`spectral_poisson_periodic`] on
/// mean-free data.
///
/// # Errors
///
/// As [`spectral_poisson_periodic`].
pub fn spectral_second_derivative(u: &[f64], length: f64) -> Result<Vec<f64>, SolveError> {
    let n = u.len();
    if n < 2 {
        return Err(SolveError::InvalidArgument("need at least two samples"));
    }
    if !length.is_finite() || length <= 0.0 {
        return Err(SolveError::InvalidArgument("the period must be positive"));
    }
    if u.iter().any(|v| !v.is_finite()) {
        return Err(SolveError::InvalidArgument("the samples must be finite"));
    }
    let spec = crate::transforms::fft::fft_any(
        &u.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>(),
    );
    let mut out = vec![Complex::new(0.0, 0.0); n];
    for m in 1..n {
        let signed = if m * 2 <= n { m as f64 } else { m as f64 - n as f64 };
        let k = std::f64::consts::TAU * signed / length;
        out[m] = Complex::new(-k * k * spec[m].re, -k * k * spec[m].im);
    }
    let inverse = crate::transforms::fft::ifft_any(&out);
    Ok(inverse.iter().map(|c| c.re).collect())
}

/// The largest error in the Chebyshev derivative of `f` at each degree
/// in `sizes`.
///
/// The point of the function is the *shape* of what it returns, not any
/// one entry. For an analytic `f` the sequence falls geometrically and a
/// log-log fit against `n` finds no fixed slope at all; for an `f` with
/// `k` continuous derivatives it falls as `n^-k` and the fit finds
/// exactly `k`. Plotting one without the other is what makes spectral
/// accuracy look like magic rather than like a statement about
/// smoothness.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] if any size is below one or the
/// interval is degenerate.
pub fn spectral_convergence_demo(
    f: &dyn Fn(f64) -> f64,
    df: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    sizes: &[usize],
) -> Result<Vec<f64>, SolveError> {
    let mut out = Vec::with_capacity(sizes.len());
    for &n in sizes {
        let x = chebyshev_points(n, a, b)?;
        let d = cheb_diff_matrix(n, a, b)?;
        let values: Vec<f64> = x.iter().map(|&xi| f(xi)).collect();
        let got = cheb_differentiate(&d, &values)?;
        let worst = got
            .iter()
            .zip(x.iter())
            .map(|(&g, &xi)| (g - df(xi)).abs())
            .fold(0.0, f64::max);
        out.push(worst);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    /// The correlation of `y` against `x`, used to ask which of two
    /// models a sequence of errors actually follows.
    fn correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
        let syy: f64 = y.iter().map(|b| (b - my) * (b - my)).sum();
        sxy / (sxx * syy).sqrt()
    }

    #[test]
    fn the_points_descend_from_one_end_to_the_other() {
        let x = chebyshev_points(6, -2.0, 3.0).unwrap();
        assert_eq!(x.len(), 7);
        assert!((x[0] - 3.0).abs() < 1e-15, "the first point is the right end");
        assert!((x[6] + 2.0).abs() < 1e-15, "the last point is the left end");
        for w in x.windows(2) {
            assert!(w[1] < w[0], "the points did not descend");
        }
        // Symmetric about the midpoint, and clustered towards the ends:
        // the outermost gap is smaller than the middle one.
        let mid = 0.5;
        for j in 0..=6 {
            assert!(((x[j] - mid) + (x[6 - j] - mid)).abs() < 1e-14);
        }
        assert!(x[0] - x[1] < x[3] - x[4], "the points did not cluster at the ends");
        assert!(chebyshev_points(0, 0.0, 1.0).is_err());
        assert!(chebyshev_points(4, 1.0, 1.0).is_err());
    }

    #[test]
    fn differentiation_is_exact_on_polynomials_it_can_represent() {
        // Not a tolerance: the interpolant of a polynomial of degree at
        // most n *is* that polynomial, so the matrix returns its
        // derivative to rounding however large n is. The residual grows
        // only as the matrix's own conditioning does.
        for n in [4usize, 9, 20] {
            let d = cheb_diff_matrix(n, -1.0, 1.0).unwrap();
            let x = chebyshev_points(n, -1.0, 1.0).unwrap();
            for degree in 0..=n {
                let v: Vec<f64> = x.iter().map(|t| t.powi(degree as i32)).collect();
                let got = cheb_differentiate(&d, &v).unwrap();
                for (k, &t) in x.iter().enumerate() {
                    let want = if degree == 0 {
                        0.0
                    } else {
                        degree as f64 * t.powi(degree as i32 - 1)
                    };
                    assert!(
                        (got[k] - want).abs() < 1e-12,
                        "n={n} degree={degree} point {k}: {} vs {want}",
                        got[k]
                    );
                }
                // And twice differentiating gives the second derivative.
                let twice = cheb_differentiate(&d, &got).unwrap();
                for (k, &t) in x.iter().enumerate() {
                    let want = if degree < 2 {
                        0.0
                    } else {
                        (degree * (degree - 1)) as f64 * t.powi(degree as i32 - 2)
                    };
                    assert!((twice[k] - want).abs() < 1e-9, "n={n} degree={degree} second");
                }
            }
        }
    }

    #[test]
    fn the_matrix_has_the_symmetries_its_point_set_forces() {
        for n in [5usize, 12, 25] {
            let d = cheb_diff_matrix(n, -1.0, 1.0).unwrap();
            for i in 0..=n {
                // Constants have zero derivative, so every row sums to
                // nothing -- which the negative sum trick arranges by
                // construction rather than by luck.
                let row: f64 = (0..=n).map(|j| d.get(i, j)).sum();
                assert!(row.abs() < 1e-12, "row {i} of {n} summed to {row}");
                // The point set is symmetric about the midpoint and
                // differentiation is odd, so the matrix is
                // centro-antisymmetric.
                for j in 0..=n {
                    let mirrored = d.get(n - i, n - j);
                    assert!(
                        (d.get(i, j) + mirrored).abs() < 1e-10 * (1.0 + d.get(i, j).abs()),
                        "n={n} entry ({i},{j}) is not centro-antisymmetric"
                    );
                }
            }
        }
    }

    #[test]
    fn moving_to_another_interval_only_rescales_the_matrix() {
        // d/dx on [a, b] is (2 / (b - a)) times d/dx on [-1, 1], and
        // nothing else changes. A missing Jacobian here would leave every
        // test on [-1, 1] passing.
        let n = 10;
        let unit = cheb_diff_matrix(n, -1.0, 1.0).unwrap();
        for (a, b) in [(0.0, 1.0), (-3.0, 2.5), (10.0, 10.5)] {
            let scaled = cheb_diff_matrix(n, a, b).unwrap();
            let factor = 2.0 / (b - a);
            for i in 0..=n {
                for j in 0..=n {
                    let want = factor * unit.get(i, j);
                    assert!(
                        (scaled.get(i, j) - want).abs() < 1e-10 * (1.0 + want.abs()),
                        "({a}, {b}) entry ({i}, {j})"
                    );
                }
            }
        }
        assert!(cheb_diff_matrix(0, 0.0, 1.0).is_err());
    }

    #[test]
    fn the_periodic_solver_is_exact_on_what_the_grid_can_hold() {
        // For a trigonometric polynomial inside the band there is no
        // truncation at all, so the answer is exact to rounding rather
        // than merely accurate.
        let n = 32;
        let l = 2.0 * PI;
        let at = |i: usize| l * i as f64 / n as f64;
        let f: Vec<f64> =
            (0..n).map(|i| -(3.0 * at(i)).sin() - 4.0 * (2.0 * at(i)).cos()).collect();
        let u = spectral_poisson_periodic(&f, l).unwrap();
        for i in 0..n {
            let want = (3.0 * at(i)).sin() / 9.0 + (2.0 * at(i)).cos();
            assert!((u[i] - want).abs() < 1e-13, "sample {i}: {} vs {want}", u[i]);
        }
        // The solution is mean-free, which is the only normalisation a
        // periodic problem admits.
        let mean = u.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-13, "the mean was {mean}");
        // Differentiating twice with the same symbol undoes it exactly.
        let back = spectral_second_derivative(&u, l).unwrap();
        for i in 0..n {
            assert!((back[i] - f[i]).abs() < 1e-12, "round trip at {i}");
        }
    }

    #[test]
    fn the_spectral_symbol_and_the_difference_symbol_disagree_by_h_squared() {
        // The module claims these are different solvers. This is the
        // demonstration: the spectral solution satisfies the *continuum*
        // equation to rounding and the three-point difference equation
        // only to second order. A finite-difference solver would have it
        // exactly the other way round.
        let l = 2.0 * PI;
        let mut previous = f64::INFINITY;
        for n in [16usize, 32, 64] {
            let at = |i: usize| l * i as f64 / n as f64;
            let f: Vec<f64> = (0..n).map(|i| -(3.0 * at(i)).sin()).collect();
            let u = spectral_poisson_periodic(&f, l).unwrap();
            let h = l / n as f64;
            let discrete: f64 = (0..n)
                .map(|i| {
                    let left = u[(i + n - 1) % n];
                    let right = u[(i + 1) % n];
                    ((left - 2.0 * u[i] + right) / (h * h) - f[i]).abs()
                })
                .fold(0.0, f64::max);
            let exact = spectral_second_derivative(&u, l)
                .unwrap()
                .iter()
                .zip(f.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(exact < 1e-12, "the spectral residual was {exact}");
            assert!(discrete > 1e-4, "the difference residual vanished too: {discrete}");
            if previous.is_finite() {
                let ratio = previous / discrete;
                assert!((ratio - 4.0).abs() < 0.2, "the difference residual fell by {ratio}");
            }
            previous = discrete;
        }
    }

    #[test]
    fn collocation_reaches_rounding_on_a_smooth_problem() {
        let n = 24;
        let u = chebyshev_collocation_bvp(
            &|_| 1.0,
            &|_| 0.0,
            &|x: f64| PI * PI * (PI * x).sin(),
            0.0,
            1.0,
            (0.0, 0.0),
            n,
        )
        .unwrap();
        let x = chebyshev_points(n, 0.0, 1.0).unwrap();
        for (k, &xi) in x.iter().enumerate() {
            assert!((u[k] - (PI * xi).sin()).abs() < 1e-12, "point {k}");
        }
        // A polynomial the space contains comes back untouched, whatever
        // the coefficients: -( (1+x) u' )' with u = x^2 - x.
        let m = 8;
        let v = chebyshev_collocation_bvp(
            &|x: f64| 1.0 + x,
            &|_| 0.0,
            // -(p u')' = -(p' u' + p u'') = -((2x - 1) + (1 + x) * 2).
            &|x: f64| -((2.0 * x - 1.0) + 2.0 * (1.0 + x)),
            0.0,
            1.0,
            (0.0, 0.0),
            m,
        )
        .unwrap();
        let y = chebyshev_points(m, 0.0, 1.0).unwrap();
        for (k, &xi) in y.iter().enumerate() {
            assert!((v[k] - (xi * xi - xi)).abs() < 1e-12, "patch point {k}: {}", v[k]);
        }
    }

    #[test]
    fn collocation_agrees_with_the_finite_element_solver() {
        // Two entirely different discretisations of the same operator.
        // Twenty-four collocation points against four hundred linear
        // elements, and they meet to the accuracy of the weaker one.
        use crate::fem::fem1d::{fem_1d_general, Bc, Fem1dSolution};
        let p = |x: f64| 1.0 + x * x;
        let q = |x: f64| 0.5 + x;
        let f = |x: f64| (2.0 * x).sin() + 1.0;
        let n = 24;
        let spectral =
            chebyshev_collocation_bvp(&p, &q, &f, 0.0, 1.0, (0.2, -0.3), n).unwrap();
        let elements = Fem1dSolution::new(
            0.0,
            1.0,
            1,
            fem_1d_general(
                &p,
                &q,
                &f,
                0.0,
                1.0,
                (Bc::Dirichlet(0.2), Bc::Dirichlet(-0.3)),
                400,
            )
            .unwrap(),
        )
        .unwrap();
        let x = chebyshev_points(n, 0.0, 1.0).unwrap();
        for (k, &xi) in x.iter().enumerate() {
            let want = elements.eval(xi);
            assert!(
                (spectral[k] - want).abs() < 1e-4 * (1.0 + want.abs()),
                "point {k} at {xi}: {} against {want}",
                spectral[k]
            );
        }
    }

    #[test]
    fn smooth_data_converges_geometrically_and_rough_data_does_not() {
        // The distinction is not "fast against slow" -- it is which
        // model the sequence of errors follows. For an analytic function
        // the log error is linear in n; for one with a few derivatives
        // it is linear in log n. Asking which fit is better tells the
        // two apart without having to name a rate.
        let analytic: Vec<usize> = vec![8, 12, 16, 20];
        let rough: Vec<usize> = vec![8, 12, 16, 20, 24, 28, 32, 36];
        let fits = |sizes: &[usize], e: &[f64]| {
            let ln_e: Vec<f64> = e.iter().map(|v| v.ln()).collect();
            let n: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
            let ln_n: Vec<f64> = n.iter().map(|v| v.ln()).collect();
            (correlation(&ln_n, &ln_e), correlation(&n, &ln_e))
        };
        let smooth = spectral_convergence_demo(
            &|x: f64| x.sin().exp(),
            &|x: f64| x.cos() * x.sin().exp(),
            -1.0,
            1.0,
            &analytic,
        )
        .unwrap();
        let (power, exponential) = fits(&analytic, &smooth);
        assert!(
            exponential < power,
            "an analytic function fitted a power law better: {exponential} against {power}"
        );
        assert!(smooth[3] < 1e-11, "n = 20 left {}", smooth[3]);
        assert!(smooth[0] / smooth[3] > 1e7, "the error barely moved");

        // |x|^3 has two continuous derivatives, so its derivative error
        // falls as a power of n and no faster.
        let kinked = spectral_convergence_demo(
            &|x: f64| x.abs().powi(3),
            &|x: f64| 3.0 * x * x * x.signum(),
            -1.0,
            1.0,
            &rough,
        )
        .unwrap();
        let (power, exponential) = fits(&rough, &kinked);
        assert!(
            power < exponential,
            "a kinked function fitted an exponential better: {power} against {exponential}"
        );
        assert!(power < -0.99, "the power law fit was poor: {power}");
        // Smoother kinks converge faster: |x|^5 has four derivatives.
        let smoother = spectral_convergence_demo(
            &|x: f64| x.abs().powi(5),
            &|x: f64| 5.0 * x.powi(4) * x.signum(),
            -1.0,
            1.0,
            &rough,
        )
        .unwrap();
        for k in 0..rough.len() {
            assert!(smoother[k] < kinked[k], "|x|^5 was not smoother than |x|^3 at {k}");
        }
        assert!(spectral_convergence_demo(&|_| 0.0, &|_| 0.0, 0.0, 1.0, &[0]).is_err());
    }

    #[test]
    fn the_solvers_refuse_impossible_arguments() {
        assert!(chebyshev_collocation_bvp(&|_| 1.0, &|_| 0.0, &|_| 1.0, 0.0, 1.0, (0.0, 0.0), 1)
            .is_err());
        assert!(
            chebyshev_collocation_bvp(&|_| -1.0, &|_| 0.0, &|_| 1.0, 0.0, 1.0, (0.0, 0.0), 6)
                .is_err()
        );
        assert!(chebyshev_collocation_bvp(
            &|_| 1.0,
            &|_| f64::NAN,
            &|_| 1.0,
            0.0,
            1.0,
            (0.0, 0.0),
            6
        )
        .is_err());
        assert!(chebyshev_collocation_bvp(
            &|_| 1.0,
            &|_| 0.0,
            &|_| 1.0,
            0.0,
            1.0,
            (f64::NAN, 0.0),
            6
        )
        .is_err());
        assert!(chebyshev_collocation_bvp(&|_| 1.0, &|_| 0.0, &|_| 1.0, 1.0, 0.0, (0.0, 0.0), 6)
            .is_err());
        assert!(spectral_poisson_periodic(&[1.0], 1.0).is_err());
        assert!(spectral_poisson_periodic(&[1.0, 2.0], 0.0).is_err());
        assert!(spectral_poisson_periodic(&[1.0, f64::NAN], 1.0).is_err());
        assert!(spectral_second_derivative(&[1.0], 1.0).is_err());
        assert!(spectral_second_derivative(&[1.0, 2.0], -1.0).is_err());
        assert!(spectral_second_derivative(&[f64::INFINITY, 2.0], 1.0).is_err());
        let d = cheb_diff_matrix(4, 0.0, 1.0).unwrap();
        assert!(cheb_differentiate(&d, &[1.0, 2.0]).is_err());
    }
}
