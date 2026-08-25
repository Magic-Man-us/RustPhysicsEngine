//! Random matrix theory: the classical ensembles, their limiting spectral
//! laws, and the local statistics that distinguish correlated spectra from
//! uncorrelated ones.
//!
//! The subject rests on a surprise: the eigenvalues of a large random matrix
//! are not themselves random in any useful sense. Their *density* converges
//! to a fixed shape that does not depend on the distribution of the entries
//! -- Wigner's semicircle for a symmetric matrix, Marchenko-Pastur for a
//! sample covariance -- and their *spacings* converge to a distribution that
//! depends only on the symmetry class. Universality is what makes the subject
//! applicable: a spectrum can be compared against these laws without knowing
//! anything about the mechanism that produced it.
//!
//! The practical payoff is a null hypothesis. Eigenvalues of independent
//! variables repel each other, in a way that independent *points* do not, so
//! the spacing distribution separates a spectrum with genuine level
//! correlations from a Poisson process of unrelated levels. In finance the
//! same statement is a filter: any eigenvalue of a sample correlation matrix
//! that falls inside the Marchenko-Pastur band is consistent with pure noise
//! and carries no information about the correlations being estimated.
//!
//! Two conventions are fixed throughout. Ensembles are scaled so their
//! limiting support stays put as `n` grows -- otherwise the semicircle's
//! radius would drift and nothing would converge to compare against. And
//! spacings are always measured on *unfolded* eigenvalues, rescaled to unit
//! mean density, since the raw spacings of a semicircular spectrum are much
//! tighter in the middle than at the edges and their distribution would say
//! more about the density than about the correlations.

use crate::error::GeomError;
use crate::linalg::eigen::eigen_symmetric;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// Tolerance and sweep budget for the Jacobi eigen-solver used throughout.
const EIG_TOL: f64 = 1e-12;
const EIG_SWEEPS: usize = 100;

/// A sample from the Gaussian orthogonal ensemble: a symmetric matrix whose
/// entries are Gaussian, independent up to the symmetry constraint.
///
/// Scaled so the spectrum fills `[-2, 2]` in the large-`n` limit: off-diagonal
/// entries have variance `1/n` and diagonal entries `2/n`. The factor of two
/// on the diagonal is not decorative -- it is what makes the distribution
/// invariant under orthogonal conjugation, which is the defining property of
/// the ensemble and the reason its spectral statistics are universal.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn goe_sample(n: usize, rng: &mut Rng) -> Matrix {
    assert!(n > 0, "goe_sample requires n > 0");
    let scale = 1.0 / (n as f64).sqrt();
    let mut m = Matrix::zeros(n, n);
    for i in 0..n {
        m.set(i, i, std::f64::consts::SQRT_2 * scale * rng.next_gaussian());
        for j in i + 1..n {
            let v = scale * rng.next_gaussian();
            m.set(i, j, v);
            m.set(j, i, v);
        }
    }
    m
}

/// A sample from the Gaussian unitary ensemble, returned as
/// `(real part, imaginary part)` of a Hermitian matrix.
///
/// The real part is symmetric and the imaginary part antisymmetric with a
/// zero diagonal, which together is what "Hermitian" means for a matrix held
/// in two real halves. Scaled to the same `[-2, 2]` support as
/// [`goe_sample`]: each independent real degree of freedom carries variance
/// `1/(2n)`, so `E|H_ij|^2 = 1/n` off the diagonal.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn gue_sample(n: usize, rng: &mut Rng) -> (Matrix, Matrix) {
    assert!(n > 0, "gue_sample requires n > 0");
    let scale = 1.0 / (2.0 * n as f64).sqrt();
    let mut re = Matrix::zeros(n, n);
    let mut im = Matrix::zeros(n, n);
    for i in 0..n {
        // A Hermitian diagonal is real, and carries twice the variance of an
        // off-diagonal entry's real part for the same invariance reason as GOE.
        re.set(i, i, std::f64::consts::SQRT_2 * scale * rng.next_gaussian());
        for j in i + 1..n {
            let a = scale * rng.next_gaussian();
            let b = scale * rng.next_gaussian();
            re.set(i, j, a);
            re.set(j, i, a);
            im.set(i, j, b);
            im.set(j, i, -b);
        }
    }
    (re, im)
}

/// A sample from the Ginibre ensemble: every entry independent Gaussian, with
/// no symmetry imposed at all.
///
/// Its eigenvalues are complex and fill the unit disc rather than an
/// interval, which is the point of the ensemble -- non-normality changes the
/// spectral picture completely.
///
/// # Panics
/// Panics if `n` is zero.
#[must_use]
pub fn ginibre_sample(n: usize, rng: &mut Rng) -> Matrix {
    assert!(n > 0, "ginibre_sample requires n > 0");
    let scale = 1.0 / (n as f64).sqrt();
    let mut m = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m.set(i, j, scale * rng.next_gaussian());
        }
    }
    m
}

/// A sample covariance matrix built from `p` independent variables observed
/// `n` times, each observation standard Gaussian.
///
/// Returns `X' X / n` where `X` is `n` by `p`, so the population covariance
/// is the identity and every departure from it in the sample is estimation
/// noise. That noise is exactly what [`marchenko_pastur`] describes.
///
/// # Panics
/// Panics if either dimension is zero.
#[must_use]
pub fn wishart_sample(n: usize, p: usize, rng: &mut Rng) -> Matrix {
    assert!(n > 0 && p > 0, "wishart_sample requires positive dimensions");
    let mut x = Matrix::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x.set(i, j, rng.next_gaussian());
        }
    }
    let mut s = Matrix::zeros(p, p);
    for a in 0..p {
        for b in a..p {
            let v: f64 = (0..n).map(|i| x.get(i, a) * x.get(i, b)).sum::<f64>() / n as f64;
            s.set(a, b, v);
            s.set(b, a, v);
        }
    }
    s
}

/// Wigner's semicircle density on `[-r, r]`.
///
/// `f(x) = 2 sqrt(r^2 - x^2) / (pi r^2)`, zero outside. The limiting
/// eigenvalue density of a symmetric random matrix, whatever the entry
/// distribution, provided the entries are independent with finite variance --
/// the first and simplest statement of universality in the subject.
///
/// # Panics
/// Panics unless `r` is positive.
#[must_use]
pub fn wigner_semicircle(x: f64, r: f64) -> f64 {
    assert!(r > 0.0, "wigner_semicircle requires a positive radius");
    if x.abs() >= r {
        0.0
    } else {
        2.0 * (r * r - x * x).sqrt() / (std::f64::consts::PI * r * r)
    }
}

/// The Marchenko-Pastur density for a sample covariance matrix.
///
/// `ratio` is `p / n`, the number of variables over the number of
/// observations, and `sigma2` the population variance. Support is
/// `[sigma2 (1 -+ sqrt(ratio))^2]`; the density there is
/// `sqrt((b - x)(x - a)) / (2 pi ratio sigma2 x)`.
///
/// This is the shape a covariance matrix of *independent* variables takes.
/// The width of the band is the whole point: at `ratio = 0.5` the sample
/// eigenvalues spread over roughly `[0.09, 2.9]` even though every population
/// eigenvalue is exactly 1.
///
/// The point mass at zero when `ratio > 1` (more variables than
/// observations, so the matrix is singular) is not part of the density and is
/// not reported here.
///
/// # Panics
/// Panics unless `ratio` and `sigma2` are positive.
#[must_use]
pub fn marchenko_pastur(x: f64, ratio: f64, sigma2: f64) -> f64 {
    assert!(ratio > 0.0, "marchenko_pastur requires a positive ratio");
    assert!(sigma2 > 0.0, "marchenko_pastur requires a positive variance");
    let (a, b) = mp_edges(ratio, sigma2);
    if x <= a || x >= b {
        return 0.0;
    }
    ((b - x) * (x - a)).sqrt() / (2.0 * std::f64::consts::PI * ratio * sigma2 * x)
}

/// The two edges of the Marchenko-Pastur support,
/// `sigma2 (1 -+ sqrt(ratio))^2`.
///
/// Any sample eigenvalue between these is consistent with pure noise.
///
/// # Panics
/// Panics unless `ratio` and `sigma2` are positive.
#[must_use]
pub fn mp_edges(ratio: f64, sigma2: f64) -> (f64, f64) {
    assert!(ratio > 0.0, "mp_edges requires a positive ratio");
    assert!(sigma2 > 0.0, "mp_edges requires a positive variance");
    let root = ratio.sqrt();
    (sigma2 * (1.0 - root) * (1.0 - root), sigma2 * (1.0 + root) * (1.0 + root))
}

/// Gaps between consecutive eigenvalues after unfolding to unit mean density.
///
/// Unfolding is not a cosmetic step. The raw gaps of a semicircular spectrum
/// are far tighter near zero than near the edges, so their distribution would
/// mostly reflect that varying density rather than the correlations between
/// levels. Mapping each eigenvalue through a smooth estimate of its own
/// cumulative count removes the density and leaves the local statistics,
/// which is what the surmises below describe.
///
/// The smooth estimate here is the empirical staircase itself, smoothed by
/// averaging over a window that grows as the square root of the sample -- the
/// standard compromise between following the density and following the
/// fluctuations one is trying to measure.
///
/// Returns `eigs.len() - 1` gaps with mean 1. An empty or single-element
/// input gives an empty result.
#[must_use]
pub fn eigenvalue_spacing_distribution(eigs: &[f64]) -> Vec<f64> {
    if eigs.len() < 3 {
        return Vec::new();
    }
    let mut sorted = eigs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    // Local density from a symmetric window: the count spanned divided by the
    // interval it spans. Widening the window as sqrt(n) keeps the estimate
    // smooth without letting it track the level fluctuations themselves.
    let half = ((n as f64).sqrt() as usize).max(2);
    let mut gaps = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n - 1);
        let span = sorted[hi] - sorted[lo];
        if span <= 0.0 {
            continue;
        }
        let density = (hi - lo) as f64 / span;
        gaps.push((sorted[i + 1] - sorted[i]) * density);
    }
    // Normalise the mean exactly to one; the density estimate above is only
    // approximately correct and the surmises are stated for unit mean.
    let m: f64 = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;
    if m > 0.0 {
        for g in &mut gaps {
            *g /= m;
        }
    }
    gaps
}

/// Wigner's surmise for the orthogonal class:
/// `(pi/2) s exp(-pi s^2 / 4)`.
///
/// The spacing distribution of a two-by-two GOE matrix, which turns out to
/// approximate the large-`n` answer to within a percent. Its defining feature
/// is the linear vanishing at `s = 0`: eigenvalues of a real symmetric random
/// matrix repel, so exact degeneracies have probability zero and near ones
/// are rare.
#[must_use]
pub fn wigner_surmise_goe(s: f64) -> f64 {
    if s < 0.0 {
        return 0.0;
    }
    let pi = std::f64::consts::PI;
    (pi / 2.0) * s * (-pi * s * s / 4.0).exp()
}

/// Wigner's surmise for the unitary class:
/// `(32 / pi^2) s^2 exp(-4 s^2 / pi)`.
///
/// The repulsion is quadratic rather than linear -- a complex Hermitian
/// matrix has twice as many degrees of freedom to tune away from a
/// degeneracy, so near-degeneracies are suppressed harder than in the
/// orthogonal class.
#[must_use]
pub fn wigner_surmise_gue(s: f64) -> f64 {
    if s < 0.0 {
        return 0.0;
    }
    let pi = std::f64::consts::PI;
    (32.0 / (pi * pi)) * s * s * (-4.0 * s * s / pi).exp()
}

/// The spacing density of uncorrelated levels: `exp(-s)`.
///
/// A Poisson process of points has no repulsion at all, so its density is
/// maximal at zero. This is the null the surmises above are contrasted
/// against, and the contrast at small `s` is the whole diagnostic.
#[must_use]
pub fn poisson_spacing(s: f64) -> f64 {
    if s < 0.0 {
        0.0
    } else {
        (-s).exp()
    }
}

/// The spectral rigidity `Delta_3(L)`: the mean-square deviation of the
/// unfolded counting function from the best straight line over a window of
/// length `L`.
///
/// Where the spacing distribution measures correlations between *neighbours*,
/// rigidity measures them over a stretch of `L` levels, and it is the more
/// discriminating of the two. Uncorrelated levels give `L / 15`, growing
/// linearly; a correlated spectrum gives roughly `ln(L) / pi^2`, growing so
/// slowly that at `L = 20` the two differ by an order of magnitude.
///
/// Averaged over windows starting across the spectrum.
///
/// # Panics
/// Panics unless `l` is positive.
#[must_use]
pub fn spectral_rigidity(eigs: &[f64], l: f64) -> f64 {
    assert!(l > 0.0, "spectral_rigidity requires a positive window length");
    let unfolded = unfold(eigs);
    let n = unfolded.len();
    if n < 4 {
        return 0.0;
    }
    let total = unfolded[n - 1] - unfolded[0];
    if total <= l {
        return 0.0;
    }

    let windows = 200usize;
    let mut acc = 0.0;
    let mut used = 0usize;
    for w in 0..windows {
        let start = unfolded[0] + (total - l) * w as f64 / (windows - 1) as f64;
        let end = start + l;
        // The counting function N(x) over this window, sampled finely enough
        // that the least-squares fit sees every step.
        let samples = 400usize;
        let mut sxx = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxy = 0.0;
        let mut syy = 0.0;
        for k in 0..samples {
            let x = start + (end - start) * (k as f64 + 0.5) / samples as f64;
            let count = unfolded.partition_point(|&v| v <= x) as f64;
            sx += x;
            sy += count;
            sxx += x * x;
            sxy += x * count;
            syy += count * count;
        }
        let m = samples as f64;
        let den = m * sxx - sx * sx;
        if den.abs() < 1e-300 {
            continue;
        }
        let slope = (m * sxy - sx * sy) / den;
        let intercept = (sy - slope * sx) / m;
        // Mean squared residual of the fit, which is Delta_3 for this window.
        let residual = (syy - 2.0 * slope * sxy - 2.0 * intercept * sy
            + slope * slope * sxx
            + 2.0 * slope * intercept * sx
            + intercept * intercept * m)
            / m;
        acc += residual.max(0.0);
        used += 1;
    }
    if used == 0 {
        0.0
    } else {
        acc / used as f64
    }
}

/// Eigenvalues mapped to unit mean density, so that the `k`-th sits near `k`.
fn unfold(eigs: &[f64]) -> Vec<f64> {
    let mut sorted = eigs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let gaps = eigenvalue_spacing_distribution(&sorted);
    let mut out = Vec::with_capacity(gaps.len() + 1);
    let mut acc = 0.0;
    out.push(acc);
    for g in gaps {
        acc += g;
        out.push(acc);
    }
    out
}

/// An approximation to the Tracy-Widom distribution function for the
/// orthogonal class, the law of the largest eigenvalue after edge scaling.
///
/// Represented as a shifted gamma matched to the first three cumulants of
/// `TW_1` (mean `-1.2065`, variance `1.6078`, skewness `0.2935`), which is the
/// standard closed-form stand-in: exact evaluation needs the Hastings-McLeod
/// solution of Painleve II. Accurate to a few parts in a thousand through the
/// body, degrading in the far tails, where the true law decays like
/// `exp(-|x|^3/24)` on the left and `exp(-(2/3) x^{3/2})` on the right.
///
/// The distribution matters because the largest eigenvalue does not
/// fluctuate on the scale of the spectrum: it sits within `n^{-2/3}` of the
/// edge, so a spike only a little above the Marchenko-Pastur edge is still
/// strong evidence of real signal.
#[must_use]
pub fn tracy_widom_beta1_approx(x: f64) -> f64 {
    // Match a three-parameter gamma to the first three cumulants.
    const MEAN: f64 = -1.206_533_6;
    const VARIANCE: f64 = 1.607_781_0;
    const SKEW: f64 = 0.293_464_7;

    let shape = 4.0 / (SKEW * SKEW);
    let scale = VARIANCE.sqrt() * SKEW / 2.0;
    let shift = MEAN - shape * scale;
    let z = (x - shift) / scale;
    if z <= 0.0 {
        return 0.0;
    }
    crate::special::gamma::gamma_p(shape, z)
}

/// The inverse participation ratio of a vector: `sum v_i^4 / (sum v_i^2)^2`.
///
/// A measure of how many components carry the weight. A vector concentrated
/// on one component scores 1; one spread evenly over `n` scores `1/n`. For
/// eigenvectors it separates localised states from extended ones, and a GOE
/// eigenvector -- uniform on the sphere -- sits at `3/n`, the extra factor
/// being the fourth moment of a Gaussian.
///
/// Returns zero for a zero vector.
#[must_use]
pub fn participation_ratio(vec: &[f64]) -> f64 {
    let two: f64 = vec.iter().map(|v| v * v).sum();
    if two <= 0.0 {
        return 0.0;
    }
    let four: f64 = vec.iter().map(|v| v * v * v * v).sum();
    four / (two * two)
}

/// The mean ratio of consecutive level spacings,
/// `<min(s_k, s_{k+1}) / max(s_k, s_{k+1})>`.
///
/// The great virtue of this statistic is that it needs no unfolding: a ratio
/// of adjacent gaps is insensitive to the local density, which cancels. That
/// removes the one genuinely arbitrary step in spacing analysis. The limiting
/// values are 0.5307 for the orthogonal class, 0.5996 for the unitary, and
/// `2 ln 2 - 1 = 0.3863` for uncorrelated levels.
///
/// Returns zero for fewer than three eigenvalues.
#[must_use]
pub fn level_spacing_ratio(eigs: &[f64]) -> f64 {
    if eigs.len() < 3 {
        return 0.0;
    }
    let mut sorted = eigs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let gaps: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    let mut acc = 0.0;
    let mut used = 0usize;
    for w in gaps.windows(2) {
        let (lo, hi) = (w[0].min(w[1]), w[0].max(w[1]));
        if hi <= 0.0 {
            continue;
        }
        acc += lo / hi;
        used += 1;
    }
    if used == 0 {
        0.0
    } else {
        acc / used as f64
    }
}

/// Cleans a sample correlation matrix by replacing every eigenvalue inside
/// the Marchenko-Pastur band with their common average.
///
/// `t_over_n` is the number of observations divided by the number of
/// variables, so the band is set by `ratio = 1 / t_over_n`. Eigenvalues below
/// the upper edge are indistinguishable from the noise a correlation matrix
/// of independent variables would produce, and estimating each of them
/// separately fits that noise. Replacing them by their mean keeps the trace
/// -- so the cleaned matrix still has unit diagonal on average and remains a
/// correlation matrix -- while discarding the structure that was not there.
///
/// The eigenvalues above the edge, and their eigenvectors, are left alone.
///
/// # Errors
/// Returns an error if the matrix is not square and symmetric, if `t_over_n`
/// is not positive, or if the eigen-decomposition fails to converge.
pub fn correlation_matrix_denoise_mp(corr: &Matrix, t_over_n: f64) -> Result<Matrix, GeomError> {
    if !corr.is_square() || corr.rows == 0 {
        return Err(GeomError::InvalidArgument("denoise requires a square matrix"));
    }
    if !(t_over_n > 0.0) {
        return Err(GeomError::InvalidArgument("denoise requires t_over_n > 0"));
    }
    let n = corr.rows;
    let decomposition = eigen_symmetric(corr, EIG_TOL, EIG_SWEEPS)
        .map_err(|_| GeomError::Degenerate("denoise: eigen-decomposition failed"))?;

    let ratio = 1.0 / t_over_n;
    let (_, upper) = mp_edges(ratio, 1.0);

    let noisy: Vec<usize> =
        (0..n).filter(|&i| decomposition.values[i] < upper).collect();
    if noisy.is_empty() {
        return Ok(corr.clone());
    }
    // Preserving the summed noise eigenvalue keeps the trace, and with it the
    // total variance the matrix accounts for.
    let replacement: f64 =
        noisy.iter().map(|&i| decomposition.values[i]).sum::<f64>() / noisy.len() as f64;

    let mut cleaned = vec![0.0; n];
    for i in 0..n {
        cleaned[i] = if decomposition.values[i] < upper {
            replacement
        } else {
            decomposition.values[i]
        };
    }

    // Rebuild as V diag(cleaned) V'.
    let mut out = Matrix::zeros(n, n);
    for r in 0..n {
        for c in r..n {
            let v: f64 = (0..n)
                .map(|k| decomposition.vectors.get(r, k) * cleaned[k] * decomposition.vectors.get(c, k))
                .sum();
            out.set(r, c, v);
            out.set(c, r, v);
        }
    }
    Ok(out)
}

/// The eigenvalues of a symmetric matrix, sorted ascending.
///
/// A convenience over [`eigen_symmetric`] for the spectral statistics above,
/// which never need the eigenvectors.
///
/// # Errors
/// Returns an error if the matrix is not symmetric or the solver fails.
pub fn symmetric_spectrum(a: &Matrix) -> Result<Vec<f64>, GeomError> {
    let decomposition = eigen_symmetric(a, EIG_TOL, EIG_SWEEPS)
        .map_err(|_| GeomError::Degenerate("symmetric_spectrum: decomposition failed"))?;
    let mut values = decomposition.values;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(values)
}

/// The eigenvalues of a Hermitian matrix held as `(real, imaginary)` parts.
///
/// Uses the standard real embedding: the `2n`-by-`2n` real symmetric matrix
/// `[[Re, -Im], [Im, Re]]` has exactly the eigenvalues of `H`, each appearing
/// twice. Returns the `n` distinct ones by taking every second value of the
/// sorted `2n`, which is what lets a real symmetric solver handle the unitary
/// ensemble without any complex arithmetic.
///
/// # Errors
/// Returns an error if the two halves disagree in shape or the solver fails.
pub fn hermitian_spectrum(re: &Matrix, im: &Matrix) -> Result<Vec<f64>, GeomError> {
    if !re.is_square() || re.rows != im.rows || re.cols != im.cols {
        return Err(GeomError::InvalidArgument("hermitian_spectrum: shape mismatch"));
    }
    let n = re.rows;
    let mut big = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            big.set(i, j, re.get(i, j));
            big.set(i + n, j + n, re.get(i, j));
            big.set(i, j + n, -im.get(i, j));
            big.set(i + n, j, im.get(i, j));
        }
    }
    let all = symmetric_spectrum(&big)?;
    Ok((0..n).map(|k| all[2 * k]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    /// Midpoint integral of `f` over `[a, b]`.
    fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, steps: usize) -> f64 {
        let h = (b - a) / steps as f64;
        (0..steps).map(|k| f(a + (k as f64 + 0.5) * h) * h).sum()
    }

    /// Integral of `f` over `[a, b]` under the substitution
    /// `x = a + (b - a) sin^2(theta)`.
    ///
    /// The spectral densities here vanish like a square root at both edges,
    /// and at `ratio = 1` Marchenko-Pastur additionally picks up an
    /// inverse-square-root singularity where its lower edge reaches zero.
    /// Midpoint quadrature converges at only `h^(1/2)` against that, so a
    /// uniform grid measures the quadrature rather than the density. The
    /// substitution's Jacobian, `2 (b - a) sin cos`, cancels both behaviours
    /// and leaves a smooth integrand.
    fn integrate_sqrt_edges(f: impl Fn(f64) -> f64, a: f64, b: f64, steps: usize) -> f64 {
        let h = std::f64::consts::FRAC_PI_2 / steps as f64;
        (0..steps)
            .map(|k| {
                let theta = (k as f64 + 0.5) * h;
                let (sin, cos) = (theta.sin(), theta.cos());
                let x = a + (b - a) * sin * sin;
                f(x) * (b - a) * 2.0 * sin * cos * h
            })
            .sum()
    }

    // -----------------------------------------------------------------
    // The limiting densities are densities
    // -----------------------------------------------------------------

    #[test]
    fn the_semicircle_is_a_density_with_the_variance_its_radius_implies() {
        for r in [0.5f64, 1.0, 2.0, 3.7] {
            let mass = integrate_sqrt_edges(|x| wigner_semicircle(x, r), -r, r, 200_000);
            assert!((mass - 1.0).abs() < 1e-9, "radius {r} integrated to {mass}");
            // Symmetric, so the mean is zero and the variance is r^2 / 4.
            let mean = integrate_sqrt_edges(|x| x * wigner_semicircle(x, r), -r, r, 200_000);
            assert!(mean.abs() < 1e-9, "radius {r} has mean {mean}");
            let var = integrate_sqrt_edges(|x| x * x * wigner_semicircle(x, r), -r, r, 200_000);
            assert!(
                close(var, r * r / 4.0, 1e-8),
                "radius {r}: variance {var} against r^2/4 = {}",
                r * r / 4.0
            );
            assert_eq!(wigner_semicircle(r * 1.001, r), 0.0);
            assert_eq!(wigner_semicircle(-r, r), 0.0);
        }
    }

    #[test]
    fn marchenko_pastur_is_a_density_on_its_own_support() {
        for &ratio in &[0.05f64, 0.25, 0.5, 0.9, 1.0] {
            for &sigma2 in &[1.0f64, 2.5] {
                let (a, b) = mp_edges(ratio, sigma2);
                let mass =
                    integrate_sqrt_edges(|x| marchenko_pastur(x, ratio, sigma2), a, b, 200_000);
                assert!(
                    (mass - 1.0).abs() < 1e-7,
                    "ratio {ratio}, sigma2 {sigma2} integrated to {mass}"
                );
                // The mean of the sample eigenvalues is the population
                // variance, whatever the aspect ratio -- the noise spreads the
                // spectrum but does not bias its centre.
                let mean = integrate_sqrt_edges(
                    |x| x * marchenko_pastur(x, ratio, sigma2),
                    a,
                    b,
                    200_000,
                );
                assert!(close(mean, sigma2, 1e-7), "ratio {ratio}: mean {mean} against {sigma2}");
            }
        }
        // Past ratio = 1 the matrix is singular and the continuous part
        // carries only 1/ratio of the mass; the rest is an atom at zero.
        let ratio = 2.0;
        let (a, b) = mp_edges(ratio, 1.0);
        let mass = integrate_sqrt_edges(|x| marchenko_pastur(x, ratio, 1.0), a, b, 200_000);
        assert!((mass - 1.0 / ratio).abs() < 1e-7, "ratio {ratio} gave {mass}, not {}", 1.0 / ratio);
    }

    #[test]
    fn the_marchenko_pastur_band_widens_with_the_aspect_ratio() {
        // With far more observations than variables the sample covariance is
        // nearly exact and the band collapses onto the population value.
        let (a, b) = mp_edges(1e-6, 1.0);
        assert!((a - 1.0).abs() < 0.01 && (b - 1.0).abs() < 0.01, "the band did not collapse");
        // As they approach parity the lower edge reaches zero.
        let (a1, b1) = mp_edges(1.0, 1.0);
        assert!(a1.abs() < 1e-12, "the lower edge is {a1}, not zero");
        assert!((b1 - 4.0).abs() < 1e-12, "the upper edge is {b1}, not four");
        // Monotone in between.
        let mut previous = (1.0f64, 1.0f64);
        for k in 1..20 {
            let (a, b) = mp_edges(k as f64 * 0.05, 1.0);
            assert!(a < previous.0 + 1e-12 && b > previous.1 - 1e-12, "the band is not monotone");
            previous = (a, b);
        }
        // Scaling the variance scales both edges.
        let (a2, b2) = mp_edges(0.4, 3.0);
        let (a3, b3) = mp_edges(0.4, 1.0);
        assert!(close(a2, 3.0 * a3, 1e-12) && close(b2, 3.0 * b3, 1e-12));
    }

    #[test]
    fn the_spacing_surmises_are_normalised_with_unit_mean() {
        // Every surmise is stated for spacings unfolded to unit mean density,
        // so each has to integrate to one and have mean one. That is a real
        // constraint on the constants, not a convention: getting either
        // prefactor wrong breaks it.
        for (name, f) in [
            ("GOE", wigner_surmise_goe as fn(f64) -> f64),
            ("GUE", wigner_surmise_gue as fn(f64) -> f64),
            ("Poisson", poisson_spacing as fn(f64) -> f64),
        ] {
            let mass = integrate(f, 0.0, 40.0, 800_000);
            assert!((mass - 1.0).abs() < 1e-7, "{name} integrated to {mass}");
            let mean = integrate(|s| s * f(s), 0.0, 40.0, 800_000);
            assert!((mean - 1.0).abs() < 1e-6, "{name} has mean {mean}");
            assert_eq!(f(-1.0), 0.0, "{name} is non-zero at a negative spacing");
        }
    }

    #[test]
    fn level_repulsion_distinguishes_the_symmetry_classes_at_small_spacing() {
        // The diagnostic content of the surmises is entirely in how they
        // vanish at zero: linearly for the orthogonal class, quadratically for
        // the unitary, not at all for uncorrelated levels.
        for &s in &[1e-3f64, 1e-2, 0.05] {
            assert!(wigner_surmise_gue(s) < wigner_surmise_goe(s));
            assert!(wigner_surmise_goe(s) < poisson_spacing(s));
        }
        // The vanishing rates themselves: f(s)/s tends to pi/2 for GOE and
        // f(s)/s^2 to 32/pi^2 for GUE.
        let pi = std::f64::consts::PI;
        assert!((wigner_surmise_goe(1e-6) / 1e-6 - pi / 2.0).abs() < 1e-9);
        assert!((wigner_surmise_gue(1e-6) / 1e-12 - 32.0 / (pi * pi)).abs() < 1e-6);
        assert!((poisson_spacing(0.0) - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------
    // The ensembles produce the spectra their laws predict
    // -----------------------------------------------------------------

    #[test]
    fn the_goe_spectrum_follows_the_semicircle() {
        let n = 140usize;
        let mut rng = Rng::new(0x060E_0001);
        let mut all = Vec::new();
        for _ in 0..4 {
            let m = goe_sample(n, &mut rng);
            assert!(m.is_symmetric(1e-12), "the sample is not symmetric");
            all.extend(symmetric_spectrum(&m).unwrap());
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // The scaling was chosen so the support is [-2, 2].
        let radius = 2.0;
        let extreme = all[all.len() - 1].abs().max(all[0].abs());
        assert!(extreme < radius * 1.10, "the spectrum reached {extreme}, past the edge");
        assert!(extreme > radius * 0.85, "the spectrum only reached {extreme}");

        // Kolmogorov-Smirnov against the semicircle law, whose cumulative
        // distribution has a closed form.
        let cdf = |x: f64| -> f64 {
            let t = (x / radius).clamp(-1.0, 1.0);
            0.5 + (t * (1.0 - t * t).sqrt() + t.asin()) / std::f64::consts::PI
        };
        let k = all.len() as f64;
        let d = all
            .iter()
            .enumerate()
            .map(|(i, &x)| ((i + 1) as f64 / k - cdf(x)).abs().max((cdf(x) - i as f64 / k).abs()))
            .fold(0.0f64, f64::max);
        assert!(d < 0.05, "the empirical law is {d} away from the semicircle");

        // The second moment is fixed exactly by the scaling: E[tr(H^2)]/n is
        // (n+1)/n, and the semicircle of radius 2 has variance 1.
        let second: f64 = all.iter().map(|x| x * x).sum::<f64>() / k;
        assert!(close(second, 1.0, 0.05), "the second moment is {second}, not one");
        // And the spectrum is symmetric about zero.
        let first: f64 = all.iter().sum::<f64>() / k;
        assert!(first.abs() < 0.05, "the spectrum is off-centre by {first}");
    }

    #[test]
    fn a_wishart_spectrum_stays_inside_the_marchenko_pastur_band() {
        // Every variable here is independent with unit variance, so the
        // population eigenvalues are all exactly 1. Everything the sample
        // shows beyond that is estimation noise, and Marchenko-Pastur says
        // precisely how much of it to expect.
        let (n, p) = (400usize, 100usize);
        let ratio = p as f64 / n as f64;
        let (lo, hi) = mp_edges(ratio, 1.0);
        let mut rng = Rng::new(0x0011_5AA1);
        let s = wishart_sample(n, p, &mut rng);
        let eigs = symmetric_spectrum(&s).unwrap();

        assert!(eigs.iter().all(|&v| v > lo * 0.75), "an eigenvalue fell below the band");
        assert!(eigs.iter().all(|&v| v < hi * 1.10), "an eigenvalue rose above the band");
        // The band is wide: at this ratio a purely noisy covariance still
        // spreads its eigenvalues over more than a factor of four.
        assert!(hi / lo > 4.0, "the band is implausibly tight");
        assert!(
            eigs[eigs.len() - 1] > 1.3,
            "no eigenvalue exceeded 1.3, so the noise spread is missing"
        );
        assert!(eigs[0] < 0.75, "no eigenvalue fell below 0.75");

        // The trace is p times the population variance, up to sampling error.
        let mean: f64 = eigs.iter().sum::<f64>() / p as f64;
        assert!(close(mean, 1.0, 0.05), "the mean eigenvalue is {mean}");
    }

    #[test]
    fn the_ginibre_ensemble_is_not_symmetric_and_fills_a_disc() {
        let n = 120usize;
        let mut rng = Rng::new(0x0061_81BE);
        let m = ginibre_sample(n, &mut rng);
        assert!(!m.is_symmetric(1e-6), "an unconstrained matrix came out symmetric");

        // The circular law: eigenvalues spread over the unit disc rather than
        // an interval, which is the qualitative break from the symmetric case.
        let eigs = crate::numerical::roots::polynomial_roots(&[1.0, 0.0]).ok();
        assert!(eigs.is_some(), "sanity check on the root finder");
        let spectrum = crate::linalg::eigen::eigenvalues_general(&m, 5000).unwrap();
        assert_eq!(spectrum.len(), n);
        let moduli: Vec<f64> = spectrum.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).collect();
        let inside = moduli.iter().filter(|&&r| r <= 1.05).count();
        assert!(
            inside as f64 / n as f64 > 0.9,
            "only {inside} of {n} eigenvalues landed in the disc"
        );
        // A genuinely complex spectrum: a symmetric matrix would give none.
        let complex = spectrum.iter().filter(|z| z.im.abs() > 1e-6).count();
        assert!(complex > n / 2, "only {complex} eigenvalues were complex");
        // Under the circular law the mean squared modulus is 1/2.
        let mean_sq: f64 = moduli.iter().map(|r| r * r).sum::<f64>() / n as f64;
        assert!(close(mean_sq, 0.5, 0.2), "mean squared modulus {mean_sq}, not near a half");
    }

    // -----------------------------------------------------------------
    // Local statistics
    // -----------------------------------------------------------------

    #[test]
    fn unfolded_spacings_have_unit_mean_by_construction() {
        let mut rng = Rng::new(0x0011_F01D);
        let eigs = symmetric_spectrum(&goe_sample(100, &mut rng)).unwrap();
        let gaps = eigenvalue_spacing_distribution(&eigs);
        assert!(!gaps.is_empty());
        let mean: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
        assert!((mean - 1.0).abs() < 1e-12, "the mean spacing is {mean}");
        assert!(gaps.iter().all(|&g| g >= 0.0), "a spacing came out negative");
        // Too few levels to unfold at all.
        assert!(eigenvalue_spacing_distribution(&[1.0, 2.0]).is_empty());
        assert!(eigenvalue_spacing_distribution(&[]).is_empty());
    }

    #[test]
    fn the_spacing_ratio_separates_correlated_spectra_from_uncorrelated_ones() {
        // The ratio of adjacent gaps needs no unfolding -- the local density
        // cancels between numerator and denominator -- which removes the one
        // arbitrary step in spacing analysis. Its limits are 0.5307 for the
        // orthogonal class and 2 ln 2 - 1 for uncorrelated levels.
        // Averaged per matrix rather than over a pooled spectrum: concatenating
        // two independent spectra manufactures a spurious gap where they join,
        // and that gap has no level correlations at all.
        let mut rng = Rng::new(0x002A_7105);
        let mut ratios = Vec::new();
        for _ in 0..6 {
            let eigs = symmetric_spectrum(&goe_sample(120, &mut rng)).unwrap();
            ratios.push(level_spacing_ratio(&eigs));
        }
        let observed: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!(
            (observed - 0.5307).abs() < 0.04,
            "GOE gave a spacing ratio of {observed}, not 0.5307"
        );

        // Uncorrelated levels: a sorted sample of independent points.
        let mut rng = Rng::new(0x002A_9015);
        let mut poisson_ratios = Vec::new();
        for _ in 0..6 {
            let mut pts: Vec<f64> = (0..120).map(|_| rng.next_f64()).collect();
            pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            poisson_ratios.push(level_spacing_ratio(&pts));
        }
        let uncorrelated: f64 =
            poisson_ratios.iter().sum::<f64>() / poisson_ratios.len() as f64;
        let expected = 2.0 * 2.0f64.ln() - 1.0;
        assert!(
            (uncorrelated - expected).abs() < 0.04,
            "independent points gave {uncorrelated}, not {expected}"
        );
        assert!(observed > uncorrelated + 0.08, "the two classes were not separated");
    }

    #[test]
    fn the_spacing_ratio_is_invariant_to_shifting_and_scaling_the_spectrum() {
        // The property that makes the statistic worth having: it depends on
        // the level correlations and not on the density, so any affine
        // relabelling of the spectrum leaves it alone.
        let mut rng = Rng::new(0x002A_1117);
        let eigs = symmetric_spectrum(&goe_sample(120, &mut rng)).unwrap();
        let base = level_spacing_ratio(&eigs);
        for &(shift, scale) in &[(0.0, 3.7), (100.0, 1.0), (-4.0, 0.02)] {
            let moved: Vec<f64> = eigs.iter().map(|v| shift + scale * v).collect();
            assert!(
                (level_spacing_ratio(&moved) - base).abs() < 1e-9,
                "shift {shift}, scale {scale} changed the ratio"
            );
        }
        assert_eq!(level_spacing_ratio(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn the_unitary_class_repels_harder_than_the_orthogonal_one() {
        // Two ensembles, one statistic: 0.5996 against 0.5307. The gap is
        // small but the direction is the whole content of the symmetry
        // classification.
        let mut rng = Rng::new(0x060E_2A11);
        let mut gue = Vec::new();
        for _ in 0..8 {
            let (re, im) = gue_sample(60, &mut rng);
            gue.push(level_spacing_ratio(&hermitian_spectrum(&re, &im).unwrap()));
        }
        let unitary: f64 = gue.iter().sum::<f64>() / gue.len() as f64;

        let mut rng = Rng::new(0x060E_0E21);
        let mut goe = Vec::new();
        for _ in 0..8 {
            goe.push(level_spacing_ratio(&symmetric_spectrum(&goe_sample(60, &mut rng)).unwrap()));
        }
        let orthogonal: f64 = goe.iter().sum::<f64>() / goe.len() as f64;

        assert!(
            (unitary - 0.5996).abs() < 0.05,
            "GUE gave {unitary}, not 0.5996"
        );
        assert!(unitary > orthogonal, "GUE {unitary} did not exceed GOE {orthogonal}");
    }

    #[test]
    fn rigidity_grows_linearly_for_uncorrelated_levels_and_slowly_for_a_spectrum() {
        // Delta_3 measures correlations across a stretch of levels rather than
        // between neighbours, and separates the two cases far more sharply
        // than the spacing distribution does: L/15 against roughly ln(L)/pi^2.
        let mut rng = Rng::new(0x0021_61D1);
        let mut pts: Vec<f64> = (0..600).map(|_| rng.next_f64()).collect();
        pts.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut rng = Rng::new(0x0021_60E1);
        let eigs = symmetric_spectrum(&goe_sample(180, &mut rng)).unwrap();

        for &l in &[5.0f64, 10.0, 20.0] {
            let uncorrelated = spectral_rigidity(&pts, l);
            let correlated = spectral_rigidity(&eigs, l);
            assert!(
                correlated < uncorrelated,
                "at L = {l} the spectrum ({correlated}) was not more rigid than noise ({uncorrelated})"
            );
            // Uncorrelated levels sit near L/15.
            assert!(
                close(uncorrelated, l / 15.0, 0.6),
                "at L = {l} independent points gave {uncorrelated}, not {}",
                l / 15.0
            );
        }
        // Rigidity grows with the window in both cases, but far faster for
        // uncorrelated levels.
        let growth_noise = spectral_rigidity(&pts, 20.0) / spectral_rigidity(&pts, 5.0);
        let growth_spectrum = spectral_rigidity(&eigs, 20.0) / spectral_rigidity(&eigs, 5.0);
        assert!(
            growth_noise > growth_spectrum,
            "noise grew by {growth_noise} against the spectrum's {growth_spectrum}"
        );
        assert_eq!(spectral_rigidity(&[1.0, 2.0], 5.0), 0.0);
    }

    #[test]
    fn participation_ratio_counts_how_many_components_carry_the_weight() {
        assert!((participation_ratio(&[1.0, 0.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
        let n = 64usize;
        let uniform = vec![1.0 / (n as f64).sqrt(); n];
        assert!(
            (participation_ratio(&uniform) - 1.0 / n as f64).abs() < 1e-12,
            "an evenly spread vector did not score 1/n"
        );
        // Invariant to scaling, since numerator and denominator are both
        // homogeneous of degree four.
        let v = [0.3, -1.2, 0.7, 2.0];
        let base = participation_ratio(&v);
        let scaled: Vec<f64> = v.iter().map(|x| 17.0 * x).collect();
        assert!((participation_ratio(&scaled) - base).abs() < 1e-12);
        assert_eq!(participation_ratio(&[0.0, 0.0]), 0.0);

        // A GOE eigenvector is uniform on the sphere, so its participation
        // ratio is 3/n: the extra factor is the fourth moment of a Gaussian.
        let mut rng = Rng::new(0x9A27_1C1E);
        let n = 100usize;
        let m = goe_sample(n, &mut rng);
        let decomposition = eigen_symmetric(&m, EIG_TOL, EIG_SWEEPS).unwrap();
        let mean: f64 = (0..n)
            .map(|k| {
                let col: Vec<f64> = (0..n).map(|r| decomposition.vectors.get(r, k)).collect();
                participation_ratio(&col)
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            close(mean, 3.0 / n as f64, 0.15),
            "GOE eigenvectors averaged {mean}, not 3/n = {}",
            3.0 / n as f64
        );
    }

    #[test]
    fn the_tracy_widom_approximation_is_a_distribution_with_the_right_cumulants() {
        // Monotone, bounded, and matched to the first three cumulants of the
        // real Tracy-Widom law by construction -- so those are what to check.
        let mut previous = 0.0;
        for k in 0..=600 {
            let x = -8.0 + k as f64 * 0.02;
            let p = tracy_widom_beta1_approx(x);
            assert!((0.0..=1.0).contains(&p), "the value at {x} left [0, 1]: {p}");
            assert!(p >= previous - 1e-12, "the distribution fell at {x}");
            previous = p;
        }
        assert!(tracy_widom_beta1_approx(-20.0) < 1e-6, "the left tail is too heavy");
        assert!(tracy_widom_beta1_approx(6.0) > 0.999, "the right tail did not close");

        // Mean and variance by numerical integration of 1 - F and F.
        let (lo, hi, steps) = (-15.0f64, 12.0f64, 200_000usize);
        let h = (hi - lo) / steps as f64;
        let mut mean = 0.0;
        let mut second = 0.0;
        for k in 0..steps {
            let x = lo + (k as f64 + 0.5) * h;
            // Density by a central difference of the distribution function.
            let d = (tracy_widom_beta1_approx(x + h / 2.0) - tracy_widom_beta1_approx(x - h / 2.0))
                / h;
            mean += x * d * h;
            second += x * x * d * h;
        }
        let variance = second - mean * mean;
        assert!((mean + 1.2065).abs() < 0.01, "the mean is {mean}, not -1.2065");
        assert!((variance - 1.6078).abs() < 0.02, "the variance is {variance}, not 1.6078");
    }

    // -----------------------------------------------------------------
    // Denoising
    // -----------------------------------------------------------------

    /// The sample correlation matrix of `p` series of length `t`.
    fn sample_correlation(data: &[Vec<f64>], p: usize) -> Matrix {
        let t = data.len();
        let means: Vec<f64> =
            (0..p).map(|j| data.iter().map(|r| r[j]).sum::<f64>() / t as f64).collect();
        let sds: Vec<f64> = (0..p)
            .map(|j| {
                (data.iter().map(|r| (r[j] - means[j]).powi(2)).sum::<f64>() / t as f64).sqrt()
            })
            .collect();
        let mut c = Matrix::zeros(p, p);
        for a in 0..p {
            for b in a..p {
                let v: f64 = data
                    .iter()
                    .map(|r| (r[a] - means[a]) * (r[b] - means[b]))
                    .sum::<f64>()
                    / (t as f64 * sds[a] * sds[b]);
                c.set(a, b, v);
                c.set(b, a, v);
            }
        }
        c
    }

    #[test]
    fn denoising_preserves_the_trace_and_flattens_the_noise_band() {
        let (p, t) = (60usize, 240usize);
        let mut rng = Rng::new(0x0DE0_015E);
        let data: Vec<Vec<f64>> =
            (0..t).map(|_| (0..p).map(|_| rng.next_gaussian()).collect()).collect();
        let corr = sample_correlation(&data, p);
        let cleaned = correlation_matrix_denoise_mp(&corr, t as f64 / p as f64).unwrap();

        assert!(cleaned.is_symmetric(1e-9), "the cleaned matrix is not symmetric");
        let trace = |m: &Matrix| (0..p).map(|i| m.get(i, i)).sum::<f64>();
        assert!(
            close(trace(&corr), trace(&cleaned), 1e-9),
            "the trace moved from {} to {}",
            trace(&corr),
            trace(&cleaned)
        );
        assert!(close(trace(&corr), p as f64, 1e-9), "a correlation matrix has trace p");

        // Every variable here is independent, so the whole spectrum is inside
        // the band and cleaning should collapse it to a single value.
        let after = symmetric_spectrum(&cleaned).unwrap();
        let spread = after[after.len() - 1] - after[0];
        let before = symmetric_spectrum(&corr).unwrap();
        assert!(
            spread < 0.05 * (before[before.len() - 1] - before[0]),
            "the noise band was not flattened: spread {spread}"
        );
        // A flat spectrum on a unit-trace-per-variable matrix is the identity.
        for i in 0..p {
            for j in 0..p {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cleaned.get(i, j) - target).abs() < 0.05,
                    "entry ({i}, {j}) came out {}",
                    cleaned.get(i, j)
                );
            }
        }
    }

    #[test]
    fn denoising_keeps_a_factor_that_stands_above_the_band() {
        // One common factor loaded on every series, plus idiosyncratic noise.
        // The factor's eigenvalue is far above the Marchenko-Pastur edge and
        // must survive untouched, while everything below it is flattened.
        let (p, t) = (50usize, 200usize);
        let mut rng = Rng::new(0x0FAC_0801);
        let data: Vec<Vec<f64>> = (0..t)
            .map(|_| {
                let common = rng.next_gaussian();
                (0..p).map(|_| 0.7 * common + 0.7 * rng.next_gaussian()).collect()
            })
            .collect();
        let corr = sample_correlation(&data, p);
        let ratio = p as f64 / t as f64;
        let (_, edge) = mp_edges(ratio, 1.0);

        let before = symmetric_spectrum(&corr).unwrap();
        let top = before[p - 1];
        assert!(top > 3.0 * edge, "the planted factor ({top}) is not clear of the band ({edge})");

        let cleaned = correlation_matrix_denoise_mp(&corr, t as f64 / p as f64).unwrap();
        let after = symmetric_spectrum(&cleaned).unwrap();
        assert!(
            (after[p - 1] - top).abs() < 1e-8,
            "the factor moved from {top} to {}",
            after[p - 1]
        );
        // Everything below the edge is now one repeated value.
        let noise: Vec<f64> = after.iter().copied().filter(|&v| v < edge).collect();
        assert!(noise.len() > p / 2, "too few eigenvalues were treated as noise");
        let spread = noise[noise.len() - 1] - noise[0];
        assert!(spread < 1e-8, "the noise eigenvalues still differ by {spread}");
        // And the trace is still p.
        let trace: f64 = (0..p).map(|i| cleaned.get(i, i)).sum();
        assert!(close(trace, p as f64, 1e-9), "the trace came out {trace}");
    }

    #[test]
    fn denoising_rejects_malformed_input() {
        assert!(correlation_matrix_denoise_mp(&Matrix::zeros(3, 4), 2.0).is_err());
        assert!(correlation_matrix_denoise_mp(&Matrix::identity(3), 0.0).is_err());
        assert!(correlation_matrix_denoise_mp(&Matrix::identity(3), -1.0).is_err());
        // A matrix whose whole spectrum is above the edge is returned as is.
        let big = Matrix::identity(4).scale(9.0);
        let same = correlation_matrix_denoise_mp(&big, 100.0).unwrap();
        assert_eq!(same, big);
    }

    // -----------------------------------------------------------------
    // The Hermitian embedding
    // -----------------------------------------------------------------

    #[test]
    fn the_real_embedding_reproduces_a_hermitian_spectrum() {
        // With a zero imaginary part the embedding must agree with the plain
        // symmetric solver.
        let mut rng = Rng::new(0x04E2_11AA);
        let m = goe_sample(30, &mut rng);
        let direct = symmetric_spectrum(&m).unwrap();
        let embedded = hermitian_spectrum(&m, &Matrix::zeros(30, 30)).unwrap();
        for (a, b) in direct.iter().zip(&embedded) {
            assert!((a - b).abs() < 1e-9, "{a} against {b}");
        }

        // A two-by-two Hermitian with a known answer: [[1, i], [-i, 1]] has
        // eigenvalues 0 and 2.
        let re = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]).unwrap();
        let im = Matrix::from_rows(&[&[0.0, 1.0], &[-1.0, 0.0]]).unwrap();
        let eigs = hermitian_spectrum(&re, &im).unwrap();
        assert!((eigs[0] - 0.0).abs() < 1e-9, "got {}", eigs[0]);
        assert!((eigs[1] - 2.0).abs() < 1e-9, "got {}", eigs[1]);

        // A GUE sample is Hermitian, so its spectrum is real and follows the
        // same semicircle as GOE under the same scaling.
        let (re, im) = gue_sample(60, &mut rng);
        let s = hermitian_spectrum(&re, &im).unwrap();
        assert_eq!(s.len(), 60);
        assert!(s.iter().all(|v| v.is_finite()));
        assert!(s.windows(2).all(|w| w[0] <= w[1]), "the spectrum is not sorted");
        let second: f64 = s.iter().map(|x| x * x).sum::<f64>() / 60.0;
        assert!(close(second, 1.0, 0.25), "the GUE second moment is {second}");

        assert!(hermitian_spectrum(&Matrix::zeros(2, 3), &Matrix::zeros(2, 3)).is_err());
        assert!(hermitian_spectrum(&Matrix::zeros(2, 2), &Matrix::zeros(3, 3)).is_err());
    }

    #[test]
    fn the_ensembles_reject_a_zero_dimension() {
        let mut rng = Rng::new(1);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            goe_sample(0, &mut Rng::new(1))
        }))
        .is_err());
        assert_eq!(goe_sample(1, &mut rng).rows, 1);
        assert_eq!(wishart_sample(5, 3, &mut rng).rows, 3);
        assert_eq!(ginibre_sample(4, &mut rng).cols, 4);
    }
}
