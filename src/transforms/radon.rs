//! Radon transform and tomographic reconstruction, plus Hankel/Abel
//! transforms and Hough voting.
//!
//! Images are row-major (index = y·w + x) with the projection geometry
//! centered on the image; a projection at angle θ integrates along
//! lines perpendicular to the direction (cos θ, sin θ).

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::special::bessel_jn;
use crate::transforms::fft::{fft, ifft, next_power_of_two};

/// Bilinear sample with zero outside the image.
fn sample_bilinear(img: &[f64], w: usize, h: usize, x: f64, y: f64) -> f64 {
    if x < 0.0 || y < 0.0 || x > (w - 1) as f64 || y > (h - 1) as f64 {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let v00 = img[y0 * w + x0];
    let v10 = img[y0 * w + x1];
    let v01 = img[y1 * w + x0];
    let v11 = img[y1 * w + x1];
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Forward Radon transform: one projection (length `n_rays`) per angle
/// (radians). Ray offsets span the image diagonal; line integrals use
/// unit-pixel steps with bilinear interpolation.
///
/// # Panics
/// Panics unless `img.len() == w * h`.
#[must_use]
pub fn radon(img: &[f64], w: usize, h: usize, angles: &[f64], n_rays: usize) -> Vec<Vec<f64>> {
    assert_eq!(img.len(), w * h, "radon expects w*h samples");
    let cx = (w - 1) as f64 / 2.0;
    let cy = (h - 1) as f64 / 2.0;
    let diag = ((w * w + h * h) as f64).sqrt();
    let n_steps = diag.ceil() as usize + 1;
    angles
        .iter()
        .map(|&theta| {
            let (sin_t, cos_t) = theta.sin_cos();
            (0..n_rays)
                .map(|ri| {
                    // Offset along the (cos θ, sin θ) direction.
                    let s = (ri as f64 / (n_rays - 1).max(1) as f64 - 0.5) * diag;
                    let mut acc = 0.0;
                    for ti in 0..n_steps {
                        let t = ti as f64 - diag / 2.0;
                        let x = cx + s * cos_t - t * sin_t;
                        let y = cy + s * sin_t + t * cos_t;
                        acc += sample_bilinear(img, w, h, x, y);
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

/// Filter kernels for filtered back-projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbpFilter {
    RamLak,
    SheppLogan,
    Cosine,
    Hamming,
    Hann,
}

fn fbp_filter_gain(filter: FbpFilter, f: f64) -> f64 {
    // f is |frequency| normalized to [0, 0.5]; Ram-Lak is |f|, the rest
    // taper the high end.
    let ram = f;
    if f == 0.0 {
        return 0.0;
    }
    match filter {
        FbpFilter::RamLak => ram,
        FbpFilter::SheppLogan => ram * (PI * f).sin() / (PI * f),
        FbpFilter::Cosine => ram * (PI * f).cos(),
        FbpFilter::Hamming => ram * (0.54 + 0.46 * (2.0 * PI * f).cos()),
        FbpFilter::Hann => ram * (0.5 + 0.5 * (2.0 * PI * f).cos()),
    }
}

/// Filtered back-projection onto an `out`×`out` image (pixels outside
/// the inscribed circle are zero). `sino` is \[angle\]\[ray\] as produced
/// by [`radon`] on a square image of side `out`.
#[must_use]
pub fn inverse_radon_fbp(
    sino: &[Vec<f64>],
    angles: &[f64],
    out: usize,
    filter: FbpFilter,
) -> Vec<f64> {
    if sino.is_empty() || out == 0 {
        return vec![0.0; out * out];
    }
    let n_rays = sino[0].len();
    let nfft = next_power_of_two(2 * n_rays);
    // Ray spacing in pixels (rays span the diagonal of the out-square).
    let ds = ((2 * out * out) as f64).sqrt() / (n_rays - 1).max(1) as f64;
    // Frequency-domain ramp filtering of every projection.
    let filtered: Vec<Vec<f64>> = sino
        .iter()
        .map(|proj| {
            let mut buf = vec![Complex::new(0.0, 0.0); nfft];
            for (i, &v) in proj.iter().enumerate() {
                buf[i] = Complex::new(v, 0.0);
            }
            let mut spec = fft(&buf);
            for (k, s) in spec.iter_mut().enumerate() {
                let f = if k <= nfft / 2 { k as f64 } else { (nfft - k) as f64 } / nfft as f64;
                let g = fbp_filter_gain(filter, f) / ds;
                *s = Complex::new(s.re * g, s.im * g);
            }
            ifft(&spec).iter().take(n_rays).map(|c| c.re).collect()
        })
        .collect();
    // Back-project.
    let c = (out - 1) as f64 / 2.0;
    let diag = ((2 * out * out) as f64).sqrt();
    let radius = out as f64 / 2.0;
    let mut img = vec![0.0; out * out];
    for (a, &theta) in angles.iter().enumerate() {
        let (sin_t, cos_t) = theta.sin_cos();
        for y in 0..out {
            for x in 0..out {
                let dx = x as f64 - c;
                let dy = y as f64 - c;
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let s = dx * cos_t + dy * sin_t;
                let ray = (s / diag + 0.5) * (n_rays - 1) as f64;
                if ray < 0.0 || ray > (n_rays - 1) as f64 {
                    continue;
                }
                let r0 = ray.floor() as usize;
                let r1 = (r0 + 1).min(n_rays - 1);
                let fr = ray - r0 as f64;
                img[y * out + x] += filtered[a][r0] * (1.0 - fr) + filtered[a][r1] * fr;
            }
        }
    }
    let scale = PI / angles.len() as f64;
    for v in img.iter_mut() {
        *v *= scale;
    }
    img
}

/// Simultaneous algebraic reconstruction (SART): iterate over angles,
/// forward-project the estimate, and back-project the normalized ray
/// residuals.
#[must_use]
pub fn inverse_radon_sart(
    sino: &[Vec<f64>],
    angles: &[f64],
    out: usize,
    iters: usize,
) -> Vec<f64> {
    if sino.is_empty() || out == 0 {
        return vec![0.0; out * out];
    }
    let n_rays = sino[0].len();
    let mut img = vec![0.0; out * out];
    let diag = ((2 * out * out) as f64).sqrt();
    let n_steps = diag.ceil() as usize + 1;
    let cx = (out - 1) as f64 / 2.0;
    let relax = 0.25;
    for _ in 0..iters {
        for (a, &theta) in angles.iter().enumerate() {
            let (sin_t, cos_t) = theta.sin_cos();
            for (ri, &target) in sino[a].iter().enumerate() {
                let s = (ri as f64 / (n_rays - 1).max(1) as f64 - 0.5) * diag;
                // Forward project this ray and remember the touched pixels.
                let mut acc = 0.0;
                let mut touched: Vec<(usize, f64)> = Vec::new();
                for ti in 0..n_steps {
                    let t = ti as f64 - diag / 2.0;
                    let x = cx + s * cos_t - t * sin_t;
                    let y = cx + s * sin_t + t * cos_t;
                    if x < 0.0 || y < 0.0 || x > (out - 1) as f64 || y > (out - 1) as f64 {
                        continue;
                    }
                    let x0 = x.floor() as usize;
                    let y0 = y.floor() as usize;
                    let x1 = (x0 + 1).min(out - 1);
                    let y1 = (y0 + 1).min(out - 1);
                    let fx = x - x0 as f64;
                    let fy = y - y0 as f64;
                    let weights = [
                        (y0 * out + x0, (1.0 - fx) * (1.0 - fy)),
                        (y0 * out + x1, fx * (1.0 - fy)),
                        (y1 * out + x0, (1.0 - fx) * fy),
                        (y1 * out + x1, fx * fy),
                    ];
                    for &(idx, wgt) in &weights {
                        acc += img[idx] * wgt;
                        if wgt > 0.0 {
                            touched.push((idx, wgt));
                        }
                    }
                }
                if touched.is_empty() {
                    continue;
                }
                let wsum: f64 = touched.iter().map(|&(_, w)| w).sum();
                let residual = (target - acc) / wsum;
                for &(idx, wgt) in &touched {
                    img[idx] += relax * residual * wgt;
                }
            }
        }
    }
    img
}

/// The classic Shepp-Logan head phantom on an n×n grid (values in the
/// original low-contrast scale).
#[must_use]
pub fn shepp_logan_phantom(n: usize) -> Vec<f64> {
    // (additive intensity, a, b, x0, y0, phi degrees)
    const ELLIPSES: [(f64, f64, f64, f64, f64, f64); 10] = [
        (2.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.98, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
        (-0.02, 0.1100, 0.3100, 0.22, 0.0, -18.0),
        (-0.02, 0.1600, 0.4100, -0.22, 0.0, 18.0),
        (0.01, 0.2100, 0.2500, 0.0, 0.35, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, 0.1, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, -0.1, 0.0),
        (0.01, 0.0460, 0.0230, -0.08, -0.605, 0.0),
        (0.01, 0.0230, 0.0230, 0.0, -0.606, 0.0),
        (0.01, 0.0230, 0.0460, 0.06, -0.605, 0.0),
    ];
    let mut img = vec![0.0; n * n];
    let c = (n - 1) as f64 / 2.0;
    for y in 0..n {
        for x in 0..n {
            // Normalized coordinates in [-1, 1] (y up).
            let xn = (x as f64 - c) / (n as f64 / 2.0);
            let yn = -(y as f64 - c) / (n as f64 / 2.0);
            let mut v = 0.0;
            for &(a_int, a, b, x0, y0, phi) in &ELLIPSES {
                let p = phi * PI / 180.0;
                let (sp, cp) = p.sin_cos();
                let dx = xn - x0;
                let dy = yn - y0;
                let u = dx * cp + dy * sp;
                let w = -dx * sp + dy * cp;
                if (u / a).powi(2) + (w / b).powi(2) <= 1.0 {
                    v += a_int;
                }
            }
            img[y * n + x] = v;
        }
    }
    img
}

/// Hankel transform of order `order`: ∫₀^rmax f(r)·J_ν(k·r)·r dr by
/// composite Simpson quadrature with n panels.
#[must_use]
pub fn hankel_transform(f: &dyn Fn(f64) -> f64, k: f64, order: u32, r_max: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n.max(2) } else { n + 1 };
    let h = r_max / n as f64;
    let g = |r: f64| f(r) * bessel_jn(order, k * r) * r;
    let mut acc = g(0.0) + g(r_max);
    for i in 1..n {
        let w = if !i.is_multiple_of(2) { 4.0 } else { 2.0 };
        acc += w * g(i as f64 * h);
    }
    acc * h / 3.0
}

/// Forward Abel transform F(y) = 2∫_y^rmax f(r)·r/√(r²−y²) dr, computed
/// singularity-free with the substitution r = √(y² + u²).
#[must_use]
pub fn abel_transform(f: &dyn Fn(f64) -> f64, y: f64, r_max: f64, n: usize) -> f64 {
    if y >= r_max {
        return 0.0;
    }
    let u_max = (r_max * r_max - y * y).sqrt();
    let n = if n.is_multiple_of(2) { n.max(2) } else { n + 1 };
    let h = u_max / n as f64;
    let g = |u: f64| f((y * y + u * u).sqrt());
    let mut acc = g(0.0) + g(u_max);
    for i in 1..n {
        let w = if !i.is_multiple_of(2) { 4.0 } else { 2.0 };
        acc += w * g(i as f64 * h);
    }
    2.0 * acc * h / 3.0
}

/// Inverse Abel transform of a projection sampled at y_i = i·dr:
/// f(r) = −(1/π)∫_r^R F′(y)/√(y²−r²) dy, with a central-difference F′
/// and the same singularity-removing substitution.
#[must_use]
pub fn inverse_abel(data: &[f64], dr: f64) -> Vec<f64> {
    let n = data.len();
    if n < 3 {
        return vec![0.0; n];
    }
    // F' on the sample grid.
    let deriv: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                (data[1] - data[0]) / dr
            } else if i == n - 1 {
                (data[n - 1] - data[n - 2]) / dr
            } else {
                (data[i + 1] - data[i - 1]) / (2.0 * dr)
            }
        })
        .collect();
    let r_max = (n - 1) as f64 * dr;
    let dlin = |y: f64| -> f64 {
        let t = (y / dr).clamp(0.0, (n - 1) as f64);
        let i = t.floor() as usize;
        let j = (i + 1).min(n - 1);
        let f = t - i as f64;
        deriv[i] * (1.0 - f) + deriv[j] * f
    };
    (0..n)
        .map(|ri| {
            let r = ri as f64 * dr;
            if r >= r_max {
                return 0.0;
            }
            // y = sqrt(r² + u²), dy = u/y du → integrand F'(y)/y du... times y/√(y²−r²)=y/u:
            // ∫ F'(y)/√(y²−r²) dy = ∫ F'(√(r²+u²)) du / ... substitute:
            // dy/√(y²−r²) = du/y·y/u·u = du·(1/y)·y = du... careful: dy = (u/y)du and
            // √(y²−r²) = u, so dy/√(y²−r²) = du/y.
            let u_max = (r_max * r_max - r * r).sqrt();
            let m = 200usize;
            let h = u_max / m as f64;
            let g = |u: f64| {
                let y = (r * r + u * u).sqrt();
                if y <= 1e-12 {
                    0.0
                } else {
                    dlin(y) / y
                }
            };
            let mut acc = g(0.0) + g(u_max);
            for i in 1..m {
                let w = if !i.is_multiple_of(2) { 4.0 } else { 2.0 };
                acc += w * g(i as f64 * h);
            }
            -(acc * h / 3.0) / PI
        })
        .collect()
}

/// Hough line accumulator: votes\[θ\]\[ρ\] with θ over \[0, π) in
/// `n_theta` steps and ρ over \[−D, D\] (D = image diagonal) in `n_rho`
/// bins.
#[must_use]
pub fn hough_lines(
    edges: &[bool],
    w: usize,
    h: usize,
    n_theta: usize,
    n_rho: usize,
) -> Vec<Vec<u32>> {
    let diag = ((w * w + h * h) as f64).sqrt();
    let mut acc = vec![vec![0u32; n_rho]; n_theta];
    for y in 0..h {
        for x in 0..w {
            if !edges[y * w + x] {
                continue;
            }
            for (ti, row) in acc.iter_mut().enumerate() {
                let theta = PI * ti as f64 / n_theta as f64;
                let rho = x as f64 * theta.cos() + y as f64 * theta.sin();
                let bin = ((rho / diag + 1.0) / 2.0 * (n_rho - 1) as f64).round() as usize;
                if bin < n_rho {
                    row[bin] += 1;
                }
            }
        }
    }
    acc
}

/// Hough circle detection: returns candidate (cx, cy, r, votes) sorted
/// by votes, keeping local maxima with at least half the top vote.
#[must_use]
pub fn hough_circles(
    edges: &[bool],
    w: usize,
    h: usize,
    r_min: usize,
    r_max: usize,
) -> Vec<(usize, usize, usize, u32)> {
    let mut results: Vec<(usize, usize, usize, u32)> = Vec::new();
    let mut best = 0u32;
    for r in r_min..=r_max {
        let mut acc = vec![0u32; w * h];
        let n_steps = (2.0 * PI * r as f64).ceil() as usize;
        for y in 0..h {
            for x in 0..w {
                if !edges[y * w + x] {
                    continue;
                }
                for s in 0..n_steps {
                    let ang = 2.0 * PI * s as f64 / n_steps as f64;
                    let cx = x as f64 - r as f64 * ang.cos();
                    let cy = y as f64 - r as f64 * ang.sin();
                    if cx >= 0.0 && cy >= 0.0 {
                        let (cxi, cyi) = (cx.round() as usize, cy.round() as usize);
                        if cxi < w && cyi < h {
                            acc[cyi * w + cxi] += 1;
                        }
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let v = acc[y * w + x];
                if v > 0 {
                    best = best.max(v);
                    results.push((x, y, r, v));
                }
            }
        }
    }
    let cutoff = best / 2;
    let mut peaks: Vec<(usize, usize, usize, u32)> =
        results.into_iter().filter(|&(_, _, _, v)| v > cutoff && v >= 8).collect();
    peaks.sort_by(|a, b| b.3.cmp(&a.3));
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radon_uniform_disk_projection() {
        // The projection of a uniform disk is 2√(R²−s²): check the peak.
        let n = 64;
        let mut img = vec![0.0; n * n];
        let c = (n - 1) as f64 / 2.0;
        let radius = 20.0;
        for y in 0..n {
            for x in 0..n {
                if (x as f64 - c).powi(2) + (y as f64 - c).powi(2) <= radius * radius {
                    img[y * n + x] = 1.0;
                }
            }
        }
        let n_rays = 95;
        let sino = radon(&img, n, n, &[0.0, PI / 4.0, PI / 2.0], n_rays);
        for proj in &sino {
            let peak = proj.iter().cloned().fold(0.0_f64, f64::max);
            assert!((peak - 2.0 * radius).abs() < 2.0, "peak {peak}");
            // Symmetric about the center ray.
            let mid = n_rays / 2;
            for k in 1..10 {
                assert!((proj[mid - k] - proj[mid + k]).abs() < 1.5);
            }
        }
    }

    #[test]
    fn test_fbp_phantom_psnr() {
        let n = 128;
        let phantom = shepp_logan_phantom(n);
        let angles: Vec<f64> = (0..180).map(|i| PI * i as f64 / 180.0).collect();
        let n_rays = 2 * ((n as f64) * std::f64::consts::SQRT_2).ceil() as usize + 1;
        let sino = radon(&phantom, n, n, &angles, n_rays);
        let rec = inverse_radon_fbp(&sino, &angles, n, FbpFilter::RamLak);
        // PSNR inside the inscribed circle.
        let c = (n - 1) as f64 / 2.0;
        let radius = n as f64 / 2.0 - 2.0;
        let mut mse = 0.0;
        let mut count = 0usize;
        let mut peak = 0.0_f64;
        for y in 0..n {
            for x in 0..n {
                let dx = x as f64 - c;
                let dy = y as f64 - c;
                if dx * dx + dy * dy < radius * radius {
                    let d = rec[y * n + x] - phantom[y * n + x];
                    mse += d * d;
                    count += 1;
                    peak = peak.max(phantom[y * n + x].abs());
                }
            }
        }
        mse /= count as f64;
        let psnr = 10.0 * (peak * peak / mse).log10();
        assert!(psnr > 25.0, "PSNR {psnr} dB");
    }

    #[test]
    fn test_sart_reconstructs_disk() {
        let n = 32;
        let mut img = vec![0.0; n * n];
        let c = (n - 1) as f64 / 2.0;
        for y in 0..n {
            for x in 0..n {
                if (x as f64 - c).powi(2) + (y as f64 - c).powi(2) <= 64.0 {
                    img[y * n + x] = 1.0;
                }
            }
        }
        let angles: Vec<f64> = (0..36).map(|i| PI * i as f64 / 36.0).collect();
        let sino = radon(&img, n, n, &angles, 47);
        let rec = inverse_radon_sart(&sino, &angles, n, 8);
        // Center should be near 1, far corner near 0.
        let center = rec[(n / 2) * n + n / 2];
        assert!((center - 1.0).abs() < 0.25, "center {center}");
        assert!(rec[0].abs() < 0.25, "corner {}", rec[0]);
    }

    #[test]
    fn test_phantom_values() {
        let n = 128;
        let p = shepp_logan_phantom(n);
        // Center is inside the two big ellipses: 2 − 0.98 = 1.02.
        let center = p[(n / 2) * n + n / 2];
        assert!((center - 1.02).abs() < 1e-9, "{center}");
        // Corner outside the skull: 0.
        assert_eq!(p[0], 0.0);
    }

    #[test]
    fn test_hankel_gaussian() {
        // ∫₀^∞ e^{-r²/2} J₀(kr) r dr = e^{-k²/2}
        let f = |r: f64| (-r * r / 2.0).exp();
        for &k in &[0.5, 1.0, 2.0] {
            let v = hankel_transform(&f, k, 0, 12.0, 2000);
            let expect = (-k * k / 2.0).exp();
            assert!((v - expect).abs() < 1e-8, "k={k}: {v} vs {expect}");
        }
    }

    #[test]
    fn test_abel_gaussian_roundtrip() {
        // Abel of e^{-r²/σ²} is σ√π e^{-y²/σ²}.
        let sigma = 1.0_f64;
        let f = |r: f64| (-r * r / (sigma * sigma)).exp();
        let v = abel_transform(&f, 0.5, 10.0, 2000);
        let expect = sigma * PI.sqrt() * (-0.25 / (sigma * sigma)).exp();
        assert!((v - expect).abs() < 1e-6, "{v} vs {expect}");
        // Inverse recovers the original profile.
        let dr = 0.02;
        let n = 400;
        let proj: Vec<f64> = (0..n)
            .map(|i| sigma * PI.sqrt() * (-(i as f64 * dr).powi(2) / (sigma * sigma)).exp())
            .collect();
        let rec = inverse_abel(&proj, dr);
        for &ri in &[10usize, 25, 50, 100] {
            let r = ri as f64 * dr;
            let expect = (-r * r / (sigma * sigma)).exp();
            assert!(
                (rec[ri] - expect).abs() < 0.02,
                "r={r}: {} vs {expect}",
                rec[ri]
            );
        }
    }

    #[test]
    fn test_hough_lines_finds_horizontal() {
        let (w, h) = (64, 64);
        let mut edges = vec![false; w * h];
        for x in 0..w {
            edges[20 * w + x] = true; // y = 20
        }
        let acc = hough_lines(&edges, w, h, 90, 128);
        // Peak should be at θ = π/2 (normal points along y), ρ = 20.
        let mut best = (0, 0, 0u32);
        for (t, row) in acc.iter().enumerate() {
            for (r, &v) in row.iter().enumerate() {
                if v > best.2 {
                    best = (t, r, v);
                }
            }
        }
        let theta = PI * best.0 as f64 / 90.0;
        let diag = ((w * w + h * h) as f64).sqrt();
        let rho = (best.1 as f64 / 127.0 * 2.0 - 1.0) * diag;
        assert!((theta - PI / 2.0).abs() < 0.05, "theta {theta}");
        assert!((rho - 20.0).abs() < 1.5, "rho {rho}");
        assert_eq!(best.2, w as u32);
    }

    #[test]
    fn test_hough_circles_finds_circle() {
        let (w, h) = (48, 48);
        let mut edges = vec![false; w * h];
        let (cx, cy, r) = (24.0_f64, 20.0_f64, 10.0_f64);
        for s in 0..720 {
            let a = 2.0 * PI * s as f64 / 720.0;
            let x = (cx + r * a.cos()).round() as isize;
            let y = (cy + r * a.sin()).round() as isize;
            if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                edges[y as usize * w + x as usize] = true;
            }
        }
        let found = hough_circles(&edges, w, h, 8, 12);
        assert!(!found.is_empty());
        let (fx, fy, fr, _) = found[0];
        assert!((fx as f64 - cx).abs() <= 1.0);
        assert!((fy as f64 - cy).abs() <= 1.0);
        assert!((fr as f64 - r).abs() <= 1.0);
    }
}
