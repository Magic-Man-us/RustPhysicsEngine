//! Fast Fourier transforms.
//!
//! The power-of-two core is the iterative radix-2 Cooley-Tukey from
//! Press et al., *Numerical Recipes*, §12.2. [`fft_any`] extends it to
//! arbitrary lengths with a recursive mixed-radix 2/3/5 decomposition and
//! a Bluestein chirp-z fallback for lengths with other prime factors.

use crate::error::SolveError;
use crate::fractals::Complex;
use crate::math::constants::PI;

/// Smallest power of two ≥ n (returns 1 for n = 0).
#[must_use]
pub fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two().max(1)
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

#[inline]
fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

/// In-place iterative radix-2 FFT core. `sign` is −1 for the forward
/// transform and +1 for the inverse (no 1/n scaling here).
fn fft_in_place(buf: &mut [Complex], sign: f64) {
    let n = buf.len();
    // Bit-reversal permutation.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Danielson-Lanczos butterflies.
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let wlen = cis(ang);
        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = buf[start + k];
                let v = buf[start + k + len / 2] * w;
                buf[start + k] = u + v;
                buf[start + k + len / 2] = u - v;
                w = w * wlen;
            }
        }
        len <<= 1;
    }
}

/// Forward FFT: X[k] = Σ x[n]·e^(−j2πkn/N).
///
/// # Panics
/// Panics unless `input.len()` is a power of two. Use [`fft_any`] for
/// arbitrary lengths.
#[must_use]
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    assert!(is_power_of_two(input.len()), "fft requires a power-of-two length");
    let mut buf = input.to_vec();
    fft_in_place(&mut buf, -1.0);
    buf
}

/// Inverse FFT: x[n] = (1/N)·Σ X[k]·e^(j2πkn/N).
///
/// # Panics
/// Panics unless `input.len()` is a power of two. Use [`ifft_any`] for
/// arbitrary lengths.
#[must_use]
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    assert!(is_power_of_two(input.len()), "ifft requires a power-of-two length");
    let mut buf = input.to_vec();
    fft_in_place(&mut buf, 1.0);
    let inv_n = 1.0 / buf.len() as f64;
    for v in buf.iter_mut() {
        *v = Complex::new(v.re * inv_n, v.im * inv_n);
    }
    buf
}

/// Strip the factors 2, 3, and 5 out of n; returns the co-factor.
fn residual_after_235(mut n: usize) -> usize {
    for p in [2, 3, 5] {
        while n.is_multiple_of(p) {
            n /= p;
        }
    }
    n
}

/// Recursive mixed-radix decimation-in-time for lengths whose prime
/// factors are all in {2, 3, 5}. Unscaled; `sign` as in [`fft_in_place`].
fn fft_mixed_radix(x: &[Complex], sign: f64) -> Vec<Complex> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    if is_power_of_two(n) {
        let mut buf = x.to_vec();
        fft_in_place(&mut buf, sign);
        return buf;
    }
    let p = if n.is_multiple_of(2) {
        2
    } else if n.is_multiple_of(3) {
        3
    } else {
        5
    };
    let m = n / p;
    let subs: Vec<Vec<Complex>> = (0..p)
        .map(|r| {
            let sub: Vec<Complex> = (0..m).map(|a| x[a * p + r]).collect();
            fft_mixed_radix(&sub, sign)
        })
        .collect();
    let mut out = vec![ZERO; n];
    let base = sign * 2.0 * PI / n as f64;
    for s in 0..p {
        for k in 0..m {
            let idx = k + s * m;
            let mut acc = ZERO;
            for (r, sub) in subs.iter().enumerate() {
                acc = acc + sub[k] * cis(base * (r * idx) as f64);
            }
            out[idx] = acc;
        }
    }
    out
}

/// Bluestein chirp-z algorithm: DFT of arbitrary length n via a linear
/// convolution of size ≥ 2n−1 done with the power-of-two FFT. Unscaled.
fn fft_bluestein(x: &[Complex], sign: f64) -> Vec<Complex> {
    let n = x.len();
    // Chirp w_j = e^(sign·jπ j²/n); reduce j² mod 2n to keep the angle small.
    let chirp: Vec<Complex> = (0..n)
        .map(|j| cis(sign * PI * ((j * j) % (2 * n)) as f64 / n as f64))
        .collect();
    let m = next_power_of_two(2 * n - 1);
    let mut a = vec![ZERO; m];
    for j in 0..n {
        a[j] = x[j] * chirp[j];
    }
    // b_j = conj(chirp_j) laid out for circular convolution: b[0..n) and
    // the mirrored negative indices at the end of the buffer.
    let mut b = vec![ZERO; m];
    for j in 0..n {
        let c = chirp[j].conjugate();
        b[j] = c;
        if j != 0 {
            b[m - j] = c;
        }
    }
    fft_in_place(&mut a, -1.0);
    fft_in_place(&mut b, -1.0);
    for (u, v) in a.iter_mut().zip(&b) {
        *u = *u * *v;
    }
    let conv = ifft(&a);
    (0..n).map(|k| conv[k] * chirp[k]).collect()
}

fn dft_dispatch(x: &[Complex], sign: f64) -> Vec<Complex> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    if residual_after_235(n) == 1 {
        fft_mixed_radix(x, sign)
    } else {
        fft_bluestein(x, sign)
    }
}

/// Forward DFT of any length: mixed radix 2/3/5 with a Bluestein
/// fallback for lengths containing other prime factors. O(n log n).
#[must_use]
pub fn fft_any(x: &[Complex]) -> Vec<Complex> {
    dft_dispatch(x, -1.0)
}

/// Inverse DFT of any length (includes the 1/n scaling).
#[must_use]
pub fn ifft_any(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    let mut out = dft_dispatch(x, 1.0);
    let inv_n = 1.0 / n as f64;
    for v in out.iter_mut() {
        *v = Complex::new(v.re * inv_n, v.im * inv_n);
    }
    out
}

/// FFT of a real signal, returning the n/2 + 1 non-redundant bins
/// (bins k > n/2 satisfy X[n−k] = X[k]*). Any length.
#[must_use]
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let buf: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut full = fft_any(&buf);
    full.truncate(n / 2 + 1);
    full
}

/// Inverse of [`rfft`]: rebuilds the full conjugate-symmetric spectrum
/// and returns the length-n real signal.
///
/// # Panics
/// Panics unless `x.len() == n / 2 + 1`.
#[must_use]
pub fn irfft(x: &[Complex], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(x.len(), n / 2 + 1, "irfft expects n/2 + 1 bins");
    let mut full = vec![ZERO; n];
    full[..x.len()].copy_from_slice(x);
    for k in x.len()..n {
        full[k] = x[n - k].conjugate();
    }
    ifft_any(&full).iter().map(|c| c.re).collect()
}

/// 2D FFT of row-major data (index = y·w + x): transform rows, then columns.
///
/// # Panics
/// Panics unless `x.len() == w * h`.
#[must_use]
pub fn fft_2d(x: &[Complex], w: usize, h: usize) -> Vec<Complex> {
    fft_2d_dir(x, w, h, false)
}

/// Inverse 2D FFT (includes the 1/(w·h) scaling).
///
/// # Panics
/// Panics unless `x.len() == w * h`.
#[must_use]
pub fn ifft_2d(x: &[Complex], w: usize, h: usize) -> Vec<Complex> {
    fft_2d_dir(x, w, h, true)
}

fn fft_2d_dir(x: &[Complex], w: usize, h: usize, inverse: bool) -> Vec<Complex> {
    assert_eq!(x.len(), w * h, "fft_2d expects w*h samples");
    let xform = if inverse { ifft_any } else { fft_any };
    let mut data = x.to_vec();
    for row in 0..h {
        let out = xform(&data[row * w..(row + 1) * w]);
        data[row * w..(row + 1) * w].copy_from_slice(&out);
    }
    let mut col = vec![ZERO; h];
    for cx in 0..w {
        for (cy, v) in col.iter_mut().enumerate() {
            *v = data[cy * w + cx];
        }
        let out = xform(&col);
        for (cy, v) in out.iter().enumerate() {
            data[cy * w + cx] = *v;
        }
    }
    data
}

/// 3D FFT of data indexed as `(z·ny + y)·nx + x`.
///
/// # Panics
/// Panics unless `x.len() == nx * ny * nz`.
#[must_use]
pub fn fft_3d(x: &[Complex], nx: usize, ny: usize, nz: usize) -> Vec<Complex> {
    fft_3d_dir(x, nx, ny, nz, false)
}

/// Inverse 3D FFT (includes the 1/(nx·ny·nz) scaling).
///
/// # Panics
/// Panics unless `x.len() == nx * ny * nz`.
#[must_use]
pub fn ifft_3d(x: &[Complex], nx: usize, ny: usize, nz: usize) -> Vec<Complex> {
    fft_3d_dir(x, nx, ny, nz, true)
}

fn fft_3d_dir(x: &[Complex], nx: usize, ny: usize, nz: usize, inverse: bool) -> Vec<Complex> {
    assert_eq!(x.len(), nx * ny * nz, "fft_3d expects nx*ny*nz samples");
    let xform = if inverse { ifft_any } else { fft_any };
    let mut data = x.to_vec();
    // Along x.
    for z in 0..nz {
        for y in 0..ny {
            let start = (z * ny + y) * nx;
            let out = xform(&data[start..start + nx]);
            data[start..start + nx].copy_from_slice(&out);
        }
    }
    // Along y.
    let mut line = vec![ZERO; ny];
    for z in 0..nz {
        for x_i in 0..nx {
            for (y, v) in line.iter_mut().enumerate() {
                *v = data[(z * ny + y) * nx + x_i];
            }
            let out = xform(&line);
            for (y, v) in out.iter().enumerate() {
                data[(z * ny + y) * nx + x_i] = *v;
            }
        }
    }
    // Along z.
    let mut line = vec![ZERO; nz];
    for y in 0..ny {
        for x_i in 0..nx {
            for (z, v) in line.iter_mut().enumerate() {
                *v = data[(z * ny + y) * nx + x_i];
            }
            let out = xform(&line);
            for (z, v) in out.iter().enumerate() {
                data[(z * ny + y) * nx + x_i] = *v;
            }
        }
    }
    data
}

/// 2D FFT of real row-major data, keeping only the non-redundant half
/// along x: output is row-major with width `w/2 + 1` and height `h`
/// (full transform along y).
///
/// # Panics
/// Panics unless `x.len() == w * h`.
#[must_use]
pub fn rfft_2d(x: &[f64], w: usize, h: usize) -> Vec<Complex> {
    assert_eq!(x.len(), w * h, "rfft_2d expects w*h samples");
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let half_w = w / 2 + 1;
    let mut data = vec![ZERO; half_w * h];
    for row in 0..h {
        let out = rfft(&x[row * w..(row + 1) * w]);
        data[row * half_w..(row + 1) * half_w].copy_from_slice(&out);
    }
    let mut col = vec![ZERO; h];
    for cx in 0..half_w {
        for (cy, v) in col.iter_mut().enumerate() {
            *v = data[cy * half_w + cx];
        }
        let out = fft_any(&col);
        for (cy, v) in out.iter().enumerate() {
            data[cy * half_w + cx] = *v;
        }
    }
    data
}

/// Swap spectrum halves in place so the zero-frequency bin moves to the
/// center (numpy `fftshift`; for odd n the extra bin lands left of center).
pub fn fft_shift(x: &mut [Complex]) {
    let n = x.len();
    if n > 1 {
        x.rotate_left(n.div_ceil(2));
    }
}

/// Frequencies (Hz) of the DFT bins for sample spacing `dt`, in FFT
/// order: 0, 1/(n·dt), …, then the negative frequencies.
#[must_use]
pub fn fft_freqs(n: usize, dt: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let df = 1.0 / (n as f64 * dt);
    (0..n)
        .map(|k| {
            if k <= (n - 1) / 2 {
                k as f64 * df
            } else {
                (k as f64 - n as f64) * df
            }
        })
        .collect()
}

/// Circular (periodic) 2D convolution of two w×h real fields via FFT.
///
/// # Panics
/// Panics unless both inputs have `w * h` samples.
#[must_use]
pub fn fft_convolve_2d(a: &[f64], b: &[f64], w: usize, h: usize) -> Vec<f64> {
    assert_eq!(a.len(), w * h, "fft_convolve_2d expects w*h samples");
    assert_eq!(b.len(), w * h, "fft_convolve_2d expects w*h samples");
    let fa = fft_2d(&to_complex(a), w, h);
    let fb = fft_2d(&to_complex(b), w, h);
    let prod: Vec<Complex> = fa.iter().zip(&fb).map(|(x, y)| *x * *y).collect();
    ifft_2d(&prod, w, h).iter().map(|c| c.re).collect()
}

/// Linear convolution of two real signals via zero-padded FFT.
/// Matches `signal_processing::convolve` (output length a + b − 1).
#[must_use]
pub fn fft_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_power_of_two(out_len);
    let mut fa = to_complex(a);
    let mut fb = to_complex(b);
    fa.resize(n, ZERO);
    fb.resize(n, ZERO);
    fft_in_place(&mut fa, -1.0);
    fft_in_place(&mut fb, -1.0);
    for (x, y) in fa.iter_mut().zip(fb.iter()) {
        *x = *x * *y;
    }
    let prod = ifft(&fa);
    prod.into_iter().take(out_len).map(|c| c.re).collect()
}

/// Cross-correlation via FFT; matches
/// `signal_processing::cross_correlate` (length a + b − 1).
#[must_use]
pub fn fft_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let reversed: Vec<f64> = b.iter().rev().copied().collect();
    fft_convolve(a, &reversed)
}

/// Band-limited (sinc) interpolation by an integer factor: zero-pad the
/// spectrum and inverse transform at length n·factor.
///
/// # Panics
/// Panics if `factor == 0`.
#[must_use]
pub fn fft_interpolate(x: &[f64], factor: usize) -> Vec<f64> {
    assert!(factor > 0, "interpolation factor must be positive");
    let n = x.len();
    if n == 0 || factor == 1 {
        return x.to_vec();
    }
    let spec = fft_any(&to_complex(x));
    let m = n * factor;
    let mut padded = vec![ZERO; m];
    let half = n / 2;
    // Positive frequencies (and DC).
    let top = half.min(n - 1) + 1;
    padded[..top].copy_from_slice(&spec[..top]);
    // Negative frequencies at the tail.
    for k in 1..n - half {
        padded[m - k] = spec[n - k];
    }
    if n.is_multiple_of(2) {
        // Split the Nyquist bin between +n/2 and −n/2.
        let half_nyq = Complex::new(spec[half].re * 0.5, spec[half].im * 0.5);
        padded[half] = half_nyq;
        padded[m - half] = half_nyq;
    }
    let scale = factor as f64;
    ifft_any(&padded).iter().map(|c| c.re * scale).collect()
}

/// Spectral derivative of a periodic signal sampled at spacing `dt`:
/// multiply each bin by jω and transform back (Nyquist bin zeroed).
#[must_use]
pub fn fft_differentiate(x: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut spec = fft_any(&to_complex(x));
    let freqs = fft_freqs(n, dt);
    for (k, s) in spec.iter_mut().enumerate() {
        let omega = 2.0 * PI * freqs[k];
        *s = *s * Complex::new(0.0, omega);
    }
    if n.is_multiple_of(2) {
        spec[n / 2] = ZERO;
    }
    ifft_any(&spec).iter().map(|c| c.re).collect()
}

/// Spectral antiderivative of a periodic signal: divide each nonzero bin
/// by jω; the DC bin is zeroed, so the result is the zero-mean periodic
/// antiderivative of the mean-removed input.
#[must_use]
pub fn fft_integrate(x: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut spec = fft_any(&to_complex(x));
    let freqs = fft_freqs(n, dt);
    spec[0] = ZERO;
    for (k, s) in spec.iter_mut().enumerate().skip(1) {
        let omega = 2.0 * PI * freqs[k];
        *s = *s / Complex::new(0.0, omega);
    }
    ifft_any(&spec).iter().map(|c| c.re).collect()
}

/// Solve the periodic Poisson problem ∇²u = rhs on a w×h grid with
/// spacing `dx`, using the eigenvalues of the discrete 5-point Laplacian
/// so the discrete residual is at roundoff. The k=0 mode is set to zero
/// (the mean-free solution; a pure-Neumann/periodic problem only
/// determines u up to a constant, and requires a mean-free rhs).
///
/// # Panics
/// Panics unless `rhs.len() == w * h`.
#[must_use]
pub fn fft_poisson_2d(rhs: &[f64], w: usize, h: usize, dx: f64) -> Vec<f64> {
    assert_eq!(rhs.len(), w * h, "fft_poisson_2d expects w*h samples");
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut spec = fft_2d(&to_complex(rhs), w, h);
    let inv_dx2 = 1.0 / (dx * dx);
    for ky in 0..h {
        for kx in 0..w {
            let idx = ky * w + kx;
            if kx == 0 && ky == 0 {
                spec[idx] = ZERO;
                continue;
            }
            let lambda = (2.0 * (2.0 * PI * kx as f64 / w as f64).cos()
                + 2.0 * (2.0 * PI * ky as f64 / h as f64).cos()
                - 4.0)
                * inv_dx2;
            spec[idx] = Complex::new(spec[idx].re / lambda, spec[idx].im / lambda);
        }
    }
    ifft_2d(&spec, w, h).iter().map(|c| c.re).collect()
}

fn to_complex(x: &[f64]) -> Vec<Complex> {
    x.iter().map(|&v| Complex::new(v, 0.0)).collect()
}

/// Precomputed twiddle factors and bit-reversal permutation for repeated
/// power-of-two FFTs of one size.
pub struct FftPlan {
    n: usize,
    /// Forward twiddles w_n^k = e^(−2πik/n) for k in 0..n/2.
    twiddles: Vec<Complex>,
    bit_rev: Vec<usize>,
}

impl FftPlan {
    /// Build a plan for length n.
    ///
    /// # Panics
    /// Panics unless n is a power of two.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(is_power_of_two(n), "FftPlan requires a power-of-two length");
        let twiddles: Vec<Complex> = (0..n / 2).map(|k| cis(-2.0 * PI * k as f64 / n as f64)).collect();
        let mut bit_rev = vec![0usize; n];
        let mut j = 0;
        for slot in bit_rev.iter_mut().skip(1) {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            *slot = j;
        }
        Self { n, twiddles, bit_rev }
    }

    /// Planned transform length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True for the (degenerate) length-0 plan; present for clippy's
    /// `len_without_is_empty` convention.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn run(&self, x: &mut [Complex], inverse: bool) {
        let n = self.n;
        assert_eq!(x.len(), n, "FftPlan::execute expects the planned length");
        for i in 1..n {
            let j = self.bit_rev[i];
            if i < j {
                x.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let mut w = self.twiddles[k * stride];
                    if inverse {
                        w = w.conjugate();
                    }
                    let u = x[start + k];
                    let v = x[start + k + len / 2] * w;
                    x[start + k] = u + v;
                    x[start + k + len / 2] = u - v;
                }
            }
            len <<= 1;
        }
    }

    /// In-place forward FFT of the planned length.
    ///
    /// # Panics
    /// Panics unless `x.len()` equals the planned length.
    pub fn execute(&self, x: &mut [Complex]) {
        self.run(x, false);
    }

    /// In-place inverse FFT of the planned length (with 1/n scaling).
    ///
    /// # Panics
    /// Panics unless `x.len()` equals the planned length.
    pub fn execute_inverse(&self, x: &mut [Complex]) {
        self.run(x, true);
        let inv_n = 1.0 / self.n as f64;
        for v in x.iter_mut() {
            *v = Complex::new(v.re * inv_n, v.im * inv_n);
        }
    }
}

/// Kept so downstream code has a `Result` constructor site for plan
/// creation without panicking.
impl TryFrom<usize> for FftPlan {
    type Error = SolveError;

    fn try_from(n: usize) -> Result<Self, Self::Error> {
        if is_power_of_two(n) {
            Ok(Self::new(n))
        } else {
            Err(SolveError::InvalidArgument("FftPlan requires a power-of-two length"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn naive_dft(x: &[Complex], sign: f64) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex::new(0.0, 0.0);
                for (j, &v) in x.iter().enumerate() {
                    acc = acc + v * cis(sign * 2.0 * PI * (k * j) as f64 / n as f64);
                }
                acc
            })
            .collect()
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(16), 16);
        assert_eq!(next_power_of_two(17), 32);
    }

    #[test]
    fn test_fft_impulse_is_flat() {
        let mut x = vec![Complex::new(0.0, 0.0); 8];
        x[0] = Complex::new(1.0, 0.0);
        let spec = fft(&x);
        for c in spec {
            assert!(approx(c.re, 1.0, 1e-12) && approx(c.im, 0.0, 1e-12));
        }
    }

    #[test]
    fn test_fft_dc_component() {
        let x = vec![Complex::new(1.0, 0.0); 8];
        let spec = fft(&x);
        assert!(approx(spec[0].re, 8.0, 1e-12));
        for c in &spec[1..] {
            assert!(c.norm() < 1e-12);
        }
    }

    #[test]
    fn test_ifft_roundtrip() {
        let x: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let back = ifft(&fft(&x));
        for (orig, rec) in x.iter().zip(&back) {
            assert!(approx(orig.re, rec.re, 1e-12) && approx(orig.im, rec.im, 1e-12));
        }
    }

    #[test]
    fn test_fft_any_matches_naive_awkward_lengths() {
        // 12 = 2²·3 (mixed radix), 7 and 13 prime (Bluestein), 30 = 2·3·5.
        for &n in &[2usize, 3, 5, 7, 12, 13, 30, 100] {
            let x: Vec<Complex> = (0..n)
                .map(|i| Complex::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
                .collect();
            let fast = fft_any(&x);
            let slow = naive_dft(&x, -1.0);
            for (a, b) in fast.iter().zip(&slow) {
                assert!(approx(a.re, b.re, 1e-8), "n={n}");
                assert!(approx(a.im, b.im, 1e-8), "n={n}");
            }
        }
    }

    #[test]
    fn test_rfft_length_and_symmetry() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.7).sin()).collect();
        let half = rfft(&x);
        assert_eq!(half.len(), 9);
        let full = fft(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
        for (a, b) in half.iter().zip(full.iter()) {
            assert!(approx(a.re, b.re, 1e-12) && approx(a.im, b.im, 1e-12));
        }
        for k in 9..16 {
            assert!(approx(full[k].re, full[16 - k].re, 1e-10));
            assert!(approx(full[k].im, -full[16 - k].im, 1e-10));
        }
    }

    #[test]
    fn test_irfft_roundtrip_even_odd() {
        for &n in &[8usize, 9, 15, 16] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.9).sin() + 0.3).collect();
            let back = irfft(&rfft(&x), n);
            assert_eq!(back.len(), n);
            for (a, b) in x.iter().zip(&back) {
                assert!(approx(*a, *b, 1e-10), "n={n}");
            }
        }
    }

    #[test]
    fn test_fft_2d_roundtrip_and_dc() {
        let (w, h) = (6, 4);
        let x: Vec<Complex> = (0..w * h)
            .map(|i| Complex::new(1.0 + (i as f64).sin(), 0.0))
            .collect();
        let spec = fft_2d(&x, w, h);
        let sum: f64 = x.iter().map(|c| c.re).sum();
        assert!(approx(spec[0].re, sum, 1e-9));
        let back = ifft_2d(&spec, w, h);
        for (a, b) in x.iter().zip(&back) {
            assert!(approx(a.re, b.re, 1e-9) && approx(a.im, b.im, 1e-9));
        }
    }

    #[test]
    fn test_fft_3d_roundtrip() {
        let (nx, ny, nz) = (4, 3, 2);
        let x: Vec<Complex> = (0..nx * ny * nz)
            .map(|i| Complex::new((i as f64 * 0.31).cos(), (i as f64 * 0.17).sin()))
            .collect();
        let back = ifft_3d(&fft_3d(&x, nx, ny, nz), nx, ny, nz);
        for (a, b) in x.iter().zip(&back) {
            assert!(approx(a.re, b.re, 1e-9) && approx(a.im, b.im, 1e-9));
        }
    }

    #[test]
    fn test_rfft_2d_matches_full() {
        let (w, h) = (8, 4);
        let x: Vec<f64> = (0..w * h).map(|i| (i as f64 * 0.37).sin()).collect();
        let half = rfft_2d(&x, w, h);
        let full = fft_2d(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>(), w, h);
        let half_w = w / 2 + 1;
        for row in 0..h {
            for col in 0..half_w {
                let a = half[row * half_w + col];
                let b = full[row * w + col];
                assert!(approx(a.re, b.re, 1e-9) && approx(a.im, b.im, 1e-9));
            }
        }
    }

    #[test]
    fn test_fft_shift_even_odd() {
        let mut x: Vec<Complex> = (0..4).map(|i| Complex::new(i as f64, 0.0)).collect();
        fft_shift(&mut x);
        let vals: Vec<f64> = x.iter().map(|c| c.re).collect();
        assert_eq!(vals, vec![2.0, 3.0, 0.0, 1.0]);
        let mut y: Vec<Complex> = (0..5).map(|i| Complex::new(i as f64, 0.0)).collect();
        fft_shift(&mut y);
        let vals: Vec<f64> = y.iter().map(|c| c.re).collect();
        assert_eq!(vals, vec![3.0, 4.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_fft_freqs() {
        let f = fft_freqs(4, 0.5);
        assert_eq!(f, vec![0.0, 0.5, -1.0, -0.5]);
        let f = fft_freqs(5, 1.0);
        assert_eq!(f, vec![0.0, 0.2, 0.4, -0.4, -0.2]);
    }

    #[test]
    fn test_fft_convolve_small() {
        let y = fft_convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5]);
        let expected = [0.0, 1.0, 2.5, 4.0, 1.5];
        assert_eq!(y.len(), expected.len());
        for (a, b) in y.iter().zip(&expected) {
            assert!(approx(*a, *b, 1e-12));
        }
    }

    #[test]
    fn test_fft_convolve_empty() {
        assert!(fft_convolve(&[], &[1.0]).is_empty());
        assert!(fft_convolve(&[1.0], &[]).is_empty());
    }

    #[test]
    fn test_fft_convolve_2d_impulse() {
        let (w, h) = (4, 3);
        let mut a = vec![0.0; w * h];
        a[0] = 1.0; // delta at origin: convolution returns b unchanged
        let b: Vec<f64> = (0..w * h).map(|i| i as f64).collect();
        let y = fft_convolve_2d(&a, &b, w, h);
        for (u, v) in y.iter().zip(&b) {
            assert!(approx(*u, *v, 1e-9));
        }
    }

    #[test]
    fn test_fft_correlate_matches_direct() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, -1.0, 0.5];
        let direct = crate::signal_processing::cross_correlate(&a, &b);
        let fast = fft_correlate(&a, &b);
        assert_eq!(direct.len(), fast.len());
        for (d, f) in direct.iter().zip(&fast) {
            assert!(approx(*d, *f, 1e-10));
        }
    }

    #[test]
    fn test_fft_interpolate_tone() {
        // A pure tone interpolated 4x should hit the fine-grid samples.
        let n = 16;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin()).collect();
        let y = fft_interpolate(&x, 4);
        assert_eq!(y.len(), 4 * n);
        for (i, v) in y.iter().enumerate() {
            let exact = (2.0 * PI * 3.0 * i as f64 / (4 * n) as f64).sin();
            assert!(approx(*v, exact, 1e-9), "i={i}: {v} vs {exact}");
        }
    }

    #[test]
    fn test_fft_differentiate_sin_gives_cos() {
        let n = 64;
        let dt = 2.0 * PI / n as f64; // one period of sin(t)
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * dt).sin()).collect();
        let d = fft_differentiate(&x, dt);
        for (i, v) in d.iter().enumerate() {
            assert!(approx(*v, (i as f64 * dt).cos(), 1e-8));
        }
    }

    #[test]
    fn test_fft_integrate_cos_gives_sin() {
        let n = 64;
        let dt = 2.0 * PI / n as f64;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * dt).cos()).collect();
        let int = fft_integrate(&x, dt);
        for (i, v) in int.iter().enumerate() {
            assert!(approx(*v, (i as f64 * dt).sin(), 1e-8));
        }
    }

    #[test]
    fn test_fft_poisson_2d_residual() {
        let (w, h) = (16, 12);
        let dx = 0.1;
        // Mean-free rhs.
        let mut rhs: Vec<f64> = (0..w * h).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();
        let mean = rhs.iter().sum::<f64>() / rhs.len() as f64;
        for v in rhs.iter_mut() {
            *v -= mean;
        }
        let u = fft_poisson_2d(&rhs, w, h, dx);
        // Discrete periodic 5-point Laplacian residual.
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let xp = y * w + (x + 1) % w;
                let xm = y * w + (x + w - 1) % w;
                let yp = ((y + 1) % h) * w + x;
                let ym = ((y + h - 1) % h) * w + x;
                let lap = (u[xp] + u[xm] + u[yp] + u[ym] - 4.0 * u[idx]) / (dx * dx);
                assert!(approx(lap, rhs[idx], 1e-9), "residual too large at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_fft_plan_matches_fft() {
        let n = 32;
        let x: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.6).sin(), (i as f64 * 0.4).cos()))
            .collect();
        let plan = FftPlan::new(n);
        assert_eq!(plan.len(), n);
        assert!(!plan.is_empty());
        let mut buf = x.clone();
        plan.execute(&mut buf);
        let expected = fft(&x);
        for (a, b) in buf.iter().zip(&expected) {
            assert!(approx(a.re, b.re, 1e-10) && approx(a.im, b.im, 1e-10));
        }
        plan.execute_inverse(&mut buf);
        for (a, b) in buf.iter().zip(&x) {
            assert!(approx(a.re, b.re, 1e-10) && approx(a.im, b.im, 1e-10));
        }
    }

    #[test]
    fn test_fft_plan_try_from_rejects() {
        assert!(FftPlan::try_from(12).is_err());
        assert!(FftPlan::try_from(16).is_ok());
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn test_fft_rejects_non_power_of_two() {
        let _ = fft(&[Complex::new(0.0, 0.0); 3]);
    }
}
