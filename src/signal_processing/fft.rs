//! Fast Fourier transform (iterative radix-2 Cooley-Tukey).
//!
//! Reference: Press et al., *Numerical Recipes*, §12.2. All transforms
//! require power-of-two lengths; use [`next_power_of_two`] to size
//! zero-padded buffers.

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
        let wlen = Complex::new(ang.cos(), ang.sin());
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
/// Panics unless `input.len()` is a power of two.
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
/// Panics unless `input.len()` is a power of two.
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

/// FFT of a real signal, returning the n/2 + 1 non-redundant bins
/// (bins k > n/2 satisfy X[n−k] = X[k]*).
///
/// # Panics
/// Panics unless `input.len()` is a power of two.
#[must_use]
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    assert!(is_power_of_two(input.len()), "rfft requires a power-of-two length");
    let buf: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut full = fft(&buf);
    full.truncate(input.len() / 2 + 1);
    full
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
    let mut fa: Vec<Complex> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut fb: Vec<Complex> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fa.resize(n, Complex::new(0.0, 0.0));
    fb.resize(n, Complex::new(0.0, 0.0));
    fft_in_place(&mut fa, -1.0);
    fft_in_place(&mut fb, -1.0);
    for (x, y) in fa.iter_mut().zip(fb.iter()) {
        *x = *x * *y;
    }
    let prod = ifft(&fa);
    prod.into_iter().take(out_len).map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
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
    fn test_rfft_length_and_symmetry() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.7).sin()).collect();
        let half = rfft(&x);
        assert_eq!(half.len(), 9);
        let full = fft(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
        for (a, b) in half.iter().zip(full.iter()) {
            assert!(approx(a.re, b.re, 1e-12) && approx(a.im, b.im, 1e-12));
        }
        // Conjugate symmetry of the discarded half.
        for k in 9..16 {
            assert!(approx(full[k].re, full[16 - k].re, 1e-10));
            assert!(approx(full[k].im, -full[16 - k].im, 1e-10));
        }
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
    #[should_panic(expected = "power-of-two")]
    fn test_fft_rejects_non_power_of_two() {
        let _ = fft(&[Complex::new(0.0, 0.0); 3]);
    }
}
