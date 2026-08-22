//! Properties for `signal_processing::fft`.

use rust_physics_engine::fractals::Complex;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::signal_processing::{convolve, fft, fft_convolve, ifft};
use rust_physics_engine::statistics::dft;

/// ifft(fft(x)) == x for random complex signals.
#[test]
fn prop_fft_roundtrip() {
    let mut rng = Rng::new(11);
    for &n in &[2usize, 8, 64, 256] {
        let x: Vec<Complex> = (0..n)
            .map(|_| Complex::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0))
            .collect();
        let back = ifft(&fft(&x));
        for (orig, rec) in x.iter().zip(&back) {
            assert!((orig.re - rec.re).abs() < 1e-10);
            assert!((orig.im - rec.im).abs() < 1e-10);
        }
    }
}

/// fft matches the direct DFT on real signals of length 8, 16, 64.
#[test]
fn prop_fft_matches_dft() {
    let mut rng = Rng::new(12);
    for &n in &[8usize, 16, 64] {
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let slow = dft(&x);
        let fast = fft(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
        for (s, f) in slow.iter().zip(&fast) {
            assert!((s.0 - f.re).abs() < 1e-9, "re mismatch at n={n}");
            assert!((s.1 - f.im).abs() < 1e-9, "im mismatch at n={n}");
        }
    }
}

/// fft_convolve matches direct convolution.
#[test]
fn prop_fft_convolve_matches_direct() {
    let mut rng = Rng::new(13);
    for _ in 0..20 {
        let la = 1 + (rng.next_u64() % 40) as usize;
        let lb = 1 + (rng.next_u64() % 40) as usize;
        let a: Vec<f64> = (0..la).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let b: Vec<f64> = (0..lb).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let direct = convolve(&a, &b);
        let fast = fft_convolve(&a, &b);
        assert_eq!(direct.len(), fast.len());
        for (d, f) in direct.iter().zip(&fast) {
            assert!((d - f).abs() < 1e-9, "convolve mismatch: {d} vs {f}");
        }
    }
}
