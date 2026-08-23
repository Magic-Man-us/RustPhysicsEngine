//! Properties for `transforms::fft` (roadmap Part 3, item 1).

use rust_physics_engine::fractals::Complex;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::transforms::fft::{
    fft_any, fft_differentiate, fft_poisson_2d, ifft_any, rfft,
};

/// ifft_any(fft_any(x)) == x for every n in 1..=200, primes included.
#[test]
fn prop_fft_any_roundtrip_all_lengths() {
    let mut rng = Rng::new(31);
    for n in 1..=200usize {
        let x: Vec<Complex> = (0..n)
            .map(|_| Complex::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0))
            .collect();
        let back = ifft_any(&fft_any(&x));
        for (orig, rec) in x.iter().zip(&back) {
            assert!((orig.re - rec.re).abs() < 1e-9, "re mismatch at n={n}");
            assert!((orig.im - rec.im).abs() < 1e-9, "im mismatch at n={n}");
        }
    }
}

/// rfft matches the first n/2 + 1 bins of fft_any for random real input.
#[test]
fn prop_rfft_matches_fft_any_half() {
    let mut rng = Rng::new(32);
    for &n in &[1usize, 2, 7, 12, 13, 30, 97, 128, 150] {
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let half = rfft(&x);
        assert_eq!(half.len(), n / 2 + 1);
        let full = fft_any(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
        for (a, b) in half.iter().zip(full.iter()) {
            assert!((a.re - b.re).abs() < 1e-9, "n={n}");
            assert!((a.im - b.im).abs() < 1e-9, "n={n}");
        }
    }
}

/// fft_poisson_2d leaves a discrete-Laplacian residual below 1e-10 on a
/// random mean-free right-hand side.
#[test]
fn prop_fft_poisson_2d_residual() {
    let mut rng = Rng::new(33);
    let (w, h) = (24, 18);
    let dx = 0.05;
    let mut rhs: Vec<f64> = (0..w * h).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
    let mean = rhs.iter().sum::<f64>() / rhs.len() as f64;
    for v in rhs.iter_mut() {
        *v -= mean;
    }
    let u = fft_poisson_2d(&rhs, w, h, dx);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let lap = (u[y * w + (x + 1) % w]
                + u[y * w + (x + w - 1) % w]
                + u[((y + 1) % h) * w + x]
                + u[((y + h - 1) % h) * w + x]
                - 4.0 * u[idx])
                / (dx * dx);
            assert!((lap - rhs[idx]).abs() < 1e-10, "residual at ({x},{y})");
        }
    }
}

/// Spectral differentiation of a sampled sine is its cosine to 1e-8.
#[test]
fn prop_fft_differentiate_sin_is_cos() {
    use std::f64::consts::PI;
    for &n in &[32usize, 100, 128] {
        let dt = 2.0 * PI / n as f64;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * dt).sin()).collect();
        let d = fft_differentiate(&x, dt);
        for (i, v) in d.iter().enumerate() {
            assert!((*v - (i as f64 * dt).cos()).abs() < 1e-8, "n={n}, i={i}");
        }
    }
}
