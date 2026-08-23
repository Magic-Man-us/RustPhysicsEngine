//! Discrete transforms: FFT (any length), DCT/DST, STFT, wavelets,
//! Hilbert, Laplace inversion, Radon, and spectral estimation.
//!
//! The radix-2 FFT that used to live in `signal_processing::fft` moved
//! here; the old paths re-export everything so no caller changes.

pub mod fft;

pub use fft::{
    fft, fft_2d, fft_3d, fft_any, fft_convolve, fft_convolve_2d, fft_correlate,
    fft_differentiate, fft_freqs, fft_integrate, fft_interpolate, fft_poisson_2d, fft_shift,
    ifft, ifft_2d, ifft_3d, ifft_any, irfft, next_power_of_two, rfft, rfft_2d, FftPlan,
};
