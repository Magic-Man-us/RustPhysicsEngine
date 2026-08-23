//! Discrete transforms: FFT (any length), DCT/DST, STFT, wavelets,
//! Hilbert, Laplace inversion, Radon, and spectral estimation.
//!
//! The radix-2 FFT that used to live in `signal_processing::fft` moved
//! here; the old paths re-export everything so no caller changes.

pub mod dct;
pub mod fft;
pub mod hilbert;
pub mod stft;
pub mod wavelet;
pub(crate) mod wavelet_tables;

pub use dct::{
    dct_2d, dct_compress, dct_i, dct_ii, dct_iii, dct_iv, dct_poisson_1d, dst_i, dst_ii,
    hartley, idct_2d, idct_ii, Bc,
};
pub use stft::{
    chirp_z, constant_q_transform, dtmf_decode, goertzel, goertzel_bank, mel_filterbank,
    mel_spectrogram, reassigned_spectrogram, spectrogram, zoom_fft, Stft,
};
pub use wavelet::{
    cwt, dwt, dwt_2d, idwt, idwt_2d, lifting_dwt_53, lifting_idwt_53,
    multiresolution_analysis, scale_to_frequency, scalogram, wavedec, wavelet_compress,
    wavelet_denoise, wavelet_filters, wavelet_packet_decompose, waverec, Mother, PadMode,
    Threshold, Wavelet,
};
pub use hilbert::{
    am_demodulate, analytic_signal, empirical_mode_decomposition, envelope, fm_demodulate,
    hilbert, hilbert_fir, hilbert_huang_spectrum, instantaneous_frequency,
    instantaneous_phase, kramers_kronig, minimum_phase_from_magnitude, ssb_modulate,
};
pub use fft::{
    fft, fft_2d, fft_3d, fft_any, fft_convolve, fft_convolve_2d, fft_correlate,
    fft_differentiate, fft_freqs, fft_integrate, fft_interpolate, fft_poisson_2d, fft_shift,
    ifft, ifft_2d, ifft_3d, ifft_any, irfft, next_power_of_two, rfft, rfft_2d, FftPlan,
};
