//! Digital signal processing: window functions, FIR/IIR filter design,
//! resampling, and phase utilities.
//!
//! The window generators and first-order RC filters that used to live in
//! `signal_processing` moved here; the old paths re-export them.

pub mod fir;
pub mod iir;
pub mod windows;

pub use fir::{
    filtfilt_fir, fir_apply, fir_apply_fft, fir_bandpass, fir_bandstop, fir_differentiator,
    fir_freq_response, fir_gaussian, fir_group_delay, fir_highpass, fir_hilbert,
    fir_kaiser_design, fir_least_squares, fir_lowpass, fir_parks_mcclellan, fir_raised_cosine,
    fir_root_raised_cosine, fir_savitzky_golay, FirState,
};
pub use iir::{first_order_highpass, first_order_lowpass};
pub use windows::{
    blackman_window, hamming_window, hann_window, kaiser_beta_for_attenuation, rectangular_window,
    window, window_metrics, WindowKind, WindowMetrics,
};
