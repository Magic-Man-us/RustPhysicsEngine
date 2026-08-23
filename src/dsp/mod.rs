//! Digital signal processing: window functions, FIR/IIR filter design,
//! resampling, and phase utilities.
//!
//! The window generators and first-order RC filters that used to live in
//! `signal_processing` moved here; the old paths re-export them.

pub mod iir;
pub mod windows;

pub use iir::{first_order_highpass, first_order_lowpass};
pub use windows::{blackman_window, hamming_window, hann_window, rectangular_window};
