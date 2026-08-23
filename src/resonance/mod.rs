//! Resonance and vibration: single and coupled oscillators, acoustic
//! and electromagnetic cavities, nonlinear resonance, and structural
//! dynamics.

pub mod coupled;
pub mod oscillator;

pub use coupled::{
    huygens_sync_simulate, kuramoto, kuramoto_critical_coupling, tuned_mass_damper_design,
    two_pendulums_coupled, wilberforce_pendulum, CoupledOscillators,
};
pub use oscillator::{
    bode_plot, lorentzian, lorentzian_fit, nyquist_plot, q_from_ringdown, q_from_spectrum,
    quality_factor_combined, resonance_curve, transmissibility, DampedOscillator, Damping,
};
