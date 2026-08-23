//! Resonance and vibration: single and coupled oscillators, acoustic
//! and electromagnetic cavities, nonlinear resonance, and structural
//! dynamics.

pub mod cavity;
pub mod coupled;
pub mod nonlinear;
pub mod structural;
pub mod oscillator;

pub use cavity::{
    beam_mode_shape, beam_modes, bell_modes_approx, cavity_photon_lifetime, cavity_q,
    chladni_pattern, chladni_pattern_mixed, circular_membrane_modes, circular_membrane_shape,
    conical_tube_modes, coupled_cavity_splitting, cylindrical_cavity_modes,
    fabry_perot_finesse, fabry_perot_fsr, fabry_perot_transmission, helmholtz_q,
    helmholtz_resonator, inharmonicity_coefficient, microwave_cavity_modes_rect,
    quarter_wave_resonator, rectangular_membrane_modes, rectangular_plate_modes,
    resonance_overlap, room_mode_density, room_modes, schroeder_frequency,
    stiff_string_modes, string_mode_shape, string_modes, tube_end_correction, tube_modes,
    tuning_fork_frequency, BeamBc, PlateBc, Rlc,
};
pub use structural::{
    circle_fit, experimental_modal_peak_picking, half_power_bandwidth,
    operational_deflection_shape, shock_response_spectrum,
    stochastic_subspace_identification, ModalModel,
};
pub use nonlinear::{
    autoresonance_threshold, describing_function, duffing_backbone, duffing_jump_frequencies,
    duffing_poincare, duffing_response_amplitude, duffing_simulate, fano_fit, fano_lineshape,
    frequency_pulling, harmonic_balance, hysteresis_loop, injection_locking_range,
    kapitza_pendulum_stable, mathieu_stability, mathieu_stability_chart,
    parametric_resonance_threshold, stochastic_resonance_snr, subharmonic_response,
    van_der_pol_entrainment_range, van_der_pol_limit_cycle_amplitude, van_der_pol_simulate,
};
pub use coupled::{
    huygens_sync_simulate, kuramoto, kuramoto_critical_coupling, tuned_mass_damper_design,
    two_pendulums_coupled, wilberforce_pendulum, CoupledOscillators,
};
pub use oscillator::{
    bode_plot, lorentzian, lorentzian_fit, nyquist_plot, q_from_ringdown, q_from_spectrum,
    quality_factor_combined, resonance_curve, transmissibility, DampedOscillator, Damping,
};
