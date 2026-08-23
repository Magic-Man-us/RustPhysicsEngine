//! Audio synthesis, analysis, effects, and I/O.

pub mod envelope;
pub mod oscillators;
pub mod wav;

pub use envelope::{
    apply_envelope, crossfade, envelope_follower, exponential_decay_envelope, fade_in, fade_out,
    peak_envelope, portamento, rms_envelope, Adsr, AdsrExp, Ar, FadeShape, Lfo,
};
pub use oscillators::{
    additive_saw, additive_square, additive_triangle, band_limited_impulse_train, chirp_hyperbolic,
    chirp_linear, chirp_exponential, dc, impulse, multisine, polyblep_saw, polyblep_square,
    polyblep_triangle, pulse_train, schroeder_phase_multisine, sine_sweep_with_inverse, NoiseColor,
    NoiseGen, Oscillator, Waveform, Wavetable,
};
pub use wav::{
    from_interleaved, to_interleaved, to_mono, wav_info, wav_read, wav_read_file, wav_write,
    wav_write_file, WavData,
};
