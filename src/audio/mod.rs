//! Audio synthesis, analysis, effects, and I/O.

pub mod effects;
pub mod envelope;
pub mod oscillators;
pub mod synthesis;
pub mod wav;

pub use effects::{
    bitcrush, convolution_reverb, dc_offset_remove, declick, distortion_foldback,
    distortion_hard_clip, distortion_soft_clip, distortion_tube, dither_tpdf, gain_db, haas_delay,
    measure_lufs, noise_shaping_dither, normalize_lufs, normalize_peak, normalize_rms,
    oversample_process, pitch_shift_simple, spectral_gate, synthesize_ir_exponential, true_peak,
    AllpassFilter, Chorus, CombFilter, Compressor, DeEsser, DelayLine, Eq, Exciter, Expander, Fdn,
    Flanger, Freeverb, Limiter, NoiseGate, PartitionedConvolver, Phaser, SchroederReverb,
    StereoWidener, Tremolo, Vibrato,
};
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
pub use synthesis::{
    additive, additive_evolving, am, chebyshev_waveshaper, drum_clap, drum_hihat, drum_kick,
    drum_snare, drum_tom, fm_bessel_sidebands, fm_simple, formant_synth, granular,
    hard_sync_osc, karplus_strong, karplus_strong_extended, mix, phase_distortion, pm_simple,
    pulsar_synthesis, render_note, render_sequence, ring_mod, sample_playback, subtractive,
    supersaw, vector_synth, vowel_formants, waveshaper, FmOperator, FmSynth, Synth, Voice,
};
pub use wav::{
    from_interleaved, to_interleaved, to_mono, wav_info, wav_read, wav_read_file, wav_write,
    wav_write_file, WavData,
};
