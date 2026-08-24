//! Audio synthesis, analysis, effects, and I/O.

pub mod analysis;
pub mod effects;
pub mod envelope;
pub mod oscillators;
pub mod physical;
pub mod spatial;
pub mod synthesis;
pub mod tuning;
pub mod vocoder;
pub mod wav;

pub use analysis::{
    audio_fingerprint, autocorrelation_fft, beat_track, c50, c80, chord_estimate, chroma, d50,
    delta_features, dynamic_time_warping, edt_from_ir, enob, estimate_snr, fluctuation_strength,
    formant_track, harmonic_to_noise_ratio, impulse_response_from_sweep, inharmonicity_measure,
    key_estimate, loudness_sone, lpc, lpc_spectrum, lpc_to_formants, lpc_to_lsp, lsp_to_lpc, mfcc,
    onset_complex_domain, onset_detect, onset_hfc, onset_strength, peak_pick,
    pitch_autocorrelation, pitch_cepstral, pitch_hps, pitch_mpm, pitch_to_midi_track, pitch_track,
    pitch_yin, roughness, rt60_from_ir, sharpness, silence_detect, sinad, spectral_centroid,
    spectral_crest, spectral_decrease, spectral_entropy_mag, spectral_features_track,
    spectral_flatness_mag, spectral_flux, spectral_kurtosis, spectral_rolloff, spectral_skewness,
    spectral_slope, spectral_spread, sti_approx, tempo_estimate, thd_n, transient_detect,
    zero_crossing_rate, PitchMethod, SpectralFeatures,
};
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
pub use physical::{
    banded_waveguide, commuted_synthesis, glottal_pulse_lf, hammer_string_interaction,
    inharmonic_partials, jet_nonlinearity, lip_model, reed_nonlinearity, rosenberg_pulse,
    string_tension_from_freq, sympathetic_resonance, vocal_tract, BowedString, KellyLochbaum,
    MassSpringString, Membrane2D, ModalSynth, Plate2D, WaveguideString, WaveguideTube,
};
pub use spatial::{
    air_absorption_filter, ambisonics_decode, ambisonics_encode, ambisonics_encode_1st,
    ambisonics_rotate, beamforming_delay_sum, beamforming_mvdr, binaural_simple, distance_gain,
    doppler_resample, early_reflections, ild_spherical_head, image_source_ir, itd_woodworth,
    localize_tdoa, pan_constant_power, pan_linear, pan_minus_4_5_db, pan_vbap_2d, pan_vbap_3d,
    ray_tracing_ir, sonar_equation, sonar_range, speaker_baffle_step, speaker_crossover_lr4,
    spherical_head_hrtf, tdoa_gcc_phat, thiele_small_response,
};
pub use tuning::{
    bohlen_pierce, cents_between, cents_to_ratio, chord_tones, circle_of_fifths,
    consonance_plomp_levelt, dissonance_curve, equal_temperament, harmonic_series_scale,
    interval_name, just_intonation_5limit, kirnberger_iii, meantone_quarter_comma,
    midi_to_freq_tuned, nearest_note, pythagorean, pythagorean_comma, ratio_to_cents, scala_parse,
    scale_degrees, schisma, stretch_tuning_railsback, syntonic_comma, werckmeister_iii, young,
    ChordQuality, Mode,
};
pub use vocoder::{
    autotune, channel_vocoder, cross_synthesis, harmonizer, lpc_vocoder, psola_pitch_shift,
    spectral_morph, wsola_time_stretch, Excitation, PhaseVocoder,
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
