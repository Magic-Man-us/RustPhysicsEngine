use crate::math::constants::PI;

// ── Sabine constant: 0.161 (derived from 24×ln(10)/c ≈ 0.161 for speed of sound ~343 m/s) ──
const SABINE_CONSTANT: f64 = 0.161;

// ── A-weighting reference frequencies (Hz) and correction offset (dB) ──
const A_WEIGHT_F1: f64 = 20.6;
const A_WEIGHT_F2: f64 = 107.7;
const A_WEIGHT_F3: f64 = 737.9;
const A_WEIGHT_F4: f64 = 12194.0;
const A_WEIGHT_OFFSET_DB: f64 = 2.0;

// ── Mass law constant (dB) ──
const MASS_LAW_OFFSET_DB: f64 = 47.0;

// ── Mel scale constants ──
const MEL_BREAKPOINT: f64 = 2595.0;
const MEL_DIVISOR: f64 = 700.0;

// ── Bark scale constants ──
const BARK_LOW_DIVISOR: f64 = 1000.0;
const BARK_LOW_COEFF: f64 = 13.0;
const BARK_LOW_FACTOR: f64 = 0.76;
const BARK_HIGH_DIVISOR: f64 = 7500.0;
const BARK_HIGH_COEFF: f64 = 3.5;

// ── Atmospheric absorption constants ──
const ATMOS_BASE_COEFF: f64 = 0.01;
const ATMOS_FREQ_EXPONENT: f64 = 1.7;
const ATMOS_TEMP_REF: f64 = 20.0;
const ATMOS_TEMP_FACTOR: f64 = 0.01;
const ATMOS_HUMIDITY_FACTOR: f64 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════
// Room Acoustics
// ═══════════════════════════════════════════════════════════════════════════

/// Sabine reverberation time: T60 = 0.161 V / A (seconds).
#[must_use]
pub fn sabine_reverberation(volume: f64, total_absorption: f64) -> f64 {
    assert!(total_absorption > 0.0, "total_absorption must be positive");
    SABINE_CONSTANT * volume / total_absorption
}

/// Eyring reverberation time: T60 = 0.161 V / (-S × ln(1 - ā)) (seconds).
#[must_use]
pub fn eyring_reverberation(volume: f64, surface_area: f64, avg_absorption_coeff: f64) -> f64 {
    assert!(surface_area > 0.0, "surface_area must be positive");
    assert!(avg_absorption_coeff < 1.0, "avg_absorption_coeff must be less than 1");
    SABINE_CONSTANT * volume / (-surface_area * (1.0 - avg_absorption_coeff).ln())
}

/// Total absorption area: A = Σ(Si × αi).
/// Each tuple is (area_m2, absorption_coefficient).
#[must_use]
pub fn total_absorption(surfaces: &[(f64, f64)]) -> f64 {
    surfaces.iter().map(|(area, coeff)| area * coeff).sum()
}

/// Room constant: R = S × ā / (1 - ā).
#[must_use]
pub fn room_constant(surface_area: f64, avg_absorption: f64) -> f64 {
    assert!(avg_absorption < 1.0, "avg_absorption must be less than 1");
    surface_area * avg_absorption / (1.0 - avg_absorption)
}

/// Critical distance: dc = √(Q × R / (16π)).
#[must_use]
pub fn critical_distance(room_constant: f64, directivity: f64) -> f64 {
    (directivity * room_constant / (16.0 * PI)).sqrt()
}

/// Axial/tangential/oblique room mode frequency:
/// f = (c/2) × √((nx/L)² + (ny/W)² + (nz/H)²).
#[must_use]
pub fn room_mode_frequency(
    length: f64,
    width: f64,
    height: f64,
    nx: u32,
    ny: u32,
    nz: u32,
    speed: f64,
) -> f64 {
    assert!(length > 0.0, "length must be positive");
    assert!(width > 0.0, "width must be positive");
    assert!(height > 0.0, "height must be positive");
    let nx = f64::from(nx);
    let ny = f64::from(ny);
    let nz = f64::from(nz);
    (speed / 2.0)
        * ((nx / length).powi(2) + (ny / width).powi(2) + (nz / height).powi(2)).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Sound Level Operations
// ═══════════════════════════════════════════════════════════════════════════

/// Energetic sum of two decibel levels: 10 × log₁₀(10^(dB1/10) + 10^(dB2/10)).
#[must_use]
pub fn add_db(db1: f64, db2: f64) -> f64 {
    let sum = 10.0_f64.powf(db1 / 10.0) + 10.0_f64.powf(db2 / 10.0);
    10.0 * sum.log10()
}

/// Energetic sum of multiple decibel levels.
#[must_use]
pub fn add_db_multiple(levels: &[f64]) -> f64 {
    let sum: f64 = levels.iter().map(|db| 10.0_f64.powf(db / 10.0)).sum();
    10.0 * sum.log10()
}

/// Subtract background noise: 10 × log₁₀(10^(total/10) - 10^(bg/10)).
#[must_use]
pub fn subtract_db(total_db: f64, background_db: f64) -> f64 {
    let diff = 10.0_f64.powf(total_db / 10.0) - 10.0_f64.powf(background_db / 10.0);
    10.0 * diff.log10()
}

/// Inverse-square distance attenuation: L2 = L1 - 20 × log₁₀(d2 / d1).
#[must_use]
pub fn distance_attenuation(db_at_ref: f64, ref_distance: f64, distance: f64) -> f64 {
    assert!(ref_distance > 0.0, "ref_distance must be positive");
    assert!(distance > 0.0, "distance must be positive");
    db_at_ref - 20.0 * (distance / ref_distance).log10()
}

/// A-weighting filter approximation (dBA relative weighting at a given frequency).
#[must_use]
pub fn a_weighting(frequency: f64) -> f64 {
    let f2 = frequency * frequency;
    let numerator = A_WEIGHT_F4 * A_WEIGHT_F4 * f2 * f2;
    let denominator = (f2 + A_WEIGHT_F1 * A_WEIGHT_F1)
        * (f2 + A_WEIGHT_F4 * A_WEIGHT_F4)
        * ((f2 + A_WEIGHT_F2 * A_WEIGHT_F2) * (f2 + A_WEIGHT_F3 * A_WEIGHT_F3)).sqrt();
    let ra = numerator / denominator;
    20.0 * ra.log10() + A_WEIGHT_OFFSET_DB
}

// ═══════════════════════════════════════════════════════════════════════════
// Psychoacoustics
// ═══════════════════════════════════════════════════════════════════════════

/// Rough equal-loudness approximation in phon.
/// At 1 kHz the phon value equals the SPL. At other frequencies a simple
/// A-weighting-derived correction is applied. This is NOT a full ISO 226
/// implementation.
#[must_use]
pub fn equal_loudness_phon(spl: f64, frequency: f64) -> f64 {
    let correction = a_weighting(frequency) - a_weighting(1000.0);
    spl + correction
}

/// Bark critical-band rate: z = 13 × atan(0.76 f/1000) + 3.5 × atan((f/7500)²).
#[must_use]
pub fn bark_scale(frequency: f64) -> f64 {
    BARK_LOW_COEFF * (BARK_LOW_FACTOR * frequency / BARK_LOW_DIVISOR).atan()
        + BARK_HIGH_COEFF * (frequency / BARK_HIGH_DIVISOR).powi(2).atan()
}

/// Mel scale: m = 2595 × log₁₀(1 + f/700).
#[must_use]
pub fn mel_scale(frequency: f64) -> f64 {
    MEL_BREAKPOINT * (1.0 + frequency / MEL_DIVISOR).log10()
}

/// Inverse mel scale: f = 700 × (10^(m/2595) - 1).
#[must_use]
pub fn frequency_from_mel(mel: f64) -> f64 {
    MEL_DIVISOR * (10.0_f64.powf(mel / MEL_BREAKPOINT) - 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Noise
// ═══════════════════════════════════════════════════════════════════════════

/// Noise reduction through a partition: NR = TL + 10 × log₁₀(A / S).
#[must_use]
pub fn noise_reduction(tl: f64, receiving_absorption: f64, common_area: f64) -> f64 {
    assert!(common_area > 0.0, "common_area must be positive");
    assert!(receiving_absorption > 0.0, "receiving_absorption must be positive");
    tl + 10.0 * (receiving_absorption / common_area).log10()
}

/// Single-panel mass law transmission loss: TL = 20 × log₁₀(m × f) - 47.
#[must_use]
pub fn transmission_loss_mass_law(surface_density: f64, frequency: f64) -> f64 {
    20.0 * (surface_density * frequency).log10() - MASS_LAW_OFFSET_DB
}

/// Rough STC estimate (≈ TL at 500 Hz).
#[must_use]
pub fn sound_transmission_class_estimate(tl_500: f64) -> f64 {
    tl_500
}

// ═══════════════════════════════════════════════════════════════════════════
// Outdoor Sound
// ═══════════════════════════════════════════════════════════════════════════

/// Simplified atmospheric absorption coefficient in dB/km.
/// α ≈ 0.01 × (f/1000)^1.7 × (1 + 0.01×(T-20)) × (1 - 0.005×RH).
#[must_use]
pub fn atmospheric_absorption_coeff(frequency: f64, temperature: f64, humidity: f64) -> f64 {
    ATMOS_BASE_COEFF
        * (frequency / BARK_LOW_DIVISOR).powf(ATMOS_FREQ_EXPONENT)
        * (1.0 + ATMOS_TEMP_FACTOR * (temperature - ATMOS_TEMP_REF))
        * (1.0 - ATMOS_HUMIDITY_FACTOR * humidity)
}

/// Simplified excess ground attenuation (dB) for propagation over soft ground.
/// Uses a basic geometric model based on path-length difference.
#[must_use]
pub fn ground_effect_excess(distance: f64, source_height: f64, receiver_height: f64) -> f64 {
    let direct = (distance * distance
        + (receiver_height - source_height).powi(2))
    .sqrt();
    let reflected = (distance * distance
        + (receiver_height + source_height).powi(2))
    .sqrt();
    let path_diff = reflected - direct;
    // Approximate excess attenuation from destructive interference geometry
    -10.0 * (1.0 + (path_diff / direct).powi(2)).log10()
}

// ── Harmonics & Musical Acoustics ──

/// Nth harmonic frequency: f_n = n × f_fundamental
#[must_use]
pub fn harmonic_frequency(fundamental: f64, n: u32) -> f64 {
    fundamental * n as f64
}

/// Generate harmonic series up to max_harmonic: [f, 2f, 3f, ..., nf]
#[must_use]
pub fn harmonic_series(fundamental: f64, max_harmonic: u32) -> Vec<f64> {
    (1..=max_harmonic).map(|n| fundamental * n as f64).collect()
}

/// Frequency ratio between two musical interval semitones (equal temperament):
/// ratio = 2^(semitones/12)
#[must_use]
pub fn equal_temperament_ratio(semitones: f64) -> f64 {
    2.0_f64.powf(semitones / 12.0)
}

/// Frequency of a note in equal temperament given A4=440Hz reference:
/// f = 440 × 2^((midi_note - 69)/12) where midi_note 69 = A4
#[must_use]
pub fn midi_to_frequency(midi_note: f64) -> f64 {
    const A4_FREQ: f64 = 440.0;
    const A4_MIDI: f64 = 69.0;
    A4_FREQ * 2.0_f64.powf((midi_note - A4_MIDI) / 12.0)
}

/// MIDI note number from frequency: n = 69 + 12×log₂(f/440)
#[must_use]
pub fn frequency_to_midi(frequency: f64) -> f64 {
    assert!(frequency > 0.0, "frequency must be positive");
    const A4_FREQ: f64 = 440.0;
    const A4_MIDI: f64 = 69.0;
    A4_MIDI + 12.0 * (frequency / A4_FREQ).log2()
}

/// Cents difference between two frequencies: c = 1200 × log₂(f2/f1)
#[must_use]
pub fn cents(f1: f64, f2: f64) -> f64 {
    assert!(f1 > 0.0, "f1 must be positive");
    assert!(f2 > 0.0, "f2 must be positive");
    1200.0 * (f2 / f1).log2()
}

/// Frequency of a circular membrane mode (drum):
/// f_mn = (α_mn × v) / (2π × r)
/// where α_mn are zeros of Bessel functions. Common modes:
/// (0,1)=2.405, (1,1)=3.832, (2,1)=5.136, (0,2)=5.520
#[must_use]
pub fn circular_membrane_frequency(bessel_zero: f64, wave_speed: f64, radius: f64) -> f64 {
    assert!(radius > 0.0, "radius must be positive");
    bessel_zero * wave_speed / (2.0 * PI * radius)
}

/// Rectangular plate fundamental frequency:
/// f = (π/(2L²)) × √(D/(ρh)) where D = Eh³/(12(1-ν²))
/// Simplified: takes flexural rigidity D, density×thickness (ρh), and length
#[must_use]
pub fn rectangular_plate_fundamental(length: f64, flexural_rigidity: f64, mass_per_area: f64) -> f64 {
    assert!(length > 0.0, "length must be positive");
    assert!(mass_per_area > 0.0, "mass_per_area must be positive");
    (PI / (2.0 * length * length)) * (flexural_rigidity / mass_per_area).sqrt()
}

/// Harmonic distortion: THD = √(Σ V_n²) / V_1 for n=2..N
/// Takes fundamental amplitude and harmonic amplitudes [2nd, 3rd, ...]
#[must_use]
pub fn total_harmonic_distortion(fundamental_amplitude: f64, harmonic_amplitudes: &[f64]) -> f64 {
    if fundamental_amplitude <= 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = harmonic_amplitudes.iter().map(|a| a * a).sum();
    sum_sq.sqrt() / fundamental_amplitude
}

/// Synthesize a waveform from harmonics at a given time:
/// y(t) = Σ a_n × sin(2π × n × f × t + φ_n)
/// Each tuple is (harmonic_number, amplitude, phase)
#[must_use]
pub fn harmonic_synthesis(fundamental: f64, harmonics: &[(u32, f64, f64)], t: f64) -> f64 {
    let two_pi = 2.0 * PI;
    harmonics.iter().map(|&(n, amp, phase)| {
        amp * (two_pi * n as f64 * fundamental * t + phase).sin()
    }).sum()
}

/// Inharmonicity coefficient for a stiff string (piano):
/// f_n = n × f₁ × √(1 + B × n²) where B = π³Ed⁴/(64TL²)
/// Takes the precomputed inharmonicity coefficient B
#[must_use]
pub fn inharmonic_frequency(fundamental: f64, n: u32, b_coeff: f64) -> f64 {
    let n_f = n as f64;
    n_f * fundamental * (1.0 + b_coeff * n_f * n_f).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;
    const LOOSE_TOLERANCE: f64 = 0.5;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_sabine_reverberation() {
        // 200 m³ room, total absorption 10 m² → T60 = 0.161×200/10 = 3.22 s
        let t60 = sabine_reverberation(200.0, 10.0);
        assert!(approx(t60, 3.22, TOLERANCE), "got {t60}");
    }

    #[test]
    fn test_eyring_reverberation() {
        let t60 = eyring_reverberation(200.0, 100.0, 0.1);
        assert!(approx(t60, 3.056_2, 1e-3), "got {t60}");
    }

    #[test]
    fn test_total_absorption() {
        let surfaces = [(10.0, 0.1), (20.0, 0.3), (5.0, 0.5)];
        let a = total_absorption(&surfaces);
        assert!(approx(a, 9.5, TOLERANCE));
    }

    #[test]
    fn test_add_db_two_equal_sources() {
        // Two equal dB levels → +3 dB
        let result = add_db(80.0, 80.0);
        assert!(
            approx(result, 83.0103, 0.001),
            "two 80 dB sources should give ~83 dB, got {result}"
        );
    }

    #[test]
    fn test_add_db_multiple() {
        let result = add_db_multiple(&[80.0, 80.0, 80.0]);
        // Three equal sources → +4.77 dB
        assert!(approx(result, 84.771, 0.01), "got {result}");
    }

    #[test]
    fn test_subtract_db() {
        // Subtracting equal background from +3dB combined should yield original
        let combined = add_db(80.0, 74.0);
        let signal = subtract_db(combined, 74.0);
        assert!(approx(signal, 80.0, 0.01), "got {signal}");
    }

    #[test]
    fn test_distance_doubling_minus_6db() {
        // Doubling distance → -6 dB
        let l2 = distance_attenuation(90.0, 1.0, 2.0);
        assert!(
            approx(l2, 90.0 - 6.0206, 0.001),
            "doubling distance should give ~-6 dB, got {l2}"
        );
    }

    #[test]
    fn test_mel_scale_1000hz() {
        let mel = mel_scale(1000.0);
        assert!(
            approx(mel, 1000.0, LOOSE_TOLERANCE),
            "mel(1000 Hz) should be ~1000, got {mel}"
        );
    }

    #[test]
    fn test_mel_roundtrip() {
        let freq = 440.0;
        let mel = mel_scale(freq);
        let recovered = frequency_from_mel(mel);
        assert!(approx(recovered, freq, TOLERANCE), "got {recovered}");
    }

    #[test]
    fn test_bark_scale() {
        let z = bark_scale(1000.0);
        // At 1 kHz, Bark ≈ 8.5–9
        assert!(z > 8.0 && z < 10.0, "bark(1000) should be ~8.5–9, got {z}");
    }

    #[test]
    fn test_room_mode_frequency() {
        // First axial mode along length 5m, speed 343 m/s → f = 343/(2×5) = 34.3 Hz
        let f = room_mode_frequency(5.0, 4.0, 3.0, 1, 0, 0, 343.0);
        assert!(approx(f, 34.3, TOLERANCE), "got {f}");
    }

    #[test]
    fn test_mass_law_tl() {
        let tl = transmission_loss_mass_law(10.0, 500.0);
        assert!(approx(tl, 26.9794, 1e-3), "got {tl}");
    }

    #[test]
    fn test_a_weighting_1khz() {
        // A-weighting at 1 kHz should be approximately 0 dB
        let w = a_weighting(1000.0);
        assert!(w.abs() < 1.0, "A-weighting at 1 kHz should be near 0, got {w}");
    }

    #[test]
    fn test_atmospheric_absorption() {
        let coeff = atmospheric_absorption_coeff(4000.0, 20.0, 50.0);
        assert!(coeff > 0.0, "absorption must be positive, got {coeff}");
    }

    #[test]
    fn test_critical_distance() {
        let rc = room_constant(200.0, 0.2);
        let dc = critical_distance(rc, 1.0);
        assert!(dc > 0.0, "critical distance must be positive, got {dc}");
    }

    #[test]
    fn test_noise_reduction() {
        let nr = noise_reduction(40.0, 20.0, 10.0);
        // NR = 40 + 10×log10(20/10) = 40 + 3.0103 ≈ 43.01
        assert!(approx(nr, 43.0103, 0.001), "got {nr}");
    }

    #[test]
    fn test_ground_effect_excess() {
        let excess = ground_effect_excess(100.0, 1.5, 1.5);
        // Should be a small negative value (attenuation)
        assert!(excess <= 0.0, "excess ground attenuation should be <= 0, got {excess}");
    }

    #[test]
    fn test_equal_loudness_at_1khz() {
        // At 1 kHz, phon == SPL
        let phon = equal_loudness_phon(70.0, 1000.0);
        assert!(approx(phon, 70.0, TOLERANCE), "got {phon}");
    }

    #[test]
    fn test_stc_estimate() {
        assert!(approx(sound_transmission_class_estimate(45.0), 45.0, TOLERANCE));
    }

    // ── Harmonics & Musical Acoustics ──

    #[test]
    fn test_harmonic_frequency() {
        assert!(approx(harmonic_frequency(110.0, 3), 330.0, TOLERANCE));
    }

    #[test]
    fn test_harmonic_series() {
        let series = harmonic_series(100.0, 4);
        assert_eq!(series.len(), 4);
        assert!(approx(series[0], 100.0, TOLERANCE));
        assert!(approx(series[3], 400.0, TOLERANCE));
    }

    #[test]
    fn test_equal_temperament_octave() {
        assert!(approx(equal_temperament_ratio(12.0), 2.0, TOLERANCE));
    }

    #[test]
    fn test_midi_a4() {
        assert!(approx(midi_to_frequency(69.0), 440.0, TOLERANCE));
    }

    #[test]
    fn test_midi_roundtrip() {
        let f = 261.63; // Middle C
        let midi = frequency_to_midi(f);
        let f_back = midi_to_frequency(midi);
        assert!(approx(f_back, f, 0.01));
    }

    #[test]
    fn test_cents_octave() {
        assert!(approx(cents(440.0, 880.0), 1200.0, TOLERANCE));
    }

    #[test]
    fn test_thd_pure_tone() {
        assert!(approx(total_harmonic_distortion(1.0, &[]), 0.0, TOLERANCE));
    }

    #[test]
    fn test_thd_with_harmonics() {
        // 3% second harmonic, 4% third = √(0.03²+0.04²) = 0.05 = 5%
        let thd = total_harmonic_distortion(1.0, &[0.03, 0.04]);
        assert!(approx(thd, 0.05, TOLERANCE));
    }

    #[test]
    fn test_harmonic_synthesis_single() {
        // Single fundamental at t=0, phase=0 → sin(0) = 0
        let y = harmonic_synthesis(440.0, &[(1, 1.0, 0.0)], 0.0);
        assert!(approx(y, 0.0, TOLERANCE));
    }

    #[test]
    fn test_inharmonic_b_zero_is_harmonic() {
        // B=0 should give exact harmonics
        assert!(approx(inharmonic_frequency(100.0, 3, 0.0), 300.0, TOLERANCE));
    }

    #[test]
    fn test_inharmonic_stretches_higher() {
        // With B>0, higher harmonics should be sharper (higher) than exact
        let exact = harmonic_frequency(100.0, 5);
        let stretched = inharmonic_frequency(100.0, 5, 0.001);
        assert!(stretched > exact);
    }

    #[test]
    fn test_circular_membrane_frequency() {
        let f = circular_membrane_frequency(2.405, 100.0, 0.5);
        assert!(approx(f, 76.5535, 1e-3), "got {f}");
    }

    #[test]
    fn test_rectangular_plate_fundamental() {
        let f = rectangular_plate_fundamental(1.0, 100.0, 25.0);
        assert!(approx(f, 3.14159, 1e-4), "got {f}");
    }

    #[test]
    fn test_thd_zero_fundamental() {
        let thd = total_harmonic_distortion(0.0, &[0.1, 0.2]);
        assert!(approx(thd, 0.0, 1e-15));
        let thd_neg = total_harmonic_distortion(-1.0, &[0.1]);
        assert!(approx(thd_neg, 0.0, 1e-15));
    }
}
