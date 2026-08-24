//! Musical tuning: temperaments, interval math, Scala parsing,
//! consonance models, stretch tuning, and pitch-class utilities.

use crate::error::SolveError;

/// MIDI frequencies (128 entries) for an equal temperament with
/// `n_divisions` steps per octave anchored at (`base_midi`, `base_hz`).
#[must_use]
pub fn equal_temperament(n_divisions: u32, base_hz: f64, base_midi: u8) -> Vec<f64> {
    (0..128)
        .map(|m| base_hz * 2.0_f64.powf((m as f64 - base_midi as f64) / n_divisions as f64))
        .collect()
}

/// 5-limit just intonation ratios from the tonic.
#[must_use]
pub fn just_intonation_5limit() -> [f64; 12] {
    [
        1.0,
        16.0 / 15.0,
        9.0 / 8.0,
        6.0 / 5.0,
        5.0 / 4.0,
        4.0 / 3.0,
        45.0 / 32.0,
        3.0 / 2.0,
        8.0 / 5.0,
        5.0 / 3.0,
        9.0 / 5.0,
        15.0 / 8.0,
    ]
}

/// Pythagorean (3-limit) chromatic scale ratios.
#[must_use]
pub fn pythagorean() -> [f64; 12] {
    [
        1.0,
        256.0 / 243.0,
        9.0 / 8.0,
        32.0 / 27.0,
        81.0 / 64.0,
        4.0 / 3.0,
        729.0 / 512.0,
        3.0 / 2.0,
        128.0 / 81.0,
        27.0 / 16.0,
        16.0 / 9.0,
        243.0 / 128.0,
    ]
}

fn ratios_from_fifth(fifth: f64) -> [f64; 12] {
    // Chain of fifths from -5 (Ab) to +6 (F#), reduced into one octave,
    // mapped to chromatic degrees: degree of k fifths = (7k) mod 12.
    let mut out = [0.0_f64; 12];
    for k in -5..=6_i32 {
        let degree = (7 * k).rem_euclid(12) as usize;
        let mut r = fifth.powi(k);
        while r < 1.0 {
            r *= 2.0;
        }
        while r >= 2.0 {
            r /= 2.0;
        }
        out[degree] = r;
    }
    out
}

/// Quarter-comma meantone: fifths flattened so major thirds are pure 5/4.
#[must_use]
pub fn meantone_quarter_comma() -> [f64; 12] {
    ratios_from_fifth(5.0_f64.powf(0.25))
}

/// Werckmeister III well temperament (1691), as ratios from C.
#[must_use]
pub fn werckmeister_iii() -> [f64; 12] {
    const CENTS: [f64; 12] = [
        0.0, 90.225, 192.18, 294.135, 390.225, 498.045, 588.27, 696.09, 792.18, 888.27,
        996.09, 1092.18,
    ];
    CENTS.map(cents_to_ratio)
}

/// Kirnberger III well temperament, as ratios from C.
#[must_use]
pub fn kirnberger_iii() -> [f64; 12] {
    const CENTS: [f64; 12] = [
        0.0, 90.225, 193.157, 294.135, 386.314, 498.045, 590.224, 696.578, 792.18, 889.735,
        996.09, 1088.269,
    ];
    CENTS.map(cents_to_ratio)
}

/// Thomas Young's 1799 well temperament (Young II), as ratios from C.
#[must_use]
pub fn young() -> [f64; 12] {
    const CENTS: [f64; 12] = [
        0.0, 93.9, 195.8, 297.8, 391.7, 499.9, 591.9, 697.9, 795.8, 893.8, 999.8, 1091.8,
    ];
    CENTS.map(cents_to_ratio)
}

/// Bohlen-Pierce scale: 13 equal divisions of the tritave (3:1); returns
/// the 14 ratios including both endpoints.
#[must_use]
pub fn bohlen_pierce() -> Vec<f64> {
    (0..=13).map(|k| 3.0_f64.powf(k as f64 / 13.0)).collect()
}

/// Harmonic-series scale: partials n..2n reduced to ratios from 1 to 2.
#[must_use]
pub fn harmonic_series_scale(n: usize) -> Vec<f64> {
    (0..=n).map(|k| (n + k) as f64 / n as f64).collect()
}

/// Parse a Scala `.scl` file body into cents values (one per scale
/// degree, ending with the octave entry). Ratios like `3/2` and cents
/// like `701.955` are both accepted.
pub fn scala_parse(scl: &str) -> Result<Vec<f64>, SolveError> {
    let mut lines = scl.lines().filter(|l| !l.trim_start().starts_with('!'));
    let _description = lines.next().ok_or(SolveError::InvalidArgument("malformed .scl data"))?;
    let count: usize = lines
        .next()
        .ok_or(SolveError::InvalidArgument("malformed .scl data"))?
        .trim()
        .parse()
        .map_err(|_| SolveError::InvalidArgument("malformed .scl data"))?;
    let mut out = Vec::with_capacity(count);
    for line in lines {
        let token = line.split_whitespace().next().unwrap_or("");
        if token.is_empty() {
            continue;
        }
        let cents = if token.contains('/') {
            let mut parts = token.splitn(2, '/');
            let num: f64 = parts
                .next()
                .unwrap()
                .parse()
                .map_err(|_| SolveError::InvalidArgument("malformed .scl data"))?;
            let den: f64 = parts
                .next()
                .unwrap()
                .parse()
                .map_err(|_| SolveError::InvalidArgument("malformed .scl data"))?;
            if den <= 0.0 || num <= 0.0 {
                return Err(SolveError::InvalidArgument("malformed .scl data"));
            }
            ratio_to_cents(num / den)
        } else if token.contains('.') {
            token.parse().map_err(|_| SolveError::InvalidArgument("malformed .scl data"))?
        } else {
            // Integer without a dot is a ratio numerator (e.g. "2" = 2/1).
            let r: f64 = token.parse().map_err(|_| SolveError::InvalidArgument("malformed .scl data"))?;
            if r <= 0.0 {
                return Err(SolveError::InvalidArgument("malformed .scl data"));
            }
            ratio_to_cents(r)
        };
        out.push(cents);
        if out.len() == count {
            break;
        }
    }
    if out.len() != count {
        return Err(SolveError::InvalidArgument("malformed .scl data"));
    }
    Ok(out)
}

/// Signed interval from `f1` to `f2` in cents.
#[must_use]
pub fn cents_between(f1: f64, f2: f64) -> f64 {
    1200.0 * (f2 / f1).log2()
}

/// Frequency ratio to cents.
#[must_use]
pub fn ratio_to_cents(r: f64) -> f64 {
    1200.0 * r.log2()
}

/// Cents to frequency ratio.
#[must_use]
pub fn cents_to_ratio(c: f64) -> f64 {
    2.0_f64.powf(c / 1200.0)
}

/// Nearest 12-TET MIDI note to `freq` for the given A4: returns
/// (midi, cents deviation from that note).
#[must_use]
pub fn nearest_note(freq: f64, a4: f64) -> (u8, f64) {
    let midi_f = 69.0 + 12.0 * (freq / a4).log2();
    let midi = midi_f.round().clamp(0.0, 127.0);
    (midi as u8, 100.0 * (midi_f - midi))
}

/// Name of the just interval closest to `ratio` (within 6 cents), or
/// "unknown".
#[must_use]
pub fn interval_name(ratio: f64) -> &'static str {
    const TABLE: [(f64, &str); 14] = [
        (1.0, "unison"),
        (16.0 / 15.0, "minor second"),
        (9.0 / 8.0, "major second"),
        (6.0 / 5.0, "minor third"),
        (5.0 / 4.0, "major third"),
        (4.0 / 3.0, "perfect fourth"),
        (45.0 / 32.0, "tritone"),
        (3.0 / 2.0, "perfect fifth"),
        (8.0 / 5.0, "minor sixth"),
        (5.0 / 3.0, "major sixth"),
        (16.0 / 9.0, "minor seventh"),
        (15.0 / 8.0, "major seventh"),
        (2.0, "octave"),
        (3.0, "tritave"),
    ];
    for (r, name) in TABLE {
        if ratio_to_cents(ratio / r).abs() < 6.0 {
            return name;
        }
    }
    "unknown"
}

fn plomp_levelt_dissonance(f1: f64, f2: f64) -> f64 {
    let (lo, hi) = if f1 < f2 { (f1, f2) } else { (f2, f1) };
    let s = 0.24 / (0.021 * lo + 19.0);
    let x = s * (hi - lo);
    (-3.5 * x).exp() - (-5.75 * x).exp()
}

/// Plomp-Levelt consonance of two pure tones: 1 at unison, minimum near
/// a quarter of a critical band apart.
#[must_use]
pub fn consonance_plomp_levelt(f1: f64, f2: f64) -> f64 {
    // The dissonance kernel peaks at ≈ 0.1808.
    1.0 - plomp_levelt_dissonance(f1, f2) / 0.180_86
}

/// Sethares dissonance curve: total pairwise Plomp-Levelt dissonance of
/// two copies of a `partials` timbre (`(ratio, amplitude)` relative to
/// `base` Hz) as the second copy sweeps through `ratio_range`. Returns
/// `n` points of (interval ratio, dissonance).
#[must_use]
pub fn dissonance_curve(
    base: f64,
    partials: &[(f64, f64)],
    ratio_range: (f64, f64),
    n: usize,
) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let r = ratio_range.0
                + (ratio_range.1 - ratio_range.0) * i as f64 / (n - 1).max(1) as f64;
            let mut d = 0.0;
            for &(r1, a1) in partials {
                for &(r2, a2) in partials {
                    d += a1 * a2 * plomp_levelt_dissonance(base * r1, base * r * r2);
                }
            }
            (r, d)
        })
        .collect()
}

/// Piano stretch tuning deviation (cents from 12-TET) for a constant
/// string inharmonicity coefficient `b`: octaves are widened so partial 2
/// of the lower note matches the fundamental of its octave.
#[must_use]
pub fn stretch_tuning_railsback(midi: f64, b: f64) -> f64 {
    let stretched_octave = 2.0 * ((1.0 + 4.0 * b) / (1.0 + b)).sqrt();
    let excess = ratio_to_cents(stretched_octave) - 1200.0;
    (midi - 69.0) / 12.0 * excess
}

/// The syntonic comma 81/80.
#[must_use]
pub fn syntonic_comma() -> f64 {
    81.0 / 80.0
}

/// The Pythagorean comma 3¹²/2¹⁹.
#[must_use]
pub fn pythagorean_comma() -> f64 {
    3.0_f64.powi(12) / 2.0_f64.powi(19)
}

/// The schisma 32805/32768 (Pythagorean comma / syntonic comma).
#[must_use]
pub fn schisma() -> f64 {
    32805.0 / 32768.0
}

/// Frequency of a MIDI note in a 12-tone `temperament` (ratios from the
/// tonic C), anchored so that A4 (MIDI 69) sounds at `a4`.
#[must_use]
pub fn midi_to_freq_tuned(midi: u8, a4: f64, temperament: &[f64]) -> f64 {
    assert_eq!(temperament.len(), 12, "temperament must have 12 degrees");
    let c4 = a4 / temperament[9];
    let rel = midi as i32 - 60;
    let octave = rel.div_euclid(12);
    let pc = rel.rem_euclid(12) as usize;
    c4 * 2.0_f64.powi(octave) * temperament[pc]
}

/// Pitch classes reached by successive fifths from `start`.
#[must_use]
pub fn circle_of_fifths(start: u8, n: usize) -> Vec<u8> {
    (0..n).map(|k| ((start as usize + 7 * k) % 12) as u8).collect()
}

/// Diatonic modes and common scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Ionian,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Aeolian,
    Locrian,
    HarmonicMinor,
    MelodicMinor,
    MajorPentatonic,
    MinorPentatonic,
    Blues,
    WholeTone,
    Chromatic,
}

/// Pitch classes of a scale on `root` (semitones 0-11, ascending).
#[must_use]
pub fn scale_degrees(root: u8, mode: Mode) -> Vec<u8> {
    let steps: &[u8] = match mode {
        Mode::Ionian => &[0, 2, 4, 5, 7, 9, 11],
        Mode::Dorian => &[0, 2, 3, 5, 7, 9, 10],
        Mode::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
        Mode::Lydian => &[0, 2, 4, 6, 7, 9, 11],
        Mode::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
        Mode::Aeolian => &[0, 2, 3, 5, 7, 8, 10],
        Mode::Locrian => &[0, 1, 3, 5, 6, 8, 10],
        Mode::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
        Mode::MelodicMinor => &[0, 2, 3, 5, 7, 9, 11],
        Mode::MajorPentatonic => &[0, 2, 4, 7, 9],
        Mode::MinorPentatonic => &[0, 3, 5, 7, 10],
        Mode::Blues => &[0, 3, 5, 6, 7, 10],
        Mode::WholeTone => &[0, 2, 4, 6, 8, 10],
        Mode::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    };
    steps.iter().map(|s| (root + s) % 12).collect()
}

/// Chord qualities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordQuality {
    Major,
    Minor,
    Diminished,
    Augmented,
    Major7,
    Minor7,
    Dominant7,
    HalfDiminished7,
    Diminished7,
    Sus2,
    Sus4,
}

/// Pitch classes of a chord on `root` (semitones 0-11).
#[must_use]
pub fn chord_tones(root: u8, quality: ChordQuality) -> Vec<u8> {
    let steps: &[u8] = match quality {
        ChordQuality::Major => &[0, 4, 7],
        ChordQuality::Minor => &[0, 3, 7],
        ChordQuality::Diminished => &[0, 3, 6],
        ChordQuality::Augmented => &[0, 4, 8],
        ChordQuality::Major7 => &[0, 4, 7, 11],
        ChordQuality::Minor7 => &[0, 3, 7, 10],
        ChordQuality::Dominant7 => &[0, 4, 7, 10],
        ChordQuality::HalfDiminished7 => &[0, 3, 6, 10],
        ChordQuality::Diminished7 => &[0, 3, 6, 9],
        ChordQuality::Sus2 => &[0, 2, 7],
        ChordQuality::Sus4 => &[0, 5, 7],
    };
    steps.iter().map(|s| (root + s) % 12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_temperament_and_intervals() {
        let et = equal_temperament(12, 440.0, 69);
        assert!((et[69] - 440.0).abs() < 1e-12);
        for m in 1..128 {
            assert!((cents_between(et[m - 1], et[m]) - 100.0).abs() < 1e-9);
        }
        assert!((et[60] - 261.6256).abs() < 1e-3);
        let et19 = equal_temperament(19, 440.0, 69);
        assert!((cents_between(et19[69], et19[70]) - 1200.0 / 19.0).abs() < 1e-9);
        assert!((ratio_to_cents(cents_to_ratio(345.6)) - 345.6).abs() < 1e-12);
    }

    #[test]
    fn test_temperaments() {
        let ji = just_intonation_5limit();
        assert!((ratio_to_cents(ji[7]) - 701.955).abs() < 1e-3);
        let py = pythagorean();
        assert!((py[7] - 1.5).abs() < 1e-12);
        // Quarter-comma meantone has a pure major third: C-E = 5/4.
        let mt = meantone_quarter_comma();
        assert!((mt[4] - 1.25).abs() < 1e-9, "meantone third {}", mt[4]);
        // Werckmeister III: pure fourth C-F.
        let w = werckmeister_iii();
        assert!((ratio_to_cents(w[5]) - 498.045).abs() < 0.01);
        let k = kirnberger_iii();
        assert!((ratio_to_cents(k[4]) - 386.314).abs() < 0.01); // pure third
        let y = young();
        assert_eq!(y.len(), 12);
        let bp = bohlen_pierce();
        assert_eq!(bp.len(), 14);
        assert!((bp[13] - 3.0).abs() < 1e-12);
        let hs = harmonic_series_scale(8);
        assert!((hs[8] - 2.0).abs() < 1e-12);
        assert!((hs[4] - 1.5).abs() < 1e-12);
        // Commas.
        assert!((ratio_to_cents(pythagorean_comma()) - 23.46).abs() < 0.01);
        assert!((ratio_to_cents(syntonic_comma()) - 21.506).abs() < 0.01);
        assert!(
            (pythagorean_comma() / syntonic_comma() - schisma()).abs() < 1e-12
        );
    }

    #[test]
    fn test_scala_and_notes() {
        let scl = "! example.scl\n!\nA 5-note test scale\n5\n 100.0\n 9/8\n 300.0\n 3/2\n 2\n";
        let cents = scala_parse(scl).unwrap();
        assert_eq!(cents.len(), 5);
        assert!((cents[0] - 100.0).abs() < 1e-9);
        assert!((cents[1] - 203.91).abs() < 0.01);
        assert!((cents[3] - 701.955).abs() < 0.01);
        assert!((cents[4] - 1200.0).abs() < 1e-9);
        assert!(scala_parse("only a description").is_err());
        let (midi, off) = nearest_note(442.0, 440.0);
        assert_eq!(midi, 69);
        assert!((off - 7.85).abs() < 0.05);
        assert_eq!(interval_name(1.5), "perfect fifth");
        assert_eq!(interval_name(1.251), "major third");
        assert_eq!(interval_name(1.26), "unknown"); // 13.8 cents from 5/4
        assert_eq!(interval_name(1.33), "perfect fourth");
        assert_eq!(interval_name(1.111), "unknown");
    }

    #[test]
    fn test_consonance_and_stretch() {
        // Unison fully consonant; a ~30 Hz gap at 440 is rough.
        assert!(consonance_plomp_levelt(440.0, 440.0) > 0.999);
        let rough = consonance_plomp_levelt(440.0, 470.0);
        assert!(rough < 0.3, "roughness at 30 Hz gap: {rough}");
        // Harmonic timbre: the octave is a deep minimum of dissonance.
        let partials: Vec<(f64, f64)> =
            (1..=6).map(|k| (k as f64, 1.0 / k as f64)).collect();
        let curve = dissonance_curve(261.63, &partials, (1.85, 2.15), 61);
        let d_at = |r: f64| -> f64 {
            curve
                .iter()
                .min_by(|a, b| (a.0 - r).abs().partial_cmp(&(b.0 - r).abs()).unwrap())
                .unwrap()
                .1
        };
        assert!(d_at(2.0) < d_at(1.93));
        assert!(d_at(2.0) < d_at(2.07));
        // Stretch tuning: zero for ideal strings, widening octaves else.
        assert_eq!(stretch_tuning_railsback(93.0, 0.0), 0.0);
        let up = stretch_tuning_railsback(93.0, 4e-4);
        let down = stretch_tuning_railsback(45.0, 4e-4);
        assert!(up > 0.5 && down < -0.5, "stretch {up} / {down}");
    }

    #[test]
    fn test_pitch_class_utilities() {
        let et: Vec<f64> = vec![
            1.0,
            cents_to_ratio(100.0),
            cents_to_ratio(200.0),
            cents_to_ratio(300.0),
            cents_to_ratio(400.0),
            cents_to_ratio(500.0),
            cents_to_ratio(600.0),
            cents_to_ratio(700.0),
            cents_to_ratio(800.0),
            cents_to_ratio(900.0),
            cents_to_ratio(1000.0),
            cents_to_ratio(1100.0),
        ];
        assert!((midi_to_freq_tuned(69, 440.0, &et) - 440.0).abs() < 1e-9);
        assert!((midi_to_freq_tuned(60, 440.0, &et) - 261.6256).abs() < 1e-3);
        // Just-intoned E4 above C4 anchored at A4=440: C4 = 440/(5/3).
        let ji = just_intonation_5limit();
        let e4 = midi_to_freq_tuned(64, 440.0, &ji);
        assert!((e4 - 440.0 / (5.0 / 3.0) * 1.25).abs() < 1e-9);
        let fifths = circle_of_fifths(0, 13);
        assert_eq!(fifths[1], 7);
        assert_eq!(fifths[12], 0); // closes after 12 steps
        assert_eq!(scale_degrees(0, Mode::Ionian), vec![0, 2, 4, 5, 7, 9, 11]);
        assert_eq!(scale_degrees(2, Mode::Dorian), vec![2, 4, 5, 7, 9, 11, 0]);
        assert_eq!(scale_degrees(0, Mode::Chromatic).len(), 12);
        assert_eq!(chord_tones(0, ChordQuality::Major), vec![0, 4, 7]);
        assert_eq!(chord_tones(9, ChordQuality::Minor7), vec![9, 0, 4, 7]);
        assert_eq!(chord_tones(11, ChordQuality::Diminished), vec![11, 2, 5]);
    }
}
