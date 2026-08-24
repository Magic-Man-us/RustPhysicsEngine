//! Audio-rate oscillators and test signals: PolyBLEP anti-aliased
//! classics, additive resynthesis, mipmapped wavetables, colored noise,
//! chirps, and measurement sweeps.

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::monte_carlo::Rng;
use crate::transforms::fft::{fft_any, ifft_any};

const TWO_PI: f64 = 2.0 * PI;

/// Noise spectra for [`NoiseGen`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseColor {
    White,
    /// −3 dB/octave (Paul Kellet's economy pink filter).
    Pink,
    /// −6 dB/octave (leaky integral of white).
    Brown,
    /// +3 dB/octave (differentiated pink).
    Blue,
    /// +6 dB/octave (differentiated white).
    Violet,
    /// Rough equal-loudness weighting (low + high shelf mix).
    Grey,
}

/// Deterministic colored-noise generator.
pub struct NoiseGen {
    rng: Rng,
    state: [f64; 7],
    color: NoiseColor,
    prev_white: f64,
    prev_pink: f64,
    brown: f64,
}

impl NoiseGen {
    /// Seeded generator of the requested color.
    #[must_use]
    pub fn new(seed: u64, color: NoiseColor) -> Self {
        Self {
            rng: Rng::new(seed),
            state: [0.0; 7],
            color,
            prev_white: 0.0,
            prev_pink: 0.0,
            brown: 0.0,
        }
    }

    fn white(&mut self) -> f64 {
        2.0 * self.rng.next_f64() - 1.0
    }

    fn pink(&mut self, white: f64) -> f64 {
        // Paul Kellet's refined pink filter (−3 dB/oct within ±0.05 dB).
        let b = &mut self.state;
        b[0] = 0.99886 * b[0] + white * 0.0555179;
        b[1] = 0.99332 * b[1] + white * 0.0750759;
        b[2] = 0.96900 * b[2] + white * 0.1538520;
        b[3] = 0.86650 * b[3] + white * 0.3104856;
        b[4] = 0.55000 * b[4] + white * 0.5329522;
        b[5] = -0.7616 * b[5] - white * 0.0168980;
        let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
        b[6] = white * 0.115926;
        pink * 0.11
    }

    /// Next sample, roughly unit peak scale.
    #[allow(clippy::should_implement_trait)] // roadmap API name
    pub fn next(&mut self) -> f64 {
        let w = self.white();
        match self.color {
            NoiseColor::White => w,
            NoiseColor::Pink => self.pink(w),
            NoiseColor::Brown => {
                self.brown = (self.brown + 0.02 * w) / 1.02;
                self.brown * 3.5
            }
            NoiseColor::Blue => {
                let p = self.pink(w);
                let out = p - self.prev_pink;
                self.prev_pink = p;
                out * 4.0
            }
            NoiseColor::Violet => {
                let out = w - self.prev_white;
                self.prev_white = w;
                out * 0.7
            }
            NoiseColor::Grey => {
                // Equal-loudness-ish: boosted lows and highs around a
                // quieter midband (rough approximation).
                self.brown = (self.brown + 0.02 * w) / 1.02;
                let violet = w - self.prev_white;
                self.prev_white = w;
                2.0 * self.brown + 0.3 * w + 0.5 * violet
            }
        }
    }
}

/// Waveform selector for [`Oscillator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform {
    Sine,
    Saw,
    Square { duty: f64 },
    Triangle,
    Noise(NoiseColor),
    /// Index into the oscillator's attached wavetable list.
    Wavetable(usize),
}

/// The PolyBLEP step-correction residual for phase t ∈ \[0, 1) with
/// per-sample increment dt.
fn poly_blep(t: f64, dt: f64) -> f64 {
    if t < dt {
        let u = t / dt;
        u + u - u * u - 1.0
    } else if t > 1.0 - dt {
        let u = (t - 1.0) / dt;
        u * u + u + u + 1.0
    } else {
        0.0
    }
}

/// Anti-aliased sawtooth (−1..1) at phase t with increment dt.
#[must_use]
pub fn polyblep_saw(phase: f64, dt: f64) -> f64 {
    let t = phase.rem_euclid(1.0);
    2.0 * t - 1.0 - poly_blep(t, dt)
}

/// Anti-aliased pulse with the given duty cycle.
#[must_use]
pub fn polyblep_square(phase: f64, dt: f64, duty: f64) -> f64 {
    let t = phase.rem_euclid(1.0);
    let naive = if t < duty { 1.0 } else { -1.0 };
    naive + poly_blep(t, dt) - poly_blep((t - duty).rem_euclid(1.0), dt)
}

/// PolyBLAMP corner residual (integral of the BLEP step residual).
fn poly_blamp(t: f64, dt: f64) -> f64 {
    if t < dt {
        let u = t / dt - 1.0;
        -u * u * u / 3.0
    } else if t > 1.0 - dt {
        let u = (t - 1.0) / dt + 1.0;
        u * u * u / 3.0
    } else {
        0.0
    }
}

/// Anti-aliased triangle (corner smoothing by polyBLAMP; triangle
/// aliasing is already −12 dB/oct so the correction is mild).
#[must_use]
pub fn polyblep_triangle(phase: f64, dt: f64) -> f64 {
    let t = phase.rem_euclid(1.0);
    let naive = 2.0 * (2.0 * (t - (t + 0.5).floor())).abs() - 1.0; // corners at 0 and 0.5
    // Slope changes: +8 at t = 0 (from −4 to +4), −8 at t = 0.5.
    let c0 = 8.0 * dt / 2.0 * poly_blamp(t, dt);
    let c1 = -8.0 * dt / 2.0 * poly_blamp((t + 0.5).rem_euclid(1.0), dt);
    naive + c0 + c1
}

/// Band-limited saw from its Fourier series (n harmonics).
#[must_use]
pub fn additive_saw(phase: f64, n_harmonics: usize) -> f64 {
    let mut acc = 0.0;
    for k in 1..=n_harmonics {
        acc += (TWO_PI * k as f64 * phase).sin() / k as f64;
    }
    -2.0 / PI * acc
}

/// Band-limited square from its Fourier series.
#[must_use]
pub fn additive_square(phase: f64, n_harmonics: usize) -> f64 {
    let mut acc = 0.0;
    let mut k = 1usize;
    while k <= n_harmonics {
        acc += (TWO_PI * k as f64 * phase).sin() / k as f64;
        k += 2;
    }
    4.0 / PI * acc
}

/// Band-limited triangle from its Fourier series.
#[must_use]
pub fn additive_triangle(phase: f64, n_harmonics: usize) -> f64 {
    let mut acc = 0.0;
    let mut k = 1usize;
    let mut sign = 1.0;
    while k <= n_harmonics {
        acc += sign * (TWO_PI * k as f64 * phase).sin() / (k * k) as f64;
        sign = -sign;
        k += 2;
    }
    8.0 / (PI * PI) * acc
}

/// Mipmapped single-cycle wavetable: table m is band-limited so that it
/// can play at up to `base_freqs[m]` without aliasing.
pub struct Wavetable {
    pub tables: Vec<Vec<f64>>,
    pub base_freqs: Vec<f64>,
}

impl Wavetable {
    /// Build mips from one cycle of an arbitrary waveform f(phase),
    /// phase ∈ \[0, 1): each mip keeps the harmonics that stay below
    /// Nyquist at its maximum playback frequency (octave-spaced mips
    /// starting at 20 Hz).
    #[must_use]
    pub fn from_fn(f: impl Fn(f64) -> f64, size: usize, n_mips: usize, fs: f64) -> Self {
        let cycle: Vec<Complex> = (0..size)
            .map(|i| Complex::new(f(i as f64 / size as f64), 0.0))
            .collect();
        let spec = fft_any(&cycle);
        Self::from_spectrum(&spec, size, n_mips, fs)
    }

    /// Build mips from harmonic amplitudes (index 0 = fundamental).
    #[must_use]
    pub fn from_harmonics(amps: &[f64], size: usize, n_mips: usize, fs: f64) -> Self {
        let mut spec = vec![Complex::new(0.0, 0.0); size];
        for (k, &a) in amps.iter().enumerate() {
            let h = k + 1;
            if h < size / 2 {
                // sine harmonics: X[h] = −i·a·size/2
                spec[h] = Complex::new(0.0, -a * size as f64 / 2.0);
                spec[size - h] = Complex::new(0.0, a * size as f64 / 2.0);
            }
        }
        Self::from_spectrum(&spec, size, n_mips, fs)
    }

    fn from_spectrum(spec: &[Complex], size: usize, n_mips: usize, fs: f64) -> Self {
        let mut tables = Vec::with_capacity(n_mips);
        let mut base_freqs = Vec::with_capacity(n_mips);
        for m in 0..n_mips {
            let f_max = 20.0 * 2.0_f64.powi(m as i32);
            let h_max = ((fs / 2.0) / f_max).floor().max(1.0) as usize;
            let mut band = vec![Complex::new(0.0, 0.0); size];
            band[0] = spec[0];
            for h in 1..=h_max.min(size / 2 - 1) {
                band[h] = spec[h];
                band[size - h] = spec[size - h];
            }
            let cycle = ifft_any(&band);
            tables.push(cycle.iter().map(|c| c.re).collect());
            base_freqs.push(f_max);
        }
        Self { tables, base_freqs }
    }

    /// Linear-interpolated lookup at phase ∈ \[0, 1), mip-selected for
    /// playback frequency `freq`.
    #[must_use]
    pub fn lookup(&self, phase: f64, freq: f64) -> f64 {
        // Smallest mip whose rated frequency covers `freq`.
        let m = self
            .base_freqs
            .iter()
            .position(|&b| b >= freq)
            .unwrap_or(self.base_freqs.len() - 1);
        let table = &self.tables[m];
        let n = table.len();
        let x = phase.rem_euclid(1.0) * n as f64;
        let i0 = x.floor() as usize % n;
        let i1 = (i0 + 1) % n;
        let f = x - x.floor();
        table[i0] * (1.0 - f) + table[i1] * f
    }

    /// Band-limited sawtooth wavetable.
    #[must_use]
    pub fn saw(fs: f64) -> Self {
        let amps: Vec<f64> = (1..=512).map(|k| -2.0 / (PI * k as f64)).collect();
        Self::from_harmonics(&amps, 2048, 10, fs)
    }

    /// Band-limited square wavetable.
    #[must_use]
    pub fn square(fs: f64) -> Self {
        let amps: Vec<f64> = (1..=512)
            .map(|k| if k % 2 == 1 { 4.0 / (PI * k as f64) } else { 0.0 })
            .collect();
        Self::from_harmonics(&amps, 2048, 10, fs)
    }

    /// Band-limited triangle wavetable.
    #[must_use]
    pub fn triangle(fs: f64) -> Self {
        let amps: Vec<f64> = (1..=512)
            .map(|k| {
                if k % 2 == 1 {
                    let sign = if (k / 2) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * 8.0 / (PI * PI * (k * k) as f64)
                } else {
                    0.0
                }
            })
            .collect();
        Self::from_harmonics(&amps, 2048, 10, fs)
    }
}

/// Phase-accumulating audio oscillator (PolyBLEP for saw/square,
/// polyBLAMP triangle, seeded noise, optional wavetables).
pub struct Oscillator {
    pub phase: f64,
    pub freq: f64,
    pub fs: f64,
    pub kind: Waveform,
    pub wavetables: Vec<Wavetable>,
    noise: NoiseGen,
}

impl Oscillator {
    /// New oscillator at the given frequency and sample rate.
    #[must_use]
    pub fn new(kind: Waveform, freq: f64, fs: f64) -> Self {
        let color = if let Waveform::Noise(c) = kind { c } else { NoiseColor::White };
        Self { phase: 0.0, freq, fs, kind, wavetables: Vec::new(), noise: NoiseGen::new(0x5EED, color) }
    }

    fn render(&mut self) -> f64 {
        let dt = self.freq / self.fs;
        match self.kind {
            Waveform::Sine => (TWO_PI * self.phase).sin(),
            Waveform::Saw => polyblep_saw(self.phase, dt),
            Waveform::Square { duty } => polyblep_square(self.phase, dt, duty),
            Waveform::Triangle => polyblep_triangle(self.phase, dt),
            Waveform::Noise(_) => self.noise.next(),
            Waveform::Wavetable(i) => {
                if i < self.wavetables.len() {
                    self.wavetables[i].lookup(self.phase, self.freq)
                } else {
                    0.0
                }
            }
        }
    }

    /// Next sample.
    #[allow(clippy::should_implement_trait)] // roadmap API name
    pub fn next(&mut self) -> f64 {
        let out = self.render();
        self.phase = (self.phase + self.freq / self.fs).rem_euclid(1.0);
        out
    }

    /// Change frequency (phase-continuous).
    pub fn set_freq(&mut self, freq: f64) {
        self.freq = freq;
    }

    /// Jump to an absolute phase in \[0, 1).
    pub fn set_phase(&mut self, phase: f64) {
        self.phase = phase.rem_euclid(1.0);
    }

    /// Render a block of n samples.
    pub fn block(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }

    /// One sample with instantaneous frequency modulation (adds
    /// `mod_hz` to the base frequency for this sample).
    pub fn fm(&mut self, mod_hz: f64) -> f64 {
        let out = self.render();
        self.phase = (self.phase + (self.freq + mod_hz) / self.fs).rem_euclid(1.0);
        out
    }

    /// Hard sync: retrigger the phase when `reset` is true.
    pub fn hard_sync(&mut self, reset: bool) {
        if reset {
            self.phase = 0.0;
        }
    }
}

/// Linear chirp from f0 to f1 over `duration` seconds.
#[must_use]
pub fn chirp_linear(f0: f64, f1: f64, duration: f64, fs: f64) -> Vec<f64> {
    chirp_by_freq(|t| f0 + (f1 - f0) * t / duration, duration, fs)
}

/// Exponential (logarithmic-sweep) chirp.
#[must_use]
pub fn chirp_exponential(f0: f64, f1: f64, duration: f64, fs: f64) -> Vec<f64> {
    let k = f1 / f0;
    chirp_by_freq(move |t| f0 * k.powf(t / duration), duration, fs)
}

/// Hyperbolic chirp (linear period sweep).
#[must_use]
pub fn chirp_hyperbolic(f0: f64, f1: f64, duration: f64, fs: f64) -> Vec<f64> {
    chirp_by_freq(
        move |t| f0 * f1 * duration / (f1 * duration - (f1 - f0) * t),
        duration,
        fs,
    )
}

fn chirp_by_freq(f_of_t: impl Fn(f64) -> f64, duration: f64, fs: f64) -> Vec<f64> {
    let n = (duration * fs).round() as usize;
    let dt = 1.0 / fs;
    let mut phase = 0.0;
    (0..n)
        .map(|i| {
            let out = (TWO_PI * phase).sin();
            phase += f_of_t(i as f64 * dt) * dt;
            out
        })
        .collect()
}

/// Farina exponential sweep and its inverse filter: convolving the two
/// yields (a delayed) impulse, the standard impulse-response
/// measurement pair.
#[must_use]
pub fn sine_sweep_with_inverse(f0: f64, f1: f64, duration: f64, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = (duration * fs).round() as usize;
    let l = duration / (f1 / f0).ln();
    let k = TWO_PI * f0 * l;
    let sweep: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (k * ((t / l).exp() - 1.0)).sin()
        })
        .collect();
    // Inverse: time-reversed sweep with an exponential amplitude
    // envelope compensating the sweep's pink energy distribution.
    let mut inverse: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            sweep[n - 1 - i] * (-t / l).exp()
        })
        .collect();
    // Normalize so sweep ⊛ inverse peaks at 1.
    let energy: f64 = sweep
        .iter()
        .zip(inverse.iter().rev())
        .map(|(a, b)| a * b)
        .sum();
    if energy.abs() > 1e-300 {
        for v in inverse.iter_mut() {
            *v /= energy;
        }
    }
    (sweep, inverse)
}

/// Unit impulse at `pos` in an n-sample buffer.
#[must_use]
pub fn impulse(n: usize, pos: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    if pos < n {
        x[pos] = 1.0;
    }
    x
}

/// Constant (DC) buffer.
#[must_use]
pub fn dc(n: usize, level: f64) -> Vec<f64> {
    vec![level; n]
}

/// Sum of sinusoids with per-tone amplitude and phase.
///
/// # Panics
/// Panics if the parameter arrays differ in length.
#[must_use]
pub fn multisine(freqs: &[f64], amps: &[f64], phases: &[f64], n: usize, fs: f64) -> Vec<f64> {
    assert!(
        freqs.len() == amps.len() && freqs.len() == phases.len(),
        "freqs/amps/phases must match"
    );
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            freqs
                .iter()
                .zip(amps.iter().zip(phases))
                .map(|(&f, (&a, &p))| a * (TWO_PI * f * t + p).sin())
                .sum()
        })
        .collect()
}

/// Schroeder-phase multisine of `n_tones` bin-aligned harmonics of
/// fs/n: near-minimal crest factor for broadband excitation.
#[must_use]
pub fn schroeder_phase_multisine(n_tones: usize, n: usize, fs: f64) -> Vec<f64> {
    let df = fs / n as f64;
    let freqs: Vec<f64> = (1..=n_tones).map(|k| k as f64 * df).collect();
    let amps = vec![1.0 / (n_tones as f64).sqrt(); n_tones];
    let phases: Vec<f64> = (1..=n_tones)
        .map(|k| -PI * (k * (k - 1)) as f64 / n_tones as f64)
        .collect();
    multisine(&freqs, &amps, &phases, n, fs)
}

/// Rectangular pulse train: `width` seconds high per period.
#[must_use]
pub fn pulse_train(freq: f64, width: f64, n: usize, fs: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = (i as f64 / fs).rem_euclid(1.0 / freq);
            if t < width {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Band-limited impulse train (all cosine harmonics up to Nyquist,
/// unit DC component).
#[must_use]
pub fn band_limited_impulse_train(freq: f64, n: usize, fs: f64) -> Vec<f64> {
    let h = ((fs / 2.0) / freq).floor() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let mut acc = 1.0;
            for k in 1..=h {
                acc += 2.0 * (TWO_PI * k as f64 * freq * t).cos();
            }
            acc * freq / fs * 2.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::windows::{window, WindowKind};
    use crate::transforms::fft::rfft;

    fn spectrum_db(x: &[f64]) -> Vec<f64> {
        let w = window(WindowKind::BlackmanHarris, x.len(), true);
        let tapered: Vec<f64> = x.iter().zip(&w).map(|(v, wv)| v * wv).collect();
        rfft(&tapered).iter().map(|c| 20.0 * c.norm().max(1e-300).log10()).collect()
    }

    #[test]
    fn test_polyblep_saw_reduces_aliasing() {
        let fs = 48000.0;
        let freq = 440.0;
        let n = 16384;
        let dt = freq / fs;
        let mut phase: f64 = 0.0;
        let mut naive = Vec::with_capacity(n);
        let mut blep = Vec::with_capacity(n);
        for _ in 0..n {
            naive.push(2.0 * phase - 1.0);
            blep.push(polyblep_saw(phase, dt));
            phase = (phase + dt).rem_euclid(1.0);
        }
        let sn = spectrum_db(&naive);
        let sb = spectrum_db(&blep);
        let bin0 = (freq / fs * n as f64).round() as usize;
        // Worst spectral line away from every harmonic (> 60 Hz off).
        let alias_worst = |s: &[f64]| -> f64 {
            let mut worst = f64::MIN;
            for (k, &db) in s.iter().enumerate().skip(bin0) {
                let f = k as f64 * fs / n as f64;
                let harmonic_dist = (f / freq - (f / freq).round()).abs() * freq;
                if harmonic_dist > 60.0 {
                    worst = worst.max(db);
                }
            }
            worst
        };
        let naive_alias = alias_worst(&sn) - sn[bin0];
        let blep_alias = alias_worst(&sb) - sb[bin0];
        assert!(naive_alias > -50.0, "naive aliasing {naive_alias} dB");
        assert!(blep_alias < -65.0, "polyblep aliasing {blep_alias} dB");
        assert!(blep_alias < naive_alias - 25.0, "improvement too small");
    }

    #[test]
    fn test_polyblep_square_triangle_shape() {
        let dt = 0.01;
        // Square averages to 2·duty − 1.
        let n = 1000;
        let mean: f64 = (0..n)
            .map(|i| polyblep_square(i as f64 / n as f64, dt, 0.25))
            .sum::<f64>()
            / n as f64;
        assert!((mean - (-0.5)).abs() < 0.02, "duty mean {mean}");
        // Triangle stays close to the naive triangle away from corners.
        for &p in &[0.1, 0.2, 0.35, 0.7, 0.9] {
            let naive = 2.0 * (2.0 * (p - (p + 0.5_f64).floor())).abs() - 1.0;
            assert!((polyblep_triangle(p, dt) - naive).abs() < 0.05, "at {p}");
        }
    }

    #[test]
    fn test_additive_waveforms_converge() {
        // With many harmonics the additive forms approach the ideals.
        for &p in &[0.13, 0.4, 0.77] {
            assert!((additive_saw(p, 4000) - (2.0 * p - 1.0)).abs() < 0.01, "saw {p}");
            let sq = if p < 0.5 { 1.0 } else { -1.0 };
            assert!((additive_square(p, 4001) - sq).abs() < 0.02, "square {p}");
            let tri = 2.0 / PI * (TWO_PI * p).sin().asin();
            assert!((additive_triangle(p, 401) - tri).abs() < 0.01, "tri {p}");
        }
    }

    #[test]
    fn test_wavetable_mips_and_lookup() {
        let fs = 48000.0;
        let wt = Wavetable::saw(fs);
        assert_eq!(wt.tables.len(), 10);
        // Low-frequency lookup approximates a saw.
        let v = wt.lookup(0.25, 30.0);
        assert!((v - (-0.5)).abs() < 0.05, "saw at 0.25: {v}");
        // High-frequency mip has far fewer harmonics: smoother table.
        let hi_mip = wt.base_freqs.len() - 1;
        let max_slope_hi: f64 = wt.tables[hi_mip]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f64::max);
        let max_slope_lo: f64 =
            wt.tables[0].windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        assert!(max_slope_hi < max_slope_lo, "{max_slope_hi} vs {max_slope_lo}");
        // Oscillator drives the table.
        let mut osc = Oscillator::new(Waveform::Wavetable(0), 110.0, fs);
        osc.wavetables.push(wt);
        let block = osc.block(1000);
        assert!(block.iter().any(|v| v.abs() > 0.5));
    }

    #[test]
    fn test_pink_noise_slope() {
        let mut gen = NoiseGen::new(42, NoiseColor::Pink);
        let n = 1 << 16;
        let x: Vec<f64> = (0..n).map(|_| gen.next()).collect();
        let (freqs, psd) = crate::transforms::spectral::welch(
            &x,
            48000.0,
            4096,
            2048,
            WindowKind::Hann,
        );
        // Fit PSD ∝ f^−α over 20 Hz–20 kHz (3 decades): α ≈ 1
        // (−10 dB/decade) within 1 dB/decade (±0.1 in α).
        let (alpha, _) = crate::transforms::spectral::power_law_fit(&freqs, &psd, 20.0, 20000.0);
        assert!((alpha - 1.0).abs() < 0.1, "pink slope alpha = {alpha}");
    }

    #[test]
    fn test_noise_colors_relative_slopes() {
        let slope = |color: NoiseColor| -> f64 {
            let mut g = NoiseGen::new(7, color);
            let x: Vec<f64> = (0..1 << 15).map(|_| g.next()).collect();
            let (freqs, psd) = crate::transforms::spectral::welch(
                &x,
                48000.0,
                2048,
                1024,
                WindowKind::Hann,
            );
            let (alpha, _) =
                crate::transforms::spectral::power_law_fit(&freqs, &psd, 100.0, 15000.0);
            alpha
        };
        assert!(slope(NoiseColor::White).abs() < 0.15);
        assert!((slope(NoiseColor::Brown) - 2.0).abs() < 0.3);
        assert!((slope(NoiseColor::Violet) + 2.0).abs() < 0.3);
        assert!((slope(NoiseColor::Blue) + 1.0).abs() < 0.3);
    }

    #[test]
    fn test_chirps_track_frequency() {
        let fs = 8000.0;
        for (chirp, expect_mid) in [
            (chirp_linear(100.0, 900.0, 1.0, fs), 500.0),
            (chirp_exponential(100.0, 900.0, 1.0, fs), 300.0),
        ] {
            let mid = &chirp[3600..4400];
            let crossings = mid.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
            let f_est = crossings as f64 / (mid.len() as f64 / fs);
            assert!(
                (f_est - expect_mid).abs() / expect_mid < 0.06,
                "estimated {f_est} vs {expect_mid}"
            );
        }
    }

    #[test]
    fn test_farina_sweep_gives_impulse() {
        let fs = 8000.0;
        let (sweep, inverse) = sine_sweep_with_inverse(50.0, 3000.0, 0.5, fs);
        let conv = crate::transforms::fft::fft_convolve(&sweep, &inverse);
        let (peak_idx, peak) = conv
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, &v)| (i, v.abs()))
            .unwrap();
        // Energy concentrates at the peak: sidelobes at least 20 dB down
        // outside a 10 ms window.
        let guard = (0.005 * fs) as usize;
        let side = conv
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i as i64 - peak_idx as i64).unsigned_abs() as usize > guard)
            .map(|(_, &v)| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(side < 0.1 * peak, "sidelobe {side} vs peak {peak}");
    }

    #[test]
    fn test_multisine_and_schroeder_crest() {
        let fs = 4096.0;
        let n = 4096;
        let x = schroeder_phase_multisine(31, n, fs);
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let peak = x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let crest = peak / rms;
        assert!(crest < 2.6, "Schroeder crest {crest}");
        // Random-phase reference is worse.
        let phases: Vec<f64> = (0..31).map(|k| ((k * 2654435761_usize) % 628) as f64 / 100.0).collect();
        let freqs: Vec<f64> = (1..=31).map(|k| k as f64).collect();
        let amps = vec![1.0 / (31.0_f64).sqrt(); 31];
        let y = multisine(&freqs, &amps, &phases, n, fs);
        let rms_y = (y.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let crest_y = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) / rms_y;
        assert!(crest < crest_y, "{crest} vs {crest_y}");
    }

    #[test]
    fn test_oscillator_pitch_and_fm() {
        let fs = 48000.0;
        let mut osc = Oscillator::new(Waveform::Sine, 440.0, fs);
        let x = osc.block(4800);
        let crossings = x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        assert!((crossings as f64 - 44.0).abs() <= 1.0);
        osc.set_phase(0.0);
        osc.set_freq(880.0);
        let x2 = osc.block(4800);
        let crossings2 = x2.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        assert!((crossings2 as f64 - 88.0).abs() <= 1.0);
        // FM with +440: doubles the rate again.
        let mut fmv = Vec::with_capacity(4800);
        for _ in 0..4800 {
            fmv.push(osc.fm(880.0));
        }
        let crossings3 = fmv.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        assert!((crossings3 as f64 - 176.0).abs() <= 2.0);
        osc.hard_sync(true);
        assert_eq!(osc.phase, 0.0);
    }

    #[test]
    fn test_pulse_trains_and_misc() {
        let fs = 1000.0;
        let pt = pulse_train(10.0, 0.02, 1000, fs);
        let high = pt.iter().filter(|&&v| v > 0.5).count();
        assert!((high as f64 - 200.0).abs() < 15.0, "duty count {high}");
        let blit = band_limited_impulse_train(100.0, 1000, fs);
        // Periodic with period fs/freq = 10 samples.
        for i in 0..900 {
            assert!((blit[i] - blit[i + 10]).abs() < 1e-9);
        }
        assert_eq!(impulse(8, 3)[3], 1.0);
        assert_eq!(dc(4, 0.5), vec![0.5; 4]);
    }
}
