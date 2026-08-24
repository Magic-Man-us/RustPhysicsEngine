//! Audio effects: delays, reverbs (Schroeder, Freeverb, FDN),
//! convolution, modulation effects, dynamics, distortion, EQ, imaging,
//! loudness (ITU-R BS.1770), and dithering.

use crate::audio::envelope::Lfo;
use crate::audio::oscillators::{Oscillator, Waveform};
use crate::dsp::iir::Biquad;
use crate::fractals::Complex;
use crate::linalg::Matrix;
use crate::monte_carlo::Rng;
use crate::transforms::fft::{fft, ifft, next_power_of_two};

// --- Delay primitives ----------------------------------------------------

/// Circular delay line with fractional read.
pub struct DelayLine {
    buf: Vec<f64>,
    pos: usize,
}

impl DelayLine {
    /// New line holding up to `max_samples`.
    #[must_use]
    pub fn new(max_samples: usize) -> Self {
        Self { buf: vec![0.0; max_samples.max(1)], pos: 0 }
    }

    /// Push one input sample.
    pub fn write(&mut self, x: f64) {
        self.pos = (self.pos + 1) % self.buf.len();
        self.buf[self.pos] = x;
    }

    /// Read `delay_samples` behind the write head.
    #[must_use]
    pub fn read(&self, delay_samples: usize) -> f64 {
        let n = self.buf.len();
        self.buf[(self.pos + n - delay_samples.min(n - 1)) % n]
    }

    /// Linear-interpolated fractional read.
    #[must_use]
    pub fn read_interp(&self, delay_frac: f64) -> f64 {
        let d0 = delay_frac.floor();
        let f = delay_frac - d0;
        let a = self.read(d0 as usize);
        let b = self.read(d0 as usize + 1);
        a * (1.0 - f) + b * f
    }

    /// Alias of [`read`] for tap taps.
    #[must_use]
    pub fn tap(&self, d: usize) -> f64 {
        self.read(d)
    }
}

/// Feedback comb filter with a one-pole damping low-pass in the loop.
pub struct CombFilter {
    delay: DelayLine,
    delay_samples: usize,
    pub feedback: f64,
    pub damping: f64,
    lp_state: f64,
}

impl CombFilter {
    /// New comb of the given loop length.
    #[must_use]
    pub fn new(delay_samples: usize, feedback: f64, damping: f64) -> Self {
        Self {
            delay: DelayLine::new(delay_samples + 2),
            delay_samples,
            feedback,
            damping,
            lp_state: 0.0,
        }
    }

    /// One sample through the comb.
    pub fn process(&mut self, x: f64) -> f64 {
        // Read happens before this step's write, so the sample written
        // `delay_samples` steps ago sits at index `delay_samples - 1`.
        let out = self.delay.read(self.delay_samples - 1);
        self.lp_state = out * (1.0 - self.damping) + self.lp_state * self.damping;
        self.delay.write(x + self.feedback * self.lp_state);
        out
    }
}

/// Schroeder all-pass diffuser.
pub struct AllpassFilter {
    delay: DelayLine,
    delay_samples: usize,
    pub gain: f64,
}

impl AllpassFilter {
    /// New all-pass of the given loop length.
    #[must_use]
    pub fn new(delay_samples: usize, gain: f64) -> Self {
        Self { delay: DelayLine::new(delay_samples + 2), delay_samples, gain }
    }

    /// One sample through the all-pass.
    pub fn process(&mut self, x: f64) -> f64 {
        let delayed = self.delay.read(self.delay_samples - 1);
        let w = x + self.gain * delayed;
        self.delay.write(w);
        delayed - self.gain * w
    }
}

// --- Reverbs -------------------------------------------------------------

/// Classic Schroeder reverberator: four parallel combs into two series
/// all-passes.
pub struct SchroederReverb {
    combs: [CombFilter; 4],
    allpasses: [AllpassFilter; 2],
    fs: f64,
}

impl SchroederReverb {
    const COMB_MS: [f64; 4] = [29.7, 37.1, 41.1, 43.7];

    /// New reverb at the given sample rate (RT60 initially 1 s).
    #[must_use]
    pub fn new(fs: f64) -> Self {
        let mk = |ms: f64| CombFilter::new((ms * 1e-3 * fs) as usize, 0.8, 0.2);
        let mut rv = Self {
            combs: [mk(Self::COMB_MS[0]), mk(Self::COMB_MS[1]), mk(Self::COMB_MS[2]), mk(Self::COMB_MS[3])],
            allpasses: [
                AllpassFilter::new((5.0e-3 * fs) as usize, 0.7),
                AllpassFilter::new((1.7e-3 * fs) as usize, 0.7),
            ],
            fs,
        };
        rv.set_rt60(1.0);
        rv
    }

    /// Set the decay time: comb feedback g = 10^(−3·delay/RT60).
    pub fn set_rt60(&mut self, t: f64) {
        for (comb, &ms) in self.combs.iter_mut().zip(&Self::COMB_MS) {
            let loop_t = ms * 1e-3 * (self.fs * ms * 1e-3).round() / (self.fs * ms * 1e-3).max(1.0);
            let _ = loop_t;
            let delay_t = (ms * 1e-3 * self.fs).floor() / self.fs;
            comb.feedback = 10.0_f64.powf(-3.0 * delay_t / t.max(1e-3));
        }
    }

    /// Set high-frequency damping (0..1) inside the comb loops.
    pub fn set_damping(&mut self, d: f64) {
        for comb in self.combs.iter_mut() {
            comb.damping = d.clamp(0.0, 0.99);
        }
    }

    /// One (wet-only) sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let sum: f64 = self.combs.iter_mut().map(|c| c.process(x)).sum();
        let mut v = sum / 4.0;
        for ap in self.allpasses.iter_mut() {
            v = ap.process(v);
        }
        v
    }
}

/// Jezar's Freeverb topology: 8 combs + 4 all-passes per channel with
/// a fixed stereo spread.
pub struct Freeverb {
    combs_l: Vec<CombFilter>,
    combs_r: Vec<CombFilter>,
    aps_l: Vec<AllpassFilter>,
    aps_r: Vec<AllpassFilter>,
}

impl Freeverb {
    const COMBS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
    const APS: [usize; 4] = [556, 441, 341, 225];
    const SPREAD: usize = 23;

    /// New Freeverb; the classic tunings are for 44.1 kHz and are
    /// scaled to the requested rate.
    #[must_use]
    pub fn new(fs: f64) -> Self {
        let scale = fs / 44100.0;
        let mk = |d: usize, off: usize| {
            CombFilter::new(((d + off) as f64 * scale) as usize, 0.84, 0.2)
        };
        let ap = |d: usize, off: usize| {
            AllpassFilter::new(((d + off) as f64 * scale) as usize, 0.5)
        };
        Self {
            combs_l: Self::COMBS.iter().map(|&d| mk(d, 0)).collect(),
            combs_r: Self::COMBS.iter().map(|&d| mk(d, Self::SPREAD)).collect(),
            aps_l: Self::APS.iter().map(|&d| ap(d, 0)).collect(),
            aps_r: Self::APS.iter().map(|&d| ap(d, Self::SPREAD)).collect(),
        }
    }

    /// One input sample to a stereo pair (wet only).
    pub fn process(&mut self, x: f64) -> (f64, f64) {
        let mut l: f64 = self.combs_l.iter_mut().map(|c| c.process(x)).sum();
        let mut r: f64 = self.combs_r.iter_mut().map(|c| c.process(x)).sum();
        l /= 8.0;
        r /= 8.0;
        for ap in self.aps_l.iter_mut() {
            l = ap.process(l);
        }
        for ap in self.aps_r.iter_mut() {
            r = ap.process(r);
        }
        (l, r)
    }
}

/// Feedback delay network reverb with an orthogonal mixing matrix.
pub struct Fdn {
    delays: Vec<DelayLine>,
    delay_samples: Vec<usize>,
    pub matrix: Matrix,
    pub gains: Vec<f64>,
    filters: Vec<Biquad>,
}

impl Fdn {
    /// New FDN with explicit delays (samples) and an identity-free
    /// Householder mixing matrix.
    #[must_use]
    pub fn new(n: usize, delay_samples: &[usize], fs: f64) -> Self {
        let _ = fs;
        assert_eq!(delay_samples.len(), n, "need one delay per line");
        Self {
            delays: delay_samples.iter().map(|&d| DelayLine::new(d + 2)).collect(),
            delay_samples: delay_samples.to_vec(),
            matrix: Self::householder_matrix(n),
            gains: vec![1.0; n],
            filters: vec![Biquad::identity(); n],
        }
    }

    /// Householder reflection H = I − (2/n)·11ᵀ (orthogonal, lossless).
    #[must_use]
    pub fn householder_matrix(n: usize) -> Matrix {
        Matrix::from_fn(n, n, |i, j| {
            let base = -2.0 / n as f64;
            if i == j {
                1.0 + base
            } else {
                base
            }
        })
    }

    /// Hadamard mixing matrix (n must be a power of two), scaled to be
    /// orthogonal.
    ///
    /// # Panics
    /// Panics unless n is a power of two.
    #[must_use]
    pub fn hadamard_matrix(n: usize) -> Matrix {
        assert!(n.is_power_of_two(), "Hadamard needs a power-of-two size");
        let scale = 1.0 / (n as f64).sqrt();
        Matrix::from_fn(n, n, |i, j| {
            let bits = (i & j).count_ones();
            if bits % 2 == 0 {
                scale
            } else {
                -scale
            }
        })
    }

    /// Set a frequency-dependent decay: RT60 at low frequencies and a
    /// (shorter) RT60 above ~2 kHz via one-pole shelving in each line.
    pub fn set_rt60(&mut self, t60_low: f64, t60_high: f64, fs: f64) {
        for (i, &d) in self.delay_samples.iter().enumerate() {
            let delay_t = d as f64 / fs;
            let g_low = 10.0_f64.powf(-3.0 * delay_t / t60_low.max(1e-3));
            let g_high = 10.0_f64.powf(-3.0 * delay_t / t60_high.max(1e-3));
            self.gains[i] = g_low;
            // High-shelf cut implementing the extra high-frequency loss.
            let cut_db = 20.0 * (g_high / g_low).log10();
            self.filters[i] = Biquad::highshelf(2000.0, fs, 1.0, cut_db);
        }
    }

    /// One (wet) sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let n = self.delays.len();
        let outs: Vec<f64> = (0..n)
            .map(|i| self.delays[i].read(self.delay_samples[i] - 1))
            .collect();
        // Mix and write back with per-line gain and damping filter.
        for i in 0..n {
            let mixed: f64 = (0..n).map(|j| self.matrix.get(i, j) * outs[j]).sum();
            let filtered = self.filters[i].process(mixed * self.gains[i]);
            self.delays[i].write(x + filtered);
        }
        outs.iter().sum::<f64>() / n as f64
    }

    /// One sample to a decorrelated stereo pair (alternating-sign taps).
    pub fn process_stereo(&mut self, x: f64) -> (f64, f64) {
        let n = self.delays.len();
        let outs: Vec<f64> = (0..n)
            .map(|i| self.delays[i].read(self.delay_samples[i] - 1))
            .collect();
        for i in 0..n {
            let mixed: f64 = (0..n).map(|j| self.matrix.get(i, j) * outs[j]).sum();
            let filtered = self.filters[i].process(mixed * self.gains[i]);
            self.delays[i].write(x + filtered);
        }
        let l: f64 = outs.iter().sum::<f64>() / n as f64;
        let r: f64 = outs
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
            .sum::<f64>()
            / n as f64;
        (l, r)
    }
}

/// Convolution reverb via the partitioned FFT convolver (matches direct
/// convolution; output length x + ir − 1).
#[must_use]
pub fn convolution_reverb(x: &[f64], ir: &[f64]) -> Vec<f64> {
    if x.is_empty() || ir.is_empty() {
        return Vec::new();
    }
    let block = next_power_of_two(ir.len().max(64));
    let mut conv = PartitionedConvolver::new(ir, block);
    let out_len = x.len() + ir.len() - 1;
    let mut out = Vec::with_capacity(out_len);
    let mut fed = 0usize;
    while out.len() < out_len {
        let mut chunk = vec![0.0; block];
        for slot in chunk.iter_mut() {
            if fed < x.len() {
                *slot = x[fed];
                fed += 1;
            }
        }
        out.extend_from_slice(&conv.process_block(&chunk));
    }
    out.truncate(out_len);
    out
}

/// Uniform partitioned (overlap-add, frequency-domain) convolver for
/// streaming long impulse responses.
pub struct PartitionedConvolver {
    block: usize,
    partitions: Vec<Vec<Complex>>,
    history: Vec<Vec<Complex>>,
    head: usize,
    overlap: Vec<f64>,
}

impl PartitionedConvolver {
    /// Partition an impulse response into `block_size` chunks.
    ///
    /// # Panics
    /// Panics if the IR is empty or block_size is zero.
    #[must_use]
    pub fn new(ir: &[f64], block_size: usize) -> Self {
        assert!(!ir.is_empty() && block_size > 0, "need IR and block size");
        let block = next_power_of_two(block_size);
        let nfft = 2 * block;
        let partitions: Vec<Vec<Complex>> = ir
            .chunks(block)
            .map(|chunk| {
                let mut buf = vec![Complex::new(0.0, 0.0); nfft];
                for (i, &v) in chunk.iter().enumerate() {
                    buf[i] = Complex::new(v, 0.0);
                }
                fft(&buf)
            })
            .collect();
        let n_parts = partitions.len();
        Self {
            block,
            partitions,
            history: vec![vec![Complex::new(0.0, 0.0); nfft]; n_parts],
            head: 0,
            overlap: vec![0.0; block],
        }
    }

    /// Convolve one input block (length ≤ block size); returns exactly
    /// one block of output.
    pub fn process_block(&mut self, x: &[f64]) -> Vec<f64> {
        let nfft = 2 * self.block;
        let mut buf = vec![Complex::new(0.0, 0.0); nfft];
        for (i, &v) in x.iter().take(self.block).enumerate() {
            buf[i] = Complex::new(v, 0.0);
        }
        let spec = fft(&buf);
        let n_parts = self.partitions.len();
        self.head = (self.head + n_parts - 1) % n_parts;
        self.history[self.head] = spec;
        // Accumulate partition products.
        let mut acc = vec![Complex::new(0.0, 0.0); nfft];
        for (p, part) in self.partitions.iter().enumerate() {
            let h = &self.history[(self.head + p) % n_parts];
            for k in 0..nfft {
                acc[k] = acc[k] + part[k] * h[k];
            }
        }
        let time = ifft(&acc);
        let mut out = vec![0.0; self.block];
        for i in 0..self.block {
            out[i] = time[i].re + self.overlap[i];
            self.overlap[i] = time[self.block + i].re;
        }
        out
    }
}

/// Synthetic exponential-decay impulse response with optional discrete
/// early reflections (time s, gain).
#[must_use]
pub fn synthesize_ir_exponential(
    rt60: f64,
    fs: f64,
    early_reflections: &[(f64, f64)],
    rng: &mut Rng,
) -> Vec<f64> {
    let n = (rt60 * 1.2 * fs) as usize;
    let tau = rt60 / 6.9078; // ln(10^3)
    let mut ir: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * rng.next_f64() - 1.0) * (-t / tau).exp()
        })
        .collect();
    for &(t, g) in early_reflections {
        let idx = (t * fs) as usize;
        if idx < n {
            ir[idx] += g;
        }
    }
    if !ir.is_empty() {
        ir[0] = 1.0;
    }
    ir
}

// --- Modulation effects --------------------------------------------------

/// Chorus: LFO-modulated fractional delay mixed with the dry path.
pub struct Chorus {
    delay: DelayLine,
    lfo: Lfo,
    pub base_ms: f64,
    pub depth_ms: f64,
    pub mix: f64,
    fs: f64,
}

impl Chorus {
    /// Typical chorus (20 ms base, 5 ms depth, 0.8 Hz).
    #[must_use]
    pub fn new(fs: f64) -> Self {
        Self {
            delay: DelayLine::new((0.06 * fs) as usize),
            lfo: Lfo { osc: Oscillator::new(Waveform::Sine, 0.8, fs), depth: 1.0, offset: 0.0 },
            base_ms: 20.0,
            depth_ms: 5.0,
            mix: 0.5,
            fs,
        }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        self.delay.write(x);
        let d = (self.base_ms + self.depth_ms * 0.5 * (1.0 + self.lfo.next())) * 1e-3 * self.fs;
        (1.0 - self.mix) * x + self.mix * self.delay.read_interp(d)
    }
}

/// Flanger: short modulated delay with feedback.
pub struct Flanger {
    delay: DelayLine,
    lfo: Lfo,
    pub depth_ms: f64,
    pub feedback: f64,
    pub mix: f64,
    fs: f64,
    fb_sample: f64,
}

impl Flanger {
    /// Typical flanger (0–5 ms sweep at 0.25 Hz).
    #[must_use]
    pub fn new(fs: f64) -> Self {
        Self {
            delay: DelayLine::new((0.02 * fs) as usize),
            lfo: Lfo { osc: Oscillator::new(Waveform::Sine, 0.25, fs), depth: 1.0, offset: 0.0 },
            depth_ms: 5.0,
            feedback: 0.5,
            mix: 0.5,
            fs,
            fb_sample: 0.0,
        }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        self.delay.write(x + self.feedback * self.fb_sample);
        let d = (0.5 + self.depth_ms * 0.5 * (1.0 + self.lfo.next())) * 1e-3 * self.fs;
        let wet = self.delay.read_interp(d.max(1.0));
        self.fb_sample = wet;
        (1.0 - self.mix) * x + self.mix * wet
    }
}

/// Phaser: cascaded LFO-swept all-pass biquads.
pub struct Phaser {
    allpasses: Vec<Biquad>,
    lfo: Lfo,
    pub mix: f64,
    fs: f64,
}

impl Phaser {
    /// 4-stage phaser sweeping 300–1500 Hz at 0.5 Hz.
    #[must_use]
    pub fn new(fs: f64) -> Self {
        Self {
            allpasses: vec![Biquad::allpass(800.0, fs, 0.7); 4],
            lfo: Lfo { osc: Oscillator::new(Waveform::Sine, 0.5, fs), depth: 1.0, offset: 0.0 },
            mix: 0.5,
            fs,
        }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let sweep = 900.0 + 600.0 * self.lfo.next();
        let mut v = x;
        for ap in self.allpasses.iter_mut() {
            let fresh = Biquad::allpass(sweep, self.fs, 0.7);
            // Keep the state, swap the coefficients.
            let mut updated = *ap;
            updated.b0 = fresh.b0;
            updated.b1 = fresh.b1;
            updated.b2 = fresh.b2;
            updated.a1 = fresh.a1;
            updated.a2 = fresh.a2;
            *ap = updated;
            v = ap.process(v);
        }
        (1.0 - self.mix) * x + self.mix * v
    }
}

/// Tremolo (amplitude modulation by an LFO).
pub struct Tremolo {
    lfo: Lfo,
    pub depth: f64,
}

impl Tremolo {
    /// Tremolo at the given rate/depth.
    #[must_use]
    pub fn new(rate_hz: f64, depth: f64, fs: f64) -> Self {
        Self {
            lfo: Lfo { osc: Oscillator::new(Waveform::Sine, rate_hz, fs), depth: 1.0, offset: 0.0 },
            depth,
        }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        x * (1.0 - self.depth * 0.5 * (1.0 + self.lfo.next()))
    }
}

/// Vibrato (pitch modulation via modulated delay).
pub struct Vibrato {
    delay: DelayLine,
    lfo: Lfo,
    pub depth_ms: f64,
    fs: f64,
}

impl Vibrato {
    /// Vibrato at the given rate and delay depth.
    #[must_use]
    pub fn new(rate_hz: f64, depth_ms: f64, fs: f64) -> Self {
        Self {
            delay: DelayLine::new((0.05 * fs) as usize),
            lfo: Lfo { osc: Oscillator::new(Waveform::Sine, rate_hz, fs), depth: 1.0, offset: 0.0 },
            depth_ms,
            fs,
        }
    }

    /// One sample (wet only).
    pub fn process(&mut self, x: f64) -> f64 {
        self.delay.write(x);
        let d = (self.depth_ms * 0.5 * (1.0 + self.lfo.next()) + 1.0) * 1e-3 * self.fs;
        self.delay.read_interp(d)
    }
}

// --- Dynamics ------------------------------------------------------------

/// Feed-forward compressor with soft knee and log-domain smoothing.
pub struct Compressor {
    pub threshold_db: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub knee_db: f64,
    pub makeup_db: f64,
    env: f64,
    gr_db: f64,
    fs: f64,
}

impl Compressor {
    /// New compressor.
    #[must_use]
    pub fn new(threshold_db: f64, ratio: f64, attack_ms: f64, release_ms: f64, fs: f64) -> Self {
        Self {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            knee_db: 0.0,
            makeup_db: 0.0,
            env: 0.0,
            gr_db: 0.0,
            fs,
        }
    }

    fn static_gain_db(&self, level_db: f64) -> f64 {
        let over = level_db - self.threshold_db;
        let knee = self.knee_db.max(0.0);
        let compressed_over = if knee > 0.0 && over > -knee / 2.0 && over < knee / 2.0 {
            // Quadratic soft knee.
            let x = over + knee / 2.0;
            over + (1.0 / self.ratio - 1.0) * x * x / (2.0 * knee)
        } else if over > 0.0 {
            over / self.ratio
        } else {
            over
        };
        compressed_over - over
    }

    /// One sample with the input as its own detector.
    pub fn process(&mut self, x: f64) -> f64 {
        self.sidechain(x, x)
    }

    /// One sample with an external detector (key) signal.
    pub fn sidechain(&mut self, x: f64, key: f64) -> f64 {
        let att = (-1.0 / (self.attack_ms.max(1e-3) * 1e-3 * self.fs)).exp();
        let rel = (-1.0 / (self.release_ms.max(1e-3) * 1e-3 * self.fs)).exp();
        let a = key.abs();
        let coef = if a > self.env { att } else { rel };
        self.env = a + coef * (self.env - a);
        let level_db = 20.0 * self.env.max(1e-9).log10();
        self.gr_db = self.static_gain_db(level_db);
        x * 10.0_f64.powf((self.gr_db + self.makeup_db) / 20.0)
    }

    /// Gain reduction applied to the most recent sample (≤ 0 dB).
    #[must_use]
    pub fn gain_reduction_db(&self) -> f64 {
        self.gr_db
    }
}

/// Brickwall limiter with lookahead.
pub struct Limiter {
    lookahead: DelayLine,
    lookahead_samples: usize,
    pub ceiling: f64,
    env: f64,
    release_coef: f64,
}

impl Limiter {
    /// Limiter with the given ceiling (linear) and lookahead/release.
    #[must_use]
    pub fn new(ceiling: f64, lookahead_ms: f64, release_ms: f64, fs: f64) -> Self {
        let n = ((lookahead_ms * 1e-3 * fs) as usize).max(1);
        Self {
            lookahead: DelayLine::new(n + 2),
            lookahead_samples: n,
            ceiling,
            env: 0.0,
            release_coef: (-1.0 / (release_ms.max(0.1) * 1e-3 * fs)).exp(),
        }
    }

    /// One sample; output never exceeds the ceiling.
    pub fn process(&mut self, x: f64) -> f64 {
        self.lookahead.write(x);
        // Envelope sees the incoming (future) sample.
        let a = x.abs();
        self.env = if a > self.env { a } else { a + self.release_coef * (self.env - a) };
        let delayed = self.lookahead.read(self.lookahead_samples);
        let gain = if self.env > self.ceiling { self.ceiling / self.env } else { 1.0 };
        (delayed * gain).clamp(-self.ceiling, self.ceiling)
    }
}

/// Downward noise gate.
pub struct NoiseGate {
    pub threshold: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    env: f64,
    gain: f64,
    fs: f64,
}

impl NoiseGate {
    /// Gate at a linear threshold.
    #[must_use]
    pub fn new(threshold: f64, attack_ms: f64, release_ms: f64, fs: f64) -> Self {
        Self { threshold, attack_ms, release_ms, env: 0.0, gain: 0.0, fs }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let att = (-1.0 / (self.attack_ms.max(1e-3) * 1e-3 * self.fs)).exp();
        let rel = (-1.0 / (self.release_ms.max(1e-3) * 1e-3 * self.fs)).exp();
        let a = x.abs();
        self.env = a + (if a > self.env { att } else { rel }) * (self.env - a);
        let target = if self.env > self.threshold { 1.0 } else { 0.0 };
        let coef = if target > self.gain { att } else { rel };
        self.gain = target + coef * (self.gain - target);
        x * self.gain
    }
}

/// Downward expander (gentler than a gate).
pub struct Expander {
    pub threshold_db: f64,
    pub ratio: f64,
    env: f64,
    fs: f64,
}

impl Expander {
    /// Expander below the threshold with the given ratio.
    #[must_use]
    pub fn new(threshold_db: f64, ratio: f64, fs: f64) -> Self {
        Self { threshold_db, ratio, env: 0.0, fs }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let coef = (-1.0 / (5.0e-3 * self.fs)).exp();
        let a = x.abs();
        self.env = a + coef * (self.env - a);
        let level_db = 20.0 * self.env.max(1e-9).log10();
        let under = self.threshold_db - level_db;
        let gain_db = if under > 0.0 { -under * (self.ratio - 1.0) } else { 0.0 };
        x * 10.0_f64.powf(gain_db.max(-60.0) / 20.0)
    }
}

/// De-esser: sibilance-band compressor (band-passed key).
pub struct DeEsser {
    key_filter: Biquad,
    comp: Compressor,
}

impl DeEsser {
    /// De-esser keyed around `freq` (typically 5–8 kHz).
    #[must_use]
    pub fn new(freq: f64, threshold_db: f64, fs: f64) -> Self {
        Self {
            key_filter: Biquad::bandpass(freq, fs, 2.0),
            comp: Compressor::new(threshold_db, 4.0, 0.5, 30.0, fs),
        }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let key = self.key_filter.process(x);
        self.comp.sidechain(x, key)
    }
}

// --- Distortion ----------------------------------------------------------

/// tanh soft clipper.
#[must_use]
pub fn distortion_soft_clip(x: f64, drive: f64) -> f64 {
    (x * drive).tanh()
}

/// Hard clipper at ±threshold.
#[must_use]
pub fn distortion_hard_clip(x: f64, threshold: f64) -> f64 {
    x.clamp(-threshold, threshold)
}

/// Asymmetric "tube" shaper (bias shifts the operating point).
#[must_use]
pub fn distortion_tube(x: f64, drive: f64, bias: f64) -> f64 {
    ((x + bias) * drive).tanh() - (bias * drive).tanh()
}

/// Foldback distortion.
#[must_use]
pub fn distortion_foldback(x: f64, threshold: f64) -> f64 {
    let t = threshold.abs().max(1e-9);
    let period = 4.0 * t;
    let wrapped = (x - t).rem_euclid(period);
    if wrapped < 2.0 * t {
        t - wrapped
    } else {
        wrapped - 3.0 * t
    }
}

/// Bit crusher: quantize to `bits` and hold every `sample_hold_factor`
/// samples (stateful via the counter/held pair).
#[must_use]
pub fn bitcrush(
    x: f64,
    bits: u32,
    sample_hold_factor: usize,
    counter: &mut usize,
    held: &mut f64,
) -> f64 {
    if *counter == 0 {
        let levels = 2.0_f64.powi(bits.max(1) as i32 - 1);
        *held = (x * levels).round() / levels;
    }
    *counter = (*counter + 1) % sample_hold_factor.max(1);
    *held
}

/// Run a memoryless nonlinearity oversampled by `factor` (anti-aliased:
/// upsample, apply, decimate).
#[must_use]
pub fn oversample_process(x: &[f64], factor: usize, f: &dyn Fn(f64) -> f64) -> Vec<f64> {
    if factor <= 1 {
        return x.iter().map(|&v| f(v)).collect();
    }
    let up = crate::dsp::resample::upsample(x, factor);
    let shaped: Vec<f64> = up.iter().map(|&v| f(v)).collect();
    crate::dsp::resample::decimate(&shaped, factor)
}

// --- EQ and imaging ------------------------------------------------------

/// A bank of peaking/shelf biquads.
pub struct Eq {
    pub bands: Vec<Biquad>,
    fs: f64,
    centers: Vec<f64>,
}

impl Eq {
    /// Standard 10-band graphic EQ (31.25 Hz–16 kHz octaves), flat.
    #[must_use]
    pub fn graphic_10_band(fs: f64) -> Self {
        let centers: Vec<f64> = (0..10).map(|i| 31.25 * 2.0_f64.powi(i)).collect();
        Self {
            bands: centers.iter().map(|&f| Biquad::peaking(f, fs, 1.41, 0.0)).collect(),
            fs,
            centers,
        }
    }

    /// Parametric EQ from (freq, Q, gain dB) bands.
    #[must_use]
    pub fn parametric(bands: &[(f64, f64, f64)], fs: f64) -> Self {
        Self {
            bands: bands.iter().map(|&(f, q, g)| Biquad::peaking(f, fs, q, g)).collect(),
            fs,
            centers: bands.iter().map(|&(f, _, _)| f).collect(),
        }
    }

    /// One sample through every band.
    pub fn process(&mut self, x: f64) -> f64 {
        self.bands.iter_mut().fold(x, |v, b| b.process(v))
    }

    /// Set band i's gain (graphic-EQ style, Q 1.41).
    pub fn set_gain(&mut self, i: usize, db: f64) {
        if i < self.bands.len() {
            self.bands[i] = Biquad::peaking(self.centers[i], self.fs, 1.41, db);
        }
    }
}

/// Harmonic exciter: high-passed signal through a soft shaper, mixed in.
pub struct Exciter {
    hp: Biquad,
    pub drive: f64,
    pub mix: f64,
}

impl Exciter {
    /// Exciter brightening content above `freq`.
    #[must_use]
    pub fn new(freq: f64, drive: f64, mix: f64, fs: f64) -> Self {
        Self { hp: Biquad::highpass(freq, fs, std::f64::consts::FRAC_1_SQRT_2), drive, mix }
    }

    /// One sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let bright = distortion_soft_clip(self.hp.process(x), self.drive);
        x + self.mix * bright
    }
}

/// Mid/side stereo widener.
pub struct StereoWidener {
    pub width: f64,
}

impl StereoWidener {
    /// One stereo frame: width 1 = unchanged, > 1 wider, 0 = mono.
    #[must_use]
    pub fn process(&self, l: f64, r: f64) -> (f64, f64) {
        let mid = 0.5 * (l + r);
        let side = 0.5 * (l - r) * self.width;
        (mid + side, mid - side)
    }
}

/// Haas effect: (dry, delayed) pair for pseudo-stereo width.
#[must_use]
pub fn haas_delay(x: &[f64], ms: f64, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let d = (ms * 1e-3 * fs) as usize;
    let right: Vec<f64> = (0..x.len())
        .map(|i| if i >= d { x[i - d] } else { 0.0 })
        .collect();
    (x.to_vec(), right)
}

/// Delay-line (Doppler) pitch shifter with two crossfaded taps.
#[must_use]
pub fn pitch_shift_simple(x: &[f64], semitones: f64, fs: f64) -> Vec<f64> {
    let ratio = 2.0_f64.powf(semitones / 12.0);
    let window = (0.03 * fs) as usize; // 30 ms grains
    let mut delay = DelayLine::new(2 * window + 4);
    let mut phase = 0.0_f64; // sweeping delay in samples
    let rate = 1.0 - ratio;
    (0..x.len())
        .map(|i| {
            delay.write(x[i]);
            phase = (phase + rate).rem_euclid(window as f64);
            let d1 = phase;
            let d2 = (phase + window as f64 / 2.0).rem_euclid(window as f64);
            // Triangle crossfade based on tap position.
            let w1 = 1.0 - (d1 / window as f64 * 2.0 - 1.0).abs();
            let w2 = 1.0 - (d2 / window as f64 * 2.0 - 1.0).abs();
            let sum = (w1 + w2).max(1e-9);
            (delay.read_interp(d1 + 1.0) * w1 + delay.read_interp(d2 + 1.0) * w2) / sum
        })
        .collect()
}

// --- Loudness and utility ------------------------------------------------

/// Apply a gain in dB in place.
pub fn gain_db(x: &mut [f64], db: f64) {
    let g = 10.0_f64.powf(db / 20.0);
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// Normalize the peak to `target_db` (dBFS) in place.
pub fn normalize_peak(x: &mut [f64], target_db: f64) {
    let peak = x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if peak > 0.0 {
        let g = 10.0_f64.powf(target_db / 20.0) / peak;
        for v in x.iter_mut() {
            *v *= g;
        }
    }
}

/// Normalize the RMS to `target_db` in place.
pub fn normalize_rms(x: &mut [f64], target_db: f64) {
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt();
    if rms > 0.0 {
        let g = 10.0_f64.powf(target_db / 20.0) / rms;
        for v in x.iter_mut() {
            *v *= g;
        }
    }
}

/// BS.1770 K-weighting prefilter pair for an arbitrary sample rate
/// (exact published coefficients at 48 kHz; RBJ-designed equivalents
/// otherwise).
fn k_weighting(fs: f64) -> (Biquad, Biquad) {
    if (fs - 48000.0).abs() < 1e-6 {
        let shelf = Biquad::from_coeffs(
            1.535_124_859_586_97,
            -2.691_696_189_406_38,
            1.198_392_810_852_85,
            -1.690_659_293_182_41,
            0.732_480_774_215_85,
        );
        let rlb = Biquad::from_coeffs(1.0, -2.0, 1.0, -1.990_047_454_833_98, 0.990_072_250_366_21);
        (shelf, rlb)
    } else {
        (
            Biquad::highshelf(1681.97, fs, 1.0, 3.999_84),
            Biquad::highpass(38.135, fs, 0.5),
        )
    }
}

/// Integrated loudness (LUFS) per ITU-R BS.1770-4: K-weighting, 400 ms
/// blocks with 75% overlap, absolute −70 LUFS and relative −10 LU
/// gating.
#[must_use]
pub fn measure_lufs(x: &[f64], fs: f64) -> f64 {
    let (mut shelf, mut rlb) = k_weighting(fs);
    let weighted: Vec<f64> = x.iter().map(|&v| rlb.process(shelf.process(v))).collect();
    let block = (0.4 * fs) as usize;
    let hop = block / 4;
    if weighted.len() < block {
        let ms = weighted.iter().map(|v| v * v).sum::<f64>() / weighted.len().max(1) as f64;
        return -0.691 + 10.0 * ms.max(1e-15).log10();
    }
    let mut blocks = Vec::new();
    let mut start = 0;
    while start + block <= weighted.len() {
        let ms = weighted[start..start + block].iter().map(|v| v * v).sum::<f64>() / block as f64;
        blocks.push(ms);
        start += hop;
    }
    let loud = |ms: f64| -0.691 + 10.0 * ms.max(1e-15).log10();
    // Absolute gate.
    let above: Vec<f64> = blocks.iter().copied().filter(|&ms| loud(ms) > -70.0).collect();
    if above.is_empty() {
        return -70.0;
    }
    let mean_ms = above.iter().sum::<f64>() / above.len() as f64;
    let rel_gate = loud(mean_ms) - 10.0;
    let gated: Vec<f64> = above.into_iter().filter(|&ms| loud(ms) > rel_gate).collect();
    if gated.is_empty() {
        return rel_gate;
    }
    loud(gated.iter().sum::<f64>() / gated.len() as f64)
}

/// Normalize integrated loudness to `target_lufs` in place.
pub fn normalize_lufs(x: &mut [f64], target_lufs: f64, fs: f64) {
    let current = measure_lufs(x, fs);
    gain_db(x, target_lufs - current);
}

/// Inter-sample true peak (4× oversampled), linear.
#[must_use]
pub fn true_peak(x: &[f64], fs: f64) -> f64 {
    let _ = fs;
    let up = crate::dsp::resample::upsample(x, 4);
    up.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

/// TPDF dither to `bits` (quantized output in −1..1).
#[must_use]
pub fn dither_tpdf(x: &[f64], bits: u32, rng: &mut Rng) -> Vec<f64> {
    let q = 1.0 / 2.0_f64.powi(bits.max(2) as i32 - 1);
    x.iter()
        .map(|&v| {
            let noise = (rng.next_f64() - rng.next_f64()) * q;
            ((v + noise) / q).round() * q
        })
        .collect()
}

/// First-order noise-shaped dither (error feedback pushes quantization
/// noise upward in frequency).
#[must_use]
pub fn noise_shaping_dither(x: &[f64], bits: u32, rng: &mut Rng) -> Vec<f64> {
    let q = 1.0 / 2.0_f64.powi(bits.max(2) as i32 - 1);
    let mut err = 0.0;
    x.iter()
        .map(|&v| {
            let noise = (rng.next_f64() - rng.next_f64()) * q;
            let target = v - err + noise;
            let out = (target / q).round() * q;
            err = out - (v - err);
            out
        })
        .collect()
}

/// Remove the mean in place.
pub fn dc_offset_remove(x: &mut [f64]) {
    let mean = x.iter().sum::<f64>() / x.len().max(1) as f64;
    for v in x.iter_mut() {
        *v -= mean;
    }
}

/// Replace samples whose second difference exceeds `threshold` with a
/// linear interpolation of their neighbors (simple click repair).
#[must_use]
pub fn declick(x: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = x.to_vec();
    for i in 1..x.len().saturating_sub(1) {
        let second_diff = x[i + 1] - 2.0 * x[i] + x[i - 1];
        if second_diff.abs() > threshold {
            out[i] = 0.5 * (x[i - 1] + x[i + 1]);
        }
    }
    out
}

/// Spectral gate denoiser: attenuate STFT bins that fall below the
/// noise profile (per-bin magnitude) plus `threshold_db`.
#[must_use]
pub fn spectral_gate(
    x: &[f64],
    noise_profile: &[f64],
    threshold_db: f64,
    n_fft: usize,
    hop: usize,
) -> Vec<f64> {
    use crate::dsp::windows::{window, WindowKind};
    use crate::transforms::stft::Stft;
    let stft = Stft::new(window(WindowKind::Hann, n_fft, true), hop, n_fft);
    let mut frames = stft.forward(x);
    let gate = 10.0_f64.powf(threshold_db / 20.0);
    for frame in frames.iter_mut() {
        for (k, bin) in frame.iter_mut().enumerate() {
            let floor = noise_profile.get(k).copied().unwrap_or(0.0) * gate;
            if bin.norm() < floor {
                *bin = Complex::new(0.0, 0.0);
            }
        }
    }
    let mut out = stft.inverse(&frames);
    out.resize(x.len(), 0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    const TWO_PI: f64 = 2.0 * PI;

    #[test]
    fn test_delay_line_and_comb() {
        let mut dl = DelayLine::new(16);
        for i in 0..10 {
            dl.write(i as f64);
        }
        assert_eq!(dl.read(0), 9.0);
        assert_eq!(dl.read(3), 6.0);
        assert!((dl.read_interp(2.5) - 6.5).abs() < 1e-12);
        assert_eq!(dl.tap(1), 8.0);
        // Comb echoes at its period.
        let mut comb = CombFilter::new(10, 0.5, 0.0);
        let mut out = Vec::new();
        for i in 0..40 {
            out.push(comb.process(if i == 0 { 1.0 } else { 0.0 }));
        }
        assert!((out[10] - 1.0).abs() < 1e-12);
        assert!((out[20] - 0.5).abs() < 1e-12);
        assert!((out[30] - 0.25).abs() < 1e-12);
        // Allpass has unit-magnitude response: energy of the impulse
        // response equals 1.
        let mut ap = AllpassFilter::new(7, 0.6);
        let mut e = 0.0;
        for i in 0..5000 {
            let y = ap.process(if i == 0 { 1.0 } else { 0.0 });
            e += y * y;
        }
        assert!((e - 1.0).abs() < 1e-9, "allpass energy {e}");
    }

    #[test]
    fn test_schroeder_rt60() {
        let fs = 16000.0;
        let mut rv = SchroederReverb::new(fs);
        let target = 0.8;
        rv.set_rt60(target);
        rv.set_damping(0.0);
        let n = (2.0 * fs) as usize;
        let ir: Vec<f64> = (0..n)
            .map(|i| rv.process(if i == 0 { 1.0 } else { 0.0 }))
            .collect();
        // Schroeder backward integration → RT60 from the −5..−25 dB slope.
        let mut energy: Vec<f64> = ir.iter().rev().map(|v| v * v).collect();
        for i in 1..energy.len() {
            energy[i] += energy[i - 1];
        }
        energy.reverse();
        let db: Vec<f64> = energy.iter().map(|&e| 10.0 * (e / energy[0]).log10()).collect();
        let t_at = |level: f64| -> f64 {
            db.iter().position(|&d| d < level).unwrap_or(db.len() - 1) as f64 / fs
        };
        let rt60 = (t_at(-25.0) - t_at(-5.0)) * 3.0;
        assert!(
            (rt60 - target).abs() / target < 0.15,
            "measured RT60 {rt60} vs {target}"
        );
    }

    #[test]
    fn test_freeverb_topology_predelay_and_decay() {
        let fs = 44100.0; // the classic Freeverb tunings are exact here
        let mut fv = Freeverb::new(fs);
        let n = (2.0 * fs) as usize;
        let mut l = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = fv.process(if i == 0 { 1.0 } else { 0.0 });
            l.push(a);
            r.push(b);
        }
        // Pre-delay: nothing escapes before the shortest comb loop
        // (1116 samples on the left, +23 stereo spread on the right).
        assert!(l[..1116].iter().all(|&v| v == 0.0), "left leaks before its comb");
        assert!(r[..1139].iter().all(|&v| v == 0.0), "right leaks before its comb");
        assert!(l[1116..1139].iter().any(|&v| v.abs() > 1e-6), "left never fires");
        // The stereo spread genuinely decorrelates the two channels.
        assert!(l != r, "stereo channels are identical");
        // Stable over the whole tail.
        assert!(l.iter().chain(&r).all(|v| v.is_finite() && v.abs() < 4.0), "unstable");
        // The wet output is a real reverb tail, not a single echo.
        let e: f64 = l.iter().map(|v| v * v).sum();
        assert!(e > 1e-4, "wet energy {e}");

        // Cross-check the documented topology against the same 8 combs and
        // 4 all-passes built from the independently tested primitives.
        let mut combs: Vec<CombFilter> = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
            .iter()
            .map(|&d| CombFilter::new(d, 0.84, 0.2))
            .collect();
        let mut aps: Vec<AllpassFilter> = [556, 441, 341, 225]
            .iter()
            .map(|&d| AllpassFilter::new(d, 0.5))
            .collect();
        for (i, &want) in l.iter().enumerate().take(20000) {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let mut v: f64 = combs.iter_mut().map(|c| c.process(x)).sum::<f64>() / 8.0;
            for ap in aps.iter_mut() {
                v = ap.process(v);
            }
            assert!((v - want).abs() < 1e-12, "topology mismatch at {i}: {v} vs {want}");
        }

        // Exponential decay. Every comb has DC loop gain 0.84 (the damping
        // one-pole is unity at DC), so over half a second a comb of length
        // D decays by 0.84^(0.5·fs/D). The tail must fall between the
        // fastest (D = 1116) and slowest (D = 1617) of the eight.
        let rms = |s: &[f64]| (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64).sqrt();
        let a = rms(&l[(0.5 * fs) as usize..(0.6 * fs) as usize]);
        let b = rms(&l[(1.0 * fs) as usize..(1.1 * fs) as usize]);
        let slowest = 0.84_f64.powf(0.5 * fs / 1617.0);
        let fastest = 0.84_f64.powf(0.5 * fs / 1116.0);
        let ratio = b / a;
        assert!(
            ratio > 0.5 * fastest && ratio < 1.5 * slowest,
            "half-second decay {ratio} outside [{fastest}, {slowest}]"
        );
        // Later windows are quieter than earlier ones, monotonically.
        let mut prev = f64::MAX;
        for w in 0..8 {
            let lo = ((0.2 + 0.2 * w as f64) * fs) as usize;
            let cur = rms(&l[lo..lo + (0.1 * fs) as usize]);
            assert!(cur < prev, "energy grew in window {w}");
            prev = cur;
        }
    }

    #[test]
    fn test_eq_parametric_hits_its_band_gains() {
        let fs = 48000.0;
        // A 0 dB band is exactly transparent (numerator and denominator of
        // the RBJ peaking design coincide when A = 1).
        let mut flat = Eq::parametric(&[(1000.0, 1.0, 0.0)], fs);
        for i in 0..500 {
            let x = ((i * 7919) % 200) as f64 / 100.0 - 1.0;
            assert!((flat.process(x) - x).abs() < 1e-12, "0 dB band is not transparent");
        }

        let bands = [(120.0, 1.0, 8.0), (3000.0, 4.0, -10.0)];
        let mut eq = Eq::parametric(&bands, fs);
        assert_eq!(eq.bands.len(), 2);
        // parametric() must build exactly one RBJ peaking section per
        // (freq, Q, gain) triple.
        let design: Vec<Biquad> = bands
            .iter()
            .map(|&(f, q, g)| Biquad::peaking(f, fs, q, g))
            .collect();
        assert_eq!(eq.bands, design, "parametric bands are not RBJ peaking sections");
        // Analytic cascade response, straight from the biquad coefficients.
        let analytic_db = |f: f64| -> f64 {
            let h = design
                .iter()
                .fold(Complex::new(1.0, 0.0), |acc, b| acc * b.freq_response(f, fs));
            20.0 * h.norm().log10()
        };
        // Each band reaches its design gain at its own center; the two are
        // far enough apart (4.6 octaves) that the cross-talk stays small.
        assert!((analytic_db(120.0) - 8.0).abs() < 0.3, "120 Hz {}", analytic_db(120.0));
        assert!((analytic_db(3000.0) + 10.0).abs() < 0.3, "3 kHz {}", analytic_db(3000.0));
        // Far outside both bands the EQ is essentially flat.
        assert!(analytic_db(20.0).abs() < 0.5, "20 Hz {}", analytic_db(20.0));
        assert!(analytic_db(20000.0).abs() < 0.6, "20 kHz {}", analytic_db(20000.0));

        // Measured steady-state tone gain must match that analytic response.
        let mut tone_gain = |f: f64| -> f64 {
            let n = (0.5 * fs) as usize;
            let mut peak = 0.0_f64;
            for i in 0..n {
                let y = eq.process((TWO_PI * f * i as f64 / fs).sin());
                if i > n / 2 {
                    peak = peak.max(y.abs());
                }
            }
            20.0 * peak.log10()
        };
        for &f in &[120.0, 400.0, 1000.0, 3000.0, 9000.0] {
            let measured = tone_gain(f);
            let want = analytic_db(f);
            assert!(
                (measured - want).abs() < 0.05,
                "at {f} Hz: measured {measured} dB vs response {want} dB"
            );
        }
        // Boost really boosts and cut really cuts, relative to a bypass.
        assert!(tone_gain(120.0) > 7.0);
        assert!(tone_gain(3000.0) < -9.0);
    }

    #[test]
    fn test_fdn_lossless_and_decaying() {
        let fs = 16000.0;
        // Unit gains + orthogonal matrix: energy neither explodes nor
        // dies (checked over a short window).
        let mut fdn = Fdn::new(4, &[149, 211, 263, 293], fs);
        let mut energy_early = 0.0;
        let mut energy_late = 0.0;
        for i in 0..8000 {
            let y = fdn.process(if i == 0 { 1.0 } else { 0.0 });
            if i < 2000 {
                energy_early += y * y;
            }
            if i >= 6000 {
                energy_late += y * y;
            }
        }
        assert!(
            energy_late > 0.2 * energy_early,
            "lossless FDN decayed: {energy_late} vs {energy_early}"
        );
        // With RT60 set, it decays.
        let mut fdn2 = Fdn::new(4, &[149, 211, 263, 293], fs);
        fdn2.set_rt60(0.3, 0.15, fs);
        let mut e_early = 0.0;
        let mut e_late = 0.0;
        for i in 0..8000 {
            let y = fdn2.process(if i == 0 { 1.0 } else { 0.0 });
            if i < 2000 {
                e_early += y * y;
            }
            if i >= 6000 {
                e_late += y * y;
            }
        }
        assert!(e_late < 0.05 * e_early, "damped FDN didn't decay");
        // Hadamard matrix is orthogonal.
        let h = Fdn::hadamard_matrix(4);
        let hth = h.transpose().mul(&h).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((hth.get(i, j) - expect).abs() < 1e-12);
            }
        }
        let (l, r) = fdn2.process_stereo(0.5);
        assert!(l.is_finite() && r.is_finite());
    }

    #[test]
    fn test_convolution_reverb_matches_direct() {
        let x: Vec<f64> = (0..300).map(|i| ((i * 37) % 41) as f64 / 20.0 - 1.0).collect();
        let ir: Vec<f64> = (0..90).map(|i| (-(i as f64) / 25.0).exp() * ((i % 7) as f64 - 3.0)).collect();
        let fast = convolution_reverb(&x, &ir);
        let direct = crate::signal_processing::convolve(&x, &ir);
        assert_eq!(fast.len(), direct.len());
        for (a, b) in fast.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        // Streaming convolver with small blocks agrees too.
        let mut pc = PartitionedConvolver::new(&ir, 32);
        let mut streamed = Vec::new();
        for chunk in x.chunks(32) {
            let mut padded = chunk.to_vec();
            padded.resize(32, 0.0);
            streamed.extend_from_slice(&pc.process_block(&padded));
        }
        for _ in 0..4 {
            streamed.extend_from_slice(&pc.process_block(&vec![0.0; 32]));
        }
        for (i, d) in direct.iter().enumerate() {
            assert!((streamed[i] - d).abs() < 1e-9, "stream idx {i}");
        }
        let mut rng = Rng::new(4);
        let ir2 = synthesize_ir_exponential(0.3, 8000.0, &[(0.01, 0.5)], &mut rng);
        assert!(ir2[0] == 1.0 && ir2.len() > 2000);
    }

    #[test]
    fn test_compressor_ratio_math() {
        let fs = 48000.0;
        // Static curve: 2x over threshold (in dB terms +6 dB) with ratio
        // r leaves 6/r dB, i.e. reduction 6·(1 − 1/r) dB.
        for &ratio in &[2.0, 4.0, 10.0] {
            let mut comp = Compressor::new(-20.0, ratio, 0.05, 100.0, fs);
            // Feed a constant level 6 dB above threshold until settled.
            let level = 10.0_f64.powf(-14.0 / 20.0);
            let mut out = 0.0;
            for _ in 0..48000 {
                out = comp.process(level);
            }
            let gr = comp.gain_reduction_db();
            let expect = -6.0 * (1.0 - 1.0 / ratio);
            assert!((gr - expect).abs() < 0.1, "ratio {ratio}: GR {gr} vs {expect}");
            let out_db = 20.0 * (out / level).log10();
            assert!((out_db - expect).abs() < 0.1);
        }
    }

    #[test]
    fn test_limiter_gate_expander_deesser() {
        let fs = 48000.0;
        let mut lim = Limiter::new(0.5, 1.0, 50.0, fs);
        let mut peak = 0.0_f64;
        for i in 0..4800 {
            let x = (TWO_PI * 100.0 * i as f64 / fs).sin() * 0.9;
            peak = peak.max(lim.process(x).abs());
        }
        assert!(peak <= 0.5 + 1e-9, "limiter leak {peak}");
        let mut gate = NoiseGate::new(0.1, 1.0, 20.0, fs);
        let mut quiet_out = 0.0_f64;
        for _ in 0..4800 {
            quiet_out = quiet_out.max(gate.process(0.01).abs());
        }
        assert!(quiet_out < 0.005, "gate leak {quiet_out}");
        let mut exp = Expander::new(-20.0, 2.0, fs);
        let mut small = 0.0;
        for _ in 0..4800 {
            small = exp.process(0.01);
        }
        assert!(small.abs() < 0.01);
        let mut de = DeEsser::new(6000.0, -30.0, fs);
        assert!(de.process(0.1).is_finite());
    }

    #[test]
    fn test_distortions_and_bitcrush() {
        assert!((distortion_soft_clip(10.0, 1.0) - 1.0).abs() < 1e-4);
        assert_eq!(distortion_hard_clip(2.0, 0.7), 0.7);
        // Tube shaper is asymmetric with bias.
        let pos = distortion_tube(0.5, 2.0, 0.2);
        let neg = distortion_tube(-0.5, 2.0, 0.2);
        assert!((pos + neg).abs() > 1e-3, "should be asymmetric");
        assert!(distortion_tube(0.0, 2.0, 0.2).abs() < 1e-12);
        // Foldback stays within ±t and folds.
        for &v in &[0.2, 0.9, 1.7, -2.4] {
            let y = distortion_foldback(v, 0.5);
            assert!(y.abs() <= 0.5 + 1e-12, "fold {v} -> {y}");
        }
        assert!((distortion_foldback(0.2, 0.5) - 0.2).abs() < 1e-12);
        assert!((distortion_foldback(0.7, 0.5) - 0.3).abs() < 1e-12);
        let mut counter = 0usize;
        let mut held = 0.0;
        let a = bitcrush(0.33, 4, 2, &mut counter, &mut held);
        let b = bitcrush(0.9, 4, 2, &mut counter, &mut held);
        assert_eq!(a, b, "sample-hold should keep the value");
        // 4-bit levels of 1/8.
        assert!((a * 8.0 - (a * 8.0).round()).abs() < 1e-12);
    }

    #[test]
    fn test_oversampled_clip_reduces_aliasing() {
        let fs = 48000.0;
        let f0 = 5000.0;
        let n = 4096;
        let x: Vec<f64> = (0..n).map(|i| 0.9 * (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let clip = |v: f64| distortion_soft_clip(v, 3.0);
        let plain: Vec<f64> = x.iter().map(|&v| clip(v)).collect();
        let over = oversample_process(&x, 4, &clip);
        let spec = |v: &Vec<f64>| crate::transforms::fft::rfft(v);
        let sp = spec(&plain);
        let so = spec(&over);
        // Alias of the 3rd harmonic: 3f0 = 15 kHz stays in band, 5f0 =
        // 25 kHz folds to 23 kHz. Compare that alias line.
        let alias_bin = ((fs - 5.0 * f0) * n as f64 / fs).round() as usize;
        let fund_bin = (f0 * n as f64 / fs).round() as usize;
        let alias_p = sp[alias_bin].norm() / sp[fund_bin].norm();
        let alias_o = so[alias_bin].norm() / so[fund_bin].norm();
        assert!(alias_o < 0.15 * alias_p, "alias {alias_o} vs plain {alias_p}");
    }

    #[test]
    fn test_eq_and_imaging() {
        let fs = 48000.0;
        let mut eq = Eq::graphic_10_band(fs);
        eq.set_gain(5, 6.0); // 1 kHz band
        // Measure via a tone.
        let f0 = 1000.0;
        let mut out = Vec::new();
        for i in 0..9600 {
            out.push(eq.process((TWO_PI * f0 * i as f64 / fs).sin()));
        }
        let amp = out[4800..].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!((20.0 * amp.log10() - 6.0).abs() < 0.5, "1 kHz gain {}", 20.0 * amp.log10());
        let wide = StereoWidener { width: 2.0 };
        let (l, r) = wide.process(1.0, 0.5);
        assert!((l - 1.25).abs() < 1e-12 && (r - 0.25).abs() < 1e-12);
        let (dry, wet) = haas_delay(&[1.0, 0.0, 0.0], 0.5, 4000.0);
        assert_eq!(dry[0], 1.0);
        assert_eq!(wet[2], 1.0);
        let mut ex = Exciter::new(3000.0, 2.0, 0.5, fs);
        assert!(ex.process(0.5).is_finite());
    }

    #[test]
    fn test_modulation_effects_run() {
        let fs = 8000.0;
        let mut ch = Chorus::new(fs);
        let mut fl = Flanger::new(fs);
        let mut ph = Phaser::new(fs);
        let mut tr = Tremolo::new(4.0, 0.8, fs);
        let mut vb = Vibrato::new(5.0, 2.0, fs);
        let mut acc = 0.0;
        for i in 0..4000 {
            let x = (TWO_PI * 220.0 * i as f64 / fs).sin();
            acc += ch.process(x) + fl.process(x) + ph.process(x) + tr.process(x) + vb.process(x);
        }
        assert!(acc.is_finite());
        // Tremolo actually modulates the amplitude.
        let mut tr2 = Tremolo::new(2.0, 1.0, fs);
        let env: Vec<f64> = (0..8000)
            .map(|i| tr2.process((TWO_PI * 500.0 * i as f64 / fs).sin()).abs())
            .collect();
        // Sine LFO from phase 0: gain 0.5 - 0.5 sin(2*pi*2*t), so the null is
        // at t = T/4 (sample 1000) and unity gain at 3T/4 (sample 3000).
        let dip = env[950..1050].iter().cloned().fold(0.0_f64, f64::max);
        let peak = env[2900..3100].iter().cloned().fold(0.0_f64, f64::max);
        assert!(dip < 0.1 * peak, "tremolo dip {dip} vs peak {peak}");
        assert!(peak > 0.9, "tremolo peak {peak}");
    }

    #[test]
    fn test_lufs_of_reference_sine() {
        let fs = 48000.0;
        // Sine whose RMS sits 23 dB below a full-scale sine's RMS.
        let amp = 10.0_f64.powf(-23.0 / 20.0);
        let x: Vec<f64> = (0..(5.0 * fs) as usize)
            .map(|i| amp * (TWO_PI * 997.0 * i as f64 / fs).sin())
            .collect();
        let lufs = measure_lufs(&x, fs);
        // Full-scale 997 Hz sine reads −3.01 LUFS; −23 dB below that.
        assert!((lufs - (-26.01)).abs() < 0.1, "LUFS {lufs}");
        // normalize_lufs hits its target.
        let mut y = x.clone();
        normalize_lufs(&mut y, -23.0, fs);
        assert!((measure_lufs(&y, fs) + 23.0).abs() < 0.05);
    }

    #[test]
    fn test_true_peak_and_dither() {
        // A sine sampled near its crest has inter-sample peaks above the
        // sample peak.
        let fs = 48000.0;
        let f0 = 11987.0;
        let x: Vec<f64> = (0..4800).map(|i| 0.9 * (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let sample_peak = x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let tp = true_peak(&x, fs);
        assert!(tp >= sample_peak - 1e-9);
        assert!(tp > 0.89, "true peak {tp}");
        let mut rng = Rng::new(11);
        let sig: Vec<f64> = (0..8000).map(|i| 0.5 * (TWO_PI * 440.0 * i as f64 / fs).sin()).collect();
        let dithered = dither_tpdf(&sig, 8, &mut rng);
        let q = 1.0 / 128.0;
        for v in &dithered {
            assert!((v / q - (v / q).round()).abs() < 1e-9);
        }
        let err_rms = (sig
            .iter()
            .zip(&dithered)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / 8000.0)
            .sqrt();
        assert!(err_rms < 2.5 * q, "dither error {err_rms}");
        // Noise shaping moves error energy to high frequencies.
        let mut rng2 = Rng::new(11);
        let shaped = noise_shaping_dither(&sig, 8, &mut rng2);
        let err: Vec<f64> = sig.iter().zip(&shaped).map(|(a, b)| a - b).collect();
        let spec = crate::transforms::fft::rfft(&err);
        let low: f64 = spec[..1000].iter().map(|c| c.norm_sq()).sum();
        let high: f64 = spec[3000..4000].iter().map(|c| c.norm_sq()).sum();
        assert!(high > 2.0 * low, "shaping low {low} vs high {high}");
    }

    #[test]
    fn test_utilities() {
        let mut x = vec![0.5, -0.25, 0.1];
        gain_db(&mut x, 6.0206);
        assert!((x[0] - 1.0002).abs() < 1e-3);
        normalize_peak(&mut x, 0.0);
        assert!((x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) - 1.0).abs() < 1e-12);
        let mut y = vec![1.0, 1.0, 3.0];
        dc_offset_remove(&mut y);
        assert!(y.iter().sum::<f64>().abs() < 1e-12);
        let mut z: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        z[50] = 5.0;
        let fixed = declick(&z, 1.0);
        assert!((fixed[50] - 0.5 * (z[49] + z[51])).abs() < 1e-12);
        let mut r = vec![0.1, 0.2, 0.3];
        normalize_rms(&mut r, -20.0);
        let rms = (r.iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt();
        assert!((20.0 * rms.log10() + 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_pitch_shift_and_spectral_gate() {
        let fs = 32000.0;
        let f0 = 440.0;
        let x: Vec<f64> = (0..32000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let up = pitch_shift_simple(&x, 12.0, fs);
        // Count zero crossings in the interior.
        let seg = &up[8000..24000];
        let crossings = seg.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        let f_est = crossings as f64 / (seg.len() as f64 / fs);
        assert!((f_est - 880.0).abs() / 880.0 < 0.05, "shifted pitch {f_est}");
        // Spectral gate keeps a loud tone, kills low-level noise.
        let mut rng = Rng::new(2);
        let noisy: Vec<f64> = x
            .iter()
            .map(|&v| v + 0.02 * (2.0 * rng.next_f64() - 1.0))
            .collect();
        let profile = vec![0.5; 129]; // generous noise floor estimate per bin
        let clean = spectral_gate(&noisy, &profile, 0.0, 256, 64);
        // The tone's bin survives.
        let spec = crate::transforms::fft::rfft(&clean[1000..1000 + 8192]);
        let bin = (f0 * 8192.0 / fs).round() as usize;
        assert!(spec[bin].norm() > 100.0, "tone was gated away");
    }
}
