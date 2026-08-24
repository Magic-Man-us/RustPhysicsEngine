//! Sound synthesis: additive, FM (DX7-style operator routing),
//! Karplus-Strong, subtractive, granular, formant, waveshaping, drums,
//! and note/sequence rendering.

use crate::audio::envelope::Adsr;
use crate::audio::oscillators::{polyblep_saw, polyblep_square, polyblep_triangle, NoiseGen};
use crate::audio::oscillators::{NoiseColor, Waveform};
use crate::dsp::iir::Sos;
use crate::math::constants::PI;
use crate::monte_carlo::Rng;
use crate::special::bessel_jn;

const TWO_PI: f64 = 2.0 * PI;

/// Additive synthesis from (ratio, amplitude, phase) partials of a
/// fundamental `freq`.
#[must_use]
pub fn additive(harmonics: &[(f64, f64, f64)], freq: f64, n: usize, fs: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            harmonics
                .iter()
                .map(|&(ratio, amp, phase)| amp * (TWO_PI * freq * ratio * t + phase).sin())
                .sum()
        })
        .collect()
}

/// Additive synthesis with a per-partial amplitude envelope
/// (each envelope is resampled to n output samples).
#[must_use]
pub fn additive_evolving(harmonics: &[(f64, Vec<f64>)], freq: f64, n: usize, fs: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            harmonics
                .iter()
                .map(|(ratio, env)| {
                    let pos = i as f64 / n.max(1) as f64 * (env.len() - 1).max(0) as f64;
                    let i0 = pos.floor() as usize;
                    let i1 = (i0 + 1).min(env.len() - 1);
                    let frac = pos - i0 as f64;
                    let a = env[i0] * (1.0 - frac) + env[i1] * frac;
                    a * (TWO_PI * freq * ratio * t).sin()
                })
                .sum()
        })
        .collect()
}

/// One FM operator: frequency ratio, modulation index (as output
/// amplitude when used as a modulator), envelope, and self-feedback.
pub struct FmOperator {
    pub ratio: f64,
    pub index: f64,
    pub env: Adsr,
    pub feedback: f64,
    pub phase: f64,
    last_out: f64,
}

impl FmOperator {
    /// New operator with the given ratio/index/envelope.
    #[must_use]
    pub fn new(ratio: f64, index: f64, env: Adsr) -> Self {
        Self { ratio, index, env, feedback: 0.0, phase: 0.0, last_out: 0.0 }
    }

    fn tick(&mut self, base_freq: f64, pm_input: f64, fs: f64) -> f64 {
        let out = (TWO_PI * self.phase + pm_input + self.feedback * self.last_out).sin()
            * self.env.next();
        self.phase = (self.phase + base_freq * self.ratio / fs).rem_euclid(1.0);
        self.last_out = out;
        out
    }
}

/// DX7-style FM synth: `algorithm[i]` lists the operators that modulate
/// operator i (an empty list means it is a carrier unless someone else
/// consumes it; operators that appear in no modulation list sum into
/// the output).
pub struct FmSynth {
    pub ops: Vec<FmOperator>,
    pub algorithm: Vec<Vec<usize>>,
    pub fs: f64,
    freq: f64,
}

impl FmSynth {
    /// Build a synth from operators and a routing table.
    #[must_use]
    pub fn new(ops: Vec<FmOperator>, algorithm: Vec<Vec<usize>>, fs: f64) -> Self {
        Self { ops, algorithm, fs, freq: 440.0 }
    }

    /// A few classic 6-op routing tables (1-indexed DX7 numbering
    /// reduced to the topology): 1 = single stack pairs, 32 = all
    /// carriers. Unknown numbers fall back to algorithm 32.
    #[must_use]
    pub fn dx7_algorithm(n: u8) -> Vec<Vec<usize>> {
        match n {
            // Algorithm 1: 2→1, 4→3, 5→4? (simplified two stacks + chain)
            1 => vec![vec![1], vec![], vec![3], vec![4], vec![5], vec![]],
            // Algorithm 5: three 2-op stacks.
            5 => vec![vec![1], vec![], vec![3], vec![], vec![5], vec![]],
            // Algorithm 16: one carrier fed by a tree.
            16 => vec![vec![1, 2, 4], vec![], vec![3], vec![], vec![5], vec![]],
            // Algorithm 32: six parallel carriers.
            _ => vec![vec![], vec![], vec![], vec![], vec![], vec![]],
        }
    }

    /// Gate every operator envelope on at the given frequency.
    pub fn note_on(&mut self, freq: f64) {
        self.freq = freq;
        for op in self.ops.iter_mut() {
            op.phase = 0.0;
            op.env.gate_on();
        }
    }

    /// Gate every envelope off.
    pub fn note_off(&mut self) {
        for op in self.ops.iter_mut() {
            op.env.gate_off();
        }
    }

    /// One output sample (modulators evaluated depth-first each tick).
    #[allow(clippy::should_implement_trait)] // roadmap API name
    pub fn next(&mut self) -> f64 {
        let n_ops = self.ops.len();
        let mut outputs = vec![0.0; n_ops];
        // Evaluate in reverse index order: by convention modulators have
        // higher indices than the operators they feed.
        for i in (0..n_ops).rev() {
            let pm: f64 = self.algorithm[i]
                .iter()
                .map(|&m| self.ops[m].index * outputs[m])
                .sum();
            outputs[i] = {
                let freq = self.freq;
                let fs = self.fs;
                self.ops[i].tick(freq, pm, fs)
            };
        }
        // Carriers: operators no one consumes.
        let mut consumed = vec![false; n_ops];
        for mods in &self.algorithm {
            for &m in mods {
                consumed[m] = true;
            }
        }
        let carriers: Vec<usize> = (0..n_ops).filter(|&i| !consumed[i]).collect();
        let sum: f64 = carriers.iter().map(|&i| outputs[i]).sum();
        sum / (carriers.len().max(1) as f64)
    }

    /// Render a full note (attack at t = 0, release at 70% duration).
    pub fn render(&mut self, freq: f64, duration: f64) -> Vec<f64> {
        let n = (duration * self.fs) as usize;
        self.note_on(freq);
        let release_at = (0.7 * n as f64) as usize;
        (0..n)
            .map(|i| {
                if i == release_at {
                    self.note_off();
                }
                self.next()
            })
            .collect()
    }
}

/// Two-operator FM: sin(2πf_c·t + I·sin(2πf_m·t)).
#[must_use]
pub fn fm_simple(carrier: f64, modulator: f64, index: f64, n: usize, fs: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (TWO_PI * carrier * t + index * (TWO_PI * modulator * t).sin()).sin()
        })
        .collect()
}

/// Bessel sideband amplitudes |J_k(I)| for k = 0..n_sidebands.
#[must_use]
pub fn fm_bessel_sidebands(index: f64, n_sidebands: usize) -> Vec<f64> {
    (0..=n_sidebands).map(|k| bessel_jn(k as u32, index).abs()).collect()
}

/// Phase modulation (identical spectrum to [`fm_simple`] for a sine
/// modulator).
#[must_use]
pub fn pm_simple(carrier: f64, modulator: f64, index: f64, n: usize, fs: f64) -> Vec<f64> {
    fm_simple(carrier, modulator, index, n, fs)
}

/// Amplitude modulation (1 + depth·sin(2πf_m t))·sin(2πf_c t).
#[must_use]
pub fn am(carrier: f64, modulator: f64, depth: f64, n: usize, fs: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (1.0 + depth * (TWO_PI * modulator * t).sin()) * (TWO_PI * carrier * t).sin()
        })
        .collect()
}

/// Ring modulation a·b.
#[must_use]
pub fn ring_mod(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

/// Karplus-Strong plucked string: noise burst through the averaging
/// loop. `decay` scales the loop gain, `blend` the averaging strength.
#[must_use]
pub fn karplus_strong(
    freq: f64,
    duration: f64,
    fs: f64,
    decay: f64,
    blend: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let n = (duration * fs) as usize;
    let period = (fs / freq).round().max(2.0) as usize;
    let mut delay: Vec<f64> = (0..period).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let mut pos = 0usize;
    (0..n)
        .map(|_| {
            let cur = delay[pos];
            let next = delay[(pos + 1) % period];
            delay[pos] = decay * ((1.0 - blend) * cur + blend * 0.5 * (cur + next));
            pos = (pos + 1) % period;
            cur
        })
        .collect()
}

/// Extended Karplus-Strong: pick position comb, pick-direction width
/// low-pass, and a dynamics low-pass on the excitation.
#[must_use]
#[allow(clippy::too_many_arguments)] // signature fixed by the roadmap
pub fn karplus_strong_extended(
    freq: f64,
    duration: f64,
    fs: f64,
    pick_pos: f64,
    pick_width: f64,
    decay: f64,
    dynamics: f64,
) -> Vec<f64> {
    let n = (duration * fs) as usize;
    let period = (fs / freq).round().max(2.0) as usize;
    let mut rng = Rng::new(0xC0FFEE);
    // Excitation: noise → dynamics low-pass → pick-position comb.
    let mut burst: Vec<f64> = (0..period).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let mut state = 0.0;
    for v in burst.iter_mut() {
        state += dynamics.clamp(0.0, 1.0) * (*v - state);
        *v = state;
    }
    let comb_delay = ((pick_pos.clamp(0.01, 0.99)) * period as f64) as usize;
    let combed: Vec<f64> = (0..period)
        .map(|i| burst[i] - (1.0 - pick_width.clamp(0.0, 1.0)) * burst[(i + period - comb_delay.max(1)) % period])
        .collect();
    let mut delay = combed;
    let mut pos = 0usize;
    (0..n)
        .map(|_| {
            let cur = delay[pos];
            let next = delay[(pos + 1) % period];
            delay[pos] = decay * 0.5 * (cur + next);
            pos = (pos + 1) % period;
            cur
        })
        .collect()
}

/// Subtractive synthesis: raw oscillator through a filter with an
/// amplitude envelope (`filter_env_amount` scales a per-sample cutoff
/// bias applied as post-gain tilt on the filtered signal — a simple
/// stand-in for a modulated-cutoff filter).
#[must_use]
pub fn subtractive(
    source: Waveform,
    freq: f64,
    filter: &mut Sos,
    env: &mut Adsr,
    filter_env_amount: f64,
    n: usize,
) -> Vec<f64> {
    let fs = 48000.0;
    let mut phase = 0.0_f64;
    let dt = freq / fs;
    let mut noise = NoiseGen::new(1234, NoiseColor::White);
    env.gate_on();
    (0..n)
        .map(|i| {
            let raw = match source {
                Waveform::Sine => (TWO_PI * phase).sin(),
                Waveform::Saw => polyblep_saw(phase, dt),
                Waveform::Square { duty } => polyblep_square(phase, dt, duty),
                Waveform::Triangle => polyblep_triangle(phase, dt),
                Waveform::Noise(_) => noise.next(),
                Waveform::Wavetable(_) => (TWO_PI * phase).sin(),
            };
            phase = (phase + dt).rem_euclid(1.0);
            if i == (0.8 * n as f64) as usize {
                env.gate_off();
            }
            let e = env.next();
            let filtered = filter.process(raw);
            filtered * e * (1.0 + filter_env_amount * (e - 1.0))
        })
        .collect()
}

/// Granular synthesis: Hann-windowed grains read from a source buffer
/// at `position` (0..1, with `spread` jitter), pitch shifted by
/// resampled playback, `density` grains per second.
#[must_use]
#[allow(clippy::too_many_arguments)] // signature fixed by the roadmap
pub fn granular(
    grain_source: &[f64],
    grain_size: f64,
    density: f64,
    pitch_shift: f64,
    position: f64,
    spread: f64,
    n: usize,
    fs: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut out = vec![0.0; n];
    if grain_source.is_empty() {
        return out;
    }
    let grain_len = (grain_size * fs).max(8.0) as usize;
    let hop = (fs / density.max(0.1)) as usize;
    let mut start = 0usize;
    while start < n {
        let src_center = ((position + spread * (2.0 * rng.next_f64() - 1.0)).clamp(0.0, 1.0)
            * grain_source.len() as f64) as usize;
        for j in 0..grain_len {
            let oi = start + j;
            if oi >= n {
                break;
            }
            let window = 0.5 * (1.0 - (TWO_PI * j as f64 / grain_len as f64).cos());
            let src_pos = src_center as f64 + j as f64 * pitch_shift;
            let s0 = src_pos.floor() as usize;
            if s0 + 1 < grain_source.len() {
                let f = src_pos - s0 as f64;
                let v = grain_source[s0] * (1.0 - f) + grain_source[s0 + 1] * f;
                out[oi] += window * v;
            }
        }
        start += hop.max(1);
    }
    out
}

/// Voice types for [`vowel_formants`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Voice {
    Male,
    Female,
}

/// Classic (Peterson-Barney-style) formant tables for the vowels
/// a, e, i, o, u: (frequency, bandwidth, amplitude) triples.
#[must_use]
pub fn vowel_formants(vowel: char, voice: Voice) -> Vec<(f64, f64, f64)> {
    let scale = match voice {
        Voice::Male => 1.0,
        Voice::Female => 1.15,
    };
    let base: &[(f64, f64, f64)] = match vowel.to_ascii_lowercase() {
        'a' => &[(730.0, 90.0, 1.0), (1090.0, 110.0, 0.5), (2440.0, 170.0, 0.25)],
        'e' => &[(530.0, 80.0, 1.0), (1840.0, 120.0, 0.45), (2480.0, 180.0, 0.25)],
        'i' => &[(270.0, 60.0, 1.0), (2290.0, 130.0, 0.35), (3010.0, 200.0, 0.2)],
        'o' => &[(570.0, 80.0, 1.0), (840.0, 100.0, 0.6), (2410.0, 170.0, 0.15)],
        _ => &[(300.0, 60.0, 1.0), (870.0, 100.0, 0.35), (2240.0, 170.0, 0.1)],
    };
    base.iter().map(|&(f, b, a)| (f * scale, b, a)).collect()
}

/// Formant synthesis: a pulse-train glottal source through parallel
/// resonators (freq, bandwidth, amp).
#[must_use]
pub fn formant_synth(f0: f64, formants: &[(f64, f64, f64)], n: usize, fs: f64) -> Vec<f64> {
    // Parallel two-pole resonators.
    struct Res {
        b0: f64,
        a1: f64,
        a2: f64,
        z1: f64,
        z2: f64,
        amp: f64,
    }
    let mut bank: Vec<Res> = formants
        .iter()
        .map(|&(f, bw, amp)| {
            let r = (-PI * bw / fs).exp();
            let a1 = -2.0 * r * (TWO_PI * f / fs).cos();
            let a2 = r * r;
            Res { b0: 1.0 - r, a1, a2, z1: 0.0, z2: 0.0, amp }
        })
        .collect();
    let period = (fs / f0).round().max(2.0) as usize;
    (0..n)
        .map(|i| {
            let src = if i % period == 0 { 1.0 } else { 0.0 };
            bank.iter_mut()
                .map(|r| {
                    let y = r.b0 * src - r.a1 * r.z1 - r.a2 * r.z2;
                    r.z2 = r.z1;
                    r.z1 = y;
                    r.amp * y
                })
                .sum()
        })
        .collect()
}

/// Pulsar synthesis: a formant-frequency sinusoid burst repeated at f0
/// with the given duty cycle.
#[must_use]
pub fn pulsar_synthesis(f0: f64, formant: f64, duty: f64, n: usize, fs: f64) -> Vec<f64> {
    let period = fs / f0;
    (0..n)
        .map(|i| {
            let ph = (i as f64).rem_euclid(period) / period;
            if ph < duty {
                let u = ph / duty;
                let window = 0.5 * (1.0 - (TWO_PI * u).cos());
                // Elapsed time inside the pulsaret is u·duty/f0 seconds.
                window * (TWO_PI * formant / f0 * u * duty).sin()
            } else {
                0.0
            }
        })
        .collect()
}

/// Casio CZ-style phase distortion: warp the phase ramp before the
/// cosine lookup. `kind` 0 = knee (saw-like), 1 = resonant sweep.
#[must_use]
pub fn phase_distortion(phase: f64, amount: f64, kind: usize) -> f64 {
    let p = phase.rem_euclid(1.0);
    let a = amount.clamp(0.0, 0.99);
    let warped = match kind {
        0 => {
            // Move the ramp midpoint earlier: knee at x = (1−a)/2.
            let knee = (1.0 - a) / 2.0;
            if p < knee {
                0.5 * p / knee
            } else {
                0.5 + 0.5 * (p - knee) / (1.0 - knee)
            }
        }
        _ => (p * (1.0 + 7.0 * a)).min(1.0),
    };
    -(TWO_PI * warped).cos()
}

/// Apply an arbitrary waveshaper.
#[must_use]
pub fn waveshaper(x: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    f(x)
}

/// Chebyshev waveshaper: Σ a_k·T_k(x) turns a pure cosine at amplitude
/// 1 into exactly the requested harmonic mix.
#[must_use]
pub fn chebyshev_waveshaper(x: f64, harmonic_amps: &[f64]) -> f64 {
    // Iterative Chebyshev: T0 = 1, T1 = x, T_{k+1} = 2x·T_k − T_{k−1}.
    let mut t_prev = 1.0;
    let mut t_cur = x;
    let mut acc = 0.0;
    for (k, &a) in harmonic_amps.iter().enumerate() {
        let t_k = if k == 0 {
            t_cur
        } else {
            let t_next = 2.0 * x * t_cur - t_prev;
            t_prev = t_cur;
            t_cur = t_next;
            t_next
        };
        acc += a * t_k;
    }
    acc
}

/// Hard-synced sawtooth: a slave saw retriggered at the master rate.
#[must_use]
pub fn hard_sync_osc(master_freq: f64, slave_freq: f64, n: usize, fs: f64) -> Vec<f64> {
    let mut slave_phase = 0.0_f64;
    let mut master_phase = 0.0_f64;
    let dts = slave_freq / fs;
    let dtm = master_freq / fs;
    (0..n)
        .map(|_| {
            let out = polyblep_saw(slave_phase, dts);
            slave_phase += dts;
            master_phase += dtm;
            if master_phase >= 1.0 {
                master_phase -= 1.0;
                slave_phase = 0.0;
            }
            out
        })
        .collect()
}

/// Detuned saw stack (JP-8000 style supersaw): `detune` is the maximum
/// relative detune of the outer voices.
#[must_use]
pub fn supersaw(freq: f64, detune: f64, n_voices: usize, n: usize, fs: f64) -> Vec<f64> {
    let voices = n_voices.max(1);
    let mut phases: Vec<f64> = (0..voices).map(|v| v as f64 * 0.37 % 1.0).collect();
    let freqs: Vec<f64> = (0..voices)
        .map(|v| {
            let spread = if voices > 1 {
                2.0 * v as f64 / (voices - 1) as f64 - 1.0
            } else {
                0.0
            };
            freq * (1.0 + detune * spread)
        })
        .collect();
    (0..n)
        .map(|_| {
            let mut acc = 0.0;
            for (p, &f) in phases.iter_mut().zip(&freqs) {
                acc += polyblep_saw(*p, f / fs);
                *p = (*p + f / fs).rem_euclid(1.0);
            }
            acc / voices as f64
        })
        .collect()
}

/// 2D vector synthesis: bilinear mix of four source buffers at (x, y).
#[must_use]
pub fn vector_synth(sources: [&[f64]; 4], x: f64, y: f64) -> Vec<f64> {
    let n = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    let (x, y) = (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
    let w = [
        (1.0 - x) * (1.0 - y),
        x * (1.0 - y),
        (1.0 - x) * y,
        x * y,
    ];
    (0..n)
        .map(|i| {
            sources
                .iter()
                .zip(&w)
                .map(|(s, &wv)| wv * s[i])
                .sum()
        })
        .collect()
}

/// Sample playback with loop points and linear-interpolated rate
/// conversion.
#[must_use]
pub fn sample_playback(
    sample: &[f64],
    rate_ratio: f64,
    loop_start: usize,
    loop_end: usize,
    n: usize,
) -> Vec<f64> {
    if sample.is_empty() {
        return vec![0.0; n];
    }
    let loop_end = loop_end.min(sample.len()).max(loop_start + 1);
    let mut pos = 0.0_f64;
    (0..n)
        .map(|_| {
            if pos >= loop_end as f64 {
                pos = loop_start as f64 + (pos - loop_end as f64);
            }
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(sample.len() - 1);
            let f = pos - i0 as f64;
            let v = sample[i0.min(sample.len() - 1)] * (1.0 - f) + sample[i1] * f;
            pos += rate_ratio;
            v
        })
        .collect()
}

/// Kick drum: exponential pitch sweep with an exponential amplitude
/// decay.
#[must_use]
pub fn drum_kick(fs: f64, pitch_start: f64, pitch_end: f64, decay: f64) -> Vec<f64> {
    let n = (decay * 6.0 * fs) as usize;
    let mut phase = 0.0_f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let f = pitch_end + (pitch_start - pitch_end) * (-t / (decay * 0.2)).exp();
            phase += f / fs;
            (TWO_PI * phase).sin() * (-t / decay).exp()
        })
        .collect()
}

/// Snare: tone plus band-passed noise, both decaying.
#[must_use]
pub fn drum_snare(fs: f64) -> Vec<f64> {
    let n = (0.25 * fs) as usize;
    let mut noise = NoiseGen::new(99, NoiseColor::White);
    let mut phase = 0.0_f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            phase += 180.0 / fs;
            let tone = (TWO_PI * phase).sin() * (-t / 0.06).exp();
            let hiss = noise.next() * (-t / 0.12).exp();
            0.5 * tone + 0.5 * hiss
        })
        .collect()
}

/// Hi-hat: short bright filtered noise burst.
#[must_use]
pub fn drum_hihat(fs: f64) -> Vec<f64> {
    let n = (0.08 * fs) as usize;
    let mut noise = NoiseGen::new(7, NoiseColor::Violet);
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            noise.next() * (-t / 0.02).exp()
        })
        .collect()
}

/// Clap: a few staggered noise bursts.
#[must_use]
pub fn drum_clap(fs: f64) -> Vec<f64> {
    let n = (0.25 * fs) as usize;
    let mut noise = NoiseGen::new(13, NoiseColor::Pink);
    let bursts = [0.0, 0.011, 0.023, 0.036];
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let env: f64 = bursts
                .iter()
                .map(|&b| if t >= b { (-(t - b) / 0.03).exp() } else { 0.0 })
                .fold(0.0, f64::max);
            noise.next() * env
        })
        .collect()
}

/// Tom: pitch-swept sine, longer than a kick.
#[must_use]
pub fn drum_tom(fs: f64, pitch: f64) -> Vec<f64> {
    drum_kick(fs, pitch * 1.4, pitch, 0.12)
}

/// A monophonic note generator.
pub trait Synth {
    /// Start a note at frequency (Hz) and velocity 0..1.
    fn note_on(&mut self, freq: f64, vel: f64);
    /// Release the current note.
    fn note_off(&mut self);
    /// Next output sample.
    fn next(&mut self) -> f64;
}

impl Synth for FmSynth {
    fn note_on(&mut self, freq: f64, _vel: f64) {
        FmSynth::note_on(self, freq);
    }
    fn note_off(&mut self) {
        FmSynth::note_off(self);
    }
    fn next(&mut self) -> f64 {
        FmSynth::next(self)
    }
}

/// MIDI note number to frequency (A4 = 69 = 440 Hz).
fn midi_to_freq(midi: u8) -> f64 {
    440.0 * 2.0_f64.powf((midi as f64 - 69.0) / 12.0)
}

/// Render one note through any [`Synth`].
pub fn render_note(
    synth: &mut dyn Synth,
    midi: u8,
    velocity: f64,
    duration: f64,
    fs: f64,
) -> Vec<f64> {
    let n = (duration * fs) as usize;
    synth.note_on(midi_to_freq(midi), velocity);
    let release_at = (0.75 * n as f64) as usize;
    (0..n)
        .map(|i| {
            if i == release_at {
                synth.note_off();
            }
            velocity * synth.next()
        })
        .collect()
}

/// Render a (time s, midi, duration s, velocity) note list into one
/// buffer.
pub fn render_sequence(
    synth: &mut dyn Synth,
    notes: &[(f64, u8, f64, f64)],
    fs: f64,
) -> Vec<f64> {
    let end = notes
        .iter()
        .map(|&(t, _, d, _)| t + d)
        .fold(0.0_f64, f64::max);
    let mut out = vec![0.0; (end * fs).ceil() as usize + 1];
    for &(t, midi, dur, vel) in notes {
        let rendered = render_note(synth, midi, vel, dur, fs);
        let start = (t * fs) as usize;
        for (i, v) in rendered.iter().enumerate() {
            if start + i < out.len() {
                out[start + i] += v;
            }
        }
    }
    out
}

/// Mix tracks with per-track gains (output as long as the longest track).
#[must_use]
pub fn mix(tracks: &[Vec<f64>], gains: &[f64]) -> Vec<f64> {
    let n = tracks.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![0.0; n];
    for (track, &g) in tracks.iter().zip(gains) {
        for (o, &v) in out.iter_mut().zip(track) {
            *o += g * v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::fft::rfft;

    fn spectrum_peaks(x: &[f64], fs: f64) -> Vec<(f64, f64)> {
        let n = x.len();
        let spec = rfft(x);
        let mags: Vec<f64> = spec.iter().map(|c| 2.0 * c.norm() / n as f64).collect();
        let mut peaks = Vec::new();
        for i in 1..mags.len() - 1 {
            if mags[i] > mags[i - 1] && mags[i] > mags[i + 1] && mags[i] > 1e-3 {
                peaks.push((i as f64 * fs / n as f64, mags[i]));
            }
        }
        peaks
    }

    #[test]
    fn test_additive_single_harmonic_is_sine() {
        let fs = 8000.0;
        let x = additive(&[(1.0, 0.7, 0.0)], 200.0, 4000, fs);
        for (i, &v) in x.iter().enumerate() {
            let expect = 0.7 * (TWO_PI * 200.0 * i as f64 / fs).sin();
            assert!((v - expect).abs() < 1e-12);
        }
        // Evolving version fades its partial.
        let y = additive_evolving(&[(1.0, vec![1.0, 0.0])], 200.0, 4000, fs);
        assert!(y[3999].abs() < 0.01);
    }

    #[test]
    fn test_fm_spectrum_matches_bessel() {
        let fs = 32768.0;
        let (fc, fm, index) = (4096.0, 512.0, 1.5);
        let n = 32768;
        let x = fm_simple(fc, fm, index, n, fs);
        let spec = rfft(&x);
        let mag_at = |f: f64| {
            let bin = (f * n as f64 / fs).round() as usize;
            2.0 * spec[bin].norm() / n as f64
        };
        let bessel = fm_bessel_sidebands(index, 3);
        // Carrier and first three sideband pairs match |J_k(I)| within 1 dB.
        for (k, &jk) in bessel.iter().enumerate() {
            if jk < 1e-3 {
                continue;
            }
            let up = mag_at(fc + k as f64 * fm);
            let ratio_db = 20.0 * (up / jk).log10();
            assert!(ratio_db.abs() < 1.0, "sideband {k}: {up} vs J = {jk}");
        }
    }

    #[test]
    fn test_karplus_strong_pitch() {
        let fs = 44100.0;
        let target = 220.0;
        let mut rng = Rng::new(5);
        let x = karplus_strong(target, 1.0, fs, 0.998, 1.0, &mut rng);
        // Measure pitch from autocorrelation of the sustained portion.
        let seg = &x[8000..30000];
        let min_lag = (fs / 500.0) as usize;
        let max_lag = (fs / 100.0) as usize;
        let mut best = (0usize, f64::MIN);
        for lag in min_lag..max_lag {
            let mut acc = 0.0;
            for i in 0..seg.len() - lag {
                acc += seg[i] * seg[i + lag];
            }
            if acc > best.1 {
                best = (lag, acc);
            }
        }
        let f_est = fs / best.0 as f64;
        let cents = 1200.0 * (f_est / target).log2().abs();
        // The plain KS loop is tuned to the rounded period plus the
        // half-sample averaging delay: allow that quantization plus
        // integer-lag measurement error.
        assert!(cents < 20.0, "pitch off by {cents} cents ({f_est} Hz)");
        // Decays over time.
        let early: f64 = x[..4000].iter().map(|v| v * v).sum();
        let late: f64 = x[40000..].iter().map(|v| v * v).sum::<f64>() * (4000.0 / 4100.0);
        assert!(late < early);
    }

    #[test]
    fn test_chebyshev_waveshaper_exact_harmonics() {
        let fs = 8192.0;
        let n = 8192;
        let amps = [0.0, 0.5, 0.3]; // only 2nd and 3rd harmonics
        let f0 = 128.0;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let c = (TWO_PI * f0 * i as f64 / fs).cos();
                chebyshev_waveshaper(c, &amps)
            })
            .collect();
        let spec = rfft(&x);
        let mag_at = |f: f64| {
            let bin = (f * n as f64 / fs).round() as usize;
            2.0 * spec[bin].norm() / n as f64
        };
        assert!(mag_at(f0) < 1e-9, "fundamental leak {}", mag_at(f0));
        assert!((mag_at(2.0 * f0) - 0.5).abs() < 1e-9);
        assert!((mag_at(3.0 * f0) - 0.3).abs() < 1e-9);
        assert!(mag_at(4.0 * f0) < 1e-9);
    }

    #[test]
    fn test_fm_synth_routing_and_render() {
        let fs = 16000.0;
        let env = || {
            let mut e = Adsr::new(0.005, 0.01, 0.8, 0.05, fs);
            e.set_curve(false);
            e
        };
        let ops = vec![
            FmOperator::new(1.0, 2.0, env()),
            FmOperator::new(2.0, 2.0, env()),
        ];
        let mut synth = FmSynth::new(ops, vec![vec![1], vec![]], fs);
        let out = synth.render(200.0, 0.3);
        assert_eq!(out.len(), 4800);
        let peak = out.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(peak > 0.3 && peak <= 1.2, "peak {peak}");
        // Modulated spectrum is richer than a bare sine: count peaks.
        let peaks = spectrum_peaks(&out[500..4000], fs);
        assert!(peaks.len() >= 3, "spectrum too clean: {peaks:?}");
        // Trait-object note rendering.
        let mut synth2 = FmSynth::new(
            vec![FmOperator::new(1.0, 1.0, env())],
            vec![vec![]],
            fs,
        );
        let note = render_note(&mut synth2, 69, 0.5, 0.2, fs);
        assert_eq!(note.len(), 3200);
    }

    #[test]
    fn test_formant_synth_peaks_near_formants() {
        let fs = 16000.0;
        let formants = vowel_formants('a', Voice::Male);
        let x = formant_synth(110.0, &formants, 16000, fs);
        let spec = rfft(&x);
        let mags: Vec<f64> = spec.iter().map(|c| c.norm()).collect();
        // Energy near F1 (730 Hz) dominates energy near 3.5 kHz valley.
        let band = |f_lo: f64, f_hi: f64| -> f64 {
            let lo = (f_lo * x.len() as f64 / fs) as usize;
            let hi = (f_hi * x.len() as f64 / fs) as usize;
            mags[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        };
        assert!(band(650.0, 820.0) > 5.0 * band(3300.0, 3700.0));
        // Female scaling moves formants up.
        let f_f = vowel_formants('a', Voice::Female);
        assert!(f_f[0].0 > formants[0].0);
    }

    #[test]
    fn test_granular_and_sample_playback() {
        let mut rng = Rng::new(3);
        let source: Vec<f64> = (0..8000).map(|i| (i as f64 * 0.05).sin()).collect();
        let g = granular(&source, 0.03, 40.0, 1.0, 0.5, 0.1, 4000, 8000.0, &mut rng);
        assert_eq!(g.len(), 4000);
        assert!(g.iter().any(|v| v.abs() > 0.05));
        // Sample playback at 2x loops and doubles frequency.
        let sample: Vec<f64> = (0..1000).map(|i| (TWO_PI * i as f64 / 100.0).sin()).collect();
        let played = sample_playback(&sample, 2.0, 0, 1000, 2000);
        let crossings = played.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        // 2000 output samples advance 4000 source samples = 40 cycles.
        assert!((crossings as i64 - 40).abs() <= 2, "{crossings}");
    }

    #[test]
    fn test_supersaw_hard_sync_vector() {
        let fs = 48000.0;
        let ss = supersaw(220.0, 0.01, 7, 4800, fs);
        assert!(ss.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) <= 1.0 + 1e-9);
        // Master period chosen to be an integer number of samples so the
        // output is exactly periodic sample-to-sample.
        let hs = hard_sync_osc(240.0, 550.0, 4800, fs);
        let period = (fs / 240.0).round() as usize;
        let mut dev = 0.0_f64;
        for i in 500..1500 {
            dev = dev.max((hs[i] - hs[i + period]).abs());
        }
        assert!(dev < 0.15, "sync periodicity dev {dev}");
        let a = vec![1.0; 100];
        let b = vec![2.0; 100];
        let c = vec![3.0; 100];
        let d = vec![4.0; 100];
        let v = vector_synth([&a, &b, &c, &d], 1.0, 1.0);
        assert!((v[0] - 4.0).abs() < 1e-12);
        let v2 = vector_synth([&a, &b, &c, &d], 0.5, 0.5);
        assert!((v2[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_drums_and_sequences() {
        let fs = 16000.0;
        let kick = drum_kick(fs, 120.0, 45.0, 0.15);
        assert!(kick.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) > 0.5);
        // Kick ends quiet.
        assert!(kick[kick.len() - 1].abs() < 0.01);
        assert!(!drum_snare(fs).is_empty());
        assert!(!drum_hihat(fs).is_empty());
        assert!(!drum_clap(fs).is_empty());
        assert!(!drum_tom(fs, 100.0).is_empty());
        // Sequence render places notes at their times.
        let env = Adsr::new(0.005, 0.01, 0.8, 0.02, fs);
        let mut synth = FmSynth::new(vec![FmOperator::new(1.0, 1.0, env)], vec![vec![]], fs);
        let seq = render_sequence(&mut synth, &[(0.0, 60, 0.1, 0.8), (0.2, 64, 0.1, 0.8)], fs);
        let early: f64 = seq[..1600].iter().map(|v| v * v).sum();
        let gap: f64 = seq[2600..3100].iter().map(|v| v * v).sum();
        let second: f64 = seq[3300..4700].iter().map(|v| v * v).sum();
        assert!(early > 10.0 * gap && second > 10.0 * gap);
        let m = mix(&[vec![1.0; 10], vec![1.0; 20]], &[0.5, 0.25]);
        assert_eq!(m.len(), 20);
        assert!((m[0] - 0.75).abs() < 1e-12 && (m[15] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_waveshaper_identity_and_cubic_harmonics() {
        // The identity shaping function is a no-op for every input.
        for &x in &[-2.0_f64, -0.5, 0.0, 0.25, 1.0, 7.5] {
            assert_eq!(waveshaper(x, &|v| v), x);
        }
        // A hard clipper bounds the output and passes its linear region
        // through untouched.
        let clip = |v: f64| v.clamp(-0.5, 0.5);
        for i in 0..=200 {
            let x = -2.0 + 4.0 * i as f64 / 200.0;
            let y = waveshaper(x, &clip);
            assert!(y.abs() <= 0.5 + 1e-12, "clip {x} -> {y}");
            if x.abs() <= 0.5 {
                assert!((y - x).abs() < 1e-12);
            }
        }
        // Cubic shaping of a unit cosine has the closed form
        // cos³θ = (3cosθ + cos3θ)/4: exactly 0.75 at the fundamental,
        // 0.25 at the third harmonic, nothing anywhere else.
        let fs = 8192.0;
        let n = 8192;
        let f0 = 256.0;
        let y: Vec<f64> = (0..n)
            .map(|i| waveshaper((TWO_PI * f0 * i as f64 / fs).cos(), &|v| v * v * v))
            .collect();
        let spec = rfft(&y);
        let mag_at = |f: f64| 2.0 * spec[(f * n as f64 / fs).round() as usize].norm() / n as f64;
        assert!((mag_at(f0) - 0.75).abs() < 1e-9, "fundamental {}", mag_at(f0));
        assert!((mag_at(3.0 * f0) - 0.25).abs() < 1e-9, "3rd {}", mag_at(3.0 * f0));
        assert!(mag_at(2.0 * f0) < 1e-9, "even harmonic leak");
        assert!(mag_at(5.0 * f0) < 1e-9, "5th harmonic leak");
    }

    #[test]
    fn test_subtractive_lowpass_removes_high_harmonics() {
        use crate::dsp::iir::{butterworth, IirKind};
        let fs = 48000.0; // subtractive() is hard-wired to 48 kHz
        let freq = 200.0;
        let n = 16384;
        let mut filter = butterworth(4, IirKind::Lowpass(600.0), fs);
        // sustain = 1 means the envelope is exactly 1 from the end of the
        // attack until the gate-off at 0.8n.
        let mut env = Adsr::new(0.001, 0.001, 1.0, 0.05, fs);
        let out = subtractive(Waveform::Saw, freq, &mut filter, &mut env, 0.0, n);
        assert_eq!(out.len(), n);
        // Independent path: the same PolyBLEP saw through a freshly
        // designed copy of the same filter must agree sample for sample.
        let dt = freq / fs;
        let mut phase = 0.0_f64;
        let raw: Vec<f64> = (0..n)
            .map(|_| {
                let v = polyblep_saw(phase, dt);
                phase = (phase + dt).rem_euclid(1.0);
                v
            })
            .collect();
        let mut check = butterworth(4, IirKind::Lowpass(600.0), fs);
        let filtered = check.process_block(&raw);
        for i in 100..12000 {
            assert!(
                (out[i] - filtered[i]).abs() < 1e-12,
                "at {i}: {} vs {}",
                out[i],
                filtered[i]
            );
        }
        // A 4th-order 600 Hz low-pass is ~42 dB down at 2 kHz and steeper
        // above, so the filtered saw keeps < 0.1% of the raw saw's energy
        // there.
        let band_energy = |x: &[f64]| -> f64 {
            let spec = rfft(x);
            let lo = (2000.0 * x.len() as f64 / fs) as usize;
            spec[lo..].iter().map(|c| c.norm_sq()).sum()
        };
        let e_raw = band_energy(&raw[2048..10240]);
        let e_lp = band_energy(&out[2048..10240]);
        assert!(e_lp < 1e-3 * e_raw, "high-band energy {e_lp} vs raw {e_raw}");
        // The fundamental survives essentially untouched (|H(200 Hz)| ≈ 1).
        let fund = |x: &[f64]| {
            let spec = rfft(x);
            2.0 * spec[(freq * x.len() as f64 / fs).round() as usize].norm() / x.len() as f64
        };
        let ratio = fund(&out[2048..10240]) / fund(&raw[2048..10240]);
        let expect = filter.freq_response(freq, fs).norm();
        assert!((ratio - expect).abs() < 0.02, "fundamental gain {ratio} vs {expect}");
    }

    #[test]
    fn test_pulsar_synthesis_duty_cycle_and_formant() {
        let fs = 48000.0;
        let f0 = 100.0;
        let period = 480usize; // fs/f0 exactly
        let n = 8 * period;
        let duty = 0.25;
        let x = pulsar_synthesis(f0, 800.0, duty, n, fs);
        assert_eq!(x.len(), n);
        // Every sample whose phase is past the duty cycle is exactly silent.
        for (i, &v) in x.iter().enumerate() {
            let ph = (i % period) as f64 / period as f64;
            if ph >= duty {
                assert_eq!(v, 0.0, "sample {i} (phase {ph}) should be silent");
            }
        }
        // 3/4 of every period is silent, plus the Hann window's own zero at
        // the start of each pulsaret.
        let zeros = x.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 3 * n / 4 + n / period, "{zeros} silent samples");
        // The pulsaret train is not silent.
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        assert!(rms > 0.05, "rms {rms}");
        // Exactly periodic at the fundamental.
        for i in 0..(n - period) {
            assert!((x[i] - x[i + period]).abs() < 1e-12, "period mismatch at {i}");
        }
        // The spectrum is a harmonic series of f0 whose envelope peaks at
        // the formant frequency (here 8 harmonics up).
        let spec = rfft(&x);
        let bin_hz = fs / n as f64;
        let (peak_bin, _) = spec
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .unwrap();
        let peak_hz = peak_bin as f64 * bin_hz;
        assert!((peak_hz - 800.0).abs() < 150.0, "formant peak at {peak_hz} Hz");
        // Energy lives on multiples of f0 only.
        for k in 1..30 {
            let mid = ((k as f64 + 0.5) * f0 / bin_hz).round() as usize;
            let on = spec[(k as f64 * f0 / bin_hz).round() as usize].norm();
            assert!(spec[mid].norm() < 1e-9 * on.max(1e-9) + 1e-9, "inter-harmonic {k}");
        }
    }

    #[test]
    fn test_karplus_strong_extended_pitch_and_decay() {
        let fs = 44100.0;
        let target = 220.0;
        let x = karplus_strong_extended(target, 1.0, fs, 0.25, 0.5, 0.99, 0.5);
        assert_eq!(x.len(), 44100);
        // Pitch from the autocorrelation of the sustained portion.
        let seg = &x[4000..24000];
        let min_lag = (fs / 500.0) as usize;
        let max_lag = (fs / 100.0) as usize;
        let mut best = (0usize, f64::MIN);
        for lag in min_lag..max_lag {
            let mut acc = 0.0;
            for i in 0..seg.len() - lag {
                acc += seg[i] * seg[i + lag];
            }
            if acc > best.1 {
                best = (lag, acc);
            }
        }
        let f_est = fs / best.0 as f64;
        let cents = 1200.0 * (f_est / target).log2().abs();
        // The loop is quantized to the rounded period (200 samples) plus
        // the half-sample delay of the averaging filter, so the achievable
        // pitch is 219.95 Hz; integer-lag measurement adds ~5 cents.
        assert!(cents < 25.0, "pitch off by {cents} cents ({f_est} Hz)");
        // Loop gain 0.99 per round trip over 220 round trips leaves ~11%
        // of the amplitude: the string decays without dying immediately.
        let energy = |s: &[f64]| s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64;
        let early = energy(&x[..4000]);
        let late = energy(&x[40000..]);
        assert!(late < 0.2 * early, "no decay: {early} -> {late}");
        assert!(late > 1e-6 * early, "died too fast: {early} -> {late}");
        // A faster decay setting really does decay faster.
        let quick = karplus_strong_extended(target, 1.0, fs, 0.25, 0.5, 0.95, 0.5);
        assert!(energy(&quick[40000..]) < late, "decay parameter has no effect");
    }

    #[test]
    fn test_pm_simple_matches_fm_and_zero_index_is_a_sine() {
        let fs = 8192.0;
        let n = 8192;
        let (fc, fm) = (1024.0, 128.0);
        // Documented equivalence: phase modulation by a sine is the same
        // signal as frequency modulation by a sine.
        assert_eq!(pm_simple(fc, fm, 2.0, n, fs), fm_simple(fc, fm, 2.0, n, fs));
        // Zero modulation index collapses to the bare carrier sine.
        let plain = pm_simple(fc, fm, 0.0, n, fs);
        for (i, &v) in plain.iter().enumerate() {
            let expect = (TWO_PI * fc * i as f64 / fs).sin();
            assert!((v - expect).abs() < 1e-12, "at {i}");
        }
        let spec = rfft(&plain);
        let mag_at = |f: f64| 2.0 * spec[(f * n as f64 / fs).round() as usize].norm() / n as f64;
        assert!((mag_at(fc) - 1.0).abs() < 1e-9, "carrier {}", mag_at(fc));
        for k in 1..=4 {
            assert!(mag_at(fc + k as f64 * fm) < 1e-9, "upper sideband {k}");
            assert!(mag_at(fc - k as f64 * fm) < 1e-9, "lower sideband {k}");
        }
        // Parseval: all of the energy is in the carrier bin.
        let total: f64 = spec.iter().map(|c| c.norm_sq()).sum();
        let carrier_bin = (fc * n as f64 / fs).round() as usize;
        assert!(spec[carrier_bin].norm_sq() > 0.999_999 * total, "energy leaked");
    }

    #[test]
    fn test_dx7_algorithm_routing_selects_carriers() {
        // Algorithm 32 is six independent carriers; unknown numbers fall
        // back to it.
        let a32 = FmSynth::dx7_algorithm(32);
        assert_eq!(a32.len(), 6);
        assert!(a32.iter().all(Vec::is_empty));
        assert_eq!(FmSynth::dx7_algorithm(200), a32);
        // Algorithm 1 stacks 2→1 and chains 6→5→4→3.
        assert_eq!(
            FmSynth::dx7_algorithm(1),
            vec![vec![1], vec![], vec![3], vec![4], vec![5], vec![]]
        );
        // Algorithm 16 feeds one carrier from a tree.
        assert_eq!(FmSynth::dx7_algorithm(16)[0], vec![1, 2, 4]);

        // Algorithm 5 is three 2-op stacks, so operators 0, 2 and 4 are the
        // carriers. Silencing the modulators (index 0) must leave exactly
        // the three carrier partials, each at 1/3 amplitude.
        let fs = 8192.0;
        let freq = 128.0;
        let env = || {
            // sustain = 1: the envelope is exactly 1 after ~9 samples.
            Adsr::new(0.001, 0.001, 1.0, 0.1, fs)
        };
        let ops: Vec<FmOperator> = (1..=6)
            .map(|r| {
                let index = if r % 2 == 0 { 0.0 } else { 3.0 };
                FmOperator::new(r as f64, index, env())
            })
            .collect();
        let mut synth = FmSynth::new(ops, FmSynth::dx7_algorithm(5), fs);
        synth.note_on(freq);
        let n = 8192;
        let all: Vec<f64> = (0..n + 512).map(|_| synth.next()).collect();
        let spec = rfft(&all[512..]);
        let mag_at = |h: f64| 2.0 * spec[(h * freq * n as f64 / fs).round() as usize].norm()
            / n as f64;
        for h in [1.0, 3.0, 5.0] {
            assert!((mag_at(h) - 1.0 / 3.0).abs() < 1e-3, "carrier h{h}: {}", mag_at(h));
        }
        for h in [2.0, 4.0, 6.0] {
            assert!(mag_at(h) < 1e-3, "modulator h{h} leaked: {}", mag_at(h));
        }
        // Algorithm 32 with the same operators sums all six as carriers.
        let ops32: Vec<FmOperator> = (1..=6)
            .map(|r| FmOperator::new(r as f64, 0.0, env()))
            .collect();
        let mut synth32 = FmSynth::new(ops32, FmSynth::dx7_algorithm(32), fs);
        synth32.note_on(freq);
        let all32: Vec<f64> = (0..n + 512).map(|_| synth32.next()).collect();
        let spec32 = rfft(&all32[512..]);
        for h in 1..=6 {
            let m = 2.0 * spec32[(h as f64 * freq * n as f64 / fs).round() as usize].norm()
                / n as f64;
            assert!((m - 1.0 / 6.0).abs() < 1e-3, "algorithm 32 h{h}: {m}");
        }
    }

    #[test]
    fn test_am_ring_pd() {
        let fs = 8192.0;
        let n = 8192;
        let x = am(1024.0, 128.0, 0.5, n, fs);
        let spec = rfft(&x);
        let mag_at = |f: f64| 2.0 * spec[(f * n as f64 / fs).round() as usize].norm() / n as f64;
        assert!((mag_at(1024.0) - 1.0).abs() < 0.01);
        assert!((mag_at(1024.0 + 128.0) - 0.25).abs() < 0.01);
        assert!((mag_at(1024.0 - 128.0) - 0.25).abs() < 0.01);
        let r = ring_mod(&[2.0, 3.0], &[0.5, -1.0]);
        assert_eq!(r, vec![1.0, -3.0]);
        // Phase distortion adds harmonics vs a pure cosine.
        let pd: Vec<f64> = (0..n)
            .map(|i| phase_distortion(128.0 * i as f64 / fs, 0.8, 0))
            .collect();
        let ps = rfft(&pd);
        let h2 = 2.0 * ps[(2.0 * 128.0 * n as f64 / fs) as usize].norm() / n as f64;
        assert!(h2 > 0.05, "PD second harmonic {h2}");
    }
}
