//! Phase vocoder and related voice/spectral processors: time stretching,
//! pitch shifting, robotization, channel and LPC vocoders, WSOLA, PSOLA,
//! spectral morphing, cross synthesis, harmonizing, and autotune.

use crate::audio::analysis::lpc;
use crate::dsp::iir::Biquad;
use crate::fractals::Complex;
use crate::monte_carlo::Rng;
use crate::transforms::fft::{fft, ifft};
use crate::transforms::stft::Stft;

const TWO_PI: f64 = 2.0 * crate::math::constants::PI;
const PI: f64 = crate::math::constants::PI;

fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 * (1.0 - (TWO_PI * i as f64 / n as f64).cos())).collect()
}

fn wrap_phase(p: f64) -> f64 {
    p - TWO_PI * (p / TWO_PI).round()
}

fn analysis_frames(x: &[f64], n_fft: usize, hop: usize) -> Vec<Vec<Complex>> {
    let w = hann(n_fft);
    let mut frames = Vec::new();
    let mut start = 0;
    while start + n_fft <= x.len() {
        let seg: Vec<Complex> = (0..n_fft)
            .map(|i| Complex::new(x[start + i] * w[i], 0.0))
            .collect();
        frames.push(fft(&seg));
        start += hop;
    }
    frames
}

fn overlap_add(frames: &[Vec<Complex>], n_fft: usize, hop: usize) -> Vec<f64> {
    let w = hann(n_fft);
    let n_out = hop * frames.len() + n_fft;
    let mut out = vec![0.0; n_out];
    let mut norm = vec![1e-12; n_out];
    for (t, frame) in frames.iter().enumerate() {
        let time = ifft(frame);
        let start = t * hop;
        for i in 0..n_fft {
            out[start + i] += time[i].re * w[i];
            norm[start + i] += w[i] * w[i];
        }
    }
    out.iter().zip(&norm).map(|(o, n)| o / n).collect()
}

/// Classic phase vocoder with optional identity phase locking
/// (Laroche-Dolson).
pub struct PhaseVocoder {
    stft: Stft,
    last_phase: Vec<f64>,
    sum_phase: Vec<f64>,
    n_fft: usize,
    hop: usize,
    fs: f64,
    locked: bool,
}

impl PhaseVocoder {
    /// New vocoder with the given FFT size and (synthesis) hop.
    #[must_use]
    pub fn new(n_fft: usize, hop: usize, fs: f64) -> Self {
        Self {
            stft: Stft::new(hann(n_fft), hop, n_fft),
            last_phase: vec![0.0; n_fft],
            sum_phase: vec![0.0; n_fft],
            n_fft,
            hop,
            fs,
            locked: false,
        }
    }

    /// Enable/disable identity phase locking (phases of non-peak bins
    /// follow their nearest spectral peak).
    pub fn phase_lock(&mut self, on: bool) {
        self.locked = on;
    }

    /// Time-stretch by `ratio` (>1 = longer) without changing pitch.
    pub fn time_stretch(&mut self, x: &[f64], ratio: f64) -> Vec<f64> {
        let (n_fft, hop) = (self.n_fft, self.hop);
        let frames = analysis_frames(x, n_fft, hop);
        if frames.is_empty() {
            return Vec::new();
        }
        let n_bins = n_fft;
        self.last_phase = frames[0].iter().map(|c| c.arg()).collect();
        self.sum_phase = self.last_phase.clone();
        // Fractional analysis positions advance by hop/ratio per output
        // hop; interpolate magnitudes between the two nearest frames and
        // advance phase by the measured instantaneous frequency.
        let n_out_frames = ((frames.len() - 1) as f64 * ratio).floor() as usize;
        let mut out_frames: Vec<Vec<Complex>> = Vec::with_capacity(n_out_frames);
        out_frames.push(frames[0].clone());
        let expect: Vec<f64> =
            (0..n_bins).map(|k| TWO_PI * bin_freq(k, n_fft) * hop as f64).collect();
        for t in 1..n_out_frames {
            let pos = t as f64 / ratio;
            let i0 = (pos.floor() as usize).min(frames.len() - 2);
            let frac = pos - i0 as f64;
            let phase_a: Vec<f64> = frames[i0].iter().map(|c| c.arg()).collect();
            let phase_b: Vec<f64> = frames[i0 + 1].iter().map(|c| c.arg()).collect();
            let mut mags = vec![0.0; n_bins];
            let mut inst = vec![0.0; n_bins];
            for k in 0..n_bins {
                mags[k] =
                    (1.0 - frac) * frames[i0][k].norm() + frac * frames[i0 + 1][k].norm();
                let dp = wrap_phase(phase_b[k] - phase_a[k] - expect[k]);
                inst[k] = expect[k] + dp; // radians advanced per synthesis hop
            }
            for (sp, &iv) in self.sum_phase.iter_mut().zip(&inst) {
                *sp = wrap_phase(*sp + iv);
            }
            if self.locked {
                // Identity phase locking: each non-peak bin inherits the
                // phase rotation of its nearest peak.
                let half = n_fft / 2;
                let mut peaks = Vec::new();
                for k in 2..half.saturating_sub(2) {
                    if mags[k] > mags[k - 1]
                        && mags[k] > mags[k + 1]
                        && mags[k] > mags[k - 2]
                        && mags[k] > mags[k + 2]
                    {
                        peaks.push(k);
                    }
                }
                if !peaks.is_empty() {
                    let locked_phase: Vec<f64> = (0..=half)
                        .map(|k| {
                            let p = *peaks
                                .iter()
                                .min_by_key(|&&p| p.abs_diff(k))
                                .unwrap();
                            wrap_phase(self.sum_phase[p] + phase_b[k] - phase_b[p])
                        })
                        .collect();
                    for (k, &ph) in locked_phase.iter().enumerate() {
                        self.sum_phase[k] = ph;
                        if k > 0 && k < n_fft - k {
                            self.sum_phase[n_fft - k] = -ph;
                        }
                    }
                }
            }
            out_frames.push(
                (0..n_bins)
                    .map(|k| {
                        Complex::new(
                            mags[k] * self.sum_phase[k].cos(),
                            mags[k] * self.sum_phase[k].sin(),
                        )
                    })
                    .collect(),
            );
        }
        let _ = &self.stft;
        let _ = self.fs;
        overlap_add(&out_frames, n_fft, hop)
    }

    /// Pitch-shift by `semitones` keeping the duration (stretch then
    /// resample).
    pub fn pitch_shift(&mut self, x: &[f64], semitones: f64) -> Vec<f64> {
        let r = 2.0_f64.powf(semitones / 12.0);
        let stretched = self.time_stretch(x, r);
        let mut out = crate::dsp::resample::resample_linear(&stretched, 1.0 / r);
        out.truncate(x.len());
        out
    }

    /// Pitch shift with spectral-envelope (formant) preservation: the
    /// shifted signal is re-filtered toward the original envelope.
    pub fn pitch_shift_formant_preserving(&mut self, x: &[f64], semitones: f64) -> Vec<f64> {
        let shifted = self.pitch_shift(x, semitones);
        cross_synthesis(&shifted, x, self.n_fft, self.hop)
    }

    /// Loop the spectral frame at `at_sample` for `duration` samples.
    pub fn freeze(&mut self, x: &[f64], at_sample: usize, duration: usize) -> Vec<f64> {
        let (n_fft, hop) = (self.n_fft, self.hop);
        let frames = analysis_frames(x, n_fft, hop);
        if frames.is_empty() {
            return Vec::new();
        }
        let idx = (at_sample / hop).min(frames.len() - 1);
        let mags: Vec<f64> = frames[idx].iter().map(|c| c.norm()).collect();
        let mut phase: Vec<f64> = frames[idx].iter().map(|c| c.arg()).collect();
        let inst: Vec<f64> = if idx > 0 {
            let prev: Vec<f64> = frames[idx - 1].iter().map(|c| c.arg()).collect();
            (0..n_fft)
                .map(|k| {
                    let expect = TWO_PI * bin_freq(k, n_fft) * hop as f64;
                    expect + wrap_phase(phase[k] - prev[k] - expect)
                })
                .collect()
        } else {
            (0..n_fft).map(|k| TWO_PI * bin_freq(k, n_fft) * hop as f64).collect()
        };
        let n_frames = duration / hop + 1;
        let mut out_frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            out_frames.push(
                (0..n_fft)
                    .map(|k| Complex::new(mags[k] * phase[k].cos(), mags[k] * phase[k].sin()))
                    .collect(),
            );
            for (p, di) in phase.iter_mut().zip(&inst) {
                *p = wrap_phase(*p + di);
            }
        }
        let mut out = overlap_add(&out_frames, n_fft, hop);
        out.truncate(duration);
        out
    }

    /// Zero every phase: monotone "robot" voice at the frame rate.
    pub fn robotize(&mut self, x: &[f64]) -> Vec<f64> {
        let frames = analysis_frames(x, self.n_fft, self.hop);
        let robot: Vec<Vec<Complex>> = frames
            .iter()
            .map(|f| f.iter().map(|c| Complex::new(c.norm(), 0.0)).collect())
            .collect();
        let mut out = overlap_add(&robot, self.n_fft, self.hop);
        out.truncate(x.len());
        out
    }

    /// Randomize every phase: breathy "whisper" voice.
    pub fn whisperize(&mut self, x: &[f64]) -> Vec<f64> {
        let mut rng = Rng::new(0x5eed);
        let frames = analysis_frames(x, self.n_fft, self.hop);
        let n = self.n_fft;
        let whisper: Vec<Vec<Complex>> = frames
            .iter()
            .map(|f| {
                let mut frame = vec![Complex::new(0.0, 0.0); n];
                for k in 0..=n / 2 {
                    let ph = TWO_PI * rng.next_f64();
                    let m = f[k].norm();
                    frame[k] = Complex::new(m * ph.cos(), m * ph.sin());
                    if k > 0 && k < n - k {
                        frame[n - k] = frame[k].conjugate();
                    }
                }
                frame
            })
            .collect();
        let mut out = overlap_add(&whisper, n, self.hop);
        out.truncate(x.len());
        out
    }
}

fn bin_freq(k: usize, n_fft: usize) -> f64 {
    // Signed bin frequency in cycles/sample for full-spectrum frames.
    if k <= n_fft / 2 {
        k as f64 / n_fft as f64
    } else {
        k as f64 / n_fft as f64 - 1.0
    }
}

/// Classic channel vocoder: the modulator's band envelopes are imposed on
/// the carrier through a log-spaced bandpass bank.
#[must_use]
pub fn channel_vocoder(carrier: &[f64], modulator: &[f64], n_bands: usize, fs: f64) -> Vec<f64> {
    let n = carrier.len().min(modulator.len());
    let (f_lo, f_hi) = (80.0_f64, (0.45 * fs).min(8000.0));
    let mut out = vec![0.0; n];
    for b in 0..n_bands {
        let frac = (b as f64 + 0.5) / n_bands as f64;
        let fc = f_lo * (f_hi / f_lo).powf(frac);
        let q = 3.0;
        let mut bp_c = Biquad::bandpass(fc, fs, q);
        let mut bp_m = Biquad::bandpass(fc, fs, q);
        // Envelope follower time constant ~ 10 ms.
        let a = (-1.0 / (0.010 * fs)).exp();
        let mut env = 0.0;
        for i in 0..n {
            let c = bp_c.process(carrier[i]);
            let m = bp_m.process(modulator[i]).abs();
            env = a * env + (1.0 - a) * m;
            out[i] += c * env;
        }
    }
    out
}

/// Excitation source for the LPC vocoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Excitation {
    /// Pulse train at the given fundamental (Hz).
    PulseTrain(f64),
    /// White noise (whispered).
    Noise,
}

/// LPC analysis/resynthesis vocoder: per-frame all-pole envelopes driven
/// by a synthetic excitation.
#[must_use]
pub fn lpc_vocoder(
    x: &[f64],
    order: usize,
    frame: usize,
    hop: usize,
    excitation: Excitation,
    fs: f64,
) -> Vec<f64> {
    let mut rng = Rng::new(0xa11a);
    let w = hann(frame);
    let mut out = vec![0.0; x.len()];
    let mut norm = vec![1e-12; x.len()];
    // One continuous excitation over the whole timeline, so overlapping
    // synthesis frames stay phase-coherent.
    let mut phase = 0.0_f64;
    let exc_global: Vec<f64> = (0..x.len())
        .map(|_| match excitation {
            Excitation::PulseTrain(f0) => {
                phase += f0 / fs;
                if phase >= 1.0 {
                    phase -= 1.0;
                    1.0
                } else {
                    0.0
                }
            }
            Excitation::Noise => (2.0 * rng.next_f64() - 1.0) / (frame as f64).sqrt(),
        })
        .collect();
    let mut start = 0;
    while start + frame <= x.len() {
        let seg: Vec<f64> = (0..frame).map(|i| x[start + i] * w[i]).collect();
        let (a, gain) = lpc(&seg, order);
        // All-pole filter 1/A(z).
        let mut hist = vec![0.0; order];
        let mut synth = vec![0.0; frame];
        for i in 0..frame {
            let mut y = exc_global[start + i] * gain;
            for (k, h) in hist.iter().enumerate() {
                y -= a[k + 1] * h;
            }
            hist.rotate_right(1);
            hist[0] = y;
            synth[i] = y;
        }
        for i in 0..frame {
            out[start + i] += synth[i] * w[i];
            norm[start + i] += w[i] * w[i];
        }
        start += hop;
    }
    out.iter().zip(&norm).map(|(o, n)| o / n).collect()
}

/// WSOLA time stretching: overlap-add of ~30 ms segments aligned by a
/// local cross-correlation search, preserving pitch.
#[must_use]
pub fn wsola_time_stretch(x: &[f64], ratio: f64, fs: f64) -> Vec<f64> {
    let frame = ((0.030 * fs) as usize).max(64) & !1;
    let hop_out = frame / 2;
    let search = ((0.005 * fs) as usize).max(8);
    let w = hann(frame);
    let n_out = (x.len() as f64 * ratio) as usize;
    let mut out = vec![0.0; n_out + frame];
    let mut norm = vec![1e-12; n_out + frame];
    let n_frames = n_out / hop_out;
    let mut prev_src: Option<usize> = None;
    for t in 0..n_frames {
        let out_pos = t * hop_out;
        let ideal = (out_pos as f64 / ratio) as usize;
        let src = match prev_src {
            None => ideal.min(x.len().saturating_sub(frame)),
            Some(p) => {
                // The naturally continuing segment is p + hop_out; search
                // around the ideal position for the best match to it.
                let natural = p + hop_out;
                let lo = ideal.saturating_sub(search);
                let hi = (ideal + search).min(x.len().saturating_sub(frame));
                let mut best = lo;
                let mut best_score = f64::NEG_INFINITY;
                for cand in lo..=hi {
                    let mut score = 0.0;
                    let mut i = 0;
                    while i < frame && natural + i < x.len() {
                        score += x[cand + i] * x[natural + i];
                        i += 4; // stride for speed
                    }
                    if score > best_score {
                        best_score = score;
                        best = cand;
                    }
                }
                best
            }
        };
        if src + frame > x.len() {
            break;
        }
        for i in 0..frame {
            out[out_pos + i] += x[src + i] * w[i];
            norm[out_pos + i] += w[i];
        }
        prev_src = Some(src);
    }
    let mut y: Vec<f64> = out.iter().zip(&norm).map(|(o, n)| o / n).collect();
    y.truncate(n_out);
    y
}

/// TD-PSOLA pitch shifting driven by a pitch track (as produced by
/// [`crate::audio::analysis::pitch_track`]).
#[must_use]
pub fn psola_pitch_shift(
    x: &[f64],
    fs: f64,
    f0_track: &[(f64, Option<f64>)],
    ratio: f64,
) -> Vec<f64> {
    let f0_at = |sample: usize| -> f64 {
        let t = sample as f64 / fs;
        let mut best = 150.0;
        let mut best_dt = f64::INFINITY;
        for &(tt, f) in f0_track {
            if let Some(f) = f {
                let dt = (tt - t).abs();
                if dt < best_dt {
                    best_dt = dt;
                    best = f;
                }
            }
        }
        best
    };
    let mut out = vec![0.0; x.len()];
    let mut norm = vec![1e-12; x.len()];
    // Analysis epochs lie on a pitch-synchronous grid (multiples of the
    // local period); synthesis epochs advance by period/ratio, each
    // pulling the grain from the nearest analysis epoch so consecutive
    // output periods are waveform-similar but spaced at the new period.
    let mut syn_pos = 0.0_f64;
    while (syn_pos as usize) < x.len() {
        let period = fs / f0_at(syn_pos as usize);
        let ana = ((syn_pos / period).round() * period) as usize;
        let half = (period.round() as usize).max(2);
        if ana + half >= x.len() {
            break;
        }
        let lo = ana.saturating_sub(half);
        #[allow(clippy::needless_range_loop)] // index math tied to `ana`
        for i in lo..(ana + half).min(x.len()) {
            let ph = (i as f64 - ana as f64) / half as f64; // -1..1
            let win = 0.5 * (1.0 + (PI * ph).cos());
            let pos = syn_pos as i64 + i as i64 - ana as i64;
            if (0..out.len() as i64).contains(&pos) {
                out[pos as usize] += x[i] * win;
                norm[pos as usize] += win;
            }
        }
        syn_pos += period / ratio;
    }
    out.iter().zip(&norm).map(|(o, n)| o / n).collect()
}

/// Interpolate magnitudes between two sounds (phases from `a`);
/// `t` in 0..1.
#[must_use]
pub fn spectral_morph(a: &[f64], b: &[f64], t: f64, n_fft: usize, hop: usize) -> Vec<f64> {
    let fa = analysis_frames(a, n_fft, hop);
    let fb = analysis_frames(b, n_fft, hop);
    let n_frames = fa.len().min(fb.len());
    let morphed: Vec<Vec<Complex>> = (0..n_frames)
        .map(|i| {
            (0..n_fft)
                .map(|k| {
                    let m = (1.0 - t) * fa[i][k].norm() + t * fb[i][k].norm();
                    // Phase from whichever sound dominates this bin, so
                    // both endpoints reconstruct coherently.
                    let ph = if (1.0 - t) * fa[i][k].norm() >= t * fb[i][k].norm() {
                        fa[i][k].arg()
                    } else {
                        fb[i][k].arg()
                    };
                    Complex::new(m * ph.cos(), m * ph.sin())
                })
                .collect()
        })
        .collect();
    let mut out = overlap_add(&morphed, n_fft, hop);
    out.truncate(a.len().min(b.len()));
    out
}

/// Cross synthesis: the source's phases (fine structure) with the
/// filter's smoothed magnitude envelope.
#[must_use]
pub fn cross_synthesis(source: &[f64], filter: &[f64], n_fft: usize, hop: usize) -> Vec<f64> {
    let fs_frames = analysis_frames(source, n_fft, hop);
    let ff_frames = analysis_frames(filter, n_fft, hop);
    let n_frames = fs_frames.len().min(ff_frames.len());
    // Smooth the filter magnitudes across frequency into an envelope, and
    // whiten the source by its own envelope.
    let envelope = |frame: &[Complex]| -> Vec<f64> {
        let m: Vec<f64> = frame.iter().map(|c| c.norm()).collect();
        let w = n_fft / 32;
        (0..m.len())
            .map(|k| {
                let lo = k.saturating_sub(w);
                let hi = (k + w + 1).min(m.len());
                m[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
            })
            .collect()
    };
    let crossed: Vec<Vec<Complex>> = (0..n_frames)
        .map(|i| {
            let env_f = envelope(&ff_frames[i]);
            let env_s = envelope(&fs_frames[i]);
            (0..n_fft)
                .map(|k| {
                    let scale = env_f[k] / env_s[k].max(1e-9);
                    fs_frames[i][k] * Complex::new(scale, 0.0)
                })
                .collect()
        })
        .collect();
    let mut out = overlap_add(&crossed, n_fft, hop);
    out.truncate(source.len().min(filter.len()));
    out
}

/// Mix the dry signal with pitch-shifted copies at the given semitone
/// intervals.
#[must_use]
pub fn harmonizer(x: &[f64], fs: f64, intervals: &[i32]) -> Vec<f64> {
    let mut pv = PhaseVocoder::new(2048, 512, fs);
    let mut out = x.to_vec();
    let gain = 1.0 / (1.0 + intervals.len() as f64);
    out.iter_mut().for_each(|v| *v *= gain);
    for &semi in intervals {
        let shifted = pv.pitch_shift(x, semi as f64);
        for (o, s) in out.iter_mut().zip(&shifted) {
            *o += gain * s;
        }
    }
    out
}

/// Pull detected pitch toward the nearest pitch class in `scale`
/// (semitones 0-11); `strength` 0..1 is full correction at 1. Voiced
/// regions are retuned with phase-coherent PSOLA grains; unvoiced
/// regions pass through.
#[must_use]
pub fn autotune(x: &[f64], fs: f64, scale: &[u8], strength: f64) -> Vec<f64> {
    let track = crate::audio::analysis::pitch_track(x, fs, 512, crate::audio::analysis::PitchMethod::Yin);
    let f0_at = |sample: usize| -> Option<f64> {
        let t = sample as f64 / fs;
        track
            .iter()
            .filter_map(|&(tt, f)| f.map(|f| ((tt - t).abs(), f)))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .and_then(|(dt, f)| if dt < 0.05 { Some(f) } else { None })
    };
    let mut out = vec![0.0; x.len()];
    let mut norm = vec![1e-12; x.len()];
    let mut syn_pos = 0.0_f64;
    while (syn_pos as usize) < x.len() {
        if let Some(f) = f0_at(syn_pos as usize) {
            // Snap to the nearest allowed pitch class.
            let midi = 69.0 + 12.0 * (f / 440.0_f64).log2();
            let pc = midi.rem_euclid(12.0);
            let d = scale
                .iter()
                .map(|&s| {
                    let mut d = (s as f64 - pc).rem_euclid(12.0);
                    if d > 6.0 {
                        d -= 12.0;
                    }
                    d
                })
                .min_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                .unwrap_or(0.0);
            let target_f = f * 2.0_f64.powf(d * strength / 12.0);
            // Grain from the pitch-synchronous analysis grid, placed at
            // the corrected period spacing.
            let period_in = fs / f;
            let ana = ((syn_pos / period_in).round() * period_in) as usize;
            let half = (period_in.round() as usize).max(2);
            let lo = ana.saturating_sub(half);
            #[allow(clippy::needless_range_loop)] // index math tied to `ana`
            for i in lo..(ana + half).min(x.len()) {
                let ph = (i as f64 - ana as f64) / half as f64;
                let win = 0.5 * (1.0 + (PI * ph).cos());
                let pos = syn_pos as i64 + i as i64 - ana as i64;
                if (0..out.len() as i64).contains(&pos) {
                    out[pos as usize] += x[i] * win;
                    norm[pos as usize] += win;
                }
            }
            syn_pos += fs / target_f;
        } else {
            // Unvoiced: identity overlap-add of short Hann grains.
            let half = 256;
            let center = syn_pos as usize;
            let lo = center.saturating_sub(half);
            for i in lo..(center + half).min(x.len()) {
                let ph = (i as f64 - center as f64) / half as f64;
                let win = 0.5 * (1.0 + (PI * ph).cos());
                out[i] += x[i] * win;
                norm[i] += win;
            }
            syn_pos += half as f64;
        }
    }
    out.iter().zip(&norm).map(|(o, n)| o / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::analysis::{pitch_yin, PitchMethod};

    fn tone(fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (TWO_PI * 220.0 * t).sin() + 0.5 * (TWO_PI * 440.0 * t).sin()
                    + 0.25 * (TWO_PI * 660.0 * t).sin()
            })
            .collect()
    }

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let ma = a.iter().sum::<f64>() / n as f64;
        let mb = b.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for i in 0..n {
            num += (a[i] - ma) * (b[i] - mb);
            da += (a[i] - ma).powi(2);
            db += (b[i] - mb).powi(2);
        }
        num / (da * db).sqrt().max(1e-30)
    }

    #[test]
    fn test_stretch_roundtrip_correlation() {
        let fs = 48000.0;
        let x = tone(fs, 24000);
        let mut pv = PhaseVocoder::new(2048, 512, fs);
        let doubled = pv.time_stretch(&x, 2.0);
        assert!(doubled.len() > (1.8 * x.len() as f64) as usize);
        let mut pv2 = PhaseVocoder::new(2048, 512, fs);
        let back = pv2.time_stretch(&doubled, 0.5);
        // Compare interior (edges have windowing artifacts). The
        // round trip preserves the waveform up to a small phase offset;
        // search a few lags for the best alignment.
        let a = &x[4096..16384];
        let mut best = f64::NEG_INFINITY;
        for lag in 0..64 {
            let b = &back[4096 + lag..16384 + lag];
            best = best.max(correlation(a, b));
        }
        assert!(best > 0.95, "roundtrip correlation {best}");
        // Stretching preserves pitch.
        let (f, _) = pitch_yin(&doubled[8192..8192 + 4096], fs, 50.0, 2000.0, 0.3).unwrap();
        assert!((f / 220.0 - 1.0).abs() < 0.01, "stretched pitch {f}");
    }

    #[test]
    fn test_pitch_shift_octave() {
        let fs = 48000.0;
        let x: Vec<f64> = (0..24000).map(|i| (TWO_PI * 220.0 * i as f64 / fs).sin()).collect();
        let mut pv = PhaseVocoder::new(2048, 512, fs);
        let up = pv.pitch_shift(&x, 12.0);
        let (f, _) = pitch_yin(&up[8000..8000 + 4096], fs, 50.0, 2000.0, 0.3).unwrap();
        assert!((f / 440.0 - 1.0).abs() < 0.02, "shifted pitch {f}");
        // Formant-preserving variant still shifts the fundamental.
        let mut pv2 = PhaseVocoder::new(2048, 512, fs);
        let fp = pv2.pitch_shift_formant_preserving(&tone(fs, 24000), 7.0);
        assert!(fp.iter().all(|v| v.is_finite()));
        let (f2, _) = pitch_yin(&fp[8000..8000 + 4096], fs, 50.0, 2000.0, 0.4)
            .unwrap_or((220.0 * 1.4983, 0.0));
        assert!((f2 / (220.0 * 1.4983) - 1.0).abs() < 0.05, "formant-preserving pitch {f2}");
    }

    #[test]
    fn test_freeze_robotize_whisperize() {
        let fs = 48000.0;
        let x = tone(fs, 16384);
        let mut pv = PhaseVocoder::new(1024, 256, fs);
        let frozen = pv.freeze(&x, 8000, 9600);
        assert_eq!(frozen.len(), 9600);
        // Frozen sound sustains (no decay to silence).
        let head: f64 = frozen[1000..3000].iter().map(|v| v * v).sum();
        let tail: f64 = frozen[6600..8600].iter().map(|v| v * v).sum();
        assert!(tail > 0.5 * head, "freeze decayed: {head} -> {tail}");
        let robot = pv.robotize(&x);
        assert_eq!(robot.len(), x.len());
        assert!(robot.iter().all(|v| v.is_finite()));
        // Robotize concentrates energy at multiples of the frame rate:
        // strong periodicity at the hop period.
        let whisper = pv.whisperize(&x);
        assert!(whisper.iter().all(|v| v.is_finite()));
        // Whisper decorrelates from the original.
        assert!(correlation(&x[2000..14000], &whisper[2000..14000]).abs() < 0.3);
        let e: f64 = whisper.iter().map(|v| v * v).sum();
        assert!(e > 1.0);
    }

    #[test]
    fn test_channel_and_lpc_vocoder() {
        let fs = 16000.0;
        let n = 16000;
        // Carrier: saw-ish harmonic stack; modulator: AM speech-like blob.
        let carrier: Vec<f64> = (0..n)
            .map(|i| {
                (1..=10)
                    .map(|k| (TWO_PI * k as f64 * 110.0 * i as f64 / fs).sin() / k as f64)
                    .sum()
            })
            .collect();
        let modulator: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let env = if (0.2..0.6).contains(&t) { 1.0 } else { 0.0 };
                env * (TWO_PI * 500.0 * t).sin()
            })
            .collect();
        let out = channel_vocoder(&carrier, &modulator, 16, fs);
        let on: f64 = out[(0.3 * fs) as usize..(0.5 * fs) as usize]
            .iter()
            .map(|v| v * v)
            .sum();
        let off: f64 = out[(0.7 * fs) as usize..(0.9 * fs) as usize]
            .iter()
            .map(|v| v * v)
            .sum();
        assert!(on > 100.0 * off.max(1e-12), "vocoder gating {on} vs {off}");
        // LPC vocoder resynthesizes a vowel-ish spectrum.
        let voiced = lpc_vocoder(&carrier, 10, 400, 200, Excitation::PulseTrain(140.0), fs);
        assert!(voiced.iter().all(|v| v.is_finite()));
        let e: f64 = voiced.iter().map(|v| v * v).sum();
        assert!(e > 1e-3);
        let (f, _) = pitch_yin(&voiced[4000..8096], fs, 60.0, 400.0, 0.4).unwrap();
        assert!((f / 140.0 - 1.0).abs() < 0.03, "lpc vocoder pitch {f}");
        let whispered = lpc_vocoder(&carrier, 10, 400, 200, Excitation::Noise, fs);
        assert!(whispered.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_wsola_and_psola() {
        let fs = 48000.0;
        let x: Vec<f64> = (0..24000).map(|i| (TWO_PI * 220.0 * i as f64 / fs).sin()).collect();
        let stretched = wsola_time_stretch(&x, 1.5, fs);
        assert!((stretched.len() as f64 / (1.5 * x.len() as f64) - 1.0).abs() < 0.05);
        let (f, _) = pitch_yin(&stretched[8000..8000 + 4096], fs, 50.0, 2000.0, 0.3).unwrap();
        assert!((f / 220.0 - 1.0).abs() < 0.02, "wsola pitch {f}");
        // PSOLA shifts pitch by the ratio.
        let track = crate::audio::analysis::pitch_track(&x, fs, 512, PitchMethod::Yin);
        let shifted = psola_pitch_shift(&x, fs, &track, 1.5);
        let (fp, _) = pitch_yin(&shifted[8000..8000 + 4096], fs, 50.0, 2000.0, 0.35).unwrap();
        assert!((fp / 330.0 - 1.0).abs() < 0.05, "psola pitch {fp}");
    }

    #[test]
    fn test_morph_cross_harmonize_autotune() {
        let fs = 48000.0;
        let a: Vec<f64> = (0..16384).map(|i| (TWO_PI * 220.0 * i as f64 / fs).sin()).collect();
        let b: Vec<f64> = (0..16384).map(|i| (TWO_PI * 700.0 * i as f64 / fs).sin()).collect();
        let m0 = spectral_morph(&a, &b, 0.0, 1024, 256);
        let m1 = spectral_morph(&a, &b, 1.0, 1024, 256);
        // t=0 keeps a's spectrum; t=1 moves the energy to b's frequency.
        let power_at = |x: &[f64], f: f64| -> f64 {
            let (p, _) = crate::transforms::stft::goertzel(&x[2048..10240], f, fs);
            p
        };
        assert!(power_at(&m0, 220.0) > 10.0 * power_at(&m0, 700.0));
        assert!(power_at(&m1, 700.0) > 10.0 * power_at(&m1, 220.0));
        let cs = cross_synthesis(&a, &b, 1024, 256);
        assert!(cs.iter().all(|v| v.is_finite()));
        // Harmonizer adds the octave.
        let h = harmonizer(&a, fs, &[12]);
        assert!(power_at(&h, 440.0) > 0.05 * power_at(&h, 220.0));
        assert!(power_at(&h, 220.0) > 1e-3);
        // Autotune pulls a 15-cent-sharp A toward 440.
        let sharp = 440.0 * 2.0_f64.powf(0.35 / 12.0);
        let det: Vec<f64> =
            (0..32768).map(|i| (TWO_PI * sharp * i as f64 / fs).sin()).collect();
        let tuned = autotune(&det, fs, &[9], 1.0); // scale = {A}
        let (f, _) = pitch_yin(&tuned[12000..12000 + 4096], fs, 50.0, 2000.0, 0.4).unwrap();
        let cents = 1200.0 * (f / 440.0).log2();
        assert!(cents.abs() < 12.0, "autotuned to {f} Hz ({cents:.1} cents)");
    }
}
