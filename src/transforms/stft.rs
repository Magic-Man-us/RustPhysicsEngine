//! Short-time Fourier transform, spectrograms, Goertzel, chirp-z, and
//! constant-Q analysis.

use crate::dsp::windows::{window, WindowKind};
use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::transforms::fft::{fft, ifft, irfft, next_power_of_two, rfft};

const TWO_PI: f64 = 2.0 * PI;
const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

/// Short-time Fourier transform configuration: an analysis window, hop
/// size in samples, and FFT length (≥ window length; frames are
/// zero-padded up to it).
pub struct Stft {
    pub window: Vec<f64>,
    pub hop: usize,
    pub n_fft: usize,
}

impl Stft {
    /// Build an STFT plan.
    ///
    /// # Panics
    /// Panics if the hop is zero, the window is empty, or longer than n_fft.
    #[must_use]
    pub fn new(window: Vec<f64>, hop: usize, n_fft: usize) -> Self {
        assert!(hop > 0, "hop must be positive");
        assert!(!window.is_empty(), "window must be non-empty");
        assert!(window.len() <= n_fft, "window must fit in n_fft");
        Self { window, hop, n_fft }
    }

    /// Forward STFT: one spectrum of n_fft/2 + 1 bins per frame. Frames
    /// start at k·hop and the last full frame ends within the signal.
    #[must_use]
    pub fn forward(&self, x: &[f64]) -> Vec<Vec<Complex>> {
        let wl = self.window.len();
        if x.len() < wl {
            return Vec::new();
        }
        let n_frames = 1 + (x.len() - wl) / self.hop;
        (0..n_frames)
            .map(|f| {
                let start = f * self.hop;
                let mut frame = vec![0.0; self.n_fft];
                for (i, w) in self.window.iter().enumerate() {
                    frame[i] = x[start + i] * w;
                }
                rfft(&frame)
            })
            .collect()
    }

    /// Inverse STFT by weighted overlap-add: each frame is inverse
    /// transformed, windowed again, accumulated, and normalized by the
    /// accumulated squared window. Exact wherever the window overlap
    /// covers the signal (COLA-satisfying window/hop pairs).
    #[must_use]
    pub fn inverse(&self, frames: &[Vec<Complex>]) -> Vec<f64> {
        if frames.is_empty() {
            return Vec::new();
        }
        let wl = self.window.len();
        let out_len = (frames.len() - 1) * self.hop + wl;
        let mut num = vec![0.0; out_len];
        let mut den = vec![0.0; out_len];
        for (f, spec) in frames.iter().enumerate() {
            let time = irfft(spec, self.n_fft);
            let start = f * self.hop;
            for (i, &w) in self.window.iter().enumerate() {
                num[start + i] += time[i] * w;
                den[start + i] += w * w;
            }
        }
        num.iter()
            .zip(&den)
            .map(|(&n, &d)| if d > 1e-12 { n / d } else { 0.0 })
            .collect()
    }

    /// Magnitude of each bin per frame.
    #[must_use]
    pub fn magnitude(frames: &[Vec<Complex>]) -> Vec<Vec<f64>> {
        frames
            .iter()
            .map(|f| f.iter().map(|c| c.norm()).collect())
            .collect()
    }

    /// Power in dB (10 log₁₀|X|², floored at −300 dB).
    #[must_use]
    pub fn power_db(frames: &[Vec<Complex>]) -> Vec<Vec<f64>> {
        frames
            .iter()
            .map(|f| {
                f.iter()
                    .map(|c| (10.0 * c.norm_sq().log10()).max(-300.0))
                    .collect()
            })
            .collect()
    }

    /// Frame-center times (seconds) for a signal of `n_samples` at `fs`.
    #[must_use]
    pub fn times(&self, n_samples: usize, fs: f64) -> Vec<f64> {
        let wl = self.window.len();
        if n_samples < wl {
            return Vec::new();
        }
        let n_frames = 1 + (n_samples - wl) / self.hop;
        (0..n_frames)
            .map(|f| (f * self.hop) as f64 / fs + wl as f64 / (2.0 * fs))
            .collect()
    }

    /// Bin center frequencies (Hz) for sample rate `fs`.
    #[must_use]
    pub fn freqs(&self, fs: f64) -> Vec<f64> {
        (0..=self.n_fft / 2)
            .map(|k| k as f64 * fs / self.n_fft as f64)
            .collect()
    }

    /// True when the squared window overlap-adds to a constant at this
    /// hop (perfect weighted-OLA reconstruction in the interior).
    #[must_use]
    pub fn is_cola(&self) -> bool {
        let wl = self.window.len();
        if self.hop > wl {
            return false;
        }
        // Accumulate Σ w²(n − m·hop) for interior samples.
        let reps = 3 * wl / self.hop + 3;
        let total = reps * self.hop + wl;
        let mut acc = vec![0.0; total];
        for m in 0..=reps {
            for (i, &w) in self.window.iter().enumerate() {
                acc[m * self.hop + i] += w * w;
            }
        }
        let mid = &acc[wl..total - wl];
        let mean = mid.iter().sum::<f64>() / mid.len() as f64;
        mid.iter().all(|&v| (v - mean).abs() < 1e-9 * mean.max(1e-300))
    }
}

/// Power spectrogram: (frame times, bin frequencies, |X|² per frame).
#[must_use]
pub fn spectrogram(
    x: &[f64],
    fs: f64,
    n_fft: usize,
    hop: usize,
    window_kind: WindowKind,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let stft = Stft::new(window(window_kind, n_fft, true), hop, n_fft);
    let frames = stft.forward(x);
    let power: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| f.iter().map(|c| c.norm_sq()).collect())
        .collect();
    (stft.times(x.len(), fs), stft.freqs(fs), power)
}

/// Frequency in mels (HTK convention).
fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
}

/// Triangular mel filterbank: `n_mels` rows of n_fft/2 + 1 weights.
///
/// # Panics
/// Panics if the frequency range is empty or fmax exceeds Nyquist.
#[must_use]
pub fn mel_filterbank(n_fft: usize, fs: f64, n_mels: usize, fmin: f64, fmax: f64) -> Vec<Vec<f64>> {
    assert!(fmin >= 0.0 && fmin < fmax, "need 0 <= fmin < fmax");
    assert!(fmax <= fs / 2.0 + 1e-9, "fmax must not exceed Nyquist");
    let n_bins = n_fft / 2 + 1;
    let mel_lo = hz_to_mel(fmin);
    let mel_hi = hz_to_mel(fmax);
    let centers: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_lo + (mel_hi - mel_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    (0..n_mels)
        .map(|m| {
            let (lo, mid, hi) = (centers[m], centers[m + 1], centers[m + 2]);
            (0..n_bins)
                .map(|k| {
                    let f = k as f64 * fs / n_fft as f64;
                    if f <= lo || f >= hi {
                        0.0
                    } else if f <= mid {
                        (f - lo) / (mid - lo)
                    } else {
                        (hi - f) / (hi - mid)
                    }
                })
                .collect()
        })
        .collect()
}

/// Mel-scale power spectrogram: one vector of `n_mels` band energies per
/// frame (Hann window).
#[must_use]
pub fn mel_spectrogram(
    x: &[f64],
    fs: f64,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<Vec<f64>> {
    let (_, _, power) = spectrogram(x, fs, n_fft, hop, WindowKind::Hann);
    let fb = mel_filterbank(n_fft, fs, n_mels, fmin, fmax);
    power
        .iter()
        .map(|frame| {
            fb.iter()
                .map(|filt| filt.iter().zip(frame).map(|(w, p)| w * p).sum())
                .collect()
        })
        .collect()
}

/// Goertzel single-bin DFT at an arbitrary frequency: returns the
/// (magnitude, phase) of Σ x\[n\]·e^(−jωn), ω = 2π·target/fs.
#[must_use]
pub fn goertzel(x: &[f64], target_freq: f64, fs: f64) -> (f64, f64) {
    let omega = TWO_PI * target_freq / fs;
    let coeff = 2.0 * omega.cos();
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for &v in x {
        let s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    // X = s1 − e^{−jω} s2, then compensate the phase reference to n = 0.
    let z = Complex::new(s1 - omega.cos() * s2, omega.sin() * s2);
    let ref_ph = cis(-omega * (x.len() as f64 - 1.0));
    let out = z * ref_ph;
    (out.norm(), out.arg())
}

/// Goertzel magnitudes for a set of frequencies.
#[must_use]
pub fn goertzel_bank(x: &[f64], freqs: &[f64], fs: f64) -> Vec<f64> {
    freqs.iter().map(|&f| goertzel(x, f, fs).0).collect()
}

/// Decode one DTMF digit from a tone burst; None when no clear
/// row/column pair dominates.
#[must_use]
pub fn dtmf_decode(x: &[f64], fs: f64) -> Option<char> {
    const ROWS: [f64; 4] = [697.0, 770.0, 852.0, 941.0];
    const COLS: [f64; 4] = [1209.0, 1336.0, 1477.0, 1633.0];
    const KEYS: [[char; 4]; 4] = [
        ['1', '2', '3', 'A'],
        ['4', '5', '6', 'B'],
        ['7', '8', '9', 'C'],
        ['*', '0', '#', 'D'],
    ];
    let rmag = goertzel_bank(x, &ROWS, fs);
    let cmag = goertzel_bank(x, &COLS, fs);
    let (ri, rbest) = argmax(&rmag)?;
    let (ci, cbest) = argmax(&cmag)?;
    // The winning pair must dominate the runners-up clearly.
    let rsecond = rmag.iter().enumerate().filter(|&(i, _)| i != ri).map(|(_, &v)| v).fold(0.0, f64::max);
    let csecond = cmag.iter().enumerate().filter(|&(i, _)| i != ci).map(|(_, &v)| v).fold(0.0, f64::max);
    if rbest > 4.0 * rsecond && cbest > 4.0 * csecond && rbest > 0.0 && cbest > 0.0 {
        Some(KEYS[ri][ci])
    } else {
        None
    }
}

fn argmax(v: &[f64]) -> Option<(usize, f64)> {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &m)| (i, m))
}

/// Chirp-z transform: X\[k\] = Σ x\[n\]·a^(−n)·w^(nk) for k = 0..m−1,
/// evaluated in O((n+m) log(n+m)) by Bluestein's substitution.
#[must_use]
pub fn chirp_z(x: &[Complex], m: usize, w: Complex, a: Complex) -> Vec<Complex> {
    let n = x.len();
    if n == 0 || m == 0 {
        return vec![ZERO; m];
    }
    // w^{nk} = w^{(n² + k² − (k−n)²)/2}
    let pow_half = |base: Complex, e2: f64| -> Complex {
        // base^(e2/2) via polar form.
        let r = base.norm();
        let th = base.arg();
        let mag = r.powf(e2 / 2.0);
        Complex::new(mag * (th * e2 / 2.0).cos(), mag * (th * e2 / 2.0).sin())
    };
    let nfft = next_power_of_two(n + m - 1);
    let mut fa = vec![ZERO; nfft];
    for j in 0..n {
        let aj = {
            let r = a.norm();
            let th = a.arg();
            let mag = r.powf(-(j as f64));
            Complex::new(mag * (-th * j as f64).cos(), mag * (-th * j as f64).sin())
        };
        fa[j] = x[j] * aj * pow_half(w, (j * j) as f64);
    }
    let mut fb = vec![ZERO; nfft];
    for (k, slot) in fb.iter_mut().enumerate().take(m) {
        *slot = pow_half(w, -((k * k) as f64));
    }
    for j in 1..n {
        fb[nfft - j] = pow_half(w, -((j * j) as f64));
    }
    let sa = fft(&fa);
    let sb = fft(&fb);
    let prod: Vec<Complex> = sa.iter().zip(&sb).map(|(u, v)| *u * *v).collect();
    let conv = ifft(&prod);
    (0..m).map(|k| conv[k] * pow_half(w, (k * k) as f64)).collect()
}

/// Zoom FFT: m spectrum samples evenly spaced over [f_lo, f_hi] Hz.
#[must_use]
pub fn zoom_fft(x: &[f64], fs: f64, f_lo: f64, f_hi: f64, m: usize) -> Vec<Complex> {
    let xc: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let a = cis(TWO_PI * f_lo / fs);
    let step = if m > 1 { (f_hi - f_lo) / (m - 1) as f64 } else { 0.0 };
    let w = cis(-TWO_PI * step / fs);
    chirp_z(&xc, m, w, a)
}

/// Time-frequency reassigned spectrogram (Hann window): each bin's
/// energy is moved to its instantaneous time and frequency. Returns
/// (time s, frequency Hz, power) for every bin above −80 dB of the peak.
#[must_use]
pub fn reassigned_spectrogram(
    x: &[f64],
    fs: f64,
    n_fft: usize,
    hop: usize,
) -> Vec<(f64, f64, f64)> {
    let w = window(WindowKind::Hann, n_fft, true);
    // Time-weighted and derivative windows.
    let c = (n_fft as f64 - 1.0) / 2.0;
    let tw: Vec<f64> = w.iter().enumerate().map(|(i, &v)| (i as f64 - c) * v).collect();
    let mut dw = vec![0.0; n_fft];
    for i in 0..n_fft {
        let prev = if i > 0 { w[i - 1] } else { 0.0 };
        let next = if i + 1 < n_fft { w[i + 1] } else { 0.0 };
        dw[i] = 0.5 * (next - prev);
    }
    let hop_stft = |win: &[f64]| -> Vec<Vec<Complex>> {
        Stft::new(win.to_vec(), hop, n_fft).forward(x)
    };
    let s = hop_stft(&w);
    let st = hop_stft(&tw);
    let sd = hop_stft(&dw);
    let mut peak = 0.0_f64;
    for f in &s {
        for c in f {
            peak = peak.max(c.norm_sq());
        }
    }
    let floor = peak * 1e-8;
    let mut out = Vec::new();
    for (fi, frame) in s.iter().enumerate() {
        for (k, &v) in frame.iter().enumerate() {
            let p = v.norm_sq();
            if p <= floor || p == 0.0 {
                continue;
            }
            let ratio_t = st[fi][k] / v;
            let ratio_d = sd[fi][k] / v;
            let t_hat = (fi * hop) as f64 / fs + c / fs + ratio_t.re / fs;
            let f_hat = k as f64 * fs / n_fft as f64 - ratio_d.im * fs / TWO_PI;
            out.push((t_hat, f_hat, p));
        }
    }
    out
}

/// Constant-Q transform magnitudes: bins at fmin·2^(k/bins_per_octave),
/// each analyzed with its own Hann-windowed complex kernel whose length
/// keeps Q constant. Returns one vector of `n_bins` magnitudes per hop
/// of half the longest kernel.
#[must_use]
pub fn constant_q_transform(
    x: &[f64],
    fs: f64,
    fmin: f64,
    bins_per_octave: usize,
    n_bins: usize,
) -> Vec<Vec<f64>> {
    assert!(fmin > 0.0 && bins_per_octave > 0, "invalid CQT parameters");
    let q = 1.0 / (2.0_f64.powf(1.0 / bins_per_octave as f64) - 1.0);
    let lengths: Vec<usize> = (0..n_bins)
        .map(|k| {
            let f = fmin * 2.0_f64.powf(k as f64 / bins_per_octave as f64);
            ((q * fs / f).ceil() as usize).max(2).min(x.len().max(2))
        })
        .collect();
    let max_len = lengths.iter().copied().max().unwrap_or(2);
    let hop = (max_len / 2).max(1);
    if x.len() < max_len {
        return Vec::new();
    }
    let n_frames = 1 + (x.len() - max_len) / hop;
    (0..n_frames)
        .map(|fr| {
            let start = fr * hop;
            (0..n_bins)
                .map(|k| {
                    let nl = lengths[k];
                    let f = fmin * 2.0_f64.powf(k as f64 / bins_per_octave as f64);
                    let win = window(WindowKind::Hann, nl, true);
                    let wsum: f64 = win.iter().sum();
                    let mut acc = ZERO;
                    for i in 0..nl {
                        let ph = cis(-TWO_PI * f * i as f64 / fs);
                        acc = acc + ph * Complex::new(x[start + i] * win[i], 0.0);
                    }
                    acc.norm() / wsum
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_stft_roundtrip_hann_quarter_hop() {
        let n_fft = 64;
        let stft = Stft::new(window(WindowKind::Hann, n_fft, true), n_fft / 4, n_fft);
        assert!(stft.is_cola());
        let x: Vec<f64> = (0..1024)
            .map(|i| (i as f64 * 0.13).sin() + 0.4 * (i as f64 * 0.71).cos())
            .collect();
        let y = stft.inverse(&stft.forward(&x));
        // Interior samples (full window overlap) reconstruct exactly.
        for i in n_fft..y.len() - n_fft {
            assert!(approx(x[i], y[i], 1e-10), "at {i}: {} vs {}", x[i], y[i]);
        }
    }

    #[test]
    fn test_stft_frame_geometry() {
        let stft = Stft::new(vec![1.0; 8], 4, 8);
        let x = vec![0.0; 32];
        let frames = stft.forward(&x);
        assert_eq!(frames.len(), 7);
        assert_eq!(frames[0].len(), 5);
        assert_eq!(stft.freqs(8000.0).len(), 5);
        assert!(approx(stft.freqs(8000.0)[4], 4000.0, 1e-9));
        assert_eq!(stft.times(32, 1.0).len(), 7);
    }

    #[test]
    fn test_spectrogram_tone_peaks_at_bin() {
        let fs = 1000.0;
        let n_fft = 128;
        let f0 = 125.0; // bin 16
        let x: Vec<f64> = (0..2000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let (times, freqs, power) = spectrogram(&x, fs, n_fft, 32, WindowKind::Hann);
        assert!(!times.is_empty());
        for frame in &power {
            let (k, _) = argmax(frame).unwrap();
            assert!(approx(freqs[k], f0, fs / n_fft as f64), "peak at {}", freqs[k]);
        }
    }

    #[test]
    fn test_goertzel_matches_fft_bin() {
        let fs = 1024.0;
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (TWO_PI * 32.0 * i as f64 / fs).sin() + 0.5 * (TWO_PI * 100.0 * i as f64 / fs).cos())
            .collect();
        let spec = rfft(&x);
        // Bin 8 corresponds to 32 Hz (fs/n = 4 Hz per bin).
        let (mag, _) = goertzel(&x, 32.0, fs);
        assert!(approx(mag, spec[8].norm(), 1e-6 * spec[8].norm().max(1.0)));
        let (mag2, _) = goertzel(&x, 100.0, fs);
        assert!(approx(mag2, spec[25].norm(), 1e-6 * spec[25].norm().max(1.0)));
    }

    #[test]
    fn test_goertzel_bank_and_dtmf() {
        let fs = 8000.0;
        // '5' = 770 Hz + 1336 Hz.
        let x: Vec<f64> = (0..800)
            .map(|i| {
                let t = i as f64 / fs;
                (TWO_PI * 770.0 * t).sin() + (TWO_PI * 1336.0 * t).sin()
            })
            .collect();
        assert_eq!(dtmf_decode(&x, fs), Some('5'));
        // Silence decodes to nothing.
        assert_eq!(dtmf_decode(&vec![0.0; 800], fs), None);
    }

    #[test]
    fn test_chirp_z_matches_dft() {
        // CZT with a = 1, w = e^{-2πi/n}, m = n is the plain DFT.
        let n = 12;
        let x: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.5).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let w = cis(-TWO_PI / n as f64);
        let a = Complex::new(1.0, 0.0);
        let czt = chirp_z(&x, n, w, a);
        let dft = crate::transforms::fft::fft_any(&x);
        for (u, v) in czt.iter().zip(&dft) {
            assert!(approx(u.re, v.re, 1e-8) && approx(u.im, v.im, 1e-8));
        }
    }

    #[test]
    fn test_zoom_fft_locates_tone() {
        let fs = 1000.0;
        let f0 = 123.4;
        let x: Vec<f64> = (0..1000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let m = 201;
        let spec = zoom_fft(&x, fs, 100.0, 150.0, m);
        let mags: Vec<f64> = spec.iter().map(|c| c.norm()).collect();
        let (k, _) = argmax(&mags).unwrap();
        let f_est = 100.0 + 50.0 * k as f64 / (m - 1) as f64;
        assert!(approx(f_est, f0, 0.5), "estimated {f_est}");
    }

    #[test]
    fn test_mel_filterbank_shapes() {
        let fb = mel_filterbank(256, 8000.0, 20, 0.0, 4000.0);
        assert_eq!(fb.len(), 20);
        assert_eq!(fb[0].len(), 129);
        // Every filter has some mass and peaks at 1 (triangles).
        for filt in &fb {
            let peak = filt.iter().cloned().fold(0.0_f64, f64::max);
            assert!(peak > 0.5 && peak <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn test_mel_spectrogram_tone_band() {
        let fs = 8000.0;
        let f0 = 1000.0;
        let x: Vec<f64> = (0..4000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let mels = mel_spectrogram(&x, fs, 256, 128, 20, 0.0, 4000.0);
        assert!(!mels.is_empty());
        // The peak band center should be near 1 kHz in mel space.
        let frame = &mels[mels.len() / 2];
        let (band, _) = argmax(frame).unwrap();
        let mel_lo = hz_to_mel(0.0);
        let mel_hi = hz_to_mel(4000.0);
        let center = mel_to_hz(mel_lo + (mel_hi - mel_lo) * (band + 1) as f64 / 21.0);
        assert!((center - f0).abs() < 300.0, "band center {center}");
    }

    #[test]
    fn test_reassigned_spectrogram_tone() {
        let fs = 1000.0;
        let f0 = 130.0; // off-bin on purpose
        let x: Vec<f64> = (0..2000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let points = reassigned_spectrogram(&x, fs, 128, 32);
        assert!(!points.is_empty());
        // Power-weighted mean reassigned frequency should land on f0
        // much more precisely than the 7.8 Hz bin spacing.
        let mut wsum = 0.0;
        let mut fsum = 0.0;
        for &(_, f, p) in &points {
            wsum += p;
            fsum += f * p;
        }
        let f_est = fsum / wsum;
        assert!((f_est - f0).abs() < 1.0, "reassigned mean {f_est}");
    }

    #[test]
    fn test_cqt_octave_tone() {
        let fs = 8000.0;
        let f0 = 440.0;
        let x: Vec<f64> = (0..8000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let bins_per_octave = 12;
        let n_bins = 36; // 3 octaves from 220 Hz
        let frames = constant_q_transform(&x, fs, 220.0, bins_per_octave, n_bins);
        assert!(!frames.is_empty());
        let frame = &frames[frames.len() / 2];
        let (k, _) = argmax(frame).unwrap();
        // 440 = 220·2^(12/12): bin 12.
        assert!((k as i64 - 12).abs() <= 1, "CQT peak at bin {k}");
    }
}
