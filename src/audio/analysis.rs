//! Audio analysis: pitch detection (YIN, autocorrelation, cepstral, HPS,
//! McLeod), onset/tempo/beat tracking, MFCCs, LPC and formants, LSPs,
//! spectral descriptors, chroma/key/chord estimation, psychoacoustic
//! approximations, distortion metrics, room-acoustics measures from
//! impulse responses, DTW, and constellation fingerprinting.

use crate::transforms::dct::dct_ii;
use crate::transforms::fft::{irfft, rfft};
use crate::transforms::stft::mel_spectrogram;

const TWO_PI: f64 = 2.0 * crate::math::constants::PI;
const PI: f64 = crate::math::constants::PI;

// --- Small shared helpers ------------------------------------------------

fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 * (1.0 - (TWO_PI * i as f64 / n as f64).cos())).collect()
}

fn magnitude_frames(x: &[f64], n_fft: usize, hop: usize) -> Vec<Vec<f64>> {
    let w = hann(n_fft);
    let mut out = Vec::new();
    let mut start = 0;
    while start + n_fft <= x.len() {
        let seg: Vec<f64> = (0..n_fft).map(|i| x[start + i] * w[i]).collect();
        out.push(rfft(&seg).iter().map(|c| c.norm()).collect());
        start += hop;
    }
    out
}

/// Parabolic interpolation of a local extremum at index `i` of `y`;
/// returns the fractional offset in (-0.5, 0.5).
fn parabolic_offset(y: &[f64], i: usize) -> f64 {
    if i == 0 || i + 1 >= y.len() {
        return 0.0;
    }
    let denom = y[i - 1] - 2.0 * y[i] + y[i + 1];
    if denom.abs() < 1e-30 {
        0.0
    } else {
        (0.5 * (y[i - 1] - y[i + 1]) / denom).clamp(-0.5, 0.5)
    }
}

/// Raw (biased, un-normalized) autocorrelation of `x` computed with FFTs;
/// returns lags 0..x.len().
#[must_use]
pub fn autocorrelation_fft(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let m = crate::transforms::fft::next_power_of_two(2 * n);
    let mut padded = x.to_vec();
    padded.resize(m, 0.0);
    let spec = rfft(&padded);
    let power: Vec<crate::fractals::Complex> =
        spec.iter().map(|c| crate::fractals::Complex::new(c.norm_sq(), 0.0)).collect();
    let r = irfft(&power, m);
    r[..n].to_vec()
}

// --- Pitch detection -----------------------------------------------------

/// YIN pitch detector. Returns (frequency, confidence in 0..1) or `None`
/// when no lag drops below `threshold` (unvoiced).
#[must_use]
pub fn pitch_yin(x: &[f64], fs: f64, f_min: f64, f_max: f64, threshold: f64) -> Option<(f64, f64)> {
    let n = x.len();
    let w = n / 2;
    let tau_min = ((fs / f_max).floor() as usize).max(2);
    let tau_max = ((fs / f_min).ceil() as usize).min(w.saturating_sub(1));
    if tau_max <= tau_min {
        return None;
    }
    // Difference function d(τ) = e0 + eτ - 2 r(τ), with r by direct sum.
    let sq: Vec<f64> = x.iter().map(|v| v * v).collect();
    let mut cum = vec![0.0; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + sq[i];
    }
    // r over the first window only is approximated by full autocorr terms;
    // compute d directly for correctness.
    let mut d = vec![0.0; tau_max + 2];
    for (tau, dv) in d.iter_mut().enumerate().skip(1) {
        // e0 over [0,w), e_tau over [tau, tau+w).
        let e0 = cum[w];
        let et = cum[tau + w] - cum[tau];
        let mut rt = 0.0;
        for i in 0..w {
            rt += x[i] * x[i + tau];
        }
        *dv = e0 + et - 2.0 * rt;
    }
    // Cumulative-mean normalized difference.
    let mut dp = vec![1.0; tau_max + 2];
    let mut running = 0.0;
    for tau in 1..=tau_max + 1 {
        running += d[tau];
        dp[tau] = if running > 0.0 { d[tau] * tau as f64 / running } else { 1.0 };
    }
    // First dip under threshold, descended to its local minimum.
    let mut tau = tau_min;
    let mut best: Option<usize> = None;
    while tau <= tau_max {
        if dp[tau] < threshold {
            while tau < tau_max && dp[tau + 1] < dp[tau] {
                tau += 1;
            }
            best = Some(tau);
            break;
        }
        tau += 1;
    }
    let tau = best?;
    let offset = parabolic_offset(&dp, tau);
    let freq = fs / (tau as f64 + offset);
    Some((freq, (1.0 - dp[tau]).clamp(0.0, 1.0)))
}

/// Autocorrelation pitch: highest normalized-autocorrelation peak in the
/// lag range; `None` if the peak is weak (< 0.3).
#[must_use]
pub fn pitch_autocorrelation(x: &[f64], fs: f64, f_min: f64, f_max: f64) -> Option<f64> {
    let r = autocorrelation_fft(x);
    if r[0] <= 0.0 {
        return None;
    }
    let tau_min = ((fs / f_max).floor() as usize).max(2);
    let tau_max = ((fs / f_min).ceil() as usize).min(r.len() - 2);
    let norm: Vec<f64> = r.iter().map(|v| v / r[0]).collect();
    let mut best = tau_min;
    for tau in tau_min..=tau_max {
        if norm[tau] > norm[best] {
            best = tau;
        }
    }
    if norm[best] < 0.3 || norm[best] <= norm[best - 1].min(norm[best + 1]) - 1e-12 {
        return None;
    }
    let offset = parabolic_offset(&norm, best);
    Some(fs / (best as f64 + offset))
}

/// Cepstral pitch: peak of the real cepstrum in the expected quefrency
/// range.
#[must_use]
pub fn pitch_cepstral(x: &[f64], fs: f64, f_min: f64, f_max: f64) -> Option<f64> {
    let n = x.len();
    let w = hann(n);
    let xw: Vec<f64> = x.iter().zip(&w).map(|(a, b)| a * b).collect();
    let ceps = crate::transforms::spectral::cepstrum_real(&xw);
    let q_min = ((fs / f_max).floor() as usize).max(2);
    let q_max = ((fs / f_min).ceil() as usize).min(ceps.len() / 2);
    if q_max <= q_min {
        return None;
    }
    let (mut best, mut best_v) = (q_min, f64::NEG_INFINITY);
    for (q, &v) in ceps.iter().enumerate().take(q_max).skip(q_min) {
        if v > best_v {
            best_v = v;
            best = q;
        }
    }
    let mean = ceps[q_min..q_max].iter().sum::<f64>() / (q_max - q_min) as f64;
    let var = ceps[q_min..q_max].iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / (q_max - q_min) as f64;
    if best_v < mean + 2.0 * var.sqrt() {
        return None;
    }
    let offset = parabolic_offset(&ceps, best);
    Some(fs / (best as f64 + offset))
}

/// Harmonic product spectrum pitch estimate.
#[must_use]
pub fn pitch_hps(x: &[f64], fs: f64, n_harmonics: usize) -> Option<f64> {
    let n = x.len();
    let w = hann(n);
    let m = crate::transforms::fft::next_power_of_two(4 * n);
    let mut seg: Vec<f64> = x.iter().zip(&w).map(|(a, b)| a * b).collect();
    seg.resize(m, 0.0);
    let mag: Vec<f64> = rfft(&seg).iter().map(|c| c.norm()).collect();
    let df = fs / m as f64;
    let k_lo = (40.0 / df).ceil() as usize;
    let k_hi = mag.len() / n_harmonics.max(2);
    if k_hi <= k_lo {
        return None;
    }
    let hps: Vec<f64> = (0..k_hi)
        .map(|k| {
            if k < k_lo {
                0.0
            } else {
                (1..=n_harmonics).map(|h| mag[k * h].max(1e-12).ln()).sum::<f64>()
            }
        })
        .collect();
    let best = (k_lo..k_hi).max_by(|&a, &b| hps[a].partial_cmp(&hps[b]).unwrap())?;
    let offset = parabolic_offset(&hps, best);
    Some((best as f64 + offset) * df)
}

/// McLeod pitch method (NSDF). Returns (frequency, clarity).
#[must_use]
pub fn pitch_mpm(x: &[f64], fs: f64) -> Option<(f64, f64)> {
    let n = x.len();
    let r = autocorrelation_fft(x);
    let sq: Vec<f64> = x.iter().map(|v| v * v).collect();
    let mut cum = vec![0.0; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + sq[i];
    }
    let max_tau = n / 2;
    let nsdf: Vec<f64> = (0..max_tau)
        .map(|tau| {
            let m = cum[n - tau] + (cum[n] - cum[tau]);
            if m > 0.0 { 2.0 * r[tau] / m } else { 0.0 }
        })
        .collect();
    // Key maxima between positive-going zero crossings.
    let mut peaks: Vec<usize> = Vec::new();
    let mut i = 1;
    while i < max_tau && nsdf[i] > 0.0 {
        i += 1; // skip the lag-0 lobe
    }
    while i < max_tau {
        while i < max_tau && nsdf[i] <= 0.0 {
            i += 1;
        }
        let mut best = i;
        while i < max_tau && nsdf[i] > 0.0 {
            if nsdf[i] > nsdf[best] {
                best = i;
            }
            i += 1;
        }
        if best < max_tau && nsdf[best] > 0.0 {
            peaks.push(best);
        }
    }
    let global = peaks.iter().map(|&p| nsdf[p]).fold(f64::NEG_INFINITY, f64::max);
    if !global.is_finite() || global < 0.3 {
        return None;
    }
    let chosen = *peaks.iter().find(|&&p| nsdf[p] >= 0.8 * global)?;
    let offset = parabolic_offset(&nsdf, chosen);
    Some((fs / (chosen as f64 + offset), nsdf[chosen].clamp(0.0, 1.0)))
}

/// Frame-wise pitch detection method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchMethod {
    Yin,
    Autocorrelation,
    Cepstral,
    Hps,
    Mpm,
}

/// Frame-wise pitch track: (time s, f0) per hop, 2048-sample frames.
#[must_use]
pub fn pitch_track(x: &[f64], fs: f64, hop: usize, method: PitchMethod) -> Vec<(f64, Option<f64>)> {
    let frame = 2048.min(x.len());
    let mut out = Vec::new();
    let mut start = 0;
    while start + frame <= x.len() {
        let seg = &x[start..start + frame];
        let f = match method {
            PitchMethod::Yin => pitch_yin(seg, fs, 50.0, 2000.0, 0.2).map(|p| p.0),
            PitchMethod::Autocorrelation => pitch_autocorrelation(seg, fs, 50.0, 2000.0),
            PitchMethod::Cepstral => pitch_cepstral(seg, fs, 50.0, 2000.0),
            PitchMethod::Hps => pitch_hps(seg, fs, 4),
            PitchMethod::Mpm => pitch_mpm(seg, fs).map(|p| p.0),
        };
        out.push(((start as f64 + frame as f64 / 2.0) / fs, f));
        start += hop;
    }
    out
}

/// Segment a pitch track into notes: (start time, duration, MIDI note).
/// Runs of at least 3 voiced frames on the same rounded MIDI number
/// become one note.
#[must_use]
pub fn pitch_to_midi_track(track: &[(f64, Option<f64>)]) -> Vec<(f64, f64, u8)> {
    let dt = if track.len() > 1 { track[1].0 - track[0].0 } else { 0.0 };
    let midi: Vec<Option<i32>> = track
        .iter()
        .map(|(_, f)| {
            f.and_then(|f| {
                if f > 0.0 {
                    Some((69.0 + 12.0 * (f / 440.0).log2()).round() as i32)
                } else {
                    None
                }
            })
        })
        .collect();
    let mut notes = Vec::new();
    let mut run_start = 0;
    let mut i = 0;
    while i <= midi.len() {
        let boundary = i == midi.len() || midi[i] != midi[run_start];
        if boundary {
            if let Some(Some(m)) = midi.get(run_start).copied() {
                let len = i - run_start;
                if len >= 3 && (0..=127).contains(&m) {
                    notes.push((track[run_start].0, len as f64 * dt, m as u8));
                }
            }
            run_start = i;
        }
        i += 1;
    }
    notes
}

// --- Onsets, tempo, beats ------------------------------------------------

/// Spectral-flux onset strength envelope (one value per STFT frame).
#[must_use]
pub fn onset_strength(x: &[f64], _fs: f64, n_fft: usize, hop: usize) -> Vec<f64> {
    let mags = magnitude_frames(x, n_fft, hop);
    let mut out = vec![0.0; mags.len()];
    for t in 1..mags.len() {
        out[t] = mags[t]
            .iter()
            .zip(&mags[t - 1])
            .map(|(c, p)| (c - p).max(0.0))
            .sum();
    }
    out
}

/// Pick onsets from a strength envelope: local maxima above `threshold`
/// separated by at least `min_gap` frames.
#[must_use]
pub fn onset_detect(strength: &[f64], threshold: f64, min_gap: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for i in 1..strength.len().saturating_sub(1) {
        if strength[i] >= strength[i - 1]
            && strength[i] > strength[i + 1]
            && strength[i] > threshold
            && out.last().is_none_or(|&l| i - l >= min_gap)
        {
            out.push(i);
        }
    }
    out
}

/// High-frequency-content onset function: Σ k |X_k|² per frame.
#[must_use]
pub fn onset_hfc(x: &[f64], _fs: f64, n_fft: usize, hop: usize) -> Vec<f64> {
    magnitude_frames(x, n_fft, hop)
        .iter()
        .map(|m| m.iter().enumerate().map(|(k, v)| k as f64 * v * v).sum())
        .collect()
}

/// Complex-domain onset function: deviation of each frame from the
/// magnitude/phase prediction of the previous frames.
#[must_use]
pub fn onset_complex_domain(x: &[f64], _fs: f64, n_fft: usize, hop: usize) -> Vec<f64> {
    let w = hann(n_fft);
    let mut frames = Vec::new();
    let mut start = 0;
    while start + n_fft <= x.len() {
        let seg: Vec<f64> = (0..n_fft).map(|i| x[start + i] * w[i]).collect();
        frames.push(rfft(&seg));
        start += hop;
    }
    let mut out = vec![0.0; frames.len()];
    for t in 2..frames.len() {
        let mut dev = 0.0;
        for (k, cur) in frames[t].iter().enumerate() {
            let mag_prev = frames[t - 1][k].norm();
            let phi1 = frames[t - 1][k].arg();
            let phi2 = frames[t - 2][k].arg();
            let pred_phase = 2.0 * phi1 - phi2;
            let pred = crate::fractals::Complex::new(
                mag_prev * pred_phase.cos(),
                mag_prev * pred_phase.sin(),
            );
            let d = *cur - pred;
            dev += d.norm();
        }
        out[t] = dev;
    }
    out
}

/// Tempo (BPM) from an onset-strength envelope sampled at `fs` frames/s,
/// via the autocorrelation peak in the 40-240 BPM range (preferring the
/// shortest strong lag, i.e. the fastest consistent pulse).
#[must_use]
pub fn tempo_estimate(onsets: &[f64], fs: f64) -> f64 {
    let n = onsets.len();
    let mean = onsets.iter().sum::<f64>() / n.max(1) as f64;
    let centered: Vec<f64> = onsets.iter().map(|v| v - mean).collect();
    let r = autocorrelation_fft(&centered);
    let lag_min = ((fs * 60.0 / 240.0).round() as usize).max(1);
    let lag_max = ((fs * 60.0 / 40.0).round() as usize).min(n.saturating_sub(2));
    if lag_max <= lag_min {
        return 0.0;
    }
    let peak = r[lag_min..=lag_max].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut best = lag_min;
    for lag in lag_min..=lag_max {
        if r[lag] > r[lag - 1] && r[lag] >= r[lag + 1] && r[lag] >= 0.8 * peak {
            best = lag;
            break;
        }
    }
    if r[best] < 0.8 * peak {
        best = (lag_min..=lag_max).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap();
    }
    let offset = parabolic_offset(&r, best);
    60.0 * fs / (best as f64 + offset)
}

/// Beat tracking by dynamic programming (Ellis 2007): onset envelope,
/// global tempo, then a penalized best-predecessor recursion. Returns
/// beat times in seconds.
#[must_use]
pub fn beat_track(x: &[f64], fs: f64) -> Vec<f64> {
    let (n_fft, hop) = (1024, 256);
    let env = onset_strength(x, fs, n_fft, hop);
    if env.len() < 8 {
        return Vec::new();
    }
    let env_rate = fs / hop as f64;
    let bpm = tempo_estimate(&env, env_rate);
    if bpm <= 0.0 {
        return Vec::new();
    }
    let period = env_rate * 60.0 / bpm;
    let tightness = 100.0;
    let n = env.len();
    let mut score = env.clone();
    let mut back = vec![usize::MAX; n];
    for i in 0..n {
        let lo = (i as f64 - 2.0 * period).max(0.0) as usize;
        let hi = (i as f64 - 0.5 * period).max(0.0) as usize;
        let mut best = f64::NEG_INFINITY;
        let mut best_j = usize::MAX;
        for (j, &sj) in score.iter().enumerate().take(hi).skip(lo) {
            let interval = (i - j) as f64;
            let pen = tightness * (interval / period).ln().powi(2);
            let s = sj - pen;
            if s > best {
                best = s;
                best_j = j;
            }
        }
        if best_j != usize::MAX && best > 0.0 {
            score[i] += best;
            back[i] = best_j;
        }
    }
    // Backtrack from the best score in the last period.
    let tail_start = n.saturating_sub(period.ceil() as usize + 1);
    let mut cur = (tail_start..n)
        .max_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap())
        .unwrap_or(n - 1);
    let mut beats = vec![cur];
    while back[cur] != usize::MAX {
        cur = back[cur];
        beats.push(cur);
    }
    beats.reverse();
    beats.iter().map(|&i| i as f64 * hop as f64 / fs).collect()
}

// --- Cepstral / LPC features ---------------------------------------------

/// MFCCs: log-mel spectrogram followed by a DCT-II, keeping `n_mfcc`
/// coefficients per frame.
#[must_use]
pub fn mfcc(
    x: &[f64],
    fs: f64,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    n_mfcc: usize,
) -> Vec<Vec<f64>> {
    let mel = mel_spectrogram(x, fs, n_fft, hop, n_mels, 0.0, fs / 2.0);
    mel.iter()
        .map(|frame| {
            let logm: Vec<f64> = frame.iter().map(|v| (v + 1e-10).ln()).collect();
            let c = dct_ii(&logm);
            c[..n_mfcc.min(c.len())].to_vec()
        })
        .collect()
}

/// Regression delta features over ±`width` frames.
#[must_use]
pub fn delta_features(f: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    let t_max = f.len();
    let w = width.max(1);
    let denom: f64 = 2.0 * (1..=w).map(|k| (k * k) as f64).sum::<f64>();
    (0..t_max)
        .map(|t| {
            (0..f[t].len())
                .map(|d| {
                    (1..=w)
                        .map(|k| {
                            let hi = f[(t + k).min(t_max - 1)][d];
                            let lo = f[t.saturating_sub(k)][d];
                            k as f64 * (hi - lo)
                        })
                        .sum::<f64>()
                        / denom
                })
                .collect()
        })
        .collect()
}

/// Linear prediction by the autocorrelation method (Levinson-Durbin).
/// Returns the coefficients of A(z) = 1 + a₁z⁻¹ + … (length order+1,
/// leading 1) and the residual gain √E.
#[must_use]
pub fn lpc(x: &[f64], order: usize) -> (Vec<f64>, f64) {
    let n = x.len();
    let mut r = vec![0.0; order + 1];
    for (lag, rv) in r.iter_mut().enumerate() {
        *rv = (0..n - lag).map(|i| x[i] * x[i + lag]).sum();
    }
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut e = r[0];
    if e <= 0.0 {
        return (a, 0.0);
    }
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / e;
        let prev = a.clone();
        for j in 1..=i {
            a[j] = prev[j] + k * prev[i - j];
        }
        e *= 1.0 - k * k;
        if e <= 0.0 {
            break;
        }
    }
    (a, e.max(0.0).sqrt())
}

/// Formants (frequency, bandwidth in Hz) from LPC coefficients, via the
/// roots of A(z).
#[must_use]
pub fn lpc_to_formants(coeffs: &[f64], fs: f64) -> Vec<(f64, f64)> {
    let (_, poles, _) = crate::dsp::iir::tf_to_zpk(&[1.0], coeffs);
    let mut out: Vec<(f64, f64)> = poles
        .iter()
        .filter(|p| p.im > 1e-9)
        .map(|p| {
            let f = p.arg() * fs / TWO_PI;
            let bw = -p.norm().max(1e-12).ln() * fs / PI;
            (f, bw)
        })
        .filter(|&(f, bw)| f > 90.0 && f < fs / 2.0 - 90.0 && bw < 600.0)
        .collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

/// Frame-wise formant tracking (25 ms Hamming frames, pre-emphasis).
#[must_use]
pub fn formant_track(x: &[f64], fs: f64, order: usize, hop: usize) -> Vec<Vec<(f64, f64)>> {
    let frame = ((0.025 * fs) as usize).max(order * 4);
    let pre: Vec<f64> =
        std::iter::once(x[0]).chain(x.windows(2).map(|w| w[1] - 0.97 * w[0])).collect();
    let win: Vec<f64> = (0..frame)
        .map(|i| 0.54 - 0.46 * (TWO_PI * i as f64 / (frame - 1) as f64).cos())
        .collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start + frame <= pre.len() {
        let seg: Vec<f64> = (0..frame).map(|i| pre[start + i] * win[i]).collect();
        let (a, _) = lpc(&seg, order);
        out.push(lpc_to_formants(&a, fs));
        start += hop;
    }
    out
}

/// LPC envelope magnitude spectrum: gain/|A(e^{jω})| at `n` frequencies
/// from 0 to fs/2.
#[must_use]
pub fn lpc_spectrum(coeffs: &[f64], gain: f64, n: usize, fs: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let omega = PI * i as f64 / n as f64;
            let _ = fs;
            let (mut re, mut im) = (0.0, 0.0);
            for (k, &c) in coeffs.iter().enumerate() {
                re += c * (omega * k as f64).cos();
                im -= c * (omega * k as f64).sin();
            }
            gain / (re * re + im * im).sqrt().max(1e-30)
        })
        .collect()
}

fn poly_eval_on_circle(p: &[f64], omega: f64) -> f64 {
    // For a (anti)palindromic p of length n+1, e^{jnω/2}·P(e^{-jω}) is
    // real (or imaginary); return the rotated real part plus rotated
    // imaginary part, whichever carries the signal.
    let n = p.len() - 1;
    let (mut re, mut im) = (0.0, 0.0);
    for (k, &c) in p.iter().enumerate() {
        re += c * (omega * k as f64).cos();
        im -= c * (omega * k as f64).sin();
    }
    let half = 0.5 * n as f64 * omega;
    let (c, s) = (half.cos(), half.sin());
    // Rotate by e^{j n ω/2}.
    let rr = re * c - im * s;
    let ri = re * s + im * c;
    if rr.abs() >= ri.abs() { rr } else { ri }
}

fn unit_circle_roots(p: &[f64]) -> Vec<f64> {
    let steps = 4096;
    let mut roots = Vec::new();
    let mut prev_w = PI / steps as f64 * 0.5;
    let mut prev_v = poly_eval_on_circle(p, prev_w);
    for i in 1..steps {
        let w = PI * (i as f64 + 0.5) / steps as f64;
        let v = poly_eval_on_circle(p, w);
        if prev_v == 0.0 {
            roots.push(prev_w);
        } else if prev_v * v < 0.0 {
            let (mut lo, mut hi) = (prev_w, w);
            let (mut flo, _) = (prev_v, v);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let fm = poly_eval_on_circle(p, mid);
                if flo * fm <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
        prev_w = w;
        prev_v = v;
    }
    roots
}

/// Line spectral pairs (radian frequencies in (0, π), sorted) of an LPC
/// polynomial with leading 1. The order must be even.
#[must_use]
pub fn lpc_to_lsp(coeffs: &[f64]) -> Vec<f64> {
    let m = coeffs.len() - 1;
    assert!(m >= 2 && m.is_multiple_of(2), "LSP conversion requires an even LPC order");
    let mut p = vec![0.0; m + 2];
    let mut q = vec![0.0; m + 2];
    for i in 0..=m + 1 {
        let a_i = if i <= m { coeffs[i] } else { 0.0 };
        let a_rev = if m + 1 - i <= m { coeffs[m + 1 - i] } else { 0.0 };
        p[i] = a_i + a_rev;
        q[i] = a_i - a_rev;
    }
    // Remove trivial roots: P(-1) = 0, Q(1) = 0.
    let p_red = deflate(&p, -1.0);
    let q_red = deflate(&q, 1.0);
    let mut roots = unit_circle_roots(&p_red);
    roots.extend(unit_circle_roots(&q_red));
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots.truncate(m);
    roots
}

fn deflate(p: &[f64], root: f64) -> Vec<f64> {
    // Synthetic division of p (ascending powers of z⁻¹) by (1 - root z⁻¹).
    let mut out = vec![0.0; p.len() - 1];
    let mut carry = 0.0;
    for i in 0..p.len() - 1 {
        out[i] = p[i] + root * carry;
        carry = out[i];
    }
    out
}

/// Reconstruct LPC coefficients (leading 1) from line spectral pairs.
#[must_use]
pub fn lsp_to_lpc(lsp: &[f64]) -> Vec<f64> {
    let m = lsp.len();
    assert!(m >= 2 && m.is_multiple_of(2), "even LSP count required");
    // Sorted ascending, the LSPs alternate starting with a P root:
    // ω_p1 < ω_q1 < ω_p2 < … (Q owns the trivial root at ω = 0).
    let mut p_poly = vec![1.0, 1.0]; // (1 + z⁻¹) trivial factor of P
    let mut q_poly = vec![1.0, -1.0]; // (1 - z⁻¹) trivial factor of Q
    for (idx, &w) in lsp.iter().enumerate() {
        let quad = [1.0, -2.0 * w.cos(), 1.0];
        let target = if idx % 2 == 0 { &mut p_poly } else { &mut q_poly };
        *target = conv(target, &quad);
    }
    let mut a = vec![0.0; m + 1];
    for (i, av) in a.iter_mut().enumerate() {
        *av = 0.5 * (p_poly[i] + q_poly[i]);
    }
    a
}

fn conv(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

// --- Spectral shape descriptors ------------------------------------------

/// Amplitude-weighted mean frequency.
#[must_use]
pub fn spectral_centroid(mag: &[f64], freqs: &[f64]) -> f64 {
    let total: f64 = mag.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    mag.iter().zip(freqs).map(|(m, f)| m * f).sum::<f64>() / total
}

/// Standard deviation of the spectral distribution.
#[must_use]
pub fn spectral_spread(mag: &[f64], freqs: &[f64]) -> f64 {
    let c = spectral_centroid(mag, freqs);
    let total: f64 = mag.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    (mag.iter().zip(freqs).map(|(m, f)| m * (f - c).powi(2)).sum::<f64>() / total).sqrt()
}

/// Third standardized moment of the spectral distribution.
#[must_use]
pub fn spectral_skewness(mag: &[f64], freqs: &[f64]) -> f64 {
    let c = spectral_centroid(mag, freqs);
    let s = spectral_spread(mag, freqs);
    let total: f64 = mag.iter().sum();
    if total <= 0.0 || s <= 0.0 {
        return 0.0;
    }
    mag.iter().zip(freqs).map(|(m, f)| m * (f - c).powi(3)).sum::<f64>() / (total * s.powi(3))
}

/// Fourth standardized moment of the spectral distribution.
#[must_use]
pub fn spectral_kurtosis(mag: &[f64], freqs: &[f64]) -> f64 {
    let c = spectral_centroid(mag, freqs);
    let s = spectral_spread(mag, freqs);
    let total: f64 = mag.iter().sum();
    if total <= 0.0 || s <= 0.0 {
        return 0.0;
    }
    mag.iter().zip(freqs).map(|(m, f)| m * (f - c).powi(4)).sum::<f64>() / (total * s.powi(4))
}

/// Frequency below which `pct` (0..1) of the spectral energy lies.
#[must_use]
pub fn spectral_rolloff(mag: &[f64], freqs: &[f64], pct: f64) -> f64 {
    let total: f64 = mag.iter().map(|m| m * m).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for (m, &f) in mag.iter().zip(freqs) {
        acc += m * m;
        if acc >= pct * total {
            return f;
        }
    }
    *freqs.last().unwrap_or(&0.0)
}

/// Half-wave rectified spectral flux between consecutive magnitude
/// frames.
#[must_use]
pub fn spectral_flux(prev: &[f64], cur: &[f64]) -> f64 {
    cur.iter().zip(prev).map(|(c, p)| (c - p).max(0.0)).sum()
}

/// Geometric-to-arithmetic mean ratio (1 = white, 0 = tonal).
#[must_use]
pub fn spectral_flatness_mag(mag: &[f64]) -> f64 {
    let n = mag.len() as f64;
    let am: f64 = mag.iter().sum::<f64>() / n;
    if am <= 0.0 {
        return 0.0;
    }
    let gm = (mag.iter().map(|m| (m + 1e-30).ln()).sum::<f64>() / n).exp();
    gm / am
}

/// Peak-to-mean spectral ratio.
#[must_use]
pub fn spectral_crest(mag: &[f64]) -> f64 {
    let am: f64 = mag.iter().sum::<f64>() / mag.len() as f64;
    if am <= 0.0 {
        return 0.0;
    }
    mag.iter().cloned().fold(0.0_f64, f64::max) / am
}

/// Linear-regression slope of magnitude vs frequency.
#[must_use]
pub fn spectral_slope(mag: &[f64], freqs: &[f64]) -> f64 {
    let n = mag.len() as f64;
    let fm = freqs.iter().sum::<f64>() / n;
    let mm = mag.iter().sum::<f64>() / n;
    let num: f64 = mag.iter().zip(freqs).map(|(m, f)| (f - fm) * (m - mm)).sum();
    let den: f64 = freqs.iter().map(|f| (f - fm).powi(2)).sum();
    if den > 0.0 { num / den } else { 0.0 }
}

/// Spectral decrease (perceptual measure of how fast magnitude falls off
/// with bin index).
#[must_use]
pub fn spectral_decrease(mag: &[f64]) -> f64 {
    let denom: f64 = mag.iter().skip(1).sum();
    if denom <= 0.0 {
        return 0.0;
    }
    mag.iter()
        .enumerate()
        .skip(1)
        .map(|(k, m)| (m - mag[0]) / k as f64)
        .sum::<f64>()
        / denom
}

/// Normalized Shannon entropy of the magnitude distribution (0..1).
#[must_use]
pub fn spectral_entropy_mag(mag: &[f64]) -> f64 {
    let total: f64 = mag.iter().sum();
    if total <= 0.0 || mag.len() < 2 {
        return 0.0;
    }
    let h: f64 = mag
        .iter()
        .filter(|&&m| m > 0.0)
        .map(|m| {
            let p = m / total;
            -p * p.ln()
        })
        .sum();
    h / (mag.len() as f64).ln()
}

/// One frame of spectral descriptors.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectralFeatures {
    pub centroid: f64,
    pub spread: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub rolloff85: f64,
    pub flux: f64,
    pub flatness: f64,
    pub crest: f64,
    pub slope: f64,
    pub decrease: f64,
    pub entropy: f64,
}

/// Frame-wise spectral descriptor track.
#[must_use]
pub fn spectral_features_track(x: &[f64], fs: f64, n_fft: usize, hop: usize) -> Vec<SpectralFeatures> {
    let mags = magnitude_frames(x, n_fft, hop);
    let freqs: Vec<f64> = (0..n_fft / 2 + 1).map(|k| k as f64 * fs / n_fft as f64).collect();
    let mut out = Vec::with_capacity(mags.len());
    for t in 0..mags.len() {
        let m = &mags[t];
        out.push(SpectralFeatures {
            centroid: spectral_centroid(m, &freqs),
            spread: spectral_spread(m, &freqs),
            skewness: spectral_skewness(m, &freqs),
            kurtosis: spectral_kurtosis(m, &freqs),
            rolloff85: spectral_rolloff(m, &freqs, 0.85),
            flux: if t > 0 { spectral_flux(&mags[t - 1], m) } else { 0.0 },
            flatness: spectral_flatness_mag(m),
            crest: spectral_crest(m),
            slope: spectral_slope(m, &freqs),
            decrease: spectral_decrease(m),
            entropy: spectral_entropy_mag(m),
        });
    }
    out
}

/// Frame-wise zero-crossing rate (fraction of adjacent sample pairs that
/// change sign).
#[must_use]
pub fn zero_crossing_rate(x: &[f64], frame: usize, hop: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut start = 0;
    while start + frame <= x.len() {
        let seg = &x[start..start + frame];
        let zc = seg.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
        out.push(zc as f64 / (frame - 1) as f64);
        start += hop;
    }
    out
}

/// Harmonic-to-noise ratio (dB) from the normalized autocorrelation at
/// the period of `f0`.
#[must_use]
pub fn harmonic_to_noise_ratio(x: &[f64], fs: f64, f0: f64) -> f64 {
    let r = autocorrelation_fft(x);
    if r[0] <= 0.0 {
        return 0.0;
    }
    let lag0 = fs / f0;
    let lo = ((lag0 * 0.9) as usize).max(1);
    let hi = ((lag0 * 1.1).ceil() as usize).min(r.len() - 1);
    let mut best = lo;
    for lag in lo..=hi {
        if r[lag] > r[best] {
            best = lag;
        }
    }
    // Unbiased normalization at this lag.
    let n = x.len() as f64;
    let rn = (r[best] / (n - best as f64) * n / r[0]).clamp(1e-6, 1.0 - 1e-6);
    10.0 * (rn / (1.0 - rn)).log10()
}

/// Piano-style inharmonicity coefficient B fitted from the measured
/// partial frequencies: f_k ≈ k f0 √(1 + B k²).
#[must_use]
pub fn inharmonicity_measure(x: &[f64], fs: f64, f0: f64) -> f64 {
    let n = x.len();
    let w = hann(n);
    let m = crate::transforms::fft::next_power_of_two(4 * n);
    let mut seg: Vec<f64> = x.iter().zip(&w).map(|(a, b)| a * b).collect();
    seg.resize(m, 0.0);
    let mag: Vec<f64> = rfft(&seg).iter().map(|c| c.norm()).collect();
    let df = fs / m as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for k in 1..=6_usize {
        let center = k as f64 * f0;
        if center * 1.1 >= fs / 2.0 {
            break;
        }
        let lo = ((center * 0.93 / df) as usize).max(1);
        let hi = ((center * 1.07 / df) as usize).min(mag.len() - 2);
        if hi <= lo {
            break;
        }
        let peak = (lo..=hi).max_by(|&a, &b| mag[a].partial_cmp(&mag[b]).unwrap()).unwrap();
        let fk = (peak as f64 + parabolic_offset(&mag, peak)) * df;
        let y = (fk / (k as f64 * f0)).powi(2) - 1.0;
        let k2 = (k * k) as f64;
        num += y * k2;
        den += k2 * k2;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

// --- Chroma, key, chords -------------------------------------------------

/// Frame-wise 12-bin chroma (C, C#, …, B), energy-normalized per frame.
#[must_use]
pub fn chroma(x: &[f64], fs: f64, n_fft: usize, hop: usize) -> Vec<[f64; 12]> {
    let mags = magnitude_frames(x, n_fft, hop);
    let df = fs / n_fft as f64;
    mags.iter()
        .map(|m| {
            let mut c = [0.0_f64; 12];
            for (k, v) in m.iter().enumerate().skip(1) {
                let f = k as f64 * df;
                if !(27.5..5000.0).contains(&f) {
                    continue;
                }
                let pc_a = (12.0 * (f / 440.0).log2()).round() as i64;
                let pc = ((pc_a + 9).rem_euclid(12)) as usize;
                c[pc] += v * v;
            }
            let total: f64 = c.iter().sum();
            if total > 0.0 {
                c.iter_mut().for_each(|v| *v /= total);
            }
            c
        })
        .collect()
}

const KRUMHANSL_MAJOR: [f64; 12] =
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KRUMHANSL_MINOR: [f64; 12] =
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

fn correlate12(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let ma = a.iter().sum::<f64>() / 12.0;
    let mb = b.iter().sum::<f64>() / 12.0;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..12 {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma).powi(2);
        db += (b[i] - mb).powi(2);
    }
    if da > 0.0 && db > 0.0 { num / (da * db).sqrt() } else { 0.0 }
}

/// Krumhansl-Schmuckler key estimate from a chroma track:
/// (tonic pitch class with C = 0, is_major).
#[must_use]
pub fn key_estimate(chroma_track: &[[f64; 12]]) -> (u8, bool) {
    let mut avg = [0.0_f64; 12];
    for c in chroma_track {
        for i in 0..12 {
            avg[i] += c[i];
        }
    }
    let mut best = (0_u8, true);
    let mut best_r = f64::NEG_INFINITY;
    for root in 0..12_usize {
        for (profile, major) in [(&KRUMHANSL_MAJOR, true), (&KRUMHANSL_MINOR, false)] {
            let mut rotated = [0.0_f64; 12];
            for i in 0..12 {
                rotated[(i + root) % 12] = profile[i];
            }
            let r = correlate12(&avg, &rotated);
            if r > best_r {
                best_r = r;
                best = (root as u8, major);
            }
        }
    }
    best
}

/// Template chord match on one chroma frame: returns e.g. "C", "Am",
/// "Bdim", "Faug".
#[must_use]
pub fn chord_estimate(chroma_frame: &[f64; 12]) -> String {
    const NAMES: [&str; 12] =
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let qualities: [(&str, [usize; 3]); 4] =
        [("", [0, 4, 7]), ("m", [0, 3, 7]), ("dim", [0, 3, 6]), ("aug", [0, 4, 8])];
    let mut best = String::new();
    let mut best_score = f64::NEG_INFINITY;
    for root in 0..12_usize {
        for (suffix, tones) in &qualities {
            let score: f64 = tones.iter().map(|&t| chroma_frame[(root + t) % 12]).sum();
            if score > best_score {
                best_score = score;
                best = format!("{}{}", NAMES[root], suffix);
            }
        }
    }
    best
}

// --- Psychoacoustic approximations ---------------------------------------

/// Approximate loudness in sones (Zwicker-style power law on the overall
/// level, full scale taken as 94 dB SPL).
#[must_use]
pub fn loudness_sone(x: &[f64], _fs: f64) -> f64 {
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt();
    if rms <= 0.0 {
        return 0.0;
    }
    let phon = 94.0 + 20.0 * rms.log10();
    if phon >= 40.0 {
        2.0_f64.powf((phon - 40.0) / 10.0)
    } else {
        (phon.max(0.0) / 40.0).powf(2.642)
    }
}

fn bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0).powi(2)).atan()
}

/// Approximate sharpness in acum (Bark-weighted specific-loudness
/// centroid, von Bismarck weighting).
#[must_use]
pub fn sharpness(x: &[f64], fs: f64) -> f64 {
    let n = 4096.min(x.len());
    let w = hann(n);
    let seg: Vec<f64> = x[..n].iter().zip(&w).map(|(a, b)| a * b).collect();
    let mag: Vec<f64> = rfft(&seg).iter().map(|c| c.norm()).collect();
    let df = fs / n as f64;
    let mut band_e = [0.0_f64; 24];
    for (k, m) in mag.iter().enumerate().skip(1) {
        let z = bark(k as f64 * df);
        let b = (z as usize).min(23);
        band_e[b] += m * m;
    }
    let (mut num, mut den) = (0.0, 0.0);
    for (b, &e) in band_e.iter().enumerate() {
        let nprime = e.powf(0.23);
        let z = b as f64 + 0.5;
        let g = if z < 16.0 { 1.0 } else { 0.066 * (0.171 * z).exp() };
        num += nprime * g * z;
        den += nprime;
    }
    if den > 0.0 { 0.11 * num / den } else { 0.0 }
}

fn envelope_fluctuation_energy(x: &[f64], fs: f64, f_lo: f64, f_hi: f64) -> f64 {
    // RMS envelope at ~1 kHz frame rate, then band energy fraction of its
    // fluctuation spectrum.
    let block = ((fs / 1000.0) as usize).max(1);
    let env: Vec<f64> = x
        .chunks(block)
        .map(|c| (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    if env.len() < 16 {
        return 0.0;
    }
    let mean = env.iter().sum::<f64>() / env.len() as f64;
    let centered: Vec<f64> = env.iter().map(|v| v - mean).collect();
    let mag: Vec<f64> = rfft(&centered).iter().map(|c| c.norm_sq()).collect();
    let env_fs = fs / block as f64;
    let df = env_fs / centered.len() as f64;
    let total: f64 = mag.iter().skip(1).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let band: f64 = mag
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(k, _)| {
            let f = *k as f64 * df;
            f >= f_lo && f <= f_hi
        })
        .map(|(_, v)| v)
        .sum();
    band / total
}

/// Approximate roughness: fraction of envelope fluctuation energy in the
/// 20-300 Hz modulation range.
#[must_use]
pub fn roughness(x: &[f64], fs: f64) -> f64 {
    envelope_fluctuation_energy(x, fs, 20.0, 300.0)
}

/// Approximate fluctuation strength: fraction of envelope fluctuation
/// energy in the 1-10 Hz modulation range (maximal near 4 Hz).
#[must_use]
pub fn fluctuation_strength(x: &[f64], fs: f64) -> f64 {
    envelope_fluctuation_energy(x, fs, 1.0, 10.0)
}

// --- Segmentation and quality metrics ------------------------------------

/// Silent regions as (start, end) sample ranges: block RMS below
/// `threshold_db` (dBFS) for at least `min_len` samples.
#[must_use]
pub fn silence_detect(x: &[f64], threshold_db: f64, min_len: usize) -> Vec<(usize, usize)> {
    let block = 256;
    let thresh = 10.0_f64.powf(threshold_db / 20.0);
    let quiet: Vec<bool> = x
        .chunks(block)
        .map(|c| (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt() < thresh)
        .collect();
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &q) in quiet.iter().enumerate() {
        match (q, start) {
            (true, None) => start = Some(i * block),
            (false, Some(s)) => {
                let end = i * block;
                if end - s >= min_len {
                    out.push((s, end));
                }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        if x.len() - s >= min_len {
            out.push((s, x.len()));
        }
    }
    out
}

/// Transient (attack) sample positions from a high-frequency-content
/// envelope with an adaptive threshold.
#[must_use]
pub fn transient_detect(x: &[f64], fs: f64) -> Vec<usize> {
    let (n_fft, hop) = (256, 64);
    let h = onset_hfc(x, fs, n_fft, hop);
    if h.len() < 3 {
        return Vec::new();
    }
    let mut d: Vec<f64> = vec![0.0; h.len()];
    for i in 1..h.len() {
        d[i] = (h[i] - h[i - 1]).max(0.0);
    }
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    let std = (d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d.len() as f64).sqrt();
    onset_detect(&d, mean + 2.0 * std, (0.02 * fs / hop as f64) as usize)
        .iter()
        .map(|&i| i * hop)
        .collect()
}

/// SNR (dB) of a signal given a noise-only reference segment.
#[must_use]
pub fn estimate_snr(x: &[f64], noise_segment: &[f64]) -> f64 {
    let px = x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64;
    let pn = noise_segment.iter().map(|v| v * v).sum::<f64>() / noise_segment.len().max(1) as f64;
    if pn <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * ((px - pn).max(1e-30) / pn).log10()
}

fn fundamental_and_total_power(x: &[f64], fs: f64, f0: f64) -> (f64, f64) {
    let n = x.len();
    let w = hann(n);
    let seg: Vec<f64> = x.iter().zip(&w).map(|(a, b)| a * b).collect();
    let p: Vec<f64> = rfft(&seg).iter().map(|c| c.norm_sq()).collect();
    let k0 = (f0 * n as f64 / fs).round() as usize;
    let lo = k0.saturating_sub(3);
    let hi = (k0 + 3).min(p.len() - 1);
    let fund: f64 = p[lo..=hi].iter().sum();
    // Exclude DC leakage.
    let total: f64 = p.iter().skip(3).sum();
    (fund, total)
}

/// THD+N as a linear ratio: √((P_total − P_fund)/P_fund).
#[must_use]
pub fn thd_n(x: &[f64], fs: f64, f0: f64) -> f64 {
    let (fund, total) = fundamental_and_total_power(x, fs, f0);
    if fund <= 0.0 {
        return f64::INFINITY;
    }
    ((total - fund).max(0.0) / fund).sqrt()
}

/// SINAD in dB: 10 log₁₀(P_fund / (P_total − P_fund)).
#[must_use]
pub fn sinad(x: &[f64], fs: f64, f0: f64) -> f64 {
    let (fund, total) = fundamental_and_total_power(x, fs, f0);
    let rest = (total - fund).max(1e-30);
    10.0 * (fund / rest).log10()
}

/// Effective number of bits from a SINAD measurement (dB).
#[must_use]
pub fn enob(sinad_db: f64) -> f64 {
    (sinad_db - 1.76) / 6.02
}

// --- Room acoustics from impulse responses -------------------------------

/// Deconvolve a Farina sweep measurement: convolve the recording with the
/// inverse sweep and align so the direct impulse response starts at 0.
#[must_use]
pub fn impulse_response_from_sweep(recorded: &[f64], inverse_sweep: &[f64]) -> Vec<f64> {
    let full = crate::audio::effects::convolution_reverb(recorded, inverse_sweep);
    full[inverse_sweep.len() - 1..].to_vec()
}

fn schroeder_db(ir: &[f64]) -> Vec<f64> {
    let mut edc: Vec<f64> = ir.iter().rev().map(|v| v * v).collect();
    for i in 1..edc.len() {
        edc[i] += edc[i - 1];
    }
    edc.reverse();
    let e0 = edc[0].max(1e-300);
    edc.iter().map(|e| 10.0 * (e / e0).max(1e-300).log10()).collect()
}

fn time_at_db(db: &[f64], level: f64, fs: f64) -> f64 {
    for (i, &v) in db.iter().enumerate() {
        if v <= level {
            return i as f64 / fs;
        }
    }
    db.len() as f64 / fs
}

/// RT60 via Schroeder backward integration, extrapolated from the
/// −5..−25 dB decay slope.
#[must_use]
pub fn rt60_from_ir(ir: &[f64], fs: f64) -> f64 {
    let db = schroeder_db(ir);
    let t5 = time_at_db(&db, -5.0, fs);
    let t25 = time_at_db(&db, -25.0, fs);
    3.0 * (t25 - t5)
}

/// Early decay time: the 0..−10 dB slope extrapolated to 60 dB.
#[must_use]
pub fn edt_from_ir(ir: &[f64], fs: f64) -> f64 {
    let db = schroeder_db(ir);
    6.0 * time_at_db(&db, -10.0, fs)
}

fn early_late_ratio(ir: &[f64], fs: f64, split_ms: f64) -> (f64, f64) {
    let split = ((split_ms * 1e-3 * fs) as usize).min(ir.len());
    let early: f64 = ir[..split].iter().map(|v| v * v).sum();
    let late: f64 = ir[split..].iter().map(|v| v * v).sum();
    (early, late)
}

/// Clarity C50 (dB): early (< 50 ms) to late energy ratio.
#[must_use]
pub fn c50(ir: &[f64], fs: f64) -> f64 {
    let (e, l) = early_late_ratio(ir, fs, 50.0);
    10.0 * (e / l.max(1e-300)).log10()
}

/// Clarity C80 (dB): early (< 80 ms) to late energy ratio.
#[must_use]
pub fn c80(ir: &[f64], fs: f64) -> f64 {
    let (e, l) = early_late_ratio(ir, fs, 80.0);
    10.0 * (e / l.max(1e-300)).log10()
}

/// Definition D50: fraction of energy arriving within 50 ms.
#[must_use]
pub fn d50(ir: &[f64], fs: f64) -> f64 {
    let (e, l) = early_late_ratio(ir, fs, 50.0);
    e / (e + l).max(1e-300)
}

/// Single-band STI approximation from the impulse response's modulation
/// transfer function at the 14 standard modulation frequencies.
#[must_use]
pub fn sti_approx(ir: &[f64], fs: f64) -> f64 {
    const MOD_FREQS: [f64; 14] =
        [0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5];
    let total: f64 = ir.iter().map(|v| v * v).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut sum_ti = 0.0;
    for &fm in &MOD_FREQS {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &v) in ir.iter().enumerate() {
            let ph = TWO_PI * fm * i as f64 / fs;
            re += v * v * ph.cos();
            im -= v * v * ph.sin();
        }
        let m = ((re * re + im * im).sqrt() / total).clamp(1e-6, 1.0 - 1e-6);
        let snr = (10.0 * (m / (1.0 - m)).log10()).clamp(-15.0, 15.0);
        sum_ti += (snr + 15.0) / 30.0;
    }
    sum_ti / MOD_FREQS.len() as f64
}

// --- Generic utilities ---------------------------------------------------

/// Local maxima above `threshold`, greedily thinned so accepted peaks are
/// at least `min_distance` apart (strongest first). Returns sorted
/// indices.
#[must_use]
pub fn peak_pick(x: &[f64], threshold: f64, min_distance: usize) -> Vec<usize> {
    let mut candidates: Vec<usize> = (1..x.len().saturating_sub(1))
        .filter(|&i| x[i] >= x[i - 1] && x[i] > x[i + 1] && x[i] > threshold)
        .collect();
    candidates.sort_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap());
    let mut kept: Vec<usize> = Vec::new();
    for c in candidates {
        if kept.iter().all(|&k| k.abs_diff(c) >= min_distance) {
            kept.push(c);
        }
    }
    kept.sort_unstable();
    kept
}

/// Dynamic time warping between two feature sequences (Euclidean local
/// cost). Returns (total cost, warping path from (0,0) to (n-1,m-1)).
#[must_use]
pub fn dynamic_time_warping(a: &[Vec<f64>], b: &[Vec<f64>]) -> (f64, Vec<(usize, usize)>) {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return (0.0, Vec::new());
    }
    let dist = |i: usize, j: usize| -> f64 {
        a[i].iter().zip(&b[j]).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    };
    let mut d = vec![vec![f64::INFINITY; m]; n];
    d[0][0] = dist(0, 0);
    for i in 0..n {
        for j in 0..m {
            if i == 0 && j == 0 {
                continue;
            }
            let mut best = f64::INFINITY;
            if i > 0 {
                best = best.min(d[i - 1][j]);
            }
            if j > 0 {
                best = best.min(d[i][j - 1]);
            }
            if i > 0 && j > 0 {
                best = best.min(d[i - 1][j - 1]);
            }
            d[i][j] = dist(i, j) + best;
        }
    }
    let mut path = vec![(n - 1, m - 1)];
    let (mut i, mut j) = (n - 1, m - 1);
    while i > 0 || j > 0 {
        let (pi, pj) = if i == 0 {
            (0, j - 1)
        } else if j == 0 {
            (i - 1, 0)
        } else {
            let diag = d[i - 1][j - 1];
            let up = d[i - 1][j];
            let left = d[i][j - 1];
            if diag <= up && diag <= left {
                (i - 1, j - 1)
            } else if up <= left {
                (i - 1, j)
            } else {
                (i, j - 1)
            }
        };
        path.push((pi, pj));
        i = pi;
        j = pj;
    }
    path.reverse();
    (d[n - 1][m - 1], path)
}

/// Shazam-style constellation fingerprint: spectrogram peaks paired into
/// (f_anchor, f_target, Δt) hashes.
#[must_use]
pub fn audio_fingerprint(x: &[f64], _fs: f64) -> Vec<u32> {
    let (n_fft, hop) = (1024, 256);
    let mags = magnitude_frames(x, n_fft, hop);
    // Local peaks per frame: top bins that are 3x the frame mean.
    let mut constellation: Vec<(usize, usize)> = Vec::new(); // (frame, bin)
    for (t, m) in mags.iter().enumerate() {
        let mean = m.iter().sum::<f64>() / m.len() as f64;
        let mut peaks: Vec<usize> = (2..m.len() - 2)
            .filter(|&k| {
                m[k] > 3.0 * mean
                    && m[k] >= m[k - 1]
                    && m[k] > m[k + 1]
                    && m[k] >= m[k - 2]
                    && m[k] > m[k + 2]
            })
            .collect();
        peaks.sort_by(|&a, &b| m[b].partial_cmp(&m[a]).unwrap());
        peaks.truncate(5);
        for p in peaks {
            constellation.push((t, p));
        }
    }
    constellation.sort_unstable();
    let mut hashes = Vec::new();
    for (i, &(t1, f1)) in constellation.iter().enumerate() {
        let mut paired = 0;
        for &(t2, f2) in &constellation[i + 1..] {
            let dt = t2 - t1;
            if dt == 0 {
                continue;
            }
            if dt > 32 || paired >= 3 {
                break;
            }
            hashes.push(((f1 as u32 & 0x3FF) << 22) | ((f2 as u32 & 0x3FF) << 12) | (dt as u32 & 0xFFF));
            paired += 1;
        }
    }
    hashes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn sine(f: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (TWO_PI * f * i as f64 / fs).sin()).collect()
    }

    #[test]
    fn test_pitch_methods_on_sine() {
        let fs = 48000.0;
        let x = sine(330.0, fs, 4096);
        let (fy, conf) = pitch_yin(&x, fs, 50.0, 2000.0, 0.2).unwrap();
        assert!((fy - 330.0).abs() < 1.0, "yin {fy}");
        assert!(conf > 0.9);
        let fa = pitch_autocorrelation(&x, fs, 50.0, 2000.0).unwrap();
        assert!((fa - 330.0).abs() < 2.0, "acf {fa}");
        // Cepstral pitch needs a harmonic-rich signal (a lone sinusoid has
        // no rahmonic peak).
        let rich: Vec<f64> = (0..4096)
            .map(|i| {
                (1..=6)
                    .map(|k| (TWO_PI * k as f64 * 330.0 * i as f64 / fs).sin() / k as f64)
                    .sum()
            })
            .collect();
        let fc = pitch_cepstral(&rich, fs, 50.0, 2000.0).unwrap();
        assert!((fc / 330.0 - 1.0).abs() < 0.02, "cepstral {fc}");
        let fh = pitch_hps(&rich, fs, 3).unwrap();
        assert!((fh / 330.0 - 1.0).abs() < 0.01, "hps {fh}");
        let (fm, clarity) = pitch_mpm(&x, fs).unwrap();
        assert!((fm - 330.0).abs() < 1.0, "mpm {fm}");
        assert!(clarity > 0.9);
        // Unvoiced: white noise mostly rejected by YIN.
        let mut rng = Rng::new(7);
        let noise: Vec<f64> = (0..4096).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        assert!(pitch_yin(&noise, fs, 50.0, 2000.0, 0.1).is_none());
    }

    #[test]
    fn test_yin_tracks_vibrato() {
        let fs = 48000.0;
        let (f0, depth, rate) = (220.0, 2.0, 3.0);
        let n = 48000;
        // Phase-integrated vibrato tone.
        let mut phase = 0.0;
        let mut inst = Vec::with_capacity(n);
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let f = f0 + depth * (TWO_PI * rate * i as f64 / fs).sin();
                inst.push(f);
                phase += TWO_PI * f / fs;
                phase.sin()
            })
            .collect();
        let track = pitch_track(&x, fs, 512, PitchMethod::Yin);
        let frame = 2048;
        for (idx, (t, f)) in track.iter().enumerate() {
            let f = f.expect("vibrato frame unvoiced");
            let start = idx * 512;
            let mean_inst: f64 =
                inst[start..start + frame].iter().sum::<f64>() / frame as f64;
            assert!(
                (f - mean_inst).abs() < 1.0,
                "at t={t}: yin {f} vs instantaneous {mean_inst}"
            );
        }
        // Note segmentation on a steady tone.
        let steady = pitch_track(&sine(440.0, fs, 24000), fs, 512, PitchMethod::Yin);
        let notes = pitch_to_midi_track(&steady);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].2, 69);
    }

    #[test]
    fn test_onsets_and_tempo() {
        let fs = 48000.0;
        let (n_fft, hop) = (1024, 256);
        // Click train at 120 BPM: one click every 0.5 s.
        let n = (16.0 * fs) as usize;
        let mut x = vec![0.0; n];
        let mut true_frames = Vec::new();
        // Start away from 0: spectral flux cannot flag an onset in the
        // very first frame (there is no preceding frame to increase from).
        let mut t = 24000;
        while t < n {
            for k in 0..64.min(n - t) {
                x[t + k] += (1.0 - k as f64 / 64.0) * if k % 2 == 0 { 1.0 } else { -1.0 };
            }
            true_frames.push(t / hop);
            t += 24000;
        }
        let strength = onset_strength(&x, fs, n_fft, hop);
        let thresh = 0.3 * strength.iter().cloned().fold(0.0_f64, f64::max);
        let onsets = onset_detect(&strength, thresh, 20);
        assert_eq!(onsets.len(), true_frames.len(), "onset count");
        for (o, tf) in onsets.iter().zip(&true_frames) {
            assert!(o.abs_diff(*tf) <= 4, "onset at frame {o}, expected {tf}");
        }
        let bpm = tempo_estimate(&strength, fs / hop as f64);
        assert!((bpm - 120.0).abs() < 1.0, "tempo {bpm}");
        // Beat tracking: intervals near 0.5 s.
        let beats = beat_track(&x, fs);
        assert!(beats.len() >= 10);
        for w in beats.windows(2) {
            let dt = w[1] - w[0];
            assert!((dt - 0.5).abs() < 0.06, "beat interval {dt}");
        }
        // HFC and complex-domain functions spike at the clicks too.
        let hfc = onset_hfc(&x, fs, n_fft, hop);
        let cd = onset_complex_domain(&x, fs, n_fft, hop);
        let peak_h = hfc.iter().cloned().fold(0.0_f64, f64::max);
        let local = hfc[true_frames[1] - 2..true_frames[1] + 4]
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(local > 0.3 * peak_h, "hfc {local} vs peak {peak_h}");
        assert!(cd.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mfcc_white_noise_flat() {
        let fs = 48000.0;
        let mut rng = Rng::new(3);
        let x: Vec<f64> = (0..48000).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let c = mfcc(&x, fs, 1024, 512, 26, 13);
        assert!(!c.is_empty());
        let n_frames = c.len() as f64;
        let mean: Vec<f64> = (0..13)
            .map(|k| c.iter().map(|f| f[k]).sum::<f64>() / n_frames)
            .collect();
        // Flat log-mel spectrum: all DCT coefficients above c0 are small.
        for (k, m) in mean.iter().enumerate().skip(1) {
            assert!(m.abs() < 0.12 * mean[0].abs(), "c{k} = {m}, c0 = {}", mean[0]);
        }
        let d = delta_features(&c, 2);
        assert_eq!(d.len(), c.len());
        assert!(d[5].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lpc_formants_of_synthetic_vowel() {
        let fs = 16000.0;
        // Two-resonator "vowel": F1 = 700 Hz (bw 90), F2 = 1220 Hz (bw 110).
        let make_pole = |f: f64, bw: f64| {
            let r = (-PI * bw / fs).exp();
            (r, TWO_PI * f / fs)
        };
        let poles = [make_pole(700.0, 90.0), make_pole(1220.0, 110.0)];
        let n = 16000;
        let mut x = vec![0.0; n];
        for i in (0..n).step_by(160) {
            x[i] = 1.0; // 100 Hz pulse train
        }
        for &(r, th) in &poles {
            let (a1, a2) = (-2.0 * r * th.cos(), r * r);
            let mut y1 = 0.0;
            let mut y2 = 0.0;
            for v in x.iter_mut() {
                let y = *v - a1 * y1 - a2 * y2;
                y2 = y1;
                y1 = y;
                *v = y;
            }
        }
        let seg = &x[2000..2000 + 800];
        let win: Vec<f64> = (0..800)
            .map(|i| 0.54 - 0.46 * (TWO_PI * i as f64 / 799.0).cos())
            .collect();
        let wseg: Vec<f64> = seg.iter().zip(&win).map(|(a, b)| a * b).collect();
        let (a, gain) = lpc(&wseg, 8);
        assert!(gain > 0.0);
        let formants = lpc_to_formants(&a, fs);
        assert!(formants.len() >= 2, "formants: {formants:?}");
        assert!((formants[0].0 - 700.0).abs() < 50.0, "F1 {}", formants[0].0);
        assert!((formants[1].0 - 1220.0).abs() < 50.0, "F2 {}", formants[1].0);
        // The LPC spectrum peaks near F1.
        let spec = lpc_spectrum(&a, gain, 256, fs);
        let peak = (1..256).max_by(|&i, &j| spec[i].partial_cmp(&spec[j]).unwrap()).unwrap();
        let peak_f = peak as f64 * fs / 2.0 / 256.0;
        assert!((peak_f - 700.0).abs() < 100.0, "LPC spectrum peak {peak_f}");
        // Formant tracking runs.
        let track = formant_track(&x, fs, 8, 800);
        assert!(!track.is_empty());
    }

    #[test]
    fn test_lsp_roundtrip() {
        let fs = 16000.0;
        let x = sine(500.0, fs, 2048)
            .iter()
            .zip(sine(1300.0, fs, 2048))
            .map(|(a, b)| a + 0.7 * b)
            .collect::<Vec<f64>>();
        let (a, _) = lpc(&x, 8);
        let lsp = lpc_to_lsp(&a);
        assert_eq!(lsp.len(), 8);
        for w in lsp.windows(2) {
            assert!(w[1] > w[0], "LSPs not sorted: {lsp:?}");
        }
        let a2 = lsp_to_lpc(&lsp);
        for (u, v) in a.iter().zip(&a2) {
            assert!((u - v).abs() < 1e-6, "roundtrip {a:?} vs {a2:?}");
        }
    }

    #[test]
    fn test_spectral_descriptors() {
        let fs = 48000.0;
        let mut rng = Rng::new(11);
        let noise: Vec<f64> = (0..16384).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let tone = sine(1000.0, fs, 16384);
        let nf = spectral_features_track(&noise, fs, 2048, 1024);
        let tf = spectral_features_track(&tone, fs, 2048, 1024);
        let n_mean = |f: fn(&SpectralFeatures) -> f64, t: &[SpectralFeatures]| {
            t.iter().map(f).sum::<f64>() / t.len() as f64
        };
        let c = n_mean(|s| s.centroid, &nf);
        assert!((c / (fs / 4.0) - 1.0).abs() < 0.15, "noise centroid {c}");
        assert!(n_mean(|s| s.flatness, &nf) > 0.4);
        assert!(n_mean(|s| s.flatness, &tf) < 0.05);
        assert!(n_mean(|s| s.entropy, &nf) > n_mean(|s| s.entropy, &tf));
        assert!(n_mean(|s| s.crest, &tf) > n_mean(|s| s.crest, &nf));
        let t_c = n_mean(|s| s.centroid, &tf);
        assert!((t_c - 1000.0).abs() < 100.0, "tone centroid {t_c}");
        let roll = n_mean(|s| s.rolloff85, &nf);
        assert!(roll > 0.5 * fs / 2.0 && roll < fs / 2.0, "rolloff {roll}");
        // ZCR of a 1 kHz tone ~ 2 f / fs.
        let z = zero_crossing_rate(&tone, 1024, 1024);
        let zm = z.iter().sum::<f64>() / z.len() as f64;
        assert!((zm - 2.0 * 1000.0 / fs).abs() < 0.005, "zcr {zm}");
    }

    #[test]
    fn test_hnr_and_inharmonicity() {
        let fs = 48000.0;
        let tone = sine(220.0, fs, 8192);
        let hnr_t = harmonic_to_noise_ratio(&tone, fs, 220.0);
        assert!(hnr_t > 20.0, "tone HNR {hnr_t}");
        let mut rng = Rng::new(5);
        let noise: Vec<f64> = (0..8192).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let hnr_n = harmonic_to_noise_ratio(&noise, fs, 220.0);
        assert!(hnr_n < 8.0, "noise HNR {hnr_n}");
        // Stretched partials with B = 1e-3.
        let b = 1e-3;
        let f0 = 200.0;
        let n = 16384;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                (1..=6)
                    .map(|k| {
                        let fk = k as f64 * f0 * (1.0 + b * (k * k) as f64).sqrt();
                        (TWO_PI * fk * i as f64 / fs).sin() / k as f64
                    })
                    .sum()
            })
            .collect();
        let b_est = inharmonicity_measure(&x, fs, f0);
        assert!((b_est / b - 1.0).abs() < 0.3, "B {b_est} vs {b}");
        let harmonic: Vec<f64> = (0..n)
            .map(|i| (1..=6).map(|k| (TWO_PI * k as f64 * f0 * i as f64 / fs).sin()).sum())
            .collect();
        assert!(inharmonicity_measure(&harmonic, fs, f0).abs() < 2e-4);
    }

    #[test]
    fn test_chroma_key_chord() {
        let fs = 48000.0;
        // C major triad: C4, E4, G4.
        let n = 32768;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                [261.63, 329.63, 392.0]
                    .iter()
                    .map(|f| (TWO_PI * f * i as f64 / fs).sin())
                    .sum()
            })
            .collect();
        let ch = chroma(&x, fs, 4096, 2048);
        assert!(!ch.is_empty());
        let frame = &ch[ch.len() / 2];
        assert_eq!(chord_estimate(frame), "C");
        let (tonic, major) = key_estimate(&ch);
        assert_eq!(tonic, 0);
        assert!(major);
        // A minor triad.
        let y: Vec<f64> = (0..n)
            .map(|i| {
                [220.0, 261.63, 329.63]
                    .iter()
                    .map(|f| (TWO_PI * f * i as f64 / fs).sin())
                    .sum()
            })
            .collect();
        let chy = chroma(&y, fs, 4096, 2048);
        assert_eq!(chord_estimate(&chy[chy.len() / 2]), "Am");
    }

    #[test]
    fn test_psychoacoustics_and_segmentation() {
        let fs = 48000.0;
        let quiet: Vec<f64> = sine(1000.0, fs, 24000).iter().map(|v| 0.01 * v).collect();
        let loud = sine(1000.0, fs, 24000);
        assert!(loudness_sone(&loud, fs) > loudness_sone(&quiet, fs));
        assert!(sharpness(&sine(8000.0, fs, 8192), fs) > sharpness(&sine(200.0, fs, 8192), fs));
        // 70 Hz AM raises roughness; 4 Hz AM raises fluctuation strength.
        let am = |fm: f64| -> Vec<f64> {
            (0..48000)
                .map(|i| {
                    let t = i as f64 / fs;
                    (1.0 + (TWO_PI * fm * t).cos()) * (TWO_PI * 1000.0 * t).sin()
                })
                .collect()
        };
        assert!(roughness(&am(70.0), fs) > roughness(&loud, fs) + 0.1);
        assert!(fluctuation_strength(&am(4.0), fs) > fluctuation_strength(&am(70.0), fs));
        // Silence detection.
        let mut sig = sine(440.0, fs, 12000);
        sig.extend(vec![0.0; 12000]);
        sig.extend(sine(440.0, fs, 12000));
        let silent = silence_detect(&sig, -60.0, 4800);
        assert_eq!(silent.len(), 1);
        assert!(silent[0].0 >= 11800 && silent[0].0 <= 12500, "{silent:?}");
        // Transients on clicks.
        let mut clicks = vec![0.0; 48000];
        for &p in &[8000_usize, 24000, 40000] {
            for k in 0..32 {
                clicks[p + k] = if k % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let trans = transient_detect(&clicks, fs);
        assert_eq!(trans.len(), 3, "{trans:?}");
        for (t, p) in trans.iter().zip([8000_usize, 24000, 40000]) {
            assert!(t.abs_diff(p) < 600, "transient {t} vs {p}");
        }
        // SNR.
        let mut rng = Rng::new(2);
        let noise: Vec<f64> = (0..24000).map(|_| 0.1 * (2.0 * rng.next_f64() - 1.0)).collect();
        let noisy: Vec<f64> = loud.iter().zip(&noise).map(|(s, n)| s + n).collect();
        let snr = estimate_snr(&noisy, &noise);
        let expected = 10.0 * (0.5_f64 / (0.01 / 3.0)).log10();
        assert!((snr - expected).abs() < 1.5, "snr {snr} vs {expected}");
    }

    #[test]
    fn test_thd_sinad_enob() {
        let fs = 48000.0;
        let n = 4800;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (TWO_PI * 1000.0 * t).sin() + 0.01 * (TWO_PI * 3000.0 * t).sin()
            })
            .collect();
        let thd = thd_n(&x, fs, 1000.0);
        assert!((thd / 0.01 - 1.0).abs() < 0.15, "thd {thd}");
        let s = sinad(&x, fs, 1000.0);
        assert!((s - 40.0).abs() < 1.0, "sinad {s}");
        assert!((enob(s) - (s - 1.76) / 6.02).abs() < 1e-12);
    }

    #[test]
    fn test_rt60_and_room_metrics() {
        let fs = 16000.0;
        let rt = 0.8;
        let mut rng = Rng::new(9);
        let n = (1.5 * fs) as usize;
        // Exponentially decaying noise IR with exact -60 dB at t = rt.
        let ir: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * rng.next_f64() - 1.0) * 10.0_f64.powf(-3.0 * t / rt)
            })
            .collect();
        let measured = rt60_from_ir(&ir, fs);
        assert!((measured / rt - 1.0).abs() < 0.05, "rt60 {measured}");
        let edt = edt_from_ir(&ir, fs);
        assert!((edt / rt - 1.0).abs() < 0.15, "edt {edt}");
        assert!(c80(&ir, fs) > c50(&ir, fs));
        let d = d50(&ir, fs);
        assert!(d > 0.0 && d < 1.0);
        // Shorter reverberation → higher clarity and STI.
        let short: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * rng.next_f64() - 1.0) * 10.0_f64.powf(-3.0 * t / 0.1)
            })
            .collect();
        assert!(c50(&short, fs) > c50(&ir, fs));
        assert!(sti_approx(&short, fs) > sti_approx(&ir, fs));
        assert!(sti_approx(&short, fs) <= 1.0);
    }

    #[test]
    fn test_sweep_deconvolution() {
        let fs = 24000.0;
        let (sweep, inverse) = crate::audio::oscillators::sine_sweep_with_inverse(
            40.0, 10000.0, 1.0, fs,
        );
        // System: delta at 100 plus half-amplitude echo at 400.
        let mut recorded = vec![0.0; sweep.len() + 500];
        for (i, &s) in sweep.iter().enumerate() {
            recorded[i + 100] += s;
            recorded[i + 400] += 0.5 * s;
        }
        let ir = impulse_response_from_sweep(&recorded, &inverse);
        let peak = ir.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let p1 = (0..ir.len()).max_by(|&a, &b| ir[a].abs().partial_cmp(&ir[b].abs()).unwrap()).unwrap();
        assert!(p1.abs_diff(100) <= 2, "direct path at {p1}");
        let echo_peak = ir[395..405].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!((echo_peak / peak - 0.5).abs() < 0.1, "echo ratio {}", echo_peak / peak);
    }

    #[test]
    fn test_utilities() {
        // peak_pick honors threshold and distance.
        let x = [0.0, 1.0, 0.0, 0.9, 0.0, 0.2, 0.0, 3.0, 0.0];
        assert_eq!(peak_pick(&x, 0.5, 1), vec![1, 3, 7]);
        assert_eq!(peak_pick(&x, 0.5, 3), vec![1, 7]);
        // Autocorrelation of a sine has a peak at its period.
        let fs = 8000.0;
        let x = sine(200.0, fs, 2048);
        let r = autocorrelation_fft(&x);
        assert!(r[0] > 0.0);
        let period = (fs / 200.0) as usize;
        assert!(r[period] > 0.8 * r[0] * (1.0 - period as f64 / 2048.0));
        // DTW: identical sequences cost ~0 with diagonal path.
        let a: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let (cost, path) = dynamic_time_warping(&a, &a);
        assert!(cost < 1e-12);
        assert_eq!(path.len(), 10);
        assert!(path.iter().enumerate().all(|(i, &(p, q))| p == i && q == i));
        let b: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 + 0.5]).collect();
        let (cost2, _) = dynamic_time_warping(&a, &b);
        assert!(cost2 > cost);
        // Fingerprints: deterministic and content-sensitive.
        let s1: Vec<f64> = sine(440.0, 8000.0, 16384)
            .iter()
            .zip(sine(1250.0, 8000.0, 16384))
            .map(|(a, b)| a + b)
            .collect();
        let f1 = audio_fingerprint(&s1, 8000.0);
        let f1b = audio_fingerprint(&s1, 8000.0);
        assert!(!f1.is_empty());
        assert_eq!(f1, f1b);
        let s2: Vec<f64> = sine(523.0, 8000.0, 16384)
            .iter()
            .zip(sine(1770.0, 8000.0, 16384))
            .map(|(a, b)| a + b)
            .collect();
        let f2 = audio_fingerprint(&s2, 8000.0);
        let common = f1.iter().filter(|h| f2.contains(h)).count();
        assert!(common < f1.len() / 2, "fingerprints too similar");
    }
}
