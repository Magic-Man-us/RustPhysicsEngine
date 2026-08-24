//! Envelopes, LFOs, followers, fades, and glides.

use crate::audio::oscillators::Oscillator;
use crate::math::constants::PI;

/// ADSR state machine phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdsrState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Linear (optionally exponential-curved) ADSR envelope; times in
/// seconds, sustain as a level in \[0, 1\].
pub struct Adsr {
    pub attack: f64,
    pub decay: f64,
    pub sustain: f64,
    pub release: f64,
    fs: f64,
    state: AdsrState,
    level: f64,
    exp_curve: bool,
}

impl Adsr {
    /// New idle envelope.
    #[must_use]
    pub fn new(attack: f64, decay: f64, sustain: f64, release: f64, fs: f64) -> Self {
        Self {
            attack,
            decay,
            sustain,
            release,
            fs,
            state: AdsrState::Idle,
            level: 0.0,
            exp_curve: false,
        }
    }

    /// Start the attack from the current level.
    pub fn gate_on(&mut self) {
        self.state = AdsrState::Attack;
    }

    /// Enter the release phase.
    pub fn gate_off(&mut self) {
        self.state = AdsrState::Release;
    }

    /// Use exponential (one-pole style) segment shapes instead of
    /// linear ramps.
    pub fn set_curve(&mut self, exp: bool) {
        self.exp_curve = exp;
    }

    /// Envelope value for the next sample.
    #[allow(clippy::should_implement_trait)] // roadmap API name
    pub fn next(&mut self) -> f64 {
        let dt = 1.0 / self.fs;
        match self.state {
            AdsrState::Idle => {
                self.level = 0.0;
            }
            AdsrState::Attack => {
                if self.attack <= dt {
                    self.level = 1.0;
                } else if self.exp_curve {
                    let coef = (-(dt) / (self.attack / 5.0)).exp();
                    self.level = 1.15 - (1.15 - self.level) * coef;
                    self.level = self.level.min(1.0);
                } else {
                    self.level += dt / self.attack;
                }
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.state = AdsrState::Decay;
                }
            }
            AdsrState::Decay => {
                if self.decay <= dt {
                    self.level = self.sustain;
                } else if self.exp_curve {
                    let coef = (-(dt) / (self.decay / 5.0)).exp();
                    self.level = self.sustain + (self.level - self.sustain) * coef;
                } else {
                    self.level -= dt * (1.0 - self.sustain) / self.decay;
                }
                if self.level <= self.sustain + 1e-9 {
                    self.level = self.sustain;
                    self.state = AdsrState::Sustain;
                }
            }
            AdsrState::Sustain => {
                self.level = self.sustain;
            }
            AdsrState::Release => {
                if self.release <= dt {
                    self.level = 0.0;
                } else if self.exp_curve {
                    let coef = (-(dt) / (self.release / 5.0)).exp();
                    self.level *= coef;
                } else {
                    self.level -= dt * self.sustain.max(1e-9) / self.release;
                }
                if self.level <= 1e-6 {
                    self.level = 0.0;
                    self.state = AdsrState::Idle;
                }
            }
        }
        self.level
    }

    /// True while the envelope is producing signal.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state != AdsrState::Idle
    }
}

/// Simple linear attack-release envelope (a one-shot AD when the gate
/// is released immediately).
pub struct Ar {
    pub attack: f64,
    pub release: f64,
    fs: f64,
    level: f64,
    rising: bool,
    active: bool,
}

impl Ar {
    /// New idle attack-release envelope.
    #[must_use]
    pub fn new(attack: f64, release: f64, fs: f64) -> Self {
        Self { attack, release, fs, level: 0.0, rising: false, active: false }
    }

    /// Trigger the attack.
    pub fn gate_on(&mut self) {
        self.rising = true;
        self.active = true;
    }

    /// Begin the release.
    pub fn gate_off(&mut self) {
        self.rising = false;
    }

    /// Next envelope value.
    #[allow(clippy::should_implement_trait)] // roadmap API name
    pub fn next(&mut self) -> f64 {
        let dt = 1.0 / self.fs;
        if self.active {
            if self.rising {
                self.level = (self.level + dt / self.attack.max(dt)).min(1.0);
            } else {
                self.level -= dt / self.release.max(dt);
                if self.level <= 0.0 {
                    self.level = 0.0;
                    self.active = false;
                }
            }
        }
        self.level
    }

    /// True while producing signal.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Exponential ADSR driven by RC time constants (τ per segment).
pub struct AdsrExp {
    pub attack_tau: f64,
    pub decay_tau: f64,
    pub sustain: f64,
    pub release_tau: f64,
    fs: f64,
    state: AdsrState,
    level: f64,
}

impl AdsrExp {
    /// New idle exponential envelope.
    #[must_use]
    pub fn new(attack_tau: f64, decay_tau: f64, sustain: f64, release_tau: f64, fs: f64) -> Self {
        Self { attack_tau, decay_tau, sustain, release_tau, fs, state: AdsrState::Idle, level: 0.0 }
    }

    /// Trigger the attack.
    pub fn gate_on(&mut self) {
        self.state = AdsrState::Attack;
    }

    /// Begin the release.
    pub fn gate_off(&mut self) {
        self.state = AdsrState::Release;
    }

    /// Next envelope value.
    #[allow(clippy::should_implement_trait)] // roadmap API name
    pub fn next(&mut self) -> f64 {
        let dt = 1.0 / self.fs;
        match self.state {
            AdsrState::Idle => self.level = 0.0,
            AdsrState::Attack => {
                let coef = (-dt / self.attack_tau).exp();
                // Aim above 1 so the attack actually reaches it.
                self.level = 1.2 - (1.2 - self.level) * coef;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.state = AdsrState::Decay;
                }
            }
            AdsrState::Decay | AdsrState::Sustain => {
                let coef = (-dt / self.decay_tau).exp();
                self.level = self.sustain + (self.level - self.sustain) * coef;
                self.state = AdsrState::Sustain;
            }
            AdsrState::Release => {
                let coef = (-dt / self.release_tau).exp();
                self.level *= coef;
                if self.level < 1e-6 {
                    self.level = 0.0;
                    self.state = AdsrState::Idle;
                }
            }
        }
        self.level
    }

    /// True while producing signal.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state != AdsrState::Idle
    }
}

/// Low-frequency oscillator: scaled/offset wrapper over [`Oscillator`].
pub struct Lfo {
    pub osc: Oscillator,
    pub depth: f64,
    pub offset: f64,
}

impl Lfo {
    /// Next LFO value offset + depth·osc.
    #[allow(clippy::should_implement_trait)] // roadmap API name
    pub fn next(&mut self) -> f64 {
        self.offset + self.depth * self.osc.next()
    }

    /// Reset the LFO phase.
    pub fn sync(&mut self) {
        self.osc.set_phase(0.0);
    }
}

/// Peak envelope follower with attack/release time constants (ms).
#[must_use]
pub fn envelope_follower(x: &[f64], attack_ms: f64, release_ms: f64, fs: f64) -> Vec<f64> {
    let att = (-1.0 / (attack_ms * 1e-3 * fs)).exp();
    let rel = (-1.0 / (release_ms * 1e-3 * fs)).exp();
    let mut env = 0.0_f64;
    x.iter()
        .map(|&v| {
            let a = v.abs();
            let coef = if a > env { att } else { rel };
            env = a + coef * (env - a);
            env
        })
        .collect()
}

/// Sliding-window peak magnitude (centered).
#[must_use]
pub fn peak_envelope(x: &[f64], window: usize) -> Vec<f64> {
    let half = window.max(1) / 2;
    (0..x.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(x.len());
            x[lo..hi].iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
        })
        .collect()
}

/// Sliding-window RMS (centered).
#[must_use]
pub fn rms_envelope(x: &[f64], window: usize) -> Vec<f64> {
    let half = window.max(1) / 2;
    (0..x.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(x.len());
            let s: f64 = x[lo..hi].iter().map(|v| v * v).sum();
            (s / (hi - lo) as f64).sqrt()
        })
        .collect()
}

/// e^(−t/τ) sampled for n samples.
#[must_use]
pub fn exponential_decay_envelope(n: usize, tau: f64, fs: f64) -> Vec<f64> {
    (0..n).map(|i| (-(i as f64) / (tau * fs)).exp()).collect()
}

/// Multiply a signal by an envelope in place.
pub fn apply_envelope(x: &mut [f64], env: &[f64]) {
    for (v, &e) in x.iter_mut().zip(env) {
        *v *= e;
    }
}

/// Fade curve shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeShape {
    Linear,
    EqualPower,
    Exponential,
    SCurve,
}

fn fade_gain(shape: FadeShape, t: f64) -> f64 {
    // t ∈ [0, 1]: 0 = silent, 1 = full.
    match shape {
        FadeShape::Linear => t,
        FadeShape::EqualPower => (t * PI / 2.0).sin(),
        FadeShape::Exponential => t * t * t,
        FadeShape::SCurve => 0.5 - 0.5 * (PI * t).cos(),
    }
}

/// Fade in the first n samples in place.
pub fn fade_in(x: &mut [f64], n: usize, shape: FadeShape) {
    let n = n.min(x.len());
    for (i, v) in x.iter_mut().enumerate().take(n) {
        *v *= fade_gain(shape, i as f64 / n.max(1) as f64);
    }
}

/// Fade out the last n samples in place.
pub fn fade_out(x: &mut [f64], n: usize, shape: FadeShape) {
    let len = x.len();
    let n = n.min(len);
    for i in 0..n {
        x[len - 1 - i] *= fade_gain(shape, i as f64 / n.max(1) as f64);
    }
}

/// Full-length crossfade from a to b.
///
/// # Panics
/// Panics if the inputs differ in length.
#[must_use]
pub fn crossfade(a: &[f64], b: &[f64], shape: FadeShape) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "crossfade inputs must match");
    let n = a.len();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1).max(1) as f64;
            let gb = fade_gain(shape, t);
            let ga = fade_gain(shape, 1.0 - t);
            a[i] * ga + b[i] * gb
        })
        .collect()
}

/// Pitch glide trajectory (Hz per sample): linear or exponential
/// (constant cents/second) from one frequency to another.
#[must_use]
pub fn portamento(from_hz: f64, to_hz: f64, n: usize, fs: f64, exponential: bool) -> Vec<f64> {
    let _ = fs;
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1).max(1) as f64;
            if exponential {
                from_hz * (to_hz / from_hz).powf(t)
            } else {
                from_hz + (to_hz - from_hz) * t
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::oscillators::Waveform;

    #[test]
    fn test_adsr_reaches_sustain_after_attack_plus_decay() {
        let fs = 1000.0;
        let mut env = Adsr::new(0.1, 0.2, 0.6, 0.3, fs);
        env.gate_on();
        let mut last = 0.0;
        let steps = ((0.1 + 0.2) * fs) as usize + 2;
        let mut peak = 0.0_f64;
        for _ in 0..steps {
            last = env.next();
            peak = peak.max(last);
        }
        assert!((peak - 1.0).abs() < 1e-9, "peak {peak}");
        assert!((last - 0.6).abs() < 1e-6, "sustain {last}");
        // Holds at sustain.
        for _ in 0..500 {
            last = env.next();
        }
        assert!((last - 0.6).abs() < 1e-9);
        // Releases to zero.
        env.gate_off();
        for _ in 0..((0.35 * fs) as usize) {
            last = env.next();
        }
        assert!(last < 1e-6, "release leak {last}");
        assert!(!env.is_active());
    }

    #[test]
    fn test_ar_and_exp_envelopes() {
        let fs = 1000.0;
        let mut ar = Ar::new(0.05, 0.05, fs);
        ar.gate_on();
        for _ in 0..60 {
            ar.next();
        }
        assert!(ar.next() > 0.99);
        ar.gate_off();
        for _ in 0..70 {
            ar.next();
        }
        assert!(!ar.is_active());
        let mut ex = AdsrExp::new(0.01, 0.05, 0.5, 0.05, fs);
        ex.gate_on();
        let mut peak = 0.0_f64;
        for _ in 0..200 {
            peak = peak.max(ex.next());
        }
        assert!((peak - 1.0).abs() < 1e-9);
        for _ in 0..1000 {
            ex.next();
        }
        assert!((ex.next() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_adsr_exp_release_is_a_pure_rc_decay() {
        let fs = 1000.0;
        let (attack_tau, decay_tau, sustain, release_tau) = (0.005, 0.02, 0.5, 0.05);
        let mut env = AdsrExp::new(attack_tau, decay_tau, sustain, release_tau, fs);
        // Idle envelopes are not active and produce nothing.
        assert!(!env.is_active());
        assert_eq!(env.next(), 0.0);

        env.gate_on();
        assert!(env.is_active());
        // Settle onto the sustain level.
        let mut level = 0.0;
        for _ in 0..500 {
            level = env.next();
        }
        assert!((level - sustain).abs() < 1e-9, "sustain {level}");
        assert!(env.is_active(), "sustaining envelope must be active");

        // Release: level(k) = level0·exp(−k·dt/τ) exactly, until it is
        // snapped to zero below 1e-6.
        env.gate_off();
        assert!(env.is_active(), "release must still be active");
        let level0 = level;
        let dt = 1.0 / fs;
        let mut k = 0usize;
        loop {
            k += 1;
            let v = env.next();
            let expect = level0 * (-(k as f64) * dt / release_tau).exp();
            if expect < 1e-6 {
                break;
            }
            assert!((v - expect).abs() < 1e-12, "release step {k}: {v} vs {expect}");
            assert!(env.is_active(), "went idle at step {k} while still audible");
        }
        // One time constant into the release the level is down by 1/e.
        let tau_samples = (release_tau * fs) as usize;
        assert!(tau_samples > 10);
        // The tail terminates exactly at zero and reports itself finished.
        for _ in 0..200 {
            env.next();
        }
        assert!(!env.is_active(), "release never completed");
        assert_eq!(env.next(), 0.0, "idle envelope must output silence");

        // Re-gating restarts the attack from the current (zero) level.
        env.gate_on();
        assert!(env.is_active());
        let mut peak = 0.0_f64;
        for _ in 0..200 {
            peak = peak.max(env.next());
        }
        assert!((peak - 1.0).abs() < 1e-12, "re-attack peak {peak}");

        // Releasing from a higher level takes proportionally longer to
        // reach the same floor: t = τ·ln(level0/floor).
        let count_release = |from_sustain: f64| -> usize {
            let mut e = AdsrExp::new(attack_tau, 0.001, from_sustain, release_tau, fs);
            e.gate_on();
            for _ in 0..2000 {
                e.next();
            }
            e.gate_off();
            let mut n = 0;
            while e.is_active() {
                e.next();
                n += 1;
            }
            n
        };
        let n_loud = count_release(0.8) as f64;
        let n_quiet = count_release(0.1) as f64;
        let expect = release_tau * fs * (0.8_f64 / 0.1).ln();
        assert!(
            (n_loud - n_quiet - expect).abs() < 2.0,
            "release lengths {n_loud} - {n_quiet} vs {expect}"
        );
    }

    #[test]
    fn test_lfo_and_followers() {
        let fs = 1000.0;
        let osc = Oscillator::new(Waveform::Sine, 2.0, fs);
        let mut lfo = Lfo { osc, depth: 0.5, offset: 1.0 };
        let vals: Vec<f64> = (0..1000).map(|_| lfo.next()).collect();
        let min = vals.iter().cloned().fold(f64::MAX, f64::min);
        let max = vals.iter().cloned().fold(f64::MIN, f64::max);
        assert!(min > 0.49 && max < 1.51);
        lfo.sync();
        assert_eq!(lfo.osc.phase, 0.0);
        // Follower tracks a burst and decays after it.
        let mut x = vec![0.0; 1000];
        for v in x.iter_mut().take(500).skip(100) {
            *v = 1.0;
        }
        let env = envelope_follower(&x, 1.0, 50.0, fs);
        assert!(env[400] > 0.95);
        assert!(env[600] < env[499] && env[600] > 0.05);
        let pk = peak_envelope(&x, 21);
        assert_eq!(pk[300], 1.0);
        let rms = rms_envelope(&vec![0.5; 100], 11);
        assert!((rms[50] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_equal_power_crossfade_keeps_rms() {
        // Two uncorrelated noise-ish signals: equal-power crossfade keeps
        // the summed power constant.
        let n = 8000;
        let a: Vec<f64> = (0..n).map(|i| ((i * 7919) % 200) as f64 / 100.0 - 1.0).collect();
        let b: Vec<f64> = (0..n).map(|i| ((i * 104729 + 13) % 200) as f64 / 100.0 - 1.0).collect();
        let rms = |x: &[f64]| (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
        let (ra, rb) = (rms(&a), rms(&b));
        let cross = crossfade(&a, &b, FadeShape::EqualPower);
        // Windowed RMS through the fade stays near the endpoint RMS.
        for seg in 0..8 {
            let lo = seg * 1000;
            let r = rms(&cross[lo..lo + 1000]);
            let target = 0.5 * (ra + rb);
            assert!((r - target).abs() / target < 0.1, "segment {seg}: {r} vs {target}");
        }
        // Linear crossfade dips in the middle.
        let lin = crossfade(&a, &b, FadeShape::Linear);
        let mid = rms(&lin[3500..4500]);
        assert!(mid < 0.85 * ra, "linear midpoint {mid} vs {ra}");
    }

    #[test]
    fn test_fades_and_portamento() {
        let mut x = vec![1.0; 100];
        fade_in(&mut x, 50, FadeShape::Linear);
        assert_eq!(x[0], 0.0);
        assert!((x[25] - 0.5).abs() < 0.03);
        assert_eq!(x[99], 1.0);
        let mut y = vec![1.0; 100];
        fade_out(&mut y, 50, FadeShape::SCurve);
        assert_eq!(y[99], 0.0);
        assert_eq!(y[0], 1.0);
        let gl = portamento(220.0, 440.0, 101, 1000.0, true);
        assert!((gl[0] - 220.0).abs() < 1e-9);
        assert!((gl[100] - 440.0).abs() < 1e-9);
        // Exponential glide hits the geometric mean at the midpoint.
        assert!((gl[50] - (220.0 * 440.0_f64).sqrt()).abs() < 0.5);
        let env = exponential_decay_envelope(100, 0.01, 1000.0);
        assert!((env[10] - (-1.0_f64).exp()).abs() < 1e-9);
        let mut sig = vec![2.0; 3];
        apply_envelope(&mut sig, &[0.5, 1.0, 0.0]);
        assert_eq!(sig, vec![1.0, 2.0, 0.0]);
    }
}
