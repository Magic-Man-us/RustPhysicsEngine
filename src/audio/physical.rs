//! Physical modeling synthesis: digital waveguides (plucked/struck/bowed
//! strings, clarinet and flute bores), modal synthesis (bars, membranes,
//! plates, bells, glasses), finite-difference membranes and Kirchhoff
//! plates, a brute-force mass-spring string for validation, the
//! Kelly-Lochbaum vocal tract, and glottal source models.

// The roadmap fixes the sample-generator method name as `next()`.
#![allow(clippy::should_implement_trait)]

use crate::dsp::iir::Biquad;
use crate::monte_carlo::Rng;
use crate::resonance::cavity::{
    beam_modes, bell_modes_approx, circular_membrane_modes, rectangular_plate_modes, BeamBc,
    PlateBc,
};
use crate::sim::wave_sim::WaveEquation2D;

const TWO_PI: f64 = 2.0 * crate::math::constants::PI;
const PI: f64 = crate::math::constants::PI;

// --- Shared helpers ------------------------------------------------------

/// Phase delay of a biquad in samples at frequency `f`.
fn phase_delay_samples(bq: &Biquad, f: f64, fs: f64) -> f64 {
    let h = bq.freq_response(f, fs);
    let omega = TWO_PI * f / fs;
    if omega <= 0.0 {
        return 0.0;
    }
    // Principal value is enough for the sub-π phases of the short loop
    // filters used here.
    let mut phase = h.arg();
    if phase > 0.0 {
        phase -= TWO_PI;
    }
    -phase / omega
}

/// Phase delay in samples of the first-order allpass
/// H(z) = (η + z⁻¹)/(1 + η z⁻¹) at radian frequency `omega` (rad/sample).
fn allpass_phase_delay(eta: f64, omega: f64) -> f64 {
    let (c, s) = (omega.cos(), omega.sin());
    // H(e^{-jω}) numerator: η + cos ω − j sin ω; denominator: 1 + η cos ω − j η sin ω.
    let num = f64::atan2(-s, eta + c);
    let den = f64::atan2(-eta * s, 1.0 + eta * c);
    -(num - den) / omega
}

/// Solve for the allpass coefficient giving phase delay `d` samples at
/// `omega` (rad/sample). Valid for d roughly in (0.1, 3) at audio rates.
fn allpass_coeff_for_delay(d: f64, omega: f64) -> f64 {
    // Phase delay decreases monotonically with η.
    let (mut lo, mut hi) = (-0.999, 0.999);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if allpass_phase_delay(mid, omega) > d {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// STK-style bow friction table: reflection coefficient as a function of
/// the bow/string velocity difference.
fn bow_friction(delta_v: f64, slope: f64) -> f64 {
    ((delta_v * slope).abs() + 0.75).powi(-4).min(1.0)
}

// --- Waveguide string ----------------------------------------------------

/// Bidirectional digital waveguide string with bridge damping filter,
/// optional stiffness allpass, and an internal tuning allpass keeping the
/// pitch exact at the fundamental.
pub struct WaveguideString {
    delay_l: Vec<f64>,
    delay_r: Vec<f64>,
    pos: usize,
    freq: f64,
    fs: f64,
    /// Per-reflection bridge loss (loop gain), slightly below 1.
    pub damping: f64,
    /// Dispersion (stiffness) allpass in the bridge loop.
    pub stiffness_allpass: Biquad,
    /// Bridge reflection lowpass.
    pub bridge_filter: Biquad,
    /// Default pluck/strike position as a fraction of the string length.
    pub pluck_pos: f64,
    n: usize,
    tune_eta: f64,
    tune_x1: f64,
    tune_y1: f64,
    bow: Option<(f64, f64, f64)>, // (force, velocity, position)
}

impl WaveguideString {
    /// New string tuned to `freq` at sample rate `fs`.
    #[must_use]
    pub fn new(freq: f64, fs: f64) -> Self {
        let s = 0.4;
        let mut wg = Self {
            delay_l: Vec::new(),
            delay_r: Vec::new(),
            pos: 0,
            freq,
            fs,
            damping: 0.996,
            stiffness_allpass: Biquad::identity(),
            bridge_filter: Biquad::from_coeffs(1.0 - s, 0.0, 0.0, -s, 0.0),
            pluck_pos: 0.28,
            n: 1,
            tune_eta: 0.0,
            tune_x1: 0.0,
            tune_y1: 0.0,
            bow: None,
        };
        wg.set_freq(freq);
        wg
    }

    /// Retune to a new fundamental, sizing the rails and the tuning
    /// allpass so the loop delay is exactly `fs/freq` samples at `freq`.
    pub fn set_freq(&mut self, freq: f64) {
        self.freq = freq;
        let total = self.fs / freq;
        let fixed = phase_delay_samples(&self.bridge_filter, freq, self.fs)
            + phase_delay_samples(&self.stiffness_allpass, freq, self.fs);
        let n = (((total - fixed - 0.5) / 2.0).floor().max(1.0)) as usize;
        let frac = total - fixed - 2.0 * n as f64;
        let omega = TWO_PI * freq / self.fs;
        self.tune_eta = allpass_coeff_for_delay(frac, omega);
        self.n = n;
        self.delay_l.resize(n, 0.0);
        self.delay_r.resize(n, 0.0);
        self.pos %= n;
    }

    /// Current fundamental (Hz).
    #[must_use]
    pub fn freq(&self) -> f64 {
        self.freq
    }

    /// Pluck: load a smoothed triangular displacement peaked at `pos`
    /// (0..1) with amplitude `amp`; `width` (0..1) smooths the corner.
    pub fn pluck(&mut self, pos: f64, amp: f64, width: f64) {
        let n = self.n;
        let p = pos.clamp(0.05, 0.95);
        let mut shape: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / (n - 1).max(1) as f64;
                if x < p { x / p } else { (1.0 - x) / (1.0 - p) }
            })
            .collect();
        // One box-smoothing pass of width ~ width * n.
        let w = ((width * n as f64) as usize).max(1);
        if w > 1 {
            let mut acc = vec![0.0; n + 1];
            for i in 0..n {
                acc[i + 1] = acc[i] + shape[i];
            }
            shape = (0..n)
                .map(|i| {
                    let a = i.saturating_sub(w / 2);
                    let b = (i + w / 2 + 1).min(n);
                    (acc[b] - acc[a]) / (b - a) as f64
                })
                .collect();
        }
        let peak = shape.iter().cloned().fold(0.0_f64, f64::max).max(1e-12);
        self.pluck_pos = p;
        for (i, &v) in shape.iter().enumerate() {
            let v = 0.5 * amp * v / peak;
            self.delay_r[(self.pos + i) % n] += v;
            self.delay_l[(self.pos + n - 1 - i) % n] += v;
        }
    }

    /// Strike: inject a narrow raised-cosine velocity pulse at `pos`.
    pub fn strike(&mut self, pos: f64, vel: f64) {
        let n = self.n;
        let center = (pos.clamp(0.0, 1.0) * (n - 1) as f64) as usize;
        let hw = (n / 20).max(1);
        for k in 0..(2 * hw + 1) {
            let i = (center + k + n - hw) % n;
            let x = (k as f64 - hw as f64) / hw as f64;
            let v = 0.5 * vel * 0.5 * (1.0 + (PI * x).cos());
            self.delay_r[(self.pos + i) % n] += v;
            self.delay_l[(self.pos + n - 1 - i) % n] += v;
        }
    }

    /// Start (force > 0) or stop (force <= 0) bowing at `pos` with the
    /// given bow force (0..1) and velocity.
    pub fn bow(&mut self, force: f64, velocity: f64, pos: f64) {
        self.bow = if force > 0.0 {
            Some((force.min(1.0), velocity, pos.clamp(0.05, 0.95)))
        } else {
            None
        };
    }

    /// Advance one sample; returns the wave arriving at the bridge.
    pub fn next(&mut self) -> f64 {
        if let Some((force, velocity, pos)) = self.bow {
            let n = self.n;
            let bi = ((pos * (n - 1) as f64) as usize).min(n - 1);
            let jr = (self.pos + bi) % n;
            let jl = (self.pos + n - 1 - bi) % n;
            let v = self.delay_r[jr] + self.delay_l[jl];
            let dv = velocity - v;
            let f = 0.2 * force * dv * bow_friction(dv, 5.0 - 4.0 * force);
            self.delay_r[jr] += 0.5 * f;
            self.delay_l[jl] += 0.5 * f;
        }
        let i = self.pos;
        let to_bridge = self.delay_r[i];
        let to_nut = self.delay_l[i];
        // Tuning allpass y[n] = η x[n] + x[n-1] - η y[n-1].
        let ap = self.tune_eta * to_bridge + self.tune_x1 - self.tune_eta * self.tune_y1;
        self.tune_x1 = to_bridge;
        self.tune_y1 = ap;
        let filtered = self.bridge_filter.process(self.stiffness_allpass.process(ap));
        self.delay_l[i] = -self.damping * filtered;
        self.delay_r[i] = -to_nut;
        self.pos = (self.pos + 1) % self.n;
        to_bridge
    }

    /// Displacement at fractional position `pos` (0..1).
    #[must_use]
    pub fn output_at(&self, pos: f64) -> f64 {
        let n = self.n;
        let i = ((pos.clamp(0.0, 1.0) * (n - 1) as f64) as usize).min(n - 1);
        self.delay_r[(self.pos + i) % n] + self.delay_l[(self.pos + n - 1 - i) % n]
    }

    /// Inject an excitation sample at the pluck point (used by commuted
    /// synthesis and coupling helpers).
    fn inject(&mut self, x: f64) {
        let n = self.n;
        let i = ((self.pluck_pos * (n - 1) as f64) as usize).min(n - 1);
        self.delay_r[(self.pos + i) % n] += 0.5 * x;
        self.delay_l[(self.pos + n - 1 - i) % n] += 0.5 * x;
    }
}

// --- Waveguide tubes -----------------------------------------------------

enum TubeKind {
    Clarinet,
    Flute,
}

/// Single-reed (clarinet) or jet (flute) waveguide wind instrument.
pub struct WaveguideTube {
    kind: TubeKind,
    bore: Vec<f64>,
    pos: usize,
    jet: Vec<f64>,
    jet_pos: usize,
    breath: f64,
    reflection_filter: Biquad,
    dc_block: Biquad,
    rng: Rng,
}

impl WaveguideTube {
    /// Clarinet model: closed-open bore (round trip fs/(2 f)) with a reed
    /// reflection nonlinearity.
    #[must_use]
    pub fn clarinet(freq: f64, fs: f64) -> Self {
        let m = ((fs / (2.0 * freq) - 1.0).round().max(2.0)) as usize;
        Self {
            kind: TubeKind::Clarinet,
            bore: vec![0.0; m],
            pos: 0,
            jet: Vec::new(),
            jet_pos: 0,
            breath: 0.0,
            reflection_filter: Biquad::from_coeffs(0.5, 0.5, 0.0, 0.0, 0.0),
            dc_block: crate::dsp::iir::dc_blocker(0.995),
            rng: Rng::new(0x0dd5),
        }
    }

    /// Flute model: open-open bore (round trip fs/f) with a jet delay of
    /// half the bore and a cubic jet nonlinearity.
    #[must_use]
    pub fn flute(freq: f64, fs: f64) -> Self {
        let m = ((fs / freq - 2.0).round().max(4.0)) as usize;
        Self {
            kind: TubeKind::Flute,
            bore: vec![0.0; m],
            pos: 0,
            jet: vec![0.0; (m / 2).max(1)],
            jet_pos: 0,
            breath: 0.0,
            reflection_filter: Biquad::from_coeffs(0.7, 0.0, 0.0, -0.3, 0.0),
            dc_block: crate::dsp::iir::dc_blocker(0.995),
            rng: Rng::new(0xf1e7),
        }
    }

    /// Set the blowing pressure (0..~1).
    pub fn set_breath(&mut self, p: f64) {
        self.breath = p;
    }

    /// Advance one sample of bore output.
    pub fn next(&mut self) -> f64 {
        let noise = 2.0 * self.rng.next_f64() - 1.0;
        let bore_out = self.bore[self.pos];
        let out = match self.kind {
            TubeKind::Clarinet => {
                let breath = self.breath * (1.0 + 0.02 * noise);
                let refl = -0.95 * self.reflection_filter.process(bore_out);
                let dp = refl - breath;
                let r = reed_nonlinearity(dp, 0.3, 1.0);
                self.bore[self.pos] = breath + dp * r;
                bore_out
            }
            TubeKind::Flute => {
                let breath = self.breath * (1.0 + 0.05 * noise);
                let temp = self.dc_block.process(self.reflection_filter.process(bore_out));
                let jet_in = breath - 0.5 * temp;
                let jet_out = self.jet[self.jet_pos];
                self.jet[self.jet_pos] = jet_in;
                self.jet_pos = (self.jet_pos + 1) % self.jet.len();
                self.bore[self.pos] = jet_nonlinearity(jet_out) + 0.5 * temp;
                bore_out
            }
        };
        self.pos = (self.pos + 1) % self.bore.len();
        out
    }
}

// --- Modal synthesis -----------------------------------------------------

/// Bank of two-pole resonators driven by an excitation buffer.
pub struct ModalSynth {
    /// (frequency Hz, T60 seconds, gain) per mode.
    pub modes: Vec<(f64, f64, f64)>,
    resonators: Vec<Biquad>,
    excitation: Vec<f64>,
    exc_pos: usize,
    fs: f64,
}

impl ModalSynth {
    /// Build from an explicit (freq, t60, gain) mode list; modes at or
    /// above Nyquist are dropped.
    #[must_use]
    pub fn from_modes(modes: &[(f64, f64, f64)], fs: f64) -> Self {
        let kept: Vec<(f64, f64, f64)> =
            modes.iter().copied().filter(|m| m.0 > 0.0 && m.0 < 0.49 * fs).collect();
        let resonators = kept
            .iter()
            .map(|&(f, t60, gain)| {
                let r = (-3.0 * std::f64::consts::LN_10 / (fs * t60.max(1e-3))).exp();
                let theta = TWO_PI * f / fs;
                Biquad::from_coeffs(gain * (1.0 - r), 0.0, 0.0, -2.0 * r * theta.cos(), r * r)
            })
            .collect();
        Self { modes: kept, resonators, excitation: Vec::new(), exc_pos: 0, fs }
    }

    /// Free-free bar of rectangular cross section (Euler-Bernoulli).
    #[must_use]
    pub fn bar(length: f64, width: f64, thickness: f64, young: f64, rho: f64, fs: f64) -> Self {
        let i_area = width * thickness.powi(3) / 12.0;
        let area = width * thickness;
        let freqs = beam_modes(length, young, i_area, rho, area, BeamBc::FreeFree, 8);
        Self::from_bare_freqs(&freqs, fs)
    }

    /// Circular membrane (drum head) modal model.
    #[must_use]
    pub fn membrane_circular(radius: f64, tension: f64, sigma: f64, fs: f64) -> Self {
        let modes = circular_membrane_modes(radius, tension, sigma, 3, 3);
        let freqs: Vec<f64> = modes.iter().map(|m| m.2).collect();
        Self::from_bare_freqs(&freqs, fs)
    }

    /// Simply supported rectangular plate modal model.
    #[allow(clippy::too_many_arguments)] // physical parameter list
    #[must_use]
    pub fn plate(a: f64, b: f64, thickness: f64, young: f64, rho: f64, nu: f64, fs: f64) -> Self {
        let modes =
            rectangular_plate_modes(a, b, thickness, young, nu, rho, PlateBc::SimplySupported, 4, 4);
        let freqs: Vec<f64> = modes.iter().map(|m| m.2).collect();
        Self::from_bare_freqs(&freqs, fs)
    }

    /// Thin-ring bell flexural modes.
    #[must_use]
    pub fn bell(radius: f64, thickness: f64, young: f64, rho: f64, nu: f64, fs: f64) -> Self {
        let freqs = bell_modes_approx(radius, thickness, young, rho, nu, 8);
        Self::from_bare_freqs(&freqs, fs)
    }

    /// Wine-glass model from measured partial ratios of the (n,0) rim
    /// modes: 1, 2.32, 4.25, 6.63, 9.38.
    #[must_use]
    pub fn glass(f0: f64, fs: f64) -> Self {
        let ratios = [1.0, 2.32, 4.25, 6.63, 9.38];
        let freqs: Vec<f64> = ratios.iter().map(|r| r * f0).collect();
        Self::from_bare_freqs(&freqs, fs)
    }

    fn from_bare_freqs(freqs: &[f64], fs: f64) -> Self {
        let f1 = freqs.first().copied().unwrap_or(1.0).max(1.0);
        let modes: Vec<(f64, f64, f64)> = freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| (f, 3.0 * (f1 / f.max(f1)).sqrt(), 1.0 / (1.0 + i as f64)))
            .collect();
        Self::from_modes(&modes, fs)
    }

    /// Queue an arbitrary excitation signal.
    pub fn excite(&mut self, impulse: &[f64]) {
        self.excitation = impulse.to_vec();
        self.exc_pos = 0;
    }

    /// Strike with a raised-cosine impulse; `hardness` in (0..1], harder
    /// strikes are shorter (brighter).
    pub fn strike(&mut self, hardness: f64) {
        let h = hardness.clamp(0.05, 1.0);
        let w = (2.0 + (1.0 - h) * 1.5e-3 * self.fs) as usize;
        let pulse: Vec<f64> =
            (0..w).map(|i| 0.5 * (1.0 - (TWO_PI * (i as f64 + 0.5) / w as f64).cos())).collect();
        self.excite(&pulse);
    }

    /// Advance one output sample.
    pub fn next(&mut self) -> f64 {
        let x = if self.exc_pos < self.excitation.len() {
            let v = self.excitation[self.exc_pos];
            self.exc_pos += 1;
            v
        } else {
            0.0
        };
        self.resonators.iter_mut().map(|r| r.process(x)).sum()
    }
}

// --- Bowed string --------------------------------------------------------

/// Bowed string: two waveguide segments joined at the bow point with a
/// stick-slip friction curve producing Helmholtz motion.
pub struct BowedString {
    neck: Vec<f64>,
    bridge: Vec<f64>,
    npos: usize,
    bpos: usize,
    /// Bow speed (sets the string amplitude scale).
    pub bow_velocity: f64,
    /// Bow pressure 0..1 (sets the friction-curve slope).
    pub bow_force: f64,
    reflection: Biquad,
}

impl BowedString {
    /// New bowed string at `freq`; the bow sits at ~1/8 of the length.
    #[must_use]
    pub fn new(freq: f64, fs: f64) -> Self {
        let total = fs / freq - 2.0;
        let beta = 0.127;
        let nb = ((total * beta).round().max(1.0)) as usize;
        let nn = ((total - nb as f64).round().max(1.0)) as usize;
        let s = 0.35;
        Self {
            neck: vec![0.0; nn],
            bridge: vec![0.0; nb],
            npos: 0,
            bpos: 0,
            bow_velocity: 0.1,
            bow_force: 0.5,
            reflection: Biquad::from_coeffs(1.0 - s, 0.0, 0.0, -s, 0.0),
        }
    }

    /// Advance one sample of bridge output.
    pub fn next(&mut self) -> f64 {
        let bo = self.bridge[self.bpos];
        let no = self.neck[self.npos];
        let bridge_refl = -0.95 * self.reflection.process(bo);
        let nut_refl = -no;
        let string_vel = bridge_refl + nut_refl;
        let dv = self.bow_velocity - string_vel;
        let slope = 5.0 - 4.0 * self.bow_force.clamp(0.0, 1.0);
        let new_vel = dv * bow_friction(dv, slope);
        self.neck[self.npos] = bridge_refl + new_vel;
        self.bridge[self.bpos] = nut_refl + new_vel;
        self.npos = (self.npos + 1) % self.neck.len();
        self.bpos = (self.bpos + 1) % self.bridge.len();
        bo
    }
}

// --- Finite-difference membrane and plate --------------------------------

/// Circular drum head on a masked finite-difference grid, audio-rate.
pub struct Membrane2D {
    grid: WaveEquation2D,
    mask: Vec<bool>,
    substeps: usize,
    dt_sub: f64,
    fs: f64,
    pickup: (usize, usize),
    radius: f64,
}

impl Membrane2D {
    /// Circular drum of physical radius (m), membrane tension (N/m), and
    /// surface density (kg/m²) on a res × res grid.
    #[must_use]
    pub fn drum(radius_m: f64, tension: f64, density: f64, res: usize, fs: f64) -> Self {
        let c = (tension / density).sqrt();
        let dx = 2.0 * radius_m / (res - 1) as f64;
        let mut grid = WaveEquation2D::new(res, res, dx, dx, c);
        grid.set_damping(1.0);
        let dt = 1.0 / fs;
        let stable = grid.stable_dt();
        let substeps = (dt / (0.9 * stable)).ceil().max(1.0) as usize;
        let half = 0.5 * (res - 1) as f64;
        let mask = (0..res * res)
            .map(|idx| {
                let (i, j) = (idx % res, idx / res);
                let x = (i as f64 - half) * dx;
                let y = (j as f64 - half) * dx;
                (x * x + y * y).sqrt() > radius_m
            })
            .collect();
        Self {
            grid,
            mask,
            substeps,
            dt_sub: dt / substeps as f64,
            fs,
            pickup: (res / 2 + res / 8, res / 2),
            radius: radius_m,
        }
    }

    /// Strike at (x, y) in units of the radius (-1..1) with velocity
    /// `vel`, as a smooth velocity bump.
    pub fn strike(&mut self, x: f64, y: f64, vel: f64) {
        let res = self.grid.nx;
        let half = 0.5 * (res - 1) as f64;
        let cx = half + x.clamp(-0.95, 0.95) * half;
        let cy = half + y.clamp(-0.95, 0.95) * half;
        let sigma = self.radius / 6.0 / self.grid.dx;
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                if self.mask[idx] {
                    continue;
                }
                let r2 = (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2);
                self.grid.u_current[idx] += vel / self.fs * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    /// Advance one audio sample; returns the displacement at the pickup.
    pub fn next(&mut self) -> f64 {
        for _ in 0..self.substeps {
            self.grid.step(self.dt_sub);
            for (idx, &m) in self.mask.iter().enumerate() {
                if m {
                    self.grid.u_current[idx] = 0.0;
                }
            }
        }
        self.grid.u_current[self.pickup.1 * self.grid.nx + self.pickup.0]
    }
}

/// Simply supported Kirchhoff plate (u_tt = -κ² ∇⁴u) on a
/// finite-difference grid, audio-rate.
pub struct Plate2D {
    u: Vec<f64>,
    u_prev: Vec<f64>,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    kappa: f64,
    /// Viscous damping γ (s⁻¹).
    pub damping: f64,
    substeps: usize,
    dt_sub: f64,
    pickup: (usize, usize),
    fs: f64,
}

impl Plate2D {
    /// Plate of size a × b (m), thickness h, Young's modulus, density,
    /// and Poisson ratio on a res-wide grid.
    #[allow(clippy::too_many_arguments)] // physical parameter list
    #[must_use]
    pub fn new(
        a: f64,
        b: f64,
        thickness: f64,
        young: f64,
        rho: f64,
        nu: f64,
        res: usize,
        fs: f64,
    ) -> Self {
        let nx = res;
        let ny = ((res as f64 * b / a).round().max(4.0)) as usize;
        let dx = a / (nx - 1) as f64;
        let dy = b / (ny - 1) as f64;
        let d = young * thickness.powi(3) / (12.0 * (1.0 - nu * nu));
        let kappa = (d / (rho * thickness)).sqrt();
        let lap_max = 4.0 / (dx * dx) + 4.0 / (dy * dy);
        let dt_stable = 2.0 / (kappa * lap_max);
        let dt = 1.0 / fs;
        let substeps = (dt / (0.8 * dt_stable)).ceil().max(1.0) as usize;
        Self {
            u: vec![0.0; nx * ny],
            u_prev: vec![0.0; nx * ny],
            nx,
            ny,
            dx,
            dy,
            kappa,
            damping: 1.0,
            substeps,
            dt_sub: dt / substeps as f64,
            pickup: (nx / 3, ny / 3),
            fs,
        }
    }

    /// Strike at fractional position (0..1, 0..1) with velocity `vel`.
    pub fn strike(&mut self, x: f64, y: f64, vel: f64) {
        let cx = x.clamp(0.1, 0.9) * (self.nx - 1) as f64;
        let cy = y.clamp(0.1, 0.9) * (self.ny - 1) as f64;
        let sigma = (self.nx as f64 / 12.0).max(1.0);
        for j in 1..self.ny - 1 {
            for i in 1..self.nx - 1 {
                let r2 = (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2);
                self.u[j * self.nx + i] += vel / self.fs * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    fn laplacian(&self, f: &[f64]) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let (ix2, iy2) = (1.0 / (self.dx * self.dx), 1.0 / (self.dy * self.dy));
        let mut out = vec![0.0; nx * ny];
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let c = j * nx + i;
                out[c] = (f[c + 1] - 2.0 * f[c] + f[c - 1]) * ix2
                    + (f[c + nx] - 2.0 * f[c] + f[c - nx]) * iy2;
            }
        }
        out
    }

    /// Advance one audio sample; returns the displacement at the pickup.
    pub fn next(&mut self) -> f64 {
        for _ in 0..self.substeps {
            let lap = self.laplacian(&self.u);
            let bih = self.laplacian(&lap);
            let dt = self.dt_sub;
            let gdt = self.damping * dt;
            let k2dt2 = self.kappa * self.kappa * dt * dt;
            let mut u_next = vec![0.0; self.u.len()];
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    let c = j * self.nx + i;
                    u_next[c] = (2.0 * self.u[c] - (1.0 - gdt) * self.u_prev[c]
                        - k2dt2 * bih[c])
                        / (1.0 + gdt);
                }
            }
            self.u_prev.copy_from_slice(&self.u);
            self.u.copy_from_slice(&u_next);
        }
        self.u[self.pickup.1 * self.nx + self.pickup.0]
    }
}

// --- Mass-spring string (validation) -------------------------------------

/// Brute-force lumped mass-spring string with fixed ends, for validating
/// the waveguide against a direct Newtonian simulation.
pub struct MassSpringString {
    /// Node masses (kg).
    pub masses: Vec<f64>,
    /// Transverse displacements.
    pub positions: Vec<f64>,
    /// Transverse velocities.
    pub velocities: Vec<f64>,
    /// Inter-node transverse stiffness T/a (N/m).
    pub k: f64,
    /// Viscous damping per unit mass (s⁻¹).
    pub damping: f64,
    substeps: usize,
    dt_sub: f64,
    pickup: usize,
}

impl MassSpringString {
    /// String of `n` moving masses tuned so the continuum limit has
    /// fundamental `freq` (unit length, unit line density).
    #[must_use]
    pub fn new(freq: f64, n: usize, fs: f64) -> Self {
        let a = 1.0 / (n + 1) as f64;
        let c = 2.0 * freq; // L = 1, f1 = c / 2L
        let m = a; // mu = 1
        let k = c * c / a; // T/a with T = mu c^2
        let omega_max = 2.0 * (k / m).sqrt();
        let dt = 1.0 / fs;
        let substeps = (dt * omega_max / 1.5).ceil().max(1.0) as usize;
        Self {
            masses: vec![m; n],
            positions: vec![0.0; n],
            velocities: vec![0.0; n],
            k,
            damping: 0.0,
            substeps,
            dt_sub: dt / substeps as f64,
            pickup: n / 3,
        }
    }

    /// Triangular pluck peaked at fractional position `pos`.
    pub fn pluck(&mut self, pos: f64, amp: f64) {
        let n = self.positions.len();
        let p = pos.clamp(0.05, 0.95);
        for i in 0..n {
            let x = (i + 1) as f64 / (n + 1) as f64;
            self.positions[i] = amp * if x < p { x / p } else { (1.0 - x) / (1.0 - p) };
        }
        self.velocities.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Advance one audio sample (semi-implicit Euler, substepped for
    /// stability); returns the displacement at the pickup node.
    pub fn next(&mut self) -> f64 {
        let n = self.positions.len();
        for _ in 0..self.substeps {
            for i in 0..n {
                let left = if i > 0 { self.positions[i - 1] } else { 0.0 };
                let right = if i + 1 < n { self.positions[i + 1] } else { 0.0 };
                let f = self.k * (left - 2.0 * self.positions[i] + right)
                    - self.damping * self.masses[i] * self.velocities[i];
                self.velocities[i] += f / self.masses[i] * self.dt_sub;
            }
            for i in 0..n {
                self.positions[i] += self.velocities[i] * self.dt_sub;
            }
        }
        self.positions[self.pickup]
    }
}

// --- Excitation and coupling helpers -------------------------------------

/// One waveguide per band: (center frequency, T60 seconds) pairs, as used
/// in banded waveguide synthesis of stiff/inharmonic objects.
#[must_use]
pub fn banded_waveguide(freq: f64, bands: &[(f64, f64)], fs: f64) -> Vec<WaveguideString> {
    bands
        .iter()
        .map(|&(ratio, t60)| {
            let f = freq * ratio;
            let mut s = WaveguideString::new(f, fs);
            // Per-reflection loss from the band's T60.
            let period = 1.0 / f;
            s.damping = 10.0_f64.powf(-3.0 * period / t60.max(1e-3));
            s
        })
        .collect()
}

/// Commuted synthesis: the body impulse response is convolved into the
/// excitation and fed through the string, avoiding a body filter at
/// synthesis time.
pub fn commuted_synthesis(
    body_ir: &[f64],
    excitation: &[f64],
    string: &mut WaveguideString,
    n: usize,
) -> Vec<f64> {
    let drive = crate::audio::effects::convolution_reverb(excitation, body_ir);
    (0..n)
        .map(|t| {
            if t < drive.len() {
                string.inject(drive[t]);
            }
            string.next()
        })
        .collect()
}

/// Piano hammer-string contact: a hammer of mass `hammer_mass` (kg) with
/// initial velocity `hammer_vel` compresses a nonlinear felt spring
/// F = k ξ^p against the string. Returns the contact force history
/// (one sample per tick until separation); the string is excited in place.
pub fn hammer_string_interaction(
    string: &mut WaveguideString,
    hammer_mass: f64,
    hammer_vel: f64,
    stiffness_exp: f64,
    k: f64,
) -> Vec<f64> {
    let fs = string.fs;
    let dt = 1.0 / fs;
    let mut y_h = 0.0;
    let mut v_h = hammer_vel;
    let mut force = Vec::new();
    let max_steps = fs as usize; // 1 s guard
    for _ in 0..max_steps {
        let y_s = string.output_at(string.pluck_pos);
        let xi = y_h - y_s;
        if xi <= 0.0 && !force.is_empty() {
            break; // hammer separated
        }
        let f = if xi > 0.0 { k * xi.powf(stiffness_exp) } else { 0.0 };
        v_h -= f / hammer_mass * dt;
        y_h += v_h * dt;
        string.inject(f * dt);
        string.next();
        force.push(f);
        if v_h < 0.0 && xi <= 0.0 {
            break;
        }
    }
    force
}

/// Single-reed reflection coefficient as a function of the pressure
/// difference across the reed (STK-style linear table, clamped to ±1).
#[must_use]
pub fn reed_nonlinearity(delta_p: f64, stiffness: f64, closing_p: f64) -> f64 {
    (0.7 - stiffness * delta_p / closing_p).clamp(-1.0, 1.0)
}

/// Flute jet nonlinearity x - x³, clamped to ±1.
#[must_use]
pub fn jet_nonlinearity(x: f64) -> f64 {
    (x - x * x * x).clamp(-1.0, 1.0)
}

/// Brass lip valve: pressure-controlled transmission coefficient; the
/// lips open on positive mouth-bore pressure difference.
#[must_use]
pub fn lip_model(delta_p: f64, lip_tension: f64) -> f64 {
    if delta_p <= 0.0 {
        0.0
    } else {
        (delta_p * delta_p / (lip_tension + delta_p * delta_p)).clamp(0.0, 1.0)
    }
}

/// Kelly-Lochbaum piecewise-cylindrical vocal tract lattice.
pub struct KellyLochbaum {
    refl: Vec<f64>,
    fwd: Vec<f64>,
    bwd: Vec<f64>,
    /// Reflection at the glottis end (near +1).
    pub glottal_reflection: f64,
    /// Reflection at the lips (near -1); output is the transmitted part.
    pub lip_reflection: f64,
}

/// Build a Kelly-Lochbaum lattice from a tract area function (cm² or any
/// consistent unit); each section is one sample of travel at `fs`.
#[must_use]
pub fn vocal_tract(area_function: &[f64], _fs: f64) -> KellyLochbaum {
    let refl = area_function
        .windows(2)
        .map(|w| (w[0] - w[1]) / (w[0] + w[1]))
        .collect::<Vec<f64>>();
    KellyLochbaum {
        refl,
        fwd: vec![0.0; area_function.len()],
        bwd: vec![0.0; area_function.len()],
        glottal_reflection: 0.9,
        lip_reflection: -0.85,
    }
}

impl KellyLochbaum {
    /// Update the reflection coefficients from a new area function.
    pub fn set_areas(&mut self, area_function: &[f64]) {
        self.refl =
            area_function.windows(2).map(|w| (w[0] - w[1]) / (w[0] + w[1])).collect();
        self.fwd.resize(area_function.len(), 0.0);
        self.bwd.resize(area_function.len(), 0.0);
    }

    /// One sample: feed the glottal source in, return the lip output.
    pub fn next(&mut self, glottal: f64) -> f64 {
        let s = self.fwd.len();
        let mut fwd_next = vec![0.0; s];
        let mut bwd_next = vec![0.0; s];
        fwd_next[0] = glottal + self.glottal_reflection * self.bwd[0];
        for (i, &k) in self.refl.iter().enumerate() {
            let f = self.fwd[i];
            let b = self.bwd[i + 1];
            fwd_next[i + 1] = (1.0 + k) * f - k * b;
            bwd_next[i] = k * f + (1.0 - k) * b;
        }
        let out = (1.0 + self.lip_reflection) * self.fwd[s - 1];
        bwd_next[s - 1] = self.lip_reflection * self.fwd[s - 1];
        self.fwd = fwd_next;
        self.bwd = bwd_next;
        out
    }
}

/// Simplified Liljencrants-Fant glottal flow *derivative* pulse over one
/// period t ∈ [0, t0): exponentially growing sinusoid up to `te` (peak of
/// the sinusoid at `tp`), then an exponential return phase with time
/// constant `ta`.
#[must_use]
pub fn glottal_pulse_lf(t: f64, t0: f64, te: f64, tp: f64, ta: f64) -> f64 {
    let t = t.rem_euclid(t0);
    if t < te {
        // Growth rate chosen so the envelope rises ~e^2 over the open
        // phase; normalized so E(te) = -1, continuous with the return.
        let alpha = 2.0 / te;
        let norm = (alpha * te).exp() * (PI * te / tp).sin().abs().max(1e-6);
        (alpha * t).exp() * (PI * t / tp).sin() / norm
    } else {
        // Return phase: -E_e/(ε ta)(e^{-ε(t-te)} - e^{-ε(t0-te)}); solve
        // ε ta = 1 - e^{-ε(t0-te)} by fixed-point iteration.
        let tr = t0 - te;
        let mut eps = 1.0 / ta;
        for _ in 0..30 {
            eps = (1.0 - (-eps * tr).exp()) / ta;
        }
        let e_end = (-eps * tr).exp();
        -((-eps * (t - te)).exp() - e_end) / (eps * ta).max(1e-9)
    }
}

/// Rosenberg glottal flow pulse: raised-cosine rise over the first 2/3 of
/// the open phase, cosine fall over the last 1/3, zero when closed.
/// `phase` in [0, 1), `open_quotient` in (0, 1].
#[must_use]
pub fn rosenberg_pulse(phase: f64, open_quotient: f64) -> f64 {
    let p = phase.rem_euclid(1.0);
    let oq = open_quotient.clamp(0.05, 1.0);
    let t1 = 2.0 / 3.0 * oq;
    let t2 = oq - t1;
    if p < t1 {
        0.5 * (1.0 - (PI * p / t1).cos())
    } else if p < oq {
        ((PI / 2.0) * (p - t1) / t2).cos()
    } else {
        0.0
    }
}

/// Tension (N) needed for a string of `length` (m) and line density `mu`
/// (kg/m) to sound at `freq`: T = μ (2 L f)².
#[must_use]
pub fn string_tension_from_freq(freq: f64, length: f64, mu: f64) -> f64 {
    mu * (2.0 * length * freq).powi(2)
}

/// Piano-style stretched partials f_k = k f0 √(1 + B k²).
#[must_use]
pub fn inharmonic_partials(f0: f64, b: f64, n: usize) -> Vec<f64> {
    (1..=n)
        .map(|k| {
            let kf = k as f64;
            kf * f0 * (1.0 + b * kf * kf).sqrt()
        })
        .collect()
}

/// Couple strings through a shared bridge: each sample, every string
/// except the driving one at `excitation_idx` receives `coupling` times
/// the sum of the others' bridge outputs. The string at `excitation_idx`
/// should already be excited. Returns the summed bridge output over `n`
/// samples.
pub fn sympathetic_resonance(
    strings: &mut [WaveguideString],
    coupling: f64,
    excitation_idx: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut outs = vec![0.0; strings.len()];
    for _ in 0..n {
        for (i, s) in strings.iter_mut().enumerate() {
            outs[i] = s.next();
        }
        let total: f64 = outs.iter().sum();
        for (i, s) in strings.iter_mut().enumerate() {
            if i != excitation_idx {
                s.inject(coupling * (total - outs[i]));
            }
        }
        out.push(total);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::fft::rfft;

    /// Precise frequency estimate: peak bin of a Hann-windowed FFT plus
    /// the phase advance between two frames `hop` samples apart.
    fn measure_freq(x: &[f64], fs: f64) -> f64 {
        let n = 4096;
        let hop = 200;
        assert!(x.len() >= n + hop, "signal too short");
        let w: Vec<f64> =
            (0..n).map(|i| 0.5 * (1.0 - (TWO_PI * i as f64 / n as f64).cos())).collect();
        let seg1: Vec<f64> = (0..n).map(|i| x[i] * w[i]).collect();
        let seg2: Vec<f64> = (0..n).map(|i| x[i + hop] * w[i]).collect();
        let s1 = rfft(&seg1);
        let s2 = rfft(&seg2);
        let k = (2..s1.len() - 1)
            .max_by(|&a, &b| s1[a].norm().partial_cmp(&s1[b].norm()).unwrap())
            .unwrap();
        let dphi = (s2[k] * s1[k].conjugate()).arg();
        let expected = TWO_PI * k as f64 / n as f64 * hop as f64;
        let mut d = dphi - expected;
        d -= TWO_PI * (d / TWO_PI).round();
        (expected + d) / hop as f64 / TWO_PI * fs
    }

    fn cents(f: f64, target: f64) -> f64 {
        1200.0 * (f / target).log2()
    }

    #[test]
    fn test_waveguide_pitch_accuracy() {
        let fs = 48000.0;
        for &f0 in &[55.0, 110.0, 220.0, 440.0, 880.0, 1320.0, 1760.0] {
            let mut s = WaveguideString::new(f0, fs);
            s.damping = 0.9995;
            s.pluck(0.28, 0.7, 0.05);
            let x: Vec<f64> = (0..6000).map(|_| s.next()).collect();
            let f = measure_freq(&x[1000..], fs);
            let err = cents(f, f0);
            assert!(err.abs() < 2.0, "pitch {f0}: got {f} Hz ({err:.2} cents)");
        }
    }

    #[test]
    fn test_waveguide_decays_monotonically() {
        let fs = 48000.0;
        let mut s = WaveguideString::new(220.0, fs);
        s.damping = 0.98;
        s.pluck(0.3, 1.0, 0.05);
        let x: Vec<f64> = (0..48000).map(|_| s.next()).collect();
        let block = 4800;
        let rms: Vec<f64> = x
            .chunks(block)
            .map(|c| (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt())
            .collect();
        for w in rms.windows(2) {
            assert!(w[1] < w[0] * 1.02, "rms not decaying: {rms:?}");
        }
        assert!(rms[rms.len() - 1] < 0.05 * rms[0]);
    }

    #[test]
    fn test_waveguide_strike_bow_and_output_at() {
        let fs = 48000.0;
        let mut s = WaveguideString::new(220.0, fs);
        s.strike(0.2, 1.0);
        let e: f64 = (0..2000).map(|_| s.next().powi(2)).sum();
        assert!(e > 1e-6 && e.is_finite());
        let mid = s.output_at(0.5);
        assert!(mid.is_finite());
        // Bowing sustains the tone.
        let mut b = WaveguideString::new(220.0, fs);
        b.damping = 0.98;
        b.bow(0.6, 0.15, 0.13);
        let x: Vec<f64> = (0..24000).map(|_| b.next()).collect();
        let early: f64 = x[4000..8000].iter().map(|v| v * v).sum();
        let late: f64 = x[20000..24000].iter().map(|v| v * v).sum();
        assert!(late > 0.2 * early, "bowed tone died: {early} -> {late}");
    }

    #[test]
    fn test_clarinet_and_flute() {
        let fs = 48000.0;
        let mut cl = WaveguideTube::clarinet(200.0, fs);
        cl.set_breath(0.6);
        let x: Vec<f64> = (0..24000).map(|_| cl.next()).collect();
        let tail = &x[12000..];
        let rms = (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(rms > 0.05, "clarinet did not speak: rms {rms}");
        let mut d = tail.to_vec();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        d.iter_mut().for_each(|v| *v -= mean);
        let f = measure_freq(&d, fs);
        assert!((f / 200.0 - 1.0).abs() < 0.08, "clarinet pitch {f}");

        let mut fl = WaveguideTube::flute(440.0, fs);
        fl.set_breath(0.35);
        let y: Vec<f64> = (0..36000).map(|_| fl.next()).collect();
        let tail = &y[24000..];
        let rms = (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(rms > 0.01, "flute did not speak: rms {rms}");
        assert!(tail.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_modal_bar_matches_beam_ratios() {
        let fs = 48000.0;
        // Aluminum bar 30 cm x 3 cm x 1 cm.
        let bar = ModalSynth::bar(0.30, 0.03, 0.01, 69e9, 2700.0, fs);
        let reference = beam_modes(0.30, 69e9, 0.03 * 0.01_f64.powi(3) / 12.0, 2700.0,
            0.03 * 0.01, BeamBc::FreeFree, 4);
        assert!(bar.modes.len() >= 3);
        for (i, m) in bar.modes.iter().take(3).enumerate() {
            let ratio = m.0 / bar.modes[0].0;
            let expected = reference[i] / reference[0];
            assert!(
                (ratio - expected).abs() < 1e-9,
                "mode {i}: ratio {ratio} vs beam {expected}"
            );
        }
        // Rendered spectrum peaks at the first mode.
        let mut synth = ModalSynth::bar(0.30, 0.03, 0.01, 69e9, 2700.0, fs);
        synth.strike(0.9);
        let x: Vec<f64> = (0..6000).map(|_| synth.next()).collect();
        let f = measure_freq(&x[500..], fs);
        let f1 = synth.modes[0].0;
        assert!((f / f1 - 1.0).abs() < 0.02, "bar peak {f} vs mode {f1}");
    }

    #[test]
    fn test_modal_constructors() {
        let fs = 48000.0;
        let mut m = ModalSynth::membrane_circular(0.2, 1000.0, 0.3, fs);
        assert!(!m.modes.is_empty());
        m.strike(0.5);
        assert!((0..100).map(|_| m.next()).all(|v| v.is_finite()));
        let p = ModalSynth::plate(0.5, 0.4, 0.003, 200e9, 7800.0, 0.3, fs);
        assert!(!p.modes.is_empty());
        let b = ModalSynth::bell(0.1, 0.01, 100e9, 8600.0, 0.34, fs);
        assert!(!b.modes.is_empty());
        let g = ModalSynth::glass(600.0, fs);
        assert!((g.modes[1].0 / g.modes[0].0 - 2.32).abs() < 1e-9);
    }

    #[test]
    fn test_bowed_string_sustains_near_pitch() {
        let fs = 48000.0;
        let f0 = 220.0;
        let mut b = BowedString::new(f0, fs);
        b.bow_velocity = 0.12;
        b.bow_force = 0.6;
        let x: Vec<f64> = (0..36000).map(|_| b.next()).collect();
        let tail = &x[24000..];
        let rms = (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(rms > 0.01, "bowed string silent: {rms}");
        let f = measure_freq(tail, fs);
        assert!((f / f0 - 1.0).abs() < 0.1, "bowed pitch {f}");
    }

    #[test]
    fn test_membrane_fundamental() {
        let fs = 16000.0;
        let radius = 0.5;
        let tension = 10000.0;
        let sigma = 1.0;
        let mut drum = Membrane2D::drum(radius, tension, sigma, 64, fs);
        drum.strike(0.0, 0.0, 1.0);
        let x: Vec<f64> = (0..4400).map(|_| drum.next()).collect();
        let f = measure_freq(&x[100..], fs);
        let modes = circular_membrane_modes(radius, tension, sigma, 0, 1);
        let f01 = modes[0].2;
        assert!(
            (f / f01 - 1.0).abs() < 0.03,
            "membrane fundamental {f} vs analytic {f01}"
        );
    }

    #[test]
    fn test_plate_runs_and_decays() {
        let fs = 8000.0;
        let mut p = Plate2D::new(0.5, 0.5, 0.002, 69e9, 2700.0, 0.33, 24, fs);
        p.damping = 30.0;
        p.strike(0.4, 0.55, 1.0);
        let x: Vec<f64> = (0..4000).map(|_| p.next()).collect();
        assert!(x.iter().all(|v| v.is_finite()));
        let early: f64 = x[..1000].iter().map(|v| v * v).sum();
        let late: f64 = x[3000..].iter().map(|v| v * v).sum();
        assert!(early > 0.0);
        assert!(late < early, "plate not decaying: {early} -> {late}");
    }

    #[test]
    fn test_mass_spring_converges_to_waveguide() {
        let fs = 48000.0;
        let f0 = 220.0;
        let mut errs = Vec::new();
        for &n in &[8_usize, 32] {
            let mut ms = MassSpringString::new(f0, n, fs);
            ms.pluck(0.5, 0.5);
            let x: Vec<f64> = (0..6000).map(|_| ms.next()).collect();
            let f = measure_freq(&x, fs);
            errs.push((f / f0 - 1.0).abs());
        }
        assert!(errs[1] < errs[0], "no convergence: {errs:?}");
        assert!(errs[1] < 2e-3, "n=32 error too large: {errs:?}");
        // The waveguide itself is exact to within 2 cents (~0.12%).
        let mut wg = WaveguideString::new(f0, fs);
        wg.damping = 0.9995;
        wg.pluck(0.5, 0.5, 0.02);
        let y: Vec<f64> = (0..6000).map(|_| wg.next()).collect();
        let fw = measure_freq(&y[500..], fs);
        assert!((fw / f0 - 1.0).abs() < 2e-3);
    }

    #[test]
    fn test_excitation_helpers() {
        // Reed: monotonically decreasing in delta_p, clamped to [-1, 1].
        let mut prev = f64::INFINITY;
        for i in 0..50 {
            let dp = -3.0 + i as f64 * 0.25;
            let r = reed_nonlinearity(dp, 0.3, 1.0);
            assert!((-1.0..=1.0).contains(&r));
            assert!(r <= prev);
            prev = r;
        }
        assert_eq!(jet_nonlinearity(0.0), 0.0);
        assert!(jet_nonlinearity(0.5) > 0.0);
        assert!(jet_nonlinearity(5.0).abs() <= 1.0);
        assert_eq!(lip_model(-1.0, 1.0), 0.0);
        assert!(lip_model(2.0, 1.0) > lip_model(0.5, 1.0));
        assert!(lip_model(100.0, 1.0) <= 1.0);
        // Rosenberg pulse: in [0, 1], closed phase is exactly zero.
        for i in 0..100 {
            let v = rosenberg_pulse(i as f64 / 100.0, 0.6);
            assert!((0.0..=1.0).contains(&v));
        }
        assert_eq!(rosenberg_pulse(0.8, 0.6), 0.0);
        assert!(rosenberg_pulse(0.2, 0.6) > 0.5);
        // LF pulse: finite over a cycle, strongest negative spike near te.
        let (t0, te, tp, ta) = (0.01, 0.006, 0.004, 0.0004);
        let vals: Vec<f64> =
            (0..1000).map(|i| glottal_pulse_lf(i as f64 * t0 / 1000.0, t0, te, tp, ta)).collect();
        assert!(vals.iter().all(|v| v.is_finite()));
        let (imin, _) = vals
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let t_min = imin as f64 * t0 / 1000.0;
        assert!((t_min - te).abs() < 0.15 * t0, "LF minimum at {t_min}, te {te}");
        // String tension: 440 Hz on 0.65 m at 6 g/m.
        let t = string_tension_from_freq(440.0, 0.65, 0.006);
        assert!((t - 0.006 * (2.0 * 0.65 * 440.0_f64).powi(2)).abs() < 1e-9);
        // Inharmonicity: b = 0 gives exact harmonics, b > 0 stretches.
        let h = inharmonic_partials(100.0, 0.0, 5);
        assert!((h[4] - 500.0).abs() < 1e-9);
        let s = inharmonic_partials(100.0, 1e-3, 5);
        assert!(s[4] > 500.0);
        assert!(s[4] / s[0] > h[4] / h[0]);
    }

    #[test]
    fn test_hammer_commuted_banded_sympathetic() {
        let fs = 48000.0;
        // Hammer contact: finite force pulse, string rings afterwards.
        let mut s = WaveguideString::new(261.6, fs);
        let force = hammer_string_interaction(&mut s, 0.005, 2.0, 2.3, 1e8);
        assert!(!force.is_empty() && force.len() < fs as usize);
        assert!(force.iter().all(|f| f.is_finite() && *f >= 0.0));
        assert!(force.iter().cloned().fold(0.0_f64, f64::max) > 0.0);
        let e: f64 = (0..2000).map(|_| s.next().powi(2)).sum();
        assert!(e > 0.0 && e.is_finite());
        // Commuted synthesis: n samples, nonzero, finite.
        let ir: Vec<f64> = (0..200).map(|i| (-(i as f64) / 40.0).exp()).collect();
        let exc = vec![1.0, 0.5, 0.25];
        let mut s2 = WaveguideString::new(220.0, fs);
        let out = commuted_synthesis(&ir, &exc, &mut s2, 4000);
        assert_eq!(out.len(), 4000);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().map(|v| v * v).sum::<f64>() > 1e-9);
        // Banded waveguide: one string per band at the right frequency.
        let bands = [(1.0, 2.0), (2.756, 1.0), (5.404, 0.5)];
        let strings = banded_waveguide(200.0, &bands, fs);
        assert_eq!(strings.len(), 3);
        assert!((strings[1].freq() - 551.2).abs() < 1e-6);
        assert!(strings.iter().all(|s| s.damping < 1.0));
        // Sympathetic resonance: a silent unison string picks up energy.
        let mut pair = vec![WaveguideString::new(220.0, fs), WaveguideString::new(220.0, fs)];
        pair[0].pluck(0.3, 0.8, 0.05);
        sympathetic_resonance(&mut pair, 0.02, 0, 9600);
        let e1: f64 = (0..2000).map(|_| pair[1].next().powi(2)).sum();
        assert!(e1 > 1e-8, "sympathetic string stayed silent: {e1}");
    }

    #[test]
    fn test_kelly_lochbaum_tract() {
        let fs = 48000.0;
        // Rough /a/ area function, glottis to lips.
        let areas = [0.45, 0.2, 0.26, 0.21, 0.32, 0.3, 1.2, 2.6, 4.0, 5.0, 5.0, 4.6];
        let mut kl = vocal_tract(&areas, fs);
        let f0 = 110.0;
        let out: Vec<f64> = (0..4800)
            .map(|i| {
                let phase = f0 * i as f64 / fs;
                kl.next(rosenberg_pulse(phase, 0.6))
            })
            .collect();
        assert!(out.iter().all(|v| v.is_finite()));
        let e: f64 = out[1000..].iter().map(|v| v * v).sum();
        assert!(e > 1e-6, "tract output silent");
        kl.set_areas(&[1.0, 1.0, 1.0, 1.0]);
        assert!(kl.next(0.5).is_finite());
    }
}
