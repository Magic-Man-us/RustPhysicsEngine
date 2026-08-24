//! The damped harmonic oscillator m·x″ + c·x′ + k·x = F(t): closed-form
//! responses in every damping regime, frequency-domain descriptions, and
//! resonance measurement (Lorentzian fits, Q extraction).

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::numerical::rk4_step_vec;
use crate::optimization::levenberg_marquardt;

const TWO_PI: f64 = 2.0 * PI;

/// Damping regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Damping {
    Under,
    Critical,
    Over,
}

/// Mass-damper-spring oscillator m·x″ + c·x′ + k·x = F.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DampedOscillator {
    pub m: f64,
    pub c: f64,
    pub k: f64,
}

impl DampedOscillator {
    /// Undamped natural frequency ω₀ = √(k/m) (rad/s).
    #[must_use]
    pub fn natural_frequency(&self) -> f64 {
        (self.k / self.m).sqrt()
    }

    /// Damping ratio ζ = c/(2√(km)).
    #[must_use]
    pub fn damping_ratio(&self) -> f64 {
        self.c / (2.0 * (self.k * self.m).sqrt())
    }

    /// Quality factor Q = 1/(2ζ).
    #[must_use]
    pub fn q_factor(&self) -> f64 {
        1.0 / (2.0 * self.damping_ratio())
    }

    /// Damped oscillation frequency ω₀√(1−ζ²); None at or beyond
    /// critical damping.
    #[must_use]
    pub fn damped_frequency(&self) -> Option<f64> {
        let zeta = self.damping_ratio();
        if zeta < 1.0 {
            Some(self.natural_frequency() * (1.0 - zeta * zeta).sqrt())
        } else {
            None
        }
    }

    /// Damping regime (critical within 1 part in 10⁹ of ζ = 1).
    #[must_use]
    pub fn regime(&self) -> Damping {
        let zeta = self.damping_ratio();
        if (zeta - 1.0).abs() < 1e-9 {
            Damping::Critical
        } else if zeta < 1.0 {
            Damping::Under
        } else {
            Damping::Over
        }
    }

    /// Closed-form free response x(t) from initial position and
    /// velocity, valid in all three regimes.
    #[must_use]
    pub fn free_response(&self, x0: f64, v0: f64, t: f64) -> f64 {
        let w0 = self.natural_frequency();
        let zeta = self.damping_ratio();
        match self.regime() {
            Damping::Under => {
                let wd = w0 * (1.0 - zeta * zeta).sqrt();
                let decay = (-zeta * w0 * t).exp();
                decay * (x0 * (wd * t).cos() + (v0 + zeta * w0 * x0) / wd * (wd * t).sin())
            }
            Damping::Critical => {
                let decay = (-w0 * t).exp();
                decay * (x0 + (v0 + w0 * x0) * t)
            }
            Damping::Over => {
                let r = w0 * (zeta * zeta - 1.0).sqrt();
                let decay = (-zeta * w0 * t).exp();
                decay * (x0 * (r * t).cosh() + (v0 + zeta * w0 * x0) / r * (r * t).sinh())
            }
        }
    }

    /// Steady-state amplitude under forcing F = f₀·cos(ωt):
    /// X = (f₀/m)/√((ω₀²−ω²)² + (2ζω₀ω)²).
    #[must_use]
    pub fn steady_state_amplitude(&self, f0: f64, omega: f64) -> f64 {
        let w0 = self.natural_frequency();
        let zeta = self.damping_ratio();
        (f0 / self.m)
            / ((w0 * w0 - omega * omega).powi(2) + (2.0 * zeta * w0 * omega).powi(2)).sqrt()
    }

    /// Steady-state phase lag φ ∈ \[0, π\] of x behind the forcing:
    /// x = X·cos(ωt − φ).
    #[must_use]
    pub fn steady_state_phase(&self, omega: f64) -> f64 {
        let w0 = self.natural_frequency();
        let zeta = self.damping_ratio();
        (2.0 * zeta * w0 * omega).atan2(w0 * w0 - omega * omega)
    }

    /// Transfer function X(s)/F(s) = 1/(m·s² + c·s + k).
    #[must_use]
    pub fn transfer_function(&self, s: Complex) -> Complex {
        let den = s * s * Complex::new(self.m, 0.0)
            + s * Complex::new(self.c, 0.0)
            + Complex::new(self.k, 0.0);
        Complex::new(1.0, 0.0) / den
    }

    /// H(jω).
    #[must_use]
    pub fn frequency_response(&self, omega: f64) -> Complex {
        self.transfer_function(Complex::new(0.0, omega))
    }

    /// Frequency of maximum amplitude ω₀√(1−2ζ²); None for ζ ≥ 1/√2
    /// (no resonant peak).
    #[must_use]
    pub fn resonant_frequency(&self) -> Option<f64> {
        let zeta = self.damping_ratio();
        if zeta < std::f64::consts::FRAC_1_SQRT_2 {
            Some(self.natural_frequency() * (1.0 - 2.0 * zeta * zeta).sqrt())
        } else {
            None
        }
    }

    /// Half-power (full) bandwidth Δω = ω₀/Q = c/m (rad/s).
    #[must_use]
    pub fn bandwidth(&self) -> f64 {
        self.c / self.m
    }

    /// Impulse response h(t): free response to a unit impulse (velocity
    /// jump 1/m).
    #[must_use]
    pub fn impulse_response(&self, t: f64) -> f64 {
        self.free_response(0.0, 1.0 / self.m, t)
    }

    /// Response to a unit step force (final value 1/k).
    #[must_use]
    pub fn step_response(&self, t: f64) -> f64 {
        1.0 / self.k - self.free_response(1.0 / self.k, 0.0, t)
    }

    /// RK4 integration of the forced equation: samples of (t, x, v)
    /// every dt up to t_end.
    #[must_use]
    pub fn forced_response_numeric(
        &self,
        force: &dyn Fn(f64) -> f64,
        x0: f64,
        v0: f64,
        t_end: f64,
        dt: f64,
    ) -> Vec<(f64, f64, f64)> {
        let (m, c, k) = (self.m, self.c, self.k);
        let f = move |t: f64, y: &[f64]| -> Vec<f64> {
            vec![y[1], (force(t) - c * y[1] - k * y[0]) / m]
        };
        let mut y = vec![x0, v0];
        let mut t = 0.0;
        let mut out = vec![(0.0, x0, v0)];
        while t < t_end - 1e-12 {
            y = rk4_step_vec(&f, t, &y, dt);
            t += dt;
            out.push((t, y[0], y[1]));
        }
        out
    }

    /// Mechanical energy ½mv² + ½kx².
    #[must_use]
    pub fn energy(&self, x: f64, v: f64) -> f64 {
        0.5 * self.m * v * v + 0.5 * self.k * x * x
    }

    /// Time for the displacement envelope to decay to `fraction` of its
    /// initial value: t = −ln(fraction)/(ζω₀).
    ///
    /// # Panics
    /// Panics unless 0 < fraction < 1.
    #[must_use]
    pub fn decay_time(&self, fraction: f64) -> f64 {
        assert!(fraction > 0.0 && fraction < 1.0, "fraction must be in (0, 1)");
        -fraction.ln() / (self.damping_ratio() * self.natural_frequency())
    }

    /// Logarithmic decrement δ = 2πζ/√(1−ζ²) (underdamped).
    #[must_use]
    pub fn logarithmic_decrement(&self) -> f64 {
        let zeta = self.damping_ratio();
        TWO_PI * zeta / (1.0 - zeta * zeta).sqrt()
    }

    /// Build from natural frequency, quality factor, and mass.
    #[must_use]
    pub fn from_q(omega0: f64, q: f64, m: f64) -> Self {
        Self { m, c: m * omega0 / q, k: m * omega0 * omega0 }
    }
}

/// Magnitude (dB) and phase (degrees) of a transfer function over a
/// frequency grid.
#[must_use]
pub fn bode_plot(tf: &dyn Fn(Complex) -> Complex, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut mags = Vec::with_capacity(omega.len());
    let mut phases = Vec::with_capacity(omega.len());
    for &w in omega {
        let h = tf(Complex::new(0.0, w));
        mags.push(20.0 * h.norm().log10());
        phases.push(h.arg() * 180.0 / PI);
    }
    (mags, phases)
}

/// Nyquist locus H(jω) over a frequency grid.
#[must_use]
pub fn nyquist_plot(tf: &dyn Fn(Complex) -> Complex, omega: &[f64]) -> Vec<Complex> {
    omega.iter().map(|&w| tf(Complex::new(0.0, w))).collect()
}

/// Area-normalized Lorentzian lineshape
/// (1/π)·(γ/2)/((ω−ω₀)² + (γ/2)²).
#[must_use]
pub fn lorentzian(omega: f64, omega0: f64, gamma: f64) -> f64 {
    let hg = gamma / 2.0;
    hg / (PI * ((omega - omega0).powi(2) + hg * hg))
}

/// Fit y(ω) ≈ A·(γ/2)²/((ω−ω₀)² + (γ/2)²) (peak-amplitude Lorentzian)
/// by Levenberg-Marquardt; returns (ω₀, γ, A).
///
/// # Panics
/// Panics if the fit fails to converge or fewer than 4 points are given.
#[must_use]
pub fn lorentzian_fit(omega: &[f64], y: &[f64]) -> (f64, f64, f64) {
    assert!(omega.len() >= 4 && omega.len() == y.len(), "need >= 4 (omega, y) points");
    // Initial guesses from the data.
    let (imax, &ymax) = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let w_peak = omega[imax];
    // Half-max width estimate.
    let half = ymax / 2.0;
    let mut lo = omega[0];
    let mut hi = *omega.last().unwrap();
    for i in (0..imax).rev() {
        if y[i] < half {
            lo = omega[i];
            break;
        }
    }
    for i in imax..y.len() {
        if y[i] < half {
            hi = omega[i];
            break;
        }
    }
    let gamma0 = (hi - lo).max(1e-6);
    let om = omega.to_vec();
    let yv = y.to_vec();
    let resid = move |p: &[f64]| -> Vec<f64> {
        let (w0, g, a) = (p[0], p[1], p[2]);
        om.iter()
            .zip(&yv)
            .map(|(&w, &yy)| {
                let hg = g / 2.0;
                a * hg * hg / ((w - w0).powi(2) + hg * hg) - yy
            })
            .collect()
    };
    let fit = levenberg_marquardt(&resid, None, &[w_peak, gamma0, ymax], 1e-12, 500)
        .expect("Lorentzian fit failed");
    (fit.params[0], fit.params[1].abs(), fit.params[2])
}

/// Estimate (f₀ Hz, Q) from a free ring-down record: frequency from
/// interpolated zero crossings, decay rate from a log-linear fit to the
/// rectified peaks.
///
/// # Panics
/// Panics if the record has fewer than 4 zero crossings.
#[must_use]
pub fn q_from_ringdown(x: &[f64], fs: f64) -> (f64, f64) {
    let times = crate::dsp::phase::zero_crossing_times(x, fs);
    assert!(times.len() >= 4, "ring-down too short");
    // Every crossing is half a period.
    let spans = times.len() - 1;
    let f0 = spans as f64 / (2.0 * (times[times.len() - 1] - times[0]));
    // Peak magnitudes between crossings → exponential fit.
    let mut ts = Vec::new();
    let mut logs = Vec::new();
    for w in times.windows(2) {
        let i0 = (w[0] * fs).ceil() as usize;
        let i1 = ((w[1] * fs).floor() as usize).min(x.len() - 1);
        if i1 <= i0 {
            continue;
        }
        let (imax, peak) = x[i0..=i1]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if peak > 0.0 {
            ts.push((i0 + imax) as f64 / fs);
            logs.push(peak.ln());
        }
    }
    // Least-squares slope of ln|peak| vs t = −σ.
    let n = ts.len() as f64;
    let sx: f64 = ts.iter().sum();
    let sy: f64 = logs.iter().sum();
    let sxx: f64 = ts.iter().map(|v| v * v).sum();
    let sxy: f64 = ts.iter().zip(&logs).map(|(a, b)| a * b).sum();
    let sigma = -(n * sxy - sx * sy) / (n * sxx - sx * sx);
    let q = PI * f0 / sigma;
    (f0, q)
}

/// Estimate (f₀, Q) from a power spectrum by the −3 dB method with
/// linear interpolation of the half-power crossings.
///
/// # Panics
/// Panics on an empty spectrum.
#[must_use]
pub fn q_from_spectrum(f: &[f64], psd: &[f64]) -> (f64, f64) {
    assert!(!f.is_empty() && f.len() == psd.len(), "need matching (f, psd)");
    let (imax, &pmax) = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let half = pmax / 2.0;
    let mut f_lo = f[0];
    for i in (1..=imax).rev() {
        if psd[i - 1] < half {
            let t = (half - psd[i - 1]) / (psd[i] - psd[i - 1]);
            f_lo = f[i - 1] + t * (f[i] - f[i - 1]);
            break;
        }
    }
    let mut f_hi = f[f.len() - 1];
    for i in imax..f.len() - 1 {
        if psd[i + 1] < half {
            let t = (psd[i] - half) / (psd[i] - psd[i + 1]);
            f_hi = f[i] + t * (f[i + 1] - f[i]);
            break;
        }
    }
    let f0 = f[imax];
    (f0, f0 / (f_hi - f_lo))
}

/// Steady-state amplitude of the oscillator (unit force) at each ω.
#[must_use]
pub fn resonance_curve(osc: &DampedOscillator, omega: &[f64]) -> Vec<f64> {
    omega.iter().map(|&w| osc.steady_state_amplitude(1.0, w)).collect()
}

/// Base-excitation transmissibility at frequency ratio r = ω/ω₀:
/// √((1+(2ζr)²)/((1−r²)² + (2ζr)²)).
#[must_use]
pub fn transmissibility(omega_ratio: f64, zeta: f64) -> f64 {
    let r2 = omega_ratio * omega_ratio;
    let d = 2.0 * zeta * omega_ratio;
    ((1.0 + d * d) / ((1.0 - r2).powi(2) + d * d)).sqrt()
}

/// Combined quality factor of independent loss channels:
/// 1/Q = Σ 1/Qᵢ.
#[must_use]
pub fn quality_factor_combined(qs: &[f64]) -> f64 {
    1.0 / qs.iter().map(|q| 1.0 / q).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_free_response_matches_rk4_all_regimes() {
        for &(c, _regime) in &[(0.4, "under"), (4.0, "critical"), (9.0, "over")] {
            let osc = DampedOscillator { m: 1.0, c, k: 4.0 };
            let traj = osc.forced_response_numeric(&|_| 0.0, 1.0, -0.5, 5.0, 0.0005);
            for &(t, x, _) in traj.iter().step_by(500) {
                let exact = osc.free_response(1.0, -0.5, t);
                assert!(approx(x, exact, 1e-6), "c={c}, t={t}: {x} vs {exact}");
            }
        }
    }

    #[test]
    fn test_basic_quantities() {
        let osc = DampedOscillator { m: 2.0, c: 0.8, k: 50.0 };
        assert!(approx(osc.natural_frequency(), 5.0, 1e-12));
        assert!(approx(osc.damping_ratio(), 0.8 / 20.0, 1e-12));
        assert!(approx(osc.q_factor(), 12.5, 1e-9));
        assert_eq!(osc.regime(), Damping::Under);
        let osc2 = DampedOscillator::from_q(5.0, 12.5, 2.0);
        assert!(approx(osc2.c, 0.8, 1e-12) && approx(osc2.k, 50.0, 1e-12));
        assert!(osc.damped_frequency().unwrap() < 5.0);
        assert!(approx(osc.bandwidth(), 0.4, 1e-12));
    }

    #[test]
    fn test_steady_state_peak_at_resonant_frequency() {
        let osc = DampedOscillator { m: 1.0, c: 0.3, k: 9.0 };
        let wr = osc.resonant_frequency().unwrap();
        let a_res = osc.steady_state_amplitude(1.0, wr);
        for &dw in &[-0.05, 0.05] {
            assert!(osc.steady_state_amplitude(1.0, wr + dw) < a_res);
        }
        // Phase: ~0 well below, π/2 at ω0, ~π well above.
        assert!(osc.steady_state_phase(0.01) < 0.01);
        assert!(approx(osc.steady_state_phase(3.0), PI / 2.0, 1e-9));
        assert!(osc.steady_state_phase(30.0) > 3.0);
        // |H(jω)|·f0/m identity.
        let w = 2.7;
        assert!(approx(
            osc.frequency_response(w).norm(),
            osc.steady_state_amplitude(1.0, w),
            1e-12
        ));
    }

    #[test]
    fn test_energy_decays_at_c_over_m() {
        // Lightly damped: E(t) ≈ E0·exp(−c·t/m).
        let osc = DampedOscillator { m: 1.0, c: 0.05, k: 100.0 };
        let traj = osc.forced_response_numeric(&|_| 0.0, 1.0, 0.0, 8.0, 0.0005);
        let e0 = osc.energy(1.0, 0.0);
        for &(t, x, v) in traj.iter().skip(4000).step_by(4000) {
            let e = osc.energy(x, v);
            let expect = e0 * (-osc.c * t / osc.m).exp();
            assert!((e - expect).abs() / expect < 0.03, "t={t}: {e} vs {expect}");
        }
    }

    #[test]
    fn test_impulse_step_and_decrement() {
        let osc = DampedOscillator { m: 1.0, c: 0.2, k: 25.0 };
        // Impulse response initial slope = 1/m.
        let dt = 1e-6;
        assert!(approx(osc.impulse_response(dt) / dt, 1.0, 1e-3));
        // Step response settles at 1/k.
        assert!(approx(osc.step_response(200.0), 1.0 / 25.0, 1e-9));
        // Logarithmic decrement matches successive peak ratio.
        let wd = osc.damped_frequency().unwrap();
        let t1 = 0.7;
        let x1 = osc.free_response(1.0, 0.0, t1);
        let x2 = osc.free_response(1.0, 0.0, t1 + TWO_PI / wd);
        assert!(approx((x1 / x2).ln(), osc.logarithmic_decrement(), 1e-9));
        // Decay time.
        let tf = osc.decay_time(0.01);
        assert!(approx((-osc.damping_ratio() * osc.natural_frequency() * tf).exp(), 0.01, 1e-12));
    }

    #[test]
    fn test_bode_and_nyquist() {
        let osc = DampedOscillator { m: 1.0, c: 0.5, k: 4.0 };
        let omega = [0.01, 2.0, 100.0];
        let tf = move |s: Complex| osc.transfer_function(s);
        let (mag, phase) = bode_plot(&tf, &omega);
        assert!(approx(mag[0], 20.0 * 0.25_f64.log10(), 0.01)); // 1/k at DC
        assert!(approx(phase[1], -90.0, 1.0)); // −90° at ω0
        assert!(phase[2] < -170.0);
        let nyq = nyquist_plot(&tf, &omega);
        assert!(nyq[1].im < 0.0 && nyq[1].re.abs() < 0.01);
    }

    #[test]
    fn test_lorentzian_and_fit() {
        // Area of the normalized Lorentzian ≈ 1.
        let (w0, g) = (10.0, 0.5);
        let mut area = 0.0;
        let dw = 0.001;
        let mut w = -200.0;
        while w < 220.0 {
            area += lorentzian(w, w0, g) * dw;
            w += dw;
        }
        assert!(approx(area, 1.0, 2e-3), "area {area}");
        // Fit recovers parameters.
        let omega: Vec<f64> = (0..400).map(|i| 8.0 + 4.0 * i as f64 / 400.0).collect();
        let y: Vec<f64> = omega
            .iter()
            .map(|&w| 3.0 * (g / 2.0) * (g / 2.0) / ((w - w0).powi(2) + (g / 2.0) * (g / 2.0)))
            .collect();
        let (w0f, gf, af) = lorentzian_fit(&omega, &y);
        assert!(approx(w0f, w0, 1e-6) && approx(gf, g, 1e-6) && approx(af, 3.0, 1e-6));
    }

    #[test]
    fn test_q_from_ringdown_and_spectrum() {
        let osc = DampedOscillator::from_q(TWO_PI * 50.0, 40.0, 1.0);
        let fs = 5000.0;
        let n = 20000;
        let x: Vec<f64> = (0..n).map(|i| osc.free_response(1.0, 0.0, i as f64 / fs)).collect();
        let (f0, q) = q_from_ringdown(&x, fs);
        assert!(approx(f0, 50.0, 0.3), "f0 {f0}");
        assert!((q - 40.0).abs() / 40.0 < 0.02, "Q {q}");
        // Spectrum method on the analytic resonance curve.
        let freqs: Vec<f64> = (0..4000).map(|i| 40.0 + 20.0 * i as f64 / 4000.0).collect();
        let psd: Vec<f64> = freqs
            .iter()
            .map(|&f| osc.steady_state_amplitude(1.0, TWO_PI * f).powi(2))
            .collect();
        let (f0s, qs) = q_from_spectrum(&freqs, &psd);
        assert!(approx(f0s, 50.0, 0.1));
        assert!((qs - 40.0).abs() / 40.0 < 0.05, "spectrum Q {qs}");
    }

    #[test]
    fn test_resonance_curve_peak_and_sharpening() {
        let osc = DampedOscillator { m: 2.0, c: 0.6, k: 50.0 };
        let w0 = osc.natural_frequency();
        let zeta = osc.damping_ratio();
        let omega: Vec<f64> = (0..=4000).map(|i| 0.01 + 12.0 * i as f64 / 4000.0).collect();
        let curve = resonance_curve(&osc, &omega);
        assert_eq!(curve.len(), omega.len());
        // Each point is the unit-force steady-state amplitude.
        for (i, &w) in omega.iter().enumerate() {
            assert!(approx(curve[i], osc.steady_state_amplitude(1.0, w), 1e-15));
        }
        // Static (ω → 0) deflection is 1/k.
        assert!(approx(curve[0], 1.0 / osc.k, 1e-6), "static {}", curve[0]);
        // The peak sits at ω₀√(1−2ζ²), not at ω₀.
        let wr = osc.resonant_frequency().unwrap();
        let (imax, &peak) = curve
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            (omega[imax] - wr).abs() < 2.0 * (omega[1] - omega[0]),
            "peak at {} vs ω_r {wr}",
            omega[imax]
        );
        // Peak height has the closed form 1/(k·2ζ√(1−ζ²)).
        let expect = 1.0 / (osc.k * 2.0 * zeta * (1.0 - zeta * zeta).sqrt());
        assert!(approx(peak, expect, 1e-4 * expect), "peak {peak} vs {expect}");
        // Evaluated exactly at ω_r it matches to full precision.
        let at_wr = resonance_curve(&osc, &[wr])[0];
        assert!(approx(at_wr, expect, 1e-12 * expect));
        assert!(at_wr >= peak, "grid peak should not beat the analytic one");
        // Monotone rise then fall around the peak.
        for i in 1..imax {
            assert!(curve[i] > curve[i - 1], "not rising at {i}");
        }
        for i in imax + 1..curve.len() {
            assert!(curve[i] < curve[i - 1], "not falling at {i}");
        }

        // Lighter damping means a taller, sharper peak: the height scales
        // as 1/(2ζ√(1−ζ²)) and the −3 dB width narrows as ω₀/Q.
        let mut last_peak = 0.0_f64;
        let mut last_width = f64::MAX;
        for &c in &[1.2_f64, 0.6, 0.3, 0.15] {
            let o = DampedOscillator { m: 2.0, c, k: 50.0 };
            let z = o.damping_ratio();
            let curve = resonance_curve(&o, &omega);
            let p = curve.iter().cloned().fold(f64::MIN, f64::max);
            let want = 1.0 / (o.k * 2.0 * z * (1.0 - z * z).sqrt());
            assert!(approx(p, want, 2e-3 * want), "c = {c}: peak {p} vs {want}");
            assert!(p > last_peak, "peak did not grow as damping fell (c = {c})");
            last_peak = p;
            // Half-power width from the curve (as a power spectrum)
            // recovers Q = 1/(2ζ) through the independent −3 dB estimator.
            let psd: Vec<f64> = curve.iter().map(|v| v * v).collect();
            let (f_peak, q_est) = q_from_spectrum(&omega, &psd);
            assert!(
                (q_est - o.q_factor()).abs() / o.q_factor() < 0.05,
                "c = {c}: Q {q_est} vs {}",
                o.q_factor()
            );
            assert!((f_peak - o.resonant_frequency().unwrap()).abs() < 0.05);
            let width = f_peak / q_est;
            assert!(width < last_width, "resonance did not sharpen at c = {c}");
            last_width = width;
        }

        // Above the half-power damping ζ = 1/√2 there is no peak at all:
        // the curve falls monotonically from its static value.
        let heavy = DampedOscillator { m: 2.0, c: 30.0, k: 50.0 };
        assert!(heavy.damping_ratio() > std::f64::consts::FRAC_1_SQRT_2);
        assert!(heavy.resonant_frequency().is_none());
        let flat = resonance_curve(&heavy, &omega);
        for i in 1..flat.len() {
            assert!(flat[i] < flat[i - 1], "overdamped curve rose at {i}");
        }
        assert!(approx(flat[0], 1.0 / heavy.k, 1e-6));
        // An empty grid gives an empty curve.
        assert!(resonance_curve(&osc, &[]).is_empty());
        // Far above resonance the response rolls off as 1/(mω²).
        let high = resonance_curve(&osc, &[100.0 * w0])[0];
        assert!(
            approx(high, 1.0 / (osc.m * (100.0 * w0).powi(2)), 1e-3 * high),
            "mass-controlled asymptote {high}"
        );
    }

    #[test]
    fn test_transmissibility_and_combined_q() {
        // At r = √2 every ζ gives T = 1.
        for &z in &[0.05, 0.2, 0.7] {
            assert!(approx(transmissibility(std::f64::consts::SQRT_2, z), 1.0, 1e-12));
        }
        // Below √2 amplification, above isolation (small ζ).
        assert!(transmissibility(1.0, 0.1) > 1.0);
        assert!(transmissibility(3.0, 0.1) < 1.0);
        assert!(approx(quality_factor_combined(&[100.0, 100.0]), 50.0, 1e-12));
    }
}
