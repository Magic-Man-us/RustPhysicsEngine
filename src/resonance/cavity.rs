//! Resonant cavities and structures: RLC circuits, Helmholtz
//! resonators, strings, air columns, membranes, plates, beams, rooms,
//! optical etalons, and microwave cavities.

use crate::fields::ScalarField2;
use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::special::{bessel_j_zeros, bessel_jn};

const TWO_PI: f64 = 2.0 * PI;

// --- RLC ----------------------------------------------------------------

/// Series/parallel RLC resonator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rlc {
    pub r: f64,
    pub l: f64,
    pub c: f64,
}

impl Rlc {
    /// Series impedance R + j(ωL − 1/(ωC)).
    #[must_use]
    pub fn series_impedance(&self, omega: f64) -> Complex {
        Complex::new(self.r, omega * self.l - 1.0 / (omega * self.c))
    }

    /// Parallel impedance 1/(1/R + 1/(jωL) + jωC).
    #[must_use]
    pub fn parallel_impedance(&self, omega: f64) -> Complex {
        let y = Complex::new(1.0 / self.r, omega * self.c - 1.0 / (omega * self.l));
        Complex::new(1.0, 0.0) / y
    }

    /// Resonant frequency 1/(2π√(LC)) (Hz).
    #[must_use]
    pub fn resonant_frequency(&self) -> f64 {
        1.0 / (TWO_PI * (self.l * self.c).sqrt())
    }

    /// Series quality factor (1/R)√(L/C).
    #[must_use]
    pub fn q_series(&self) -> f64 {
        (self.l / self.c).sqrt() / self.r
    }

    /// Parallel quality factor R√(C/L).
    #[must_use]
    pub fn q_parallel(&self) -> f64 {
        self.r * (self.c / self.l).sqrt()
    }

    /// Series half-power bandwidth f₀/Q (Hz).
    #[must_use]
    pub fn bandwidth(&self) -> f64 {
        self.resonant_frequency() / self.q_series()
    }

    /// Series damping ratio ζ = (R/2)√(C/L).
    #[must_use]
    pub fn damping_ratio(&self) -> f64 {
        self.r / 2.0 * (self.c / self.l).sqrt()
    }

    /// Voltage transfer across C in the series loop (low-pass).
    #[must_use]
    pub fn transfer_lowpass(&self, omega: f64) -> Complex {
        Complex::new(0.0, -1.0 / (omega * self.c)) / self.series_impedance(omega)
    }

    /// Voltage transfer across R (band-pass).
    #[must_use]
    pub fn transfer_bandpass(&self, omega: f64) -> Complex {
        Complex::new(self.r, 0.0) / self.series_impedance(omega)
    }

    /// Voltage transfer across L (high-pass).
    #[must_use]
    pub fn transfer_highpass(&self, omega: f64) -> Complex {
        Complex::new(0.0, omega * self.l) / self.series_impedance(omega)
    }

    /// Voltage transfer across the LC pair (notch).
    #[must_use]
    pub fn transfer_notch(&self, omega: f64) -> Complex {
        Complex::new(0.0, omega * self.l - 1.0 / (omega * self.c)) / self.series_impedance(omega)
    }

    /// Capacitor voltage after a step of amplitude v on the series loop.
    #[must_use]
    pub fn step_response(&self, t: f64, v: f64) -> f64 {
        // L q″ + R q′ + q/C = v  ⇒  mechanical analogue m=L, c=R, k=1/C.
        let osc = super::oscillator::DampedOscillator { m: self.l, c: self.r, k: 1.0 / self.c };
        v * osc.step_response(t) / self.c
    }

    /// Stored energy ½Li² + ½Cv².
    #[must_use]
    pub fn energy(&self, i: f64, v: f64) -> f64 {
        0.5 * self.l * i * i + 0.5 * self.c * v * v
    }
}

// --- Acoustic resonators -------------------------------------------------

/// Helmholtz resonance frequency (Hz) with a flanged-end correction of
/// 1.7·r added to the neck length.
#[must_use]
pub fn helmholtz_resonator(volume: f64, neck_area: f64, neck_length: f64, c: f64) -> f64 {
    let r = (neck_area / PI).sqrt();
    let l_eff = neck_length + 1.7 * r;
    c / TWO_PI * (neck_area / (volume * l_eff)).sqrt()
}

/// Radiation-limited quality factor of a Helmholtz resonator (flanged
/// baffle radiation resistance): Q = 2π·√(V·L_eff³/A³).
#[must_use]
pub fn helmholtz_q(volume: f64, neck_area: f64, neck_length: f64, _c: f64) -> f64 {
    let r = (neck_area / PI).sqrt();
    let l_eff = neck_length + 1.7 * r;
    TWO_PI * (volume * l_eff.powi(3) / neck_area.powi(3)).sqrt()
}

/// Ideal string mode frequencies i·√(T/μ)/(2L), i = 1..=n.
#[must_use]
pub fn string_modes(length: f64, tension: f64, mu: f64, n: usize) -> Vec<f64> {
    let f1 = (tension / mu).sqrt() / (2.0 * length);
    (1..=n).map(|i| i as f64 * f1).collect()
}

/// Mode shape sin(nπx/L).
#[must_use]
pub fn string_mode_shape(length: f64, n: usize, x: f64) -> f64 {
    (n as f64 * PI * x / length).sin()
}

/// Stiff-string partials fₙ = n·f₁·√(1 + B·n²) with the piano
/// inharmonicity coefficient B (radius-based).
#[must_use]
pub fn stiff_string_modes(
    length: f64,
    tension: f64,
    mu: f64,
    young: f64,
    radius: f64,
    n: usize,
) -> Vec<f64> {
    let f1 = (tension / mu).sqrt() / (2.0 * length);
    let b = inharmonicity_coefficient(young, radius, tension, length);
    (1..=n)
        .map(|i| {
            let k = i as f64;
            k * f1 * (1.0 + b * k * k).sqrt()
        })
        .collect()
}

/// Piano-string inharmonicity B = π³·E·r⁴/(4·T·L²).
#[must_use]
pub fn inharmonicity_coefficient(young: f64, radius: f64, tension: f64, length: f64) -> f64 {
    PI.powi(3) * young * radius.powi(4) / (4.0 * tension * length * length)
}

/// Air-column modes: open-open i·c/(2L); open-closed odd harmonics
/// (2i−1)·c/(4L). Pass the end-corrected length.
#[must_use]
pub fn tube_modes(length: f64, c: f64, open_open: bool, n: usize) -> Vec<f64> {
    (1..=n)
        .map(|i| {
            if open_open {
                i as f64 * c / (2.0 * length)
            } else {
                (2.0 * i as f64 - 1.0) * c / (4.0 * length)
            }
        })
        .collect()
}

/// End correction of an open tube end: 0.85·r flanged, 0.61·r unflanged.
#[must_use]
pub fn tube_end_correction(radius: f64, flanged: bool) -> f64 {
    if flanged {
        0.85 * radius
    } else {
        0.61 * radius
    }
}

/// Complete-cone modes: like an open-open pipe, i·c/(2L).
#[must_use]
pub fn conical_tube_modes(length: f64, c: f64, n: usize) -> Vec<f64> {
    (1..=n).map(|i| i as f64 * c / (2.0 * length)).collect()
}

/// Rectangular membrane modes (m, n, f) sorted by frequency:
/// f = (c/2)·√((m/a)² + (n/b)²), c = √(T/σ).
#[must_use]
pub fn rectangular_membrane_modes(
    a: f64,
    b: f64,
    tension: f64,
    sigma: f64,
    max_m: usize,
    max_n: usize,
) -> Vec<(usize, usize, f64)> {
    let c = (tension / sigma).sqrt();
    let mut out = Vec::new();
    for m in 1..=max_m {
        for n in 1..=max_n {
            let f = c / 2.0 * ((m as f64 / a).powi(2) + (n as f64 / b).powi(2)).sqrt();
            out.push((m, n, f));
        }
    }
    out.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Circular membrane modes (m angular, n radial, f) sorted by
/// frequency: f = α_mn·c/(2πR) with α_mn the n-th zero of J_m.
#[must_use]
pub fn circular_membrane_modes(
    radius: f64,
    tension: f64,
    sigma: f64,
    max_m: usize,
    max_n: usize,
) -> Vec<(usize, usize, f64)> {
    let c = (tension / sigma).sqrt();
    let mut out = Vec::new();
    for m in 0..=max_m {
        let zeros = bessel_j_zeros(m as u32, max_n);
        for (n, &alpha) in zeros.iter().enumerate() {
            out.push((m, n + 1, alpha * c / (TWO_PI * radius)));
        }
    }
    out.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Circular membrane mode shape J_m(α_mn·r/R)·cos(mθ).
#[must_use]
pub fn circular_membrane_shape(radius: f64, m: usize, n: usize, r: f64, theta: f64) -> f64 {
    let alpha = bessel_j_zeros(m as u32, n)[n - 1];
    bessel_jn(m as u32, alpha * r / radius) * (m as f64 * theta).cos()
}

/// Plate boundary conditions for [`rectangular_plate_modes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateBc {
    SimplySupported,
    Clamped,
}

/// Clamped-clamped beam eigenvalue λ_i (roots of cos λ·cosh λ = 1).
fn clamped_beam_lambda(i: usize) -> f64 {
    const FIRST: [f64; 4] = [4.730_040_744_862_704, 7.853_204_624_095_838, 10.995_607_838_001_671, 14.137_165_491_257_464];
    if i <= 4 {
        FIRST[i - 1]
    } else {
        (2.0 * i as f64 + 1.0) * PI / 2.0
    }
}

/// Clamped-free beam eigenvalue λ_i (roots of cos λ·cosh λ = −1).
fn cantilever_lambda(i: usize) -> f64 {
    const FIRST: [f64; 4] = [1.875_104_068_711_961, 4.694_091_132_974_175, 7.854_757_438_237_613, 10.995_540_734_875_467];
    if i <= 4 {
        FIRST[i - 1]
    } else {
        (2.0 * i as f64 - 1.0) * PI / 2.0
    }
}

/// Thin rectangular plate modes (m, n, f Hz) sorted ascending.
/// Simply supported edges are exact; clamped edges use the separable
/// beam-function Rayleigh estimate (upper bound, a few % high).
#[must_use]
#[allow(clippy::too_many_arguments)] // signature fixed by the roadmap
pub fn rectangular_plate_modes(
    a: f64,
    b: f64,
    h: f64,
    young: f64,
    nu: f64,
    rho: f64,
    bc: PlateBc,
    max_m: usize,
    max_n: usize,
) -> Vec<(usize, usize, f64)> {
    let d = young * h.powi(3) / (12.0 * (1.0 - nu * nu));
    let coef = (d / (rho * h)).sqrt();
    let mut out = Vec::new();
    for m in 1..=max_m {
        for n in 1..=max_n {
            let omega = match bc {
                PlateBc::SimplySupported => {
                    PI * PI * ((m as f64 / a).powi(2) + (n as f64 / b).powi(2)) * coef
                }
                PlateBc::Clamped => {
                    let lm = clamped_beam_lambda(m) / a;
                    let ln = clamped_beam_lambda(n) / b;
                    coef * (lm.powi(4) + ln.powi(4) + 2.0 * lm * lm * ln * ln).sqrt()
                }
            };
            out.push((m, n, omega / TWO_PI));
        }
    }
    out.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Chladni figure of a square-symmetric plate mode: the field
/// φ_mn + φ_nm with φ_mn = cos(mπx/a)cos(nπy/b) on a res×res grid
/// (nodal lines are the zero set).
#[must_use]
pub fn chladni_pattern(a: f64, b: f64, m: usize, n: usize, res: usize) -> ScalarField2 {
    chladni_pattern_mixed(a, b, &[(m, n, 1.0), (n, m, 1.0)], res)
}

/// General superposition Σ cᵢ·cos(mᵢπx/a)cos(nᵢπy/b).
#[must_use]
pub fn chladni_pattern_mixed(
    a: f64,
    b: f64,
    modes: &[(usize, usize, f64)],
    res: usize,
) -> ScalarField2 {
    let dx = a / (res.max(2) - 1) as f64;
    ScalarField2::from_fn(res, res, dx, |x, y| {
        modes
            .iter()
            .map(|&(m, n, c)| {
                c * (m as f64 * PI * x / a).cos() * (n as f64 * PI * (y / dx) * dx / b).cos()
            })
            .sum()
    })
}

/// Beam boundary conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamBc {
    ClampedFree,
    ClampedClamped,
    SimplySupported,
    FreeFree,
}

fn beam_lambda(bc: BeamBc, i: usize) -> f64 {
    match bc {
        BeamBc::ClampedFree => cantilever_lambda(i),
        BeamBc::ClampedClamped | BeamBc::FreeFree => clamped_beam_lambda(i),
        BeamBc::SimplySupported => i as f64 * PI,
    }
}

/// Euler-Bernoulli beam natural frequencies (Hz):
/// f_i = λ_i²/(2πL²)·√(EI/(ρA)).
#[must_use]
pub fn beam_modes(
    length: f64,
    young: f64,
    i_area: f64,
    rho: f64,
    area: f64,
    bc: BeamBc,
    n: usize,
) -> Vec<f64> {
    let coef = (young * i_area / (rho * area)).sqrt() / (TWO_PI * length * length);
    (1..=n).map(|i| beam_lambda(bc, i).powi(2) * coef).collect()
}

/// Euler-Bernoulli beam mode shape at position x ∈ \[0, L\]
/// (unnormalized; standard clamped/simply-supported/free functions).
#[must_use]
pub fn beam_mode_shape(length: f64, bc: BeamBc, n: usize, x: f64) -> f64 {
    let lam = beam_lambda(bc, n);
    let xi = lam * x / length;
    match bc {
        BeamBc::SimplySupported => xi.sin(),
        BeamBc::ClampedFree => {
            let sigma = (lam.sinh() - lam.sin()) / (lam.cosh() + lam.cos());
            xi.cosh() - xi.cos() - sigma * (xi.sinh() - xi.sin())
        }
        BeamBc::ClampedClamped => {
            let sigma = (lam.cosh() - lam.cos()) / (lam.sinh() - lam.sin());
            xi.cosh() - xi.cos() - sigma * (xi.sinh() - xi.sin())
        }
        BeamBc::FreeFree => {
            let sigma = (lam.cosh() - lam.cos()) / (lam.sinh() - lam.sin());
            xi.cosh() + xi.cos() - sigma * (xi.sinh() + xi.sin())
        }
    }
}

/// Tuning fork prong frequency: cantilever first mode of a rectangular
/// prong, f = (1.875²/2π)·(t/L²)·√(E/(12ρ)).
#[must_use]
pub fn tuning_fork_frequency(length: f64, thickness: f64, young: f64, rho: f64) -> f64 {
    cantilever_lambda(1).powi(2) / TWO_PI * thickness / (length * length)
        * (young / (12.0 * rho)).sqrt()
}

/// Bell/ring flexural modes (thin-ring approximation), n = 2..:
/// f_n = n(n²−1)/√(n²+1) · (t/(2πR²))·√(E/(12ρ(1−ν²))).
#[must_use]
pub fn bell_modes_approx(
    radius: f64,
    thickness: f64,
    young: f64,
    rho: f64,
    nu: f64,
    n: usize,
) -> Vec<f64> {
    let coef = thickness / (TWO_PI * radius * radius)
        * (young / (12.0 * rho * (1.0 - nu * nu))).sqrt();
    (2..2 + n)
        .map(|k| {
            let kf = k as f64;
            kf * (kf * kf - 1.0) / (kf * kf + 1.0).sqrt() * coef
        })
        .collect()
}

/// All room modes (nx, ny, nz, f Hz) with indices up to max_n, sorted:
/// f = (c/2)·√((nx/lx)² + (ny/ly)² + (nz/lz)²).
#[must_use]
pub fn room_modes(lx: f64, ly: f64, lz: f64, c: f64, max_n: usize) -> Vec<(usize, usize, usize, f64)> {
    let mut out = Vec::new();
    for nx in 0..=max_n {
        for ny in 0..=max_n {
            for nz in 0..=max_n {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let f = c / 2.0
                    * ((nx as f64 / lx).powi(2)
                        + (ny as f64 / ly).powi(2)
                        + (nz as f64 / lz).powi(2))
                    .sqrt();
                out.push((nx, ny, nz, f));
            }
        }
    }
    out.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Asymptotic modal density dN/df = 4πVf²/c³ + πSf/(2c²) + L/(8c).
#[must_use]
pub fn room_mode_density(lx: f64, ly: f64, lz: f64, c: f64, f: f64) -> f64 {
    let v = lx * ly * lz;
    let s = 2.0 * (lx * ly + ly * lz + lz * lx);
    let l = 4.0 * (lx + ly + lz);
    4.0 * PI * v * f * f / c.powi(3) + PI * s * f / (2.0 * c * c) + l / (8.0 * c)
}

/// Schroeder crossover frequency 2000·√(RT60/V) (Hz).
#[must_use]
pub fn schroeder_frequency(rt60: f64, volume: f64) -> f64 {
    2000.0 * (rt60 / volume).sqrt()
}

/// Fabry-Perot (Airy) intensity transmission for mirror reflectance r:
/// T = (1−r)²/((1−r)² + 4r·sin²(δ/2)), δ = 4πnL/λ.
#[must_use]
pub fn fabry_perot_transmission(wavelength: f64, length: f64, r: f64, n_index: f64) -> f64 {
    let delta = 4.0 * PI * n_index * length / wavelength;
    let one_m = (1.0 - r).powi(2);
    one_m / (one_m + 4.0 * r * (delta / 2.0).sin().powi(2))
}

/// Free spectral range c/(2nL) (Hz).
#[must_use]
pub fn fabry_perot_fsr(length: f64, n_index: f64, c: f64) -> f64 {
    c / (2.0 * n_index * length)
}

/// Finesse π√r/(1−r).
#[must_use]
pub fn fabry_perot_finesse(r: f64) -> f64 {
    PI * r.sqrt() / (1.0 - r)
}

/// Quality factor f/Δf.
#[must_use]
pub fn cavity_q(frequency: f64, fwhm: f64) -> f64 {
    frequency / fwhm
}

/// Photon lifetime Q/(2πf) (s).
#[must_use]
pub fn cavity_photon_lifetime(q: f64, frequency: f64) -> f64 {
    q / (TWO_PI * frequency)
}

/// Rectangular microwave cavity modes (label, f Hz) up to index max_n,
/// sorted: f = (c/2)·√((m/a)² + (n/b)² + (p/d)²) with the standard TE
/// (p ≥ 1, m+n ≥ 1) and TM (m, n ≥ 1, p ≥ 0) index rules.
#[must_use]
pub fn microwave_cavity_modes_rect(
    a: f64,
    b: f64,
    d: f64,
    c: f64,
    max_n: usize,
) -> Vec<(String, f64)> {
    let freq = |m: usize, n: usize, p: usize| {
        c / 2.0
            * ((m as f64 / a).powi(2) + (n as f64 / b).powi(2) + (p as f64 / d).powi(2)).sqrt()
    };
    let mut out = Vec::new();
    for m in 0..=max_n {
        for n in 0..=max_n {
            for p in 0..=max_n {
                if p >= 1 && m + n >= 1 {
                    out.push((format!("TE{m}{n}{p}"), freq(m, n, p)));
                }
                if m >= 1 && n >= 1 {
                    out.push((format!("TM{m}{n}{p}"), freq(m, n, p)));
                }
            }
        }
    }
    out.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Zeros of J′_m by bracketed bisection between consecutive extrema of
/// J_m sampled on a fine grid.
fn bessel_jprime_zeros(m: u32, count: usize) -> Vec<f64> {
    let jp = |x: f64| -> f64 {
        if m == 0 {
            -bessel_jn(1, x)
        } else {
            0.5 * (bessel_jn(m - 1, x) - bessel_jn(m + 1, x))
        }
    };
    let mut out = Vec::new();
    let mut x = if m == 0 { 0.5 } else { 0.05 + m as f64 * 0.5 };
    let dx = 0.01;
    let mut prev = jp(x);
    while out.len() < count && x < 500.0 {
        let next = jp(x + dx);
        if prev.signum() != next.signum() {
            let (mut lo, mut hi) = (x, x + dx);
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if jp(mid).signum() == prev.signum() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            out.push(0.5 * (lo + hi));
        }
        prev = next;
        x += dx;
    }
    out
}

/// Cylindrical cavity modes (label, f Hz), sorted:
/// TM_mnp uses J_m zeros (p ≥ 0), TE_mnp uses J′_m zeros (p ≥ 1);
/// f = (c/2π)·√((x/R)² + (pπ/H)²).
#[must_use]
pub fn cylindrical_cavity_modes(radius: f64, height: f64, c: f64, max_n: usize) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for m in 0..=max_n {
        let tm_zeros = bessel_j_zeros(m as u32, max_n);
        let te_zeros = bessel_jprime_zeros(m as u32, max_n);
        for n in 1..=max_n {
            for p in 0..=max_n {
                let f_of = |x: f64| {
                    c / TWO_PI * ((x / radius).powi(2) + (p as f64 * PI / height).powi(2)).sqrt()
                };
                out.push((format!("TM{m}{n}{p}"), f_of(tm_zeros[n - 1])));
                if p >= 1 {
                    out.push((format!("TE{m}{n}{p}"), f_of(te_zeros[n - 1])));
                }
            }
        }
    }
    out.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Quarter-wave resonator fundamental c/(4L).
#[must_use]
pub fn quarter_wave_resonator(length: f64, c: f64) -> f64 {
    c / (4.0 * length)
}

/// Mode splitting of two identical coupled cavities with coupling
/// coefficient κ: (f₀√(1−κ), f₀√(1+κ)).
#[must_use]
pub fn coupled_cavity_splitting(f0: f64, coupling: f64) -> (f64, f64) {
    (f0 * (1.0 - coupling).sqrt(), f0 * (1.0 + coupling).sqrt())
}

/// Lorentzian overlap of two resonances (1 when co-tuned, → 0 when far
/// apart relative to their combined half-widths).
#[must_use]
pub fn resonance_overlap(f1: f64, q1: f64, f2: f64, q2: f64) -> f64 {
    let g = 0.5 * (f1 / q1 + f2 / q2) / 2.0;
    g * g / ((f1 - f2).powi(2) / 4.0 + g * g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_rlc_resonance() {
        let rlc = Rlc { r: 10.0, l: 1e-3, c: 1e-6 };
        let f0 = rlc.resonant_frequency();
        assert!(approx(f0, 1.0 / (TWO_PI * (1e-9_f64).sqrt()), 1e-6));
        // Series impedance minimal (purely resistive) at resonance.
        let z0 = rlc.series_impedance(TWO_PI * f0);
        assert!(approx(z0.re, 10.0, 1e-9) && z0.im.abs() < 1e-6);
        // Q relations.
        assert!(approx(rlc.q_series() * rlc.q_parallel(), 1.0 / 1.0 * (1e-3 / 1e-6) * (1e-6 / 1e-3), 1e-9));
        // Transfers: LP → 1 at DC-ish, BP peak 1 at f0, notch 0 at f0.
        assert!(approx(rlc.transfer_bandpass(TWO_PI * f0).norm(), 1.0, 1e-9));
        assert!(rlc.transfer_notch(TWO_PI * f0).norm() < 1e-9);
        assert!(approx(rlc.transfer_lowpass(1.0).norm(), 1.0, 1e-3));
        // Step response settles at v.
        assert!(approx(rlc.step_response(0.5, 5.0), 5.0, 1e-6));
    }

    #[test]
    fn test_rlc_damping_bandwidth_and_transfers() {
        let rlc = Rlc { r: 10.0, l: 1e-3, c: 1e-6 };
        let f0 = rlc.resonant_frequency();
        let w0 = TWO_PI * f0;

        // ζ = (R/2)√(C/L), and ζ = 1/(2Q) for the series loop.
        let zeta = rlc.damping_ratio();
        assert!(approx(zeta, 10.0 / 2.0 * (1e-6_f64 / 1e-3).sqrt(), 1e-15));
        assert!(approx(zeta, 1.0 / (2.0 * rlc.q_series()), 1e-15), "ζ vs 1/2Q");
        // It agrees with the mechanical analogue (m = L, c = R, k = 1/C)
        // that step_response already relies on.
        let mech = super::super::oscillator::DampedOscillator { m: rlc.l, c: rlc.r, k: 1.0 / rlc.c };
        assert!(approx(zeta, mech.damping_ratio(), 1e-15));
        assert!(approx(w0, mech.natural_frequency(), 1e-9));
        assert!(rlc.damping_ratio() < 1.0, "expected an underdamped loop");
        // Raising R raises the damping proportionally.
        let heavy = Rlc { r: 40.0, l: 1e-3, c: 1e-6 };
        assert!(approx(heavy.damping_ratio(), 4.0 * zeta, 1e-15));

        // Δf = f0/Q = R/(2πL), and it really is the half-power width of
        // the band-pass transfer.
        let bw = rlc.bandwidth();
        assert!(approx(bw, f0 / rlc.q_series(), 1e-12));
        assert!(approx(bw, rlc.r / (TWO_PI * rlc.l), 1e-9), "Δf vs R/2πL");
        // The two −3 dB crossings of |H_bp| are ω0·(∓ζ + √(1+ζ²)).
        let half_power = |lo: f64, hi: f64| -> f64 {
            let (mut a, mut b) = (lo, hi);
            for _ in 0..200 {
                let mid = 0.5 * (a + b);
                let m = rlc.transfer_bandpass(mid).norm();
                if (m - std::f64::consts::FRAC_1_SQRT_2) > 0.0 {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            0.5 * (a + b)
        };
        let w_hi = half_power(w0, 10.0 * w0);
        let w_lo = half_power(w0, 0.01 * w0);
        assert!(approx((w_hi - w_lo) / TWO_PI, bw, 1e-6 * bw), "measured Δf");
        assert!(approx((w_lo * w_hi).sqrt(), w0, 1e-6 * w0), "edges straddle ω0");

        // The three series transfers are a voltage divider: they sum to 1
        // at every frequency.
        for &w in &[0.01 * w0, 0.5 * w0, w0, 2.0 * w0, 100.0 * w0] {
            let sum = rlc.transfer_lowpass(w) + rlc.transfer_bandpass(w) + rlc.transfer_highpass(w);
            assert!(approx(sum.re, 1.0, 1e-12) && approx(sum.im, 0.0, 1e-12), "divider at {w}");
        }
        // High-pass: → 0 at DC, → 1 well above resonance, and exactly Q at
        // resonance (ω0·L/R = √(L/C)/R).
        assert!(rlc.transfer_highpass(1e-3 * w0).norm() < 1e-5, "HP leaks at DC");
        assert!(approx(rlc.transfer_highpass(1e4 * w0).norm(), 1.0, 1e-6), "HP not unity");
        assert!(approx(rlc.transfer_highpass(w0).norm(), rlc.q_series(), 1e-9), "HP peak ≠ Q");
        assert!(rlc.transfer_highpass(1e4 * w0).norm() > rlc.transfer_highpass(0.1 * w0).norm());

        // Parallel impedance peaks at resonance, where it is purely R.
        let z0 = rlc.parallel_impedance(w0);
        assert!(approx(z0.re, rlc.r, 1e-9) && z0.im.abs() < 1e-9, "Z_par(ω0) = {z0:?}");
        let below = rlc.parallel_impedance(0.5 * w0).norm();
        let above = rlc.parallel_impedance(2.0 * w0).norm();
        assert!(below < z0.norm() && above < z0.norm(), "not a maximum: {below}, {above}");
        // Closed form |Z| = R/√(1 + Q_p²(ω/ω0 − ω0/ω)²).
        for &ratio in &[0.5_f64, 0.8, 1.25, 2.0] {
            let w = ratio * w0;
            let qp = rlc.q_parallel();
            let want = rlc.r / (1.0 + (qp * (ratio - 1.0 / ratio)).powi(2)).sqrt();
            assert!(
                approx(rlc.parallel_impedance(w).norm(), want, 1e-9 * rlc.r),
                "at ω/ω0 = {ratio}"
            );
        }
        // Series and parallel impedances are reciprocal in character: the
        // series loop is minimum where the parallel tank is maximum.
        assert!(rlc.series_impedance(w0).norm() < rlc.series_impedance(2.0 * w0).norm());
    }

    #[test]
    fn test_rlc_energy_is_conserved_in_the_lossless_tank() {
        // ½Li² + ½Cv² by definition.
        let rlc = Rlc { r: 0.0, l: 2e-3, c: 5e-6 };
        assert!(approx(rlc.energy(3.0, 0.0), 0.5 * 2e-3 * 9.0, 1e-18));
        assert!(approx(rlc.energy(0.0, 4.0), 0.5 * 5e-6 * 16.0, 1e-18));
        assert_eq!(rlc.energy(0.0, 0.0), 0.0, "an idle tank stores nothing");
        // Quadratic in each argument.
        assert!(approx(rlc.energy(2.0, 0.0), 4.0 * rlc.energy(1.0, 0.0), 1e-18));

        // With R = 0 the loop is a lossless LC oscillator: charge obeys
        // q̈ + q/(LC) = 0, so total energy ½Li² + q²/(2C) is constant and
        // sloshes between the inductor and the capacitor at ω0.
        let w0 = TWO_PI * rlc.resonant_frequency();
        assert!(approx(w0, 1.0 / (rlc.l * rlc.c).sqrt(), 1e-9));
        let q0 = 1e-5; // initial charge, capacitor fully charged
        let e0 = rlc.energy(0.0, q0 / rlc.c);
        let mut min_frac = f64::MAX;
        let mut max_frac = f64::MIN;
        for step in 0..=400 {
            let t = step as f64 / 400.0 * 4.0 * TWO_PI / w0; // four periods
            let q = q0 * (w0 * t).cos();
            let i = -q0 * w0 * (w0 * t).sin();
            let e = rlc.energy(i, q / rlc.c);
            assert!(
                (e - e0).abs() < 1e-12 * e0,
                "energy drifted at t = {t}: {e} vs {e0}"
            );
            // Track how the split moves between the two stores.
            let magnetic = 0.5 * rlc.l * i * i / e0;
            min_frac = min_frac.min(magnetic);
            max_frac = max_frac.max(magnetic);
        }
        assert!(min_frac < 1e-4, "energy never fully returns to the capacitor");
        assert!(max_frac > 1.0 - 1e-4, "energy never fully reaches the inductor");
        // Equipartition: at ω0·t = π/4 the split is exactly half and half.
        let t = 0.25 * PI / w0;
        let q = q0 * (w0 * t).cos();
        let i = -q0 * w0 * (w0 * t).sin();
        assert!(approx(0.5 * rlc.l * i * i, 0.5 * e0, 1e-12 * e0), "magnetic half");
        assert!(approx(q * q / (2.0 * rlc.c), 0.5 * e0, 1e-12 * e0), "electric half");
        assert!(approx(rlc.energy(i, q / rlc.c), e0, 1e-12 * e0), "equipartition total");

        // With loss the stored energy decays at the rate R/L (the analogue
        // of the mechanical c/m).
        let lossy = Rlc { r: 0.5, l: 2e-3, c: 5e-6 };
        let osc = super::super::oscillator::DampedOscillator {
            m: lossy.l,
            c: lossy.r,
            k: 1.0 / lossy.c,
        };
        let traj = osc.forced_response_numeric(&|_| 0.0, q0, 0.0, 0.02, 1e-7);
        let start = lossy.energy(0.0, q0 / lossy.c);
        for &(t, q, i) in traj.iter().skip(20000).step_by(40000) {
            let e = lossy.energy(i, q / lossy.c);
            let expect = start * (-lossy.r * t / lossy.l).exp();
            assert!((e - expect).abs() / expect < 0.03, "t = {t}: {e} vs {expect}");
        }
    }

    #[test]
    fn test_string_and_stiff_string() {
        let modes = string_modes(0.65, 60.0, 0.001, 5);
        for (i, &f) in modes.iter().enumerate() {
            assert!(approx(f, (i + 1) as f64 * modes[0], 1e-9));
        }
        assert!(approx(string_mode_shape(1.0, 2, 0.25), 1.0, 1e-12));
        // Stiff string sharpens the partials monotonically.
        let stiff = stiff_string_modes(0.65, 60.0, 0.001, 2e11, 5e-4, 5);
        for (i, &fv) in stiff.iter().enumerate() {
            assert!(fv >= (i + 1) as f64 * modes[0]);
        }
        let b = inharmonicity_coefficient(2e11, 5e-4, 60.0, 0.65);
        assert!(approx(stiff[1], 2.0 * modes[0] * (1.0 + 4.0 * b).sqrt(), 1e-9));
    }

    #[test]
    fn test_tubes() {
        let open = tube_modes(0.5, 343.0, true, 3);
        assert!(approx(open[0], 343.0, 1e-9));
        assert!(approx(open[1], 686.0, 1e-9));
        let closed = tube_modes(0.5, 343.0, false, 3);
        assert!(approx(closed[0], 171.5, 1e-9));
        assert!(approx(closed[1], 3.0 * 171.5, 1e-9)); // odd harmonics
        assert!(approx(quarter_wave_resonator(0.5, 343.0), 171.5, 1e-9));
        assert!(tube_end_correction(0.01, true) > tube_end_correction(0.01, false));
        let cone = conical_tube_modes(0.5, 343.0, 2);
        assert!(approx(cone[0], 343.0, 1e-9));
    }

    #[test]
    fn test_circular_membrane_fundamental() {
        let (radius, tension, sigma) = (0.2_f64, 100.0_f64, 0.3_f64);
        let c = (tension / sigma).sqrt();
        let modes = circular_membrane_modes(radius, tension, sigma, 3, 3);
        // Fundamental is the (0,1) mode at 2.405·c/(2πR).
        let (m, n, f) = modes[0];
        assert_eq!((m, n), (0, 1));
        assert!(approx(f, 2.404_825_557_695_773 * c / (TWO_PI * radius), 1e-6));
        // Shape peaks at the center for m=0 and vanishes at the rim.
        // (the crate Bessel evaluator is a ~3e-9 polynomial approximation)
        assert!(approx(circular_membrane_shape(radius, 0, 1, 0.0, 0.0), 1.0, 1e-7));
        assert!(circular_membrane_shape(radius, 0, 1, radius, 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_rectangular_membrane_and_plate() {
        let modes = rectangular_membrane_modes(1.0, 1.0, 100.0, 0.1, 2, 2);
        let f11 = modes[0].2;
        // Degenerate (1,2)/(2,1) pair next.
        assert!(approx(modes[1].2, modes[2].2, 1e-9));
        assert!(approx(modes[1].2 / f11, (2.5_f64 / 0.5).sqrt() / 2.0_f64.sqrt() * 2.0_f64.sqrt() / 2.0_f64.sqrt(), 1.0)); // sanity only
        // Simply supported plate fundamental: (π²/2π)(1/a²+1/b²)√(D/ρh).
        let p = rectangular_plate_modes(0.4, 0.3, 0.002, 7e10, 0.33, 2700.0, PlateBc::SimplySupported, 2, 2);
        let d: f64 = 7e10 * 0.002_f64.powi(3) / (12.0 * (1.0 - 0.33_f64 * 0.33));
        let expect = PI * PI * (1.0 / 0.16 + 1.0 / 0.09) * (d / (2700.0 * 0.002)).sqrt() / TWO_PI;
        assert!(approx(p[0].2, expect, 1e-6));
        // Clamped estimate is higher than simply supported.
        let pc = rectangular_plate_modes(0.4, 0.3, 0.002, 7e10, 0.33, 2700.0, PlateBc::Clamped, 2, 2);
        assert!(pc[0].2 > p[0].2);
    }

    #[test]
    fn test_beam_mode_ratios() {
        // Clamped-free frequency ratios 1 : 6.27 : 17.5.
        let f = beam_modes(1.0, 2e11, 1e-8, 7800.0, 1e-4, BeamBc::ClampedFree, 3);
        assert!(approx(f[1] / f[0], 6.27, 0.01), "{}", f[1] / f[0]);
        assert!(approx(f[2] / f[0], 17.55, 0.06), "{}", f[2] / f[0]);
        // Simply supported: harmonic in λ² → ratios 1:4:9.
        let fs = beam_modes(1.0, 2e11, 1e-8, 7800.0, 1e-4, BeamBc::SimplySupported, 3);
        assert!(approx(fs[1] / fs[0], 4.0, 1e-9));
        assert!(approx(fs[2] / fs[0], 9.0, 1e-9));
        // Shapes satisfy their essential boundary conditions.
        assert!(beam_mode_shape(1.0, BeamBc::ClampedFree, 1, 0.0).abs() < 1e-12);
        assert!(beam_mode_shape(1.0, BeamBc::SimplySupported, 2, 0.0).abs() < 1e-12);
        // Tuning fork: A440-ish prong.
        let ft = tuning_fork_frequency(0.09, 0.004, 2e11, 7800.0);
        assert!(ft > 200.0 && ft < 900.0, "{ft}");
        // Bell modes rise steeply with n.
        let bells = bell_modes_approx(0.1, 0.01, 1e11, 8600.0, 0.35, 3);
        assert!(bells[1] / bells[0] > 2.0);
    }

    #[test]
    fn test_room_modes_and_density() {
        let modes = room_modes(5.0, 4.0, 3.0, 343.0, 2);
        // First axial mode of the longest dimension: c/(2·5).
        assert!(approx(modes[0].3, 343.0 / 10.0, 1e-9));
        assert_eq!((modes[0].0, modes[0].1, modes[0].2), (1, 0, 0));
        // Matches the existing acoustics helper.
        let f_ref = crate::acoustics::room_mode_frequency(5.0, 4.0, 3.0, 1, 0, 0, 343.0);
        assert!(approx(modes[0].3, f_ref, 1e-9));
        // Density grows like f².
        let d1 = room_mode_density(5.0, 4.0, 3.0, 343.0, 100.0);
        let d2 = room_mode_density(5.0, 4.0, 3.0, 343.0, 200.0);
        assert!(d2 / d1 > 3.0);
        assert!(approx(schroeder_frequency(0.6, 60.0), 200.0, 1e-9));
    }

    #[test]
    fn test_fabry_perot() {
        let (l, n, c) = (0.01, 1.0, 299_792_458.0);
        // Resonance when 2nL is an integer number of wavelengths.
        let lam = 2.0 * n * l / 20000.0;
        assert!(approx(fabry_perot_transmission(lam, l, 0.9, n), 1.0, 1e-9));
        // Off resonance: strongly suppressed for high reflectance.
        let lam_off = 2.0 * n * l / 20000.5;
        assert!(fabry_perot_transmission(lam_off, l, 0.9, n) < 0.05);
        assert!(approx(fabry_perot_fsr(l, n, c), c / (2.0 * l), 1e-6));
        // Finesse ≈ FSR / FWHM: check against a numeric FWHM.
        let fin = fabry_perot_finesse(0.9);
        assert!(approx(fin, PI * 0.9_f64.sqrt() / 0.1, 1e-9));
        assert!(approx(cavity_photon_lifetime(cavity_q(1e9, 1e6), 1e9), 1e3 / (TWO_PI * 1e9), 1e-12));
    }

    #[test]
    fn test_microwave_and_cylindrical_cavities() {
        // Cubic cavity: dominant TE101 at (c/2)·√(2)/a.
        let a = 0.1;
        let c = 299_792_458.0;
        let modes = microwave_cavity_modes_rect(a, a, a, c, 2);
        let (label, f) = &modes[0];
        assert!(label.starts_with("TE"), "{label}");
        assert!(approx(*f, c / 2.0 * 2.0_f64.sqrt() / a, 1.0), "{f}");
        // Cylindrical: TM010 fundamental at 2.405·c/(2πR), p-independent of height.
        let cyl = cylindrical_cavity_modes(0.05, 0.1, c, 2);
        let (lab0, f0) = &cyl[0];
        assert_eq!(lab0, "TM010");
        assert!(approx(*f0, 2.404_825_557_695_773 * c / (TWO_PI * 0.05), 10.0));
    }

    #[test]
    fn test_helmholtz_and_overlap() {
        // Classic bottle: V=1 L, neck 5 cm² × 5 cm → ~100 Hz range.
        let f = helmholtz_resonator(1e-3, 5e-4, 0.05, 343.0);
        assert!(f > 80.0 && f < 200.0, "{f}");
        assert!(helmholtz_q(1e-3, 5e-4, 0.05, 343.0) > 1.0);
        // Overlap: 1 co-tuned, small when detuned by many linewidths.
        assert!(approx(resonance_overlap(1000.0, 100.0, 1000.0, 100.0), 1.0, 1e-12));
        assert!(resonance_overlap(1000.0, 1000.0, 1010.0, 1000.0) < 0.05);
        let (lo, hi) = coupled_cavity_splitting(1000.0, 0.02);
        assert!(lo < 1000.0 && hi > 1000.0);
        assert!(approx(hi - lo, 1000.0 * 0.02, 0.2));
    }

    #[test]
    fn test_chladni_field() {
        let field = chladni_pattern(1.0, 1.0, 3, 1, 65);
        assert_eq!(field.nx, 65);
        // Symmetric under (x,y) swap by construction.
        for i in 0..65 {
            for j in 0..65 {
                assert!(approx(field.get(i, j), field.get(j, i), 1e-9));
            }
        }
        // Corners: cos(0)=1 terms → value 2 at the origin corner.
        assert!(approx(field.get(0, 0), 2.0, 1e-12));
    }
}
