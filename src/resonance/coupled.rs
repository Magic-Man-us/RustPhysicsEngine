//! Coupled linear oscillators: normal modes, modal superposition,
//! receptance, classic two-body systems, Kuramoto synchronization, and
//! tuned-mass-damper design.

use crate::linalg::{eigen_symmetric, lu_decompose, solve, Matrix};
use crate::math::constants::PI;
use crate::numerical::rk4_step_vec;

const TWO_PI: f64 = 2.0 * PI;

/// N-degree-of-freedom system M·x″ + C·x′ + K·x = F.
#[derive(Debug, Clone)]
pub struct CoupledOscillators {
    pub masses: Vec<f64>,
    pub stiffness: Matrix,
    pub damping: Matrix,
}

impl CoupledOscillators {
    fn n(&self) -> usize {
        self.masses.len()
    }

    /// Chain of n masses, each grounded with stiffness k and coupled to
    /// its neighbors by k_coupling (free ends).
    #[must_use]
    pub fn chain(n: usize, m: f64, k: f64, k_coupling: f64) -> Self {
        let mut kk = Matrix::zeros(n, n);
        for i in 0..n {
            let mut diag = k;
            if i > 0 {
                diag += k_coupling;
                kk.set(i, i - 1, -k_coupling);
            }
            if i + 1 < n {
                diag += k_coupling;
                kk.set(i, i + 1, -k_coupling);
            }
            kk.set(i, i, diag);
        }
        Self { masses: vec![m; n], stiffness: kk, damping: Matrix::zeros(n, n) }
    }

    /// Chain of n masses joined in series by springs k with both ends
    /// fixed to walls (the textbook fixed-fixed chain).
    #[must_use]
    pub fn chain_fixed_ends(n: usize, m: f64, k: f64) -> Self {
        let mut kk = Matrix::zeros(n, n);
        for i in 0..n {
            kk.set(i, i, 2.0 * k);
            if i > 0 {
                kk.set(i, i - 1, -k);
            }
            if i + 1 < n {
                kk.set(i, i + 1, -k);
            }
        }
        Self { masses: vec![m; n], stiffness: kk, damping: Matrix::zeros(n, n) }
    }

    /// Ring of n masses joined by springs k (periodic chain).
    #[must_use]
    pub fn ring(n: usize, m: f64, k: f64) -> Self {
        let mut kk = Matrix::zeros(n, n);
        for i in 0..n {
            kk.set(i, i, 2.0 * k);
            let prev = (i + n - 1) % n;
            let next = (i + 1) % n;
            kk.set(i, prev, kk.get(i, prev) - k);
            kk.set(i, next, kk.get(i, next) - k);
        }
        Self { masses: vec![m; n], stiffness: kk, damping: Matrix::zeros(n, n) }
    }

    /// Build from an explicit spring list: (i, j, k) couples masses i
    /// and j; (i, i, k) grounds mass i with stiffness k.
    #[must_use]
    pub fn from_springs(masses: &[f64], springs: &[(usize, usize, f64)]) -> Self {
        let n = masses.len();
        let mut kk = Matrix::zeros(n, n);
        for &(i, j, k) in springs {
            if i == j {
                kk.set(i, i, kk.get(i, i) + k);
            } else {
                kk.set(i, i, kk.get(i, i) + k);
                kk.set(j, j, kk.get(j, j) + k);
                kk.set(i, j, kk.get(i, j) - k);
                kk.set(j, i, kk.get(j, i) - k);
            }
        }
        Self { masses: masses.to_vec(), stiffness: kk, damping: Matrix::zeros(n, n) }
    }

    /// Undamped normal modes: frequencies (rad/s, ascending) and
    /// mass-orthonormal mode shapes as matrix columns (ΦᵀMΦ = I), from
    /// the symmetric eigenproblem M^(−1/2)·K·M^(−1/2).
    ///
    /// # Panics
    /// Panics if the eigen solve fails.
    #[must_use]
    pub fn normal_modes(&self) -> (Vec<f64>, Matrix) {
        let n = self.n();
        let inv_sqrt_m: Vec<f64> = self.masses.iter().map(|&m| 1.0 / m.sqrt()).collect();
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, inv_sqrt_m[i] * self.stiffness.get(i, j) * inv_sqrt_m[j]);
            }
        }
        let eig = eigen_symmetric(&a, 1e-12, 200).expect("normal mode eigen solve failed");
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&x, &y| {
            eig.values[x].partial_cmp(&eig.values[y]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let freqs: Vec<f64> = order.iter().map(|&i| eig.values[i].max(0.0).sqrt()).collect();
        let mut shapes = Matrix::zeros(n, n);
        for (col, &src) in order.iter().enumerate() {
            for (row, &ism) in inv_sqrt_m.iter().enumerate() {
                shapes.set(row, col, ism * eig.vectors.get(row, src));
            }
        }
        (freqs, shapes)
    }

    /// Mode shape i (ascending frequency order).
    #[must_use]
    pub fn mode_shape(&self, i: usize) -> Vec<f64> {
        let (_, shapes) = self.normal_modes();
        (0..self.n()).map(|r| shapes.get(r, i)).collect()
    }

    /// Modal participation q = ΦᵀM·x₀ of an initial displacement.
    #[must_use]
    pub fn modal_participation(&self, x0: &[f64]) -> Vec<f64> {
        let (_, shapes) = self.normal_modes();
        let n = self.n();
        (0..n)
            .map(|mode| {
                (0..n).map(|r| shapes.get(r, mode) * self.masses[r] * x0[r]).sum()
            })
            .collect()
    }

    /// Modal damping ratios ζᵢ = φᵢᵀCφᵢ/(2ωᵢ) (proportional-damping
    /// assumption).
    #[must_use]
    pub fn modal_damping_ratios(&self) -> Vec<f64> {
        let (freqs, shapes) = self.normal_modes();
        let n = self.n();
        (0..n)
            .map(|mode| {
                let mut cm = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        cm += shapes.get(i, mode) * self.damping.get(i, j) * shapes.get(j, mode);
                    }
                }
                if freqs[mode] > 1e-12 {
                    cm / (2.0 * freqs[mode])
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Free response x(t) by modal superposition (proportional damping).
    ///
    /// # Panics
    /// Panics on dimension mismatch.
    #[must_use]
    pub fn response(&self, x0: &[f64], v0: &[f64], t: f64) -> Vec<f64> {
        let n = self.n();
        assert!(x0.len() == n && v0.len() == n, "state dimension mismatch");
        let (freqs, shapes) = self.normal_modes();
        let zetas = self.modal_damping_ratios();
        // Modal initial conditions.
        let q0: Vec<f64> = (0..n)
            .map(|mode| (0..n).map(|r| shapes.get(r, mode) * self.masses[r] * x0[r]).sum())
            .collect();
        let qd0: Vec<f64> = (0..n)
            .map(|mode| (0..n).map(|r| shapes.get(r, mode) * self.masses[r] * v0[r]).sum())
            .collect();
        let mut x = vec![0.0; n];
        for mode in 0..n {
            let w = freqs[mode];
            let z = zetas[mode].min(0.999_999);
            let qt = if w < 1e-12 {
                q0[mode] + qd0[mode] * t
            } else {
                let wd = w * (1.0 - z * z).sqrt();
                (-z * w * t).exp()
                    * (q0[mode] * (wd * t).cos()
                        + (qd0[mode] + z * w * q0[mode]) / wd * (wd * t).sin())
            };
            for (r, xv) in x.iter_mut().enumerate() {
                *xv += shapes.get(r, mode) * qt;
            }
        }
        x
    }

    /// Steady-state complex response X to harmonic forcing F·e^{jωt},
    /// solving (K + jωC − ω²M)X = F as a doubled real system.
    ///
    /// # Panics
    /// Panics on dimension mismatch or singular dynamic stiffness.
    #[must_use]
    pub fn forced_response(&self, force: &[f64], omega: f64) -> Vec<crate::fractals::Complex> {
        let n = self.n();
        assert_eq!(force.len(), n, "force dimension mismatch");
        // [K−ω²M  −ωC][Re]   [F]
        // [ωC   K−ω²M][Im] = [0]
        let mut big = Matrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                let a = self.stiffness.get(i, j)
                    - if i == j { omega * omega * self.masses[i] } else { 0.0 };
                let c = omega * self.damping.get(i, j);
                big.set(i, j, a);
                big.set(n + i, n + j, a);
                big.set(i, n + j, -c);
                big.set(n + i, j, c);
            }
        }
        let mut rhs = vec![0.0; 2 * n];
        rhs[..n].copy_from_slice(force);
        let sol = solve(&big, &rhs).expect("dynamic stiffness singular");
        (0..n).map(|i| crate::fractals::Complex::new(sol[i], sol[n + i])).collect()
    }

    /// Undamped receptance matrix (K − ω²M)^(−1).
    ///
    /// # Panics
    /// Panics at an exact resonance (singular matrix).
    #[must_use]
    pub fn frequency_response_matrix(&self, omega: f64) -> Matrix {
        let n = self.n();
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                d.set(
                    i,
                    j,
                    self.stiffness.get(i, j)
                        - if i == j { omega * omega * self.masses[i] } else { 0.0 },
                );
            }
        }
        lu_decompose(&d).and_then(|lu| lu.inverse()).expect("receptance singular at resonance")
    }

    /// Beat (envelope) angular frequency ω₂ − ω₁ for a two-mode system.
    #[must_use]
    pub fn beat_frequency(&self) -> Option<f64> {
        if self.n() != 2 {
            return None;
        }
        let (freqs, _) = self.normal_modes();
        Some(freqs[1] - freqs[0])
    }

    /// Time for the energy to migrate fully between two identical
    /// coupled oscillators: π/(ω₂ − ω₁).
    #[must_use]
    pub fn energy_transfer_time(&self) -> Option<f64> {
        self.beat_frequency().map(|dw| PI / dw)
    }

    /// Dispersion relation ω(q) of the uniform chain this system was
    /// built as (q in radians per lattice site):
    /// ω = √((k_ground + 4·k_c·sin²(q/2))/m), with k_c and k_ground read
    /// off the stiffness matrix interior.
    #[must_use]
    pub fn dispersion_relation(&self, k_wave: f64) -> f64 {
        let n = self.n();
        let kc = if n > 1 { -self.stiffness.get(0, 1) } else { 0.0 };
        let interior = if n > 2 { self.stiffness.get(1, 1) } else { self.stiffness.get(0, 0) };
        let k_ground = (interior - 2.0 * kc).max(0.0);
        ((k_ground + 4.0 * kc * (k_wave / 2.0).sin().powi(2)) / self.masses[0]).sqrt()
    }

    /// Velocity-Verlet simulation (with damping forces); returns the
    /// position vector at every step, including t = 0.
    ///
    /// # Panics
    /// Panics on dimension mismatch.
    #[must_use]
    pub fn simulate(&self, x0: &[f64], v0: &[f64], t_end: f64, dt: f64) -> Vec<Vec<f64>> {
        let n = self.n();
        assert!(x0.len() == n && v0.len() == n, "state dimension mismatch");
        let accel = |x: &[f64], v: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let mut f = 0.0;
                    for j in 0..n {
                        f -= self.stiffness.get(i, j) * x[j] + self.damping.get(i, j) * v[j];
                    }
                    f / self.masses[i]
                })
                .collect()
        };
        let mut x = x0.to_vec();
        let mut v = v0.to_vec();
        let mut a = accel(&x, &v);
        let steps = (t_end / dt).round() as usize;
        let mut out = Vec::with_capacity(steps + 1);
        out.push(x.clone());
        for _ in 0..steps {
            for i in 0..n {
                x[i] += v[i] * dt + 0.5 * a[i] * dt * dt;
            }
            // Half-kick, recompute, half-kick (velocity Verlet with
            // velocity-dependent force via a predictor half step).
            let v_half: Vec<f64> = (0..n).map(|i| v[i] + 0.5 * a[i] * dt).collect();
            let a_new = accel(&x, &v_half);
            for i in 0..n {
                v[i] += 0.5 * (a[i] + a_new[i]) * dt;
            }
            a = a_new;
            out.push(x.clone());
        }
        out
    }

    /// Dunkerley lower-bound estimate of the fundamental frequency:
    /// 1/ω² ≈ Σ mᵢ·(K⁻¹)ᵢᵢ.
    ///
    /// # Panics
    /// Panics if K is singular.
    #[must_use]
    pub fn dunkerley_estimate(&self) -> f64 {
        let inv = lu_decompose(&self.stiffness)
            .and_then(|lu| lu.inverse())
            .expect("stiffness singular");
        let s: f64 = self.masses.iter().enumerate().map(|(i, &m)| m * inv.get(i, i)).sum();
        (1.0 / s).sqrt()
    }

    /// Rayleigh quotient ω² estimate φᵀKφ/φᵀMφ for a trial shape.
    #[must_use]
    pub fn rayleigh_quotient(&self, shape: &[f64]) -> f64 {
        let n = self.n();
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            den += self.masses[i] * shape[i] * shape[i];
            for j in 0..n {
                num += shape[i] * self.stiffness.get(i, j) * shape[j];
            }
        }
        num / den
    }

    /// Anti-resonance frequencies of the receptance H_ij(ω): zeros
    /// located by scanning for sign changes of the undamped H_ij up to
    /// 1.2× the highest natural frequency.
    #[must_use]
    pub fn anti_resonance_frequencies(&self, i: usize, j: usize) -> Vec<f64> {
        let (freqs, _) = self.normal_modes();
        let w_max = freqs.last().copied().unwrap_or(1.0) * 1.2;
        let steps = 4000;
        let h = |w: f64| -> Option<f64> {
            // Skip points too close to a pole.
            for &f in &freqs {
                if (w - f).abs() < w_max * 1e-4 {
                    return None;
                }
            }
            Some(self.frequency_response_matrix(w).get(i, j))
        };
        let mut out = Vec::new();
        let mut prev: Option<(f64, f64)> = None;
        for s in 1..steps {
            let w = w_max * s as f64 / steps as f64;
            if let Some(v) = h(w) {
                if let Some((wp, vp)) = prev {
                    if vp.signum() != v.signum() {
                        // Make sure no pole sits between the two samples.
                        let pole_between = freqs.iter().any(|&f| f > wp && f < w);
                        if !pole_between {
                            // Bisect.
                            let (mut a, mut b) = (wp, w);
                            for _ in 0..60 {
                                let mid = 0.5 * (a + b);
                                match h(mid) {
                                    Some(vm) if vm.signum() == vp.signum() => a = mid,
                                    Some(_) => b = mid,
                                    None => break,
                                }
                            }
                            out.push(0.5 * (a + b));
                        }
                    }
                }
                prev = Some((w, v));
            } else {
                prev = None;
            }
        }
        out
    }
}

/// Two pendulums (length l, mass m) coupled by a spring k: 2-dof
/// small-angle system in the displacement coordinates.
#[must_use]
pub fn two_pendulums_coupled(l: f64, g: f64, k: f64, m: f64) -> CoupledOscillators {
    let kg = m * g / l;
    let mut kk = Matrix::zeros(2, 2);
    kk.set(0, 0, kg + k);
    kk.set(1, 1, kg + k);
    kk.set(0, 1, -k);
    kk.set(1, 0, -k);
    CoupledOscillators { masses: vec![m, m], stiffness: kk, damping: Matrix::zeros(2, 2) }
}

/// Wilberforce pendulum: vertical bounce (mass m, spring k) coupled to
/// torsion (inertia i, stiffness kappa) through the cross term eps.
#[must_use]
pub fn wilberforce_pendulum(m: f64, k: f64, i: f64, kappa: f64, eps: f64) -> CoupledOscillators {
    let mut kk = Matrix::zeros(2, 2);
    kk.set(0, 0, k);
    kk.set(1, 1, kappa);
    kk.set(0, 1, eps / 2.0);
    kk.set(1, 0, eps / 2.0);
    CoupledOscillators { masses: vec![m, i], stiffness: kk, damping: Matrix::zeros(2, 2) }
}

/// Two weakly coupled phase oscillators (Huygens' clocks abstraction):
/// φ̇₁ = ω₁ + κ·sin(φ₂−φ₁), φ̇₂ = ω₂ + κ·sin(φ₁−φ₂). Returns
/// (t, wrapped phase difference) per step.
#[must_use]
pub fn huygens_sync_simulate(
    omega1: f64,
    omega2: f64,
    coupling: f64,
    t_end: f64,
    dt: f64,
) -> Vec<(f64, f64)> {
    let f = |_t: f64, y: &[f64]| -> Vec<f64> {
        vec![
            omega1 + coupling * (y[1] - y[0]).sin(),
            omega2 + coupling * (y[0] - y[1]).sin(),
        ]
    };
    let mut y = vec![0.0, PI / 2.0];
    let mut t = 0.0;
    let mut out = vec![(0.0, crate::dsp::phase::wrap_phase(y[1] - y[0]))];
    while t < t_end - 1e-12 {
        y = rk4_step_vec(&f, t, &y, dt);
        t += dt;
        out.push((t, crate::dsp::phase::wrap_phase(y[1] - y[0])));
    }
    out
}

/// Kuramoto model of n phase oscillators with global coupling K:
/// returns the phase history (per step) and the order parameter r(t).
///
/// # Panics
/// Panics unless `omegas` and `theta0` both have length n.
#[must_use]
pub fn kuramoto(
    n: usize,
    k: f64,
    omegas: &[f64],
    theta0: &[f64],
    t_end: f64,
    dt: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    assert!(omegas.len() == n && theta0.len() == n, "need n frequencies and phases");
    let f = |_t: f64, th: &[f64]| -> Vec<f64> {
        // Mean-field form via the order parameter.
        let (mut sx, mut sy) = (0.0, 0.0);
        for &p in th {
            sx += p.cos();
            sy += p.sin();
        }
        let r = (sx * sx + sy * sy).sqrt() / n as f64;
        let psi = sy.atan2(sx);
        th.iter()
            .zip(omegas)
            .map(|(&p, &w)| w + k * r * (psi - p).sin())
            .collect()
    };
    let mut th = theta0.to_vec();
    let mut t = 0.0;
    let order = |th: &[f64]| -> f64 {
        let (mut sx, mut sy) = (0.0, 0.0);
        for &p in th {
            sx += p.cos();
            sy += p.sin();
        }
        (sx * sx + sy * sy).sqrt() / n as f64
    };
    let mut phases = vec![th.clone()];
    let mut rs = vec![order(&th)];
    while t < t_end - 1e-12 {
        th = rk4_step_vec(&f, t, &th, dt);
        t += dt;
        phases.push(th.clone());
        rs.push(order(&th));
    }
    (phases, rs)
}

/// Critical coupling estimate K_c = 2/(π·g(0)) with the frequency
/// density at the center estimated by a Gaussian kernel (Silverman
/// bandwidth) around the mean frequency.
#[must_use]
pub fn kuramoto_critical_coupling(omegas: &[f64]) -> f64 {
    let n = omegas.len() as f64;
    let mean = omegas.iter().sum::<f64>() / n;
    let var = omegas.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt().max(1e-12);
    let bw = 1.06 * sd * n.powf(-0.2);
    let g0 = omegas
        .iter()
        .map(|&w| (-((w - mean) / bw).powi(2) / 2.0).exp())
        .sum::<f64>()
        / (n * bw * (TWO_PI).sqrt());
    2.0 / (PI * g0)
}

/// Den Hartog tuned-mass-damper design for an undamped primary
/// (m_primary, k_primary) and absorber mass ratio μ: returns the
/// absorber stiffness, damping, and optimal tuning ratio f = 1/(1+μ).
#[must_use]
pub fn tuned_mass_damper_design(
    m_primary: f64,
    k_primary: f64,
    mass_ratio: f64,
) -> (f64, f64, f64) {
    let mu = mass_ratio;
    let m_abs = mu * m_primary;
    let wp = (k_primary / m_primary).sqrt();
    let f_opt = 1.0 / (1.0 + mu);
    let zeta_opt = (3.0 * mu / (8.0 * (1.0 + mu).powi(3))).sqrt();
    let w_abs = f_opt * wp;
    let k_abs = m_abs * w_abs * w_abs;
    let c_abs = 2.0 * m_abs * w_abs * zeta_opt;
    (k_abs, c_abs, f_opt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_fixed_chain_frequencies() {
        let (n, m, k) = (6usize, 1.5, 40.0);
        let sys = CoupledOscillators::chain_fixed_ends(n, m, k);
        let (freqs, shapes) = sys.normal_modes();
        for (i, &f) in freqs.iter().enumerate() {
            let exact =
                2.0 * (k / m).sqrt() * ((i + 1) as f64 * PI / (2.0 * (n as f64 + 1.0))).sin();
            assert!(approx(f, exact, 1e-9), "mode {i}: {f} vs {exact}");
        }
        // Mass-orthonormality ΦᵀMΦ = I.
        for a in 0..n {
            for b in 0..n {
                let dot: f64 =
                    (0..n).map(|r| shapes.get(r, a) * m * shapes.get(r, b)).sum();
                let expect = if a == b { 1.0 } else { 0.0 };
                assert!(approx(dot, expect, 1e-9), "({a},{b}): {dot}");
            }
        }
    }

    #[test]
    fn test_two_coupled_beat() {
        // Two identical grounded oscillators with weak coupling.
        let sys = CoupledOscillators::chain(2, 1.0, 100.0, 1.0);
        let (freqs, _) = sys.normal_modes();
        assert!(approx(freqs[0], 10.0, 1e-9)); // in-phase mode
        assert!(approx(freqs[1], (102.0_f64).sqrt(), 1e-9)); // out-of-phase
        let beat = sys.beat_frequency().unwrap();
        assert!(approx(beat, (102.0_f64).sqrt() - 10.0, 1e-12));
        // Simulate: energy starts in mass 0, transfers to mass 1 after
        // the predicted transfer time.
        let t_transfer = sys.energy_transfer_time().unwrap();
        let dt = 1e-4;
        let traj = sys.simulate(&[1.0, 0.0], &[0.0, 0.0], t_transfer, dt);
        // Around t_transfer the envelope has fully migrated: compare the
        // peak amplitudes over one carrier period at the end.
        let carrier_steps = (TWO_PI / 10.0 / dt) as usize;
        let tail = &traj[traj.len().saturating_sub(carrier_steps)..];
        let peak0 = tail.iter().map(|s| s[0].abs()).fold(0.0_f64, f64::max);
        let peak1 = tail.iter().map(|s| s[1].abs()).fold(0.0_f64, f64::max);
        assert!(peak0 < 0.1, "mass 0 envelope {peak0}");
        assert!(peak1 > 0.9, "mass 1 envelope {peak1}");
        // Modal response matches the simulation midway too.
        let mid = traj.len() / 2;
        let analytic = sys.response(&[1.0, 0.0], &[0.0, 0.0], t_transfer / 2.0);
        assert!(approx(analytic[0], traj[mid][0], 1e-3));
        assert!(approx(analytic[1], traj[mid][1], 1e-3));
    }

    #[test]
    fn test_mode_shape_symmetry_and_analytic_chain() {
        // Two identical coupled masses: the lower mode is symmetric (1,1)
        // and the upper is antisymmetric (1,−1), both mass-normalized so
        // each component is 1/√(2m).
        let (m, k, kc) = (2.0_f64, 100.0_f64, 5.0_f64);
        let sys = CoupledOscillators::chain(2, m, k, kc);
        let (freqs, shapes) = sys.normal_modes();
        let sym = sys.mode_shape(0);
        let anti = sys.mode_shape(1);
        assert_eq!(sym.len(), 2);
        // mode_shape(i) is column i of the normal-mode matrix.
        for r in 0..2 {
            assert!(approx(sym[r], shapes.get(r, 0), 1e-15));
            assert!(approx(anti[r], shapes.get(r, 1), 1e-15));
        }
        let unit = 1.0 / (2.0 * m).sqrt();
        assert!(approx(sym[0], sym[1], 1e-9), "mode 0 is not symmetric: {sym:?}");
        assert!(approx(sym[0].abs(), unit, 1e-9), "normalization {}", sym[0]);
        assert!(approx(anti[0], -anti[1], 1e-9), "mode 1 is not antisymmetric: {anti:?}");
        assert!(approx(anti[0].abs(), unit, 1e-9), "normalization {}", anti[0]);
        // The symmetric mode ignores the coupling spring (ω = √(k/m));
        // the antisymmetric one stretches it (ω = √((k+2k_c)/m)).
        assert!(approx(freqs[0], (k / m).sqrt(), 1e-9), "in-phase {}", freqs[0]);
        assert!(approx(freqs[1], ((k + 2.0 * kc) / m).sqrt(), 1e-9), "out-of-phase {}", freqs[1]);
        // Mass-orthogonality between the two shapes.
        let cross: f64 = (0..2).map(|r| sym[r] * m * anti[r]).sum();
        assert!(cross.abs() < 1e-12, "modes not M-orthogonal: {cross}");
        // Each shape's Rayleigh quotient returns its own ω².
        assert!(approx(sys.rayleigh_quotient(&sym), freqs[0].powi(2), 1e-9));
        assert!(approx(sys.rayleigh_quotient(&anti), freqs[1].powi(2), 1e-9));

        // A fixed-fixed chain has the analytic eigenvectors
        // φ_i[r] ∝ sin((i+1)(r+1)π/(n+1)).
        let n = 5usize;
        let chain = CoupledOscillators::chain_fixed_ends(n, 1.5, 40.0);
        for i in 0..n {
            let shape = chain.mode_shape(i);
            assert_eq!(shape.len(), n);
            let exact: Vec<f64> = (0..n)
                .map(|r| ((i + 1) as f64 * (r + 1) as f64 * PI / (n as f64 + 1.0)).sin())
                .collect();
            // Compare directions (the eigen solve fixes neither sign nor
            // scale): the normalized vectors must be parallel.
            let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let dot: f64 = shape.iter().zip(&exact).map(|(a, b)| a * b).sum();
            let cos = dot / (norm(&shape) * norm(&exact));
            assert!(
                approx(cos.abs(), 1.0, 1e-9),
                "mode {i} is not the analytic shape (cos = {cos})"
            );
            // Mode i has exactly i sign changes (i interior nodes).
            let sign_changes = shape
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            assert_eq!(sign_changes, i, "mode {i} node count");
            // Mass-normalized: Σ mᵢφᵢ² = 1.
            let mass_norm: f64 = shape.iter().map(|x| 1.5 * x * x).sum();
            assert!(approx(mass_norm, 1.0, 1e-9), "mode {i} normalization {mass_norm}");
        }
        // Out-of-range indices are the caller's problem, but index n−1 is
        // the highest mode and must be the most oscillatory.
        let top = chain.mode_shape(n - 1);
        assert_eq!(
            top.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count(),
            n - 1
        );
    }

    #[test]
    fn test_modal_participation_and_rayleigh() {
        let sys = CoupledOscillators::chain_fixed_ends(4, 1.0, 10.0);
        let (freqs, shapes) = sys.normal_modes();
        let phi0: Vec<f64> = (0..4).map(|r| shapes.get(r, 0)).collect();
        // Participation of a pure mode is a unit vector.
        let q = sys.modal_participation(&phi0);
        assert!(approx(q[0], 1.0, 1e-9));
        for &v in &q[1..] {
            assert!(v.abs() < 1e-9);
        }
        // Rayleigh quotient of a mode shape = ω².
        let rq = sys.rayleigh_quotient(&phi0);
        assert!(approx(rq, freqs[0] * freqs[0], 1e-9));
        // Any trial shape bounds the fundamental from above.
        assert!(sys.rayleigh_quotient(&[1.0, 1.0, 1.0, 1.0]) >= freqs[0] * freqs[0] - 1e-12);
        // Dunkerley bounds it from below.
        assert!(sys.dunkerley_estimate() <= freqs[0] + 1e-9);
    }

    #[test]
    fn test_forced_response_and_receptance() {
        let mut sys = CoupledOscillators::chain(2, 1.0, 100.0, 10.0);
        sys.damping.set(0, 0, 0.5);
        sys.damping.set(1, 1, 0.5);
        let omega = 8.0;
        let x = sys.forced_response(&[1.0, 0.0], omega);
        // Verify (K + jωC − ω²M)X = F by residual.
        let n = 2;
        for i in 0..n {
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, xj) in x.iter().enumerate() {
                let kij = sys.stiffness.get(i, j)
                    - if i == j { omega * omega * sys.masses[i] } else { 0.0 };
                let cij = omega * sys.damping.get(i, j);
                re += kij * xj.re - cij * xj.im;
                im += kij * xj.im + cij * xj.re;
            }
            let f = if i == 0 { 1.0 } else { 0.0 };
            assert!(approx(re, f, 1e-9) && approx(im, 0.0, 1e-9), "row {i}");
        }
        // Undamped receptance solves (K−ω²M)H = I.
        let sys0 = CoupledOscillators::chain(2, 1.0, 100.0, 10.0);
        let h = sys0.frequency_response_matrix(omega);
        let mut d = Matrix::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                d.set(i, j, sys0.stiffness.get(i, j) - if i == j { omega * omega } else { 0.0 });
            }
        }
        let prod = d.mul(&h).unwrap();
        assert!(approx(prod.get(0, 0), 1.0, 1e-9) && approx(prod.get(0, 1), 0.0, 1e-9));
    }

    #[test]
    fn test_dispersion_and_ring() {
        let sys = CoupledOscillators::chain_fixed_ends(8, 1.0, 25.0);
        // Fixed chain modes sample the dispersion at q = iπ/(n+1).
        let (freqs, _) = sys.normal_modes();
        for (i, &f) in freqs.iter().enumerate() {
            let q = (i + 1) as f64 * PI / 9.0;
            assert!(approx(sys.dispersion_relation(q), f, 1e-9), "mode {i}");
        }
        // Ring has a zero mode (rigid rotation) and doubly degenerate pairs.
        let ring = CoupledOscillators::ring(6, 1.0, 10.0);
        let (rf, _) = ring.normal_modes();
        assert!(rf[0].abs() < 1e-6);
        assert!(approx(rf[1], rf[2], 1e-9));
    }

    #[test]
    fn test_two_pendulums_and_wilberforce() {
        let sys = two_pendulums_coupled(1.0, 9.81, 2.0, 0.5);
        let (freqs, _) = sys.normal_modes();
        // In-phase mode: √(g/l); out-of-phase: √(g/l + 2k/m).
        assert!(approx(freqs[0], 9.81_f64.sqrt(), 1e-9));
        assert!(approx(freqs[1], (9.81_f64 + 2.0 * 2.0 / 0.5).sqrt(), 1e-9));
        // Wilberforce: tuned k/m = kappa/i gives symmetric splitting.
        let w = wilberforce_pendulum(0.5, 5.0, 1e-4, 1e-3, 1e-3);
        let (wf, _) = w.normal_modes();
        assert!(wf[0] < wf[1]);
        let center = (5.0_f64 / 0.5).sqrt();
        assert!(wf[0] < center && wf[1] > center);
    }

    #[test]
    fn test_kuramoto_synchronization() {
        // Deterministic spread of natural frequencies.
        let n = 40;
        let omegas: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.2 * ((i as f64 / (n - 1) as f64) - 0.5))
            .collect();
        let theta0: Vec<f64> = (0..n).map(|i| (i as f64 * 2.399) % TWO_PI).collect();
        let kc = kuramoto_critical_coupling(&omegas);
        assert!(kc > 0.0);
        // Well above critical: strong synchronization.
        let (_, r_hi) = kuramoto(n, 4.0 * kc, &omegas, &theta0, 60.0, 0.02);
        let tail_hi = r_hi[r_hi.len() - 100..].iter().sum::<f64>() / 100.0;
        assert!(tail_hi > 0.9, "r above critical: {tail_hi}");
        // Well below critical: stays incoherent.
        let (_, r_lo) = kuramoto(n, 0.05 * kc, &omegas, &theta0, 60.0, 0.02);
        let tail_lo = r_lo[r_lo.len() - 100..].iter().sum::<f64>() / 100.0;
        assert!(tail_lo < tail_hi, "r below critical {tail_lo} vs {tail_hi}");
    }

    #[test]
    fn test_huygens_phase_locking() {
        // Identical clocks with coupling: phase difference decays to 0.
        let hist = huygens_sync_simulate(2.0, 2.0, 0.5, 30.0, 0.01);
        let (_, d_end) = hist[hist.len() - 1];
        assert!(d_end.abs() < 1e-3, "residual phase diff {d_end}");
    }

    #[test]
    fn test_tuned_mass_damper_reduces_peak() {
        let (mp, kp) = (10.0, 1000.0);
        let (ka, ca, f_opt) = tuned_mass_damper_design(mp, kp, 0.05);
        assert!(approx(f_opt, 1.0 / 1.05, 1e-12));
        // Bare primary with light damping vs primary + TMD.
        let mut bare = CoupledOscillators::from_springs(&[mp], &[(0, 0, kp)]);
        bare.damping.set(0, 0, 1.0);
        let mut with_tmd =
            CoupledOscillators::from_springs(&[mp, 0.5], &[(0, 0, kp), (0, 1, ka)]);
        with_tmd.damping.set(0, 0, 1.0);
        with_tmd.damping.set(0, 0, with_tmd.damping.get(0, 0) + ca);
        with_tmd.damping.set(1, 1, ca);
        with_tmd.damping.set(0, 1, -ca);
        with_tmd.damping.set(1, 0, -ca);
        let peak = |sys: &CoupledOscillators, dof: usize| -> f64 {
            let mut best = 0.0_f64;
            for i in 1..600 {
                let w = 15.0 * i as f64 / 600.0;
                let f = vec![1.0, 0.0][..sys.masses.len()].to_vec();
                let x = sys.forced_response(&f, w);
                best = best.max(x[dof].norm());
            }
            best
        };
        let p_bare = peak(&bare, 0);
        let p_tmd = peak(&with_tmd, 0);
        assert!(p_tmd < 0.25 * p_bare, "TMD peak {p_tmd} vs bare {p_bare}");
    }

    #[test]
    fn test_anti_resonances_interlace() {
        // Point receptance anti-resonances interlace the resonances.
        let sys = CoupledOscillators::chain(3, 1.0, 50.0, 20.0);
        let (freqs, _) = sys.normal_modes();
        let anti = sys.anti_resonance_frequencies(0, 0);
        assert_eq!(anti.len(), 2, "anti-resonances {anti:?}");
        assert!(anti[0] > freqs[0] && anti[0] < freqs[1]);
        assert!(anti[1] > freqs[1] && anti[1] < freqs[2]);
    }
}
