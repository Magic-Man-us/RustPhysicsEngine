//! Turbulence modelling and statistics.
//!
//! Kolmogorov scaling, energy spectra, LES subgrid models (Smagorinsky,
//! dynamic Smagorinsky, WALE, Vreman), vortex identification criteria,
//! RANS models (k-epsilon, k-omega SST, Spalart-Allmaras), synthetic
//! turbulence generation, and canonical spectra (von Karman, Pao).

use crate::fractals::Complex;
use crate::linalg::Mat3;
use crate::transforms::fft::{fft_2d, fft_3d, rfft};

const KOLMOGOROV_CONST: f64 = 1.5;
const KAPPA_VK: f64 = 0.41;

// ---------------------------------------------------------------------------
// Kolmogorov scaling
// ---------------------------------------------------------------------------

/// Kolmogorov -5/3 inertial-range energy spectrum E(k) = C eps^{2/3} k^{-5/3}.
pub fn kolmogorov_spectrum(k: f64, dissipation: f64) -> f64 {
    if k <= 0.0 {
        return 0.0;
    }
    KOLMOGOROV_CONST * dissipation.powf(2.0 / 3.0) * k.powf(-5.0 / 3.0)
}

/// Kolmogorov length, time and velocity scales `(eta, tau, u)` from
/// kinematic viscosity and dissipation rate.
pub fn kolmogorov_scales(nu: f64, dissipation: f64) -> (f64, f64, f64) {
    let eta = (nu.powi(3) / dissipation).powf(0.25);
    let tau = (nu / dissipation).sqrt();
    let u = (nu * dissipation).powf(0.25);
    (eta, tau, u)
}

/// Taylor microscale lambda = sqrt(15 nu u'^2 / eps).
pub fn taylor_microscale(u_rms: f64, nu: f64, dissipation: f64) -> f64 {
    (15.0 * nu * u_rms * u_rms / dissipation).sqrt()
}

/// Integral length scale estimate L = u'^3 / eps.
pub fn integral_scale(u_rms: f64, dissipation: f64) -> f64 {
    u_rms.powi(3) / dissipation
}

/// Taylor-microscale Reynolds number Re_lambda = u' lambda / nu.
pub fn re_lambda(u_rms: f64, nu: f64, dissipation: f64) -> f64 {
    u_rms * taylor_microscale(u_rms, nu, dissipation) / nu
}

// ---------------------------------------------------------------------------
// Spectra from velocity fields
// ---------------------------------------------------------------------------

/// One-dimensional energy spectrum of a periodic velocity signal sampled at
/// spacing `dx`. Returns `(k, e_k)` with k in rad per unit length. The sum of
/// `e_k` times dk equals half the mean-square fluctuation.
pub fn energy_spectrum_1d(u: &[f64], dx: f64) -> (Vec<f64>, Vec<f64>) {
    let n = u.len();
    let mean = u.iter().sum::<f64>() / n as f64;
    let fluct: Vec<f64> = u.iter().map(|&v| v - mean).collect();
    let spec = rfft(&fluct);
    let l = n as f64 * dx;
    let dk = 2.0 * std::f64::consts::PI / l;
    let nk = n / 2;
    let mut ks = Vec::with_capacity(nk);
    let mut es = Vec::with_capacity(nk);
    for (m, bin) in spec.iter().enumerate().take(nk + 1).skip(1) {
        let re = bin.re / n as f64;
        let im = bin.im / n as f64;
        let mut e = re * re + im * im;
        // one-sided: double all but the Nyquist bin (n even, m == n/2)
        if !(n.is_multiple_of(2) && m == n / 2) {
            e *= 2.0;
        }
        // E(k) dk = 0.5 |u_hat|^2 contribution
        ks.push(m as f64 * dk);
        es.push(0.5 * e / dk);
    }
    (ks, es)
}

/// Shell-averaged 2D energy spectrum of a periodic (u, v) field on an
/// `n` x `n` grid with spacing `dx`. Returns `(k, e_k)` binned on integer
/// wavenumber shells.
pub fn energy_spectrum_2d(u: &[f64], v: &[f64], n: usize, dx: f64) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u.len(), n * n);
    assert_eq!(v.len(), n * n);
    let cu: Vec<Complex> = u.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let cv: Vec<Complex> = v.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let uh = fft_2d(&cu, n, n);
    let vh = fft_2d(&cv, n, n);
    let l = n as f64 * dx;
    let dk = 2.0 * std::f64::consts::PI / l;
    let nk = n / 2;
    let mut e = vec![0.0; nk + 1];
    let norm = 1.0 / (n as f64 * n as f64);
    for kj in 0..n {
        let kjs = if kj <= n / 2 { kj as f64 } else { kj as f64 - n as f64 };
        for ki in 0..n {
            let kis = if ki <= n / 2 { ki as f64 } else { ki as f64 - n as f64 };
            let km = (kis * kis + kjs * kjs).sqrt();
            let shell = km.round() as usize;
            if shell == 0 || shell > nk {
                continue;
            }
            let idx = kj * n + ki;
            let eu = (uh[idx].re * norm).powi(2) + (uh[idx].im * norm).powi(2);
            let ev = (vh[idx].re * norm).powi(2) + (vh[idx].im * norm).powi(2);
            e[shell] += 0.5 * (eu + ev);
        }
    }
    let ks: Vec<f64> = (1..=nk).map(|m| m as f64 * dk).collect();
    let es: Vec<f64> = (1..=nk).map(|m| e[m] / dk).collect();
    (ks, es)
}

/// Shell-averaged 3D energy spectrum of periodic (u, v, w) on an n^3 grid
/// (index `(k*n + j)*n + i`) with spacing `dx`.
pub fn energy_spectrum_3d(
    u: &[f64],
    v: &[f64],
    w: &[f64],
    n: usize,
    dx: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u.len(), n * n * n);
    let cu: Vec<Complex> = u.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let cv: Vec<Complex> = v.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let cw: Vec<Complex> = w.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let uh = fft_3d(&cu, n, n, n);
    let vh = fft_3d(&cv, n, n, n);
    let wh = fft_3d(&cw, n, n, n);
    let l = n as f64 * dx;
    let dk = 2.0 * std::f64::consts::PI / l;
    let nk = n / 2;
    let mut e = vec![0.0; nk + 1];
    let norm = 1.0 / (n as f64).powi(3);
    for kk in 0..n {
        let kks = if kk <= n / 2 { kk as f64 } else { kk as f64 - n as f64 };
        for kj in 0..n {
            let kjs = if kj <= n / 2 { kj as f64 } else { kj as f64 - n as f64 };
            for ki in 0..n {
                let kis = if ki <= n / 2 { ki as f64 } else { ki as f64 - n as f64 };
                let km = (kis * kis + kjs * kjs + kks * kks).sqrt();
                let shell = km.round() as usize;
                if shell == 0 || shell > nk {
                    continue;
                }
                let idx = (kk * n + kj) * n + ki;
                let es = ((uh[idx].re * norm).powi(2) + (uh[idx].im * norm).powi(2))
                    + ((vh[idx].re * norm).powi(2) + (vh[idx].im * norm).powi(2))
                    + ((wh[idx].re * norm).powi(2) + (wh[idx].im * norm).powi(2));
                e[shell] += 0.5 * es;
            }
        }
    }
    let ks: Vec<f64> = (1..=nk).map(|m| m as f64 * dk).collect();
    let es: Vec<f64> = (1..=nk).map(|m| e[m] / dk).collect();
    (ks, es)
}

/// Dissipation rate from a spectrum: eps = 2 nu integral k^2 E(k) dk
/// (trapezoidal).
pub fn dissipation_rate_from_spectrum(k: &[f64], e: &[f64], nu: f64) -> f64 {
    assert_eq!(k.len(), e.len());
    let mut integral = 0.0;
    for i in 1..k.len() {
        let f0 = k[i - 1] * k[i - 1] * e[i - 1];
        let f1 = k[i] * k[i] * e[i];
        integral += 0.5 * (f0 + f1) * (k[i] - k[i - 1]);
    }
    2.0 * nu * integral
}

/// Longitudinal structure function of order `p` at separation `r` (in
/// samples times dx) for a periodic 1D signal: <|u(x+r) - u(x)|^p>.
pub fn structure_function(u: &[f64], sep: usize, order: i32) -> f64 {
    let n = u.len();
    let mut acc = 0.0;
    for i in 0..n {
        let d = (u[(i + sep) % n] - u[i]).abs();
        acc += d.powi(order);
    }
    acc / n as f64
}

/// Two-point autocorrelation of a periodic 1D signal at separation `sep`
/// (samples), normalized to R(0) = 1.
pub fn two_point_correlation(u: &[f64], sep: usize) -> f64 {
    let n = u.len();
    let mean = u.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let a = u[i] - mean;
        let b = u[(i + sep) % n] - mean;
        num += a * b;
        den += a * a;
    }
    if den == 0.0 { 0.0 } else { num / den }
}

// ---------------------------------------------------------------------------
// Velocity-gradient tensors and LES subgrid models
// ---------------------------------------------------------------------------

/// Symmetric strain-rate tensor S = (grad u + grad u^T)/2 from a velocity
/// gradient tensor g where `g.data[i][j] = du_i/dx_j`.
pub fn strain_tensor(g: &Mat3) -> Mat3 {
    let mut s = [[0.0; 3]; 3];
    for (i, row) in s.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = 0.5 * (g.data[i][j] + g.data[j][i]);
        }
    }
    Mat3 { data: s }
}

/// Antisymmetric rotation-rate tensor W = (grad u - grad u^T)/2.
pub fn rotation_tensor(g: &Mat3) -> Mat3 {
    let mut w = [[0.0; 3]; 3];
    for (i, row) in w.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = 0.5 * (g.data[i][j] - g.data[j][i]);
        }
    }
    Mat3 { data: w }
}

fn frob2(m: &Mat3) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += m.data[i][j] * m.data[i][j];
        }
    }
    s
}

/// Smagorinsky eddy viscosity nu_t = (Cs Delta)^2 |S|, |S| = sqrt(2 S:S).
pub fn smagorinsky_nu_t(g: &Mat3, delta: f64, cs: f64) -> f64 {
    let s = strain_tensor(g);
    let s_mag = (2.0 * frob2(&s)).sqrt();
    (cs * delta).powi(2) * s_mag
}

/// Germano-Lilly dynamic Smagorinsky coefficient from resolved and
/// test-filtered fields. `l` is the Leonard stress tensor and `m` the model
/// difference tensor; returns Cs^2 = <L:M> / <M:M> clipped at zero.
pub fn dynamic_smagorinsky_cs(l: &Mat3, m: &Mat3) -> f64 {
    let mut lm = 0.0;
    let mut mm = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            lm += l.data[i][j] * m.data[i][j];
            mm += m.data[i][j] * m.data[i][j];
        }
    }
    if mm <= 0.0 { 0.0 } else { (lm / mm).max(0.0) }
}

/// WALE subgrid eddy viscosity (Nicoud & Ducros 1999) with constant `cw`
/// (typically 0.325 to 0.5).
pub fn wale_nu_t(g: &Mat3, delta: f64, cw: f64) -> f64 {
    // gd = symmetric traceless part of g^2
    let mut g2 = [[0.0; 3]; 3];
    for (i, row) in g2.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            for (k, gk) in g.data.iter().enumerate() {
                *v += g.data[i][k] * gk[j];
            }
        }
    }
    let tr = (g2[0][0] + g2[1][1] + g2[2][2]) / 3.0;
    let mut sd2 = 0.0;
    for (i, row) in g2.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let mut sd = 0.5 * (v + g2[j][i]);
            if i == j {
                sd -= tr;
            }
            sd2 += sd * sd;
        }
    }
    let s = strain_tensor(g);
    let ss = frob2(&s);
    let denom = ss.powf(2.5) + sd2.powf(1.25);
    if denom <= 0.0 {
        return 0.0;
    }
    (cw * delta).powi(2) * sd2.powf(1.5) / denom
}

/// Vreman subgrid eddy viscosity (Vreman 2004) with constant `c`
/// (approximately 2.5 Cs^2, so about 0.07).
pub fn vreman_nu_t(g: &Mat3, delta: f64, c: f64) -> f64 {
    // alpha_ij = du_j/dx_i (transpose of our g); beta = delta^2 alpha^T alpha
    let a = &g.data;
    let mut aa = 0.0;
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // alpha[i][j] = a[j][i]
            aa += a[j][i] * a[j][i];
            for am in a {
                b[i][j] += delta * delta * am[i] * am[j];
            }
        }
    }
    let bb = b[0][0] * b[1][1] - b[0][1] * b[0][1] + b[0][0] * b[2][2] - b[0][2] * b[0][2]
        + b[1][1] * b[2][2] - b[1][2] * b[1][2];
    if aa <= 1e-30 || bb <= 0.0 {
        return 0.0;
    }
    c * (bb / aa).sqrt()
}

// ---------------------------------------------------------------------------
// Vortex identification
// ---------------------------------------------------------------------------

/// Q-criterion: Q = (|W|^2 - |S|^2)/2. Positive Q marks vortical regions.
pub fn q_criterion(g: &Mat3) -> f64 {
    let s = strain_tensor(g);
    let w = rotation_tensor(g);
    0.5 * (frob2(&w) - frob2(&s))
}

/// Lambda-2 criterion: middle eigenvalue of S^2 + W^2. Negative values mark
/// vortex cores.
pub fn lambda2_criterion(g: &Mat3) -> f64 {
    let s = strain_tensor(g);
    let w = rotation_tensor(g);
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            for k in 0..3 {
                *v += s.data[i][k] * s.data[k][j] + w.data[i][k] * w.data[k][j];
            }
        }
    }
    // symmetric 3x3 eigenvalues via trigonometric formula
    let p1 = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
    let q = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
    if p1.abs() < 1e-30 {
        let mut d = [m[0][0], m[1][1], m[2][2]];
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return d[1];
    }
    let p2 = (m[0][0] - q).powi(2) + (m[1][1] - q).powi(2) + (m[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let mut bmat = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            bmat[i][j] = (m[i][j] - if i == j { q } else { 0.0 }) / p;
        }
    }
    let detb = bmat[0][0] * (bmat[1][1] * bmat[2][2] - bmat[1][2] * bmat[2][1])
        - bmat[0][1] * (bmat[1][0] * bmat[2][2] - bmat[1][2] * bmat[2][0])
        + bmat[0][2] * (bmat[1][0] * bmat[2][1] - bmat[1][1] * bmat[2][0]);
    let r = (detb / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    3.0 * q - e1 - e3
}

/// Delta criterion: Delta = (Q/3)^3 + (det(g)/2)^2 > 0 marks complex
/// eigenvalues of the velocity gradient (swirling motion).
pub fn delta_criterion(g: &Mat3) -> f64 {
    let q = q_criterion(g);
    let det = g.determinant();
    (q / 3.0).powi(3) + (det / 2.0).powi(2)
}

/// Identify vortical cells in a 2D periodic velocity field on an n x n grid
/// (spacing `dx`) by the 2D Q-criterion; returns flags where Q > `threshold`.
pub fn vortex_identify_q(
    u: &[f64],
    v: &[f64],
    n: usize,
    dx: f64,
    threshold: f64,
) -> Vec<bool> {
    let idx = |i: usize, j: usize| j * n + i;
    let mut out = vec![false; n * n];
    for j in 0..n {
        let jp = (j + 1) % n;
        let jm = (j + n - 1) % n;
        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;
            let dudx = (u[idx(ip, j)] - u[idx(im, j)]) / (2.0 * dx);
            let dudy = (u[idx(i, jp)] - u[idx(i, jm)]) / (2.0 * dx);
            let dvdx = (v[idx(ip, j)] - v[idx(im, j)]) / (2.0 * dx);
            let dvdy = (v[idx(i, jp)] - v[idx(i, jm)]) / (2.0 * dx);
            // 2D: Q = -(dudx*dvdy - dudy*dvdx) sign convention via S/W norms
            let s2 = dudx * dudx + dvdy * dvdy + 0.5 * (dudy + dvdx).powi(2);
            let w2 = 0.5 * (dudy - dvdx).powi(2);
            out[idx(i, j)] = 0.5 * (w2 - s2) > threshold;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// RANS models
// ---------------------------------------------------------------------------

/// Which k-epsilon variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KEpsilonVariant {
    Standard,
    Rng,
    Realizable,
}

/// Homogeneous (0D) k-epsilon model state advanced by production balance.
#[derive(Debug, Clone)]
pub struct KEpsilon {
    pub k: f64,
    pub epsilon: f64,
    pub nu: f64,
    pub variant: KEpsilonVariant,
    pub c_mu: f64,
    pub c1: f64,
    pub c2: f64,
    pub sigma_k: f64,
    pub sigma_eps: f64,
}

impl KEpsilon {
    pub fn new(k0: f64, eps0: f64, nu: f64, variant: KEpsilonVariant) -> Self {
        let (c_mu, c1, c2) = match variant {
            KEpsilonVariant::Standard => (0.09, 1.44, 1.92),
            KEpsilonVariant::Rng => (0.0845, 1.42, 1.68),
            KEpsilonVariant::Realizable => (0.09, 1.44, 1.9),
        };
        Self {
            k: k0,
            epsilon: eps0,
            nu,
            variant,
            c_mu,
            c1,
            c2,
            sigma_k: 1.0,
            sigma_eps: 1.3,
        }
    }

    /// Initialize from turbulence intensity `ti` (fraction), mean speed and a
    /// length scale: k = 1.5 (ti U)^2, eps = C_mu^{3/4} k^{3/2} / L.
    pub fn init_from_intensity(ti: f64, u_mean: f64, length: f64, nu: f64) -> Self {
        let k = 1.5 * (ti * u_mean).powi(2);
        let eps = 0.09_f64.powf(0.75) * k.powf(1.5) / length;
        Self::new(k, eps, nu, KEpsilonVariant::Standard)
    }

    /// Eddy viscosity nu_t = C_mu k^2 / eps.
    pub fn nu_t(&self) -> f64 {
        if self.epsilon <= 0.0 {
            return 0.0;
        }
        self.c_mu * self.k * self.k / self.epsilon
    }

    /// Production term P_k = nu_t |S|^2 with |S| = sqrt(2 S:S) given as
    /// `s_mag`.
    pub fn production(&self, s_mag: f64) -> f64 {
        self.nu_t() * s_mag * s_mag
    }

    /// Advance the homogeneous model one step of size `dt` under mean strain
    /// magnitude `s_mag`:
    /// dk/dt = P - eps, deps/dt = (C1 P - C2 eps) eps / k.
    pub fn step(&mut self, s_mag: f64, dt: f64) {
        let p = self.production(s_mag);
        let mut c1 = self.c1;
        if self.variant == KEpsilonVariant::Rng {
            // RNG strain correction
            let eta = s_mag * self.k / self.epsilon.max(1e-30);
            let eta0 = 4.38;
            let beta = 0.012;
            c1 -= eta * (1.0 - eta / eta0) / (1.0 + beta * eta.powi(3));
        }
        let k_new = (self.k + dt * (p - self.epsilon)).max(1e-12);
        let eps_new = (self.epsilon
            + dt * (c1 * p - self.c2 * self.epsilon) * self.epsilon / self.k.max(1e-30))
        .max(1e-14);
        self.k = k_new;
        self.epsilon = eps_new;
    }

    /// Standard wall-function values at wall distance `y` for friction
    /// velocity `u_tau`: returns `(k, epsilon)` in the log layer.
    pub fn wall_function(&self, u_tau: f64, y: f64) -> (f64, f64) {
        let k = u_tau * u_tau / self.c_mu.sqrt();
        let eps = u_tau.powi(3) / (KAPPA_VK * y);
        (k, eps)
    }
}

/// Homogeneous k-omega SST model (Menter 1994), blended toward k-omega near
/// `blend = 1` and k-epsilon at `blend = 0`.
#[derive(Debug, Clone)]
pub struct KOmegaSst {
    pub k: f64,
    pub omega: f64,
    pub nu: f64,
    pub a1: f64,
    pub beta_star: f64,
}

impl KOmegaSst {
    pub fn new(k0: f64, omega0: f64, nu: f64) -> Self {
        Self {
            k: k0,
            omega: omega0,
            nu,
            a1: 0.31,
            beta_star: 0.09,
        }
    }

    /// SST eddy viscosity with the shear limiter:
    /// nu_t = a1 k / max(a1 omega, |S| F2); pass F2 = 1 in free shear.
    pub fn nu_t(&self, s_mag: f64, f2: f64) -> f64 {
        self.a1 * self.k / (self.a1 * self.omega).max(s_mag * f2)
    }

    /// Advance the homogeneous model with blending function `f1` in [0, 1]
    /// (1 = inner k-omega constants, 0 = transformed k-epsilon constants).
    pub fn step(&mut self, s_mag: f64, f1: f64, dt: f64) {
        let (alpha1, beta1) = (5.0 / 9.0, 0.075);
        let (alpha2, beta2) = (0.44, 0.0828);
        let alpha = f1 * alpha1 + (1.0 - f1) * alpha2;
        let beta = f1 * beta1 + (1.0 - f1) * beta2;
        let nu_t = self.nu_t(s_mag, 1.0);
        let p = (nu_t * s_mag * s_mag).min(10.0 * self.beta_star * self.k * self.omega);
        let k_new = (self.k + dt * (p - self.beta_star * self.k * self.omega)).max(1e-12);
        let om_new = (self.omega
            + dt * (alpha * s_mag * s_mag - beta * self.omega * self.omega))
        .max(1e-12);
        self.k = k_new;
        self.omega = om_new;
    }
}

/// Homogeneous Spalart-Allmaras one-equation model (no wall term).
#[derive(Debug, Clone)]
pub struct SpalartAllmaras {
    pub nu_tilde: f64,
    pub nu: f64,
}

impl SpalartAllmaras {
    pub fn new(nu_tilde0: f64, nu: f64) -> Self {
        Self {
            nu_tilde: nu_tilde0,
            nu,
        }
    }

    /// Eddy viscosity nu_t = nu_tilde fv1, fv1 = chi^3/(chi^3 + cv1^3).
    pub fn nu_t(&self) -> f64 {
        let cv1: f64 = 7.1;
        let chi = self.nu_tilde / self.nu;
        let fv1 = chi.powi(3) / (chi.powi(3) + cv1.powi(3));
        self.nu_tilde * fv1
    }

    /// Advance with mean vorticity magnitude `omega_mag` and wall distance
    /// `d` (destruction active for finite d).
    pub fn step(&mut self, omega_mag: f64, d: f64, dt: f64) {
        let cb1 = 0.1355;
        let cb2: f64 = 0.622;
        let sigma = 2.0 / 3.0;
        let cw1 = cb1 / (KAPPA_VK * KAPPA_VK) + (1.0 + cb2) / sigma;
        let cv1: f64 = 7.1;
        let chi = self.nu_tilde / self.nu;
        let fv1 = chi.powi(3) / (chi.powi(3) + cv1.powi(3));
        let fv2 = 1.0 - chi / (1.0 + chi * fv1);
        let s_tilde = (omega_mag
            + self.nu_tilde / (KAPPA_VK * KAPPA_VK * d * d) * fv2)
            .max(0.3 * omega_mag);
        let r = (self.nu_tilde / (s_tilde * KAPPA_VK * KAPPA_VK * d * d).max(1e-30)).min(10.0);
        let cw2 = 0.3;
        let cw3: f64 = 2.0;
        let g = r + cw2 * (r.powi(6) - r);
        let fw = g * ((1.0 + cw3.powi(6)) / (g.powi(6) + cw3.powi(6))).powf(1.0 / 6.0);
        let prod = cb1 * s_tilde * self.nu_tilde;
        let dest = cw1 * fw * (self.nu_tilde / d).powi(2);
        self.nu_tilde = (self.nu_tilde + dt * (prod - dest)).max(0.0);
    }
}

/// Common interface for homogeneous RANS models.
pub trait RansModel {
    /// Current eddy viscosity.
    fn eddy_viscosity(&self) -> f64;
    /// Advance one step under mean strain magnitude `s_mag`.
    fn advance(&mut self, s_mag: f64, dt: f64);
}

impl RansModel for KEpsilon {
    fn eddy_viscosity(&self) -> f64 {
        self.nu_t()
    }
    fn advance(&mut self, s_mag: f64, dt: f64) {
        self.step(s_mag, dt);
    }
}

impl RansModel for KOmegaSst {
    fn eddy_viscosity(&self) -> f64 {
        self.nu_t(0.0, 1.0)
    }
    fn advance(&mut self, s_mag: f64, dt: f64) {
        self.step(s_mag, 1.0, dt);
    }
}

impl RansModel for SpalartAllmaras {
    fn eddy_viscosity(&self) -> f64 {
        self.nu_t()
    }
    fn advance(&mut self, s_mag: f64, dt: f64) {
        // interpret s_mag as vorticity magnitude far from walls
        self.step(s_mag, 1e6, dt);
    }
}

/// Advance any RANS model `n` steps and return the eddy-viscosity history.
pub fn rans_step(model: &mut dyn RansModel, s_mag: f64, dt: f64, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        model.advance(s_mag, dt);
        out.push(model.eddy_viscosity());
    }
    out
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Turbulence intensity: rms fluctuation over mean speed.
pub fn turbulence_intensity(u: &[f64]) -> f64 {
    let n = u.len() as f64;
    let mean = u.iter().sum::<f64>() / n;
    let var = u.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    if mean.abs() < 1e-30 {
        return 0.0;
    }
    var.sqrt() / mean.abs()
}

/// Reynolds stress component <u'v'> from paired samples.
pub fn reynolds_stress(u: &[f64], v: &[f64]) -> f64 {
    assert_eq!(u.len(), v.len());
    let n = u.len() as f64;
    let um = u.iter().sum::<f64>() / n;
    let vm = v.iter().sum::<f64>() / n;
    u.iter()
        .zip(v)
        .map(|(&a, &b)| (a - um) * (b - vm))
        .sum::<f64>()
        / n
}

// ---------------------------------------------------------------------------
// Synthetic turbulence
// ---------------------------------------------------------------------------

/// Divergence-free synthetic 2D turbulence (Kraichnan-style random Fourier
/// modes) on an n x n periodic grid of size `l`. Each of `n_modes` modes has a
/// random wavevector with magnitude near `k_peak` and an amplitude direction
/// perpendicular to it. Returns `(u, v)`.
pub fn synthetic_turbulence_kraichnan(
    n: usize,
    l: f64,
    n_modes: usize,
    k_peak: f64,
    u_rms: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let dk = 2.0 * std::f64::consts::PI / l;
    let mut modes = Vec::with_capacity(n_modes);
    for _ in 0..n_modes {
        // wavevector on the integer lattice near k_peak with random direction
        let theta = 2.0 * std::f64::consts::PI * rand();
        let km = k_peak * (0.5 + rand());
        let kx = (km * theta.cos() / dk).round() * dk;
        let ky = (km * theta.sin() / dk).round() * dk;
        if kx == 0.0 && ky == 0.0 {
            continue;
        }
        let phase = 2.0 * std::f64::consts::PI * rand();
        let amp = rand();
        modes.push((kx, ky, phase, amp));
    }
    let mut u = vec![0.0; n * n];
    let mut v = vec![0.0; n * n];
    let dx = l / n as f64;
    for j in 0..n {
        let y = j as f64 * dx;
        for i in 0..n {
            let x = i as f64 * dx;
            let mut uu = 0.0;
            let mut vv = 0.0;
            for &(kx, ky, phase, amp) in &modes {
                let km = (kx * kx + ky * ky).sqrt();
                // unit vector perpendicular to k gives div-free mode
                let (ex, ey) = (-ky / km, kx / km);
                let c = (kx * x + ky * y + phase).cos();
                uu += amp * ex * c;
                vv += amp * ey * c;
            }
            u[j * n + i] = uu;
            v[j * n + i] = vv;
        }
    }
    // rescale to requested rms
    let ms = u
        .iter()
        .zip(&v)
        .map(|(&a, &b)| a * a + b * b)
        .sum::<f64>()
        / (n * n) as f64;
    let rms = (ms / 2.0).sqrt();
    if rms > 0.0 {
        let sc = u_rms / rms;
        for (a, b) in u.iter_mut().zip(v.iter_mut()) {
            *a *= sc;
            *b *= sc;
        }
    }
    (u, v)
}

/// Synthetic eddy method: superpose `n_eddies` compact Gaussian eddies with
/// random centers and signs in a periodic box of size `l`, sampled at `n_pts`
/// points along a line. Returns a fluctuation signal scaled to `u_rms`.
pub fn synthetic_eddy_method(
    n_pts: usize,
    l: f64,
    n_eddies: usize,
    eddy_size: f64,
    u_rms: f64,
    seed: u64,
) -> Vec<f64> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    let mut rand = || {
        state = state
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut u = vec![0.0; n_pts];
    let dx = l / n_pts as f64;
    for _ in 0..n_eddies {
        let xc = l * rand();
        let sign = if rand() < 0.5 { -1.0 } else { 1.0 };
        for (i, ui) in u.iter_mut().enumerate() {
            let mut r = (i as f64 * dx - xc).abs();
            if r > l / 2.0 {
                r = l - r;
            }
            *ui += sign * (-0.5 * (r / eddy_size).powi(2)).exp();
        }
    }
    let mean = u.iter().sum::<f64>() / n_pts as f64;
    let var = u.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_pts as f64;
    if var > 0.0 {
        let sc = u_rms / var.sqrt();
        for ui in u.iter_mut() {
            *ui = (*ui - mean) * sc;
        }
    }
    u
}

// ---------------------------------------------------------------------------
// Model spectra
// ---------------------------------------------------------------------------

/// Von Karman model spectrum with integral-scale wavenumber `ke` and
/// Kolmogorov cutoff `k_eta`:
/// E(k) ~ (k/ke)^4 / (1 + (k/ke)^2)^{17/6} * exp(-2 (k/k_eta)^2) scaled so
/// the peak region carries energy `k_energy` overall (approximate).
pub fn von_karman_spectrum(k: f64, k_energy: f64, ke: f64, k_eta: f64) -> f64 {
    if k <= 0.0 {
        return 0.0;
    }
    let x = k / ke;
    let shape = x.powi(4) / (1.0 + x * x).powf(17.0 / 6.0);
    // normalization: integral of shape dk over (0, inf) without cutoff is
    // ke * B(5/2, 1/3) / 2 = ke * 1.032496
    let a = k_energy / (1.032496 * ke);
    a * shape * (-2.0 * (k / k_eta).powi(2)).exp()
}

/// Pao dissipation-range spectrum:
/// E(k) = C eps^{2/3} k^{-5/3} exp(-1.5 C (k eta)^{4/3}).
pub fn pao_spectrum(k: f64, dissipation: f64, nu: f64) -> f64 {
    if k <= 0.0 {
        return 0.0;
    }
    let (eta, _, _) = kolmogorov_scales(nu, dissipation);
    KOLMOGOROV_CONST
        * dissipation.powf(2.0 / 3.0)
        * k.powf(-5.0 / 3.0)
        * (-1.5 * KOLMOGOROV_CONST * (k * eta).powf(4.0 / 3.0)).exp()
}

// ---------------------------------------------------------------------------
// Wall-bounded utilities
// ---------------------------------------------------------------------------

/// Fit the log law u+ = (1/kappa) ln y+ + B to a profile `(y, u)` for a fluid
/// of viscosity `nu`, returning `(u_tau, b)`. The friction velocity comes from
/// the slope of u versus ln y: u_tau = kappa * slope.
pub fn log_law_fit(y: &[f64], u: &[f64], nu: f64) -> (f64, f64) {
    assert_eq!(y.len(), u.len());
    let n = y.len() as f64;
    let lx: Vec<f64> = y.iter().map(|&yy| yy.ln()).collect();
    let mx = lx.iter().sum::<f64>() / n;
    let my = u.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (l, &uu) in lx.iter().zip(u) {
        num += (l - mx) * (uu - my);
        den += (l - mx) * (l - mx);
    }
    let slope = num / den;
    let intercept = my - slope * mx;
    let u_tau = KAPPA_VK * slope;
    // u = u_tau/kappa (ln y + ln(u_tau/nu)) + B u_tau
    let b = intercept / u_tau - (u_tau / nu).ln() / KAPPA_VK;
    (u_tau, b)
}

/// Reference mean-velocity profile for turbulent channel flow at friction
/// Reynolds number `re_tau`: Reichardt's composite profile evaluated at
/// `n` points from the wall to the centerline. Returns `(y_plus, u_plus)`.
pub fn channel_flow_dns_reference(re_tau: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut yp = Vec::with_capacity(n);
    let mut up = Vec::with_capacity(n);
    for i in 1..=n {
        let y = re_tau * i as f64 / n as f64;
        let u = (1.0 / KAPPA_VK) * (1.0 + KAPPA_VK * y).ln()
            + 7.8 * (1.0 - (-y / 11.0).exp() - (y / 11.0) * (-y / 3.0).exp());
        yp.push(y);
        up.push(u);
    }
    (yp, up)
}

/// Log-log slope of a spectrum over the wavenumber band `[k_lo, k_hi]`
/// (least squares). For an inertial range this returns about -5/3.
pub fn inertial_range_exponent(k: &[f64], e: &[f64], k_lo: f64, k_hi: f64) -> f64 {
    let mut lx = Vec::new();
    let mut ly = Vec::new();
    for (&kk, &ee) in k.iter().zip(e) {
        if kk >= k_lo && kk <= k_hi && ee > 0.0 {
            lx.push(kk.ln());
            ly.push(ee.ln());
        }
    }
    let n = lx.len() as f64;
    let mx = lx.iter().sum::<f64>() / n;
    let my = ly.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (l, yv) in lx.iter().zip(&ly) {
        num += (l - mx) * (yv - my);
        den += (l - mx) * (l - mx);
    }
    num / den
}

/// Eddy-turnover (cascade) time at scale `l` with velocity `u_l`:
/// tau = l / u_l.
pub fn richardson_cascade_time(l: f64, u_l: f64) -> f64 {
    l / u_l
}

/// Turbulent diffusivity nu_t / Pr_t.
pub fn turbulent_diffusivity(nu_t: f64, pr_t: f64) -> f64 {
    nu_t / pr_t
}

/// Set up a decaying isotropic turbulence run: a StableFluid3 on an n^3
/// periodic box seeded with divergence-free random modes, advanced to
/// `t_end`. Returns the fluid for inspection. Keep `n` small (16 or so).
pub fn decaying_isotropic_turbulence(
    n: usize,
    nu: f64,
    t_end: f64,
) -> crate::cfd::stable_fluids::StableFluid3 {
    let l = 2.0 * std::f64::consts::PI;
    let dx = l / n as f64;
    let mut fluid = crate::cfd::stable_fluids::StableFluid3::new(n, n, n, dx);
    // seed with a Taylor-Green-like divergence-free field plus a second mode
    // so it decays through nonlinear interaction; write MAC faces directly
    for k in 0..n {
        let zc = (k as f64 + 0.5) * dx;
        for j in 0..n {
            let yc = (j as f64 + 0.5) * dx;
            for i in 0..=n {
                let xf = i as f64 * dx;
                fluid.grid.u[(k * n + j) * (n + 1) + i] = xf.sin() * yc.cos() * zc.cos()
                    + 0.3 * (2.0 * xf).sin() * (2.0 * yc).cos() * zc.cos();
            }
        }
    }
    for k in 0..n {
        let zc = (k as f64 + 0.5) * dx;
        for j in 0..=n {
            let yf = j as f64 * dx;
            for i in 0..n {
                let xc = (i as f64 + 0.5) * dx;
                fluid.grid.v[(k * (n + 1) + j) * n + i] = -xc.cos() * yf.sin() * zc.cos()
                    - 0.3 * (2.0 * xc).cos() * (2.0 * yf).sin() * zc.cos();
            }
        }
    }
    let dt = 0.5 * dx;
    let steps = (t_end / dt).ceil() as usize;
    // explicit viscous damping factor applied per step (dominant k^2 ~ 3 for
    // the seeded modes; kept simple since this is a demonstration setup)
    let damp = (-3.0 * nu * dt).exp();
    for _ in 0..steps.max(1) {
        fluid.step(dt);
        for f in fluid
            .grid
            .u
            .iter_mut()
            .chain(fluid.grid.v.iter_mut())
            .chain(fluid.grid.w.iter_mut())
        {
            *f *= damp;
        }
    }
    fluid
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: {a} vs {b}");
    }

    #[test]
    fn test_kolmogorov_scaling() {
        let nu = 1e-5;
        let eps = 0.1;
        let (eta, tau, ueta) = kolmogorov_scales(nu, eps);
        // Re at Kolmogorov scale is unity
        assert_close(ueta * eta / nu, 1.0, 1e-10, "kolmogorov Re");
        assert_close(tau, (nu / eps).sqrt(), 1e-15, "tau");
        // spectrum slope is exactly -5/3 in log-log
        let k1 = 10.0_f64;
        let k2 = 100.0_f64;
        let slope = (kolmogorov_spectrum(k2, eps) / kolmogorov_spectrum(k1, eps)).ln()
            / (k2 / k1).ln();
        assert_close(slope, -5.0 / 3.0, 1e-12, "-5/3 slope");
        // scale relations
        let u_rms = 1.0;
        let lam = taylor_microscale(u_rms, nu, eps);
        let big_l = integral_scale(u_rms, eps);
        assert!(eta < lam && lam < big_l, "eta < lambda < L");
        assert!(re_lambda(u_rms, nu, eps) > 100.0);
    }

    #[test]
    fn test_energy_spectrum_parseval() {
        // single-mode field: all energy in one shell
        let n = 64;
        let dx = 2.0 * std::f64::consts::PI / n as f64;
        let u: Vec<f64> = (0..n).map(|i| (3.0 * i as f64 * dx).sin()).collect();
        let (k, e) = energy_spectrum_1d(&u, dx);
        // total energy = 0.5 <u^2> = 0.25
        let dk = k[1] - k[0];
        let total: f64 = e.iter().map(|&ee| ee * dk).sum();
        assert_close(total, 0.25, 1e-10, "parseval 1d");
        // energy concentrated at k = 3
        let imax = (0..e.len()).max_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap()).unwrap();
        assert_close(k[imax], 3.0, 1e-10, "peak wavenumber");

        // 2D single mode
        let mut u2 = vec![0.0; n * n];
        let v2 = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                u2[j * n + i] = (4.0 * j as f64 * dx).cos();
            }
        }
        let (k2, e2) = energy_spectrum_2d(&u2, &v2, n, dx);
        let dk2 = k2[1] - k2[0];
        let tot2: f64 = e2.iter().map(|&ee| ee * dk2).sum();
        assert_close(tot2, 0.25, 1e-10, "parseval 2d");

        // dissipation of the single mode: 2 nu k^2 E => 2*nu*9*0.25
        let nu = 0.01;
        let eps = dissipation_rate_from_spectrum(&k, &e, nu);
        assert_close(eps, 2.0 * nu * 9.0 * 0.25, 0.05 * 2.0 * nu * 9.0 * 0.25, "eps from spectrum");
    }

    #[test]
    fn test_energy_spectrum_3d_parseval() {
        use std::f64::consts::PI;
        let n = 32;
        let dx = 2.0 * PI / n as f64;
        // Single solenoidal mode: u = A cos(k0 z), v = w = 0 with k0 = 3
        // along z. It is divergence free (u depends only on z) and carries
        // kinetic energy 0.5 <u²> = A²/4 per unit mass.
        let amp = 0.8;
        let k0 = 3.0;
        let mut u = vec![0.0; n * n * n];
        let v = vec![0.0; n * n * n];
        let w = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    u[(k * n + j) * n + i] = amp * (k0 * k as f64 * dx).cos();
                }
            }
        }
        let (ks, es) = energy_spectrum_3d(&u, &v, &w, n, dx);
        assert_eq!(ks.len(), n / 2);
        assert_eq!(es.len(), n / 2);
        // The box is 2π long, so dk = 1 and the shells sit on integers.
        let dk = ks[1] - ks[0];
        assert_close(dk, 1.0, 1e-12, "shell spacing");
        // Parseval: Σ E(k) dk = 0.5 <|u|²> = A²/4.
        let total: f64 = es.iter().map(|e| e * dk).sum();
        assert_close(total, amp * amp / 4.0, 1e-10, "parseval 3d");
        // All of the energy sits in the k = 3 shell.
        let peak = (0..es.len())
            .max_by(|&a, &b| es[a].partial_cmp(&es[b]).unwrap())
            .unwrap();
        assert_close(ks[peak], k0, 1e-12, "peak wavenumber");
        for (m, e) in es.iter().enumerate() {
            if m != peak {
                assert!(*e < 1e-18, "leakage {e} into shell {}", ks[m]);
            }
        }
        // Spectra are non-negative by construction.
        assert!(es.iter().all(|&e| e >= 0.0));

        // Isotropy: the same mode oriented along x or y gives the same
        // spectrum, and three orthogonal modes superpose additively.
        let mut ux = vec![0.0; n * n * n];
        let mut vy = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let c = (k * n + j) * n + i;
                    // u depends on y, v depends on x: still divergence free.
                    ux[c] = amp * (k0 * j as f64 * dx).cos();
                    vy[c] = amp * (k0 * i as f64 * dx).cos();
                }
            }
        }
        let (_, es_x) = energy_spectrum_3d(&ux, &v, &w, n, dx);
        for (a, b) in es_x.iter().zip(&es) {
            assert_close(*a, *b, 1e-12, "isotropy of the shell average");
        }
        let (_, es_sum) = energy_spectrum_3d(&ux, &vy, &u, n, dx);
        let total_sum: f64 = es_sum.iter().map(|e| e * dk).sum();
        assert_close(total_sum, 3.0 * amp * amp / 4.0, 1e-9, "three modes superpose");

        // Two modes in different shells land in their own shells with the
        // right split of the energy.
        let (a1, k1) = (0.5_f64, 2.0_f64);
        let (a2, k2) = (0.3_f64, 7.0_f64);
        let mut two = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    two[(k * n + j) * n + i] = a1 * (k1 * k as f64 * dx).sin()
                        + a2 * (k2 * k as f64 * dx).sin();
                }
            }
        }
        let (ks2, es2) = energy_spectrum_3d(&two, &v, &w, n, dx);
        let shell = |target: f64| -> f64 {
            let idx = ks2.iter().position(|k| (k - target).abs() < 1e-9).unwrap();
            es2[idx] * dk
        };
        assert_close(shell(k1), a1 * a1 / 4.0, 1e-12, "k = 2 shell energy");
        assert_close(shell(k2), a2 * a2 / 4.0, 1e-12, "k = 7 shell energy");
        let total2: f64 = es2.iter().map(|e| e * dk).sum();
        assert_close(total2, (a1 * a1 + a2 * a2) / 4.0, 1e-10, "parseval, two modes");

        // Dissipation of a single mode: eps = 2 nu k² E integrated over k
        // reduces to 2 nu k0² × (A²/4).
        let nu = 0.02;
        let eps = dissipation_rate_from_spectrum(&ks, &es, nu);
        let eps_exact = 2.0 * nu * k0 * k0 * amp * amp / 4.0;
        assert_close(eps, eps_exact, 0.02 * eps_exact, "eps from the 3D spectrum");
    }

    #[test]
    fn test_rans_trait_objects_match_closed_form_solutions() {
        // k-omega SST through the trait: nu_t = a1 k / max(a1 omega, 0) =
        // k / omega when the shear limiter is inactive.
        let sst = KOmegaSst::new(2.0, 8.0, 1e-5);
        {
            let m: &dyn RansModel = &sst;
            assert_close(m.eddy_viscosity(), 2.0 / 8.0, 1e-15, "SST nu_t = k/omega");
            assert!(m.eddy_viscosity() >= 0.0);
        }

        // Homogeneous decay (s_mag = 0) with the trait's f1 = 1 inner
        // constants has the closed-form solution
        //   omega(t) = omega0 / (1 + beta omega0 t),
        //   k(t)     = k0 (1 + beta omega0 t)^{-beta*/beta}
        // with beta = 0.075 and beta* = 0.09.
        let (k0, om0) = (1.0_f64, 10.0_f64);
        let mut decay: Box<dyn RansModel> = Box::new(KOmegaSst::new(k0, om0, 1e-5));
        let dt = 1e-4_f64;
        let steps = 10_000; // t = 1
        let hist = rans_step(decay.as_mut(), 0.0, dt, steps);
        assert_eq!(hist.len(), steps);
        assert!(hist.iter().all(|v| v.is_finite() && *v >= 0.0), "nu_t must stay >= 0");
        // The working variables decay monotonically without production.
        assert!(
            hist.windows(2).all(|w| w[1] <= w[0] + 1e-15),
            "eddy viscosity is not monotone in free decay"
        );
        let t = dt * steps as f64;
        let (beta, beta_star) = (0.075_f64, 0.09_f64);
        let s = 1.0 + beta * om0 * t;
        let om_exact = om0 / s;
        let k_exact = k0 * s.powf(-beta_star / beta);
        let nu_exact = k_exact / om_exact;
        // Forward Euler at dt = 1e-4 (omega dt = 1e-3) is first order, so
        // ~0.1% is the expected agreement.
        assert_close(
            *hist.last().unwrap(),
            nu_exact,
            0.01 * nu_exact,
            "SST free decay vs the exact solution",
        );
        // Halving the step halves the error: the discrete solution really
        // is converging to that closed form.
        let err_at = |dt: f64| -> f64 {
            let steps = (t / dt).round() as usize;
            let mut m: Box<dyn RansModel> = Box::new(KOmegaSst::new(k0, om0, 1e-5));
            let h = rans_step(m.as_mut(), 0.0, dt, steps);
            (h.last().unwrap() - nu_exact).abs()
        };
        let ratio = err_at(2e-4) / err_at(1e-4);
        assert!((1.7..2.4).contains(&ratio), "SST time-order ratio {ratio}");

        // Spalart-Allmaras through the trait: `advance` reads s_mag as the
        // vorticity magnitude and places the wall infinitely far away, so
        // the destruction term vanishes and the model reduces to pure
        // exponential production d(nu~)/dt = c_b1 Omega nu~.
        let sa = SpalartAllmaras::new(1e-3, 1e-5);
        {
            let m: &dyn RansModel = &sa;
            // chi = 100 >> c_v1 = 7.1, so f_v1 -> 1 and nu_t -> nu~.
            let chi: f64 = 1e-3 / 1e-5;
            let fv1 = chi.powi(3) / (chi.powi(3) + 7.1_f64.powi(3));
            assert_close(m.eddy_viscosity(), 1e-3 * fv1, 1e-18, "SA nu_t");
            assert!(m.eddy_viscosity() > 0.0);
            assert!(m.eddy_viscosity() < 1e-3, "f_v1 must damp nu~");
        }
        let nu_tilde0 = 1e-3;
        let omega_mag = 4.0;
        let dt = 1e-3;
        let steps = 300;
        let mut sa_model: Box<dyn RansModel> = Box::new(SpalartAllmaras::new(nu_tilde0, 1e-5));
        let hist = rans_step(sa_model.as_mut(), omega_mag, dt, steps);
        assert_eq!(hist.len(), steps);
        assert!(hist.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(
            hist.windows(2).all(|w| w[1] >= w[0]),
            "SA nu_t must grow under vorticity"
        );
        let cb1 = 0.1355_f64;
        // Discrete forward-Euler solution: nu~_N = nu~_0 (1 + c_b1 Omega dt)^N.
        let nu_tilde_n = nu_tilde0 * (1.0 + cb1 * omega_mag * dt).powi(steps as i32);
        let chi_n = nu_tilde_n / 1e-5;
        let fv1_n = chi_n.powi(3) / (chi_n.powi(3) + 7.1_f64.powi(3));
        let nu_t_n = nu_tilde_n * fv1_n;
        assert_close(
            *hist.last().unwrap(),
            nu_t_n,
            1e-6 * nu_t_n,
            "SA growth vs the exact Euler solution",
        );
        // Which in turn approximates exp(c_b1 Omega t) to O(dt).
        let nu_tilde_exp = nu_tilde0 * (cb1 * omega_mag * dt * steps as f64).exp();
        assert_close(nu_tilde_n, nu_tilde_exp, 0.01 * nu_tilde_exp, "SA exponential growth");

        // Zero vorticity leaves the model at rest far from any wall.
        let mut quiet: Box<dyn RansModel> = Box::new(SpalartAllmaras::new(nu_tilde0, 1e-5));
        let flat = rans_step(quiet.as_mut(), 0.0, dt, 200);
        assert_close(
            *flat.last().unwrap(),
            quiet.eddy_viscosity(),
            1e-18,
            "SA at rest",
        );
        assert!(
            (flat.last().unwrap() / (nu_tilde0 * {
                let chi: f64 = nu_tilde0 / 1e-5;
                chi.powi(3) / (chi.powi(3) + 7.1_f64.powi(3))
            }) - 1.0)
                .abs()
                < 1e-9,
            "SA drifted without production"
        );

        // Both models satisfy the trait contract of a non-negative eddy
        // viscosity when driven through a common `dyn` loop.
        let mut zoo: Vec<Box<dyn RansModel>> = vec![
            Box::new(KOmegaSst::new(0.5, 20.0, 1e-5)),
            Box::new(SpalartAllmaras::new(5e-4, 1e-5)),
            Box::new(KEpsilon::new(0.5, 0.5, 1e-5, KEpsilonVariant::Standard)),
        ];
        for m in zoo.iter_mut() {
            let h = rans_step(m.as_mut(), 1.5, 1e-3, 100);
            assert!(h.iter().all(|&v| v.is_finite() && v >= 0.0), "negative nu_t: {h:?}");
        }
    }

    #[test]
    fn test_structure_and_correlation() {
        let n = 256;
        let u: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();
        // R(0) = 1, R falls off, R(n/2) = -1 for a single mode
        assert_close(two_point_correlation(&u, 0), 1.0, 1e-12, "R(0)");
        assert_close(two_point_correlation(&u, n / 2), -1.0, 1e-9, "R(T/2)");
        // second-order structure function relates to correlation:
        // S2(r) = 2 u'^2 (1 - R(r))
        let var = 0.5;
        let s2 = structure_function(&u, n / 4, 2);
        let r = two_point_correlation(&u, n / 4);
        assert_close(s2, 2.0 * var * (1.0 - r), 1e-9, "S2 vs R");
    }

    #[test]
    fn test_les_models() {
        // pure shear du/dy = s
        let s = 2.0;
        let g = Mat3 {
            data: [[0.0, s, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };
        let delta = 0.1;
        let nu_smag = smagorinsky_nu_t(&g, delta, 0.17);
        // |S| = s for pure shear
        assert_close(nu_smag, (0.17 * delta) * (0.17 * delta) * s, 1e-12, "smagorinsky shear");
        // solid-body rotation has zero strain: nu_t = 0
        let grot = Mat3 {
            data: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };
        assert_close(smagorinsky_nu_t(&grot, delta, 0.17), 0.0, 1e-12, "smag rotation");
        // WALE vanishes for pure shear (its key property) and is finite for
        // a strained-rotational field
        assert_close(wale_nu_t(&g, delta, 0.5), 0.0, 1e-12, "wale shear");
        let gmix = Mat3 {
            data: [[1.0, 0.5, 0.0], [-0.3, -1.0, 0.2], [0.1, 0.0, 0.0]],
        };
        assert!(wale_nu_t(&gmix, delta, 0.5) > 0.0);
        // Vreman is constructed to vanish for pure shear (its key property)
        // and to be positive for genuinely 3D gradients
        assert_close(vreman_nu_t(&g, delta, 0.07), 0.0, 1e-12, "vreman shear");
        assert!(vreman_nu_t(&gmix, delta, 0.07) > 0.0);
        let gzero = Mat3 {
            data: [[0.0; 3]; 3],
        };
        assert_close(vreman_nu_t(&gzero, delta, 0.07), 0.0, 1e-12, "vreman zero");
        // dynamic Cs recovers a known ratio
        let l = Mat3 {
            data: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        };
        let m = Mat3 {
            data: [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        };
        assert_close(dynamic_smagorinsky_cs(&l, &m), 0.5, 1e-12, "dynamic cs");
    }

    #[test]
    fn test_vortex_criteria() {
        // solid-body rotation: Q > 0, lambda2 < 0, delta > 0
        let grot = Mat3 {
            data: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };
        assert!(q_criterion(&grot) > 0.0, "Q rotation");
        assert!(lambda2_criterion(&grot) < 0.0, "lambda2 rotation");
        assert!(delta_criterion(&grot) > 0.0, "delta rotation");
        // pure strain: Q < 0, lambda2 > 0 (middle eig of S^2 is >= 0)
        let gstr = Mat3 {
            data: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        };
        assert!(q_criterion(&gstr) < 0.0, "Q strain");
        assert!(lambda2_criterion(&gstr) >= 0.0, "lambda2 strain");
        assert!(delta_criterion(&gstr) <= 0.0, "delta strain");

        // grid identification: a Taylor-Green vortex array flags cores
        let n = 32;
        let dx = 2.0 * std::f64::consts::PI / n as f64;
        let mut u = vec![0.0; n * n];
        let mut v = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (i as f64 * dx, j as f64 * dx);
                u[j * n + i] = x.sin() * y.cos();
                v[j * n + i] = -x.cos() * y.sin();
            }
        }
        let flags = vortex_identify_q(&u, &v, n, dx, 0.1);
        let count = flags.iter().filter(|&&f| f).count();
        assert!(count > 0, "some vortical cells");
        assert!(count < n * n, "not everything vortical");
        // the cell at a vortex center (x = y = pi/2 -> i = j = n/4) is flagged
        assert!(flags[(n / 4) * n + n / 4], "core flagged");
    }

    #[test]
    fn test_rans_models() {
        // k-epsilon homogeneous shear reaches production/dissipation balance
        // behavior: k grows for strong shear, decays for zero shear
        let mut ke = KEpsilon::new(1.0, 1.0, 1e-5, KEpsilonVariant::Standard);
        let k0 = ke.k;
        for _ in 0..200 {
            ke.step(0.0, 0.005);
        }
        assert!(ke.k < k0, "k decays without production");
        // decaying isotropic k-eps has the exact solution
        // k(t) = k0 (1 + t/tau)^{-n}, n = 1/(C2-1), tau = k0/((C2-1) eps0)
        let mut ke2 = KEpsilon::new(1.0, 1.0, 1e-5, KEpsilonVariant::Standard);
        let dt = 1e-4_f64;
        let t_end = 5.0_f64;
        let steps = (t_end / dt).round() as usize;
        for _ in 0..steps {
            ke2.step(0.0, dt);
        }
        let n_dec = 1.0 / (1.92 - 1.0);
        let tau = 1.0 / (1.92 - 1.0);
        let k_exact = (1.0 + t_end / tau).powf(-n_dec);
        assert_close(ke2.k, k_exact, 0.01 * k_exact, "k-eps decay solution");

        // strong shear: k grows
        let mut ke3 = KEpsilon::new(0.01, 0.01, 1e-5, KEpsilonVariant::Standard);
        for _ in 0..100 {
            ke3.step(5.0, 0.01);
        }
        assert!(ke3.k > 0.01, "k grows under shear");
        assert!(ke3.nu_t() > 0.0);

        // wall function sanity
        let (kw, ew) = ke.wall_function(0.05, 0.01);
        assert_close(kw, 0.05 * 0.05 / 0.3, 1e-10, "wall k");
        assert!(ew > 0.0);

        // init from intensity
        let ki = KEpsilon::init_from_intensity(0.05, 10.0, 0.1, 1e-5);
        assert_close(ki.k, 1.5 * 0.25, 1e-10, "k from TI");

        // SST decays without shear and stays positive
        let mut sst = KOmegaSst::new(1.0, 10.0, 1e-5);
        for _ in 0..500 {
            sst.step(0.0, 1.0, 0.001);
        }
        assert!(sst.k < 1.0 && sst.k > 0.0, "sst k decays");

        // Spalart-Allmaras grows under vorticity far from wall, and nu_t
        // approaches nu_tilde for large chi
        let mut sa = SpalartAllmaras::new(1e-3, 1e-5);
        let nt0 = sa.nu_t();
        for _ in 0..100 {
            sa.step(10.0, 1e6, 0.01);
        }
        assert!(sa.nu_t() > nt0, "SA production");

        // trait object dispatch
        let mut m: Box<dyn RansModel> = Box::new(KEpsilon::new(1.0, 1.0, 1e-5, KEpsilonVariant::Rng));
        let hist = rans_step(m.as_mut(), 2.0, 0.001, 50);
        assert_eq!(hist.len(), 50);
        assert!(hist.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn test_synthetic_turbulence() {
        // Kraichnan field: divergence-free to discretization accuracy,
        // correct rms
        let n = 48;
        let l = 2.0 * std::f64::consts::PI;
        let (u, v) = synthetic_turbulence_kraichnan(n, l, 40, 4.0, 0.7, 42);
        let ms = u.iter().zip(&v).map(|(&a, &b)| a * a + b * b).sum::<f64>() / (n * n) as f64;
        assert_close((ms / 2.0).sqrt(), 0.7, 1e-9, "kraichnan rms");
        // divergence check: modes are exact transverse lattice waves, but a
        // central difference of sin/cos leaves an O((k dx)^2) residual, so
        // compare discrete divergence to the gradient magnitude loosely
        let dx = l / n as f64;
        let idx = |i: usize, j: usize| j * n + i;
        let mut div_rms = 0.0;
        let mut grad_rms = 0.0;
        for j in 0..n {
            let (jp, jm) = ((j + 1) % n, (j + n - 1) % n);
            for i in 0..n {
                let (ip, im) = ((i + 1) % n, (i + n - 1) % n);
                let dudx = (u[idx(ip, j)] - u[idx(im, j)]) / (2.0 * dx);
                let dvdy = (v[idx(i, jp)] - v[idx(i, jm)]) / (2.0 * dx);
                div_rms += (dudx + dvdy).powi(2);
                grad_rms += dudx * dudx + dvdy * dvdy;
            }
        }
        assert!(
            div_rms.sqrt() < 0.1 * grad_rms.sqrt().max(1e-30),
            "divergence-free: {} vs {}",
            div_rms.sqrt(),
            grad_rms.sqrt()
        );

        // SEM: zero mean, target rms
        let sem = synthetic_eddy_method(256, 1.0, 60, 0.05, 0.3, 7);
        let mean = sem.iter().sum::<f64>() / 256.0;
        let var = sem.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 256.0;
        assert_close(mean, 0.0, 1e-10, "sem mean");
        assert_close(var.sqrt(), 0.3, 1e-9, "sem rms");
    }

    #[test]
    fn test_model_spectra_and_fits() {
        // Pao spectrum matches Kolmogorov in the inertial range and falls
        // below it near the cutoff
        let nu = 1e-5;
        let eps = 0.1;
        let (eta, _, _) = kolmogorov_scales(nu, eps);
        let k_in = 0.001 / eta;
        assert_close(
            pao_spectrum(k_in, eps, nu) / kolmogorov_spectrum(k_in, eps),
            1.0,
            0.02,
            "pao inertial",
        );
        assert!(pao_spectrum(1.0 / eta, eps, nu) < 0.5 * kolmogorov_spectrum(1.0 / eta, eps));

        // von Karman: integrates to roughly the requested energy, peaks near
        // ke (large keta so the dissipation cutoff is negligible; the k^-5/3
        // tail beyond the integration limit carries about 1.5% of the energy)
        let ke = 5.0;
        let keta = 1e7;
        let mut total = 0.0;
        let dk = 0.02;
        let mut kk = dk;
        let mut peak_k = 0.0;
        let mut peak_e = 0.0;
        while kk < 5000.0 {
            let e = von_karman_spectrum(kk, 2.0, ke, keta);
            total += e * dk;
            if e > peak_e {
                peak_e = e;
                peak_k = kk;
            }
            kk += dk;
        }
        assert_close(total, 2.0, 0.06, "von karman energy");
        // peak of x^4/(1+x^2)^{17/6} is at x = sqrt(12/5)
        assert_close(peak_k, ke * (12.0_f64 / 5.0).sqrt(), 0.1, "von karman peak");

        // inertial-range exponent recovered from an exact -5/3 spectrum
        let ks: Vec<f64> = (1..200).map(|i| i as f64).collect();
        let es: Vec<f64> = ks.iter().map(|&k| kolmogorov_spectrum(k, eps)).collect();
        assert_close(inertial_range_exponent(&ks, &es, 5.0, 100.0), -5.0 / 3.0, 1e-10, "fit slope");

        // log-law fit: generate a synthetic log profile and recover u_tau, B
        let u_tau = 0.05;
        let nu2 = 1e-6;
        let ys: Vec<f64> = (1..50).map(|i| 30.0 * nu2 / u_tau * (1.1_f64).powi(i)).collect();
        let us: Vec<f64> = ys
            .iter()
            .map(|&y| u_tau * ((y * u_tau / nu2).ln() / KAPPA_VK + 5.0))
            .collect();
        let (ut_fit, b_fit) = log_law_fit(&ys, &us, nu2);
        assert_close(ut_fit, u_tau, 1e-9, "u_tau fit");
        assert_close(b_fit, 5.0, 1e-7, "B fit");

        // Reichardt reference: near-wall u+ ~ y+, log region slope ~ 1/kappa
        let (yp, up) = channel_flow_dns_reference(550.0, 550);
        assert_close(up[0] / yp[0], 1.0, 0.05, "viscous sublayer");
        // slope in log region between y+ = 100 and 400
        let i1 = 99;
        let i2 = 399;
        let slope = (up[i2] - up[i1]) / (yp[i2] / yp[i1]).ln();
        assert_close(slope, 1.0 / KAPPA_VK, 0.1, "log slope");

        // misc
        assert_close(richardson_cascade_time(2.0, 4.0), 0.5, 1e-15, "turnover");
        assert_close(turbulent_diffusivity(0.09, 0.9), 0.1, 1e-12, "Pr_t");
        let ti = turbulence_intensity(&[9.0, 10.0, 11.0, 10.0]);
        assert!(ti > 0.0 && ti < 0.2);
        let rs = reynolds_stress(&[1.0, -1.0, 1.0, -1.0], &[1.0, -1.0, 1.0, -1.0]);
        assert_close(rs, 1.0, 1e-12, "reynolds stress");
    }

    #[test]
    fn test_decaying_isotropic_turbulence() {
        let n = 8;
        let f0 = decaying_isotropic_turbulence(n, 0.05, 1e-9);
        let f1 = decaying_isotropic_turbulence(n, 0.05, 1.0);
        let energy = |f: &crate::cfd::stable_fluids::StableFluid3| {
            f.grid
                .u
                .iter()
                .chain(f.grid.v.iter())
                .chain(f.grid.w.iter())
                .map(|&x| x * x)
                .sum::<f64>()
        };
        let e0 = energy(&f0);
        let e1 = energy(&f1);
        assert!(e0 > 0.0, "seeded with energy");
        assert!(e1 < e0, "kinetic energy decays: {e1} vs {e0}");
        assert!(f1.grid.u.iter().all(|v| v.is_finite()));
    }
}
