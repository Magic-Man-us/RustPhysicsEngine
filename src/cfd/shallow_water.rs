//! Shallow water equations: well-balanced HLL finite volumes with
//! hydrostatic reconstruction and wet/dry handling (1D and 2D), the
//! Stoker dam-break solution, water-wave dispersion relations, ocean
//! spectra, and Gerstner waves.

use crate::math::{Vec2, Vec3};
use crate::monte_carlo::Rng;

const PI: f64 = crate::math::constants::PI;
const TWO_PI: f64 = 2.0 * PI;

/// HLL flux for the 1D shallow water system (h, hu) with gravity g.
fn swe_hll_flux(hl: f64, hul: f64, hr: f64, hur: f64, g: f64, dry: f64) -> (f64, f64) {
    let ul = if hl > dry { hul / hl } else { 0.0 };
    let ur = if hr > dry { hur / hr } else { 0.0 };
    let cl = (g * hl.max(0.0)).sqrt();
    let cr = (g * hr.max(0.0)).sqrt();
    let sl = (ul - cl).min(ur - cr).min(0.0);
    let sr = (ul + cl).max(ur + cr).max(0.0);
    let fl = (hul, hul * ul + 0.5 * g * hl * hl);
    let fr = (hur, hur * ur + 0.5 * g * hr * hr);
    if sl >= 0.0 {
        fl
    } else if sr <= 0.0 {
        fr
    } else {
        let inv = 1.0 / (sr - sl);
        (
            (sr * fl.0 - sl * fr.0 + sl * sr * (hr - hl)) * inv,
            (sr * fl.1 - sl * fr.1 + sl * sr * (hur - hul)) * inv,
        )
    }
}

/// 2D shallow water solver on a square grid: HLL fluxes with Audusse
/// hydrostatic reconstruction (well-balanced over bathymetry), Manning
/// friction, Coriolis, and wet/dry tolerance.
pub struct ShallowWater2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
    pub bathymetry: Vec<f64>,
    pub g: f64,
    pub manning_n: f64,
    pub coriolis: f64,
    pub dry_tol: f64,
    pub time: f64,
}

impl ShallowWater2D {
    /// New dry basin with flat bathymetry.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64, g: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            h: vec![0.0; nx * ny],
            hu: vec![0.0; nx * ny],
            hv: vec![0.0; nx * ny],
            bathymetry: vec![0.0; nx * ny],
            g,
            manning_n: 0.0,
            coriolis: 0.0,
            dry_tol: 1e-6,
            time: 0.0,
        }
    }

    /// Set the bed elevation from a function of world (x, y).
    pub fn set_bathymetry(&mut self, f: impl Fn(f64, f64) -> f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                self.bathymetry[j * self.nx + i] =
                    f((i as f64 + 0.5) * self.dx, (j as f64 + 0.5) * self.dx);
            }
        }
    }

    /// Dam break: depth `h_l` left of world x = `x_split`, `h_r` right.
    pub fn set_dam_break(&mut self, x_split: f64, h_l: f64, h_r: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = (i as f64 + 0.5) * self.dx;
                let c = j * self.nx + i;
                self.h[c] = if x < x_split { h_l } else { h_r };
                self.hu[c] = 0.0;
                self.hv[c] = 0.0;
            }
        }
        self.time = 0.0;
    }

    /// Gaussian free-surface bump of amplitude `amp` on still depth
    /// `base_depth`.
    pub fn set_gaussian_bump(&mut self, cx: f64, cy: f64, amp: f64, sigma: f64, base_depth: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = (i as f64 + 0.5) * self.dx - cx;
                let y = (j as f64 + 0.5) * self.dx - cy;
                let c = j * self.nx + i;
                let eta = amp * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
                self.h[c] = (base_depth + eta - self.bathymetry[c]).max(0.0);
                self.hu[c] = 0.0;
                self.hv[c] = 0.0;
            }
        }
    }

    /// Add water volume at cell (i, j) at the given rate × 1 step.
    pub fn add_source(&mut self, i: usize, j: usize, rate: f64) {
        self.h[j * self.nx + i] += rate;
    }

    /// One step at the CFL number; returns the dt used. Reflective walls.
    pub fn step(&mut self, cfl: f64) -> f64 {
        let (nx, ny, dx, g) = (self.nx, self.ny, self.dx, self.g);
        let dry = self.dry_tol;
        let mut smax = (g * 1e-6_f64).sqrt();
        for c in 0..nx * ny {
            if self.h[c] > dry {
                let u = self.hu[c] / self.h[c];
                let v = self.hv[c] / self.h[c];
                let cw = (g * self.h[c]).sqrt();
                smax = smax.max(u.abs().max(v.abs()) + cw);
            }
        }
        let dt = cfl * dx / smax;
        let lam = dt / dx;
        let mut dh = vec![0.0; nx * ny];
        let mut dhu = vec![0.0; nx * ny];
        let mut dhv = vec![0.0; nx * ny];
        // x faces with hydrostatic reconstruction.
        let get = |i: i64, j: i64| -> (f64, f64, f64, f64) {
            // Reflective ghost: mirror interior with flipped normal
            // momentum handled at the face by the caller.
            let ii = i.clamp(0, nx as i64 - 1) as usize;
            let jj = j.clamp(0, ny as i64 - 1) as usize;
            let c = jj * nx + ii;
            (self.h[c], self.hu[c], self.hv[c], self.bathymetry[c])
        };
        for j in 0..ny as i64 {
            for fi in 0..=(nx as i64) {
                let (hl, mut hul, hvl, bl) = get(fi - 1, j);
                let (hr, mut hur, hvr, br) = get(fi, j);
                let wall = fi == 0 || fi == nx as i64;
                if wall {
                    // Mirror the normal momentum.
                    if fi == 0 {
                        hul = -hur;
                    } else {
                        hur = -hul;
                    }
                }
                // Hydrostatic reconstruction (Audusse et al.).
                let bmax = bl.max(br);
                let hl_s = (hl + bl - bmax).max(0.0);
                let hr_s = (hr + br - bmax).max(0.0);
                let ul = if hl > dry { hul / hl } else { 0.0 };
                let ur = if hr > dry { hur / hr } else { 0.0 };
                let vl = if hl > dry { hvl / hl } else { 0.0 };
                let vr = if hr > dry { hvr / hr } else { 0.0 };
                let (fh, fhu_s) = swe_hll_flux(hl_s, hl_s * ul, hr_s, hr_s * ur, g, dry);
                // Transverse momentum upwinded with the mass flux.
                let fhv = fh * if fh >= 0.0 { vl } else { vr };
                // Source correction restoring well-balancedness.
                let src_l = 0.5 * g * (hl * hl - hl_s * hl_s);
                let src_r = 0.5 * g * (hr * hr - hr_s * hr_s);
                if fi > 0 {
                    let c = (j as usize) * nx + (fi - 1) as usize;
                    dh[c] -= lam * fh;
                    dhu[c] -= lam * (fhu_s + src_l);
                    dhv[c] -= lam * fhv;
                }
                if fi < nx as i64 {
                    let c = (j as usize) * nx + fi as usize;
                    dh[c] += lam * fh;
                    dhu[c] += lam * (fhu_s + src_r);
                    dhv[c] += lam * fhv;
                }
            }
        }
        // y faces (roles of hu/hv swapped).
        for i in 0..nx as i64 {
            for fj in 0..=(ny as i64) {
                let (hl, hul, mut hvl, bl) = get(i, fj - 1);
                let (hr, hur, mut hvr, br) = get(i, fj);
                let wall = fj == 0 || fj == ny as i64;
                if wall {
                    if fj == 0 {
                        hvl = -hvr;
                    } else {
                        hvr = -hvl;
                    }
                }
                let bmax = bl.max(br);
                let hl_s = (hl + bl - bmax).max(0.0);
                let hr_s = (hr + br - bmax).max(0.0);
                let vl = if hl > dry { hvl / hl } else { 0.0 };
                let vr = if hr > dry { hvr / hr } else { 0.0 };
                let ul = if hl > dry { hul / hl } else { 0.0 };
                let ur = if hr > dry { hur / hr } else { 0.0 };
                let (fh, fhv_s) = swe_hll_flux(hl_s, hl_s * vl, hr_s, hr_s * vr, g, dry);
                let fhu = fh * if fh >= 0.0 { ul } else { ur };
                let src_l = 0.5 * g * (hl * hl - hl_s * hl_s);
                let src_r = 0.5 * g * (hr * hr - hr_s * hr_s);
                if fj > 0 {
                    let c = ((fj - 1) as usize) * nx + i as usize;
                    dh[c] -= lam * fh;
                    dhv[c] -= lam * (fhv_s + src_l);
                    dhu[c] -= lam * fhu;
                }
                if fj < ny as i64 {
                    let c = (fj as usize) * nx + i as usize;
                    dh[c] += lam * fh;
                    dhv[c] += lam * (fhv_s + src_r);
                    dhu[c] += lam * fhu;
                }
            }
        }
        for c in 0..nx * ny {
            self.h[c] = (self.h[c] + dh[c]).max(0.0);
            self.hu[c] += dhu[c];
            self.hv[c] += dhv[c];
            if self.h[c] <= dry {
                self.hu[c] = 0.0;
                self.hv[c] = 0.0;
            }
        }
        // Manning friction (semi-implicit) and Coriolis rotation.
        if self.manning_n > 0.0 || self.coriolis != 0.0 {
            for c in 0..nx * ny {
                if self.h[c] <= dry {
                    continue;
                }
                if self.manning_n > 0.0 {
                    let u = self.hu[c] / self.h[c];
                    let v = self.hv[c] / self.h[c];
                    let speed = (u * u + v * v).sqrt();
                    let cf = g * self.manning_n * self.manning_n * speed
                        / self.h[c].powf(4.0 / 3.0);
                    let fac = 1.0 / (1.0 + dt * cf);
                    self.hu[c] *= fac;
                    self.hv[c] *= fac;
                }
                if self.coriolis != 0.0 {
                    let th = self.coriolis * dt;
                    let (c_t, s_t) = (th.cos(), th.sin());
                    let (mu, mv) = (self.hu[c], self.hv[c]);
                    self.hu[c] = c_t * mu + s_t * mv;
                    self.hv[c] = -s_t * mu + c_t * mv;
                }
            }
        }
        self.time += dt;
        dt
    }

    /// Step until time `t`.
    pub fn run_until(&mut self, t: f64) {
        while self.time < t - 1e-12 {
            self.step(0.45);
        }
    }

    /// Number of wet cells.
    #[must_use]
    pub fn wet_cells(&self) -> usize {
        self.h.iter().filter(|&&h| h > self.dry_tol).count()
    }

    /// Total water volume.
    #[must_use]
    pub fn total_volume(&self) -> f64 {
        self.h.iter().sum::<f64>() * self.dx * self.dx
    }

    /// Total mechanical energy (kinetic + potential).
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        let mut e = 0.0;
        for c in 0..self.nx * self.ny {
            let h = self.h[c];
            if h > self.dry_tol {
                let u = self.hu[c] / h;
                let v = self.hv[c] / h;
                e += 0.5 * h * (u * u + v * v)
                    + 0.5 * self.g * h * h
                    + self.g * h * self.bathymetry[c];
            }
        }
        e * self.dx * self.dx
    }

    /// Froude number per cell.
    #[must_use]
    pub fn froude_field(&self) -> Vec<f64> {
        (0..self.nx * self.ny)
            .map(|c| {
                let h = self.h[c];
                if h > self.dry_tol {
                    let u = self.hu[c] / h;
                    let v = self.hv[c] / h;
                    (u * u + v * v).sqrt() / (self.g * h).sqrt()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Deepest water column.
    #[must_use]
    pub fn max_depth(&self) -> f64 {
        self.h.iter().cloned().fold(0.0, f64::max)
    }
}

/// Stoker's exact wet-bed dam-break solution: (h, u) at position x
/// (dam at x = 0) and time t.
#[must_use]
pub fn swe_1d_exact_dam_break(x: f64, t: f64, h_l: f64, h_r: f64, g: f64) -> (f64, f64) {
    if t <= 0.0 {
        return if x < 0.0 { (h_l, 0.0) } else { (h_r, 0.0) };
    }
    let cl = (g * h_l).sqrt();
    let cr = (g * h_r).sqrt();
    // Middle depth from matching the rarefaction to the shock.
    let f = |hm: f64| -> f64 {
        let cm = (g * hm).sqrt();
        let u_rare = 2.0 * (cl - cm);
        let u_shock = (hm - h_r) * ((g * (hm + h_r)) / (2.0 * hm * h_r)).sqrt();
        u_rare - u_shock
    };
    let (mut lo, mut hi) = (h_r, h_l);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let hm = 0.5 * (lo + hi);
    let cm = (g * hm).sqrt();
    let um = 2.0 * (cl - cm);
    // Shock speed from mass conservation.
    let s = if (hm - h_r).abs() > 1e-12 { hm * um / (hm - h_r) } else { um + cr };
    let xi = x / t;
    if xi <= -cl {
        (h_l, 0.0)
    } else if xi <= um - cm {
        // Rarefaction fan.
        let c = (2.0 * cl - xi) / 3.0;
        let u = 2.0 * (xi + cl) / 3.0;
        (c * c / g, u)
    } else if xi <= s {
        (hm, um)
    } else {
        (h_r, 0.0)
    }
}

/// 1D well-balanced SWE step on columns with bathymetry (helper used by
/// the tsunami run-up model and `ShallowWater1D::step_hll`).
#[allow(clippy::too_many_arguments)] // solver state slices
pub fn swe_1d_step_hll(
    h: &mut [f64],
    hu: &mut [f64],
    b: &[f64],
    dx: f64,
    g: f64,
    dt: f64,
    dry: f64,
    reflective: bool,
) {
    let n = h.len();
    let lam = dt / dx;
    let get = |arr: &[f64], i: i64| arr[i.clamp(0, n as i64 - 1) as usize];
    let mut dh = vec![0.0; n];
    let mut dhu = vec![0.0; n];
    for f in 0..=(n as i64) {
        let (hl, mut hul, bl) = (get(h, f - 1), get(hu, f - 1), get(b, f - 1));
        let (hr, mut hur, br) = (get(h, f), get(hu, f), get(b, f));
        let wall = f == 0 || f == n as i64;
        if wall && reflective {
            if f == 0 {
                hul = -hur;
            } else {
                hur = -hul;
            }
        }
        let bmax = bl.max(br);
        let hl_s = (hl + bl - bmax).max(0.0);
        let hr_s = (hr + br - bmax).max(0.0);
        let ul = if hl > dry { hul / hl } else { 0.0 };
        let ur = if hr > dry { hur / hr } else { 0.0 };
        let (fh, fhu_s) = swe_hll_flux(hl_s, hl_s * ul, hr_s, hr_s * ur, g, dry);
        let src_l = 0.5 * g * (hl * hl - hl_s * hl_s);
        let src_r = 0.5 * g * (hr * hr - hr_s * hr_s);
        if f > 0 {
            dh[(f - 1) as usize] -= lam * fh;
            dhu[(f - 1) as usize] -= lam * (fhu_s + src_l);
        }
        if f < n as i64 {
            dh[f as usize] += lam * fh;
            dhu[f as usize] += lam * (fhu_s + src_r);
        }
    }
    for i in 0..n {
        h[i] = (h[i] + dh[i]).max(0.0);
        hu[i] += dhu[i];
        if h[i] <= dry {
            hu[i] = 0.0;
        }
    }
}

/// Tsunami run-up on a plane beach of slope `slope`: an offshore
/// Gaussian wave of amplitude `a` (width `sigma`, centered at world
/// x = x0) propagates onto the beach. Returns the shoreline position
/// sampled at each of `n_samples` uniform times up to `t_end`.
#[allow(clippy::too_many_arguments)] // scenario parameters
#[must_use]
pub fn tsunami_runup_1d(
    slope: f64,
    a: f64,
    sigma: f64,
    x0: f64,
    n: usize,
    domain: f64,
    t_end: f64,
    n_samples: usize,
) -> Vec<f64> {
    let dx = domain / n as f64;
    let g = 9.81;
    // Bathymetry rises to the right: b = slope·(x − x_beach), sea level 0.
    let x_beach = 0.7 * domain;
    let b: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx;
            slope * (x - x_beach)
        })
        .collect();
    let mut h: Vec<f64> = b
        .iter()
        .enumerate()
        .map(|(i, &bi)| {
            let x = (i as f64 + 0.5) * dx;
            let eta = a * (-((x - x0) / sigma).powi(2) / 2.0).exp();
            (eta - bi).max(0.0)
        })
        .collect();
    let mut hu = vec![0.0; n];
    let mut out = Vec::with_capacity(n_samples);
    let mut t = 0.0;
    let mut next_sample = 0;
    let dry = 1e-6;
    while next_sample < n_samples {
        let target = (next_sample as f64 + 1.0) * t_end / n_samples as f64;
        while t < target {
            // CFL time step.
            let mut smax: f64 = 1e-3;
            for i in 0..n {
                if h[i] > dry {
                    smax = smax.max((hu[i] / h[i]).abs() + (g * h[i]).sqrt());
                }
            }
            let dt = (0.4 * dx / smax).min(target - t + 1e-12);
            swe_1d_step_hll(&mut h, &mut hu, &b, dx, g, dt, dry, true);
            t += dt;
        }
        // Shoreline: last wet cell.
        let shoreline = (0..n)
            .rev()
            .find(|&i| h[i] > 10.0 * dry)
            .map(|i| (i as f64 + 0.5) * dx)
            .unwrap_or(0.0);
        out.push(shoreline);
        next_sample += 1;
    }
    out
}

/// Shallow-water wave speed √(gh).
#[must_use]
pub fn wave_speed_shallow(h: f64, g: f64) -> f64 {
    (g * h).sqrt()
}

/// Shallow-water dispersion ω = k√(gh).
#[must_use]
pub fn dispersion_shallow(k: f64, h: f64, g: f64) -> f64 {
    k * (g * h).sqrt()
}

/// Deep-water dispersion ω = √(gk).
#[must_use]
pub fn dispersion_deep(k: f64, g: f64) -> f64 {
    (g * k).sqrt()
}

/// Full linear dispersion ω = √(gk tanh(kh)).
#[must_use]
pub fn dispersion_full(k: f64, h: f64, g: f64) -> f64 {
    (g * k * (k * h).tanh()).sqrt()
}

/// Stokes drift at the surface: U = a²ωk cosh(2kh)/(2 sinh²(kh)).
#[must_use]
pub fn stokes_drift(a: f64, k: f64, h: f64, g: f64) -> f64 {
    let omega = dispersion_full(k, h, g);
    let kh = k * h;
    if kh > 20.0 {
        a * a * omega * k
    } else {
        a * a * omega * k * (2.0 * kh).cosh() / (2.0 * kh.sinh().powi(2))
    }
}

/// Miche/steepness breaking criterion: waves break when H/λ exceeds
/// 1/7.
#[must_use]
pub fn wave_breaking_criterion(h_over_lambda: f64) -> bool {
    h_over_lambda > 1.0 / 7.0
}

/// JONSWAP spectrum S(f) (m²/Hz) for significant wave height `hs`, peak
/// period `tp`, and peak-enhancement `gamma` (≈3.3), normalized so that
/// ∫S df = hs²/16.
#[must_use]
pub fn jonswap_spectrum(f: f64, hs: f64, tp: f64, gamma: f64) -> f64 {
    let fp = 1.0 / tp;
    let shape = |f: f64| -> f64 {
        if f <= 0.0 {
            return 0.0;
        }
        let sigma = if f <= fp { 0.07 } else { 0.09 };
        let r = (-((f - fp).powi(2)) / (2.0 * sigma * sigma * fp * fp)).exp();
        f.powi(-5) * (-1.25 * (fp / f).powi(4)).exp() * gamma.powf(r)
    };
    // Normalize numerically (coarse quadrature over the support).
    let n = 400;
    let f_hi = 6.0 * fp;
    let df = f_hi / n as f64;
    let integral: f64 = (1..=n).map(|k| shape(k as f64 * df)).sum::<f64>() * df;
    let m0_target = hs * hs / 16.0;
    shape(f) * m0_target / integral.max(1e-300)
}

/// Pierson-Moskowitz spectrum for wind speed `u10` (m/s) at 10 m.
#[must_use]
pub fn pierson_moskowitz(f: f64, u10: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let g = 9.81;
    let alpha = 8.1e-3;
    let f0 = g / (2.0 * PI * u10 * 1.026);
    alpha * g * g * (TWO_PI).powi(-4) * f.powi(-5) * (-1.25 * (f0 / f).powi(4)).exp()
}

/// Synthesize a sea-surface elevation time series from a one-sided
/// spectrum (random phases): `n` samples at rate `fs`.
#[must_use]
pub fn wave_field_from_spectrum(
    spectrum: &dyn Fn(f64) -> f64,
    n: usize,
    fs: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let df = fs / n as f64;
    let n_bins = n / 2;
    let comps: Vec<(f64, f64, f64)> = (1..n_bins)
        .map(|k| {
            let f = k as f64 * df;
            let amp = (2.0 * spectrum(f) * df).max(0.0).sqrt();
            (amp, TWO_PI * f, TWO_PI * rng.next_f64())
        })
        .collect();
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            comps.iter().map(|&(a, w, ph)| a * (w * t + ph).cos()).sum()
        })
        .collect()
}

/// Gerstner (trochoidal) wave displacement of the surface point whose
/// rest position is `p`, summing waves given as (amplitude, wavelength,
/// speed, direction).
#[must_use]
pub fn gerstner_wave(p: Vec2, t: f64, waves: &[(f64, f64, f64, Vec2)]) -> Vec3 {
    let mut out = Vec3::new(p.x, p.y, 0.0);
    for &(a, lambda, speed, dir) in waves {
        let d = dir.normalized();
        let k = TWO_PI / lambda;
        let phase = k * (d.dot(&p) - speed * t);
        out.x -= d.x * a * phase.sin();
        out.y -= d.y * a * phase.sin();
        out.z += a * phase.cos();
    }
    out
}

/// Kelvin ship-wake half angle arcsin(1/3) ≈ 19.47°.
#[must_use]
pub fn kelvin_wake_angle() -> f64 {
    (1.0_f64 / 3.0).asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lake_at_rest_well_balanced() {
        let n = 32;
        let mut sw = ShallowWater2D::new(n, n, 1.0 / n as f64, 9.81);
        sw.set_bathymetry(|x, y| {
            0.3 * (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.02).exp()
        });
        // Flat free surface at level 1: h = 1 − b.
        for j in 0..n {
            for i in 0..n {
                let c = j * n + i;
                sw.h[c] = (1.0 - sw.bathymetry[c]).max(0.0);
            }
        }
        for _ in 0..50 {
            sw.step(0.4);
        }
        let max_speed = (0..n * n)
            .map(|c| {
                if sw.h[c] > sw.dry_tol {
                    (sw.hu[c].abs() + sw.hv[c].abs()) / sw.h[c]
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);
        assert!(max_speed < 1e-12, "lake at rest disturbed: {max_speed}");
    }

    #[test]
    fn test_stoker_dam_break() {
        let n = 400;
        let domain = 40.0;
        let mut sw = ShallowWater2D::new(n, 4, domain / n as f64, 9.81);
        sw.set_dam_break(0.5 * domain, 2.0, 1.0);
        let t_end = 1.0;
        sw.run_until(t_end);
        let dx = domain / n as f64;
        // L1 error (a pointwise metric at the smeared front is O(1) for
        // any shock-capturing scheme).
        let mut l1 = 0.0_f64;
        for i in 0..n {
            let x = (i as f64 + 0.5) * dx - 0.5 * domain;
            let (he, _) = swe_1d_exact_dam_break(x, sw.time, 2.0, 1.0, 9.81);
            let hn = sw.h[n + i]; // second row (uniform in y)
            l1 += (hn - he).abs();
        }
        l1 /= n as f64 * 2.0;
        assert!(l1 < 0.01, "Stoker L1 mismatch {l1}");
        // The star-region plateau depth matches within 3%.
        let (hm_exact, um_exact) = swe_1d_exact_dam_break(1.0, sw.time, 2.0, 1.0, 9.81);
        let i_mid = ((1.0 + 0.5 * domain) / dx) as usize;
        let h_mid = sw.h[n + i_mid];
        assert!(
            (h_mid / hm_exact - 1.0).abs() < 0.03,
            "plateau depth {h_mid} vs {hm_exact}"
        );
        let u_mid = sw.hu[n + i_mid] / h_mid;
        assert!((u_mid / um_exact - 1.0).abs() < 0.05, "plateau speed {u_mid} vs {um_exact}");
        // Volume conserved.
        let expect = (2.0 * 0.5 + 1.0 * 0.5) * domain * (4.0 * dx);
        assert!((sw.total_volume() / expect - 1.0).abs() < 1e-12, "volume drift");
    }

    #[test]
    fn test_bump_propagation_speed_and_energy() {
        let n = 96;
        let mut sw = ShallowWater2D::new(n, n, 1.0 / n as f64, 9.81);
        let depth = 0.1;
        sw.set_gaussian_bump(0.5, 0.5, 0.005, 0.04, depth);
        let e0 = sw.total_energy();
        let v0 = sw.total_volume();
        // Track the ring radius: run to t and find the crest along +x.
        sw.run_until(0.15);
        let mut best = (0, 0.0);
        for i in n / 2..n {
            let h = sw.h[(n / 2) * n + i];
            let eta = h - depth;
            if eta > best.1 {
                best = (i, eta);
            }
        }
        let r = (best.0 as f64 + 0.5) / n as f64 - 0.5;
        let c_expect = (9.81 * depth).sqrt();
        let r_expect = c_expect * sw.time;
        assert!(
            (r - r_expect).abs() < 0.04,
            "ring at {r}, expected {r_expect}"
        );
        assert!((sw.total_volume() / v0 - 1.0).abs() < 1e-12);
        // Energy non-increasing (HLL is dissipative).
        assert!(sw.total_energy() <= e0 * (1.0 + 1e-9));
        assert!(sw.max_depth() > 0.0);
        assert_eq!(sw.wet_cells(), n * n);
        assert!(sw.froude_field().iter().all(|f| f.is_finite()));
    }

    #[test]
    fn test_tsunami_runup() {
        let shoreline = tsunami_runup_1d(0.05, 0.02, 4.0, 20.0, 200, 100.0, 40.0, 20);
        assert_eq!(shoreline.len(), 20);
        assert!(shoreline.iter().all(|s| s.is_finite() && *s > 0.0));
        // The shoreline moves (run-up then drawdown).
        let max_s = shoreline.iter().cloned().fold(f64::MIN, f64::max);
        let min_s = shoreline.iter().cloned().fold(f64::MAX, f64::min);
        assert!(max_s - min_s > 0.2, "shoreline static: {min_s}..{max_s}");
    }

    #[test]
    fn test_dispersion_and_spectra() {
        let g = 9.81;
        // Limits of the full dispersion relation.
        let k = 0.01; // long wave in h = 10 m
        assert!(
            (dispersion_full(k, 10.0, g) / dispersion_shallow(k, 10.0, g) - 1.0).abs() < 0.01
        );
        let k2 = 5.0; // short wave
        assert!((dispersion_full(k2, 10.0, g) / dispersion_deep(k2, g) - 1.0).abs() < 1e-6);
        assert!((wave_speed_shallow(10.0, g) - (98.1_f64).sqrt()).abs() < 1e-12);
        // Stokes drift: deep-water limit a²ωk.
        let (a, kk) = (0.5, 0.1);
        let deep = stokes_drift(a, kk, 1e4, g);
        assert!((deep / (a * a * dispersion_deep(kk, g) * kk) - 1.0).abs() < 1e-6);
        assert!(!wave_breaking_criterion(0.1));
        assert!(wave_breaking_criterion(0.15));
        // JONSWAP: integrates to hs²/16, peaks near fp.
        let (hs, tp) = (2.0, 8.0);
        let n = 2000;
        let df = 1.0 / n as f64;
        let m0: f64 = (1..n).map(|k| jonswap_spectrum(k as f64 * df, hs, tp, 3.3)).sum::<f64>() * df;
        assert!((m0 / (hs * hs / 16.0) - 1.0).abs() < 0.02, "m0 {m0}");
        let s_peak = jonswap_spectrum(1.0 / tp, hs, tp, 3.3);
        assert!(s_peak > jonswap_spectrum(0.5 / tp, hs, tp, 3.3));
        assert!(s_peak > jonswap_spectrum(2.0 / tp, hs, tp, 3.3));
        // PM peak frequency scales inversely with wind speed.
        let pm_peak = |u: f64| -> f64 {
            let mut best = (0.0, 0.0);
            for k in 1..500 {
                let f = k as f64 * 0.002;
                let s = pierson_moskowitz(f, u);
                if s > best.1 {
                    best = (f, s);
                }
            }
            best.0
        };
        assert!(pm_peak(20.0) < pm_peak(10.0));
        // Wave synthesis: variance matches the spectrum integral.
        let mut rng = Rng::new(12);
        let spec = |f: f64| jonswap_spectrum(f, hs, tp, 3.3);
        let ts = wave_field_from_spectrum(&spec, 4096, 2.0, &mut rng);
        let var = ts.iter().map(|v| v * v).sum::<f64>() / ts.len() as f64;
        assert!((var / (hs * hs / 16.0) - 1.0).abs() < 0.1, "variance {var}");
        // Gerstner wave: circular particle path of radius a.
        let waves = [(0.5, 10.0, 3.0, Vec2::new(1.0, 0.0))];
        let p0 = gerstner_wave(Vec2::new(0.0, 0.0), 0.0, &waves);
        let mut max_z = f64::MIN;
        let mut min_z = f64::MAX;
        for k in 0..100 {
            let s = gerstner_wave(Vec2::new(0.0, 0.0), k as f64 * 0.1, &waves);
            max_z = max_z.max(s.z);
            min_z = min_z.min(s.z);
        }
        assert!((max_z - 0.5).abs() < 1e-6 && (min_z + 0.5).abs() < 1e-6);
        assert!(p0.z <= 0.5 + 1e-12);
        assert!((kelvin_wake_angle().to_degrees() - 19.47).abs() < 0.01);
    }
}
