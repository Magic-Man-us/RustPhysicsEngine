//! Vortex methods: regularized Biot-Savart particle methods in 2D and 3D,
//! classical vortex solutions (Lamb-Oseen, Rankine, Burgers, Hill), point
//! vortex dynamics with a symplectic integrator, and vortex phenomenology
//! (shedding, Crow instability, tip vortices).

use crate::math::{Vec2, Vec3};

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 3D vortex particle method
// ---------------------------------------------------------------------------

/// A vector-valued vortex particle: strength is circulation times length
/// (the integral of vorticity over the particle's volume).
#[derive(Debug, Clone, Copy)]
pub struct VortexParticle {
    pub pos: Vec3,
    pub strength: Vec3,
    pub core: f64,
}

/// Regularization kernel for the Biot-Savart sum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VortexKernel {
    /// Bare 1/r^3 kernel (singular at particle positions).
    Singular,
    /// Rosenhead-Moore algebraic smoothing with radius delta.
    Rosenhead(f64),
    /// Gaussian core of radius sigma.
    Gaussian(f64),
    /// Winckelmans-Leonard high-order algebraic kernel with radius delta.
    HighOrder(f64),
}

impl VortexKernel {
    /// The factor K(r) multiplying (p - x) x alpha / (4 pi) so that the
    /// singular case is 1/r^3.
    fn factor(&self, r: f64) -> f64 {
        match *self {
            VortexKernel::Singular => {
                if r < 1e-12 {
                    0.0
                } else {
                    1.0 / (r * r * r)
                }
            }
            VortexKernel::Rosenhead(delta) => {
                let d2 = r * r + delta * delta;
                1.0 / (d2 * d2.sqrt())
            }
            VortexKernel::Gaussian(sigma) => {
                if r < 1e-12 * sigma.max(1e-30) {
                    return 0.0;
                }
                let rho = r / sigma;
                let g = crate::special::erf(rho / std::f64::consts::SQRT_2)
                    - rho * (2.0 / std::f64::consts::PI).sqrt() * (-0.5 * rho * rho).exp();
                g / (r * r * r)
            }
            VortexKernel::HighOrder(delta) => {
                let d2 = r * r + delta * delta;
                (r * r + 2.5 * delta * delta) / (d2 * d2 * d2.sqrt())
            }
        }
    }
}

/// 3D vortex particle method with direct Biot-Savart summation.
#[derive(Debug, Clone)]
pub struct VortexMethod3 {
    pub particles: Vec<VortexParticle>,
    pub kernel: VortexKernel,
}

impl VortexMethod3 {
    #[must_use]
    pub fn new(particles: Vec<VortexParticle>, kernel: VortexKernel) -> Self {
        Self { particles, kernel }
    }

    /// Velocity induced at `p` by all particles:
    /// u = -(1/4pi) sum (p - x_i) x alpha_i K(|p - x_i|).
    #[must_use]
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        let mut u = Vec3::new(0.0, 0.0, 0.0);
        for q in &self.particles {
            let d = p - q.pos;
            let k = self.kernel.factor(d.magnitude());
            u = u - d.cross(&q.strength) * (k / FOUR_PI);
        }
        u
    }

    /// Velocity gradient at `p` by central differences with spacing `h`.
    fn velocity_gradient(&self, p: Vec3, h: f64) -> [[f64; 3]; 3] {
        let mut g = [[0.0; 3]; 3];
        let ex = [
            Vec3::new(h, 0.0, 0.0),
            Vec3::new(0.0, h, 0.0),
            Vec3::new(0.0, 0.0, h),
        ];
        for (j, e) in ex.iter().enumerate() {
            let up = self.velocity_at(p + *e);
            let um = self.velocity_at(p - *e);
            g[0][j] = (up.x - um.x) / (2.0 * h);
            g[1][j] = (up.y - um.y) / (2.0 * h);
            g[2][j] = (up.z - um.z) / (2.0 * h);
        }
        g
    }

    /// One step: RK2 advection of positions, vortex stretching of strengths
    /// (classical scheme, alpha' = (alpha . grad) u), and viscous diffusion
    /// by core spreading (the kernel radius grows as sigma^2 += 2 nu dt).
    pub fn step(&mut self, dt: f64, nu: f64) {
        let h = self
            .particles
            .iter()
            .map(|p| p.core)
            .fold(f64::INFINITY, f64::min)
            .max(1e-6)
            * 0.5;
        // RK2 midpoint for positions; strengths stretched with midpoint
        // gradients.
        let snapshot = self.clone();
        let mids: Vec<Vec3> = self
            .particles
            .iter()
            .map(|p| p.pos + snapshot.velocity_at(p.pos) * (0.5 * dt))
            .collect();
        let new_state: Vec<(Vec3, Vec3)> = self
            .particles
            .iter()
            .zip(&mids)
            .map(|(p, &m)| {
                let u_mid = snapshot.velocity_at(m);
                let g = snapshot.velocity_gradient(m, h);
                let a = p.strength;
                // classical scheme: d alpha_i/dt = alpha_j du_i/dx_j
                let da = Vec3::new(
                    g[0][0] * a.x + g[0][1] * a.y + g[0][2] * a.z,
                    g[1][0] * a.x + g[1][1] * a.y + g[1][2] * a.z,
                    g[2][0] * a.x + g[2][1] * a.y + g[2][2] * a.z,
                );
                (p.pos + u_mid * dt, a + da * dt)
            })
            .collect();
        for (p, (pos, alpha)) in self.particles.iter_mut().zip(new_state) {
            p.pos = pos;
            p.strength = alpha;
            p.core = (p.core * p.core + 2.0 * nu * dt).sqrt();
        }
        // keep the kernel radius in sync with the (uniform) particle core
        let mean_core =
            self.particles.iter().map(|p| p.core).sum::<f64>() / self.particles.len() as f64;
        self.kernel = match self.kernel {
            VortexKernel::Singular => VortexKernel::Singular,
            VortexKernel::Rosenhead(_) => VortexKernel::Rosenhead(mean_core),
            VortexKernel::Gaussian(_) => VortexKernel::Gaussian(mean_core),
            VortexKernel::HighOrder(_) => VortexKernel::HighOrder(mean_core),
        };
    }

    /// A discretized circular vortex ring of circulation `strength` in the
    /// plane perpendicular to `normal`.
    #[must_use]
    pub fn vortex_ring(
        center: Vec3,
        radius: f64,
        strength: f64,
        core: f64,
        n: usize,
    ) -> Self {
        Self::ring_with_normal(center, radius, strength, core, n, Vec3::new(0.0, 0.0, 1.0))
    }

    fn ring_with_normal(
        center: Vec3,
        radius: f64,
        strength: f64,
        core: f64,
        n: usize,
        normal: Vec3,
    ) -> Self {
        let nrm = normal.normalized();
        // orthonormal basis in the ring plane
        let helper = if nrm.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let e1 = nrm.cross(&helper).normalized();
        let e2 = nrm.cross(&e1);
        let seg = TWO_PI * radius / n as f64;
        let particles = (0..n)
            .map(|i| {
                let th = TWO_PI * i as f64 / n as f64;
                let pos = center + e1 * (radius * th.cos()) + e2 * (radius * th.sin());
                let tangent = e2 * th.cos() - e1 * th.sin();
                VortexParticle {
                    pos,
                    strength: tangent * (strength * seg),
                    core,
                }
            })
            .collect();
        Self::new(particles, VortexKernel::HighOrder(core))
    }

    /// Two coaxial rings of the same sign separated by `gap` along z: the
    /// classical leapfrogging configuration.
    #[must_use]
    pub fn two_rings_leapfrog(radius: f64, strength: f64, core: f64, gap: f64, n: usize) -> Self {
        let mut a = Self::vortex_ring(Vec3::new(0.0, 0.0, 0.0), radius, strength, core, n);
        let b = Self::vortex_ring(Vec3::new(0.0, 0.0, gap), radius, strength, core, n);
        a.particles.extend(b.particles);
        a
    }

    /// Discrete helicity sum u(x_i) . alpha_i.
    #[must_use]
    pub fn helicity(&self) -> f64 {
        self.particles
            .iter()
            .map(|p| self.velocity_at(p.pos).dot(&p.strength))
            .sum()
    }

    /// Particle-strength enstrophy proxy sum |alpha_i|^2.
    #[must_use]
    pub fn enstrophy(&self) -> f64 {
        self.particles.iter().map(|p| p.strength.magnitude_squared()).sum()
    }

    /// Kinetic energy estimate E = (1/8pi) sum_{i != j} alpha_i . alpha_j /
    /// r_ij (regularized with the particle core).
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let mut e = 0.0;
        for (i, a) in self.particles.iter().enumerate() {
            for b in self.particles.iter().skip(i + 1) {
                let r = (a.pos - b.pos).magnitude().max(0.5 * (a.core + b.core));
                e += a.strength.dot(&b.strength) / r;
            }
        }
        e / FOUR_PI
    }

    /// Hydrodynamic impulse I = (1/2) sum x_i x alpha_i.
    #[must_use]
    pub fn impulse(&self) -> Vec3 {
        let mut i = Vec3::new(0.0, 0.0, 0.0);
        for p in &self.particles {
            i = i + p.pos.cross(&p.strength) * 0.5;
        }
        i
    }

    /// Merge particles onto a lattice of the given spacing: strengths add,
    /// positions are strength-weighted centroids.
    pub fn remesh(&mut self, spacing: f64) {
        use std::collections::HashMap;
        // accumulator: (weighted position, strength sum, weight, weighted core)
        type CellAcc = (Vec3, Vec3, f64, f64);
        let mut cells: HashMap<(i64, i64, i64), CellAcc> = HashMap::new();
        for p in &self.particles {
            let key = (
                (p.pos.x / spacing).round() as i64,
                (p.pos.y / spacing).round() as i64,
                (p.pos.z / spacing).round() as i64,
            );
            let w = p.strength.magnitude().max(1e-30);
            let e = cells
                .entry(key)
                .or_insert((Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0.0, 0.0));
            e.0 = e.0 + p.pos * w;
            e.1 = e.1 + p.strength;
            e.2 += w;
            e.3 += p.core * w;
        }
        self.particles = cells
            .into_values()
            .map(|(pw, alpha, w, cw)| VortexParticle {
                pos: pw * (1.0 / w),
                strength: alpha,
                core: cw / w,
            })
            .collect();
    }
}

// ---------------------------------------------------------------------------
// 2D vortex particle method
// ---------------------------------------------------------------------------

/// 2D vortex blob method: particles carry scalar circulation.
#[derive(Debug, Clone)]
pub struct VortexMethod2 {
    /// (position, circulation) pairs.
    pub particles: Vec<(Vec2, f64)>,
    /// Rosenhead smoothing radius.
    pub delta: f64,
}

impl VortexMethod2 {
    #[must_use]
    pub fn new(particles: Vec<(Vec2, f64)>, delta: f64) -> Self {
        Self { particles, delta }
    }

    /// Velocity at `p`: u = sum Gamma_i/(2 pi) (-dy, dx)/(r^2 + delta^2).
    #[must_use]
    pub fn velocity_at(&self, p: Vec2) -> Vec2 {
        let mut u = Vec2::new(0.0, 0.0);
        for &(x, g) in &self.particles {
            let d = p - x;
            let r2 = d.magnitude_squared() + self.delta * self.delta;
            if r2 < 1e-24 {
                continue;
            }
            u = u + Vec2::new(-d.y, d.x) * (g / (TWO_PI * r2));
        }
        u
    }

    /// Velocity at `p` excluding the contribution of particle `skip` (used
    /// during time stepping so a particle's own slightly-displaced old
    /// position cannot induce a spurious self-velocity).
    fn velocity_excluding(&self, p: Vec2, skip: usize) -> Vec2 {
        let mut u = Vec2::new(0.0, 0.0);
        for (j, &(x, g)) in self.particles.iter().enumerate() {
            if j == skip {
                continue;
            }
            let d = p - x;
            let r2 = d.magnitude_squared() + self.delta * self.delta;
            if r2 < 1e-24 {
                continue;
            }
            u = u + Vec2::new(-d.y, d.x) * (g / (TWO_PI * r2));
        }
        u
    }

    /// One step: RK2 advection; viscosity by core spreading of `delta`.
    pub fn step(&mut self, dt: f64, nu: f64) {
        let snapshot = self.clone();
        let mids: Vec<Vec2> = self
            .particles
            .iter()
            .enumerate()
            .map(|(i, &(x, _))| x + snapshot.velocity_excluding(x, i) * (0.5 * dt))
            .collect();
        for (i, ((x, _), m)) in self.particles.iter_mut().zip(mids).enumerate() {
            *x = *x + snapshot.velocity_excluding(m, i) * dt;
        }
        self.delta = (self.delta * self.delta + 2.0 * nu * dt).sqrt();
    }

    /// Flat vortex sheet of total circulation `gamma_total` along the x axis
    /// from 0 to `length`, discretized into `n` blobs.
    #[must_use]
    pub fn vortex_sheet(n: usize, gamma_total: f64, length: f64, delta: f64) -> Self {
        let particles = (0..n)
            .map(|i| {
                let x = length * (i as f64 + 0.5) / n as f64;
                (Vec2::new(x, 0.0), gamma_total / n as f64)
            })
            .collect();
        Self::new(particles, delta)
    }

    /// Sinusoidally perturbed periodic vortex sheet (one wavelength) for
    /// Kelvin-Helmholtz roll-up studies.
    #[must_use]
    pub fn kelvin_helmholtz_roll_up(n: usize, delta_u: f64, wavelength: f64, amp: f64) -> Self {
        // sheet strength = velocity jump
        let gamma_total = delta_u * wavelength;
        let delta = 0.3 * wavelength / n as f64 * 10.0;
        let particles = (0..n)
            .map(|i| {
                let s = wavelength * (i as f64 + 0.5) / n as f64;
                let y = amp * (TWO_PI * s / wavelength).sin();
                (Vec2::new(s, y), gamma_total / n as f64)
            })
            .collect();
        Self::new(particles, delta)
    }

    /// Two counter-rotating point vortices separated by `d` (propagate
    /// together at Gamma/(2 pi d)).
    #[must_use]
    pub fn point_vortex_pair(gamma: f64, d: f64) -> Self {
        Self::new(
            vec![
                (Vec2::new(0.0, 0.5 * d), gamma),
                (Vec2::new(0.0, -0.5 * d), -gamma),
            ],
            0.0,
        )
    }

    /// Discretize a Lamb-Oseen vortex of circulation `gamma` and core radius
    /// `r_c` into `n` rings of blobs.
    #[must_use]
    pub fn lamb_oseen_init(gamma: f64, r_c: f64, n: usize) -> Self {
        let mut particles = Vec::new();
        let r_max = 3.0 * r_c;
        let dr = r_max / n as f64;
        for i in 0..n {
            let r0 = i as f64 * dr;
            let r1 = r0 + dr;
            let rm = 0.5 * (r0 + r1);
            // circulation in the annulus from the Gaussian vorticity
            let g_ann = gamma
                * ((-r0 * r0 / (r_c * r_c)).exp() - (-r1 * r1 / (r_c * r_c)).exp());
            let n_th = (8.max(i * 6)).max(1);
            for k in 0..n_th {
                let th = TWO_PI * k as f64 / n_th as f64;
                particles.push((Vec2::new(rm * th.cos(), rm * th.sin()), g_ann / n_th as f64));
            }
        }
        Self::new(particles, 0.5 * dr)
    }

    /// Merge blobs closer than `eps`: circulation adds, position is the
    /// circulation-magnitude-weighted centroid.
    pub fn merge_near(&mut self, eps: f64) {
        let mut merged: Vec<(Vec2, f64)> = Vec::new();
        let mut used = vec![false; self.particles.len()];
        for i in 0..self.particles.len() {
            if used[i] {
                continue;
            }
            let (mut pos_w, mut g_sum) = (Vec2::new(0.0, 0.0), 0.0);
            let mut w_sum = 0.0;
            let anchor = self.particles[i].0;
            for (j, uj) in used.iter_mut().enumerate().skip(i) {
                if *uj {
                    continue;
                }
                if (self.particles[j].0 - anchor).magnitude() <= eps {
                    let w = self.particles[j].1.abs().max(1e-30);
                    pos_w = pos_w + self.particles[j].0 * w;
                    g_sum += self.particles[j].1;
                    w_sum += w;
                    *uj = true;
                }
            }
            merged.push((pos_w * (1.0 / w_sum), g_sum));
        }
        self.particles = merged;
    }

    /// Sample the blob velocity field onto a MAC grid covering the particle
    /// bounding box (padded by 10%).
    #[must_use]
    pub fn to_grid(&self, nx: usize, ny: usize) -> crate::cfd::grid::MacGrid2 {
        let (mut lo, mut hi) = (Vec2::new(f64::MAX, f64::MAX), Vec2::new(f64::MIN, f64::MIN));
        for &(p, _) in &self.particles {
            lo.x = lo.x.min(p.x);
            lo.y = lo.y.min(p.y);
            hi.x = hi.x.max(p.x);
            hi.y = hi.y.max(p.y);
        }
        let span = ((hi.x - lo.x).max(hi.y - lo.y)).max(1e-9);
        let dx = 1.2 * span / nx.max(ny) as f64;
        let origin = Vec2::new(
            0.5 * (lo.x + hi.x) - 0.5 * nx as f64 * dx,
            0.5 * (lo.y + hi.y) - 0.5 * ny as f64 * dx,
        );
        let mut g = crate::cfd::grid::MacGrid2::new(nx, ny, dx);
        for j in 0..ny {
            for i in 0..=nx {
                let p = origin + Vec2::new(i as f64 * dx, (j as f64 + 0.5) * dx);
                g.u[j * (nx + 1) + i] = self.velocity_at(p).x;
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                let p = origin + Vec2::new((i as f64 + 0.5) * dx, j as f64 * dx);
                g.v[j * nx + i] = self.velocity_at(p).y;
            }
        }
        g
    }
}

// ---------------------------------------------------------------------------
// Biot-Savart building blocks and classical vortices
// ---------------------------------------------------------------------------

/// Velocity induced at `p` by a straight vortex segment from `a` to `b`
/// carrying circulation `gamma`.
#[must_use]
pub fn biot_savart_segment(p: Vec3, a: Vec3, b: Vec3, gamma: f64) -> Vec3 {
    let r1 = p - a;
    let r2 = p - b;
    let r0 = b - a;
    let c = r1.cross(&r2);
    let c2 = c.magnitude_squared();
    if c2 < 1e-24 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let m1 = r1.magnitude();
    let m2 = r2.magnitude();
    c * (gamma / (FOUR_PI * c2) * (r0.dot(&r1) / m1 - r0.dot(&r2) / m2))
}

/// Velocity induced at `p` by a circular vortex ring discretized into
/// `n_seg` straight segments.
#[must_use]
pub fn biot_savart_ring(
    p: Vec3,
    center: Vec3,
    radius: f64,
    gamma: f64,
    normal: Vec3,
    n_seg: usize,
) -> Vec3 {
    let nrm = normal.normalized();
    let helper = if nrm.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let e1 = nrm.cross(&helper).normalized();
    let e2 = nrm.cross(&e1);
    let mut u = Vec3::new(0.0, 0.0, 0.0);
    let node = |k: usize| {
        let th = TWO_PI * k as f64 / n_seg as f64;
        center + e1 * (radius * th.cos()) + e2 * (radius * th.sin())
    };
    for k in 0..n_seg {
        u = u + biot_savart_segment(p, node(k), node(k + 1), gamma);
    }
    u
}

/// Kelvin's formula for the self-induced translation speed of a thin vortex
/// ring: U = Gamma/(4 pi R) (ln(8R/a) - 1/4).
#[must_use]
pub fn vortex_ring_self_velocity(gamma: f64, r: f64, core: f64) -> f64 {
    gamma / (FOUR_PI * r) * ((8.0 * r / core).ln() - 0.25)
}

/// Lamb-Oseen azimuthal velocity at radius `r` and time `t`:
/// v = Gamma/(2 pi r) (1 - exp(-r^2/(4 nu t))).
#[must_use]
pub fn lamb_oseen_velocity(r: f64, t: f64, gamma: f64, nu: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    gamma / (TWO_PI * r) * (1.0 - (-r * r / (4.0 * nu * t)).exp())
}

/// Rankine vortex: solid-body rotation inside `r_core`, potential outside.
#[must_use]
pub fn rankine_vortex(r: f64, r_core: f64, gamma: f64) -> f64 {
    if r <= r_core {
        gamma * r / (TWO_PI * r_core * r_core)
    } else {
        gamma / (TWO_PI * r)
    }
}

/// Burgers vortex azimuthal velocity: the steady balance of diffusion
/// against axial strain `strain` (units 1/s):
/// v = Gamma/(2 pi r) (1 - exp(-strain r^2/(4 nu))).
#[must_use]
pub fn burgers_vortex(r: f64, gamma: f64, nu: f64, strain: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    gamma / (TWO_PI * r) * (1.0 - (-strain * r * r / (4.0 * nu)).exp())
}

/// Hill's spherical vortex of radius `a` in the co-moving frame: far-field
/// velocity is -u along z, the sphere surface is a stream surface, and the
/// poles are stagnation points.
#[must_use]
pub fn hill_spherical_vortex(p: Vec3, u: f64, a: f64) -> Vec3 {
    let w2 = p.x * p.x + p.y * p.y;
    let w = w2.sqrt();
    let z = p.z;
    let r2 = w2 + z * z;
    // Stokes stream function: interior psi = (3u/4) w^2 (1 - r^2/a^2),
    // exterior psi = (u/2) w^2 (a^3/r^3 - 1); velocities
    // u_z = (1/w) dpsi/dw, u_w = -(1/w) dpsi/dz. Continuous at r = a with
    // stagnation points at the poles.
    let a2 = a * a;
    let a3 = a2 * a;
    let (uw, uz) = if r2 <= a2 {
        let uz = 1.5 * u * (1.0 - (2.0 * w2 + z * z) / a2);
        let uw = 1.5 * u * w * z / a2;
        (uw, uz)
    } else {
        let r = r2.sqrt();
        let r5 = r2 * r2 * r;
        let uz = u * (a3 / (r2 * r) - 1.0) - 1.5 * u * a3 * w2 / r5;
        let uw = 1.5 * u * a3 * w * z / r5;
        (uw, uz)
    };
    if w < 1e-14 {
        Vec3::new(0.0, 0.0, uz)
    } else {
        Vec3::new(uw * p.x / w, uw * p.y / w, uz)
    }
}

/// Translation speed of a counter-rotating vortex pair: Gamma/(2 pi d).
#[must_use]
pub fn vortex_pair_velocity(gamma: f64, d: f64) -> f64 {
    gamma / (TWO_PI * d)
}

/// Point-vortex Hamiltonian H = -(1/4 pi) sum_{i<j} G_i G_j ln r_ij.
#[must_use]
pub fn point_vortex_hamiltonian(pos: &[Vec2], gammas: &[f64]) -> f64 {
    let mut h = 0.0;
    for i in 0..pos.len() {
        for j in (i + 1)..pos.len() {
            let r = (pos[i] - pos[j]).magnitude();
            h -= gammas[i] * gammas[j] * r.ln() / FOUR_PI;
        }
    }
    h
}

fn point_vortex_rhs(pos: &[Vec2], gammas: &[f64], out: &mut [Vec2]) {
    for (i, o) in out.iter_mut().enumerate() {
        let mut u = Vec2::new(0.0, 0.0);
        for (j, &g) in gammas.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = pos[i] - pos[j];
            let r2 = d.magnitude_squared();
            if r2 < 1e-24 {
                continue;
            }
            u = u + Vec2::new(-d.y, d.x) * (g / (TWO_PI * r2));
        }
        *o = u;
    }
}

/// One implicit-midpoint step of the point-vortex system (symplectic; the
/// Hamiltonian error stays bounded over long integrations).
pub fn point_vortex_step(pos: &mut [Vec2], gammas: &[f64], dt: f64) {
    let n = pos.len();
    let start: Vec<Vec2> = pos.to_vec();
    let mut mid: Vec<Vec2> = pos.to_vec();
    let mut vel = vec![Vec2::new(0.0, 0.0); n];
    for _ in 0..100 {
        point_vortex_rhs(&mid, gammas, &mut vel);
        let mut max_change = 0.0_f64;
        for i in 0..n {
            let new_mid = start[i] + vel[i] * (0.5 * dt);
            max_change = max_change.max((new_mid - mid[i]).magnitude());
            mid[i] = new_mid;
        }
        if max_change < 1e-15 {
            break;
        }
    }
    point_vortex_rhs(&mid, gammas, &mut vel);
    for i in 0..n {
        pos[i] = start[i] + vel[i] * dt;
    }
}

// ---------------------------------------------------------------------------
// Vortex phenomenology
// ---------------------------------------------------------------------------

/// Inviscid Kelvin-Helmholtz growth rate for a velocity jump `delta_u`:
/// sigma = k delta_u / 2.
#[must_use]
pub fn kelvin_helmholtz_growth_exact(k: f64, delta_u: f64) -> f64 {
    0.5 * k * delta_u
}

/// Shedding frequency f = St U / D.
#[must_use]
pub fn vortex_shedding_frequency(strouhal: f64, u: f64, d: f64) -> f64 {
    strouhal * u / d
}

/// Roshko-style Strouhal-Reynolds correlation for a circular cylinder.
#[must_use]
pub fn strouhal_from_re(re: f64) -> f64 {
    if re < 50.0 {
        0.0
    } else if re < 200.0 {
        0.212 * (1.0 - 21.2 / re)
    } else {
        0.198 * (1.0 - 19.7 / re)
    }
}

/// Crow instability growth rate for a trailing-vortex pair of spacing `b`
/// (approximate peak rate ~0.83 Gamma/(2 pi b^2), weakly dependent on core).
#[must_use]
pub fn crow_instability_growth(b: f64, gamma: f64, core: f64) -> f64 {
    let correction = 1.0 - 0.5 * (core / b).min(0.5);
    0.83 * gamma / (TWO_PI * b * b) * correction
}

/// Peak swirl velocity of a decaying tip vortex at time `t` (Lamb-Oseen core
/// growth from initial core radius `r_core0`).
#[must_use]
pub fn tip_vortex_decay(gamma: f64, r_core0: f64, nu: f64, t: f64) -> f64 {
    let alpha = 1.256_43;
    let r_c = (r_core0 * r_core0 + 4.0 * alpha * nu * t).sqrt();
    // peak of Lamb-Oseen profile: 0.6382 Gamma/(2 pi r_c)
    0.638_2 * gamma / (TWO_PI * r_c)
}

/// Helicity density u . omega.
#[must_use]
pub fn helicity_density(u: Vec3, omega: Vec3) -> f64 {
    u.dot(&omega)
}

/// Trace a vortex line through a vorticity field by RK4 along the normalized
/// field direction, with arc-length step `ds`.
pub fn vortex_line_trace(
    omega_field: &dyn Fn(Vec3) -> Vec3,
    seed: Vec3,
    steps: usize,
    ds: f64,
) -> Vec<Vec3> {
    let dir = |p: Vec3| -> Vec3 {
        let w = omega_field(p);
        let m = w.magnitude();
        if m < 1e-30 {
            Vec3::new(0.0, 0.0, 0.0)
        } else {
            w * (1.0 / m)
        }
    };
    let mut out = Vec::with_capacity(steps + 1);
    let mut p = seed;
    out.push(p);
    for _ in 0..steps {
        let k1 = dir(p);
        let k2 = dir(p + k1 * (0.5 * ds));
        let k3 = dir(p + k2 * (0.5 * ds));
        let k4 = dir(p + k3 * ds);
        p = p + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (ds / 6.0);
        out.push(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biot_savart_ring_on_axis() {
        // on-axis closed form: u_z = Gamma R^2 / (2 (R^2 + z^2)^{3/2})
        let gamma = 2.0;
        let r = 1.5;
        for &z in &[0.0, 0.5, 2.0] {
            let u = biot_savart_ring(
                Vec3::new(0.0, 0.0, z),
                Vec3::new(0.0, 0.0, 0.0),
                r,
                gamma,
                Vec3::new(0.0, 0.0, 1.0),
                720,
            );
            let exact = gamma * r * r / (2.0 * (r * r + z * z).powf(1.5));
            assert!((u.z.abs() - exact).abs() < 1e-4 * exact, "z={z}: {} vs {exact}", u.z);
            assert!(u.x.abs() < 1e-10 && u.y.abs() < 1e-10);
        }
        // infinite-line limit via a long segment: u = Gamma/(2 pi d)
        let u = biot_savart_segment(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1e4, 0.0, 0.0),
            Vec3::new(1e4, 0.0, 0.0),
            1.0,
        );
        assert!((u.z.abs() - 1.0 / TWO_PI).abs() < 1e-6);
    }

    #[test]
    fn test_vortex_ring_translation() {
        // a discretized thin ring translates near the Kelvin speed; the
        // cores must overlap (spacing 2 pi/64 < core) for smooth gradients
        let (r, gamma, core) = (1.0, 1.0, 0.15);
        let mut ring = VortexMethod3::vortex_ring(Vec3::new(0.0, 0.0, 0.0), r, gamma, core, 64);
        let u_kelvin = vortex_ring_self_velocity(gamma, r, core);
        let z0 = ring.particles.iter().map(|p| p.pos.z).sum::<f64>() / 64.0;
        let dt = 0.02;
        for _ in 0..25 {
            ring.step(dt, 0.0);
        }
        let z1 = ring.particles.iter().map(|p| p.pos.z).sum::<f64>() / 64.0;
        let u_meas = (z1 - z0) / (25.0 * dt);
        // a blob-discretized ring translates somewhat below the thin-core
        // Kelvin speed; check direction and magnitude to within a factor
        assert!(
            u_meas > 0.5 * u_kelvin && u_meas < 1.2 * u_kelvin,
            "ring speed {u_meas} vs Kelvin {u_kelvin}"
        );
        // impulse magnitude ~ Gamma pi R^2 along z, roughly conserved
        let imp = ring.impulse();
        assert!(
            (imp.z - gamma * std::f64::consts::PI * r * r).abs() < 0.15 * imp.z.abs(),
            "impulse {} vs {}",
            imp.z,
            gamma * std::f64::consts::PI * r * r
        );
    }

    #[test]
    fn test_two_rings_leapfrog() {
        // two same-sign coaxial rings leapfrog: the trailing ring shrinks?
        // no — it widens/contracts alternately; test that they pass through
        // each other at least once (mean z of ring B crosses ring A's)
        let mut sys = VortexMethod3::two_rings_leapfrog(1.0, 1.0, 0.1, 0.5, 32);
        let n = 32;
        let mean_z = |s: &VortexMethod3, lo: usize| {
            s.particles[lo..lo + n].iter().map(|p| p.pos.z).sum::<f64>() / n as f64
        };
        let mut crossed = false;
        let mut prev_gap = mean_z(&sys, n) - mean_z(&sys, 0);
        for _ in 0..600 {
            sys.step(0.02, 0.0);
            let gap = mean_z(&sys, n) - mean_z(&sys, 0);
            if gap.signum() != prev_gap.signum() {
                crossed = true;
                break;
            }
            prev_gap = gap;
        }
        assert!(crossed, "rings never passed through each other");
    }

    #[test]
    fn test_classical_vortices() {
        // Lamb-Oseen conserves circulation: 2 pi r v -> Gamma for large r
        let (gamma, nu, t) = (3.0, 1e-3, 2.0);
        let r_big = 20.0 * (4.0_f64 * nu * t).sqrt();
        assert!(
            (TWO_PI * r_big * lamb_oseen_velocity(r_big, t, gamma, nu) - gamma).abs() < 1e-10
        );
        // small r: solid-body-like, v ~ Gamma r/(8 pi nu t)
        let r_small = 1e-4;
        let v = lamb_oseen_velocity(r_small, t, gamma, nu);
        assert!((v - gamma * r_small / (8.0 * std::f64::consts::PI * nu * t)).abs() / v < 1e-3);
        // Rankine continuous at core edge
        let (rc, g) = (0.3, 1.0);
        assert!(
            (rankine_vortex(rc - 1e-12, rc, g) - rankine_vortex(rc + 1e-12, rc, g)).abs() < 1e-9
        );
        // Burgers: far field potential
        let vb = burgers_vortex(10.0, g, 1e-3, 1.0);
        assert!((vb - g / (TWO_PI * 10.0)).abs() < 1e-12);
        // Hill's vortex: poles are stagnation points, far field -> -u z
        let (u0, a) = (1.0, 1.0);
        let pole = hill_spherical_vortex(Vec3::new(0.0, 0.0, a), u0, a);
        assert!(pole.magnitude() < 1e-10, "pole speed {}", pole.magnitude());
        let far = hill_spherical_vortex(Vec3::new(0.0, 0.0, 50.0), u0, a);
        assert!((far.z + u0).abs() < 1e-4);
        // center: co-moving internal jet u_z = +3u/2
        let c = hill_spherical_vortex(Vec3::new(0.0, 0.0, 0.0), u0, a);
        assert!((c.z - 1.5 * u0).abs() < 1e-12, "center {}", c.z);
        // interior/exterior z-velocity continuous at the equator boundary
        let vi = hill_spherical_vortex(Vec3::new(a - 1e-9, 0.0, 0.0), u0, a);
        let ve = hill_spherical_vortex(Vec3::new(a + 1e-9, 0.0, 0.0), u0, a);
        assert!((vi.z - ve.z).abs() < 1e-6, "equator jump {} vs {}", vi.z, ve.z);
    }

    #[test]
    fn test_point_vortex_symplectic() {
        // pair of like-signed vortices orbits; H conserved to 1e-8 over 1e4
        // steps
        let mut pos = vec![Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0)];
        let gammas = vec![1.0, 1.0];
        let h0 = point_vortex_hamiltonian(&pos, &gammas);
        let dt = 1e-3;
        for _ in 0..10_000 {
            point_vortex_step(&mut pos, &gammas, dt);
        }
        let h1 = point_vortex_hamiltonian(&pos, &gammas);
        assert!((h1 - h0).abs() < 1e-8, "dH = {}", h1 - h0);
        // counter-rotating pair translates at Gamma/(2 pi d)
        let mut pv = VortexMethod2::point_vortex_pair(1.0, 0.4);
        let x0 = pv.particles[0].0.x;
        for _ in 0..100 {
            pv.step(1e-3, 0.0);
        }
        let u_meas = (pv.particles[0].0.x - x0) / 0.1;
        assert!(
            (u_meas - vortex_pair_velocity(1.0, 0.4)).abs() < 1e-3,
            "pair speed {u_meas} vs {}",
            vortex_pair_velocity(1.0, 0.4)
        );
    }

    #[test]
    fn test_vortex_method_2d() {
        // Lamb-Oseen blob discretization: azimuthal velocity near exact
        let vm = VortexMethod2::lamb_oseen_init(1.0, 0.5, 30);
        let total: f64 = vm.particles.iter().map(|&(_, g)| g).sum();
        // discretized circulation within 3 r_c of the total (1 - e^{-9})
        assert!((total - (1.0 - (-9.0_f64).exp())).abs() < 1e-6);
        let r_test = 1.0;
        let v = vm.velocity_at(Vec2::new(r_test, 0.0)).y;
        // steady profile equivalent: v = G/(2 pi r)(1 - exp(-r^2/rc^2))
        let exact = 1.0 / (TWO_PI * r_test) * (1.0 - (-r_test * r_test / 0.25).exp());
        assert!((v - exact).abs() < 0.05 * exact, "{v} vs {exact}");
        // merge_near conserves circulation
        let mut vm2 = vm.clone();
        let before: f64 = vm2.particles.iter().map(|&(_, g)| g).sum();
        vm2.merge_near(0.2);
        let after: f64 = vm2.particles.iter().map(|&(_, g)| g).sum();
        assert!((before - after).abs() < 1e-12);
        assert!(vm2.particles.len() < vm.particles.len());
        // KH sheet rolls up: initially near-flat sheet develops y spread
        let mut kh = VortexMethod2::kelvin_helmholtz_roll_up(64, 1.0, 1.0, 0.01);
        let spread0: f64 = kh.particles.iter().map(|&(p, _)| p.y * p.y).sum::<f64>();
        for _ in 0..200 {
            kh.step(0.005, 0.0);
        }
        let spread1: f64 = kh.particles.iter().map(|&(p, _)| p.y * p.y).sum::<f64>();
        assert!(spread1 > 4.0 * spread0, "no roll-up: {spread0} -> {spread1}");
        // to_grid produces finite velocities
        let g = vm.to_grid(16, 16);
        assert!(g.u.iter().chain(g.v.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_phenomenology() {
        assert!((kelvin_helmholtz_growth_exact(2.0, 3.0) - 3.0).abs() < 1e-15);
        assert!((vortex_shedding_frequency(0.2, 10.0, 0.1) - 20.0).abs() < 1e-12);
        // Strouhal: zero below onset, ~0.19-0.21 at high Re, continuous-ish
        assert!(strouhal_from_re(30.0) == 0.0);
        assert!((strouhal_from_re(1e5) - 0.198).abs() < 1e-3);
        assert!(strouhal_from_re(150.0) > 0.15 && strouhal_from_re(150.0) < 0.21);
        assert!(crow_instability_growth(10.0, 100.0, 1.0) > 0.0);
        // tip vortex peak swirl decays in time
        let v0 = tip_vortex_decay(1.0, 0.05, 1e-5, 0.0);
        let v1 = tip_vortex_decay(1.0, 0.05, 1e-5, 100.0);
        assert!(v1 < v0);
        assert!((helicity_density(Vec3::new(1.0, 2.0, 3.0), Vec3::new(3.0, 2.0, 1.0)) - 10.0).abs() < 1e-15);
        // vortex line of a uniform omega-z field is a straight z line
        let field = |_: Vec3| Vec3::new(0.0, 0.0, 2.0);
        let line = vortex_line_trace(&field, Vec3::new(1.0, 0.0, 0.0), 10, 0.1);
        assert!((line.last().unwrap().z - 1.0).abs() < 1e-12);
        assert!((line.last().unwrap().x - 1.0).abs() < 1e-12);
    }
}
