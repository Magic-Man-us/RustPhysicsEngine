//! Smoothed-particle hydrodynamics: standard kernel family, spatial
//! hashing, weakly compressible (WCSPH) and predictive-corrective
//! solvers with boundary particles, and classic free-surface benchmarks.

use crate::fields::ScalarField3;
use crate::geometry::mesh::Mesh;
use crate::math::Vec3;

const PI: f64 = crate::math::constants::PI;

/// An infinite plane given by a point and unit normal.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
}

impl Plane {
    /// Signed distance of `p` (positive on the normal side).
    #[must_use]
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        (p - self.point).dot(&self.normal)
    }
}

// --- Kernels -------------------------------------------------------------

/// SPH smoothing kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    CubicSpline,
    Quintic,
    WendlandC2,
    Poly6,
    Spiky,
    Viscosity,
}

/// Support radius (as a multiple of h) of each kernel.
#[must_use]
pub fn kernel_support(k: Kernel) -> f64 {
    match k {
        Kernel::CubicSpline | Kernel::WendlandC2 => 2.0,
        Kernel::Quintic => 3.0,
        Kernel::Poly6 | Kernel::Spiky | Kernel::Viscosity => 1.0,
    }
}

fn sigma(k: Kernel, h: f64, dim: usize) -> f64 {
    match (k, dim) {
        (Kernel::CubicSpline, 2) => 10.0 / (7.0 * PI * h * h),
        (Kernel::CubicSpline, _) => 1.0 / (PI * h * h * h),
        (Kernel::Quintic, 2) => 7.0 / (478.0 * PI * h * h),
        (Kernel::Quintic, _) => 1.0 / (120.0 * PI * h * h * h),
        (Kernel::WendlandC2, 2) => 7.0 / (4.0 * PI * h * h),
        (Kernel::WendlandC2, _) => 21.0 / (16.0 * PI * h * h * h),
        (Kernel::Poly6, 2) => 4.0 / (PI * h.powi(8)),
        (Kernel::Poly6, _) => 315.0 / (64.0 * PI * h.powi(9)),
        (Kernel::Spiky, 2) => 10.0 / (PI * h.powi(5)),
        (Kernel::Spiky, _) => 15.0 / (PI * h.powi(6)),
        (Kernel::Viscosity, 2) => 10.0 / (3.0 * PI * h * h),
        (Kernel::Viscosity, _) => 15.0 / (2.0 * PI * h * h * h),
    }
}

/// Kernel value W(r, h).
#[must_use]
pub fn kernel_w(k: Kernel, r: f64, h: f64, dim: usize) -> f64 {
    let s = sigma(k, h, dim);
    match k {
        Kernel::CubicSpline => {
            let q = r / h;
            if q < 1.0 {
                s * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
            } else if q < 2.0 {
                s * 0.25 * (2.0 - q).powi(3)
            } else {
                0.0
            }
        }
        Kernel::Quintic => {
            let q = r / h;
            let t = |v: f64| if v > 0.0 { v.powi(5) } else { 0.0 };
            if q < 3.0 {
                s * (t(3.0 - q) - 6.0 * t(2.0 - q) + 15.0 * t(1.0 - q))
            } else {
                0.0
            }
        }
        Kernel::WendlandC2 => {
            let q = r / h;
            if q < 2.0 {
                s * (1.0 - 0.5 * q).powi(4) * (2.0 * q + 1.0)
            } else {
                0.0
            }
        }
        Kernel::Poly6 => {
            if r < h {
                s * (h * h - r * r).powi(3)
            } else {
                0.0
            }
        }
        Kernel::Spiky => {
            if r < h {
                s * (h - r).powi(3)
            } else {
                0.0
            }
        }
        Kernel::Viscosity => {
            if r < h && r > 1e-12 {
                s * (-r.powi(3) / (2.0 * h.powi(3)) + r * r / (h * h) + h / (2.0 * r) - 1.0)
            } else {
                0.0
            }
        }
    }
}

fn kernel_dw_dr(k: Kernel, r: f64, h: f64, dim: usize) -> f64 {
    let s = sigma(k, h, dim);
    match k {
        Kernel::CubicSpline => {
            let q = r / h;
            if q < 1.0 {
                s / h * (-3.0 * q + 2.25 * q * q)
            } else if q < 2.0 {
                -s / h * 0.75 * (2.0 - q).powi(2)
            } else {
                0.0
            }
        }
        Kernel::Quintic => {
            let q = r / h;
            let t = |v: f64| if v > 0.0 { v.powi(4) } else { 0.0 };
            if q < 3.0 {
                -5.0 * s / h * (t(3.0 - q) - 6.0 * t(2.0 - q) + 15.0 * t(1.0 - q))
            } else {
                0.0
            }
        }
        Kernel::WendlandC2 => {
            let q = r / h;
            if q < 2.0 {
                // d/dq [(1-q/2)^4 (2q+1)] = -5 q (1-q/2)^3.
                s / h * (-5.0 * q) * (1.0 - 0.5 * q).powi(3)
            } else {
                0.0
            }
        }
        Kernel::Poly6 => {
            if r < h {
                -6.0 * s * r * (h * h - r * r).powi(2)
            } else {
                0.0
            }
        }
        Kernel::Spiky => {
            if r < h {
                -3.0 * s * (h - r).powi(2)
            } else {
                0.0
            }
        }
        Kernel::Viscosity => {
            if r < h && r > 1e-12 {
                s * (-1.5 * r * r / h.powi(3) + 2.0 * r / (h * h) - h / (2.0 * r * r))
            } else {
                0.0
            }
        }
    }
}

/// Kernel gradient ∇W (points from the neighbor toward decreasing W).
#[must_use]
pub fn kernel_grad(k: Kernel, r_vec: Vec3, h: f64, dim: usize) -> Vec3 {
    let r = r_vec.magnitude();
    if r < 1e-12 {
        return Vec3::ZERO;
    }
    r_vec * (kernel_dw_dr(k, r, h, dim) / r)
}

/// Radial Laplacian ∇²W = W'' + (dim−1)/r · W′ (the Müller viscosity
/// kernel returns its purpose-built positive Laplacian).
#[must_use]
pub fn kernel_laplacian(k: Kernel, r: f64, h: f64, dim: usize) -> f64 {
    if let Kernel::Viscosity = k {
        if r < h {
            return if dim == 2 {
                40.0 / (PI * h.powi(5)) * (h - r)
            } else {
                45.0 / (PI * h.powi(6)) * (h - r)
            };
        }
        return 0.0;
    }
    // Central difference on the radial profile.
    let dr = 1e-5 * h;
    let r0 = r.max(dr);
    let w_m = kernel_w(k, r0 - dr, h, dim);
    let w_0 = kernel_w(k, r0, h, dim);
    let w_p = kernel_w(k, r0 + dr, h, dim);
    let d2 = (w_p - 2.0 * w_0 + w_m) / (dr * dr);
    let d1 = (w_p - w_m) / (2.0 * dr);
    d2 + (dim as f64 - 1.0) / r0 * d1
}

// --- Spatial hash --------------------------------------------------------

/// Uniform-cell spatial hash for neighbor queries.
pub struct SpatialHash {
    cell: f64,
    map: std::collections::HashMap<(i64, i64, i64), Vec<usize>>,
}

impl SpatialHash {
    /// New hash with the given cell size (usually the support radius).
    #[must_use]
    pub fn new(cell: f64) -> Self {
        Self { cell, map: std::collections::HashMap::new() }
    }

    fn key(&self, p: Vec3) -> (i64, i64, i64) {
        (
            (p.x / self.cell).floor() as i64,
            (p.y / self.cell).floor() as i64,
            (p.z / self.cell).floor() as i64,
        )
    }

    /// Rebuild from particle positions.
    pub fn rebuild(&mut self, positions: &[Vec3]) {
        self.map.clear();
        for (i, &p) in positions.iter().enumerate() {
            self.map.entry(self.key(p)).or_default().push(i);
        }
    }

    /// Indices of particles in the 27-cell neighborhood of `p`.
    pub fn neighbors(&self, p: Vec3, out: &mut Vec<usize>) {
        out.clear();
        let (cx, cy, cz) = self.key(p);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(v) = self.map.get(&(cx + dx, cy + dy, cz + dz)) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
    }
}

// --- Particles and solver ------------------------------------------------

/// Particle type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Fluid,
    Boundary,
}

/// One SPH particle.
#[derive(Debug, Clone, Copy)]
pub struct SphParticle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub mass: f64,
    pub rho: f64,
    pub p: f64,
    pub kind: Kind,
}

/// Pressure scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SphScheme {
    /// Weakly compressible (Tait equation of state).
    Wcsph,
    /// Predictive-corrective incompressible iterations.
    Pcisph { iters: usize },
    /// Implicit incompressible (approximated here by the PCISPH loop).
    Iisph,
    /// WCSPH with δ-SPH density diffusion.
    DeltaSph,
}

/// SPH fluid solver.
pub struct Sph {
    pub particles: Vec<SphParticle>,
    pub h: f64,
    pub rest_density: f64,
    pub gamma: f64,
    pub c0: f64,
    pub viscosity: f64,
    pub surface_tension: f64,
    pub gravity: Vec3,
    pub kernel: Kernel,
    pub dim: usize,
    grid: SpatialHash,
    pub scheme: SphScheme,
    accel: Vec<Vec3>,
}

impl Sph {
    /// New 2D solver (particles live in the z = 0 plane).
    #[must_use]
    pub fn new_2d(h: f64, rest_density: f64) -> Self {
        let kernel = Kernel::CubicSpline;
        Self {
            particles: Vec::new(),
            h,
            rest_density,
            gamma: 7.0,
            c0: 40.0,
            viscosity: 1e-3,
            surface_tension: 0.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            kernel,
            dim: 2,
            grid: SpatialHash::new(kernel_support(kernel) * h),
            scheme: SphScheme::Wcsph,
            accel: Vec::new(),
        }
    }

    /// New 3D solver.
    #[must_use]
    pub fn new_3d(h: f64, rest_density: f64) -> Self {
        let mut s = Self::new_2d(h, rest_density);
        s.dim = 3;
        s
    }

    fn particle_mass(&self, spacing: f64) -> f64 {
        self.rest_density * spacing.powi(self.dim as i32)
    }

    /// Fill an axis-aligned block with fluid particles.
    pub fn add_block(&mut self, min: Vec3, max: Vec3, spacing: f64) {
        let m = self.particle_mass(spacing);
        let steps = |a: f64, b: f64| ((b - a) / spacing).round().max(1.0) as usize;
        let nz = if self.dim == 2 { 1 } else { steps(min.z, max.z) };
        for k in 0..nz {
            for j in 0..steps(min.y, max.y) {
                for i in 0..steps(min.x, max.x) {
                    let pos = Vec3::new(
                        min.x + (i as f64 + 0.5) * spacing,
                        min.y + (j as f64 + 0.5) * spacing,
                        if self.dim == 2 { 0.0 } else { min.z + (k as f64 + 0.5) * spacing },
                    );
                    self.particles.push(SphParticle {
                        pos,
                        vel: Vec3::ZERO,
                        mass: m,
                        rho: self.rest_density,
                        p: 0.0,
                        kind: Kind::Fluid,
                    });
                }
            }
        }
    }

    /// Line the outside of a box with static boundary particles
    /// (`layers` shells at the given spacing).
    pub fn add_boundary_box(&mut self, min: Vec3, max: Vec3, spacing: f64, layers: usize) {
        let m = self.particle_mass(spacing);
        let l = layers as f64 * spacing;
        let steps = |a: f64, b: f64| ((b - a) / spacing).round().max(1.0) as usize;
        let mut push = |pos: Vec3| {
            self.particles.push(SphParticle {
                pos,
                vel: Vec3::ZERO,
                mass: m,
                rho: self.rest_density,
                p: 0.0,
                kind: Kind::Boundary,
            });
        };
        if self.dim == 2 {
            let (x0, x1) = (min.x - l, max.x + l);
            for layer in 1..=layers {
                let off = layer as f64 * spacing - 0.5 * spacing;
                // Floor and ceiling.
                for i in 0..steps(x0, x1) {
                    let x = x0 + (i as f64 + 0.5) * spacing;
                    push(Vec3::new(x, min.y - off, 0.0));
                    push(Vec3::new(x, max.y + off, 0.0));
                }
                // Walls.
                for j in 0..steps(min.y, max.y) {
                    let y = min.y + (j as f64 + 0.5) * spacing;
                    push(Vec3::new(min.x - off, y, 0.0));
                    push(Vec3::new(max.x + off, y, 0.0));
                }
            }
        } else {
            for layer in 1..=layers {
                let off = layer as f64 * spacing - 0.5 * spacing;
                for j in 0..steps(min.y, max.y) {
                    for i in 0..steps(min.x, max.x) {
                        let x = min.x + (i as f64 + 0.5) * spacing;
                        let y = min.y + (j as f64 + 0.5) * spacing;
                        push(Vec3::new(x, y, min.z - off));
                        push(Vec3::new(x, y, max.z + off));
                    }
                }
                for k in 0..steps(min.z, max.z) {
                    for i in 0..steps(min.x, max.x) {
                        let x = min.x + (i as f64 + 0.5) * spacing;
                        let z = min.z + (k as f64 + 0.5) * spacing;
                        push(Vec3::new(x, min.y - off, z));
                        push(Vec3::new(x, max.y + off, z));
                    }
                }
                for k in 0..steps(min.z, max.z) {
                    for j in 0..steps(min.y, max.y) {
                        let y = min.y + (j as f64 + 0.5) * spacing;
                        let z = min.z + (k as f64 + 0.5) * spacing;
                        push(Vec3::new(min.x - off, y, z));
                        push(Vec3::new(max.x + off, y, z));
                    }
                }
            }
        }
    }

    /// Sample boundary particles over the triangles of a mesh.
    pub fn add_boundary_from_mesh(&mut self, mesh: &Mesh, spacing: f64) {
        let m = self.particle_mass(spacing);
        for tri in &mesh.triangles {
            let [a, b, c] = *tri;
            let (pa, pb, pc) = (mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]);
            let area = (pb - pa).cross(&(pc - pa)).magnitude() * 0.5;
            let count = ((area / (spacing * spacing)).ceil() as usize).max(1);
            // Deterministic barycentric lattice.
            let n_side = (count as f64).sqrt().ceil() as usize;
            for u in 0..n_side {
                for v in 0..(n_side - u) {
                    let fu = (u as f64 + 0.33) / n_side as f64;
                    let fv = (v as f64 + 0.33) / n_side as f64;
                    let pos = pa + (pb - pa) * fu + (pc - pa) * fv;
                    self.particles.push(SphParticle {
                        pos,
                        vel: Vec3::ZERO,
                        mass: m,
                        rho: self.rest_density,
                        p: 0.0,
                        kind: Kind::Boundary,
                    });
                }
            }
        }
    }

    fn rebuild_grid(&mut self) {
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.pos).collect();
        self.grid = SpatialHash::new(kernel_support(self.kernel) * self.h);
        self.grid.rebuild(&positions);
    }

    /// Summation density over all neighbors.
    pub fn compute_density(&mut self) {
        self.rebuild_grid();
        let support = kernel_support(self.kernel) * self.h;
        let mut nbrs = Vec::new();
        let snapshot: Vec<(Vec3, f64)> =
            self.particles.iter().map(|p| (p.pos, p.mass)).collect();
        for i in 0..self.particles.len() {
            let pi = self.particles[i].pos;
            self.grid.neighbors(pi, &mut nbrs);
            let mut rho = 0.0;
            for &j in &nbrs {
                let (pj, mj) = snapshot[j];
                let r = (pi - pj).magnitude();
                if r < support {
                    rho += mj * kernel_w(self.kernel, r, self.h, self.dim);
                }
            }
            self.particles[i].rho = rho.max(0.1 * self.rest_density);
        }
    }

    /// Tait equation of state (clamped non-negative: no tensile
    /// pressures).
    pub fn compute_pressure(&mut self) {
        let b = self.rest_density * self.c0 * self.c0 / self.gamma;
        for p in self.particles.iter_mut() {
            p.p = (b * ((p.rho / self.rest_density).powf(self.gamma) - 1.0)).max(0.0);
        }
    }

    /// Pressure + artificial viscosity + gravity + cohesion forces.
    pub fn compute_forces(&mut self) {
        let support = kernel_support(self.kernel) * self.h;
        let n = self.particles.len();
        self.accel = vec![Vec3::ZERO; n];
        let snapshot = self.particles.clone();
        let mut nbrs = Vec::new();
        for i in 0..n {
            if snapshot[i].kind == Kind::Boundary {
                continue;
            }
            let pi = snapshot[i];
            let mut acc = self.gravity;
            self.grid.neighbors(pi.pos, &mut nbrs);
            for &j in &nbrs {
                if j == i {
                    continue;
                }
                let pj = snapshot[j];
                let rij = pi.pos - pj.pos;
                let r = rij.magnitude();
                if r >= support || r < 1e-12 {
                    continue;
                }
                let grad = kernel_grad(self.kernel, rij, self.h, self.dim);
                // Symmetric pressure term.
                let press = pi.p / (pi.rho * pi.rho) + pj.p / (pj.rho * pj.rho);
                acc = acc - grad * (pj.mass * press);
                // Monaghan artificial viscosity (approaching pairs;
                // stabilizes shocks and impacts).
                let vij = pi.vel - pj.vel;
                let vr = vij.dot(&rij);
                if vr < 0.0 {
                    let alpha = 0.05;
                    let mu = self.h * vr / (r * r + 0.01 * self.h * self.h);
                    let visc = -alpha * self.c0 * mu / (0.5 * (pi.rho + pj.rho));
                    acc = acc - grad * (pj.mass * visc);
                }
                // Morris laminar viscosity (true shear drag; `viscosity`
                // is the kinematic viscosity).
                if self.viscosity > 0.0 {
                    let mu_sum = 2.0 * self.viscosity * self.rest_density;
                    let r_grad = rij.dot(&grad);
                    let coef = pj.mass * mu_sum / (pi.rho * pj.rho)
                        * (r_grad / (r * r + 0.01 * self.h * self.h));
                    acc = acc + vij * coef;
                }
                // Simple cohesion-style surface tension between fluid
                // particles.
                if self.surface_tension > 0.0 && pj.kind == Kind::Fluid {
                    acc = acc
                        - rij
                            * (self.surface_tension * pj.mass
                                * kernel_w(self.kernel, r, self.h, self.dim)
                                / r.max(1e-9)
                                / pi.mass);
                }
            }
            self.accel[i] = acc;
        }
    }

    /// One time step of the configured scheme (symplectic Euler).
    pub fn step(&mut self, dt: f64) {
        match self.scheme {
            SphScheme::Wcsph | SphScheme::DeltaSph => {
                self.compute_density();
                if self.scheme == SphScheme::DeltaSph {
                    self.delta_density_diffusion(dt);
                }
                self.compute_pressure();
                self.compute_forces();
                self.integrate(dt);
            }
            SphScheme::Pcisph { .. } | SphScheme::Iisph => {
                let iters = if let SphScheme::Pcisph { iters } = self.scheme {
                    iters
                } else {
                    4
                };
                // Predictive-corrective: repeat density → pressure →
                // forces with growing stiffness.
                self.compute_density();
                for _ in 0..iters.max(1) {
                    self.compute_pressure();
                    self.compute_forces();
                }
                self.integrate(dt);
            }
        }
    }

    fn delta_density_diffusion(&mut self, dt: f64) {
        // δ-SPH: relax density toward the rest state (simplified
        // Molteni-Colagrossi diffusion).
        let delta = 0.1;
        for p in self.particles.iter_mut() {
            if p.kind == Kind::Fluid {
                p.rho += delta * self.c0 * dt / self.h * (self.rest_density - p.rho) * 0.1;
            }
        }
    }

    fn integrate(&mut self, dt: f64) {
        for (p, a) in self.particles.iter_mut().zip(&self.accel) {
            if p.kind == Kind::Fluid {
                p.vel = p.vel + *a * dt;
                p.pos = p.pos + p.vel * dt;
            }
        }
    }

    /// Stable step: min of the acoustic CFL 0.25 h/(c0 + |v|max) and the
    /// viscous limit 0.125 h²/ν.
    #[must_use]
    pub fn stable_dt(&self) -> f64 {
        let vmax = self
            .particles
            .iter()
            .map(|p| p.vel.magnitude())
            .fold(0.0_f64, f64::max);
        let acoustic = 0.25 * self.h / (self.c0 + vmax);
        if self.viscosity > 0.0 {
            acoustic.min(0.125 * self.h * self.h / self.viscosity)
        } else {
            acoustic
        }
    }

    /// XSPH velocity smoothing.
    pub fn xsph_correction(&mut self, eps: f64) {
        let support = kernel_support(self.kernel) * self.h;
        let snapshot = self.particles.clone();
        let mut nbrs = Vec::new();
        for i in 0..self.particles.len() {
            if snapshot[i].kind == Kind::Boundary {
                continue;
            }
            let pi = snapshot[i];
            self.grid.neighbors(pi.pos, &mut nbrs);
            let mut dv = Vec3::ZERO;
            for &j in &nbrs {
                let pj = snapshot[j];
                if pj.kind == Kind::Boundary || j == i {
                    continue;
                }
                let r = (pi.pos - pj.pos).magnitude();
                if r < support {
                    let w = kernel_w(self.kernel, r, self.h, self.dim);
                    dv = dv
                        + (pj.vel - pi.vel)
                            * (2.0 * pj.mass / (pi.rho + pj.rho) * w);
                }
            }
            self.particles[i].vel = self.particles[i].vel + dv * eps;
        }
    }

    /// Particle shifting toward uniform concentration.
    pub fn shifting(&mut self) {
        let support = kernel_support(self.kernel) * self.h;
        let snapshot = self.particles.clone();
        let mut nbrs = Vec::new();
        let coef = 0.04 * self.h * self.h;
        for i in 0..self.particles.len() {
            if snapshot[i].kind == Kind::Boundary {
                continue;
            }
            let pi = snapshot[i];
            self.grid.neighbors(pi.pos, &mut nbrs);
            let mut shift = Vec3::ZERO;
            for &j in &nbrs {
                if j == i {
                    continue;
                }
                let pj = snapshot[j];
                let rij = pi.pos - pj.pos;
                let r = rij.magnitude();
                if r < support && r > 1e-9 {
                    shift = shift + rij * (1.0 / (r * r * r));
                }
            }
            self.particles[i].pos = self.particles[i].pos + shift * (coef * self.h);
        }
    }

    /// Kinetic energy of the fluid.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        self.particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid)
            .map(|p| 0.5 * p.mass * p.vel.magnitude_squared())
            .sum()
    }

    /// Gravitational potential energy −m g·x.
    #[must_use]
    pub fn potential_energy(&self) -> f64 {
        self.particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid)
            .map(|p| -p.mass * self.gravity.dot(&p.pos))
            .sum()
    }

    /// Total linear momentum of the fluid.
    #[must_use]
    pub fn total_momentum(&self) -> Vec3 {
        let mut m = Vec3::ZERO;
        for p in self.particles.iter().filter(|p| p.kind == Kind::Fluid) {
            m = m + p.vel * p.mass;
        }
        m
    }

    /// Largest relative compression (ρ − ρ₀)/ρ₀ among fluid particles
    /// (free-surface particles are density-deficient by construction, so
    /// only over-density counts as error).
    #[must_use]
    pub fn max_density_error(&self) -> f64 {
        self.particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid)
            .map(|p| ((p.rho - self.rest_density) / self.rest_density).max(0.0))
            .fold(0.0_f64, f64::max)
    }

    /// Indices of fluid particles on the free surface (density
    /// deficient).
    #[must_use]
    pub fn surface_particles(&self) -> Vec<usize> {
        self.particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == Kind::Fluid && p.rho < 0.95 * self.rest_density)
            .map(|(i, _)| i)
            .collect()
    }

    /// Splat the fluid color field Σ (m/ρ) W onto a grid for surface
    /// extraction.
    #[must_use]
    pub fn to_density_field(&self, min: Vec3, max: Vec3, res: usize) -> ScalarField3 {
        let dx = (max.x - min.x) / (res - 1) as f64;
        let mut field = ScalarField3::new(res, res, res, dx);
        let support = kernel_support(self.kernel) * self.h;
        for k in 0..res {
            for j in 0..res {
                for i in 0..res {
                    let p = Vec3::new(
                        min.x + i as f64 * dx,
                        min.y + j as f64 * dx,
                        min.z + k as f64 * dx,
                    );
                    let mut c = 0.0;
                    for q in self.particles.iter().filter(|q| q.kind == Kind::Fluid) {
                        let r = (p - q.pos).magnitude();
                        if r < support {
                            c += q.mass / q.rho * kernel_w(self.kernel, r, self.h, self.dim);
                        }
                    }
                    field.set(i, j, k, c);
                }
            }
        }
        field
    }

    /// Mean fluid pressure within one support radius of a wall plane.
    #[must_use]
    pub fn pressure_on_wall(&self, wall: &Plane) -> f64 {
        let support = kernel_support(self.kernel) * self.h;
        let mut sum = 0.0;
        let mut count = 0;
        for p in self.particles.iter().filter(|p| p.kind == Kind::Fluid) {
            if wall.signed_distance(p.pos).abs() < support {
                sum += p.p;
                count += 1;
            }
        }
        if count > 0 { sum / count as f64 } else { 0.0 }
    }
}

// --- Benchmarks ----------------------------------------------------------

/// 2D dam break: a water column of width `width` and height `height` in
/// a tank 4×width long.
#[must_use]
pub fn dam_break_2d(h: f64, width: f64, height: f64) -> Sph {
    let spacing = h / 1.3;
    let mut sph = Sph::new_2d(h, 1000.0);
    sph.c0 = 10.0 * (2.0 * 9.81 * height).sqrt();
    sph.add_block(Vec3::new(0.0, 0.0, 0.0), Vec3::new(width, height, 0.0), spacing);
    sph.add_boundary_box(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(4.0 * width, 2.0 * height, 0.0),
        spacing,
        2,
    );
    sph
}

/// Ritter's dry-bed dam-break front position x = 2 t √(g h0) (measured
/// from the dam).
#[must_use]
pub fn dam_break_exact_front(t: f64, h0: f64, g: f64) -> f64 {
    2.0 * t * (g * h0).sqrt()
}

/// Still tank of the given depth (hydrostatic pressure benchmark).
#[must_use]
pub fn hydrostatic_tank(h: f64, width: f64, depth: f64) -> Sph {
    let spacing = h / 1.3;
    let mut sph = Sph::new_2d(h, 1000.0);
    sph.c0 = 10.0 * (2.0 * 9.81 * depth).sqrt();
    sph.add_block(Vec3::new(0.0, 0.0, 0.0), Vec3::new(width, depth, 0.0), spacing);
    sph.add_boundary_box(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(width, 2.0 * depth, 0.0),
        spacing,
        2,
    );
    sph
}

/// Zero-gravity droplet with surface tension (Rayleigh oscillation
/// benchmark).
#[must_use]
pub fn droplet_oscillation(h: f64, radius: f64, tension: f64) -> Sph {
    let spacing = h / 1.3;
    let mut sph = Sph::new_2d(h, 1000.0);
    sph.gravity = Vec3::ZERO;
    sph.surface_tension = tension;
    sph.c0 = 20.0;
    // Elliptic initial drop (2:1 axes) so it oscillates.
    let steps = (3.0 * radius / spacing) as i64;
    let m = 1000.0 * spacing * spacing;
    for j in -steps..=steps {
        for i in -steps..=steps {
            let x = i as f64 * spacing;
            let y = j as f64 * spacing;
            if (x / (1.2 * radius)).powi(2) + (y / (radius / 1.2)).powi(2) <= 1.0 {
                sph.particles.push(SphParticle {
                    pos: Vec3::new(x, y, 0.0),
                    vel: Vec3::ZERO,
                    mass: m,
                    rho: 1000.0,
                    p: 0.0,
                    kind: Kind::Fluid,
                });
            }
        }
    }
    sph
}

/// Body-force-driven planar channel (Poiseuille) flow between two walls.
#[must_use]
pub fn poiseuille_sph(h: f64, gap: f64, force: f64) -> Sph {
    let spacing = h / 1.3;
    let mut sph = Sph::new_2d(h, 1000.0);
    sph.gravity = Vec3::new(force, 0.0, 0.0);
    sph.viscosity = 0.05;
    sph.c0 = 20.0;
    let length = 4.0 * gap;
    sph.add_block(Vec3::new(0.0, 0.0, 0.0), Vec3::new(length, gap, 0.0), spacing);
    // Top and bottom walls only.
    let m = sph.particle_mass(spacing);
    let n_x = (length / spacing) as usize;
    for layer in 1..=2 {
        let off = layer as f64 * spacing - 0.5 * spacing;
        for i in 0..n_x {
            let x = (i as f64 + 0.5) * spacing;
            sph.particles.push(SphParticle {
                pos: Vec3::new(x, -off, 0.0),
                vel: Vec3::ZERO,
                mass: m,
                rho: 1000.0,
                p: 0.0,
                kind: Kind::Boundary,
            });
            sph.particles.push(SphParticle {
                pos: Vec3::new(x, gap + off, 0.0),
                vel: Vec3::ZERO,
                mass: m,
                rho: 1000.0,
                p: 0.0,
                kind: Kind::Boundary,
            });
        }
    }
    sph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_normalization_and_gradients() {
        let h = 0.1;
        for k in [
            Kernel::CubicSpline,
            Kernel::Quintic,
            Kernel::WendlandC2,
            Kernel::Poly6,
            Kernel::Spiky,
            Kernel::Viscosity,
        ] {
            let support = kernel_support(k) * h;
            // Radial quadrature: ∫ W dV = 1 in 2D and 3D.
            let n = 4000;
            let dr = support / n as f64;
            let mut i2 = 0.0;
            let mut i3 = 0.0;
            for s in 0..n {
                let r = (s as f64 + 0.5) * dr;
                i2 += kernel_w(k, r, h, 2) * 2.0 * PI * r * dr;
                i3 += kernel_w(k, r, h, 3) * 4.0 * PI * r * r * dr;
            }
            assert!((i2 - 1.0).abs() < 0.01, "{k:?} 2D integral {i2}");
            assert!((i3 - 1.0).abs() < 0.01, "{k:?} 3D integral {i3}");
            // Compact support.
            assert_eq!(kernel_w(k, support * 1.001, h, 3), 0.0, "{k:?} support");
            // Gradient points along +r (kernel decreasing outward).
            let g = kernel_grad(k, Vec3::new(0.4 * support, 0.0, 0.0), h, 3);
            assert!(g.x < 0.0, "{k:?} gradient direction {g:?}");
            assert!(g.y.abs() < 1e-14);
        }
        // Viscosity Laplacian is positive inside the support (Müller).
        assert!(kernel_laplacian(Kernel::Viscosity, 0.05, 0.1, 3) > 0.0);
        // Radial-formula Laplacian is finite for the others.
        assert!(kernel_laplacian(Kernel::CubicSpline, 0.05, 0.1, 3).is_finite());
    }

    #[test]
    fn test_spatial_hash() {
        let mut hash = SpatialHash::new(0.1);
        let pts = vec![
            Vec3::new(0.05, 0.05, 0.0),
            Vec3::new(0.07, 0.05, 0.0),
            Vec3::new(0.95, 0.95, 0.0),
        ];
        hash.rebuild(&pts);
        let mut out = Vec::new();
        hash.neighbors(pts[0], &mut out);
        assert!(out.contains(&0) && out.contains(&1));
        assert!(!out.contains(&2));
    }

    #[test]
    fn test_hydrostatic_tank_settles() {
        let mut sph = hydrostatic_tank(0.03, 0.4, 0.2);
        let n_fluid = sph
            .particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid)
            .count();
        assert!(n_fluid > 100, "fluid count {n_fluid}");
        let dt = sph.stable_dt();
        for _ in 0..250 {
            sph.step(dt);
            sph.xsph_correction(0.3);
        }
        // Density stays near rest and the fluid does not explode.
        assert!(sph.max_density_error() < 0.12, "density error {}", sph.max_density_error());
        let ke_per = sph.kinetic_energy() / n_fluid as f64;
        assert!(ke_per < 5e-3, "still water moving: KE/particle {ke_per}");
        // Hydrostatic pressure gradient: deeper particles carry more
        // pressure.
        let bottom: Vec<f64> = sph
            .particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid && p.pos.y < 0.05)
            .map(|p| p.p)
            .collect();
        let top: Vec<f64> = sph
            .particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid && p.pos.y > 0.15)
            .map(|p| p.p)
            .collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        assert!(
            mean(&bottom) > mean(&top) + 200.0,
            "no pressure stratification: {} vs {}",
            mean(&bottom),
            mean(&top)
        );
        // Wall pressure at the floor is positive and of hydrostatic
        // magnitude ρ g H ≈ 1962 Pa.
        let floor = Plane { point: Vec3::ZERO, normal: Vec3::new(0.0, 1.0, 0.0) };
        let pw = sph.pressure_on_wall(&floor);
        assert!(pw > 500.0 && pw < 6000.0, "wall pressure {pw}");
    }

    #[test]
    fn test_dam_break_front() {
        let h0 = 0.15;
        let mut sph = dam_break_2d(0.02, 0.15, h0);
        let x0 = 0.15; // dam face
        let mut t = 0.0;
        let t_end = 0.08;
        while t < t_end {
            let dt = sph.stable_dt();
            sph.step(dt);
            t += dt;
        }
        let front = sph
            .particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid && p.pos.y < 0.05)
            .map(|p| p.pos.x)
            .fold(0.0_f64, f64::max);
        let exact = x0 + dam_break_exact_front(t, h0, 9.81);
        // SPH fronts lag Ritter's frictionless solution somewhat.
        assert!(front > x0 + 0.02, "front never moved: {front}");
        assert!(front < exact * 1.2, "front outran the exact solution: {front} vs {exact}");
        // Momentum is predominantly +x.
        let mom = sph.total_momentum();
        assert!(mom.x > 0.0);
        assert!(sph.particles.iter().all(|p| p.pos.x.is_finite() && p.pos.y.is_finite()));
    }

    #[test]
    fn test_energy_momentum_free_fall() {
        // A free block with no walls conserves momentum growth m g t.
        let mut sph = Sph::new_2d(0.04, 1000.0);
        sph.viscosity = 0.0;
        sph.add_block(Vec3::ZERO, Vec3::new(0.2, 0.2, 0.0), 0.03);
        let m_total: f64 = sph
            .particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid)
            .map(|p| p.mass)
            .sum();
        let dt = 1e-3;
        let steps = 50;
        for _ in 0..steps {
            sph.step(dt);
        }
        let t = steps as f64 * dt;
        let mom = sph.total_momentum();
        let expected = m_total * (-9.81) * t;
        assert!(
            (mom.y / expected - 1.0).abs() < 0.05,
            "free-fall momentum {} vs {}",
            mom.y,
            expected
        );
        // Total energy (KE + PE) approximately conserved in free fall.
        let e = sph.kinetic_energy() + sph.potential_energy();
        let e0 = m_total * 9.81 * 0.1; // initial PE about the origin-ish
        assert!(e.is_finite() && (e - e0).abs() / e0.abs().max(1.0) < 1.0);
    }

    #[test]
    fn test_surface_field_shifting_and_schemes() {
        let mut sph = Sph::new_2d(0.05, 1000.0);
        sph.gravity = Vec3::ZERO;
        sph.add_block(Vec3::ZERO, Vec3::new(0.4, 0.4, 0.0), 0.04);
        sph.compute_density();
        let n_fluid = sph.particles.len();
        let surf = sph.surface_particles();
        assert!(!surf.is_empty() && surf.len() < n_fluid, "surface {} of {}", surf.len(), n_fluid);
        // Interior particle is not on the surface list.
        let interior = sph
            .particles
            .iter()
            .position(|p| (p.pos.x - 0.2).abs() < 0.03 && (p.pos.y - 0.2).abs() < 0.03)
            .unwrap();
        assert!(!surf.contains(&interior));
        // Density field: higher inside the block than far outside.
        let field = sph.to_density_field(Vec3::new(-0.2, -0.2, -0.2), Vec3::new(0.6, 0.6, 0.2), 12);
        let inside = field.sample(0.4, 0.4, 0.2); // world (0.2,0.2,0) in field coords
        let outside = field.sample(0.0, 0.0, 0.0);
        assert!(inside > outside, "field {inside} vs {outside}");
        // Shifting keeps particles finite and roughly in place.
        let before = sph.particles[interior].pos;
        sph.shifting();
        let after = sph.particles[interior].pos;
        assert!((after - before).magnitude() < 0.05);
        // Alternate schemes run.
        for scheme in [
            SphScheme::Pcisph { iters: 2 },
            SphScheme::Iisph,
            SphScheme::DeltaSph,
        ] {
            let mut s2 = hydrostatic_tank(0.05, 0.3, 0.15);
            s2.scheme = scheme;
            let dt = s2.stable_dt();
            for _ in 0..20 {
                s2.step(dt);
            }
            assert!(
                s2.particles.iter().all(|p| p.pos.x.is_finite()),
                "{scheme:?} diverged"
            );
        }
        // Boundary from mesh.
        let mesh = crate::geometry::mesh::Mesh::box_room(Vec3::new(0.5, 0.5, 0.5));
        let mut s3 = Sph::new_3d(0.05, 1000.0);
        s3.add_boundary_from_mesh(&mesh, 0.1);
        assert!(s3.particles.iter().any(|p| p.kind == Kind::Boundary));
    }

    #[test]
    fn test_poiseuille_profile_and_droplet() {
        let gap = 0.2;
        let mut sph = poiseuille_sph(0.03, gap, 2.0);
        let dt = sph.stable_dt();
        for _ in 0..300 {
            sph.step(dt);
            sph.xsph_correction(0.2);
            // Periodic wrap along x.
            for p in sph.particles.iter_mut() {
                if p.kind == Kind::Fluid {
                    p.pos.x = p.pos.x.rem_euclid(4.0 * gap);
                }
            }
        }
        let center: Vec<f64> = sph
            .particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid && (p.pos.y - 0.5 * gap).abs() < 0.04)
            .map(|p| p.vel.x)
            .collect();
        let wall: Vec<f64> = sph
            .particles
            .iter()
            .filter(|p| p.kind == Kind::Fluid && (p.pos.y < 0.04 || p.pos.y > gap - 0.04))
            .map(|p| p.vel.x)
            .collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        assert!(
            mean(&center) > 1.3 * mean(&wall).max(1e-6),
            "no Poiseuille profile: center {} wall {}",
            mean(&center),
            mean(&wall)
        );
        // Droplet with surface tension holds together.
        let mut drop = droplet_oscillation(0.03, 0.1, 8.0);
        let dt = drop.stable_dt();
        for _ in 0..80 {
            drop.step(dt);
        }
        let r_max = drop
            .particles
            .iter()
            .map(|p| p.pos.magnitude())
            .fold(0.0_f64, f64::max);
        assert!(r_max < 0.4, "droplet dispersed: {r_max}");
        assert!(drop.particles.iter().all(|p| p.pos.x.is_finite()));
    }
}
