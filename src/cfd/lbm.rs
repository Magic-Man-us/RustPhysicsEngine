//! Lattice Boltzmann method: D2Q9 with BGK/TRT/MRT/cumulant-style
//! collisions, bounce-back solids, Zou-He open boundaries, Guo forcing,
//! D3Q19 and D3Q27 lattices, and classic benchmarks.

use crate::math::{Vec2, Vec3};

/// D2Q9 lattice velocities (0 = rest, 1-4 axis, 5-8 diagonal).
const E2: [(i64, i64); 9] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
];
const W2: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
/// Opposite direction of each D2Q9 velocity.
const OPP2: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Collision operator selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Collision {
    /// Single-relaxation-time BGK.
    Bgk,
    /// Two-relaxation-time with the given "magic" parameter Λ
    /// (Λ = 1/4 gives optimal stability, 3/16 exact walls).
    Trt { magic: f64 },
    /// Multiple-relaxation-time (Lallemand-Luo relaxation set).
    Mrt,
    /// Cumulant-style: MRT with aggressively relaxed high-order moments.
    Cumulant,
}

fn feq2(rho: f64, u: Vec2, k: usize) -> f64 {
    let (ex, ey) = (E2[k].0 as f64, E2[k].1 as f64);
    let eu = ex * u.x + ey * u.y;
    let u2 = u.x * u.x + u.y * u.y;
    W2[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
}

/// D2Q9 lattice Boltzmann solver.
pub struct LbmD2Q9 {
    pub nx: usize,
    pub ny: usize,
    pub f: Vec<[f64; 9]>,
    pub tau: f64,
    pub solid: Vec<bool>,
    pub force: Vec2,
    pub collision: Collision,
    /// Periodic wrap in x/y where no other boundary applies.
    pub periodic_x: bool,
    pub periodic_y: bool,
    /// Inlet velocity on the left wall (Zou-He), if set.
    inlet: Option<f64>,
    /// Outlet density on the right wall (Zou-He), if set.
    outlet: Option<f64>,
}

impl LbmD2Q9 {
    /// Quiescent unit-density fluid.
    #[must_use]
    pub fn new(nx: usize, ny: usize, tau: f64) -> Self {
        let mut s = Self {
            nx,
            ny,
            f: vec![[0.0; 9]; nx * ny],
            tau,
            solid: vec![false; nx * ny],
            force: Vec2::ZERO,
            collision: Collision::Bgk,
            periodic_x: true,
            periodic_y: true,
            inlet: None,
            outlet: None,
        };
        s.init_equilibrium(1.0, Vec2::ZERO);
        s
    }

    /// Reset every node to equilibrium at (rho, u).
    pub fn init_equilibrium(&mut self, rho: f64, u: Vec2) {
        for node in self.f.iter_mut() {
            for (k, fk) in node.iter_mut().enumerate() {
                *fk = feq2(rho, u, k);
            }
        }
    }

    /// Kinematic viscosity ν = (τ − 1/2)/3 in lattice units.
    #[must_use]
    pub fn viscosity(&self) -> f64 {
        (self.tau - 0.5) / 3.0
    }

    /// Densities at all nodes.
    #[must_use]
    pub fn density(&self) -> Vec<f64> {
        self.f.iter().map(|n| n.iter().sum()).collect()
    }

    /// Velocities at all nodes (forcing shift included).
    #[must_use]
    pub fn velocity(&self) -> Vec<Vec2> {
        self.f
            .iter()
            .map(|n| {
                let rho: f64 = n.iter().sum();
                let mut u = Vec2::ZERO;
                for k in 0..9 {
                    u.x += E2[k].0 as f64 * n[k];
                    u.y += E2[k].1 as f64 * n[k];
                }
                (u + self.force * 0.5) * (1.0 / rho.max(1e-12))
            })
            .collect()
    }

    /// Largest lattice Mach number |u|/c_s.
    #[must_use]
    pub fn mach_max(&self) -> f64 {
        let cs = (1.0_f64 / 3.0).sqrt();
        self.velocity()
            .iter()
            .map(Vec2::magnitude)
            .fold(0.0_f64, f64::max)
            / cs
    }

    /// Collision step (with Guo forcing).
    pub fn collide(&mut self) {
        let tau = self.tau;
        let omega = 1.0 / tau;
        let force = self.force;
        for (c, node) in self.f.iter_mut().enumerate() {
            if self.solid[c] {
                continue;
            }
            let rho: f64 = node.iter().sum();
            let mut u = Vec2::ZERO;
            for k in 0..9 {
                u.x += E2[k].0 as f64 * node[k];
                u.y += E2[k].1 as f64 * node[k];
            }
            u = (u + force * 0.5) * (1.0 / rho.max(1e-12));
            let feq: [f64; 9] = std::array::from_fn(|k| feq2(rho, u, k));
            match self.collision {
                Collision::Bgk => {
                    for k in 0..9 {
                        node[k] += omega * (feq[k] - node[k]);
                    }
                }
                Collision::Trt { magic } => {
                    // Symmetric part relaxes with ω⁺ = 1/τ; antisymmetric
                    // with ω⁻ chosen through the magic parameter
                    // Λ = (τ⁺ − 1/2)(τ⁻ − 1/2).
                    let lam_plus = tau - 0.5;
                    let tau_minus = magic / lam_plus + 0.5;
                    let om_m = 1.0 / tau_minus;
                    let f_old = *node;
                    for k in 0..9 {
                        let ko = OPP2[k];
                        let f_p = 0.5 * (f_old[k] + f_old[ko]);
                        let f_m = 0.5 * (f_old[k] - f_old[ko]);
                        let feq_p = 0.5 * (feq[k] + feq[ko]);
                        let feq_m = 0.5 * (feq[k] - feq[ko]);
                        node[k] = (f_p + omega * (feq_p - f_p)) + (f_m + om_m * (feq_m - f_m));
                    }
                }
                Collision::Mrt | Collision::Cumulant => {
                    // Moment-space relaxation (Lallemand-Luo D2Q9 set).
                    // Moments: rho, e, eps, jx, qx, jy, qy, pxx, pxy.
                    let f = *node;
                    let m = [
                        f.iter().sum::<f64>(),
                        -4.0 * f[0] - f[1] - f[2] - f[3] - f[4]
                            + 2.0 * (f[5] + f[6] + f[7] + f[8]),
                        4.0 * f[0] - 2.0 * (f[1] + f[2] + f[3] + f[4])
                            + f[5]
                            + f[6]
                            + f[7]
                            + f[8],
                        f[1] - f[3] + f[5] - f[6] - f[7] + f[8],
                        -2.0 * (f[1] - f[3]) + f[5] - f[6] - f[7] + f[8],
                        f[2] - f[4] + f[5] + f[6] - f[7] - f[8],
                        -2.0 * (f[2] - f[4]) + f[5] + f[6] - f[7] - f[8],
                        f[1] - f[2] + f[3] - f[4],
                        f[5] - f[6] + f[7] - f[8],
                    ];
                    let jx = m[3];
                    let jy = m[5];
                    let rho_m = m[0];
                    let meq = [
                        rho_m,
                        -2.0 * rho_m + 3.0 * (jx * jx + jy * jy) / rho_m.max(1e-12),
                        rho_m - 3.0 * (jx * jx + jy * jy) / rho_m.max(1e-12),
                        jx,
                        -jx,
                        jy,
                        -jy,
                        (jx * jx - jy * jy) / rho_m.max(1e-12),
                        jx * jy / rho_m.max(1e-12),
                    ];
                    // Relaxation rates: shear moments (pxx, pxy) use 1/τ;
                    // others fixed (cumulant relaxes bulk faster).
                    let s_bulk = if self.collision == Collision::Cumulant { 1.9 } else { 1.4 };
                    let s = [
                        0.0, s_bulk, 1.4, 0.0, 1.2, 0.0, 1.2, omega, omega,
                    ];
                    let mut m_post = [0.0; 9];
                    for i in 0..9 {
                        m_post[i] = m[i] + s[i] * (meq[i] - m[i]);
                    }
                    // Inverse moment transform (orthogonal basis norms).
                    let n0 = m_post[0] / 9.0;
                    let n1 = m_post[1] / 36.0;
                    let n2 = m_post[2] / 36.0;
                    let n3 = m_post[3] / 6.0;
                    let n4 = m_post[4] / 12.0;
                    let n5 = m_post[5] / 6.0;
                    let n6 = m_post[6] / 12.0;
                    let n7 = m_post[7] / 4.0;
                    let n8 = m_post[8] / 4.0;
                    node[0] = n0 - 4.0 * n1 + 4.0 * n2;
                    node[1] = n0 - n1 - 2.0 * n2 + n3 - 2.0 * n4 + n7;
                    node[2] = n0 - n1 - 2.0 * n2 + n5 - 2.0 * n6 - n7;
                    node[3] = n0 - n1 - 2.0 * n2 - n3 + 2.0 * n4 + n7;
                    node[4] = n0 - n1 - 2.0 * n2 - n5 + 2.0 * n6 - n7;
                    node[5] = n0 + 2.0 * n1 + n2 + n3 + n4 + n5 + n6 + n8;
                    node[6] = n0 + 2.0 * n1 + n2 - n3 - n4 + n5 + n6 - n8;
                    node[7] = n0 + 2.0 * n1 + n2 - n3 - n4 - n5 - n6 + n8;
                    node[8] = n0 + 2.0 * n1 + n2 + n3 + n4 - n5 - n6 - n8;
                }
            }
            // Guo forcing term with relaxation-matched prefactors: the
            // odd (momentum-carrying) part uses the odd rate, the even
            // part the even rate; plain BGK uses ω for both.
            if force.x != 0.0 || force.y != 0.0 {
                let (fac_odd, fac_even) = match self.collision {
                    Collision::Bgk => (1.0 - 0.5 * omega, 1.0 - 0.5 * omega),
                    Collision::Trt { magic } => {
                        let om_m = 1.0 / (magic / (tau - 0.5) + 0.5);
                        (1.0 - 0.5 * om_m, 1.0 - 0.5 * omega)
                    }
                    // Momentum moments are conserved (s = 0) in the MRT
                    // set, so their force projection enters in full; the
                    // stress projection uses ω.
                    Collision::Mrt | Collision::Cumulant => (1.0, 1.0 - 0.5 * omega),
                };
                for k in 0..9 {
                    let (ex, ey) = (E2[k].0 as f64, E2[k].1 as f64);
                    let eu = ex * u.x + ey * u.y;
                    let s_odd = 3.0 * (ex * force.x + ey * force.y);
                    let s_even = -3.0 * (u.x * force.x + u.y * force.y)
                        + 9.0 * eu * (ex * force.x + ey * force.y);
                    node[k] += W2[k] * (fac_odd * s_odd + fac_even * s_even);
                }
            }
        }
    }

    /// Streaming step (periodic wrap; solids handled by bounce-back).
    pub fn stream(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        let mut new_f = self.f.clone();
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let c = (j as usize) * nx + i as usize;
                for k in 0..9 {
                    let (di, dj) = E2[k];
                    let mut ti = i + di;
                    let mut tj = j + dj;
                    let mut blocked = false;
                    if ti < 0 || ti >= nx as i64 {
                        if self.periodic_x {
                            ti = ti.rem_euclid(nx as i64);
                        } else {
                            blocked = true;
                        }
                    }
                    if tj < 0 || tj >= ny as i64 {
                        if self.periodic_y {
                            tj = tj.rem_euclid(ny as i64);
                        } else {
                            blocked = true;
                        }
                    }
                    if blocked {
                        // Domain wall without wrap: bounce back in place.
                        new_f[c][OPP2[k]] = self.f[c][k];
                    } else {
                        new_f[(tj as usize) * nx + ti as usize][k] = self.f[c][k];
                    }
                }
            }
        }
        self.f = new_f;
    }

    /// Half-way bounce-back on solid nodes.
    pub fn bounce_back(&mut self) {
        for c in 0..self.nx * self.ny {
            if self.solid[c] {
                let node = self.f[c];
                for k in 0..9 {
                    self.f[c][k] = node[OPP2[k]];
                }
            }
        }
    }

    /// Zou-He velocity inlet on the left wall (x = 0), horizontal
    /// velocity `u`.
    pub fn zou_he_velocity_inlet(&mut self, u: f64) {
        self.inlet = Some(u);
        self.periodic_x = false;
        let nx = self.nx;
        for j in 0..self.ny {
            let c = j * nx;
            let f = &mut self.f[c];
            let rho = (f[0] + f[2] + f[4] + 2.0 * (f[3] + f[6] + f[7])) / (1.0 - u);
            f[1] = f[3] + 2.0 / 3.0 * rho * u;
            f[5] = f[7] - 0.5 * (f[2] - f[4]) + rho * u / 6.0;
            f[8] = f[6] + 0.5 * (f[2] - f[4]) + rho * u / 6.0;
        }
    }

    /// Zou-He pressure (density) outlet on the right wall.
    pub fn zou_he_pressure_outlet(&mut self, rho: f64) {
        self.outlet = Some(rho);
        self.periodic_x = false;
        let nx = self.nx;
        for j in 0..self.ny {
            let c = j * nx + (nx - 1);
            let f = &mut self.f[c];
            let u = (f[0] + f[2] + f[4] + 2.0 * (f[1] + f[5] + f[8])) / rho - 1.0;
            f[3] = f[1] - 2.0 / 3.0 * rho * u;
            f[7] = f[5] + 0.5 * (f[2] - f[4]) - rho * u / 6.0;
            f[6] = f[8] - 0.5 * (f[2] - f[4]) - rho * u / 6.0;
        }
    }

    /// Make both axes periodic (clears open boundaries).
    pub fn periodic(&mut self) {
        self.periodic_x = true;
        self.periodic_y = true;
        self.inlet = None;
        self.outlet = None;
    }

    /// One full update: collide, stream, bounce-back, boundaries.
    pub fn step(&mut self) {
        self.collide();
        self.stream();
        self.bounce_back();
        if let Some(u) = self.inlet {
            self.zou_he_velocity_inlet(u);
        }
        if let Some(rho) = self.outlet {
            self.zou_he_pressure_outlet(rho);
        }
    }

    /// Run `n` steps.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Momentum-exchange drag on the solid set (lattice units).
    #[must_use]
    pub fn drag_on_solid(&self) -> Vec2 {
        let (nx, ny) = (self.nx, self.ny);
        let mut fx = 0.0;
        let mut fy = 0.0;
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let c = (j as usize) * nx + i as usize;
                if self.solid[c] {
                    continue;
                }
                for (k, &(di, dj)) in E2.iter().enumerate().skip(1) {
                    let ti = (i + di).rem_euclid(nx as i64) as usize;
                    let tj = (j + dj).rem_euclid(ny as i64) as usize;
                    if self.solid[tj * nx + ti] {
                        // Momentum transferred by the bounced population.
                        let transfer = 2.0 * self.f[c][k];
                        fx += transfer * di as f64;
                        fy += transfer * dj as f64;
                    }
                }
            }
        }
        Vec2::new(fx, fy)
    }

    /// Vorticity at interior nodes (central differences).
    #[must_use]
    pub fn vorticity(&self) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let vel = self.velocity();
        let mut w = vec![0.0; nx * ny];
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let dv_dx = (vel[j * nx + i + 1].y - vel[j * nx + i - 1].y) / 2.0;
                let du_dy = (vel[(j + 1) * nx + i].x - vel[(j - 1) * nx + i].x) / 2.0;
                w[j * nx + i] = dv_dx - du_dy;
            }
        }
        w
    }
}

// --- D3Q19 ---------------------------------------------------------------

const E19: [(i64, i64, i64); 19] = [
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (-1, -1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, -1),
    (1, 0, -1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, -1),
    (0, 1, -1),
    (0, -1, 1),
];

fn w19(k: usize) -> f64 {
    if k == 0 {
        1.0 / 3.0
    } else if k <= 6 {
        1.0 / 18.0
    } else {
        1.0 / 36.0
    }
}

fn opp19(k: usize) -> usize {
    const OPP: [usize; 19] =
        [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];
    OPP[k]
}

/// D3Q19 BGK lattice Boltzmann solver (periodic + bounce-back).
pub struct LbmD3Q19 {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub f: Vec<[f64; 19]>,
    pub tau: f64,
    pub solid: Vec<bool>,
    pub force: Vec3,
}

impl LbmD3Q19 {
    /// Quiescent unit-density fluid.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64) -> Self {
        let mut s = Self {
            nx,
            ny,
            nz,
            f: vec![[0.0; 19]; nx * ny * nz],
            tau,
            solid: vec![false; nx * ny * nz],
            force: Vec3::ZERO,
        };
        for node in s.f.iter_mut() {
            for (k, fk) in node.iter_mut().enumerate() {
                *fk = w19(k);
            }
        }
        s
    }

    /// Kinematic viscosity.
    #[must_use]
    pub fn viscosity(&self) -> f64 {
        (self.tau - 0.5) / 3.0
    }

    /// Node velocities.
    #[must_use]
    pub fn velocity(&self) -> Vec<Vec3> {
        self.f
            .iter()
            .map(|n| {
                let rho: f64 = n.iter().sum();
                let mut u = Vec3::ZERO;
                for (k, &fk) in n.iter().enumerate() {
                    u.x += E19[k].0 as f64 * fk;
                    u.y += E19[k].1 as f64 * fk;
                    u.z += E19[k].2 as f64 * fk;
                }
                (u + self.force * 0.5) * (1.0 / rho.max(1e-12))
            })
            .collect()
    }

    /// One BGK step with Guo forcing.
    pub fn step(&mut self) {
        let omega = 1.0 / self.tau;
        let force = self.force;
        // Collide.
        for (c, node) in self.f.iter_mut().enumerate() {
            if self.solid[c] {
                continue;
            }
            let rho: f64 = node.iter().sum();
            let mut u = Vec3::ZERO;
            for (k, &fk) in node.iter().enumerate() {
                u.x += E19[k].0 as f64 * fk;
                u.y += E19[k].1 as f64 * fk;
                u.z += E19[k].2 as f64 * fk;
            }
            u = (u + force * 0.5) * (1.0 / rho.max(1e-12));
            let u2 = u.magnitude_squared();
            for (k, fk) in node.iter_mut().enumerate() {
                let eu =
                    E19[k].0 as f64 * u.x + E19[k].1 as f64 * u.y + E19[k].2 as f64 * u.z;
                let feq = w19(k) * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
                *fk += omega * (feq - *fk);
                let ef = E19[k].0 as f64 * force.x
                    + E19[k].1 as f64 * force.y
                    + E19[k].2 as f64 * force.z;
                let uf = u.dot(&force);
                *fk += w19(k) * (1.0 - 0.5 * omega) * (3.0 * ef - 3.0 * uf + 9.0 * eu * ef);
            }
        }
        // Stream (fully periodic).
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let mut new_f = self.f.clone();
        for k in 0..19 {
            let (di, dj, dk) = E19[k];
            for z in 0..nz as i64 {
                for y in 0..ny as i64 {
                    for x in 0..nx as i64 {
                        let c = ((z as usize) * ny + y as usize) * nx + x as usize;
                        let tx = (x + di).rem_euclid(nx as i64) as usize;
                        let ty = (y + dj).rem_euclid(ny as i64) as usize;
                        let tz = (z + dk).rem_euclid(nz as i64) as usize;
                        new_f[(tz * ny + ty) * nx + tx][k] = self.f[c][k];
                    }
                }
            }
        }
        self.f = new_f;
        // Bounce back.
        for c in 0..nx * ny * nz {
            if self.solid[c] {
                let node = self.f[c];
                for k in 0..19 {
                    self.f[c][k] = node[opp19(k)];
                }
            }
        }
    }
}

/// D3Q27 lattice constants (velocities and weights), for custom solvers.
pub struct LbmD3Q27;

impl LbmD3Q27 {
    /// The 27 lattice velocities.
    #[must_use]
    pub fn velocities() -> Vec<(i64, i64, i64)> {
        let mut v = Vec::with_capacity(27);
        for z in -1..=1_i64 {
            for y in -1..=1_i64 {
                for x in -1..=1_i64 {
                    v.push((x, y, z));
                }
            }
        }
        v
    }

    /// Weight of velocity (x, y, z).
    #[must_use]
    pub fn weight(e: (i64, i64, i64)) -> f64 {
        match e.0.abs() + e.1.abs() + e.2.abs() {
            0 => 8.0 / 27.0,
            1 => 2.0 / 27.0,
            2 => 1.0 / 54.0,
            _ => 1.0 / 216.0,
        }
    }
}

// --- Benchmarks ----------------------------------------------------------

/// Body-force-driven Poiseuille channel: solid walls at j = 0 and
/// j = ny−1, periodic in x.
#[must_use]
pub fn lbm_poiseuille_2d(nx: usize, ny: usize, tau: f64, force: f64) -> LbmD2Q9 {
    let mut lbm = LbmD2Q9::new(nx, ny, tau);
    lbm.force = Vec2::new(force, 0.0);
    for i in 0..nx {
        lbm.solid[i] = true;
        lbm.solid[(ny - 1) * nx + i] = true;
    }
    lbm
}

/// Exact Poiseuille profile u(y) for channel half-width walls at y = 0
/// and y = h.
#[must_use]
pub fn poiseuille_exact(y: f64, h: f64, force: f64, nu: f64) -> f64 {
    force / (2.0 * nu) * y * (h - y)
}

/// Flow past a cylinder at Reynolds number `re` (Zou-He inlet/outlet).
#[must_use]
pub fn lbm_cylinder(nx: usize, ny: usize, re: f64) -> LbmD2Q9 {
    let u_in = 0.08;
    let d = ny as f64 / 5.0;
    let nu = u_in * d / re;
    let tau = 3.0 * nu + 0.5;
    let mut lbm = LbmD2Q9::new(nx, ny, tau);
    lbm.periodic_x = false;
    // Cylinder at (nx/4, ny/2 + tiny offset to trigger shedding).
    let (cx, cy) = (nx as f64 / 4.0, ny as f64 / 2.0 + 0.5);
    for j in 0..ny {
        for i in 0..nx {
            let dx = i as f64 - cx;
            let dy = j as f64 - cy;
            if (dx * dx + dy * dy).sqrt() < d / 2.0 {
                lbm.solid[j * nx + i] = true;
            }
        }
    }
    // Top/bottom walls.
    for i in 0..nx {
        lbm.solid[i] = true;
        lbm.solid[(ny - 1) * nx + i] = true;
    }
    lbm.periodic_y = false;
    lbm.init_equilibrium(1.0, Vec2::new(u_in, 0.0));
    lbm.inlet = Some(u_in);
    lbm.outlet = Some(1.0);
    lbm
}

/// Lid-driven cavity at Reynolds number `re` (moving top wall via
/// bounce-back with wall velocity).
#[must_use]
pub fn lbm_lid_cavity(n: usize, re: f64) -> LbmD2Q9 {
    let u_lid = 0.1;
    let nu = u_lid * n as f64 / re;
    let tau = 3.0 * nu + 0.5;
    let mut lbm = LbmD2Q9::new(n, n, tau);
    lbm.periodic_x = false;
    lbm.periodic_y = false;
    // Side and bottom walls solid; the lid handled in the driver below.
    for j in 0..n {
        lbm.solid[j * n] = true;
        lbm.solid[j * n + n - 1] = true;
    }
    for i in 0..n {
        lbm.solid[i] = true;
    }
    lbm
}

/// Apply the moving-lid boundary to a cavity solver for one step: after
/// the regular step, impose the lid velocity on the top row (Zou-He).
pub fn lbm_cavity_step(lbm: &mut LbmD2Q9, u_lid: f64) {
    lbm.step();
    let (nx, ny) = (lbm.nx, lbm.ny);
    for i in 0..nx {
        let c = (ny - 1) * nx + i;
        let f = &mut lbm.f[c];
        // Zou-He moving top wall (v = 0, u = u_lid).
        let rho = f[0] + f[1] + f[3] + 2.0 * (f[2] + f[5] + f[6]);
        f[4] = f[2];
        f[7] = f[5] + 0.5 * (f[1] - f[3]) - 0.5 * rho * u_lid;
        f[8] = f[6] - 0.5 * (f[1] - f[3]) + 0.5 * rho * u_lid;
    }
}

/// Double-distribution thermal LBM: returns (flow lattice, temperature
/// lattice); the temperature field advects with the flow and feeds back
/// as a Boussinesq force. Step both manually with `thermal_step`.
#[must_use]
pub fn lbm_thermal(nx: usize, ny: usize, tau_f: f64, tau_g: f64) -> (LbmD2Q9, LbmD2Q9) {
    let flow = LbmD2Q9::new(nx, ny, tau_f);
    let mut temp = LbmD2Q9::new(nx, ny, tau_g);
    // Initialize a hot stripe at the bottom.
    for j in 0..ny {
        for i in 0..nx {
            let t0: f64 = if j < ny / 8 { 1.0 } else { 0.0 };
            for (k, fk) in temp.f[j * nx + i].iter_mut().enumerate() {
                *fk = W2[k] * t0.max(1e-6);
            }
        }
    }
    (flow, temp)
}

/// One coupled Boussinesq step of the double-distribution system.
pub fn thermal_step(flow: &mut LbmD2Q9, temp: &mut LbmD2Q9, buoyancy: f64) {
    // Advect the temperature with the flow velocity (equilibrium built
    // from the flow's u).
    let vel = flow.velocity();
    let omega_g = 1.0 / temp.tau;
    for (c, node) in temp.f.iter_mut().enumerate() {
        let t: f64 = node.iter().sum();
        for (k, fk) in node.iter_mut().enumerate() {
            let feq = feq2(t, vel[c], k);
            *fk += omega_g * (feq - *fk);
        }
    }
    temp.stream();
    // Boussinesq: buoyancy force from the temperature.
    let t_field: Vec<f64> = temp.f.iter().map(|n| n.iter().sum()).collect();
    let t_mean = t_field.iter().sum::<f64>() / t_field.len() as f64;
    // Uniform mean force per node is applied globally (spatially varying
    // forcing needs per-node force; approximate with the mean of the
    // positive anomaly acting upward).
    let anomaly = t_field.iter().map(|t| t - t_mean).fold(0.0_f64, f64::max);
    flow.force = Vec2::new(0.0, buoyancy * anomaly);
    flow.step();
}

/// Convert a lattice velocity to physical units given the lattice
/// spacing and time step.
#[must_use]
pub fn lbm_to_physical(u_lattice: f64, dx: f64, dt: f64) -> f64 {
    u_lattice * dx / dt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_and_viscosity() {
        let mut lbm = LbmD2Q9::new(24, 24, 0.8);
        // Random-ish perturbation.
        for (c, node) in lbm.f.iter_mut().enumerate() {
            node[1] += 0.01 * ((c * 7919 % 13) as f64 / 13.0 - 0.5);
        }
        let mass0: f64 = lbm.density().iter().sum();
        lbm.run(50);
        let mass: f64 = lbm.density().iter().sum();
        assert!((mass / mass0 - 1.0).abs() < 1e-12, "mass drift");
        assert!((lbm.viscosity() - 0.1).abs() < 1e-12);
        assert!(lbm.mach_max() < 0.3);
        // Momentum conserved without forces/solids.
        let vel = lbm.velocity();
        let rho = lbm.density();
        let px: f64 = vel.iter().zip(&rho).map(|(v, r)| v.x * r).sum();
        lbm.run(20);
        let vel2 = lbm.velocity();
        let rho2 = lbm.density();
        let px2: f64 = vel2.iter().zip(&rho2).map(|(v, r)| v.x * r).sum();
        assert!((px - px2).abs() < 1e-10, "momentum drift {px} vs {px2}");
    }

    #[test]
    fn test_poiseuille_matches_exact() {
        let (nx, ny) = (8, 33);
        let force = 1e-6;
        let tau = 0.9;
        let mut lbm = lbm_poiseuille_2d(nx, ny, tau, force);
        lbm.run(6000);
        let nu = lbm.viscosity();
        let vel = lbm.velocity();
        // Channel interior spans j = 1..ny-1; walls act at half-way
        // between the solid and fluid nodes.
        let h = (ny - 2) as f64; // wall planes sit half-way past the fluid
        let mut max_rel = 0.0_f64;
        let u_max_exact = poiseuille_exact(h / 2.0, h, force, nu);
        for j in 1..ny - 1 {
            let y = (j - 1) as f64 + 0.5; // distance from the lower wall plane
            let u = vel[j * nx + 4].x;
            let ue = poiseuille_exact(y, h, force, nu);
            max_rel = max_rel.max((u - ue).abs() / u_max_exact);
        }
        assert!(max_rel < 0.03, "Poiseuille profile error {max_rel}");
        // Center beats near-wall velocity.
        let u_c = vel[(ny / 2) * nx + 4].x;
        let u_w = vel[nx + 4].x;
        assert!(u_c > 3.0 * u_w, "no parabola: {u_c} vs {u_w}");
    }

    #[test]
    fn test_collision_operators_stable() {
        for coll in [
            Collision::Bgk,
            Collision::Trt { magic: 0.25 },
            Collision::Mrt,
            Collision::Cumulant,
        ] {
            let mut lbm = lbm_poiseuille_2d(8, 17, 0.7, 1e-6);
            lbm.collision = coll;
            lbm.run(6000);
            let vel = lbm.velocity();
            assert!(
                vel.iter().all(|v| v.x.is_finite() && v.x.abs() < 0.3),
                "{coll:?} unstable"
            );
            // Profile still parabola-ish.
            let u_c = vel[8 * 8 + 4].x;
            let u_w = vel[8 + 4].x;
            assert!(u_c > u_w, "{coll:?} profile");
            // TRT/MRT match BGK viscosity closely on this laminar flow.
            let nu = lbm.viscosity();
            let h = 15.0; // ny - 2 with half-way wall planes
            let ue = poiseuille_exact(h / 2.0, h, 1e-6, nu);
            assert!((u_c / ue - 1.0).abs() < 0.1, "{coll:?} center {u_c} vs {ue}");
        }
    }

    #[test]
    fn test_cylinder_and_drag() {
        let mut lbm = lbm_cylinder(80, 40, 60.0);
        lbm.run(800);
        let vel = lbm.velocity();
        assert!(vel.iter().all(|v| v.magnitude().is_finite()));
        // Drag pushes the cylinder downstream (+x).
        let drag = lbm.drag_on_solid();
        assert!(drag.x > 0.0, "drag {drag:?}");
        // Wake: velocity right behind the cylinder is below the inlet.
        let (cx, cy) = (80 / 4, 20);
        let behind = vel[cy * 80 + cx + 8].x;
        assert!(behind < 0.08, "no wake deficit: {behind}");
        // Vorticity of opposite signs above/below the cylinder.
        let w = lbm.vorticity();
        let above = w[(cy + 4) * 80 + cx + 4];
        let below = w[(cy - 4) * 80 + cx + 4];
        assert!(above * below < 0.0, "no shear layers: {above} vs {below}");
    }

    #[test]
    fn test_lid_cavity_circulation() {
        let n = 32;
        let mut lbm = lbm_lid_cavity(n, 100.0);
        for _ in 0..3000 {
            lbm_cavity_step(&mut lbm, 0.1);
        }
        let vel = lbm.velocity();
        // Flow near the lid follows it; return flow at the bottom is
        // opposite.
        let top = vel[(n - 2) * n + n / 2].x;
        let bottom = vel[2 * n + n / 2].x;
        assert!(top > 0.01, "no lid drag: {top}");
        assert!(bottom < 0.0, "no return flow: {bottom}");
        // A single primary vortex: net vorticity is single-signed.
        let w = lbm.vorticity();
        let sum: f64 = w.iter().sum();
        assert!(sum.abs() > 1e-3);
    }

    #[test]
    fn test_d3q19_and_d3q27() {
        // Weights normalize.
        let vs = LbmD3Q27::velocities();
        assert_eq!(vs.len(), 27);
        let total: f64 = vs.iter().map(|&e| LbmD3Q27::weight(e)).sum();
        assert!((total - 1.0).abs() < 1e-12);
        let w_sum: f64 = (0..19).map(w19).sum();
        assert!((w_sum - 1.0).abs() < 1e-12);
        // 3D body-driven channel flow between z walls.
        let (nx, ny, nz) = (4, 4, 17);
        let mut lbm = LbmD3Q19::new(nx, ny, nz, 0.8);
        lbm.force = Vec3::new(1e-6, 0.0, 0.0);
        for j in 0..ny {
            for i in 0..nx {
                lbm.solid[j * nx + i] = true;
                lbm.solid[((nz - 1) * ny + j) * nx + i] = true;
            }
        }
        for _ in 0..2000 {
            lbm.step();
        }
        let vel = lbm.velocity();
        let center = vel[((nz / 2) * ny + 2) * nx + 2].x;
        let wall = vel[(ny + 2) * nx + 2].x;
        assert!(center > 2.0 * wall.max(1e-9), "3D channel: {center} vs {wall}");
        assert!(vel.iter().all(|v| v.magnitude().is_finite()));
        // Mass conserved.
        let mass: f64 = lbm.f.iter().map(|n| n.iter().sum::<f64>()).sum();
        assert!((mass / (nx * ny * nz) as f64 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_periodic_restores_wraparound() {
        // A lattice configured with Zou-He open boundaries is closed in x;
        // `periodic()` must put the torus back and drop the open-boundary
        // bookkeeping.
        let (nx, ny) = (16, 8);
        let u0 = Vec2::new(0.05, 0.0);

        let mut torus = LbmD2Q9::new(nx, ny, 0.8);
        torus.periodic_x = false;
        torus.periodic_y = false;
        torus.periodic();
        assert!(torus.periodic_x && torus.periodic_y);
        torus.init_equilibrium(1.0, u0);

        // On a torus with no solids and no forcing, a uniform equilibrium
        // state is an exact fixed point: collision leaves f = f_eq and
        // streaming just permutes identical values. Mass and momentum are
        // conserved to round-off.
        let mass0: f64 = torus.density().iter().sum();
        let px0: f64 = torus
            .velocity()
            .iter()
            .zip(torus.density())
            .map(|(v, r)| v.x * r)
            .sum();
        torus.run(60);
        for (c, (rho, v)) in torus.density().iter().zip(torus.velocity()).enumerate() {
            assert!((rho - 1.0).abs() < 1e-12, "density {rho} at {c}");
            assert!((v.x - u0.x).abs() < 1e-12, "u {v:?} at {c}");
            assert!(v.y.abs() < 1e-12, "v {v:?} at {c}");
        }
        let mass: f64 = torus.density().iter().sum();
        let px: f64 = torus
            .velocity()
            .iter()
            .zip(torus.density())
            .map(|(v, r)| v.x * r)
            .sum();
        assert!((mass - mass0).abs() < 1e-12, "mass drift {mass} vs {mass0}");
        assert!((px - px0).abs() < 1e-12, "momentum drift {px} vs {px0}");

        // The same uniform state on a closed lattice does not survive:
        // the stream bounces off the walls, so this is not a vacuous
        // check of the torus.
        let mut closed = LbmD2Q9::new(nx, ny, 0.8);
        closed.periodic_x = false;
        closed.periodic_y = false;
        closed.init_equilibrium(1.0, u0);
        closed.run(60);
        let closed_dev = closed
            .velocity()
            .iter()
            .map(|v| (v.x - u0.x).abs())
            .fold(0.0_f64, f64::max);
        assert!(closed_dev > 1e-3, "closed lattice was undisturbed: {closed_dev}");

        // `periodic()` also drops the Zou-He inlet/outlet: a uniform
        // ρ = 1.2 lattice at rest is then an exact fixed point, whereas a
        // live pressure outlet would drive the last column back to ρ = 1.
        let mut cleared = LbmD2Q9::new(nx, ny, 0.8);
        cleared.zou_he_velocity_inlet(0.05);
        cleared.zou_he_pressure_outlet(1.0);
        cleared.periodic();
        cleared.init_equilibrium(1.2, Vec2::ZERO);
        cleared.run(30);
        let rho_dev = cleared
            .density()
            .iter()
            .map(|r| (r - 1.2).abs())
            .fold(0.0_f64, f64::max);
        assert!(rho_dev < 1e-12, "open boundaries still active: {rho_dev}");
        let mut still_open = LbmD2Q9::new(nx, ny, 0.8);
        still_open.zou_he_velocity_inlet(0.05);
        still_open.zou_he_pressure_outlet(1.0);
        still_open.init_equilibrium(1.2, Vec2::ZERO);
        still_open.run(30);
        let open_dev = still_open
            .density()
            .iter()
            .map(|r| (r - 1.2).abs())
            .fold(0.0_f64, f64::max);
        assert!(open_dev > 1e-3, "the outlet should have pulled ρ down: {open_dev}");

        // Wrap-around distance: with the collision switched off (τ → ∞,
        // ω ≈ 0) a population injected in the +x direction advects one
        // node per step and must return to its origin after exactly nx
        // steps.
        let mut ballistic = LbmD2Q9::new(nx, ny, 1e9);
        ballistic.periodic();
        let (i0, j0) = (3_usize, 4_usize);
        let delta = 0.02;
        let base = ballistic.f[j0 * nx + i0][1];
        ballistic.f[j0 * nx + i0][1] += delta;
        let mass_b: f64 = ballistic.density().iter().sum();
        for s in 1..=nx {
            ballistic.step();
            let here = (i0 + s) % nx;
            let carried = ballistic.f[j0 * nx + here][1] - base;
            assert!(
                (carried - delta).abs() < 1e-6,
                "step {s}: perturbation {carried} not at column {here}"
            );
        }
        // Back where it started after a full lap.
        assert!(
            (ballistic.f[j0 * nx + i0][1] - base - delta).abs() < 1e-6,
            "no wrap-around after {nx} steps"
        );
        assert!(
            (ballistic.density().iter().sum::<f64>() - mass_b).abs() < 1e-12,
            "mass lost during the lap"
        );
    }

    #[test]
    fn test_d3q19_viscosity_matches_shear_wave_decay() {
        // ν = (τ − 1/2)/3 in lattice units (c_s² = 1/3, dt = 1).
        for &tau in &[0.6_f64, 0.8, 1.0, 1.5] {
            let lbm = LbmD3Q19::new(2, 2, 2, tau);
            assert!(
                (lbm.viscosity() - (tau - 0.5) / 3.0).abs() < 1e-15,
                "tau {tau}: nu {}",
                lbm.viscosity()
            );
        }
        // Physical check: a transverse shear wave u_x = U sin(k z) on a
        // fully periodic lattice decays as exp(−ν k² t). Measuring the
        // decay recovers the viscosity the formula reports.
        let (nx, ny, nz) = (4_usize, 4_usize, 16_usize);
        let tau = 0.8;
        let mut lbm = LbmD3Q19::new(nx, ny, nz, tau);
        let k = 2.0 * std::f64::consts::PI / nz as f64;
        let amp = 0.01;
        for z in 0..nz {
            let ux = amp * (k * z as f64).sin();
            for y in 0..ny {
                for x in 0..nx {
                    let c = (z * ny + y) * nx + x;
                    let u2 = ux * ux;
                    for k19 in 0..19 {
                        let eu = E19[k19].0 as f64 * ux;
                        lbm.f[c][k19] =
                            w19(k19) * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
                    }
                }
            }
        }
        // Fourier amplitude of the sin(kz) mode of the plane-averaged u_x.
        let mode = |lbm: &LbmD3Q19| -> f64 {
            let vel = lbm.velocity();
            let mut a = 0.0;
            for z in 0..nz {
                let mut mean = 0.0;
                for y in 0..ny {
                    for x in 0..nx {
                        mean += vel[(z * ny + y) * nx + x].x;
                    }
                }
                mean /= (nx * ny) as f64;
                a += mean * (k * z as f64).sin();
            }
            2.0 * a / nz as f64
        };
        // Skip a short start-up transient, then measure the decay rate
        // over a window.
        for _ in 0..20 {
            lbm.step();
        }
        let a0 = mode(&lbm);
        let window = 100;
        for _ in 0..window {
            lbm.step();
        }
        let a1 = mode(&lbm);
        assert!(a0 > 0.5 * amp && a1 > 0.0, "mode collapsed: {a0} -> {a1}");
        let nu_measured = -(a1 / a0).ln() / (k * k * window as f64);
        let nu_formula = lbm.viscosity();
        // The lattice-Boltzmann shear mode decays at exactly ν k² up to
        // O((k dx)²) lattice corrections; here k dx = 0.39, so a few
        // percent is the expected accuracy.
        assert!(
            (nu_measured / nu_formula - 1.0).abs() < 0.05,
            "measured nu {nu_measured} vs (tau-0.5)/3 = {nu_formula}"
        );
        // Doubling (τ − 1/2) doubles the viscosity and so doubles the
        // decay rate: an independent confirmation of the linear relation.
        let tau2 = 0.5 + 2.0 * (tau - 0.5);
        let mut fast = LbmD3Q19::new(nx, ny, nz, tau2);
        for z in 0..nz {
            let ux = amp * (k * z as f64).sin();
            for y in 0..ny {
                for x in 0..nx {
                    let c = (z * ny + y) * nx + x;
                    let u2 = ux * ux;
                    for k19 in 0..19 {
                        let eu = E19[k19].0 as f64 * ux;
                        fast.f[c][k19] =
                            w19(k19) * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
                    }
                }
            }
        }
        assert!((fast.viscosity() - 2.0 * nu_formula).abs() < 1e-15);
        for _ in 0..20 {
            fast.step();
        }
        let b0 = mode(&fast);
        for _ in 0..window {
            fast.step();
        }
        let b1 = mode(&fast);
        let rate_ratio = (b0 / b1).ln() / (a0 / a1).ln();
        assert!(
            (rate_ratio - 2.0).abs() < 0.1,
            "decay-rate ratio {rate_ratio} for doubled viscosity"
        );
    }

    #[test]
    fn test_thermal_plume_and_units() {
        let (mut flow, mut temp) = lbm_thermal(24, 24, 0.7, 0.7);
        for _ in 0..200 {
            thermal_step(&mut flow, &mut temp, 1e-4);
        }
        let vel = flow.velocity();
        // Buoyancy drives net upward motion.
        let vy_mean: f64 = vel.iter().map(|v| v.y).sum::<f64>() / vel.len() as f64;
        assert!(vy_mean > 0.0, "no buoyant rise: {vy_mean}");
        assert!(vel.iter().all(|v| v.magnitude().is_finite()));
        assert!((lbm_to_physical(0.1, 1e-3, 1e-5) - 10.0).abs() < 1e-12);
    }
}
