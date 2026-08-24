//! Stable-fluids incompressible solver on a MAC grid: MacCormack
//! advection, implicit viscosity, buoyancy, vorticity confinement, and a
//! pressure projection with a choice of Poisson solvers, plus classic
//! benchmark configurations.

use crate::cfd::grid::{CellField2, FluidBc, MacGrid2, MacGrid3};
use crate::math::{Vec2, Vec3};

const PI: f64 = crate::math::constants::PI;

/// Pressure Poisson solver choice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressureSolver {
    /// Fixed Jacobi iteration count.
    Jacobi(usize),
    /// Fixed Gauss-Seidel iteration count.
    GaussSeidel(usize),
    /// Conjugate gradients to tolerance.
    Cg { tol: f64, max_iter: usize },
    /// Diagonally preconditioned CG (tol 1e-10, 800 iterations).
    Pcg,
    /// DCT-based direct solve (rectangular domains without solids).
    Fft,
    /// Geometric multigrid V-cycles (count).
    Multigrid(usize),
}

/// Masked 5-point Laplacian application: A p = Σ_{fluid nbr} (p_c − p_n)/dx²
/// (SPD form of −∇² with Neumann walls/solids).
fn apply_neumann_laplacian(
    p: &[f64],
    out: &mut [f64],
    solid: &[bool],
    nx: usize,
    ny: usize,
    dx: f64,
) {
    let inv = 1.0 / (dx * dx);
    for j in 0..ny {
        for i in 0..nx {
            let c = j * nx + i;
            if solid[c] {
                out[c] = 0.0;
                continue;
            }
            let mut acc = 0.0;
            let mut count = 0.0;
            let mut visit = |n: usize| {
                if !solid[n] {
                    acc += p[n];
                    count += 1.0;
                }
            };
            if i > 0 {
                visit(c - 1);
            }
            if i + 1 < nx {
                visit(c + 1);
            }
            if j > 0 {
                visit(c - nx);
            }
            if j + 1 < ny {
                visit(c + nx);
            }
            out[c] = (count * p[c] - acc) * inv;
        }
    }
}

/// Solve ∇²p = rhs with Neumann boundaries (and solid cells) by
/// conjugate gradients; the mean of `rhs` over fluid cells is removed
/// for compatibility and the solution has zero mean.
#[allow(clippy::too_many_arguments)] // grid geometry parameters
#[must_use]
pub fn pressure_poisson_cg(
    div: &[f64],
    solid: &[bool],
    nx: usize,
    ny: usize,
    dx: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = nx * ny;
    let fluid_count = solid.iter().filter(|s| !**s).count().max(1) as f64;
    let mean = div
        .iter()
        .zip(solid)
        .filter(|(_, s)| !**s)
        .map(|(v, _)| v)
        .sum::<f64>()
        / fluid_count;
    // A p = b with A = −∇² (SPD) and b = −(rhs − mean).
    let b: Vec<f64> = div
        .iter()
        .zip(solid)
        .map(|(v, s)| if *s { 0.0 } else { -(v - mean) })
        .collect();
    let mut p = vec![0.0; n];
    let mut r = b.clone();
    let mut d = r.clone();
    let mut ap = vec![0.0; n];
    let mut rs_old: f64 = r.iter().map(|v| v * v).sum();
    let b_norm = rs_old.sqrt().max(1e-300);
    for _ in 0..max_iter {
        if rs_old.sqrt() / b_norm < tol {
            break;
        }
        apply_neumann_laplacian(&d, &mut ap, solid, nx, ny, dx);
        let dad: f64 = d.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if dad.abs() < 1e-300 {
            break;
        }
        let alpha = rs_old / dad;
        for i in 0..n {
            p[i] += alpha * d[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rs_new / rs_old;
        for i in 0..n {
            d[i] = r[i] + beta * d[i];
        }
        rs_old = rs_new;
    }
    // Zero mean over fluid cells.
    let p_mean = p
        .iter()
        .zip(solid)
        .filter(|(_, s)| !**s)
        .map(|(v, _)| v)
        .sum::<f64>()
        / fluid_count;
    p.iter_mut().zip(solid).for_each(|(v, s)| {
        if *s {
            *v = 0.0;
        } else {
            *v -= p_mean;
        }
    });
    p
}

/// CG solve of the node-centered Neumann pressure system used by
/// [`crate::sim::fluid_sim::EulerFluid2D`] (column-major layout
/// `i*ny + j`, mirror boundary nodes, anisotropic spacing). This is the
/// Part 3 rewire target for that solver's Poisson step.
#[allow(clippy::too_many_arguments)] // grid geometry parameters
#[must_use]
pub fn poisson_neumann_cg_rect(
    rhs: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let idx = |i: usize, j: usize| i * ny + j;
    let (ix2, iy2) = (1.0 / (dx * dx), 1.0 / (dy * dy));
    // Compatibility: remove the interior mean of rhs.
    let count = ((nx - 2) * (ny - 2)).max(1) as f64;
    let mut mean = 0.0;
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            mean += rhs[idx(i, j)];
        }
    }
    mean /= count;
    // A p = b with A = −∇² (mirror/Neumann boundaries folded in).
    let apply = |x: &[f64], out: &mut [f64]| {
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let mirror =
                    |ii: usize, jj: usize| x[idx(ii.clamp(1, nx - 2), jj.clamp(1, ny - 2))];
                let lap = (mirror(i + 1, j) + mirror(i.saturating_sub(1).max(1), j)
                    - 2.0 * x[idx(i, j)])
                    * ix2
                    + (mirror(i, j + 1) + mirror(i, j.saturating_sub(1).max(1))
                        - 2.0 * x[idx(i, j)])
                        * iy2;
                out[idx(i, j)] = -lap;
            }
        }
        for j in 0..ny {
            out[idx(0, j)] = x[idx(0, j)];
            out[idx(nx - 1, j)] = x[idx(nx - 1, j)];
        }
        for i in 0..nx {
            out[idx(i, 0)] = x[idx(i, 0)];
            out[idx(i, ny - 1)] = x[idx(i, ny - 1)];
        }
    };
    let mut b = vec![0.0; nx * ny];
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            b[idx(i, j)] = -(rhs[idx(i, j)] - mean);
        }
    }
    let mut p = cg_generic(&b, apply, tol, max_iter);
    // Mirror onto boundary nodes, corners from the diagonal neighbor.
    for j in 1..ny - 1 {
        p[idx(0, j)] = p[idx(1, j)];
        p[idx(nx - 1, j)] = p[idx(nx - 2, j)];
    }
    for i in 0..nx {
        let i_in = i.clamp(1, nx - 2);
        p[idx(i, 0)] = p[idx(i_in, 1)];
        p[idx(i, ny - 1)] = p[idx(i_in, ny - 2)];
    }
    p
}

// --- Standalone Dirichlet multigrid --------------------------------------

fn dirichlet_residual(u: &[f64], rhs: &[f64], n: usize, h: f64) -> Vec<f64> {
    let inv = 1.0 / (h * h);
    let at = |u: &[f64], i: i64, j: i64| -> f64 {
        if i < 0 || j < 0 || i >= n as i64 || j >= n as i64 {
            0.0
        } else {
            u[(j as usize) * n + i as usize]
        }
    };
    let mut r = vec![0.0; n * n];
    for j in 0..n as i64 {
        for i in 0..n as i64 {
            let lap = (at(u, i - 1, j) + at(u, i + 1, j) + at(u, i, j - 1) + at(u, i, j + 1)
                - 4.0 * at(u, i, j))
                * inv;
            r[(j as usize) * n + i as usize] = rhs[(j as usize) * n + i as usize] - lap;
        }
    }
    r
}

fn dirichlet_smooth(u: &mut [f64], rhs: &[f64], n: usize, h: f64, sweeps: usize) {
    let h2 = h * h;
    for _ in 0..sweeps {
        for j in 0..n {
            for i in 0..n {
                let c = j * n + i;
                let mut s = 0.0;
                if i > 0 {
                    s += u[c - 1];
                }
                if i + 1 < n {
                    s += u[c + 1];
                }
                if j > 0 {
                    s += u[c - n];
                }
                if j + 1 < n {
                    s += u[c + n];
                }
                u[c] = (s - h2 * rhs[c]) / 4.0;
            }
        }
    }
}

fn mg_cycle(rhs: &[f64], n: usize, h: f64, levels: usize, pre: usize, post: usize) -> Vec<f64> {
    let mut u = vec![0.0; n * n];
    if levels == 0 || n < 3 {
        dirichlet_smooth(&mut u, rhs, n, h, 60);
        return u;
    }
    dirichlet_smooth(&mut u, rhs, n, h, pre);
    let res = dirichlet_residual(&u, rhs, n, h);
    // Full-weighting restriction onto the (n-1)/2 grid.
    let nc = (n - 1) / 2;
    if nc < 1 {
        dirichlet_smooth(&mut u, rhs, n, h, 40);
        return u;
    }
    let mut coarse_rhs = vec![0.0; nc * nc];
    for jc in 0..nc {
        for ic in 0..nc {
            let (i, j) = (2 * ic + 1, 2 * jc + 1);
            let at = |di: i64, dj: i64| -> f64 {
                let ii = i as i64 + di;
                let jj = j as i64 + dj;
                if ii < 0 || jj < 0 || ii >= n as i64 || jj >= n as i64 {
                    0.0
                } else {
                    res[(jj as usize) * n + ii as usize]
                }
            };
            coarse_rhs[jc * nc + ic] = 0.25 * at(0, 0)
                + 0.125 * (at(-1, 0) + at(1, 0) + at(0, -1) + at(0, 1))
                + 0.0625 * (at(-1, -1) + at(1, -1) + at(-1, 1) + at(1, 1));
        }
    }
    let coarse_u = mg_cycle(&coarse_rhs, nc, 2.0 * h, levels - 1, pre, post);
    // Bilinear prolongation and correction.
    for j in 0..n {
        for i in 0..n {
            // Coarse coordinates of fine node (i, j): coarse node ic sits
            // at fine 2ic+1.
            let x = (i as f64 - 1.0) / 2.0;
            let y = (j as f64 - 1.0) / 2.0;
            let i0 = x.floor();
            let j0 = y.floor();
            let fx = x - i0;
            let fy = y - j0;
            let cval = |ic: f64, jc: f64| -> f64 {
                if ic < 0.0 || jc < 0.0 || ic >= nc as f64 || jc >= nc as f64 {
                    0.0
                } else {
                    coarse_u[(jc as usize) * nc + ic as usize]
                }
            };
            let interp = cval(i0, j0) * (1.0 - fx) * (1.0 - fy)
                + cval(i0 + 1.0, j0) * fx * (1.0 - fy)
                + cval(i0, j0 + 1.0) * (1.0 - fx) * fy
                + cval(i0 + 1.0, j0 + 1.0) * fx * fy;
            u[j * n + i] += interp;
        }
    }
    dirichlet_smooth(&mut u, rhs, n, h, post);
    u
}

/// One geometric multigrid V-cycle for the Dirichlet Poisson problem
/// ∇²u = rhs on the unit square (rhs is an n × n interior-node grid with
/// n = √len, spacing h = 1/(n+1)); returns the V-cycle approximation to
/// the solution from a zero initial guess. Iterate on the residual for a
/// full solve.
#[must_use]
pub fn multigrid_vcycle(rhs: &[f64], levels: usize, pre: usize, post: usize) -> Vec<f64> {
    let n = (rhs.len() as f64).sqrt().round() as usize;
    assert_eq!(n * n, rhs.len(), "rhs must be a square grid");
    let h = 1.0 / (n + 1) as f64;
    mg_cycle(rhs, n, h, levels, pre, post)
}

// --- The solver ----------------------------------------------------------

/// 2D stable-fluids solver.
pub struct StableFluid2 {
    pub grid: MacGrid2,
    pub density: CellField2,
    pub temperature: CellField2,
    pub viscosity: f64,
    pub buoyancy: f64,
    pub vorticity_confinement: f64,
    pressure_solver: PressureSolver,
    /// Domain boundary condition.
    pub bc: FluidBc,
    /// Tangential velocity of the top wall (lid-driven cavity).
    pub lid_velocity: f64,
    /// Thermal diffusivity for the temperature field.
    pub thermal_diffusivity: f64,
}

impl StableFluid2 {
    /// New quiescent solver.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Self {
            grid: MacGrid2::new(nx, ny, dx),
            density: CellField2::new(nx, ny, dx),
            temperature: CellField2::new(nx, ny, dx),
            viscosity: 0.0,
            buoyancy: 0.0,
            vorticity_confinement: 0.0,
            pressure_solver: PressureSolver::Cg { tol: 1e-10, max_iter: 800 },
            bc: FluidBc::NoSlip,
            lid_velocity: 0.0,
            thermal_diffusivity: 0.0,
        }
    }

    /// Choose the pressure solver.
    pub fn set_pressure_solver(&mut self, s: PressureSolver) {
        self.pressure_solver = s;
    }

    /// One full step: advect, buoyancy, diffuse, confine vorticity,
    /// project.
    pub fn step(&mut self, dt: f64) {
        self.advect_velocity_maccormack(dt);
        if self.buoyancy != 0.0 {
            self.apply_buoyancy(dt);
        }
        if self.viscosity > 0.0 {
            self.diffuse(dt);
        }
        if self.vorticity_confinement > 0.0 {
            self.apply_vorticity_confinement(dt);
        }
        self.grid.apply_bc(self.bc);
        self.project();
        self.grid.apply_bc(self.bc);
        // Scalar transport.
        self.density = crate::cfd::advection::advect_maccormack_2d(&self.density, &self.grid, dt);
        self.temperature =
            crate::cfd::advection::advect_maccormack_2d(&self.temperature, &self.grid, dt);
        if self.thermal_diffusivity > 0.0 {
            let kappa = self.thermal_diffusivity;
            self.temperature = diffuse_cell_field(&self.temperature, kappa, dt);
        }
    }

    fn advect_velocity_maccormack(&mut self, dt: f64) {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let bicubic = |data: &[f64], ni: usize, nj: usize, gx: f64, gy: f64| -> f64 {
            let x = gx.clamp(0.0, (ni - 1) as f64);
            let y = gy.clamp(0.0, (nj - 1) as f64);
            let i = (x.floor() as usize).min(ni.saturating_sub(2));
            let j = (y.floor() as usize).min(nj.saturating_sub(2));
            let (fx, fy) = (x - i as f64, y - j as f64);
            let cat = |m1: f64, p0: f64, p1: f64, p2: f64, t: f64| -> f64 {
                let a = 0.5 * (-m1 + 3.0 * p0 - 3.0 * p1 + p2);
                let b = m1 - 2.5 * p0 + 2.0 * p1 - 0.5 * p2;
                let c = 0.5 * (p1 - m1);
                ((a * t + b) * t + c) * t + p0
            };
            let gi = |ii: i64, jj: i64| -> f64 {
                let ii = ii.clamp(0, ni as i64 - 1) as usize;
                let jj = jj.clamp(0, nj as i64 - 1) as usize;
                data[jj * ni + ii]
            };
            let rows: [f64; 4] = std::array::from_fn(|k| {
                let dj = k as i64 - 1;
                cat(
                    gi(i as i64 - 1, j as i64 + dj),
                    gi(i as i64, j as i64 + dj),
                    gi(i as i64 + 1, j as i64 + dj),
                    gi(i as i64 + 2, j as i64 + dj),
                    fx,
                )
            });
            cat(rows[0], rows[1], rows[2], rows[3], fy)
        };
        let g = &self.grid;
        let advect = |data: &[f64], ni: usize, nj: usize, ox: f64, oy: f64| -> Vec<f64> {
            let mut out = vec![0.0; ni * nj];
            for j in 0..nj {
                for i in 0..ni {
                    let p = Vec2::new((i as f64 + ox) * dx, (j as f64 + oy) * dx);
                    let vmid = g.velocity_at(p - g.velocity_at(p) * (0.5 * dt));
                    let back = p - vmid * dt;
                    let bgx = back.x / dx - ox;
                    let bgy = back.y / dx - oy;
                    let fwd = bicubic(data, ni, nj, bgx, bgy);
                    // Retrace forward from the backtraced point.
                    let vmid2 = g.velocity_at(back + g.velocity_at(back) * (0.5 * dt));
                    let again = back + vmid2 * dt;
                    let orig = data[j * ni + i];
                    let err =
                        0.5 * (orig - bicubic(data, ni, nj, again.x / dx - ox, again.y / dx - oy));
                    // Clamp to the 4 lattice values around the backtrace.
                    let i0 = (bgx.clamp(0.0, (ni - 1) as f64).floor() as usize)
                        .min(ni.saturating_sub(2));
                    let j0 = (bgy.clamp(0.0, (nj - 1) as f64).floor() as usize)
                        .min(nj.saturating_sub(2));
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for dj in 0..2 {
                        for di in 0..2 {
                            let v = data[(j0 + dj) * ni + (i0 + di).min(ni - 1)];
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                    // Soft clamp: permit a half-range overshoot so smooth
                    // extrema (vortex cores) are not flattened.
                    let margin = 0.5 * (hi - lo);
                    out[j * ni + i] = (fwd + err).clamp(lo - margin, hi + margin);
                }
            }
            out
        };
        let new_u = advect(&g.u, nx + 1, ny, 0.0, 0.5);
        let new_v = advect(&g.v, nx, ny + 1, 0.5, 0.0);
        self.grid.u = new_u;
        self.grid.v = new_v;
    }

    fn apply_buoyancy(&mut self, dt: f64) {
        let (nx, ny) = (self.grid.nx, self.grid.ny);
        let t_mean =
            self.temperature.data.iter().sum::<f64>() / self.temperature.data.len() as f64;
        for j in 1..ny {
            for i in 0..nx {
                // Temperature at the v face (i, j).
                let t = 0.5 * (self.temperature.at(i, j - 1) + self.temperature.at(i, j));
                let idx = self.grid.v_idx(i, j);
                self.grid.v[idx] += dt * self.buoyancy * (t - t_mean);
            }
        }
    }

    /// Implicit viscosity solve (I − ν dt ∇²) u = u, per component; the
    /// wall ghosts follow the domain BC (no-slip mirrors with sign flip,
    /// free-slip mirrors) and the moving lid enters as a source term.
    pub fn diffuse(&mut self, dt: f64) {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let lam = self.viscosity * dt / (dx * dx);
        let wall_sign = if self.bc == FluidBc::NoSlip { -1.0 } else { 1.0 };
        // u component: interior unknowns i in 1..nx (boundary faces are
        // fixed 0), j in 0..ny with wall ghosts.
        let apply_u = |x: &[f64], out: &mut [f64]| {
            for j in 0..ny {
                for i in 0..=nx {
                    let c = j * (nx + 1) + i;
                    if i == 0 || i == nx {
                        out[c] = x[c];
                        continue;
                    }
                    let left = x[c - 1];
                    let right = x[c + 1];
                    // Wall-tangential ghosts at bottom/top.
                    let down = if j > 0 { x[c - (nx + 1)] } else { wall_sign * x[c] };
                    let up = if j + 1 < ny { x[c + (nx + 1)] } else { wall_sign * x[c] };
                    out[c] = x[c] - lam * (left + right + down + up - 4.0 * x[c]);
                }
            }
        };
        // RHS: u + lid source (top ghost = 2*lid - u ⇒ constant 2*lid).
        let mut rhs_u = self.grid.u.clone();
        if self.lid_velocity != 0.0 {
            for i in 1..nx {
                let c = (ny - 1) * (nx + 1) + i;
                rhs_u[c] += lam * 2.0 * self.lid_velocity;
            }
        }
        self.grid.u = cg_generic(&rhs_u, apply_u, 1e-10, 400);
        let apply_v = |x: &[f64], out: &mut [f64]| {
            for j in 0..=ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    if j == 0 || j == ny {
                        out[c] = x[c];
                        continue;
                    }
                    let down = x[c - nx];
                    let up = x[c + nx];
                    let left = if i > 0 { x[c - 1] } else { wall_sign * x[c] };
                    let right = if i + 1 < nx { x[c + 1] } else { wall_sign * x[c] };
                    out[c] = x[c] - lam * (left + right + down + up - 4.0 * x[c]);
                }
            }
        };
        let rhs_v = self.grid.v.clone();
        self.grid.v = cg_generic(&rhs_v, apply_v, 1e-10, 400);
    }

    /// Vorticity confinement force ε dx (N × ω).
    pub fn apply_vorticity_confinement(&mut self, dt: f64) {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let w = self.grid.curl();
        let mag: Vec<f64> = w.iter().map(|v| v.abs()).collect();
        let at = |f: &[f64], i: i64, j: i64| -> f64 {
            let i = i.clamp(0, nx as i64 - 1) as usize;
            let j = j.clamp(0, ny as i64 - 1) as usize;
            f[j * nx + i]
        };
        let eps = self.vorticity_confinement;
        // Force at cell centers, then splat to faces.
        let mut fx = vec![0.0; nx * ny];
        let mut fy = vec![0.0; nx * ny];
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let gx = (at(&mag, i + 1, j) - at(&mag, i - 1, j)) / (2.0 * dx);
                let gy = (at(&mag, i, j + 1) - at(&mag, i, j - 1)) / (2.0 * dx);
                let norm = (gx * gx + gy * gy).sqrt().max(1e-12);
                let (nxv, nyv) = (gx / norm, gy / norm);
                let wc = at(&w, i, j);
                fx[(j as usize) * nx + i as usize] = eps * dx * nyv * wc;
                fy[(j as usize) * nx + i as usize] = -eps * dx * nxv * wc;
            }
        }
        for j in 0..ny {
            for i in 1..nx {
                let idx = self.grid.u_idx(i, j);
                self.grid.u[idx] += dt * 0.5 * (fx[j * nx + i - 1] + fx[j * nx + i]);
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let idx = self.grid.v_idx(i, j);
                self.grid.v[idx] += dt * 0.5 * (fy[(j - 1) * nx + i] + fy[j * nx + i]);
            }
        }
    }

    /// Pressure projection: solve the Neumann Poisson problem for the
    /// divergence and subtract the pressure gradient.
    pub fn project(&mut self) {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let div = self.grid.divergence();
        let p = match self.pressure_solver {
            PressureSolver::Cg { tol, max_iter } => {
                pressure_poisson_cg(&div, &self.grid.solid, nx, ny, dx, tol, max_iter)
            }
            PressureSolver::Pcg => {
                pressure_poisson_cg(&div, &self.grid.solid, nx, ny, dx, 1e-10, 800)
            }
            PressureSolver::Jacobi(iters) => self.relax_pressure(&div, iters, false),
            PressureSolver::GaussSeidel(iters) => self.relax_pressure(&div, iters, true),
            PressureSolver::Fft => {
                if self.grid.solid.iter().any(|&s| s) {
                    pressure_poisson_cg(&div, &self.grid.solid, nx, ny, dx, 1e-10, 800)
                } else {
                    poisson_neumann_dct(&div, nx, ny, dx)
                }
            }
            PressureSolver::Multigrid(cycles) => {
                if self.grid.solid.iter().any(|&s| s) {
                    pressure_poisson_cg(&div, &self.grid.solid, nx, ny, dx, 1e-10, 800)
                } else {
                    neumann_multigrid(&div, nx, ny, dx, cycles)
                }
            }
        };
        self.grid.p = p;
        // Subtract the gradient on faces between two fluid cells.
        for j in 0..ny {
            for i in 1..nx {
                let (cl, cr) = (self.grid.c_idx(i - 1, j), self.grid.c_idx(i, j));
                if !self.grid.solid[cl] && !self.grid.solid[cr] {
                    let idx = self.grid.u_idx(i, j);
                    self.grid.u[idx] -= (self.grid.p[cr] - self.grid.p[cl]) / dx;
                }
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let (cb, ct) = (self.grid.c_idx(i, j - 1), self.grid.c_idx(i, j));
                if !self.grid.solid[cb] && !self.grid.solid[ct] {
                    let idx = self.grid.v_idx(i, j);
                    self.grid.v[idx] -= (self.grid.p[ct] - self.grid.p[cb]) / dx;
                }
            }
        }
    }

    fn relax_pressure(&self, div: &[f64], iters: usize, gauss_seidel: bool) -> Vec<f64> {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let solid = &self.grid.solid;
        let h2 = dx * dx;
        let mut p = vec![0.0; nx * ny];
        let mut next = vec![0.0; nx * ny];
        for _ in 0..iters {
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    if solid[c] {
                        continue;
                    }
                    let src: &[f64] = if gauss_seidel { &next } else { &p };
                    let prev: &[f64] = &p;
                    let mut acc = 0.0;
                    let mut count = 0.0;
                    // Gauss-Seidel uses already-updated west/south values.
                    if i > 0 && !solid[c - 1] {
                        acc += src[c - 1];
                        count += 1.0;
                    }
                    if j > 0 && !solid[c - nx] {
                        acc += src[c - nx];
                        count += 1.0;
                    }
                    if i + 1 < nx && !solid[c + 1] {
                        acc += prev[c + 1];
                        count += 1.0;
                    }
                    if j + 1 < ny && !solid[c + nx] {
                        acc += prev[c + nx];
                        count += 1.0;
                    }
                    next[c] = if count > 0.0 { (acc - h2 * div[c]) / count } else { 0.0 };
                }
            }
            std::mem::swap(&mut p, &mut next);
            if gauss_seidel {
                next.copy_from_slice(&p);
            }
        }
        p
    }

    /// Splat density into a Gaussian blob at world (x, y).
    pub fn add_density(&mut self, x: f64, y: f64, amount: f64) {
        splat(&mut self.density, x, y, amount);
    }

    /// Splat heat (temperature) at world (x, y).
    pub fn add_heat(&mut self, x: f64, y: f64, amount: f64) {
        splat(&mut self.temperature, x, y, amount);
    }

    /// Add velocity in a Gaussian blob at world (x, y).
    pub fn add_velocity(&mut self, x: f64, y: f64, v: Vec2) {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let sigma = 2.0 * dx;
        for j in 0..ny {
            for i in 0..=nx {
                let px = i as f64 * dx;
                let py = (j as f64 + 0.5) * dx;
                let w = (-((px - x).powi(2) + (py - y).powi(2)) / (2.0 * sigma * sigma)).exp();
                let idx = self.grid.u_idx(i, j);
                self.grid.u[idx] += v.x * w;
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                let px = (i as f64 + 0.5) * dx;
                let py = j as f64 * dx;
                let w = (-((px - x).powi(2) + (py - y).powi(2)) / (2.0 * sigma * sigma)).exp();
                let idx = self.grid.v_idx(i, j);
                self.grid.v[idx] += v.y * w;
            }
        }
    }

    /// Largest cell divergence magnitude.
    #[must_use]
    pub fn divergence_max(&self) -> f64 {
        self.grid.divergence().iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
    }

    /// Integrate streamlines from seed points (RK2, fixed step).
    #[must_use]
    pub fn streamlines(&self, seeds: &[Vec2], steps: usize, dt: f64) -> Vec<Vec<Vec2>> {
        seeds
            .iter()
            .map(|&s| {
                let mut p = s;
                let mut line = vec![p];
                for _ in 0..steps {
                    let vmid = self.grid.velocity_at(p + self.grid.velocity_at(p) * (0.5 * dt));
                    p = p + vmid * dt;
                    line.push(p);
                }
                line
            })
            .collect()
    }

    /// Advect passive tracer particles one step (RK2).
    pub fn particles_advect(&self, pts: &mut [Vec2], dt: f64) {
        for p in pts.iter_mut() {
            let vmid = self.grid.velocity_at(*p + self.grid.velocity_at(*p) * (0.5 * dt));
            *p = *p + vmid * dt;
        }
    }

    /// Pressure force on the solid cells (unit density): F = Σ p n dx,
    /// with n the outward normal of the fluid at each solid boundary
    /// face.
    #[must_use]
    pub fn drag_on_solid(&self) -> Vec2 {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let mut f = Vec2::ZERO;
        for j in 0..ny {
            for i in 0..nx {
                if !self.grid.solid[self.grid.c_idx(i, j)] {
                    continue;
                }
                // Fluid neighbors push on this solid cell.
                if i > 0 && !self.grid.solid[self.grid.c_idx(i - 1, j)] {
                    f.x += self.grid.p[self.grid.c_idx(i - 1, j)] * dx;
                }
                if i + 1 < nx && !self.grid.solid[self.grid.c_idx(i + 1, j)] {
                    f.x -= self.grid.p[self.grid.c_idx(i + 1, j)] * dx;
                }
                if j > 0 && !self.grid.solid[self.grid.c_idx(i, j - 1)] {
                    f.y += self.grid.p[self.grid.c_idx(i, j - 1)] * dx;
                }
                if j + 1 < ny && !self.grid.solid[self.grid.c_idx(i, j + 1)] {
                    f.y -= self.grid.p[self.grid.c_idx(i, j + 1)] * dx;
                }
            }
        }
        f
    }

    /// Lift (transverse pressure force) on the solid.
    #[must_use]
    pub fn lift_on_solid(&self) -> f64 {
        self.drag_on_solid().y
    }

    /// Cell-centered vorticity as a field.
    #[must_use]
    pub fn vorticity_field(&self) -> CellField2 {
        let mut f = CellField2::new(self.grid.nx, self.grid.ny, self.grid.dx);
        f.data = self.grid.curl();
        f
    }

    /// Stream function ψ with ∇²ψ = −ω and ψ = 0 on the boundary.
    #[must_use]
    pub fn stream_function(&self) -> CellField2 {
        let (nx, ny, dx) = (self.grid.nx, self.grid.ny, self.grid.dx);
        let w = self.grid.curl();
        let rhs: Vec<f64> = w.iter().map(|v| -v).collect();
        // Dirichlet CG.
        let apply = |x: &[f64], out: &mut [f64]| {
            let inv = 1.0 / (dx * dx);
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    let mut acc = 4.0 * x[c];
                    if i > 0 {
                        acc -= x[c - 1];
                    }
                    if i + 1 < nx {
                        acc -= x[c + 1];
                    }
                    if j > 0 {
                        acc -= x[c - nx];
                    }
                    if j + 1 < ny {
                        acc -= x[c + nx];
                    }
                    out[c] = acc * inv;
                }
            }
        };
        let b: Vec<f64> = rhs.iter().map(|v| -v).collect();
        let mut f = CellField2::new(nx, ny, dx);
        f.data = cg_generic(&b, apply, 1e-10, 2000);
        f
    }
}

fn splat(field: &mut CellField2, x: f64, y: f64, amount: f64) {
    let (nx, ny, dx) = (field.nx, field.ny, field.dx);
    let sigma = 2.0 * dx;
    for j in 0..ny {
        for i in 0..nx {
            let px = (i as f64 + 0.5) * dx;
            let py = (j as f64 + 0.5) * dx;
            let w = (-((px - x).powi(2) + (py - y).powi(2)) / (2.0 * sigma * sigma)).exp();
            field.data[j * nx + i] += amount * w;
        }
    }
}

fn diffuse_cell_field(f: &CellField2, kappa: f64, dt: f64) -> CellField2 {
    let (nx, ny, dx) = (f.nx, f.ny, f.dx);
    let lam = kappa * dt / (dx * dx);
    let apply = move |x: &[f64], out: &mut [f64]| {
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                // Neumann (insulated) walls.
                let left = if i > 0 { x[c - 1] } else { x[c] };
                let right = if i + 1 < nx { x[c + 1] } else { x[c] };
                let down = if j > 0 { x[c - nx] } else { x[c] };
                let up = if j + 1 < ny { x[c + nx] } else { x[c] };
                out[c] = x[c] - lam * (left + right + down + up - 4.0 * x[c]);
            }
        }
    };
    let mut out = f.clone();
    out.data = cg_generic(&f.data, apply, 1e-10, 400);
    out
}

/// Generic CG for SPD operators given as closures.
fn cg_generic(b: &[f64], apply: impl Fn(&[f64], &mut [f64]), tol: f64, max_iter: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut d = r.clone();
    let mut ad = vec![0.0; n];
    let mut rs: f64 = r.iter().map(|v| v * v).sum();
    let b_norm = rs.sqrt().max(1e-300);
    for _ in 0..max_iter {
        if rs.sqrt() / b_norm < tol {
            break;
        }
        apply(&d, &mut ad);
        let dad: f64 = d.iter().zip(&ad).map(|(a, b)| a * b).sum();
        if dad.abs() < 1e-300 {
            break;
        }
        let alpha = rs / dad;
        for i in 0..n {
            x[i] += alpha * d[i];
            r[i] -= alpha * ad[i];
        }
        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rs_new / rs;
        for i in 0..n {
            d[i] = r[i] + beta * d[i];
        }
        rs = rs_new;
    }
    x
}

/// Direct Neumann Poisson solve via DCT-II eigenvalues (no solids).
fn poisson_neumann_dct(rhs: &[f64], nx: usize, ny: usize, dx: f64) -> Vec<f64> {
    use crate::transforms::dct::{dct_ii, dct_iii};
    // Forward DCT-II along rows then columns.
    let mut coef = vec![0.0; nx * ny];
    for j in 0..ny {
        let row: Vec<f64> = (0..nx).map(|i| rhs[j * nx + i]).collect();
        let t = dct_ii(&row);
        for i in 0..nx {
            coef[j * nx + i] = t[i];
        }
    }
    for i in 0..nx {
        let col: Vec<f64> = (0..ny).map(|j| coef[j * nx + i]).collect();
        let t = dct_ii(&col);
        for j in 0..ny {
            coef[j * nx + i] = t[j];
        }
    }
    // Divide by the discrete Neumann Laplacian eigenvalues.
    for j in 0..ny {
        for i in 0..nx {
            let lx = 2.0 * ((PI * i as f64 / (2.0 * nx as f64)).sin()).powi(2) * 2.0;
            let ly = 2.0 * ((PI * j as f64 / (2.0 * ny as f64)).sin()).powi(2) * 2.0;
            let eig = -(lx + ly) / (dx * dx);
            let c = j * nx + i;
            if i == 0 && j == 0 {
                coef[c] = 0.0;
            } else {
                coef[c] /= eig;
            }
        }
    }
    // Inverse: DCT-III with 1/(2n) scaling per axis.
    let mut p = coef;
    for i in 0..nx {
        let col: Vec<f64> = (0..ny).map(|j| p[j * nx + i]).collect();
        let t = dct_iii(&col);
        for j in 0..ny {
            p[j * nx + i] = t[j] / (2.0 * ny as f64);
        }
    }
    for j in 0..ny {
        let row: Vec<f64> = (0..nx).map(|i| p[j * nx + i]).collect();
        let t = dct_iii(&row);
        for i in 0..nx {
            p[j * nx + i] = t[i] / (2.0 * nx as f64);
        }
    }
    p
}

/// Neumann multigrid for the pressure problem (square-ish grids, no
/// solids): V-cycles with damped Jacobi smoothing.
fn neumann_multigrid(rhs: &[f64], nx: usize, ny: usize, dx: f64, cycles: usize) -> Vec<f64> {
    // Iterate CG-quality solutions cheaply: reuse the Dirichlet MG
    // machinery on the mean-removed problem is not directly applicable
    // (different BCs), so run V-cycles with a Neumann smoother.
    let n = nx * ny;
    let mean = rhs.iter().sum::<f64>() / n as f64;
    let b: Vec<f64> = rhs.iter().map(|v| v - mean).collect();
    let solid = vec![false; n];
    let mut p = vec![0.0; n];
    let mut r = vec![0.0; n];
    for _ in 0..cycles {
        // Residual of ∇²p = b: r = b − ∇²p = b + A p (A = −∇²).
        apply_neumann_laplacian(&p, &mut r, &solid, nx, ny, dx);
        for i in 0..n {
            r[i] += b[i];
        }
        let corr = neumann_vcycle(&r, nx, ny, dx, 3);
        for i in 0..n {
            p[i] += corr[i];
        }
        let pm = p.iter().sum::<f64>() / n as f64;
        p.iter_mut().for_each(|v| *v -= pm);
    }
    p
}

fn neumann_smooth(u: &mut [f64], rhs: &[f64], nx: usize, ny: usize, dx: f64, sweeps: usize) {
    let h2 = dx * dx;
    for _ in 0..sweeps {
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                let mut acc = 0.0;
                let mut count = 0.0_f64;
                if i > 0 {
                    acc += u[c - 1];
                    count += 1.0;
                }
                if i + 1 < nx {
                    acc += u[c + 1];
                    count += 1.0;
                }
                if j > 0 {
                    acc += u[c - nx];
                    count += 1.0;
                }
                if j + 1 < ny {
                    acc += u[c + nx];
                    count += 1.0;
                }
                u[c] = (acc - h2 * rhs[c]) / count.max(1.0);
            }
        }
    }
}

fn neumann_vcycle(b: &[f64], nx: usize, ny: usize, dx: f64, depth: usize) -> Vec<f64> {
    // Solve A u = -b i.e. ∇²u = b approximately.
    let mut u = vec![0.0; nx * ny];
    if depth == 0 || nx < 4 || ny < 4 || !nx.is_multiple_of(2) || !ny.is_multiple_of(2) {
        neumann_smooth(&mut u, b, nx, ny, dx, 40);
        return u;
    }
    neumann_smooth(&mut u, b, nx, ny, dx, 3);
    // Residual of ∇²u = b.
    let solid = vec![false; nx * ny];
    let mut au = vec![0.0; nx * ny];
    apply_neumann_laplacian(&u, &mut au, &solid, nx, ny, dx); // A = −∇²
    let res: Vec<f64> = (0..nx * ny).map(|i| b[i] + au[i]).collect(); // b − ∇²u
    // Restrict by 2×2 averaging.
    let (cx, cy) = (nx / 2, ny / 2);
    let mut cb = vec![0.0; cx * cy];
    for j in 0..cy {
        for i in 0..cx {
            cb[j * cx + i] = 0.25
                * (res[(2 * j) * nx + 2 * i]
                    + res[(2 * j) * nx + 2 * i + 1]
                    + res[(2 * j + 1) * nx + 2 * i]
                    + res[(2 * j + 1) * nx + 2 * i + 1]);
        }
    }
    let cu = neumann_vcycle(&cb, cx, cy, 2.0 * dx, depth - 1);
    // Prolong by injection (piecewise constant) and correct.
    for j in 0..ny {
        for i in 0..nx {
            u[j * nx + i] += cu[(j / 2) * cx + i / 2];
        }
    }
    neumann_smooth(&mut u, b, nx, ny, dx, 3);
    u
}

// --- 3D solver -----------------------------------------------------------

/// Minimal 3D stable-fluids solver (semi-Lagrangian advection + CG
/// projection).
pub struct StableFluid3 {
    pub grid: MacGrid3,
    pub density: Vec<f64>,
    pub buoyancy: f64,
    pub temperature: Vec<f64>,
}

impl StableFluid3 {
    /// New quiescent 3D solver.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self {
            grid: MacGrid3::new(nx, ny, nz, dx),
            density: vec![0.0; nx * ny * nz],
            buoyancy: 0.0,
            temperature: vec![0.0; nx * ny * nz],
        }
    }

    /// One step: semi-Lagrangian velocity advection, buoyancy, project.
    pub fn step(&mut self, dt: f64) {
        let (nx, ny, nz, dx) = (self.grid.nx, self.grid.ny, self.grid.nz, self.grid.dx);
        // Advect each face component.
        let g = &self.grid;
        let mut new_u = g.u.clone();
        let mut new_v = g.v.clone();
        let mut new_w = g.w.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let p = Vec3::new(i as f64 * dx, (j as f64 + 0.5) * dx, (k as f64 + 0.5) * dx);
                    let back = p - g.velocity_at(p) * dt;
                    new_u[(k * ny + j) * (nx + 1) + i] = g.velocity_at(back).x;
                }
            }
        }
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    let p = Vec3::new((i as f64 + 0.5) * dx, j as f64 * dx, (k as f64 + 0.5) * dx);
                    let back = p - g.velocity_at(p) * dt;
                    new_v[(k * (ny + 1) + j) * nx + i] = g.velocity_at(back).y;
                }
            }
        }
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = Vec3::new((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx, k as f64 * dx);
                    let back = p - g.velocity_at(p) * dt;
                    new_w[(k * ny + j) * nx + i] = g.velocity_at(back).z;
                }
            }
        }
        self.grid.u = new_u;
        self.grid.v = new_v;
        self.grid.w = new_w;
        // Buoyancy on w faces? Use +y as up for consistency with 2D: v.
        if self.buoyancy != 0.0 {
            let t_mean = self.temperature.iter().sum::<f64>() / self.temperature.len() as f64;
            for k in 0..nz {
                for j in 1..ny {
                    for i in 0..nx {
                        let t0 = self.temperature[(k * ny + j - 1) * nx + i];
                        let t1 = self.temperature[(k * ny + j) * nx + i];
                        self.grid.v[(k * (ny + 1) + j) * nx + i] +=
                            dt * self.buoyancy * (0.5 * (t0 + t1) - t_mean);
                    }
                }
            }
        }
        // Zero domain-normal faces then project.
        self.apply_no_penetration();
        self.project();
        self.apply_no_penetration();
    }

    fn apply_no_penetration(&mut self) {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        for k in 0..nz {
            for j in 0..ny {
                self.grid.u[(k * ny + j) * (nx + 1)] = 0.0;
                self.grid.u[(k * ny + j) * (nx + 1) + nx] = 0.0;
            }
        }
        for k in 0..nz {
            for i in 0..nx {
                self.grid.v[(k * (ny + 1)) * nx + i] = 0.0;
                self.grid.v[(k * (ny + 1) + ny) * nx + i] = 0.0;
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                self.grid.w[(j) * nx + i] = 0.0;
                self.grid.w[(nz * ny + j) * nx + i] = 0.0;
            }
        }
    }

    /// PCG pressure projection.
    pub fn project(&mut self) {
        let (nx, ny, nz, dx) = (self.grid.nx, self.grid.ny, self.grid.nz, self.grid.dx);
        let div = self.grid.divergence();
        let n = nx * ny * nz;
        let mean = div.iter().sum::<f64>() / n as f64;
        let b: Vec<f64> = div.iter().map(|v| -(v - mean)).collect();
        let apply = |x: &[f64], out: &mut [f64]| {
            let inv = 1.0 / (dx * dx);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let c = (k * ny + j) * nx + i;
                        let mut acc = 0.0;
                        let mut count = 0.0;
                        if i > 0 {
                            acc += x[c - 1];
                            count += 1.0;
                        }
                        if i + 1 < nx {
                            acc += x[c + 1];
                            count += 1.0;
                        }
                        if j > 0 {
                            acc += x[c - nx];
                            count += 1.0;
                        }
                        if j + 1 < ny {
                            acc += x[c + nx];
                            count += 1.0;
                        }
                        if k > 0 {
                            acc += x[c - nx * ny];
                            count += 1.0;
                        }
                        if k + 1 < nz {
                            acc += x[c + nx * ny];
                            count += 1.0;
                        }
                        out[c] = (count * x[c] - acc) * inv;
                    }
                }
            }
        };
        let p = cg_generic(&b, apply, 1e-8, 600);
        for k in 0..nz {
            for j in 0..ny {
                for i in 1..nx {
                    let (cl, cr) = ((k * ny + j) * nx + i - 1, (k * ny + j) * nx + i);
                    self.grid.u[(k * ny + j) * (nx + 1) + i] -= (p[cr] - p[cl]) / dx;
                }
            }
        }
        for k in 0..nz {
            for j in 1..ny {
                for i in 0..nx {
                    let (cb, ct) = ((k * ny + j - 1) * nx + i, (k * ny + j) * nx + i);
                    self.grid.v[(k * (ny + 1) + j) * nx + i] -= (p[ct] - p[cb]) / dx;
                }
            }
        }
        for k in 1..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (cb, ct) = (((k - 1) * ny + j) * nx + i, (k * ny + j) * nx + i);
                    self.grid.w[(k * ny + j) * nx + i] -= (p[ct] - p[cb]) / dx;
                }
            }
        }
        self.grid.p = p;
    }
}

// --- Benchmarks ----------------------------------------------------------

/// Lid-driven cavity at Reynolds number `re` on an n × n unit box,
/// stepped to `t_end` (lid speed 1).
#[must_use]
pub fn lid_driven_cavity(n: usize, re: f64, t_end: f64) -> StableFluid2 {
    let dx = 1.0 / n as f64;
    let mut f = StableFluid2::new(n, n, dx);
    f.viscosity = 1.0 / re;
    f.lid_velocity = 1.0;
    f.bc = FluidBc::NoSlip;
    let dt = (0.6 * dx).min(0.02);
    let steps = (t_end / dt).ceil() as usize;
    for _ in 0..steps {
        f.step(dt);
    }
    f
}

/// Uniform inflow past a circular cylinder (diameter ~ny/5 cells) at
/// Reynolds number `re` (inflow speed 1).
#[must_use]
pub fn flow_past_cylinder(nx: usize, ny: usize, re: f64) -> StableFluid2 {
    let dx = 1.0 / ny as f64;
    let mut f = StableFluid2::new(nx, ny, dx);
    let d = 0.2; // cylinder diameter in domain units
    f.viscosity = d / re;
    f.bc = FluidBc::Inflow(Vec2::new(1.0, 0.0));
    f.grid.set_solid_circle(0.3 * nx as f64 * dx, 0.5, 0.5 * d);
    // Start with uniform flow plus a slight asymmetry to trigger
    // shedding.
    for j in 0..ny {
        for i in 0..=nx {
            let idx = f.grid.u_idx(i, j);
            f.grid.u[idx] = 1.0;
        }
    }
    for j in 0..=ny {
        for i in 0..nx {
            let idx = f.grid.v_idx(i, j);
            f.grid.v[idx] = 0.01 * ((j as f64) * 0.7).sin();
        }
    }
    f.grid.apply_bc(f.bc);
    f
}

/// Rayleigh-Bénard convection cell: Rayleigh number `ra`, Prandtl `pr`,
/// hot floor and cold ceiling encoded in the initial temperature.
#[must_use]
pub fn rayleigh_benard(nx: usize, ny: usize, ra: f64, pr: f64) -> StableFluid2 {
    let dx = 1.0 / ny as f64;
    let mut f = StableFluid2::new(nx, ny, dx);
    // Nondimensionalization with buoyancy coefficient 1, ΔT = 1, H = 1:
    // ν = sqrt(Pr/Ra), κ = 1/sqrt(Ra Pr).
    f.viscosity = (pr / ra).sqrt();
    f.thermal_diffusivity = 1.0 / (ra * pr).sqrt();
    f.buoyancy = 1.0;
    f.bc = FluidBc::NoSlip;
    f.temperature = CellField2::from_fn(nx, ny, dx, |x, y| {
        (1.0 - y) + 0.01 * (7.3 * x).sin() * (PI * y).sin()
    });
    f
}

/// Taylor-Green vortex in a free-slip unit box:
/// u = sin(πx) cos(πy), v = −cos(πx) sin(πy), decaying as e^{−2π²νt}.
#[must_use]
pub fn taylor_green_vortex(n: usize, nu: f64) -> StableFluid2 {
    let dx = 1.0 / n as f64;
    let mut f = StableFluid2::new(n, n, dx);
    f.viscosity = nu;
    f.bc = FluidBc::FreeSlip;
    for j in 0..n {
        for i in 0..=n {
            let x = i as f64 * dx;
            let y = (j as f64 + 0.5) * dx;
            let idx = f.grid.u_idx(i, j);
            f.grid.u[idx] = (PI * x).sin() * (PI * y).cos();
        }
    }
    for j in 0..=n {
        for i in 0..n {
            let x = (i as f64 + 0.5) * dx;
            let y = j as f64 * dx;
            let idx = f.grid.v_idx(i, j);
            f.grid.v[idx] = -(PI * x).cos() * (PI * y).sin();
        }
    }
    f
}

/// Exact Taylor-Green velocity at (x, y, t).
#[must_use]
pub fn taylor_green_exact(x: f64, y: f64, t: f64, nu: f64) -> Vec2 {
    let decay = (-2.0 * PI * PI * nu * t).exp();
    Vec2::new(
        (PI * x).sin() * (PI * y).cos() * decay,
        -(PI * x).cos() * (PI * y).sin() * decay,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    #[test]
    fn test_projection_kills_divergence() {
        let n = 32;
        let mut f = StableFluid2::new(n, n, 1.0 / n as f64);
        let mut rng = Rng::new(4);
        for u in f.grid.u.iter_mut() {
            *u = 2.0 * rng.next_f64() - 1.0;
        }
        for v in f.grid.v.iter_mut() {
            *v = 2.0 * rng.next_f64() - 1.0;
        }
        f.grid.apply_bc(FluidBc::NoSlip);
        f.set_pressure_solver(PressureSolver::Cg { tol: 1e-12, max_iter: 4000 });
        f.project();
        assert!(f.divergence_max() < 1e-8, "residual divergence {}", f.divergence_max());
        // With a solid obstacle too.
        let mut f2 = StableFluid2::new(n, n, 1.0 / n as f64);
        for u in f2.grid.u.iter_mut() {
            *u = 2.0 * rng.next_f64() - 1.0;
        }
        f2.grid.set_solid_circle(0.5, 0.5, 0.15);
        f2.grid.apply_bc(FluidBc::NoSlip);
        f2.set_pressure_solver(PressureSolver::Cg { tol: 1e-12, max_iter: 4000 });
        f2.project();
        // Divergence in fluid cells only.
        let div = f2.grid.divergence();
        let max_fluid_div = div
            .iter()
            .zip(&f2.grid.solid)
            .filter(|(_, s)| !**s)
            .map(|(d, _)| d.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_fluid_div < 1e-8, "solid-case divergence {max_fluid_div}");
    }

    #[test]
    fn test_pressure_solver_variants_agree() {
        let n = 32;
        let dx = 1.0 / n as f64;
        let mut base = StableFluid2::new(n, n, dx);
        let mut rng = Rng::new(9);
        for u in base.grid.u.iter_mut() {
            *u = 2.0 * rng.next_f64() - 1.0;
        }
        for v in base.grid.v.iter_mut() {
            *v = 2.0 * rng.next_f64() - 1.0;
        }
        base.grid.apply_bc(FluidBc::NoSlip);
        let solvers = [
            PressureSolver::Cg { tol: 1e-12, max_iter: 4000 },
            PressureSolver::Fft,
            PressureSolver::Multigrid(30),
            PressureSolver::GaussSeidel(4000),
        ];
        let mut divs = Vec::new();
        for s in solvers {
            let mut f = StableFluid2::new(n, n, dx);
            f.grid.u = base.grid.u.clone();
            f.grid.v = base.grid.v.clone();
            f.set_pressure_solver(s);
            f.project();
            divs.push(f.divergence_max());
        }
        assert!(divs[0] < 1e-8, "CG {divs:?}");
        assert!(divs[1] < 1e-6, "FFT {divs:?}");
        assert!(divs[2] < 1e-4, "MG {divs:?}");
        assert!(divs[3] < 1e-3, "GS {divs:?}");
    }

    #[test]
    fn test_multigrid_vcycle_convergence() {
        // Dirichlet Poisson: iterate V-cycles, residual to 1e-6 in < 10.
        let n = 63;
        let h = 1.0 / (n + 1) as f64;
        let rhs: Vec<f64> = (0..n * n)
            .map(|c| {
                let (i, j) = (c % n, c / n);
                let x = (i + 1) as f64 * h;
                let y = (j + 1) as f64 * h;
                (2.0 * PI * x).sin() * (PI * y).sin()
            })
            .collect();
        let mut u = vec![0.0; n * n];
        let apply_res = |u: &[f64]| -> Vec<f64> { dirichlet_residual(u, &rhs, n, h) };
        let norm0 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut cycles = 0;
        loop {
            let res = apply_res(&u);
            let rn = res.iter().map(|v| v * v).sum::<f64>().sqrt() / norm0;
            if rn < 1e-6 {
                break;
            }
            assert!(cycles < 10, "multigrid too slow: residual {rn} after {cycles}");
            let corr = multigrid_vcycle(&res, 5, 2, 2);
            u.iter_mut().zip(&corr).for_each(|(a, b)| *a += b);
            cycles += 1;
        }
        assert!(cycles < 10, "took {cycles} cycles");
    }

    #[test]
    fn test_taylor_green_decay() {
        let n = 64;
        let nu = 0.01;
        let mut f = taylor_green_vortex(n, nu);
        f.set_pressure_solver(PressureSolver::Cg { tol: 1e-10, max_iter: 2000 });
        let ke0 = f.grid.kinetic_energy();
        let t_end = 0.2_f64;
        let dt = 0.01;
        let steps = (t_end / dt).round() as usize;
        for _ in 0..steps {
            f.step(dt);
        }
        // Kinetic energy decays as e^{-4π²νt}.
        let ke = f.grid.kinetic_energy();
        let expected = ke0 * (-4.0 * PI * PI * nu * t_end).exp();
        assert!(
            (ke / expected - 1.0).abs() < 0.02,
            "TG decay: KE {ke} vs {expected}"
        );
        // Pointwise check against the exact field.
        let exact = taylor_green_exact(0.3, 0.4, t_end, nu);
        let got = f.grid.velocity_at(Vec2::new(0.3, 0.4));
        assert!((got.x - exact.x).abs() < 0.02, "u {got:?} vs {exact:?}");
        assert!((got.y - exact.y).abs() < 0.02, "v {got:?} vs {exact:?}");
    }

    #[test]
    fn test_lid_driven_cavity_ghia() {
        // Re 100 cavity: u along the vertical centerline vs Ghia (1982).
        let f = lid_driven_cavity(32, 100.0, 8.0);
        let ghia: [(f64, f64); 5] = [
            (0.9688, 0.78871),
            (0.8516, 0.23151),
            (0.5000, -0.20581),
            (0.2813, -0.15662),
            (0.1016, -0.06434),
        ];
        for &(y, u_ref) in &ghia {
            let u = f.grid.velocity_at(Vec2::new(0.5, y)).x;
            assert!(
                (u - u_ref).abs() < 0.05,
                "cavity u({y}) = {u} vs Ghia {u_ref}"
            );
        }
        // Mass (net flux) is conserved: total divergence tiny.
        assert!(f.divergence_max() < 1e-6);
    }

    #[test]
    fn test_buoyancy_and_confinement_and_tools() {
        let n = 24;
        let mut f = StableFluid2::new(n, n, 1.0 / n as f64);
        f.buoyancy = 5.0;
        f.vorticity_confinement = 0.5;
        f.thermal_diffusivity = 1e-4;
        f.add_heat(0.5, 0.25, 3.0);
        f.add_density(0.5, 0.25, 1.0);
        let mass0: f64 = f.density.data.iter().sum();
        for _ in 0..30 {
            f.step(0.01);
        }
        // Heat rises: net upward velocity above the blob.
        let v_above = f.grid.velocity_at(Vec2::new(0.5, 0.5)).y;
        assert!(v_above > 0.0, "plume did not rise: {v_above}");
        // Density mass approximately conserved by limited MacCormack.
        let mass: f64 = f.density.data.iter().sum();
        assert!((mass / mass0 - 1.0).abs() < 0.05, "mass drift {mass} vs {mass0}");
        assert!(f.divergence_max() < 1e-6);
        // Streamlines and particles follow the flow.
        let lines = f.streamlines(&[Vec2::new(0.5, 0.3)], 20, 0.01);
        assert_eq!(lines[0].len(), 21);
        assert!(lines[0][20].y >= lines[0][0].y - 1e-9);
        let mut pts = [Vec2::new(0.5, 0.3)];
        f.particles_advect(&mut pts, 0.01);
        assert!(pts[0].y >= 0.3);
        // Vorticity and stream function fields are consistent in scale.
        let w = f.vorticity_field();
        let psi = f.stream_function();
        assert!(w.data.iter().any(|v| v.abs() > 1e-6));
        assert!(psi.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_add_velocity_injects_the_requested_momentum() {
        let n = 64;
        let dx = 1.0 / n as f64;
        let sigma = 2.0 * dx;
        let (x0, y0) = (0.5, 0.5);
        let v = Vec2::new(0.7, -0.4);

        let mut f = StableFluid2::new(n, n, dx);
        f.add_velocity(x0, y0, v);

        // Every face gets exactly v times the Gaussian weight at that
        // face's own position.
        for j in 0..n {
            for i in 0..=n {
                let (px, py) = (i as f64 * dx, (j as f64 + 0.5) * dx);
                let w = (-((px - x0).powi(2) + (py - y0).powi(2)) / (2.0 * sigma * sigma)).exp();
                assert!(
                    (f.grid.u[f.grid.u_idx(i, j)] - v.x * w).abs() < 1e-15,
                    "u face ({i},{j})"
                );
            }
        }
        for j in 0..=n {
            for i in 0..n {
                let (px, py) = ((i as f64 + 0.5) * dx, j as f64 * dx);
                let w = (-((px - x0).powi(2) + (py - y0).powi(2)) / (2.0 * sigma * sigma)).exp();
                assert!(
                    (f.grid.v[f.grid.v_idx(i, j)] - v.y * w).abs() < 1e-15,
                    "v face ({i},{j})"
                );
            }
        }

        // Integrated momentum: Σ u dx² approximates v_x ∫∫ exp(−r²/2σ²)
        // = v_x · 2πσ². The blob is 8 cells wide and sits well inside the
        // domain, so the midpoint sum is accurate to well under a percent.
        let px: f64 = f.grid.u.iter().sum::<f64>() * dx * dx;
        let py: f64 = f.grid.v.iter().sum::<f64>() * dx * dx;
        let gauss = 2.0 * PI * sigma * sigma;
        assert!(
            (px / (v.x * gauss) - 1.0).abs() < 5e-3,
            "x momentum {px} vs {}",
            v.x * gauss
        );
        assert!(
            (py / (v.y * gauss) - 1.0).abs() < 5e-3,
            "y momentum {py} vs {}",
            v.y * gauss
        );

        // The injection is linear: two calls add, and the peak face value
        // is the requested velocity (weight 1 at the blob centre, which
        // lies on a u face when x0 is a multiple of dx).
        let mut g = StableFluid2::new(n, n, dx);
        g.add_velocity(x0, y0, Vec2::new(0.3, -0.1));
        g.add_velocity(x0, y0, Vec2::new(0.4, -0.3));
        for (a, b) in g.grid.u.iter().zip(&f.grid.u) {
            assert!((a - b).abs() < 1e-15, "add_velocity is not additive: {a} vs {b}");
        }
        for (a, b) in g.grid.v.iter().zip(&f.grid.v) {
            assert!((a - b).abs() < 1e-15);
        }
        let centre_u = f.grid.u[f.grid.u_idx((x0 / dx).round() as usize, n / 2)];
        // The u faces sit at y = (j+0.5)dx, half a cell off the blob
        // centre, so the peak weight is exp(−(dx/2)²/(2σ²)).
        let peak_w = (-(0.5 * dx).powi(2) / (2.0 * sigma * sigma)).exp();
        assert!(
            (centre_u - v.x * peak_w).abs() < 1e-14,
            "peak u {centre_u} vs {}",
            v.x * peak_w
        );

        // Opposite injections cancel exactly.
        let mut z = StableFluid2::new(n, n, dx);
        z.add_velocity(x0, y0, v);
        z.add_velocity(x0, y0, Vec2::new(-v.x, -v.y));
        assert!(z.grid.u.iter().all(|u| u.abs() < 1e-16));
        assert!(z.grid.v.iter().all(|v| v.abs() < 1e-16));

        // A blob of momentum injected into a quiescent, incompressible
        // box is mostly removed by the projection (a pure translation is
        // not a possible motion of an enclosed incompressible fluid), and
        // what survives is divergence free.
        let mut proj = StableFluid2::new(n, n, dx);
        proj.set_pressure_solver(PressureSolver::Cg { tol: 1e-12, max_iter: 4000 });
        proj.add_velocity(x0, y0, v);
        let ke_before = proj.grid.kinetic_energy();
        proj.grid.apply_bc(FluidBc::NoSlip);
        proj.project();
        assert!(proj.divergence_max() < 1e-8, "divergence {}", proj.divergence_max());
        assert!(
            proj.grid.kinetic_energy() < ke_before,
            "projection must remove the compressive part"
        );
    }

    #[test]
    fn test_lift_on_solid_symmetry() {
        // Uniform flow at incidence over a circular obstacle placed on the
        // horizontal mid-line of the box. Reflecting the box about that
        // line maps the +alpha problem onto the −alpha one, so the lift is
        // an odd function of the incidence and the drag an even one.
        let (nx, ny) = (64_usize, 48_usize);
        let dx = 1.0 / ny as f64;
        let force_at = |alpha: f64| -> (f64, f64) {
            let mut f = StableFluid2::new(nx, ny, dx);
            f.set_pressure_solver(PressureSolver::Cg { tol: 1e-13, max_iter: 8000 });
            // Centre the circle on the horizontal mid-line so the geometry
            // is exactly mirror symmetric across it.
            f.grid.set_solid_circle(0.4 * nx as f64 * dx, 0.5 * ny as f64 * dx, 0.18);
            for j in 0..ny {
                for i in 0..=nx {
                    let idx = f.grid.u_idx(i, j);
                    f.grid.u[idx] = alpha.cos();
                }
            }
            for j in 0..=ny {
                for i in 0..nx {
                    let idx = f.grid.v_idx(i, j);
                    f.grid.v[idx] = alpha.sin();
                }
            }
            f.grid.apply_bc(FluidBc::FreeSlip);
            f.project();
            (f.drag_on_solid().x, f.lift_on_solid())
        };

        // Symmetric case: the lift of a symmetric body in an aligned
        // stream vanishes.
        let (drag0, lift0) = force_at(0.0);
        assert!(
            lift0.abs() < 1e-9 * drag0.abs().max(1e-12),
            "symmetric configuration has lift {lift0} (drag {drag0})"
        );

        // Antisymmetry in the incidence.
        let alpha = 0.35;
        let (drag_p, lift_p) = force_at(alpha);
        let (drag_m, lift_m) = force_at(-alpha);
        assert!(lift_p.is_finite() && lift_m.is_finite());
        assert!(
            lift_p.abs() > 1e-6,
            "an inclined stream should load the body: {lift_p}"
        );
        assert!(
            (lift_p + lift_m).abs() < 1e-6 * lift_p.abs(),
            "lift is not odd in the incidence: {lift_p} vs {lift_m}"
        );
        assert!(
            (drag_p - drag_m).abs() < 1e-6 * drag_p.abs().max(1e-12),
            "drag is not even in the incidence: {drag_p} vs {drag_m}"
        );
        // `lift_on_solid` is exactly the transverse component of the
        // pressure force.
        let mut check = StableFluid2::new(nx, ny, dx);
        check.set_pressure_solver(PressureSolver::Cg { tol: 1e-13, max_iter: 8000 });
        check.grid.set_solid_circle(0.4 * nx as f64 * dx, 0.5 * ny as f64 * dx, 0.18);
        for j in 0..ny {
            for i in 0..=nx {
                let idx = check.grid.u_idx(i, j);
                check.grid.u[idx] = alpha.cos();
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                let idx = check.grid.v_idx(i, j);
                check.grid.v[idx] = alpha.sin();
            }
        }
        check.grid.apply_bc(FluidBc::FreeSlip);
        check.project();
        assert!((check.lift_on_solid() - check.drag_on_solid().y).abs() < 1e-15);

        // With no solid at all there is nothing to load.
        let empty = StableFluid2::new(16, 16, 1.0 / 16.0);
        assert!(empty.lift_on_solid().abs() < 1e-15);
        assert!(empty.drag_on_solid().magnitude() < 1e-15);

        // A real wake case stays finite and produces downstream drag.
        let mut cyl = flow_past_cylinder(48, 24, 100.0);
        let dt = 0.4 * cyl.grid.dx;
        for _ in 0..40 {
            cyl.grid.apply_bc(cyl.bc);
            cyl.step(dt);
        }
        assert!(cyl.lift_on_solid().is_finite(), "lift is not finite");
        assert!(cyl.drag_on_solid().x > 0.0, "no downstream drag");
        // `flow_past_cylinder` seeds an up-down asymmetric perturbation to
        // trip vortex shedding, so the transverse load is genuinely
        // non-zero rather than cancelling by symmetry.
        assert!(
            cyl.lift_on_solid().abs() > 0.0,
            "an asymmetric wake must load the cylinder transversely"
        );
    }

    #[test]
    fn test_cylinder_smoke_and_drag() {
        let mut f = flow_past_cylinder(48, 24, 100.0);
        let dt = 0.4 * f.grid.dx;
        for _ in 0..40 {
            f.grid.apply_bc(f.bc);
            f.step(dt);
        }
        assert!(f.grid.u.iter().all(|v| v.is_finite()));
        // Pressure drag pushes the cylinder downstream.
        let drag = f.drag_on_solid();
        assert!(drag.x > 0.0, "drag {drag:?}");
    }

    #[test]
    fn test_rayleigh_benard_onset() {
        // Well above critical Ra: convection grows from the perturbation.
        let mut f = rayleigh_benard(32, 16, 1e5, 1.0);
        let dt = 0.01;
        let mut ke_early = 0.0;
        for s in 0..80 {
            // Hold the plates at fixed temperature.
            let ny = f.temperature.ny;
            let nx = f.temperature.nx;
            for i in 0..nx {
                f.temperature.data[i] = 1.0;
                f.temperature.data[(ny - 1) * nx + i] = 0.0;
            }
            f.step(dt);
            if s == 10 {
                ke_early = f.grid.kinetic_energy();
            }
        }
        let ke_late = f.grid.kinetic_energy();
        assert!(ke_late > ke_early, "no convective growth: {ke_early} -> {ke_late}");
        assert!(ke_late.is_finite());
    }

    #[test]
    fn test_stable_fluid_3d() {
        let mut f = StableFluid3::new(12, 12, 12, 1.0 / 12.0);
        let mut rng = Rng::new(6);
        for u in f.grid.u.iter_mut() {
            *u = 2.0 * rng.next_f64() - 1.0;
        }
        for v in f.grid.v.iter_mut() {
            *v = 2.0 * rng.next_f64() - 1.0;
        }
        f.step(0.01);
        let div = f.grid.divergence();
        let max_div = div.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_div < 1e-6, "3D divergence {max_div}");
        // Buoyant plume rises along +y.
        let mut b = StableFluid3::new(10, 10, 10, 0.1);
        b.buoyancy = 4.0;
        for k in 3..7 {
            for j in 1..4 {
                for i in 3..7 {
                    b.temperature[(k * 10 + j) * 10 + i] = 1.0;
                }
            }
        }
        for _ in 0..10 {
            b.step(0.02);
        }
        let v_mid = b.grid.velocity_at(Vec3::new(0.5, 0.5, 0.5)).y;
        assert!(v_mid > 0.0, "3D plume did not rise: {v_mid}");
    }
}
