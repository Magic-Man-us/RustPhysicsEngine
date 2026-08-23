//! Staggered (MAC) grids and cell-centered scalar fields for
//! incompressible flow solvers.

use crate::math::{Vec2, Vec3};

/// Boundary condition for the velocity field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FluidBc {
    /// Wrap-around domain.
    Periodic,
    /// Zero normal and tangential velocity at walls and solids.
    NoSlip,
    /// Zero normal velocity only.
    FreeSlip,
    /// Fixed velocity on the left (-x) boundary, outflow on the right.
    Inflow(Vec2),
    /// Zero-gradient everywhere.
    Outflow,
}

/// 2D marker-and-cell grid: `u` on vertical faces ((nx+1) × ny), `v` on
/// horizontal faces (nx × (ny+1)), pressure and solid flags at cell
/// centers. Cell (i, j) spans [i·dx, (i+1)·dx) × [j·dx, (j+1)·dx).
pub struct MacGrid2 {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub p: Vec<f64>,
    pub solid: Vec<bool>,
}

impl MacGrid2 {
    /// New grid with all fields zero and no solids.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            u: vec![0.0; (nx + 1) * ny],
            v: vec![0.0; nx * (ny + 1)],
            p: vec![0.0; nx * ny],
            solid: vec![false; nx * ny],
        }
    }

    /// Index into `u` (i in 0..=nx, j in 0..ny).
    #[inline]
    #[must_use]
    pub fn u_idx(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }

    /// Index into `v` (i in 0..nx, j in 0..=ny).
    #[inline]
    #[must_use]
    pub fn v_idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Index into cell-centered arrays.
    #[inline]
    #[must_use]
    pub fn c_idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// u face value at (i, j).
    #[must_use]
    pub fn u_at(&self, i: usize, j: usize) -> f64 {
        self.u[self.u_idx(i, j)]
    }

    /// v face value at (i, j).
    #[must_use]
    pub fn v_at(&self, i: usize, j: usize) -> f64 {
        self.v[self.v_idx(i, j)]
    }

    fn bilinear(data: &[f64], nx: usize, ny: usize, x: f64, y: f64) -> f64 {
        // Sample a lattice whose node (i, j) sits at grid coords (i, j).
        let x = x.clamp(0.0, (nx - 1) as f64);
        let y = y.clamp(0.0, (ny - 1) as f64);
        let i = (x.floor() as usize).min(nx.saturating_sub(2));
        let j = (y.floor() as usize).min(ny.saturating_sub(2));
        let fx = x - i as f64;
        let fy = y - j as f64;
        let idx = |i: usize, j: usize| j * nx + i;
        let i1 = (i + 1).min(nx - 1);
        let j1 = (j + 1).min(ny - 1);
        data[idx(i, j)] * (1.0 - fx) * (1.0 - fy)
            + data[idx(i1, j)] * fx * (1.0 - fy)
            + data[idx(i, j1)] * (1.0 - fx) * fy
            + data[idx(i1, j1)] * fx * fy
    }

    /// Bilinearly interpolated velocity at a world position.
    #[must_use]
    pub fn velocity_at(&self, p: Vec2) -> Vec2 {
        let gx = p.x / self.dx;
        let gy = p.y / self.dx;
        // u nodes at (i, j + 0.5), v nodes at (i + 0.5, j).
        let u = Self::bilinear(&self.u, self.nx + 1, self.ny, gx, gy - 0.5);
        let v = Self::bilinear(&self.v, self.nx, self.ny + 1, gx - 0.5, gy);
        Vec2::new(u, v)
    }

    /// Cell-centered divergence (1/s).
    #[must_use]
    pub fn divergence(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.nx * self.ny];
        for j in 0..self.ny {
            for i in 0..self.nx {
                d[self.c_idx(i, j)] = (self.u_at(i + 1, j) - self.u_at(i, j)
                    + self.v_at(i, j + 1)
                    - self.v_at(i, j))
                    / self.dx;
            }
        }
        d
    }

    /// Cell-centered vorticity ω = ∂v/∂x − ∂u/∂y (central differences of
    /// face-averaged components).
    #[must_use]
    pub fn curl(&self) -> Vec<f64> {
        let mut w = vec![0.0; self.nx * self.ny];
        let uc = |i: usize, j: usize| 0.5 * (self.u_at(i, j) + self.u_at(i + 1, j));
        let vc = |i: usize, j: usize| 0.5 * (self.v_at(i, j) + self.v_at(i, j + 1));
        for j in 0..self.ny {
            for i in 0..self.nx {
                let ip = (i + 1).min(self.nx - 1);
                let im = i.saturating_sub(1);
                let jp = (j + 1).min(self.ny - 1);
                let jm = j.saturating_sub(1);
                let dv_dx = (vc(ip, j) - vc(im, j)) / ((ip - im).max(1) as f64 * self.dx);
                let du_dy = (uc(i, jp) - uc(i, jm)) / ((jp - jm).max(1) as f64 * self.dx);
                w[self.c_idx(i, j)] = dv_dx - du_dy;
            }
        }
        w
    }

    /// Largest face-velocity magnitude.
    #[must_use]
    pub fn max_velocity(&self) -> f64 {
        let mu = self.u.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let mv = self.v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        (mu * mu + mv * mv).sqrt()
    }

    /// Time step honoring the CFL number.
    #[must_use]
    pub fn cfl_dt(&self, cfl: f64) -> f64 {
        let vmax = self.max_velocity().max(1e-9);
        cfl * self.dx / vmax
    }

    /// Mark cells inside a world-space axis-aligned box as solid and
    /// zero their faces.
    pub fn set_solid_box(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let cx = (i as f64 + 0.5) * self.dx;
                let cy = (j as f64 + 0.5) * self.dx;
                if cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1 {
                    let c = self.c_idx(i, j);
                    self.solid[c] = true;
                }
            }
        }
        self.zero_solid_faces();
    }

    /// Mark cells inside a circle as solid and zero their faces.
    pub fn set_solid_circle(&mut self, cx: f64, cy: f64, r: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let x = (i as f64 + 0.5) * self.dx - cx;
                let y = (j as f64 + 0.5) * self.dx - cy;
                if (x * x + y * y).sqrt() <= r {
                    let c = self.c_idx(i, j);
                    self.solid[c] = true;
                }
            }
        }
        self.zero_solid_faces();
    }

    fn zero_solid_faces(&mut self) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                if self.solid[self.c_idx(i, j)] {
                    let (u0, u1) = (self.u_idx(i, j), self.u_idx(i + 1, j));
                    let (v0, v1) = (self.v_idx(i, j), self.v_idx(i, j + 1));
                    self.u[u0] = 0.0;
                    self.u[u1] = 0.0;
                    self.v[v0] = 0.0;
                    self.v[v1] = 0.0;
                }
            }
        }
    }

    /// Apply a domain boundary condition to the face velocities.
    pub fn apply_bc(&mut self, bc: FluidBc) {
        let (nx, ny) = (self.nx, self.ny);
        match bc {
            FluidBc::Periodic => {
                for j in 0..ny {
                    let right = self.u_at(nx, j);
                    let left = self.u_at(0, j);
                    let avg = 0.5 * (right + left);
                    let (i0, i1) = (self.u_idx(0, j), self.u_idx(nx, j));
                    self.u[i0] = avg;
                    self.u[i1] = avg;
                }
                for i in 0..nx {
                    let top = self.v_at(i, ny);
                    let bot = self.v_at(i, 0);
                    let avg = 0.5 * (top + bot);
                    let (i0, i1) = (self.v_idx(i, 0), self.v_idx(i, ny));
                    self.v[i0] = avg;
                    self.v[i1] = avg;
                }
            }
            FluidBc::NoSlip => {
                for j in 0..ny {
                    let (l, r) = (self.u_idx(0, j), self.u_idx(nx, j));
                    self.u[l] = 0.0;
                    self.u[r] = 0.0;
                }
                for i in 0..nx {
                    let (b, t) = (self.v_idx(i, 0), self.v_idx(i, ny));
                    self.v[b] = 0.0;
                    self.v[t] = 0.0;
                }
                // Tangential: zero the first interior tangential faces
                // adjacent to the walls (ghost-free approximation).
                for i in 0..=nx {
                    let (b, t) = (self.u_idx(i.min(nx), 0), self.u_idx(i.min(nx), ny - 1));
                    self.u[b] = 0.0;
                    self.u[t] = 0.0;
                }
                self.zero_solid_faces();
            }
            FluidBc::FreeSlip => {
                for j in 0..ny {
                    let (l, r) = (self.u_idx(0, j), self.u_idx(nx, j));
                    self.u[l] = 0.0;
                    self.u[r] = 0.0;
                }
                for i in 0..nx {
                    let (b, t) = (self.v_idx(i, 0), self.v_idx(i, ny));
                    self.v[b] = 0.0;
                    self.v[t] = 0.0;
                }
                self.zero_solid_faces();
            }
            FluidBc::Inflow(vin) => {
                for j in 0..ny {
                    let l = self.u_idx(0, j);
                    self.u[l] = vin.x;
                    // Outflow (zero gradient) on the right.
                    let (r, r1) = (self.u_idx(nx, j), self.u_idx(nx - 1, j));
                    self.u[r] = self.u[r1];
                }
                for i in 0..nx {
                    let (b, t, t1) = (self.v_idx(i, 0), self.v_idx(i, ny), self.v_idx(i, ny - 1));
                    self.v[b] = vin.y;
                    self.v[t] = self.v[t1];
                }
            }
            FluidBc::Outflow => {
                for j in 0..ny {
                    let (l, l1) = (self.u_idx(0, j), self.u_idx(1, j));
                    let (r, r1) = (self.u_idx(nx, j), self.u_idx(nx - 1, j));
                    self.u[l] = self.u[l1];
                    self.u[r] = self.u[r1];
                }
                for i in 0..nx {
                    let (b, b1) = (self.v_idx(i, 0), self.v_idx(i, 1));
                    let (t, t1) = (self.v_idx(i, ny), self.v_idx(i, ny - 1));
                    self.v[b] = self.v[b1];
                    self.v[t] = self.v[t1];
                }
            }
        }
    }

    /// Total kinetic energy 0.5 Σ |v|² dx² (unit density).
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let mut e = 0.0;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let uc = 0.5 * (self.u_at(i, j) + self.u_at(i + 1, j));
                let vc = 0.5 * (self.v_at(i, j) + self.v_at(i, j + 1));
                e += 0.5 * (uc * uc + vc * vc);
            }
        }
        e * self.dx * self.dx
    }

    /// Total enstrophy 0.5 Σ ω² dx².
    #[must_use]
    pub fn enstrophy(&self) -> f64 {
        self.curl().iter().map(|w| 0.5 * w * w).sum::<f64>() * self.dx * self.dx
    }
}

/// 3D MAC grid (faces staggered per axis).
pub struct MacGrid3 {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
    pub p: Vec<f64>,
    pub solid: Vec<bool>,
}

impl MacGrid3 {
    /// New grid with all fields zero.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            u: vec![0.0; (nx + 1) * ny * nz],
            v: vec![0.0; nx * (ny + 1) * nz],
            w: vec![0.0; nx * ny * (nz + 1)],
            p: vec![0.0; nx * ny * nz],
            solid: vec![false; nx * ny * nz],
        }
    }

    /// u face value.
    #[must_use]
    pub fn u_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.u[(k * self.ny + j) * (self.nx + 1) + i]
    }

    /// v face value.
    #[must_use]
    pub fn v_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.v[(k * (self.ny + 1) + j) * self.nx + i]
    }

    /// w face value.
    #[must_use]
    pub fn w_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.w[(k * self.ny + j) * self.nx + i]
    }

    fn trilinear(
        data: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        let x = x.clamp(0.0, (nx - 1) as f64);
        let y = y.clamp(0.0, (ny - 1) as f64);
        let z = z.clamp(0.0, (nz - 1) as f64);
        let i = (x.floor() as usize).min(nx.saturating_sub(2));
        let j = (y.floor() as usize).min(ny.saturating_sub(2));
        let k = (z.floor() as usize).min(nz.saturating_sub(2));
        let (fx, fy, fz) = (x - i as f64, y - j as f64, z - k as f64);
        let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let i1 = (i + 1).min(nx - 1);
        let j1 = (j + 1).min(ny - 1);
        let k1 = (k + 1).min(nz - 1);
        let c00 = data[idx(i, j, k)] * (1.0 - fx) + data[idx(i1, j, k)] * fx;
        let c10 = data[idx(i, j1, k)] * (1.0 - fx) + data[idx(i1, j1, k)] * fx;
        let c01 = data[idx(i, j, k1)] * (1.0 - fx) + data[idx(i1, j, k1)] * fx;
        let c11 = data[idx(i, j1, k1)] * (1.0 - fx) + data[idx(i1, j1, k1)] * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }

    /// Trilinearly interpolated velocity at a world position.
    #[must_use]
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        let (gx, gy, gz) = (p.x / self.dx, p.y / self.dx, p.z / self.dx);
        let u = Self::trilinear(&self.u, self.nx + 1, self.ny, self.nz, gx, gy - 0.5, gz - 0.5);
        let v = Self::trilinear(&self.v, self.nx, self.ny + 1, self.nz, gx - 0.5, gy, gz - 0.5);
        let w = Self::trilinear(&self.w, self.nx, self.ny, self.nz + 1, gx - 0.5, gy - 0.5, gz);
        Vec3::new(u, v, w)
    }

    /// Cell-centered divergence.
    #[must_use]
    pub fn divergence(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.nx * self.ny * self.nz];
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    d[(k * self.ny + j) * self.nx + i] = (self.u_at(i + 1, j, k)
                        - self.u_at(i, j, k)
                        + self.v_at(i, j + 1, k)
                        - self.v_at(i, j, k)
                        + self.w_at(i, j, k + 1)
                        - self.w_at(i, j, k))
                        / self.dx;
                }
            }
        }
        d
    }

    /// Largest face-velocity magnitude bound.
    #[must_use]
    pub fn max_velocity(&self) -> f64 {
        let mu = self.u.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let mv = self.v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let mw = self.w.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        (mu * mu + mv * mv + mw * mw).sqrt()
    }

    /// Time step honoring the CFL number.
    #[must_use]
    pub fn cfl_dt(&self, cfl: f64) -> f64 {
        cfl * self.dx / self.max_velocity().max(1e-9)
    }
}

/// Cell-centered scalar field on the same layout as [`MacGrid2`]
/// pressure cells; node (i, j) sits at world ((i+0.5) dx, (j+0.5) dx).
#[derive(Debug, Clone)]
pub struct CellField2 {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub data: Vec<f64>,
}

impl CellField2 {
    /// New zero field.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Self { nx, ny, dx, data: vec![0.0; nx * ny] }
    }

    /// Build from a function of cell-center world coordinates.
    #[must_use]
    pub fn from_fn(nx: usize, ny: usize, dx: f64, f: impl Fn(f64, f64) -> f64) -> Self {
        let mut field = Self::new(nx, ny, dx);
        for j in 0..ny {
            for i in 0..nx {
                field.data[j * nx + i] = f((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx);
            }
        }
        field
    }

    /// Value at cell (i, j).
    #[inline]
    #[must_use]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.nx + i]
    }

    /// Mutable value at cell (i, j).
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[j * self.nx + i]
    }

    /// Bilinear sample at a world position (clamped at the borders).
    #[must_use]
    pub fn sample(&self, p: Vec2) -> f64 {
        MacGrid2::bilinear(
            &self.data,
            self.nx,
            self.ny,
            p.x / self.dx - 0.5,
            p.y / self.dx - 0.5,
        )
    }

    /// Clamped Catmull-Rom (monotone-limited) sample at a world position.
    #[must_use]
    pub fn sample_cubic(&self, p: Vec2) -> f64 {
        let gx = (p.x / self.dx - 0.5).clamp(0.0, (self.nx - 1) as f64);
        let gy = (p.y / self.dx - 0.5).clamp(0.0, (self.ny - 1) as f64);
        let i = gx.floor() as usize;
        let j = gy.floor() as usize;
        let fx = gx - i as f64;
        let fy = gy - j as f64;
        let cat = |m1: f64, p0: f64, p1: f64, p2: f64, t: f64| -> f64 {
            let a = 0.5 * (-m1 + 3.0 * p0 - 3.0 * p1 + p2);
            let b = m1 - 2.5 * p0 + 2.0 * p1 - 0.5 * p2;
            let c = 0.5 * (p1 - m1);
            let v = ((a * t + b) * t + c) * t + p0;
            // Clamp to the local hull for monotonicity.
            v.clamp(p0.min(p1), p0.max(p1))
        };
        let gi = |ii: i64, jj: i64| -> f64 {
            let ii = ii.clamp(0, self.nx as i64 - 1) as usize;
            let jj = jj.clamp(0, self.ny as i64 - 1) as usize;
            self.at(ii, jj)
        };
        let rows: Vec<f64> = (-1..=2)
            .map(|dj| {
                cat(
                    gi(i as i64 - 1, j as i64 + dj),
                    gi(i as i64, j as i64 + dj),
                    gi(i as i64 + 1, j as i64 + dj),
                    gi(i as i64 + 2, j as i64 + dj),
                    fx,
                )
            })
            .collect();
        cat(rows[0], rows[1], rows[2], rows[3], fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_grid_basics() {
        let mut g = MacGrid2::new(16, 12, 0.5);
        // Uniform flow: zero divergence, zero curl, correct sampling.
        g.u.iter_mut().for_each(|u| *u = 2.0);
        g.v.iter_mut().for_each(|v| *v = -1.0);
        assert!(g.divergence().iter().all(|d| d.abs() < 1e-12));
        assert!(g.curl().iter().all(|w| w.abs() < 1e-12));
        let vel = g.velocity_at(Vec2::new(3.3, 2.2));
        assert!((vel.x - 2.0).abs() < 1e-12 && (vel.y + 1.0).abs() < 1e-12);
        assert!((g.max_velocity() - (5.0_f64).sqrt()).abs() < 1e-12);
        assert!((g.cfl_dt(0.5) - 0.5 * 0.5 / 5.0_f64.sqrt()).abs() < 1e-12);
        let ke = g.kinetic_energy();
        let expected = 0.5 * 5.0 * (16.0 * 12.0) * 0.25;
        assert!((ke - expected).abs() < 1e-9, "KE {ke} vs {expected}");
    }

    #[test]
    fn test_rigid_rotation_curl() {
        // u = -Ω y, v = Ω x about the domain center: ω = 2Ω.
        let n = 32;
        let dx = 1.0 / n as f64;
        let omega = 3.0;
        let mut g = MacGrid2::new(n, n, dx);
        let c = 0.5;
        for j in 0..n {
            for i in 0..=n {
                let y = (j as f64 + 0.5) * dx - c;
                let idx = g.u_idx(i, j);
                g.u[idx] = -omega * y;
            }
        }
        for j in 0..=n {
            for i in 0..n {
                let x = (i as f64 + 0.5) * dx - c;
                let idx = g.v_idx(i, j);
                g.v[idx] = omega * x;
            }
        }
        let w = g.curl();
        // Interior cells: ω = 2Ω exactly for this linear field.
        for j in 2..n - 2 {
            for i in 2..n - 2 {
                assert!(
                    (w[g.c_idx(i, j)] - 2.0 * omega).abs() < 1e-9,
                    "curl {} at ({i},{j})",
                    w[g.c_idx(i, j)]
                );
            }
        }
        assert!(g.enstrophy() > 0.0);
        assert!(g.divergence().iter().all(|d| d.abs() < 1e-9));
    }

    #[test]
    fn test_solids_and_bcs() {
        let mut g = MacGrid2::new(16, 16, 1.0);
        g.u.iter_mut().for_each(|u| *u = 1.0);
        g.set_solid_circle(8.0, 8.0, 2.0);
        assert!(g.solid[g.c_idx(8, 8)]);
        assert!(!g.solid[g.c_idx(1, 1)]);
        assert_eq!(g.u_at(8, 8), 0.0);
        let mut g2 = MacGrid2::new(8, 8, 1.0);
        g2.u.iter_mut().for_each(|u| *u = 3.0);
        g2.v.iter_mut().for_each(|v| *v = 2.0);
        g2.apply_bc(FluidBc::NoSlip);
        assert_eq!(g2.u_at(0, 3), 0.0);
        assert_eq!(g2.u_at(8, 3), 0.0);
        assert_eq!(g2.v_at(3, 0), 0.0);
        assert_eq!(g2.v_at(3, 8), 0.0);
        let mut g3 = MacGrid2::new(8, 8, 1.0);
        g3.apply_bc(FluidBc::Inflow(Vec2::new(1.5, 0.0)));
        assert_eq!(g3.u_at(0, 4), 1.5);
        let mut g4 = MacGrid2::new(8, 8, 1.0);
        for j in 0..8 {
            let idx = g4.u_idx(1, j);
            g4.u[idx] = 2.5;
        }
        g4.apply_bc(FluidBc::Outflow);
        assert_eq!(g4.u_at(0, 4), 2.5);
        g4.set_solid_box(2.0, 2.0, 4.0, 4.0);
        assert!(g4.solid[g4.c_idx(3, 3)]);
    }

    #[test]
    fn test_mac3_and_cell_field() {
        let mut g = MacGrid3::new(8, 8, 8, 0.25);
        g.u.iter_mut().for_each(|u| *u = 1.0);
        g.v.iter_mut().for_each(|v| *v = 2.0);
        g.w.iter_mut().for_each(|w| *w = -1.0);
        assert!(g.divergence().iter().all(|d| d.abs() < 1e-12));
        let vel = g.velocity_at(Vec3::new(1.0, 1.0, 1.0));
        assert!((vel.x - 1.0).abs() < 1e-12);
        assert!((vel.y - 2.0).abs() < 1e-12);
        assert!((vel.z + 1.0).abs() < 1e-12);
        assert!(g.cfl_dt(1.0) > 0.0);
        // CellField2 sampling reproduces a linear function exactly.
        let f = CellField2::from_fn(32, 32, 0.1, |x, y| 2.0 * x + 3.0 * y);
        let p = Vec2::new(1.234, 2.345);
        assert!((f.sample(p) - (2.0 * p.x + 3.0 * p.y)).abs() < 1e-9);
        // Cubic sampling is exact on linears too and bounded on steps.
        assert!((f.sample_cubic(p) - (2.0 * p.x + 3.0 * p.y)).abs() < 1e-9);
        let step = CellField2::from_fn(32, 32, 0.1, |x, _| if x > 1.6 { 1.0 } else { 0.0 });
        for k in 0..40 {
            let v = step.sample_cubic(Vec2::new(1.4 + 0.01 * k as f64, 1.0));
            assert!((-1e-12..=1.0 + 1e-12).contains(&v), "overshoot {v}");
        }
    }
}
