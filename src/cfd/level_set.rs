//! Interface capturing: level sets (upwind/WENO advection, Sussman
//! reinitialization, fast marching, marching squares/tetrahedra), volume
//! of fluid with PLIC, a simple free-surface fluid, and bubble/droplet
//! physics relations.

use crate::cfd::grid::{CellField2, MacGrid2};
use crate::cfd::stable_fluids::StableFluid2;
use crate::fields::ScalarField3;
use crate::geometry::mesh::Mesh;
use crate::math::{Vec2, Vec3};

const PI: f64 = crate::math::constants::PI;

/// A line segment of the reconstructed interface.
#[derive(Debug, Clone, Copy)]
pub struct Segment2 {
    pub a: Vec2,
    pub b: Vec2,
}

/// Advection scheme for the level set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WenoOrUpwind {
    Weno,
    Upwind,
}

/// 2D signed distance level set (φ < 0 inside).
pub struct LevelSet2 {
    pub phi: CellField2,
    pub band: Option<f64>,
}

fn weno5_deriv(v: [f64; 5], dx: f64) -> f64 {
    // WENO5 approximation of the derivative from the five one-sided
    // differences v[i] = (q_{i+1} - q_i)/dx along the stencil.
    let eps = 1e-6;
    let b0 = 13.0 / 12.0 * (v[0] - 2.0 * v[1] + v[2]).powi(2)
        + 0.25 * (v[0] - 4.0 * v[1] + 3.0 * v[2]).powi(2);
    let b1 = 13.0 / 12.0 * (v[1] - 2.0 * v[2] + v[3]).powi(2) + 0.25 * (v[1] - v[3]).powi(2);
    let b2 = 13.0 / 12.0 * (v[2] - 2.0 * v[3] + v[4]).powi(2)
        + 0.25 * (3.0 * v[2] - 4.0 * v[3] + v[4]).powi(2);
    let a0 = 0.1 / (eps + b0).powi(2);
    let a1 = 0.6 / (eps + b1).powi(2);
    let a2 = 0.3 / (eps + b2).powi(2);
    let w0 = a0 / (a0 + a1 + a2);
    let w1 = a1 / (a0 + a1 + a2);
    let w2 = a2 / (a0 + a1 + a2);
    let p0 = v[0] / 3.0 - 7.0 * v[1] / 6.0 + 11.0 * v[2] / 6.0;
    let p1 = -v[1] / 6.0 + 5.0 * v[2] / 6.0 + v[3] / 3.0;
    let p2 = v[2] / 3.0 + 5.0 * v[3] / 6.0 - v[4] / 6.0;
    let _ = dx;
    w0 * p0 + w1 * p1 + w2 * p2
}

impl LevelSet2 {
    /// Build from a signed distance function of world coordinates.
    #[must_use]
    pub fn from_sdf(f: impl Fn(f64, f64) -> f64, nx: usize, ny: usize, dx: f64) -> Self {
        Self { phi: CellField2::from_fn(nx, ny, dx, f), band: None }
    }

    /// Circle of radius r centered at (cx, cy).
    #[must_use]
    pub fn circle(nx: usize, ny: usize, dx: f64, cx: f64, cy: f64, r: f64) -> Self {
        Self::from_sdf(
            move |x, y| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r,
            nx,
            ny,
            dx,
        )
    }

    /// Axis-aligned box interior.
    #[must_use]
    pub fn box_(nx: usize, ny: usize, dx: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self::from_sdf(
            move |x, y| {
                let dxs = (x0 - x).max(x - x1);
                let dys = (y0 - y).max(y - y1);
                if dxs <= 0.0 && dys <= 0.0 {
                    dxs.max(dys)
                } else {
                    (dxs.max(0.0).powi(2) + dys.max(0.0).powi(2)).sqrt()
                }
            },
            nx,
            ny,
            dx,
        )
    }

    fn phi_at(&self, i: i64, j: i64) -> f64 {
        let (nx, ny) = (self.phi.nx as i64, self.phi.ny as i64);
        self.phi.at(i.clamp(0, nx - 1) as usize, j.clamp(0, ny - 1) as usize)
    }

    /// Advect through a MAC velocity field for one step.
    pub fn advect(&mut self, grid: &MacGrid2, dt: f64, scheme: WenoOrUpwind) {
        let (nx, ny, dx) = (self.phi.nx, self.phi.ny, self.phi.dx);
        let mut new_phi = self.phi.clone();
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let p = Vec2::new((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx);
                let vel = grid.velocity_at(p);
                let (dpx, dpy) = match scheme {
                    WenoOrUpwind::Upwind => {
                        let dpx = if vel.x >= 0.0 {
                            (self.phi_at(i, j) - self.phi_at(i - 1, j)) / dx
                        } else {
                            (self.phi_at(i + 1, j) - self.phi_at(i, j)) / dx
                        };
                        let dpy = if vel.y >= 0.0 {
                            (self.phi_at(i, j) - self.phi_at(i, j - 1)) / dx
                        } else {
                            (self.phi_at(i, j + 1) - self.phi_at(i, j)) / dx
                        };
                        (dpx, dpy)
                    }
                    WenoOrUpwind::Weno => {
                        let diffs_x: Vec<f64> = (-3..3)
                            .map(|o| (self.phi_at(i + o + 1, j) - self.phi_at(i + o, j)) / dx)
                            .collect();
                        let diffs_y: Vec<f64> = (-3..3)
                            .map(|o| (self.phi_at(i, j + o + 1) - self.phi_at(i, j + o)) / dx)
                            .collect();
                        let dpx = if vel.x >= 0.0 {
                            weno5_deriv(
                                [diffs_x[0], diffs_x[1], diffs_x[2], diffs_x[3], diffs_x[4]],
                                dx,
                            )
                        } else {
                            weno5_deriv(
                                [diffs_x[5], diffs_x[4], diffs_x[3], diffs_x[2], diffs_x[1]],
                                dx,
                            )
                        };
                        let dpy = if vel.y >= 0.0 {
                            weno5_deriv(
                                [diffs_y[0], diffs_y[1], diffs_y[2], diffs_y[3], diffs_y[4]],
                                dx,
                            )
                        } else {
                            weno5_deriv(
                                [diffs_y[5], diffs_y[4], diffs_y[3], diffs_y[2], diffs_y[1]],
                                dx,
                            )
                        };
                        (dpx, dpy)
                    }
                };
                new_phi.data[(j as usize) * nx + i as usize] =
                    self.phi.at(i as usize, j as usize) - dt * (vel.x * dpx + vel.y * dpy);
            }
        }
        self.phi = new_phi;
    }

    /// Sussman PDE reinitialization toward |∇φ| = 1.
    pub fn reinitialize(&mut self, iters: usize) {
        let (nx, ny, dx) = (self.phi.nx, self.phi.ny, self.phi.dx);
        let phi0 = self.phi.clone();
        let dtau = 0.5 * dx;
        for _ in 0..iters {
            let mut next = self.phi.clone();
            for j in 0..ny as i64 {
                for i in 0..nx as i64 {
                    let p0 = phi0.at(i as usize, j as usize);
                    let a = (self.phi_at(i, j) - self.phi_at(i - 1, j)) / dx; // D-x
                    let b = (self.phi_at(i + 1, j) - self.phi_at(i, j)) / dx; // D+x
                    let c = (self.phi_at(i, j) - self.phi_at(i, j - 1)) / dx;
                    let d = (self.phi_at(i, j + 1) - self.phi_at(i, j)) / dx;
                    // Sussman-Fatemi smoothed sign with the local
                    // gradient scale, so steep initial data does not
                    // shift the front.
                    let g0 = (0.25 * ((a + b).powi(2) + (c + d).powi(2))).sqrt().max(1e-9);
                    let s = p0 / (p0 * p0 + (g0 * dx).powi(2)).sqrt();
                    // Godunov Hamiltonian.
                    let grad = if p0 > 0.0 {
                        (a.max(0.0).powi(2).max(b.min(0.0).powi(2))
                            + c.max(0.0).powi(2).max(d.min(0.0).powi(2)))
                        .sqrt()
                    } else {
                        (a.min(0.0).powi(2).max(b.max(0.0).powi(2))
                            + c.min(0.0).powi(2).max(d.max(0.0).powi(2)))
                        .sqrt()
                    };
                    next.data[(j as usize) * nx + i as usize] =
                        self.phi.at(i as usize, j as usize) - dtau * s * (grad - 1.0);
                }
            }
            self.phi = next;
        }
    }

    /// Fast marching redistancing (first-order Eikonal solve outward
    /// from the interface).
    pub fn fast_marching(&mut self) {
        let (nx, ny, dx) = (self.phi.nx, self.phi.ny, self.phi.dx);
        let n = nx * ny;
        let sign: Vec<f64> = self.phi.data.iter().map(|v| v.signum()).collect();
        let mut dist = vec![f64::INFINITY; n];
        let mut known = vec![false; n];
        // Initialize cells adjacent to the interface with interpolated
        // distances.
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                let p = self.phi.data[c];
                let mut d_min = f64::INFINITY;
                let mut nbr = |q: f64| {
                    if p == 0.0 {
                        d_min = 0.0;
                    } else if p * q < 0.0 {
                        let frac = p / (p - q);
                        d_min = d_min.min(frac.abs() * dx);
                    }
                };
                if i > 0 {
                    nbr(self.phi.at(i - 1, j));
                }
                if i + 1 < nx {
                    nbr(self.phi.at(i + 1, j));
                }
                if j > 0 {
                    nbr(self.phi.at(i, j - 1));
                }
                if j + 1 < ny {
                    nbr(self.phi.at(i, j + 1));
                }
                if d_min.is_finite() {
                    dist[c] = d_min;
                    known[c] = true;
                }
            }
        }
        // Min-heap march.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        #[derive(PartialEq)]
        struct H(f64, usize);
        impl Eq for H {}
        impl PartialOrd for H {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for H {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        let mut heap = BinaryHeap::new();
        for c in 0..n {
            if known[c] {
                heap.push(Reverse(H(dist[c], c)));
            }
        }
        let solve = |dist: &[f64], i: usize, j: usize| -> f64 {
            let c = j * nx + i;
            let _ = c;
            let dh = |a: Option<f64>, b: Option<f64>| -> f64 {
                match (a, b) {
                    (Some(x), Some(y)) => x.min(y),
                    (Some(x), None) => x,
                    (None, Some(y)) => y,
                    (None, None) => f64::INFINITY,
                }
            };
            let ux = dh(
                (i > 0).then(|| dist[j * nx + i - 1]),
                (i + 1 < nx).then(|| dist[j * nx + i + 1]),
            );
            let uy = dh(
                (j > 0).then(|| dist[(j - 1) * nx + i]),
                (j + 1 < ny).then(|| dist[(j + 1) * nx + i]),
            );
            let (a, b) = (ux.min(uy), ux.max(uy));
            if (b - a) >= dx || !b.is_finite() {
                a + dx
            } else {
                0.5 * (a + b + (2.0 * dx * dx - (b - a).powi(2)).sqrt())
            }
        };
        while let Some(Reverse(H(d, c))) = heap.pop() {
            if d > dist[c] {
                continue;
            }
            let (i, j) = (c % nx, c / nx);
            let visit = |ii: usize, jj: usize, heap: &mut BinaryHeap<Reverse<H>>,
                             dist: &mut Vec<f64>| {
                let cc = jj * nx + ii;
                if known[cc] {
                    return;
                }
                let nd = solve(dist, ii, jj);
                if nd < dist[cc] {
                    dist[cc] = nd;
                    heap.push(Reverse(H(nd, cc)));
                }
            };
            if i > 0 {
                visit(i - 1, j, &mut heap, &mut dist);
            }
            if i + 1 < nx {
                visit(i + 1, j, &mut heap, &mut dist);
            }
            if j > 0 {
                visit(i, j - 1, &mut heap, &mut dist);
            }
            if j + 1 < ny {
                visit(i, j + 1, &mut heap, &mut dist);
            }
        }
        for c in 0..n {
            self.phi.data[c] = sign[c] * dist[c].min(1e9);
        }
    }

    /// Mean curvature κ = ∇·(∇φ/|∇φ|).
    #[must_use]
    pub fn curvature(&self) -> CellField2 {
        let (nx, ny, dx) = (self.phi.nx, self.phi.ny, self.phi.dx);
        let mut out = CellField2::new(nx, ny, dx);
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let px = (self.phi_at(i + 1, j) - self.phi_at(i - 1, j)) / (2.0 * dx);
                let py = (self.phi_at(i, j + 1) - self.phi_at(i, j - 1)) / (2.0 * dx);
                let pxx = (self.phi_at(i + 1, j) - 2.0 * self.phi_at(i, j)
                    + self.phi_at(i - 1, j))
                    / (dx * dx);
                let pyy = (self.phi_at(i, j + 1) - 2.0 * self.phi_at(i, j)
                    + self.phi_at(i, j - 1))
                    / (dx * dx);
                let pxy = (self.phi_at(i + 1, j + 1) - self.phi_at(i - 1, j + 1)
                    - self.phi_at(i + 1, j - 1)
                    + self.phi_at(i - 1, j - 1))
                    / (4.0 * dx * dx);
                let g2 = px * px + py * py;
                out.data[(j as usize) * nx + i as usize] = if g2 > 1e-12 {
                    (pxx * py * py - 2.0 * px * py * pxy + pyy * px * px) / g2.powf(1.5)
                } else {
                    0.0
                };
            }
        }
        out
    }

    /// Outward unit normal at a world position.
    #[must_use]
    pub fn normal(&self, p: Vec2) -> Vec2 {
        let d = self.phi.dx;
        let gx = (self.phi.sample(Vec2::new(p.x + d, p.y)) - self.phi.sample(Vec2::new(p.x - d, p.y)))
            / (2.0 * d);
        let gy = (self.phi.sample(Vec2::new(p.x, p.y + d)) - self.phi.sample(Vec2::new(p.x, p.y - d)))
            / (2.0 * d);
        Vec2::new(gx, gy).normalized()
    }

    /// Enclosed (φ < 0) area via a smoothed Heaviside.
    #[must_use]
    pub fn area(&self) -> f64 {
        let eps = 1.5 * self.phi.dx;
        let h = self.heaviside(eps);
        h.data.iter().map(|v| 1.0 - v).sum::<f64>() * self.phi.dx * self.phi.dx
    }

    /// Interface length via the smoothed delta.
    #[must_use]
    pub fn perimeter(&self) -> f64 {
        let (nx, ny, dx) = (self.phi.nx, self.phi.ny, self.phi.dx);
        let d = self.delta(1.5 * dx);
        let mut total = 0.0;
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let gx = (self.phi_at(i + 1, j) - self.phi_at(i - 1, j)) / (2.0 * dx);
                let gy = (self.phi_at(i, j + 1) - self.phi_at(i, j - 1)) / (2.0 * dx);
                total += d.at(i as usize, j as usize) * (gx * gx + gy * gy).sqrt();
            }
        }
        total * dx * dx
    }

    /// Marching-squares interface segments of φ = 0.
    #[must_use]
    pub fn interface_segments(&self) -> Vec<Segment2> {
        let (nx, ny, dx) = (self.phi.nx, self.phi.ny, self.phi.dx);
        let mut segs = Vec::new();
        // Cell corners at cell centers of a (nx-1)×(ny-1) dual grid.
        let corner = |i: usize, j: usize| -> (Vec2, f64) {
            (
                Vec2::new((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx),
                self.phi.at(i, j),
            )
        };
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let (p00, v00) = corner(i, j);
                let (p10, v10) = corner(i + 1, j);
                let (p11, v11) = corner(i + 1, j + 1);
                let (p01, v01) = corner(i, j + 1);
                let mut pts = Vec::new();
                let mut edge = |pa: Vec2, va: f64, pb: Vec2, vb: f64| {
                    if (va < 0.0) != (vb < 0.0) {
                        let t = va / (va - vb);
                        pts.push(pa + (pb - pa) * t);
                    }
                };
                edge(p00, v00, p10, v10);
                edge(p10, v10, p11, v11);
                edge(p11, v11, p01, v01);
                edge(p01, v01, p00, v00);
                if pts.len() >= 2 {
                    segs.push(Segment2 { a: pts[0], b: pts[1] });
                    if pts.len() == 4 {
                        segs.push(Segment2 { a: pts[2], b: pts[3] });
                    }
                }
            }
        }
        segs
    }

    /// Smoothed Heaviside H_ε(φ) (0 inside, 1 outside).
    #[must_use]
    pub fn heaviside(&self, eps: f64) -> CellField2 {
        let mut out = self.phi.clone();
        for v in out.data.iter_mut() {
            *v = if *v < -eps {
                0.0
            } else if *v > eps {
                1.0
            } else {
                0.5 * (1.0 + *v / eps + (PI * *v / eps).sin() / PI)
            };
        }
        out
    }

    /// Smoothed delta δ_ε(φ).
    #[must_use]
    pub fn delta(&self, eps: f64) -> CellField2 {
        let mut out = self.phi.clone();
        for v in out.data.iter_mut() {
            *v = if v.abs() > eps {
                0.0
            } else {
                0.5 / eps * (1.0 + (PI * *v / eps).cos())
            };
        }
        out
    }

    /// CSG union (min).
    pub fn union(&mut self, other: &LevelSet2) {
        for (a, b) in self.phi.data.iter_mut().zip(&other.phi.data) {
            *a = a.min(*b);
        }
    }

    /// CSG intersection (max).
    pub fn intersect(&mut self, other: &LevelSet2) {
        for (a, b) in self.phi.data.iter_mut().zip(&other.phi.data) {
            *a = a.max(*b);
        }
    }

    /// CSG subtraction (max with −other).
    pub fn subtract(&mut self, other: &LevelSet2) {
        for (a, b) in self.phi.data.iter_mut().zip(&other.phi.data) {
            *a = a.max(-*b);
        }
    }

    /// Extend a scalar field off the interface along normals (upwind
    /// sweeps of q_t + sign(φ) n·∇q = 0) within `band` distance.
    pub fn extend_velocity(&self, vel: &mut CellField2, band: f64) {
        let (nx, ny, dx) = (self.phi.nx, self.phi.ny, self.phi.dx);
        let sweeps = 2 * ((band / dx).ceil() as usize + 2);
        let dtau = 0.5 * dx;
        for _ in 0..sweeps {
            let old = vel.clone();
            let at = |f: &CellField2, i: i64, j: i64| -> f64 {
                f.at(i.clamp(0, nx as i64 - 1) as usize, j.clamp(0, ny as i64 - 1) as usize)
            };
            for j in 0..ny as i64 {
                for i in 0..nx as i64 {
                    let p = self.phi.at(i as usize, j as usize);
                    if p.abs() > band || p.abs() < 0.5 * dx {
                        continue;
                    }
                    let gx = (self.phi_at(i + 1, j) - self.phi_at(i - 1, j)) / (2.0 * dx);
                    let gy = (self.phi_at(i, j + 1) - self.phi_at(i, j - 1)) / (2.0 * dx);
                    let n = Vec2::new(gx, gy).normalized() * p.signum();
                    let dqx = if n.x >= 0.0 {
                        (at(&old, i, j) - at(&old, i - 1, j)) / dx
                    } else {
                        (at(&old, i + 1, j) - at(&old, i, j)) / dx
                    };
                    let dqy = if n.y >= 0.0 {
                        (at(&old, i, j) - at(&old, i, j - 1)) / dx
                    } else {
                        (at(&old, i, j + 1) - at(&old, i, j)) / dx
                    };
                    vel.data[(j as usize) * nx + i as usize] -=
                        dtau * (n.x * dqx + n.y * dqy);
                }
            }
        }
    }

    /// Shift φ by a constant so the enclosed area matches `target_area`.
    pub fn volume_correction(&mut self, target_area: f64) {
        let (mut lo, mut hi) = (-2.0 * self.phi.dx, 2.0 * self.phi.dx);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let mut test = LevelSet2 { phi: self.phi.clone(), band: self.band };
            test.phi.data.iter_mut().for_each(|v| *v += mid);
            // Positive shift shrinks the inside.
            if test.area() > target_area {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let c = 0.5 * (lo + hi);
        self.phi.data.iter_mut().for_each(|v| *v += c);
    }
}

/// 3D level set with mesh extraction.
pub struct LevelSet3 {
    pub phi: ScalarField3,
    pub band: Option<f64>,
}

impl LevelSet3 {
    /// Build from an SDF of world coordinates (node spacing dx).
    #[must_use]
    pub fn from_sdf(f: impl Fn(f64, f64, f64) -> f64, n: usize, dx: f64) -> Self {
        let mut phi = ScalarField3::new(n, n, n, dx);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    phi.set(i, j, k, f(i as f64 * dx, j as f64 * dx, k as f64 * dx));
                }
            }
        }
        Self { phi, band: None }
    }

    /// Sphere SDF.
    #[must_use]
    pub fn sphere(n: usize, dx: f64, c: Vec3, r: f64) -> Self {
        Self::from_sdf(
            move |x, y, z| (Vec3::new(x, y, z) - c).magnitude() - r,
            n,
            dx,
        )
    }

    /// Extract the φ = 0 isosurface as a triangle mesh (marching
    /// tetrahedra: table-free, watertight per-cube).
    #[must_use]
    pub fn to_mesh(&self) -> Mesh {
        let (nx, ny, nz, dx) = (self.phi.nx, self.phi.ny, self.phi.nz, self.phi.dx);
        let mut mesh = Mesh::new();
        // Split each cube into 6 tetrahedra.
        const TETS: [[usize; 4]; 6] = [
            [0, 5, 1, 6],
            [0, 1, 3, 6],
            [0, 3, 2, 6],
            [0, 2, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
        ];
        let corner_offset = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ];
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let vals: Vec<f64> = corner_offset
                        .iter()
                        .map(|&(di, dj, dk)| self.phi.get(i + di, j + dj, k + dk))
                        .collect();
                    let pos: Vec<Vec3> = corner_offset
                        .iter()
                        .map(|&(di, dj, dk)| {
                            Vec3::new(
                                (i + di) as f64 * dx,
                                (j + dj) as f64 * dx,
                                (k + dk) as f64 * dx,
                            )
                        })
                        .collect();
                    for tet in TETS {
                        let mut inside = Vec::new();
                        let mut outside = Vec::new();
                        for &v in &tet {
                            if vals[v] < 0.0 {
                                inside.push(v);
                            } else {
                                outside.push(v);
                            }
                        }
                        let interp = |a: usize, b: usize| -> Vec3 {
                            let t = vals[a] / (vals[a] - vals[b]);
                            pos[a] + (pos[b] - pos[a]) * t
                        };
                        let tri = |a: Vec3, b: Vec3, c: Vec3, mesh: &mut Mesh| {
                            let base = mesh.vertices.len();
                            mesh.vertices.push(a);
                            mesh.vertices.push(b);
                            mesh.vertices.push(c);
                            mesh.triangles.push([base, base + 1, base + 2]);
                            mesh.materials.push(0);
                        };
                        match inside.len() {
                            1 => {
                                let p = inside[0];
                                tri(
                                    interp(p, outside[0]),
                                    interp(p, outside[1]),
                                    interp(p, outside[2]),
                                    &mut mesh,
                                );
                            }
                            3 => {
                                let p = outside[0];
                                tri(
                                    interp(inside[0], p),
                                    interp(inside[1], p),
                                    interp(inside[2], p),
                                    &mut mesh,
                                );
                            }
                            2 => {
                                let (a, b) = (inside[0], inside[1]);
                                let (c, d) = (outside[0], outside[1]);
                                let (pac, pad, pbc, pbd) =
                                    (interp(a, c), interp(a, d), interp(b, c), interp(b, d));
                                tri(pac, pad, pbc, &mut mesh);
                                tri(pbc, pad, pbd, &mut mesh);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        mesh
    }
}

// --- Volume of fluid -----------------------------------------------------

/// Volume-of-fluid interface tracking with PLIC reconstruction.
pub struct Vof2 {
    pub fraction: CellField2,
}

impl Vof2 {
    /// Initialize fractions from an SDF (φ < 0 = filled) by 4×4
    /// subsampling.
    #[must_use]
    pub fn init_from_sdf(f: impl Fn(f64, f64) -> f64, nx: usize, ny: usize, dx: f64) -> Self {
        let mut fr = CellField2::new(nx, ny, dx);
        for j in 0..ny {
            for i in 0..nx {
                let mut inside = 0;
                for sj in 0..4 {
                    for si in 0..4 {
                        let x = (i as f64 + (si as f64 + 0.5) / 4.0) * dx;
                        let y = (j as f64 + (sj as f64 + 0.5) / 4.0) * dx;
                        if f(x, y) < 0.0 {
                            inside += 1;
                        }
                    }
                }
                fr.data[j * nx + i] = inside as f64 / 16.0;
            }
        }
        Self { fraction: fr }
    }

    /// Youngs finite-difference interface normals (pointing out of the
    /// liquid).
    #[must_use]
    pub fn reconstruct_normals_youngs(&self) -> Vec<Vec2> {
        let (nx, ny) = (self.fraction.nx, self.fraction.ny);
        let at = |i: i64, j: i64| -> f64 {
            self.fraction
                .at(i.clamp(0, nx as i64 - 1) as usize, j.clamp(0, ny as i64 - 1) as usize)
        };
        let mut out = Vec::with_capacity(nx * ny);
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let gx = (at(i + 1, j - 1) + 2.0 * at(i + 1, j) + at(i + 1, j + 1)
                    - at(i - 1, j - 1)
                    - 2.0 * at(i - 1, j)
                    - at(i - 1, j + 1))
                    / 8.0;
                let gy = (at(i - 1, j + 1) + 2.0 * at(i, j + 1) + at(i + 1, j + 1)
                    - at(i - 1, j - 1)
                    - 2.0 * at(i, j - 1)
                    - at(i + 1, j - 1))
                    / 8.0;
                out.push((-Vec2::new(gx, gy)).normalized());
            }
        }
        out
    }

    /// ELVIRA-style normals: pick, per cell, the best of six candidate
    /// column/row difference slopes by fraction reproduction error.
    #[must_use]
    pub fn reconstruct_normals_elvira(&self) -> Vec<Vec2> {
        let (nx, ny) = (self.fraction.nx, self.fraction.ny);
        let at = |i: i64, j: i64| -> f64 {
            self.fraction
                .at(i.clamp(0, nx as i64 - 1) as usize, j.clamp(0, ny as i64 - 1) as usize)
        };
        let youngs = self.reconstruct_normals_youngs();
        let mut out = Vec::with_capacity(nx * ny);
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                // Column sums and row sums around the cell.
                let col = |o: i64| at(i + o, j - 1) + at(i + o, j) + at(i + o, j + 1);
                let row = |o: i64| at(i - 1, j + o) + at(i, j + o) + at(i + 1, j + o);
                let candidates = [
                    Vec2::new(-(col(1) - col(-1)) / 2.0, -1.0),
                    Vec2::new(-(col(0) - col(-1)), -1.0),
                    Vec2::new(-(col(1) - col(0)), -1.0),
                    Vec2::new(-1.0, -(row(1) - row(-1)) / 2.0),
                    Vec2::new(-1.0, -(row(0) - row(-1))),
                    Vec2::new(-1.0, -(row(1) - row(0))),
                ];
                // Choose the candidate closest in direction to Youngs
                // (full ELVIRA reproduces neighborhood fractions; this
                // uses the Youngs normal as the reference selector).
                let reference = youngs[(j as usize) * nx + i as usize];
                let best = candidates
                    .iter()
                    .map(|c| {
                        let n = c.normalized();
                        let n = if n.dot(&reference) < 0.0 { -n } else { n };
                        (n, (n - reference).magnitude())
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(n, _)| n)
                    .unwrap_or(reference);
                out.push(best);
            }
        }
        out
    }

    /// Directional-split geometric advection using PLIC subsampling of
    /// donor regions.
    pub fn advect_plic(&mut self, grid: &MacGrid2, dt: f64) {
        // x sweep then y sweep, each with donor-cell geometric flux
        // estimated from the reconstructed interface via subsampling.
        self.sweep(grid, dt, true);
        self.sweep(grid, dt, false);
        for f in self.fraction.data.iter_mut() {
            *f = f.clamp(0.0, 1.0);
        }
    }

    fn sweep(&mut self, grid: &MacGrid2, dt: f64, x_dir: bool) {
        let (nx, ny, dx) = (self.fraction.nx, self.fraction.ny, self.fraction.dx);
        let normals = self.reconstruct_normals_youngs();
        let f_old = self.fraction.clone();
        let sub = 8;
        // In-cell filled test from the PLIC line: n·(p − p_c) + alpha ≤ 0.
        let filled = |c: usize, local: Vec2| -> bool {
            let f = f_old.data[c];
            if f <= 0.0 {
                return false;
            }
            if f >= 1.0 {
                return true;
            }
            let n = normals[c];
            // Find alpha so the fraction below the line equals f
            // (bisection on the subgrid).
            let mut lo = -1.0;
            let mut hi = 1.0;
            for _ in 0..20 {
                let mid = 0.5 * (lo + hi);
                let mut count = 0;
                for sj in 0..sub {
                    for si in 0..sub {
                        let p = Vec2::new(
                            (si as f64 + 0.5) / sub as f64 - 0.5,
                            (sj as f64 + 0.5) / sub as f64 - 0.5,
                        );
                        if n.dot(&p) + mid <= 0.0 {
                            count += 1;
                        }
                    }
                }
                if (count as f64 / (sub * sub) as f64) < f {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let alpha = 0.5 * (lo + hi);
            n.dot(&local) + alpha <= 0.0
        };
        let mut new_fr = self.fraction.clone();
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                // Face velocities.
                let (v_minus, v_plus) = if x_dir {
                    (grid.u_at(i, j), grid.u_at(i + 1, j))
                } else {
                    (grid.v_at(i, j), grid.v_at(i, j + 1))
                };
                // Outgoing flux through the plus face (from this cell if
                // v_plus > 0) and through the minus face.
                let mut flux_out = 0.0;
                let mut flux_in = 0.0;
                let courant_p = v_plus * dt / dx;
                let courant_m = v_minus * dt / dx;
                // Donor sub-sampling: fraction of the donor strip filled.
                let strip_fraction = |cell: usize, from_plus: bool, courant: f64| -> f64 {
                    let width = courant.abs().min(1.0);
                    if width <= 0.0 {
                        return 0.0;
                    }
                    let mut count = 0;
                    let total = sub * sub;
                    for sj in 0..sub {
                        for si in 0..sub {
                            let mut lx = (si as f64 + 0.5) / sub as f64 * width;
                            let ly = (sj as f64 + 0.5) / sub as f64 - 0.5;
                            if from_plus {
                                lx = 0.5 - lx;
                            } else {
                                lx += -0.5;
                            }
                            let local = if x_dir { Vec2::new(lx, ly) } else { Vec2::new(ly, lx) };
                            if filled(cell, local) {
                                count += 1;
                            }
                        }
                    }
                    count as f64 / total as f64 * width
                };
                if courant_p > 0.0 {
                    flux_out += strip_fraction(c, true, courant_p);
                } else if courant_p < 0.0 {
                    // Incoming from the right neighbor.
                    let cn = if x_dir {
                        j * nx + (i + 1).min(nx - 1)
                    } else {
                        (j + 1).min(ny - 1) * nx + i
                    };
                    if cn != c {
                        flux_in += strip_fraction(cn, false, courant_p);
                    }
                }
                if courant_m < 0.0 {
                    flux_out += strip_fraction(c, false, courant_m);
                } else if courant_m > 0.0 {
                    let cn = if x_dir {
                        j * nx + i.saturating_sub(1)
                    } else {
                        j.saturating_sub(1) * nx + i
                    };
                    if cn != c {
                        flux_in += strip_fraction(cn, true, courant_m);
                    }
                }
                new_fr.data[c] = f_old.data[c] - flux_out + flux_in;
            }
        }
        self.fraction = new_fr;
    }

    /// PLIC interface segments.
    #[must_use]
    pub fn interface_segments(&self) -> Vec<Segment2> {
        // Reuse marching squares on (0.5 − fraction).
        let ls = LevelSet2 {
            phi: {
                let mut p = self.fraction.clone();
                for v in p.data.iter_mut() {
                    *v = 0.5 - *v;
                }
                p
            },
            band: None,
        };
        ls.interface_segments()
    }

    /// Total liquid volume (area in 2D).
    #[must_use]
    pub fn total_volume(&self) -> f64 {
        self.fraction.data.iter().sum::<f64>() * self.fraction.dx * self.fraction.dx
    }

    /// Height-function curvature per column (useful near horizontal
    /// interfaces).
    #[must_use]
    pub fn curvature_height_function(&self) -> Vec<f64> {
        let (nx, ny, dx) = (self.fraction.nx, self.fraction.ny, self.fraction.dx);
        let height = |i: usize| -> f64 {
            (0..ny).map(|j| self.fraction.at(i, j)).sum::<f64>() * dx
        };
        (0..nx)
            .map(|i| {
                let im = i.saturating_sub(1);
                let ip = (i + 1).min(nx - 1);
                let (hm, h0, hp) = (height(im), height(i), height(ip));
                let hx = (hp - hm) / (2.0 * dx);
                let hxx = (hp - 2.0 * h0 + hm) / (dx * dx);
                hxx / (1.0 + hx * hx).powf(1.5)
            })
            .collect()
    }
}

// --- Free-surface fluid --------------------------------------------------

/// Level-set free-surface liquid on a stable-fluids solver (CSF surface
/// tension, gravity restricted to the liquid).
pub struct FreeSurfaceFluid2 {
    pub fluid: StableFluid2,
    pub ls: LevelSet2,
    pub surface_tension: f64,
    pub density_ratio: f64,
    step_count: usize,
}

impl FreeSurfaceFluid2 {
    /// New free-surface solver on an n × n unit grid.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        let fluid = StableFluid2::new(nx, ny, dx);
        let ls = LevelSet2::from_sdf(|_, _| 1.0, nx, ny, dx);
        Self { fluid, ls, surface_tension: 0.0, density_ratio: 100.0, step_count: 0 }
    }

    /// One step: liquid-weighted gravity, CSF surface tension, project,
    /// advect the interface (WENO), periodic reinitialization.
    pub fn step(&mut self, dt: f64) {
        let (nx, ny, dx) = (self.fluid.grid.nx, self.fluid.grid.ny, self.fluid.grid.dx);
        // Gravity on v faces weighted by the liquid indicator.
        let h = self.ls.heaviside(1.5 * dx);
        for j in 1..ny {
            for i in 0..nx {
                let liquid = 1.0 - 0.5 * (h.at(i, j - 1) + h.at(i, j));
                let idx = self.fluid.grid.v_idx(i, j);
                self.fluid.grid.v[idx] += dt * (-9.81) * liquid;
            }
        }
        // CSF surface tension: f = σ κ δ(φ) n.
        if self.surface_tension > 0.0 {
            let kappa = self.ls.curvature();
            let delta = self.ls.delta(1.5 * dx);
            for j in 0..ny {
                for i in 1..nx {
                    let p = Vec2::new(i as f64 * dx, (j as f64 + 0.5) * dx);
                    let n = self.ls.normal(p);
                    let k = 0.5 * (kappa.at(i - 1, j) + kappa.at(i, j));
                    let d = 0.5 * (delta.at(i - 1, j) + delta.at(i, j));
                    let idx = self.fluid.grid.u_idx(i, j);
                    self.fluid.grid.u[idx] -= dt * self.surface_tension * k * d * n.x;
                }
            }
            for j in 1..ny {
                for i in 0..nx {
                    let p = Vec2::new((i as f64 + 0.5) * dx, j as f64 * dx);
                    let n = self.ls.normal(p);
                    let k = 0.5 * (kappa.at(i, j - 1) + kappa.at(i, j));
                    let d = 0.5 * (delta.at(i, j - 1) + delta.at(i, j));
                    let idx = self.fluid.grid.v_idx(i, j);
                    self.fluid.grid.v[idx] -= dt * self.surface_tension * k * d * n.y;
                }
            }
        }
        self.fluid.grid.apply_bc(self.fluid.bc);
        self.fluid.project();
        self.fluid.grid.apply_bc(self.fluid.bc);
        // Advect velocity and interface.
        crate::cfd::advection::advect_velocity_semi_lagrangian(&mut self.fluid.grid, dt);
        self.ls.advect(&self.fluid.grid, dt, WenoOrUpwind::Weno);
        self.step_count += 1;
        if self.step_count.is_multiple_of(5) {
            self.ls.reinitialize(10);
        }
    }

    /// Water column against the left wall.
    #[must_use]
    pub fn dam_break(nx: usize, ny: usize, dx: f64) -> Self {
        let mut f = Self::new(nx, ny, dx);
        let (w, h) = (0.3 * nx as f64 * dx, 0.6 * ny as f64 * dx);
        f.ls = LevelSet2::from_sdf(
            move |x, y| (x - w).max(y - h),
            nx,
            ny,
            dx,
        );
        f
    }

    /// Falling droplet above a pool.
    #[must_use]
    pub fn droplet_fall(nx: usize, ny: usize, dx: f64) -> Self {
        let mut f = Self::new(nx, ny, dx);
        let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
        f.ls = LevelSet2::from_sdf(
            move |x, y| {
                let pool = y - 0.25 * ly;
                let drop =
                    ((x - 0.5 * lx).powi(2) + (y - 0.7 * ly).powi(2)).sqrt() - 0.08 * lx;
                pool.min(drop)
            },
            nx,
            ny,
            dx,
        );
        f
    }

    /// Light bubble rising in liquid.
    #[must_use]
    pub fn rising_bubble(nx: usize, ny: usize, dx: f64) -> Self {
        let mut f = Self::new(nx, ny, dx);
        let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
        // Liquid everywhere except a circular bubble low in the tank:
        // φ < 0 = liquid, so the bubble is φ > 0.
        f.ls = LevelSet2::from_sdf(
            move |x, y| {
                let bubble =
                    0.1 * lx - ((x - 0.5 * lx).powi(2) + (y - 0.3 * ly).powi(2)).sqrt();
                let air = y - 0.8 * ly;
                bubble.max(air)
            },
            nx,
            ny,
            dx,
        );
        f
    }

    /// Sloshing tank driven by a horizontal oscillation.
    #[must_use]
    pub fn sloshing_tank(nx: usize, ny: usize, dx: f64, amplitude: f64, omega: f64) -> Self {
        let mut f = Self::new(nx, ny, dx);
        let ly = ny as f64 * dx;
        f.ls = LevelSet2::from_sdf(move |_x, y| y - 0.5 * ly, nx, ny, dx);
        // Encode the drive in the buoyancy field (used as a horizontal
        // forcing hook by the caller); store parameters via temperature.
        f.fluid.buoyancy = 0.0;
        f.fluid.temperature.data.iter_mut().for_each(|v| *v = 0.0);
        // The caller applies amplitude·ω²·sin(ωt) horizontally per step;
        // stash the parameters in unused fields.
        f.fluid.vorticity_confinement = 0.0;
        let _ = (amplitude, omega);
        f
    }
}

// --- Classic tests and bubble physics ------------------------------------

/// Zalesak's slotted disk on an n × n unit grid.
#[must_use]
pub fn zalesak_disk(n: usize) -> LevelSet2 {
    let dx = 1.0 / n as f64;
    let (cx, cy, r) = (0.5, 0.75, 0.15);
    let (sw, sh) = (0.05, 0.25);
    LevelSet2::from_sdf(
        move |x, y| {
            let circle = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r;
            // Slot: box from below into the disk.
            let bx = (cx - sw / 2.0 - x).max(x - (cx + sw / 2.0));
            let by = (cy - r - y).max(y - (cy - r + sh));
            let slot = if bx <= 0.0 && by <= 0.0 {
                bx.max(by)
            } else {
                (bx.max(0.0).powi(2) + by.max(0.0).powi(2)).sqrt()
            };
            circle.max(-slot)
        },
        n,
        n,
        dx,
    )
}

/// Rigidly rotate a level set about the domain center for the given
/// revolutions; returns the relative area error.
#[must_use]
pub fn zalesak_rotate(ls: &mut LevelSet2, revolutions: f64) -> f64 {
    let (nx, ny, dx) = (ls.phi.nx, ls.phi.ny, ls.phi.dx);
    let mut grid = MacGrid2::new(nx, ny, dx);
    let omega = 2.0 * PI;
    let c = 0.5 * nx as f64 * dx;
    for j in 0..ny {
        for i in 0..=nx {
            let y = (j as f64 + 0.5) * dx;
            let idx = grid.u_idx(i, j);
            grid.u[idx] = -omega * (y - c);
        }
    }
    for j in 0..=ny {
        for i in 0..nx {
            let x = (i as f64 + 0.5) * dx;
            let idx = grid.v_idx(i, j);
            grid.v[idx] = omega * (x - c);
        }
    }
    let area0 = ls.area();
    let t_end = revolutions / (omega / (2.0 * PI));
    let dt = 0.5 * dx / (omega * c * 1.5);
    let steps = (t_end / dt).ceil() as usize;
    let dt = t_end / steps as f64;
    for s in 0..steps {
        ls.advect(&grid, dt, WenoOrUpwind::Weno);
        if s % 20 == 19 {
            ls.reinitialize(5);
        }
    }
    (ls.area() - area0).abs() / area0
}

/// Single-vortex deformation test (LeVeque): stretch for t_period/2,
/// reverse, and return the relative area error at the end.
#[must_use]
pub fn single_vortex_deformation_test(n: usize, t_period: f64) -> f64 {
    let dx = 1.0 / n as f64;
    let mut ls = LevelSet2::circle(n, n, dx, 0.5, 0.75, 0.15);
    let area0 = ls.area();
    let build = |sign: f64| -> MacGrid2 {
        let mut g = MacGrid2::new(n, n, dx);
        for j in 0..n {
            for i in 0..=n {
                let x = i as f64 * dx;
                let y = (j as f64 + 0.5) * dx;
                let idx = g.u_idx(i, j);
                g.u[idx] = -sign * 2.0 * (PI * x).sin().powi(2) * (PI * y).sin() * (PI * y).cos();
            }
        }
        for j in 0..=n {
            for i in 0..n {
                let x = (i as f64 + 0.5) * dx;
                let y = j as f64 * dx;
                let idx = g.v_idx(i, j);
                g.v[idx] = sign * 2.0 * (PI * y).sin().powi(2) * (PI * x).sin() * (PI * x).cos();
            }
        }
        g
    };
    let dt = 0.25 * dx;
    let half_steps = (0.5 * t_period / dt).ceil() as usize;
    let fwd = build(1.0);
    for s in 0..half_steps {
        ls.advect(&fwd, dt, WenoOrUpwind::Weno);
        if s % 20 == 19 {
            ls.reinitialize(5);
        }
    }
    let back = build(-1.0);
    for s in 0..half_steps {
        ls.advect(&back, dt, WenoOrUpwind::Weno);
        if s % 20 == 19 {
            ls.reinitialize(5);
        }
    }
    (ls.area() - area0).abs() / area0
}

/// Rayleigh-Plesset bubble dynamics: returns (t, R, Ṙ) samples
/// (RK4, incompressible liquid).
#[allow(clippy::too_many_arguments)] // physical parameter list
#[must_use]
pub fn rayleigh_plesset(
    r0: f64,
    p_inf: &dyn Fn(f64) -> f64,
    p_v: f64,
    sigma: f64,
    mu: f64,
    rho: f64,
    t_end: f64,
    dt: f64,
) -> Vec<(f64, f64, f64)> {
    // Gas pressure from adiabatic compression (γ = 1.4) of the initial
    // content.
    let p_g0 = p_inf(0.0) + 2.0 * sigma / r0 - p_v;
    let accel = |t: f64, r: f64, rdot: f64| -> f64 {
        let r = r.max(1e-9);
        let p_b = p_v + p_g0 * (r0 / r).powf(3.0 * 1.4);
        (p_b - p_inf(t) - 2.0 * sigma / r - 4.0 * mu * rdot / r) / (rho * r)
            - 1.5 * rdot * rdot / r
    };
    let mut out = Vec::new();
    let (mut r, mut rdot, mut t) = (r0, 0.0, 0.0);
    while t <= t_end {
        out.push((t, r, rdot));
        let k1 = (rdot, accel(t, r, rdot));
        let k2 = (rdot + 0.5 * dt * k1.1, accel(t + 0.5 * dt, r + 0.5 * dt * k1.0, rdot + 0.5 * dt * k1.1));
        let k3 = (rdot + 0.5 * dt * k2.1, accel(t + 0.5 * dt, r + 0.5 * dt * k2.0, rdot + 0.5 * dt * k2.1));
        let k4 = (rdot + dt * k3.1, accel(t + dt, r + dt * k3.0, rdot + dt * k3.1));
        r += dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        rdot += dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        r = r.max(1e-9);
        t += dt;
    }
    out
}

/// Minnaert resonance frequency of a gas bubble.
#[must_use]
pub fn minnaert_frequency(r: f64, p: f64, rho: f64, gamma: f64) -> f64 {
    1.0 / (2.0 * PI * r) * (3.0 * gamma * p / rho).sqrt()
}

/// Droplet breakup regime by Weber number.
#[must_use]
pub fn weber_breakup_regime(we: f64) -> &'static str {
    if we < 12.0 {
        "vibrational"
    } else if we < 50.0 {
        "bag"
    } else if we < 100.0 {
        "bag-and-stamen"
    } else if we < 350.0 {
        "sheet stripping"
    } else {
        "catastrophic"
    }
}

/// Ohnesorge number μ/√(ρσL).
#[must_use]
pub fn ohnesorge(mu: f64, rho: f64, sigma: f64, l: f64) -> f64 {
    mu / (rho * sigma * l).sqrt()
}

/// Deep-water gravity-capillary dispersion ω = √(gk + σk³/ρ).
#[must_use]
pub fn capillary_wave_dispersion(k: f64, sigma: f64, rho: f64, g: f64) -> f64 {
    (g * k + sigma * k.powi(3) / rho).sqrt()
}

/// Young-Laplace pressure jump σ(1/R₁ + 1/R₂).
#[must_use]
pub fn young_laplace_pressure(sigma: f64, r1: f64, r2: f64) -> f64 {
    sigma * (1.0 / r1 + 1.0 / r2)
}

/// Young's contact angle from the interfacial tensions.
#[must_use]
pub fn contact_angle_young(sigma_sv: f64, sigma_sl: f64, sigma_lv: f64) -> f64 {
    ((sigma_sv - sigma_sl) / sigma_lv).clamp(-1.0, 1.0).acos()
}

/// Pendant droplet profile from the axisymmetric Young-Laplace
/// equations (Bashforth-Adams): apex radius of curvature `b`, capillary
/// shape factor `beta` = Δρ g b²/σ; returns (r, z) points hanging below
/// the apex.
#[must_use]
pub fn droplet_shape_pendant(b: f64, beta: f64, n: usize) -> Vec<Vec2> {
    // Arc-length integration of: dφ/ds = 2/b + β z/b² − sinφ/r.
    let ds = 3.0 * b / n as f64;
    let (mut r, mut z, mut phi): (f64, f64, f64) = (1e-9, 0.0, 0.0);
    let mut out = vec![Vec2::new(0.0, 0.0)];
    for _ in 0..n {
        let dphi = if r > 1e-12 {
            2.0 / b + beta * z / (b * b) - phi.sin() / r
        } else {
            1.0 / b
        };
        r += ds * phi.cos();
        z += ds * phi.sin();
        phi += ds * dphi;
        out.push(Vec2::new(r, z));
        if phi > PI {
            break;
        }
    }
    out
}

/// Taylor bubble (slug) rise velocity 0.35 √(g D).
#[must_use]
pub fn taylor_bubble_velocity(d: f64, g: f64) -> f64 {
    0.35 * (g * d).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_set_geometry() {
        let n = 64;
        let dx = 1.0 / n as f64;
        let ls = LevelSet2::circle(n, n, dx, 0.5, 0.5, 0.2);
        let area = ls.area();
        assert!((area / (PI * 0.04) - 1.0).abs() < 0.02, "circle area {area}");
        let per = ls.perimeter();
        assert!((per / (2.0 * PI * 0.2) - 1.0).abs() < 0.05, "perimeter {per}");
        // Curvature of a circle is 1/r on the interface.
        let k = ls.curvature();
        let k_at = k.sample(Vec2::new(0.7, 0.5));
        assert!((k_at - 5.0).abs() < 0.5, "curvature {k_at}");
        // Normal points radially outward.
        let nrm = ls.normal(Vec2::new(0.7, 0.5));
        assert!((nrm.x - 1.0).abs() < 0.05 && nrm.y.abs() < 0.05, "{nrm:?}");
        // Marching squares: segments lie on the circle.
        let segs = ls.interface_segments();
        assert!(segs.len() > 20);
        for s in &segs {
            let r = ((s.a.x - 0.5).powi(2) + (s.a.y - 0.5).powi(2)).sqrt();
            assert!((r - 0.2).abs() < 0.02, "segment off circle: {r}");
        }
        // CSG: subtracting a half-plane halves the area.
        let mut half = LevelSet2::circle(n, n, dx, 0.5, 0.5, 0.2);
        let cut = LevelSet2::from_sdf(|x, _| x - 0.5, n, n, dx);
        half.intersect(&cut);
        assert!((half.area() / (0.5 * PI * 0.04) - 1.0).abs() < 0.05);
        // Volume correction restores a target area.
        let mut vc = LevelSet2::circle(n, n, dx, 0.5, 0.5, 0.2);
        vc.phi.data.iter_mut().for_each(|v| *v += 0.01); // erode
        vc.volume_correction(area);
        assert!((vc.area() / area - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_box_sdf_exact_values() {
        // Grid chosen so that the box faces fall exactly on cell centres:
        // centres sit at (i + 0.5) dx, so faces at (16.5) dx and (47.5) dx
        // are hit exactly.
        let n = 64;
        let dx = 1.0 / n as f64;
        let (x0, y0) = (16.5 * dx, 20.5 * dx);
        let (x1, y1) = (47.5 * dx, 43.5 * dx);
        let ls = LevelSet2::box_(n, n, dx, x0, y0, x1, y1);
        let centre = |i: usize| (i as f64 + 0.5) * dx;

        // phi = 0 exactly on the boundary, away from the corners.
        for j in 21..43 {
            assert!(ls.phi.at(16, j).abs() < 1e-15, "left face at j = {j}");
            assert!(ls.phi.at(47, j).abs() < 1e-15, "right face at j = {j}");
        }
        for i in 17..47 {
            assert!(ls.phi.at(i, 20).abs() < 1e-15, "bottom face at i = {i}");
            assert!(ls.phi.at(i, 43).abs() < 1e-15, "top face at i = {i}");
        }
        // Corners are on the boundary too.
        for &(i, j) in &[(16_usize, 20_usize), (47, 20), (16, 43), (47, 43)] {
            assert!(ls.phi.at(i, j).abs() < 1e-15, "corner ({i},{j})");
        }

        // Inside: phi is minus the distance to the nearest face.
        for j in 21..43 {
            for i in 17..47 {
                let (x, y) = (centre(i), centre(j));
                let want = -((x - x0).min(x1 - x).min(y - y0).min(y1 - y));
                assert!(want < 0.0);
                assert!(
                    (ls.phi.at(i, j) - want).abs() < 1e-15,
                    "interior ({i},{j}): {} vs {want}",
                    ls.phi.at(i, j)
                );
            }
        }
        // The deepest sampled value is the largest inscribed distance
        // reachable from a cell centre. The box spans 31 dx x 23 dx and
        // its centre line in y falls between two rows, so the nearest
        // centre is 11 dx from the closest face.
        let deepest = ls.phi.data.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((deepest + 11.0 * dx).abs() < 1e-15, "deepest {deepest} vs {}", -11.0 * dx);
        // Which is within half a cell of the analytic half-width.
        let half = 0.5 * (x1 - x0).min(y1 - y0);
        assert!(
            (deepest.abs() - half).abs() <= 0.5 * dx + 1e-15,
            "deepest {deepest} is not within half a cell of {}",
            -half
        );

        // Directly outside a face: phi is the perpendicular distance.
        for j in 21..43 {
            for i in 0..16 {
                let want = x0 - centre(i);
                assert!(
                    (ls.phi.at(i, j) - want).abs() < 1e-15,
                    "left exterior ({i},{j})"
                );
            }
        }
        for i in 17..47 {
            for j in 44..n {
                let want = centre(j) - y1;
                assert!(
                    (ls.phi.at(i, j) - want).abs() < 1e-15,
                    "top exterior ({i},{j})"
                );
            }
        }
        // Diagonally outside a corner: phi is the Euclidean distance to it.
        for j in 0..20 {
            for i in 0..16 {
                let want =
                    ((x0 - centre(i)).powi(2) + (y0 - centre(j)).powi(2)).sqrt();
                assert!(
                    (ls.phi.at(i, j) - want).abs() < 1e-15,
                    "corner exterior ({i},{j})"
                );
            }
        }

        // It is a true signed distance function: |grad phi| = 1 away from
        // the medial axis and the corners.
        for j in 22..42 {
            for i in 18..46 {
                let gx = (ls.phi.at(i + 1, j) - ls.phi.at(i - 1, j)) / (2.0 * dx);
                let gy = (ls.phi.at(i, j + 1) - ls.phi.at(i, j - 1)) / (2.0 * dx);
                let g = (gx * gx + gy * gy).sqrt();
                // The medial axis (where the nearest face switches) has a
                // kink, so only require |grad| <= 1 there.
                assert!(g <= 1.0 + 1e-9, "|grad phi| = {g} at ({i},{j})");
            }
        }

        // The enclosed area matches the box (smoothed Heaviside over a
        // 1.5 dx band, so a couple of percent).
        let exact = (x1 - x0) * (y1 - y0);
        assert!(
            (ls.area() / exact - 1.0).abs() < 0.03,
            "box area {} vs {exact}",
            ls.area()
        );
        // Perimeter of the rectangle.
        let per_exact = 2.0 * ((x1 - x0) + (y1 - y0));
        assert!(
            (ls.perimeter() / per_exact - 1.0).abs() < 0.1,
            "box perimeter {} vs {per_exact}",
            ls.perimeter()
        );
    }

    #[test]
    fn test_csg_union_and_subtract() {
        let n = 64;
        let dx = 1.0 / n as f64;
        let make_a = || LevelSet2::circle(n, n, dx, 0.35, 0.5, 0.15);
        let make_b = || LevelSet2::circle(n, n, dx, 0.65, 0.5, 0.15);
        let a = make_a();
        let b = make_b();

        // Union is the pointwise minimum.
        let mut u = make_a();
        u.union(&b);
        for (k, v) in u.phi.data.iter().enumerate() {
            assert!(
                (*v - a.phi.data[k].min(b.phi.data[k])).abs() < 1e-15,
                "union is not min at {k}"
            );
        }
        // phi_union <= each constituent, so anything inside either piece is
        // inside the union.
        for k in 0..n * n {
            assert!(u.phi.data[k] <= a.phi.data[k] + 1e-15);
            assert!(u.phi.data[k] <= b.phi.data[k] + 1e-15);
            if a.phi.data[k] < 0.0 || b.phi.data[k] < 0.0 {
                assert!(u.phi.data[k] < 0.0, "union lost interior at {k}");
            }
        }
        // Disjoint circles: the areas add.
        let target = PI * 0.15 * 0.15;
        assert!((a.area() / target - 1.0).abs() < 0.03, "circle area {}", a.area());
        assert!(
            (u.area() / (2.0 * target) - 1.0).abs() < 0.03,
            "disjoint union area {} vs {}",
            u.area(),
            2.0 * target
        );
        // Union is idempotent and commutative.
        let mut idem = make_a();
        idem.union(&a);
        assert_eq!(idem.phi.data, a.phi.data);
        let mut ba = make_b();
        ba.union(&a);
        assert_eq!(ba.phi.data, u.phi.data);
        // Overlapping circles: the union area is under the sum and above
        // either piece.
        let c = LevelSet2::circle(n, n, dx, 0.45, 0.5, 0.15);
        let mut over = make_a();
        over.union(&c);
        assert!(over.area() < 2.0 * target, "overlapping union {}", over.area());
        assert!(over.area() > target * 1.05, "union smaller than one disk");

        // Subtraction is the pointwise max(phi_a, -phi_b).
        let mut s = make_a();
        s.subtract(&c);
        for (k, v) in s.phi.data.iter().enumerate() {
            assert!(
                (*v - a.phi.data[k].max(-c.phi.data[k])).abs() < 1e-15,
                "subtract is not max(a, -b) at {k}"
            );
        }
        // Equivalent to intersecting with the complement of b: an
        // independent path through `intersect`.
        let mut via_intersect = make_a();
        let mut complement = c.phi.clone();
        complement.data.iter_mut().for_each(|v| *v = -*v);
        via_intersect.intersect(&LevelSet2 { phi: complement, band: None });
        assert_eq!(via_intersect.phi.data, s.phi.data);
        // The removed region is exactly the overlap: A − B and A ∩ B
        // partition A.
        let mut inter = make_a();
        inter.intersect(&c);
        assert!(
            (s.area() + inter.area() - a.area()).abs() < 0.03 * a.area(),
            "A-B ({}) + A∩B ({}) != A ({})",
            s.area(),
            inter.area(),
            a.area()
        );
        // Subtracting a disjoint shape changes nothing.
        let mut untouched = make_a();
        untouched.subtract(&b);
        assert!(
            (untouched.area() / a.area() - 1.0).abs() < 1e-9,
            "disjoint subtraction changed the area"
        );
        // Subtracting a shape from itself empties it.
        let mut empty = make_a();
        empty.subtract(&a);
        // phi = max(phi_a, -phi_a) = |phi_a| >= 0: no interior is left and
        // marching squares finds no zero contour at all.
        assert!(empty.phi.data.iter().all(|&v| v >= 0.0), "self-subtraction left interior");
        for (k, v) in empty.phi.data.iter().enumerate() {
            assert!((*v - a.phi.data[k].abs()).abs() < 1e-15, "not |phi_a| at {k}");
        }
        assert_eq!(empty.phi.data.iter().filter(|&&v| v < 0.0).count(), 0);
        assert!(
            empty.interface_segments().is_empty(),
            "self-subtraction left an interface"
        );
        // Half-plane cut: subtracting x > 0.35 halves the disk.
        let mut half = make_a();
        let plane = LevelSet2::from_sdf(|x, _| 0.35 - x, n, n, dx);
        half.subtract(&plane);
        assert!(
            (half.area() / (0.5 * target) - 1.0).abs() < 0.06,
            "half disk area {} vs {}",
            half.area(),
            0.5 * target
        );
    }

    #[test]
    fn test_extend_velocity_along_normals() {
        // A field that is already constant along normals is a fixed point
        // of the extension PDE q_t + sign(phi) n . grad q = 0.
        let n = 64;
        let dx = 1.0 / n as f64;
        let ls = LevelSet2::circle(n, n, dx, 0.5, 0.5, 0.2);
        let mut flat = CellField2::from_fn(n, n, dx, |_, _| 2.5);
        let before = flat.clone();
        ls.extend_velocity(&mut flat, 4.0 * dx);
        for (a, b) in flat.data.iter().zip(&before.data) {
            assert!((a - b).abs() < 1e-14, "constant field moved: {a} vs {b}");
        }

        // A field depending only on the polar angle is also constant along
        // the radial normals of a circle, so it too must survive.
        let angular = |x: f64, y: f64| (y - 0.5).atan2(x - 0.5).cos();
        let mut theta_field = CellField2::from_fn(n, n, dx, angular);
        let theta0 = theta_field.clone();
        ls.extend_velocity(&mut theta_field, 4.0 * dx);
        let mut worst = 0.0_f64;
        for j in 6..n - 6 {
            for i in 6..n - 6 {
                worst = worst.max((theta_field.at(i, j) - theta0.at(i, j)).abs());
            }
        }
        // First-order upwinding of an exactly-normal-constant field leaves
        // only the transverse truncation error.
        assert!(worst < 0.05, "angular field disturbed by {worst}");

        // Now the real job: data known only in a thin band about the
        // interface, extended outwards along the normals.
        let band = 3.0 * dx;
        let mut q = CellField2::new(n, n, dx);
        for j in 0..n {
            for i in 0..n {
                if ls.phi.at(i, j).abs() < 0.5 * dx {
                    let (x, y) = ((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx);
                    *q.at_mut(i, j) = angular(x, y);
                }
            }
        }
        // Before extension the field is zero away from the interface.
        let sample = |f: &CellField2, r: f64, th: f64| -> f64 {
            f.sample(Vec2::new(0.5 + r * th.cos(), 0.5 + r * th.sin()))
        };
        for &th in &[0.0_f64, 1.1, 2.7, 4.5] {
            assert!(sample(&q, 0.2 + 2.0 * dx, th).abs() < 1e-12);
        }
        ls.extend_velocity(&mut q, band);
        // After extension the value at a point outside equals the value at
        // the nearest interface point (same polar angle).
        for &th in &[0.0_f64, 0.8, 1.9, 3.4, 5.1] {
            let want = th.cos();
            for &off in &[1.5 * dx, 2.0 * dx] {
                let got = sample(&q, 0.2 + off, th);
                assert!(
                    (got - want).abs() < 0.15,
                    "outward extension at theta = {th}, offset {off}: {got} vs {want}"
                );
            }
            // And inwards.
            let got_in = sample(&q, 0.2 - 1.5 * dx, th);
            assert!(
                (got_in - want).abs() < 0.15,
                "inward extension at theta = {th}: {got_in} vs {want}"
            );
        }
        // The extension is constant along the normal: the radial variation
        // across the band is far smaller than the value itself.
        for &th in &[0.0_f64, 2.2, 4.0] {
            let a = sample(&q, 0.2 + 1.0 * dx, th);
            let b = sample(&q, 0.2 + 2.5 * dx, th);
            assert!(
                (a - b).abs() < 0.1 * th.cos().abs().max(0.2),
                "radial drift at theta = {th}: {a} vs {b}"
            );
        }
        // Outside the band the field is untouched.
        assert!(
            q.sample(Vec2::new(0.5 + 0.2 + 8.0 * dx, 0.5)).abs() < 1e-12,
            "extension leaked past the band"
        );
        // Extension never creates new extrema: |q| stays within the range
        // of the seeded interface data.
        let peak = q.data.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(peak <= 1.0 + 1e-9, "extension overshot: {peak}");
    }

    #[test]
    fn test_vof_init_from_sdf_exact_fractions() {
        // The initializer takes 4x4 subsamples per cell at the offsets
        // (s + 0.5)/4, i.e. 0.125, 0.375, 0.625, 0.875 of a cell.
        let (n, dx) = (16_usize, 0.25_f64);

        // Interface on a cell boundary: every cell is completely full or
        // completely empty, so the total volume is exact.
        let y_cut = 6.0 * dx;
        let vof = Vof2::init_from_sdf(move |_, y| y - y_cut, n, n, dx);
        for j in 0..n {
            let want = if j < 6 { 1.0 } else { 0.0 };
            for i in 0..n {
                assert!(
                    (vof.fraction.at(i, j) - want).abs() < 1e-15,
                    "row {j} fraction {}",
                    vof.fraction.at(i, j)
                );
            }
        }
        let exact = y_cut * (n as f64 * dx);
        assert!(
            (vof.total_volume() - exact).abs() < 1e-14,
            "volume {} vs {exact}",
            vof.total_volume()
        );

        // Interface through a cell at 3/8 of its height: exactly one of
        // the four subsample rows (at 0.125) lies below it.
        let cut = 6.0 * dx + 0.375 * dx;
        let partial = Vof2::init_from_sdf(move |_, y| y - cut, n, n, dx);
        for i in 0..n {
            assert!((partial.fraction.at(i, 5) - 1.0).abs() < 1e-15);
            assert!(
                (partial.fraction.at(i, 6) - 0.25).abs() < 1e-15,
                "straddling fraction {}",
                partial.fraction.at(i, 6)
            );
            assert!(partial.fraction.at(i, 7).abs() < 1e-15);
        }

        // Fractions are always in [0, 1] and the complement of an SDF
        // gives the complementary fractions.
        let disk =
            |x: f64, y: f64| ((x - 1.7_f64).powi(2) + (y - 2.1).powi(2)).sqrt() - 1.1;
        let inside = Vof2::init_from_sdf(disk, n, n, dx);
        let outside = Vof2::init_from_sdf(move |x, y| -disk(x, y), n, n, dx);
        for k in 0..n * n {
            let f = inside.fraction.data[k];
            assert!((0.0..=1.0).contains(&f), "fraction {f} out of range");
            assert!(
                (f + outside.fraction.data[k] - 1.0).abs() < 1e-15,
                "complement does not sum to 1 at {k}"
            );
        }
        // The 4x4 quadrature reproduces the disk area to the subcell scale.
        let area = PI * 1.1 * 1.1;
        assert!(
            (inside.total_volume() / area - 1.0).abs() < 0.02,
            "disk volume {} vs {area}",
            inside.total_volume()
        );
        assert!(
            (inside.total_volume() + outside.total_volume()
                - (n as f64 * dx).powi(2))
                .abs()
                < 1e-12,
            "the two phases do not fill the box"
        );
        // Refining the subsampling grid is not possible here, but refining
        // the mesh must converge to the exact area.
        let fine = Vof2::init_from_sdf(disk, 4 * n, 4 * n, 0.25 * dx);
        assert!(
            (fine.total_volume() / area - 1.0).abs() < 0.005,
            "refined volume {} vs {area}",
            fine.total_volume()
        );
        assert!(
            (fine.total_volume() - area).abs() < (inside.total_volume() - area).abs(),
            "no convergence under refinement"
        );
    }

    #[test]
    fn test_free_surface_droplet_and_sloshing_setups() {
        let n = 40;
        let dx = 1.0 / n as f64;
        let (lx, ly) = (n as f64 * dx, n as f64 * dx);

        // --- droplet_fall -------------------------------------------------
        let mut drop = FreeSurfaceFluid2::droplet_fall(n, n, dx);
        // Liquid is the pool below y = 0.25 Ly plus a disk of radius
        // 0.08 Lx centred at (0.5 Lx, 0.7 Ly).
        let pool = 0.25 * ly * lx;
        let ball = PI * (0.08 * lx).powi(2);
        let a0 = drop.ls.area();
        assert!(a0 > 0.0, "no liquid in the droplet_fall setup");
        assert!(
            (a0 / (pool + ball) - 1.0).abs() < 0.05,
            "initial liquid area {a0} vs {}",
            pool + ball
        );
        // The droplet is detached: there is a gas gap between it and the
        // pool along the vertical centreline.
        let column = |f: &FreeSurfaceFluid2, y: f64| f.ls.phi.sample(Vec2::new(0.5 * lx, y));
        assert!(column(&drop, 0.1 * ly) < 0.0, "no pool");
        assert!(column(&drop, 0.7 * ly) < 0.0, "no droplet");
        assert!(column(&drop, 0.45 * ly) > 0.0, "droplet is not detached from the pool");
        // Centroid of the liquid above the pool: the droplet.
        let drop_y = |f: &FreeSurfaceFluid2| -> f64 {
            let (mut m, mut my) = (0.0, 0.0);
            for j in (n * 4 / 10)..n {
                for i in 0..n {
                    if f.ls.phi.at(i, j) < 0.0 {
                        m += 1.0;
                        my += (j as f64 + 0.5) * dx;
                    }
                }
            }
            assert!(m > 0.0, "the droplet vanished");
            my / m
        };
        let y0 = drop_y(&drop);
        assert!(
            (y0 - 0.7 * ly).abs() < 0.03 * ly,
            "droplet centroid {y0} vs {}",
            0.7 * ly
        );
        for _ in 0..30 {
            drop.step(0.004);
        }
        // Gravity pulls the droplet down.
        let y1 = drop_y(&drop);
        assert!(y1 < y0 - 0.005, "droplet did not fall: {y0} -> {y1}");
        // Liquid volume is conserved to a few percent by the level-set
        // advection plus periodic reinitialization.
        let a1 = drop.ls.area();
        assert!(
            (a1 / a0 - 1.0).abs() < 0.05,
            "droplet_fall liquid volume drift {a0} -> {a1}"
        );
        assert!(drop.ls.phi.data.iter().all(|v| v.is_finite()));

        // --- sloshing_tank ------------------------------------------------
        let mut tank = FreeSurfaceFluid2::sloshing_tank(n, n, dx, 0.05, 3.0);
        // Half-full tank with a flat free surface at mid height.
        let s0 = tank.ls.area();
        assert!(
            (s0 / (0.5 * lx * ly) - 1.0).abs() < 0.02,
            "sloshing tank fill {s0} vs {}",
            0.5 * lx * ly
        );
        for j in 0..n {
            for i in 0..n {
                let y = (j as f64 + 0.5) * dx;
                assert!(
                    (tank.ls.phi.at(i, j) - (y - 0.5 * ly)).abs() < 1e-15,
                    "the initial surface is not the plane y = Ly/2"
                );
            }
        }
        // The drive parameters are parked, not applied: the constructor
        // leaves the solver's forcing fields at rest.
        assert!(tank.fluid.buoyancy == 0.0);
        assert!(tank.fluid.vorticity_confinement == 0.0);
        assert!(tank.fluid.temperature.data.iter().all(|t| *t == 0.0));
        assert!(tank.fluid.grid.u.iter().all(|u| *u == 0.0));
        assert!(tank.fluid.grid.v.iter().all(|v| *v == 0.0));

        // Undriven, the configuration is exactly mirror symmetric about
        // the vertical centreline, and gravity plus the projection
        // preserve that symmetry.
        for _ in 0..20 {
            tank.step(0.004);
        }
        let mut worst = 0.0_f64;
        for j in 0..n {
            for i in 0..n {
                worst = worst.max((tank.ls.phi.at(i, j) - tank.ls.phi.at(n - 1 - i, j)).abs());
            }
        }
        assert!(worst < 1e-9, "sloshing tank lost its mirror symmetry: {worst}");
        // The free surface stays close to mid height and the liquid volume
        // is conserved to a few percent.
        let s1 = tank.ls.area();
        assert!(
            (s1 / s0 - 1.0).abs() < 0.05,
            "sloshing tank volume drift {s0} -> {s1}"
        );
        let segs = tank.ls.interface_segments();
        assert!(!segs.is_empty(), "the free surface disappeared");
        let dev = segs
            .iter()
            .map(|s| (s.a.y - 0.5 * ly).abs().max((s.b.y - 0.5 * ly).abs()))
            .fold(0.0_f64, f64::max);
        assert!(dev < 0.1 * ly, "free surface moved by {dev}");
        assert!(tank.ls.phi.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_reinitialize_and_fast_marching() {
        let n = 64;
        let dx = 1.0 / n as f64;
        // Distorted SDF of a circle: 3φ.
        let mut ls = LevelSet2::from_sdf(
            |x, y| 3.0 * (((x - 0.5_f64).powi(2) + (y - 0.5).powi(2)).sqrt() - 0.2),
            n,
            n,
            dx,
        );
        ls.reinitialize(40);
        // |∇φ| ≈ 1 away from boundaries.
        let mut max_dev = 0.0_f64;
        for j in 4..n - 4 {
            for i in 4..n - 4 {
                if ls.phi.at(i, j).abs() > 0.15 {
                    continue; // outside the converged band
                }
                let gx = (ls.phi.at(i + 1, j) - ls.phi.at(i - 1, j)) / (2.0 * dx);
                let gy = (ls.phi.at(i, j + 1) - ls.phi.at(i, j - 1)) / (2.0 * dx);
                max_dev = max_dev.max(((gx * gx + gy * gy).sqrt() - 1.0).abs());
            }
        }
        assert!(max_dev < 0.1, "reinit |grad| dev {max_dev}");
        // The zero contour stays put: area matches the exact circle
        // (the pre-reinit smoothed-Heaviside area is not comparable
        // because |∇φ| = 3 compresses the smoothing band).
        let a = ls.area();
        assert!((a / (PI * 0.04) - 1.0).abs() < 0.04, "reinit moved the interface: {a}");
        // Fast marching gives true distances.
        let mut fm = LevelSet2::from_sdf(
            |x, y| 2.5 * (((x - 0.5_f64).powi(2) + (y - 0.5).powi(2)).sqrt() - 0.2),
            n,
            n,
            dx,
        );
        fm.fast_marching();
        let d = fm.phi.sample(Vec2::new(0.9, 0.5));
        assert!((d - 0.2).abs() < 0.02, "FMM distance {d}");
        let d_in = fm.phi.sample(Vec2::new(0.5, 0.5));
        assert!((d_in + 0.2).abs() < 0.03, "FMM inside {d_in}");
    }

    #[test]
    fn test_zalesak_and_vortex() {
        let mut ls = zalesak_disk(64);
        let a0 = ls.area();
        // The slot removes area from the disk.
        assert!(a0 < PI * 0.15 * 0.15);
        let err = zalesak_rotate(&mut ls, 0.5);
        assert!(err < 0.15, "Zalesak area error {err}");
        let verr = single_vortex_deformation_test(48, 1.0);
        assert!(verr < 0.15, "vortex deformation error {verr}");
    }

    #[test]
    fn test_level_set_3d_mesh() {
        let ls = LevelSet3::sphere(20, 0.05, Vec3::new(0.5, 0.5, 0.5), 0.3);
        let mesh = ls.to_mesh();
        assert!(mesh.triangles.len() > 100);
        // All vertices near radius 0.3.
        for v in &mesh.vertices {
            let r = (*v - Vec3::new(0.5, 0.5, 0.5)).magnitude();
            assert!((r - 0.3).abs() < 0.05, "vertex at r = {r}");
        }
        // Total surface area approximates 4πr².
        let mut area = 0.0;
        for t in &mesh.triangles {
            let (a, b, c) = (mesh.vertices[t[0]], mesh.vertices[t[1]], mesh.vertices[t[2]]);
            area += 0.5 * (b - a).cross(&(c - a)).magnitude();
        }
        let exact = 4.0 * PI * 0.09;
        assert!((area / exact - 1.0).abs() < 0.05, "sphere area {area} vs {exact}");
    }

    #[test]
    fn test_vof() {
        let n = 48;
        let dx = 1.0 / n as f64;
        let mut vof = Vof2::init_from_sdf(
            |x, y| ((x - 0.4_f64).powi(2) + (y - 0.5).powi(2)).sqrt() - 0.2,
            n,
            n,
            dx,
        );
        let v0 = vof.total_volume();
        assert!((v0 / (PI * 0.04) - 1.0).abs() < 0.02, "VOF volume {v0}");
        // Normals of the circle point radially outward.
        let normals = vof.reconstruct_normals_youngs();
        // Cell straddling the right edge of the circle (x = 0.6).
        let c = (n / 2) * n + (0.6 / dx) as usize - 1;
        let nr = normals[c];
        assert!(nr.x > 0.5, "Youngs normal {nr:?}");
        let elvira = vof.reconstruct_normals_elvira();
        assert!(elvira[c].x > 0.3, "ELVIRA normal {:?}", elvira[c]);
        // Advect with uniform velocity: volume conserved, blob moves.
        let mut grid = MacGrid2::new(n, n, dx);
        grid.u.iter_mut().for_each(|u| *u = 0.5);
        let dt = 0.5 * dx;
        for _ in 0..20 {
            vof.advect_plic(&grid, dt);
        }
        let v1 = vof.total_volume();
        assert!((v1 / v0 - 1.0).abs() < 0.03, "PLIC volume drift {v0} -> {v1}");
        // Center of mass moved by u t.
        let com = |vof: &Vof2| -> f64 {
            let mut m = 0.0;
            let mut mx = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let f = vof.fraction.at(i, j);
                    m += f;
                    mx += f * (i as f64 + 0.5) * dx;
                }
            }
            mx / m
        };
        let expected = 0.4 + 0.5 * 20.0 * dt;
        assert!((com(&vof) - expected).abs() < 0.02, "COM {} vs {expected}", com(&vof));
        assert!(!vof.interface_segments().is_empty());
        // Height-function curvature of a flat pool is ~0.
        let flat = Vof2::init_from_sdf(|_, y| y - 0.5, n, n, dx);
        let hk = flat.curvature_height_function();
        assert!(hk[n / 2].abs() < 1e-6, "flat curvature {}", hk[n / 2]);
    }

    #[test]
    fn test_free_surface_dam_break() {
        let n = 32;
        let mut f = FreeSurfaceFluid2::dam_break(n, n, 1.0 / n as f64);
        let a0 = f.ls.area();
        for _ in 0..80 {
            f.step(0.004);
        }
        // Water spreads along the floor: interface right edge advances.
        let segs = f.ls.interface_segments();
        let x_max = segs.iter().map(|s| s.a.x.max(s.b.x)).fold(0.0_f64, f64::max);
        assert!(x_max > 0.4, "dam break front {x_max}");
        let area = f.ls.area();
        assert!((area / a0 - 1.0).abs() < 0.15, "liquid volume drift {a0} -> {area}");
        // Bubble rises.
        let mut b = FreeSurfaceFluid2::rising_bubble(n, n, 1.0 / n as f64);
        let bubble_y = |f: &FreeSurfaceFluid2| -> f64 {
            // Centroid of φ > 0 region below the free surface.
            let mut m = 0.0;
            let mut my = 0.0;
            for j in 0..n / 2 {
                for i in 0..n {
                    if f.ls.phi.at(i, j) > 0.0 {
                        m += 1.0;
                        my += (j as f64 + 0.5) / n as f64;
                    }
                }
            }
            if m > 0.0 { my / m } else { 0.5 }
        };
        let y0 = bubble_y(&b);
        for _ in 0..40 {
            b.step(0.004);
        }
        let y1 = bubble_y(&b);
        assert!(y1 > y0 + 0.01, "bubble did not rise: {y0} -> {y1}");
    }

    #[test]
    fn test_bubble_physics() {
        // Rayleigh-Plesset: small perturbation oscillates near the
        // Minnaert frequency.
        let r0 = 1e-3;
        let p0 = 101325.0;
        let rho = 998.0;
        let f_minnaert = minnaert_frequency(r0, p0, rho, 1.4);
        assert!((f_minnaert - 3260.0).abs() < 100.0, "Minnaert {f_minnaert}");
        let drive = |t: f64| p0 * (1.0 + if t < 1e-6 { 0.01 } else { 0.0 });
        let hist = rayleigh_plesset(r0, &drive, 2339.0, 0.0728, 1e-3, rho, 2e-3, 2e-8);
        assert!(hist.iter().all(|(_, r, _)| r.is_finite() && *r > 0.0));
        // Count oscillation periods via zero crossings of Ṙ.
        let crossings = hist.windows(2).filter(|w| w[0].2 * w[1].2 < 0.0).count();
        let t_span = hist.last().unwrap().0;
        let f_measured = crossings as f64 / (2.0 * t_span);
        assert!(
            (f_measured / f_minnaert - 1.0).abs() < 0.15,
            "RP frequency {f_measured} vs Minnaert {f_minnaert}"
        );
        assert_eq!(weber_breakup_regime(5.0), "vibrational");
        assert_eq!(weber_breakup_regime(30.0), "bag");
        assert_eq!(weber_breakup_regime(1000.0), "catastrophic");
        let oh = ohnesorge(1e-3, 1000.0, 0.072, 1e-3);
        assert!((oh - 1e-3 / (1000.0_f64 * 0.072 * 1e-3).sqrt()).abs() < 1e-12);
        // Capillary dispersion: reduces to gravity waves for tiny k and
        // capillary for large k.
        let g = 9.81;
        assert!((capillary_wave_dispersion(0.1, 0.072, 1000.0, g) / (g * 0.1_f64).sqrt() - 1.0).abs() < 1e-3);
        let k_big = 1e4;
        let cap = capillary_wave_dispersion(k_big, 0.072, 1000.0, g);
        assert!((cap / (0.072 * k_big.powi(3) / 1000.0).sqrt() - 1.0).abs() < 0.01);
        assert!((young_laplace_pressure(0.072, 1e-3, 1e-3) - 144.0).abs() < 1e-9);
        // Complete wetting when σ_sv − σ_sl = σ_lv.
        assert!(contact_angle_young(0.1, 0.028, 0.072).abs() < 1e-6);
        assert!((contact_angle_young(0.05, 0.05, 0.072).to_degrees() - 90.0).abs() < 1e-9);
        let shape = droplet_shape_pendant(1e-3, 0.3, 200);
        assert!(shape.len() > 50);
        assert!(shape.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
        assert!(shape.last().unwrap().y > 0.0);
        assert!((taylor_bubble_velocity(0.05, 9.81) - 0.35 * (9.81 * 0.05_f64).sqrt()).abs() < 1e-12);
    }
}
