//! Advection schemes: classic 1D finite-volume methods (upwind,
//! Lax-Wendroff, MUSCL with slope limiters, WENO5), 2D semi-Lagrangian
//! transport with BFECC/MacCormack error compensation, SSP-RK3, Burgers
//! solvers, and the Cole-Hopf exact solution.

use crate::cfd::grid::{CellField2, MacGrid2};
use crate::math::Vec2;

fn pidx(i: i64, n: usize) -> usize {
    i.rem_euclid(n as i64) as usize
}

/// First-order upwind advection of a periodic 1D field by constant
/// velocity `u` for one step.
#[must_use]
pub fn advect_upwind_1d(q: &[f64], u: f64, dx: f64, dt: f64) -> Vec<f64> {
    let n = q.len();
    let c = u * dt / dx;
    (0..n as i64)
        .map(|i| {
            if u >= 0.0 {
                q[pidx(i, n)] - c * (q[pidx(i, n)] - q[pidx(i - 1, n)])
            } else {
                q[pidx(i, n)] - c * (q[pidx(i + 1, n)] - q[pidx(i, n)])
            }
        })
        .collect()
}

/// Second-order Lax-Wendroff advection (dispersive near discontinuities).
#[must_use]
pub fn advect_lax_wendroff_1d(q: &[f64], u: f64, dx: f64, dt: f64) -> Vec<f64> {
    let n = q.len();
    let c = u * dt / dx;
    (0..n as i64)
        .map(|i| {
            let qm = q[pidx(i - 1, n)];
            let q0 = q[pidx(i, n)];
            let qp = q[pidx(i + 1, n)];
            q0 - 0.5 * c * (qp - qm) + 0.5 * c * c * (qp - 2.0 * q0 + qm)
        })
        .collect()
}

/// Slope limiters for MUSCL-type schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limiter {
    Minmod,
    VanLeer,
    Superbee,
    Mc,
    VanAlbada,
    Koren,
}

fn limiter_phi(l: Limiter, r: f64) -> f64 {
    match l {
        Limiter::Minmod => r.clamp(0.0, 1.0),
        Limiter::VanLeer => {
            if r <= 0.0 { 0.0 } else { 2.0 * r / (1.0 + r) }
        }
        Limiter::Superbee => (2.0 * r).min(1.0).max(r.min(2.0)).max(0.0),
        Limiter::Mc => (2.0 * r).min(0.5 * (1.0 + r)).clamp(0.0, 2.0),
        Limiter::VanAlbada => {
            if r <= 0.0 { 0.0 } else { r * (1.0 + r) / (1.0 + r * r) }
        }
        Limiter::Koren => (2.0 * r).min((1.0 + 2.0 * r) / 3.0).clamp(0.0, 2.0),
    }
}

/// MUSCL advection with a slope limiter (TVD for Minmod/VanLeer/…).
#[must_use]
pub fn advect_muscl_1d(q: &[f64], u: f64, dx: f64, dt: f64, limiter: Limiter) -> Vec<f64> {
    let n = q.len();
    let c = u * dt / dx;
    // Limited upwind-biased face states at i+1/2 for u >= 0 (mirror for
    // u < 0).
    let face = |i: i64| -> f64 {
        if u >= 0.0 {
            let qm = q[pidx(i - 1, n)];
            let q0 = q[pidx(i, n)];
            let qp = q[pidx(i + 1, n)];
            let denom = qp - q0;
            let r = if denom.abs() < 1e-300 { 0.0 } else { (q0 - qm) / denom };
            q0 + 0.5 * limiter_phi(limiter, r) * (qp - q0)
        } else {
            let q0 = q[pidx(i + 1, n)];
            let qm = q[pidx(i + 2, n)];
            let qp = q[pidx(i, n)];
            let denom = qp - q0;
            let r = if denom.abs() < 1e-300 { 0.0 } else { (q0 - qm) / denom };
            q0 + 0.5 * limiter_phi(limiter, r) * (qp - q0)
        }
    };
    (0..n as i64)
        .map(|i| q[pidx(i, n)] - c * (face(i) - face(i - 1)))
        .collect()
}

/// WENO5 reconstruction of face states at every i+1/2: returns
/// (left-biased, right-biased) values, both length n (periodic).
#[must_use]
pub fn weno5_reconstruct(q: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = q.len();
    let eps = 1e-6;
    let weno_left = |a: f64, b: f64, c: f64, d: f64, e: f64| -> f64 {
        // Stencil values q[i-2..=i+2], reconstruct at i+1/2 from the left.
        let p0 = (2.0 * a - 7.0 * b + 11.0 * c) / 6.0;
        let p1 = (-b + 5.0 * c + 2.0 * d) / 6.0;
        let p2 = (2.0 * c + 5.0 * d - e) / 6.0;
        let b0 = 13.0 / 12.0 * (a - 2.0 * b + c).powi(2) + 0.25 * (a - 4.0 * b + 3.0 * c).powi(2);
        let b1 = 13.0 / 12.0 * (b - 2.0 * c + d).powi(2) + 0.25 * (b - d).powi(2);
        let b2 = 13.0 / 12.0 * (c - 2.0 * d + e).powi(2) + 0.25 * (3.0 * c - 4.0 * d + e).powi(2);
        let a0 = 0.1 / (eps + b0).powi(2);
        let a1 = 0.6 / (eps + b1).powi(2);
        let a2 = 0.3 / (eps + b2).powi(2);
        (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2)
    };
    let mut left = vec![0.0; n];
    let mut right = vec![0.0; n];
    for i in 0..n as i64 {
        left[i as usize] = weno_left(
            q[pidx(i - 2, n)],
            q[pidx(i - 1, n)],
            q[pidx(i, n)],
            q[pidx(i + 1, n)],
            q[pidx(i + 2, n)],
        );
        // Right-biased state at i+1/2 is the mirrored reconstruction.
        right[i as usize] = weno_left(
            q[pidx(i + 3, n)],
            q[pidx(i + 2, n)],
            q[pidx(i + 1, n)],
            q[pidx(i, n)],
            q[pidx(i - 1, n)],
        );
    }
    (left, right)
}

/// WENO5 upwind advection (Euler step in time).
#[must_use]
pub fn advect_weno5_1d(q: &[f64], u: f64, dx: f64, dt: f64) -> Vec<f64> {
    let n = q.len();
    let (left, right) = weno5_reconstruct(q);
    let face = |i: i64| -> f64 {
        if u >= 0.0 { left[pidx(i, n)] } else { right[pidx(i, n)] }
    };
    let c = u * dt / dx;
    (0..n as i64)
        .map(|i| q[pidx(i, n)] - c * (face(i) - face(i - 1)))
        .collect()
}

/// Semi-Lagrangian advection of a cell field through a MAC velocity
/// field (RK2 backtrace, bilinear sampling). Unconditionally stable.
#[must_use]
pub fn advect_semi_lagrangian_2d(q: &CellField2, grid: &MacGrid2, dt: f64) -> CellField2 {
    let mut out = q.clone();
    for j in 0..q.ny {
        for i in 0..q.nx {
            let p = Vec2::new((i as f64 + 0.5) * q.dx, (j as f64 + 0.5) * q.dx);
            let vmid = grid.velocity_at(p - grid.velocity_at(p) * (0.5 * dt));
            let back = p - vmid * dt;
            out.data[j * q.nx + i] = q.sample(back);
        }
    }
    out
}

/// Back-and-forth error compensation and correction (BFECC): second
/// order, limited to the local min/max to avoid new extrema.
#[must_use]
pub fn advect_bfecc_2d(q: &CellField2, grid: &MacGrid2, dt: f64) -> CellField2 {
    let forward = advect_semi_lagrangian_2d(q, grid, dt);
    let back = advect_semi_lagrangian_2d(&forward, grid, -dt);
    let mut compensated = q.clone();
    for (c, (&q0, &b)) in compensated.data.iter_mut().zip(q.data.iter().zip(&back.data)) {
        *c = q0 + 0.5 * (q0 - b);
    }
    let mut out = advect_semi_lagrangian_2d(&compensated, grid, dt);
    clamp_to_source(&mut out, q, grid, dt);
    out
}

/// Unsplit MacCormack advection with min/max limiting.
#[must_use]
pub fn advect_maccormack_2d(q: &CellField2, grid: &MacGrid2, dt: f64) -> CellField2 {
    let forward = advect_semi_lagrangian_2d(q, grid, dt);
    let back = advect_semi_lagrangian_2d(&forward, grid, -dt);
    let mut out = forward.clone();
    for (o, (&q0, &b)) in out.data.iter_mut().zip(q.data.iter().zip(&back.data)) {
        *o += 0.5 * (q0 - b);
    }
    clamp_to_source(&mut out, q, grid, dt);
    out
}

fn clamp_to_source(out: &mut CellField2, q: &CellField2, grid: &MacGrid2, dt: f64) {
    // Limit each result to the min/max of the 4 source cells around the
    // backtraced point (standard MacCormack/BFECC limiter).
    for j in 0..q.ny {
        for i in 0..q.nx {
            let p = Vec2::new((i as f64 + 0.5) * q.dx, (j as f64 + 0.5) * q.dx);
            let back = p - grid.velocity_at(p) * dt;
            let gx = (back.x / q.dx - 0.5).clamp(0.0, (q.nx - 1) as f64);
            let gy = (back.y / q.dx - 0.5).clamp(0.0, (q.ny - 1) as f64);
            let i0 = (gx.floor() as usize).min(q.nx - 2);
            let j0 = (gy.floor() as usize).min(q.ny - 2);
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for dj in 0..2 {
                for di in 0..2 {
                    let v = q.at(i0 + di, j0 + dj);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let c = &mut out.data[j * q.nx + i];
            *c = c.clamp(lo, hi);
        }
    }
}

/// First-order upwind advection on the 2D grid using face velocities.
#[must_use]
pub fn advect_upwind_2d(q: &CellField2, grid: &MacGrid2, dt: f64) -> CellField2 {
    let (nx, ny, dx) = (q.nx, q.ny, q.dx);
    let mut out = q.clone();
    let at = |i: i64, j: i64| -> f64 {
        q.at(i.clamp(0, nx as i64 - 1) as usize, j.clamp(0, ny as i64 - 1) as usize)
    };
    for j in 0..ny {
        for i in 0..nx {
            let u = 0.5 * (grid.u_at(i, j) + grid.u_at(i + 1, j));
            let v = 0.5 * (grid.v_at(i, j) + grid.v_at(i, j + 1));
            let (ii, jj) = (i as i64, j as i64);
            let dqdx = if u >= 0.0 {
                at(ii, jj) - at(ii - 1, jj)
            } else {
                at(ii + 1, jj) - at(ii, jj)
            };
            let dqdy = if v >= 0.0 {
                at(ii, jj) - at(ii, jj - 1)
            } else {
                at(ii, jj + 1) - at(ii, jj)
            };
            out.data[j * nx + i] = q.at(i, j) - dt / dx * (u * dqdx + v * dqdy);
        }
    }
    out
}

/// Advect the MAC velocity field itself semi-Lagrangianly (each face
/// component backtraced from its own staggered position).
pub fn advect_velocity_semi_lagrangian(grid: &mut MacGrid2, dt: f64) {
    let (nx, ny, dx) = (grid.nx, grid.ny, grid.dx);
    let mut new_u = grid.u.clone();
    let mut new_v = grid.v.clone();
    for j in 0..ny {
        for i in 0..=nx {
            let p = Vec2::new(i as f64 * dx, (j as f64 + 0.5) * dx);
            let back = p - grid.velocity_at(p) * dt;
            new_u[j * (nx + 1) + i] = grid.velocity_at(back).x;
        }
    }
    for j in 0..=ny {
        for i in 0..nx {
            let p = Vec2::new((i as f64 + 0.5) * dx, j as f64 * dx);
            let back = p - grid.velocity_at(p) * dt;
            new_v[j * nx + i] = grid.velocity_at(back).y;
        }
    }
    grid.u = new_u;
    grid.v = new_v;
}

/// Dimensionally split flux-limited (MUSCL) advection on the 2D grid.
#[must_use]
pub fn advect_flux_limited_2d(
    q: &CellField2,
    grid: &MacGrid2,
    dt: f64,
    limiter: Limiter,
) -> CellField2 {
    let (nx, ny) = (q.nx, q.ny);
    let mut mid = q.clone();
    // x sweep per row with the row-averaged u (fields advected by a MAC
    // grid velocity that varies smoothly; adequate for the split scheme).
    for j in 0..ny {
        let row: Vec<f64> = (0..nx).map(|i| q.at(i, j)).collect();
        let u_row = (0..nx)
            .map(|i| 0.5 * (grid.u_at(i, j) + grid.u_at(i + 1, j)))
            .sum::<f64>()
            / nx as f64;
        let adv = advect_muscl_1d(&row, u_row, q.dx, dt, limiter);
        for (i, v) in adv.iter().enumerate() {
            mid.data[j * nx + i] = *v;
        }
    }
    let mut out = mid.clone();
    for i in 0..nx {
        let col: Vec<f64> = (0..ny).map(|j| mid.at(i, j)).collect();
        let v_col = (0..ny)
            .map(|j| 0.5 * (grid.v_at(i, j) + grid.v_at(i, j + 1)))
            .sum::<f64>()
            / ny as f64;
        let adv = advect_muscl_1d(&col, v_col, q.dx, dt, limiter);
        for (j, v) in adv.iter().enumerate() {
            out.data[j * nx + i] = *v;
        }
    }
    out
}

/// Strong-stability-preserving third-order Runge-Kutta step for
/// dq/dt = rhs(q).
#[must_use]
pub fn rk3_ssp(q: &[f64], rhs: &dyn Fn(&[f64]) -> Vec<f64>, dt: f64) -> Vec<f64> {
    let n = q.len();
    let k1 = rhs(q);
    let q1: Vec<f64> = (0..n).map(|i| q[i] + dt * k1[i]).collect();
    let k2 = rhs(&q1);
    let q2: Vec<f64> = (0..n)
        .map(|i| 0.75 * q[i] + 0.25 * (q1[i] + dt * k2[i]))
        .collect();
    let k3 = rhs(&q2);
    (0..n)
        .map(|i| q[i] / 3.0 + 2.0 / 3.0 * (q2[i] + dt * k3[i]))
        .collect()
}

/// Spatial scheme selector for Burgers / advection-diffusion steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scheme {
    Upwind,
    LaxWendroff,
    Muscl(Limiter),
    Weno5,
}

fn burgers_flux_divergence(u: &[f64], dx: f64, scheme: Scheme) -> Vec<f64> {
    let n = u.len();
    // Godunov flux for f(u) = u²/2 at each face i+1/2 from (ul, ur).
    let godunov = |ul: f64, ur: f64| -> f64 {
        if ul <= ur {
            // Rarefaction: minimum of the flux over [ul, ur].
            if ul > 0.0 {
                0.5 * ul * ul
            } else if ur < 0.0 {
                0.5 * ur * ur
            } else {
                0.0
            }
        } else {
            // Shock: maximum.
            (0.5 * ul * ul).max(0.5 * ur * ur)
        }
    };
    let (fl, fr): (Vec<f64>, Vec<f64>) = match scheme {
        Scheme::Weno5 => weno5_reconstruct(u),
        Scheme::Muscl(l) => {
            let mut left = vec![0.0; n];
            let mut right = vec![0.0; n];
            for i in 0..n as i64 {
                let qm = u[pidx(i - 1, n)];
                let q0 = u[pidx(i, n)];
                let qp = u[pidx(i + 1, n)];
                let d = qp - q0;
                let r = if d.abs() < 1e-300 { 0.0 } else { (q0 - qm) / d };
                left[i as usize] = q0 + 0.5 * limiter_phi(l, r) * d;
                let q0r = qp;
                let qmr = u[pidx(i + 2, n)];
                let dr = q0 - q0r;
                let rr = if dr.abs() < 1e-300 { 0.0 } else { (q0r - qmr) / dr };
                right[i as usize] = q0r + 0.5 * limiter_phi(l, rr) * dr;
            }
            (left, right)
        }
        _ => {
            // Piecewise-constant states.
            let left: Vec<f64> = (0..n as i64).map(|i| u[pidx(i, n)]).collect();
            let right: Vec<f64> = (0..n as i64).map(|i| u[pidx(i + 1, n)]).collect();
            (left, right)
        }
    };
    let flux: Vec<f64> = (0..n).map(|i| godunov(fl[i], fr[i])).collect();
    (0..n as i64)
        .map(|i| (flux[pidx(i, n)] - flux[pidx(i - 1, n)]) / dx)
        .collect()
}

/// One explicit step of viscous Burgers u_t + (u²/2)_x = ν u_xx on a
/// periodic domain.
#[must_use]
pub fn burgers_step(u: &[f64], dx: f64, dt: f64, nu: f64, scheme: Scheme) -> Vec<f64> {
    let n = u.len();
    match scheme {
        Scheme::LaxWendroff => {
            // Richtmyer two-step Lax-Wendroff for the flux + viscosity.
            let f = |v: f64| 0.5 * v * v;
            let half: Vec<f64> = (0..n as i64)
                .map(|i| {
                    let q0 = u[pidx(i, n)];
                    let qp = u[pidx(i + 1, n)];
                    0.5 * (q0 + qp) - 0.5 * dt / dx * (f(qp) - f(q0))
                })
                .collect();
            (0..n as i64)
                .map(|i| {
                    let visc = nu * (u[pidx(i + 1, n)] - 2.0 * u[pidx(i, n)] + u[pidx(i - 1, n)])
                        / (dx * dx);
                    u[pidx(i, n)] - dt / dx * (f(half[pidx(i, n)]) - f(half[pidx(i - 1, n)]))
                        + dt * visc
                })
                .collect()
        }
        _ => {
            let dfdx = burgers_flux_divergence(u, dx, scheme);
            (0..n as i64)
                .map(|i| {
                    let visc = nu * (u[pidx(i + 1, n)] - 2.0 * u[pidx(i, n)] + u[pidx(i - 1, n)])
                        / (dx * dx);
                    u[pidx(i, n)] - dt * dfdx[pidx(i, n)] + dt * visc
                })
                .collect()
        }
    }
}

/// Exact viscous Burgers solution by the Cole-Hopf transform:
/// u(x,t) = ∫ ((x−y)/t) e^{−G/2ν} dy / ∫ e^{−G/2ν} dy with
/// G(y) = (x−y)²/(2t) + ∫₀^y u₀.
#[must_use]
pub fn burgers_exact_cole_hopf(x: f64, t: f64, nu: f64, u0: &dyn Fn(f64) -> f64) -> f64 {
    if t <= 0.0 {
        return u0(x);
    }
    let half_width = 12.0 * (nu * t).sqrt().max(1.0);
    let n = 4001;
    let dy = 2.0 * half_width / (n - 1) as f64;
    // Cumulative ∫ u0 from the left edge (the constant offset cancels in
    // the ratio).
    let mut f_cum = vec![0.0; n];
    let mut prev = u0(x - half_width);
    let mut acc = 0.0;
    for (k, fc) in f_cum.iter_mut().enumerate().skip(1) {
        let y = x - half_width + k as f64 * dy;
        let cur = u0(y);
        acc += 0.5 * (prev + cur) * dy;
        *fc = acc;
        prev = cur;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    // Normalize the exponent to avoid underflow.
    let g_min = (0..n)
        .map(|k| {
            let y = x - half_width + k as f64 * dy;
            (x - y).powi(2) / (2.0 * t) + f_cum[k]
        })
        .fold(f64::INFINITY, f64::min);
    for (k, &fc) in f_cum.iter().enumerate() {
        let y = x - half_width + k as f64 * dy;
        let g = (x - y).powi(2) / (2.0 * t) + fc;
        let w = (-(g - g_min) / (2.0 * nu)).exp();
        num += (x - y) / t * w;
        den += w;
    }
    num / den.max(1e-300)
}

/// One explicit step of 1D advection-diffusion q_t + u q_x = D q_xx.
#[must_use]
pub fn advection_diffusion_1d(
    q: &[f64],
    u: f64,
    d: f64,
    dx: f64,
    dt: f64,
    scheme: Scheme,
) -> Vec<f64> {
    let n = q.len();
    let advected = match scheme {
        Scheme::Upwind => advect_upwind_1d(q, u, dx, dt),
        Scheme::LaxWendroff => advect_lax_wendroff_1d(q, u, dx, dt),
        Scheme::Muscl(l) => advect_muscl_1d(q, u, dx, dt, l),
        Scheme::Weno5 => advect_weno5_1d(q, u, dx, dt),
    };
    (0..n as i64)
        .map(|i| {
            advected[pidx(i, n)]
                + d * dt / (dx * dx)
                    * (q[pidx(i + 1, n)] - 2.0 * q[pidx(i, n)] + q[pidx(i - 1, n)])
        })
        .collect()
}

/// Cell Péclet number u dx / D.
#[must_use]
pub fn peclet_cell(u: f64, dx: f64, d: f64) -> f64 {
    u * dx / d
}

/// Total variation Σ |q_{i+1} − q_i| (periodic).
#[must_use]
pub fn total_variation(q: &[f64]) -> f64 {
    let n = q.len();
    (0..n as i64).map(|i| (q[pidx(i + 1, n)] - q[pidx(i, n)]).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfd::grid::MacGrid2;
    use crate::monte_carlo::Rng;

    const TWO_PI: f64 = 2.0 * crate::math::constants::PI;

    fn advect_periodic_error(
        n: usize,
        step: impl Fn(&[f64], f64, f64, f64) -> Vec<f64>,
    ) -> f64 {
        // Advect sin over one full period at CFL 0.5, L2 error vs exact.
        let dx = 1.0 / n as f64;
        let u = 1.0;
        let dt = 0.5 * dx / u;
        let steps = (1.0 / (u * dt)).round() as usize;
        let mut q: Vec<f64> = (0..n).map(|i| (TWO_PI * i as f64 * dx).sin()).collect();
        for _ in 0..steps {
            q = step(&q, u, dx, dt);
        }
        let exact: Vec<f64> = (0..n).map(|i| (TWO_PI * i as f64 * dx).sin()).collect();
        (q.iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt()
    }

    #[test]
    fn test_upwind_monotone_tvd() {
        let mut rng = Rng::new(3);
        let mut q: Vec<f64> = (0..128).map(|_| rng.next_f64()).collect();
        let tv0 = total_variation(&q);
        let mut tv_prev = tv0;
        for _ in 0..100 {
            q = advect_upwind_1d(&q, 1.0, 0.01, 0.005);
            let tv = total_variation(&q);
            assert!(tv <= tv_prev + 1e-12, "TV grew: {tv} > {tv_prev}");
            tv_prev = tv;
        }
        // Monotone: no new extrema beyond initial bounds.
        assert!(q.iter().all(|&v| (-1e-12..=1.0 + 1e-12).contains(&v)));
    }

    #[test]
    fn test_lax_wendroff_second_order() {
        let e1 = advect_periodic_error(64, advect_lax_wendroff_1d);
        let e2 = advect_periodic_error(128, advect_lax_wendroff_1d);
        let ratio = e1 / e2;
        assert!((3.2..5.0).contains(&ratio), "LW order ratio {ratio}");
        // Upwind is only first order by comparison.
        let u1 = advect_periodic_error(64, advect_upwind_1d);
        let u2 = advect_periodic_error(128, advect_upwind_1d);
        let uratio = u1 / u2;
        assert!((1.6..2.6).contains(&uratio), "upwind order ratio {uratio}");
    }

    #[test]
    fn test_weno5_fifth_order_reconstruction() {
        // Face-value reconstruction error on smooth data scales as h^5.
        let err = |n: usize| -> f64 {
            let dx = 1.0 / n as f64;
            let q: Vec<f64> = (0..n)
                .map(|i| {
                    // Cell average of sin over the cell (exact).
                    let a = i as f64 * dx;
                    let b = a + dx;
                    ((TWO_PI * a).cos() - (TWO_PI * b).cos()) / (TWO_PI * dx)
                })
                .collect();
            let (left, _) = weno5_reconstruct(&q);
            (0..n)
                .map(|i| {
                    let xf = (i as f64 + 1.0) * dx;
                    (left[i] - (TWO_PI * xf).sin()).abs()
                })
                .fold(0.0_f64, f64::max)
        };
        let e1 = err(32);
        let e2 = err(64);
        let ratio = e1 / e2;
        assert!(ratio > 24.0, "WENO5 order ratio {ratio} (want ~32)");
    }

    #[test]
    fn test_muscl_minmod_tvd_on_step() {
        let n = 128;
        let mut q: Vec<f64> = (0..n).map(|i| if (32..64).contains(&i) { 1.0 } else { 0.0 }).collect();
        let tv0 = total_variation(&q);
        for _ in 0..200 {
            q = advect_muscl_1d(&q, 1.0, 1.0 / n as f64, 0.4 / n as f64, Limiter::Minmod);
            assert!(total_variation(&q) <= tv0 + 1e-9);
        }
        assert!(q.iter().all(|&v| (-1e-9..=1.0 + 1e-9).contains(&v)));
        // MUSCL is sharper than upwind on the same problem.
        let mut qu: Vec<f64> =
            (0..n).map(|i| if (32..64).contains(&i) { 1.0 } else { 0.0 }).collect();
        for _ in 0..200 {
            qu = advect_upwind_1d(&qu, 1.0, 1.0 / n as f64, 0.4 / n as f64);
        }
        assert!(total_variation(&q) > total_variation(&qu));
    }

    #[test]
    fn test_semi_lagrangian_stability_and_accuracy() {
        let n = 32;
        let dx = 1.0 / n as f64;
        let mut grid = MacGrid2::new(n, n, dx);
        grid.u.iter_mut().for_each(|u| *u = 1.0);
        grid.v.iter_mut().for_each(|v| *v = 0.5);
        let q = CellField2::from_fn(n, n, dx, |x, y| {
            (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.02).exp()
        });
        // CFL 10: dt = 10 dx / |u|.
        let dt = 10.0 * dx / 1.0;
        let mut cur = q.clone();
        let max0 = q.data.iter().cloned().fold(0.0_f64, f64::max);
        for _ in 0..100 {
            cur = advect_semi_lagrangian_2d(&cur, &grid, dt);
            let max = cur.data.iter().cloned().fold(0.0_f64, f64::max);
            assert!(max <= max0 + 1e-9, "SL blew up: {max}");
            assert!(cur.data.iter().all(|v| v.is_finite()));
        }
        // BFECC and MacCormack reduce the SL error on a small step.
        let dt_small = 0.5 * dx;
        let exact = |x: f64, y: f64, t: f64| {
            (-((x - 0.5 - t).powi(2) + (y - 0.5 - 0.5 * t).powi(2)) / 0.02).exp()
        };
        let steps = 8;
        let mut sl = q.clone();
        let mut bf = q.clone();
        let mut mc = q.clone();
        for _ in 0..steps {
            sl = advect_semi_lagrangian_2d(&sl, &grid, dt_small);
            bf = advect_bfecc_2d(&bf, &grid, dt_small);
            mc = advect_maccormack_2d(&mc, &grid, dt_small);
        }
        let t = steps as f64 * dt_small;
        let err = |f: &CellField2| -> f64 {
            let mut e = 0.0;
            for j in 4..n - 4 {
                for i in 4..n - 4 {
                    let x = (i as f64 + 0.5) * dx;
                    let y = (j as f64 + 0.5) * dx;
                    e += (f.at(i, j) - exact(x, y, t)).powi(2);
                }
            }
            e.sqrt()
        };
        let (e_sl, e_bf, e_mc) = (err(&sl), err(&bf), err(&mc));
        assert!(e_bf < 0.7 * e_sl, "BFECC {e_bf} vs SL {e_sl}");
        assert!(e_mc < 0.7 * e_sl, "MacCormack {e_mc} vs SL {e_sl}");
    }

    #[test]
    fn test_2d_upwind_flux_limited_and_velocity_advect() {
        let n = 32;
        let dx = 1.0 / n as f64;
        let mut grid = MacGrid2::new(n, n, dx);
        grid.u.iter_mut().for_each(|u| *u = 0.7);
        let q = CellField2::from_fn(n, n, dx, |x, _| (TWO_PI * x).sin());
        let up = advect_upwind_2d(&q, &grid, 0.5 * dx);
        assert!(up.data.iter().all(|v| v.is_finite()));
        let fl = advect_flux_limited_2d(&q, &grid, 0.5 * dx, Limiter::VanLeer);
        assert!(fl.data.iter().all(|v| v.is_finite()));
        // Uniform velocity field is a fixed point of self-advection.
        let mut g2 = MacGrid2::new(n, n, dx);
        g2.u.iter_mut().for_each(|u| *u = 1.0);
        g2.v.iter_mut().for_each(|v| *v = -0.5);
        advect_velocity_semi_lagrangian(&mut g2, 0.02);
        assert!(g2.u.iter().all(|&u| (u - 1.0).abs() < 1e-9));
        assert!(g2.v.iter().all(|&v| (v + 0.5).abs() < 1e-9));
    }

    #[test]
    fn test_rk3_ssp_order() {
        // dq/dt = -q: compare against exp for two step sizes.
        let rhs = |q: &[f64]| -> Vec<f64> { q.iter().map(|v| -v).collect() };
        let solve = |dt: f64| -> f64 {
            let mut q = vec![1.0];
            let steps = (1.0 / dt) as usize;
            for _ in 0..steps {
                q = rk3_ssp(&q, &rhs, dt);
            }
            (q[0] - (-1.0_f64).exp()).abs()
        };
        let e1 = solve(0.1);
        let e2 = solve(0.05);
        let ratio = e1 / e2;
        assert!((6.0..11.0).contains(&ratio), "RK3 order ratio {ratio}");
    }

    #[test]
    fn test_burgers_vs_cole_hopf() {
        let n = 256;
        let dx = 1.0 / n as f64;
        let nu = 0.05;
        let u0 = |x: f64| (TWO_PI * x).sin();
        let mut u: Vec<f64> = (0..n).map(|i| u0((i as f64 + 0.5) * dx)).collect();
        let t_end = 0.1;
        let dt = 0.8 * (dx * dx / (2.0 * nu)).min(0.2 * dx);
        let steps = (t_end / dt).ceil() as usize;
        let dt = t_end / steps as f64;
        for _ in 0..steps {
            u = burgers_step(&u, dx, dt, nu, Scheme::Weno5);
        }
        let mut max_err = 0.0_f64;
        for (i, &ui) in u.iter().enumerate() {
            let x = (i as f64 + 0.5) * dx;
            let exact = burgers_exact_cole_hopf(x, t_end, nu, &u0);
            max_err = max_err.max((ui - exact).abs());
        }
        assert!(max_err < 1e-3, "Burgers error {max_err}");
        // Lax-Wendroff variant also runs stably here.
        let mut ul: Vec<f64> = (0..n).map(|i| u0((i as f64 + 0.5) * dx)).collect();
        for _ in 0..steps {
            ul = burgers_step(&ul, dx, dt, nu, Scheme::LaxWendroff);
        }
        assert!(ul.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_advection_diffusion_and_misc() {
        let n = 64;
        let dx = 1.0 / n as f64;
        // Pure diffusion decay rate of the k=1 mode: exp(-D (2πk)² t).
        let d = 0.01;
        let dt = 0.2 * dx * dx / d;
        let mut q: Vec<f64> = (0..n).map(|i| (TWO_PI * i as f64 * dx).sin()).collect();
        let steps = 200;
        for _ in 0..steps {
            q = advection_diffusion_1d(&q, 0.0, d, dx, dt, Scheme::Upwind);
        }
        let t = steps as f64 * dt;
        let expected = (-d * TWO_PI * TWO_PI * t).exp();
        let amp = q.iter().cloned().fold(0.0_f64, f64::max);
        assert!((amp / expected - 1.0).abs() < 0.02, "diffusion decay {amp} vs {expected}");
        assert!((peclet_cell(2.0, 0.1, 0.05) - 4.0).abs() < 1e-12);
        let tv = total_variation(&[0.0, 1.0, 0.0]);
        assert!((tv - 2.0).abs() < 1e-12);
        // All limiters are bounded and pass through φ(1) = 1.
        for l in [
            Limiter::Minmod,
            Limiter::VanLeer,
            Limiter::Superbee,
            Limiter::Mc,
            Limiter::VanAlbada,
            Limiter::Koren,
        ] {
            assert!((limiter_phi(l, 1.0) - 1.0).abs() < 1e-12, "{l:?} at 1");
            assert_eq!(limiter_phi(l, -1.0), 0.0, "{l:?} negative r");
            for k in 0..50 {
                let r = 0.1 * k as f64;
                let phi = limiter_phi(l, r);
                assert!((0.0..=2.0 + 1e-12).contains(&phi), "{l:?} out of TVD bounds");
            }
        }
    }
}
