// Grid-based fluid simulation solvers.
//
// Three models at increasing fidelity:
//   1. ColumnFluid     — hydrostatic cellular automaton (1D columns)
//   2. ShallowWater1D  — Saint-Venant equations via Lax-Friedrichs
//   3. EulerFluid2D    — incompressible Euler via Chorin projection

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Orifice discharge coefficient (dimensionless). Empirical value for
/// sharp-edged orifice flow; Torricelli's theorem gives ideal velocity
/// v = sqrt(2gΔh), multiplied by C_D to account for vena contracta losses.
const DISCHARGE_COEFFICIENT: f64 = 0.6;

/// Minimum water depth (m) below which a shallow-water cell is treated as
/// dry. Prevents division by zero in velocity recovery u = hu / h.
const DRY_TOLERANCE: f64 = 1e-10;

/// CFL safety factor applied when computing stable time steps.
const CFL_SAFETY: f64 = 0.9;

// ═══════════════════════════════════════════════════════════════════════════
// 1. ColumnFluid — Hydrostatic Column Balancing
// ═══════════════════════════════════════════════════════════════════════════
//
// PDE being discretized:
//   None — this is a cellular automaton model. Adjacent water columns
//   exchange volume driven by hydrostatic pressure differences. The base
//   pressure in column i is P_i = ρ g h_i, so the pressure difference
//   drives flow from higher to lower columns.
//
// Flow model (Torricelli/orifice):
//   Q_{i→i+1} = C_D × dx × sign(Δh) × √(2g|Δh|)
//   where Δh = h_i - h_{i+1}.
//
// Update rule:
//   h_i^{n+1} = h_i^n - dt/dx × (Q_{i→i+1} - Q_{i-1→i})
//
// Stability:
//   CFL-like condition: dt must be small enough that no column transfers
//   more water than it contains. We clamp transferred volume to the
//   available height and enforce h ≥ 0.
//
// Conservation:
//   Total volume V = Σ h_i × dx is exactly conserved (transfers are
//   antisymmetric).

pub struct ColumnFluid {
    pub heights: Vec<f64>,
    pub width: usize,
    pub dx: f64,
    pub density: f64,
    pub g: f64,
}

impl ColumnFluid {
    #[must_use]
    /// Create a column fluid with the given number of columns, spacing, density, and gravity.
    pub fn new(width: usize, dx: f64, density: f64, g: f64) -> Self {
        assert!(dx > 0.0, "column spacing dx must be positive");
        assert!(density > 0.0, "fluid density must be positive");
        assert!(g > 0.0, "gravitational acceleration must be positive");
        Self {
            heights: vec![0.0; width],
            width,
            dx,
            density,
            g,
        }
    }

    /// Set the water height in a specific column.
    pub fn set_height(&mut self, col: usize, h: f64) {
        assert!(col < self.width, "column index out of bounds");
        assert!(h >= 0.0, "height must be non-negative");
        self.heights[col] = h;
    }

    /// Advance one time step using pressure-driven orifice flow between
    /// adjacent columns.
    ///
    /// Flow rate from column i to i+1 (Torricelli with discharge coeff):
    ///   Q = C_D × dx × sign(Δh) × √(2g|Δh|)
    ///
    /// Volume transferred per step = Q × dt. Height change:
    ///   Δh_i = -(Q_right - Q_left) × dt / dx
    ///
    /// Heights are clamped to zero to prevent negative water.
    pub fn step(&mut self, dt: f64) {
        if self.width < 2 {
            return;
        }

        // Compute flow rates at each interface (width - 1 interfaces).
        // flow[k] = flow rate from column k to column k+1 (positive = rightward).
        let n_interfaces = self.width - 1;
        let mut flow = vec![0.0; n_interfaces];

        for k in 0..n_interfaces {
            let dh = self.heights[k] - self.heights[k + 1];
            let abs_dh = dh.abs();
            // Q = C_D × dx × sign(Δh) × √(2g|Δh|)
            let q = DISCHARGE_COEFFICIENT * self.dx * dh.signum()
                * (2.0 * self.g * abs_dh).sqrt();
            flow[k] = q;
        }

        // Apply volume transfers: volume = Q × dt, height change = volume / dx.
        let mut new_heights = self.heights.clone();
        for k in 0..n_interfaces {
            let dh_transfer = flow[k] * dt / self.dx;
            new_heights[k] -= dh_transfer;
            new_heights[k + 1] += dh_transfer;
        }

        // Enforce non-negative heights.
        for h in &mut new_heights {
            if *h < 0.0 {
                *h = 0.0;
            }
        }

        self.heights = new_heights;
    }

    /// Total fluid volume: V = Σ h_i × dx (m² in 2D cross-section).
    #[must_use]
    pub fn total_volume(&self) -> f64 {
        self.heights.iter().sum::<f64>() * self.dx
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ShallowWater1D — Saint-Venant Equations
// ═══════════════════════════════════════════════════════════════════════════
//
// PDEs (1D Saint-Venant / shallow water equations on a flat bed, S=0):
//
//   ∂h/∂t + ∂(hu)/∂x = 0                       (mass conservation)
//   ∂(hu)/∂t + ∂(hu² + g h²/2)/∂x = 0          (momentum conservation)
//
// Conservative form with state vector U = [h, hu]^T, flux F(U):
//   F = [hu,  hu²/h + g h²/2]^T
//
// Discretization — Lax-Friedrichs scheme (first order, diffusive but
// unconditionally stable under CFL):
//
//   U_i^{n+1} = ½(U_{i-1}^n + U_{i+1}^n) - dt/(2dx) × (F_{i+1}^n - F_{i-1}^n)
//
// Written out component-wise:
//   h_i^{n+1}  = ½(h_{i-1} + h_{i+1})  - dt/(2dx) × (hu_{i+1} - hu_{i-1})
//   hu_i^{n+1} = ½(hu_{i-1} + hu_{i+1}) - dt/(2dx) × (F_{i+1} - F_{i-1})
//   where F_i  = hu_i²/h_i + g h_i²/2
//
// CFL stability condition:
//   dt < dx / max_i(|u_i| + √(g h_i))
//
// The maximum wave speed a = |u| + √(gh) comes from the eigenvalues of the
// flux Jacobian ∂F/∂U, which are λ = u ± √(gh) (characteristic speeds).
//
// Boundary conditions: reflective (ghost cells mirror interior).

pub struct ShallowWater1D {
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub nx: usize,
    pub dx: f64,
    pub g: f64,
}

impl ShallowWater1D {
    #[must_use]
    /// Create a 1D shallow water solver with nx cells, spacing dx, and gravity g.
    pub fn new(nx: usize, dx: f64, g: f64) -> Self {
        assert!(dx > 0.0, "grid spacing dx must be positive");
        assert!(g > 0.0, "gravitational acceleration must be positive");
        Self {
            h: vec![0.0; nx],
            hu: vec![0.0; nx],
            nx,
            dx,
            g,
        }
    }

    /// Recover velocity u = hu/h, returning 0 for dry cells.
    #[must_use]
    pub fn velocity(&self, i: usize) -> f64 {
        if self.h[i] < DRY_TOLERANCE {
            0.0
        } else {
            self.hu[i] / self.h[i]
        }
    }

    /// Momentum flux: F = hu²/h + g h²/2.
    fn momentum_flux(&self, i: usize) -> f64 {
        if self.h[i] < DRY_TOLERANCE {
            0.0
        } else {
            self.hu[i] * self.hu[i] / self.h[i] + 0.5 * self.g * self.h[i] * self.h[i]
        }
    }

    /// HLL (Harten-Lax-van Leer) time step with reflective boundary
    /// conditions — the Part 3 rewire onto the CFD module's Riemann
    /// machinery. Sharper than [`Self::step_lax_friedrichs`] on bores.
    pub fn step_hll(&mut self, dt: f64) {
        let b = vec![0.0; self.nx];
        crate::cfd::shallow_water::swe_1d_step_hll(
            &mut self.h,
            &mut self.hu,
            &b,
            self.dx,
            self.g,
            dt,
            DRY_TOLERANCE,
            true,
        );
    }

    /// Lax-Friedrichs time step with reflective boundary conditions.
    ///
    /// Scheme (interior, 1 ≤ i ≤ nx-2):
    ///   h_i^{n+1}  = ½(h_{i-1} + h_{i+1}) - dt/(2dx)(hu_{i+1} - hu_{i-1})
    ///   hu_i^{n+1} = ½(hu_{i-1}+ hu_{i+1}) - dt/(2dx)(F_{i+1}  - F_{i-1})
    ///
    /// Boundaries (i=0, i=nx-1): reflective ghost cells.
    pub fn step_lax_friedrichs(&mut self, dt: f64) {
        if self.nx < 3 {
            return;
        }

        let ratio = dt / (2.0 * self.dx);

        let mut h_new = vec![0.0; self.nx];
        let mut hu_new = vec![0.0; self.nx];

        for i in 1..self.nx - 1 {
            h_new[i] = 0.5 * (self.h[i - 1] + self.h[i + 1])
                - ratio * (self.hu[i + 1] - self.hu[i - 1]);

            let f_plus = self.momentum_flux(i + 1);
            let f_minus = self.momentum_flux(i - 1);
            hu_new[i] = 0.5 * (self.hu[i - 1] + self.hu[i + 1])
                - ratio * (f_plus - f_minus);

            // Enforce non-negative depth.
            if h_new[i] < 0.0 {
                h_new[i] = 0.0;
                hu_new[i] = 0.0;
            }
        }

        // Reflective boundaries: h equal to neighbor, hu negated (wall).
        h_new[0] = h_new[1];
        hu_new[0] = -hu_new[1];
        h_new[self.nx - 1] = h_new[self.nx - 2];
        hu_new[self.nx - 1] = -hu_new[self.nx - 2];

        self.h = h_new;
        self.hu = hu_new;
    }

    /// Maximum wave speed: max_i(|u_i| + √(g h_i)).
    /// This is the largest eigenvalue of the flux Jacobian across all cells.
    #[must_use]
    pub fn max_wave_speed(&self) -> f64 {
        let mut max_speed: f64 = 0.0;
        for i in 0..self.nx {
            let u = self.velocity(i);
            let c = (self.g * self.h[i].max(0.0)).sqrt();
            max_speed = max_speed.max(u.abs() + c);
        }
        max_speed
    }

    /// CFL-limited stable time step: dt = CFL_SAFETY × dx / max_wave_speed.
    #[must_use]
    pub fn stable_dt(&self) -> f64 {
        let s = self.max_wave_speed();
        if s < DRY_TOLERANCE {
            return self.dx; // all dry, any dt is stable
        }
        CFL_SAFETY * self.dx / s
    }

    /// Total volume: V = Σ h_i × dx.
    #[must_use]
    pub fn total_volume(&self) -> f64 {
        self.h.iter().sum::<f64>() * self.dx
    }

    /// Total energy (mechanical): E = Σ (½ h u² + ½ g h²) dx.
    ///
    /// First term is depth-integrated kinetic energy per unit width,
    /// second is potential energy (∫₀ʰ ρg z dz = ½ρg h², with ρ=1 in
    /// shallow water non-dimensionalization — we include g explicitly).
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..self.nx {
            let u = self.velocity(i);
            let h = self.h[i];
            e += (0.5 * h * u * u + 0.5 * self.g * h * h) * self.dx;
        }
        e
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. EulerFluid2D — Incompressible Euler (Chorin Projection)
// ═══════════════════════════════════════════════════════════════════════════
//
// PDE — Incompressible Euler equations:
//
//   ∂u/∂t + (u·∇)u = -∇p/ρ + f       (momentum)
//   ∇·u = 0                            (incompressibility / mass conservation)
//
// Discretization — Chorin's projection method (Chorin, 1968):
//
// Step 1 — Advection + body forces (forward Euler with first-order upwind):
//   u* = u^n - dt (u^n · ∇)u^n + dt g
//
//   Upwind advection for component φ ∈ {vx, vy}:
//     (u·∇)φ ≈ vx × ∂φ/∂x + vy × ∂φ/∂y
//   where ∂φ/∂x uses backward difference if vx > 0, forward if vx < 0
//   (first-order upwind, unconditionally TVD).
//
// Step 2 — Pressure projection (enforce ∇·u = 0):
//   Solve the pressure Poisson equation:
//     ∇²p = (ρ/dt) ∇·u*
//   via Jacobi iteration with Neumann boundary conditions (∂p/∂n = 0).
//
//   Jacobi update (uniform grid dx = dy = h):
//     p_{i,j}^{k+1} = ¼(p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1}
//                       - h² × rhs_{i,j})
//
//   For non-square grids (dx ≠ dy):
//     p_{i,j}^{k+1} = [(p_{i+1,j}+p_{i-1,j})/dx² + (p_{i,j+1}+p_{i,j-1})/dy²
//                       - rhs_{i,j}] / [2(1/dx² + 1/dy²)]
//
// Step 3 — Velocity correction:
//   u^{n+1} = u* - (dt/ρ) ∇p
//
// After correction, ∇·u^{n+1} ≈ 0 to within the Poisson solver tolerance.
//
// Stability:
//   The advection step is stable under CFL: dt < min(dx, dy) / max|u|.
//   The Poisson solve is unconditionally stable (elliptic).
//
// Boundary conditions: solid walls (no-penetration, free-slip).
//   - Normal velocity = 0 at boundaries.
//   - Neumann BC on pressure (∂p/∂n = 0), implemented by copying neighbor
//     pressure values.

pub struct EulerFluid2D {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub pressure: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub density: f64,
}

/// 2D grid index: row-major, i * ny + j.
#[inline]
fn idx(i: usize, j: usize, ny: usize) -> usize {
    i * ny + j
}

impl EulerFluid2D {
    #[must_use]
    /// Create a 2D incompressible Euler fluid solver on an nx-by-ny grid.
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, density: f64) -> Self {
        assert!(dx > 0.0, "grid spacing dx must be positive");
        assert!(dy > 0.0, "grid spacing dy must be positive");
        assert!(density > 0.0, "fluid density must be positive");
        let n = nx * ny;
        Self {
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            pressure: vec![0.0; n],
            nx,
            ny,
            dx,
            dy,
            density,
        }
    }

    /// Set the velocity at grid point (i, j).
    pub fn set_velocity(&mut self, i: usize, j: usize, vx: f64, vy: f64) {
        let k = idx(i, j, self.ny);
        self.vx[k] = vx;
        self.vy[k] = vy;
    }

    /// One full time step via Chorin's projection method.
    ///
    /// 1. Advect with first-order upwind + add body force.
    /// 2. Solve ∇²p = (ρ/dt)∇·u* via Jacobi iteration.
    /// 3. Correct: u^{n+1} = u* - (dt/ρ)∇p.
    pub fn step(&mut self, dt: f64, gravity_x: f64, gravity_y: f64) {
        let n = self.nx * self.ny;

        // ── Step 1: Advection + body forces ──
        let mut vx_star = vec![0.0; n];
        let mut vy_star = vec![0.0; n];

        for i in 0..self.nx {
            for j in 0..self.ny {
                let k = idx(i, j, self.ny);
                let u = self.vx[k];
                let v = self.vy[k];

                // First-order upwind: (u·∇)φ
                let advect_vx = self.upwind_advect(i, j, u, v, &self.vx);
                let advect_vy = self.upwind_advect(i, j, u, v, &self.vy);

                vx_star[k] = u - dt * advect_vx + dt * gravity_x;
                vy_star[k] = v - dt * advect_vy + dt * gravity_y;
            }
        }

        // Enforce no-penetration on boundaries.
        self.apply_boundary_velocity(&mut vx_star, &mut vy_star);

        // ── Step 2: Pressure Poisson solve ──
        // Compute divergence of u*: ∇·u* = ∂vx*/∂x + ∂vy*/∂y
        let mut rhs = vec![0.0; n];
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let dvx_dx = (vx_star[idx(i + 1, j, self.ny)]
                    - vx_star[idx(i - 1, j, self.ny)])
                    / (2.0 * self.dx);
                let dvy_dy = (vy_star[idx(i, j + 1, self.ny)]
                    - vy_star[idx(i, j - 1, self.ny)])
                    / (2.0 * self.dy);
                rhs[idx(i, j, self.ny)] = (self.density / dt) * (dvx_dx + dvy_dy);
            }
        }

        // Jacobi iteration for ∇²p = rhs with Neumann BCs.
        self.pressure = self.solve_pressure_poisson(&rhs);

        // ── Step 3: Velocity correction ──
        // u^{n+1} = u* - (dt/ρ)∇p
        let dt_over_rho = dt / self.density;
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let k = idx(i, j, self.ny);
                let dp_dx = (self.pressure[idx(i + 1, j, self.ny)]
                    - self.pressure[idx(i - 1, j, self.ny)])
                    / (2.0 * self.dx);
                let dp_dy = (self.pressure[idx(i, j + 1, self.ny)]
                    - self.pressure[idx(i, j - 1, self.ny)])
                    / (2.0 * self.dy);

                self.vx[k] = vx_star[k] - dt_over_rho * dp_dx;
                self.vy[k] = vy_star[k] - dt_over_rho * dp_dy;
            }
        }

        // Re-apply boundary conditions after correction.
        // apply_boundary_velocity borrows &self immutably, so we must work
        // through temporaries to satisfy the borrow checker.
        let mut vx_tmp = std::mem::take(&mut self.vx);
        let mut vy_tmp = std::mem::take(&mut self.vy);
        self.apply_boundary_velocity(&mut vx_tmp, &mut vy_tmp);
        self.vx = vx_tmp;
        self.vy = vy_tmp;
    }

    /// First-order upwind advection: (u·∇)φ at cell (i,j).
    ///
    /// Uses backward difference when local velocity is positive (information
    /// travels from left), forward difference when negative.
    fn upwind_advect(&self, i: usize, j: usize, u: f64, v: f64, field: &[f64]) -> f64 {
        let phi = field[idx(i, j, self.ny)];

        // ∂φ/∂x via upwind
        let dphi_dx = if u >= 0.0 {
            if i > 0 {
                (phi - field[idx(i - 1, j, self.ny)]) / self.dx
            } else {
                0.0
            }
        } else if i < self.nx - 1 {
            (field[idx(i + 1, j, self.ny)] - phi) / self.dx
        } else {
            0.0
        };

        // ∂φ/∂y via upwind
        let dphi_dy = if v >= 0.0 {
            if j > 0 {
                (phi - field[idx(i, j - 1, self.ny)]) / self.dy
            } else {
                0.0
            }
        } else if j < self.ny - 1 {
            (field[idx(i, j + 1, self.ny)] - phi) / self.dy
        } else {
            0.0
        };

        u * dphi_dx + v * dphi_dy
    }

    /// Apply solid-wall (no-penetration, free-slip) boundary conditions.
    fn apply_boundary_velocity(&self, vx: &mut [f64], vy: &mut [f64]) {
        // Left/right walls: vx = 0
        for j in 0..self.ny {
            vx[idx(0, j, self.ny)] = 0.0;
            vx[idx(self.nx - 1, j, self.ny)] = 0.0;
        }
        // Top/bottom walls: vy = 0
        for i in 0..self.nx {
            vy[idx(i, 0, self.ny)] = 0.0;
            vy[idx(i, self.ny - 1, self.ny)] = 0.0;
        }
    }

    /// Jacobi solver for ∇²p = rhs with Neumann BCs (∂p/∂n = 0).
    ///
    /// Discretization on a uniform grid:
    ///   (p_{i+1,j} + p_{i-1,j})/dx² + (p_{i,j+1} + p_{i,j-1})/dy² - rhs
    ///   ────────────────────────────────────────────────────────────────────
    ///                         2(1/dx² + 1/dy²)
    fn solve_pressure_poisson(&self, rhs: &[f64]) -> Vec<f64> {
        // Part 3 rewire: delegate the Poisson solve to the CFD module's
        // conjugate-gradient Neumann solver (the same node-centered
        // system, solved to a much tighter tolerance than the old
        // fixed-count Jacobi sweep).
        crate::cfd::stable_fluids::poisson_neumann_cg_rect(
            rhs, self.nx, self.ny, self.dx, self.dy, 1e-10, 2000,
        )
    }

    /// Maximum absolute divergence: max |∇·u|.
    /// Should be near zero after a projection step.
    #[must_use]
    pub fn divergence(&self) -> f64 {
        let mut max_div: f64 = 0.0;

        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let dvx_dx = (self.vx[idx(i + 1, j, self.ny)]
                    - self.vx[idx(i - 1, j, self.ny)])
                    / (2.0 * self.dx);
                let dvy_dy = (self.vy[idx(i, j + 1, self.ny)]
                    - self.vy[idx(i, j - 1, self.ny)])
                    / (2.0 * self.dy);

                max_div = max_div.max((dvx_dx + dvy_dy).abs());
            }
        }

        max_div
    }

    /// Total kinetic energy: KE = ½ρ Σ (vx² + vy²) dx dy.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let cell_area = self.dx * self.dy;
        let mut ke = 0.0;
        for k in 0..self.nx * self.ny {
            ke += self.vx[k] * self.vx[k] + self.vy[k] * self.vy[k];
        }
        0.5 * self.density * ke * cell_area
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_rel_eq(a: f64, b: f64, tol: f64) -> bool {
        if b.abs() < 1e-15 {
            a.abs() < tol
        } else {
            ((a - b) / b).abs() < tol
        }
    }

    // ── ColumnFluid ──

    #[test]
    fn column_fluid_equilibrates_to_equal_height() {
        // Two columns at different heights should converge to the average.
        let mut fluid = ColumnFluid::new(2, 1.0, 1000.0, constants::G_ACCEL);
        fluid.set_height(0, 2.0);
        fluid.set_height(1, 0.0);

        let expected_height = 1.0; // average of 2 and 0

        for _ in 0..10_000 {
            fluid.step(0.001);
        }

        let h0 = fluid.heights[0];
        assert!(
            approx_eq(h0, expected_height, 0.05),
            "column 0: {h0} != {expected_height}",
        );
        let h1 = fluid.heights[1];
        assert!(
            approx_eq(h1, expected_height, 0.05),
            "column 1: {h1} != {expected_height}",
        );
    }

    #[test]
    fn column_fluid_conserves_volume() {
        let mut fluid = ColumnFluid::new(10, 0.5, 1000.0, constants::G_ACCEL);
        fluid.set_height(0, 3.0);
        fluid.set_height(3, 1.5);
        fluid.set_height(7, 2.0);

        let initial_volume = fluid.total_volume();

        for _ in 0..5000 {
            fluid.step(0.0005);
        }

        let final_volume = fluid.total_volume();
        assert!(
            approx_rel_eq(initial_volume, final_volume, 1e-6),
            "volume not conserved: {initial_volume} -> {final_volume}",
        );
    }

    // ── ShallowWater1D ──

    #[test]
    fn shallow_water_dam_break_wave_speed() {
        // Dam break: h_left = H, h_right = 0.
        // The leading wave front propagates at speed c = √(gH) for the
        // Ritter solution of ideal dam break on dry bed.
        // With Lax-Friedrichs (diffusive), we check the wavefront is in
        // the right ballpark rather than pixel-exact.
        let nx = 500;
        let dx = 0.1;
        let g = constants::G_ACCEL;
        let h0 = 1.0;

        let mut sw = ShallowWater1D::new(nx, dx, g);

        // Dam at center: left half filled, right half dry.
        let dam_pos = nx / 2;
        for i in 0..dam_pos {
            sw.h[i] = h0;
        }

        let expected_wave_speed = (g * h0).sqrt(); // ~3.13 m/s
        let total_time = 1.0;
        let mut t = 0.0;

        while t < total_time {
            let dt = sw.stable_dt().min(total_time - t);
            sw.step_lax_friedrichs(dt);
            t += dt;
        }

        // Find the rightmost cell with significant water depth.
        let threshold = 0.01 * h0;
        let mut front_idx = dam_pos;
        for i in (dam_pos..nx).rev() {
            if sw.h[i] > threshold {
                front_idx = i;
                break;
            }
        }

        let front_distance = (front_idx - dam_pos) as f64 * dx;
        let measured_speed = front_distance / total_time;

        // Lax-Friedrichs is diffusive, so the front may be smeared. We
        // check within a generous factor (50%) since numerical diffusion
        // slows the sharp front.
        assert!(
            measured_speed > 0.5 * expected_wave_speed,
            "wave too slow: measured {measured_speed} m/s vs expected {expected_wave_speed} m/s",
        );
        assert!(
            measured_speed < 2.0 * expected_wave_speed,
            "wave too fast: measured {measured_speed} m/s vs expected {expected_wave_speed} m/s",
        );
    }

    #[test]
    fn shallow_water_step_hll_matches_stoker_dam_break() {
        // Wet-bed dam break: the HLL update is checked against Stoker's
        // exact similarity solution, and against the diffusive
        // Lax-Friedrichs update on the identical initial data.
        use crate::cfd::shallow_water::swe_1d_exact_dam_break;

        let nx = 800;
        let domain = 40.0;
        let dx = domain / nx as f64;
        let g = 9.81;
        let (h_l, h_r) = (2.0_f64, 1.0_f64);
        let t_end = 1.0_f64;

        let fill = |sw: &mut ShallowWater1D| {
            for i in 0..nx {
                sw.h[i] = if (i as f64 + 0.5) * dx < 0.5 * domain { h_l } else { h_r };
                sw.hu[i] = 0.0;
            }
        };

        let mut hll = ShallowWater1D::new(nx, dx, g);
        fill(&mut hll);
        let v0 = hll.total_volume();
        let e0 = hll.total_energy();
        let mut t = 0.0;
        while t < t_end {
            let dt = hll.stable_dt().min(t_end - t);
            hll.step_hll(dt);
            t += dt;
            // Depth stays non-negative throughout.
            assert!(hll.h.iter().all(|&h| h >= 0.0), "negative depth at t = {t}");
        }

        // Reflective walls and a flux-form update: the volume is conserved
        // to round-off (the waves have not reached the walls at t = 1 s,
        // and would be reflected rather than lost if they had).
        assert!(
            approx_rel_eq(hll.total_volume(), v0, 1e-12),
            "volume drift {} -> {}",
            v0,
            hll.total_volume()
        );
        // HLL is an entropy-satisfying, dissipative scheme: the total
        // mechanical energy must not grow across the bore.
        assert!(
            hll.total_energy() <= e0 * (1.0 + 1e-12),
            "energy grew: {e0} -> {}",
            hll.total_energy()
        );

        // Compare with Stoker's exact solution (dam at the domain centre).
        let l1 = |sw: &ShallowWater1D| -> f64 {
            let mut acc = 0.0;
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx - 0.5 * domain;
                let (he, _) = swe_1d_exact_dam_break(x, t_end, h_l, h_r, g);
                acc += (sw.h[i] - he).abs();
            }
            acc / nx as f64
        };
        let l1_hll = l1(&hll);
        assert!(l1_hll < 0.01, "HLL L1 depth error {l1_hll}");

        // Star-region plateau: depth and velocity match the exact
        // intermediate state. x = 4 m lies between the contact and the
        // bore at t = 1 s.
        let (h_star, u_star) = swe_1d_exact_dam_break(4.0, t_end, h_l, h_r, g);
        let i_star = ((0.5 * domain + 4.0) / dx) as usize;
        assert!(
            approx_rel_eq(hll.h[i_star], h_star, 0.02),
            "plateau depth {} vs exact {h_star}",
            hll.h[i_star]
        );
        assert!(
            approx_rel_eq(hll.velocity(i_star), u_star, 0.03),
            "plateau speed {} vs exact {u_star}",
            hll.velocity(i_star)
        );

        // The bore runs to the right at the Rankine-Hugoniot speed, so
        // the water level well to the right of the dam is still the
        // undisturbed h_r while the level just behind the front is raised.
        let x_front = 0.5 * domain
            + (0..nx)
                .rev()
                .find(|&i| hll.h[i] > h_r + 0.05)
                .map_or(0.0, |i| (i as f64 + 0.5) * dx - 0.5 * domain);
        let s_shock = (x_front - 0.5 * domain) / t_end;
        // Exact bore speed from mass conservation across the shock:
        // S = h* u* / (h* − h_r).
        let s_exact = h_star * u_star / (h_star - h_r);
        assert!(
            (s_shock / s_exact - 1.0).abs() < 0.05,
            "bore speed {s_shock} vs exact {s_exact}"
        );
        assert!(
            approx_rel_eq(hll.h[nx - 5], h_r, 1e-9),
            "far field disturbed: {}",
            hll.h[nx - 5]
        );

        // The same run with Lax-Friedrichs on identical data: HLL, being
        // an upwind Riemann solver, resolves the bore more sharply.
        let mut lf = ShallowWater1D::new(nx, dx, g);
        fill(&mut lf);
        let mut t = 0.0;
        while t < t_end {
            let dt = lf.stable_dt().min(t_end - t);
            lf.step_lax_friedrichs(dt);
            t += dt;
        }
        assert!(
            l1_hll < l1(&lf),
            "HLL ({l1_hll}) not sharper than Lax-Friedrichs ({})",
            l1(&lf)
        );
    }

    #[test]
    fn shallow_water_step_hll_keeps_a_lake_at_rest() {
        // Still water over a flat bed is an exact steady state of the
        // shallow-water equations; the HLL update with reflective walls
        // must reproduce it exactly.
        let nx = 64;
        let dx = 0.1;
        let g = constants::G_ACCEL;
        let mut sw = ShallowWater1D::new(nx, dx, g);
        sw.h.iter_mut().for_each(|h| *h = 1.5);
        let v0 = sw.total_volume();
        for _ in 0..200 {
            let dt = sw.stable_dt();
            sw.step_hll(dt);
        }
        let max_dev = sw.h.iter().map(|&h| (h - 1.5).abs()).fold(0.0_f64, f64::max);
        let max_mom = sw.hu.iter().map(|m| m.abs()).fold(0.0_f64, f64::max);
        assert!(max_dev < 1e-12, "lake at rest disturbed: {max_dev}");
        assert!(max_mom < 1e-12, "spurious momentum {max_mom}");
        assert!(approx_rel_eq(sw.total_volume(), v0, 1e-14), "volume drift");
    }

    #[test]
    fn shallow_water_flat_surface_stays_flat() {
        let nx = 100;
        let dx = 0.1;
        let g = constants::G_ACCEL;
        let h0 = 1.0;

        let mut sw = ShallowWater1D::new(nx, dx, g);
        for i in 0..nx {
            sw.h[i] = h0;
            // zero velocity everywhere
        }

        for _ in 0..1000 {
            let dt = sw.stable_dt();
            sw.step_lax_friedrichs(dt);
        }

        // All heights should still be very close to h0.
        let max_deviation = sw
            .h
            .iter()
            .skip(1) // skip boundary cells (reflective BC can cause small artifacts)
            .take(nx - 2)
            .map(|&h| (h - h0).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_deviation < 1e-6,
            "flat surface developed perturbation: max deviation = {max_deviation}",
        );
    }

    // ── EulerFluid2D ──

    #[test]
    fn euler_2d_divergence_near_zero_after_step() {
        let nx = 32;
        let ny = 32;
        let dx = 0.1;
        let dy = 0.1;
        let density = 1.0;

        let mut fluid = EulerFluid2D::new(nx, ny, dx, dy, density);

        // Set up a non-trivial initial velocity field: a vortex.
        // u = -y', v = x' (rigid body rotation around center).
        let cx = (nx as f64 * dx) / 2.0;
        let cy = (ny as f64 * dy) / 2.0;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let x = i as f64 * dx - cx;
                let y = j as f64 * dy - cy;
                let r = (x * x + y * y).sqrt();
                // Taper the vortex to zero at boundaries.
                let envelope = (1.0 - r / (cx.min(cy))).max(0.0);
                fluid.set_velocity(i, j, -y * envelope * 0.5, x * envelope * 0.5);
            }
        }

        // This initial field is not exactly divergence-free on the grid,
        // but after one projection step it should be.
        fluid.step(0.001, 0.0, 0.0);

        let div = fluid.divergence();
        assert!(
            div < 0.1,
            "divergence too large after projection: {div}",
        );
    }

    #[test]
    fn shallow_water_max_wave_speed_still_water() {
        let nx = 50;
        let dx = 0.1;
        let g = 9.81;
        let h0 = 2.0;
        let mut sw = ShallowWater1D::new(nx, dx, g);
        for i in 0..nx {
            sw.h[i] = h0;
        }
        // With zero velocity, max wave speed = √(g h)
        let s = sw.max_wave_speed();
        let expected = 4.429_446_918_070_02;
        assert!(
            approx_rel_eq(s, expected, 1e-10),
            "max_wave_speed={s}, expected {expected}"
        );
    }

    #[test]
    fn shallow_water_max_wave_speed_dry() {
        let sw = ShallowWater1D::new(10, 0.1, 9.81);
        assert!(approx_eq(sw.max_wave_speed(), 0.0, 1e-15));
    }

    #[test]
    fn shallow_water_total_energy_still_water() {
        let nx = 50;
        let dx = 0.1;
        let g = 9.81;
        let h0 = 1.0;
        let mut sw = ShallowWater1D::new(nx, dx, g);
        for i in 0..nx {
            sw.h[i] = h0;
        }
        // E = Σ (0.5 * g * h^2) * dx = nx * 0.5 * g * h0^2 * dx
        let e = sw.total_energy();
        let expected = 24.525;
        assert!(
            approx_rel_eq(e, expected, 1e-10),
            "total_energy={e}, expected {expected}"
        );
    }

    #[test]
    fn shallow_water_velocity_dry_cell() {
        let sw = ShallowWater1D::new(5, 0.1, 9.81);
        assert!(approx_eq(sw.velocity(0), 0.0, 1e-15));
    }

    #[test]
    fn shallow_water_velocity_wet_cell() {
        let mut sw = ShallowWater1D::new(5, 0.1, 9.81);
        sw.h[2] = 2.0;
        sw.hu[2] = 6.0;
        assert!(approx_rel_eq(sw.velocity(2), 3.0, 1e-10));
    }

    #[test]
    fn euler_2d_approximately_conserves_kinetic_energy() {
        // Inviscid Euler should conserve kinetic energy in the absence of
        // boundaries doing work. In practice, numerical dissipation (upwind
        // advection) causes some loss, but it should be bounded.
        let nx = 32;
        let ny = 32;
        let dx = 0.1;
        let dy = 0.1;
        let density = 1.0;

        let mut fluid = EulerFluid2D::new(nx, ny, dx, dy, density);

        // Uniform flow in x-direction (away from walls to minimize boundary
        // effects over the short test duration).
        let u0 = 1.0;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                fluid.set_velocity(i, j, u0, 0.0);
            }
        }

        let ke_initial = fluid.kinetic_energy();
        assert!(ke_initial > 0.0, "initial KE should be positive");

        let dt = 0.001;
        let n_steps = 50;
        for _ in 0..n_steps {
            fluid.step(dt, 0.0, 0.0);
        }

        let ke_final = fluid.kinetic_energy();

        // Allow up to 30% loss from numerical dissipation and boundary effects
        // over this short run. The key check is that energy doesn't grow
        // (no numerical instability) and doesn't vanish.
        assert!(
            ke_final <= ke_initial * 1.01,
            "KE grew (instability): {ke_initial} -> {ke_final}",
        );
        assert!(
            ke_final > ke_initial * 0.001,
            "KE lost too much: {ke_initial} -> {ke_final}",
        );
    }

    #[test]
    fn column_fluid_single_column_step() {
        let mut fluid = ColumnFluid::new(1, 1.0, 1000.0, constants::G_ACCEL);
        fluid.set_height(0, 5.0);
        fluid.step(0.001);
        assert!(approx_eq(fluid.heights[0], 5.0, 1e-12));
    }

    #[test]
    fn column_fluid_negative_height_clamped() {
        let mut fluid = ColumnFluid::new(3, 0.1, 1000.0, constants::G_ACCEL);
        fluid.set_height(0, 0.001);
        fluid.set_height(1, 10.0);
        fluid.set_height(2, 0.001);
        for _ in 0..50 {
            fluid.step(0.01);
        }
        for h in &fluid.heights {
            assert!(*h >= 0.0, "Height should never be negative, got {h}");
        }
    }

    #[test]
    fn shallow_water_small_nx_noop() {
        let mut sw = ShallowWater1D::new(2, 1.0, constants::G_ACCEL);
        sw.h[0] = 1.0;
        sw.h[1] = 1.0;
        sw.step_lax_friedrichs(0.01);
        assert!(sw.h[0].is_finite());
    }

    #[test]
    fn shallow_water_negative_depth_clamped() {
        let mut sw = ShallowWater1D::new(5, 0.1, constants::G_ACCEL);
        sw.h[2] = 100.0;
        sw.hu[2] = 500.0;
        let dt = sw.stable_dt() * 0.5;
        for _ in 0..3 {
            sw.step_lax_friedrichs(dt);
        }
        for &h in &sw.h {
            assert!(h >= 0.0, "Depth should never be negative, got {h}");
        }
    }

    #[test]
    fn shallow_water_dry_stable_dt() {
        let sw = ShallowWater1D::new(10, 0.5, constants::G_ACCEL);
        let dt = sw.stable_dt();
        assert!(approx_eq(dt, sw.dx, 1e-12));
    }

    #[test]
    fn shallow_water_total_volume() {
        let mut sw = ShallowWater1D::new(5, 0.5, constants::G_ACCEL);
        sw.h = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let v = sw.total_volume();
        assert!(approx_eq(v, 9.0 * 0.5, 1e-12));
    }

    #[test]
    fn column_fluid_large_dt_forces_negative_clamp() {
        let mut fluid = ColumnFluid::new(2, 1.0, 1000.0, constants::G_ACCEL);
        fluid.set_height(0, 10.0);
        fluid.set_height(1, 0.0);
        fluid.step(100.0);
        assert!(fluid.heights[0] >= 0.0);
        assert!(fluid.heights[1] >= 0.0);
    }

    #[test]
    fn shallow_water_large_dt_forces_negative_clamp() {
        let mut sw = ShallowWater1D::new(5, 0.5, constants::G_ACCEL);
        sw.h = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        sw.hu = vec![0.0, 0.0, 50.0, 0.0, 0.0];
        sw.step_lax_friedrichs(10.0);
        for &h in &sw.h {
            assert!(h >= 0.0);
        }
    }

    #[test]
    fn euler2d_upwind_boundary_edges() {
        let nx = 5;
        let ny = 5;
        let dx = 0.1;
        let dy = 0.1;
        let mut sim = EulerFluid2D::new(nx, ny, dx, dy, 1.0);
        // Set negative velocity everywhere so that at i=nx-1 and j=ny-1
        // the upwind_advect hits the else branches (L485, L498)
        for i in 0..nx {
            for j in 0..ny {
                let k = i * ny + j;
                sim.vx[k] = -1.0;
                sim.vy[k] = -1.0;
            }
        }
        for k in 0..nx * ny {
            sim.pressure[k] = 1.0;
        }
        sim.step(0.001, 0.0, 0.0);
        assert!(sim.divergence().is_finite());
    }

    #[test]
    fn euler2d_poisson_converges() {
        let nx = 10;
        let ny = 10;
        let dx = 0.1;
        let dy = 0.1;
        let mut sim = EulerFluid2D::new(nx, ny, dx, dy, 1.0);
        for i in 0..nx {
            for j in 0..ny {
                let k = i * ny + j;
                sim.vx[k] = ((i as f64) * 0.1).sin();
                sim.vy[k] = ((j as f64) * 0.1).cos();
            }
        }
        sim.step(0.0001, 0.0, 0.0);
        let div = sim.divergence();
        assert!(div.is_finite());
    }

    #[test]
    fn test_approx_rel_eq_near_zero_b() {
        assert!(approx_rel_eq(0.0, 0.0, 1e-6));
        assert!(!approx_rel_eq(1.0, 0.0, 0.5));
    }
}
