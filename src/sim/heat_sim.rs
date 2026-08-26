//! Heat conduction and convection-diffusion on a grid.
//!
//! Explicit finite differences in two and three dimensions, with Dirichlet
//! and Neumann boundaries, sources, and an advection term for
//! convection-diffusion.
//!
//! Explicit stepping is only conditionally stable: the step must satisfy
//! `α Δt / Δx² ≤ 1/4` in 2-D and `1/6` in 3-D, so halving the grid spacing
//! quarters the allowable time step. The stability limit is provided as a
//! function rather than left to the caller to remember.

// Heat conduction and convection simulation on structured grids.
//
// ## Governing PDEs
//
// **Heat equation (conduction):**
//   ∂T/∂t = α∇²T + Q/(ρcₚ)
// where α = k/(ρcₚ) is thermal diffusivity [m²/s].
//
// **Advection-diffusion (convection-diffusion):**
//   ∂T/∂t + v·∂T/∂x = α·∂²T/∂x²
//
// ## Discretization
//
// ### Explicit (FTCS — Forward Time, Central Space):
//   T_ij^(n+1) = T_ij^n + α·dt·( (T_{i+1,j} - 2T_ij + T_{i-1,j})/dx²
//                                + (T_{i,j+1} - 2T_ij + T_{i,j-1})/dy² )
//
// ### Implicit (Backward Euler + Jacobi iteration):
//   (1 + 2α·dt/dx² + 2α·dt/dy²)·T_ij^(n+1)
//     - α·dt/dx²·(T_{i-1,j}^(n+1) + T_{i+1,j}^(n+1))
//     - α·dt/dy²·(T_{i,j-1}^(n+1) + T_{i,j+1}^(n+1)) = T_ij^n
//
// ### Upwind (1D advection-diffusion):
//   v > 0: ∂T/∂x ≈ (T_i - T_{i-1})/dx
//   v < 0: ∂T/∂x ≈ (T_{i+1} - T_i)/dx
//   Diffusion: α·(T_{i+1} - 2T_i + T_{i-1})/dx²
//
// ## Stability conditions
//
// Explicit 2D: dt < 1 / (2α·(1/dx² + 1/dy²))
// Explicit 3D: dt < 1 / (2α·(1/dx² + 1/dy² + 1/dz²))
// Upwind 1D:   dt < min(dx/|v|, dx²/(2α))
// Implicit:    Unconditionally stable (for any dt).

/// 2D heat conduction on a uniform Cartesian grid with Dirichlet boundaries.
pub struct HeatConduction2D {
    /// Temperature field stored in row-major order: index = i*ny + j.
    pub temperature: Vec<f64>,
    /// Number of grid points in x.
    pub nx: usize,
    /// Number of grid points in y.
    pub ny: usize,
    /// Grid spacing in x (m).
    pub dx: f64,
    /// Grid spacing in y (m).
    pub dy: f64,
    /// Thermal diffusivity α = k/(ρcₚ) [m²/s].
    pub diffusivity: f64,
}

impl HeatConduction2D {
    /// Create a grid initialized to uniform temperature 0.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, diffusivity: f64) -> Self {
        assert!(dx > 0.0, "grid spacing dx must be positive");
        assert!(dy > 0.0, "grid spacing dy must be positive");
        assert!(diffusivity > 0.0, "thermal diffusivity must be positive");
        Self {
            temperature: vec![0.0; nx * ny],
            nx,
            ny,
            dx,
            dy,
            diffusivity,
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ny + j
    }

    /// Set temperature at grid point (i, j).
    pub fn set_temperature(&mut self, i: usize, j: usize, temp: f64) {
        let idx = self.idx(i, j);
        self.temperature[idx] = temp;
    }

    #[must_use]
    /// Get temperature at grid point (i, j).
    pub fn get_temperature(&self, i: usize, j: usize) -> f64 {
        self.temperature[self.idx(i, j)]
    }

    /// FTCS explicit step. Boundary cells (i=0, i=nx-1, j=0, j=ny-1) are
    /// held fixed (Dirichlet).
    ///
    /// Stability requires dt < stable_dt().
    pub fn step_explicit(&mut self, dt: f64) {
        let old = self.temperature.clone();
        let rx = self.diffusivity * dt / (self.dx * self.dx);
        let ry = self.diffusivity * dt / (self.dy * self.dy);

        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let c = i * self.ny + j;
                let laplacian_x = old[c + self.ny] - 2.0 * old[c] + old[c - self.ny];
                let laplacian_y = old[c + 1] - 2.0 * old[c] + old[c - 1];
                self.temperature[c] = old[c] + rx * laplacian_x + ry * laplacian_y;
            }
        }
    }

    /// Backward Euler solved via Jacobi iteration (unconditionally stable).
    ///
    /// Solves:
    ///   (1 + 2·rx + 2·ry)·T_ij^(n+1)
    ///     − rx·(T_{i±1,j}^(n+1)) − ry·(T_{i,j±1}^(n+1)) = T_ij^n
    /// where rx = α·dt/dx², ry = α·dt/dy².
    pub fn step_implicit_jacobi(&mut self, dt: f64, iterations: usize) {
        let rhs = self.temperature.clone();
        let rx = self.diffusivity * dt / (self.dx * self.dx);
        let ry = self.diffusivity * dt / (self.dy * self.dy);
        let diag = 1.0 + 2.0 * rx + 2.0 * ry;

        for _ in 0..iterations {
            let prev = self.temperature.clone();
            for i in 1..self.nx - 1 {
                for j in 1..self.ny - 1 {
                    let c = i * self.ny + j;
                    let neighbors = rx * (prev[c + self.ny] + prev[c - self.ny])
                        + ry * (prev[c + 1] + prev[c - 1]);
                    self.temperature[c] = (rhs[c] + neighbors) / diag;
                }
            }
        }
    }

    /// Maximum stable time step for the explicit FTCS scheme.
    /// dt_max = 1 / (2α·(1/dx² + 1/dy²))
    #[must_use]
    pub fn stable_dt(&self) -> f64 {
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let inv_dy2 = 1.0 / (self.dy * self.dy);
        1.0 / (2.0 * self.diffusivity * (inv_dx2 + inv_dy2))
    }

    /// Total thermal energy proxy: Σ T_ij · dx · dy.
    /// Proportional to total thermal energy when ρcₚ is uniform.
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        let cell_area = self.dx * self.dy;
        self.temperature.iter().sum::<f64>() * cell_area
    }

    #[must_use]
    /// Maximum temperature in the field.
    pub fn max_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    #[must_use]
    /// Minimum temperature in the field.
    pub fn min_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    #[must_use]
    /// Average temperature across the entire grid.
    pub fn average_temperature(&self) -> f64 {
        self.temperature.iter().sum::<f64>() / self.temperature.len() as f64
    }

    /// Explicit step with volumetric heat source.
    ///
    /// PDE: ∂T/∂t = α∇²T + Q(x,y)/(ρcₚ)
    ///
    /// `sources` is the Q/(ρcₚ) term at each grid point, same layout as
    /// `temperature` (row-major, length nx*ny). Units: [K/s].
    pub fn step_with_source(&mut self, dt: f64, sources: &[f64]) {
        assert_eq!(
            sources.len(),
            self.nx * self.ny,
            "sources length must equal nx * ny"
        );
        let old = self.temperature.clone();
        let rx = self.diffusivity * dt / (self.dx * self.dx);
        let ry = self.diffusivity * dt / (self.dy * self.dy);

        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let c = i * self.ny + j;
                let laplacian_x = old[c + self.ny] - 2.0 * old[c] + old[c - self.ny];
                let laplacian_y = old[c + 1] - 2.0 * old[c] + old[c - 1];
                self.temperature[c] =
                    old[c] + rx * laplacian_x + ry * laplacian_y + dt * sources[c];
            }
        }
    }
}

/// 3D heat conduction on a uniform Cartesian grid with Dirichlet boundaries.
pub struct HeatConduction3D {
    /// Temperature field: index = i*ny*nz + j*nz + k.
    pub temperature: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Thermal diffusivity α = k/(ρcₚ) [m²/s].
    pub diffusivity: f64,
}

impl HeatConduction3D {
    #[must_use]
    /// Create a 3D heat conduction grid initialized to uniform temperature 0.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        diffusivity: f64,
    ) -> Self {
        assert!(dx > 0.0, "grid spacing dx must be positive");
        assert!(dy > 0.0, "grid spacing dy must be positive");
        assert!(dz > 0.0, "grid spacing dz must be positive");
        assert!(diffusivity > 0.0, "thermal diffusivity must be positive");
        Self {
            temperature: vec![0.0; nx * ny * nz],
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            diffusivity,
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    /// Set temperature at grid point (i, j, k).
    pub fn set_temperature(&mut self, i: usize, j: usize, k: usize, temp: f64) {
        let idx = self.idx(i, j, k);
        self.temperature[idx] = temp;
    }

    #[must_use]
    /// Get temperature at grid point (i, j, k).
    pub fn get_temperature(&self, i: usize, j: usize, k: usize) -> f64 {
        self.temperature[self.idx(i, j, k)]
    }

    /// FTCS explicit step in 3D. Boundary cells held fixed (Dirichlet).
    ///
    /// Stability requires dt < stable_dt().
    pub fn step_explicit(&mut self, dt: f64) {
        let old = self.temperature.clone();
        let rx = self.diffusivity * dt / (self.dx * self.dx);
        let ry = self.diffusivity * dt / (self.dy * self.dy);
        let rz = self.diffusivity * dt / (self.dz * self.dz);
        let stride_i = self.ny * self.nz;
        let stride_j = self.nz;

        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                for k in 1..self.nz - 1 {
                    let c = i * stride_i + j * stride_j + k;
                    let lap_x = old[c + stride_i] - 2.0 * old[c] + old[c - stride_i];
                    let lap_y = old[c + stride_j] - 2.0 * old[c] + old[c - stride_j];
                    let lap_z = old[c + 1] - 2.0 * old[c] + old[c - 1];
                    self.temperature[c] =
                        old[c] + rx * lap_x + ry * lap_y + rz * lap_z;
                }
            }
        }
    }

    /// dt_max = 1 / (2α·(1/dx² + 1/dy² + 1/dz²))
    #[must_use]
    pub fn stable_dt(&self) -> f64 {
        let inv = 1.0 / (self.dx * self.dx)
            + 1.0 / (self.dy * self.dy)
            + 1.0 / (self.dz * self.dz);
        1.0 / (2.0 * self.diffusivity * inv)
    }

    /// Total thermal energy proxy: Σ T·dx·dy·dz.
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        let cell_vol = self.dx * self.dy * self.dz;
        self.temperature.iter().sum::<f64>() * cell_vol
    }

    /// Average temperature across the entire 3D grid.
    #[must_use]
    pub fn average_temperature(&self) -> f64 {
        self.temperature.iter().sum::<f64>() / self.temperature.len() as f64
    }
}

/// 1D convection-diffusion (advection-diffusion) solver.
///
/// PDE: ∂T/∂t + v·∂T/∂x = α·∂²T/∂x²
///
/// Uses first-order upwind for the advection term and central differencing
/// for the diffusion term.
pub struct ConvectionDiffusion1D {
    /// Scalar field values.
    pub field: Vec<f64>,
    /// Number of grid points.
    pub nx: usize,
    /// Grid spacing (m).
    pub dx: f64,
    /// Advection velocity [m/s].
    pub velocity: f64,
    /// Thermal diffusivity [m²/s].
    pub diffusivity: f64,
}

impl ConvectionDiffusion1D {
    #[must_use]
    /// Create a 1D convection-diffusion solver with the given parameters.
    pub fn new(nx: usize, dx: f64, velocity: f64, diffusivity: f64) -> Self {
        assert!(dx > 0.0, "grid spacing dx must be positive");
        assert!(diffusivity > 0.0, "thermal diffusivity must be positive");
        Self {
            field: vec![0.0; nx],
            nx,
            dx,
            velocity,
            diffusivity,
        }
    }

    /// Upwind advection + central diffusion explicit step.
    /// Boundaries held fixed (Dirichlet).
    ///
    /// Stability requires dt < stable_dt().
    pub fn step_upwind(&mut self, dt: f64) {
        let old = self.field.clone();
        let r_diff = self.diffusivity * dt / (self.dx * self.dx);

        for i in 1..self.nx - 1 {
            let diffusion = r_diff * (old[i + 1] - 2.0 * old[i] + old[i - 1]);

            let advection = if self.velocity > 0.0 {
                self.velocity * (old[i] - old[i - 1]) / self.dx
            } else {
                self.velocity * (old[i + 1] - old[i]) / self.dx
            };

            self.field[i] = old[i] - dt * advection + diffusion;
        }
    }

    /// Grid Peclet number: Pe = v·dx/α.
    /// For stability of central differencing the advection term, Pe < 2 is
    /// required. The upwind scheme used here is stable for any Pe but
    /// introduces numerical diffusion proportional to Pe.
    #[must_use]
    pub fn peclet_number(&self) -> f64 {
        self.velocity * self.dx / self.diffusivity
    }

    /// Maximum stable time step.
    /// CFL condition: dt < dx/|v| AND diffusive limit: dt < dx²/(2α).
    /// Returns the minimum of both limits.
    #[must_use]
    pub fn stable_dt(&self) -> f64 {
        let dt_cfl = if self.velocity.abs() > 0.0 {
            self.dx / self.velocity.abs()
        } else {
            f64::INFINITY
        };
        let dt_diff = self.dx * self.dx / (2.0 * self.diffusivity);
        dt_cfl.min(dt_diff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }



    // ── 2D: hot center diffuses outward, max temp decreases ──

    #[test]
    fn test_2d_hot_center_diffuses() {
        let nx = 21;
        let ny = 21;
        let dx = 0.01;
        let dy = 0.01;
        let alpha = 1e-4;
        let mut grid = HeatConduction2D::new(nx, ny, dx, dy, alpha);

        let mid_i = nx / 2;
        let mid_j = ny / 2;
        grid.set_temperature(mid_i, mid_j, 100.0);

        let initial_max = grid.max_temperature();
        assert!(approx(initial_max, 100.0, TOLERANCE));

        let dt = grid.stable_dt() * 0.4;
        for _ in 0..50 {
            grid.step_explicit(dt);
        }

        // Max temperature must decrease as heat spreads
        assert!(grid.max_temperature() < initial_max);
        // Neighbors of center should have gained heat
        assert!(grid.get_temperature(mid_i + 1, mid_j) > 0.0);
        assert!(grid.get_temperature(mid_i, mid_j + 1) > 0.0);
    }

    // ── 2D: total energy decreases with Dirichlet=0 boundaries ──

    #[test]
    fn test_2d_energy_decreases_dirichlet_zero() {
        let nx = 11;
        let ny = 11;
        let dx = 0.01;
        let dy = 0.01;
        let alpha = 1e-4;
        let mut grid = HeatConduction2D::new(nx, ny, dx, dy, alpha);

        // Set interior to 50, boundaries stay at 0 (Dirichlet)
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                grid.set_temperature(i, j, 50.0);
            }
        }

        let energy_before = grid.total_energy();
        let dt = grid.stable_dt() * 0.4;
        for _ in 0..100 {
            grid.step_explicit(dt);
        }
        let energy_after = grid.total_energy();

        // Heat flows out through zero-temperature boundaries
        assert!(energy_after < energy_before);
    }

    // ── 2D: explicit vs implicit give similar results for stable dt ──

    #[test]
    fn test_2d_explicit_vs_implicit_agreement() {
        let nx = 15;
        let ny = 15;
        let dx = 0.01;
        let dy = 0.01;
        let alpha = 1e-4;

        let mut grid_explicit = HeatConduction2D::new(nx, ny, dx, dy, alpha);
        let mut grid_implicit = HeatConduction2D::new(nx, ny, dx, dy, alpha);

        // Same initial condition: hot spot in center
        let mid_i = nx / 2;
        let mid_j = ny / 2;
        grid_explicit.set_temperature(mid_i, mid_j, 100.0);
        grid_implicit.set_temperature(mid_i, mid_j, 100.0);

        // Use a small dt well within stability for both to converge
        let dt = grid_explicit.stable_dt() * 0.3;
        let steps = 30;
        let jacobi_iters = 200;

        for _ in 0..steps {
            grid_explicit.step_explicit(dt);
            grid_implicit.step_implicit_jacobi(dt, jacobi_iters);
        }

        // They should agree to within a few percent
        for i in 0..nx {
            for j in 0..ny {
                let te = grid_explicit.get_temperature(i, j);
                let ti = grid_implicit.get_temperature(i, j);
                assert!(
                    approx(te, ti, 0.5),
                    "Mismatch at ({i},{j}): explicit={te}, implicit={ti}"
                );
            }
        }
    }

    // ── 2D: stability check returns correct value ──

    #[test]
    fn test_2d_stable_dt_value() {
        let dx = 0.01;
        let dy = 0.02;
        let alpha = 1e-4;
        let grid = HeatConduction2D::new(5, 5, dx, dy, alpha);

        let expected = 0.4;

        assert!(approx(grid.stable_dt(), expected, TOLERANCE));
    }

    // ── 3D: symmetric initial condition stays symmetric ──

    #[test]
    fn test_3d_symmetry_preserved() {
        let n = 11;
        let d = 0.01;
        let alpha = 1e-4;
        let mut grid = HeatConduction3D::new(n, n, n, d, d, d, alpha);

        let mid = n / 2;
        grid.set_temperature(mid, mid, mid, 100.0);

        let dt = grid.stable_dt() * 0.3;
        for _ in 0..20 {
            grid.step_explicit(dt);
        }

        // All six face-adjacent neighbors of center should be equal
        let t_px = grid.get_temperature(mid + 1, mid, mid);
        let t_mx = grid.get_temperature(mid - 1, mid, mid);
        let t_py = grid.get_temperature(mid, mid + 1, mid);
        let t_my = grid.get_temperature(mid, mid - 1, mid);
        let t_pz = grid.get_temperature(mid, mid, mid + 1);
        let t_mz = grid.get_temperature(mid, mid, mid - 1);

        assert!(approx(t_px, t_mx, 1e-12), "x-symmetry broken");
        assert!(approx(t_py, t_my, 1e-12), "y-symmetry broken");
        assert!(approx(t_pz, t_mz, 1e-12), "z-symmetry broken");
        // Cubic symmetry: all six neighbors equal
        assert!(approx(t_px, t_py, 1e-12), "cubic symmetry x vs y");
        assert!(approx(t_py, t_pz, 1e-12), "cubic symmetry y vs z");

        // Center should have decreased
        assert!(grid.get_temperature(mid, mid, mid) < 100.0);
    }

    // ── Convection-diffusion: pulse advects in correct direction ──

    #[test]
    fn test_convection_pulse_advects_positive() {
        let nx = 101;
        let dx = 0.01;
        let velocity = 1.0;
        let alpha = 1e-4;
        let mut solver = ConvectionDiffusion1D::new(nx, dx, velocity, alpha);

        // Place a pulse at i=20
        let pulse_idx = 20;
        solver.field[pulse_idx] = 1.0;

        // Compute centroid of field
        let centroid_before: f64 = solver
            .field
            .iter()
            .enumerate()
            .map(|(i, &t)| i as f64 * t)
            .sum::<f64>()
            / solver.field.iter().sum::<f64>();

        let dt = solver.stable_dt() * 0.4;
        for _ in 0..10 {
            solver.step_upwind(dt);
        }

        let total: f64 = solver.field.iter().sum();
        assert!(total > 1e-15, "total should remain above threshold");
        let centroid_after: f64 = solver
            .field
            .iter()
            .enumerate()
            .map(|(i, &t)| i as f64 * t)
            .sum::<f64>()
            / total;

        assert!(
            centroid_after > centroid_before,
            "Pulse should advect rightward: before={centroid_before}, after={centroid_after}"
        );
    }

    #[test]
    fn test_convection_pulse_advects_negative() {
        let nx = 101;
        let dx = 0.01;
        let velocity = -1.0;
        let alpha = 1e-4;
        let mut solver = ConvectionDiffusion1D::new(nx, dx, velocity, alpha);

        let pulse_idx = 80;
        solver.field[pulse_idx] = 1.0;

        let centroid_before: f64 = solver
            .field
            .iter()
            .enumerate()
            .map(|(i, &t)| i as f64 * t)
            .sum::<f64>()
            / solver.field.iter().sum::<f64>();

        let dt = solver.stable_dt() * 0.4;
        for _ in 0..10 {
            solver.step_upwind(dt);
        }

        let total: f64 = solver.field.iter().sum();
        assert!(total > 1e-15, "total should remain above threshold");
        let centroid_after: f64 = solver
            .field
            .iter()
            .enumerate()
            .map(|(i, &t)| i as f64 * t)
            .sum::<f64>()
            / total;

        assert!(
            centroid_after < centroid_before,
            "Pulse should advect leftward: before={centroid_before}, after={centroid_after}"
        );
    }

    // ── Convection-diffusion: pure diffusion (v=0) matches heat equation ──

    #[test]
    fn test_convection_pure_diffusion_matches_heat_equation() {
        let nx = 51;
        let dx = 0.01;
        let alpha = 1e-4;

        // ConvectionDiffusion1D with v=0
        let mut cd = ConvectionDiffusion1D::new(nx, dx, 0.0, alpha);
        // HeatConduction2D used as 1D reference (ny=1 doesn't work with
        // Dirichlet on edges, so we use the 1D heat_equation_step from
        // thermodynamics). Instead, build a direct 1D reference.
        let mut ref_field = vec![0.0; nx];

        // Same IC: pulse at center
        let mid = nx / 2;
        cd.field[mid] = 100.0;
        ref_field[mid] = 100.0;

        let dt = cd.stable_dt() * 0.4;
        let steps = 40;

        for _ in 0..steps {
            cd.step_upwind(dt);
            // Manual 1D heat equation step (FTCS)
            let old = ref_field.clone();
            let r = alpha * dt / (dx * dx);
            for i in 1..nx - 1 {
                ref_field[i] = old[i] + r * (old[i + 1] - 2.0 * old[i] + old[i - 1]);
            }
        }

        for i in 0..nx {
            let (cd_val, ref_val) = (cd.field[i], ref_field[i]);
            assert!(
                approx(cd_val, ref_val, 1e-10),
                "Mismatch at i={i}: cd={cd_val}, ref={ref_val}",
            );
        }
    }

    // ── Peclet number ──

    #[test]
    fn test_peclet_number() {
        let solver = ConvectionDiffusion1D::new(10, 0.1, 2.0, 0.05);
        // Pe = v·dx/α = 2.0 * 0.1 / 0.05 = 4.0
        assert!(approx(solver.peclet_number(), 4.0, TOLERANCE));
    }

    // ── Stable dt for convection-diffusion ──

    #[test]
    fn test_convection_stable_dt() {
        let dx = 0.01;
        let v = 2.0;
        let alpha = 1e-4;
        let solver = ConvectionDiffusion1D::new(10, dx, v, alpha);

        let expected = 0.005;

        assert!(approx(solver.stable_dt(), expected, TOLERANCE));
    }

    // ── 2D: step_with_source adds energy ──

    #[test]
    fn test_2d_source_adds_energy() {
        let nx = 11;
        let ny = 11;
        let dx = 0.01;
        let dy = 0.01;
        let alpha = 1e-4;
        let mut grid = HeatConduction2D::new(nx, ny, dx, dy, alpha);

        // Uniform source in interior
        let mut sources = vec![0.0; nx * ny];
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                sources[i * ny + j] = 10.0; // 10 K/s
            }
        }

        let dt = grid.stable_dt() * 0.4;
        for _ in 0..10 {
            grid.step_with_source(dt, &sources);
        }

        // Interior should have heated up
        assert!(grid.get_temperature(5, 5) > 0.0);
        assert!(grid.average_temperature() > 0.0);
    }

    // ── 3D: stable_dt value ──

    #[test]
    fn test_3d_stable_dt() {
        let dx = 0.01;
        let dy = 0.02;
        let dz = 0.015;
        let alpha = 1e-4;
        let grid = HeatConduction3D::new(5, 5, 5, dx, dy, dz, alpha);

        let expected = 0.29508196721311475;
        assert!(approx(grid.stable_dt(), expected, TOLERANCE));
    }

    // ── 2D: min/max/average ──

    #[test]
    fn test_2d_statistics() {
        let mut grid = HeatConduction2D::new(3, 3, 0.01, 0.01, 1e-4);
        grid.set_temperature(0, 0, 10.0);
        grid.set_temperature(1, 1, 50.0);
        grid.set_temperature(2, 2, -5.0);

        assert!(approx(grid.max_temperature(), 50.0, TOLERANCE));
        assert!(approx(grid.min_temperature(), -5.0, TOLERANCE));
        // avg = (10 + 50 + (-5)) / 9 = 55/9
        assert!(approx(grid.average_temperature(), 55.0 / 9.0, 1e-12));
    }

    #[test]
    fn test_3d_total_energy() {
        let mut grid = HeatConduction3D::new(3, 3, 3, 0.01, 0.01, 0.01, 1e-4);
        grid.set_temperature(1, 1, 1, 100.0);
        let e = grid.total_energy();
        assert!(e > 0.0, "Total energy should be positive");
    }

    #[test]
    fn test_3d_average_temperature() {
        let mut grid = HeatConduction3D::new(3, 3, 3, 0.01, 0.01, 0.01, 1e-4);
        grid.set_temperature(1, 1, 1, 27.0);
        let avg = grid.average_temperature();
        assert!(approx(avg, 27.0 / 27.0, 1e-12));
    }
}
