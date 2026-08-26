// Wave equation simulation on structured grids using finite differences.
//
// PDE (1D): ∂²u/∂t² = c² ∂²u/∂x²
// PDE (2D): ∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)
//
// Discretization: second-order central differences in both space and time
// (leapfrog / Verlet scheme).
//
// 1D update rule:
//   u_i^{n+1} = 2 u_i^n - u_i^{n-1} + r² (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)
//   where r = c dt / dx  (Courant number)
//
// 2D update rule (no damping):
//   u_{ij}^{n+1} = 2 u_{ij}^n - u_{ij}^{n-1}
//       + c² dt² [ (u_{i+1,j} - 2u_{ij} + u_{i-1,j}) / dx²
//                + (u_{i,j+1} - 2u_{ij} + u_{i,j-1}) / dy² ]
//
// 2D update rule (with damping term -γ ∂u/∂t):
//   u_{ij}^{n+1} = [ 2 u_{ij}^n - (1 - γ dt) u_{ij}^{n-1}
//                   + c² dt² ∇²u_{ij}^n ] / (1 + γ dt)
//
// Stability (CFL condition):
//   1D: r = c dt / dx ≤ 1    →  dt_max = dx / c
//   2D: c dt √(1/dx² + 1/dy²) ≤ 1  →  dt_max = 1 / (c √(1/dx² + 1/dy²))

/// 1D wave equation solver on a uniform grid with fixed (Dirichlet) endpoints.
pub struct WaveEquation1D {
    /// Current displacement field u^n.
    pub u_current: Vec<f64>,
    /// Previous time-step displacement field u^{n-1}.
    pub u_previous: Vec<f64>,
    /// Number of grid points (including boundary points).
    pub nx: usize,
    /// Grid spacing (m).
    pub dx: f64,
    /// Phase speed c (m/s).
    pub wave_speed: f64,
    /// Accumulated simulation time (s).
    pub time: f64,
}

impl WaveEquation1D {
    /// Create a new 1D wave equation solver. All displacements start at zero.
    pub fn new(nx: usize, dx: f64, wave_speed: f64) -> Self {
        assert!(dx > 0.0, "grid spacing dx must be positive");
        assert!(wave_speed > 0.0, "wave speed must be positive");
        Self {
            u_current: vec![0.0; nx],
            u_previous: vec![0.0; nx],
            nx,
            dx,
            wave_speed,
            time: 0.0,
        }
    }

    /// Set initial displacement u(x, 0) and initial velocity ∂u/∂t(x, 0).
    ///
    /// The previous time-step field is computed from the Taylor expansion that
    /// is second-order accurate in dt:
    ///   u_i^{-1} = u_i^0 - v_i dt + 0.5 c² dt² / dx² (u_{i+1}^0 - 2 u_i^0 + u_{i-1}^0)
    ///
    /// This avoids the first-order error that arises from the naive
    /// u_prev = u_current - velocity * dt approximation.
    pub fn set_initial(&mut self, displacement: &[f64], velocity: &[f64], dt: f64) {
        assert_eq!(displacement.len(), self.nx);
        assert_eq!(velocity.len(), self.nx);

        self.u_current.copy_from_slice(displacement);

        let r2 = (self.wave_speed * dt / self.dx).powi(2);

        // Boundary points stay at zero (fixed endpoints).
        self.u_previous[0] = 0.0;
        self.u_previous[self.nx - 1] = 0.0;

        for i in 1..self.nx - 1 {
            let laplacian = displacement[i + 1] - 2.0 * displacement[i] + displacement[i - 1];
            self.u_previous[i] =
                displacement[i] - velocity[i] * dt + 0.5 * r2 * laplacian;
        }

        self.time = 0.0;
    }

    /// Advance one time step using the leapfrog scheme with fixed endpoints
    /// `u[0] = u[nx-1] = 0`.
    pub fn step(&mut self, dt: f64) {
        let r2 = (self.wave_speed * dt / self.dx).powi(2);

        let mut u_next = vec![0.0; self.nx];
        // Boundaries stay zero (Dirichlet).
        for i in 1..self.nx - 1 {
            let laplacian =
                self.u_current[i + 1] - 2.0 * self.u_current[i] + self.u_current[i - 1];
            u_next[i] = 2.0 * self.u_current[i] - self.u_previous[i] + r2 * laplacian;
        }

        self.u_previous.copy_from_slice(&self.u_current);
        self.u_current.copy_from_slice(&u_next);
        self.time += dt;
    }

    /// Advance one time step with Mur's first-order absorbing boundary conditions.
    ///
    /// Interior update is identical to `step`. At the boundaries the outgoing
    /// characteristic is approximated:
    ///
    /// ```text
    /// u[0]^{n+1}     = u[1]^n      + (r - 1)/(r + 1) (u[1]^{n+1}     - u[0]^n)
    /// u[nx-1]^{n+1}  = u[nx-2]^n   + (r - 1)/(r + 1) (u[nx-2]^{n+1}  - u[nx-1]^n)
    /// ```
    ///
    /// where `r = c dt / dx`.
    ///
    /// These conditions absorb normally-incident waves perfectly (first order
    /// in angle of incidence for oblique waves).
    pub fn step_absorbing(&mut self, dt: f64) {
        let r = self.wave_speed * dt / self.dx;
        let r2 = r * r;
        let abc_coeff = (r - 1.0) / (r + 1.0);

        let mut u_next = vec![0.0; self.nx];

        // Interior leapfrog update.
        for i in 1..self.nx - 1 {
            let laplacian =
                self.u_current[i + 1] - 2.0 * self.u_current[i] + self.u_current[i - 1];
            u_next[i] = 2.0 * self.u_current[i] - self.u_previous[i] + r2 * laplacian;
        }

        // Mur ABC — left boundary.
        u_next[0] = self.u_current[1] + abc_coeff * (u_next[1] - self.u_current[0]);

        // Mur ABC — right boundary.
        let n = self.nx - 1;
        u_next[n] = self.u_current[n - 1] + abc_coeff * (u_next[n - 1] - self.u_current[n]);

        self.u_previous.copy_from_slice(&self.u_current);
        self.u_current.copy_from_slice(&u_next);
        self.time += dt;
    }

    /// Courant number r = c dt / dx.
    pub fn courant_number(&self, dt: f64) -> f64 {
        self.wave_speed * dt / self.dx
    }

    /// Maximum stable time step from the CFL condition: dt_max = dx / c.
    pub fn stable_dt(&self) -> f64 {
        self.dx / self.wave_speed
    }

    /// Discrete total energy (kinetic + potential) of the grid.
    ///
    /// E = 0.5 dx Σ_i [ ((u_i^n - u_i^{n-1}) / dt)²  +  c² ((u_{i+1}^n - u_i^n) / dx)² ]
    ///
    /// The first term is the kinetic energy density (velocity squared), the
    /// second is the potential / strain energy density (spatial gradient squared).
    /// Both are summed with the cell volume dx.
    pub fn total_energy(&self, dt: f64) -> f64 {
        let mut energy = 0.0;
        let c2 = self.wave_speed * self.wave_speed;
        let inv_dt2 = 1.0 / (dt * dt);
        let inv_dx2 = 1.0 / (self.dx * self.dx);

        for i in 0..self.nx {
            let kinetic = (self.u_current[i] - self.u_previous[i]).powi(2) * inv_dt2;
            let potential = if i < self.nx - 1 {
                c2 * (self.u_current[i + 1] - self.u_current[i]).powi(2) * inv_dx2
            } else {
                0.0
            };
            energy += kinetic + potential;
        }

        0.5 * self.dx * energy
    }

    /// Inject a continuous point source by adding amplitude to the current
    /// displacement field at the given grid index.
    pub fn add_source(&mut self, position: usize, amplitude: f64) {
        assert!(position < self.nx);
        self.u_current[position] += amplitude;
    }
}

/// 2D wave equation solver on a uniform grid with fixed (Dirichlet) boundaries.
///
/// Grid layout: row-major, index = j * nx + i  (i along x, j along y).
pub struct WaveEquation2D {
    /// Current displacement field u^n, length nx * ny, row-major.
    pub u_current: Vec<f64>,
    /// Previous time-step displacement field u^{n-1}.
    pub u_previous: Vec<f64>,
    /// Grid points in x.
    pub nx: usize,
    /// Grid points in y.
    pub ny: usize,
    /// Grid spacing in x (m).
    pub dx: f64,
    /// Grid spacing in y (m).
    pub dy: f64,
    /// Phase speed c (m/s).
    pub wave_speed: f64,
    /// Accumulated simulation time (s).
    pub time: f64,
    /// Damping coefficient γ (s⁻¹). Zero means no damping.
    /// Adds a -γ ∂u/∂t term to the PDE.
    pub damping: f64,
}

impl WaveEquation2D {
    /// Create a new 2D wave equation solver with all displacements at zero.
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, wave_speed: f64) -> Self {
        assert!(dx > 0.0, "grid spacing dx must be positive");
        assert!(dy > 0.0, "grid spacing dy must be positive");
        assert!(wave_speed > 0.0, "wave speed must be positive");
        let n = nx * ny;
        Self {
            u_current: vec![0.0; n],
            u_previous: vec![0.0; n],
            nx,
            ny,
            dx,
            dy,
            wave_speed,
            time: 0.0,
            damping: 0.0,
        }
    }

    /// Enable viscous damping (-γ ∂u/∂t).
    pub fn set_damping(&mut self, damping: f64) {
        self.damping = damping;
    }

    /// Linear index for grid point (i, j).
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Advance one time step.
    ///
    /// Without damping (γ = 0):
    ///   u_{ij}^{n+1} = 2 u_{ij}^n - u_{ij}^{n-1} + c² dt² ∇²u_{ij}^n
    ///
    /// With damping (γ > 0):
    ///   u_{ij}^{n+1} = [ 2 u_{ij}^n - (1 - γ dt) u_{ij}^{n-1}
    ///                   + c² dt² ∇²u_{ij}^n ] / (1 + γ dt)
    ///
    /// Boundary points (i=0, i=nx-1, j=0, j=ny-1) are held at zero.
    pub fn step(&mut self, dt: f64) {
        let c2dt2 = self.wave_speed * self.wave_speed * dt * dt;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let inv_dy2 = 1.0 / (self.dy * self.dy);
        let gamma_dt = self.damping * dt;
        let denom = 1.0 + gamma_dt;
        let prev_coeff = 1.0 - gamma_dt;

        let n = self.nx * self.ny;
        let mut u_next = vec![0.0; n];

        for j in 1..self.ny - 1 {
            for i in 1..self.nx - 1 {
                let c = self.idx(i, j);
                let laplacian_x = (self.u_current[self.idx(i + 1, j)]
                    - 2.0 * self.u_current[c]
                    + self.u_current[self.idx(i - 1, j)])
                    * inv_dx2;
                let laplacian_y = (self.u_current[self.idx(i, j + 1)]
                    - 2.0 * self.u_current[c]
                    + self.u_current[self.idx(i, j - 1)])
                    * inv_dy2;

                u_next[c] = (2.0 * self.u_current[c] - prev_coeff * self.u_previous[c]
                    + c2dt2 * (laplacian_x + laplacian_y))
                    / denom;
            }
        }

        self.u_previous.copy_from_slice(&self.u_current);
        self.u_current.copy_from_slice(&u_next);
        self.time += dt;
    }

    /// Maximum stable time step from the 2D CFL condition:
    ///   dt_max = 1 / (c √(1/dx² + 1/dy²))
    pub fn stable_dt(&self) -> f64 {
        1.0 / (self.wave_speed * (1.0 / (self.dx * self.dx) + 1.0 / (self.dy * self.dy)).sqrt())
    }

    /// Discrete total energy of the 2D field.
    ///
    /// E = 0.5 dx dy Σ_{i,j} [ ((u_{ij}^n - u_{ij}^{n-1})/dt)²
    ///     + c² ((u_{i+1,j} - u_{ij})²/dx² + (u_{i,j+1} - u_{ij})²/dy²) ]
    pub fn total_energy(&self, dt: f64) -> f64 {
        let c2 = self.wave_speed * self.wave_speed;
        let inv_dt2 = 1.0 / (dt * dt);
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let inv_dy2 = 1.0 / (self.dy * self.dy);
        let mut energy = 0.0;

        for j in 0..self.ny {
            for i in 0..self.nx {
                let idx = self.idx(i, j);
                let kinetic =
                    (self.u_current[idx] - self.u_previous[idx]).powi(2) * inv_dt2;

                let grad_x = if i < self.nx - 1 {
                    c2 * (self.u_current[self.idx(i + 1, j)] - self.u_current[idx]).powi(2)
                        * inv_dx2
                } else {
                    0.0
                };

                let grad_y = if j < self.ny - 1 {
                    c2 * (self.u_current[self.idx(i, j + 1)] - self.u_current[idx]).powi(2)
                        * inv_dy2
                } else {
                    0.0
                };

                energy += kinetic + grad_x + grad_y;
            }
        }

        0.5 * self.dx * self.dy * energy
    }

    /// Set the displacement at grid point (i, j) in the current time step.
    pub fn set_point(&mut self, i: usize, j: usize, value: f64) {
        let idx = self.idx(i, j);
        self.u_current[idx] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;
    const ENERGY_TOLERANCE: f64 = 0.05;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── 1D Tests ──

    #[test]
    fn test_1d_gaussian_splits_into_two_pulses() {
        // A Gaussian initial displacement with zero velocity splits into two
        // identical half-amplitude pulses traveling in opposite directions.
        let nx = 201;
        let dx = 0.01;
        let c = 1.0;
        let dt = dx / c; // Courant number = 1 (exact transport)
        let center = (nx / 2) as f64 * dx;
        let sigma = 0.05;

        let mut sim = WaveEquation1D::new(nx, dx, c);

        let displacement: Vec<f64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                (-((x - center).powi(2)) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let velocity = vec![0.0; nx];
        sim.set_initial(&displacement, &velocity, dt);

        // Advance enough steps for the pulse to clearly split.
        let n_steps = 40;
        for _ in 0..n_steps {
            sim.step(dt);
        }

        // The original center should have much less displacement than the
        // two traveling peaks.
        let center_idx = nx / 2;
        let left_peak_idx = center_idx - n_steps;
        let right_peak_idx = center_idx + n_steps;

        // With r=1 the leapfrog scheme is exact: the center should be near zero
        // and the two peaks should each be approximately half the original amplitude.
        let center_val = sim.u_current[center_idx];
        assert!(
            center_val.abs() < 0.05,
            "center should be near zero, got {center_val}",
        );

        // The peaks travel at speed c. After n_steps time steps of dt=dx/c,
        // they've moved n_steps grid cells.
        let left_val = sim.u_current[left_peak_idx];
        assert!(
            left_val.abs() > 0.3,
            "left peak too small: {left_val}",
        );
        let right_val = sim.u_current[right_peak_idx];
        assert!(
            right_val.abs() > 0.3,
            "right peak too small: {right_val}",
        );
    }

    #[test]
    fn test_1d_energy_conservation() {
        // With fixed endpoints and no damping, total energy is conserved.
        let nx = 101;
        let dx = 0.01;
        let c = 2.0;
        let dt = 0.8 * dx / c; // r = 0.8 < 1 (stable)

        let mut sim = WaveEquation1D::new(nx, dx, c);

        let displacement: Vec<f64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                let center = 0.5;
                (-((x - center).powi(2)) / (2.0 * 0.03_f64.powi(2))).exp()
            })
            .collect();
        let velocity = vec![0.0; nx];
        sim.set_initial(&displacement, &velocity, dt);

        let e0 = sim.total_energy(dt);

        for _ in 0..200 {
            sim.step(dt);
        }

        let e_final = sim.total_energy(dt);

        let rel_error = ((e_final - e0) / e0).abs();
        assert!(
            rel_error < ENERGY_TOLERANCE,
            "energy not conserved: E0={e0}, E_final={e_final}, rel_error={rel_error}"
        );
    }

    #[test]
    fn test_1d_courant_one_exact_transport() {
        // When r = c dt / dx = 1 the finite-difference scheme reproduces the
        // exact solution of the wave equation (no numerical dispersion).
        // A right-traveling pulse should translate exactly one grid cell per step.
        let nx = 101;
        let dx = 0.01;
        let c = 1.0;
        let dt = dx / c; // r = 1

        let mut sim = WaveEquation1D::new(nx, dx, c);

        // Right-traveling pulse: u(x,0) = f(x), v(x,0) = c f'(x)
        // so that the solution is u(x,t) = f(x - ct).
        let sigma = 0.03;
        let center = 0.5;
        let displacement: Vec<f64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                (-((x - center).powi(2)) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        // Velocity = -c * du/dx for a right-traveling wave.
        let velocity: Vec<f64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                let du_dx = -(x - center) / (sigma * sigma)
                    * (-((x - center).powi(2)) / (2.0 * sigma * sigma)).exp();
                -c * du_dx
            })
            .collect();
        sim.set_initial(&displacement, &velocity, dt);

        let n_shift = 10;
        for _ in 0..n_shift {
            sim.step(dt);
        }

        // The pulse should have shifted exactly n_shift cells to the right.
        // Compare interior points away from boundaries.
        let mut max_error = 0.0_f64;
        for i in 5..nx - 5 - n_shift {
            let expected = displacement[i]; // original profile at index i
            let actual = sim.u_current[i + n_shift]; // should match
            max_error = max_error.max((actual - expected).abs());
        }

        assert!(
            max_error < 0.05,
            "r=1 should give approximate transport, max_error={max_error}"
        );
    }

    #[test]
    fn test_1d_stable_dt() {
        let dx = 0.05;
        let c = 340.0;
        let sim = WaveEquation1D::new(100, dx, c);
        let expected = 1.4705882352941176e-4;
        let dt = sim.stable_dt();
        assert!(
            approx(dt, expected, TOLERANCE),
            "stable_dt={dt}, expected={expected}",
        );
    }

    // ── 2D Tests ──

    #[test]
    fn test_2d_circular_wave_symmetry() {
        // A point source at the center of a square grid should produce a
        // circularly-symmetric wavefront (up to grid anisotropy).
        let n = 51;
        let dx = 0.01;
        let dy = 0.01;
        let c = 1.0;

        let mut sim = WaveEquation2D::new(n, n, dx, dy, c);

        let center = n / 2;
        sim.set_point(center, center, 1.0);

        let dt = 0.5 * sim.stable_dt();
        for _ in 0..20 {
            sim.step(dt);
        }

        // Check four points equidistant from center along axes: should have
        // equal displacement.
        let offset = 10;
        let up = sim.u_current[sim.idx(center, center + offset)];
        let down = sim.u_current[sim.idx(center, center - offset)];
        let left = sim.u_current[sim.idx(center - offset, center)];
        let right = sim.u_current[sim.idx(center + offset, center)];

        let mean = (up + down + left + right) / 4.0;
        assert!(
            (up - mean).abs() < 1e-12 && (down - mean).abs() < 1e-12
                && (left - mean).abs() < 1e-12 && (right - mean).abs() < 1e-12,
            "axial symmetry broken: up={up}, down={down}, left={left}, right={right}"
        );
    }

    #[test]
    fn test_2d_damping_decreases_energy() {
        let n = 31;
        let dx = 0.01;
        let c = 1.0;

        let mut sim = WaveEquation2D::new(n, n, dx, dx, c);
        sim.set_damping(5.0);
        sim.set_point(n / 2, n / 2, 1.0);

        let dt = 0.5 * sim.stable_dt();

        // Let it propagate one step so there is non-zero velocity.
        sim.step(dt);
        let e_start = sim.total_energy(dt);
        assert!(e_start > 0.0, "initial energy must be positive");

        for _ in 0..50 {
            sim.step(dt);
        }

        let e_end = sim.total_energy(dt);
        assert!(
            e_end < e_start,
            "damping should decrease energy: E_start={e_start}, E_end={e_end}"
        );
    }

    #[test]
    fn test_2d_stable_dt() {
        let dx = 0.1;
        let dy = 0.2;
        let c = 3.0;
        let sim = WaveEquation2D::new(10, 10, dx, dy, c);
        let expected = 0.0298142396999972;
        let dt = sim.stable_dt();
        assert!(
            approx(dt, expected, TOLERANCE),
            "stable_dt={dt}, expected={expected}",
        );
    }

    #[test]
    fn test_1d_add_source() {
        let nx = 50;
        let dx = 0.01;
        let c = 1.0;
        let mut sim = WaveEquation1D::new(nx, dx, c);
        sim.add_source(25, 3.0);
        let val1 = sim.u_current[25];
        assert!(
            approx(val1, 3.0, TOLERANCE),
            "add_source should add amplitude, got {val1}",
        );
        sim.add_source(25, 2.0);
        let val2 = sim.u_current[25];
        assert!(
            approx(val2, 5.0, TOLERANCE),
            "add_source should accumulate, got {val2}",
        );
    }

    #[test]
    fn test_1d_courant_number() {
        let dx = 0.05;
        let c = 340.0;
        let sim = WaveEquation1D::new(100, dx, c);
        let dt = 0.0001;
        let r = sim.courant_number(dt);
        let expected = 0.68;
        assert!(
            approx(r, expected, TOLERANCE),
            "courant_number={r}, expected {expected}"
        );
    }

    // ── Absorbing BC Test ──

    #[test]
    fn test_absorbing_bc_no_reflection() {
        // Launch a right-traveling pulse toward the right boundary with
        // absorbing BC. After it exits, the field should be nearly zero.
        let nx = 201;
        let dx = 0.01;
        let c = 1.0;
        let dt = dx / c; // r = 1 (ABC is exact for r = 1)

        let mut sim = WaveEquation1D::new(nx, dx, c);

        // Narrow Gaussian starting near the right boundary.
        let center = 0.85 * (nx as f64) * dx;
        let sigma = 0.02;
        let displacement: Vec<f64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                (-((x - center).powi(2)) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        // Right-traveling: v = -c du/dx
        let velocity: Vec<f64> = (0..nx)
            .map(|i| {
                let x = i as f64 * dx;
                let du_dx = -(x - center) / (sigma * sigma)
                    * (-((x - center).powi(2)) / (2.0 * sigma * sigma)).exp();
                -c * du_dx
            })
            .collect();
        sim.set_initial(&displacement, &velocity, dt);

        // Run long enough for the pulse to exit.
        for _ in 0..100 {
            sim.step_absorbing(dt);
        }

        // With perfect absorption, the field should be essentially zero.
        let max_residual: f64 = sim
            .u_current
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(
            max_residual < 0.05,
            "absorbing BC should reduce reflections, max residual={max_residual}"
        );
    }
}
