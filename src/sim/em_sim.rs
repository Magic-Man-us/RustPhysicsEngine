use crate::math::constants::{C, EPSILON_0, MU_0, PI};

// ── Mur ABC boundary storage ──

struct MurBoundary {
    ez_left_prev: f64,
    ez_left_curr: f64,
    ez_right_prev: f64,
    ez_right_curr: f64,
}

// ── FDTD 1D (TM mode: Ez, Hy) ──

pub struct Fdtd1D {
    pub ez: Vec<f64>,
    pub hy: Vec<f64>,
    pub nx: usize,
    pub dx: f64,
    pub dt: f64,
    pub epsilon: Vec<f64>,
    pub mu: Vec<f64>,
    pub conductivity: Vec<f64>,
    pub time: f64,
    pub time_step: u64,
    mur: MurBoundary,
}

const COURANT_FACTOR_1D: f64 = 0.5;

impl Fdtd1D {
    #[must_use]
    pub fn new(nx: usize, dx: f64) -> Self {
        let dt = Self::stable_dt_for_dx(dx);
        Self {
            ez: vec![0.0; nx],
            hy: vec![0.0; nx],
            nx,
            dx,
            dt,
            epsilon: vec![1.0; nx],
            mu: vec![1.0; nx],
            conductivity: vec![0.0; nx],
            time: 0.0,
            time_step: 0,
            mur: MurBoundary {
                ez_left_prev: 0.0,
                ez_left_curr: 0.0,
                ez_right_prev: 0.0,
                ez_right_curr: 0.0,
            },
        }
    }

    pub fn set_material(&mut self, start: usize, end: usize, epsilon_r: f64, mu_r: f64, sigma: f64) {
        for i in start..end.min(self.nx) {
            self.epsilon[i] = epsilon_r;
            self.mu[i] = mu_r;
            self.conductivity[i] = sigma;
        }
    }

    pub fn step(&mut self) {
        let dt = self.dt;
        let dx = self.dx;

        // Store boundary values for Mur ABC before update
        self.mur.ez_left_prev = self.mur.ez_left_curr;
        self.mur.ez_left_curr = self.ez[1];
        self.mur.ez_right_prev = self.mur.ez_right_curr;
        self.mur.ez_right_curr = self.ez[self.nx - 2];

        // Update Hy (leapfrog half-step)
        // Hy[i]^(n+1/2) = Hy[i]^(n-1/2) - (dt/(mu_0 * mu_r[i] * dx)) * (Ez[i+1]^n - Ez[i]^n)
        for i in 0..self.nx - 1 {
            self.hy[i] -= (dt / (MU_0 * self.mu[i] * dx)) * (self.ez[i + 1] - self.ez[i]);
        }

        // Update Ez (full step) with lossy material support
        // For lossy: Ez[i] = ca * Ez[i] - cb * (Hy[i] - Hy[i-1])
        // ca = (1 - sigma*dt/(2*eps_0*eps_r)) / (1 + sigma*dt/(2*eps_0*eps_r))
        // cb = (dt/(eps_0*eps_r*dx)) / (1 + sigma*dt/(2*eps_0*eps_r))
        for i in 1..self.nx - 1 {
            let sigma = self.conductivity[i];
            let eps_r = self.epsilon[i];
            let loss_term = sigma * dt / (2.0 * EPSILON_0 * eps_r);
            let ca = (1.0 - loss_term) / (1.0 + loss_term);
            let cb = (dt / (EPSILON_0 * eps_r * dx)) / (1.0 + loss_term);
            self.ez[i] = ca * self.ez[i] - cb * (self.hy[i] - self.hy[i - 1]);
        }

        self.time += dt;
        self.time_step += 1;
    }

    pub fn add_source_soft(&mut self, position: usize, value: f64) {
        self.ez[position] += value;
    }

    pub fn add_source_hard(&mut self, position: usize, value: f64) {
        self.ez[position] = value;
    }

    #[must_use]
    pub fn gaussian_pulse(t: f64, t0: f64, spread: f64) -> f64 {
        (-((t - t0) / spread).powi(2)).exp()
    }

    #[must_use]
    pub fn sinusoidal_source(t: f64, frequency: f64) -> f64 {
        (2.0 * PI * frequency * t).sin()
    }

    pub fn apply_abc_mur(&mut self) {
        let coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx);

        // Left boundary: Ez[0]^(n+1) = Ez[1]^n + coeff * (Ez[1]^(n+1) - Ez[0]^n)
        self.ez[0] = self.mur.ez_left_prev + coeff * (self.ez[1] - self.mur.ez_left_curr);

        // Right boundary: Ez[nx-1]^(n+1) = Ez[nx-2]^n + coeff * (Ez[nx-2]^(n+1) - Ez[nx-1]^n)
        let last = self.nx - 1;
        self.ez[last] = self.mur.ez_right_prev + coeff * (self.ez[last - 1] - self.mur.ez_right_curr);
    }

    #[must_use]
    pub fn total_energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.nx {
            energy += 0.5 * EPSILON_0 * self.epsilon[i] * self.ez[i] * self.ez[i] * self.dx;
        }
        for i in 0..self.nx {
            energy += 0.5 * MU_0 * self.mu[i] * self.hy[i] * self.hy[i] * self.dx;
        }
        energy
    }

    #[must_use]
    pub fn stable_dt_for_dx(dx: f64) -> f64 {
        // CFL for 1D: dt <= dx / (c * sqrt(1)) with Courant factor
        dx * COURANT_FACTOR_1D / C
    }
}

// ── FDTD 2D (TM mode: Ez, Hx, Hy) ──

pub struct Fdtd2D {
    pub ez: Vec<f64>,
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub dt: f64,
    pub epsilon: Vec<f64>,
    pub time_step: u64,
}

impl Fdtd2D {
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        let dt = Self::stable_dt_for_grid(dx, dy);
        let n = nx * ny;
        Self {
            ez: vec![0.0; n],
            hx: vec![0.0; n],
            hy: vec![0.0; n],
            nx,
            ny,
            dx,
            dy,
            dt,
            epsilon: vec![1.0; n],
            time_step: 0,
        }
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ny + j
    }

    pub fn step(&mut self) {
        let dt = self.dt;
        let dx = self.dx;
        let dy = self.dy;

        // Update Hx: dHx/dt = -(1/mu_0) * dEz/dy
        // Hx[i,j] -= (dt/(mu_0*dy)) * (Ez[i,j+1] - Ez[i,j])
        for i in 0..self.nx {
            for j in 0..self.ny - 1 {
                let idx = self.idx(i, j);
                let idx_jp1 = self.idx(i, j + 1);
                self.hx[idx] -= (dt / (MU_0 * dy)) * (self.ez[idx_jp1] - self.ez[idx]);
            }
        }

        // Update Hy: dHy/dt = (1/mu_0) * dEz/dx
        // Hy[i,j] += (dt/(mu_0*dx)) * (Ez[i+1,j] - Ez[i,j])
        for i in 0..self.nx - 1 {
            for j in 0..self.ny {
                let idx = self.idx(i, j);
                let idx_ip1 = self.idx(i + 1, j);
                self.hy[idx] += (dt / (MU_0 * dx)) * (self.ez[idx_ip1] - self.ez[idx]);
            }
        }

        // Update Ez: dEz/dt = (1/(eps_0*eps_r)) * (dHy/dx - dHx/dy)
        // Ez[i,j] += (dt/(eps_0*eps_r)) * ((Hy[i,j]-Hy[i-1,j])/dx - (Hx[i,j]-Hx[i,j-1])/dy)
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let idx = self.idx(i, j);
                let idx_im1 = self.idx(i - 1, j);
                let idx_jm1 = self.idx(i, j - 1);
                let eps_r = self.epsilon[idx];
                self.ez[idx] += (dt / (EPSILON_0 * eps_r))
                    * ((self.hy[idx] - self.hy[idx_im1]) / dx
                        - (self.hx[idx] - self.hx[idx_jm1]) / dy);
            }
        }

        self.time_step += 1;
    }

    pub fn add_source(&mut self, i: usize, j: usize, value: f64) {
        let idx = self.idx(i, j);
        self.ez[idx] += value;
    }

    #[must_use]
    pub fn total_energy(&self) -> f64 {
        let cell_area = self.dx * self.dy;
        let mut energy = 0.0;
        for i in 0..self.nx * self.ny {
            energy += 0.5 * EPSILON_0 * self.epsilon[i] * self.ez[i] * self.ez[i] * cell_area;
            energy += 0.5 * MU_0 * (self.hx[i] * self.hx[i] + self.hy[i] * self.hy[i]) * cell_area;
        }
        energy
    }

    #[must_use]
    pub fn stable_dt_for_grid(dx: f64, dy: f64) -> f64 {
        // CFL for 2D: dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2))
        // Apply Courant factor 0.5 for safety
        COURANT_FACTOR_1D / (C * (1.0 / (dx * dx) + 1.0 / (dy * dy)).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REL_TOL: f64 = 0.05;

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b.abs() < 1e-30 {
            return a.abs() < tol;
        }
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_1d_vacuum_pulse_speed() {
        // A pulse in vacuum should travel at speed c.
        // Place a Gaussian pulse at the center, run for N steps,
        // verify the peak has moved c*dt*N cells.
        let nx = 500;
        let dx = 1e-3;
        let mut sim = Fdtd1D::new(nx, dx);
        let source_pos = nx / 2;

        let t0 = 30.0 * sim.dt;
        let spread = 10.0 * sim.dt;

        // Inject a Gaussian pulse for enough steps to build it
        let inject_steps = 60_u64;
        for _ in 0..inject_steps {
            let val = Fdtd1D::gaussian_pulse(sim.time, t0, spread);
            sim.add_source_soft(source_pos, val);
            sim.step();
            sim.apply_abc_mur();
        }

        // Find peak position after injection
        let peak_before = sim
            .ez
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Run more steps without source
        let travel_steps = 200_u64;
        for _ in 0..travel_steps {
            sim.step();
            sim.apply_abc_mur();
        }

        let peak_after = sim
            .ez
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let cells_traveled = (peak_after as f64 - peak_before as f64).abs();
        let expected_cells = C * sim.dt * travel_steps as f64 / dx;

        assert!(
            approx_rel(cells_traveled, expected_cells, 0.1),
            "Pulse traveled {cells_traveled} cells, expected {expected_cells}"
        );
    }

    #[test]
    fn test_1d_energy_conservation_vacuum() {
        // In lossless vacuum with ABC boundaries, energy should be roughly conserved
        // (ABC absorbs outgoing waves, so energy decreases but never increases).
        let nx = 300;
        let dx = 1e-3;
        let mut sim = Fdtd1D::new(nx, dx);
        let source_pos = nx / 2;

        let t0 = 20.0 * sim.dt;
        let spread = 8.0 * sim.dt;

        // Build pulse
        for _ in 0..40 {
            let val = Fdtd1D::gaussian_pulse(sim.time, t0, spread);
            sim.add_source_soft(source_pos, val);
            sim.step();
            sim.apply_abc_mur();
        }

        let energy_after_injection = sim.total_energy();
        assert!(energy_after_injection > 0.0, "Should have nonzero energy after injection");

        // Propagate without source — energy should not increase
        let mut max_energy = energy_after_injection;
        for _ in 0..100 {
            sim.step();
            sim.apply_abc_mur();
            let e = sim.total_energy();
            // Allow tiny floating-point overshoot
            if e > max_energy * 1.001 {
                panic!(
                    "Energy increased from {max_energy} to {e}, simulation is unstable"
                );
            }
            if e > max_energy {
                max_energy = e;
            }
        }
    }

    #[test]
    fn test_1d_dielectric_slab_slows_wave() {
        // A wave in a dielectric with epsilon_r = 4 should travel at c/2.
        let nx = 1000;
        let dx = 1e-3;
        let mut sim = Fdtd1D::new(nx, dx);

        let epsilon_r = 4.0;
        // Fill entire domain with dielectric
        sim.set_material(0, nx, epsilon_r, 1.0, 0.0);

        let source_pos = 100;
        let t0 = 30.0 * sim.dt;
        let spread = 10.0 * sim.dt;

        for _ in 0..60 {
            let val = Fdtd1D::gaussian_pulse(sim.time, t0, spread);
            sim.add_source_soft(source_pos, val);
            sim.step();
        }

        let peak_before = sim.ez[source_pos + 1..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i + source_pos + 1)
            .unwrap();

        let travel_steps = 400_u64;
        for _ in 0..travel_steps {
            sim.step();
        }

        let peak_after = sim.ez[source_pos + 1..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i + source_pos + 1)
            .unwrap();

        let cells_traveled = (peak_after as f64 - peak_before as f64).abs();
        let n = epsilon_r.sqrt(); // refractive index
        let expected_cells = (C / n) * sim.dt * travel_steps as f64 / dx;

        assert!(
            approx_rel(cells_traveled, expected_cells, 0.15),
            "In dielectric (n={n}), pulse traveled {cells_traveled} cells, expected {expected_cells}"
        );
    }

    #[test]
    fn test_1d_lossy_material_attenuates() {
        // A wave in a lossy material should lose energy over time.
        let nx = 500;
        let dx = 1e-3;
        let mut sim = Fdtd1D::new(nx, dx);

        // Set conductivity in the right half
        let sigma = 0.01;
        sim.set_material(nx / 2, nx, 1.0, 1.0, sigma);

        let source_pos = nx / 4;
        let t0 = 20.0 * sim.dt;
        let spread = 8.0 * sim.dt;

        // Inject pulse
        for _ in 0..40 {
            let val = Fdtd1D::gaussian_pulse(sim.time, t0, spread);
            sim.add_source_soft(source_pos, val);
            sim.step();
        }

        let energy_before = sim.total_energy();

        // Let pulse enter lossy region
        for _ in 0..500 {
            sim.step();
        }

        let energy_after = sim.total_energy();

        assert!(
            energy_after < energy_before,
            "Lossy material should attenuate: before={energy_before}, after={energy_after}"
        );
    }

    #[test]
    fn test_2d_cfl_stability() {
        let dx = 1e-3;
        let dy = 1e-3;
        let dt = Fdtd2D::stable_dt_for_grid(dx, dy);
        let cfl_limit = 1.0 / (C * (1.0 / (dx * dx) + 1.0 / (dy * dy)).sqrt());

        assert!(
            dt <= cfl_limit,
            "dt={dt} exceeds CFL limit={cfl_limit}"
        );
        assert!(dt > 0.0, "dt must be positive");

        // Verify the Courant factor is applied
        assert!(
            approx_rel(dt, cfl_limit * COURANT_FACTOR_1D, 1e-10),
            "dt should be CFL_limit * Courant_factor"
        );
    }

    #[test]
    fn test_2d_point_source_symmetry() {
        // A point source at the center of a uniform grid should produce
        // a circularly symmetric expanding wave.
        let n = 101;
        let dx = 1e-3;
        let dy = 1e-3;
        let mut sim = Fdtd2D::new(n, n, dx, dy);

        let center = n / 2;
        let t0 = 20.0 * sim.dt;
        let spread = 8.0 * sim.dt;

        // Inject point source at center
        for step in 0..40_u64 {
            let t = step as f64 * sim.dt;
            let val = Fdtd1D::gaussian_pulse(t, t0, spread);
            sim.add_source(center, center, val);
            sim.step();
        }

        // Propagate
        for _ in 0..30 {
            sim.step();
        }

        // Check 4-fold symmetry: Ez at (center+r, center) should approximately
        // equal Ez at (center-r, center), (center, center+r), (center, center-r)
        let check_radius = 10;
        let i = center;
        let j = center;

        let ez_right = sim.ez[sim.idx(i + check_radius, j)];
        let ez_left = sim.ez[sim.idx(i - check_radius, j)];
        let ez_up = sim.ez[sim.idx(i, j + check_radius)];
        let ez_down = sim.ez[sim.idx(i, j - check_radius)];

        let avg = (ez_right.abs() + ez_left.abs() + ez_up.abs() + ez_down.abs()) / 4.0;
        assert!(avg > 1e-15, "Fields should be nonzero at radius {check_radius}");

        // Each direction should be within REL_TOL of the average
        assert!(
            approx_rel(ez_right.abs(), avg, REL_TOL),
            "Right={ez_right}, avg={avg}"
        );
        assert!(
            approx_rel(ez_left.abs(), avg, REL_TOL),
            "Left={ez_left}, avg={avg}"
        );
        assert!(
            approx_rel(ez_up.abs(), avg, REL_TOL),
            "Up={ez_up}, avg={avg}"
        );
        assert!(
            approx_rel(ez_down.abs(), avg, REL_TOL),
            "Down={ez_down}, avg={avg}"
        );
    }

    #[test]
    fn test_gaussian_pulse_shape() {
        let peak = Fdtd1D::gaussian_pulse(5.0, 5.0, 1.0);
        assert!(approx_rel(peak, 1.0, 1e-12), "Peak should be 1.0");

        let off = Fdtd1D::gaussian_pulse(0.0, 5.0, 1.0);
        assert!(off < 1e-10, "Far from center should be ~0");
    }

    #[test]
    fn test_sinusoidal_source_values() {
        let val = Fdtd1D::sinusoidal_source(0.0, 1.0);
        assert!(val.abs() < 1e-12, "sin(0) = 0");

        let val_quarter = Fdtd1D::sinusoidal_source(0.25, 1.0);
        assert!(
            approx_rel(val_quarter, 1.0, 1e-10),
            "sin(pi/2) = 1, got {val_quarter}"
        );
    }

    #[test]
    fn test_1d_stable_dt() {
        let dx = 1e-3;
        let dt = Fdtd1D::stable_dt_for_dx(dx);
        let cfl_limit = dx / C;
        assert!(dt < cfl_limit, "dt must be strictly below CFL limit");
        assert!(dt > 0.0);
    }
}
