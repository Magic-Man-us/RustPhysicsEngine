use crate::math::constants::K_B;

const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
const MANTISSA_BITS: u32 = 53;
const LN_MIN_CLAMP: f64 = 1e-300;

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> (64 - MANTISSA_BITS)) as f64 / (1u64 << MANTISSA_BITS) as f64
    }

    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(LN_MIN_CLAMP);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Monte Carlo Integration
// ---------------------------------------------------------------------------

pub fn mc_integrate_1d(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    n: usize,
    rng: &mut Rng,
) -> f64 {
    let width = b - a;
    let sum: f64 = (0..n).map(|_| f(a + rng.next_f64() * width)).sum();
    width * sum / n as f64
}

pub fn mc_integrate_2d(
    f: &dyn Fn(f64, f64) -> f64,
    x_range: (f64, f64),
    y_range: (f64, f64),
    n: usize,
    rng: &mut Rng,
) -> f64 {
    let x_width = x_range.1 - x_range.0;
    let y_width = y_range.1 - y_range.0;
    let area = x_width * y_width;
    let sum: f64 = (0..n)
        .map(|_| {
            let x = x_range.0 + rng.next_f64() * x_width;
            let y = y_range.0 + rng.next_f64() * y_width;
            f(x, y)
        })
        .sum();
    area * sum / n as f64
}

pub fn mc_estimate_pi(n: usize, rng: &mut Rng) -> f64 {
    let inside = (0..n)
        .filter(|_| {
            let x = rng.next_f64();
            let y = rng.next_f64();
            x * x + y * y <= 1.0
        })
        .count();
    4.0 * inside as f64 / n as f64
}

pub fn mc_integrate_importance(
    f: &dyn Fn(f64) -> f64,
    pdf: &dyn Fn(f64) -> f64,
    sampler: &dyn Fn(&mut Rng) -> f64,
    n: usize,
    rng: &mut Rng,
) -> f64 {
    let sum: f64 = (0..n)
        .map(|_| {
            let x = sampler(rng);
            let p = pdf(x);
            if p > 0.0 {
                f(x) / p
            } else {
                0.0
            }
        })
        .sum();
    sum / n as f64
}

// ---------------------------------------------------------------------------
// Random Walks
// ---------------------------------------------------------------------------

pub fn random_walk_1d(steps: usize, step_size: f64, rng: &mut Rng) -> Vec<f64> {
    let mut positions = Vec::with_capacity(steps + 1);
    positions.push(0.0);
    for _ in 0..steps {
        let last = *positions.last().expect("positions is never empty");
        let direction = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
        positions.push(last + direction * step_size);
    }
    positions
}

pub fn random_walk_2d(steps: usize, step_size: f64, rng: &mut Rng) -> Vec<(f64, f64)> {
    let mut positions = Vec::with_capacity(steps + 1);
    positions.push((0.0, 0.0));
    for _ in 0..steps {
        let (x, y) = *positions.last().expect("positions is never empty");
        let angle = rng.next_f64() * 2.0 * std::f64::consts::PI;
        positions.push((x + step_size * angle.cos(), y + step_size * angle.sin()));
    }
    positions
}

pub fn random_walk_3d(steps: usize, step_size: f64, rng: &mut Rng) -> Vec<(f64, f64, f64)> {
    let mut positions = Vec::with_capacity(steps + 1);
    positions.push((0.0, 0.0, 0.0));
    for _ in 0..steps {
        let (x, y, z) = *positions.last().expect("positions is never empty");
        // Uniform point on unit sphere via Marsaglia's method: use Gaussian components
        let gx = rng.next_gaussian();
        let gy = rng.next_gaussian();
        let gz = rng.next_gaussian();
        let norm = (gx * gx + gy * gy + gz * gz).sqrt();
        if norm < 1e-300 {
            positions.push((x, y, z));
            continue;
        }
        let dx = step_size * gx / norm;
        let dy = step_size * gy / norm;
        let dz = step_size * gz / norm;
        positions.push((x + dx, y + dy, z + dz));
    }
    positions
}

// ---------------------------------------------------------------------------
// Stochastic Processes
// ---------------------------------------------------------------------------

pub fn wiener_process(n_steps: usize, dt: f64, rng: &mut Rng) -> Vec<f64> {
    let sqrt_dt = dt.sqrt();
    let mut path = Vec::with_capacity(n_steps + 1);
    path.push(0.0);
    for _ in 0..n_steps {
        let last = *path.last().expect("path is never empty");
        path.push(last + sqrt_dt * rng.next_gaussian());
    }
    path
}

#[allow(clippy::too_many_arguments)]
pub fn ornstein_uhlenbeck(
    n_steps: usize,
    dt: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    x0: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let sqrt_dt = dt.sqrt();
    let mut path = Vec::with_capacity(n_steps + 1);
    path.push(x0);
    for _ in 0..n_steps {
        let x = *path.last().expect("path is never empty");
        let drift = theta * (mu - x) * dt;
        let diffusion = sigma * sqrt_dt * rng.next_gaussian();
        path.push(x + drift + diffusion);
    }
    path
}

#[allow(clippy::too_many_arguments)]
pub fn langevin_step(
    x: f64,
    v: f64,
    force: f64,
    mass: f64,
    gamma: f64,
    temperature: f64,
    dt: f64,
    rng: &mut Rng,
) -> (f64, f64) {
    let sqrt_dt = dt.sqrt();
    let noise_amplitude = (2.0 * gamma * K_B * temperature / mass).sqrt();
    let v_new = v + (force / mass - gamma * v) * dt + noise_amplitude * sqrt_dt * rng.next_gaussian();
    let x_new = x + v_new * dt;
    (x_new, v_new)
}

// ---------------------------------------------------------------------------
// Metropolis Algorithm
// ---------------------------------------------------------------------------

pub fn metropolis_step(
    energy_current: f64,
    energy_proposed: f64,
    temperature: f64,
    rng: &mut Rng,
) -> bool {
    let delta_e = energy_proposed - energy_current;
    if delta_e < 0.0 {
        return true;
    }
    let acceptance_prob = (-delta_e / (K_B * temperature)).exp();
    rng.next_f64() < acceptance_prob
}

pub fn metropolis_sample(
    energy_fn: &dyn Fn(f64) -> f64,
    proposal: &dyn Fn(f64, &mut Rng) -> f64,
    x0: f64,
    n_samples: usize,
    temperature: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(n_samples);
    let mut x = x0;
    let mut e = energy_fn(x);
    for _ in 0..n_samples {
        let x_proposed = proposal(x, rng);
        let e_proposed = energy_fn(x_proposed);
        if metropolis_step(e, e_proposed, temperature, rng) {
            x = x_proposed;
            e = e_proposed;
        }
        samples.push(x);
    }
    samples
}

// ---------------------------------------------------------------------------
// Ising Model (1D)
// ---------------------------------------------------------------------------

pub fn ising_energy_1d(spins: &[i8], j_coupling: f64, h_field: f64) -> f64 {
    let n = spins.len();
    let mut interaction_sum: f64 = 0.0;
    for i in 0..n.saturating_sub(1) {
        interaction_sum += (spins[i] as f64) * (spins[i + 1] as f64);
    }
    let field_sum: f64 = spins.iter().map(|&s| s as f64).sum();
    -j_coupling * interaction_sum - h_field * field_sum
}

pub fn ising_magnetization(spins: &[i8]) -> f64 {
    let sum: f64 = spins.iter().map(|&s| s as f64).sum();
    sum / spins.len() as f64
}

pub fn ising_step_1d(
    spins: &mut [i8],
    j_coupling: f64,
    h_field: f64,
    temperature: f64,
    rng: &mut Rng,
) {
    let n = spins.len();
    if n == 0 {
        return;
    }
    let idx = (rng.next_u64() as usize) % n;

    // Compute local energy change from flipping spin at idx
    let s = spins[idx] as f64;
    let mut neighbor_sum = 0.0;
    if idx > 0 {
        neighbor_sum += spins[idx - 1] as f64;
    }
    if idx + 1 < n {
        neighbor_sum += spins[idx + 1] as f64;
    }
    // ΔE = 2 J s_i Σ s_neighbors + 2 H s_i
    let delta_e = 2.0 * j_coupling * s * neighbor_sum + 2.0 * h_field * s;

    if delta_e < 0.0 || rng.next_f64() < (-delta_e / (K_B * temperature)).exp() {
        spins[idx] = -spins[idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SEED: u64 = 42;
    const PI_TOLERANCE: f64 = 0.05;
    const INTEGRAL_TOLERANCE: f64 = 0.10;

    #[test]
    fn test_mc_estimate_pi() {
        let mut rng = Rng::new(TEST_SEED);
        let pi_est = mc_estimate_pi(10_000, &mut rng);
        let relative_error = (pi_est - std::f64::consts::PI).abs() / std::f64::consts::PI;
        assert!(
            relative_error < PI_TOLERANCE,
            "Pi estimate {pi_est} has relative error {relative_error} exceeding {PI_TOLERANCE}"
        );
    }

    #[test]
    fn test_mc_integrate_sin() {
        let mut rng = Rng::new(TEST_SEED);
        let result = mc_integrate_1d(&f64::sin, 0.0, std::f64::consts::PI, 10_000, &mut rng);
        let expected = 2.0;
        let relative_error = (result - expected).abs() / expected;
        assert!(
            relative_error < INTEGRAL_TOLERANCE,
            "Integral of sin(x) from 0 to pi = {result}, expected ~{expected}, relative error {relative_error}"
        );
    }

    #[test]
    fn test_wiener_process_starts_at_zero() {
        let mut rng = Rng::new(TEST_SEED);
        let path = wiener_process(100, 0.01, &mut rng);
        assert_eq!(path[0], 0.0);
        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_ou_process_mean_approaches_mu() {
        let mut rng = Rng::new(TEST_SEED);
        let mu = 5.0;
        let theta = 10.0;
        let sigma = 0.1;
        let dt = 0.01;
        let n_steps = 10_000;
        let path = ornstein_uhlenbeck(n_steps, dt, theta, mu, sigma, 0.0, &mut rng);
        // Average over the second half of the path (after equilibration)
        let second_half = &path[n_steps / 2..];
        let mean: f64 = second_half.iter().sum::<f64>() / second_half.len() as f64;
        assert!(
            (mean - mu).abs() < 0.5,
            "OU process mean {mean} should approach mu={mu}"
        );
    }

    #[test]
    fn test_metropolis_always_accepts_lower_energy() {
        let mut rng = Rng::new(TEST_SEED);
        for _ in 0..100 {
            assert!(metropolis_step(10.0, 5.0, 300.0, &mut rng));
        }
    }

    #[test]
    fn test_ising_all_up_energy() {
        let n = 10;
        let spins = vec![1i8; n];
        let j = 1.0;
        let h = 0.0;
        let energy = ising_energy_1d(&spins, j, h);
        // All aligned: E = -J * (N-1) * 1 = -(N-1)
        let expected = -j * (n as f64 - 1.0);
        assert!(
            (energy - expected).abs() < 1e-12,
            "All-up energy {energy} != expected {expected}"
        );
    }

    #[test]
    fn test_random_walk_1d_starts_at_zero_and_correct_length() {
        let mut rng = Rng::new(TEST_SEED);
        let steps = 50;
        let walk = random_walk_1d(steps, 1.0, &mut rng);
        assert_eq!(walk[0], 0.0);
        assert_eq!(walk.len(), steps + 1);
    }
}
