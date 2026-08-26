//! Strange attractors: 3-D chaotic flows and 2-D chaotic maps with
//! trajectory integration, Lyapunov spectra (Benettin
//! renormalization), Kaplan-Yorke dimension, Poincaré sections,
//! bifurcation diagrams, and dimension estimators
//! (Grassberger-Procaccia correlation dimension, box counting,
//! Rosenstein's largest-Lyapunov method, Feigenbaum ratios).

use crate::math::{Vec2, Vec3};
use crate::spatial::primitives::{Plane, Rect};

/// A 3-D autonomous flow ẋ = f(x) with a preferred time step.
pub struct Attractor3 {
    pub derivs: Box<dyn Fn(Vec3) -> Vec3>,
    pub dt: f64,
}

/// Fixed-step integration schemes (`Rk45` takes two half steps and
/// keeps the fifth-order combination, giving adaptive-quality
/// accuracy at fixed cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    Euler,
    Rk4,
    Rk45,
}

fn rk4_step(f: &dyn Fn(Vec3) -> Vec3, x: Vec3, dt: f64) -> Vec3 {
    let k1 = f(x);
    let k2 = f(x + k1 * (dt / 2.0));
    let k3 = f(x + k2 * (dt / 2.0));
    let k4 = f(x + k3 * dt);
    x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
}

impl Attractor3 {
    fn step(&self, x: Vec3, method: Integrator) -> Vec3 {
        match method {
            Integrator::Euler => x + (self.derivs)(x) * self.dt,
            Integrator::Rk4 => rk4_step(&self.derivs, x, self.dt),
            Integrator::Rk45 => {
                // Two half RK4 steps with Richardson extrapolation:
                // fifth-order accurate combination.
                let full = rk4_step(&self.derivs, x, self.dt);
                let half = rk4_step(&self.derivs, x, self.dt / 2.0);
                let two = rk4_step(&self.derivs, half, self.dt / 2.0);
                two + (two - full) * (1.0 / 15.0)
            }
        }
    }

    /// Integrates `n` steps from `x0`, returning the n+1 visited
    /// states (including the start).
    #[must_use]
    pub fn trajectory(&self, x0: Vec3, n: usize, method: Integrator) -> Vec<Vec3> {
        let mut out = Vec::with_capacity(n + 1);
        let mut x = x0;
        out.push(x);
        for _ in 0..n {
            x = self.step(x, method);
            out.push(x);
        }
        out
    }

    /// Full Lyapunov spectrum by the Benettin method: three
    /// orthonormal perturbation vectors evolved through the
    /// finite-difference flow map and re-orthonormalized (modified
    /// Gram-Schmidt) each step; the exponents are the average log
    /// stretching factors. Uses RK4 with step `dt`.
    ///
    /// # Panics
    /// Panics unless `n >= 100` and `dt > 0`.
    #[must_use]
    pub fn lyapunov_spectrum(&self, x0: Vec3, n: usize, dt: f64) -> [f64; 3] {
        assert!(n >= 100, "need enough steps to average");
        assert!(dt > 0.0, "dt must be positive");
        let eps = 1e-7;
        let flow = |x: Vec3| rk4_step(&self.derivs, x, dt);
        // Settle onto the attractor first.
        let mut x = x0;
        for _ in 0..1000 {
            x = flow(x);
        }
        let mut basis = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let mut sums = [0.0f64; 3];
        for _ in 0..n {
            let fx = flow(x);
            // Propagate each basis vector through the linearized flow
            // by finite differences.
            let mut w = [Vec3::ZERO; 3];
            for (wk, bk) in w.iter_mut().zip(&basis) {
                *wk = (flow(x + *bk * eps) - fx) * (1.0 / eps);
            }
            // Modified Gram-Schmidt with growth bookkeeping.
            for k in 0..3 {
                for bj in basis.iter().take(k) {
                    let proj = w[k].dot(bj);
                    w[k] = w[k] - *bj * proj;
                }
                let norm = w[k].magnitude();
                sums[k] += norm.max(1e-300).ln();
                basis[k] = w[k] * (1.0 / norm.max(1e-300));
            }
            x = fx;
        }
        let t = n as f64 * dt;
        [sums[0] / t, sums[1] / t, sums[2] / t]
    }

    /// Points where the trajectory crosses the plane in the +normal
    /// direction, linearly interpolated between steps.
    #[must_use]
    pub fn poincare_section(&self, x0: Vec3, n: usize, plane: &Plane) -> Vec<Vec3> {
        let mut out = Vec::new();
        let mut x = x0;
        let mut d_prev = plane.normal.dot(&x) + plane.d;
        for _ in 0..n {
            let next = self.step(x, Integrator::Rk4);
            let d_next = plane.normal.dot(&next) + plane.d;
            if d_prev < 0.0 && d_next >= 0.0 {
                let t = d_prev / (d_prev - d_next);
                out.push(x + (next - x) * t);
            }
            x = next;
            d_prev = d_next;
        }
        out
    }

    /// Histogram of the trajectory projected onto the xy plane over
    /// `bounds` (row-major, x fastest). The first 100 steps are
    /// discarded as transient.
    ///
    /// # Panics
    /// Panics on an empty grid or degenerate bounds.
    #[must_use]
    pub fn density_map(
        &self,
        x0: Vec3,
        n: usize,
        res: (usize, usize),
        bounds: &Rect,
    ) -> Vec<u32> {
        assert!(res.0 > 0 && res.1 > 0, "grid must be non-empty");
        let size = bounds.max - bounds.min;
        assert!(size.x > 0.0 && size.y > 0.0, "bounds must have positive area");
        let mut grid = vec![0u32; res.0 * res.1];
        let mut x = x0;
        for i in 0..n {
            x = self.step(x, Integrator::Rk4);
            if i < 100 {
                continue;
            }
            let ix = ((x.x - bounds.min.x) / size.x * res.0 as f64).floor();
            let iy = ((x.y - bounds.min.y) / size.y * res.1 as f64).floor();
            if ix >= 0.0 && iy >= 0.0 && (ix as usize) < res.0 && (iy as usize) < res.1 {
                grid[iy as usize * res.0 + ix as usize] += 1;
            }
        }
        grid
    }
}

/// Kaplan-Yorke (Lyapunov) dimension of a spectrum: with exponents
/// sorted descending and k the largest index with Σ₁ᵏ λᵢ ≥ 0,
/// D = k + Σ₁ᵏ λᵢ / |λ_{k+1}|.
#[must_use]
pub fn kaplan_yorke_dimension(spectrum: [f64; 3]) -> f64 {
    let mut s = spectrum;
    s.sort_by(|a, b| b.total_cmp(a));
    let mut acc = 0.0;
    for (k, &sk) in s.iter().enumerate() {
        if acc + sk < 0.0 {
            return k as f64 + if sk < 0.0 { acc / -sk } else { 0.0 };
        }
        acc += sk;
    }
    3.0
}

/// A discrete 2-D map x ← step(x).
pub struct Attractor2Map {
    pub step: Box<dyn Fn(Vec2) -> Vec2>,
}

impl Attractor2Map {
    /// Iterates the map, discarding `burn_in` steps then keeping `n`.
    #[must_use]
    pub fn trajectory(&self, x0: Vec2, n: usize, burn_in: usize) -> Vec<Vec2> {
        let mut x = x0;
        let mut out = Vec::with_capacity(n);
        for i in 0..n + burn_in {
            x = (self.step)(x);
            if i >= burn_in {
                out.push(x);
            }
        }
        out
    }

    /// Histogram of `n` iterates over `bounds` (row-major).
    ///
    /// # Panics
    /// Panics on an empty grid or degenerate bounds.
    #[must_use]
    pub fn density_map(&self, x0: Vec2, n: usize, res: (usize, usize), bounds: &Rect) -> Vec<u32> {
        assert!(res.0 > 0 && res.1 > 0, "grid must be non-empty");
        let size = bounds.max - bounds.min;
        assert!(size.x > 0.0 && size.y > 0.0, "bounds must have positive area");
        let mut grid = vec![0u32; res.0 * res.1];
        for p in self.trajectory(x0, n, 100) {
            let ix = ((p.x - bounds.min.x) / size.x * res.0 as f64).floor();
            let iy = ((p.y - bounds.min.y) / size.y * res.1 as f64).floor();
            if ix >= 0.0 && iy >= 0.0 && (ix as usize) < res.0 && (iy as usize) < res.1 {
                grid[iy as usize * res.0 + ix as usize] += 1;
            }
        }
        grid
    }

    /// Largest Lyapunov exponent per iteration, by renormalized
    /// finite-difference perturbations.
    ///
    /// # Panics
    /// Panics unless `n >= 100`.
    #[must_use]
    pub fn lyapunov(&self, x0: Vec2, n: usize) -> f64 {
        assert!(n >= 100, "need enough iterations to average");
        let eps = 1e-8;
        let mut x = x0;
        for _ in 0..200 {
            x = (self.step)(x);
        }
        let mut v = Vec2::new(1.0, 0.0);
        let mut sum = 0.0;
        for _ in 0..n {
            let fx = (self.step)(x);
            let w = ((self.step)(x + v * eps) - fx) * (1.0 / eps);
            let norm = w.magnitude().max(1e-300);
            sum += norm.ln();
            v = w * (1.0 / norm);
            x = fx;
        }
        sum / n as f64
    }

    /// Bifurcation diagram: for each of `n_params` parameter values
    /// in `param_range`, the map from `f` is iterated `transient`
    /// times and the next `samples` x coordinates are recorded as
    /// (parameter, x) pairs.
    #[must_use]
    pub fn bifurcation_diagram(
        param_range: (f64, f64),
        n_params: usize,
        transient: usize,
        samples: usize,
        f: &dyn Fn(f64) -> Attractor2Map,
    ) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(n_params * samples);
        for i in 0..n_params {
            let p = param_range.0
                + (param_range.1 - param_range.0) * i as f64 / (n_params.max(2) - 1) as f64;
            let map = f(p);
            let mut x = Vec2::new(0.5, 0.5);
            for _ in 0..transient {
                x = (map.step)(x);
            }
            for _ in 0..samples {
                x = (map.step)(x);
                out.push((p, x.x));
            }
        }
        out
    }
}

/// Named systems: 3-D flows with customary parameters and time
/// steps, plus the classic 2-D chaotic maps.
pub mod presets {
    use super::{Attractor2Map, Attractor3};
    use crate::math::{Vec2, Vec3};

    fn flow(dt: f64, f: impl Fn(Vec3) -> Vec3 + 'static) -> Attractor3 {
        Attractor3 { derivs: Box::new(f), dt }
    }

    fn map(f: impl Fn(Vec2) -> Vec2 + 'static) -> Attractor2Map {
        Attractor2Map { step: Box::new(f) }
    }

    /// Lorenz 1963: ẋ = σ(y−x), ẏ = x(ρ−z) − y, ż = xy − βz.
    #[must_use]
    pub fn lorenz(sigma: f64, rho: f64, beta: f64) -> Attractor3 {
        flow(0.01, move |p| {
            Vec3::new(
                sigma * (p.y - p.x),
                p.x * (rho - p.z) - p.y,
                p.x * p.y - beta * p.z,
            )
        })
    }

    /// Rössler 1976: ẋ = −y−z, ẏ = x+ay, ż = b + z(x−c).
    #[must_use]
    pub fn rossler(a: f64, b: f64, c: f64) -> Attractor3 {
        flow(0.02, move |p| Vec3::new(-p.y - p.z, p.x + a * p.y, b + p.z * (p.x - c)))
    }

    /// Aizawa attractor (a sphere-wrapped scroll).
    #[must_use]
    pub fn aizawa() -> Attractor3 {
        let (a, b, c, d, e, f) = (0.95, 0.7, 0.6, 3.5, 0.25, 0.1);
        flow(0.01, move |p| {
            Vec3::new(
                (p.z - b) * p.x - d * p.y,
                d * p.x + (p.z - b) * p.y,
                c + a * p.z - p.z * p.z * p.z / 3.0
                    - (p.x * p.x + p.y * p.y) * (1.0 + e * p.z)
                    + f * p.z * p.x * p.x * p.x,
            )
        })
    }

    /// Thomas' cyclically symmetric attractor: ẋ = sin y − bx, ...
    #[must_use]
    pub fn thomas(b: f64) -> Attractor3 {
        flow(0.05, move |p| {
            Vec3::new(p.y.sin() - b * p.x, p.z.sin() - b * p.y, p.x.sin() - b * p.z)
        })
    }

    /// Chen 1999 (a = 35, b = 3, c = 28).
    #[must_use]
    pub fn chen() -> Attractor3 {
        let (a, b, c) = (35.0, 3.0, 28.0);
        flow(0.002, move |p| {
            Vec3::new(
                a * (p.y - p.x),
                (c - a) * p.x - p.x * p.z + c * p.y,
                p.x * p.y - b * p.z,
            )
        })
    }

    /// Lü 2002 (a = 36, b = 3, c = 20), bridging Lorenz and Chen.
    #[must_use]
    pub fn lu() -> Attractor3 {
        let (a, b, c) = (36.0, 3.0, 20.0);
        flow(0.002, move |p| {
            Vec3::new(a * (p.y - p.x), -p.x * p.z + c * p.y, p.x * p.y - b * p.z)
        })
    }

    /// Halvorsen's cyclic attractor: ẋ = −ax − 4y − 4z − y².
    #[must_use]
    pub fn halvorsen(a: f64) -> Attractor3 {
        flow(0.005, move |p| {
            Vec3::new(
                -a * p.x - 4.0 * p.y - 4.0 * p.z - p.y * p.y,
                -a * p.y - 4.0 * p.z - 4.0 * p.x - p.z * p.z,
                -a * p.z - 4.0 * p.x - 4.0 * p.y - p.x * p.x,
            )
        })
    }

    /// Sprott case B: ẋ = yz, ẏ = x − y, ż = 1 − xy.
    #[must_use]
    pub fn sprott_b() -> Attractor3 {
        flow(0.01, |p| Vec3::new(p.y * p.z, p.x - p.y, 1.0 - p.x * p.y))
    }

    /// Dadras attractor.
    #[must_use]
    pub fn dadras() -> Attractor3 {
        let (a, b, c, d, e) = (3.0, 2.7, 1.7, 2.0, 9.0);
        flow(0.005, move |p| {
            Vec3::new(
                p.y - a * p.x + b * p.y * p.z,
                c * p.y - p.x * p.z + p.z,
                d * p.x * p.y - e * p.z,
            )
        })
    }

    /// Rabinovich-Fabrikant: ẋ = y(z − 1 + x²) + γx, ...
    #[must_use]
    pub fn rabinovich_fabrikant(a: f64, g: f64) -> Attractor3 {
        flow(0.002, move |p| {
            Vec3::new(
                p.y * (p.z - 1.0 + p.x * p.x) + g * p.x,
                p.x * (3.0 * p.z + 1.0 - p.x * p.x) + g * p.y,
                -2.0 * p.z * (a + p.x * p.y),
            )
        })
    }

    /// Three-scroll unified chaotic system (TSUCS-1).
    #[must_use]
    pub fn three_scroll() -> Attractor3 {
        let (a, b, c, d, e, f) = (32.48, 45.84, 1.18, 0.13, 0.57, 14.7);
        flow(0.001, move |p| {
            Vec3::new(
                a * (p.y - p.x) + d * p.x * p.z,
                b * p.x - p.x * p.z + f * p.y,
                c * p.z + p.x * p.y - e * p.x * p.x,
            )
        })
    }

    /// Arneodo-Coullet: ẋ = y, ẏ = z, ż = ax − by − z − x³
    /// with (a, b) = (5.5, 3.5).
    #[must_use]
    pub fn arneodo() -> Attractor3 {
        flow(0.01, |p| {
            Vec3::new(p.y, p.z, 5.5 * p.x - 3.5 * p.y - p.z - p.x * p.x * p.x)
        })
    }

    /// Nosé-Hoover oscillator (Sprott A): ẋ = y, ẏ = −x + yz,
    /// ż = 1 − y².
    #[must_use]
    pub fn nose_hoover() -> Attractor3 {
        flow(0.01, |p| Vec3::new(p.y, -p.x + p.y * p.z, 1.0 - p.y * p.y))
    }

    /// Four-wing attractor.
    #[must_use]
    pub fn four_wing() -> Attractor3 {
        let (a, b, c) = (0.2, 0.01, -0.4);
        flow(0.02, move |p| {
            Vec3::new(
                a * p.x + p.y * p.z,
                b * p.x + c * p.y - p.x * p.z,
                -p.z - p.x * p.y,
            )
        })
    }

    /// Chua's circuit with the piecewise-linear diode
    /// characteristic f(x) = m₁x + (m₀−m₁)(|x+1| − |x−1|)/2.
    #[must_use]
    pub fn chua(alpha: f64, beta: f64, m0: f64, m1: f64) -> Attractor3 {
        flow(0.005, move |p| {
            let fx = m1 * p.x + 0.5 * (m0 - m1) * ((p.x + 1.0).abs() - (p.x - 1.0).abs());
            Vec3::new(alpha * (p.y - p.x - fx), p.x - p.y + p.z, -beta * p.y)
        })
    }

    /// Rikitake two-disc dynamo (μ = 1, a = 5).
    #[must_use]
    pub fn rikitake() -> Attractor3 {
        let (mu, a) = (1.0, 5.0);
        flow(0.005, move |p| {
            Vec3::new(-mu * p.x + p.z * p.y, -mu * p.y + p.x * (p.z - a), 1.0 - p.x * p.y)
        })
    }

    /// Forced Duffing oscillator as an autonomous 3-D flow with
    /// z = ωt: ẋ = y, ẏ = −δy − αx − βx³ + γ cos z, ż = ω.
    #[must_use]
    pub fn duffing_forced(delta: f64, alpha: f64, beta: f64, gamma: f64, omega: f64) -> Attractor3 {
        flow(0.01, move |p| {
            Vec3::new(
                p.y,
                -delta * p.y - alpha * p.x - beta * p.x * p.x * p.x + gamma * p.z.cos(),
                omega,
            )
        })
    }

    /// Clifford attractor: x' = sin(ay) + c cos(ax), ...
    #[must_use]
    pub fn clifford(a: f64, b: f64, c: f64, d: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new(
                (a * p.y).sin() + c * (a * p.x).cos(),
                (b * p.x).sin() + d * (b * p.y).cos(),
            )
        })
    }

    /// Peter de Jong attractor.
    #[must_use]
    pub fn de_jong(a: f64, b: f64, c: f64, d: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new((a * p.y).sin() - (b * p.x).cos(), (c * p.x).sin() - (d * p.y).cos())
        })
    }

    /// Ikeda map with t = 0.4 − 6/(1 + x² + y²).
    #[must_use]
    pub fn ikeda(u: f64) -> Attractor2Map {
        map(move |p| {
            let t = 0.4 - 6.0 / (1.0 + p.x * p.x + p.y * p.y);
            let (s, c) = t.sin_cos();
            Vec2::new(1.0 + u * (p.x * c - p.y * s), u * (p.x * s + p.y * c))
        })
    }

    /// Tinkerbell map.
    #[must_use]
    pub fn tinkerbell(a: f64, b: f64, c: f64, d: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new(
                p.x * p.x - p.y * p.y + a * p.x + b * p.y,
                2.0 * p.x * p.y + c * p.x + d * p.y,
            )
        })
    }

    /// Gingerbreadman map: x' = 1 − y + |x|, y' = x.
    #[must_use]
    pub fn gingerbreadman() -> Attractor2Map {
        map(|p| Vec2::new(1.0 - p.y + p.x.abs(), p.x))
    }

    /// Hénon map: x' = 1 − ax² + y, y' = bx.
    #[must_use]
    pub fn henon(a: f64, b: f64) -> Attractor2Map {
        map(move |p| Vec2::new(1.0 - a * p.x * p.x + p.y, b * p.x))
    }

    /// Duffing map: x' = y, y' = −bx + ay − y³.
    #[must_use]
    pub fn duffing_map(a: f64, b: f64) -> Attractor2Map {
        map(move |p| Vec2::new(p.y, -b * p.x + a * p.y - p.y * p.y * p.y))
    }

    /// Bogdanov map.
    #[must_use]
    pub fn bogdanov(eps: f64, k: f64, mu: f64) -> Attractor2Map {
        map(move |p| {
            let y = p.y * (1.0 + eps) + k * p.x * (p.x - 1.0) + mu * p.x * p.y;
            Vec2::new(p.x + y, y)
        })
    }

    /// Chirikov standard map on the torus [0, 2π)²:
    /// p' = p + k sin θ, θ' = θ + p'. Area-preserving.
    #[must_use]
    pub fn standard_map(k: f64) -> Attractor2Map {
        let tau = std::f64::consts::TAU;
        map(move |q| {
            let p = (q.y + k * q.x.sin()).rem_euclid(tau);
            Vec2::new((q.x + p).rem_euclid(tau), p)
        })
    }

    /// Gumowski-Mira map with g(x) = ax + 2(1−a)x²/(1+x²).
    #[must_use]
    pub fn gumowski_mira(a: f64, b: f64) -> Attractor2Map {
        let g = move |x: f64| a * x + 2.0 * (1.0 - a) * x * x / (1.0 + x * x);
        map(move |p| {
            let x1 = b * p.y + g(p.x);
            Vec2::new(x1, g(x1) - p.x)
        })
    }

    /// Barry Martin's hopalong: x' = y − sign(x)√|bx − c|, y' = a − x.
    #[must_use]
    pub fn hopalong(a: f64, b: f64, c: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new(p.y - p.x.signum() * (b * p.x - c).abs().sqrt(), a - p.x)
        })
    }

    /// Bedhead attractor.
    #[must_use]
    pub fn bedhead(a: f64, b: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new(
                (p.x * p.y / b).sin() * p.y + (a * p.x - p.y).cos(),
                p.x + p.y.sin() / b,
            )
        })
    }

    /// Johnny Svensson's attractor.
    #[must_use]
    pub fn svensson(a: f64, b: f64, c: f64, d: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new(
                d * (a * p.x).sin() - (b * p.y).sin(),
                c * (a * p.x).cos() + (b * p.y).cos(),
            )
        })
    }

    /// "Fractal dream" attractor.
    #[must_use]
    pub fn fractal_dream(a: f64, b: f64, c: f64, d: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new((p.y * b).sin() + c * (p.x * b).sin(), (p.x * a).sin() + d * (p.y * a).sin())
        })
    }

    /// Popcorn map: x' = x − h sin(y + tan 3y), ...
    #[must_use]
    pub fn popcorn(h: f64) -> Attractor2Map {
        map(move |p| {
            Vec2::new(
                p.x - h * (p.y + (3.0 * p.y).tan()).sin(),
                p.y - h * (p.x + (3.0 * p.x).tan()).sin(),
            )
        })
    }

    /// Arnold's cat map on the unit torus.
    #[must_use]
    pub fn arnold_cat() -> Attractor2Map {
        map(|p| Vec2::new((2.0 * p.x + p.y).rem_euclid(1.0), (p.x + p.y).rem_euclid(1.0)))
    }

    /// Baker's map on the unit square.
    #[must_use]
    pub fn baker() -> Attractor2Map {
        map(|p| {
            if p.x < 0.5 {
                Vec2::new(2.0 * p.x, 0.5 * p.y)
            } else {
                Vec2::new(2.0 * p.x - 1.0, 0.5 * p.y + 0.5)
            }
        })
    }

    /// Zaslavskii map (ε forcing, ν rotation, r damping).
    #[must_use]
    pub fn zaslavskii(eps: f64, nu: f64, r: f64) -> Attractor2Map {
        let tau = std::f64::consts::TAU;
        let mu = (1.0 - (-r).exp()) / r;
        map(move |p| {
            let y = (tau * p.x).cos() + (-r).exp() * p.y;
            Vec2::new(
                (p.x + nu * (1.0 + mu * y) + eps * nu * mu * (tau * p.x).cos()).rem_euclid(1.0),
                y,
            )
        })
    }
}

/// Grassberger-Procaccia correlation dimension: the least-squares
/// slope of ln C(r) against ln r over `n_r` log-spaced radii, where
/// C(r) is the fraction of point pairs closer than r.
///
/// # Panics
/// Panics unless there are >= 100 points, `0 < r_min < r_max`, and
/// `n_r >= 2`.
#[must_use]
pub fn correlation_dimension(points: &[Vec3], r_min: f64, r_max: f64, n_r: usize) -> f64 {
    assert!(points.len() >= 100, "need at least 100 points");
    assert!(r_min > 0.0 && r_max > r_min, "need 0 < r_min < r_max");
    assert!(n_r >= 2, "need at least two radii");
    let radii: Vec<f64> = (0..n_r)
        .map(|i| r_min * (r_max / r_min).powf(i as f64 / (n_r - 1) as f64))
        .collect();
    let mut counts = vec![0u64; n_r];
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            let d = points[i].distance_to(&points[j]);
            // Radii are sorted: count into the smallest bin >= d.
            for (k, &r) in radii.iter().enumerate() {
                if d < r {
                    counts[k] += 1;
                }
            }
        }
    }
    let total = (points.len() * (points.len() - 1) / 2) as f64;
    // Least-squares fit over bins with non-zero counts.
    let pts: Vec<(f64, f64)> = radii
        .iter()
        .zip(&counts)
        .filter(|&(_, &c)| c > 0)
        .map(|(&r, &c)| (r.ln(), (c as f64 / total).ln()))
        .collect();
    fit_slope(&pts)
}

fn fit_slope(pts: &[(f64, f64)]) -> f64 {
    assert!(pts.len() >= 2, "need at least two samples to fit a slope");
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

/// Box-counting dimension of a 3-D point set: slope of ln N(s)
/// versus ln(1/s) over the given box sizes.
///
/// # Panics
/// Panics unless points and at least two positive scales are given.
#[must_use]
pub fn box_counting_dimension_3d(points: &[Vec3], scales: &[f64]) -> f64 {
    assert!(!points.is_empty(), "need points");
    assert!(scales.len() >= 2 && scales.iter().all(|&s| s > 0.0), "need >= 2 positive scales");
    let mut fit = Vec::with_capacity(scales.len());
    for &s in scales {
        let mut boxes = std::collections::HashSet::new();
        for p in points {
            boxes.insert((
                (p.x / s).floor() as i64,
                (p.y / s).floor() as i64,
                (p.z / s).floor() as i64,
            ));
        }
        fit.push(((1.0 / s).ln(), (boxes.len() as f64).ln()));
    }
    fit_slope(&fit)
}

/// Time-delay embedding: vectors [x(i), x(i+τ), ..., x(i+(m−1)τ)].
///
/// # Panics
/// Panics unless the series is long enough for one vector.
#[must_use]
pub fn delay_embedding(series: &[f64], dim: usize, delay: usize) -> Vec<Vec<f64>> {
    assert!(dim >= 1 && delay >= 1, "dimension and delay must be positive");
    let span = (dim - 1) * delay;
    assert!(series.len() > span, "series too short for this embedding");
    (0..series.len() - span)
        .map(|i| (0..dim).map(|k| series[i + k * delay]).collect())
        .collect()
}

/// Recurrence plot: `R[i,j]` is true when embedded states i and j are
/// within `eps` (row-major over the n embedded points).
#[must_use]
pub fn recurrence_plot(series: &[f64], embed_dim: usize, delay: usize, eps: f64) -> Vec<bool> {
    let embedded = delay_embedding(series, embed_dim, delay);
    let n = embedded.len();
    let mut out = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            let d2: f64 = embedded[i]
                .iter()
                .zip(&embedded[j])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            out[i * n + j] = d2.sqrt() <= eps;
        }
    }
    out
}

/// Largest Lyapunov exponent from a scalar series by Rosenstein's
/// method: embed, pair each point with its nearest neighbor at
/// temporal distance > `mean_period`, and fit the slope of the mean
/// log divergence over `max_iter` steps. Returned per sample step.
///
/// # Panics
/// Panics on a series too short for the embedding and tracking.
#[must_use]
pub fn largest_lyapunov_rosenstein(
    series: &[f64],
    embed_dim: usize,
    delay: usize,
    mean_period: usize,
    max_iter: usize,
) -> f64 {
    let embedded = delay_embedding(series, embed_dim, delay);
    let n = embedded.len();
    assert!(n > max_iter + mean_period + 1, "series too short");
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    };
    // Nearest neighbors outside the Theiler window.
    let mut neighbor = vec![usize::MAX; n];
    for i in 0..n - max_iter {
        let mut best = f64::INFINITY;
        for j in 0..n - max_iter {
            if i.abs_diff(j) <= mean_period {
                continue;
            }
            let d = dist(&embedded[i], &embedded[j]);
            if d < best && d > 0.0 {
                best = d;
                neighbor[i] = j;
            }
        }
    }
    let mut mean_log = Vec::with_capacity(max_iter);
    for k in 1..=max_iter {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..n - max_iter {
            let j = neighbor[i];
            if j == usize::MAX || i + k >= n || j + k >= n {
                continue;
            }
            let d = dist(&embedded[i + k], &embedded[j + k]);
            if d > 0.0 {
                sum += d.ln();
                count += 1;
            }
        }
        if count > 0 {
            mean_log.push((k as f64, sum / count as f64));
        }
    }
    // Fit the initial (approximately linear) growth region: the
    // first half of the curve before saturation.
    let cut = (mean_log.len() / 2).max(2);
    fit_slope(&mean_log[..cut])
}

/// Feigenbaum δ estimate from successive bifurcation parameters:
/// δₙ = (bₙ₋₁ − bₙ₋₂)/(bₙ − bₙ₋₁) for the last triple.
///
/// # Panics
/// Panics unless at least 3 bifurcation points are given.
#[must_use]
pub fn feigenbaum_estimate(bifurcations: &[f64]) -> f64 {
    assert!(bifurcations.len() >= 3, "need at least three bifurcation points");
    let n = bifurcations.len();
    (bifurcations[n - 2] - bifurcations[n - 3]) / (bifurcations[n - 1] - bifurcations[n - 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Vec2, Vec3};

    #[test]
    fn test_lorenz_lyapunov_spectrum_and_dimension() {
        let lorenz = presets::lorenz(10.0, 28.0, 8.0 / 3.0);
        let spectrum = lorenz.lyapunov_spectrum(Vec3::new(1.0, 1.0, 1.0), 60_000, 0.005);
        // Literature: (0.9056, 0, -14.572).
        assert!(
            (spectrum[0] - 0.9056).abs() < 0.05 * 0.9056 + 0.02,
            "largest exponent {} vs 0.9056",
            spectrum[0]
        );
        assert!(spectrum[1].abs() < 0.05, "middle exponent ~0 ({})", spectrum[1]);
        // The sum equals the divergence -(sigma + 1 + beta).
        let sum: f64 = spectrum.iter().sum();
        assert!((sum + 10.0 + 1.0 + 8.0 / 3.0).abs() < 0.4, "trace check ({sum})");
        let dky = kaplan_yorke_dimension(spectrum);
        assert!((dky - 2.06).abs() < 0.05, "Kaplan-Yorke dimension {dky}");
    }

    #[test]
    fn test_trajectory_integrators() {
        // Circular test flow: ẋ = -y, ẏ = x, ż = 0 preserves radius.
        let circle = Attractor3 {
            derivs: Box::new(|p: Vec3| Vec3::new(-p.y, p.x, 0.0)),
            dt: 0.05,
        };
        let start = Vec3::new(1.0, 0.0, 0.0);
        for (method, tol) in
            [(Integrator::Euler, 0.3), (Integrator::Rk4, 1e-6), (Integrator::Rk45, 1e-8)]
        {
            let traj = circle.trajectory(start, 200, method);
            assert_eq!(traj.len(), 201);
            let r = traj.last().unwrap().magnitude();
            assert!((r - 1.0).abs() < tol, "{method:?} radius drift {}", (r - 1.0).abs());
        }
    }

    #[test]
    fn test_poincare_and_density() {
        let lorenz = presets::lorenz(10.0, 28.0, 8.0 / 3.0);
        // Section through z = 27 (the classic Lorenz section).
        let plane = Plane { normal: Vec3::new(0.0, 0.0, 1.0), d: -27.0 };
        let pts = lorenz.poincare_section(Vec3::new(1.0, 1.0, 20.0), 40_000, &plane);
        assert!(pts.len() > 50, "many crossings ({})", pts.len());
        for p in &pts {
            assert!((p.z - 27.0).abs() < 0.5, "crossings near the plane");
        }
        let bounds = Rect { min: Vec2::new(-25.0, -30.0), max: Vec2::new(25.0, 30.0) };
        let grid = lorenz.density_map(Vec3::new(1.0, 1.0, 20.0), 20_000, (32, 32), &bounds);
        let hits: u32 = grid.iter().sum();
        assert!(hits > 15_000, "trajectory stays in the window ({hits})");
    }

    #[test]
    fn test_henon_lyapunov_and_correlation_dimension() {
        let henon = presets::henon(1.4, 0.3);
        let l = henon.lyapunov(Vec2::new(0.1, 0.1), 30_000);
        assert!((l - 0.419).abs() < 0.02, "Henon largest exponent {l}");
        let pts2 = henon.trajectory(Vec2::new(0.1, 0.1), 3000, 200);
        let pts: Vec<Vec3> = pts2.iter().map(|p| Vec3::new(p.x, p.y, 0.0)).collect();
        let d = correlation_dimension(&pts, 0.005, 0.08, 10);
        assert!((d - 1.21).abs() < 0.05, "Henon correlation dimension {d}");
    }

    #[test]
    fn test_standard_map_area_preserving() {
        let k = 0.971;
        let map = presets::standard_map(k);
        let h = 1e-6;
        let mut rng_state = 12345u64;
        let mut rand = move || {
            rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (rng_state >> 33) as f64 / (1u64 << 31) as f64
        };
        for _ in 0..50 {
            let p = Vec2::new(rand() * 6.0 + 0.1, rand() * 6.0 + 0.1);
            // Numeric Jacobian determinant (away from the torus seam).
            let f0 = (map.step)(p);
            let fx = (map.step)(p + Vec2::new(h, 0.0));
            let fy = (map.step)(p + Vec2::new(0.0, h));
            let j00 = (fx.x - f0.x) / h;
            let j01 = (fy.x - f0.x) / h;
            let j10 = (fx.y - f0.y) / h;
            let j11 = (fy.y - f0.y) / h;
            let det = (j00 * j11 - j01 * j10).abs();
            // Skip samples that straddled the mod-2pi seam.
            if !(0.2..=5.0).contains(&det) {
                continue;
            }
            assert!((det - 1.0).abs() < 1e-4, "standard map det {det}");
        }
    }

    #[test]
    fn test_logistic_feigenbaum() {
        // Period-doubling parameters of the logistic map found by
        // bisection on the detected attractor period.
        let period_at = |r: f64| -> usize {
            let mut x = 0.5f64;
            for _ in 0..200_000 {
                x = r * x * (1.0 - x);
            }
            let anchor = x;
            let mut p = 0;
            for k in 1..=64 {
                x = r * x * (1.0 - x);
                if (x - anchor).abs() < 1e-10 {
                    p = k;
                    break;
                }
            }
            p
        };
        let mut bifurcations = Vec::new();
        let mut lo = 2.9;
        for target in [2usize, 4, 8, 16] {
            let mut a = lo;
            let mut b = 3.6;
            // Find where the period first reaches `target`.
            for _ in 0..48 {
                let mid = 0.5 * (a + b);
                let p = period_at(mid);
                if p >= target || p == 0 {
                    b = mid;
                } else {
                    a = mid;
                }
            }
            bifurcations.push(0.5 * (a + b));
            lo = 0.5 * (a + b);
        }
        // Known doubling points: 3, 3.44949, 3.54409, 3.56441.
        assert!((bifurcations[0] - 3.0).abs() < 1e-3);
        assert!((bifurcations[1] - 3.449_49).abs() < 1e-3);
        let delta = feigenbaum_estimate(&bifurcations);
        assert!((delta - 4.6692).abs() < 0.05, "Feigenbaum delta {delta}");
        // The bifurcation diagram shows one branch before r = 3 and
        // several after.
        let diagram = Attractor2Map::bifurcation_diagram((2.8, 3.5), 8, 3000, 40, &|r| {
            Attractor2Map { step: Box::new(move |p: Vec2| Vec2::new(r * p.x * (1.0 - p.x), 0.0)) }
        });
        assert_eq!(diagram.len(), 8 * 40);
        // Per-parameter spread: a single point below r = 3, several
        // branches by r = 3.5.
        let spread_at = |r0: f64| -> f64 {
            let xs: Vec<f64> = diagram
                .iter()
                .filter(|&&(r, _)| (r - r0).abs() < 1e-9)
                .map(|&(_, x)| x)
                .collect();
            assert!(!xs.is_empty(), "diagram sampled r = {r0}");
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - xs.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        assert!(spread_at(2.8) < 1e-6, "fixed point at r = 2.8");
        assert!(spread_at(2.9) < 1e-6, "fixed point at r = 2.9");
        assert!(spread_at(3.5) > 0.2, "periodic branches at r = 3.5");
    }

    #[test]
    fn test_flow_presets_stay_bounded() {
        let cases: Vec<(&str, Attractor3, Vec3, f64)> = vec![
            ("lorenz", presets::lorenz(10.0, 28.0, 8.0 / 3.0), Vec3::new(1.0, 1.0, 1.0), 100.0),
            ("rossler", presets::rossler(0.2, 0.2, 5.7), Vec3::new(1.0, 1.0, 1.0), 100.0),
            ("aizawa", presets::aizawa(), Vec3::new(0.1, 0.0, 0.0), 10.0),
            ("thomas", presets::thomas(0.208_186), Vec3::new(1.0, 0.0, 0.0), 20.0),
            ("chen", presets::chen(), Vec3::new(-0.1, 0.5, -0.6), 200.0),
            ("lu", presets::lu(), Vec3::new(0.1, 0.3, -0.6), 200.0),
            ("halvorsen", presets::halvorsen(1.89), Vec3::new(-1.48, -1.51, 2.04), 50.0),
            ("sprott_b", presets::sprott_b(), Vec3::new(0.05, 0.05, 0.05), 50.0),
            ("dadras", presets::dadras(), Vec3::new(1.1, 2.1, -2.0), 100.0),
            (
                "rabinovich",
                presets::rabinovich_fabrikant(1.1, 0.87),
                Vec3::new(-1.0, 0.0, 0.5),
                50.0,
            ),
            ("three_scroll", presets::three_scroll(), Vec3::new(-0.29, -0.25, -0.59), 500.0),
            ("arneodo", presets::arneodo(), Vec3::new(0.1, 0.0, 0.0), 50.0),
            ("nose_hoover", presets::nose_hoover(), Vec3::new(0.1, 0.0, 0.0), 50.0),
            ("four_wing", presets::four_wing(), Vec3::new(1.3, -0.18, 0.01), 20.0),
            ("chua", presets::chua(15.6, 28.0, -1.143, -0.714), Vec3::new(0.7, 0.0, 0.0), 30.0),
            ("rikitake", presets::rikitake(), Vec3::new(-1.4, -3.0, 4.4), 50.0),
            (
                "duffing",
                presets::duffing_forced(0.3, -1.0, 1.0, 0.5, 1.2),
                Vec3::new(0.5, 0.0, 0.0),
                1e4,
            ),
        ];
        for (name, a, x0, bound) in cases {
            let traj = a.trajectory(x0, 5000, Integrator::Rk4);
            let last = traj.last().unwrap();
            assert!(
                last.magnitude().is_finite() && last.magnitude() < bound,
                "{name} stayed bounded ({})",
                last.magnitude()
            );
        }
    }

    #[test]
    fn test_map_presets_stay_bounded() {
        let cases: Vec<(&str, Attractor2Map, Vec2, f64)> = vec![
            ("clifford", presets::clifford(-1.4, 1.6, 1.0, 0.7), Vec2::new(0.1, 0.1), 4.0),
            ("de_jong", presets::de_jong(-2.0, -2.0, -1.2, 2.0), Vec2::new(0.1, 0.1), 4.0),
            ("ikeda", presets::ikeda(0.918), Vec2::new(0.1, 0.1), 10.0),
            ("tinkerbell", presets::tinkerbell(0.9, -0.6013, 2.0, 0.5), Vec2::new(-0.72, -0.64), 4.0),
            ("gingerbreadman", presets::gingerbreadman(), Vec2::new(0.5, 3.7), 20.0),
            ("henon", presets::henon(1.4, 0.3), Vec2::new(0.1, 0.1), 3.0),
            ("duffing_map", presets::duffing_map(2.75, 0.2), Vec2::new(0.1, 0.1), 5.0),
            ("bogdanov", presets::bogdanov(0.0, 1.2, 0.0), Vec2::new(0.1, 0.1), 10.0),
            ("standard", presets::standard_map(0.971), Vec2::new(0.5, 0.5), 10.0),
            ("gumowski_mira", presets::gumowski_mira(-0.192, 0.982), Vec2::new(0.1, 0.1), 30.0),
            ("hopalong", presets::hopalong(0.4, 1.0, 0.0), Vec2::new(0.1, 0.1), 50.0),
            ("bedhead", presets::bedhead(-0.81, -0.92), Vec2::new(1.0, 1.0), 10.0),
            ("svensson", presets::svensson(1.4, 1.56, 1.4, -6.56), Vec2::new(0.1, 0.1), 10.0),
            ("dream", presets::fractal_dream(-0.966, 2.879, 0.765, 0.744), Vec2::new(0.1, 0.1), 5.0),
            ("popcorn", presets::popcorn(0.05), Vec2::new(0.1, 0.1), 10.0),
            ("arnold_cat", presets::arnold_cat(), Vec2::new(0.3, 0.7), 1.5),
            ("baker", presets::baker(), Vec2::new(0.3, 0.7), 1.5),
            ("zaslavskii", presets::zaslavskii(0.3, 400.0 / 3.0, 3.0), Vec2::new(0.1, 0.1), 10.0),
        ];
        for (name, m, x0, bound) in cases {
            let traj = m.trajectory(x0, 3000, 100);
            for p in traj.iter().step_by(97) {
                assert!(
                    p.x.is_finite() && p.y.is_finite() && p.magnitude() < bound,
                    "{name} bounded ({p:?})"
                );
            }
        }
        // Density map covers multiple cells.
        let bounds = Rect { min: Vec2::new(-2.5, -2.5), max: Vec2::new(2.5, 2.5) };
        let grid = presets::clifford(-1.4, 1.6, 1.0, 0.7).density_map(
            Vec2::new(0.1, 0.1),
            20_000,
            (24, 24),
            &bounds,
        );
        assert!(grid.iter().filter(|&&c| c > 0).count() > 50);
    }

    #[test]
    fn test_embedding_recurrence_rosenstein() {
        // A clean sine series: recurrence plot has a periodic
        // diagonal structure and near-zero largest exponent.
        let series: Vec<f64> = (0..600).map(|i| (i as f64 * 0.1).sin()).collect();
        let emb = delay_embedding(&series, 3, 5);
        assert_eq!(emb.len(), 600 - 10);
        assert_eq!(emb[0].len(), 3);
        let rp = recurrence_plot(&series[..200], 2, 5, 0.1);
        let n = 200 - 5;
        assert_eq!(rp.len(), n * n);
        assert!(rp[0], "diagonal is recurrent");
        let density = rp.iter().filter(|&&b| b).count() as f64 / (n * n) as f64;
        assert!(density > 0.01 && density < 0.5, "sparse but non-empty ({density})");
        let lam_sine = largest_lyapunov_rosenstein(&series, 3, 5, 62, 40);
        assert!(lam_sine.abs() < 0.05, "periodic signal has ~0 exponent ({lam_sine})");
        // Chaotic logistic series has a clearly positive exponent.
        let mut x = 0.4;
        let logistic: Vec<f64> = (0..800)
            .map(|_| {
                x = 4.0 * x * (1.0 - x);
                x
            })
            .collect();
        let lam = largest_lyapunov_rosenstein(&logistic, 2, 1, 1, 8);
        assert!(lam > 0.3, "chaotic series diverges ({lam})");
        // Box-counting sanity: points on a line have dimension ~1.
        let line: Vec<Vec3> = (0..2000)
            .map(|i| Vec3::new(i as f64 / 2000.0, 0.0, 0.0))
            .collect();
        let d = box_counting_dimension_3d(&line, &[0.05, 0.025, 0.0125, 0.00625]);
        assert!((d - 1.0).abs() < 0.1, "line dimension {d}");
    }
}
