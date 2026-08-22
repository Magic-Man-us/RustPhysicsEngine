//! Symplectic integrators for second-order systems x'' = a(x).
//!
//! These preserve phase-space volume, so energy errors stay bounded
//! instead of drifting. References: Verlet (1967); Yoshida, "Construction
//! of higher order symplectic integrators", Phys. Lett. A 150 (1990).

/// One velocity-Verlet step (kick-drift-kick):
/// v½ = v + a(x)·dt/2; x₁ = x + v½·dt; v₁ = v½ + a(x₁)·dt/2.
///
/// # Panics
/// Panics if `x` and `v` differ in length or `acc` returns the wrong
/// length.
pub fn velocity_verlet(acc: &dyn Fn(&[f64]) -> Vec<f64>, x: &mut [f64], v: &mut [f64], dt: f64) {
    assert!(x.len() == v.len(), "velocity_verlet requires x.len() == v.len()");
    let n = x.len();
    let a0 = acc(x);
    assert!(a0.len() == n, "acc must return one value per coordinate");
    for i in 0..n {
        v[i] += 0.5 * dt * a0[i];
        x[i] += dt * v[i];
    }
    let a1 = acc(x);
    assert!(a1.len() == n, "acc must return one value per coordinate");
    for i in 0..n {
        v[i] += 0.5 * dt * a1[i];
    }
}

/// Alias of the kick-drift-kick leapfrog scheme (identical to velocity
/// Verlet in this synchronized form).
pub fn leapfrog_kick_drift_kick(
    acc: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &mut [f64],
    v: &mut [f64],
    dt: f64,
) {
    velocity_verlet(acc, x, v, dt);
}

/// One 4th-order Yoshida step: composition of three velocity-Verlet
/// sub-steps with weights w1, w0, w1 where
/// w1 = 1/(2 − 2^(1/3)), w0 = −2^(1/3)·w1.
pub fn yoshida4(acc: &dyn Fn(&[f64]) -> Vec<f64>, x: &mut [f64], v: &mut [f64], dt: f64) {
    let cbrt2 = 2.0_f64.powf(1.0 / 3.0);
    let w1 = 1.0 / (2.0 - cbrt2);
    let w0 = -cbrt2 * w1;
    velocity_verlet(acc, x, v, w1 * dt);
    velocity_verlet(acc, x, v, w0 * dt);
    velocity_verlet(acc, x, v, w1 * dt);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oscillator_energy(x: &[f64], v: &[f64]) -> f64 {
        0.5 * (v[0] * v[0] + x[0] * x[0])
    }

    #[test]
    fn test_verlet_harmonic_energy_bounded() {
        let acc = |x: &[f64]| vec![-x[0]];
        let mut x = vec![1.0];
        let mut v = vec![0.0];
        let e0 = oscillator_energy(&x, &v);
        let dt = 0.05;
        let mut max_err = 0.0_f64;
        for _ in 0..100_000 {
            velocity_verlet(&acc, &mut x, &mut v, dt);
            max_err = max_err.max((oscillator_energy(&x, &v) - e0).abs());
        }
        // Bounded oscillation, no secular drift: error stays O(dt^2).
        assert!(max_err < 1e-3, "energy error {max_err}");
    }

    #[test]
    fn test_yoshida4_more_accurate_than_verlet() {
        let acc = |x: &[f64]| vec![-x[0]];
        let dt = 0.1;
        let steps = 1000;

        let mut xv = (vec![1.0], vec![0.0]);
        let mut xy = (vec![1.0], vec![0.0]);
        for _ in 0..steps {
            velocity_verlet(&acc, &mut xv.0, &mut xv.1, dt);
            yoshida4(&acc, &mut xy.0, &mut xy.1, dt);
        }
        let t = steps as f64 * dt;
        let exact = t.cos();
        let err_verlet = (xv.0[0] - exact).abs();
        let err_yoshida = (xy.0[0] - exact).abs();
        assert!(err_yoshida < err_verlet / 10.0, "verlet {err_verlet}, yoshida {err_yoshida}");
    }

    #[test]
    fn test_yoshida4_energy_bounded_long_run() {
        let acc = |x: &[f64]| vec![-x[0]];
        let mut x = vec![1.0];
        let mut v = vec![0.0];
        let e0 = oscillator_energy(&x, &v);
        for _ in 0..100_000 {
            yoshida4(&acc, &mut x, &mut v, 0.05);
        }
        assert!((oscillator_energy(&x, &v) - e0).abs() < 1e-6);
    }

    #[test]
    fn test_leapfrog_matches_verlet() {
        let acc = |x: &[f64]| vec![-x[0]];
        let mut a = (vec![1.0], vec![0.5]);
        let mut b = (vec![1.0], vec![0.5]);
        for _ in 0..100 {
            velocity_verlet(&acc, &mut a.0, &mut a.1, 0.03);
            leapfrog_kick_drift_kick(&acc, &mut b.0, &mut b.1, 0.03);
        }
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "x.len() == v.len()")]
    fn test_mismatched_lengths_panic() {
        let acc = |x: &[f64]| x.to_vec();
        velocity_verlet(&acc, &mut [1.0], &mut [1.0, 2.0], 0.1);
    }
}
