//! Explicit fixed-step ODE integrators.

/// Single forward Euler step: y_next = y + dt * f(t, y).
#[must_use]
pub fn euler_step(f: &dyn Fn(f64, f64) -> f64, t: f64, y: f64, dt: f64) -> f64 {
    y + dt * f(t, y)
}

/// Single step of the classic 4th-order Runge-Kutta method.
#[must_use]
pub fn rk4_step(f: &dyn Fn(f64, f64) -> f64, t: f64, y: f64, dt: f64) -> f64 {
    let k1 = f(t, y);
    let k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1);
    let k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2);
    let k4 = f(t + dt, y + dt * k3);
    y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

/// Full RK4 integration of dy/dt = f(t, y) from t0 to t_end, returning (t, y) pairs.
#[must_use]
pub fn rk4_solve(
    f: &dyn Fn(f64, f64) -> f64,
    t0: f64,
    y0: f64,
    t_end: f64,
    dt: f64,
) -> Vec<(f64, f64)> {
    let mut t = t0;
    let mut y = y0;
    let mut results = vec![(t, y)];
    while t < t_end - dt * 0.5 {
        y = rk4_step(f, t, y, dt);
        t += dt;
        results.push((t, y));
    }
    results
}

/// Single RK4 step for a system of ODEs (vector state).
/// `f(t, y)` returns a `Vec<f64>` of derivatives matching the length of `y`.
#[must_use]
pub fn rk4_step_vec(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    t: f64,
    y: &[f64],
    dt: f64,
) -> Vec<f64> {
    let n = y.len();
    let k1 = f(t, y);

    let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k1[i]).collect();
    let k2 = f(t + 0.5 * dt, &y2);

    let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k2[i]).collect();
    let k3 = f(t + 0.5 * dt, &y3);

    let y4: Vec<f64> = (0..n).map(|i| y[i] + dt * k3[i]).collect();
    let k4 = f(t + dt, &y4);

    (0..n)
        .map(|i| y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}
