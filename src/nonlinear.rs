// Chaos theory and nonlinear dynamics: logistic maps, Lyapunov exponents,
// strange attractors, fractal dimension estimators, and fixed-point analysis.

// ── Logistic Map ──────────────────────────────────────────────────────────────

#[must_use]
pub fn logistic_map(r: f64, x: f64) -> f64 {
    r * x * (1.0 - x)
}

#[must_use]
pub fn logistic_map_iterate(r: f64, x0: f64, n: usize) -> Vec<f64> {
    let mut trajectory = Vec::with_capacity(n + 1);
    let mut x = x0;
    trajectory.push(x);
    for _ in 0..n {
        x = logistic_map(r, x);
        trajectory.push(x);
    }
    trajectory
}

#[must_use]
pub fn logistic_map_converge(r: f64, x0: f64, transient: usize, samples: usize) -> Vec<f64> {
    let mut x = x0;
    for _ in 0..transient {
        x = logistic_map(r, x);
    }
    let mut result = Vec::with_capacity(samples);
    for _ in 0..samples {
        x = logistic_map(r, x);
        result.push(x);
    }
    result
}

// ── Lyapunov Exponent ─────────────────────────────────────────────────────────

#[must_use]
pub fn lyapunov_exponent_logistic(r: f64, x0: f64, n: usize) -> f64 {
    let mut x = x0;
    let mut sum = 0.0;
    for _ in 0..n {
        let derivative = r * (1.0 - 2.0 * x);
        let abs_d = derivative.abs();
        if abs_d == 0.0 {
            return f64::NEG_INFINITY;
        }
        sum += abs_d.ln();
        x = logistic_map(r, x);
    }
    sum / n as f64
}

#[must_use]
pub fn lyapunov_exponent_1d(
    f: &dyn Fn(f64) -> f64,
    df: &dyn Fn(f64) -> f64,
    x0: f64,
    n: usize,
) -> f64 {
    let mut x = x0;
    let mut sum = 0.0;
    for _ in 0..n {
        let abs_d = df(x).abs();
        if abs_d == 0.0 {
            return f64::NEG_INFINITY;
        }
        sum += abs_d.ln();
        x = f(x);
    }
    sum / n as f64
}

// ── Classic Attractors ────────────────────────────────────────────────────────

#[must_use]
pub fn lorenz_derivatives(state: &[f64; 3], sigma: f64, rho: f64, beta: f64) -> [f64; 3] {
    let [x, y, z] = *state;
    [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]
}

#[must_use]
pub fn rossler_derivatives(state: &[f64; 3], a: f64, b: f64, c: f64) -> [f64; 3] {
    let [x, y, z] = *state;
    [
        -y - z,
        x + a * y,
        b + z * (x - c),
    ]
}

#[must_use]
pub fn henon_map(x: f64, y: f64, a: f64, b: f64) -> (f64, f64) {
    (1.0 - a * x * x + y, b * x)
}

#[must_use]
pub fn henon_iterate(x0: f64, y0: f64, a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n + 1);
    let mut x = x0;
    let mut y = y0;
    points.push((x, y));
    for _ in 0..n {
        let (xn, yn) = henon_map(x, y, a, b);
        x = xn;
        y = yn;
        points.push((x, y));
    }
    points
}

// ── Fractal Dimensions ────────────────────────────────────────────────────────

#[must_use]
pub fn correlation_dimension_estimate(distances: &[f64], r: f64) -> f64 {
    let total_pairs = distances.len();
    if total_pairs == 0 {
        return 0.0;
    }
    let count = distances.iter().filter(|&&d| d < r).count();
    count as f64 / total_pairs as f64
}

#[must_use]
pub fn box_counting_dimension(occupied_boxes: &[(usize, usize)], grid_sizes: &[usize]) -> f64 {
    // For each grid size, count distinct occupied boxes at that resolution,
    // then do least-squares fit of log(N) vs log(1/epsilon).
    if grid_sizes.len() < 2 {
        return 0.0;
    }

    let mut log_inv_eps: Vec<f64> = Vec::with_capacity(grid_sizes.len());
    let mut log_n: Vec<f64> = Vec::with_capacity(grid_sizes.len());

    for &size in grid_sizes {
        if size == 0 {
            continue;
        }
        let mut boxes_at_scale: Vec<(usize, usize)> = occupied_boxes
            .iter()
            .map(|&(bx, by)| (bx / size, by / size))
            .collect();
        boxes_at_scale.sort_unstable();
        boxes_at_scale.dedup();

        let n = boxes_at_scale.len();
        if n == 0 {
            continue;
        }

        log_inv_eps.push((1.0 / size as f64).ln());
        log_n.push((n as f64).ln());
    }

    if log_inv_eps.len() < 2 {
        return 0.0;
    }

    linear_regression_slope(&log_inv_eps, &log_n)
}

fn linear_regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = xs.iter().map(|x| x * x).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }
    (n * sum_xy - sum_x * sum_y) / denom
}

// ── Fixed Points & Stability ──────────────────────────────────────────────────

#[must_use]
pub fn fixed_point_iterate(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let x_next = f(x);
        if (x_next - x).abs() < tol {
            return Some(x_next);
        }
        x = x_next;
    }
    None
}

#[must_use]
pub fn is_stable_fixed_point(df_at_fixed: f64) -> bool {
    df_at_fixed.abs() < 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn logistic_map_r2_converges_to_half() {
        let trajectory = logistic_map_iterate(2.0, 0.1, 100);
        let final_val = *trajectory.last().unwrap();
        assert!(approx(final_val, 0.5, TOL), "expected ~0.5, got {final_val}");
    }

    #[test]
    fn lyapunov_negative_for_stable_r2() {
        let lambda = lyapunov_exponent_logistic(2.0, 0.1, 10_000);
        assert!(lambda < 0.0, "expected negative Lyapunov for r=2.0, got {lambda}");
    }

    #[test]
    fn lyapunov_positive_for_chaotic_r4() {
        let lambda = lyapunov_exponent_logistic(4.0, 0.1, 10_000);
        assert!(lambda > 0.0, "expected positive Lyapunov for r=4.0, got {lambda}");
    }

    #[test]
    fn lorenz_at_origin_is_zero() {
        let d = lorenz_derivatives(&[0.0, 0.0, 0.0], 10.0, 28.0, 8.0 / 3.0);
        assert!(approx(d[0], 0.0, TOL));
        assert!(approx(d[1], 0.0, TOL));
        assert!(approx(d[2], 0.0, TOL));
    }

    #[test]
    fn henon_classic_does_not_diverge() {
        let points = henon_iterate(0.1, 0.1, 1.4, 0.3, 1000);
        for &(x, y) in &points {
            assert!(
                x.is_finite() && y.is_finite(),
                "Henon map diverged at ({x}, {y})"
            );
        }
    }

    #[test]
    fn fixed_point_of_cos() {
        let fp = fixed_point_iterate(&f64::cos, 1.0, 1e-10, 1000);
        let val = fp.expect("fixed_point_iterate should converge for cos");
        assert!(
            approx(val, 0.7390851332, 1e-4),
            "expected ~0.7391, got {val}"
        );
    }

    #[test]
    fn stable_vs_unstable_fixed_point() {
        assert!(is_stable_fixed_point(0.5));
        assert!(is_stable_fixed_point(-0.9));
        assert!(!is_stable_fixed_point(1.1));
        assert!(!is_stable_fixed_point(-1.5));
    }

    #[test]
    fn correlation_dimension_basic() {
        let distances = vec![0.1, 0.5, 0.9, 1.5, 2.0];
        let c = correlation_dimension_estimate(&distances, 1.0);
        // 3 out of 5 distances are < 1.0
        assert!(approx(c, 0.6, TOL));
    }

    #[test]
    fn logistic_map_converge_samples() {
        let samples = logistic_map_converge(2.0, 0.1, 200, 50);
        assert_eq!(samples.len(), 50);
        for &s in &samples {
            assert!(approx(s, 0.5, 1e-4), "expected ~0.5 after transient, got {s}");
        }
    }

    #[test]
    fn rossler_derivatives_at_origin() {
        let d = rossler_derivatives(&[0.0, 0.0, 0.0], 0.2, 0.2, 5.7);
        assert!(approx(d[0], 0.0, TOL));
        assert!(approx(d[1], 0.0, TOL));
        assert!(approx(d[2], 0.2, TOL)); // b = 0.2
    }

    #[test]
    fn lyapunov_1d_matches_logistic_specialization() {
        let r = 3.5;
        let x0 = 0.1;
        let n = 5000;
        let lambda_specific = lyapunov_exponent_logistic(r, x0, n);
        let lambda_general = lyapunov_exponent_1d(
            &|x| r * x * (1.0 - x),
            &|x| r * (1.0 - 2.0 * x),
            x0,
            n,
        );
        assert!(
            approx(lambda_specific, lambda_general, 1e-10),
            "specific={lambda_specific}, general={lambda_general}"
        );
    }
}
