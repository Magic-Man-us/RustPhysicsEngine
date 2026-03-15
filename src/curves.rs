use crate::math::Vec3;
use crate::math::constants::PI;

// ---------------------------------------------------------------------------
// Conic Sections
// ---------------------------------------------------------------------------

pub fn circle_area(radius: f64) -> f64 {
    PI * radius * radius
}

pub fn circle_circumference(radius: f64) -> f64 {
    2.0 * PI * radius
}

/// Returns (x-cx)^2 + (y-cy)^2 - r^2. Zero means the point lies on the circle.
pub fn circle_equation(x: f64, y: f64, cx: f64, cy: f64, r: f64) -> f64 {
    let dx = x - cx;
    let dy = y - cy;
    dx * dx + dy * dy - r * r
}

/// Ramanujan approximation: pi * (3(a+b) - sqrt((3a+b)(a+3b)))
pub fn ellipse_circumference_approx(a: f64, b: f64) -> f64 {
    PI * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt())
}

/// Returns x^2/a^2 + y^2/b^2 - 1. Zero means the point lies on the ellipse.
pub fn ellipse_equation(x: f64, y: f64, a: f64, b: f64) -> f64 {
    (x * x) / (a * a) + (y * y) / (b * b) - 1.0
}

/// Eccentricity e = sqrt(1 - b^2/a^2) for a > b.
pub fn ellipse_eccentricity(a: f64, b: f64) -> f64 {
    (1.0 - (b * b) / (a * a)).sqrt()
}

/// Focus distance f = 1/(4a) for parabola y = ax^2.
pub fn parabola_focus(a: f64) -> f64 {
    1.0 / (4.0 * a)
}

pub fn parabola_equation(x: f64, a: f64) -> f64 {
    a * x * x
}

/// Eccentricity e = sqrt(1 + b^2/a^2).
pub fn hyperbola_eccentricity(a: f64, b: f64) -> f64 {
    (1.0 + (b * b) / (a * a)).sqrt()
}

pub fn hyperbola_asymptote_slope(a: f64, b: f64) -> f64 {
    b / a
}

/// Discriminant B^2 - 4AC for general conic Ax^2 + Bxy + Cy^2 + ...
/// Negative => ellipse, zero => parabola, positive => hyperbola.
pub fn conic_discriminant(a: f64, b: f64, c: f64) -> f64 {
    b * b - 4.0 * a * c
}

// ---------------------------------------------------------------------------
// Bezier Curves
// ---------------------------------------------------------------------------

pub fn bezier_quadratic(
    t: f64,
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
) -> (f64, f64) {
    let u = 1.0 - t;
    let uu = u * u;
    let tt = t * t;
    (
        uu * p0.0 + 2.0 * u * t * p1.0 + tt * p2.0,
        uu * p0.1 + 2.0 * u * t * p1.1 + tt * p2.1,
    )
}

pub fn bezier_cubic(
    t: f64,
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
) -> (f64, f64) {
    let u = 1.0 - t;
    let uu = u * u;
    let uuu = uu * u;
    let tt = t * t;
    let ttt = tt * t;
    (
        uuu * p0.0 + 3.0 * uu * t * p1.0 + 3.0 * u * tt * p2.0 + ttt * p3.0,
        uuu * p0.1 + 3.0 * uu * t * p1.1 + 3.0 * u * tt * p2.1 + ttt * p3.1,
    )
}

pub fn bezier_quadratic_3d(t: f64, p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3 {
    let u = 1.0 - t;
    p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)
}

pub fn bezier_cubic_3d(t: f64, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Vec3 {
    let u = 1.0 - t;
    let uu = u * u;
    let tt = t * t;
    p0 * (uu * u) + p1 * (3.0 * uu * t) + p2 * (3.0 * u * tt) + p3 * (tt * t)
}

/// Sample n+1 evenly spaced points along a cubic Bezier curve (t from 0 to 1).
pub fn bezier_sample(
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    n: usize,
) -> Vec<(f64, f64)> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            bezier_cubic(t, p0, p1, p2, p3)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Parametric Curves
// ---------------------------------------------------------------------------

pub fn parametric_circle(t: f64, r: f64) -> (f64, f64) {
    (r * t.cos(), r * t.sin())
}

pub fn parametric_ellipse(t: f64, a: f64, b: f64) -> (f64, f64) {
    (a * t.cos(), b * t.sin())
}

/// Archimedean spiral: r = a + b*t.
pub fn parametric_spiral(t: f64, a: f64, b: f64) -> (f64, f64) {
    let r = a + b * t;
    (r * t.cos(), r * t.sin())
}

/// Lissajous figure: (sin(a*t + delta), sin(b*t)).
pub fn parametric_lissajous(t: f64, a: f64, b: f64, delta: f64) -> (f64, f64) {
    ((a * t + delta).sin(), (b * t).sin())
}

/// Cycloid: (r(t - sin(t)), r(1 - cos(t))).
pub fn parametric_cycloid(t: f64, r: f64) -> (f64, f64) {
    (r * (t - t.sin()), r * (1.0 - t.cos()))
}

/// Helix: (r cos(t), r sin(t), pitch * t / (2*pi)).
pub fn parametric_helix(t: f64, radius: f64, pitch: f64) -> (f64, f64, f64) {
    let two_pi = 2.0 * PI;
    (radius * t.cos(), radius * t.sin(), pitch * t / two_pi)
}

// ---------------------------------------------------------------------------
// Arc Length & Curvature
// ---------------------------------------------------------------------------

/// Numerical arc length via piecewise linear approximation with n segments.
pub fn arc_length_parametric(
    fx: &dyn Fn(f64) -> f64,
    fy: &dyn Fn(f64) -> f64,
    t0: f64,
    t1: f64,
    n: usize,
) -> f64 {
    let dt = (t1 - t0) / n as f64;
    let mut length = 0.0;
    let mut prev_x = fx(t0);
    let mut prev_y = fy(t0);
    for i in 1..=n {
        let t = t0 + i as f64 * dt;
        let cur_x = fx(t);
        let cur_y = fy(t);
        let dx = cur_x - prev_x;
        let dy = cur_y - prev_y;
        length += (dx * dx + dy * dy).sqrt();
        prev_x = cur_x;
        prev_y = cur_y;
    }
    length
}

pub fn arc_length_circle(radius: f64, angle: f64) -> f64 {
    radius * angle
}

/// Curvature kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2).
pub fn curvature_2d(dxdt: f64, dydt: f64, d2xdt2: f64, d2ydt2: f64) -> f64 {
    let numerator = (dxdt * d2ydt2 - dydt * d2xdt2).abs();
    let speed_sq = dxdt * dxdt + dydt * dydt;
    let denominator = speed_sq * speed_sq.sqrt();
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -- Conic Sections --

    #[test]
    fn test_circle_area() {
        assert!(approx(circle_area(1.0), PI));
        assert!(approx(circle_area(2.0), 4.0 * PI));
    }

    #[test]
    fn test_circle_circumference() {
        assert!(approx(circle_circumference(1.0), 2.0 * PI));
        assert!(approx(circle_circumference(3.0), 6.0 * PI));
    }

    #[test]
    fn test_circle_equation_on_circle() {
        assert!(approx(circle_equation(1.0, 0.0, 0.0, 0.0, 1.0), 0.0));
        assert!(approx(circle_equation(3.0, 4.0, 3.0, 4.0, 0.0), 0.0));
    }

    #[test]
    fn test_circle_equation_inside_outside() {
        assert!(circle_equation(0.5, 0.0, 0.0, 0.0, 1.0) < 0.0);
        assert!(circle_equation(2.0, 0.0, 0.0, 0.0, 1.0) > 0.0);
    }

    #[test]
    fn test_ellipse_equation_on_ellipse() {
        assert!(approx(ellipse_equation(2.0, 0.0, 2.0, 1.0), 0.0));
        assert!(approx(ellipse_equation(0.0, 3.0, 5.0, 3.0), 0.0));
    }

    #[test]
    fn test_ellipse_eccentricity_circle() {
        assert!(approx(ellipse_eccentricity(5.0, 5.0), 0.0));
    }

    #[test]
    fn test_ellipse_eccentricity_known() {
        let e = ellipse_eccentricity(5.0, 3.0);
        assert!(approx(e, 0.8));
    }

    #[test]
    fn test_ellipse_circumference_circle() {
        let c = ellipse_circumference_approx(1.0, 1.0);
        assert!(approx_rel(c, 2.0 * PI, 0.01));
    }

    #[test]
    fn test_parabola_focus() {
        assert!(approx(parabola_focus(1.0), 0.25));
        assert!(approx(parabola_focus(0.5), 0.5));
    }

    #[test]
    fn test_parabola_equation() {
        assert!(approx(parabola_equation(3.0, 2.0), 18.0));
    }

    #[test]
    fn test_hyperbola_eccentricity() {
        let e = hyperbola_eccentricity(3.0, 4.0);
        assert!(approx(e, 5.0 / 3.0));
    }

    #[test]
    fn test_hyperbola_asymptote_slope() {
        assert!(approx(hyperbola_asymptote_slope(3.0, 4.0), 4.0 / 3.0));
    }

    #[test]
    fn test_conic_discriminant() {
        // Ellipse: B^2 - 4AC < 0
        assert!(conic_discriminant(1.0, 0.0, 1.0) < 0.0);
        // Parabola: B^2 - 4AC = 0
        assert!(approx(conic_discriminant(1.0, 2.0, 1.0), 0.0));
        // Hyperbola: B^2 - 4AC > 0
        assert!(conic_discriminant(1.0, 0.0, -1.0) > 0.0);
    }

    // -- Bezier Curves --

    #[test]
    fn test_bezier_quadratic_endpoints() {
        let p0 = (0.0, 0.0);
        let p1 = (0.5, 1.0);
        let p2 = (1.0, 0.0);
        let start = bezier_quadratic(0.0, p0, p1, p2);
        let end = bezier_quadratic(1.0, p0, p1, p2);
        assert!(approx(start.0, p0.0) && approx(start.1, p0.1));
        assert!(approx(end.0, p2.0) && approx(end.1, p2.1));
    }

    #[test]
    fn test_bezier_cubic_endpoints() {
        let p0 = (0.0, 0.0);
        let p1 = (0.25, 1.0);
        let p2 = (0.75, 1.0);
        let p3 = (1.0, 0.0);
        let start = bezier_cubic(0.0, p0, p1, p2, p3);
        let end = bezier_cubic(1.0, p0, p1, p2, p3);
        assert!(approx(start.0, p0.0) && approx(start.1, p0.1));
        assert!(approx(end.0, p3.0) && approx(end.1, p3.1));
    }

    #[test]
    fn test_bezier_cubic_midpoint_straight_line() {
        let p0 = (0.0, 0.0);
        let p1 = (1.0 / 3.0, 1.0 / 3.0);
        let p2 = (2.0 / 3.0, 2.0 / 3.0);
        let p3 = (1.0, 1.0);
        let mid = bezier_cubic(0.5, p0, p1, p2, p3);
        assert!(approx(mid.0, 0.5) && approx(mid.1, 0.5));
    }

    #[test]
    fn test_bezier_quadratic_3d_endpoints() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(1.0, 2.0, 0.0);
        let p2 = Vec3::new(2.0, 0.0, 0.0);
        let start = bezier_quadratic_3d(0.0, p0, p1, p2);
        let end = bezier_quadratic_3d(1.0, p0, p1, p2);
        assert!(approx(start.x, 0.0) && approx(start.y, 0.0));
        assert!(approx(end.x, 2.0) && approx(end.y, 0.0));
    }

    #[test]
    fn test_bezier_cubic_3d_endpoints() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(1.0, 1.0, 1.0);
        let p2 = Vec3::new(2.0, 1.0, 1.0);
        let p3 = Vec3::new(3.0, 0.0, 0.0);
        let start = bezier_cubic_3d(0.0, p0, p1, p2, p3);
        let end = bezier_cubic_3d(1.0, p0, p1, p2, p3);
        assert!(approx(start.x, 0.0) && approx(start.z, 0.0));
        assert!(approx(end.x, 3.0) && approx(end.z, 0.0));
    }

    #[test]
    fn test_bezier_sample_count() {
        let pts = bezier_sample((0.0, 0.0), (0.5, 1.0), (0.5, 1.0), (1.0, 0.0), 10);
        assert_eq!(pts.len(), 11);
    }

    #[test]
    fn test_bezier_sample_endpoints() {
        let p0 = (0.0, 0.0);
        let p3 = (1.0, 0.0);
        let pts = bezier_sample(p0, (0.3, 1.0), (0.7, 1.0), p3, 20);
        assert!(approx(pts[0].0, p0.0) && approx(pts[0].1, p0.1));
        assert!(approx(pts[20].0, p3.0) && approx(pts[20].1, p3.1));
    }

    // -- Parametric Curves --

    #[test]
    fn test_parametric_circle() {
        let (x, y) = parametric_circle(0.0, 5.0);
        assert!(approx(x, 5.0) && approx(y, 0.0));
        let (x2, y2) = parametric_circle(PI / 2.0, 5.0);
        assert!(approx(x2, 0.0) && approx(y2, 5.0));
    }

    #[test]
    fn test_parametric_ellipse() {
        let (x, y) = parametric_ellipse(0.0, 3.0, 2.0);
        assert!(approx(x, 3.0) && approx(y, 0.0));
        let (x2, y2) = parametric_ellipse(PI / 2.0, 3.0, 2.0);
        assert!(approx(x2, 0.0) && approx(y2, 2.0));
    }

    #[test]
    fn test_parametric_spiral_origin() {
        let (x, y) = parametric_spiral(0.0, 1.0, 2.0);
        assert!(approx(x, 1.0) && approx(y, 0.0));
    }

    #[test]
    fn test_parametric_lissajous_origin() {
        let (x, y) = parametric_lissajous(0.0, 1.0, 1.0, 0.0);
        assert!(approx(x, 0.0) && approx(y, 0.0));
    }

    #[test]
    fn test_parametric_cycloid_origin() {
        let (x, y) = parametric_cycloid(0.0, 5.0);
        assert!(approx(x, 0.0) && approx(y, 0.0));
    }

    #[test]
    fn test_parametric_cycloid_top() {
        let (x, y) = parametric_cycloid(PI, 1.0);
        assert!(approx(x, PI) && approx(y, 2.0));
    }

    #[test]
    fn test_parametric_helix() {
        let (x, y, z) = parametric_helix(0.0, 2.0, 5.0);
        assert!(approx(x, 2.0) && approx(y, 0.0) && approx(z, 0.0));
        let (_, _, z2) = parametric_helix(2.0 * PI, 2.0, 5.0);
        assert!(approx(z2, 5.0));
    }

    // -- Arc Length & Curvature --

    #[test]
    fn test_arc_length_circle() {
        assert!(approx(arc_length_circle(1.0, 2.0 * PI), 2.0 * PI));
        assert!(approx(arc_length_circle(3.0, PI), 3.0 * PI));
    }

    #[test]
    fn test_arc_length_parametric_straight_line() {
        // Straight line from (0,0) to (3,4), length = 5
        let fx = |t: f64| 3.0 * t;
        let fy = |t: f64| 4.0 * t;
        let len = arc_length_parametric(&fx, &fy, 0.0, 1.0, 1000);
        assert!(approx_rel(len, 5.0, 1e-6));
    }

    #[test]
    fn test_arc_length_parametric_circle() {
        let r = 1.0;
        let fx = |t: f64| r * t.cos();
        let fy = |t: f64| r * t.sin();
        let len = arc_length_parametric(&fx, &fy, 0.0, 2.0 * PI, 10_000);
        assert!(approx_rel(len, 2.0 * PI, 1e-4));
    }

    #[test]
    fn test_curvature_2d_circle() {
        // Unit circle: x = cos(t), y = sin(t)
        // x' = -sin(t), y' = cos(t), x'' = -cos(t), y'' = -sin(t)
        // At t=0: x'=0, y'=1, x''=-1, y''=0 => kappa = |0*0 - 1*(-1)| / (0+1)^(3/2) = 1
        let kappa = curvature_2d(0.0, 1.0, -1.0, 0.0);
        assert!(approx(kappa, 1.0));
    }

    #[test]
    fn test_curvature_2d_straight_line() {
        // x' = 1, y' = 2, x'' = 0, y'' = 0 => kappa = 0
        let kappa = curvature_2d(1.0, 2.0, 0.0, 0.0);
        assert!(approx(kappa, 0.0));
    }

    #[test]
    fn test_curvature_2d_zero_speed() {
        let kappa = curvature_2d(0.0, 0.0, 1.0, 1.0);
        assert!(approx(kappa, 0.0));
    }
}
