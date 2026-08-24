pub mod attractors;
pub mod automata;
pub mod escape_time;
pub mod ifs;
pub mod lsystem;
pub mod noise;

use std::ops::{Add, Div, Mul, Sub};

use crate::math::constants::PI;

const ESCAPE_RADIUS_SQ: f64 = 4.0;
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
const LCG_SHIFT: u32 = 33;
const LCG_DIVISOR: f64 = (1u64 << 31) as f64;

// Barnsley fern probability thresholds (cumulative)
const FERN_STEM_THRESHOLD: f64 = 0.01;
const FERN_SMALL_LEAFLET_THRESHOLD: f64 = 0.08;
const FERN_LEFT_THRESHOLD: f64 = 0.15;

// Newton fractal
const NEWTON_ROOT_COUNT: usize = 3;

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    /// Create a complex number from real and imaginary parts.
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    /// Squared norm (modulus squared): |z|² = re² + im²
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    /// Norm (modulus): |z| = √(re² + im²)
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    /// Argument (phase angle): arg(z) = atan2(im, re)
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline]
    /// Complex conjugate: z* = re - im·i
    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    /// Complex division: z/w = z·w̄ / |w|²
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

// --- Mandelbrot ---

#[must_use]
/// Mandelbrot escape-time iteration count: z_{n+1} = z_n² + c, returns iterations to escape |z| > 2.
pub fn mandelbrot_iterations(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    let c = Complex::new(c_re, c_im);
    let mut z = Complex::new(0.0, 0.0);
    for i in 0..max_iter {
        z = z * z + c;
        if z.norm_sq() > ESCAPE_RADIUS_SQ {
            return i + 1;
        }
    }
    max_iter
}

#[must_use]
/// Smooth Mandelbrot iteration count using continuous escape-time coloring.
pub fn mandelbrot_smooth(c_re: f64, c_im: f64, max_iter: u32) -> f64 {
    let c = Complex::new(c_re, c_im);
    let mut z = Complex::new(0.0, 0.0);
    for i in 0..max_iter {
        z = z * z + c;
        if z.norm_sq() > ESCAPE_RADIUS_SQ {
            let iter = i + 1;
            let log_zn = z.norm_sq().ln() / 2.0;
            return iter as f64 - log_zn.ln() / std::f64::consts::LN_2;
        }
    }
    max_iter as f64
}

#[must_use]
/// Compute Mandelbrot iteration counts for an entire grid of pixels.
pub fn mandelbrot_grid(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
    max_iter: u32,
) -> Vec<u32> {
    assert!(width > 0, "width must be positive");
    assert!(height > 0, "height must be positive");
    let mut result = Vec::with_capacity(width * height);
    let dx = (x_max - x_min) / width as f64;
    let dy = (y_max - y_min) / height as f64;
    for row in 0..height {
        let ci = y_min + (row as f64 + 0.5) * dy;
        for col in 0..width {
            let cr = x_min + (col as f64 + 0.5) * dx;
            result.push(mandelbrot_iterations(cr, ci, max_iter));
        }
    }
    result
}

// --- Julia Sets ---

#[must_use]
/// Julia set escape-time iteration count for a fixed c: z_{n+1} = z_n² + c.
pub fn julia_iterations(z_re: f64, z_im: f64, c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    let c = Complex::new(c_re, c_im);
    let mut z = Complex::new(z_re, z_im);
    for i in 0..max_iter {
        z = z * z + c;
        if z.norm_sq() > ESCAPE_RADIUS_SQ {
            return i + 1;
        }
    }
    max_iter
}

#[must_use]
/// Compute Julia set iteration counts for an entire grid of pixels.
pub fn julia_grid(
    c_re: f64,
    c_im: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
    max_iter: u32,
) -> Vec<u32> {
    assert!(width > 0, "width must be positive");
    assert!(height > 0, "height must be positive");
    let mut result = Vec::with_capacity(width * height);
    let dx = (x_max - x_min) / width as f64;
    let dy = (y_max - y_min) / height as f64;
    for row in 0..height {
        let zi = y_min + (row as f64 + 0.5) * dy;
        for col in 0..width {
            let zr = x_min + (col as f64 + 0.5) * dx;
            result.push(julia_iterations(zr, zi, c_re, c_im, max_iter));
        }
    }
    result
}

// --- Burning Ship ---

#[must_use]
/// Burning Ship fractal iteration count: z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)² + c.
pub fn burning_ship_iterations(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    let mut z_re: f64 = 0.0;
    let mut z_im: f64 = 0.0;
    for i in 0..max_iter {
        let abs_re = z_re.abs();
        let abs_im = z_im.abs();
        let new_re = abs_re * abs_re - abs_im * abs_im + c_re;
        let new_im = 2.0 * abs_re * abs_im + c_im;
        z_re = new_re;
        z_im = new_im;
        if z_re * z_re + z_im * z_im > ESCAPE_RADIUS_SQ {
            return i + 1;
        }
    }
    max_iter
}

// --- Newton Fractal ---

#[must_use]
/// Newton fractal for f(z) = z³ - 1: returns (iterations, root_index) for convergence.
pub fn newton_fractal_iterations(
    z_re: f64,
    z_im: f64,
    max_iter: u32,
    tolerance: f64,
) -> (u32, u32) {
    let roots = [
        Complex::new(1.0, 0.0),
        Complex::new((2.0 * PI / 3.0).cos(), (2.0 * PI / 3.0).sin()),
        Complex::new((4.0 * PI / 3.0).cos(), (4.0 * PI / 3.0).sin()),
    ];

    let mut z = Complex::new(z_re, z_im);
    let tol_sq = tolerance * tolerance;

    for i in 0..max_iter {
        // Check convergence to any root
        for (ri, root) in roots.iter().enumerate() {
            let diff = z - *root;
            if diff.norm_sq() < tol_sq {
                return (i, ri as u32);
            }
        }

        // Newton step for f(z) = z^3 - 1: z_next = z - f(z)/f'(z) = z - (z^3 - 1)/(3z^2)
        let z2 = z * z;
        let z3 = z2 * z;
        let denom = Complex::new(3.0, 0.0) * z2;
        let denom_norm_sq = denom.norm_sq();
        if denom_norm_sq < 1e-30 {
            return (max_iter, 0);
        }
        let numerator = z3 - Complex::new(1.0, 0.0);
        // complex division: numerator / denom = numerator * conj(denom) / |denom|^2
        let conj_d = denom.conjugate();
        let prod = numerator * conj_d;
        let ratio = Complex::new(prod.re / denom_norm_sq, prod.im / denom_norm_sq);
        z = z - ratio;
    }

    // Find closest root
    let mut closest: u32 = 0;
    let mut min_dist = f64::MAX;
    for (ri, root) in roots.iter().enumerate() {
        let dist = (z - *root).norm_sq();
        if dist < min_dist {
            min_dist = dist;
            closest = ri as u32;
        }
    }
    (max_iter, closest)
}

// --- Iterated Function Systems ---

fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
    (*state >> LCG_SHIFT) as f64 / LCG_DIVISOR
}

#[must_use]
/// Generate Sierpinski triangle points via the chaos game (iterated function system).
pub fn sierpinski_point(x: f64, y: f64, iterations: usize) -> Vec<(f64, f64)> {
    let vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, (3.0_f64).sqrt() / 2.0)];
    let mut points = Vec::with_capacity(iterations);
    let mut px = x;
    let mut py = y;
    let mut rng_state: u64 = 12_345;

    for _ in 0..iterations {
        let r = lcg_next(&mut rng_state);
        let idx = (r * NEWTON_ROOT_COUNT as f64) as usize % NEWTON_ROOT_COUNT;
        let (vx, vy) = vertices[idx];
        px = (px + vx) / 2.0;
        py = (py + vy) / 2.0;
        points.push((px, py));
    }
    points
}

#[must_use]
/// Generate Barnsley fern points via the iterated function system with four affine maps.
pub fn barnsley_fern_point(x: f64, y: f64, iterations: usize) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(iterations);
    let mut px = x;
    let mut py = y;
    let mut rng_state: u64 = 12_345;

    for _ in 0..iterations {
        let r = lcg_next(&mut rng_state);
        let (nx, ny) = if r < FERN_STEM_THRESHOLD {
            // Stem
            (0.0, 0.16 * py)
        } else if r < FERN_SMALL_LEAFLET_THRESHOLD {
            // Small leaflet
            (0.20 * px - 0.26 * py, 0.23 * px + 0.22 * py + 1.6)
        } else if r < FERN_LEFT_THRESHOLD {
            // Left leaflet
            (-0.15 * px + 0.28 * py, 0.26 * px + 0.24 * py + 0.44)
        } else {
            // Right (main)
            (0.85 * px + 0.04 * py, -0.04 * px + 0.85 * py + 1.6)
        };
        px = nx;
        py = ny;
        points.push((px, py));
    }
    points
}

// --- Fractal Dimension ---

#[must_use]
/// Count occupied grid cells for box-counting fractal dimension estimation.
pub fn box_count_2d(
    points: &[(f64, f64)],
    grid_size: usize,
    bounds: (f64, f64, f64, f64),
) -> usize {
    if points.is_empty() || grid_size == 0 {
        return 0;
    }

    let (x_min, x_max, y_min, y_max) = bounds;
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    if x_range <= 0.0 || y_range <= 0.0 {
        return 0;
    }

    let mut occupied = vec![false; grid_size * grid_size];

    for &(px, py) in points {
        let col = ((px - x_min) / x_range * grid_size as f64) as usize;
        let row = ((py - y_min) / y_range * grid_size as f64) as usize;
        let col = col.min(grid_size - 1);
        let row = row.min(grid_size - 1);
        occupied[row * grid_size + col] = true;
    }

    occupied.iter().filter(|&&b| b).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    // --- Complex arithmetic ---

    #[test]
    fn complex_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a + b;
        assert!(approx(c.re, 4.0));
        assert!(approx(c.im, 6.0));
    }

    #[test]
    fn complex_sub() {
        let a = Complex::new(5.0, 3.0);
        let b = Complex::new(2.0, 1.0);
        let c = a - b;
        assert!(approx(c.re, 3.0));
        assert!(approx(c.im, 2.0));
    }

    #[test]
    fn complex_mul() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        assert!(approx(c.re, -5.0));
        assert!(approx(c.im, 10.0));
    }

    #[test]
    fn complex_norm_sq() {
        let z = Complex::new(3.0, 4.0);
        assert!(approx(z.norm_sq(), 25.0));
    }

    #[test]
    fn complex_norm() {
        let z = Complex::new(3.0, 4.0);
        assert!(approx(z.norm(), 5.0));
    }

    #[test]
    fn complex_arg() {
        let z = Complex::new(1.0, 1.0);
        assert!(approx(z.arg(), PI / 4.0));
    }

    #[test]
    fn complex_conjugate() {
        let z = Complex::new(3.0, 4.0);
        let c = z.conjugate();
        assert!(approx(c.re, 3.0));
        assert!(approx(c.im, -4.0));
    }

    // --- Mandelbrot ---

    #[test]
    fn mandelbrot_origin_in_set() {
        let max_iter = 1000;
        assert_eq!(mandelbrot_iterations(0.0, 0.0, max_iter), max_iter);
    }

    #[test]
    fn mandelbrot_escape_at_two() {
        assert_eq!(mandelbrot_iterations(2.0, 0.0, 1000), 2);
    }

    #[test]
    fn mandelbrot_grid_dimensions() {
        let width = 100;
        let height = 50;
        let grid = mandelbrot_grid(-2.0, 1.0, -1.0, 1.0, width, height, 100);
        assert_eq!(grid.len(), width * height);
    }

    // --- Julia ---

    #[test]
    fn julia_origin_with_c_zero_in_set() {
        let max_iter = 1000;
        assert_eq!(julia_iterations(0.0, 0.0, 0.0, 0.0, max_iter), max_iter);
    }

    // --- Burning Ship ---

    #[test]
    fn burning_ship_origin_in_set() {
        let max_iter = 1000;
        assert_eq!(burning_ship_iterations(0.0, 0.0, max_iter), max_iter);
    }

    // --- Newton Fractal ---

    #[test]
    fn newton_at_root_zero_converges_immediately() {
        let (iters, root) = newton_fractal_iterations(1.0, 0.0, 100, 1e-6);
        assert_eq!(iters, 0);
        assert_eq!(root, 0);
    }

    // --- IFS ---

    #[test]
    fn sierpinski_produces_correct_count() {
        let points = sierpinski_point(0.5, 0.5, 100);
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn barnsley_fern_produces_correct_count() {
        let points = barnsley_fern_point(0.0, 0.0, 100);
        assert_eq!(points.len(), 100);
    }

    // --- Box counting ---

    #[test]
    fn box_count_single_point() {
        let points = vec![(0.5, 0.5)];
        let count = box_count_2d(&points, 10, (0.0, 1.0, 0.0, 1.0));
        assert_eq!(count, 1);
    }

    #[test]
    fn box_count_empty() {
        let count = box_count_2d(&[], 10, (0.0, 1.0, 0.0, 1.0));
        assert_eq!(count, 0);
    }

    #[test]
    fn julia_grid_dimensions() {
        let width = 80;
        let height = 60;
        let grid = julia_grid(-0.7, 0.27015, -1.5, 1.5, -1.0, 1.0, width, height, 100);
        assert_eq!(grid.len(), width * height);
    }

    #[test]
    fn julia_grid_origin_c_zero_all_in_set() {
        // c = 0: the Julia set is the unit circle; origin should be in the set
        let grid = julia_grid(0.0, 0.0, -0.1, 0.1, -0.1, 0.1, 3, 3, 1000);
        // The center pixel (z near origin) should reach max_iter
        assert_eq!(grid[4], 1000);
    }

    #[test]
    fn mandelbrot_smooth_in_set_returns_max_iter() {
        let val = mandelbrot_smooth(0.0, 0.0, 100);
        assert!(approx(val, 100.0), "origin should return max_iter, got {val}");
    }

    #[test]
    fn mandelbrot_smooth_escape_gives_fractional_value() {
        let val = mandelbrot_smooth(2.0, 0.0, 1000);
        // Should escape quickly and give a smooth fractional iteration count
        assert!(val < 5.0, "should escape quickly, got {val}");
        // Should be fractional (not integer)
        assert!((val - val.round()).abs() > 0.01, "smooth coloring should be fractional, got {val}");
    }

    // --- Burning Ship escape path ---

    #[test]
    fn burning_ship_escapes_far_from_origin() {
        // c = (10, 0) is far outside the set and must escape quickly
        let iters = burning_ship_iterations(10.0, 0.0, 1000);
        assert!(iters < 1000, "far-out point should escape, got {iters}");
        assert!(iters >= 1, "should take at least one iteration");
    }

    // --- Newton fractal: exercising the iteration loop ---

    #[test]
    fn newton_converges_to_root_after_iterations() {
        // Start away from all roots; Newton's method should converge in a few steps
        let (iters, root) = newton_fractal_iterations(0.5, 0.5, 200, 1e-6);
        assert!(iters > 0, "should require at least one Newton step");
        assert!(iters < 200, "should converge before max_iter");
        assert!(root < 3, "root_index must be 0, 1, or 2");
    }

    #[test]
    fn newton_max_iter_fallback_finds_closest_root() {
        // With max_iter = 1 and a point far from roots, should hit the closest-root fallback
        let (iters, root) = newton_fractal_iterations(0.5, 0.5, 1, 1e-15);
        assert_eq!(iters, 1, "should exhaust max_iter");
        assert!(root < 3, "closest root index must be valid");
    }

    #[test]
    fn newton_degenerate_at_origin() {
        // z = 0 makes f'(z) = 3z² = 0, triggering the degenerate denominator guard
        let (iters, root) = newton_fractal_iterations(0.0, 0.0, 100, 1e-6);
        assert_eq!(iters, 100, "degenerate point should return max_iter");
        assert_eq!(root, 0);
    }

    // --- Barnsley fern: ensure all branches are hit ---

    #[test]
    fn barnsley_fern_all_branches_with_many_iterations() {
        // The LCG is deterministic. With enough iterations all four probability
        // branches (stem at p < 0.01, small leaflet, left leaflet, main) will fire.
        let points = barnsley_fern_point(0.0, 0.0, 10_000);
        assert_eq!(points.len(), 10_000);
        // Fern points should stay in a bounded region (roughly x in [-3, 3], y in [0, 10])
        for &(x, y) in &points {
            assert!(x > -5.0 && x < 5.0, "x out of expected range: {x}");
            assert!(y > -1.0 && y < 12.0, "y out of expected range: {y}");
        }
    }

    // --- Box counting: invalid bounds ---

    #[test]
    fn box_count_zero_grid_size() {
        let points = vec![(0.5, 0.5)];
        assert_eq!(box_count_2d(&points, 0, (0.0, 1.0, 0.0, 1.0)), 0);
    }

    #[test]
    fn box_count_inverted_bounds() {
        // x_max < x_min => range <= 0 => returns 0
        let points = vec![(0.5, 0.5)];
        assert_eq!(box_count_2d(&points, 10, (1.0, 0.0, 0.0, 1.0)), 0);
    }

    #[test]
    fn box_count_zero_y_range() {
        let points = vec![(0.5, 0.5)];
        assert_eq!(box_count_2d(&points, 10, (0.0, 1.0, 1.0, 1.0)), 0);
    }
}
