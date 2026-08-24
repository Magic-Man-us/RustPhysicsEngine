//! Escape-time fractals: a generic iteration engine with smooth
//! coloring, orbit traps, and distance estimation, the classic
//! quadratic families (Mandelbrot, Julia, tricorn, burning ship),
//! Newton/nova and magnet fractals, Lyapunov fractals, Buddhabrot
//! accumulation, and perturbation iteration for deep zooms.

use crate::fractals::Complex;
use crate::monte_carlo::Rng;
use crate::spatial::primitives::Rect;

/// Orbit trap shapes: the result records the minimum distance from
/// the orbit to the trap.
#[derive(Debug, Clone, Copy)]
pub enum OrbitTrap {
    /// Distance to a fixed point.
    Point(Complex),
    /// Distance to the coordinate axes (min of |re|, |im|).
    Cross,
    /// Distance to the circle of the given radius about the origin.
    Circle(f64),
    /// Distance to the line through the origin at the given angle.
    Line(f64),
}

/// Iteration parameters.
#[derive(Debug, Clone, Copy)]
pub struct EscapeParams {
    pub max_iter: u32,
    /// Escape radius (|z| > bailout escapes). Use >= 2 for the
    /// quadratic families; larger values smooth the fractional
    /// iteration count.
    pub bailout: f64,
    /// Request a distance estimate (honored by the derivative-aware
    /// entry points and the wrappers built on them).
    pub compute_distance: bool,
    pub trap: Option<OrbitTrap>,
}

impl Default for EscapeParams {
    fn default() -> Self {
        Self { max_iter: 256, bailout: 4.0, compute_distance: false, trap: None }
    }
}

/// Result of iterating one point.
#[derive(Debug, Clone, Copy)]
pub struct EscapeResult {
    /// Iterations completed before escape (or `max_iter`).
    pub iterations: u32,
    pub escaped: bool,
    /// Fractional iteration count n + 1 − log₂ ln |z| (the smooth
    /// coloring of quadratic maps; approximate for other powers).
    /// Equal to `max_iter` for interior points.
    pub smooth: f64,
    pub final_z: Complex,
    /// Exterior distance estimate |z| ln |z| / |dz| when a
    /// derivative was tracked and the point escaped.
    pub distance: Option<f64>,
    /// Minimum distance from the orbit to the requested trap.
    pub orbit_trap: Option<f64>,
}

fn trap_distance(trap: OrbitTrap, z: Complex) -> f64 {
    match trap {
        OrbitTrap::Point(p) => (z - p).norm(),
        OrbitTrap::Cross => z.re.abs().min(z.im.abs()),
        OrbitTrap::Circle(r) => (z.norm() - r).abs(),
        OrbitTrap::Line(angle) => {
            let (s, c) = angle.sin_cos();
            (z.re * s - z.im * c).abs()
        }
    }
}

fn smooth_count(iterations: u32, z: Complex) -> f64 {
    let n = z.norm();
    if n > 1.0 {
        f64::from(iterations) + 1.0 - n.ln().ln() / std::f64::consts::LN_2
    } else {
        f64::from(iterations)
    }
}

/// Iterates z ← f(z, c) until |z| exceeds the bailout, recording
/// smooth iteration counts and orbit-trap distances. `distance` is
/// `None` here (no derivative is tracked); use
/// [`escape_time_with_derivative`] for distance estimates.
///
/// # Panics
/// Panics unless `max_iter >= 1` and `bailout > 1`.
#[must_use]
pub fn escape_time(
    f: &dyn Fn(Complex, Complex) -> Complex,
    z0: Complex,
    c: Complex,
    params: &EscapeParams,
) -> EscapeResult {
    assert!(params.max_iter >= 1, "max_iter must be >= 1");
    assert!(params.bailout > 1.0, "bailout must exceed 1");
    let bb = params.bailout * params.bailout;
    let mut z = z0;
    let mut trap = params.trap.map(|t| trap_distance(t, z));
    for i in 0..params.max_iter {
        if z.norm_sq() > bb {
            return EscapeResult {
                iterations: i,
                escaped: true,
                smooth: smooth_count(i, z),
                final_z: z,
                distance: None,
                orbit_trap: trap,
            };
        }
        z = f(z, c);
        if let (Some(t), Some(d)) = (params.trap, trap.as_mut()) {
            *d = d.min(trap_distance(t, z));
        }
    }
    EscapeResult {
        iterations: params.max_iter,
        escaped: false,
        smooth: f64::from(params.max_iter),
        final_z: z,
        distance: None,
        orbit_trap: trap,
    }
}

/// Escape-time iteration that also tracks the parameter-space
/// derivative dz ← (∂f/∂z)·dz + 1 (the recurrence for sets like the
/// Mandelbrot set where c varies per pixel and z₀ is fixed), giving
/// the exterior distance estimate |z| ln|z| / |dz| on escape.
///
/// # Panics
/// Panics unless `max_iter >= 1` and `bailout > 1`.
#[must_use]
pub fn escape_time_with_derivative(
    f: &dyn Fn(Complex, Complex) -> Complex,
    df_dz: &dyn Fn(Complex, Complex) -> Complex,
    z0: Complex,
    c: Complex,
    params: &EscapeParams,
) -> EscapeResult {
    escape_with_derivative(f, df_dz, z0, c, params, true)
}

fn escape_with_derivative(
    f: &dyn Fn(Complex, Complex) -> Complex,
    df_dz: &dyn Fn(Complex, Complex) -> Complex,
    z0: Complex,
    c: Complex,
    params: &EscapeParams,
    param_space: bool,
) -> EscapeResult {
    assert!(params.max_iter >= 1, "max_iter must be >= 1");
    assert!(params.bailout > 1.0, "bailout must exceed 1");
    let bb = params.bailout * params.bailout;
    let mut z = z0;
    let mut dz = if param_space { Complex::new(0.0, 0.0) } else { Complex::new(1.0, 0.0) };
    let mut trap = params.trap.map(|t| trap_distance(t, z));
    for i in 0..params.max_iter {
        if z.norm_sq() > bb {
            let n = z.norm();
            let dn = dz.norm();
            let distance = if params.compute_distance && dn > 0.0 {
                Some(n * n.ln() / dn)
            } else {
                None
            };
            return EscapeResult {
                iterations: i,
                escaped: true,
                smooth: smooth_count(i, z),
                final_z: z,
                distance,
                orbit_trap: trap,
            };
        }
        dz = df_dz(z, c) * dz;
        if param_space {
            dz = dz + Complex::new(1.0, 0.0);
        }
        z = f(z, c);
        if let (Some(t), Some(d)) = (params.trap, trap.as_mut()) {
            *d = d.min(trap_distance(t, z));
        }
    }
    EscapeResult {
        iterations: params.max_iter,
        escaped: false,
        smooth: f64::from(params.max_iter),
        final_z: z,
        distance: None,
        orbit_trap: trap,
    }
}

fn sq(z: Complex) -> Complex {
    z * z
}

fn cscale(z: Complex, s: f64) -> Complex {
    Complex::new(z.re * s, z.im * s)
}

/// z^p for real p by the principal branch.
fn cpow(z: Complex, p: f64) -> Complex {
    let r = z.norm();
    if r == 0.0 {
        return Complex::new(0.0, 0.0);
    }
    let theta = z.arg();
    let rp = r.powf(p);
    Complex::new(rp * (p * theta).cos(), rp * (p * theta).sin())
}

/// The Mandelbrot set iteration z ← z² + c from z₀ = 0; tracks the
/// derivative for distance estimates when requested.
#[must_use]
pub fn mandelbrot(c: Complex, params: &EscapeParams) -> EscapeResult {
    escape_time_with_derivative(
        &|z, c| sq(z) + c,
        &|z, _| cscale(z, 2.0),
        Complex::new(0.0, 0.0),
        c,
        params,
    )
}

/// The multibrot iteration z ← z^power + c from z₀ = 0.
///
/// # Panics
/// Panics unless `power > 1`.
#[must_use]
pub fn multibrot(c: Complex, power: f64, params: &EscapeParams) -> EscapeResult {
    assert!(power > 1.0, "multibrot needs power > 1");
    escape_time_with_derivative(
        &move |z, c| cpow(z, power) + c,
        &move |z, _| cscale(cpow(z, power - 1.0), power),
        Complex::new(0.0, 0.0),
        c,
        params,
    )
}

/// The Julia set iteration z ← z² + c from the given z; tracks the
/// dynamic-space derivative for distance estimates when requested.
#[must_use]
pub fn julia(z: Complex, c: Complex, params: &EscapeParams) -> EscapeResult {
    escape_with_derivative(&|z, c| sq(z) + c, &|z, _| cscale(z, 2.0), z, c, params, false)
}

/// The tricorn (Mandelbar): z ← conj(z)² + c.
#[must_use]
pub fn tricorn(c: Complex, params: &EscapeParams) -> EscapeResult {
    escape_time(&|z, c| sq(z.conjugate()) + c, Complex::new(0.0, 0.0), c, params)
}

/// The burning ship: z ← (|Re z| + i |Im z|)² + c.
#[must_use]
pub fn burning_ship(c: Complex, params: &EscapeParams) -> EscapeResult {
    escape_time(
        &|z, c| sq(Complex::new(z.re.abs(), z.im.abs())) + c,
        Complex::new(0.0, 0.0),
        c,
        params,
    )
}

/// The Phoenix fractal: z_{n+1} = z_n² + c + p·z_{n−1}.
#[must_use]
pub fn phoenix(z: Complex, c: Complex, p: Complex, params: &EscapeParams) -> EscapeResult {
    assert!(params.max_iter >= 1, "max_iter must be >= 1");
    assert!(params.bailout > 1.0, "bailout must exceed 1");
    let bb = params.bailout * params.bailout;
    let mut prev = Complex::new(0.0, 0.0);
    let mut cur = z;
    let mut trap = params.trap.map(|t| trap_distance(t, cur));
    for i in 0..params.max_iter {
        if cur.norm_sq() > bb {
            return EscapeResult {
                iterations: i,
                escaped: true,
                smooth: smooth_count(i, cur),
                final_z: cur,
                distance: None,
                orbit_trap: trap,
            };
        }
        let next = sq(cur) + c + p * prev;
        prev = cur;
        cur = next;
        if let (Some(t), Some(d)) = (params.trap, trap.as_mut()) {
            *d = d.min(trap_distance(t, cur));
        }
    }
    EscapeResult {
        iterations: params.max_iter,
        escaped: false,
        smooth: f64::from(params.max_iter),
        final_z: cur,
        distance: None,
        orbit_trap: trap,
    }
}

fn poly_eval(poly: &[Complex], z: Complex) -> Complex {
    let mut acc = Complex::new(0.0, 0.0);
    for &c in poly.iter().rev() {
        acc = acc * z + c;
    }
    acc
}

fn poly_derivative(poly: &[Complex]) -> Vec<Complex> {
    poly.iter()
        .enumerate()
        .skip(1)
        .map(|(k, &c)| cscale(c, k as f64))
        .collect()
}

/// Roots of a complex polynomial by Durand-Kerner iteration,
/// sorted by (re, im) for stable indexing.
fn poly_roots(poly: &[Complex]) -> Vec<Complex> {
    let deg = poly.len() - 1;
    let lead = poly[deg];
    let monic: Vec<Complex> = poly.iter().map(|&c| c / lead).collect();
    // Standard starting points: powers of a non-real seed.
    let seed = Complex::new(0.4, 0.9);
    let mut roots: Vec<Complex> = (0..deg)
        .map(|k| {
            let mut p = Complex::new(1.0, 0.0);
            for _ in 0..k {
                p = p * seed;
            }
            p
        })
        .collect();
    for _ in 0..200 {
        let mut moved = 0.0f64;
        for i in 0..deg {
            let mut denom = Complex::new(1.0, 0.0);
            for j in 0..deg {
                if j != i {
                    denom = denom * (roots[i] - roots[j]);
                }
            }
            if denom.norm_sq() == 0.0 {
                continue;
            }
            let delta = poly_eval(&monic, roots[i]) / denom;
            roots[i] = roots[i] - delta;
            moved = moved.max(delta.norm());
        }
        if moved < 1e-14 {
            break;
        }
    }
    roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    roots
}

/// Newton fractal for the polynomial with coefficients `poly`
/// (constant term first): iterates z ← z − p(z)/p′(z) and returns
/// the index of the root reached (roots sorted by real then
/// imaginary part) and the iteration count. Index `degree` (one past
/// the last root) marks failure to converge within `max_iter`.
///
/// # Panics
/// Panics unless the polynomial has degree >= 2.
#[must_use]
pub fn newton_fractal(z: Complex, poly: &[Complex], params: &EscapeParams) -> (usize, u32) {
    assert!(
        poly.len() >= 3 && poly.last().map(|c| c.norm_sq() > 0.0) == Some(true),
        "Newton fractal needs a polynomial of degree >= 2"
    );
    let roots = poly_roots(poly);
    let dpoly = poly_derivative(poly);
    let mut z = z;
    for i in 0..params.max_iter {
        for (k, &r) in roots.iter().enumerate() {
            if (z - r).norm() < 1e-9 {
                return (k, i);
            }
        }
        let d = poly_eval(&dpoly, z);
        if d.norm_sq() == 0.0 {
            return (roots.len(), i);
        }
        z = z - poly_eval(poly, z) / d;
    }
    (roots.len(), params.max_iter)
}

/// Nova fractal: relaxed Newton iteration on z^power − 1 with an
/// added constant, z ← z − relax·(z^p − 1)/(p·z^{p−1}) + c.
/// `escaped = true` records convergence to a fixed point (|Δz| <
/// 1e-9); `iterations` counts steps to convergence.
///
/// # Panics
/// Panics unless `power > 1`.
#[must_use]
pub fn nova_fractal(
    z: Complex,
    c: Complex,
    power: f64,
    relax: f64,
    params: &EscapeParams,
) -> EscapeResult {
    assert!(power > 1.0, "nova needs power > 1");
    let mut cur = z;
    let mut trap = params.trap.map(|t| trap_distance(t, cur));
    for i in 0..params.max_iter {
        let denom = cscale(cpow(cur, power - 1.0), power);
        if denom.norm_sq() == 0.0 {
            break;
        }
        let next = cur - cscale((cpow(cur, power) - Complex::new(1.0, 0.0)) / denom, relax) + c;
        let step = (next - cur).norm();
        cur = next;
        if let (Some(t), Some(d)) = (params.trap, trap.as_mut()) {
            *d = d.min(trap_distance(t, cur));
        }
        if step < 1e-9 {
            return EscapeResult {
                iterations: i + 1,
                escaped: true,
                smooth: f64::from(i + 1),
                final_z: cur,
                distance: None,
                orbit_trap: trap,
            };
        }
    }
    EscapeResult {
        iterations: params.max_iter,
        escaped: false,
        smooth: f64::from(params.max_iter),
        final_z: cur,
        distance: None,
        orbit_trap: trap,
    }
}

/// Magnet iterations: the exterior attractor is the fixed point
/// z = 1 (which plays the role infinity plays for the Mandelbrot
/// set), so `escaped` records either |z| > bailout or convergence
/// to 1; interior parameters orbit other attracting cycles.
fn magnet_escape(
    step: impl Fn(Complex, Complex) -> Complex,
    c: Complex,
    params: &EscapeParams,
) -> EscapeResult {
    assert!(params.max_iter >= 1, "max_iter must be >= 1");
    let bb = params.bailout * params.bailout;
    let one = Complex::new(1.0, 0.0);
    let mut z = Complex::new(0.0, 0.0);
    let mut trap = params.trap.map(|t| trap_distance(t, z));
    for i in 0..params.max_iter {
        if z.norm_sq() > bb || (z - one).norm_sq() < 1e-12 {
            return EscapeResult {
                iterations: i,
                escaped: true,
                smooth: smooth_count(i, z),
                final_z: z,
                distance: None,
                orbit_trap: trap,
            };
        }
        z = step(z, c);
        if let (Some(t), Some(d)) = (params.trap, trap.as_mut()) {
            *d = d.min(trap_distance(t, z));
        }
    }
    EscapeResult {
        iterations: params.max_iter,
        escaped: false,
        smooth: f64::from(params.max_iter),
        final_z: z,
        distance: None,
        orbit_trap: trap,
    }
}

/// Magnet fractal type I: z ← ((z² + c − 1)/(2z + c − 2))².
#[must_use]
pub fn magnet_type1(c: Complex, params: &EscapeParams) -> EscapeResult {
    magnet_escape(
        |z, c| {
            let num = sq(z) + c - Complex::new(1.0, 0.0);
            let den = cscale(z, 2.0) + c - Complex::new(2.0, 0.0);
            if den.norm_sq() == 0.0 { Complex::new(1e18, 0.0) } else { sq(num / den) }
        },
        c,
        params,
    )
}

/// Magnet fractal type II:
/// z ← ((z³ + 3(c−1)z + (c−1)(c−2)) / (3z² + 3(c−2)z + (c−1)(c−2) + 1))².
#[must_use]
pub fn magnet_type2(c: Complex, params: &EscapeParams) -> EscapeResult {
    magnet_escape(
        |z, c| {
            let cm1 = c - Complex::new(1.0, 0.0);
            let cm2 = c - Complex::new(2.0, 0.0);
            let num = z * z * z + cscale(cm1 * z, 3.0) + cm1 * cm2;
            let den = cscale(sq(z), 3.0) + cscale(cm2 * z, 3.0) + cm1 * cm2 + Complex::new(1.0, 0.0);
            if den.norm_sq() == 0.0 { Complex::new(1e18, 0.0) } else { sq(num / den) }
        },
        c,
        params,
    )
}

/// Lyapunov exponent of the forced logistic map x ← r·x(1−x) where r
/// alternates between `a` and `b` according to `sequence` (a string
/// of 'A's and 'B's, cycled). Negative values mark stability
/// (colored regions of Markus-Lyapunov fractals), positive chaos.
///
/// # Panics
/// Panics unless the sequence is non-empty and made of A/B, and
/// `iterations >= 1`.
#[must_use]
pub fn lyapunov_fractal(a: f64, b: f64, sequence: &str, iterations: usize, warmup: usize) -> f64 {
    assert!(iterations >= 1, "need at least one iteration");
    let rs: Vec<f64> = sequence
        .chars()
        .map(|ch| match ch {
            'A' | 'a' => a,
            'B' | 'b' => b,
            _ => panic!("Lyapunov sequence must be A/B"),
        })
        .collect();
    assert!(!rs.is_empty(), "Lyapunov sequence must be non-empty");
    // 0.49 rather than the critical point 0.5: starting exactly at
    // 0.5 lands on the superstable orbit 0.5 -> 1 -> 0 when a factor
    // is 4, poisoning the average.
    let mut x = 0.49;
    for i in 0..warmup {
        x = rs[i % rs.len()] * x * (1.0 - x);
    }
    let mut sum = 0.0;
    for i in 0..iterations {
        let r = rs[(warmup + i) % rs.len()];
        x = r * x * (1.0 - x);
        sum += (r * (1.0 - 2.0 * x)).abs().max(1e-300).ln();
    }
    sum / iterations as f64
}

/// Period of the attracting cycle at parameter c, by iterating to
/// the attractor and then measuring the first return within 1e-9.
/// `None` when the orbit escapes or no cycle of period <= 64 is
/// found within `max_iter` settling steps.
#[must_use]
pub fn mandelbrot_period(c: Complex, max_iter: u32) -> Option<u32> {
    let mut z = Complex::new(0.0, 0.0);
    for _ in 0..max_iter {
        z = sq(z) + c;
        if z.norm_sq() > 4.0 {
            return None;
        }
    }
    let anchor = z;
    for p in 1..=64u32 {
        z = sq(z) + c;
        if z.norm_sq() > 4.0 {
            return None;
        }
        if (z - anchor).norm() < 1e-9 {
            return Some(p);
        }
    }
    None
}

/// True inside the main cardioid, where the fixed point is
/// attracting: q(q + Re c − 1/4) < (Im c)²/4 with q = |c − 1/4|².
#[must_use]
pub fn mandelbrot_in_main_cardioid(c: Complex) -> bool {
    let xq = c.re - 0.25;
    let q = xq * xq + c.im * c.im;
    q * (q + xq) < 0.25 * c.im * c.im
}

/// True inside the period-2 bulb |c + 1| < 1/4.
#[must_use]
pub fn mandelbrot_in_period2_bulb(c: Complex) -> bool {
    let dx = c.re + 1.0;
    dx * dx + c.im * c.im < 0.0625
}

/// Buddhabrot: accumulates the escape orbits of random starting
/// parameters into a `res.0` × `res.1` grid over `bounds`
/// (row-major, x fastest).
///
/// # Panics
/// Panics on an empty grid or degenerate bounds.
#[must_use]
pub fn buddhabrot(
    samples: usize,
    max_iter: u32,
    res: (usize, usize),
    bounds: &Rect,
    rng: &mut Rng,
) -> Vec<u32> {
    assert!(res.0 > 0 && res.1 > 0, "grid must be non-empty");
    let size = bounds.max - bounds.min;
    assert!(size.x > 0.0 && size.y > 0.0, "bounds must have positive area");
    let mut grid = vec![0u32; res.0 * res.1];
    let mut orbit = Vec::with_capacity(max_iter as usize);
    for _ in 0..samples {
        let c = Complex::new(
            -2.5 + 4.0 * rng.next_f64(),
            -2.0 + 4.0 * rng.next_f64(),
        );
        orbit.clear();
        let mut z = Complex::new(0.0, 0.0);
        let mut escaped = false;
        for _ in 0..max_iter {
            z = sq(z) + c;
            orbit.push(z);
            if z.norm_sq() > 4.0 {
                escaped = true;
                break;
            }
        }
        if !escaped {
            continue;
        }
        for w in &orbit {
            let ix = ((w.re - bounds.min.x) / size.x * res.0 as f64).floor();
            let iy = ((w.im - bounds.min.y) / size.y * res.1 as f64).floor();
            if ix >= 0.0 && iy >= 0.0 && (ix as usize) < res.0 && (iy as usize) < res.1 {
                grid[iy as usize * res.0 + ix as usize] += 1;
            }
        }
    }
    grid
}

/// Random points near the Mandelbrot set boundary: rejection
/// sampling keeping parameters whose escape time falls in
/// [20, max_iter), i.e. neither deep exterior nor interior.
///
/// # Panics
/// Panics unless `n >= 1`.
#[must_use]
pub fn mandelbrot_boundary_points(n: usize, rng: &mut Rng) -> Vec<Complex> {
    assert!(n >= 1, "need at least one point");
    let params = EscapeParams { max_iter: 500, ..EscapeParams::default() };
    let mut out = Vec::with_capacity(n);
    let mut attempts = 0usize;
    while out.len() < n && attempts < n * 100_000 {
        attempts += 1;
        let c = Complex::new(
            -2.5 + 3.5 * rng.next_f64(),
            -1.5 + 3.0 * rng.next_f64(),
        );
        let r = mandelbrot(c, &params);
        if r.escaped && r.iterations >= 20 {
            out.push(c);
        }
    }
    out
}

/// Evaluates `f` at every pixel center of a `width` × `height` grid
/// over `bounds` (row-major, x fastest, y increasing upward).
///
/// # Panics
/// Panics on an empty grid.
#[must_use]
pub fn render_grid(
    f: &dyn Fn(Complex) -> EscapeResult,
    bounds: &Rect,
    width: usize,
    height: usize,
) -> Vec<EscapeResult> {
    assert!(width > 0 && height > 0, "grid must be non-empty");
    let size = bounds.max - bounds.min;
    let mut out = Vec::with_capacity(width * height);
    for j in 0..height {
        for i in 0..width {
            let c = Complex::new(
                bounds.min.x + (i as f64 + 0.5) / width as f64 * size.x,
                bounds.min.y + (j as f64 + 0.5) / height as f64 * size.y,
            );
            out.push(f(c));
        }
    }
    out
}

/// Supersampled smooth iteration counts: each pixel averages
/// `samples` × `samples` sub-pixel evaluations.
///
/// # Panics
/// Panics on an empty grid or `samples == 0`.
#[must_use]
pub fn render_grid_supersampled(
    f: &dyn Fn(Complex) -> EscapeResult,
    bounds: &Rect,
    width: usize,
    height: usize,
    samples: usize,
) -> Vec<f64> {
    assert!(width > 0 && height > 0, "grid must be non-empty");
    assert!(samples > 0, "need at least one sample per pixel");
    let size = bounds.max - bounds.min;
    let mut out = Vec::with_capacity(width * height);
    for j in 0..height {
        for i in 0..width {
            let mut acc = 0.0;
            for sj in 0..samples {
                for si in 0..samples {
                    let c = Complex::new(
                        bounds.min.x
                            + (i as f64 + (si as f64 + 0.5) / samples as f64) / width as f64
                                * size.x,
                        bounds.min.y
                            + (j as f64 + (sj as f64 + 0.5) / samples as f64) / height as f64
                                * size.y,
                    );
                    acc += f(c).smooth;
                }
            }
            out.push(acc / (samples * samples) as f64);
        }
    }
    out
}

/// Perturbation iteration for deep zooms: iterates the offset
/// δ ← 2·Z_n·δ + δ² + δ₀ against a precomputed reference orbit
/// Z_n (the orbit of `center_hi`), so pixels near the reference
/// need only f64 offsets. The reference orbit must start at
/// Z_0 = c_ref (the first iterate of 0). When the reference orbit
/// is shorter than the escape time, iteration continues directly.
///
/// # Panics
/// Panics on an empty reference orbit.
#[must_use]
pub fn perturbation_mandelbrot(
    center_hi: (f64, f64),
    delta: Complex,
    reference_orbit: &[Complex],
    params: &EscapeParams,
) -> EscapeResult {
    assert!(!reference_orbit.is_empty(), "need a reference orbit");
    let bb = params.bailout * params.bailout;
    let c_ref = Complex::new(center_hi.0, center_hi.1);
    let mut dz = delta;
    let mut i = 0u32;
    // Perturbed phase: z_n = Z_n + dz_n.
    while (i as usize) < reference_orbit.len() && i < params.max_iter {
        let z = reference_orbit[i as usize] + dz;
        if z.norm_sq() > bb {
            return EscapeResult {
                iterations: i,
                escaped: true,
                smooth: smooth_count(i, z),
                final_z: z,
                distance: None,
                orbit_trap: None,
            };
        }
        dz = cscale(reference_orbit[i as usize] * dz, 2.0) + sq(dz) + delta;
        i += 1;
    }
    // Direct continuation from the reconstructed value.
    let mut z = if (i as usize) < reference_orbit.len() {
        reference_orbit[i as usize] + dz
    } else {
        reference_orbit[reference_orbit.len() - 1] + dz
    };
    let c = c_ref + delta;
    while i < params.max_iter {
        if z.norm_sq() > bb {
            return EscapeResult {
                iterations: i,
                escaped: true,
                smooth: smooth_count(i, z),
                final_z: z,
                distance: None,
                orbit_trap: None,
            };
        }
        z = sq(z) + c;
        i += 1;
    }
    EscapeResult {
        iterations: params.max_iter,
        escaped: false,
        smooth: f64::from(params.max_iter),
        final_z: z,
        distance: None,
        orbit_trap: None,
    }
}

/// Applies a palette to the smooth iteration count (interior points
/// map to palette(0)).
#[must_use]
pub fn color_smooth_iter(r: &EscapeResult, palette: &dyn Fn(f64) -> [f64; 3]) -> [f64; 3] {
    if r.escaped { palette(r.smooth) } else { palette(0.0) }
}

/// Distance-estimate shading: 0 on the set, saturating to 1 at a
/// few pixels away; d/pixel_size clamped to [0, 1]. Interior points
/// (no distance) shade to 0.
///
/// # Panics
/// Panics unless `pixel_size > 0`.
#[must_use]
pub fn color_distance_estimate(r: &EscapeResult, pixel_size: f64) -> f64 {
    assert!(pixel_size > 0.0, "pixel size must be positive");
    r.distance.map_or(0.0, |d| (d / pixel_size).clamp(0.0, 1.0))
}

/// Reference orbit of the Mandelbrot iteration at c (Z_0 = c),
/// for [`perturbation_mandelbrot`]. Stops early on escape.
#[must_use]
pub fn mandelbrot_reference_orbit(c: Complex, max_iter: u32) -> Vec<Complex> {
    let mut out = Vec::with_capacity(max_iter as usize);
    let mut z = Complex::new(0.0, 0.0);
    for _ in 0..max_iter {
        z = sq(z) + c;
        out.push(z);
        if z.norm_sq() > 1e6 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec2;

    fn c(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    #[test]
    fn test_cardioid_and_bulb_interior() {
        let params = EscapeParams { max_iter: 10_000, ..EscapeParams::default() };
        for (re, im) in [(0.0, 0.0), (-0.1, 0.2), (0.2, 0.1), (-0.12, -0.4)] {
            assert!(mandelbrot_in_main_cardioid(c(re, im)), "({re}, {im}) in cardioid");
            assert!(!mandelbrot(c(re, im), &params).escaped);
        }
        for (re, im) in [(-1.0, 0.0), (-1.1, 0.1)] {
            assert!(mandelbrot_in_period2_bulb(c(re, im)));
            assert!(!mandelbrot(c(re, im), &params).escaped);
        }
        assert!(!mandelbrot_in_main_cardioid(c(-1.0, 0.0)));
        assert!(mandelbrot(c(0.3, 0.6), &params).escaped);
    }

    #[test]
    fn test_julia_unit_circle() {
        // c = 0: the filled Julia set is the closed unit disk.
        let params = EscapeParams { max_iter: 2000, ..EscapeParams::default() };
        for angle in 0..12 {
            let a = f64::from(angle) * 0.5;
            let inside = julia(c(0.9 * a.cos(), 0.9 * a.sin()), c(0.0, 0.0), &params);
            assert!(!inside.escaped, "|z| < 1 stays bounded");
            let outside = julia(c(1.1 * a.cos(), 1.1 * a.sin()), c(0.0, 0.0), &params);
            assert!(outside.escaped, "|z| > 1 escapes");
        }
    }

    #[test]
    fn test_distance_estimate_near_minus_two() {
        // The set contains the segment [-2, 0.25] of the real axis;
        // at c = -2 - d the true distance is d.
        let params = EscapeParams {
            max_iter: 5000,
            bailout: 100.0,
            compute_distance: true,
            trap: None,
        };
        for d in [0.01, 0.005, 0.001] {
            let r = mandelbrot(c(-2.0 - d, 0.0), &params);
            assert!(r.escaped);
            let est = r.distance.expect("distance requested");
            // The Koebe 1/4 bound puts the estimate within a factor
            // of ~2 of the truth for smooth boundaries; on the tip
            // it is within ~50%: the estimate halves ln|z|.
            assert!(
                est > 0.4 * d && est < 2.5 * d,
                "distance estimate {est} vs true {d}"
            );
        }
    }

    #[test]
    fn test_escape_time_deterministic_and_traps() {
        let params = EscapeParams {
            trap: Some(OrbitTrap::Point(c(0.0, 0.0))),
            ..EscapeParams::default()
        };
        let f = |z: Complex, cc: Complex| z * z + cc;
        let r1 = escape_time(&f, c(0.0, 0.0), c(0.3, 0.5), &params);
        let r2 = escape_time(&f, c(0.0, 0.0), c(0.3, 0.5), &params);
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.smooth, r2.smooth);
        let trap = r1.orbit_trap.expect("trap requested");
        assert!(trap >= 0.0 && trap.is_finite());
        // Cross trap on the real-axis orbit c = -1 is zero.
        let params2 = EscapeParams {
            trap: Some(OrbitTrap::Cross),
            max_iter: 50,
            ..EscapeParams::default()
        };
        let r = mandelbrot(c(-1.0, 0.0), &params2);
        assert!(r.orbit_trap.expect("trap") < 1e-12, "real orbit touches the axes");
    }

    #[test]
    fn test_tricorn_ship_phoenix_multibrot() {
        let params = EscapeParams::default();
        assert!(!tricorn(c(0.0, 0.0), &params).escaped);
        assert!(tricorn(c(2.0, 1.0), &params).escaped);
        assert!(!burning_ship(c(0.0, 0.0), &params).escaped);
        assert!(burning_ship(c(1.0, 1.0), &params).escaped);
        assert!(!multibrot(c(0.0, 0.0), 3.0, &params).escaped);
        assert!(multibrot(c(1.0, 0.5), 3.0, &params).escaped);
        // Phoenix with p = 0 reduces to Julia.
        let a = phoenix(c(0.3, 0.2), c(-0.4, 0.1), c(0.0, 0.0), &params);
        let b = julia(c(0.3, 0.2), c(-0.4, 0.1), &params);
        assert_eq!(a.iterations, b.iterations);
    }

    #[test]
    fn test_newton_and_nova() {
        // z^3 - 1: roots at 1 and e^{±2πi/3}.
        let poly = [c(-1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
        let params = EscapeParams { max_iter: 100, ..EscapeParams::default() };
        let (root_a, it_a) = newton_fractal(c(1.2, 0.1), &poly, &params);
        assert!(root_a < 3, "converged to a root");
        assert!(it_a < 100);
        let (root_b, _) = newton_fractal(c(-0.6, 0.7), &poly, &params);
        assert!(root_b < 3);
        assert_ne!(root_a, root_b, "different basins");
        // All three roots reachable and distinct.
        let roots = poly_roots(&poly);
        assert_eq!(roots.len(), 3);
        for r in &roots {
            assert!((poly_eval(&poly, *r)).norm() < 1e-9);
        }
        // Nova with c = 0, relax = 1 is plain Newton: converges fast.
        let r = nova_fractal(c(0.9, 0.3), c(0.0, 0.0), 3.0, 1.0, &params);
        assert!(r.escaped, "nova converges");
        assert!((cpow(r.final_z, 3.0) - c(1.0, 0.0)).norm() < 1e-6);
    }

    #[test]
    fn test_magnet_fractals() {
        let params = EscapeParams { max_iter: 300, ..EscapeParams::default() };
        // c = 1: the orbit is superstable at z = 0 for both types.
        assert!(!magnet_type1(c(1.0, 0.0), &params).escaped);
        assert!(magnet_type1(c(2.0, 0.5), &params).escaped, "past the boundary");
        assert!(!magnet_type2(c(1.0, 0.0), &params).escaped);
        assert!(!magnet_type2(c(1.8, 0.1), &params).escaped);
        assert!(magnet_type2(c(3.5, 2.5), &params).escaped, "exterior converges to 1");
    }

    #[test]
    fn test_lyapunov_known_values() {
        // Fixed point of the logistic map at r = 2.5: lambda = ln|2 - r|.
        let l = lyapunov_fractal(2.5, 2.5, "AB", 4000, 500);
        assert!((l - 0.5f64.ln()).abs() < 1e-6, "lambda {l}");
        // Fully chaotic r = 4: lambda = ln 2.
        let l4 = lyapunov_fractal(4.0, 4.0, "AB", 200_000, 100);
        assert!((l4 - 2.0f64.ln()).abs() < 0.05, "lambda {l4}");
        // The classic stable window at (3.4, 2.9) with sequence AB.
        let ls = lyapunov_fractal(3.4, 2.9, "AB", 5000, 500);
        assert!(ls < 0.0, "periodic regime is stable ({ls})");
    }

    #[test]
    fn test_periods() {
        assert_eq!(mandelbrot_period(c(0.0, 0.0), 2000), Some(1));
        assert_eq!(mandelbrot_period(c(-1.0, 0.0), 2000), Some(2));
        // Superstable period-3 parameter on the real axis.
        assert_eq!(mandelbrot_period(c(-1.754_877_666, 0.0), 4000), Some(3));
        assert_eq!(mandelbrot_period(c(0.5, 0.5), 2000), None, "escaping point");
    }

    #[test]
    fn test_buddhabrot_and_boundary() {
        let mut rng = Rng::new(31);
        let bounds = Rect { min: Vec2::new(-2.5, -2.0), max: Vec2::new(1.5, 2.0) };
        let grid = buddhabrot(4000, 200, (32, 32), &bounds, &mut rng);
        assert_eq!(grid.len(), 1024);
        assert!(grid.iter().any(|&h| h > 0), "orbits accumulate");
        let pts = mandelbrot_boundary_points(50, &mut rng);
        assert_eq!(pts.len(), 50);
        let params = EscapeParams { max_iter: 500, ..EscapeParams::default() };
        for p in &pts {
            let r = mandelbrot(*p, &params);
            assert!(r.escaped && r.iterations >= 20, "point near the boundary");
        }
    }

    #[test]
    fn test_render_grids() {
        let bounds = Rect { min: Vec2::new(-2.5, -1.5), max: Vec2::new(1.0, 1.5) };
        let params = EscapeParams::default();
        let grid = render_grid(&|cc| mandelbrot(cc, &params), &bounds, 24, 16);
        assert_eq!(grid.len(), 24 * 16);
        let interior = grid.iter().filter(|r| !r.escaped).count();
        assert!(interior > 0 && interior < grid.len());
        let smooth = render_grid_supersampled(&|cc| mandelbrot(cc, &params), &bounds, 12, 8, 2);
        assert_eq!(smooth.len(), 96);
        assert!(smooth.iter().all(|s| s.is_finite()));
        // Palette helpers.
        let r = mandelbrot(c(0.3, 0.6), &params);
        let rgb = color_smooth_iter(&r, &|t| [t, t * 0.5, 1.0 - t]);
        assert!(rgb[0] >= 0.0);
        let pd = EscapeParams { compute_distance: true, ..EscapeParams::default() };
        let rd = mandelbrot(c(-2.1, 0.0), &pd);
        assert!(color_distance_estimate(&rd, 0.01) > 0.0);
    }

    #[test]
    fn test_perturbation_matches_direct() {
        let center = (-0.7436438870371587, 0.1318259042053119); // deep zoom site
        let c_ref = c(center.0, center.1);
        let orbit = mandelbrot_reference_orbit(c_ref, 3000);
        let params = EscapeParams { max_iter: 3000, ..EscapeParams::default() };
        for (dx, dy) in [(1e-6, 0.0), (0.0, 1e-6), (-2e-6, 1.5e-6)] {
            let delta = c(dx, dy);
            let pert = perturbation_mandelbrot(center, delta, &orbit, &params);
            let direct = mandelbrot(c(center.0 + dx, center.1 + dy), &params);
            let diff = i64::from(pert.iterations).abs_diff(i64::from(direct.iterations));
            assert!(diff <= 2, "perturbation {} vs direct {}", pert.iterations, direct.iterations);
        }
    }
}
