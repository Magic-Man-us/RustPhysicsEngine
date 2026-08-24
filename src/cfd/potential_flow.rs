//! Incompressible potential flow: elementary singularities, complex
//! potentials, Joukowski and Karman-Trefftz airfoils, NACA sections, the
//! Hess-Smith panel method, thin-airfoil and lifting-line theory, a
//! simple vortex lattice, and added-mass results.

use crate::fractals::Complex;
use crate::linalg::{lu_decompose, Matrix};
use crate::math::Vec2;

const PI: f64 = crate::math::constants::PI;
const TWO_PI: f64 = 2.0 * PI;

/// A wall line for the method of images.
#[derive(Debug, Clone, Copy)]
pub struct Plane2 {
    pub point: Vec2,
    pub normal: Vec2,
}

/// Elementary potential-flow element.
#[derive(Debug, Clone, Copy)]
pub enum Element {
    /// Uniform stream of speed u at incidence alpha.
    Uniform { u: f64, alpha: f64 },
    /// Point source (strength m > 0) at pos.
    Source { m: f64, pos: Vec2 },
    /// Point vortex of circulation gamma (positive counterclockwise).
    Vortex { gamma: f64, pos: Vec2 },
    /// Doublet of strength kappa oriented at `angle`.
    Doublet { kappa: f64, pos: Vec2, angle: f64 },
}

fn cx(v: Vec2) -> Complex {
    Complex::new(v.x, v.y)
}

fn c_ln(z: Complex) -> Complex {
    Complex::new(z.norm().max(1e-300).ln(), z.arg())
}

fn c_exp(z: Complex) -> Complex {
    let e = z.re.exp();
    Complex::new(e * z.im.cos(), e * z.im.sin())
}

impl Element {
    /// Complex potential W(z).
    #[must_use]
    pub fn complex_potential(&self, z: Complex) -> Complex {
        match *self {
            Element::Uniform { u, alpha } => z * c_exp(Complex::new(0.0, -alpha)) * Complex::new(u, 0.0),
            Element::Source { m, pos } => c_ln(z - cx(pos)) * Complex::new(m / TWO_PI, 0.0),
            Element::Vortex { gamma, pos } => {
                c_ln(z - cx(pos)) * Complex::new(0.0, -gamma / TWO_PI)
            }
            Element::Doublet { kappa, pos, angle } => {
                let d = z - cx(pos);
                c_exp(Complex::new(0.0, angle)) * Complex::new(kappa / TWO_PI, 0.0) / d
            }
        }
    }

    /// Complex velocity dW/dz = u − i v.
    #[must_use]
    pub fn complex_velocity(&self, z: Complex) -> Complex {
        match *self {
            Element::Uniform { u, alpha } => c_exp(Complex::new(0.0, -alpha)) * Complex::new(u, 0.0),
            Element::Source { m, pos } => {
                Complex::new(m / TWO_PI, 0.0) / (z - cx(pos))
            }
            Element::Vortex { gamma, pos } => {
                Complex::new(0.0, -gamma / TWO_PI) / (z - cx(pos))
            }
            Element::Doublet { kappa, pos, angle } => {
                let d = z - cx(pos);
                Complex::new(-kappa / TWO_PI, 0.0) * c_exp(Complex::new(0.0, angle)) / (d * d)
            }
        }
    }
}

/// Closure form of a uniform stream: returns (velocity, potential,
/// stream function) at a point.
pub fn uniform_flow(u: f64, alpha: f64) -> impl Fn(Vec2) -> (Vec2, f64, f64) {
    move |p: Vec2| {
        let e = Element::Uniform { u, alpha };
        let w = e.complex_potential(cx(p));
        let cv = e.complex_velocity(cx(p));
        (Vec2::new(cv.re, -cv.im), w.re, w.im)
    }
}

/// Closure form of a point source.
pub fn source(strength: f64, pos: Vec2) -> impl Fn(Vec2) -> (Vec2, f64, f64) {
    move |p: Vec2| {
        let e = Element::Source { m: strength, pos };
        let w = e.complex_potential(cx(p));
        let cv = e.complex_velocity(cx(p));
        (Vec2::new(cv.re, -cv.im), w.re, w.im)
    }
}

/// Closure form of a sink (negative source).
pub fn sink(strength: f64, pos: Vec2) -> impl Fn(Vec2) -> (Vec2, f64, f64) {
    source(-strength, pos)
}

/// Closure form of a point vortex.
pub fn vortex(gamma: f64, pos: Vec2) -> impl Fn(Vec2) -> (Vec2, f64, f64) {
    move |p: Vec2| {
        let e = Element::Vortex { gamma, pos };
        let w = e.complex_potential(cx(p));
        let cv = e.complex_velocity(cx(p));
        (Vec2::new(cv.re, -cv.im), w.re, w.im)
    }
}

/// Closure form of a doublet.
pub fn doublet(kappa: f64, pos: Vec2, angle: f64) -> impl Fn(Vec2) -> (Vec2, f64, f64) {
    move |p: Vec2| {
        let e = Element::Doublet { kappa, pos, angle };
        let w = e.complex_potential(cx(p));
        let cv = e.complex_velocity(cx(p));
        (Vec2::new(cv.re, -cv.im), w.re, w.im)
    }
}

/// Superposition of potential-flow elements.
pub struct PotentialFlow2 {
    pub elements: Vec<Element>,
}

impl PotentialFlow2 {
    /// Velocity at a point.
    #[must_use]
    pub fn velocity(&self, p: Vec2) -> Vec2 {
        let cv = self.complex_velocity(cx(p));
        Vec2::new(cv.re, -cv.im)
    }

    /// Velocity potential φ.
    #[must_use]
    pub fn potential(&self, p: Vec2) -> f64 {
        self.complex_potential(cx(p)).re
    }

    /// Stream function ψ.
    #[must_use]
    pub fn stream_function(&self, p: Vec2) -> f64 {
        self.complex_potential(cx(p)).im
    }

    /// Complex potential W(z).
    #[must_use]
    pub fn complex_potential(&self, z: Complex) -> Complex {
        self.elements
            .iter()
            .fold(Complex::new(0.0, 0.0), |acc, e| acc + e.complex_potential(z))
    }

    /// Complex velocity dW/dz.
    #[must_use]
    pub fn complex_velocity(&self, z: Complex) -> Complex {
        self.elements
            .iter()
            .fold(Complex::new(0.0, 0.0), |acc, e| acc + e.complex_velocity(z))
    }

    /// Pressure coefficient 1 − |v|²/U∞².
    #[must_use]
    pub fn pressure_coefficient(&self, p: Vec2, u_inf: f64) -> f64 {
        1.0 - self.velocity(p).magnitude_squared() / (u_inf * u_inf)
    }

    /// Stagnation points found by Newton iteration on dW/dz = 0 from a
    /// grid of starts.
    #[must_use]
    pub fn stagnation_points(&self) -> Vec<Vec2> {
        let mut found: Vec<Vec2> = Vec::new();
        for j in 0..14 {
            for i in 0..14 {
                let mut z = Complex::new(
                    -3.5 + 7.0 * i as f64 / 13.0,
                    -3.5 + 7.0 * j as f64 / 13.0,
                );
                let mut ok = false;
                for _ in 0..60 {
                    let v = self.complex_velocity(z);
                    if v.norm() < 1e-11 {
                        ok = true;
                        break;
                    }
                    // Numerical derivative of the complex velocity.
                    let h = Complex::new(1e-6, 0.0);
                    let dv = (self.complex_velocity(z + h) - v) / h;
                    if dv.norm() < 1e-14 {
                        break;
                    }
                    // Damped Newton: backtrack until |v| decreases (the
                    // far field has v → U∞ with vanishing derivative, so
                    // undamped steps fly off).
                    let mut step = v / dv;
                    let mut accepted = false;
                    for _ in 0..25 {
                        let trial = z - step;
                        if self.complex_velocity(trial).norm() < v.norm() {
                            z = trial;
                            accepted = true;
                            break;
                        }
                        step = step * Complex::new(0.5, 0.0);
                    }
                    if !accepted || z.norm() > 50.0 {
                        break;
                    }
                }
                if ok && found.iter().all(|f| (f.x - z.re).abs() + (f.y - z.im).abs() > 1e-5) {
                    found.push(Vec2::new(z.re, z.im));
                }
            }
        }
        found
    }

    /// Trace streamlines by RK2.
    #[must_use]
    pub fn streamlines(&self, seeds: &[Vec2], steps: usize, dt: f64) -> Vec<Vec<Vec2>> {
        seeds
            .iter()
            .map(|&s| {
                let mut p = s;
                let mut line = vec![p];
                for _ in 0..steps {
                    let v1 = self.velocity(p);
                    let v2 = self.velocity(p + v1 * (0.5 * dt));
                    p = p + v2 * dt;
                    line.push(p);
                }
                line
            })
            .collect()
    }

    /// Kutta-Joukowski lift per unit span L' = ρ U Γ (sum of vortex
    /// circulations).
    #[must_use]
    pub fn lift_kutta_joukowski(&self, u_inf: f64, rho: f64) -> f64 {
        let gamma: f64 = self
            .elements
            .iter()
            .map(|e| if let Element::Vortex { gamma, .. } = e { *gamma } else { 0.0 })
            .sum();
        -rho * u_inf * gamma
    }
}

/// Flow past a circular cylinder with circulation Γ.
#[must_use]
pub fn cylinder_flow(u_inf: f64, r: f64, gamma: f64) -> PotentialFlow2 {
    PotentialFlow2 {
        elements: vec![
            Element::Uniform { u: u_inf, alpha: 0.0 },
            Element::Doublet { kappa: TWO_PI * u_inf * r * r, pos: Vec2::ZERO, angle: 0.0 },
            Element::Vortex { gamma, pos: Vec2::ZERO },
        ],
    }
}

/// Rankine oval: source and sink of strength m at (±a, 0) in a stream.
#[must_use]
pub fn rankine_oval(u: f64, m: f64, a: f64) -> PotentialFlow2 {
    PotentialFlow2 {
        elements: vec![
            Element::Uniform { u, alpha: 0.0 },
            Element::Source { m, pos: Vec2::new(-a, 0.0) },
            Element::Source { m: -m, pos: Vec2::new(a, 0.0) },
        ],
    }
}

/// Exact surface pressure coefficient of the rotating cylinder
/// (Γ counterclockwise-positive, matching [`Element::Vortex`]).
#[must_use]
pub fn cylinder_cp_exact(theta: f64, gamma: f64, u: f64, r: f64) -> f64 {
    let term = 2.0 * theta.sin() - gamma / (TWO_PI * r * u);
    1.0 - term * term
}

/// Joukowski map ζ = z + c²/z.
#[must_use]
pub fn joukowski_transform(z: Complex, c: f64) -> Complex {
    z + Complex::new(c * c, 0.0) / z
}

/// Inverse Joukowski map (branch with |z| ≥ c).
#[must_use]
pub fn inverse_joukowski(zeta: Complex, c: f64) -> Complex {
    let half = zeta * Complex::new(0.5, 0.0);
    let disc = half * half - Complex::new(c * c, 0.0);
    // Complex square root.
    let r = disc.norm().sqrt();
    let th = disc.arg() / 2.0;
    let sq = Complex::new(r * th.cos(), r * th.sin());
    let z1 = half + sq;
    let z2 = half - sq;
    if z1.norm() >= z2.norm() { z1 } else { z2 }
}

/// Joukowski airfoil: image of the circle through (c, 0) centered at
/// `center`.
#[must_use]
pub fn joukowski_airfoil(center: Complex, c: f64, n_points: usize) -> Vec<Vec2> {
    let rel = Complex::new(c, 0.0) - center;
    let r = rel.norm();
    let th0 = rel.arg(); // start at the trailing-edge preimage
    (0..n_points)
        .map(|k| {
            let th = th0 + TWO_PI * k as f64 / n_points as f64;
            let z = center + Complex::new(r * th.cos(), r * th.sin());
            let zeta = joukowski_transform(z, c);
            Vec2::new(zeta.re, zeta.im)
        })
        .collect()
}

/// Joukowski airfoil with the Kutta condition: returns (surface points,
/// surface cp, lift coefficient).
#[must_use]
pub fn joukowski_airfoil_flow(
    center: Complex,
    c: f64,
    alpha: f64,
    u_inf: f64,
) -> (Vec<Vec2>, Vec<f64>, f64) {
    let r = (Complex::new(c, 0.0) - center).norm();
    // Kutta condition: rear stagnation point at the trailing-edge
    // preimage z = c.
    // TE preimage at angle −β on the circle: sin β = y_c/R.
    let beta = (center.im / r).asin();
    // Clockwise circulation (negative in the ccw-positive convention).
    let gamma = -4.0 * PI * u_inf * r * (alpha + beta).sin();
    let n = 100;
    let mut pts = Vec::with_capacity(n);
    let mut cps = Vec::with_capacity(n);
    for k in 0..n {
        let th = TWO_PI * (k as f64 + 0.5) / n as f64;
        let z = center + Complex::new(r * th.cos(), r * th.sin());
        let zeta = joukowski_transform(z, c);
        pts.push(Vec2::new(zeta.re, zeta.im));
        // Velocity on the circle mapped through dζ/dz.
        let zc = z - center;
        let w_circle = c_exp(Complex::new(0.0, -alpha)) * Complex::new(u_inf, 0.0)
            - c_exp(Complex::new(0.0, alpha)) * Complex::new(u_inf, 0.0)
                * Complex::new(r * r, 0.0)
                / (zc * zc)
            + Complex::new(0.0, -gamma / TWO_PI) / zc;
        let dzeta_dz = Complex::new(1.0, 0.0) - Complex::new(c * c, 0.0) / (z * z);
        let v = if dzeta_dz.norm() > 1e-6 {
            (w_circle / dzeta_dz).norm()
        } else {
            0.0
        };
        cps.push(1.0 - (v / u_inf).powi(2));
    }
    // Chord from the surface extent.
    let x_min = pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let x_max = pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let chord = x_max - x_min;
    let cl = -2.0 * gamma / (u_inf * chord);
    (pts, cps, cl)
}

/// Karman-Trefftz airfoil (finite trailing-edge angle set by `n_exp`
/// slightly below 2).
#[must_use]
pub fn karman_trefftz_airfoil(center: Complex, c: f64, n_exp: f64, n_points: usize) -> Vec<Vec2> {
    let rel = Complex::new(c, 0.0) - center;
    let r = rel.norm();
    let th0 = rel.arg();
    let cpow = |z: Complex, p: f64| -> Complex {
        let rr = z.norm().max(1e-300).powf(p);
        let th = z.arg() * p;
        Complex::new(rr * th.cos(), rr * th.sin())
    };
    (0..n_points)
        .map(|k| {
            let th = th0 + TWO_PI * k as f64 / n_points as f64;
            let z = center + Complex::new(r * th.cos(), r * th.sin());
            let a = cpow((z - Complex::new(c, 0.0)) / (z + Complex::new(c, 0.0)), n_exp);
            let zeta = Complex::new(n_exp * c, 0.0) * (Complex::new(1.0, 0.0) + a)
                / (Complex::new(1.0, 0.0) - a);
            Vec2::new(zeta.re, zeta.im)
        })
        .collect()
}

/// NACA 4-digit airfoil ("2412" etc.), chord 1, from the trailing edge
/// over the top and back along the bottom.
#[must_use]
pub fn naca4(code: &str, n_points: usize, closed_te: bool) -> Vec<Vec2> {
    let digits: Vec<u32> = code.chars().filter_map(|c| c.to_digit(10)).collect();
    assert_eq!(digits.len(), 4, "NACA 4-digit code");
    let m = digits[0] as f64 / 100.0;
    let p = digits[1] as f64 / 10.0;
    let t = (digits[2] * 10 + digits[3]) as f64 / 100.0;
    let a4 = if closed_te { -0.1036 } else { -0.1015 };
    let half = n_points / 2;
    let thickness = |x: f64| -> f64 {
        5.0 * t
            * (0.2969 * x.sqrt() - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x
                + a4 * x * x * x * x)
    };
    let camber = |x: f64| -> (f64, f64) {
        if m == 0.0 || p == 0.0 {
            (0.0, 0.0)
        } else if x < p {
            (m / (p * p) * (2.0 * p * x - x * x), 2.0 * m / (p * p) * (p - x))
        } else {
            (
                m / (1.0 - p).powi(2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x),
                2.0 * m / (1.0 - p).powi(2) * (p - x),
            )
        }
    };
    let mut pts = Vec::with_capacity(2 * half);
    // Upper surface TE → LE (cosine spacing).
    for k in 0..half {
        let x = 0.5 * (1.0 + (PI * k as f64 / (half - 1) as f64).cos());
        let (yc, dyc) = camber(x);
        let th = dyc.atan();
        let yt = thickness(x);
        pts.push(Vec2::new(x - yt * th.sin(), yc + yt * th.cos()));
    }
    // Lower surface LE → TE.
    for k in 1..half {
        let x = 0.5 * (1.0 - (PI * k as f64 / (half - 1) as f64).cos());
        let (yc, dyc) = camber(x);
        let th = dyc.atan();
        let yt = thickness(x);
        pts.push(Vec2::new(x + yt * th.sin(), yc - yt * th.cos()));
    }
    pts
}

/// NACA 5-digit airfoil ("23012" etc.).
#[must_use]
pub fn naca5(code: &str, n_points: usize) -> Vec<Vec2> {
    let digits: Vec<u32> = code.chars().filter_map(|c| c.to_digit(10)).collect();
    assert_eq!(digits.len(), 5, "NACA 5-digit code");
    let t = (digits[3] * 10 + digits[4]) as f64 / 100.0;
    // Standard camber-line constants for the 210..250 series.
    let series = digits[0] * 100 + digits[1] * 10 + digits[2];
    let (r_c, k1) = match series {
        210 => (0.0580, 361.4),
        220 => (0.1260, 51.64),
        230 => (0.2025, 15.957),
        240 => (0.2900, 6.643),
        250 => (0.3910, 3.230),
        _ => (0.2025, 15.957),
    };
    let half = n_points / 2;
    let thickness = |x: f64| -> f64 {
        5.0 * t
            * (0.2969 * x.sqrt() - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x
                - 0.1015 * x * x * x * x)
    };
    let camber = |x: f64| -> (f64, f64) {
        if x < r_c {
            (
                k1 / 6.0 * (x.powi(3) - 3.0 * r_c * x * x + r_c * r_c * (3.0 - r_c) * x),
                k1 / 6.0 * (3.0 * x * x - 6.0 * r_c * x + r_c * r_c * (3.0 - r_c)),
            )
        } else {
            (k1 * r_c.powi(3) / 6.0 * (1.0 - x), -k1 * r_c.powi(3) / 6.0)
        }
    };
    let mut pts = Vec::with_capacity(2 * half);
    for k in 0..half {
        let x = 0.5 * (1.0 + (PI * k as f64 / (half - 1) as f64).cos());
        let (yc, dyc) = camber(x);
        let th = dyc.atan();
        let yt = thickness(x);
        pts.push(Vec2::new(x - yt * th.sin(), yc + yt * th.cos()));
    }
    for k in 1..half {
        let x = 0.5 * (1.0 - (PI * k as f64 / (half - 1) as f64).cos());
        let (yc, dyc) = camber(x);
        let th = dyc.atan();
        let yt = thickness(x);
        pts.push(Vec2::new(x + yt * th.sin(), yc - yt * th.cos()));
    }
    pts
}

// --- Panel method --------------------------------------------------------

struct Panel {
    a: Vec2,
    mid: Vec2,
    len: f64,
    /// Panel angle: tangent direction.
    theta: f64,
}

/// Hess-Smith source/vortex panel method.
pub struct PanelMethod {
    panels: Vec<Panel>,
    pub alpha: f64,
    pub u_inf: f64,
    /// Source strength per panel (after solve).
    pub sources: Vec<f64>,
    /// Global vortex strength per unit length (after solve).
    pub gamma: f64,
}

impl PanelMethod {
    /// Build panels from airfoil surface points (TE → upper → LE →
    /// lower → TE ordering as produced by [`naca4`]).
    #[must_use]
    pub fn new(airfoil: &[Vec2]) -> Self {
        let n = airfoil.len();
        let mut panels = Vec::with_capacity(n);
        for k in 0..n {
            let a = airfoil[k];
            let b = airfoil[(k + 1) % n];
            let d = b - a;
            let len = d.magnitude();
            if len < 1e-12 {
                continue;
            }
            panels.push(Panel {
                a,
                mid: (a + b) * 0.5,
                len,
                theta: d.y.atan2(d.x),
            });
        }
        Self { panels, alpha: 0.0, u_inf: 1.0, sources: Vec::new(), gamma: 0.0 }
    }

    /// Velocity induced at `p` by unit source and unit vortex densities
    /// on panel `j` (local integrals).
    fn induced(&self, j: usize, p: Vec2) -> (Vec2, Vec2) {
        let pan = &self.panels[j];
        // Local panel coordinates.
        let (ct, st) = (pan.theta.cos(), pan.theta.sin());
        let rel = p - pan.a;
        let x = rel.x * ct + rel.y * st;
        let y = -rel.x * st + rel.y * ct;
        let l = pan.len;
        let r1_sq = (x * x + y * y).max(1e-20);
        let r2_sq = ((x - l) * (x - l) + y * y).max(1e-20);
        let th1 = y.atan2(x);
        let th2 = y.atan2(x - l);
        let dth = th2 - th1;
        // Local induced velocities per unit density (vortex density is
        // counterclockwise-positive).
        let us = -(r2_sq / r1_sq).ln() / (2.0 * TWO_PI); // (1/2π) ln(r1/r2)
        let vs = dth / TWO_PI;
        let uv = -dth / TWO_PI;
        let vv = -(r2_sq / r1_sq).ln() / (2.0 * TWO_PI);
        // Rotate back to global coordinates.
        let rot = |u: f64, v: f64| Vec2::new(u * ct - v * st, u * st + v * ct);
        (rot(us, vs), rot(uv, vv))
    }

    /// Induced velocity at the midpoint of panel i from panel j, taking
    /// the exterior-limit self terms analytically (points are ordered
    /// counterclockwise, so the fluid lies on the local −y side).
    fn induced_surface(&self, j: usize, i: usize) -> (Vec2, Vec2) {
        if i == j {
            let pan = &self.panels[i];
            let ni = Vec2::new(-pan.theta.sin(), pan.theta.cos());
            let t = Vec2::new(pan.theta.cos(), pan.theta.sin());
            // Exterior limits (fluid on the local −y side): source
            // −σ/2 along the interior normal, ccw vortex +γ/2 along the
            // tangent.
            (ni * -0.5, t * 0.5)
        } else {
            self.induced(j, self.panels[i].mid)
        }
    }

    /// Assemble and solve the Hess-Smith system: no-penetration source
    /// strengths for two trial vortex strengths, then the Kutta
    /// condition (equal trailing-edge tangential speeds evaluated
    /// through the surface-velocity operator) fixes γ by linearity.
    pub fn solve(&mut self) {
        let n = self.panels.len();
        let mut a = Matrix::zeros(n, n);
        let mut b_vortex = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        let free = Vec2::new(self.u_inf * self.alpha.cos(), self.u_inf * self.alpha.sin());
        for i in 0..n {
            let ni = Vec2::new(-self.panels[i].theta.sin(), self.panels[i].theta.cos());
            for j in 0..n {
                let (vs, vv) = self.induced_surface(j, i);
                a.set(i, j, vs.dot(&ni));
                b_vortex[i] += vv.dot(&ni);
            }
            rhs[i] = -free.dot(&ni);
        }
        let lu = lu_decompose(&a).expect("panel system solvable");
        // γ = 0 solution and the sensitivity to γ = 1.
        let sigma0 = lu.solve(&rhs).expect("panel solve");
        let rhs1: Vec<f64> = rhs.iter().zip(&b_vortex).map(|(r, b)| r - b).collect();
        let sigma1 = lu.solve(&rhs1).expect("panel solve");
        // Kutta functional f(γ) = V·t_first + V·t_last, linear in γ.
        let kutta = |sources: &[f64], gamma: f64, s: &Self| -> f64 {
            let mut total = 0.0;
            for &i in &[0usize, n - 1] {
                let pan = &s.panels[i];
                let tan = Vec2::new(pan.theta.cos(), pan.theta.sin());
                let mut vt = free.dot(&tan);
                for (j, &sj) in sources.iter().enumerate() {
                    let (vs, vv) = s.induced_surface(j, i);
                    vt += (vs * sj + vv * gamma).dot(&tan);
                }
                total += vt;
            }
            total
        };
        let f0 = kutta(&sigma0, 0.0, self);
        let f1 = kutta(&sigma1, 1.0, self);
        let gamma = if (f1 - f0).abs() > 1e-14 { -f0 / (f1 - f0) } else { 0.0 };
        self.gamma = gamma;
        self.sources = sigma0
            .iter()
            .zip(&sigma1)
            .map(|(s0, s1)| s0 + gamma * (s1 - s0))
            .collect();
    }

    /// Velocity at an arbitrary field point.
    #[must_use]
    pub fn velocity_at(&self, p: Vec2) -> Vec2 {
        let mut v = Vec2::new(self.u_inf * self.alpha.cos(), self.u_inf * self.alpha.sin());
        for j in 0..self.panels.len() {
            let (vs, vv) = self.induced(j, p);
            v = v + vs * self.sources[j] + vv * self.gamma;
        }
        v
    }

    /// Surface (x/c, cp) at panel midpoints.
    #[must_use]
    pub fn cp_distribution(&self) -> Vec<(f64, f64)> {
        self.panels
            .iter()
            .enumerate()
            .map(|(i, pan)| {
                let tan = Vec2::new(pan.theta.cos(), pan.theta.sin());
                let mut vt = Vec2::new(
                    self.u_inf * self.alpha.cos(),
                    self.u_inf * self.alpha.sin(),
                )
                .dot(&tan);
                for j in 0..self.panels.len() {
                    let (vs, vv) = self.induced_surface(j, i);
                    vt += (vs * self.sources[j] + vv * self.gamma).dot(&tan);
                }
                (pan.mid.x, 1.0 - (vt / self.u_inf).powi(2))
            })
            .collect()
    }

    /// Lift coefficient from the total circulation.
    #[must_use]
    pub fn cl(&self) -> f64 {
        let total_len: f64 = self.panels.iter().map(|p| p.len).sum();
        let x_min = self.panels.iter().map(|p| p.mid.x).fold(f64::INFINITY, f64::min);
        let x_max = self.panels.iter().map(|p| p.mid.x).fold(f64::NEG_INFINITY, f64::max);
        let chord = (x_max - x_min).max(1e-9);
        // Counterclockwise panel ordering makes positive lift correspond
        // to negative sheet strength.
        -2.0 * self.gamma * total_len / (self.u_inf * chord)
    }

    /// Quarter-chord moment coefficient from the cp distribution.
    #[must_use]
    pub fn cm_quarter_chord(&self) -> f64 {
        let cps = self.cp_distribution();
        let mut cm = 0.0;
        for (k, pan) in self.panels.iter().enumerate() {
            let cp = cps[k].1;
            let ni = Vec2::new(-pan.theta.sin(), pan.theta.cos());
            // ni points into the body, so the surface force is +cp ni.
            let f = ni * (cp * pan.len);
            let arm = pan.mid - Vec2::new(0.25, 0.0);
            cm += arm.x * f.y - arm.y * f.x;
        }
        cm
    }

    /// Streamlines around the airfoil.
    #[must_use]
    pub fn streamlines(&self, seeds: &[Vec2], steps: usize, dt: f64) -> Vec<Vec<Vec2>> {
        seeds
            .iter()
            .map(|&s| {
                let mut p = s;
                let mut line = vec![p];
                for _ in 0..steps {
                    let v = self.velocity_at(p);
                    p = p + v * dt;
                    line.push(p);
                }
                line
            })
            .collect()
    }

    /// Center of pressure x/c from the cp distribution.
    #[must_use]
    pub fn pressure_center(&self) -> f64 {
        let cps = self.cp_distribution();
        let mut fy = 0.0;
        let mut mx = 0.0;
        for (k, pan) in self.panels.iter().enumerate() {
            let cp = cps[k].1;
            let ni = Vec2::new(-pan.theta.sin(), pan.theta.cos());
            let f = ni * (cp * pan.len);
            fy += f.y;
            mx += pan.mid.x * f.y;
        }
        if fy.abs() > 1e-12 { mx / fy } else { 0.25 }
    }
}

// --- Airfoil/wing theory -------------------------------------------------

/// Thin-airfoil lift coefficient for a camber-line slope dz/dx given on
/// x ∈ [0, 1]: cl = 2π(α − α_L0).
#[must_use]
pub fn thin_airfoil_cl(alpha: f64, camber_slope: &dyn Fn(f64) -> f64) -> f64 {
    // α_L0 = −(1/π) ∫ (dz/dx)(cosθ − 1) dθ.
    let n = 400;
    let mut integral = 0.0;
    for k in 0..n {
        let th = PI * (k as f64 + 0.5) / n as f64;
        let x = 0.5 * (1.0 - th.cos());
        integral += camber_slope(x) * (th.cos() - 1.0) * PI / n as f64;
    }
    let alpha_l0 = -integral / PI;
    TWO_PI * (alpha - alpha_l0)
}

/// Flat-plate thin-airfoil lift 2πα.
#[must_use]
pub fn thin_airfoil_cl_flat(alpha: f64) -> f64 {
    TWO_PI * alpha
}

/// Prandtl lifting line with a Fourier sine series: returns
/// (CL, CDi, circulation at the collocation stations).
#[must_use]
pub fn lifting_line(
    span: f64,
    chord: &dyn Fn(f64) -> f64,
    alpha: &dyn Fn(f64) -> f64,
    n_terms: usize,
    u_inf: f64,
) -> (f64, f64, Vec<f64>) {
    let n = n_terms;
    let mut a = Matrix::zeros(n, n);
    let mut rhs = vec![0.0; n];
    let stations: Vec<f64> = (0..n).map(|k| PI * (k as f64 + 0.5) / n as f64).collect();
    for (i, &th) in stations.iter().enumerate() {
        let y = -0.5 * span * th.cos();
        let c = chord(y);
        let mu = c * PI / (2.0 * span); // πc/(2b) with a0 = 2π
        for j in 0..n {
            let nn = (j + 1) as f64;
            a.set(
                i,
                j,
                (nn * th).sin() * (nn * mu / th.sin() + 1.0),
            );
        }
        rhs[i] = mu * alpha(y);
    }
    let lu = lu_decompose(&a).expect("lifting line system");
    let coefs = lu.solve(&rhs).expect("lifting line solve");
    let ar = span * span / {
        // Mean chord for the reference area.
        let m = 64;
        let s: f64 = (0..m)
            .map(|k| chord(-0.5 * span + span * (k as f64 + 0.5) / m as f64))
            .sum::<f64>()
            / m as f64;
        s * span
    };
    let cl = PI * ar * coefs[0];
    let delta: f64 = coefs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(j, &c)| (j + 1) as f64 * (c / coefs[0].max(1e-300)).powi(2))
        .sum();
    let cdi = cl * cl / (PI * ar) * (1.0 + delta);
    let circulation: Vec<f64> = stations
        .iter()
        .map(|&th| {
            2.0 * span
                * u_inf
                * coefs
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| c * ((j + 1) as f64 * th).sin())
                    .sum::<f64>()
        })
        .collect();
    (cl, cdi, circulation)
}

/// Elliptic-wing lift slope: CL = 2πα/(1 + 2/AR).
#[must_use]
pub fn elliptic_wing_cl(ar: f64, alpha: f64) -> f64 {
    TWO_PI * alpha / (1.0 + 2.0 / ar)
}

/// Induced drag CL²/(π e AR).
#[must_use]
pub fn induced_drag(cl: f64, ar: f64, e: f64) -> f64 {
    cl * cl / (PI * e * ar)
}

/// Raymer's straight-wing Oswald efficiency estimate with a sweep
/// correction.
#[must_use]
pub fn oswald_efficiency_estimate(ar: f64, sweep: f64) -> f64 {
    if sweep.abs() < 0.05 {
        1.78 * (1.0 - 0.045 * ar.powf(0.68)) - 0.64
    } else {
        4.61 * (1.0 - 0.045 * ar.powf(0.68)) * sweep.cos().powf(0.15) - 3.1
    }
}

/// Simple wing planform for the vortex lattice.
#[derive(Debug, Clone, Copy)]
pub struct WingGeometry {
    pub span: f64,
    pub root_chord: f64,
    pub tip_chord: f64,
    /// Quarter-chord sweep (radians).
    pub sweep: f64,
}

/// Single-lattice-row vortex lattice method (horseshoe vortices at the
/// quarter chord, collocation at 3/4 chord): returns (CL, CDi,
/// circulation per strip).
#[must_use]
pub fn vortex_lattice(
    wing: &WingGeometry,
    alpha: f64,
    n_span: usize,
    _n_chord: usize,
    u_inf: f64,
) -> (f64, f64, Vec<f64>) {
    let b = wing.span;
    let n = n_span;
    let dy = b / n as f64;
    // Strip geometry.
    let chord_at = |y: f64| -> f64 {
        let frac = (2.0 * y / b).abs();
        wing.root_chord + (wing.tip_chord - wing.root_chord) * frac
    };
    let quarter_x = |y: f64| -> f64 { 0.25 * chord_at(y) + y.abs() * wing.sweep.tan() };
    // Horseshoe induced downwash at collocation points (Biot-Savart of
    // the bound leg + two trailing legs).
    let induced_w = |xc: f64, yc: f64, x1: f64, y1: f64, x2: f64, y2: f64| -> f64 {
        // Unit-circulation horseshoe in the z = 0 plane; w is the
        // z-velocity at (xc, yc).
        let seg = |xa: f64, ya: f64, xb: f64, yb: f64| -> f64 {
            // w = 1/(4π) · (r1 × r2)_z/|r1 × r2|² · (L·r̂1 − L·r̂2) for a
            // unit-circulation segment in the lattice plane.
            let (r1x, r1y) = (xc - xa, yc - ya);
            let (r2x, r2y) = (xc - xb, yc - yb);
            let cross = r1x * r2y - r1y * r2x;
            if cross.abs() < 1e-9 {
                return 0.0;
            }
            let r1 = (r1x * r1x + r1y * r1y).sqrt();
            let r2 = (r2x * r2x + r2y * r2y).sqrt();
            let dot1 = (xb - xa) * r1x + (yb - ya) * r1y;
            let dot2 = (xb - xa) * r2x + (yb - ya) * r2y;
            1.0 / (2.0 * TWO_PI) / cross * (dot1 / r1 - dot2 / r2)
        };
        // Bound leg + trailing legs to downstream infinity.
        let far = 1e4;
        seg(x1, y1, x2, y2) + seg(x2, y2, far, y2) + seg(far, y1, x1, y1)
    };
    let mut a = Matrix::zeros(n, n);
    let mut rhs = vec![0.0; n];
    let ys: Vec<f64> = (0..n).map(|k| -0.5 * b + (k as f64 + 0.5) * dy).collect();
    for (i, &yc) in ys.iter().enumerate() {
        let xc = quarter_x(yc) + 0.5 * chord_at(yc); // 3/4 chord
        for (j, &yj) in ys.iter().enumerate() {
            let (y1, y2) = (yj - 0.5 * dy, yj + 0.5 * dy);
            let xq = quarter_x(yj);
            a.set(i, j, induced_w(xc, yc, xq, y1, xq, y2));
        }
        rhs[i] = -u_inf * alpha;
    }
    let lu = lu_decompose(&a).expect("VLM system");
    let gammas = lu.solve(&rhs).expect("VLM solve");
    let s_ref: f64 = ys.iter().map(|&y| chord_at(y) * dy).sum();
    let lift: f64 = gammas.iter().sum::<f64>() * dy * u_inf; // ρ = 1
    let cl = 2.0 * lift / (u_inf * u_inf * s_ref);
    // Induced drag from the downwash at the bound vortices.
    let mut di = 0.0;
    for (i, &yc) in ys.iter().enumerate() {
        let mut w = 0.0;
        for (j, &yj) in ys.iter().enumerate() {
            let (y1, y2) = (yj - 0.5 * dy, yj + 0.5 * dy);
            let xq = quarter_x(yj);
            w += gammas[j] * induced_w(quarter_x(yc), yc, xq, y1, xq, y2);
        }
        di += -w * gammas[i] * dy;
    }
    let cdi = 2.0 * di / (u_inf * u_inf * s_ref);
    (cl, cdi, gammas)
}

/// McCormick ground-effect induced-drag factor (16 h/b)²/(1 + (16 h/b)²).
#[must_use]
pub fn ground_effect_factor(h_over_b: f64) -> f64 {
    let x = 16.0 * h_over_b;
    x * x / (1.0 + x * x)
}

/// Velocity of a base flow seen through a conformal map at the physical
/// point z (numerical dW/dζ via the chain rule).
#[must_use]
pub fn conformal_map_flow(
    map: &dyn Fn(Complex) -> Complex,
    base: &PotentialFlow2,
    z: Complex,
) -> Vec2 {
    // v(ζ)* = dW/dz · dz/dζ⁻¹ with ζ = map(z).
    let h = 1e-6;
    let dmap = (map(z + Complex::new(h, 0.0)) - map(z - Complex::new(h, 0.0)))
        / Complex::new(2.0 * h, 0.0);
    let w = base.complex_velocity(map(z));
    let v = w * dmap;
    Vec2::new(v.re, -v.im)
}

/// Mirror every element across a wall (method of images).
#[must_use]
pub fn method_of_images_wall(elements: &[Element], wall: Plane2) -> PotentialFlow2 {
    let n = wall.normal.normalized();
    let mirror = |p: Vec2| -> Vec2 {
        let d = (p - wall.point).dot(&n);
        p - n * (2.0 * d)
    };
    let mut all = elements.to_vec();
    for e in elements {
        all.push(match *e {
            Element::Uniform { u, alpha } => Element::Uniform { u, alpha: -alpha },
            Element::Source { m, pos } => Element::Source { m, pos: mirror(pos) },
            Element::Vortex { gamma, pos } => {
                Element::Vortex { gamma: -gamma, pos: mirror(pos) }
            }
            Element::Doublet { kappa, pos, angle } => {
                Element::Doublet { kappa, pos: mirror(pos), angle: -angle }
            }
        });
    }
    PotentialFlow2 { elements: all }
}

/// Added mass of an accelerating cylinder per unit length ρπr².
#[must_use]
pub fn added_mass_cylinder(rho: f64, r: f64) -> f64 {
    rho * PI * r * r
}

/// Added mass of a sphere (2/3)ρπr³.
#[must_use]
pub fn added_mass_sphere(rho: f64, r: f64) -> f64 {
    2.0 / 3.0 * rho * PI * r * r * r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elements_and_cylinder() {
        // Uniform flow closure.
        let flow = uniform_flow(2.0, 0.0);
        let (v, phi, psi) = flow(Vec2::new(1.0, 1.0));
        assert!((v.x - 2.0).abs() < 1e-12 && v.y.abs() < 1e-12);
        assert!((phi - 2.0).abs() < 1e-12);
        assert!((psi - 2.0).abs() < 1e-12);
        // Source radial speed m/(2πr).
        let s = source(1.0, Vec2::ZERO);
        let (vs, _, _) = s(Vec2::new(2.0, 0.0));
        assert!((vs.x - 1.0 / (TWO_PI * 2.0)).abs() < 1e-12);
        // Vortex tangential speed Γ/(2πr), counterclockwise.
        let vx = vortex(1.0, Vec2::ZERO);
        let (vv, _, _) = vx(Vec2::new(2.0, 0.0));
        assert!(vv.x.abs() < 1e-12);
        assert!((vv.y - 1.0 / (TWO_PI * 2.0)).abs() < 1e-12, "{vv:?}");
        // Cylinder: no penetration on the surface, |v| = 2U sinθ.
        let cyl = cylinder_flow(1.0, 0.5, 0.0);
        for k in 1..8 {
            let th = PI * k as f64 / 8.0;
            let p = Vec2::new(0.5 * th.cos(), 0.5 * th.sin());
            let v = cyl.velocity(p);
            let radial = v.dot(&p.normalized());
            assert!(radial.abs() < 1e-9, "penetration {radial} at {th}");
            let speed = v.magnitude();
            assert!((speed - 2.0 * th.sin().abs()).abs() < 1e-9, "surface speed {speed}");
        }
        // Stagnation points at (±r, 0) without circulation.
        let stag = cyl.stagnation_points();
        assert!(stag.iter().any(|p| (p.x - 0.5).abs() < 1e-6 && p.y.abs() < 1e-6));
        assert!(stag.iter().any(|p| (p.x + 0.5).abs() < 1e-6 && p.y.abs() < 1e-6));
        // Kutta-Joukowski lift and exact cp.
        let lifting = cylinder_flow(1.0, 0.5, -2.0);
        assert!((lifting.lift_kutta_joukowski(1.0, 1.2) - 2.4).abs() < 1e-12);
        let cp = lifting.pressure_coefficient(Vec2::new(0.0, 0.5), 1.0);
        let cp_exact = cylinder_cp_exact(PI / 2.0, -2.0, 1.0, 0.5);
        assert!((cp - cp_exact).abs() < 1e-9, "{cp} vs {cp_exact}");
        // Rankine oval: stagnation on the x axis.
        let oval = rankine_oval(1.0, 2.0, 0.5);
        let stag_o = oval.stagnation_points();
        assert!(stag_o.iter().any(|p| p.y.abs() < 1e-6 && p.x < -0.5));
        // Method of images: wall is a streamline.
        let img = method_of_images_wall(
            &[Element::Vortex { gamma: 1.0, pos: Vec2::new(0.0, 1.0) }],
            Plane2 { point: Vec2::ZERO, normal: Vec2::new(0.0, 1.0) },
        );
        let v_wall = img.velocity(Vec2::new(0.7, 0.0));
        assert!(v_wall.y.abs() < 1e-12, "wall penetration {v_wall:?}");
    }

    #[test]
    fn test_sink_and_doublet_singularities() {
        // A sink is a source of the opposite sign: every returned
        // quantity must be the exact negation of the source's.
        let m = 2.3;
        let pos = Vec2::new(0.4, -0.2);
        let src = source(m, pos);
        let snk = sink(m, pos);
        for &p in &[
            Vec2::new(1.0, 0.0),
            Vec2::new(-0.7, 1.3),
            Vec2::new(0.4, 1.1),
            Vec2::new(2.5, -2.0),
        ] {
            let (vs, phis, psis) = src(p);
            let (vk, phik, psik) = snk(p);
            assert!((vs.x + vk.x).abs() < 1e-14 && (vs.y + vk.y).abs() < 1e-14, "{vs:?} {vk:?}");
            assert!((phis + phik).abs() < 1e-14, "phi {phis} {phik}");
            assert!((psis + psik).abs() < 1e-14, "psi {psis} {psik}");
        }
        // The sink draws fluid in: the radial component points inward and
        // the net flux through any enclosing circle is exactly −m
        // (divergence theorem for a point sink of strength m).
        let n = 2000;
        let radius = 0.9;
        let mut flux = 0.0;
        let mut max_radial = f64::NEG_INFINITY;
        for k in 0..n {
            let th = TWO_PI * (k as f64 + 0.5) / n as f64;
            let normal = Vec2::new(th.cos(), th.sin());
            let p = pos + normal * radius;
            let (v, _, _) = snk(p);
            max_radial = max_radial.max(v.dot(&normal));
            flux += v.dot(&normal) * radius * TWO_PI / n as f64;
        }
        assert!(max_radial < 0.0, "sink flow is not inward: {max_radial}");
        assert!((flux + m).abs() < 1e-9, "sink flux {flux} vs {}", -m);
        // Outside the singularity the sink field is irrotational: the
        // circulation around the same circle vanishes.
        let mut circulation = 0.0;
        for k in 0..n {
            let th = TWO_PI * (k as f64 + 0.5) / n as f64;
            let tangent = Vec2::new(-th.sin(), th.cos());
            let p = pos + Vec2::new(th.cos(), th.sin()) * radius;
            let (v, _, _) = snk(p);
            circulation += v.dot(&tangent) * radius * TWO_PI / n as f64;
        }
        assert!(circulation.abs() < 1e-9, "sink circulation {circulation}");

        // The doublet velocity decays exactly as 1/r²: doubling the
        // distance quarters the speed.
        let kappa = 1.7;
        let angle = 0.6;
        let dbl = doublet(kappa, Vec2::ZERO, angle);
        for &th in &[0.0_f64, 0.9, 2.4, -1.3] {
            let dir = Vec2::new(th.cos(), th.sin());
            let (v1, _, _) = dbl(dir * 1.0);
            let (v2, _, _) = dbl(dir * 2.0);
            let (v4, _, _) = dbl(dir * 4.0);
            assert!(
                (v1.magnitude() / v2.magnitude() - 4.0).abs() < 1e-9,
                "doublet decay 1->2: {}",
                v1.magnitude() / v2.magnitude()
            );
            assert!(
                (v2.magnitude() / v4.magnitude() - 4.0).abs() < 1e-9,
                "doublet decay 2->4: {}",
                v2.magnitude() / v4.magnitude()
            );
        }
        // On the doublet axis the potential is κ cos0/(2πr) = κ/(2πr).
        let (_, phi_axis, _) = dbl(Vec2::new(angle.cos(), angle.sin()) * 3.0);
        assert!(
            (phi_axis - kappa / (TWO_PI * 3.0)).abs() < 1e-12,
            "axis potential {phi_axis}"
        );

        // A doublet is the limit of a source/sink pair separated by d
        // along the doublet axis with κ = m·d. Compare against the
        // independent source and sink closures at finite, small d.
        let d = 1e-4;
        let axis = Vec2::new(angle.cos(), angle.sin());
        let pair_src = source(kappa / d, axis * (-0.5 * d));
        let pair_snk = sink(kappa / d, axis * (0.5 * d));
        for &p in &[Vec2::new(1.0, 0.5), Vec2::new(-2.0, 1.0), Vec2::new(0.3, -1.4)] {
            let (vd, phid, psid) = dbl(p);
            let (v1, phi1, psi1) = pair_src(p);
            let (v2, phi2, psi2) = pair_snk(p);
            let v = v1 + v2;
            // The remainder of the expansion is O(d²) relative.
            assert!(
                (v - vd).magnitude() < 1e-6 * vd.magnitude(),
                "doublet vs pair at {p:?}: {v:?} vs {vd:?}"
            );
            assert!((phi1 + phi2 - phid).abs() < 1e-6 * phid.abs().max(1e-3));
            assert!((psi1 + psi2 - psid).abs() < 1e-6 * psid.abs().max(1e-3));
        }
    }

    #[test]
    fn test_potential_stream_function_and_streamlines() {
        // A superposition with all four element kinds: uniform stream,
        // source, vortex and doublet.
        let flow = PotentialFlow2 {
            elements: vec![
                Element::Uniform { u: 1.3, alpha: 0.2 },
                Element::Source { m: 0.7, pos: Vec2::new(-0.6, 0.1) },
                Element::Vortex { gamma: -0.9, pos: Vec2::new(0.3, -0.4) },
                Element::Doublet { kappa: 0.5, pos: Vec2::new(0.1, 0.6), angle: 0.4 },
            ],
        };
        let probes = [
            Vec2::new(2.0, 1.0),
            Vec2::new(-1.5, 2.2),
            Vec2::new(1.1, -1.7),
            Vec2::new(-2.4, -0.9),
        ];
        for &p in &probes {
            // W(z) = φ + iψ by definition.
            let w = flow.complex_potential(Complex::new(p.x, p.y));
            assert!((w.re - flow.potential(p)).abs() < 1e-14, "Re W != phi");
            assert!((w.im - flow.stream_function(p)).abs() < 1e-14, "Im W != psi");

            // φ and ψ are harmonic conjugates: the Cauchy-Riemann
            // relations ∂φ/∂x = ∂ψ/∂y and ∂φ/∂y = −∂ψ/∂x hold, and ∇φ is
            // the velocity.
            let h = 1e-5;
            let dx = |f: &dyn Fn(Vec2) -> f64| {
                (f(Vec2::new(p.x + h, p.y)) - f(Vec2::new(p.x - h, p.y))) / (2.0 * h)
            };
            let dy = |f: &dyn Fn(Vec2) -> f64| {
                (f(Vec2::new(p.x, p.y + h)) - f(Vec2::new(p.x, p.y - h))) / (2.0 * h)
            };
            let phi_x = dx(&|q| flow.potential(q));
            let phi_y = dy(&|q| flow.potential(q));
            let psi_x = dx(&|q| flow.stream_function(q));
            let psi_y = dy(&|q| flow.stream_function(q));
            // Central differences at h = 1e-5 carry O(h²)|f'''| ≈ 1e-9.
            assert!((phi_x - psi_y).abs() < 1e-7, "CR-1 at {p:?}: {phi_x} vs {psi_y}");
            assert!((phi_y + psi_x).abs() < 1e-7, "CR-2 at {p:?}: {phi_y} vs {}", -psi_x);
            let v = flow.velocity(p);
            assert!((phi_x - v.x).abs() < 1e-7, "grad phi != u: {phi_x} vs {}", v.x);
            assert!((phi_y - v.y).abs() < 1e-7, "grad phi != v: {phi_y} vs {}", v.y);
            // ψ is the stream function: ∂ψ/∂y = u, −∂ψ/∂x = v.
            assert!((psi_y - v.x).abs() < 1e-7);
            assert!((psi_x + v.y).abs() < 1e-7);

            // Both are harmonic away from the singularities.
            let lap = |f: &dyn Fn(Vec2) -> f64| {
                let hh = 1e-3;
                (f(Vec2::new(p.x + hh, p.y)) + f(Vec2::new(p.x - hh, p.y))
                    + f(Vec2::new(p.x, p.y + hh))
                    + f(Vec2::new(p.x, p.y - hh))
                    - 4.0 * f(p))
                    / (hh * hh)
            };
            assert!(lap(&|q| flow.potential(q)).abs() < 1e-5, "phi not harmonic at {p:?}");
            assert!(
                lap(&|q| flow.stream_function(q)).abs() < 1e-5,
                "psi not harmonic at {p:?}"
            );
        }

        // The cylinder surface is the streamline ψ = 0 and the stagnation
        // streamline through it.
        let cyl = cylinder_flow(1.0, 0.5, 0.0);
        for k in 0..16 {
            let th = TWO_PI * k as f64 / 16.0;
            let p = Vec2::new(0.5 * th.cos(), 0.5 * th.sin());
            assert!(
                cyl.stream_function(p).abs() < 1e-12,
                "cylinder surface is not psi = 0: {}",
                cyl.stream_function(p)
            );
        }
        // Far upstream the potential reduces to the free stream Ux.
        let far = Vec2::new(-500.0, 3.0);
        assert!((cyl.potential(far) - far.x).abs() < 1e-3, "{}", cyl.potential(far));

        // Streamlines of a uniform stream are exactly straight lines
        // covering |U| dt per step.
        let uni = PotentialFlow2 {
            elements: vec![Element::Uniform { u: 2.0, alpha: 0.3 }],
        };
        let dt = 0.05;
        let lines = uni.streamlines(&[Vec2::new(-1.0, 0.5), Vec2::new(0.0, -1.0)], 25, dt);
        for line in &lines {
            assert_eq!(line.len(), 26);
            let dir = Vec2::new(0.3_f64.cos(), 0.3_f64.sin());
            for (k, p) in line.iter().enumerate() {
                let want = line[0] + dir * (2.0 * dt * k as f64);
                assert!((*p - want).magnitude() < 1e-12, "step {k}: {p:?} vs {want:?}");
            }
        }

        // A streamline of the cylinder flow is a level set of ψ: the
        // stream function is constant along it. RK2 with dt = 0.01 keeps
        // the drift to O(dt³) per step.
        let seed = Vec2::new(-3.0, 0.35);
        let psi0 = cyl.stream_function(seed);
        let path = cyl.streamlines(&[seed], 600, 0.01);
        for p in &path[0] {
            assert!(
                (cyl.stream_function(*p) - psi0).abs() < 1e-4,
                "psi drifted to {} from {psi0} at {p:?}",
                cyl.stream_function(*p)
            );
            // A streamline outside the body never enters it.
            assert!(p.magnitude() > 0.5 - 1e-6, "streamline entered the cylinder at {p:?}");
        }
        // It flows downstream and passes over the cylinder.
        assert!(path[0].last().unwrap().x > 1.0, "streamline did not advance");
    }

    #[test]
    fn test_joukowski() {
        let c = 0.25;
        let center = Complex::new(-0.025, 0.05);
        // Round trip of the transform away from the cut.
        let z = Complex::new(0.4, 0.3);
        let zeta = joukowski_transform(z, c);
        let back = inverse_joukowski(zeta, c);
        assert!((back - z).norm() < 1e-9);
        // Airfoil closes at the trailing edge 2c.
        let foil = joukowski_airfoil(center, c, 100);
        assert!(foil.iter().any(|p| (p.x - 2.0 * c).abs() < 1e-6));
        // Lift curve close to thin-airfoil 2π slope at small alpha.
        let (pts, cps, cl) = joukowski_airfoil_flow(center, c, 0.05, 1.0);
        assert_eq!(pts.len(), cps.len());
        // Cambered Joukowski: cl = 2π sin(α + β) · 4R/chord with
        // sin β = y_c/R; check the sign and the thin-airfoil scale.
        let r = (Complex::new(c, 0.0) - center).norm();
        let beta = (center.im / r).asin();
        let cl_expect = TWO_PI * (0.05 + beta).sin();
        assert!(cl > 0.0, "camber+alpha must lift: {cl}");
        assert!((cl / cl_expect - 1.0).abs() < 0.3, "cl {cl} vs {cl_expect}");
        // Lift grows with incidence.
        let (_, _, cl2) = joukowski_airfoil_flow(center, c, 0.10, 1.0);
        assert!(cl2 > cl);
        // cp bounded by the stagnation value 1.
        assert!(cps.iter().all(|&cp| cp <= 1.0 + 1e-9));
        // Zero-camber symmetric foil at α = 0 has zero lift.
        let (_, _, cl0) = joukowski_airfoil_flow(Complex::new(-0.03, 0.0), c, 0.0, 1.0);
        assert!(cl0.abs() < 1e-9, "symmetric cl {cl0}");
        // Karman-Trefftz reduces to Joukowski at n = 2.
        let kt = karman_trefftz_airfoil(center, c, 2.0, 64);
        let jk = joukowski_airfoil(center, c, 64);
        for (a, b) in kt.iter().zip(&jk) {
            assert!((a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6);
        }
    }

    #[test]
    fn test_naca_sections() {
        let foil = naca4("2412", 80, true);
        assert!(foil.len() > 70);
        // Max thickness ~12% and max camber ~2%.
        let y_max = foil.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let y_min = foil.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        assert!((y_max - y_min - 0.12).abs() < 0.02, "thickness {}", y_max - y_min);
        assert!(y_max > 0.06 && y_min > -0.07, "camber asymmetry");
        // Symmetric 0012: top/bottom mirror.
        let sym = naca4("0012", 80, true);
        let y_max_s = sym.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let y_min_s = sym.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        assert!((y_max_s + y_min_s).abs() < 1e-9);
        // 5-digit: 23012 has thickness 12%.
        let five = naca5("23012", 80);
        let t5 = five.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max)
            - five.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        assert!((t5 - 0.12).abs() < 0.02, "naca5 thickness {t5}");
    }

    #[test]
    fn test_panel_method() {
        let foil = naca4("0012", 120, true);
        let mut pm = PanelMethod::new(&foil);
        pm.alpha = 5.0_f64.to_radians();
        pm.u_inf = 1.0;
        pm.solve();
        let cl = pm.cl();
        let cl_thin = TWO_PI * pm.alpha;
        assert!(
            (cl / cl_thin - 1.0).abs() < 0.25,
            "panel cl {cl} vs thin {cl_thin}"
        );
        // cp: suction peak on the upper surface near the LE; stagnation
        // cp ≈ 1 somewhere.
        let cps = pm.cp_distribution();
        let cp_min = cps.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let cp_max = cps.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
        assert!(cp_min < -1.0, "no suction peak: {cp_min}");
        assert!(cp_max > 0.9 && cp_max <= 1.05, "no stagnation: {cp_max}");
        // α = 0 symmetric: no lift.
        let mut pm0 = PanelMethod::new(&naca4("0012", 120, true));
        pm0.solve();
        assert!(pm0.cl().abs() < 1e-6, "symmetric cl {}", pm0.cl());
        // Velocity field: far upstream is the free stream.
        let v_far = pm.velocity_at(Vec2::new(-30.0, 0.0));
        assert!((v_far.magnitude() - 1.0).abs() < 0.01);
        // Center of pressure near the quarter chord for small alpha.
        let xcp = pm.pressure_center();
        assert!((0.1..0.5).contains(&xcp), "x_cp {xcp}");
        assert!(pm.cm_quarter_chord().abs() < 0.1);
    }

    #[test]
    fn test_panel_method_streamlines() {
        let alpha = 6.0_f64.to_radians();
        let foil = naca4("2412", 140, true);
        let mut pm = PanelMethod::new(&foil);
        pm.alpha = alpha;
        pm.u_inf = 1.0;
        pm.solve();

        let seeds = [
            Vec2::new(-2.0, -0.30),
            Vec2::new(-2.0, -0.10),
            Vec2::new(-2.0, 0.10),
            Vec2::new(-2.0, 0.30),
        ];
        let dt = 0.01;
        let steps = 500;
        let lines = pm.streamlines(&seeds, steps, dt);
        assert_eq!(lines.len(), seeds.len());

        // Each traced point is an explicit Euler step along the velocity
        // field: p_{k+1} − p_k = u(p_k) dt exactly.
        for line in &lines {
            assert_eq!(line.len(), steps + 1);
            for w in line.windows(2) {
                let want = pm.velocity_at(w[0]) * dt;
                let got = w[1] - w[0];
                assert!(
                    (got - want).magnitude() < 1e-12,
                    "streamline step {got:?} is not u dt = {want:?}"
                );
            }
        }

        // Streamlines are material lines of a solid body: none of them may
        // cross into the airfoil. The section is thin, so a point is
        // "inside" if it is within the chord and closer to the camber line
        // than the local half thickness; testing against the polygon
        // directly is cleaner.
        let inside = |p: Vec2| -> bool {
            let mut hits = 0;
            for k in 0..foil.len() {
                let (a, b) = (foil[k], foil[(k + 1) % foil.len()]);
                if (a.y > p.y) != (b.y > p.y) {
                    let x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
                    if x > p.x {
                        hits += 1;
                    }
                }
            }
            hits % 2 == 1
        };
        assert!(inside(Vec2::new(0.4, 0.02)), "the inside test must work");
        assert!(!inside(Vec2::new(0.4, 0.5)));
        for (s, line) in lines.iter().enumerate() {
            for p in line {
                assert!(!inside(*p), "seed {s} entered the airfoil at {p:?}");
            }
        }

        // Upwash ahead, downwash behind: the bound circulation of a
        // lifting section turns the flow up in front of it and down
        // behind it, so the local flow angle straddles the incidence.
        // The speed is still close to the free stream two chords away.
        let free_dir = Vec2::new(alpha.cos(), alpha.sin());
        assert!(pm.cl() > 0.3, "expected a lifting section, cl = {}", pm.cl());
        for (s, line) in lines.iter().enumerate() {
            let head = (line[1] - line[0]) * (1.0 / dt);
            assert!(
                (head.magnitude() / pm.u_inf - 1.0).abs() < 0.05,
                "seed {s} upstream speed {} is not the free stream",
                head.magnitude()
            );
            let angle_in = head.y.atan2(head.x);
            assert!(
                angle_in > alpha,
                "seed {s} upstream angle {angle_in} shows no upwash (alpha {alpha})"
            );

            let tail = *line.last().unwrap();
            assert!(tail.x > 1.5, "seed {s} never reached the wake: {tail:?}");
            let out = (tail - line[line.len() - 2]).normalized();
            let angle_out = out.y.atan2(out.x);
            assert!(
                angle_out < alpha,
                "seed {s} wake angle {angle_out} is not deflected below alpha {alpha}"
            );
            assert!(
                angle_out < angle_in,
                "seed {s} was not turned downwards: {angle_in} -> {angle_out}"
            );
        }
        // The far-field disturbance is small: two chords upstream the flow
        // direction is within a few degrees of the free stream.
        let head0 = (lines[0][1] - lines[0][0]).normalized();
        assert!(
            (head0 - free_dir).magnitude() < 0.05,
            "far upstream too disturbed: {head0:?} vs {free_dir:?}"
        );

        // Streamlines cannot cross: the vertical ordering of the seeds is
        // preserved all the way downstream.
        for k in 0..lines.len() - 1 {
            let (lo, hi) = (lines[k].last().unwrap(), lines[k + 1].last().unwrap());
            assert!(lo.y < hi.y, "streamlines {k} and {} crossed", k + 1);
        }

        // Far from the section the flow is a bound vortex of strength Γ,
        // whose induced velocity falls off as 1/r. A streamline released
        // a distance R below the airfoil therefore bows away from the
        // straight free-stream line by an amount ∝ 1/R: doubling R halves
        // the bow.
        let bow = |depth: f64| -> f64 {
            let seed = Vec2::new(-4.0, -depth);
            let line = &pm.streamlines(&[seed], 400, 0.02)[0];
            line.iter()
                .map(|p| {
                    let rel = *p - seed;
                    (rel.x * free_dir.y - rel.y * free_dir.x).abs()
                })
                .fold(0.0_f64, f64::max)
        };
        let (b8, b16) = (bow(8.0), bow(16.0));
        assert!(b8 > 0.0 && b8 < 0.05, "near-field bow out of range: {b8}");
        assert!(
            (1.5..3.0).contains(&(b8 / b16)),
            "far-field decay is not 1/r: bow {b8} at R=8 vs {b16} at R=16"
        );
    }

    #[test]
    fn test_wing_theory() {
        // Thin airfoil: flat plate 2πα; parabolic camber shifts α_L0.
        assert!((thin_airfoil_cl_flat(0.1) - TWO_PI * 0.1).abs() < 1e-12);
        let flat = thin_airfoil_cl(0.1, &|_| 0.0);
        assert!((flat - TWO_PI * 0.1).abs() < 1e-6);
        // Parabolic camber z = 4 z_max x(1−x): α_L0 = −2 z_max.
        let zmax = 0.02;
        let cambered = thin_airfoil_cl(0.0, &move |x| 4.0 * zmax * (1.0 - 2.0 * x));
        assert!(
            (cambered - TWO_PI * 2.0 * zmax).abs() < 0.01 * TWO_PI * 2.0 * zmax + 1e-9,
            "cambered {cambered} vs {}",
            TWO_PI * 2.0 * zmax
        );
        // Lifting line: elliptic-like rectangular wing approaches the
        // monoplane equation result; induced drag ≈ CL²/(πAR)(1+δ).
        let span = 10.0;
        let chord = 1.0;
        let (cl, cdi, circ) = lifting_line(span, &|_| chord, &|_| 0.1, 20, 1.0);
        let ar = span / chord;
        let cl_expect = TWO_PI * 0.1 / (1.0 + 2.0 / ar);
        assert!((cl / cl_expect - 1.0).abs() < 0.1, "LL CL {cl} vs {cl_expect}");
        assert!(cdi > cl * cl / (PI * ar) * 0.95, "CDi {cdi} below elliptic bound");
        assert!(cdi < cl * cl / (PI * ar) * 1.25, "CDi {cdi} too large");
        // Circulation peaks at the root, vanishes toward the tips.
        assert!(circ[10] > circ[0].max(circ[19]));
        assert!((elliptic_wing_cl(8.0, 0.1) - TWO_PI * 0.1 / 1.25).abs() < 1e-12);
        assert!((induced_drag(0.5, 8.0, 0.9) - 0.25 / (PI * 0.9 * 8.0)).abs() < 1e-12);
        let e = oswald_efficiency_estimate(8.0, 0.0);
        assert!((0.6..1.0).contains(&e), "oswald {e}");
        // VLM: rectangular AR 10 wing lift slope below 2π, near lifting
        // line.
        let wing = WingGeometry { span: 10.0, root_chord: 1.0, tip_chord: 1.0, sweep: 0.0 };
        let (cl_vlm, cdi_vlm, gam) = vortex_lattice(&wing, 0.1, 24, 1, 1.0);
        assert!(cl_vlm > 0.3 && cl_vlm < TWO_PI * 0.1, "VLM CL {cl_vlm}");
        assert!(cdi_vlm > 0.0 && cdi_vlm < 0.1, "VLM CDi {cdi_vlm}");
        assert!(gam[12] > gam[0], "VLM circulation shape");
        assert!(ground_effect_factor(10.0) > 0.99);
        assert!(ground_effect_factor(0.1) < 0.8);
        assert!((added_mass_cylinder(1000.0, 0.5) - 1000.0 * PI * 0.25).abs() < 1e-9);
        assert!((added_mass_sphere(1000.0, 0.5) - 2.0 / 3.0 * 1000.0 * PI * 0.125).abs() < 1e-9);
        // Conformal map: identity map reproduces the base flow.
        let base = cylinder_flow(1.0, 0.5, 0.0);
        let v = conformal_map_flow(&|z| z, &base, Complex::new(2.0, 1.0));
        let direct = base.velocity(Vec2::new(2.0, 1.0));
        assert!((v - direct).magnitude() < 1e-6);
    }
}
