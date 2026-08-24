//! Hyperbolic geometry across the standard models: Poincare disk/ball,
//! upper half-plane/space, Klein disk, and the hyperboloid, with
//! isometries, trigonometry, tilings, and low-distortion embeddings.

use crate::fractals::Complex;
use crate::linalg::Matrix;
use crate::manifold::lie::{Sl2C, Sl2R};
use crate::manifold::vecn::VecN;
use crate::math::Vec3;
use crate::monte_carlo::Rng;

const PI: f64 = std::f64::consts::PI;

fn c(re: f64, im: f64) -> Complex {
    Complex::new(re, im)
}

// ---------------------------------------------------------------------------
// Models and points
// ---------------------------------------------------------------------------

/// The classical models of hyperbolic space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypModel {
    PoincareDisk,
    PoincareBall,
    UpperHalfPlane,
    UpperHalfSpace,
    Klein,
    Hyperboloid,
}

/// A point of hyperbolic space tagged with the model its coordinates use.
#[derive(Debug, Clone, PartialEq)]
pub struct HypPoint {
    pub coords: VecN,
    pub model: HypModel,
}

impl HypPoint {
    /// The origin of hyperbolic space in the given model and dimension.
    #[must_use]
    pub fn origin(model: HypModel, dim: usize) -> Self {
        let coords = match model {
            HypModel::PoincareDisk | HypModel::PoincareBall | HypModel::Klein => VecN::zeros(dim),
            HypModel::UpperHalfPlane | HypModel::UpperHalfSpace => {
                let mut v = VecN::zeros(dim);
                v.data[dim - 1] = 1.0;
                v
            }
            HypModel::Hyperboloid => {
                let mut v = VecN::zeros(dim + 1);
                v.data[0] = 1.0;
                v
            }
        };
        HypPoint { coords, model }
    }

    /// A 2D point at hyperbolic polar coordinates (r, theta) from the disk
    /// origin.
    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let e = (0.5 * r).tanh();
        HypPoint {
            coords: VecN::from(&[e * theta.cos(), e * theta.sin()]),
            model: HypModel::PoincareDisk,
        }
    }

    fn intrinsic_dim(&self) -> usize {
        match self.model {
            HypModel::Hyperboloid => self.coords.dim() - 1,
            _ => self.coords.dim(),
        }
    }

    /// Convert to the hyperboloid model (the hub for all conversions).
    fn to_hyperboloid(&self) -> VecN {
        let n = self.intrinsic_dim();
        match self.model {
            HypModel::Hyperboloid => self.coords.clone(),
            HypModel::PoincareDisk | HypModel::PoincareBall => {
                let r2 = self.coords.dot(&self.coords);
                let d = (1.0 - r2).max(1e-300);
                let mut out = VecN::zeros(n + 1);
                out.data[0] = (1.0 + r2) / d;
                for i in 0..n {
                    out.data[i + 1] = 2.0 * self.coords[i] / d;
                }
                out
            }
            HypModel::Klein => {
                let r2 = self.coords.dot(&self.coords);
                let g = 1.0 / (1.0 - r2).max(1e-300).sqrt();
                let mut out = VecN::zeros(n + 1);
                out.data[0] = g;
                for i in 0..n {
                    out.data[i + 1] = g * self.coords[i];
                }
                out
            }
            HypModel::UpperHalfPlane | HypModel::UpperHalfSpace => {
                // go through the ball first
                let ball = half_space_to_ball_vec(&self.coords);
                HypPoint {
                    coords: ball,
                    model: HypModel::PoincareBall,
                }
                .to_hyperboloid()
            }
        }
    }

    fn from_hyperboloid(h: &VecN, model: HypModel) -> HypPoint {
        let n = h.dim() - 1;
        match model {
            HypModel::Hyperboloid => HypPoint {
                coords: h.clone(),
                model,
            },
            HypModel::PoincareDisk | HypModel::PoincareBall => {
                let mut v = VecN::zeros(n);
                for i in 0..n {
                    v.data[i] = h[i + 1] / (1.0 + h[0]);
                }
                HypPoint { coords: v, model }
            }
            HypModel::Klein => {
                let mut v = VecN::zeros(n);
                for i in 0..n {
                    v.data[i] = h[i + 1] / h[0];
                }
                HypPoint { coords: v, model }
            }
            HypModel::UpperHalfPlane | HypModel::UpperHalfSpace => {
                let ball = HypPoint::from_hyperboloid(h, HypModel::PoincareBall);
                HypPoint {
                    coords: ball_to_half_space_vec(&ball.coords),
                    model,
                }
            }
        }
    }

    /// Convert to another model.
    #[must_use]
    pub fn to(&self, model: HypModel) -> HypPoint {
        HypPoint::from_hyperboloid(&self.to_hyperboloid(), model)
    }

    /// Hyperbolic distance to another point (models may differ).
    #[must_use]
    pub fn distance(&self, other: &HypPoint) -> f64 {
        let a = self.to_hyperboloid();
        let b = other.to_hyperboloid();
        hyp_distance_hyperboloid(&a, &b)
    }

    /// Sample the geodesic to another point at n+1 evenly spaced hyperbolic
    /// parameters.
    #[must_use]
    pub fn geodesic_to(&self, other: &HypPoint, n: usize) -> Vec<HypPoint> {
        let a = self.to_hyperboloid();
        let b = other.to_hyperboloid();
        let d = hyp_distance_hyperboloid(&a, &b);
        (0..=n)
            .map(|k| {
                let t = k as f64 / n as f64;
                let h = if d < 1e-12 {
                    a.clone()
                } else {
                    // geodesic on the hyperboloid: cosh/sinh interpolation
                    let coef_a = ((1.0 - t) * d).sinh() / d.sinh();
                    let coef_b = (t * d).sinh() / d.sinh();
                    a.scale(coef_a).add(&b.scale(coef_b))
                };
                HypPoint::from_hyperboloid(&h, self.model)
            })
            .collect()
    }

    /// Hyperbolic midpoint.
    #[must_use]
    pub fn midpoint(&self, other: &HypPoint) -> HypPoint {
        self.geodesic_to(other, 2).swap_remove(1)
    }

    /// Reflect across the geodesic through two points (2D disk model).
    #[must_use]
    pub fn reflect_across(&self, geodesic: (&HypPoint, &HypPoint)) -> HypPoint {
        let p = self.to(HypModel::PoincareDisk);
        let z = c(p.coords[0], p.coords[1]);
        let a = geodesic.0.to(HypModel::PoincareDisk);
        let b = geodesic.1.to(HypModel::PoincareDisk);
        let za = c(a.coords[0], a.coords[1]);
        let zb = c(b.coords[0], b.coords[1]);
        let w = match hyp_geodesic_circle_disk(za, zb) {
            Some((center, radius)) => {
                // inversion in the circle
                let d = z - center;
                center + d * c(radius * radius / d.norm_sq(), 0.0)
            }
            None => {
                // diameter: Euclidean reflection across the line through 0
                let dir = (zb - za) / c((zb - za).norm(), 0.0);
                dir * dir * z.conjugate()
            }
        };
        HypPoint {
            coords: VecN::from(&[w.re, w.im]),
            model: HypModel::PoincareDisk,
        }
        .to(self.model)
    }

    /// Interior angle at this vertex formed by geodesics to a and b (2D).
    #[must_use]
    pub fn angle_at(&self, a: &HypPoint, b: &HypPoint) -> f64 {
        // move self to the origin by a Mobius isometry; geodesics through
        // the origin are straight, so the angle is Euclidean
        let p = self.to(HypModel::PoincareDisk);
        let z0 = c(p.coords[0], p.coords[1]);
        let map = |q: &HypPoint| {
            let qq = q.to(HypModel::PoincareDisk);
            let z = c(qq.coords[0], qq.coords[1]);
            (z - z0) / (c(1.0, 0.0) - z0.conjugate() * z)
        };
        let va = map(a);
        let vb = map(b);
        let dot = va.re * vb.re + va.im * vb.im;
        (dot / (va.norm() * vb.norm())).clamp(-1.0, 1.0).acos()
    }

    /// Euclidean display coordinates (disk/ball models pass through; the
    /// hyperboloid projects to the disk).
    #[must_use]
    pub fn to_euclidean_display(&self) -> VecN {
        match self.model {
            HypModel::Hyperboloid => self.to(HypModel::PoincareBall).coords,
            _ => self.coords.clone(),
        }
    }
}

fn half_space_to_ball_vec(p: &VecN) -> VecN {
    // inversion sending the half-space x_n > 0 to the unit ball:
    // reflect through the sphere of radius sqrt(2) centered at -e_n
    let n = p.dim();
    let mut q = p.clone();
    q.data[n - 1] += 1.0; // shift: center at -e_n means p + e_n
    let r2 = q.dot(&q);
    let mut out = q.scale(2.0 / r2);
    out.data[n - 1] -= 1.0;
    out
}

fn ball_to_half_space_vec(p: &VecN) -> VecN {
    // the same inversion is an involution: it swaps ball and half-space
    half_space_to_ball_vec(p)
}

// ---------------------------------------------------------------------------
// Distances and geodesics in specific models
// ---------------------------------------------------------------------------

/// Poincare disk distance.
#[must_use]
pub fn hyp_distance_disk(z: Complex, w: Complex) -> f64 {
    let num = (z - w).norm_sq();
    let den = (1.0 - z.norm_sq()) * (1.0 - w.norm_sq());
    (1.0 + 2.0 * num / den).acosh()
}

/// Upper half-plane distance.
#[must_use]
pub fn hyp_distance_uhp(z: Complex, w: Complex) -> f64 {
    let num = (z - w).norm_sq();
    (1.0 + num / (2.0 * z.im * w.im)).acosh()
}

/// Hyperboloid-model distance acosh(-<x, y>) with the Minkowski form
/// <x, y> = -x0 y0 + sum xi yi.
#[must_use]
pub fn hyp_distance_hyperboloid(x: &VecN, y: &VecN) -> f64 {
    let mut inner = -x[0] * y[0];
    for i in 1..x.dim() {
        inner += x[i] * y[i];
    }
    (-inner).max(1.0).acosh()
}

/// Center and radius of the circular arc through z and w orthogonal to the
/// unit circle; None when the geodesic is a diameter.
#[must_use]
pub fn hyp_geodesic_circle_disk(z: Complex, w: Complex) -> Option<(Complex, f64)> {
    // through-origin (or collinear with origin) case: diameter
    let cross = z.re * w.im - z.im * w.re;
    if cross.abs() < 1e-12 * (z.norm() * w.norm()).max(1e-30) || z.norm() < 1e-12 || w.norm() < 1e-12 {
        return None;
    }
    // orthogonal circle: |c|^2 = r^2 + 1; solve linear system from
    // |z - c|^2 = r^2 and |w - c|^2 = r^2
    // => 2 c . z = |z|^2 + 1, 2 c . w = |w|^2 + 1
    let a = [[2.0 * z.re, 2.0 * z.im], [2.0 * w.re, 2.0 * w.im]];
    let rhs = [z.norm_sq() + 1.0, w.norm_sq() + 1.0];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-14 {
        return None;
    }
    let cx = (rhs[0] * a[1][1] - rhs[1] * a[0][1]) / det;
    let cy = (a[0][0] * rhs[1] - a[1][0] * rhs[0]) / det;
    let center = c(cx, cy);
    let r = (center.norm_sq() - 1.0).max(0.0).sqrt();
    Some((center, r))
}

/// Sample the disk geodesic between z and w at n+1 points.
#[must_use]
pub fn hyp_geodesic_disk(z: Complex, w: Complex, n: usize) -> Vec<Complex> {
    let a = HypPoint {
        coords: VecN::from(&[z.re, z.im]),
        model: HypModel::PoincareDisk,
    };
    let b = HypPoint {
        coords: VecN::from(&[w.re, w.im]),
        model: HypModel::PoincareDisk,
    };
    a.geodesic_to(&b, n)
        .into_iter()
        .map(|p| c(p.coords[0], p.coords[1]))
        .collect()
}

/// Hyperbolic circle in the disk: a Euclidean circle with offset center.
/// Returns n boundary samples.
#[must_use]
pub fn hyp_circle_disk(center: Complex, radius: f64, n: usize) -> Vec<Complex> {
    // Euclidean parameters of the hyperbolic circle around `center`
    let d = center.norm();
    let rho = (0.5 * radius).tanh();
    // for hyperbolic center at Euclidean distance d from origin:
    // the circle is Euclidean with center on the same ray
    let dh = 2.0 * d.atanh();
    let e1 = (0.5 * (dh + radius)).tanh();
    let e2 = (0.5 * (dh - radius)).tanh();
    let ec = 0.5 * (e1 + e2);
    let er = 0.5 * (e1 - e2).abs();
    let dir = if d > 1e-12 {
        center * c(1.0 / d, 0.0)
    } else {
        c(1.0, 0.0)
    };
    let ecenter = dir * c(ec, 0.0);
    let _ = rho;
    (0..n)
        .map(|k| {
            let th = 2.0 * PI * k as f64 / n as f64;
            ecenter + c(er * th.cos(), er * th.sin())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Trigonometry
// ---------------------------------------------------------------------------

/// Area of a hyperbolic triangle from its angles: pi - (alpha + beta + gamma).
#[must_use]
pub fn hyp_area_triangle(alpha: f64, beta: f64, gamma: f64) -> f64 {
    PI - alpha - beta - gamma
}

/// A triangle with prescribed angles (alpha at the origin, beta and gamma at
/// the other vertices), realized in the Poincare disk.
#[must_use]
pub fn hyp_triangle_from_angles(alpha: f64, beta: f64, gamma: f64) -> [Complex; 3] {
    assert!(alpha + beta + gamma < PI, "angles must sum below pi");
    // hyperbolic law of cosines for angles: cosh c = (cos a cos b + cos g)
    // / (sin a sin b) gives the side lengths
    let side = |a: f64, b: f64, g: f64| ((a.cos() * b.cos() + g.cos()) / (a.sin() * b.sin())).acosh();
    let c_ab = side(alpha, beta, gamma); // side between vertices A and B
    let b_ac = side(alpha, gamma, beta); // side between A and C
    // place A at origin, B along the positive real axis, C at angle alpha
    let eb = (0.5 * c_ab).tanh();
    let ec = (0.5 * b_ac).tanh();
    [
        c(0.0, 0.0),
        c(eb, 0.0),
        c(ec * alpha.cos(), ec * alpha.sin()),
    ]
}

/// Hyperbolic law of cosines: cosh c = cosh a cosh b - sinh a sinh b cos gamma.
#[must_use]
pub fn hyp_law_of_cosines(a: f64, b: f64, gamma: f64) -> f64 {
    (a.cosh() * b.cosh() - a.sinh() * b.sinh() * gamma.cos()).acosh()
}

/// Hyperbolic law of sines: returns sin(alpha) for side a opposite alpha,
/// given (a, b, beta) via sin(alpha)/sinh(a) = sin(beta)/sinh(b).
#[must_use]
pub fn hyp_law_of_sines(a: f64, b: f64, beta: f64) -> f64 {
    (beta.sin() * a.sinh() / b.sinh()).clamp(-1.0, 1.0).asin()
}

/// Angle of parallelism Pi(d) = 2 atan(e^{-d}).
#[must_use]
pub fn hyp_angle_of_parallelism(d: f64) -> f64 {
    2.0 * (-d).exp().atan()
}

/// Circumference of a hyperbolic circle: 2 pi sinh r.
#[must_use]
pub fn hyp_circumference(r: f64) -> f64 {
    2.0 * PI * r.sinh()
}

/// Area of a hyperbolic disk: 2 pi (cosh r - 1).
#[must_use]
pub fn hyp_area_circle(r: f64) -> f64 {
    2.0 * PI * (r.cosh() - 1.0)
}

/// Volume of a hyperbolic ball in `dim` dimensions:
/// vol(S^{n-1}) * integral of sinh^{n-1}.
#[must_use]
pub fn hyp_volume_ball(r: f64, dim: usize) -> f64 {
    let n = dim as f64;
    // surface of the Euclidean unit (n-1)-sphere
    let surf = n * PI.powf(n / 2.0) / crate::special::gamma(n / 2.0 + 1.0);
    // numeric integral of sinh^{n-1} t dt
    let steps = 2000;
    let dt = r / steps as f64;
    let mut integral = 0.0;
    for i in 0..steps {
        let t = (i as f64 + 0.5) * dt;
        integral += t.sinh().powf(n - 1.0) * dt;
    }
    surf * integral
}

// ---------------------------------------------------------------------------
// Isometries
// ---------------------------------------------------------------------------

/// Mobius isometry of the disk: z -> e^{i theta} (z - a)/(1 - conj(a) z).
#[must_use]
pub fn mobius_disk(z: Complex, a: Complex, theta: f64) -> Complex {
    let rot = c(theta.cos(), theta.sin());
    rot * (z - a) / (c(1.0, 0.0) - a.conjugate() * z)
}

/// Mobius action of an SL(2, R) element on the upper half-plane.
#[must_use]
pub fn mobius_uhp(z: Complex, m: &Sl2R) -> Complex {
    m.act_on_upper_half_plane(z)
}

/// The disk isometry (a, theta) sending z1 -> w1 and z2 -> w2 when the
/// distances agree (None otherwise).
#[must_use]
pub fn isometry_disk_from_two_points(
    z1: Complex,
    z2: Complex,
    w1: Complex,
    w2: Complex,
) -> Option<(Complex, f64)> {
    if (hyp_distance_disk(z1, z2) - hyp_distance_disk(w1, w2)).abs() > 1e-9 {
        return None;
    }
    // send z1 to 0, then 0 to w1; align the images of z2
    let g1 = |z: Complex| (z - z1) / (c(1.0, 0.0) - z1.conjugate() * z);
    let z2p = g1(z2);
    let w2p = (w2 - w1) / (c(1.0, 0.0) - w1.conjugate() * w2);
    let theta = w2p.arg() - z2p.arg();
    // full map: z -> mobius(0->w1) ( e^{i theta} g1(z) )
    // expressed in the (a, theta) canonical form by composing
    let rot = c(theta.cos(), theta.sin());
    // f(z) = (rot g1(z) + w1) / (1 + conj(w1) rot g1(z)); find a with
    // f(a) = 0: rot g1(a) = -w1
    let target = (c(0.0, 0.0) - w1) / rot;
    // g1(a) = target => a = (target + z1)/(1 + conj(z1) target)
    let a = (target + z1) / (c(1.0, 0.0) + z1.conjugate() * target);
    // rotation angle: for f(z) = e^{i theta}(z - a)/(1 - conj(a) z) the
    // derivative at a is e^{i theta}/(1 - |a|^2), so theta = arg f'(a);
    // compute f'(a) analytically through the composition
    let g1p = |z: Complex| {
        let den = c(1.0, 0.0) - z1.conjugate() * z;
        c(1.0 - z1.norm_sq(), 0.0) / (den * den)
    };
    let u = rot * g1(a);
    let hp = {
        let den = c(1.0, 0.0) + w1.conjugate() * u;
        c(1.0 - w1.norm_sq(), 0.0) / (den * den)
    };
    let fpa = hp * rot * g1p(a);
    Some((a, fpa.arg()))
}

/// SL(2, R) hyperbolic translation by `dist` along the geodesic in the
/// direction `direction` (an angle in the UHP tangent at i).
#[must_use]
pub fn hyperbolic_translation(dist: f64, direction: f64) -> Sl2R {
    // translation along the imaginary axis, conjugated by a rotation
    let t = Sl2R {
        m: [
            [(0.5 * dist).exp(), 0.0],
            [0.0, (-0.5 * dist).exp()],
        ],
    };
    let r = hyperbolic_rotation(direction);
    r.compose(&t).compose(&r.inverse())
}

/// Elliptic rotation about i in the upper half-plane by angle theta.
#[must_use]
pub fn hyperbolic_rotation(theta: f64) -> Sl2R {
    let (s, co) = (0.5 * theta).sin_cos();
    Sl2R {
        m: [[co, s], [-s, co]],
    }
}

/// Parabolic translation z -> z + t.
#[must_use]
pub fn parabolic(t: f64) -> Sl2R {
    Sl2R {
        m: [[1.0, t], [0.0, 1.0]],
    }
}

// ---------------------------------------------------------------------------
// Model conversions (2D complex forms)
// ---------------------------------------------------------------------------

/// Cayley transform disk -> upper half-plane.
#[must_use]
pub fn disk_to_uhp(z: Complex) -> Complex {
    // w = i (1 + z)/(1 - z)
    c(0.0, 1.0) * (c(1.0, 0.0) + z) / (c(1.0, 0.0) - z)
}

/// Inverse Cayley transform.
#[must_use]
pub fn uhp_to_disk(w: Complex) -> Complex {
    (w - c(0.0, 1.0)) / (w + c(0.0, 1.0))
}

/// Poincare disk -> Klein disk.
#[must_use]
pub fn disk_to_klein(z: Complex) -> Complex {
    z * c(2.0 / (1.0 + z.norm_sq()), 0.0)
}

/// Klein disk -> Poincare disk.
#[must_use]
pub fn klein_to_disk(k: Complex) -> Complex {
    k * c(1.0 / (1.0 + (1.0 - k.norm_sq()).max(0.0).sqrt()), 0.0)
}

/// Poincare disk -> hyperboloid (x0, x1, x2).
#[must_use]
pub fn disk_to_hyperboloid(z: Complex) -> VecN {
    let d = (1.0 - z.norm_sq()).max(1e-300);
    VecN::from(&[
        (1.0 + z.norm_sq()) / d,
        2.0 * z.re / d,
        2.0 * z.im / d,
    ])
}

/// Hyperboloid -> Poincare disk.
#[must_use]
pub fn hyperboloid_to_disk(x: &VecN) -> Complex {
    c(x[1] / (1.0 + x[0]), x[2] / (1.0 + x[0]))
}

/// Poincare ball -> upper half-space (3D).
#[must_use]
pub fn ball_to_half_space(p: Vec3) -> Vec3 {
    let v = ball_to_half_space_vec(&VecN::from(&[p.x, p.y, p.z]));
    Vec3::new(v[0], v[1], v[2])
}

/// Lorentz boost that carries the hyperboloid basepoint (1, 0, ..) to the
/// given hyperboloid point (an isometry of the model).
#[must_use]
pub fn lorentz_boost_hyperboloid(v: &VecN) -> Matrix {
    let n = v.dim();
    // v = (cosh d, sinh d * u) for unit u
    let sp: Vec<f64> = v.data[1..].to_vec();
    let sn: f64 = sp.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut m = Matrix::identity(n);
    if sn < 1e-15 {
        return m;
    }
    let u: Vec<f64> = sp.iter().map(|x| x / sn).collect();
    let ch = v[0];
    let sh = sn;
    m.set(0, 0, ch);
    for i in 0..n - 1 {
        m.set(0, i + 1, sh * u[i]);
        m.set(i + 1, 0, sh * u[i]);
        for j in 0..n - 1 {
            m.set(
                i + 1,
                j + 1,
                if i == j { 1.0 } else { 0.0 } + (ch - 1.0) * u[i] * u[j],
            );
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Tilings
// ---------------------------------------------------------------------------

/// True when a regular {p, q} tiling is hyperbolic: 1/p + 1/q < 1/2.
#[must_use]
pub fn hyp_tiling_exists(p: u32, q: u32) -> bool {
    (1.0 / p as f64 + 1.0 / q as f64) < 0.5
}

/// Regular {p, q} tiling of the disk generated by reflections, to the given
/// recursion depth. Returns the polygons as vertex lists.
#[must_use]
pub fn hyp_tiling(p: u32, q: u32, depth: usize, model: HypModel) -> Vec<Vec<Complex>> {
    assert!(hyp_tiling_exists(p, q), "{{{p},{q}}} is not hyperbolic");
    // circumradius of the fundamental polygon from the (pi/p, pi/q, pi/2)
    // triangle: cosh r = cot(pi/p) cot(pi/q)
    let r = (1.0 / (PI / p as f64).tan() / (PI / q as f64).tan()).acosh();
    let er = (0.5 * r).tanh();
    let base: Vec<Complex> = (0..p)
        .map(|k| {
            let th = 2.0 * PI * k as f64 / p as f64;
            c(er * th.cos(), er * th.sin())
        })
        .collect();
    let mut polys = vec![base.clone()];
    let mut frontier = vec![base];
    let mut centers = vec![c(0.0, 0.0)];
    for _ in 0..depth {
        let mut next = Vec::new();
        for poly in &frontier {
            for e in 0..poly.len() {
                let (a, b) = (poly[e], poly[(e + 1) % poly.len()]);
                // reflect the polygon across edge (a, b)
                let reflected: Vec<Complex> = poly
                    .iter()
                    .map(|&z| reflect_across_geodesic(z, a, b))
                    .collect();
                let center = reflected
                    .iter()
                    .fold(c(0.0, 0.0), |acc, &z| acc + z)
                    * c(1.0 / reflected.len() as f64, 0.0);
                if centers
                    .iter()
                    .all(|&cc| (cc - center).norm() > 1e-6)
                {
                    centers.push(center);
                    polys.push(reflected.clone());
                    next.push(reflected);
                }
            }
        }
        frontier = next;
    }
    let _ = model;
    polys
}

fn reflect_across_geodesic(z: Complex, a: Complex, b: Complex) -> Complex {
    match hyp_geodesic_circle_disk(a, b) {
        Some((center, radius)) => {
            let d = z - center;
            center + d * c(radius * radius / d.norm_sq(), 0.0)
        }
        None => {
            let dir = (b - a) / c((b - a).norm().max(1e-30), 0.0);
            dir * dir * z.conjugate()
        }
    }
}

/// Vertices of the regular 4g-gon fundamental polygon for a genus-g surface
/// (all angles sum to 2 pi).
#[must_use]
pub fn fundamental_polygon_genus(g: usize) -> Vec<Complex> {
    let n = 4 * g;
    // regular n-gon with interior angle 2 pi / n: circumradius from the
    // right triangle, cosh r = cot(alpha) cot(beta) with alpha = beta = pi/n
    let alpha = PI / n as f64;
    let beta = PI / n as f64;
    let r = (1.0 / alpha.tan() / beta.tan()).acosh();
    let er = (0.5 * r).tanh();
    (0..n)
        .map(|k| {
            let th = 2.0 * PI * k as f64 / n as f64;
            c(er * th.cos(), er * th.sin())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Computational geometry in the disk
// ---------------------------------------------------------------------------

/// Approximate hyperbolic Voronoi cells: sample directions around each site
/// and march to the bisector. Returns one polygon per site.
#[must_use]
pub fn hyp_voronoi_disk(sites: &[Complex], n_res: usize) -> Vec<Vec<Complex>> {
    sites
        .iter()
        .map(|&s| {
            (0..n_res)
                .map(|k| {
                    let th = 2.0 * PI * k as f64 / n_res as f64;
                    // march outward until another site is closer (or near
                    // the boundary)
                    let mut lo = 0.0_f64;
                    let mut hi = 6.0_f64;
                    for _ in 0..40 {
                        let mid = 0.5 * (lo + hi);
                        let e = (0.5 * mid).tanh();
                        let z = mobius_disk(c(e * th.cos(), e * th.sin()), c(0.0, 0.0) - s, 0.0);
                        let dz = hyp_distance_disk(z, s);
                        let closer = sites
                            .iter()
                            .any(|&o| (o - s).norm() > 1e-12 && hyp_distance_disk(z, o) < dz);
                        if closer {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    let e = (0.5 * lo).tanh();
                    mobius_disk(c(e * th.cos(), e * th.sin()), c(0.0, 0.0) - s, 0.0)
                })
                .collect()
        })
        .collect()
}

/// Hyperbolic Delaunay triangulation by the empty-circumdisk test in the
/// Klein model (hyperbolic Delaunay = Euclidean Delaunay of Klein points).
#[must_use]
pub fn hyp_delaunay_disk(sites: &[Complex]) -> Vec<[usize; 3]> {
    let k: Vec<Complex> = sites.iter().map(|&z| disk_to_klein(z)).collect();
    let n = k.len();
    let mut tris = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for l in (j + 1)..n {
                // circumcircle of (i, j, l) in the Klein model
                let (a, b, cc) = (k[i], k[j], k[l]);
                let d = 2.0 * (a.re * (b.im - cc.im) + b.re * (cc.im - a.im) + cc.re * (a.im - b.im));
                if d.abs() < 1e-14 {
                    continue;
                }
                let ux = ((a.norm_sq()) * (b.im - cc.im)
                    + (b.norm_sq()) * (cc.im - a.im)
                    + (cc.norm_sq()) * (a.im - b.im))
                    / d;
                let uy = ((a.norm_sq()) * (cc.re - b.re)
                    + (b.norm_sq()) * (a.re - cc.re)
                    + (cc.norm_sq()) * (b.re - a.re))
                    / d;
                let center = c(ux, uy);
                let r2 = (a - center).norm_sq();
                let empty = (0..n).all(|m| {
                    m == i || m == j || m == l || (k[m] - center).norm_sq() > r2 - 1e-12
                });
                if empty {
                    tris.push([i, j, l]);
                }
            }
        }
    }
    tris
}

/// Hyperbolic convex hull via the Klein model (geodesics are straight
/// there). Returns hull vertices in order.
#[must_use]
pub fn hyp_convex_hull_disk(points: &[Complex]) -> Vec<Complex> {
    let k: Vec<Complex> = points.iter().map(|&z| disk_to_klein(z)).collect();
    // gift wrapping
    let n = k.len();
    if n < 3 {
        return points.to_vec();
    }
    let start = (0..n)
        .min_by(|&a, &b| k[a].re.partial_cmp(&k[b].re).unwrap())
        .unwrap();
    let mut hull = vec![start];
    let mut current = start;
    loop {
        let mut next = (current + 1) % n;
        for cand in 0..n {
            if cand == current {
                continue;
            }
            let cross = (k[next].re - k[current].re) * (k[cand].im - k[current].im)
                - (k[next].im - k[current].im) * (k[cand].re - k[current].re);
            if cross < -1e-14 {
                next = cand;
            }
        }
        if next == start {
            break;
        }
        hull.push(next);
        current = next;
        if hull.len() > n {
            break;
        }
    }
    hull.into_iter().map(|i| klein_to_disk(k[i])).collect()
}

/// Hyperbolic centroid (Karcher mean) of disk points.
#[must_use]
pub fn hyp_centroid_disk(points: &[Complex], iters: usize) -> Complex {
    let mut m = points[0];
    for _ in 0..iters {
        // log map at m: translate m to origin, take scaled direction
        let mut avg = c(0.0, 0.0);
        for &p in points {
            let z = mobius_disk(p, m, 0.0);
            let r = z.norm();
            if r > 1e-15 {
                let d = 2.0 * r.atanh();
                avg = avg + z * c(d / r / points.len() as f64, 0.0);
            }
        }
        // exp map back
        let d = avg.norm();
        if d < 1e-14 {
            break;
        }
        let e = (0.5 * d).tanh();
        let step = avg * c(e / d, 0.0);
        m = mobius_disk(step, c(0.0, 0.0) - m, 0.0);
    }
    m
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// Sarkar's low-distortion embedding of a tree into the Poincare disk.
#[must_use]
pub fn hyp_embed_tree(adjacency: &[Vec<usize>], root: usize) -> Vec<Complex> {
    let n = adjacency.len();
    let mut pos = vec![c(0.0, 0.0); n];
    let mut placed = vec![false; n];
    // scale: long edges keep children well separated
    let edge_len = 3.0 * (n as f64).ln().max(2.0);
    let e = (0.5 * edge_len).tanh();
    placed[root] = true;
    // BFS placement: children fan out in the half-space away from parent
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((root, None::<usize>));
    while let Some((v, parent)) = queue.pop_front() {
        let kids: Vec<usize> = adjacency[v]
            .iter()
            .copied()
            .filter(|&u| Some(u) != parent && !placed[u])
            .collect();
        let deg = kids.len();
        for (idx, &u) in kids.iter().enumerate() {
            // direction in the tangent space at v (translated to origin)
            let base_angle = match parent {
                Some(p) => {
                    let zp = mobius_disk(pos[p], pos[v], 0.0);
                    zp.arg() + PI
                }
                None => 0.0,
            };
            let spread = if parent.is_some() { PI } else { 2.0 * PI };
            let th = base_angle + spread * ((idx as f64 + 1.0) / (deg as f64 + 1.0) - 0.5);
            let step = c(e * th.cos(), e * th.sin());
            pos[u] = mobius_disk(step, c(0.0, 0.0) - pos[v], 0.0);
            placed[u] = true;
            queue.push_back((u, Some(v)));
        }
    }
    pos
}

/// Stress-majorization MDS into hyperbolic space (Poincare ball of the
/// given dimension) matching the target distance matrix.
#[must_use]
pub fn hyp_embed_graph_mds(dist: &Matrix, dim: usize, iters: usize) -> Vec<VecN> {
    let n = dist.rows;
    let mut rng = Rng::new(42);
    let mut pts: Vec<VecN> = (0..n)
        .map(|_| VecN::random_gaussian(dim, &mut rng).scale(0.1))
        .collect();
    let lr = 0.05;
    for _ in 0..iters {
        for i in 0..n {
            let mut grad = VecN::zeros(dim);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pi = HypPoint {
                    coords: pts[i].clone(),
                    model: HypModel::PoincareBall,
                };
                let pj = HypPoint {
                    coords: pts[j].clone(),
                    model: HypModel::PoincareBall,
                };
                let d = pi.distance(&pj).max(1e-9);
                let err = d - dist.get(i, j);
                // Euclidean-chord surrogate gradient direction
                let dir = pts[i].sub(&pts[j]);
                let dn = dir.norm().max(1e-12);
                grad = grad.add(&dir.scale(err / dn));
            }
            pts[i] = pts[i].sub(&grad.scale(lr / n as f64));
            // keep inside the ball
            let r = pts[i].norm();
            if r > 0.95 {
                pts[i] = pts[i].scale(0.95 / r);
            }
        }
    }
    pts
}

/// Nickel-Kiela Poincare embedding of a graph by Riemannian SGD on
/// edge-distance loss (connected pairs pulled together, random negatives
/// pushed apart).
#[must_use]
pub fn poincare_embedding_train(
    graph_edges: &[(usize, usize)],
    dim: usize,
    epochs: usize,
    lr: f64,
    rng: &mut Rng,
) -> Vec<VecN> {
    let n = graph_edges
        .iter()
        .map(|&(a, b)| a.max(b))
        .max()
        .unwrap_or(0)
        + 1;
    let mut pts: Vec<VecN> = (0..n)
        .map(|_| VecN::random_gaussian(dim, rng).scale(0.01))
        .collect();
    let dist = |a: &VecN, b: &VecN| {
        let num = a.sub(b).dot(&a.sub(b));
        let den = (1.0 - a.dot(a)) * (1.0 - b.dot(b));
        (1.0 + 2.0 * num / den.max(1e-12)).acosh()
    };
    for _ in 0..epochs {
        for &(u, v) in graph_edges {
            // attract u, v; repel u and a random negative
            let neg = (rng.next_u64() as usize) % n;
            let d_pos = dist(&pts[u], &pts[v]);
            let d_neg = dist(&pts[u], &pts[neg]).max(1e-6);
            for (target, sign, dd) in [(v, 1.0, d_pos), (neg, -0.5, d_neg)] {
                if target == u {
                    continue;
                }
                let dir = pts[u].sub(&pts[target]);
                let dn = dir.norm().max(1e-9);
                // Riemannian scaling: conformal factor correction
                let conf = (1.0 - pts[u].dot(&pts[u])).powi(2) / 4.0;
                let step = dir.scale(-sign * lr * conf * dd.min(3.0) / dn);
                pts[u] = pts[u].add(&step);
                let r = pts[u].norm();
                if r > 0.98 {
                    pts[u] = pts[u].scale(0.98 / r);
                }
            }
        }
    }
    pts
}

/// Curve-shortening flow of a closed disk polygon under the hyperbolic
/// metric (explicit steps toward the hyperbolic midpoint of neighbors).
#[must_use]
pub fn hyp_mean_curvature_flow(curve: &[Complex], dt: f64, steps: usize) -> Vec<Complex> {
    let mut pts = curve.to_vec();
    let n = pts.len();
    for _ in 0..steps {
        let snapshot = pts.clone();
        for i in 0..n {
            let a = HypPoint {
                coords: VecN::from(&[snapshot[(i + n - 1) % n].re, snapshot[(i + n - 1) % n].im]),
                model: HypModel::PoincareDisk,
            };
            let b = HypPoint {
                coords: VecN::from(&[snapshot[(i + 1) % n].re, snapshot[(i + 1) % n].im]),
                model: HypModel::PoincareDisk,
            };
            let mid = a.midpoint(&b);
            let target = c(mid.coords[0], mid.coords[1]);
            pts[i] = snapshot[i] + (target - snapshot[i]) * c(dt, 0.0);
        }
    }
    pts
}

// ---------------------------------------------------------------------------
// Horocycles, equidistants, limit sets
// ---------------------------------------------------------------------------

/// Horocycle at the ideal point through a given interior point: a Euclidean
/// circle tangent to the boundary at the ideal point.
#[must_use]
pub fn horocycle_disk(ideal_point: Complex, through: Complex, n: usize) -> Vec<Complex> {
    let u = ideal_point * c(1.0 / ideal_point.norm(), 0.0);
    // circle tangent to boundary at u passing through `through`:
    // center = u (1 - r), radius r; |through - u(1-r)| = r
    let a = through - u;
    // |a + u r|^2 = r^2 -> |a|^2 + 2 r Re(a conj(u)) + r^2 |u|^2 = r^2
    // with |u| = 1: r = -|a|^2 / (2 Re(a conj(u)))
    let denom = 2.0 * (a * u.conjugate()).re;
    let r = -a.norm_sq() / denom;
    let center = u * c(1.0 - r, 0.0);
    (0..n)
        .map(|k| {
            let th = 2.0 * PI * k as f64 / n as f64;
            center + c(r * th.cos(), r * th.sin())
        })
        .collect()
}

/// Equidistant curve at hyperbolic distance `d` from the geodesic through
/// two boundary-anchored points (sampled along one side).
#[must_use]
pub fn equidistant_curve_disk(geodesic: (Complex, Complex), d: f64, n: usize) -> Vec<Complex> {
    let pts = hyp_geodesic_disk(geodesic.0, geodesic.1, n);
    pts.windows(2)
        .map(|w| {
            let (z, znext) = (w[0], w[1]);
            // move z perpendicular to the geodesic by distance d: translate
            // z to origin, direction of geodesic is (znext - z) rotated 90
            let dir = mobius_disk(znext, z, 0.0);
            let perp = c(-dir.im, dir.re) * c(1.0 / dir.norm().max(1e-30), 0.0);
            let e = (0.5 * d).tanh();
            mobius_disk(perp * c(e, 0.0), c(0.0, 0.0) - z, 0.0)
        })
        .collect()
}

/// Limit set of a Schottky-like group: orbit of a basepoint under words in
/// the generators up to the given length, keeping the deepest images.
#[must_use]
pub fn limit_set_schottky(generators: &[Sl2C], depth: usize) -> Vec<Complex> {
    let mut gens = generators.to_vec();
    let inverses: Vec<Sl2C> = generators.iter().map(Sl2C::inverse).collect();
    gens.extend(inverses);
    let mut frontier = vec![(Sl2C::identity(), usize::MAX)];
    for _ in 0..depth {
        let mut next = Vec::new();
        for &(g, last) in &frontier {
            for (k, h) in gens.iter().enumerate() {
                // avoid immediate backtracking g h h^-1
                let n_gen = gens.len() / 2;
                let is_inverse_of_last =
                    last != usize::MAX && (k + n_gen) % gens.len() == last;
                if is_inverse_of_last {
                    continue;
                }
                next.push((g.compose(h), k));
            }
        }
        frontier = next;
    }
    frontier
        .iter()
        .map(|(g, _)| g.mobius(c(0.0, 0.1)))
        .collect()
}

/// Circles of an Apollonian gasket generated from the classic
/// (-1, 2, 2, 3) Descartes configuration by Vieta reflection; returns
/// (center, radius) pairs (the bounding circle first).
#[must_use]
pub fn apollonian_from_mobius(depth: usize) -> Vec<(Complex, f64)> {
    // (center, signed curvature); the bounding circle has curvature -1
    let mut circ: Vec<(Complex, f64)> = vec![
        (c(0.0, 0.0), -1.0),
        (c(-0.5, 0.0), 2.0),
        (c(0.5, 0.0), 2.0),
        (c(0.0, 2.0 / 3.0), 3.0),
        (c(0.0, -2.0 / 3.0), 3.0),
    ];
    // quadruples (a, b, c, d): reflect d across the mutually tangent triple
    // (a, b, c); note circles 3 and 4 are not tangent to each other, so the
    // valid seeds come from the quadruples {0,1,2,3} and {0,1,2,4}
    let mut frontier = vec![
        (0usize, 1, 3, 2),
        (0, 2, 3, 1),
        (1, 2, 3, 0),
        (0, 1, 4, 2),
        (0, 2, 4, 1),
        (1, 2, 4, 0),
    ];
    for _ in 0..depth {
        let mut next = Vec::new();
        for &(a, b, cc, d) in &frontier {
            let (za, ka) = circ[a];
            let (zb, kb) = circ[b];
            let (zc, kc) = circ[cc];
            let (zd, kd) = circ[d];
            let k_new = 2.0 * (ka + kb + kc) - kd;
            if k_new <= 0.0 || k_new > 1e5 {
                continue;
            }
            let z_new = (za * c(ka, 0.0) + zb * c(kb, 0.0) + zc * c(kc, 0.0))
                * c(2.0 / k_new, 0.0)
                - zd * c(kd / k_new, 0.0);
            // dedup
            if circ
                .iter()
                .any(|&(z, k)| (z - z_new).norm() < 1e-9 && (k - k_new).abs() < 1e-6)
            {
                continue;
            }
            circ.push((z_new, k_new));
            let idx = circ.len() - 1;
            next.push((a, b, idx, cc));
            next.push((a, cc, idx, b));
            next.push((b, cc, idx, a));
        }
        frontier = next;
    }
    circ.into_iter()
        .map(|(z, k)| (z, 1.0 / k.abs()))
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distances_and_invariance() {
        let z = c(0.2, 0.1);
        let w = c(-0.3, 0.4);
        let d = hyp_distance_disk(z, w);
        // invariance under Mobius isometries of the disk
        let a = c(0.3, -0.2);
        let d2 = hyp_distance_disk(mobius_disk(z, a, 0.7), mobius_disk(w, a, 0.7));
        assert!((d - d2).abs() < 1e-12, "{d} vs {d2}");
        // agreement across models
        let zu = disk_to_uhp(z);
        let wu = disk_to_uhp(w);
        assert!((hyp_distance_uhp(zu, wu) - d).abs() < 1e-10);
        let zh = disk_to_hyperboloid(z);
        let wh = disk_to_hyperboloid(w);
        assert!((hyp_distance_hyperboloid(&zh, &wh) - d).abs() < 1e-10);
        // distance from origin: 2 atanh r
        let r = 0.5;
        assert!((hyp_distance_disk(c(0.0, 0.0), c(r, 0.0)) - 2.0 * r.atanh()).abs() < 1e-12);
        // UHP invariance under SL(2, R)
        let m = hyperbolic_translation(0.8, 0.3);
        let d3 = hyp_distance_uhp(mobius_uhp(zu, &m), mobius_uhp(wu, &m));
        assert!((d3 - d).abs() < 1e-9);
    }

    #[test]
    fn test_model_conversions_roundtrip() {
        let z = c(0.35, -0.2);
        assert!((uhp_to_disk(disk_to_uhp(z)) - z).norm() < 1e-12);
        assert!((klein_to_disk(disk_to_klein(z)) - z).norm() < 1e-12);
        assert!((hyperboloid_to_disk(&disk_to_hyperboloid(z)) - z).norm() < 1e-12);
        // HypPoint conversion chain through every model
        let p = HypPoint {
            coords: VecN::from(&[0.35, -0.2]),
            model: HypModel::PoincareDisk,
        };
        for m in [
            HypModel::Klein,
            HypModel::UpperHalfPlane,
            HypModel::Hyperboloid,
            HypModel::PoincareDisk,
        ] {
            let round = p.to(m).to(HypModel::PoincareDisk);
            assert!(
                (round.coords[0] - 0.35).abs() < 1e-9 && (round.coords[1] + 0.2).abs() < 1e-9,
                "roundtrip through {m:?}"
            );
        }
        // 3D ball <-> half space
        let q = Vec3::new(0.2, -0.1, 0.3);
        let hs = ball_to_half_space(q);
        assert!(hs.z > 0.0, "half-space coordinate positive");
        // distances agree between ball and half-space points
        let p1 = HypPoint {
            coords: VecN::from(&[0.2, -0.1, 0.3]),
            model: HypModel::PoincareBall,
        };
        let p2 = HypPoint {
            coords: VecN::from(&[-0.1, 0.15, 0.05]),
            model: HypModel::PoincareBall,
        };
        let d_ball = p1.distance(&p2);
        let d_hs = p1
            .to(HypModel::UpperHalfSpace)
            .distance(&p2.to(HypModel::UpperHalfSpace));
        assert!((d_ball - d_hs).abs() < 1e-9, "{d_ball} vs {d_hs}");
        // hyperboloid points satisfy -x0^2 + |x|^2 = -1
        let h = p1.to(HypModel::Hyperboloid);
        let mink = -h.coords[0] * h.coords[0]
            + h.coords[1] * h.coords[1]
            + h.coords[2] * h.coords[2]
            + h.coords[3] * h.coords[3];
        assert!((mink + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangles_and_trig() {
        // triangle area = pi - angle sum via explicit construction
        let (al, be, ga) = (0.5, 0.6, 0.7);
        let tri = hyp_triangle_from_angles(al, be, ga);
        let pts: Vec<HypPoint> = tri
            .iter()
            .map(|z| HypPoint {
                coords: VecN::from(&[z.re, z.im]),
                model: HypModel::PoincareDisk,
            })
            .collect();
        let a0 = pts[0].angle_at(&pts[1], &pts[2]);
        let a1 = pts[1].angle_at(&pts[2], &pts[0]);
        let a2 = pts[2].angle_at(&pts[0], &pts[1]);
        assert!((a0 - al).abs() < 1e-9, "alpha {a0}");
        assert!((a1 - be).abs() < 1e-9, "beta {a1}");
        assert!((a2 - ga).abs() < 1e-9, "gamma {a2}");
        assert!((hyp_area_triangle(a0, a1, a2) - (PI - al - be - ga)).abs() < 1e-9);
        // law of cosines consistency with measured side lengths
        let side_ab = pts[0].distance(&pts[1]);
        let side_ac = pts[0].distance(&pts[2]);
        let side_bc = pts[1].distance(&pts[2]);
        let from_loc = hyp_law_of_cosines(side_ab, side_ac, a0);
        assert!((from_loc - side_bc).abs() < 1e-9);
        // law of sines
        let alpha_from_los = hyp_law_of_sines(side_bc, side_ac, a1);
        assert!((alpha_from_los - a0).abs() < 1e-9);
        // angle of parallelism decreasing, Pi(0) = pi/2
        assert!((hyp_angle_of_parallelism(0.0) - PI / 2.0).abs() < 1e-12);
        assert!(hyp_angle_of_parallelism(1.0) < hyp_angle_of_parallelism(0.5));
        // circle geometry
        assert!((hyp_circumference(1.0) - 2.0 * PI * 1.0_f64.sinh()).abs() < 1e-12);
        assert!((hyp_area_circle(1.0) - 2.0 * PI * (1.0_f64.cosh() - 1.0)).abs() < 1e-12);
        // 2D volume matches area formula
        assert!((hyp_volume_ball(1.0, 2) - hyp_area_circle(1.0)).abs() < 1e-3);
        // hyperbolic circle in the disk has correct radius from center
        let ctr = c(0.3, 0.1);
        let circ = hyp_circle_disk(ctr, 0.7, 32);
        for z in circ {
            assert!((hyp_distance_disk(z, ctr) - 0.7).abs() < 1e-9);
        }
    }

    #[test]
    fn test_geodesics_and_isometries() {
        // geodesic arc is orthogonal to the boundary: the circle satisfies
        // |c|^2 = r^2 + 1
        let z = c(0.5, 0.2);
        let w = c(-0.1, 0.6);
        let (center, r) = hyp_geodesic_circle_disk(z, w).unwrap();
        assert!((center.norm_sq() - (r * r + 1.0)).abs() < 1e-10);
        // sampled geodesic has evenly spaced hyperbolic arc lengths
        let path = hyp_geodesic_disk(z, w, 10);
        let total = hyp_distance_disk(z, w);
        for (k, seg) in path.windows(2).enumerate() {
            let d = hyp_distance_disk(seg[0], seg[1]);
            assert!((d - total / 10.0).abs() < 1e-9, "segment {k}");
        }
        // midpoint equidistant
        let a = HypPoint {
            coords: VecN::from(&[z.re, z.im]),
            model: HypModel::PoincareDisk,
        };
        let b = HypPoint {
            coords: VecN::from(&[w.re, w.im]),
            model: HypModel::PoincareDisk,
        };
        let mid = a.midpoint(&b);
        assert!((mid.distance(&a) - mid.distance(&b)).abs() < 1e-9);
        // reflection across a geodesic is an isometry that fixes the geodesic
        let p = HypPoint {
            coords: VecN::from(&[0.1, 0.3]),
            model: HypModel::PoincareDisk,
        };
        let refl = p.reflect_across((&a, &b));
        assert!((refl.distance(&a) - p.distance(&a)).abs() < 1e-9);
        let double = refl.reflect_across((&a, &b));
        assert!(double.distance(&p) < 1e-9, "involution");
        // isometry from two points maps them correctly
        let (ma, mth) = isometry_disk_from_two_points(z, w, c(0.0, 0.0), c(0.55, 0.1)).unwrap_or((c(0.0,0.0), 0.0));
        let img1 = mobius_disk(z, ma, mth);
        // only require the distance pattern (the pair may differ in distance)
        let _ = img1;
        // constructive test with consistent distances:
        let w1 = c(0.1, -0.2);
        let d0 = hyp_distance_disk(z, w);
        // build w2 at the same distance from w1 (translate w by the map
        // sending z to w1)
        let (aa, tt) = isometry_disk_from_two_points(
            z,
            w,
            w1,
            {
                // image of w under some isometry sending z -> w1
                let g = |x: Complex| mobius_disk(x, z, 0.0); // z -> 0
                let h = |x: Complex| {
                    // 0 -> w1
                    (x + w1) / (c(1.0, 0.0) + w1.conjugate() * x)
                };
                h(g(w))
            },
        )
        .expect("isometry exists");
        let i1 = mobius_disk(z, aa, tt);
        assert!((i1 - w1).norm() < 1e-10, "isometry image {i1:?}");
        let i2 = mobius_disk(w, aa, tt);
        assert!((hyp_distance_disk(i1, i2) - d0).abs() < 1e-8);
        // horocycle passes through the requested point and is tangent to
        // the boundary
        let horo = horocycle_disk(c(1.0, 0.0), c(0.2, 0.0), 64);
        let on = horo
            .iter()
            .any(|&h| (h - c(0.2, 0.0)).norm() < 0.05);
        assert!(on, "horocycle through point");
        assert!(horo.iter().all(|h| h.norm() < 1.0 + 1e-9));
        // equidistant curve stays at constant distance from the geodesic
        let eq = equidistant_curve_disk((z, w), 0.5, 24);
        for &q in eq.iter().skip(2).take(20) {
            // distance from q to the geodesic (minimum over samples)
            let dmin = path
                .iter()
                .map(|&g| hyp_distance_disk(q, g))
                .fold(f64::MAX, f64::min);
            assert!((dmin - 0.5).abs() < 0.02, "equidistant {dmin}");
        }
    }

    #[test]
    fn test_tiling() {
        assert!(hyp_tiling_exists(7, 3));
        assert!(!hyp_tiling_exists(4, 4));
        assert!(!hyp_tiling_exists(6, 3));
        // {7,3}: all heptagons congruent — equal hyperbolic edge lengths
        let tiles = hyp_tiling(7, 3, 1, HypModel::PoincareDisk);
        assert!(tiles.len() > 1, "tiling produced {} tiles", tiles.len());
        let mut lengths = Vec::new();
        for tile in &tiles {
            for e in 0..tile.len() {
                lengths.push(hyp_distance_disk(tile[e], tile[(e + 1) % tile.len()]));
            }
        }
        let l0 = lengths[0];
        for (k, l) in lengths.iter().enumerate() {
            assert!((l - l0).abs() < 1e-6, "edge {k}: {l} vs {l0}");
        }
        // fundamental polygon for genus 2: octagon with total angle 2 pi
        let oct = fundamental_polygon_genus(2);
        assert_eq!(oct.len(), 8);
        let pts: Vec<HypPoint> = oct
            .iter()
            .map(|z| HypPoint {
                coords: VecN::from(&[z.re, z.im]),
                model: HypModel::PoincareDisk,
            })
            .collect();
        let total: f64 = (0..8)
            .map(|i| pts[i].angle_at(&pts[(i + 7) % 8], &pts[(i + 1) % 8]))
            .sum();
        assert!((total - 2.0 * PI).abs() < 1e-6, "angle sum {total}");
    }

    #[test]
    fn test_computational_geometry() {
        let sites = vec![c(0.0, 0.0), c(0.4, 0.0), c(0.0, 0.4), c(-0.3, -0.2)];
        // Delaunay produces triangles covering all sites
        let tris = hyp_delaunay_disk(&sites);
        assert!(!tris.is_empty());
        let mut used = vec![false; sites.len()];
        for t in &tris {
            for &v in t {
                used[v] = true;
            }
        }
        assert!(used.iter().all(|&u| u));
        // Voronoi cell of a site contains points closer to it
        let cells = hyp_voronoi_disk(&sites, 16);
        assert_eq!(cells.len(), sites.len());
        for z in &cells[1] {
            let d1 = hyp_distance_disk(*z, sites[1]);
            for (k, &s) in sites.iter().enumerate() {
                if k != 1 {
                    assert!(hyp_distance_disk(*z, s) >= d1 - 1e-6);
                }
            }
        }
        // convex hull contains all points on the correct side (Klein test)
        let pts = vec![c(0.0, 0.0), c(0.5, 0.0), c(0.0, 0.5), c(0.2, 0.15), c(-0.4, 0.1)];
        let hull = hyp_convex_hull_disk(&pts);
        assert!(hull.len() >= 3 && hull.len() <= pts.len());
        // interior point excluded
        assert!(hull.iter().all(|&h| (h - c(0.2, 0.15)).norm() > 1e-9));
        // centroid of a symmetric configuration is the center
        let sym = vec![c(0.3, 0.0), c(-0.3, 0.0), c(0.0, 0.3), c(0.0, -0.3)];
        let ctr = hyp_centroid_disk(&sym, 40);
        assert!(ctr.norm() < 1e-8, "centroid {ctr:?}");
        // mean curvature flow shrinks a circle
        let circ = hyp_circle_disk(c(0.0, 0.0), 1.0, 32);
        let flowed = hyp_mean_curvature_flow(&circ, 0.5, 20);
        let r_before: f64 = circ.iter().map(|z| z.norm()).sum::<f64>() / 32.0;
        let r_after: f64 = flowed.iter().map(|z| z.norm()).sum::<f64>() / 32.0;
        assert!(r_after < r_before);
    }

    #[test]
    fn test_embeddings() {
        // binary tree, depth 3
        let mut adj = vec![Vec::new(); 15];
        for i in 0..7 {
            adj[i].push(2 * i + 1);
            adj[i].push(2 * i + 2);
            adj[2 * i + 1].push(i);
            adj[2 * i + 2].push(i);
        }
        let pos = hyp_embed_tree(&adj, 0);
        // tree distance vs hyperbolic distance distortion:
        // compute graph distances by BFS
        let bfs = |start: usize| -> Vec<usize> {
            let mut d = vec![usize::MAX; 15];
            d[start] = 0;
            let mut q = std::collections::VecDeque::from([start]);
            while let Some(v) = q.pop_front() {
                for &u in &adj[v] {
                    if d[u] == usize::MAX {
                        d[u] = d[v] + 1;
                        q.push_back(u);
                    }
                }
            }
            d
        };
        let edge_len = 2.0 * (15.0_f64).ln();
        let mut max_ratio = 0.0_f64;
        let mut min_ratio = f64::MAX;
        for i in 0..15 {
            let gd = bfs(i);
            for j in 0..15 {
                if i == j {
                    continue;
                }
                let hd = hyp_distance_disk(pos[i], pos[j]);
                let td = gd[j] as f64 * edge_len;
                let ratio = hd / td;
                max_ratio = max_ratio.max(ratio);
                min_ratio = min_ratio.min(ratio);
            }
        }
        let distortion = max_ratio / min_ratio;
        assert!(distortion < 1.1, "tree distortion {distortion}");
        // MDS embedding roughly reproduces a small metric
        let mut dist = Matrix::zeros(3, 3);
        dist.set(0, 1, 1.0);
        dist.set(1, 0, 1.0);
        dist.set(0, 2, 1.0);
        dist.set(2, 0, 1.0);
        dist.set(1, 2, 1.5);
        dist.set(2, 1, 1.5);
        let emb = hyp_embed_graph_mds(&dist, 2, 400);
        let p = |k: usize| HypPoint {
            coords: emb[k].clone(),
            model: HypModel::PoincareBall,
        };
        let d01 = p(0).distance(&p(1));
        let d12 = p(1).distance(&p(2));
        assert!((d01 - 1.0).abs() < 0.15, "mds d01 {d01}");
        assert!((d12 - 1.5).abs() < 0.2, "mds d12 {d12}");
        // Poincare embedding: connected nodes end closer than random pairs
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)];
        let mut rng = Rng::new(17);
        let emb2 = poincare_embedding_train(&edges, 2, 200, 0.05, &mut rng);
        let dist2 = |a: usize, b: usize| {
            let num = emb2[a].sub(&emb2[b]).dot(&emb2[a].sub(&emb2[b]));
            let den = (1.0 - emb2[a].dot(&emb2[a])) * (1.0 - emb2[b].dot(&emb2[b]));
            (1.0 + 2.0 * num / den).acosh()
        };
        let connected: f64 = edges.iter().map(|&(a, b)| dist2(a, b)).sum::<f64>() / 8.0;
        let cross = dist2(0, 4) + dist2(1, 5) + dist2(2, 6) + dist2(3, 7);
        assert!(connected < cross / 4.0, "embedding separates components");
    }

    #[test]
    fn test_lorentz_and_limit_sets() {
        // boost carries the basepoint to the target hyperboloid point
        let p = disk_to_hyperboloid(c(0.3, -0.2));
        let b = lorentz_boost_hyperboloid(&p);
        let base = vec![1.0, 0.0, 0.0];
        let img = b.mul_vec(&base).unwrap();
        for (a, e) in img.iter().zip(&p.data) {
            assert!((a - e).abs() < 1e-10);
        }
        // boost preserves the Minkowski form
        let q = disk_to_hyperboloid(c(-0.1, 0.4));
        let bq = VecN::from(&b.mul_vec(&q.data).unwrap());
        let mink = -bq[0] * bq[0] + bq[1] * bq[1] + bq[2] * bq[2];
        assert!((mink + 1.0).abs() < 1e-10);
        // Schottky limit set points accumulate near the boundary/real axis
        let g1 = Sl2C {
            m: [
                [c(3.0, 0.0), c(0.0, 0.0)],
                [c(0.0, 0.0), c(1.0 / 3.0, 0.0)],
            ],
        };
        let g2 = Sl2C {
            m: [
                [c(1.66, 0.0), c(1.33, 0.0)],
                [c(1.33, 0.0), c(1.66, 0.0)],
            ],
        };
        let limit = limit_set_schottky(&[g1, g2], 5);
        assert!(!limit.is_empty());
        assert!(limit.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        // deep words compress toward the limit set: imaginary parts shrink
        let mut ims: Vec<f64> = limit.iter().map(|z| z.im.abs()).collect();
        ims.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q75 = ims[3 * ims.len() / 4];
        assert!(q75 < 0.1, "limit set near real axis: q75 = {q75}");
        // Apollonian circles are tangent to their parents
        let circles = apollonian_from_mobius(2);
        assert!(circles.len() > 5);
        for (z, r) in circles.iter().skip(5) {
            // each generated circle is tangent to at least two others
            let tangencies = circles
                .iter()
                .filter(|(z2, r2)| {
                    let d = (*z - *z2).norm();
                    (d - (r + r2)).abs() < 1e-5 || (d - (r - r2).abs()).abs() < 1e-5
                })
                .count();
            assert!(tangencies >= 3, "tangencies {tangencies}");
        }
    }

    #[test]
    fn test_origin_polar_and_display_coords() {
        // the origin of each 2D model, and the fact that they are all the
        // same point of hyperbolic space
        let disk = HypPoint::origin(HypModel::PoincareDisk, 2);
        let klein = HypPoint::origin(HypModel::Klein, 2);
        let uhp = HypPoint::origin(HypModel::UpperHalfPlane, 2);
        let hyp = HypPoint::origin(HypModel::Hyperboloid, 2);
        assert_eq!(disk.coords.dim(), 2);
        assert!(disk.coords.norm() < 1e-15, "disk origin at 0");
        assert!(klein.coords.norm() < 1e-15, "klein origin at 0");
        assert!(
            uhp.coords[0].abs() < 1e-15 && (uhp.coords[1] - 1.0).abs() < 1e-15,
            "upper half-plane origin is i"
        );
        assert_eq!(hyp.coords.dim(), 3);
        assert!(
            (hyp.coords[0] - 1.0).abs() < 1e-15
                && hyp.coords[1].abs() < 1e-15
                && hyp.coords[2].abs() < 1e-15,
            "hyperboloid origin is the apex"
        );
        for a in [&disk, &klein, &uhp, &hyp] {
            assert!(a.distance(a) < 1e-12, "distance to itself");
            for b in [&disk, &klein, &uhp, &hyp] {
                assert!(a.distance(b) < 1e-12, "origins coincide across models");
            }
        }
        // 3D origins too: the ball origin is the center, the half-space
        // origin sits one unit above the boundary plane
        let ball3 = HypPoint::origin(HypModel::PoincareBall, 3);
        let half3 = HypPoint::origin(HypModel::UpperHalfSpace, 3);
        assert!(ball3.coords.norm() < 1e-15 && ball3.coords.dim() == 3);
        assert!((half3.coords[2] - 1.0).abs() < 1e-15 && half3.coords[0].abs() < 1e-15);
        assert!(ball3.distance(&half3) < 1e-12);
        // to_euclidean_display: disk-like models pass through, and the
        // hyperboloid is projected into the unit ball
        assert!(disk.to_euclidean_display().norm() < 1e-15, "origin displays at 0");
        assert!(hyp.to_euclidean_display().norm() < 1e-15);
        assert_eq!(hyp.to_euclidean_display().dim(), 2);
        let p = HypPoint {
            coords: VecN::from(&[0.35, -0.2]),
            model: HypModel::PoincareDisk,
        };
        let shown = p.to_euclidean_display();
        assert!(
            (shown[0] - 0.35).abs() < 1e-15 && (shown[1] + 0.2).abs() < 1e-15,
            "disk coordinates pass through"
        );
        let shown_h = p.to(HypModel::Hyperboloid).to_euclidean_display();
        assert!(
            (shown_h[0] - 0.35).abs() < 1e-12 && (shown_h[1] + 0.2).abs() < 1e-12,
            "hyperboloid displays as its disk image {shown_h:?}"
        );
        assert!(shown_h.norm() < 1.0, "display stays inside the unit disk");
        // from_polar: hyperbolic radius r, Euclidean radius tanh(r/2)
        for &r in &[0.0_f64, 0.25, 1.0, 3.0] {
            for &th in &[0.0_f64, 0.7, 2.9, -1.4] {
                let q = HypPoint::from_polar(r, th);
                assert_eq!(q.model, HypModel::PoincareDisk);
                assert!(
                    (q.distance(&disk) - r).abs() < 1e-10,
                    "polar radius {r} came back as {}",
                    q.distance(&disk)
                );
                let z = c(q.coords[0], q.coords[1]);
                assert!(
                    (hyp_distance_disk(c(0.0, 0.0), z) - r).abs() < 1e-10,
                    "disk-model distance"
                );
                assert!((z.norm() - (0.5 * r).tanh()).abs() < 1e-12, "euclidean radius");
                if r > 1e-9 {
                    let ang = z.im.atan2(z.re);
                    let diff = (ang - th + PI).rem_euclid(2.0 * PI) - PI;
                    assert!(diff.abs() < 1e-12, "polar angle {ang} vs {th}");
                }
            }
        }
        // hyperbolic law of cosines for two polar points sharing the origin:
        // cosh d = cosh r1 cosh r2 - sinh r1 sinh r2 cos(dtheta)
        let (r1, r2) = (0.8_f64, 1.7_f64);
        for &dth in &[0.0_f64, 0.5, 2.0, PI] {
            let a = HypPoint::from_polar(r1, 0.3);
            let b = HypPoint::from_polar(r2, 0.3 + dth);
            let d = a.distance(&b);
            let want = (r1.cosh() * r2.cosh() - r1.sinh() * r2.sinh() * dth.cos()).acosh();
            assert!(
                (d - want).abs() < 1e-9,
                "law of cosines at dtheta={dth}: {d} vs {want}"
            );
            // and the closed form matches the crate's own law of cosines
            let via_trig = hyp_law_of_cosines(r1, r2, dth);
            assert!((d - via_trig).abs() < 1e-9, "vs hyp_law_of_cosines");
        }
        assert!(HypPoint::from_polar(0.0, 1.2).distance(&disk) < 1e-12);
    }

    #[test]
    fn test_parabolic_isometry() {
        // z -> z + t on the upper half-plane
        let t = 0.75;
        let par = parabolic(t);
        assert!((par.m[0][0] - 1.0).abs() < 1e-15 && (par.m[1][1] - 1.0).abs() < 1e-15);
        assert!((par.m[1][0]).abs() < 1e-15 && (par.m[0][1] - t).abs() < 1e-15);
        let det = par.m[0][0] * par.m[1][1] - par.m[0][1] * par.m[1][0];
        assert!((det - 1.0).abs() < 1e-15, "unimodular");
        let trace = par.m[0][0] + par.m[1][1];
        assert!((trace.abs() - 2.0).abs() < 1e-15, "|trace| = 2 is parabolic");
        let pts = [c(0.0, 1.0), c(0.4, 2.5), c(-1.3, 0.2)];
        for &z in &pts {
            let img = mobius_uhp(z, &par);
            assert!((img - (z + c(t, 0.0))).norm() < 1e-12, "translation {img:?}");
            assert!(img.im > 0.0, "half-plane preserved");
        }
        // it is an isometry of the hyperbolic metric
        for &z in &pts {
            for &w in &pts {
                let d = hyp_distance_uhp(z, w);
                let d2 = hyp_distance_uhp(mobius_uhp(z, &par), mobius_uhp(w, &par));
                assert!((d - d2).abs() < 1e-10, "distance {d} vs {d2}");
            }
        }
        // one-parameter group: p(a) p(b) = p(a + b), and p(0) is the identity
        let comp = parabolic(0.3).compose(&parabolic(-1.1));
        let want = parabolic(-0.8);
        for i in 0..2 {
            for j in 0..2 {
                assert!((comp.m[i][j] - want.m[i][j]).abs() < 1e-12, "group law");
            }
        }
        let z = c(0.2, 1.4);
        assert!((mobius_uhp(z, &parabolic(0.0)) - z).norm() < 1e-15, "identity");
        assert!(
            (mobius_uhp(mobius_uhp(z, &par), &parabolic(-t)) - z).norm() < 1e-12,
            "inverse"
        );
        // no interior fixed point (z + t = z is unsolvable), and the single
        // fixed boundary point is infinity: in the disk model, 1
        for &z in &pts {
            assert!((mobius_uhp(z, &par) - z).norm() > 0.5 * t, "no interior fixed point");
        }
        let far = c(0.0, 1e7);
        let disk_far = uhp_to_disk(far);
        let disk_img = uhp_to_disk(mobius_uhp(far, &par));
        assert!((disk_far - c(1.0, 0.0)).norm() < 1e-6, "boundary point 1");
        assert!(
            (disk_img - disk_far).norm() < 1e-6,
            "the ideal fixed point is unmoved: {disk_img:?}"
        );
        // parabolic elements have no interior fixed point, unlike elliptic
        // rotations, which fix i
        let rot = hyperbolic_rotation(0.6);
        assert!((mobius_uhp(c(0.0, 1.0), &rot) - c(0.0, 1.0)).norm() < 1e-12);
    }
}
