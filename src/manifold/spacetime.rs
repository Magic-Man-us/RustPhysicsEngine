//! Special and general relativity: four-vectors and Lorentz transforms,
//! Rindler and Kruskal coordinates, Schwarzschild and Kerr geodesics,
//! gravitational lensing, black hole thermodynamics, cosmological
//! distances, inspiral waveforms, and Kaluza-Klein reduction.
//!
//! Kinematics and geometry use geometric units (G = c = 1) with the
//! mostly-minus signature (+, -, -, -) unless stated otherwise; the
//! thermodynamic and cosmological helpers use SI units.

use crate::linalg::{Mat3, Mat4};
use crate::manifold::geodesic::{GeodesicState, Integrator};
use crate::manifold::lie::{Sl2C, So3};
use crate::manifold::metric::{frw_metric, schwarzschild_metric_fn, Metric, Sig};
use crate::manifold::vecn::VecN;
use crate::math::constants::{C, G, HBAR, K_B};
use crate::math::Vec3;

const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// four-vectors
// ---------------------------------------------------------------------------

/// A spacetime four-vector (t, x) in units with c = 1.
#[derive(Debug, Clone, Copy)]
pub struct FourVector {
    pub t: f64,
    pub x: Vec3,
}

impl FourVector {
    #[must_use]
    pub fn new(t: f64, x: Vec3) -> Self {
        FourVector { t, x }
    }

    /// Minkowski inner product in the mostly-minus convention
    /// (+, -, -, -): a.b = a_t b_t - a_x . b_x. See
    /// [`FourVector::minkowski_dot_sig`] for the other signature.
    #[must_use]
    pub fn minkowski_dot(&self, o: &FourVector) -> f64 {
        self.t * o.t - self.x.dot(&o.x)
    }

    /// Minkowski inner product with an explicit signature convention.
    #[must_use]
    pub fn minkowski_dot_sig(&self, o: &FourVector, sig: Sig) -> f64 {
        match sig {
            Sig::MostlyMinus => self.minkowski_dot(o),
            Sig::MostlyPlus => -self.minkowski_dot(o),
        }
    }

    /// Invariant norm squared t^2 - |x|^2 (positive for timelike vectors).
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.minkowski_dot(self)
    }

    #[must_use]
    pub fn is_timelike(&self) -> bool {
        self.norm_squared() > 0.0
    }

    #[must_use]
    pub fn is_spacelike(&self) -> bool {
        self.norm_squared() < 0.0
    }

    #[must_use]
    pub fn is_null(&self, tol: f64) -> bool {
        self.norm_squared().abs() <= tol * (self.t * self.t + self.x.magnitude_squared())
    }

    /// Active boost: the transform that maps a particle at rest to
    /// three-velocity `v` (|v| < 1).
    #[must_use]
    pub fn boost(&self, v: Vec3) -> FourVector {
        LorentzTransform::boost(v).apply(*self)
    }

    /// Spatial rotation of the vector part.
    #[must_use]
    pub fn rotate(&self, r: &So3) -> FourVector {
        FourVector {
            t: self.t,
            x: r.0.mul_vec(self.x),
        }
    }

    /// Four-velocity gamma (1, v) of a particle moving at `v` (|v| < 1).
    #[must_use]
    pub fn from_velocity(v: Vec3) -> FourVector {
        let gamma = 1.0 / (1.0 - v.magnitude_squared()).sqrt();
        FourVector {
            t: gamma,
            x: v * gamma,
        }
    }

    /// Four-momentum m gamma (1, v) of a mass m moving at `v`.
    #[must_use]
    pub fn from_momentum(m: f64, v: Vec3) -> FourVector {
        let u = FourVector::from_velocity(v);
        u.scale(m)
    }

    /// Energy (time component, c = 1).
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.t
    }

    /// Spatial momentum (vector part).
    #[must_use]
    pub fn spatial_momentum(&self) -> Vec3 {
        self.x
    }

    /// Rapidity of the three-velocity x/t: atanh(|x|/t).
    #[must_use]
    pub fn rapidity(&self) -> f64 {
        (self.x.magnitude() / self.t).atanh()
    }

    #[must_use]
    pub fn scale(&self, k: f64) -> FourVector {
        FourVector {
            t: self.t * k,
            x: self.x * k,
        }
    }

    /// Proper time along the straight worldline from this event to `o`
    /// (zero if the separation is not timelike).
    #[must_use]
    pub fn proper_time_to(&self, o: &FourVector) -> f64 {
        let d = *o - *self;
        d.norm_squared().max(0.0).sqrt()
    }
}

impl std::ops::Add for FourVector {
    type Output = FourVector;
    fn add(self, o: FourVector) -> FourVector {
        FourVector {
            t: self.t + o.t,
            x: self.x + o.x,
        }
    }
}

impl std::ops::Sub for FourVector {
    type Output = FourVector;
    fn sub(self, o: FourVector) -> FourVector {
        FourVector {
            t: self.t - o.t,
            x: self.x - o.x,
        }
    }
}

// ---------------------------------------------------------------------------
// Lorentz transforms
// ---------------------------------------------------------------------------

/// A Lorentz transformation as a 4x4 matrix acting on (t, x, y, z).
#[derive(Debug, Clone, Copy)]
pub struct LorentzTransform(pub Mat4);

impl LorentzTransform {
    #[must_use]
    pub fn identity() -> Self {
        LorentzTransform(Mat4::identity())
    }

    /// Active boost taking a particle at rest to three-velocity `v`.
    #[must_use]
    pub fn boost(v: Vec3) -> Self {
        let b2 = v.magnitude_squared();
        if b2 < 1e-300 {
            return Self::identity();
        }
        let gamma = 1.0 / (1.0 - b2).sqrt();
        let mut m = Mat4::identity();
        m.data[0][0] = gamma;
        let n = [v.x, v.y, v.z];
        for i in 0..3 {
            m.data[0][i + 1] = gamma * n[i];
            m.data[i + 1][0] = gamma * n[i];
            for j in 0..3 {
                m.data[i + 1][j + 1] =
                    (if i == j { 1.0 } else { 0.0 }) + (gamma - 1.0) * n[i] * n[j] / b2;
            }
        }
        LorentzTransform(m)
    }

    /// Boost along +x with speed `beta`.
    #[must_use]
    pub fn boost_x(beta: f64) -> Self {
        Self::boost(Vec3::new(beta, 0.0, 0.0))
    }

    /// Spatial rotation embedded as a Lorentz transform.
    #[must_use]
    pub fn rotation(r: &So3) -> Self {
        let mut m = Mat4::identity();
        for i in 0..3 {
            for j in 0..3 {
                m.data[i + 1][j + 1] = r.0.data[i][j];
            }
        }
        LorentzTransform(m)
    }

    #[must_use]
    pub fn compose(&self, o: &LorentzTransform) -> LorentzTransform {
        LorentzTransform(self.0.mul_mat(&o.0))
    }

    /// Inverse via eta Lambda^T eta (exact for Lorentz matrices).
    #[must_use]
    pub fn inverse(&self) -> LorentzTransform {
        let lt = self.0.transpose();
        let mut m = Mat4::zero();
        for r in 0..4 {
            for c in 0..4 {
                let sr = if r == 0 { 1.0 } else { -1.0 };
                let sc = if c == 0 { 1.0 } else { -1.0 };
                m.data[r][c] = sr * sc * lt.data[r][c];
            }
        }
        LorentzTransform(m)
    }

    #[must_use]
    pub fn apply(&self, v: FourVector) -> FourVector {
        let out = self.0.mul_vec4([v.t, v.x.x, v.x.y, v.x.z]);
        FourVector {
            t: out[0],
            x: Vec3::new(out[1], out[2], out[3]),
        }
    }

    /// Check Lambda^T eta Lambda = eta to tolerance `tol`.
    #[must_use]
    pub fn is_lorentz(&self, tol: f64) -> bool {
        let l = &self.0;
        for a in 0..4 {
            for b in 0..4 {
                let mut s = 0.0;
                for m in 0..4 {
                    let eta = if m == 0 { 1.0 } else { -1.0 };
                    s += l.data[m][a] * eta * l.data[m][b];
                }
                let want = if a != b {
                    0.0
                } else if a == 0 {
                    1.0
                } else {
                    -1.0
                };
                if (s - want).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Thomas-Wigner rotation of the composition B(v1) B(v2): the residual
    /// spatial rotation R with B(v1) B(v2) = B(v1 + v2) R.
    #[must_use]
    pub fn thomas_wigner_rotation(v1: Vec3, v2: Vec3) -> So3 {
        let l = Self::boost(v1).compose(&Self::boost(v2));
        let rest = l.apply(FourVector::new(1.0, Vec3::new(0.0, 0.0, 0.0)));
        let u = rest.x * (1.0 / rest.t);
        let r4 = Self::boost(u).inverse().compose(&l);
        let mut r = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                r.data[i][j] = r4.0.data[i + 1][j + 1];
            }
        }
        So3::project(&r)
    }

    /// Relativistic velocity addition: the velocity of a particle moving at
    /// `v` in a frame that itself moves at `u` (i.e. B(u) applied to the
    /// four-velocity of `v`).
    #[must_use]
    pub fn velocity_addition(u: Vec3, v: Vec3) -> Vec3 {
        let w = Self::boost(u).apply(FourVector::from_velocity(v));
        w.x * (1.0 / w.t)
    }

    /// The Lorentz transform covered by an SL(2, C) element.
    #[must_use]
    pub fn from_sl2c(m: &Sl2C) -> LorentzTransform {
        LorentzTransform(m.to_lorentz())
    }

    /// An SL(2, C) element covering this transform (defined up to sign),
    /// via the polar split Lambda = B(u) R.
    #[must_use]
    pub fn to_sl2c(&self) -> Sl2C {
        let rest = self.apply(FourVector::new(1.0, Vec3::new(0.0, 0.0, 0.0)));
        let u = rest.x * (1.0 / rest.t);
        let speed = u.magnitude();
        let a_boost = if speed > 1e-14 {
            Sl2C::from_lorentz_boost(u * (1.0 / speed), speed.atanh())
        } else {
            Sl2C::identity()
        };
        let r4 = Self::boost(u).inverse().compose(self);
        let mut r = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                r.data[i][j] = r4.0.data[i + 1][j + 1];
            }
        }
        let axis_angle = So3::project(&r).log();
        let angle = axis_angle.magnitude();
        let a_rot = if angle > 1e-14 {
            su2_from_axis_angle(axis_angle * (1.0 / angle), angle)
        } else {
            Sl2C::identity()
        };
        a_boost.compose(&a_rot)
    }
}

fn su2_from_axis_angle(n: Vec3, angle: f64) -> Sl2C {
    use crate::fractals::Complex;
    let c = (0.5 * angle).cos();
    let s = (0.5 * angle).sin();
    // exp(-i angle/2 n.sigma)
    Sl2C {
        m: [
            [
                Complex::new(c, -s * n.z),
                Complex::new(-s * n.y, -s * n.x),
            ],
            [
                Complex::new(s * n.y, -s * n.x),
                Complex::new(c, s * n.z),
            ],
        ],
    }
}

/// Causal relation of event `b` relative to event `a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Causal {
    Past,
    Future,
    Spacelike,
    Null,
}

/// Classify the separation b - a on the light cone of `a`.
#[must_use]
pub fn light_cone_check(a: FourVector, b: FourVector) -> Causal {
    let d = b - a;
    let s2 = d.norm_squared();
    let scale = d.t * d.t + d.x.magnitude_squared();
    if s2.abs() <= 1e-12 * scale.max(1e-300) {
        Causal::Null
    } else if s2 < 0.0 {
        Causal::Spacelike
    } else if d.t > 0.0 {
        Causal::Future
    } else {
        Causal::Past
    }
}

/// A hyperplane of events: all x with normal . x = offset (Minkowski dot).
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: FourVector,
    pub offset: f64,
}

/// The simultaneity hyperplane through `event` for an observer moving at
/// `observer_vel`: events x with u . (x - event) = 0 for the observer
/// four-velocity u.
#[must_use]
pub fn simultaneity_plane(observer_vel: Vec3, event: FourVector) -> Plane {
    let u = FourVector::from_velocity(observer_vel);
    Plane {
        normal: u,
        offset: u.minkowski_dot(&event),
    }
}

/// Ages (stay-at-home, traveler) after coordinate time `t_coordinate` with
/// the traveler cruising at speed `v`.
#[must_use]
pub fn twin_paradox_ages(v: f64, t_coordinate: f64) -> (f64, f64) {
    (t_coordinate, t_coordinate * (1.0 - v * v).sqrt())
}

/// Relativistic rocket with constant proper acceleration: returns
/// (coordinate time, distance, speed) after proper time `tau`.
#[must_use]
pub fn relativistic_rocket(accel_proper: f64, tau: f64) -> (f64, f64, f64) {
    let a = accel_proper;
    (
        (a * tau).sinh() / a,
        ((a * tau).cosh() - 1.0) / a,
        (a * tau).tanh(),
    )
}

/// Rindler coordinates (eta, xi) of the Minkowski event (t, x) in the
/// right wedge x > |t|, normalized so the observer at proper acceleration
/// `a` sits at xi = 1/a: t = xi sinh(a eta), x = xi cosh(a eta).
#[must_use]
pub fn rindler_coords(t: f64, x: f64, a: f64) -> (f64, f64) {
    ((t / x).atanh() / a, (x * x - t * t).max(0.0).sqrt())
}

/// Distance from a uniformly accelerated observer to their Rindler
/// horizon: c^2 / a (geometric units: 1/a).
#[must_use]
pub fn rindler_horizon(a: f64) -> f64 {
    1.0 / a
}

/// Unruh temperature of a uniformly accelerated observer (SI units):
/// T = hbar a / (2 pi c k_B).
#[must_use]
pub fn unruh_temperature(a: f64) -> f64 {
    HBAR * a / (2.0 * PI * C * K_B)
}

// ---------------------------------------------------------------------------
// Schwarzschild coordinates and orbits
// ---------------------------------------------------------------------------

/// Kruskal-Szekeres coordinates (T, X) of the Schwarzschild event (t, r),
/// smooth across the horizon r = 2M (exterior region I and interior
/// region II).
#[must_use]
pub fn kruskal_from_schwarzschild(t: f64, r: f64, m: f64) -> (f64, f64) {
    let f = r / (2.0 * m) - 1.0;
    let e = (r / (4.0 * m)).exp();
    let (sh, ch) = ((t / (4.0 * m)).sinh(), (t / (4.0 * m)).cosh());
    if f >= 0.0 {
        let rho = f.sqrt() * e;
        (rho * sh, rho * ch)
    } else {
        let rho = (-f).sqrt() * e;
        (rho * ch, rho * sh)
    }
}

/// Penrose diagram coordinates: Kruskal null coordinates compactified with
/// arctangent; returns (T, X) of the conformal diagram.
#[must_use]
pub fn penrose_diagram_coords(t: f64, r: f64, m: f64) -> (f64, f64) {
    let (kt, kx) = kruskal_from_schwarzschild(t, r, m);
    let u = (kt - kx).atan();
    let v = (kt + kx).atan();
    ((v + u) / 2.0, (v - u) / 2.0)
}

/// Ingoing Eddington-Finkelstein null coordinate v = t + r* with the
/// tortoise coordinate r* = r + 2M ln|r/2M - 1|.
#[must_use]
pub fn eddington_finkelstein(t: f64, r: f64, m: f64) -> f64 {
    t + r + 2.0 * m * (r / (2.0 * m) - 1.0).abs().max(1e-300).ln()
}

/// The Schwarzschild metric wired into the finite-difference [`Metric`]
/// machinery (coordinates t, r, theta, phi; signature -+++).
#[must_use]
pub fn schwarzschild_geodesic_metric(m: f64) -> Metric {
    Metric::new(4, schwarzschild_metric_fn(m))
}

/// Full timelike Schwarzschild orbit in the equatorial plane from the
/// first integrals: energy `e` and angular momentum `l` per unit mass,
/// starting at r0 (infalling if r0 is not a turning point). Returns
/// (t, r, phi, tau) samples every proper-time step `dt`.
#[must_use]
pub fn orbit_schwarzschild_full(
    m: f64,
    e: f64,
    l: f64,
    r0: f64,
    tau_end: f64,
    dt: f64,
) -> Vec<(f64, f64, f64, f64)> {
    let f = |r: f64| 1.0 - 2.0 * m / r;
    let rdd = |r: f64| -m / (r * r) + l * l / (r * r * r) - 3.0 * m * l * l / (r * r * r * r);
    let rd0_sq = e * e - f(r0) * (1.0 + l * l / (r0 * r0));
    let mut y = [0.0, r0, -rd0_sq.max(0.0).sqrt(), 0.0]; // t, r, rdot, phi
    let deriv = |y: &[f64; 4]| -> [f64; 4] {
        [e / f(y[1]), y[2], rdd(y[1]), l / (y[1] * y[1])]
    };
    let steps = (tau_end / dt).ceil().max(1.0) as usize;
    let h = tau_end / steps as f64;
    let mut out = Vec::with_capacity(steps + 1);
    out.push((y[0], y[1], y[3], 0.0));
    for s in 0..steps {
        let k1 = deriv(&y);
        let mut y2 = y;
        for i in 0..4 {
            y2[i] = y[i] + 0.5 * h * k1[i];
        }
        let k2 = deriv(&y2);
        for i in 0..4 {
            y2[i] = y[i] + 0.5 * h * k2[i];
        }
        let k3 = deriv(&y2);
        for i in 0..4 {
            y2[i] = y[i] + h * k3[i];
        }
        let k4 = deriv(&y2);
        for i in 0..4 {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        if y[1] <= 2.0 * m * 1.0001 {
            break;
        }
        out.push((y[0], y[1], y[3], (s + 1) as f64 * h));
    }
    out
}

/// Photon trajectory around a Schwarzschild black hole with impact
/// parameter `b`, from the orbit equation u'' + u = 3 M u^2 starting at
/// infinity. Returns (phi, r) samples; stops at `phi_max`, escape, or
/// capture inside the photon sphere.
#[must_use]
pub fn photon_ray_trace_schwarzschild(m: f64, b: f64, phi_max: f64) -> Vec<(f64, f64)> {
    let mut u = 0.0_f64;
    let mut up = 1.0 / b;
    let rhs = |u: f64| 3.0 * m * u * u - u;
    let dphi = 1e-3;
    let mut phi = 0.0;
    let mut out = Vec::new();
    while phi < phi_max {
        if u > 0.0 {
            out.push((phi, 1.0 / u));
        }
        let k1u = up;
        let k1p = rhs(u);
        let k2u = up + 0.5 * dphi * k1p;
        let k2p = rhs(u + 0.5 * dphi * k1u);
        let k3u = up + 0.5 * dphi * k2p;
        let k3p = rhs(u + 0.5 * dphi * k2u);
        let k4u = up + dphi * k3p;
        let k4p = rhs(u + dphi * k3u);
        u += dphi / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        up += dphi / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        phi += dphi;
        if u < 0.0 || u > 1.0 / (2.0 * m) {
            break; // escaped past the start radius, or captured
        }
    }
    out
}

/// Apparent black hole shadow radius for a Kerr hole of spin `a` seen at
/// `inclination` (radians from the spin axis), averaged over the shadow
/// boundary via Bardeen's celestial coordinates; sqrt(27) M for a = 0.
#[must_use]
pub fn black_hole_shadow_radius(m: f64, a: f64, inclination: f64) -> f64 {
    if a.abs() < 1e-9 * m {
        return 27.0_f64.sqrt() * m;
    }
    let cos_i = inclination.cos();
    // photon-orbit radii between prograde and retrograde circular photon
    // orbits: r = 2M (1 + cos(2/3 acos(∓a/M)))
    let r_min = 2.0 * m * (1.0 + (2.0 / 3.0 * (-a / m).acos()).cos());
    let r_max = 2.0 * m * (1.0 + (2.0 / 3.0 * (a / m).acos()).cos());
    let n = 400;
    let mut sum = 0.0;
    let mut count = 0.0;
    for k in 0..=n {
        let r = r_min + (r_max - r_min) * k as f64 / n as f64;
        let xi = (r * r * (r - 3.0 * m) + a * a * (r + m)) / (a * (r - m));
        let eta = r * r * r * (4.0 * m * a * a - r * (r - 3.0 * m) * (r - 3.0 * m))
            / (a * a * (r - m) * (r - m));
        let beta2 = eta + a * a * cos_i * cos_i
            - xi * xi * cos_i * cos_i / (1.0 - cos_i * cos_i).max(1e-12);
        if beta2 >= 0.0 {
            sum += (xi * xi + eta + a * a * cos_i * cos_i).max(0.0).sqrt();
            count += 1.0;
        }
    }
    if count > 0.0 {
        sum / count
    } else {
        27.0_f64.sqrt() * m
    }
}

/// Carter constants of motion for a timelike Kerr geodesic: energy `e`,
/// axial angular momentum `l`, and Carter constant `q` per unit mass.
#[derive(Debug, Clone, Copy)]
pub struct KerrConstants {
    pub m: f64,
    pub a: f64,
    pub e: f64,
    pub l: f64,
    pub q: f64,
}

impl KerrConstants {
    /// Radial potential R(r): Sigma^2 (dr/dtau)^2 = R(r).
    #[must_use]
    pub fn radial_potential(&self, r: f64) -> f64 {
        let delta = r * r - 2.0 * self.m * r + self.a * self.a;
        let p = self.e * (r * r + self.a * self.a) - self.l * self.a;
        p * p
            - delta
                * (r * r + (self.l - self.a * self.e) * (self.l - self.a * self.e) + self.q)
    }

    /// Polar potential Theta(theta): Sigma^2 (dtheta/dtau)^2 = Theta.
    #[must_use]
    pub fn theta_potential(&self, theta: f64) -> f64 {
        let c2 = theta.cos() * theta.cos();
        let s2 = (1.0 - c2).max(1e-300);
        self.q - c2 * (self.a * self.a * (1.0 - self.e * self.e) + self.l * self.l / s2)
    }
}

/// Bundle the Kerr geodesic constants (energy, axial angular momentum, and
/// Carter constant per unit rest mass) with their potentials.
#[must_use]
pub fn kerr_geodesic_constants(m: f64, a: f64, e: f64, l: f64, q: f64) -> KerrConstants {
    KerrConstants { m, a, e, l, q }
}

// ---------------------------------------------------------------------------
// lensing and black hole thermodynamics
// ---------------------------------------------------------------------------

/// Einstein ring angular radius (geometric units, angles in radians):
/// theta_E = sqrt(4 M d_ls / (d_l d_s)).
#[must_use]
pub fn gravitational_lens_einstein_radius(m: f64, d_l: f64, d_s: f64, d_ls: f64) -> f64 {
    (4.0 * m * d_ls / (d_l * d_s)).sqrt()
}

/// Total point-lens magnification at impact parameter u (in Einstein
/// radii): (u^2 + 2) / (u sqrt(u^2 + 4)).
#[must_use]
pub fn point_lens_magnification(u: f64) -> f64 {
    (u * u + 2.0) / (u * (u * u + 4.0).sqrt())
}

/// Solve the lens equation beta = theta - alpha(theta) for image positions
/// given the deflection profile `mass_model` (alpha as a function of
/// theta, odd in theta). Returns all real images found on both sides.
#[must_use]
pub fn lens_equation_solve(beta: f64, mass_model: &dyn Fn(f64) -> f64) -> Vec<f64> {
    let g = |theta: f64| theta - mass_model(theta) - beta;
    let mut images = Vec::new();
    // scan for sign changes on both sides of the lens, then bisect
    let scan = |lo: f64, hi: f64, images: &mut Vec<f64>| {
        let n = 4000;
        let mut prev_t = lo;
        let mut prev_g = g(lo);
        for k in 1..=n {
            let t = lo + (hi - lo) * k as f64 / n as f64;
            let gt = g(t);
            if prev_g == 0.0 {
                images.push(prev_t);
            } else if prev_g * gt < 0.0 {
                let (mut a, mut b) = (prev_t, t);
                for _ in 0..80 {
                    let mid = 0.5 * (a + b);
                    if g(a) * g(mid) <= 0.0 {
                        b = mid;
                    } else {
                        a = mid;
                    }
                }
                images.push(0.5 * (a + b));
            }
            prev_t = t;
            prev_g = gt;
        }
    };
    let span = (beta.abs() + 1.0) * 10.0;
    scan(1e-9, span, &mut images);
    scan(-span, -1e-9, &mut images);
    images.sort_by(|a, b| a.partial_cmp(b).unwrap());
    images
}

/// Hawking temperature of a Schwarzschild black hole (SI):
/// T = hbar c^3 / (8 pi G M k_B).
#[must_use]
pub fn hawking_temperature(m: f64) -> f64 {
    HBAR * C * C * C / (8.0 * PI * G * m * K_B)
}

/// Bekenstein-Hawking entropy (SI): S = 4 pi G M^2 k_B / (hbar c).
#[must_use]
pub fn bekenstein_entropy(m: f64) -> f64 {
    4.0 * PI * G * m * m * K_B / (HBAR * C)
}

/// Black hole evaporation time (SI): t = 5120 pi G^2 M^3 / (hbar c^4).
#[must_use]
pub fn evaporation_time(m: f64) -> f64 {
    5120.0 * PI * G * G * m * m * m / (HBAR * C * C * C * C)
}

// ---------------------------------------------------------------------------
// cosmology and gravitational waves
// ---------------------------------------------------------------------------

/// Geodesic in an FRW universe with scale factor `a_fn` and curvature `k`,
/// integrated with the [`Metric`] machinery (coordinates t, r, theta,
/// phi). Returns the positions along the geodesic.
#[must_use]
pub fn frw_geodesic(
    a_fn: fn(f64) -> f64,
    k: f64,
    x0: &VecN,
    v0: &VecN,
    tau_end: f64,
    dt: f64,
) -> Vec<VecN> {
    let metric = frw_metric(a_fn, k);
    metric
        .geodesic(x0, v0, tau_end, dt, Integrator::Rk4)
        .into_iter()
        .map(|s| s.x)
        .collect()
}

/// Cosmological distances in a flat-ish FRW universe (SI: `h0` in 1/s,
/// distances in meters, lookback time in seconds): returns (comoving,
/// angular-diameter, luminosity, lookback).
#[must_use]
pub fn cosmological_distances(z: f64, h0: f64, omega_m: f64, omega_l: f64) -> (f64, f64, f64, f64) {
    let omega_k = 1.0 - omega_m - omega_l;
    let e_of_z = |z: f64| {
        (omega_m * (1.0 + z).powi(3) + omega_k * (1.0 + z) * (1.0 + z) + omega_l).sqrt()
    };
    // Simpson integration of 1/E and 1/((1+z)E)
    let n = 2000;
    let h = z / n as f64;
    let mut int_com = 0.0;
    let mut int_look = 0.0;
    for i in 0..n {
        let z0 = i as f64 * h;
        let z1 = z0 + 0.5 * h;
        let z2 = z0 + h;
        int_com += h / 6.0 * (1.0 / e_of_z(z0) + 4.0 / e_of_z(z1) + 1.0 / e_of_z(z2));
        int_look += h / 6.0
            * (1.0 / ((1.0 + z0) * e_of_z(z0))
                + 4.0 / ((1.0 + z1) * e_of_z(z1))
                + 1.0 / ((1.0 + z2) * e_of_z(z2)));
    }
    let d_h = C / h0;
    let d_c = d_h * int_com;
    let d_m = if omega_k.abs() < 1e-12 {
        d_c
    } else if omega_k > 0.0 {
        d_h / omega_k.sqrt() * (omega_k.sqrt() * d_c / d_h).sinh()
    } else {
        d_h / (-omega_k).sqrt() * ((-omega_k).sqrt() * d_c / d_h).sin()
    };
    (d_c, d_m / (1.0 + z), d_m * (1.0 + z), int_look / h0)
}

/// Chirp mass (m1 m2)^(3/5) / (m1 + m2)^(1/5).
#[must_use]
pub fn gw_chirp_mass(m1: f64, m2: f64) -> f64 {
    (m1 * m2).powf(0.6) / (m1 + m2).powf(0.2)
}

/// Leading-order (Newtonian chirp) inspiral waveform at luminosity
/// distance `d` (SI units, face-on): returns (h_plus, h_cross) sampled at
/// the times `t`, with coalescence at the last sample.
#[must_use]
pub fn gw_waveform_inspiral(m1: f64, m2: f64, d: f64, t: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mc = gw_chirp_mass(m1, m2);
    let gm = G * mc / (C * C * C); // chirp mass in seconds
    let t_c = t.last().copied().unwrap_or(0.0);
    let mut hp = Vec::with_capacity(t.len());
    let mut hx = Vec::with_capacity(t.len());
    for &ti in t {
        let theta = (t_c - ti).max(1e-6);
        // orbital phase to coalescence and strain amplitude
        let phase = -2.0 * (theta / (5.0 * gm)).powf(0.625);
        let amp = (G * mc / (C * C)) / d * (5.0 * gm / theta).powf(0.25);
        hp.push(amp * phase.cos());
        hx.push(amp * phase.sin());
    }
    (hp, hx)
}

// ---------------------------------------------------------------------------
// Kaluza-Klein
// ---------------------------------------------------------------------------

/// Five-dimensional Kaluza-Klein metric from a 4D base metric, gauge
/// potential `a` (lower index), dilaton `phi`, and compactification
/// `radius` (the fifth coordinate is an angle scaled by `radius`):
/// g_55 = (radius phi)^2, g_mu5 = radius phi^2 A_mu,
/// g_munu = g4_munu + phi^2 A_mu A_nu.
#[must_use]
pub fn kaluza_klein_metric(
    g4: Metric,
    a: impl Fn(&VecN) -> VecN + 'static,
    phi: impl Fn(&VecN) -> f64 + 'static,
    radius: f64,
) -> Metric {
    Metric::new(5, move |p: &VecN| {
        let p4 = VecN::from(&p.data[..4]);
        let g = (g4.g)(&p4);
        let av = a(&p4);
        let ph2 = phi(&p4) * phi(&p4);
        crate::linalg::Matrix::from_fn(5, 5, |i, j| match (i, j) {
            (4, 4) => ph2 * radius * radius,
            (4, mu) => ph2 * av[mu] * radius,
            (mu, 4) => ph2 * av[mu] * radius,
            (mu, nu) => g.get(mu, nu) + ph2 * av[mu] * av[nu],
        })
    })
}

/// Reduce a 5D Kaluza-Klein geodesic to 4D charged-particle data: the
/// conserved fifth momentum gives the charge-to-mass ratio (valid when
/// the gauge potential vanishes at the initial point and phi = 1), and
/// the positions project to the 4D worldline.
#[must_use]
pub fn kk_reduce_geodesic_to_charged(geo5: &[GeodesicState], radius: f64) -> (f64, Vec<VecN>) {
    let q_over_m = geo5
        .first()
        .map(|s| s.v[4] * radius * radius)
        .unwrap_or(0.0);
    let path = geo5
        .iter()
        .map(|s| VecN::from(&s.x.data[..4]))
        .collect();
    (q_over_m, path)
}

/// Kaluza-Klein tower masses n / R for mode numbers 0..=n_max.
#[must_use]
pub fn kk_compactification_mass_spectrum(radius: f64, n_max: usize) -> Vec<f64> {
    (0..=n_max).map(|n| n as f64 / radius).collect()
}

/// Gravitational force law with `n_extra` compact extra dimensions of size
/// `size` (normalized to 1/r^2 at large r): 1/r^2 outside, continuously
/// matched to size^n / r^(2+n) inside.
#[must_use]
pub fn extra_dimension_gravity_law(r: f64, n_extra: usize, size: f64) -> f64 {
    if r >= size {
        1.0 / (r * r)
    } else {
        size.powi(n_extra as i32) / r.powi(2 + n_extra as i32)
    }
}

/// Cross-validate the spacetime algebra boost rotor against the matrix
/// Lorentz boost: the maximum component difference over a set of basis
/// events (the STA rotor R e R~ realizes the inverse boost, so it is
/// compared against B(-v)).
#[must_use]
pub fn sta_vs_matrix_lorentz_check(v: Vec3) -> f64 {
    use crate::manifold::clifford::sta;
    let rotor = sta::boost(v);
    let matrix = LorentzTransform::boost(v * -1.0);
    let events = [
        (1.0, Vec3::new(0.0, 0.0, 0.0)),
        (0.0, Vec3::new(1.0, 0.0, 0.0)),
        (0.0, Vec3::new(0.0, 1.0, 0.0)),
        (0.0, Vec3::new(0.0, 0.0, 1.0)),
        (1.0, Vec3::new(0.3, -0.4, 0.5)),
    ];
    let mut worst = 0.0_f64;
    for &(t, x) in &events {
        let e = sta::event(t, x);
        let mapped = sta::lorentz_apply(&rotor, &e);
        let mt = mapped.coeffs[0b0001];
        let mx = Vec3::new(
            mapped.coeffs[0b0010],
            mapped.coeffs[0b0100],
            mapped.coeffs[0b1000],
        );
        let want = matrix.apply(FourVector::new(t, x));
        worst = worst
            .max((mt - want.t).abs())
            .max((mx - want.x).magnitude());
    }
    worst
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::general_relativity as gr;
    use crate::monte_carlo::Rng;

    #[test]
    fn test_fourvector_and_boosts() {
        let mut rng = Rng::new(11);
        for _ in 0..20 {
            let v = Vec3::new(
                0.8 * (rng.next_f64() - 0.5),
                0.8 * (rng.next_f64() - 0.5),
                0.8 * (rng.next_f64() - 0.5),
            );
            let p = FourVector::new(
                2.0 * rng.next_f64() + 1.5,
                Vec3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()),
            );
            // boost preserves the Minkowski norm
            let boosted = p.boost(v);
            assert!(
                (boosted.norm_squared() - p.norm_squared()).abs() < 1e-10,
                "norm not preserved"
            );
            assert!(LorentzTransform::boost(v).is_lorentz(1e-10));
            // velocity addition never exceeds c
            let u = Vec3::new(
                0.9 * (rng.next_f64() - 0.5),
                0.9 * (rng.next_f64() - 0.5),
                0.0,
            );
            let w = LorentzTransform::velocity_addition(u, v);
            assert!(w.magnitude() < 1.0, "superluminal addition {}", w.magnitude());
        }
        // four-velocity is unit and its rapidity matches atanh
        let v = Vec3::new(0.6, 0.0, 0.0);
        let u4 = FourVector::from_velocity(v);
        assert!((u4.norm_squared() - 1.0).abs() < 1e-12);
        assert!((u4.rapidity() - 0.6_f64.atanh()).abs() < 1e-12);
        // collinear addition matches the scalar formula
        let w = LorentzTransform::velocity_addition(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
        );
        assert!((w.x - (0.5 + 0.6) / (1.0 + 0.3)).abs() < 1e-12);
        // energy-momentum of a moving mass
        let p = FourVector::from_momentum(2.0, v);
        assert!((p.energy() - 2.0 / 0.8).abs() < 1e-12);
        // causal classification
        let o = FourVector::new(0.0, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(
            light_cone_check(o, FourVector::new(2.0, Vec3::new(1.0, 0.0, 0.0))),
            Causal::Future
        );
        assert_eq!(
            light_cone_check(o, FourVector::new(-2.0, Vec3::new(1.0, 0.0, 0.0))),
            Causal::Past
        );
        assert_eq!(
            light_cone_check(o, FourVector::new(0.5, Vec3::new(1.0, 0.0, 0.0))),
            Causal::Spacelike
        );
        assert_eq!(
            light_cone_check(o, FourVector::new(1.0, Vec3::new(1.0, 0.0, 0.0))),
            Causal::Null
        );
        // twin paradox and rocket
        let (home, traveler) = twin_paradox_ages(0.8, 10.0);
        assert!((home - 10.0).abs() < 1e-12 && (traveler - 6.0).abs() < 1e-12);
        let (t, x, vr) = relativistic_rocket(1.0, 2.0);
        assert!((t - 2.0_f64.sinh()).abs() < 1e-12);
        assert!((x - (2.0_f64.cosh() - 1.0)).abs() < 1e-12);
        assert!((vr - 2.0_f64.tanh()).abs() < 1e-12);
        // simultaneity plane contains the event and is u-orthogonal
        let ev = FourVector::new(1.0, Vec3::new(2.0, 0.0, 0.0));
        let pl = simultaneity_plane(Vec3::new(0.5, 0.0, 0.0), ev);
        assert!((pl.normal.minkowski_dot(&ev) - pl.offset).abs() < 1e-12);
    }

    #[test]
    fn test_thomas_wigner_and_sl2c() {
        // perpendicular boosts: rotation angle matches the closed formula
        let v1 = Vec3::new(0.5, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 0.4, 0.0);
        let r = LorentzTransform::thomas_wigner_rotation(v1, v2);
        let angle = r.log().magnitude();
        let g1 = 1.0 / (1.0 - v1.magnitude_squared()).sqrt();
        let g2 = 1.0 / (1.0 - v2.magnitude_squared()).sqrt();
        let g12 = g1 * g2 * (1.0 + v1.dot(&v2));
        let cosw = (1.0 + g1 + g2 + g12) * (1.0 + g1 + g2 + g12)
            / ((1.0 + g1) * (1.0 + g2) * (1.0 + g12))
            - 1.0;
        assert!(
            (angle.cos() - cosw).abs() < 1e-9,
            "wigner angle {} vs formula {}",
            angle.cos(),
            cosw
        );
        // parallel boosts commute: no rotation
        let r0 = LorentzTransform::thomas_wigner_rotation(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
        );
        assert!(r0.log().magnitude() < 1e-9);
        // SL(2, C) roundtrip: boost * rotation
        let rot = So3::exp(Vec3::new(0.2, -0.3, 0.4));
        let l = LorentzTransform::boost(Vec3::new(0.3, 0.2, -0.1))
            .compose(&LorentzTransform::rotation(&rot));
        let back = LorentzTransform::from_sl2c(&l.to_sl2c());
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (back.0.data[i][j] - l.0.data[i][j]).abs() < 1e-9,
                    "sl2c roundtrip at {i},{j}: {} vs {}",
                    back.0.data[i][j],
                    l.0.data[i][j]
                );
            }
        }
        // STA rotor agrees with the matrix boost
        assert!(sta_vs_matrix_lorentz_check(Vec3::new(0.4, -0.2, 0.3)) < 1e-9);
        assert!(sta_vs_matrix_lorentz_check(Vec3::new(0.0, 0.0, 0.7)) < 1e-9);
    }

    #[test]
    fn test_kruskal_rindler_thermo() {
        let m = 1.0;
        // Kruskal identity T^2 - X^2 = (1 - r/2M) e^(r/2M) on both sides of
        // the horizon, and continuity across it
        for &r in &[1.7, 1.9, 1.99, 2.01, 2.1, 3.0] {
            let (t_k, x_k) = kruskal_from_schwarzschild(0.7, r, m);
            let want = (1.0 - r / (2.0 * m)) * (r / (2.0 * m)).exp();
            assert!(
                (t_k * t_k - x_k * x_k - want).abs() < 1e-10,
                "kruskal identity at r={r}"
            );
        }
        // smooth through r = 2M: along an ingoing null ray (fixed
        // Eddington-Finkelstein v), the Kruskal null coordinate T + X
        // equals e^(v/4M) on both sides of the horizon
        let v_ef = 3.0;
        for k in 0..200 {
            let r = 1.8 + 0.4 * k as f64 / 199.0;
            if (r - 2.0 * m).abs() < 1e-3 {
                continue;
            }
            let r_star = eddington_finkelstein(0.0, r, m); // v(t=0) = r*
            let t = v_ef - r_star;
            let (t_k, x_k) = kruskal_from_schwarzschild(t, r, m);
            let want = (v_ef / (4.0 * m)).exp();
            assert!(
                (t_k + x_k - want).abs() < 1e-8 * want,
                "Kruskal V not smooth at r={r}: {} vs {want}",
                t_k + x_k
            );
        }
        // penrose coords are bounded
        let (pt, px) = penrose_diagram_coords(0.5, 3.0, m);
        assert!(pt.abs() < PI / 2.0 && px.abs() < PI / 2.0);
        // Rindler roundtrip: t = xi sinh(a eta), x = xi cosh(a eta)
        let (t0, x0, a) = (0.3, 1.7, 2.0);
        let (eta, xi) = rindler_coords(t0, x0, a);
        assert!((xi * (a * eta).sinh() - t0).abs() < 1e-12);
        assert!((xi * (a * eta).cosh() - x0).abs() < 1e-12);
        assert!((rindler_horizon(a) - 0.5).abs() < 1e-15);
        // Eddington-Finkelstein v is finite and increasing in t
        assert!(eddington_finkelstein(1.0, 3.0, m) > eddington_finkelstein(0.0, 3.0, m));
        // thermodynamics for a solar-mass hole
        let msun = 1.989e30;
        let t_h = hawking_temperature(msun);
        assert!((t_h - 6.17e-8).abs() / 6.17e-8 < 0.02, "hawking {t_h}");
        // entropy ~ 1.05e54 J/K per solar mass squared
        let s = bekenstein_entropy(msun);
        assert!(s > 1e53 && s < 1e55, "entropy {s}");
        // evaporation ~ 6.6e74 s * (M/Msun)^3
        let tev = evaporation_time(msun);
        assert!((tev - 6.6e74).abs() / 6.6e74 < 0.1, "evaporation {tev}");
        // Unruh: ~4e-20 K at g = 9.81
        let tu = unruh_temperature(9.81);
        assert!((tu - 3.98e-20).abs() / 3.98e-20 < 0.02, "unruh {tu}");
    }

    #[test]
    fn test_schwarzschild_orbit_matches_geodesic_module() {
        let m = 1.0;
        let r0 = 20.0;
        let l = 4.5;
        // energy of a turning point at r0 (dr = 0 there)
        let f: f64 = 1.0 - 2.0 * m / r0;
        let e = (f * (1.0 + l * l / (r0 * r0))).sqrt();
        let full = orbit_schwarzschild_full(m, e, l, r0, 600.0, 0.01);
        let reference = crate::manifold::geodesic::schwarzschild_orbit(
            m,
            r0,
            l,
            e,
            2.0 * PI,
            1e-3,
        );
        // compare r(phi) between the two integrations
        for &(_, r, phi, _) in &full {
            if phi > 2.0 * PI - 0.05 {
                break;
            }
            let idx = (phi / 1e-3).round() as usize;
            if idx < reference.len() {
                let r_ref = reference[idx].1;
                assert!(
                    (r - r_ref).abs() / r_ref < 2e-3,
                    "orbit mismatch at phi={phi}: {r} vs {r_ref}"
                );
            }
        }
        // proper time runs slower than coordinate time
        let last = full.last().unwrap();
        assert!(last.0 > last.3, "coordinate time should exceed proper time");
        // photon deflection at large impact parameter matches 4M/b
        let b = 1000.0;
        let trace = photon_ray_trace_schwarzschild(m, b, PI + 0.1);
        // extrapolate the u = 0 exit crossing: near exit du/dphi = -1/b
        let (phi_last, r_last) = *trace.last().unwrap();
        let deflection = phi_last + b / r_last - PI;
        let expect = crate::manifold::geodesic::light_deflection(m, b);
        assert!(
            (deflection - expect).abs() / expect < 0.05,
            "deflection {deflection} vs {expect}"
        );
        // shadow of a Schwarzschild hole
        assert!((black_hole_shadow_radius(m, 0.0, 0.5) - 27.0_f64.sqrt()).abs() < 1e-12);
        // Kerr shadow shrinks a bit prograde but stays order sqrt(27) M
        let rs = black_hole_shadow_radius(m, 0.7, PI / 2.0);
        assert!(rs > 4.0 && rs < 6.0, "kerr shadow {rs}");
        // Kerr constants: Schwarzschild circular orbit sits at a double
        // root of the radial potential
        let rc = 8.0;
        let ec = (1.0 - 2.0 * m / rc) / (1.0 - 3.0 * m / rc).sqrt();
        let lc = rc * (m / (rc - 3.0 * m)).sqrt();
        // the same formulas in SI via general_relativity (M chosen so
        // GM/c^2 = 1 meter, r in meters)
        let m_si = C * C / G;
        assert!((gr::circular_orbit_energy(m_si, rc) - ec).abs() < 1e-9);
        assert!((gr::circular_orbit_angular_momentum(m_si, rc) - lc).abs() < 1e-9);
        let kc = kerr_geodesic_constants(m, 0.0, ec, lc, 0.0);
        let scale = rc.powi(4);
        assert!(kc.radial_potential(rc).abs() / scale < 1e-10, "R(rc) != 0");
        let dr = 1e-5;
        let slope =
            (kc.radial_potential(rc + dr) - kc.radial_potential(rc - dr)) / (2.0 * dr);
        assert!(slope.abs() / scale < 1e-4, "R'(rc) != 0: {slope}");
        assert!(kc.theta_potential(PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_lensing_cosmology_gw() {
        // Einstein radius formula and cross-check against the deflection
        // angle: theta_E^2 d_l d_s / d_ls = 4M
        let (m, d_l, d_s, d_ls) = (1.0, 1e9, 2e9, 1e9);
        let te = gravitational_lens_einstein_radius(m, d_l, d_s, d_ls);
        assert!((te * te * d_l * d_s / d_ls - 4.0 * m).abs() < 1e-12);
        // magnification: large u -> 1, u = 1 -> 3/sqrt(5)
        assert!((point_lens_magnification(100.0) - 1.0).abs() < 1e-3);
        assert!((point_lens_magnification(1.0) - 3.0 / 5.0_f64.sqrt()).abs() < 1e-12);
        // point lens images: theta^2 - beta theta - theta_E^2 = 0
        let theta_e = 1.0;
        let beta = 0.5;
        let images = lens_equation_solve(beta, &|th: f64| theta_e * theta_e / th);
        assert_eq!(images.len(), 2, "point lens must give 2 images");
        let want_plus = 0.5 * (beta + (beta * beta + 4.0 * theta_e * theta_e).sqrt());
        let want_minus = 0.5 * (beta - (beta * beta + 4.0 * theta_e * theta_e).sqrt());
        assert!((images[1] - want_plus).abs() < 1e-6);
        assert!((images[0] - want_minus).abs() < 1e-6);
        // cosmological distances agree with the low-z expansions in
        // general_relativity
        let h0 = 2.2e-18; // ~ 68 km/s/Mpc in 1/s
        let z = 0.01;
        let (d_c, d_a, d_lum, look) = cosmological_distances(z, h0, 0.3, 0.7);
        assert!((d_c - gr::cosmological_redshift_distance(z, h0)).abs() / d_c < 0.01);
        assert!((d_lum - gr::luminosity_distance(z, h0)).abs() / d_lum < 0.01);
        assert!((look - gr::lookback_time(z, h0)).abs() / look < 0.01);
        assert!((d_a - d_lum / (1.0 + z) / (1.0 + z)).abs() / d_a < 1e-9);
        // chirp mass matches astrophysics
        let (m1, m2) = (2.8e31, 1.6e31);
        assert!(
            (gw_chirp_mass(m1, m2) - crate::astrophysics::gravitational_waves::chirp_mass(m1, m2))
                .abs()
                / gw_chirp_mass(m1, m2)
                < 1e-12
        );
        // inspiral: amplitude grows toward coalescence, phase accelerates
        let t: Vec<f64> = (0..4000).map(|i| i as f64 * 1e-3).collect();
        let (hp, hx) = gw_waveform_inspiral(m1, m2, 3.1e24, &t);
        let amp_early = (hp[10] * hp[10] + hx[10] * hx[10]).sqrt();
        let amp_late = (hp[3800] * hp[3800] + hx[3800] * hx[3800]).sqrt();
        assert!(amp_late > amp_early * 1.5, "no chirp: {amp_early} {amp_late}");
        assert!(amp_early > 0.0 && hp.len() == 4000);
        // frequency increases: count zero crossings in early vs late windows
        let crossings = |s: &[f64]| s.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(crossings(&hp[3000..3900]) > crossings(&hp[0..900]));
    }

    #[test]
    fn test_kaluza_klein_lorentz_force() {
        // flat 4D metric, uniform electric field E_x from A_t = -E x
        let e_field = 0.01;
        let g4 = Metric::minkowski(4, Sig::MostlyPlus);
        let a_fn = move |p: &VecN| {
            let mut a = VecN::zeros(4);
            a.data[0] = -e_field * p[1];
            a
        };
        let g5 = kaluza_klein_metric(g4, a_fn, |_| 1.0, 1.0);
        // start at rest with fifth-velocity = charge/mass
        let q_over_m = 0.5;
        let x0 = VecN::from(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let v0 = VecN::from(&[1.0, 0.0, 0.0, 0.0, q_over_m]);
        let tau_end = 2.0;
        let geo = g5.geodesic(&x0, &v0, tau_end, 0.01, Integrator::Rk4);
        let (q_rec, path) = kk_reduce_geodesic_to_charged(&geo, 1.0);
        assert!((q_rec - q_over_m).abs() < 1e-9, "recovered charge {q_rec}");
        // reference: charged particle in flat spacetime, du/dtau = (q/m) F u
        // with F_tx = E: du_t = (q/m) E u_x... using signature (-,+,+,+):
        // u_t'' analog - integrate the same first-order system
        let mut u = [1.0, 0.0]; // (u^t, u^x)
        let mut pos = [0.0, 0.0]; // (t, x)
        let n = 2000;
        let h = tau_end / n as f64;
        let mut worst = 0.0_f64;
        let mut checked = false;
        for s in 0..=n {
            let tau = s as f64 * h;
            // compare against the KK geodesic sample at the same tau
            let idx = (tau / 0.01).round() as usize;
            if idx < path.len() && (idx as f64 * 0.01 - tau).abs() < 1e-9 {
                worst = worst
                    .max((path[idx][0] - pos[0]).abs())
                    .max((path[idx][1] - pos[1]).abs());
                checked = true;
            }
            // RK4 on (u_t, u_x, t, x): du^t = -qE u^x, du^x = -qE u^t
            // (the sign follows the potential A_t = -E x)
            let f = |u: &[f64; 2]| [-q_over_m * e_field * u[1], -q_over_m * e_field * u[0]];
            let k1 = f(&u);
            let k2 = f(&[u[0] + 0.5 * h * k1[0], u[1] + 0.5 * h * k1[1]]);
            let k3 = f(&[u[0] + 0.5 * h * k2[0], u[1] + 0.5 * h * k2[1]]);
            let k4 = f(&[u[0] + h * k3[0], u[1] + h * k3[1]]);
            let du = [
                h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            ];
            pos[0] += h * u[0] + 0.5 * h * du[0];
            pos[1] += h * u[1] + 0.5 * h * du[1];
            u[0] += du[0];
            u[1] += du[1];
        }
        assert!(checked, "no comparison samples");
        assert!(worst < 5e-3, "KK vs Lorentz force deviation {worst}");
        // mass tower and modified gravity law
        let tower = kk_compactification_mass_spectrum(2.0, 3);
        assert_eq!(tower.len(), 4);
        assert!((tower[3] - 1.5).abs() < 1e-15);
        let inside = extra_dimension_gravity_law(0.5, 2, 1.0);
        let outside = extra_dimension_gravity_law(2.0, 2, 1.0);
        assert!((outside - 0.25).abs() < 1e-15);
        assert!((inside - 1.0 / 0.5_f64.powi(4)).abs() < 1e-12);
        // continuity at r = size
        let a = extra_dimension_gravity_law(1.0 - 1e-9, 2, 1.0);
        let b = extra_dimension_gravity_law(1.0 + 1e-9, 2, 1.0);
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn test_lorentz_identity_and_boost_x() {
        // identity acts trivially and is a Lorentz transform
        let id = LorentzTransform::identity();
        assert!(id.is_lorentz(1e-15));
        let mut rng = Rng::new(4242);
        for _ in 0..5 {
            let v = FourVector::new(
                rng.next_gaussian(),
                Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian()),
            );
            let out = id.apply(v);
            assert!((out.t - v.t).abs() < 1e-15 && (out.x - v.x).magnitude() < 1e-15);
        }
        // boost_x(beta) is the boost along +x, with the closed-form matrix
        let beta = 0.6_f64;
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let bx = LorentzTransform::boost_x(beta);
        let bv = LorentzTransform::boost(Vec3::new(beta, 0.0, 0.0));
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (bx.0.data[i][j] - bv.0.data[i][j]).abs() < 1e-15,
                    "boost_x vs boost at {i},{j}"
                );
            }
        }
        let want = [
            [gamma, gamma * beta, 0.0, 0.0],
            [gamma * beta, gamma, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (bx.0.data[i][j] - want[i][j]).abs() < 1e-14,
                    "boost_x matrix at {i},{j}: {}",
                    bx.0.data[i][j]
                );
            }
        }
        assert!(bx.is_lorentz(1e-12));
        // a particle at rest is taken to three-velocity beta
        let moved = bx.apply(FourVector::new(1.0, Vec3::new(0.0, 0.0, 0.0)));
        assert!((moved.t - gamma).abs() < 1e-14);
        assert!((moved.x - Vec3::new(gamma * beta, 0.0, 0.0)).magnitude() < 1e-14);
        // zero speed degenerates to the identity, and inverse boosts cancel
        let zero = LorentzTransform::boost_x(0.0);
        for i in 0..4 {
            for j in 0..4 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((zero.0.data[i][j] - e).abs() < 1e-15);
            }
        }
        let round = LorentzTransform::boost_x(-beta).compose(&bx);
        for i in 0..4 {
            for j in 0..4 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((round.0.data[i][j] - e).abs() < 1e-12, "B(-b) B(b) = I");
            }
        }
        // collinear boosts add rapidities
        let b1 = LorentzTransform::boost_x(0.3);
        let b2 = LorentzTransform::boost_x(0.5);
        let comp = b1.compose(&b2);
        let want_beta = (0.3_f64.atanh() + 0.5_f64.atanh()).tanh();
        let img = comp.apply(FourVector::new(1.0, Vec3::new(0.0, 0.0, 0.0)));
        assert!(
            (img.x.x / img.t - want_beta).abs() < 1e-12,
            "rapidity addition {} vs {want_beta}",
            img.x.x / img.t
        );
    }

    #[test]
    fn test_fourvector_causal_predicates_and_ops() {
        let timelike = FourVector::new(2.0, Vec3::new(1.0, 0.0, 0.0));
        let spacelike = FourVector::new(0.5, Vec3::new(1.0, 0.0, 0.0));
        let null = FourVector::new(1.0, Vec3::new(0.6, 0.8, 0.0));
        // the three predicates partition the sample consistently with the
        // sign of the invariant norm
        for v in [timelike, spacelike, null] {
            let s2 = v.norm_squared();
            assert_eq!(v.is_timelike(), s2 > 0.0);
            assert_eq!(v.is_spacelike(), s2 < 0.0);
            assert!(!(v.is_timelike() && v.is_spacelike()));
        }
        assert!(timelike.is_timelike() && !timelike.is_null(1e-12));
        assert!(spacelike.is_spacelike() && !spacelike.is_null(1e-12));
        assert!(null.is_null(1e-12) && !null.is_timelike() && !null.is_spacelike());
        assert!((null.norm_squared()).abs() < 1e-15);
        // the classification is boost-invariant
        let v = Vec3::new(0.4, -0.2, 0.1);
        assert!(timelike.boost(v).is_timelike());
        assert!(spacelike.boost(v).is_spacelike());
        assert!(null.boost(v).is_null(1e-12));
        // four-velocities are timelike and unit; photon momenta are null
        assert!(FourVector::from_velocity(Vec3::new(0.9, 0.0, 0.0)).is_timelike());
        // spatial_momentum is the vector part, and E^2 - |p|^2 = m^2
        let p = FourVector::from_momentum(3.0, Vec3::new(0.5, -0.25, 0.0));
        assert!((p.spatial_momentum() - p.x).magnitude() < 1e-15);
        assert!(
            (p.energy().powi(2) - p.spatial_momentum().magnitude_squared() - 9.0).abs() < 1e-12,
            "mass shell"
        );
        // Add is componentwise and interacts linearly with the Minkowski dot
        let a = FourVector::new(1.5, Vec3::new(0.3, -0.7, 2.0));
        let b = FourVector::new(-0.5, Vec3::new(1.0, 0.25, -0.5));
        let sum = a + b;
        assert!((sum.t - 1.0).abs() < 1e-15);
        assert!((sum.x - Vec3::new(1.3, -0.45, 1.5)).magnitude() < 1e-15);
        assert!(((sum - b).t - a.t).abs() < 1e-15 && ((sum - b).x - a.x).magnitude() < 1e-15);
        let c = FourVector::new(0.2, Vec3::new(0.1, 0.2, 0.3));
        assert!(
            ((a + b).minkowski_dot(&c) - a.minkowski_dot(&c) - b.minkowski_dot(&c)).abs() < 1e-14,
            "Minkowski dot is additive"
        );
        // boosting is linear over addition too
        let boosted = (a + b).boost(v);
        let sep = a.boost(v) + b.boost(v);
        assert!((boosted.t - sep.t).abs() < 1e-12 && (boosted.x - sep.x).magnitude() < 1e-12);
        // minkowski_dot_sig: MostlyPlus is the negative of MostlyMinus
        let dm = a.minkowski_dot_sig(&b, Sig::MostlyMinus);
        let dp = a.minkowski_dot_sig(&b, Sig::MostlyPlus);
        assert!((dm - a.minkowski_dot(&b)).abs() < 1e-15);
        assert!((dm + dp).abs() < 1e-15, "signature flip");
        assert!(
            (dp - (-a.t * b.t + a.x.dot(&b.x))).abs() < 1e-15,
            "mostly-plus form"
        );
        // proper_time_to is the square root of the timelike interval and is
        // boost-invariant; spacelike separations give zero
        let o = FourVector::new(0.0, Vec3::new(0.0, 0.0, 0.0));
        let ev = FourVector::new(2.0, Vec3::new(1.2, 0.0, 0.0));
        let tau = o.proper_time_to(&ev);
        assert!((tau - (4.0_f64 - 1.44).sqrt()).abs() < 1e-14, "proper time {tau}");
        assert!((tau * tau - (ev - o).norm_squared()).abs() < 1e-13);
        assert!(
            (o.boost(v).proper_time_to(&ev.boost(v)) - tau).abs() < 1e-12,
            "proper time invariant"
        );
        let spacelike_tau = o.proper_time_to(&FourVector::new(0.1, Vec3::new(1.0, 0.0, 0.0)));
        assert!(spacelike_tau.abs() < 1e-300, "spacelike proper time clamped");
        // a clock moving at speed 0.6 for coordinate time 1 ages 0.8
        let worldline = FourVector::new(1.0, Vec3::new(0.6, 0.0, 0.0));
        assert!((o.proper_time_to(&worldline) - 0.8).abs() < 1e-14);
        // rotate preserves t and |x| and agrees with the So3 action
        let r = So3::exp(Vec3::new(0.2, -0.5, 0.7));
        let rot = a.rotate(&r);
        assert!((rot.t - a.t).abs() < 1e-15, "time untouched");
        assert!(
            (rot.x.magnitude() - a.x.magnitude()).abs() < 1e-13,
            "spatial length preserved"
        );
        assert!((rot.norm_squared() - a.norm_squared()).abs() < 1e-13);
        let via_matrix = LorentzTransform::rotation(&r).apply(a);
        assert!((rot.t - via_matrix.t).abs() < 1e-15);
        assert!((rot.x - via_matrix.x).magnitude() < 1e-13, "rotate vs matrix");
    }

    #[test]
    fn test_kaluza_klein_and_schwarzschild_metrics() {
        // constant dilaton with no gauge field: g55 = (radius phi)^2 and the
        // 4D block is untouched
        let radius = 2.5;
        let phi_const = 0.75;
        let g5 = kaluza_klein_metric(
            Metric::minkowski(4, Sig::MostlyPlus),
            |_: &VecN| VecN::zeros(4),
            move |_: &VecN| phi_const,
            radius,
        );
        let p = VecN::from(&[0.3, 1.0, -0.5, 2.0, 0.1]);
        let g = (g5.g)(&p);
        assert!(
            (g.get(4, 4) - (radius * phi_const).powi(2)).abs() < 1e-14,
            "g55 = (R phi)^2: {}",
            g.get(4, 4)
        );
        for mu in 0..4 {
            assert!(g.get(4, mu).abs() < 1e-15 && g.get(mu, 4).abs() < 1e-15, "A = 0");
            let want = if mu == 0 { -1.0 } else { 1.0 };
            assert!((g.get(mu, mu) - want).abs() < 1e-15, "4D block preserved");
        }
        // a nontrivial dilaton and gauge potential: check every block
        // against the closed form, and symmetry of the 5D metric
        let e_field = 0.3;
        let a_fn = move |q: &VecN| {
            let mut a = VecN::zeros(4);
            a.data[0] = -e_field * q[1];
            a.data[2] = 0.2 * q[3];
            a
        };
        let phi_fn = |q: &VecN| 1.0 + 0.1 * q[1] * q[1];
        let g5b = kaluza_klein_metric(
            Metric::minkowski(4, Sig::MostlyPlus),
            a_fn,
            phi_fn,
            radius,
        );
        let gb = (g5b.g)(&p);
        let p4 = VecN::from(&p.data[..4]);
        let phi2 = phi_fn(&p4) * phi_fn(&p4);
        let av = {
            let mut a = VecN::zeros(4);
            a.data[0] = -e_field * p4[1];
            a.data[2] = 0.2 * p4[3];
            a
        };
        assert!((gb.get(4, 4) - phi2 * radius * radius).abs() < 1e-14);
        for mu in 0..4 {
            assert!(
                (gb.get(4, mu) - phi2 * av[mu] * radius).abs() < 1e-14,
                "g_5mu at {mu}"
            );
            assert!((gb.get(4, mu) - gb.get(mu, 4)).abs() < 1e-15, "symmetry");
            for nu in 0..4 {
                let flat = if mu == nu {
                    if mu == 0 { -1.0 } else { 1.0 }
                } else {
                    0.0
                };
                assert!(
                    (gb.get(mu, nu) - (flat + phi2 * av[mu] * av[nu])).abs() < 1e-14,
                    "g_munu at {mu},{nu}"
                );
                assert!((gb.get(mu, nu) - gb.get(nu, mu)).abs() < 1e-15);
            }
        }
        // Schwarzschild as a Metric: exact components and the flat limit
        let m = 1.0;
        let met = schwarzschild_geodesic_metric(m);
        assert_eq!(met.dim, 4);
        for &r in &[3.0_f64, 10.0, 1e6] {
            let theta = PI / 3.0;
            let gm = (met.g)(&VecN::from(&[0.0, r, theta, 0.0]));
            let f = 1.0 - 2.0 * m / r;
            assert!((gm.get(0, 0) + f).abs() < 1e-12, "g_tt at r={r}");
            assert!((gm.get(1, 1) - 1.0 / f).abs() < 1e-12, "g_rr at r={r}");
            assert!((gm.get(2, 2) - r * r).abs() < 1e-9 * r * r, "g_thth");
            assert!(
                (gm.get(3, 3) - r * r * theta.sin().powi(2)).abs() < 1e-9 * r * r,
                "g_phph"
            );
            assert!(
                (gm.get(0, 0) * gm.get(1, 1) + 1.0).abs() < 1e-12,
                "g_tt g_rr = -1"
            );
            for i in 0..4 {
                for j in 0..4 {
                    if i != j {
                        assert!(gm.get(i, j).abs() < 1e-15, "diagonal metric");
                    }
                }
            }
        }
        // far field is Minkowski (-,+) in the t, r block
        let far = (met.g)(&VecN::from(&[0.0, 1e9, PI / 2.0, 0.0]));
        assert!((far.get(0, 0) + 1.0).abs() < 1e-8, "g_tt -> -1");
        assert!((far.get(1, 1) - 1.0).abs() < 1e-8, "g_rr -> 1");
        // and the horizon is where g_tt vanishes
        let hor = (met.g)(&VecN::from(&[0.0, 2.0 * m, PI / 2.0, 0.0]));
        assert!(hor.get(0, 0).abs() < 1e-15, "g_tt(2M) = 0");
    }

    #[test]
    fn test_frw_geodesic() {
        // static universe (a = 1): radial geodesics are straight lines
        fn a_one(_t: f64) -> f64 {
            1.0
        }
        let x0 = VecN::from(&[0.0, 1.0, PI / 2.0, 0.0]);
        let v0 = VecN::from(&[1.2, 0.5, 0.0, 0.0]);
        let path = frw_geodesic(a_one, 0.0, &x0, &v0, 1.0, 0.01);
        let end = path.last().unwrap();
        assert!((end[0] - 1.2).abs() < 1e-6, "t drift {}", end[0]);
        assert!((end[1] - 1.5).abs() < 1e-6, "r drift {}", end[1]);
    }
}
