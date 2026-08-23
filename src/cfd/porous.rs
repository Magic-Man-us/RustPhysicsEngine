//! Porous-media flow: Darcy's law and extensions, unsaturated flow
//! (Richards equation with Van Genuchten retention), well hydraulics,
//! solute transport, and two-phase relations (Leverett, Corey,
//! Buckley-Leverett).

use crate::cfd::grid::CellField2;

const G: f64 = 9.806_65;

// ---------------------------------------------------------------------------
// Darcy and friends
// ---------------------------------------------------------------------------

/// Darcy velocity (specific discharge) q = -(k/mu) grad p; returns the
/// magnitude for a pressure gradient `grad_p` (Pa/m).
#[must_use]
pub fn darcy_velocity(k: f64, mu: f64, grad_p: f64) -> f64 {
    -k / mu * grad_p
}

/// Volumetric flow rate through area `a` over length `l` under pressure
/// difference `dp`: Q = k A dp / (mu L).
#[must_use]
pub fn darcy_flow_rate(k: f64, a: f64, mu: f64, dp: f64, l: f64) -> f64 {
    k * a * dp / (mu * l)
}

/// Kozeny-Carman permeability of a packed bed of spheres of diameter
/// `d_particle` (Ergun-consistent constant 150):
/// k = phi^3 d^2 / (150 (1-phi)^2).
#[must_use]
pub fn permeability_kozeny_carman(porosity: f64, d_particle: f64) -> f64 {
    porosity.powi(3) * d_particle * d_particle / (150.0 * (1.0 - porosity).powi(2))
}

/// Approximate Kozeny-type permeability of a random fiber mat with fiber
/// diameter `d_fiber` (Kozeny constant ~ 5 with the fiber specific surface
/// 4/d): k = phi^3 d^2 / (80 (1-phi)^2).
#[must_use]
pub fn carman_kozeny_fibers(porosity: f64, d_fiber: f64) -> f64 {
    porosity.powi(3) * d_fiber * d_fiber / (80.0 * (1.0 - porosity).powi(2))
}

/// Ergun pressure drop over bed length `l` for superficial velocity `u`:
/// dP/L = 150 mu u (1-phi)^2/(phi^3 d^2) + 1.75 rho u^2 (1-phi)/(phi^3 d).
#[must_use]
pub fn ergun_pressure_drop(u: f64, d: f64, porosity: f64, mu: f64, rho: f64, l: f64) -> f64 {
    let e = porosity;
    let visc = 150.0 * mu * u * (1.0 - e).powi(2) / (e.powi(3) * d * d);
    let inert = 1.75 * rho * u * u * (1.0 - e) / (e.powi(3) * d);
    (visc + inert) * l
}

/// Forchheimer pressure gradient magnitude: dp/dx = mu u / k + beta rho u^2.
#[must_use]
pub fn forchheimer(k: f64, beta: f64, mu: f64, rho: f64, u: f64) -> f64 {
    mu * u / k + beta * rho * u * u
}

/// Brinkman flow in a porous channel of height `h` driven by `dp_dx`:
/// u(y) = -(k/mu) dp/dx [1 - cosh((y - h/2)/sqrt(k)) / cosh(h/(2 sqrt(k)))].
#[must_use]
pub fn brinkman_velocity_profile(y: f64, h: f64, k: f64, mu: f64, dp_dx: f64) -> f64 {
    let s = k.sqrt();
    -(k / mu) * dp_dx * (1.0 - ((y - 0.5 * h) / s).cosh() / (0.5 * h / s).cosh())
}

/// Hydraulic conductivity K = k rho g / mu (m/s).
#[must_use]
pub fn hydraulic_conductivity(k: f64, rho: f64, g: f64, mu: f64) -> f64 {
    k * rho * g / mu
}

// ---------------------------------------------------------------------------
// Unsaturated flow: Van Genuchten + Richards
// ---------------------------------------------------------------------------

/// Van Genuchten soil retention parameters (`alpha` in 1/m, `k_s` in m/s).
#[derive(Debug, Clone, Copy)]
pub struct VanGenuchten {
    pub theta_r: f64,
    pub theta_s: f64,
    pub alpha: f64,
    pub n: f64,
    pub k_s: f64,
}

impl VanGenuchten {
    fn m(&self) -> f64 {
        1.0 - 1.0 / self.n
    }

    /// Water content at pressure head `h` (m, negative when unsaturated).
    #[must_use]
    pub fn theta(&self, h: f64) -> f64 {
        if h >= 0.0 {
            return self.theta_s;
        }
        let m = self.m();
        self.theta_r
            + (self.theta_s - self.theta_r)
                / (1.0 + (self.alpha * h.abs()).powf(self.n)).powf(m)
    }

    /// Effective saturation at pressure head `h`.
    #[must_use]
    pub fn effective_saturation(&self, h: f64) -> f64 {
        (self.theta(h) - self.theta_r) / (self.theta_s - self.theta_r)
    }

    /// Unsaturated hydraulic conductivity K(h) by Mualem-Van Genuchten.
    #[must_use]
    pub fn k(&self, h: f64) -> f64 {
        if h >= 0.0 {
            return self.k_s;
        }
        let m = self.m();
        let se = self.effective_saturation(h);
        self.k_s * se.sqrt() * (1.0 - (1.0 - se.powf(1.0 / m)).powf(m)).powi(2)
    }

    /// Specific moisture capacity C(h) = d theta / dh.
    #[must_use]
    pub fn capacity(&self, h: f64) -> f64 {
        if h >= 0.0 {
            return 0.0;
        }
        let m = self.m();
        let ah = (self.alpha * h.abs()).powf(self.n);
        (self.theta_s - self.theta_r) * m * self.n * self.alpha * ah
            / (self.alpha * h.abs() * (1.0 + ah).powf(m + 1.0))
    }

    /// Pressure head from water content (inverse retention curve).
    #[must_use]
    pub fn head_from_theta(&self, theta: f64) -> f64 {
        if theta >= self.theta_s {
            return 0.0;
        }
        let se = ((theta - self.theta_r) / (self.theta_s - self.theta_r)).clamp(1e-9, 1.0);
        let m = self.m();
        -(se.powf(-1.0 / m) - 1.0).powf(1.0 / self.n) / self.alpha
    }

    /// Typical sand.
    #[must_use]
    pub fn sand() -> Self {
        Self {
            theta_r: 0.045,
            theta_s: 0.43,
            alpha: 14.5,
            n: 2.68,
            k_s: 8.25e-5,
        }
    }

    /// Typical loam.
    #[must_use]
    pub fn loam() -> Self {
        Self {
            theta_r: 0.078,
            theta_s: 0.43,
            alpha: 3.6,
            n: 1.56,
            k_s: 2.89e-6,
        }
    }

    /// Typical clay.
    #[must_use]
    pub fn clay() -> Self {
        Self {
            theta_r: 0.068,
            theta_s: 0.38,
            alpha: 0.8,
            n: 1.09,
            k_s: 5.56e-7,
        }
    }
}

/// Brooks-Corey effective saturation for entry pressure head `h_b` (m) and
/// pore-size index `lambda`.
#[must_use]
pub fn brooks_corey(h: f64, h_b: f64, lambda: f64) -> f64 {
    if h >= -h_b.abs() {
        1.0
    } else {
        (h_b.abs() / h.abs()).powf(lambda)
    }
}

/// Explicit finite-volume Richards equation in 1D (z positive downward),
/// theta-form so mass is conserved exactly. Boundary conditions are water
/// fluxes (m/s, positive downward): `bc_top` enters the first cell,
/// `bc_bottom` leaves the last. Returns the water-content profile after each
/// step (`steps + 1` rows including the initial state).
#[must_use]
pub fn richards_equation_1d(
    theta0: &[f64],
    soil: &VanGenuchten,
    dz: f64,
    dt: f64,
    steps: usize,
    bc_top: f64,
    bc_bottom: f64,
) -> Vec<Vec<f64>> {
    let n = theta0.len();
    let mut theta = theta0.to_vec();
    let mut out = Vec::with_capacity(steps + 1);
    out.push(theta.clone());
    for _ in 0..steps {
        let h: Vec<f64> = theta.iter().map(|&t| soil.head_from_theta(t)).collect();
        let k: Vec<f64> = h.iter().map(|&hh| soil.k(hh)).collect();
        // interface fluxes q_{i+1/2} = -K (dh/dz - 1) (z down: gravity aids
        // downward flow)
        let mut q = vec![0.0; n + 1];
        q[0] = bc_top;
        q[n] = bc_bottom;
        for i in 1..n {
            let k_face = 0.5 * (k[i - 1] + k[i]);
            q[i] = -k_face * ((h[i] - h[i - 1]) / dz - 1.0);
        }
        for i in 0..n {
            theta[i] += dt / dz * (q[i] - q[i + 1]);
            theta[i] = theta[i].clamp(soil.theta_r + 1e-9, soil.theta_s);
        }
        out.push(theta.clone());
    }
    out
}

// ---------------------------------------------------------------------------
// Well hydraulics
// ---------------------------------------------------------------------------

/// Theis transient drawdown at radius `r` and time `t` for pumping rate `q`,
/// transmissivity `t_coeff` and storativity `s`:
/// s_d = Q/(4 pi T) W(u), u = r^2 S/(4 T t), W = E1.
#[must_use]
pub fn theis_drawdown(q: f64, t_coeff: f64, s: f64, r: f64, t: f64) -> f64 {
    let u = r * r * s / (4.0 * t_coeff * t);
    q / (4.0 * std::f64::consts::PI * t_coeff) * crate::special::e1(u)
}

/// Thiem steady-state head at radius `r2` given head `h1` at `r1`:
/// h2 = h1 + Q/(2 pi T) ln(r2/r1).
#[must_use]
pub fn thiem_steady(q: f64, t_coeff: f64, r1: f64, r2: f64, h1: f64) -> f64 {
    h1 + q / (2.0 * std::f64::consts::PI * t_coeff) * (r2 / r1).ln()
}

/// Dupuit unconfined flow between heads `h1` and `h2` over length `l`:
/// the water-table height at distance `x`.
#[must_use]
pub fn dupuit_unconfined(h1: f64, h2: f64, l: f64, x: f64) -> f64 {
    (h1 * h1 - (h1 * h1 - h2 * h2) * x / l).max(0.0).sqrt()
}

/// Steady 2D groundwater flow: solve div(K grad h) = -recharge with
/// Dirichlet head fixed at the listed `(i, j, head)` cells and no-flow
/// elsewhere on the boundary. Gauss-Seidel with harmonic-mean face
/// conductivities.
#[must_use]
pub fn groundwater_flow_2d(
    k_field: &CellField2,
    bc: &[(usize, usize, f64)],
    recharge: &CellField2,
) -> CellField2 {
    let (nx, ny, dx) = (k_field.nx, k_field.ny, k_field.dx);
    let mut h = CellField2::new(nx, ny, dx);
    let mut fixed = vec![false; nx * ny];
    for &(i, j, head) in bc {
        h.data[j * nx + i] = head;
        fixed[j * nx + i] = true;
    }
    let kf = |a: f64, b: f64| 2.0 * a * b / (a + b).max(1e-300);
    for _ in 0..20_000 {
        let mut max_change = 0.0_f64;
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                if fixed[c] {
                    continue;
                }
                let kc = k_field.data[c];
                let mut num = recharge.data[c] * dx * dx;
                let mut den = 0.0;
                if i > 0 {
                    let kw = kf(kc, k_field.data[c - 1]);
                    num += kw * h.data[c - 1];
                    den += kw;
                }
                if i + 1 < nx {
                    let ke = kf(kc, k_field.data[c + 1]);
                    num += ke * h.data[c + 1];
                    den += ke;
                }
                if j > 0 {
                    let ks = kf(kc, k_field.data[c - nx]);
                    num += ks * h.data[c - nx];
                    den += ks;
                }
                if j + 1 < ny {
                    let kn = kf(kc, k_field.data[c + nx]);
                    num += kn * h.data[c + nx];
                    den += kn;
                }
                if den == 0.0 {
                    continue;
                }
                let new = num / den;
                max_change = max_change.max((new - h.data[c]).abs());
                h.data[c] = new;
            }
        }
        if max_change < 1e-12 {
            break;
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Solute transport
// ---------------------------------------------------------------------------

/// Peclet number for porous transport: Pe = u l / D.
#[must_use]
pub fn peclet_porous(u: f64, l: f64, d: f64) -> f64 {
    u * l / d
}

/// Hydrodynamic dispersion coefficient D = alpha_L u + D_m.
#[must_use]
pub fn dispersion_coefficient(alpha_l: f64, u: f64, d_m: f64) -> f64 {
    alpha_l * u + d_m
}

/// One explicit step of the 1D advection-dispersion-reaction equation with
/// retardation factor R and first-order decay:
/// R dc/dt + u dc/dx = D d2c/dx2 - R lambda c (upwind advection).
#[must_use]
pub fn advection_dispersion_1d(
    c: &[f64],
    u: f64,
    d: f64,
    dx: f64,
    dt: f64,
    retardation: f64,
    decay: f64,
) -> Vec<f64> {
    let n = c.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let cm = if i > 0 { c[i - 1] } else { c[0] };
        let cp = if i + 1 < n { c[i + 1] } else { c[n - 1] };
        let adv = if u >= 0.0 {
            u * (c[i] - cm) / dx
        } else {
            u * (cp - c[i]) / dx
        };
        let disp = d * (cp - 2.0 * c[i] + cm) / (dx * dx);
        out[i] = c[i] + dt * ((disp - adv) / retardation - decay * c[i]);
    }
    out
}

/// Ogata-Banks solution for continuous injection at x = 0 into an initially
/// clean semi-infinite column: c/c0 at (x, t).
#[must_use]
pub fn ogata_banks(x: f64, t: f64, u: f64, d: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let s = 2.0 * (d * t).sqrt();
    let term1 = crate::special::erfc((x - u * t) / s);
    // guard the exponentially growing factor with its decaying erfc partner
    let arg = u * x / d;
    let term2 = if arg > 700.0 {
        0.0
    } else {
        arg.exp() * crate::special::erfc((x + u * t) / s)
    };
    0.5 * (term1 + term2)
}

// ---------------------------------------------------------------------------
// Two-phase relations
// ---------------------------------------------------------------------------

/// Leverett J-function scaling of capillary pressure:
/// Pc = sigma cos(theta) sqrt(phi/k) J(Sw), with J = 0.5 Sw^{-1/2}.
#[must_use]
pub fn capillary_pressure_leverett(sw: f64, porosity: f64, k: f64, sigma: f64, theta: f64) -> f64 {
    let j = 0.5 / sw.max(1e-9).sqrt();
    sigma * theta.cos() * (porosity / k).sqrt() * j
}

/// Corey relative permeabilities `(k_rw, k_ro)` with residual saturations
/// and exponent `n`.
#[must_use]
pub fn relative_permeability_corey(sw: f64, sw_r: f64, so_r: f64, n: f64) -> (f64, f64) {
    let se = ((sw - sw_r) / (1.0 - sw_r - so_r)).clamp(0.0, 1.0);
    (se.powf(n), (1.0 - se).powf(n))
}

fn fractional_flow(sw: f64, sw_r: f64, so_r: f64, mu_w: f64, mu_o: f64) -> f64 {
    let (krw, kro) = relative_permeability_corey(sw, sw_r, so_r, 2.0);
    let lw = krw / mu_w;
    let lo = kro / mu_o;
    if lw + lo == 0.0 { 0.0 } else { lw / (lw + lo) }
}

/// Buckley-Leverett water saturation at position `x` and time `t` for total
/// (Darcy) velocity `u_total` injected into a column at connate water
/// saturation, using quadratic Corey curves. Returns Sw(x, t) including the
/// Welge shock front.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn buckley_leverett(
    x: f64,
    t: f64,
    u_total: f64,
    porosity: f64,
    mu_w: f64,
    mu_o: f64,
    sw_r: f64,
    so_r: f64,
) -> f64 {
    if t <= 0.0 || x < 0.0 {
        return sw_r;
    }
    let f = |s: f64| fractional_flow(s, sw_r, so_r, mu_w, mu_o);
    let dfds = |s: f64| {
        let h = 1e-6;
        (f(s + h) - f(s - h)) / (2.0 * h)
    };
    let s_max = 1.0 - so_r;
    // Welge tangent from Sw = sw_r: f'(S_f) = f(S_f)/(S_f - sw_r)
    let mut s_front = s_max - 1e-6;
    let (mut lo, mut hi) = (sw_r + 1e-4, s_max - 1e-6);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let g = dfds(mid) - f(mid) / (mid - sw_r);
        if g > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        s_front = 0.5 * (lo + hi);
    }
    let xi = porosity * x / (u_total * t); // = f'(S) on the rarefaction
    let xi_front = f(s_front) / (s_front - sw_r);
    if xi >= xi_front {
        return sw_r;
    }
    // invert f'(S) = xi on [s_front, s_max] where f' is decreasing in S
    let (mut lo, mut hi) = (s_front, s_max);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if dfds(mid) > xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Porosity reduction by biofilm growth: phi = phi0 - biomass/rho_biofilm.
#[must_use]
pub fn bioclogging_porosity_change(phi0: f64, biomass: f64, rho_biofilm: f64) -> f64 {
    (phi0 - biomass / rho_biofilm).max(0.0)
}

/// Effective thermal conductivity of a saturated porous medium (geometric
/// mean mixing): k_eff = k_s^(1-phi) k_f^phi.
#[must_use]
pub fn effective_thermal_conductivity_porous(k_s: f64, k_f: f64, porosity: f64) -> f64 {
    k_s.powf(1.0 - porosity) * k_f.powf(porosity)
}

/// Gravity number: ratio of gravity to viscous forces in porous flow
/// (used in the tests for scaling sanity).
#[must_use]
pub fn gravity_number(k: f64, rho: f64, mu: f64, u: f64) -> f64 {
    k * rho * G / (mu * u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_darcy_ergun_consistency() {
        // laminar Ergun limit equals Darcy with Kozeny-Carman permeability
        let (phi, d, mu, rho, l) = (0.4, 1e-3, 1e-3, 1000.0, 2.0);
        let u = 1e-5; // slow enough that inertia is negligible
        let dp_ergun = ergun_pressure_drop(u, d, phi, mu, rho, l);
        let k = permeability_kozeny_carman(phi, d);
        let dp_darcy = mu * u * l / k;
        assert!(
            (dp_ergun - dp_darcy).abs() / dp_darcy < 0.01,
            "{dp_ergun} vs {dp_darcy}"
        );
        // Forchheimer reduces to Darcy at low speed
        let fchh = forchheimer(k, 0.0, mu, rho, u);
        assert!((fchh - mu * u / k).abs() < 1e-18);
        // Brinkman: symmetric channel profile, zero at walls, positive core
        let (h, kk, dpdx) = (1.0, 1e-4, -1.0);
        assert!(brinkman_velocity_profile(0.0, h, kk, mu, dpdx).abs() < 1e-10);
        assert!(brinkman_velocity_profile(h, h, kk, mu, dpdx).abs() < 1e-10);
        let mid = brinkman_velocity_profile(0.5 * h, h, kk, mu, dpdx);
        assert!(mid > 0.0);
        // deep-channel core approaches the Darcy plug velocity
        assert!((mid - kk / mu).abs() / (kk / mu) < 1e-6);
        assert!(hydraulic_conductivity(1e-12, 1000.0, 9.81, 1e-3) > 0.0);
        assert!(darcy_velocity(1e-12, 1e-3, -1e4) > 0.0);
        assert!(darcy_flow_rate(1e-12, 1.0, 1e-3, 1e4, 1.0) > 0.0);
        assert!(carman_kozeny_fibers(0.7, 1e-5) > 0.0);
    }

    #[test]
    fn test_van_genuchten() {
        let s = VanGenuchten::sand();
        // saturated at h >= 0
        assert!((s.theta(0.0) - s.theta_s).abs() < 1e-15);
        assert!((s.k(0.0) - s.k_s).abs() < 1e-20);
        // dry limit approaches residual
        assert!((s.theta(-100.0) - s.theta_r).abs() < 0.01);
        // monotone in h
        assert!(s.theta(-0.1) > s.theta(-1.0));
        assert!(s.k(-0.1) > s.k(-1.0));
        // capacity is the numerical derivative of theta
        let h = -0.5;
        let dh = 1e-6;
        let c_num = (s.theta(h + dh) - s.theta(h - dh)) / (2.0 * dh);
        assert!((s.capacity(h) - c_num).abs() / c_num < 1e-6);
        // inverse retention consistent
        let th = s.theta(-0.7);
        assert!((s.head_from_theta(th) + 0.7).abs() < 1e-9);
        // soils ordered by conductivity: sand > loam > clay
        assert!(VanGenuchten::sand().k_s > VanGenuchten::loam().k_s);
        assert!(VanGenuchten::loam().k_s > VanGenuchten::clay().k_s);
        // Brooks-Corey saturated above entry pressure
        assert!((brooks_corey(-0.05, 0.1, 2.0) - 1.0).abs() < 1e-15);
        assert!(brooks_corey(-1.0, 0.1, 2.0) < 0.02);
    }

    #[test]
    fn test_richards_mass_conservation() {
        // closed column (zero-flux BCs): total water conserved exactly
        let soil = VanGenuchten::loam();
        let n = 30;
        let dz = 0.02;
        // wet top half, drier bottom half
        let theta0: Vec<f64> = (0..n)
            .map(|i| if i < n / 2 { 0.35 } else { 0.15 })
            .collect();
        let hist = richards_equation_1d(&theta0, &soil, dz, 5.0, 400, 0.0, 0.0);
        let mass0: f64 = hist[0].iter().sum::<f64>() * dz;
        let mass1: f64 = hist.last().unwrap().iter().sum::<f64>() * dz;
        assert!(
            (mass1 - mass0).abs() < 1e-12 * mass0.max(1.0),
            "mass {mass0} -> {mass1}"
        );
        // water moved downward (gravity): bottom half gained
        let bottom0: f64 = hist[0][n / 2..].iter().sum();
        let bottom1: f64 = hist.last().unwrap()[n / 2..].iter().sum();
        assert!(bottom1 > bottom0, "no downward redistribution");
        // infiltration BC adds the right mass
        let hist2 = richards_equation_1d(&theta0, &soil, dz, 5.0, 100, 1e-6, 0.0);
        let mass2: f64 = hist2.last().unwrap().iter().sum::<f64>() * dz;
        let expected = mass0 + 1e-6 * 5.0 * 100.0;
        assert!((mass2 - expected).abs() < 1e-12, "{mass2} vs {expected}");
    }

    #[test]
    fn test_well_hydraulics() {
        // Theis at late time matches the Jacob approximation
        // s = Q/(4 pi T) [ln(2.25 T t/(r^2 S))]
        let (q, t_c, s, r) = (1e-3, 1e-3, 1e-4, 10.0);
        let t = 1e7; // u << 0.01
        let dd = theis_drawdown(q, t_c, s, r, t);
        let u = r * r * s / (4.0 * t_c * t);
        assert!(u < 0.01);
        let jacob =
            q / (4.0 * std::f64::consts::PI * t_c) * (2.25 * t_c * t / (r * r * s)).ln();
        assert!((dd - jacob).abs() / jacob < 0.01, "{dd} vs {jacob}");
        // drawdown decreases with distance
        assert!(theis_drawdown(q, t_c, s, 50.0, t) < dd);
        // Thiem: head rises away from a pumping well
        let h2 = thiem_steady(1e-3, 1e-3, 1.0, 100.0, 5.0);
        assert!(h2 > 5.0);
        // Dupuit endpoints
        assert!((dupuit_unconfined(10.0, 6.0, 100.0, 0.0) - 10.0).abs() < 1e-12);
        assert!((dupuit_unconfined(10.0, 6.0, 100.0, 100.0) - 6.0).abs() < 1e-12);
        // steady 2D groundwater: uniform K, fixed heads left/right ->
        // linear profile
        let nx = 21;
        let ny = 5;
        let mut kf = CellField2::new(nx, ny, 1.0);
        kf.data.iter_mut().for_each(|v| *v = 1e-4);
        let recharge = CellField2::new(nx, ny, 1.0);
        let mut bc = Vec::new();
        for j in 0..ny {
            bc.push((0, j, 10.0));
            bc.push((nx - 1, j, 5.0));
        }
        let h = groundwater_flow_2d(&kf, &bc, &recharge);
        let mid = h.data[(ny / 2) * nx + nx / 2];
        assert!((mid - 7.5).abs() < 1e-6, "midpoint head {mid}");
    }

    #[test]
    fn test_solute_transport() {
        // Ogata-Banks vs explicit advection-dispersion within 1%
        let (u, d) = (1e-3, 5e-4);
        let dx = 0.01;
        let n = 400;
        let dt = 0.4 * dx * dx / d;
        let t_end = 1000.0_f64;
        let steps = (t_end / dt).ceil() as usize;
        let dt = t_end / steps as f64;
        let mut c = vec![0.0; n];
        c[0] = 1.0;
        for _ in 0..steps {
            c = advection_dispersion_1d(&c, u, d, dx, dt, 1.0, 0.0);
            c[0] = 1.0; // continuous injection boundary
        }
        // compare in the mid-plume region
        for &xi in &[0.4, 0.8, 1.2] {
            let i = (xi / dx).round() as usize;
            let exact = ogata_banks(xi, t_end, u, d);
            assert!(
                (c[i] - exact).abs() < 0.01,
                "x={xi}: {} vs {exact}",
                c[i]
            );
        }
        // retardation slows the front
        let mut cr = vec![0.0; n];
        cr[0] = 1.0;
        for _ in 0..steps {
            cr = advection_dispersion_1d(&cr, u, d, dx, dt, 3.0, 0.0);
            cr[0] = 1.0;
        }
        let front = |cc: &[f64]| cc.iter().position(|&v| v < 0.5).unwrap_or(n);
        assert!(front(&cr) < front(&c), "retardation did not slow front");
        assert!((peclet_porous(1e-3, 1.0, 5e-4) - 2.0).abs() < 1e-12);
        assert!((dispersion_coefficient(0.1, 1e-3, 1e-9) - 1.00001e-4).abs() < 1e-15);
    }

    #[test]
    fn test_buckley_leverett() {
        let (u, phi, mu_w, mu_o, swr, sor) = (1e-5, 0.25, 1e-3, 4e-3, 0.15, 0.2);
        let t = 5e5;
        // behind the front: high saturation; far ahead: connate
        let s_near = buckley_leverett(0.05, t, u, phi, mu_w, mu_o, swr, sor);
        let s_far = buckley_leverett(50.0, t, u, phi, mu_w, mu_o, swr, sor);
        assert!(s_near > 0.6, "near saturation {s_near}");
        assert!((s_far - swr).abs() < 1e-12);
        // saturation profile is monotone decreasing in x
        let mut prev = f64::MAX;
        for k in 1..40 {
            let s = buckley_leverett(k as f64 * 0.5, t, u, phi, mu_w, mu_o, swr, sor);
            assert!(s <= prev + 1e-9);
            prev = s;
        }
        // shock front position scales linearly with time
        let find_front = |tt: f64| {
            let mut x = 0.0;
            for k in 1..4000 {
                let xx = k as f64 * 0.01;
                if buckley_leverett(xx, tt, u, phi, mu_w, mu_o, swr, sor) <= swr + 1e-9 {
                    x = xx;
                    break;
                }
            }
            x
        };
        let x1 = find_front(2e5);
        let x2 = find_front(4e5);
        assert!((x2 / x1 - 2.0).abs() < 0.05, "front {x1} -> {x2}");
        // misc
        assert!(capillary_pressure_leverett(0.5, 0.2, 1e-12, 0.03, 0.0) > 0.0);
        let (krw, kro) = relative_permeability_corey(0.5, 0.1, 0.1, 2.0);
        assert!((krw - 0.25).abs() < 1e-12 && (kro - 0.25).abs() < 1e-12);
        assert!((bioclogging_porosity_change(0.3, 10.0, 100.0) - 0.2).abs() < 1e-12);
        let ke = effective_thermal_conductivity_porous(2.0, 0.6, 0.4);
        assert!(ke > 0.6 && ke < 2.0);
        assert!(gravity_number(1e-12, 1000.0, 1e-3, 1e-5) > 0.0);
    }
}
