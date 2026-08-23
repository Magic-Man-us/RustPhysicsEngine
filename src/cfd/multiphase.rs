//! Multiphase flow correlations: mixture properties, drift-flux and void
//! fraction models, two-phase pressure drop, flow-pattern maps, bubble and
//! droplet dynamics, population balance, boiling and condensation, sprays,
//! and dispersed-particle transport.

use crate::math::Vec3;

const G: f64 = 9.806_65;
const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Mixture properties
// ---------------------------------------------------------------------------

/// Mixture density rho_m = alpha rho_g + (1 - alpha) rho_l for void
/// fraction `alpha`.
#[must_use]
pub fn mixture_density(alpha: f64, rho_g: f64, rho_l: f64) -> f64 {
    alpha * rho_g + (1.0 - alpha) * rho_l
}

/// McAdams homogeneous mixture viscosity from quality `x`:
/// 1/mu_m = x/mu_g + (1-x)/mu_l.
#[must_use]
pub fn mixture_viscosity_mcadams(x: f64, mu_g: f64, mu_l: f64) -> f64 {
    1.0 / (x / mu_g + (1.0 - x) / mu_l)
}

/// Dukler mixture viscosity: mu_m = rho_m (x mu_g/rho_g + (1-x) mu_l/rho_l).
#[must_use]
pub fn mixture_viscosity_dukler(x: f64, rho_g: f64, rho_l: f64, mu_g: f64, mu_l: f64) -> f64 {
    let rho_m = 1.0 / (x / rho_g + (1.0 - x) / rho_l);
    rho_m * (x * mu_g / rho_g + (1.0 - x) * mu_l / rho_l)
}

// ---------------------------------------------------------------------------
// Drift flux and void fraction
// ---------------------------------------------------------------------------

/// Drift-flux gas velocity v_g = C0 j + v_gj with total superficial
/// velocity j = j_g + j_l.
#[must_use]
pub fn drift_flux_velocity(j_g: f64, j_l: f64, c0: f64, v_gj: f64) -> f64 {
    c0 * (j_g + j_l) + v_gj
}

/// Homogeneous (no-slip) void fraction from quality `x`.
#[must_use]
pub fn void_fraction_homogeneous(x: f64, rho_g: f64, rho_l: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 + (1.0 - x) / x * rho_g / rho_l)
}

/// Drift-flux void fraction alpha = j_g / (C0 j + v_gj).
#[must_use]
pub fn void_fraction_drift_flux(j_g: f64, j_l: f64, c0: f64, v_gj: f64) -> f64 {
    j_g / drift_flux_velocity(j_g, j_l, c0, v_gj)
}

/// Lockhart-Martinelli void fraction (turbulent-turbulent):
/// alpha = 1 - 1/sqrt(1 + 20/X + 1/X^2) with the Martinelli parameter from
/// quality and fluid properties.
#[must_use]
pub fn void_fraction_lockhart_martinelli(
    x: f64,
    rho_g: f64,
    rho_l: f64,
    mu_g: f64,
    mu_l: f64,
) -> f64 {
    let xtt = martinelli_parameter(x, rho_g, rho_l, mu_g, mu_l);
    1.0 - 1.0 / (1.0 + 20.0 / xtt + 1.0 / (xtt * xtt)).sqrt()
}

/// Turbulent-turbulent Martinelli parameter
/// X_tt = ((1-x)/x)^0.9 (rho_g/rho_l)^0.5 (mu_l/mu_g)^0.1.
#[must_use]
pub fn martinelli_parameter(x: f64, rho_g: f64, rho_l: f64, mu_g: f64, mu_l: f64) -> f64 {
    ((1.0 - x) / x).powf(0.9) * (rho_g / rho_l).sqrt() * (mu_l / mu_g).powf(0.1)
}

// ---------------------------------------------------------------------------
// Two-phase pressure drop
// ---------------------------------------------------------------------------

/// Lockhart-Martinelli two-phase pressure gradient from the single-phase
/// liquid and gas gradients: dp_tp = dp_l phi_l^2 with
/// phi_l^2 = 1 + C/X + 1/X^2, X^2 = dp_l/dp_g.
#[must_use]
pub fn two_phase_pressure_drop_lockhart_martinelli(dp_l: f64, dp_g: f64, c: f64) -> f64 {
    let x = (dp_l / dp_g).sqrt();
    dp_l * (1.0 + c / x + 1.0 / (x * x))
}

/// Chisholm C coefficient from the flow regimes of each phase
/// (turbulent-turbulent 20, viscous-turbulent 12, turbulent-viscous 10,
/// viscous-viscous 5).
#[must_use]
pub fn chisholm(re_l: f64, re_g: f64) -> f64 {
    let lt = re_l > 2300.0;
    let gt = re_g > 2300.0;
    match (lt, gt) {
        (true, true) => 20.0,
        (false, true) => 12.0,
        (true, false) => 10.0,
        (false, false) => 5.0,
    }
}

/// Simplified Friedel two-phase multiplier phi_lo^2 for the
/// liquid-only pressure gradient, using the homogeneous density, Froude and
/// Weber corrections.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn friedel_correlation(
    x: f64,
    rho_g: f64,
    rho_l: f64,
    mu_g: f64,
    mu_l: f64,
    sigma: f64,
    d: f64,
    mass_flux: f64,
) -> f64 {
    let rho_h = 1.0 / (x / rho_g + (1.0 - x) / rho_l);
    let e = (1.0 - x).powi(2) + x * x * rho_l / rho_g * (mu_g / mu_l).powf(0.25);
    let f = x.powf(0.78) * (1.0 - x).powf(0.224);
    let h = (rho_l / rho_g).powf(0.91) * (mu_g / mu_l).powf(0.19)
        * (1.0 - mu_g / mu_l).powf(0.7);
    let fr = mass_flux * mass_flux / (G * d * rho_h * rho_h);
    let we = mass_flux * mass_flux * d / (rho_h * sigma);
    e + 3.24 * f * h / (fr.powf(0.045) * we.powf(0.035))
}

// ---------------------------------------------------------------------------
// Flow pattern map
// ---------------------------------------------------------------------------

/// Horizontal / near-horizontal two-phase flow patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowPattern {
    Stratified,
    Intermittent,
    Annular,
    DispersedBubble,
}

/// Simplified Taitel-Dukler flow-pattern map for a pipe of diameter `d` at
/// inclination `inclination` (radians from horizontal), from superficial
/// velocities `j_g`, `j_l`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn flow_pattern_taitel_dukler(
    j_g: f64,
    j_l: f64,
    d: f64,
    rho_g: f64,
    rho_l: f64,
    mu_g: f64,
    mu_l: f64,
    inclination: f64,
) -> FlowPattern {
    let cos_i = inclination.cos().max(1e-6);
    // superficial friction pressure gradients (Blasius / laminar)
    let grad = |j: f64, rho: f64, mu: f64| -> f64 {
        let re = (rho * j * d / mu).max(1e-9);
        let f = if re < 2300.0 { 16.0 / re } else { 0.046 / re.powf(0.2) };
        2.0 * f * rho * j * j / d
    };
    let dp_l = grad(j_l, rho_l, mu_l);
    let dp_g = grad(j_g, rho_g, mu_g);
    let x2 = dp_l / dp_g;
    let x = x2.sqrt();
    // approximate equilibrium liquid level (turbulent-turbulent fit)
    let h_tilde = (x.powf(0.85) / (1.0 + x.powf(0.85))).clamp(1e-4, 1.0 - 1e-4);
    // circular-pipe geometry at dimensionless level h_tilde
    let phi = 2.0 * (1.0 - 2.0 * h_tilde).acos(); // wetted angle
    let a_total = PI / 4.0;
    let a_l = (phi - phi.sin()) / 8.0;
    let a_g = a_total - a_l;
    let u_g_hat = a_total / a_g; // dimensionless gas velocity
    let dadh = (1.0 - (2.0 * h_tilde - 1.0).powi(2)).sqrt();
    // Taitel-Dukler modified Froude number
    let f = (rho_g / ((rho_l - rho_g) * d * G * cos_i)).sqrt() * j_g;
    // criterion A: stratified -> non-stratified transition
    let stratified_stable =
        f * f * (u_g_hat * u_g_hat * dadh / a_g) / ((1.0 - h_tilde).powi(2)) < 1.0;
    if stratified_stable {
        return FlowPattern::Stratified;
    }
    // non-stratified: annular if the liquid level is low
    if h_tilde < 0.35 {
        return FlowPattern::Annular;
    }
    // intermittent vs dispersed bubble: T parameter (turbulence vs buoyancy)
    let t = (dp_l / ((rho_l - rho_g) * G * cos_i)).sqrt();
    let s_i = dadh; // interface width (dimensionless)
    let u_l_hat = a_total / a_l.max(1e-9);
    let dispersed = t * t >= 8.0 * a_g / (s_i * u_l_hat * u_l_hat);
    if dispersed {
        FlowPattern::DispersedBubble
    } else {
        FlowPattern::Intermittent
    }
}

// ---------------------------------------------------------------------------
// Bubbles and droplets
// ---------------------------------------------------------------------------

/// Eotvos (Bond) number Eo = delta_rho g d^2 / sigma.
#[must_use]
pub fn eotvos(delta_rho: f64, g: f64, d: f64, sigma: f64) -> f64 {
    delta_rho * g * d * d / sigma
}

/// Morton number Mo = g mu^4 / (rho sigma^3) (continuous-phase properties,
/// density difference folded into g for near-unit density ratios).
#[must_use]
pub fn morton_number(g: f64, mu: f64, rho: f64, sigma: f64) -> f64 {
    g * mu.powi(4) / (rho * sigma.powi(3))
}

/// Tomiyama drag coefficient for a contaminated bubble:
/// Cd = max(24/Re (1 + 0.15 Re^0.687), 8 Eo / (3 (Eo + 4))).
#[must_use]
pub fn bubble_drag_coefficient(re: f64, eo: f64, _mo: f64) -> f64 {
    let cd_visc = 24.0 / re * (1.0 + 0.15 * re.powf(0.687));
    let cd_dist = 8.0 * eo / (3.0 * (eo + 4.0));
    cd_visc.max(cd_dist)
}

/// Terminal rise velocity of a bubble of diameter `d`: force balance with
/// the Tomiyama contaminated drag law, solved by bisection. Reduces to the
/// Stokes settling formula for tiny bubbles and to the Eotvos-limited cap
/// regime for large ones.
#[must_use]
pub fn bubble_rise_velocity(d: f64, rho_l: f64, rho_g: f64, mu_l: f64, sigma: f64) -> f64 {
    let drho = (rho_l - rho_g).max(0.0);
    if drho == 0.0 {
        return 0.0;
    }
    let eo = eotvos(drho, G, d, sigma);
    let mo = morton_number(G, mu_l, rho_l, sigma);
    let balance = |u: f64| -> f64 {
        let re = (rho_l * u * d / mu_l).max(1e-12);
        let cd = bubble_drag_coefficient(re, eo, mo);
        // (4/3) d g drho - Cd rho_l u^2
        4.0 / 3.0 * d * G * drho - cd * rho_l * u * u
    };
    let (mut lo, mut hi) = (1e-12, 20.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if balance(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Stokes terminal velocity of a small droplet in a continuous phase:
/// u = g d^2 (rho_d - rho_c) / (18 mu_c).
#[must_use]
pub fn droplet_terminal_velocity(d: f64, rho_d: f64, rho_c: f64, mu_c: f64) -> f64 {
    G * d * d * (rho_d - rho_c) / (18.0 * mu_c)
}

/// Sauter mean diameter d32 = sum d^3 / sum d^2.
#[must_use]
pub fn sauter_mean_diameter(diameters: &[f64]) -> f64 {
    let d3: f64 = diameters.iter().map(|&d| d * d * d).sum();
    let d2: f64 = diameters.iter().map(|&d| d * d).sum();
    if d2 == 0.0 { 0.0 } else { d3 / d2 }
}

/// Rosin-Rammler cumulative mass fraction below diameter `d`:
/// F = 1 - exp(-(d/d_mean)^n).
#[must_use]
pub fn rosin_rammler(d: f64, d_mean: f64, n: f64) -> f64 {
    1.0 - (-(d / d_mean).powf(n)).exp()
}

/// Simplified Luo-Svendsen breakup rate for a bubble/droplet of diameter
/// `d` in turbulence of dissipation `eps` at dispersed-phase fraction
/// `alpha`: rate ~ 0.923 (1-alpha) (eps/d^2)^{1/3}
/// exp(-12 sigma / (2.05 rho_c eps^{2/3} d^{5/3})).
#[must_use]
pub fn breakup_rate_luo_svendsen(alpha: f64, eps: f64, d: f64, sigma: f64, rho_c: f64) -> f64 {
    let we_crit = 12.0 * sigma / (2.05 * rho_c * eps.powf(2.0 / 3.0) * d.powf(5.0 / 3.0));
    0.923 * (1.0 - alpha) * (eps / (d * d)).powf(1.0 / 3.0) * (-we_crit).exp()
}

/// Simplified Prince-Blanch coalescence kernel for bubbles of diameters
/// `d1`, `d2` (turbulent collision frequency times a film-drainage
/// efficiency).
#[must_use]
pub fn coalescence_rate_prince_blanch(
    d1: f64,
    d2: f64,
    eps: f64,
    rho_c: f64,
    sigma: f64,
) -> f64 {
    let rc = 0.25 * (d1 + d2);
    let collision = 0.089 * PI * (d1 + d2).powi(2)
        * (d1.powf(2.0 / 3.0) + d2.powf(2.0 / 3.0)).sqrt()
        * eps.powf(1.0 / 3.0);
    let we = rho_c * eps.powf(2.0 / 3.0) * (2.0 * rc).powf(5.0 / 3.0) / sigma;
    collision * (-(we / 4.0).sqrt()).exp()
}

/// One explicit step of a discrete population balance on size classes
/// `sizes` (diameters) with number densities `n`: binary breakage into two
/// equal-volume daughters and pairwise coalescence into the nearest class by
/// volume. Number densities update; volume moves between resolved classes.
#[must_use]
pub fn population_balance_1d(
    n: &[f64],
    sizes: &[f64],
    breakup: &dyn Fn(f64) -> f64,
    coalescence: &dyn Fn(f64, f64) -> f64,
    dt: f64,
) -> Vec<f64> {
    let nc = n.len();
    assert_eq!(nc, sizes.len());
    let vols: Vec<f64> = sizes.iter().map(|&d| PI / 6.0 * d * d * d).collect();
    let mut dn = vec![0.0; nc];
    // fixed-pivot deposit: split `amount` particles of volume `v` between the
    // bracketing classes so both number and volume are conserved; outside the
    // grid, scale the count to conserve volume
    let deposit = |dn: &mut [f64], v: f64, amount: f64| {
        if v <= vols[0] {
            dn[0] += amount * v / vols[0];
        } else if v >= vols[nc - 1] {
            dn[nc - 1] += amount * v / vols[nc - 1];
        } else {
            let k = vols.partition_point(|&vv| vv <= v) - 1;
            let f = (vols[k + 1] - v) / (vols[k + 1] - vols[k]);
            dn[k] += amount * f;
            dn[k + 1] += amount * (1.0 - f);
        }
    };
    // binary breakage into equal-volume daughters
    for (i, (&ni, &di)) in n.iter().zip(sizes).enumerate() {
        let rate = breakup(di) * ni;
        if rate <= 0.0 {
            continue;
        }
        dn[i] -= rate;
        deposit(&mut dn, 0.5 * vols[i], 2.0 * rate);
    }
    // pairwise coalescence
    for i in 0..nc {
        for j in i..nc {
            let sym = if i == j { 0.5 } else { 1.0 };
            let rate = sym * coalescence(sizes[i], sizes[j]) * n[i] * n[j];
            if rate <= 0.0 {
                continue;
            }
            dn[i] -= rate;
            dn[j] -= rate;
            deposit(&mut dn, vols[i] + vols[j], rate);
        }
    }
    n.iter()
        .zip(&dn)
        .map(|(&ni, &di)| (ni + dt * di).max(0.0))
        .collect()
}

// ---------------------------------------------------------------------------
// Phase change
// ---------------------------------------------------------------------------

/// Cavitation number sigma_c = (p - p_v) / (rho u^2 / 2).
#[must_use]
pub fn cavitation_number(p: f64, p_v: f64, rho: f64, u: f64) -> f64 {
    (p - p_v) / (0.5 * rho * u * u)
}

/// Saturated-fluid property bundle for boiling correlations.
#[derive(Debug, Clone, Copy)]
pub struct SaturatedFluid {
    pub mu_l: f64,
    pub h_fg: f64,
    pub rho_l: f64,
    pub rho_g: f64,
    pub sigma: f64,
    pub cp_l: f64,
    pub pr_l: f64,
}

impl SaturatedFluid {
    /// Water at 1 atm saturation.
    #[must_use]
    pub fn water_1atm() -> Self {
        Self {
            mu_l: 2.79e-4,
            h_fg: 2.257e6,
            rho_l: 957.9,
            rho_g: 0.596,
            sigma: 0.0589,
            cp_l: 4217.0,
            pr_l: 1.76,
        }
    }
}

/// Rohsenow nucleate-boiling heat flux for wall superheat `delta_t` (K)
/// with surface constant `c_sf` (0.013 for water on polished surfaces).
#[must_use]
pub fn boiling_heat_flux_rohsenow(delta_t: f64, fluid: &SaturatedFluid, c_sf: f64) -> f64 {
    let f = fluid;
    f.mu_l
        * f.h_fg
        * (G * (f.rho_l - f.rho_g) / f.sigma).sqrt()
        * (f.cp_l * delta_t / (c_sf * f.h_fg * f.pr_l)).powi(3)
}

/// Zuber critical heat flux:
/// q_chf = 0.131 h_fg rho_g^{1/2} (sigma g (rho_l - rho_g))^{1/4}.
#[must_use]
pub fn critical_heat_flux_zuber(fluid: &SaturatedFluid) -> f64 {
    0.131
        * fluid.h_fg
        * fluid.rho_g.sqrt()
        * (fluid.sigma * G * (fluid.rho_l - fluid.rho_g)).powf(0.25)
}

/// Nusselt laminar film condensation coefficient on a vertical plate of
/// height `height` with wall subcooling `delta_t` and liquid conductivity
/// `k_l`: h = 0.943 [rho_l (rho_l - rho_g) g h_fg k^3 / (mu dT L)]^{1/4}.
#[must_use]
pub fn condensation_nusselt_film(
    fluid: &SaturatedFluid,
    k_l: f64,
    delta_t: f64,
    height: f64,
) -> f64 {
    0.943
        * (fluid.rho_l * (fluid.rho_l - fluid.rho_g) * G * fluid.h_fg * k_l.powi(3)
            / (fluid.mu_l * delta_t * height))
            .powf(0.25)
}

/// Hertz-Knudsen maximum evaporation mass flux (kg/m^2/s) for molar mass
/// `m` (kg/mol) at temperature `t`: J = (p_sat - p) sqrt(m / (2 pi R T)).
#[must_use]
pub fn evaporation_rate_hertz_knudsen(p_sat: f64, p: f64, t: f64, m: f64) -> f64 {
    (p_sat - p) * (m / (2.0 * PI * 8.314_462_618 * t)).sqrt()
}

/// Hiroyasu spray tip penetration for injection pressure drop `delta_p`
/// into gas of density `rho_a` through a nozzle of diameter `d_nozzle`,
/// at time `t` after start of injection.
#[must_use]
pub fn spray_penetration_hiroyasu(
    delta_p: f64,
    rho_l: f64,
    rho_a: f64,
    d_nozzle: f64,
    t: f64,
) -> f64 {
    let t_break = 28.65 * rho_l * d_nozzle / (rho_a * delta_p).sqrt();
    if t < t_break {
        0.39 * (2.0 * delta_p / rho_l).sqrt() * t
    } else {
        2.95 * (delta_p / rho_a).powf(0.25) * (d_nozzle * t).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Dispersed particles
// ---------------------------------------------------------------------------

/// Particle response time tau_p = rho_p d^2 / (18 mu).
#[must_use]
pub fn particle_response_time(rho_p: f64, d_p: f64, mu: f64) -> f64 {
    rho_p * d_p * d_p / (18.0 * mu)
}

/// Stokes number St = tau_p u / l.
#[must_use]
pub fn stokes_number(rho_p: f64, d_p: f64, u: f64, mu: f64, l: f64) -> f64 {
    particle_response_time(rho_p, d_p, mu) * u / l
}

/// One-way-coupled particle step with exact exponential integration of
/// dv/dt = (u_f - v)/tau_p + g. `p` is (position, velocity).
pub fn particle_tracking_step(
    p: &mut (Vec3, Vec3),
    fluid_vel: Vec3,
    tau_p: f64,
    g: Vec3,
    dt: f64,
) {
    let v_inf = fluid_vel + g * tau_p; // terminal velocity in this frame
    let decay = (-dt / tau_p).exp();
    let v0 = p.1;
    let v1 = v_inf + (v0 - v_inf) * decay;
    // exact position integral of v(t)
    p.0 = p.0 + v_inf * dt + (v0 - v_inf) * (tau_p * (1.0 - decay));
    p.1 = v1;
}

/// Terminal settling velocity of a sphere with the Schiller-Naumann drag
/// Cd = 24/Re (1 + 0.15 Re^0.687), solved by bisection.
#[must_use]
pub fn settling_velocity(d: f64, rho_p: f64, rho_f: f64, mu: f64, g: f64) -> f64 {
    let drho = rho_p - rho_f;
    if drho <= 0.0 {
        return 0.0;
    }
    let balance = |u: f64| -> f64 {
        let re = (rho_f * u * d / mu).max(1e-12);
        let cd = 24.0 / re * (1.0 + 0.15 * re.powf(0.687));
        4.0 / 3.0 * d * g * drho - cd * rho_f * u * u
    };
    let (mut lo, mut hi) = (1e-12, 100.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if balance(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Minimum fluidization velocity from the Ergun equation balanced against
/// the bed weight at voidage `porosity`, solved by bisection.
#[must_use]
pub fn fluidization_minimum_velocity(
    d: f64,
    rho_p: f64,
    rho_f: f64,
    mu: f64,
    porosity: f64,
) -> f64 {
    let e = porosity;
    let weight = (1.0 - e) * (rho_p - rho_f) * G;
    let ergun = |u: f64| {
        150.0 * mu * u * (1.0 - e).powi(2) / (e.powi(3) * d * d)
            + 1.75 * rho_f * u * u * (1.0 - e) / (e.powi(3) * d)
    };
    let (mut lo, mut hi) = (0.0, 100.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if ergun(mid) < weight {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Richardson-Zaki hindered settling velocity u = u_t phi^n where `phi` is
/// the fluid voidage.
#[must_use]
pub fn sedimentation_richardson_zaki(u_t: f64, porosity: f64, n: f64) -> f64 {
    u_t * porosity.powf(n)
}

/// Richardson-Zaki exponent as a function of particle Reynolds number.
#[must_use]
pub fn hindered_settling_exponent(re: f64) -> f64 {
    if re < 0.2 {
        4.65
    } else if re < 1.0 {
        4.4 * re.powf(-0.03)
    } else if re < 500.0 {
        4.4 * re.powf(-0.1)
    } else {
        2.4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixture_and_void() {
        assert!((mixture_density(0.3, 1.0, 1000.0) - 700.3).abs() < 1e-9);
        // McAdams between the two pure viscosities, closer to gas at high x
        let m1 = mixture_viscosity_mcadams(0.9, 1e-5, 1e-3);
        assert!(m1 > 1e-5 && m1 < 1e-3 && m1 < 5e-5);
        let m2 = mixture_viscosity_dukler(0.5, 1.0, 1000.0, 1e-5, 1e-3);
        assert!(m2 > 1e-5 && m2 < 1e-3);
        // homogeneous void: x = 0 -> 0, high x with light gas -> near 1
        assert!(void_fraction_homogeneous(0.0, 1.0, 1000.0) == 0.0);
        assert!(void_fraction_homogeneous(0.5, 1.0, 1000.0) > 0.99);
        // drift flux reduces to homogeneous for C0 = 1, v_gj = 0
        let (jg, jl) = (0.5, 1.0);
        let a = void_fraction_drift_flux(jg, jl, 1.0, 0.0);
        assert!((a - jg / (jg + jl)).abs() < 1e-12);
        // slip (C0 > 1) lowers void fraction
        assert!(void_fraction_drift_flux(jg, jl, 1.2, 0.2) < a);
        // Lockhart-Martinelli void between 0 and 1, increasing in quality
        let vf1 = void_fraction_lockhart_martinelli(0.1, 1.0, 1000.0, 1e-5, 1e-3);
        let vf2 = void_fraction_lockhart_martinelli(0.5, 1.0, 1000.0, 1e-5, 1e-3);
        assert!(vf1 > 0.0 && vf2 < 1.0 && vf2 > vf1);
    }

    #[test]
    fn test_pressure_drop() {
        // L-M multiplier exceeds both single-phase gradients
        let (dpl, dpg) = (100.0, 400.0);
        let dp = two_phase_pressure_drop_lockhart_martinelli(dpl, dpg, 20.0);
        assert!(dp > dpl && dp > dpg);
        // X = 0.5 -> phi_l^2 = 1 + 40 + 4 = 45
        assert!((dp - dpl * 45.0).abs() < 1e-9);
        assert!((chisholm(1e4, 1e4) - 20.0).abs() < 1e-12);
        assert!((chisholm(1e3, 1e3) - 5.0).abs() < 1e-12);
        // Friedel multiplier > 1 for typical air-water
        let phi2 = friedel_correlation(0.1, 1.2, 998.0, 1.8e-5, 1e-3, 0.072, 0.025, 500.0);
        assert!(phi2 > 1.0, "friedel {phi2}");
    }

    #[test]
    fn test_taitel_dukler() {
        let (d, rho_g, rho_l, mu_g, mu_l) = (0.05, 1.2, 998.0, 1.8e-5, 1e-3);
        // low velocities, horizontal -> stratified
        let p = flow_pattern_taitel_dukler(0.05, 0.01, d, rho_g, rho_l, mu_g, mu_l, 0.0);
        assert_eq!(p, FlowPattern::Stratified);
        // high gas, little liquid -> annular
        let p = flow_pattern_taitel_dukler(30.0, 0.02, d, rho_g, rho_l, mu_g, mu_l, 0.0);
        assert_eq!(p, FlowPattern::Annular);
        // both high with dominant liquid -> dispersed bubble
        let p = flow_pattern_taitel_dukler(0.3, 4.0, d, rho_g, rho_l, mu_g, mu_l, 0.0);
        assert_eq!(p, FlowPattern::DispersedBubble);
        // moderate gas and liquid -> intermittent (slug/plug)
        let p = flow_pattern_taitel_dukler(1.0, 0.5, d, rho_g, rho_l, mu_g, mu_l, 0.0);
        assert_eq!(p, FlowPattern::Intermittent);
    }

    #[test]
    fn test_bubbles_droplets() {
        let (rho_l, rho_g, mu_l, sigma) = (998.0, 1.2, 1e-3, 0.072);
        // tiny bubble: Stokes regime matches the settling formula magnitude
        let d = 3e-5;
        let u = bubble_rise_velocity(d, rho_l, rho_g, mu_l, sigma);
        let u_stokes = settling_velocity(d, rho_l, rho_g, mu_l, G); // same balance
        assert!((u - u_stokes).abs() / u_stokes < 0.01, "{u} vs {u_stokes}");
        // millimetric air bubble in water rises O(20-30 cm/s)
        let u_mm = bubble_rise_velocity(1.5e-3, rho_l, rho_g, mu_l, sigma);
        assert!(u_mm > 0.05 && u_mm < 0.5, "u_mm = {u_mm}");
        // rise velocity increases with size in the small-bubble regime
        assert!(bubble_rise_velocity(1e-4, rho_l, rho_g, mu_l, sigma) > u);
        // droplet Stokes formula sign: heavy droplet falls (positive down)
        assert!(droplet_terminal_velocity(1e-4, 998.0, 1.2, 1.8e-5) > 0.0);
        // Sauter mean of equal diameters is that diameter
        assert!((sauter_mean_diameter(&[2e-3, 2e-3]) - 2e-3).abs() < 1e-18);
        // Rosin-Rammler: F(d_mean) = 1 - 1/e
        assert!((rosin_rammler(1e-3, 1e-3, 2.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
        // breakup rate increases with dissipation, coalescence positive
        let b1 = breakup_rate_luo_svendsen(0.05, 0.1, 5e-3, sigma, rho_l);
        let b2 = breakup_rate_luo_svendsen(0.05, 10.0, 5e-3, sigma, rho_l);
        assert!(b2 > b1 && b1 >= 0.0);
        assert!(coalescence_rate_prince_blanch(1e-3, 2e-3, 1.0, rho_l, sigma) > 0.0);
        assert!((eotvos(997.0, G, 1e-3, sigma) - 997.0 * G * 1e-6 / sigma).abs() < 1e-9);
        // water Morton number ~ 2.6e-11
        let mo = morton_number(G, 1e-3, 998.0, 0.072);
        assert!(mo > 1e-11 && mo < 1e-10);
    }

    #[test]
    fn test_population_balance() {
        // pure breakage: numbers grow, total volume conserved
        let sizes: Vec<f64> = (0..8).map(|k| 1e-3 * 2.0_f64.powf(k as f64 / 3.0)).collect();
        // classes are volume-doubling every 3 steps: halves land 3 below
        let n0 = {
            let mut v = vec![0.0; 8];
            v[7] = 100.0;
            v
        };
        let breakup = |_d: f64| 0.1;
        let none = |_a: f64, _b: f64| 0.0;
        let vol = |d: f64| d * d * d;
        let total_vol = |n: &[f64]| -> f64 {
            n.iter().zip(&sizes).map(|(&ni, &d)| ni * vol(d)).sum()
        };
        let v0 = total_vol(&n0);
        let mut n = n0.clone();
        for _ in 0..10 {
            n = population_balance_1d(&n, &sizes, &breakup, &none, 0.1);
        }
        assert!((total_vol(&n) - v0).abs() / v0 < 1e-9, "breakage volume");
        assert!(n.iter().sum::<f64>() > 100.0, "breakage increases count");
        // pure coalescence: numbers shrink, volume conserved
        let n1 = {
            let mut v = vec![0.0; 8];
            v[0] = 100.0;
            v
        };
        let nob = |_d: f64| 0.0;
        let coal = |_a: f64, _b: f64| 1e-3;
        let v1 = total_vol(&n1);
        let mut n = n1.clone();
        for _ in 0..10 {
            n = population_balance_1d(&n, &sizes, &nob, &coal, 0.1);
        }
        assert!((total_vol(&n) - v1).abs() / v1 < 1e-9, "coalescence volume");
        assert!(n.iter().sum::<f64>() < 100.0, "coalescence reduces count");
    }

    #[test]
    fn test_phase_change() {
        let w = SaturatedFluid::water_1atm();
        // Zuber CHF for water ~ 1.1 MW/m^2
        let chf = critical_heat_flux_zuber(&w);
        assert!(chf > 0.8e6 && chf < 1.5e6, "CHF {chf}");
        // Rohsenow: q ~ dT^3, so doubling superheat gives 8x flux
        let q1 = boiling_heat_flux_rohsenow(5.0, &w, 0.013);
        let q2 = boiling_heat_flux_rohsenow(10.0, &w, 0.013);
        assert!((q2 / q1 - 8.0).abs() < 1e-9);
        assert!(q1 > 0.0);
        // film condensation h decreases with plate height (thicker film)
        let h1 = condensation_nusselt_film(&w, 0.68, 10.0, 0.1);
        let h2 = condensation_nusselt_film(&w, 0.68, 10.0, 1.0);
        assert!(h1 > h2 && h2 > 0.0);
        // Hertz-Knudsen: zero at equilibrium, positive for supersaturation
        assert!(evaporation_rate_hertz_knudsen(1e5, 1e5, 373.0, 0.018).abs() < 1e-12);
        assert!(evaporation_rate_hertz_knudsen(1.1e5, 1e5, 373.0, 0.018) > 0.0);
        // cavitation number decreases with speed
        assert!(
            cavitation_number(1e5, 2.3e3, 998.0, 10.0)
                > cavitation_number(1e5, 2.3e3, 998.0, 20.0)
        );
        // spray penetration: continuous at breakup time, grows with sqrt(t)
        let (dp, rl, ra, dn) = (5e7_f64, 800.0_f64, 25.0_f64, 2e-4_f64);
        let tb = 28.65 * rl * dn / (ra * dp).sqrt();
        let s1 = spray_penetration_hiroyasu(dp, rl, ra, dn, tb * 0.999);
        let s2 = spray_penetration_hiroyasu(dp, rl, ra, dn, tb * 1.001);
        assert!((s1 - s2).abs() / s1 < 0.1, "jump at breakup: {s1} vs {s2}");
        let s4 = spray_penetration_hiroyasu(dp, rl, ra, dn, 4.0 * tb);
        assert!((s4 / s2 - 2.0).abs() < 0.05, "sqrt(t) growth");
    }

    #[test]
    fn test_particles() {
        // response time and Stokes number scale as d^2
        let tau = particle_response_time(2500.0, 1e-5, 1.8e-5);
        assert!((stokes_number(2500.0, 1e-5, 1.0, 1.8e-5, 0.01) - tau * 100.0).abs() < 1e-12);
        // settling: small particle matches Stokes formula
        let d = 1e-5;
        let u = settling_velocity(d, 2500.0, 1.2, 1.8e-5, G);
        let stokes = G * d * d * (2500.0 - 1.2) / (18.0 * 1.8e-5);
        assert!((u - stokes).abs() / stokes < 0.01, "{u} vs {stokes}");
        // particle tracking relaxes to terminal velocity
        let mut p = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
        let gvec = Vec3::new(0.0, -G, 0.0);
        for _ in 0..200 {
            particle_tracking_step(&mut p, Vec3::new(0.0, 0.0, 0.0), tau, gvec, tau);
        }
        assert!((p.1.y + G * tau).abs() < 1e-9, "terminal velocity");
        assert!(p.0.y < 0.0);
        // still fluid, no gravity: particle keeps initial position
        let mut q = (Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 0.0));
        particle_tracking_step(&mut q, Vec3::new(0.0, 0.0, 0.0), tau, Vec3::new(0.0, 0.0, 0.0), 1.0);
        assert!((q.0 - Vec3::new(1.0, 2.0, 3.0)).magnitude() < 1e-15);
        // fluidization velocity below single-particle settling
        let umf = fluidization_minimum_velocity(5e-4, 2500.0, 1000.0, 1e-3, 0.45);
        let ut = settling_velocity(5e-4, 2500.0, 1000.0, 1e-3, G);
        assert!(umf > 0.0 && umf < ut, "umf {umf} vs ut {ut}");
        // Richardson-Zaki: hindered slower than free settling
        let n_rz = hindered_settling_exponent(1000.0);
        assert!((n_rz - 2.4).abs() < 1e-12);
        assert!(sedimentation_richardson_zaki(ut, 0.6, 4.65) < ut);
        assert!(hindered_settling_exponent(0.1) > hindered_settling_exponent(10.0));
    }
}
