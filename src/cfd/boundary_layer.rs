//! Boundary layers: Blasius and Falkner-Skan similarity solutions
//! (shooting), Thwaites and Head integral methods, turbulent wall laws,
//! transition and separation criteria, rotating and oscillating layers,
//! and flat-plate heat transfer.

use crate::math::Vec2;
use crate::special::erfc;

/// Blasius flat-plate similarity solution: rows (η, f, f′, f″) with
/// f‴ + ½ f f″ = 0, solved by RK4 shooting on f″(0).
#[must_use]
pub fn blasius_solve(eta_max: f64, n: usize) -> Vec<(f64, f64, f64, f64)> {
    let integrate = |fpp0: f64, record: bool| -> (f64, Vec<(f64, f64, f64, f64)>) {
        let h = eta_max / n as f64;
        let (mut f, mut fp, mut fpp) = (0.0, 0.0, fpp0);
        let mut rows = Vec::with_capacity(if record { n + 1 } else { 0 });
        if record {
            rows.push((0.0, f, fp, fpp));
        }
        let deriv = |f: f64, fp: f64, fpp: f64| -> (f64, f64, f64) {
            (fp, fpp, -0.5 * f * fpp)
        };
        for i in 0..n {
            let k1 = deriv(f, fp, fpp);
            let k2 = deriv(f + 0.5 * h * k1.0, fp + 0.5 * h * k1.1, fpp + 0.5 * h * k1.2);
            let k3 = deriv(f + 0.5 * h * k2.0, fp + 0.5 * h * k2.1, fpp + 0.5 * h * k2.2);
            let k4 = deriv(f + h * k3.0, fp + h * k3.1, fpp + h * k3.2);
            f += h / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
            fp += h / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
            fpp += h / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);
            if record {
                rows.push(((i + 1) as f64 * h, f, fp, fpp));
            }
        }
        (fp, rows)
    };
    // Bisection on f″(0) for f′(∞) = 1.
    let (mut lo, mut hi) = (0.1, 0.6);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let (fp_end, _) = integrate(mid, false);
        if fp_end < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    integrate(0.5 * (lo + hi), true).1
}

/// Blasius streamwise velocity u(y) at station x.
#[must_use]
pub fn blasius_profile(y: f64, x: f64, u_inf: f64, nu: f64) -> f64 {
    let eta = y * (u_inf / (nu * x)).sqrt();
    let sol = blasius_solve(10.0, 400);
    let h = 10.0 / 400.0;
    if eta >= 10.0 {
        return u_inf;
    }
    let i = (eta / h) as usize;
    let frac = eta / h - i as f64;
    let fp = sol[i].2 * (1.0 - frac) + sol[(i + 1).min(400)].2 * frac;
    u_inf * fp
}

/// Blasius thicknesses (δ99, δ*, θ) at station x.
#[must_use]
pub fn blasius_thickness(x: f64, u_inf: f64, nu: f64) -> (f64, f64, f64) {
    let scale = (nu * x / u_inf).sqrt();
    (4.91 * scale, 1.7208 * scale, 0.664 * scale)
}

/// Local Blasius skin friction 0.664/√Re_x.
#[must_use]
pub fn blasius_cf(x: f64, u_inf: f64, nu: f64) -> f64 {
    0.664 / (u_inf * x / nu).sqrt()
}

/// Total laminar drag of one side of a flat plate.
#[must_use]
pub fn blasius_drag_plate(l: f64, width: f64, u_inf: f64, nu: f64, rho: f64) -> f64 {
    let re_l = u_inf * l / nu;
    1.328 / re_l.sqrt() * 0.5 * rho * u_inf * u_inf * l * width
}

/// Falkner-Skan wedge-flow similarity: rows (η, f, f′) for
/// f‴ + f f″ + β(1 − f′²) = 0.
#[must_use]
pub fn falkner_skan_solve(beta: f64, eta_max: f64, n: usize) -> Vec<(f64, f64, f64)> {
    let integrate = |fpp0: f64, record: bool| -> (f64, Vec<(f64, f64, f64)>) {
        let h = eta_max / n as f64;
        let (mut f, mut fp, mut fpp) = (0.0, 0.0, fpp0);
        let mut rows = Vec::new();
        if record {
            rows.push((0.0, f, fp));
        }
        let deriv = |f: f64, fp: f64, fpp: f64| -> (f64, f64, f64) {
            (fp, fpp, -f * fpp - beta * (1.0 - fp * fp))
        };
        for i in 0..n {
            let k1 = deriv(f, fp, fpp);
            let k2 = deriv(f + 0.5 * h * k1.0, fp + 0.5 * h * k1.1, fpp + 0.5 * h * k1.2);
            let k3 = deriv(f + 0.5 * h * k2.0, fp + 0.5 * h * k2.1, fpp + 0.5 * h * k2.2);
            let k4 = deriv(f + h * k3.0, fp + h * k3.1, fpp + h * k3.2);
            f += h / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
            fp += h / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
            fpp += h / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);
            if !fp.is_finite() {
                break;
            }
            if record {
                rows.push(((i + 1) as f64 * h, f, fp));
            }
        }
        (fp, rows)
    };
    // fp(eta_max) is not monotone in f''(0) for adverse-gradient beta —
    // near separation it grazes 1 at the physical solution — so minimize
    // |fp(eta_max) - 1| by ternary search rather than bisecting a crossing.
    let score = |fpp0: f64| -> f64 {
        let (fp_end, _) = integrate(fpp0, false);
        if fp_end.is_finite() {
            (fp_end - 1.0).abs()
        } else {
            f64::INFINITY
        }
    };
    let (mut lo, mut hi) = (-0.2, 2.0);
    for _ in 0..150 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if score(m1) <= score(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    integrate(0.5 * (lo + hi), true).1
}

/// Falkner-Skan separation parameter β = −0.1988.
#[must_use]
pub fn falkner_skan_separation_beta() -> f64 {
    -0.1988
}

/// Thwaites integral method along stations `x` with edge velocity
/// `u_e(x)`: rows (θ, λ, H, cf).
#[must_use]
pub fn thwaites_method(
    u_e: &dyn Fn(f64) -> f64,
    x: &[f64],
    nu: f64,
) -> Vec<(f64, f64, f64, f64)> {
    // θ²(x) = 0.45 ν U⁻⁶ ∫₀ˣ U⁵ dx (cumulative trapezoid).
    let mut integral = 0.0;
    let mut prev_x = x[0];
    let mut prev_u5 = u_e(x[0]).powi(5);
    x.iter()
        .map(|&xi| {
            let u = u_e(xi).max(1e-12);
            let u5 = u.powi(5);
            integral += 0.5 * (prev_u5 + u5) * (xi - prev_x);
            prev_x = xi;
            prev_u5 = u5;
            let theta2 = 0.45 * nu / u.powi(6) * integral;
            let theta = theta2.max(0.0).sqrt();
            let due = ((u_e(xi + 1e-6) - u_e(xi - 1e-6)) / 2e-6).clamp(-1e6, 1e6);
            let lambda = theta2 / nu * due;
            // Standard Thwaites correlations for H(λ) and l(λ).
            let (h, l) = if lambda >= 0.0 {
                (
                    2.61 - 3.75 * lambda + 5.24 * lambda * lambda,
                    0.22 + 1.57 * lambda - 1.8 * lambda * lambda,
                )
            } else {
                (
                    2.088 + 0.0731 / (lambda + 0.14),
                    0.22 + 1.402 * lambda + 0.018 * lambda / (lambda + 0.107),
                )
            };
            let cf = if theta > 0.0 { 2.0 * nu * l / (u * theta) } else { 0.0 };
            (theta, lambda, h, cf)
        })
        .collect()
}

/// First station where Thwaites' λ drops below −0.09 (separation).
#[must_use]
pub fn thwaites_separation_point(u_e: &dyn Fn(f64) -> f64, x: &[f64], nu: f64) -> Option<f64> {
    let rows = thwaites_method(u_e, x, nu);
    x.iter()
        .zip(&rows)
        .find(|(_, r)| r.1 <= -0.09)
        .map(|(xi, _)| *xi)
}

/// Pohlhausen quartic velocity profile u/U at η = y/δ with shape
/// parameter λ.
#[must_use]
pub fn pohlhausen_profile(eta: f64, lambda: f64) -> f64 {
    let e = eta.clamp(0.0, 1.0);
    (2.0 * e - 2.0 * e.powi(3) + e.powi(4)) + lambda / 6.0 * e * (1.0 - e).powi(3)
}

/// Turbulent 1/n power-law profile.
#[must_use]
pub fn turbulent_bl_power_law(y: f64, delta: f64, n: f64) -> f64 {
    (y / delta).clamp(0.0, 1.0).powf(1.0 / n)
}

/// Local turbulent skin friction (Prandtl 1/5-power law) 0.0592 Re⁻⅕.
#[must_use]
pub fn turbulent_cf_prandtl(re_x: f64) -> f64 {
    0.0592 / re_x.powf(0.2)
}

/// Schlichting's local turbulent skin friction (2 log₁₀Re − 0.65)⁻²·³.
#[must_use]
pub fn turbulent_cf_schlichting(re_x: f64) -> f64 {
    (2.0 * re_x.log10() - 0.65).powf(-2.3)
}

/// Turbulent boundary-layer thickness δ = 0.37 x / Re_x^{1/5}.
#[must_use]
pub fn turbulent_thickness_1_7(x: f64, re_x: f64) -> f64 {
    0.37 * x / re_x.powf(0.2)
}

/// Logarithmic law of the wall u⁺ = ln(y⁺)/κ + B.
#[must_use]
pub fn law_of_the_wall(y_plus: f64, kappa: f64, b: f64) -> f64 {
    y_plus.max(1e-12).ln() / kappa + b
}

/// Spalding's composite wall profile: u⁺(y⁺) by inverting
/// y⁺ = u⁺ + e^{−κB}(e^{κu⁺} − 1 − κu⁺ − (κu⁺)²/2 − (κu⁺)³/6).
#[must_use]
pub fn spalding(y_plus: f64) -> f64 {
    let (kappa, b) = (0.41, 5.0);
    let y_of_u = |u: f64| -> f64 {
        let ku = kappa * u;
        u + (-kappa * b).exp() * (ku.exp() - 1.0 - ku - ku * ku / 2.0 - ku.powi(3) / 6.0)
    };
    let (mut lo, mut hi) = (0.0, 40.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if y_of_u(mid) < y_plus {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Van Driest near-wall damping 1 − e^{−y⁺/A}.
#[must_use]
pub fn van_driest_damping(y_plus: f64, a: f64) -> f64 {
    1.0 - (-y_plus / a).exp()
}

/// Wall coordinate y⁺ = y u_τ/ν.
#[must_use]
pub fn y_plus(y: f64, u_tau: f64, nu: f64) -> f64 {
    y * u_tau / nu
}

/// Friction velocity √(τ_w/ρ).
#[must_use]
pub fn u_tau(tau_w: f64, rho: f64) -> f64 {
    (tau_w / rho).sqrt()
}

/// First-cell height for a target y⁺ on a plate of length `l`
/// (turbulent flat-plate friction estimate).
#[must_use]
pub fn first_cell_height(y_plus_target: f64, u_inf: f64, nu: f64, re_l: f64, l: f64) -> f64 {
    let cf = 0.026 / re_l.powf(1.0 / 7.0);
    let tau = 0.5 * cf * u_inf * u_inf; // ρ = 1
    let ut = tau.sqrt();
    let _ = l;
    y_plus_target * nu / ut
}

/// Mayle-style transition estimate: Re_θt ≈ 400 Ti^{−5/8} (Ti in
/// percent), converted to Re_x with the Blasius relation θ =
/// 0.664 x/√Re_x.
#[must_use]
pub fn transition_re_x_estimate(turbulence_intensity: f64) -> f64 {
    let ti_pct = (turbulence_intensity * 100.0).max(0.05);
    let re_theta_t = 400.0 * ti_pct.powf(-5.0 / 8.0);
    (re_theta_t / 0.664).powi(2)
}

/// Michel's transition criterion: Re_θ > 1.174 (1 + 22400/Re_x) Re_x^0.46.
#[must_use]
pub fn michel_transition_criterion(re_theta: f64, re_x: f64) -> bool {
    re_theta > 1.174 * (1.0 + 22400.0 / re_x) * re_x.powf(0.46)
}

/// Head's entrainment integral method for turbulent boundary layers:
/// rows (θ, H, cf) marched along `x`.
#[must_use]
pub fn head_entrainment_method(
    u_e: &dyn Fn(f64) -> f64,
    x: &[f64],
    nu: f64,
    theta0: f64,
    h0: f64,
) -> Vec<(f64, f64, f64)> {
    let h1_of_h = |h: f64| -> f64 {
        if h <= 1.6 {
            3.3 + 0.8234 * (h - 1.1).max(1e-3).powf(-1.287)
        } else {
            3.3 + 1.5501 * (h - 0.6778).max(1e-3).powf(-3.064)
        }
    };
    let h_of_h1 = |h1: f64| -> f64 {
        // Invert numerically.
        let (mut lo, mut hi) = (1.05, 3.0);
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if h1_of_h(mid) > h1 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let mut theta = theta0;
    let mut h = h0;
    let mut out = Vec::with_capacity(x.len());
    for (k, &xi) in x.iter().enumerate() {
        let u = u_e(xi).max(1e-9);
        let due = (u_e(xi + 1e-6) - u_e(xi - 1e-6)) / 2e-6;
        let re_theta = (u * theta / nu).max(20.0);
        // Ludwieg-Tillmann skin friction.
        let cf = 0.246 * 10.0_f64.powf(-0.678 * h) * re_theta.powf(-0.268);
        out.push((theta, h, cf));
        if k + 1 >= x.len() {
            break;
        }
        let dx = x[k + 1] - xi;
        // Momentum integral.
        let dtheta = (0.5 * cf - (h + 2.0) * theta / u * due) * dx;
        // Entrainment: d(U θ H1)/dx = U · 0.0306 (H1 − 3)^{−0.6169}.
        let h1 = h1_of_h(h);
        let f_ent = 0.0306 * (h1 - 3.0).max(1e-3).powf(-0.6169);
        let uth1 = u * theta * h1 + u * f_ent * dx;
        theta = (theta + dtheta).max(1e-9);
        let u_next = u_e(x[k + 1]).max(1e-9);
        let h1_next = (uth1 / (u_next * theta)).max(3.05);
        h = h_of_h1(h1_next).clamp(1.05, 3.0);
    }
    out
}

/// Stratford's turbulent separation criterion:
/// Cp √(x dCp/dx) ≥ 0.39 (10⁻⁶ Re_x)^{0.1}.
#[must_use]
pub fn stratford_separation_criterion(cp: f64, x: f64, dcp_dx: f64, re_x: f64) -> bool {
    if cp <= 0.0 || dcp_dx <= 0.0 {
        return false;
    }
    cp * (x * dcp_dx).sqrt() >= 0.39 * (1e-6 * re_x).powf(0.1)
}

/// Ekman spiral velocity (u, v) at height z for geostrophic wind `u_g`.
#[must_use]
pub fn ekman_spiral(z: f64, u_g: f64, f: f64, nu: f64) -> Vec2 {
    let d = (2.0 * nu / f).sqrt();
    let zeta = z / d;
    Vec2::new(
        u_g * (1.0 - (-zeta).exp() * zeta.cos()),
        u_g * (-zeta).exp() * zeta.sin(),
    )
}

/// Ekman layer depth π√(2ν/f).
#[must_use]
pub fn ekman_depth(nu: f64, f: f64) -> f64 {
    crate::math::constants::PI * (2.0 * nu / f).sqrt()
}

/// Stokes' second problem (oscillating plate): u(y, t).
#[must_use]
pub fn stokes_second_problem(y: f64, t: f64, u0: f64, omega: f64, nu: f64) -> f64 {
    let k = (omega / (2.0 * nu)).sqrt();
    u0 * (-k * y).exp() * (omega * t - k * y).cos()
}

/// Stokes' first problem (impulsively started plate): u = u0 erfc(η).
#[must_use]
pub fn stokes_first_problem(y: f64, t: f64, u0: f64, nu: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    u0 * erfc(y / (2.0 * (nu * t).sqrt()))
}

/// Plane Couette-Poiseuille flow u(y) between plates 0 and h.
#[must_use]
pub fn couette_flow(y: f64, h: f64, u_wall: f64, dp_dx: f64, mu: f64) -> f64 {
    u_wall * y / h - dp_dx / (2.0 * mu) * y * (h - y)
}

/// Laminar flat-plate local Nusselt number 0.332 Re_x^½ Pr^⅓.
#[must_use]
pub fn flat_plate_heat_transfer_laminar(re_x: f64, pr: f64) -> f64 {
    0.332 * re_x.sqrt() * pr.cbrt()
}

/// Thermal to velocity boundary-layer thickness ratio ≈ Pr^{−1/3}.
#[must_use]
pub fn thermal_bl_ratio(pr: f64) -> f64 {
    pr.powf(-1.0 / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blasius() {
        let sol = blasius_solve(10.0, 800);
        let fpp0 = sol[0].3;
        assert!((fpp0 - 0.33206).abs() < 1e-4, "f''(0) = {fpp0}");
        // f'(∞) → 1, monotonically.
        assert!((sol.last().unwrap().2 - 1.0).abs() < 1e-6);
        for w in sol.windows(2) {
            assert!(w[1].2 >= w[0].2 - 1e-12, "profile not monotone");
        }
        // η99 ≈ 4.91.
        let eta99 = sol.iter().find(|r| r.2 >= 0.99).unwrap().0;
        assert!((eta99 - 4.91).abs() < 0.05, "eta99 {eta99}");
        // Profile helper consistent with thickness constants.
        let (u_inf, nu, x) = (2.0, 1.5e-5, 0.5);
        let (d99, dstar, theta) = blasius_thickness(x, u_inf, nu);
        assert!((blasius_profile(d99, x, u_inf, nu) / u_inf - 0.99).abs() < 0.005);
        assert!(dstar / theta > 2.5 && dstar / theta < 2.65, "H = {}", dstar / theta);
        assert!((blasius_cf(x, u_inf, nu) - 0.664 / (u_inf * x / nu).sqrt()).abs() < 1e-15);
        let drag = blasius_drag_plate(1.0, 1.0, 2.0, 1.5e-5, 1.2);
        assert!(drag > 0.0 && drag < 0.5 * 1.2 * 4.0);
    }

    #[test]
    fn test_falkner_skan() {
        // β = 0 recovers Blasius: in the standard FS form f''' + f f'' +
        // β(1 − f'^2) = 0 the wall shear is 0.46960 = sqrt(2) · 0.33206.
        // second-order one-sided difference for f''(0) from the fp column
        let wall_shear = |rows: &[(f64, f64, f64)]| {
            let h = rows[1].0 - rows[0].0;
            (-3.0 * rows[0].2 + 4.0 * rows[1].2 - rows[2].2) / (2.0 * h)
        };
        let b0 = falkner_skan_solve(0.0, 10.0, 800);
        let slope0 = wall_shear(&b0);
        assert!((slope0 - 0.46960).abs() < 5e-3, "FS β=0 f''(0) {slope0}");
        assert!(
            (slope0 / 2.0_f64.sqrt() - 0.33206).abs() < 5e-3,
            "FS β=0 rescaled to Blasius"
        );
        // Stagnation flow β = 1: f''(0) = 1.2326.
        let b1 = falkner_skan_solve(1.0, 8.0, 800);
        let slope1 = wall_shear(&b1);
        assert!((slope1 - 1.2326).abs() < 5e-3, "FS β=1 f''(0) {slope1}");
        // Near separation the wall shear vanishes.
        let bs = falkner_skan_solve(falkner_skan_separation_beta(), 10.0, 800);
        let slope_s = wall_shear(&bs);
        assert!(slope_s.abs() < 0.03, "FS separation f''(0) {slope_s}");
    }

    #[test]
    fn test_thwaites_howarth() {
        // Howarth's linearly decelerated flow U = 1 − x: Thwaites
        // separates near x = 0.123 (exact 0.1199).
        let nu = 1e-5;
        let xs: Vec<f64> = (1..=2000).map(|k| k as f64 * 0.0001).collect();
        let sep = thwaites_separation_point(&|x| 1.0 - x, &xs, nu).expect("separation");
        assert!((0.11..0.13).contains(&sep), "Howarth separation at {sep}");
        // Favorable gradient never separates.
        assert!(thwaites_separation_point(&|x| 1.0 + x, &xs, nu).is_none());
        // Flat plate: Thwaites θ within 3% of Blasius.
        let rows = thwaites_method(&|_| 1.0, &xs, nu);
        let theta_b = 0.664 * (nu * 0.2 / 1.0_f64).sqrt();
        let idx = xs.iter().position(|&x| x >= 0.2).unwrap();
        assert!(
            (rows[idx].0 / theta_b - 1.0).abs() < 0.03,
            "Thwaites θ {} vs Blasius {theta_b}",
            rows[idx].0
        );
        // H ≈ 2.61 on a flat plate, cf positive.
        assert!((rows[idx].2 - 2.61).abs() < 0.05);
        assert!(rows[idx].3 > 0.0);
        // Pohlhausen: no-slip and edge matching.
        assert_eq!(pohlhausen_profile(0.0, 7.0), 0.0);
        assert!((pohlhausen_profile(1.0, 7.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_wall_laws_and_transition() {
        // Spalding merges to the log law at large y+.
        let u_log = law_of_the_wall(100.0, 0.41, 5.0);
        let u_sp = spalding(100.0);
        assert!((u_sp / u_log - 1.0).abs() < 0.05, "{u_sp} vs {u_log}");
        // And to u+ = y+ in the sublayer.
        assert!((spalding(2.0) - 2.0).abs() < 0.15);
        assert!((van_driest_damping(26.0, 26.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
        assert!((y_plus(1e-4, 0.05, 1e-6) - 5.0).abs() < 1e-9);
        assert!((u_tau(0.09, 900.0) - 0.01).abs() < 1e-12);
        let h = first_cell_height(1.0, 10.0, 1.5e-5, 1e6, 1.5);
        assert!(h > 1e-7 && h < 1e-4, "first cell {h}");
        // Transition Reynolds falls with turbulence intensity.
        assert!(transition_re_x_estimate(0.001) > transition_re_x_estimate(0.05));
        assert!(transition_re_x_estimate(0.01) > 1e5);
        // Michel: laminar Re_theta below the curve, turbulent above.
        assert!(!michel_transition_criterion(300.0, 1e6));
        assert!(michel_transition_criterion(2000.0, 1e6));
        // Turbulent correlations decrease with Re and stay positive.
        assert!(turbulent_cf_prandtl(1e7) < turbulent_cf_prandtl(1e6));
        assert!(turbulent_cf_schlichting(1e6) > 0.0);
        let ratio = turbulent_cf_schlichting(1e6) / turbulent_cf_prandtl(1e6);
        assert!((0.6..1.6).contains(&ratio), "cf correlations differ: {ratio}");
        assert!(turbulent_thickness_1_7(1.0, 1e6) < 0.05);
        assert!((turbulent_bl_power_law(0.5, 1.0, 7.0) - 0.5_f64.powf(1.0 / 7.0)).abs() < 1e-12);
        // Stratford.
        assert!(!stratford_separation_criterion(0.05, 0.5, 0.1, 1e6));
        assert!(stratford_separation_criterion(0.6, 1.0, 1.0, 1e6));
    }

    #[test]
    fn test_head_method() {
        // Flat-plate turbulent layer: θ grows, H ≈ 1.3-1.5, cf sensible.
        let nu = 1.5e-5;
        let xs: Vec<f64> = (0..200).map(|k| 0.1 + k as f64 * 0.01).collect();
        let rows = head_entrainment_method(&|_| 10.0, &xs, nu, 2e-4, 1.4);
        assert!(rows.last().unwrap().0 > rows[0].0, "θ must grow");
        for r in &rows {
            assert!((1.2..1.9).contains(&r.1), "H drifted: {}", r.1);
            assert!(r.2 > 5e-4 && r.2 < 8e-3, "cf {}", r.2);
        }
    }

    #[test]
    fn test_rotating_and_unsteady_layers() {
        let (u_g, f, nu) = (10.0, 1e-4, 0.1);
        // Surface cross-isobar angle is 45°.
        let v_near = ekman_spiral(1e-4, u_g, f, nu);
        let angle = v_near.y.atan2(v_near.x).to_degrees();
        assert!((angle - 45.0).abs() < 0.5, "Ekman surface angle {angle}");
        // Far above the layer the wind is geostrophic.
        let deep = ekman_spiral(10.0 * ekman_depth(nu, f), u_g, f, nu);
        assert!((deep.x - u_g).abs() < 1e-3 && deep.y.abs() < 1e-3);
        assert!((ekman_depth(nu, f) - crate::math::constants::PI * (2.0_f64 * 0.1 / 1e-4).sqrt()).abs() < 1e-9);
        // Stokes problems.
        let u2 = stokes_second_problem(0.0, 0.0, 1.0, 5.0, 1e-3);
        assert!((u2 - 1.0).abs() < 1e-12);
        // Amplitude decays by e^{-1} at y = √(2ν/ω).
        let y_decay = (2.0 * 1e-3 / 5.0_f64).sqrt();
        let mut max_amp = 0.0_f64;
        for k in 0..200 {
            let t = k as f64 * 0.01;
            max_amp = max_amp.max(stokes_second_problem(y_decay, t, 1.0, 5.0, 1e-3).abs());
        }
        assert!((max_amp - (-1.0_f64).exp()).abs() < 0.01, "Stokes II decay {max_amp}");
        assert!((stokes_first_problem(0.0, 1.0, 3.0, 1e-3) - 3.0).abs() < 1e-9);
        assert!(stokes_first_problem(0.5, 1.0, 3.0, 1e-3) < 3.0);
        assert!(stokes_first_problem(0.5, 10.0, 3.0, 1e-3) > stokes_first_problem(0.5, 1.0, 3.0, 1e-3));
        // Couette-Poiseuille.
        assert!((couette_flow(0.5, 1.0, 2.0, 0.0, 1.0) - 1.0).abs() < 1e-12);
        let poise = couette_flow(0.5, 1.0, 0.0, -8.0, 1.0);
        assert!((poise - 1.0).abs() < 1e-12, "Poiseuille center {poise}");
        // Heat transfer.
        assert!((flat_plate_heat_transfer_laminar(1e4, 1.0) - 33.2).abs() < 0.1);
        assert!(thermal_bl_ratio(7.0) < 1.0);
        assert!(thermal_bl_ratio(0.01) > 1.0);
    }
}
