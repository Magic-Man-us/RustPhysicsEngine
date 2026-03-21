use crate::math::constants::{G, C, PI};

const C2: f64 = C * C;

// ── Schwarzschild Metric ────────────────────────────────────────────────

/// Schwarzschild radius: r_s = 2GM/c²
#[must_use]
pub fn schwarzschild_radius(mass: f64) -> f64 {
    2.0 * G * mass / C2
}

/// Event horizon radius (alias for schwarzschild_radius).
#[must_use]
pub fn event_horizon_radius(mass: f64) -> f64 {
    schwarzschild_radius(mass)
}

/// Time-time component of the Schwarzschild metric: g_tt = -(1 - r_s/r)
#[must_use]
pub fn schwarzschild_metric_tt(mass: f64, r: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    let rs = schwarzschild_radius(mass);
    -(1.0 - rs / r)
}

/// Radial-radial component of the Schwarzschild metric: g_rr = 1/(1 - r_s/r)
#[must_use]
pub fn schwarzschild_metric_rr(mass: f64, r: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    let rs = schwarzschild_radius(mass);
    assert!((1.0 - rs / r) != 0.0, "r must not equal Schwarzschild radius");
    1.0 / (1.0 - rs / r)
}

/// Proper time factor: dτ/dt = √(1 - r_s/r)
#[must_use]
pub fn proper_time_factor(mass: f64, r: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    let rs = schwarzschild_radius(mass);
    (1.0 - rs / r).sqrt()
}

/// Gravitational redshift factor between emitter at r_emit and observer at r_obs:
/// z_factor = √((1 - r_s/r_obs) / (1 - r_s/r_emit))
#[must_use]
pub fn gravitational_redshift_factor(mass: f64, r_emit: f64, r_obs: f64) -> f64 {
    assert!(r_emit > 0.0, "emitter radius must be positive");
    assert!(r_obs > 0.0, "observer radius must be positive");
    let rs = schwarzschild_radius(mass);
    assert!((1.0 - rs / r_emit) != 0.0, "r_emit must not equal Schwarzschild radius");
    ((1.0 - rs / r_obs) / (1.0 - rs / r_emit)).sqrt()
}

/// Innermost stable circular orbit for Schwarzschild: r_isco = 3 r_s = 6GM/c²
#[must_use]
pub fn isco_radius(mass: f64) -> f64 {
    3.0 * schwarzschild_radius(mass)
}

/// Photon sphere radius: r_ph = 1.5 r_s = 3GM/c²
#[must_use]
pub fn photon_sphere_radius(mass: f64) -> f64 {
    1.5 * schwarzschild_radius(mass)
}

// ── Kerr Metric (rotating black holes) ──────────────────────────────────

/// Kerr outer event horizon: r+ = GM/c² + √((GM/c²)² - a²)
/// where a is the spin parameter (dimensions of length).
#[must_use]
pub fn kerr_event_horizon(mass: f64, spin: f64) -> f64 {
    let m = G * mass / C2;
    m + (m * m - spin * spin).sqrt()
}

/// Kerr ergosphere radius at polar angle theta:
/// r_ergo = GM/c² + √((GM/c²)² - a²cos²θ)
#[must_use]
pub fn kerr_ergosphere_radius(mass: f64, spin: f64, theta: f64) -> f64 {
    let m = G * mass / C2;
    let a_cos = spin * theta.cos();
    m + (m * m - a_cos * a_cos).sqrt()
}

/// ISCO radius for a Kerr black hole using the exact Bardeen-Press-Teukolsky formula.
/// `spin` is the dimensionless spin parameter a/M (in geometric units, a* = Jc/(GM²)).
/// `prograde` selects co-rotating (true) or counter-rotating (false) orbits.
/// Returns the ISCO in metres.
#[must_use]
pub fn kerr_isco(mass: f64, spin: f64, prograde: bool) -> f64 {
    let a = spin.clamp(-1.0, 1.0);

    let z1 = 1.0 + (1.0 - a * a).cbrt() * ((1.0 + a).cbrt() + (1.0 - a).cbrt());
    let z2 = (3.0 * a * a + z1 * z1).sqrt();

    let r_over_m = if prograde {
        3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
    } else {
        3.0 + z2 + ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
    };

    r_over_m * G * mass / C2
}

/// Frame-dragging angular velocity (weak-field / Lense-Thirring limit):
/// Ω = 2GMa / (c²r³)
/// Here `spin` is the spin parameter a with dimensions of length.
#[must_use]
pub fn frame_dragging_rate(mass: f64, spin: f64, r: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    2.0 * G * mass * spin / (C2 * r * r * r)
}

// ── Geodesic Motion ─────────────────────────────────────────────────────

/// Radial geodesic acceleration in Schwarzschild spacetime (effective potential approach):
/// d²r/dτ² = -GM/r² + l²(r - 3GM/c²) / r⁴
/// where l is the specific angular momentum (per unit mass).
#[must_use]
pub fn geodesic_acceleration_schwarzschild(mass: f64, r: f64, _dr_dtau: f64, l: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    let gm = G * mass;
    let r2 = r * r;
    -gm / r2 + l * l * (r - 3.0 * gm / C2) / (r2 * r2)
}

/// Effective potential for a massive particle in Schwarzschild spacetime:
/// V_eff = -GMm/r + l²/(2mr²) - GMl²/(mc²r³)
#[must_use]
pub fn effective_potential_schwarzschild(mass: f64, r: f64, l: f64, particle_mass: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    assert!(particle_mass > 0.0, "particle_mass must be positive");
    let gm = G * mass;
    let m = particle_mass;
    -gm * m / r + l * l / (2.0 * m * r * r) - gm * l * l / (m * C2 * r * r * r)
}

/// Specific energy of a circular orbit in Schwarzschild:
/// E/(mc²) = (1 - 2GM/(rc²)) / √(1 - 3GM/(rc²))
#[must_use]
pub fn circular_orbit_energy(mass: f64, r: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    let x = G * mass / (r * C2);
    assert!((1.0 - 3.0 * x) > 0.0, "r must be outside the photon sphere");
    (1.0 - 2.0 * x) / (1.0 - 3.0 * x).sqrt()
}

/// Specific angular momentum of a circular orbit in Schwarzschild:
/// L/(mc) = r √(GM / (r c² - 3GM))  →  simplified from the exact expression.
/// Returns L/(mc) (dimensionless when r and GM/c² share length units, but here in SI
/// it carries dimensions of length).
#[must_use]
pub fn circular_orbit_angular_momentum(mass: f64, r: f64) -> f64 {
    assert!(r > 0.0, "radial coordinate must be positive");
    let gm = G * mass;
    assert!((r * C2 - 3.0 * gm) > 0.0, "r must be outside the photon sphere");
    r * (gm / (r * C2 - 3.0 * gm)).sqrt()
}

// ── Cosmology ───────────────────────────────────────────────────────────

/// Friedmann equation (flat universe, matter-dominated):
/// H = √(8πGρ/3)
/// For the full form with curvature and cosmological constant the caller should
/// construct the effective density; this returns the Hubble parameter for a given
/// energy density.
#[must_use]
pub fn friedmann_hubble(density: f64, _curvature: f64, _cosmological_constant: f64) -> f64 {
    (8.0 * PI * G * density / 3.0).sqrt()
}

/// Critical density of the universe: ρ_c = 3H² / (8πG)
#[must_use]
pub fn critical_density(hubble: f64) -> f64 {
    3.0 * hubble * hubble / (8.0 * PI * G)
}

/// Comoving distance via Hubble's law (valid for z << 1): d ≈ cz / H₀
#[must_use]
pub fn cosmological_redshift_distance(redshift: f64, hubble: f64) -> f64 {
    assert!(hubble > 0.0, "Hubble parameter must be positive");
    C * redshift / hubble
}

/// Luminosity distance (first-order expansion): d_L = (c/H₀) z (1 + z/2)
#[must_use]
pub fn luminosity_distance(redshift: f64, hubble: f64) -> f64 {
    assert!(hubble > 0.0, "Hubble parameter must be positive");
    (C / hubble) * redshift * (1.0 + redshift / 2.0)
}

/// Lookback time (matter-dominated approximation): t ≈ z / (H₀ (1+z))
#[must_use]
pub fn lookback_time(redshift: f64, hubble: f64) -> f64 {
    assert!(hubble > 0.0, "Hubble parameter must be positive");
    assert!((1.0 + redshift) != 0.0, "redshift must not equal -1");
    redshift / (hubble * (1.0 + redshift))
}

/// Scale factor from cosmological redshift: a = 1/(1+z)
#[must_use]
pub fn scale_factor_from_redshift(redshift: f64) -> f64 {
    assert!((1.0 + redshift) != 0.0, "redshift must not equal -1");
    1.0 / (1.0 + redshift)
}

/// CMB temperature at a given redshift: T = T₀ (1+z)
#[must_use]
pub fn temperature_at_redshift(t0: f64, redshift: f64) -> f64 {
    t0 * (1.0 + redshift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::{SOLAR_MASS, HUBBLE};

    fn rel_err(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn schwarzschild_radius_of_sun() {
        let rs = schwarzschild_radius(SOLAR_MASS);
        // Expected ≈ 2953 m (commonly cited ~2954 m depending on constants)
        assert!(rel_err(rs, 2954.0) < 0.01, "r_s = {rs}");
    }

    #[test]
    fn isco_is_three_times_rs() {
        let rs = schwarzschild_radius(SOLAR_MASS);
        let r_isco = isco_radius(SOLAR_MASS);
        assert!(rel_err(r_isco, 3.0 * rs) < 1e-12);
    }

    #[test]
    fn photon_sphere_is_1_5_rs() {
        let rs = schwarzschild_radius(SOLAR_MASS);
        let r_ph = photon_sphere_radius(SOLAR_MASS);
        assert!(rel_err(r_ph, 1.5 * rs) < 1e-12);
    }

    #[test]
    fn proper_time_at_infinity_is_one() {
        let factor = proper_time_factor(SOLAR_MASS, 1e20);
        assert!(rel_err(factor, 1.0) < 1e-6, "factor = {factor}");
    }

    #[test]
    fn redshift_factor_equal_radii_is_one() {
        let r = 1e7;
        let z = gravitational_redshift_factor(SOLAR_MASS, r, r);
        assert!(rel_err(z, 1.0) < 1e-12);
    }

    #[test]
    fn metric_consistency() {
        let mass = SOLAR_MASS;
        let r = 1e7;
        let gtt = schwarzschild_metric_tt(mass, r);
        let grr = schwarzschild_metric_rr(mass, r);
        // g_tt * g_rr should equal -1 for diagonal Schwarzschild metric
        assert!(rel_err(gtt * grr, -1.0) < 1e-12);
    }

    #[test]
    fn critical_density_with_hubble() {
        let rho_c = critical_density(HUBBLE);
        // ρ_c ≈ 9.5e-27 kg/m³ for H₀ ≈ 70 km/s/Mpc
        assert!(rho_c > 1e-28 && rho_c < 1e-25, "rho_c = {rho_c}");
    }

    #[test]
    fn scale_factor_at_z_zero_is_one() {
        assert!((scale_factor_from_redshift(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn kerr_reduces_to_schwarzschild_at_zero_spin() {
        let rs = schwarzschild_radius(SOLAR_MASS);
        let rk = kerr_event_horizon(SOLAR_MASS, 0.0);
        assert!(rel_err(rk, rs) < 1e-12);
    }

    #[test]
    fn kerr_isco_zero_spin_matches_schwarzschild() {
        let r_schwarz = isco_radius(SOLAR_MASS);
        let r_kerr = kerr_isco(SOLAR_MASS, 0.0, true);
        assert!(rel_err(r_kerr, r_schwarz) < 1e-10, "schwarz={r_schwarz}, kerr={r_kerr}");
    }

    #[test]
    fn kerr_isco_prograde_less_than_retrograde() {
        let pro = kerr_isco(SOLAR_MASS, 0.5, true);
        let retro = kerr_isco(SOLAR_MASS, 0.5, false);
        assert!(pro < retro, "prograde={pro}, retrograde={retro}");
    }

    #[test]
    fn temperature_scales_with_redshift() {
        let t0 = 2.725;
        let z = 1100.0;
        let t = temperature_at_redshift(t0, z);
        assert!(rel_err(t, 3000.225) < 1e-12);
    }

    #[test]
    fn event_horizon_is_schwarzschild_alias() {
        assert!((event_horizon_radius(SOLAR_MASS) - schwarzschild_radius(SOLAR_MASS)).abs() < 1e-15);
    }

    #[test]
    fn luminosity_distance_low_z() {
        let z = 0.01;
        let d_l = luminosity_distance(z, HUBBLE);
        let d_h = cosmological_redshift_distance(z, HUBBLE);
        // At very low z, d_L ≈ d_comoving
        assert!(rel_err(d_l, d_h) < 0.01);
    }

    #[test]
    fn circular_orbit_energy_at_isco() {
        // At ISCO (r = 6GM/c²), E/mc² = (1 - 2/6) / √(1 - 3/6) = (2/3)/√(1/2) = 2√2/3 ≈ 0.9428
        let r = isco_radius(SOLAR_MASS);
        let e = circular_orbit_energy(SOLAR_MASS, r);
        assert!(rel_err(e, 0.942_809_041_582_063_4) < 1e-6, "E={e}");
    }

    #[test]
    fn circular_orbit_angular_momentum_positive() {
        let r = 1e7; // well outside Schwarzschild radius
        let l = circular_orbit_angular_momentum(SOLAR_MASS, r);
        assert!(l > 0.0, "angular momentum should be positive, got {l}");
    }

    #[test]
    fn effective_potential_schwarzschild_newtonian_limit() {
        // At large r, the GR correction term is small; potential ≈ -GMm/r + l²/(2mr²)
        let mass = SOLAR_MASS;
        let r = 1e12;
        let l = 1e10;
        let m = 1.0;
        let v = effective_potential_schwarzschild(mass, r, l, m);
        assert!(rel_err(v, -1.327_518_27e8) < 1e-3, "v={v}");
    }

    #[test]
    fn frame_dragging_rate_positive_for_positive_spin() {
        let omega = frame_dragging_rate(SOLAR_MASS, 1000.0, 1e7);
        assert!(omega > 0.0, "frame dragging should be positive for positive spin");
    }

    #[test]
    fn frame_dragging_rate_falls_off_as_r_cubed() {
        let r1 = 1e7;
        let r2 = 2e7;
        let omega1 = frame_dragging_rate(SOLAR_MASS, 1000.0, r1);
        let omega2 = frame_dragging_rate(SOLAR_MASS, 1000.0, r2);
        let ratio = omega1 / omega2;
        assert!(rel_err(ratio, 8.0) < 1e-10, "should fall as r^3, ratio={ratio}");
    }

    #[test]
    fn friedmann_hubble_from_critical_density() {
        // H = √(8πGρ/3). If we feed in the critical density for H₀, we should recover H₀.
        let rho_c = critical_density(HUBBLE);
        let h_recovered = friedmann_hubble(rho_c, 0.0, 0.0);
        assert!(rel_err(h_recovered, HUBBLE) < 1e-6, "recovered H={h_recovered}, expected {HUBBLE}");
    }

    #[test]
    fn geodesic_acceleration_schwarzschild_zero_angular_momentum() {
        // With l=0, a = -GM/r²
        let mass = SOLAR_MASS;
        let r = 1e7;
        let a = geodesic_acceleration_schwarzschild(mass, r, 0.0, 0.0);
        assert!(rel_err(a, -1.327_518_27e6) < 1e-10, "a={a}");
    }

    #[test]
    fn kerr_ergosphere_at_equator_exceeds_horizon() {
        let spin = 1000.0;
        let r_ergo = kerr_ergosphere_radius(SOLAR_MASS, spin, PI / 2.0);
        let r_horizon = kerr_event_horizon(SOLAR_MASS, spin);
        assert!(
            r_ergo >= r_horizon,
            "ergosphere at equator should be >= horizon: ergo={r_ergo}, horizon={r_horizon}"
        );
    }

    #[test]
    fn kerr_ergosphere_at_pole_equals_horizon() {
        let spin = 500.0;
        let r_ergo = kerr_ergosphere_radius(SOLAR_MASS, spin, 0.0);
        let r_horizon = kerr_event_horizon(SOLAR_MASS, spin);
        assert!(rel_err(r_ergo, r_horizon) < 1e-10,
            "ergosphere at pole should equal horizon: ergo={r_ergo}, horizon={r_horizon}");
    }

    #[test]
    fn lookback_time_positive_and_bounded() {
        let z = 1.0;
        let t = lookback_time(z, HUBBLE);
        assert!(t > 0.0, "lookback time should be positive");
        // t = z / (H₀(1+z)) = 1/(2 H₀) ≈ 2.222e17 s ≈ 7 Gyr
        assert!(rel_err(t, 2.222_222_222_222_222e17) < 1e-10);
    }
}
