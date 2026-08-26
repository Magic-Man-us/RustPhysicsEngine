//! The solid Earth: gravity, seismology, and heat.
//!
//! Gravity surveying -- the latitude formula, the free-air and Bouguer
//! corrections, the resulting anomaly, and Airy isostatic compensation.
//!
//! Seismology: P- and S-wave travel times, epicentral distance from the
//! S−P lag, the Richter and moment magnitude scales, and seismic moment
//! and energy. The moment magnitude is the one to use for large events,
//! because Richter saturates.
//!
//! Heat flow: pressure and temperature with depth, the geothermal
//! gradient, and geothermal power. Plate tectonics closes the module with
//! Euler-pole plate velocities and the square-root-of-age law for
//! seafloor depth.

use crate::math::constants;

// ── Gravity & Geoid constants ────────────────────────────────────────

/// Equatorial gravity (m/s²) — WGS84 / International gravity formula
const G_EQUATORIAL: f64 = 9.780_327;

/// Coefficient of sin²φ in the International gravity formula
const IGF_SIN2_COEFF: f64 = 0.005_302_4;

/// Coefficient of sin²2φ in the International gravity formula
const IGF_SIN2_2PHI_COEFF: f64 = 0.000_005_8;

/// Free-air correction factor (mGal/m)
const FREE_AIR_FACTOR: f64 = -0.308_6;

// ── Earth structure constants ────────────────────────────────────────

/// Average continental Moho depth (m)
const MOHO_CONTINENTAL_M: f64 = 35_000.0;

/// Average oceanic Moho depth (m)
const MOHO_OCEANIC_M: f64 = 7_000.0;

/// Core-mantle boundary depth (m)
const CMB_DEPTH_M: f64 = 2_891_000.0;

// ── Seismology constants ─────────────────────────────────────────────

/// Moment magnitude offset for M₀ in N·m
const MW_OFFSET_NM: f64 = 6.07;

/// Gutenberg-Richter energy constant (exponent offset)
const GR_ENERGY_OFFSET: f64 = 4.8;

/// Richter magnitude distance coefficient
const RICHTER_DIST_COEFF: f64 = 2.76;

/// Richter magnitude constant offset
const RICHTER_OFFSET: f64 = 2.48;

// ── Plate tectonics constants ────────────────────────────────────────

/// Parsons-Sclater ridge depth (m)
const RIDGE_DEPTH_M: f64 = 2_500.0;

/// Parsons-Sclater subsidence coefficient (m/Myr^0.5)
const SUBSIDENCE_COEFF: f64 = 350.0;

// ── Gravity & Geoid ─────────────────────────────────────────────────

/// International gravity formula: g = 9.780327(1 + 0.0053024 sin²φ − 0.0000058 sin²2φ).
/// `latitude_rad` is the geodetic latitude in radians.
#[must_use]
pub fn gravity_at_latitude(latitude_rad: f64) -> f64 {
    let sin_phi = latitude_rad.sin();
    let sin2_phi = sin_phi * sin_phi;
    let sin_2phi = (2.0 * latitude_rad).sin();
    let sin2_2phi = sin_2phi * sin_2phi;
    G_EQUATORIAL * (1.0 + IGF_SIN2_COEFF * sin2_phi - IGF_SIN2_2PHI_COEFF * sin2_2phi)
}

/// Free-air gravity correction: Δg = −0.3086 × h (mGal).
/// `height` is meters above the geoid.
#[must_use]
pub fn free_air_correction(height: f64) -> f64 {
    FREE_AIR_FACTOR * height
}

/// Bouguer slab correction: Δg = 2πGρh.
/// Accounts for the gravitational attraction of material between the
/// station and the geoid. Returns the correction in m/s².
#[must_use]
pub fn bouguer_correction(height: f64, density: f64) -> f64 {
    2.0 * constants::PI * constants::G * density * height
}

/// Bouguer anomaly: Δg_B = g_obs − g_lat + free_air − bouguer.
/// All values in consistent units (typically mGal or m/s²).
#[must_use]
pub fn bouguer_anomaly(
    observed_g: f64,
    latitude_g: f64,
    free_air: f64,
    bouguer: f64,
) -> f64 {
    observed_g - latitude_g + free_air - bouguer
}

/// Airy isostatic compensation depth: d = elevation × ρ_c / (ρ_m − ρ_c).
/// Returns the depth of the crustal root below normal Moho.
#[must_use]
pub fn isostatic_compensation_depth(
    elevation: f64,
    crust_density: f64,
    mantle_density: f64,
) -> f64 {
    assert!((mantle_density - crust_density) != 0.0, "mantle and crust densities must differ");
    elevation * crust_density / (mantle_density - crust_density)
}

// ── Seismology ──────────────────────────────────────────────────────

/// P-wave travel time: t = d / v.
#[must_use]
pub fn p_wave_travel_time(distance: f64, velocity: f64) -> f64 {
    assert!(velocity > 0.0, "velocity must be positive");
    distance / velocity
}

/// S-wave travel time: t = d / v.
#[must_use]
pub fn s_wave_travel_time(distance: f64, velocity: f64) -> f64 {
    assert!(velocity > 0.0, "velocity must be positive");
    distance / velocity
}

/// Epicentral distance from S−P lag time: d = Δt × vp × vs / (vp − vs).
#[must_use]
pub fn epicentral_distance_from_lag(t_s_minus_p: f64, vp: f64, vs: f64) -> f64 {
    assert!((vp - vs) != 0.0, "P-wave and S-wave velocities must differ");
    t_s_minus_p * vp * vs / (vp - vs)
}

/// Simplified Richter local magnitude: M_L = log₁₀(A) + 2.76 log₁₀(Δ) − 2.48.
/// `amplitude` is the maximum trace amplitude in mm, `distance_km` in km.
#[must_use]
pub fn richter_magnitude(amplitude: f64, distance_km: f64) -> f64 {
    assert!(amplitude > 0.0, "amplitude must be positive");
    assert!(distance_km > 0.0, "distance_km must be positive");
    amplitude.log10() + RICHTER_DIST_COEFF * distance_km.log10() - RICHTER_OFFSET
}

/// Moment magnitude from seismic moment (N·m): M_w = (2/3) log₁₀(M₀) − 6.07.
#[must_use]
pub fn moment_magnitude(seismic_moment: f64) -> f64 {
    assert!(seismic_moment > 0.0, "seismic_moment must be positive");
    (2.0 / 3.0) * seismic_moment.log10() - MW_OFFSET_NM
}

/// Seismic moment from moment magnitude (returns N·m): M₀ = 10^(1.5(M_w + 6.07)).
#[must_use]
pub fn seismic_moment(magnitude: f64) -> f64 {
    10.0_f64.powf(1.5 * (magnitude + MW_OFFSET_NM))
}

/// Seismic energy from magnitude (Gutenberg-Richter): E = 10^(1.5M + 4.8) joules.
#[must_use]
pub fn seismic_energy(magnitude: f64) -> f64 {
    10.0_f64.powf(1.5 * magnitude + GR_ENERGY_OFFSET)
}

// ── Earth Structure ─────────────────────────────────────────────────

/// Lithostatic pressure at depth (simplified): P ≈ ρgd.
/// `depth` in meters, `surface_density` in kg/m³, `g` in m/s².
#[must_use]
pub fn pressure_at_depth(depth: f64, surface_density: f64, g: f64) -> f64 {
    surface_density * g * depth
}

/// Temperature at depth assuming a linear geothermal gradient:
/// T = T₀ + (dT/dz) × z.
/// `geothermal_gradient` is in K/m (typical ~0.025–0.030 K/m).
#[must_use]
pub fn temperature_at_depth(surface_temp: f64, geothermal_gradient: f64, depth: f64) -> f64 {
    surface_temp + geothermal_gradient * depth
}

/// Average continental Moho depth: ~35 km.
#[must_use]
pub const fn moho_depth_continental() -> f64 {
    MOHO_CONTINENTAL_M
}

/// Average oceanic Moho depth: ~7 km.
#[must_use]
pub const fn moho_depth_oceanic() -> f64 {
    MOHO_OCEANIC_M
}

/// Core-mantle boundary depth: 2891 km.
#[must_use]
pub const fn core_mantle_boundary_depth() -> f64 {
    CMB_DEPTH_M
}

// ── Geothermal ──────────────────────────────────────────────────────

/// Fourier heat flow: q = k × dT/dz (W/m²).
/// `conductivity` in W/(m·K), `temperature_gradient` in K/m.
#[must_use]
pub fn heat_flow(conductivity: f64, temperature_gradient: f64) -> f64 {
    conductivity * temperature_gradient
}

/// Geothermal power extracted from a fluid: P = ṁ c ΔT.
/// `flow_rate` in kg/s, `specific_heat` in J/(kg·K), `delta_temp` in K.
#[must_use]
pub fn geothermal_power(flow_rate: f64, specific_heat: f64, delta_temp: f64) -> f64 {
    flow_rate * specific_heat * delta_temp
}

// ── Plate Tectonics ─────────────────────────────────────────────────

/// Plate velocity from an Euler pole: v = ω R sin(θ).
/// `omega` in rad/s, `radius` in meters, `angular_distance` in radians.
#[must_use]
pub fn plate_velocity_euler(omega: f64, radius: f64, angular_distance: f64) -> f64 {
    omega * radius * angular_distance.sin()
}

/// Age of seafloor from distance to ridge: t = d / (2v).
/// `distance` in meters, `spreading_rate` is the full rate in m/s.
/// Division by 2 accounts for the half-spreading rate.
#[must_use]
pub fn age_of_seafloor(distance: f64, spreading_rate: f64) -> f64 {
    assert!(spreading_rate > 0.0, "spreading_rate must be positive");
    distance / (2.0 * spreading_rate)
}

/// Ocean depth from seafloor age (Parsons-Sclater model):
/// d = 2500 + 350√t.
/// `age_myr` is in millions of years; returns depth in meters.
#[must_use]
pub fn ocean_depth_from_age(age_myr: f64) -> f64 {
    RIDGE_DEPTH_M + SUBSIDENCE_COEFF * age_myr.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn rel_error(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    // ── Gravity tests ────────────────────────────────────────────────

    #[test]
    fn test_gravity_at_equator() {
        let g = gravity_at_latitude(0.0);
        assert!(
            approx_eq(g, 9.780_327, 0.001),
            "equatorial gravity should be ~9.780 m/s², got {g}"
        );
    }

    #[test]
    fn test_gravity_at_pole() {
        let g = gravity_at_latitude(constants::PI / 2.0);
        assert!(
            approx_eq(g, 9.832, 0.002),
            "polar gravity should be ~9.832 m/s², got {g}"
        );
    }

    #[test]
    fn test_gravity_increases_with_latitude() {
        let g_eq = gravity_at_latitude(0.0);
        let g_45 = gravity_at_latitude(constants::PI / 4.0);
        let g_pole = gravity_at_latitude(constants::PI / 2.0);
        assert!(g_45 > g_eq, "gravity at 45° should exceed equatorial");
        assert!(g_pole > g_45, "polar gravity should exceed 45°");
    }

    #[test]
    fn test_free_air_correction() {
        let correction = free_air_correction(100.0);
        assert!(
            approx_eq(correction, -30.86, 0.01),
            "free-air correction at 100 m should be ~-30.86 mGal, got {correction}"
        );
    }

    #[test]
    fn test_bouguer_correction_positive() {
        let bc = bouguer_correction(100.0, 2670.0);
        assert!(bc > 0.0, "Bouguer correction should be positive for positive height/density");
    }

    #[test]
    fn test_bouguer_anomaly_identity() {
        let anomaly = bouguer_anomaly(10.0, 10.0, 5.0, 5.0);
        assert!(
            approx_eq(anomaly, 0.0, 1e-10),
            "anomaly should cancel when corrections equal difference"
        );
    }

    #[test]
    fn test_isostatic_compensation() {
        let depth = isostatic_compensation_depth(1000.0, 2700.0, 3300.0);
        assert!(
            approx_eq(depth, 4500.0, 1.0),
            "1 km elevation with typical densities → ~4.5 km root, got {depth}"
        );
    }

    // ── Seismology tests ─────────────────────────────────────────────

    #[test]
    fn test_wave_travel_times() {
        let tp = p_wave_travel_time(100_000.0, 5000.0);
        let ts = s_wave_travel_time(100_000.0, 3000.0);
        assert!(approx_eq(tp, 20.0, 1e-10));
        assert!(approx_eq(ts, 100_000.0 / 3000.0, 1e-10));
        assert!(ts > tp, "S-wave should arrive after P-wave");
    }

    #[test]
    fn test_epicentral_distance_from_lag() {
        let vp = 6000.0;
        let vs = 3500.0;
        let dist = 100_000.0;
        let lag = dist / vs - dist / vp;
        let recovered = epicentral_distance_from_lag(lag, vp, vs);
        assert!(
            rel_error(recovered, dist) < 1e-10,
            "round-trip epicentral distance should match, got {recovered}"
        );
    }

    #[test]
    fn test_moment_magnitude_roundtrip() {
        let mw = 9.0;
        let m0 = seismic_moment(mw);
        let mw_back = moment_magnitude(m0);
        assert!(
            approx_eq(mw_back, mw, 1e-10),
            "M_w 9.0 → M₀ → M_w should round-trip, got {mw_back}"
        );
    }

    #[test]
    fn test_seismic_moment_magnitude_9() {
        let m0 = seismic_moment(9.0);
        assert!(
            m0 > 1e22,
            "M₀ for M_w 9.0 should be > 10²² N·m, got {m0}"
        );
    }

    #[test]
    fn test_seismic_energy_increases_with_magnitude() {
        let e5 = seismic_energy(5.0);
        let e6 = seismic_energy(6.0);
        let e7 = seismic_energy(7.0);
        assert!(e6 > e5);
        assert!(e7 > e6);
        let ratio = e6 / e5;
        assert!(
            approx_eq(ratio, 10.0_f64.powf(1.5), 1.0),
            "one magnitude step ≈ 31.6× energy, got {ratio}"
        );
    }

    #[test]
    fn test_richter_magnitude_positive() {
        let ml = richter_magnitude(10.0, 100.0);
        assert!(ml > 0.0, "Richter magnitude should be positive for reasonable inputs");
    }

    // ── Earth structure tests ────────────────────────────────────────

    #[test]
    fn test_moho_depths_reasonable() {
        let continental = moho_depth_continental();
        let oceanic = moho_depth_oceanic();
        assert!(
            continental > oceanic,
            "continental Moho should be deeper than oceanic"
        );
        assert!(
            approx_eq(continental, 35_000.0, 1.0),
            "continental Moho ~35 km"
        );
        assert!(approx_eq(oceanic, 7_000.0, 1.0), "oceanic Moho ~7 km");
    }

    #[test]
    fn test_cmb_depth() {
        let cmb = core_mantle_boundary_depth();
        assert!(
            approx_eq(cmb, 2_891_000.0, 1.0),
            "CMB depth should be 2891 km"
        );
    }

    #[test]
    fn test_pressure_at_depth() {
        let p = pressure_at_depth(10_000.0, 2700.0, 9.81);
        let expected = 2.6487e8;
        assert!(approx_eq(p, expected, 1.0));
    }

    #[test]
    fn test_temperature_at_depth() {
        let t = temperature_at_depth(288.0, 0.025, 1_000.0);
        assert!(
            approx_eq(t, 313.0, 0.01),
            "1 km depth with 25°C/km gradient → ~313 K, got {t}"
        );
    }

    // ── Geothermal tests ─────────────────────────────────────────────

    #[test]
    fn test_heat_flow() {
        let q = heat_flow(3.0, 0.030);
        assert!(
            approx_eq(q, 0.09, 1e-10),
            "k=3 W/(m·K), grad=30°C/km → q=0.09 W/m²"
        );
    }

    #[test]
    fn test_geothermal_power() {
        let p = geothermal_power(50.0, 4186.0, 80.0);
        let expected = 1.6744e7;
        assert!(approx_eq(p, expected, 1.0));
    }

    // ── Plate tectonics tests ────────────────────────────────────────

    #[test]
    fn test_plate_velocity_euler() {
        let v = plate_velocity_euler(1e-9, constants::EARTH_RADIUS, constants::PI / 2.0);
        let expected = 0.006371;
        assert!(
            rel_error(v, expected) < 1e-10,
            "at 90° angular distance, sin(θ)=1, so v=ωR"
        );
    }

    #[test]
    fn test_age_of_seafloor() {
        let rate = 0.04; // 4 cm/yr ~ 1.27e-9 m/s full rate
        let distance = 1000.0;
        let age = age_of_seafloor(distance, rate);
        assert!(
            approx_eq(age, 12_500.0, 1e-6),
            "1000 m at 4 cm/yr full rate → 12500 time units"
        );
    }

    #[test]
    fn test_ocean_depth_from_age_at_ridge() {
        let d = ocean_depth_from_age(0.0);
        assert!(
            approx_eq(d, 2_500.0, 1e-10),
            "zero-age seafloor should be at ridge depth 2500 m"
        );
    }

    #[test]
    fn test_ocean_depth_increases_with_age() {
        let d10 = ocean_depth_from_age(10.0);
        let d50 = ocean_depth_from_age(50.0);
        let d100 = ocean_depth_from_age(100.0);
        assert!(d50 > d10);
        assert!(d100 > d50);
    }
}
