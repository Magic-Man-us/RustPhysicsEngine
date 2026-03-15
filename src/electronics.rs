use crate::math::constants::{E_CHARGE, K_B};

// ---------------------------------------------------------------------------
// Semiconductor basics
// ---------------------------------------------------------------------------

/// Intrinsic carrier concentration: ni = sqrt(Nc * Nv) * exp(-Eg / (2kT))
#[must_use]
pub fn intrinsic_carrier_concentration(
    nc: f64,
    nv: f64,
    band_gap: f64,
    temperature: f64,
) -> f64 {
    (nc * nv).sqrt() * (-band_gap / (2.0 * K_B * temperature)).exp()
}

/// Fermi-Dirac distribution: f(E) = 1 / (1 + exp((E - Ef) / (kT)))
#[must_use]
pub fn fermi_dirac(energy: f64, fermi_level: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + ((energy - fermi_level) / (K_B * temperature)).exp())
}

/// Thermal voltage: Vt = kT / q
#[must_use]
pub fn thermal_voltage(temperature: f64) -> f64 {
    K_B * temperature / E_CHARGE
}

/// Electrical conductivity: sigma = n * q * mu
#[must_use]
pub fn conductivity(carrier_density: f64, mobility: f64, charge: f64) -> f64 {
    carrier_density * charge * mobility
}

/// Resistivity: rho = 1 / sigma
#[must_use]
pub fn resistivity(conductivity: f64) -> f64 {
    1.0 / conductivity
}

/// Drift velocity: vd = mu * E
#[must_use]
pub fn drift_velocity(mobility: f64, electric_field: f64) -> f64 {
    mobility * electric_field
}

/// Einstein relation for diffusion coefficient: D = mu * kT / q
#[must_use]
pub fn diffusion_coefficient_einstein(mobility: f64, temperature: f64) -> f64 {
    mobility * K_B * temperature / E_CHARGE
}

/// Built-in potential of a p-n junction: Vbi = (kT/q) * ln(Na * Nd / ni^2)
#[must_use]
pub fn built_in_potential(na: f64, nd: f64, ni: f64, temperature: f64) -> f64 {
    thermal_voltage(temperature) * (na * nd / (ni * ni)).ln()
}

/// Depletion region width: W = sqrt(2 * epsilon * Vbi * (1/Na + 1/Nd) / q)
#[must_use]
pub fn depletion_width(
    epsilon: f64,
    vbi: f64,
    na: f64,
    nd: f64,
    charge: f64,
) -> f64 {
    (2.0 * epsilon * vbi * (1.0 / na + 1.0 / nd) / charge).sqrt()
}

// ---------------------------------------------------------------------------
// Diode
// ---------------------------------------------------------------------------

/// Shockley diode equation: I = Is * (exp(V / (n * Vt)) - 1)
#[must_use]
pub fn diode_current(is: f64, voltage: f64, temperature: f64, n: f64) -> f64 {
    let vt = thermal_voltage(temperature);
    is * ((voltage / (n * vt)).exp() - 1.0)
}

/// Reverse saturation current (symmetric approximation):
/// Is = q * A * ni^2 * (Dn/Ln + Dp/Lp)
#[must_use]
pub fn diode_reverse_saturation(
    area: f64,
    ni: f64,
    dn: f64,
    dp: f64,
    ln: f64,
    lp: f64,
    charge: f64,
) -> f64 {
    charge * area * ni * ni * (dn / ln + dp / lp)
}

// ---------------------------------------------------------------------------
// MOSFET (simplified, long-channel)
// ---------------------------------------------------------------------------

/// MOSFET drain current in the linear region:
/// Id = mu * Cox * (W/L) * ((Vgs - Vth) * Vds - Vds^2 / 2)
#[must_use]
pub fn mosfet_drain_current_linear(
    mu: f64,
    cox: f64,
    w: f64,
    l: f64,
    vgs: f64,
    vth: f64,
    vds: f64,
) -> f64 {
    mu * cox * (w / l) * ((vgs - vth) * vds - vds * vds / 2.0)
}

/// MOSFET drain current in the saturation region:
/// Id = (mu * Cox / 2) * (W/L) * (Vgs - Vth)^2
#[must_use]
pub fn mosfet_drain_current_saturation(
    mu: f64,
    cox: f64,
    w: f64,
    l: f64,
    vgs: f64,
    vth: f64,
) -> f64 {
    let vov = vgs - vth;
    (mu * cox / 2.0) * (w / l) * vov * vov
}

// ---------------------------------------------------------------------------
// Photovoltaic
// ---------------------------------------------------------------------------

/// Solar cell current: I = Iph - I0 * (exp(V / Vt) - 1)
#[must_use]
pub fn solar_cell_current(
    photocurrent: f64,
    dark_current: f64,
    voltage: f64,
    temperature: f64,
) -> f64 {
    let vt = thermal_voltage(temperature);
    photocurrent - dark_current * ((voltage / vt).exp() - 1.0)
}

/// Open-circuit voltage: Voc = Vt * ln(Iph / I0 + 1)
#[must_use]
pub fn open_circuit_voltage(
    photocurrent: f64,
    dark_current: f64,
    temperature: f64,
) -> f64 {
    let vt = thermal_voltage(temperature);
    vt * (photocurrent / dark_current + 1.0).ln()
}

/// Fill factor: FF = Pmax / (Voc * Isc)
#[must_use]
pub fn fill_factor(voc: f64, isc: f64, pmax: f64) -> f64 {
    pmax / (voc * isc)
}

/// Solar cell efficiency: eta = Pmax / Pin
#[must_use]
pub fn solar_cell_efficiency(pmax: f64, incident_power: f64) -> f64 {
    pmax / incident_power
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool {
        if b.abs() < 1e-30 {
            (a - b).abs() < 1e-30
        } else {
            ((a - b) / b).abs() < TOLERANCE
        }
    }

    // -- Semiconductor basics --

    #[test]
    fn test_thermal_voltage_at_300k() {
        let vt = thermal_voltage(300.0);
        let expected = K_B * 300.0 / E_CHARGE;
        assert!(approx(vt, expected), "Vt={vt}, expected={expected}");
    }

    #[test]
    fn test_fermi_dirac_at_fermi_level() {
        let f = fermi_dirac(1.0, 1.0, 300.0);
        assert!(approx(f, 0.5), "f(Ef)={f}, expected 0.5");
    }

    #[test]
    fn test_fermi_dirac_well_below() {
        let f = fermi_dirac(0.0, 1.0, 300.0);
        assert!(f > 0.99, "f(E << Ef) should be ~1.0, got {f}");
    }

    #[test]
    fn test_fermi_dirac_well_above() {
        let f = fermi_dirac(2.0, 1.0, 300.0);
        assert!(f < 0.01, "f(E >> Ef) should be ~0.0, got {f}");
    }

    #[test]
    fn test_intrinsic_carrier_concentration() {
        let nc = 2.8e25;
        let nv = 1.04e25;
        let eg = 1.12 * E_CHARGE; // silicon band gap in joules
        let t = 300.0;
        let ni = intrinsic_carrier_concentration(nc, nv, eg, t);
        // ni for silicon at 300K is approximately 1.5e16 m^-3 (1.5e10 cm^-3)
        assert!(ni > 1e15 && ni < 1e17, "ni={ni:.3e}, expected ~1.5e16");
    }

    #[test]
    fn test_conductivity_and_resistivity() {
        let n = 1e22; // carrier density (m^-3)
        let mu = 0.14; // mobility (m^2/(V·s))
        let q = E_CHARGE;
        let sigma = conductivity(n, mu, q);
        let rho = resistivity(sigma);
        assert!(approx(sigma, n * q * mu));
        assert!(approx(rho, 1.0 / sigma));
    }

    #[test]
    fn test_drift_velocity() {
        let mu = 0.14;
        let e_field = 1000.0;
        let vd = drift_velocity(mu, e_field);
        assert!(approx(vd, 140.0));
    }

    #[test]
    fn test_diffusion_coefficient_einstein() {
        let mu = 0.14;
        let t = 300.0;
        let d = diffusion_coefficient_einstein(mu, t);
        let expected = mu * K_B * t / E_CHARGE;
        assert!(approx(d, expected));
    }

    #[test]
    fn test_built_in_potential() {
        let na = 1e23; // m^-3
        let nd = 1e21; // m^-3
        let ni = 1.5e16; // m^-3 (silicon)
        let t = 300.0;
        let vbi = built_in_potential(na, nd, ni, t);
        // Vbi for typical silicon junction is ~0.6-0.8 V
        assert!(vbi > 0.5 && vbi < 1.0, "Vbi={vbi}, expected 0.5-1.0 V");
    }

    #[test]
    fn test_depletion_width() {
        let epsilon = 11.7 * 8.854e-12; // silicon permittivity
        let vbi = 0.7;
        let na = 1e23;
        let nd = 1e21;
        let q = E_CHARGE;
        let w = depletion_width(epsilon, vbi, na, nd, q);
        // Depletion width is typically on the order of micrometers
        assert!(w > 1e-8 && w < 1e-4, "W={w:.3e}, expected ~um range");
    }

    // -- Diode --

    #[test]
    fn test_diode_current_forward_bias() {
        let is = 1e-12;
        let v = 0.6;
        let t = 300.0;
        let n = 1.0;
        let i = diode_current(is, v, t, n);
        assert!(i > 0.0, "Forward-biased diode should have positive current");
        // At 0.6V forward bias, current should be significant
        assert!(i > 1e-3, "I={i:.3e}, expected > 1mA at 0.6V");
    }

    #[test]
    fn test_diode_current_reverse_bias() {
        let is = 1e-12;
        let v = -1.0;
        let t = 300.0;
        let n = 1.0;
        let i = diode_current(is, v, t, n);
        // Reverse bias current approaches -Is
        assert!(approx(i, -is), "I={i:.3e}, expected ~{is:.3e}");
    }

    #[test]
    fn test_diode_current_zero_bias() {
        let is = 1e-12;
        let i = diode_current(is, 0.0, 300.0, 1.0);
        assert!(i.abs() < 1e-20, "Zero bias should give zero current, got {i}");
    }

    #[test]
    fn test_diode_reverse_saturation() {
        let area = 1e-8; // m^2
        let ni = 1.5e16;
        let dn = 3.5e-3; // m^2/s
        let dp = 1.2e-3;
        let ln = 5e-6; // m
        let lp = 3e-6;
        let q = E_CHARGE;
        let is = diode_reverse_saturation(area, ni, dn, dp, ln, lp, q);
        let expected = q * area * ni * ni * (dn / ln + dp / lp);
        assert!(approx(is, expected));
        assert!(is > 0.0);
    }

    // -- MOSFET --

    #[test]
    fn test_mosfet_linear_region() {
        let mu = 0.04; // m^2/(V·s)
        let cox = 3.45e-3; // F/m^2
        let w = 10e-6;
        let l = 1e-6;
        let vgs = 2.0;
        let vth = 0.7;
        let vds = 0.1;
        let id = mosfet_drain_current_linear(mu, cox, w, l, vgs, vth, vds);
        let expected = mu * cox * (w / l) * ((vgs - vth) * vds - vds * vds / 2.0);
        assert!(approx(id, expected));
        assert!(id > 0.0);
    }

    #[test]
    fn test_mosfet_saturation_region() {
        let mu = 0.04;
        let cox = 3.45e-3;
        let w = 10e-6;
        let l = 1e-6;
        let vgs = 2.0;
        let vth = 0.7;
        let id = mosfet_drain_current_saturation(mu, cox, w, l, vgs, vth);
        let vov = vgs - vth;
        let expected = (mu * cox / 2.0) * (w / l) * vov * vov;
        assert!(approx(id, expected));
    }

    #[test]
    fn test_mosfet_saturation_greater_than_linear() {
        let mu = 0.04;
        let cox = 3.45e-3;
        let w = 10e-6;
        let l = 1e-6;
        let vgs = 2.0;
        let vth = 0.7;
        let vds_sat = vgs - vth; // boundary
        let id_sat = mosfet_drain_current_saturation(mu, cox, w, l, vgs, vth);
        let id_lin = mosfet_drain_current_linear(mu, cox, w, l, vgs, vth, vds_sat);
        // At the boundary Vds = Vgs - Vth, linear and saturation should match
        assert!(approx(id_sat, id_lin), "sat={id_sat}, lin={id_lin}");
    }

    // -- Photovoltaic --

    #[test]
    fn test_solar_cell_current_short_circuit() {
        let iph = 0.035; // 35 mA
        let i0 = 1e-10;
        let t = 300.0;
        let i = solar_cell_current(iph, i0, 0.0, t);
        // At V=0, I = Iph (short circuit)
        assert!(approx(i, iph), "Isc={i}, expected {iph}");
    }

    #[test]
    fn test_open_circuit_voltage() {
        let iph = 0.035;
        let i0 = 1e-10;
        let t = 300.0;
        let voc = open_circuit_voltage(iph, i0, t);
        // Voc should be positive and on the order of ~0.5-0.7 V for typical values
        assert!(voc > 0.4 && voc < 0.8, "Voc={voc}, expected 0.4-0.8 V");
        // Verify that current at Voc is approximately zero
        let i_at_voc = solar_cell_current(iph, i0, voc, t);
        assert!(i_at_voc.abs() < 1e-10, "I(Voc)={i_at_voc}, expected ~0");
    }

    #[test]
    fn test_fill_factor() {
        let voc = 0.6;
        let isc = 0.035;
        let pmax = 0.015;
        let ff = fill_factor(voc, isc, pmax);
        let expected = pmax / (voc * isc);
        assert!(approx(ff, expected));
        assert!(ff > 0.0 && ff < 1.0);
    }

    #[test]
    fn test_solar_cell_efficiency() {
        let pmax = 0.015;
        let pin = 0.1;
        let eta = solar_cell_efficiency(pmax, pin);
        assert!(approx(eta, 0.15));
    }
}
