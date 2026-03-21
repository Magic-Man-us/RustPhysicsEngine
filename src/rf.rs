use crate::math::constants;

// ── EM Spectrum Classification ──

/// Wavelength to frequency: f = c / λ
pub fn wavelength_to_frequency(wavelength: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    constants::C / wavelength
}

/// Frequency to wavelength: λ = c / f
pub fn frequency_to_wavelength(frequency: f64) -> f64 {
    assert!(frequency > 0.0, "frequency must be positive");
    constants::C / frequency
}

/// Photon energy from frequency: E = hf
pub fn frequency_to_energy(frequency: f64) -> f64 {
    constants::H * frequency
}

// ── RF Propagation ──

/// Free-space path loss in dB: FSPL = 20log₁₀(d) + 20log₁₀(f) + 20log₁₀(4π/c)
pub fn free_space_path_loss(distance: f64, frequency: f64) -> f64 {
    20.0 * distance.log10()
        + 20.0 * frequency.log10()
        + 20.0 * (4.0 * constants::PI / constants::C).log10()
}

/// Friis transmission equation (linear): Pr = Pt × Gt × Gr × (λ/(4πd))²
pub fn friis_received_power(
    pt: f64,
    gt: f64,
    gr: f64,
    wavelength: f64,
    distance: f64,
) -> f64 {
    assert!(distance > 0.0, "distance must be positive");
    let ratio = wavelength / (4.0 * constants::PI * distance);
    pt * gt * gr * ratio * ratio
}

/// Link budget in dB: Pr = Pt + Gt + Gr - PathLoss
pub fn link_budget_db(pt_dbm: f64, gt_dbi: f64, gr_dbi: f64, path_loss_db: f64) -> f64 {
    pt_dbm + gt_dbi + gr_dbi - path_loss_db
}

/// Skin depth in a conductor: δ = 1 / √(πfμσ)
pub fn skin_depth_conductor(frequency: f64, permeability: f64, conductivity: f64) -> f64 {
    assert!(frequency > 0.0, "frequency must be positive");
    assert!(permeability > 0.0, "permeability must be positive");
    assert!(conductivity > 0.0, "conductivity must be positive");
    1.0 / (constants::PI * frequency * permeability * conductivity).sqrt()
}

/// Fade margin: FM = received_dBm - sensitivity_dBm
pub fn fade_margin_db(
    _transmitted_dbm: f64,
    received_dbm: f64,
    sensitivity_dbm: f64,
) -> f64 {
    received_dbm - sensitivity_dbm
}

// ── Antenna Theory ──

/// Antenna gain from effective area: G = 4πA_e / λ²
pub fn antenna_gain_from_area(effective_area: f64, wavelength: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    4.0 * constants::PI * effective_area / (wavelength * wavelength)
}

/// Effective aperture from gain: A_e = Gλ² / (4π)
pub fn effective_area_from_gain(gain: f64, wavelength: f64) -> f64 {
    gain * wavelength * wavelength / (4.0 * constants::PI)
}

/// Half-wave dipole gain (linear): G ≈ 1.64 (2.15 dBi)
const HALF_WAVE_DIPOLE_GAIN: f64 = 1.64;

/// Returns the half-wave dipole gain (linear): G ≈ 1.64 (2.15 dBi).
pub fn half_wave_dipole_gain() -> f64 {
    HALF_WAVE_DIPOLE_GAIN
}

/// Effective isotropic radiated power: EIRP = P × G
pub fn eirp(power: f64, gain: f64) -> f64 {
    power * gain
}

/// Approximate antenna beamwidth in degrees: θ ≈ 70λ / D
const BEAMWIDTH_FACTOR: f64 = 70.0;

/// Approximate antenna beamwidth in degrees: θ ≈ 70λ / D.
pub fn beamwidth_approximate(wavelength: f64, aperture: f64) -> f64 {
    assert!(aperture > 0.0, "aperture must be positive");
    BEAMWIDTH_FACTOR * wavelength / aperture
}

/// Antenna directivity from gain and efficiency: D = G / η
pub fn antenna_directivity(gain: f64, efficiency: f64) -> f64 {
    assert!(efficiency > 0.0, "efficiency must be positive");
    gain / efficiency
}

// ── Transmission Lines ──

/// Characteristic impedance of coaxial cable: Z₀ = (138/√εr) × log₁₀(D/d)
const COAX_IMPEDANCE_FACTOR: f64 = 138.0;

/// Characteristic impedance of coaxial cable: Z₀ = (138/√εr) × log₁₀(D/d).
pub fn characteristic_impedance_coax(
    outer_radius: f64,
    inner_radius: f64,
    permittivity_rel: f64,
) -> f64 {
    assert!(inner_radius > 0.0, "inner_radius must be positive");
    assert!(permittivity_rel > 0.0, "permittivity_rel must be positive");
    (COAX_IMPEDANCE_FACTOR / permittivity_rel.sqrt())
        * (outer_radius / inner_radius).log10()
}

/// Velocity factor: VF = 1 / √εr
pub fn velocity_factor(permittivity_rel: f64) -> f64 {
    assert!(permittivity_rel > 0.0, "permittivity_rel must be positive");
    1.0 / permittivity_rel.sqrt()
}

/// Wavelength in a transmission line: λ_line = λ₀ × VF
pub fn wavelength_in_line(free_space_wavelength: f64, velocity_factor: f64) -> f64 {
    free_space_wavelength * velocity_factor
}

/// Voltage standing wave ratio: VSWR = (1 + |Γ|) / (1 - |Γ|)
pub fn vswr(reflection_coeff: f64) -> f64 {
    let gamma = reflection_coeff.abs();
    assert!(gamma < 1.0, "reflection coefficient magnitude must be less than 1");
    (1.0 + gamma) / (1.0 - gamma)
}

/// Return loss in dB: RL = -20log₁₀(|Γ|)
pub fn return_loss(reflection_coeff: f64) -> f64 {
    -20.0 * reflection_coeff.abs().log10()
}

/// Mismatch loss in dB: ML = -10log₁₀(1 - ((VSWR-1)/(VSWR+1))²)
pub fn mismatch_loss(vswr: f64) -> f64 {
    let gamma = (vswr - 1.0) / (vswr + 1.0);
    -10.0 * (1.0 - gamma * gamma).log10()
}

// ── Signal & Modulation Basics ──

/// Convert dBm to watts: P = 10^((dBm - 30) / 10)
pub fn dbm_to_watts(dbm: f64) -> f64 {
    10.0_f64.powf((dbm - 30.0) / 10.0)
}

/// Convert watts to dBm: dBm = 10log₁₀(P) + 30
pub fn watts_to_dbm(watts: f64) -> f64 {
    10.0 * watts.log10() + 30.0
}

/// Convert dB to linear ratio: ratio = 10^(dB / 10)
pub fn db_to_ratio(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

/// Convert linear ratio to dB: dB = 10log₁₀(ratio)
pub fn ratio_to_db(ratio: f64) -> f64 {
    10.0 * ratio.log10()
}

/// Thermal noise power: N = k_B × T × B
pub fn noise_power(bandwidth: f64, temperature: f64) -> f64 {
    constants::K_B * temperature * bandwidth
}

/// Signal-to-noise ratio in dB: SNR = 10log₁₀(S / N)
pub fn snr_db(signal_power: f64, noise_power: f64) -> f64 {
    assert!(noise_power > 0.0, "noise_power must be positive");
    10.0 * (signal_power / noise_power).log10()
}

/// Thermal noise floor in dBm: 10log₁₀(k_B × T × B) + 30
pub fn thermal_noise_floor_dbm(bandwidth: f64, temperature: f64) -> f64 {
    10.0 * (constants::K_B * temperature * bandwidth).log10() + 30.0
}

/// Shannon-Hartley channel capacity: C = B × log₂(1 + SNR)
pub fn shannon_capacity(bandwidth: f64, snr_linear: f64) -> f64 {
    bandwidth * (1.0 + snr_linear).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_rel(a: f64, b: f64, rel_tol: f64) -> bool {
        if b == 0.0 {
            return a.abs() < rel_tol;
        }
        ((a - b) / b).abs() < rel_tol
    }

    // ── EM Spectrum Classification ──

    #[test]
    fn test_wavelength_to_frequency() {
        let freq = wavelength_to_frequency(1.0);
        assert!(approx(freq, constants::C));
    }

    #[test]
    fn test_frequency_to_wavelength() {
        let wl = frequency_to_wavelength(constants::C);
        assert!(approx(wl, 1.0));
    }

    #[test]
    fn test_wavelength_frequency_roundtrip() {
        let freq = 2.4e9;
        let wl = frequency_to_wavelength(freq);
        let freq_back = wavelength_to_frequency(wl);
        assert!(approx_rel(freq_back, freq, 1e-10));
    }

    #[test]
    fn test_frequency_to_energy() {
        let energy = frequency_to_energy(1.0);
        assert!(approx(energy, constants::H));
    }

    // ── RF Propagation ──

    #[test]
    fn test_free_space_path_loss() {
        let fspl = free_space_path_loss(1000.0, 2.4e9);
        assert!(fspl > 0.0);
        // FSPL increases with distance
        let fspl_far = free_space_path_loss(2000.0, 2.4e9);
        assert!(fspl_far > fspl);
    }

    #[test]
    fn test_friis_received_power() {
        let wavelength = frequency_to_wavelength(2.4e9);
        let pr = friis_received_power(1.0, 1.0, 1.0, wavelength, 100.0);
        assert!(pr > 0.0);
        assert!(pr < 1.0);
        // Power decreases with distance
        let pr_far = friis_received_power(1.0, 1.0, 1.0, wavelength, 200.0);
        assert!(pr_far < pr);
    }

    #[test]
    fn test_link_budget_db() {
        let pr = link_budget_db(20.0, 10.0, 5.0, 100.0);
        assert!(approx(pr, -65.0));
    }

    #[test]
    fn test_skin_depth_conductor() {
        let delta = skin_depth_conductor(1e6, constants::MU_0, 5.8e7);
        assert!(delta > 0.0);
        // Higher frequency => smaller skin depth
        let delta_high = skin_depth_conductor(1e9, constants::MU_0, 5.8e7);
        assert!(delta_high < delta);
    }

    #[test]
    fn test_fade_margin_db() {
        let fm = fade_margin_db(20.0, -60.0, -90.0);
        assert!(approx(fm, 30.0));
    }

    // ── Antenna Theory ──

    #[test]
    fn test_antenna_gain_area_roundtrip() {
        let wavelength = 0.125;
        let area = 0.5;
        let gain = antenna_gain_from_area(area, wavelength);
        let area_back = effective_area_from_gain(gain, wavelength);
        assert!(approx_rel(area_back, area, 1e-10));
    }

    #[test]
    fn test_half_wave_dipole_gain() {
        let g = half_wave_dipole_gain();
        assert!(approx(g, 1.64));
    }

    #[test]
    fn test_eirp() {
        let result = eirp(100.0, 10.0);
        assert!(approx(result, 1000.0));
    }

    #[test]
    fn test_beamwidth_approximate() {
        let bw = beamwidth_approximate(0.03, 1.0);
        assert!(approx(bw, 2.1));
    }

    #[test]
    fn test_antenna_directivity() {
        let d = antenna_directivity(10.0, 0.5);
        assert!(approx(d, 20.0));
    }

    // ── Transmission Lines ──

    #[test]
    fn test_characteristic_impedance_coax() {
        // 50 ohm coax approximation: D/d ≈ 2.3 with air dielectric
        let z = characteristic_impedance_coax(2.3, 1.0, 1.0);
        assert!(approx_rel(z, 49.77, 0.01));
    }

    #[test]
    fn test_velocity_factor() {
        let vf = velocity_factor(2.25);
        assert!(approx_rel(vf, 1.0 / 1.5, 1e-10));
    }

    #[test]
    fn test_wavelength_in_line() {
        let wl = wavelength_in_line(1.0, 0.66);
        assert!(approx(wl, 0.66));
    }

    #[test]
    fn test_vswr() {
        let v = vswr(0.5);
        assert!(approx(v, 3.0));

        // Perfect match
        let v_perfect = vswr(0.0);
        assert!(approx(v_perfect, 1.0));
    }

    #[test]
    fn test_return_loss() {
        let rl = return_loss(0.1);
        assert!(approx_rel(rl, 20.0, 1e-6));
    }

    #[test]
    fn test_mismatch_loss() {
        // Perfect match: VSWR = 1.0, no mismatch loss
        let ml = mismatch_loss(1.0);
        assert!(approx(ml, 0.0));

        // Higher VSWR means more mismatch loss
        let ml_high = mismatch_loss(3.0);
        assert!(ml_high > 0.0);
    }

    // ── Signal & Modulation Basics ──

    #[test]
    fn test_dbm_watts_roundtrip() {
        let dbm = 30.0;
        let watts = dbm_to_watts(dbm);
        assert!(approx(watts, 1.0));
        let dbm_back = watts_to_dbm(watts);
        assert!(approx(dbm_back, dbm));
    }

    #[test]
    fn test_dbm_to_watts_zero() {
        let watts = dbm_to_watts(0.0);
        assert!(approx(watts, 0.001));
    }

    #[test]
    fn test_db_ratio_roundtrip() {
        let db = 3.0;
        let ratio = db_to_ratio(db);
        let db_back = ratio_to_db(ratio);
        assert!(approx(db_back, db));
    }

    #[test]
    fn test_noise_power() {
        let n = noise_power(1.0e6, 290.0);
        let expected = 4.003882100000000e-15;
        assert!(approx_rel(n, expected, 1e-10));
    }

    #[test]
    fn test_snr_db() {
        let snr = snr_db(100.0, 1.0);
        assert!(approx(snr, 20.0));
    }

    #[test]
    fn test_thermal_noise_floor_dbm() {
        // At 290K with 1 Hz bandwidth: approximately -174 dBm
        let floor = thermal_noise_floor_dbm(1.0, 290.0);
        assert!(approx_rel(floor, -174.0, 0.01));
    }

    #[test]
    fn test_shannon_capacity() {
        let cap = shannon_capacity(1.0e6, 1000.0);
        // C = 1e6 * log2(1001)
        let expected = 9.967226258835994e6;
        assert!(approx_rel(cap, expected, 1e-10));
    }

    #[test]
    fn test_approx_rel_zero_b() {
        assert!(approx_rel(0.0, 0.0, 1e-6));
        assert!(!approx_rel(1.0, 0.0, 0.5));
    }
}
