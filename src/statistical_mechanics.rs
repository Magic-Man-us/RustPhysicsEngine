use crate::math::constants;

// ── Brownian Motion & Diffusion ──

/// Stokes-Einstein diffusion coefficient: D = k_B × T / (6π × μ × r)
pub fn einstein_diffusion(temperature: f64, dynamic_viscosity: f64, particle_radius: f64) -> f64 {
    assert!(dynamic_viscosity > 0.0, "dynamic_viscosity must be positive");
    assert!(particle_radius > 0.0, "particle_radius must be positive");
    constants::K_B * temperature
        / (6.0 * constants::PI * dynamic_viscosity * particle_radius)
}

/// Mean square displacement: ⟨r²⟩ = 2nDt where n = number of spatial dimensions
pub fn mean_square_displacement(diffusion_coeff: f64, time: f64, dimensions: u32) -> f64 {
    2.0 * dimensions as f64 * diffusion_coeff * time
}

/// Root-mean-square displacement: √(⟨r²⟩)
pub fn rms_displacement(diffusion_coeff: f64, time: f64, dimensions: u32) -> f64 {
    mean_square_displacement(diffusion_coeff, time, dimensions).sqrt()
}

/// Fick's first law: J = -D × (dc/dx)
pub fn fick_first_law(diffusion_coeff: f64, concentration_gradient: f64) -> f64 {
    -diffusion_coeff * concentration_gradient
}

/// Fick's second law via explicit finite-difference step in 1D.
/// Updates `concentrations` in place. Boundary cells (first and last) are held fixed.
pub fn fick_second_law_step_1d(
    concentrations: &mut [f64],
    dx: f64,
    dt: f64,
    diffusion_coeff: f64,
) {
    assert!(dx > 0.0, "dx must be positive");
    let n = concentrations.len();
    if n < 3 {
        return;
    }

    let alpha = diffusion_coeff * dt / (dx * dx);

    // Snapshot current values so the update uses consistent data
    let old: Vec<f64> = concentrations.to_vec();

    for i in 1..n - 1 {
        concentrations[i] = old[i] + alpha * (old[i + 1] - 2.0 * old[i] + old[i - 1]);
    }
}

/// Characteristic diffusion length: L = √(2Dt)
pub fn diffusion_length(diffusion_coeff: f64, time: f64) -> f64 {
    (2.0 * diffusion_coeff * time).sqrt()
}

/// Time required to diffuse a given length: t = L² / (2D)
pub fn diffusion_time(diffusion_coeff: f64, length: f64) -> f64 {
    assert!(diffusion_coeff > 0.0, "diffusion_coeff must be positive");
    length * length / (2.0 * diffusion_coeff)
}

// ── Maxwell-Boltzmann Distribution ──

/// Maxwell speed distribution: f(v) = 4π × (m/(2πk_BT))^(3/2) × v² × exp(-mv²/(2k_BT))
pub fn maxwell_speed_distribution(mass: f64, temperature: f64, speed: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    assert!(temperature > 0.0, "temperature must be positive");
    let a = mass / (2.0 * constants::PI * constants::K_B * temperature);
    4.0 * constants::PI * a.powf(1.5) * speed * speed
        * (-mass * speed * speed / (2.0 * constants::K_B * temperature)).exp()
}

/// Most probable speed: v_p = √(2k_BT / m)
pub fn most_probable_speed(mass: f64, temperature: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    (2.0 * constants::K_B * temperature / mass).sqrt()
}

/// Mean speed: v̄ = √(8k_BT / (πm))
pub fn mean_speed(mass: f64, temperature: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    (8.0 * constants::K_B * temperature / (constants::PI * mass)).sqrt()
}

/// RMS speed from Maxwell-Boltzmann: v_rms = √(3k_BT / m)
pub fn rms_speed_maxwell(mass: f64, temperature: f64) -> f64 {
    assert!(mass > 0.0, "mass must be positive");
    (3.0 * constants::K_B * temperature / mass).sqrt()
}

// ── Equipartition & Statistical Averages ──

/// Equipartition energy: E = (f/2) × k_B × T
pub fn equipartition_energy(degrees_of_freedom: u32, temperature: f64) -> f64 {
    degrees_of_freedom as f64 / 2.0 * constants::K_B * temperature
}

/// Equipartition heat capacity per particle: Cv = (f/2) × k_B
pub fn equipartition_heat_capacity(degrees_of_freedom: u32) -> f64 {
    degrees_of_freedom as f64 / 2.0 * constants::K_B
}

/// Boltzmann factor: exp(-E / (k_B × T))
pub fn boltzmann_factor(energy: f64, temperature: f64) -> f64 {
    assert!(temperature > 0.0, "temperature must be positive");
    (-energy / (constants::K_B * temperature)).exp()
}

/// Boltzmann probability: P = exp(-E/(k_BT)) / Z
pub fn boltzmann_probability(energy: f64, temperature: f64, partition_function: f64) -> f64 {
    assert!(partition_function > 0.0, "partition_function must be positive");
    boltzmann_factor(energy, temperature) / partition_function
}

/// Partition function for a quantum harmonic oscillator: Z = 1 / (1 - exp(-hf/(k_BT)))
pub fn partition_function_harmonic(temperature: f64, frequency: f64) -> f64 {
    assert!(temperature > 0.0, "temperature must be positive");
    let x = constants::H * frequency / (constants::K_B * temperature);
    1.0 / (1.0 - (-x).exp())
}

/// Mean energy of a quantum harmonic oscillator (includes zero-point energy):
/// ⟨E⟩ = hf / (exp(hf/(k_BT)) - 1) + hf/2
pub fn mean_energy_harmonic(temperature: f64, frequency: f64) -> f64 {
    assert!(temperature > 0.0, "temperature must be positive");
    let hf = constants::H * frequency;
    let x = hf / (constants::K_B * temperature);
    hf / (x.exp() - 1.0) + hf / 2.0
}

// ── Debye Model ──

/// Debye temperature: Θ_D = h × f_max / k_B
pub fn debye_temperature(max_frequency: f64) -> f64 {
    constants::H * max_frequency / constants::K_B
}

/// Dulong-Petit limit (high-T Debye heat capacity): Cv = 3Nk_B
pub fn debye_heat_capacity_high_t(n_atoms: f64) -> f64 {
    3.0 * n_atoms * constants::K_B
}

/// Low-temperature Debye heat capacity: Cv = (12/5)π⁴Nk_B(T/Θ_D)³
pub fn debye_heat_capacity_low_t(
    n_atoms: f64,
    temperature: f64,
    debye_temp: f64,
) -> f64 {
    assert!(debye_temp > 0.0, "debye_temp must be positive");
    let pi4 = constants::PI.powi(4);
    let ratio = temperature / debye_temp;
    (12.0 / 5.0) * pi4 * n_atoms * constants::K_B * ratio.powi(3)
}

/// Einstein model heat capacity:
/// Cv = 3Nk_B × (Θ_E/T)² × exp(Θ_E/T) / (exp(Θ_E/T) - 1)²
pub fn einstein_heat_capacity(
    n_atoms: f64,
    temperature: f64,
    einstein_temp: f64,
) -> f64 {
    assert!(temperature > 0.0, "temperature must be positive");
    let x = einstein_temp / temperature;
    let ex = x.exp();
    let denom = ex - 1.0;
    3.0 * n_atoms * constants::K_B * x * x * ex / (denom * denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool {
        if b.abs() < 1e-30 {
            return (a - b).abs() < 1e-30;
        }
        ((a - b) / b).abs() < TOLERANCE
    }

    // ── Brownian Motion & Diffusion ──

    #[test]
    fn test_einstein_diffusion() {
        // Water at 300 K, viscosity ~1e-3 Pa·s, 1 μm particle
        let d = einstein_diffusion(300.0, 1e-3, 1e-6);
        let expected = 2.197371130248822e-13;
        assert!(approx(d, expected), "got {d}, expected {expected}");
    }

    #[test]
    fn test_mean_square_displacement() {
        let msd = mean_square_displacement(1e-12, 10.0, 3);
        // ⟨r²⟩ = 2 × 3 × 1e-12 × 10 = 6e-11
        assert!(approx(msd, 6e-11));
    }

    #[test]
    fn test_rms_displacement() {
        let rms = rms_displacement(1e-12, 10.0, 3);
        assert!(approx(rms, (6e-11_f64).sqrt()));
    }

    #[test]
    fn test_fick_first_law() {
        let j = fick_first_law(2.0, 5.0);
        assert!(approx(j, -10.0));
    }

    #[test]
    fn test_fick_second_law_step_1d() {
        // Step function: left half at 1.0, right half at 0.0
        let mut c = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let dx = 1.0;
        let dt = 0.1;
        let d = 1.0;
        fick_second_law_step_1d(&mut c, dx, dt, d);
        // Boundaries unchanged
        assert_eq!(c[0], 1.0);
        assert_eq!(c[5], 0.0);
        // Interior cell at index 3 was 0.0 with neighbors (1.0, 0.0) -> 0.0 + 0.1*(1.0 - 0.0 + 0.0) = 0.1
        assert!(approx(c[3], 0.1));
    }

    #[test]
    fn test_fick_second_law_short_array() {
        let mut c = vec![1.0, 0.0];
        fick_second_law_step_1d(&mut c, 1.0, 0.1, 1.0);
        // Should be a no-op for arrays shorter than 3
        assert_eq!(c, vec![1.0, 0.0]);
    }

    #[test]
    fn test_diffusion_length() {
        let l = diffusion_length(1e-9, 100.0);
        assert!(approx(l, (2e-7_f64).sqrt()));
    }

    #[test]
    fn test_diffusion_time() {
        let t = diffusion_time(1e-9, 1e-3);
        // t = (1e-3)^2 / (2 × 1e-9) = 1e-6 / 2e-9 = 500
        assert!(approx(t, 500.0));
    }

    #[test]
    fn test_diffusion_length_time_roundtrip() {
        let d = 2.5e-10;
        let t_orig = 42.0;
        let l = diffusion_length(d, t_orig);
        let t_back = diffusion_time(d, l);
        assert!(approx(t_back, t_orig));
    }

    // ── Maxwell-Boltzmann Distribution ──

    #[test]
    fn test_maxwell_speed_distribution_peak_near_most_probable() {
        let mass = constants::M_PROTON;
        let temp = 300.0;
        let vp = most_probable_speed(mass, temp);
        let f_at_vp = maxwell_speed_distribution(mass, temp, vp);
        let f_below = maxwell_speed_distribution(mass, temp, vp * 0.9);
        let f_above = maxwell_speed_distribution(mass, temp, vp * 1.1);
        assert!(f_at_vp > f_below);
        assert!(f_at_vp > f_above);
    }

    #[test]
    fn test_maxwell_speed_distribution_zero_at_zero() {
        let f = maxwell_speed_distribution(constants::M_PROTON, 300.0, 0.0);
        assert!(f.abs() < 1e-30);
    }

    #[test]
    fn test_most_probable_speed() {
        let vp = most_probable_speed(constants::M_PROTON, 300.0);
        let expected = 2225.452730128216;
        assert!(approx(vp, expected));
    }

    #[test]
    fn test_mean_speed() {
        let vm = mean_speed(constants::M_PROTON, 300.0);
        let expected = 2511.154498032511;
        assert!(approx(vm, expected));
    }

    #[test]
    fn test_rms_speed_maxwell() {
        let vrms = rms_speed_maxwell(constants::M_PROTON, 300.0);
        let expected = 2725.611817748942;
        assert!(approx(vrms, expected));
    }

    #[test]
    fn test_speed_ordering() {
        // v_p < v_mean < v_rms
        let m = constants::M_PROTON;
        let t = 500.0;
        let vp = most_probable_speed(m, t);
        let vm = mean_speed(m, t);
        let vrms = rms_speed_maxwell(m, t);
        assert!(vp < vm);
        assert!(vm < vrms);
    }

    // ── Equipartition & Statistical Averages ──

    #[test]
    fn test_equipartition_energy_monatomic() {
        // Monatomic ideal gas: 3 translational DOF
        let e = equipartition_energy(3, 300.0);
        let expected = 6.2129205e-21;
        assert!(approx(e, expected));
    }

    #[test]
    fn test_equipartition_heat_capacity() {
        let cv = equipartition_heat_capacity(5);
        assert!(approx(cv, 3.4516225e-23));
    }

    #[test]
    fn test_boltzmann_factor() {
        let bf = boltzmann_factor(0.0, 300.0);
        assert!(approx(bf, 1.0));
    }

    #[test]
    fn test_boltzmann_factor_high_energy() {
        // At energy >> k_B*T, factor approaches 0
        let bf = boltzmann_factor(1.0, 300.0);
        assert!(bf < 1e-10);
    }

    #[test]
    fn test_boltzmann_probability() {
        let z = 5.0;
        let p = boltzmann_probability(0.0, 300.0, z);
        assert!(approx(p, 1.0 / z));
    }

    #[test]
    fn test_partition_function_harmonic_high_t() {
        // At high T, Z ≈ k_BT / (hf)
        let freq = 1e12;
        let temp = 1e6;
        let z = partition_function_harmonic(temp, freq);
        let z_classical = constants::K_B * temp / (constants::H * freq);
        assert!(((z - z_classical) / z_classical).abs() < 0.01);
    }

    #[test]
    fn test_mean_energy_harmonic_high_t() {
        // At high T, ⟨E⟩ ≈ k_BT (classical limit, zero-point negligible relative to k_BT)
        let freq = 1e12;
        let temp = 1e6;
        let e = mean_energy_harmonic(temp, freq);
        let kbt = constants::K_B * temp;
        assert!(((e - kbt) / kbt).abs() < 0.01);
    }

    #[test]
    fn test_mean_energy_harmonic_low_t() {
        // At T → 0, ⟨E⟩ → hf/2 (zero-point energy dominates)
        let freq = 1e13;
        let temp = 1.0;
        let e = mean_energy_harmonic(temp, freq);
        let zpe = constants::H * freq / 2.0;
        assert!(approx(e, zpe));
    }

    // ── Debye Model ──

    #[test]
    fn test_debye_temperature() {
        let f_max = 1e13;
        let theta = debye_temperature(f_max);
        let expected = 479.9243073366221;
        assert!(approx(theta, expected));
    }

    #[test]
    fn test_debye_heat_capacity_high_t() {
        // 1 mole of atoms
        let n = constants::N_A;
        let cv = debye_heat_capacity_high_t(n);
        // Should equal 3R per mole ≈ 24.9 J/(mol·K)
        let expected = 24.94338785445972;
        assert!(approx(cv, expected));
    }

    #[test]
    fn test_debye_heat_capacity_low_t_cubic_scaling() {
        let n = 1.0;
        let theta_d = 400.0;
        let cv1 = debye_heat_capacity_low_t(n, 10.0, theta_d);
        let cv2 = debye_heat_capacity_low_t(n, 20.0, theta_d);
        // Doubling T should multiply Cv by 2^3 = 8
        assert!(approx(cv2 / cv1, 8.0));
    }

    #[test]
    fn test_einstein_heat_capacity_high_t() {
        // At T >> Θ_E, Cv → 3Nk_B (Dulong-Petit)
        let n = 1.0;
        let theta_e = 200.0;
        let temp = 1e6;
        let cv = einstein_heat_capacity(n, temp, theta_e);
        let dp = 3.0 * n * constants::K_B;
        assert!(((cv - dp) / dp).abs() < 0.01);
    }

    #[test]
    fn test_einstein_heat_capacity_low_t() {
        // At T << Θ_E, Cv → 0 exponentially
        let cv = einstein_heat_capacity(1.0, 50.0, 1000.0);
        assert!(cv < 1e-6, "Cv at low T should be negligible, got {cv}");
    }

    #[test]
    fn test_approx_near_zero_b() {
        assert!(approx(0.0, 0.0));
        assert!(!approx(1.0, 0.0));
    }
}
