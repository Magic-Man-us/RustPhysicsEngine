use crate::math::constants;

// ── Wave Basics ──

/// Wave speed: v = f * λ
pub fn wave_speed(frequency: f64, wavelength: f64) -> f64 {
    frequency * wavelength
}

/// Wavelength from speed and frequency: λ = v / f
pub fn wavelength(speed: f64, frequency: f64) -> f64 {
    speed / frequency
}

/// Frequency from speed and wavelength: f = v / λ
pub fn frequency(speed: f64, wavelength: f64) -> f64 {
    speed / wavelength
}

/// Period: T = 1 / f
pub fn period(frequency: f64) -> f64 {
    1.0 / frequency
}

/// Angular frequency: ω = 2πf
pub fn angular_frequency(frequency: f64) -> f64 {
    2.0 * constants::PI * frequency
}

/// Wave number: k = 2π / λ
pub fn wave_number(wavelength: f64) -> f64 {
    2.0 * constants::PI / wavelength
}

/// Transverse wave displacement: y(x,t) = A * sin(kx - ωt + φ)
pub fn wave_displacement(
    amplitude: f64,
    wave_number: f64,
    x: f64,
    angular_freq: f64,
    t: f64,
    phase: f64,
) -> f64 {
    amplitude * (wave_number * x - angular_freq * t + phase).sin()
}

// ── Wave Energy ──

/// Energy of a wave (proportional): E ∝ A^2 * f^2
/// Returns the energy for a given amplitude and frequency (with a constant factor).
pub fn wave_energy_density(amplitude: f64, frequency: f64, linear_density: f64) -> f64 {
    0.5 * linear_density * (2.0 * constants::PI * frequency).powi(2) * amplitude * amplitude
}

/// Intensity of a wave: I = P / A (power per unit area)
pub fn wave_intensity(power: f64, area: f64) -> f64 {
    power / area
}

/// Intensity falls off with distance (spherical wave): I = P / (4πr^2)
pub fn spherical_wave_intensity(power: f64, distance: f64) -> f64 {
    power / (4.0 * constants::PI * distance * distance)
}

/// Decibel level: β = 10 * log10(I / I_0)
pub fn decibel_level(intensity: f64, reference_intensity: f64) -> f64 {
    10.0 * (intensity / reference_intensity).log10()
}

/// Intensity from decibel level: I = I_0 * 10^(β/10)
pub fn intensity_from_decibels(decibels: f64, reference_intensity: f64) -> f64 {
    reference_intensity * 10.0_f64.powf(decibels / 10.0)
}

// ── Doppler Effect ──

/// Doppler effect (sound): f' = f * (v + v_observer) / (v + v_source)
/// Convention: positive v_observer = observer moving toward source,
/// positive v_source = source moving away from observer.
pub fn doppler_frequency(
    source_freq: f64,
    wave_speed: f64,
    observer_velocity: f64,
    source_velocity: f64,
) -> f64 {
    source_freq * (wave_speed + observer_velocity) / (wave_speed + source_velocity)
}

/// Relativistic Doppler effect: f' = f * sqrt((1 + β) / (1 - β))
/// where β = v/c, positive β = approaching.
pub fn relativistic_doppler(source_freq: f64, beta: f64) -> f64 {
    source_freq * ((1.0 + beta) / (1.0 - beta)).sqrt()
}

/// Mach number: M = v_object / v_sound
pub fn mach_number(object_speed: f64, sound_speed: f64) -> f64 {
    object_speed / sound_speed
}

/// Mach cone half-angle: sin(θ) = v_sound / v_object = 1/M
pub fn mach_cone_angle(mach: f64) -> f64 {
    (1.0 / mach).asin()
}

// ── Standing Waves ──

/// Frequencies of standing waves on a string fixed at both ends:
/// f_n = n * v / (2L)
pub fn standing_wave_frequency(harmonic: u32, wave_speed: f64, length: f64) -> f64 {
    harmonic as f64 * wave_speed / (2.0 * length)
}

/// Fundamental frequency of a string: f = (1/(2L)) * sqrt(T/μ)
/// T = tension, μ = linear mass density
pub fn string_fundamental(length: f64, tension: f64, linear_density: f64) -> f64 {
    (1.0 / (2.0 * length)) * (tension / linear_density).sqrt()
}

/// Standing waves in an open pipe: f_n = n * v / (2L) (all harmonics)
pub fn open_pipe_frequency(harmonic: u32, sound_speed: f64, length: f64) -> f64 {
    harmonic as f64 * sound_speed / (2.0 * length)
}

/// Standing waves in a closed pipe: f_n = n * v / (4L) (odd harmonics only)
pub fn closed_pipe_frequency(odd_harmonic: u32, sound_speed: f64, length: f64) -> f64 {
    odd_harmonic as f64 * sound_speed / (4.0 * length)
}

// ── Superposition ──

/// Beat frequency: f_beat = |f1 - f2|
pub fn beat_frequency(f1: f64, f2: f64) -> f64 {
    (f1 - f2).abs()
}

/// Superposition of two waves at a point (same frequency):
/// A_resultant = sqrt(A1^2 + A2^2 + 2*A1*A2*cos(Δφ))
pub fn superposition_amplitude(a1: f64, a2: f64, phase_diff: f64) -> f64 {
    (a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * phase_diff.cos()).sqrt()
}

// ── Wave Speed in Media ──

/// Speed of sound in an ideal gas: v = sqrt(γ * R * T / M)
/// γ = heat capacity ratio, M = molar mass
pub fn speed_of_sound_gas(gamma: f64, temperature: f64, molar_mass: f64) -> f64 {
    (gamma * constants::R * temperature / molar_mass).sqrt()
}

/// Wave speed on a string: v = sqrt(T / μ)
pub fn wave_speed_string(tension: f64, linear_density: f64) -> f64 {
    (tension / linear_density).sqrt()
}

// ── Wave Propagation ──

/// Phase velocity: v_p = ω/k
pub fn phase_velocity(angular_freq: f64, wave_number: f64) -> f64 {
    angular_freq / wave_number
}

/// Group velocity: v_g = dω/dk
pub fn group_velocity(d_omega: f64, d_k: f64) -> f64 {
    d_omega / d_k
}

/// Acoustic impedance: Z = ρv
pub fn wave_impedance(density: f64, wave_speed: f64) -> f64 {
    density * wave_speed
}

/// Amplitude reflection coefficient: R = (Z2 - Z1)/(Z2 + Z1)
pub fn reflection_coefficient(z1: f64, z2: f64) -> f64 {
    (z2 - z1) / (z2 + z1)
}

/// Amplitude transmission coefficient: T = 2Z2/(Z1 + Z2)
pub fn transmission_coefficient(z1: f64, z2: f64) -> f64 {
    2.0 * z2 / (z1 + z2)
}

/// Intensity reflection coefficient: R_I = ((Z2 - Z1)/(Z2 + Z1))²
pub fn intensity_reflection(z1: f64, z2: f64) -> f64 {
    let r = reflection_coefficient(z1, z2);
    r * r
}

/// Intensity transmission coefficient: T_I = 4Z1Z2/(Z1 + Z2)²
pub fn intensity_transmission(z1: f64, z2: f64) -> f64 {
    let sum = z1 + z2;
    4.0 * z1 * z2 / (sum * sum)
}

// ── Wave Attenuation ──

/// Attenuated amplitude: A = A₀ × e^(-αx)
pub fn attenuated_amplitude(initial: f64, attenuation_coeff: f64, distance: f64) -> f64 {
    initial * (-attenuation_coeff * distance).exp()
}

/// Absorption coefficient from dB/m: α = dB × ln(10)/20
pub fn absorption_coefficient_from_db(db_per_meter: f64) -> f64 {
    db_per_meter * 10.0_f64.ln() / 20.0
}

/// Penetration depth (skin depth): δ = 1/α
pub fn penetration_depth(attenuation_coeff: f64) -> f64 {
    1.0 / attenuation_coeff
}

// ── Acoustic Physics ──

/// Sound pressure level: SPL = 20 × log10(p/p_ref)
pub fn sound_pressure_level(pressure: f64, reference: f64) -> f64 {
    20.0 * (pressure / reference).log10()
}

/// Acoustic power: P = p²A/Z
pub fn acoustic_power(pressure: f64, area: f64, impedance: f64) -> f64 {
    pressure * pressure * area / impedance
}

/// Resonant frequency of open tube: f = v/(2L)
pub fn resonant_frequency_tube_open(length: f64, speed: f64) -> f64 {
    speed / (2.0 * length)
}

/// Resonant frequency of closed tube: f = v/(4L)
pub fn resonant_frequency_tube_closed(length: f64, speed: f64) -> f64 {
    speed / (4.0 * length)
}

/// Acoustic intensity from pressure: I = p²/Z
pub fn acoustic_intensity_from_pressure(pressure: f64, impedance: f64) -> f64 {
    pressure * pressure / impedance
}

/// Wavelength in a medium: λ = v/f
pub fn wavelength_in_medium(frequency: f64, speed_in_medium: f64) -> f64 {
    speed_in_medium / frequency
}

// ── Seismic / Mechanical Waves ──

/// P-wave speed: vp = √((K + 4G/3)/ρ)
pub fn p_wave_speed(bulk_modulus: f64, shear_modulus: f64, density: f64) -> f64 {
    ((bulk_modulus + 4.0 * shear_modulus / 3.0) / density).sqrt()
}

/// S-wave speed: vs = √(G/ρ)
pub fn s_wave_speed(shear_modulus: f64, density: f64) -> f64 {
    (shear_modulus / density).sqrt()
}

/// Rayleigh wave speed approximation: vR ≈ vs × (0.862 + 1.14ν)/(1 + ν)
pub fn rayleigh_wave_speed(shear_speed: f64, poisson_ratio: f64) -> f64 {
    shear_speed * (0.862 + 1.14 * poisson_ratio) / (1.0 + poisson_ratio)
}

/// Love wave speed range: between vs_layer and vs_halfspace
pub fn love_wave_speed_range(
    shear_speed_layer: f64,
    shear_speed_halfspace: f64,
) -> (f64, f64) {
    let low = shear_speed_layer.min(shear_speed_halfspace);
    let high = shear_speed_layer.max(shear_speed_halfspace);
    (low, high)
}

// ── Wave Interference & Diffraction ──

/// Constructive interference path difference: Δ = mλ
pub fn path_difference_constructive(order: i32, wavelength: f64) -> f64 {
    order as f64 * wavelength
}

/// Destructive interference path difference: Δ = (m + 0.5)λ
pub fn path_difference_destructive(order: i32, wavelength: f64) -> f64 {
    (order as f64 + 0.5) * wavelength
}

/// Fraunhofer single-slit intensity: I/I₀ = (sin(β)/β)² where β = πa sin(θ)/λ
/// Returns 1.0 at θ = 0 (central maximum).
pub fn fraunhofer_single_slit_intensity(
    angle: f64,
    slit_width: f64,
    wavelength: f64,
) -> f64 {
    let beta = constants::PI * slit_width * angle.sin() / wavelength;
    if beta.abs() < 1e-12 {
        return 1.0;
    }
    let sinc = beta.sin() / beta;
    sinc * sinc
}

/// Airy disk radius: r = 1.22λf/D
pub fn airy_disk_radius(wavelength: f64, focal_length: f64, aperture: f64) -> f64 {
    1.22 * wavelength * focal_length / aperture
}

/// Fresnel number: F = a²/(λL)
pub fn fresnel_number(aperture: f64, distance: f64, wavelength: f64) -> f64 {
    aperture * aperture / (wavelength * distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_wave_speed() {
        assert!(approx(wave_speed(440.0, 0.773), 340.12, 0.01));
    }

    #[test]
    fn test_doppler_approaching() {
        // Source approaching at 30 m/s, sound speed 340 m/s
        let f = doppler_frequency(440.0, 340.0, 0.0, -30.0);
        assert!(f > 440.0); // Frequency increases
    }

    #[test]
    fn test_doppler_receding() {
        let f = doppler_frequency(440.0, 340.0, 0.0, 30.0);
        assert!(f < 440.0); // Frequency decreases
    }

    #[test]
    fn test_decibel_level() {
        let db = decibel_level(1e-10, 1e-12);
        assert!(approx(db, 20.0, 0.01));
    }

    #[test]
    fn test_standing_wave() {
        // 2nd harmonic on 1m string with wave speed 340 m/s
        let f = standing_wave_frequency(2, 340.0, 1.0);
        assert!(approx(f, 340.0, 1e-6));
    }

    #[test]
    fn test_beat_frequency() {
        assert!(approx(beat_frequency(440.0, 442.0), 2.0, 1e-9));
    }

    #[test]
    fn test_superposition_constructive() {
        let a = superposition_amplitude(1.0, 1.0, 0.0);
        assert!(approx(a, 2.0, 1e-9));
    }

    #[test]
    fn test_superposition_destructive() {
        let a = superposition_amplitude(1.0, 1.0, constants::PI);
        assert!(approx(a, 0.0, 1e-9));
    }

    #[test]
    fn test_speed_of_sound_air() {
        // Air: γ=1.4, T=293K, M=0.029 kg/mol
        let v = speed_of_sound_gas(1.4, 293.0, 0.029);
        assert!(approx_rel(v, 343.0, 0.02));
    }

    #[test]
    fn test_spherical_wave_intensity() {
        let i = spherical_wave_intensity(100.0, 10.0);
        assert!(approx_rel(i, 0.0796, 0.01));
    }

    // ── Wave Propagation Tests ──

    #[test]
    fn test_phase_velocity() {
        // ω = 100 rad/s, k = 5 rad/m → v_p = 20 m/s
        assert!(approx(phase_velocity(100.0, 5.0), 20.0, 1e-9));
    }

    #[test]
    fn test_group_velocity() {
        assert!(approx(group_velocity(10.0, 2.0), 5.0, 1e-9));
    }

    #[test]
    fn test_wave_impedance() {
        // Air: ρ ≈ 1.225 kg/m³, v ≈ 343 m/s → Z ≈ 420.175
        let z = wave_impedance(1.225, 343.0);
        assert!(approx(z, 420.175, 0.001));
    }

    #[test]
    fn test_reflection_coefficient() {
        // Same medium → R = 0
        assert!(approx(reflection_coefficient(400.0, 400.0), 0.0, 1e-9));
        // Z1=400, Z2=1600 → R = 1200/2000 = 0.6
        assert!(approx(reflection_coefficient(400.0, 1600.0), 0.6, 1e-9));
    }

    #[test]
    fn test_transmission_coefficient() {
        // Same medium → T = 1
        assert!(approx(transmission_coefficient(400.0, 400.0), 1.0, 1e-9));
        // Z1=400, Z2=1600 → T = 3200/2000 = 1.6
        assert!(approx(transmission_coefficient(400.0, 1600.0), 1.6, 1e-9));
    }

    #[test]
    fn test_intensity_reflection() {
        // Z1=400, Z2=1600 → R_I = 0.36
        assert!(approx(intensity_reflection(400.0, 1600.0), 0.36, 1e-9));
    }

    #[test]
    fn test_intensity_transmission() {
        // Z1=400, Z2=1600 → T_I = 4*400*1600/2000² = 0.64
        assert!(approx(intensity_transmission(400.0, 1600.0), 0.64, 1e-9));
    }

    #[test]
    fn test_intensity_reflection_transmission_sum() {
        // R_I + T_I = 1 (energy conservation)
        let r = intensity_reflection(420.0, 1500.0);
        let t = intensity_transmission(420.0, 1500.0);
        assert!(approx(r + t, 1.0, 1e-9));
    }

    // ── Wave Attenuation Tests ──

    #[test]
    fn test_attenuated_amplitude() {
        // At distance 0, amplitude unchanged
        assert!(approx(attenuated_amplitude(5.0, 0.1, 0.0), 5.0, 1e-9));
        // At α=1, x=1: A = 5*e^(-1) ≈ 1.8394
        assert!(approx(attenuated_amplitude(5.0, 1.0, 1.0), 5.0 * (-1.0_f64).exp(), 1e-9));
    }

    #[test]
    fn test_absorption_coefficient_from_db() {
        // 1 dB/m → α = ln(10)/20 ≈ 0.11513
        let alpha = absorption_coefficient_from_db(1.0);
        assert!(approx(alpha, 10.0_f64.ln() / 20.0, 1e-9));
    }

    #[test]
    fn test_penetration_depth() {
        // α = 0.5 → δ = 2.0
        assert!(approx(penetration_depth(0.5), 2.0, 1e-9));
    }

    // ── Acoustic Physics Tests ──

    #[test]
    fn test_sound_pressure_level() {
        // 1 Pa relative to 2e-5 Pa → 20*log10(50000) ≈ 93.98 dB
        let spl = sound_pressure_level(1.0, 2e-5);
        assert!(approx(spl, 93.979, 0.01));
    }

    #[test]
    fn test_acoustic_power() {
        // p=2 Pa, A=0.5 m², Z=400 → P = 4*0.5/400 = 0.005 W
        assert!(approx(acoustic_power(2.0, 0.5, 400.0), 0.005, 1e-9));
    }

    #[test]
    fn test_resonant_frequency_tube_open() {
        // L=0.5 m, v=340 m/s → f = 340/1.0 = 340 Hz
        assert!(approx(resonant_frequency_tube_open(0.5, 340.0), 340.0, 1e-9));
    }

    #[test]
    fn test_resonant_frequency_tube_closed() {
        // L=0.5 m, v=340 m/s → f = 340/2.0 = 170 Hz
        assert!(approx(resonant_frequency_tube_closed(0.5, 340.0), 170.0, 1e-9));
    }

    #[test]
    fn test_acoustic_intensity_from_pressure() {
        // p=2 Pa, Z=400 → I = 4/400 = 0.01 W/m²
        assert!(approx(acoustic_intensity_from_pressure(2.0, 400.0), 0.01, 1e-9));
    }

    #[test]
    fn test_wavelength_in_medium() {
        // f=1000 Hz, v=1500 m/s (water) → λ = 1.5 m
        assert!(approx(wavelength_in_medium(1000.0, 1500.0), 1.5, 1e-9));
    }

    // ── Seismic / Mechanical Wave Tests ──

    #[test]
    fn test_p_wave_speed() {
        // K=40e9 Pa, G=30e9 Pa, ρ=2700 kg/m³
        // vp = sqrt((40e9 + 4*30e9/3) / 2700) = sqrt((40e9 + 40e9)/2700)
        let vp = p_wave_speed(40e9, 30e9, 2700.0);
        let expected = ((40e9_f64 + 4.0 * 30e9 / 3.0) / 2700.0).sqrt();
        assert!(approx(vp, expected, 0.01));
    }

    #[test]
    fn test_s_wave_speed() {
        // G=30e9 Pa, ρ=2700 → vs = sqrt(30e9/2700) ≈ 3333.3 m/s
        let vs = s_wave_speed(30e9, 2700.0);
        assert!(approx(vs, (30e9_f64 / 2700.0).sqrt(), 0.01));
    }

    #[test]
    fn test_p_wave_faster_than_s_wave() {
        let vp = p_wave_speed(40e9, 30e9, 2700.0);
        let vs = s_wave_speed(30e9, 2700.0);
        assert!(vp > vs);
    }

    #[test]
    fn test_rayleigh_wave_speed() {
        // vs=3000 m/s, ν=0.25 → vR = 3000*(0.862+0.285)/1.25 = 3000*0.9176 = 2752.8
        let vr = rayleigh_wave_speed(3000.0, 0.25);
        let expected = 3000.0 * (0.862 + 1.14 * 0.25) / 1.25;
        assert!(approx(vr, expected, 0.01));
    }

    #[test]
    fn test_rayleigh_slower_than_shear() {
        // Rayleigh waves are always slower than shear waves
        let vs = 3000.0;
        let vr = rayleigh_wave_speed(vs, 0.25);
        assert!(vr < vs);
    }

    #[test]
    fn test_love_wave_speed_range() {
        let (low, high) = love_wave_speed_range(2000.0, 3500.0);
        assert!(approx(low, 2000.0, 1e-9));
        assert!(approx(high, 3500.0, 1e-9));
    }

    #[test]
    fn test_love_wave_speed_range_reversed() {
        let (low, high) = love_wave_speed_range(3500.0, 2000.0);
        assert!(approx(low, 2000.0, 1e-9));
        assert!(approx(high, 3500.0, 1e-9));
    }

    // ── Wave Interference & Diffraction Tests ──

    #[test]
    fn test_path_difference_constructive() {
        // m=2, λ=0.5 → Δ = 1.0
        assert!(approx(path_difference_constructive(2, 0.5), 1.0, 1e-9));
        // m=0 → Δ = 0
        assert!(approx(path_difference_constructive(0, 0.5), 0.0, 1e-9));
    }

    #[test]
    fn test_path_difference_destructive() {
        // m=0, λ=0.5 → Δ = 0.25
        assert!(approx(path_difference_destructive(0, 0.5), 0.25, 1e-9));
        // m=1, λ=0.5 → Δ = 0.75
        assert!(approx(path_difference_destructive(1, 0.5), 0.75, 1e-9));
    }

    #[test]
    fn test_fraunhofer_central_maximum() {
        // At θ=0, intensity is 1.0
        assert!(approx(fraunhofer_single_slit_intensity(0.0, 1e-3, 500e-9), 1.0, 1e-9));
    }

    #[test]
    fn test_fraunhofer_first_minimum() {
        // First minimum at sin(θ) = λ/a → θ = asin(λ/a)
        let a = 1e-3;
        let lambda = 500e-9;
        let theta = (lambda as f64 / a as f64).asin();
        let intensity = fraunhofer_single_slit_intensity(theta, a, lambda);
        assert!(approx(intensity, 0.0, 1e-6));
    }

    #[test]
    fn test_fraunhofer_symmetry() {
        let a = 1e-3;
        let lambda = 500e-9;
        let theta = 0.01;
        let pos = fraunhofer_single_slit_intensity(theta, a, lambda);
        let neg = fraunhofer_single_slit_intensity(-theta, a, lambda);
        assert!(approx(pos, neg, 1e-12));
    }

    #[test]
    fn test_airy_disk_radius() {
        // λ=500nm, f=100mm, D=50mm → r = 1.22*500e-9*0.1/0.05 = 1.22e-6 m
        let r = airy_disk_radius(500e-9, 0.1, 0.05);
        assert!(approx(r, 1.22e-6, 1e-12));
    }

    #[test]
    fn test_fresnel_number() {
        // a=1mm, L=1m, λ=500nm → F = (1e-3)²/(500e-9 * 1) = 2.0
        let f = fresnel_number(1e-3, 1.0, 500e-9);
        assert!(approx(f, 2.0, 1e-6));
    }
}
