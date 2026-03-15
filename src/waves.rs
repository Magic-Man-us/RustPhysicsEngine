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
}
