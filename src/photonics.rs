use crate::math::constants;

// ── Fiber Optics Constants ──────────────────────────────────────────────────

/// Single-mode cutoff: V < 2.405 (first zero of J₀ Bessel function)
const SINGLE_MODE_CUTOFF: f64 = 2.405;

// ── Gaussian Beam Propagation ───────────────────────────────────────────────

/// Beam waist from far-field divergence: w₀ = λ/(π×θ)
pub fn beam_waist_from_divergence(wavelength: f64, divergence_half_angle: f64) -> f64 {
    assert!(divergence_half_angle > 0.0, "divergence_half_angle must be positive");
    wavelength / (constants::PI * divergence_half_angle)
}

/// Rayleigh range: z_R = πw₀²/λ
pub fn rayleigh_range(waist: f64, wavelength: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    constants::PI * waist * waist / wavelength
}

/// Beam radius at axial position z: w(z) = w₀√(1+(z/z_R)²)
pub fn beam_radius(waist: f64, z: f64, rayleigh: f64) -> f64 {
    assert!(rayleigh > 0.0, "rayleigh range must be positive");
    waist * (1.0 + (z / rayleigh).powi(2)).sqrt()
}

/// Far-field half-angle divergence: θ = λ/(πw₀)
pub fn beam_divergence(waist: f64, wavelength: f64) -> f64 {
    assert!(waist > 0.0, "waist must be positive");
    wavelength / (constants::PI * waist)
}

/// Radius of curvature of the wavefront: R(z) = z(1+(z_R/z)²)
pub fn beam_curvature(z: f64, rayleigh: f64) -> f64 {
    assert!(z != 0.0, "z must not be zero");
    z * (1.0 + (rayleigh / z).powi(2))
}

/// Gouy phase shift: ψ(z) = atan(z/z_R)
pub fn gouy_phase(z: f64, rayleigh: f64) -> f64 {
    assert!(rayleigh > 0.0, "rayleigh range must be positive");
    (z / rayleigh).atan()
}

/// Peak-normalized Gaussian beam intensity at radial offset r and axial position z:
/// I = (2P/(πw²)) exp(-2r²/w²), where w = w(z).
pub fn beam_intensity(power: f64, waist: f64, r: f64, z: f64, rayleigh: f64) -> f64 {
    let w = beam_radius(waist, z, rayleigh);
    let w_sq = w * w;
    (2.0 * power / (constants::PI * w_sq)) * (-2.0 * r * r / w_sq).exp()
}

/// Combined beam parameter: returns (w(z), R(z)) at axial position z.
pub fn beam_parameter(waist: f64, z: f64, rayleigh: f64) -> (f64, f64) {
    (beam_radius(waist, z, rayleigh), beam_curvature(z, rayleigh))
}

// ── Fiber Optics ────────────────────────────────────────────────────────────

/// Numerical aperture of a step-index fiber: NA = √(n_core² - n_clad²)
pub fn numerical_aperture(n_core: f64, n_clad: f64) -> f64 {
    (n_core * n_core - n_clad * n_clad).sqrt()
}

/// Maximum acceptance half-angle: θ_max = arcsin(NA)
pub fn acceptance_angle(na: f64) -> f64 {
    na.asin()
}

/// Normalized frequency (V-number): V = 2πr×NA/λ
pub fn v_number(radius: f64, na: f64, wavelength: f64) -> f64 {
    assert!(wavelength > 0.0, "wavelength must be positive");
    constants::TAU * radius * na / wavelength
}

/// True when the fiber supports only the fundamental mode (V < 2.405).
pub fn is_single_mode(v_number: f64) -> bool {
    v_number < SINGLE_MODE_CUTOFF
}

/// Approximate mode count for a step-index multimode fiber: M ≈ V²/2
pub fn number_of_modes(v_number: f64) -> f64 {
    v_number * v_number / 2.0
}

/// Output power after propagation through a lossy fiber:
/// P_out = P_in × 10^(-αL/10), where α is in dB/km and L in km.
pub fn fiber_attenuation(input_power: f64, attenuation_db_per_km: f64, length_km: f64) -> f64 {
    input_power * 10.0_f64.powf(-attenuation_db_per_km * length_km / 10.0)
}

/// Chromatic dispersion pulse broadening: Δt = D × L × Δλ
/// D in ps/(nm·km), L in km, Δλ in nm → Δt in ps.
pub fn dispersion_broadening(dispersion: f64, length: f64, spectral_width: f64) -> f64 {
    dispersion * length * spectral_width
}

/// Critical angle for total internal reflection inside the fiber core:
/// θc = arcsin(n_clad / n_core)
pub fn critical_angle_fiber(n_core: f64, n_clad: f64) -> f64 {
    assert!(n_core > 0.0, "n_core must be positive");
    (n_clad / n_core).asin()
}

// ── Thin Lens / ABCD Ray-Transfer Matrices ──────────────────────────────────

/// Type alias for a 2×2 ray-transfer (ABCD) matrix.
pub type RayMatrix = [[f64; 2]; 2];

/// Thin lens matrix: [[1, 0], [-1/f, 1]]
pub fn thin_lens_matrix(focal_length: f64) -> RayMatrix {
    assert!(focal_length != 0.0, "focal_length must not be zero");
    [
        [1.0, 0.0],
        [-1.0 / focal_length, 1.0],
    ]
}

/// Free-space propagation matrix: [[1, d], [0, 1]]
pub fn free_space_matrix(distance: f64) -> RayMatrix {
    [
        [1.0, distance],
        [0.0, 1.0],
    ]
}

/// Multiplies two 2×2 ray-transfer matrices: result = m1 × m2.
pub fn multiply_ray_matrices(m1: &RayMatrix, m2: &RayMatrix) -> RayMatrix {
    [
        [
            m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0],
            m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1],
        ],
        [
            m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0],
            m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1],
        ],
    ]
}

/// Applies a ray-transfer matrix to a ray (height, angle), returning (y', θ').
pub fn apply_ray_matrix(matrix: &RayMatrix, height: f64, angle: f64) -> (f64, f64) {
    (
        matrix[0][0] * height + matrix[0][1] * angle,
        matrix[1][0] * height + matrix[1][1] * angle,
    )
}

/// Image distance for a thick lens via ABCD matrix composition.
///
/// Constructs the system matrix from: refraction at R1, propagation through
/// the lens of thickness `t` and index `n`, refraction at R2, then solves
/// for the image distance using the thin-lens-equivalent focal length.
pub fn image_distance_thick_lens(
    n: f64,
    r1: f64,
    r2: f64,
    thickness: f64,
    object_dist: f64,
) -> f64 {
    assert!(r1 != 0.0, "r1 must not be zero");
    assert!(r2 != 0.0, "r2 must not be zero");
    assert!(n > 0.0, "refractive index must be positive");
    // Refraction matrix at surface with radius R entering medium of index n_to from n_from:
    // [[1, 0], [-(n_to - n_from)/R, 1]]
    let refraction_r1: RayMatrix = [
        [1.0, 0.0],
        [-(n - 1.0) / r1, 1.0],
    ];
    let propagation: RayMatrix = [
        [1.0, thickness / n],
        [0.0, 1.0],
    ];
    let refraction_r2: RayMatrix = [
        [1.0, 0.0],
        [(n - 1.0) / r2, 1.0],
    ];

    // System matrix = R2 × P × R1 (right to left)
    let rp = multiply_ray_matrices(&propagation, &refraction_r1);
    let system = multiply_ray_matrices(&refraction_r2, &rp);

    // Effective focal length: f = -1 / C  (system matrix element [1][0])
    let f_eff = -1.0 / system[1][0];

    // Standard thin-lens imaging equation: 1/f = 1/do + 1/di → di = f*do/(do - f)
    f_eff * object_dist / (object_dist - f_eff)
}

// ── Interference & Coherence ────────────────────────────────────────────────

/// Temporal coherence length: L_c = λ²/Δλ
pub fn coherence_length(wavelength: f64, bandwidth: f64) -> f64 {
    assert!(bandwidth > 0.0, "bandwidth must be positive");
    wavelength * wavelength / bandwidth
}

/// Coherence time from frequency bandwidth: τ_c = 1/Δf
pub fn coherence_time(bandwidth_hz: f64) -> f64 {
    assert!(bandwidth_hz > 0.0, "bandwidth_hz must be positive");
    1.0 / bandwidth_hz
}

/// Fringe visibility (contrast): V = (I_max - I_min) / (I_max + I_min)
pub fn fringe_visibility(i_max: f64, i_min: f64) -> f64 {
    assert!(i_max + i_min != 0.0, "i_max + i_min must not be zero");
    (i_max - i_min) / (i_max + i_min)
}

/// Fabry-Perot etalon transmission (Airy function):
/// T = (1-R)² / ((1-R)² + 4R sin²(δ/2))
pub fn fabry_perot_transmission(reflectance: f64, phase: f64) -> f64 {
    let one_minus_r = 1.0 - reflectance;
    let one_minus_r_sq = one_minus_r * one_minus_r;
    let sin_half = (phase / 2.0).sin();
    one_minus_r_sq / (one_minus_r_sq + 4.0 * reflectance * sin_half * sin_half)
}

/// Free spectral range of a Fabry-Perot cavity: FSR = c/(2nL) in Hz.
pub fn free_spectral_range(cavity_length: f64, n: f64) -> f64 {
    assert!(cavity_length > 0.0, "cavity_length must be positive");
    assert!(n > 0.0, "refractive index must be positive");
    constants::C / (2.0 * n * cavity_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn rel_approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() / b.abs() < tol
    }

    // ── Gaussian beam tests ─────────────────────────────────────────────

    #[test]
    fn test_beam_waist_divergence_roundtrip() {
        let wavelength = 1.064e-6;
        let w0 = 50.0e-6;
        let theta = beam_divergence(w0, wavelength);
        let w0_back = beam_waist_from_divergence(wavelength, theta);
        assert!(rel_approx(w0, w0_back, 1e-12));
    }

    #[test]
    fn test_rayleigh_range_known() {
        let w0 = 100.0e-6;
        let wavelength = 632.8e-9;
        let z_r = rayleigh_range(w0, wavelength);
        // π × (100×10⁻⁶)² / 632.8×10⁻⁹ ≈ 0.04964591…
        assert!(rel_approx(z_r, 0.049_645_930_057_553, 1e-6));
    }

    #[test]
    fn test_beam_radius_at_zero_equals_waist() {
        let w0 = 50.0e-6;
        let z_r = 1.0;
        assert!(approx(beam_radius(w0, 0.0, z_r), w0));
    }

    #[test]
    fn test_beam_radius_at_rayleigh_range() {
        let w0 = 50.0e-6;
        let z_r = 0.01;
        let w_at_zr = beam_radius(w0, z_r, z_r);
        assert!(rel_approx(w_at_zr, 7.071_067_811_865_476e-5, 1e-12));
    }

    #[test]
    fn test_gouy_phase_at_rayleigh() {
        let z_r = 0.01;
        let psi = gouy_phase(z_r, z_r);
        assert!(rel_approx(psi, constants::PI / 4.0, 1e-12));
    }

    #[test]
    fn test_beam_intensity_on_axis() {
        let power = 1.0;
        let w0 = 100.0e-6;
        let z_r = rayleigh_range(w0, 632.8e-9);
        let i_peak = beam_intensity(power, w0, 0.0, 0.0, z_r);
        assert!(rel_approx(i_peak, 6.366_197_723_675_814e7, 1e-9));
    }

    #[test]
    fn test_beam_parameter_returns_pair() {
        let w0 = 50.0e-6;
        let z_r = 0.01;
        let z = 0.005;
        let (w, r_curv) = beam_parameter(w0, z, z_r);
        assert!(approx(w, beam_radius(w0, z, z_r)));
        assert!(approx(r_curv, beam_curvature(z, z_r)));
    }

    // ── Fiber optics tests ──────────────────────────────────────────────

    #[test]
    fn test_numerical_aperture() {
        let na = numerical_aperture(1.48, 1.46);
        assert!(rel_approx(na, 0.242_487_113_059_643, 1e-12));
    }

    #[test]
    fn test_acceptance_angle() {
        let na = 0.22;
        let theta = acceptance_angle(na);
        assert!(rel_approx(theta.sin(), na, 1e-12));
    }

    #[test]
    fn test_single_mode_below_cutoff() {
        assert!(is_single_mode(2.0));
        assert!(!is_single_mode(3.0));
        assert!(!is_single_mode(SINGLE_MODE_CUTOFF));
    }

    #[test]
    fn test_number_of_modes() {
        let v = 30.0;
        assert!(approx(number_of_modes(v), 450.0));
    }

    #[test]
    fn test_fiber_attenuation_no_loss() {
        let p_out = fiber_attenuation(1.0, 0.0, 100.0);
        assert!(approx(p_out, 1.0));
    }

    #[test]
    fn test_fiber_attenuation_10db() {
        let p_out = fiber_attenuation(1.0, 10.0, 1.0);
        assert!(rel_approx(p_out, 0.1, 1e-9));
    }

    #[test]
    fn test_critical_angle_fiber() {
        let theta = critical_angle_fiber(1.48, 1.46);
        assert!(rel_approx(theta, 1.406_211_640_313_002, 1e-6));
    }

    // ── ABCD matrix tests ───────────────────────────────────────────────

    #[test]
    fn test_thin_lens_matrix_structure() {
        let f = 0.1;
        let m = thin_lens_matrix(f);
        assert!(approx(m[0][0], 1.0));
        assert!(approx(m[0][1], 0.0));
        assert!(approx(m[1][0], -10.0));
        assert!(approx(m[1][1], 1.0));
    }

    #[test]
    fn test_free_space_matrix_structure() {
        let d = 0.5;
        let m = free_space_matrix(d);
        assert!(approx(m[0][0], 1.0));
        assert!(approx(m[0][1], 0.5));
        assert!(approx(m[1][0], 0.0));
        assert!(approx(m[1][1], 1.0));
    }

    #[test]
    fn test_multiply_identity() {
        let identity: RayMatrix = [[1.0, 0.0], [0.0, 1.0]];
        let lens = thin_lens_matrix(0.2);
        let result = multiply_ray_matrices(&identity, &lens);
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx(result[i][j], lens[i][j]));
            }
        }
    }

    #[test]
    fn test_apply_ray_matrix_free_space() {
        let d = 1.0;
        let m = free_space_matrix(d);
        let (y, theta) = apply_ray_matrix(&m, 0.0, 0.1);
        assert!(approx(y, 0.1));
        assert!(approx(theta, 0.1));
    }

    #[test]
    fn test_thick_lens_thin_limit() {
        // A thick lens with zero thickness should match the lensmaker's equation.
        let n = 1.5;
        let r1 = 0.1;
        let r2 = -0.1;
        let f_thin = 1.0 / ((n - 1.0) * (1.0 / r1 - 1.0 / r2));
        let obj = 0.5;
        let di_thin = f_thin * obj / (obj - f_thin);
        let di_thick = image_distance_thick_lens(n, r1, r2, 0.0, obj);
        assert!(rel_approx(di_thick, di_thin, 1e-9));
    }

    // ── Interference & coherence tests ──────────────────────────────────

    #[test]
    fn test_coherence_length() {
        let lc = coherence_length(632.8e-9, 0.1e-9);
        assert!(rel_approx(lc, 4.004_358_4e-3, 1e-9));
    }

    #[test]
    fn test_coherence_time() {
        let tau = coherence_time(1e9);
        assert!(approx(tau, 1e-9));
    }

    #[test]
    fn test_fringe_visibility_perfect() {
        let v = fringe_visibility(1.0, 0.0);
        assert!(approx(v, 1.0));
    }

    #[test]
    fn test_fringe_visibility_symmetric() {
        let v = fringe_visibility(0.8, 0.2);
        assert!(approx(v, 0.6));
    }

    #[test]
    fn test_fabry_perot_resonance() {
        // At resonance (δ = 0), transmission should be 1.0 regardless of reflectance.
        let t = fabry_perot_transmission(0.9, 0.0);
        assert!(rel_approx(t, 1.0, 1e-12));
    }

    #[test]
    fn test_fabry_perot_anti_resonance() {
        // At anti-resonance (δ = π), high reflectance means low transmission.
        let r = 0.9;
        let t = fabry_perot_transmission(r, constants::PI);
        assert!(rel_approx(t, 0.002_770_083_102_493_075, 1e-6));
    }

    #[test]
    fn test_free_spectral_range() {
        let fsr = free_spectral_range(0.01, 1.0);
        assert!(rel_approx(fsr, 1.498_962_29e10, 1e-12));
    }

    #[test]
    fn test_dispersion_broadening() {
        // D=17 ps/(nm*km), L=100 km, Δλ=0.1 nm => Δt = 17 * 100 * 0.1 = 170 ps
        let dt = dispersion_broadening(17.0, 100.0, 0.1);
        assert!(approx(dt, 170.0), "got {dt}");
    }

    #[test]
    fn test_v_number_known() {
        // V = 2π * r * NA / λ
        let radius = 4.0e-6;
        let na = 0.12;
        let wavelength = 1.3e-6;
        let v = v_number(radius, na, wavelength);
        assert!(rel_approx(v, 2.319_945_344_189_386, 1e-6));
    }

    #[test]
    fn test_v_number_single_mode_check() {
        // A typical single-mode fiber at 1550nm: r~4.1um, NA~0.12
        let v = v_number(4.1e-6, 0.12, 1.55e-6);
        assert!(is_single_mode(v), "V={v} should be single mode");
    }
}
