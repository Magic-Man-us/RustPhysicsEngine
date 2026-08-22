use std::ops::{Add, Sub, Mul, Neg};

/// 3D vector used throughout the physics engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    /// Constructs a new 3D vector from x, y, z components.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Computes the Euclidean length of this vector: |v| = sqrt(x² + y² + z²).
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Computes the squared length of this vector: x² + y² + z² (avoids a sqrt).
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Returns the unit vector in the same direction: v / |v|. Returns ZERO for zero-length vectors.
    pub fn normalized(&self) -> Self {
        let m = self.magnitude();
        if m == 0.0 {
            return Vec3::ZERO;
        }
        *self * (1.0 / m)
    }

    /// Computes the dot product of two vectors: a · b = ax*bx + ay*by + az*bz.
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product of two vectors: a × b, yielding a vector perpendicular to both.
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Computes the Euclidean distance between two points: |self - other|.
    pub fn distance_to(&self, other: &Vec3) -> f64 {
        (*self - *other).magnitude()
    }

    /// Computes the angle in radians between two vectors: θ = acos((a · b) / (|a| |b|)).
    pub fn angle_between(&self, other: &Vec3) -> f64 {
        let d = self.dot(other);
        let m = self.magnitude() * other.magnitude();
        if m == 0.0 {
            return 0.0;
        }
        (d / m).clamp(-1.0, 1.0).acos()
    }

    /// Linearly interpolates between two vectors: result = self*(1-t) + other*t.
    pub fn lerp(&self, other: &Vec3, t: f64) -> Vec3 {
        *self * (1.0 - t) + *other * t
    }

    /// Projects this vector onto another: proj_b(a) = b * (a · b) / (b · b).
    pub fn project_onto(&self, other: &Vec3) -> Vec3 {
        let d = other.magnitude_squared();
        if d == 0.0 {
            return Vec3::ZERO;
        }
        *other * (self.dot(other) / d)
    }

    /// Reflects this vector about a surface normal: r = v - 2(v · n)n.
    pub fn reflect(&self, normal: &Vec3) -> Vec3 {
        *self - *normal * (2.0 * self.dot(normal))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// 2-D vector of f64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    /// Creates a vector from components.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean length: √(x² + y²).
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Squared length: x² + y².
    #[must_use]
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Unit vector in the same direction (ZERO stays ZERO).
    #[must_use]
    pub fn normalized(&self) -> Self {
        let m = self.magnitude();
        if m == 0.0 {
            Self::ZERO
        } else {
            Self::new(self.x / m, self.y / m)
        }
    }

    /// Dot product: x₁x₂ + y₁y₂.
    #[must_use]
    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2-D cross product (z of the 3-D cross): x₁y₂ − y₁x₂.
    #[must_use]
    pub fn cross(&self, other: &Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Counter-clockwise perpendicular: (−y, x).
    #[must_use]
    pub fn perp(&self) -> Vec2 {
        Vec2::new(-self.y, self.x)
    }

    /// Distance to another point.
    #[must_use]
    pub fn distance_to(&self, other: &Vec2) -> f64 {
        (*self - *other).magnitude()
    }

    /// Unsigned angle to another vector in [0, π].
    #[must_use]
    pub fn angle_between(&self, other: &Vec2) -> f64 {
        let denom = self.magnitude() * other.magnitude();
        if denom == 0.0 {
            return 0.0;
        }
        (self.dot(other) / denom).clamp(-1.0, 1.0).acos()
    }

    /// Linear interpolation: self + t·(other − self).
    #[must_use]
    pub fn lerp(&self, other: &Vec2, t: f64) -> Vec2 {
        Vec2::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
        )
    }

    /// Rotation by `angle` radians counter-clockwise about the origin.
    #[must_use]
    pub fn rotate(&self, angle: f64) -> Vec2 {
        let (s, c) = angle.sin_cos();
        Vec2::new(c * self.x - s * self.y, s * self.x + c * self.y)
    }

    /// Embedding into 3-D with z = 0.
    #[must_use]
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, 0.0)
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f64) -> Vec2 {
        Vec2::new(self.x * rhs, self.y * rhs)
    }
}

impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}

/// Physical and mathematical constants (NIST CODATA 2018 / 2019 SI redefinition).
pub mod constants {
    // ── Mathematical constants ──────────────────────────────────────────

    /// Pi (π)
    pub const PI: f64 = std::f64::consts::PI;
    /// Tau (2π)
    pub const TAU: f64 = 2.0 * std::f64::consts::PI;
    /// Euler's number (e)
    pub const E: f64 = std::f64::consts::E;
    /// Square root of 2
    pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
    /// Natural logarithm of 2
    pub const LN_2: f64 = std::f64::consts::LN_2;
    /// Natural logarithm of 10
    pub const LN_10: f64 = std::f64::consts::LN_10;

    // ── Fundamental constants ───────────────────────────────────────────

    /// Speed of light in vacuum (m/s) — exact
    pub const C: f64 = 299_792_458.0;
    /// Gravitational constant (m³ kg⁻¹ s⁻²) — ±0.000_15e-11
    pub const G: f64 = 6.674_30e-11;
    /// Planck's constant (J·s) — exact (2019 SI redefinition)
    pub const H: f64 = 6.626_070_15e-34;
    /// Reduced Planck's constant ℏ = h/(2π) (J·s) — exact
    pub const HBAR: f64 = 1.054_571_817e-34;
    /// Boltzmann constant (J/K) — exact (2019 SI redefinition)
    pub const K_B: f64 = 1.380_649e-23;
    /// Elementary charge (C) — exact (2019 SI redefinition)
    pub const E_CHARGE: f64 = 1.602_176_634e-19;
    /// Avogadro constant (mol⁻¹) — exact (2019 SI redefinition)
    pub const N_A: f64 = 6.022_140_76e23;
    /// Molar gas constant R = N_A × k_B (J mol⁻¹ K⁻¹) — exact
    pub const R: f64 = 8.314_462_618;
    /// Standard gravitational acceleration (m/s²) — exact by definition
    pub const G_ACCEL: f64 = 9.806_65;

    // ── Particle masses ─────────────────────────────────────────────────

    /// Electron mass (kg)
    pub const M_ELECTRON: f64 = 9.109_383_7015e-31;
    /// Proton mass (kg)
    pub const M_PROTON: f64 = 1.672_621_923_69e-27;
    /// Neutron mass (kg)
    pub const M_NEUTRON: f64 = 1.674_927_498_04e-27;
    /// Atomic mass unit / dalton (kg)
    pub const AMU: f64 = 1.660_539_066_60e-27;

    // ── Electromagnetic constants ───────────────────────────────────────

    /// Vacuum permittivity ε₀ (F/m)
    pub const EPSILON_0: f64 = 8.854_187_8128e-12;
    /// Vacuum permeability μ₀ (H/m)
    pub const MU_0: f64 = 1.256_637_062_12e-6;
    /// Coulomb constant k_e = 1/(4πε₀) (N·m²/C²)
    pub const K_E: f64 = 8.987_551_7923e9;
    /// Impedance of free space Z₀ = μ₀c (Ω)
    pub const VACUUM_IMPEDANCE: f64 = 376.730_313_668;
    /// Magnetic flux quantum Φ₀ = h/(2e) (Wb)
    pub const MAGNETIC_FLUX_QUANTUM: f64 = 2.067_833_848e-15;
    /// Conductance quantum G₀ = 2e²/h (S)
    pub const CONDUCTANCE_QUANTUM: f64 = 7.748_091_729e-5;
    /// Von Klitzing constant R_K = h/e² (Ω)
    pub const VON_KLITZING: f64 = 25_812.807_45;
    /// Josephson constant K_J = 2e/h (Hz/V)
    pub const JOSEPHSON: f64 = 483_597.848_4e9;

    // ── Thermodynamic constants ─────────────────────────────────────────

    /// Stefan-Boltzmann constant σ (W m⁻² K⁻⁴)
    pub const SIGMA: f64 = 5.670_374_419e-8;
    /// Wien displacement law constant b (m·K)
    pub const WIEN_DISPLACEMENT: f64 = 2.897_771_955e-3;
    /// First radiation constant c₁ = 2πhc² (W·m²)
    pub const FIRST_RADIATION: f64 = 3.741_771_852e-16;
    /// Second radiation constant c₂ = hc/k_B (m·K)
    pub const SECOND_RADIATION: f64 = 1.438_776_877e-2;

    // ── Atomic & nuclear constants ──────────────────────────────────────

    /// Rydberg constant R∞ (m⁻¹)
    pub const RYDBERG: f64 = 1.097_373_156_8160e7;
    /// Rydberg energy (J) ≈ 13.6 eV
    pub const RYDBERG_ENERGY: f64 = 2.179_872_361_1035e-18;
    /// Bohr radius a₀ (m)
    pub const BOHR_RADIUS: f64 = 5.291_772_109_03e-11;
    /// Bohr magneton μ_B (J/T)
    pub const BOHR_MAGNETON: f64 = 9.274_010_0783e-24;
    /// Nuclear magneton μ_N (J/T)
    pub const NUCLEAR_MAGNETON: f64 = 5.050_783_7461e-27;
    /// Fine-structure constant α ≈ 1/137
    pub const ALPHA: f64 = 7.297_352_5693e-3;
    /// Inverse fine-structure constant 1/α
    pub const ALPHA_INV: f64 = 137.035_999_084;

    // ── Planck units ────────────────────────────────────────────────────

    /// Planck mass √(ℏc/G) (kg)
    pub const PLANCK_MASS: f64 = 2.176_434e-8;
    /// Planck length √(ℏG/c³) (m)
    pub const PLANCK_LENGTH: f64 = 1.616_255e-35;
    /// Planck time √(ℏG/c⁵) (s)
    pub const PLANCK_TIME: f64 = 5.391_247e-44;
    /// Planck temperature √(ℏc⁵/(G·k_B²)) (K)
    pub const PLANCK_TEMPERATURE: f64 = 1.416_784e32;
    /// Planck charge √(4πε₀ℏc) (C)
    pub const PLANCK_CHARGE: f64 = 1.875_546e-18;

    // ── Astronomical constants ──────────────────────────────────────────

    /// Solar mass M☉ (kg)
    pub const SOLAR_MASS: f64 = 1.989e30;
    /// Solar radius R☉ (m)
    pub const SOLAR_RADIUS: f64 = 6.957e8;
    /// Solar luminosity L☉ (W)
    pub const SOLAR_LUMINOSITY: f64 = 3.828e26;
    /// Solar effective temperature T☉ (K)
    pub const SOLAR_TEMPERATURE: f64 = 5778.0;
    /// Earth mass M⊕ (kg)
    pub const EARTH_MASS: f64 = 5.972_17e24;
    /// Earth mean radius R⊕ (m)
    pub const EARTH_RADIUS: f64 = 6.371e6;
    /// Mean Earth-Moon distance (m)
    pub const EARTH_MOON_DISTANCE: f64 = 3.844e8;
    /// Astronomical unit (m) — exact (IAU 2012)
    pub const AU: f64 = 1.495_978_707e11;
    /// Light-year (m) — exact
    pub const LIGHT_YEAR: f64 = 9.460_730_472_5808e15;
    /// Parsec (m)
    pub const PARSEC: f64 = 3.085_677_581e16;
    /// Hubble constant H₀ ≈ 69.8 km/s/Mpc (s⁻¹)
    pub const HUBBLE: f64 = 2.25e-18;
    /// Cosmic microwave background temperature (K)
    pub const CMB_TEMPERATURE: f64 = 2.725_5;

    // ── Conversion factors ──────────────────────────────────────────────

    /// Electron-volt to joules (J/eV) — same value as E_CHARGE
    pub const EV_TO_JOULES: f64 = 1.602_176_634e-19;
    /// Thermochemical calorie (J)
    pub const CALORIE: f64 = 4.184;
    /// Standard atmosphere (Pa) — exact
    pub const ATM: f64 = 101_325.0;
    /// Torr (Pa)
    pub const TORR: f64 = 133.322_387_415;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_vec3_magnitude() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(v.magnitude(), 5.0));
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx(a.dot(&b), 32.0));
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!(approx(c.x, 0.0) && approx(c.y, 0.0) && approx(c.z, 1.0));
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(0.0, 3.0, 4.0);
        let n = v.normalized();
        assert!(approx(n.magnitude(), 1.0));
    }

    #[test]
    fn test_vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let r = v.reflect(&normal);
        assert!(approx(r.x, 1.0) && approx(r.y, 1.0) && approx(r.z, 0.0));
    }

    fn rel_error(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn test_hbar_equals_h_over_2pi() {
        let derived = constants::H / (2.0 * constants::PI);
        assert!(rel_error(constants::HBAR, derived) < 1e-9);
    }

    #[test]
    fn test_coulomb_constant_from_epsilon0() {
        let derived = 1.0 / (4.0 * constants::PI * constants::EPSILON_0);
        assert!(rel_error(constants::K_E, derived) < 1e-6);
    }

    #[test]
    fn test_vacuum_impedance_equals_mu0_times_c() {
        let derived = constants::MU_0 * constants::C;
        assert!(rel_error(constants::VACUUM_IMPEDANCE, derived) < 1e-6);
    }

    #[test]
    fn test_gas_constant_equals_na_times_kb() {
        let derived = constants::N_A * constants::K_B;
        assert!(rel_error(constants::R, derived) < 1e-9);
    }

    #[test]
    fn test_fine_structure_constant() {
        let derived = constants::E_CHARGE.powi(2)
            / (4.0 * constants::PI * constants::EPSILON_0 * constants::HBAR * constants::C);
        assert!(rel_error(constants::ALPHA, derived) < 1e-6);
    }

    #[test]
    fn test_magnitude_squared() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(v.magnitude_squared(), 25.0));
    }

    #[test]
    fn test_distance_to() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 3.0);
        // distance = sqrt(9 + 16 + 0) = 5
        assert!(approx(a.distance_to(&b), 5.0));
    }

    #[test]
    fn test_angle_between_perpendicular() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let angle = a.angle_between(&b);
        assert!(approx(angle, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn test_angle_between_parallel() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 0.0, 0.0);
        assert!(approx(a.angle_between(&b), 0.0));
    }

    #[test]
    fn test_lerp_endpoints() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 20.0, 30.0);
        let at0 = a.lerp(&b, 0.0);
        let at1 = a.lerp(&b, 1.0);
        assert!(approx(at0.x, 0.0) && approx(at0.y, 0.0) && approx(at0.z, 0.0));
        assert!(approx(at1.x, 10.0) && approx(at1.y, 20.0) && approx(at1.z, 30.0));
    }

    #[test]
    fn test_lerp_midpoint() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 20.0, 30.0);
        let mid = a.lerp(&b, 0.5);
        assert!(approx(mid.x, 5.0) && approx(mid.y, 10.0) && approx(mid.z, 15.0));
    }

    #[test]
    fn test_project_onto() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let proj = a.project_onto(&b);
        assert!(approx(proj.x, 3.0) && approx(proj.y, 0.0) && approx(proj.z, 0.0));
    }

    #[test]
    fn test_project_onto_self() {
        let v = Vec3::new(3.0, 4.0, 5.0);
        let proj = v.project_onto(&v);
        assert!(approx(proj.x, v.x) && approx(proj.y, v.y) && approx(proj.z, v.z));
    }

    #[test]
    fn test_normalized_zero_vector() {
        let v = Vec3::ZERO;
        let n = v.normalized();
        assert!(approx(n.x, 0.0) && approx(n.y, 0.0) && approx(n.z, 0.0));
    }

    #[test]
    fn test_angle_between_zero_vector() {
        let a = Vec3::ZERO;
        let b = Vec3::new(1.0, 0.0, 0.0);
        let angle = a.angle_between(&b);
        assert!(approx(angle, 0.0));
    }

    #[test]
    fn test_project_onto_zero_vector() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let zero = Vec3::ZERO;
        let proj = a.project_onto(&zero);
        assert!(approx(proj.x, 0.0) && approx(proj.y, 0.0) && approx(proj.z, 0.0));
    }
}

#[cfg(test)]
mod vec2_tests {
    use super::*;
    use constants::PI;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn test_basics() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx(v.magnitude(), 5.0));
        assert!(approx(v.magnitude_squared(), 25.0));
        let n = v.normalized();
        assert!(approx(n.magnitude(), 1.0));
        assert_eq!(Vec2::ZERO.normalized(), Vec2::ZERO);
    }

    #[test]
    fn test_dot_cross_perp() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 2.0);
        assert!(approx(a.dot(&b), 0.0));
        assert!(approx(a.cross(&b), 2.0));
        assert!(approx(b.cross(&a), -2.0));
        let p = a.perp();
        assert!(approx(p.x, 0.0) && approx(p.y, 1.0));
        assert!(approx(a.dot(&p), 0.0));
    }

    #[test]
    fn test_rotate_and_angle() {
        let v = Vec2::new(1.0, 0.0).rotate(PI / 2.0);
        assert!(approx(v.x, 0.0) && approx(v.y, 1.0));
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 3.0);
        assert!(approx(a.angle_between(&b), PI / 2.0));
        assert!(approx(a.angle_between(&Vec2::ZERO), 0.0));
    }

    #[test]
    fn test_lerp_distance_ops() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 4.0);
        let m = a.lerp(&b, 0.5);
        assert!(approx(m.x, 1.0) && approx(m.y, 2.0));
        assert!(approx(a.distance_to(&b), 20.0_f64.sqrt()));
        let s = a + b - b;
        assert!(approx(s.x, 0.0) && approx(s.y, 0.0));
        let d = -(b * 0.5);
        assert!(approx(d.x, -1.0) && approx(d.y, -2.0));
        assert_eq!(b.to_vec3(), Vec3::new(2.0, 4.0, 0.0));
    }
}
