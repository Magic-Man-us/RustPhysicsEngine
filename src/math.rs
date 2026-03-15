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

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(&self) -> Self {
        let m = self.magnitude();
        if m == 0.0 {
            return Vec3::ZERO;
        }
        *self * (1.0 / m)
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn distance_to(&self, other: &Vec3) -> f64 {
        (*self - *other).magnitude()
    }

    pub fn angle_between(&self, other: &Vec3) -> f64 {
        let d = self.dot(other);
        let m = self.magnitude() * other.magnitude();
        if m == 0.0 {
            return 0.0;
        }
        (d / m).clamp(-1.0, 1.0).acos()
    }

    pub fn lerp(&self, other: &Vec3, t: f64) -> Vec3 {
        *self * (1.0 - t) + *other * t
    }

    pub fn project_onto(&self, other: &Vec3) -> Vec3 {
        let d = other.magnitude_squared();
        if d == 0.0 {
            return Vec3::ZERO;
        }
        *other * (self.dot(other) / d)
    }

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

/// Common physical constants.
pub mod constants {
    /// Speed of light in vacuum (m/s)
    pub const C: f64 = 2.998e8;
    /// Gravitational constant (m^3 kg^-1 s^-2)
    pub const G: f64 = 6.674e-11;
    /// Planck's constant (J·s)
    pub const H: f64 = 6.626e-34;
    /// Reduced Planck's constant (J·s)
    pub const HBAR: f64 = 1.055e-34;
    /// Boltzmann constant (J/K)
    pub const K_B: f64 = 1.381e-23;
    /// Elementary charge (C)
    pub const E_CHARGE: f64 = 1.602e-19;
    /// Electron mass (kg)
    pub const M_ELECTRON: f64 = 9.109e-31;
    /// Proton mass (kg)
    pub const M_PROTON: f64 = 1.673e-27;
    /// Neutron mass (kg)
    pub const M_NEUTRON: f64 = 1.675e-27;
    /// Avogadro's number (mol^-1)
    pub const N_A: f64 = 6.022e23;
    /// Universal gas constant (J/(mol·K))
    pub const R: f64 = 8.314;
    /// Vacuum permittivity (F/m)
    pub const EPSILON_0: f64 = 8.854e-12;
    /// Vacuum permeability (H/m)
    pub const MU_0: f64 = 1.257e-6;
    /// Stefan-Boltzmann constant (W/(m^2·K^4))
    pub const SIGMA: f64 = 5.670e-8;
    /// Coulomb constant (N·m^2/C^2)
    pub const K_E: f64 = 8.988e9;
    /// Standard gravitational acceleration (m/s^2)
    pub const G_ACCEL: f64 = 9.80665;
    /// Atomic mass unit (kg)
    pub const AMU: f64 = 1.661e-27;
    /// Pi
    pub const PI: f64 = std::f64::consts::PI;
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
}
