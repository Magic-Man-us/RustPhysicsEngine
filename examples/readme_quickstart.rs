//! The Quick start snippet from README.md, kept compiling so it cannot rot.
//! Any edit here must be mirrored in the README and vice versa.

use rust_physics_engine::classical::projectile_range;
use rust_physics_engine::exact::rational::Rational;
use rust_physics_engine::math::constants::{C, G};
use rust_physics_engine::units::quantity::{Dim, Quantity};

fn main() {

    // Ballistics: v₀ = 50 m/s, θ = 45°, g = 9.81 m/s²
    let range = projectile_range(50.0, std::f64::consts::FRAC_PI_4, 9.81);
    assert!((range - 254.841_997_961).abs() < 1e-9);

    // Constants come from one table. A black hole's Schwarzschild radius:
    let solar_mass = 1.989e30;
    let r_s = 2.0 * G * solar_mass / (C * C);        // about 2.95 km

    // Quantities carry their dimensions, and addition checks them.
    let v = Quantity::new(3.0, Dim::new(1, 0, -1, 0, 0, 0, 0)); // m/s
    let t = Quantity::new(2.0, Dim::TIME);
    let d = v.mul(&t).unwrap();                      // 6 m — a length, exactly
    assert!(v.add(&t).is_err());                     // a velocity is not a time

    // Exact rational arithmetic over arbitrary-precision integers.
    let third = Rational::from_i64(1, 3);
    let one = third.mul(&Rational::from_i64(3, 1));
    assert_eq!(one, Rational::one());                // not 0.9999999999999999

    println!("range = {range:.6} m");
    println!("r_s   = {:.1} m", r_s);
    println!("d     = {} {}", d.value, d.dim);
}
