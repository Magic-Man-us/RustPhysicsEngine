//! Guide, chapter 2: a spacecraft in orbit.
//!
//! Run with `cargo run --example guide_02_orbit`. This file is the source
//! for that chapter of `docs/GUIDE.md`, and CI runs it, so the guide
//! cannot describe code that does not work.

use rust_physics_engine::astrophysics::orbital_elements::OrbitalElements;
use rust_physics_engine::math::constants::{EARTH_MASS, EARTH_RADIUS, G};
use rust_physics_engine::math::Vec3;
use rust_physics_engine::propulsion::hohmann_delta_v;

fn main() {
    // The gravitational parameter is what orbital mechanics actually uses;
    // G and M never appear apart.
    let mu = G * EARTH_MASS;

    // A circular orbit 400 km up, near enough the ISS.
    let r = EARTH_RADIUS + 400e3;
    let speed = (mu / r).sqrt();
    let position = Vec3::new(r, 0.0, 0.0);
    let velocity = Vec3::new(0.0, speed, 0.0);

    // Going from a state vector to elements is the first thing you do with
    // tracking data, because elements are what you can reason about.
    let elements = OrbitalElements::from_state_vectors(position, velocity, mu);
    println!("circular orbit at 400 km");
    println!("  speed          {:.0} m/s", speed);
    println!("  semi-major     {:.1} km", elements.semi_major_axis / 1e3);
    println!("  eccentricity   {:.2e}", elements.eccentricity);
    println!("  period         {:.1} min", elements.period(mu) / 60.0);
    println!("  bound?         {}", elements.is_bound());

    // A circular orbit has e = 0 to within rounding, and its period is
    // Kepler's third law. Both are worth asserting rather than eyeballing.
    assert!(elements.eccentricity < 1e-12);
    let kepler = 2.0 * std::f64::consts::PI * (r.powi(3) / mu).sqrt();
    assert!((elements.period(mu) - kepler).abs() < 1e-6);

    // Now raise it to geostationary. A Hohmann transfer is two burns: one
    // to enter an ellipse that touches both circles, one to circularise.
    let r_geo = 42_164e3;
    let (dv1, dv2) = hohmann_delta_v(mu, r, r_geo);
    println!();
    println!("Hohmann transfer to geostationary");
    println!("  burn 1         {:.0} m/s", dv1);
    println!("  burn 2         {:.0} m/s", dv2);
    println!("  total          {:.0} m/s", dv1 + dv2);

    // Both burns are prograde, and the first is the larger of the two --
    // it does most of the work of raising the apoapsis.
    assert!(dv1 > 0.0 && dv2 > 0.0);
    assert!(dv1 > dv2);

    // The transfer ellipse touches both circles, so its semi-major axis is
    // the mean of the two radii and its period follows. Half of that is the
    // flight time.
    let a_transfer = 0.5 * (r + r_geo);
    let transfer_time = std::f64::consts::PI * (a_transfer.powi(3) / mu).sqrt();
    println!("  flight time    {:.1} hours", transfer_time / 3600.0);

    // Check the ellipse really does touch both circles, via vis-viva:
    // v² = μ(2/r − 1/a). The speed at its low point is the circular speed
    // plus the first burn.
    let v_peri = (mu * (2.0 / r - 1.0 / a_transfer)).sqrt();
    assert!((v_peri - (speed + dv1)).abs() < 1e-6);
    println!();
    println!("vis-viva at perigee agrees with circular speed + burn 1");
}
