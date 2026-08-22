//! Kani harnesses for `crate::spatial`.

use crate::math::Vec3;
use crate::spatial::intersect::ray_aabb;
use crate::spatial::primitives::{Aabb, Ray};

/// `ray_aabb` never reports an interval with t_enter > t_exit.
#[kani::proof]
fn ray_aabb_interval_ordered() {
    let ox: f64 = kani::any();
    let oy: f64 = kani::any();
    let oz: f64 = kani::any();
    let dx: f64 = kani::any();
    let dy: f64 = kani::any();
    let dz: f64 = kani::any();
    kani::assume(ox.is_finite() && oy.is_finite() && oz.is_finite());
    kani::assume(dx.is_finite() && dy.is_finite() && dz.is_finite());
    kani::assume(ox.abs() < 1e50 && oy.abs() < 1e50 && oz.abs() < 1e50);
    let d = Vec3::new(dx, dy, dz);
    kani::assume(d.magnitude_squared() > 1e-12 && d.magnitude_squared() < 1e50);
    let lo_x: f64 = kani::any();
    let lo_y: f64 = kani::any();
    let lo_z: f64 = kani::any();
    let ex: f64 = kani::any();
    let ey: f64 = kani::any();
    let ez: f64 = kani::any();
    kani::assume(lo_x.is_finite() && lo_y.is_finite() && lo_z.is_finite());
    kani::assume(lo_x.abs() < 1e50 && lo_y.abs() < 1e50 && lo_z.abs() < 1e50);
    kani::assume(ex.is_finite() && ex >= 0.0 && ex < 1e50);
    kani::assume(ey.is_finite() && ey >= 0.0 && ey < 1e50);
    kani::assume(ez.is_finite() && ez >= 0.0 && ez < 1e50);
    let min = Vec3::new(lo_x, lo_y, lo_z);
    let b = Aabb { min, max: min + Vec3::new(ex, ey, ez) };
    let r = Ray::new(Vec3::new(ox, oy, oz), d);
    if let Some((t_enter, t_exit)) = ray_aabb(&r, &b) {
        assert!(t_enter <= t_exit);
        assert!(t_enter >= 0.0);
    }
}
