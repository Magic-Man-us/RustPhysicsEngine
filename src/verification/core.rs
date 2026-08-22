//! Kani harnesses for `crate::core`.

use crate::core::interval::Interval;

/// `Interval::mul` contains all four corner products for symbolic
/// finite operands.
#[kani::proof]
fn interval_mul_contains_corner_products() {
    let a_lo: f64 = kani::any();
    let a_hi: f64 = kani::any();
    let b_lo: f64 = kani::any();
    let b_hi: f64 = kani::any();
    kani::assume(a_lo.is_finite() && a_hi.is_finite() && a_lo <= a_hi);
    kani::assume(b_lo.is_finite() && b_hi.is_finite() && b_lo <= b_hi);
    kani::assume(a_lo.abs() < 1e100 && a_hi.abs() < 1e100);
    kani::assume(b_lo.abs() < 1e100 && b_hi.abs() < 1e100);
    let a = Interval::new(a_lo, a_hi);
    let b = Interval::new(b_lo, b_hi);
    let p = a * b;
    for x in [a_lo, a_hi] {
        for y in [b_lo, b_hi] {
            assert!(p.contains(x * y));
        }
    }
}
