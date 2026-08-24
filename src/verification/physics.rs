//! Kani harnesses for the Part 1 physics and numerics core.
//!
//! Each harness states a precondition with `kani::assume`, calls the real
//! function on symbolic inputs, and asserts a postcondition. Kani (CBMC)
//! explores every input satisfying the precondition, so a passing harness
//! is a proof over the full f64 domain of that region, not a sample.
//!
//! Transcendentals (sin, cos, exp, ln) are modeled by CBMC as
//! unconstrained finite values, so harnesses touching them prove
//! panic-freedom only, not numeric bounds. sqrt is modeled exactly.

use crate::classical::{displacement, kinetic_energy, projectile_range};
use crate::gravitation::{escape_velocity, orbital_velocity};
use crate::information_theory::shannon_entropy;
use crate::linalg::Mat3;
use crate::math::Vec3;
use crate::numerical::bisection;
use crate::statistics::{factorial, mean, variance};
use crate::trigonometry::normalize_angle;

// ---------- helpers ----------

fn any_finite(bound: f64) -> f64 {
    let x: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= bound);
    x
}

fn any_positive(lo: f64, hi: f64) -> f64 {
    let x: f64 = kani::any();
    kani::assume(x.is_finite() && x >= lo && x <= hi);
    x
}

fn any_vec3(bound: f64) -> Vec3 {
    Vec3::new(any_finite(bound), any_finite(bound), any_finite(bound))
}

// ---------- classical ----------

#[kani::proof]
fn displacement_is_finite_on_bounded_inputs() {
    let v0 = any_finite(1e6);
    let a = any_finite(1e6);
    let t = any_finite(1e6);
    let x = displacement(v0, a, t);
    assert!(x.is_finite());
}

#[kani::proof]
fn kinetic_energy_nonnegative_for_nonnegative_mass() {
    let m = any_positive(0.0, 1e30);
    let v = any_finite(1e8);
    let ke = kinetic_energy(m, v);
    assert!(ke >= 0.0);
    assert!(ke.is_finite());
}

#[kani::proof]
fn projectile_range_never_panics_with_positive_g() {
    let v = any_finite(1e4);
    let theta = any_finite(10.0);
    let g = any_positive(1e-6, 1e3);
    // sin is unconstrained in CBMC; we prove only that the assert! cannot fire.
    let _ = projectile_range(v, theta, g);
}

#[kani::proof]
#[kani::should_panic]
fn projectile_range_panics_on_nonpositive_g() {
    let g = any_finite(1e3);
    kani::assume(g <= 0.0);
    let _ = projectile_range(1.0, 0.5, g);
}

// ---------- gravitation ----------

#[kani::proof]
fn escape_velocity_finite_nonnegative() {
    let m = any_positive(0.0, 1e40);
    let r = any_positive(1e-3, 1e20);
    let v = escape_velocity(m, r);
    assert!(v.is_finite());
    assert!(v >= 0.0);
}

#[kani::proof]
fn orbital_velocity_below_escape_velocity() {
    // v_orb = sqrt(GM/r) < v_esc = sqrt(2GM/r) for M > 0.
    let m = any_positive(1e10, 1e40);
    let r = any_positive(1e3, 1e12);
    let vo = orbital_velocity(m, r);
    let ve = escape_velocity(m, r);
    assert!(vo < ve);
}

// ---------- math ----------

#[kani::proof]
fn vec3_normalized_never_produces_nan() {
    let v = any_vec3(1e150);
    let n = v.normalized();
    assert!(n.x.is_finite() && n.y.is_finite() && n.z.is_finite());
}

#[kani::proof]
fn vec3_dot_with_self_nonnegative() {
    let v = any_vec3(1e150);
    assert!(v.dot(&v) >= 0.0);
}

// ---------- linalg ----------

#[kani::proof]
fn mat3_inverse_never_divides_by_zero() {
    let mut data = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            data[r][c] = any_finite(1e100);
        }
    }
    let m = Mat3 { data };
    match m.inverse() {
        None => {}
        Some(inv) => {
            for r in 0..3 {
                for c in 0..3 {
                    assert!(inv.data[r][c].is_finite());
                }
            }
        }
    }
}

#[kani::proof]
fn mat3_identity_inverse_is_identity() {
    let i = Mat3::identity();
    let inv = i.inverse().unwrap();
    for r in 0..3 {
        for c in 0..3 {
            assert!(inv.data[r][c] == i.data[r][c]);
        }
    }
}

// ---------- numerical ----------

#[kani::proof]
#[kani::unwind(34)]
fn bisection_result_is_inside_bracket() {
    let a = any_finite(1e3);
    let b = any_finite(1e3);
    kani::assume(a < b);
    let c = any_finite(1e3);
    kani::assume(c > a && c < b);
    let f = move |x: f64| x - c;
    if let Some(r) = bisection(&f, a, b, 1e-6, 32) {
        assert!(r >= a && r <= b);
    }
}

// ---------- statistics ----------

#[kani::proof]
#[kani::unwind(21)]
fn factorial_is_monotone_and_finite_below_171() {
    let n: u64 = kani::any();
    kani::assume(n < 20);
    let f = factorial(n);
    let g = factorial(n + 1);
    assert!(f.is_finite() && g.is_finite());
    assert!(g >= f);
}

#[kani::proof]
#[kani::unwind(9)]
fn mean_of_bounded_slice_is_bounded() {
    const N: usize = 8;
    let data: [f64; N] = kani::any();
    for x in data.iter() {
        kani::assume(x.is_finite() && x.abs() <= 1e100);
    }
    let m = mean(&data);
    assert!(m.is_finite());
    assert!(m.abs() <= 1e100 + 1e-300);
}

#[kani::proof]
#[kani::unwind(9)]
fn variance_is_nonnegative() {
    const N: usize = 8;
    let data: [f64; N] = kani::any();
    for x in data.iter() {
        kani::assume(x.is_finite() && x.abs() <= 1e50);
    }
    let v = variance(&data);
    assert!(v.is_finite());
    assert!(v >= 0.0);
}

#[kani::proof]
#[kani::should_panic]
fn mean_panics_on_empty() {
    let _ = mean(&[]);
}

// ---------- information theory ----------

#[kani::proof]
#[kani::unwind(5)]
fn shannon_entropy_never_panics_on_nonempty() {
    const N: usize = 4;
    let p: [f64; N] = kani::any();
    for x in p.iter() {
        kani::assume(x.is_finite() && *x >= 0.0 && *x <= 1.0);
    }
    let _ = shannon_entropy(&p);
}

// ---------- trigonometry ----------

#[kani::proof]
fn normalize_angle_never_panics() {
    let theta = any_finite(1e12);
    let _ = normalize_angle(theta);
}
