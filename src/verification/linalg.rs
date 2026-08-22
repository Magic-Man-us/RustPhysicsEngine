//! Kani harnesses for `crate::linalg`.

use crate::linalg::{lu_decompose, Matrix};

/// `lu_decompose` on a symbolic 3×3 matrix with finite entries either
/// returns `Err` or a factorization whose entries are all finite.
#[kani::proof]
#[kani::unwind(6)]
fn lu_decompose_3x3_finite_or_err() {
    let mut data = Vec::with_capacity(9);
    for _ in 0..9 {
        let v: f64 = kani::any();
        kani::assume(v.is_finite());
        kani::assume(v.abs() < 1e100);
        data.push(v);
    }
    let a = Matrix { rows: 3, cols: 3, data };
    if let Ok(f) = lu_decompose(&a) {
        for &v in &f.lu.data {
            assert!(v.is_finite());
        }
    }
}
