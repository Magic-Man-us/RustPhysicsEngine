//! Properties for `linalg`: Matrix algebra, LU, and Cholesky.

use rust_physics_engine::linalg::{cholesky, lu_decompose, solve, Mat3, Matrix};
use rust_physics_engine::monte_carlo::Rng;

fn random_matrix(rng: &mut Rng, rows: usize, cols: usize) -> Matrix {
    let mut m = Matrix::zeros(rows, cols);
    for v in m.data.iter_mut() {
        *v = rng.next_f64() * 2.0 - 1.0;
    }
    m
}

/// Random matrix made safely non-singular by diagonal dominance.
fn random_dominant(rng: &mut Rng, n: usize) -> Matrix {
    let mut m = random_matrix(rng, n, n);
    for i in 0..n {
        let v = m.get(i, i) + n as f64 + 1.0;
        m.set(i, i, v);
    }
    m
}

fn matrices_close(a: &Matrix, b: &Matrix, tol: f64) -> bool {
    a.rows == b.rows
        && a.cols == b.cols
        && a.data.iter().zip(&b.data).all(|(&x, &y)| (x - y).abs() <= tol)
}

/// (A·B)ᵀ == Bᵀ·Aᵀ
#[test]
fn prop_transpose_of_product() {
    let mut rng = Rng::new(1);
    for _ in 0..50 {
        let a = random_matrix(&mut rng, 4, 3);
        let b = random_matrix(&mut rng, 3, 5);
        let lhs = a.mul(&b).unwrap().transpose();
        let rhs = b.transpose().mul(&a.transpose()).unwrap();
        assert!(matrices_close(&lhs, &rhs, 1e-12));
    }
}

/// A·I == A
#[test]
fn prop_identity_is_neutral() {
    let mut rng = Rng::new(2);
    for _ in 0..50 {
        let a = random_matrix(&mut rng, 4, 4);
        let ai = a.mul(&Matrix::identity(4)).unwrap();
        assert!(matrices_close(&a, &ai, 0.0));
    }
}

/// A · solve(A, b) == b for random well-conditioned A.
#[test]
fn prop_lu_solve_residual() {
    let mut rng = Rng::new(3);
    for _ in 0..50 {
        let a = random_dominant(&mut rng, 5);
        let b: Vec<f64> = (0..5).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
        let x = solve(&a, &b).unwrap();
        let back = a.mul_vec(&x).unwrap();
        for (got, want) in back.iter().zip(&b) {
            assert!((got - want).abs() < 1e-9, "residual too large: {got} vs {want}");
        }
    }
}

/// LU determinant matches the Mat3 cofactor determinant on 3×3 input.
#[test]
fn prop_lu_determinant_matches_mat3() {
    let mut rng = Rng::new(4);
    for _ in 0..50 {
        let m3 = Mat3::from_rows(
            [rng.next_f64(), rng.next_f64(), rng.next_f64()],
            [rng.next_f64(), rng.next_f64(), rng.next_f64()],
            [rng.next_f64() + 2.0, rng.next_f64(), rng.next_f64() + 3.0],
        );
        let a = Matrix::from_mat3(&m3);
        match lu_decompose(&a) {
            Ok(f) => {
                let expected = m3.determinant();
                assert!(
                    (f.determinant() - expected).abs() < 1e-10 * expected.abs().max(1.0),
                    "lu det {} vs mat3 det {}",
                    f.determinant(),
                    expected
                );
            }
            Err(_) => {
                assert!(m3.determinant().abs() < 1e-6);
            }
        }
    }
}

/// L·Lᵀ == A for A = M·Mᵀ + n·I (symmetric positive definite by construction).
#[test]
fn prop_cholesky_reconstructs_spd() {
    let mut rng = Rng::new(5);
    for _ in 0..50 {
        let n = 4;
        let m = random_matrix(&mut rng, n, n);
        let a = m
            .mul(&m.transpose())
            .unwrap()
            .add(&Matrix::identity(n).scale(n as f64))
            .unwrap();
        let l = cholesky(&a).unwrap();
        let llt = l.mul(&l.transpose()).unwrap();
        assert!(matrices_close(&llt, &a, 1e-9));
    }
}
