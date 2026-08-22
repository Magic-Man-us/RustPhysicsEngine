//! Properties for `linalg`: Matrix algebra, LU, and Cholesky.

use rust_physics_engine::linalg::{
    cholesky, eigen_symmetric, eigenvalues_general, lu_decompose, qr_householder, solve, svd,
    thomas_solve, Mat3, Matrix, Qr, Svd,
};
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

/// QᵀQ == I, Q·R == A, and R is upper triangular, for random rectangular A.
#[test]
fn prop_qr_invariants() {
    let mut rng = Rng::new(6);
    for trial in 0..50 {
        let (m, n) = if trial % 2 == 0 { (5, 3) } else { (4, 4) };
        let a = random_matrix(&mut rng, m, n);
        let Qr { q, r } = qr_householder(&a);

        let qtq = q.transpose().mul(&q).unwrap();
        assert!(matrices_close(&qtq, &Matrix::identity(m), 1e-11), "Q not orthogonal");

        let qr = q.mul(&r).unwrap();
        assert!(matrices_close(&qr, &a, 1e-11), "QR != A");

        for i in 1..m {
            for j in 0..i.min(n) {
                assert!(r.get(i, j).abs() <= 1e-12, "R not upper triangular");
            }
        }
    }
}

/// thomas_solve matches lu::solve on the equivalent dense system.
#[test]
fn prop_thomas_matches_dense_lu() {
    let mut rng = Rng::new(8);
    for _ in 0..50 {
        let n = 6;
        let diag: Vec<f64> = (0..n).map(|_| rng.next_f64() + 4.0).collect();
        let sub: Vec<f64> = (0..n - 1).map(|_| rng.next_f64() - 0.5).collect();
        let sup: Vec<f64> = (0..n - 1).map(|_| rng.next_f64() - 0.5).collect();
        let rhs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();

        let mut dense = Matrix::zeros(n, n);
        for i in 0..n {
            dense.set(i, i, diag[i]);
            if i + 1 < n {
                dense.set(i + 1, i, sub[i]);
                dense.set(i, i + 1, sup[i]);
            }
        }
        let x_thomas = thomas_solve(&sub, &diag, &sup, &rhs).unwrap();
        let x_dense = solve(&dense, &rhs).unwrap();
        for (a, b) in x_thomas.iter().zip(&x_dense) {
            assert!((a - b).abs() < 1e-10, "thomas {a} vs dense {b}");
        }
    }
}

/// Symmetric eigen: A·v == λ·v per pair; eigenvalues sum to the trace
/// and multiply to the determinant.
#[test]
fn prop_symmetric_eigen_invariants() {
    let mut rng = Rng::new(31);
    for _ in 0..30 {
        let n = 4;
        let m = random_matrix(&mut rng, n, n);
        // Symmetrize: A = (M + Mᵀ)/2.
        let a = m.add(&m.transpose()).unwrap().scale(0.5);
        let e = eigen_symmetric(&a, 1e-13, 100).unwrap();

        for k in 0..n {
            let v: Vec<f64> = (0..n).map(|r| e.vectors.get(r, k)).collect();
            let av = a.mul_vec(&v).unwrap();
            for r in 0..n {
                assert!((av[r] - e.values[k] * v[r]).abs() < 1e-9, "A v != lambda v");
            }
        }

        let trace: f64 = (0..n).map(|i| a.get(i, i)).sum();
        let sum: f64 = e.values.iter().sum();
        assert!((trace - sum).abs() < 1e-9);

        let det = lu_decompose(&a).map(|f| f.determinant()).unwrap_or(0.0);
        let prod: f64 = e.values.iter().product();
        assert!((det - prod).abs() < 1e-8 * det.abs().max(1.0));
    }
}

/// General eigenvalues sum to the trace (complex parts cancel).
#[test]
fn prop_general_eigenvalues_trace() {
    let mut rng = Rng::new(32);
    for _ in 0..30 {
        let n = 5;
        let a = random_matrix(&mut rng, n, n);
        let eig = eigenvalues_general(&a, 60).unwrap();
        let trace: f64 = (0..n).map(|i| a.get(i, i)).sum();
        let sum_re: f64 = eig.iter().map(|c| c.re).sum();
        let sum_im: f64 = eig.iter().map(|c| c.im).sum();
        assert!((trace - sum_re).abs() < 1e-8, "trace {trace} vs {sum_re}");
        assert!(sum_im.abs() < 1e-8);
    }
}

/// U·Σ·Vᵀ == A; singular values descending and non-negative.
#[test]
fn prop_svd_reconstruction() {
    let mut rng = Rng::new(33);
    for trial in 0..30 {
        let (m, n) = if trial % 2 == 0 { (5, 3) } else { (3, 5) };
        let a = random_matrix(&mut rng, m, n);
        let Svd { u, sigma, vt } = svd(&a).unwrap();

        for w in sigma.windows(2) {
            assert!(w[0] >= w[1], "singular values not descending");
        }
        assert!(sigma.iter().all(|&s| s >= 0.0));

        let k = sigma.len();
        let mut sig = Matrix::zeros(k, k);
        for i in 0..k {
            sig.set(i, i, sigma[i]);
        }
        let back = u.mul(&sig).unwrap().mul(&vt).unwrap();
        assert!(matrices_close(&back, &a, 1e-9), "U S Vt != A");
    }
}

/// CG/PCG drive the residual below tol on an SPD system.
#[test]
fn prop_cg_residual_below_tol() {
    use rust_physics_engine::linalg::{conjugate_gradient, pcg_jacobi, CsrMatrix};
    let mut rng = Rng::new(34);
    for _ in 0..10 {
        let a = CsrMatrix::laplacian_2d(8, 8, 0.25);
        let n = 64;
        let b: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let tol = 1e-10;
        let bnorm = b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        for x in [
            conjugate_gradient(&a, &b, &vec![0.0; n], tol, 10_000).unwrap(),
            pcg_jacobi(&a, &b, tol, 10_000).unwrap(),
        ] {
            let ax = a.mul_vec(&x);
            let res: f64 = ax
                .iter()
                .zip(&b)
                .map(|(axi, bi)| (axi - bi) * (axi - bi))
                .sum::<f64>()
                .sqrt();
            assert!(res <= tol * bnorm * 1.01, "residual {res} above tol");
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
