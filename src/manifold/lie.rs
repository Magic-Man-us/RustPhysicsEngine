//! Lie groups and algebras: rotation and rigid-motion groups in 2/3/4
//! dimensions, SU(2) and SL(2) groups, matrix exponentials and logarithms,
//! representation-theory helpers (Wigner d, Clebsch-Gordan), and estimation
//! algorithms on these manifolds (pose graphs, hand-eye, Umeyama).

#![allow(non_camel_case_types)]

use crate::error::SolveError;
use crate::fractals::Complex;
use crate::linalg::{lu_decompose, svd, Mat3, Mat4, Matrix};
use crate::manifold::vecn::TensorN;
use crate::math::{Vec2, Vec3};
use crate::monte_carlo::Rng;
use crate::quaternion::Quaternion;

const EPS_ANGLE: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Matrix functions
// ---------------------------------------------------------------------------

/// Commutator [A, B] = AB - BA.
pub fn lie_bracket_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    let ab = a.mul(b).expect("dimension mismatch");
    let ba = b.mul(a).expect("dimension mismatch");
    Matrix::from_fn(a.rows, a.cols, |i, j| ab.get(i, j) - ba.get(i, j))
}

/// Matrix exponential by scaling-and-squaring with a Taylor/Pade core.
#[must_use]
pub fn matrix_exp(m: &Matrix) -> Matrix {
    let n = m.rows;
    let norm = m.data.iter().map(|v| v.abs()).fold(0.0, f64::max) * n as f64;
    let s = norm.log2().ceil().max(0.0) as u32;
    let scale = 1.0 / 2.0_f64.powi(s as i32);
    let a = m.scale(scale);
    // Taylor series to machine precision on the scaled matrix
    let mut term = Matrix::identity(n);
    let mut sum = Matrix::identity(n);
    for k in 1..=30 {
        term = term.mul(&a).unwrap().scale(1.0 / k as f64);
        sum = sum.add(&term).unwrap();
        if term.frobenius_norm() < 1e-18 * sum.frobenius_norm().max(1.0) {
            break;
        }
    }
    let mut out = sum;
    for _ in 0..s {
        out = out.mul(&out).unwrap();
    }
    out
}

/// Principal matrix square root by the Denman-Beavers iteration.
pub fn matrix_sqrt(m: &Matrix) -> Result<Matrix, SolveError> {
    let n = m.rows;
    let mut y = m.clone();
    let mut z = Matrix::identity(n);
    for _ in 0..80 {
        let y_inv = lu_decompose(&y)?.inverse()?;
        let z_inv = lu_decompose(&z)?.inverse()?;
        let y_next = y.add(&z_inv)?.scale(0.5);
        let z_next = z.add(&y_inv)?.scale(0.5);
        let delta = y_next
            .add(&y.scale(-1.0))
            .unwrap()
            .frobenius_norm();
        y = y_next;
        z = z_next;
        if delta < 1e-14 * y.frobenius_norm().max(1.0) {
            break;
        }
    }
    Ok(y)
}

/// Principal matrix logarithm by inverse scaling-and-squaring with a
/// Gregory series core. Requires eigenvalues off the negative real axis.
pub fn matrix_log(m: &Matrix) -> Result<Matrix, SolveError> {
    let n = m.rows;
    let mut a = m.clone();
    let mut k = 0u32;
    // repeated square roots until close to the identity
    loop {
        let dist = a
            .add(&Matrix::identity(n).scale(-1.0))
            .unwrap()
            .frobenius_norm();
        if dist < 0.3 || k >= 40 {
            break;
        }
        a = matrix_sqrt(&a)?;
        k += 1;
    }
    // log(I + X) with X = A - I via the Gregory (atanh) series:
    // log A = 2 atanh(Z), Z = (A - I)(A + I)^{-1}
    let id = Matrix::identity(n);
    let z = a
        .add(&id.scale(-1.0))
        .unwrap()
        .mul(&lu_decompose(&a.add(&id).unwrap())?.inverse()?)
        .unwrap();
    let z2 = z.mul(&z).unwrap();
    let mut term = z.clone();
    let mut sum = z.clone();
    for j in 1..=40 {
        term = term.mul(&z2).unwrap();
        let add = term.scale(1.0 / (2 * j + 1) as f64);
        sum = sum.add(&add).unwrap();
        if add.frobenius_norm() < 1e-18 {
            break;
        }
    }
    Ok(sum.scale(2.0 * 2.0_f64.powi(k as i32)))
}

/// Killing form B_ij = tr(ad_i ad_j) for a Lie algebra basis of matrices.
#[must_use]
pub fn killing_form(basis: &[Matrix]) -> Matrix {
    let c = structure_constants(basis);
    let n = basis.len();
    Matrix::from_fn(n, n, |i, j| {
        let mut s = 0.0;
        for k in 0..n {
            for l in 0..n {
                s += c.get(&[k, i, l]) * c.get(&[l, j, k]);
            }
        }
        s
    })
}

/// Structure constants c^k_{ij} with [b_i, b_j] = c^k_{ij} b_k, obtained by
/// least squares in the vectorized basis.
#[must_use]
pub fn structure_constants(basis: &[Matrix]) -> TensorN {
    let n = basis.len();
    let dim = basis[0].rows * basis[0].cols;
    // basis Gram matrix under the Frobenius inner product
    let gram = Matrix::from_fn(n, n, |i, j| {
        basis[i]
            .data
            .iter()
            .zip(&basis[j].data)
            .map(|(a, b)| a * b)
            .sum()
    });
    let gram_lu = lu_decompose(&gram).expect("basis must be linearly independent");
    let mut c = TensorN::zeros(&[n, n, n]);
    for i in 0..n {
        for j in 0..n {
            let br = lie_bracket_matrix(&basis[i], &basis[j]);
            let rhs: Vec<f64> = (0..n)
                .map(|k| {
                    basis[k]
                        .data
                        .iter()
                        .zip(&br.data)
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect();
            let coef = gram_lu.solve(&rhs).unwrap();
            for (k, &v) in coef.iter().enumerate() {
                c.set(&[k, i, j], v);
            }
        }
    }
    let _ = dim;
    c
}

// ---------------------------------------------------------------------------
// SO(3)
// ---------------------------------------------------------------------------

/// A 3D rotation stored as a matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So3(pub Mat3);

/// so(3) algebra element (axis-angle vector).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct so3(pub Vec3);

impl So3 {
    #[must_use]
    pub fn identity() -> Self {
        So3(Mat3::identity())
    }

    /// Hat operator: w -> skew-symmetric matrix.
    #[must_use]
    pub fn hat(w: Vec3) -> Mat3 {
        Mat3::from_rows(
            [0.0, -w.z, w.y],
            [w.z, 0.0, -w.x],
            [-w.y, w.x, 0.0],
        )
    }

    /// Vee operator: skew-symmetric matrix -> vector.
    #[must_use]
    pub fn vee(m: &Mat3) -> Vec3 {
        Vec3::new(m.data[2][1], m.data[0][2], m.data[1][0])
    }

    /// Rodrigues exponential.
    #[must_use]
    pub fn exp(w: Vec3) -> Self {
        let th = w.magnitude();
        if th < EPS_ANGLE {
            let k = So3::hat(w);
            let mut m = Mat3::identity() + k;
            m = m + k.mul_mat(&k).mul_scalar(0.5);
            return So3(m);
        }
        let k = So3::hat(w * (1.0 / th));
        let m = Mat3::identity() + k.mul_scalar(th.sin())
            + k.mul_mat(&k).mul_scalar(1.0 - th.cos());
        So3(m)
    }

    /// Logarithm: rotation vector with |w| in [0, pi].
    #[must_use]
    pub fn log(&self) -> Vec3 {
        let m = &self.0;
        let cos_th = ((m.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
        let th = cos_th.acos();
        if th < EPS_ANGLE {
            return So3::vee(&(*m - m.transpose())).scale_half();
        }
        if (std::f64::consts::PI - th) < 1e-6 {
            // near pi: extract axis from the symmetric part
            let a = Mat3::identity() + m.mul_scalar(1.0); // I + R
            // columns of I + R are parallel to the axis
            let mut best = Vec3::new(a.data[0][0], a.data[1][0], a.data[2][0]);
            for c in 1..3 {
                let v = Vec3::new(a.data[0][c], a.data[1][c], a.data[2][c]);
                if v.magnitude() > best.magnitude() {
                    best = v;
                }
            }
            let axis = best.normalized();
            // fix the sign using the skew part
            let skew = So3::vee(&(*m - m.transpose()));
            let axis = if axis.dot(&skew) < 0.0 { axis * -1.0 } else { axis };
            return axis * th;
        }
        So3::vee(&(*m - m.transpose())) * (th / (2.0 * th.sin()))
    }

    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        So3::exp(axis.normalized() * angle)
    }

    #[must_use]
    pub fn from_quat(q: &Quaternion) -> Self {
        let m = q.normalize().to_rotation_matrix();
        So3(Mat3 { data: m })
    }

    #[must_use]
    pub fn to_quat(&self) -> Quaternion {
        let w = self.log();
        let th = w.magnitude();
        if th < EPS_ANGLE {
            return Quaternion::new(1.0, 0.5 * w.x, 0.5 * w.y, 0.5 * w.z).normalize();
        }
        Quaternion::from_axis_angle(w * (1.0 / th), th)
    }

    #[must_use]
    pub fn compose(&self, other: &So3) -> So3 {
        So3(self.0.mul_mat(&other.0))
    }

    #[must_use]
    pub fn inverse(&self) -> So3 {
        So3(self.0.transpose())
    }

    /// Adjoint of SO(3) is the rotation matrix itself.
    #[must_use]
    pub fn adjoint(&self) -> Mat3 {
        self.0
    }

    #[must_use]
    pub fn apply(&self, v: Vec3) -> Vec3 {
        self.0.mul_vec(v)
    }

    /// Rotation angle in [0, pi].
    #[must_use]
    pub fn angle(&self) -> f64 {
        ((self.0.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
    }

    /// Geodesic distance (relative rotation angle).
    #[must_use]
    pub fn distance(&self, other: &So3) -> f64 {
        self.inverse().compose(other).angle()
    }

    /// Geodesic interpolation R exp(t log(R^-1 S)).
    #[must_use]
    pub fn interpolate(&self, other: &So3, t: f64) -> So3 {
        let rel = self.inverse().compose(other).log();
        self.compose(&So3::exp(rel * t))
    }

    /// Uniform random rotation (via random unit quaternion).
    #[must_use]
    pub fn random(rng: &mut Rng) -> So3 {
        let q = Quaternion::new(
            rng.next_gaussian(),
            rng.next_gaussian(),
            rng.next_gaussian(),
            rng.next_gaussian(),
        )
        .normalize();
        So3::from_quat(&q)
    }

    /// Nearest rotation to an arbitrary matrix (polar projection via SVD).
    #[must_use]
    pub fn project(m: &Mat3) -> So3 {
        let a = Matrix::from_mat3(m);
        let s = svd(&a).expect("svd failed");
        let u = s.u;
        let vt = s.vt;
        let mut r = u.mul(&vt).unwrap();
        // fix improper rotation
        let det = lu_decompose(&r).map(|lu| lu.determinant()).unwrap_or(1.0);
        if det < 0.0 {
            // flip the last column of U
            let mut uf = u.clone();
            for i in 0..3 {
                let v = uf.get(i, 2);
                uf.set(i, 2, -v);
            }
            r = uf.mul(&vt).unwrap();
        }
        So3(Mat3 {
            data: [
                [r.get(0, 0), r.get(0, 1), r.get(0, 2)],
                [r.get(1, 0), r.get(1, 1), r.get(1, 2)],
                [r.get(2, 0), r.get(2, 1), r.get(2, 2)],
            ],
        })
    }

    /// Left Jacobian of SO(3).
    #[must_use]
    pub fn left_jacobian(w: Vec3) -> Mat3 {
        let th = w.magnitude();
        let k = So3::hat(w);
        if th < EPS_ANGLE {
            return Mat3::identity() + k.mul_scalar(0.5) + k.mul_mat(&k).mul_scalar(1.0 / 6.0);
        }
        let a = (1.0 - th.cos()) / (th * th);
        let b = (th - th.sin()) / (th * th * th);
        Mat3::identity() + k.mul_scalar(a) + k.mul_mat(&k).mul_scalar(b)
    }

    /// Right Jacobian: J_r(w) = J_l(-w).
    #[must_use]
    pub fn right_jacobian(w: Vec3) -> Mat3 {
        So3::left_jacobian(w * -1.0)
    }

    /// Inverse left Jacobian.
    #[must_use]
    pub fn left_jacobian_inv(w: Vec3) -> Mat3 {
        let th = w.magnitude();
        let k = So3::hat(w);
        if th < EPS_ANGLE {
            return Mat3::identity() + k.mul_scalar(-0.5) + k.mul_mat(&k).mul_scalar(1.0 / 12.0);
        }
        let cot_half = 1.0 / (0.5 * th).tan();
        let b = (1.0 / (th * th)) * (1.0 - 0.5 * th * cot_half);
        Mat3::identity() + k.mul_scalar(-0.5) + k.mul_mat(&k).mul_scalar(b)
    }

    /// Inverse right Jacobian.
    #[must_use]
    pub fn right_jacobian_inv(w: Vec3) -> Mat3 {
        So3::left_jacobian_inv(w * -1.0)
    }

    /// Baker-Campbell-Hausdorff series for so(3) to the given order
    /// (1, 2, or 3): log(exp a exp b).
    #[must_use]
    pub fn bch(a: Vec3, b: Vec3, order: usize) -> Vec3 {
        let mut out = a + b;
        if order >= 2 {
            out = out + a.cross(&b) * 0.5;
        }
        if order >= 3 {
            out = out + (a.cross(&a.cross(&b)) + b.cross(&b.cross(&a))) * (1.0 / 12.0);
        }
        out
    }

    /// Geodesic (Karcher) mean of rotations.
    #[must_use]
    pub fn geodesic_mean(rots: &[So3], iters: usize) -> So3 {
        let mut m = rots[0];
        for _ in 0..iters {
            let mut avg = Vec3::new(0.0, 0.0, 0.0);
            for r in rots {
                avg = avg + m.inverse().compose(r).log();
            }
            avg = avg * (1.0 / rots.len() as f64);
            if avg.magnitude() < 1e-14 {
                break;
            }
            m = m.compose(&So3::exp(avg));
        }
        m
    }
}

trait ScaleHalf {
    fn scale_half(self) -> Self;
}

impl ScaleHalf for Vec3 {
    fn scale_half(self) -> Self {
        self * 0.5
    }
}

// ---------------------------------------------------------------------------
// SE(3)
// ---------------------------------------------------------------------------

/// Rigid transform: rotation then translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    pub r: So3,
    pub t: Vec3,
}

/// se(3) algebra element: linear part rho, angular part phi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct se3 {
    pub rho: Vec3,
    pub phi: Vec3,
}

impl Se3 {
    #[must_use]
    pub fn identity() -> Self {
        Se3 {
            r: So3::identity(),
            t: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    /// Exponential map: t = J_l(phi) rho.
    #[must_use]
    pub fn exp(xi: se3) -> Self {
        let r = So3::exp(xi.phi);
        let t = So3::left_jacobian(xi.phi).mul_vec(xi.rho);
        Se3 { r, t }
    }

    /// Logarithm map.
    #[must_use]
    pub fn log(&self) -> se3 {
        let phi = self.r.log();
        let rho = So3::left_jacobian_inv(phi).mul_vec(self.t);
        se3 { rho, phi }
    }

    #[must_use]
    pub fn compose(&self, other: &Se3) -> Se3 {
        Se3 {
            r: self.r.compose(&other.r),
            t: self.r.apply(other.t) + self.t,
        }
    }

    #[must_use]
    pub fn inverse(&self) -> Se3 {
        let rinv = self.r.inverse();
        Se3 {
            r: rinv,
            t: rinv.apply(self.t * -1.0),
        }
    }

    /// 6x6 adjoint [[R, hat(t) R], [0, R]], ordering (rho, phi).
    #[must_use]
    pub fn adjoint(&self) -> [[f64; 6]; 6] {
        let r = self.r.0;
        let tr = So3::hat(self.t).mul_mat(&r);
        let mut a = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = r.data[i][j];
                a[i][j + 3] = tr.data[i][j];
                a[i + 3][j + 3] = r.data[i][j];
            }
        }
        a
    }

    #[must_use]
    pub fn apply_point(&self, p: Vec3) -> Vec3 {
        self.r.apply(p) + self.t
    }

    #[must_use]
    pub fn apply_vector(&self, v: Vec3) -> Vec3 {
        self.r.apply(v)
    }

    #[must_use]
    pub fn to_mat4(&self) -> Mat4 {
        let r = &self.r.0.data;
        Mat4::from_rows(
            [r[0][0], r[0][1], r[0][2], self.t.x],
            [r[1][0], r[1][1], r[1][2], self.t.y],
            [r[2][0], r[2][1], r[2][2], self.t.z],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    #[must_use]
    pub fn from_mat4(m: &Mat4) -> Se3 {
        let d = &m.data;
        Se3 {
            r: So3(Mat3::from_rows(
                [d[0][0], d[0][1], d[0][2]],
                [d[1][0], d[1][1], d[1][2]],
                [d[2][0], d[2][1], d[2][2]],
            )),
            t: Vec3::new(d[0][3], d[1][3], d[2][3]),
        }
    }

    /// Build from a rotation frame and origin.
    #[must_use]
    pub fn from_frame(r: &Mat3, origin: Vec3) -> Se3 {
        Se3 {
            r: So3(*r),
            t: origin,
        }
    }

    /// Extract (rotation, origin).
    #[must_use]
    pub fn to_frame(&self) -> (Mat3, Vec3) {
        (self.r.0, self.t)
    }

    /// Screw-motion interpolation: exp(t log(self^-1 other)) composed on
    /// the left with self.
    #[must_use]
    pub fn interpolate(&self, other: &Se3, t: f64) -> Se3 {
        let rel = self.inverse().compose(other).log();
        self.compose(&Se3::exp(se3 {
            rho: rel.rho * t,
            phi: rel.phi * t,
        }))
    }

    /// Screw axis of the motion: (direction, point on axis, angle,
    /// translation along the axis).
    #[must_use]
    pub fn screw_axis(&self) -> (Vec3, Vec3, f64, f64) {
        let xi = self.log();
        let th = xi.phi.magnitude();
        if th < EPS_ANGLE {
            let d = xi.rho.magnitude();
            let dir = if d > 0.0 { xi.rho * (1.0 / d) } else { Vec3::new(1.0, 0.0, 0.0) };
            return (dir, Vec3::new(0.0, 0.0, 0.0), 0.0, d);
        }
        let dir = xi.phi * (1.0 / th);
        let d_along = dir.dot(&xi.rho);
        // point on axis: q = (phi x rho) / theta^2
        let q = xi.phi.cross(&xi.rho) * (1.0 / (th * th));
        (dir, q, th, d_along)
    }

    /// Velocity of a point under a twist: v = rho + phi x p.
    #[must_use]
    pub fn twist_to_velocity(xi: &se3, p: Vec3) -> Vec3 {
        xi.rho + xi.phi.cross(&p)
    }

    /// Left Jacobian of SE(3) (block form with the Barfoot Q matrix),
    /// ordering (rho, phi).
    #[must_use]
    pub fn jacobian_left(xi: &se3) -> [[f64; 6]; 6] {
        let j = So3::left_jacobian(xi.phi);
        let q = se3_q_matrix(xi.rho, xi.phi);
        let mut out = [[0.0; 6]; 6];
        for i in 0..3 {
            for k in 0..3 {
                out[i][k] = j.data[i][k];
                out[i][k + 3] = q.data[i][k];
                out[i + 3][k + 3] = j.data[i][k];
            }
        }
        out
    }

    /// Right Jacobian: J_r(xi) = J_l(-xi).
    #[must_use]
    pub fn jacobian_right(xi: &se3) -> [[f64; 6]; 6] {
        Se3::jacobian_left(&se3 {
            rho: xi.rho * -1.0,
            phi: xi.phi * -1.0,
        })
    }

    /// Weighted distance: sqrt(|t_rel|^2 + weight * angle^2).
    #[must_use]
    pub fn distance(&self, other: &Se3, weight: f64) -> f64 {
        let rel = self.inverse().compose(other);
        (rel.t.magnitude_squared() + weight * rel.r.angle().powi(2)).sqrt()
    }

    #[must_use]
    pub fn random(rng: &mut Rng) -> Se3 {
        Se3 {
            r: So3::random(rng),
            t: Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian()),
        }
    }

    /// Iterative mean of poses in the group.
    #[must_use]
    pub fn mean(poses: &[Se3], iters: usize) -> Se3 {
        let mut m = poses[0];
        for _ in 0..iters {
            let mut rho = Vec3::new(0.0, 0.0, 0.0);
            let mut phi = Vec3::new(0.0, 0.0, 0.0);
            for p in poses {
                let l = m.inverse().compose(p).log();
                rho = rho + l.rho;
                phi = phi + l.phi;
            }
            let k = 1.0 / poses.len() as f64;
            let step = se3 {
                rho: rho * k,
                phi: phi * k,
            };
            if step.rho.magnitude() + step.phi.magnitude() < 1e-14 {
                break;
            }
            m = m.compose(&Se3::exp(step));
        }
        m
    }

    /// Relative transform a^-1 b.
    #[must_use]
    pub fn relative(a: &Se3, b: &Se3) -> Se3 {
        a.inverse().compose(b)
    }
}

/// Barfoot's Q matrix for the SE(3) Jacobian.
fn se3_q_matrix(rho: Vec3, phi: Vec3) -> Mat3 {
    let th = phi.magnitude();
    let rx = So3::hat(rho);
    let px = So3::hat(phi);
    if th < EPS_ANGLE {
        return rx.mul_scalar(0.5);
    }
    let th2 = th * th;
    let th3 = th2 * th;
    let th4 = th3 * th;
    let th5 = th4 * th;
    let (s, c) = th.sin_cos();
    let m1 = rx.mul_scalar(0.5);
    let pxrx = px.mul_mat(&rx);
    let rxpx = rx.mul_mat(&px);
    let a2 = (th - s) / th3;
    // Barfoot, State Estimation for Robotics, eq. (7.86)
    let term2 = (pxrx + rxpx + px.mul_mat(&rxpx)).mul_scalar(a2);
    let a3 = (th2 + 2.0 * c - 2.0) / (2.0 * th4);
    let term3 = (px.mul_mat(&pxrx) + rxpx.mul_mat(&px) + px.mul_mat(&rxpx).mul_scalar(-3.0))
        .mul_scalar(a3);
    let a4 = (2.0 * th - 3.0 * s + th * c) / (2.0 * th5);
    let term4 = (px.mul_mat(&rxpx).mul_mat(&px) + px.mul_mat(&px).mul_mat(&rxpx))
        .mul_scalar(a4);
    m1 + term2 + term3 + term4
}

// ---------------------------------------------------------------------------
// Sim(3), SO(2), SE(2)
// ---------------------------------------------------------------------------

/// Similarity transform: scale, rotation, translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sim3 {
    pub s: f64,
    pub r: So3,
    pub t: Vec3,
}

impl Sim3 {
    #[must_use]
    pub fn identity() -> Self {
        Sim3 {
            s: 1.0,
            r: So3::identity(),
            t: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    #[must_use]
    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.r.apply(p) * self.s + self.t
    }

    #[must_use]
    pub fn compose(&self, other: &Sim3) -> Sim3 {
        Sim3 {
            s: self.s * other.s,
            r: self.r.compose(&other.r),
            t: self.r.apply(other.t) * self.s + self.t,
        }
    }

    #[must_use]
    pub fn inverse(&self) -> Sim3 {
        let rinv = self.r.inverse();
        Sim3 {
            s: 1.0 / self.s,
            r: rinv,
            t: rinv.apply(self.t * (-1.0 / self.s)),
        }
    }
}

/// Planar rotation by an angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So2(pub f64);

impl So2 {
    #[must_use]
    pub fn compose(&self, other: &So2) -> So2 {
        So2(self.0 + other.0)
    }

    #[must_use]
    pub fn inverse(&self) -> So2 {
        So2(-self.0)
    }

    #[must_use]
    pub fn apply(&self, v: Vec2) -> Vec2 {
        let (s, c) = self.0.sin_cos();
        Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
    }
}

/// Planar rigid transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se2 {
    pub theta: f64,
    pub t: Vec2,
}

impl Se2 {
    #[must_use]
    pub fn identity() -> Self {
        Se2 {
            theta: 0.0,
            t: Vec2::new(0.0, 0.0),
        }
    }

    /// Exponential of (vx, vy, omega).
    #[must_use]
    pub fn exp(v: [f64; 3]) -> Se2 {
        let (vx, vy, w) = (v[0], v[1], v[2]);
        if w.abs() < EPS_ANGLE {
            return Se2 {
                theta: w,
                t: Vec2::new(vx, vy),
            };
        }
        let (s, c) = w.sin_cos();
        // V matrix for SE(2)
        let a = s / w;
        let b = (1.0 - c) / w;
        Se2 {
            theta: w,
            t: Vec2::new(a * vx - b * vy, b * vx + a * vy),
        }
    }

    /// Logarithm: (vx, vy, omega).
    #[must_use]
    pub fn log(&self) -> [f64; 3] {
        let w = self.theta;
        if w.abs() < EPS_ANGLE {
            return [self.t.x, self.t.y, w];
        }
        let (s, c) = w.sin_cos();
        let a = s / w;
        let b = (1.0 - c) / w;
        let det = a * a + b * b;
        let vx = (a * self.t.x + b * self.t.y) / det;
        let vy = (-b * self.t.x + a * self.t.y) / det;
        [vx, vy, w]
    }

    #[must_use]
    pub fn compose(&self, other: &Se2) -> Se2 {
        Se2 {
            theta: self.theta + other.theta,
            t: So2(self.theta).apply(other.t) + self.t,
        }
    }

    #[must_use]
    pub fn inverse(&self) -> Se2 {
        Se2 {
            theta: -self.theta,
            t: So2(-self.theta).apply(self.t * -1.0),
        }
    }

    /// 3x3 adjoint on (v, omega).
    #[must_use]
    pub fn adjoint(&self) -> [[f64; 3]; 3] {
        let (s, c) = self.theta.sin_cos();
        [
            [c, -s, self.t.y],
            [s, c, -self.t.x],
            [0.0, 0.0, 1.0],
        ]
    }

    #[must_use]
    pub fn apply(&self, p: Vec2) -> Vec2 {
        So2(self.theta).apply(p) + self.t
    }

    #[must_use]
    pub fn interpolate(&self, other: &Se2, t: f64) -> Se2 {
        let rel = self.inverse().compose(other).log();
        self.compose(&Se2::exp([rel[0] * t, rel[1] * t, rel[2] * t]))
    }

    /// Row-major 2x3 affine matrix [R | t].
    #[must_use]
    pub fn to_affine2(&self) -> [[f64; 3]; 2] {
        let (s, c) = self.theta.sin_cos();
        [[c, -s, self.t.x], [s, c, self.t.y]]
    }
}

/// Common Lie-group interface.
pub trait LieGroup: Sized {
    type Algebra;
    fn exp(a: &Self::Algebra) -> Self;
    fn log(&self) -> Self::Algebra;
    fn compose(&self, o: &Self) -> Self;
    fn inverse(&self) -> Self;
    fn identity() -> Self;
    fn adjoint_action(&self, a: &Self::Algebra) -> Self::Algebra;
}

impl LieGroup for So3 {
    type Algebra = Vec3;
    fn exp(a: &Vec3) -> Self {
        So3::exp(*a)
    }
    fn log(&self) -> Vec3 {
        So3::log(self)
    }
    fn compose(&self, o: &Self) -> Self {
        So3::compose(self, o)
    }
    fn inverse(&self) -> Self {
        So3::inverse(self)
    }
    fn identity() -> Self {
        So3::identity()
    }
    fn adjoint_action(&self, a: &Vec3) -> Vec3 {
        self.apply(*a)
    }
}

impl LieGroup for Se3 {
    type Algebra = se3;
    fn exp(a: &se3) -> Self {
        Se3::exp(*a)
    }
    fn log(&self) -> se3 {
        Se3::log(self)
    }
    fn compose(&self, o: &Self) -> Self {
        Se3::compose(self, o)
    }
    fn inverse(&self) -> Self {
        Se3::inverse(self)
    }
    fn identity() -> Self {
        Se3::identity()
    }
    fn adjoint_action(&self, a: &se3) -> se3 {
        let ad = self.adjoint();
        let x = [a.rho.x, a.rho.y, a.rho.z, a.phi.x, a.phi.y, a.phi.z];
        let mut y = [0.0; 6];
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = ad[i].iter().zip(&x).map(|(m, v)| m * v).sum();
        }
        se3 {
            rho: Vec3::new(y[0], y[1], y[2]),
            phi: Vec3::new(y[3], y[4], y[5]),
        }
    }
}

// ---------------------------------------------------------------------------
// SO(4)
// ---------------------------------------------------------------------------

/// A 4D rotation matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So4(pub Mat4);

fn quat_mul(a: &Quaternion, b: &Quaternion) -> Quaternion {
    *a * *b
}

fn quat_as_array(q: &Quaternion) -> [f64; 4] {
    [q.w, q.x, q.y, q.z]
}

impl So4 {
    #[must_use]
    pub fn identity() -> Self {
        So4(Mat4::identity())
    }

    /// Exponential of a bivector (b01, b02, b03, b12, b13, b23): builds the
    /// antisymmetric generator and exponentiates.
    #[must_use]
    pub fn exp(bivector: [f64; 6]) -> Self {
        let [b01, b02, b03, b12, b13, b23] = bivector;
        let gen = Matrix::from_fn(4, 4, |i, j| match (i, j) {
            (0, 1) => -b01,
            (1, 0) => b01,
            (0, 2) => -b02,
            (2, 0) => b02,
            (0, 3) => -b03,
            (3, 0) => b03,
            (1, 2) => -b12,
            (2, 1) => b12,
            (1, 3) => -b13,
            (3, 1) => b13,
            (2, 3) => -b23,
            (3, 2) => b23,
            _ => 0.0,
        });
        So4(Mat4::from_matrix(&matrix_exp(&gen)))
    }

    /// Logarithm: bivector components in the same ordering as [`So4::exp`].
    #[must_use]
    pub fn log(&self) -> [f64; 6] {
        let l = matrix_log(&self.0.to_matrix()).expect("rotation log failed");
        // antisymmetrize against numerical noise
        [
            0.5 * (l.get(1, 0) - l.get(0, 1)),
            0.5 * (l.get(2, 0) - l.get(0, 2)),
            0.5 * (l.get(3, 0) - l.get(0, 3)),
            0.5 * (l.get(2, 1) - l.get(1, 2)),
            0.5 * (l.get(3, 1) - l.get(1, 3)),
            0.5 * (l.get(3, 2) - l.get(2, 3)),
        ]
    }

    /// Rotation p -> l p r_conj with quaternions acting on (w, x, y, z).
    #[must_use]
    pub fn from_double_quaternion(l: Quaternion, r: Quaternion) -> Self {
        let ln = l.normalize();
        let rn = r.normalize();
        let basis = [
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Quaternion::new(0.0, 1.0, 0.0, 0.0),
            Quaternion::new(0.0, 0.0, 1.0, 0.0),
            Quaternion::new(0.0, 0.0, 0.0, 1.0),
        ];
        let mut m = Mat4::zero();
        for (c, e) in basis.iter().enumerate() {
            let img = quat_mul(&quat_mul(&ln, e), &rn.conjugate());
            let arr = quat_as_array(&img);
            for (rr, &v) in arr.iter().enumerate() {
                m.data[rr][c] = v;
            }
        }
        So4(m)
    }

    /// Factor into the double quaternion pair (l, r), unique up to a common
    /// sign.
    #[must_use]
    pub fn to_double_quaternion(&self) -> (Quaternion, Quaternion) {
        // images of the basis quaternions under the rotation
        let col = |c: usize| {
            Quaternion::new(
                self.0.data[0][c],
                self.0.data[1][c],
                self.0.data[2][c],
                self.0.data[3][c],
            )
        };
        let q0 = col(0);
        // r rotates (as SO(3)) the imaginary units onto Im(q0* q_i)
        let v = |c: usize| {
            let p = quat_mul(&q0.conjugate(), &col(c));
            Vec3::new(p.x, p.y, p.z)
        };
        let (v1, v2, v3) = (v(1), v(2), v(3));
        let rot = Mat3::from_rows(
            [v1.x, v2.x, v3.x],
            [v1.y, v2.y, v3.y],
            [v1.z, v2.z, v3.z],
        );
        let r = So3::project(&rot).to_quat();
        let l = quat_mul(&q0, &r);
        (l.normalize(), r.normalize())
    }

    /// Left-isoclinic rotation p -> q p.
    #[must_use]
    pub fn isoclinic_left(q: Quaternion) -> Self {
        So4::from_double_quaternion(q, Quaternion::identity())
    }

    /// Right-isoclinic rotation p -> p q_conj.
    #[must_use]
    pub fn isoclinic_right(q: Quaternion) -> Self {
        So4::from_double_quaternion(Quaternion::identity(), q)
    }

    #[must_use]
    pub fn compose(&self, other: &So4) -> So4 {
        So4(self.0.mul_mat(&other.0))
    }

    #[must_use]
    pub fn inverse(&self) -> So4 {
        So4(self.0.transpose())
    }

    #[must_use]
    pub fn apply(&self, p: [f64; 4]) -> [f64; 4] {
        self.0.mul_vec4(p)
    }

    #[must_use]
    pub fn random(rng: &mut Rng) -> So4 {
        let l = Quaternion::new(
            rng.next_gaussian(),
            rng.next_gaussian(),
            rng.next_gaussian(),
            rng.next_gaussian(),
        )
        .normalize();
        let r = Quaternion::new(
            rng.next_gaussian(),
            rng.next_gaussian(),
            rng.next_gaussian(),
            rng.next_gaussian(),
        )
        .normalize();
        So4::from_double_quaternion(l, r)
    }

    /// Rotation by `angle` in a single coordinate plane.
    #[must_use]
    pub fn simple_rotation(plane: (usize, usize), angle: f64) -> So4 {
        let (a, b) = plane;
        let (s, c) = angle.sin_cos();
        let mut m = Mat4::identity();
        m.data[a][a] = c;
        m.data[b][b] = c;
        m.data[a][b] = -s;
        m.data[b][a] = s;
        So4(m)
    }

    /// Double rotation: `angle1` in the (0,1) plane, `angle2` in (2,3).
    #[must_use]
    pub fn double_rotation(angle1: f64, angle2: f64) -> So4 {
        So4::simple_rotation((0, 1), angle1).compose(&So4::simple_rotation((2, 3), angle2))
    }
}

// ---------------------------------------------------------------------------
// SU(2)
// ---------------------------------------------------------------------------

/// SU(2) element stored as a unit quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su2(pub Quaternion);

impl Su2 {
    /// Exponential: rotation by angle |a| about a/|a| (double cover of
    /// SO(3): the quaternion carries the half angle).
    #[must_use]
    pub fn exp(a: Vec3) -> Self {
        let th = a.magnitude();
        if th < EPS_ANGLE {
            return Su2(Quaternion::new(1.0, 0.5 * a.x, 0.5 * a.y, 0.5 * a.z).normalize());
        }
        Su2(Quaternion::from_axis_angle(a * (1.0 / th), th))
    }

    /// Logarithm: axis times angle, angle in [0, 2 pi).
    #[must_use]
    pub fn log(&self) -> Vec3 {
        let q = self.0.normalize();
        let (axis, angle) = q.to_axis_angle();
        axis * angle
    }

    /// Double cover onto SO(3): q and -q map to the same rotation.
    #[must_use]
    pub fn to_so3(&self) -> So3 {
        So3::from_quat(&self.0)
    }

    /// Coefficients (c0, c1, c2, c3) with U = c0 I + c1 s1 + c2 s2 + c3 s3
    /// in the Pauli basis: U = w I - i (x s1 + y s2 + z s3).
    #[must_use]
    pub fn pauli_decompose(&self) -> [Complex; 4] {
        let q = self.0.normalize();
        [
            Complex::new(q.w, 0.0),
            Complex::new(0.0, -q.x),
            Complex::new(0.0, -q.y),
            Complex::new(0.0, -q.z),
        ]
    }

    /// The 2x2 complex matrix U = w I - i (x s1 + y s2 + z s3).
    #[must_use]
    pub fn to_matrix_2x2(&self) -> [[Complex; 2]; 2] {
        let q = self.0.normalize();
        [
            [Complex::new(q.w, -q.z), Complex::new(-q.y, -q.x)],
            [Complex::new(q.y, -q.x), Complex::new(q.w, q.z)],
        ]
    }

    /// Recover the quaternion from a 2x2 SU(2) matrix.
    #[must_use]
    pub fn from_matrix_2x2(m: [[Complex; 2]; 2]) -> Self {
        let w = 0.5 * (m[0][0].re + m[1][1].re);
        let z = 0.5 * (m[1][1].im - m[0][0].im);
        let y = 0.5 * (m[1][0].re - m[0][1].re);
        let x = -0.5 * (m[0][1].im + m[1][0].im);
        Su2(Quaternion::new(w, x, y, z).normalize())
    }

    #[must_use]
    pub fn compose(&self, other: &Su2) -> Su2 {
        Su2(quat_mul(&self.0, &other.0))
    }

    #[must_use]
    pub fn inverse(&self) -> Su2 {
        Su2(self.0.conjugate())
    }

    /// Matrix trace (real: 2w).
    #[must_use]
    pub fn trace(&self) -> f64 {
        2.0 * self.0.normalize().w
    }

    /// Character of the spin-j representation:
    /// chi_j(theta) = sin((2j+1) theta/2)/sin(theta/2) where theta is the
    /// SO(3) rotation angle.
    #[must_use]
    pub fn character(&self, j: f64) -> f64 {
        let w = self.0.normalize().w.clamp(-1.0, 1.0);
        let half_theta = w.acos();
        if half_theta.abs() < 1e-9 {
            return 2.0 * j + 1.0;
        }
        ((2.0 * j + 1.0) * half_theta).sin() / half_theta.sin()
    }
}

// ---------------------------------------------------------------------------
// SL(2, R) and SL(2, C)
// ---------------------------------------------------------------------------

/// Classification of Mobius/SL(2) elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sl2Class {
    Elliptic,
    Parabolic,
    Hyperbolic,
    Loxodromic,
}

/// SL(2, R) matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sl2R {
    pub m: [[f64; 2]; 2],
}

impl Sl2R {
    #[must_use]
    pub fn identity() -> Self {
        Sl2R {
            m: [[1.0, 0.0], [0.0, 1.0]],
        }
    }

    /// Exponential of a traceless 2x2 matrix (closed form).
    #[must_use]
    pub fn exp(a: [[f64; 2]; 2]) -> Self {
        let half_tr = 0.5 * (a[0][0] + a[1][1]);
        let b = [
            [a[0][0] - half_tr, a[0][1]],
            [a[1][0], a[1][1] - half_tr],
        ];
        // B^2 = -det(B) I
        let d = b[0][0] * b[1][1] - b[0][1] * b[1][0];
        let (cosw, sincw) = if d > 0.0 {
            let w = d.sqrt();
            (w.cos(), w.sin() / w)
        } else if d < 0.0 {
            let w = (-d).sqrt();
            (w.cosh(), w.sinh() / w)
        } else {
            (1.0, 1.0)
        };
        let e = half_tr.exp();
        Sl2R {
            m: [
                [e * (cosw + sincw * b[0][0]), e * sincw * b[0][1]],
                [e * sincw * b[1][0], e * (cosw + sincw * b[1][1])],
            ],
        }
    }

    /// Logarithm via the general matrix log.
    #[must_use]
    pub fn log(&self) -> [[f64; 2]; 2] {
        let m = Matrix::from_fn(2, 2, |i, j| self.m[i][j]);
        let l = matrix_log(&m).expect("log failed");
        [[l.get(0, 0), l.get(0, 1)], [l.get(1, 0), l.get(1, 1)]]
    }

    #[must_use]
    pub fn compose(&self, o: &Sl2R) -> Sl2R {
        let a = &self.m;
        let b = &o.m;
        Sl2R {
            m: [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                ],
            ],
        }
    }

    #[must_use]
    pub fn inverse(&self) -> Sl2R {
        let a = &self.m;
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        Sl2R {
            m: [
                [a[1][1] / det, -a[0][1] / det],
                [-a[1][0] / det, a[0][0] / det],
            ],
        }
    }

    /// Mobius action on the upper half-plane: z -> (az + b)/(cz + d).
    #[must_use]
    pub fn act_on_upper_half_plane(&self, z: Complex) -> Complex {
        let a = Complex::new(self.m[0][0], 0.0);
        let b = Complex::new(self.m[0][1], 0.0);
        let c = Complex::new(self.m[1][0], 0.0);
        let d = Complex::new(self.m[1][1], 0.0);
        (a * z + b) / (c * z + d)
    }

    /// Classification by |trace|.
    #[must_use]
    pub fn classify(&self) -> Sl2Class {
        let tr = (self.m[0][0] + self.m[1][1]).abs();
        if (tr - 2.0).abs() < 1e-12 {
            Sl2Class::Parabolic
        } else if tr < 2.0 {
            Sl2Class::Elliptic
        } else {
            Sl2Class::Hyperbolic
        }
    }

    /// Fixed points of the Mobius action (roots of c z^2 + (d - a) z - b).
    #[must_use]
    pub fn fixed_points(&self) -> Vec<Complex> {
        let (a, b, c, d) = (self.m[0][0], self.m[0][1], self.m[1][0], self.m[1][1]);
        if c.abs() < 1e-15 {
            if (d - a).abs() < 1e-15 {
                return vec![]; // identity-like: everything fixed
            }
            return vec![Complex::new(b / (d - a), 0.0)];
        }
        let disc = (d - a) * (d - a) + 4.0 * b * c;
        if disc >= 0.0 {
            let r = disc.sqrt();
            vec![
                Complex::new((a - d + r) / (2.0 * c), 0.0),
                Complex::new((a - d - r) / (2.0 * c), 0.0),
            ]
        } else {
            let r = (-disc).sqrt();
            vec![
                Complex::new((a - d) / (2.0 * c), r / (2.0 * c)),
                Complex::new((a - d) / (2.0 * c), -r / (2.0 * c)),
            ]
        }
    }

    /// Translation length of a hyperbolic element: 2 acosh(|tr|/2).
    #[must_use]
    pub fn translation_length(&self) -> f64 {
        let tr = (self.m[0][0] + self.m[1][1]).abs();
        if tr <= 2.0 {
            0.0
        } else {
            2.0 * (tr / 2.0).acosh()
        }
    }
}

/// SL(2, C) matrix.
#[derive(Debug, Clone, Copy)]
pub struct Sl2C {
    pub m: [[Complex; 2]; 2],
}

impl Sl2C {
    #[must_use]
    pub fn identity() -> Self {
        Sl2C {
            m: [
                [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            ],
        }
    }

    /// Mobius action on the Riemann sphere.
    #[must_use]
    pub fn mobius(&self, z: Complex) -> Complex {
        (self.m[0][0] * z + self.m[0][1]) / (self.m[1][0] * z + self.m[1][1])
    }

    #[must_use]
    pub fn compose(&self, o: &Sl2C) -> Sl2C {
        let mut m = [[Complex::new(0.0, 0.0); 2]; 2];
        for (i, mi) in m.iter_mut().enumerate() {
            for (j, mij) in mi.iter_mut().enumerate() {
                *mij = self.m[i][0] * o.m[0][j] + self.m[i][1] * o.m[1][j];
            }
        }
        Sl2C { m }
    }

    #[must_use]
    pub fn inverse(&self) -> Sl2C {
        let det = self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0];
        let inv = |c: Complex| c / det;
        Sl2C {
            m: [
                [inv(self.m[1][1]), inv(Complex::new(0.0, 0.0) - self.m[0][1])],
                [inv(Complex::new(0.0, 0.0) - self.m[1][0]), inv(self.m[0][0])],
            ],
        }
    }

    /// Double cover onto SO(3,1): X = t I + x s1 + y s2 + z s3 maps as
    /// X -> A X A^dagger; returns the 4x4 Lorentz matrix acting on
    /// (t, x, y, z).
    #[must_use]
    pub fn to_lorentz(&self) -> Mat4 {
        // basis Hermitian matrices for (t, x, y, z)
        let herm = |t: f64, x: f64, y: f64, z: f64| -> [[Complex; 2]; 2] {
            [
                [Complex::new(t + z, 0.0), Complex::new(x, -y)],
                [Complex::new(x, y), Complex::new(t - z, 0.0)],
            ]
        };
        let unherm = |m: [[Complex; 2]; 2]| -> [f64; 4] {
            let t = 0.5 * (m[0][0].re + m[1][1].re);
            let z = 0.5 * (m[0][0].re - m[1][1].re);
            let x = m[1][0].re;
            let y = m[1][0].im;
            [t, x, y, z]
        };
        let a = &self.m;
        let adag = [
            [a[0][0].conjugate(), a[1][0].conjugate()],
            [a[0][1].conjugate(), a[1][1].conjugate()],
        ];
        let mul = |p: &[[Complex; 2]; 2], q: &[[Complex; 2]; 2]| {
            let mut out = [[Complex::new(0.0, 0.0); 2]; 2];
            for (i, oi) in out.iter_mut().enumerate() {
                for (j, oij) in oi.iter_mut().enumerate() {
                    *oij = p[i][0] * q[0][j] + p[i][1] * q[1][j];
                }
            }
            out
        };
        let mut l = Mat4::zero();
        let basis = [
            herm(1.0, 0.0, 0.0, 0.0),
            herm(0.0, 1.0, 0.0, 0.0),
            herm(0.0, 0.0, 1.0, 0.0),
            herm(0.0, 0.0, 0.0, 1.0),
        ];
        for (c, x) in basis.iter().enumerate() {
            let img = mul(&mul(a, x), &adag);
            let v = unherm(img);
            for (r, &vr) in v.iter().enumerate() {
                l.data[r][c] = vr;
            }
        }
        l
    }

    /// Pure boost with rapidity `phi` along the unit direction `n`:
    /// A = cosh(phi/2) I + sinh(phi/2) (n . sigma).
    #[must_use]
    pub fn from_lorentz_boost(n: Vec3, phi: f64) -> Sl2C {
        let ch = (0.5 * phi).cosh();
        let sh = (0.5 * phi).sinh();
        Sl2C {
            m: [
                [
                    Complex::new(ch + sh * n.z, 0.0),
                    Complex::new(sh * n.x, -sh * n.y),
                ],
                [
                    Complex::new(sh * n.x, sh * n.y),
                    Complex::new(ch - sh * n.z, 0.0),
                ],
            ],
        }
    }

    /// Classification by the (complex) trace.
    #[must_use]
    pub fn classify(&self) -> Sl2Class {
        let tr = self.m[0][0] + self.m[1][1];
        if tr.im.abs() > 1e-12 {
            return Sl2Class::Loxodromic;
        }
        let t = tr.re.abs();
        if (t - 2.0).abs() < 1e-12 {
            Sl2Class::Parabolic
        } else if t < 2.0 {
            Sl2Class::Elliptic
        } else {
            Sl2Class::Hyperbolic
        }
    }
}

/// The 3D Heisenberg group with coordinates (x, y, z) and product
/// (x, y, z)(x', y', z') = (x + x', y + y', z + z' + x y').
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heisenberg3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Heisenberg3 {
    #[must_use]
    pub fn identity() -> Self {
        Heisenberg3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[must_use]
    pub fn compose(&self, o: &Heisenberg3) -> Heisenberg3 {
        Heisenberg3 {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z + self.x * o.y,
        }
    }

    #[must_use]
    pub fn inverse(&self) -> Heisenberg3 {
        Heisenberg3 {
            x: -self.x,
            y: -self.y,
            z: -self.z + self.x * self.y,
        }
    }

    /// Group commutator a b a^-1 b^-1 (lands in the center).
    #[must_use]
    pub fn commutator(&self, o: &Heisenberg3) -> Heisenberg3 {
        self.compose(o)
            .compose(&self.inverse())
            .compose(&o.inverse())
    }
}

// ---------------------------------------------------------------------------
// U(n)
// ---------------------------------------------------------------------------

/// A unitary matrix U(n) with complex entries.
#[derive(Debug, Clone)]
pub struct Unitary {
    pub m: Vec<Vec<Complex>>,
}

fn cmat_mul(a: &[Vec<Complex>], b: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
    let n = a.len();
    let mut out = vec![vec![Complex::new(0.0, 0.0); n]; n];
    for (i, oi) in out.iter_mut().enumerate() {
        for (j, oij) in oi.iter_mut().enumerate() {
            let mut s = Complex::new(0.0, 0.0);
            for (k, bk) in b.iter().enumerate() {
                s = s + a[i][k] * bk[j];
            }
            *oij = s;
        }
    }
    out
}

impl Unitary {
    /// U = exp(-i H t) for Hermitian H, by Taylor with scaling-squaring.
    #[must_use]
    pub fn from_hermitian_exp(h: &[Vec<Complex>], t: f64) -> Self {
        let n = h.len();
        // A = -i H t
        let a: Vec<Vec<Complex>> = h
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&c| Complex::new(c.im * t, -c.re * t))
                    .collect()
            })
            .collect();
        let norm: f64 = a
            .iter()
            .flat_map(|r| r.iter())
            .map(|c| c.norm())
            .fold(0.0, f64::max)
            * n as f64;
        let s = norm.log2().ceil().max(0.0) as u32;
        let scale = 1.0 / 2.0_f64.powi(s as i32);
        let asc: Vec<Vec<Complex>> = a
            .iter()
            .map(|r| r.iter().map(|&c| Complex::new(c.re * scale, c.im * scale)).collect())
            .collect();
        let mut sum = vec![vec![Complex::new(0.0, 0.0); n]; n];
        let mut term = vec![vec![Complex::new(0.0, 0.0); n]; n];
        for i in 0..n {
            sum[i][i] = Complex::new(1.0, 0.0);
            term[i][i] = Complex::new(1.0, 0.0);
        }
        for k in 1..=30 {
            term = cmat_mul(&term, &asc);
            let inv_k = 1.0 / k as f64;
            for r in term.iter_mut() {
                for c in r.iter_mut() {
                    *c = Complex::new(c.re * inv_k, c.im * inv_k);
                }
            }
            for (sr, tr) in sum.iter_mut().zip(&term) {
                for (sc, tc) in sr.iter_mut().zip(tr) {
                    *sc = *sc + *tc;
                }
            }
        }
        let mut u = sum;
        for _ in 0..s {
            u = cmat_mul(&u, &u);
        }
        Unitary { m: u }
    }

    /// Check U U^dagger = I within `tol`.
    #[must_use]
    pub fn is_unitary(&self, tol: f64) -> bool {
        let n = self.m.len();
        let d = self.dagger();
        let p = cmat_mul(&self.m, &d.m);
        for (i, row) in p.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                if (c.re - want).abs() > tol || c.im.abs() > tol {
                    return false;
                }
            }
        }
        n > 0
    }

    #[must_use]
    pub fn compose(&self, o: &Unitary) -> Unitary {
        Unitary {
            m: cmat_mul(&self.m, &o.m),
        }
    }

    /// Conjugate transpose.
    #[must_use]
    pub fn dagger(&self) -> Unitary {
        let n = self.m.len();
        let mut out = vec![vec![Complex::new(0.0, 0.0); n]; n];
        for (i, oi) in out.iter_mut().enumerate() {
            for (j, oij) in oi.iter_mut().enumerate() {
                *oij = self.m[j][i].conjugate();
            }
        }
        Unitary { m: out }
    }

    /// Haar-random unitary via Gram-Schmidt of a complex Gaussian matrix
    /// with phase normalization.
    #[must_use]
    pub fn random_haar(n: usize, rng: &mut Rng) -> Unitary {
        let mut cols: Vec<Vec<Complex>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| Complex::new(rng.next_gaussian(), rng.next_gaussian()))
                    .collect()
            })
            .collect();
        for i in 0..n {
            for j in 0..i {
                // proj = <col_j, col_i>
                let colj = cols[j].clone();
                let mut dot = Complex::new(0.0, 0.0);
                for (a, b) in colj.iter().zip(&cols[i]) {
                    dot = dot + a.conjugate() * *b;
                }
                for (a, b) in colj.iter().zip(cols[i].iter_mut()) {
                    *b = *b - *a * dot;
                }
            }
            let norm: f64 = cols[i].iter().map(|c| c.norm_sq()).sum::<f64>().sqrt();
            // divide by norm and fix the phase of the leading entry
            let lead = cols[i][0];
            let phase = if lead.norm() > 1e-30 {
                Complex::new(lead.re / lead.norm(), lead.im / lead.norm())
            } else {
                Complex::new(1.0, 0.0)
            };
            let scale = phase.conjugate();
            for c in cols[i].iter_mut() {
                *c = *c * scale;
                *c = Complex::new(c.re / norm, c.im / norm);
            }
        }
        // columns -> matrix
        let mut m = vec![vec![Complex::new(0.0, 0.0); n]; n];
        for (j, col) in cols.iter().enumerate() {
            for (i, &c) in col.iter().enumerate() {
                m[i][j] = c;
            }
        }
        Unitary { m }
    }
}

// ---------------------------------------------------------------------------
// Representation theory
// ---------------------------------------------------------------------------

/// Casimir eigenvalue of the spin-j representation of so(3): j(j+1).
#[must_use]
pub fn casimir_so3(j: f64) -> f64 {
    j * (j + 1.0)
}

fn factorial(n: i64) -> f64 {
    (1..=n).map(|v| v as f64).product::<f64>().max(1.0)
}

/// Wigner small-d matrix element d^j_{m1 m2}(beta) (Wigner's sum formula).
#[must_use]
pub fn wigner_d_small(j: f64, m1: f64, m2: f64, beta: f64) -> f64 {
    let jm1 = (j + m1).round() as i64;
    let jm2 = (j + m2).round() as i64;
    let jn1 = (j - m1).round() as i64;
    let jn2 = (j - m2).round() as i64;
    let pref = (factorial(jm1) * factorial(jn1) * factorial(jm2) * factorial(jn2)).sqrt();
    let (cb, sb) = ((0.5 * beta).cos(), (0.5 * beta).sin());
    let mut sum = 0.0;
    for k in 0..=(jm1.min(jn2)).max(0) {
        let a = jm1 - k;
        let b = jn2 - k;
        let c = k + (m2 - m1).round() as i64;
        if a < 0 || b < 0 || c < 0 {
            continue;
        }
        let denom = factorial(a) * factorial(b) * factorial(k) * factorial(c);
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        // cos^(2j + m1 - m2 - 2k), sin^(m2 - m1 + 2k)
        let ce = 2.0 * j + m1 - m2 - 2.0 * k as f64;
        let se = m2 - m1 + 2.0 * k as f64;
        sum += sign / denom * cb.powf(ce) * sb.powf(se);
    }
    pref * sum
}

/// Full Wigner D-matrix element
/// D^j_{m1 m2}(alpha, beta, gamma) = e^{-i m1 alpha} d^j_{m1 m2}(beta)
/// e^{-i m2 gamma}.
#[must_use]
pub fn wigner_d(j: f64, m1: f64, m2: f64, alpha: f64, beta: f64, gamma: f64) -> Complex {
    let d = wigner_d_small(j, m1, m2, beta);
    let phase = -(m1 * alpha + m2 * gamma);
    Complex::new(d * phase.cos(), d * phase.sin())
}

/// Clebsch-Gordan coefficient <j1 m1 j2 m2 | j m> (Racah's formula).
#[must_use]
pub fn clebsch_gordan(j1: f64, m1: f64, j2: f64, m2: f64, j: f64, m: f64) -> f64 {
    if (m1 + m2 - m).abs() > 1e-9 {
        return 0.0;
    }
    if j < (j1 - j2).abs() - 1e-9 || j > j1 + j2 + 1e-9 {
        return 0.0;
    }
    let f = factorial;
    let i = |x: f64| x.round() as i64;
    let pref = ((2.0 * j + 1.0)
        * f(i(j1 + j2 - j))
        * f(i(j1 - j2 + j))
        * f(i(-j1 + j2 + j))
        / f(i(j1 + j2 + j + 1.0)))
    .sqrt()
        * (f(i(j1 + m1))
            * f(i(j1 - m1))
            * f(i(j2 + m2))
            * f(i(j2 - m2))
            * f(i(j + m))
            * f(i(j - m)))
        .sqrt();
    let mut sum = 0.0;
    for k in 0..=i(j1 + j2 - j).max(0) {
        let t1 = i(j1 + j2 - j) - k;
        let t2 = i(j1 - m1) - k;
        let t3 = i(j2 + m2) - k;
        let t4 = i(j - j2 + m1) + k;
        let t5 = i(j - j1 - m2) + k;
        if t1 < 0 || t2 < 0 || t3 < 0 || t4 < 0 || t5 < 0 {
            continue;
        }
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign / (f(k) * f(t1) * f(t2) * f(t3) * f(t4) * f(t5));
    }
    pref * sum
}

/// Rotate the degree-l band of complex spherical-harmonic coefficients
/// (ordered m = -l..l) by the rotation `r` using the Wigner D-matrix with
/// zyz Euler angles.
#[must_use]
pub fn rotate_spherical_harmonics(coeffs: &[Complex], l: usize, r: &So3) -> Vec<Complex> {
    assert_eq!(coeffs.len(), 2 * l + 1);
    // zyz Euler angles from the rotation matrix
    let m = &r.0.data;
    let beta = m[2][2].clamp(-1.0, 1.0).acos();
    let (alpha, gamma) = if beta.abs() < 1e-9 || (std::f64::consts::PI - beta).abs() < 1e-9 {
        (m[1][0].atan2(m[0][0]), 0.0)
    } else {
        (m[1][2].atan2(m[0][2]), m[2][1].atan2(-m[2][0]))
    };
    let lf = l as f64;
    let mut out = vec![Complex::new(0.0, 0.0); 2 * l + 1];
    for (mi, o) in out.iter_mut().enumerate() {
        let mm = mi as f64 - lf;
        let mut s = Complex::new(0.0, 0.0);
        for (ni, &c) in coeffs.iter().enumerate() {
            let nn = ni as f64 - lf;
            s = s + wigner_d(lf, mm, nn, alpha, beta, gamma) * c;
        }
        *o = s;
    }
    out
}

/// Near-uniform deterministic grid on SO(3) built from a Fibonacci sphere
/// of axes and a golden-ratio sweep of angles.
#[must_use]
pub fn so3_uniform_grid(n: usize) -> Vec<So3> {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|k| {
            let kf = k as f64 + 0.5;
            let z = 1.0 - 2.0 * kf / n as f64;
            let rr = (1.0 - z * z).max(0.0).sqrt();
            let th = 2.0 * std::f64::consts::PI * kf * golden;
            let axis = Vec3::new(rr * th.cos(), rr * th.sin(), z);
            // Haar-uniform angle: inverse CDF of (1-cos)/pi; use a
            // different irrational than the axis sweep to decorrelate
            let u = (k as f64 * std::f64::consts::SQRT_2) % 1.0;
            // solve (angle - sin angle)/pi = u by bisection
            let (mut lo, mut hi) = (0.0, std::f64::consts::PI);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if (mid - mid.sin()) / std::f64::consts::PI < u {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            So3::from_axis_angle(axis, 0.5 * (lo + hi))
        })
        .collect()
}

/// Haar measure density over the rotation angle in [0, pi]:
/// rho(theta) = (1 - cos theta)/pi, normalized to integrate to 1.
#[must_use]
pub fn so3_haar_measure_density(angle: f64) -> f64 {
    (1.0 - angle.cos()) / std::f64::consts::PI
}

// ---------------------------------------------------------------------------
// Estimation on manifolds
// ---------------------------------------------------------------------------

/// Gauss-Newton pose-graph optimization: edges are (i, j, z_ij, info)
/// with residual log(z^-1 x_i^-1 x_j). Pose 0 is held fixed. Returns the
/// final total squared residual.
pub fn pose_graph_optimize(
    poses: &mut [Se3],
    edges: &[(usize, usize, Se3, [[f64; 6]; 6])],
    iters: usize,
) -> f64 {
    let n = poses.len();
    let dof = 6 * (n - 1); // pose 0 fixed
    let residual = |poses: &[Se3]| -> Vec<(usize, Vec<f64>)> {
        edges
            .iter()
            .enumerate()
            .map(|(e, &(i, j, ref z, _))| {
                let r = z
                    .inverse()
                    .compose(&poses[i].inverse())
                    .compose(&poses[j])
                    .log();
                (
                    e,
                    vec![r.rho.x, r.rho.y, r.rho.z, r.phi.x, r.phi.y, r.phi.z],
                )
            })
            .collect()
    };
    let total = |poses: &[Se3]| -> f64 {
        residual(poses)
            .iter()
            .map(|(_, r)| r.iter().map(|v| v * v).sum::<f64>())
            .sum()
    };
    let hstep = 1e-6;
    for _ in 0..iters {
        let r0 = residual(poses);
        // numerical Jacobian: 6 dof per non-fixed pose
        let mut jt_j = Matrix::zeros(dof, dof);
        let mut jt_r = vec![0.0; dof];
        for (e, &(i, j, ref _z, info)) in edges.iter().enumerate() {
            // local Jacobians wrt poses i and j
            let mut cols: Vec<(usize, Vec<f64>)> = Vec::new();
            for (&p, base) in [(i, 0usize), (j, 0)].iter().zip([i, j]) {
                let _ = p;
                if base == 0 {
                    continue;
                }
                for d in 0..6 {
                    let mut xi = [0.0; 6];
                    xi[d] = hstep;
                    let pert = Se3::exp(se3 {
                        rho: Vec3::new(xi[0], xi[1], xi[2]),
                        phi: Vec3::new(xi[3], xi[4], xi[5]),
                    });
                    let saved = poses[base];
                    poses[base] = saved.compose(&pert);
                    let rp = {
                        let (ii, jj, z) = (edges[e].0, edges[e].1, &edges[e].2);
                        let r = z
                            .inverse()
                            .compose(&poses[ii].inverse())
                            .compose(&poses[jj])
                            .log();
                        vec![r.rho.x, r.rho.y, r.rho.z, r.phi.x, r.phi.y, r.phi.z]
                    };
                    poses[base] = saved;
                    let col: Vec<f64> = rp
                        .iter()
                        .zip(&r0[e].1)
                        .map(|(a, b)| (a - b) / hstep)
                        .collect();
                    cols.push(((base - 1) * 6 + d, col));
                }
            }
            // accumulate weighted normal equations
            for (ca, va) in &cols {
                for (cb, vb) in &cols {
                    let mut s = 0.0;
                    for r in 0..6 {
                        for c in 0..6 {
                            s += va[r] * info[r][c] * vb[c];
                        }
                    }
                    jt_j.set(*ca, *cb, jt_j.get(*ca, *cb) + s);
                }
                let mut s = 0.0;
                for (var, inforow) in va.iter().zip(&info) {
                    for (infoc, rc) in inforow.iter().zip(&r0[e].1) {
                        s += var * infoc * rc;
                    }
                }
                jt_r[*ca] += s;
            }
        }
        // Levenberg damping for robustness
        for d in 0..dof {
            jt_j.set(d, d, jt_j.get(d, d) + 1e-9);
        }
        let delta = match lu_decompose(&jt_j).and_then(|lu| lu.solve(&jt_r)) {
            Ok(d) => d,
            Err(_) => break,
        };
        for (p, pose) in poses.iter_mut().enumerate().skip(1) {
            let base = (p - 1) * 6;
            let step = Se3::exp(se3 {
                rho: Vec3::new(-delta[base], -delta[base + 1], -delta[base + 2]),
                phi: Vec3::new(-delta[base + 3], -delta[base + 4], -delta[base + 5]),
            });
            *pose = pose.compose(&step);
        }
    }
    total(poses)
}

/// Park-Martin hand-eye calibration: solve AX = XB from motion pairs.
#[must_use]
pub fn hand_eye_calibration(a: &[Se3], b: &[Se3]) -> Se3 {
    // rotation: M = sum beta alpha^T, Rx = (M^T M)^{-1/2} M^T
    let mut m = Matrix::zeros(3, 3);
    for (ai, bi) in a.iter().zip(b) {
        let alpha = ai.r.log();
        let beta = bi.r.log();
        for r in 0..3 {
            for c in 0..3 {
                let br = [beta.x, beta.y, beta.z][r];
                let ac = [alpha.x, alpha.y, alpha.z][c];
                m.set(r, c, m.get(r, c) + br * ac);
            }
        }
    }
    let mtm = m.transpose().mul(&m).unwrap();
    let inv_sqrt = matrix_sqrt(&mtm)
        .and_then(|s| lu_decompose(&s)?.inverse())
        .expect("hand-eye rotation solve failed");
    let rx = inv_sqrt.mul(&m.transpose()).unwrap();
    let rx3 = So3::project(&Mat3::from_rows(
        [rx.get(0, 0), rx.get(0, 1), rx.get(0, 2)],
        [rx.get(1, 0), rx.get(1, 1), rx.get(1, 2)],
        [rx.get(2, 0), rx.get(2, 1), rx.get(2, 2)],
    ));
    // translation least squares: (Ra - I) tx = Rx tb - ta
    let rows = 3 * a.len();
    let mut amat = Matrix::zeros(rows, 3);
    let mut rhs = vec![0.0; rows];
    for (k, (ai, bi)) in a.iter().zip(b).enumerate() {
        let ra = ai.r.0;
        let rbt = rx3.apply(bi.t);
        for r in 0..3 {
            for c in 0..3 {
                amat.set(3 * k + r, c, ra.data[r][c] - if r == c { 1.0 } else { 0.0 });
            }
        }
        let diff = rbt - ai.t;
        rhs[3 * k] = diff.x;
        rhs[3 * k + 1] = diff.y;
        rhs[3 * k + 2] = diff.z;
    }
    let tx = crate::linalg::least_squares(&amat, &rhs).unwrap_or(vec![0.0; 3]);
    Se3 {
        r: rx3,
        t: Vec3::new(tx[0], tx[1], tx[2]),
    }
}

/// Umeyama similarity alignment: the Sim3 (or Se3 when `with_scale` is
/// false) minimizing sum |dst_i - (s R src_i + t)|^2.
#[must_use]
pub fn umeyama_alignment(src: &[Vec3], dst: &[Vec3], with_scale: bool) -> Sim3 {
    let n = src.len() as f64;
    let cs = src.iter().fold(Vec3::new(0.0, 0.0, 0.0), |a, &b| a + b) * (1.0 / n);
    let cd = dst.iter().fold(Vec3::new(0.0, 0.0, 0.0), |a, &b| a + b) * (1.0 / n);
    let mut cov = Matrix::zeros(3, 3);
    let mut var_src = 0.0;
    for (s, d) in src.iter().zip(dst) {
        let ss = *s - cs;
        let dd = *d - cd;
        var_src += ss.magnitude_squared();
        let svec = [ss.x, ss.y, ss.z];
        let dvec = [dd.x, dd.y, dd.z];
        for (r, dr) in dvec.iter().enumerate() {
            for (c, sc) in svec.iter().enumerate() {
                cov.set(r, c, cov.get(r, c) + dr * sc);
            }
        }
    }
    cov = cov.scale(1.0 / n);
    var_src /= n;
    let sv = svd(&cov).expect("svd failed");
    let det_uv = lu_decompose(&sv.u.mul(&sv.vt).unwrap())
        .map(|lu| lu.determinant())
        .unwrap_or(1.0);
    let mut d_fix = [1.0, 1.0, 1.0];
    if det_uv < 0.0 {
        d_fix[2] = -1.0;
    }
    let mut r = Matrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for (k, &df) in d_fix.iter().enumerate() {
                s += sv.u.get(i, k) * df * sv.vt.get(k, j);
            }
            r.set(i, j, s);
        }
    }
    let scale = if with_scale {
        let trace_ds: f64 = sv
            .sigma
            .iter()
            .zip(&d_fix)
            .map(|(sg, df)| sg * df)
            .sum();
        trace_ds / var_src
    } else {
        1.0
    };
    let r3 = Mat3::from_rows(
        [r.get(0, 0), r.get(0, 1), r.get(0, 2)],
        [r.get(1, 0), r.get(1, 1), r.get(1, 2)],
        [r.get(2, 0), r.get(2, 1), r.get(2, 2)],
    );
    let rot = So3(r3);
    let t = cd + rot.apply(cs) * (-scale);
    Sim3 { s: scale, r: rot, t }
}

/// Rotation averaging: chordal L2 mean (projected arithmetic mean of the
/// matrices) refined by a few IRLS iterations in the tangent space.
#[must_use]
pub fn rotation_averaging(rots: &[So3], weights: &[f64]) -> So3 {
    let mut m = Mat3::zero();
    for (r, &w) in rots.iter().zip(weights) {
        m = m + r.0.mul_scalar(w);
    }
    let mut mean = So3::project(&m);
    // IRLS refinement with geodesic weights
    for _ in 0..10 {
        let mut avg = Vec3::new(0.0, 0.0, 0.0);
        let mut wsum = 0.0;
        for (r, &w) in rots.iter().zip(weights) {
            let l = mean.inverse().compose(r).log();
            let d = l.magnitude().max(1e-9);
            let irls = w / d.max(0.05);
            avg = avg + l * irls;
            wsum += irls;
        }
        avg = avg * (1.0 / wsum);
        if avg.magnitude() < 1e-13 {
            break;
        }
        mean = mean.compose(&So3::exp(avg));
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_so3_roundtrips() {
        let mut rng = Rng::new(11);
        for _ in 0..20 {
            let r = So3::random(&mut rng);
            let r2 = So3::exp(r.log());
            for i in 0..3 {
                for j in 0..3 {
                    assert!(close(r.0.data[i][j], r2.0.data[i][j], 1e-12));
                }
            }
        }
        // log(exp(w)) = w for |w| < pi
        for &mag in &[0.001, 0.5, 2.0, 3.0] {
            let w = Vec3::new(0.3, -0.5, 0.8).normalized() * mag;
            let back = So3::exp(w).log();
            assert!((back - w).magnitude() < 1e-10, "mag {mag}");
        }
        // quaternion roundtrip
        let r = So3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.2);
        let q = r.to_quat();
        let r2 = So3::from_quat(&q);
        assert!(r.distance(&r2) < 1e-12);
        // project recovers a rotation from a noisy matrix
        let noisy = r.0 + Mat3::from_rows([0.01, 0.0, 0.0], [0.0, -0.02, 0.01], [0.0, 0.0, 0.015]);
        let pr = So3::project(&noisy);
        assert!(pr.0.determinant() > 0.999 && pr.0.determinant() < 1.001);
        assert!(pr.distance(&r) < 0.05);
        // Jacobian identity: J_l(w) J_l_inv(w) = I
        let w = Vec3::new(0.4, -0.2, 0.7);
        let prod = So3::left_jacobian(w).mul_mat(&So3::left_jacobian_inv(w));
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(close(prod.data[i][j], want, 1e-12));
            }
        }
        // first-order identity: exp(w + dw) ~ exp(w) exp(Jr(w) dw)
        let dw = Vec3::new(1e-6, -2e-6, 1.5e-6);
        let lhs = So3::exp(w + dw);
        let rhs = So3::exp(w).compose(&So3::exp(So3::right_jacobian(w).mul_vec(dw)));
        assert!(lhs.distance(&rhs) < 1e-9);
    }

    #[test]
    fn test_bch_and_mean() {
        let a = Vec3::new(0.15, -0.1, 0.05);
        let b = Vec3::new(-0.05, 0.2, 0.1);
        let exact = So3::exp(a).compose(&So3::exp(b)).log();
        let b3 = So3::bch(a, b, 3);
        assert!((b3 - exact).magnitude() < 3.0 * (0.25_f64.powi(4)), "bch3 err {}", (b3 - exact).magnitude());
        // order 1 is worse than order 3
        let b1 = So3::bch(a, b, 1);
        assert!((b1 - exact).magnitude() > (b3 - exact).magnitude());
        // geodesic mean of symmetric perturbations recovers the center
        let center = So3::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.7);
        let rots = vec![
            center.compose(&So3::exp(Vec3::new(0.1, 0.0, 0.0))),
            center.compose(&So3::exp(Vec3::new(-0.1, 0.0, 0.0))),
            center.compose(&So3::exp(Vec3::new(0.0, 0.1, 0.0))),
            center.compose(&So3::exp(Vec3::new(0.0, -0.1, 0.0))),
        ];
        // acos near the identity has a ~sqrt(eps) noise floor, hence 1e-6
        let m = So3::geodesic_mean(&rots, 30);
        assert!(m.distance(&center) < 1e-6);
    }

    #[test]
    fn test_se3() {
        let xi = se3 {
            rho: Vec3::new(0.3, -0.2, 0.5),
            phi: Vec3::new(0.4, 0.1, -0.3),
        };
        let g = Se3::exp(xi);
        let back = g.log();
        assert!((back.rho - xi.rho).magnitude() < 1e-12);
        assert!((back.phi - xi.phi).magnitude() < 1e-12);
        // compose/inverse
        let gi = g.inverse();
        let id = g.compose(&gi);
        assert!(id.t.magnitude() < 1e-12 && id.r.angle() < 1e-6);
        // mat4 roundtrip
        let m = g.to_mat4();
        let g2 = Se3::from_mat4(&m);
        assert!(g.distance(&g2, 1.0) < 1e-6);
        // screw interpolation has constant twist: relative log between
        // consecutive samples is constant
        let mut rng = Rng::new(3);
        let a = Se3::random(&mut rng);
        let b = Se3::random(&mut rng);
        let mut prev: Option<se3> = None;
        for k in 0..4 {
            let t0 = k as f64 * 0.25;
            let p0 = a.interpolate(&b, t0);
            let p1 = a.interpolate(&b, t0 + 0.25);
            let rel = p0.inverse().compose(&p1).log();
            if let Some(pr) = prev {
                assert!((rel.rho - pr.rho).magnitude() < 1e-9, "screw rho");
                assert!((rel.phi - pr.phi).magnitude() < 1e-9, "screw phi");
            }
            prev = Some(rel);
        }
        // adjoint consistency: Ad_g xi = log(g exp(xi) g^-1)
        let small = se3 {
            rho: Vec3::new(1e-4, 2e-4, -1e-4),
            phi: Vec3::new(-2e-4, 1e-4, 3e-4),
        };
        let lhs = LieGroup::adjoint_action(&g, &small);
        let rhs = g.compose(&Se3::exp(small)).compose(&g.inverse()).log();
        assert!((lhs.rho - rhs.rho).magnitude() < 1e-9);
        assert!((lhs.phi - rhs.phi).magnitude() < 1e-9);
        // screw axis of a pure rotation about z through origin
        let rot = Se3 {
            r: So3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.8),
            t: Vec3::new(0.0, 0.0, 0.0),
        };
        let (dir, _q, angle, slide) = rot.screw_axis();
        assert!((dir.z.abs() - 1.0).abs() < 1e-12 && (angle - 0.8).abs() < 1e-12);
        assert!(slide.abs() < 1e-12);
        // twist velocity
        let v = Se3::twist_to_velocity(
            &se3 {
                rho: Vec3::new(0.0, 0.0, 0.0),
                phi: Vec3::new(0.0, 0.0, 1.0),
            },
            Vec3::new(1.0, 0.0, 0.0),
        );
        assert!((v.y - 1.0).abs() < 1e-15 && v.x.abs() < 1e-15);
        // group mean
        let poses = vec![g, g, g];
        let mean = Se3::mean(&poses, 5);
        assert!(mean.distance(&g, 1.0) < 1e-6);
    }

    #[test]
    fn test_se2_so2() {
        let g = Se2::exp([1.0, 0.5, 0.8]);
        let l = g.log();
        assert!(close(l[0], 1.0, 1e-12) && close(l[1], 0.5, 1e-12) && close(l[2], 0.8, 1e-12));
        let gi = g.inverse();
        let id = g.compose(&gi);
        assert!(id.theta.abs() < 1e-12 && id.t.magnitude() < 1e-12);
        // pure rotation exp moves a point along an arc
        let quarter = Se2::exp([0.0, 0.0, std::f64::consts::FRAC_PI_2]);
        let p = quarter.apply(Vec2::new(1.0, 0.0));
        assert!(p.x.abs() < 1e-12 && (p.y - 1.0).abs() < 1e-12);
        // interpolate halfway
        let h = Se2::identity().interpolate(&g, 0.5);
        let full = h.compose(&h.inverse().compose(&g));
        assert!(close(full.theta, g.theta, 1e-12));
        let aff = g.to_affine2();
        assert!(close(aff[0][2], g.t.x, 1e-15));
        let r = So2(0.3).compose(&So2(-0.3));
        assert!(r.0.abs() < 1e-15);
    }

    #[test]
    fn test_so4() {
        let mut rng = Rng::new(21);
        // double quaternion construction is orthogonal with det 1
        let r = So4::random(&mut rng);
        let should_be_id = r.0.mul_mat(&r.0.transpose());
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(close(should_be_id.data[i][j], want, 1e-12));
            }
        }
        assert!(close(r.0.determinant(), 1.0, 1e-9));
        // to_double_quaternion roundtrip (up to overall sign)
        let (l, rr) = r.to_double_quaternion();
        let r2 = So4::from_double_quaternion(l, rr);
        let mut ok = true;
        for i in 0..4 {
            for j in 0..4 {
                if !close(r.0.data[i][j], r2.0.data[i][j], 1e-9) {
                    ok = false;
                }
            }
        }
        assert!(ok, "double quaternion roundtrip");
        // exp/log roundtrip
        let biv = [0.3, -0.2, 0.1, 0.4, -0.1, 0.25];
        let g = So4::exp(biv);
        let back = g.log();
        for (a, b) in biv.iter().zip(&back) {
            assert!(close(*a, *b, 1e-9), "{a} vs {b}");
        }
        // simple rotation acts only in its plane
        let s = So4::simple_rotation((1, 2), 0.7);
        let p = s.apply([1.0, 0.0, 0.0, 1.0]);
        assert!(close(p[0], 1.0, 1e-15) && close(p[3], 1.0, 1e-15));
        // double rotation composes commuting planes
        let d = So4::double_rotation(0.5, 0.9);
        let d2 = So4::simple_rotation((2, 3), 0.9).compose(&So4::simple_rotation((0, 1), 0.5));
        for i in 0..4 {
            for j in 0..4 {
                assert!(close(d.0.data[i][j], d2.0.data[i][j], 1e-12));
            }
        }
        // isoclinic left/right compose to the double rotation form
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.6);
        let combo = So4::isoclinic_left(q).compose(&So4::isoclinic_right(q));
        let direct = So4::from_double_quaternion(q, q);
        for i in 0..4 {
            for j in 0..4 {
                assert!(close(combo.0.data[i][j], direct.0.data[i][j], 1e-12));
            }
        }
    }

    #[test]
    fn test_su2() {
        // double cover: q and -q give the same rotation
        let a = Vec3::new(0.4, -0.7, 0.2);
        let u = Su2::exp(a);
        let neg = Su2(Quaternion::new(-u.0.w, -u.0.x, -u.0.y, -u.0.z));
        let r1 = u.to_so3();
        let r2 = neg.to_so3();
        assert!(r1.distance(&r2) < 1e-12);
        // exp/log roundtrip
        let back = u.log();
        assert!((back - a).magnitude() < 1e-10);
        // matrix roundtrip and unitarity of the 2x2 form
        let m = u.to_matrix_2x2();
        let u2 = Su2::from_matrix_2x2(m);
        assert!(u.0.dot(&u2.0).abs() > 1.0 - 1e-10);
        // det(U) = 1
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        assert!(close(det.re, 1.0, 1e-12) && close(det.im, 0.0, 1e-12));
        // Pauli decomposition reassembles the matrix
        let c = u.pauli_decompose();
        // entry (0,0) = c0 + c3 (sigma_z contribution)
        let m00 = c[0] + c[3];
        assert!(close(m00.re, m[0][0].re, 1e-12) && close(m00.im, m[0][0].im, 1e-12));
        // characters: identity gives 2j+1, general element bounded by it
        let id = Su2(Quaternion::identity());
        assert!(close(id.character(1.5), 4.0, 1e-12));
        assert!(u.character(1.0).abs() <= 3.0 + 1e-12);
        // trace = 2 cos(theta/2)
        let th = u.to_so3().angle();
        assert!(close(u.trace(), 2.0 * (0.5 * th).cos(), 1e-9));
    }

    #[test]
    fn test_sl2() {
        // closed-form exp matches the general matrix exp
        let a = [[0.3, 0.7], [0.2, -0.3]];
        let g = Sl2R::exp(a);
        let am = Matrix::from_rows(&[&a[0][..], &a[1][..]]).unwrap();
        let em = matrix_exp(&am);
        for i in 0..2 {
            for j in 0..2 {
                assert!(close(g.m[i][j], em.get(i, j), 1e-12));
            }
        }
        // det = 1 for traceless generator
        let det = g.m[0][0] * g.m[1][1] - g.m[0][1] * g.m[1][0];
        assert!(close(det, 1.0, 1e-12));
        // Mobius action preserves the upper half-plane
        let z = Complex::new(0.3, 1.2);
        let w = g.act_on_upper_half_plane(z);
        assert!(w.im > 0.0);
        // classification
        let rot = Sl2R::exp([[0.0, -0.5], [0.5, 0.0]]);
        assert_eq!(rot.classify(), Sl2Class::Elliptic);
        let boost = Sl2R::exp([[0.5, 0.0], [0.0, -0.5]]);
        assert_eq!(boost.classify(), Sl2Class::Hyperbolic);
        // translation length of diag(e^t, e^-t) is 2t
        assert!(close(boost.translation_length(), 1.0, 1e-10));
        // hyperbolic fixed points are real
        for f in boost.fixed_points() {
            assert!(f.im.abs() < 1e-12);
        }
        // SL(2, C): boost maps to a Lorentz transformation
        let b = Sl2C::from_lorentz_boost(Vec3::new(0.0, 0.0, 1.0), 0.8);
        let lam = b.to_lorentz();
        // Lambda^T eta Lambda = eta with eta = diag(1, -1, -1, -1)
        let eta = Mat4::from_rows(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        );
        let should_be_eta = lam.transpose().mul_mat(&eta).mul_mat(&lam);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    close(should_be_eta.data[i][j], eta.data[i][j], 1e-10),
                    "Lorentz check at ({i},{j})"
                );
            }
        }
        // boost entries: gamma = cosh(0.8) mixing t and z
        assert!(close(lam.data[0][0], 0.8_f64.cosh(), 1e-10));
        assert!(close(lam.data[0][3], 0.8_f64.sinh(), 1e-10));
        assert_eq!(b.classify(), Sl2Class::Hyperbolic);
        // Heisenberg: commutator is central (x = y = 0)
        let h1 = Heisenberg3 { x: 1.0, y: 0.5, z: 0.2 };
        let h2 = Heisenberg3 { x: -0.3, y: 0.8, z: 0.1 };
        let c = h1.commutator(&h2);
        assert!(c.x.abs() < 1e-12 && c.y.abs() < 1e-12);
        assert!(close(c.z, h1.x * h2.y - h2.x * h1.y, 1e-12));
    }

    #[test]
    fn test_matrix_functions() {
        // nilpotent: exp equals the truncated series exactly
        let n = Matrix::from_rows(&[&[0.0, 1.0, 2.0], &[0.0, 0.0, 3.0], &[0.0, 0.0, 0.0]])
            .unwrap();
        let e = matrix_exp(&n);
        // I + N + N^2/2
        let n2 = n.mul(&n).unwrap();
        let expect = Matrix::identity(3).add(&n).unwrap().add(&n2.scale(0.5)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(close(e.get(i, j), expect.get(i, j), 1e-12));
            }
        }
        // log(exp(A)) = A
        let a = Matrix::from_rows(&[&[0.1, 0.4, -0.2], &[0.0, -0.3, 0.2], &[0.1, 0.0, 0.2]])
            .unwrap();
        let back = matrix_log(&matrix_exp(&a)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(close(back.get(i, j), a.get(i, j), 1e-9));
            }
        }
        // sqrt squared
        let spd = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let s = matrix_sqrt(&spd).unwrap();
        let sq = s.mul(&s).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(close(sq.get(i, j), spd.get(i, j), 1e-10));
            }
        }
        // so(3) structure constants are the Levi-Civita symbol and the
        // Killing form is -2 I
        let basis = vec![
            Matrix::from_mat3(&So3::hat(Vec3::new(1.0, 0.0, 0.0))),
            Matrix::from_mat3(&So3::hat(Vec3::new(0.0, 1.0, 0.0))),
            Matrix::from_mat3(&So3::hat(Vec3::new(0.0, 0.0, 1.0))),
        ];
        let c = structure_constants(&basis);
        assert!(close(c.get(&[2, 0, 1]), 1.0, 1e-12));
        assert!(close(c.get(&[2, 1, 0]), -1.0, 1e-12));
        assert!(close(c.get(&[0, 0, 1]), 0.0, 1e-12));
        let k = killing_form(&basis);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { -2.0 } else { 0.0 };
                assert!(close(k.get(i, j), want, 1e-12));
            }
        }
        // bracket antisymmetry
        let br = lie_bracket_matrix(&basis[0], &basis[1]);
        let br2 = lie_bracket_matrix(&basis[1], &basis[0]);
        for i in 0..3 {
            for j in 0..3 {
                assert!(close(br.get(i, j), -br2.get(i, j), 1e-15));
            }
        }
    }

    #[test]
    fn test_wigner_and_cg() {
        // d^1_{00}(beta) = cos beta
        assert!(close(wigner_d_small(1.0, 0.0, 0.0, 0.7), 0.7_f64.cos(), 1e-12));
        // d matrix rows are orthonormal (unitarity)
        let j = 1.5;
        let beta = 0.9;
        let ms = [-1.5, -0.5, 0.5, 1.5];
        for &m1 in &ms {
            for &m2 in &ms {
                let mut s = 0.0;
                for &mm in &ms {
                    s += wigner_d_small(j, m1, mm, beta) * wigner_d_small(j, m2, mm, beta);
                }
                let want = if (m1 - m2).abs() < 1e-9 { 1.0 } else { 0.0 };
                assert!(close(s, want, 1e-10), "unitarity ({m1},{m2}): {s}");
            }
        }
        // Clebsch-Gordan: <1/2 1/2 1/2 -1/2 | 1 0> = 1/sqrt(2)
        let cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1.0, 0.0);
        assert!(close(cg, 1.0 / 2.0_f64.sqrt(), 1e-12), "cg = {cg}");
        // and | 0 0 > = 1/sqrt(2) with sign convention
        let cg0 = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0.0, 0.0);
        assert!(close(cg0.abs(), 1.0 / 2.0_f64.sqrt(), 1e-12));
        // orthogonality: sum over (m1, m2) of products for different j
        let mut dot = 0.0;
        let mut n1 = 0.0;
        for &m1 in &[-0.5, 0.5] {
            for &m2 in &[-0.5, 0.5] {
                let a = clebsch_gordan(0.5, m1, 0.5, m2, 1.0, 0.0);
                let b = clebsch_gordan(0.5, m1, 0.5, m2, 0.0, 0.0);
                dot += a * b;
                n1 += a * a;
            }
        }
        assert!(dot.abs() < 1e-12, "CG orthogonality {dot}");
        assert!(close(n1, 1.0, 1e-12), "CG normalization {n1}");
        // casimir
        assert!(close(casimir_so3(1.5), 3.75, 1e-15));
        // rotating SH coefficients preserves the norm (D is unitary)
        let coeffs = vec![
            Complex::new(0.3, 0.1),
            Complex::new(-0.2, 0.4),
            Complex::new(0.5, 0.0),
        ];
        let r = So3::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.8);
        let rot = rotate_spherical_harmonics(&coeffs, 1, &r);
        let n_before: f64 = coeffs.iter().map(|c| c.norm_sq()).sum();
        let n_after: f64 = rot.iter().map(|c| c.norm_sq()).sum();
        assert!(close(n_before, n_after, 1e-10));
        // haar density normalizes
        let mut integral = 0.0;
        let n_int = 2000;
        for i in 0..n_int {
            let th = std::f64::consts::PI * (i as f64 + 0.5) / n_int as f64;
            integral += so3_haar_measure_density(th) * std::f64::consts::PI / n_int as f64;
        }
        assert!(close(integral, 1.0, 1e-4));
        // uniform grid: mean rotation matrix is near zero (uniformity)
        let grid = so3_uniform_grid(200);
        let mut mean = Mat3::zero();
        for g in &grid {
            mean = mean + g.0.mul_scalar(1.0 / 200.0);
        }
        let frob: f64 = mean
            .data
            .iter()
            .flat_map(|r| r.iter())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(frob < 0.15, "grid mean {frob}");
    }

    #[test]
    fn test_unitary() {
        // exp of Pauli-z Hermitian
        let h = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            vec![Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)],
        ];
        let u = Unitary::from_hermitian_exp(&h, 0.7);
        assert!(u.is_unitary(1e-12));
        // diagonal phases e^{-i t}, e^{+i t}
        assert!(close(u.m[0][0].re, 0.7_f64.cos(), 1e-12));
        assert!(close(u.m[0][0].im, -0.7_f64.sin(), 1e-12));
        // haar random unitary
        let mut rng = Rng::new(5);
        let hr = Unitary::random_haar(4, &mut rng);
        assert!(hr.is_unitary(1e-10));
        // compose with dagger gives identity
        let prod = hr.compose(&hr.dagger());
        assert!(close(prod.m[2][2].re, 1.0, 1e-10) && prod.m[0][1].norm() < 1e-10);
    }

    #[test]
    fn test_pose_graph_and_calibration() {
        let mut rng = Rng::new(9);
        // ground-truth square loop
        let truth = [
            Se3::identity(),
            Se3 {
                r: So3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.5),
                t: Vec3::new(1.0, 0.0, 0.0),
            },
            Se3 {
                r: So3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.0),
                t: Vec3::new(1.0, 1.0, 0.0),
            },
            Se3 {
                r: So3::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.4),
                t: Vec3::new(0.0, 1.0, 0.2),
            },
        ];
        let info = {
            let mut m = [[0.0; 6]; 6];
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            m
        };
        let mut edges = Vec::new();
        for k in 0..4 {
            let (i, j) = (k, (k + 1) % 4);
            edges.push((i, j, Se3::relative(&truth[i], &truth[j]), info));
        }
        // noisy initialization
        let mut poses: Vec<Se3> = truth
            .iter()
            .map(|p| {
                p.compose(&Se3::exp(se3 {
                    rho: Vec3::new(
                        0.1 * rng.next_gaussian(),
                        0.1 * rng.next_gaussian(),
                        0.1 * rng.next_gaussian(),
                    ),
                    phi: Vec3::new(
                        0.05 * rng.next_gaussian(),
                        0.05 * rng.next_gaussian(),
                        0.05 * rng.next_gaussian(),
                    ),
                }))
            })
            .collect();
        poses[0] = truth[0];
        let before: f64 = edges
            .iter()
            .map(|&(i, j, ref z, _)| {
                let r = z.inverse().compose(&poses[i].inverse()).compose(&poses[j]).log();
                r.rho.magnitude_squared() + r.phi.magnitude_squared()
            })
            .sum();
        let after = pose_graph_optimize(&mut poses, &edges, 15);
        assert!(after < 0.01 * before, "pose graph {before} -> {after}");
        // hand-eye: B = X^-1 A X
        let x_true = Se3 {
            r: So3::from_axis_angle(Vec3::new(0.2, 1.0, 0.4), 0.9),
            t: Vec3::new(0.1, -0.2, 0.3),
        };
        let mut as_ = Vec::new();
        let mut bs = Vec::new();
        for _ in 0..8 {
            let a = Se3::random(&mut rng);
            let b = x_true.inverse().compose(&a).compose(&x_true);
            as_.push(a);
            bs.push(b);
        }
        let x_est = hand_eye_calibration(&as_, &bs);
        assert!(
            x_est.r.distance(&x_true.r) < 1e-6,
            "hand-eye rotation {}",
            x_est.r.distance(&x_true.r)
        );
        assert!((x_est.t - x_true.t).magnitude() < 1e-6, "hand-eye translation");
        // Umeyama with scale
        let src: Vec<Vec3> = (0..10)
            .map(|_| {
                Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian())
            })
            .collect();
        let s_true = 1.7;
        let r_true = So3::from_axis_angle(Vec3::new(1.0, 0.2, -0.3), 0.6);
        let t_true = Vec3::new(0.5, -1.0, 2.0);
        let dst: Vec<Vec3> = src.iter().map(|&p| r_true.apply(p) * s_true + t_true).collect();
        let sim = umeyama_alignment(&src, &dst, true);
        assert!(close(sim.s, s_true, 1e-10));
        assert!(sim.r.distance(&r_true) < 1e-10);
        assert!((sim.t - t_true).magnitude() < 1e-9);
        // rotation averaging
        let center = So3::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5);
        let rots = vec![
            center.compose(&So3::exp(Vec3::new(0.05, 0.0, 0.0))),
            center.compose(&So3::exp(Vec3::new(-0.05, 0.0, 0.0))),
            center.compose(&So3::exp(Vec3::new(0.0, 0.0, 0.05))),
            center.compose(&So3::exp(Vec3::new(0.0, 0.0, -0.05))),
        ];
        let avg = rotation_averaging(&rots, &[1.0; 4]);
        assert!(avg.distance(&center) < 0.01);
    }
}
