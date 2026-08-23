//! Clifford (geometric) algebras Cl(p, q, r): a dense multivector type
//! over any signature, with the geometric/outer/inner products, versors and
//! rotors, and specialized models — Euclidean [`cl3`], projective [`pga3`],
//! conformal [`cga3`], and spacetime [`sta`] geometric algebra.

use crate::fractals::Complex;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::quaternion::Quaternion;

/// A dense multivector in Cl(p, q, r): 2^(p+q+r) coefficients indexed by
/// basis-blade bitmask (bit i set means basis vector i is a factor; bits
/// 0..p square to +1, the next q to -1, the last r to 0).
#[derive(Debug, Clone, PartialEq)]
pub struct Multivector {
    pub p: usize,
    pub q: usize,
    pub r: usize,
    pub coeffs: Vec<f64>,
}

/// Sign from reordering the product of two basis blades into canonical
/// order (excluding metric contractions).
fn reorder_sign(a: usize, b: usize) -> f64 {
    let mut a = a >> 1;
    let mut swaps = 0u32;
    while a != 0 {
        swaps += (a & b).count_ones();
        a >>= 1;
    }
    if swaps.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Product of two basis blades under the (p, q, r) signature:
/// (sign, result mask); sign 0.0 when a degenerate vector squares.
fn blade_product(a: usize, b: usize, p: usize, q: usize, _r: usize) -> (f64, usize) {
    let mut sign = reorder_sign(a, b);
    let common = a & b;
    let mut bit = 0;
    let mut c = common;
    while c != 0 {
        if c & 1 == 1 {
            if bit >= p + q {
                return (0.0, 0); // degenerate: squares to zero
            }
            if bit >= p {
                sign = -sign; // negative-signature vector
            }
        }
        c >>= 1;
        bit += 1;
    }
    (sign, a ^ b)
}

impl Multivector {
    #[must_use]
    pub fn zero(p: usize, q: usize, r: usize) -> Self {
        Multivector {
            p,
            q,
            r,
            coeffs: vec![0.0; 1 << (p + q + r)],
        }
    }

    #[must_use]
    pub fn scalar(s: f64, p: usize, q: usize, r: usize) -> Self {
        let mut m = Self::zero(p, q, r);
        m.coeffs[0] = s;
        m
    }

    /// Grade-1 vector from components (one per basis vector).
    #[must_use]
    pub fn vector(v: &[f64], p: usize, q: usize, r: usize) -> Self {
        let mut m = Self::zero(p, q, r);
        for (i, &c) in v.iter().enumerate() {
            m.coeffs[1 << i] = c;
        }
        m
    }

    /// Unit basis blade with the given bitmask.
    #[must_use]
    pub fn basis_blade(mask: usize, p: usize, q: usize, r: usize) -> Self {
        let mut m = Self::zero(p, q, r);
        m.coeffs[mask] = 1.0;
        m
    }

    /// The unit pseudoscalar e_1...e_n.
    #[must_use]
    pub fn pseudoscalar(p: usize, q: usize, r: usize) -> Self {
        let n = p + q + r;
        Self::basis_blade((1 << n) - 1, p, q, r)
    }

    fn dim(&self) -> usize {
        self.p + self.q + self.r
    }

    /// Full geometric product.
    #[must_use]
    pub fn geometric(&self, o: &Self) -> Self {
        let mut out = Self::zero(self.p, self.q, self.r);
        for (a, &ca) in self.coeffs.iter().enumerate() {
            if ca == 0.0 {
                continue;
            }
            for (b, &cb) in o.coeffs.iter().enumerate() {
                if cb == 0.0 {
                    continue;
                }
                let (s, m) = blade_product(a, b, self.p, self.q, self.r);
                if s != 0.0 {
                    out.coeffs[m] += s * ca * cb;
                }
            }
        }
        out
    }

    /// Outer (wedge) product: blade terms with no common factors.
    #[must_use]
    pub fn wedge(&self, o: &Self) -> Self {
        let mut out = Self::zero(self.p, self.q, self.r);
        for (a, &ca) in self.coeffs.iter().enumerate() {
            if ca == 0.0 {
                continue;
            }
            for (b, &cb) in o.coeffs.iter().enumerate() {
                if cb == 0.0 || a & b != 0 {
                    continue;
                }
                let s = reorder_sign(a, b);
                out.coeffs[a | b] += s * ca * cb;
            }
        }
        out
    }

    /// Left contraction a ⌋ b: for basis blades, the grade
    /// |grade(b)| - |grade(a)| part of the geometric product, nonzero only
    /// when a's factors all lie inside b.
    #[must_use]
    pub fn inner(&self, o: &Self) -> Self {
        let mut out = Self::zero(self.p, self.q, self.r);
        for (a, &ca) in self.coeffs.iter().enumerate() {
            if ca == 0.0 {
                continue;
            }
            for (b, &cb) in o.coeffs.iter().enumerate() {
                if cb == 0.0 || a & !b != 0 {
                    continue;
                }
                let (s, m) = blade_product(a, b, self.p, self.q, self.r);
                if s != 0.0 {
                    out.coeffs[m] += s * ca * cb;
                }
            }
        }
        out
    }

    /// Scalar product <a b>_0.
    #[must_use]
    pub fn scalar_product(&self, o: &Self) -> f64 {
        let mut s = 0.0;
        for (a, &ca) in self.coeffs.iter().enumerate() {
            if ca == 0.0 {
                continue;
            }
            let cb = o.coeffs[a];
            if cb == 0.0 {
                continue;
            }
            let (sg, m) = blade_product(a, a, self.p, self.q, self.r);
            if m == 0 && sg != 0.0 {
                s += sg * ca * cb;
            }
        }
        s
    }

    /// Commutator product (ab - ba)/2.
    #[must_use]
    pub fn commutator(&self, o: &Self) -> Self {
        self.geometric(o).sub(&o.geometric(self)).scale(0.5)
    }

    /// Regressive product a ∨ b = undual(dual(a) ∧ dual(b)).
    #[must_use]
    pub fn regressive(&self, o: &Self) -> Self {
        self.dual().wedge(&o.dual()).undual()
    }

    /// Reverse: (-1)^{k(k-1)/2} per grade.
    #[must_use]
    pub fn reverse(&self) -> Self {
        let mut out = self.clone();
        for (m, c) in out.coeffs.iter_mut().enumerate() {
            let k = m.count_ones() as usize;
            if (k * (k.saturating_sub(1)) / 2) % 2 == 1 {
                *c = -*c;
            }
        }
        out
    }

    /// Grade involution: (-1)^k per grade.
    #[must_use]
    pub fn grade_involution(&self) -> Self {
        let mut out = self.clone();
        for (m, c) in out.coeffs.iter_mut().enumerate() {
            if m.count_ones() % 2 == 1 {
                *c = -*c;
            }
        }
        out
    }

    /// Clifford conjugation: reverse of the grade involution.
    #[must_use]
    pub fn clifford_conjugate(&self) -> Self {
        self.grade_involution().reverse()
    }

    /// Dual: right complement. For non-degenerate algebras this is x I^-1;
    /// for degenerate ones (r > 0) the Poincare complement mask map with
    /// the reordering sign (so that blade ∧ dual(blade) = pseudoscalar).
    #[must_use]
    pub fn dual(&self) -> Self {
        let n = self.dim();
        let full = (1usize << n) - 1;
        let mut out = Self::zero(self.p, self.q, self.r);
        if self.r == 0 {
            let inv_i = {
                let i = Self::pseudoscalar(self.p, self.q, self.r);
                let ii = i.geometric(&i).coeffs[0];
                i.scale(1.0 / ii)
            };
            return self.geometric(&inv_i);
        }
        for (m, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let comp = full & !m;
            let s = reorder_sign(m, comp);
            out.coeffs[comp] += s * c;
        }
        out
    }

    /// Inverse of [`Multivector::dual`].
    #[must_use]
    pub fn undual(&self) -> Self {
        let n = self.dim();
        let full = (1usize << n) - 1;
        if self.r == 0 {
            // dual is x I^-1, so undual is x I
            let i = Self::pseudoscalar(self.p, self.q, self.r);
            return self.geometric(&i);
        }
        let mut out = Self::zero(self.p, self.q, self.r);
        for (m, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let comp = full & !m;
            // sign such that dual(undual(x)) = x
            let s = reorder_sign(comp, m);
            out.coeffs[comp] += s * c;
        }
        out
    }

    /// Grade-k part.
    #[must_use]
    pub fn grade(&self, k: usize) -> Self {
        let mut out = Self::zero(self.p, self.q, self.r);
        for (m, &c) in self.coeffs.iter().enumerate() {
            if m.count_ones() as usize == k {
                out.coeffs[m] = c;
            }
        }
        out
    }

    /// The grades present (nonzero above tolerance).
    #[must_use]
    pub fn grades(&self) -> Vec<usize> {
        let mut set = std::collections::BTreeSet::new();
        for (m, &c) in self.coeffs.iter().enumerate() {
            if c.abs() > 1e-12 {
                set.insert(m.count_ones() as usize);
            }
        }
        set.into_iter().collect()
    }

    /// Heuristic blade check: single grade and X X~ is a scalar.
    #[must_use]
    pub fn is_blade(&self) -> bool {
        let g = self.grades();
        if g.len() != 1 {
            return false;
        }
        let prod = self.geometric(&self.reverse());
        prod.grades().iter().all(|&k| k == 0)
    }

    /// Heuristic versor check: X X~ is a nonzero scalar and X has only even
    /// or only odd grades.
    #[must_use]
    pub fn is_versor(&self) -> bool {
        let prod = self.geometric(&self.reverse());
        if !prod.grades().iter().all(|&k| k == 0) || prod.coeffs[0].abs() < 1e-12 {
            return false;
        }
        let g = self.grades();
        g.iter().all(|k| k % 2 == 0) || g.iter().all(|k| k % 2 == 1)
    }

    /// Squared magnitude <X~ X>_0 (may be negative in mixed signature).
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.reverse().geometric(self).coeffs[0]
    }

    #[must_use]
    pub fn norm(&self) -> f64 {
        self.norm_squared().abs().sqrt()
    }

    #[must_use]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-300 {
            self.clone()
        } else {
            self.scale(1.0 / n)
        }
    }

    /// Inverse for versor-like elements: X~/(X X~) when X X~ is scalar.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let rev = self.reverse();
        let prod = self.geometric(&rev);
        if !prod.grades().iter().all(|&k| k == 0) {
            return None;
        }
        let s = prod.coeffs[0];
        if s.abs() < 1e-300 {
            return None;
        }
        Some(rev.scale(1.0 / s))
    }

    /// Exponential: closed form for blades with scalar square, series with
    /// scaling-and-squaring otherwise.
    #[must_use]
    pub fn exp(&self) -> Self {
        let sq = self.geometric(self);
        if sq.grades().iter().all(|&k| k == 0) {
            let s = sq.coeffs[0];
            let one = Self::scalar(1.0, self.p, self.q, self.r);
            if s.abs() < 1e-300 {
                return one.add(self);
            }
            if s < 0.0 {
                let th = (-s).sqrt();
                return one.scale(th.cos()).add(&self.scale(th.sin() / th));
            }
            let th = s.sqrt();
            return one.scale(th.cosh()).add(&self.scale(th.sinh() / th));
        }
        // general series with scaling and squaring
        let norm: f64 = self.coeffs.iter().map(|c| c.abs()).sum();
        let k = norm.log2().ceil().max(0.0) as u32;
        let scaled = self.scale(1.0 / 2.0_f64.powi(k as i32));
        let mut sum = Self::scalar(1.0, self.p, self.q, self.r);
        let mut term = Self::scalar(1.0, self.p, self.q, self.r);
        for j in 1..=24 {
            term = term.geometric(&scaled).scale(1.0 / j as f64);
            sum = sum.add(&term);
        }
        let mut out = sum;
        for _ in 0..k {
            out = out.geometric(&out);
        }
        out
    }

    /// Logarithm of a rotor R = <R>_0 + <R>_2 (bivector generator).
    #[must_use]
    pub fn log(&self) -> Option<Self> {
        let s = self.coeffs[0];
        let b = self.grade(2);
        let b2 = b.geometric(&b).coeffs[0];
        if b.norm() < 1e-14 {
            return Some(Self::zero(self.p, self.q, self.r));
        }
        if b2 < 0.0 {
            // elliptic rotor: R = cos t + B sin t / |B|
            let bn = (-b2).sqrt();
            let t = bn.atan2(s);
            Some(b.scale(t / bn))
        } else {
            // hyperbolic (boost-like)
            let bn = b2.sqrt();
            let t = (bn / s).atanh();
            Some(b.scale(t / bn))
        }
    }

    /// Versor sandwich R x R~.
    #[must_use]
    pub fn sandwich(&self, x: &Self) -> Self {
        self.geometric(x).geometric(&self.reverse())
    }

    /// Rotor rotating unit vector a to unit vector b: (1 + b a)/|1 + b a|.
    #[must_use]
    pub fn rotor_from_vectors(a: &Self, b: &Self) -> Self {
        let one = Self::scalar(1.0, a.p, a.q, a.r);
        one.add(&b.normalized().geometric(&a.normalized())).normalized()
    }

    /// Rotor for a rotation by `angle` in the plane of unit bivector `b`:
    /// exp(-b angle/2).
    #[must_use]
    pub fn rotor_from_plane_angle(b: &Self, angle: f64) -> Self {
        b.normalized().scale(-0.5 * angle).exp()
    }

    /// Rotor interpolation R1 (R1^-1 R2)^t.
    #[must_use]
    pub fn rotor_interpolate(&self, o: &Self, t: f64) -> Self {
        let rel = self.inverse().expect("rotor inverse").geometric(o);
        let l = rel.log().expect("rotor log");
        self.geometric(&l.scale(t).exp())
    }

    /// Quaternion from the even subalgebra of Cl(3,0):
    /// i = -e23, j = -e31 = e13, k = -e12.
    #[must_use]
    pub fn to_quaternion(&self) -> Option<Quaternion> {
        if self.dim() != 3 || self.q != 0 || self.r != 0 {
            return None;
        }
        Some(Quaternion::new(
            self.coeffs[0b000],
            -self.coeffs[0b110],
            self.coeffs[0b101],
            -self.coeffs[0b011],
        ))
    }

    /// Rotor in Cl(3,0) from a quaternion (inverse of
    /// [`Multivector::to_quaternion`]).
    #[must_use]
    pub fn from_quaternion(q: &Quaternion) -> Self {
        let mut m = Self::zero(3, 0, 0);
        m.coeffs[0b000] = q.w;
        m.coeffs[0b110] = -q.x;
        m.coeffs[0b101] = q.y;
        m.coeffs[0b011] = -q.z;
        m
    }

    /// Matrix of left multiplication by this multivector on the coefficient
    /// space (a faithful 2^n-dimensional representation).
    #[must_use]
    pub fn to_matrix_rep(&self) -> Matrix {
        let n = 1 << self.dim();
        let mut m = Matrix::zeros(n, n);
        for (a, &ca) in self.coeffs.iter().enumerate() {
            if ca == 0.0 {
                continue;
            }
            for b in 0..n {
                let (s, res) = blade_product(a, b, self.p, self.q, self.r);
                if s != 0.0 {
                    m.set(res, b, m.get(res, b) + s * ca);
                }
            }
        }
        m
    }

    /// Meet (intersection) via the regressive product.
    #[must_use]
    pub fn meet(&self, o: &Self) -> Self {
        self.regressive(o)
    }

    /// Join (union): the wedge when independent, otherwise the larger blade.
    #[must_use]
    pub fn join(&self, o: &Self) -> Self {
        let w = self.wedge(o);
        if w.norm() > 1e-12 {
            w
        } else if self.grades().last() >= o.grades().last() {
            self.clone()
        } else {
            o.clone()
        }
    }

    /// Factor a blade into orthogonal grade-1 vectors.
    #[must_use]
    pub fn blade_factor(&self) -> Vec<Multivector> {
        let g = self.grades();
        if g.len() != 1 {
            return Vec::new();
        }
        let k = g[0];
        if k == 0 {
            return Vec::new();
        }
        let mut rest = self.normalized();
        let mut out = Vec::new();
        for _ in 0..k {
            let g = rest.grades();
            if g.is_empty() || g[0] == 0 {
                break;
            }
            let gr = g[0];
            if gr == 1 {
                out.push(rest.normalized());
                break;
            }
            // project each basis vector onto the remaining blade, keep the
            // largest as the next factor
            let mut best: Option<Multivector> = None;
            let mut best_norm = 0.0;
            for i in 0..self.dim() {
                let e = Self::basis_blade(1 << i, self.p, self.q, self.r);
                let proj = e.project_onto_blade(&rest);
                let pn = proj.norm();
                if pn > best_norm {
                    best_norm = pn;
                    best = Some(proj);
                }
            }
            let f = match best {
                Some(b) => b.normalized(),
                None => break,
            };
            let finv = match f.inverse() {
                Some(v) => v,
                None => break,
            };
            rest = finv.geometric(&rest).grade(gr - 1);
            out.push(f);
        }
        out
    }

    /// Projection of x onto blade B: (x ⌋ B) B^-1.
    #[must_use]
    pub fn project_onto_blade(&self, b: &Self) -> Self {
        match b.inverse() {
            Some(binv) => self.inner(b).geometric(&binv),
            None => Self::zero(self.p, self.q, self.r),
        }
    }

    /// Rejection from a blade.
    #[must_use]
    pub fn reject_from_blade(&self, b: &Self) -> Self {
        self.sub(&self.project_onto_blade(b))
    }

    /// Reflection through the line of vector n: n X n^-1.
    #[must_use]
    pub fn reflect_in_vector(&self, n: &Self) -> Self {
        let ninv = n.inverse().expect("vector must be invertible");
        n.geometric(self).geometric(&ninv)
    }

    /// Reflection in the hyperplane orthogonal to n: n X̂ n^-1.
    #[must_use]
    pub fn reflect_in_hyperplane(&self, n: &Self) -> Self {
        let ninv = n.inverse().expect("vector must be invertible");
        n.geometric(&self.grade_involution()).geometric(&ninv)
    }

    #[must_use]
    pub fn add(&self, o: &Self) -> Self {
        Multivector {
            p: self.p,
            q: self.q,
            r: self.r,
            coeffs: self
                .coeffs
                .iter()
                .zip(&o.coeffs)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    #[must_use]
    pub fn sub(&self, o: &Self) -> Self {
        Multivector {
            p: self.p,
            q: self.q,
            r: self.r,
            coeffs: self
                .coeffs
                .iter()
                .zip(&o.coeffs)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    #[must_use]
    pub fn scale(&self, k: f64) -> Self {
        Multivector {
            p: self.p,
            q: self.q,
            r: self.r,
            coeffs: self.coeffs.iter().map(|c| c * k).collect(),
        }
    }

    /// Alias for [`Multivector::scale`].
    #[must_use]
    pub fn mul_scalar(&self, k: f64) -> Self {
        self.scale(k)
    }

    /// Human-readable blade expansion, e.g. "1.5 + 2e12 - 0.3e123".
    #[must_use]
    pub fn to_string_blades(&self) -> String {
        let mut parts = Vec::new();
        for (m, &c) in self.coeffs.iter().enumerate() {
            if c.abs() < 1e-12 {
                continue;
            }
            let name = blade_name(m, self.p, self.q, self.r);
            let term = if m == 0 {
                format!("{c}")
            } else if (c - 1.0).abs() < 1e-12 {
                name.clone()
            } else if (c + 1.0).abs() < 1e-12 {
                format!("-{name}")
            } else {
                format!("{c}{name}")
            };
            parts.push(term);
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            let mut s = parts[0].clone();
            for t in &parts[1..] {
                if let Some(stripped) = t.strip_prefix('-') {
                    s.push_str(" - ");
                    s.push_str(stripped);
                } else {
                    s.push_str(" + ");
                    s.push_str(t);
                }
            }
            s
        }
    }
}

/// Basis-blade multiplication table: table[a][b] = (sign, result mask).
#[must_use]
pub fn cayley_table(p: usize, q: usize, r: usize) -> Vec<Vec<(f64, usize)>> {
    let n = 1 << (p + q + r);
    (0..n)
        .map(|a| (0..n).map(|b| blade_product(a, b, p, q, r)).collect())
        .collect()
}

/// Name of a basis blade, e.g. "e12" (1-indexed factors).
#[must_use]
pub fn blade_name(mask: usize, p: usize, q: usize, r: usize) -> String {
    let _ = (p, q, r);
    if mask == 0 {
        return "1".to_string();
    }
    let mut s = "e".to_string();
    for i in 0..64 {
        if mask >> i & 1 == 1 {
            s.push_str(&format!("{}", i + 1));
        }
    }
    s
}

/// Dimension of the algebra: 2^(p+q+r).
#[must_use]
pub fn algebra_dimension(p: usize, q: usize, r: usize) -> usize {
    1 << (p + q + r)
}

/// Classification of small Clifford algebras by isomorphism type.
#[must_use]
pub fn is_isomorphic_to_known(p: usize, q: usize, r: usize) -> &'static str {
    match (p, q, r) {
        (0, 0, 0) => "reals",
        (0, 1, 0) => "complex",
        (1, 0, 0) => "split-complex",
        (0, 0, 1) => "dual numbers",
        (0, 2, 0) => "quaternions",
        (2, 0, 0) | (1, 1, 0) => "M2(R)",
        (3, 0, 0) => "M2(C)",
        (0, 3, 0) => "quaternions x quaternions",
        (1, 3, 0) => "M2(H)",
        (3, 1, 0) => "M4(R)",
        (4, 1, 0) => "M4(C)",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// cl3: Euclidean 3D geometric algebra Cl(3,0)
// ---------------------------------------------------------------------------

/// Euclidean 3D geometric algebra Cl(3, 0).
pub mod cl3 {
    use super::{Multivector, Quaternion, Vec3};

    /// Grade-1 vector.
    #[must_use]
    pub fn vec(v: Vec3) -> Multivector {
        Multivector::vector(&[v.x, v.y, v.z], 3, 0, 0)
    }

    /// Bivector dual to the vector b (the plane with normal b).
    #[must_use]
    pub fn bivec(b: Vec3) -> Multivector {
        // dual of vector: v I with I = e123
        vec(b).geometric(&pseudoscalar())
    }

    /// The pseudoscalar e123.
    #[must_use]
    pub fn pseudoscalar() -> Multivector {
        Multivector::pseudoscalar(3, 0, 0)
    }

    /// Rotor for a rotation about `axis` by `angle` (matches quaternion
    /// rotation).
    #[must_use]
    pub fn rotor(axis: Vec3, angle: f64) -> Multivector {
        let b = bivec(axis.normalized());
        b.scale(-0.5 * angle).exp()
    }

    /// Rotate a vector with a rotor: R v R~.
    #[must_use]
    pub fn rotate(v: Vec3, r: &Multivector) -> Vec3 {
        let out = r.sandwich(&vec(v));
        to_vec3(&out).expect("sandwich of a vector is a vector")
    }

    /// The cross product via the wedge: a x b = -I (a ∧ b).
    #[must_use]
    pub fn cross_via_wedge(a: Vec3, b: Vec3) -> Vec3 {
        let w = vec(a).wedge(&vec(b));
        let d = w.geometric(&pseudoscalar()).scale(-1.0);
        to_vec3(&d).expect("dual of a bivector is a vector")
    }

    /// Reflect v in the plane with unit normal n.
    #[must_use]
    pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
        let out = vec(v).reflect_in_hyperplane(&vec(n.normalized()));
        to_vec3(&out).expect("reflection of a vector is a vector")
    }

    /// Extract the grade-1 part as a Vec3 (None if other grades dominate).
    #[must_use]
    pub fn to_vec3(m: &Multivector) -> Option<Vec3> {
        let g1 = m.grade(1);
        let rest = m.sub(&g1);
        if rest.norm() > 1e-9 * m.norm().max(1.0) {
            return None;
        }
        Some(Vec3::new(g1.coeffs[0b001], g1.coeffs[0b010], g1.coeffs[0b100]))
    }

    /// The plane (bivector) through three points, with weight twice the
    /// triangle area.
    #[must_use]
    pub fn plane_from_points(a: Vec3, b: Vec3, c: Vec3) -> Multivector {
        vec(b - a).wedge(&vec(c - a))
    }

    /// The line direction blade through two points (their difference).
    #[must_use]
    pub fn line_from_points(a: Vec3, b: Vec3) -> Multivector {
        vec(b - a)
    }

    /// Rotor as quaternion.
    #[must_use]
    pub fn rotor_to_quaternion(r: &Multivector) -> Option<Quaternion> {
        r.to_quaternion()
    }
}

// ---------------------------------------------------------------------------
// pga3: projective geometric algebra Cl(3,0,1)
// ---------------------------------------------------------------------------

/// Plane-based projective geometric algebra Cl(3, 0, 1): planes are
/// vectors, points are trivectors, and rigid motions are motors. Basis
/// vectors e1, e2, e3 are bits 0..2; the degenerate e0 is bit 3.
pub mod pga3 {
    use super::{Multivector, Vec3};
    use crate::manifold::lie::{Se3, So3};
    use crate::quaternion::Quaternion;

    fn e(i: usize) -> Multivector {
        Multivector::basis_blade(1 << i, 3, 0, 1)
    }

    fn e0() -> Multivector {
        Multivector::basis_blade(1 << 3, 3, 0, 1)
    }

    /// The plane n . x + d = 0 as a grade-1 element.
    #[must_use]
    pub fn plane(n: Vec3, d: f64) -> Multivector {
        e(0).scale(n.x)
            .add(&e(1).scale(n.y))
            .add(&e(2).scale(n.z))
            .add(&e0().scale(d))
    }

    /// A Euclidean point as the meet of three axis-aligned planes.
    #[must_use]
    pub fn point(p: Vec3) -> Multivector {
        let px = plane(Vec3::new(1.0, 0.0, 0.0), -p.x);
        let py = plane(Vec3::new(0.0, 1.0, 0.0), -p.y);
        let pz = plane(Vec3::new(0.0, 0.0, 1.0), -p.z);
        px.wedge(&py).wedge(&pz)
    }

    /// Ideal (infinite) point in direction d.
    #[must_use]
    pub fn point_at_infinity(d: Vec3) -> Multivector {
        point(d).sub(&point(Vec3::new(0.0, 0.0, 0.0)))
    }

    /// Line through two points (their join).
    #[must_use]
    pub fn line_from_points(a: Vec3, b: Vec3) -> Multivector {
        join(&point(a), &point(b))
    }

    /// Line as the meet of two planes.
    #[must_use]
    pub fn line_from_planes(p: &Multivector, q: &Multivector) -> Multivector {
        p.wedge(q)
    }

    /// Plane through three points (their join).
    #[must_use]
    pub fn plane_from_points(a: Vec3, b: Vec3, c: Vec3) -> Multivector {
        join(&join(&point(a), &point(b)), &point(c))
    }

    /// Meet (intersection): the outer product in the plane-based algebra.
    #[must_use]
    pub fn meet(a: &Multivector, b: &Multivector) -> Multivector {
        a.wedge(b)
    }

    /// Join (span): the regressive product.
    #[must_use]
    pub fn join(a: &Multivector, b: &Multivector) -> Multivector {
        a.regressive(b)
    }

    /// Euclidean coordinates of a (normalized or unnormalized) point.
    #[must_use]
    pub fn to_vec3(p: &Multivector) -> Option<Vec3> {
        // decompose against the basis trivectors by wedging test planes:
        // coordinates recovered from meets with coordinate planes
        let w = weight(p);
        if w.abs() < 1e-12 {
            return None;
        }
        // meet with coordinate planes yields scalars via the pseudoscalar
        let coord = |n: Vec3| {
            let pl = plane(n, 0.0);
            // pl ^ point = (n . x + d) * pseudoscalar-ish
            let m = pl.wedge(p);
            m.coeffs[0b1111]
        };
        let x = coord(Vec3::new(1.0, 0.0, 0.0));
        let y = coord(Vec3::new(0.0, 1.0, 0.0));
        let z = coord(Vec3::new(0.0, 0.0, 1.0));
        // sign/scale fixed by the weight
        Some(Vec3::new(x / w, y / w, z / w))
    }

    /// Weight of a point (the coefficient pairing with a plane at its
    /// location); used to normalize.
    fn weight(p: &Multivector) -> f64 {
        // wedge with the plane x*0 + 1*e0? e0 ^ P picks the Euclidean part
        let m = e0().wedge(p);
        m.coeffs[0b1111]
    }

    /// True for ideal (infinite) elements: zero weight.
    #[must_use]
    pub fn is_ideal(x: &Multivector) -> bool {
        weight(x).abs() < 1e-9 * x.norm().max(1e-30)
    }

    /// Signed distance from a point to a plane (both normalized inside).
    #[must_use]
    pub fn distance_point_plane(p: &Multivector, pl: &Multivector) -> f64 {
        let n = Vec3::new(pl.coeffs[0b0001], pl.coeffs[0b0010], pl.coeffs[0b0100]);
        let nn = n.magnitude();
        let m = pl.wedge(p);
        m.coeffs[0b1111] / (nn * weight(p))
    }

    /// Two distinct points on a line (by meeting with coordinate planes).
    fn two_points_on_line(l: &Multivector) -> Option<(Vec3, Vec3)> {
        let mut found = Vec::new();
        for d in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ] {
            for off in [0.0, 1.0, -1.0, 2.0] {
                let pl = plane(d, off);
                let pt = meet(l, &pl);
                if weight(&pt).abs() > 1e-9 {
                    if let Some(v) = to_vec3(&pt) {
                        if found
                            .iter()
                            .all(|f: &Vec3| (*f - v).magnitude() > 1e-6)
                        {
                            found.push(v);
                        }
                    }
                }
                if found.len() == 2 {
                    return Some((found[0], found[1]));
                }
            }
        }
        None
    }

    /// Distance from a point to a line.
    #[must_use]
    pub fn distance_point_line(p: &Multivector, l: &Multivector) -> f64 {
        let pv = to_vec3(p).expect("finite point");
        let (a, b) = two_points_on_line(l).expect("line");
        let d = (b - a).normalized();
        let r = pv - a;
        (r - d * r.dot(&d)).magnitude()
    }

    /// Distance between two lines.
    #[must_use]
    pub fn distance_lines(l1: &Multivector, l2: &Multivector) -> f64 {
        let (a1, b1) = two_points_on_line(l1).expect("line 1");
        let (a2, b2) = two_points_on_line(l2).expect("line 2");
        let d1 = (b1 - a1).normalized();
        let d2 = (b2 - a2).normalized();
        let n = d1.cross(&d2);
        if n.magnitude() < 1e-9 {
            // parallel
            let r = a2 - a1;
            return (r - d1 * r.dot(&d1)).magnitude();
        }
        ((a2 - a1).dot(&n.normalized())).abs()
    }

    /// Angle between two planes.
    #[must_use]
    pub fn angle_planes(p: &Multivector, q: &Multivector) -> f64 {
        let n1 = Vec3::new(p.coeffs[0b0001], p.coeffs[0b0010], p.coeffs[0b0100]);
        let n2 = Vec3::new(q.coeffs[0b0001], q.coeffs[0b0010], q.coeffs[0b0100]);
        (n1.dot(&n2) / (n1.magnitude() * n2.magnitude()))
            .clamp(-1.0, 1.0)
            .acos()
    }

    /// Angle between two lines.
    #[must_use]
    pub fn angle_lines(l1: &Multivector, l2: &Multivector) -> f64 {
        let (a1, b1) = two_points_on_line(l1).expect("line 1");
        let (a2, b2) = two_points_on_line(l2).expect("line 2");
        let d1 = (b1 - a1).normalized();
        let d2 = (b2 - a2).normalized();
        d1.dot(&d2).abs().clamp(0.0, 1.0).acos()
    }

    /// Motor translating by t.
    #[must_use]
    pub fn motor_translation(t: Vec3) -> Multivector {
        // T = 1 + (1/2)(t . e) e0-part bivector; sign fixed so that
        // T point(0) T~ = point(t)
        let b = e0().wedge(
            &e(0).scale(t.x).add(&e(1).scale(t.y)).add(&e(2).scale(t.z)),
        );
        Multivector::scalar(1.0, 3, 0, 1).sub(&b.scale(0.5))
    }

    /// Rotor in Cl(3,0,1) about an axis direction through the origin.
    fn origin_rotor(axis: Vec3, angle: f64) -> Multivector {
        // same coefficients as the Cl(3,0) rotor, embedded (masks without
        // bit 3)
        let q = Quaternion::from_axis_angle(axis.normalized(), angle);
        let mut m = Multivector::zero(3, 0, 1);
        m.coeffs[0b0000] = q.w;
        m.coeffs[0b0110] = -q.x;
        m.coeffs[0b0101] = q.y;
        m.coeffs[0b0011] = -q.z;
        m
    }

    /// Motor rotating by `angle` about an axis line.
    #[must_use]
    pub fn motor_rotation(axis_line: &Multivector, angle: f64) -> Multivector {
        let (a, b) = two_points_on_line(axis_line).expect("axis line");
        let dir = (b - a).normalized();
        let t = motor_translation(a);
        let tinv = motor_translation(a * -1.0);
        t.geometric(&origin_rotor(dir, angle)).geometric(&tinv)
    }

    /// Screw motor: rotate by `angle` about the line while translating
    /// `dist` along it.
    #[must_use]
    pub fn motor_screw(line: &Multivector, angle: f64, dist: f64) -> Multivector {
        let (a, b) = two_points_on_line(line).expect("screw line");
        let dir = (b - a).normalized();
        motor_translation(dir * dist).geometric(&motor_rotation(line, angle))
    }

    /// Motor from a rigid transform.
    #[must_use]
    pub fn motor_from_se3(m: &Se3) -> Multivector {
        let q = m.r.to_quat();
        motor_translation(m.t).geometric(&origin_rotor_from_quat(&q))
    }

    fn origin_rotor_from_quat(q: &Quaternion) -> Multivector {
        let mut m = Multivector::zero(3, 0, 1);
        m.coeffs[0b0000] = q.w;
        m.coeffs[0b0110] = -q.x;
        m.coeffs[0b0101] = q.y;
        m.coeffs[0b0011] = -q.z;
        m
    }

    /// Rigid transform from a motor.
    #[must_use]
    pub fn motor_to_se3(m: &Multivector) -> Se3 {
        // rotation from the Euclidean even part
        let q = Quaternion::new(
            m.coeffs[0b0000],
            -m.coeffs[0b0110],
            m.coeffs[0b0101],
            -m.coeffs[0b0011],
        )
        .normalize();
        let r = So3::from_quat(&q);
        // translation from the image of the origin
        let img = motor_apply(m, &point(Vec3::new(0.0, 0.0, 0.0)));
        let t = to_vec3(&img).expect("motor image of origin");
        Se3 { r, t }
    }

    /// Screw interpolation between motors (through Se3's exact screw).
    #[must_use]
    pub fn motor_interpolate(a: &Multivector, b: &Multivector, t: f64) -> Multivector {
        let sa = motor_to_se3(a);
        let sb = motor_to_se3(b);
        motor_from_se3(&sa.interpolate(&sb, t))
    }

    /// Apply a motor by the sandwich product.
    #[must_use]
    pub fn motor_apply(m: &Multivector, x: &Multivector) -> Multivector {
        m.sandwich(x)
    }

    /// Orthogonal projection of a point onto a line.
    #[must_use]
    pub fn project_point_on_line(p: &Multivector, l: &Multivector) -> Multivector {
        let pv = to_vec3(p).expect("finite point");
        let (a, b) = two_points_on_line(l).expect("line");
        let d = (b - a).normalized();
        point(a + d * (pv - a).dot(&d))
    }

    /// Orthogonal projection of a point onto a plane.
    #[must_use]
    pub fn project_point_on_plane(p: &Multivector, pl: &Multivector) -> Multivector {
        let pv = to_vec3(p).expect("finite point");
        let n = Vec3::new(pl.coeffs[0b0001], pl.coeffs[0b0010], pl.coeffs[0b0100]);
        let nn = n.magnitude();
        let d = pl.coeffs[0b1000] / nn;
        let nu = n * (1.0 / nn);
        point(pv - nu * (nu.dot(&pv) + d))
    }

    /// Orthogonal projection of a line onto a plane.
    #[must_use]
    pub fn project_line_on_plane(l: &Multivector, pl: &Multivector) -> Multivector {
        let (a, b) = two_points_on_line(l).expect("line");
        let pa = project_point_on_plane(&point(a), pl);
        let pb = project_point_on_plane(&point(b), pl);
        join(&pa, &pb)
    }

    /// One explicit step of PGA rigid-body dynamics (Gunn): the motor
    /// advances by its body-frame rate bivector.
    pub fn rigid_body_step(motor: &mut Multivector, rate: &Multivector, dt: f64) {
        let step = rate.scale(-0.5 * dt).exp();
        *motor = motor.geometric(&step);
        // renormalize the versor
        let n = motor.norm();
        if n > 1e-12 {
            *motor = motor.scale(1.0 / n);
        }
    }

    /// Diagonal inertia map on body-rate bivectors: scales the rotational
    /// components by (ixx, iyy, izz) and the translational by the mass.
    #[must_use]
    pub fn inertia_dual_map(rate: &Multivector, inertia: [f64; 3], mass: f64) -> Multivector {
        let mut out = rate.clone();
        // rotational bivectors e23, e13, e12 (masks 6, 5, 3)
        out.coeffs[0b0110] *= inertia[0];
        out.coeffs[0b0101] *= inertia[1];
        out.coeffs[0b0011] *= inertia[2];
        // translational e0i (masks 9, 10, 12)
        out.coeffs[0b1001] *= mass;
        out.coeffs[0b1010] *= mass;
        out.coeffs[0b1100] *= mass;
        out
    }

    /// Forque (force + torque) bivector of a force applied at a point: the
    /// weighted line through the point in the force direction.
    #[must_use]
    pub fn forque(force: Vec3, application_point: Vec3) -> Multivector {
        let mag = force.magnitude();
        if mag < 1e-30 {
            return Multivector::zero(3, 0, 1);
        }
        line_from_points(application_point, application_point + force).scale(mag)
    }
}

// ---------------------------------------------------------------------------
// cga3: conformal geometric algebra Cl(4,1)
// ---------------------------------------------------------------------------

/// Conformal geometric algebra Cl(4, 1): points, spheres, circles, lines
/// and planes as blades, with conformal versors. Basis: e1..e3 (bits 0..2,
/// +1), e+ (bit 3, +1), e- (bit 4, -1); null vectors e_inf = e- + e+ and
/// e_0 = (e- - e+)/2.
pub mod cga3 {
    use super::{Multivector, Vec3};
    use crate::manifold::lie::Sim3;
    use crate::manifold::polytope4::Vec4;

    /// Kinds of CGA object.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CgaObject {
        Point,
        PointPair,
        Line,
        Circle,
        Plane,
        Sphere,
        ImaginarySphere,
        Ideal,
        Unknown,
    }

    fn e(i: usize) -> Multivector {
        Multivector::basis_blade(1 << i, 4, 1, 0)
    }

    /// The null vector at infinity.
    #[must_use]
    pub fn e_inf() -> Multivector {
        e(4).add(&e(3))
    }

    /// The null origin vector.
    #[must_use]
    pub fn e_0() -> Multivector {
        e(4).sub(&e(3)).scale(0.5)
    }

    /// The positive-signature extra basis vector.
    #[must_use]
    pub fn e_plus() -> Multivector {
        e(3)
    }

    /// The negative-signature extra basis vector.
    #[must_use]
    pub fn e_minus() -> Multivector {
        e(4)
    }

    fn evec(v: Vec3) -> Multivector {
        e(0).scale(v.x).add(&e(1).scale(v.y)).add(&e(2).scale(v.z))
    }

    /// Conformal up-projection of a Euclidean point:
    /// P = p + (1/2) p^2 e_inf + e_0.
    #[must_use]
    pub fn point(p: Vec3) -> Multivector {
        evec(p)
            .add(&e_inf().scale(0.5 * p.magnitude_squared()))
            .add(&e_0())
    }

    /// Euclidean coordinates of a conformal point (None for ideal points).
    #[must_use]
    pub fn down(x: &Multivector) -> Option<Vec3> {
        let w = -x.scalar_product(&e_inf());
        if w.abs() < 1e-12 {
            return None;
        }
        Some(Vec3::new(
            x.coeffs[0b00001] / w,
            x.coeffs[0b00010] / w,
            x.coeffs[0b00100] / w,
        ))
    }

    /// IPNS sphere with the given center and radius.
    #[must_use]
    pub fn sphere(center: Vec3, r: f64) -> Multivector {
        point(center).sub(&e_inf().scale(0.5 * r * r))
    }

    /// IPNS plane n . x = d (unit normal recommended).
    #[must_use]
    pub fn plane(n: Vec3, d: f64) -> Multivector {
        evec(n.normalized()).add(&e_inf().scale(d))
    }

    /// IPNS circle through three points.
    #[must_use]
    pub fn circle_from_points(a: Vec3, b: Vec3, c: Vec3) -> Multivector {
        point(a).wedge(&point(b)).wedge(&point(c)).dual()
    }

    /// IPNS line through two points.
    #[must_use]
    pub fn line_from_points(a: Vec3, b: Vec3) -> Multivector {
        point(a).wedge(&point(b)).wedge(&e_inf()).dual()
    }

    /// IPNS point pair.
    #[must_use]
    pub fn point_pair(a: Vec3, b: Vec3) -> Multivector {
        point(a).wedge(&point(b)).dual()
    }

    /// IPNS sphere through four points.
    #[must_use]
    pub fn sphere_from_points(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Multivector {
        point(a)
            .wedge(&point(b))
            .wedge(&point(c))
            .wedge(&point(d))
            .dual()
    }

    /// Meet of IPNS objects (their intersection): the outer product.
    #[must_use]
    pub fn meet(a: &Multivector, b: &Multivector) -> Multivector {
        a.wedge(b)
    }

    /// The sphere-pencil basis of a grade-k IPNS round/flat: grade-1
    /// elements v with v ^ X = 0.
    fn pencil(x: &Multivector) -> Vec<Multivector> {
        let mut out = Vec::new();
        // solve v ^ x = 0 over the 5D grade-1 space by least squares
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for i in 0..5 {
            let v = e(i);
            let w = v.wedge(x);
            cols.push(w.coeffs.clone());
        }
        // nullspace of the 32 x 5 matrix via Gram matrix eigen
        let mut gram = [[0.0_f64; 5]; 5];
        for (i, gi) in gram.iter_mut().enumerate() {
            for (j, gij) in gi.iter_mut().enumerate() {
                *gij = cols[i]
                    .iter()
                    .zip(&cols[j])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            }
        }
        let m = crate::linalg::Matrix::from_fn(5, 5, |i, j| gram[i][j]);
        if let Ok(eig) = crate::linalg::eigen_symmetric(&m, 1e-12, 200) {
            for k in 0..5 {
                if eig.values[k].abs() < 1e-8 {
                    let mut v = Multivector::zero(4, 1, 0);
                    for i in 0..5 {
                        v.coeffs[1 << i] = eig.vectors.get(i, k);
                    }
                    out.push(v);
                }
            }
        }
        out
    }

    /// Center, radius, and plane normal of an IPNS circle.
    #[must_use]
    pub fn circle_center_radius_normal(c: &Multivector) -> (Vec3, f64, Vec3) {
        let pen = pencil(c);
        assert!(pen.len() >= 2, "not a circle pencil");
        // find the plane member (zero e_0 weight) and a sphere member
        let weight = |v: &Multivector| -v.scalar_product(&e_inf());
        // plane member: combination with zero weight; sphere member: any
        // pencil element with nonzero weight, normalized to weight 1
        let w1 = weight(&pen[0]);
        let w2 = weight(&pen[1]);
        let pl = if w1.abs() < 1e-9 {
            pen[0].clone()
        } else if w2.abs() < 1e-9 {
            pen[1].clone()
        } else {
            pen[0].scale(1.0 / w1).sub(&pen[1].scale(1.0 / w2))
        };
        let sp = if w1.abs() > 1e-9 {
            pen[0].scale(1.0 / w1)
        } else {
            pen[1].scale(1.0 / w2)
        };
        // plane: n + d e_inf (unnormalized)
        let n = Vec3::new(pl.coeffs[0b00001], pl.coeffs[0b00010], pl.coeffs[0b00100]);
        let nn = n.magnitude().max(1e-30);
        let nu = n * (1.0 / nn);
        // plane offset: for pl = alpha(n_u + d e_inf): d from the e_0-free
        // decomposition: coefficient of e_inf = (e_plus + e_minus)/...
        // recover d via  pl . e_0 = -alpha d? use scalar product with e_0:
        let d = -pl.scalar_product(&e_0()) / nn;
        // sphere: normalized weight 1 -> center and radius
        let cs = Vec3::new(sp.coeffs[0b00001], sp.coeffs[0b00010], sp.coeffs[0b00100]);
        let r2 = sp.scalar_product(&sp);
        // circle = sphere ∩ plane
        let dist = nu.dot(&cs) - d;
        let center = cs - nu * dist;
        let radius = (r2 - dist * dist).max(0.0).sqrt();
        (center, radius, nu)
    }

    /// Center and radius of an IPNS sphere.
    #[must_use]
    pub fn sphere_center_radius(s: &Multivector) -> (Vec3, f64) {
        let w = -s.scalar_product(&e_inf());
        let sn = s.scale(1.0 / w);
        let c = Vec3::new(sn.coeffs[0b00001], sn.coeffs[0b00010], sn.coeffs[0b00100]);
        let r2 = sn.scalar_product(&sn);
        (c, r2.max(0.0).sqrt())
    }

    /// A point on an IPNS line and its direction.
    #[must_use]
    pub fn line_point_direction(l: &Multivector) -> (Vec3, Vec3) {
        let pen = pencil(l);
        assert!(pen.len() >= 2, "not a line pencil");
        // both members are planes (weight ~ 0); extract two planes
        let plane_of = |v: &Multivector| {
            let n = Vec3::new(v.coeffs[0b00001], v.coeffs[0b00010], v.coeffs[0b00100]);
            let nn = n.magnitude().max(1e-30);
            let d = -v.scalar_product(&e_0()) / nn;
            (n * (1.0 / nn), d)
        };
        let (n1, d1) = plane_of(&pen[0]);
        let (n2, d2) = plane_of(&pen[1]);
        let dir = n1.cross(&n2).normalized();
        // point: solve n1.x = d1, n2.x = d2, dir.x = 0
        let a = crate::linalg::Matrix::from_rows(&[
            &[n1.x, n1.y, n1.z],
            &[n2.x, n2.y, n2.z],
            &[dir.x, dir.y, dir.z],
        ])
        .unwrap();
        let sol = crate::linalg::lu_decompose(&a)
            .and_then(|lu| lu.solve(&[d1, d2, 0.0]))
            .expect("line solve");
        (Vec3::new(sol[0], sol[1], sol[2]), dir)
    }

    /// Normal and offset of an IPNS plane (n . x = d).
    #[must_use]
    pub fn plane_normal_distance(pl: &Multivector) -> (Vec3, f64) {
        let n = Vec3::new(pl.coeffs[0b00001], pl.coeffs[0b00010], pl.coeffs[0b00100]);
        let nn = n.magnitude().max(1e-30);
        let d = -pl.scalar_product(&e_0()) / nn;
        (n * (1.0 / nn), d)
    }

    /// Classify an IPNS object by grade and flatness.
    #[must_use]
    pub fn classify(x: &Multivector) -> CgaObject {
        let grades = x.grades();
        if grades.len() != 1 {
            return CgaObject::Unknown;
        }
        let flat = e_inf().inner(x).norm() < 1e-9 * x.norm().max(1e-30);
        match grades[0] {
            1 => {
                if flat {
                    CgaObject::Plane
                } else {
                    let (_, r) = sphere_center_radius(x);
                    let w = -x.scalar_product(&e_inf());
                    let r2 = x.scale(1.0 / w).scalar_product(&x.scale(1.0 / w));
                    if r2 < -1e-9 {
                        CgaObject::ImaginarySphere
                    } else if r < 1e-6 {
                        CgaObject::Point
                    } else {
                        CgaObject::Sphere
                    }
                }
            }
            2 => {
                if flat {
                    CgaObject::Line
                } else {
                    CgaObject::Circle
                }
            }
            3 => {
                if flat {
                    CgaObject::Ideal
                } else {
                    CgaObject::PointPair
                }
            }
            _ => CgaObject::Unknown,
        }
    }

    /// Euclidean distance between two conformal points: d^2 = -2 A . B.
    #[must_use]
    pub fn distance(a: &Multivector, b: &Multivector) -> f64 {
        let wa = -a.scalar_product(&e_inf());
        let wb = -b.scalar_product(&e_inf());
        (-2.0 * a.scale(1.0 / wa).scalar_product(&b.scale(1.0 / wb)))
            .max(0.0)
            .sqrt()
    }

    /// True when the point lies strictly inside the sphere.
    #[must_use]
    pub fn is_inside_sphere(p: &Multivector, s: &Multivector) -> bool {
        let wp = -p.scalar_product(&e_inf());
        let ws = -s.scalar_product(&e_inf());
        p.scale(1.0 / wp).scalar_product(&s.scale(1.0 / ws)) > 0.0
    }

    /// Translator versor: T = 1 - (1/2) t e_inf.
    #[must_use]
    pub fn translator(t: Vec3) -> Multivector {
        Multivector::scalar(1.0, 4, 1, 0).sub(&evec(t).geometric(&e_inf()).scale(0.5))
    }

    /// Euclidean rotor about an axis through the origin.
    #[must_use]
    pub fn rotor(axis: Vec3, angle: f64) -> Multivector {
        let a = axis.normalized();
        // bivector dual to the axis within the Euclidean subalgebra
        let b = e(1).geometric(&e(2)).scale(a.x)
            .add(&e(2).geometric(&e(0)).scale(a.y))
            .add(&e(0).geometric(&e(1)).scale(a.z));
        b.scale(-0.5 * angle).exp()
    }

    /// Dilator scaling by `scale` about the origin.
    #[must_use]
    pub fn dilator(scale: f64) -> Multivector {
        let ec = e_inf().wedge(&e_0());
        ec.scale(-0.5 * scale.ln()).exp()
    }

    /// Transversor (special conformal) versor.
    #[must_use]
    pub fn transversor(v: Vec3) -> Multivector {
        Multivector::scalar(1.0, 4, 1, 0).add(&e_0().geometric(&evec(v)))
    }

    /// Inversion versor in a sphere (the sphere itself acts by sandwich).
    #[must_use]
    pub fn inversion_in_sphere(s: &Multivector) -> Multivector {
        s.clone()
    }

    /// Rigid motor: translation then rotation.
    #[must_use]
    pub fn motor(t: Vec3, axis: Vec3, angle: f64) -> Multivector {
        translator(t).geometric(&rotor(axis, angle))
    }

    /// Conformal versor for a similarity transform.
    #[must_use]
    pub fn conformal_from_similarity(s: &Sim3) -> Multivector {
        let (axis, angle) = s.r.to_quat().to_axis_angle();
        translator(s.t)
            .geometric(&rotor(axis, angle))
            .geometric(&dilator(s.s))
    }

    /// Apply a versor by the sandwich product (with the grade involution
    /// for odd versors such as spheres and planes).
    #[must_use]
    pub fn apply(versor: &Multivector, x: &Multivector) -> Multivector {
        let odd = versor.grades().iter().all(|g| g % 2 == 1);
        let target = if odd { x.grade_involution() } else { x.clone() };
        let out = versor.geometric(&target).geometric(&versor.reverse());
        // normalize scale using the reverse-square of the versor
        let s = versor.geometric(&versor.reverse()).coeffs[0];
        if s.abs() > 1e-30 {
            out.scale(1.0 / s)
        } else {
            out
        }
    }

    /// Sphere inversion as a reflection: S X S normalized.
    #[must_use]
    pub fn reflect_in_sphere(x: &Multivector, s: &Multivector) -> Multivector {
        apply(s, x)
    }

    /// Linear versor interpolation with renormalization.
    #[must_use]
    pub fn interpolate_versor(a: &Multivector, b: &Multivector, t: f64) -> Multivector {
        let mix = a.scale(1.0 - t).add(&b.scale(t));
        let n = mix.norm();
        if n > 1e-12 {
            mix.scale(1.0 / n)
        } else {
            mix
        }
    }

    /// Apollonius problem: spheres tangent to three given spheres (solved
    /// in Euclidean form, returned as IPNS spheres).
    #[must_use]
    pub fn apollonius_problem(
        c1: &Multivector,
        c2: &Multivector,
        c3: &Multivector,
    ) -> Vec<Multivector> {
        let (p1, r1) = sphere_center_radius(c1);
        let (p2, r2) = sphere_center_radius(c2);
        let (p3, r3) = sphere_center_radius(c3);
        let mut out = Vec::new();
        for signs in 0..8u32 {
            let s1 = if signs & 1 == 0 { 1.0 } else { -1.0 };
            let s2 = if signs & 2 == 0 { 1.0 } else { -1.0 };
            let s3 = if signs & 4 == 0 { 1.0 } else { -1.0 };
            // |c - p_i| = r + s_i r_i; subtract pairs to linearize:
            // -2 c.(p_i - p_j) + (p_i^2 - p_j^2) = 2 r (s_i r_i - s_j r_j)
            //   + (s_i r_i)^2 - (s_j r_j)^2
            // a . c = 2 (ra - rb) r + (ra^2 - rb^2) - (|pa|^2 - |pb|^2)
            // with a = -2 (pa - pb)
            let row = |pa: Vec3, ra: f64, pb: Vec3, rb: f64| {
                (
                    (pa - pb) * -2.0,
                    2.0 * (ra - rb),
                    ra * ra - rb * rb - pa.magnitude_squared() + pb.magnitude_squared(),
                )
            };
            let (a1, b1, k1) = row(p1, s1 * r1, p2, s2 * r2);
            let (a2, b2, k2) = row(p1, s1 * r1, p3, s3 * r3);
            // c = c0 + r * cdir from the two linear equations plus a third
            // free direction; solve with least squares over (c, r): 2 eqs,
            // 4 unknowns -> parametrize c = alpha + r beta via pseudo-solve:
            // pick the 2x2 system in the plane spanned by a1, a2
            // solve A [c; r] = k as under-determined; use direct approach:
            // write c = u + r v with A u = k - 0, A v = -b
            // minimum-norm solution of the underdetermined 2x3 system:
            // x = A^T (A A^T)^-1 k
            let min_norm = |k1: f64, k2: f64| -> Option<Vec3> {
                let g11 = a1.dot(&a1);
                let g12 = a1.dot(&a2);
                let g22 = a2.dot(&a2);
                let det = g11 * g22 - g12 * g12;
                if det.abs() < 1e-12 {
                    return None;
                }
                let y1 = (g22 * k1 - g12 * k2) / det;
                let y2 = (-g12 * k1 + g11 * k2) / det;
                Some(a1 * y1 + a2 * y2)
            };
            let u = match min_norm(k1, k2) {
                Some(v) => v,
                None => continue,
            };
            let v = match min_norm(b1, b2) {
                Some(w) => w,
                None => continue,
            };
            // With three spheres in 3D the tangent spheres form a
            // one-parameter family; take the canonical member whose center
            // lies in the affine span of the linear system (the centers'
            // plane) and solve the single remaining tangency for r.
            let g = |rr: f64| {
                let c = u + v * rr;
                (c - p1).magnitude() - (rr + s1 * r1)
            };
            let mut r = (r1 + r2 + r3) / 3.0;
            let mut ok = false;
            for _ in 0..100 {
                let val = g(r);
                if val.abs() < 1e-12 {
                    ok = true;
                    break;
                }
                let h = 1e-7;
                let dg = (g(r + h) - val) / h;
                if dg.abs() < 1e-14 {
                    break;
                }
                r -= val / dg;
            }
            if !ok || r <= 1e-9 || !r.is_finite() {
                continue;
            }
            let c = u + v * r;
            // dedup
            if out.iter().all(|s: &Multivector| {
                let (cc, rr) = sphere_center_radius(s);
                (cc - c).magnitude() > 1e-6 || (rr - r).abs() > 1e-6
            }) {
                out.push(sphere(c, r));
            }
        }
        out
    }

    /// The circle in which two spheres intersect.
    #[must_use]
    pub fn circle_through_intersection(s1: &Multivector, s2: &Multivector) -> Multivector {
        s1.wedge(s2)
    }

    /// Tangent plane to a sphere at a point on it.
    #[must_use]
    pub fn tangent_at(surface: &Multivector, pt: &Multivector) -> Multivector {
        let (c, _r) = sphere_center_radius(surface);
        let p = down(pt).expect("finite point");
        let n = (p - c).normalized();
        plane(n, n.dot(&p))
    }

    /// The flat carrier of a round: the plane containing a circle.
    #[must_use]
    pub fn carrier(x: &Multivector) -> Multivector {
        let (center, _r, n) = circle_center_radius_normal(x);
        plane(n, n.dot(&center))
    }

    /// IPNS <-> OPNS dualization.
    #[must_use]
    pub fn dual_cga(x: &Multivector) -> Multivector {
        x.dual()
    }

    /// Length of the tangent from a point to a sphere.
    #[must_use]
    pub fn point_to_sphere_tangent_distance(p: &Multivector, s: &Multivector) -> f64 {
        let (c, r) = sphere_center_radius(s);
        let pv = down(p).expect("finite point");
        ((pv - c).magnitude_squared() - r * r).max(0.0).sqrt()
    }

    /// Inverse stereographic projection R3 -> S3 via the conformal model.
    #[must_use]
    pub fn stereographic_via_cga(p: Vec3) -> Vec4 {
        let r2 = p.magnitude_squared();
        let s = 1.0 / (r2 + 1.0);
        Vec4::new(2.0 * p.x * s, 2.0 * p.y * s, 2.0 * p.z * s, (r2 - 1.0) * s)
    }
}

// ---------------------------------------------------------------------------
// sta: spacetime algebra Cl(1,3)
// ---------------------------------------------------------------------------

/// Spacetime algebra Cl(1, 3): gamma_0 squares to +1 (bit 0), the spatial
/// gamma_i to -1 (bits 1..3). Relative vectors are the bivectors
/// sigma_i = gamma_i gamma_0.
pub mod sta {
    use super::{Complex, Multivector, Vec3};
    use crate::quaternion::Quaternion;

    fn gamma(i: usize) -> Multivector {
        Multivector::basis_blade(1 << i, 1, 3, 0)
    }

    fn sigma(i: usize) -> Multivector {
        gamma(i).geometric(&gamma(0))
    }

    /// Spacetime event t gamma_0 + x . gamma.
    #[must_use]
    pub fn event(t: f64, x: Vec3) -> Multivector {
        gamma(0)
            .scale(t)
            .add(&gamma(1).scale(x.x))
            .add(&gamma(2).scale(x.y))
            .add(&gamma(3).scale(x.z))
    }

    /// Boost rotor for velocity v (|v| < 1, c = 1).
    #[must_use]
    pub fn boost(v: Vec3) -> Multivector {
        let speed = v.magnitude();
        if speed < 1e-15 {
            return Multivector::scalar(1.0, 1, 3, 0);
        }
        let alpha = rapidity(speed);
        let n = v * (1.0 / speed);
        let b = sigma(1).scale(n.x).add(&sigma(2).scale(n.y)).add(&sigma(3).scale(n.z));
        // R = cosh(a/2) - sinh(a/2) B
        Multivector::scalar((0.5 * alpha).cosh(), 1, 3, 0).sub(&b.scale((0.5 * alpha).sinh()))
    }

    /// Spatial rotation rotor.
    #[must_use]
    pub fn rotation(axis: Vec3, angle: f64) -> Multivector {
        let a = axis.normalized();
        // spatial bivectors: gamma_j gamma_k
        let b = gamma(3).geometric(&gamma(2)).scale(a.x)
            .add(&gamma(1).geometric(&gamma(3)).scale(a.y))
            .add(&gamma(2).geometric(&gamma(1)).scale(a.z));
        b.scale(-0.5 * angle).exp()
    }

    /// Apply a Lorentz rotor to an event: R e R~.
    #[must_use]
    pub fn lorentz_apply(r: &Multivector, e: &Multivector) -> Multivector {
        r.sandwich(e)
    }

    /// Faraday bivector F = E . sigma + I B . sigma.
    #[must_use]
    pub fn bivector_em(e_field: Vec3, b_field: Vec3) -> Multivector {
        let i = Multivector::pseudoscalar(1, 3, 0);
        let es = sigma(1)
            .scale(e_field.x)
            .add(&sigma(2).scale(e_field.y))
            .add(&sigma(3).scale(e_field.z));
        let bs = sigma(1)
            .scale(b_field.x)
            .add(&sigma(2).scale(b_field.y))
            .add(&sigma(3).scale(b_field.z));
        es.add(&i.geometric(&bs))
    }

    /// Electromagnetic invariants from F^2 = (E^2 - B^2) + 2 (E . B) I:
    /// returns (E^2 - B^2, E . B).
    #[must_use]
    pub fn em_invariants(f: &Multivector) -> (f64, f64) {
        let f2 = f.geometric(f);
        let s = f2.coeffs[0];
        let pseudo = f2.coeffs[0b1111];
        (s, 0.5 * pseudo)
    }

    /// Lorentz force: dp/dtau = q F . v (grade-1 contraction), for a
    /// particle of charge q and mass m returns the 4-acceleration.
    #[must_use]
    pub fn lorentz_force_sta(
        f: &Multivector,
        velocity: &Multivector,
        q: f64,
        m: f64,
    ) -> Multivector {
        f.geometric(velocity)
            .sub(&velocity.geometric(f))
            .scale(0.5 * q / m)
            .grade(1)
    }

    /// Proper time along a piecewise-linear worldline of events.
    #[must_use]
    pub fn proper_time(path: &[Multivector]) -> f64 {
        let mut tau = 0.0;
        for w in path.windows(2) {
            let d = w[1].sub(&w[0]);
            let dt = d.coeffs[0b0001];
            let dx = Vec3::new(d.coeffs[0b0010], d.coeffs[0b0100], d.coeffs[0b1000]);
            let s2 = dt * dt - dx.magnitude_squared();
            tau += s2.max(0.0).sqrt();
        }
        tau
    }

    /// Rapidity of a speed: atanh(v).
    #[must_use]
    pub fn rapidity(v: f64) -> f64 {
        v.atanh()
    }

    /// Split an event into (time, space) relative to an observer 4-velocity
    /// (default observer: gamma_0).
    #[must_use]
    pub fn spacetime_split(x: &Multivector, observer: &Multivector) -> (f64, Vec3) {
        // x gamma_obs = t + spatial sigma components
        let prod = x.geometric(observer);
        let t = prod.coeffs[0];
        // sigma_i = gamma_i gamma_0 = -(gamma_0 gamma_i): the canonical
        // coefficients (masks (1<<i)|1, increasing-index order) carry a
        // minus sign relative to the sigma components
        let sx = -prod.coeffs[0b0011];
        let sy = -prod.coeffs[0b0101];
        let sz = -prod.coeffs[0b1001];
        (t, Vec3::new(sx, sy, sz))
    }

    /// Residual of Maxwell's equation nabla F = J at a spacetime point,
    /// with nabla = gamma^mu d_mu (reciprocal frame: gamma^0 = gamma_0,
    /// gamma^i = -gamma_i) by central differences with step h.
    #[must_use]
    pub fn maxwell_residual(
        f: &dyn Fn(&[f64; 4]) -> Multivector,
        j: &Multivector,
        x: &[f64; 4],
        h: f64,
    ) -> Multivector {
        let mut nabla_f = Multivector::zero(1, 3, 0);
        for mu in 0..4 {
            let mut xp = *x;
            let mut xm = *x;
            xp[mu] += h;
            xm[mu] -= h;
            let df = f(&xp).sub(&f(&xm)).scale(1.0 / (2.0 * h));
            let g = if mu == 0 {
                gamma(0)
            } else {
                gamma(mu).scale(-1.0)
            };
            nabla_f = nabla_f.add(&g.geometric(&df));
        }
        nabla_f.sub(j)
    }

    /// The Dirac gamma matrices (Dirac basis) as 4x4 complex matrices.
    #[must_use]
    pub fn dirac_gamma_matrices() -> [[[Complex; 4]; 4]; 4] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let i = Complex::new(0.0, 1.0);
        let n = Complex::new(-1.0, 0.0);
        let ni = Complex::new(0.0, -1.0);
        [
            // gamma^0 = diag(1, 1, -1, -1)
            [[o, z, z, z], [z, o, z, z], [z, z, n, z], [z, z, z, n]],
            // gamma^1
            [[z, z, z, o], [z, z, o, z], [z, n, z, z], [n, z, z, z]],
            // gamma^2
            [[z, z, z, ni], [z, z, i, z], [z, i, z, z], [ni, z, z, z]],
            // gamma^3
            [[z, z, o, z], [z, z, z, n], [n, z, z, z], [z, o, z, z]],
        ]
    }

    /// Map a Pauli/quaternion rotation to the STA spatial rotor.
    #[must_use]
    pub fn pauli_to_sta(q: &Quaternion) -> Multivector {
        let (axis, angle) = q.normalize().to_axis_angle();
        rotation(axis, angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn random_mv(p: usize, q: usize, r: usize, rng: &mut Rng) -> Multivector {
        let mut m = Multivector::zero(p, q, r);
        for c in m.coeffs.iter_mut() {
            *c = rng.next_gaussian();
        }
        m
    }

    #[test]
    fn test_geometric_product_associative() {
        let mut rng = Rng::new(7);
        for &(p, q, r) in &[(3usize, 0usize, 0usize), (1, 3, 0), (3, 0, 1), (4, 1, 0)] {
            for _ in 0..4 {
                let a = random_mv(p, q, r, &mut rng);
                let b = random_mv(p, q, r, &mut rng);
                let c = random_mv(p, q, r, &mut rng);
                let lhs = a.geometric(&b).geometric(&c);
                let rhs = a.geometric(&b.geometric(&c));
                let err = lhs.sub(&rhs).norm();
                assert!(err < 1e-10, "associativity in Cl({p},{q},{r}): {err}");
                // distributivity
                let d1 = a.geometric(&b.add(&c));
                let d2 = a.geometric(&b).add(&a.geometric(&c));
                assert!(d1.sub(&d2).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_small_algebra_isomorphisms() {
        // Cl(0,1): e1^2 = -1, complex numbers
        let e1 = Multivector::basis_blade(1, 0, 1, 0);
        let sq = e1.geometric(&e1);
        assert!((sq.coeffs[0] + 1.0).abs() < 1e-15, "i^2 = -1");
        assert_eq!(is_isomorphic_to_known(0, 1, 0), "complex");
        // Cl(0,2): quaternions: i = e1, j = e2, k = e12
        let i = Multivector::basis_blade(0b01, 0, 2, 0);
        let j = Multivector::basis_blade(0b10, 0, 2, 0);
        let k = i.geometric(&j);
        assert!((i.geometric(&i).coeffs[0] + 1.0).abs() < 1e-15);
        assert!((j.geometric(&j).coeffs[0] + 1.0).abs() < 1e-15);
        assert!((k.geometric(&k).coeffs[0] + 1.0).abs() < 1e-15, "k^2 = -1");
        // ij = k, jk = i, ki = j
        assert!(i.geometric(&j).sub(&k).norm() < 1e-15);
        let jk = j.geometric(&k);
        assert!(jk.sub(&i).norm() < 1e-15, "jk = i");
        let ki = k.geometric(&i);
        assert!(ki.sub(&j).norm() < 1e-15, "ki = j");
        assert_eq!(is_isomorphic_to_known(0, 2, 0), "quaternions");
        // Cl(1,0): split-complex e1^2 = +1
        let e = Multivector::basis_blade(1, 1, 0, 0);
        assert!((e.geometric(&e).coeffs[0] - 1.0).abs() < 1e-15);
        // Cl(0,0,1): dual numbers eps^2 = 0
        let eps = Multivector::basis_blade(1, 0, 0, 1);
        assert!(eps.geometric(&eps).norm() < 1e-15);
        assert_eq!(algebra_dimension(4, 1, 0), 32);
    }

    #[test]
    fn test_products_and_involutions() {
        let mut rng = Rng::new(11);
        let a = random_mv(3, 0, 0, &mut rng);
        let b = random_mv(3, 0, 0, &mut rng);
        // wedge antisymmetry on vectors
        let v1 = a.grade(1);
        let v2 = b.grade(1);
        let w12 = v1.wedge(&v2);
        let w21 = v2.wedge(&v1);
        assert!(w12.add(&w21).norm() < 1e-12);
        // wedge with self vanishes
        assert!(v1.wedge(&v1).norm() < 1e-12);
        // geometric = inner + wedge for vectors
        let g = v1.geometric(&v2);
        let split = v1.inner(&v2).add(&w12);
        assert!(g.sub(&split).norm() < 1e-12);
        // reverse reverses products
        let ab_rev = a.geometric(&b).reverse();
        let ba_rev = b.reverse().geometric(&a.reverse());
        assert!(ab_rev.sub(&ba_rev).norm() < 1e-10);
        // dual/undual roundtrip (Cl(3,0) and PGA)
        for &(p, q, r) in &[(3usize, 0usize, 0usize), (3, 0, 1)] {
            let x = random_mv(p, q, r, &mut rng);
            let round = x.dual().undual();
            assert!(round.sub(&x).norm() < 1e-10, "dual roundtrip Cl({p},{q},{r})");
        }
        // projection + rejection = identity
        let blade = v1.wedge(&v2); // plane blade
        let x = random_mv(3, 0, 0, &mut rng).grade(1);
        let recon = x.project_onto_blade(&blade).add(&x.reject_from_blade(&blade));
        assert!(recon.sub(&x).norm() < 1e-10);
        // projection lies in the blade (wedge vanishes)
        assert!(x.project_onto_blade(&blade).wedge(&blade).norm() < 1e-10);
        // blade factorization reassembles the blade direction
        let factors = blade.normalized().blade_factor();
        assert_eq!(factors.len(), 2);
        let re_wedge = factors[0].wedge(&factors[1]);
        // same blade up to sign and scale
        let cosang = re_wedge.scalar_product(&blade.normalized()).abs()
            / re_wedge.norm();
        assert!(cosang > 1.0 - 1e-9, "blade factor alignment {cosang}");
        // is_blade / is_versor
        assert!(blade.is_blade());
        assert!(!a.is_blade() || a.grades().len() == 1);
        let rot = cl3::rotor(Vec3::new(0.0, 0.0, 1.0), 0.7);
        assert!(rot.is_versor());
    }

    #[test]
    fn test_rotors_match_quaternions() {
        let mut rng = Rng::new(3);
        for _ in 0..6 {
            let axis = Vec3::new(
                rng.next_gaussian(),
                rng.next_gaussian(),
                rng.next_gaussian(),
            )
            .normalized();
            let angle = 2.0 * rng.next_f64();
            let v = Vec3::new(
                rng.next_gaussian(),
                rng.next_gaussian(),
                rng.next_gaussian(),
            );
            let r = cl3::rotor(axis, angle);
            let by_rotor = cl3::rotate(v, &r);
            let q = Quaternion::from_axis_angle(axis, angle);
            let by_quat = q.rotate_vec(v);
            assert!(
                (by_rotor - by_quat).magnitude() < 1e-10,
                "rotor vs quaternion: {by_rotor:?} vs {by_quat:?}"
            );
            // quaternion conversion roundtrip
            let qq = r.to_quaternion().unwrap();
            let r2 = Multivector::from_quaternion(&qq);
            assert!(r.sub(&r2).norm() < 1e-12);
            // rotor log/exp roundtrip
            let l = r.log().unwrap();
            let r3 = l.exp();
            assert!(r.sub(&r3).norm() < 1e-10, "rotor log/exp");
        }
        // rotor_from_vectors takes a to b
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 1.0).normalized();
        let r = Multivector::rotor_from_vectors(&cl3::vec(a), &cl3::vec(b));
        let img = cl3::rotate(a, &r);
        assert!((img - b).magnitude() < 1e-10);
        // rotor interpolation midpoint has half the angle
        let r1 = cl3::rotor(Vec3::new(0.0, 0.0, 1.0), 0.0);
        let r2 = cl3::rotor(Vec3::new(0.0, 0.0, 1.0), 1.0);
        let mid = r1.rotor_interpolate(&r2, 0.5);
        let img2 = cl3::rotate(Vec3::new(1.0, 0.0, 0.0), &mid);
        assert!((img2.y.atan2(img2.x) - 0.5).abs() < 1e-10);
        // cross product via wedge
        let cx = cl3::cross_via_wedge(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!((cx - Vec3::new(0.0, 0.0, 1.0)).magnitude() < 1e-12);
        // reflection
        let refl = cl3::reflect(Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!((refl - Vec3::new(1.0, -1.0, 0.0)).magnitude() < 1e-12);
        // to_string_blades sanity
        let s = cl3::vec(Vec3::new(1.0, 0.0, -1.0)).to_string_blades();
        assert!(s.contains("e1") && s.contains("e3"), "{s}");
    }

    #[test]
    fn test_matrix_rep_and_meet_join() {
        // matrix representation is multiplicative
        let mut rng = Rng::new(5);
        let a = random_mv(2, 1, 0, &mut rng);
        let b = random_mv(2, 1, 0, &mut rng);
        let ma = a.to_matrix_rep();
        let mb = b.to_matrix_rep();
        let mab = a.geometric(&b).to_matrix_rep();
        let prod = ma.mul(&mb).unwrap();
        let mut err = 0.0_f64;
        for i in 0..8 {
            for j in 0..8 {
                err = err.max((prod.get(i, j) - mab.get(i, j)).abs());
            }
        }
        assert!(err < 1e-10, "matrix rep homomorphism {err}");
        // meet of two planes in Cl(3,0) is their common line direction
        let p1 = cl3::vec(Vec3::new(0.0, 0.0, 1.0)).dual(); // xy plane bivector
        let p2 = cl3::vec(Vec3::new(0.0, 1.0, 0.0)).dual(); // xz plane bivector
        let line = p1.meet(&p2);
        // expect the x axis direction (grade 1)
        let lv = cl3::to_vec3(&line.normalized());
        assert!(lv.is_some());
        let lv = lv.unwrap();
        assert!(lv.y.abs() < 1e-9 && lv.z.abs() < 1e-9 && lv.x.abs() > 0.9);
        // join of two vectors is their plane
        let j = cl3::vec(Vec3::new(1.0, 0.0, 0.0)).join(&cl3::vec(Vec3::new(0.0, 1.0, 0.0)));
        assert_eq!(j.grades(), vec![2]);
    }

    #[test]
    fn test_pga3_basics() {
        use super::pga3;
        // point roundtrip
        let p = Vec3::new(1.0, -2.0, 0.5);
        let pt = pga3::point(p);
        let back = pga3::to_vec3(&pt).unwrap();
        assert!((back - p).magnitude() < 1e-12, "point roundtrip {back:?}");
        // plane-point distance
        let pl = pga3::plane(Vec3::new(0.0, 0.0, 1.0), -1.0); // z = 1
        let d = pga3::distance_point_plane(&pga3::point(Vec3::new(5.0, 3.0, 4.0)), &pl);
        assert!((d.abs() - 3.0).abs() < 1e-9, "point-plane {d}");
        // meet of two planes is a line; meet with a third gives a point
        let px = pga3::plane(Vec3::new(1.0, 0.0, 0.0), -1.0); // x = 1
        let py = pga3::plane(Vec3::new(0.0, 1.0, 0.0), -2.0); // y = 2
        let line = pga3::line_from_planes(&px, &py);
        let pz = pga3::plane(Vec3::new(0.0, 0.0, 1.0), -3.0); // z = 3
        let ppt = pga3::meet(&line, &pz);
        let v = pga3::to_vec3(&ppt).unwrap();
        assert!((v - Vec3::new(1.0, 2.0, 3.0)).magnitude() < 1e-9, "triple meet {v:?}");
        // join of two points, then distance from a third point
        let l = pga3::line_from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let dl = pga3::distance_point_line(&pga3::point(Vec3::new(0.5, 2.0, 0.0)), &l);
        assert!((dl - 2.0).abs() < 1e-9, "point-line {dl}");
        // plane from three points
        let p3 = pga3::plane_from_points(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        );
        let dp = pga3::distance_point_plane(&pga3::point(Vec3::new(7.0, -2.0, 4.0)), &p3);
        assert!((dp.abs() - 3.0).abs() < 1e-9, "plane from points {dp}");
        // angles
        let a = pga3::angle_planes(
            &pga3::plane(Vec3::new(1.0, 0.0, 0.0), 0.0),
            &pga3::plane(Vec3::new(0.0, 1.0, 0.0), 0.0),
        );
        assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let al = pga3::angle_lines(
            &pga3::line_from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            &pga3::line_from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)),
        );
        assert!((al - std::f64::consts::FRAC_PI_4).abs() < 1e-9);
        // skew line distance
        let l1 = pga3::line_from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = pga3::line_from_points(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 1.0, 2.0));
        assert!((pga3::distance_lines(&l1, &l2) - 2.0).abs() < 1e-9);
        // ideal points
        assert!(pga3::is_ideal(&pga3::point_at_infinity(Vec3::new(1.0, 0.0, 0.0))));
        assert!(!pga3::is_ideal(&pt));
        // projections
        let proj = pga3::project_point_on_plane(&pga3::point(Vec3::new(1.0, 1.0, 5.0)), &pl);
        let pv = pga3::to_vec3(&proj).unwrap();
        assert!((pv - Vec3::new(1.0, 1.0, 1.0)).magnitude() < 1e-9);
        let projl = pga3::project_point_on_line(&pga3::point(Vec3::new(0.3, 4.0, 0.0)), &l);
        let plv = pga3::to_vec3(&projl).unwrap();
        assert!((plv - Vec3::new(0.3, 0.0, 0.0)).magnitude() < 1e-9);
    }

    #[test]
    fn test_pga3_motors_match_se3() {
        use super::pga3;
        use crate::manifold::lie::{Se3, So3};
        let mut rng = Rng::new(23);
        // translation motor
        let t = Vec3::new(0.5, -1.0, 2.0);
        let tm = pga3::motor_translation(t);
        let img = pga3::to_vec3(&pga3::motor_apply(&tm, &pga3::point(Vec3::new(1.0, 1.0, 1.0))))
            .unwrap();
        assert!(
            (img - Vec3::new(1.5, 0.0, 3.0)).magnitude() < 1e-9,
            "translator image {img:?}"
        );
        // general motor matches Se3 action on points
        for _ in 0..5 {
            let se = Se3 {
                r: So3::random(&mut rng),
                t: Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian()),
            };
            let m = pga3::motor_from_se3(&se);
            let x = Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian());
            let via_motor =
                pga3::to_vec3(&pga3::motor_apply(&m, &pga3::point(x))).unwrap();
            let via_se3 = se.apply_point(x);
            assert!(
                (via_motor - via_se3).magnitude() < 1e-9,
                "motor vs Se3: {via_motor:?} vs {via_se3:?}"
            );
            // roundtrip
            let se2 = pga3::motor_to_se3(&m);
            assert!(se.distance(&se2, 1.0) < 1e-9, "motor_to_se3 roundtrip");
        }
        // rotation about an off-origin axis: points on the axis are fixed
        let axis = pga3::line_from_points(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 1.0));
        let rm = pga3::motor_rotation(&axis, 1.2);
        let fixed = pga3::to_vec3(&pga3::motor_apply(&rm, &pga3::point(Vec3::new(1.0, 0.0, 0.5))))
            .unwrap();
        assert!((fixed - Vec3::new(1.0, 0.0, 0.5)).magnitude() < 1e-9);
        // screw motor: interpolation endpoint correctness
        let m0 = pga3::motor_from_se3(&Se3::identity());
        let m1 = pga3::motor_screw(&axis, 0.8, 0.5);
        let half = pga3::motor_interpolate(&m0, &m1, 1.0);
        let x = Vec3::new(0.0, 1.0, 0.0);
        let a1 = pga3::to_vec3(&pga3::motor_apply(&m1, &pga3::point(x))).unwrap();
        let a2 = pga3::to_vec3(&pga3::motor_apply(&half, &pga3::point(x))).unwrap();
        assert!((a1 - a2).magnitude() < 1e-9);
        // rigid body step advances the motor smoothly and keeps it a versor
        let mut motor = m0.clone();
        let rate = Multivector::basis_blade(0b0011, 3, 0, 1).scale(2.0); // spin about z
        for _ in 0..10 {
            pga3::rigid_body_step(&mut motor, &rate, 0.05);
        }
        assert!(motor.is_versor());
        let spun = pga3::to_vec3(&pga3::motor_apply(&motor, &pga3::point(Vec3::new(1.0, 0.0, 0.0))))
            .unwrap();
        assert!((spun.magnitude() - 1.0).abs() < 1e-9, "rotation preserves radius");
        // inertia map and forque are well-formed
        let momentum = pga3::inertia_dual_map(&rate, [2.0, 3.0, 4.0], 1.5);
        assert!((momentum.coeffs[0b0011] - rate.coeffs[0b0011] * 4.0).abs() < 1e-12);
        let fq = pga3::forque(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(fq.norm() > 0.0);
    }

    #[test]
    fn test_cga3_objects() {
        use super::cga3;
        // distance between points matches Euclidean
        let a = Vec3::new(1.0, 2.0, -0.5);
        let b = Vec3::new(-1.0, 0.5, 2.0);
        let d = cga3::distance(&cga3::point(a), &cga3::point(b));
        assert!((d - (a - b).magnitude()).abs() < 1e-10, "cga distance {d}");
        // down(point(p)) roundtrip
        assert!((cga3::down(&cga3::point(a)).unwrap() - a).magnitude() < 1e-12);
        // sphere extraction roundtrip
        let s = cga3::sphere(Vec3::new(0.5, -1.0, 2.0), 1.7);
        let (c, r) = cga3::sphere_center_radius(&s);
        assert!((c - Vec3::new(0.5, -1.0, 2.0)).magnitude() < 1e-10);
        assert!((r - 1.7).abs() < 1e-10);
        // meet of two spheres is a circle with the analytic center/radius
        let s1 = cga3::sphere(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let s2 = cga3::sphere(Vec3::new(1.2, 0.0, 0.0), 1.0);
        let circ = cga3::meet(&s1, &s2);
        assert_eq!(cga3::classify(&circ), cga3::CgaObject::Circle);
        let (cc, cr, cn) = cga3::circle_center_radius_normal(&circ);
        assert!((cc - Vec3::new(0.6, 0.0, 0.0)).magnitude() < 1e-9, "circle center {cc:?}");
        let exact_r = (1.0_f64 - 0.36).sqrt();
        assert!((cr - exact_r).abs() < 1e-9, "circle radius {cr} vs {exact_r}");
        assert!(cn.x.abs() > 0.999, "circle normal {cn:?}");
        // circle through three points
        let c3 = cga3::circle_from_points(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
        );
        let (uc, ur, un) = cga3::circle_center_radius_normal(&c3);
        assert!(uc.magnitude() < 1e-9 && (ur - 1.0).abs() < 1e-9);
        assert!(un.z.abs() > 0.999);
        // line extraction
        let l = cga3::line_from_points(Vec3::new(0.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0));
        assert_eq!(cga3::classify(&l), cga3::CgaObject::Line);
        let (lp, ld) = cga3::line_point_direction(&l);
        assert!(ld.x.abs() > 0.999, "line dir {ld:?}");
        assert!((lp.y - 1.0).abs() < 1e-9 && lp.z.abs() < 1e-9, "line point {lp:?}");
        // plane classification and extraction
        let pl = cga3::plane(Vec3::new(0.0, 0.0, 1.0), 2.0);
        assert_eq!(cga3::classify(&pl), cga3::CgaObject::Plane);
        let (pn, pd) = cga3::plane_normal_distance(&pl);
        assert!(pn.z > 0.999 && (pd - 2.0).abs() < 1e-9);
        // inside test
        assert!(cga3::is_inside_sphere(&cga3::point(Vec3::new(0.1, 0.0, 0.0)), &s1));
        assert!(!cga3::is_inside_sphere(&cga3::point(Vec3::new(2.0, 0.0, 0.0)), &s1));
        // tangent distance: 3-4-5 triangle
        let s5 = cga3::sphere(Vec3::new(0.0, 0.0, 0.0), 3.0);
        let td = cga3::point_to_sphere_tangent_distance(&cga3::point(Vec3::new(5.0, 0.0, 0.0)), &s5);
        assert!((td - 4.0).abs() < 1e-10);
        // carrier of a circle is its plane
        let car = cga3::carrier(&c3);
        let (cnorm, cd) = cga3::plane_normal_distance(&car);
        assert!(cnorm.z.abs() > 0.999 && cd.abs() < 1e-9);
        // sphere through four points
        let s4 = cga3::sphere_from_points(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let (c4, r4) = cga3::sphere_center_radius(&s4);
        assert!(c4.magnitude() < 1e-9 && (r4 - 1.0).abs() < 1e-9);
        // tangent plane at a point of a sphere
        let tp = cga3::tangent_at(&s1, &cga3::point(Vec3::new(1.0, 0.0, 0.0)));
        let (tn, tdd) = cga3::plane_normal_distance(&tp);
        assert!(tn.x > 0.999 && (tdd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cga3_versors() {
        use super::cga3;
        // translator
        let t = cga3::translator(Vec3::new(1.0, 2.0, 3.0));
        let img = cga3::down(&cga3::apply(&t, &cga3::point(Vec3::new(0.5, 0.0, 0.0)))).unwrap();
        assert!((img - Vec3::new(1.5, 2.0, 3.0)).magnitude() < 1e-10, "translator {img:?}");
        // rotor
        let r = cga3::rotor(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let ri = cga3::down(&cga3::apply(&r, &cga3::point(Vec3::new(1.0, 0.0, 0.0)))).unwrap();
        assert!((ri - Vec3::new(0.0, 1.0, 0.0)).magnitude() < 1e-10, "rotor {ri:?}");
        // dilator: scales distances from origin
        let dl = cga3::dilator(2.0);
        let di = cga3::down(&cga3::apply(&dl, &cga3::point(Vec3::new(1.0, 1.0, 0.0)))).unwrap();
        let scale = di.magnitude() / Vec3::new(1.0, 1.0, 0.0).magnitude();
        assert!(
            (scale - 2.0).abs() < 1e-9 || (scale - 0.5).abs() < 1e-9,
            "dilator scale {scale}"
        );
        // motor: translation + rotation composition applies to spheres too
        let m = cga3::motor(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.7);
        let ms = cga3::apply(&m, &cga3::sphere(Vec3::new(0.0, 0.0, 0.0), 0.5));
        let (mc, mr) = cga3::sphere_center_radius(&ms);
        assert!((mr - 0.5).abs() < 1e-9, "sphere radius preserved {mr}");
        assert!((mc - Vec3::new(1.0, 0.0, 0.0)).magnitude() < 1e-9, "sphere center moved {mc:?}");
        // inversion in the unit sphere: r -> 1/r
        let s = cga3::sphere(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let inv = cga3::reflect_in_sphere(&cga3::point(Vec3::new(2.0, 0.0, 0.0)), &s);
        let iv = cga3::down(&inv).unwrap();
        assert!((iv - Vec3::new(0.5, 0.0, 0.0)).magnitude() < 1e-9, "inversion {iv:?}");
        // conformal from similarity
        use crate::manifold::lie::Sim3;
        let sim = Sim3 {
            s: 1.5,
            r: crate::manifold::lie::So3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.3),
            t: Vec3::new(0.5, 0.0, 0.0),
        };
        let cv = cga3::conformal_from_similarity(&sim);
        let x = Vec3::new(1.0, -0.5, 0.7);
        let via_cga = cga3::down(&cga3::apply(&cv, &cga3::point(x))).unwrap();
        let via_sim = sim.apply(x);
        assert!((via_cga - via_sim).magnitude() < 1e-8, "{via_cga:?} vs {via_sim:?}");
        // versor interpolation stays versor-like
        let vi = cga3::interpolate_versor(&cga3::translator(Vec3::new(1.0, 0.0, 0.0)), &cga3::translator(Vec3::new(3.0, 0.0, 0.0)), 0.5);
        let vip = cga3::down(&cga3::apply(&vi, &cga3::point(Vec3::new(0.0, 0.0, 0.0)))).unwrap();
        assert!((vip.x - 2.0).abs() < 0.2, "interp translator {vip:?}");
        // transversor maps points to points (possibly ideal)
        let tv = cga3::transversor(Vec3::new(0.1, 0.0, 0.0));
        let tvp = cga3::apply(&tv, &cga3::point(Vec3::new(1.0, 1.0, 0.0)));
        assert_eq!(tvp.grades(), vec![1]);
        // stereographic via cga lands on S3
        let s3 = cga3::stereographic_via_cga(Vec3::new(0.3, -0.4, 0.7));
        assert!((s3.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cga3_apollonius() {
        use super::cga3;
        // three spheres in a plane; tangent spheres exist
        let s1 = cga3::sphere(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let s2 = cga3::sphere(Vec3::new(4.0, 0.0, 0.0), 1.0);
        let s3 = cga3::sphere(Vec3::new(2.0, 3.0, 0.0), 1.0);
        let sols = cga3::apollonius_problem(&s1, &s2, &s3);
        assert!(!sols.is_empty(), "no Apollonius solutions");
        for sol in &sols {
            let (c, r) = cga3::sphere_center_radius(sol);
            for (sc, sr) in [
                (Vec3::new(0.0, 0.0, 0.0), 1.0),
                (Vec3::new(4.0, 0.0, 0.0), 1.0),
                (Vec3::new(2.0, 3.0, 0.0), 1.0),
            ] {
                let d = (c - sc).magnitude();
                let tangent =
                    (d - (r + sr)).abs() < 1e-9 || (d - (r - sr).abs()).abs() < 1e-9;
                assert!(tangent, "not tangent: d={d}, r={r}, sr={sr}");
            }
        }
    }

    #[test]
    fn test_sta() {
        use super::sta;
        // boost matches the Lorentz matrix
        let v = 0.6;
        let gamma = 1.0 / (1.0_f64 - v * v).sqrt();
        let b = sta::boost(Vec3::new(v, 0.0, 0.0));
        let e = sta::event(1.0, Vec3::new(0.5, 0.0, 0.0));
        let e2 = sta::lorentz_apply(&b, &e);
        let (t2, x2) = sta::spacetime_split(&e2, &Multivector::basis_blade(1, 1, 3, 0));
        // active boost: t' = gamma (t - v x)... sign convention: check both
        let expect_t = gamma * (1.0 - v * 0.5);
        let expect_x = gamma * (0.5 - v * 1.0);
        let alt_t = gamma * (1.0 + v * 0.5);
        let alt_x = gamma * (0.5 + v * 1.0);
        let matches_minus = (t2 - expect_t).abs() < 1e-10 && (x2.x - expect_x).abs() < 1e-10;
        let matches_plus = (t2 - alt_t).abs() < 1e-10 && (x2.x - alt_x).abs() < 1e-10;
        assert!(matches_minus || matches_plus, "boost: t'={t2}, x'={x2:?}");
        // interval invariance
        let s_before = 1.0 - 0.25;
        let s_after = t2 * t2 - x2.magnitude_squared();
        assert!((s_before - s_after).abs() < 1e-10, "interval invariant");
        // rotation acts on the spatial part only
        let r = sta::rotation(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let er = sta::lorentz_apply(&r, &sta::event(2.0, Vec3::new(1.0, 0.0, 0.0)));
        let (tr, xr) = sta::spacetime_split(&er, &Multivector::basis_blade(1, 1, 3, 0));
        assert!((tr - 2.0).abs() < 1e-10);
        assert!((xr.y.abs() - 1.0).abs() < 1e-10 && xr.x.abs() < 1e-10, "rotation {xr:?}");
        // EM invariants: values and boost invariance
        let ef = Vec3::new(0.3, -0.2, 0.5);
        let bf = Vec3::new(0.1, 0.4, -0.3);
        let f = sta::bivector_em(ef, bf);
        let (i1, i2) = sta::em_invariants(&f);
        assert!(
            (i1 - (ef.magnitude_squared() - bf.magnitude_squared())).abs() < 1e-10,
            "E^2-B^2: {i1}"
        );
        assert!((i2.abs() - ef.dot(&bf).abs()) < 1e-10, "E.B: {i2}");
        let fb = b.sandwich(&f);
        let (j1, j2) = sta::em_invariants(&fb);
        assert!((i1 - j1).abs() < 1e-9 && (i2.abs() - j2.abs()).abs() < 1e-9, "invariants under boost");
        // proper time: moving clock runs slow
        let path = vec![
            sta::event(0.0, Vec3::new(0.0, 0.0, 0.0)),
            sta::event(1.0, Vec3::new(0.6, 0.0, 0.0)),
        ];
        assert!((sta::proper_time(&path) - 0.8).abs() < 1e-12);
        assert!((sta::rapidity(0.6) - 0.6931471805599453).abs() < 1e-10);
        // Lorentz force: static E field accelerates along E
        let fe = sta::bivector_em(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
        let u = sta::event(1.0, Vec3::new(0.0, 0.0, 0.0)); // rest 4-velocity
        let acc = sta::lorentz_force_sta(&fe, &u, 1.0, 1.0);
        let (at, ax) = sta::spacetime_split(&acc, &Multivector::basis_blade(1, 1, 3, 0));
        let _ = at;
        assert!(ax.x.abs() > 0.9 && ax.y.abs() < 1e-12, "Lorentz force {ax:?}");
        // Maxwell residual vanishes for a constant field with no current
        let f_const = move |_: &[f64; 4]| sta::bivector_em(ef, bf);
        let j0 = Multivector::zero(1, 3, 0);
        let res = sta::maxwell_residual(&f_const, &j0, &[0.0, 0.0, 0.0, 0.0], 1e-4);
        assert!(res.norm() < 1e-9, "constant-field Maxwell residual");
        // gamma matrices: anticommutators {g^mu, g^nu} = 2 eta
        let g = sta::dirac_gamma_matrices();
        let mul = |a: &[[Complex; 4]; 4], b: &[[Complex; 4]; 4]| {
            let mut out = [[Complex::new(0.0, 0.0); 4]; 4];
            for (i, oi) in out.iter_mut().enumerate() {
                for (j, oij) in oi.iter_mut().enumerate() {
                    for k in 0..4 {
                        *oij = *oij + a[i][k] * b[k][j];
                    }
                }
            }
            out
        };
        let eta = [1.0, -1.0, -1.0, -1.0];
        for mu in 0..4 {
            for nu in 0..4 {
                let ac1 = mul(&g[mu], &g[nu]);
                let ac2 = mul(&g[nu], &g[mu]);
                for i in 0..4 {
                    for j in 0..4 {
                        let sum = ac1[i][j] + ac2[i][j];
                        let want = if mu == nu && i == j { 2.0 * eta[mu] } else { 0.0 };
                        assert!(
                            (sum.re - want).abs() < 1e-12 && sum.im.abs() < 1e-12,
                            "anticommutator ({mu},{nu})"
                        );
                    }
                }
            }
        }
        // pauli_to_sta rotor rotates the same way
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.8);
        let rq = sta::pauli_to_sta(&q);
        let ev = sta::lorentz_apply(&rq, &sta::event(0.0, Vec3::new(1.0, 0.0, 0.0)));
        let (_, xv) = sta::spacetime_split(&ev, &Multivector::basis_blade(1, 1, 3, 0));
        let expect = q.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            ((xv - expect).magnitude() < 1e-9) || ((xv + expect).magnitude() < 1e-9),
            "pauli rotor {xv:?} vs {expect:?}"
        );
    }
}
