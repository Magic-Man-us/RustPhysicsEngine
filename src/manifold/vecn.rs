//! n-dimensional vectors and arbitrary-rank tensors: the generic machinery
//! behind the metric-driven differential geometry in this module tree.

use crate::error::SolveError;
use crate::linalg::Matrix;
use crate::math::{Vec2, Vec3};
use crate::monte_carlo::Rng;

// ---------------------------------------------------------------------------
// VecN
// ---------------------------------------------------------------------------

/// A dense n-dimensional vector.
#[derive(Debug, Clone, PartialEq)]
pub struct VecN {
    pub data: Vec<f64>,
}

impl VecN {
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    #[must_use]
    pub fn ones(n: usize) -> Self {
        Self { data: vec![1.0; n] }
    }

    /// The i-th standard basis vector in n dimensions.
    #[must_use]
    pub fn unit(n: usize, i: usize) -> Self {
        let mut v = Self::zeros(n);
        v.data[i] = 1.0;
        v
    }

    #[must_use]
    pub fn from(slice: &[f64]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn dot(&self, other: &VecN) -> f64 {
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    #[must_use]
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    #[must_use]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            self.clone()
        } else {
            self.scale(1.0 / n)
        }
    }

    #[must_use]
    pub fn add(&self, other: &VecN) -> Self {
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    #[must_use]
    pub fn sub(&self, other: &VecN) -> Self {
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    #[must_use]
    pub fn scale(&self, k: f64) -> Self {
        Self {
            data: self.data.iter().map(|a| a * k).collect(),
        }
    }

    /// Outer product a b^T as a matrix.
    #[must_use]
    pub fn outer(&self, other: &VecN) -> Matrix {
        Matrix::from_fn(self.dim(), other.dim(), |i, j| self.data[i] * other.data[j])
    }

    /// Projection of self onto `other`.
    #[must_use]
    pub fn project_onto(&self, other: &VecN) -> Self {
        let d = other.dot(other);
        if d == 0.0 {
            return Self::zeros(self.dim());
        }
        other.scale(self.dot(other) / d)
    }

    #[must_use]
    pub fn angle_between(&self, other: &VecN) -> f64 {
        (self.dot(other) / (self.norm() * other.norm())).clamp(-1.0, 1.0).acos()
    }

    #[must_use]
    pub fn lerp(&self, other: &VecN, t: f64) -> Self {
        self.scale(1.0 - t).add(&other.scale(t))
    }

    /// Cross product, defined only in three dimensions.
    #[must_use]
    pub fn cross_3d(&self, other: &VecN) -> Option<Vec3> {
        if self.dim() != 3 || other.dim() != 3 {
            return None;
        }
        let (a, b) = (&self.data, &other.data);
        Some(Vec3::new(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ))
    }

    #[must_use]
    pub fn to_vec3(&self) -> Option<Vec3> {
        if self.dim() == 3 {
            Some(Vec3::new(self.data[0], self.data[1], self.data[2]))
        } else {
            None
        }
    }

    #[must_use]
    pub fn to_vec2(&self) -> Option<Vec2> {
        if self.dim() == 2 {
            Some(Vec2::new(self.data[0], self.data[1]))
        } else {
            None
        }
    }

    /// Gram-Schmidt orthonormalization; near-dependent vectors are dropped.
    #[must_use]
    pub fn gram_schmidt(vectors: &[VecN]) -> Vec<VecN> {
        let mut basis: Vec<VecN> = Vec::new();
        for v in vectors {
            let mut w = v.clone();
            for b in &basis {
                w = w.sub(&b.scale(w.dot(b)));
            }
            let n = w.norm();
            if n > 1e-12 {
                basis.push(w.scale(1.0 / n));
            }
        }
        basis
    }

    /// Uniform random direction on the unit (n-1)-sphere.
    #[must_use]
    pub fn random_unit(n: usize, rng: &mut Rng) -> Self {
        Self::random_gaussian(n, rng).normalized()
    }

    /// Standard normal components.
    #[must_use]
    pub fn random_gaussian(n: usize, rng: &mut Rng) -> Self {
        Self {
            data: (0..n).map(|_| rng.next_gaussian()).collect(),
        }
    }
}

impl std::ops::Add for VecN {
    type Output = VecN;
    fn add(self, rhs: VecN) -> VecN {
        VecN::add(&self, &rhs)
    }
}

impl std::ops::Sub for VecN {
    type Output = VecN;
    fn sub(self, rhs: VecN) -> VecN {
        VecN::sub(&self, &rhs)
    }
}

impl std::ops::Mul<f64> for VecN {
    type Output = VecN;
    fn mul(self, k: f64) -> VecN {
        self.scale(k)
    }
}

impl std::ops::Neg for VecN {
    type Output = VecN;
    fn neg(self) -> VecN {
        self.scale(-1.0)
    }
}

impl std::ops::Index<usize> for VecN {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

// ---------------------------------------------------------------------------
// TensorN
// ---------------------------------------------------------------------------

/// A dense tensor of arbitrary rank, stored row-major (last index fastest).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorN {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn flat_index(shape: &[usize], idx: &[usize]) -> usize {
    debug_assert_eq!(shape.len(), idx.len());
    let st = strides(shape);
    idx.iter().zip(&st).map(|(&i, &s)| i * s).sum()
}

/// Iterate all multi-indices of `shape`, calling `f` with each.
fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut idx = vec![0usize; shape.len()];
    loop {
        f(&idx);
        let mut k = shape.len();
        loop {
            if k == 0 {
                return;
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
}

impl TensorN {
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product::<usize>().max(1)],
        }
    }

    #[must_use]
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![1.0; shape.iter().product::<usize>().max(1)],
        }
    }

    /// Rank-2 identity (Kronecker delta) of dimension n.
    #[must_use]
    pub fn identity_2(n: usize) -> Self {
        Self::from_fn(&[n, n], |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 })
    }

    #[must_use]
    pub fn from_fn(shape: &[usize], f: impl Fn(&[usize]) -> f64) -> Self {
        let mut t = Self::zeros(shape);
        let st = strides(shape);
        for_each_index(shape, |idx| {
            let flat: usize = idx.iter().zip(&st).map(|(&i, &s)| i * s).sum();
            t.data[flat] = f(idx);
        });
        t
    }

    #[must_use]
    pub fn from_matrix(m: &Matrix) -> Self {
        Self {
            shape: vec![m.rows, m.cols],
            data: m.data.clone(),
        }
    }

    /// Rank-2 tensors convert back to a matrix.
    #[must_use]
    pub fn to_matrix(&self) -> Option<Matrix> {
        if self.rank() != 2 {
            return None;
        }
        Some(Matrix {
            rows: self.shape[0],
            cols: self.shape[1],
            data: self.data.clone(),
        })
    }

    #[must_use]
    pub fn get(&self, idx: &[usize]) -> f64 {
        self.data[flat_index(&self.shape, idx)]
    }

    pub fn set(&mut self, idx: &[usize], v: f64) {
        let f = flat_index(&self.shape, idx);
        self.data[f] = v;
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Trace over indices `i` and `j` (which must have equal extent),
    /// producing a tensor of rank two less. A rank-2 trace yields a rank-0
    /// tensor (shape `[]` with a single entry).
    #[must_use]
    pub fn contract(&self, i: usize, j: usize) -> TensorN {
        assert_ne!(i, j);
        assert_eq!(self.shape[i], self.shape[j]);
        let (lo, hi) = (i.min(j), i.max(j));
        let n = self.shape[i];
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != lo && k != hi)
            .map(|(_, &s)| s)
            .collect();
        let mut out = TensorN::zeros(&out_shape);
        let out_st = strides(&out_shape);
        let mut full = vec![0usize; self.rank()];
        for_each_index(&out_shape, |idx| {
            let mut m = 0;
            for (k, slot) in full.iter_mut().enumerate() {
                if k == lo || k == hi {
                    continue;
                }
                *slot = idx[m];
                m += 1;
            }
            let mut sum = 0.0;
            for d in 0..n {
                full[lo] = d;
                full[hi] = d;
                sum += self.get(&full);
            }
            let flat: usize = idx.iter().zip(&out_st).map(|(&a, &s)| a * s).sum();
            out.data[flat] = sum;
        });
        out
    }

    /// Tensor (outer) product: rank adds.
    #[must_use]
    pub fn tensor_product(&self, other: &TensorN) -> TensorN {
        let mut shape = self.shape.clone();
        shape.extend_from_slice(&other.shape);
        let mut out = TensorN::zeros(&shape);
        for (a, &va) in self.data.iter().enumerate() {
            for (b, &vb) in other.data.iter().enumerate() {
                out.data[a * other.data.len() + b] = va * vb;
            }
        }
        out
    }

    /// Generalized matrix multiplication: contract index `i_self` of self
    /// with index `j_other` of other.
    #[must_use]
    pub fn contract_with(&self, other: &TensorN, i_self: usize, j_other: usize) -> TensorN {
        let prod = self.tensor_product(other);
        prod.contract(i_self, self.rank() + j_other)
    }

    /// Permute indices: `perm[k]` names which original axis becomes axis k.
    #[must_use]
    pub fn transpose(&self, perm: &[usize]) -> TensorN {
        assert_eq!(perm.len(), self.rank());
        let new_shape: Vec<usize> = perm.iter().map(|&p| self.shape[p]).collect();
        let mut out = TensorN::zeros(&new_shape);
        let out_st = strides(&new_shape);
        let mut src = vec![0usize; self.rank()];
        for_each_index(&new_shape, |idx| {
            for (k, &p) in perm.iter().enumerate() {
                src[p] = idx[k];
            }
            let flat: usize = idx.iter().zip(&out_st).map(|(&a, &s)| a * s).sum();
            out.data[flat] = self.get(&src);
        });
        out
    }

    /// Symmetrize over indices i and j.
    #[must_use]
    pub fn symmetrize(&self, i: usize, j: usize) -> TensorN {
        let mut perm: Vec<usize> = (0..self.rank()).collect();
        perm.swap(i, j);
        let t = self.transpose(&perm);
        self.add(&t).scale(0.5)
    }

    /// Antisymmetrize over indices i and j.
    #[must_use]
    pub fn antisymmetrize(&self, i: usize, j: usize) -> TensorN {
        let mut perm: Vec<usize> = (0..self.rank()).collect();
        perm.swap(i, j);
        let t = self.transpose(&perm);
        self.sub(&t).scale(0.5)
    }

    /// True when exchange of indices i and j leaves the tensor unchanged
    /// within `tol`.
    #[must_use]
    pub fn is_symmetric(&self, i: usize, j: usize, tol: f64) -> bool {
        let mut perm: Vec<usize> = (0..self.rank()).collect();
        perm.swap(i, j);
        let t = self.transpose(&perm);
        self.data
            .iter()
            .zip(&t.data)
            .all(|(a, b)| (a - b).abs() <= tol)
    }

    /// Raise index `i` with the inverse metric.
    #[must_use]
    pub fn raise_index(&self, i: usize, metric_inv: &Matrix) -> TensorN {
        self.apply_metric(i, metric_inv)
    }

    /// Lower index `i` with the metric.
    #[must_use]
    pub fn lower_index(&self, i: usize, metric: &Matrix) -> TensorN {
        self.apply_metric(i, metric)
    }

    fn apply_metric(&self, i: usize, g: &Matrix) -> TensorN {
        let gt = TensorN::from_matrix(g);
        // contract g_{a b} with index i, then move the new axis (appended
        // last) back into position i
        let c = self.contract_with(&gt, i, 1);
        // axes of c: original axes without i, then the metric's free axis
        let r = self.rank();
        let mut perm: Vec<usize> = Vec::with_capacity(r);
        let mut m = 0;
        for k in 0..r {
            if k == i {
                perm.push(r - 1);
            } else {
                perm.push(m);
                m += 1;
            }
        }
        c.transpose(&perm)
    }

    /// Fix `axis` to `idx`, dropping that axis.
    #[must_use]
    pub fn slice(&self, axis: usize, idx: usize) -> TensorN {
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != axis)
            .map(|(_, &s)| s)
            .collect();
        let mut out = TensorN::zeros(&out_shape);
        let out_st = strides(&out_shape);
        let mut full = vec![0usize; self.rank()];
        for_each_index(&out_shape, |oidx| {
            let mut m = 0;
            for (k, slot) in full.iter_mut().enumerate() {
                if k == axis {
                    *slot = idx;
                } else {
                    *slot = oidx[m];
                    m += 1;
                }
            }
            let flat: usize = oidx.iter().zip(&out_st).map(|(&a, &s)| a * s).sum();
            out.data[flat] = self.get(&full);
        });
        out
    }

    #[must_use]
    pub fn norm_frobenius(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[must_use]
    pub fn map(&self, f: impl Fn(f64) -> f64) -> TensorN {
        TensorN {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    #[must_use]
    pub fn add(&self, other: &TensorN) -> TensorN {
        assert_eq!(self.shape, other.shape);
        TensorN {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    #[must_use]
    pub fn sub(&self, other: &TensorN) -> TensorN {
        assert_eq!(self.shape, other.shape);
        TensorN {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    #[must_use]
    pub fn scale(&self, k: f64) -> TensorN {
        TensorN {
            shape: self.shape.clone(),
            data: self.data.iter().map(|v| v * k).collect(),
        }
    }

    /// Einstein summation over named indices, e.g. `"ij,jk->ik"`,
    /// `"ijk,k->ij"`, `"ii->"`. Each operand's index string length must
    /// match its rank; repeated labels are summed.
    pub fn einsum(spec: &str, tensors: &[&TensorN]) -> Result<TensorN, SolveError> {
        let (lhs, rhs) = spec
            .split_once("->")
            .ok_or(SolveError::InvalidArgument("einsum spec needs ->"))?;
        let operands: Vec<&str> = lhs.split(',').collect();
        if operands.len() != tensors.len() {
            return Err(SolveError::InvalidArgument(
                "einsum operand count mismatch",
            ));
        }
        // map labels to extents
        let mut extent: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for (labels, t) in operands.iter().zip(tensors) {
            if labels.chars().count() != t.rank() {
                return Err(SolveError::InvalidArgument("einsum label/rank mismatch"));
            }
            for (c, &s) in labels.chars().zip(&t.shape) {
                if let Some(&e) = extent.get(&c) {
                    if e != s {
                        return Err(SolveError::InvalidArgument("einsum extent mismatch"));
                    }
                } else {
                    extent.insert(c, s);
                }
            }
        }
        let out_labels: Vec<char> = rhs.chars().collect();
        for c in &out_labels {
            if !extent.contains_key(c) {
                return Err(SolveError::InvalidArgument("einsum unknown output label"));
            }
        }
        let mut sum_labels: Vec<char> = extent
            .keys()
            .filter(|c| !out_labels.contains(c))
            .copied()
            .collect();
        sum_labels.sort_unstable();
        let out_shape: Vec<usize> = out_labels.iter().map(|c| extent[c]).collect();
        let sum_shape: Vec<usize> = sum_labels.iter().map(|c| extent[c]).collect();
        let mut out = TensorN::zeros(&out_shape);
        let out_st = strides(&out_shape);
        let mut assign: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for_each_index(&out_shape, |oidx| {
            for (k, &c) in out_labels.iter().enumerate() {
                assign.insert(c, oidx[k]);
            }
            let mut total = 0.0;
            for_each_index(&sum_shape, |sidx| {
                for (k, &c) in sum_labels.iter().enumerate() {
                    assign.insert(c, sidx[k]);
                }
                let mut prod = 1.0;
                for (labels, t) in operands.iter().zip(tensors) {
                    let idx: Vec<usize> = labels.chars().map(|c| assign[&c]).collect();
                    prod *= t.get(&idx);
                }
                total += prod;
            });
            let flat: usize = oidx.iter().zip(&out_st).map(|(&a, &s)| a * s).sum();
            out.data[flat] = total;
        });
        Ok(out)
    }

    /// The rank-n Levi-Civita symbol in n dimensions.
    #[must_use]
    pub fn levi_civita(n: usize) -> TensorN {
        let shape = vec![n; n];
        TensorN::from_fn(&shape, |idx| {
            // permutation sign, zero on repeats
            let mut seen = vec![false; n];
            for &i in idx {
                if seen[i] {
                    return 0.0;
                }
                seen[i] = true;
            }
            let mut perm: Vec<usize> = idx.to_vec();
            let mut sign = 1.0;
            for i in 0..n {
                while perm[i] != i {
                    let j = perm[i];
                    perm.swap(i, j);
                    sign = -sign;
                }
            }
            sign
        })
    }

    /// Kronecker delta as a rank-2 tensor.
    #[must_use]
    pub fn kronecker(n: usize) -> TensorN {
        Self::identity_2(n)
    }

    /// Hodge dual of a fully antisymmetric rank-k tensor with respect to
    /// `metric`: (*T)_{j...} = (1/k!) sqrt|g| T^{i...} eps_{i... j...}, with
    /// indices raised by the inverse metric.
    #[must_use]
    pub fn hodge_dual_vector(&self, metric: &Matrix) -> TensorN {
        let n = metric.rows;
        let k = self.rank();
        assert!(k <= n);
        let lu = crate::linalg::lu_decompose(metric).expect("metric must be invertible");
        let ginv = lu.inverse().expect("metric must be invertible");
        let detg = lu.determinant();
        let mut up = self.clone();
        for i in 0..k {
            up = up.raise_index(i, &ginv);
        }
        let eps = TensorN::levi_civita(n).scale(detg.abs().sqrt());
        let kfact: f64 = (1..=k).map(|v| v as f64).product::<f64>().max(1.0);
        let out_shape = vec![n; n - k];
        let mut out = TensorN::zeros(&out_shape);
        let out_st = strides(&out_shape);
        let sum_shape = vec![n; k];
        let mut eps_idx = vec![0usize; n];
        for_each_index(&out_shape.clone(), |oidx| {
            let mut total = 0.0;
            for_each_index(&sum_shape, |sidx| {
                for (m, &v) in sidx.iter().enumerate() {
                    eps_idx[m] = v;
                }
                for (m, &v) in oidx.iter().enumerate() {
                    eps_idx[k + m] = v;
                }
                total += up.get(sidx) * eps.get(&eps_idx);
            });
            let flat: usize = oidx.iter().zip(&out_st).map(|(&a, &s)| a * s).sum();
            out.data[flat] = total / kfact;
        });
        out
    }
}

/// Wedge product of antisymmetric forms: antisymmetrization of the tensor
/// product with the standard combinatorial normalization
/// (a ^ b)_{i...j...} = (p+q)!/(p! q!) Alt(a (x) b).
#[must_use]
pub fn wedge(a: &TensorN, b: &TensorN) -> TensorN {
    let p = a.rank();
    let q = b.rank();
    let prod = a.tensor_product(b);
    let r = p + q;
    // full antisymmetrization over all r indices
    let mut out = TensorN::zeros(&prod.shape);
    for (perm, sign) in permutations_list(r) {
        out = out.add(&prod.transpose(&perm).scale(sign));
    }
    let pfact: f64 = (1..=p).map(|v| v as f64).product::<f64>().max(1.0);
    let qfact: f64 = (1..=q).map(|v| v as f64).product::<f64>().max(1.0);
    out.scale(1.0 / (pfact * qfact))
}

/// Numerical exterior derivative of a k-form field at `p` by central
/// differences with step `h`: (d omega)_{i j...} = Alt(partial_i omega_{j...}).
#[must_use]
pub fn exterior_derivative_numeric(
    omega: &dyn Fn(&VecN) -> TensorN,
    p: &VecN,
    h: f64,
) -> TensorN {
    let n = p.dim();
    let sample = omega(p);
    let k = sample.rank();
    let mut d_shape = vec![n];
    d_shape.extend_from_slice(&sample.shape);
    // partial derivatives: axis 0 is the derivative index
    let mut dt = TensorN::zeros(&d_shape);
    for i in 0..n {
        let mut pp = p.clone();
        let mut pm = p.clone();
        pp.data[i] += h;
        pm.data[i] -= h;
        let diff = omega(&pp).sub(&omega(&pm)).scale(1.0 / (2.0 * h));
        // write into slice i of axis 0
        let st = strides(&d_shape);
        for_each_index(&sample.shape, |idx| {
            let mut full = Vec::with_capacity(k + 1);
            full.push(i);
            full.extend_from_slice(idx);
            let flat: usize = full.iter().zip(&st).map(|(&a, &s)| a * s).sum();
            dt.data[flat] = diff.get(idx);
        });
    }
    // antisymmetrize over all k+1 indices with (k+1) terms of the cyclic
    // alternation: full antisymmetrization
    let r = k + 1;
    let mut out = TensorN::zeros(&d_shape);
    for (perm, sign) in permutations_list(r) {
        out = out.add(&dt.transpose(&perm).scale(sign));
    }
    let rfact: f64 = (1..=r).map(|v| v as f64).product();
    out.scale((r as f64) / rfact)
}

/// Straightforward permutation generation with signs.
fn permutations_list(n: usize) -> Vec<(Vec<usize>, f64)> {
    fn go(current: &mut Vec<usize>, remaining: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, f64)>) {
        if remaining.is_empty() {
            let sign = permutation_sign(current);
            out.push((current.clone(), sign));
            return;
        }
        for i in 0..remaining.len() {
            let v = remaining.remove(i);
            current.push(v);
            go(current, remaining, out);
            current.pop();
            remaining.insert(i, v);
        }
    }
    let mut out = Vec::new();
    go(&mut Vec::new(), &mut (0..n).collect(), &mut out);
    out
}

fn permutation_sign(perm: &[usize]) -> f64 {
    let mut p = perm.to_vec();
    let mut sign = 1.0;
    for i in 0..p.len() {
        while p[i] != i {
            let j = p[i];
            p.swap(i, j);
            sign = -sign;
        }
    }
    sign
}

/// Determinant of a square matrix via LU decomposition.
#[must_use]
pub fn determinant_n(m: &Matrix) -> f64 {
    match crate::linalg::lu_decompose(m) {
        Ok(lu) => lu.determinant(),
        Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vecn_basics() {
        let a = VecN::from(&[1.0, 2.0, 2.0]);
        assert_eq!(a.dim(), 3);
        assert!((a.norm() - 3.0).abs() < 1e-15);
        assert!((a.normalized().norm() - 1.0).abs() < 1e-15);
        let b = VecN::unit(3, 0);
        assert!((a.dot(&b) - 1.0).abs() < 1e-15);
        assert!((a.angle_between(&a)).abs() < 1e-7);
        let c = a.clone().add(&b);
        assert!((c[0] - 2.0).abs() < 1e-15);
        // projection: a onto e0 = (1, 0, 0)
        let p = a.project_onto(&b);
        assert!((p[0] - 1.0).abs() < 1e-15 && p[1].abs() < 1e-15);
        // cross product matches Vec3
        let x = VecN::unit(3, 0);
        let y = VecN::unit(3, 1);
        let z = x.cross_3d(&y).unwrap();
        assert!((z.z - 1.0).abs() < 1e-15);
        assert!(VecN::zeros(4).cross_3d(&VecN::zeros(4)).is_none());
        // Gram-Schmidt produces an orthonormal set and drops dependents
        let vs = vec![
            VecN::from(&[1.0, 1.0, 0.0]),
            VecN::from(&[1.0, 0.0, 0.0]),
            VecN::from(&[2.0, 1.0, 0.0]), // dependent
        ];
        let basis = VecN::gram_schmidt(&vs);
        assert_eq!(basis.len(), 2);
        assert!(basis[0].dot(&basis[1]).abs() < 1e-12);
        assert!((basis[0].norm() - 1.0).abs() < 1e-12);
        // random unit vectors have unit norm
        let mut rng = Rng::new(7);
        let u = VecN::random_unit(5, &mut rng);
        assert!((u.norm() - 1.0).abs() < 1e-12);
        // operators
        let d = VecN::from(&[1.0, 2.0]) * 3.0;
        assert!((d[1] - 6.0).abs() < 1e-15);
        let e = -VecN::from(&[1.0, -2.0]);
        assert!((e[1] - 2.0).abs() < 1e-15);
        assert!(a.to_vec3().is_some() && a.to_vec2().is_none());
        let l = VecN::zeros(2).lerp(&VecN::ones(2), 0.25);
        assert!((l[0] - 0.25).abs() < 1e-15);
        let o = VecN::from(&[1.0, 2.0]).outer(&VecN::from(&[3.0, 4.0]));
        assert!((o.get(1, 0) - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_tensor_basics() {
        // einsum matrix product matches Matrix::mul
        let a = Matrix::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
        let b = Matrix::from_fn(4, 2, |i, j| (i as f64) - (j as f64));
        let ta = TensorN::from_matrix(&a);
        let tb = TensorN::from_matrix(&b);
        let prod = TensorN::einsum("ij,jk->ik", &[&ta, &tb]).unwrap();
        let exact = a.mul(&b).unwrap();
        for i in 0..3 {
            for k in 0..2 {
                assert!((prod.get(&[i, k]) - exact.get(i, k)).abs() < 1e-12);
            }
        }
        // contract of identity gives n
        let id = TensorN::identity_2(5);
        let tr = id.contract(0, 1);
        assert_eq!(tr.rank(), 0);
        assert!((tr.data[0] - 5.0).abs() < 1e-15);
        // einsum trace spelling
        let tr2 = TensorN::einsum("ii->", &[&id]).unwrap();
        assert!((tr2.data[0] - 5.0).abs() < 1e-15);
        // matrix-vector einsum
        let v = TensorN {
            shape: vec![4],
            data: vec![1.0, 0.0, -1.0, 2.0],
        };
        let mv = TensorN::einsum("ij,j->i", &[&ta, &v]).unwrap();
        let exact_mv = a.mul_vec(&v.data).unwrap();
        for i in 0..3 {
            assert!((mv.get(&[i]) - exact_mv[i]).abs() < 1e-12);
        }
        // transpose round trip
        let t3 = TensorN::from_fn(&[2, 3, 4], |idx| (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64);
        let tt = t3.transpose(&[2, 0, 1]);
        assert_eq!(tt.shape, vec![4, 2, 3]);
        assert!((tt.get(&[3, 1, 2]) - t3.get(&[1, 2, 3])).abs() < 1e-15);
        // slice
        let s = t3.slice(1, 2);
        assert_eq!(s.shape, vec![2, 4]);
        assert!((s.get(&[1, 3]) - t3.get(&[1, 2, 3])).abs() < 1e-15);
        // contract_with as generalized matmul
        let cw = ta.contract_with(&tb, 1, 0);
        assert!((cw.get(&[2, 1]) - exact.get(2, 1)).abs() < 1e-12);
    }

    #[test]
    fn test_raise_lower_roundtrip() {
        // non-trivial symmetric metric
        let g = Matrix::from_rows(&[&[2.0, 0.3, 0.0], &[0.3, 1.5, 0.1], &[0.0, 0.1, 1.0]])
            .unwrap();
        let ginv = crate::linalg::lu_decompose(&g).unwrap().inverse().unwrap();
        let t = TensorN::from_fn(&[3, 3], |idx| ((idx[0] + 1) * (idx[1] + 2)) as f64 * 0.37);
        for axis in 0..2 {
            let round = t.raise_index(axis, &ginv).lower_index(axis, &g);
            for (a, b) in round.data.iter().zip(&t.data) {
                assert!((a - b).abs() < 1e-12, "axis {axis}");
            }
        }
        // raising with the identity is a no-op
        let id = Matrix::identity(3);
        let same = t.raise_index(0, &id);
        assert!(same
            .data
            .iter()
            .zip(&t.data)
            .all(|(a, b)| (a - b).abs() < 1e-15));
    }

    #[test]
    fn test_levi_civita_and_wedge() {
        // eps contracted with itself gives n!
        for n in 2..=4 {
            let eps = TensorN::levi_civita(n);
            let total: f64 = eps.data.iter().map(|v| v * v).sum();
            let nfact: f64 = (1..=n).map(|v| v as f64).product();
            assert!((total - nfact).abs() < 1e-12, "n = {n}");
        }
        // wedge(a, a) = 0 for 1-forms
        let a = TensorN {
            shape: vec![3],
            data: vec![1.0, 2.0, 3.0],
        };
        let waa = wedge(&a, &a);
        assert!(waa.norm_frobenius() < 1e-14);
        // wedge of e1, e2 = antisymmetric with (1,2) entry 1
        let e1 = TensorN {
            shape: vec![3],
            data: vec![1.0, 0.0, 0.0],
        };
        let e2 = TensorN {
            shape: vec![3],
            data: vec![0.0, 1.0, 0.0],
        };
        let w = wedge(&e1, &e2);
        assert!((w.get(&[0, 1]) - 1.0).abs() < 1e-14);
        assert!((w.get(&[1, 0]) + 1.0).abs() < 1e-14);
        assert!(w.get(&[0, 0]).abs() < 1e-14);
        // antisymmetrize/symmetrize split reconstructs the tensor
        let t = TensorN::from_fn(&[3, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
        let rec = t.symmetrize(0, 1).add(&t.antisymmetrize(0, 1));
        assert!(rec
            .data
            .iter()
            .zip(&t.data)
            .all(|(a, b)| (a - b).abs() < 1e-14));
        assert!(t.symmetrize(0, 1).is_symmetric(0, 1, 1e-14));
    }

    #[test]
    fn test_hodge_and_exterior() {
        // Hodge dual of a 1-form in Euclidean 3-space: *dx = dy ^ dz
        let g = Matrix::identity(3);
        let dx = TensorN {
            shape: vec![3],
            data: vec![1.0, 0.0, 0.0],
        };
        let star = dx.hodge_dual_vector(&g);
        assert_eq!(star.shape, vec![3, 3]);
        assert!((star.get(&[1, 2]) - 1.0).abs() < 1e-12);
        assert!((star.get(&[2, 1]) + 1.0).abs() < 1e-12);
        // rank-0 dual: *1 = volume form (eps itself)
        let one = TensorN {
            shape: vec![],
            data: vec![2.0],
        };
        let vol = one.hodge_dual_vector(&g);
        assert!((vol.get(&[0, 1, 2]) - 2.0).abs() < 1e-12);
        // exterior derivative of f dx for f = y: d(y dx) = dy ^ dx
        let omega = |p: &VecN| TensorN {
            shape: vec![3],
            data: vec![p[1], 0.0, 0.0],
        };
        let d = exterior_derivative_numeric(&omega, &VecN::from(&[0.3, 0.7, -0.2]), 1e-5);
        // (d omega)_{yx} = +1/... convention: components antisymmetric with
        // d(y dx) = dy ^ dx = -dx ^ dy: entry (1,0) = +1, (0,1) = -1 up to
        // the alternation normalization; check antisymmetry and magnitude
        assert!((d.get(&[1, 0]) + d.get(&[0, 1])).abs() < 1e-8);
        assert!(d.get(&[1, 0]).abs() > 0.4, "magnitude {}", d.get(&[1, 0]));
        // d^2 = 0 numerically on a smooth 0-form -> 1-form -> 2-form chain
        let f_grad = |p: &VecN| {
            // omega = df for f = x^2 y: exact gradient
            TensorN {
                shape: vec![3],
                data: vec![2.0 * p[0] * p[1], p[0] * p[0], 0.0],
            }
        };
        let dd = exterior_derivative_numeric(&f_grad, &VecN::from(&[0.5, -0.3, 0.9]), 1e-4);
        assert!(dd.norm_frobenius() < 1e-6, "d^2 = {}", dd.norm_frobenius());
        // determinant via lu
        let m = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]).unwrap();
        assert!((determinant_n(&m) - 5.0).abs() < 1e-12);
    }
}
