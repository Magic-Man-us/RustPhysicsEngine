//! Metric geometry on n-dimensional manifolds: a metric is a function from
//! coordinates to a matrix g_ij, and everything else — Christoffel symbols,
//! Riemann/Ricci/Weyl curvature, covariant derivatives, geodesic machinery
//! inputs — is derived from it by finite differences.

use crate::linalg::{lu_decompose, Matrix};
use crate::manifold::vecn::{TensorN, VecN};
use crate::math::Vec3;

/// Signature convention for Minkowski-type metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sig {
    /// (-, +, +, ...): time first, mostly plus.
    MostlyPlus,
    /// (+, -, -, ...): time first, mostly minus.
    MostlyMinus,
}

/// A (pseudo-)Riemannian metric given by a coordinate chart function.
pub struct Metric {
    pub dim: usize,
    pub g: Box<dyn Fn(&VecN) -> Matrix>,
    /// Finite-difference step for all derived quantities.
    pub h: f64,
}

impl Metric {
    pub fn new(dim: usize, g: impl Fn(&VecN) -> Matrix + 'static) -> Self {
        Self {
            dim,
            g: Box::new(g),
            h: 1e-4,
        }
    }

    /// Flat Euclidean space in Cartesian coordinates.
    #[must_use]
    pub fn euclidean(n: usize) -> Self {
        Self::new(n, move |_| Matrix::identity(n))
    }

    /// Flat Minkowski space with the time coordinate first.
    #[must_use]
    pub fn minkowski(n: usize, signature: Sig) -> Self {
        let s = match signature {
            Sig::MostlyPlus => -1.0,
            Sig::MostlyMinus => 1.0,
        };
        Self::new(n, move |_| {
            Matrix::from_fn(n, n, |i, j| {
                if i != j {
                    0.0
                } else if i == 0 {
                    s
                } else {
                    -s
                }
            })
        })
    }

    /// Round n-sphere of radius `r` in hyperspherical angles
    /// (theta_1, ..., theta_n): g_ii = r^2 prod_{k<i} sin^2 theta_k.
    #[must_use]
    pub fn sphere(n: usize, r: f64) -> Self {
        Self::new(n, move |p: &VecN| {
            Matrix::from_fn(n, n, |i, j| {
                if i != j {
                    return 0.0;
                }
                let mut g = r * r;
                for k in 0..i {
                    let s = p[k].sin();
                    g *= s * s;
                }
                g
            })
        })
    }

    /// Poincare ball model of hyperbolic n-space (curvature -1):
    /// g = 4 delta / (1 - |x|^2)^2.
    #[must_use]
    pub fn hyperbolic_ball(n: usize) -> Self {
        Self::new(n, move |p: &VecN| {
            let r2 = p.dot(p);
            let f = 2.0 / (1.0 - r2);
            Matrix::from_fn(n, n, |i, j| if i == j { f * f } else { 0.0 })
        })
    }

    /// Poincare half-space model of hyperbolic n-space: g = delta / x_n^2
    /// (last coordinate positive).
    #[must_use]
    pub fn poincare_half_space(n: usize) -> Self {
        Self::new(n, move |p: &VecN| {
            let y = p[n - 1];
            Matrix::from_fn(n, n, |i, j| if i == j { 1.0 / (y * y) } else { 0.0 })
        })
    }

    /// Flat n-torus with circumference radii `radii` (angle coordinates).
    #[must_use]
    pub fn torus_flat(n: usize, radii: &[f64]) -> Self {
        let r = radii.to_vec();
        Self::new(n, move |_| {
            Matrix::from_fn(r.len(), r.len(), |i, j| if i == j { r[i] * r[i] } else { 0.0 })
        })
    }

    /// Schwarzschild exterior in coordinates (t, r, theta, phi), G = c = 1.
    #[must_use]
    pub fn schwarzschild(m: f64) -> Self {
        Self::new(4, schwarzschild_metric_fn(m))
    }

    /// Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi).
    #[must_use]
    pub fn kerr(m: f64, a: f64) -> Self {
        kerr_boyer_lindquist(m, a)
    }

    /// Spatially flat/open/closed FRW in (t, r, theta, phi) with scale
    /// factor `a(t)` and curvature parameter `k`.
    #[must_use]
    pub fn frw(a: fn(f64) -> f64, k: f64) -> Self {
        frw_metric(a, k)
    }

    /// De Sitter space in static coordinates with Hubble length `l`.
    #[must_use]
    pub fn de_sitter(l: f64) -> Self {
        Self::new(4, move |p: &VecN| {
            let r = p[1];
            let f = 1.0 - r * r / (l * l);
            spherical_static_metric(f, r, p[2])
        })
    }

    /// Anti-de Sitter space in static coordinates with AdS radius `l`.
    #[must_use]
    pub fn anti_de_sitter(l: f64) -> Self {
        Self::new(4, move |p: &VecN| {
            let r = p[1];
            let f = 1.0 + r * r / (l * l);
            spherical_static_metric(f, r, p[2])
        })
    }

    /// Kaluza-Klein 5D metric from a 4D metric, gauge potential `a_mu` and
    /// constant dilaton `phi`: g_{mu nu} + phi^2 A_mu A_nu, g_{mu 5} =
    /// phi^2 A_mu, g_55 = phi^2. The extra coordinate is last.
    #[must_use]
    pub fn kaluza_klein_5d(
        g4: Metric,
        a_mu: impl Fn(&VecN) -> VecN + 'static,
        phi: f64,
    ) -> Self {
        Self::new(5, move |p: &VecN| {
            let p4 = VecN::from(&p.data[..4]);
            let g = (g4.g)(&p4);
            let a = a_mu(&p4);
            Matrix::from_fn(5, 5, |i, j| match (i, j) {
                (4, 4) => phi * phi,
                (4, mu) => phi * phi * a[mu],
                (mu, 4) => phi * phi * a[mu],
                (mu, nu) => g.get(mu, nu) + phi * phi * a[mu] * a[nu],
            })
        })
    }

    /// Induced 2D metric of an embedded surface (u, v) -> R^3; the Gaussian
    /// curvature of this metric is the surface's curvature.
    #[must_use]
    pub fn gaussian_curvature_surface(f: fn(f64, f64) -> Vec3) -> Self {
        surface_metric_from_parametrization(f)
    }

    /// Induced metric from an embedding of an n-manifold into R^m:
    /// g_ij = d(embed)/dx^i . d(embed)/dx^j (finite differences).
    #[must_use]
    pub fn induced_from_embedding(dim: usize, embed: impl Fn(&VecN) -> VecN + 'static) -> Self {
        let h = 1e-5;
        Self::new(dim, move |p: &VecN| {
            let mut jac: Vec<VecN> = Vec::with_capacity(dim);
            for i in 0..dim {
                let mut pp = p.clone();
                let mut pm = p.clone();
                pp.data[i] += h;
                pm.data[i] -= h;
                jac.push(embed(&pp).sub(&embed(&pm)).scale(1.0 / (2.0 * h)));
            }
            Matrix::from_fn(dim, dim, |i, j| jac[i].dot(&jac[j]))
        })
    }

    // -- pointwise evaluation ------------------------------------------------

    #[must_use]
    pub fn at(&self, p: &VecN) -> Matrix {
        (self.g)(p)
    }

    #[must_use]
    pub fn inverse_at(&self, p: &VecN) -> Matrix {
        lu_decompose(&self.at(p))
            .and_then(|lu| lu.inverse())
            .expect("metric is singular at this point")
    }

    #[must_use]
    pub fn det_at(&self, p: &VecN) -> f64 {
        lu_decompose(&self.at(p)).map_or(0.0, |lu| lu.determinant())
    }

    /// Metric signature at `p`: (number of positive, number of negative)
    /// eigenvalues.
    #[must_use]
    pub fn signature(&self, p: &VecN) -> (usize, usize) {
        let g = self.at(p);
        let eig = crate::linalg::eigenvalues_general(&g, 500)
            .expect("eigenvalue iteration failed for metric signature");
        let pos = eig.iter().filter(|e| e.re > 0.0).count();
        let neg = eig.iter().filter(|e| e.re < 0.0).count();
        (pos, neg)
    }

    /// d g_ij / d x^k by central differences.
    #[must_use]
    pub fn dg(&self, p: &VecN, k: usize) -> Matrix {
        let mut pp = p.clone();
        let mut pm = p.clone();
        pp.data[k] += self.h;
        pm.data[k] -= self.h;
        let gp = self.at(&pp);
        let gm = self.at(&pm);
        Matrix::from_fn(self.dim, self.dim, |i, j| {
            (gp.get(i, j) - gm.get(i, j)) / (2.0 * self.h)
        })
    }

    /// Christoffel symbols of the second kind Gamma^i_{jk}, shape [n, n, n].
    #[must_use]
    pub fn christoffel(&self, p: &VecN) -> TensorN {
        let n = self.dim;
        let ginv = self.inverse_at(p);
        let dgs: Vec<Matrix> = (0..n).map(|k| self.dg(p, k)).collect();
        TensorN::from_fn(&[n, n, n], |idx| {
            let (i, j, k) = (idx[0], idx[1], idx[2]);
            let mut sum = 0.0;
            for l in 0..n {
                sum += ginv.get(i, l)
                    * (dgs[j].get(l, k) + dgs[k].get(l, j) - dgs[l].get(j, k));
            }
            0.5 * sum
        })
    }

    /// Christoffel symbols of the first kind Gamma_{ijk} =
    /// (1/2)(d_j g_ik + d_k g_ij - d_i g_jk).
    #[must_use]
    pub fn christoffel_first_kind(&self, p: &VecN) -> TensorN {
        let n = self.dim;
        let dgs: Vec<Matrix> = (0..n).map(|k| self.dg(p, k)).collect();
        TensorN::from_fn(&[n, n, n], |idx| {
            let (i, j, k) = (idx[0], idx[1], idx[2]);
            0.5 * (dgs[j].get(i, k) + dgs[k].get(i, j) - dgs[i].get(j, k))
        })
    }

    /// Riemann tensor R^i_{jkl} = d_k Gamma^i_{lj} - d_l Gamma^i_{kj}
    /// + Gamma^i_{km} Gamma^m_{lj} - Gamma^i_{lm} Gamma^m_{kj}.
    #[must_use]
    pub fn riemann(&self, p: &VecN) -> TensorN {
        let n = self.dim;
        let gamma = self.christoffel(p);
        // dGamma[k] = d Gamma / d x^k
        let mut dgamma: Vec<TensorN> = Vec::with_capacity(n);
        for k in 0..n {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp.data[k] += self.h;
            pm.data[k] -= self.h;
            let gp = self.christoffel(&pp);
            let gm = self.christoffel(&pm);
            dgamma.push(gp.sub(&gm).scale(1.0 / (2.0 * self.h)));
        }
        TensorN::from_fn(&[n, n, n, n], |idx| {
            let (i, j, k, l) = (idx[0], idx[1], idx[2], idx[3]);
            let mut r = dgamma[k].get(&[i, l, j]) - dgamma[l].get(&[i, k, j]);
            for m in 0..n {
                r += gamma.get(&[i, k, m]) * gamma.get(&[m, l, j])
                    - gamma.get(&[i, l, m]) * gamma.get(&[m, k, j]);
            }
            r
        })
    }

    /// Fully lowered Riemann tensor R_{ijkl}.
    #[must_use]
    pub fn riemann_lowered(&self, p: &VecN) -> TensorN {
        self.riemann(p).lower_index(0, &self.at(p))
    }

    /// Ricci tensor R_{jl} = R^i_{jil}.
    #[must_use]
    pub fn ricci(&self, p: &VecN) -> Matrix {
        self.riemann(p).contract(0, 2).to_matrix().unwrap()
    }

    /// Ricci scalar R = g^{jl} R_{jl}.
    #[must_use]
    pub fn ricci_scalar(&self, p: &VecN) -> f64 {
        let ric = self.ricci(p);
        let ginv = self.inverse_at(p);
        let mut r = 0.0;
        for j in 0..self.dim {
            for l in 0..self.dim {
                r += ginv.get(j, l) * ric.get(j, l);
            }
        }
        r
    }

    /// Einstein tensor G_ij = R_ij - (R/2) g_ij.
    #[must_use]
    pub fn einstein_tensor(&self, p: &VecN) -> Matrix {
        let ric = self.ricci(p);
        let r = self.ricci_scalar(p);
        let g = self.at(p);
        Matrix::from_fn(self.dim, self.dim, |i, j| {
            ric.get(i, j) - 0.5 * r * g.get(i, j)
        })
    }

    /// Weyl conformal tensor C_{ijkl} (dimension at least 3).
    #[must_use]
    pub fn weyl(&self, p: &VecN) -> TensorN {
        let n = self.dim;
        assert!(n >= 3, "Weyl tensor needs dim >= 3");
        let rl = self.riemann_lowered(p);
        let ric = self.ricci(p);
        let r = self.ricci_scalar(p);
        let g = self.at(p);
        let nf = n as f64;
        TensorN::from_fn(&[n, n, n, n], |idx| {
            let (i, j, k, l) = (idx[0], idx[1], idx[2], idx[3]);
            let ricci_part = (g.get(i, k) * ric.get(j, l) - g.get(i, l) * ric.get(j, k)
                + g.get(j, l) * ric.get(i, k)
                - g.get(j, k) * ric.get(i, l))
                / (nf - 2.0);
            let scalar_part = r * (g.get(i, k) * g.get(j, l) - g.get(i, l) * g.get(j, k))
                / ((nf - 1.0) * (nf - 2.0));
            rl.get(&[i, j, k, l]) - ricci_part + scalar_part
        })
    }

    /// Kretschmann scalar K = R_{ijkl} R^{ijkl}.
    #[must_use]
    pub fn kretschmann(&self, p: &VecN) -> f64 {
        let rl = self.riemann_lowered(p);
        let ginv = self.inverse_at(p);
        let mut up = rl.clone();
        for axis in 0..4 {
            up = up.raise_index(axis, &ginv);
        }
        rl.data.iter().zip(&up.data).map(|(a, b)| a * b).sum()
    }

    /// Sectional curvature of the plane spanned by `u`, `v` at `p`.
    #[must_use]
    pub fn sectional_curvature(&self, p: &VecN, u: &VecN, v: &VecN) -> f64 {
        let rl = self.riemann_lowered(p);
        let n = self.dim;
        let mut num = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        num += rl.get(&[i, j, k, l]) * u[i] * v[j] * u[k] * v[l];
                    }
                }
            }
        }
        let guu = self.inner(p, u, u);
        let gvv = self.inner(p, v, v);
        let guv = self.inner(p, u, v);
        num / (guu * gvv - guv * guv)
    }

    /// Gaussian curvature (dimension 2 only): K = R_{0101} / det g.
    #[must_use]
    pub fn gaussian_curvature(&self, p: &VecN) -> Option<f64> {
        if self.dim != 2 {
            return None;
        }
        let rl = self.riemann_lowered(p);
        Some(rl.get(&[0, 1, 0, 1]) / self.det_at(p))
    }

    /// True when the Riemann tensor vanishes within `tol` (per component,
    /// relative to the metric scale).
    #[must_use]
    pub fn is_flat(&self, p: &VecN, tol: f64) -> bool {
        self.riemann(p).data.iter().all(|v| v.abs() <= tol)
    }

    /// True when Ricci = (R/n) g within `tol`.
    #[must_use]
    pub fn is_einstein(&self, p: &VecN, tol: f64) -> bool {
        let ric = self.ricci(p);
        let r = self.ricci_scalar(p);
        let g = self.at(p);
        let n = self.dim as f64;
        (0..self.dim).all(|i| {
            (0..self.dim).all(|j| (ric.get(i, j) - r / n * g.get(i, j)).abs() <= tol)
        })
    }

    /// Covariant derivative of a vector field along `direction`:
    /// (nabla_d v)^i = d^j d_j v^i + Gamma^i_{jk} d^j v^k.
    #[must_use]
    pub fn covariant_derivative_vector(
        &self,
        v: &dyn Fn(&VecN) -> VecN,
        p: &VecN,
        direction: &VecN,
    ) -> VecN {
        let n = self.dim;
        let gamma = self.christoffel(p);
        let vp = v(p);
        let mut out = VecN::zeros(n);
        // directional derivative by finite differences
        for j in 0..n {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp.data[j] += self.h;
            pm.data[j] -= self.h;
            let dv = v(&pp).sub(&v(&pm)).scale(1.0 / (2.0 * self.h));
            for i in 0..n {
                out.data[i] += direction[j] * dv[i];
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    out.data[i] += gamma.get(&[i, j, k]) * direction[j] * vp[k];
                }
            }
        }
        out
    }

    /// Covariant derivative nabla_k of a tensor field with all indices
    /// covariant (lower): (nabla_k T)_{i...} = d_k T_{i...}
    /// - sum_m Gamma^l_{k i_m} T_{... l ...}.
    #[must_use]
    pub fn covariant_derivative_tensor(
        &self,
        t: &dyn Fn(&VecN) -> TensorN,
        p: &VecN,
        k: usize,
    ) -> TensorN {
        let gamma = self.christoffel(p);
        let mut pp = p.clone();
        let mut pm = p.clone();
        pp.data[k] += self.h;
        pm.data[k] -= self.h;
        let mut out = t(&pp).sub(&t(&pm)).scale(1.0 / (2.0 * self.h));
        let tp = t(p);
        let rank = tp.rank();
        let n = self.dim;
        for axis in 0..rank {
            // subtract Gamma^l_{k i_axis} T_{... l ...}
            let mut corr = TensorN::zeros(&tp.shape);
            let shape = tp.shape.clone();
            let mut idx_l = vec![0usize; rank];
            for flat in 0..corr.data.len() {
                // decode flat index
                let mut rem = flat;
                let mut idx = vec![0usize; rank];
                for ax in (0..rank).rev() {
                    idx[ax] = rem % shape[ax];
                    rem /= shape[ax];
                }
                let mut sum = 0.0;
                for l in 0..n {
                    idx_l.clone_from_slice(&idx);
                    idx_l[axis] = l;
                    sum += gamma.get(&[l, k, idx[axis]]) * tp.get(&idx_l);
                }
                corr.data[flat] = sum;
            }
            out = out.sub(&corr);
        }
        out
    }

    /// Covariant divergence of a vector field:
    /// nabla_i v^i = (1/sqrt|g|) d_i (sqrt|g| v^i).
    #[must_use]
    pub fn divergence(&self, v: &dyn Fn(&VecN) -> VecN, p: &VecN) -> f64 {
        let n = self.dim;
        let mut sum = 0.0;
        for i in 0..n {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp.data[i] += self.h;
            pm.data[i] -= self.h;
            let fp = self.volume_element(&pp) * v(&pp)[i];
            let fm = self.volume_element(&pm) * v(&pm)[i];
            sum += (fp - fm) / (2.0 * self.h);
        }
        sum / self.volume_element(p)
    }

    /// Laplace-Beltrami operator on a scalar field:
    /// (1/sqrt|g|) d_i (sqrt|g| g^{ij} d_j f).
    #[must_use]
    pub fn laplace_beltrami(&self, f: &dyn Fn(&VecN) -> f64, p: &VecN) -> f64 {
        let n = self.dim;
        let flux = |q: &VecN, i: usize| -> f64 {
            let ginv = self.inverse_at(q);
            let vol = self.volume_element(q);
            let mut s = 0.0;
            for j in 0..n {
                let mut qp = q.clone();
                let mut qm = q.clone();
                qp.data[j] += self.h;
                qm.data[j] -= self.h;
                s += ginv.get(i, j) * (f(&qp) - f(&qm)) / (2.0 * self.h);
            }
            vol * s
        };
        let mut sum = 0.0;
        for i in 0..n {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp.data[i] += self.h;
            pm.data[i] -= self.h;
            sum += (flux(&pp, i) - flux(&pm, i)) / (2.0 * self.h);
        }
        sum / self.volume_element(p)
    }

    /// Riemannian gradient (index raised): (grad f)^i = g^{ij} d_j f.
    #[must_use]
    pub fn gradient(&self, f: &dyn Fn(&VecN) -> f64, p: &VecN) -> VecN {
        let n = self.dim;
        let ginv = self.inverse_at(p);
        let mut df = VecN::zeros(n);
        for j in 0..n {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp.data[j] += self.h;
            pm.data[j] -= self.h;
            df.data[j] = (f(&pp) - f(&pm)) / (2.0 * self.h);
        }
        let mut out = VecN::zeros(n);
        for i in 0..n {
            for j in 0..n {
                out.data[i] += ginv.get(i, j) * df[j];
            }
        }
        out
    }

    /// Volume element sqrt |det g|.
    #[must_use]
    pub fn volume_element(&self, p: &VecN) -> f64 {
        self.det_at(p).abs().sqrt()
    }

    /// Midpoint-rule integral of `f` against the volume element over a
    /// coordinate box.
    #[must_use]
    pub fn volume_integrate(
        &self,
        f: &dyn Fn(&VecN) -> f64,
        bounds: &[(f64, f64)],
        n_per_dim: usize,
    ) -> f64 {
        assert_eq!(bounds.len(), self.dim);
        let n = self.dim;
        let widths: Vec<f64> = bounds.iter().map(|&(a, b)| (b - a) / n_per_dim as f64).collect();
        let cell: f64 = widths.iter().product();
        let mut idx = vec![0usize; n];
        let mut total = 0.0;
        loop {
            let p = VecN::from(
                &idx.iter()
                    .enumerate()
                    .map(|(d, &i)| bounds[d].0 + (i as f64 + 0.5) * widths[d])
                    .collect::<Vec<f64>>(),
            );
            total += f(&p) * self.volume_element(&p) * cell;
            let mut d = n;
            loop {
                if d == 0 {
                    return total;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < n_per_dim {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    /// Length of a curve c(t) from t0 to t1 using n midpoint samples of
    /// sqrt |g(c', c')|.
    #[must_use]
    pub fn length_of_curve(&self, c: &dyn Fn(f64) -> VecN, t0: f64, t1: f64, n: usize) -> f64 {
        let dt = (t1 - t0) / n as f64;
        let mut len = 0.0;
        for i in 0..n {
            let t = t0 + (i as f64 + 0.5) * dt;
            let dc = c(t + 0.5 * self.h).sub(&c(t - 0.5 * self.h)).scale(1.0 / self.h);
            len += self.inner(&c(t), &dc, &dc).abs().sqrt() * dt;
        }
        len
    }

    /// Metric inner product g(u, v) at p.
    #[must_use]
    pub fn inner(&self, p: &VecN, u: &VecN, v: &VecN) -> f64 {
        let g = self.at(p);
        let mut s = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                s += g.get(i, j) * u[i] * v[j];
            }
        }
        s
    }

    /// Metric norm sqrt |g(v, v)|.
    #[must_use]
    pub fn norm(&self, p: &VecN, v: &VecN) -> f64 {
        self.inner(p, v, v).abs().sqrt()
    }

    /// Angle between u and v in the metric.
    #[must_use]
    pub fn angle(&self, p: &VecN, u: &VecN, v: &VecN) -> f64 {
        (self.inner(p, u, v) / (self.norm(p, u) * self.norm(p, v)))
            .clamp(-1.0, 1.0)
            .acos()
    }

    /// Orthonormal frame (vielbein) at `p`: rows are frame vectors obtained
    /// by Gram-Schmidt of the coordinate basis in the metric inner product,
    /// normalized by sqrt |g(e, e)| (works for Lorentzian signatures).
    #[must_use]
    pub fn orthonormal_frame(&self, p: &VecN) -> Matrix {
        let n = self.dim;
        let mut frame: Vec<VecN> = Vec::with_capacity(n);
        for i in 0..n {
            let mut e = VecN::unit(n, i);
            for b in &frame {
                let gbb = self.inner(p, b, b);
                if gbb.abs() > 1e-30 {
                    let coef = self.inner(p, &e, b) / gbb;
                    e = e.sub(&b.scale(coef));
                }
            }
            let nn = self.norm(p, &e);
            if nn > 1e-30 {
                e = e.scale(1.0 / nn);
            }
            frame.push(e);
        }
        Matrix::from_fn(n, n, |i, j| frame[i][j])
    }

    /// Lie derivative of the metric along `xi`:
    /// (L_xi g)_{ij} = xi^k d_k g_ij + g_kj d_i xi^k + g_ik d_j xi^k.
    #[must_use]
    pub fn lie_derivative_metric(&self, xi: &dyn Fn(&VecN) -> VecN, p: &VecN) -> Matrix {
        let n = self.dim;
        let g = self.at(p);
        let xp = xi(p);
        // d xi^k / d x^i
        let mut dxi = vec![vec![0.0; n]; n]; // dxi[i][k]
        for (i, row) in dxi.iter_mut().enumerate() {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp.data[i] += self.h;
            pm.data[i] -= self.h;
            let d = xi(&pp).sub(&xi(&pm)).scale(1.0 / (2.0 * self.h));
            row.clone_from_slice(&d.data);
        }
        let dgs: Vec<Matrix> = (0..n).map(|k| self.dg(p, k)).collect();
        Matrix::from_fn(n, n, |i, j| {
            let mut s = 0.0;
            for k in 0..n {
                s += xp[k] * dgs[k].get(i, j)
                    + g.get(k, j) * dxi[i][k]
                    + g.get(i, k) * dxi[j][k];
            }
            s
        })
    }

    /// True when `xi` is a Killing field at `p` within `tol`.
    #[must_use]
    pub fn killing_check(&self, xi: &dyn Fn(&VecN) -> VecN, p: &VecN, tol: f64) -> bool {
        self.lie_derivative_metric(xi, p)
            .data
            .iter()
            .all(|v| v.abs() <= tol)
    }

    /// If g = lambda * g_other at `p` (componentwise, consistent), return
    /// lambda.
    #[must_use]
    pub fn conformal_factor_to(&self, other: &Metric, p: &VecN) -> Option<f64> {
        let a = self.at(p);
        let b = other.at(p);
        let mut lambda = None;
        for i in 0..self.dim {
            for j in 0..self.dim {
                let (x, y) = (a.get(i, j), b.get(i, j));
                if y.abs() < 1e-14 {
                    if x.abs() > 1e-10 {
                        return None;
                    }
                    continue;
                }
                let r = x / y;
                match lambda {
                    None => lambda = Some(r),
                    Some(l) if (l - r).abs() > 1e-6 * l.abs().max(1.0) => return None,
                    _ => {}
                }
            }
        }
        lambda
    }

    /// Max residual of the first Bianchi identity
    /// R_{i[jkl]} : R_ijkl + R_iklj + R_iljk = 0, normalized by the largest
    /// Riemann component.
    #[must_use]
    pub fn bianchi_identity_residual(&self, p: &VecN) -> f64 {
        let rl = self.riemann_lowered(p);
        let n = self.dim;
        let scale = rl.data.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-30);
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let r = rl.get(&[i, j, k, l])
                            + rl.get(&[i, k, l, j])
                            + rl.get(&[i, l, j, k]);
                        worst = worst.max(r.abs());
                    }
                }
            }
        }
        worst / scale
    }
}

fn spherical_static_metric(f: f64, r: f64, theta: f64) -> Matrix {
    Matrix::from_fn(4, 4, |i, j| {
        if i != j {
            return 0.0;
        }
        match i {
            0 => -f,
            1 => 1.0 / f,
            2 => r * r,
            _ => r * r * theta.sin().powi(2),
        }
    })
}

/// Schwarzschild metric function in (t, r, theta, phi), G = c = 1.
pub fn schwarzschild_metric_fn(m: f64) -> impl Fn(&VecN) -> Matrix {
    move |p: &VecN| {
        let r = p[1];
        let f = 1.0 - 2.0 * m / r;
        spherical_static_metric(f, r, p[2])
    }
}

/// Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi).
#[must_use]
pub fn kerr_boyer_lindquist(m: f64, a: f64) -> Metric {
    Metric::new(4, move |p: &VecN| {
        let (r, th) = (p[1], p[2]);
        let (s, c) = th.sin_cos();
        let sigma = r * r + a * a * c * c;
        let delta = r * r - 2.0 * m * r + a * a;
        let mut g = Matrix::zeros(4, 4);
        g.set(0, 0, -(1.0 - 2.0 * m * r / sigma));
        g.set(1, 1, sigma / delta);
        g.set(2, 2, sigma);
        g.set(
            3,
            3,
            (r * r + a * a + 2.0 * m * r * a * a * s * s / sigma) * s * s,
        );
        let gtp = -2.0 * m * r * a * s * s / sigma;
        g.set(0, 3, gtp);
        g.set(3, 0, gtp);
        g
    })
}

/// FRW metric in (t, r, theta, phi) with scale factor a(t) and curvature k.
#[must_use]
pub fn frw_metric(a: fn(f64) -> f64, k: f64) -> Metric {
    Metric::new(4, move |p: &VecN| {
        let (t, r, th) = (p[0], p[1], p[2]);
        let a2 = a(t) * a(t);
        Matrix::from_fn(4, 4, |i, j| {
            if i != j {
                return 0.0;
            }
            match i {
                0 => -1.0,
                1 => a2 / (1.0 - k * r * r),
                2 => a2 * r * r,
                _ => a2 * r * r * th.sin().powi(2),
            }
        })
    })
}

/// Induced first fundamental form of a parametrized surface in R^3.
#[must_use]
pub fn surface_metric_from_parametrization(f: fn(f64, f64) -> Vec3) -> Metric {
    let h = 1e-5;
    Metric::new(2, move |p: &VecN| {
        let (u, v) = (p[0], p[1]);
        let fu = (f(u + h, v) - f(u - h, v)) * (1.0 / (2.0 * h));
        let fv = (f(u, v + h) - f(u, v - h)) * (1.0 / (2.0 * h));
        Matrix::from_fn(2, 2, |i, j| {
            let a = if i == 0 { fu } else { fv };
            let b = if j == 0 { fu } else { fv };
            a.dot(&b)
        })
    })
}

/// Warped product metric: block diagonal with the fiber block scaled by
/// warp(base_point)^2. Base coordinates come first.
#[must_use]
pub fn warped_product(
    base: Metric,
    fiber: Metric,
    warp: impl Fn(&VecN) -> f64 + 'static,
) -> Metric {
    let nb = base.dim;
    let nf = fiber.dim;
    Metric::new(nb + nf, move |p: &VecN| {
        let pb = VecN::from(&p.data[..nb]);
        let pf = VecN::from(&p.data[nb..]);
        let gb = (base.g)(&pb);
        let gf = (fiber.g)(&pf);
        let w2 = warp(&pb).powi(2);
        Matrix::from_fn(nb + nf, nb + nf, |i, j| {
            if i < nb && j < nb {
                gb.get(i, j)
            } else if i >= nb && j >= nb {
                w2 * gf.get(i - nb, j - nb)
            } else {
                0.0
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_spaces() {
        let e = Metric::euclidean(3);
        let p = VecN::from(&[0.4, -1.2, 2.0]);
        // Christoffels vanish
        assert!(e.christoffel(&p).norm_frobenius() < 1e-12);
        assert!(e.is_flat(&p, 1e-8));
        assert!((e.ricci_scalar(&p)).abs() < 1e-8);
        // Minkowski flat too, signature (3, 1) for mostly-plus
        let mk = Metric::minkowski(4, Sig::MostlyPlus);
        let q = VecN::from(&[0.0, 1.0, 2.0, 3.0]);
        assert!(mk.is_flat(&q, 1e-8));
        assert_eq!(mk.signature(&q), (3, 1));
        assert_eq!(Metric::minkowski(4, Sig::MostlyMinus).signature(&q), (1, 3));
        // flat torus
        let t = Metric::torus_flat(2, &[1.0, 2.0]);
        assert!(t.is_flat(&VecN::from(&[0.3, 0.9]), 1e-8));
        assert!((t.volume_element(&VecN::zeros(2)) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sphere_curvature() {
        let r = 2.0;
        let s2 = Metric::sphere(2, r);
        let p = VecN::from(&[1.0, 0.5]); // theta, phi away from poles
        // Gaussian curvature 1/r^2, Ricci scalar 2/r^2
        let k = s2.gaussian_curvature(&p).unwrap();
        assert!((k - 1.0 / (r * r)).abs() < 1e-6, "K = {k}");
        assert!((s2.ricci_scalar(&p) - 2.0 / (r * r)).abs() < 1e-5);
        // general n: R = n(n-1)/r^2
        let s3 = Metric::sphere(3, r);
        let p3 = VecN::from(&[1.1, 0.8, 0.4]);
        assert!(
            (s3.ricci_scalar(&p3) - 6.0 / (r * r)).abs() < 1e-4,
            "R3 = {}",
            s3.ricci_scalar(&p3)
        );
        // sectional curvature of any plane is 1/r^2
        let u = VecN::unit(3, 0);
        let v = VecN::unit(3, 2);
        assert!((s3.sectional_curvature(&p3, &u, &v) - 1.0 / (r * r)).abs() < 1e-4);
        // spheres are Einstein manifolds
        assert!(s3.is_einstein(&p3, 1e-4));
        // Bianchi identity residual small
        assert!(s2.bianchi_identity_residual(&p) < 1e-5);
        // area of the 2-sphere: integrate 1 over the chart
        let area = s2.volume_integrate(&|_| 1.0, &[(0.01, std::f64::consts::PI - 0.01), (0.0, 2.0 * std::f64::consts::PI)], 60);
        assert!((area - 4.0 * std::f64::consts::PI * r * r).abs() < 0.05 * area);
    }

    #[test]
    fn test_hyperbolic() {
        // Poincare ball: sectional curvature -1 everywhere
        let h2 = Metric::hyperbolic_ball(2);
        for &pt in &[[0.0, 0.0], [0.3, 0.1], [-0.2, 0.5]] {
            let p = VecN::from(&pt);
            let k = h2.gaussian_curvature(&p).unwrap();
            assert!((k + 1.0).abs() < 1e-5, "K = {k} at {pt:?}");
        }
        let h3 = Metric::hyperbolic_ball(3);
        let p = VecN::from(&[0.1, 0.2, -0.3]);
        let u = VecN::unit(3, 0);
        let v = VecN::unit(3, 1);
        assert!((h3.sectional_curvature(&p, &u, &v) + 1.0).abs() < 1e-4);
        // half-space model also curvature -1
        let hs = Metric::poincare_half_space(2);
        let q = VecN::from(&[0.7, 1.3]);
        assert!((hs.gaussian_curvature(&q).unwrap() + 1.0).abs() < 1e-5);
        // hyperbolic ball geodesic through origin: length of diameter chord
        // from -a to a is 2 * 2 atanh(a)
        let a = 0.5;
        let c = |t: f64| VecN::from(&[t, 0.0]);
        let len = h2.length_of_curve(&c, -a, a, 400);
        assert!((len - 4.0 * a.atanh()).abs() < 1e-4);
    }

    #[test]
    fn test_schwarzschild() {
        let m = 1.0;
        let s = Metric::schwarzschild(m);
        let p = VecN::from(&[0.0, 6.0, std::f64::consts::FRAC_PI_2, 0.3]);
        // vacuum: Ricci = 0
        let ric = s.ricci(&p);
        for v in &ric.data {
            assert!(v.abs() < 1e-6, "Ricci component {v}");
        }
        // Kretschmann = 48 M^2 / r^6
        let kr = s.kretschmann(&p);
        let exact = 48.0 * m * m / 6.0_f64.powi(6);
        assert!((kr - exact).abs() < 1e-5 * exact.max(1e-10) + 1e-8, "K = {kr} vs {exact}");
        // Bianchi residual small
        assert!(s.bianchi_identity_residual(&p) < 1e-5);
        // time translation is a Killing vector
        let xi = |_: &VecN| VecN::unit(4, 0);
        assert!(s.killing_check(&xi, &p, 1e-8));
        // radial translation is not
        let xr = |_: &VecN| VecN::unit(4, 1);
        assert!(!s.killing_check(&xr, &p, 1e-8));
        // Kerr with a = 0.6 is also vacuum
        let kerr = Metric::kerr(1.0, 0.6);
        let pk = VecN::from(&[0.0, 5.0, 1.1, 0.2]);
        let rk = kerr.ricci(&pk);
        for v in &rk.data {
            assert!(v.abs() < 1e-4, "Kerr Ricci component {v}");
        }
        // Kerr reduces to Schwarzschild at a = 0
        let k0 = Metric::kerr(1.0, 0.0);
        let g1 = k0.at(&p);
        let g2 = s.at(&p);
        for (a, b) in g1.data.iter().zip(&g2.data) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cosmological() {
        // de Sitter is an Einstein space with R = 12/l^2
        let l = 3.0;
        let ds = Metric::de_sitter(l);
        let p = VecN::from(&[0.0, 0.8, 1.2, 0.4]);
        assert!(ds.is_einstein(&p, 1e-4));
        assert!((ds.ricci_scalar(&p) - 12.0 / (l * l)).abs() < 1e-4);
        // anti-de Sitter: R = -12/l^2
        let ads = Metric::anti_de_sitter(l);
        assert!((ads.ricci_scalar(&p) + 12.0 / (l * l)).abs() < 1e-4);
        // FRW with a(t) = exp(H t), H = 0.2, k = 0: R = 12 H^2
        fn a_exp(t: f64) -> f64 {
            (0.2 * t).exp()
        }
        let frw = Metric::frw(a_exp, 0.0);
        let q = VecN::from(&[1.0, 0.5, 1.0, 0.7]);
        let r_exact = 12.0 * 0.2 * 0.2;
        assert!(
            (frw.ricci_scalar(&q) - r_exact).abs() < 1e-4,
            "FRW R = {}",
            frw.ricci_scalar(&q)
        );
        // Schwarzschild Weyl is nonzero (vacuum but curved), sphere Weyl ~ 0
        let s = Metric::schwarzschild(1.0);
        let ps = VecN::from(&[0.0, 6.0, 1.2, 0.3]);
        assert!(s.weyl(&ps).norm_frobenius() > 1e-3);
        let s3 = Metric::sphere(3, 1.0);
        let p3 = VecN::from(&[1.1, 0.8, 0.4]);
        // dimension-3 Weyl vanishes identically
        assert!(s3.weyl(&p3).norm_frobenius() < 1e-3);
    }

    #[test]
    fn test_kaluza_klein() {
        // constant phi = 1, zero gauge field: 5D = 4D x flat circle
        let g4 = Metric::minkowski(4, Sig::MostlyPlus);
        let kk = Metric::kaluza_klein_5d(g4, |_| VecN::zeros(4), 1.0);
        let p = VecN::from(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let g = kk.at(&p);
        assert!((g.get(4, 4) - 1.0).abs() < 1e-14);
        assert!(g.get(0, 4).abs() < 1e-14);
        assert!((g.get(0, 0) + 1.0).abs() < 1e-14);
        // flat 5D
        assert!(kk.is_flat(&p, 1e-8));
        // warped product: R x_f S^2 with constant warp r is a round sphere
        let base = Metric::euclidean(1);
        let fiber = Metric::sphere(2, 1.0);
        let wp = warped_product(base, fiber, |_| 2.0);
        let q = VecN::from(&[0.0, 1.0, 0.5]);
        // fiber block = 4 * (round unit sphere) = round sphere radius 2
        let g = wp.at(&q);
        assert!((g.get(1, 1) - 4.0).abs() < 1e-12);
        // scalar curvature of the product: only fiber curves, R = 2/r^2 with
        // r = 2 (constant warp adds nothing)
        assert!((wp.ricci_scalar(&q) - 0.5).abs() < 1e-4, "R = {}", wp.ricci_scalar(&q));
    }

    #[test]
    fn test_surface_and_operators() {
        // torus surface embedded in R^3: K = cos v / (r (R + r cos v))
        fn torus(u: f64, v: f64) -> Vec3 {
            let (big_r, r) = (2.0, 0.5);
            Vec3::new(
                (big_r + r * v.cos()) * u.cos(),
                (big_r + r * v.cos()) * u.sin(),
                r * v.sin(),
            )
        }
        let tm = surface_metric_from_parametrization(torus);
        for &v in &[0.5_f64, 2.0, 4.0] {
            let p = VecN::from(&[1.0, v]);
            let k = tm.gaussian_curvature(&p).unwrap();
            let exact = v.cos() / (0.5 * (2.0 + 0.5 * v.cos()));
            assert!((k - exact).abs() < 5e-3, "torus K at v={v}: {k} vs {exact}");
        }
        // Laplace-Beltrami on Euclidean space = ordinary Laplacian
        let e = Metric::euclidean(2);
        let f = |p: &VecN| p[0] * p[0] + 3.0 * p[1] * p[1];
        let lb = e.laplace_beltrami(&f, &VecN::from(&[0.3, -0.4]));
        assert!((lb - 8.0).abs() < 1e-5, "lb = {lb}");
        // gradient raises correctly on a scaled metric
        let scaled = Metric::new(2, |_| {
            Matrix::from_fn(2, 2, |i, j| if i == j { 4.0 } else { 0.0 })
        });
        let gr = scaled.gradient(&f, &VecN::from(&[1.0, 0.0]));
        assert!((gr[0] - 0.5).abs() < 1e-6); // g^xx df/dx = (1/4) * 2
        // divergence of a linear field in Euclidean space
        let vfield = |p: &VecN| VecN::from(&[2.0 * p[0], -p[1]]);
        assert!((e.divergence(&vfield, &VecN::from(&[0.2, 0.7])) - 1.0).abs() < 1e-6);
        // covariant derivative in Euclidean space = directional derivative
        let cd = e.covariant_derivative_vector(
            &|p: &VecN| VecN::from(&[p[1], p[0] * p[0]]),
            &VecN::from(&[1.0, 2.0]),
            &VecN::from(&[1.0, 0.0]),
        );
        assert!(cd[0].abs() < 1e-8 && (cd[1] - 2.0).abs() < 1e-6);
        // sphere: covariant derivative of the phi-basis vector along itself
        // points toward the pole (parallel transport curls); just check the
        // machinery returns finite values on a curved space
        let s2 = Metric::sphere(2, 1.0);
        let cds = s2.covariant_derivative_vector(
            &|_| VecN::unit(2, 1),
            &VecN::from(&[1.0, 0.5]),
            &VecN::unit(2, 1),
        );
        assert!(cds.data.iter().all(|v| v.is_finite()));
        // covariant derivative of the metric itself vanishes
        let s2b = Metric::sphere(2, 1.0);
        let gfield = move |p: &VecN| TensorN::from_matrix(&s2b.at(p));
        let nabla_g = s2.covariant_derivative_tensor(&gfield, &VecN::from(&[1.0, 0.5]), 0);
        assert!(nabla_g.norm_frobenius() < 1e-6, "nabla g = {}", nabla_g.norm_frobenius());
        // orthonormal frame: e_i . e_j = delta (Euclidean check on sphere
        // metric inner product)
        let fr = s2.orthonormal_frame(&VecN::from(&[1.0, 0.5]));
        let p = VecN::from(&[1.0, 0.5]);
        for i in 0..2 {
            for j in 0..2 {
                let ei = VecN::from(fr.row(i));
                let ej = VecN::from(fr.row(j));
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s2.inner(&p, &ei, &ej) - want).abs() < 1e-10);
            }
        }
        // conformal factor: hyperbolic ball at origin is 4 * Euclidean
        let h2 = Metric::hyperbolic_ball(2);
        let lam = h2.conformal_factor_to(&Metric::euclidean(2), &VecN::zeros(2));
        assert!((lam.unwrap() - 4.0).abs() < 1e-12);
        // christoffel first kind lowers the second kind
        let p = VecN::from(&[1.0, 0.5]);
        let g1 = s2.christoffel_first_kind(&p);
        let g2 = s2.christoffel(&p).lower_index(0, &s2.at(&p));
        assert!(g1.sub(&g2).norm_frobenius() < 1e-8);
    }

    #[test]
    fn test_einstein_tensor() {
        // Schwarzschild is a vacuum solution: G_ij = 0
        let s = Metric::schwarzschild(1.0);
        let p = VecN::from(&[0.0, 6.0, std::f64::consts::FRAC_PI_2, 0.3]);
        let g = s.einstein_tensor(&p);
        for v in &g.data {
            assert!(v.abs() < 1e-6, "Schwarzschild Einstein component {v}");
        }
        // in two dimensions G vanishes identically (R_ij = (R/2) g_ij)
        let s2 = Metric::sphere(2, 1.5);
        let g2 = s2.einstein_tensor(&VecN::from(&[1.0, 0.5]));
        for v in &g2.data {
            assert!(v.abs() < 1e-6, "2D Einstein component {v}");
        }
        // de Sitter solves G_ij + Lambda g_ij = 0 with Lambda = 3/l^2
        let l = 3.0;
        let ds = Metric::de_sitter(l);
        let pd = VecN::from(&[0.0, 0.8, 1.2, 0.4]);
        let gd = ds.einstein_tensor(&pd);
        let metric = ds.at(&pd);
        let lambda = 3.0 / (l * l);
        for i in 0..4 {
            for j in 0..4 {
                let r = gd.get(i, j) + lambda * metric.get(i, j);
                assert!(r.abs() < 1e-5, "de Sitter G + Lambda g at ({i},{j}) = {r}");
            }
        }
        // trace identity in n dimensions: g^ij G_ij = (1 - n/2) R
        for (m, pt) in [
            (Metric::sphere(3, 1.3), VecN::from(&[1.1, 0.8, 0.4])),
            (Metric::hyperbolic_ball(3), VecN::from(&[0.1, 0.2, -0.3])),
        ] {
            let ein = m.einstein_tensor(&pt);
            let ginv = m.inverse_at(&pt);
            let mut tr = 0.0;
            for i in 0..m.dim {
                for j in 0..m.dim {
                    tr += ginv.get(i, j) * ein.get(i, j);
                }
            }
            let want = (1.0 - m.dim as f64 / 2.0) * m.ricci_scalar(&pt);
            assert!((tr - want).abs() < 1e-4, "Einstein trace {tr} vs {want}");
        }
        // the Einstein tensor is symmetric
        let ein = Metric::kerr(1.0, 0.6).einstein_tensor(&VecN::from(&[0.0, 5.0, 1.1, 0.2]));
        for i in 0..4 {
            for j in 0..4 {
                assert!((ein.get(i, j) - ein.get(j, i)).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_warped_product_closed_forms() {
        // dr^2 + r^2 dOmega^2 is flat Euclidean 3-space in spherical
        // coordinates
        let flat = warped_product(Metric::euclidean(1), Metric::sphere(2, 1.0), |p: &VecN| p[0]);
        let q = VecN::from(&[1.3, 0.9, 0.4]);
        let g = flat.at(&q);
        assert!((g.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((g.get(1, 1) - 1.3 * 1.3).abs() < 1e-14);
        assert!((g.get(2, 2) - 1.3 * 1.3 * 0.9_f64.sin().powi(2)).abs() < 1e-14);
        // off-diagonal blocks vanish for a warped product
        assert!(g.get(0, 1).abs() + g.get(0, 2).abs() + g.get(1, 2).abs() == 0.0);
        assert!(flat.is_flat(&q, 1e-6), "spherical coordinates must be flat");
        // volume element r^2 sin(theta)
        let vol = flat.volume_element(&q);
        assert!((vol - 1.3 * 1.3 * 0.9_f64.sin()).abs() < 1e-12, "vol = {vol}");

        // dr^2 + sin^2(r) dOmega^2 is the round unit 3-sphere:
        // R = n(n-1) = 6 and the space is Einstein
        let s3 = warped_product(
            Metric::euclidean(1),
            Metric::sphere(2, 1.0),
            |p: &VecN| p[0].sin(),
        );
        let q3 = VecN::from(&[0.9, 1.1, 0.4]);
        assert!((s3.at(&q3).get(1, 1) - 0.9_f64.sin().powi(2)).abs() < 1e-14);
        let r = s3.ricci_scalar(&q3);
        assert!((r - 6.0).abs() < 1e-4, "unit S^3 scalar curvature {r}");
        assert!(s3.is_einstein(&q3, 1e-4));
        // every 2-plane has sectional curvature 1
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let k = s3.sectional_curvature(&q3, &VecN::unit(3, i), &VecN::unit(3, j));
            assert!((k - 1.0).abs() < 1e-3, "sectional curvature ({i},{j}) = {k}");
        }
        // a constant warp w just rescales the fiber: it is the round sphere
        // of radius w, with fiber curvature 1/w^2
        let cone = warped_product(Metric::euclidean(1), Metric::sphere(2, 1.0), |_| 2.5);
        let qc = VecN::from(&[0.4, 1.0, 0.2]);
        assert!((cone.at(&qc).get(1, 1) - 6.25).abs() < 1e-13);
        assert!((cone.ricci_scalar(&qc) - 2.0 / 6.25).abs() < 1e-4);
    }

    #[test]
    fn test_surface_and_embedded_metrics() {
        // --- gaussian_curvature_surface -----------------------------------
        // sphere of radius R: K = 1/R^2 everywhere
        fn sphere_1_5(u: f64, v: f64) -> Vec3 {
            let r = 1.5;
            Vec3::new(r * u.sin() * v.cos(), r * u.sin() * v.sin(), r * u.cos())
        }
        let sm = Metric::gaussian_curvature_surface(sphere_1_5);
        for &u in &[0.7_f64, 1.4, 2.2] {
            let p = VecN::from(&[u, 0.3]);
            let k = sm.gaussian_curvature(&p).unwrap();
            assert!((k - 1.0 / 2.25).abs() < 1e-3, "sphere K at u={u}: {k}");
            // the induced first fundamental form is R^2 diag(1, sin^2 u)
            let g = sm.at(&p);
            assert!((g.get(0, 0) - 2.25).abs() < 1e-8);
            assert!((g.get(1, 1) - 2.25 * u.sin().powi(2)).abs() < 1e-8);
            assert!(g.get(0, 1).abs() < 1e-8, "coordinate lines are orthogonal");
        }
        // a cylinder is developable: K = 0 despite being curved in space
        fn cylinder(u: f64, v: f64) -> Vec3 {
            Vec3::new(u.cos(), u.sin(), v)
        }
        let cm = Metric::gaussian_curvature_surface(cylinder);
        let pc = VecN::from(&[0.7, 0.2]);
        assert!(cm.gaussian_curvature(&pc).unwrap().abs() < 1e-6);
        // and its induced metric is the flat one: du^2 + dv^2
        let gc = cm.at(&pc);
        assert!((gc.get(0, 0) - 1.0).abs() < 1e-8 && (gc.get(1, 1) - 1.0).abs() < 1e-8);
        assert!(gc.get(0, 1).abs() < 1e-8);

        // --- induced_from_embedding ----------------------------------------
        // the unit-sphere embedding reproduces the round metric
        let emb = Metric::induced_from_embedding(2, |p: &VecN| {
            VecN::from(&[
                p[0].sin() * p[1].cos(),
                p[0].sin() * p[1].sin(),
                p[0].cos(),
            ])
        });
        let round = Metric::sphere(2, 1.0);
        for &pt in &[[1.0_f64, 0.5], [2.0, -1.2], [0.6, 3.0]] {
            let p = VecN::from(&pt);
            let a = emb.at(&p);
            let b = round.at(&p);
            for (x, y) in a.data.iter().zip(&b.data) {
                assert!((x - y).abs() < 1e-8, "induced vs round: {x} vs {y}");
            }
        }
        // and therefore has Gaussian curvature 1
        let k = emb.gaussian_curvature(&VecN::from(&[1.0, 0.5])).unwrap();
        assert!((k - 1.0).abs() < 1e-3, "induced sphere K = {k}");
        // a linear embedding gives the constant Gram matrix J^T J and is flat
        let lin = Metric::induced_from_embedding(2, |p: &VecN| {
            VecN::from(&[p[0], p[1], p[0] + 2.0 * p[1]])
        });
        let pl = VecN::from(&[0.3, -0.7]);
        let gl = lin.at(&pl);
        assert!((gl.get(0, 0) - 2.0).abs() < 1e-8);
        assert!((gl.get(0, 1) - 2.0).abs() < 1e-8);
        assert!((gl.get(1, 0) - 2.0).abs() < 1e-8);
        assert!((gl.get(1, 1) - 5.0).abs() < 1e-8);
        assert!((lin.det_at(&pl) - 6.0).abs() < 1e-7);
        assert!(lin.is_flat(&pl, 1e-6));
        // lengths measured with the induced metric equal lengths of the image
        // curve in the ambient space: the u-curve of the sphere is a meridian
        let curve = |t: f64| VecN::from(&[t, 0.4]);
        let len = emb.length_of_curve(&curve, 0.5, 1.5, 400);
        assert!((len - 1.0).abs() < 1e-6, "meridian arc length {len}");
    }

    #[test]
    fn test_metric_angle() {
        // Euclidean: the metric angle is the ordinary Euclidean angle
        let e = Metric::euclidean(3);
        let p = VecN::zeros(3);
        let u = VecN::from(&[1.0, 2.0, -0.5]);
        let v = VecN::from(&[-0.3, 1.0, 0.8]);
        assert!((e.angle(&p, &u, &v) - u.angle_between(&v)).abs() < 1e-12);
        // orthogonal coordinate directions of a diagonal metric are at pi/2
        let diag = Metric::new(3, |_| {
            Matrix::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 })
        });
        let half_pi = std::f64::consts::FRAC_PI_2;
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let a = diag.angle(&p, &VecN::unit(3, i), &VecN::unit(3, j));
            assert!((a - half_pi).abs() < 1e-12, "angle({i},{j}) = {a}");
        }
        // a vector makes a zero angle with itself and pi with its negative
        assert!(diag.angle(&p, &u, &u) < 1e-7);
        assert!((diag.angle(&p, &u, &u.scale(-1.0)) - std::f64::consts::PI).abs() < 1e-7);
        // the angle is scale invariant and symmetric
        assert!(
            (diag.angle(&p, &u.scale(2.0), &v.scale(3.0)) - diag.angle(&p, &u, &v)).abs() < 1e-12
        );
        assert!((diag.angle(&p, &u, &v) - diag.angle(&p, &v, &u)).abs() < 1e-15);
        // on the unit sphere g = diag(1, sin^2 theta), so the angle between
        // e_theta and e_theta + e_phi is atan(sin theta)
        let s2 = Metric::sphere(2, 1.0);
        for &th in &[0.4_f64, 1.0, 2.5] {
            let q = VecN::from(&[th, 0.9]);
            let a = s2.angle(&q, &VecN::from(&[1.0, 1.0]), &VecN::unit(2, 0));
            assert!((a - th.sin().atan()).abs() < 1e-12, "theta = {th}: {a}");
            // meridians and parallels always meet at a right angle
            let ortho = s2.angle(&q, &VecN::unit(2, 0), &VecN::unit(2, 1));
            assert!((ortho - half_pi).abs() < 1e-12);
        }
    }

    #[test]
    fn test_kaluza_klein_5d() {
        let phi = 1.3_f64;
        let a_const = [0.0, 0.4, -0.2, 0.0];
        let kk = Metric::kaluza_klein_5d(
            Metric::minkowski(4, Sig::MostlyPlus),
            move |_| VecN::from(&a_const),
            phi,
        );
        let p = VecN::from(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let g = kk.at(&p);
        // hand-computed components: g_55 = phi^2, g_{mu 5} = phi^2 A_mu,
        // g_{mu nu} = eta_{mu nu} + phi^2 A_mu A_nu
        assert!((g.get(4, 4) - phi * phi).abs() < 1e-14);
        for mu in 0..4 {
            let want = phi * phi * a_const[mu];
            assert!((g.get(4, mu) - want).abs() < 1e-14, "g_{{5,{mu}}}");
            assert!((g.get(mu, 4) - want).abs() < 1e-14, "symmetry of g_{{mu,5}}");
        }
        assert!((g.get(0, 0) + 1.0).abs() < 1e-14); // A_0 = 0, eta_00 = -1
        assert!((g.get(1, 1) - (1.0 + phi * phi * 0.16)).abs() < 1e-14);
        assert!((g.get(1, 2) - phi * phi * 0.4 * -0.2).abs() < 1e-14);
        assert!((g.get(3, 3) - 1.0).abs() < 1e-14);
        // a constant gauge potential is pure gauge (F = 0): the 5D metric is
        // constant, hence flat
        assert!(kk.is_flat(&p, 1e-8));

        // a potential linear in x^1 carries a constant field strength
        // F_12 = b, and the Kaluza-Klein reduction gives
        // R_5 = R_4 - (1/4) phi^2 F_{mu nu} F^{mu nu} = -phi^2 b^2 / 2
        for &b in &[0.4_f64, 0.8] {
            let kk2 = Metric::kaluza_klein_5d(
                Metric::minkowski(4, Sig::MostlyPlus),
                move |q: &VecN| VecN::from(&[0.0, 0.0, b * q[1], 0.0]),
                1.0,
            );
            let r5 = kk2.ricci_scalar(&p);
            assert!(
                (r5 - (-0.5 * b * b)).abs() < 1e-6,
                "KK scalar curvature for b={b}: {r5} vs {}",
                -0.5 * b * b
            );
            // a nonzero field strength genuinely curves the 5D space
            assert!(!kk2.is_flat(&p, 1e-3));
        }
    }
}
