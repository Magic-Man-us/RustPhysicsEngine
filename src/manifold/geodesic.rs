//! Geodesics, parallel transport, Jacobi fields, and relativistic orbits,
//! all driven by the finite-difference [`Metric`] machinery.

use crate::error::SolveError;
use crate::geometry::mesh::Mesh;
use crate::linalg::{lu_decompose, Matrix};
use crate::manifold::metric::Metric;
use crate::manifold::vecn::{TensorN, VecN};
use crate::math::Vec3;

/// A point on a geodesic: position, velocity, affine parameter.
#[derive(Debug, Clone)]
pub struct GeodesicState {
    pub x: VecN,
    pub v: VecN,
    pub tau: f64,
}

/// Time integrator selection for geodesic integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    Rk4,
    DormandPrince,
}

fn gamma_vv(gamma: &TensorN, v: &VecN, w: &VecN) -> VecN {
    let n = v.dim();
    let mut out = VecN::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            for k in 0..n {
                s += gamma.get(&[i, j, k]) * v[j] * w[k];
            }
        }
        out.data[i] = s;
    }
    out
}

impl Metric {
    /// Right-hand side of the geodesic equation: x' = v,
    /// v'^i = -Gamma^i_{jk} v^j v^k.
    #[must_use]
    pub fn geodesic_rhs(&self, s: &GeodesicState) -> (VecN, VecN) {
        let gamma = self.christoffel(&s.x);
        (s.v.clone(), gamma_vv(&gamma, &s.v, &s.v).scale(-1.0))
    }

    fn rk4_step(&self, x: &VecN, v: &VecN, dt: f64) -> (VecN, VecN) {
        let f = |x: &VecN, v: &VecN| -> (VecN, VecN) {
            let gamma = self.christoffel(x);
            (v.clone(), gamma_vv(&gamma, v, v).scale(-1.0))
        };
        let (k1x, k1v) = f(x, v);
        let (k2x, k2v) = f(
            &x.add(&k1x.scale(0.5 * dt)),
            &v.add(&k1v.scale(0.5 * dt)),
        );
        let (k3x, k3v) = f(
            &x.add(&k2x.scale(0.5 * dt)),
            &v.add(&k2v.scale(0.5 * dt)),
        );
        let (k4x, k4v) = f(&x.add(&k3x.scale(dt)), &v.add(&k3v.scale(dt)));
        let xn = x.add(
            &k1x.add(&k2x.scale(2.0))
                .add(&k3x.scale(2.0))
                .add(&k4x)
                .scale(dt / 6.0),
        );
        let vn = v.add(
            &k1v.add(&k2v.scale(2.0))
                .add(&k3v.scale(2.0))
                .add(&k4v)
                .scale(dt / 6.0),
        );
        (xn, vn)
    }

    /// Integrate the geodesic from (x0, v0) to affine parameter `tau_end`.
    #[must_use]
    pub fn geodesic(
        &self,
        x0: &VecN,
        v0: &VecN,
        tau_end: f64,
        dt: f64,
        method: Integrator,
    ) -> Vec<GeodesicState> {
        match method {
            Integrator::Rk4 => {
                let steps = (tau_end / dt).ceil().max(1.0) as usize;
                let dt = tau_end / steps as f64;
                let mut out = Vec::with_capacity(steps + 1);
                let mut x = x0.clone();
                let mut v = v0.clone();
                out.push(GeodesicState {
                    x: x.clone(),
                    v: v.clone(),
                    tau: 0.0,
                });
                for i in 0..steps {
                    let (xn, vn) = self.rk4_step(&x, &v, dt);
                    x = xn;
                    v = vn;
                    out.push(GeodesicState {
                        x: x.clone(),
                        v: v.clone(),
                        tau: (i + 1) as f64 * dt,
                    });
                }
                out
            }
            Integrator::DormandPrince => self.geodesic_adaptive(x0, v0, tau_end, 1e-8),
        }
    }

    /// Adaptive RK45 (Dormand-Prince) geodesic integration with relative
    /// tolerance `rtol`.
    #[must_use]
    pub fn geodesic_adaptive(
        &self,
        x0: &VecN,
        v0: &VecN,
        tau_end: f64,
        rtol: f64,
    ) -> Vec<GeodesicState> {
        // pack (x, v) into one state vector
        let n = x0.dim();
        let f = |y: &Vec<f64>| -> Vec<f64> {
            let x = VecN::from(&y[..n]);
            let v = VecN::from(&y[n..]);
            let gamma = self.christoffel(&x);
            let a = gamma_vv(&gamma, &v, &v).scale(-1.0);
            let mut dy = Vec::with_capacity(2 * n);
            dy.extend_from_slice(&v.data);
            dy.extend_from_slice(&a.data);
            dy
        };
        // Dormand-Prince coefficients
        const A: [[f64; 6]; 6] = [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 2187.0,
                64448.0 / 6561.0,
                -212.0 / 729.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -5103.0 / 18656.0,
                0.0,
            ],
            [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
            ],
        ];
        const B5: [f64; 7] = [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ];
        const B4: [f64; 7] = [
            5179.0 / 57600.0,
            0.0,
            7571.0 / 16695.0,
            393.0 / 640.0,
            -92097.0 / 339200.0,
            187.0 / 2100.0,
            1.0 / 40.0,
        ];
        let mut y: Vec<f64> = x0.data.iter().chain(&v0.data).copied().collect();
        let mut tau = 0.0;
        let mut h = tau_end / 100.0;
        let mut out = vec![GeodesicState {
            x: x0.clone(),
            v: v0.clone(),
            tau: 0.0,
        }];
        let mut guard = 0;
        while tau < tau_end && guard < 100_000 {
            guard += 1;
            h = h.min(tau_end - tau);
            let mut k: Vec<Vec<f64>> = Vec::with_capacity(7);
            k.push(f(&y));
            for row in &A {
                let mut ys = y.clone();
                for (m, km) in k.iter().enumerate() {
                    let a = row[m];
                    if a != 0.0 {
                        for (s, &kv) in ys.iter_mut().zip(km) {
                            *s += h * a * kv;
                        }
                    }
                }
                k.push(f(&ys));
            }
            let mut y5 = y.clone();
            let mut y4 = y.clone();
            for (m, km) in k.iter().enumerate() {
                for i in 0..y.len() {
                    y5[i] += h * B5[m] * km[i];
                    y4[i] += h * B4[m] * km[i];
                }
            }
            let scale: f64 = y.iter().map(|v| v.abs()).fold(1.0, f64::max);
            let err = y5
                .iter()
                .zip(&y4)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
                / scale;
            if err <= rtol || h < 1e-12 {
                tau += h;
                y = y5;
                out.push(GeodesicState {
                    x: VecN::from(&y[..n]),
                    v: VecN::from(&y[n..]),
                    tau,
                });
            }
            let fac = if err > 0.0 {
                0.9 * (rtol / err).powf(0.2)
            } else {
                2.0
            };
            h *= fac.clamp(0.2, 5.0);
        }
        out
    }

    /// Discrete geodesic between two points by string relaxation of the
    /// discrete geodesic equation.
    pub fn geodesic_between(
        &self,
        a: &VecN,
        b: &VecN,
        n_iter: usize,
    ) -> Result<Vec<VecN>, SolveError> {
        let n_pts = 33;
        let mut path: Vec<VecN> = (0..n_pts)
            .map(|i| a.lerp(b, i as f64 / (n_pts - 1) as f64))
            .collect();
        let dt = 1.0 / (n_pts - 1) as f64;
        for _ in 0..n_iter {
            let snapshot = path.clone();
            for i in 1..n_pts - 1 {
                let v = snapshot[i + 1].sub(&snapshot[i - 1]).scale(1.0 / (2.0 * dt));
                let gamma = self.christoffel(&snapshot[i]);
                let corr = gamma_vv(&gamma, &v, &v).scale(0.5 * dt * dt);
                path[i] = snapshot[i + 1]
                    .add(&snapshot[i - 1])
                    .scale(0.5)
                    .add(&corr);
            }
        }
        if path.iter().all(|p| p.data.iter().all(|v| v.is_finite())) {
            Ok(path)
        } else {
            Err(SolveError::NoConvergence {
                iters: n_iter,
                residual: f64::NAN,
            })
        }
    }

    /// Geodesic distance by relaxing a path and measuring its length.
    pub fn geodesic_distance(&self, a: &VecN, b: &VecN) -> Result<f64, SolveError> {
        let path = self.geodesic_between(a, b, 400)?;
        let mut len = 0.0;
        for w in path.windows(2) {
            let mid = w[0].lerp(&w[1], 0.5);
            let d = w[1].sub(&w[0]);
            len += self.inner(&mid, &d, &d).abs().sqrt();
        }
        Ok(len)
    }

    /// Exponential map: the geodesic from `p` with initial velocity `v`
    /// evaluated at affine parameter 1.
    #[must_use]
    pub fn exp_map(&self, p: &VecN, v: &VecN) -> VecN {
        let path = self.geodesic(p, v, 1.0, 0.01, Integrator::Rk4);
        path.last().unwrap().x.clone()
    }

    /// Logarithm map: find v with exp_p(v) = q by Newton shooting.
    pub fn log_map(&self, p: &VecN, q: &VecN) -> Result<VecN, SolveError> {
        let n = self.dim;
        let mut v = q.sub(p);
        for _ in 0..60 {
            let fx = self.exp_map(p, &v).sub(q);
            let err = fx.norm();
            if err < 1e-10 {
                return Ok(v);
            }
            // finite-difference Jacobian d exp / d v
            let hstep = 1e-6 * v.norm().max(1.0);
            let mut jac = Matrix::zeros(n, n);
            for j in 0..n {
                let mut vp = v.clone();
                vp.data[j] += hstep;
                let d = self.exp_map(p, &vp).sub(q).sub(&fx).scale(1.0 / hstep);
                for i in 0..n {
                    jac.set(i, j, d[i]);
                }
            }
            let delta = lu_decompose(&jac)?.solve(&fx.data)?;
            for (vi, di) in v.data.iter_mut().zip(&delta) {
                *vi -= di;
            }
        }
        let fx = self.exp_map(p, &v).sub(q);
        if fx.norm() < 1e-6 {
            Ok(v)
        } else {
            Err(SolveError::NoConvergence {
                iters: 60,
                residual: fx.norm(),
            })
        }
    }

    /// Discrete parallel transport of `v` along a polyline path:
    /// v <- v - Gamma(dx, v) per segment.
    #[must_use]
    pub fn parallel_transport(&self, v: &VecN, path: &[VecN]) -> VecN {
        let mut vt = v.clone();
        for w in path.windows(2) {
            let dx = w[1].sub(&w[0]);
            let mid = w[0].lerp(&w[1], 0.5);
            let gamma = self.christoffel(&mid);
            vt = vt.sub(&gamma_vv(&gamma, &dx, &vt));
        }
        vt
    }

    /// Parallel transport along a geodesic by integrating
    /// dv^i/dtau = -Gamma^i_{jk} xdot^j v^k with RK4.
    #[must_use]
    pub fn parallel_transport_along_geodesic(
        &self,
        v: &VecN,
        x0: &VecN,
        v0: &VecN,
        tau_end: f64,
        dt: f64,
    ) -> VecN {
        let steps = (tau_end / dt).ceil().max(1.0) as usize;
        let dt = tau_end / steps as f64;
        let mut x = x0.clone();
        let mut u = v0.clone();
        let mut w = v.clone();
        for _ in 0..steps {
            // RK4 on the combined system (x, u, w)
            let f = |x: &VecN, u: &VecN, w: &VecN| -> (VecN, VecN, VecN) {
                let gamma = self.christoffel(x);
                (
                    u.clone(),
                    gamma_vv(&gamma, u, u).scale(-1.0),
                    gamma_vv(&gamma, u, w).scale(-1.0),
                )
            };
            let (k1x, k1u, k1w) = f(&x, &u, &w);
            let (k2x, k2u, k2w) = f(
                &x.add(&k1x.scale(0.5 * dt)),
                &u.add(&k1u.scale(0.5 * dt)),
                &w.add(&k1w.scale(0.5 * dt)),
            );
            let (k3x, k3u, k3w) = f(
                &x.add(&k2x.scale(0.5 * dt)),
                &u.add(&k2u.scale(0.5 * dt)),
                &w.add(&k2w.scale(0.5 * dt)),
            );
            let (k4x, k4u, k4w) = f(
                &x.add(&k3x.scale(dt)),
                &u.add(&k3u.scale(dt)),
                &w.add(&k3w.scale(dt)),
            );
            x = x.add(
                &k1x.add(&k2x.scale(2.0))
                    .add(&k3x.scale(2.0))
                    .add(&k4x)
                    .scale(dt / 6.0),
            );
            u = u.add(
                &k1u.add(&k2u.scale(2.0))
                    .add(&k3u.scale(2.0))
                    .add(&k4u)
                    .scale(dt / 6.0),
            );
            w = w.add(
                &k1w.add(&k2w.scale(2.0))
                    .add(&k3w.scale(2.0))
                    .add(&k4w)
                    .scale(dt / 6.0),
            );
        }
        w
    }

    /// Transport `v` around a closed loop; the difference from `v` measures
    /// the enclosed curvature.
    #[must_use]
    pub fn holonomy(&self, loop_path: &[VecN], v: &VecN) -> VecN {
        self.parallel_transport(v, loop_path)
    }

    /// Rotation angle picked up by parallel transport around a closed loop
    /// on a 2D manifold (equals the integral of Gaussian curvature by
    /// Gauss-Bonnet).
    #[must_use]
    pub fn holonomy_angle_2d(&self, loop_path: &[VecN]) -> f64 {
        assert_eq!(self.dim, 2);
        let p0 = &loop_path[0];
        let frame = self.orthonormal_frame(p0);
        let e1 = VecN::from(frame.row(0));
        let e2 = VecN::from(frame.row(1));
        let vt = self.parallel_transport(&e1, loop_path);
        let c = self.inner(p0, &vt, &e1);
        let s = self.inner(p0, &vt, &e2);
        s.atan2(c)
    }

    /// Tidal acceleration from the Jacobi equation:
    /// D^2 J / dtau^2 = -R(J, v)v, returned as
    /// A^i = -R^i_{jkl} v^j J^k v^l.
    #[must_use]
    pub fn geodesic_deviation(
        &self,
        s: &GeodesicState,
        separation: &VecN,
        _sep_vel: &VecN,
    ) -> VecN {
        let n = self.dim;
        let r = self.riemann(&s.x);
        let mut a = VecN::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        sum += r.get(&[i, j, k, l]) * s.v[j] * separation[k] * s.v[l];
                    }
                }
            }
            a.data[i] = -sum;
        }
        a
    }

    /// Integrate the Jacobi field along a geodesic (coordinate form obtained
    /// by varying the geodesic equation):
    /// J'' + 2 Gamma(v, J') + (d_l Gamma)(v, v) J^l = 0.
    /// Returns J at each step.
    #[must_use]
    pub fn jacobi_field(
        &self,
        x0: &VecN,
        v0: &VecN,
        j0: &VecN,
        j0_dot: &VecN,
        tau_end: f64,
        dt: f64,
    ) -> Vec<VecN> {
        let n = self.dim;
        let steps = (tau_end / dt).ceil().max(1.0) as usize;
        let dt = tau_end / steps as f64;
        let rhs = |x: &VecN, v: &VecN, j: &VecN, jd: &VecN| -> (VecN, VecN, VecN, VecN) {
            let gamma = self.christoffel(x);
            let acc = gamma_vv(&gamma, v, v).scale(-1.0);
            // d Gamma in direction of each coordinate, contracted
            let mut jacc = gamma_vv(&gamma, v, jd).scale(-2.0);
            for l in 0..n {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp.data[l] += self.h;
                xm.data[l] -= self.h;
                let gp = self.christoffel(&xp);
                let gm = self.christoffel(&xm);
                let dg = gp.sub(&gm).scale(1.0 / (2.0 * self.h));
                let term = gamma_vv(&dg, v, v).scale(-j[l]);
                jacc = jacc.add(&term);
            }
            (v.clone(), acc, jd.clone(), jacc)
        };
        let mut x = x0.clone();
        let mut v = v0.clone();
        let mut j = j0.clone();
        let mut jd = j0_dot.clone();
        let mut out = vec![j.clone()];
        for _ in 0..steps {
            let (k1x, k1v, k1j, k1jd) = rhs(&x, &v, &j, &jd);
            let (k2x, k2v, k2j, k2jd) = rhs(
                &x.add(&k1x.scale(0.5 * dt)),
                &v.add(&k1v.scale(0.5 * dt)),
                &j.add(&k1j.scale(0.5 * dt)),
                &jd.add(&k1jd.scale(0.5 * dt)),
            );
            let (k3x, k3v, k3j, k3jd) = rhs(
                &x.add(&k2x.scale(0.5 * dt)),
                &v.add(&k2v.scale(0.5 * dt)),
                &j.add(&k2j.scale(0.5 * dt)),
                &jd.add(&k2jd.scale(0.5 * dt)),
            );
            let (k4x, k4v, k4j, k4jd) = rhs(
                &x.add(&k3x.scale(dt)),
                &v.add(&k3v.scale(dt)),
                &j.add(&k3j.scale(dt)),
                &jd.add(&k3jd.scale(dt)),
            );
            x = x.add(
                &k1x.add(&k2x.scale(2.0)).add(&k3x.scale(2.0)).add(&k4x).scale(dt / 6.0),
            );
            v = v.add(
                &k1v.add(&k2v.scale(2.0)).add(&k3v.scale(2.0)).add(&k4v).scale(dt / 6.0),
            );
            j = j.add(
                &k1j.add(&k2j.scale(2.0)).add(&k3j.scale(2.0)).add(&k4j).scale(dt / 6.0),
            );
            jd = jd.add(
                &k1jd
                    .add(&k2jd.scale(2.0))
                    .add(&k3jd.scale(2.0))
                    .add(&k4jd)
                    .scale(dt / 6.0),
            );
            out.push(j.clone());
        }
        out
    }

    /// Affine parameters of conjugate points: zeros of |J| for a Jacobi
    /// field with J(0) = 0.
    #[must_use]
    pub fn conjugate_points(
        &self,
        x0: &VecN,
        v0: &VecN,
        tau_max: f64,
        dt: f64,
    ) -> Vec<f64> {
        let n = self.dim;
        // seed with a normalized J' orthogonal to v0 in the metric
        let frame = self.orthonormal_frame(x0);
        let mut jdot = VecN::from(frame.row(n - 1));
        // make sure it's not parallel to v0
        if self.inner(x0, &jdot, v0).abs() > 0.9 * self.norm(x0, &jdot) * self.norm(x0, v0) {
            jdot = VecN::from(frame.row(0));
        }
        let js = self.jacobi_field(x0, v0, &VecN::zeros(n), &jdot, tau_max, dt);
        let steps = js.len() - 1;
        let dtau = tau_max / steps as f64;
        let mut out = Vec::new();
        for i in 2..js.len() {
            let a = js[i - 1].norm();
            let b = js[i].norm();
            // |J| dips to ~0 and grows again: detect near-zero minima or
            // sign-like crossings of the projection
            if a < b && i >= 2 {
                let prev = js[i - 2].norm();
                if a < prev && a < 1e-2 * js.iter().map(VecN::norm).fold(0.0, f64::max) {
                    out.push((i - 1) as f64 * dtau);
                }
            }
        }
        out
    }

    /// Rough cut-locus estimate on a 2D manifold: for each direction, the
    /// point where the first conjugate point occurs (clamped to `tau_max`).
    #[must_use]
    pub fn cut_locus_estimate_2d(
        &self,
        p: &VecN,
        n_directions: usize,
        tau_max: f64,
    ) -> Vec<VecN> {
        assert_eq!(self.dim, 2);
        let frame = self.orthonormal_frame(p);
        let e1 = VecN::from(frame.row(0));
        let e2 = VecN::from(frame.row(1));
        (0..n_directions)
            .map(|k| {
                let th = 2.0 * std::f64::consts::PI * k as f64 / n_directions as f64;
                let dir = e1.scale(th.cos()).add(&e2.scale(th.sin()));
                let conj = self.conjugate_points(p, &dir, tau_max, tau_max / 200.0);
                let tau = conj.first().copied().unwrap_or(tau_max);
                let path = self.geodesic(p, &dir, tau, tau / 100.0, Integrator::Rk4);
                path.last().unwrap().x.clone()
            })
            .collect()
    }

    /// Null geodesic for a Lorentzian metric: the time component of `k0` is
    /// rescaled so g(k, k) = 0, then the geodesic is integrated.
    #[must_use]
    pub fn null_geodesic(
        &self,
        x0: &VecN,
        k0: &VecN,
        lambda_end: f64,
        dt: f64,
    ) -> Vec<GeodesicState> {
        // solve g_00 a^2 k0^2 + 2 a k0 g_{0i} k^i + g_ij k^i k^j = 0 for the
        // time-component scaling a
        let g = self.at(x0);
        let n = self.dim;
        let mut spatial = 0.0;
        let mut cross = 0.0;
        for i in 1..n {
            cross += g.get(0, i) * k0[i];
            for j in 1..n {
                spatial += g.get(i, j) * k0[i] * k0[j];
            }
        }
        let (aa, bb, cc) = (g.get(0, 0) * k0[0] * k0[0], 2.0 * k0[0] * cross, spatial);
        let mut k = k0.clone();
        if aa.abs() > 1e-30 {
            let disc = (bb * bb - 4.0 * aa * cc).max(0.0).sqrt();
            let a1 = (-bb + disc) / (2.0 * aa);
            let a2 = (-bb - disc) / (2.0 * aa);
            let a = if (a1 - 1.0).abs() < (a2 - 1.0).abs() { a1 } else { a2 };
            k.data[0] *= a;
        }
        self.geodesic(x0, &k, lambda_end, dt, Integrator::Rk4)
    }

    /// True when the polyline satisfies the discrete geodesic equation
    /// within `tol` (relative to segment length).
    #[must_use]
    pub fn is_geodesic(&self, path: &[VecN], tol: f64) -> bool {
        if path.len() < 3 {
            return true;
        }
        let dt = 1.0;
        for i in 1..path.len() - 1 {
            let v = path[i + 1].sub(&path[i - 1]).scale(1.0 / (2.0 * dt));
            let gamma = self.christoffel(&path[i]);
            let acc = path[i + 1]
                .sub(&path[i].scale(2.0))
                .add(&path[i - 1])
                .scale(1.0 / (dt * dt));
            let resid = acc.add(&gamma_vv(&gamma, &v, &v));
            let scale = v.norm().max(1e-30);
            if resid.norm() > tol * scale.max(scale * scale) {
                return false;
            }
        }
        true
    }

    /// Points at geodesic distance `r` from `p` in `n_dirs` directions.
    #[must_use]
    pub fn geodesic_sphere(&self, p: &VecN, r: f64, n_dirs: usize) -> Vec<VecN> {
        let n = self.dim;
        let frame = self.orthonormal_frame(p);
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        (0..n_dirs)
            .map(|k| {
                let dir = if n == 2 {
                    let th = 2.0 * std::f64::consts::PI * k as f64 / n_dirs as f64;
                    VecN::from(frame.row(0))
                        .scale(th.cos())
                        .add(&VecN::from(frame.row(1)).scale(th.sin()))
                } else {
                    let mut d = VecN::zeros(n);
                    for i in 0..n {
                        d = d.add(&VecN::from(frame.row(i)).scale(rand()));
                    }
                    let nn = self.norm(p, &d).max(1e-12);
                    d.scale(1.0 / nn)
                };
                self.exp_map(p, &dir.scale(r))
            })
            .collect()
    }

    /// Geodesic polar coordinates on a 2D manifold: exp_p(r (cos theta e1 +
    /// sin theta e2)).
    #[must_use]
    pub fn geodesic_polar_coords(&self, p: &VecN, r: f64, theta: f64) -> VecN {
        assert_eq!(self.dim, 2);
        let frame = self.orthonormal_frame(p);
        let dir = VecN::from(frame.row(0))
            .scale(theta.cos())
            .add(&VecN::from(frame.row(1)).scale(theta.sin()));
        self.exp_map(p, &dir.scale(r))
    }

    /// Karcher (Frechet) mean by iterating m <- exp_m(mean of log_m(p_i)).
    #[must_use]
    pub fn karcher_mean(&self, points: &[VecN], iters: usize) -> VecN {
        let mut m = points[0].clone();
        for _ in 0..iters {
            let mut avg = VecN::zeros(self.dim);
            let mut count = 0.0;
            for p in points {
                if let Ok(v) = self.log_map(&m, p) {
                    avg = avg.add(&v);
                    count += 1.0;
                }
            }
            if count == 0.0 {
                break;
            }
            avg = avg.scale(1.0 / count);
            if avg.norm() < 1e-12 {
                break;
            }
            m = self.exp_map(&m, &avg);
        }
        m
    }

    /// Geodesic interpolation: exp_a(t log_a(b)).
    #[must_use]
    pub fn geodesic_interpolate(&self, a: &VecN, b: &VecN, t: f64) -> VecN {
        match self.log_map(a, b) {
            Ok(v) => self.exp_map(a, &v.scale(t)),
            Err(_) => a.lerp(b, t),
        }
    }

    /// Geodesic regression: fit (base point at t = 0, velocity) to samples
    /// (t_i, p_i) by one Karcher-mean step and a linear fit in the tangent
    /// space at the mean.
    #[must_use]
    pub fn geodesic_regression(&self, ts: &[f64], points: &[VecN]) -> (VecN, VecN) {
        let m = self.karcher_mean(points, 20);
        let t_mean = ts.iter().sum::<f64>() / ts.len() as f64;
        let mut num = VecN::zeros(self.dim);
        let mut den = 0.0;
        let logs: Vec<VecN> = points
            .iter()
            .map(|p| self.log_map(&m, p).unwrap_or_else(|_| VecN::zeros(self.dim)))
            .collect();
        for (t, l) in ts.iter().zip(&logs) {
            num = num.add(&l.scale(t - t_mean));
            den += (t - t_mean) * (t - t_mean);
        }
        let v = if den > 0.0 { num.scale(1.0 / den) } else { num };
        let p0 = self.exp_map(&m, &v.scale(-t_mean));
        (p0, v)
    }
}

// ---------------------------------------------------------------------------
// Relativistic orbits and free functions
// ---------------------------------------------------------------------------

/// Verify that sphere geodesics are great circles: shoot a unit-speed
/// geodesic along the equator and check it closes after 2 pi r.
#[must_use]
pub fn great_circle_check(r: f64) -> bool {
    let s = Metric::sphere(2, r);
    let p0 = VecN::from(&[std::f64::consts::FRAC_PI_2, 0.0]);
    let v0 = VecN::from(&[0.0, 1.0 / r]); // unit speed along phi
    let path = s.geodesic(&p0, &v0, 2.0 * std::f64::consts::PI * r, 0.01 * r, Integrator::Rk4);
    let end = &path.last().unwrap().x;
    let dth = (end[0] - p0[0]).abs();
    let dph = (end[1] - p0[1] - 2.0 * std::f64::consts::PI).abs();
    dth < 1e-6 && dph < 1e-6
}

/// Equatorial Schwarzschild orbit r(phi) starting at r0 with dr/dphi = 0,
/// angular momentum `l` per unit mass (the energy parameter `_e` is
/// determined by the turning-point condition and kept for signature
/// compatibility). Integrates u'' + u = M/L^2 + 3 M u^2 with RK4. Returns
/// (phi, r) samples.
#[must_use]
pub fn schwarzschild_orbit(
    m: f64,
    r0: f64,
    l: f64,
    _e: f64,
    phi_end: f64,
    dt: f64,
) -> Vec<(f64, f64)> {
    let mut u = 1.0 / r0;
    let mut up = 0.0;
    let rhs = |u: f64| m / (l * l) + 3.0 * m * u * u - u;
    let steps = (phi_end / dt).ceil() as usize;
    let dphi = phi_end / steps as f64;
    let mut out = Vec::with_capacity(steps + 1);
    out.push((0.0, r0));
    for i in 0..steps {
        let k1u = up;
        let k1p = rhs(u);
        let k2u = up + 0.5 * dphi * k1p;
        let k2p = rhs(u + 0.5 * dphi * k1u);
        let k3u = up + 0.5 * dphi * k2p;
        let k3p = rhs(u + 0.5 * dphi * k2u);
        let k4u = up + dphi * k3p;
        let k4p = rhs(u + dphi * k3u);
        u += dphi / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        up += dphi / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        out.push(((i + 1) as f64 * dphi, 1.0 / u));
    }
    out
}

/// Leading-order perihelion precession per orbit: 6 pi M / (a (1 - e^2)).
#[must_use]
pub fn perihelion_precession(m: f64, a: f64, e: f64) -> f64 {
    6.0 * std::f64::consts::PI * m / (a * (1.0 - e * e))
}

/// Leading-order light deflection by a mass: 4 M / b.
#[must_use]
pub fn light_deflection(m: f64, b: f64) -> f64 {
    4.0 * m / b
}

/// Shapiro time delay for a signal grazing at impact parameter `b` between
/// radii r1 and r2: 2M ln(4 r1 r2 / b^2).
#[must_use]
pub fn shapiro_delay(m: f64, r1: f64, r2: f64, b: f64) -> f64 {
    2.0 * m * (4.0 * r1 * r2 / (b * b)).ln()
}

/// Lyapunov instability exponent of the circular photon orbit at r = 3M:
/// lambda = 1 / (3 sqrt(3) M) per unit affine time.
#[must_use]
pub fn photon_orbit_stability(m: f64) -> f64 {
    1.0 / (3.0 * 3.0_f64.sqrt() * m)
}

/// Shortest path between two mesh vertices along mesh edges (Dijkstra —
/// an upper bound on the exact geodesic). Returns the vertex positions.
#[must_use]
pub fn geodesics_on_mesh_exact(mesh: &Mesh, a: usize, b: usize) -> Vec<Vec3> {
    let nv = mesh.vertices.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nv];
    for t in &mesh.triangles {
        for e in 0..3 {
            let (i, j) = (t[e], t[(e + 1) % 3]);
            let w = (mesh.vertices[i] - mesh.vertices[j]).magnitude();
            adj[i].push((j, w));
            adj[j].push((i, w));
        }
    }
    let mut dist = vec![f64::INFINITY; nv];
    let mut prev = vec![usize::MAX; nv];
    let mut visited = vec![false; nv];
    dist[a] = 0.0;
    for _ in 0..nv {
        let mut best = usize::MAX;
        let mut bd = f64::INFINITY;
        for (i, (&d, &vis)) in dist.iter().zip(&visited).enumerate() {
            if !vis && d < bd {
                bd = d;
                best = i;
            }
        }
        if best == usize::MAX || best == b {
            break;
        }
        visited[best] = true;
        for &(j, w) in &adj[best] {
            if dist[best] + w < dist[j] {
                dist[j] = dist[best] + w;
                prev[j] = best;
            }
        }
    }
    let mut path = Vec::new();
    let mut cur = b;
    while cur != usize::MAX {
        path.push(mesh.vertices[cur]);
        if cur == a {
            break;
        }
        cur = prev[cur];
    }
    path.reverse();
    path
}

/// Heat-method geodesic distance from a source vertex (Crane et al.):
/// diffuse heat for time `t`, normalize the gradient per face, solve a
/// Poisson equation for the distance. Dense solves; suitable for small
/// meshes.
#[must_use]
pub fn heat_method_geodesic(mesh: &Mesh, source: usize, t: f64) -> Vec<f64> {
    let nv = mesh.vertices.len();
    let nf = mesh.triangles.len();
    // cotangent Laplacian L and lumped mass M
    let mut l = Matrix::zeros(nv, nv);
    let mut mass = vec![0.0; nv];
    for tri in &mesh.triangles {
        let [i, j, k] = *tri;
        let (pi, pj, pk) = (mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]);
        let area = 0.5 * (pj - pi).cross(&(pk - pi)).magnitude();
        for v in [i, j, k] {
            mass[v] += area / 3.0;
        }
        let cot = |a: Vec3, b: Vec3| a.dot(&b) / a.cross(&b).magnitude().max(1e-30);
        // opposite-angle cotangents
        let entries = [
            (j, k, cot(pj - pi, pk - pi)),
            (k, i, cot(pk - pj, pi - pj)),
            (i, j, cot(pi - pk, pj - pk)),
        ];
        for (a, b, c) in entries {
            let w = 0.5 * c;
            l.set(a, b, l.get(a, b) - w);
            l.set(b, a, l.get(b, a) - w);
            l.set(a, a, l.get(a, a) + w);
            l.set(b, b, l.get(b, b) + w);
        }
    }
    // heat step: (M + t L) u = delta_source
    let a = Matrix::from_fn(nv, nv, |r, c| {
        t * l.get(r, c) + if r == c { mass[r] } else { 0.0 }
    });
    let mut rhs = vec![0.0; nv];
    rhs[source] = 1.0;
    let u = match lu_decompose(&a).and_then(|lu| lu.solve(&rhs)) {
        Ok(v) => v,
        Err(_) => return vec![0.0; nv],
    };
    // normalized negative gradient per face, then vertex divergence
    let mut div = vec![0.0; nv];
    for f in 0..nf {
        let [i, j, k] = mesh.triangles[f];
        let (pi, pj, pk) = (mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]);
        let n = (pj - pi).cross(&(pk - pi));
        let area2 = n.magnitude().max(1e-30);
        let nn = n * (1.0 / area2);
        // gradient of a linear function on the triangle
        let grad = (nn.cross(&(pk - pj)) * u[i]
            + nn.cross(&(pi - pk)) * u[j]
            + nn.cross(&(pj - pi)) * u[k])
            * (1.0 / area2);
        let g = grad.magnitude();
        if g < 1e-30 {
            continue;
        }
        let x = grad * (-1.0 / g); // unit vector toward increasing distance
        // divergence contribution per vertex
        let cot = |a: Vec3, b: Vec3| a.dot(&b) / a.cross(&b).magnitude().max(1e-30);
        let cot_k = cot(pi - pk, pj - pk); // angle at k, opposite edge ij
        let cot_i = cot(pj - pi, pk - pi); // angle at i, opposite edge jk
        let cot_j = cot(pk - pj, pi - pj); // angle at j, opposite edge ki
        // each outgoing edge pairs with the cotangent of its opposite angle
        div[i] += 0.5 * (cot_k * (pj - pi).dot(&x) + cot_j * (pk - pi).dot(&x));
        div[j] += 0.5 * (cot_i * (pk - pj).dot(&x) + cot_k * (pi - pj).dot(&x));
        div[k] += 0.5 * (cot_j * (pi - pk).dot(&x) + cot_i * (pj - pk).dot(&x));
    }
    // Poisson: our L is the positive-definite cotan operator (~ -nabla^2),
    // so nabla^2 phi = div X becomes L phi = -div. L is singular; pin the
    // source row.
    let mut lp = l.clone();
    for c in 0..nv {
        lp.set(source, c, if c == source { 1.0 } else { 0.0 });
    }
    let mut rhs2: Vec<f64> = div.iter().map(|v| -v).collect();
    rhs2[source] = 0.0;
    let phi = match lu_decompose(&lp).and_then(|lu| lu.solve(&rhs2)) {
        Ok(v) => v,
        Err(_) => return vec![0.0; nv],
    };
    let min = phi.iter().cloned().fold(f64::INFINITY, f64::min);
    phi.iter().map(|v| v - min).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::metric::Sig;

    #[test]
    fn test_sphere_geodesics() {
        assert!(great_circle_check(1.0));
        assert!(great_circle_check(2.5));
        // unit-speed geodesic stays unit speed (metric norm conserved)
        let s = Metric::sphere(2, 1.0);
        let p0 = VecN::from(&[1.0, 0.3]);
        let v0dir = VecN::from(&[0.6, 0.5]);
        let n0 = s.norm(&p0, &v0dir);
        let v0 = v0dir.scale(1.0 / n0);
        let path = s.geodesic(&p0, &v0, 2.0, 0.01, Integrator::Rk4);
        for st in path.iter().step_by(50) {
            assert!((s.norm(&st.x, &st.v) - 1.0).abs() < 1e-6);
        }
        // adaptive integrator agrees with RK4
        let pa = s.geodesic_adaptive(&p0, &v0, 2.0, 1e-10);
        let end_rk = &path.last().unwrap().x;
        let end_dp = &pa.last().unwrap().x;
        assert!(end_rk.sub(end_dp).norm() < 1e-6);
        // geodesic path satisfies is_geodesic (as polyline it is close)
        let poly: Vec<VecN> = path.iter().map(|st| st.x.clone()).collect();
        assert!(s.is_geodesic(&poly, 0.1));
    }

    #[test]
    fn test_exp_log_and_distance() {
        let s = Metric::sphere(2, 1.0);
        let p = VecN::from(&[1.2, 0.4]);
        let q = VecN::from(&[1.0, 1.1]);
        // exp(log(p, q)) = q
        let v = s.log_map(&p, &q).expect("log_map");
        let q2 = s.exp_map(&p, &v);
        assert!(q2.sub(&q).norm() < 1e-8, "exp log error {}", q2.sub(&q).norm());
        // geodesic distance equals |log| in the metric
        let d_log = s.norm(&p, &v);
        let d_relax = s.geodesic_distance(&p, &q).unwrap();
        assert!((d_log - d_relax).abs() < 0.01 * d_log, "{d_log} vs {d_relax}");
        // and equals the central angle on the unit sphere (embed to check)
        let to3 = |x: &VecN| {
            Vec3::new(
                x[0].sin() * x[1].cos(),
                x[0].sin() * x[1].sin(),
                x[0].cos(),
            )
        };
        let angle = to3(&p).dot(&to3(&q)).clamp(-1.0, 1.0).acos();
        assert!((d_log - angle).abs() < 1e-4, "{d_log} vs {angle}");
        // interpolation midpoint is equidistant
        let mid = s.geodesic_interpolate(&p, &q, 0.5);
        let d1 = s.geodesic_distance(&p, &mid).unwrap();
        let d2 = s.geodesic_distance(&mid, &q).unwrap();
        assert!((d1 - d2).abs() < 0.02 * d1);
    }

    #[test]
    fn test_parallel_transport_and_holonomy() {
        let s = Metric::sphere(2, 1.0);
        // transport along a geodesic preserves the metric norm
        let p0 = VecN::from(&[1.0, 0.2]);
        let v0 = VecN::from(&[0.3, 0.8]);
        let w0 = VecN::from(&[0.5, -0.1]);
        let n0 = s.norm(&p0, &w0);
        let w1 = s.parallel_transport_along_geodesic(&w0, &p0, &v0, 1.5, 0.005);
        let path = s.geodesic(&p0, &v0, 1.5, 0.005, Integrator::Rk4);
        let n1 = s.norm(&path.last().unwrap().x, &w1);
        assert!((n1 - n0).abs() < 1e-5, "norm {n0} -> {n1}");
        // holonomy around a latitude circle theta = theta0:
        // angle = 2 pi (1 - cos theta0)
        let th0 = 1.0;
        let loop_path: Vec<VecN> = (0..=400)
            .map(|i| {
                VecN::from(&[th0, 2.0 * std::f64::consts::PI * i as f64 / 400.0])
            })
            .collect();
        let ang = s.holonomy_angle_2d(&loop_path).abs();
        let exact = 2.0 * std::f64::consts::PI * (1.0 - th0.cos());
        let ang_mod = if ang > std::f64::consts::PI { 2.0 * std::f64::consts::PI - ang } else { ang };
        let exact_mod = if exact > std::f64::consts::PI { 2.0 * std::f64::consts::PI - exact } else { exact };
        assert!(
            (ang_mod - exact_mod).abs() < 0.01,
            "holonomy {ang_mod} vs {exact_mod}"
        );
        // holonomy vector differs from the original (curvature detected)
        let e = VecN::from(&[1.0, 0.0]);
        let he = s.holonomy(&loop_path, &e);
        assert!(he.sub(&e).norm() > 0.1);
    }

    #[test]
    fn test_jacobi_and_conjugate() {
        // unit sphere: J(tau) = sin(tau) for J(0)=0, |J'(0)|=1; conjugate
        // point at tau = pi
        let s = Metric::sphere(2, 1.0);
        let p0 = VecN::from(&[std::f64::consts::FRAC_PI_2, 0.0]);
        let v0 = VecN::from(&[0.0, 1.0]); // unit speed along equator
        let j0dot = VecN::from(&[1.0, 0.0]); // orthogonal, unit
        let js = s.jacobi_field(&p0, &v0, &VecN::zeros(2), &j0dot, 3.5, 0.005);
        let steps = js.len() - 1;
        let dtau = 3.5 / steps as f64;
        // |J| ~ |sin(tau)| in the metric; check at tau = pi/2 and near pi
        let at = |tau: f64| {
            let i = (tau / dtau).round() as usize;
            &js[i]
        };
        // metric norm of J at the point (approximate using start frame is
        // fine since g is diagonal along the equator)
        let mid = at(std::f64::consts::FRAC_PI_2);
        assert!((mid.norm() - 1.0).abs() < 0.01, "J(pi/2) = {}", mid.norm());
        let near_pi = at(std::f64::consts::PI);
        assert!(near_pi.norm() < 0.02, "J(pi) = {}", near_pi.norm());
        let conj = s.conjugate_points(&p0, &v0, 3.5, 0.005);
        assert!(
            conj.iter().any(|&t| (t - std::f64::consts::PI).abs() < 0.05),
            "conjugate points {conj:?}"
        );
        // radius-2 sphere: conjugate at 2 pi
        let s2 = Metric::sphere(2, 2.0);
        let v0b = VecN::from(&[0.0, 0.5]);
        let conj2 = s2.conjugate_points(&p0, &v0b, 7.0, 0.01);
        assert!(
            conj2.iter().any(|&t| (t - 2.0 * std::f64::consts::PI).abs() < 0.1),
            "r=2 conjugate {conj2:?}"
        );
        // geodesic deviation on the sphere: A = -K J for J orthogonal to v
        let st = GeodesicState {
            x: p0.clone(),
            v: v0.clone(),
            tau: 0.0,
        };
        let dev = s.geodesic_deviation(&st, &j0dot, &VecN::zeros(2));
        assert!((dev[0] + 1.0).abs() < 1e-4, "deviation {dev:?}");
    }

    #[test]
    fn test_karcher_and_regression() {
        let s = Metric::sphere(2, 1.0);
        let c = VecN::from(&[std::f64::consts::FRAC_PI_2, 0.0]);
        let pts = vec![
            VecN::from(&[std::f64::consts::FRAC_PI_2 + 0.3, 0.0]),
            VecN::from(&[std::f64::consts::FRAC_PI_2 - 0.3, 0.0]),
            VecN::from(&[std::f64::consts::FRAC_PI_2, 0.3]),
            VecN::from(&[std::f64::consts::FRAC_PI_2, -0.3]),
        ];
        let m = s.karcher_mean(&pts, 20);
        assert!(m.sub(&c).norm() < 1e-6, "karcher mean {m:?}");
        // regression through points along the equator recovers the velocity
        let ts = [0.0, 0.5, 1.0];
        let pts2: Vec<VecN> = ts
            .iter()
            .map(|&t| VecN::from(&[std::f64::consts::FRAC_PI_2, 0.4 * t]))
            .collect();
        let (p0, v) = s.geodesic_regression(&ts, &pts2);
        assert!((p0[1]).abs() < 1e-3 && (p0[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-3);
        assert!((v[1] - 0.4).abs() < 0.01, "regression v = {v:?}");
    }

    #[test]
    fn test_schwarzschild_orbits() {
        // Mercury-like precession within 1% of 6 pi M / (a (1 - e^2))
        let m = 1.0_f64;
        let a = 1500.0_f64;
        let e = 0.2_f64;
        // Newtonian ellipse: L^2 = M a (1 - e^2); start at aphelion
        let l = (m * a * (1.0 - e * e)).sqrt();
        let r_ap = a * (1.0 + e);
        let orbit = schwarzschild_orbit(m, r_ap, l, 0.0, 4.2 * std::f64::consts::PI, 1e-3);
        // successive aphelia: local maxima of r refined by parabolic
        // interpolation (grid resolution alone is coarser than the shift)
        let mut aphelia = Vec::new();
        for i in 1..orbit.len() - 1 {
            let (rm, r0, rp) = (orbit[i - 1].1, orbit[i].1, orbit[i + 1].1);
            if r0 > rm && r0 > rp {
                let dphi = orbit[i].0 - orbit[i - 1].0;
                let offset = 0.5 * (rm - rp) / (rm - 2.0 * r0 + rp);
                aphelia.push(orbit[i].0 + offset * dphi);
            }
        }
        assert!(aphelia.len() >= 2, "need two aphelia, got {aphelia:?}");
        let advance = aphelia[1] - aphelia[0] - 2.0 * std::f64::consts::PI;
        let exact = perihelion_precession(m, a, e);
        assert!(
            (advance - exact).abs() < 0.01 * exact,
            "precession {advance} vs {exact}"
        );
        // light deflection: integrate the photon equation
        // u'' + u = 3 M u^2 from u = 0 with u'(0) = 1/b. At b = 100 M the
        // second-order term (15 pi/4)(M/b)^2 contributes ~3%, so compare
        // against the leading order at b = 1000 M where it is ~0.3%.
        let b = 1000.0 * m;
        let mut u = 1e-9_f64;
        let mut up = 1.0 / b;
        let dphi = 1e-4;
        let mut phi = 0.0;
        let mut u_prev = u;
        while u >= 0.0 && phi < 2.0 * std::f64::consts::PI {
            u_prev = u;
            let rhs = |u: f64| 3.0 * m * u * u - u;
            let k1u = up;
            let k1p = rhs(u);
            let k2u = up + 0.5 * dphi * k1p;
            let k2p = rhs(u + 0.5 * dphi * k1u);
            let k3u = up + 0.5 * dphi * k2p;
            let k3p = rhs(u + 0.5 * dphi * k2u);
            let k4u = up + dphi * k3p;
            let k4p = rhs(u + dphi * k3u);
            u += dphi / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            up += dphi / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
            phi += dphi;
        }
        // interpolate the exit angle where u crosses zero
        let phi_exit = phi - dphi + dphi * u_prev / (u_prev - u);
        let deflection = phi_exit - std::f64::consts::PI;
        let exact_defl = light_deflection(m, b);
        assert!(
            (deflection - exact_defl).abs() < 0.01 * exact_defl,
            "deflection {deflection} vs {exact_defl}"
        );
        // misc formulas
        assert!(shapiro_delay(1.0, 1e6, 1e6, 100.0) > 0.0);
        assert!(photon_orbit_stability(1.0) > 0.0);
    }

    #[test]
    fn test_null_geodesic_minkowski() {
        let mk = Metric::minkowski(4, Sig::MostlyPlus);
        let x0 = VecN::zeros(4);
        let k0 = VecN::from(&[2.0, 1.0, 0.0, 0.0]); // will be rescaled null
        let path = mk.null_geodesic(&x0, &k0, 1.0, 0.05);
        let kf = &path.last().unwrap().v;
        // g(k, k) = 0 preserved: -t_dot^2 + x_dot^2 = 0
        let null_res = -kf[0] * kf[0] + kf[1] * kf[1];
        assert!(null_res.abs() < 1e-10, "null violation {null_res}");
        // straight line: x = t
        let xe = &path.last().unwrap().x;
        assert!((xe[0] - xe[1]).abs() < 1e-10);
    }

    #[test]
    fn test_mesh_geodesics() {
        // small grid mesh on the plane z = 0
        let n = 8;
        let mut mesh = Mesh::new();
        for j in 0..=n {
            for i in 0..=n {
                mesh.vertices.push(Vec3::new(i as f64, j as f64, 0.0));
            }
        }
        let idx = |i: usize, j: usize| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                mesh.triangles.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                mesh.triangles.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        mesh.materials = vec![0; mesh.triangles.len()];
        // Dijkstra path between opposite corners exists and has finite length
        let path = geodesics_on_mesh_exact(&mesh, idx(0, 0), idx(n, n));
        assert!(path.len() >= 2);
        let len: f64 = path.windows(2).map(|w| (w[1] - w[0]).magnitude()).sum();
        let exact = (2.0 * (n as f64).powi(2)).sqrt();
        // edge paths overestimate the straight diagonal, but the diagonal
        // edges keep it exact here
        assert!(len >= exact - 1e-9 && len < 1.5 * exact, "len {len} vs {exact}");
        // heat method: distances grow with true distance, zero at source
        let d = heat_method_geodesic(&mesh, idx(0, 0), 2.0);
        // the min-shift normalization leaves the source at (near) zero
        assert!(d[idx(0, 0)] < 0.15 * d[idx(n, n)], "source {}", d[idx(0, 0)]);
        assert!(d[idx(4, 0)] > d[idx(2, 0)]);
        assert!(d[idx(6, 6)] > d[idx(3, 3)]);
        // rough magnitude: corner distance within 30% of Euclidean
        let rel = d[idx(n, 0)] / n as f64;
        assert!(rel > 0.7 && rel < 1.3, "heat distance ratio {rel}");
    }

    #[test]
    fn test_polar_and_sphere_points() {
        let h = Metric::hyperbolic_ball(2);
        let p = VecN::zeros(2);
        // geodesic circle of radius r around origin: points at Euclidean
        // radius tanh(r/2)
        let pts = h.geodesic_sphere(&p, 1.0, 8);
        for q in &pts {
            let er = q.norm();
            assert!((er - (0.5_f64).tanh()).abs() < 1e-4, "radius {er}");
        }
        let q = h.geodesic_polar_coords(&p, 1.0, 0.7);
        assert!((q.norm() - (0.5_f64).tanh()).abs() < 1e-4);
    }
}
