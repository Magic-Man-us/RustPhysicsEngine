//! Structural dynamics: finite-element bars and beams, modal analysis
//! with general (consistent) mass matrices, Rayleigh damping, implicit
//! time integration (Newmark-β, HHT-α), model reduction, response
//! spectra, and experimental modal analysis tools.

use crate::fractals::Complex;
use crate::linalg::{cholesky, eigen_symmetric, eigenvalues_general, lu_decompose, solve, Matrix};
use crate::math::constants::PI;

const TWO_PI: f64 = 2.0 * PI;

/// Structural model M·x″ + C·x′ + K·x = F(t) with full matrices.
#[derive(Debug, Clone)]
pub struct ModalModel {
    pub m: Matrix,
    pub c: Matrix,
    pub k: Matrix,
}

fn mat_dim(m: &Matrix) -> usize {
    m.rows
}

impl ModalModel {
    fn n(&self) -> usize {
        mat_dim(&self.k)
    }

    /// Clamped-free axial bar discretized with n_elem consistent-mass
    /// two-node elements (dofs are the axial displacements of nodes
    /// 1..=n_elem; node 0 is fixed).
    #[must_use]
    pub fn from_fem_1d_bar(n_elem: usize, length: f64, area: f64, young: f64, rho: f64) -> Self {
        let le = length / n_elem as f64;
        let ke = young * area / le;
        let me = rho * area * le / 6.0;
        let n_nodes = n_elem + 1;
        let mut k = Matrix::zeros(n_nodes, n_nodes);
        let mut m = Matrix::zeros(n_nodes, n_nodes);
        for e in 0..n_elem {
            let idx = [e, e + 1];
            let ke_loc = [[ke, -ke], [-ke, ke]];
            let me_loc = [[2.0 * me, me], [me, 2.0 * me]];
            for a in 0..2 {
                for b in 0..2 {
                    k.set(idx[a], idx[b], k.get(idx[a], idx[b]) + ke_loc[a][b]);
                    m.set(idx[a], idx[b], m.get(idx[a], idx[b]) + me_loc[a][b]);
                }
            }
        }
        // Remove the clamped node 0.
        let reduce = |full: &Matrix| {
            Matrix::from_fn(n_elem, n_elem, |i, j| full.get(i + 1, j + 1))
        };
        Self { m: reduce(&m), c: Matrix::zeros(n_elem, n_elem), k: reduce(&k) }
    }

    /// Cantilever Euler-Bernoulli beam with n_elem two-node elements
    /// (dofs per free node: transverse deflection w and rotation θ).
    #[must_use]
    pub fn from_fem_beam(
        n_elem: usize,
        length: f64,
        young: f64,
        i_area: f64,
        rho: f64,
        area: f64,
    ) -> Self {
        let le = length / n_elem as f64;
        let ei = young * i_area / le.powi(3);
        let ml = rho * area * le / 420.0;
        let ke_loc = [
            [12.0 * ei, 6.0 * ei * le, -12.0 * ei, 6.0 * ei * le],
            [6.0 * ei * le, 4.0 * ei * le * le, -6.0 * ei * le, 2.0 * ei * le * le],
            [-12.0 * ei, -6.0 * ei * le, 12.0 * ei, -6.0 * ei * le],
            [6.0 * ei * le, 2.0 * ei * le * le, -6.0 * ei * le, 4.0 * ei * le * le],
        ];
        let me_loc = [
            [156.0 * ml, 22.0 * le * ml, 54.0 * ml, -13.0 * le * ml],
            [22.0 * le * ml, 4.0 * le * le * ml, 13.0 * le * ml, -3.0 * le * le * ml],
            [54.0 * ml, 13.0 * le * ml, 156.0 * ml, -22.0 * le * ml],
            [-13.0 * le * ml, -3.0 * le * le * ml, -22.0 * le * ml, 4.0 * le * le * ml],
        ];
        let n_dof_full = 2 * (n_elem + 1);
        let mut k = Matrix::zeros(n_dof_full, n_dof_full);
        let mut m = Matrix::zeros(n_dof_full, n_dof_full);
        for e in 0..n_elem {
            let idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3];
            for a in 0..4 {
                for b in 0..4 {
                    k.set(idx[a], idx[b], k.get(idx[a], idx[b]) + ke_loc[a][b]);
                    m.set(idx[a], idx[b], m.get(idx[a], idx[b]) + me_loc[a][b]);
                }
            }
        }
        // Clamp node 0 (w0 = θ0 = 0).
        let nf = n_dof_full - 2;
        let reduce = |full: &Matrix| Matrix::from_fn(nf, nf, |i, j| full.get(i + 2, j + 2));
        Self { m: reduce(&m), c: Matrix::zeros(nf, nf), k: reduce(&k) }
    }

    /// Lumped-parameter model from mass, spring, and damper lists
    /// ((i, i, v) grounds dof i; (i, j, v) couples i and j).
    #[must_use]
    pub fn from_lumped(
        masses: &[f64],
        springs: &[(usize, usize, f64)],
        dampers: &[(usize, usize, f64)],
    ) -> Self {
        let n = masses.len();
        let assemble = |list: &[(usize, usize, f64)]| {
            let mut a = Matrix::zeros(n, n);
            for &(i, j, v) in list {
                if i == j {
                    a.set(i, i, a.get(i, i) + v);
                } else {
                    a.set(i, i, a.get(i, i) + v);
                    a.set(j, j, a.get(j, j) + v);
                    a.set(i, j, a.get(i, j) - v);
                    a.set(j, i, a.get(j, i) - v);
                }
            }
            a
        };
        let mut m = Matrix::zeros(n, n);
        for (i, &mv) in masses.iter().enumerate() {
            m.set(i, i, mv);
        }
        Self { m, c: assemble(dampers), k: assemble(springs) }
    }

    /// Set C = αM + βK.
    pub fn rayleigh_damping(&mut self, alpha: f64, beta: f64) {
        let n = self.n();
        self.c = Matrix::from_fn(n, n, |i, j| alpha * self.m.get(i, j) + beta * self.k.get(i, j));
    }

    /// Choose Rayleigh α, β to hit damping ratios ζ₁ at f₁ and ζ₂ at f₂
    /// (Hz): ζ = α/(2ω) + βω/2.
    pub fn rayleigh_from_ratios(&mut self, zeta1: f64, f1: f64, zeta2: f64, f2: f64) {
        let w1 = TWO_PI * f1;
        let w2 = TWO_PI * f2;
        let beta = 2.0 * (zeta2 * w2 - zeta1 * w1) / (w2 * w2 - w1 * w1);
        let alpha = 2.0 * zeta1 * w1 - beta * w1 * w1;
        self.rayleigh_damping(alpha, beta);
    }

    /// Undamped modes of the generalized problem K·φ = ω²M·φ via the
    /// Cholesky reduction L⁻¹KL⁻ᵀ: (frequencies rad/s ascending,
    /// mass-orthonormal mode shapes as columns).
    ///
    /// # Panics
    /// Panics if M is not positive definite or the eigen solve fails.
    #[must_use]
    pub fn modes(&self) -> (Vec<f64>, Matrix) {
        let n = self.n();
        let l = cholesky(&self.m).expect("mass matrix must be positive definite");
        // B = L⁻¹ K L⁻ᵀ: solve L Y = K, then L Z = Yᵀ (Z = L⁻¹ K L⁻ᵀ as rows).
        let fwd = |l: &Matrix, b: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut acc = b[i];
                for (j, &yj) in y.iter().enumerate().take(i) {
                    acc -= l.get(i, j) * yj;
                }
                y[i] = acc / l.get(i, i);
            }
            y
        };
        let back = |l: &Matrix, b: &[f64]| -> Vec<f64> {
            // Solve Lᵀ x = b.
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let mut acc = b[i];
                for (j, &xj) in x.iter().enumerate().skip(i + 1) {
                    acc -= l.get(j, i) * xj;
                }
                x[i] = acc / l.get(i, i);
            }
            x
        };
        // Column c of Y = L⁻¹ K.
        let mut y = Matrix::zeros(n, n);
        for c in 0..n {
            let col: Vec<f64> = (0..n).map(|r| self.k.get(r, c)).collect();
            let sol = fwd(&l, &col);
            for (r, &v) in sol.iter().enumerate() {
                y.set(r, c, v);
            }
        }
        // B = Y L⁻ᵀ ⇒ Bᵀ = L⁻¹ Yᵀ; B symmetric so compute rows the same way.
        let mut b = Matrix::zeros(n, n);
        for r in 0..n {
            let row: Vec<f64> = (0..n).map(|c| y.get(r, c)).collect();
            let sol = fwd(&l, &row);
            for (c, &v) in sol.iter().enumerate() {
                b.set(r, c, v);
            }
        }
        // Symmetrize against roundoff.
        let bsym = Matrix::from_fn(n, n, |i, j| 0.5 * (b.get(i, j) + b.get(j, i)));
        let eig = eigen_symmetric(&bsym, 1e-12, 300).expect("modal eigen solve failed");
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&x, &yv| {
            eig.values[x].partial_cmp(&eig.values[yv]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let freqs: Vec<f64> = order.iter().map(|&i| eig.values[i].max(0.0).sqrt()).collect();
        let mut shapes = Matrix::zeros(n, n);
        for (col, &src) in order.iter().enumerate() {
            let u: Vec<f64> = (0..n).map(|r| eig.vectors.get(r, src)).collect();
            let phi = back(&l, &u); // φ = L⁻ᵀ u
            for (r, &v) in phi.iter().enumerate() {
                shapes.set(r, col, v);
            }
        }
        (freqs, shapes)
    }

    /// State-space eigen solution of the damped system: eigenvalues λ
    /// with Im λ ≥ 0 (one per underdamped mode) and their complex mode
    /// shapes, from A = \[\[0, I\], \[−M⁻¹K, −M⁻¹C\]\] with eigenvectors
    /// recovered by shifted inverse iteration.
    ///
    /// # Panics
    /// Panics if the eigen machinery fails.
    #[must_use]
    pub fn damped_modes(&self) -> Vec<(Complex, Vec<Complex>)> {
        let n = self.n();
        let m_lu = lu_decompose(&self.m).expect("mass matrix singular");
        // M⁻¹K and M⁻¹C column by column.
        let mut mk = Matrix::zeros(n, n);
        let mut mc = Matrix::zeros(n, n);
        for c in 0..n {
            let kc: Vec<f64> = (0..n).map(|r| self.k.get(r, c)).collect();
            let cc: Vec<f64> = (0..n).map(|r| self.c.get(r, c)).collect();
            let sk = m_lu.solve(&kc).expect("solve failed");
            let sc = m_lu.solve(&cc).expect("solve failed");
            for (r, (&kv, &cv)) in sk.iter().zip(&sc).enumerate() {
                mk.set(r, c, kv);
                mc.set(r, c, cv);
            }
        }
        let dim = 2 * n;
        let a = Matrix::from_fn(dim, dim, |i, j| {
            if i < n {
                if j == i + n {
                    1.0
                } else {
                    0.0
                }
            } else if j < n {
                -mk.get(i - n, j)
            } else {
                -mc.get(i - n, j - n)
            }
        });
        let eigs = eigenvalues_general(&a, 100).expect("state-space eigenvalues failed");
        // Keep one of each conjugate pair (Im ≥ 0), sorted by |λ|.
        let mut lambdas: Vec<Complex> = eigs.into_iter().filter(|l| l.im >= -1e-12).collect();
        lambdas.sort_by(|x, y| x.norm().partial_cmp(&y.norm()).unwrap_or(std::cmp::Ordering::Equal));
        lambdas
            .into_iter()
            .map(|lambda| {
                let v = inverse_iterate(&a, lambda);
                (lambda, v[..n].to_vec())
            })
            .collect()
    }

    /// Damping ratios ζᵢ = −Re λᵢ/|λᵢ| from the damped modes.
    #[must_use]
    pub fn modal_damping_ratios(&self) -> Vec<f64> {
        self.damped_modes()
            .iter()
            .filter(|(l, _)| l.norm() > 1e-12)
            .map(|(l, _)| -l.re / l.norm())
            .collect()
    }

    /// Receptance FRF H_ij(ω) = \[(K + jωC − ω²M)⁻¹\]_ij.
    #[must_use]
    pub fn frf(&self, i: usize, j: usize, omega: f64) -> Complex {
        let col = self.dynamic_solve_unit(j, omega);
        col[i]
    }

    /// Magnitude receptance matrix |H(ω)|.
    #[must_use]
    pub fn frf_matrix(&self, omega: f64) -> Matrix {
        let n = self.n();
        let mut h = Matrix::zeros(n, n);
        for j in 0..n {
            let col = self.dynamic_solve_unit(j, omega);
            for (i, cv) in col.iter().enumerate() {
                h.set(i, j, cv.norm());
            }
        }
        h
    }

    /// Solve (K + jωC − ω²M)·x = e_j via the doubled real system.
    fn dynamic_solve_unit(&self, j: usize, omega: f64) -> Vec<Complex> {
        let n = self.n();
        let mut big = Matrix::zeros(2 * n, 2 * n);
        for r in 0..n {
            for c in 0..n {
                let a = self.k.get(r, c) - omega * omega * self.m.get(r, c);
                let b = omega * self.c.get(r, c);
                big.set(r, c, a);
                big.set(n + r, n + c, a);
                big.set(r, n + c, -b);
                big.set(n + r, c, b);
            }
        }
        let mut rhs = vec![0.0; 2 * n];
        rhs[j] = 1.0;
        let sol = solve(&big, &rhs).expect("dynamic stiffness singular");
        (0..n).map(|r| Complex::new(sol[r], sol[n + r])).collect()
    }

    /// Newmark-β implicit integration under a time-varying force;
    /// returns the displacement vector at every step (t = 0 included).
    /// β = 1/4, γ = 1/2 is the unconditionally stable trapezoid rule.
    ///
    /// # Panics
    /// Panics on dimension mismatches or a singular effective stiffness.
    #[must_use]
    #[allow(clippy::too_many_arguments)] // signature fixed by the roadmap
    pub fn newmark_beta(
        &self,
        force: &dyn Fn(f64) -> Vec<f64>,
        x0: &[f64],
        v0: &[f64],
        t_end: f64,
        dt: f64,
        beta: f64,
        gamma: f64,
    ) -> Vec<Vec<f64>> {
        let n = self.n();
        assert!(x0.len() == n && v0.len() == n, "state dimension mismatch");
        let mat_vec = |a: &Matrix, x: &[f64]| -> Vec<f64> {
            (0..n).map(|i| (0..n).map(|j| a.get(i, j) * x[j]).sum()).collect()
        };
        // Initial acceleration from equilibrium at t = 0.
        let m_lu = lu_decompose(&self.m).expect("mass matrix singular");
        let f0 = force(0.0);
        let rhs0: Vec<f64> = (0..n)
            .map(|i| f0[i] - mat_vec(&self.c, v0)[i] - mat_vec(&self.k, x0)[i])
            .collect();
        let mut acc = m_lu.solve(&rhs0).expect("initial acceleration failed");
        let a0 = 1.0 / (beta * dt * dt);
        let a1 = gamma / (beta * dt);
        let keff = Matrix::from_fn(n, n, |i, j| {
            self.k.get(i, j) + a0 * self.m.get(i, j) + a1 * self.c.get(i, j)
        });
        let keff_lu = lu_decompose(&keff).expect("effective stiffness singular");
        let mut x = x0.to_vec();
        let mut v = v0.to_vec();
        let steps = (t_end / dt).round() as usize;
        let mut out = Vec::with_capacity(steps + 1);
        out.push(x.clone());
        let mut t = 0.0;
        for _ in 0..steps {
            t += dt;
            let f = force(t);
            // Predictors.
            let xp: Vec<f64> = (0..n)
                .map(|i| a0 * x[i] + v[i] / (beta * dt) + (0.5 / beta - 1.0) * acc[i])
                .collect();
            let vp: Vec<f64> = (0..n)
                .map(|i| {
                    a1 * x[i] + (gamma / beta - 1.0) * v[i]
                        + dt * (gamma / (2.0 * beta) - 1.0) * acc[i]
                })
                .collect();
            let m_xp = mat_vec(&self.m, &xp);
            let c_vp = mat_vec(&self.c, &vp);
            let rhs: Vec<f64> = (0..n).map(|i| f[i] + m_xp[i] + c_vp[i]).collect();
            let x_new = keff_lu.solve(&rhs).expect("Newmark solve failed");
            let acc_new: Vec<f64> = (0..n).map(|i| a0 * (x_new[i] - x[i]) - v[i] / (beta * dt) - (0.5 / beta - 1.0) * acc[i]).collect();
            let v_new: Vec<f64> = (0..n)
                .map(|i| v[i] + dt * ((1.0 - gamma) * acc[i] + gamma * acc_new[i]))
                .collect();
            x = x_new;
            v = v_new;
            acc = acc_new;
            out.push(x.clone());
        }
        out
    }

    /// HHT-α integration (α ∈ [−1/3, 0]; β = (1−α)²/4, γ = 1/2 − α):
    /// numerically damps spurious high-frequency content while keeping
    /// second-order accuracy.
    ///
    /// # Panics
    /// Panics on dimension mismatches or singular matrices.
    #[must_use]
    pub fn hht_alpha(
        &self,
        force: &dyn Fn(f64) -> Vec<f64>,
        x0: &[f64],
        v0: &[f64],
        t_end: f64,
        dt: f64,
        alpha: f64,
    ) -> Vec<Vec<f64>> {
        let n = self.n();
        assert!(x0.len() == n && v0.len() == n, "state dimension mismatch");
        let beta = (1.0 - alpha).powi(2) / 4.0;
        let gamma = 0.5 - alpha;
        let mat_vec = |a: &Matrix, x: &[f64]| -> Vec<f64> {
            (0..n).map(|i| (0..n).map(|j| a.get(i, j) * x[j]).sum()).collect()
        };
        let m_lu = lu_decompose(&self.m).expect("mass matrix singular");
        let f0 = force(0.0);
        let rhs0: Vec<f64> = (0..n)
            .map(|i| f0[i] - mat_vec(&self.c, v0)[i] - mat_vec(&self.k, x0)[i])
            .collect();
        let mut acc = m_lu.solve(&rhs0).expect("initial acceleration failed");
        let a0 = 1.0 / (beta * dt * dt);
        let keff = Matrix::from_fn(n, n, |i, j| {
            a0 * self.m.get(i, j)
                + (1.0 + alpha) * gamma / (beta * dt) * self.c.get(i, j)
                + (1.0 + alpha) * self.k.get(i, j)
        });
        let keff_lu = lu_decompose(&keff).expect("effective stiffness singular");
        let mut x = x0.to_vec();
        let mut v = v0.to_vec();
        let steps = (t_end / dt).round() as usize;
        let mut out = Vec::with_capacity(steps + 1);
        out.push(x.clone());
        let mut t = 0.0;
        for _ in 0..steps {
            let t_new = t + dt;
            let f_new = force(t_new);
            let f_old = force(t);
            // Newmark predictors folded to the right-hand side:
            // a_{n+1} = a0(x_{n+1} − x_n) − v_n/(βΔt) − (1/2β − 1)a_n
            // v_{n+1} = γ/(βΔt)·x_{n+1} + w_n
            let m_pred: Vec<f64> = (0..n)
                .map(|i| a0 * x[i] + v[i] / (beta * dt) + (0.5 / beta - 1.0) * acc[i])
                .collect();
            let w_n: Vec<f64> = (0..n)
                .map(|i| {
                    v[i] * (1.0 - gamma / beta)
                        + dt * acc[i] * (1.0 - gamma / (2.0 * beta))
                        - gamma / (beta * dt) * x[i]
                })
                .collect();
            let m_term = mat_vec(&self.m, &m_pred);
            let c_wn = mat_vec(&self.c, &w_n);
            let c_vn = mat_vec(&self.c, &v);
            let k_xn = mat_vec(&self.k, &x);
            let rhs: Vec<f64> = (0..n)
                .map(|i| {
                    (1.0 + alpha) * f_new[i] - alpha * f_old[i]
                        + m_term[i]
                        - (1.0 + alpha) * c_wn[i]
                        + alpha * c_vn[i]
                        + alpha * k_xn[i]
                })
                .collect();
            let x_new = keff_lu.solve(&rhs).expect("HHT solve failed");
            let acc_new: Vec<f64> = (0..n)
                .map(|i| a0 * (x_new[i] - x[i]) - v[i] / (beta * dt) - (0.5 / beta - 1.0) * acc[i])
                .collect();
            let v_new: Vec<f64> = (0..n)
                .map(|i| v[i] + dt * ((1.0 - gamma) * acc[i] + gamma * acc_new[i]))
                .collect();
            x = x_new;
            v = v_new;
            acc = acc_new;
            t = t_new;
            out.push(x.clone());
        }
        out
    }

    /// Reduced model keeping the lowest n_modes modal coordinates
    /// (unit modal masses, diagonal stiffness ω², modal damping).
    #[must_use]
    pub fn modal_truncation(&self, n_modes: usize) -> ModalModel {
        let (freqs, shapes) = self.modes();
        let n = self.n();
        let nm = n_modes.min(n);
        let mut kr = Matrix::zeros(nm, nm);
        let mut cr = Matrix::zeros(nm, nm);
        for (a, &fa) in freqs.iter().enumerate().take(nm) {
            kr.set(a, a, fa * fa);
            for b in 0..nm {
                let mut cv = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        cv += shapes.get(i, a) * self.c.get(i, j) * shapes.get(j, b);
                    }
                }
                cr.set(a, b, cv);
            }
        }
        ModalModel { m: Matrix::identity(nm), c: cr, k: kr }
    }

    /// Guyan (static) condensation onto the given master dofs.
    ///
    /// # Panics
    /// Panics if the slave stiffness block is singular.
    #[must_use]
    pub fn guyan_reduction(&self, master_dofs: &[usize]) -> ModalModel {
        let n = self.n();
        let slaves: Vec<usize> = (0..n).filter(|i| !master_dofs.contains(i)).collect();
        let nm = master_dofs.len();
        let ns = slaves.len();
        // T = [I; −Kss⁻¹ Ksm]
        let kss = Matrix::from_fn(ns, ns, |i, j| self.k.get(slaves[i], slaves[j]));
        let ksm = Matrix::from_fn(ns, nm, |i, j| self.k.get(slaves[i], master_dofs[j]));
        let kss_lu = lu_decompose(&kss).expect("slave stiffness singular");
        let mut t = Matrix::zeros(n, nm);
        for (r, &d) in master_dofs.iter().enumerate() {
            t.set(d, r, 1.0);
        }
        for c in 0..nm {
            let col: Vec<f64> = (0..ns).map(|r| ksm.get(r, c)).collect();
            let sol = kss_lu.solve(&col).expect("Guyan solve failed");
            for (&d, &v) in slaves.iter().zip(&sol) {
                t.set(d, c, -v);
            }
        }
        let project = |a: &Matrix| -> Matrix {
            // TᵀAT
            let at = a.mul(&t).expect("dims");
            t.transpose().mul(&at).expect("dims")
        };
        ModalModel { m: project(&self.m), c: project(&self.c), k: project(&self.k) }
    }

    /// Modal response spectrum for a ground acceleration record: for
    /// each mode, (natural frequency Hz, peak SDOF relative-displacement
    /// response at damping ζ).
    #[must_use]
    pub fn response_spectrum(&self, ground_accel: &[f64], dt: f64, zeta: f64) -> Vec<(f64, f64)> {
        let (freqs, _) = self.modes();
        freqs
            .iter()
            .map(|&w| {
                let f_hz = w / TWO_PI;
                let peak = sdof_peak_displacement(ground_accel, dt, w, zeta);
                (f_hz, peak)
            })
            .collect()
    }

    /// Modal assurance criterion between two mode shapes.
    #[must_use]
    pub fn mac(phi1: &[f64], phi2: &[f64]) -> f64 {
        let dot: f64 = phi1.iter().zip(phi2).map(|(a, b)| a * b).sum();
        let n1: f64 = phi1.iter().map(|v| v * v).sum();
        let n2: f64 = phi2.iter().map(|v| v * v).sum();
        dot * dot / (n1 * n2).max(1e-300)
    }

    /// Campbell diagram data: (rpm, natural frequencies in Hz) over a
    /// speed range (structure frequencies here are speed-independent;
    /// intersect with the order lines to find criticals).
    #[must_use]
    pub fn campbell_diagram(
        &self,
        rpm_range: (f64, f64),
        n: usize,
        orders: &[f64],
    ) -> Vec<(f64, Vec<f64>)> {
        let _ = orders;
        let (freqs, _) = self.modes();
        let f_hz: Vec<f64> = freqs.iter().map(|w| w / TWO_PI).collect();
        (0..n)
            .map(|i| {
                let rpm =
                    rpm_range.0 + (rpm_range.1 - rpm_range.0) * i as f64 / (n - 1).max(1) as f64;
                (rpm, f_hz.clone())
            })
            .collect()
    }

    /// Critical speeds (rpm) where an excitation order line crosses a
    /// natural frequency: rpm = 60·fᵢ/order.
    #[must_use]
    pub fn critical_speeds(&self, orders: &[f64]) -> Vec<f64> {
        let (freqs, _) = self.modes();
        let mut out = Vec::new();
        for &w in &freqs {
            for &ord in orders {
                if ord > 0.0 {
                    out.push(60.0 * w / TWO_PI / ord);
                }
            }
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Relative frequency margin of each excitation to the nearest
    /// natural frequency: min |f_exc − fᵢ|/fᵢ.
    #[must_use]
    pub fn resonance_margins(&self, excitation_freqs: &[f64]) -> Vec<f64> {
        let (freqs, _) = self.modes();
        excitation_freqs
            .iter()
            .map(|&fe| {
                freqs
                    .iter()
                    .map(|&w| {
                        let f = w / TWO_PI;
                        (fe - f).abs() / f.max(1e-300)
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }
}

/// Peak |x| of an SDOF oscillator (ω, ζ) under base acceleration
/// (Newmark average-acceleration integration).
fn sdof_peak_displacement(ground_accel: &[f64], dt: f64, omega: f64, zeta: f64) -> f64 {
    let (mut x, mut v) = (0.0_f64, 0.0_f64);
    let mut a = -ground_accel.first().copied().unwrap_or(0.0);
    let (beta, gamma) = (0.25, 0.5);
    let keff = omega * omega + gamma / (beta * dt) * (2.0 * zeta * omega) + 1.0 / (beta * dt * dt);
    let mut peak = 0.0_f64;
    for &ag in &ground_accel[1..] {
        let rhs = -ag
            + (x / (beta * dt * dt) + v / (beta * dt) + (0.5 / beta - 1.0) * a)
            + 2.0 * zeta * omega
                * (gamma / (beta * dt) * x + (gamma / beta - 1.0) * v
                    + dt * (gamma / (2.0 * beta) - 1.0) * a);
        let x_new = rhs / keff;
        let a_new = (x_new - x) / (beta * dt * dt) - v / (beta * dt) - (0.5 / beta - 1.0) * a;
        let v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new);
        x = x_new;
        v = v_new;
        a = a_new;
        peak = peak.max(x.abs());
    }
    peak
}

/// Shifted inverse iteration for one complex eigenvector of a real
/// matrix (doubled real system with a small regularizing offset).
fn inverse_iterate(a: &Matrix, lambda: Complex) -> Vec<Complex> {
    let n = mat_dim(a);
    let shift = lambda + Complex::new(1e-8 * (1.0 + lambda.norm()), 1e-8);
    let big = Matrix::from_fn(2 * n, 2 * n, |i, j| {
        let (r, c) = (i % n, j % n);
        let are = a.get(r, c) - if r == c { shift.re } else { 0.0 };
        let aim = if r == c { shift.im } else { 0.0 };
        match (i < n, j < n) {
            (true, true) | (false, false) => are,
            (true, false) => aim,
            (false, true) => -aim,
        }
    });
    let lu = match lu_decompose(&big) {
        Ok(l) => l,
        Err(_) => return vec![Complex::new(0.0, 0.0); n],
    };
    let mut v: Vec<f64> = (0..2 * n).map(|i| ((i * 2654435761) % 97) as f64 / 97.0 - 0.5).collect();
    for _ in 0..4 {
        let w = match lu.solve(&v) {
            Ok(w) => w,
            Err(_) => break,
        };
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-300 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();
    }
    (0..n).map(|i| Complex::new(v[i], v[i + n])).collect()
}

/// Peak-picking experimental modal analysis on a receptance FRF:
/// (natural frequency, damping ratio) per resolved peak via half-power
/// bandwidths.
#[must_use]
pub fn experimental_modal_peak_picking(frf: &[Complex], freqs: &[f64]) -> Vec<(f64, f64)> {
    let mag: Vec<f64> = frf.iter().map(|c| c.norm()).collect();
    let mut sorted = mag.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let mut out = Vec::new();
    for i in 1..mag.len().saturating_sub(1) {
        if mag[i] > mag[i - 1] && mag[i] > mag[i + 1] && mag[i] > 5.0 * median {
            let bw = half_power_bandwidth(&mag, freqs, i);
            if bw > 0.0 {
                out.push((freqs[i], bw / (2.0 * freqs[i])));
            }
        }
    }
    out
}

/// Half-power (−3 dB) bandwidth around the FRF magnitude peak at
/// `peak_idx`, with linear interpolation; 0 when a crossing is missing.
#[must_use]
pub fn half_power_bandwidth(frf_mag: &[f64], freqs: &[f64], peak_idx: usize) -> f64 {
    let target = frf_mag[peak_idx] / std::f64::consts::SQRT_2;
    let mut f_lo = None;
    for i in (1..=peak_idx).rev() {
        if frf_mag[i - 1] < target {
            let t = (target - frf_mag[i - 1]) / (frf_mag[i] - frf_mag[i - 1]);
            f_lo = Some(freqs[i - 1] + t * (freqs[i] - freqs[i - 1]));
            break;
        }
    }
    let mut f_hi = None;
    for i in peak_idx..frf_mag.len() - 1 {
        if frf_mag[i + 1] < target {
            let t = (frf_mag[i] - target) / (frf_mag[i] - frf_mag[i + 1]);
            f_hi = Some(freqs[i] + t * (freqs[i + 1] - freqs[i]));
            break;
        }
    }
    match (f_lo, f_hi) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    }
}

/// Kasa circle fit of an FRF arc in the Nyquist plane around a
/// resonance; returns (f₀, ζ) using the angular-sweep-rate maximum for
/// f₀ and the standard circle-fit damping formula.
///
/// # Panics
/// Panics if fewer than 5 points are given.
#[must_use]
pub fn circle_fit(frf: &[Complex], freqs: &[f64], window: usize) -> (f64, f64) {
    assert!(frf.len() >= 5 && frf.len() == freqs.len(), "need >= 5 FRF points");
    // Restrict to `window` points around the magnitude peak.
    let peak = frf
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let half = window.max(4) / 2;
    let lo = peak.saturating_sub(half);
    let hi = (peak + half + 1).min(frf.len());
    let pts: Vec<(f64, f64)> = frf[lo..hi].iter().map(|c| (c.re, c.im)).collect();
    let fs = &freqs[lo..hi];
    // Kasa least squares: minimize Σ (x² + y² + D x + E y + F)².
    let mut ata = Matrix::zeros(3, 3);
    let mut rhs = vec![0.0; 3];
    for &(x, y) in &pts {
        let row = [x, y, 1.0];
        let b = -(x * x + y * y);
        for i in 0..3 {
            rhs[i] += row[i] * b;
            for j in 0..3 {
                ata.set(i, j, ata.get(i, j) + row[i] * row[j]);
            }
        }
    }
    let sol = solve(&ata, &rhs).expect("circle fit singular");
    let cx = -sol[0] / 2.0;
    let cy = -sol[1] / 2.0;
    // Angles around the center; f0 at maximum angular sweep rate.
    let angles: Vec<f64> = pts.iter().map(|&(x, y)| (y - cy).atan2(x - cx)).collect();
    let unwrapped = crate::dsp::phase::unwrap_phase(&angles);
    let mut best = (1usize, 0.0_f64);
    for i in 1..unwrapped.len() {
        let rate = (unwrapped[i] - unwrapped[i - 1]).abs() / (fs[i] - fs[i - 1]).abs().max(1e-300);
        if rate > best.1 {
            best = (i, rate);
        }
    }
    let i0 = best.0;
    let f0 = 0.5 * (fs[i0 - 1] + fs[i0]);
    // Circle-fit damping from two flanking points a (below) and b (above):
    // ζ = (f_b² − f_a²)/(2 f₀² (tan(θ_a/2) + tan(θ_b/2))), angles measured
    // from the resonance point on the circle.
    let theta_res = 0.5 * (unwrapped[i0 - 1] + unwrapped[i0]);
    let pick = |idx: usize| -> Option<(f64, f64)> {
        if idx < unwrapped.len() {
            let dth = (unwrapped[idx] - theta_res).abs();
            if dth > 1e-6 && dth < PI {
                return Some((fs[idx], dth));
            }
        }
        None
    };
    let below = (0..i0).rev().find_map(pick);
    let above = (i0..unwrapped.len()).find_map(pick);
    let zeta = match (below, above) {
        (Some((fa, ta)), Some((fb, tb))) => {
            (fb * fb - fa * fa).abs() / (2.0 * f0 * f0 * ((ta / 2.0).tan() + (tb / 2.0).tan()))
        }
        _ => 0.0,
    };
    (f0, zeta)
}

/// Operational deflection shape: the complex amplitude of every
/// measured channel at frequency ω (single-bin correlation at fs).
#[must_use]
pub fn operational_deflection_shape(
    responses: &[Vec<f64>],
    omega: f64,
    fs: f64,
) -> Vec<Complex> {
    responses
        .iter()
        .map(|ch| {
            let mut re = 0.0;
            let mut im = 0.0;
            for (i, &v) in ch.iter().enumerate() {
                let ang = omega * i as f64 / fs;
                re += v * ang.cos();
                im -= v * ang.sin();
            }
            let scale = 2.0 / ch.len().max(1) as f64;
            Complex::new(re * scale, im * scale)
        })
        .collect()
}

/// Covariance-driven stochastic subspace identification: output-only
/// modal frequencies and damping ratios from response channels sampled
/// at fs. `order` is the state dimension (≥ 2 per expected mode).
///
/// # Panics
/// Panics if the SVD or eigen machinery fails, or the data is shorter
/// than 4·order.
#[must_use]
pub fn stochastic_subspace_identification(
    outputs: &[Vec<f64>],
    fs: f64,
    order: usize,
) -> Vec<(f64, f64)> {
    let n_ch = outputs.len();
    let len = outputs[0].len();
    let block = 2 * order;
    assert!(len > 4 * block, "record too short for the requested order");
    // Output covariance sequence R_k (n_ch × n_ch), k = 1..2·block.
    let covs: Vec<Matrix> = (1..=2 * block)
        .map(|k| {
            Matrix::from_fn(n_ch, n_ch, |a, b| {
                let mut acc = 0.0;
                for t in 0..len - k {
                    acc += outputs[a][t + k] * outputs[b][t];
                }
                acc / (len - k) as f64
            })
        })
        .collect();
    // Block Hankel of covariances.
    let rows = block * n_ch;
    let cols = block * n_ch;
    let hankel = Matrix::from_fn(rows, cols, |i, j| {
        let (bi, ri) = (i / n_ch, i % n_ch);
        let (bj, rj) = (j / n_ch, j % n_ch);
        covs[bi + bj].get(ri, rj)
    });
    let dec = crate::linalg::svd(&hankel).expect("SSI SVD failed");
    // Observability O = U₁·√S₁ (first `order` singular values).
    let obs = Matrix::from_fn(rows, order, |i, j| dec.u.get(i, j) * dec.sigma[j].max(0.0).sqrt());
    // A from the shift-invariance of O: O_up · A = O_down (least squares).
    let up_rows = rows - n_ch;
    let o_up = Matrix::from_fn(up_rows, order, |i, j| obs.get(i, j));
    let o_dn = Matrix::from_fn(up_rows, order, |i, j| obs.get(i + n_ch, j));
    // Normal equations.
    let otu = o_up.transpose().mul(&o_up).expect("dims");
    let otd = o_up.transpose().mul(&o_dn).expect("dims");
    let lu = lu_decompose(&otu).expect("SSI normal equations singular");
    let mut a = Matrix::zeros(order, order);
    for c in 0..order {
        let col: Vec<f64> = (0..order).map(|r| otd.get(r, c)).collect();
        let sol = lu.solve(&col).expect("solve failed");
        for (r, &v) in sol.iter().enumerate() {
            a.set(r, c, v);
        }
    }
    let eigs = eigenvalues_general(&a, 100).expect("SSI eigenvalues failed");
    let mut out = Vec::new();
    for l in eigs {
        if l.im <= 1e-12 || l.norm() >= 1.0 || l.norm() < 1e-6 {
            continue; // keep one of each conjugate pair, stable only
        }
        // Continuous-time pole: s = ln(λ)·fs.
        let s_re = l.norm().ln() * fs;
        let s_im = l.arg() * fs;
        let wn = (s_re * s_re + s_im * s_im).sqrt();
        if wn < 1e-9 {
            continue;
        }
        out.push((wn / TWO_PI, -s_re / wn));
    }
    out.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Shock response spectrum: peak absolute SDOF displacement response at
/// each requested natural frequency (Hz) for a base acceleration pulse.
#[must_use]
pub fn shock_response_spectrum(
    accel: &[f64],
    dt: f64,
    freqs: &[f64],
    zeta: f64,
) -> Vec<f64> {
    freqs
        .iter()
        .map(|&f| sdof_peak_displacement(accel, dt, TWO_PI * f, zeta))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_fem_bar_converges_to_clamped_free() {
        // Continuum: f_n = (2n−1)c/(4L).
        let (length, area, young, rho) = (2.0_f64, 1e-4_f64, 2e11_f64, 7800.0_f64);
        let c: f64 = (young / rho).sqrt();
        let model = ModalModel::from_fem_1d_bar(40, length, area, young, rho);
        let (freqs, shapes) = model.modes();
        for nmode in 1..=3 {
            let exact = (2.0 * nmode as f64 - 1.0) * c / (4.0 * length) * TWO_PI;
            assert!(
                (freqs[nmode - 1] - exact).abs() / exact < 0.005,
                "mode {nmode}: {} vs {exact}",
                freqs[nmode - 1]
            );
        }
        // Mass-orthonormal shapes.
        let n = 40;
        let mut dot = 0.0;
        for i in 0..n {
            for j in 0..n {
                dot += shapes.get(i, 0) * model.m.get(i, j) * shapes.get(j, 0);
            }
        }
        assert!(approx(dot, 1.0, 1e-8));
    }

    #[test]
    fn test_fem_beam_matches_cantilever_theory() {
        let (length, young, i_area, rho, area) = (1.0, 2e11, 1e-8, 7800.0, 1e-4);
        let model = ModalModel::from_fem_beam(20, length, young, i_area, rho, area);
        let (freqs, _) = model.modes();
        let coef = (young * i_area / (rho * area)).sqrt() / (length * length);
        let exact1 = 1.875_104_068_711_961_f64.powi(2) * coef;
        let exact2 = 4.694_091_132_974_175_f64.powi(2) * coef;
        assert!((freqs[0] - exact1).abs() / exact1 < 0.001, "{} vs {exact1}", freqs[0]);
        assert!((freqs[1] - exact2).abs() / exact2 < 0.005, "{} vs {exact2}", freqs[1]);
    }

    #[test]
    fn test_rayleigh_damping_hits_ratios() {
        let mut model = ModalModel::from_lumped(
            &[1.0, 1.0],
            &[(0, 0, 400.0), (0, 1, 100.0), (1, 1, 900.0)],
            &[],
        );
        let (freqs, _) = model.modes();
        let (f1, f2) = (freqs[0] / TWO_PI, freqs[1] / TWO_PI);
        model.rayleigh_from_ratios(0.02, f1, 0.05, f2);
        let zetas = model.modal_damping_ratios();
        // Damped modes give ratios matching the targets.
        assert!(approx(zetas[0], 0.02, 1e-3), "{zetas:?}");
        assert!(approx(zetas[zetas.len() - 1], 0.05, 1e-3), "{zetas:?}");
    }

    #[test]
    fn test_damped_modes_and_frf_peak() {
        let mut model = ModalModel::from_lumped(&[1.0], &[(0, 0, 100.0)], &[]);
        model.rayleigh_damping(0.0, 0.002); // ζ = βω/2 = 0.01
        let dm = model.damped_modes();
        let (lambda, shape) = &dm[0];
        assert!(approx(lambda.norm(), 10.0, 1e-6));
        assert!(approx(-lambda.re / lambda.norm(), 0.01, 1e-6));
        // (state vector is velocity-dominated at |λ| = 10: x-part ≈ 1/√(1+|λ|²))
        assert!(shape[0].norm() > 0.05);
        // FRF magnitude peaks near ω0 with |H| ≈ 1/(2ζk) = 1/(0.02·100).
        let h_res = model.frf(0, 0, 10.0).norm();
        assert!(approx(h_res, 0.5, 0.01), "{h_res}");
        let h_dc = model.frf(0, 0, 0.01).norm();
        assert!(approx(h_dc, 0.01, 1e-4));
        let fm = model.frf_matrix(10.0);
        assert!(approx(fm.get(0, 0), h_res, 1e-9));
    }

    #[test]
    fn test_newmark_energy_conservation_and_accuracy() {
        // Undamped SDOF: β=1/4, γ=1/2 conserves energy.
        let model = ModalModel::from_lumped(&[1.0], &[(0, 0, 25.0)], &[]);
        let hist = model.newmark_beta(&|_| vec![0.0], &[1.0], &[0.0], 20.0, 0.01, 0.25, 0.5);
        let last_amp = hist[hist.len() - 200..]
            .iter()
            .map(|x| x[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(approx(last_amp, 1.0, 1e-3), "amplitude drift {last_amp}");
        // Matches the exact cosine at a checkpoint.
        let idx = 500; // t = 5
        let exact = (5.0 * 5.0_f64).cos();
        assert!(approx(hist[idx][0], exact, 5e-3), "{} vs {exact}", hist[idx][0]);
        // HHT with α = −0.1 stays close but damps slightly.
        let hht = model.hht_alpha(&|_| vec![0.0], &[1.0], &[0.0], 20.0, 0.01, -0.1);
        let hht_amp = hht[hht.len() - 200..].iter().map(|x| x[0].abs()).fold(0.0_f64, f64::max);
        assert!(hht_amp <= 1.0 + 1e-9 && hht_amp > 0.8, "HHT amplitude {hht_amp}");
    }

    #[test]
    fn test_modal_truncation_and_guyan() {
        let model = ModalModel::from_lumped(
            &[1.0, 2.0, 1.5],
            &[(0, 0, 500.0), (0, 1, 200.0), (1, 2, 300.0), (2, 2, 100.0)],
            &[],
        );
        let (freqs, _) = model.modes();
        let red = model.modal_truncation(2);
        let (rf, _) = red.modes();
        assert!(approx(rf[0], freqs[0], 1e-9) && approx(rf[1], freqs[1], 1e-9));
        // Guyan keeps the fundamental within a few percent.
        let guy = model.guyan_reduction(&[0, 2]);
        let (gf, _) = guy.modes();
        assert!((gf[0] - freqs[0]).abs() / freqs[0] < 0.1, "{} vs {}", gf[0], freqs[0]);
        assert!(gf[0] >= freqs[0] - 1e-9); // static condensation stiffens
    }

    #[test]
    fn test_mac_identity_and_orthogonality() {
        let model = ModalModel::from_fem_1d_bar(10, 1.0, 1e-4, 2e11, 7800.0);
        let (_, shapes) = model.modes();
        let phi0: Vec<f64> = (0..10).map(|r| shapes.get(r, 0)).collect();
        let phi1: Vec<f64> = (0..10).map(|r| shapes.get(r, 1)).collect();
        assert!(approx(ModalModel::mac(&phi0, &phi0), 1.0, 1e-12));
        assert!(ModalModel::mac(&phi0, &phi1) < 0.01);
    }

    #[test]
    fn test_response_and_shock_spectra() {
        // Half-sine base pulse: SRS peaks near the pulse's own frequency
        // scale and falls off for very stiff systems.
        let dt = 0.001;
        let t_pulse = 0.05;
        let accel: Vec<f64> = (0..2000)
            .map(|i| {
                let t = i as f64 * dt;
                if t < t_pulse {
                    9.81 * (PI * t / t_pulse).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let freqs = [1.0, 5.0, 10.0, 50.0, 200.0];
        let srs = shock_response_spectrum(&accel, dt, &freqs, 0.05);
        // Displacement SRS decreases with stiffness at the high end.
        assert!(srs[4] < srs[2]);
        assert!(srs.iter().all(|&v| v > 0.0));
        let model = ModalModel::from_lumped(&[1.0], &[(0, 0, (TWO_PI * 10.0) * (TWO_PI * 10.0))], &[]);
        let rs = model.response_spectrum(&accel, dt, 0.05);
        assert!(approx(rs[0].0, 10.0, 1e-6));
        assert!(approx(rs[0].1, srs[2], 1e-9));
    }

    #[test]
    fn test_campbell_and_margins() {
        let model = ModalModel::from_lumped(&[1.0], &[(0, 0, (TWO_PI * 50.0) * (TWO_PI * 50.0))], &[]);
        let crit = model.critical_speeds(&[1.0, 2.0]);
        assert!(approx(crit[0], 1500.0, 1e-6)); // order 2: 50 Hz at 1500 rpm
        assert!(approx(crit[1], 3000.0, 1e-6));
        let cd = model.campbell_diagram((0.0, 6000.0), 5, &[1.0]);
        assert_eq!(cd.len(), 5);
        assert!(approx(cd[0].1[0], 50.0, 1e-9));
        let margins = model.resonance_margins(&[45.0, 100.0]);
        assert!(approx(margins[0], 0.1, 1e-9));
        assert!(approx(margins[1], 1.0, 1e-9));
    }

    #[test]
    fn test_peak_picking_and_circle_fit() {
        // Synthetic SDOF receptance.
        let (f0, zeta) = (35.0, 0.02);
        let w0 = TWO_PI * f0;
        let freqs: Vec<f64> = (0..2000).map(|i| 10.0 + 50.0 * i as f64 / 2000.0).collect();
        let frf: Vec<Complex> = freqs
            .iter()
            .map(|&f| {
                let w = TWO_PI * f;
                Complex::new(1.0, 0.0)
                    / Complex::new(w0 * w0 - w * w, 2.0 * zeta * w0 * w)
            })
            .collect();
        let picked = experimental_modal_peak_picking(&frf, &freqs);
        assert_eq!(picked.len(), 1, "{picked:?}");
        assert!(approx(picked[0].0, f0, 0.2));
        assert!((picked[0].1 - zeta).abs() / zeta < 0.05, "ζ {}", picked[0].1);
        let (fc, zc) = circle_fit(&frf, &freqs, 60);
        assert!(approx(fc, f0, 0.2), "circle f0 {fc}");
        assert!((zc - zeta).abs() / zeta < 0.1, "circle ζ {zc}");
    }

    #[test]
    fn test_ods_and_ssi() {
        // Two-mode free decay observed at two dofs.
        let model = ModalModel::from_lumped(
            &[1.0, 1.0],
            &[(0, 0, 400.0), (0, 1, 150.0), (1, 1, 900.0)],
            &[],
        );
        let mut damped = model.clone();
        damped.rayleigh_damping(0.1, 1e-4);
        let fs = 200.0;
        let hist = damped.newmark_beta(&|_| vec![0.0, 0.0], &[1.0, -0.4], &[0.0, 0.0], 40.0, 1.0 / fs, 0.25, 0.5);
        let ch0: Vec<f64> = hist.iter().map(|x| x[0]).collect();
        let ch1: Vec<f64> = hist.iter().map(|x| x[1]).collect();
        let (freqs, _) = model.modes();
        // ODS at the first natural frequency has both dofs responding.
        let ods = operational_deflection_shape(&[ch0.clone(), ch1.clone()], freqs[0], fs);
        assert!(ods[0].norm() > 0.0 && ods[1].norm() > 0.0);
        // SSI recovers both modal frequencies within 2%.
        let ident = stochastic_subspace_identification(&[ch0, ch1], fs, 4);
        assert!(!ident.is_empty(), "no stable poles found");
        for target in [freqs[0] / TWO_PI, freqs[1] / TWO_PI] {
            let hit = ident.iter().any(|&(f, z)| (f - target).abs() / target < 0.02 && z >= 0.0);
            assert!(hit, "missing {target} Hz in {ident:?}");
        }
    }
}
