//! 1D/2D compressible Euler equations: exact and approximate Riemann
//! solvers (HLL, HLLC, Roe with entropy fix, Rusanov, AUSM+), MUSCL
//! finite-volume drivers, classic shock-tube problems, and gas-dynamic
//! shock/expansion relations.

const PI: f64 = crate::math::constants::PI;

/// Primitive state (density, velocity, pressure).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prim {
    pub rho: f64,
    pub u: f64,
    pub p: f64,
}

/// Conserved state (density, momentum, total energy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cons {
    pub rho: f64,
    pub mom: f64,
    pub e: f64,
}

/// Primitive → conserved.
#[must_use]
pub fn prim_to_cons(p: Prim, gamma: f64) -> Cons {
    Cons {
        rho: p.rho,
        mom: p.rho * p.u,
        e: p.p / (gamma - 1.0) + 0.5 * p.rho * p.u * p.u,
    }
}

/// Conserved → primitive.
#[must_use]
pub fn cons_to_prim(c: Cons, gamma: f64) -> Prim {
    let rho = c.rho.max(1e-14);
    let u = c.mom / rho;
    Prim {
        rho,
        u,
        p: ((c.e - 0.5 * rho * u * u) * (gamma - 1.0)).max(1e-14),
    }
}

/// Physical flux F(U).
#[must_use]
pub fn flux(c: Cons, gamma: f64) -> Cons {
    let p = cons_to_prim(c, gamma);
    Cons {
        rho: c.mom,
        mom: c.mom * p.u + p.p,
        e: p.u * (c.e + p.p),
    }
}

/// Speed of sound √(γp/ρ).
#[must_use]
pub fn sound_speed(p: Prim, gamma: f64) -> f64 {
    (gamma * p.p / p.rho.max(1e-14)).sqrt()
}

fn pressure_function(p: f64, k: Prim, gamma: f64) -> (f64, f64) {
    // Toro's f_K(p) and its derivative.
    let a = sound_speed(k, gamma);
    if p > k.p {
        // Shock branch.
        let ak = 2.0 / ((gamma + 1.0) * k.rho);
        let bk = (gamma - 1.0) / (gamma + 1.0) * k.p;
        let sq = (ak / (p + bk)).sqrt();
        let f = (p - k.p) * sq;
        let df = sq * (1.0 - 0.5 * (p - k.p) / (p + bk));
        (f, df)
    } else {
        // Rarefaction branch.
        let pr = (p / k.p).powf((gamma - 1.0) / (2.0 * gamma));
        let f = 2.0 * a / (gamma - 1.0) * (pr - 1.0);
        let df = 1.0 / (k.rho * a) * (p / k.p).powf(-(gamma + 1.0) / (2.0 * gamma));
        (f, df)
    }
}

/// Star-region pressure and velocity of the exact Riemann problem
/// (Newton iteration, Toro ch. 4).
#[must_use]
pub fn riemann_exact_star(l: Prim, r: Prim, gamma: f64) -> (f64, f64) {
    let du = r.u - l.u;
    // PVRS initial guess, clipped positive.
    let a_l = sound_speed(l, gamma);
    let a_r = sound_speed(r, gamma);
    let p_guess = (0.5 * (l.p + r.p) - 0.125 * du * (l.rho + r.rho) * (a_l + a_r)).max(1e-8);
    let mut p = p_guess;
    for _ in 0..60 {
        let (fl, dfl) = pressure_function(p, l, gamma);
        let (fr, dfr) = pressure_function(p, r, gamma);
        let f = fl + fr + du;
        let df = dfl + dfr;
        let step = f / df.max(1e-300);
        let new_p = (p - step).max(1e-10);
        if (new_p - p).abs() / p.max(1e-10) < 1e-12 {
            p = new_p;
            break;
        }
        p = new_p;
    }
    let (fl, _) = pressure_function(p, l, gamma);
    let (fr, _) = pressure_function(p, r, gamma);
    let u = 0.5 * (l.u + r.u) + 0.5 * (fr - fl);
    (p, u)
}

/// Sample the exact Riemann solution at similarity coordinate ξ = x/t.
#[must_use]
pub fn riemann_exact(l: Prim, r: Prim, gamma: f64, x_over_t: f64) -> Prim {
    let (p_star, u_star) = riemann_exact_star(l, r, gamma);
    let xi = x_over_t;
    let g1 = (gamma - 1.0) / (gamma + 1.0);
    if xi <= u_star {
        // Left of the contact.
        let a_l = sound_speed(l, gamma);
        if p_star > l.p {
            // Left shock.
            let s_l = l.u - a_l * ((gamma + 1.0) / (2.0 * gamma) * p_star / l.p
                + (gamma - 1.0) / (2.0 * gamma))
                .sqrt();
            if xi <= s_l {
                l
            } else {
                let rho = l.rho * ((p_star / l.p + g1) / (g1 * p_star / l.p + 1.0));
                Prim { rho, u: u_star, p: p_star }
            }
        } else {
            // Left rarefaction.
            let rho_star = l.rho * (p_star / l.p).powf(1.0 / gamma);
            let a_star = sound_speed(Prim { rho: rho_star, u: u_star, p: p_star }, gamma);
            let head = l.u - a_l;
            let tail = u_star - a_star;
            if xi <= head {
                l
            } else if xi >= tail {
                Prim { rho: rho_star, u: u_star, p: p_star }
            } else {
                // Inside the fan.
                let u = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * l.u + xi);
                let a = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * (l.u - xi));
                let rho = l.rho * (a / a_l).powf(2.0 / (gamma - 1.0));
                let p = l.p * (a / a_l).powf(2.0 * gamma / (gamma - 1.0));
                Prim { rho, u, p }
            }
        }
    } else {
        // Right of the contact (mirror).
        let a_r = sound_speed(r, gamma);
        if p_star > r.p {
            let s_r = r.u + a_r * ((gamma + 1.0) / (2.0 * gamma) * p_star / r.p
                + (gamma - 1.0) / (2.0 * gamma))
                .sqrt();
            if xi >= s_r {
                r
            } else {
                let rho = r.rho * ((p_star / r.p + g1) / (g1 * p_star / r.p + 1.0));
                Prim { rho, u: u_star, p: p_star }
            }
        } else {
            let rho_star = r.rho * (p_star / r.p).powf(1.0 / gamma);
            let a_star = sound_speed(Prim { rho: rho_star, u: u_star, p: p_star }, gamma);
            let head = r.u + a_r;
            let tail = u_star + a_star;
            if xi >= head {
                r
            } else if xi <= tail {
                Prim { rho: rho_star, u: u_star, p: p_star }
            } else {
                let u = 2.0 / (gamma + 1.0) * (-a_r + (gamma - 1.0) / 2.0 * r.u + xi);
                let a = 2.0 / (gamma + 1.0) * (a_r - (gamma - 1.0) / 2.0 * (r.u - xi));
                let rho = r.rho * (a / a_r).powf(2.0 / (gamma - 1.0));
                let p = r.p * (a / a_r).powf(2.0 * gamma / (gamma - 1.0));
                Prim { rho, u, p }
            }
        }
    }
}

/// Einfeldt (Roe-averaged) wave speed bounds (S_L, S_R).
#[must_use]
pub fn wave_speeds_einfeldt(l: Cons, r: Cons, gamma: f64) -> (f64, f64) {
    let pl = cons_to_prim(l, gamma);
    let pr = cons_to_prim(r, gamma);
    let al = sound_speed(pl, gamma);
    let ar = sound_speed(pr, gamma);
    let sl = pl.rho.sqrt();
    let sr = pr.rho.sqrt();
    let u_roe = (sl * pl.u + sr * pr.u) / (sl + sr);
    let hl = (l.e + pl.p) / pl.rho;
    let hr = (r.e + pr.p) / pr.rho;
    let h_roe = (sl * hl + sr * hr) / (sl + sr);
    let a_roe = ((gamma - 1.0) * (h_roe - 0.5 * u_roe * u_roe)).max(1e-14).sqrt();
    ((pl.u - al).min(u_roe - a_roe), (pr.u + ar).max(u_roe + a_roe))
}

/// HLL flux.
#[must_use]
pub fn flux_hll(l: Cons, r: Cons, gamma: f64) -> Cons {
    let (sl, sr) = wave_speeds_einfeldt(l, r, gamma);
    let fl = flux(l, gamma);
    let fr = flux(r, gamma);
    if sl >= 0.0 {
        fl
    } else if sr <= 0.0 {
        fr
    } else {
        let inv = 1.0 / (sr - sl);
        Cons {
            rho: (sr * fl.rho - sl * fr.rho + sl * sr * (r.rho - l.rho)) * inv,
            mom: (sr * fl.mom - sl * fr.mom + sl * sr * (r.mom - l.mom)) * inv,
            e: (sr * fl.e - sl * fr.e + sl * sr * (r.e - l.e)) * inv,
        }
    }
}

/// HLLC flux (restores the contact wave).
#[must_use]
pub fn flux_hllc(l: Cons, r: Cons, gamma: f64) -> Cons {
    let pl = cons_to_prim(l, gamma);
    let pr = cons_to_prim(r, gamma);
    let (sl, sr) = wave_speeds_einfeldt(l, r, gamma);
    let fl = flux(l, gamma);
    let fr = flux(r, gamma);
    if sl >= 0.0 {
        return fl;
    }
    if sr <= 0.0 {
        return fr;
    }
    let s_star = (pr.p - pl.p + pl.rho * pl.u * (sl - pl.u) - pr.rho * pr.u * (sr - pr.u))
        / (pl.rho * (sl - pl.u) - pr.rho * (sr - pr.u));
    let star = |k: Cons, pk: Prim, s_k: f64| -> Cons {
        let coef = pk.rho * (s_k - pk.u) / (s_k - s_star);
        Cons {
            rho: coef,
            mom: coef * s_star,
            e: coef
                * (k.e / pk.rho
                    + (s_star - pk.u) * (s_star + pk.p / (pk.rho * (s_k - pk.u)))),
        }
    };
    if s_star >= 0.0 {
        let ul_star = star(l, pl, sl);
        Cons {
            rho: fl.rho + sl * (ul_star.rho - l.rho),
            mom: fl.mom + sl * (ul_star.mom - l.mom),
            e: fl.e + sl * (ul_star.e - l.e),
        }
    } else {
        let ur_star = star(r, pr, sr);
        Cons {
            rho: fr.rho + sr * (ur_star.rho - r.rho),
            mom: fr.mom + sr * (ur_star.mom - r.mom),
            e: fr.e + sr * (ur_star.e - r.e),
        }
    }
}

/// Roe flux with a Harten entropy fix.
#[must_use]
pub fn flux_roe(l: Cons, r: Cons, gamma: f64) -> Cons {
    let pl = cons_to_prim(l, gamma);
    let pr = cons_to_prim(r, gamma);
    let sl = pl.rho.sqrt();
    let sr = pr.rho.sqrt();
    let u = (sl * pl.u + sr * pr.u) / (sl + sr);
    let hl = (l.e + pl.p) / pl.rho;
    let hr = (r.e + pr.p) / pr.rho;
    let h = (sl * hl + sr * hr) / (sl + sr);
    let a = ((gamma - 1.0) * (h - 0.5 * u * u)).max(1e-14).sqrt();
    let drho = pr.rho - pl.rho;
    let du = pr.u - pl.u;
    let dp = pr.p - pl.p;
    // Wave strengths with the Roe-averaged density ρ̃ = √(ρL ρR).
    let rho_roe = sl * sr;
    let alpha1 = (dp - rho_roe * a * du) / (2.0 * a * a);
    let alpha2 = drho - dp / (a * a);
    let alpha3 = (dp + rho_roe * a * du) / (2.0 * a * a);
    let lambdas = [u - a, u, u + a];
    // Entropy fix.
    let eps = 0.1 * a;
    let fix = |lam: f64| -> f64 {
        if lam.abs() < eps { (lam * lam + eps * eps) / (2.0 * eps) } else { lam.abs() }
    };
    let k1 = [1.0, u - a, h - u * a];
    let k2 = [1.0, u, 0.5 * u * u];
    let k3 = [1.0, u + a, h + u * a];
    let fl = flux(l, gamma);
    let fr = flux(r, gamma);
    let mut out = [
        0.5 * (fl.rho + fr.rho),
        0.5 * (fl.mom + fr.mom),
        0.5 * (fl.e + fr.e),
    ];
    for c in 0..3 {
        out[c] -= 0.5
            * (alpha1 * fix(lambdas[0]) * k1[c]
                + alpha2 * fix(lambdas[1]) * k2[c]
                + alpha3 * fix(lambdas[2]) * k3[c]);
    }
    Cons { rho: out[0], mom: out[1], e: out[2] }
}

/// Rusanov (local Lax-Friedrichs) flux.
#[must_use]
pub fn flux_rusanov(l: Cons, r: Cons, gamma: f64) -> Cons {
    let pl = cons_to_prim(l, gamma);
    let pr = cons_to_prim(r, gamma);
    let s = (pl.u.abs() + sound_speed(pl, gamma)).max(pr.u.abs() + sound_speed(pr, gamma));
    let fl = flux(l, gamma);
    let fr = flux(r, gamma);
    Cons {
        rho: 0.5 * (fl.rho + fr.rho) - 0.5 * s * (r.rho - l.rho),
        mom: 0.5 * (fl.mom + fr.mom) - 0.5 * s * (r.mom - l.mom),
        e: 0.5 * (fl.e + fr.e) - 0.5 * s * (r.e - l.e),
    }
}

/// AUSM+ flux (Liou 1996).
#[must_use]
pub fn flux_ausm_plus(l: Cons, r: Cons, gamma: f64) -> Cons {
    let pl = cons_to_prim(l, gamma);
    let pr = cons_to_prim(r, gamma);
    let a_half = 0.5 * (sound_speed(pl, gamma) + sound_speed(pr, gamma));
    let ml = pl.u / a_half;
    let mr = pr.u / a_half;
    let m4 = |m: f64, sign: f64| -> f64 {
        if m.abs() >= 1.0 {
            0.5 * (m + sign * m.abs())
        } else {
            sign * 0.25 * (m + sign).powi(2) + sign * 0.125 * (m * m - 1.0).powi(2)
        }
    };
    let p5 = |m: f64, sign: f64| -> f64 {
        if m.abs() >= 1.0 {
            0.5 * (1.0 + sign * m.signum())
        } else {
            0.25 * (m + sign).powi(2) * (2.0 - sign * m)
                + sign * 3.0 / 16.0 * m * (m * m - 1.0).powi(2)
        }
    };
    let m_half = m4(ml, 1.0) + m4(mr, -1.0);
    let p_half = p5(ml, 1.0) * pl.p + p5(mr, -1.0) * pr.p;
    let hl = (l.e + pl.p) / pl.rho;
    let hr = (r.e + pr.p) / pr.rho;
    let mdot = a_half * m_half * if m_half > 0.0 { pl.rho } else { pr.rho };
    let (rho_u, h_u) = if m_half > 0.0 { (pl.u, hl) } else { (pr.u, hr) };
    Cons {
        rho: mdot,
        mom: mdot * rho_u + p_half,
        e: mdot * h_u,
    }
}

/// Numerical flux selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxKind {
    Exact,
    Hll,
    Hllc,
    Roe,
    Rusanov,
    AusmPlus,
}

fn numerical_flux(kind: FluxKind, l: Cons, r: Cons, gamma: f64) -> Cons {
    match kind {
        FluxKind::Exact => {
            let s = riemann_exact(cons_to_prim(l, gamma), cons_to_prim(r, gamma), gamma, 0.0);
            flux(prim_to_cons(s, gamma), gamma)
        }
        FluxKind::Hll => flux_hll(l, r, gamma),
        FluxKind::Hllc => flux_hllc(l, r, gamma),
        FluxKind::Roe => flux_roe(l, r, gamma),
        FluxKind::Rusanov => flux_rusanov(l, r, gamma),
        FluxKind::AusmPlus => flux_ausm_plus(l, r, gamma),
    }
}

/// Boundary condition for [`Euler1D`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerBc {
    Transmissive,
    Reflective,
    Periodic,
}

/// 1D finite-volume Euler solver (order 1, or 2 with minmod MUSCL).
pub struct Euler1D {
    pub cells: Vec<Cons>,
    pub dx: f64,
    pub gamma: f64,
    pub flux: FluxKind,
    pub order: usize,
    pub bc: EulerBc,
    pub time: f64,
}

impl Euler1D {
    /// Uniform quiescent gas.
    #[must_use]
    pub fn new(n: usize, dx: f64, gamma: f64) -> Self {
        Self {
            cells: vec![prim_to_cons(Prim { rho: 1.0, u: 0.0, p: 1.0 }, gamma); n],
            dx,
            gamma,
            flux: FluxKind::Hllc,
            order: 2,
            bc: EulerBc::Transmissive,
            time: 0.0,
        }
    }

    /// Set piecewise-constant left/right states split at fraction
    /// `x_split` of the domain.
    pub fn set_riemann_problem(&mut self, l: Prim, r: Prim, x_split: f64) {
        let n = self.cells.len();
        for (i, c) in self.cells.iter_mut().enumerate() {
            let x = (i as f64 + 0.5) / n as f64;
            *c = prim_to_cons(if x < x_split { l } else { r }, self.gamma);
        }
        self.time = 0.0;
    }

    fn ghost(&self, i: i64) -> Cons {
        let n = self.cells.len() as i64;
        match self.bc {
            EulerBc::Transmissive => self.cells[i.clamp(0, n - 1) as usize],
            EulerBc::Periodic => self.cells[i.rem_euclid(n) as usize],
            EulerBc::Reflective => {
                if i < 0 {
                    let mut c = self.cells[(-i - 1) as usize];
                    c.mom = -c.mom;
                    c
                } else if i >= n {
                    let mut c = self.cells[(2 * n - 1 - i) as usize];
                    c.mom = -c.mom;
                    c
                } else {
                    self.cells[i as usize]
                }
            }
        }
    }

    /// One step at the given CFL number; returns the dt used.
    pub fn step(&mut self, cfl: f64) -> f64 {
        let n = self.cells.len();
        let mut smax = 1e-12_f64;
        for c in &self.cells {
            let p = cons_to_prim(*c, self.gamma);
            smax = smax.max(p.u.abs() + sound_speed(p, self.gamma));
        }
        let dt = cfl * self.dx / smax;
        // Reconstruct primitive states at faces (minmod slopes).
        let minmod = |a: f64, b: f64| -> f64 {
            if a * b <= 0.0 {
                0.0
            } else if a.abs() < b.abs() {
                a
            } else {
                b
            }
        };
        let prim_at = |c: Cons| cons_to_prim(c, self.gamma);
        // Face i+1/2 for i in -1..n-1 → n+1 faces.
        let mut fluxes = Vec::with_capacity(n + 1);
        for f in -1..(n as i64) {
            let (ul, ur) = if self.order >= 2 {
                let get = |i: i64| prim_at(self.ghost(i));
                let pl_m = get(f - 1);
                let pl_0 = get(f);
                let pl_p = get(f + 1);
                let pr_0 = get(f + 1);
                let pr_p = get(f + 2);
                let slope = |m: Prim, c: Prim, p: Prim| -> (f64, f64, f64) {
                    (
                        minmod(c.rho - m.rho, p.rho - c.rho),
                        minmod(c.u - m.u, p.u - c.u),
                        minmod(c.p - m.p, p.p - c.p),
                    )
                };
                let (sr, su, sp) = slope(pl_m, pl_0, pl_p);
                let left = Prim {
                    rho: (pl_0.rho + 0.5 * sr).max(1e-10),
                    u: pl_0.u + 0.5 * su,
                    p: (pl_0.p + 0.5 * sp).max(1e-10),
                };
                let (sr2, su2, sp2) = slope(pl_0, pr_0, pr_p);
                let right = Prim {
                    rho: (pr_0.rho - 0.5 * sr2).max(1e-10),
                    u: pr_0.u - 0.5 * su2,
                    p: (pr_0.p - 0.5 * sp2).max(1e-10),
                };
                (prim_to_cons(left, self.gamma), prim_to_cons(right, self.gamma))
            } else {
                (self.ghost(f), self.ghost(f + 1))
            };
            fluxes.push(numerical_flux(self.flux, ul, ur, self.gamma));
        }
        let lam = dt / self.dx;
        for i in 0..n {
            let fl = fluxes[i];
            let fr = fluxes[i + 1];
            self.cells[i].rho -= lam * (fr.rho - fl.rho);
            self.cells[i].mom -= lam * (fr.mom - fl.mom);
            self.cells[i].e -= lam * (fr.e - fl.e);
        }
        self.time += dt;
        dt
    }

    /// Step until time `t`.
    ///
    /// The CFL number follows the stability limit of the configured
    /// reconstruction: the first-order update is TVD up to CFL 1, while
    /// the MUSCL reconstruction advanced with a single forward-Euler
    /// update is only TVD for CFL ≤ 1/2 (Harten's condition; see Toro,
    /// *Riemann Solvers and Numerical Methods*, ch. 13). Running the
    /// second-order scheme at CFL 0.9 does not converge under grid
    /// refinement.
    pub fn run_until(&mut self, t: f64) {
        let cfl = if self.order >= 2 { 0.45 } else { 0.9 };
        while self.time < t - 1e-12 {
            let remaining = t - self.time;
            let dt = self.step(cfl);
            if dt > remaining {
                // Undo overshoot proportionally: cheap fix — step was
                // already applied; accept slight overshoot instead.
                break;
            }
        }
    }

    /// Primitive states of all cells.
    #[must_use]
    pub fn primitives(&self) -> Vec<Prim> {
        self.cells.iter().map(|c| cons_to_prim(*c, self.gamma)).collect()
    }

    /// Total mass ∫ρ dx.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.cells.iter().map(|c| c.rho).sum::<f64>() * self.dx
    }

    /// Total energy ∫E dx.
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.cells.iter().map(|c| c.e).sum::<f64>() * self.dx
    }

    /// Position (domain fraction) of the steepest density gradient.
    #[must_use]
    pub fn shock_position(&self) -> Option<f64> {
        let n = self.cells.len();
        if n < 3 {
            return None;
        }
        let (mut best, mut best_g) = (0, 0.0);
        for i in 0..n - 1 {
            let g = (self.cells[i + 1].rho - self.cells[i].rho).abs();
            if g > best_g {
                best_g = g;
                best = i;
            }
        }
        if best_g < 1e-9 {
            None
        } else {
            Some((best as f64 + 1.0) / n as f64)
        }
    }
}

/// Sod's shock tube on n cells (γ = 1.4, unit domain).
#[must_use]
pub fn sod_shock_tube(n: usize) -> Euler1D {
    let mut e = Euler1D::new(n, 1.0 / n as f64, 1.4);
    e.set_riemann_problem(
        Prim { rho: 1.0, u: 0.0, p: 1.0 },
        Prim { rho: 0.125, u: 0.0, p: 0.1 },
        0.5,
    );
    e
}

/// Lax's problem.
#[must_use]
pub fn lax_problem(n: usize) -> Euler1D {
    let mut e = Euler1D::new(n, 1.0 / n as f64, 1.4);
    e.set_riemann_problem(
        Prim { rho: 0.445, u: 0.698, p: 3.528 },
        Prim { rho: 0.5, u: 0.0, p: 0.571 },
        0.5,
    );
    e
}

/// Shu-Osher shock/entropy-wave interaction.
#[must_use]
pub fn shu_osher(n: usize) -> Euler1D {
    let mut e = Euler1D::new(n, 10.0 / n as f64, 1.4);
    for (i, c) in e.cells.iter_mut().enumerate() {
        let x = (i as f64 + 0.5) * 10.0 / n as f64 - 5.0;
        let prim = if x < -4.0 {
            Prim { rho: 3.857143, u: 2.629369, p: 10.33333 }
        } else {
            Prim { rho: 1.0 + 0.2 * (5.0 * x).sin(), u: 0.0, p: 1.0 }
        };
        *c = prim_to_cons(prim, 1.4);
    }
    e
}

/// Woodward-Colella interacting blast waves (reflective walls).
#[must_use]
pub fn blast_wave_woodward_colella(n: usize) -> Euler1D {
    let mut e = Euler1D::new(n, 1.0 / n as f64, 1.4);
    e.bc = EulerBc::Reflective;
    for (i, c) in e.cells.iter_mut().enumerate() {
        let x = (i as f64 + 0.5) / n as f64;
        let p = if x < 0.1 {
            1000.0
        } else if x > 0.9 {
            100.0
        } else {
            0.01
        };
        *c = prim_to_cons(Prim { rho: 1.0, u: 0.0, p }, 1.4);
    }
    e
}

/// Sedov point blast in 1D planar symmetry.
#[must_use]
pub fn sedov_1d(n: usize) -> Euler1D {
    let mut e = Euler1D::new(n, 1.0 / n as f64, 1.4);
    for (i, c) in e.cells.iter_mut().enumerate() {
        let center = i == n / 2;
        let p = if center { 1e4 / (1.0 / n as f64) } else { 1e-6 };
        *c = prim_to_cons(Prim { rho: 1.0, u: 0.0, p }, 1.4);
    }
    e
}

/// Exact Sod solution at (x, t) on the unit domain split at 0.5.
#[must_use]
pub fn sod_exact(x: f64, t: f64) -> Prim {
    let l = Prim { rho: 1.0, u: 0.0, p: 1.0 };
    let r = Prim { rho: 0.125, u: 0.0, p: 0.1 };
    if t <= 0.0 {
        return if x < 0.5 { l } else { r };
    }
    riemann_exact(l, r, 1.4, (x - 0.5) / t)
}

// --- 2D Euler ------------------------------------------------------------

/// 2D finite-volume Euler solver (dimensional splitting, MUSCL + HLLC).
pub struct Euler2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub gamma: f64,
    pub rho: Vec<f64>,
    pub momx: Vec<f64>,
    pub momy: Vec<f64>,
    pub e: Vec<f64>,
    pub solid: Vec<bool>,
    pub periodic: bool,
    /// Gravity in −y applied as a source term.
    pub gravity: f64,
    pub time: f64,
}

impl Euler2D {
    /// Uniform gas at rest.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64, gamma: f64) -> Self {
        let n = nx * ny;
        let e0 = 1.0 / (gamma - 1.0);
        Self {
            nx,
            ny,
            dx,
            gamma,
            rho: vec![1.0; n],
            momx: vec![0.0; n],
            momy: vec![0.0; n],
            e: vec![e0; n],
            solid: vec![false; n],
            periodic: false,
            gravity: 0.0,
            time: 0.0,
        }
    }

    /// Set the primitive state of one cell.
    pub fn set_cell(&mut self, i: usize, j: usize, rho: f64, u: f64, v: f64, p: f64) {
        let c = j * self.nx + i;
        self.rho[c] = rho;
        self.momx[c] = rho * u;
        self.momy[c] = rho * v;
        self.e[c] = p / (self.gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    }

    fn sweep_x(&mut self, dt: f64) {
        // 4-component HLLC along x: state (ρ, ρu, ρv, E); the transverse
        // velocity is contact-advected in the star region.
        let (nx, ny) = (self.nx, self.ny);
        let gamma = self.gamma;
        let hllc4 = |sl_state: [f64; 4], sr_state: [f64; 4]| -> [f64; 4] {
            let prim = |q: [f64; 4]| -> (f64, f64, f64, f64) {
                let rho = q[0].max(1e-14);
                let u = q[1] / rho;
                let v = q[2] / rho;
                let p = ((q[3] - 0.5 * rho * (u * u + v * v)) * (gamma - 1.0)).max(1e-14);
                (rho, u, v, p)
            };
            let phys = |q: [f64; 4]| -> [f64; 4] {
                let (rho, u, v, p) = prim(q);
                [rho * u, rho * u * u + p, rho * u * v, u * (q[3] + p)]
            };
            let (rl, ul, _vl, pl) = prim(sl_state);
            let (rr, ur, _vr, pr) = prim(sr_state);
            let al = (gamma * pl / rl).sqrt();
            let ar = (gamma * pr / rr).sqrt();
            // Roe-averaged bounds.
            let (wl, wr) = (rl.sqrt(), rr.sqrt());
            let u_roe = (wl * ul + wr * ur) / (wl + wr);
            let hl = (sl_state[3] + pl) / rl;
            let hr = (sr_state[3] + pr) / rr;
            let h_roe = (wl * hl + wr * hr) / (wl + wr);
            let v_roe = (wl * sl_state[2] / rl + wr * sr_state[2] / rr) / (wl + wr);
            let a_roe = ((gamma - 1.0)
                * (h_roe - 0.5 * (u_roe * u_roe + v_roe * v_roe)))
                .max(1e-14)
                .sqrt();
            let s_l = (ul - al).min(u_roe - a_roe);
            let s_r = (ur + ar).max(u_roe + a_roe);
            let fl = phys(sl_state);
            let fr = phys(sr_state);
            if s_l >= 0.0 {
                return fl;
            }
            if s_r <= 0.0 {
                return fr;
            }
            let s_star = (pr - pl + rl * ul * (s_l - ul) - rr * ur * (s_r - ur))
                / (rl * (s_l - ul) - rr * (s_r - ur));
            let star = |q: [f64; 4], s_k: f64| -> [f64; 4] {
                let (rho, u, v, p) = prim(q);
                let coef = rho * (s_k - u) / (s_k - s_star);
                [
                    coef,
                    coef * s_star,
                    coef * v,
                    coef * (q[3] / rho + (s_star - u) * (s_star + p / (rho * (s_k - u)))),
                ]
            };
            if s_star >= 0.0 {
                let us = star(sl_state, s_l);
                std::array::from_fn(|c| fl[c] + s_l * (us[c] - sl_state[c]))
            } else {
                let us = star(sr_state, s_r);
                std::array::from_fn(|c| fr[c] + s_r * (us[c] - sr_state[c]))
            }
        };
        let lam = dt / self.dx;
        for j in 0..ny {
            let get = |i: i64| -> [f64; 4] {
                let idx = if self.periodic {
                    j * nx + i.rem_euclid(nx as i64) as usize
                } else {
                    j * nx + i.clamp(0, nx as i64 - 1) as usize
                };
                [self.rho[idx], self.momx[idx], self.momy[idx], self.e[idx]]
            };
            // Minmod-limited MUSCL reconstruction per component.
            let minmod = |a: f64, b: f64| -> f64 {
                if a * b <= 0.0 {
                    0.0
                } else if a.abs() < b.abs() {
                    a
                } else {
                    b
                }
            };
            let mut fluxes = Vec::with_capacity(nx + 1);
            for f in -1..(nx as i64) {
                let qm = get(f - 1);
                let q0 = get(f);
                let qp = get(f + 1);
                let qpp = get(f + 2);
                let left: [f64; 4] = std::array::from_fn(|c| {
                    q0[c] + 0.5 * minmod(q0[c] - qm[c], qp[c] - q0[c])
                });
                let right: [f64; 4] = std::array::from_fn(|c| {
                    qp[c] - 0.5 * minmod(qp[c] - q0[c], qpp[c] - qp[c])
                });
                fluxes.push(hllc4(left, right));
            }
            for i in 0..nx {
                let c = j * nx + i;
                if self.solid[c] {
                    continue;
                }
                let (fl, fr) = (fluxes[i], fluxes[i + 1]);
                self.rho[c] -= lam * (fr[0] - fl[0]);
                self.momx[c] -= lam * (fr[1] - fl[1]);
                self.momy[c] -= lam * (fr[2] - fl[2]);
                self.e[c] -= lam * (fr[3] - fl[3]);
            }
        }
    }

    fn transpose_state(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        let t = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; nx * ny];
            for j in 0..ny {
                for i in 0..nx {
                    out[i * ny + j] = v[j * nx + i];
                }
            }
            out
        };
        self.rho = t(&self.rho);
        let mx = t(&self.momx);
        let my = t(&self.momy);
        self.momx = my;
        self.momy = mx;
        self.e = t(&self.e);
        let s = {
            let mut out = vec![false; nx * ny];
            for j in 0..ny {
                for i in 0..nx {
                    out[i * ny + j] = self.solid[j * nx + i];
                }
            }
            out
        };
        self.solid = s;
        std::mem::swap(&mut self.nx, &mut self.ny);
    }

    /// One dimensionally split step (x then y sweep); returns dt.
    pub fn step(&mut self, cfl: f64) -> f64 {
        let gamma = self.gamma;
        let mut smax = 1e-12_f64;
        for c in 0..self.rho.len() {
            let rho = self.rho[c].max(1e-14);
            let u = self.momx[c] / rho;
            let v = self.momy[c] / rho;
            let p = ((self.e[c] - 0.5 * rho * (u * u + v * v)) * (gamma - 1.0)).max(1e-14);
            let a = (gamma * p / rho).sqrt();
            smax = smax.max(u.abs().max(v.abs()) + a);
        }
        let dt = cfl * self.dx / smax;
        self.sweep_x(0.5 * dt);
        self.transpose_state();
        self.sweep_x(dt);
        self.transpose_state();
        self.sweep_x(0.5 * dt);
        if self.gravity != 0.0 {
            for c in 0..self.rho.len() {
                let v_old = self.momy[c] / self.rho[c].max(1e-14);
                self.momy[c] -= dt * self.gravity * self.rho[c];
                let v_new = self.momy[c] / self.rho[c].max(1e-14);
                self.e[c] += 0.5 * self.rho[c] * (v_new * v_new - v_old * v_old);
            }
        }
        self.time += dt;
        dt
    }

    /// Kelvin-Helmholtz double shear layer with a velocity perturbation.
    pub fn kelvin_helmholtz_init(&mut self) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        self.periodic = true;
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let y = (j as f64 + 0.5) * dx / (ny as f64 * dx);
                let inner = (0.25..0.75).contains(&y);
                let (rho, u) = if inner { (2.0, 0.5) } else { (1.0, -0.5) };
                let v = 0.01 * (2.0 * PI * 4.0 * x).sin();
                self.set_cell(i, j, rho, u, v, 2.5);
            }
        }
    }

    /// Rayleigh-Taylor: heavy over light in gravity `g`.
    pub fn rayleigh_taylor_init(&mut self, g: f64) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        self.gravity = g;
        let height = ny as f64 * dx;
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let y = (j as f64 + 0.5) * dx;
                let rho = if y > 0.5 * height { 2.0 } else { 1.0 };
                // Hydrostatic pressure.
                let p = 2.5 + rho * g * (height - y);
                let v = 0.01
                    * (2.0 * PI * x / (nx as f64 * dx)).cos()
                    * (-((y - 0.5 * height) / (0.05 * height)).powi(2)).exp();
                self.set_cell(i, j, rho, 0.0, v, p);
            }
        }
    }

    /// Mach-10 double Mach reflection initial wedge state (simplified,
    /// vertical shock at x = 1/6).
    pub fn double_mach_reflection_init(&mut self) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                if x < 1.0 / 6.0 {
                    self.set_cell(i, j, 8.0, 8.25, 0.0, 116.5);
                } else {
                    self.set_cell(i, j, 1.4, 0.0, 0.0, 1.0);
                }
            }
        }
    }

    /// Shock hitting a low-density bubble.
    pub fn shock_bubble_init(&mut self) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let height = ny as f64 * dx;
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let y = (j as f64 + 0.5) * dx;
                let in_bubble =
                    ((x - 0.5).powi(2) + (y - 0.5 * height).powi(2)).sqrt() < 0.15;
                if x < 0.2 {
                    // Post-shock state of a Mach 1.5 shock in air.
                    let post = rankine_hugoniot(1.0, 1.0, 1.5, 1.4);
                    self.set_cell(i, j, post.rho, post.u, 0.0, post.p);
                } else if in_bubble {
                    self.set_cell(i, j, 0.1, 0.0, 0.0, 1.0);
                } else {
                    self.set_cell(i, j, 1.0, 0.0, 0.0, 1.0);
                }
            }
        }
    }

    /// Numerical schlieren |∇ρ| normalized to [0, 1].
    #[must_use]
    pub fn schlieren(&self) -> Vec<f64> {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let at = |i: i64, j: i64| -> f64 {
            let i = i.clamp(0, nx as i64 - 1) as usize;
            let j = j.clamp(0, ny as i64 - 1) as usize;
            self.rho[j * nx + i]
        };
        let mut out = vec![0.0; nx * ny];
        let mut maxg = 1e-300_f64;
        for j in 0..ny as i64 {
            for i in 0..nx as i64 {
                let gx = (at(i + 1, j) - at(i - 1, j)) / (2.0 * dx);
                let gy = (at(i, j + 1) - at(i, j - 1)) / (2.0 * dx);
                let g = (gx * gx + gy * gy).sqrt();
                out[(j as usize) * nx + i as usize] = g;
                maxg = maxg.max(g);
            }
        }
        out.iter_mut().for_each(|v| *v /= maxg);
        out
    }
}

// --- Gas-dynamic relations -----------------------------------------------

/// Post-shock primitive state behind a normal shock of Mach `mach`
/// moving into gas at (p1, rho1) at rest (lab frame).
#[must_use]
pub fn rankine_hugoniot(p1: f64, rho1: f64, mach: f64, gamma: f64) -> Prim {
    let (p_ratio, rho_ratio, _, _) = normal_shock_relations(mach, gamma);
    let a1 = (gamma * p1 / rho1).sqrt();
    let ws = mach * a1; // shock speed into still gas
    let rho2 = rho1 * rho_ratio;
    // Mass conservation across the shock in the shock frame.
    let u2 = ws * (1.0 - 1.0 / rho_ratio);
    Prim { rho: rho2, u: u2, p: p1 * p_ratio }
}

/// Normal shock relations: (p2/p1, ρ2/ρ1, T2/T1, M2).
#[must_use]
pub fn normal_shock_relations(mach: f64, gamma: f64) -> (f64, f64, f64, f64) {
    let m2 = mach * mach;
    let p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (m2 - 1.0);
    let rho_ratio = (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0);
    let t_ratio = p_ratio / rho_ratio;
    let m2_post = (((gamma - 1.0) * m2 + 2.0) / (2.0 * gamma * m2 - (gamma - 1.0))).sqrt();
    (p_ratio, rho_ratio, t_ratio, m2_post)
}

/// Oblique shock wave angles (weak, strong) in radians for a flow
/// deflection; `None` if the deflection exceeds the maximum attached
/// angle.
#[must_use]
pub fn oblique_shock_angle(mach: f64, deflection: f64, gamma: f64) -> Option<(f64, f64)> {
    let theta = deflection;
    // θ-β-M relation: tanθ = 2 cotβ (M² sin²β − 1)/(M²(γ + cos 2β) + 2).
    let f = |beta: f64| -> f64 {
        let m2 = mach * mach;
        let num = m2 * beta.sin().powi(2) - 1.0;
        (2.0 / beta.tan()) * num / (m2 * (gamma + (2.0 * beta).cos()) + 2.0) - theta.tan()
    };
    let mach_angle = (1.0 / mach).asin();
    // Find the maximum of θ(β) by scanning.
    let mut roots = Vec::new();
    let steps = 2000;
    let mut prev_b = mach_angle + 1e-6;
    let mut prev_v = f(prev_b);
    for k in 1..=steps {
        let b = mach_angle + (PI / 2.0 - mach_angle - 2e-6) * k as f64 / steps as f64;
        let v = f(b);
        if prev_v * v < 0.0 {
            let (mut lo, mut hi, mut flo) = (prev_b, b, prev_v);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let fm = f(mid);
                if flo * fm <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
        prev_b = b;
        prev_v = v;
    }
    match roots.len() {
        0 => None,
        1 => Some((roots[0], roots[0])),
        _ => Some((roots[0], roots[roots.len() - 1])),
    }
}

/// Prandtl-Meyer function ν(M) in radians.
#[must_use]
pub fn prandtl_meyer(mach: f64, gamma: f64) -> f64 {
    if mach <= 1.0 {
        return 0.0;
    }
    let gp = (gamma + 1.0) / (gamma - 1.0);
    gp.sqrt() * ((mach * mach - 1.0) / gp).sqrt().atan() - (mach * mach - 1.0).sqrt().atan()
}

/// Isentropic area ratio A/A* for a given Mach number.
#[must_use]
pub fn nozzle_area_ratio(mach: f64, gamma: f64) -> f64 {
    let g1 = (gamma + 1.0) / (2.0 * (gamma - 1.0));
    (1.0 / mach)
        * ((2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * mach * mach)).powf(g1)
}

/// Invert the area ratio for the subsonic or supersonic branch.
#[must_use]
pub fn nozzle_mach_from_area(ratio: f64, gamma: f64, supersonic: bool) -> f64 {
    let (mut lo, mut hi) = if supersonic { (1.0, 50.0) } else { (1e-6, 1.0) };
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let a = nozzle_area_ratio(mid, gamma);
        // A/A* decreases with M below 1 and increases above 1.
        let too_big = a > ratio;
        if supersonic == too_big {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Quasi-1D isentropic nozzle solution: `area(x)` on x ∈ [0, 1] with a
/// single interior throat; chooses the supersonic branch downstream when
/// `p_exit` is below the critical exit pressure. Returns primitives at
/// `n` stations (R = 287 J/kg·K).
#[must_use]
pub fn quasi_1d_nozzle(
    area: &dyn Fn(f64) -> f64,
    n: usize,
    p0: f64,
    t0: f64,
    p_exit: f64,
    gamma: f64,
) -> Vec<Prim> {
    let r_gas = 287.0;
    // Locate the throat.
    let mut x_throat = 0.5;
    let mut a_min = f64::INFINITY;
    for k in 0..=200 {
        let x = k as f64 / 200.0;
        let a = area(x);
        if a < a_min {
            a_min = a;
            x_throat = x;
        }
    }
    // Choked supersonic if the exit pressure is below the supersonic
    // isentropic exit pressure bound.
    let exit_ratio = area(1.0) / a_min;
    let m_exit_sup = nozzle_mach_from_area(exit_ratio, gamma, true);
    let p_exit_sup = p0 * (1.0 + (gamma - 1.0) / 2.0 * m_exit_sup * m_exit_sup)
        .powf(-gamma / (gamma - 1.0));
    let supersonic_downstream = p_exit <= p_exit_sup * 1.5;
    (0..n)
        .map(|k| {
            let x = (k as f64 + 0.5) / n as f64;
            let ratio = (area(x) / a_min).max(1.0);
            let m = if x < x_throat || !supersonic_downstream {
                nozzle_mach_from_area(ratio, gamma, false)
            } else {
                nozzle_mach_from_area(ratio, gamma, true)
            };
            let fac = 1.0 + (gamma - 1.0) / 2.0 * m * m;
            let p = p0 * fac.powf(-gamma / (gamma - 1.0));
            let t = t0 / fac;
            let rho = p / (r_gas * t);
            let a_sound = (gamma * r_gas * t).sqrt();
            Prim { rho, u: m * a_sound, p }
        })
        .collect()
}

/// Isentropic vortex (strength 5) advected by a uniform (1, 1)
/// background on a 10 × 10 periodic domain: returns (ρ, u, v, p).
#[must_use]
pub fn isentropic_vortex_exact(x: f64, y: f64, t: f64, gamma: f64) -> (f64, f64, f64, f64) {
    let beta = 5.0;
    let l = 10.0;
    let xc = (5.0 + t).rem_euclid(l);
    let yc = (5.0 + t).rem_euclid(l);
    // Nearest periodic image.
    let mut dx = x - xc;
    let mut dy = y - yc;
    if dx > l / 2.0 {
        dx -= l;
    }
    if dx < -l / 2.0 {
        dx += l;
    }
    if dy > l / 2.0 {
        dy -= l;
    }
    if dy < -l / 2.0 {
        dy += l;
    }
    let r2 = dx * dx + dy * dy;
    let ex = ((1.0 - r2) / 2.0).exp();
    let du = -beta / (2.0 * PI) * ex * dy;
    let dv = beta / (2.0 * PI) * ex * dx;
    let dt_temp = -(gamma - 1.0) * beta * beta / (8.0 * gamma * PI * PI) * (1.0 - r2).exp();
    let temp = 1.0 + dt_temp;
    let rho = temp.powf(1.0 / (gamma - 1.0));
    let p = temp.powf(gamma / (gamma - 1.0));
    (rho, 1.0 + du, 1.0 + dv, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversions_and_exact_star() {
        let p = Prim { rho: 1.3, u: 0.7, p: 2.1 };
        let back = cons_to_prim(prim_to_cons(p, 1.4), 1.4);
        assert!((back.rho - p.rho).abs() < 1e-12);
        assert!((back.u - p.u).abs() < 1e-12);
        assert!((back.p - p.p).abs() < 1e-12);
        // Toro test 1 (Sod): p* = 0.30313, u* = 0.92745.
        let l = Prim { rho: 1.0, u: 0.0, p: 1.0 };
        let r = Prim { rho: 0.125, u: 0.0, p: 0.1 };
        let (ps, us) = riemann_exact_star(l, r, 1.4);
        assert!((ps - 0.30313).abs() < 1e-4, "p* {ps}");
        assert!((us - 0.92745).abs() < 1e-4, "u* {us}");
        // Toro test 2 (123 problem): p* = 0.00189.
        let l2 = Prim { rho: 1.0, u: -2.0, p: 0.4 };
        let r2 = Prim { rho: 1.0, u: 2.0, p: 0.4 };
        let (ps2, us2) = riemann_exact_star(l2, r2, 1.4);
        assert!((ps2 - 0.00189).abs() < 5e-4, "p* {ps2}");
        assert!(us2.abs() < 1e-8, "u* {us2}");
    }

    #[test]
    fn test_hllc_sod_matches_exact() {
        let n = 400;
        let mut e = sod_shock_tube(n);
        e.flux = FluxKind::Hllc;
        let t_end = 0.2;
        while e.time < t_end {
            e.step(0.9);
        }
        let prims = e.primitives();
        let mut l1 = 0.0;
        for (i, pr) in prims.iter().enumerate() {
            let x = (i as f64 + 0.5) / n as f64;
            let ex = sod_exact(x, e.time);
            l1 += (pr.rho - ex.rho).abs();
        }
        l1 /= n as f64;
        // Mean exact density is ~0.5; 2% relative L1.
        assert!(l1 < 0.01, "Sod L1 density error {l1}");
        // Mass and energy conserved to roundoff before waves hit walls.
        let mut e2 = sod_shock_tube(200);
        let m0 = e2.total_mass();
        let en0 = e2.total_energy();
        for _ in 0..40 {
            e2.step(0.9);
        }
        assert!((e2.total_mass() - m0).abs() < 1e-12, "mass drift");
        assert!((e2.total_energy() - en0).abs() < 1e-11, "energy drift");
        // Shock position moves right of the split.
        assert!(e.shock_position().unwrap() > 0.7);
    }

    #[test]
    fn test_lax_problem_and_run_until() {
        // Lax's shock tube: the standard left/right states on a unit
        // domain with gamma = 1.4, split at the mid-point.
        let n = 400;
        let mut e = lax_problem(n);
        assert!((e.gamma - 1.4).abs() < 1e-15);
        assert!((e.dx - 1.0 / n as f64).abs() < 1e-15);
        assert_eq!(e.cells.len(), n);
        let l = Prim { rho: 0.445, u: 0.698, p: 3.528 };
        let r = Prim { rho: 0.5, u: 0.0, p: 0.571 };
        let p0 = e.primitives();
        for (i, pr) in p0.iter().enumerate() {
            let want = if (i as f64 + 0.5) / (n as f64) < 0.5 { l } else { r };
            assert!((pr.rho - want.rho).abs() < 1e-12, "rho at {i}: {}", pr.rho);
            assert!((pr.u - want.u).abs() < 1e-12, "u at {i}: {}", pr.u);
            assert!((pr.p - want.p).abs() < 1e-12, "p at {i}: {}", pr.p);
        }
        assert!(e.time == 0.0);

        // `run_until` advances to (at least) the requested time without
        // overshooting by more than a single step.
        let t_end = 0.13;
        e.flux = FluxKind::Hllc;
        e.run_until(t_end);
        assert!(e.time >= t_end - 1e-12, "run_until stopped short at {}", e.time);
        let dt_bound = 0.45 * e.dx
            / e.primitives()
                .iter()
                .map(|p| p.u.abs() + sound_speed(*p, e.gamma))
                .fold(0.0_f64, f64::max)
            * 2.0;
        assert!(
            e.time <= t_end + dt_bound,
            "run_until overshot: {} vs {t_end}",
            e.time
        );
        // Positivity of the numerical solution.
        assert!(
            e.primitives().iter().all(|p| p.rho > 0.0 && p.p > 0.0),
            "lost positivity"
        );

        // Compare against the exact self-similar solution.
        let l1_against_exact = |sim: &Euler1D| -> (f64, f64) {
            let m = sim.cells.len();
            let (mut l1_rho, mut l1_p) = (0.0, 0.0);
            for (i, pr) in sim.primitives().iter().enumerate() {
                let x = (i as f64 + 0.5) / (m as f64) - 0.5;
                let ex = riemann_exact(l, r, 1.4, x / sim.time);
                l1_rho += (pr.rho - ex.rho).abs();
                l1_p += (pr.p - ex.p).abs();
            }
            (l1_rho / m as f64, l1_p / m as f64)
        };
        let prims = e.primitives();
        let (l1_rho, l1_p) = l1_against_exact(&e);
        // The mean exact density over the tube is ~0.9 and the mean
        // pressure ~1.7; the scheme smears the contact and the shock over
        // a few cells, leaving well under 1% of L1 at 400 cells.
        assert!(l1_rho < 0.01, "Lax L1 density error {l1_rho}");
        assert!(l1_p < 0.01, "Lax L1 pressure error {l1_p}");

        // The solution converges under grid refinement (first order at the
        // discontinuities, so the error falls by roughly the mesh ratio's
        // square root or better). Measured: 6.5e-3 at n = 400 and 2.1e-3
        // at n = 1600.
        let mut fine = lax_problem(4 * n);
        fine.flux = FluxKind::Hllc;
        fine.run_until(t_end);
        let (l1_rho_fine, _) = l1_against_exact(&fine);
        assert!(
            l1_rho_fine < 0.5 * l1_rho,
            "no convergence under refinement: {l1_rho} -> {l1_rho_fine}"
        );

        // Star-region plateau: constant pressure and velocity between the
        // contact and the shock, matching the exact star state.
        let (p_star, u_star) = riemann_exact_star(l, r, 1.4);
        let i_shock = (0..n)
            .rev()
            .find(|&i| prims[i].p > 0.5 * (p_star + r.p))
            .expect("the shock must be inside the domain");
        assert!(i_shock < n - 5, "shock reached the boundary");
        for i in (i_shock - 20)..(i_shock - 6) {
            assert!(
                (prims[i].p / p_star - 1.0).abs() < 0.02,
                "star pressure {} vs {p_star} at cell {i}",
                prims[i].p
            );
            assert!(
                (prims[i].u / u_star - 1.0).abs() < 0.02,
                "star velocity {} vs {u_star} at cell {i}",
                prims[i].u
            );
        }
        // The contact sits to the left of the shock and moves at u*, so
        // density jumps there while pressure and velocity do not.
        let i_contact = ((0.5 + u_star * e.time) * n as f64) as usize;
        assert!(i_contact < i_shock, "contact must trail the shock");
        assert!(
            (prims[i_contact - 12].p / prims[i_contact + 12].p - 1.0).abs() < 0.02,
            "pressure jumps across the contact"
        );
        assert!(
            prims[i_contact - 12].rho / prims[i_contact + 12].rho < 0.85,
            "no density jump across the contact"
        );
    }

    #[test]
    fn test_run_until_conserves_mass_and_energy_periodically() {
        // With periodic boundaries the finite-volume update telescopes, so
        // `run_until` must conserve the integrals of rho and E exactly.
        let n = 200;
        let mut e = lax_problem(n);
        e.bc = EulerBc::Periodic;
        let m0 = e.total_mass();
        let en0 = e.total_energy();
        // Analytic mass of the initial data: half the domain at each state.
        assert!(
            (m0 - 0.5 * (0.445 + 0.5)).abs() < 1e-12,
            "initial mass {m0} vs {}",
            0.5 * (0.445 + 0.5)
        );
        e.run_until(0.05);
        assert!(e.time >= 0.05 - 1e-12);
        assert!((e.total_mass() - m0).abs() < 1e-12, "mass drift {} vs {m0}", e.total_mass());
        assert!(
            (e.total_energy() - en0).abs() < 1e-11,
            "energy drift {} vs {en0}",
            e.total_energy()
        );
        assert!(e.primitives().iter().all(|p| p.rho > 0.0 && p.p > 0.0));

        // `run_until` is just repeated `step` up to the target time:
        // reaching the same time by hand gives the same state.
        let mut manual = lax_problem(n);
        manual.bc = EulerBc::Periodic;
        while manual.time < 0.05 - 1e-12 {
            let remaining = 0.05 - manual.time;
            let dt = manual.step(0.45);
            if dt > remaining {
                break;
            }
        }
        assert!((manual.time - e.time).abs() < 1e-14, "times differ");
        for (a, b) in manual.cells.iter().zip(&e.cells) {
            assert!((a.rho - b.rho).abs() < 1e-14);
            assert!((a.mom - b.mom).abs() < 1e-14);
            assert!((a.e - b.e).abs() < 1e-14);
        }

        // Calling `run_until` again with a time already passed is a no-op.
        let before = e.time;
        e.run_until(0.0);
        assert!((e.time - before).abs() < 1e-15);
    }

    #[test]
    fn test_flux_functions_agree_on_smooth_and_run() {
        let gamma = 1.4;
        let l = prim_to_cons(Prim { rho: 1.0, u: 0.3, p: 1.0 }, gamma);
        let r = prim_to_cons(Prim { rho: 0.99, u: 0.31, p: 0.995 }, gamma);
        let hllc = flux_hllc(l, r, gamma);
        let roe = flux_roe(l, r, gamma);
        let hll = flux_hll(l, r, gamma);
        let exact = numerical_flux(FluxKind::Exact, l, r, gamma);
        for (a, b) in [(hllc.rho, roe.rho), (hllc.mom, roe.mom), (hllc.e, roe.e)] {
            assert!((a - b).abs() < 5e-3, "HLLC {a} vs Roe {b}");
        }
        assert!((hllc.rho - exact.rho).abs() < 5e-3);
        assert!((hll.rho - hllc.rho).abs() < 5e-3);
        // Consistency: F(u, u) = F(u).
        let f_phys = flux(l, gamma);
        for kind in [
            FluxKind::Exact,
            FluxKind::Hll,
            FluxKind::Hllc,
            FluxKind::Roe,
            FluxKind::Rusanov,
            FluxKind::AusmPlus,
        ] {
            let f = numerical_flux(kind, l, l, gamma);
            assert!((f.rho - f_phys.rho).abs() < 1e-10, "{kind:?} rho");
            assert!((f.mom - f_phys.mom).abs() < 1e-10, "{kind:?} mom");
            assert!((f.e - f_phys.e).abs() < 1e-10, "{kind:?} e");
        }
        // All fluxes run Sod stably.
        for kind in [FluxKind::Roe, FluxKind::Rusanov, FluxKind::AusmPlus, FluxKind::Hll] {
            let mut e = sod_shock_tube(100);
            e.flux = kind;
            for _ in 0..60 {
                e.step(0.8);
            }
            assert!(
                e.primitives().iter().all(|p| p.rho > 0.0 && p.p > 0.0),
                "{kind:?} lost positivity"
            );
        }
        // Blast wave and Shu-Osher survive with reflective/transmissive BCs.
        let mut b = blast_wave_woodward_colella(200);
        for _ in 0..100 {
            b.step(0.6);
        }
        assert!(b.primitives().iter().all(|p| p.rho > 0.0 && p.p > 0.0));
        let mut s = shu_osher(200);
        while s.time < 1.0 {
            s.step(0.8);
        }
        assert!(s.primitives().iter().all(|p| p.rho > 0.0));
        let mut sed = sedov_1d(101);
        for _ in 0..50 {
            sed.step(0.5);
        }
        assert!(sed.primitives().iter().all(|p| p.rho > 0.0 && p.p > 0.0));
    }

    #[test]
    fn test_normal_and_oblique_shocks() {
        let (p_ratio, rho_ratio, t_ratio, m2) = normal_shock_relations(2.0, 1.4);
        assert!((p_ratio - 4.5).abs() < 1e-12, "p2/p1 {p_ratio}");
        assert!((rho_ratio - 8.0 / 3.0).abs() < 1e-12, "rho ratio {rho_ratio}");
        assert!((t_ratio - 4.5 * 3.0 / 8.0).abs() < 1e-12);
        assert!((m2 - 0.57735).abs() < 1e-4, "M2 {m2}");
        // Rankine-Hugoniot post-shock state is consistent.
        let post = rankine_hugoniot(1.0, 1.4, 2.0, 1.4);
        assert!((post.p - 4.5).abs() < 1e-9);
        assert!((post.rho - 1.4 * 8.0 / 3.0).abs() < 1e-9);
        assert!(post.u > 0.0);
        // Oblique shock: M=2, deflection 10°: weak β ≈ 39.31°.
        let (weak, strong) = oblique_shock_angle(2.0, 10.0_f64.to_radians(), 1.4).unwrap();
        assert!((weak.to_degrees() - 39.31).abs() < 0.1, "weak {}", weak.to_degrees());
        assert!(strong.to_degrees() > 80.0, "strong {}", strong.to_degrees());
        // Detached for too-large deflection.
        assert!(oblique_shock_angle(1.5, 35.0_f64.to_radians(), 1.4).is_none());
        // Prandtl-Meyer ν(2) = 26.38°.
        assert!((prandtl_meyer(2.0, 1.4).to_degrees() - 26.3798).abs() < 1e-3);
        assert_eq!(prandtl_meyer(0.8, 1.4), 0.0);
    }

    #[test]
    fn test_nozzle_relations() {
        // A/A* at M=2, γ=1.4 is 1.6875.
        assert!((nozzle_area_ratio(2.0, 1.4) - 1.6875).abs() < 1e-4);
        assert!((nozzle_area_ratio(1.0, 1.4) - 1.0).abs() < 1e-12);
        let m = nozzle_mach_from_area(1.6875, 1.4, true);
        assert!((m - 2.0).abs() < 1e-3);
        let m_sub = nozzle_mach_from_area(1.6875, 1.4, false);
        assert!((nozzle_area_ratio(m_sub, 1.4) - 1.6875).abs() < 1e-3);
        assert!(m_sub < 1.0);
        // Converging-diverging nozzle: Mach increases through the throat
        // when choked.
        let area = |x: f64| 1.0 + 3.0 * (x - 0.5) * (x - 0.5);
        let states = quasi_1d_nozzle(&area, 50, 1e5, 300.0, 1e3, 1.4);
        assert_eq!(states.len(), 50);
        let a0 = (1.4 * 287.0 * 300.0_f64).sqrt();
        let m_in = states[2].u / (1.4 * states[2].p / states[2].rho).sqrt();
        let m_out = states[47].u / (1.4 * states[47].p / states[47].rho).sqrt();
        assert!(m_in < 1.0 && m_out > 1.0, "nozzle {m_in} -> {m_out}");
        assert!(states[47].u > a0, "supersonic exit");
        assert!(states.iter().all(|s| s.p > 0.0 && s.rho > 0.0));
    }

    #[test]
    fn test_euler2d_isentropic_vortex_second_order() {
        let err_at = |n: usize| -> f64 {
            let dx = 10.0 / n as f64;
            let mut e = Euler2D::new(n, n, dx, 1.4);
            e.periodic = true;
            for j in 0..n {
                for i in 0..n {
                    let x = (i as f64 + 0.5) * dx;
                    let y = (j as f64 + 0.5) * dx;
                    let (rho, u, v, p) = isentropic_vortex_exact(x, y, 0.0, 1.4);
                    e.set_cell(i, j, rho, u, v, p);
                }
            }
            let t_end = 2.0; // partial advection (full period t=10 is costly)
            while e.time < t_end {
                e.step(0.6);
            }
            let mut l1 = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let x = (i as f64 + 0.5) * dx;
                    let y = (j as f64 + 0.5) * dx;
                    let (rho, _, _, _) = isentropic_vortex_exact(x, y, e.time, 1.4);
                    l1 += (e.rho[j * n + i] - rho).abs();
                }
            }
            l1 / (n * n) as f64
        };
        let e16 = err_at(16);
        let e32 = err_at(32);
        assert!(
            e16 / e32 > 1.8,
            "vortex convergence ratio {} (e16 {e16}, e32 {e32})",
            e16 / e32
        );
    }

    #[test]
    fn test_euler2d_instabilities_and_schlieren() {
        let mut kh = Euler2D::new(32, 32, 1.0 / 32.0, 1.4);
        kh.kelvin_helmholtz_init();
        for _ in 0..20 {
            kh.step(0.5);
        }
        assert!(kh.rho.iter().all(|r| r.is_finite() && *r > 0.0));
        let s = kh.schlieren();
        assert!(s.iter().cloned().fold(0.0_f64, f64::max) <= 1.0 + 1e-12);
        assert!(s.iter().any(|v| *v > 0.5));
        let mut rt = Euler2D::new(16, 48, 1.0 / 16.0, 1.4);
        rt.rayleigh_taylor_init(1.0);
        for _ in 0..20 {
            rt.step(0.5);
        }
        assert!(rt.rho.iter().all(|r| r.is_finite() && *r > 0.0));
        let mut sb = Euler2D::new(48, 24, 1.0 / 24.0, 1.4);
        sb.shock_bubble_init();
        for _ in 0..20 {
            sb.step(0.5);
        }
        assert!(sb.rho.iter().all(|r| r.is_finite() && *r > 0.0));
        let mut dm = Euler2D::new(48, 16, 1.0 / 16.0, 1.4);
        dm.double_mach_reflection_init();
        for _ in 0..10 {
            dm.step(0.4);
        }
        assert!(dm.rho.iter().all(|r| r.is_finite() && *r > 0.0));
    }
}
