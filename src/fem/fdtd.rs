//! Finite-difference time domain: Maxwell's equations on a Yee grid.
//!
//! # Why the grid is staggered
//!
//! Maxwell's curl equations couple the two fields' time derivatives to
//! each other's spatial derivatives. Yee's arrangement puts `E` and `H`
//! half a cell apart in space *and* half a step apart in time, so that
//! every derivative in the scheme is a centred difference straddling the
//! point it is evaluated at. Nothing is interpolated and nothing is
//! averaged: the update is second-order accurate while using the
//! narrowest possible stencil, and it is explicit, so a step costs one
//! pass over the arrays.
//!
//! The arrangement also makes the discrete divergence of `B` exactly
//! conserved -- the update adds a discrete curl, and the discrete
//! divergence of a discrete curl is identically zero on this grid. A
//! collocated scheme has to enforce that separately or watch it drift.
//!
//! # The Courant limit is not a guideline
//!
//! With `S = c dt / dx`, the scheme's numerical dispersion relation
//! admits a real wavenumber for every real frequency only while
//! `S <= 1` in one dimension, or `S <= 1/sqrt(d)` in `d` dimensions.
//! Past that the scheme has a mode that grows geometrically, and it
//! grows from rounding noise if nothing else. This is not accuracy
//! degrading gently; it is a hard threshold, and
//! [`fdtd_courant_check`] reports which side of it a set of parameters
//! falls on.
//!
//! # The magic time step
//!
//! At exactly `S = 1` in one dimension the numerical dispersion relation
//! becomes the exact one, and the update degenerates into a shift: a
//! pulse moves one cell per step with its shape unchanged, to machine
//! precision, forever. One dimension is the only place this happens --
//! in two or three the dispersion error depends on the propagation angle
//! and cannot be cancelled at all angles at once, which is why a
//! two-dimensional simulation is run at a Courant number safely below
//! the limit rather than at it.
//!
//! # Fields are normalised
//!
//! The updates here track `E` and `eta_0 H` rather than `E` and `H`,
//! which removes the free-space impedance from every line of the update
//! and leaves the Courant number as the only coefficient. It also makes
//! the two fields comparable in magnitude, which matters because the
//! conserved energy adds their squares -- in unnormalised units one term
//! would be `1e5` times the other and the sum would be numerical
//! nonsense.

use crate::error::SolveError;

/// Whether a set of parameters satisfies the one-dimensional Courant
/// condition `c dt <= dx`.
///
/// Equality is admissible and is in fact the best possible choice in one
/// dimension: see the module note on the magic time step.
pub fn fdtd_courant_check(dx: f64, dt: f64, c: f64) -> bool {
    dx.is_finite()
        && dt.is_finite()
        && c.is_finite()
        && dx > 0.0
        && dt > 0.0
        && c > 0.0
        && c * dt <= dx * (1.0 + 1e-12)
}

/// The two-dimensional Courant condition, `c dt <= 1 / sqrt(1/dx^2 +
/// 1/dy^2)`.
///
/// On a square grid that is `dx / (c sqrt(2))`, and unlike the
/// one-dimensional case the bound is not a good place to sit: the
/// dispersion error at the limit vanishes along the diagonals and is
/// worst along the axes, so no single Courant number is exact for every
/// direction.
pub fn fdtd_courant_check_2d(dx: f64, dy: f64, dt: f64, c: f64) -> bool {
    if !(dx.is_finite() && dy.is_finite() && dt.is_finite() && c.is_finite()) {
        return false;
    }
    if dx <= 0.0 || dy <= 0.0 || dt <= 0.0 || c <= 0.0 {
        return false;
    }
    let limit = 1.0 / (1.0 / (dx * dx) + 1.0 / (dy * dy)).sqrt();
    c * dt <= limit * (1.0 + 1e-12)
}

/// The result of a one-dimensional run: the electric field at every
/// step, and the magnetic field alongside it.
#[derive(Debug, Clone, PartialEq)]
pub struct Fdtd1d {
    /// `steps + 1` snapshots of `E_z`, each with one value per cell.
    pub e: Vec<Vec<f64>>,
    /// The matching snapshots of the normalised `eta_0 H_y`, which lives
    /// half a cell to the right of each `E` sample and half a step
    /// later in time. One shorter than `e` in space.
    pub h: Vec<Vec<f64>>,
}

impl Fdtd1d {
    /// The exactly conserved energy of the leapfrog at snapshot `n`.
    ///
    /// Not the obvious `sum eps E^2 + sum H^2`, which oscillates by a
    /// term of order `dt` forever without drifting. What leapfrog
    /// conserves is the form with the magnetic term taken as the product
    /// of the two half-steps straddling the electric one,
    ///
    /// ```text
    /// U^n = (1/2) sum eps_r (E^n)^2 + (1/2) sum H^{n-1/2} H^{n+1/2}
    /// ```
    ///
    /// which is the discrete analogue of evaluating both fields at the
    /// same instant. It is conserved to rounding in a closed lossless
    /// domain, and it is the quantity whose boundedness is what
    /// stability means.
    ///
    /// Snapshot `k` of [`Fdtd1d::h`] holds `H^{k-1/2}`, so the two
    /// half-steps straddling `E^n` are `h[n]` and `h[n+1]`. Returns
    /// `None` for the final snapshot, which has only the earlier of the
    /// two available.
    pub fn energy(&self, eps_r: &[f64], n: usize) -> Option<f64> {
        if n + 1 >= self.h.len() || n >= self.e.len() || eps_r.len() != self.e[n].len() {
            return None;
        }
        let electric: f64 =
            self.e[n].iter().zip(eps_r.iter()).map(|(v, e)| e * v * v).sum::<f64>() * 0.5;
        let magnetic: f64 =
            self.h[n].iter().zip(self.h[n + 1].iter()).map(|(a, b)| a * b).sum::<f64>() * 0.5;
        Some(electric + magnetic)
    }
}

/// What to do at the ends of a one-dimensional grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary1d {
    /// A perfect electric conductor: `E = 0`, so a wave reflects with
    /// its sign flipped and no energy leaves. This is what a closed
    /// resonator is, and it is the setting in which the discrete energy
    /// is conserved exactly.
    Conductor,
    /// Mur's first-order absorbing condition, which extrapolates along
    /// the characteristic leaving the grid. It is exact for a wave at
    /// normal incidence and at the frequency the local Courant number
    /// was matched to, and leaks a little otherwise -- in one dimension
    /// there is only normal incidence, so it is very good indeed.
    Mur,
}

/// Marches the one-dimensional Yee scheme.
///
/// `eps_r` gives the relative permittivity of each cell, `courant` is
/// `c dt / dx` in vacuum, and `source` is added to `E` at `source_cell`
/// at every step -- a soft source, which a wave passes through rather
/// than reflecting off, unlike overwriting the cell.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a grid shorter than three cells,
/// a non-positive or non-finite permittivity, a source cell outside the
/// grid, a Courant number outside `(0, 1]`, or a Courant number above
/// the limit the grid's *fastest* medium sets -- past the limit the
/// scheme is unconditionally unstable and running it would produce
/// numbers rather than an answer.
pub fn fdtd_1d(
    eps_r: &[f64],
    source: &dyn Fn(usize) -> f64,
    source_cell: usize,
    courant: f64,
    steps: usize,
    boundary: Boundary1d,
) -> Result<Fdtd1d, SolveError> {
    let n = eps_r.len();
    if n < 3 {
        return Err(SolveError::InvalidArgument("need at least three cells"));
    }
    if eps_r.iter().any(|&e| !e.is_finite() || e <= 0.0) {
        return Err(SolveError::InvalidArgument("permittivity must be positive and finite"));
    }
    if source_cell >= n {
        return Err(SolveError::InvalidArgument("the source is outside the grid"));
    }
    if !courant.is_finite() || courant <= 0.0 || courant > 1.0 + 1e-12 {
        return Err(SolveError::InvalidArgument("the Courant number must lie in (0, 1]"));
    }
    // The stability bound is set by the *fastest* wave in the grid, so
    // it is the smallest permittivity that matters, not the vacuum one.
    // A permittivity below one -- a plasma above its cutoff, or an
    // engineered medium -- has a phase speed above c and tightens the
    // limit by exactly its index. Checking only the nominal Courant
    // number would let such a grid through to blow up.
    let slowest = eps_r.iter().copied().fold(f64::INFINITY, f64::min);
    if courant > slowest.sqrt() * (1.0 + 1e-12) {
        return Err(SolveError::InvalidArgument(
            "the Courant number exceeds the limit set by the fastest medium in the grid",
        ));
    }
    let mut e = vec![0.0; n];
    let mut h = vec![0.0; n - 1];
    let mut e_hist = Vec::with_capacity(steps + 1);
    let mut h_hist = Vec::with_capacity(steps + 1);
    e_hist.push(e.clone());
    h_hist.push(h.clone());
    for step in 0..steps {
        // The magnetic half-step. H[i] sits between E[i] and E[i+1].
        for i in 0..n - 1 {
            h[i] += courant * (e[i + 1] - e[i]);
        }
        // Mur reads both the edge cell and its neighbour at the old
        // time, so capture them before the interior update overwrites
        // the neighbour.
        let (old_edge_l, old_next_l) = (e[0], e[1]);
        let (old_edge_r, old_next_r) = (e[n - 1], e[n - 2]);
        // The electric step. The interior sees both neighbours; the ends
        // are handled by the boundary condition below.
        for i in 1..n - 1 {
            e[i] += courant / eps_r[i] * (h[i] - h[i - 1]);
        }
        match boundary {
            Boundary1d::Conductor => {
                e[0] = 0.0;
                e[n - 1] = 0.0;
            }
            Boundary1d::Mur => {
                // E^{n+1}[0] = E^n[1] + k (E^{n+1}[1] - E^n[0]), which
                // is the statement that the field is constant along the
                // characteristic leaving the grid. The local Courant
                // number carries the refractive index of the edge cell,
                // which is what makes the condition exact against a
                // medium other than vacuum.
                let coeff = |cell: usize| {
                    let s = courant / eps_r[cell].sqrt();
                    (s - 1.0) / (s + 1.0)
                };
                e[0] = old_next_l + coeff(0) * (e[1] - old_edge_l);
                e[n - 1] = old_next_r + coeff(n - 1) * (e[n - 2] - old_edge_r);
            }
        }
        e[source_cell] += source(step);
        if !e.iter().all(|v| v.is_finite()) {
            return Err(SolveError::NoConvergence { iters: step, residual: f64::INFINITY });
        }
        e_hist.push(e.clone());
        h_hist.push(h.clone());
    }
    Ok(Fdtd1d { e: e_hist, h: h_hist })
}

/// The photonic band gaps of an infinite `a`/`b` bilayer stack, found
/// from the Bloch dispersion relation.
///
/// A period of the stack has a transfer matrix, and Bloch's theorem says
/// the propagating states are those whose transfer matrix has unit
/// modulus eigenvalues. For a two-layer period that reduces to
///
/// ```text
/// cos(K L) = cos(k_a d_a) cos(k_b d_b)
///          - (1/2)(n_a/n_b + n_b/n_a) sin(k_a d_a) sin(k_b d_b)
/// ```
///
/// with `k_i = omega n_i / c`. A frequency for which the right-hand side
/// exceeds one in magnitude has no real `K`: nothing propagates, and
/// that is a gap. The prefactor `(n_a/n_b + n_b/n_a)/2` is at least one
/// with equality only when the two indices agree, which is the whole
/// reason a gap exists at all -- a homogeneous "stack" has none.
///
/// Frequencies are angular and the speed of light is taken as one, so a
/// frequency is really `omega L / c` in disguise; scaling every
/// thickness by a factor scales every gap edge by its reciprocal.
///
/// Returns the gaps below `omega_max` as `(low, high)` pairs, ascending,
/// with the edges refined by bisection rather than left at the sampling
/// resolution.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for non-positive permittivities or
/// thicknesses, a non-positive frequency ceiling, or fewer than two
/// samples.
pub fn photonic_crystal_bandgap_1d(
    eps_a: f64,
    eps_b: f64,
    d_a: f64,
    d_b: f64,
    omega_max: f64,
    samples: usize,
) -> Result<Vec<(f64, f64)>, SolveError> {
    for v in [eps_a, eps_b, d_a, d_b, omega_max] {
        if !v.is_finite() || v <= 0.0 {
            return Err(SolveError::InvalidArgument("stack parameters must be positive"));
        }
    }
    if samples < 2 {
        return Err(SolveError::InvalidArgument("need at least two samples"));
    }
    let (na, nb) = (eps_a.sqrt(), eps_b.sqrt());
    let mix = 0.5 * (na / nb + nb / na);
    // The Bloch trace, whose magnitude exceeding one marks a gap.
    let trace = |w: f64| {
        let (pa, pb) = (w * na * d_a, w * nb * d_b);
        pa.cos() * pb.cos() - mix * pa.sin() * pb.sin()
    };
    let gap = |w: f64| trace(w).abs() - 1.0;
    let mut edges = Vec::new();
    let step = omega_max / samples as f64;
    // Start just above zero: the trace is exactly one at omega = 0 for
    // every stack, which is the long-wavelength limit where the layers
    // are invisible, and is a tangency rather than a crossing.
    let mut previous = step * 1e-6;
    let mut previous_gap = gap(previous);
    for k in 1..=samples {
        let w = k as f64 * step;
        let g = gap(w);
        if previous_gap.signum() != g.signum() && previous_gap != 0.0 {
            // Bisect for the crossing. The trace is smooth, so a
            // bisection to the last representable bit is cheap and
            // leaves the edge exact rather than sampling-limited.
            let (mut lo, mut hi) = (previous, w);
            let lo_sign = previous_gap.signum();
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if hi - lo <= 1e-15 * (1.0 + hi.abs()) {
                    break;
                }
                if gap(mid).signum() == lo_sign {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            edges.push((0.5 * (lo + hi), g > 0.0));
        }
        previous = w;
        previous_gap = g;
    }
    // Pair the openings with the closings that follow them. A gap left
    // open at the ceiling is reported up to the ceiling rather than
    // dropped, since dropping it would hide a gap that is there.
    let mut gaps = Vec::new();
    let mut open: Option<f64> = None;
    for (w, rising) in edges {
        if rising {
            open = Some(w);
        } else if let Some(start) = open.take() {
            gaps.push((start, w));
        }
    }
    if let Some(start) = open {
        gaps.push((start, omega_max));
    }
    Ok(gaps)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth pulse, wide enough that the grid resolves it well.
    fn pulse(step: usize) -> f64 {
        let t = step as f64 - 30.0;
        (-t * t / 120.0).exp()
    }

    /// A Hann burst that is *identically* zero from step 60 onwards.
    ///
    /// A Gaussian never quite stops -- at step 70 the one above is still
    /// injecting about 1e-6, which is small but is not nothing, and it
    /// swamps a conservation test at machine precision. Compact support
    /// makes "after the source stops" an exact statement rather than an
    /// approximate one.
    fn burst(step: usize) -> f64 {
        if step >= 60 {
            return 0.0;
        }
        let x = step as f64 / 60.0;
        0.5 * (1.0 - (std::f64::consts::TAU * x).cos())
    }

    #[test]
    fn the_magic_time_step_is_an_exact_shift() {
        // At a Courant number of one in vacuum the numerical dispersion
        // relation becomes the exact one and the update degenerates into
        // a translation. Not approximately: the two snapshots agree bit
        // for bit.
        let n = 200;
        let eps = vec![1.0; n];
        let r = fdtd_1d(&eps, &pulse, 100, 1.0, 90, Boundary1d::Conductor).unwrap();
        let mut worst: f64 = 0.0;
        for step in 60..80 {
            for i in 120..180 {
                worst = worst.max((r.e[step + 1][i] - r.e[step][i - 1]).abs());
            }
        }
        assert_eq!(worst, 0.0, "the pulse did not translate exactly");
        // And it is still a pulse rather than a decayed smear.
        let peak = r.e[70].iter().cloned().fold(0.0f64, f64::max);
        assert!(peak > 0.4, "the pulse faded to {peak}");
    }

    #[test]
    fn the_leapfrog_conserves_its_own_energy_and_not_the_obvious_one() {
        // Leapfrog conserves the form whose magnetic term is the product
        // of the two half-steps straddling the electric one. The obvious
        // sum of squares is not conserved -- it wobbles by a term of
        // order dt -- and asserting the wrong one would be asserting a
        // tolerance rather than an invariant.
        let n = 200;
        let eps: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * (i as f64 / n as f64)).collect();
        let r = fdtd_1d(&eps, &burst, 100, 0.9, 120, Boundary1d::Conductor).unwrap();
        let reference = r.energy(&eps, 70).unwrap();
        assert!(reference > 0.1, "there was no energy to conserve");
        for step in 70..=119 {
            let u = r.energy(&eps, step).unwrap();
            assert!(
                (u - reference).abs() < 1e-12 * reference,
                "step {step} drifted to {u} from {reference}"
            );
        }
        let naive = |k: usize| -> f64 {
            0.5 * r.e[k].iter().zip(eps.iter()).map(|(v, e)| e * v * v).sum::<f64>()
                + 0.5 * r.h[k].iter().map(|v| v * v).sum::<f64>()
        };
        let spread = (70..=119).map(naive).fold(f64::NEG_INFINITY, f64::max)
            - (70..=119).map(naive).fold(f64::INFINITY, f64::min);
        assert!(spread > 1e-9 * reference, "the naive energy was conserved after all");
        // Before anything has happened there is no energy, and the last
        // snapshot is missing the later of its two magnetic half-steps.
        assert_eq!(r.energy(&eps, 0), Some(0.0));
        assert!(r.energy(&eps, 120).is_none());
        assert!(r.energy(&eps[..3], 5).is_none());
    }

    #[test]
    fn the_courant_conditions_bound_what_they_should() {
        assert!(fdtd_courant_check(1.0, 1.0, 1.0));
        assert!(fdtd_courant_check(1.0, 0.5, 1.0));
        assert!(!fdtd_courant_check(1.0, 1.001, 1.0));
        assert!(!fdtd_courant_check(0.0, 1.0, 1.0));
        assert!(!fdtd_courant_check(1.0, -1.0, 1.0));
        assert!(!fdtd_courant_check(1.0, 1.0, f64::NAN));
        // On a square grid the two-dimensional limit is dx / (c sqrt 2),
        // strictly tighter than the one-dimensional one.
        let root2 = 2.0f64.sqrt();
        assert!(fdtd_courant_check_2d(1.0, 1.0, 1.0 / root2, 1.0));
        assert!(!fdtd_courant_check_2d(1.0, 1.0, 1.0 / root2 * 1.001, 1.0));
        assert!(fdtd_courant_check(1.0, 1.0 / root2 * 1.001, 1.0));
        // A grid fine in one direction is limited by that direction.
        assert!(fdtd_courant_check_2d(1.0, 0.1, 0.09, 1.0));
        assert!(!fdtd_courant_check_2d(1.0, 0.1, 0.11, 1.0));
        assert!(!fdtd_courant_check_2d(1.0, 0.0, 0.1, 1.0));
    }

    #[test]
    fn a_medium_faster_than_vacuum_tightens_the_limit() {
        // The stability bound belongs to the fastest wave in the grid.
        // A permittivity below one has a phase speed above c, so a
        // Courant number that vacuum would allow is unstable there --
        // and is refused rather than run.
        let mut eps = vec![1.0; 60];
        for e in eps.iter_mut().skip(30).take(10) {
            *e = 0.25;
        }
        assert!(fdtd_1d(&eps, &pulse, 5, 0.9, 10, Boundary1d::Mur).is_err());
        // Half the index means half the allowed Courant number, exactly.
        assert!(fdtd_1d(&eps, &pulse, 5, 0.5, 10, Boundary1d::Mur).is_ok());
        assert!(fdtd_1d(&eps, &pulse, 5, 0.51, 10, Boundary1d::Mur).is_err());
    }

    #[test]
    fn the_absorbing_boundary_beats_a_wall_by_orders_of_magnitude() {
        let n = 400;
        let eps = vec![1.0; n];
        let residual = |b| {
            let r = fdtd_1d(&eps, &pulse, n / 2, 1.0, 400, b).unwrap();
            r.e[400].iter().cloned().fold(0.0f64, |a, v| a.max(v.abs()))
        };
        let mur = residual(Boundary1d::Mur);
        let wall = residual(Boundary1d::Conductor);
        assert!(wall > 0.4, "the wall did not reflect the pulse: {wall}");
        assert!(mur < 1e-3, "the absorbing boundary left {mur} behind");
        assert!(wall / mur > 1e3, "the absorber was only {}x better", wall / mur);
    }

    #[test]
    fn a_dielectric_interface_reproduces_the_fresnel_coefficients() {
        // Normal incidence from vacuum onto an index of two: the
        // reflected amplitude is (n1 - n2)/(n1 + n2) = -1/3 and the
        // transmitted one is 2 n1/(n1 + n2) = 2/3. The sign matters as
        // much as the magnitude -- reflection off a denser medium
        // inverts the field.
        let n = 400;
        let mut eps = vec![1.0; n];
        for e in eps.iter_mut().skip(n / 2) {
            *e = 4.0;
        }
        let r = fdtd_1d(&eps, &pulse, 60, 1.0, 260, Boundary1d::Mur).unwrap();
        let extreme = |v: &[f64]| v.iter().copied().fold(0.0f64, |a, x| if x.abs() > a.abs() { x } else { a });
        let incident = extreme(&r.e[120][..190]);
        let reflected = extreme(&r.e[250][..190]);
        let transmitted = extreme(&r.e[250][210..]);
        assert!(incident > 0.4, "no incident pulse: {incident}");
        assert!(
            (reflected / incident + 1.0 / 3.0).abs() < 0.02,
            "reflection was {}",
            reflected / incident
        );
        assert!(
            (transmitted / incident - 2.0 / 3.0).abs() < 0.02,
            "transmission was {}",
            transmitted / incident
        );
    }

    #[test]
    fn the_quarter_wave_stack_has_the_gaps_the_theory_gives_it() {
        // Layers of equal optical thickness n d = lambda_0 / 4 put a gap
        // centred exactly on the design frequency, and on every odd
        // multiple of it. The even multiples are closed: there the two
        // layers are each a half wave and the period is invisible.
        let (ea, eb): (f64, f64) = (1.0, 4.0);
        let (na, nb) = (ea.sqrt(), eb.sqrt());
        let (da, db) = (1.0, na / nb);
        let w0 = std::f64::consts::PI / (2.0 * na * da);
        let gaps = photonic_crystal_bandgap_1d(ea, eb, da, db, 4.5 * w0, 4000).unwrap();
        assert!(gaps.len() >= 2, "found only {} gaps", gaps.len());
        for (m, (lo, hi)) in [(1.0, gaps[0]), (3.0, gaps[1])] {
            let centre = 0.5 * (lo + hi);
            assert!((centre / w0 - m).abs() < 1e-9, "gap centred at {} w0", centre / w0);
            // The relative width of the m-th odd gap is
            // (4 / (m pi)) arcsin(|na - nb| / (na + nb)).
            let want = 4.0 / (m * std::f64::consts::PI)
                * ((nb - na) / (nb + na)).asin();
            let got = (hi - lo) / centre;
            assert!((got - want).abs() < 1e-6, "gap {m}: width {got}, theory {want}");
        }
        // Nothing straddles the even multiple.
        assert!(
            !gaps.iter().any(|&(lo, hi)| lo < 2.0 * w0 && hi > 2.0 * w0),
            "the second-order gap did not close"
        );
    }

    #[test]
    fn a_homogeneous_stack_has_no_gaps_and_scaling_moves_them_all() {
        // With equal indices the mixing factor is exactly one and the
        // trace is cos of the total phase, which never leaves [-1, 1].
        assert!(photonic_crystal_bandgap_1d(2.25, 2.25, 1.0, 0.7, 40.0, 3000).unwrap().is_empty());
        // Thicknesses and frequencies are reciprocal, so doubling every
        // layer halves every gap edge.
        let base = photonic_crystal_bandgap_1d(1.0, 4.0, 1.0, 0.5, 12.0, 6000).unwrap();
        let stretched = photonic_crystal_bandgap_1d(1.0, 4.0, 2.0, 1.0, 6.0, 6000).unwrap();
        assert!(!base.is_empty());
        assert_eq!(base.len(), stretched.len());
        for (a, b) in base.iter().zip(stretched.iter()) {
            assert!((a.0 - 2.0 * b.0).abs() < 1e-8 * a.0);
            assert!((a.1 - 2.0 * b.1).abs() < 1e-8 * a.1);
        }
    }

    #[test]
    fn the_solvers_refuse_impossible_arguments() {
        let eps = vec![1.0; 10];
        assert!(fdtd_1d(&eps[..2], &pulse, 0, 0.5, 3, Boundary1d::Mur).is_err());
        assert!(fdtd_1d(&[1.0, 0.0, 1.0], &pulse, 0, 0.5, 3, Boundary1d::Mur).is_err());
        assert!(fdtd_1d(&eps, &pulse, 99, 0.5, 3, Boundary1d::Mur).is_err());
        assert!(fdtd_1d(&eps, &pulse, 0, 0.0, 3, Boundary1d::Mur).is_err());
        assert!(fdtd_1d(&eps, &pulse, 0, 1.5, 3, Boundary1d::Mur).is_err());
        assert!(photonic_crystal_bandgap_1d(0.0, 1.0, 1.0, 1.0, 1.0, 10).is_err());
        assert!(photonic_crystal_bandgap_1d(1.0, 1.0, 1.0, 1.0, -1.0, 10).is_err());
        assert!(photonic_crystal_bandgap_1d(1.0, 2.0, 1.0, 1.0, 1.0, 1).is_err());
    }
}
