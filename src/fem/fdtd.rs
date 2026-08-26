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

/// The state of a two-dimensional run.
///
/// The final snapshot alone is close to useless for a driven problem --
/// it is whatever phase the oscillation happened to land on -- so the
/// envelope is carried alongside it. That is a deliberate departure from
/// returning a bare field: what a steady-state calculation is *for* is
/// the amplitude, and reconstructing it from one snapshot is not
/// possible.
#[derive(Debug, Clone, PartialEq)]
pub struct Fdtd2d {
    /// Cells across.
    pub nx: usize,
    /// Cells down.
    pub ny: usize,
    /// The final `E_z`, row-major with `index = j * nx + i`.
    pub ez: Vec<f64>,
    /// The largest `|E_z|` each cell reached over the last quarter of
    /// the march, which for a driven problem is its steady amplitude
    /// and for a pulsed one is what passed through.
    pub envelope: Vec<f64>,
}

/// The dimensionless per-step loss of a polynomially graded perfectly
/// matched layer at continuous position `pos` along an axis of `n`
/// cells.
///
/// The grading matters. A layer that switches its conductivity on
/// abruptly reflects from the discontinuity far more than it absorbs,
/// which defeats the point; a polynomial ramp of order three is the
/// usual compromise between a gentle entry and a short layer. The peak
/// value follows from the round-trip attenuation a layer of this depth
/// and profile gives: integrating the loss through the layer and back
/// out gives `exp(-2 s_max D / ((m+1) S))`, so aiming at a reflection
/// `R0` fixes `s_max`.
fn pml_loss(pos: f64, n: usize, pml: usize, courant: f64, reflection: f64) -> f64 {
    if pml == 0 {
        return 0.0;
    }
    const ORDER: f64 = 3.0;
    let d = pml as f64;
    let depth = if pos < d {
        d - pos
    } else if pos > (n - 1) as f64 - d {
        pos - ((n - 1) as f64 - d)
    } else {
        return 0.0;
    };
    let s_max = -(ORDER + 1.0) * courant * reflection.ln() / (2.0 * d);
    s_max * (depth / d).powf(ORDER)
}

/// Marches the two-dimensional transverse-magnetic Yee scheme with a
/// Berenger split-field perfectly matched layer.
///
/// `eps_r` is row-major over `nx * ny` cells. `source` gives the value
/// added softly at `source_pos` on each step, exactly as in
/// [`fdtd_1d`] -- a continuous sinusoid at `f` cycles per step is
/// `|s| (TAU * f * s as f64).sin()`, and a pulse is anything with
/// compact support. Taking the waveform rather than a frequency is what
/// lets a caller switch the drive off, which is the only way to measure
/// what a boundary reflects: with a source still running, the field near
/// it is the source's own and says nothing about the layer.
///
/// Ramp a continuous drive on rather than switching it: a step
/// broadcasts across the whole band the grid can carry, and none of it
/// is what was asked for.
///
/// # Why the field is split
///
/// A lossy layer absorbs, but an ordinary lossy layer also *reflects*,
/// because its impedance differs from the vacuum it adjoins. Berenger's
/// construction splits `E_z` into the two parts that the two spatial
/// derivatives feed, and damps each with the loss belonging to its own
/// axis. The resulting medium is matched at every angle and every
/// frequency, which no single isotropic conductivity can be: what is
/// left is only the reflection from grading the profile over a finite
/// depth, and that is what the `reflection` target controls.
///
/// The layer is backed by a conductor. That is not a flaw -- anything
/// that reaches the backing has crossed the graded layer twice and comes
/// back attenuated by the round-trip factor the grading was designed
/// for.
///
/// `pml` gives the depth on each axis separately, `(x, y)`. A depth of
/// zero on an axis leaves plain conducting walls there, which is what a
/// waveguide wants: absorbing its side walls would stop it being a
/// waveguide, while absorbing its ends stops the switch-on transient
/// rattling around forever and swamping the field being measured.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a grid smaller than the layers
/// need, a permittivity array of the wrong length or with a non-positive
/// entry, a source outside the grid, a non-finite frequency, a
/// reflection target outside `(0, 1)`, or a Courant number above the
/// two-dimensional limit for the fastest medium present.
#[allow(clippy::too_many_arguments)]
pub fn fdtd_2d_tm(
    eps_r: &[f64],
    source_pos: (usize, usize),
    source: &dyn Fn(usize) -> f64,
    nx: usize,
    ny: usize,
    steps: usize,
    pml: (usize, usize),
    courant: f64,
    reflection: f64,
) -> Result<Fdtd2d, SolveError> {
    if source_pos.0 >= nx || source_pos.1 >= ny {
        return Err(SolveError::InvalidArgument("the source is outside the grid"));
    }
    let profile = [(source_pos.1 * nx + source_pos.0, 1.0)];
    march_2d(eps_r, &profile, source, nx, ny, steps, pml, courant, reflection)
}

/// The shared march. `sources` gives flat cell indices and the weight
/// each carries, which is what lets a caller excite a whole transverse
/// profile at once -- a single cell excites every mode a guide has, and
/// only a profile matching one of them excites that one alone.
#[allow(clippy::too_many_arguments)]
fn march_2d(
    eps_r: &[f64],
    sources: &[(usize, f64)],
    source: &dyn Fn(usize) -> f64,
    nx: usize,
    ny: usize,
    steps: usize,
    pml: (usize, usize),
    courant: f64,
    reflection: f64,
) -> Result<Fdtd2d, SolveError> {
    if nx < 5 || ny < 5 {
        return Err(SolveError::InvalidArgument("the grid must be at least five cells across"));
    }
    if eps_r.len() != nx * ny {
        return Err(SolveError::DimensionMismatch { expected: nx * ny, got: eps_r.len() });
    }
    if eps_r.iter().any(|&e| !e.is_finite() || e <= 0.0) {
        return Err(SolveError::InvalidArgument("permittivity must be positive and finite"));
    }
    if 2 * pml.0 + 1 >= nx || 2 * pml.1 + 1 >= ny {
        return Err(SolveError::InvalidArgument("the absorbing layers leave no interior"));
    }
    if sources.iter().any(|&(k, w)| k >= nx * ny || !w.is_finite()) {
        return Err(SolveError::InvalidArgument("a source is outside the grid or not finite"));
    }
    if !reflection.is_finite() || reflection <= 0.0 || reflection >= 1.0 {
        return Err(SolveError::InvalidArgument("the reflection target must lie in (0, 1)"));
    }
    let slowest = eps_r.iter().copied().fold(f64::INFINITY, f64::min);
    if !courant.is_finite()
        || courant <= 0.0
        || courant > (slowest / 2.0).sqrt() * (1.0 + 1e-12)
    {
        return Err(SolveError::InvalidArgument(
            "the Courant number exceeds the two-dimensional limit for the fastest medium",
        ));
    }

    // Update coefficients, one per grid line. The E lines sit on the
    // integers and the H lines half a cell off, so each axis needs both.
    let coeffs = |n: usize, depth: usize, offset: f64| -> (Vec<f64>, Vec<f64>) {
        (0..n)
            .map(|k| {
                let a = 0.5 * pml_loss(k as f64 + offset, n, depth, courant, reflection);
                ((1.0 - a) / (1.0 + a), courant / (1.0 + a))
            })
            .unzip()
    };
    let (cax, cbx) = coeffs(nx, pml.0, 0.0);
    let (cay, cby) = coeffs(ny, pml.1, 0.0);
    let (dax, dbx) = coeffs(nx, pml.0, 0.5);
    let (day, dby) = coeffs(ny, pml.1, 0.5);

    let mut ezx = vec![0.0; nx * ny];
    let mut ezy = vec![0.0; nx * ny];
    let mut ez = vec![0.0; nx * ny];
    // Hy[j][i] straddles Ez[j][i] and Ez[j][i+1]; Hx[j][i] straddles
    // Ez[j][i] and Ez[j+1][i].
    let mut hy = vec![0.0; ny * (nx - 1)];
    let mut hx = vec![0.0; (ny - 1) * nx];
    let mut envelope = vec![0.0; nx * ny];
    let record_from = steps - steps / 4;

    for step in 0..steps {
        for j in 0..ny {
            for i in 0..nx - 1 {
                let k = j * (nx - 1) + i;
                hy[k] = dax[i] * hy[k] + dbx[i] * (ez[j * nx + i + 1] - ez[j * nx + i]);
            }
        }
        for j in 0..ny - 1 {
            for i in 0..nx {
                let k = j * nx + i;
                hx[k] = day[j] * hx[k] - dby[j] * (ez[(j + 1) * nx + i] - ez[j * nx + i]);
            }
        }
        // The outermost ring is the conductor backing the layer, so it
        // is left at zero and the interior is updated.
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                let e = eps_r[k];
                ezx[k] = cax[i] * ezx[k]
                    + cbx[i] / e * (hy[j * (nx - 1) + i] - hy[j * (nx - 1) + i - 1]);
                ezy[k] = cay[j] * ezy[k]
                    - cby[j] / e * (hx[j * nx + i] - hx[(j - 1) * nx + i]);
                ez[k] = ezx[k] + ezy[k];
            }
        }
        let drive = source(step);
        if !drive.is_finite() {
            return Err(SolveError::InvalidArgument("the source must be finite"));
        }
        if drive != 0.0 {
            for &(k, weight) in sources {
                ezx[k] += 0.5 * drive * weight;
                ezy[k] += 0.5 * drive * weight;
                ez[k] = ezx[k] + ezy[k];
            }
        }
        if !ez.iter().all(|v| v.is_finite()) {
            return Err(SolveError::NoConvergence { iters: step, residual: f64::INFINITY });
        }
        if step >= record_from {
            for (slot, v) in envelope.iter_mut().zip(ez.iter()) {
                *slot = f64::max(*slot, v.abs());
            }
        }
    }
    Ok(Fdtd2d { nx, ny, ez, envelope })
}

/// Infers a parallel-plate waveguide's cutoff frequency from the
/// evanescent decay it shows when driven below that cutoff.
///
/// The guide is `width` cells between conducting plates, driven in its
/// `mode`-th transverse pattern at angular frequency `omega` in radians
/// per unit *time*, with the cell size and the speed of light both one
/// -- so a step advances the phase by `omega * S`, not by `omega`.
/// Below cutoff nothing propagates: the field falls off as
/// `exp(-alpha x)`, and measuring `alpha` down the guide gives the
/// cutoff back.
///
/// # Which cutoff comes back
///
/// Not the textbook `m pi c / a`. The grid has its own dispersion
/// relation,
///
/// ```text
/// sin^2(omega S / 2) / S^2 = sin^2(k_x / 2) + sin^2(k_y / 2)
/// ```
///
/// and an evanescent `k_x = i alpha` turns the first term on the right
/// into `-sinh^2(alpha / 2)`. Solving for where `alpha` vanishes gives
/// the *numerical* cutoff
///
/// ```text
/// omega_c = (2 / S) arcsin(S sin(k_y / 2)),   k_y = m pi / a
/// ```
///
/// which is what this returns and what the simulation actually has. It
/// approaches the continuum value as the guide is resolved more finely,
/// from below -- the grid is always a little slow -- and the difference
/// is second order in the cell size. Reporting the continuum figure
/// would be reporting what the answer ought to be rather than what it
/// is.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a mode outside `1..width`, a
/// guide too short to measure a decay in, or a frequency at or above the
/// numerical cutoff, where there is no decay to measure;
/// [`SolveError::NoConvergence`] if the measured profile is not a clean
/// exponential, which is the honest answer when a mode is close to the
/// grid's resolution limit: three half-waves across ten cells decays
/// within a couple of cells, leaving too little of the profile above the
/// numerical floor to fit a slope to. Widening the guide fixes it.
pub fn waveguide_cutoff_check_fdtd(
    width: usize,
    length: usize,
    mode: usize,
    omega: f64,
    courant: f64,
    steps: usize,
) -> Result<f64, SolveError> {
    if width < 4 || mode == 0 || mode >= width {
        return Err(SolveError::InvalidArgument("the mode must lie in 1..width"));
    }
    if length < 80 {
        return Err(SolveError::InvalidArgument("the guide is too short to measure a decay"));
    }
    if !omega.is_finite() || omega <= 0.0 {
        return Err(SolveError::InvalidArgument("the frequency must be positive and finite"));
    }
    if !courant.is_finite() || courant <= 0.0 || courant > 0.5f64.sqrt() * (1.0 + 1e-12) {
        return Err(SolveError::InvalidArgument("the Courant number exceeds the plane limit"));
    }
    let ky = std::f64::consts::PI * mode as f64 / width as f64;
    let numerical_cutoff = 2.0 / courant * (courant * (0.5 * ky).sin()).asin();
    if omega >= numerical_cutoff {
        return Err(SolveError::InvalidArgument(
            "the drive is at or above cutoff, so there is no evanescent decay to measure",
        ));
    }
    // The guide runs along x with conducting plates at j = 0 and
    // j = width. Ez is clamped there, which is what a plate is.
    let (nx, ny) = (length, width + 1);
    let eps = vec![1.0; nx * ny];
    // Absorb the ends but not the plates. Without this the guide is a
    // closed box: the transient the drive radiates when it switches on
    // contains frequencies above cutoff, those propagate, nothing damps
    // them, and after a few thousand steps they are what the envelope
    // is measuring rather than the evanescent field.
    let pad = 12usize;
    // Drive the whole section with the mode's own transverse pattern.
    // The discrete sine vectors are exactly orthogonal, so this excites
    // that mode and no other -- which matters, because a lower mode has
    // a lower cutoff and might be propagating at a frequency where this
    // one is not, and a single point source would excite it.
    let column = pad + 3;
    let profile: Vec<(usize, f64)> = (1..ny - 1)
        .map(|j| {
            (j * nx + column, (ky * j as f64).sin())
        })
        .collect();
    // Ramp the drive on over twenty periods, then hold it: the decay
    // being measured is the steady-state one, and a step would put a
    // broadband transient down the guide that propagates where the
    // wanted frequency does not.
    // `omega` is radians per unit *time*, and a step advances time by
    // dt = S (the cell size and the speed of light are both one), so the
    // phase advances by omega * S per step. Driving at omega per step
    // instead would silently simulate a frequency 1/S times too high --
    // which still produces a clean exponential, just the wrong one.
    let period_steps = std::f64::consts::TAU / (omega * courant);
    let ramp = 20.0 * period_steps;
    let drive = |step: usize| {
        let t = step as f64;
        let window = if t < ramp {
            0.5 * (1.0 - (std::f64::consts::PI * t / ramp).cos())
        } else {
            1.0
        };
        window * (omega * courant * t).sin()
    };
    let run = march_2d(&eps, &profile, &drive, nx, ny, steps, (pad, 0), courant, 1e-6)?;
    // Amplitude down the guide, summed across the section so that a
    // node of the transverse pattern does not read as a zero.
    let profile: Vec<f64> = (0..nx)
        .map(|i| (1..ny - 1).map(|j| run.envelope[j * nx + i]).sum::<f64>())
        .collect();
    // Fit the log slope well away from the source and from the far end,
    // where the reflection off the terminating wall contaminates it.
    // Only a few cells downstream. The source is the mode's own
    // transverse pattern and the discrete sine vectors are exactly
    // orthogonal, so no other mode is excited and there is nothing to
    // wait out -- only the source cell's own near field, which is two
    // or three cells wide. Backing off a whole guide width instead
    // would cost most of the dynamic range, and a fast-decaying high
    // mode has little to spare.
    let start = column + 3;
    let end = nx - pad - width - 2;
    if end <= start + 8 {
        return Err(SolveError::InvalidArgument("the guide is too short to measure a decay"));
    }
    // Fit only across the clean exponential. Two things bound it. Near
    // the source the higher modes are still present, and they decay
    // faster, so the profile starts steeper than the mode being
    // measured -- hence beginning a guide width downstream. Far from it
    // the evanescent field reaches a floor, set by whatever the layers
    // failed to absorb, and beyond that the profile flattens; including
    // that stretch does not merely add scatter, it drags the fitted
    // slope towards zero, and with a long enough guide it reports no
    // decay at all.
    //
    // The floor is the smallest value in the window, and the fit stops
    // where the signal is still fifty times above it, which keeps its
    // contribution to the slope in the third digit.
    let floor = (start..end).map(|i| profile[i]).fold(f64::INFINITY, f64::min);
    if !(profile[start] > 0.0) {
        return Err(SolveError::NoConvergence { iters: steps, residual: profile[start] });
    }
    let threshold = (50.0 * floor).max(profile[start] * 1e-13);
    let mut stop = start;
    while stop < end && profile[stop] > threshold {
        stop += 1;
    }
    let points: Vec<(f64, f64)> = (start..stop)
        .filter(|&i| profile[i] > 0.0)
        .map(|i| (i as f64, profile[i].ln()))
        .collect();
    if points.len() < 10 {
        return Err(SolveError::NoConvergence { iters: steps, residual: points.len() as f64 });
    }
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let syy: f64 = points.iter().map(|p| (p.1 - my) * (p.1 - my)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    if sxx <= 0.0 || syy <= 0.0 {
        return Err(SolveError::NoConvergence { iters: steps, residual: f64::INFINITY });
    }
    // A pure exponential gives a correlation of exactly -1 in the log.
    // Anything appreciably short of that means the profile is not one,
    // and the slope would be a number rather than a measurement.
    let correlation = sxy / (sxx * syy).sqrt();
    if correlation > -0.999 {
        return Err(SolveError::NoConvergence { iters: steps, residual: correlation });
    }
    let alpha = -sxy / sxx;
    if alpha <= 0.0 {
        return Err(SolveError::NoConvergence { iters: steps, residual: alpha });
    }
    // Invert the numerical dispersion relation for the cutoff:
    // sin^2(w_c S/2)/S^2 = sin^2(ky/2) = sinh^2(alpha/2) + sin^2(w S/2)/S^2.
    let rhs = (0.5 * alpha).sinh().powi(2)
        + (0.5 * omega * courant).sin().powi(2) / (courant * courant);
    let arg = courant * rhs.sqrt();
    if !(0.0..=1.0).contains(&arg) {
        return Err(SolveError::NoConvergence { iters: steps, residual: arg });
    }
    Ok(2.0 / courant * arg.asin())
}

/// The numerical cutoff a parallel-plate guide of this width has on a
/// grid at this Courant number, `(2/S) arcsin(S sin(k_y/2))`.
///
/// The continuum answer is `m pi / a`; this is what the grid actually
/// gives, and it is always the smaller of the two.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a mode outside `1..width` or a
/// Courant number outside the plane limit.
pub fn waveguide_cutoff_numerical(
    width: usize,
    mode: usize,
    courant: f64,
) -> Result<f64, SolveError> {
    if width == 0 || mode == 0 || mode >= width {
        return Err(SolveError::InvalidArgument("the mode must lie in 1..width"));
    }
    if !courant.is_finite() || courant <= 0.0 || courant > 0.5f64.sqrt() * (1.0 + 1e-12) {
        return Err(SolveError::InvalidArgument("the Courant number exceeds the plane limit"));
    }
    let ky = std::f64::consts::PI * mode as f64 / width as f64;
    Ok(2.0 / courant * (courant * (0.5 * ky).sin()).asin())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

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
    fn the_matched_layer_absorbs_what_a_wall_reflects() {
        // A pulsed source, so that the field left in the interior after
        // it has stopped is entirely what came back. With the source
        // still running the interior is its own near field and says
        // nothing about the boundary at all.
        let (nx, ny) = (60usize, 60usize);
        let eps = vec![1.0; nx * ny];
        let s = 0.5f64.sqrt() * 0.99;
        let src = |step: usize| -> f64 {
            if step >= 100 {
                return 0.0;
            }
            let x = step as f64 / 100.0;
            0.5 * (1.0 - (std::f64::consts::TAU * x).cos())
                * (std::f64::consts::TAU * 0.07 * step as f64).sin()
        };
        let residual = |pml: usize| {
            let r =
                fdtd_2d_tm(&eps, (nx / 2, ny / 2), &src, nx, ny, 500, (pml, pml), s, 1e-6)
                    .unwrap();
            let mut peak: f64 = 0.0;
            for j in pml + 3..ny - pml - 3 {
                for i in pml + 3..nx - pml - 3 {
                    peak = peak.max(r.envelope[j * nx + i]);
                }
            }
            peak
        };
        let wall = residual(0);
        let thin = residual(4);
        let thick = residual(10);
        assert!(wall > 1e-2, "the conductor did not reflect: {wall}");
        assert!(wall / thin > 1e3, "four cells of layer were only {}x better", wall / thin);
        assert!(thick < thin, "a deeper layer absorbed less");
    }

    #[test]
    fn the_plane_scheme_respects_the_symmetry_of_its_own_grid() {
        // A source at the exact centre of a square vacuum box: the Yee
        // arrangement is symmetric under reflecting either axis and
        // under exchanging them, so the field must be too, bit for bit.
        // An index slip in the staggering breaks this immediately while
        // still producing a picture that looks like a wave.
        let n = 41usize;
        let eps = vec![1.0; n * n];
        let s = 0.5f64.sqrt() * 0.9;
        let src = |step: usize| -> f64 {
            let t = step as f64 - 20.0;
            (-t * t / 60.0).exp()
        };
        let r = fdtd_2d_tm(&eps, (n / 2, n / 2), &src, n, n, 120, (0, 0), s, 1e-6).unwrap();
        for j in 0..n {
            for i in 0..n {
                let v = r.ez[j * n + i];
                assert_eq!(v, r.ez[j * n + (n - 1 - i)], "not mirrored in x at ({i}, {j})");
                assert_eq!(v, r.ez[(n - 1 - j) * n + i], "not mirrored in y at ({i}, {j})");
                assert_eq!(v, r.ez[i * n + j], "not symmetric under transposition");
            }
        }
    }

    #[test]
    fn the_numerical_cutoff_sits_below_the_continuum_one_and_approaches_it() {
        // (2/S) arcsin(S sin(ky/2)) against m pi / a. The grid is always
        // a little slow, so its cutoff is always the lower of the two,
        // and the gap closes as the square of the cell size.
        let s = 0.5f64.sqrt() * 0.99;
        let mut previous = f64::INFINITY;
        for width in [8usize, 16, 32, 64] {
            let got = waveguide_cutoff_numerical(width, 1, s).unwrap();
            let continuum = PI / width as f64;
            assert!(got < continuum, "the grid cutoff was not below the continuum one");
            let relative = (continuum - got) / continuum;
            assert!(relative < previous, "refining did not close the gap");
            previous = relative;
        }
        assert!(previous < 1e-3, "sixty-four cells still left {previous}");
        // Higher modes cut off higher, wider guides lower.
        let a = waveguide_cutoff_numerical(20, 1, s).unwrap();
        let b = waveguide_cutoff_numerical(20, 2, s).unwrap();
        let c = waveguide_cutoff_numerical(40, 1, s).unwrap();
        assert!(b > a && c < a);
        assert!(waveguide_cutoff_numerical(10, 0, s).is_err());
        assert!(waveguide_cutoff_numerical(10, 10, s).is_err());
        assert!(waveguide_cutoff_numerical(10, 1, 1.0).is_err());
    }

    #[test]
    fn the_measured_evanescent_decay_gives_the_cutoff_back() {
        // Drive below cutoff, measure the decay, invert the *numerical*
        // dispersion relation. The answer that comes back is the grid's
        // own cutoff, which the simulation actually has, rather than the
        // continuum figure it is approximating.
        let s = 0.5f64.sqrt() * 0.99;
        let width = 16;
        let want = waveguide_cutoff_numerical(width, 1, s).unwrap();
        for frac in [0.5, 0.85] {
            let got = waveguide_cutoff_check_fdtd(width, 200, 1, frac * want, s, 6000).unwrap();
            assert!(
                (got - want).abs() < 5e-3 * want,
                "at {frac} of cutoff the measurement gave {got}, wanted {want}"
            );
        }
        // At or above cutoff there is nothing evanescent to measure, and
        // saying so beats returning a number.
        assert!(waveguide_cutoff_check_fdtd(width, 200, 1, want, s, 500).is_err());
        assert!(waveguide_cutoff_check_fdtd(width, 200, 1, 2.0 * want, s, 500).is_err());
    }

    #[test]
    fn the_plane_solver_refuses_impossible_arguments() {
        let eps = vec![1.0; 40 * 40];
        let quiet = |_: usize| 0.0;
        let s = 0.5f64.sqrt() * 0.9;
        assert!(fdtd_2d_tm(&eps[..16], (0, 0), &quiet, 4, 4, 5, (0, 0), s, 1e-6).is_err());
        assert!(fdtd_2d_tm(&eps[..100], (0, 0), &quiet, 40, 40, 5, (0, 0), s, 1e-6).is_err());
        assert!(fdtd_2d_tm(&eps, (99, 0), &quiet, 40, 40, 5, (0, 0), s, 1e-6).is_err());
        assert!(fdtd_2d_tm(&eps, (0, 0), &quiet, 40, 40, 5, (25, 0), s, 1e-6).is_err());
        assert!(fdtd_2d_tm(&eps, (0, 0), &quiet, 40, 40, 5, (0, 0), 0.9, 1e-6).is_err());
        assert!(fdtd_2d_tm(&eps, (0, 0), &quiet, 40, 40, 5, (0, 0), s, 0.0).is_err());
        assert!(fdtd_2d_tm(&eps, (0, 0), &quiet, 40, 40, 5, (0, 0), s, 1.5).is_err());
        let mut bad = eps.clone();
        bad[7] = -1.0;
        assert!(fdtd_2d_tm(&bad, (0, 0), &quiet, 40, 40, 5, (0, 0), s, 1e-6).is_err());
        assert!(waveguide_cutoff_check_fdtd(16, 20, 1, 0.05, s, 500).is_err());
        assert!(waveguide_cutoff_check_fdtd(2, 200, 1, 0.05, s, 500).is_err());
        assert!(waveguide_cutoff_check_fdtd(16, 200, 1, -1.0, s, 500).is_err());
        assert!(waveguide_cutoff_check_fdtd(16, 200, 1, 0.05, 1.0, 500).is_err());
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
