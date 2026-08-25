//! Properties of the finite-difference time domain module.
//!
//! Explicit time stepping is a setting where the exact statements and the
//! approximate ones are easy to confuse, so the tests keep them apart.
//!
//! *Exact.* The Yee leapfrog conserves a particular discrete energy to
//! the last bit in a closed lossless domain -- not the obvious sum of
//! squares, which wobbles forever, but the form whose magnetic term is
//! the product of the two half-steps straddling the electric one. At a
//! Courant number of exactly one in one dimension the update degenerates
//! into a shift, so a pulse translates bit for bit. The scheme is linear
//! in its source. And the Bloch dispersion relation behind the band gaps
//! is closed-form, so the quarter-wave stack's gap centres and widths
//! are analytic and can be checked to nine digits.
//!
//! *Approximate, but for a reason.* Reflection and transmission at a
//! dielectric interface approach the Fresnel coefficients as the pulse
//! is better resolved; the absorbing boundary leaks a little. These are
//! asserted with tolerances that say what the discretisation costs,
//! rather than with tolerances chosen to make them pass.
//!
//! *A threshold, not a slope.* The Courant condition is a hard boundary,
//! and the limit belongs to the fastest medium in the grid rather than
//! to vacuum -- a permittivity below one tightens it by exactly its
//! index.

use rust_physics_engine::fem::fdtd::{
    fdtd_1d, fdtd_2d_tm, fdtd_courant_check, fdtd_courant_check_2d,
    photonic_crystal_bandgap_1d, waveguide_cutoff_check_fdtd, waveguide_cutoff_numerical,
    Boundary1d,
};
use rust_physics_engine::monte_carlo::Rng;

const PI: f64 = std::f64::consts::PI;

/// A Hann burst of the given length, identically zero afterwards.
///
/// Compact support is what makes "after the source stops" an exact
/// statement: a Gaussian is still injecting something at every step, and
/// that something swamps a conservation test at machine precision.
fn burst(length: usize, amplitude: f64) -> impl Fn(usize) -> f64 {
    move |step: usize| {
        if step >= length {
            0.0
        } else {
            amplitude * 0.5 * (1.0 - (std::f64::consts::TAU * step as f64 / length as f64).cos())
        }
    }
}

#[test]
fn prop_the_leapfrog_conserves_its_energy_to_the_last_bit() {
    // Exact for any permittivity profile and any admissible Courant
    // number -- the cancellation in the update is algebraic, not
    // asymptotic. The obvious sum of squares is not conserved, and
    // asserting that one would be asserting a tolerance instead of an
    // invariant.
    let mut rng = Rng::new(0x2ce4_71b0);
    for _ in 0..25 {
        let n = 120 + (rng.next_u64() % 80) as usize;
        let eps: Vec<f64> = (0..n).map(|_| 1.0 + 3.0 * rng.next_f64()).collect();
        let courant = 0.2 + 0.8 * rng.next_f64();
        let src = burst(50, 0.5 + rng.next_f64());
        let steps = 150;
        let r = fdtd_1d(&eps, &src, n / 2, courant, steps, Boundary1d::Conductor).unwrap();
        let reference = r.energy(&eps, 60).unwrap();
        assert!(reference > 1e-3, "there was no energy to conserve");
        for step in 60..steps {
            let u = r.energy(&eps, step).unwrap();
            assert!(
                (u - reference).abs() < 1e-11 * reference,
                "step {step} drifted to {u} from {reference}"
            );
        }
        let naive = |k: usize| -> f64 {
            0.5 * r.e[k].iter().zip(eps.iter()).map(|(v, e)| e * v * v).sum::<f64>()
                + 0.5 * r.h[k].iter().map(|v| v * v).sum::<f64>()
        };
        let spread = (60..steps).map(naive).fold(f64::NEG_INFINITY, f64::max)
            - (60..steps).map(naive).fold(f64::INFINITY, f64::min);
        assert!(spread > 1e-10 * reference, "the naive form was conserved too");
    }
}

#[test]
fn prop_the_magic_time_step_translates_a_pulse_bit_for_bit() {
    // Only in one dimension, and only at a Courant number of exactly
    // one: there the numerical dispersion relation is the exact one and
    // the update is a shift. Any other Courant number is not, which the
    // second half of the test confirms so that the first is not passing
    // for some duller reason.
    let mut rng = Rng::new(0x59d0_c3f7);
    for _ in 0..20 {
        let n = 200;
        let eps = vec![1.0; n];
        let src = burst(40, 0.3 + rng.next_f64());
        let r = fdtd_1d(&eps, &src, 100, 1.0, 90, Boundary1d::Conductor).unwrap();
        let mut worst: f64 = 0.0;
        for step in 60..85 {
            for i in 130..190 {
                worst = worst.max((r.e[step + 1][i] - r.e[step][i - 1]).abs());
            }
        }
        assert_eq!(worst, 0.0, "the pulse did not translate exactly");
        let peak = r.e[70].iter().cloned().fold(0.0f64, f64::max);
        assert!(peak > 0.1, "there was no pulse to translate: {peak}");
        // Below the magic step the scheme disperses, so the same
        // comparison fails by a visible margin.
        let slow = fdtd_1d(&eps, &src, 100, 0.6, 90, Boundary1d::Conductor).unwrap();
        let mut drift: f64 = 0.0;
        for i in 130..190 {
            drift = drift.max((slow.e[71][i] - slow.e[70][i - 1]).abs());
        }
        assert!(drift > 1e-6 * peak, "a Courant number of 0.6 translated exactly too");
    }
}

#[test]
fn prop_the_march_is_linear_in_its_source() {
    // Maxwell's equations are linear and so is the scheme. Nothing in
    // the update or in either boundary condition is allowed to be
    // affine.
    let mut rng = Rng::new(0x74a2_1e58);
    for _ in 0..20 {
        let n = 100;
        let eps: Vec<f64> = (0..n).map(|_| 1.0 + 2.0 * rng.next_f64()).collect();
        let courant = 0.3 + 0.6 * rng.next_f64();
        let (a, b) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let s1 = burst(30, a);
        let s2 = burst(45, b);
        let boundary =
            if rng.next_f64() < 0.5 { Boundary1d::Mur } else { Boundary1d::Conductor };
        let run = |s: &dyn Fn(usize) -> f64| {
            fdtd_1d(&eps, s, 40, courant, 80, boundary).unwrap()
        };
        let r1 = run(&s1);
        let r2 = run(&s2);
        let both = run(&|k| s1(k) + s2(k));
        for step in [10usize, 40, 80] {
            for i in 0..n {
                let want = r1.e[step][i] + r2.e[step][i];
                assert!(
                    (both.e[step][i] - want).abs() < 1e-12 * (1.0 + want.abs()),
                    "step {step} cell {i}"
                );
            }
        }
    }
}

#[test]
fn prop_a_dielectric_interface_gives_the_fresnel_coefficients() {
    // At normal incidence the reflected amplitude is
    // (n1 - n2)/(n1 + n2) and the transmitted one 2 n1/(n1 + n2). The
    // sign is half the content: reflection off a denser medium inverts
    // the field, and a scheme with a sign error would still conserve
    // energy.
    let mut rng = Rng::new(0x0a37_92d1);
    for _ in 0..20 {
        let n = 400;
        let n2 = 1.4 + 1.6 * rng.next_f64();
        let mut eps = vec![1.0; n];
        for e in eps.iter_mut().skip(n / 2) {
            *e = n2 * n2;
        }
        let src = burst(60, 1.0);
        let r = fdtd_1d(&eps, &src, 60, 1.0, 260, Boundary1d::Mur).unwrap();
        let extreme = |v: &[f64]| {
            v.iter().copied().fold(0.0f64, |a, x| if x.abs() > a.abs() { x } else { a })
        };
        let incident = extreme(&r.e[120][..190]);
        let reflected = extreme(&r.e[250][..190]);
        let transmitted = extreme(&r.e[250][210..]);
        assert!(incident > 0.1, "no incident pulse: {incident}");
        let want_r = (1.0 - n2) / (1.0 + n2);
        let want_t = 2.0 / (1.0 + n2);
        assert!(
            (reflected / incident - want_r).abs() < 0.03,
            "reflection {} against {want_r}",
            reflected / incident
        );
        assert!(
            (transmitted / incident - want_t).abs() < 0.04,
            "transmission {} against {want_t}",
            transmitted / incident
        );
        // Reflection off a denser medium inverts the sign, always.
        assert!(reflected < 0.0, "the reflection did not invert");
    }
}

#[test]
fn prop_the_absorbing_boundary_beats_the_wall_by_orders_of_magnitude() {
    let mut rng = Rng::new(0x3b8f_04ac);
    for _ in 0..15 {
        let n = 300;
        let eps = vec![1.0; n];
        let src = burst(50, 0.5 + rng.next_f64());
        let residual = |b| {
            let r = fdtd_1d(&eps, &src, n / 2, 1.0, 320, b).unwrap();
            r.e[320].iter().cloned().fold(0.0f64, |a, v| a.max(v.abs()))
        };
        let mur = residual(Boundary1d::Mur);
        let wall = residual(Boundary1d::Conductor);
        assert!(wall > 0.05, "the wall did not reflect: {wall}");
        assert!(wall / mur > 500.0, "the absorber was only {}x better", wall / mur);
    }
}

#[test]
fn prop_the_stability_limit_belongs_to_the_fastest_medium() {
    // A permittivity below one has a phase speed above c, so it tightens
    // the bound by exactly its index. The threshold is sharp: the run is
    // accepted at the limit and refused just past it.
    let mut rng = Rng::new(0x18cd_66e2);
    for _ in 0..30 {
        let fastest = 0.1 + 0.8 * rng.next_f64();
        let mut eps: Vec<f64> = (0..50).map(|_| 1.0 + rng.next_f64()).collect();
        eps[20 + (rng.next_u64() % 10) as usize] = fastest;
        let limit = fastest.sqrt();
        let src = burst(10, 1.0);
        assert!(fdtd_1d(&eps, &src, 5, limit * 0.999, 5, Boundary1d::Mur).is_ok());
        assert!(fdtd_1d(&eps, &src, 5, limit * 1.01, 5, Boundary1d::Mur).is_err());
        // Vacuum alone would have allowed anything up to one.
        if limit < 0.95 {
            assert!(fdtd_courant_check(1.0, 1.0, 1.0));
            assert!(fdtd_1d(&eps, &src, 5, 1.0, 5, Boundary1d::Mur).is_err());
        }
    }
}

#[test]
fn prop_the_courant_checks_agree_with_their_own_formulas() {
    let mut rng = Rng::new(0x6d43_9f01);
    for _ in 0..60 {
        let dx = 0.1 + 3.0 * rng.next_f64();
        let dy = 0.1 + 3.0 * rng.next_f64();
        let c = 0.2 + 2.0 * rng.next_f64();
        let limit_1d = dx / c;
        assert!(fdtd_courant_check(dx, limit_1d * 0.999, c));
        assert!(!fdtd_courant_check(dx, limit_1d * 1.001, c));
        let limit_2d = 1.0 / (1.0 / (dx * dx) + 1.0 / (dy * dy)).sqrt() / c;
        assert!(fdtd_courant_check_2d(dx, dy, limit_2d * 0.999, c));
        assert!(!fdtd_courant_check_2d(dx, dy, limit_2d * 1.001, c));
        // Two dimensions is always stricter than one, and on a square
        // grid it is stricter by exactly sqrt(2).
        assert!(limit_2d < limit_1d);
        let square = 1.0 / (2.0 / (dx * dx)).sqrt() / c;
        assert!((square * 2.0f64.sqrt() - limit_1d).abs() < 1e-12 * limit_1d);
        // Nonsense is rejected rather than interpreted.
        assert!(!fdtd_courant_check(-dx, 1.0, c));
        assert!(!fdtd_courant_check_2d(dx, dy, 1.0, f64::INFINITY));
    }
}

#[test]
fn prop_the_quarter_wave_stack_matches_its_analytic_gaps() {
    // Layers of equal optical thickness put a gap centred exactly on the
    // design frequency and on every odd multiple of it, with relative
    // width (4 / (m pi)) arcsin(|na - nb| / (na + nb)). The even
    // multiples are closed, because there each layer is a half wave and
    // the period is invisible.
    let mut rng = Rng::new(0x4e91_c7b6);
    for _ in 0..25 {
        let na = 1.0 + rng.next_f64();
        let nb = na + 0.3 + 2.0 * rng.next_f64();
        let (ea, eb) = (na * na, nb * nb);
        let da = 0.5 + rng.next_f64();
        let db = da * na / nb;
        let w0 = PI / (2.0 * na * da);
        let gaps = photonic_crystal_bandgap_1d(ea, eb, da, db, 4.5 * w0, 6000).unwrap();
        assert!(gaps.len() >= 2, "found only {} gaps", gaps.len());
        for (m, &(lo, hi)) in [(1.0, &gaps[0]), (3.0, &gaps[1])] {
            let centre = 0.5 * (lo + hi);
            assert!((centre / w0 - m).abs() < 1e-9, "gap centred at {} w0", centre / w0);
            let want = 4.0 / (m * PI) * ((nb - na) / (nb + na)).asin();
            let got = (hi - lo) / centre;
            assert!((got - want).abs() < 1e-8, "gap {m}: width {got}, theory {want}");
        }
        assert!(
            !gaps.iter().any(|&(lo, hi)| lo < 2.0 * w0 && hi > 2.0 * w0),
            "the even-order gap did not close"
        );
        // More contrast, wider gap: arcsin is increasing in the index
        // mismatch and nothing else enters the formula.
        let wider = photonic_crystal_bandgap_1d(
            ea,
            (nb + 1.0) * (nb + 1.0),
            da,
            da * na / (nb + 1.0),
            4.5 * w0,
            6000,
        )
        .unwrap();
        let relative = |g: &(f64, f64)| (g.1 - g.0) / (0.5 * (g.0 + g.1));
        assert!(relative(&wider[0]) > relative(&gaps[0]), "more contrast gave a narrower gap");
    }
}

#[test]
fn prop_a_stack_is_the_same_crystal_however_it_is_described() {
    // Swapping which layer is called `a` describes the same periodic
    // medium, so the gaps must be identical; scaling every thickness
    // scales every gap edge by the reciprocal; and a stack of one
    // material has no gaps at all, since the mixing factor is then
    // exactly one and the Bloch trace is a plain cosine.
    let mut rng = Rng::new(0x2f60_bb34);
    for _ in 0..25 {
        let ea = 1.0 + 3.0 * rng.next_f64();
        let eb = 1.0 + 3.0 * rng.next_f64();
        let da = 0.4 + rng.next_f64();
        let db = 0.4 + rng.next_f64();
        let ceiling = 25.0;
        let forward = photonic_crystal_bandgap_1d(ea, eb, da, db, ceiling, 8000).unwrap();
        let swapped = photonic_crystal_bandgap_1d(eb, ea, db, da, ceiling, 8000).unwrap();
        assert_eq!(forward.len(), swapped.len(), "swapping the layers changed the gap count");
        for (a, b) in forward.iter().zip(swapped.iter()) {
            assert!((a.0 - b.0).abs() < 1e-8 * a.0.max(1.0));
            assert!((a.1 - b.1).abs() < 1e-8 * a.1.max(1.0));
        }
        let s = 1.5 + rng.next_f64();
        let stretched =
            photonic_crystal_bandgap_1d(ea, eb, s * da, s * db, ceiling / s, 8000).unwrap();
        assert_eq!(forward.len(), stretched.len());
        for (a, b) in forward.iter().zip(stretched.iter()) {
            assert!((a.0 - s * b.0).abs() < 1e-7 * a.0.max(1.0), "{} vs {}", a.0, s * b.0);
            assert!((a.1 - s * b.1).abs() < 1e-7 * a.1.max(1.0));
        }
        assert!(photonic_crystal_bandgap_1d(ea, ea, da, db, ceiling, 8000).unwrap().is_empty());
    }
}

/// A comfortable Courant number for the plane: below the limit, since
/// unlike one dimension there is no magic value there.
const PLANE_COURANT: f64 = 0.7;

/// A pulsed drive of the given length at the given frequency in cycles
/// per step, identically zero once it has finished.
fn pulsed(length: usize, freq: f64, amplitude: f64) -> impl Fn(usize) -> f64 {
    move |step: usize| {
        if step >= length {
            return 0.0;
        }
        let x = step as f64 / length as f64;
        amplitude
            * 0.5
            * (1.0 - (std::f64::consts::TAU * x).cos())
            * (std::f64::consts::TAU * freq * step as f64).sin()
    }
}

#[test]
fn prop_the_plane_scheme_keeps_the_symmetry_of_its_grid() {
    // A source at the exact centre of a square vacuum box. The Yee
    // arrangement is symmetric under reflecting either axis and under
    // exchanging them, so the field is too -- bit for bit, since nothing
    // in the update breaks it. An index slip in the staggering shows up
    // here while still producing something that looks like a wave.
    let mut rng = Rng::new(0x11f7_36ea);
    for _ in 0..12 {
        // Odd, so the centre is a cell.
        let n = 21 + 2 * (rng.next_u64() % 6) as usize;
        let eps = vec![1.0; n * n];
        let src = pulsed(30 + (rng.next_u64() % 30) as usize, 0.05 + 0.1 * rng.next_f64(), 1.0);
        let pml = (rng.next_u64() % 4) as usize;
        let r = fdtd_2d_tm(
            &eps,
            (n / 2, n / 2),
            &src,
            n,
            n,
            90,
            (pml, pml),
            PLANE_COURANT,
            1e-6,
        )
        .unwrap();
        let mut nonzero = false;
        for j in 0..n {
            for i in 0..n {
                let v = r.ez[j * n + i];
                nonzero |= v != 0.0;
                assert_eq!(v, r.ez[j * n + (n - 1 - i)], "not mirrored in x");
                assert_eq!(v, r.ez[(n - 1 - j) * n + i], "not mirrored in y");
                assert_eq!(v, r.ez[i * n + j], "not symmetric under transposition");
            }
        }
        assert!(nonzero, "the field never left the source");
    }
}

#[test]
fn prop_the_plane_march_is_linear_in_its_source() {
    let mut rng = Rng::new(0x4c02_a9d5);
    for _ in 0..12 {
        let (nx, ny) = (30, 26);
        let eps: Vec<f64> = (0..nx * ny).map(|_| 1.0 + 2.0 * rng.next_f64()).collect();
        let pml = (rng.next_u64() % 5) as usize;
        let s1 = pulsed(25, 0.06, 2.0 * rng.next_f64() - 1.0);
        let s2 = pulsed(40, 0.09, 2.0 * rng.next_f64() - 1.0);
        // The fastest medium sets the limit, and here it is vacuum.
        let courant = 0.6;
        let run = |src: &dyn Fn(usize) -> f64| {
            fdtd_2d_tm(&eps, (7, 9), src, nx, ny, 70, (pml, pml), courant, 1e-6).unwrap()
        };
        let a = run(&s1);
        let b = run(&s2);
        let both = run(&|k| s1(k) + s2(k));
        for i in 0..nx * ny {
            let want = a.ez[i] + b.ez[i];
            assert!((both.ez[i] - want).abs() < 1e-12 * (1.0 + want.abs()), "cell {i}");
        }
    }
}

#[test]
fn prop_a_matched_layer_absorbs_what_a_conductor_returns() {
    // With the drive switched off, everything left in the interior came
    // back from the boundary. A conductor returns essentially all of it;
    // a graded layer a few cells deep returns a thousandth or less, and
    // a deeper one less still.
    let mut rng = Rng::new(0x7ae1_2c93);
    for _ in 0..10 {
        let (nx, ny) = (56, 56);
        let eps = vec![1.0; nx * ny];
        let src = pulsed(90, 0.05 + 0.05 * rng.next_f64(), 0.5 + rng.next_f64());
        let residual = |pml: usize| {
            let r = fdtd_2d_tm(
                &eps,
                (nx / 2, ny / 2),
                &src,
                nx,
                ny,
                460,
                (pml, pml),
                PLANE_COURANT,
                1e-6,
            )
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
        assert!(wall > 1e-3, "the conductor returned nothing: {wall}");
        assert!(wall / thin > 500.0, "four cells were only {}x better", wall / thin);
        assert!(thick <= thin, "a deeper layer absorbed less");
    }
}

#[test]
fn prop_the_numerical_cutoff_is_exactly_where_the_decay_vanishes() {
    // omega_c = (2/S) arcsin(S sin(ky/2)) is defined as the frequency at
    // which the evanescent decay reaches zero, so substituting it back
    // into the numerical dispersion relation must close to rounding.
    // The continuum value m pi / a does not, and the gap between them is
    // second order in the cell size -- with the grid always the slower
    // of the two.
    let mut rng = Rng::new(0x63b8_c410);
    for _ in 0..40 {
        let width = 6 + (rng.next_u64() % 60) as usize;
        let mode = 1 + (rng.next_u64() % (width as u64 - 1)) as usize;
        let courant = 0.2 + 0.5 * rng.next_f64();
        let wc = waveguide_cutoff_numerical(width, mode, courant).unwrap();
        let ky = PI * mode as f64 / width as f64;
        let lhs = (0.5 * wc * courant).sin().powi(2) / (courant * courant);
        let rhs = (0.5 * ky).sin().powi(2);
        assert!((lhs - rhs).abs() < 1e-12 * rhs, "the cutoff does not zero the decay");
        let continuum = ky;
        assert!(wc < continuum, "the grid was not the slower of the two");
        // Second order in the cell size: doubling the resolution at the
        // same physical width quarters the shortfall.
        let coarse = 1.0 - wc / continuum;
        let fine_width = 2 * width;
        let fine_mode = mode;
        let fine = 1.0
            - waveguide_cutoff_numerical(fine_width, fine_mode, courant).unwrap()
                / (PI * fine_mode as f64 / fine_width as f64);
        assert!(fine < coarse, "refining did not close the gap");
        // The shortfall is (ky/2)^2 (1 - S^2) / 6 to leading order, so
        // halving ky quarters it -- but that expansion is a cubic one
        // and only applies while the mode is well resolved. Three half
        // waves across six cells is not, and there the shortfall is
        // simply large rather than quadratically small.
        if ky < 0.4 {
            let ratio = coarse / fine;
            assert!((ratio - 4.0).abs() < 0.6, "the shortfall fell by {ratio}, not 4");
            let predicted = (0.5 * ky).powi(2) * (1.0 - courant * courant) / 6.0;
            assert!(
                (coarse - predicted).abs() < 0.1 * predicted,
                "shortfall {coarse} against the predicted {predicted}"
            );
        }
    }
}

#[test]
fn prop_the_measured_decay_recovers_the_grids_own_cutoff() {
    // Drive a guide below cutoff, fit the evanescent decay, invert the
    // numerical dispersion relation. What comes back is the cutoff the
    // simulation actually has, to a couple of parts in a thousand,
    // across widths, modes and drive frequencies. Matching the
    // *numerical* cutoff rather than the continuum one is the point:
    // the two differ by more than this tolerance at these widths, so
    // the test would fail against the textbook figure.
    let mut rng = Rng::new(0x2d94_51fb);
    let s = 0.5f64.sqrt() * 0.99;
    for _ in 0..10 {
        // Modes two and three in a well-resolved guide. Mode one is
        // deliberately not used here: at these widths the numerical and
        // continuum cutoffs differ by about as much as the measurement
        // error, so the comparison below could not tell them apart and
        // would be asserting noise.
        let width = 16 + 2 * (rng.next_u64() % 3) as usize;
        let mode = 2 + (rng.next_u64() % 2) as usize;
        let frac = 0.4 + 0.35 * rng.next_f64();
        let want = waveguide_cutoff_numerical(width, mode, s).unwrap();
        let got = waveguide_cutoff_check_fdtd(width, 200, mode, frac * want, s, 7000).unwrap();
        assert!(
            (got - want).abs() < 5e-3 * want,
            "width {width} mode {mode} at {frac}: got {got}, wanted {want}"
        );
        // The continuum figure is further off than the measurement is,
        // so the measurement really is picking out the grid's value.
        let continuum = PI * mode as f64 / width as f64;
        assert!(
            (got - want).abs() < (continuum - want).abs(),
            "the measurement was no closer to the grid cutoff than the continuum one is"
        );
        // At or above cutoff there is no decay, and saying so beats
        // returning a number.
        assert!(waveguide_cutoff_check_fdtd(width, 200, mode, want, s, 400).is_err());
    }
}
