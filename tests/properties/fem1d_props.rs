//! Properties of the one-dimensional finite element module.
//!
//! Finite elements are unusually well supplied with theorems that are
//! sharp rather than asymptotic, and those are the ones worth testing.
//!
//! *Optimality.* Galerkin orthogonality makes the discrete solution the
//! orthogonal projection of the true one in the energy inner product, so
//! its energy-norm error is no larger than that of **any** other function
//! in the space -- the nodal interpolant included, with a constant of
//! exactly one. Equivalently it minimises the energy functional, so
//! refining a mesh, or raising the polynomial degree on the same mesh,
//! can only lower the computed energy. Both are inequalities with no
//! fudge factor, and a sign error anywhere in the assembly violates them.
//!
//! *Exactness.* A solution already in the element space must be returned
//! untouched, and for the Laplacian specifically the nodal values are
//! exact whatever the mesh, because the Green's function of a node is
//! itself piecewise linear.
//!
//! *Structure.* The problem is linear, so superposition holds; the
//! operator is symmetric under reflecting the interval, so the solution
//! is; testing against the constant function gives an exact discrete
//! conservation law; and with no reaction term the stiffness matrix is an
//! M-matrix, which is what a discrete maximum principle amounts to.
//!
//! *Rates.* Everything above holds on a single mesh. The convergence
//! orders -- `h^2` and `h^3` in `L2`, one less in `H1` -- are what say the
//! space is the one it claims to be, and a quadratic element that
//! converges at second order is a quadratic element with a broken shape
//! function.

use rust_physics_engine::error::SolveError;
use rust_physics_engine::fem::fem1d::{
    convergence_rate, fem_1d_error_h1, fem_1d_error_h1_seminorm, fem_1d_error_l2, fem_1d_general,
    fem_1d_poisson, fem_1d_quadratic, Bc, Fem1dSolution,
};
use rust_physics_engine::monte_carlo::Rng;

const GAUSS_X: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683_1,
    0.0,
    0.538_469_310_105_683_1,
    0.906_179_845_938_664,
];
const GAUSS_W: [f64; 5] = [
    0.236_926_885_056_189_1,
    0.478_628_670_499_366_5,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_5,
    0.236_926_885_056_189_1,
];

/// Element-by-element five-point Gauss, matching what the assembly uses,
/// so that an identity the assembly satisfies exactly comes out exact
/// here too.
fn integrate(a: f64, b: f64, elements: usize, g: &dyn Fn(f64) -> f64) -> f64 {
    let h = (b - a) / elements as f64;
    let mut total = 0.0;
    for e in 0..elements {
        let left = a + e as f64 * h;
        for (&xi, &w) in GAUSS_X.iter().zip(GAUSS_W.iter()) {
            total += 0.5 * w * h * g(left + 0.5 * (xi + 1.0) * h);
        }
    }
    total
}

/// A random polynomial with coefficients in `[-1, 1]`, and its first two
/// derivatives. Polynomial data keeps every quadrature in the assembly
/// exact, so the exactness properties are testable at machine precision
/// rather than at quadrature precision.
fn poly(rng: &mut Rng, degree: usize) -> Vec<f64> {
    (0..=degree).map(|_| 2.0 * rng.next_f64() - 1.0).collect()
}

fn eval(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &a| acc * x + a)
}

fn deriv(c: &[f64]) -> Vec<f64> {
    c.iter().enumerate().skip(1).map(|(k, &a)| k as f64 * a).collect()
}

/// A strictly positive coefficient built from a random polynomial by
/// shifting it clear of zero.
fn positive(rng: &mut Rng, degree: usize) -> Vec<f64> {
    let mut c = poly(rng, degree);
    c[0] += 3.0;
    c
}

/// The energy functional the Ritz method minimises,
/// `J(v) = (1/2) integral p v'^2 - integral f v`.
fn energy(
    u_h: &Fem1dSolution,
    p: &dyn Fn(f64) -> f64,
    f: &dyn Fn(f64) -> f64,
    elements: usize,
) -> f64 {
    let quad = elements.max(u_h.elements());
    let stiff = integrate(u_h.a, u_h.b, quad, &|x| {
        let d = u_h.eval_derivative(x);
        p(x) * d * d
    });
    let load = integrate(u_h.a, u_h.b, quad, &|x| f(x) * u_h.eval(x));
    0.5 * stiff - load
}

/// The nodal interpolant of `u` on the same mesh as `u_h`.
fn interpolant(u_h: &Fem1dSolution, u: &dyn Fn(f64) -> f64) -> Fem1dSolution {
    let values = u_h.nodes().iter().map(|&x| u(x)).collect();
    Fem1dSolution::new(u_h.a, u_h.b, u_h.degree, values).unwrap()
}

#[test]
fn prop_the_error_is_orthogonal_to_everything_representable() {
    // Galerkin orthogonality, stated as the Pythagoras identity it is
    // equivalent to: for *any* v_h in the space with the right boundary
    // values,
    //
    //     ||u - v_h||_a^2 = ||u - u_h||_a^2 + ||u_h - v_h||_a^2.
    //
    // Cea's lemma is the corollary got by dropping the last term, so
    // testing the identity tests more than the inequality does -- and it
    // is an equality, which an inequality satisfied by accident is not.
    //
    // The variable coefficient here is worth a word. Linear elements
    // have a constant derivative on each element, so the five-point
    // quadrature in the assembly replaces p by its element average
    // exactly. That changes nothing for two functions in the space, so
    // the discrete bilinear form still agrees with the true one on
    // V_h x V_h, and the discrete solution is the true a-orthogonal
    // projection rather than an approximation of one.
    let mut rng = Rng::new(0x5f3e_1a77);
    let mut smallest_gap = f64::INFINITY;
    for _ in 0..40 {
        let pc = positive(&mut rng, 2);
        let uc = poly(&mut rng, 4);
        let duc = deriv(&uc);
        let dduc = deriv(&duc);
        let dpc = deriv(&pc);
        let p = |x: f64| eval(&pc, x);
        let u = |x: f64| eval(&uc, x);
        let du = |x: f64| eval(&duc, x);
        // -(p u')' = -(p' u' + p u'').
        let f = |x: f64| -(eval(&dpc, x) * eval(&duc, x) + eval(&pc, x) * eval(&dduc, x));
        let n = 6;
        let sol = Fem1dSolution::new(
            0.0,
            1.0,
            1,
            fem_1d_general(
                &p,
                &|_| 0.0,
                &f,
                0.0,
                1.0,
                (Bc::Dirichlet(u(0.0)), Bc::Dirichlet(u(1.0))),
                n,
            )
            .unwrap(),
        )
        .unwrap();
        // The energy norm carries p, so weight the seminorm by it.
        let err = |s: &Fem1dSolution| {
            integrate(0.0, 1.0, 4 * n, &|x| {
                let e = du(x) - s.eval_derivative(x);
                p(x) * e * e
            })
        };
        let gap = |s: &Fem1dSolution| {
            let d: Vec<f64> =
                sol.values.iter().zip(s.values.iter()).map(|(a, b)| a - b).collect();
            let d = Fem1dSolution::new(0.0, 1.0, 1, d).unwrap();
            integrate(0.0, 1.0, 4 * n, &|x| {
                let g = d.eval_derivative(x);
                p(x) * g * g
            })
        };
        let best = err(&sol);
        // Against the nodal interpolant, and against random members of
        // the space -- the identity holds for every one of them.
        let mut candidates = vec![interpolant(&sol, &u)];
        for _ in 0..4 {
            let mut c = sol.clone();
            for value in c.values.iter_mut().take(n).skip(1) {
                *value += 0.5 * (2.0 * rng.next_f64() - 1.0);
            }
            candidates.push(c);
        }
        for c in &candidates {
            let total = err(c);
            let side = gap(c);
            assert!(
                (total - best - side).abs() < 1e-10 * (1.0 + total),
                "orthogonality failed: {total} vs {best} + {side}"
            );
            // Cea's lemma follows, and is worth stating separately
            // because it is the statement with the physical content.
            assert!(best <= total * (1.0 + 1e-12), "the projection was not the best fit");
        }
        // The identity would also hold if the interpolant and the
        // projection coincided, so confirm they do not.
        smallest_gap = smallest_gap.min(gap(&candidates[0]).sqrt() / best.sqrt());
    }
    assert!(
        smallest_gap > 1e-4,
        "the projection never moved off the interpolant (closest {smallest_gap})"
    );
}

#[test]
fn prop_quadratic_elements_also_beat_their_interpolant() {
    // The same optimality for the plain Laplacian at degree two, where
    // the interpolant and the solution differ at the midsides.
    let mut rng = Rng::new(0x21ab_44c1);
    for _ in 0..30 {
        let uc = poly(&mut rng, 6);
        let duc = deriv(&uc);
        let dduc = deriv(&duc);
        let u = |x: f64| eval(&uc, x);
        let du = |x: f64| eval(&duc, x);
        let f = |x: f64| -eval(&dduc, x);
        let n = 4;
        let v = fem_1d_quadratic(
            &|_| 1.0,
            &|_| 0.0,
            &f,
            0.0,
            1.0,
            (Bc::Dirichlet(u(0.0)), Bc::Dirichlet(u(1.0))),
            n,
        )
        .unwrap();
        let sol = Fem1dSolution::new(0.0, 1.0, 2, v).unwrap();
        let fem = fem_1d_error_h1_seminorm(&sol, &du);
        let lag = fem_1d_error_h1_seminorm(&interpolant(&sol, &u), &du);
        assert!(fem <= lag * (1.0 + 1e-9), "fem {fem} was worse than interpolant {lag}");
    }
}

#[test]
fn prop_the_solution_minimises_the_energy_over_the_space() {
    // The Ritz characterisation. Perturbing the discrete solution in any
    // direction that respects the Dirichlet data must raise the energy,
    // and because the functional is quadratic the rise is exactly the
    // energy norm of the perturbation.
    let mut rng = Rng::new(0x77c1_0e23);
    for _ in 0..30 {
        let fc = poly(&mut rng, 3);
        let f = |x: f64| eval(&fc, x);
        let n = 7;
        let v =
            fem_1d_poisson(&f, 0.0, 1.0, (Bc::Dirichlet(0.3), Bc::Dirichlet(-0.4)), n).unwrap();
        let sol = Fem1dSolution::new(0.0, 1.0, 1, v).unwrap();
        let base = energy(&sol, &|_| 1.0, &f, 4 * n);
        for _ in 0..5 {
            let mut bumped = sol.clone();
            // Interior nodes only: the boundary values are prescribed.
            for value in bumped.values.iter_mut().take(n).skip(1) {
                *value += 0.4 * (2.0 * rng.next_f64() - 1.0);
            }
            let raised = energy(&bumped, &|_| 1.0, &f, 4 * n);
            assert!(raised > base, "a perturbation lowered the energy: {raised} < {base}");
            // The excess is exactly half the energy norm of the
            // difference, which is what "quadratic functional" means.
            let diff: Vec<f64> =
                bumped.values.iter().zip(sol.values.iter()).map(|(a, b)| a - b).collect();
            let d = Fem1dSolution::new(0.0, 1.0, 1, diff).unwrap();
            let half = 0.5
                * integrate(0.0, 1.0, 4 * n, &|x| {
                    let g = d.eval_derivative(x);
                    g * g
                });
            assert!((raised - base - half).abs() < 1e-10 * (1.0 + half));
        }
    }
}

#[test]
fn prop_poisson_is_nodally_exact_at_every_mesh_size() {
    // Not an asymptotic statement: three elements are as exact at their
    // nodes as three hundred.
    let mut rng = Rng::new(0x1c9e_b0d5);
    for _ in 0..40 {
        let uc = poly(&mut rng, 5);
        let dduc = deriv(&deriv(&uc));
        let u = |x: f64| eval(&uc, x);
        let n = 2 + (rng.next_u64() % 12) as usize;
        let v = fem_1d_poisson(
            &|x: f64| -eval(&dduc, x),
            -1.0,
            2.0,
            (Bc::Dirichlet(u(-1.0)), Bc::Dirichlet(u(2.0))),
            n,
        )
        .unwrap();
        for (i, got) in v.iter().enumerate() {
            let x = -1.0 + 3.0 * i as f64 / n as f64;
            assert!((got - u(x)).abs() < 1e-11, "n={n} node {i} off by {}", got - u(x));
        }
    }
}

#[test]
fn prop_quadratic_elements_are_exact_at_vertices_only() {
    // The vertex Green's function is piecewise linear and lies in the
    // quadratic space; the midside one kinks inside an element and does
    // not. So vertices are exact and midsides merely accurate.
    let mut rng = Rng::new(0x3e77_9a01);
    for _ in 0..25 {
        let uc = poly(&mut rng, 6);
        let dduc = deriv(&deriv(&uc));
        let u = |x: f64| eval(&uc, x);
        let n = 5;
        let v = fem_1d_quadratic(
            &|_| 1.0,
            &|_| 0.0,
            &|x: f64| -eval(&dduc, x),
            0.0,
            1.0,
            (Bc::Dirichlet(u(0.0)), Bc::Dirichlet(u(1.0))),
            n,
        )
        .unwrap();
        let err = |i: usize| (v[i] - u(i as f64 / (2 * n) as f64)).abs();
        let vertex = (0..=2 * n).step_by(2).map(err).fold(0.0, f64::max);
        let midside = (1..2 * n).step_by(2).map(err).fold(0.0, f64::max);
        assert!(vertex < 1e-12, "vertices off by {vertex}");
        assert!(midside > 1e2 * vertex.max(1e-16), "midsides were exact too");
    }
}

#[test]
fn prop_a_solution_already_in_the_space_comes_back_untouched() {
    // The patch test, at both degrees, with the exact solution chosen to
    // be representable: linear for P1 and quadratic for P2.
    let mut rng = Rng::new(0x9b2c_7710);
    for _ in 0..30 {
        let lc = poly(&mut rng, 1);
        let n = 3 + (rng.next_u64() % 6) as usize;
        let v = fem_1d_poisson(
            &|_| 0.0,
            0.0,
            2.0,
            (Bc::Dirichlet(eval(&lc, 0.0)), Bc::Dirichlet(eval(&lc, 2.0))),
            n,
        )
        .unwrap();
        for (i, got) in v.iter().enumerate() {
            let x = 2.0 * i as f64 / n as f64;
            assert!((got - eval(&lc, x)).abs() < 1e-12);
        }
        let qc = poly(&mut rng, 2);
        let dd = deriv(&deriv(&qc));
        let v2 = fem_1d_quadratic(
            &|_| 1.0,
            &|_| 0.0,
            &|x: f64| -eval(&dd, x),
            0.0,
            2.0,
            (Bc::Dirichlet(eval(&qc, 0.0)), Bc::Dirichlet(eval(&qc, 2.0))),
            n,
        )
        .unwrap();
        for (i, got) in v2.iter().enumerate() {
            let x = 2.0 * i as f64 / (2 * n) as f64;
            assert!((got - eval(&qc, x)).abs() < 1e-12, "P2 patch off by {}", got - eval(&qc, x));
        }
    }
}

#[test]
fn prop_the_problem_is_linear_in_its_data() {
    // Superposition, over the load and the boundary values together.
    // Nothing in the assembly is allowed to be affine in the data.
    let mut rng = Rng::new(0x4d10_ee62);
    for _ in 0..30 {
        let (f1c, f2c) = (poly(&mut rng, 3), poly(&mut rng, 3));
        let pc = positive(&mut rng, 1);
        let qc = positive(&mut rng, 1);
        let (g0, g1) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let (h0, h1) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let n = 9;
        let solve = |fc: &[f64], a: f64, b: f64| {
            let fc = fc.to_vec();
            fem_1d_general(
                &|x: f64| eval(&pc, x),
                &|x: f64| eval(&qc, x),
                &|x: f64| eval(&fc, x),
                0.0,
                1.0,
                (Bc::Dirichlet(a), Bc::Neumann(b)),
                n,
            )
            .unwrap()
        };
        let a = solve(&f1c, g0, g1);
        let b = solve(&f2c, h0, h1);
        let sum: Vec<f64> = f1c.iter().zip(f2c.iter()).map(|(x, y)| x + y).collect();
        let c = solve(&sum, g0 + h0, g1 + h1);
        for i in 0..=n {
            let want = a[i] + b[i];
            assert!((c[i] - want).abs() < 1e-10 * (1.0 + want.abs()), "node {i}");
        }
    }
}

#[test]
fn prop_a_nonnegative_load_gives_a_nonnegative_solution() {
    // With no reaction term the linear-element stiffness matrix has
    // negative off-diagonals and nonnegative row sums, which makes it an
    // M-matrix: its inverse is entrywise nonnegative. That is the
    // discrete maximum principle, and unlike the continuous one it can
    // fail for a badly assembled matrix.
    let mut rng = Rng::new(0x6ac2_1f30);
    for _ in 0..40 {
        let pc = positive(&mut rng, 2);
        // A square keeps the load nonnegative without making it constant.
        let fc = poly(&mut rng, 2);
        let n = 12;
        let v = fem_1d_general(
            &|x: f64| eval(&pc, x),
            &|_| 0.0,
            &|x: f64| eval(&fc, x).powi(2),
            0.0,
            1.0,
            (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)),
            n,
        )
        .unwrap();
        assert!(v.iter().all(|&y| y >= -1e-13), "the solution dipped to {:?}", v.iter().cloned().fold(f64::INFINITY, f64::min));
    }
}

#[test]
fn prop_a_harmonic_solution_stays_between_its_boundary_values() {
    // With no load and no reaction the solution has no interior extremum
    // -- the discrete version of a harmonic function attaining its
    // extremes on the boundary.
    let mut rng = Rng::new(0x0b7d_5522);
    for _ in 0..40 {
        let pc = positive(&mut rng, 3);
        let (g0, g1) = (4.0 * rng.next_f64() - 2.0, 4.0 * rng.next_f64() - 2.0);
        let v = fem_1d_general(
            &|x: f64| eval(&pc, x),
            &|_| 0.0,
            &|_| 0.0,
            0.0,
            1.0,
            (Bc::Dirichlet(g0), Bc::Dirichlet(g1)),
            10,
        )
        .unwrap();
        let (lo, hi) = (g0.min(g1), g0.max(g1));
        for (i, &y) in v.iter().enumerate() {
            assert!(y >= lo - 1e-12 && y <= hi + 1e-12, "node {i} left the range at {y}");
        }
        // And it is monotone, since a variable p only rescales the flux.
        let rising = g1 > g0;
        for w in v.windows(2) {
            assert_eq!(w[1] >= w[0] - 1e-12, rising || (g1 - g0).abs() < 1e-12);
        }
    }
}

#[test]
fn prop_refining_or_enriching_the_space_lowers_the_energy() {
    // V_n sits inside V_2n, and the linear space on a mesh sits inside
    // the quadratic space on the same mesh. A minimiser over a larger set
    // cannot do worse, so both refinements lower the computed energy.
    let mut rng = Rng::new(0x2f88_ac41);
    for _ in 0..25 {
        let fc = poly(&mut rng, 4);
        let f = |x: f64| eval(&fc, x);
        let bc = (Bc::Dirichlet(0.2), Bc::Dirichlet(-0.5));
        let n = 5;
        let coarse =
            Fem1dSolution::new(0.0, 1.0, 1, fem_1d_poisson(&f, 0.0, 1.0, bc, n).unwrap()).unwrap();
        let fine =
            Fem1dSolution::new(0.0, 1.0, 1, fem_1d_poisson(&f, 0.0, 1.0, bc, 2 * n).unwrap())
                .unwrap();
        let rich = Fem1dSolution::new(
            0.0,
            1.0,
            2,
            fem_1d_quadratic(&|_| 1.0, &|_| 0.0, &f, 0.0, 1.0, bc, n).unwrap(),
        )
        .unwrap();
        let j = |s: &Fem1dSolution| energy(s, &|_| 1.0, &f, 8 * n);
        assert!(j(&fine) <= j(&coarse) + 1e-12, "refining raised the energy");
        assert!(j(&rich) <= j(&coarse) + 1e-12, "enriching raised the energy");
    }
}

#[test]
fn prop_testing_against_the_constant_gives_an_exact_conservation_law() {
    // With no Dirichlet end the constant function is admissible, and the
    // discrete equation it produces is the sum of all the others. What it
    // says is a balance: the reaction consumes exactly what the source
    // supplies plus what crosses the two ends. It holds on any mesh, at
    // machine precision, because it is one of the equations solved.
    let mut rng = Rng::new(0x18e4_63b9);
    for _ in 0..30 {
        let pc = positive(&mut rng, 2);
        let fc = poly(&mut rng, 3);
        let c = 0.5 + rng.next_f64();
        let (g0, g1) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let n = 11;
        let v = fem_1d_general(
            &|x: f64| eval(&pc, x),
            &|_| c,
            &|x: f64| eval(&fc, x),
            0.0,
            1.0,
            (Bc::Neumann(g0), Bc::Neumann(g1)),
            n,
        )
        .unwrap();
        let sol = Fem1dSolution::new(0.0, 1.0, 1, v).unwrap();
        let reaction = c * integrate(0.0, 1.0, n, &|x| sol.eval(x));
        let source = integrate(0.0, 1.0, n, &|x| eval(&fc, x));
        let balance = reaction - source - g0 - g1;
        assert!(balance.abs() < 1e-11 * (1.0 + source.abs()), "balance was off by {balance}");
    }
}

#[test]
fn prop_a_robin_end_becomes_a_dirichlet_end_as_its_coefficient_grows() {
    // p du/dn + alpha u = alpha * U forces u towards U at the rate 1/alpha:
    // the flux term is bounded, so the residual is the flux over alpha.
    let mut rng = Rng::new(0x7d31_0c04);
    for _ in 0..20 {
        let fc = poly(&mut rng, 2);
        let target = 2.0 * rng.next_f64() - 1.0;
        let n = 10;
        let mut previous = f64::INFINITY;
        for alpha in [1e2, 1e4, 1e6] {
            let v = fem_1d_poisson(
                &|x: f64| eval(&fc, x),
                0.0,
                1.0,
                (Bc::Dirichlet(0.0), Bc::Robin { alpha, g: alpha * target }),
                n,
            )
            .unwrap();
            let gap = (v[n] - target).abs();
            assert!(gap < previous, "raising alpha did not tighten the end value");
            previous = gap;
        }
        assert!(previous < 1e-4, "the Robin end never reached its target: {previous}");
    }
}

#[test]
fn prop_reflecting_the_interval_reflects_the_solution() {
    // The operator is unchanged by x -> a + b - x provided the
    // coefficients and the boundary conditions are carried along, and the
    // outward-normal convention is exactly what makes the flux values
    // transfer unchanged rather than with a sign flip.
    let mut rng = Rng::new(0x55aa_3391);
    for _ in 0..30 {
        let pc = positive(&mut rng, 3);
        let fc = poly(&mut rng, 3);
        let qc = positive(&mut rng, 1);
        let (g0, g1) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let n = 9;
        let forward = fem_1d_general(
            &|x: f64| eval(&pc, x),
            &|x: f64| eval(&qc, x),
            &|x: f64| eval(&fc, x),
            0.0,
            1.0,
            (Bc::Dirichlet(g0), Bc::Neumann(g1)),
            n,
        )
        .unwrap();
        let mirrored = fem_1d_general(
            &|x: f64| eval(&pc, 1.0 - x),
            &|x: f64| eval(&qc, 1.0 - x),
            &|x: f64| eval(&fc, 1.0 - x),
            0.0,
            1.0,
            (Bc::Neumann(g1), Bc::Dirichlet(g0)),
            n,
        )
        .unwrap();
        for i in 0..=n {
            let want = forward[n - i];
            assert!((mirrored[i] - want).abs() < 1e-11 * (1.0 + want.abs()), "node {i}");
        }
    }
}

#[test]
fn prop_the_error_obeys_the_poincare_inequality() {
    // The error of a two-ended Dirichlet problem vanishes at both ends,
    // so Poincare-Friedrichs applies with its sharp constant
    // (b - a)/pi -- the reciprocal of the square root of the first
    // eigenvalue of the Laplacian on the interval. An L2 error larger
    // than that would mean the error function is not what it claims.
    let mut rng = Rng::new(0x6612_bb28);
    let pi = std::f64::consts::PI;
    for _ in 0..30 {
        let a = rng.next_f64();
        let b = a + 0.5 + 2.0 * rng.next_f64();
        let k = 1.0 + 3.0 * rng.next_f64();
        // Transcendental data, so the error is genuinely nonzero.
        let u = |x: f64| (k * (x - a)).sin();
        let du = |x: f64| k * (k * (x - a)).cos();
        let f = |x: f64| k * k * (k * (x - a)).sin();
        let n = 7;
        let sol = Fem1dSolution::new(
            a,
            b,
            1,
            fem_1d_poisson(&f, a, b, (Bc::Dirichlet(u(a)), Bc::Dirichlet(u(b))), n).unwrap(),
        )
        .unwrap();
        let l2 = fem_1d_error_l2(&sol, &u);
        let semi = fem_1d_error_h1_seminorm(&sol, &du);
        assert!(l2 > 0.0 && semi > 0.0, "the error vanished, so the test proves nothing");
        assert!(l2 <= (b - a) / pi * semi * (1.0 + 1e-9), "L2 {l2} beat Poincare on {semi}");
        // And the full norm is the hypotenuse of the two.
        let full = fem_1d_error_h1(&sol, &u, &du);
        assert!((full - l2.hypot(semi)).abs() < 1e-14 * full);
    }
}

#[test]
fn prop_the_convergence_orders_are_the_ones_the_spaces_promise() {
    // Second order in L2 and first in H1 for linear elements, one better
    // for quadratic. These are the statements that identify the space:
    // a quadratic element with a mistyped shape function still converges,
    // just at the linear rate.
    let mut rng = Rng::new(0x4e0a_9c17);
    for _ in 0..8 {
        let k = 2.0 + 3.0 * rng.next_f64();
        let phase = rng.next_f64();
        let u = |x: f64| (k * x + phase).sin();
        let du = |x: f64| k * (k * x + phase).cos();
        let f = |x: f64| k * k * (k * x + phase).sin();
        let bc = (Bc::Dirichlet(u(0.0)), Bc::Dirichlet(u(1.0)));
        let mut hs = Vec::new();
        let (mut l1, mut h1, mut l2n, mut h2n) = (vec![], vec![], vec![], vec![]);
        for n in [8usize, 16, 32, 64] {
            let p1 = Fem1dSolution::new(
                0.0,
                1.0,
                1,
                fem_1d_poisson(&f, 0.0, 1.0, bc, n).unwrap(),
            )
            .unwrap();
            let p2 = Fem1dSolution::new(
                0.0,
                1.0,
                2,
                fem_1d_quadratic(&|_| 1.0, &|_| 0.0, &f, 0.0, 1.0, bc, n).unwrap(),
            )
            .unwrap();
            l1.push(fem_1d_error_l2(&p1, &u));
            h1.push(fem_1d_error_h1_seminorm(&p1, &du));
            l2n.push(fem_1d_error_l2(&p2, &u));
            h2n.push(fem_1d_error_h1_seminorm(&p2, &du));
            hs.push(1.0 / n as f64);
        }
        for (errors, want) in [(&l1, 2.0), (&h1, 1.0), (&l2n, 3.0), (&h2n, 2.0)] {
            let rate = convergence_rate(errors, &hs).unwrap();
            assert!((rate - want).abs() < 0.06, "wanted order {want}, measured {rate}");
        }
    }
}

#[test]
fn prop_the_convergence_rate_is_a_slope_and_ignores_the_scales() {
    // A log-log slope does not see a constant factor on either axis, and
    // does not see the order the refinements were listed in. Anything
    // that did would be fitting something other than the exponent.
    let mut rng = Rng::new(0x39fd_71aa);
    for _ in 0..40 {
        let k = 4.0 * rng.next_f64() - 1.0;
        let hs: Vec<f64> = (0..5).map(|i| 0.7f64.powi(i) * (0.5 + rng.next_f64())).collect();
        let c = 0.1 + 4.0 * rng.next_f64();
        let errors: Vec<f64> = hs.iter().map(|h| c * h.powf(k)).collect();
        let base = convergence_rate(&errors, &hs).unwrap();
        assert!((base - k).abs() < 1e-9, "wanted {k}, got {base}");
        let scaled: Vec<f64> = errors.iter().map(|e| 37.0 * e).collect();
        assert!((convergence_rate(&scaled, &hs).unwrap() - k).abs() < 1e-9);
        let stretched: Vec<f64> = hs.iter().map(|h| 0.13 * h).collect();
        let restretched: Vec<f64> = stretched.iter().map(|h| c * h.powf(k)).collect();
        assert!((convergence_rate(&restretched, &stretched).unwrap() - k).abs() < 1e-9);
        let (mut re, mut rh) = (errors.clone(), hs.clone());
        re.reverse();
        rh.reverse();
        assert!((convergence_rate(&re, &rh).unwrap() - base).abs() < 1e-12);
    }
}

#[test]
fn prop_the_evaluator_agrees_with_its_own_derivative() {
    // Inside an element the piecewise polynomial is smooth, so a centred
    // difference of eval must match eval_derivative. Straddling an
    // element boundary it need not, and that discontinuity is the point:
    // a finite element solution is continuous but its derivative is not.
    let mut rng = Rng::new(0x0f2a_4d6e);
    for _ in 0..40 {
        let n = 4 + (rng.next_u64() % 5) as usize;
        for degree in [1usize, 2] {
            let values: Vec<f64> =
                (0..=degree * n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
            let s = Fem1dSolution::new(0.0, 1.0, degree, values).unwrap();
            let h = s.h();
            for e in 0..n {
                let x = 0.0 + (e as f64 + 0.5) * h;
                let d = 1e-5 * h;
                let fd = (s.eval(x + d) - s.eval(x - d)) / (2.0 * d);
                let exact = s.eval_derivative(x);
                assert!((fd - exact).abs() < 1e-6 * (1.0 + exact.abs()), "degree {degree}");
            }
        }
    }
}

#[test]
fn prop_a_pure_flux_problem_is_singular_exactly_when_nothing_pins_it() {
    // The constant is in the kernel unless a Dirichlet end, a reaction
    // term or a Robin coefficient removes it. Each of those three
    // independently makes the same data solvable, and none of them is
    // needed twice.
    let mut rng = Rng::new(0x60b1_2fe7);
    for _ in 0..30 {
        let pc = positive(&mut rng, 2);
        let fc = poly(&mut rng, 2);
        let (g0, g1) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let n = 8;
        let run = |q: &dyn Fn(f64) -> f64, bc: (Bc, Bc)| {
            fem_1d_general(
                &|x: f64| eval(&pc, x),
                q,
                &|x: f64| eval(&fc, x),
                0.0,
                1.0,
                bc,
                n,
            )
        };
        assert_eq!(
            run(&|_| 0.0, (Bc::Neumann(g0), Bc::Neumann(g1))),
            Err(SolveError::Singular)
        );
        assert!(run(&|_| 0.7, (Bc::Neumann(g0), Bc::Neumann(g1))).is_ok());
        assert!(run(&|_| 0.0, (Bc::Robin { alpha: 0.9, g: g0 }, Bc::Neumann(g1))).is_ok());
        assert!(run(&|_| 0.0, (Bc::Dirichlet(g0), Bc::Neumann(g1))).is_ok());
        // A Robin end with a zero coefficient is a flux condition and
        // pins nothing, which is the boundary case the check has to get
        // right rather than treating "Robin" as a keyword.
        assert_eq!(
            run(&|_| 0.0, (Bc::Robin { alpha: 0.0, g: g0 }, Bc::Neumann(g1))),
            Err(SolveError::Singular)
        );
    }
}
