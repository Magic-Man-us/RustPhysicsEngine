//! Properties of the spectral methods module.
//!
//! Spectral discretisations are unusually rich in *exact* statements,
//! because the interpolant of a function the basis can represent is that
//! function rather than an approximation of it.
//!
//! *Exact.* The Chebyshev differentiation matrix returns the exact
//! derivative of any polynomial of degree at most `N`, at machine
//! precision and for every `N`. It annihilates constants, it is
//! centro-antisymmetric because its point set is symmetric and
//! differentiation is odd, and moving to another interval multiplies it
//! by the Jacobian and does nothing else. The periodic Poisson solver is
//! exact for any trigonometric polynomial inside the grid's band, its
//! answer is mean-free because a periodic problem admits no other
//! normalisation, and differentiating that answer twice returns the data.
//!
//! *A statement about smoothness.* The convergence rate is not a
//! property of the method but of the function it is given. For analytic
//! data the log of the error is linear in `N`; for data with a few
//! continuous derivatives it is linear in `log N`. Asking which of the
//! two models the errors actually follow separates the cases without
//! having to name a rate, and it is a sharper question than "is the
//! error small".
//!
//! *Cross-checks.* Chebyshev collocation and linear finite elements are
//! entirely different discretisations of the same operator. Where they
//! agree, both are probably right; the agreement is asserted directly.

use rust_physics_engine::fem::fem1d::{fem_1d_general, Bc, Fem1dSolution};
use rust_physics_engine::fem::spectral_pde::{
    cheb_differentiate, cheb_diff_matrix, chebyshev_collocation_bvp, chebyshev_points,
    spectral_convergence_demo, spectral_poisson_periodic, spectral_second_derivative,
};
use rust_physics_engine::monte_carlo::Rng;

const TAU: f64 = std::f64::consts::TAU;

fn poly(rng: &mut Rng, degree: usize) -> Vec<f64> {
    (0..=degree).map(|_| 2.0 * rng.next_f64() - 1.0).collect()
}

fn eval(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &a| acc * x + a)
}

fn deriv(c: &[f64]) -> Vec<f64> {
    c.iter().enumerate().skip(1).map(|(k, &a)| k as f64 * a).collect()
}

/// The correlation of `y` against `x`.
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let syy: f64 = y.iter().map(|b| (b - my) * (b - my)).sum();
    sxy / (sxx * syy).sqrt()
}

#[test]
fn prop_differentiation_is_exact_on_every_polynomial_it_can_hold() {
    // The interpolant of a polynomial of degree at most N *is* that
    // polynomial, so this is exact rather than accurate -- and it stays
    // exact as N grows, which no fixed-order difference formula does.
    let mut rng = Rng::new(0x51c0_3ea7);
    for _ in 0..30 {
        let n = 3 + (rng.next_u64() % 18) as usize;
        let a = -2.0 + 2.0 * rng.next_f64();
        let b = a + 0.5 + 3.0 * rng.next_f64();
        let d = cheb_diff_matrix(n, a, b).unwrap();
        let x = chebyshev_points(n, a, b).unwrap();
        let degree = (rng.next_u64() as usize) % (n + 1);
        let c = poly(&mut rng, degree);
        let dc = deriv(&c);
        let values: Vec<f64> = x.iter().map(|&t| eval(&c, t)).collect();
        let got = cheb_differentiate(&d, &values).unwrap();
        let scale = values.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (k, &t) in x.iter().enumerate() {
            let want = eval(&dc, t);
            assert!(
                (got[k] - want).abs() < 1e-9 * scale,
                "n={n} degree={degree} point {k}: {} vs {want}",
                got[k]
            );
        }
        // Twice differentiating gives the second derivative, which is
        // the statement that D squared is the second-derivative matrix
        // on this space.
        let twice = cheb_differentiate(&d, &got).unwrap();
        let ddc = deriv(&dc);
        for (k, &t) in x.iter().enumerate() {
            assert!((twice[k] - eval(&ddc, t)).abs() < 1e-6 * scale, "second at {k}");
        }
    }
}

#[test]
fn prop_the_matrix_carries_the_symmetries_of_its_point_set() {
    let mut rng = Rng::new(0x2ff4_8b13);
    for _ in 0..30 {
        let n = 2 + (rng.next_u64() % 24) as usize;
        let a = -3.0 + 4.0 * rng.next_f64();
        let b = a + 0.3 + 4.0 * rng.next_f64();
        let d = cheb_diff_matrix(n, a, b).unwrap();
        let scale = (0..=n)
            .flat_map(|i| (0..=n).map(move |j| (i, j)))
            .map(|(i, j)| d.get(i, j).abs())
            .fold(1.0f64, f64::max);
        for i in 0..=n {
            let row: f64 = (0..=n).map(|j| d.get(i, j)).sum();
            assert!(row.abs() < 1e-10 * scale, "row {i} summed to {row}");
            for j in 0..=n {
                assert!(
                    (d.get(i, j) + d.get(n - i, n - j)).abs() < 1e-9 * scale,
                    "entry ({i},{j}) is not centro-antisymmetric"
                );
            }
        }
        // The Jacobian is the only thing an interval change introduces.
        let unit = cheb_diff_matrix(n, -1.0, 1.0).unwrap();
        let factor = 2.0 / (b - a);
        for i in 0..=n {
            for j in 0..=n {
                let want = factor * unit.get(i, j);
                assert!((d.get(i, j) - want).abs() < 1e-9 * (1.0 + want.abs()));
            }
        }
        // The points are symmetric about the midpoint and cluster.
        let x = chebyshev_points(n, a, b).unwrap();
        let mid = 0.5 * (a + b);
        for j in 0..=n {
            assert!(((x[j] - mid) + (x[n - j] - mid)).abs() < 1e-12 * (b - a));
        }
    }
}

#[test]
fn prop_the_periodic_solver_is_exact_within_the_band_and_mean_free() {
    // Build the source from the exact solution rather than the other way
    // round, so that what is being tested is a solve and not an
    // identity: u is a random trigonometric polynomial, f is its second
    // derivative in closed form, and the solver has to recover u.
    let mut rng = Rng::new(0x7b1e_44c9);
    for _ in 0..30 {
        let n = 16 + 8 * (rng.next_u64() % 4) as usize;
        let length = 0.5 + 4.0 * rng.next_f64();
        let modes = 1 + (rng.next_u64() % 4) as usize;
        let coeffs: Vec<(f64, f64)> =
            (0..modes).map(|_| (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0)).collect();
        // Keep every mode strictly inside the band, so nothing is
        // truncated and the answer is exact rather than approximate.
        let u_at = |x: f64| -> f64 {
            coeffs
                .iter()
                .enumerate()
                .map(|(m, &(c, s))| {
                    let k = TAU * (m + 1) as f64 / length;
                    c * (k * x).cos() + s * (k * x).sin()
                })
                .sum()
        };
        let f_at = |x: f64| -> f64 {
            coeffs
                .iter()
                .enumerate()
                .map(|(m, &(c, s))| {
                    let k = TAU * (m + 1) as f64 / length;
                    -k * k * (c * (k * x).cos() + s * (k * x).sin())
                })
                .sum()
        };
        let at = |i: usize| length * i as f64 / n as f64;
        let f: Vec<f64> = (0..n).map(|i| f_at(at(i))).collect();
        let u = spectral_poisson_periodic(&f, length).unwrap();
        let scale = (0..n).map(|i| u_at(at(i)).abs()).fold(1.0f64, f64::max);
        for i in 0..n {
            assert!((u[i] - u_at(at(i))).abs() < 1e-10 * scale, "sample {i}");
        }
        let mean = u.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-11 * scale, "the mean was {mean}");
        // Differentiating twice with the same symbol undoes the solve.
        let back = spectral_second_derivative(&u, length).unwrap();
        let fscale = f.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for i in 0..n {
            assert!((back[i] - f[i]).abs() < 1e-10 * fscale, "round trip at {i}");
        }
        // A constant added to the source is dropped, because a periodic
        // problem with a nonzero mean has no solution at all and the
        // mean-free part is the most that can be answered.
        let shifted: Vec<f64> = f.iter().map(|v| v + 3.7).collect();
        let shifted_solution = spectral_poisson_periodic(&shifted, length).unwrap();
        for i in 0..n {
            assert!((shifted_solution[i] - u[i]).abs() < 1e-10 * scale, "shifted at {i}");
        }
    }
}

#[test]
fn prop_the_periodic_solver_is_linear() {
    let mut rng = Rng::new(0x0a6d_92f5);
    for _ in 0..30 {
        let n = 12 + (rng.next_u64() % 20) as usize;
        let length = 0.5 + 3.0 * rng.next_f64();
        let f1: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let f2: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let a = spectral_poisson_periodic(&f1, length).unwrap();
        let b = spectral_poisson_periodic(&f2, length).unwrap();
        let sum: Vec<f64> = f1.iter().zip(&f2).map(|(x, y)| x + y).collect();
        let c = spectral_poisson_periodic(&sum, length).unwrap();
        let scale = a.iter().chain(b.iter()).fold(1.0f64, |m, v| m.max(v.abs()));
        for i in 0..n {
            assert!((c[i] - a[i] - b[i]).abs() < 1e-10 * scale, "sample {i}");
        }
    }
}

#[test]
fn prop_collocation_reproduces_what_its_space_contains() {
    // The patch test for a spectral method: a polynomial of degree at
    // most n solves the discrete equations exactly, whatever the
    // coefficient functions, because the collocation derivative of a
    // polynomial is exact.
    let mut rng = Rng::new(0x63a0_c7e2);
    for _ in 0..25 {
        let n = 6 + (rng.next_u64() % 8) as usize;
        // Degrees well inside the space; p is a polynomial too, so that
        // -(p u')' stays one and every evaluation is exact.
        let mut pc = poly(&mut rng, 2);
        pc[0] += 3.0;
        let uc = poly(&mut rng, 4.min(n));
        let duc = deriv(&uc);
        let dduc = deriv(&duc);
        let dpc = deriv(&pc);
        let qc = poly(&mut rng, 1);
        let (a, b) = (0.0, 1.0 + rng.next_f64());
        let p = |x: f64| eval(&pc, x);
        let q = |x: f64| eval(&qc, x);
        let u = |x: f64| eval(&uc, x);
        let f = |x: f64| {
            -(eval(&dpc, x) * eval(&duc, x) + eval(&pc, x) * eval(&dduc, x)) + eval(&qc, x) * u(x)
        };
        let got = chebyshev_collocation_bvp(&p, &q, &f, a, b, (u(a), u(b)), n).unwrap();
        let x = chebyshev_points(n, a, b).unwrap();
        let scale = x.iter().map(|&t| u(t).abs()).fold(1.0f64, f64::max);
        for (k, &t) in x.iter().enumerate() {
            assert!((got[k] - u(t)).abs() < 1e-8 * scale, "point {k} at {t}: {}", got[k]);
        }
    }
}

#[test]
fn prop_collocation_and_finite_elements_meet_in_the_middle() {
    // Two discretisations with nothing in common but the operator. If
    // both are right they agree; if either has a sign error they do not,
    // and no self-consistency check on one of them would notice.
    let mut rng = Rng::new(0x18b7_5d40);
    for _ in 0..12 {
        let k = 1.0 + 2.0 * rng.next_f64();
        let c = 0.5 + rng.next_f64();
        let p = move |x: f64| 1.0 + c * x * x;
        let q = move |x: f64| 0.3 + x;
        let f = move |x: f64| (k * x).sin() + 1.0;
        let (ga, gb) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let n = 24;
        let spectral = chebyshev_collocation_bvp(&p, &q, &f, 0.0, 1.0, (ga, gb), n).unwrap();
        let elements = Fem1dSolution::new(
            0.0,
            1.0,
            1,
            fem_1d_general(
                &p,
                &q,
                &f,
                0.0,
                1.0,
                (Bc::Dirichlet(ga), Bc::Dirichlet(gb)),
                400,
            )
            .unwrap(),
        )
        .unwrap();
        let x = chebyshev_points(n, 0.0, 1.0).unwrap();
        for (j, &xi) in x.iter().enumerate() {
            let want = elements.eval(xi);
            assert!(
                (spectral[j] - want).abs() < 1e-4 * (1.0 + want.abs()),
                "point {j} at {xi}: {} against {want}",
                spectral[j]
            );
        }
    }
}

#[test]
fn prop_the_convergence_model_follows_the_smoothness_of_the_data() {
    // The question is not how small the error is but which curve it
    // lies on. Analytic data puts log(error) on a line against n;
    // data with a few derivatives puts it on a line against log(n).
    // Comparing the two fits separates them without naming a rate, and
    // it is the honest form of the claim that spectral methods converge
    // "exponentially" -- they do so exactly when the function lets them.
    let rough: Vec<usize> = vec![8, 12, 16, 20, 24, 28, 32, 36];
    let fits = |sizes: &[usize], e: &[f64]| {
        let ln_e: Vec<f64> = e.iter().map(|v| v.ln()).collect();
        let n: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
        let ln_n: Vec<f64> = n.iter().map(|v| v.ln()).collect();
        (correlation(&ln_n, &ln_e), correlation(&n, &ln_e))
    };

    type Pair<'a> = (&'a dyn Fn(f64) -> f64, &'a dyn Fn(f64) -> f64);
    let smooth: Vec<Pair> = vec![
        (&|x: f64| x.sin().exp(), &|x: f64| x.cos() * x.sin().exp()),
        (&|x: f64| (2.0 * x).cos(), &|x: f64| -2.0 * (2.0 * x).sin()),
        (&|x: f64| 1.0 / (2.0 + x), &|x: f64| -1.0 / (2.0 + x).powi(2)),
    ];
    // Geometric convergence runs into the rounding floor, and where it
    // does the recorded "errors" are cancellation noise rather than
    // truncation. How soon depends on the function -- cos(2x) is entire
    // and is there by n = 16, while 1/(2+x) has a pole a unit away from
    // the interval and is still converging at n = 24 -- so the window is
    // chosen per function rather than fixed. Fitting a model to the
    // floor would be fitting nothing.
    let ladder: Vec<usize> = vec![4, 6, 8, 10, 12, 14, 16, 18, 20, 24];
    for (f, df) in smooth {
        let all = spectral_convergence_demo(f, df, -1.0, 1.0, &ladder).unwrap();
        let keep = all.iter().position(|&v| v < 1e-12).unwrap_or(all.len());
        assert!(keep >= 5, "only {keep} sizes stayed above the rounding floor");
        let sizes = &ladder[..keep];
        let e = &all[..keep];
        let (power, exponential) = fits(sizes, e);
        assert!(
            exponential < power,
            "analytic data fitted a power law better: {exponential} against {power}"
        );
        assert!(e[0] / e[keep - 1] > 1e5, "the error only fell by {}", e[0] / e[keep - 1]);
    }

    let kinked: Vec<(Pair, f64)> = vec![
        ((&|x: f64| x.abs().powi(3), &|x: f64| 3.0 * x * x * x.signum()), 2.0),
        ((&|x: f64| x.abs().powi(5), &|x: f64| 5.0 * x.powi(4) * x.signum()), 4.0),
    ];
    let mut previous: Option<Vec<f64>> = None;
    for ((f, df), order) in kinked {
        let e = spectral_convergence_demo(f, df, -1.0, 1.0, &rough).unwrap();
        let (power, exponential) = fits(&rough, &e);
        assert!(
            power < exponential,
            "kinked data fitted an exponential better: {power} against {exponential}"
        );
        assert!(power < -0.99, "the power law fit was poor: {power}");
        let ln_e: Vec<f64> = e.iter().map(|v| v.ln()).collect();
        let ln_n: Vec<f64> = rough.iter().map(|&s| (s as f64).ln()).collect();
        let m = ln_n.len() as f64;
        let mx = ln_n.iter().sum::<f64>() / m;
        let my = ln_e.iter().sum::<f64>() / m;
        let slope: f64 = ln_n.iter().zip(&ln_e).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>()
            / ln_n.iter().map(|a| (a - mx) * (a - mx)).sum::<f64>();
        assert!(
            (slope + order).abs() < 0.8,
            "|x|^k with {order} derivatives converged at {slope}"
        );
        // A smoother kink converges faster at every size.
        if let Some(coarser) = &previous {
            for j in 0..rough.len() {
                assert!(e[j] < coarser[j], "the smoother function was not more accurate");
            }
        }
        previous = Some(e);
    }
}
