//! Properties of the continuous optimisation modules.
//!
//! Convex optimisation is the part of the subject where randomised checking
//! bites hardest, because optimality has an exact certificate. A proximal
//! operator is not "approximately" the right point: it is the unique
//! minimiser of a strongly convex function, and the variational inequality
//! that characterises it can be tested against arbitrary competitors. The
//! lasso's subgradient conditions are equalities on the support and
//! inequalities off it, with no tolerance in the mathematics. So the tests
//! below check certificates rather than convergence wherever a certificate
//! exists.
//!
//! The stochastic searches admit less. What can still be demanded exactly of
//! them is internal consistency -- the reported value is the objective at the
//! reported point, the reported permutation is a permutation, the reported
//! front is exactly the non-dominated set -- and those are the bugs that
//! actually occur.

use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::optimization::convex::{
    admm_lasso, bfgs, conjugate_gradient_nonlinear, exact_line_search, frank_wolfe,
    lasso_coordinate_descent, lbfgs, newton_method_nd, projected_gradient, prox_box, prox_l1,
    prox_l2, prox_simplex, ridge_closed_form,
};
use rust_physics_engine::optimization::metaheuristics::{
    benchmark_functions, cma_es, convergence_curve, differential_evolution,
    genetic_algorithm_permutation, hypervolume_2d, multistart_local, pareto_front,
    particle_swarm, pattern_search, GaConfig,
};

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

fn spread(rng: &mut Rng, half_width: f64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * half_width
}

/// A random positive definite matrix, built as `L L' + I` so that it is
/// definite by construction rather than by luck.
fn random_spd(rng: &mut Rng, n: usize) -> Matrix {
    let mut l = Matrix::zeros(n, n);
    for r in 0..n {
        for c in 0..=r {
            l.set(r, c, spread(rng, 1.5));
        }
    }
    Matrix::from_fn(n, n, |r, c| {
        let dot: f64 = (0..n).map(|k| l.get(r, k) * l.get(c, k)).sum();
        dot + if r == c { 1.0 } else { 0.0 }
    })
}

/// A dense matrix of independent draws.
fn random_matrix(rng: &mut Rng, rows: usize, cols: usize, half_width: f64) -> Matrix {
    let mut a = Matrix::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            a.set(r, c, spread(rng, half_width));
        }
    }
    a
}

fn quadratic_value(q: &Matrix, c: &[f64], x: &[f64]) -> f64 {
    let n = x.len();
    let mut acc = 0.0;
    for i in 0..n {
        acc += c[i] * x[i];
        for j in 0..n {
            acc += 0.5 * q.get(i, j) * x[i] * x[j];
        }
    }
    acc
}

fn quadratic_gradient(q: &Matrix, c: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|i| c[i] + (0..n).map(|j| q.get(i, j) * x[j]).sum::<f64>()).collect()
}

// ---------------------------------------------------------------------------
// Proximal operators: certificates, not convergence
// ---------------------------------------------------------------------------

#[test]
fn prop_soft_thresholding_satisfies_the_subgradient_condition_of_its_own_problem() {
    // prox_l1 minimises ||x - v||^2 / 2 + t ||x||_1. Optimality is
    // v - x in t * d||x||_1, which is an equality where x is non-zero and a
    // bound where it is not. Both are exact statements about the returned
    // vector, so no tolerance beyond rounding is allowed.
    let mut rng = Rng::new(0x_C0FE_0001);
    for _ in 0..500 {
        let n = 1 + pick(&mut rng, 6);
        let v: Vec<f64> = (0..n).map(|_| spread(&mut rng, 4.0)).collect();
        let t = rng.next_f64() * 3.0;
        let p = prox_l1(&v, t);

        for i in 0..n {
            let residual = v[i] - p[i];
            if p[i] != 0.0 {
                assert!(
                    (residual - t * p[i].signum()).abs() < 1e-12,
                    "coordinate {i} is non-zero at {} but v - p is {residual}, not {}",
                    p[i],
                    t * p[i].signum()
                );
            } else {
                assert!(
                    residual.abs() <= t + 1e-12,
                    "coordinate {i} was zeroed although |v| = {} exceeds t = {t}",
                    v[i].abs()
                );
            }
            // Shrinkage never overshoots: the result cannot cross zero.
            assert!(p[i] * v[i] >= 0.0, "coordinate {i} changed sign");
            assert!(p[i].abs() <= v[i].abs() + 1e-12, "coordinate {i} grew");
        }
    }
}

#[test]
fn prop_the_group_threshold_shrinks_the_norm_by_exactly_the_threshold() {
    // prox_l2 is the vector analogue: the direction is untouched and the
    // length becomes max(0, ||v|| - t). Both halves are checkable exactly.
    let mut rng = Rng::new(0x_C0FE_0002);
    for _ in 0..400 {
        let n = 1 + pick(&mut rng, 5);
        let v: Vec<f64> = (0..n).map(|_| spread(&mut rng, 2.0)).collect();
        let t = rng.next_f64() * 3.0;
        let p = prox_l2(&v, t);

        let vn: f64 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let pn: f64 = p.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(
            (pn - (vn - t).max(0.0)).abs() < 1e-12,
            "||v|| = {vn} and t = {t} but the result has norm {pn}"
        );
        if pn > 0.0 {
            // Parallel to v: the cross terms of the normalised vectors agree.
            for i in 0..n {
                assert!(
                    (p[i] / pn - v[i] / vn).abs() < 1e-12,
                    "the direction changed at coordinate {i}"
                );
            }
        }
    }
}

#[test]
fn prop_the_projections_satisfy_the_variational_inequality_against_random_competitors() {
    // The defining property of a Euclidean projection onto a convex set C:
    // for every q in C, (v - p) . (q - p) <= 0. It is what makes p the
    // closest point, and it is exact -- so it can be thrown at a few hundred
    // random competitors per draw rather than at a grid.
    let mut rng = Rng::new(0x_C0FE_0003);
    for _ in 0..300 {
        let n = 2 + pick(&mut rng, 5);
        let v: Vec<f64> = (0..n).map(|_| spread(&mut rng, 5.0)).collect();

        // The simplex.
        let p = prox_simplex(&v);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{p:?} does not sum to one");
        assert!(p.iter().all(|x| *x >= -1e-15), "{p:?} has a negative entry");
        for _ in 0..40 {
            // A random point of the simplex, from normalised positive weights.
            let raw: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
            let total: f64 = raw.iter().sum();
            let q: Vec<f64> = raw.iter().map(|a| a / total).collect();
            let inner: f64 =
                (0..n).map(|i| (v[i] - p[i]) * (q[i] - p[i])).sum();
            assert!(inner <= 1e-9, "the simplex projection is beaten in direction {q:?}");
        }

        // A box, whose projection is the clamp.
        let lo = spread(&mut rng, 2.0);
        let hi = lo + rng.next_f64() * 4.0 + 0.1;
        let p = prox_box(&v, lo, hi);
        for _ in 0..40 {
            let q: Vec<f64> = (0..n).map(|_| lo + rng.next_f64() * (hi - lo)).collect();
            let inner: f64 = (0..n).map(|i| (v[i] - p[i]) * (q[i] - p[i])).sum();
            assert!(inner <= 1e-9, "the box projection is beaten in direction {q:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Smooth minimisation against closed forms
// ---------------------------------------------------------------------------

#[test]
fn prop_the_exact_search_zeroes_the_directional_derivative() {
    // Whatever the quadratic and whatever the descent direction, the returned
    // step is the one where the slope along the line vanishes, and it agrees
    // with -(g . d) / (d' Q d) computed directly.
    let mut rng = Rng::new(0x_C0FE_0004);
    for _ in 0..200 {
        let n = 2 + pick(&mut rng, 4);
        let q = random_spd(&mut rng, n);
        let c: Vec<f64> = (0..n).map(|_| spread(&mut rng, 2.0)).collect();
        let x: Vec<f64> = (0..n).map(|_| spread(&mut rng, 3.0)).collect();
        let grad = |z: &[f64]| quadratic_gradient(&q, &c, z);

        let g = grad(&x);
        let mut direction: Vec<f64> = (0..n).map(|_| spread(&mut rng, 1.0)).collect();
        let slope: f64 = g.iter().zip(&direction).map(|(a, b)| a * b).sum();
        if slope.abs() < 1e-8 {
            continue;
        }
        if slope > 0.0 {
            for d in &mut direction {
                *d = -*d;
            }
        }
        let slope: f64 = g.iter().zip(&direction).map(|(a, b)| a * b).sum();
        let curvature: f64 = (0..n)
            .map(|i| (0..n).map(|j| direction[i] * q.get(i, j) * direction[j]).sum::<f64>())
            .sum();
        let expected = -slope / curvature;

        let t = exact_line_search(&grad, &x, &direction).expect("a definite quadratic is bounded");
        assert!(
            (t - expected).abs() <= 1e-8 * expected.abs().max(1.0),
            "the closed-form step is {expected} but the search returned {t}"
        );
    }
}

#[test]
fn prop_every_smooth_method_lands_on_the_same_closed_form_minimiser() {
    // The minimiser of a positive definite quadratic solves Q x = -c, which
    // is available without an optimiser. Newton, BFGS, L-BFGS and conjugate
    // gradients are each measured against that, never against each other.
    let mut rng = Rng::new(0x_C0FE_0005);
    for _ in 0..120 {
        let n = 2 + pick(&mut rng, 4);
        let q = random_spd(&mut rng, n);
        let c: Vec<f64> = (0..n).map(|_| spread(&mut rng, 3.0)).collect();
        let f = |z: &[f64]| quadratic_value(&q, &c, z);
        let grad = |z: &[f64]| quadratic_gradient(&q, &c, z);
        let hess = |_: &[f64]| q.clone();

        let negated: Vec<f64> = c.iter().map(|v| -v).collect();
        let Ok(exact) = rust_physics_engine::linalg::lu::solve(&q, &negated) else {
            continue;
        };
        let start: Vec<f64> = (0..n).map(|_| spread(&mut rng, 4.0)).collect();

        // Newton on a quadratic is exact after a single step, since the
        // quadratic model it minimises *is* the function.
        let (one_step, _) = newton_method_nd(&f, &grad, &hess, &start, 0.0, 1).unwrap();
        for i in 0..n {
            assert!(
                (one_step[i] - exact[i]).abs() < 1e-6 * (1.0 + exact[i].abs()),
                "one Newton step gave {one_step:?} against {exact:?}"
            );
        }

        for (name, got) in [
            ("bfgs", bfgs(&f, &grad, &start, 1e-11, 500).unwrap().0),
            ("lbfgs", lbfgs(&f, &grad, &start, 6, 1e-11, 500).unwrap().0),
            ("cg", conjugate_gradient_nonlinear(&f, &grad, &start, 1e-11, 400).unwrap().0),
        ] {
            for i in 0..n {
                assert!(
                    (got[i] - exact[i]).abs() < 1e-5 * (1.0 + exact[i].abs()),
                    "{name} gave {got:?} against {exact:?}"
                );
            }
            // The value can only be checked downward: it is a minimum.
            assert!(
                f(&got) <= f(&exact) + 1e-9 * (1.0 + f(&exact).abs()),
                "{name} reports a value below the true minimum"
            );
        }
    }
}

#[test]
fn prop_ridge_solves_its_normal_equations_and_shrinks_with_the_penalty() {
    // The ridge estimate is defined by (A'A + lambda I) x = A'b. Substituting
    // the returned vector back is a complete check of correctness, and the
    // monotone shrinkage of ||x|| in lambda is a separate consequence.
    let mut rng = Rng::new(0x_C0FE_0006);
    for _ in 0..200 {
        let rows = 4 + pick(&mut rng, 8);
        let cols = 1 + pick(&mut rng, 4);
        let a = random_matrix(&mut rng, rows, cols, 2.0);
        let b: Vec<f64> = (0..rows).map(|_| spread(&mut rng, 3.0)).collect();

        let mut previous = f64::INFINITY;
        for lambda in [0.01f64, 0.1, 1.0, 10.0, 100.0] {
            let Ok(x) = ridge_closed_form(&a, &b, lambda) else {
                continue;
            };
            for i in 0..cols {
                let lhs: f64 = (0..cols)
                    .map(|j| {
                        let g: f64 = (0..rows).map(|r| a.get(r, i) * a.get(r, j)).sum();
                        (g + if i == j { lambda } else { 0.0 }) * x[j]
                    })
                    .sum();
                let rhs: f64 = (0..rows).map(|r| a.get(r, i) * b[r]).sum();
                assert!(
                    (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                    "row {i} of the normal equations reads {lhs} against {rhs}"
                );
            }
            let magnitude: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                magnitude <= previous + 1e-8,
                "the coefficients grew from {previous} to {magnitude} as lambda rose to {lambda}"
            );
            previous = magnitude;
        }
    }
}

#[test]
fn prop_both_lasso_solvers_meet_the_subgradient_conditions_and_agree() {
    // For the objective ||A x - b||^2 / (2n) + lambda ||x||_1, optimality is
    // a'_j (A x - b) / n = -lambda sign(x_j) on the support and
    // |a'_j (A x - b) / n| <= lambda off it. That certificate is what makes
    // the two solvers comparable: each is checked against the mathematics
    // first, and only then against the other.
    let mut rng = Rng::new(0x_C0FE_0007);
    let mut sparse_seen = 0usize;
    for _ in 0..150 {
        let rows = 12 + pick(&mut rng, 12);
        let cols = 2 + pick(&mut rng, 4);
        let a = random_matrix(&mut rng, rows, cols, 1.5);
        let b: Vec<f64> = (0..rows).map(|_| spread(&mut rng, 2.0)).collect();
        let lambda = 0.02 + rng.next_f64() * 0.4;

        let cd = lasso_coordinate_descent(&a, &b, lambda, 4000).unwrap();
        let Ok(admm) = admm_lasso(&a, &b, lambda, 1.0, 4000) else {
            continue;
        };

        let correlation = |x: &[f64], j: usize| -> f64 {
            let residual: Vec<f64> = (0..rows)
                .map(|r| (0..cols).map(|k| a.get(r, k) * x[k]).sum::<f64>() - b[r])
                .collect();
            (0..rows).map(|r| a.get(r, j) * residual[r]).sum::<f64>() / rows as f64
        };

        for j in 0..cols {
            let c = correlation(&cd, j);
            if cd[j].abs() > 1e-9 {
                assert!(
                    (c + lambda * cd[j].signum()).abs() < 1e-6,
                    "on the support, coordinate {j} has correlation {c} against lambda {lambda}"
                );
            } else {
                sparse_seen += 1;
                assert!(
                    c.abs() <= lambda + 1e-6,
                    "coordinate {j} is zero although its correlation {c} exceeds lambda {lambda}"
                );
            }
            // ADMM solves the same problem, so it must land in the same place.
            assert!(
                (cd[j] - admm[j]).abs() < 5e-3,
                "coordinate {j}: coordinate descent {} against ADMM {}",
                cd[j],
                admm[j]
            );
        }
    }
    assert!(sparse_seen > 50, "only {sparse_seen} zero coefficients arose, so sparsity is untested");
}

#[test]
fn prop_the_constrained_methods_never_leave_the_feasible_set() {
    // Projected gradient and Frank-Wolfe differ in how they stay feasible --
    // one projects, one takes convex combinations of vertices -- but the
    // guarantee is the same and it holds at the returned point whatever the
    // objective and starting point.
    let mut rng = Rng::new(0x_C0FE_0008);
    for _ in 0..200 {
        let n = 2 + pick(&mut rng, 4);
        let q = random_spd(&mut rng, n);
        let c: Vec<f64> = (0..n).map(|_| spread(&mut rng, 3.0)).collect();
        let grad = |z: &[f64]| quadratic_gradient(&q, &c, z);
        let start: Vec<f64> = (0..n).map(|_| spread(&mut rng, 4.0)).collect();

        let projected = projected_gradient(&grad, &|z| prox_simplex(z), &start, 0.02, 400);
        assert!(
            (projected.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "the projected iterate {projected:?} left the simplex"
        );
        assert!(projected.iter().all(|v| *v >= -1e-12), "{projected:?} has a negative entry");

        // The linear oracle over the simplex is the vertex of steepest
        // descent, so every Frank-Wolfe iterate is a convex combination of
        // vertices and stays inside.
        let oracle = |g: &[f64]| -> Vec<f64> {
            let best = (0..n)
                .min_by(|&i, &j| g[i].partial_cmp(&g[j]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            (0..n).map(|i| if i == best { 1.0 } else { 0.0 }).collect()
        };
        let inside = prox_simplex(&start);
        let fw = frank_wolfe(&grad, &oracle, &inside, 300);
        assert!(
            (fw.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "the Frank-Wolfe iterate {fw:?} left the simplex"
        );
        assert!(fw.iter().all(|v| *v >= -1e-12), "{fw:?} has a negative entry");
    }
}

// ---------------------------------------------------------------------------
// Stochastic search: internal consistency and the exact combinatorics
// ---------------------------------------------------------------------------

#[test]
fn prop_the_searches_report_the_value_they_actually_found() {
    // A population method that returns a point and a value has two chances to
    // disagree with itself, and bookkeeping errors here are invisible in a
    // convergence test that only looks at the value. Recomputing the
    // objective at the returned point catches them exactly.
    let mut rng = Rng::new(0x_C0FE_0009);
    for _ in 0..40 {
        let n = 2 + pick(&mut rng, 2);
        let centre: Vec<f64> = (0..n).map(|_| spread(&mut rng, 2.0)).collect();
        let weights: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64() * 3.0).collect();
        let f = |x: &[f64]| -> f64 {
            x.iter()
                .zip(&centre)
                .zip(&weights)
                .map(|((a, m), w)| w * (a - m) * (a - m))
                .sum()
        };
        let bounds: Vec<(f64, f64)> = vec![(-6.0, 6.0); n];
        let start: Vec<f64> = (0..n).map(|_| spread(&mut rng, 5.0)).collect();

        let cases: Vec<(&str, (Vec<f64>, f64))> = vec![
            ("pattern", pattern_search(&f, &start, 0.5, 1e-10, 4000)),
            ("de", differential_evolution(&f, &bounds, 20, 0.9, 0.8, 200, &mut rng)),
            ("pso", particle_swarm(&f, &bounds, 25, 0.7, 1.5, 1.5, 200, &mut rng)),
            ("cma", cma_es(&f, &start, 1.0, 300, &mut rng)),
            ("multistart", multistart_local(&f, &bounds, 8, &mut rng)),
        ];
        for (name, (x, value)) in &cases {
            assert!(
                (f(x) - value).abs() < 1e-9 * (1.0 + value.abs()),
                "{name} reported {value} at a point worth {}",
                f(x)
            );
            assert!(
                *value <= f(&start) + 1e-12,
                "{name} returned a point worse than where it started"
            );
            // A separable convex bowl has a known minimiser, so this much can
            // be demanded of every one of them.
            for i in 0..n {
                assert!(
                    (x[i] - centre[i]).abs() < 0.2,
                    "{name} stopped at {x:?}, away from {centre:?}"
                );
            }
        }
    }
}

#[test]
fn prop_the_permutation_search_returns_a_permutation() {
    // Order crossover exists precisely because the arithmetic crossovers do
    // not close on permutations. The closure property is exact and is the one
    // thing that must never fail, whatever the tour costs happen to be.
    let mut rng = Rng::new(0x_C0FE_000A);
    for _ in 0..30 {
        let n = 4 + pick(&mut rng, 6);
        let cost_matrix: Vec<Vec<f64>> =
            (0..n).map(|_| (0..n).map(|_| rng.next_f64() * 10.0).collect()).collect();
        let cost = |p: &[usize]| -> f64 {
            (0..p.len()).map(|i| cost_matrix[p[i]][p[(i + 1) % p.len()]]).sum()
        };
        let config = GaConfig { population: 24, generations: 40, elite: 2, ..GaConfig::default() };
        let (tour, value) = genetic_algorithm_permutation(&cost, n, &config, &mut rng);

        assert_eq!(tour.len(), n, "the tour has the wrong length");
        let mut seen = vec![false; n];
        for &city in &tour {
            assert!(city < n, "the tour visits {city}, which is out of range");
            assert!(!seen[city], "the tour visits {city} twice");
            seen[city] = true;
        }
        assert!(
            (cost(&tour) - value).abs() < 1e-9,
            "the reported cost {value} is not the tour's cost {}",
            cost(&tour)
        );
    }
}

#[test]
fn prop_the_pareto_front_is_exactly_the_non_dominated_set() {
    // The specification is short enough to restate independently: index i is
    // in the front when nothing dominates it. Comparing the returned indices
    // against that brute-force definition is a complete test, not a sample.
    let mut rng = Rng::new(0x_C0FE_000B);
    for _ in 0..300 {
        let count = 1 + pick(&mut rng, 20);
        let dimension = 2 + pick(&mut rng, 3);
        // Round the coordinates so ties -- the case the strict and non-strict
        // comparisons disagree on -- actually occur.
        let points: Vec<Vec<f64>> = (0..count)
            .map(|_| (0..dimension).map(|_| (rng.next_f64() * 4.0).round()).collect())
            .collect();

        let front = pareto_front(&points);
        let dominates = |a: &[f64], b: &[f64]| -> bool {
            a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
        };
        for i in 0..count {
            let dominated =
                (0..count).any(|j| j != i && dominates(&points[j], &points[i]));
            assert_eq!(
                front.contains(&i),
                !dominated,
                "point {i} = {:?} is {}dominated but the front {}contains it",
                points[i],
                if dominated { "" } else { "not " },
                if front.contains(&i) { "" } else { "does not " }
            );
        }
        assert!(!front.is_empty(), "a non-empty set always has a non-dominated member");
        assert!(front.windows(2).all(|w| w[0] < w[1]), "the indices are not in order");
    }
}

#[test]
fn prop_the_hypervolume_is_monotone_and_blind_to_dominated_additions() {
    // Two exact statements. Adding any point cannot shrink the dominated
    // region, and adding a point that is already dominated cannot change it
    // at all, since it contributes no area of its own.
    let mut rng = Rng::new(0x_C0FE_000C);
    for _ in 0..300 {
        let count = 1 + pick(&mut rng, 8);
        let reference = (6.0f64, 6.0f64);
        let mut front: Vec<Vec<f64>> = (0..count)
            .map(|_| vec![rng.next_f64() * 5.0, rng.next_f64() * 5.0])
            .collect();
        let base = hypervolume_2d(&front, reference);
        assert!(base >= 0.0 && base.is_finite(), "the hypervolume is {base}");

        let addition = vec![rng.next_f64() * 5.0, rng.next_f64() * 5.0];
        let mut grown = front.clone();
        grown.push(addition.clone());
        let after = hypervolume_2d(&grown, reference);
        assert!(after >= base - 1e-9, "adding a point shrank the hypervolume from {base} to {after}");

        // Now a point strictly dominated by an existing one: no new area.
        let victim = front[pick(&mut rng, front.len())].clone();
        front.push(vec![victim[0] + 0.5, victim[1] + 0.5]);
        let unchanged = hypervolume_2d(&front, reference);
        assert!(
            (unchanged - base).abs() < 1e-9,
            "a dominated point moved the hypervolume from {base} to {unchanged}"
        );

        // A front entirely beyond the reference dominates nothing.
        let beyond: Vec<Vec<f64>> = (0..count)
            .map(|_| vec![reference.0 + rng.next_f64(), reference.1 + rng.next_f64()])
            .collect();
        assert_eq!(hypervolume_2d(&beyond, reference), 0.0);
    }
}

#[test]
fn prop_the_convergence_curve_is_the_running_minimum() {
    // Non-increasing, and each entry equals the minimum of the prefix. The
    // second is the definition; the first follows, and both are exact.
    let mut rng = Rng::new(0x_C0FE_000D);
    for _ in 0..300 {
        let n = 1 + pick(&mut rng, 40);
        let history: Vec<f64> = (0..n).map(|_| spread(&mut rng, 50.0)).collect();
        let curve = convergence_curve(&history);
        assert_eq!(curve.len(), n);
        for i in 0..n {
            let prefix = history[..=i]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            assert_eq!(curve[i], prefix, "entry {i} is not the minimum so far");
            if i > 0 {
                assert!(curve[i] <= curve[i - 1], "the curve rose at entry {i}");
            }
        }
    }
    assert!(convergence_curve(&[]).is_empty());
}

#[test]
fn prop_no_random_point_beats_a_benchmark_s_recorded_optimum() {
    // The recorded optima are constants in a table, and a table can be wrong.
    // Sampling the box heavily cannot prove the value but can refute it, and
    // a refutation is what would matter.
    let mut rng = Rng::new(0x_C0FE_000E);
    for benchmark in benchmark_functions() {
        let n = benchmark.bounds.len();
        for _ in 0..20_000 {
            let x: Vec<f64> = benchmark
                .bounds
                .iter()
                .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
                .collect();
            let value = (benchmark.f)(&x);
            assert!(
                value >= benchmark.optimum - 1e-9,
                "{} evaluates to {value} at {x:?}, below its recorded optimum {}",
                benchmark.name,
                benchmark.optimum
            );
            assert!(value.is_finite(), "{} returned a non-finite value", benchmark.name);
        }
        assert_eq!(n, 2, "{} is documented as two-dimensional", benchmark.name);
    }
}
