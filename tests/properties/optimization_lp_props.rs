//! Properties of the linear programming module.
//!
//! The theorems of linear programming are unusually well suited to randomised
//! checking, because they are exact rather than asymptotic. Strong duality is
//! an equation, not a bound; complementary slackness holds at every optimal
//! basis, not on average; and the two solvers must agree to solver tolerance
//! on every instance rather than typically. So these run over hundreds of
//! random programs and demand equality.

use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::optimization::lp::{
    chebyshev_center, interior_point, l1_regression_lp, linf_regression_lp, lp_dual,
    sensitivity_ranges, simplex, two_player_zero_sum_lp, Cmp, LpProblem, LpResult,
};

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random program whose feasible region always contains the origin, so only
/// unboundedness can prevent an optimum.
fn random_bounded_lp(rng: &mut Rng) -> LpProblem {
    let m = 2 + pick(rng, 4);
    let n = 2 + pick(rng, 4);
    let mut a = Matrix::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            // Non-negative coefficients with a non-negative right-hand side
            // make the region a bounded simplex-like body, so every random
            // draw has an optimum rather than running off to infinity.
            a.set(i, j, (rng.next_f64() * 4.0).round() + 1.0);
        }
    }
    let b: Vec<f64> = (0..m).map(|_| (rng.next_f64() * 30.0).round() + 5.0).collect();
    let c: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 10.0).round() - 3.0).collect();
    LpProblem::new(c, a, b, true).unwrap()
}

#[test]
fn prop_strong_duality_holds_exactly_at_every_optimum() {
    // The objective equals b . y, where y is the vector of shadow prices.
    // Nothing in the solver imposes this: the duals are read off the optimal
    // basis and the objective from the primal solution.
    let mut rng = Rng::new(0x_0D0A_1001);
    let mut checked = 0usize;
    for _ in 0..400 {
        let p = random_bounded_lp(&mut rng);
        let LpResult::Optimal { x, objective, duals, reduced_costs } = simplex(&p).unwrap() else {
            continue;
        };
        checked += 1;
        assert!(p.is_feasible(&x, 1e-7), "the reported point is infeasible: {x:?}");

        let by: f64 = p.b.iter().zip(&duals).map(|(a, b)| a * b).sum();
        assert!(
            (by - objective).abs() < 1e-6 * (1.0 + objective.abs()),
            "b . y = {by} against objective {objective}"
        );

        // Complementary slackness, both halves.
        for (j, &xj) in x.iter().enumerate() {
            if xj > 1e-7 {
                assert!(
                    reduced_costs[j].abs() < 1e-6,
                    "variable {j} is in use with reduced cost {}",
                    reduced_costs[j]
                );
            }
        }
        for i in 0..p.m() {
            let row: f64 = (0..p.n()).map(|j| p.a.get(i, j) * x[j]).sum();
            if row < p.b[i] - 1e-7 {
                assert!(duals[i].abs() < 1e-6, "slack row {i} priced at {}", duals[i]);
            }
        }
        // For a maximisation over `<=` rows, no resource has a negative price.
        assert!(duals.iter().all(|&v| v > -1e-7), "a shadow price went negative: {duals:?}");
    }
    assert!(checked > 300, "only {checked} of 400 programs had an optimum");
}

#[test]
fn prop_the_dual_of_the_dual_returns_the_primal_value() {
    let mut rng = Rng::new(0x_0D0A_1002);
    let mut checked = 0usize;
    for _ in 0..200 {
        let p = random_bounded_lp(&mut rng);
        let Some(primal) = simplex(&p).unwrap().objective() else { continue };
        let d = lp_dual(&p).unwrap();
        let Some(dual) = simplex(&d).unwrap().objective() else { continue };
        checked += 1;
        assert!(
            (primal - dual).abs() < 1e-6 * (1.0 + primal.abs()),
            "primal {primal} against dual {dual}"
        );
        // Transposing twice returns to the original value.
        let back = simplex(&lp_dual(&d).unwrap()).unwrap().objective().unwrap();
        assert!(
            (primal - back).abs() < 1e-6 * (1.0 + primal.abs()),
            "dual of dual gave {back}, not {primal}"
        );
        assert_eq!(d.maximize, !p.maximize);
    }
    assert!(checked > 150, "only {checked} of 200 duals were solvable");
}

#[test]
fn prop_the_two_solvers_land_on_the_same_value() {
    // The simplex method walks the boundary and stops at a vertex; the
    // interior point method approaches through the middle and never reaches
    // one. They share only the standardisation step.
    let mut rng = Rng::new(0x_0117_1003);
    let mut compared = 0usize;
    for _ in 0..150 {
        let p = random_bounded_lp(&mut rng);
        let s = simplex(&p).unwrap();
        let i = interior_point(&p, 1e-9).unwrap();
        let (LpResult::Optimal { objective: so, .. }, LpResult::Optimal { objective: io, x: ix, .. }) =
            (&s, &i)
        else {
            continue;
        };
        compared += 1;
        assert!(
            (so - io).abs() < 1e-4 * (1.0 + so.abs()),
            "simplex {so} against interior point {io}"
        );
        assert!(p.is_feasible(ix, 1e-4), "the interior point answer is infeasible");
    }
    assert!(compared > 100, "only {compared} of 150 programs were comparable");
}

#[test]
fn prop_a_perturbed_right_hand_side_moves_the_objective_at_its_shadow_price() {
    // The definition the module commits to: duals[i] is d(objective)/d(b[i]).
    // Inside the reported range that derivative is exact, not approximate.
    let mut rng = Rng::new(0x_5E45_1004);
    let mut checked = 0usize;
    for _ in 0..120 {
        let p = random_bounded_lp(&mut rng);
        let LpResult::Optimal { objective, duals, .. } = simplex(&p).unwrap() else { continue };
        let Ok((c_ranges, b_ranges)) = sensitivity_ranges(&p) else { continue };
        assert_eq!(b_ranges.len(), p.m());
        assert_eq!(c_ranges.len(), p.n());

        for i in 0..p.m() {
            let (lo, hi) = b_ranges[i];
            assert!(
                lo <= p.b[i] + 1e-7 && hi >= p.b[i] - 1e-7,
                "row {i}: the current value {} sits outside its own range {lo}..{hi}",
                p.b[i]
            );
            for fraction in [0.4f64, -0.4] {
                let span = if fraction > 0.0 { hi - p.b[i] } else { p.b[i] - lo };
                if !span.is_finite() || span <= 1e-9 {
                    continue;
                }
                let delta = fraction.signum() * fraction.abs() * span;
                let mut q = p.clone();
                q.b[i] += delta;
                let Some(moved) = simplex(&q).unwrap().objective() else { continue };
                checked += 1;
                assert!(
                    (moved - objective - duals[i] * delta).abs() < 1e-6 * (1.0 + moved.abs()),
                    "row {i}, delta {delta}: objective {moved}, predicted {}",
                    objective + duals[i] * delta
                );
            }
        }
    }
    assert!(checked > 200, "only {checked} perturbations were exercised");
}

#[test]
fn prop_an_objective_coefficient_inside_its_range_leaves_the_vertex_alone() {
    let mut rng = Rng::new(0x_5E45_1005);
    let mut checked = 0usize;
    for _ in 0..120 {
        let p = random_bounded_lp(&mut rng);
        let LpResult::Optimal { x, .. } = simplex(&p).unwrap() else { continue };
        let Ok((c_ranges, _)) = sensitivity_ranges(&p) else { continue };

        for j in 0..p.n() {
            let (lo, hi) = c_ranges[j];
            assert!(
                lo <= p.c[j] + 1e-7 && hi >= p.c[j] - 1e-7,
                "coefficient {j}: {} sits outside its own range {lo}..{hi}",
                p.c[j]
            );
            for target in [lo, hi] {
                if !target.is_finite() || (target - p.c[j]).abs() < 1e-9 {
                    continue;
                }
                let mut q = p.clone();
                q.c[j] = p.c[j] + 0.9 * (target - p.c[j]);
                let Some(moved) = simplex(&q).unwrap().solution().map(<[f64]>::to_vec) else {
                    continue;
                };
                checked += 1;
                // The objective at the new coefficients, evaluated at the old
                // point, must be optimal -- the vertex has not moved.
                let held = q.objective_at(&x);
                let achieved = q.objective_at(&moved);
                assert!(
                    (held - achieved).abs() < 1e-6 * (1.0 + achieved.abs()),
                    "coefficient {j} at {}: the old vertex is worth {held}, the new one {achieved}",
                    q.c[j]
                );
            }
        }
    }
    assert!(checked > 100, "only {checked} coefficient perturbations were exercised");
}

#[test]
fn prop_a_zero_sum_game_has_a_value_neither_player_can_beat() {
    // Von Neumann's minimax theorem, which here is a corollary of duality:
    // the two players' programs are duals, so their values coincide. Against
    // the optimal strategy, no pure response does better than the value --
    // and since a mixed strategy is a convex combination of pure ones, that
    // covers every response.
    let mut rng = Rng::new(0x_6A3E_1006);
    for _ in 0..150 {
        let m = 2 + pick(&mut rng, 4);
        let n = 2 + pick(&mut rng, 4);
        let mut payoff = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                payoff.set(i, j, (rng.next_f64() * 10.0 - 5.0).round());
            }
        }
        let (row, column, value) = two_player_zero_sum_lp(&payoff).unwrap();

        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-7, "row strategy {row:?}");
        assert!((column.iter().sum::<f64>() - 1.0).abs() < 1e-7, "column {column:?}");
        assert!(row.iter().chain(&column).all(|&v| v > -1e-7), "a negative probability");

        for j in 0..n {
            let got: f64 = (0..m).map(|i| row[i] * payoff.get(i, j)).sum();
            assert!(got >= value - 1e-6, "column {j} held the row player to {got} below {value}");
        }
        for i in 0..m {
            let got: f64 = (0..n).map(|j| column[j] * payoff.get(i, j)).sum();
            assert!(got <= value + 1e-6, "row {i} earned {got} above {value}");
        }

        // A game and its transposed negation swap the players, so the value
        // negates: what one can guarantee, the other must concede.
        let mut mirrored = Matrix::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                mirrored.set(j, i, -payoff.get(i, j));
            }
        }
        let (_, _, mirror_value) = two_player_zero_sum_lp(&mirrored).unwrap();
        assert!(
            (mirror_value + value).abs() < 1e-6,
            "the mirrored game is worth {mirror_value}, not {}",
            -value
        );
    }
}

#[test]
fn prop_the_chebyshev_ball_fits_and_nothing_larger_does() {
    let mut rng = Rng::new(0x_C4E5_1007);
    for _ in 0..150 {
        let n = 2 + pick(&mut rng, 3);
        let m = n + 2 + pick(&mut rng, 4);
        // Half-spaces with outward normals spread over the sphere and a
        // positive offset, so the origin is always strictly inside.
        let mut a = Matrix::zeros(m, n);
        let mut b = vec![0.0; m];
        for i in 0..m {
            let mut norm = 0.0;
            for j in 0..n {
                let v = rng.next_gaussian();
                a.set(i, j, v);
                norm += v * v;
            }
            if norm <= 1e-12 {
                a.set(i, 0, 1.0);
            }
            b[i] = 1.0 + rng.next_f64() * 3.0;
        }
        let Ok((centre, radius)) = chebyshev_center(&a, &b) else { continue };
        if !radius.is_finite() {
            continue;
        }
        assert!(radius > 0.0, "the origin is inside, so the radius must be positive");

        // The ball fits: every face is at least `radius` from the centre.
        for i in 0..m {
            let norm: f64 = (0..n).map(|j| a.get(i, j) * a.get(i, j)).sum::<f64>().sqrt();
            let slack = b[i] - (0..n).map(|j| a.get(i, j) * centre[j]).sum::<f64>();
            assert!(
                slack / norm >= radius - 1e-6,
                "face {i} is only {} from the centre, less than the radius {radius}",
                slack / norm
            );
        }
        // And nothing larger does: some face is exactly `radius` away, or the
        // radius was not maximal.
        let touching = (0..m)
            .filter(|&i| {
                let norm: f64 = (0..n).map(|j| a.get(i, j) * a.get(i, j)).sum::<f64>().sqrt();
                let slack = b[i] - (0..n).map(|j| a.get(i, j) * centre[j]).sum::<f64>();
                (slack / norm - radius).abs() < 1e-6
            })
            .count();
        assert!(touching >= 1, "the ball touches no face, so it is not maximal");
    }
}

#[test]
fn prop_each_regression_minimises_the_norm_it_claims_to() {
    // The defining property of each fit, checked against the other and against
    // ordinary least squares: whichever norm a fit minimises, it must score no
    // worse than the other two under that norm.
    let mut rng = Rng::new(0x_2E62_1008);
    for _ in 0..60 {
        let n = 8 + pick(&mut rng, 20);
        let mut design = Matrix::zeros(n, 2);
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = rng.next_f64() * 10.0;
            design.set(i, 0, 1.0);
            design.set(i, 1, t);
            y[i] = 2.0 + 0.5 * t + rng.next_gaussian();
        }
        // One gross outlier, which is where the norms disagree most.
        y[pick(&mut rng, n)] += 40.0;

        let l1 = l1_regression_lp(&design, &y).unwrap();
        let li = linf_regression_lp(&design, &y).unwrap();
        let l2 = crate_least_squares(&design, &y);

        let residual = |beta: &[f64], i: usize| y[i] - beta[0] - beta[1] * design.get(i, 1);
        let sum_abs = |beta: &[f64]| (0..n).map(|i| residual(beta, i).abs()).sum::<f64>();
        let worst = |beta: &[f64]| {
            (0..n).map(|i| residual(beta, i).abs()).fold(0.0f64, f64::max)
        };

        assert!(sum_abs(&l1) <= sum_abs(&li) + 1e-6, "the L1 fit lost on the L1 norm");
        assert!(sum_abs(&l1) <= sum_abs(&l2) + 1e-6, "the L1 fit lost to least squares");
        assert!(worst(&li) <= worst(&l1) + 1e-6, "the minimax fit lost on the sup norm");
        assert!(worst(&li) <= worst(&l2) + 1e-6, "the minimax fit lost to least squares");

        // The minimax residual is attained at least three times for two
        // parameters -- the fit is pinned by its extreme points.
        let top = worst(&li);
        let attained =
            (0..n).filter(|&i| (residual(&li, i).abs() - top).abs() < 1e-6).count();
        assert!(attained >= 3, "only {attained} residuals reached the minimax value");
    }
}

/// Ordinary least squares, for comparison against the two LP fits.
fn crate_least_squares(design: &Matrix, y: &[f64]) -> Vec<f64> {
    rust_physics_engine::linalg::qr::least_squares(design, y).unwrap()
}

#[test]
fn prop_every_constraint_sense_is_handled_consistently() {
    // Mixed `<=`, `>=` and `=` rows, checked by the one thing that must hold
    // whatever the senses: the reported point is feasible and the shadow
    // prices reproduce the objective.
    let mut rng = Rng::new(0x_5E45_1009);
    let mut solved = 0usize;
    for _ in 0..300 {
        let m = 2 + pick(&mut rng, 3);
        let n = 2 + pick(&mut rng, 3);
        let mut a = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a.set(i, j, (rng.next_f64() * 4.0).round() + 1.0);
            }
        }
        let b: Vec<f64> = (0..m).map(|_| (rng.next_f64() * 20.0).round() + 5.0).collect();
        let c: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 8.0).round() + 1.0).collect();
        let senses: Vec<Cmp> = (0..m)
            .map(|_| match pick(&mut rng, 3) {
                0 => Cmp::Le,
                1 => Cmp::Ge,
                _ => Cmp::Eq,
            })
            .collect();
        let p = LpProblem {
            c,
            a,
            b,
            constraint_types: senses,
            bounds: vec![(0.0, f64::INFINITY); n],
            maximize: false,
        };
        let LpResult::Optimal { x, objective, duals, .. } = simplex(&p).unwrap() else {
            continue;
        };
        solved += 1;
        assert!(p.is_feasible(&x, 1e-6), "infeasible point {x:?} for senses {:?}", p.constraint_types);
        let by: f64 = p.b.iter().zip(&duals).map(|(a, b)| a * b).sum();
        assert!(
            (by - objective).abs() < 1e-6 * (1.0 + objective.abs()),
            "b . y = {by} against objective {objective}"
        );
    }
    assert!(solved > 100, "only {solved} of 300 mixed-sense programs had an optimum");
}
