//! Properties of the game theory module.
//!
//! Equilibrium has a certificate: no player gains by deviating, and since
//! payoffs are linear in one's own mixture, only pure deviations need
//! checking. That makes every equilibrium claim falsifiable on a random
//! instance, independent of the algorithm that produced it -- so the three
//! methods here are each measured against the definition rather than against
//! one another.
//!
//! The cooperative side is checked against its axioms, which are equations:
//! efficiency and symmetry hold exactly on every game, not typically.

use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::optimization::game_theory::{
    backward_induction, banzhaf_index, best_response, colonel_blotto_sim,
    correlated_equilibrium_lp, dominated_strategies, evolutionarily_stable_check,
    iterated_elimination, minimax_value, nash_2x2, nash_bimatrix_lemke_howson,
    nash_deviation_gain, nash_support_enumeration, replicator_dynamics, shapley_value,
    vcg_auction, GameTree,
};

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random payoff matrix with small integer entries, so that ties -- the
/// case degenerate-game handling turns on -- actually arise.
fn random_payoff(rng: &mut Rng, rows: usize, cols: usize, spread: f64) -> Matrix {
    let mut m = Matrix::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            m.set(r, c, (rng.next_f64() * spread).round() - spread / 2.0);
        }
    }
    m
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn prop_the_minimax_theorem_holds_on_every_random_zero_sum_game() {
    // The row player's guaranteed floor equals the column player's ceiling.
    // Both are recomputed here from the returned mixtures, so the solver's
    // own reported value is never taken on trust.
    let mut rng = Rng::new(0x_9A11_0001);
    for _ in 0..400 {
        let rows = 1 + pick(&mut rng, 5);
        let cols = 1 + pick(&mut rng, 5);
        let game = random_payoff(&mut rng, rows, cols, 12.0);
        let (value, p, q) = minimax_value(&game).unwrap();

        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-6, "the row mixture is {p:?}");
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-6, "the column mixture is {q:?}");
        assert!(p.iter().all(|v| *v >= -1e-9) && q.iter().all(|v| *v >= -1e-9));

        let floor = (0..cols)
            .map(|j| (0..rows).map(|i| p[i] * game.get(i, j)).sum::<f64>())
            .fold(f64::INFINITY, f64::min);
        let ceiling = (0..rows)
            .map(|i| dot(game.row(i), &q))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (floor - ceiling).abs() < 1e-6,
            "the floor {floor} and ceiling {ceiling} differ, which minimax forbids"
        );
        assert!((value - floor).abs() < 1e-6, "the reported value {value} is not the floor");

        // The value lies between the pure maximin and the pure minimax, and
        // mixing is what closes the gap between them.
        let pure_maximin = (0..rows)
            .map(|i| (0..cols).map(|j| game.get(i, j)).fold(f64::INFINITY, f64::min))
            .fold(f64::NEG_INFINITY, f64::max);
        let pure_minimax = (0..cols)
            .map(|j| (0..rows).map(|i| game.get(i, j)).fold(f64::NEG_INFINITY, f64::max))
            .fold(f64::INFINITY, f64::min);
        assert!(
            value >= pure_maximin - 1e-6 && value <= pure_minimax + 1e-6,
            "the value {value} escapes [{pure_maximin}, {pure_minimax}]"
        );
    }
}

#[test]
fn prop_transposing_and_negating_a_zero_sum_game_swaps_the_players() {
    // The column player's problem is the row player's on the negated
    // transpose, so the value must negate. Nothing in the solver enforces
    // this -- it solves the row program either way -- so it is a real check
    // on the duality it relies on.
    let mut rng = Rng::new(0x_9A11_0002);
    for _ in 0..300 {
        let rows = 1 + pick(&mut rng, 4);
        let cols = 1 + pick(&mut rng, 4);
        let game = random_payoff(&mut rng, rows, cols, 10.0);
        let (value, _, _) = minimax_value(&game).unwrap();
        let (mirrored, _, _) = minimax_value(&game.transpose().scale(-1.0)).unwrap();
        assert!(
            (value + mirrored).abs() < 1e-6,
            "the value is {value} one way and {mirrored} the other"
        );

        // Adding a constant to every payoff shifts the value by it.
        let shifted = Matrix::from_fn(rows, cols, |i, j| game.get(i, j) + 7.0);
        let (bumped, _, _) = minimax_value(&shifted).unwrap();
        assert!((bumped - value - 7.0).abs() < 1e-6, "shifting moved the value to {bumped}");
    }
}

#[test]
fn prop_dominated_strategies_carry_no_weight_and_elimination_preserves_equilibria() {
    // Strict domination is the elimination that is safe. Both halves are
    // checked: a dominated row gets zero weight in the optimal mixture, and
    // what survives iterated elimination still contains the equilibrium.
    let mut rng = Rng::new(0x_9A11_0003);
    let mut with_dominated = 0usize;
    for _ in 0..300 {
        let rows = 2 + pick(&mut rng, 4);
        let cols = 2 + pick(&mut rng, 4);
        let game = random_payoff(&mut rng, rows, cols, 10.0);
        let dominated = dominated_strategies(&game);
        if !dominated.is_empty() {
            with_dominated += 1;
        }
        // The claim restated: some other row beats it everywhere.
        for &i in &dominated {
            assert!(
                (0..rows).any(|k| {
                    k != i && (0..cols).all(|j| game.get(k, j) > game.get(i, j) + 1e-9)
                }),
                "row {i} was reported dominated but nothing dominates it"
            );
        }
        let (_, p, _) = minimax_value(&game).unwrap();
        for &i in &dominated {
            assert!(p[i].abs() < 1e-6, "the dominated row {i} carries weight {}", p[i]);
        }

        // Iterated elimination on the zero-sum pair keeps the support.
        let (surviving_rows, surviving_cols) =
            iterated_elimination(&game, &game.scale(-1.0)).unwrap();
        assert!(!surviving_rows.is_empty() && !surviving_cols.is_empty());
        let (_, p, q) = minimax_value(&game).unwrap();
        for i in 0..rows {
            if p[i] > 1e-6 {
                assert!(surviving_rows.contains(&i), "row {i} is played but was eliminated");
            }
        }
        for j in 0..cols {
            if q[j] > 1e-6 {
                assert!(surviving_cols.contains(&j), "column {j} is played but was eliminated");
            }
        }
    }
    assert!(with_dominated > 20, "only {with_dominated} games had a dominated row");
}

#[test]
fn prop_every_method_returns_a_profile_no_player_wants_to_leave() {
    // The definition of equilibrium, applied to the output of three
    // independent methods on the same random games.
    let mut rng = Rng::new(0x_9A11_0004);
    let mut lemke_solved = 0usize;
    let mut mixed_seen = 0usize;
    for _ in 0..300 {
        let a = random_payoff(&mut rng, 2, 2, 8.0);
        let b = random_payoff(&mut rng, 2, 2, 8.0);

        let all = nash_2x2(&a, &b).unwrap();
        assert!(!all.is_empty(), "Nash's theorem guarantees one: {a:?} against {b:?}");
        for (p, q) in &all {
            let gain = nash_deviation_gain(&a, &b, p, q).unwrap();
            assert!(gain < 1e-6, "nash_2x2 returned {p:?}, {q:?} with a gain of {gain}");
            if p[0] > 1e-6 && p[0] < 1.0 - 1e-6 {
                mixed_seen += 1;
            }
        }

        let enumerated = nash_support_enumeration(&a, &b, 2).unwrap();
        for (p, q) in &enumerated {
            let gain = nash_deviation_gain(&a, &b, p, q).unwrap();
            assert!(gain < 1e-6, "support enumeration returned a gain of {gain}");
        }
        assert!(!enumerated.is_empty(), "enumeration found nothing on {a:?}, {b:?}");

        // Lemke-Howson refuses degenerate games rather than guessing, so a
        // failure is allowed; a wrong answer is not.
        for label in 0..4 {
            if let Ok((p, q)) = nash_bimatrix_lemke_howson(&a, &b, label) {
                let gain = nash_deviation_gain(&a, &b, &p, &q).unwrap();
                assert!(
                    gain < 1e-6,
                    "Lemke-Howson from label {label} returned {p:?}, {q:?} with a gain of {gain}"
                );
                lemke_solved += 1;
            }
        }
    }
    assert!(lemke_solved > 300, "Lemke-Howson only solved {lemke_solved} of 1200 attempts");
    assert!(mixed_seen > 50, "only {mixed_seen} genuinely mixed equilibria arose");
}

#[test]
fn prop_a_best_response_is_the_only_thing_worth_playing() {
    // Every reported best response ties for the maximum, nothing outside the
    // set reaches it, and any mixture over the set earns the same as any
    // single member of it -- which is exactly the indifference that mixed
    // equilibria rest on.
    let mut rng = Rng::new(0x_9A11_0005);
    for _ in 0..400 {
        let rows = 1 + pick(&mut rng, 5);
        let cols = 1 + pick(&mut rng, 5);
        let payoff = random_payoff(&mut rng, rows, cols, 8.0);
        let raw: Vec<f64> = (0..cols).map(|_| rng.next_f64()).collect();
        let total: f64 = raw.iter().sum();
        let opponent: Vec<f64> = raw.iter().map(|v| v / total).collect();

        let best = best_response(&payoff, &opponent);
        assert!(!best.is_empty(), "there is always a best response");
        let values: Vec<f64> = (0..rows).map(|i| dot(payoff.row(i), &opponent)).collect();
        let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for i in 0..rows {
            assert_eq!(
                best.contains(&i),
                values[i] >= peak - 1e-9,
                "row {i} is worth {} against a peak of {peak}",
                values[i]
            );
        }

        // Any mixture over the set is worth the same.
        let weights: Vec<f64> = best.iter().map(|_| rng.next_f64() + 1e-9).collect();
        let weight_total: f64 = weights.iter().sum();
        let mixed: f64 = best
            .iter()
            .zip(&weights)
            .map(|(&i, w)| w / weight_total * values[i])
            .sum();
        assert!((mixed - peak).abs() < 1e-9, "a mixture of best responses is worth {mixed}");
    }
}

#[test]
fn prop_a_correlated_equilibrium_obeys_its_incentive_constraints() {
    // The distribution is a distribution, and obeying the recommendation
    // beats every deviation conditional on receiving it. Both are linear
    // conditions and both are exact.
    let mut rng = Rng::new(0x_9A11_0006);
    for _ in 0..150 {
        let rows = 2 + pick(&mut rng, 2);
        let cols = 2 + pick(&mut rng, 2);
        let a = random_payoff(&mut rng, rows, cols, 8.0);
        let b = random_payoff(&mut rng, rows, cols, 8.0);
        let Ok(joint) = correlated_equilibrium_lp(&a, &b) else {
            continue;
        };

        let mass: f64 = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .map(|(i, j)| joint.get(i, j))
            .sum();
        assert!((mass - 1.0).abs() < 1e-6, "the distribution has mass {mass}");
        assert!((0..rows).all(|i| (0..cols).all(|j| joint.get(i, j) >= -1e-9)));

        for i in 0..rows {
            for k in 0..cols.min(rows) {
                let gain: f64 =
                    (0..cols).map(|j| joint.get(i, j) * (a.get(k, j) - a.get(i, j))).sum();
                assert!(gain <= 1e-6, "the row player gains {gain} by playing {k} when told {i}");
            }
        }
        for j in 0..cols {
            for l in 0..cols {
                let gain: f64 =
                    (0..rows).map(|i| joint.get(i, j) * (b.get(i, l) - b.get(i, j))).sum();
                assert!(gain <= 1e-6, "the column player gains {gain} by playing {l}");
            }
        }
    }
}

#[test]
fn prop_the_replicator_dynamic_keeps_its_iterates_on_the_simplex() {
    // The simplex is invariant under the flow, and a strategy that starts at
    // zero can never appear -- growth is proportional to current share, which
    // is what makes this a model of reproduction rather than of learning.
    let mut rng = Rng::new(0x_9A11_0007);
    for _ in 0..150 {
        let n = 2 + pick(&mut rng, 3);
        let payoff = random_payoff(&mut rng, n, n, 6.0);
        let raw: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let total: f64 = raw.iter().sum();
        let mut start: Vec<f64> = raw.iter().map(|v| v / total).collect();
        // Extinguish one strategy on half the draws.
        let extinct = if rng.next_f64() < 0.5 {
            let k = pick(&mut rng, n);
            start[k] = 0.0;
            let renormalise: f64 = start.iter().sum();
            for v in &mut start {
                *v /= renormalise;
            }
            Some(k)
        } else {
            None
        };

        let trajectory = replicator_dynamics(&payoff, &start, 8.0, 0.01).unwrap();
        for state in &trajectory {
            assert!((state.iter().sum::<f64>() - 1.0).abs() < 1e-8, "left the simplex: {state:?}");
            assert!(state.iter().all(|v| *v >= -1e-12), "went negative: {state:?}");
            if let Some(k) = extinct {
                assert!(state[k] < 1e-12, "an extinct strategy reappeared at {}", state[k]);
            }
        }

        // A vertex of the simplex is a fixed point whatever the payoffs.
        let mut vertex = vec![0.0; n];
        vertex[pick(&mut rng, n)] = 1.0;
        for state in &replicator_dynamics(&payoff, &vertex, 5.0, 0.01).unwrap() {
            for (a, b) in state.iter().zip(&vertex) {
                assert!((a - b).abs() < 1e-9, "a vertex moved to {state:?}");
            }
        }
    }
}

#[test]
fn prop_an_evolutionarily_stable_strategy_is_a_symmetric_equilibrium() {
    // Stability is strictly stronger than equilibrium, so anything the check
    // passes must also survive the deviation test against itself. The
    // converse fails, and the failures are what make the concept worth
    // having.
    let mut rng = Rng::new(0x_9A11_0008);
    let mut stable_seen = 0usize;
    let mut unstable_equilibria = 0usize;
    for _ in 0..400 {
        let n = 2 + pick(&mut rng, 2);
        let payoff = random_payoff(&mut rng, n, n, 6.0);
        // Test the pure strategies, where both conditions are cheap to state.
        for k in 0..n {
            let mut strategy = vec![0.0; n];
            strategy[k] = 1.0;
            let stable = evolutionarily_stable_check(&payoff, &strategy, 1e-9).unwrap();
            let equilibrium =
                nash_deviation_gain(&payoff, &payoff.transpose(), &strategy, &strategy).unwrap()
                    < 1e-9;
            if stable {
                stable_seen += 1;
                assert!(equilibrium, "strategy {k} is called stable but is not an equilibrium");
            } else if equilibrium {
                unstable_equilibria += 1;
            }
        }
    }
    assert!(stable_seen > 50, "only {stable_seen} stable strategies arose");
    assert!(
        unstable_equilibria > 10,
        "only {unstable_equilibria} equilibria failed stability, so the extra condition is untested"
    );
}

#[test]
fn prop_the_shapley_value_is_efficient_symmetric_and_null_respecting() {
    // The three axioms, on random characteristic functions. Efficiency is an
    // equation, symmetry is checked by constructing interchangeable players,
    // and a null player is added to each game explicitly.
    let mut rng = Rng::new(0x_9A11_0009);
    for _ in 0..200 {
        let n = 2 + pick(&mut rng, 4);
        let weights: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 6.0).round()).collect();
        let curvature = rng.next_f64() * 2.0;
        let characteristic = |c: u64| -> f64 {
            let total: f64 = (0..n).filter(|&i| c >> i & 1 == 1).map(|i| weights[i]).sum();
            total + curvature * total * total / 10.0
        };
        let values = shapley_value(&characteristic, n).unwrap();
        let grand = (1u64 << n) - 1;
        assert!(
            (values.iter().sum::<f64>() - characteristic(grand)).abs() < 1e-9,
            "efficiency fails: {values:?} against {}",
            characteristic(grand)
        );
        // Equal weights mean interchangeable players, so equal values.
        for i in 0..n {
            for j in 0..n {
                if (weights[i] - weights[j]).abs() < 1e-12 {
                    assert!(
                        (values[i] - values[j]).abs() < 1e-9,
                        "symmetry fails between {i} and {j}: {values:?}"
                    );
                }
            }
        }

        // Adding a player who contributes nothing anywhere gives them zero
        // and leaves everyone else untouched.
        if n < 6 {
            let extended = |c: u64| characteristic(c & grand);
            let bigger = shapley_value(&extended, n + 1).unwrap();
            assert!(bigger[n].abs() < 1e-9, "the null player got {}", bigger[n]);
            for i in 0..n {
                assert!(
                    (bigger[i] - values[i]).abs() < 1e-9,
                    "adding a null player moved player {i}"
                );
            }
        }

        // The Banzhaf index shares efficiency's normalisation but not its
        // values, and both are non-negative on a monotone game. A game where
        // every coalition is worth nothing -- which the random weights do
        // produce -- has no power to apportion and comes back as zeros.
        let banzhaf = banzhaf_index(&characteristic, n).unwrap();
        assert!(banzhaf.iter().all(|v| *v >= -1e-9), "{banzhaf:?}");
        let mass: f64 = banzhaf.iter().sum();
        if characteristic(grand).abs() > 1e-12 {
            assert!((mass - 1.0).abs() < 1e-9, "{banzhaf:?} does not sum to one");
        } else {
            assert!(mass.abs() < 1e-12, "a game worth nothing apportioned {mass}");
        }
    }
}

#[test]
fn prop_vcg_is_efficient_and_never_charges_more_than_a_bidder_bid() {
    // Two guarantees that hold on every instance: the assignment maximises
    // total value, checked against every alternative; and no winner pays
    // above their own bid, so truthful bidding is never regretted.
    let mut rng = Rng::new(0x_9A11_000A);
    for _ in 0..200 {
        let bidders = 2 + pick(&mut rng, 3);
        let items = 1 + pick(&mut rng, 3);
        let bids: Vec<Vec<f64>> = (0..bidders)
            .map(|_| (0..items).map(|_| (rng.next_f64() * 10.0).round()).collect())
            .collect();
        let (assignment, payments) = vcg_auction(&bids, items).unwrap();

        // No item goes to two bidders.
        let mut taken = vec![false; items];
        for choice in assignment.iter().flatten() {
            assert!(!taken[*choice], "item {choice} was assigned twice");
            taken[*choice] = true;
        }

        let welfare: f64 = assignment
            .iter()
            .enumerate()
            .filter_map(|(i, choice)| choice.map(|k| bids[i][k]))
            .sum();
        // Exhaustive comparison: every assignment of items to distinct
        // bidders, encoded as a choice per item.
        let mut best = 0.0f64;
        let mut counters = vec![0usize; items];
        loop {
            let mut used = vec![false; bidders];
            let mut total = 0.0;
            let mut valid = true;
            for (k, &choice) in counters.iter().enumerate() {
                if choice == bidders {
                    continue;
                }
                if used[choice] {
                    valid = false;
                    break;
                }
                used[choice] = true;
                total += bids[choice][k];
            }
            if valid {
                best = best.max(total);
            }
            let mut carry = 0usize;
            while carry < items {
                counters[carry] += 1;
                if counters[carry] <= bidders {
                    break;
                }
                counters[carry] = 0;
                carry += 1;
            }
            if carry == items {
                break;
            }
        }
        assert!(
            welfare >= best - 1e-9,
            "the assignment is worth {welfare}, below the best {best}"
        );

        for (i, choice) in assignment.iter().enumerate() {
            assert!(payments[i] >= -1e-9, "bidder {i} was paid {}", payments[i]);
            match choice {
                Some(k) => assert!(
                    payments[i] <= bids[i][*k] + 1e-9,
                    "bidder {i} paid {} for something they valued at {}",
                    payments[i],
                    bids[i][*k]
                ),
                None => assert!(payments[i] < 1e-9, "a loser paid {}", payments[i]),
            }
        }
    }
}

#[test]
fn prop_backward_induction_returns_a_path_that_reaches_its_own_payoffs() {
    // The reported payoffs must be the leaf the reported path arrives at, and
    // at every decision node the mover's payoff must be the best available.
    // Both are exact and both catch the errors that actually happen.
    let mut rng = Rng::new(0x_9A11_000B);

    fn build(rng: &mut Rng, depth: usize, players: usize) -> GameTree {
        if depth == 0 {
            return GameTree::Leaf(
                (0..players).map(|_| (rng.next_f64() * 20.0).round() - 10.0).collect(),
            );
        }
        let branching = 2 + pick(rng, 2);
        GameTree::Node {
            player: pick(rng, players),
            children: (0..branching).map(|_| build(rng, depth - 1, players)).collect(),
        }
    }

    for _ in 0..300 {
        let players = 2 + pick(&mut rng, 2);
        let depth = 1 + pick(&mut rng, 3);
        let tree = build(&mut rng, depth, players);
        let (path, payoffs) = backward_induction(&tree).unwrap();

        // Walk the path and confirm it lands on those payoffs.
        let mut node = &tree;
        for &step in &path {
            match node {
                GameTree::Node { children, .. } => {
                    assert!(step < children.len(), "the path leaves the tree");
                    node = &children[step];
                }
                GameTree::Leaf(_) => panic!("the path continues past a leaf"),
            }
        }
        match node {
            GameTree::Leaf(reached) => {
                assert_eq!(reached, &payoffs, "the path does not reach the reported payoffs");
            }
            GameTree::Node { .. } => panic!("the path stops short of a leaf"),
        }

        // At the root, no other child gives the mover more.
        if let GameTree::Node { player, children } = &tree {
            for child in children {
                let (_, alternative) = backward_induction(child).unwrap();
                assert!(
                    alternative[*player] <= payoffs[*player] + 1e-9,
                    "the mover could have had {} instead of {}",
                    alternative[*player],
                    payoffs[*player]
                );
            }
        }
    }
}

#[test]
fn prop_no_blotto_allocation_dominates_and_the_result_is_antisymmetric() {
    // Beating and being beaten are mirror images, and no sampled allocation
    // beats every other -- the absence of a dominant plan is the game.
    let mut rng = Rng::new(0x_9A11_000C);
    for _ in 0..100 {
        let fields = 2 + pick(&mut rng, 4);
        let troops = fields * (2 + pick(&mut rng, 20));
        let count = 6 + pick(&mut rng, 10);
        let table = colonel_blotto_sim(fields, troops, count, &mut rng).unwrap();
        assert_eq!((table.rows, table.cols), (count, count));
        for i in 0..count {
            assert!(table.get(i, i).abs() < 1e-12, "a plan beat itself");
            for j in 0..count {
                assert!(
                    (table.get(i, j) + table.get(j, i)).abs() < 1e-12,
                    "the table is not antisymmetric at ({i}, {j})"
                );
                assert!(table.get(i, j).abs() <= fields as f64 + 1e-12);
            }
        }
        // The total score across all pairings is zero, since it is a
        // zero-sum contest however the troops are split.
        let grand: f64 = (0..count)
            .flat_map(|i| (0..count).map(move |j| (i, j)))
            .map(|(i, j)| table.get(i, j))
            .sum();
        assert!(grand.abs() < 1e-9, "the scores sum to {grand}");
    }
}
