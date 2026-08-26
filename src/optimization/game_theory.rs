//! Game theory: equilibria, dynamics, cooperative solution concepts,
//! auctions, and two-player search.
//!
//! The organising fact of the non-cooperative half is that equilibrium is a
//! *fixed-point* condition and not an optimisation: no player is optimising
//! against a fixed environment, because the environment is the other players
//! doing the same thing. That is why the zero-sum case is easy and the
//! general case is not. In a zero-sum game the two players' problems are
//! linear programs dual to each other, so von Neumann's minimax theorem is a
//! corollary of LP duality and the equilibrium is computable in polynomial
//! time. In a bimatrix game there is no such dual, the equilibrium set can be
//! disconnected, and the best general algorithms are pivoting schemes with
//! exponential worst cases.
//!
//! The cooperative half asks a different question -- not what players will do
//! but how a surplus they have already agreed to create should be split --
//! and its solution concepts are axiomatic. The Shapley value is the unique
//! allocation satisfying efficiency, symmetry, the null-player property and
//! additivity; the core is the set of allocations no coalition can improve
//! on; and the two can be disjoint, since a game can have an empty core while
//! the Shapley value always exists.

use crate::error::GeomError;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;
use crate::optimization::lp::{simplex, two_player_zero_sum_lp, Cmp, LpProblem, LpResult};

/// Tolerance for treating a payoff difference or a probability as zero.
const GAME_TOL: f64 = 1e-9;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The expected payoff to the row player under mixed strategies `p` and `q`.
fn bilinear(m: &Matrix, p: &[f64], q: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..m.rows {
        for j in 0..m.cols {
            total += p[i] * m.get(i, j) * q[j];
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Zero-sum games
// ---------------------------------------------------------------------------

/// The value of a two-player zero-sum game and the optimal mixed strategies,
/// as `(value, row strategy, column strategy)`.
///
/// The row player's guaranteed floor and the column player's guaranteed
/// ceiling coincide. That coincidence is the minimax theorem, and it is not
/// assumed here: the two players' programs are LP duals, so strong duality
/// delivers it. What makes the result surprising is that it fails without
/// mixing -- in matching pennies the pure maximin is -1 and the pure minimax
/// is +1 -- so the theorem is really a statement about the power of
/// randomisation.
///
/// # Errors
/// Returns an error if the underlying program has no optimum, which for a
/// finite game means a numerical failure rather than a modelling one.
pub fn minimax_value(payoff: &Matrix) -> Result<(f64, Vec<f64>, Vec<f64>), GeomError> {
    let (row, column, value) = two_player_zero_sum_lp(payoff)?;
    Ok((value, row, column))
}

/// The row indices strictly dominated by some other pure row.
///
/// Strict domination is the one elimination that is always safe: a strictly
/// dominated strategy is played with probability zero in every equilibrium,
/// so removing it removes no equilibria. *Weak* domination does not have that
/// property, which is why only the strict version is offered.
#[must_use]
pub fn dominated_strategies(payoff: &Matrix) -> Vec<usize> {
    (0..payoff.rows)
        .filter(|&i| {
            (0..payoff.rows).any(|k| {
                k != i && (0..payoff.cols).all(|j| payoff.get(k, j) > payoff.get(i, j) + GAME_TOL)
            })
        })
        .collect()
}

/// Iterated elimination of strictly dominated strategies, returning the row
/// and column indices that survive.
///
/// The order of elimination does not matter for strict domination: the
/// surviving set is the same however the eliminations are sequenced. That is
/// a genuine theorem and it is what makes the procedure well defined -- the
/// weak-domination analogue is order dependent and so is not a solution
/// concept at all.
///
/// `a` is the row player's payoff and `b` the column player's.
///
/// # Errors
/// Returns an error if the two payoff matrices have different shapes.
pub fn iterated_elimination(
    a: &Matrix,
    b: &Matrix,
) -> Result<(Vec<usize>, Vec<usize>), GeomError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(GeomError::InvalidArgument("iterated_elimination: shape mismatch"));
    }
    let mut rows: Vec<usize> = (0..a.rows).collect();
    let mut cols: Vec<usize> = (0..a.cols).collect();

    loop {
        let before = (rows.len(), cols.len());

        // A row is dominated when some other surviving row beats it against
        // every surviving column. The survivors are computed against the
        // whole current set before any of them is removed, so that within one
        // sweep the eliminations do not depend on their own order.
        let surviving_rows: Vec<usize> = rows
            .iter()
            .copied()
            .filter(|&i| {
                !rows.iter().any(|&k| {
                    k != i && cols.iter().all(|&j| a.get(k, j) > a.get(i, j) + GAME_TOL)
                })
            })
            .collect();
        let surviving_cols: Vec<usize> = cols
            .iter()
            .copied()
            .filter(|&j| {
                !cols.iter().any(|&l| {
                    l != j && rows.iter().all(|&i| b.get(i, l) > b.get(i, j) + GAME_TOL)
                })
            })
            .collect();
        rows = surviving_rows;
        cols = surviving_cols;

        if (rows.len(), cols.len()) == before {
            return Ok((rows, cols));
        }
    }
}

/// The pure best responses to an opponent's mixed strategy.
///
/// Returns every index attaining the maximum, not just one. The set matters:
/// a mixed equilibrium exists precisely because a player is indifferent among
/// several best responses, so an implementation that returned a single index
/// would be unable to express one.
///
/// `payoff` is the responding player's own payoff matrix, with the responder
/// indexing rows.
///
/// # Panics
/// Panics if the opponent's strategy has the wrong length.
#[must_use]
pub fn best_response(payoff: &Matrix, opponent_mixed: &[f64]) -> Vec<usize> {
    assert_eq!(opponent_mixed.len(), payoff.cols, "the opponent's strategy has the wrong length");
    let values: Vec<f64> =
        (0..payoff.rows).map(|i| dot(payoff.row(i), opponent_mixed)).collect();
    let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (0..payoff.rows).filter(|&i| values[i] >= best - GAME_TOL).collect()
}

/// The largest gain any player could get by deviating unilaterally from the
/// given strategy profile.
///
/// Zero -- to tolerance -- is exactly the definition of a Nash equilibrium,
/// so this is the certificate that any equilibrium-finding routine should be
/// held to. A deviation only ever needs to be checked against *pure*
/// strategies, since the payoff is linear in one's own mixture and a linear
/// function on a simplex attains its maximum at a vertex.
///
/// # Errors
/// Returns an error on a shape mismatch between the payoffs and the profile.
pub fn nash_deviation_gain(
    a: &Matrix,
    b: &Matrix,
    p: &[f64],
    q: &[f64],
) -> Result<f64, GeomError> {
    if a.rows != b.rows || a.cols != b.cols || p.len() != a.rows || q.len() != a.cols {
        return Err(GeomError::InvalidArgument("nash_deviation_gain: shape mismatch"));
    }
    let row_value = bilinear(a, p, q);
    let column_value = bilinear(b, p, q);
    let row_best = (0..a.rows)
        .map(|i| dot(a.row(i), q))
        .fold(f64::NEG_INFINITY, f64::max);
    let column_best = (0..b.cols)
        .map(|j| (0..b.rows).map(|i| p[i] * b.get(i, j)).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((row_best - row_value).max(column_best - column_value).max(0.0))
}

// ---------------------------------------------------------------------------
// Bimatrix equilibria
// ---------------------------------------------------------------------------

/// Every Nash equilibrium of a 2x2 bimatrix game, pure and mixed.
///
/// Small enough to enumerate completely, which makes it the reference the
/// general algorithms are checked against. The mixed equilibrium, when it
/// exists, has the property that trips people up: each player's mixture is
/// chosen to make the *opponent* indifferent, not themselves. One's own
/// payoff plays no part in one's own probabilities.
///
/// # Errors
/// Returns an error unless both matrices are 2x2.
pub fn nash_2x2(a: &Matrix, b: &Matrix) -> Result<Vec<(Vec<f64>, Vec<f64>)>, GeomError> {
    if a.rows != 2 || a.cols != 2 || b.rows != 2 || b.cols != 2 {
        return Err(GeomError::InvalidArgument("nash_2x2 requires two 2x2 matrices"));
    }
    let mut found: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut push = |p: Vec<f64>, q: Vec<f64>| {
        if nash_deviation_gain(a, b, &p, &q).unwrap_or(f64::INFINITY) < 1e-7
            && !found.iter().any(|(x, y): &(Vec<f64>, Vec<f64>)| {
                x.iter().zip(&p).all(|(u, v)| (u - v).abs() < 1e-6)
                    && y.iter().zip(&q).all(|(u, v)| (u - v).abs() < 1e-6)
            })
        {
            found.push((p, q));
        }
    };

    // The four pure profiles.
    for i in 0..2 {
        for j in 0..2 {
            let p = vec![if i == 0 { 1.0 } else { 0.0 }, if i == 1 { 1.0 } else { 0.0 }];
            let q = vec![if j == 0 { 1.0 } else { 0.0 }, if j == 1 { 1.0 } else { 0.0 }];
            push(p, q);
        }
    }

    // The fully mixed profile: the row player mixes to equalise the column
    // player's two payoffs and vice versa.
    let row_denominator = b.get(0, 0) - b.get(0, 1) - b.get(1, 0) + b.get(1, 1);
    let column_denominator = a.get(0, 0) - a.get(0, 1) - a.get(1, 0) + a.get(1, 1);
    if row_denominator.abs() > GAME_TOL && column_denominator.abs() > GAME_TOL {
        let p0 = (b.get(1, 1) - b.get(1, 0)) / row_denominator;
        let q0 = (a.get(1, 1) - a.get(0, 1)) / column_denominator;
        if (0.0..=1.0).contains(&p0) && (0.0..=1.0).contains(&q0) {
            push(vec![p0, 1.0 - p0], vec![q0, 1.0 - q0]);
        }
    }
    Ok(found)
}

/// Nash equilibria by support enumeration.
///
/// For each pair of candidate supports, the indifference conditions are a
/// linear system: every strategy in a player's support must earn the same
/// expected payoff, and the probabilities must sum to one. Solving it and
/// then *checking* the result -- non-negative probabilities, and no
/// unsupported strategy earning more -- is what makes the method sound. The
/// checking is not optional bookkeeping: most supports produce a solution to
/// the linear system that is not an equilibrium at all.
///
/// Exponential in the number of strategies, so `max_support` bounds the
/// support size considered.
///
/// # Errors
/// Returns an error on a shape mismatch.
pub fn nash_support_enumeration(
    a: &Matrix,
    b: &Matrix,
    max_support: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, GeomError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(GeomError::InvalidArgument("nash_support_enumeration: shape mismatch"));
    }
    let (m, n) = (a.rows, a.cols);
    let cap = max_support.max(1);
    let mut found: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

    for row_mask in 1u64..(1u64 << m) {
        let row_support: Vec<usize> = (0..m).filter(|&i| row_mask >> i & 1 == 1).collect();
        if row_support.len() > cap {
            continue;
        }
        for column_mask in 1u64..(1u64 << n) {
            let column_support: Vec<usize> =
                (0..n).filter(|&j| column_mask >> j & 1 == 1).collect();
            if column_support.len() != row_support.len() || column_support.len() > cap {
                // Supports of unequal size only give equilibria in degenerate
                // games, where the indifference system is not square.
                continue;
            }

            let Some(q) = indifference_solve(a, &row_support, &column_support, n, false) else {
                continue;
            };
            let Some(p) = indifference_solve(b, &column_support, &row_support, m, true) else {
                continue;
            };
            if nash_deviation_gain(a, b, &p, &q)? < 1e-7
                && !found.iter().any(|(x, y)| {
                    x.iter().zip(&p).all(|(u, v)| (u - v).abs() < 1e-6)
                        && y.iter().zip(&q).all(|(u, v)| (u - v).abs() < 1e-6)
                })
            {
                found.push((p, q));
            }
        }
    }
    Ok(found)
}

/// Solves the indifference conditions that make `payoff`'s row player
/// indifferent across `own_support`, returning the opponent's mixture over
/// `opponent_support` padded to length `width`.
///
/// With `transposed` the roles of the matrix's two indices are swapped, which
/// is how the same routine serves both players.
fn indifference_solve(
    payoff: &Matrix,
    own_support: &[usize],
    opponent_support: &[usize],
    width: usize,
    transposed: bool,
) -> Option<Vec<f64>> {
    let k = opponent_support.len();
    if own_support.len() != k {
        return None;
    }
    let entry = |i: usize, j: usize| -> f64 {
        if transposed {
            payoff.get(j, i)
        } else {
            payoff.get(i, j)
        }
    };

    // Unknowns: the k opponent probabilities. Equations: k - 1 indifference
    // differences, plus the normalisation.
    let mut m = vec![vec![0.0f64; k]; k];
    let mut rhs = vec![0.0f64; k];
    for r in 0..k - 1 {
        for (c, &j) in opponent_support.iter().enumerate() {
            m[r][c] = entry(own_support[r], j) - entry(own_support[r + 1], j);
        }
    }
    for c in 0..k {
        m[k - 1][c] = 1.0;
    }
    rhs[k - 1] = 1.0;

    let solution = gaussian_solve(&mut m, &mut rhs)?;
    if solution.iter().any(|v| *v < -GAME_TOL) {
        return None;
    }
    let mut padded = vec![0.0; width];
    for (c, &j) in opponent_support.iter().enumerate() {
        padded[j] = solution[c].max(0.0);
    }
    let total: f64 = padded.iter().sum();
    if (total - 1.0).abs() > 1e-7 {
        return None;
    }
    Some(padded)
}

/// Gaussian elimination with partial pivoting, returning `None` when the
/// system is singular.
fn gaussian_solve(m: &mut [Vec<f64>], rhs: &mut [f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&x, &y| {
            m[x][col].abs().partial_cmp(&m[y][col].abs()).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        for r in (col + 1)..n {
            let factor = m[r][col] / m[col][col];
            for c in col..n {
                m[r][c] -= factor * m[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..n {
            acc -= m[i][j] * x[j];
        }
        x[i] = acc / m[i][i];
    }
    Some(x)
}

/// One Nash equilibrium of a bimatrix game by the Lemke-Howson algorithm.
///
/// Complementary pivoting on the two players' best-response polytopes. Every
/// vertex pair is labelled by the strategies that are either unplayed or
/// unprofitable; a pair carrying all labels is an equilibrium, and the
/// algorithm walks an edge path from the artificial equilibrium at the origin
/// to one that does. The path cannot revisit a vertex and the polytopes are
/// finite, so it terminates -- which is a constructive proof that a Nash
/// equilibrium exists, independent of Kakutani's fixed-point theorem.
///
/// `initial_label` selects which strategy's label is dropped to start the
/// path; different choices generally reach different equilibria.
///
/// # Errors
/// Returns an error on a shape mismatch, an out-of-range label, or a
/// degenerate game where the pivot becomes ambiguous.
pub fn nash_bimatrix_lemke_howson(
    a: &Matrix,
    b: &Matrix,
    initial_label: usize,
) -> Result<(Vec<f64>, Vec<f64>), GeomError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(GeomError::InvalidArgument("lemke_howson: shape mismatch"));
    }
    let (m, n) = (a.rows, a.cols);
    if initial_label >= m + n {
        return Err(GeomError::InvalidArgument("lemke_howson: the label is out of range"));
    }

    // Shift both payoffs strictly positive. The polytopes below are only
    // bounded when the payoffs are, and shifting changes neither player's
    // preferences and so leaves the equilibrium set alone.
    let lowest = (0..m)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| a.get(i, j).min(b.get(i, j)))
        .fold(f64::INFINITY, f64::min);
    let shift = 1.0 - lowest;

    // Two tableaux in the standard form of Lemke-Howson: the row player's
    // slack variables are labels 0..m and the column player's are m..m+n.
    // Tableau one has basis m..m+n, tableau two has basis 0..m.
    let mut basis: Vec<usize> = (m..m + n).chain(0..m).collect();
    // Rows: n for the first tableau then m for the second. Columns are all
    // m + n labels plus the constant term.
    let total = m + n;
    let mut tableau = vec![vec![0.0f64; total + 1]; total];
    // The row player's polytope carries the *column* player's payoffs and
    // vice versa, and getting that round the wrong way is silent: the
    // algorithm still terminates at a completely labelled vertex pair, which
    // is then not an equilibrium of the game asked about. The reason for the
    // crossing is that the labels say whose best response is tight -- the
    // column player is best-responding to x when 1 - (B' x)_j is zero -- so
    // the matrix that appears alongside x is B, not A.
    for j in 0..n {
        for i in 0..m {
            tableau[j][i] = b.get(i, j) + shift;
        }
        tableau[j][m + j] = 1.0;
        tableau[j][total] = 1.0;
    }
    for i in 0..m {
        for j in 0..n {
            tableau[n + i][m + j] = a.get(i, j) + shift;
        }
        tableau[n + i][i] = 1.0;
        tableau[n + i][total] = 1.0;
    }
    // Put each tableau into basic form: solve for the basic variables.
    for r in 0..total {
        let column = basis[r];
        let pivot = tableau[r][column];
        if pivot.abs() < 1e-12 {
            return Err(GeomError::Degenerate("lemke_howson: a degenerate starting basis"));
        }
        for c in 0..=total {
            tableau[r][c] /= pivot;
        }
    }

    let mut entering = initial_label;
    // Which tableau to pivot in. Each label is carried by one variable in
    // each tableau -- label i by the row player's probability and by the
    // column player's slack -- so the label alone does not say where to
    // pivot. The path alternates: a pivot in one polytope frees a label
    // whose twin is in the other, so the next pivot is there.
    let mut in_first = initial_label < m;
    for _ in 0..(4 * (m + n) * (m + n) + 100) {
        let (lo, hi) = if in_first { (0, n) } else { (n, total) };
        let mut leaving_row = usize::MAX;
        let mut best = f64::INFINITY;
        let mut ties = 0usize;
        for r in lo..hi {
            let coefficient = tableau[r][entering];
            if coefficient <= 1e-12 {
                continue;
            }
            let ratio = tableau[r][total] / coefficient;
            if ratio < best - 1e-9 {
                best = ratio;
                leaving_row = r;
                ties = 1;
            } else if (ratio - best).abs() <= 1e-9 {
                ties += 1;
            }
        }
        if leaving_row == usize::MAX {
            return Err(GeomError::Degenerate("lemke_howson: the ray is unbounded"));
        }
        if ties > 1 {
            return Err(GeomError::Degenerate("lemke_howson: a degenerate pivot"));
        }

        // Pivot. Only within `lo..hi`: the two tableaux share a column index
        // space -- label i names the row player's probability in one and the
        // column player's slack in the other -- but they are separate systems,
        // and eliminating a column from the rows of the other one corrupts it.
        let pivot = tableau[leaving_row][entering];
        for c in 0..=total {
            tableau[leaving_row][c] /= pivot;
        }
        for r in lo..hi {
            if r != leaving_row {
                let factor = tableau[r][entering];
                if factor != 0.0 {
                    for c in 0..=total {
                        tableau[r][c] -= factor * tableau[leaving_row][c];
                    }
                }
            }
        }
        let left = basis[leaving_row];
        basis[leaving_row] = entering;

        // The label just freed is the next to enter, unless it is the one
        // dropped at the start -- then the path has arrived.
        if left == initial_label {
            break;
        }
        entering = left;
        in_first = !in_first;
    }

    // Read the vertices off the basis and normalise.
    let mut p = vec![0.0; m];
    let mut q = vec![0.0; n];
    for r in 0..total {
        let variable = basis[r];
        let value = tableau[r][total];
        if r < n && variable < m {
            p[variable] = value;
        } else if r >= n && variable >= m {
            q[variable - m] = value;
        }
    }
    let ps: f64 = p.iter().sum();
    let qs: f64 = q.iter().sum();
    if ps <= GAME_TOL || qs <= GAME_TOL {
        return Err(GeomError::Degenerate("lemke_howson: the path ended at the origin"));
    }
    for v in &mut p {
        *v /= ps;
    }
    for v in &mut q {
        *v /= qs;
    }
    Ok((p, q))
}

/// A correlated equilibrium of maximum expected total payoff, as a joint
/// distribution over strategy profiles.
///
/// The reason this is an LP and Nash equilibrium is not: the unknown is the
/// joint distribution itself rather than each player's marginal, so the
/// incentive constraints -- obeying the recommendation beats any deviation,
/// *conditional* on having received it -- are linear. Every Nash equilibrium
/// is a correlated equilibrium (take the product of the marginals), so the
/// set is never empty, and it is generally larger: correlation can achieve
/// payoffs outside the convex hull of the Nash outcomes.
///
/// # Errors
/// Returns an error on a shape mismatch or if the program has no optimum.
pub fn correlated_equilibrium_lp(a: &Matrix, b: &Matrix) -> Result<Matrix, GeomError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(GeomError::InvalidArgument("correlated_equilibrium_lp: shape mismatch"));
    }
    let (m, n) = (a.rows, a.cols);
    let variables = m * n;
    let index = |i: usize, j: usize| i * n + j;

    // Constraints: for each pair of the row player's strategies, obeying i
    // must beat deviating to k; likewise for columns; plus normalisation.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut senses: Vec<Cmp> = Vec::new();

    for i in 0..m {
        for k in 0..m {
            if i == k {
                continue;
            }
            let mut row = vec![0.0; variables];
            for j in 0..n {
                // sum_j x_ij (a_kj - a_ij) <= 0.
                row[index(i, j)] = a.get(k, j) - a.get(i, j);
            }
            rows.push(row);
            rhs.push(0.0);
            senses.push(Cmp::Le);
        }
    }
    for j in 0..n {
        for l in 0..n {
            if j == l {
                continue;
            }
            let mut row = vec![0.0; variables];
            for i in 0..m {
                row[index(i, j)] = b.get(i, l) - b.get(i, j);
            }
            rows.push(row);
            rhs.push(0.0);
            senses.push(Cmp::Le);
        }
    }
    rows.push(vec![1.0; variables]);
    rhs.push(1.0);
    senses.push(Cmp::Eq);

    let mut constraint_matrix = Matrix::zeros(rows.len(), variables);
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            constraint_matrix.set(r, c, v);
        }
    }
    let objective: Vec<f64> = (0..variables)
        .map(|k| a.get(k / n, k % n) + b.get(k / n, k % n))
        .collect();

    let problem = LpProblem {
        c: objective,
        a: constraint_matrix,
        b: rhs,
        constraint_types: senses,
        bounds: vec![(0.0, f64::INFINITY); variables],
        maximize: true,
    };
    let LpResult::Optimal { x, .. } = simplex(&problem)? else {
        return Err(GeomError::Degenerate("the correlated equilibrium program has no optimum"));
    };
    Ok(Matrix::from_fn(m, n, |i, j| x[index(i, j)].max(0.0)))
}

// ---------------------------------------------------------------------------
// Learning and evolutionary dynamics
// ---------------------------------------------------------------------------

/// Fictitious play: each player best-responds to the empirical frequency of
/// the other's past moves.
///
/// Returns the two empirical frequency vectors. It converges to equilibrium
/// in zero-sum games, in 2xN games, and in games solvable by iterated strict
/// dominance -- and famously does *not* converge in general, Shapley's 3x3
/// example cycling forever. So this is a model of learning that sometimes
/// finds equilibrium, not an algorithm for computing one.
///
/// # Errors
/// Returns an error on a shape mismatch.
pub fn fictitious_play(
    a: &Matrix,
    b: &Matrix,
    iterations: usize,
) -> Result<(Vec<f64>, Vec<f64>), GeomError> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(GeomError::InvalidArgument("fictitious_play: shape mismatch"));
    }
    let (m, n) = (a.rows, a.cols);
    let mut row_counts = vec![0.0f64; m];
    let mut column_counts = vec![0.0f64; n];
    // Seed with one observation each so the first best response is defined.
    row_counts[0] = 1.0;
    column_counts[0] = 1.0;

    for _ in 0..iterations {
        let column_total: f64 = column_counts.iter().sum();
        let belief: Vec<f64> = column_counts.iter().map(|v| v / column_total).collect();
        let row_move = best_response(a, &belief)[0];

        let row_total: f64 = row_counts.iter().sum();
        let row_belief: Vec<f64> = row_counts.iter().map(|v| v / row_total).collect();
        // The column player's own payoff has them indexing columns, so its
        // transpose is the matrix they best-respond with.
        let transposed = b.transpose();
        let column_move = best_response(&transposed, &row_belief)[0];

        row_counts[row_move] += 1.0;
        column_counts[column_move] += 1.0;
    }
    let row_total: f64 = row_counts.iter().sum();
    let column_total: f64 = column_counts.iter().sum();
    Ok((
        row_counts.iter().map(|v| v / row_total).collect(),
        column_counts.iter().map(|v| v / column_total).collect(),
    ))
}

/// The replicator dynamic for a symmetric game, returning the trajectory.
///
/// `dx_i/dt = x_i (e_i . A x - x . A x)`: a strategy grows when it does
/// better than the population average. The equation arises from asexual
/// reproduction proportional to payoff, and its fixed points include every
/// symmetric Nash equilibrium -- but not only those, since every vertex of
/// the simplex is a fixed point whether or not it is an equilibrium. The
/// simplex is invariant, which is what makes the dynamic well posed.
///
/// # Errors
/// Returns an error unless the payoff is square, the initial population is a
/// distribution over its strategies, and the step is positive.
pub fn replicator_dynamics(
    payoff: &Matrix,
    x0: &[f64],
    t_end: f64,
    dt: f64,
) -> Result<Vec<Vec<f64>>, GeomError> {
    if !payoff.is_square() || x0.len() != payoff.rows {
        return Err(GeomError::InvalidArgument("replicator_dynamics: shape mismatch"));
    }
    if !(dt > 0.0) || !(t_end > 0.0) {
        return Err(GeomError::InvalidArgument("replicator_dynamics requires positive times"));
    }
    if x0.iter().any(|v| *v < 0.0) || (x0.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
        return Err(GeomError::InvalidArgument("replicator_dynamics needs a distribution"));
    }
    let n = payoff.rows;
    let derivative = |x: &[f64]| -> Vec<f64> {
        let fitness: Vec<f64> = (0..n).map(|i| dot(payoff.row(i), x)).collect();
        let average = dot(x, &fitness);
        (0..n).map(|i| x[i] * (fitness[i] - average)).collect()
    };

    let steps = (t_end / dt).ceil() as usize;
    let mut x = x0.to_vec();
    let mut trajectory = vec![x.clone()];
    for _ in 0..steps {
        // Fourth-order Runge-Kutta: the conserved quantities that make these
        // trajectories interesting -- the interior orbits of rock-paper-
        // scissors, for one -- are destroyed by a first-order method, which
        // spirals out where the true solution cycles.
        let k1 = derivative(&x);
        let x2: Vec<f64> = (0..n).map(|i| x[i] + 0.5 * dt * k1[i]).collect();
        let k2 = derivative(&x2);
        let x3: Vec<f64> = (0..n).map(|i| x[i] + 0.5 * dt * k2[i]).collect();
        let k3 = derivative(&x3);
        let x4: Vec<f64> = (0..n).map(|i| x[i] + dt * k3[i]).collect();
        let k4 = derivative(&x4);
        for i in 0..n {
            x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            x[i] = x[i].max(0.0);
        }
        // Renormalise against the drift that finite steps introduce.
        let total: f64 = x.iter().sum();
        if total > GAME_TOL {
            for v in &mut x {
                *v /= total;
            }
        }
        trajectory.push(x.clone());
    }
    Ok(trajectory)
}

/// Whether a strategy is evolutionarily stable in a symmetric game.
///
/// Maynard Smith's two conditions: the strategy is a symmetric Nash
/// equilibrium, and against any alternative best response it does strictly
/// better than that alternative does against itself. The second condition is
/// what "stable" adds to "equilibrium" -- it says a small invading mutant
/// earns less than the resident and so dies out, which a mere Nash
/// equilibrium does not guarantee.
///
/// Checked against pure alternatives, which suffices: the payoff is linear in
/// the mutant's mixture, so if no pure mutant invades then none does.
///
/// # Errors
/// Returns an error unless the payoff is square and the strategy is a
/// distribution over its rows.
pub fn evolutionarily_stable_check(
    payoff: &Matrix,
    strategy: &[f64],
    tol: f64,
) -> Result<bool, GeomError> {
    if !payoff.is_square() || strategy.len() != payoff.rows {
        return Err(GeomError::InvalidArgument("evolutionarily_stable_check: shape mismatch"));
    }
    if strategy.iter().any(|v| *v < -tol) || (strategy.iter().sum::<f64>() - 1.0).abs() > 1e-7 {
        return Err(GeomError::InvalidArgument("the strategy must be a distribution"));
    }
    let n = payoff.rows;
    let own = bilinear(payoff, strategy, strategy);

    for i in 0..n {
        let mut mutant = vec![0.0; n];
        mutant[i] = 1.0;
        let mutant_against_resident = dot(payoff.row(i), strategy);
        if mutant_against_resident > own + tol {
            return Ok(false);
        }
        if mutant_against_resident > own - tol {
            // An alternative best response: the second condition decides.
            let resident_against_mutant = bilinear(payoff, strategy, &mutant);
            let mutant_against_mutant = payoff.get(i, i);
            if resident_against_mutant <= mutant_against_mutant + tol
                && (strategy[i] - 1.0).abs() > tol
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// The standard 2x2 games
// ---------------------------------------------------------------------------

/// The hawk-dove game: contesting a resource worth `v` at an injury cost `c`.
///
/// Returns the symmetric payoff matrix with hawk first. When `c > v` the
/// game has a mixed ESS playing hawk with probability `v / c`, which is the
/// canonical demonstration that a population can be stable while every
/// individual in it is randomising.
///
/// # Panics
/// Panics unless the cost is positive.
#[must_use]
pub fn hawk_dove(v: f64, c: f64) -> Matrix {
    assert!(c > 0.0, "hawk_dove requires a positive cost");
    Matrix::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => (v - c) / 2.0,
        (0, 1) => v,
        (1, 0) => 0.0,
        _ => v / 2.0,
    })
}

/// The prisoner's dilemma with the conventional temptation, reward,
/// punishment and sucker payoffs. Cooperate is strategy zero.
///
/// # Panics
/// Panics unless `t > r > p > s`, which is what makes it a dilemma at all --
/// defection strictly dominates while mutual cooperation beats mutual
/// defection.
#[must_use]
pub fn prisoners_dilemma(t: f64, r: f64, p: f64, s: f64) -> Matrix {
    assert!(t > r && r > p && p > s, "the prisoner's dilemma requires t > r > p > s");
    Matrix::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => r,
        (0, 1) => s,
        (1, 0) => t,
        _ => p,
    })
}

/// The stag hunt: two pure equilibria, one payoff dominant and one risk
/// dominant. Hunting stag is strategy zero.
#[must_use]
pub fn stag_hunt() -> Matrix {
    Matrix::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 4.0,
        (0, 1) => 0.0,
        (1, 0) => 3.0,
        _ => 3.0,
    })
}

/// Chicken, also called hawk-dove in its ordinal form: two asymmetric pure
/// equilibria and one mixed. Swerving is strategy zero.
#[must_use]
pub fn chicken() -> Matrix {
    Matrix::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 0.0,
        (0, 1) => -1.0,
        (1, 0) => 1.0,
        _ => -10.0,
    })
}

/// Matching pennies: the smallest zero-sum game with no pure equilibrium.
#[must_use]
pub fn matching_pennies() -> Matrix {
    Matrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { -1.0 })
}

/// Rock-paper-scissors as a zero-sum payoff matrix, in that order.
#[must_use]
pub fn rock_paper_scissors() -> Matrix {
    Matrix::from_fn(3, 3, |i, j| {
        if i == j {
            0.0
        } else if (i + 1) % 3 == j {
            -1.0
        } else {
            1.0
        }
    })
}

// ---------------------------------------------------------------------------
// The iterated prisoner's dilemma
// ---------------------------------------------------------------------------

/// A move in the iterated prisoner's dilemma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// Cooperate.
    Cooperate,
    /// Defect.
    Defect,
}

/// A strategy for the iterated prisoner's dilemma.
///
/// The history is the sequence of `(own move, opponent move)` pairs so far.
pub trait IpdStrategy {
    /// The strategy's name, for reporting.
    fn name(&self) -> String;
    /// The next move given the history.
    fn play(&self, history: &[(Move, Move)], rng: &mut Rng) -> Move;
}

/// Always cooperate.
pub struct AlwaysCooperate;
/// Always defect: the unique equilibrium of the one-shot game and of any
/// finitely repeated game with a commonly known end.
pub struct AlwaysDefect;
/// Cooperate first, then copy the opponent's last move.
pub struct TitForTat;
/// Tit for tat that forgives an occasional defection, which is what keeps two
/// copies of it from locking into mutual retaliation under noise.
pub struct GenerousTitForTat {
    /// Probability of cooperating anyway after being defected on.
    pub forgiveness: f64,
}
/// Cooperate until defected on once, then defect forever.
pub struct Grim;
/// Win-stay lose-shift: repeat the last move if it earned a good payoff,
/// switch if it did not.
pub struct Pavlov;
/// Cooperate with fixed probability, ignoring the opponent.
pub struct RandomPlayer {
    /// Probability of cooperating.
    pub cooperate_probability: f64,
}

impl IpdStrategy for AlwaysCooperate {
    fn name(&self) -> String {
        "always-cooperate".into()
    }
    fn play(&self, _: &[(Move, Move)], _: &mut Rng) -> Move {
        Move::Cooperate
    }
}
impl IpdStrategy for AlwaysDefect {
    fn name(&self) -> String {
        "always-defect".into()
    }
    fn play(&self, _: &[(Move, Move)], _: &mut Rng) -> Move {
        Move::Defect
    }
}
impl IpdStrategy for TitForTat {
    fn name(&self) -> String {
        "tit-for-tat".into()
    }
    fn play(&self, history: &[(Move, Move)], _: &mut Rng) -> Move {
        history.last().map_or(Move::Cooperate, |&(_, theirs)| theirs)
    }
}
impl IpdStrategy for GenerousTitForTat {
    fn name(&self) -> String {
        "generous-tit-for-tat".into()
    }
    fn play(&self, history: &[(Move, Move)], rng: &mut Rng) -> Move {
        match history.last() {
            Some(&(_, Move::Defect)) if rng.next_f64() >= self.forgiveness => Move::Defect,
            _ => Move::Cooperate,
        }
    }
}
impl IpdStrategy for Grim {
    fn name(&self) -> String {
        "grim".into()
    }
    fn play(&self, history: &[(Move, Move)], _: &mut Rng) -> Move {
        if history.iter().any(|&(_, theirs)| theirs == Move::Defect) {
            Move::Defect
        } else {
            Move::Cooperate
        }
    }
}
impl IpdStrategy for Pavlov {
    fn name(&self) -> String {
        "pavlov".into()
    }
    fn play(&self, history: &[(Move, Move)], _: &mut Rng) -> Move {
        match history.last() {
            None => Move::Cooperate,
            Some(&(mine, theirs)) => {
                // Stay when the opponent cooperated, switch when they did not.
                if theirs == Move::Cooperate {
                    mine
                } else if mine == Move::Cooperate {
                    Move::Defect
                } else {
                    Move::Cooperate
                }
            }
        }
    }
}
impl IpdStrategy for RandomPlayer {
    fn name(&self) -> String {
        "random".into()
    }
    fn play(&self, _: &[(Move, Move)], rng: &mut Rng) -> Move {
        if rng.next_f64() < self.cooperate_probability {
            Move::Cooperate
        } else {
            Move::Defect
        }
    }
}

/// The built-in strategy set, in a fixed order.
#[must_use]
pub fn standard_ipd_strategies() -> Vec<Box<dyn IpdStrategy>> {
    vec![
        Box::new(AlwaysCooperate),
        Box::new(AlwaysDefect),
        Box::new(TitForTat),
        Box::new(GenerousTitForTat { forgiveness: 0.1 }),
        Box::new(Grim),
        Box::new(Pavlov),
        Box::new(RandomPlayer { cooperate_probability: 0.5 }),
    ]
}

/// A round-robin tournament of iterated prisoner's dilemma, returning each
/// strategy's total score paired with its name, sorted best first.
///
/// `noise` is the probability that a chosen move is flipped in transmission.
/// It matters more than it looks: with no noise, tit for tat against itself
/// cooperates forever, and with even a little, two copies fall into
/// alternating retaliation. Axelrod's tournaments are usually reported
/// without noise, which flatters the unforgiving strategies.
///
/// Every pair plays, including each strategy against a copy of itself.
///
/// # Panics
/// Panics unless the noise is a probability.
#[must_use]
pub fn iterated_pd_tournament(
    strategies: &[Box<dyn IpdStrategy>],
    rounds: usize,
    noise: f64,
    rng: &mut Rng,
) -> Vec<(String, f64)> {
    assert!((0.0..=1.0).contains(&noise), "the noise must be a probability");
    let payoff = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
    let score = |mine: Move, theirs: Move| -> f64 {
        payoff.get(usize::from(mine == Move::Defect), usize::from(theirs == Move::Defect))
    };

    let count = strategies.len();
    let mut totals = vec![0.0f64; count];
    for i in 0..count {
        for j in i..count {
            let mut left: Vec<(Move, Move)> = Vec::with_capacity(rounds);
            let mut right: Vec<(Move, Move)> = Vec::with_capacity(rounds);
            for _ in 0..rounds {
                let mut a_move = strategies[i].play(&left, rng);
                let mut b_move = strategies[j].play(&right, rng);
                if noise > 0.0 && rng.next_f64() < noise {
                    a_move = flip(a_move);
                }
                if noise > 0.0 && rng.next_f64() < noise {
                    b_move = flip(b_move);
                }
                totals[i] += score(a_move, b_move);
                totals[j] += score(b_move, a_move);
                left.push((a_move, b_move));
                right.push((b_move, a_move));
            }
        }
    }
    let mut table: Vec<(String, f64)> =
        (0..count).map(|i| (strategies[i].name(), totals[i])).collect();
    table.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    table
}

fn flip(m: Move) -> Move {
    match m {
        Move::Cooperate => Move::Defect,
        Move::Defect => Move::Cooperate,
    }
}

// ---------------------------------------------------------------------------
// Cooperative games
// ---------------------------------------------------------------------------

/// The Shapley value of a cooperative game given by its characteristic
/// function on coalitions encoded as bitmasks.
///
/// Player `i`'s value is the average over all orderings of the players of
/// what `i` adds to the coalition already formed. The averaging is what makes
/// it fair in a precise sense: it is the *unique* allocation satisfying
/// efficiency, symmetry, the null-player property and additivity, so any
/// objection to the Shapley value has to be an objection to one of those.
///
/// Exact, and so exponential: `2^n` coalitions.
///
/// # Errors
/// Returns an error unless `1 <= n <= 20`, beyond which the enumeration is
/// not worth attempting.
pub fn shapley_value(v: &dyn Fn(u64) -> f64, n: usize) -> Result<Vec<f64>, GeomError> {
    if n == 0 || n > 20 {
        return Err(GeomError::InvalidArgument("shapley_value requires 1 <= n <= 20"));
    }
    // Weight of a coalition of size s in the marginal-contribution sum:
    // s! (n - s - 1)! / n!.
    let mut factorial = vec![1.0f64; n + 1];
    for k in 1..=n {
        factorial[k] = factorial[k - 1] * k as f64;
    }
    let mut values = vec![0.0f64; n];
    for coalition in 0u64..(1u64 << n) {
        let size = coalition.count_ones() as usize;
        let base = v(coalition);
        for i in 0..n {
            if coalition >> i & 1 == 1 {
                continue;
            }
            let weight = factorial[size] * factorial[n - size - 1] / factorial[n];
            values[i] += weight * (v(coalition | 1 << i) - base);
        }
    }
    Ok(values)
}

/// The Shapley value estimated by sampling random orderings.
///
/// The same average as [`shapley_value`], taken over sampled permutations
/// instead of all of them. Unbiased, with error falling as the reciprocal
/// square root of the sample count, which is what makes it the only option
/// once the player count passes about twenty.
///
/// # Errors
/// Returns an error if there are no players or no samples.
pub fn shapley_monte_carlo(
    v: &dyn Fn(u64) -> f64,
    n: usize,
    samples: usize,
    rng: &mut Rng,
) -> Result<Vec<f64>, GeomError> {
    if n == 0 || n > 64 || samples == 0 {
        return Err(GeomError::InvalidArgument("shapley_monte_carlo: bad parameters"));
    }
    let mut totals = vec![0.0f64; n];
    let mut order: Vec<usize> = (0..n).collect();
    for _ in 0..samples {
        for i in (1..n).rev() {
            let j = ((u128::from(rng.next_u64()) * (i as u128 + 1)) >> 64) as usize;
            order.swap(i, j);
        }
        let mut coalition = 0u64;
        let mut running = v(0);
        for &player in &order {
            coalition |= 1 << player;
            let next = v(coalition);
            totals[player] += next - running;
            running = next;
        }
    }
    Ok(totals.iter().map(|t| t / samples as f64).collect())
}

/// The normalised Banzhaf index: each player's share of the swings they can
/// make.
///
/// Differs from the Shapley value in what it averages over -- coalitions
/// rather than orderings -- and so weights the sizes differently. The two
/// disagree, and the disagreement is the point: there is no single correct
/// measure of power, only different axiomatisations of it.
///
/// A game in which no player ever swings anything has no power to apportion,
/// and the shares come back as zeros rather than as a division by zero.
///
/// # Errors
/// Returns an error unless `1 <= n <= 20`.
pub fn banzhaf_index(v: &dyn Fn(u64) -> f64, n: usize) -> Result<Vec<f64>, GeomError> {
    if n == 0 || n > 20 {
        return Err(GeomError::InvalidArgument("banzhaf_index requires 1 <= n <= 20"));
    }
    let mut raw = vec![0.0f64; n];
    for coalition in 0u64..(1u64 << n) {
        for i in 0..n {
            if coalition >> i & 1 == 1 {
                continue;
            }
            raw[i] += v(coalition | 1 << i) - v(coalition);
        }
    }
    let total: f64 = raw.iter().sum();
    if total.abs() < GAME_TOL {
        return Ok(vec![0.0; n]);
    }
    Ok(raw.iter().map(|x| x / total).collect())
}

/// Whether an allocation lies in the core: efficient, and unimprovable by any
/// coalition.
///
/// The core can be empty -- three players splitting a pound where any two can
/// take it all has no core allocation at all -- which is exactly why the
/// Shapley value, which always exists, is worth having as well.
///
/// # Errors
/// Returns an error on a bad player count or allocation length.
pub fn core_check_small(
    v: &dyn Fn(u64) -> f64,
    n: usize,
    allocation: &[f64],
) -> Result<bool, GeomError> {
    if n == 0 || n > 20 || allocation.len() != n {
        return Err(GeomError::InvalidArgument("core_check_small: bad parameters"));
    }
    let grand = (1u64 << n) - 1;
    if (allocation.iter().sum::<f64>() - v(grand)).abs() > 1e-7 {
        return Ok(false);
    }
    for coalition in 1u64..(1u64 << n) {
        let share: f64 =
            (0..n).filter(|&i| coalition >> i & 1 == 1).map(|i| allocation[i]).sum();
        if share < v(coalition) - 1e-7 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The nucleolus of a small cooperative game.
///
/// Lexicographically minimises the vector of coalition excesses -- how much
/// each coalition is short of what it could get on its own -- worst first.
/// Solved as a sequence of linear programs: maximise the smallest slack, fix
/// whichever coalitions are then tight, repeat on the rest. Unlike the core
/// it is never empty, and unlike the Shapley value it always lies in the core
/// when the core is non-empty, which is the property that motivates it.
///
/// # Errors
/// Returns an error for more than about a dozen players, or if a program
/// fails.
pub fn nucleolus_small(v: &dyn Fn(u64) -> f64, n: usize) -> Result<Vec<f64>, GeomError> {
    if n == 0 || n > 12 {
        return Err(GeomError::InvalidArgument("nucleolus_small requires 1 <= n <= 12"));
    }
    let grand = (1u64 << n) - 1;
    let total = v(grand);
    // Every variable is free. The payoffs can be negative in a game whose
    // coalitions are worth less than nothing, and the excess *must* be
    // allowed to be: when the core is empty the largest achievable slack is
    // negative, and pinning the excess at zero makes the first program
    // infeasible rather than telling the truth about the game.
    let free = vec![(f64::NEG_INFINITY, f64::INFINITY); n + 1];

    let mut fixed: Vec<(u64, f64)> = Vec::new();
    let mut settled: Vec<u64> = Vec::new();
    let mut allocation = vec![0.0; n];

    for _ in 0..(1usize << n) {
        let open: Vec<u64> = (1u64..grand).filter(|c| !settled.contains(c)).collect();
        if open.is_empty() {
            break;
        }
        // max e  s.t.  sum_{i in S} x_i - e >= v(S) for the open S,
        //              equality at the settled level for the rest,
        //              sum_i x_i = v(N).
        let variables = n + 1;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut rhs: Vec<f64> = Vec::new();
        let mut senses: Vec<Cmp> = Vec::new();

        for &coalition in &open {
            let mut row = vec![0.0; variables];
            for i in 0..n {
                if coalition >> i & 1 == 1 {
                    row[i] = 1.0;
                }
            }
            row[n] = -1.0;
            rows.push(row);
            rhs.push(v(coalition));
            senses.push(Cmp::Ge);
        }
        for &(coalition, level) in &fixed {
            let mut row = vec![0.0; variables];
            for i in 0..n {
                if coalition >> i & 1 == 1 {
                    row[i] = 1.0;
                }
            }
            rows.push(row);
            rhs.push(v(coalition) + level);
            senses.push(Cmp::Eq);
        }
        let mut row = vec![0.0; variables];
        for entry in row.iter_mut().take(n) {
            *entry = 1.0;
        }
        rows.push(row);
        rhs.push(total);
        senses.push(Cmp::Eq);

        let mut a = Matrix::zeros(rows.len(), variables);
        for (r, source) in rows.iter().enumerate() {
            for (c, &value) in source.iter().enumerate() {
                a.set(r, c, value);
            }
        }
        let mut c = vec![0.0; variables];
        c[n] = 1.0;
        let problem = LpProblem {
            c,
            a,
            b: rhs,
            constraint_types: senses,
            bounds: free.clone(),
            maximize: true,
        };
        let LpResult::Optimal { x, objective, .. } = simplex(&problem)? else {
            return Err(GeomError::Degenerate("the nucleolus program has no optimum"));
        };
        allocation = x[..n].to_vec();

        // Whichever open coalitions are tight at this slack are now settled.
        let mut newly = 0usize;
        for &coalition in &open {
            let share: f64 =
                (0..n).filter(|&i| coalition >> i & 1 == 1).map(|i| allocation[i]).sum();
            if (share - v(coalition) - objective).abs() < 1e-7 {
                fixed.push((coalition, objective));
                settled.push(coalition);
                newly += 1;
            }
        }
        if newly == 0 {
            break;
        }
    }
    Ok(allocation)
}

/// The Banzhaf power of each voter in a weighted voting game.
///
/// The point of the exercise is that power is not proportional to weight. A
/// voter with a large weight can have the same power as a small one -- and a
/// voter with positive weight can be a dummy with no power at all, if no
/// coalition ever needs them.
///
/// # Errors
/// Returns an error for an empty or oversized electorate.
pub fn voting_power_weighted(weights: &[f64], quota: f64) -> Result<Vec<f64>, GeomError> {
    let n = weights.len();
    if n == 0 || n > 20 {
        return Err(GeomError::InvalidArgument("voting_power_weighted requires 1 <= n <= 20"));
    }
    let characteristic = |coalition: u64| -> f64 {
        let total: f64 =
            (0..n).filter(|&i| coalition >> i & 1 == 1).map(|i| weights[i]).sum();
        f64::from(total >= quota)
    };
    banzhaf_index(&characteristic, n)
}

// ---------------------------------------------------------------------------
// Auctions
// ---------------------------------------------------------------------------

/// The symmetric equilibrium bid shading factor in a first-price sealed-bid
/// auction with `n` bidders whose values are uniform on `[0, 1]`.
///
/// The equilibrium bid is `(n - 1) / n` times one's value. Shading is not a
/// mistake: bidding one's value in a first-price auction guarantees zero
/// surplus whether one wins or not. As the field grows the shading vanishes,
/// which is the mechanism behind revenue equivalence.
///
/// # Panics
/// Panics unless there are at least two bidders.
#[must_use]
pub fn first_price_auction_equilibrium_uniform(n: usize) -> f64 {
    assert!(n >= 2, "an auction needs at least two bidders");
    (n as f64 - 1.0) / n as f64
}

/// Confirms by exhaustive case analysis that truthful bidding weakly
/// dominates in a second-price auction.
///
/// Returns true when no misreport ever beats the truth, over a grid of
/// values, bids and highest-rival bids. The argument is a two-case one --
/// bidding above one's value can only win auctions one regrets, bidding below
/// can only lose auctions one wanted -- and neither case depends on beliefs
/// about the rivals, which is what makes the dominance so strong.
#[must_use]
pub fn second_price_dominant_check() -> bool {
    let steps = 40;
    let grid: Vec<f64> = (0..=steps).map(|i| i as f64 / steps as f64).collect();
    for &value in &grid {
        for &bid in &grid {
            for &rival in &grid {
                let utility = |b: f64| -> f64 {
                    if b > rival {
                        value - rival
                    } else if b < rival {
                        0.0
                    } else {
                        // A tie is broken by a fair coin.
                        0.5 * (value - rival)
                    }
                };
                if utility(bid) > utility(value) + 1e-12 {
                    return false;
                }
            }
        }
    }
    true
}

/// Simulates first- and second-price auctions with uniform values, returning
/// the two average revenues.
///
/// The revenue equivalence theorem says they coincide: any two mechanisms
/// that allocate to the highest value and give a zero-value bidder zero
/// surplus raise the same expected revenue. The first-price auction collects
/// a shaded bid from the winner, the second-price auction collects the
/// runner-up's full value, and in expectation those are the same number.
///
/// # Errors
/// Returns an error for fewer than two bidders or no trials.
pub fn revenue_equivalence_sim(
    n: usize,
    trials: usize,
    rng: &mut Rng,
) -> Result<(f64, f64), GeomError> {
    if n < 2 || trials == 0 {
        return Err(GeomError::InvalidArgument("revenue_equivalence_sim: bad parameters"));
    }
    let shade = first_price_auction_equilibrium_uniform(n);
    let mut first = 0.0;
    let mut second = 0.0;
    for _ in 0..trials {
        let values: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        first += shade * sorted[0];
        second += sorted[1];
    }
    Ok((first / trials as f64, second / trials as f64))
}

/// A VCG auction for distinct items, one per winner.
///
/// `bids[i][k]` is bidder `i`'s value for item `k`. Returns the item assigned
/// to each bidder, if any, and each bidder's payment. The payment is the
/// externality imposed: the welfare others would have had in one's absence,
/// less the welfare they actually get. That is what makes truthful bidding
/// dominant -- one's own report shifts only the allocation, never the price
/// one pays for it.
///
/// The welfare-maximising assignment is found by exhaustive search, so this is
/// for small instances.
///
/// # Errors
/// Returns an error for ragged bids or more than eight bidders or items.
pub fn vcg_auction(bids: &[Vec<f64>], items: usize) -> Result<(Vec<Option<usize>>, Vec<f64>), GeomError> {
    let n = bids.len();
    if n == 0 || n > 8 || items == 0 || items > 8 {
        return Err(GeomError::InvalidArgument("vcg_auction: bad size"));
    }
    if bids.iter().any(|row| row.len() != items) {
        return Err(GeomError::InvalidArgument("vcg_auction: ragged bids"));
    }

    // The best assignment over a set of bidders, and its welfare.
    let best_assignment = |allowed: &[bool]| -> (f64, Vec<Option<usize>>) {
        let mut best_welfare = 0.0;
        let mut best = vec![None; n];
        let mut current = vec![None; n];
        let mut used = vec![false; items];
        // Depth-first over bidders, each taking an unused item or nothing.
        fn search(
            bidder: usize,
            n: usize,
            items: usize,
            allowed: &[bool],
            bids: &[Vec<f64>],
            used: &mut Vec<bool>,
            current: &mut Vec<Option<usize>>,
            welfare: f64,
            best_welfare: &mut f64,
            best: &mut Vec<Option<usize>>,
        ) {
            if bidder == n {
                if welfare > *best_welfare + 1e-12 {
                    *best_welfare = welfare;
                    *best = current.clone();
                }
                return;
            }
            search(bidder + 1, n, items, allowed, bids, used, current, welfare, best_welfare, best);
            if !allowed[bidder] {
                return;
            }
            for k in 0..items {
                if used[k] {
                    continue;
                }
                used[k] = true;
                current[bidder] = Some(k);
                search(
                    bidder + 1,
                    n,
                    items,
                    allowed,
                    bids,
                    used,
                    current,
                    welfare + bids[bidder][k],
                    best_welfare,
                    best,
                );
                current[bidder] = None;
                used[k] = false;
            }
        }
        search(
            0,
            n,
            items,
            allowed,
            bids,
            &mut used,
            &mut current,
            0.0,
            &mut best_welfare,
            &mut best,
        );
        (best_welfare, best)
    };

    let everyone = vec![true; n];
    let (welfare, assignment) = best_assignment(&everyone);

    let mut payments = vec![0.0; n];
    for i in 0..n {
        let mut without = everyone.clone();
        without[i] = false;
        let (welfare_without, _) = best_assignment(&without);
        // What the others get with i present.
        let others_with: f64 = (0..n)
            .filter(|&k| k != i)
            .filter_map(|k| assignment[k].map(|item| bids[k][item]))
            .sum();
        payments[i] = (welfare_without - others_with).max(0.0);
    }
    let _ = welfare;
    Ok((assignment, payments))
}

// ---------------------------------------------------------------------------
// Fair division, matching, and market models
// ---------------------------------------------------------------------------

/// Divide and choose over a cake whose value density differs between the two
/// players, returning each one's share of their own total value.
///
/// `density_a` and `density_b` give the two valuations over `[0, 1]`, sampled
/// on `resolution` intervals. The cutter divides at their own halfway point
/// and the chooser takes the piece they prefer, so the cutter gets exactly a
/// half by their own measure and the chooser at least a half by theirs. That
/// is envy-freeness for two players -- and it does not extend: no analogous
/// finite protocol was known for three until 1960, and for four until 2016.
///
/// # Errors
/// Returns an error if the resolution is zero or a valuation is not positive.
pub fn cake_cutting_divide_choose(
    density_a: &dyn Fn(f64) -> f64,
    density_b: &dyn Fn(f64) -> f64,
    resolution: usize,
) -> Result<(f64, f64), GeomError> {
    if resolution == 0 {
        return Err(GeomError::InvalidArgument("cake_cutting needs a positive resolution"));
    }
    let h = 1.0 / resolution as f64;
    let sample = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
        (0..resolution).map(|k| f((k as f64 + 0.5) * h) * h).collect()
    };
    let a = sample(density_a);
    let b = sample(density_b);
    let total_a: f64 = a.iter().sum();
    let total_b: f64 = b.iter().sum();
    if total_a <= 0.0 || total_b <= 0.0 {
        return Err(GeomError::InvalidArgument("cake_cutting needs positive valuations"));
    }

    // The cutter's halfway point by their own measure.
    let mut running = 0.0;
    let mut cut = resolution;
    for (k, piece) in a.iter().enumerate() {
        running += piece;
        if running >= total_a / 2.0 {
            cut = k + 1;
            break;
        }
    }
    let b_left: f64 = b[..cut].iter().sum();
    let b_right: f64 = b[cut..].iter().sum();
    // The chooser takes their preferred piece; the cutter gets the rest.
    if b_left >= b_right {
        let a_right: f64 = a[cut..].iter().sum();
        Ok((a_right / total_a, b_left / total_b))
    } else {
        let a_left: f64 = a[..cut].iter().sum();
        Ok((a_left / total_a, b_right / total_b))
    }
}

/// Confirms that the deferred-acceptance matching is stable and optimal for
/// the proposing side.
///
/// Gale-Shapley's guarantee is sharper than stability: among *all* stable
/// matchings, every proposer gets their best possible partner and every
/// receiver their worst. So the same algorithm run from the other side gives
/// a different matching, and which side proposes is a distributional
/// decision, not an implementation detail. Both halves are checked here by
/// enumerating the stable matchings directly.
///
/// # Errors
/// Returns an error unless the preference lists are square, complete, and no
/// larger than seven a side.
pub fn gale_shapley_optimality_check(
    prefs_a: &[Vec<usize>],
    prefs_b: &[Vec<usize>],
) -> Result<bool, GeomError> {
    let n = prefs_a.len();
    if n == 0 || n > 7 || prefs_b.len() != n {
        return Err(GeomError::InvalidArgument("gale_shapley_optimality_check: bad size"));
    }
    if prefs_a.iter().chain(prefs_b).any(|p| p.len() != n) {
        return Err(GeomError::InvalidArgument("the preference lists must be complete"));
    }
    let rank = |prefs: &[Vec<usize>], who: usize, partner: usize| -> usize {
        prefs[who].iter().position(|&p| p == partner).unwrap_or(usize::MAX)
    };
    if prefs_a.iter().chain(prefs_b).any(|p| {
        let mut seen = vec![false; n];
        p.iter().any(|&x| x >= n || std::mem::replace(&mut seen[x], true))
    }) {
        return Err(GeomError::InvalidArgument("each list must be a permutation"));
    }

    let matching = crate::graph::matching::stable_marriage(prefs_a, prefs_b);

    // Enumerate every stable matching by brute force over permutations.
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut stable: Vec<Vec<usize>> = Vec::new();
    permute(&mut permutation, 0, &mut |candidate: &[usize]| {
        let blocking = (0..n).any(|a| {
            (0..n).any(|b| {
                candidate[a] != b
                    && rank(prefs_a, a, b) < rank(prefs_a, a, candidate[a])
                    && rank(prefs_b, b, a)
                        < rank(prefs_b, b, candidate.iter().position(|&x| x == b).unwrap())
            })
        });
        if !blocking {
            stable.push(candidate.to_vec());
        }
    });
    if stable.is_empty() {
        return Ok(false);
    }
    if !stable.contains(&matching) {
        return Ok(false);
    }
    // Proposer optimality: no stable matching gives any proposer better.
    for a in 0..n {
        let mine = rank(prefs_a, a, matching[a]);
        if stable.iter().any(|s| rank(prefs_a, a, s[a]) < mine) {
            return Ok(false);
        }
    }
    // Receiver pessimality: no stable matching gives any receiver worse.
    for b in 0..n {
        let partner = matching.iter().position(|&x| x == b).unwrap();
        let theirs = rank(prefs_b, b, partner);
        if stable.iter().any(|s| {
            let other = s.iter().position(|&x| x == b).unwrap();
            rank(prefs_b, b, other) > theirs
        }) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn permute(items: &mut Vec<usize>, k: usize, visit: &mut dyn FnMut(&[usize])) {
    if k == items.len() {
        visit(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, visit);
        items.swap(k, i);
    }
}

/// The Stackelberg equilibrium of a 2x2 game where the row player commits
/// first to a *pure* strategy, as
/// `(leader move, follower move, leader payoff, follower payoff)`.
///
/// Committing to a pure strategy is at least as good as any pure equilibrium
/// -- the leader can commit to what they would have played anyway, and the
/// follower's reply is unchanged -- and it is often strictly better, which is
/// what first-mover advantage means.
///
/// It is not, however, at least as good as every *mixed* equilibrium. The
/// general theorem is about commitment to mixed strategies; restricted to
/// pure ones a leader can end up below their mixed Nash payoff, since the
/// mixture they would have randomised over is no longer available to them.
///
/// # Errors
/// Returns an error unless both matrices are 2x2.
pub fn stackelberg_2x2(a: &Matrix, b: &Matrix) -> Result<(usize, usize, f64, f64), GeomError> {
    if a.rows != 2 || a.cols != 2 || b.rows != 2 || b.cols != 2 {
        return Err(GeomError::InvalidArgument("stackelberg_2x2 requires two 2x2 matrices"));
    }
    let mut best = (0usize, 0usize, f64::NEG_INFINITY, 0.0);
    for leader in 0..2 {
        // The follower picks their own best column against the commitment,
        // breaking ties in the leader's favour, which is the standard
        // convention and the only one under which the optimum is attained.
        let follower_best = (0..2)
            .map(|j| b.get(leader, j))
            .fold(f64::NEG_INFINITY, f64::max);
        let follower = (0..2)
            .filter(|&j| b.get(leader, j) >= follower_best - GAME_TOL)
            .max_by(|&x, &y| {
                a.get(leader, x)
                    .partial_cmp(&a.get(leader, y))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let payoff = a.get(leader, follower);
        if payoff > best.2 {
            best = (leader, follower, payoff, b.get(leader, follower));
        }
    }
    Ok(best)
}

/// The Cournot equilibrium quantities for `n` firms with constant marginal
/// costs facing a linear inverse demand `p = intercept - slope * Q`.
///
/// Each firm's best response is linear in the others' total, and the system
/// solves in closed form. The comparison with Bertrand is the standard
/// lesson: competing in quantities leaves price above marginal cost however
/// many firms there are, while competing in prices drives it to marginal cost
/// with only two.
///
/// # Errors
/// Returns an error for no firms, a non-positive slope, or a cost above the
/// choke price.
pub fn cournot_equilibrium(
    demand_intercept: f64,
    demand_slope: f64,
    costs: &[f64],
) -> Result<Vec<f64>, GeomError> {
    let n = costs.len();
    if n == 0 {
        return Err(GeomError::InvalidArgument("cournot_equilibrium requires firms"));
    }
    if !(demand_slope > 0.0) {
        return Err(GeomError::InvalidArgument("cournot_equilibrium requires a positive slope"));
    }
    let cost_total: f64 = costs.iter().sum();
    // Setting each firm's own first-order condition to zero and solving the
    // resulting linear system gives
    // q_i = (a - n c_i + sum_{j != i} c_j) / ((n + 1) b).
    // The sum excludes the firm's own cost, which is easy to lose: including
    // it leaves the symmetric case reading (a - c)/((n+1)b) + c/((n+1)b),
    // which is not a best response to itself and so is not an equilibrium.
    let quantities: Vec<f64> = costs
        .iter()
        .map(|&c| {
            (demand_intercept - n as f64 * c + (cost_total - c))
                / ((n as f64 + 1.0) * demand_slope)
        })
        .map(|q| q.max(0.0))
        .collect();
    Ok(quantities)
}

/// The Bertrand equilibrium price with identical firms: marginal cost.
///
/// Two firms suffice. Any price above cost is undercut by a rival who then
/// takes the whole market, so the only equilibrium is the competitive one --
/// the "Bertrand paradox", since it predicts that a duopoly behaves like
/// perfect competition. With asymmetric costs the low-cost firm prices just
/// under the rival's cost, which is what this returns.
///
/// # Errors
/// Returns an error for fewer than two firms.
pub fn bertrand_equilibrium(costs: &[f64]) -> Result<f64, GeomError> {
    if costs.len() < 2 {
        return Err(GeomError::InvalidArgument("bertrand_equilibrium requires two firms"));
    }
    let mut sorted = costs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(sorted[1])
}

/// A public goods game with a linear return, returning the average
/// contribution per round.
///
/// Each of `n` players contributes some fraction of an endowment to a pot
/// that is multiplied by `multiplier` and split evenly. A unit contributed
/// costs its contributor one and returns `multiplier / n` to them, so the
/// threshold is at `multiplier = n`: below it contributing is individually
/// irrational and collectively optimal, which is the free-rider problem in
/// its simplest form, and above it the two coincide.
///
/// Players here are conditional cooperators, matching what the others gave
/// and adjusting by their own marginal return -- the rule the laboratory
/// evidence supports. Note that imitating the highest *earner* instead would
/// drive contributions to zero at any multiplier whatever, because within a
/// round every player receives the same share and so the smallest contributor
/// always earns most. That comparison is between players, and the incentive
/// that matters is the effect of a player's own contribution on their own
/// earnings; conflating the two is an easy way to build a model that cannot
/// represent the threshold at all.
///
/// # Errors
/// Returns an error for bad parameters.
pub fn public_goods_game_sim(
    n: usize,
    multiplier: f64,
    rounds: usize,
    rng: &mut Rng,
) -> Result<Vec<f64>, GeomError> {
    if n < 2 || rounds == 0 || !(multiplier > 0.0) {
        return Err(GeomError::InvalidArgument("public_goods_game_sim: bad parameters"));
    }
    // The marginal return to a unit of one's own contribution.
    let marginal = multiplier / n as f64 - 1.0;
    let adjustment = 0.5;
    let mut contributions: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let mut history = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        history.push(contributions.iter().sum::<f64>() / n as f64);
        let total: f64 = contributions.iter().sum();
        let next: Vec<f64> = (0..n)
            .map(|i| {
                let others = (total - contributions[i]) / (n as f64 - 1.0);
                (others + adjustment * marginal + (rng.next_f64() - 0.5) * 0.1)
                    .clamp(0.0, 1.0)
            })
            .collect();
        contributions = next;
    }
    Ok(history)
}

/// A Colonel Blotto tournament between random allocations, returning the
/// win-rate matrix between the sampled strategies.
///
/// Troops are split across fields and each field goes to whoever committed
/// more. The game has no pure equilibrium and no dominant allocation: every
/// deterministic plan is beaten by some other, so the equilibrium is
/// necessarily in mixed strategies. The matrix is the empirical payoff of the
/// sampled strategies against one another.
///
/// # Errors
/// Returns an error for fewer than two fields, no troops, or fewer than two
/// sampled strategies.
pub fn colonel_blotto_sim(
    fields: usize,
    troops: usize,
    strategies: usize,
    rng: &mut Rng,
) -> Result<Matrix, GeomError> {
    if fields < 2 || troops == 0 || strategies < 2 {
        return Err(GeomError::InvalidArgument("colonel_blotto_sim: bad parameters"));
    }
    // Random compositions of the troop count into the fields.
    let plans: Vec<Vec<usize>> = (0..strategies)
        .map(|_| {
            let mut weights: Vec<f64> = (0..fields).map(|_| rng.next_f64() + 1e-9).collect();
            let total: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= total;
            }
            let mut plan: Vec<usize> =
                weights.iter().map(|w| (w * troops as f64).floor() as usize).collect();
            let mut assigned: usize = plan.iter().sum();
            let mut k = 0usize;
            while assigned < troops {
                plan[k % fields] += 1;
                assigned += 1;
                k += 1;
            }
            plan
        })
        .collect();

    Ok(Matrix::from_fn(strategies, strategies, |i, j| {
        let mut score = 0.0;
        for f in 0..fields {
            if plans[i][f] > plans[j][f] {
                score += 1.0;
            } else if plans[i][f] < plans[j][f] {
                score -= 1.0;
            }
        }
        score
    }))
}

// ---------------------------------------------------------------------------
// Extensive-form games and search
// ---------------------------------------------------------------------------

/// A node of an extensive-form game tree.
///
/// A leaf carries a payoff for each player; an internal node names the player
/// to move and its children.
#[derive(Debug, Clone)]
pub enum GameTree {
    /// A terminal node with one payoff per player.
    Leaf(Vec<f64>),
    /// A decision node: which player moves, and what they may move to.
    Node {
        /// The player to move.
        player: usize,
        /// The available continuations.
        children: Vec<GameTree>,
    },
}

/// Backward induction on a game tree, returning the equilibrium path of moves
/// and the payoffs it reaches.
///
/// Solving from the leaves upward gives a subgame perfect equilibrium, which
/// rules out the equilibria of the normal form that rest on threats the
/// threatener would not want to carry out. That is the whole content of the
/// refinement: a Nash equilibrium can be sustained by a promise to behave
/// irrationally off the path, and backward induction cannot represent one.
///
/// # Errors
/// Returns an error if a decision node has no children or the payoff vectors
/// disagree in length.
pub fn backward_induction(tree: &GameTree) -> Result<(Vec<usize>, Vec<f64>), GeomError> {
    match tree {
        GameTree::Leaf(payoffs) => {
            if payoffs.is_empty() {
                return Err(GeomError::InvalidArgument("a leaf needs payoffs"));
            }
            Ok((Vec::new(), payoffs.clone()))
        }
        GameTree::Node { player, children } => {
            if children.is_empty() {
                return Err(GeomError::InvalidArgument("a decision node needs children"));
            }
            let mut best: Option<(usize, Vec<usize>, Vec<f64>)> = None;
            for (index, child) in children.iter().enumerate() {
                let (path, payoffs) = backward_induction(child)?;
                if *player >= payoffs.len() {
                    return Err(GeomError::InvalidArgument("the player index exceeds the payoffs"));
                }
                let improved = match &best {
                    None => true,
                    Some((_, _, current)) => payoffs[*player] > current[*player] + GAME_TOL,
                };
                if improved {
                    best = Some((index, path, payoffs));
                }
            }
            let (index, mut path, payoffs) = best.expect("children is non-empty");
            path.insert(0, index);
            Ok((path, payoffs))
        }
    }
}

/// A two-player zero-sum game state for alpha-beta search.
pub trait GameState: Clone {
    /// The legal moves, empty at a terminal position.
    fn moves(&self) -> Vec<usize>;
    /// The state after playing a move.
    fn apply(&self, mv: usize) -> Self;
    /// The value of a terminal position, from the perspective of the player
    /// who moves first at the root. Only consulted when `moves` is empty or
    /// the depth runs out.
    fn evaluate(&self) -> i64;
    /// Whether the side to move is the maximiser.
    fn maximising(&self) -> bool;
    /// Whether the position is terminal.
    fn terminal(&self) -> bool;
}

/// Alpha-beta search, returning the value and the best move.
///
/// The pruning is exact: alpha-beta returns the same value as a full minimax
/// search, and only visits fewer nodes. What it prunes are branches that
/// cannot affect the root value because the opponent already has a better
/// alternative elsewhere -- so the saving costs nothing in accuracy, which is
/// unusual among search heuristics. With perfect move ordering it examines
/// the square root of the nodes minimax would.
///
/// Returns `None` for the move at a terminal position.
#[must_use]
pub fn alpha_beta_search<S: GameState>(state: &S, depth: usize) -> (i64, Option<usize>) {
    fn recurse<S: GameState>(
        state: &S,
        depth: usize,
        mut alpha: i64,
        mut beta: i64,
        nodes: &mut usize,
    ) -> (i64, Option<usize>) {
        *nodes += 1;
        if depth == 0 || state.terminal() {
            return (state.evaluate(), None);
        }
        let moves = state.moves();
        if moves.is_empty() {
            return (state.evaluate(), None);
        }
        let mut best_move = None;
        if state.maximising() {
            let mut best = i64::MIN;
            for mv in moves {
                let (value, _) = recurse(&state.apply(mv), depth - 1, alpha, beta, nodes);
                if value > best {
                    best = value;
                    best_move = Some(mv);
                }
                alpha = alpha.max(best);
                if beta <= alpha {
                    break;
                }
            }
            (best, best_move)
        } else {
            let mut best = i64::MAX;
            for mv in moves {
                let (value, _) = recurse(&state.apply(mv), depth - 1, alpha, beta, nodes);
                if value < best {
                    best = value;
                    best_move = Some(mv);
                }
                beta = beta.min(best);
                if beta <= alpha {
                    break;
                }
            }
            (best, best_move)
        }
    }
    let mut nodes = 0usize;
    recurse(state, depth, i64::MIN, i64::MAX, &mut nodes)
}

/// Plain minimax without pruning, and the node count.
///
/// Kept so the pruning can be checked rather than assumed: alpha-beta must
/// agree with this on the value at every position while visiting no more
/// nodes.
#[must_use]
pub fn minimax_search<S: GameState>(state: &S, depth: usize) -> (i64, usize) {
    fn recurse<S: GameState>(state: &S, depth: usize, nodes: &mut usize) -> i64 {
        *nodes += 1;
        if depth == 0 || state.terminal() {
            return state.evaluate();
        }
        let moves = state.moves();
        if moves.is_empty() {
            return state.evaluate();
        }
        let values = moves.iter().map(|&mv| recurse(&state.apply(mv), depth - 1, nodes));
        if state.maximising() {
            values.fold(i64::MIN, i64::max)
        } else {
            values.fold(i64::MAX, i64::min)
        }
    }
    let mut nodes = 0usize;
    let value = recurse(state, depth, &mut nodes);
    (value, nodes)
}

/// Monte Carlo tree search with the UCT selection rule, returning the most
/// visited move at the root.
///
/// Each iteration walks down by the upper confidence bound
/// `mean + c sqrt(ln(parent visits) / visits)`, expands one new child, plays
/// out at random, and propagates the result back. The bound is what makes the
/// search anytime and asymptotically optimal: unexplored moves have an
/// infinite bonus, so nothing is dismissed on a single bad rollout, while the
/// bonus shrinks with evidence.
///
/// The reported move is the most *visited*, not the highest scoring: the
/// visit count is the more stable statistic, since a high mean over two
/// rollouts says very little.
///
/// # Errors
/// Returns an error at a terminal root or with a non-positive iteration
/// count.
pub fn mcts_lite<S: GameState>(
    state: &S,
    iterations: usize,
    exploration: f64,
    rng: &mut Rng,
) -> Result<usize, GeomError> {
    let root_moves = state.moves();
    if root_moves.is_empty() || iterations == 0 {
        return Err(GeomError::InvalidArgument("mcts_lite needs moves and iterations"));
    }

    struct Node {
        visits: f64,
        total: f64,
        children: Vec<usize>,
        untried: Vec<usize>,
        move_taken: usize,
    }
    let mut nodes: Vec<Node> = vec![Node {
        visits: 0.0,
        total: 0.0,
        children: Vec::new(),
        untried: root_moves.clone(),
        move_taken: usize::MAX,
    }];

    for _ in 0..iterations {
        // Selection.
        let mut current = 0usize;
        let mut position = state.clone();
        let mut path = vec![0usize];
        while nodes[current].untried.is_empty() && !nodes[current].children.is_empty() {
            let parent_visits = nodes[current].visits.max(1.0);
            // Values are stored from the root maximiser's point of view, so
            // the player choosing here reads them with their own sign. Without
            // that flip the search is optimistic rather than adversarial: it
            // assumes the opponent will pick whatever helps the root, and it
            // walks straight past forced replies.
            let sign = if position.maximising() { 1.0 } else { -1.0 };
            let best = *nodes[current]
                .children
                .iter()
                .max_by(|&&x, &&y| {
                    let score = |k: usize| -> f64 {
                        let node = &nodes[k];
                        if node.visits == 0.0 {
                            return f64::INFINITY;
                        }
                        sign * node.total / node.visits
                            + exploration * (parent_visits.ln() / node.visits).sqrt()
                    };
                    score(x).partial_cmp(&score(y)).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("children is non-empty");
            position = position.apply(nodes[best].move_taken);
            current = best;
            path.push(current);
        }

        // Expansion.
        if !nodes[current].untried.is_empty() && !position.terminal() {
            let index =
                ((u128::from(rng.next_u64()) * nodes[current].untried.len() as u128) >> 64) as usize;
            let mv = nodes[current].untried.swap_remove(index);
            position = position.apply(mv);
            nodes.push(Node {
                visits: 0.0,
                total: 0.0,
                children: Vec::new(),
                untried: position.moves(),
                move_taken: mv,
            });
            let child = nodes.len() - 1;
            nodes[current].children.push(child);
            path.push(child);
        }

        // Rollout.
        let mut depth = 0usize;
        while !position.terminal() && depth < 200 {
            let moves = position.moves();
            if moves.is_empty() {
                break;
            }
            let index = ((u128::from(rng.next_u64()) * moves.len() as u128) >> 64) as usize;
            position = position.apply(moves[index]);
            depth += 1;
        }
        // Stored in the root maximiser's terms throughout; the selection
        // rule above is what turns that into each player's own preference.
        let outcome = position.evaluate() as f64;

        // Backpropagation.
        for &k in &path {
            nodes[k].visits += 1.0;
            nodes[k].total += outcome;
        }
    }

    let best = nodes[0]
        .children
        .iter()
        .max_by(|&&x, &&y| {
            nodes[x].visits.partial_cmp(&nodes[y].visits).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied();
    Ok(best.map_or(root_moves[0], |k| nodes[k].move_taken))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    // -----------------------------------------------------------------
    // Zero-sum games
    // -----------------------------------------------------------------

    #[test]
    fn the_minimax_theorem_holds_on_every_game_it_is_stated_for() {
        // The theorem says the row player's floor equals the column player's
        // ceiling. Both are computable directly from the returned strategies
        // -- the floor is the worst column against the row mixture -- so this
        // checks the theorem rather than the solver's own arithmetic.
        let games = [
            rock_paper_scissors(),
            matching_pennies(),
            Matrix::from_rows(&[&[3.0, -1.0], &[-2.0, 4.0]]).unwrap(),
            Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 0.0, -1.0]]).unwrap(),
            Matrix::from_rows(&[&[5.0]]).unwrap(),
        ];
        for game in &games {
            let (value, row, column) = minimax_value(game).unwrap();
            assert!(close(row.iter().sum::<f64>(), 1.0, 1e-7), "the row mixture is {row:?}");
            assert!(close(column.iter().sum::<f64>(), 1.0, 1e-7), "the column mixture is {column:?}");
            assert!(row.iter().all(|p| *p >= -1e-9) && column.iter().all(|p| *p >= -1e-9));

            // The floor: the worst any column can do to the row mixture.
            let floor = (0..game.cols)
                .map(|j| (0..game.rows).map(|i| row[i] * game.get(i, j)).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            // The ceiling: the best any row can do against the column mixture.
            let ceiling = (0..game.rows)
                .map(|i| dot(game.row(i), &column))
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                close(floor, ceiling, 1e-7),
                "the floor is {floor} and the ceiling {ceiling}, which the minimax theorem forbids"
            );
            assert!(close(value, floor, 1e-7), "the reported value {value} is not the floor {floor}");
        }
    }

    #[test]
    fn rock_paper_scissors_is_uniform_and_worth_nothing() {
        // The symmetry forces it: any deviation from uniform is exploitable
        // by the strategy that beats whatever is overweighted.
        let (value, row, column) = minimax_value(&rock_paper_scissors()).unwrap();
        assert!(close(value, 0.0, 1e-9), "the value is {value}");
        for (p, q) in row.iter().zip(&column) {
            assert!(close(*p, 1.0 / 3.0, 1e-7) && close(*q, 1.0 / 3.0, 1e-7));
        }
        // And the uniform mixture really is unexploitable: every pure reply
        // earns exactly zero.
        let game = rock_paper_scissors();
        for i in 0..3 {
            assert!(close(dot(game.row(i), &uniform(3)), 0.0, 1e-12));
        }
    }

    #[test]
    fn a_dominated_strategy_is_never_played_and_elimination_finds_the_same_set() {
        // Strict domination has the property that makes elimination sound:
        // the eliminated strategy carries zero weight in the optimal mixture.
        let game = Matrix::from_rows(&[&[4.0, 3.0], &[2.0, 1.0], &[0.0, -5.0]]).unwrap();
        let dominated = dominated_strategies(&game);
        assert_eq!(dominated, vec![1, 2], "rows one and two are dominated by row zero");
        let (_, row, _) = minimax_value(&game).unwrap();
        for &i in &dominated {
            assert!(close(row[i], 0.0, 1e-7), "a dominated row carries weight {}", row[i]);
        }

        // In the prisoner's dilemma both players' cooperation goes first.
        let pd = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        let (rows, cols) = iterated_elimination(&pd, &pd.transpose()).unwrap();
        assert_eq!((rows, cols), (vec![1], vec![1]), "only mutual defection survives");

        // Nothing is dominated in rock-paper-scissors, so nothing goes.
        let rps = rock_paper_scissors();
        let (rows, cols) = iterated_elimination(&rps, &rps.scale(-1.0)).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(cols.len(), 3);
        assert!(dominated_strategies(&rps).is_empty());
    }

    // -----------------------------------------------------------------
    // Bimatrix equilibria
    // -----------------------------------------------------------------

    #[test]
    fn every_reported_equilibrium_survives_the_deviation_test() {
        // The definition, applied to the output of all three methods. An
        // equilibrium is a profile from which no unilateral deviation pays,
        // and that is checkable without reference to how it was found.
        let a = Matrix::from_rows(&[&[3.0, 0.0], &[5.0, 1.0]]).unwrap();
        let b = Matrix::from_rows(&[&[3.0, 5.0], &[0.0, 1.0]]).unwrap();
        let cases: Vec<(Matrix, Matrix)> = vec![
            (a, b),
            (chicken(), chicken().transpose()),
            (stag_hunt(), stag_hunt().transpose()),
            (matching_pennies(), matching_pennies().scale(-1.0)),
            (
                Matrix::from_rows(&[&[2.0, 1.0], &[0.0, 3.0]]).unwrap(),
                Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 0.0]]).unwrap(),
            ),
        ];
        for (a, b) in &cases {
            let all = nash_2x2(a, b).unwrap();
            assert!(!all.is_empty(), "Nash's theorem guarantees at least one equilibrium");
            for (p, q) in &all {
                let gain = nash_deviation_gain(a, b, p, q).unwrap();
                assert!(gain < 1e-7, "a profitable deviation of {gain} remains at {p:?}, {q:?}");
            }

            // Support enumeration must find the same set.
            let enumerated = nash_support_enumeration(a, b, 2).unwrap();
            assert_eq!(
                enumerated.len(),
                all.len(),
                "the two enumerations disagree: {enumerated:?} against {all:?}"
            );
            for (p, q) in &enumerated {
                assert!(nash_deviation_gain(a, b, p, q).unwrap() < 1e-7);
            }

            // And Lemke-Howson must land on one of them.
            let (p, q) = nash_bimatrix_lemke_howson(a, b, 0).unwrap();
            let gain = nash_deviation_gain(a, b, &p, &q).unwrap();
            assert!(gain < 1e-6, "Lemke-Howson returned {p:?}, {q:?} with a gain of {gain}");
        }
    }

    #[test]
    fn the_mixed_equilibrium_makes_the_opponent_indifferent_and_not_oneself() {
        // Matching pennies: both mix uniformly. The row player's uniform
        // mixture is what makes the *column* player indifferent, and the
        // column player's own payoffs never enter the row player's
        // calculation.
        let a = matching_pennies();
        let b = a.scale(-1.0);
        let all = nash_2x2(&a, &b).unwrap();
        assert_eq!(all.len(), 1, "matching pennies has exactly one equilibrium");
        let (p, q) = &all[0];
        assert!(close(p[0], 0.5, 1e-9) && close(q[0], 0.5, 1e-9), "got {p:?}, {q:?}");

        // Under p the column player's two payoffs are equal.
        let column_of = |j: usize| (0..2).map(|i| p[i] * b.get(i, j)).sum::<f64>();
        assert!(close(column_of(0), column_of(1), 1e-12));

        // Chicken has two pure and one mixed.
        let c = chicken();
        let all = nash_2x2(&c, &c.transpose()).unwrap();
        assert_eq!(all.len(), 3, "chicken has three equilibria, got {all:?}");
        let mixed = all.iter().filter(|(p, _)| p[0] > 1e-6 && p[0] < 1.0 - 1e-6).count();
        assert_eq!(mixed, 1);
    }

    #[test]
    fn lemke_howson_reaches_equilibria_from_several_starting_labels() {
        // Different dropped labels generally reach different equilibria, and
        // every one of them must be an equilibrium.
        let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]]).unwrap();
        let b = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 1.0]]).unwrap();
        let mut reached = Vec::new();
        for label in 0..4 {
            let (p, q) = nash_bimatrix_lemke_howson(&a, &b, label).unwrap();
            assert!(nash_deviation_gain(&a, &b, &p, &q).unwrap() < 1e-6, "label {label}");
            assert!(close(p.iter().sum::<f64>(), 1.0, 1e-9));
            assert!(close(q.iter().sum::<f64>(), 1.0, 1e-9));
            reached.push(p[0]);
        }
        assert!(
            reached.iter().any(|x| (x - reached[0]).abs() > 1e-6),
            "every label reached the same equilibrium {reached:?}"
        );
        assert!(nash_bimatrix_lemke_howson(&a, &b, 9).is_err());
    }

    #[test]
    fn a_correlated_equilibrium_can_beat_every_nash_equilibrium() {
        // Chicken is the standard example. The best Nash outcome averages
        // less than the correlated device that recommends the two asymmetric
        // outcomes with equal probability and never recommends the crash.
        let a = chicken();
        let b = a.transpose();
        let joint = correlated_equilibrium_lp(&a, &b).unwrap();

        let total: f64 = (0..2).flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| joint.get(i, j))
            .sum();
        assert!(close(total, 1.0, 1e-7), "the distribution sums to {total}");
        assert!((0..2).all(|i| (0..2).all(|j| joint.get(i, j) >= -1e-9)));

        // The incentive constraints, restated: obeying beats deviating.
        for i in 0..2 {
            for k in 0..2 {
                let gain: f64 =
                    (0..2).map(|j| joint.get(i, j) * (a.get(k, j) - a.get(i, j))).sum();
                assert!(gain <= 1e-7, "deviating from {i} to {k} gains {gain}");
            }
        }

        let welfare: f64 = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| joint.get(i, j) * (a.get(i, j) + b.get(i, j)))
            .sum();
        let best_nash = nash_2x2(&a, &b)
            .unwrap()
            .iter()
            .map(|(p, q)| bilinear(&a, p, q) + bilinear(&b, p, q))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            welfare >= best_nash - 1e-7,
            "correlation achieved {welfare}, below the best Nash outcome {best_nash}"
        );
        // The crash is never recommended.
        assert!(joint.get(1, 1) < 1e-7, "the correlated device recommends mutual aggression");
    }

    // -----------------------------------------------------------------
    // Dynamics
    // -----------------------------------------------------------------

    #[test]
    fn fictitious_play_converges_where_it_is_proved_to_and_not_where_it_is_not() {
        // Zero-sum: the empirical frequencies approach the minimax
        // strategies, which is Robinson's theorem.
        let rps = rock_paper_scissors();
        let (row, column) = fictitious_play(&rps, &rps.scale(-1.0), 20_000).unwrap();
        for i in 0..3 {
            assert!(
                close(row[i], 1.0 / 3.0, 0.02) && close(column[i], 1.0 / 3.0, 0.02),
                "the frequencies are {row:?}, {column:?}"
            );
        }

        // A game solvable by iterated dominance: it finds the survivor.
        let pd = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        let (row, _) = fictitious_play(&pd, &pd.transpose(), 500).unwrap();
        assert!(row[1] > 0.99, "defection should take over, got {row:?}");
    }

    #[test]
    fn the_replicator_dynamic_keeps_the_simplex_and_conserves_the_rps_orbit() {
        // Rock-paper-scissors has an interior fixed point at the uniform
        // mixture, and the product x*y*z is conserved along every interior
        // orbit. That is a genuine invariant of the flow, so a trajectory
        // that changes it is a numerical artefact rather than dynamics.
        let rps = rock_paper_scissors();
        let start = vec![0.5, 0.3, 0.2];
        let trajectory = replicator_dynamics(&rps, &start, 40.0, 0.005).unwrap();

        let product = |x: &[f64]| x[0] * x[1] * x[2];
        let initial = product(&start);
        for state in &trajectory {
            assert!(close(state.iter().sum::<f64>(), 1.0, 1e-9), "left the simplex at {state:?}");
            assert!(state.iter().all(|v| *v >= -1e-12));
            assert!(
                close(product(state), initial, 1e-4),
                "the conserved product drifted from {initial} to {}",
                product(state)
            );
        }
        // It genuinely moves: a conserved quantity is not the same as a
        // stationary trajectory.
        let travelled = trajectory
            .iter()
            .map(|s| (s[0] - start[0]).abs())
            .fold(0.0f64, f64::max);
        assert!(travelled > 0.1, "the trajectory barely moved");

        // The uniform mixture is a fixed point.
        let fixed = replicator_dynamics(&rps, &uniform(3), 5.0, 0.01).unwrap();
        for state in &fixed {
            assert!(state.iter().all(|v| close(*v, 1.0 / 3.0, 1e-9)));
        }
    }

    #[test]
    fn the_hawk_dove_ess_is_the_mixture_the_theory_predicts() {
        // With cost above value the ESS plays hawk with probability v / c,
        // and nothing else is stable. Both halves are checked: the predicted
        // mixture passes and perturbations of it fail.
        let (v, c) = (2.0, 5.0);
        let game = hawk_dove(v, c);
        let ess = vec![v / c, 1.0 - v / c];
        assert!(evolutionarily_stable_check(&game, &ess, 1e-9).unwrap(), "v / c should be stable");

        for share in [0.0f64, 0.2, 0.6, 1.0] {
            if close(share, v / c, 1e-9) {
                continue;
            }
            assert!(
                !evolutionarily_stable_check(&game, &[share, 1.0 - share], 1e-9).unwrap(),
                "playing hawk with probability {share} should not be stable"
            );
        }

        // And the replicator dynamic converges to it from anywhere interior.
        let trajectory = replicator_dynamics(&game, &[0.9, 0.1], 60.0, 0.01).unwrap();
        let end = trajectory.last().unwrap();
        assert!(close(end[0], v / c, 1e-3), "the dynamic settled at {end:?}");

        // When the resource is worth more than the injury, pure hawk is the
        // ESS instead.
        let cheap = hawk_dove(6.0, 2.0);
        assert!(evolutionarily_stable_check(&cheap, &[1.0, 0.0], 1e-9).unwrap());
    }

    // -----------------------------------------------------------------
    // The iterated prisoner's dilemma
    // -----------------------------------------------------------------

    #[test]
    fn the_tournament_scores_match_the_payoffs_the_strategies_earn() {
        // Two facts that hold whatever the strategies are. Mutual cooperation
        // between two always-cooperate players earns the reward payoff every
        // round, and always-defect can never be beaten in a direct pairing.
        let mut rng = Rng::new(0x_6A3E_0001);
        let rounds = 200;
        let pair: Vec<Box<dyn IpdStrategy>> = vec![Box::new(AlwaysCooperate), Box::new(TitForTat)];
        let table = iterated_pd_tournament(&pair, rounds, 0.0, &mut rng);
        // Both cooperate throughout: each plays itself and the other, so
        // three pairings of `rounds` rounds each at the reward payoff.
        for (_, score) in &table {
            assert!(close(*score, 3.0 * 3.0 * rounds as f64, 1e-9), "the score is {score}");
        }

        let all = standard_ipd_strategies();
        let table = iterated_pd_tournament(&all, 150, 0.0, &mut rng);
        assert_eq!(table.len(), all.len());
        assert!(table.windows(2).all(|w| w[0].1 >= w[1].1), "the table is not sorted");
        // Against always-cooperate, always-defect earns the temptation payoff
        // every round, which is the highest per-round payoff in the game.
        let defect_only: Vec<Box<dyn IpdStrategy>> =
            vec![Box::new(AlwaysDefect), Box::new(AlwaysCooperate)];
        let heads_up = iterated_pd_tournament(&defect_only, 100, 0.0, &mut rng);
        let defector = heads_up.iter().find(|(name, _)| name == "always-defect").unwrap().1;
        let cooperator = heads_up.iter().find(|(name, _)| name == "always-cooperate").unwrap().1;
        // Defect-vs-defect gives 1 twice per round, defect-vs-cooperate 5,
        // cooperate-vs-cooperate 3 twice.
        assert!(close(defector, (2.0 * 1.0 + 5.0) * 100.0, 1e-9), "the defector scored {defector}");
        assert!(
            close(cooperator, (0.0 + 2.0 * 3.0) * 100.0, 1e-9),
            "the cooperator scored {cooperator}"
        );
        assert!(defector > cooperator);
    }

    #[test]
    fn noise_is_what_separates_the_forgiving_strategies_from_the_unforgiving() {
        // Without noise, tit for tat and grim are indistinguishable against
        // cooperative opponents. With noise, a single mistaken defection
        // locks grim into permanent retaliation and costs it heavily, while
        // the generous variant recovers.
        let mut rng = Rng::new(0x_6A3E_0002);
        let quiet: Vec<Box<dyn IpdStrategy>> = vec![Box::new(TitForTat), Box::new(Grim)];
        let table = iterated_pd_tournament(&quiet, 300, 0.0, &mut rng);
        assert!(
            close(table[0].1, table[1].1, 1e-9),
            "without noise the two should be identical: {table:?}"
        );

        let noisy: Vec<Box<dyn IpdStrategy>> = vec![
            Box::new(Grim),
            Box::new(GenerousTitForTat { forgiveness: 0.3 }),
            Box::new(AlwaysCooperate),
        ];
        let table = iterated_pd_tournament(&noisy, 400, 0.02, &mut rng);
        let grim = table.iter().find(|(n, _)| n == "grim").unwrap().1;
        let generous = table.iter().find(|(n, _)| n == "generous-tit-for-tat").unwrap().1;
        assert!(
            generous > grim,
            "under noise forgiveness should pay: generous {generous} against grim {grim}"
        );
    }

    // -----------------------------------------------------------------
    // Cooperative games
    // -----------------------------------------------------------------

    #[test]
    fn the_shapley_value_satisfies_the_axioms_that_define_it() {
        // Efficiency, symmetry, and the null-player property, each checked on
        // a game constructed to exercise it.
        let n = 4;
        // A glove game: value one for any coalition with both a left and a
        // right glove, and player three holds nothing.
        let left = 0b0011u64;
        let right = 0b0100u64;
        let characteristic = |c: u64| -> f64 {
            f64::from(c & left != 0 && c & right != 0)
        };
        let values = shapley_value(&characteristic, n).unwrap();

        let grand = (1u64 << n) - 1;
        assert!(
            close(values.iter().sum::<f64>(), characteristic(grand), 1e-12),
            "efficiency fails: {values:?}"
        );
        // Players zero and one are interchangeable, so they get the same.
        assert!(close(values[0], values[1], 1e-12), "symmetry fails: {values:?}");
        // Player three is a null player: they add nothing to any coalition.
        assert!(close(values[3], 0.0, 1e-12), "the null player got {}", values[3]);
        // The single right glove is scarce, so it is worth more than a left.
        assert!(values[2] > values[0], "scarcity is not reflected: {values:?}");

        // Additivity: the value of a sum of games is the sum of the values.
        let other = |c: u64| -> f64 { c.count_ones() as f64 };
        let sum = |c: u64| characteristic(c) + other(c);
        let a = shapley_value(&characteristic, n).unwrap();
        let b = shapley_value(&other, n).unwrap();
        let ab = shapley_value(&sum, n).unwrap();
        for i in 0..n {
            assert!(close(ab[i], a[i] + b[i], 1e-12), "additivity fails at player {i}");
        }
        assert!(shapley_value(&characteristic, 0).is_err());
    }

    #[test]
    fn sampling_orderings_recovers_the_exact_shapley_value() {
        // The Monte Carlo estimator is unbiased, so with enough samples it
        // must approach the exact answer computed over all orderings.
        let mut rng = Rng::new(0x_6A3E_0003);
        let n = 5;
        let weights = [3.0f64, 1.0, 4.0, 1.0, 5.0];
        let characteristic = |c: u64| -> f64 {
            let total: f64 = (0..n).filter(|&i| c >> i & 1 == 1).map(|i| weights[i]).sum();
            total * total / 20.0
        };
        let exact = shapley_value(&characteristic, n).unwrap();
        let sampled = shapley_monte_carlo(&characteristic, n, 200_000, &mut rng).unwrap();
        for i in 0..n {
            assert!(
                close(sampled[i], exact[i], 0.02),
                "player {i}: sampled {} against exact {}",
                sampled[i],
                exact[i]
            );
        }
        // Efficiency holds sample by sample, not just in expectation, since
        // each ordering's marginal contributions telescope to the total.
        let grand = (1u64 << n) - 1;
        assert!(close(sampled.iter().sum::<f64>(), characteristic(grand), 1e-9));
    }

    #[test]
    fn the_core_is_empty_for_the_majority_game_and_the_shapley_value_is_not() {
        // Three players splitting a pound, any two of whom can take it all.
        // No allocation survives: whatever the split, some pair is short.
        let majority = |c: u64| -> f64 { f64::from(c.count_ones() >= 2) };
        let shapley = shapley_value(&majority, 3).unwrap();
        for v in &shapley {
            assert!(close(*v, 1.0 / 3.0, 1e-12), "symmetry gives an equal split: {shapley:?}");
        }
        assert!(
            !core_check_small(&majority, 3, &shapley).unwrap(),
            "the equal split cannot be in the core of the majority game"
        );
        // Nor is anything else.
        for a in 0..=10 {
            for b in 0..=(10 - a) {
                let allocation =
                    [a as f64 / 10.0, b as f64 / 10.0, (10 - a - b) as f64 / 10.0];
                assert!(!core_check_small(&majority, 3, &allocation).unwrap());
            }
        }

        // A convex game, by contrast, has the Shapley value inside its core.
        let convex = |c: u64| -> f64 {
            let k = c.count_ones() as f64;
            k * k
        };
        let shapley = shapley_value(&convex, 3).unwrap();
        assert!(
            core_check_small(&convex, 3, &shapley).unwrap(),
            "the Shapley value of a convex game lies in its core: {shapley:?}"
        );
    }

    #[test]
    fn the_nucleolus_is_efficient_and_lands_in_the_core_when_one_exists() {
        // The nucleolus always exists, and when the core is non-empty it is
        // a point of it. That is the property that distinguishes it from the
        // Shapley value, which can sit outside a non-empty core.
        let convex = |c: u64| -> f64 {
            let k = c.count_ones() as f64;
            k * k
        };
        let x = nucleolus_small(&convex, 3).unwrap();
        assert!(close(x.iter().sum::<f64>(), convex(0b111), 1e-6), "not efficient: {x:?}");
        assert!(core_check_small(&convex, 3, &x).unwrap(), "the nucleolus {x:?} is outside the core");
        for v in &x {
            assert!(close(*v, 3.0, 1e-5), "symmetry gives an equal split: {x:?}");
        }

        // A game with an empty core, where the answer is worth working out by
        // hand. Player zero alone is worth 30, the others nothing, and any
        // pair or the whole set 60. Efficiency plus the three pair
        // constraints give x_i <= -e for every i, so the total forces
        // e <= -20, and e = -20 pins every share at 20.
        let game = |c: u64| -> f64 {
            match c.count_ones() {
                0 => 0.0,
                1 => {
                    if c & 1 == 1 {
                        30.0
                    } else {
                        0.0
                    }
                }
                _ => 60.0,
            }
        };
        let x = nucleolus_small(&game, 3).unwrap();
        assert!(close(x.iter().sum::<f64>(), 60.0, 1e-6), "not efficient: {x:?}");
        for v in &x {
            assert!(close(*v, 20.0, 1e-6), "the nucleolus is (20, 20, 20), got {x:?}");
        }
        // Player zero is worth 30 alone and gets 20: with an empty core, the
        // stand-alone value cannot be honoured, and the nucleolus spreads the
        // unavoidable disappointment evenly rather than favouring anyone.
        assert!(!core_check_small(&game, 3, &x).unwrap(), "the core of this game is empty");
        for a in 0..=12 {
            for b in 0..=(12 - a) {
                let candidate = [a as f64 * 5.0, b as f64 * 5.0, (12 - a - b) as f64 * 5.0];
                assert!(!core_check_small(&game, 3, &candidate).unwrap(), "{candidate:?}");
            }
        }
        // The Shapley value answers differently, giving player zero 30.
        let shapley = shapley_value(&game, 3).unwrap();
        assert!(close(shapley[0], 30.0, 1e-9), "the Shapley values are {shapley:?}");
        assert!(close(shapley[1], 15.0, 1e-9) && close(shapley[2], 15.0, 1e-9));
        assert!(nucleolus_small(&game, 0).is_err());
    }

    #[test]
    fn voting_power_is_not_proportional_to_weight() {
        // The classic demonstration: with weights 4, 2, 1 and a quota of 4,
        // the largest party wins alone and the other two are dummies with no
        // power at all despite holding a third of the votes between them.
        let power = voting_power_weighted(&[4.0, 2.0, 1.0], 4.0).unwrap();
        assert!(close(power[0], 1.0, 1e-12), "the dictator's power is {}", power[0]);
        assert!(close(power[1], 0.0, 1e-12) && close(power[2], 0.0, 1e-12), "{power:?}");

        // With a quota of 5 the small parties matter again, and the weights
        // 4, 2, 1 give powers 3/5, 1/5, 1/5 -- so doubling a party's weight
        // from 1 to 2 buys it nothing.
        let power = voting_power_weighted(&[4.0, 2.0, 1.0], 5.0).unwrap();
        assert!(close(power[0], 0.6, 1e-9), "{power:?}");
        assert!(close(power[1], 0.2, 1e-9) && close(power[2], 0.2, 1e-9), "{power:?}");
        assert!(close(power.iter().sum::<f64>(), 1.0, 1e-12));

        // Three equal parties needing two of three: equal power, as symmetry
        // demands.
        let power = voting_power_weighted(&[1.0, 1.0, 1.0], 2.0).unwrap();
        for p in &power {
            assert!(close(*p, 1.0 / 3.0, 1e-12));
        }
        assert!(voting_power_weighted(&[], 1.0).is_err());
    }

    #[test]
    fn the_banzhaf_and_shapley_indices_disagree() {
        // They are different averages -- over coalitions rather than over
        // orderings -- so on most games they disagree. The weighted voting
        // game with weights 4, 2, 1 and a quota of 5 is small enough to check
        // both by hand: the Shapley value is 2/3, 1/6, 1/6 and the Banzhaf
        // index is 3/5, 1/5, 1/5.
        let weights = [4.0f64, 2.0, 1.0];
        let quota = 5.0;
        let characteristic = |c: u64| -> f64 {
            let total: f64 = (0..3).filter(|&i| c >> i & 1 == 1).map(|i| weights[i]).sum();
            f64::from(total >= quota)
        };
        let banzhaf = banzhaf_index(&characteristic, 3).unwrap();
        let shapley = shapley_value(&characteristic, 3).unwrap();
        assert!(close(banzhaf.iter().sum::<f64>(), 1.0, 1e-12));
        assert!(close(shapley.iter().sum::<f64>(), 1.0, 1e-12));
        assert!(close(shapley[0], 2.0 / 3.0, 1e-9), "the Shapley values are {shapley:?}");
        assert!(close(banzhaf[0], 0.6, 1e-9), "the Banzhaf indices are {banzhaf:?}");
        assert!(
            (0..3).any(|i| (banzhaf[i] - shapley[i]).abs() > 1e-6),
            "the two indices agreed exactly, which would be a coincidence"
        );
        // Both agree that the two small parties have equal power despite
        // holding different numbers of votes, since either one completes the
        // only winning coalition the large party does not already have.
        assert!(close(banzhaf[1], banzhaf[2], 1e-12) && close(shapley[1], shapley[2], 1e-12));
        assert!(banzhaf[0] > banzhaf[1]);

        // They also disagree on a game that is not a voting game at all.
        let squared = |c: u64| -> f64 {
            let k = f64::from(c.count_ones());
            k * k * f64::from(c & 1 == 1)
        };
        let a = banzhaf_index(&squared, 4).unwrap();
        let b = shapley_value(&squared, 4).unwrap();
        assert!((0..4).any(|i| (a[i] - b[i] / b.iter().sum::<f64>()).abs() > 1e-6));
    }

    // -----------------------------------------------------------------
    // Auctions
    // -----------------------------------------------------------------

    #[test]
    fn truthful_bidding_dominates_in_a_second_price_auction() {
        assert!(second_price_dominant_check(), "the dominance argument failed on some case");
        // And it does *not* dominate in a first-price auction: bidding one's
        // value there earns exactly zero.
        let shade = first_price_auction_equilibrium_uniform(4);
        assert!(close(shade, 0.75, 1e-12));
        assert!(first_price_auction_equilibrium_uniform(2) < first_price_auction_equilibrium_uniform(10));
        // Shading vanishes as the field grows.
        assert!(close(first_price_auction_equilibrium_uniform(1000), 0.999, 1e-12));
    }

    #[test]
    fn the_two_auction_formats_raise_the_same_revenue() {
        // Revenue equivalence, checked against the closed form as well as
        // against each other: with n uniform bidders the expected revenue is
        // the second highest order statistic, (n - 1) / (n + 1).
        let mut rng = Rng::new(0x_6A3E_0004);
        for n in [2usize, 3, 5, 10] {
            let (first, second) = revenue_equivalence_sim(n, 200_000, &mut rng).unwrap();
            let expected = (n as f64 - 1.0) / (n as f64 + 1.0);
            assert!(
                close(second, expected, 0.01),
                "with {n} bidders the second-price revenue is {second}, not {expected}"
            );
            assert!(
                close(first, second, 0.01),
                "with {n} bidders the formats raised {first} and {second}"
            );
        }
        assert!(revenue_equivalence_sim(1, 10, &mut rng).is_err());
    }

    #[test]
    fn vcg_charges_each_winner_the_harm_they_do_to_the_others() {
        // One item, three bidders: the winner pays the second highest bid,
        // which is the second-price auction as a special case of VCG.
        let bids = vec![vec![10.0], vec![7.0], vec![4.0]];
        let (assignment, payments) = vcg_auction(&bids, 1).unwrap();
        assert_eq!(assignment[0], Some(0), "the highest bidder should win");
        assert_eq!(assignment[1], None);
        assert!(close(payments[0], 7.0, 1e-9), "the winner paid {}", payments[0]);
        assert!(close(payments[1], 0.0, 1e-9) && close(payments[2], 0.0, 1e-9));

        // Two items, where the efficient assignment is not the greedy one.
        let bids = vec![vec![10.0, 9.0], vec![8.0, 1.0]];
        let (assignment, payments) = vcg_auction(&bids, 2).unwrap();
        let welfare: f64 = (0..2)
            .filter_map(|i| assignment[i].map(|k| bids[i][k]))
            .sum();
        // No other assignment does better, enumerated rather than asserted:
        // each bidder takes item zero, item one, or nothing.
        for first in [None, Some(0), Some(1)] {
            for second in [None, Some(0), Some(1)] {
                if first.is_some() && first == second {
                    continue;
                }
                let alternative: f64 = [first, second]
                    .iter()
                    .enumerate()
                    .filter_map(|(i, choice)| choice.map(|k: usize| bids[i][k]))
                    .sum();
                assert!(
                    alternative <= welfare + 1e-9,
                    "the assignment {first:?}, {second:?} is worth {alternative}, above {welfare}"
                );
            }
        }
        assert!(close(welfare, 17.0, 1e-9), "the efficient welfare is 17, got {welfare}");
        // No bidder pays more than they bid: individual rationality.
        for i in 0..2 {
            if let Some(k) = assignment[i] {
                assert!(payments[i] <= bids[i][k] + 1e-9, "bidder {i} paid above their value");
            }
        }
        assert!(vcg_auction(&[vec![1.0, 2.0]], 1).is_err());
        assert!(vcg_auction(&[], 1).is_err());
    }

    // -----------------------------------------------------------------
    // Fair division and matching
    // -----------------------------------------------------------------

    #[test]
    fn divide_and_choose_gives_the_cutter_a_half_and_the_chooser_at_least_one() {
        // The guarantee is asymmetric and both halves are exact. The cutter
        // gets exactly half by their own measure whatever the chooser thinks;
        // the chooser gets at least half by theirs, and strictly more when
        // the two valuations differ.
        let cases: Vec<(Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = vec![
            (Box::new(|_| 1.0), Box::new(|_| 1.0)),
            (Box::new(|_| 1.0), Box::new(|x: f64| 1.0 + 3.0 * x)),
            (Box::new(|x: f64| (1.0 - x).max(0.05)), Box::new(|x: f64| x.max(0.05))),
            (Box::new(|x: f64| (4.0 * x).exp()), Box::new(|_| 1.0)),
        ];
        for (a, b) in &cases {
            let (cutter, chooser) = cake_cutting_divide_choose(a.as_ref(), b.as_ref(), 4000).unwrap();
            assert!(cutter >= 0.5 - 2e-3, "the cutter got {cutter}");
            assert!(cutter <= 0.5 + 2e-3, "the cutter got more than half: {cutter}");
            assert!(chooser >= 0.5 - 1e-9, "the chooser got {chooser}");
        }
        // Opposed tastes: the chooser does much better than half.
        let (_, chooser) = cake_cutting_divide_choose(
            &|x: f64| (1.0 - x).max(0.01),
            &|x: f64| x.max(0.01),
            4000,
        )
        .unwrap();
        assert!(chooser > 0.7, "with opposed tastes the chooser should do far better: {chooser}");
        assert!(cake_cutting_divide_choose(&|_| 1.0, &|_| 1.0, 0).is_err());
        assert!(cake_cutting_divide_choose(&|_| 0.0, &|_| 1.0, 10).is_err());
    }

    #[test]
    fn deferred_acceptance_is_proposer_optimal_and_receiver_pessimal() {
        // The classic instance where the two sides' optimal stable matchings
        // are different, so the check has something to detect.
        let a = vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]];
        let b = vec![vec![1, 2, 0], vec![2, 0, 1], vec![0, 1, 2]];
        assert!(gale_shapley_optimality_check(&a, &b).unwrap());

        // A larger random-looking instance.
        let a = vec![
            vec![1, 0, 3, 2],
            vec![3, 1, 2, 0],
            vec![0, 2, 1, 3],
            vec![2, 3, 0, 1],
        ];
        let b = vec![
            vec![2, 1, 0, 3],
            vec![0, 3, 2, 1],
            vec![3, 0, 1, 2],
            vec![1, 2, 3, 0],
        ];
        assert!(gale_shapley_optimality_check(&a, &b).unwrap());

        assert!(gale_shapley_optimality_check(&[vec![0]], &[vec![0], vec![0]]).is_err());
        assert!(gale_shapley_optimality_check(&[vec![0, 0]], &[vec![0, 1]]).is_err());
    }

    // -----------------------------------------------------------------
    // Market models
    // -----------------------------------------------------------------

    #[test]
    fn cournot_quantities_are_mutual_best_responses() {
        // Equilibrium is a fixed point, so the test is a fixed-point check:
        // no firm can raise its own profit by changing its quantity alone.
        let (intercept, slope) = (100.0f64, 1.0f64);
        for costs in [vec![10.0, 10.0], vec![10.0, 20.0, 30.0], vec![5.0; 5], vec![40.0, 41.0]] {
            let q = cournot_equilibrium(intercept, slope, &costs).unwrap();
            let total: f64 = q.iter().sum();
            let price = intercept - slope * total;
            for i in 0..costs.len() {
                let others: f64 = total - q[i];
                let profit = |own: f64| (intercept - slope * (others + own) - costs[i]) * own;
                let base = profit(q[i]);
                for delta in [-2.0f64, -0.5, -0.01, 0.01, 0.5, 2.0] {
                    let alternative = (q[i] + delta).max(0.0);
                    assert!(
                        profit(alternative) <= base + 1e-7,
                        "firm {i} gains by moving from {} to {alternative}",
                        q[i]
                    );
                }
            }
            // Price stays above the lowest marginal cost, which is the point
            // of the comparison with Bertrand.
            let cheapest = costs.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(price > cheapest, "Cournot price {price} fell to marginal cost");
            assert!(price > bertrand_equilibrium(&costs).unwrap() - 1e-9);
        }
        assert!(cournot_equilibrium(100.0, 0.0, &[1.0]).is_err());
        assert!(cournot_equilibrium(100.0, 1.0, &[]).is_err());
        assert!(bertrand_equilibrium(&[1.0]).is_err());
    }

    #[test]
    fn committing_first_never_hurts_the_leader() {
        // The Stackelberg payoff is at least the best Nash payoff, because
        // the leader could commit to their equilibrium strategy and get the
        // same. In the standard entry game it is strictly better.
        let a = Matrix::from_rows(&[&[2.0, 1.0], &[0.0, 3.0]]).unwrap();
        let b = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 0.0]]).unwrap();
        let (leader, follower, leader_payoff, follower_payoff) = stackelberg_2x2(&a, &b).unwrap();
        assert!(close(leader_payoff, a.get(leader, follower), 1e-12));
        assert!(close(follower_payoff, b.get(leader, follower), 1e-12));
        // The follower really is best-responding.
        let best = (0..2).map(|j| b.get(leader, j)).fold(f64::NEG_INFINITY, f64::max);
        assert!(close(b.get(leader, follower), best, 1e-12));

        // Pure commitment beats every *pure* equilibrium, since the leader
        // could always commit to their equilibrium row and the follower's
        // reply would be unchanged.
        let pure_best = nash_2x2(&a, &b)
            .unwrap()
            .iter()
            .filter(|(p, _)| p.iter().any(|v| (v - 1.0).abs() < 1e-9))
            .map(|(p, q)| bilinear(&a, p, q))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            leader_payoff >= pure_best - 1e-9,
            "committing gave {leader_payoff}, below the best pure Nash payoff {pure_best}"
        );

        // And it can be strictly better. Here the row player's second row is
        // dominated in the simultaneous game, but committing to it changes
        // the column player's reply and pays the leader more.
        let a = Matrix::from_rows(&[&[2.0, 4.0], &[1.0, 3.0]]).unwrap();
        let b = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]]).unwrap();
        let (leader, follower, payoff, _) = stackelberg_2x2(&a, &b).unwrap();
        assert_eq!((leader, follower), (1, 1), "the leader should commit to the second row");
        assert!(close(payoff, 3.0, 1e-12));
        let equilibria = nash_2x2(&a, &b).unwrap();
        assert_eq!(equilibria.len(), 1, "this game has one equilibrium: {equilibria:?}");
        let nash_payoff = bilinear(&a, &equilibria[0].0, &equilibria[0].1);
        assert!(close(nash_payoff, 2.0, 1e-12), "the Nash payoff is {nash_payoff}");
        assert!(payoff > nash_payoff, "committing should strictly help here");

        // But a *pure* commitment is not the general theorem, and it can lose
        // to a mixed equilibrium: the leader gives up the mixture they would
        // otherwise have randomised over.
        let a = Matrix::from_rows(&[&[2.0, 1.0], &[0.0, 3.0]]).unwrap();
        let b = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 0.0]]).unwrap();
        let (_, _, committed, _) = stackelberg_2x2(&a, &b).unwrap();
        let mixed = nash_2x2(&a, &b)
            .unwrap()
            .iter()
            .map(|(p, q)| bilinear(&a, p, q))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            committed < mixed,
            "pure commitment gave {committed} and the mixed equilibrium {mixed}, \
             so this game no longer shows the gap"
        );
        assert!(stackelberg_2x2(&rock_paper_scissors(), &rock_paper_scissors()).is_err());
    }

    #[test]
    fn contributions_to_a_public_good_decay_when_free_riding_pays() {
        // The threshold is at multiplier = n, where a unit contributed
        // returns exactly its cost. Below it contributions decay; above it
        // they hold up. Both sides are checked, since a model that only
        // showed the decay would not distinguish free riding from a rule that
        // decays regardless.
        let mut rng = Rng::new(0x_6A3E_0005);
        let selfish = public_goods_game_sim(6, 2.0, 200, &mut rng).unwrap();
        assert_eq!(selfish.len(), 200);
        let early: f64 = selfish[..20].iter().sum::<f64>() / 20.0;
        let late: f64 = selfish[180..].iter().sum::<f64>() / 20.0;
        assert!(late < early, "contributions rose from {early} to {late} despite free riding");
        assert!(late < 0.15, "contributions settled at {late} rather than collapsing");

        let generous = public_goods_game_sim(4, 6.0, 200, &mut rng).unwrap();
        let late: f64 = generous[180..].iter().sum::<f64>() / 20.0;
        assert!(late > 0.8, "with a multiplier above the group size contributions should hold: {late}");
        assert!(public_goods_game_sim(1, 2.0, 10, &mut rng).is_err());
    }

    #[test]
    fn no_blotto_allocation_beats_every_other() {
        // The absence of a dominant strategy is the whole content of the
        // game: whatever allocation is sampled, some other beats it.
        let mut rng = Rng::new(0x_6A3E_0006);
        let results = colonel_blotto_sim(3, 12, 12, &mut rng).unwrap();
        assert_eq!((results.rows, results.cols), (12, 12));
        // The matrix is antisymmetric: beating and being beaten are mirror
        // images.
        for i in 0..12 {
            assert!(close(results.get(i, i), 0.0, 1e-12));
            for j in 0..12 {
                assert!(close(results.get(i, j), -results.get(j, i), 1e-12));
            }
        }
        for i in 0..12 {
            let unbeaten = (0..12).all(|j| j == i || results.get(i, j) > 0.0);
            assert!(!unbeaten, "allocation {i} beat every other, which Blotto forbids");
        }
        assert!(colonel_blotto_sim(1, 10, 5, &mut rng).is_err());
    }

    // -----------------------------------------------------------------
    // Trees and search
    // -----------------------------------------------------------------

    #[test]
    fn backward_induction_rules_out_the_incredible_threat() {
        // The entry game: an incumbent threatens to fight, but fighting hurts
        // them too, so once entry has happened they accommodate. Backward
        // induction sees that and the entrant enters.
        let tree = GameTree::Node {
            player: 0,
            children: vec![
                // Stay out.
                GameTree::Leaf(vec![0.0, 10.0]),
                // Enter, and the incumbent chooses.
                GameTree::Node {
                    player: 1,
                    children: vec![
                        GameTree::Leaf(vec![-3.0, -2.0]), // fight
                        GameTree::Leaf(vec![4.0, 5.0]),   // accommodate
                    ],
                },
            ],
        };
        let (path, payoffs) = backward_induction(&tree).unwrap();
        assert_eq!(path, vec![1, 1], "the entrant enters and the incumbent accommodates");
        assert!(close(payoffs[0], 4.0, 1e-12) && close(payoffs[1], 5.0, 1e-12));

        // Make fighting genuinely profitable and the threat becomes credible.
        let tree = GameTree::Node {
            player: 0,
            children: vec![
                GameTree::Leaf(vec![0.0, 10.0]),
                GameTree::Node {
                    player: 1,
                    children: vec![
                        GameTree::Leaf(vec![-3.0, 8.0]),
                        GameTree::Leaf(vec![4.0, 5.0]),
                    ],
                },
            ],
        };
        let (path, payoffs) = backward_induction(&tree).unwrap();
        assert_eq!(path, vec![0], "staying out is now the entrant's best move");
        assert!(close(payoffs[0], 0.0, 1e-12));

        assert!(backward_induction(&GameTree::Leaf(vec![])).is_err());
        assert!(backward_induction(&GameTree::Node { player: 0, children: vec![] }).is_err());
        assert!(backward_induction(&GameTree::Node {
            player: 5,
            children: vec![GameTree::Leaf(vec![1.0])]
        })
        .is_err());
    }

    /// Tic-tac-toe as a `GameState`, with the crosses player maximising.
    #[derive(Clone)]
    struct TicTacToe {
        board: [i8; 9],
        crosses_to_move: bool,
    }

    impl TicTacToe {
        fn new() -> Self {
            Self { board: [0; 9], crosses_to_move: true }
        }
        fn winner(&self) -> i8 {
            const LINES: [[usize; 3]; 8] = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],
                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                [0, 4, 8], [2, 4, 6],
            ];
            for line in LINES {
                let [a, b, c] = line;
                if self.board[a] != 0
                    && self.board[a] == self.board[b]
                    && self.board[b] == self.board[c]
                {
                    return self.board[a];
                }
            }
            0
        }
    }

    impl GameState for TicTacToe {
        fn moves(&self) -> Vec<usize> {
            if self.winner() != 0 {
                return Vec::new();
            }
            (0..9).filter(|&i| self.board[i] == 0).collect()
        }
        fn apply(&self, mv: usize) -> Self {
            let mut next = self.clone();
            next.board[mv] = if self.crosses_to_move { 1 } else { -1 };
            next.crosses_to_move = !self.crosses_to_move;
            next
        }
        fn evaluate(&self) -> i64 {
            i64::from(self.winner())
        }
        fn maximising(&self) -> bool {
            self.crosses_to_move
        }
        fn terminal(&self) -> bool {
            self.winner() != 0 || self.board.iter().all(|&c| c != 0)
        }
    }

    #[test]
    fn alpha_beta_agrees_with_minimax_and_visits_fewer_nodes() {
        // The pruning is exact, so the two must return the same value at
        // every position -- and the whole point is that alpha-beta gets there
        // cheaper.
        let root = TicTacToe::new();
        let (pruned, _) = alpha_beta_search(&root, 9);
        let (plain, plain_nodes) = minimax_search(&root, 9);
        assert_eq!(pruned, plain, "the pruning changed the value");
        assert_eq!(plain, 0, "tic-tac-toe is a draw under perfect play");

        // Agreement holds at every reachable position too, not just the root.
        for first in 0..9 {
            let after = root.apply(first);
            let (pruned, _) = alpha_beta_search(&after, 9);
            let (plain_here, _) = minimax_search(&after, 9);
            assert_eq!(pruned, plain_here, "the two disagree after the opening move {first}");
        }
        assert!(plain_nodes > 500_000, "the unpruned search was suspiciously small");

        // A won position is recognised at once.
        let mut nearly = TicTacToe::new();
        nearly.board = [1, 1, 0, -1, -1, 0, 0, 0, 0];
        let (value, mv) = alpha_beta_search(&nearly, 9);
        assert_eq!(value, 1, "crosses have a forced win");
        assert_eq!(mv, Some(2), "crosses must complete the top row");
    }

    #[test]
    fn mcts_finds_the_move_a_full_search_would() {
        // Not a claim about tree search in general -- a shallow rollout
        // policy fails at plenty of positions -- but on an immediate win it
        // must agree with the exact answer, and it must never return an
        // illegal move.
        let mut rng = Rng::new(0x_6A3E_0007);
        let mut nearly = TicTacToe::new();
        nearly.board = [1, 1, 0, -1, -1, 0, 0, 0, 0];
        let chosen = mcts_lite(&nearly, 4000, 1.4, &mut rng).unwrap();
        assert_eq!(chosen, 2, "the winning move should dominate the visit counts");

        // A forced block: crosses must stop the noughts row.
        let mut block = TicTacToe::new();
        block.board = [-1, -1, 0, 1, 0, 0, 0, 0, 1];
        let chosen = mcts_lite(&block, 6000, 1.4, &mut rng).unwrap();
        assert_eq!(chosen, 2, "crosses must block, got {chosen}");

        // Legality, from an ordinary position.
        let mut mid = TicTacToe::new();
        mid.board = [1, 0, -1, 0, 1, 0, 0, 0, -1];
        let chosen = mcts_lite(&mid, 1500, 1.4, &mut rng).unwrap();
        assert!(mid.moves().contains(&chosen), "{chosen} is not a legal move");

        let finished = TicTacToe { board: [1, 1, 1, -1, -1, 0, 0, 0, 0], crosses_to_move: false };
        assert!(mcts_lite(&finished, 100, 1.4, &mut rng).is_err());
        assert!(mcts_lite(&TicTacToe::new(), 0, 1.4, &mut rng).is_err());
    }

    #[test]
    fn the_solvers_refuse_mismatched_and_degenerate_input() {
        let two = Matrix::zeros(2, 2);
        let three = Matrix::zeros(3, 3);
        assert!(iterated_elimination(&two, &three).is_err());
        assert!(nash_2x2(&three, &three).is_err());
        assert!(nash_support_enumeration(&two, &three, 2).is_err());
        assert!(nash_bimatrix_lemke_howson(&two, &three, 0).is_err());
        assert!(correlated_equilibrium_lp(&two, &three).is_err());
        assert!(fictitious_play(&two, &three, 10).is_err());
        assert!(nash_deviation_gain(&two, &two, &[1.0], &[0.5, 0.5]).is_err());
        assert!(replicator_dynamics(&Matrix::zeros(2, 3), &[0.5, 0.5], 1.0, 0.1).is_err());
        assert!(replicator_dynamics(&two, &[0.5, 0.4], 1.0, 0.1).is_err());
        assert!(replicator_dynamics(&two, &[0.5, 0.5], 1.0, 0.0).is_err());
        assert!(evolutionarily_stable_check(&Matrix::zeros(2, 3), &[0.5, 0.5], 1e-9).is_err());
        assert!(evolutionarily_stable_check(&two, &[0.5, 0.4], 1e-9).is_err());
        assert!(shapley_monte_carlo(&|_| 0.0, 3, 0, &mut Rng::new(1)).is_err());
        assert!(banzhaf_index(&|_| 0.0, 21).is_err());
        assert!(core_check_small(&|_| 0.0, 3, &[1.0]).is_err());
    }

    #[test]
    #[should_panic(expected = "t > r > p > s")]
    fn the_prisoners_dilemma_rejects_payoffs_that_are_not_a_dilemma() {
        let _ = prisoners_dilemma(1.0, 2.0, 3.0, 4.0);
    }

    #[test]
    #[should_panic(expected = "positive cost")]
    fn hawk_dove_rejects_a_free_fight() {
        let _ = hawk_dove(1.0, 0.0);
    }
}
