//! Linear programming: the simplex method, interior point methods, duality,
//! and the classical models that reduce to a linear program.
//!
//! This module sits alongside the continuous optimisers in the parent module
//! rather than replacing them. Those search a smooth objective by following
//! gradients or shrinking a simplex, and stop at a local optimum. A linear
//! program has no local optima to stop at: the objective is linear and the
//! feasible region is a convex polyhedron, so any local optimum is global and
//! at least one optimum sits at a vertex. That is the whole reason the
//! subject exists as a separate discipline, and why an exact answer is
//! available where a nonlinear problem admits only an approximation.
//!
//! Two solvers are provided because they fail in different ways. The simplex
//! method walks vertex to vertex along the boundary, and terminates in an
//! exactly optimal basis, but its worst case is exponential and it can cycle
//! in the presence of degeneracy -- handled here by Bland's rule, which
//! guarantees termination at the cost of speed. The interior point method
//! approaches the optimum through the middle of the region, takes a number of
//! iterations that barely grows with problem size, and never lands exactly on
//! a vertex. Running both on the same problem and comparing is the cheapest
//! real check available on either.
//!
//! Duality is the organising idea. Every linear program has a dual whose
//! optimal value equals its own, and whose optimal solution is the vector of
//! rates at which the primal objective responds to relaxing each constraint.
//! Those rates -- shadow prices -- are usually worth more than the solution
//! itself, since they say which constraint to attack. The convention used
//! here is stated once and adhered to throughout:
//!
//! > `duals[i]` is the derivative of the reported objective with respect to
//! > `b[i]`.
//!
//! That definition is what makes the sensitivity ranges mean something, and
//! it is what the tests check: perturbing a right-hand side within its range
//! changes the objective by exactly `duals[i]` times the perturbation.

use crate::error::GeomError;
use crate::linalg::matrix::Matrix;

/// A pivot smaller than this is treated as numerically zero.
const PIVOT_TOL: f64 = 1e-9;
/// Reduced costs and residuals within this of zero are treated as zero.
const OPT_TOL: f64 = 1e-9;
/// Iteration cap; Bland's rule guarantees termination, so hitting this means
/// the problem is far larger than the tableau method should be used on.
const MAX_PIVOTS: usize = 200_000;

/// The sense of a constraint row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// `a . x <= b`
    Le,
    /// `a . x >= b`
    Ge,
    /// `a . x == b`
    Eq,
}

impl Cmp {
    /// The sense obtained by multiplying the row through by `-1`.
    fn flipped(self) -> Self {
        match self {
            Cmp::Le => Cmp::Ge,
            Cmp::Ge => Cmp::Le,
            Cmp::Eq => Cmp::Eq,
        }
    }
}

/// A linear program.
///
/// Minimises (or maximises) `c . x` subject to the rows of `a` compared
/// against `b` by `constraint_types`, with each variable confined to its
/// entry of `bounds`. A bound of `(0.0, f64::INFINITY)` is the default
/// non-negative variable; `(f64::NEG_INFINITY, f64::INFINITY)` makes a
/// variable free.
#[derive(Debug, Clone, PartialEq)]
pub struct LpProblem {
    /// Objective coefficients, one per variable.
    pub c: Vec<f64>,
    /// Constraint matrix, one row per constraint.
    pub a: Matrix,
    /// Right-hand sides, one per constraint.
    pub b: Vec<f64>,
    /// Sense of each constraint row.
    pub constraint_types: Vec<Cmp>,
    /// Per-variable `(lower, upper)` bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Whether to maximise rather than minimise.
    pub maximize: bool,
}

impl LpProblem {
    /// A problem in the common shape: `A x <= b`, `x >= 0`.
    ///
    /// # Errors
    /// Returns an error if the shapes disagree.
    pub fn new(c: Vec<f64>, a: Matrix, b: Vec<f64>, maximize: bool) -> Result<Self, GeomError> {
        let m = b.len();
        let p = Self {
            constraint_types: vec![Cmp::Le; m],
            bounds: vec![(0.0, f64::INFINITY); c.len()],
            c,
            a,
            b,
            maximize,
        };
        p.validate()?;
        Ok(p)
    }

    /// Number of variables.
    #[must_use]
    pub fn n(&self) -> usize {
        self.c.len()
    }

    /// Number of constraints.
    #[must_use]
    pub fn m(&self) -> usize {
        self.b.len()
    }

    /// Checks that every part of the problem has a consistent shape.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] describing the first mismatch.
    pub fn validate(&self) -> Result<(), GeomError> {
        if self.c.is_empty() {
            return Err(GeomError::InvalidArgument("an LP needs at least one variable"));
        }
        if self.a.rows != self.b.len() || self.a.cols != self.c.len() {
            return Err(GeomError::InvalidArgument("LP matrix shape does not match c and b"));
        }
        if self.constraint_types.len() != self.b.len() {
            return Err(GeomError::InvalidArgument("one constraint sense per row is required"));
        }
        if self.bounds.len() != self.c.len() {
            return Err(GeomError::InvalidArgument("one bound pair per variable is required"));
        }
        for (lo, hi) in &self.bounds {
            if lo > hi {
                return Err(GeomError::InvalidArgument("a lower bound exceeds its upper bound"));
            }
            if hi.is_infinite() && hi.is_sign_negative() {
                return Err(GeomError::InvalidArgument("an upper bound is negative infinity"));
            }
        }
        if self.c.iter().chain(&self.b).any(|v| !v.is_finite()) {
            return Err(GeomError::InvalidArgument("LP coefficients must be finite"));
        }
        Ok(())
    }

    /// The objective value at a point, in the problem's own sense.
    #[must_use]
    pub fn objective_at(&self, x: &[f64]) -> f64 {
        self.c.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    /// Whether `x` satisfies every constraint and bound to within `tol`.
    #[must_use]
    pub fn is_feasible(&self, x: &[f64], tol: f64) -> bool {
        if x.len() != self.n() {
            return false;
        }
        for (j, &v) in x.iter().enumerate() {
            let (lo, hi) = self.bounds[j];
            if v < lo - tol || v > hi + tol {
                return false;
            }
        }
        for i in 0..self.m() {
            let row: f64 = (0..self.n()).map(|j| self.a.get(i, j) * x[j]).sum();
            let ok = match self.constraint_types[i] {
                Cmp::Le => row <= self.b[i] + tol,
                Cmp::Ge => row >= self.b[i] - tol,
                Cmp::Eq => (row - self.b[i]).abs() <= tol,
            };
            if !ok {
                return false;
            }
        }
        true
    }
}

/// What a solver concluded.
#[derive(Debug, Clone, PartialEq)]
pub enum LpResult {
    /// An optimal vertex was found.
    Optimal {
        /// The optimal point.
        x: Vec<f64>,
        /// The objective there, in the problem's own sense.
        objective: f64,
        /// `duals[i]` is `d(objective) / d(b[i])`: the shadow price of row `i`.
        duals: Vec<f64>,
        /// `reduced_costs[j]` is `c[j] - sum_i duals[i] a[i][j]`, the rate at
        /// which the objective would worsen per unit of variable `j` forced
        /// into the solution. Zero for every variable already in use, which
        /// is complementary slackness.
        reduced_costs: Vec<f64>,
    },
    /// No point satisfies every constraint.
    Infeasible,
    /// The objective improves without bound inside the feasible region.
    Unbounded,
}

impl LpResult {
    /// The optimal objective, or `None` if the problem had no optimum.
    #[must_use]
    pub fn objective(&self) -> Option<f64> {
        match self {
            LpResult::Optimal { objective, .. } => Some(*objective),
            _ => None,
        }
    }

    /// The optimal point, or `None`.
    #[must_use]
    pub fn solution(&self) -> Option<&[f64]> {
        match self {
            LpResult::Optimal { x, .. } => Some(x),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Standardisation
// ---------------------------------------------------------------------------

/// How each original variable maps into the standard-form variables.
#[derive(Debug, Clone, Copy)]
enum VarMap {
    /// `x = shift + y[index]`, with `y >= 0`.
    Shifted { index: usize, shift: f64 },
    /// `x = y[plus] - y[minus]`, both non-negative: a free variable.
    Split { plus: usize, minus: usize },
}

/// The problem rewritten as `min c'y`, `A y = b`, `y >= 0`, `b >= 0`.
struct Standard {
    /// Equality constraint matrix over the standard variables and slacks.
    a: Matrix,
    b: Vec<f64>,
    c: Vec<f64>,
    /// How to read the original variables back out.
    maps: Vec<VarMap>,
    /// For each original constraint row, the standard row it became and
    /// whether it was negated to make its right-hand side non-negative.
    row_of: Vec<(usize, bool)>,
    /// Number of structural (non-slack) standard variables.
    structural: usize,
    /// Whether the caller asked to maximise, so the reported objective and
    /// duals must be negated back.
    maximize: bool,
}

/// Rewrites a problem into `min c'y`, `A y = b`, `y >= 0`, `b >= 0`.
///
/// Three transformations, in order: a variable with a non-zero finite lower
/// bound is shifted so its lower bound is zero, a free variable is split into
/// the difference of two non-negative ones, and a finite upper bound becomes
/// an ordinary row. Then slacks and surpluses turn every inequality into an
/// equality, and any row with a negative right-hand side is negated.
///
/// A maximisation is turned into a minimisation of the negated objective, and
/// undone on the way out.
fn standardize(p: &LpProblem) -> Result<Standard, GeomError> {
    p.validate()?;
    let n = p.n();

    // Lay out the standard variables and record the mapping back.
    let mut maps = Vec::with_capacity(n);
    let mut structural = 0usize;
    for &(lo, _) in &p.bounds {
        if lo.is_infinite() {
            maps.push(VarMap::Split { plus: structural, minus: structural + 1 });
            structural += 2;
        } else {
            maps.push(VarMap::Shifted { index: structural, shift: lo });
            structural += 1;
        }
    }

    // Objective in the internal (always minimising) sense.
    let sign = if p.maximize { -1.0 } else { 1.0 };
    let mut c = vec![0.0; structural];
    for (j, &cj) in p.c.iter().enumerate() {
        match maps[j] {
            VarMap::Shifted { index, shift } => {
                c[index] = sign * cj;
                let _ = shift;
            }
            VarMap::Split { plus, minus } => {
                c[plus] = sign * cj;
                c[minus] = -sign * cj;
            }
        }
    }

    // Original rows, with the shift folded into the right-hand side, plus an
    // extra row for every finite upper bound.
    let mut rows: Vec<(Vec<f64>, f64, Cmp)> = Vec::new();
    let mut row_of = Vec::with_capacity(p.m());
    for i in 0..p.m() {
        let mut coeffs = vec![0.0; structural];
        let mut rhs = p.b[i];
        for j in 0..n {
            let aij = p.a.get(i, j);
            if aij == 0.0 {
                continue;
            }
            match maps[j] {
                VarMap::Shifted { index, shift } => {
                    coeffs[index] += aij;
                    rhs -= aij * shift;
                }
                VarMap::Split { plus, minus } => {
                    coeffs[plus] += aij;
                    coeffs[minus] -= aij;
                }
            }
        }
        row_of.push((rows.len(), false));
        rows.push((coeffs, rhs, p.constraint_types[i]));
    }
    for (j, &(lo, hi)) in p.bounds.iter().enumerate() {
        if hi.is_finite() {
            let mut coeffs = vec![0.0; structural];
            match maps[j] {
                VarMap::Shifted { index, shift } => {
                    coeffs[index] = 1.0;
                    rows.push((coeffs, hi - shift, Cmp::Le));
                }
                VarMap::Split { plus, minus } => {
                    coeffs[plus] = 1.0;
                    coeffs[minus] = -1.0;
                    rows.push((coeffs, hi, Cmp::Le));
                }
            }
            debug_assert!(lo.is_infinite() || hi >= lo);
        }
    }

    // Negate any row whose right-hand side is negative, so the identity basis
    // of phase one starts feasible.
    for (i, row) in rows.iter_mut().enumerate() {
        if row.1 < 0.0 {
            for v in &mut row.0 {
                *v = -*v;
            }
            row.1 = -row.1;
            row.2 = row.2.flipped();
            if let Some(entry) = row_of.iter_mut().find(|e| e.0 == i) {
                entry.1 = true;
            }
        }
    }

    // Slack for <=, surplus for >=, nothing for =.
    let extra = rows.iter().filter(|r| r.2 != Cmp::Eq).count();
    let total = structural + extra;
    let m = rows.len();
    let mut a = Matrix::zeros(m, total);
    let mut b = vec![0.0; m];
    let mut next_slack = structural;
    for (i, (coeffs, rhs, cmp)) in rows.iter().enumerate() {
        for (j, &v) in coeffs.iter().enumerate() {
            a.set(i, j, v);
        }
        b[i] = *rhs;
        match cmp {
            Cmp::Le => {
                a.set(i, next_slack, 1.0);
                next_slack += 1;
            }
            Cmp::Ge => {
                a.set(i, next_slack, -1.0);
                next_slack += 1;
            }
            Cmp::Eq => {}
        }
    }
    c.resize(total, 0.0);

    Ok(Standard { a, b, c, maps, row_of, structural, maximize: p.maximize })
}

// ---------------------------------------------------------------------------
// The simplex method
// ---------------------------------------------------------------------------

/// A simplex tableau: the constraint rows, the objective row, and the basis.
struct Tableau {
    /// `m` rows by `n + 1` columns; the last column is the right-hand side.
    t: Vec<Vec<f64>>,
    /// Objective row of the same width; the last entry is minus the objective.
    z: Vec<f64>,
    /// Column index basic in each row.
    basis: Vec<usize>,
    m: usize,
    n: usize,
}

impl Tableau {
    /// Pivots on `(row, col)`, making that column a unit vector.
    fn pivot(&mut self, row: usize, col: usize) {
        let p = self.t[row][col];
        debug_assert!(p.abs() > PIVOT_TOL);
        for v in &mut self.t[row] {
            *v /= p;
        }
        for r in 0..self.m {
            if r == row {
                continue;
            }
            let factor = self.t[r][col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..=self.n {
                self.t[r][k] -= factor * self.t[row][k];
            }
        }
        let factor = self.z[col];
        if factor != 0.0 {
            for k in 0..=self.n {
                self.z[k] -= factor * self.t[row][k];
            }
        }
        self.basis[row] = col;
    }

    /// Runs simplex to optimality over the columns in `allowed`.
    ///
    /// Bland's rule chooses the lowest-indexed improving column and breaks
    /// ratio ties by the lowest-indexed basic variable. That is provably
    /// non-cycling: the basis sequence is lexicographically monotone, so no
    /// basis can repeat. Faster rules -- steepest edge, Dantzig -- can revisit
    /// a basis forever on a degenerate problem, which is not a hypothetical:
    /// Beale's example cycles under Dantzig's rule in six pivots.
    ///
    /// Returns `false` if the objective is unbounded below.
    fn solve(&mut self, allowed: &dyn Fn(usize) -> bool) -> bool {
        for _ in 0..MAX_PIVOTS {
            // Entering: lowest index with a negative reduced cost.
            let mut entering = None;
            for j in 0..self.n {
                if allowed(j) && self.z[j] < -OPT_TOL {
                    entering = Some(j);
                    break;
                }
            }
            let Some(col) = entering else { return true };

            // Leaving: minimum ratio, ties to the lowest basic index.
            let mut best: Option<(f64, usize, usize)> = None;
            for r in 0..self.m {
                let a = self.t[r][col];
                if a <= PIVOT_TOL {
                    continue;
                }
                let ratio = self.t[r][self.n] / a;
                let candidate = (ratio, self.basis[r], r);
                best = match best {
                    None => Some(candidate),
                    Some(current) => {
                        if ratio < current.0 - PIVOT_TOL
                            || (ratio < current.0 + PIVOT_TOL && self.basis[r] < current.1)
                        {
                            Some(candidate)
                        } else {
                            Some(current)
                        }
                    }
                };
            }
            let Some((_, _, row)) = best else {
                // No row limits the increase: the objective falls forever.
                return false;
            };
            self.pivot(row, col);
        }
        true
    }
}

/// Solves a linear program by the two-phase simplex method.
///
/// Phase one minimises the total artificial infeasibility from an
/// artificial-variable basis; a positive optimum there proves the problem
/// infeasible, since that value is the least total violation achievable.
/// Phase two then optimises the real objective from the feasible basis phase
/// one produced.
///
/// Bland's rule is used throughout, so the method terminates on any problem,
/// including degenerate ones where a faster pivoting rule would cycle.
///
/// # Errors
/// Returns an error if the problem's parts disagree in shape.
pub fn simplex(p: &LpProblem) -> Result<LpResult, GeomError> {
    Ok(solve_tableau(p)?.1)
}

/// Solves and returns the final tableau alongside the result, so that
/// sensitivity analysis can read the optimal basis rather than re-deriving it.
fn solve_tableau(p: &LpProblem) -> Result<(Option<(Tableau, Standard)>, LpResult), GeomError> {
    let s = standardize(p)?;
    let m = s.b.len();
    let n = s.c.len();
    if m == 0 {
        // No constraints at all: the objective is unbounded unless every
        // coefficient is zero, since variables are only bounded below.
        if s.c.iter().any(|&v| v < -OPT_TOL) {
            return Ok((None, LpResult::Unbounded));
        }
        let x = vec![0.0; p.n()];
        let objective = p.objective_at(&x);
        return Ok((
            None,
            LpResult::Optimal { x, objective, duals: Vec::new(), reduced_costs: p.c.clone() },
        ));
    }

    // Phase one: artificial variables form the starting basis.
    let width = n + m;
    let mut t = vec![vec![0.0; width + 1]; m];
    for i in 0..m {
        for j in 0..n {
            t[i][j] = s.a.get(i, j);
        }
        t[i][n + i] = 1.0;
        t[i][width] = s.b[i];
    }
    // Minimising the artificial sum; its reduced-cost row is minus the sum of
    // the constraint rows over the real columns.
    let mut z = vec![0.0; width + 1];
    for (j, entry) in z.iter_mut().enumerate().take(n) {
        *entry = -(0..m).map(|i| t[i][j]).sum::<f64>();
    }
    z[width] = -(0..m).map(|i| t[i][width]).sum::<f64>();

    let mut tab = Tableau { t, z, basis: (n..n + m).collect(), m, n: width };
    let real = |j: usize| j < n;
    let all = |_: usize| true;
    tab.solve(&all);
    if -tab.z[width] > 1e-7 {
        return Ok((None, LpResult::Infeasible));
    }

    // Drive any artificial still basic out of the basis. A row that cannot be
    // pivoted has no independent real column left in it, so it is redundant
    // and can be left with its artificial at zero.
    for r in 0..m {
        if tab.basis[r] >= n {
            let replacement = (0..n).find(|&j| tab.t[r][j].abs() > PIVOT_TOL);
            if let Some(col) = replacement {
                tab.pivot(r, col);
            }
        }
    }

    // Phase two: the real objective, with artificial columns barred.
    let mut z = vec![0.0; width + 1];
    z[..n].copy_from_slice(&s.c[..n]);
    for r in 0..m {
        let col = tab.basis[r];
        if col < n && s.c[col] != 0.0 {
            let factor = z[col];
            if factor != 0.0 {
                for k in 0..=width {
                    z[k] -= factor * tab.t[r][k];
                }
            }
        }
    }
    tab.z = z;
    if !tab.solve(&real) {
        return Ok((None, LpResult::Unbounded));
    }

    let result = extract(&tab, &s, p, n);
    Ok((Some((tab, s)), result))
}

/// Reads a solution, duals and reduced costs out of an optimal tableau.
fn extract(tab: &Tableau, s: &Standard, p: &LpProblem, n: usize) -> LpResult {
    let m = s.b.len();
    let mut y = vec![0.0; n];
    for r in 0..m {
        if tab.basis[r] < n {
            y[tab.basis[r]] = tab.t[r][tab.n];
        }
    }

    // Map back to the caller's variables.
    let mut x = vec![0.0; p.n()];
    for (j, map) in s.maps.iter().enumerate() {
        x[j] = match *map {
            VarMap::Shifted { index, shift } => shift + y[index],
            VarMap::Split { plus, minus } => y[plus] - y[minus],
        };
    }
    let objective = p.objective_at(&x);

    // The dual of a standard row is minus the objective-row entry under the
    // column that was basic there at the start of phase two -- for a row with
    // a slack, that is the slack column. `standardize` appends slacks in row
    // order, so counting rows with slacks recovers the column.
    let mut slack_of = vec![None; m];
    let mut next = s.structural;
    for (i, ct) in row_senses(s).iter().enumerate() {
        if *ct != Cmp::Eq {
            slack_of[i] = Some(next);
            next += 1;
        }
    }

    let sign = if s.maximize { -1.0 } else { 1.0 };
    let mut duals = vec![0.0; p.m()];
    for (i, &(row, negated)) in s.row_of.iter().enumerate() {
        let raw = match slack_of[row] {
            // For a <= row the slack enters with +1 and for a >= row with -1,
            // which is why the two read off with opposite signs.
            Some(col) => {
                let sense = row_senses(s)[row];
                match sense {
                    Cmp::Le => -tab.z[col],
                    Cmp::Ge => tab.z[col],
                    Cmp::Eq => 0.0,
                }
            }
            // An equality row has no slack column; recover its dual from the
            // artificial column that started basic there, which phase two
            // leaves carrying exactly the same information.
            None => -tab.z[s.structural + slack_count(s) + row],
        };
        // A negated row had its right-hand side sign flipped, so the
        // derivative with respect to the original b flips with it.
        let oriented = if negated { -raw } else { raw };
        duals[i] = sign * oriented;
    }

    let reduced_costs = (0..p.n())
        .map(|j| {
            p.c[j] - (0..p.m()).map(|i| duals[i] * p.a.get(i, j)).sum::<f64>()
        })
        .collect();

    LpResult::Optimal { x, objective, duals, reduced_costs }
}

/// The sense of each standard row, recovered from its slack coefficient.
fn row_senses(s: &Standard) -> Vec<Cmp> {
    let m = s.b.len();
    let mut out = vec![Cmp::Eq; m];
    let mut next = s.structural;
    for (i, entry) in out.iter_mut().enumerate() {
        if next < s.a.cols {
            let v = s.a.get(i, next);
            if v == 1.0 {
                *entry = Cmp::Le;
                next += 1;
                continue;
            } else if v == -1.0 {
                *entry = Cmp::Ge;
                next += 1;
                continue;
            }
        }
        *entry = Cmp::Eq;
    }
    out
}

/// How many standard rows carry a slack or surplus column.
fn slack_count(s: &Standard) -> usize {
    s.a.cols - s.structural
}

// ---------------------------------------------------------------------------
// Duality
// ---------------------------------------------------------------------------

/// The dual linear program.
///
/// For a minimisation `min c'x` subject to rows compared against `b` with
/// `x >= 0`, the dual is `max b'y` subject to `A'y <= c`, with each `y_i`
/// signed by the sense of its row: non-positive for a `<=` row, non-negative
/// for a `>=` row, free for an equality. Maximisation mirrors it.
///
/// Solving the dual gives the same optimal value as the primal and its
/// solution is the primal's vector of shadow prices, which is the practical
/// content of duality: the answer to "what is this constraint costing me" is
/// a solution to a different linear program of the same size.
///
/// # Errors
/// Returns an error unless every primal variable carries the default bounds
/// `(0, inf)`. A bounded variable contributes an extra dual row, which would
/// change the problem's shape rather than transpose it.
pub fn lp_dual(p: &LpProblem) -> Result<LpProblem, GeomError> {
    p.validate()?;
    if p.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi.is_finite()) {
        return Err(GeomError::InvalidArgument(
            "lp_dual requires the default non-negative variable bounds",
        ));
    }
    let (m, n) = (p.m(), p.n());
    let mut a = Matrix::zeros(n, m);
    for i in 0..m {
        for j in 0..n {
            a.set(j, i, p.a.get(i, j));
        }
    }
    // Minimising primal gives a maximising dual with `<=` rows, and the
    // reverse; a variable's sign follows from which direction relaxing its
    // row can help.
    let (sense, bounds): (Cmp, Vec<(f64, f64)>) = if p.maximize {
        (
            Cmp::Ge,
            p.constraint_types
                .iter()
                .map(|c| match c {
                    Cmp::Le => (0.0, f64::INFINITY),
                    Cmp::Ge => (f64::NEG_INFINITY, 0.0),
                    Cmp::Eq => (f64::NEG_INFINITY, f64::INFINITY),
                })
                .collect(),
        )
    } else {
        (
            Cmp::Le,
            p.constraint_types
                .iter()
                .map(|c| match c {
                    Cmp::Le => (f64::NEG_INFINITY, 0.0),
                    Cmp::Ge => (0.0, f64::INFINITY),
                    Cmp::Eq => (f64::NEG_INFINITY, f64::INFINITY),
                })
                .collect(),
        )
    };

    Ok(LpProblem {
        c: p.b.clone(),
        a,
        b: p.c.clone(),
        constraint_types: vec![sense; n],
        bounds,
        maximize: !p.maximize,
    })
}

// ---------------------------------------------------------------------------
// Sensitivity analysis
// ---------------------------------------------------------------------------

/// Ranges over which the optimal basis survives, as
/// `(objective coefficient ranges, right-hand side ranges)`.
///
/// Inside a right-hand side's range the shadow price is constant, so the
/// objective moves by exactly `duals[i]` per unit of `b[i]`. That linearity
/// is the point of the exercise and is what the tests check; outside the
/// range the basis changes and the rate does too.
///
/// Inside an objective coefficient's range the optimal *point* does not move
/// at all, only the value.
///
/// # Errors
/// Returns an error if the problem is not solved to an optimum, or if any
/// variable carries non-default bounds -- a finite upper bound becomes an
/// extra row during standardisation, and the ranges would then be reported
/// against rows the caller never wrote.
pub fn sensitivity_ranges(
    p: &LpProblem,
) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>), GeomError> {
    if p.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi.is_finite()) {
        return Err(GeomError::InvalidArgument(
            "sensitivity_ranges requires the default non-negative variable bounds",
        ));
    }
    let (solved, result) = solve_tableau(p)?;
    let LpResult::Optimal { .. } = result else {
        return Err(GeomError::Degenerate("sensitivity_ranges requires an optimal solution"));
    };
    let Some((tab, s)) = solved else {
        return Err(GeomError::Degenerate("sensitivity_ranges requires a constrained problem"));
    };

    let m = s.b.len();
    let structural = s.structural;
    let senses = row_senses(&s);
    let mut slack_of = vec![None; m];
    let mut next = structural;
    for (i, sense) in senses.iter().enumerate() {
        if *sense != Cmp::Eq {
            slack_of[i] = Some(next);
            next += 1;
        }
    }

    // Right-hand side ranges. Raising b_i by delta moves the basic solution by
    // delta times the i-th column of B inverse, which the tableau carries
    // under that row's slack column (with a sign set by the row's sense).
    let mut b_ranges = Vec::with_capacity(p.m());
    for (i, &(row, negated)) in s.row_of.iter().enumerate() {
        let Some(col) = slack_of[row] else {
            // An equality row cannot be relaxed without changing the basis.
            b_ranges.push((p.b[i], p.b[i]));
            continue;
        };
        let orientation = match senses[row] {
            Cmp::Le => 1.0,
            Cmp::Ge => -1.0,
            Cmp::Eq => 0.0,
        } * if negated { -1.0 } else { 1.0 };

        let (mut down, mut up) = (f64::NEG_INFINITY, f64::INFINITY);
        for r in 0..m {
            let direction = orientation * tab.t[r][col];
            if direction.abs() < PIVOT_TOL {
                continue;
            }
            // The basic value in row r is t[r][rhs]; it must stay non-negative.
            let limit = -tab.t[r][tab.n] / direction;
            if direction > 0.0 {
                down = down.max(limit);
            } else {
                up = up.min(limit);
            }
        }
        b_ranges.push((p.b[i] + down, p.b[i] + up));
    }

    // Objective coefficient ranges. A non-basic column may have its cost
    // lowered until its reduced cost reaches zero; a basic one is limited by
    // the ratios along its row.
    let sign = if s.maximize { -1.0 } else { 1.0 };
    let mut c_ranges = Vec::with_capacity(p.n());
    for j in 0..p.n() {
        let VarMap::Shifted { index, .. } = s.maps[j] else {
            // A free variable is two standard columns at once; its range is
            // not a single interval in this basis.
            c_ranges.push((f64::NEG_INFINITY, f64::INFINITY));
            continue;
        };
        let basic_row = (0..m).find(|&r| tab.basis[r] == index);
        let (down, up) = match basic_row {
            None => {
                // Non-basic: the reduced cost must stay non-negative, so the
                // cost may rise without limit and fall by its reduced cost.
                (-tab.z[index], f64::INFINITY)
            }
            Some(r) => {
                let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
                for k in 0..tab.n {
                    let a = tab.t[r][k];
                    if a.abs() < PIVOT_TOL || tab.basis.contains(&k) {
                        continue;
                    }
                    let ratio = tab.z[k] / a;
                    if a > 0.0 {
                        hi = hi.min(ratio);
                    } else {
                        lo = lo.max(ratio);
                    }
                }
                (lo, hi)
            }
        };
        // The internal problem always minimises, so a maximisation's ranges
        // come back mirrored.
        let (a, b) = (p.c[j] + sign * down, p.c[j] + sign * up);
        c_ranges.push((a.min(b), a.max(b)));
    }

    Ok((c_ranges, b_ranges))
}

// ---------------------------------------------------------------------------
// The dual simplex
// ---------------------------------------------------------------------------

/// The dual simplex method, started from a given basis.
///
/// Where the primal simplex keeps every basic variable non-negative and works
/// toward optimality, the dual simplex keeps the reduced costs optimal and
/// works toward feasibility. That is the right way round after a right-hand
/// side changes -- the old basis stays dual-feasible while becoming primal
/// infeasible, so re-solving costs a few pivots instead of a fresh start.
///
/// `basis` names one standard-form column per constraint row. Column indices
/// run over the structural variables first, then the slack and surplus
/// columns in row order.
///
/// # Errors
/// Returns an error if the basis has the wrong length, names a column out of
/// range, or is singular. A basis that is not dual-feasible is reported as
/// [`GeomError::Degenerate`] rather than silently repaired.
pub fn dual_simplex(p: &LpProblem, basis: &[usize]) -> Result<LpResult, GeomError> {
    let s = standardize(p)?;
    let m = s.b.len();
    let n = s.c.len();
    if basis.len() != m {
        return Err(GeomError::InvalidArgument("dual_simplex needs one basic column per row"));
    }
    if basis.iter().any(|&j| j >= n) {
        return Err(GeomError::InvalidArgument("dual_simplex basis names a column out of range"));
    }

    // Build the tableau and pivot the named columns into the basis.
    let mut t = vec![vec![0.0; n + 1]; m];
    for i in 0..m {
        for j in 0..n {
            t[i][j] = s.a.get(i, j);
        }
        t[i][n] = s.b[i];
    }
    let mut tab = Tableau { t, z: vec![0.0; n + 1], basis: vec![usize::MAX; m], m, n };
    for (r, &col) in basis.iter().enumerate() {
        if tab.t[r][col].abs() < PIVOT_TOL {
            // Try to find another row that can supply this column.
            let swap = (r + 1..m).find(|&k| tab.t[k][col].abs() > PIVOT_TOL);
            let Some(k) = swap else {
                return Err(GeomError::Degenerate("dual_simplex basis is singular"));
            };
            tab.t.swap(r, k);
        }
        tab.pivot(r, col);
    }

    // Price out the objective row against the basis.
    let mut z = s.c.clone();
    z.push(0.0);
    for r in 0..m {
        let factor = z[tab.basis[r]];
        if factor != 0.0 {
            for k in 0..=n {
                z[k] -= factor * tab.t[r][k];
            }
        }
    }
    tab.z = z;
    if tab.z[..n].iter().any(|&v| v < -OPT_TOL) {
        return Err(GeomError::Degenerate("dual_simplex requires a dual-feasible basis"));
    }

    for _ in 0..MAX_PIVOTS {
        // Leaving: the most negative basic value, ties to the lowest index.
        let mut leaving: Option<usize> = None;
        for r in 0..m {
            if tab.t[r][n] < -PIVOT_TOL
                && leaving.is_none_or(|best| tab.t[r][n] < tab.t[best][n])
            {
                leaving = Some(r);
            }
        }
        let Some(row) = leaving else {
            let result = extract(&tab, &s, p, n);
            return Ok(result);
        };

        // Entering: the ratio test runs along the row, over columns that would
        // move the infeasible basic value upward.
        let mut entering: Option<(f64, usize)> = None;
        for j in 0..n {
            let a = tab.t[row][j];
            if a >= -PIVOT_TOL {
                continue;
            }
            let ratio = tab.z[j] / -a;
            if entering.is_none_or(|(best, _)| ratio < best) {
                entering = Some((ratio, j));
            }
        }
        let Some((_, col)) = entering else {
            // No column can restore feasibility in this row: the primal is
            // infeasible, which is the dual being unbounded.
            return Ok(LpResult::Infeasible);
        };
        tab.pivot(row, col);
    }
    Err(GeomError::Degenerate("dual_simplex did not terminate"))
}

// ---------------------------------------------------------------------------
// Interior point
// ---------------------------------------------------------------------------

/// Number of Newton steps the path-following method is allowed.
const IP_MAX_ITER: usize = 200;
/// How far along a Newton step to go before hitting the boundary.
const IP_STEP_FRACTION: f64 = 0.995;
/// Centring parameter: the fraction of the current duality measure aimed at.
const IP_SIGMA: f64 = 0.2;

/// Solves a linear program by a primal-dual path-following interior point
/// method.
///
/// The method keeps `x > 0` and `s > 0` strictly, and drives the duality
/// measure `x's/n` toward zero along the central path. Each iteration solves
/// one Newton system, reduced to the normal equations `A D A' dy = r` with
/// `D = diag(x_i / s_i)` and factored by Cholesky. Unlike the simplex method
/// it never lands exactly on a vertex, and unlike the simplex method its
/// iteration count barely grows with the size of the problem.
///
/// The starting point is deliberately infeasible -- all ones -- and the primal
/// and dual residuals are driven to zero alongside the duality gap. That
/// avoids needing a phase one, but means infeasibility shows up as a failure
/// to converge rather than as a proof, so an unconverged run is reported as
/// [`LpResult::Infeasible`] only when the residuals are still large while the
/// gap has closed.
///
/// # Errors
/// Returns an error if the problem's parts disagree in shape or `tol` is not
/// positive.
pub fn interior_point(p: &LpProblem, tol: f64) -> Result<LpResult, GeomError> {
    if !(tol > 0.0) {
        return Err(GeomError::InvalidArgument("interior_point requires tol > 0"));
    }
    let s = standardize(p)?;
    let m = s.b.len();
    let n = s.c.len();
    if m == 0 {
        return simplex(p);
    }

    let mut x = vec![1.0; n];
    let mut slack = vec![1.0; n];
    let mut y = vec![0.0; m];

    let mut converged = false;
    for _ in 0..IP_MAX_ITER {
        // Residuals: primal feasibility, dual feasibility, complementarity.
        let ax: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| s.a.get(i, j) * x[j]).sum::<f64>())
            .collect();
        let r_p: Vec<f64> = (0..m).map(|i| s.b[i] - ax[i]).collect();
        let r_d: Vec<f64> = (0..n)
            .map(|j| {
                s.c[j] - (0..m).map(|i| s.a.get(i, j) * y[i]).sum::<f64>() - slack[j]
            })
            .collect();
        let mu: f64 = x.iter().zip(&slack).map(|(a, b)| a * b).sum::<f64>() / n as f64;

        let primal_err = r_p.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let dual_err = r_d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if primal_err < tol && dual_err < tol && mu < tol {
            converged = true;
            break;
        }

        // Normal equations: (A D A') dy = b - sigma mu A S^-1 e + A D r_d,
        // with D = diag(x_i / s_i).
        let d: Vec<f64> = (0..n).map(|j| x[j] / slack[j].max(1e-300)).collect();
        let mut normal = Matrix::zeros(m, m);
        for i in 0..m {
            for k in i..m {
                let v: f64 =
                    (0..n).map(|j| s.a.get(i, j) * d[j] * s.a.get(k, j)).sum();
                normal.set(i, k, v);
                normal.set(k, i, v);
            }
            // A touch of regularisation: a redundant constraint row makes the
            // normal matrix singular, and the answer is unaffected by a
            // perturbation this small.
            let diagonal = normal.get(i, i);
            normal.set(i, i, diagonal + 1e-12 * (1.0 + diagonal));
        }
        let rhs: Vec<f64> = (0..m)
            .map(|i| {
                s.b[i]
                    - IP_SIGMA
                        * mu
                        * (0..n).map(|j| s.a.get(i, j) / slack[j].max(1e-300)).sum::<f64>()
                    + (0..n).map(|j| s.a.get(i, j) * d[j] * r_d[j]).sum::<f64>()
            })
            .collect();

        let Ok(factor) = crate::linalg::cholesky::cholesky(&normal) else {
            break;
        };
        let Ok(dy) = crate::linalg::cholesky::cholesky_solve(&factor, &rhs) else {
            break;
        };

        let ds: Vec<f64> = (0..n)
            .map(|j| r_d[j] - (0..m).map(|i| s.a.get(i, j) * dy[i]).sum::<f64>())
            .collect();
        let dx: Vec<f64> = (0..n)
            .map(|j| IP_SIGMA * mu / slack[j].max(1e-300) - x[j] - d[j] * ds[j])
            .collect();
        if dx.iter().chain(&ds).chain(&dy).any(|v| !v.is_finite()) {
            break;
        }

        // Step to just short of the boundary, separately in each space.
        let step = |v: &[f64], dv: &[f64]| -> f64 {
            let mut alpha = 1.0f64;
            for (a, b) in v.iter().zip(dv) {
                if *b < 0.0 {
                    alpha = alpha.min(-a / b);
                }
            }
            (IP_STEP_FRACTION * alpha).min(1.0)
        };
        let alpha_p = step(&x, &dx);
        let alpha_d = step(&slack, &ds);
        for j in 0..n {
            x[j] = (x[j] + alpha_p * dx[j]).max(1e-300);
            slack[j] = (slack[j] + alpha_d * ds[j]).max(1e-300);
        }
        for i in 0..m {
            y[i] += alpha_d * dy[i];
        }
    }

    if !converged {
        // Either the problem has no solution or the method stalled; the
        // simplex method decides which, since it terminates on any input.
        return simplex(p);
    }

    // Map back to the caller's variables.
    let mut out = vec![0.0; p.n()];
    for (j, map) in s.maps.iter().enumerate() {
        out[j] = match *map {
            VarMap::Shifted { index, shift } => shift + x[index],
            VarMap::Split { plus, minus } => x[plus] - x[minus],
        };
    }
    let objective = p.objective_at(&out);

    let sign = if s.maximize { -1.0 } else { 1.0 };
    let mut duals = vec![0.0; p.m()];
    for (i, &(row, negated)) in s.row_of.iter().enumerate() {
        let raw = y[row];
        duals[i] = sign * if negated { -raw } else { raw };
    }
    let reduced_costs = (0..p.n())
        .map(|j| p.c[j] - (0..p.m()).map(|i| duals[i] * p.a.get(i, j)).sum::<f64>())
        .collect();

    Ok(LpResult::Optimal { x: out, objective, duals, reduced_costs })
}

// ---------------------------------------------------------------------------
// A small modelling language
// ---------------------------------------------------------------------------

/// Parses a linear program from text.
///
/// The grammar is deliberately tiny:
///
/// ```text
/// max 3x + 5y
/// subject to
///   x <= 4
///   2y <= 12
///   3x + 2y <= 18
/// bounds
///   y >= 1
///   free z
/// ```
///
/// The first line gives the sense and the objective. Everything after
/// `subject to` (or `st`, or `s.t.`) is a constraint row until an optional
/// `bounds` section, where single-variable lines set bounds rather than adding
/// rows and `free x` removes a variable's lower bound. Blank lines and `#`
/// comments are ignored, coefficients may be omitted, and variables are
/// numbered in order of first appearance.
///
/// # Errors
/// Returns [`GeomError::InvalidArgument`] naming the first thing that could
/// not be read.
pub fn lp_from_str(text: &str) -> Result<LpProblem, GeomError> {
    #[derive(PartialEq)]
    enum Section {
        Objective,
        Constraints,
        Bounds,
    }

    let mut names: Vec<String> = Vec::new();
    let mut maximize = false;
    let mut objective: Vec<(usize, f64)> = Vec::new();
    let mut rows: Vec<(Vec<(usize, f64)>, Cmp, f64)> = Vec::new();
    let mut bound_lines: Vec<(usize, Cmp, f64)> = Vec::new();
    let mut free: Vec<usize> = Vec::new();
    let mut section = Section::Objective;

    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower == "subject to" || lower == "st" || lower == "s.t." || lower == "such that" {
            section = Section::Constraints;
            continue;
        }
        if lower == "bounds" {
            section = Section::Bounds;
            continue;
        }

        match section {
            Section::Objective => {
                let rest = if let Some(r) = lower.strip_prefix("max") {
                    maximize = true;
                    &line[line.len() - r.len()..]
                } else if let Some(r) = lower.strip_prefix("min") {
                    &line[line.len() - r.len()..]
                } else {
                    return Err(GeomError::InvalidArgument(
                        "the first line must start with max or min",
                    ));
                };
                objective = parse_terms(rest, &mut names)?;
                section = Section::Constraints;
            }
            Section::Constraints => {
                let (terms, cmp, rhs) = parse_row(line, &mut names)?;
                rows.push((terms, cmp, rhs));
            }
            Section::Bounds => {
                if let Some(rest) = lower.strip_prefix("free ") {
                    let name = rest.trim().to_string();
                    let idx = index_of(&name, &mut names);
                    free.push(idx);
                    continue;
                }
                let (terms, cmp, rhs) = parse_row(line, &mut names)?;
                if terms.len() != 1 || (terms[0].1 - 1.0).abs() > 1e-12 {
                    return Err(GeomError::InvalidArgument(
                        "a bounds line must name a single variable with coefficient one",
                    ));
                }
                bound_lines.push((terms[0].0, cmp, rhs));
            }
        }
    }

    let n = names.len();
    if n == 0 {
        return Err(GeomError::InvalidArgument("the model names no variables"));
    }
    let mut c = vec![0.0; n];
    for (j, v) in objective {
        c[j] += v;
    }
    let m = rows.len();
    let mut a = Matrix::zeros(m, n);
    let mut b = vec![0.0; m];
    let mut constraint_types = Vec::with_capacity(m);
    for (i, (terms, cmp, rhs)) in rows.into_iter().enumerate() {
        for (j, v) in terms {
            a.set(i, j, a.get(i, j) + v);
        }
        b[i] = rhs;
        constraint_types.push(cmp);
    }

    let mut bounds = vec![(0.0, f64::INFINITY); n];
    for j in free {
        bounds[j].0 = f64::NEG_INFINITY;
    }
    for (j, cmp, value) in bound_lines {
        match cmp {
            Cmp::Ge => bounds[j].0 = value,
            Cmp::Le => bounds[j].1 = value,
            Cmp::Eq => bounds[j] = (value, value),
        }
    }

    let p = LpProblem { c, a, b, constraint_types, bounds, maximize };
    p.validate()?;
    Ok(p)
}

/// The index of a variable name, appending it if new.
fn index_of(name: &str, names: &mut Vec<String>) -> usize {
    if let Some(i) = names.iter().position(|n| n == name) {
        return i;
    }
    names.push(name.to_string());
    names.len() - 1
}

/// Parses `3x + 2y - z` into `(index, coefficient)` pairs.
fn parse_terms(text: &str, names: &mut Vec<String>) -> Result<Vec<(usize, f64)>, GeomError> {
    let mut out = Vec::new();
    // Normalise so every term carries its own sign, then split on spaces.
    let spaced = text.replace('+', " + ").replace('-', " - ");
    let tokens: Vec<&str> = spaced.split_whitespace().collect();
    let mut sign = 1.0f64;
    let mut i = 0usize;
    while i < tokens.len() {
        match tokens[i] {
            "+" => {
                sign = 1.0;
                i += 1;
            }
            "-" => {
                sign = -1.0;
                i += 1;
            }
            token => {
                // A term is an optional number followed by an optional name,
                // possibly separated by a `*`.
                let body = token.trim_start_matches('*');
                let split = body.find(|ch: char| ch.is_alphabetic() || ch == '_');
                let (coefficient, name) = match split {
                    Some(0) => (1.0, body),
                    Some(k) => {
                        let head = body[..k].trim_end_matches('*');
                        let value: f64 = head
                            .parse()
                            .map_err(|_| GeomError::InvalidArgument("bad coefficient"))?;
                        (value, &body[k..])
                    }
                    None => {
                        return Err(GeomError::InvalidArgument(
                            "a term with no variable appeared on the left-hand side",
                        ))
                    }
                };
                out.push((index_of(name, names), sign * coefficient));
                sign = 1.0;
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Parses `3x + 2y <= 18` into terms, a sense, and a right-hand side.
fn parse_row(
    line: &str,
    names: &mut Vec<String>,
) -> Result<(Vec<(usize, f64)>, Cmp, f64), GeomError> {
    for (token, cmp) in [("<=", Cmp::Le), (">=", Cmp::Ge), ("=<", Cmp::Le), ("=>", Cmp::Ge)] {
        if let Some(k) = line.find(token) {
            let rhs: f64 = line[k + token.len()..]
                .trim()
                .parse()
                .map_err(|_| GeomError::InvalidArgument("bad right-hand side"))?;
            return Ok((parse_terms(&line[..k], names)?, cmp, rhs));
        }
    }
    if let Some(k) = line.find('=') {
        let rhs: f64 = line[k + 1..]
            .trim()
            .parse()
            .map_err(|_| GeomError::InvalidArgument("bad right-hand side"))?;
        return Ok((parse_terms(&line[..k], names)?, Cmp::Eq, rhs));
    }
    Err(GeomError::InvalidArgument("a constraint line needs a comparison operator"))
}

// ---------------------------------------------------------------------------
// Classical models
// ---------------------------------------------------------------------------

/// Stigler's diet problem: the cheapest combination of foods meeting every
/// nutritional minimum.
///
/// `costs` gives the price per unit of each food, `nutrients` holds the amount
/// of nutrient `k` in one unit of food `j` at `(k, j)`, and `requirements`
/// the minimum of each nutrient.
///
/// # Errors
/// Returns an error if the shapes disagree.
pub fn diet_problem(
    costs: &[f64],
    nutrients: &Matrix,
    requirements: &[f64],
) -> Result<LpProblem, GeomError> {
    if nutrients.cols != costs.len() || nutrients.rows != requirements.len() {
        return Err(GeomError::InvalidArgument("diet_problem: shape mismatch"));
    }
    let p = LpProblem {
        c: costs.to_vec(),
        a: nutrients.clone(),
        b: requirements.to_vec(),
        constraint_types: vec![Cmp::Ge; requirements.len()],
        bounds: vec![(0.0, f64::INFINITY); costs.len()],
        maximize: false,
    };
    p.validate()?;
    Ok(p)
}

/// A production plan: how much of each product to make to maximise profit
/// under resource limits.
///
/// `usage` holds the amount of resource `k` consumed per unit of product `j`
/// at `(k, j)`, and `available` the stock of each resource.
///
/// # Errors
/// Returns an error if the shapes disagree.
pub fn production_planning(
    profits: &[f64],
    usage: &Matrix,
    available: &[f64],
) -> Result<LpProblem, GeomError> {
    if usage.cols != profits.len() || usage.rows != available.len() {
        return Err(GeomError::InvalidArgument("production_planning: shape mismatch"));
    }
    let p = LpProblem {
        c: profits.to_vec(),
        a: usage.clone(),
        b: available.to_vec(),
        constraint_types: vec![Cmp::Le; available.len()],
        bounds: vec![(0.0, f64::INFINITY); profits.len()],
        maximize: true,
    };
    p.validate()?;
    Ok(p)
}

/// The transportation problem: ship from sources to sinks at least cost.
///
/// `costs` holds the unit cost from source `i` to sink `j` at `(i, j)`.
/// Supply is an upper limit and demand a lower one, so unbalanced instances
/// are handled without inventing a dummy row.
///
/// The constraint matrix is totally unimodular, so with integer supplies and
/// demands the simplex optimum is automatically integral -- no branch and
/// bound is needed, which is why the problem is solved as a linear program at
/// all.
///
/// # Errors
/// Returns an error if the shapes disagree or total demand exceeds total
/// supply, which is infeasible by inspection.
pub fn transportation_problem(
    supply: &[f64],
    demand: &[f64],
    costs: &Matrix,
) -> Result<LpResult, GeomError> {
    let (m, n) = (supply.len(), demand.len());
    if costs.rows != m || costs.cols != n || m == 0 || n == 0 {
        return Err(GeomError::InvalidArgument("transportation_problem: shape mismatch"));
    }
    if supply.iter().chain(demand).any(|&v| v < 0.0) {
        return Err(GeomError::InvalidArgument("supply and demand must be non-negative"));
    }
    if demand.iter().sum::<f64>() > supply.iter().sum::<f64>() + OPT_TOL {
        return Ok(LpResult::Infeasible);
    }

    let vars = m * n;
    let mut a = Matrix::zeros(m + n, vars);
    let mut b = vec![0.0; m + n];
    let mut senses = Vec::with_capacity(m + n);
    for i in 0..m {
        for j in 0..n {
            a.set(i, i * n + j, 1.0);
        }
        b[i] = supply[i];
        senses.push(Cmp::Le);
    }
    for j in 0..n {
        for i in 0..m {
            a.set(m + j, i * n + j, 1.0);
        }
        b[m + j] = demand[j];
        senses.push(Cmp::Ge);
    }

    let c: Vec<f64> = (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| costs.get(i, j)).collect();
    let p = LpProblem {
        c,
        a,
        b,
        constraint_types: senses,
        bounds: vec![(0.0, f64::INFINITY); vars],
        maximize: false,
    };
    simplex(&p)
}

/// Solves a two-player zero-sum game, returning
/// `(row strategy, column strategy, value)`.
///
/// `payoff` holds the row player's gain at `(i, j)`. The row player maximises
/// the worst case and the column player minimises the best case, and von
/// Neumann's minimax theorem says the two coincide -- which here is not an
/// extra assumption but a consequence of LP duality, since the two players'
/// programs are duals of each other. The column strategy is read directly off
/// the row program's shadow prices.
///
/// The payoff is shifted to be strictly positive before solving, since the
/// standard formulation divides by the value; the shift is undone on the way
/// out.
///
/// # Errors
/// Returns an error if the resulting program has no optimum, which cannot
/// happen for a finite game and would indicate a numerical failure.
pub fn two_player_zero_sum_lp(payoff: &Matrix) -> Result<(Vec<f64>, Vec<f64>, f64), GeomError> {
    // A `Matrix` cannot be constructed with a zero dimension, so the game is
    // always at least one by one.
    let (m, n) = (payoff.rows, payoff.cols);
    // Shift so every entry is at least one, keeping the value positive.
    let lowest = (0..m)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| payoff.get(i, j))
        .fold(f64::INFINITY, f64::min);
    let shift = 1.0 - lowest;

    // min sum(x) s.t. for each column j, sum_i x_i (a_ij + shift) >= 1.
    let mut a = Matrix::zeros(n, m);
    for j in 0..n {
        for i in 0..m {
            a.set(j, i, payoff.get(i, j) + shift);
        }
    }
    let p = LpProblem {
        c: vec![1.0; m],
        a,
        b: vec![1.0; n],
        constraint_types: vec![Cmp::Ge; n],
        bounds: vec![(0.0, f64::INFINITY); m],
        maximize: false,
    };
    let LpResult::Optimal { x, objective, duals, .. } = simplex(&p)? else {
        return Err(GeomError::Degenerate("the game program has no optimum"));
    };
    if objective <= OPT_TOL {
        return Err(GeomError::Degenerate("the game program produced a non-positive total"));
    }

    let value = 1.0 / objective;
    let row: Vec<f64> = x.iter().map(|v| v * value).collect();
    // The duals of the row player's program are the column player's weights.
    let column: Vec<f64> = duals.iter().map(|v| v * value).collect();
    Ok((row, column, value - shift))
}

/// The Chebyshev centre of the polyhedron `{x : a_i . x <= b_i}`: the point
/// furthest from every face, and that distance.
///
/// Maximises `r` subject to `a_i . x + r ||a_i|| <= b_i`. The norm term is
/// what turns "satisfy the constraint" into "stay `r` away from it", and it is
/// why the problem is linear at all -- the distance from a point to a
/// hyperplane is linear in the point.
///
/// Returns `(centre, radius)`. The radius is always unique, but the centre
/// need not be: in a box four wide and six tall the largest inscribed circle
/// has radius two and can sit anywhere along a vertical segment. Only the
/// coordinates that the touching faces pin down are determined, and the
/// returned point is one vertex of that optimal face.
///
/// An unbounded polyhedron gives an infinite radius; an empty one is an error.
///
/// # Errors
/// Returns an error for a shape mismatch, a zero row, or an infeasible system.
pub fn chebyshev_center(a: &Matrix, b: &[f64]) -> Result<(Vec<f64>, f64), GeomError> {
    let (m, n) = (a.rows, a.cols);
    if b.len() != m {
        return Err(GeomError::InvalidArgument("chebyshev_center: shape mismatch"));
    }
    // Variables: the centre (free) then the radius (non-negative).
    let mut design = Matrix::zeros(m, n + 1);
    for i in 0..m {
        let norm: f64 = (0..n).map(|j| a.get(i, j) * a.get(i, j)).sum::<f64>().sqrt();
        if norm <= 0.0 {
            return Err(GeomError::Degenerate("chebyshev_center: a constraint row is all zeros"));
        }
        for j in 0..n {
            design.set(i, j, a.get(i, j));
        }
        design.set(i, n, norm);
    }
    let mut c = vec![0.0; n + 1];
    c[n] = 1.0;
    let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n + 1];
    bounds[n] = (0.0, f64::INFINITY);

    let p = LpProblem {
        c,
        a: design,
        b: b.to_vec(),
        constraint_types: vec![Cmp::Le; m],
        bounds,
        maximize: true,
    };
    match simplex(&p)? {
        LpResult::Optimal { x, .. } => Ok((x[..n].to_vec(), x[n])),
        LpResult::Unbounded => Ok((vec![0.0; n], f64::INFINITY)),
        LpResult::Infeasible => Err(GeomError::Degenerate("chebyshev_center: the region is empty")),
    }
}

/// Least-absolute-deviations regression, solved as a linear program.
///
/// Minimises `sum |y_i - x_i . beta|` by splitting each residual into a
/// positive and a negative part. The result is far less sensitive to an
/// outlier than a least-squares fit, because the cost of a large residual
/// grows linearly rather than quadratically -- an outlier at ten standard
/// deviations pulls a hundred times harder on a least-squares fit than on
/// this one.
///
/// `x` holds one row per observation. Add a column of ones for an intercept.
///
/// # Errors
/// Returns an error on a shape mismatch or if the program has no optimum.
pub fn l1_regression_lp(x: &Matrix, y: &[f64]) -> Result<Vec<f64>, GeomError> {
    let (n, k) = (x.rows, x.cols);
    if y.len() != n || n == 0 || k == 0 {
        return Err(GeomError::InvalidArgument("l1_regression_lp: shape mismatch"));
    }
    // Variables: beta (free, k), then u and v (non-negative, n each).
    let vars = k + 2 * n;
    let mut a = Matrix::zeros(n, vars);
    for i in 0..n {
        for j in 0..k {
            a.set(i, j, x.get(i, j));
        }
        a.set(i, k + i, 1.0);
        a.set(i, k + n + i, -1.0);
    }
    let mut c = vec![0.0; vars];
    for entry in c.iter_mut().skip(k) {
        *entry = 1.0;
    }
    let mut bounds = vec![(0.0, f64::INFINITY); vars];
    for entry in bounds.iter_mut().take(k) {
        *entry = (f64::NEG_INFINITY, f64::INFINITY);
    }

    let p = LpProblem {
        c,
        a,
        b: y.to_vec(),
        constraint_types: vec![Cmp::Eq; n],
        bounds,
        maximize: false,
    };
    match simplex(&p)? {
        LpResult::Optimal { x: sol, .. } => Ok(sol[..k].to_vec()),
        other => Err(match other {
            LpResult::Infeasible => GeomError::Degenerate("l1_regression_lp: infeasible"),
            _ => GeomError::Degenerate("l1_regression_lp: unbounded"),
        }),
    }
}

/// Chebyshev (minimax) regression, solved as a linear program.
///
/// Minimises the largest absolute residual rather than their sum. Where the
/// L1 fit ignores an outlier, this one is dominated by it -- the fit is
/// pinned by the extreme points and by nothing else, which is exactly what is
/// wanted when the residuals are bounded errors rather than noise.
///
/// # Errors
/// Returns an error on a shape mismatch or if the program has no optimum.
pub fn linf_regression_lp(x: &Matrix, y: &[f64]) -> Result<Vec<f64>, GeomError> {
    let (n, k) = (x.rows, x.cols);
    if y.len() != n || n == 0 || k == 0 {
        return Err(GeomError::InvalidArgument("linf_regression_lp: shape mismatch"));
    }
    // Variables: beta (free, k) then t (non-negative).
    let vars = k + 1;
    let mut a = Matrix::zeros(2 * n, vars);
    let mut b = vec![0.0; 2 * n];
    for i in 0..n {
        for j in 0..k {
            a.set(i, j, x.get(i, j));
            a.set(n + i, j, -x.get(i, j));
        }
        a.set(i, k, -1.0);
        a.set(n + i, k, -1.0);
        b[i] = y[i];
        b[n + i] = -y[i];
    }
    let mut c = vec![0.0; vars];
    c[k] = 1.0;
    let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); vars];
    bounds[k] = (0.0, f64::INFINITY);

    let p = LpProblem {
        c,
        a,
        b,
        constraint_types: vec![Cmp::Le; 2 * n],
        bounds,
        maximize: false,
    };
    match simplex(&p)? {
        LpResult::Optimal { x: sol, .. } => Ok(sol[..k].to_vec()),
        other => Err(match other {
            LpResult::Infeasible => GeomError::Degenerate("linf_regression_lp: infeasible"),
            _ => GeomError::Degenerate("linf_regression_lp: unbounded"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    fn optimum(r: &LpResult) -> (&[f64], f64, &[f64], &[f64]) {
        match r {
            LpResult::Optimal { x, objective, duals, reduced_costs } => {
                (x, *objective, duals, reduced_costs)
            }
            other => panic!("expected an optimum, got {other:?}"),
        }
    }

    /// The textbook example used throughout: max 3x + 5y subject to
    /// x <= 4, 2y <= 12, 3x + 2y <= 18. Optimum (2, 6) worth 36.
    fn textbook() -> LpProblem {
        let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0], &[3.0, 2.0]]).unwrap();
        LpProblem::new(vec![3.0, 5.0], a, vec![4.0, 12.0, 18.0], true).unwrap()
    }

    // -----------------------------------------------------------------
    // The simplex method against hand-worked answers
    // -----------------------------------------------------------------

    #[test]
    fn a_hand_worked_maximisation_matches_its_known_optimum() {
        let p = textbook();
        let r = simplex(&p).unwrap();
        let (x, objective, duals, reduced_costs) = optimum(&r);
        assert!((x[0] - 2.0).abs() < 1e-9 && (x[1] - 6.0).abs() < 1e-9, "x = {x:?}");
        assert!((objective - 36.0).abs() < 1e-9, "objective {objective}");
        // The first constraint is slack at the optimum, so its shadow price
        // is zero; the other two bind.
        assert!(duals[0].abs() < 1e-9, "duals {duals:?}");
        assert!((duals[1] - 1.5).abs() < 1e-9, "duals {duals:?}");
        assert!((duals[2] - 1.0).abs() < 1e-9, "duals {duals:?}");
        assert!(reduced_costs.iter().all(|v| v.abs() < 1e-9), "rc {reduced_costs:?}");
        assert!(p.is_feasible(x, 1e-9));
    }

    #[test]
    fn a_minimisation_with_ge_rows_matches_its_known_optimum() {
        // min 2x + 3y s.t. x + y >= 10, x >= 3, y >= 2. Optimum (8, 2) = 22.
        let a = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]]).unwrap();
        let p = LpProblem {
            c: vec![2.0, 3.0],
            a,
            b: vec![10.0, 3.0, 2.0],
            constraint_types: vec![Cmp::Ge; 3],
            bounds: vec![(0.0, f64::INFINITY); 2],
            maximize: false,
        };
        let r = simplex(&p).unwrap();
        let (x, objective, _, _) = optimum(&r);
        assert!((objective - 22.0).abs() < 1e-9, "objective {objective}, x = {x:?}");
        assert!(p.is_feasible(x, 1e-9));
    }

    #[test]
    fn equality_rows_free_variables_and_bounds_are_all_honoured() {
        // min x + y s.t. x + y = 5, y free.
        let p = LpProblem {
            c: vec![1.0, 1.0],
            a: Matrix::from_rows(&[&[1.0, 1.0]]).unwrap(),
            b: vec![5.0],
            constraint_types: vec![Cmp::Eq],
            bounds: vec![(0.0, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)],
            maximize: false,
        };
        let r = simplex(&p).unwrap();
        let (x, objective, duals, _) = optimum(&r);
        assert!((objective - 5.0).abs() < 1e-9);
        assert!(p.is_feasible(x, 1e-9), "x = {x:?}");
        assert!((duals[0] - 1.0).abs() < 1e-9, "the equality dual is {}", duals[0]);

        // A free variable can genuinely go negative when that helps.
        let q = LpProblem {
            c: vec![1.0, 1.0],
            a: Matrix::from_rows(&[&[1.0, -1.0]]).unwrap(),
            b: vec![4.0],
            constraint_types: vec![Cmp::Eq],
            bounds: vec![(0.0, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)],
            maximize: false,
        };
        let r = simplex(&q).unwrap();
        let (x, objective, _, _) = optimum(&r);
        assert!(q.is_feasible(x, 1e-9), "x = {x:?}");
        assert!(objective < 0.0 || x[1] < 1e-9, "the free variable stayed pinned: {x:?}");

        // max x + y s.t. x + y <= 100, 1 <= x <= 3, 2 <= y <= 4.
        let bounded = LpProblem {
            c: vec![1.0, 1.0],
            a: Matrix::from_rows(&[&[1.0, 1.0]]).unwrap(),
            b: vec![100.0],
            constraint_types: vec![Cmp::Le],
            bounds: vec![(1.0, 3.0), (2.0, 4.0)],
            maximize: true,
        };
        let r = simplex(&bounded).unwrap();
        let (x, objective, _, _) = optimum(&r);
        assert!((x[0] - 3.0).abs() < 1e-9 && (x[1] - 4.0).abs() < 1e-9, "x = {x:?}");
        assert!((objective - 7.0).abs() < 1e-9);

        // A lower bound that actually binds.
        let floored = LpProblem {
            c: vec![1.0, 1.0],
            a: Matrix::from_rows(&[&[1.0, 1.0]]).unwrap(),
            b: vec![100.0],
            constraint_types: vec![Cmp::Le],
            bounds: vec![(5.0, f64::INFINITY), (7.0, f64::INFINITY)],
            maximize: false,
        };
        let r = simplex(&floored).unwrap();
        let (x, objective, _, _) = optimum(&r);
        assert!((objective - 12.0).abs() < 1e-9, "objective {objective}, x = {x:?}");
    }

    #[test]
    fn infeasible_and_unbounded_problems_are_reported_as_such() {
        let contradictory = LpProblem {
            c: vec![1.0],
            a: Matrix::from_rows(&[&[1.0], &[1.0]]).unwrap(),
            b: vec![1.0, 5.0],
            constraint_types: vec![Cmp::Le, Cmp::Ge],
            bounds: vec![(0.0, f64::INFINITY)],
            maximize: false,
        };
        assert_eq!(simplex(&contradictory).unwrap(), LpResult::Infeasible);
        assert_eq!(contradictory.objective_at(&[3.0]), 3.0);
        assert!(simplex(&contradictory).unwrap().solution().is_none());

        let open = LpProblem {
            c: vec![1.0],
            a: Matrix::from_rows(&[&[1.0]]).unwrap(),
            b: vec![1.0],
            constraint_types: vec![Cmp::Ge],
            bounds: vec![(0.0, f64::INFINITY)],
            maximize: true,
        };
        assert_eq!(simplex(&open).unwrap(), LpResult::Unbounded);
        assert!(simplex(&open).unwrap().objective().is_none());
    }

    #[test]
    fn bland_s_rule_terminates_on_a_problem_that_cycles_without_it() {
        // Beale's example. Under Dantzig's most-negative rule the simplex
        // method returns to its starting basis after six pivots and repeats
        // forever; Bland's rule cannot, because the basis sequence it visits
        // is lexicographically monotone.
        let a = Matrix::from_rows(&[
            &[0.5, -5.5, -2.5, 9.0],
            &[0.5, -1.5, -0.5, 1.0],
            &[1.0, 0.0, 0.0, 0.0],
        ])
        .unwrap();
        let p = LpProblem {
            c: vec![-10.0, 57.0, 9.0, 24.0],
            a,
            b: vec![0.0, 0.0, 1.0],
            constraint_types: vec![Cmp::Le; 3],
            bounds: vec![(0.0, f64::INFINITY); 4],
            maximize: false,
        };
        let r = simplex(&p).unwrap();
        let (x, objective, _, _) = optimum(&r);
        assert!(p.is_feasible(x, 1e-9), "x = {x:?}");
        assert!((objective - -1.0).abs() < 1e-9, "Beale's optimum is -1, got {objective}");
    }

    #[test]
    fn a_degenerate_problem_still_terminates_with_the_right_value() {
        // Three constraints meeting at one vertex: every basis there is
        // degenerate, and the ratio test ties at every pivot.
        let a = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]]).unwrap();
        let p = LpProblem::new(vec![1.0, 1.0], a, vec![2.0, 1.0, 1.0], true).unwrap();
        let r = simplex(&p).unwrap();
        let (x, objective, _, _) = optimum(&r);
        assert!((objective - 2.0).abs() < 1e-9, "objective {objective}, x = {x:?}");
        assert!(p.is_feasible(x, 1e-9));
    }

    // -----------------------------------------------------------------
    // Duality
    // -----------------------------------------------------------------

    #[test]
    fn the_dual_reaches_the_same_value_and_carries_the_shadow_prices() {
        let p = textbook();
        let primal = simplex(&p).unwrap();
        let d = lp_dual(&p).unwrap();
        let dual = simplex(&d).unwrap();
        let (_, po, py, _) = optimum(&primal);
        let (dx, dobj, _, _) = optimum(&dual);

        assert!((po - dobj).abs() < 1e-9, "primal {po} against dual {dobj}");
        for (a, b) in py.iter().zip(dx) {
            assert!((a - b).abs() < 1e-9, "shadow prices {py:?} against dual solution {dx:?}");
        }
        // The dual of the dual returns to the primal's value.
        let back = simplex(&lp_dual(&d).unwrap()).unwrap();
        assert!((back.objective().unwrap() - po).abs() < 1e-9, "dual of dual gave {back:?}");
        assert!(!d.maximize && p.maximize, "the dual did not flip the sense");
    }

    #[test]
    fn strong_duality_and_complementary_slackness_hold_on_random_programs() {
        // Two theorems on 200 random feasible programs. Strong duality says
        // the objective equals b . y exactly; complementary slackness says a
        // variable in use has zero reduced cost and a slack row has zero
        // shadow price. Neither is imposed anywhere in the solver -- they come
        // out of the optimal basis.
        let mut rng = Rng::new(0x_0D0A_0001);
        let mut solved = 0usize;
        for _ in 0..200 {
            let m = 2 + (rng.next_u64() % 4) as usize;
            let n = 2 + (rng.next_u64() % 4) as usize;
            let mut a = Matrix::zeros(m, n);
            for i in 0..m {
                for j in 0..n {
                    a.set(i, j, (rng.next_f64() * 4.0 - 1.0).round());
                }
            }
            let b: Vec<f64> = (0..m).map(|_| (rng.next_f64() * 20.0 + 1.0).round()).collect();
            let c: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 10.0 - 2.0).round()).collect();
            // All-`<=` rows with a non-negative right-hand side: the origin is
            // always feasible, so only unboundedness can prevent an optimum.
            let p = LpProblem::new(c, a, b, true).unwrap();
            let r = simplex(&p).unwrap();
            let LpResult::Optimal { x, objective, duals, reduced_costs } = &r else {
                continue;
            };
            solved += 1;
            assert!(p.is_feasible(x, 1e-7), "the reported point is not feasible: {x:?}");

            let by: f64 = p.b.iter().zip(duals).map(|(a, b)| a * b).sum();
            assert!(
                close(by, *objective, 1e-7),
                "strong duality failed: b . y = {by}, objective = {objective}"
            );
            // A maximisation's shadow prices on `<=` rows are non-negative:
            // more of a resource cannot hurt.
            assert!(duals.iter().all(|&v| v > -1e-7), "a shadow price went negative: {duals:?}");

            for (j, &xj) in x.iter().enumerate() {
                if xj > 1e-7 {
                    assert!(
                        reduced_costs[j].abs() < 1e-6,
                        "variable {j} is in use but has reduced cost {}",
                        reduced_costs[j]
                    );
                }
            }
            for i in 0..p.m() {
                let row: f64 = (0..p.n()).map(|j| p.a.get(i, j) * x[j]).sum();
                if row < p.b[i] - 1e-7 {
                    assert!(
                        duals[i].abs() < 1e-6,
                        "row {i} is slack but priced at {}",
                        duals[i]
                    );
                }
            }
        }
        assert!(solved > 100, "only {solved} of 200 random programs had an optimum");
    }

    #[test]
    fn weak_duality_bounds_every_feasible_pair() {
        // For any primal-feasible x and dual-feasible y of a maximisation,
        // c . x <= b . y. The optimum is where they meet.
        let p = textbook();
        let d = lp_dual(&p).unwrap();
        let mut rng = Rng::new(0x_0D0A_0002);
        let LpResult::Optimal { objective, .. } = simplex(&p).unwrap() else {
            panic!("expected an optimum");
        };
        for _ in 0..500 {
            let x = vec![rng.next_f64() * 4.0, rng.next_f64() * 6.0];
            if p.is_feasible(&x, 0.0) {
                assert!(
                    p.objective_at(&x) <= objective + 1e-9,
                    "a feasible point beat the optimum"
                );
            }
            let y = vec![rng.next_f64() * 2.0, rng.next_f64() * 2.0, rng.next_f64() * 2.0];
            if d.is_feasible(&y, 0.0) {
                assert!(
                    d.objective_at(&y) >= objective - 1e-9,
                    "a dual-feasible point fell below the optimum"
                );
            }
        }
    }

    #[test]
    fn lp_dual_rejects_a_problem_it_cannot_transpose() {
        let mut p = textbook();
        p.bounds[0] = (0.0, 5.0);
        assert!(lp_dual(&p).is_err(), "a bounded variable should be refused");
        p.bounds[0] = (1.0, f64::INFINITY);
        assert!(lp_dual(&p).is_err(), "a shifted variable should be refused");
    }

    // -----------------------------------------------------------------
    // Interior point
    // -----------------------------------------------------------------

    #[test]
    fn the_two_solvers_agree_on_a_hundred_random_programs() {
        // The strongest check available on either method: they share no code
        // beyond the standardisation, walk the feasible region in completely
        // different ways, and must land on the same value.
        let mut rng = Rng::new(0x_0117_0001);
        let mut compared = 0usize;
        for _ in 0..100 {
            let m = 2 + (rng.next_u64() % 4) as usize;
            let n = 2 + (rng.next_u64() % 4) as usize;
            let mut a = Matrix::zeros(m, n);
            for i in 0..m {
                for j in 0..n {
                    a.set(i, j, (rng.next_f64() * 3.0).round() + 1.0);
                }
            }
            let b: Vec<f64> = (0..m).map(|_| (rng.next_f64() * 20.0 + 5.0).round()).collect();
            let c: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 9.0).round() + 1.0).collect();
            let p = LpProblem::new(c, a, b, true).unwrap();

            let s = simplex(&p).unwrap();
            let i = interior_point(&p, 1e-9).unwrap();
            let (LpResult::Optimal { objective: so, .. }, LpResult::Optimal { objective: io, x: ix, .. }) =
                (&s, &i)
            else {
                continue;
            };
            compared += 1;
            assert!(
                close(*so, *io, 1e-5),
                "simplex {so} against interior point {io}"
            );
            assert!(p.is_feasible(ix, 1e-5), "the interior point answer is infeasible: {ix:?}");
        }
        assert!(compared > 80, "only {compared} of 100 programs were comparable");
    }

    #[test]
    fn interior_point_handles_the_awkward_shapes_too() {
        // Equality rows, `>=` rows, and a free variable.
        let p = LpProblem {
            c: vec![2.0, 3.0, 1.0],
            a: Matrix::from_rows(&[&[1.0, 1.0, 1.0], &[1.0, -1.0, 0.0]]).unwrap(),
            b: vec![10.0, 2.0],
            constraint_types: vec![Cmp::Eq, Cmp::Ge],
            bounds: vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            maximize: false,
        };
        let s = simplex(&p).unwrap();
        let i = interior_point(&p, 1e-10).unwrap();
        let (_, so, _, _) = optimum(&s);
        let (ix, io, _, _) = optimum(&i);
        assert!(close(so, io, 1e-5), "simplex {so} against interior point {io}");
        assert!(p.is_feasible(ix, 1e-5), "x = {ix:?}");

        assert!(interior_point(&p, 0.0).is_err());
        assert!(interior_point(&p, -1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Sensitivity
    // -----------------------------------------------------------------

    #[test]
    fn a_right_hand_side_moves_the_objective_at_exactly_its_shadow_price() {
        let p = textbook();
        let (_, objective, duals, _) = {
            let r = simplex(&p).unwrap();
            let (x, o, d, rc) = optimum(&r);
            (x.to_vec(), o, d.to_vec(), rc.to_vec())
        };
        let (c_ranges, b_ranges) = sensitivity_ranges(&p).unwrap();
        assert_eq!(b_ranges.len(), p.m());
        assert_eq!(c_ranges.len(), p.n());

        for i in 0..p.m() {
            let (lo, hi) = b_ranges[i];
            assert!(lo <= p.b[i] + 1e-9 && hi >= p.b[i] - 1e-9, "row {i} range {lo}..{hi}");
            for fraction in [0.3f64, 0.7, -0.3, -0.7] {
                let span = if fraction > 0.0 { hi - p.b[i] } else { p.b[i] - lo };
                if !span.is_finite() || span <= 0.0 {
                    continue;
                }
                let delta = fraction.signum() * fraction.abs() * span;
                let mut q = p.clone();
                q.b[i] += delta;
                let moved = simplex(&q).unwrap().objective().unwrap();
                assert!(
                    (moved - objective - duals[i] * delta).abs() < 1e-7,
                    "row {i}, delta {delta}: objective {moved}, expected {}",
                    objective + duals[i] * delta
                );
            }
        }
    }

    #[test]
    fn an_objective_coefficient_inside_its_range_leaves_the_solution_put() {
        let p = textbook();
        let base = simplex(&p).unwrap();
        let (x0, _, _, _) = optimum(&base);
        let x0 = x0.to_vec();
        let (c_ranges, _) = sensitivity_ranges(&p).unwrap();

        for j in 0..p.n() {
            let (lo, hi) = c_ranges[j];
            assert!(lo <= p.c[j] + 1e-9 && hi >= p.c[j] - 1e-9, "coefficient {j} range {lo}..{hi}");
            for target in [lo, hi] {
                if !target.is_finite() {
                    continue;
                }
                // Just inside the range the optimal point must not move.
                let inside = p.c[j] + 0.95 * (target - p.c[j]);
                let mut q = p.clone();
                q.c[j] = inside;
                let moved = simplex(&q).unwrap();
                let (x1, o1, _, _) = optimum(&moved);
                for (a, b) in x0.iter().zip(x1) {
                    assert!(
                        (a - b).abs() < 1e-7,
                        "coefficient {j} at {inside} moved the solution from {x0:?} to {x1:?}"
                    );
                }
                // And the objective is the new coefficient against the old point.
                assert!(close(o1, q.objective_at(&x0), 1e-9));
            }
        }
    }

    #[test]
    fn sensitivity_declines_problems_it_cannot_report_on() {
        let mut p = textbook();
        p.bounds[0] = (0.0, 5.0);
        assert!(sensitivity_ranges(&p).is_err(), "a bounded variable should be refused");

        let unbounded = LpProblem {
            c: vec![1.0],
            a: Matrix::from_rows(&[&[1.0]]).unwrap(),
            b: vec![1.0],
            constraint_types: vec![Cmp::Ge],
            bounds: vec![(0.0, f64::INFINITY)],
            maximize: true,
        };
        assert!(sensitivity_ranges(&unbounded).is_err());
    }

    // -----------------------------------------------------------------
    // The dual simplex
    // -----------------------------------------------------------------

    #[test]
    fn the_dual_simplex_reaches_the_primal_answer_from_an_optimal_basis() {
        // The use case: a right-hand side changes, the old basis stays
        // dual-feasible, and re-solving is a few pivots rather than a fresh
        // start. The answer must match a fresh primal solve exactly.
        let p = textbook();
        // The all-slack basis of a maximisation with non-negative b is
        // dual-feasible only when every objective coefficient is non-positive,
        // so use a minimisation for a clean start.
        let q = LpProblem {
            c: vec![3.0, 5.0],
            a: p.a.clone(),
            b: vec![4.0, 12.0, 18.0],
            constraint_types: vec![Cmp::Ge; 3],
            bounds: vec![(0.0, f64::INFINITY); 2],
            maximize: false,
        };
        // Standard form has two structural columns then three surpluses;
        // starting from the surplus basis is dual-feasible for a minimisation
        // with non-negative costs.
        let fresh = simplex(&q).unwrap();
        let (fx, fo, _, _) = optimum(&fresh);
        let via_dual = dual_simplex(&q, &[2, 3, 4]).unwrap();
        let (dx, dobj, _, _) = optimum(&via_dual);
        assert!(close(fo, dobj, 1e-9), "primal {fo} against dual simplex {dobj}");
        for (a, b) in fx.iter().zip(dx) {
            assert!((a - b).abs() < 1e-7, "points differ: {fx:?} against {dx:?}");
        }
    }

    #[test]
    fn the_dual_simplex_rejects_a_basis_it_cannot_start_from() {
        let p = textbook();
        assert!(dual_simplex(&p, &[0, 1]).is_err(), "a short basis should be refused");
        assert!(dual_simplex(&p, &[0, 1, 99]).is_err(), "an out-of-range column should be refused");
        // The textbook maximisation has negative internal costs, so the
        // all-slack basis is not dual-feasible and the method says so rather
        // than quietly repairing it.
        assert!(dual_simplex(&p, &[2, 3, 4]).is_err(), "a dual-infeasible basis should be refused");
    }

    // -----------------------------------------------------------------
    // The modelling language
    // -----------------------------------------------------------------

    #[test]
    fn the_parser_reproduces_a_hand_built_problem() {
        let text = "\
max 3x + 5y
subject to
  x <= 4
  2y <= 12
  3x + 2y <= 18
";
        let parsed = lp_from_str(text).unwrap();
        let built = textbook();
        assert_eq!(parsed.c, built.c);
        assert_eq!(parsed.b, built.b);
        assert_eq!(parsed.constraint_types, built.constraint_types);
        assert_eq!(parsed.maximize, built.maximize);
        for i in 0..built.m() {
            for j in 0..built.n() {
                assert!((parsed.a.get(i, j) - built.a.get(i, j)).abs() < 1e-12);
            }
        }
        assert!(close(simplex(&parsed).unwrap().objective().unwrap(), 36.0, 1e-9));
    }

    #[test]
    fn the_parser_handles_signs_senses_comments_and_bounds() {
        let text = "\
# a comment, and a blank line follow

min 2a - 3b + c
s.t.
  a + b >= 4
  a - 2b + 3c = 6      # an equality
  -a + b <= 2
bounds
  b <= 10
  free c
";
        let p = lp_from_str(text).unwrap();
        assert!(!p.maximize);
        assert_eq!(p.c, vec![2.0, -3.0, 1.0]);
        assert_eq!(p.constraint_types, vec![Cmp::Ge, Cmp::Eq, Cmp::Le]);
        assert_eq!(p.b, vec![4.0, 6.0, 2.0]);
        assert_eq!(p.bounds[1], (0.0, 10.0));
        assert_eq!(p.bounds[2].0, f64::NEG_INFINITY);
        // Row two: a - 2b + 3c.
        assert!((p.a.get(1, 0) - 1.0).abs() < 1e-12);
        assert!((p.a.get(1, 1) + 2.0).abs() < 1e-12);
        assert!((p.a.get(1, 2) - 3.0).abs() < 1e-12);
        // Row three starts with a leading minus.
        assert!((p.a.get(2, 0) + 1.0).abs() < 1e-12);
        // And it solves.
        let r = simplex(&p).unwrap();
        if let LpResult::Optimal { x, .. } = &r {
            assert!(p.is_feasible(x, 1e-7), "x = {x:?}");
        }
    }

    #[test]
    fn the_parser_reports_what_it_could_not_read() {
        assert!(lp_from_str("").is_err());
        assert!(lp_from_str("solve 3x").is_err(), "a missing sense should be refused");
        assert!(lp_from_str("max 3x\nst\n x + y").is_err(), "a row with no operator");
        assert!(lp_from_str("max 3x\nst\n x <= abc").is_err(), "a non-numeric right-hand side");
        assert!(lp_from_str("max 3x\nst\n 4 <= 5").is_err(), "a row with no variable");
        assert!(
            lp_from_str("max 3x\nst\n x <= 4\nbounds\n 2x >= 1").is_err(),
            "a bounds line with a coefficient"
        );
    }

    // -----------------------------------------------------------------
    // Classical models
    // -----------------------------------------------------------------

    #[test]
    fn the_diet_problem_meets_every_requirement_at_least_cost() {
        // Two foods, two nutrients. Food 0 is cheap but thin; food 1 is dear
        // but rich.
        let nutrients = Matrix::from_rows(&[&[1.0, 3.0], &[2.0, 1.0]]).unwrap();
        let p = diet_problem(&[1.0, 2.0], &nutrients, &[9.0, 8.0]).unwrap();
        let r = simplex(&p).unwrap();
        let (x, objective, duals, _) = optimum(&r);
        assert!(p.is_feasible(x, 1e-7), "the diet does not meet the requirements: {x:?}");
        for k in 0..2 {
            let got: f64 = (0..2).map(|j| nutrients.get(k, j) * x[j]).sum();
            assert!(got >= 9.0f64.min(8.0) - 1e-7, "nutrient {k} came to {got}");
        }
        // A minimisation over `>=` rows prices every nutrient non-negatively:
        // needing more of something cannot make the diet cheaper.
        assert!(duals.iter().all(|&v| v > -1e-7), "duals {duals:?}");
        let by: f64 = p.b.iter().zip(duals).map(|(a, b)| a * b).sum();
        assert!(close(by, objective, 1e-7), "b . y = {by} against {objective}");

        assert!(diet_problem(&[1.0], &nutrients, &[9.0, 8.0]).is_err());
        assert!(diet_problem(&[1.0, 2.0], &nutrients, &[9.0]).is_err());
    }

    #[test]
    fn a_production_plan_exhausts_the_binding_resource() {
        let usage = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]).unwrap();
        let p = production_planning(&[5.0, 4.0], &usage, &[100.0, 90.0]).unwrap();
        let r = simplex(&p).unwrap();
        let (x, objective, duals, _) = optimum(&r);
        assert!(p.is_feasible(x, 1e-7));
        assert!(objective > 0.0);
        // Every resource with a positive shadow price must be fully used --
        // that is complementary slackness read backwards.
        for i in 0..2 {
            if duals[i] > 1e-7 {
                let used: f64 = (0..2).map(|j| usage.get(i, j) * x[j]).sum();
                assert!(
                    (used - p.b[i]).abs() < 1e-7,
                    "resource {i} is priced at {} but only {used} of {} is used",
                    duals[i],
                    p.b[i]
                );
            }
        }
        assert!(production_planning(&[5.0], &usage, &[100.0, 90.0]).is_err());
    }

    #[test]
    fn the_transportation_problem_ships_everything_demanded_at_least_cost() {
        let costs = Matrix::from_rows(&[&[4.0, 6.0, 9.0], &[5.0, 3.0, 8.0], &[7.0, 7.0, 2.0]])
            .unwrap();
        let supply = [30.0, 40.0, 50.0];
        let demand = [25.0, 35.0, 45.0];
        let r = transportation_problem(&supply, &demand, &costs).unwrap();
        let (x, objective, _, _) = optimum(&r);

        for i in 0..3 {
            let shipped: f64 = (0..3).map(|j| x[i * 3 + j]).sum();
            assert!(shipped <= supply[i] + 1e-7, "source {i} over-shipped {shipped}");
        }
        for j in 0..3 {
            let received: f64 = (0..3).map(|i| x[i * 3 + j]).sum();
            assert!(received >= demand[j] - 1e-7, "sink {j} received only {received}");
        }
        // The cheapest assignment here sends each source to its own cheapest
        // sink where possible; a greedy lower bound cannot beat the optimum.
        let greedy_bound: f64 = (0..3)
            .map(|j| {
                let cheapest =
                    (0..3).map(|i| costs.get(i, j)).fold(f64::INFINITY, f64::min);
                cheapest * demand[j]
            })
            .sum();
        assert!(objective >= greedy_bound - 1e-7, "{objective} beat the bound {greedy_bound}");
        assert!(objective <= 1e6);

        // Total unimodularity: integral supplies and demands give an integral
        // optimum with no branch and bound anywhere.
        for v in x {
            assert!((v - v.round()).abs() < 1e-7, "a shipment came out fractional: {v}");
        }

        // Demand beyond supply is infeasible by inspection.
        assert_eq!(
            transportation_problem(&[1.0], &[5.0], &Matrix::from_rows(&[&[1.0]]).unwrap())
                .unwrap(),
            LpResult::Infeasible
        );
        assert!(transportation_problem(&[1.0, 2.0], &[1.0], &costs).is_err());
        assert!(transportation_problem(&[-1.0], &[1.0], &Matrix::from_rows(&[&[1.0]]).unwrap())
            .is_err());
    }

    // -----------------------------------------------------------------
    // Games
    // -----------------------------------------------------------------

    #[test]
    fn matching_pennies_is_fair_and_played_uniformly() {
        let payoff = Matrix::from_rows(&[&[1.0, -1.0], &[-1.0, 1.0]]).unwrap();
        let (row, column, value) = two_player_zero_sum_lp(&payoff).unwrap();
        assert!(value.abs() < 1e-9, "the value should be zero, got {value}");
        for v in &row {
            assert!((v - 0.5).abs() < 1e-9, "row strategy {row:?}");
        }
        for v in &column {
            assert!((v - 0.5).abs() < 1e-9, "column strategy {column:?}");
        }
        assert!(close(row.iter().sum::<f64>(), 1.0, 1e-9));
        assert!(close(column.iter().sum::<f64>(), 1.0, 1e-9));
    }

    #[test]
    fn rock_paper_scissors_is_fair_and_played_uniformly() {
        let payoff = Matrix::from_rows(&[
            &[0.0, -1.0, 1.0],
            &[1.0, 0.0, -1.0],
            &[-1.0, 1.0, 0.0],
        ])
        .unwrap();
        let (row, column, value) = two_player_zero_sum_lp(&payoff).unwrap();
        assert!(value.abs() < 1e-9, "value {value}");
        for v in row.iter().chain(&column) {
            assert!((v - 1.0 / 3.0).abs() < 1e-9, "row {row:?} column {column:?}");
        }
    }

    #[test]
    fn the_minimax_value_is_what_both_players_can_guarantee() {
        // The real content of the theorem: neither player can do better than
        // the value against the other's optimal strategy. Check both
        // directions against every pure response, which is enough since a
        // mixed strategy is a convex combination of them.
        let payoff = Matrix::from_rows(&[&[3.0, -1.0, 2.0], &[-2.0, 4.0, 0.0]]).unwrap();
        let (row, column, value) = two_player_zero_sum_lp(&payoff).unwrap();
        assert!(close(row.iter().sum::<f64>(), 1.0, 1e-9), "row {row:?}");
        assert!(close(column.iter().sum::<f64>(), 1.0, 1e-9), "column {column:?}");
        assert!(row.iter().chain(&column).all(|&v| v > -1e-9), "a negative probability");

        // Against the row player's strategy, no column keeps the payoff below
        // the value.
        for j in 0..payoff.cols {
            let got: f64 = (0..payoff.rows).map(|i| row[i] * payoff.get(i, j)).sum();
            assert!(got >= value - 1e-7, "column {j} held the row player to {got} below {value}");
        }
        // And against the column player's strategy, no row beats it.
        for i in 0..payoff.rows {
            let got: f64 = (0..payoff.cols).map(|j| column[j] * payoff.get(i, j)).sum();
            assert!(got <= value + 1e-7, "row {i} earned {got} above {value}");
        }

        // A game with a saddle point is played purely, at that entry.
        let saddle = Matrix::from_rows(&[&[4.0, 5.0], &[2.0, 3.0]]).unwrap();
        let (r2, _, v2) = two_player_zero_sum_lp(&saddle).unwrap();
        assert!((v2 - 4.0).abs() < 1e-9, "the saddle value is 4, got {v2}");
        assert!((r2[0] - 1.0).abs() < 1e-9, "the row player should play row 0: {r2:?}");

        // A one-by-one game is the degenerate case that does exist: one row,
        // one column, and no choice for either player.
        let trivial = Matrix::from_rows(&[&[7.0]]).unwrap();
        let (r, c, v) = two_player_zero_sum_lp(&trivial).unwrap();
        assert!((v - 7.0).abs() < 1e-9 && (r[0] - 1.0).abs() < 1e-9 && (c[0] - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------
    // Geometry and regression
    // -----------------------------------------------------------------

    #[test]
    fn the_chebyshev_centre_of_a_box_is_its_middle() {
        // The box [0, 4] x [0, 6], written as four half-spaces. The inscribed
        // circle has radius 2 and touches the two nearer faces.
        let a = Matrix::from_rows(&[
            &[1.0, 0.0],
            &[-1.0, 0.0],
            &[0.0, 1.0],
            &[0.0, -1.0],
        ])
        .unwrap();
        let (centre, radius) = chebyshev_center(&a, &[4.0, 0.0, 6.0, 0.0]).unwrap();
        assert!((radius - 2.0).abs() < 1e-9, "radius {radius}");
        // The horizontal position is pinned, since the box is exactly two
        // radii wide. The vertical one is not: the circle slides freely in a
        // box six tall, so only the range is determined.
        assert!((centre[0] - 2.0).abs() < 1e-9, "centre {centre:?}");
        assert!(
            (2.0..=4.0).contains(&centre[1]),
            "centre {centre:?} puts the circle outside the box"
        );

        // The circle really does fit: every face is at least `radius` away.
        for i in 0..4 {
            let norm: f64 = (0..2).map(|j| a.get(i, j) * a.get(i, j)).sum::<f64>().sqrt();
            let slack = [4.0, 0.0, 6.0, 0.0][i]
                - (0..2).map(|j| a.get(i, j) * centre[j]).sum::<f64>();
            assert!(slack / norm >= radius - 1e-7, "face {i} is only {} away", slack / norm);
        }

        // A half-plane is unbounded, so no largest circle fits.
        let half = Matrix::from_rows(&[&[1.0, 0.0]]).unwrap();
        assert!(chebyshev_center(&half, &[1.0]).unwrap().1.is_infinite());
        // A contradictory pair has no interior at all.
        let empty = Matrix::from_rows(&[&[1.0], &[-1.0]]).unwrap();
        assert!(chebyshev_center(&empty, &[-1.0, -1.0]).is_err());
        assert!(chebyshev_center(&Matrix::zeros(1, 2), &[1.0]).is_err());
        assert!(chebyshev_center(&a, &[1.0]).is_err());
    }

    #[test]
    fn the_l1_fit_shrugs_off_an_outlier_that_drags_least_squares() {
        // Points on a straight line, with one gross outlier. The L1 fit should
        // stay on the line; a least-squares fit cannot.
        let n = 21usize;
        let mut design = Matrix::zeros(n, 2);
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = i as f64;
            design.set(i, 0, 1.0);
            design.set(i, 1, t);
            y[i] = 3.0 + 2.0 * t;
        }
        y[10] += 100.0;

        let l1 = l1_regression_lp(&design, &y).unwrap();
        assert!((l1[0] - 3.0).abs() < 1e-6, "intercept {} should be 3", l1[0]);
        assert!((l1[1] - 2.0).abs() < 1e-6, "slope {} should be 2", l1[1]);

        let l2 = crate::linalg::qr::least_squares(&design, &y).unwrap();
        assert!(
            (l2[0] - 3.0).abs() > 1.0,
            "least squares was supposed to be dragged, got {l2:?}"
        );

        // The L1 objective at the L1 fit is no worse than at the L2 fit --
        // which is what "minimises the sum of absolute deviations" means.
        let cost = |beta: &[f64]| -> f64 {
            (0..n)
                .map(|i| (y[i] - beta[0] - beta[1] * design.get(i, 1)).abs())
                .sum()
        };
        assert!(cost(&l1) <= cost(&l2) + 1e-7, "L1 {} against L2 {}", cost(&l1), cost(&l2));
        assert!(l1_regression_lp(&design, &y[..3]).is_err());
    }

    #[test]
    fn the_minimax_fit_equalises_its_largest_residuals() {
        // The Chebyshev fit is pinned by the extreme points: at the optimum
        // the largest residual is attained at least k + 1 times with
        // alternating signs, for k parameters. Here k = 2.
        let n = 12usize;
        let mut design = Matrix::zeros(n, 2);
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = i as f64;
            design.set(i, 0, 1.0);
            design.set(i, 1, t);
            // A line plus a deterministic wobble.
            y[i] = 1.0 + 0.5 * t + (t * 1.7).sin();
        }
        let beta = linf_regression_lp(&design, &y).unwrap();
        let residuals: Vec<f64> =
            (0..n).map(|i| y[i] - beta[0] - beta[1] * design.get(i, 1)).collect();
        let worst = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
        let attained = residuals.iter().filter(|r| (r.abs() - worst).abs() < 1e-7).count();
        assert!(attained >= 3, "only {attained} residuals reached the maximum {worst}");
        // Both signs appear among them, which is what makes it a minimax fit
        // rather than merely a fit with a large residual.
        let extremes: Vec<f64> =
            residuals.iter().copied().filter(|r| (r.abs() - worst).abs() < 1e-7).collect();
        assert!(
            extremes.iter().any(|&v| v > 0.0) && extremes.iter().any(|&v| v < 0.0),
            "the extreme residuals do not alternate in sign: {extremes:?}"
        );

        // And it really is minimax: no other fit has a smaller worst residual.
        let l1 = l1_regression_lp(&design, &y).unwrap();
        let l1_worst = (0..n)
            .map(|i| (y[i] - l1[0] - l1[1] * design.get(i, 1)).abs())
            .fold(0.0f64, f64::max);
        assert!(worst <= l1_worst + 1e-7, "minimax {worst} against L1's worst {l1_worst}");
        assert!(linf_regression_lp(&design, &y[..3]).is_err());
    }

    // -----------------------------------------------------------------
    // Input validation
    // -----------------------------------------------------------------

    #[test]
    fn malformed_problems_are_refused_rather_than_solved() {
        let a = Matrix::from_rows(&[&[1.0, 2.0]]).unwrap();
        assert!(LpProblem::new(vec![1.0], a.clone(), vec![1.0], false).is_err());
        assert!(LpProblem::new(vec![1.0, 2.0], a.clone(), vec![1.0, 2.0], false).is_err());
        // A `Matrix` cannot be empty, so an empty objective is reached by
        // building the struct directly.
        let empty = LpProblem {
            c: Vec::new(),
            a: Matrix::from_rows(&[&[1.0]]).unwrap(),
            b: vec![1.0],
            constraint_types: vec![Cmp::Le],
            bounds: Vec::new(),
            maximize: false,
        };
        assert!(empty.validate().is_err(), "a problem with no variables should be refused");

        let mut p = LpProblem::new(vec![1.0, 2.0], a, vec![1.0], false).unwrap();
        p.bounds[0] = (5.0, 1.0);
        assert!(p.validate().is_err(), "an inverted bound should be refused");
        p.bounds[0] = (0.0, f64::INFINITY);
        p.constraint_types.push(Cmp::Le);
        assert!(p.validate().is_err(), "a spare constraint sense should be refused");

        let mut q = textbook();
        q.c[0] = f64::NAN;
        assert!(q.validate().is_err(), "a non-finite coefficient should be refused");
        assert!(!q.is_feasible(&[1.0], 1e-9), "a wrong-length point is not feasible");
    }
}
