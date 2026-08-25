//! Dimensional analysis: Buckingham's theorem, the named groups,
//! natural units and the Planck scale.
//!
//! # Buckingham's theorem is a rank computation
//!
//! A physical relation among `n` quantities built from `r` independent
//! dimensions can be rewritten as a relation among exactly `n - r`
//! dimensionless groups. That is not a heuristic: the dimension vectors
//! form the columns of a matrix, a dimensionless product of powers is a
//! vector in its null space, and the dimension of a null space is the
//! column count minus the rank. Every part of it is linear algebra over
//! the rationals.
//!
//! Which is why [`buckingham_pi`] works in [`Rational`] rather than in
//! floating point. An exponent vector is *exactly* in the null space or
//! it is not, and a group whose dimensions cancel to `1e-16` instead of
//! to zero is not a dimensionless group -- it is a rounding error that
//! will be reported as physics. The returned exponents are exact
//! rationals for the same reason: the Reynolds number's exponents happen
//! to be integers, but the null space basis of a general problem is not
//! integral, and rounding it would silently change the group.
//!
//! The theorem says how many groups there are, not which ones. Any basis
//! of the null space works, and the conventional groups -- Reynolds,
//! Froude, Mach -- are particular choices made for physical reasons that
//! the algebra knows nothing about. [`dimensionless_groups_named`] lists
//! those conventions; `buckingham_pi` finds a basis and makes no claim
//! that it is the one anybody would name.
//!
//! # Natural units are a change of bookkeeping, not of physics
//!
//! Setting `hbar = c = 1` makes length, time and mass powers of a single
//! unit, conventionally energy: `[L] = [T] = [E]^-1` and `[M] = [E]`.
//! Nothing physical changes -- the dimensionless combinations are the
//! same -- but a quantity's dimension collapses to one integer, its
//! energy power, and [`natural_units_convert`] returns the magnitude in
//! `eV` to that power. Electromagnetic and thermal dimensions need
//! further conventions to absorb, so a dimension involving amperes,
//! kelvin, moles or candela is refused rather than guessed at.
//!
//! # Checking a formula is not the same as evaluating it
//!
//! [`dimensional_check_formula`] walks a symbolic expression and asks
//! whether it is dimensionally coherent: that every term of every sum
//! agrees, and that nothing dimensioned is handed to a sine or an
//! exponential. Neither question can be answered by running the
//! formula, because both sides of `x + v` are perfectly good floats. It
//! is the check a physicist does by eye before believing an algebra
//! step, done mechanically, and it catches the dropped factor that
//! numerical testing cannot.

use crate::exact::rational::Rational;
use crate::exact::symbolic::Expr;
use crate::units::quantity::{codata, Dim, DimError};

/// The exponent vectors of a set of quantities, as an exact matrix of
/// seven rows by `n` columns.
fn dimension_matrix(dims: &[Dim]) -> Vec<Vec<Rational>> {
    (0..7)
        .map(|row| {
            dims.iter().map(|d| Rational::from_i64(d.exponents()[row] as i64, 1)).collect()
        })
        .collect()
}

/// Reduces a matrix to row echelon form in place, returning the pivot
/// column of each row that has one.
fn row_reduce(matrix: &mut [Vec<Rational>]) -> Vec<usize> {
    let rows = matrix.len();
    let cols = if rows == 0 { 0 } else { matrix[0].len() };
    let mut pivots = Vec::new();
    let mut row = 0;
    for col in 0..cols {
        // Exact arithmetic means "is this entry zero" is a question with
        // an answer, rather than a threshold nobody can choose well.
        let Some(found) = (row..rows).find(|&r| !matrix[r][col].is_zero()) else {
            continue;
        };
        matrix.swap(row, found);
        let inverse = matrix[row][col].recip().expect("the pivot is nonzero");
        for c in col..cols {
            matrix[row][c] = matrix[row][c].mul(&inverse);
        }
        for r in 0..rows {
            if r == row || matrix[r][col].is_zero() {
                continue;
            }
            let factor = matrix[r][col].clone();
            for c in col..cols {
                let term = factor.mul(&matrix[row][c]);
                matrix[r][c] = matrix[r][c].sub(&term);
            }
        }
        pivots.push(col);
        row += 1;
        if row == rows {
            break;
        }
    }
    pivots
}

/// A basis for the dimensionless groups of a set of quantities.
///
/// Returns exactly `n - rank` vectors of `n` exact rational exponents.
/// The product of the quantities raised to those exponents is
/// dimensionless, exactly.
///
/// Any basis of the null space is a valid answer and this one is
/// whichever the elimination produces; see the module note on why that
/// is not the same as producing the groups anybody has named.
///
/// # Errors
///
/// [`DimError::Malformed`] if given no quantities.
pub fn buckingham_pi(dims: &[Dim]) -> Result<Vec<Vec<Rational>>, DimError> {
    if dims.is_empty() {
        return Err(DimError::Malformed("no quantities"));
    }
    let n = dims.len();
    let mut matrix = dimension_matrix(dims);
    let pivots = row_reduce(&mut matrix);
    let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
    let mut basis = Vec::with_capacity(free.len());
    for &f in &free {
        let mut vector = vec![Rational::zero(); n];
        vector[f] = Rational::one();
        // Each pivot variable is minus the coefficient of the free one.
        for (r, &p) in pivots.iter().enumerate() {
            vector[p] = matrix[r][f].neg();
        }
        basis.push(vector);
    }
    Ok(basis)
}

/// Checks that a vector of exponents really does cancel every dimension.
///
/// Exact: the sum of each row is compared against zero, not against a
/// tolerance.
///
/// # Errors
///
/// [`DimError::Malformed`] if the lengths disagree.
pub fn is_dimensionless_group(dims: &[Dim], exponents: &[Rational]) -> Result<bool, DimError> {
    if dims.len() != exponents.len() {
        return Err(DimError::Malformed("one exponent per quantity is needed"));
    }
    for row in 0..7 {
        let mut total = Rational::zero();
        for (d, e) in dims.iter().zip(exponents) {
            let contribution = Rational::from_i64(d.exponents()[row] as i64, 1).mul(e);
            total = total.add(&contribution);
        }
        if !total.is_zero() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The dimensionless groups that have names, with their formulas and
/// what each compares.
///
/// The formulas are the conventional ones. Each is *a* member of its
/// problem's null space rather than the only one -- see the module note.
pub fn dimensionless_groups_named() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("Reynolds", "rho v L / mu", "inertia against viscosity"),
        ("Froude", "v / sqrt(g L)", "inertia against gravity"),
        ("Weber", "rho v^2 L / sigma", "inertia against surface tension"),
        ("Mach", "v / c", "speed against the speed of sound"),
        ("Prandtl", "nu / alpha", "momentum diffusivity against thermal"),
        ("Rayleigh", "g beta dT L^3 / (nu alpha)", "buoyancy against diffusion"),
        ("Peclet", "v L / alpha", "advection against diffusion"),
        ("Nusselt", "h L / k", "total heat transfer against conduction"),
        ("Biot", "h L / k_solid", "surface against internal resistance"),
        ("Strouhal", "f L / v", "shedding frequency against flow"),
        ("Knudsen", "lambda / L", "mean free path against geometry"),
        ("Schmidt", "nu / D", "momentum diffusivity against mass"),
        ("Euler", "dp / (rho v^2)", "pressure against inertia"),
        ("Capillary", "mu v / sigma", "viscosity against surface tension"),
        ("Stokes", "tau v / L", "particle response against flow"),
    ]
}

/// The power of energy a dimension corresponds to when `hbar = c = 1`.
///
/// `[M] = [E]`, `[L] = [T] = [E]^-1`, so the power is
/// `kg - m - s`.
///
/// # Errors
///
/// [`DimError::Mismatch`] if the dimension involves amperes, kelvin,
/// moles or candela, which need further conventions to absorb and are
/// refused rather than guessed at.
pub fn natural_units_power(dim: Dim) -> Result<i32, DimError> {
    if dim.a != 0 || dim.k != 0 || dim.mol != 0 || dim.cd != 0 {
        return Err(DimError::Mismatch {
            expected: Dim::new(dim.m, dim.kg, dim.s, 0, 0, 0, 0),
            found: dim,
        });
    }
    Ok(dim.kg as i32 - dim.m as i32 - dim.s as i32)
}

/// Expresses an SI magnitude in electron volts to the power
/// [`natural_units_power`] gives.
///
/// # Errors
///
/// As [`natural_units_power`].
pub fn natural_units_convert(value: f64, dim: Dim) -> Result<f64, DimError> {
    natural_units_power(dim)?;
    // One kilogram is c^2/e electron volts; one metre is 1/(hbar c) and
    // one second is 1/hbar inverse electron volts. Each SI unit in the
    // dimension contributes its own factor.
    let c = codata("speed of light").expect("a listed constant");
    let e = codata("elementary charge").expect("a listed constant");
    let hbar = codata("reduced Planck constant").expect("a listed constant");
    let kg_in_ev = c * c / e;
    let inverse_metre_in_ev = hbar * c / e;
    let inverse_second_in_ev = hbar / e;
    Ok(value
        * kg_in_ev.powi(dim.kg as i32)
        / inverse_metre_in_ev.powi(dim.m as i32)
        / inverse_second_in_ev.powi(dim.s as i32))
}

/// The Planck units, as `(name, value, unit)`.
///
/// Each is built from `hbar`, `c` and `G` alone, which is the point:
/// they are the only combination of those three with the dimensions of a
/// length, a time, a mass and so on, so they are the scale at which
/// gravity and quantum mechanics are the same size. The defining
/// relations are checked in the tests against the CODATA values rather
/// than the numbers being copied in.
pub fn planck_units() -> Vec<(&'static str, f64, &'static str)> {
    let hbar = codata("reduced Planck constant").expect("a listed constant");
    let c = codata("speed of light").expect("a listed constant");
    let g = codata("gravitational constant").expect("a listed constant");
    let kb = codata("Boltzmann constant").expect("a listed constant");
    let length = (hbar * g / c.powi(3)).sqrt();
    let mass = (hbar * c / g).sqrt();
    let time = length / c;
    let energy = mass * c * c;
    vec![
        ("Planck length", length, "m"),
        ("Planck mass", mass, "kg"),
        ("Planck time", time, "s"),
        ("Planck energy", energy, "J"),
        ("Planck temperature", energy / kb, "K"),
    ]
}

// ---------------------------------------------------------------------------
// dimensional checking of a symbolic formula
// ---------------------------------------------------------------------------

/// The dimension of a symbolic expression, given the dimension of every
/// variable in it.
///
/// This is the check a physicist runs before believing an algebra step,
/// done mechanically. It is worth having as code because the two rules
/// it enforces are the ones a hand derivation drops:
///
/// * **Every term of a sum has to have the same dimension.** A length
///   plus a time is not a longer length, it is a mistake, and it is the
///   mistake a dropped factor produces.
/// * **A transcendental function's argument has to be dimensionless.**
///   `sin`, `exp` and `ln` are defined by their power series, and a
///   series adds `x` to `x^3` to `x^5`, so `x` can only be a pure
///   number. `exp(-t/tau)` is meaningful and `exp(-t)` is not, and the
///   difference is the missing timescale.
///
/// Neither rule can be checked by evaluating the formula: both sides of
/// `x + v` are finite floats. They are properties of the expression, and
/// this walks the expression.
///
/// `var_dims` maps each variable name to its dimension; the first
/// matching entry wins. Numeric literals are dimensionless.
///
/// # Exponents
///
/// `Pow(b, e)` needs `e` to be a literal number, because the dimension
/// of `b^e` depends on the *value* of `e` and not on its dimension.
/// When the base is dimensionless the exponent may be anything
/// dimensionless -- `2^n` is a pure number whatever `n` is -- but when
/// the base carries dimensions the exponent must be a literal -- an
/// [`Expr::Rat`] or an [`Expr::Const`] -- and the base's exponents must
/// all be divisible by the literal's denominator.
///
/// A `Const` is read as the dyadic rational it exactly is, which needs
/// no guessing: `0.5` is one half, so `Pow(x, 0.5)` is a square root
/// and behaves like one. `0.1` is not one tenth, it is the
/// power-of-two fraction the float holds, and no dimension is
/// divisible by that denominator, so `l^0.1` is reported as a root
/// that does not exist rather than quietly rounded into one that
/// does.
///
/// # Errors
///
/// [`DimError::Mismatch`] when the terms of a sum disagree or a
/// transcendental is handed something dimensioned;
/// [`DimError::UnknownVar`] for a variable missing from `var_dims`;
/// [`DimError::NotAPerfectRoot`] for a root that does not come out
/// exactly; [`DimError::Malformed`] for an exponent that is not a
/// literal; [`DimError::Overflow`] if an exponent leaves `i8`.
///
/// # Examples
///
/// ```
/// use rust_physics_engine::exact::symbolic::Expr;
/// use rust_physics_engine::units::dimensional::dimensional_check_formula;
/// use rust_physics_engine::units::quantity::Dim;
///
/// let vars = [("l", Dim::LENGTH), ("g", Dim::new(1, 0, -2, 0, 0, 0, 0))];
/// // The pendulum period really is a time.
/// let over_g = Expr::pow(Expr::var("g"), Expr::c(-1.0));
/// let period = Expr::Sqrt(Box::new(Expr::mul(vec![Expr::var("l"), over_g])));
/// assert_eq!(dimensional_check_formula(&period, &vars).unwrap(), Dim::TIME);
/// // And sin(l) is not anything.
/// let bad = Expr::Sin(Box::new(Expr::var("l")));
/// assert!(dimensional_check_formula(&bad, &vars).is_err());
/// ```
pub fn dimensional_check_formula(
    expr: &Expr,
    var_dims: &[(&str, Dim)],
) -> Result<Dim, DimError> {
    // A transcendental takes a pure number and returns one. Sharing the
    // check keeps the eight of them honest about the same rule.
    fn pure(arg: &Expr, var_dims: &[(&str, Dim)]) -> Result<Dim, DimError> {
        let d = dimensional_check_formula(arg, var_dims)?;
        if d.is_dimensionless() {
            Ok(Dim::NONE)
        } else {
            Err(DimError::Mismatch { expected: Dim::NONE, found: d })
        }
    }

    match expr {
        Expr::Const(_) | Expr::Rat(_) => Ok(Dim::NONE),
        Expr::Var(name) => var_dims
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| *d)
            .ok_or_else(|| DimError::UnknownVar(name.clone())),
        Expr::Neg(x) | Expr::Abs(x) => dimensional_check_formula(x, var_dims),
        Expr::Add(terms) => {
            // Zero is the additive identity of every dimension at once,
            // so a term that is literally zero joins any sum. This is
            // not a convenience: `diff` leaves `0 * t` sitting beside
            // `v * 1` in the product rule's output, and refusing that
            // sum would make the checker useless on any expression that
            // had been differentiated. Its subexpressions are still
            // checked -- only its claim on the sum's dimension is
            // waived.
            let mut head: Option<Dim> = None;
            for t in terms {
                let d = dimensional_check_formula(t, var_dims)?;
                if is_literal_zero(t) {
                    continue;
                }
                match head {
                    None => head = Some(d),
                    Some(h) if d != h => {
                        return Err(DimError::Mismatch { expected: h, found: d })
                    }
                    Some(_) => {}
                }
            }
            // An empty sum, or one of nothing but zeros, is zero.
            Ok(head.unwrap_or(Dim::NONE))
        }
        Expr::Mul(factors) => {
            let mut out = Dim::NONE;
            for f in factors {
                out = out.mul(&dimensional_check_formula(f, var_dims)?)?;
            }
            Ok(out)
        }
        Expr::Pow(base, exponent) => {
            let db = dimensional_check_formula(base, var_dims)?;
            let de = dimensional_check_formula(exponent, var_dims)?;
            if !de.is_dimensionless() {
                // A dimensioned exponent is meaningless whatever the
                // base is: x^(1 m) has no reading at all.
                return Err(DimError::Mismatch { expected: Dim::NONE, found: de });
            }
            if db.is_dimensionless() {
                // 2^n is a pure number for any n, so the exponent does
                // not have to be a literal here.
                return Ok(Dim::NONE);
            }
            let r = literal_rational(exponent)
                .ok_or(DimError::Malformed("a dimensioned base needs a literal rational exponent"))?;
            let (num, den) = (
                r.num.to_i64().ok_or(DimError::Overflow)?,
                r.den.to_i64().ok_or(DimError::Overflow)?,
            );
            let mut out = [0i8; 7];
            for (slot, e) in out.iter_mut().zip(db.exponents()) {
                // The root has to come out exactly. m^3 to the power 1/2
                // is not a dimension, it is a sign that the formula is
                // wrong.
                if (e as i64) % den != 0 {
                    return Err(DimError::NotAPerfectRoot(db));
                }
                let scaled = (e as i64 / den).checked_mul(num).ok_or(DimError::Overflow)?;
                *slot = i8::try_from(scaled).map_err(|_| DimError::Overflow)?;
            }
            Ok(Dim::new(out[0], out[1], out[2], out[3], out[4], out[5], out[6]))
        }
        Expr::Sqrt(x) => dimensional_check_formula(x, var_dims)?.sqrt(),
        Expr::Sin(x)
        | Expr::Cos(x)
        | Expr::Tan(x)
        | Expr::Exp(x)
        | Expr::Ln(x)
        | Expr::Atan(x)
        | Expr::Sinh(x)
        | Expr::Cosh(x) => pure(x, var_dims),
    }
}

/// Whether an expression is *syntactically* zero, which is the only
/// case in which its dimension may be ignored.
///
/// Conservative on purpose: it recognises a literal zero, a product
/// with a zero factor, and a sum of nothing but those. It never tries
/// to decide that something cancels, because a wrong yes here would
/// wave a real dimension error through.
fn is_literal_zero(e: &Expr) -> bool {
    match e {
        Expr::Const(c) => *c == 0.0,
        Expr::Rat(r) => r.is_zero(),
        Expr::Neg(x) => is_literal_zero(x),
        Expr::Mul(fs) => fs.iter().any(is_literal_zero),
        Expr::Add(ts) => !ts.is_empty() && ts.iter().all(is_literal_zero),
        _ => false,
    }
}

/// An expression's value as an exact rational, when it is written as a
/// literal. Deliberately narrow: an `f64` that is not a whole number is
/// refused rather than approximated.
fn literal_rational(e: &Expr) -> Option<Rational> {
    match e {
        Expr::Rat(r) => Some(r.clone()),
        // Every finite f64 is a dyadic rational, so this is its exact
        // value and not an approximation of it: 0.5 is one half, and
        // 0.1 is the power-of-two fraction the float actually holds --
        // whose denominator no dimension is divisible by, so `l^0.1`
        // comes back as a root that does not exist rather than as a
        // rounded one that does.
        Expr::Const(c) => Rational::from_f64_exact(*c),
        Expr::Neg(inner) => literal_rational(inner).map(|r| r.neg()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::quantity::Quantity;

    /// Density, speed, length, dynamic viscosity: the pipe-flow problem
    /// whose one dimensionless group is Reynolds.
    pub(super) fn pipe_flow() -> Vec<Dim> {
        vec![
            Dim::new(-3, 1, 0, 0, 0, 0, 0),
            Dim::new(1, 0, -1, 0, 0, 0, 0),
            Dim::LENGTH,
            Dim::new(-1, 1, -1, 0, 0, 0, 0),
        ]
    }

    #[test]
    fn buckingham_returns_variables_minus_rank_groups() {
        // Four quantities built from three independent dimensions leave
        // exactly one group, and it is Reynolds up to a power.
        let dims = super::tests::pipe_flow();
        let groups = buckingham_pi(&dims).unwrap();
        assert_eq!(groups.len(), 1, "pipe flow should have one group");
        assert!(is_dimensionless_group(&dims, &groups[0]).unwrap());
        // The exponents are (-1, -1, -1, 1) or its negative, which is
        // the reciprocal of rho v L / mu.
        let e: Vec<f64> = groups[0].iter().map(|r| r.to_f64()).collect();
        let sign = if e[3] > 0.0 { 1.0 } else { -1.0 };
        for (got, want) in e.iter().zip([-1.0, -1.0, -1.0, 1.0]) {
            assert!((got * sign - want).abs() < 1e-12, "exponents were {e:?}");
        }
        // Adding a quantity whose dimension is already spanned adds
        // exactly one group.
        let mut more = dims.clone();
        more.push(Dim::new(1, 0, -2, 0, 0, 0, 0));
        assert_eq!(buckingham_pi(&more).unwrap().len(), 2);
        // Independent base dimensions alone give no groups at all.
        let bases = vec![Dim::LENGTH, Dim::MASS, Dim::TIME];
        assert!(buckingham_pi(&bases).unwrap().is_empty());
        // And repeating one gives a group immediately.
        let repeated = vec![Dim::LENGTH, Dim::LENGTH];
        let groups = buckingham_pi(&repeated).unwrap();
        assert_eq!(groups.len(), 1);
        assert!(is_dimensionless_group(&repeated, &groups[0]).unwrap());
        // Purely dimensionless quantities are each their own group.
        let none = vec![Dim::NONE, Dim::NONE, Dim::NONE];
        assert_eq!(buckingham_pi(&none).unwrap().len(), 3);
        assert!(buckingham_pi(&[]).is_err());
    }

    #[test]
    fn the_groups_cancel_exactly_rather_than_nearly() {
        // The check is against zero in exact rational arithmetic. A
        // group whose dimensions cancelled to 1e-16 would not be a
        // group, and in floating point there would be no way to tell.
        let dims = crate::units::dimensional::tests::pipe_flow();
        let groups = buckingham_pi(&dims).unwrap();
        for g in &groups {
            assert!(is_dimensionless_group(&dims, g).unwrap());
            // Every exponent came back as an exact rational.
            assert!(g.iter().all(|r| r.to_f64().is_finite()));
        }
        // A vector that is not in the null space is rejected, and the
        // rejection is exact too.
        let wrong = vec![Rational::one(), Rational::zero(), Rational::zero(), Rational::zero()];
        assert!(!is_dimensionless_group(&dims, &wrong).unwrap());
        assert!(is_dimensionless_group(&dims, &wrong[..2]).is_err());
    }

    #[test]
    fn the_named_groups_are_listed_with_what_they_compare() {
        let groups = dimensionless_groups_named();
        assert!(groups.len() >= 12);
        for (name, formula, meaning) in &groups {
            assert!(!name.is_empty() && !formula.is_empty() && !meaning.is_empty());
        }
        let names: Vec<&str> = groups.iter().map(|(n, _, _)| *n).collect();
        for wanted in ["Reynolds", "Froude", "Mach", "Prandtl", "Rayleigh", "Weber"] {
            assert!(names.contains(&wanted), "{wanted} is missing");
        }
        // No duplicates.
        let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn natural_units_collapse_a_dimension_to_one_power_of_energy() {
        // With hbar = c = 1 a mass is an energy, a length and a time are
        // inverse energies.
        assert_eq!(natural_units_power(Dim::MASS).unwrap(), 1);
        assert_eq!(natural_units_power(Dim::LENGTH).unwrap(), -1);
        assert_eq!(natural_units_power(Dim::TIME).unwrap(), -1);
        assert_eq!(natural_units_power(Dim::NONE).unwrap(), 0);
        // Energy itself: m^2 kg s^-2 gives 1 - 2 + 2 = 1.
        assert_eq!(natural_units_power(Quantity::joules(1.0).dim).unwrap(), 1);
        // A speed is dimensionless, since c is one.
        assert_eq!(natural_units_power(Dim::new(1, 0, -1, 0, 0, 0, 0)).unwrap(), 0);
        // Electromagnetic and thermal dimensions need conventions this
        // does not choose, so they are refused rather than guessed.
        assert!(natural_units_power(Dim::CURRENT).is_err());
        assert!(natural_units_power(Dim::TEMPERATURE).is_err());
        assert!(natural_units_power(Dim::AMOUNT).is_err());
        assert!(natural_units_power(Dim::LUMINOUS).is_err());
        // The magnitudes are the standard ones.
        let kg = natural_units_convert(1.0, Dim::MASS).unwrap();
        assert!((kg / 5.609_588e35 - 1.0).abs() < 1e-5, "a kilogram came to {kg} eV");
        let m = natural_units_convert(1.0, Dim::LENGTH).unwrap();
        assert!((m / 5.067_731e6 - 1.0).abs() < 1e-5, "a metre came to {m} inverse eV");
        let s = natural_units_convert(1.0, Dim::TIME).unwrap();
        assert!((s / 1.519_267e15 - 1.0).abs() < 1e-5, "a second came to {s} inverse eV");
        // An electron volt of energy is one electron volt, exactly the
        // definition, which is the consistency check that ties the three
        // factors together.
        let ev = crate::units::quantity::codata("elementary charge").unwrap();
        let one = natural_units_convert(ev, Quantity::joules(1.0).dim).unwrap();
        assert!((one - 1.0).abs() < 1e-9, "an electron volt came to {one} eV");
        // The speed of light is one.
        let c = crate::units::quantity::codata("speed of light").unwrap();
        let unity = natural_units_convert(c, Dim::new(1, 0, -1, 0, 0, 0, 0)).unwrap();
        assert!((unity - 1.0).abs() < 1e-9, "c came to {unity}");
    }

    #[test]
    fn the_planck_units_satisfy_their_own_definitions() {
        let units = planck_units();
        let get = |n: &str| units.iter().find(|(m, _, _)| *m == n).map(|(_, v, _)| *v).unwrap();
        let hbar = crate::units::quantity::codata("reduced Planck constant").unwrap();
        let c = crate::units::quantity::codata("speed of light").unwrap();
        let g = crate::units::quantity::codata("gravitational constant").unwrap();
        let (length, mass, time) = (get("Planck length"), get("Planck mass"), get("Planck time"));
        // l_P = c t_P.
        assert!((length - c * time).abs() < 1e-12 * length);
        // l_P m_P = hbar / c, which is what "the Compton wavelength
        // equals the Schwarzschild radius" amounts to.
        assert!((length * mass - hbar / c).abs() < 1e-9 * length * mass);
        // E_P = m_P c^2.
        assert!((get("Planck energy") - mass * c * c).abs() < 1e-12 * get("Planck energy"));
        // And the Schwarzschild radius of the Planck mass is twice the
        // Planck length, which is the statement that gravity and quantum
        // mechanics meet there.
        let schwarzschild = 2.0 * g * mass / (c * c);
        assert!(
            (schwarzschild - 2.0 * length).abs() < 1e-9 * schwarzschild,
            "the Schwarzschild radius came to {schwarzschild}"
        );
        // The published values, to the precision G is known to.
        assert!((length / 1.616_255e-35 - 1.0).abs() < 1e-5);
        assert!((mass / 2.176_434e-8 - 1.0).abs() < 1e-5);
        assert!((time / 5.391_247e-44 - 1.0).abs() < 1e-5);
        assert!((get("Planck temperature") / 1.416_784e32 - 1.0).abs() < 1e-5);
    }
}


#[cfg(test)]
mod formula_tests {
    use super::*;
    use crate::units::quantity::Quantity;

    const VELOCITY: Dim = Dim::new(1, 0, -1, 0, 0, 0, 0);
    const ACCEL: Dim = Dim::new(1, 0, -2, 0, 0, 0, 0);
    const FORCE: Dim = Dim::new(1, 1, -2, 0, 0, 0, 0);
    const ENERGY: Dim = Dim::new(2, 1, -2, 0, 0, 0, 0);
    const FREQUENCY: Dim = Dim::new(0, 0, -1, 0, 0, 0, 0);

    fn vars() -> Vec<(&'static str, Dim)> {
        vec![
            ("m", Dim::MASS),
            ("v", VELOCITY),
            ("l", Dim::LENGTH),
            ("x", Dim::LENGTH),
            ("g", ACCEL),
            ("t", Dim::TIME),
            ("tau", Dim::TIME),
            ("omega", FREQUENCY),
            ("k_b", Dim::new(2, 1, -2, 0, -1, 0, 0)),
            ("temp", Dim::TEMPERATURE),
            ("n", Dim::NONE),
        ]
    }

    /// `a / b`, since the expression type has no division node.
    fn over(a: Expr, b: Expr) -> Expr {
        Expr::mul(vec![a, Expr::pow(b, Expr::c(-1.0))])
    }

    #[test]
    fn a_sum_of_unlike_terms_is_refused_and_of_like_terms_is_not() {
        let v = vars();
        // The two energies of a falling body. Both terms are energies,
        // so the sum is one too.
        let kinetic = Expr::mul(vec![
            Expr::c(0.5),
            Expr::var("m"),
            Expr::pow(Expr::var("v"), Expr::c(2.0)),
        ]);
        let potential = Expr::mul(vec![Expr::var("m"), Expr::var("g"), Expr::var("l")]);
        let total = Expr::add(vec![kinetic.clone(), potential.clone()]);
        assert_eq!(dimensional_check_formula(&total, &v).unwrap(), ENERGY);

        // Drop the velocity's square -- the single commonest slip in a
        // hand derivation -- and the sum stops being a sum of energies.
        let slipped = Expr::mul(vec![Expr::c(0.5), Expr::var("m"), Expr::var("v")]);
        let bad = Expr::add(vec![slipped, potential]);
        match dimensional_check_formula(&bad, &v) {
            Err(DimError::Mismatch { expected, found }) => {
                assert_eq!(expected, Dim::new(1, 1, -1, 0, 0, 0, 0));
                assert_eq!(found, ENERGY);
            }
            other => panic!("expected a mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_transcendental_argument_must_be_a_pure_number() {
        let v = vars();
        // Every one of these is defined by a power series that adds x to
        // x^3, so a dimensioned argument is refused by all of them.
        let dimensioned = [
            Expr::Sin(Box::new(Expr::var("t"))),
            Expr::Cos(Box::new(Expr::var("t"))),
            Expr::Tan(Box::new(Expr::var("t"))),
            Expr::Exp(Box::new(Expr::var("t"))),
            Expr::Ln(Box::new(Expr::var("t"))),
            Expr::Atan(Box::new(Expr::var("t"))),
            Expr::Sinh(Box::new(Expr::var("t"))),
            Expr::Cosh(Box::new(Expr::var("t"))),
        ];
        for e in &dimensioned {
            assert_eq!(
                dimensional_check_formula(e, &v),
                Err(DimError::Mismatch { expected: Dim::NONE, found: Dim::TIME }),
                "a dimensioned argument slipped through {e:?}"
            );
        }
        // Supply the missing timescale and each of them is fine, and
        // returns a pure number.
        let phase = Expr::mul(vec![Expr::var("omega"), Expr::var("t")]);
        for e in &dimensioned {
            let fixed = e.substitute("t", &phase);
            assert_eq!(dimensional_check_formula(&fixed, &v).unwrap(), Dim::NONE);
        }
        // exp(-t/tau) is the decay every physical model writes; exp(-t)
        // is the same model with its timescale lost.
        let decay = Expr::Exp(Box::new(Expr::Neg(Box::new(over(
            Expr::var("t"),
            Expr::var("tau"),
        )))));
        assert_eq!(dimensional_check_formula(&decay, &v).unwrap(), Dim::NONE);
    }

    #[test]
    fn a_power_is_exact_or_it_is_an_error() {
        let v = vars();
        let l_over_g = over(Expr::var("l"), Expr::var("g")); // s^2
        assert_eq!(dimensional_check_formula(&l_over_g, &v).unwrap(), Dim::new(0, 0, 2, 0, 0, 0, 0));

        // The half power of s^2 is exactly s, by both routes.
        let half = Expr::pow(l_over_g.clone(), Expr::Rat(Rational::from_i64(1, 2)));
        assert_eq!(dimensional_check_formula(&half, &v).unwrap(), Dim::TIME);
        let root = Expr::Sqrt(Box::new(l_over_g.clone()));
        assert_eq!(dimensional_check_formula(&root, &v).unwrap(), Dim::TIME);

        // The third power of s^2 is not a dimension: two is not
        // divisible by three, and there is no rounding that makes it so.
        let third = Expr::pow(l_over_g, Expr::Rat(Rational::from_i64(1, 3)));
        assert_eq!(
            dimensional_check_formula(&third, &v),
            Err(DimError::NotAPerfectRoot(Dim::new(0, 0, 2, 0, 0, 0, 0)))
        );

        // A dimensionless base takes any dimensionless exponent, literal
        // or not, because 2^n is a pure number whatever n is.
        let two_to_n = Expr::pow(Expr::c(2.0), Expr::var("n"));
        assert_eq!(dimensional_check_formula(&two_to_n, &v).unwrap(), Dim::NONE);
        // But not a dimensioned one: 2^t has no reading at all.
        let two_to_t = Expr::pow(Expr::c(2.0), Expr::var("t"));
        assert_eq!(
            dimensional_check_formula(&two_to_t, &v),
            Err(DimError::Mismatch { expected: Dim::NONE, found: Dim::TIME })
        );
        // A dimensioned base needs the exponent's value, not just its
        // dimension, so a symbolic exponent is refused rather than
        // guessed at -- as is a fractional f64, which would have to be
        // rounded into a rational to be used.
        assert!(matches!(
            dimensional_check_formula(&Expr::pow(Expr::var("l"), Expr::var("n")), &v),
            Err(DimError::Malformed(_))
        ));
        // A Const exponent is read as the dyadic rational it exactly
        // is, so a half power is a square root and behaves like one --
        // which matters because `diff` writes the derivative of a
        // square root as exactly that.
        let sq = Expr::mul(vec![Expr::var("l"), Expr::var("l")]);
        assert_eq!(
            dimensional_check_formula(&Expr::pow(sq, Expr::c(0.5)), &v).unwrap(),
            Dim::LENGTH
        );
        assert_eq!(
            dimensional_check_formula(&Expr::pow(Expr::var("l"), Expr::c(0.5)), &v),
            Err(DimError::NotAPerfectRoot(Dim::LENGTH))
        );
        // And a tenth is not one tenth: it is the float's own binary
        // fraction, whose denominator divides no dimension at all.
        assert!(matches!(
            dimensional_check_formula(&Expr::pow(Expr::var("l"), Expr::c(0.1)), &v),
            Err(DimError::NotAPerfectRoot(_))
        ));
    }

    #[test]
    fn the_checker_agrees_with_quantity_arithmetic() {
        // Two independent implementations of the same algebra: this one
        // walks an expression tree over i8 exponents, Quantity carries
        // the dimension along with a value through method calls. They
        // have no code in common, so agreeing is evidence.
        let v = vars();
        let m = Quantity::new(2.5, Dim::MASS);
        let vel = Quantity::new(3.0, VELOCITY);
        let len = Quantity::new(1.5, Dim::LENGTH);
        let acc = Quantity::new(9.81, ACCEL);

        let by_quantity = m
            .mul(&vel)
            .unwrap()
            .mul(&vel)
            .unwrap()
            .add(&m.mul(&acc).unwrap().mul(&len).unwrap())
            .unwrap();
        let by_formula = Expr::add(vec![
            Expr::mul(vec![
                Expr::var("m"),
                Expr::pow(Expr::var("v"), Expr::c(2.0)),
            ]),
            Expr::mul(vec![Expr::var("m"), Expr::var("g"), Expr::var("l")]),
        ]);
        assert_eq!(dimensional_check_formula(&by_formula, &v).unwrap(), by_quantity.dim);

        // And they agree on the refusal, not just on the success.
        assert!(m.add(&vel).is_err());
        assert!(dimensional_check_formula(
            &Expr::add(vec![Expr::var("m"), Expr::var("v")]),
            &v
        )
        .is_err());

        // A root that does not come out exactly is refused by both.
        assert!(Quantity::new(4.0, Dim::LENGTH).sqrt().is_err());
        assert!(dimensional_check_formula(&Expr::Sqrt(Box::new(Expr::var("l"))), &v).is_err());
    }

    #[test]
    fn differentiating_divides_the_dimension_by_the_variables() {
        // d/dt has the dimension of one over a time, so the derivative's
        // dimension is the function's divided by the variable's. That is
        // a theorem about the limit, and it holds term by term for every
        // rule the differentiator implements -- which makes it a check
        // on the differentiator as much as on the checker.
        let v = vars();
        let cases: Vec<(Expr, Dim)> = vec![
            // A displacement under constant acceleration, of which the
            // derivative is a velocity and the second a acceleration.
            (
                Expr::add(vec![
                    Expr::mul(vec![Expr::var("v"), Expr::var("t")]),
                    Expr::mul(vec![
                        Expr::c(0.5),
                        Expr::var("g"),
                        Expr::pow(Expr::var("t"), Expr::c(2.0)),
                    ]),
                ]),
                Dim::LENGTH,
            ),
            // A damped oscillation: still a length, however written.
            (
                Expr::mul(vec![
                    Expr::var("l"),
                    Expr::Exp(Box::new(Expr::Neg(Box::new(over(
                        Expr::var("t"),
                        Expr::var("tau"),
                    ))))),
                    Expr::Sin(Box::new(Expr::mul(vec![Expr::var("omega"), Expr::var("t")]))),
                ]),
                Dim::LENGTH,
            ),
            // A pure phase, whose derivative is a frequency.
            (
                Expr::Atan(Box::new(Expr::mul(vec![Expr::var("omega"), Expr::var("t")]))),
                Dim::NONE,
            ),
        ];
        for (e, want) in cases {
            assert_eq!(dimensional_check_formula(&e, &v).unwrap(), want);
            let mut d = e;
            let mut expected = want;
            for order in 1..=2 {
                d = d.diff("t");
                expected = expected.div(&Dim::TIME).unwrap();
                assert_eq!(
                    dimensional_check_formula(&d, &v).unwrap(),
                    expected,
                    "derivative of order {order} came out wrong"
                );
            }
        }
    }

    #[test]
    fn buckinghams_groups_check_out_as_formulas() {
        // buckingham_pi finds its groups as an exact null space over the
        // rationals; dimensional_check_formula multiplies i8 exponents
        // along an expression tree. Feeding one into the other closes
        // the loop between them.
        let dims = crate::units::dimensional::tests::pipe_flow();
        let names = ["rho", "u", "d", "mu"];
        let var_dims: Vec<(&str, Dim)> =
            names.iter().copied().zip(dims.iter().copied()).collect();
        let groups = buckingham_pi(&dims).unwrap();
        assert_eq!(groups.len(), 1, "pipe flow has exactly one group");
        for group in &groups {
            // Buckingham's exponents are rationals; write the group as
            // the product of powers it stands for.
            let factors: Vec<Expr> = group
                .iter()
                .zip(names.iter())
                .map(|(e, n)| Expr::pow(Expr::var(n), Expr::Rat(e.clone())))
                .collect();
            let expr = Expr::mul(factors);
            assert_eq!(
                dimensional_check_formula(&expr, &var_dims).unwrap(),
                Dim::NONE,
                "a group the theorem calls dimensionless is not"
            );
        }
    }

    #[test]
    fn textbook_formulas_come_out_with_their_textbook_dimensions() {
        // Each of these is a relation somebody has to get right; the
        // dimension is the cheapest part of it to check.
        let v = vars();
        let cases: Vec<(&str, Expr, Dim)> = vec![
            ("Newton's second law", Expr::mul(vec![Expr::var("m"), Expr::var("g")]), FORCE),
            (
                "the equipartition energy",
                Expr::mul(vec![Expr::c(1.5), Expr::var("k_b"), Expr::var("temp")]),
                ENERGY,
            ),
            (
                "the pendulum period",
                Expr::Sqrt(Box::new(over(Expr::var("l"), Expr::var("g")))),
                Dim::TIME,
            ),
            (
                "the thermal de Broglie speed",
                Expr::Sqrt(Box::new(over(
                    Expr::mul(vec![Expr::var("k_b"), Expr::var("temp")]),
                    Expr::var("m"),
                ))),
                VELOCITY,
            ),
            (
                "an oscillator's frequency from its period",
                over(Expr::c(1.0), Expr::var("tau")),
                FREQUENCY,
            ),
        ];
        for (name, e, want) in cases {
            assert_eq!(dimensional_check_formula(&e, &v).unwrap(), want, "{name}");
        }
    }

    #[test]
    fn zero_joins_a_sum_of_any_dimension_but_nothing_else_does() {
        let v = vars();
        // Zero is the additive identity of every dimension, so it is
        // the one term that may sit beside anything.
        let with_zero = Expr::add(vec![Expr::var("l"), Expr::c(0.0)]);
        assert_eq!(dimensional_check_formula(&with_zero, &v).unwrap(), Dim::LENGTH);
        let zero_product = Expr::add(vec![
            Expr::var("l"),
            Expr::mul(vec![Expr::c(0.0), Expr::var("t")]),
        ]);
        assert_eq!(dimensional_check_formula(&zero_product, &v).unwrap(), Dim::LENGTH);

        // One is not zero, and admitting it would defeat the check.
        let with_one = Expr::add(vec![Expr::var("l"), Expr::c(1.0)]);
        assert!(dimensional_check_formula(&with_one, &v).is_err());
        // Nor does the waiver reach inside the zero term: a nonsense
        // subexpression is still nonsense multiplied by zero.
        let zero_times_nonsense =
            Expr::mul(vec![Expr::c(0.0), Expr::Sin(Box::new(Expr::var("t")))]);
        assert!(dimensional_check_formula(
            &Expr::add(vec![Expr::var("l"), zero_times_nonsense]),
            &v
        )
        .is_err());
    }

    #[test]
    fn a_variable_with_no_dimension_given_is_named_in_the_error() {
        // Silently treating an unknown symbol as dimensionless would
        // make the check pass on exactly the formulas it exists to
        // catch, so it is an error, and it says which symbol.
        let v = vars();
        let e = Expr::add(vec![Expr::var("l"), Expr::var("height")]);
        assert_eq!(
            dimensional_check_formula(&e, &v),
            Err(DimError::UnknownVar("height".to_string()))
        );
    }
}

