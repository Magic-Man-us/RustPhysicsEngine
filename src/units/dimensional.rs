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

use crate::exact::rational::Rational;
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
    let power = natural_units_power(dim)?;
    let _ = power;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::quantity::Quantity;

    /// Density, speed, length, dynamic viscosity: the pipe-flow problem
    /// whose one dimensionless group is Reynolds.
    fn pipe_flow() -> Vec<Dim> {
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
        let dims = pipe_flow();
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
        let dims = pipe_flow();
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
