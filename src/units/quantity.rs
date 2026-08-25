//! Values that carry their dimensions.
//!
//! # Why a number alone is not a measurement
//!
//! The two most expensive unit mistakes on record -- the Mars Climate
//! Orbiter's pound-seconds fed to a newton-second interface, and the
//! Gimli Glider's kilograms of fuel loaded as pounds -- were both
//! arithmetic that a computer performed correctly on numbers that meant
//! something other than what the receiving code assumed. Neither was a
//! rounding error and neither would have been caught by testing the
//! arithmetic.
//!
//! A [`Quantity`] carries seven small integers alongside its value: the
//! exponents of metre, kilogram, second, ampere, kelvin, mole and
//! candela. Addition then checks that the two exponent vectors agree and
//! refuses if they do not, multiplication adds them, and taking a square
//! root fails unless every one of them is even. None of this is
//! approximate -- the exponents are integers and the checks are exact.
//!
//! # The gram is the prefixable unit, not the kilogram
//!
//! The SI base unit of mass is the kilogram, which is the only base unit
//! whose name already contains a prefix. The prefix system therefore
//! attaches to the *gram*: `mg` is a milligram and not a milli-kilogram,
//! and `kg` parses here as kilo applied to gram. The unit table stores
//! the gram at `1e-3`, which makes `kg` come out at exactly one and the
//! oddity disappear.
//!
//! # Parsing a unit is ambiguous and the rule has to be stated
//!
//! `m` is both the metre and the milli prefix, `T` is both the tesla and
//! tera, `min` starts with the milli prefix followed by `in`. The rule
//! used is: try the whole token as a unit name first, and only if that
//! fails split off a prefix. So `m` is a metre, `mm` is a millimetre,
//! `min` is a minute, and `T` is a tesla. It is a rule rather than a
//! deduction, and any other rule would give different answers for the
//! same strings.

use std::fmt;

/// What can go wrong when dimensions meet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimError {
    /// Two quantities that had to agree did not.
    Mismatch {
        /// The dimension expected.
        expected: Dim,
        /// The dimension found.
        found: Dim,
    },
    /// A root was taken of a dimension that does not have one.
    NotAPerfectRoot(Dim),
    /// An exponent left the range a signed byte can hold.
    Overflow,
    /// A unit name was not recognised.
    UnknownUnit(String),
    /// The text was not a quantity.
    Malformed(&'static str),
}

impl fmt::Display for DimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimError::Mismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            DimError::NotAPerfectRoot(d) => write!(f, "{d} has no exact root"),
            DimError::Overflow => write!(f, "a dimension exponent overflowed"),
            DimError::UnknownUnit(u) => write!(f, "unknown unit: {u}"),
            DimError::Malformed(m) => write!(f, "malformed quantity: {m}"),
        }
    }
}

impl std::error::Error for DimError {}

/// The seven SI base exponents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Dim {
    /// Metre.
    pub m: i8,
    /// Kilogram.
    pub kg: i8,
    /// Second.
    pub s: i8,
    /// Ampere.
    pub a: i8,
    /// Kelvin.
    pub k: i8,
    /// Mole.
    pub mol: i8,
    /// Candela.
    pub cd: i8,
}

impl Dim {
    /// A pure number.
    pub const NONE: Dim = Dim { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 };
    /// Length.
    pub const LENGTH: Dim = Dim { m: 1, ..Dim::NONE };
    /// Mass.
    pub const MASS: Dim = Dim { kg: 1, ..Dim::NONE };
    /// Time.
    pub const TIME: Dim = Dim { s: 1, ..Dim::NONE };
    /// Electric current.
    pub const CURRENT: Dim = Dim { a: 1, ..Dim::NONE };
    /// Thermodynamic temperature.
    pub const TEMPERATURE: Dim = Dim { k: 1, ..Dim::NONE };
    /// Amount of substance.
    pub const AMOUNT: Dim = Dim { mol: 1, ..Dim::NONE };
    /// Luminous intensity.
    pub const LUMINOUS: Dim = Dim { cd: 1, ..Dim::NONE };

    /// Builds a dimension from its seven exponents.
    pub const fn new(m: i8, kg: i8, s: i8, a: i8, k: i8, mol: i8, cd: i8) -> Dim {
        Dim { m, kg, s, a, k, mol, cd }
    }

    /// The exponents as an array, in the order metre, kilogram, second,
    /// ampere, kelvin, mole, candela.
    pub const fn exponents(&self) -> [i8; 7] {
        [self.m, self.kg, self.s, self.a, self.k, self.mol, self.cd]
    }

    /// Whether every exponent is zero.
    pub fn is_dimensionless(&self) -> bool {
        *self == Dim::NONE
    }

    /// Adds two exponent vectors, which is what multiplying does.
    ///
    /// # Errors
    ///
    /// [`DimError::Overflow`] if an exponent leaves `i8`.
    pub fn mul(&self, other: &Dim) -> Result<Dim, DimError> {
        let mut out = [0i8; 7];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.exponents()[i]
                .checked_add(other.exponents()[i])
                .ok_or(DimError::Overflow)?;
        }
        Ok(Dim::new(out[0], out[1], out[2], out[3], out[4], out[5], out[6]))
    }

    /// Subtracts two exponent vectors, which is what dividing does.
    ///
    /// # Errors
    ///
    /// As [`Dim::mul`].
    pub fn div(&self, other: &Dim) -> Result<Dim, DimError> {
        let mut out = [0i8; 7];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.exponents()[i]
                .checked_sub(other.exponents()[i])
                .ok_or(DimError::Overflow)?;
        }
        Ok(Dim::new(out[0], out[1], out[2], out[3], out[4], out[5], out[6]))
    }

    /// Multiplies every exponent by `n`.
    ///
    /// # Errors
    ///
    /// As [`Dim::mul`].
    pub fn pow(&self, n: i8) -> Result<Dim, DimError> {
        let mut out = [0i8; 7];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.exponents()[i].checked_mul(n).ok_or(DimError::Overflow)?;
        }
        Ok(Dim::new(out[0], out[1], out[2], out[3], out[4], out[5], out[6]))
    }

    /// Halves every exponent.
    ///
    /// # Errors
    ///
    /// [`DimError::NotAPerfectRoot`] unless every exponent is even. A
    /// dimension with an odd exponent has no square root at all -- there
    /// is no such thing as the square root of a metre -- so this is a
    /// refusal rather than a rounding decision.
    pub fn sqrt(&self) -> Result<Dim, DimError> {
        if self.exponents().iter().any(|e| e % 2 != 0) {
            return Err(DimError::NotAPerfectRoot(*self));
        }
        let e = self.exponents();
        Ok(Dim::new(e[0] / 2, e[1] / 2, e[2] / 2, e[3] / 2, e[4] / 2, e[5] / 2, e[6] / 2))
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }
        const NAMES: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];
        let mut parts = Vec::new();
        for (name, e) in NAMES.iter().zip(self.exponents()) {
            if e == 1 {
                parts.push((*name).to_string());
            } else if e != 0 {
                parts.push(format!("{name}^{e}"));
            }
        }
        write!(f, "{}", parts.join(" "))
    }
}

/// A value together with its dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    /// The magnitude, in coherent SI units.
    pub value: f64,
    /// What it measures.
    pub dim: Dim,
}

/// Declares a constructor for a unit whose SI value is the given factor.
macro_rules! unit_ctor {
    ($name:ident, $factor:expr, $dim:expr, $doc:expr) => {
        #[doc = $doc]
        pub fn $name(v: f64) -> Quantity {
            Quantity { value: v * $factor, dim: $dim }
        }
    };
}

impl Quantity {
    /// A pure number.
    pub fn number(v: f64) -> Quantity {
        Quantity { value: v, dim: Dim::NONE }
    }

    /// A value with an explicit dimension, already in SI.
    pub fn new(value: f64, dim: Dim) -> Quantity {
        Quantity { value, dim }
    }

    unit_ctor!(meters, 1.0, Dim::LENGTH, "Metres.");
    unit_ctor!(kilometers, 1e3, Dim::LENGTH, "Kilometres.");
    unit_ctor!(millimeters, 1e-3, Dim::LENGTH, "Millimetres.");
    unit_ctor!(feet, 0.304_8, Dim::LENGTH, "Feet, exactly 0.3048 m.");
    unit_ctor!(inches, 0.025_4, Dim::LENGTH, "Inches, exactly 25.4 mm.");
    unit_ctor!(miles, 1_609.344, Dim::LENGTH, "Statute miles.");
    unit_ctor!(kg, 1.0, Dim::MASS, "Kilograms.");
    unit_ctor!(grams, 1e-3, Dim::MASS, "Grams.");
    unit_ctor!(pounds, 0.453_592_37, Dim::MASS, "Pounds, exactly 0.45359237 kg.");
    unit_ctor!(seconds, 1.0, Dim::TIME, "Seconds.");
    unit_ctor!(minutes, 60.0, Dim::TIME, "Minutes.");
    unit_ctor!(hours, 3_600.0, Dim::TIME, "Hours.");
    unit_ctor!(days, 86_400.0, Dim::TIME, "Days of exactly 86400 s.");
    unit_ctor!(amperes, 1.0, Dim::CURRENT, "Amperes.");
    unit_ctor!(kelvin, 1.0, Dim::TEMPERATURE, "Kelvin.");
    unit_ctor!(moles, 1.0, Dim::AMOUNT, "Moles.");
    unit_ctor!(candela, 1.0, Dim::LUMINOUS, "Candela.");
    unit_ctor!(hertz, 1.0, Dim::new(0, 0, -1, 0, 0, 0, 0), "Hertz.");
    unit_ctor!(newtons, 1.0, Dim::new(1, 1, -2, 0, 0, 0, 0), "Newtons.");
    unit_ctor!(pascals, 1.0, Dim::new(-1, 1, -2, 0, 0, 0, 0), "Pascals.");
    unit_ctor!(joules, 1.0, Dim::new(2, 1, -2, 0, 0, 0, 0), "Joules.");
    unit_ctor!(watts, 1.0, Dim::new(2, 1, -3, 0, 0, 0, 0), "Watts.");
    unit_ctor!(coulombs, 1.0, Dim::new(0, 0, 1, 1, 0, 0, 0), "Coulombs.");
    unit_ctor!(volts, 1.0, Dim::new(2, 1, -3, -1, 0, 0, 0), "Volts.");
    unit_ctor!(farads, 1.0, Dim::new(-2, -1, 4, 2, 0, 0, 0), "Farads.");
    unit_ctor!(ohms, 1.0, Dim::new(2, 1, -3, -2, 0, 0, 0), "Ohms.");
    unit_ctor!(teslas, 1.0, Dim::new(0, 1, -2, -1, 0, 0, 0), "Teslas.");
    unit_ctor!(webers, 1.0, Dim::new(2, 1, -2, -1, 0, 0, 0), "Webers.");
    unit_ctor!(henries, 1.0, Dim::new(2, 1, -2, -2, 0, 0, 0), "Henries.");
    unit_ctor!(
        electron_volts,
        1.602_176_634e-19,
        Dim::new(2, 1, -2, 0, 0, 0, 0),
        "Electron volts, exact since the 2019 redefinition."
    );
    unit_ctor!(
        kilowatt_hours,
        3.6e6,
        Dim::new(2, 1, -2, 0, 0, 0, 0),
        "Kilowatt hours."
    );

    /// Adds two quantities.
    ///
    /// # Errors
    ///
    /// [`DimError::Mismatch`] if they do not measure the same thing.
    pub fn add(&self, other: &Quantity) -> Result<Quantity, DimError> {
        if self.dim != other.dim {
            return Err(DimError::Mismatch { expected: self.dim, found: other.dim });
        }
        Ok(Quantity { value: self.value + other.value, dim: self.dim })
    }

    /// Subtracts two quantities.
    ///
    /// # Errors
    ///
    /// As [`Quantity::add`].
    pub fn sub(&self, other: &Quantity) -> Result<Quantity, DimError> {
        if self.dim != other.dim {
            return Err(DimError::Mismatch { expected: self.dim, found: other.dim });
        }
        Ok(Quantity { value: self.value - other.value, dim: self.dim })
    }

    /// Multiplies two quantities, adding their exponents.
    ///
    /// # Errors
    ///
    /// [`DimError::Overflow`] if an exponent leaves `i8`.
    pub fn mul(&self, other: &Quantity) -> Result<Quantity, DimError> {
        Ok(Quantity { value: self.value * other.value, dim: self.dim.mul(&other.dim)? })
    }

    /// Divides two quantities, subtracting their exponents.
    ///
    /// # Errors
    ///
    /// As [`Quantity::mul`].
    pub fn div(&self, other: &Quantity) -> Result<Quantity, DimError> {
        Ok(Quantity { value: self.value / other.value, dim: self.dim.div(&other.dim)? })
    }

    /// Raises to an integer power.
    ///
    /// # Errors
    ///
    /// As [`Quantity::mul`].
    pub fn pow(&self, n: i8) -> Result<Quantity, DimError> {
        Ok(Quantity { value: self.value.powi(n as i32), dim: self.dim.pow(n)? })
    }

    /// Takes the square root.
    ///
    /// # Errors
    ///
    /// [`DimError::NotAPerfectRoot`] unless every exponent is even.
    pub fn sqrt(&self) -> Result<Quantity, DimError> {
        Ok(Quantity { value: self.value.sqrt(), dim: self.dim.sqrt()? })
    }

    /// The magnitude expressed in the named unit.
    ///
    /// # Errors
    ///
    /// [`DimError::UnknownUnit`] or [`DimError::Mismatch`] if the unit
    /// measures something else.
    pub fn to(&self, unit: &str) -> Result<f64, DimError> {
        let (factor, dim) = parse_unit(unit)?;
        if dim != self.dim {
            return Err(DimError::Mismatch { expected: self.dim, found: dim });
        }
        Ok(self.value / factor)
    }

    /// The value and its SI dimension as text.
    pub fn format_si(&self) -> String {
        format!("{} {}", self.value, self.dim)
    }
}

/// The SI prefixes, largest first so that the longest match wins.
const PREFIXES: [(&str, f64); 24] = [
    ("Q", 1e30),
    ("R", 1e27),
    ("Y", 1e24),
    ("Z", 1e21),
    ("E", 1e18),
    ("P", 1e15),
    ("T", 1e12),
    ("G", 1e9),
    ("M", 1e6),
    ("da", 1e1),
    ("k", 1e3),
    ("h", 1e2),
    ("d", 1e-1),
    ("c", 1e-2),
    ("m", 1e-3),
    ("u", 1e-6),
    ("µ", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
    ("z", 1e-21),
    ("y", 1e-24),
    ("r", 1e-27),
];

/// Every unit name recognised, with its SI factor and dimension.
///
/// The gram sits at `1e-3` rather than the kilogram at one, so that the
/// prefix system attaches where SI says it does -- see the module note.
fn base_units(name: &str) -> Option<(f64, Dim)> {
    let joule = Dim::new(2, 1, -2, 0, 0, 0, 0);
    Some(match name {
        "m" => (1.0, Dim::LENGTH),
        "g" => (1e-3, Dim::MASS),
        "s" => (1.0, Dim::TIME),
        "A" => (1.0, Dim::CURRENT),
        "K" => (1.0, Dim::TEMPERATURE),
        "mol" => (1.0, Dim::AMOUNT),
        "cd" => (1.0, Dim::LUMINOUS),
        "Hz" => (1.0, Dim::new(0, 0, -1, 0, 0, 0, 0)),
        "N" => (1.0, Dim::new(1, 1, -2, 0, 0, 0, 0)),
        "Pa" => (1.0, Dim::new(-1, 1, -2, 0, 0, 0, 0)),
        "J" => (1.0, joule),
        "W" => (1.0, Dim::new(2, 1, -3, 0, 0, 0, 0)),
        "C" => (1.0, Dim::new(0, 0, 1, 1, 0, 0, 0)),
        "V" => (1.0, Dim::new(2, 1, -3, -1, 0, 0, 0)),
        "F" => (1.0, Dim::new(-2, -1, 4, 2, 0, 0, 0)),
        "ohm" | "Ω" => (1.0, Dim::new(2, 1, -3, -2, 0, 0, 0)),
        "T" => (1.0, Dim::new(0, 1, -2, -1, 0, 0, 0)),
        "Wb" => (1.0, Dim::new(2, 1, -2, -1, 0, 0, 0)),
        "H" => (1.0, Dim::new(2, 1, -2, -2, 0, 0, 0)),
        "L" | "l" => (1e-3, Dim::new(3, 0, 0, 0, 0, 0, 0)),
        "min" => (60.0, Dim::TIME),
        "h" => (3_600.0, Dim::TIME),
        "d" => (86_400.0, Dim::TIME),
        "yr" => (31_557_600.0, Dim::TIME),
        "eV" => (1.602_176_634e-19, joule),
        "Wh" => (3_600.0, joule),
        "cal" => (4.184, joule),
        "bar" => (1e5, Dim::new(-1, 1, -2, 0, 0, 0, 0)),
        "atm" => (101_325.0, Dim::new(-1, 1, -2, 0, 0, 0, 0)),
        "ft" => (0.304_8, Dim::LENGTH),
        "in" => (0.025_4, Dim::LENGTH),
        "mi" => (1_609.344, Dim::LENGTH),
        "lb" => (0.453_592_37, Dim::MASS),
        "t" => (1e3, Dim::MASS),
        "rad" | "sr" => (1.0, Dim::NONE),
        // The bare numeral, so that "1/mol" and "1/m" read as the
        // reciprocals they are meant to be.
        "1" => (1.0, Dim::NONE),
        _ => return None,
    })
}

/// Resolves one unit token, trying the whole name before splitting off a
/// prefix -- see the module note on why the order is the rule.
fn resolve(token: &str) -> Result<(f64, Dim), DimError> {
    if let Some(found) = base_units(token) {
        return Ok(found);
    }
    for (prefix, scale) in PREFIXES {
        if let Some(rest) = token.strip_prefix(prefix) {
            if !rest.is_empty() {
                if let Some((factor, dim)) = base_units(rest) {
                    return Ok((factor * scale, dim));
                }
            }
        }
    }
    Err(DimError::UnknownUnit(token.to_string()))
}

/// Parses a unit expression such as `m/s^2`, `kg*m^2/s^3` or `J s`.
///
/// Multiplication is written `*` or a space, and division `/`. A `/`
/// applies to the single term that follows it and nothing more, so
/// `J/mol/K` is joules per mole per kelvin. Parentheses are **not**
/// supported: `J/(mol K)` is rejected as an unknown unit rather than
/// quietly parsed as something else, which is the safer of the two ways
/// to not support them.
///
/// # Errors
///
/// [`DimError::UnknownUnit`] for an unrecognised name, or
/// [`DimError::Malformed`] for a broken exponent.
pub fn parse_unit(text: &str) -> Result<(f64, Dim), DimError> {
    let text = text.trim();
    if text.is_empty() || text == "1" {
        return Ok((1.0, Dim::NONE));
    }
    let mut factor = 1.0;
    let mut dim = Dim::NONE;
    // Walk the string splitting on * and /, remembering which one
    // introduced each term.
    let mut dividing = false;
    let mut token = String::new();
    let mut terms: Vec<(bool, String)> = Vec::new();
    for c in text.chars() {
        if c == '*' || c == '/' {
            terms.push((dividing, std::mem::take(&mut token)));
            dividing = c == '/';
        } else if c.is_whitespace() {
            // Juxtaposition is multiplication: "J s" is a joule second,
            // which is how units are written everywhere outside a
            // keyboard. Leading and trailing space is not a term.
            if !token.is_empty() {
                terms.push((dividing, std::mem::take(&mut token)));
                dividing = false;
            }
        } else {
            token.push(c);
        }
    }
    terms.push((dividing, token));
    for (invert, term) in terms {
        if term.is_empty() {
            return Err(DimError::Malformed("an empty term"));
        }
        let (name, power) = match term.split_once('^') {
            Some((n, p)) => {
                (n, p.parse::<i8>().map_err(|_| DimError::Malformed("a bad exponent"))?)
            }
            None => (term.as_str(), 1),
        };
        let (f, d) = resolve(name)?;
        let signed = if invert { -power } else { power };
        factor *= f.powi(signed as i32);
        dim = dim.mul(&d.pow(signed)?)?;
    }
    Ok((factor, dim))
}

/// Parses a quantity such as `"9.81 m/s^2"` or `"3 kWh"`.
///
/// # Errors
///
/// [`DimError::Malformed`] if there is no number, and whatever
/// [`parse_unit`] reports for the rest.
pub fn parse_quantity(text: &str) -> Result<Quantity, DimError> {
    let text = text.trim();
    // The number runs until the first character that cannot continue it.
    // An `e` is only exponent notation when a digit or sign follows, so
    // that "3 eV" is three electron volts rather than a broken float.
    let bytes: Vec<char> = text.chars().collect();
    let mut end = 0;
    while end < bytes.len() {
        let c = bytes[end];
        let ok = c.is_ascii_digit()
            || c == '.'
            || ((c == '+' || c == '-') && (end == 0 || matches!(bytes[end - 1], 'e' | 'E')))
            || ((c == 'e' || c == 'E')
                && end + 1 < bytes.len()
                && (bytes[end + 1].is_ascii_digit()
                    || bytes[end + 1] == '+'
                    || bytes[end + 1] == '-'));
        if !ok {
            break;
        }
        end += 1;
    }
    if end == 0 {
        return Err(DimError::Malformed("no number"));
    }
    let value: f64 = text[..bytes[..end].iter().map(|c| c.len_utf8()).sum::<usize>()]
        .parse()
        .map_err(|_| DimError::Malformed("not a number"))?;
    let rest: String = bytes[end..].iter().collect();
    let (factor, dim) = parse_unit(&rest)?;
    Ok(Quantity { value: value * factor, dim })
}

/// Converts a value between two named units.
///
/// # Errors
///
/// [`DimError::UnknownUnit`] for an unrecognised name, or
/// [`DimError::Mismatch`] if the two measure different things -- which
/// is the whole point of the function rather than an edge case.
pub fn unit_convert(value: f64, from: &str, to: &str) -> Result<f64, DimError> {
    let (a, da) = parse_unit(from)?;
    let (b, db) = parse_unit(to)?;
    if da != db {
        return Err(DimError::Mismatch { expected: da, found: db });
    }
    Ok(value * a / b)
}

/// Formats a number with the SI prefix that brings it into `[1, 1000)`.
///
/// Returns the scaled number and the prefix, so that a caller can put
/// the unit after it. Zero and anything non-finite are returned with no
/// prefix, there being no sensible one.
pub fn si_prefixes_format(value: f64) -> (f64, &'static str) {
    if value == 0.0 || !value.is_finite() {
        return (value, "");
    }
    const STEPS: [(f64, &str); 17] = [
        (1e24, "Y"),
        (1e21, "Z"),
        (1e18, "E"),
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "u"),
        (1e-9, "n"),
        (1e-12, "p"),
        (1e-15, "f"),
        (1e-18, "a"),
        (1e-21, "z"),
        (1e-24, "y"),
    ];
    let magnitude = value.abs();
    for (scale, prefix) in STEPS {
        if magnitude >= scale {
            return (value / scale, prefix);
        }
    }
    (value / 1e-24, "y")
}

/// The 2022 CODATA constants, as `(name, value, unit)`.
///
/// Seven of these are exact by definition rather than measured: the
/// 2019 revision of the SI fixed `c`, `h`, `e`, `k`, `N_A`, the
/// caesium hyperfine frequency and the luminous efficacy, and defined
/// the kilogram, ampere, kelvin, mole and candela in terms of them. The
/// gravitational constant is not among them and remains the worst known
/// of the fundamental constants by a wide margin -- about one part in
/// forty thousand, against one part in `1e10` for the fine-structure
/// constant.
pub fn constants_codata() -> Vec<(&'static str, f64, &'static str)> {
    vec![
        ("speed of light", 299_792_458.0, "m/s"),
        ("Planck constant", 6.626_070_15e-34, "J s"),
        ("reduced Planck constant", 1.054_571_817e-34, "J s"),
        ("elementary charge", 1.602_176_634e-19, "C"),
        ("Boltzmann constant", 1.380_649e-23, "J/K"),
        ("Avogadro constant", 6.022_140_76e23, "1/mol"),
        ("molar gas constant", 8.314_462_618_153_24, "J/mol/K"),
        ("gravitational constant", 6.674_30e-11, "m^3/kg/s^2"),
        ("vacuum electric permittivity", 8.854_187_818_8e-12, "F/m"),
        ("vacuum magnetic permeability", 1.256_637_061_27e-6, "H/m"),
        ("fine-structure constant", 7.297_352_564_3e-3, "1"),
        ("electron mass", 9.109_383_713_9e-31, "kg"),
        ("proton mass", 1.672_621_925_95e-27, "kg"),
        ("neutron mass", 1.674_927_500_56e-27, "kg"),
        ("atomic mass constant", 1.660_539_068_92e-27, "kg"),
        ("Rydberg constant", 10_973_731.568_157, "1/m"),
        ("Stefan-Boltzmann constant", 5.670_374_419e-8, "W/m^2/K^4"),
        ("Bohr radius", 5.291_772_105_44e-11, "m"),
        ("standard gravity", 9.806_65, "m/s^2"),
    ]
}

/// Looks a CODATA constant up by name.
pub fn codata(name: &str) -> Option<f64> {
    constants_codata().into_iter().find(|(n, _, _)| *n == name).map(|(_, v, _)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exponent_algebra_is_exact() {
        let speed = Dim::LENGTH.div(&Dim::TIME).unwrap();
        assert_eq!(speed, Dim::new(1, 0, -1, 0, 0, 0, 0));
        let force = Dim::MASS.mul(&speed.div(&Dim::TIME).unwrap()).unwrap();
        assert_eq!(force, Dim::new(1, 1, -2, 0, 0, 0, 0));
        // Multiplying then dividing by the same thing returns exactly
        // the original exponents, not nearly.
        let energy = force.mul(&Dim::LENGTH).unwrap();
        assert_eq!(energy.div(&Dim::LENGTH).unwrap(), force);
        assert_eq!(Dim::LENGTH.pow(3).unwrap(), Dim::new(3, 0, 0, 0, 0, 0, 0));
        assert_eq!(Dim::NONE.pow(7).unwrap(), Dim::NONE);
        // A square root exists exactly when every exponent is even.
        let area = Dim::LENGTH.pow(2).unwrap();
        assert_eq!(area.sqrt().unwrap(), Dim::LENGTH);
        assert_eq!(Dim::LENGTH.sqrt(), Err(DimError::NotAPerfectRoot(Dim::LENGTH)));
        assert_eq!(Dim::NONE.sqrt().unwrap(), Dim::NONE);
        assert!(Dim::NONE.is_dimensionless());
        assert!(!Dim::LENGTH.is_dimensionless());
        // Exponents live in a byte and overflow is reported.
        let big = Dim::new(100, 0, 0, 0, 0, 0, 0);
        assert_eq!(big.mul(&big), Err(DimError::Overflow));
        assert_eq!(big.pow(2), Err(DimError::Overflow));
        // Display puts the symbols back.
        assert_eq!(Dim::NONE.to_string(), "1");
        assert_eq!(speed.to_string(), "m s^-1");
        assert_eq!(Dim::MASS.to_string(), "kg");
    }

    #[test]
    fn adding_unlike_quantities_is_refused_and_like_ones_are_not() {
        let a = Quantity::meters(3.0);
        let b = Quantity::feet(1.0);
        let t = Quantity::seconds(2.0);
        assert!((a.add(&b).unwrap().value - 3.304_8).abs() < 1e-12);
        assert!((a.sub(&b).unwrap().value - 2.695_2).abs() < 1e-12);
        assert_eq!(
            a.add(&t),
            Err(DimError::Mismatch { expected: Dim::LENGTH, found: Dim::TIME })
        );
        assert!(a.sub(&t).is_err());
        // Multiplying and dividing always work and track the exponents.
        let speed = a.div(&t).unwrap();
        assert_eq!(speed.dim, Dim::new(1, 0, -1, 0, 0, 0, 0));
        assert!((speed.value - 1.5).abs() < 1e-15);
        let back = speed.mul(&t).unwrap();
        assert_eq!(back.dim, Dim::LENGTH);
        assert!((back.value - 3.0).abs() < 1e-15);
        // Force times distance is an energy, whichever way it is built.
        let f = Quantity::newtons(4.0);
        assert_eq!(f.mul(&a).unwrap().dim, Quantity::joules(1.0).dim);
        assert_eq!(Quantity::watts(1.0).mul(&t).unwrap().dim, Quantity::joules(1.0).dim);
        assert_eq!(Quantity::volts(1.0).mul(&Quantity::amperes(1.0)).unwrap().dim,
            Quantity::watts(1.0).dim);
        assert_eq!(
            Quantity::pascals(1.0).mul(&Quantity::meters(1.0).pow(3).unwrap()).unwrap().dim,
            Quantity::joules(1.0).dim
        );
        // A root of an area is a length; of a length it is nothing.
        let area = a.pow(2).unwrap();
        assert_eq!(area.sqrt().unwrap().dim, Dim::LENGTH);
        assert!(a.sqrt().is_err());
        assert_eq!(Quantity::number(4.0).sqrt().unwrap().value, 2.0);
    }

    #[test]
    fn units_parse_the_way_the_rule_says() {
        // Whole name before prefix: m is a metre, mm a millimetre, min a
        // minute, T a tesla.
        assert_eq!(parse_unit("m").unwrap(), (1.0, Dim::LENGTH));
        assert!((parse_unit("mm").unwrap().0 - 1e-3).abs() < 1e-18);
        assert_eq!(parse_unit("min").unwrap(), (60.0, Dim::TIME));
        assert_eq!(parse_unit("T").unwrap().1, Quantity::teslas(1.0).dim);
        // The gram carries the prefixes, so a kilogram comes out at one.
        assert!((parse_unit("kg").unwrap().0 - 1.0).abs() < 1e-15);
        assert!((parse_unit("mg").unwrap().0 - 1e-6).abs() < 1e-21);
        assert!((parse_unit("g").unwrap().0 - 1e-3).abs() < 1e-18);
        // Compound expressions.
        let (f, d) = parse_unit("m/s^2").unwrap();
        assert!((f - 1.0).abs() < 1e-15);
        assert_eq!(d, Dim::new(1, 0, -2, 0, 0, 0, 0));
        let (f, d) = parse_unit("kg*m^2/s^3").unwrap();
        assert!((f - 1.0).abs() < 1e-15);
        assert_eq!(d, Quantity::watts(1.0).dim);
        assert_eq!(parse_unit("1").unwrap(), (1.0, Dim::NONE));
        assert_eq!(parse_unit("").unwrap(), (1.0, Dim::NONE));
        assert!(matches!(parse_unit("zorkmid"), Err(DimError::UnknownUnit(_))));
        assert!(matches!(parse_unit("m^x"), Err(DimError::Malformed(_))));
        assert!(matches!(parse_unit("m//s"), Err(DimError::Malformed(_))));
    }

    #[test]
    fn quantities_parse_from_text() {
        let g = parse_quantity("9.81 m/s^2").unwrap();
        assert!((g.value - 9.81).abs() < 1e-12);
        assert_eq!(g.dim, Dim::new(1, 0, -2, 0, 0, 0, 0));
        let e = parse_quantity("3 kWh").unwrap();
        assert!((e.value - 1.08e7).abs() < 1.0);
        assert_eq!(e.dim, Quantity::joules(1.0).dim);
        // An `e` is only an exponent when a digit or sign follows, so
        // "1 eV" is an electron volt rather than a broken float.
        let ev = parse_quantity("1 eV").unwrap();
        assert!((ev.value - 1.602_176_634e-19).abs() < 1e-30);
        let big = parse_quantity("2.5e3 mm").unwrap();
        assert!((big.value - 2.5).abs() < 1e-12);
        let neg = parse_quantity("-4.5 N*m").unwrap();
        assert!((neg.value + 4.5).abs() < 1e-12);
        assert_eq!(neg.dim, Quantity::joules(1.0).dim);
        let bare = parse_quantity("7").unwrap();
        assert_eq!(bare.dim, Dim::NONE);
        assert!(matches!(parse_quantity("kg"), Err(DimError::Malformed(_))));
        assert!(matches!(parse_quantity(""), Err(DimError::Malformed(_))));
        assert!(matches!(parse_quantity("1 zorkmid"), Err(DimError::UnknownUnit(_))));
    }

    #[test]
    fn conversions_agree_with_the_flat_functions_and_round_trip() {
        // The typed path and the original conversion functions are two
        // routes to the same number, so they had better agree.
        use crate::units::{feet_to_meters, kg_to_lbs, meters_to_feet};
        assert!((unit_convert(1.0, "m", "ft").unwrap() - meters_to_feet(1.0)).abs() < 1e-4);
        assert!((unit_convert(1.0, "ft", "m").unwrap() - feet_to_meters(1.0)).abs() < 1e-6);
        assert!((unit_convert(1.0, "kg", "lb").unwrap() - kg_to_lbs(1.0)).abs() < 1e-4);
        // Round trips return the value.
        for (a, b) in [("km", "mi"), ("J", "eV"), ("h", "s"), ("L", "m^3"), ("bar", "Pa")] {
            let there = unit_convert(3.5, a, b).unwrap();
            let back = unit_convert(there, b, a).unwrap();
            assert!((back - 3.5).abs() < 1e-9, "{a} to {b} and back gave {back}");
        }
        // Converting between different things is the error the function
        // exists to raise.
        assert!(matches!(unit_convert(1.0, "m", "s"), Err(DimError::Mismatch { .. })));
        assert!(matches!(unit_convert(1.0, "m", "zorkmid"), Err(DimError::UnknownUnit(_))));
        // Quantity::to is the same conversion from the other side.
        assert!((Quantity::kilometers(2.0).to("m").unwrap() - 2000.0).abs() < 1e-9);
        assert!(Quantity::kilometers(2.0).to("s").is_err());
        assert!(Quantity::meters(1.0).format_si().contains('m'));
    }

    #[test]
    fn the_prefix_formatter_lands_in_the_right_decade() {
        for (value, wanted, prefix) in [
            (1234.0, 1.234, "k"),
            (0.000_42, 420.0, "u"),
            (5.0, 5.0, ""),
            (-2.5e9, -2.5, "G"),
            (7e-15, 7.0, "f"),
        ] {
            let (scaled, got) = si_prefixes_format(value);
            assert_eq!(got, prefix, "for {value}");
            assert!((scaled - wanted).abs() < 1e-9 * wanted.abs().max(1.0), "for {value}");
        }
        assert_eq!(si_prefixes_format(0.0), (0.0, ""));
        assert!(si_prefixes_format(f64::NAN).0.is_nan());
        // The scaled value always lies in [1, 1000) unless it ran out of
        // prefixes at either end.
        for k in -20..20i32 {
            let v = 3.7 * 10f64.powi(k);
            let (scaled, _) = si_prefixes_format(v);
            assert!(scaled.abs() >= 1.0 && scaled.abs() < 1000.0, "{v} gave {scaled}");
        }
    }

    #[test]
    fn the_codata_table_is_internally_consistent() {
        let get = |n: &str| codata(n).unwrap_or_else(|| panic!("{n} is missing"));
        // Exact by definition since the 2019 revision of the SI.
        assert_eq!(get("speed of light"), 299_792_458.0);
        assert_eq!(get("Planck constant"), 6.626_070_15e-34);
        assert_eq!(get("elementary charge"), 1.602_176_634e-19);
        assert_eq!(get("Boltzmann constant"), 1.380_649e-23);
        assert_eq!(get("Avogadro constant"), 6.022_140_76e23);
        // The gas constant is the product of two exact ones, so it is
        // exact too.
        let r = get("Boltzmann constant") * get("Avogadro constant");
        assert!((r - get("molar gas constant")).abs() < 1e-12 * r);
        // hbar is h over two pi.
        let hbar = get("Planck constant") / std::f64::consts::TAU;
        assert!((hbar - get("reduced Planck constant")).abs() < 1e-9 * hbar);
        // epsilon_0 mu_0 c^2 = 1, which is what fixes the permittivity
        // once the permeability is measured.
        let one = get("vacuum electric permittivity")
            * get("vacuum magnetic permeability")
            * get("speed of light").powi(2);
        assert!((one - 1.0).abs() < 1e-9, "epsilon mu c^2 came to {one}");
        // The fine-structure constant follows from the others.
        let alpha = get("elementary charge").powi(2)
            / (2.0
                * std::f64::consts::TAU
                * get("vacuum electric permittivity")
                * get("reduced Planck constant")
                * get("speed of light"));
        assert!(
            (alpha - get("fine-structure constant")).abs() < 1e-8 * alpha,
            "alpha came to {alpha}"
        );
        // And the Rydberg constant from alpha and the electron mass.
        let rydberg = alpha * alpha * get("electron mass") * get("speed of light")
            / (2.0 * get("Planck constant"));
        assert!(
            (rydberg - get("Rydberg constant")).abs() < 1e-7 * rydberg,
            "Rydberg came to {rydberg}"
        );
        assert!(codata("phlogiston").is_none());
        // Every entry parses as the unit it claims.
        for (name, _, unit) in constants_codata() {
            assert!(parse_unit(unit).is_ok(), "{name} has an unparseable unit {unit}");
        }
    }
}
