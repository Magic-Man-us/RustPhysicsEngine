//! Letting Python values stand in for Rust ones.
//!
//! Two kinds of conversion live here. The first is coercion: a `Vec3`
//! argument accepts `(1.0, 2.0, 3.0)`, a `Matrix` accepts a list of rows.
//! Requiring `Vec3(1, 2, 3)` everywhere would be faithful to the Rust API
//! and unpleasant to use; accepting both costs one `extract` attempt and
//! reads the way a Python caller expects.
//!
//! The second is identification: three Rust types have exact Python
//! counterparts, and are translated rather than wrapped.
//!
//! | Rust | Python |
//! |---|---|
//! | `fractals::Complex` | `complex` |
//! | `exact::bigint::BigInt` | `int` |
//! | `exact::rational::Rational` | `fractions.Fraction` |
//!
//! `BigInt` and `Fraction` are both arbitrary-precision and both exact, so
//! the round trip loses nothing -- which is the test a translation has to
//! pass before it is worth doing. Everything else gets a wrapper class.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::OnceLock;
use pyo3::types::{PyComplex, PyComplexMethods, PyList, PyListMethods, PyString, PyTuple};
use pyo3::Borrowed;

use rust_physics_engine::exact::bigint::BigInt;
use rust_physics_engine::exact::rational::Rational;
use rust_physics_engine::fractals::Complex;

/// Extracts exactly `n` floats from a sequence.
///
/// Used by the generated adapters for the small fixed-width value types --
/// `Vec2`, `Vec3`, `Vec4`, `Mat3` and friends -- so that any sequence of
/// the right length is accepted.
pub fn floats_exact(obj: Borrowed<'_, '_, PyAny>, n: usize, what: &str) -> PyResult<Vec<f64>> {
    let v: Vec<f64> = obj.extract().map_err(|_| {
        PyTypeError::new_err(format!("{what} expects {n} floats or a {what} instance"))
    })?;
    if v.len() != n {
        return Err(PyTypeError::new_err(format!(
            "{what} expects {n} floats, got {}",
            v.len()
        )));
    }
    Ok(v)
}

/// Extracts a rectangular list of rows, checking that the rows agree.
pub fn rows(obj: Borrowed<'_, '_, PyAny>, what: &str) -> PyResult<Vec<Vec<f64>>> {
    let v: Vec<Vec<f64>> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("{what} expects a sequence of rows")))?;
    if v.is_empty() || v[0].is_empty() {
        return Err(PyTypeError::new_err(format!("{what} needs at least one non-empty row")));
    }
    let cols = v[0].len();
    for (i, r) in v.iter().enumerate() {
        if r.len() != cols {
            return Err(PyTypeError::new_err(format!(
                "{what}: row {i} has {} entries, row 0 has {cols}",
                r.len()
            )));
        }
    }
    Ok(v)
}

// ── Complex ─────────────────────────────────────────────────────────────

/// A `Complex` argument. Accepts `complex`, `float` and `int`.
pub struct ComplexArg(pub Complex);

impl<'a, 'py> FromPyObject<'a, 'py> for ComplexArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, PyErr> {
        let any = obj.as_any();
        if let Ok(c) = any.cast::<PyComplex>() {
            return Ok(ComplexArg(Complex::new(c.real(), c.imag())));
        }
        if let Ok(x) = any.extract::<f64>() {
            return Ok(ComplexArg(Complex::new(x, 0.0)));
        }
        // `complex()` on an object with `__complex__`.
        let called = any.call_method0("__complex__").map_err(|_| {
            PyTypeError::new_err("expected a complex, a float, or an object with __complex__")
        })?;
        let c = called
            .cast::<PyComplex>()
            .map_err(|_| PyTypeError::new_err("__complex__ did not return a complex"))?;
        Ok(ComplexArg(Complex::new(c.real(), c.imag())))
    }
}

/// Builds a Python `complex`.
pub fn complex_out(py: Python<'_>, c: Complex) -> Bound<'_, PyComplex> {
    PyComplex::from_doubles(py, c.re, c.im)
}

// ── BigInt ──────────────────────────────────────────────────────────────

static INT_CTOR: OnceLock<Py<PyAny>> = OnceLock::new();

fn int_ctor(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    if let Some(v) = INT_CTOR.get() {
        return Ok(v);
    }
    let ctor = py.import("builtins")?.getattr("int")?.unbind();
    Ok(INT_CTOR.get_or_init(|| ctor))
}

/// A `BigInt` argument: any Python `int`, of any size.
pub struct BigIntArg(pub BigInt);

impl<'a, 'py> FromPyObject<'a, 'py> for BigIntArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, PyErr> {
        Ok(BigIntArg(bigint_from(obj.as_any())?))
    }
}

/// Converts a Python `int` to a [`BigInt`].
///
/// Values inside `i64` take a direct path. Anything larger goes via a
/// hexadecimal string, which is exact -- four bits per character -- and is
/// the only conversion CPython offers that does not depend on the
/// internal digit layout.
pub fn bigint_from(obj: &Bound<'_, PyAny>) -> PyResult<BigInt> {
    if let Ok(n) = obj.extract::<i64>() {
        return Ok(BigInt::from_i64(n));
    }
    let s: String = obj
        .call_method1("__format__", ("x",))
        .map_err(|_| PyTypeError::new_err("expected an int"))?
        .extract()?;
    BigInt::from_str_radix(&s, 16).map_err(super::errors::map_geom)
}

/// Converts a [`BigInt`] to a Python `int`.
pub fn bigint_out<'py>(py: Python<'py>, v: &BigInt) -> PyResult<Bound<'py, PyAny>> {
    if let Some(n) = v.to_i64() {
        return Ok(n.into_pyobject(py)?.into_any());
    }
    let s = v.to_string_radix(16);
    let text = PyString::new(py, &s);
    int_ctor(py)?.bind(py).call1((text, 16))
}

// ── Rational ────────────────────────────────────────────────────────────

static FRACTION: OnceLock<Py<PyAny>> = OnceLock::new();

fn fraction(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    if let Some(v) = FRACTION.get() {
        return Ok(v);
    }
    let cls = py.import("fractions")?.getattr("Fraction")?.unbind();
    Ok(FRACTION.get_or_init(|| cls))
}

/// A `Rational` argument: a `Fraction`, an `int`, or a `(numerator,
/// denominator)` pair. A `float` is *not* accepted -- 0.1 is not one
/// tenth, and silently pretending otherwise in a module whose whole point
/// is exactness would be the wrong kindness. Use
/// `Fraction(1, 10)`, or `exact.rational.Rational.from_f64_approx`.
pub struct RationalArg(pub Rational);

impl<'a, 'py> FromPyObject<'a, 'py> for RationalArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, PyErr> {
        let any = obj.as_any();
        if let (Ok(n), Ok(d)) = (any.getattr("numerator"), any.getattr("denominator")) {
            let num = bigint_from(&n)?;
            let den = bigint_from(&d)?;
            return Rational::new(num, den)
                .map(RationalArg)
                .ok_or_else(|| PyTypeError::new_err("a Rational may not have a zero denominator"));
        }
        if let Ok(t) = any.cast::<PyTuple>() {
            if t.len() == 2 {
                let num = bigint_from(&t.get_item(0)?)?;
                let den = bigint_from(&t.get_item(1)?)?;
                return Rational::new(num, den).map(RationalArg).ok_or_else(|| {
                    PyTypeError::new_err("a Rational may not have a zero denominator")
                });
            }
        }
        Err(PyTypeError::new_err(
            "expected a Fraction, an int, or a (numerator, denominator) pair",
        ))
    }
}

/// Converts a [`Rational`] to a `fractions.Fraction`.
pub fn rational_out<'py>(py: Python<'py>, v: &Rational) -> PyResult<Bound<'py, PyAny>> {
    let num = bigint_out(py, &v.num)?;
    let den = bigint_out(py, &v.den)?;
    fraction(py)?.bind(py).call1((num, den))
}

// ── Arguments written through ───────────────────────────────────────────

/// Writes a mutated slice back into the Python list it came from.
///
/// A handful of routines take `&mut [f64]` and work in place. Copying the
/// values in and dropping the results on the floor would compile and give
/// wrong answers silently, which is the one outcome worth ruling out; so
/// the values go back where they came from, and an argument that cannot
/// receive them -- a tuple, a generator -- is a `TypeError` rather than a
/// quiet no-op.
pub fn write_back<T>(obj: &Bound<'_, PyAny>, values: &[T]) -> PyResult<()>
where
    T: Copy + for<'py> IntoPyObject<'py>,
{
    let list = obj.cast::<PyList>().map_err(|_| {
        PyTypeError::new_err("this argument is modified in place, so it must be a list")
    })?;
    let fresh = PyList::empty(obj.py());
    for v in values {
        fresh.append(*v)?;
    }
    list.set_slice(0, list.len(), fresh.as_any())
}

/// A [`Complex`] on its way *to* Python, as a `complex`.
///
/// Needed where the value is handed to a Python callable rather than
/// returned: `Callback::call` takes a tuple of things that convert
/// themselves, and a bare `Complex` is not one.
pub struct Cx(pub Complex);

impl<'py> IntoPyObject<'py> for Cx {
    type Target = PyComplex;
    type Output = Bound<'py, PyComplex>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyComplex::from_doubles(py, self.0.re, self.0.im))
    }
}
