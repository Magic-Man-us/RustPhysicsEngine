//! Passing a Python callable into a routine that wants `&dyn Fn`.
//!
//! Roughly two hundred functions in the library take a function: an
//! integrand, a residual, the right-hand side of an ODE. Rust's signature
//! for those has no room for failure -- `&dyn Fn(f64) -> f64` returns an
//! `f64` and nothing else -- but a Python callable can raise, and can
//! return something that is not a number.
//!
//! [`Callback`] holds the first such error instead of discarding it. When
//! the callable fails, the call returns a fallback so the Rust routine can
//! unwind normally on its own terms, every later call short-circuits to
//! that same fallback rather than compounding the failure, and the wrapper
//! re-raises the stored exception once the routine returns. What the
//! caller sees is their own `ZeroDivisionError`, with their traceback,
//! rather than a NaN that came from nowhere.

use std::cell::RefCell;

use pyo3::prelude::*;

/// A Python callable, adapted for use as a Rust `Fn`.
pub struct Callback {
    obj: Py<PyAny>,
    err: RefCell<Option<PyErr>>,
}

impl Callback {
    /// Wraps a Python object. It is not checked for callability here; a
    /// non-callable raises `TypeError` on first use, which is where the
    /// traceback is most useful.
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj, err: RefCell::new(None) }
    }

    /// Calls the object with `args`, returning `fallback` if anything goes
    /// wrong -- and remembering why.
    pub fn call<A, R>(&self, args: A, fallback: R) -> R
    where
        A: for<'py> pyo3::call::PyCallArgs<'py>,
        R: for<'a, 'py> FromPyObject<'a, 'py>,
        for<'a, 'py> <R as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
    {
        if self.err.borrow().is_some() {
            return fallback;
        }
        Python::attach(|py| {
            let outcome = self
                .obj
                .call1(py, args)
                .and_then(|v| v.bind(py).extract::<R>().map_err(Into::into));
            match outcome {
                Ok(v) => v,
                Err(e) => {
                    *self.err.borrow_mut() = Some(e);
                    fallback
                }
            }
        })
    }

    /// The first exception raised inside the callable, if there was one.
    pub fn take_err(&self) -> Option<PyErr> {
        self.err.borrow_mut().take()
    }
}

/// Re-raises the first error from any of `callbacks`, else returns `value`.
///
/// The generated wrappers call this before returning. It runs before the
/// panic check, because a panic that follows a failed callback is a
/// consequence of the fallback value, not the thing the caller needs to
/// see.
pub fn check<T>(callbacks: &[&Callback], value: T) -> PyResult<T> {
    for cb in callbacks {
        if let Some(e) = cb.take_err() {
            return Err(e);
        }
    }
    Ok(value)
}
