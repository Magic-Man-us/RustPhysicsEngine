//! The exception hierarchy, and the two things that produce it: the
//! library's error enums, and panics.
//!
//! Every exception raised by these bindings derives from `PhysicsError`,
//! so `except PhysicsError` catches all of them and nothing else. Below
//! that the tree follows the library's own two error enums, plus the
//! dimensional errors from `units`. Where a variant carries data --
//! `NoConvergence { iters, residual }`, `DimensionMismatch { expected,
//! got }` -- that data is set as attributes on the exception instance
//! rather than only formatted into its message, so a caller can branch on
//! the residual instead of parsing a string.
//!
//! Panics are the second source. Much of the library validates arguments
//! with `assert!`, which is right for Rust -- a negative mass is a
//! programming error, not a runtime condition -- but a panic crossing the
//! FFI boundary would abort the interpreter or, at best, surface as
//! `pyo3_runtime.PanicException` with a backtrace printed to stderr.
//! [`guard`] catches them and raises `InvalidArgumentError` carrying the
//! assertion's own message, which is the message the library author
//! wrote for exactly this case.

use std::cell::{Cell, RefCell};
use std::panic::AssertUnwindSafe;
use std::sync::Once;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use rust_physics_engine::error::{GeomError, SolveError};
use rust_physics_engine::units::quantity::DimError;

pyo3::create_exception!(
    rust_physics_engine,
    PhysicsError,
    pyo3::exceptions::PyException,
    "Base class for every exception raised by rust_physics_engine."
);
pyo3::create_exception!(
    rust_physics_engine,
    InvalidArgumentError,
    PhysicsError,
    "An argument violated a documented precondition."
);
pyo3::create_exception!(
    rust_physics_engine,
    SolverError,
    PhysicsError,
    "Base class for failures reported by the numerical solvers."
);
pyo3::create_exception!(
    rust_physics_engine,
    SingularMatrixError,
    SolverError,
    "The matrix is singular, or a pivot fell below the safe threshold."
);
pyo3::create_exception!(
    rust_physics_engine,
    NotPositiveDefiniteError,
    SolverError,
    "The matrix is not (numerically) symmetric positive definite."
);
pyo3::create_exception!(
    rust_physics_engine,
    ConvergenceError,
    SolverError,
    "An iteration did not converge. Carries `iterations` and `residual`."
);
pyo3::create_exception!(
    rust_physics_engine,
    DimensionMismatchError,
    SolverError,
    "Operand shapes are incompatible. Carries `expected` and `got`."
);
pyo3::create_exception!(
    rust_physics_engine,
    GeometryError,
    PhysicsError,
    "Base class for failures reported by the geometric algorithms."
);
pyo3::create_exception!(
    rust_physics_engine,
    DegenerateGeometryError,
    GeometryError,
    "Degenerate input: a zero-area triangle, coincident points, and so on."
);
pyo3::create_exception!(
    rust_physics_engine,
    NotManifoldError,
    GeometryError,
    "The mesh is not manifold where the operation requires it."
);
pyo3::create_exception!(
    rust_physics_engine,
    EmptyInputError,
    GeometryError,
    "The input contains no elements."
);
pyo3::create_exception!(
    rust_physics_engine,
    UnitsError,
    PhysicsError,
    "A dimensional check failed in `units`."
);

/// Registers the exception classes on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("PhysicsError", py.get_type::<PhysicsError>())?;
    m.add("InvalidArgumentError", py.get_type::<InvalidArgumentError>())?;
    m.add("SolverError", py.get_type::<SolverError>())?;
    m.add("SingularMatrixError", py.get_type::<SingularMatrixError>())?;
    m.add("NotPositiveDefiniteError", py.get_type::<NotPositiveDefiniteError>())?;
    m.add("ConvergenceError", py.get_type::<ConvergenceError>())?;
    m.add("DimensionMismatchError", py.get_type::<DimensionMismatchError>())?;
    m.add("GeometryError", py.get_type::<GeometryError>())?;
    m.add("DegenerateGeometryError", py.get_type::<DegenerateGeometryError>())?;
    m.add("NotManifoldError", py.get_type::<NotManifoldError>())?;
    m.add("EmptyInputError", py.get_type::<EmptyInputError>())?;
    m.add("UnitsError", py.get_type::<UnitsError>())?;
    Ok(())
}

fn with_attrs(err: PyErr, attrs: &[(&str, f64)], ints: &[(&str, usize)]) -> PyErr {
    Python::attach(|py| {
        let value = err.value(py);
        for (k, v) in attrs {
            let _ = value.setattr(*k, *v);
        }
        for (k, v) in ints {
            let _ = value.setattr(*k, *v);
        }
    });
    err
}

/// Maps a [`SolveError`] onto the matching Python exception.
pub fn map_solve(e: SolveError) -> PyErr {
    let msg = e.to_string();
    match e {
        SolveError::Singular => SingularMatrixError::new_err(msg),
        SolveError::NotPositiveDefinite => NotPositiveDefiniteError::new_err(msg),
        SolveError::NoConvergence { iters, residual } => with_attrs(
            ConvergenceError::new_err(msg),
            &[("residual", residual)],
            &[("iterations", iters)],
        ),
        SolveError::DimensionMismatch { expected, got } => with_attrs(
            DimensionMismatchError::new_err(msg),
            &[],
            &[("expected", expected), ("got", got)],
        ),
        SolveError::InvalidArgument(_) => InvalidArgumentError::new_err(msg),
    }
}

/// Maps a [`GeomError`] onto the matching Python exception.
pub fn map_geom(e: GeomError) -> PyErr {
    let msg = e.to_string();
    match e {
        GeomError::Degenerate(_) => DegenerateGeometryError::new_err(msg),
        GeomError::NotManifold => NotManifoldError::new_err(msg),
        GeomError::Empty => EmptyInputError::new_err(msg),
        GeomError::InvalidArgument(_) => InvalidArgumentError::new_err(msg),
    }
}

/// Maps a [`DimError`] onto `UnitsError`.
pub fn map_dim(e: DimError) -> PyErr {
    UnitsError::new_err(e.to_string())
}

/// Fallback for the handful of one-off error types that are plain structs
/// (`TooManyErrors`, `NegativeCycle`): anything with a `Display`.
pub fn map_display<E: std::fmt::Display>(e: E) -> PyErr {
    PhysicsError::new_err(e.to_string())
}

/// Last resort, for an error type that is data rather than a message --
/// `graph::matching` reports an odd cycle as the `Vec<usize>` of its
/// vertices. `Debug` is what such a type has, so `Debug` is what the
/// exception carries.
pub fn map_debug<E: std::fmt::Debug>(e: E) -> PyErr {
    PhysicsError::new_err(format!("{e:?}"))
}

thread_local! {
    static PANIC_MESSAGE: RefCell<Option<String>> = const { RefCell::new(None) };
    static SUPPRESS: Cell<bool> = const { Cell::new(false) };
}

static HOOK: Once = Once::new();

/// Installs a panic hook that stays quiet inside [`guard`] and defers to
/// whatever hook was already in place everywhere else.
///
/// The suppression flag is thread-local, so a panic on some other thread
/// -- one this extension knows nothing about -- still prints and still
/// reaches the previous hook. Only a panic raised inside a call that
/// [`guard`] is watching is captured, and that panic is about to become an
/// exception carrying the same message.
pub fn install_panic_hook() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if SUPPRESS.with(Cell::get) {
                let payload = info.payload();
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panicked".to_string());
                PANIC_MESSAGE.with(|slot| *slot.borrow_mut() = Some(msg));
            } else {
                previous(info);
            }
        }));
    });
}

/// Runs `f`, converting a panic into `InvalidArgumentError`.
///
/// The library asserts its preconditions -- `assert!(mass > 0.0, "mass
/// must be positive")` -- and those assertions are the argument checks a
/// Python caller should see. The message the assertion carries becomes
/// the exception's message.
pub fn guard<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    let previous = SUPPRESS.with(|s| s.replace(true));
    PANIC_MESSAGE.with(|slot| *slot.borrow_mut() = None);
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    SUPPRESS.with(|s| s.set(previous));
    match result {
        Ok(v) => Ok(v),
        Err(_) => Err(PANIC_MESSAGE
            .with(|slot| slot.borrow_mut().take())
            .unwrap_or_else(|| "the underlying Rust routine panicked".to_string())),
    }
}

/// [`guard`], with the panic already turned into a `PyErr`.
pub fn guarded<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    guard(f).map_err(InvalidArgumentError::new_err)
}
