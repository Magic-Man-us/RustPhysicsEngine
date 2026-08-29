//! Python bindings for `rust_physics_engine`.
//!
//! Almost everything under `generated/` is written by
//! `bindings/python/generate.py`, which reads the library's source and
//! emits a wrapper for each item it can bind. This file and `runtime/`
//! are the parts a generator cannot write: the module's entry point, the
//! exception hierarchy, the coercions that let a Python tuple stand in
//! for a `Vec3`, and the adapter that carries a Python callable into a
//! routine expecting `&dyn Fn`.

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod generated;
mod runtime;

/// The extension module. `numeria/__init__.py` re-exports
/// from here and installs the submodules into `sys.modules`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    runtime::install_panic_hook();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    runtime::errors::register(m)?;
    generated::register(m.py(), m)?;
    Ok(())
}
