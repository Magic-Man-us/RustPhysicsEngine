//! The hand-written half of the bindings: everything the generator calls
//! but does not itself produce.
//!
//! Three concerns live here. [`errors`] turns the library's two error
//! enums, and any panic that escapes an argument assertion, into a Python
//! exception hierarchy. [`coerce`] lets a Python tuple stand in for a
//! `Vec3` and a list of lists for a `Matrix`, so callers are not forced to
//! construct wrapper objects for values that have an obvious literal form.
//! [`callback`] passes a Python callable into a routine that expects
//! `&dyn Fn`, and carries an exception raised inside it back out to the
//! caller instead of swallowing it.

pub mod callback;
pub mod coerce;
pub mod errors;

pub use callback::Callback;
pub use errors::{guard, install_panic_hook, map_geom, map_solve};
