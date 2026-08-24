//! Kani proof harnesses. Compiled only under `cargo kani` (`#[cfg(kani)]`
//! at the inclusion site in `lib.rs`).

pub mod core;
pub mod linalg;
pub mod physics;
pub mod spatial;
