//! Property-based tests: randomized invariants checked with the crate's
//! own deterministic `Rng`, one module per library area.

mod core_props;
mod linalg_props;
mod numerical_props;
mod signal_props;
mod special_props;
