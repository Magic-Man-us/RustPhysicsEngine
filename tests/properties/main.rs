//! Property-based tests: randomized invariants checked with the crate's
//! own deterministic `Rng`, one module per library area.

// Same style allowances as the library: index loops mirror the maths, and
// table-driven cases carry deliberately explicit types.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

mod core_props;
mod discrete_props;
mod fractals_props;
mod game_theory_props;
mod geometry_props;
mod graph_flow_props;
mod graph_props;
mod graph_structure_props;
mod kinetics_props;
mod linalg_props;
mod md_props;
mod mesh_props;
mod numerical_props;
mod optimization_continuous_props;
mod optimization_discrete_props;
mod optimization_lp_props;
mod quantum_circuit_props;
mod quantum_matter_props;
mod quantum_props;
mod signal_props;
mod spatial_props;
mod special_props;
mod monte_carlo_props;
mod patterns_props;
mod statistics_props;
mod statmech_props;
mod stochastic_extremes_props;
mod stochastic_process_props;
mod transforms_props;
