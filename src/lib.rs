// Style allowances for a numerics codebase: index loops mirror the math
// notation in matrix/stencil kernels, and public signatures follow the
// roadmap's frozen APIs even where clippy would prefer fewer arguments or
// simpler types.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Argument guards use the negated form `!(x > 0.0)` on purpose: unlike
// `x <= 0.0`, it also rejects NaN inputs.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod math;
pub mod error;
pub mod core;
pub mod discrete;
pub mod exact;
pub mod special;
pub mod classical;
pub mod gravitation;
pub mod thermodynamics;
pub mod electromagnetism;
pub mod fluids;
pub mod waves;
pub mod optics;
pub mod relativity;
pub mod quantum;
pub mod nuclear;
pub mod radiation;
pub mod atmosphere;
pub mod statistical_mechanics;
pub mod astrophysics;
pub mod rf;
pub mod linalg;
pub mod numerical;
pub mod statistics;
pub mod plasma;
pub mod solid_mechanics;
pub mod chemistry;
pub mod electronics;
pub mod geometry;
pub mod graph;
pub mod propulsion;
pub mod units;
pub mod nonlinear;
pub mod finance;
pub mod learn;
pub mod fractals;
pub mod particle_physics;
pub mod quaternion;
pub mod monte_carlo;
pub mod information_theory;
pub mod vector_calculus;
pub mod optimization;
pub mod signal_processing;
pub mod transforms;
pub mod dsp;
pub mod resonance;
pub mod cfd;
pub mod manifold;
pub mod fem;
pub mod fields;
pub mod audio;
pub mod curves;
pub mod trigonometry;
pub mod neutronics;
pub mod control_systems;
pub mod color_science;
pub mod biophysics;
pub mod acoustics;
pub mod materials;
pub mod general_relativity;
pub mod geophysics;
pub mod magnetohydrodynamics;
pub mod photonics;
pub mod fluid_instabilities;
pub mod sim;
pub mod continuum_mechanics;
pub mod spatial;
pub mod stochastic;
pub mod mesh;
pub mod codes;
pub mod patterns;

#[cfg(kani)]
mod verification;
