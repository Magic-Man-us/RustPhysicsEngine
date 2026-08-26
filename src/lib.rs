//! A zero-dependency library for physics, mathematics and engineering
//! computation.
//!
//! # What this is for
//!
//! Every routine here is written so that something about it can be
//! *checked*: against a closed form, against a conservation law, against an
//! independent implementation of the same quantity, or against an exact
//! identity over integers. A test that only asserts a function ran is not
//! evidence. Where a result is approximate its error has a stated bound
//! derived from the method; where it is exact the assertion uses `==`.
//!
//! That principle decides the shape of the API. Solvers return
//! [`Result`] rather than panicking on non-convergence, so a caller can
//! tell "did not converge" from "converged to this". Functions validate
//! their arguments, and the guards are written `!(x > 0.0)` rather than
//! `x <= 0.0` so that NaN is rejected too. Physical constants come from
//! one table, [`math::constants`], and the values fixed by the 2019 SI
//! redefinition are exact.
//!
//! # Finding your way around
//!
//! The crate is wide -- 71 top-level modules. `docs/MODULE_MAP.md` in the
//! repository is a generated map of every module with its size and
//! summary. The rough shape:
//!
//! | Area | Modules |
//! |---|---|
//! | Numeric primitives | [`core`], [`math`], [`linalg`], [`numerical`], [`special`] |
//! | Exact and symbolic | [`exact`], [`discrete`], [`graph`], [`codes`] |
//! | Classical physics | [`classical`], [`gravitation`], [`solid_mechanics`], [`continuum_mechanics`], [`resonance`] |
//! | Thermal and statistical | [`thermodynamics`], [`statistical_mechanics`], [`radiation`] |
//! | Electromagnetic | [`electromagnetism`], [`electronics`], [`rf`], [`photonics`], [`plasma`], [`magnetohydrodynamics`] |
//! | Waves and signals | [`waves`], [`optics`], [`acoustics`], [`transforms`], [`dsp`], [`signal_processing`], [`audio`] |
//! | Fluids | [`fluids`], [`cfd`], [`fluid_instabilities`], [`propulsion`] |
//! | Modern physics | [`relativity`], [`general_relativity`], [`quantum`], [`particle_physics`], [`nuclear`], [`neutronics`] |
//! | Space | [`astrophysics`] |
//! | PDE solvers | [`fem`], [`sim`], [`fields`], [`vector_calculus`] |
//! | Life and chemistry | [`chemistry`], [`biophysics`] |
//! | Probability and data | [`statistics`], [`stochastic`], [`monte_carlo`], [`information_theory`], [`learn`] |
//! | Decisions | [`optimization`], [`finance`] |
//! | Geometry | [`geometry`], [`curves`], [`trigonometry`], [`quaternion`], [`manifold`], [`spatial`], [`mesh`] |
//! | Patterns | [`fractals`], [`patterns`], [`nonlinear`] |
//! | Reference and utility | [`units`], [`materials`], [`color_science`], [`control_systems`], [`atmosphere`], [`geophysics`], [`error`] |
//!
//! # Conventions
//!
//! **Units are SI** unless a function's documentation says otherwise, and
//! angles are radians. [`units`] converts, and [`units::quantity`] carries
//! dimensions in the type so that adding a velocity to a time is an error
//! rather than a number.
//!
//! **`f64` throughout**, except where exactness is the point: [`exact`]
//! works over arbitrary-precision integers and rationals, and
//! [`units::dimensional`] computes null spaces over
//! [`exact::rational::Rational`] because a group of quantities is exactly
//! dimensionless or it is not.
//!
//! **Randomness comes from [`monte_carlo::Rng`]**, a linear congruential
//! generator that returns its raw state. The low bits therefore have a
//! short period, so use [`monte_carlo::Rng::below`] for any small-integer
//! draw rather than `next_u64() % n`.
//!
//! # Example
//!
//! ```
//! use rust_physics_engine::units::quantity::{Dim, Quantity};
//!
//! let v = Quantity::new(3.0, Dim::new(1, 0, -1, 0, 0, 0, 0)); // m/s
//! let t = Quantity::new(2.0, Dim::TIME);
//! assert_eq!(v.mul(&t).unwrap().dim, Dim::LENGTH);  // exactly a length
//! assert!(v.add(&t).is_err());                      // and not a time
//! ```

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
