//! Ordinary differential equation solvers.

pub mod explicit;

pub use explicit::{euler_step, rk4_solve, rk4_step, rk4_step_vec};
