//! Ordinary differential equation solvers.

pub mod adaptive;
pub mod explicit;
pub mod implicit;
pub mod symplectic;

pub use adaptive::{dormand_prince, dormand_prince_dense, AdaptiveResult};
pub use explicit::{euler_step, rk4_solve, rk4_step, rk4_step_vec};
pub use implicit::{backward_euler, bdf2};
pub use symplectic::{leapfrog_kick_drift_kick, velocity_verlet, yoshida4};
