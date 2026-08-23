//! Computational fluid dynamics: staggered grids, advection schemes,
//! and (in later modules) incompressible solvers, shallow water, SPH,
//! LBM, level sets, and turbulence models.

pub mod advection;
pub mod grid;

pub use advection::{
    advect_bfecc_2d, advect_flux_limited_2d, advect_lax_wendroff_1d, advect_maccormack_2d,
    advect_muscl_1d, advect_semi_lagrangian_2d, advect_upwind_1d, advect_upwind_2d,
    advect_velocity_semi_lagrangian, advect_weno5_1d, advection_diffusion_1d,
    burgers_exact_cole_hopf, burgers_step, peclet_cell, rk3_ssp, total_variation,
    weno5_reconstruct, Limiter, Scheme,
};
pub use grid::{CellField2, FluidBc, MacGrid2, MacGrid3};
