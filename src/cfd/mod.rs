//! Computational fluid dynamics: staggered grids, advection schemes,
//! and (in later modules) incompressible solvers, shallow water, SPH,
//! LBM, level sets, and turbulence models.

pub mod advection;
pub mod grid;
pub mod lbm;
pub mod riemann;
pub mod sph;
pub mod shallow_water;
pub mod stable_fluids;

pub use advection::{
    advect_bfecc_2d, advect_flux_limited_2d, advect_lax_wendroff_1d, advect_maccormack_2d,
    advect_muscl_1d, advect_semi_lagrangian_2d, advect_upwind_1d, advect_upwind_2d,
    advect_velocity_semi_lagrangian, advect_weno5_1d, advection_diffusion_1d,
    burgers_exact_cole_hopf, burgers_step, peclet_cell, rk3_ssp, total_variation,
    weno5_reconstruct, Limiter, Scheme,
};
pub use grid::{CellField2, FluidBc, MacGrid2, MacGrid3};
pub use lbm::{
    lbm_cavity_step, lbm_cylinder, lbm_lid_cavity, lbm_poiseuille_2d, lbm_thermal,
    lbm_to_physical, poiseuille_exact, thermal_step, Collision, LbmD2Q9, LbmD3Q19, LbmD3Q27,
};
pub use riemann::{
    blast_wave_woodward_colella, cons_to_prim, flux, flux_ausm_plus, flux_hll, flux_hllc,
    flux_roe, flux_rusanov, isentropic_vortex_exact, lax_problem, normal_shock_relations,
    nozzle_area_ratio, nozzle_mach_from_area, oblique_shock_angle, prandtl_meyer, prim_to_cons,
    quasi_1d_nozzle, rankine_hugoniot, riemann_exact, riemann_exact_star, sedov_1d,
    shu_osher, sod_exact, sod_shock_tube, sound_speed, wave_speeds_einfeldt, Cons, Euler1D,
    Euler2D, EulerBc, FluxKind, Prim,
};
pub use sph::{
    dam_break_2d, dam_break_exact_front, droplet_oscillation, hydrostatic_tank, kernel_grad,
    kernel_laplacian, kernel_support, kernel_w, poiseuille_sph, Kernel, Kind, Plane, Sph,
    SphParticle, SphScheme, SpatialHash,
};
pub use shallow_water::{
    dispersion_deep, dispersion_full, dispersion_shallow, gerstner_wave, jonswap_spectrum,
    kelvin_wake_angle, pierson_moskowitz, stokes_drift, swe_1d_exact_dam_break, swe_1d_step_hll,
    tsunami_runup_1d, wave_breaking_criterion, wave_field_from_spectrum, wave_speed_shallow,
    ShallowWater2D,
};
pub use stable_fluids::{
    flow_past_cylinder, lid_driven_cavity, multigrid_vcycle, pressure_poisson_cg,
    rayleigh_benard, taylor_green_exact, taylor_green_vortex, PressureSolver, StableFluid2,
    StableFluid3,
};
