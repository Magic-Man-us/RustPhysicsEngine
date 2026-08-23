//! Computational fluid dynamics: staggered grids, advection schemes,
//! and (in later modules) incompressible solvers, shallow water, SPH,
//! LBM, level sets, and turbulence models.

pub mod advection;
pub mod boundary_layer;
pub mod grid;
pub mod lbm;
pub mod level_set;
pub mod potential_flow;
pub mod riemann;
pub mod sph;
pub mod shallow_water;
pub mod stable_fluids;
pub mod turbulence;

pub use advection::{
    advect_bfecc_2d, advect_flux_limited_2d, advect_lax_wendroff_1d, advect_maccormack_2d,
    advect_muscl_1d, advect_semi_lagrangian_2d, advect_upwind_1d, advect_upwind_2d,
    advect_velocity_semi_lagrangian, advect_weno5_1d, advection_diffusion_1d,
    burgers_exact_cole_hopf, burgers_step, peclet_cell, rk3_ssp, total_variation,
    weno5_reconstruct, Limiter, Scheme,
};
pub use boundary_layer::{
    blasius_cf, blasius_drag_plate, blasius_profile, blasius_solve, blasius_thickness,
    couette_flow, ekman_depth, ekman_spiral, falkner_skan_separation_beta, falkner_skan_solve,
    first_cell_height, flat_plate_heat_transfer_laminar, head_entrainment_method, law_of_the_wall,
    michel_transition_criterion, pohlhausen_profile, spalding, stokes_first_problem,
    stokes_second_problem, stratford_separation_criterion, thermal_bl_ratio, thwaites_method,
    thwaites_separation_point, transition_re_x_estimate, turbulent_bl_power_law,
    turbulent_cf_prandtl, turbulent_cf_schlichting, turbulent_thickness_1_7, u_tau,
    van_driest_damping, y_plus,
};
pub use grid::{CellField2, FluidBc, MacGrid2, MacGrid3};
pub use lbm::{
    lbm_cavity_step, lbm_cylinder, lbm_lid_cavity, lbm_poiseuille_2d, lbm_thermal,
    lbm_to_physical, poiseuille_exact, thermal_step, Collision, LbmD2Q9, LbmD3Q19, LbmD3Q27,
};
pub use level_set::{
    capillary_wave_dispersion, contact_angle_young, droplet_shape_pendant, minnaert_frequency,
    ohnesorge, rayleigh_plesset, single_vortex_deformation_test, taylor_bubble_velocity,
    weber_breakup_regime, young_laplace_pressure, zalesak_disk, zalesak_rotate,
    FreeSurfaceFluid2, LevelSet2, LevelSet3, Segment2, Vof2, WenoOrUpwind,
};
pub use potential_flow::{
    added_mass_cylinder, added_mass_sphere, conformal_map_flow, cylinder_cp_exact, cylinder_flow,
    doublet, elliptic_wing_cl, ground_effect_factor, induced_drag, inverse_joukowski,
    joukowski_airfoil, joukowski_airfoil_flow, joukowski_transform, karman_trefftz_airfoil,
    lifting_line, method_of_images_wall, naca4, naca5, oswald_efficiency_estimate, rankine_oval,
    sink, source, thin_airfoil_cl, thin_airfoil_cl_flat, uniform_flow, vortex, vortex_lattice,
    Element, PanelMethod, Plane2, PotentialFlow2, WingGeometry,
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
pub use turbulence::{
    channel_flow_dns_reference, decaying_isotropic_turbulence, delta_criterion,
    dissipation_rate_from_spectrum, dynamic_smagorinsky_cs, energy_spectrum_1d,
    energy_spectrum_2d, energy_spectrum_3d, inertial_range_exponent, integral_scale,
    kolmogorov_scales, kolmogorov_spectrum, lambda2_criterion, log_law_fit, pao_spectrum,
    q_criterion, rans_step, re_lambda, reynolds_stress, richardson_cascade_time,
    rotation_tensor, smagorinsky_nu_t, strain_tensor, structure_function,
    synthetic_eddy_method, synthetic_turbulence_kraichnan, taylor_microscale,
    turbulence_intensity, turbulent_diffusivity, two_point_correlation, von_karman_spectrum,
    vortex_identify_q, vreman_nu_t, wale_nu_t, KEpsilon, KEpsilonVariant, KOmegaSst, RansModel,
    SpalartAllmaras,
};
