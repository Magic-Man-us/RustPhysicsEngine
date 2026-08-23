//! Manifolds and higher-dimensional geometry: generic n-dimensional
//! vectors and tensors, metric-driven curvature, and (in later modules)
//! geodesics, Lie groups, constant-curvature spaces, polytopes, Clifford
//! algebras, embeddings, discrete exterior calculus, and spacetimes.

pub mod geodesic;
pub mod lie;
pub mod metric;
pub mod vecn;

pub use geodesic::{
    geodesics_on_mesh_exact, great_circle_check, heat_method_geodesic, light_deflection,
    perihelion_precession, photon_orbit_stability, schwarzschild_orbit, shapiro_delay,
    GeodesicState, Integrator,
};
pub use lie::{
    casimir_so3, clebsch_gordan, hand_eye_calibration, killing_form, lie_bracket_matrix,
    matrix_exp, matrix_log, matrix_sqrt, pose_graph_optimize, rotate_spherical_harmonics,
    rotation_averaging, se3, so3, so3_haar_measure_density, so3_uniform_grid,
    structure_constants, umeyama_alignment, wigner_d, wigner_d_small, Heisenberg3, LieGroup,
    Se2, Se3, Sim3, Sl2C, Sl2Class, Sl2R, So2, So3, So4, Su2, Unitary,
};
pub use metric::{
    frw_metric, kerr_boyer_lindquist, schwarzschild_metric_fn,
    surface_metric_from_parametrization, warped_product, Metric, Sig,
};
pub use vecn::{
    determinant_n, exterior_derivative_numeric, wedge, TensorN, VecN,
};
