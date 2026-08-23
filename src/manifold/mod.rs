//! Manifolds and higher-dimensional geometry: generic n-dimensional
//! vectors and tensors, metric-driven curvature, and (in later modules)
//! geodesics, Lie groups, constant-curvature spaces, polytopes, Clifford
//! algebras, embeddings, discrete exterior calculus, and spacetimes.

pub mod geodesic;
pub mod metric;
pub mod vecn;

pub use geodesic::{
    geodesics_on_mesh_exact, great_circle_check, heat_method_geodesic, light_deflection,
    perihelion_precession, photon_orbit_stability, schwarzschild_orbit, shapiro_delay,
    GeodesicState, Integrator,
};
pub use metric::{
    frw_metric, kerr_boyer_lindquist, schwarzschild_metric_fn,
    surface_metric_from_parametrization, warped_product, Metric, Sig,
};
pub use vecn::{
    determinant_n, exterior_derivative_numeric, wedge, TensorN, VecN,
};
