//! Manifolds and higher-dimensional geometry: generic n-dimensional
//! vectors and tensors, metric-driven curvature, and (in later modules)
//! geodesics, Lie groups, constant-curvature spaces, polytopes, Clifford
//! algebras, embeddings, discrete exterior calculus, and spacetimes.

pub mod metric;
pub mod vecn;

pub use metric::{
    frw_metric, kerr_boyer_lindquist, schwarzschild_metric_fn,
    surface_metric_from_parametrization, warped_product, Metric, Sig,
};
pub use vecn::{
    determinant_n, exterior_derivative_numeric, wedge, TensorN, VecN,
};
