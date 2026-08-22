//! Special functions: error function family, gamma family, and beta
//! functions.

pub mod bessel;
pub mod beta;
pub mod elliptic;
pub mod erf;
pub mod gamma;
pub mod legendre;

pub use bessel::{
    bessel_i0, bessel_i1, bessel_j0, bessel_j1, bessel_j_zeros, bessel_jn, bessel_k0, bessel_k1,
    bessel_y0, bessel_y1, bessel_yn,
};
pub use beta::{beta, beta_inc};
pub use elliptic::{
    ellipse_perimeter_exact, elliptic_e, elliptic_e_inc, elliptic_f, elliptic_k,
    pendulum_period_exact,
};
pub use erf::{erf, erfc, erfinv};
pub use gamma::{gamma, gamma_p, gamma_q, lgamma};
pub use legendre::{
    gauss_legendre_nodes, legendre_p, legendre_p_assoc, spherical_harmonic_real,
};
