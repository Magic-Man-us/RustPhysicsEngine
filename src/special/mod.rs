//! Special functions: error function family, gamma family, and beta
//! functions.

pub mod beta;
pub mod erf;
pub mod gamma;

pub use beta::{beta, beta_inc};
pub use erf::{erf, erfc, erfinv};
pub use gamma::{gamma, gamma_p, gamma_q, lgamma};
