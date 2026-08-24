//! Exact arithmetic: arbitrary-precision integers, exact rationals,
//! arbitrary-precision binary floating point, polynomials, and continued
//! fractions.

pub mod bigfloat;
pub mod bigint;
pub mod contfrac;
pub mod polynomial;
pub mod rational;

// pub use bigfloat::BigFloat;  // re-added once bigfloat.rs defines it
pub use bigint::BigInt;
pub use rational::Rational;
