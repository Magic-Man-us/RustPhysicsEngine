//! Exact arithmetic: arbitrary-precision integers, exact rationals,
//! arbitrary-precision binary floating point, polynomials, and continued
//! fractions.

pub mod bigfloat;
pub mod bigint;
pub mod contfrac;
pub mod polynomial;
pub mod rational;
pub mod symbolic;

pub use bigfloat::BigFloat;
pub use bigint::BigInt;
pub use rational::Rational;
pub use symbolic::Expr;
