//! Finite elements, finite-difference time domain, and spectral methods.
//!
//! Three ways of turning a differential equation into a linear system,
//! kept in one place because the interesting content is how they differ.
//!
//! A finite *difference* replaces the derivative with a difference
//! quotient and asks the equation to hold at grid points. A finite
//! *element* never differentiates the solution twice at all: it multiplies
//! by a test function, integrates by parts, and asks the resulting
//! integral identity to hold for every test function in a finite
//! dimensional space. That change of question is what buys the method its
//! two best properties -- it needs one less derivative of the solution to
//! make sense, so a kink in the coefficient is admissible rather than
//! fatal, and the answer it produces is the *best* approximation in the
//! space with respect to the energy the operator defines.
//!
//! A spectral method is the same Galerkin idea with global smooth basis
//! functions instead of local piecewise ones, which trades the sparsity
//! of the matrix for a convergence rate limited only by the smoothness of
//! the solution.

pub mod fdtd;
pub mod fem1d;
pub mod fem2d;
