//! Error types shared by the numerical solvers.
//!
//! New solver APIs return `Result<T, SolveError>`; pre-existing `Option`
//! APIs are untouched.

use std::fmt;

/// Failure modes reported by the numerical solvers in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// The matrix is singular, or a pivot fell below the safe threshold.
    Singular,
    /// The matrix is not (numerically) symmetric positive definite.
    NotPositiveDefinite,
    /// The iteration did not converge within the allowed iteration count.
    NoConvergence { iters: usize, residual: f64 },
    /// Operand dimensions are incompatible.
    DimensionMismatch { expected: usize, got: usize },
    /// An argument violated a documented precondition.
    InvalidArgument(&'static str),
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::Singular => write!(f, "matrix is singular or pivot below threshold"),
            SolveError::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            SolveError::NoConvergence { iters, residual } => {
                write!(f, "no convergence after {iters} iterations (residual {residual:e})")
            }
            SolveError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            SolveError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for SolveError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_variants() {
        assert!(SolveError::Singular.to_string().contains("singular"));
        assert!(SolveError::NotPositiveDefinite.to_string().contains("positive definite"));
        let nc = SolveError::NoConvergence { iters: 5, residual: 1e-3 };
        assert!(nc.to_string().contains('5'));
        let dm = SolveError::DimensionMismatch { expected: 3, got: 4 };
        assert!(dm.to_string().contains('3') && dm.to_string().contains('4'));
        assert!(SolveError::InvalidArgument("x").to_string().contains('x'));
    }

    #[test]
    fn test_eq_clone() {
        let e = SolveError::DimensionMismatch { expected: 2, got: 1 };
        assert_eq!(e.clone(), e);
        assert_ne!(e, SolveError::Singular);
    }
}

/// Failure modes reported by the geometric algorithms (spatial, mesh,
/// patterns modules).
#[derive(Debug, Clone, PartialEq)]
pub enum GeomError {
    /// Degenerate input: zero-area triangle, coincident points, etc.
    Degenerate(&'static str),
    /// The mesh is not manifold where the operation requires it.
    NotManifold,
    /// The input contains no elements.
    Empty,
    /// An argument violated a documented precondition.
    InvalidArgument(&'static str),
}

impl fmt::Display for GeomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeomError::Degenerate(msg) => write!(f, "degenerate geometry: {msg}"),
            GeomError::NotManifold => write!(f, "mesh is not manifold"),
            GeomError::Empty => write!(f, "empty input"),
            GeomError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for GeomError {}

#[cfg(test)]
mod geom_tests {
    use super::*;

    #[test]
    fn test_geom_error_display() {
        assert!(GeomError::Degenerate("x").to_string().contains('x'));
        assert!(GeomError::NotManifold.to_string().contains("manifold"));
        assert!(GeomError::Empty.to_string().contains("empty"));
        assert!(GeomError::InvalidArgument("y").to_string().contains('y'));
        assert_eq!(GeomError::Empty.clone(), GeomError::Empty);
    }
}
