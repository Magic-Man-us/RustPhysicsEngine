//! Graphs: representation and structure, shortest paths, network flow,
//! matchings, and spectral graph theory.

pub mod core;
pub mod coloring;
pub mod flow;
pub mod matching;
pub mod paths;
pub mod spectral;

pub use core::Graph;
