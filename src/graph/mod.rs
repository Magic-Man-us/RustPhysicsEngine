//! Graphs: representation and structure, shortest paths, network flow,
//! matchings, spectral graph theory, colouring, and drawing.

pub mod core;
pub mod coloring;
pub mod flow;
pub mod layout;
pub mod matching;
pub mod paths;
pub mod spectral;

pub use core::Graph;
