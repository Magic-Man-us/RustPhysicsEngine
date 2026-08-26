//! Reference property tables.
//!
//! Lookup data rather than computation: [`elements`] carries all 118
//! elements with atomic mass, density, melting and boiling points and
//! thermal and electrical conductivity; [`common`] carries engineering
//! solids; [`fluids`] carries liquids with density, viscosity, surface
//! tension and speed of sound; and [`gases`] carries molar mass, specific
//! heat ratio and thermal conductivity.
//!
//! Values are room-temperature and one-atmosphere unless stated. They are
//! reference figures for calculation, not a substitute for a datasheet on
//! a specific alloy or grade.

pub mod elements;
pub mod common;
pub mod gases;
pub mod fluids;
