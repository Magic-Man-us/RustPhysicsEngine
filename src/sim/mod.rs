//! Time-stepping simulation engines.
//!
//! Where the rest of the crate evaluates a relation, these advance a state
//! forward in time: [`rigid_body`] for 3-D dynamics with quaternion
//! orientation and Euler's equations, [`fluid_sim`] for shallow water and
//! 2-D incompressible Euler, [`heat_sim`] for conduction and
//! convection-diffusion, [`wave_sim`] for the wave equation with Mur
//! absorbing boundaries, [`em_sim`] for FDTD electromagnetics, and
//! [`cloth_sim`] for Verlet cloth and rope.
//!
//! These are compact, readable integrators intended for interactive use
//! and for seeing the physics behave. For the research-grade schemes --
//! Riemann solvers, WENO, lattice Boltzmann, SPH -- see [`crate::cfd`];
//! for finite elements and a Yee-grid FDTD with PML see [`crate::fem`].

pub mod fluid_sim;
pub mod heat_sim;
pub mod em_sim;
pub mod wave_sim;
pub mod rigid_body;
pub mod cloth_sim;
