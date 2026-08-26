//! Astrodynamics and astrophysics.
//!
//! Orbits are the core. [`kepler`] solves Kepler's equation for elliptic,
//! parabolic and hyperbolic orbits; [`orbital_elements`] converts between
//! state vectors and Keplerian elements; [`maneuvers`] covers Hohmann and
//! bi-elliptic transfers, plane changes, phasing and J2 secular rates; and
//! [`lambert`] solves for the transfer orbit connecting two positions in a
//! given time.
//!
//! [`time_systems`] and [`coords`] are the bookkeeping that makes those
//! answers refer to anything real -- Julian dates, UT1/TAI/TT/TDB,
//! sidereal time, and the equatorial, ecliptic, galactic, horizontal and
//! ITRF frames with precession and nutation.
//!
//! Many-body gravity is handled by [`nbody`] with a leapfrog integrator
//! and [`octree`] for Barnes-Hut O(N log N) forces. The remaining modules
//! cover [`tidal`] forces and Roche limits, [`lagrange`] points,
//! [`gravitational_waves`], [`magnetosphere`] field-line tracing,
//! [`habitable_zone`] boundaries, and [`collisions`] and impact cratering.

pub mod nbody;
/// Barnes-Hut octree (moved to `crate::spatial::octree`; re-exported here
/// for backwards compatibility).
pub mod octree {
    pub use crate::spatial::octree::*;
}
pub mod kepler;
pub mod lambert;
pub mod maneuvers;
pub mod coords;
pub mod time_systems;
pub mod orbital_elements;
pub mod tidal;
pub mod collisions;
pub mod magnetosphere;
pub mod lagrange;
pub mod habitable_zone;
pub mod gravitational_waves;
