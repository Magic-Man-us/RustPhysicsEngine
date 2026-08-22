pub mod nbody;
/// Barnes-Hut octree (moved to `crate::spatial::octree`; re-exported here
/// for backwards compatibility).
pub mod octree {
    pub use crate::spatial::octree::*;
}
pub mod orbital_elements;
pub mod tidal;
pub mod collisions;
pub mod magnetosphere;
pub mod lagrange;
pub mod habitable_zone;
pub mod gravitational_waves;
