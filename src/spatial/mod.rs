//! Spatial data structures, transforms, geometric primitives, and
//! queries.

pub mod bvh;
pub mod contain;
pub mod distance;
pub mod frame;
pub mod intersect;
pub mod kdtree;
pub mod mat4;
pub mod octree;
pub mod primitives;
pub mod projective;
pub mod quadtree;
pub mod sdf;
pub mod transform2d;

pub use bvh::Bvh;
pub use frame::Frame;
pub use kdtree::{KdTree, KdTree2, SpatialHash};
pub use quadtree::Quadtree;
pub use intersect::RayHit;
pub use mat4::Mat4;
pub use octree::Octree;
pub use primitives::{
    Aabb, Capsule, Circle, Cylinder, Obb, Plane, Polygon2, Polyline, Ray, Rect, Segment,
    Segment2, Sphere, Triangle, Triangle2,
};
pub use projective::{
    are_collinear, cross_ratio, dehomogenize, line_through, lines_intersect, point_h,
    point_on_line, rectify_quad_to_rect, Hom2, Homography,
};
pub use transform2d::Affine2;
